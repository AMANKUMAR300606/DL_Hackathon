from typing import List, Dict, Any, Optional
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.faiss_indexer import FAISSIndexer
from src.indexing.graph_indexer import GraphIndexer
from sentence_transformers import CrossEncoder
from src.core.intent_detector import IntentDetector
from src.core.explanation_generator import ExplanationGenerator
from src.models.schema import CandidateProfile, SearchResultItem


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Indexer,
        faiss_idx: FAISSIndexer,
        graph: GraphIndexer,
        profiles: Optional[List[CandidateProfile]] = None,
    ):
        self.bm25 = bm25
        self.faiss = faiss_idx
        self.graph = graph
        self.intent_detector = IntentDetector(profiles or [])
        self.explainer = ExplanationGenerator()
        print("Loading CrossEncoder model...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        faiss_results: List[Dict],
        graph_results: List[Dict],
        k: int = 60,
    ):
        rrf_scores: Dict[str, float] = {}
        all_profiles: Dict[str, Dict] = {}

        def add_to_rrf(results: List[Dict], source: str):
            for rank, res in enumerate(results):
                pid = res["profile"].id
                if pid not in all_profiles:
                    all_profiles[pid] = {"profile": res["profile"], "scores": {}}

                all_profiles[pid]["scores"][source] = res["score"]

                # Capture graph traversal data from graph results
                if source == "graph":
                    if "graph_traversal" in res:
                        all_profiles[pid]["scores"]["graph_traversal"] = res["graph_traversal"]
                    if "graph_traversal_detail" in res:
                        all_profiles[pid]["scores"]["graph_traversal_detail"] = res["graph_traversal_detail"]

                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

        add_to_rrf(bm25_results, "bm25")
        add_to_rrf(faiss_results, "faiss")
        add_to_rrf(graph_results, "graph")

        sorted_pids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)
        return sorted_pids, rrf_scores, all_profiles

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------
    def search(
        self, query: str, top_k: int = 5, explain: bool = True
    ) -> List[SearchResultItem]:
        intent = self.intent_detector.analyze_intent(query)
        expanded_query = intent["expanded_query"]

        fetch_k = max(top_k * 3, 20)

        bm25_res  = self.bm25.search(expanded_query, fetch_k)
        faiss_res = self.faiss.search(expanded_query, fetch_k)
        graph_res = self.graph.search(expanded_query, fetch_k)

        sorted_pids, rrf_scores, all_profiles = self._reciprocal_rank_fusion(
            bm25_res, faiss_res, graph_res
        )

        # 1. Apply strict YoE Filter
        min_yoe = intent.get("min_yoe", 0.0)
        if min_yoe > 0:
            filtered_pids = []
            for pid in sorted_pids:
                if all_profiles[pid]["profile"].years_of_experience >= min_yoe:
                    filtered_pids.append(pid)
            sorted_pids = filtered_pids

        # 2. Re-rank with Cross-Encoder
        rerank_pool = sorted_pids[:max(30, top_k * 2)]
        if rerank_pool:
            cross_inp = []
            for pid in rerank_pool:
                prof = all_profiles[pid]["profile"]
                skills = f"{prof.core_skills or ''} {prof.secondary_skills or ''}"
                roles = str(prof.potential_roles or '')
                doc_text = f"Roles: {roles}. Skills: {skills}. Summary: {prof.skill_summary or ''}"
                cross_inp.append((query, doc_text))
            
            ce_scores = self.cross_encoder.predict(cross_inp)
            
            # Min-Max Normalization to [0, 1]
            min_score = min(ce_scores)
            max_score = max(ce_scores)
            range_score = max_score - min_score if max_score > min_score else 1.0
            
            ce_score_map = {}
            for idx, pid in enumerate(rerank_pool):
                raw_score = float(ce_scores[idx])
                norm_score = (raw_score - min_score) / range_score
                ce_score_map[pid] = norm_score
                all_profiles[pid]["scores"]["cross_encoder"] = norm_score
            
            # Apply threshold to filter out poor matches
            # Adjust the threshold value as needed (e.g., 0.1 means bottom 10% of rerank pool are discarded)
            threshold = 0.01 
            filtered_rerank_pool = [pid for pid in rerank_pool if ce_score_map[pid] >= threshold]
            
            # Sort the remaining ones by normalized score
            sorted_pids = sorted(filtered_rerank_pool, key=lambda pid: ce_score_map[pid], reverse=True)

        final_results = []
        
        for pid in sorted_pids[:top_k]:
            profile_data = all_profiles[pid]
            profile = profile_data["profile"]
            scores  = profile_data["scores"]

            explanation_detail = None
            explanation_str = None
            if explain:
                explanation_detail = self.explainer.generate(profile, query, scores)
                explanation_str = explanation_detail.summary

            final_results.append(
                SearchResultItem(
                    profile=profile,
                    score=scores.get("cross_encoder", rrf_scores.get(pid, 0.0)),
                    explanation=explanation_str,
                    explanation_detail=explanation_detail,
                )
            )

        return final_results
