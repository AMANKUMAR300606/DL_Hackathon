from typing import Dict, Any, List
from src.models.schema import CandidateProfile, ExplanationDetail, MethodType, GraphStep


class ExplanationGenerator:
    def generate(
        self,
        profile: CandidateProfile,
        query: str,
        scores: Dict[str, Any],
    ) -> ExplanationDetail:
        """
        Produces a structured ExplanationDetail showing exactly which retrieval
        methods contributed, with scores and 2-hop graph traversal paths.
        """
        query_lower = query.lower()
        methods: List[MethodType] = []
        lexical_matches: List[str] = []
        graph_steps: List[GraphStep] = []

        bm25_score  = scores.get("bm25",  None)
        faiss_score = scores.get("faiss", None)
        graph_score = scores.get("graph", None)
        cross_encoder_score = scores.get("cross_encoder", None)

        # Rich traversal detail from graph (list of {skill, via_roles, direct})
        graph_traversal_detail: List[Dict] = scores.get("graph_traversal_detail", [])
        # Flat list of matched skills (backward compat)
        graph_traversal_flat: List[str] = scores.get("graph_traversal", [])

        # ── Lexical (BM25) ─────────────────────────────────────────────
        if bm25_score is not None and bm25_score > 0:
            methods.append(MethodType.lexical)
            query_tokens = set(t for t in query_lower.split() if len(t) > 2)
            # Check all dynamic attributes (universal — not just core_skills)
            for attr in (profile.attributes or []):
                for item in attr.value:
                    clean = str(item).strip().lower()
                    if clean and any(t in clean or clean in t for t in query_tokens):
                        if clean not in lexical_matches:
                            lexical_matches.append(clean)
            # Fallback: also check legacy fields in case attributes are empty
            if not lexical_matches:
                for col in [profile.core_skills, profile.secondary_skills]:
                    if not col:
                        continue
                    for skill in col.split(","):
                        clean = skill.split("(")[0].strip().lower()
                        if clean and any(t in clean or clean in t for t in query_tokens):
                            if clean not in lexical_matches:
                                lexical_matches.append(clean)

        # ── Semantic (FAISS) ───────────────────────────────────────────
        semantic_similarity = None
        semantic_bucket = "low"
        if faiss_score is not None and faiss_score > 0:
            methods.append(MethodType.semantic)
            semantic_similarity = float(faiss_score)
            if faiss_score >= 0.75:
                semantic_bucket = "high"
            elif faiss_score >= 0.45:
                semantic_bucket = "medium"
            else:
                semantic_bucket = "low"

        # ── Knowledge Graph (Neo4j) ────────────────────────────────────
        if graph_score is not None and graph_score > 0:
            methods.append(MethodType.graph)

            # Construct strict GraphStep objects for the new schema
            if graph_traversal_detail:
                for seg in graph_traversal_detail[:3]:
                    source = seg.get("semantic_proxy") or "Query"
                    target = seg.get("skill", "")
                    
                    if seg.get("semantic_proxy"):
                        graph_steps.append(GraphStep(source_id="Query", relation="SEMANTIC_PROXY", target_id=source))
                    
                    graph_steps.append(GraphStep(source_id=source, relation="MATCH_SKILL", target_id=target))
                    
                    for role in seg.get("via_roles", []):
                        graph_steps.append(GraphStep(source_id=target, relation="MATCH_ROLE", target_id=role))
                        target = role
                    
                    graph_steps.append(GraphStep(source_id=target, relation="HAS_CANDIDATE", target_id=profile.id))
            elif graph_traversal_flat:
                prev = "Query"
                for step in graph_traversal_flat[:4]:
                    graph_steps.append(GraphStep(source_id=prev, relation="TRAVERSAL", target_id=step))
                    prev = step

        # ── Cross-Encoder (Re-ranking) ─────────────────────────────────
        if cross_encoder_score is not None:
            methods.append(MethodType.cross_encoder)

        # ── Human-readable summary ─────────────────────────────────────
        summary_parts = []

        if MethodType.lexical in methods:
            if lexical_matches:
                summary_parts.append(
                    f"BM25 (Lexical Index): matched {', '.join(lexical_matches[:4])}"
                )
            else:
                summary_parts.append(f"BM25 (Lexical Index): matched keywords")

        if MethodType.semantic in methods:
            summary_parts.append(
                f"FAISS (Semantic Vector Index): {semantic_bucket} similarity"
            )

        if MethodType.graph in methods:
            nodes = []
            if graph_steps:
                for s in graph_steps:
                    if s.target_id and s.relation != "HAS_CANDIDATE":
                        t = s.target_id.title()
                        if not nodes or nodes[-1] != t:
                            nodes.append(t)
            elif graph_traversal_flat:
                for s in graph_traversal_flat[:3]:
                    t = str(s).title()
                    if not nodes or nodes[-1] != t:
                        nodes.append(t)

            if nodes:
                if len(nodes) == 1:
                    narrative = f"'{nodes[0]}' matched the query"
                else:
                    narrative = f"'{nodes[0]}' is going to '{nodes[1]}'"
                    for i in range(2, len(nodes)):
                        narrative += f", which goes to '{nodes[i]}'"
                
                summary_parts.append(f"Neo4j (Knowledge Graph): {narrative}, and at last these are matched so we have this output")
            else:
                summary_parts.append(f"Neo4j (Knowledge Graph): Retrieved by traversing the candidate's professional network")

        if MethodType.cross_encoder in methods:
            summary_parts.append(f"Re-ranked by Cross-Encoder (score {cross_encoder_score:.3f})")

        if not summary_parts:
            summary_parts.append("Matched via hybrid RRF scoring.")

        return ExplanationDetail(
            methods=methods,
            bm25_score=round(bm25_score, 4)  if bm25_score  is not None else None,
            faiss_score=round(faiss_score, 4) if faiss_score is not None else None,
            graph_score=round(graph_score, 4) if graph_score is not None else None,
            cross_encoder_score=round(cross_encoder_score, 4) if cross_encoder_score is not None else None,
            lexical_matches=lexical_matches[:6],
            semantic_similarity=semantic_similarity,
            graph_traversal=graph_steps,
            summary=" | ".join(summary_parts)
        )
