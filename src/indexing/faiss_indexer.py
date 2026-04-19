import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from src.models.schema import CandidateProfile

class FAISSIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_db = faiss.IndexFlatL2(self.dimension)
        self.profiles = []

    def _profile_to_text(self, p: CandidateProfile) -> str:
        """Build the text representation used for embedding a profile."""
        text_parts = []
        for a in (p.attributes or []):
            vals = " ".join(str(v) for v in a.value)
            text_parts.append(f"{a.key}: {vals}")
            text_parts.append(vals)  # boost: repeat for denser semantic embedding
        if p.core_skills:
            text_parts.append(f"Skills: {p.core_skills}")
        if p.potential_roles:
            text_parts.append(f"Roles: {p.potential_roles}")
        if p.skill_summary:
            text_parts.append(p.skill_summary)
        return " ".join(text_parts)

    def index(self, profiles: List[CandidateProfile]):
        self.profiles = profiles
        texts = [self._profile_to_text(p) for p in profiles]

        if not texts:
            return

        print(f"Embedding {len(texts)} documents for FAISS index...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        self.index_db.add(embeddings)

    def add_single(self, profile: CandidateProfile):
        """Incrementally add ONE new profile to the in-memory index (no full rebuild)."""
        text = self._profile_to_text(profile)
        emb = self.model.encode([text]).astype("float32")
        faiss.normalize_L2(emb)
        self.index_db.add(emb)
        self.profiles.append(profile)

    def update_single(self, profile: CandidateProfile):
        """
        Update an existing profile's embedding in-place.
        Uses faiss.reconstruct_n() to swap just the one vector — no full re-encoding.
        Falls back to add_single() if the profile ID isn't found.
        """
        for i, p in enumerate(self.profiles):
            if p.id == profile.id:
                self.profiles[i] = profile
                text = self._profile_to_text(profile)
                new_emb = self.model.encode([text]).astype("float32")
                faiss.normalize_L2(new_emb)
                # Reconstruct all existing vectors, swap position i, rebuild index
                n = self.index_db.ntotal
                all_vecs = self.index_db.reconstruct_n(0, n).astype("float32")
                all_vecs[i] = new_emb[0]
                self.index_db = faiss.IndexFlatL2(self.dimension)
                self.index_db.add(all_vecs)
                return
        # Profile not found — add it as new
        self.add_single(profile)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index_db.ntotal == 0:
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index_db.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                similarity = 1.0 / (1.0 + distances[0][i])
                results.append({
                    "profile": self.profiles[idx],
                    "score": float(similarity),
                    "technique": "faiss"
                })
        return results
