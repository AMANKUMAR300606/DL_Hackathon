from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from src.models.schema import CandidateProfile

STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "in", "to", "for",
    "with", "is", "it", "at", "on", "by", "as", "be", "this",
    "manager", "specialist", "coordinator", "associate", "senior",
    "junior", "lead", "expert", "professional"
}

SYNONYMS = {
    "frontend":   "frontend",
    "front-end":  "frontend",
    "back-end":   "backend",
    "fullstack":  "fullstack",
    "full-stack": "fullstack",
    "html5":      "html",
    "css3":       "css",
    "js":         "javascript",
    "react.js":   "react",
    "node.js":    "nodejs",
    "vue.js":     "vue",
    "postgres":   "postgresql",
    "ml":         "machine learning",
    "ai":         "artificial intelligence",
}

class BM25Indexer:
    def __init__(self):
        self.bm25_corpus = []
        self.corpus: list = []
        self.profiles = []
        self.bm25 = None

    def _preprocess(self, text: str) -> List[str]:
        if not text:
            return []

        # 1. Lowercase
        text = text.lower()

        # 2. Normalize special tech tokens before stripping punctuation
        text = text.replace("c++", "cplusplus")
        text = text.replace("c#",  "csharp")
        text = text.replace(".net", "dotnet")
        text = text.replace(".js",  "js")

        # 3. Remove all punctuation — keeps alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # 4. Tokenize on whitespace
        tokens = text.split()

        # 5. Remove stopwords and single-char tokens
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

        # 6. Synonym normalization
        tokens = [SYNONYMS.get(t, t) for t in tokens]

        return tokens

    def _profile_to_text(self, p: CandidateProfile) -> str:
        """Build the text representation used for BM25 indexing."""
        parts = []
        for a in (p.attributes or []):
            vals = " ".join(str(v) for v in a.value)
            parts.append(f"{a.key}: {vals}")
        if p.skill_summary:
            parts.append(p.skill_summary)
        return " ".join(parts)

    def index(self, profiles: List[CandidateProfile]):
        self.profiles = profiles
        tokenized_corpus = [self._preprocess(self._profile_to_text(p)) for p in profiles]
        self.corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_single(self, profile: CandidateProfile):
        """
        Incrementally add ONE new profile to the in-memory BM25 index.
        Appends the tokenized doc and rebuilds BM25Okapi (fast for +1 doc).
        """
        tokens = self._preprocess(self._profile_to_text(profile))
        self.corpus.append(tokens)
        self.profiles.append(profile)
        self.bm25 = BM25Okapi(self.corpus)

    def update_single(self, profile: CandidateProfile):
        """
        Update an existing profile's tokens in-place.
        Finds by ID, replaces in corpus + profiles list, rebuilds BM25Okapi.
        Falls back to add_single() if ID not found.
        """
        for i, p in enumerate(self.profiles):
            if p.id == profile.id:
                self.profiles[i] = profile
                self.corpus[i] = self._preprocess(self._profile_to_text(profile))
                self.bm25 = BM25Okapi(self.corpus)
                return
        # Not found — add as new
        self.add_single(profile)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []
        
        tokenized_query = self._preprocess(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        query_set = set(tokenized_query)

        results = []
        for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            # Hard filter: document MUST share at least one token with query
            doc_tokens = set(self.corpus[idx])
            if not (query_set & doc_tokens):
                continue  # Zero lexical overlap — skip completely

            results.append({
                "profile": self.profiles[idx],
                "score": float(scores[idx]),
                "technique": "bm25"
            })

            if len(results) >= top_k:
                break

        return results
