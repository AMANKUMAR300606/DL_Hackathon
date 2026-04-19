"""
Data-driven IntentDetector.

Instead of hardcoded skill/role lists, vocabulary is extracted at runtime
from the actual candidate profiles loaded from CSV.
"""
import re
from typing import Dict, Any, List, Set
from src.models.schema import CandidateProfile


class IntentDetector:
    def __init__(self, profiles: List[CandidateProfile]):
        self.skill_vocab: Set[str] = set()
        self.role_vocab: Set[str] = set()
        self._build_vocabulary(profiles)

    # ------------------------------------------------------------------
    # Vocabulary building
    # ------------------------------------------------------------------
    def _extract_terms(self, text: str) -> List[str]:
        if not text:
            return []
        terms = []
        for part in text.split(","):
            clean = part.split("(")[0].strip().lower()
            if clean and len(clean) > 2:
                terms.append(clean)
        return terms

    def _build_vocabulary(self, profiles: List[CandidateProfile]):
        for p in profiles:
            # Skills from core and secondary columns
            for col in [p.core_skills, p.secondary_skills]:
                for term in self._extract_terms(col or ""):
                    self.skill_vocab.add(term)
            # Roles from potential_roles column
            for term in self._extract_terms(p.potential_roles or ""):
                self.role_vocab.add(term)

        print(
            f"[IntentDetector] Vocabulary built from data: "
            f"{len(self.skill_vocab)} skills, {len(self.role_vocab)} roles."
        )

    def extend(self, profiles: List[CandidateProfile]):
        """Extend vocabulary with terms from new profiles (no full rebuild)."""
        before_skills = len(self.skill_vocab)
        before_roles  = len(self.role_vocab)
        for p in profiles:
            for col in [p.core_skills, p.secondary_skills]:
                for term in self._extract_terms(col or ""):
                    self.skill_vocab.add(term)
            for term in self._extract_terms(p.potential_roles or ""):
                self.role_vocab.add(term)
        added_skills = len(self.skill_vocab) - before_skills
        added_roles  = len(self.role_vocab)  - before_roles
        print(
            f"[IntentDetector] Vocab extended: +{added_skills} skills, "
            f"+{added_roles} roles -> {len(self.skill_vocab)} / {len(self.role_vocab)} total."
        )

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Match query tokens against the data-driven skill and role vocabularies.
        A token matches a vocabulary term if either contains the other
        (handles partial matches like 'python' matching 'python programming').
        """
        query_lower = query.lower()

        # Extract strict Years of Experience requirement
        min_yoe = 0.0
        yoe_pattern = r"(?:>|more than|over|at least)\s*(\d+)|(\d+)\+?\s*(?:years|yrs|yoe)"
        match = re.search(yoe_pattern, query_lower)
        if match:
            val = match.group(1) or match.group(2)
            if val:
                min_yoe = float(val)

        query_tokens = [t.strip() for t in query_lower.split() if len(t.strip()) > 2]

        matched_skills: List[str] = []
        matched_roles: List[str] = []

        for token in query_tokens:
            # Check skills
            for skill in self.skill_vocab:
                if token in skill or skill in token:
                    if skill not in matched_skills:
                        matched_skills.append(skill)
            # Check roles
            for role in self.role_vocab:
                if token in role or role in token:
                    if role not in matched_roles:
                        matched_roles.append(role)

        # Also do a full-phrase scan for multi-word terms
        for skill in self.skill_vocab:
            if len(skill) > 4 and skill in query_lower and skill not in matched_skills:
                matched_skills.append(skill)
        for role in self.role_vocab:
            if len(role) > 4 and role in query_lower and role not in matched_roles:
                matched_roles.append(role)

        # Limit to top 5 most relevant matches (shorter = more specific)
        matched_skills = sorted(matched_skills, key=len)[:5]
        matched_roles = sorted(matched_roles, key=len)[:5]

        intent_type = "general_search"
        if matched_roles and not matched_skills:
            intent_type = "role_search"
        elif matched_skills and not matched_roles:
            intent_type = "skill_search"
        elif matched_roles and matched_skills:
            intent_type = "role_and_skill_search"

        extra = " ".join(matched_roles + matched_skills)
        return {
            "original_query": query,
            "intent_type": intent_type,
            "extracted_roles": matched_roles,
            "extracted_skills": matched_skills,
            "expanded_query": f"{query} {extra}".strip(),
            "min_yoe": min_yoe,
        }
