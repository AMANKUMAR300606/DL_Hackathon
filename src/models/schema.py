from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, NewType
from enum import Enum
from datetime import datetime

# Enforce strict global ID typing to guarantee uniqueness across complex graphs
EntityID = NewType('EntityID', str)

class MethodType(str, Enum):
    lexical = "lexical"
    semantic = "semantic"
    graph = "graph"
    cross_encoder = "cross_encoder"

class Direction(str, Enum):
    out = "out"
    in_ = "in"

class ConfidenceMixin(BaseModel):
    confidence: Optional[float] = None

    @validator("confidence")
    def validate_confidence(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("confidence must be between 0 and 1")
        return v

class TypeLabel(ConfidenceMixin):
    name: str
    canonical_id: Optional[str] = None
    
    @validator("name")
    def validate_name(cls, v):
        return v.strip()

class Relation(ConfidenceMixin):
    relation_type: str
    canonical_relation_type: Optional[str] = None
    target_id: EntityID
    target_type: Optional[str] = None
    canonical_id: Optional[str] = None
    source: Optional[str] = None    # Provenance tracking
    direction: Direction = Direction.out
    inverse_relation: Optional[str] = None
    
    @validator("relation_type")
    def validate_relation_type(cls, v):
        if not v or not v.strip():
            raise ValueError("relation_type cannot be empty")
        return v.lower()

class Attribute(ConfidenceMixin):
    key: str
    value: Any
    value_type: Optional[str] = None
    canonical_id: Optional[str] = None
    source: Optional[str] = None    # Provenance tracking

class CandidateProfile(BaseModel):
    id: EntityID
    entity_type: str = "candidate"
    canonical_id: Optional[str] = None
    name: Optional[str] = None
    version: str = "1.0"
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    # --- Legacy Fields (retained for backward compatibility during MVP) ---
    core_skills: Optional[str] = None
    secondary_skills: Optional[str] = None
    soft_skills: Optional[str] = None
    years_of_experience: float = 0.0
    potential_roles: Optional[str] = None
    skill_summary: Optional[str] = None
    
    # --- Next-Gen Dynamic Graph Representation ---
    types: List[TypeLabel] = Field(default_factory=list)
    attributes: List[Attribute] = Field(default_factory=list)    # Replaced loose Dict with strict List[Attribute]
    relations: List[Relation] = Field(default_factory=list)
    embedding: Optional[List[float]] = None

class SearchQuery(BaseModel):
    query: str
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    explain: bool = True

class GraphStep(BaseModel):
    source_id: EntityID
    relation: str
    target_id: EntityID
    confidence: Optional[float] = None
    step_score: Optional[float] = None

class ExplanationDetail(BaseModel):
    """Structured, source-tagged explanation for a single retrieval result."""
    methods: List[MethodType] = Field(default_factory=list)
    
    # Raw scores per method (0-1 normalized)
    bm25_score: Optional[float] = None
    faiss_score: Optional[float] = None
    graph_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    
    # Lexical: which skills / terms matched 
    lexical_matches: List[str] = Field(default_factory=list)
    # Semantic: similarity score bucket
    semantic_similarity: Optional[float] = None
    # Graph: the traversal path that led to this candidate
    graph_traversal: List[GraphStep] = Field(default_factory=list)
    # Human-readable summary
    summary: str = ""

class SearchResultItem(BaseModel):
    profile: CandidateProfile
    score: float
    confidence: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None           # Added for profound transparency & debugging
    explanation: Optional[str] = None
    explanation_detail: Optional[ExplanationDetail] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
