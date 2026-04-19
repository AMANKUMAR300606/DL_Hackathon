import os
import math
from typing import List, Dict
from src.core.data_loader import load_and_clean_data
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.faiss_indexer import FAISSIndexer
from src.indexing.graph_indexer import GraphIndexer
from src.core.retriever import HybridRetriever

def is_relevant(query: str, profile) -> bool:
    """
    A programmatic heuristic to judge relevance in the absence of a labeled dataset.
    Real RAG systems either use an LLM-as-a-judge (like Ragas) or build synthetic ground truth.
    Here we use keyword heuristics to simulate an automated judge.
    """
    text = f"{profile.potential_roles} {profile.core_skills} {profile.secondary_skills} {profile.skill_summary}".lower()
    query_lower = query.lower()
    
    if "python backend" in query_lower:
        return "python" in text and ("backend" in text or "back-end" in text)
    if "regulatory affairs" in query_lower:
        return "regulatory" in text and "fda" in text
        
    return False

# Simple IR Metrics implementation
def precision_at_k(retrieved_docs, query: str, k: int):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if is_relevant(query, doc.profile))
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(retrieved_docs, query: str, total_relevant_in_corpus: int, k: int):
    if total_relevant_in_corpus == 0:
        return 0.0
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_k if is_relevant(query, doc.profile))
    return relevant_retrieved / total_relevant_in_corpus

def dcg_at_k(retrieved_docs, query: str, k: int):
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        if is_relevant(query, doc.profile):
            rel = 1  # Binary relevance
            dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(retrieved_docs, query: str, total_relevant_in_corpus: int, k: int):
    if total_relevant_in_corpus == 0:
        return 0.0
    # Ideal DCG assumes all top k slots are filled with relevant documents (if enough exist)
    ideal_hits = min(k, total_relevant_in_corpus)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += (2**1 - 1) / math.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_docs, query, k) / idcg

def evaluate_system():
    print("Loading data for evaluation...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'profiles.csv')
    profiles = load_and_clean_data(data_path)
    
    # Initialize indexers
    print("Initializing indexers...")
    bm25 = BM25Indexer()
    bm25.index(profiles)
    
    faiss_idx = FAISSIndexer()
    faiss_idx.index(profiles)
    
    # Do NOT wipe the live Neo4j database. Just connect to it.
    graph = GraphIndexer()
    graph.profiles = profiles
    graph._profile_map = {p.id: p for p in profiles}
    
    hybrid = HybridRetriever(bm25, faiss_idx, graph, profiles=profiles)
    
    test_queries = [
        "Looking for a Python Backend Developer", 
        "Regulatory Affairs Manager with FDA experience"
    ]
    
    k = 5
    for query in test_queries:
        print(f"\nEvaluating Query: '{query}'")
        
        # 1. Establish Ground Truth by scanning the entire corpus
        total_relevant = sum(1 for p in profiles if is_relevant(query, p))
        print(f"[Ground Truth] Found {total_relevant} total relevant candidates in the entire dataset.")
        
        # 2. Get hybrid results
        results = hybrid.search(query, top_k=k)
        retrieved_ids = [res.profile.id for res in results]
        print(f"Actually retrieved top {k} IDs: {retrieved_ids}")
        
        # 3. Calculate metrics dynamically based on what was retrieved vs what is relevant
        p_at_k = precision_at_k(results, query, k)
        r_at_k = recall_at_k(results, query, total_relevant, k)
        ndcg = ndcg_at_k(results, query, total_relevant, k)
        
        print(f"Hybrid Retriever -> P@{k}: {p_at_k:.2f}, R@{k}: {r_at_k:.2f}, nDCG@{k}: {ndcg:.2f}")

    print("\nNote: For comprehensive RAGAS evaluation without heuristics, you would need an OPENAI_API_KEY")
    print("to utilize LLM-as-a-judge for Context Precision and Recall.")

if __name__ == "__main__":
    evaluate_system()
