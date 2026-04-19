import os
import sys

def main():
    # RAGAS relies heavily on LLMs to grade the system.
    # By default, it uses OpenAI's models (GPT-3.5/GPT-4) as the judge.
    if not os.getenv("GOOGLE_API_KEY"):
        print("[ERROR] GOOGLE_API_KEY environment variable is not set!")
        print("RAGAS requires an LLM API key to evaluate Context Precision, Recall, Faithfulness, etc.")
        print("\nPlease set it in your terminal before running this script:")
        print("    $env:GOOGLE_API_KEY='AIzaSy...your-key-here'")
        sys.exit(1)

    print("Importing Ragas and Dataset modules...")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import llm_factory
    # from ragas.embed import embedding_factory
   
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        AnswerRelevancy,
    )
    
    from src.core.data_loader import load_and_clean_data
    from src.indexing.bm25_indexer import BM25Indexer
    from src.indexing.faiss_indexer import FAISSIndexer
    from src.indexing.graph_indexer import GraphIndexer
    from src.core.retriever import HybridRetriever
    
    print("Loading data and initializing indexers...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'profiles.csv')
    profiles = load_and_clean_data(data_path)
    
    bm25 = BM25Indexer()
    bm25.index(profiles)
    
    faiss_idx = FAISSIndexer()
    faiss_idx.index(profiles)
    
    graph = GraphIndexer()
    # Do NOT wipe the live Neo4j database.
    graph.profiles = profiles
    graph._profile_map = {p.id: p for p in profiles}
    
    hybrid = HybridRetriever(bm25, faiss_idx, graph, profiles=profiles)
    
    # Define Evaluation Data
    test_queries = [
        {
            "query": "Looking for a Python Backend Developer",
            "reference": "The system should return candidates with strong Python and backend development skills, ideally with frameworks like Django or FastAPI."
        },
        {
            "query": "Regulatory Affairs Manager with FDA experience",
            "reference": "The system should return candidates who have experience in regulatory affairs and have worked directly with the FDA."
        }
    ]
    
    # RAGAS 0.4.x expects these specific columns
    dataset_dict = {
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "reference": []
    }
    
    print("Running queries through Hybrid Retriever to gather RAGAS data...")
    for item in test_queries:
        query = item["query"]
        results = hybrid.search(query, top_k=3)
        
        # Build contexts from retrieved profiles
        contexts = []
        for res in results:
            p = res.profile
            context_str = f"Name: {p.name}. Roles: {p.potential_roles}. Skills: {p.core_skills}, {p.secondary_skills}. Summary: {p.skill_summary}"
            contexts.append(context_str)
            
        # Our retriever returns a structured object, but RAG systems usually generate a text response.
        # We will use the generated 'explanation' of the top result as a proxy for the LLM response.
        response = results[0].explanation if results and results[0].explanation else "These candidates match your query based on their skills and experience."
        
        dataset_dict["user_input"].append(query)
        dataset_dict["retrieved_contexts"].append(contexts)
        dataset_dict["response"].append(response)
        dataset_dict["reference"].append(item["reference"])
        
    eval_dataset = Dataset.from_dict(dataset_dict)
    
    print("Initializing Gemini LLM and Embeddings...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        # ragas v0.4.x might require explicit wrappers, but usually passing langchain models works.
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
# The client automatically picks up the 'GEMINI_API_KEY' environment variable
        client = genai.Client(api_key="AIzaSyD1D1HQqhchUJO5oyvXSlXDx4zOelnZ-qo")



        llm = llm_factory(
            client=client,
            model="gemini-2.5-flash",
            provider="google"
        )
        embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001",provider="google")
    except ImportError:
        print("[ERROR] Missing required packages for Gemini integration.")
        print("Please run: pip install langchain-google-genai")
        sys.exit(1)
        
    print("Starting RAGAS Evaluation (this makes calls to Gemini)...")
    # Evaluate using standard Ragas metrics
    metrics = [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm,embeddings=embeddings),
    ]
    
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as e:
        print(f"\n❌ RAGAS Evaluation Failed: {str(e)}")
        print("Note: This is often caused by column name mismatches in different RAGAS versions.")
        print("If it complains about missing columns (like 'question' or 'ground_truths'), you may need to map them.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("RAGAS EVALUATION METRICS:")
    print("="*50)
    
    df = result.to_pandas()
    
    # Print overall averages
    print("Overall Averages:")
    for m in metrics:
        metric_name = m.name
        if metric_name in df.columns:
            avg_score = df[metric_name].mean()
            print(f"{metric_name.ljust(20)}: {avg_score:.4f}")
            
    print("\nDetailed breakdown per query:")
    for _, row in df.iterrows():
        print(f"\nQuery: {row.get('user_input', row.get('question', ''))}")
        for m in metrics:
            metric_name = m.name
            if metric_name in row:
                print(f"  - {metric_name.ljust(18)}: {row[metric_name]:.4f}")

if __name__ == "__main__":
    main()
