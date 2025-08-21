
import logging
from langchain_core.vectorstores import VectorStore
from src.query_classifier import classify_query
from src.rag_strategies import (
    simple_rag_retrieval,
    comparative_rag_retrieval,
    analytical_rag_retrieval
)

# Configure logging in the decision engine
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def route_query(query: str, vector_store: VectorStore):
    """
    Routes the user's query to the appropriate RAG retrieval strategy.

    Args:
        query (str): The user's question.
        vector_store (VectorStore): The FAISS vector store object.

    Returns:
        tuple: A tuple containing the list of retrieved documents and the determined category.
               Returns (None, "Non-Sport") for non-sport queries.
    """

    category = classify_query(query)   # Classify the user's query

    retrieved_docs = []
    
    # Route to the appropriate retrieval strategy based on the category
    if category == "Factual":       # Factual queries
        logging.info("Routing to: Simple RAG Retrieval")
        retrieved_docs = simple_rag_retrieval(query, vector_store)

    elif category == "Comparative":     # Comparative queries
        logging.info("Routing to: Comparative RAG Retrieval")
        retrieved_docs = comparative_rag_retrieval(query, vector_store)

    elif category == "Analytical":     # Analytical queries
        logging.info("Routing to: Analytical RAG Retrieval")
        retrieved_docs = analytical_rag_retrieval(query, vector_store)

    elif category == "Non-Sport":      # Non-Sport queries
        logging.warning("Query identified as Non-Sport. Halting retrieval.")
        # We return an empty list for docs and the category
        return None, category
        
    else:
        # Default fallback to simple retrieval if category is unrecognized
        logging.warning(f"Unrecognized category '{category}'. Defaulting to Simple RAG.")
        retrieved_docs = simple_rag_retrieval(query, vector_store)

    return retrieved_docs, category

if __name__ == '__main__':
    # This is for testing the script directly
    from src.vector_store import load_vector_store, get_embedding_model

    print("--- Testing Decision Engine ---")

    
    embeddings = get_embedding_model()
    db = load_vector_store(embeddings=embeddings)

    if db:
        queries_to_test = [
            "What is a hat-trick in football?", # Factual
            "Compare the careers of Pele and Maradona", # Comparative
            "Why is possession important in modern football?", # Analytical
            "What's the best recipe for pasta?", # Non-Sport
        ]
        
        for q in queries_to_test:
            print(f"\n--- Processing Query ---\n'{q}'")
            
            docs, cat = route_query(q, db)
            
            print(f"  -> Detected Category: {cat}")
            
            if docs:
                print(f"  -> Retrieval Strategy Executed: Retrieved {len(docs)} documents.")
            else:
                print("  -> Retrieval Halted or No Documents Found.")
    else:
        print("Failed to load vector store. Please run 'python -m src.vector_store' first.")