
import logging         
from langchain_core.vectorstores import VectorStore    

# Configure logging in the RAG strategies
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simple_rag_retrieval(query: str, vector_store: VectorStore, k: int = 4):   # Function to perform simple RAG retrieval
    """
    Performs a simple similarity search on the vector store.

    Args:
        query (str): The user's question.
        vector_store (VectorStore): The FAISS vector store object.
        k (int): The number of relevant documents to retrieve.

    Returns:
        list: A list of retrieved document chunks.
    """
    if not vector_store:      #  if vector store is available
        logging.error("Vector store is not available.")
        return []

    logging.info(f"Performing simple RAG retrieval for query: '{query}'")     # Log the query being processed
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    try:
        relevant_docs = retriever.invoke(query)
        logging.info(f"Retrieved {len(relevant_docs)} documents for the query.")
        return relevant_docs
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return []

def comparative_rag_retrieval(query: str, vector_store: VectorStore, k: int = 6):   # Function for comparative RAG retrieval
    """
    A strategy for comparative questions. 
    It could involve searching for both entities separately and combining results.
    For now, we'll use a slightly enhanced simple retrieval.
    """
    # A more advanced implementation might parse entities (e.g., "Messi", "Ronaldo")
    # and perform separate searches. For this version, we retrieve more documents
    # to increase the chance of finding info on both.
    logging.info(f"Performing comparative RAG retrieval for query: '{query}'")
    return simple_rag_retrieval(query, vector_store, k=k)

def analytical_rag_retrieval(query: str, vector_store: VectorStore, k: int = 5):   # Function for analytical RAG retrieval
    """
    A strategy for analytical questions requiring reasoning.
    This might involve multi-hop reasoning in a more advanced system.
    For now, we retrieve a moderate number of documents to provide broader context.
    """
    logging.info(f"Performing analytical RAG retrieval for query: '{query}'")
    return simple_rag_retrieval(query, vector_store, k=k)


if __name__ == '__main__':      
    # This is for testing the script directly
    # You need the vector store to be created first (run vector_store.py)
    from src.vector_store import load_vector_store, get_embedding_model

    print("--- Testing RAG Strategies ---")

    embeddings = get_embedding_model()       # Load the embedding model
    db = load_vector_store(embeddings=embeddings)       # Load the vector store

    if db:

        test_query = "What is the offside rule in football?"     # Define a test query

        retrieved_docs = simple_rag_retrieval(test_query, db)
        
        if retrieved_docs:
            print(f"\n--- Test Query ---\n'{test_query}'")
            print(f"\n--- Retrieved {len(retrieved_docs)} Documents (Top 2 shown) ---")
            
            for i, doc in enumerate(retrieved_docs[:2]):
                print(f"\n--- Document {i+1} ---")
                print(doc.page_content)
                print(f"Source: {doc.metadata.get('source', 'N/A')}")
        else:
            print("Could not retrieve any documents. Is your knowledge base relevant to the query?")
    else:
        print("Failed to load vector store. Please run 'python -m src.vector_store' first.")