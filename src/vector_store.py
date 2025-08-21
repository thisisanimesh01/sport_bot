
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.document_processor import load_documents_from_directory, chunk_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
# We use a sentence-transformer model for creating the embeddings.
# 'all-MiniLM-L6-v2' is a great starting point as it's small, fast, and effective.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store/faiss_index"

def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """
    Loads the sentence-transformer model from Hugging Face.
    """
    logging.info(f"Loading embedding model: {model_name}")
    # We specify 'cpu' as the device to ensure it runs on standard laptops
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logging.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        return None

def create_and_save_vector_store(documents, embeddings, store_path=VECTOR_STORE_PATH):
    """
    Creates a FAISS vector store from documents and saves it to disk.
    """
    if not documents:
        logging.error("No documents provided to create vector store.")
        return None
        
    try:
        # Create the vector store from the document chunks
        logging.info("Creating FAISS vector store...")
        db = FAISS.from_documents(documents, embeddings)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        
        # Save the vector store locally
        db.save_local(store_path)
        logging.info(f"Vector store created and saved successfully at: {store_path}")
        return db
    except Exception as e:
        logging.error(f"Failed to create and save vector store: {e}")
        return None

def load_vector_store(store_path=VECTOR_STORE_PATH, embeddings=None):
    """
    Loads an existing FAISS vector store from disk.
    """
    if not os.path.exists(store_path):
        logging.error(f"Vector store not found at path: {store_path}")
        return None
        
    if not embeddings:
        logging.error("Embeddings model must be provided to load the vector store.")
        return None

    try:
        logging.info(f"Loading vector store from: {store_path}")
        db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        logging.info("Vector store loaded successfully.")
        return db
    except Exception as e:
        logging.error(f"Failed to load vector store: {e}")
        return None

if __name__ == '__main__':
    # This script can be run directly to build the vector store for the first time.
    
  
    knowledge_base_dir = 'data/sports_knowledge_base'     # Path to the sports knowledge base


    docs = load_documents_from_directory(knowledge_base_dir)     #function to load documents
    chunked_docs = chunk_documents(docs)     #function to chunk documents

    if chunked_docs:

        embeddings_model = get_embedding_model()    #function to get embedding model

        if embeddings_model:      # Check if embeddings model loaded successfully
            print("\nStarting the creation of the vector store. This may take a few minutes...")
            create_and_save_vector_store(chunked_docs, embeddings_model)
            print("\nProcess finished.")
    else:
        print("No documents were found or processed. Cannot create vector store.")