
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Configure logging in the document processor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents_from_directory(directory_path):   #function to load documents
    """
    Loads all supported documents from a specified directory.
    Supported formats: .pdf, .txt, .md
    """
    documents = []
    logging.info(f"Scanning directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                if filename.endswith(".pdf"):         # PDF files
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    logging.info(f"Successfully loaded PDF: {filename}")       
                elif filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                    logging.info(f"Successfully loaded TXT: {filename}")       # Text files
                elif filename.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                    documents.extend(loader.load())
                    logging.info(f"Successfully loaded MD: {filename}")
            except Exception as e:
                logging.error(f"Failed to load or process {filename}: {e}")       #error message
    
    if not documents:
        logging.warning("No documents were loaded. Check the directory path and file formats.")
        
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits the loaded documents into smaller chunks for processing.
    """
    logging.info(f"Starting to chunk {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunked_documents = text_splitter.split_documents(documents)
    logging.info(f"Successfully chunked documents into {len(chunked_documents)} chunks.")
    return chunked_documents

if __name__ == '__main__':
    # This is for testing the script directly
    # Make sure to adjust the path to your actual data directory relative to where you run this
    # For example, if you run from the root 'sport_bot' folder:
    knowledge_base_dir = 'data/sports_knowledge_base'
    
    if not os.path.exists(knowledge_base_dir):
        logging.error(f"Directory not found: {knowledge_base_dir}. Please create it and add your knowledge base files.")
    else:
        
        docs = load_documents_from_directory(knowledge_base_dir)     #function to load documents

        if docs:
           
            chunked_docs = chunk_documents(docs)   #function to chunk documents

            # Print a sample to verify
            print("\n--- Sample Chunk ---")
            print(chunked_docs[0].page_content)
            print("\n--- Metadata ---")
            print(chunked_docs[0].metadata)
            print(f"\nTotal documents loaded: {len(docs)}")   # Total documents loaded
            print(f"Total chunks created: {len(chunked_docs)}")    # Total chunks created