
import logging
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.vector_store import load_vector_store, get_embedding_model
from src.decision_engine import route_query

# Configure logging in the sports chatbot
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SportsChatbot:     # SportsChatbot class definition
    def __init__(self):
        """
        Initializes the chatbot by loading the necessary models and vector store.
        """
        logging.info("Initializing Sports Chatbot...")
        self.embeddings = get_embedding_model()
        self.vector_store = load_vector_store(embeddings=self.embeddings)
        
        # Check if vector store was loaded successfully
        if self.vector_store is None:
            raise RuntimeError("Failed to load vector store. Ensure 'vector_store/faiss_index' exists.")

        # Load the local LLM for response generation
        # We use a smaller model to ensure it runs on a standard laptop (<= 8GB RAM)
        logging.info("Loading local LLM...")
        try:
            self.llm = HuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-small",
                task="text2text-generation",
                pipeline_kwargs={"max_new_tokens": 200},
            )
            logging.info("LLM loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load LLM: {e}")
            raise RuntimeError("Could not load the language model. Check your internet connection for the first download or model files.")
        
        # Define the prompt template for the RAG chain
        self.prompt_template = self._create_prompt_template()
        
        # Build the RAG chain
        self.rag_chain = self._create_rag_chain()

    def _create_prompt_template(self):
        """Creates the prompt template for the RAG chain."""
        template = """
        You are an expert sports assistant. Use the following pieces of context to answer the user's question.
        If you don't know the answer from the context provided, just say that you don't have enough information.
        Keep the answer concise and relevant.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _format_docs(self, docs):
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_rag_chain(self):
        """Creates the full RAG chain for processing queries."""
        return (
            {"context": self._retriever_wrapper, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def _retriever_wrapper(self, query: str):
        """
        A wrapper that uses the decision engine to retrieve docs and then formats them.
        """
        retrieved_docs, category = route_query(query, self.vector_store)
        
        # If the query is non-sport, we will have no docs.
        # The chain will continue but the context will be empty.
        if category == "Non-Sport":
            return "NON_SPORT_QUERY" # Special flag for non-sport queries

        if not retrieved_docs:
            return "No relevant information found in the knowledge base."
            
        return self._format_docs(retrieved_docs)

    def answer(self, query: str):
        """
        The main method to get an answer from the chatbot.
        """
        logging.info(f"Received query: {query}")
        
        # First, check for non-sport queries explicitly to provide a graceful response.
        category = classify_query(query) # We need to import this
        if category == "Non-Sport":
            logging.warning("Non-sport query detected. Replying gracefully.")
            return "I am a sports intelligence chatbot and can only answer questions related to sports. Please ask me something about sports!"
        
        # If it's a sports query, invoke the RAG chain
        response = self.rag_chain.invoke(query)
        
        # Final check in case the response is empty or model refuses to answer
        if not response or "don't have enough information" in response.lower():
            return "I couldn't find a specific answer in my knowledge base. Can you try rephrasing the question?"
        
        return response

# We need to add the import for classify_query
from src.query_classifier import classify_query

if __name__ == '__main__':
    # This block is for testing the chatbot directly
    try:
        print("--- Testing Sports Chatbot ---")
        bot = SportsChatbot()
        
        print("\nBot Initialized. Ask a question (or type 'exit' to quit).")
        
        # Example query for testing
        test_query = "What is the offside rule in football?"
        print(f"\n[USER]: {test_query}")
        
        answer = bot.answer(test_query)
        print(f"[BOT]: {answer}")

        # Example non-sport query
        test_query_non_sport = "What is the capital of France?"
        print(f"\n[USER]: {test_query_non_sport}")
        answer_non_sport = bot.answer(test_query_non_sport)
        print(f"[BOT]: {answer_non_sport}")

    except RuntimeError as e:
        print(f"An error occurred during initialization: {e}")    # Log initialization errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")      # Log unexpected errors