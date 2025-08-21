🏆 Sports Intelligence Chatbot 

This project is an intelligent chatbot agent built for the AI Agent Development Internship Coding Challenge. It's an "Ask Me Anything" expert system for a specific sport (e.g., Football/Soccer), capable of answering a wide range of questions by leveraging a local knowledge base through a Retrieval-Augmented Generation (RAG) pipeline.

The agent can autonomously classify user queries and decide the best strategy to retrieve relevant information before generating a contextually accurate response.

🏛️ Architecture Overview
The chatbot operates on a RAG pipeline, which allows the Large Language Model (LLM) to access and use external knowledge before generating a response. This prevents hallucination and ensures the answers are based on factual data provided in the knowledge base.

The workflow is as follows:

Knowledge Base Ingestion: Documents (PDF, TXT, MD) from the data/sports_knowledge_base directory are loaded and split into smaller, manageable chunks.

Vector Embeddings: Each text chunk is converted into a numerical vector representation (embedding) using a sentence-transformer model from Hugging Face.

Vector Store: These embeddings are stored in a FAISS vector database, which allows for efficient similarity searches.

User Query: The user asks a question through the Streamlit interface.

Query Classification: The agent's "Decision Engine" first classifies the query into a category (e.g., Factual, Comparative, Analytical, Non-Sport).

Strategic Retrieval: Based on the category, the engine selects an appropriate RAG strategy to search the vector store for the most relevant document chunks.

Context Augmentation: The retrieved chunks (context) and the original query are combined into a detailed prompt.

Response Generation: The complete prompt is sent to a local LLM (e.g., Google's Flan-T5), which generates a human-like answer based only on the provided context.

Graceful Handling: If a query is classified as "Non-Sport," the process is halted, and a polite refusal is returned.

[User Query] -> [Query Classifier] -> [Decision Engine] -> [RAG Strategy] -> [Vector Store]
      ^                                                                         |
      |                                                                         v
[Final Answer] <- [LLM Generation] <- [Prompt Augmentation] <- [Retrieved Documents]

✨ Features
Autonomous Decision-Making: Automatically classifies user queries and routes them to the best retrieval strategy.

Multi-Format Document Support: Ingests knowledge from PDF, TXT, and Markdown files.

Local & Offline: Runs entirely on a local machine after initial model downloads. No API keys required.

Efficient Vector Search: Utilizes FAISS for fast and memory-efficient information retrieval.

Domain-Specific Focus: Gracefully handles and declines questions outside the sports domain.

Interactive UI: A simple and user-friendly chat interface built with Streamlit.

🛠️ Tech Stack
Core Framework: LangChain

LLM & Embeddings: Hugging Face (google/flan-t5-small, sentence-transformers/all-MiniLM-L6-v2)

Vector Database: FAISS (Facebook AI Similarity Search)

UI: Streamlit

Document Loaders: pypdf, Unstructured

🚀 Setup and Installation
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.9+

pip (Python package installer)

1. Clone the Repository
git clone https://github.com/your-username/sport_bot.git
cd sport_bot

2. Create a Virtual Environment (Recommended)
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

⚙️ Usage
1. Add Your Knowledge Base
Place your sports-related documents (PDF, TXT, or MD files) inside the data/sports_knowledge_base/ directory. The application will automatically detect and process them.

Important: This directory must contain at least one document for the chatbot to work.

2. Build the Vector Store
Before running the chatbot for the first time, you need to process your documents and create the FAISS vector index.

Run the following command from the root directory of the project:

python -m src.vector_store

This will create a vector_store/ directory containing the faiss_index. You only need to run this script once, or whenever you add, remove, or change the documents in your knowledge base.

3. Launch the Chatbot
Start the interactive Streamlit application with the following command:

streamlit run demo.py

Your web browser will automatically open with the chatbot interface, ready for you to ask questions!

🧠 Design Decisions
Local LLM (google/flan-t5-small): This model was chosen because it is small enough to run on a standard laptop with <= 8GB of RAM, satisfying the project's technical constraints. It's an instruction-tuned model, which makes it effective at following the RAG prompt.

Keyword-Based Query Classifier: A simple, rule-based classifier was implemented for speed and reliability. It avoids the overhead of training a separate ML model for classification and is highly effective for distinguishing between the required query types (Factual, Comparative, etc.).

Sentence Transformers for Embeddings: The all-MiniLM-L6-v2 model provides an excellent balance of performance and speed for generating high-quality embeddings, making it ideal for this local-first application.

FAISS for Vector Store: FAISS is incredibly fast and memory-efficient, making it a perfect choice for running similarity searches on a local machine without requiring a dedicated database server.

Streamlit for UI: Streamlit was chosen for its ability to rapidly create an interactive and user-friendly web interface with minimal code, allowing for a focus on the core AI logic.

📁 Directory Structure
sport_bot/
├── src/
│   ├── document_processor.py   # Loads and chunks documents
│   ├── vector_store.py         # Creates and manages the FAISS vector store
│   ├── query_classifier.py     # Classifies user queries
│   ├── rag_strategies.py       # Defines different retrieval methods
│   ├── decision_engine.py      # Routes queries to the correct strategy
│   └── sports_chatbot.py       # Core chatbot logic and RAG chain
├── data/
│   └── sports_knowledge_base/  # Place your PDF, TXT, MD files here
├── vector_store/
│   └── faiss_index/            # Generated by vector_store.py
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── demo.py                     # The Streamlit application entry point

