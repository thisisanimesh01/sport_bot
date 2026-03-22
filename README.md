# 🏆 Sports Intelligence Chatbot

An intelligent chatbot agent built for the **AI Agent Development Internship Coding Challenge**.  
This is an **"Ask Me Anything" expert system** for a specific sport (e.g., Football/Soccer), capable of answering a wide range of questions by leveraging a **local knowledge base** through a **Retrieval-Augmented Generation (RAG) pipeline**.

The agent autonomously classifies user queries and decides the best strategy to retrieve relevant information before generating a contextually accurate response.

---

## 🏛️ Architecture Overview

The chatbot operates on a **RAG pipeline**, which allows the LLM to access and use external knowledge before generating a response.  
This prevents hallucination and ensures answers are based on factual data provided in the knowledge base.

**Workflow:**

1. **Knowledge Base Ingestion** – Documents (PDF, TXT, MD) from the `data/sports_knowledge_base/` directory are loaded and split into smaller, manageable chunks.
2. **Vector Embeddings** – Each chunk is converted into a vector embedding using a Hugging Face sentence-transformer model.
3. **Vector Store** – Embeddings are stored in a **FAISS vector database** for efficient similarity searches.
4. **User Query** – Users ask a question via the **Streamlit interface**.
5. **Query Classification** – The "Decision Engine" classifies the query (Factual, Comparative, Analytical, Non-Sport).
6. **Strategic Retrieval** – Based on classification, the engine retrieves relevant chunks from FAISS.
7. **Context Augmentation** – Retrieved chunks + query are merged into a detailed prompt.
8. **Response Generation** – The prompt is sent to a local LLM (e.g., Flan-T5) to generate the final response.
9. **Graceful Handling** – If query is "Non-Sport," the chatbot politely declines.

**Pipeline Flow:**

```
[User Query] -> [Query Classifier] -> [Decision Engine] -> [RAG Strategy] -> [Vector Store]
      ^                                                                         |
      |                                                                         v
[Final Answer] <- [LLM Generation] <- [Prompt Augmentation] <- [Retrieved Documents]
```

---

## ✨ Features

- **Autonomous Decision-Making** – Classifies queries & selects best retrieval strategy.  
- **Multi-Format Document Support** – Works with PDF, TXT, Markdown files.  
- **Local & Offline** – Runs fully on local machine, no API keys required.  
- **Efficient Vector Search** – Uses FAISS for high-speed retrieval.  
- **Domain-Specific Focus** – Handles non-sport queries gracefully.  
- **Interactive UI** – Simple chat interface powered by Streamlit.  

---

## 🛠️ Tech Stack

- **Core Framework** – [LangChain](https://www.langchain.com/)  
- **LLM & Embeddings** – [Hugging Face](https://huggingface.co/)  
  - `google/flan-t5-small` (LLM)  
  - `sentence-transformers/all-MiniLM-L6-v2` (Embeddings)  
- **Vector Database** – [FAISS](https://github.com/facebookresearch/faiss)  
- **UI** – [Streamlit](https://streamlit.io/)  
- **Document Loaders** – `pypdf`, `unstructured`  

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.9+  
- pip (Python package installer)  

### 1. Clone the Repository
```bash
git clone https://github.com/thisisanimesh01/sport_bot.git
cd sport_bot
```

### 2. Create a Virtual Environment (Recommended)

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
.env\Scriptsctivate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Usage

### 1. Add Your Knowledge Base
Place your **sports-related documents (PDF, TXT, MD)** inside:
```
data/sports_knowledge_base/
```

⚠️ Must contain at least **one document** for the chatbot to work.

---

### 2. Build the Vector Store
```bash
python -m src.vector_store
```
This generates a `vector_store/` directory with the FAISS index.  
Re-run if you **add, remove, or modify** documents.

---

### 3. Launch the Chatbot
```bash
streamlit run demo.py
```

This opens the chatbot interface in your web browser. ✅

---

## 🧠 Design Decisions

- **Local LLM (`google/flan-t5-small`)** – Runs on standard laptops with ≤ 8GB RAM. Instruction-tuned for better RAG performance.  
- **Keyword-Based Query Classifier** – Simple rule-based system, fast and reliable.  
- **Sentence Transformers (`all-MiniLM-L6-v2`)** – Balance of speed and embedding quality.  
- **FAISS Vector Store** – High-speed similarity search without requiring a full DB server.  
- **Streamlit UI** – Quick and user-friendly chat interface.  

---

## 📁 Directory Structure

```
sport_bot/
├── src/
│   ├── document_processor.py   # Loads and chunks documents
│   ├── vector_store.py         # Creates and manages the FAISS vector store
│   ├── query_classifier.py     # Classifies user queries
│   ├── rag_strategies.py       # Defines retrieval methods
│   ├── decision_engine.py      # Routes queries to correct strategy
│   └── sports_chatbot.py       # Core chatbot logic & RAG chain
├── data/
│   └── sports_knowledge_base/  # Place sports docs here
├── vector_store/
│   └── faiss_index/            # Generated by vector_store.py
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── demo.py                     # Streamlit app entry point
```

---

## 📌 Future Improvements
- Add support for **multi-sport query handling**.  
- Improve query classifier with a **small ML model**.  
- Integrate advanced **retrieval strategies** for analytical queries.  




To make it more usable in various sports more files to be added soon..! 
