# 🤖 Chat with Your Data (RAG Application)

A full-stack **Retrieval-Augmented Generation (RAG)** application built using Python. This project allows users to chat with their custom text data (e.g., notes, articles, or documentation) instead of relying solely on general AI knowledge.

## 🚀 Features
- **Custom Data Ingestion:** Loads text data from a local file (`facts.txt`).
- **Vector Search:** Uses **ChromaDB** to store and retrieve relevant context based on user queries.
- **AI Integration:** Connects to the **HuggingFace Inference API** (Qwen-2.5-7B-Instruct) to generate human-like answers.
- **Interactive UI:** A clean, responsive web interface built with **Streamlit**.
- **Dynamic Updates:** Includes a "Refresh Database" feature to update knowledge without restarting the application.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **Database:** ChromaDB (Vector Store) with FastEmbed
- **LLM:** HuggingFace API

## 📂 Project Structure
- `6_streamlit_app.py` - The main application file containing the UI and RAG logic.
- `reset_database.py` - A utility script to clear and rebuild the vector database.
- `data/facts.txt` - The source file containing the custom data (e.g., F1 history, Biryani facts).
- `requirements.txt` - List of dependencies required to run the project.

## ⚙️ How to Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt