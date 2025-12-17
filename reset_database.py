import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Define base paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_folder = os.path.join(current_dir, "db", "chroma_fastembed")
data_file = os.path.join(current_dir, "data", "facts.txt")

def reset_database():
    """
    Resets the vector database by deleting the existing folder
    and re-ingesting data from facts.txt.
    """
    
    # 1. Clear the existing database
    if os.path.exists(db_folder):
        print("🗑️  Removing existing vector database...")
        shutil.rmtree(db_folder)
    else:
        print("⚠️  No existing database found. Creating a new one...")

    # 2. Load the raw data
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file not found at {data_file}")
        return

    print("📂 Loading data from 'facts.txt'...")
    loader = TextLoader(data_file)
    docs = loader.load()

    # 3. Split data into chunks (Chunking)
    print("✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    # 4. Re-build the Vector Store (Embedding)
    print("⚙️  Initializing FastEmbed and creating vector store...")
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=db_folder
    )

    print("✅ Success! Database has been refreshed. You can now run the Streamlit app.")

if __name__ == "__main__":
    reset_database()