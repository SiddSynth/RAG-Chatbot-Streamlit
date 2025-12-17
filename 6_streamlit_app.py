# 6_streamlit_app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

# 1. Setup Page
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Chat with your Data (RAG)")

# 2. Load Secrets
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    st.error("⚠️ HUGGINGFACEHUB_API_TOKEN not found in .env file!")
    st.stop()

# 3. Setup Backend (Cached for speed)
@st.cache_resource
def load_db():
    print("⏳ Loading Vector Database...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_fastembed")
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    db = Chroma(
        persist_directory=persistent_directory, 
        embedding_function=embedding_model
    )
    return db

db = load_db()
client = InferenceClient(api_key=api_key)

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Logic
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 Thinking...")

        try:
            # A. Retrieve Context
            retriever = db.as_retriever(search_kwargs={"k": 2})
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join(doc.page_content for doc in docs)

            # B. Prepare Prompt for Qwen
            system_prompt = "You are a helpful assistant. Answer based ONLY on the context provided. If unsure, say 'I don't know'."
            user_message = f"Context:\n{context_text}\n\nQuestion:\n{prompt}"

            # C. Call Qwen (Chat Mode)
            response = client.chat_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Show Answer
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            message_placeholder.error(f"Error: {e}")