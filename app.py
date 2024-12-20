from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

URI = "tcp://localhost:19530"  
collection_name = "faq_collection2"

from langchain_milvus import Milvus

vector_store = Milvus(
    embeddings,
    connection_args={"uri": URI},
    collection_name=collection_name,
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

from rag import answer_question
import streamlit as st

st.title("💬 FAQ_INOVA")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Tire aqui suas dúvidas sobre o CPSI"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    results = answer_question(prompt, retriever)
    msg = results['answer']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)