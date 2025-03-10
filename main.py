import requests
import json
import faiss
import os
import numpy as np
from langchain import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
import fitz  # PyMuPDF for PDF parsing
from sentence_transformers import SentenceTransformer
import datetime
 
st.set_page_config(page_title="Technical Query Chatbot", layout="wide")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
 
url = "https://api.openai.com/v1/chat/completions"
 
# Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
 
# Data Collection & Preparation
def fetch_academic_resources(query):
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "Provide academic content related to: " + query}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Failed to fetch data"
 
# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text
 
# Sample Queries
queries = ["Python programming", "Machine learning algorithms", "Database normalization"]
documents = [fetch_academic_resources(query) for query in queries]
 
# Embedding & Vector Store
vectors = embedding_model.encode(documents)
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype('float32'))
 
# Creating In-Memory Docstore
docstore = InMemoryDocstore({str(i): Document(page_content=doc) for i, doc in enumerate(documents)})
 
# Creating index-to-docstore mapping
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
 
# Initialize FAISS Vector Store
faiss_store = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding_model.encode)
 
# RAG Model Function
def query_rag_model(query):
    query_vector = embedding_model.encode(query)
    _, indices = index.search(np.array([query_vector]).astype('float32'), k=1)
    retrieved_doc = documents[indices[0][0]]
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "Answer the query based on the following document: " + retrieved_doc},
                      {"role": "user", "content": query}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Failed to generate response"
 
# Evaluation & Testing
def evaluate_model(queries, ground_truths):
    predictions = [query_rag_model(query) for query in queries]
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='weighted', zero_division=0)
    recall = recall_score(ground_truths, predictions, average='weighted', zero_division=0)
    f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
 
# Fine-Tuning & Optimization
def fine_tune_model(queries, temperature=0.5):
    optimized_documents = []
    for query in queries:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "system", "content": "Provide more detailed and accurate academic content related to: " + query}],
            "temperature": temperature
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            optimized_documents.append(response.json()["choices"][0]["message"]["content"])
        else:
            optimized_documents.append("Failed to fetch data")
    return optimized_documents
 
# Deployment & Integration with Streamlit
def main():
    st.title("Technical Query Chatbot")
    st.sidebar.title("Chat History")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = {}
 
    user_query = st.text_input("Enter your query:")
 
    if st.button("Submit"):
        if user_query:
            response = query_rag_model(user_query)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            time = datetime.datetime.now().strftime("%H:%M:%S")
            if timestamp not in st.session_state['chat_history']:
                st.session_state['chat_history'][timestamp] = []
            st.session_state['chat_history'][timestamp].append((time, user_query, response))
            st.write("Response:", response)
 
    for date, chats in reversed(st.session_state['chat_history'].items()):
        with st.sidebar.expander(f"{date}"):
            for time, query, response in chats:
                st.text(f"Query: {query}")
                st.text(f"Response: {response}")
 
    if st.sidebar.button("New Chat"):
        st.session_state['chat_history'] = {}
        st.rerun()
 
if __name__ == "__main__":
    main()