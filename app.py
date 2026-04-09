import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

# -------------------------------
# LOAD EMBEDDINGS + FAISS
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "medical_faiss",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    return vector_store

vector_store = load_vectorstore()

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# -------------------------------
# FORMAT CONTEXT
# -------------------------------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# -------------------------------
# RAG CHAIN
# -------------------------------
parallel_chain = RunnableParallel({
    "context": RunnableSequence(retriever, RunnableLambda(format_docs)),
    "question": RunnablePassthrough()
})

prompt = PromptTemplate(
    template="""You are a medical assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT guess
- If answer not found, say "I don't know"

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

parser = StrOutputParser()

chatbot = RunnableSequence(parallel_chain, prompt, model, parser)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Medical AI", page_icon="🩺")

st.title("🩺 Medical AI Assistant")

# session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
user_input = st.chat_input("Ask your medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chatbot.invoke(user_input)
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

            st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })