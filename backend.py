from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from logspot import setup_logs
from flask_cors import CORS

# Load environment variables
load_dotenv()

# -------------------------------
# LOAD EMBEDDINGS + FAISS STORE
# -------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "medical_faiss",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# -------------------------------
# FORMAT DOCUMENTS (CONTEXT)
# -------------------------------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# -------------------------------
# BUILD RAG CHAIN
# -------------------------------
parallel_chain = RunnableParallel({
    "context": RunnableSequence(retriever, RunnableLambda(format_docs)),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

prompt = PromptTemplate(
    template="""You are a helpful medical assistant.
Answer ONLY using the provided context. 
If the context does not contain the answer, reply: "I don't know."

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

chatbot = RunnableSequence(parallel_chain, prompt, model, parser)


# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "https://medaiui.vercel.app"]
    }
})


logger = setup_logs(app,service="medai",telegram_chat_id=os.getenv("CHATID"))

@app.route("/")
def home():
    return jsonify({
        "message": "Medical RAG API is running!",
        "endpoints": {
            "/ask (POST)": "Send a question and get model answer."
        }
    })



@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()

    if not data or "question" not in data:
        logger.error(msg="Please send JSON with a 'question' field.", notify=True)
        return jsonify({"error": "Please send JSON with a 'question' field."}), 400

    user_question = data["question"]
    logger.info(msg=user_question)
    try:
        answer = chatbot.invoke(user_question)
        logger.info(answer)
        return jsonify({
            "question": user_question,
            "answer": answer
        })
    except Exception as e:
        logger.error(str(e), notify=True)
        return jsonify({"error": str(e)}), 500


# -------------------------------
# START SERVER
# -------------------------------
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
