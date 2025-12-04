from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

loader = PyPDFLoader("Medical_book.pdf")
docs = loader.load()
# print(len(docs))
# print(docs[587].page_content)
# print(docs[587].metadata)

contents = [d.page_content for d in docs]
full_text = " ".join([d.page_content for d in docs])

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
chunks = splitter.create_documents([full_text])

# print(chunks)

embedding =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks,embedding)

vector_store.save_local("medical_faiss")

