from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "What is capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vectors = embedding.embed_documents(docs)

print(str(vectors))