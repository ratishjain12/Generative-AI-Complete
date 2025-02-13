from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)

#simple text
result = embedding.embed_query("What is capital of India")

print("text: ",str(result))
#docs
docs = [
    "What is capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(docs)

print("docs: ",str(result))



