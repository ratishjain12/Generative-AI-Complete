from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions = 300)

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about virat kohli"
query_embedding = embedding.embed_query(query)

docs_embedding = embedding.embed_documents(docs)

scores = cosine_similarity([query_embedding],docs_embedding)[0]

index, score = sorted(list(enumerate(scores)), key= lambda x:x[1])[-1]

print("similar doc: ", docs[index])
print("similarity score: ", score)