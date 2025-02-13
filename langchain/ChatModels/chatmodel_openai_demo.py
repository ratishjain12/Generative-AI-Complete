from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4")
# you can give temperature and max completion tokens as params

result = model.invoke("What is the capital of France?")

print(result.content)