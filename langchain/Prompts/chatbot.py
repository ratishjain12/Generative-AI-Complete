from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

chat_history = [
    SystemMessage("You are helpful AI Assistant")
]
while True:
    user = input("user: ")
    chat_history.append(HumanMessage(content=user))
    if(user == 'exit'):
        break;
    result = model.invoke(chat_history);
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)
