from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder("chat_history"),
    ('human', '{topic}')
])

chat_history = []

with open('chat_history.txt','r') as f:
    chat_history.extend(f.readlines());


prompt = chat_template.invoke({
    'chat_history': chat_history,
    'topic': 'where is my refund',
})

print(prompt)

