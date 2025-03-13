from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt1 = PromptTemplate(
    template="Generate a detailed report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a brief summary of: \n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({"topic": "black hole"})

print(result)

chain.get_graph().print_ascii()
