from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


prompt1 = PromptTemplate(
    template="Generate notes on: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a quiz on: {topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the notes and quiz into a single document\n notes -> {notes}\n quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"topic": "black hole"})

print(result)

chain.get_graph().print_ascii()





