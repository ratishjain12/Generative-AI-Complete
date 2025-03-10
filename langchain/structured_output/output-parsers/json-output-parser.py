from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of a fictional person\n {format_instructions}",
    input_variables = [],
    partial_variables = {"format_instructions":parser.get_format_instructions()}
)

# result = chain.invoke("Machine Learning")

chain = template | model | parser
parsed_result = chain.invoke({})
print(parsed_result)
print(type(parsed_result))










