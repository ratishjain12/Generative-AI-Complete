from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18,description="The age of the person")
    city: str = Field(description="The city of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate the name, age and city of a fictional {place} person\n {format_instructions}",
    input_variables = ["place"],
    partial_variables = {"format_instructions":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"place":"India"})

print(result)


















