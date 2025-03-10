from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


schema = [
    ResponseSchema(name="Fact 1", description="Fact 1 about the topic"),
    ResponseSchema(name="Fact 2", description="Fact 2 about the topic"),
    ResponseSchema(name="Fact 3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Generate 3 facts about {topic} \n {format_instructions}",
    input_variables = ["topic"],
    partial_variables = {"format_instructions":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic":"Machine Learning"})

print(result)













