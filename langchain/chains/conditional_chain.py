from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the following feedback as positive or negative\n\n{feedback} \n\n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Generate a response for the positive feedback\n\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Generate a response for the negative feedback\n\n{feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not classify feedback")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I hate this product!"})
print(result)

