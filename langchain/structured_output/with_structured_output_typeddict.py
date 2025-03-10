from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional,Literal,List

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Review(TypedDict):
    key_themes: Annotated[List[str], "The key themes of the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The sentiment of the review"]
    pros: Annotated[Optional[List[str]], "The pros of the product"]
    cons: Annotated[Optional[List[str]], "The cons of the product"]

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("I hate this product")

print(result)
