from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional,Literal,List
from pydantic import BaseModel,Field
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Review(BaseModel):
    key_themes: List[str] = Field(description="The key themes of the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review")
    pros: Optional[List[str]] = Field(description="The pros of the product")
    cons: Optional[List[str]] = Field(description="The cons of the product")

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("I hate this product")

print(result)
