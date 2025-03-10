from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

Review = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {
            "type": "string"
        },
        "sentiment": {
            "type": "string"
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"}
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}
structured_model = model.with_structured_output(Review)


result = structured_model.invoke("I hate this product")

print(result)
