# with_structured_output_json_small.py

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize a lighter model
model = ChatOpenAI(model="gpt-4o-mini")

# Minimal JSON schema
json_schema = {
    "title": "CricketReview",
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Short summary of the review"},
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neutral"],
            "description": "Overall sentiment"
        },
        "name": {"type": "string", "description": "Reviewer name"}
    },
    "required": ["summary", "sentiment", "name"]
}

structured_model = model.with_structured_output(json_schema)

review_text = """
Virat Kohli played a brilliant knock of 82* against Pakistan. 
India’s bowling was sharp, and the crowd was amazing. 
Only concern: middle-order failed again. 

Review by Rao Ibrar Jamal
"""

response = structured_model.invoke(review_text)

print(response)
