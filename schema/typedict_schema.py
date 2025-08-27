from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


# Typedict schema 
class Review(TypedDict):
    summary: Annotated[str, "Short summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Overall sentiment of the review"]
    name: Annotated[str, "Reviewer name"]



structured_model = model.with_structured_output(Review)

review_text = """
Virat Kohli’s 82* against Pakistan was magical! 
India’s bowling was sharp, but the middle-order struggled again. 

Review by Ahmed
"""

result = structured_model.invoke(review_text)

print(result)

