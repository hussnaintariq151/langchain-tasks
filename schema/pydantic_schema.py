from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini")

# Define schema with Pydantic
class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed in the review")
    summary: str = Field(description="A short summary of the review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Overall sentiment of the review")
    pros: Optional[list[str]] = Field(default=None, description="All pros mentioned in the review")
    cons: Optional[list[str]] = Field(default=None, description="All cons mentioned in the review")
    name: Optional[str] = Field(default=None, description="Name of the reviewer")

# Structured model
structured_model = model.with_structured_output(Review)

# Example cricket review
review_text = """
India’s recent match against Australia was outstanding! 
Virat Kohli played a brilliant knock under pressure and guided India home with a calm finish. 
Bumrah’s yorkers were unplayable and restricted the Aussies in the death overs. 

However, India’s middle-order struggled once again, and two easy catches were dropped in the field. 

Pros:
Kohli’s match-winning innings
Bumrah’s exceptional bowling

Cons:
Weak middle-order
Poor fielding efforts

Review by Rahul Sharma
"""

# Run model
result = structured_model.invoke(review_text)


print(result)


# print(f"Reviewer   : {result.name}")
# print(f"Sentiment  : {result.sentiment}")
# print(f"Summary    : {result.summary}")
