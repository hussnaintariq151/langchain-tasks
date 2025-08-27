from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

template = """
You are an assistant that classifies cricket player roles.

Examples:
Input: "Virat Kohli"
Output: "Batsman"

Input: "Jasprit Bumrah"
Output: "Bowler"

Input: "MS Dhoni"
Output: "Wicketkeeper Batsman"

Now classify the following player:

Input: "{player}"
Output:
"""

prompt = ChatPromptTemplate.from_template(template)

player_name = "Rohit Sharma"
final_prompt = prompt.format_messages(player=player_name)

response = model.invoke(final_prompt)

print(response.content)
