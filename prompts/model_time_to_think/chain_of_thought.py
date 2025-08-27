from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

target_runs = 180
current_runs = 95
overs_bowled = 12
total_overs = 20


question = f"""
A T20 team needs {target_runs} runs in {total_overs} overs to win.
They are currently {current_runs} after {overs_bowled} overs.
What is the required run rate (runs per over) for the remaining overs to reach exactly {target_runs}?
Round to 2 decimals.
"""


template = """
You are a careful sports-math assistant.
Think through the problem step by step INTERNALLY (do NOT reveal your steps).
Return ONLY the final result in the following format:

Final:
- Required run rate: <number> runs/over
- One-line check: <very short check>

Question:
```{q}```
"""

prompt = ChatPromptTemplate.from_template(template)
messages = prompt.format_messages(q=question)

response = model.invoke(messages)

print(response.content)