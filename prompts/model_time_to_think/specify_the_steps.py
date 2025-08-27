from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model="gpt-4")

template = """
You are an assistant that solves problems by following steps.

Follow these steps:
1. Understand the problem.
2. Extract key numbers and entities.
3. Perform calculations step by step.
4. Give the final answer clearly.

Problem: {problem}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

problem = "A train travels 60 km in 1.5 hours. What is its average speed?"

final_prompt = prompt.format_messages(problem=problem)

response = model.invoke(final_prompt)

print(response.content)
