from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

template = """
You are a helpful assistant.

Read the text enclosed in triple backticks and summarize it in 3 concise bullet points.

```{text}```
"""

prompt = ChatPromptTemplate.from_template(template)

input_text = """
Babar Azam is one of the most successful Pakistani cricketers. 
He is known for his sensible batting style, remarkable consistency, 
and gentle qualities. He has broken many records and is considered 
one of the modern greats of the game.
"""

final_prompt = prompt.format_messages(text=input_text)

print (final_prompt)

response = model.invoke(final_prompt)

print(response.content)