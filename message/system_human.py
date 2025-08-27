
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="You are a cricket expert."),
    HumanMessage(content="Who is Babar Azam?")
]

response = model.invoke(messages)
print(response.content)
