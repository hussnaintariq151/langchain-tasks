from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="You are a cricket expert."),
    HumanMessage(content="Who is Babar Azam?"),
    AIMessage(content="He is a Pakistani cricketer, known for his batting."),
    HumanMessage(content="What are some of his achievements?")
]

response = model.invoke(messages)
print(response.content)
