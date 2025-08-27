from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

passthrough = RunnablePassthrough()

chain = (
    passthrough
    | llm
    | RunnableLambda(lambda x: {"text": x.content})  # extract text from LLM response
)

user_input = "Explain black holes in simple words."
result = chain.invoke(user_input)

print(result["text"])
