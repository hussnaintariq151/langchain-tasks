
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

from langchain.schema.runnable import RunnableLambda


llm = ChatOpenAI(model="gpt-4o-mini")

# Wrap a Python function inside RunnableLambda
# This function takes input text and makes it uppercase

uppercase_func = RunnableLambda(lambda x: {"text": x["text"].upper()})

chain = (
    llm
    | RunnableLambda(lambda x: {"text": x.content})   # extract response text
    | uppercase_func                                  # make it uppercase
)

result = chain.invoke("Explain quantum computing in simple words.")
print(result)
