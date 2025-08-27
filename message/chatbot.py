from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

# Start conversation history with a system message
conversation = [
    SystemMessage(content="You are a helpful cricket expert. Answer briefly.")
]

# 3. Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break
    
    # Add human message to history
    conversation.append(HumanMessage(content=user_input))
    
    # Get AI response
    response = model.invoke(conversation)
    print("AI:", response.content)
    
    # Add AI response to history (so memory builds up)
    conversation.append(AIMessage(content=response.content))
