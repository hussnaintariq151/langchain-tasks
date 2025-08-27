from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

print("Google API Key:", os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)