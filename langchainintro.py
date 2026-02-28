import os
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Tools
def get_weather(city:str) -> str:
    """Get weather for a city"""
    return f"The weather today in {city} is sunny."

# Agent
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful agent"
)

# Invokation
res = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What is the weather like in Mumbai?"
    }]
})

print(res)
