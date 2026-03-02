import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model(
    "qwen/qwen3-32b",
    model_provider="groq"
)

# This classes basically provide distinction between different 
# instructions with in the same conversation
messages = [
    SystemMessage("You are an poetry expert"),
    HumanMessage("Write a poem on artificial intelligence")
]

res = model.invoke(messages)
print(res.content)
