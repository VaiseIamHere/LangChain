import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model(
    "qwen/qwen3-32b",
    model_provider="groq"
)

@tool
def get_weather(location:str) -> str:
    """Gives weather information of a particular location"""
    return f"It is hot in {location}"

# Either bind or create an agent directly
model_with_tool = model.bind_tools([get_weather])

# res = model_with_tool.invoke("What is the weather like in Delhi tell me more about it")
# print(res)

# for tool_call in res.tool_calls:
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")

# Now to actually create an agent you need to have an tool exection loop

# Collect Query & generate calls
messages = [{'role': 'user', 'content': 'what is the weather like in Delhi?'}]
ai_msg = model_with_tool.invoke(messages)
messages.append(ai_msg)

# Collect tool execution results
for tool_call in ai_msg.tool_calls:
    # here the tool is hardcoded rather we use a tool_map instead!!
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Final response
final_res = model_with_tool.invoke(messages)
print(final_res.text)
print(final_res)
