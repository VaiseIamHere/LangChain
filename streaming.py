import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google-genai"
)

# Output is sent when completely completed
print("*******Normal Invoke*******")
res = model.invoke("Write a 200 words essay on Artificial Intelligence")
print(res.content)

# Streaming: Output is sent as soon it gets generated
print("*******Streaming*******")
cnt = 0
for chunk in model.stream("Write a 1000 words essay on Attack of titans"):
    print(chunk.text)
    cnt += 1

print(f"Total chunks: {cnt}")

# Batch: Parallel processing of multiple independent questions
# config parameter is opt
print("*******Batch Processing*******")
respones = model.batch(
    [
        "What is color of sky?",
        "Who were the first men?",
        "Tell me about yourself"
    ],
    config={
        'max_concurrency': 2
    }
)

for res in respones:
    print(res.text)
