import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google-genai"
)

res = model.invoke("Why do parrots talk?")
print(res.content)
