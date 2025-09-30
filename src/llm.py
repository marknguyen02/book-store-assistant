from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
# from langchain_ollama import OllamaLLM
from config import GOOGLE_API_KEY

google_text_embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=GOOGLE_API_KEY,
    model="models/text-embedding-004"
)

# ollama_llm = OllamaLLM(
#     model="llama3.2:latest",
#     temperature=0
# )

def get_google_chat_llm(temperature):
    return ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model="gemini-2.5-flash",
        temperature=temperature
)