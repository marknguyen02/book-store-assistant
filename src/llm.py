from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

google_text_embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

def get_google_chat_llm(temperature):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature
)