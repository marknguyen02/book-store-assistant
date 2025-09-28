from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from llm import get_google_chat_llm

google_chat_llm = get_google_chat_llm(temperature=0.7)

def convert_history(messages):
    converted = []
    for m in messages:
        if m["role"] == "user":
            converted.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            converted.append(AIMessage(content=m["content"]))

    return converted

fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a friendly and helpful BookStore Assistant.
Engage in natural conversation with the user, providing clear and concise responses.
Do not include explanations outside the conversation.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{request}")
])

llm_with_prompt = fallback_prompt | google_chat_llm

def handle_fallback(request, history):
    converted_history = convert_history(history)
    response = llm_with_prompt.invoke({
        "request": request,
        "history": converted_history
    })
    return response.content
