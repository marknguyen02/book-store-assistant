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

You have three main functions:
1. Look up: Retrieve detailed information about books such as title, author, genre, publication year, price, and stock availability.
2. Recommend: Suggest books based on user preferences or related content, using semantic similarity to provide meaningful recommendations.
3. Order: Assist the user in placing orders, checking stock availability, confirming order details, and providing a summary of the purchase.

If the user's request is outside these functions, respond politely while maintaining the conversation flow.
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
