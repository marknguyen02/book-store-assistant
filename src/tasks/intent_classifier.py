from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from llm import get_google_chat_llm

google_chat_llm = get_google_chat_llm(temperature=0)

classifier_memory = InMemoryChatMessageHistory()

def get_classifier_memory():
    return classifier_memory

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a classifier for a BookStore Assistant. 
Classify the user request into exactly one of three tasks, taking into account 
the **conversation history**:

- lookup (structured lookup of price, stock, author, category)
- recommend (book suggestions, descriptions, or similar books)
- order (placing an order with book title, quantity, address, phone)

Respond with a **single word only**, exactly one of: lookup, recommend, order.
Do NOT include any extra words, punctuation, quotes, or explanation.
If uncertain, respond with: none.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{request}")
])

llm_with_prompt = classifier_prompt | google_chat_llm

classifier_chain = RunnableWithMessageHistory(
    llm_with_prompt,
    get_classifier_memory,
    input_messages_key="request",
    history_messages_key="history",
)

def classify_task(request):
    response = classifier_chain.invoke({"request": request})
    result = response.content.strip().lower()
    return result
