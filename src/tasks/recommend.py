from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from llm import get_google_chat_llm
from db import get_qdrant_retriever

google_chat_llm = get_google_chat_llm(temperature=1.0)
retriever = get_qdrant_retriever(k=10)

recommend_memory = InMemoryChatMessageHistory()

def get_recommend_memory():
    return recommend_memory

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a knowledgeable and friendly BookStore Assistant.
Answer the user's question using the context retrieved.

For each book, include:
- Book ID
- Title
- Price
- Publish Month / Year
- Stock
- Author

Write in **clear, professional, and approachable Markdown**. Use headings, bold text, and bullet points where helpful. Avoid repetitive phrases.

At the end of the list, you may include a single friendly note inviting the user to ask for more details or contact staff if they wish to order.

If a book is not fully described, include what is available and do not add unrelated info.
"""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("system", "Context:\n{context}"),
    ("human", "{input}")
])

document_chain = create_stuff_documents_chain(google_chat_llm, answer_prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_recommend_memory,
    history_messages_key="history",
    output_messages_key="answer",
)

def get_recommendation(request):
    response = conversational_rag_chain.invoke({"input": request})
    result = response["answer"]
    return result