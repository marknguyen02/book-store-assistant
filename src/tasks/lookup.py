from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from llm import get_google_chat_llm
from db import psql_db

lookup_memory = InMemoryChatMessageHistory()

def get_lookup_memory():
    return lookup_memory

google_chat_llm = get_google_chat_llm(temperature=0.2)

lookup_executor = create_sql_agent(
    llm=google_chat_llm,
    db=psql_db,
    agent_type='zero-shot-react-description',
    verbose=True,
    handle_parsing_errors=True 
)

lookup_chain = RunnableWithMessageHistory(
    lookup_executor,
    get_lookup_memory,
    input_messages_key="input",
    history_messages_key="history",
)

def lookup_book_data(request: str) -> str:
    response = lookup_chain.invoke({"input": request})
    return response['output']
