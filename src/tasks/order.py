import json
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from llm import get_google_chat_llm
from sers import insert_order_to_db

REQUIRED_FIELDS = ["customer_name", "phone", "address", "book_id", "quantity"]
data_dict = {
    "confirm": False,
    "data": {field: None for field in REQUIRED_FIELDS},
    "content": ""
}

google_chat_llm = get_google_chat_llm(temperature=0.6)
order_memory = InMemoryChatMessageHistory()

def get_order_memory():
    return order_memory

order_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Bookstore Assistant.
Your task is to extract, maintain, and confirm order information.

Important:
- You must output a single valid JSON object only. 
- Do not use Markdown formatting, code blocks, or extra explanations outside of the JSON.
- The JSON must be directly parseable with Python's json.loads().

Output format:
{{
  "confirm": boolean,
  "data": {{
    "customer_name": string or null,
    "phone": string or null,
    "address": string or null,
    "book_id": integer or null,
    "quantity": integer or null
  }},
  "content": string
}}

Rules:
1. Always output a single valid JSON object with exactly these 3 keys: confirm, data, content.
2. Field "data" must contain exactly these keys: """ + str(REQUIRED_FIELDS) + """.
3. If a field is not explicitly provided, keep its value null or unchanged if it was previously provided.
4. "confirm" = true ONLY IF:
   - The user explicitly confirms the order AND
   - All fields in "data" are non-null.
   Otherwise, "confirm" must be false.
5. Do not guess or fabricate values. Only use what the user provides.
6. The JSON must strictly comply with RFC 8259.
7. "content" must be a professional, user-facing message that:
   - Summarizes any changes made to the order.
   - Describes the current state of the order in a clear and concise manner.
   - Explicitly lists any missing fields.
   - Politely informs the user that confirmation cannot proceed until all fields are complete.
   - You may use Markdown formatting (e.g., **bold**, bullet points, newlines) inside the "content" string to make it easier for the user to read.
8. Do not add explanations outside the JSON. The response must only be the JSON object.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{request}")
])

llm_with_prompt = order_prompt | google_chat_llm

order_chain = RunnableWithMessageHistory(
    llm_with_prompt,
    get_order_memory,
    input_messages_key="request",
    history_messages_key="history"
)

def parse_order_message(request: str) -> dict:
    global data_dict
    
    response = order_chain.invoke({"request": request})
    result = response.content

    try:
        parsed = json.loads(result)

        if "confirm" in parsed and isinstance(parsed["confirm"], bool):
            data_dict["confirm"] = parsed["confirm"]

        if "data" in parsed and isinstance(parsed["data"], dict):
            for key in REQUIRED_FIELDS:
                if key in parsed["data"]:
                    data_dict["data"][key] = parsed["data"][key]

        if "content" in parsed and isinstance(parsed["content"], str):
            data_dict["content"] = parsed["content"]

    except:
        pass

def is_order_confirmable():
    global data_dict

    if not data_dict.get("confirm", False):
        return False

    data = data_dict.get("data", {})
    for field in REQUIRED_FIELDS:
        if data.get(field) is None:
            return False

    return True

def reset_order_memory_with_context():
    global order_memory, order_chain

    data_dict["confirm"] = False

    order_memory = InMemoryChatMessageHistory()

    init_prompt = (
        "I currently have the following order information stored:\n"
        + json.dumps(data_dict, ensure_ascii=False, indent=4)
        + "\nPlease continue to provide the missing details or confirm the order."
    )

    order_memory.add_user_message(init_prompt)

    order_chain = RunnableWithMessageHistory(
        llm_with_prompt,
        get_order_memory,
        input_messages_key="request",
        history_messages_key="history"
    )

def complete_confirm():
    global order_memory, data_dict

    order_memory = InMemoryChatMessageHistory()
    data_dict = {
        "confirm": False,
        "data": {field: None for field in REQUIRED_FIELDS},
        "content": ""
    }

def handle_order(request):
    parse_order_message(request)

    if data_dict["confirm"]:
        if is_order_confirmable():
            insert_order_to_db(data_dict["data"])
            complete_confirm()
        else:
            reset_order_memory_with_context()
    
    return data_dict["content"]
