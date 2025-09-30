from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from llm import get_google_chat_llm
from sers import check_order_validity, insert_order_to_db 

REQUIRE_FIELDS = ["customer_name", "phone", "address", "book_id", "quantity", "confirmed"]

EXTRACT_CHAT_PROMPT = """
You are a data extraction system that updates the order information based on both the user request and the system messages.

Rules (priority ordered):
1. Never infer, guess, or invent information. Only update fields with values explicitly provided in the conversation (user messages + system messages).
2. If the user attempts to order multiple different book IDs in one request, do not update any order data at all. Leave the order state unchanged.
3. Always persist previously collected valid information from the conversation history. 
   - Do not overwrite existing valid values with None unless:
     a) The system explicitly indicates an error for that field, OR
     b) The user explicitly provides new/updated information for that field.
4. After an order is successfully submitted:
   - Reset the order state for a new order.
   - Retain customer information (customer_name, phone, address) by default for the next order, unless the user explicitly provides updated details.
   - Reset book_id, quantity, and confirmed to None or False for the new order.
5. Always update the order data strictly according to the conversation history.
6. If any required information is missing, set that field to None.
7. If a system message indicates an error or invalid value for a field:
   - Reset that field to None.
   - Force confirmed = False (even if the user previously confirmed).
8. If the user provides all required fields and explicitly confirms in the same message (e.g., "I want book ID 123, quantity 2, confirm"), then set confirmed = True immediately.
9. If the user confirms the order (e.g., says "yes", "confirm", "okay") but there are still errors or missing fields, keep confirmed = False.
10. If all required information is complete but the user has not explicitly confirmed, keep confirmed = False until explicit confirmation is received.
11. Only set confirmed = True when the user explicitly confirms the order AND no errors or missing fields remain.
12. The output must strictly follow the `Order` schema.
"""

RESPONSE_CHAT_PROMPT = """
You are an assistant responsible for communicating with the customer about their book order.

Guidelines (priority ordered):
1. **At the very beginning of the conversation**, always greet the customer and clearly state the rules and required information for placing an order.

    Required Information:
    - Full name
    - Phone number
    - Shipping address
    - Book ID
    - Quantity
    - Confirmation

    Rules:
    - Book ID is mandatory (we cannot process orders using only book titles).
    - Only one book per order. If the customer wants multiple books, they must place separate orders.

   Note: These rules and requirements are **only stated at the start of the very first order**.  
   When a new order begins after a completed one, continue naturally without repeating the rules.

2. **Never assume or invent information.** Only use details explicitly provided in the conversation (user messages + system messages).
3. If required information is missing or invalid, clearly tell the customer what is missing and ask them to provide it.
4. If all required fields are provided but the order is not yet confirmed, summarize the order neatly and ask the customer to confirm.
5. If the user attempts to confirm while information is still missing or invalid, ignore the confirmation and explain what still needs to be fixed.
6. If a system message indicates a specific error (e.g., invalid book ID, insufficient stock), clearly explain the issue and guide the customer to correct it.
7. If a system message indicates a technical/system error, politely inform the customer with a short and clear message.
8. If the order has been successfully submitted, notify the customer in a clear, friendly, and professional way that their order has been placed successfully.
9. Maintain a natural, friendly, and professional tone. Avoid being overly formal or robotic. Use clear formatting when presenting order details or listing missing info.
"""

class Order(BaseModel):
    customer_name: Optional[str] = Field(None, description="Full name of the customer who places the order")
    phone: Optional[str] = Field(None, description="Customer's contact phone number for order confirmation and delivery")
    address: Optional[str] = Field(None, description="Shipping address where the ordered books should be delivered")
    book_id: Optional[int] = Field(None, description="Unique identifier of the book being ordered")
    quantity: Optional[int] = Field(None, description="Number of copies of the book requested in the order")
    confirmed: bool = Field(False, description="Whether the order has been confirmed")


class OrderManager:
    def __init__(self):
        self.extract_chain = self._build_extract_chain()
        self.response_chain = self._build_response_chain()
        self.chat_memory = []

        self.data = self._initialize_data()


    def _build_extract_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACT_CHAT_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ])
        extract_llm = get_google_chat_llm(temperature=0).with_structured_output(schema=Order)
        chain = prompt | extract_llm
        return chain
    
    def _build_response_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", RESPONSE_CHAT_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        response_llm = get_google_chat_llm(temperature=1)
        chain = prompt | response_llm
        return chain
    
    def _initialize_data(self):
        return Order().model_dump()
    
    def _update_data(self):
        ext_result = self.extract_chain.invoke({"messages": self.chat_memory})
        self.data = ext_result.model_dump()
    
    def _monitor_system(self):
        sys_messages = []
        missing_fields = [key for key in self.data if self.data[key] is None and key != "confirmed"]

        if len(missing_fields) != 0:
            fields_str = ", ".join(missing_fields)
            sys_messages.append(SystemMessage(
                content=f"Missing required fields: {fields_str}."
            ))
        else:
            if self.data["confirmed"]:
                try:
                    check_order_validity(self.data)   
                    insert_order_to_db(self.data)
                    self._update_data()
                    sys_messages.append(SystemMessage(
                        content="Please notify the user that their order was successfully submitted."
                    ))
                except Exception as e:
                    sys_messages.append(SystemMessage(
                        content=f"System error: {str(e)}"
                    ))
            else:
                sys_messages.append(SystemMessage(
                    content="Please review and verify the order information for accuracy before confirming the order."                
                ))

        return sys_messages
        
    def process_order(self, request):
        self.chat_memory.append(HumanMessage(content=request))

        self._update_data()
        sys_messages = self._monitor_system()
        self.chat_memory.extend(sys_messages)
        self._update_data()

        response = self.response_chain.invoke({"messages": self.chat_memory})
        ai_content = response.content

        self.chat_memory.append(AIMessage(content=ai_content))

        return ai_content    
    
order_manager = OrderManager()

def handle_order(request):
    response = order_manager.process_order(request)
    return response