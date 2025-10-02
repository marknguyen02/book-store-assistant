from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
from router import handle_user_request

st.title("Book Store Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Request anything"):
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = handle_user_request(prompt, history=st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response
        })

    except Exception as e:
        error_message = "Something went wrong while processing your request. Please try again."
        with st.chat_message("assistant"):
            st.markdown(error_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })
