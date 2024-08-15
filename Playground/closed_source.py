import os
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import sys
import torch
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# Show title and description.
st.title("Closed-Source Chatbot")
st.write(
    "This is a simple chatbot that uses Anthropic's Claude 3.5 closed source model to generate responses. "
    
)

#api_key = os.environ.get('OPENAI_API_KEY') #for OpenAI
#client = OpenAI(api_key=api_key)

api_key = os.environ.get('ANTHROPIC_API_KEY') #for Anthropic
client = Anthropic(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input(f"Type here..."):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API.
    with st.chat_message("assistant"):
        try:
            stream = client.messages.create(
                max_tokens=2024,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                model="claude-3-5-sonnet",
                stream=True,
            )
    
            response = st.write_stream(stream)

        except Exception as e:
            response = "😵‍💫 Looks like someone unplugged something"
            st.write(response)
            st.write("This was the error message:")
            st.write(e)

    st.session_state.messages.append({"role": "assistant", "content": response})
