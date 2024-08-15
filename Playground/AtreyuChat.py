import logging
import os
import sys
import torch
import base64
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values

LOGO_URL_LARGE = "images/robot_logo.png"
LOGO_URL_SMALL = "images/robot.png"
HUMAN_AVATAR = "images/human.png"
MAX_TOKENS = 2042

user_avatar = "images/human.png" # replace with your user avatar
assistant_avatar = "images/robot.png"  # replace with your bot avatar

st.set_page_config(
    page_title="Atreyu.AI",
    page_icon="images/robot.png",
    layout="wide",
    initial_sidebar_state="auto"
)

load_dotenv()

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1",
    api_key=os.environ.get('HUGGINGFACE_API_KEY')
)

# Supported models
model_links = {
    "Meta-Llama 3.1":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mixtral 8x7B":"mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Google-Gemma 7B":"google/gemma-1.1-7b-it",
    "Zephyr-7b":"HuggingFaceH4/zephyr-7b-beta"
}

# Model descriptions and logos
model_info = {
    "Meta-Llama 3.1": {
        'description': "This is the latest model developed by Meta (Llama3.1 8B). View the [Model Card](www.something.com) for the technical specs.",
        'logo': 'images/llama_logo.gif'
    },
    "Mixtral 8x7B": {
        'description': "This model was developed by Mistral.ai (Mixtral 8x7B). View the [Model Card](www.something.com) for the technical specs.",
        'logo': 'https://mistral.ai/images/logo_hubc88c4ece131b91c7cb753f40e9e1cc5_2589_256x0_resize_q97_h2_lanczos_3.webp'
    },
    "Google-Gemma 7B": {
        'description': "This model was developed by Google (Gemma 7B). View the [Model Card](www.something.com) for the technical specs.",
        'logo': 'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'
    },
    "Zephyr-7b": {
        'description': "The Zephyr model is a **Large Language Model (LLM)**. View the [Model Card](www.something.com) for the technical specs.",
        'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
    }
}

# Random dog images for error messages
random_dog = ["broken_llama3.jpeg"]

def reset_conversation():
    '''Resets Conversation'''
    st.session_state.conversation = []
    st.session_state.messages = []

st.header(f"`AtreyuChat`")

models =[key for key in model_links.keys()]
selected_model = st.sidebar.selectbox("Select Model", models)
repo_id = model_links[selected_model]

# Create a temperature slider
temp_values = st.sidebar.slider('Adjust creativity level from model responses', 0.0, 1.0, 0.5)

# Create model description
st.sidebar.markdown(model_info[selected_model]['description'])

# Add reset button to clear conversation
st.sidebar.button('Reset Chat', on_click=reset_conversation)

# Handle model change
if "prev_option" not in st.session_state:
    st.session_state.prev_option = selected_model

if st.session_state.prev_option != selected_model:
    st.session_state.messages = []
    st.session_state.prev_option = selected_model
    reset_conversation()

# Set a default model
if selected_model not in st.session_state:
    st.session_state[selected_model] = model_links[selected_model]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == 'user':
        avatar = user_avatar
    else:  # assuming the only other role is 'bot'
        avatar = assistant_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(f"Type here..."):

    # Display user message in chat message container
    with st.chat_message("user", avatar=HUMAN_AVATAR):
        st.markdown(prompt)

    # Add user message to chat history
    # Bot greets the user by their name
    if "user_name" in st.session_state:
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.markdown(f"Welcome, {st.session_state.user_name}!")
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=LOGO_URL_SMALL):
        try:
            stream = client.chat.completions.create(
                model=model_links[selected_model],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temp_values,
                stream=True,
                max_tokens=MAX_TOKENS,
            )
    
            response = st.write_stream(stream)

        except Exception as e:
            response = "😵‍💫 Looks like someone unplugged something"
            st.write(response)
            st.write("This was the error message:")
            st.write(e)

    st.session_state.messages.append({"role": "assistant", "content": response})
