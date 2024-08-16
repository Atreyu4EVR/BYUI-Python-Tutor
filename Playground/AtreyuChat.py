import logging
import os
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

LOGO_URL_LARGE = "images/robot_logo.png"
LOGO_URL_SMALL = "images/robot.png"
HUMAN_AVATAR = "images/human.png"
MAX_TOKENS = 2042

user_avatar = "images/human.png"  # replace with your user avatar
assistant_avatar = "images/robot.png"  # replace with your bot avatar

st.subheader("AtreyuChat")

load_dotenv()

# Initialize the OpenAI client
client = InferenceClient(
    base_url="https://api-inference.huggingface.co/v1",
    token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
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
        'description': "This is the latest model developed by Meta (Llama3.1 8B). View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'images/llama_logo.gif'
    },
    "Mixtral 8x7B": {
        'description': "This model was developed by Mistral.ai (Mixtral 8x7B). View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'https://mistral.ai/images/logo_hubc88c4ece131b91c7cb753f40e9e1cc5_2589_256x0_resize_q97_h2_lanczos_3.webp'
    },
    "Google-Gemma 7B": {
        'description': "This model was developed by Google (Gemma 7B). View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'
    },
    "Zephyr-7b": {
        'description': "The Zephyr model is a **Large Language Model (LLM)**. View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
    }
}

def reset_conversation():
    """Resets Conversation."""
    st.session_state.conversation = []
    st.session_state.messages = []

models = [key for key in model_links.keys()]
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
    avatar = user_avatar if message["role"] == 'user' else assistant_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type here..."):

    # Display user message in chat message container
    with st.chat_message("user", avatar=HUMAN_AVATAR):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response from the model using InferenceClient
    response_content = ""
    try:
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            stream=True,
        ):
            response_content += message.choices[0].delta.content
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            # Update the assistant's message in the UI as it comes in
            st.chat_message("assistant", avatar=LOGO_URL_SMALL).markdown(response_content)
    except Exception as e:
        error_message = "😵‍💫 Looks like someone unplugged something"
        st.markdown(error_message)
        st.markdown(f"Error details: {e}")

    st.session_state.messages.append({"role": "assistant", "content": response_content})
