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
repo_id = model_links[selected_model]  # This is your selected model repo ID

# Initialize the Hugging Face client with the selected model
client = InferenceClient(
    model=repo_id,  # Pass the selected model here
    token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

# Function to generate a response from the model using Hugging Face
def generate_response():
    prompt = []
    for dict_message in st.session_state.messages:
        role = dict_message["role"]
        content = dict_message["content"]
        prompt.append(f"{role}\n{content}")

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    for event in client.chat_completion(
        messages=[{"role": "user", "content": prompt_str}],
        max_tokens=MAX_TOKENS,
        stream=True
    ):
        yield str(event.choices[0].delta.content)

# User-provided prompt
if prompt := st.chat_input(disabled=not hf_token):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.write(prompt)

# Generate a new response if the last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    selected_model_repo = model_links[selected_model]
    with st.chat_message("assistant", avatar="./logo.svg"):
        response = generate_response(selected_model_repo)
        full_response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
