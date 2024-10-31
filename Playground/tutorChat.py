import logging
import os
import random
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

LOGO_URL_LARGE = "images/robot_logo.png"
LOGO_URL_SMALL = "images/robot.png"
HUMAN_AVATAR = "images/human.png"
MAX_TOKENS = 2042

user_avatar = "images/human.png"  # replace with your user avatar
assistant_avatar = "images/robot.png"  # replace with your bot avatar

st.subheader("AtreyuChat")

load_dotenv()

# Define the Mountain Daylight Time offset manually (UTC-6)
mountain_offset = timedelta(hours=-6)
mountain_tz = timezone(mountain_offset, name="MDT")
utc_now = datetime.now(timezone.utc)
mountain_time = utc_now.astimezone(mountain_tz)
formatted_mountain_time = mountain_time.strftime("%m-%d-%Y %H:%M:%S MDT")
current_date_time = ("Current Date/Time:", formatted_mountain_time)

# Supported models
model_links = {
    "meta-llama/Llama-3.2-3B-Instruct",
}

# Model descriptions and logos
model_info = {
    "Meta-Llama 3.1": {
        'description': "This is the latest model developed by Meta (Llama3.1 8B). View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'images/llama_logo.gif'
    }
}

models = [key for key in model_links.keys()]
selected_model = st.sidebar.selectbox("Select Model", models)
repo_id = model_links[selected_model]  # This is your selected model repo ID

system_prompt = f"""
Background: Your name is Atreyu, named after the fearless warrior from the book, Never Ending Story. You’re a helpful chatbot for a young boy named Mason. You're going to assist him with fun ideas, answer his questions, and support him in his daily tasks. You're always there to make his experience enjoyable and positive. Personalized Assistance - chat about his favorite video games like Minecraft, Fortnite, and Rocket League; help with projects or games on his dad's Raspberry Pi 5; talk about his chickens and his dad's duck, Daisy, asking questions about their care and sharing fun facts.
"""

# List of greeting variations
greetings = [
    "Hi there! I'm Atreyu, your friendly AI assistant. How can I help you with Python today?",
    "Greetings! I'm Atreyu, your go-to AI assistant. Do you have any questions about programming",
    "Hello! I'm Atreyu, your AI guide. What would you like to know about Python today?",
    "Hey! I'm Atreyu, your AI assistant. Is there anything you'd like to learn about Python",
    "Hello! I'm Atreyu, your AI assistant. What would you like to know about programming?"
]

# Function to select a random greeting
def get_random_greeting():
    return random.choice(greetings)

def reset_conversation():
    """Resets Conversation."""
    st.session_state.conversation = []
    st.session_state.messages = []

# Initialize the Hugging Face client with the selected model
client = InferenceClient(
    model=repo_id,  # Pass the selected model here
    token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    random_greeting = get_random_greeting()  # Get a random greeting
    st.session_state.messages = [{"role": "assistant", "content": random_greeting}]
    
# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] == 'user':
        avatar = user_avatar
    else:  # assuming the only other role is 'bot'
        avatar = assistant_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def clear_chat_history():
    random_greeting = get_random_greeting()  # Get a random greeting
    st.session_state.messages = [{"role": "assistant", "content": random_greeting}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

# Function to generate a response from the model using Hugging Face

def generate_response():
    prompt = []

    # Add the system message at the beginning of the prompt
    prompt.append(system_prompt)
    
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append(f"user\n{dict_message['content']}")
        else:  # This assumes the role is 'assistant' otherwise
            prompt.append(f"assistant\n{dict_message['content']}")

    # Add the final assistant role prompt where the response will be generated
    prompt.append("assistant")
    
    # Combine all the parts into a single string with line breaks
    prompt_str = "\n".join(prompt)
    
    for event in client.chat_completion(
        messages=[{"role": "user", "content": prompt_str}],
        max_tokens=MAX_TOKENS,
        stream=True
    ):
        yield str(event.choices[0].delta.content)

# User-provided prompt
if prompt := st.chat_input(f"Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.write(prompt)

# Generate a new response if the last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    model=model_links[selected_model]
    with st.chat_message("assistant", avatar=assistant_avatar):
        response = generate_response()
        full_response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
