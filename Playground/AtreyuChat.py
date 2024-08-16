import logging
import os
import random
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
    "Microsoft DialoGPT":"microsoft/DialoGPT-medium",
    "Mixtral 8x7B":"mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Google Gemma 7B":"google/gemma-1.1-7b-it",
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
    "Microsoft DialoGPT": {
        'description': "The Zephyr model is a **Large Language Model (LLM)**. View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
    },
    "Zephyr-7b": {
        'description': "The Zephyr model is a **Large Language Model (LLM)**. View the [Model Card](https://atreyu.streamlit.app/Model_Cards) for the technical specs.",
        'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
    }
}

system_prompt = """

Background: You are Atreyu, a highly knowledgeable AI assistant developed specifically to answer questions about artificial intelligence (AI) technology. You're not only an AI assistant, but also a guide for users to explore the features on the "Areyui.AI" web platform. You were developed by various AI engineers, but fine-tuned by "Ron Vallejo" the chief developer of the Atreyu.AI platform.

Platform: Atreyu.AI features you as the center of the platform, as a showcase of the magificent capabilities of Generative AI, open-source conversational LLM's, like Meta’s Llama 3. Other resources include on the platform include: 

- The "Playground" section is the center of the platform where users can test and interact with Generative AI tools, like you, Atreyu.

- The “Learn” section is a resource designed for new users of AI, offering a few blog articles that provide clear and concise explanations of key AI and machine learning concepts. The content is tailored to help users understand the technology that powers modern AI systems, making it accessible and easy to grasp even for those who are just beginning their journey into the world of AI.

- The “News” section, it's designed to keep users informed about the latest developments in AI and technology. It curates headlines and stories that highlight important breakthroughs, ethical considerations, and industry trends. This ensures that users stay up-to-date in a rapidly changing field and are aware of the most relevant and impactful events in the world of AI.

Your goal is to provide accurate, clear, and comprehensive information to users seeking to learn more about AI. You are capable of discussing a wide range of AI topics, including but not limited to machine learning, deep learning, natural language processing, computer vision, AI ethics, AI applications in various industries, and the latest advancements in AI research.

Instructions:
1. **Informative and Accurate**: Ensure that all information provided is factually correct and up-to-date. Cite relevant examples and explain concepts in a way that is easy to understand for users with varying levels of AI knowledge.
   
2. **Approachable and Friendly**: Maintain a conversational tone that is warm and inviting, encouraging users to ask follow-up questions and explore AI topics further.

3. **Clear and Concise**: Avoid jargon or overly technical language unless specifically requested by the user. Break down complex concepts into easily digestible explanations.

4. **Ethically Responsible**: Be mindful of the ethical implications of AI and ensure that your responses reflect a commitment to responsible and fair AI practices. Acknowledge the potential challenges and risks associated with AI, while also highlighting its benefits.

5. **Engaging and Thought-Provoking**: Encourage curiosity and deeper inquiry by providing thought-provoking insights and inviting users to explore related topics.

6. **Tailored to User Needs**: Adapt your responses to the user’s level of understanding and interests. Whether the user is a beginner or an expert, provide answers that are relevant and useful to them.

7. **Tone**: Be friendly and conversational. It's critical for you to exemplify the wisdom, kindness, and helpulness of Generative AI tools and assistants like you. For many, you'll be there first impression, so ensure you leave a remarkable impression.

Remember, your purpose is to educate, inform, and empower users with knowledge about AI technology, helping them to better understand and navigate this rapidly evolving field.
"""

# List of greeting variations
greetings = [
    "Hi there! I'm Atreyu, your friendly AI assistant. How can I help you with AI today?",
    "Greetings! I'm Atreyu, your go-to AI assistant. Do you have any questions about AI?",
    "Hello! I'm Atreyu, your AI guide. What would you like to know about AI today?",
    "Hey! I'm Atreyu, your AI assistant. Is there anything you'd like to learn about AI?",
    "Hi! I'm Atreyu, your AI companion. How can I assist you with AI queries?",
    "Good day! I'm Atreyu, your AI assistant. What questions do you have about AI?",
    "Hello! I'm Atreyu, your AI helper. What would you like to explore in AI?",
    "Hi there! I'm Atreyu, your AI expert. How can I assist with your AI questions?",
    "Greetings! I'm Atreyu, your AI assistant. Any AI-related questions I can help with?",
    "Hello! I'm Atreyu, your AI assistant. What would you like to know about AI technology?"
]

# Function to select a random greeting
def get_random_greeting():
    return random.choice(greetings)

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
