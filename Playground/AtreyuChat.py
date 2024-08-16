import logging
import os
import sys
import torch
import base64
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient

LOGO_URL_LARGE = "images/robot_logo.png"
LOGO_URL_SMALL = "images/robot.png"
HUMAN_AVATAR = "images/human.png"
MAX_TOKENS = 2042

user_avatar = "images/human.png"  # replace with your user avatar
assistant_avatar = "images/robot.png"  # replace with your bot avatar

st.subheader("AtreyuChat")

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
client = InferenceClient(
    base_url="https://api-inference.huggingface.co/v1",
    api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
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

# Random dog images for error messages
random_dog = ["broken_llama3.jpeg"]

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

examples = [
    {"input": "What is your name?", "output": "Thank you for asking. My name is Atreyu. What's your name?"},
    {"input": "What are the benefits of AI?", "output": "Great question! There's many benefits of AI. such as enhancing productivity, automating repetitive tasks, enabling data-driven decision-making, and making complex processes more efficient and accessible."},
    {"input": "Should I be afraid of AI?", "output": "No, but it’s important to approach AI with a balanced understanding, recognizing both its benefits and potential risks, and advocating for responsible and ethical use."},
    {"input": "Will AI take away jobs from humans?", "output": "It’s natural to worry about AI’s impact on jobs, but there’s a broader perspective to consider. While AI and automation might displace some roles—about 85 million by 2025, according to the World Economic Forum—they are also expected to create 97 million new jobs that align better with the evolving relationship between humans and machines. This shift emphasizes the importance of adapting and learning new skills, opening doors to more meaningful, creative, and innovative work. AI isn’t just taking away jobs—it’s transforming them, offering new opportunities for growth and advancement in the workforce."},
    {"input": "What is generative AI?", "output": "Generative AI is a subset of artificial intelligence that creates new content, such as text, images, music, or videos, rather than just analyzing data or making predictions."},
    {"input": "How does generative AI differ from traditional AI?", "output": "Traditional AI focuses on analyzing data and making predictions, while generative AI is trained to generate new data that resembles the original training data."},
    {"input": "What are some examples of generative AI applications?", "output": "Examples include text generation, image and video generation, music composition, and assisting designers and artists in creating new and innovative designs."},
    {"input": "When did the concept of machine-generated content first emerge?", "output": "The concept of machine-generated content emerged in the 1950s and 1960s alongside early artificial intelligence research."},
    {"input": "Who created one of the earliest examples of generative AI, and what was it?", "output": "Joseph Weizenbaum created ELIZA in the 1960s, a pioneering chatbot that simulated a psychotherapist and could engage in natural language conversations."},
    {"input": "What technological breakthrough in 2014 significantly advanced generative AI?", "output": "The introduction of Generative Adversarial Networks (GANs) by Ian Goodfellow and his colleagues in 2014 enabled the creation of highly realistic synthetic data, revolutionizing generative AI."},
    {"input": "How has generative AI evolved in recent years?", "output": "Recent advancements include DeepMind's WaveNet in 2016, NVIDIA's Progressive GANs in 2017, OpenAI's GPT-2 and GPT-3 in 2019-2020, and the release of DALL-E and ChatGPT by OpenAI in 2022."},
    {"input": "What are some potential future applications of generative AI in entertainment?", "output": "In entertainment, generative AI could be used to create personalized content such as movies, music, and video games."},
    {"input": "How might generative AI impact the education industry?", "output": "Generative AI could transform education by creating customized learning materials like textbooks, videos, and interactive simulations tailored to individual learning needs."},
    {"input": "What are some concerns associated with generative AI?", "output": "Concerns include issues of authenticity, copyright, and the impact on the value of human creativity."},
    {"input": "How did OpenAI contribute to the advancement of generative AI in 2022?", "output": "OpenAI released DALL-E, which generates images from text prompts, and ChatGPT, a conversational AI, marking significant milestones in generative AI."},
    {"input": "What role does generative AI play in healthcare?", "output": "In healthcare, generative AI could be used to create personalized treatment plans, such as customized medications and therapies."},
    {"input": "Why is responsible use of generative AI important?", "output": "Responsible use of generative AI is crucial to address concerns like authenticity, copyright, and ethical implications, ensuring the technology benefits society without negative consequences."},
    {"input": "How has generative AI impacted design and creativity?", "output": "Generative AI assists designers and artists by enabling them to create new and innovative designs, such as graphics, logos, and products, expanding the possibilities of creative expression."},
    {"input": "What is the significance of GPT-4 in the context of generative AI?", "output": "Introduced by OpenAI in 2023, GPT-4 further advanced the capabilities of large language models, enabling even more sophisticated and human-like text generation."}
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named Atreyu developed to answer questions about AI technology."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

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

    # Use Langchain with HuggingFaceHub 
    llm = HuggingFaceHub(repo_id=repo_id, 
                         model_kwargs={"temperature": temp_values, "max_new_tokens": MAX_TOKENS})

    chain = final_prompt | llm 
    response = chain.invoke({"input": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=LOGO_URL_SMALL):
        try:
            st.markdown(response)  # Change from st.write to st.markdown for consistency
        except Exception as e:
            error_message = "😵‍💫 Looks like someone unplugged something"
            st.markdown(error_message)
            st.markdown(f"Error details: {e}")

    st.session_state.messages.append({"role": "assistant", "content": response})
