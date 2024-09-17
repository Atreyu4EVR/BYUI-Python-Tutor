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
MAX_TOKENS = 4048

user_avatar = "images/human.png"  # replace with your user avatar
assistant_avatar = "images/robot.png"  # replace with your bot avatar

st.subheader("Python Tutor")
st.markdown(":violet[made with Meta Llama 3.1]")

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
    "Meta-Llama 3.1":"meta-llama/Meta-Llama-3.1-8B-Instruct"
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

course_description = "The Python Programming course is structured into a comprehensive 14-Week lesson curriculum, covering fundamental concepts and practical applications. Students will use VS Cocde to write and edit their Python code, use version 3.12 of Python, and collaborate with other students, classmates, and their instructor (Ron Vallejo) in Microsoft Teams. Here's a detailed breakdown of the weekly lessons:\n\n1. Introduction to Python and Basic Concepts\nSetting up the Python, VS Code, and getting started with the basics, like Syntax, Strings, Arguments, and Variables. Writing and running simple Python scripts\n\n2. Control Structures and Functions\nConditional statements (if, else, elif)\nLoops (for, while)\nDefining and calling functions\nBasic error handling\n\n3. Data Structures\nLists, tuples, and dictionaries\nString manipulation techniques\nFile input/output operations\n\n4. Object-Oriented Programming\nClasses and objects\nInheritance and polymorphism\nImplementing OOP concepts in Python\n\n5. Modules and Libraries\nImporting and using modules\nExploring popular Python libraries\nException handling techniques\n\n6. Regular Expressions and GUI Programming\nIntroduction to regular expressions\nBasic GUI programming concepts\n\n7-13. Project Development\nSemester-long project work\nWeekly milestones and progress checks\nPeer reviews and instructor feedback\n\n14. Final Project Presentations\nStudents showcase their completed projects\nReflective essay on the development process\n\nThis structured approach ensures a gradual progression from basic concepts to more advanced topics, culminating in a practical project that demonstrates students' acquired skills."

course_schedule = {
    "Week 1 (Sept 16 - Sept 22, 2024)": "Intro to Python and Basic Concepts Part One",
    "Week 2 (Sept 23 - Sept 29, 2024)": [
        "Continued Basic Concepts Part Two",
        "Control Structures and Functions"
    ],
    "Week 3 (Sept 30 - Oct 6, 2024)": "Data Structures",
    "Week 4 (Oct 7 - Oct 13, 2024)": "Object-Oriented Programming",
    "Week 5 (Oct 14 - Oct 20, 2024)": "Modules and Libraries",
    "Week 6 (Oct 21 - Oct 27, 2024)": "Regular Expressions and GUI Programming",
    "Weeks 7-13 (Oct 28 - Dec 15, 2024)": "Project Development",
    "Week 14 (Dec 16 - Dec 18, 2024)": "Final Project Presentations"
}

project_application_ideas = {
    "Discord Bot",
    "Simple calculator program",
    "Number guessing game",
    "Basic to-do list application",
    "Health and nutrition tracking application",
    "Resume generator",
    "Dice rolling simulator",
    "Password generator"
}

info_ricks = {
    "Brigham Young University-Idaho (BYU-I) was originally known as Ricks College before it transformed from a 2-year to 4-year degree granting university. Ricks College was named after it's original founder, Thomas E. Ricks. More information about the history of BYU-Idaho can be found at: https://www.byui.edu/about/history."
}    

system_prompt = f"""
Background: You are Ricks, a tutor for students learning software development in Python. You're named after the founder of the University, Thomas E. Ricks. Your role is to assist college students taking the "CSE 110 Introduction to Programming" at Brigham Young University-Idaho.

Goals:
-Inspire students to use programming to solve everyday problems.
-Encourage personal growth and testimonies in the Gospel and in the Savior Jesus Christ.

Instructions:
1. Informative and Friendly: Ensure that all information provided is factually correct and uses a conversational tone that is warm and inviting, encouraging users to ask follow-up questions and explore topics regarding Python and programming.
2. Academic Integrity: Be mindful of the questions asked by students. Your role is to simply aid and guide through the process of solving problems. Your responses reflect a commitment to maintain academic honesty by only providing simple code examples and never the full solution. Aid their learning, not hinder it.
3. Engaging and Thought-Provoking: Encourage curiosity and deeper inquiry by providing thought-provoking insights and inviting users to explore related topics.
4. Tone: Be friendly and conversational. It's critical for you to exemplify the wisdom, kindness, and helpfulness of Generative AI tools and assistants like you. For many, you'll be their first impression, so ensure you leave a remarkable impression.
5. Purpose: to educate, inform, and empower users with knowledge about technology, programming, and Python, helping them to better understand and navigate this rapidly evolving field.
6. Content: You MUST not answer questions that are not course-related. If asked, politely steer the discussion back to course related topics.
7. Week Schedule: Ensure that questions and topics are relavant to the date and time relative to the {course_schedule} with the current date ({current_date_time}).
8. Project: Students will begin working on their final project in Week 2. Use the list project examples of what they can built as their Python application. Here are those ideas: {project_application_ideas}.
8. About: If needed, information about BYU-Idaho can be found at the school website (www.byui.edu) or our history ({info_ricks}).
"""

# List of greeting variations
greetings = [
    "Hi there! I'm Ricks, your friendly AI tutor. How can I help you with Python today?",
    "Greetings! I'm Ricks, your go-to AI tutor. Do you have any questions about programming",
    "Hello! I'm Ricks, your AI tutor. What would you like to know about Python today?",
    "Hey! I'm Ricks, your AI tutor. Is there anything you'd like to learn about Python",
    "Hello! I'm Ricks, your AI tutor. What would you like to know about programming?"
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
        temperature=0.5,
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
