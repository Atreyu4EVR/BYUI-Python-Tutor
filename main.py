import streamlit as st

st.balloons()
# Set the title of the web page

# Introduction and Purpose
st.header("Introduction")

st.markdown("""
**Python Tutor** is a web application written entirely in Python using the Streamlit framework. This application is dedicated to supporting students enrolled in CSE 110 and showcasing the capabilities of Python. **Ricks**, the AI Assistant, is a custom AI model specifically designed to help students by answering questions about programming, the basics of Python, and other course-related topics.

**Playground**: This site leverages Meta's [LLaMA (Large Language Model Meta AI)](https://www.llama.com/), a state-of-the-art open-source conversational AI chatbot developed by Meta. **LLaMA** is designed to understand and generate human-like text, enabling students to engage in natural, meaningful conversations. Its advanced architecture allows for continuous learning and improvement, ensuring that the AI remains up-to-date with the latest programming concepts and educational content.""")

# Disclaimer
st.markdown("""
***Disclaimer***: *This application uses advanced generative AI technology to provide responses and generate content. While we strive for accuracy, the information generated on this platform may not always be correct or up-to-date. Users are strongly advised to fact-check the information and use this application at their own risk. The developers of this application are not responsible for any consequences resulting from the use of the information provided. By using this application, you acknowledge and agree to these terms.*
""")
