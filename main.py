import streamlit as st

st.balloons()
# Set the title of the web page

# Introduction and Purpose
st.header("Introduction")

st.markdown("""
Atreyu.AI is web application written entirely in Python using the Streamlit framework. This application is dedicated to showcasing the capabilities of Python for students in CSE 110. Atreyu, the AI Assistant, is a custom AI model designed specifically for the students and is tailor-made to answer questions about programming, the basics of Python, and other course related topics. 

**Playground**: At the heart of Atreyu.AI is a free, open-source conversational AI chatbot, powered by an open-source models like Meta's Llama 3. This allows students to engage in natural, human-like conversations with an AI that continuously evolves and improves.
""")

# Disclaimer
st.markdown("""
***Disclaimer***: *This application uses advanced generative AI technology to provide responses and generate content. While we strive for accuracy, the information generated on this platform may not always be correct or up-to-date. Users are strongly advised to fact-check the information and use this application at their own risk. The developers of this application are not responsible for any consequences resulting from the use of the information provided. By using this application, you acknowledge and agree to these terms.*
""")
