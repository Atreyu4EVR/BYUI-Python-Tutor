import streamlit as st
from pathlib import Path

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
    menu_items={
        'About': "Python Tutor is an AI chatbot designed specifically for BYU-Idaho students as a free programming tutoring resource. This tool serves as a 24/7 learning companion to help students understand Python concepts, debug code, and develop their programming skills while complementing classroom instruction. \nArtificial intelligence systems can occasionally generate unexpected, inaccurate, or inappropriate responses. Therefore, user discretion is advised. \n\n© 2024 Ron Vallejo. All rights reserved."
    }
)


st.subheader("Python Tutor")
st.markdown('powered by advanced AI **[:blue[made by Meta]](https://www.llama.com/)**')

LOGO = str(Path(__file__).parent / "images" / "python_tutor_logo.png")

with st.sidebar:
    st.image(LOGO, caption=None, use_container_width=True, clamp=False, channels="RGB", output_format="auto")
    st.divider()

with st.sidebar:
    st.markdown("**Rules:** This AI tool is provided as a free educational resource to support your learning. We encourage students to use this tool honestly and avoid using it to generate answers or bypass learning. Using this tool to cheat or create your code is academically dishonest.")
with st.sidebar:
    st.markdown("***Disclaimer:** While this tool is designed to be helpful and safe, artificial intelligence systems can occasionally generate unexpected, inaccurate, or inappropriate responses. Therefore, user discreption is advised.*")


# Define pages
python_tutor = st.Page("tools/chat.py", title="Python Tutor", icon=":material/smart_toy:")


# Build the navigation
pg = st.navigation({"Home": [python_tutor]})

# Run the selected page
pg.run()