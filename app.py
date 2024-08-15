import streamlit as st

# Define pages
home_page = st.Page("main.py", title="Home", icon=":material/home:", default=True)
open_source_chatbot = st.Page("Playground/open_source.py", title="AtreyuChat", icon="🤖")
learn_overview = st.Page("Learn/overview.py", title="Introduction to AI", icon="📘")
learn_demystifying_ai = st.Page("Learn/demystifying_ai.py", title="Demystifying AI", icon="🤔")
#learn_agents = st.Page("Learn/agents.py", title="AI Agents", icon="🕵️‍♂️")
#learn_future_of_ai = st.Page("Learn/future_of_ai.py", title="The Future of AI", icon="🔮")
news_ai = st.Page("under_construction.py", title="Coming Soon", icon="📰")

# Build the navigation
pg = st.navigation(
    {
        "Home": [home_page],
        "Playground": [open_source_chatbot],
        "Learn": [learn_overview, learn_demystifying_ai],
        "News": [news_ai],
    }
)

# Run the selected page
pg.run()