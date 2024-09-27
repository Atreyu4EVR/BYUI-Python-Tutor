import streamlit as st
from streamlit.components.v1 import html
from streamlit_javascript import st_javascript

# Define pages
home_page = st.Page("main.py", title="Home", icon=":material/menu:", default=True)
open_source_chatbot = st.Page("Playground/tutorChat.py", title="Python Tutor", icon=":material/smart_toy:")
#ai_websearch = st.Page("Playground/AtreyuSearch.py", title="AtreyuSearch", icon=":material/travel_explore:")
#learn_overview = st.Page("Learn/overview.py", title="Introduction to AI", icon=":material/domain_verification:")
#learn_demystifying_ai = st.Page("Learn/demystifying_ai.py", title="Demystifying AI", icon=":material/domain_verification:")
#learn_models = st.Page("Learn/Model_Cards.py", title="Model Cards", icon=":material/web:")
#learn_future_of_ai = st.Page("Learn/future_of_ai.py", title="The Future of AI", icon="🔮")
#news_ai = st.Page("under_construction.py", title="Coming Soon", icon=":material/devices_off:")

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# Build the navigation
pg = st.navigation(
    {
        "Home": [home_page],
        "Playground": [open_source_chatbot]
        #"Learn": [learn_overview, learn_demystifying_ai, learn_models],
        #"News": [news_ai],
    }
)

js_code = """<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-BPSQ4HKZ1B"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-BPSQ4HKZ1B');
</script>"""

with open("index.html") as f:
    html_string = f.read()
    html(html_string)
st_javascript(js_code)

# Run the selected page
pg.run()
