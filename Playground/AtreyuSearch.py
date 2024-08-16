import os
import streamlit as st
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Tavily client with API key
api_key = os.getenv('TAVILY_API_KEY')

# Check if API key is correctly loaded
if not api_key:
    st.error("API key not found. Please check your environment variables.")
    st.stop()

client = TavilyClient(api_key=api_key)

# Show title and description.
st.title("AI WebSearch")
st.markdown("Powered by *[Tavily](https://tavily.com/)*")

def generate_response(input_text, method='search', **kwargs):
    try:
        # Choose the appropriate method based on the user's choice
        if method == 'search':
            response = client.search(input_text, **kwargs)
        elif method == 'get_search_context':
            response = client.get_search_context(input_text, **kwargs)
        elif method == 'qna_search':
            response = client.qna_search(input_text, **kwargs)
        else:
            st.error(f"Unknown method: {method}")
            return

        # Return the response so it can be handled appropriately
        return response

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

# Form for Tavily Web Search
with st.form("web_search"):
    input_text = st.text_input("Enter search query:")
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Display user's query as a chat message
        with st.chat_message("user"):
            st.write(input_text)

        # Generate and display the assistant's response
        response = generate_response(input_text, method='get_search_context', search_depth='advanced', max_tokens=4000)

        # If response is a list or generator of chunks, accumulate them
        full_content = ""
        if isinstance(response, (list, tuple)) or hasattr(response, '__iter__'):
            for chunk in response:
                content = chunk.get('content', '') if isinstance(chunk, dict) else str(chunk)
                full_content += content
                print(content, end="", flush=True)  # Print to console if needed
        else:
            full_content = response  # Direct response if it's not a generator or list
        
        if full_content:
            with st.chat_message("assistant"):
                # Format the response using Markdown
                st.markdown(full_content)

