import os
import json
import time
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
        # Initialize progress bar
        progress_text = "Fetching AI search results."
        my_bar = st.progress(0, text=progress_text)

        my_bar.progress(20)
        # Choose the appropriate method based on the user's choice
        if method == 'search':
            response = client.search(input_text, **kwargs)
        elif method == 'get_search_context':
            response = client.get_search_context(input_text, **kwargs)
        elif method == 'qna_search':
            response = client.qna_search(input_text, **kwargs)
        else:
            st.error(f"Unknown method: {method}")
            return None

        my_bar.progress(60)

        return response

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

def parse_and_format_response(response):
    """Parse the JSON response and format it for display using Markdown."""
    try:
        # If the response is already a dict, no need to parse it
        if isinstance(response, dict):
            response_data = response
        else:
            # If the response is a JSON string, load it as a dict
            response_data = json.loads(response)

        my_bar.progress(80, text="Parsing data.")

        formatted_response = ""

        # Extract the query and response time
        query = response_data.get("query", "No query found")
        response_time = response_data.get("response_time", "N/A")

        formatted_response += f"**Query:** {query}\n"
        formatted_response += f"**Response Time:** {response_time} seconds\n\n"

        # Extract and format the search results
        results = response_data.get("results", [])
        for result in results:
            title = result.get("title", "No title")
            url = result.get("url", "#")
            content = result.get("content", "No content available")

            formatted_response += f"**[{title}]({url})**\n\n{content}\n\n"

        my_bar.progress(100, text="Complete.")

        return formatted_response
    except Exception as e:
        return f"Error parsing response: {str(e)}"

# Search Input in Form for Tavily AI Search
st.header("Search")
with st.form("web_search"):
    input_text = st.text_input("Enter search query:")
    submitted = st.form_submit_button("Submit")

# Display Results
st.header("Results")
container = st.container(border=True)

if submitted and input_text:
    # Generate and display the assistant's response
    response = generate_response(input_text, method='search', search_depth='advanced', max_results=5, include_answer=False, include_images=False, include_raw_content=False)
    
    if response:
        formatted_response = parse_and_format_response(response)
        container.markdown(formatted_response)
