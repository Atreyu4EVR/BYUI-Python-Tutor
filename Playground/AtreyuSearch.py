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

def generate_response(input_text, my_bar, method='search', **kwargs):
    try:
        # Update progress to indicate the start
        my_bar.progress(10, text="Starting search...")

        # Perform the search
        response = client.search(input_text, **kwargs) if method == 'search' else None

        # Update progress to indicate progress
        my_bar.progress(60, text="Search complete. Parsing results...")

        return response

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

def parse_and_format_response(response, my_bar):
    """Parse the response and format it for display using Markdown."""
    try:
        formatted_response = ""

        # Extract the query and response time
        query = response.get("query", "No query found")
        response_time = response.get("response_time", "N/A")

        formatted_response += f"**Query:** {query}\n"
        formatted_response += f"**Response Time:** {response_time} seconds\n\n"

        # Extract and format the search results
        results = response.get("results", [])
        for result in results:
            title = result.get("title", "No title")
            url = result.get("url", "#")
            content = result.get("content", "No content available")

            formatted_response += f"**[{title}]({url})**\n\n{content}\n\n"

        # Update progress to indicate completion
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
container = st.container()

if submitted and input_text:
    # Initialize progress bar
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    # Generate and display the assistant's response
    response = generate_response(input_text, my_bar, method='search', search_depth='advanced', max_results=5, include_answer=False, include_images=False, include_raw_content=False)
    
    if response:
        formatted_response = parse_and_format_response(response, my_bar)
        container.markdown(formatted_response)
