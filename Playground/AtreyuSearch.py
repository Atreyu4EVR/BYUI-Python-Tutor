import os
import json
import time
import logging
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
st.subheader("AtreyuSearch")
st.markdown("Powered by *[Tavily](https://tavily.com/)*")

def generate_response(input_text, my_bar, method='search', **kwargs):
    try:
        my_bar = st.progress(0)
        
        for percent_complete in range(100):
            # Update the progress text based on the percentage of completion.
            if percent_complete < 50:
                progress_text = f"Fetching Artax..."
            else:
                progress_text = f"Found the Child Empress..."
            
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Perform the search
        response = client.search(input_text, **kwargs) if method == 'search' else None

        return response

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

def parse_and_format_response(response, my_bar):
    """Parse the response and format it for display using Markdown."""
    try:
        formatted_response = ""

        # Extract the query and response time
        answer = response.get("answer", "No answer found")
        response_time = response.get("response_time", "N/A")

        formatted_response += f"### Answer:\n\n ***{answer}***\n "

        # Extract and format the search results
        results = response.get("results", [])
        for result in results:
            title = result.get("title", "No title")
            url = result.get("url", "#")
            content = result.get("content", "No content available")

            formatted_response += f"##### [{title}]({url})\n\n*{content}*\n\n"

        return formatted_response
    except Exception as e:
        return f"Error parsing response: {str(e)}"

# Search Input in Form for Tavily AI Search
with st.form("web_search", border=False):
    col1, col2 = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with col1:
        input_text = st.text_input("", placeholder="Search the web using Atreyu")
    with col2:
        submitted = st.form_submit_button("Search")

# Display Results
container = st.container(border=True)

if submitted and input_text:

    # Generate and display the assistant's response
    response = generate_response(input_text, method='search', search_depth='advanced', max_results=5, include_answer=True, include_images=False, include_raw_content=False)
    
    if response:
        formatted_response = parse_and_format_response(response)
        container.markdown(formatted_response)
