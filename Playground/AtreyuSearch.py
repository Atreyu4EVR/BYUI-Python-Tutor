import os
import json
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
st.markdown("AI Web Search Powered by *[Tavily](https://tavily.com/)*")

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
            return None

        return response

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

def parse_and_format_response(response):
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

        return formatted_response
    except Exception as e:
        return f"Error parsing response: {str(e)}"

left, right = st.columns(gap="small", vertical_alignment="top")

# Form for Tavily Web Search
with st.form("web_search"):
    input_text = st.text_input("Enter search query:")
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Display user's query as a chat message
        with st.chat_message("user"):
            st.write(input_text)

        # Generate and display the assistant's response
        response = generate_response(input_text, method='search', search_depth='advanced', max_results=5, include_answer=False, include_images=False, include_raw_content=False)
        
        if response:
            formatted_response = parse_and_format_response(response)
            with st.chat_message("assistant"):
                # Display the formatted response using Markdown
                st.markdown(formatted_response)
