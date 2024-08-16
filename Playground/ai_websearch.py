import os
import logging
import sys
from tavily import TavilyClient
from dotenv import load_dotenv, dotenv_values

load_dotenv()

api_key = os.environ.get('TAVILY_API_KEY') #for Tavily

client = TavilyClient(api_key=apy_key)

# Show title and description.
st.title("AI Powered WebSearch")

# Step 2. Executing a simple search query
response = client.search("Who is Leo Messi?")

# Step 3. That's it! You've done a Tavily Search!
print(response)
