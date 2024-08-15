#import os
#import requests
import streamlit as st
#from dotenv import load_dotenv, dotenv_values
#from streamlit_elements import elements, mui, html


# Set the title of the web page
st.title("🚧 Under Construction 🚧")

# Display the main message
st.write("This page is currently under construction. Please check back later!")

# Optionally, you can add an image or emoji to make it more visually appealing
st.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", caption="We're working on it!")

# You can also add a footer message
st.write("Thank you for your patience.")

## 

"""
#load_dotenv()

api_key=os.environ.get('NEWS_API_KEY')
query="artificial%20intelligence"

def fetch_news(api_key, query):
    url = f'https://newsapi.org/v2/top-headlines?q={query}&apiKey={api_key}'
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content
        data = response.json()
        
        # Extract the articles list
        articles = data.get("articles", [])
        
        # Create a list to store the formatted articles
        formatted_articles = []
        
        # Loop through each article and extract specific fields
        for article in articles:
            formatted_article = {
                "source_id": article['source'].get('id'),
                "source_name": article['source'].get('name'),
                "author": article.get('author'),
                "title": article.get('title'),
                "description": article.get('description'),
                "url": article.get('url'),
                "urlToImage": article.get('urlToImage'),
                "publishedAt": article.get('publishedAt'),
                "content": article.get('content')
            }
            formatted_articles.append(formatted_article)
        
        # Return the formatted articles
        return formatted_articles
    else:
        # Handle errors
        print(f"Failed to retrieve data: {response.status_code}")
        return []

articles = fetch_news(api_key, query)

# Print the formatted articles
for article in articles:
    print(f"Title: {article['title']}")
    print(f"Source: {article['source_name']}")
    print(f"Description: {article['description']}")
    print(f"Published At: {article['publishedAt']}")
    print(f"URL: {article['url']}")
    print()

pg.run()
"""
