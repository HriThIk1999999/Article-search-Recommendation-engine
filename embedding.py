import json
import torch
import pickle
import os
import requests
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from langdetect import detect  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document

#  Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load Stopwords
stop_words_english = set(stopwords.words('english'))
stop_words_german = set(stopwords.words('german'))

#  API Key
api_key = "387f971f-0b7e-4bf2-9a5c-2bba3a2c638f"

#  Sections to Fetch
sections = ["technology", "science", "sports", "health", "education", "finance"]

# Date Range (Last 30 Days)
from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
to_date = datetime.now().strftime("%Y-%m-%d")

# Guardian API URL
base_url = "https://content.guardianapis.com/search"

def fetch_articles(section, page=1):
    """Fetch articles from The Guardian API."""
    params = {
        "api-key": api_key,
        "section": section,
        "from-date": from_date,
        "to-date": to_date,
        "show-fields": "body",
        "page-size": 50,
        "page": page
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get("response", {})
    else:
        print(f"Failed to fetch articles for {section}. Status code: {response.status_code}")
        return {}

# Collect Articles
all_articles = []
for section in sections:
    page = 1
    while True:
        data = fetch_articles(section, page)
        results = data.get("results", [])
        if not results:
            break
        all_articles.extend(results)
        if page >= data.get("pages", 1):
            break
        page += 1

# Define save directory
save_dir = "/content/dataset"
os.makedirs(save_dir, exist_ok=True)

# Save articles to JSONL (only if articles exist)
jsonl_file_path = os.path.join(save_dir, "articles.jsonl")
if all_articles:
    with open(jsonl_file_path, "w", encoding="utf-8") as f:
        for article in all_articles:
            json.dump(article, f)
            f.write("\n")
    print(f" Saved {len(all_articles)} articles to {jsonl_file_path}")
else:
    print(" Warning: No articles fetched. JSONL file not saved.")

def load_embedding_model():
    """Load the embedding model from HuggingFace."""
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

def parse_jsonl(file_path):
    """Parse entire JSONL file and return a list of dictionaries."""
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data_list.append(json.loads(line))  # Load each JSON object
        return data_list
    except FileNotFoundError:
        print(f" Error: File {file_path} not found.")
        return []

def clean_content(text):
    """Clean the text by removing special characters, tokenizing, and removing stop words."""
    text = re.sub(r'[^a-zA-Z0-9\sÄäÖöÜüß]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    
    try:
        language = detect(text)  # Detect the language
    except:
        language = 'en'  # Default to English if detection fails
    
    # Choose stop words based on language
    stop_words = stop_words_german if language == 'de' else stop_words_english

    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

def create_documents(data_list):
    """Create Document objects from dataset."""
    documents = []
    for row in data_list:
        parsed_text = BeautifulSoup(row.get("fields", {}).get("body", ""), 'html.parser').get_text(separator=' ', strip=True)
        cleaned_text = clean_content(parsed_text)

        documents.append(Document(
            text=cleaned_text,
            metadata={
                "title": row.get("webTitle", ""),
                "section": row.get("sectionName", ""),
                "url": row.get("webUrl", ""),
                "date": row.get("webPublicationDate", "")
            }
        ))
    return documents

def generate_embeddings(documents, embed_model):
    """Generate embeddings for documents."""
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)  # Fixed embedding insertion
    return index

def save_embeddings(index, file_path="/content/dataset/article_embeddings.pkl"):
    """Save embeddings to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"Embeddings saved to {file_path}")

def main():
    file_path = "/content/dataset/articles.jsonl"  # Fixed path for JSONL file

    try:
        # Step 1: Load JSONL file
        data_list = parse_jsonl(file_path)
        if not data_list:
            print(" No data found. Exiting...")
            return

        print(f"Loaded {len(data_list)} records from {file_path}")

        # Step 2: Create Document objects
        documents = create_documents(data_list)

        # Step 3: Load the embedding model
        embed_model = load_embedding_model()

        # Step 4: Generate embeddings
        index = generate_embeddings(documents, embed_model)

        # Step 5: Save embeddings
        save_embeddings(index)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory

if __name__ == "__main__":
    main()
