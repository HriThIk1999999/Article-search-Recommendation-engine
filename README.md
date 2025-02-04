Article Recommendation Search Engine

📌 Overview

This project is an Article Recommendation Search Engine that scrapes articles from The Guardian and uses NLP-based embeddings to provide users with relevant articles based on their search queries. The project leverages Hugging Face sentence-transformers, LlamaIndex, and Streamlit for an interactive user experience.

🏗️ Features

Automated Web Scraping: Fetches articles from The Guardian API across multiple categories.

Preprocessing & Cleaning: Cleans HTML content, removes stopwords, and tokenizes text.

Embeddings Generation: Uses MiniLM-L6-v2 for sentence embeddings.

Vector Search Indexing: Stores and retrieves relevant articles efficiently.

Interactive UI: Users can enter a search query and receive recommended articles.

Optimized for Performance: Utilizes Torch GPU acceleration (if available) for embedding computations.

🛠️ Tech Stack

Python (Core programming language)

Streamlit (Frontend UI for searching articles)

BeautifulSoup (Web scraping)

NLTK (Text preprocessing)

Hugging Face Transformers (Embeddings)

LlamaIndex (Vector search)

Torch (CUDA-accelerated computing)
