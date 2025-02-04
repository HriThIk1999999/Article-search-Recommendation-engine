**Article Recommendation Search Engine**

**📌 Overview**

This project is an Article Recommendation Search Engine that scrapes articles from The Guardian and uses NLP-based embeddings to provide users with relevant articles based on their search queries. The project leverages Hugging Face sentence-transformers, LlamaIndex, and Streamlit for an interactive user experience.

**🏗️ Features**

Automated Web Scraping: Fetches articles from The Guardian API across multiple categories.

Preprocessing & Cleaning: Cleans HTML content, removes stopwords, and tokenizes text.

Embeddings Generation: Uses MiniLM-L6-v2 for sentence embeddings.

Vector Search Indexing: Stores and retrieves relevant articles efficiently.

Interactive UI: Users can enter a search query and receive recommended articles.

Optimized for Performance: Utilizes Torch GPU acceleration (if available) for embedding computations.

**🛠️ Tech Stack**

Python (Core programming language)

Streamlit (Frontend UI for searching articles)

BeautifulSoup (Web scraping)

NLTK (Text preprocessing)

Hugging Face Transformers (Embeddings)

LlamaIndex (Vector search)

Torch (CUDA-accelerated computing)

**📂 Project Structure**

📦 Article-Recommendation-Search-Engine
├── 📜 embedding.py  # Scrapes articles, preprocesses data & generates embeddings

├── 📜 app.py               # Streamlit-based user interface

├── 📜 requirements.txt     # List of dependencies

├── 📜 README.md            # Project documentation

├── 📂 dataset              # Directory containing the scraped articles & embeddings

│   ├── articles.jsonl      # Scraped article dataset
│   ├── article_embeddings.pkl  # Saved embeddings for retrieval


**🚀 Installation & Setup**

1️⃣ Clone the Repository

**git clone https://github.com/yourusername/Article-Recommendation-Search-Engine.git
cd Article-Recommendation-Search-Engine**

2️⃣ Install Dependencies

**pip install -r requirements.txt**

3️⃣ Run the Scraper & Generate Embeddings

**python embedding.py**

4️⃣ Start the Streamlit App

**streamlit run app.py**

**🎯 How It Works**

**1️⃣ Scraping & Preprocessing:**

Articles are fetched from The Guardian API for the past 3 months.

HTML is parsed, stopwords are removed, and text is tokenized.

**2️⃣ Generating Embeddings:**

MiniLM-L6-v2 model generates vector embeddings for articles.

Data is stored in article_embeddings.pkl for efficient retrieval.

**3️⃣ Searching for Articles:**

Users enter a search query via Streamlit UI.

The system retrieves the most relevant articles using vector similarity.

Titles, sections, and publication dates are displayed with clickable links.

**🤝 Contribution**

Feel free to fork the repository, create a pull request, or open an issue to suggest improvements!

**Contact**

Email- hrithik99singh@gmail.com
