import streamlit as st
import pickle
import torch
import logging
import os
from huggingface_hub import login
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from the environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Check if the token is correctly loaded
if HF_TOKEN is None:
    logging.error("❌ Hugging Face token is not set. Please add it to the .env file.")
else:
    logging.info("✅ Hugging Face token loaded successfully.")

# Authenticate with Hugging Face using the token
login(HF_TOKEN)

# Ensure Streamlit caches models correctly
@st.cache_resource
def load_llm_and_embed_model():
    try:
        # Load Hugging Face LLM (Ensure Model is Unlocked)
        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=500,
            generate_kwargs={"temperature": 0.5, "do_sample": False},
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",
            model_kwargs={"torch_dtype": torch.float16}
        )

        # Load Hugging Face Embeddings
        embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        logging.info("✅ Models loaded successfully!")

        return llm, embed_model

    except Exception as e:
        logging.error(f"❌ Error loading models: {e}")
        raise

# Load embeddings from saved file
def load_embeddings(file_path="/content/dataset/article_embeddings.pkl"):
    try:
        with open(file_path, 'rb') as f:
            index = pickle.load(f)
        return index
    except Exception as e:
        logging.error(f"❌ Error loading embeddings: {e}")
        raise

# Configure the query engine
def configure_query_engine(index, embed_model, llm):
    try:
        # Create Query Engine (No SentenceSplitter)
        query_engine = index.as_query_engine(llm=llm)
        return query_engine

    except Exception as e:
        logging.error(f"❌ Error configuring query engine: {e}")
        raise

# Main Streamlit Function
def main():
    st.title("📖 Article Recommendation Search Engine - The Guardian")

    # Add a description to explain the app to users
    st.markdown("Welcome to the **Article Search Engine**. Ask questions about the articles, and get relevant responses based on the search engine model.")

    try:
        # Load LLM and Embeddings
        llm, embed_model = load_llm_and_embed_model()
        index = load_embeddings()
        query_engine = configure_query_engine(index, embed_model, llm)

        # User Query Input
        query = st.text_input("🔍 Ask a question about articles:")

        if query:
            logging.info(f"Query received: {query}")
            response = query_engine.query(query)
            st.write("📝 Response:", response.response.strip())

            # Free GPU Memory After Query
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("✅ CUDA cache cleared.")

    # Handle GPU Out of Memory Errors
    except torch.cuda.OutOfMemoryError as e:
        st.error(f"⚠️ CUDA out of memory error: {str(e)}")
        logging.error(f"CUDA OOM error: {str(e)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.warning("✅ CUDA memory cleared, try again.")

    # Handle General Errors
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()

