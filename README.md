# Local Documents Chatbot
Learning project exploring a RAG system for local document-based QA.

## Overview
This project locally creates a Streamlit Python application that takes in PDF files uploaded by the user, extracts relevant text from it, and retrieves answers to queries using GPT models from OpenAI.

## Architecture
- Python: Backend logic for chatbot
- Streamlit: Python framework for user interface
- FAISS: Local vector database for efficient similarity search
- LangChain: Embedding retrieval pipeline
- OpenAI: Answer generation

## Run app locally
1. Clone this repository and cd to it
2. Install dependencies with ```$ pip install -r requirements.txt```
3. Optionally create a ```.env``` file with your Open AI key. You can also manually add your key through the UI later
4. Run the app with ```$ streamlit run app.py``` and go to URL isplayed on logs