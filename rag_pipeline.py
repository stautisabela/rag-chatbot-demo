import openai
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from document_processor import store

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def set_api_key(api_key):
    openai.api_key = api_key

def is_similar(text1, text2, threshold=0.85):
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def filter_duplicates(sources, threshold=0.85):
    unique_sources = []
    for source in sources:
        if all(not is_similar(source, existing, threshold) for existing in unique_sources):
            unique_sources.append(source)
    return unique_sources

def retrieve_and_answer(query):
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please provide a valid API key.")

    query_embedding = embedding_model.encode([query])

    # Retrieve more than needed to allow filtering
    sources, distances = store.search(query_embedding, k=10)
    sources = filter_duplicates(sources)
    
    # Limit context to top 3 unique sources
    context = "\n".join(sources[:3]) 
    
    # Use LLM to generate an answer
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion:\n{query}"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    
    # Return answer + top 5 sources
    return answer, sources[:5]