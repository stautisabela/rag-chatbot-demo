import os
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
doc_store = []

def remove_duplicates(chunks):
    unique_chunks = list(set(chunks))
    return unique_chunks

# Reinitializes FAISS index and doc_store
def reset_store():
    global index, doc_store
    index = faiss.IndexFlatL2(384)
    doc_store = []

def process_file(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the PDF
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    # Chunk text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    chunks = remove_duplicates(chunks)
    
    # Embed and store chunks in FAISS
    embeddings = embedding_model.encode(chunks)
    index.add(embeddings)
    doc_store.extend(chunks)
