import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore(dim=384)

def remove_duplicates(chunks):
    unique_chunks = list(set(chunks))
    return unique_chunks

def reset_store():
    global store
    store = VectorStore(dim=384)

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

    # Embed and store chunks in VectorStore
    embeddings = embedding_model.encode(chunks)
    store.add_documents(chunks, embeddings)