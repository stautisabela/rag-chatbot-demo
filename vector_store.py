import faiss

class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.doc_store = []

    def add_documents(self, documents, embeddings):
        self.index.add(embeddings)
        self.doc_store.extend(documents)

    def search(self, query_embedding, k=5):
        distances, indexes = self.index.search(query_embedding, k)
        return [self.doc_store[i] for i in indexes[0]], distances[0]

    def reset(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.doc_store = []
