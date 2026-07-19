import os
#FAISS (Facebook AI Similarity Search) is a library developed by Meta (formerly Facebook) for fast similarity search and vector search.
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from src.embedding import EmbeddingPipeline

# Sentence Transformers → Convert text into numerical vectors (embeddings).
# FAISS → Stores and searches those vectors efficiently.

class FaissVectorStore :
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self,documents:List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas) 
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            # L2 means it measures Euclidean distance (straight-line distance between vectors). Smaller D = more similar. FAISS compares your query vector against every stored vector and returns the closest
            self.index = faiss.IndexFlatL2(dim)
            print(f"[INFO] Created new FAISS index with dimension {dim}")
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {len(embeddings)} embeddings. Total vectors: {self.index.ntotal}")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved vector store to {self.persist_dir}")
    
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded vector store from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # D  →  distances array, shape (1, top_k) — how far each result is
        # I  →  indices array,   shape (1, top_k) — position in the stored vectors
        # self.index.search() is a FAISS library method — not something you define. It is the core similarity search engine that makes the RAG retrieval work.
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying for '{query_text}'...")
        query_emb = self.model.encode([query_text]).astype('float32')
        results = self.search(query_emb, top_k)
        return results

# Example usage
if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    # print(store.query("What is Artificial Intelligence, When It was introduced?", top_k=3))