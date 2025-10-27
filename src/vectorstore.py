from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os
from typing import List, Any
from src.embedding import EmbeddingManager
import numpy as np
import faiss
import pickle

class FaissVectorStore:
    def __init__(self, persist_dir:str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"FaissVectorStore initialized with model: {embedding_model}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")


    def build_from_documents(self, documents: List[Any]):

        embedding_manager = EmbeddingManager(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = embedding_manager.chunk_documents(documents)

        embedding = embedding_manager.embed_chunks(chunks)

        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embedding).astype(np.float32), metadatas)
        self.save()
        print(f"FaissVectorStore built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]= None):
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)
        print(f"Added faiss index and metadata to {self.persist_dir}")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index,faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded faiss index and metadata from {self.persist_dir}")

    def search(self, query: str, k: int = 5) -> List[Any]:
        embedding = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(embedding, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({
                "index": idx,
                "distance": dist,
                "metadata": meta
            })
        return results
    
    def query(self, query: str, k: int = 5) -> List[Any]:
        results = self.search(query, k)
        return results
