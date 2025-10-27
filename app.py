from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.embedding import EmbeddingManager
from src.search import RAGSearch


if __name__ == "__main__":
    print("[Info] Starting the application...")

    vector_store = FaissVectorStore(persist_dir="faiss_store")

    print("[Info] Searching for the main idea of the India and South Asia relationships document...")
    vector_store.load()

    rag_search = RAGSearch(persist_dir="faiss_store", model_name="all-MiniLM-L6-v2", llm_model="llama-3.1-8b-instant")
    response = rag_search.search_and_summarize("What is the main idea of the India and South Asia relationships document?")
    print(f"[Info] Response: {response}")



