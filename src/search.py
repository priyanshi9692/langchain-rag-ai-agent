import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", model_name: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, model_name)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not os.path.exists(faiss_path) and os.path.exists(meta_path):
            from data_loader import load_all_documents
            documents = load_all_documents(data_dir="data")
            self.vectorstore.build_from_documents(documents)
        else:
            self.vectorstore.load()
        self.llm = ChatGroq(model=llm_model, api_key=os.getenv("GROQ_API_KEY"))
        print(f"RAGSearch initialized with model: {model_name}, llm_model: {llm_model}")


    def search_and_summarize(self, query: str, k: int = 5)-> str:
        results = self.vectorstore.query(query, k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n".join(texts)

        if not context:
            return "No relevant information found"

        prompt = f"""
        You are a helpful assistant that summarizes documents.
        Here is the context:
        {context}
        Here is the query:
        {query}
        """
        response = self.llm.invoke([prompt])
        return response.content