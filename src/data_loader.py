from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader, DirectoryLoader, CSVLoader, JSONLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_all_documents(data_dir: str) -> List[Any]:
    data_path = Path(data_dir).resolve()

    print(f"[Debug] Data Path: {data_path}")

    documents = []

    pdf_files = list(data_path.glob("**/*.pdf"))

    print(f"[Debug] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"[Debug] Loading PDF file: {pdf_file}")
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            print(f"[Debug] Loaded {len(docs)} documents from {pdf_file}")
            documents.extend(docs)
        except Exception as e:
            print(f"[Error] Failed to load PDF file: {pdf_file} - {e}")
            continue

    return documents