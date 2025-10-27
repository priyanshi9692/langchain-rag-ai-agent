# LangChain RAG AI Agent

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, FAISS, and Groq LLM. This application enables semantic search and question-answering over document collections using state-of-the-art embedding models and large language models.

## Overview

This RAG system processes documents, creates vector embeddings, stores them in a FAISS index, and provides intelligent search capabilities with LLM-powered summarization. The application is designed for efficient document retrieval and contextual question-answering.

## Features

- **Multi-format Document Loading**: Support for PDF, TXT, CSV, JSON, DOCX, and Excel files
- **Intelligent Text Chunking**: Recursive character-based text splitting with configurable chunk sizes
- **Vector Embeddings**: Sentence-Transformers integration for semantic document representation
- **FAISS Vector Store**: High-performance similarity search with persistent storage
- **LLM Integration**: Groq API integration for advanced language understanding and generation
- **Modular Architecture**: Clean separation of concerns with dedicated modules for each functionality

## Architecture

The system follows a modular design pattern:

```
Document Loading → Text Chunking → Embedding Generation → Vector Storage → Semantic Search → LLM Summarization
```

### Core Components

1. **Data Loader**: Handles document ingestion from various file formats
2. **Embedding Manager**: Processes documents into chunks and generates vector embeddings
3. **Vector Store**: Manages FAISS index creation, persistence, and similarity search
4. **RAG Search**: Orchestrates retrieval and LLM-based response generation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-rag-ai-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: lang-venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

To obtain a Groq API key:
1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Generate a new API key

### Model Configuration

The system uses the following models by default:

- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence-Transformers)
- **LLM Model**: `llama-3.1-8b-instant` (Groq)

These can be configured in the initialization parameters.

## Usage

### Basic Usage

1. Place your documents in the `data/` directory:
```
data/
├── document1.pdf
├── document2.txt
└── subdirectory/
    └── document3.pdf
```

2. Run the application:
```bash
python app.py
```

### Programmatic Usage

#### Building a Vector Store

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

# Load documents
documents = load_all_documents(data_dir="data")

# Initialize and build vector store
vector_store = FaissVectorStore(persist_dir="faiss_store")
vector_store.build_from_documents(documents)
```

#### Performing RAG Search

```python
from src.search import RAGSearch

# Initialize RAG Search
rag_search = RAGSearch(
    persist_dir="faiss_store",
    model_name="all-MiniLM-L6-v2",
    llm_model="llama-3.1-8b-instant"
)

# Query the system
response = rag_search.search_and_summarize("What are the key findings?")
print(response)
```

#### Direct Vector Search

```python
from src.vectorstore import FaissVectorStore

# Load existing vector store
vector_store = FaissVectorStore(persist_dir="faiss_store")
vector_store.load()

# Perform similarity search
results = vector_store.search("query text", k=5)
for result in results:
    print(f"Distance: {result['distance']}")
    print(f"Content: {result['metadata']['text']}")
```

## Project Structure

```
langchain-rag-ai-agent/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env                        # Environment variables (create this)
├── data/                       # Document storage
│   ├── text_files/            # Text documents
│   └── *.pdf                  # PDF documents
├── faiss_store/               # Persisted vector store
│   ├── faiss.index           # FAISS index file
│   └── metadata.pkl          # Document metadata
├── notebook/                  # Jupyter notebooks
│   └── document.ipynb        # Experimental notebook
├── src/                       # Source code modules
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Document loading utilities
│   ├── embedding.py          # Embedding generation
│   ├── vectorstore.py        # FAISS vector store management
│   └── search.py             # RAG search implementation
└── lang-venv/                # Virtual environment (gitignored)
```

## Module Documentation

### data_loader.py

Handles document ingestion from various file formats including PDF, TXT, CSV, JSON, DOCX, and Excel.

**Key Function**:
- `load_all_documents(data_dir: str)`: Recursively loads all PDF documents from a directory

### embedding.py

Manages text chunking and embedding generation using Sentence-Transformers.

**Key Methods**:
- `chunk_documents(documents)`: Splits documents into manageable chunks
- `embed_chunks(chunks)`: Generates vector embeddings for text chunks

**Configuration**:
- `chunk_size`: Default 1000 characters
- `chunk_overlap`: Default 200 characters

### vectorstore.py

Implements FAISS vector store with persistence capabilities.

**Key Methods**:
- `build_from_documents(documents)`: Creates index from documents
- `add_embeddings(embeddings, metadatas)`: Adds vectors to index
- `search(query, k)`: Performs similarity search
- `save()`: Persists index to disk
- `load()`: Loads index from disk

### search.py

Orchestrates RAG workflow combining retrieval and generation.

**Key Methods**:
- `search_and_summarize(query, k)`: Retrieves relevant documents and generates LLM response

## Dependencies

Core dependencies include:

- **langchain**: Framework for LLM applications
- **langchain-community**: Community integrations
- **langchain-groq**: Groq API integration
- **sentence-transformers**: Embedding generation
- **faiss-cpu**: Vector similarity search
- **pypdf/pymupdf**: PDF processing
- **python-dotenv**: Environment variable management

See `requirements.txt` for complete list.

## Performance Considerations

### Chunking Strategy

The default chunking configuration (1000 characters, 200 overlap) balances:
- Context preservation
- Embedding quality
- Search precision

Adjust based on your use case:
- Larger chunks: Better context, fewer embeddings
- Smaller chunks: More precise retrieval, more embeddings

### Vector Store

FAISS provides efficient similarity search:
- **IndexFlatL2**: Exact search with L2 distance
- Suitable for datasets up to millions of vectors
- Consider FAISS GPU for larger datasets

### Embedding Model

`all-MiniLM-L6-v2` offers:
- 384-dimensional embeddings
- Good balance of speed and quality
- Suitable for general-purpose semantic search

Alternative models:
- `all-mpnet-base-v2`: Higher quality, slower
- `all-MiniLM-L12-v2`: Balanced option

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
- **Solution**: Ensure virtual environment is activated and dependencies are installed

**Issue**: `GROQ_API_KEY not found`
- **Solution**: Create `.env` file with valid API key

**Issue**: `Model not found` error from Groq
- **Solution**: Verify model name doesn't include provider prefix (use `llama-3.1-8b-instant`, not `groq/llama-3.1-8b-instant`)

**Issue**: FAISS index not found
- **Solution**: Run `build_from_documents()` before loading or searching

## Future Enhancements

Potential improvements:

- [ ] Support for additional document formats (Markdown, HTML, RTF)
- [ ] Multiple embedding model comparison
- [ ] Query result caching
- [ ] Streaming responses for large queries
- [ ] REST API with FastAPI
- [ ] Web interface with Streamlit/Gradio
- [ ] Multi-language support
- [ ] Document update and deletion capabilities
- [ ] Advanced filtering and metadata search
- [ ] Integration with other LLM providers (OpenAI, Anthropic)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Submit a pull request with description

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)
- [Groq](https://groq.com/)

## Contact

For questions or feedback, please open an issue in the repository.
