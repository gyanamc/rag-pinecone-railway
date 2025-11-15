# RAG Pinecone + Railway Project

A complete Retrieval-Augmented Generation (RAG) system built with FastAPI, Pinecone vector database, and OpenAI, designed for easy deployment on Railway.

## 🚀 Features

- **Document Ingestion**: Upload and process documents with automatic chunking
- **Vector Storage**: Store document embeddings in Pinecone for fast similarity search
- **RAG Query System**: Query your knowledge base with natural language questions
- **RESTful API**: Clean FastAPI endpoints for easy integration
- **Railway Ready**: Pre-configured for seamless deployment on Railway

## 📋 Prerequisites

- Python 3.8+
- Pinecone account ([sign up here](https://www.pinecone.io/))
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Railway account (optional, for deployment)

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-pinecone-railway
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-index
PINECONE_ENVIRONMENT=us-west1-gcp-free

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
API_PORT=8000
ENVIRONMENT=development
```

### 5. Run the application

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Endpoints

### Health Check

```bash
GET /
GET /health
```

### Add Documents

```bash
POST /documents
Content-Type: application/json

{
  "text": "Your document content here...",
  "metadata": {
    "source": "example.pdf",
    "author": "John Doe"
  }
}
```

### Query the RAG System

```bash
POST /query
Content-Type: application/json

{
  "question": "What is the main topic of the documents?"
}
```

**Response:**
```json
{
  "answer": "The answer based on your documents...",
  "source_documents": [
    {
      "content": "Relevant chunk of text...",
      "metadata": {
        "source": "example.pdf"
      }
    }
  ]
}
```

## 🔍 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🚢 Deployment on Railway

### 1. Install Railway CLI (optional)

```bash
npm install -g @railway/cli
```

### 2. Login to Railway

```bash
railway login
```

### 3. Initialize Railway project

```bash
railway init
```

### 4. Add environment variables

Set the following environment variables in Railway:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `OPENAI_API_KEY`

You can set them via Railway dashboard or CLI:
```bash
railway variables set PINECONE_API_KEY=your_key
railway variables set OPENAI_API_KEY=your_key
```

### 5. Deploy

```bash
railway up
```

Or push to your connected GitHub repository - Railway will auto-deploy!

## 🏗️ Project Structure

```
rag-pinecone-railway/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── pinecone_client.py   # Pinecone integration
│   ├── document_processor.py # Document chunking
│   └── rag_service.py       # RAG service logic
├── env.example              # Environment variables template
├── .gitignore
├── Procfile                 # Railway process file
├── railway.json             # Railway configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## 🔧 Configuration

### Pinecone Settings

- **Index Name**: Change `PINECONE_INDEX_NAME` to use a different index
- **Dimension**: Currently set to 1536 (for `text-embedding-3-small`)
- **Metric**: Cosine similarity

### OpenAI Settings

- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **LLM Model**: `gpt-3.5-turbo` (can be changed in `rag_service.py`)

### Document Processing

- **Chunk Size**: 1000 characters (configurable in `document_processor.py`)
- **Chunk Overlap**: 200 characters (configurable)

## 📝 Usage Example

### 1. Add a document

```python
import requests

url = "http://localhost:8000/documents"
payload = {
    "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms...",
    "metadata": {
        "source": "ml-intro.txt",
        "topic": "machine-learning"
    }
}
response = requests.post(url, json=payload)
print(response.json())
```

### 2. Query the system

```python
url = "http://localhost:8000/query"
payload = {
    "question": "What is machine learning?"
}
response = requests.post(url, json=payload)
print(response.json())
```

## 🐛 Troubleshooting

### Pinecone Index Not Found

If you get an error about the index not existing:
1. The index will be created automatically on first startup
2. Make sure your Pinecone API key has proper permissions
3. Check that you're using the correct Pinecone environment

### OpenAI API Errors

- Verify your API key is correct
- Check your OpenAI account has sufficient credits
- Ensure the API key has proper permissions

### Port Already in Use

If port 8000 is already in use, change `API_PORT` in your `.env` file or run:
```bash
uvicorn app.main:app --port 8001
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Pinecone](https://www.pinecone.io/) for vector database
- [OpenAI](https://openai.com/) for embeddings and language models
- [LangChain](https://www.langchain.com/) for RAG orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Railway](https://railway.app/) for deployment platform
