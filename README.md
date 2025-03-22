# 🧠 Document Similarity Search API

This is a FastAPI-based project that finds similar documents using sentence embeddings and vector search (FAISS). You can search for similar documents and even add new ones in real-time!

---

## 🚀 Features

- 🔍 Search for the top 5 most similar documents
- 📎 Uses Hugging Face sentence-transformers (`all-MiniLM-L6-v2`)
- 🧠 Vector search with FAISS (supports L2, cosine, dot product)
- ➕ Real-time document addition via API
- 💾 Persistent storage using `documents.csv`

---

## 📦 Tech Stack

- **Backend:** Python 3, FastAPI
- **ML Model:** Hugging Face `sentence-transformers`
- **Vector DB:** FAISS (in-memory)
- **API Docs:** Swagger UI (auto-generated)

---

## 📂 Project Structure

document_similarity_api/ 
├── main.py 
├── data_loader.py 
├── embedding.py 
├── vector_store.py 
├── documents.csv 
├── requirements.txt 
└── README.md


---

## 📥 Setup Instructions

1. Clone the Repo
git clone https://github.com/yourusername/document-similarity-api.git
cd document-similarity-api

2. Install Dependencies
pip install -r requirements.txt

3. Run the Server
uvicorn main:app --reload
Visit: http://127.0.0.1:8000/docs for the interactive API.

🔗 API Endpoints

GET /api/search?q=your+query&metric=cosine
q: search query (required)

metric: l2 (default), cosine, or dot

POST /api/add
Add a new document:
{
  "text": "This is a new document."
}

✨ Bonus Features
✅ Real-time document indexing

✅ Persistent storage in documents.csv

✅ Multiple similarity metrics

