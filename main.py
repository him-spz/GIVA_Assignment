from fastapi import FastAPI, Query
from pydantic import BaseModel
from vector_store import search_top_k, add_document

# Initialize FastAPI application
app = FastAPI()

# Define the data model for adding new documents
class NewDocument(BaseModel):
    text: str  # The text content of the new document

# Search endpoint: Returns top 5 similar documents to the given query
@app.get("/api/search")
def search(
    q: str = Query(..., description="Search query string"),  # Required query string
    metric: str = Query("l2", description="Similarity metric: l2, cosine, dot")  # Optional metric
):
    """
    Search for the top 5 most similar documents using the selected similarity metric.
    Supported metrics: l2 (default), cosine, dot
    """
    results = search_top_k(q, top_k=5, metric=metric.lower())
    return {
        "query": q,
        "metric": metric,
        "results": results
    }

# Add endpoint: Accepts a new document and adds it to the vector store
@app.post("/api/add")
def add_doc(doc: NewDocument):
    """
    Add a new document to the index in real-time.
    Also appends the document to documents.csv for persistence.
    """
    result = add_document(doc.text)
    return result
