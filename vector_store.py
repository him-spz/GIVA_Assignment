import faiss
import numpy as np
from embedding import embed_documents, embed_query
from data_loader import load_documents

# -------------------------------
# Load & Preprocess Documents
# -------------------------------

# Load documents from CSV
documents = load_documents("documents.csv")

# Generate embeddings for the documents
doc_embeddings = embed_documents(documents)
doc_embeddings_np = np.array(doc_embeddings)

# Determine embedding dimension
dim = doc_embeddings_np.shape[1]

# Normalize embeddings for cosine similarity
normed_embeddings = doc_embeddings_np / np.linalg.norm(doc_embeddings_np, axis=1, keepdims=True)

# -------------------------------
# Initialize FAISS Indexes
# -------------------------------

# 1. L2 (Euclidean Distance)
index_l2 = faiss.IndexFlatL2(dim)
index_l2.add(doc_embeddings_np)

# 2. Cosine Similarity (via normalized inner product)
index_cosine = faiss.IndexFlatIP(dim)
index_cosine.add(normed_embeddings)

# 3. Dot Product
index_dot = faiss.IndexFlatIP(dim)
index_dot.add(doc_embeddings_np)

# -------------------------------
# Search Function
# -------------------------------

def search_top_k(query, top_k=5, metric="l2"):
    """
    Search for top-k most similar documents using a specified similarity metric.

    Parameters:
    - query (str): The search query.
    - top_k (int): Number of top results to return.
    - metric (str): One of ['l2', 'cosine', 'dot'].

    Returns:
    - List[dict]: Ranked list of most similar documents with scores.
    """
    query_vector = embed_query(query)

    # Select similarity metric
    if metric == "cosine":
        query_vector = query_vector / np.linalg.norm(query_vector)
        distances, indices = index_cosine.search(np.array([query_vector]), top_k)
    elif metric == "dot":
        distances, indices = index_dot.search(np.array([query_vector]), top_k)
    else:  # Default: L2
        distances, indices = index_l2.search(np.array([query_vector]), top_k)

    # Format search results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "text": documents[idx],
            "score": float(distances[0][i])
        })
    return results

# -------------------------------
# Add New Document (Real-time)
# -------------------------------

def add_document(new_text: str):
    """
    Add a new document to the vector index and persist it to disk.

    Parameters:
    - new_text (str): The document to add.

    Returns:
    - dict: Confirmation message and document ID.
    """
    new_vector = embed_query(new_text)
    new_vector_np = np.array([new_vector])

    # Add to all FAISS indexes
    index_l2.add(new_vector_np)
    normed_vector = new_vector / np.linalg.norm(new_vector)
    index_cosine.add(np.array([normed_vector]))
    index_dot.add(new_vector_np)

    # Update in-memory list
    documents.append(new_text)

    # Persist new document to CSV
    # Persist to CSV
    import csv

# Persist to CSV properly
    with open("documents.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([new_text.strip()])



    return {
        "message": "Document added and saved",
        "id": len(documents) - 1
    }
