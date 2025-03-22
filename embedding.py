from sentence_transformers import SentenceTransformer

# Load the sentence transformer model once at module level
# Model: all-MiniLM-L6-v2 (lightweight, fast, good performance for semantic search)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_documents(docs):
    """
    Generate vector embeddings for a list of documents.

    Parameters:
    - docs (List[str]): List of text documents.

    Returns:
    - List[np.ndarray]: Corresponding list of vector embeddings.
    """
    return model.encode(docs, show_progress_bar=True)


def embed_query(query):
    """
    Generate a vector embedding for a single query.

    Parameters:
    - query (str): The query string.

    Returns:
    - np.ndarray: Vector embedding of the query.
    """
    return model.encode([query])[0]
