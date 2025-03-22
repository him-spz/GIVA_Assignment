import pandas as pd

def load_documents(file_path):
    """
    Load documents from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file containing documents.

    Returns:
    - List[str]: A list of cleaned text documents.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Drop rows where the 'text' column is missing or null
    df = df.dropna(subset=["text"])
    
    # Return the 'text' column as a list of strings
    return df["text"].tolist()
