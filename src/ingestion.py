"""Document loading, splitting, embedding, and vector-store setup."""

from pathlib import Path

DATA_DIRECTORY = Path("data/financial_docs")


def load_documents(data_directory: Path = DATA_DIRECTORY):
    """Load source documents from the financial corpus."""
    raise NotImplementedError("Implement document loading.")


def split_documents(documents):
    """Split documents into fixed-size, overlapping chunks."""
    raise NotImplementedError("Implement fixed-length chunking with overlap.")


def create_vector_store(chunks):
    """Embed chunks and store them in FAISS."""
    raise NotImplementedError("Implement embeddings and FAISS vector-store creation.")
