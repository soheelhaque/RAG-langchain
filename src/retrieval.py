"""Semantic retrieval from the FAISS vector store."""

from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class ScoredDocument(TypedDict):
    """A retrieved document with its FAISS distance and cosine similarity."""

    document: Document
    faiss_distance: float
    cosine_similarity: float


def retrieve_with_scores(
    query: str,
    vector_store: FAISS,
    k: int = 4,
) -> list[ScoredDocument]:
    """Retrieve the most relevant chunks with comparable similarity scores.

    Args:
        query: The financial research question used for similarity search.
        vector_store: The FAISS index containing embedded document chunks.
        k: The maximum number of chunks to return.

    Returns:
        Up to ``k`` retrieved chunks, each paired with its raw FAISS squared
        L2 distance and cosine similarity. FAISS distance is lower for closer
        matches; cosine similarity is higher for closer matches.
    """
    document_distances = vector_store.similarity_search_with_score(query, k=k)
    retrieved_documents = []
    for document, faiss_distance in document_distances:
        # Vectors are L2-normalised when the index is built. For unit vectors,
        # squared L2 distance is ``2 - (2 * cosine_similarity)``.
        cosine_similarity = 1 - (float(faiss_distance) / 2)
        retrieved_documents.append(
            {
                "document": document,
                "faiss_distance": float(faiss_distance),
                "cosine_similarity": cosine_similarity,
            }
        )
    return retrieved_documents


def create_chunk_preview(document: Document) -> str:
    """Create a compact preview from the beginning and end of a chunk.

    Args:
        document: The chunk to summarise.

    Returns:
        The first five words, an ellipsis, and the last five words of the
        chunk. Short chunks are returned in full.
    """
    words = document.page_content.split()
    if len(words) <= 10:
        return " ".join(words)
    return f"{' '.join(words[:5])} ... {' '.join(words[-5:])}"


def print_retrieval_debug(retrieved_documents: list[ScoredDocument]) -> None:
    """Print ranked retrieval details for inspection during the demo.

    Args:
        retrieved_documents: The scored document chunks returned by retrieval.

    Returns:
        None. The function writes retrieval details to standard output.
    """
    print("\n--- RETRIEVAL DEBUG ---")
    for rank, result in enumerate(retrieved_documents, start=1):
        document = result["document"]
        source = document.metadata.get("source", "unknown source")
        print(f"\nRank {rank}")
        print(f"FAISS distance (lower is closer): {result['faiss_distance']:.4f}")
        print(f"Cosine similarity (higher is closer): {result['cosine_similarity']:.4f}")
        print(f"Source: {source}")
        print(f"Text: {create_chunk_preview(document)}")
        print(f"Start line: {document.metadata.get('start_line', 'unknown')}")
        print(f"End line: {document.metadata.get('end_line', 'unknown')}")
    print()
