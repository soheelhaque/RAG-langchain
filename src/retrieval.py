"""Semantic retrieval from the FAISS vector store."""

from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class ScoredDocument(TypedDict):
    """A retrieved document and its relevance score."""

    document: Document
    score: float
    explanation: str


def explain_match(query: str, document: Document, score: float) -> str:
    """Describe the semantic strength and keyword overlap of a retrieved chunk.

    Args:
        query: The financial research question used for retrieval.
        document: A retrieved document chunk.
        score: The chunk's FAISS relevance score.

    Returns:
        A short explanation of the match strength and direct keyword overlap.
    """
    if score > 0.85:
        strength = "very strong semantic match"
    elif score > 0.75:
        strength = "strong semantic match"
    elif score > 0.65:
        strength = "moderate semantic match"
    else:
        strength = "weak semantic similarity"

    query_terms = set(query.lower().split())
    document_terms = set(document.page_content.lower().split())
    overlap = sorted(query_terms.intersection(document_terms))
    overlap_text = (
        f"Keyword overlap detected: {', '.join(overlap[:5])}"
        if overlap
        else "No direct keyword overlap (semantic match only)"
    )
    return f"{strength}. {overlap_text}"


def retrieve_with_scores(
    query: str,
    vector_store: FAISS,
    k: int = 4,
) -> list[ScoredDocument]:
    """Retrieve the most relevant chunks and their FAISS relevance scores.

    Args:
        query: The financial research question used for similarity search.
        vector_store: The FAISS index containing embedded document chunks.
        k: The maximum number of chunks to return.

    Returns:
        Up to ``k`` retrieved chunks, each paired with a relevance score where
        higher values indicate a closer semantic match.
    """
    document_scores = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "document": document,
            "score": float(score),
            "explanation": explain_match(query, document, float(score)),
        }
        for document, score in document_scores
    ]


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
        print(f"Score: {result['score']:.4f}")
        print(f"Explanation: {result['explanation']}")
        print(f"Source: {source}")
        print(f"Text: {create_chunk_preview(document)}")
        print(f"Start line: {document.metadata.get('start_line', 'unknown')}")
        print(f"End line: {document.metadata.get('end_line', 'unknown')}")
    print()
