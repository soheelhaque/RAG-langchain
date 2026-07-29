"""Offline tests for retrieval-debug helpers."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval import create_chunk_preview, print_retrieval_debug, retrieve_with_scores


class DeterministicEmbeddings(Embeddings):
    """Return predictable vectors so retrieval can be tested offline."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each text using a fixed keyword-based vector.

        Args:
            texts: The text strings to embed.

        Returns:
            A two-dimensional vector for each text string.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query using the same fixed keyword-based vector.

        Args:
            text: The query text to embed.

        Returns:
            A two-dimensional vector for the query.
        """
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Map interest-rate text and AI text to separate fixed vectors.

        Args:
            text: The text to map to a vector.

        Returns:
            A two-dimensional deterministic embedding vector.
        """
        return [1.0, 0.0] if "interest" in text.lower() else [0.5, 0.5]


def test_retrieve_with_scores_ranks_the_closest_chunk_first() -> None:
    """Rank the interest-rate chunk above an unrelated AI chunk."""
    documents = [
        Document(
            page_content="Interest rates affect equity valuations.",
            metadata={"source": "rates.txt"},
        ),
        Document(page_content="AI spending supports growth.", metadata={"source": "ai.txt"}),
    ]
    vector_store = FAISS.from_documents(documents, DeterministicEmbeddings())

    results = retrieve_with_scores("How do interest rates affect equities?", vector_store)

    assert results[0]["document"].metadata["source"] == "rates.txt"
    assert results[0]["score"] > results[1]["score"]
    assert "interest" in results[0]["explanation"]


def test_create_chunk_preview_shows_the_start_and_end_of_long_chunks() -> None:
    """Show the first and last five words with an ellipsis between them."""
    document = Document(page_content="one two three four five six seven eight nine ten eleven")

    preview = create_chunk_preview(document)

    assert preview == "one two three four five ... seven eight nine ten eleven"


def test_print_retrieval_debug_includes_chunk_location(capsys: object) -> None:
    """Print source, text preview, and source line range for each chunk."""
    document = Document(
        page_content="one two three four five six seven eight nine ten eleven",
        metadata={"source": "example.txt", "start_line": 3, "end_line": 5},
    )
    print_retrieval_debug(
        [{"document": document, "score": 0.9, "explanation": "very strong semantic match"}]
    )

    output = capsys.readouterr().out

    assert "Source: example.txt" in output
    assert "Text: one two three four five ... seven eight nine ten eleven" in output
    assert "Start line: 3" in output
    assert "End line: 5" in output
