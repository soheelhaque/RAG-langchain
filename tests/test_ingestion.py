"""Offline tests for document loading and chunk metadata."""

from src.ingestion import load_documents, split_documents


def test_load_documents_loads_the_financial_corpus() -> None:
    """Load every expected text document with its source metadata."""
    documents = load_documents()

    assert len(documents) == 5
    assert all(document.page_content for document in documents)
    assert all("source" in document.metadata for document in documents)


def test_split_documents_adds_source_line_ranges() -> None:
    """Attach valid original-file line ranges to every generated chunk."""
    documents = load_documents()
    chunks = split_documents(documents)

    assert chunks
    for chunk in chunks:
        start_line = chunk.metadata["start_line"]
        end_line = chunk.metadata["end_line"]

        assert isinstance(start_line, int)
        assert isinstance(end_line, int)
        assert start_line >= 1
        assert end_line >= start_line
