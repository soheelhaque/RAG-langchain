"""Offline tests for document loading and chunk metadata."""

from src.ingestion import CHUNK_OVERLAP, CHUNK_SIZE, load_documents, split_documents


def test_load_documents_loads_the_financial_corpus() -> None:
    """Load every expected text document with its source metadata."""
    documents = load_documents()

    assert len(documents) == 8
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


def test_split_documents_uses_larger_overlapping_chunks_with_note_titles() -> None:
    """Keep each chunk large enough for a full research section and self-describing."""
    documents = load_documents()
    chunks = split_documents(documents)

    assert CHUNK_SIZE == 1_000
    assert CHUNK_OVERLAP == 200
    assert all(chunk.metadata["title"].startswith("Title:") for chunk in chunks)
    assert all(chunk.page_content.startswith(f"{chunk.metadata['title']}\n\n") for chunk in chunks)
