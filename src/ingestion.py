"""Document loading, splitting, embedding, and vector-store setup."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIRECTORY = Path("data/financial_docs")
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200


def get_document_title(document: Document) -> str:
    """Return the note title from a source document's header.

    The corpus uses a ``Title:`` header in every source file. Keeping that
    title with each chunk gives excerpts enough subject context to be embedded
    and retrieved independently.
    """
    for line in document.page_content.splitlines():
        if line.startswith("Title:"):
            return line
    return ""


def load_documents(data_directory: Path = DATA_DIRECTORY) -> list[Document]:
    """Load every text file in the financial corpus as a LangChain document.

    Args:
        data_directory: Directory containing the source ``.txt`` files.

    Returns:
        A list of documents containing the file text and source-path metadata.
    """
    loader = DirectoryLoader(
        str(data_directory),
        glob="*.txt",
        loader_cls=TextLoader,
    )
    return loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for retrieval.

    Args:
        documents: The source documents to split.

    Returns:
        A list of approximately 1,000-character chunks with 200 characters of
        overlap. Every chunk is prefixed with its source note's title and
        includes its starting and ending source line in metadata.
    """
    document_texts = {
        str(document.metadata["source"]): document.page_content for document in documents
    }
    document_titles = {
        str(document.metadata["source"]): get_document_title(document) for document in documents
    }
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        source = str(chunk.metadata["source"])
        source_text = document_texts[source]
        chunk_title = document_titles[source]
        start_index = chunk.metadata["start_index"]
        end_index = start_index + len(chunk.page_content)
        chunk.metadata["start_line"] = source_text.count("\n", 0, start_index) + 1
        chunk.metadata["end_line"] = source_text.count("\n", 0, max(start_index, end_index - 1)) + 1
        chunk.metadata["title"] = chunk_title
        if chunk_title:
            chunk.page_content = f"{chunk_title}\n\n{chunk.page_content}"

    return chunks


def create_vector_store(chunks: list[Document]) -> FAISS:
    """Embed document chunks and store their vectors in an in-memory FAISS index.

    Args:
        chunks: The document chunks to embed and index.

    Returns:
        A FAISS vector store containing the chunks and their embeddings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings, normalize_L2=True)
