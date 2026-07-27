"""Assembly point for the LangChain RAG pipeline."""


def create_rag_chain():
    """Connect retrieval, prompt construction, and LLM generation."""
    raise NotImplementedError("Implement RAG chain construction.")


def run_pipeline(question: str):
    """Run the RAG pipeline for a financial research question."""
    raise NotImplementedError("Implement pipeline execution.")
