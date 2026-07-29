"""Prompt-template construction for retrieved financial context."""

from langchain_core.prompts import ChatPromptTemplate


def create_prompt_template() -> ChatPromptTemplate:
    """Create the prompt used to answer questions from retrieved financial context.

    Returns:
        A chat prompt template that accepts ``context`` and ``question`` values.
    """
    return ChatPromptTemplate.from_template(
        """You are a financial research assistant. Answer the question using only the
provided context. If the context does not contain enough information, say so clearly.

Context:
{context}

Question:
{question}

Provide a concise investment-style response with 5 to 10 bullet points. Add a short
"Risk notes" section when relevant."""
    )
