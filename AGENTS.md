# RAG-langchain

## Purpose

This project is a small LangChain and FAISS implementation of the Financial Research Assistant
defined in `docs/SPEC.md`.

## Scope

- Keep the implementation small and focused.
- Follow the architecture in `docs/SPEC.md`.
- Do not add production features, a user interface, APIs, agents, memory, evaluation, reranking,
  hybrid retrieval, or advanced query strategies unless explicitly requested.
- Use existing dependencies before adding new ones.

## Conventions

- Use Python 3.13.9 and uv.
- Use Ruff with a 100-character line length.
- Prefer straightforward, readable code over abstractions.
- Keep the four implementation modules focused on their named responsibilities.
