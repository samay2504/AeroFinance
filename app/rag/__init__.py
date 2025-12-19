"""
RAG Module - Retrieval-Augmented Generation with Vector Search.

Supports:
- Qdrant (primary) and ChromaDB (fallback) vector stores
- Smart document chunking
- Semantic search over ingested documents
- Document summarization
"""
from app.rag.ingest import (
    DocumentIngestor,
    SmartChunker,
    get_document_ingestor,
    RAGPipeline,
    get_rag_pipeline
)

__all__ = [
    "DocumentIngestor",
    "SmartChunker", 
    "get_document_ingestor",
    "RAGPipeline",
    "get_rag_pipeline"
]
