"""
Document Ingest - Smart chunking and embedding pipeline for RAG.
Qdrant primary, Chroma fallback. Multi-tenant via client_id metadata.
"""
import os
import logging
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import io

# Prevent transformers from loading torch which causes DLL issues on Windows
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Embedding model - commented out to avoid torch dependency
# To enable: uncomment and ensure torch is properly installed
# try:
#     from sentence_transformers import SentenceTransformer
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False


class SmartChunker:
    """
    Smart document chunker that handles:
    - Tables: CSV-style chunking with column context
    - Prose: Sentence/paragraph chunking
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def detect_content_type(self, text: str) -> str:
        """Detect if content is tabular or prose."""
        lines = text.strip().split("\n")
        
        # Check for common table indicators
        table_indicators = 0
        for line in lines[:10]:
            # Multiple columns separated by tabs/pipes
            if "\t" in line or "|" in line:
                table_indicators += 1
            # Numeric patterns suggesting tabular data
            if re.search(r"\d+\s+\d+\s+\d+", line):
                table_indicators += 1

        return "table" if table_indicators > 3 else "prose"

    def chunk_table(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """Chunk tabular data preserving row context."""
        lines = text.strip().split("\n")
        chunks = []
        
        # Assume first line is header
        header = lines[0] if lines else ""
        
        current_chunk = [header]
        current_size = len(header)
        
        for line in lines[1:]:
            if current_size + len(line) > self.chunk_size:
                if current_size >= self.min_chunk_size:
                    chunks.append({
                        "content": "\n".join(current_chunk),
                        "type": "table",
                        "context": context,
                        "header": header
                    })
                current_chunk = [header, line]  # Keep header for context
                current_size = len(header) + len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
        
        # Last chunk
        if current_chunk and len(current_chunk) > 1:
            chunks.append({
                "content": "\n".join(current_chunk),
                "type": "table",
                "context": context,
                "header": header
            })
        
        return chunks

    def chunk_prose(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """Chunk prose text by sentences/paragraphs."""
        chunks = []
        
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) > self.chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "type": "prose",
                        "context": context
                    })
                
                # Overlap: take last portion
                overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                current_chunk = overlap_text + " " + para
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip()
        
        # Last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append({
                "content": current_chunk.strip(),
                "type": "prose",
                "context": context
            })
        
        return chunks

    def chunk_document(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """Smart chunk document based on content type."""
        content_type = self.detect_content_type(text)
        
        if content_type == "table":
            return self.chunk_table(text, context)
        else:
            return self.chunk_prose(text, context)


class DocumentIngestor:
    """
    Document ingestion pipeline with vector storage.
    Qdrant primary, Chroma fallback.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "ai_ca_docs",
        chroma_persist_dir: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.chunker = SmartChunker()
        
        # Initialize embedding model
        self._embedder = None
        self._embedding_dim = 384  # Default for MiniLM
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(embedding_model)
                self._embedding_dim = self._embedder.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {embedding_model} (dim={self._embedding_dim})")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")

        # Initialize vector store
        self._qdrant = None
        self._chroma = None
        self._active_store = None
        
        # Try Qdrant first
        if QDRANT_AVAILABLE:
            try:
                self._qdrant = QdrantClient(url=qdrant_url, timeout=5)
                self._qdrant.get_collections()
                self._ensure_qdrant_collection()
                self._active_store = "qdrant"
                logger.info("Connected to Qdrant")
            except Exception as e:
                logger.warning(f"Qdrant unavailable: {e}")
                self._qdrant = None

        # Fallback to Chroma
        if self._active_store is None and CHROMA_AVAILABLE:
            try:
                persist_dir = chroma_persist_dir or str(Path.cwd() / "data" / "chroma")
                self._chroma = chromadb.PersistentClient(path=persist_dir)
                self._chroma_collection = self._chroma.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self._active_store = "chroma"
                logger.info("Connected to Chroma")
            except Exception as e:
                logger.warning(f"Chroma unavailable: {e}")

        if self._active_store is None:
            logger.warning("No vector store available")

    def _ensure_qdrant_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            self._qdrant.get_collection(self.collection_name)
        except Exception:
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._embedder is None:
            # Return zero vector as fallback
            return [0.0] * self._embedding_dim
        
        try:
            embedding = self._embedder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * self._embedding_dim

    def _generate_id(self, content: str, metadata: Dict) -> str:
        """Generate unique ID for chunk."""
        data = f"{content}:{metadata.get('client_id')}:{metadata.get('dataset_id')}"
        return hashlib.md5(data.encode()).hexdigest()

    def ingest_text(
        self,
        text: str,
        client_id: str,
        dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest text document into vector store.
        
        Args:
            text: Document text
            client_id: Client/tenant ID
            dataset_id: Dataset identifier
            metadata: Additional metadata
            
        Returns:
            Ingestion result with chunk count
        """
        if self._active_store is None:
            return {
                "success": False,
                "error": "No vector store available"
            }

        # Chunk document
        context = f"Document: {dataset_id}"
        chunks = self.chunker.chunk_document(text, context)
        
        if not chunks:
            return {
                "success": False,
                "error": "No chunks generated"
            }

        # Prepare metadata
        base_metadata = metadata or {}
        base_metadata["client_id"] = client_id
        base_metadata["dataset_id"] = dataset_id

        # Ingest chunks
        ingested = 0
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "chunk_type": chunk.get("type", "unknown"),
                "context": chunk.get("context", "")
            }
            
            content = chunk["content"]
            embedding = self._generate_embedding(content)
            chunk_id = self._generate_id(content, chunk_metadata)
            
            try:
                if self._active_store == "qdrant":
                    self._qdrant.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=chunk_id,
                                vector=embedding,
                                payload={"content": content, **chunk_metadata}
                            )
                        ]
                    )
                elif self._active_store == "chroma":
                    self._chroma_collection.upsert(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[chunk_metadata]
                    )
                ingested += 1
            except Exception as e:
                logger.error(f"Failed to ingest chunk {i}: {e}")
                continue

        return {
            "success": True,
            "chunks_ingested": ingested,
            "total_chunks": len(chunks),
            "store": self._active_store,
            "client_id": client_id,
            "dataset_id": dataset_id
        }

    def search(
        self,
        query: str,
        client_id: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            client_id: Filter by client ID
            top_k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        if self._active_store is None:
            return []

        query_embedding = self._generate_embedding(query)
        
        try:
            if self._active_store == "qdrant":
                results = self._qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter={
                        "must": [
                            {"key": "client_id", "match": {"value": client_id}}
                        ]
                    }
                )
                
                return [
                    {
                        "content": r.payload.get("content", ""),
                        "score": r.score,
                        "metadata": {k: v for k, v in r.payload.items() if k != "content"}
                    }
                    for r in results
                    if r.score >= score_threshold
                ]

            elif self._active_store == "chroma":
                results = self._chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where={"client_id": client_id}
                )
                
                docs = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                
                return [
                    {
                        "content": doc,
                        "score": 1 - dist,  # Convert distance to similarity
                        "metadata": meta
                    }
                    for doc, dist, meta in zip(docs, distances, metadatas)
                    if 1 - dist >= score_threshold
                ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        return []


# Singleton instance
_doc_ingestor: Optional[DocumentIngestor] = None


def get_document_ingestor() -> DocumentIngestor:
    """Get or create singleton document ingestor."""
    global _doc_ingestor
    
    if _doc_ingestor is None:
        try:
            from app.config import settings
            _doc_ingestor = DocumentIngestor(
                qdrant_url=settings.vectordb.qdrant_url,
                collection_name=settings.vectordb.qdrant_collection,
                chroma_persist_dir=settings.vectordb.chroma_persist_dir,
                embedding_model=settings.vectordb.embedding_model
            )
        except Exception as e:
            logger.warning(f"Using default config for DocumentIngestor: {e}")
            _doc_ingestor = DocumentIngestor()
    
    return _doc_ingestor


# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for document Q&A.
    
    Features:
    - Semantic search over ingested documents
    - Context assembly for LLM prompting  
    - Token-efficient context compression
    """
    
    def __init__(self, llm_wrapper=None):
        self._ingestor = get_document_ingestor()
        self._llm = llm_wrapper
        self._max_context_tokens = 2000  # Token budget for context
    
    @property
    def is_available(self) -> bool:
        """Check if RAG pipeline is operational."""
        return self._ingestor._active_store is not None
    
    def ingest_document(
        self,
        text: str,
        client_id: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest document into vector store."""
        return self._ingestor.ingest_text(text, client_id, doc_id, metadata)
    
    def query(
        self,
        question: str,
        client_id: str,
        top_k: int = 3,
        score_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        Query documents and optionally generate answer.
        
        Returns relevant chunks. If LLM is available, also generates answer.
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "RAG pipeline not available - no vector store configured",
                "contexts": []
            }
        
        # Retrieve relevant chunks
        results = self._ingestor.search(
            query=question,
            client_id=client_id,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        if not results:
            return {
                "success": True,
                "answer": "No relevant documents found for your query.",
                "contexts": [],
                "method": "rag:no_match"
            }
        
        # Assemble context with token budget
        context_parts = []
        total_chars = 0
        char_budget = self._max_context_tokens * 4  # ~4 chars per token
        
        for r in results:
            content = r.get("content", "")
            if total_chars + len(content) <= char_budget:
                context_parts.append({
                    "text": content,
                    "score": r.get("score", 0),
                    "source": r.get("metadata", {}).get("dataset_id", "unknown")
                })
                total_chars += len(content)
        
        # If LLM available, generate answer
        answer = None
        if self._llm and context_parts:
            try:
                context_text = "\n\n---\n\n".join([c["text"] for c in context_parts])
                prompt = f"""Based on the following document excerpts, answer the question.

DOCUMENT EXCERPTS:
{context_text}

QUESTION: {question}

Provide a concise, accurate answer based only on the information provided. If the answer is not in the documents, say so."""

                response = self._llm.invoke(prompt)
                answer = str(response.content) if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}")
                answer = f"Found {len(context_parts)} relevant document sections."
        else:
            answer = f"Found {len(context_parts)} relevant document sections."
        
        return {
            "success": True,
            "answer": answer,
            "contexts": context_parts,
            "num_contexts": len(context_parts),
            "method": "rag:semantic_search"
        }
    
    def summarize_document(
        self,
        client_id: str,
        doc_id: Optional[str] = None,
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a summary of ingested documents.
        
        Args:
            client_id: Client ID to filter documents
            doc_id: Optional specific document ID to summarize
            max_chunks: Maximum chunks to include in summary context
            
        Returns:
            Summary of the document(s)
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "RAG pipeline not available - no vector store configured",
                "summary": None
            }
        
        # Use a generic query to retrieve document content
        query = "summarize main topics key points financial data"
        
        results = self._ingestor.search(
            query=query,
            client_id=client_id,
            top_k=max_chunks,
            score_threshold=0.1  # Lower threshold to get more content for summarization
        )
        
        if not results:
            return {
                "success": True,
                "summary": "No documents found for this client.",
                "chunks_used": 0,
                "method": "rag:no_content"
            }
        
        # Filter by specific doc_id if provided
        if doc_id:
            results = [r for r in results if r.get("metadata", {}).get("dataset_id") == doc_id]
        
        if not results:
            return {
                "success": True,
                "summary": f"No content found for document: {doc_id}",
                "chunks_used": 0,
                "method": "rag:no_match"
            }
        
        # Assemble content for summarization
        content_parts = []
        total_chars = 0
        char_budget = 8000  # ~2000 tokens
        
        for r in results:
            content = r.get("content", "")
            if total_chars + len(content) <= char_budget:
                content_parts.append(content)
                total_chars += len(content)
        
        combined_content = "\n\n".join(content_parts)
        
        # Generate summary with LLM
        if self._llm:
            try:
                prompt = f"""Summarize the following document content. Provide a concise overview highlighting:
1. Main topics and themes
2. Key data points and figures
3. Important conclusions or insights

DOCUMENT CONTENT:
{combined_content}

SUMMARY:"""
                
                response = self._llm.invoke(prompt)
                summary = str(response.content) if hasattr(response, 'content') else str(response)
                
                return {
                    "success": True,
                    "summary": summary,
                    "chunks_used": len(content_parts),
                    "total_chars": total_chars,
                    "method": "rag:llm_summary"
                }
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
        
        # Fallback: return first few chunks as preview
        preview = combined_content[:500] + "..." if len(combined_content) > 500 else combined_content
        return {
            "success": True,
            "summary": f"Document preview ({len(content_parts)} chunks):\n{preview}",
            "chunks_used": len(content_parts),
            "method": "rag:preview"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG pipeline status and statistics."""
        return {
            "available": self.is_available,
            "vector_store": self._ingestor._active_store,
            "embedding_model": getattr(self._ingestor, '_embedding_model', 'unknown'),
            "collection": self._ingestor.collection_name,
            "llm_available": self._llm is not None
        }


# Singleton RAG pipeline
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline(llm_wrapper=None) -> RAGPipeline:
    """Get or create singleton RAG pipeline."""
    global _rag_pipeline
    
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(llm_wrapper)
    elif llm_wrapper and _rag_pipeline._llm is None:
        _rag_pipeline._llm = llm_wrapper
    
    return _rag_pipeline


__all__ = [
    "DocumentIngestor", 
    "SmartChunker", 
    "get_document_ingestor",
    "RAGPipeline",
    "get_rag_pipeline"
]

