"""
Document Ingest - Smart chunking and embedding pipeline for RAG.
Qdrant primary, Chroma fallback. Multi-tenant via client_id metadata.
"""
import logging
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import io

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

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
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
        from app.config import settings
        _doc_ingestor = DocumentIngestor(
            qdrant_url=settings.vectordb.qdrant_url,
            collection_name=settings.vectordb.qdrant_collection,
            chroma_persist_dir=settings.vectordb.chroma_persist_dir,
            embedding_model=settings.vectordb.embedding_model
        )
    
    return _doc_ingestor


__all__ = ["DocumentIngestor", "SmartChunker", "get_document_ingestor"]
