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
import subprocess
import time

# Load .env at module initialization (before accessing os.environ)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

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

# Embedding model - SentenceTransformers (works with Windows DLL fix)
# Tested and verified working: test_embeddings.py passes all tests
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
    Qdrant primary (cloud or local), Chroma fallback.
    
    Features:
    - Reads Qdrant config from .env (VECTOR_DB_QDRANT_HOST, VECTOR_DB_QDRANT_API_KEY)
    - Uses stronger embedding model (all-mpnet-base-v2 or configurable)
    - CUDA/GPU acceleration when available
    - Production-grade error handling
    """
    
    # Production-grade embedding models ranked by quality
    # all-mpnet-base-v2: Best quality, 768 dim, slower
    # all-MiniLM-L12-v2: Good balance, 384 dim
    # all-MiniLM-L6-v2: Fast, 384 dim, lower quality
    EMBEDDING_MODELS = {
        "all-mpnet-base-v2": {"dim": 768, "quality": "best"},
        "all-MiniLM-L12-v2": {"dim": 384, "quality": "balanced"},
        "all-MiniLM-L6-v2": {"dim": 384, "quality": "fast"},
        "paraphrase-multilingual-mpnet-base-v2": {"dim": 768, "quality": "multilingual"},
    }


    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = None,
        chroma_persist_dir: str = None,
        embedding_model: str = None,
        qdrant_api_key: str = None
    ):
        # Load configuration from environment with sensible defaults
        # If passed URL is localhost (default in settings), try to find a better one in env
        env_url = os.environ.get("VECTOR_DB_QDRANT_HOST")
        if qdrant_url == "http://localhost:6333" and env_url:
             self.qdrant_url = env_url
        else:
             self.qdrant_url = qdrant_url or env_url or "http://localhost:6333"

        self.qdrant_api_key = qdrant_api_key or os.environ.get("VECTOR_DB_QDRANT_API_KEY")
        # Prioritize ENV var for collection to allow easy override (fixes dimension mismatch issues)
        self.collection_name = os.environ.get("QDRANT_COLLECTION") or collection_name or "AI-CA"
        
        # Override vector DB type from env - fallback is CHROMA not FAISS
        self.vector_db_type = os.environ.get("VECTOR_DB_TYPE", "chroma").lower()
        if self.vector_db_type == "faiss":
            logger.warning("FAISS configured but deprecated. Falling back to ChromaDB for consistency.")
            self.vector_db_type = "chroma"

        # Use stronger embedding model by default (all-mpnet-base-v2)
        default_model = "all-mpnet-base-v2"
        self._embedding_model_name = embedding_model or os.environ.get("EMBEDDING_MODEL", default_model)
        
        self.chunker = SmartChunker()
        
        # Initialize embedding model handling (CUDA/CPU)
        # ... (embedding init logic matches previous state) ...
        self._embedder = None
        self._embedder_type = None
        self._embedding_dim = self.EMBEDDING_MODELS.get(self._embedding_model_name, {}).get("dim", 768)
        self._device = None
        
        # Detect CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self._device = "cpu"
                logger.info("Using CPU for embeddings (CUDA not available)")
        except ImportError:
            self._device = "cpu"
            logger.debug("PyTorch not available, defaulting to CPU")
        
        # 1. Try SentenceTransformers first
        if SENTENCE_TRANSFORMERS_AVAILABLE and not self._embedder:
            try:
                # Try strongest model first
                models_to_try = [
                    self._embedding_model_name,
                    "all-mpnet-base-v2", 
                    "all-MiniLM-L12-v2", 
                    "all-MiniLM-L6-v2"
                ]
                
                for model_name in models_to_try:
                    try:
                        self._embedder = SentenceTransformer(model_name, device=self._device)
                        self._embedder_type = "sentence_transformers"
                        self._embedding_dim = self._embedder.get_sentence_embedding_dimension()
                        self._embedding_model_name = model_name
                        logger.info(f"✅ Using SentenceTransformers: {model_name} (dim={self._embedding_dim}, device={self._device})")
                        break
                    except Exception as model_err:
                        logger.debug(f"Model {model_name} failed: {model_err}")
                        continue
            except Exception as e:
                logger.warning(f"SentenceTransformers failed: {e}")
        
        # 2-4. Embedder Fallbacks (Ollama, HF, OpenAI) - kept as is
        if not self._embedder:
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/version", timeout=2)
                if resp.status_code == 200:
                    from langchain_community.embeddings import OllamaEmbeddings
                    self._embedder = OllamaEmbeddings(model="nomic-embed-text")
                    self._embedder_type = "ollama"
                    self._embedding_dim = 768
                    logger.info("Using Ollama embeddings (nomic-embed-text)")
            except Exception as e:
                logger.debug(f"Ollama embeddings unavailable: {e}")

        hf_api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_API_KEY")
        if hf_api_key and not self._embedder:
            try:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                self._embedder = HuggingFaceInferenceAPIEmbeddings(
                    api_key=hf_api_key,
                    model_name=f"sentence-transformers/{self._embedding_model_name}"
                )
                self._embedder_type = "huggingface_api"
                logger.info(f"Using HuggingFace Inference API")
            except Exception:
                pass

        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and not self._embedder:
             try:
                from langchain_openai import OpenAIEmbeddings
                self._embedder = OpenAIEmbeddings()
                self._embedder_type = "openai"
                self._embedding_dim = 1536
                logger.info("Using OpenAI embeddings")
             except Exception:
                pass
        
        if self._embedder:
            logger.info(f"📊 Embedding provider: {self._embedder_type} (dim={self._embedding_dim})")
        else:
            logger.warning("⚠️ No embedding provider available. Using hash-based fallback.")

        # Initialize vector store
        self._qdrant = None
        self._chroma = None
        self._active_store = None
        
        # Logic to choose DB based on env
        if self.vector_db_type == "qdrant" and QDRANT_AVAILABLE:
            try:
                # Docker logic for local
                if "localhost" in self.qdrant_url or "127.0.0.1" in self.qdrant_url:
                    self._ensure_local_qdrant_running()

                # Handle Qdrant Cloud vs Local
                if self.qdrant_api_key:
                    logger.info(f"Connecting to Qdrant Cloud/Auth: {self.qdrant_url}")
                    self._qdrant = QdrantClient(
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        timeout=int(os.environ.get("QDRANT_HEALTH_CHECK_TIMEOUT", 10))
                    )
                else:
                    # Local Qdrant (no auth)
                    logger.info(f"Connecting to local Qdrant: {self.qdrant_url}")
                    self._qdrant = QdrantClient(url=self.qdrant_url, timeout=5)
                
                self._qdrant.get_collections()
                self._ensure_qdrant_collection()
                self._active_store = "qdrant"
                logger.info(f"✅ Connected to Qdrant ({self.collection_name})")
            except Exception as e:
                logger.warning(f"Qdrant unavailable: {e}. Falling back to Chroma.")
                self._qdrant = None

        # Fallback to Chroma (Default)
        if self._active_store is None and CHROMA_AVAILABLE:
            try:
                persist_dir = chroma_persist_dir or str(Path.cwd() / "data" / "chroma")
                self._chroma = chromadb.PersistentClient(path=persist_dir)
                self._chroma_collection = self._chroma.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self._active_store = "chroma"
                logger.info("✅ Connected to Chroma (Fallback)")
            except Exception as e:
                logger.warning(f"Chroma unavailable: {e}")

        if self._active_store is None:
            logger.warning("⚠️ No vector store available")

    def _ensure_local_qdrant_running(self):
        """Check if local Qdrant is running, if not, pull and start via Docker."""
        try:
            import requests
            # Check if running
            try:
                requests.get(self.qdrant_url.replace("tcp://", "http://"), timeout=1)
                return  # Running
            except:
                pass # Not running
            
            logger.info("⚠️ Local Qdrant not detected. Attempting to start via Docker...")
            
            # Check for Docker
            subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Pull Qdrant
            logger.info("🐳 Pulling qdrant/qdrant...")
            subprocess.run(["docker", "pull", "qdrant/qdrant"], check=True)
            
            # Run Qdrant
            logger.info("🚀 Starting Qdrant container...")
            subprocess.run([
                "docker", "run", "-d", 
                "-p", "6333:6333", 
                "-v", "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant"
            ], check=True)
            
            # Wait for startup
            logger.info("Waiting for Qdrant startup...")
            for _ in range(10):
                try:
                    requests.get(self.qdrant_url.replace("tcp://", "http://"), timeout=1)
                    logger.info("✅ Qdrant started successfully via Docker")
                    return
                except:
                    time.sleep(2)
        except Exception as e:
            logger.warning(f"Failed to auto-start local Qdrant: {e}. Ensure Docker is running.")

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
        """Generate embedding for text using available provider."""
        if self._embedder is not None:
            try:
                # LangChain embeddings use embed_query
                if self._embedder_type in ("huggingface_api", "ollama", "openai"):
                    embedding = self._embedder.embed_query(text)
                    return embedding
                # SentenceTransformers uses encode
                elif self._embedder_type == "sentence_transformers":
                    embedding = self._embedder.encode(text, convert_to_numpy=True)
                    return embedding.tolist()
            except Exception as e:
                logger.error(f"Embedding generation failed ({self._embedder_type}): {e}")
        
        # Fallback: simple hash-based embedding when no model available
        # This provides basic semantic matching via word hashing
        words = text.lower().split()[:100]  # Use first 100 words
        embedding = [0.0] * self._embedding_dim
        
        for i, word in enumerate(words):
            # Hash word to embedding dimension
            word_hash = hash(word) % self._embedding_dim
            embedding[word_hash] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding

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

        # PRODUCTION FIX: Normalize client_id for VectorDB compatibility
        try:
            from app.core.id_generator import normalize_client_id, sanitize_metadata_value
            safe_client_id = normalize_client_id(client_id)
        except ImportError:
            # Fallback normalization
            safe_client_id = client_id.lower().replace(' ', '_').replace(':', '_') if client_id else "default"

        # Prepare metadata with sanitized values
        base_metadata = {}
        for k, v in (metadata or {}).items():
            # Skip list/dict values - convert to strings for VectorDB
            if isinstance(v, (list, tuple)):
                base_metadata[k] = ", ".join(str(x) for x in v)
            elif isinstance(v, dict):
                base_metadata[k] = str(v)
            else:
                base_metadata[k] = v
        
        base_metadata["client_id"] = safe_client_id  # Use normalized ID
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
            client_id: Filter by client ID (will be normalized)
            top_k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        if self._active_store is None:
            return []

        # PRODUCTION FIX: Normalize client_id to match ingestion normalization
        try:
            from app.core.id_generator import normalize_client_id
            safe_client_id = normalize_client_id(client_id)
        except ImportError:
            safe_client_id = client_id.lower().replace(' ', '_').replace(':', '_') if client_id else "default"

        query_embedding = self._generate_embedding(query)
        
        try:
            if self._active_store == "qdrant":
                results = self._qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter={
                        "must": [
                            {"key": "client_id", "match": {"value": safe_client_id}}
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
                    where={"client_id": safe_client_id}  # Use normalized ID
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
    """Get or create singleton document ingestor with .env config."""
    global _doc_ingestor
    
    if _doc_ingestor is None:
        try:
            from app.config import settings
            # Prefer env var for model if available, otherwise use settings
            env_model = os.environ.get("EMBEDDING_MODEL")
            
            _doc_ingestor = DocumentIngestor(
                qdrant_url=getattr(settings.vectordb, 'qdrant_url', None),
                collection_name=getattr(settings.vectordb, 'qdrant_collection', None),
                chroma_persist_dir=getattr(settings.vectordb, 'chroma_persist_dir', None),
                embedding_model=env_model or getattr(settings.vectordb, 'embedding_model', None),
                qdrant_api_key=getattr(settings.vectordb, 'qdrant_api_key', None)
            )
        except Exception as e:
            logger.debug(f"Using direct .env config for DocumentIngestor: {e}")
            # Fallback: DocumentIngestor reads from os.environ directly
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

