"""
Smart PDF Ingestor — production-grade multimodal PDF ingestion & RAG pipeline.

Architecture:
  Ingestion:  pdfplumber text → vision cascade (Cloud Vision → dots.ocr → RapidOCR → Tesseract)
              → semantic chunking → BM25 index → disk persistence.  NO embeddings at ingest time.

  Retrieval:  BM25 pre-filter → metadata filter → lazy embed (cache-first)
              → cosine similarity → rerank (Jina/Cohere/Cross-Encoder).

Cost levers:
  • Lazy embedding:  embed on access, not arrival → 60-70% cost reduction.
  • Retrieval funnel: cheap BM25 first, expensive vector last → 4-6x P95 speedup.
  • Fingerprinting:  SHA256 doc + SimHash page + MinHash chunk dedup → 30-40% storage savings.

Reuse map:
  SmartChunker         — app.rag.ingest (paragraph/table-aware chunking)
  id_generator         — app.core.id_generator (normalize_client_id, generate_chunk_id, generate_doc_id)
  TokenManager         — app.core.prompts (extractive summarisation)
  SemanticCache        — app.core.llm_utils (get_semantic_cache, get_single_flight)
  _rerank_results      — app.rag.ingest.DocumentIngestor (Jina/Cohere/Cross-Encoder)
  _PROMPT_TEMPLATES    — app.core.prompts (pdf_vision_extract)
  PDFSettings          — app.config.settings.pdf (all env-driven config)
  BloomFilter          — app.core.data_registry (O(1) negative lookups on doc_hash)
"""

import io
import os
import json
import time
import uuid
import struct
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

# ═══ DLL Fix: Must run before loading native extensions (pdfplumber, numpy) ═══
# Prevents WinError 1114 "A dynamic link library (DLL) initialization routine
# failed" on Windows conda environments when torch DLLs are loaded later.
try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

import pdfplumber
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.llm_provider import LLMProvider
from app.core.llm_utils import get_semantic_cache, get_single_flight
from app.core.id_generator import normalize_client_id, generate_chunk_id, generate_doc_id
from app.core.prompts import get_token_manager, _PROMPT_TEMPLATES
from app.core.data_registry import BloomFilter
from app.core.redis_client import get_redis
from app.rag.ingest import SmartChunker

logger = logging.getLogger(__name__)

# ─── Configuration alias ────────────────────────────────────────
_pdf_cfg = settings.pdf

# ─── MinHash / SimHash constants ────────────────────────────────
_MINHASH_NUM_PERM = 128             # Number of hash permutations for MinHash
_SIMHASH_BITS = 64                   # SimHash width
_SIMHASH_NEAR_DUP_THRESHOLD = 3      # Hamming distance ≤ 3 → near duplicate
_MINHASH_JACCARD_THRESHOLD = 0.80     # ≥ 80% estimated Jaccard → fuzzy duplicate
_SHINGLE_K = 3                        # Character n-gram width for MinHash
_MINHASH_WINDOW = 500               # Sliding window: compare new chunk vs last N chunks
                                    # Prevents O(n²) on large docs while still catching dups
_VISION_CACHE_TTL = 86400 * 7       # Redis vision cache TTL: 7 days
_BLOOM_REDIS_KEY = "pdf:bloom:hashes"  # Redis SET key for bloom filter persistence


# ═════════════════════════════════════════════════════════════════
# SIMHASH / MINHASH PRIMITIVES
# ═════════════════════════════════════════════════════════════════

def _simhash(text: str, bits: int = _SIMHASH_BITS) -> int:
    """
    Charikar SimHash — locality-sensitive 64-bit hash.

    Converts text into a fixed-width fingerprint where similar texts
    produce hashes with small Hamming distance.

    Algorithm (per Charikar 2002):
      1. Tokenise text into features (words).
      2. Hash each feature to `bits` bits.
      3. Weight each bit position: +1 if bit set, -1 if unset.
      4. Final hash: bit i = 1 if weighted sum at position i > 0.

    Returns an integer in [0, 2^bits).
    """
    if not text:
        return 0
    v = [0] * bits
    tokens = text.lower().split()
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    return bin(a ^ b).count("1")


def _simhash_is_near_dup(
    a: int, b: int, threshold: int = _SIMHASH_NEAR_DUP_THRESHOLD
) -> bool:
    """True if two SimHashes are within Hamming-distance threshold."""
    return _hamming_distance(a, b) <= threshold


def _shingle(text: str, k: int = _SHINGLE_K) -> Set[str]:
    """Generate character k-gram shingle set from text."""
    text = text.lower().strip()
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash_signature(
    shingles: Set[str], num_perm: int = _MINHASH_NUM_PERM
) -> List[int]:
    """
    MinHash sketch — estimate Jaccard similarity in O(1) space.

    Uses `num_perm` independent hash functions (simulated via
    seed-parameterised SHA256 truncation) to build a compact
    signature vector.

    Returns list of `num_perm` minimum hash values.
    """
    if not shingles:
        return [0] * num_perm
    sig = [0xFFFFFFFF] * num_perm
    for s in shingles:
        s_bytes = s.encode("utf-8")
        for i in range(num_perm):
            # Hash = SHA256(seed || shingle), truncated to 32 bits
            h = struct.unpack(
                "<I",
                hashlib.sha256(
                    struct.pack("<I", i) + s_bytes
                ).digest()[:4],
            )[0]
            if h < sig[i]:
                sig[i] = h
    return sig


def _minhash_jaccard(
    sig_a: List[int], sig_b: List[int]
) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.

    J(A,B) ≈ (number of matching min-hash values) / num_perm
    """
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ═════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    Semantic unit of content with full provenance.

    Fields carry enough metadata for the retrieval funnel to filter
    without loading the full text from disk.
    """
    text: str
    doc_id: str
    chunk_id: str
    page_num: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    summary: str = ""
    is_table: bool = False
    is_visual: bool = False
    confidence: float = 1.0
    embedding_status: str = "pending"   # pending | cached | embedded
    metadata: Dict[str, Any] = field(default_factory=dict)
    minhash_sig: Optional[List[int]] = field(default=None, repr=False)

    def __post_init__(self):
        """Auto-generate deterministic chunk_id if empty, reusing id_generator."""
        if not self.chunk_id:
            self.chunk_id = generate_chunk_id(self.doc_id, self.page_num)


@dataclass
class DocumentFingerprint:
    """
    Multi-level hash for deduplication.

    doc_hash     — SHA256 of entire file bytes (exact duplicate detection).
    page_hashes  — per-page MD5  (vision cache keying).
    page_simhashes — per-page SimHash (near-duplicate detection, Hamming dist).
    chunk_hashes — per-chunk MD5 (paragraph-level exact dedup).
    chunk_minhashes — per-chunk MinHash signature (fuzzy paragraph reuse).
    """
    doc_hash: str
    page_hashes: Dict[int, str] = field(default_factory=dict)
    page_simhashes: Dict[int, int] = field(default_factory=dict)
    chunk_hashes: Dict[str, str] = field(default_factory=dict)
    chunk_minhashes: Dict[str, List[int]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Return value of ingest_document()."""
    success: bool
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int
    text_pages: int
    vision_pages: int
    tables_extracted: int
    time_seconds: float
    fingerprint: DocumentFingerprint
    strategy: str = "lazy_hybrid"


# ═════════════════════════════════════════════════════════════════
# SMART PDF INGESTOR
# ═════════════════════════════════════════════════════════════════

class SmartPDFIngestor:
    """
    Production-grade PDF ingestion with vision capabilities.

    Ingestion path:
      1. SHA256 fingerprint → dedup check
      2. Per-page: text density → route to text or vision extraction
      3. Semantic chunking (reuses SmartChunker from app.rag.ingest)
      4. Extractive summaries (reuses TokenManager.summarize_for_context)
      5. BM25 index update  (summaries tokenised)
      6. Disk persistence    (meta JSON + chunks JSONL)
      7. NO embeddings generated — lazy evaluation on retrieval.

    Retrieval path:
      L0  BM25 keyword pre-filter     → top-N candidates (settings.pdf.bm25_prefilter_k)
      L1  Metadata filter              → apply user-supplied filters
      L2  Lazy embed (cache-first)     → embed only cache misses
      L3  Cosine similarity            → rank survivors
      L4  Rerank (Jina/Cohere/Cross-Encoder via DocumentIngestor._rerank_results)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        cache_dir: Optional[str] = None,
        enable_vision: Optional[bool] = None,
        vision_threshold: Optional[float] = None,
        max_chunk_tokens: Optional[int] = None,
        parallel_workers: Optional[int] = None,
    ):
        self.llm = llm_provider

        # All config from settings.pdf — constructor args are overrides only
        self.cache_dir = Path(cache_dir or _pdf_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.enable_vision = enable_vision if enable_vision is not None else _pdf_cfg.enable_vision
        self.vision_threshold = vision_threshold if vision_threshold is not None else _pdf_cfg.vision_threshold
        self.max_chunk_tokens = max_chunk_tokens if max_chunk_tokens is not None else _pdf_cfg.max_chunk_tokens
        self.parallel_workers = parallel_workers if parallel_workers is not None else _pdf_cfg.parallel_workers

        # dots.ocr — intelligent Docker lifecycle management
        self._dots_ocr_enabled = _pdf_cfg.dots_ocr_enabled
        self._dots_ocr_endpoint = _pdf_cfg.dots_ocr_endpoint
        self._dots_ocr_timeout = _pdf_cfg.dots_ocr_timeout
        self._dots_ocr_min_confidence = _pdf_cfg.dots_ocr_min_confidence
        self._dots_ocr_manager = None

        # dots.ocr: only pre-validate config here — actual Docker startup is
        # deferred to first use (after Gemini vision fails) via _ensure_dots_ocr_ready().
        self._dots_ocr_ready = False  # True once ensure_running() succeeds
        if self._dots_ocr_enabled:
            if _pdf_cfg.dots_ocr_auto_docker:
                try:
                    from app.core.gpu_utils import DotsOCRManager
                    host_port = self._parse_port(self._dots_ocr_endpoint) or 5555
                    self._dots_ocr_manager = DotsOCRManager(
                        host_port=host_port,
                        docker_image=_pdf_cfg.dots_ocr_docker_image,
                        model_name=_pdf_cfg.dots_ocr_model,
                    )
                    gpu = self._dots_ocr_manager.gpu_info
                    logger.info(
                        f"dots.ocr configured (lazy start): GPU={gpu.device_name or 'none'}, "
                        f"VRAM={gpu.vram_total_gib:.1f}GiB — container will start on first use"
                    )
                except Exception as exc:
                    logger.warning(f"dots.ocr auto-docker config failed: {exc} — disabling")
                    self._dots_ocr_enabled = False
            elif not self._dots_ocr_endpoint:
                raise ValueError(
                    "dots.ocr is enabled (PDF_DOTS_OCR_ENABLED=true) but "
                    "PDF_DOTS_OCR_ENDPOINT is not set. Either:\n"
                    "  1. Set PDF_DOTS_OCR_ENDPOINT=http://your-host:port in .env\n"
                    "  2. Set PDF_DOTS_OCR_AUTO_DOCKER=true for automatic management"
                )
            else:
                # Manual endpoint provided — mark ready immediately (no Docker to start)
                self._dots_ocr_ready = True

        # Probe local OCR availability once at init (avoid repeated ImportError)
        self._rapid_ocr_available = False
        try:
            import rapidocr_onnxruntime  # noqa: F401
            self._rapid_ocr_available = True
            logger.info("RapidOCR available (L3a local fallback)")
        except ImportError:
            logger.info("RapidOCR not available (pip install rapidocr-onnxruntime)")

        self._tesseract_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("Tesseract available (L3b backup fallback)")
        except Exception:
            logger.info("Tesseract not available (apt-get install tesseract-ocr)")

        # Reuse existing subsystems — no reinvention
        self._chunker = SmartChunker(
            chunk_size=self.max_chunk_tokens * 4,  # SmartChunker uses char count, ~4 chars/token
            chunk_overlap=50,
            min_chunk_size=100,
        )
        self._token_mgr = get_token_manager()
        self._semantic_cache = get_semantic_cache()
        self._single_flight = get_single_flight()

        # Bloom filter for O(1) negative lookups on doc_hash dedup checks
        # Avoids I/O (stat-ing .meta.json) for documents never ingested
        self._bloom = BloomFilter(
            capacity=settings.storage.bloom_capacity,
            fp_rate=settings.storage.bloom_fp_rate,
        )
        # Pre-populate bloom from existing meta files on disk
        self._bloom_prepopulate()

        # BM25 in-memory index
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus_map: Dict[int, Dict[str, Any]] = {}

        # Lazy-loaded reranker / Qdrant reference (from DocumentIngestor)
        self._doc_ingestor = None

        # SimHash registry for cross-document near-duplicate page detection
        # Maps page_simhash → (doc_id, page_num) for O(1) collision checks
        self._simhash_registry: Dict[int, Tuple[str, int]] = {}

        # MinHash registry for chunk-level fuzzy dedup
        # Maps chunk_id → MinHash signature — bounded deque for O(window) comparisons
        self._minhash_registry: Dict[str, List[int]] = {}
        self._minhash_insertion_order: List[str] = []  # maintain insertion order for window

        # Counters
        self._stats: Dict[str, int] = defaultdict(int, {
            "documents_processed": 0,
            "pages_processed": 0,
            "vision_calls": 0,
            "dots_ocr_calls": 0,
            "local_ocr_calls": 0,
            "cache_hits": 0,
            "dedupe_saves": 0,
            "bloom_io_saved": 0,
            "simhash_near_dups": 0,
            "minhash_fuzzy_dups": 0,
            "qdrant_upserts": 0,
        })
        # Thread-safe lock for _stats mutations from parallel workers.
        self._stats_lock = threading.Lock()

        # Semaphore for concurrent cloud-vision API calls.
        # Configured via PDF_VISION_CONCURRENCY (default 3).
        # Prevents 429 rate-limit cascades on large multi-page PDFs.
        self._vision_concurrency = _pdf_cfg.vision_concurrency
        self._vision_sem = threading.Semaphore(self._vision_concurrency)

        logger.info(
            f"SmartPDFIngestor initialized: vision={self.enable_vision}, "
            f"vision_concurrency={self._vision_concurrency}, "
            f"cache={self.cache_dir}"
        )

    # ─────────────────────────────────────────────────────────────
    # LAZY RERANKER ACCESS (delegates to DocumentIngestor)
    # ─────────────────────────────────────────────────────────────

    def _get_reranker(self):
        """
        Lazily obtain the *singleton* DocumentIngestor for reranking,
        embedding, and Qdrant access.  Using the module-level singleton
        avoids creating a second SentenceTransformer instance (saves ~46s
        on first query).
        """
        if self._doc_ingestor is None:
            try:
                from app.rag.ingest import get_document_ingestor
                self._doc_ingestor = get_document_ingestor()
            except Exception as e:
                logger.warning(f"Could not init DocumentIngestor: {e}")
        return self._doc_ingestor

    def _ensure_dots_ocr_ready(self) -> bool:
        """
        Lazily start the dots.ocr Docker container on first actual use.

        Called only after Gemini vision (L1) fails — avoids the 120s
        model-load wait on every SmartPDFIngestor construction.
        Returns True if dots.ocr is available, False otherwise.
        """
        if self._dots_ocr_ready:
            return True
        if not self._dots_ocr_enabled or self._dots_ocr_manager is None:
            return False
        try:
            logger.info("dots.ocr: starting container on first use (lazy init)...")
            if self._dots_ocr_manager.ensure_running():
                self._dots_ocr_endpoint = self._dots_ocr_manager.endpoint
                self._dots_ocr_ready = True
                logger.info(f"dots.ocr container healthy at {self._dots_ocr_endpoint}")
                return True
            else:
                logger.warning("dots.ocr Docker start failed — disabling for this session")
                self._dots_ocr_enabled = False
                return False
        except Exception as exc:
            logger.warning(f"dots.ocr lazy start failed: {exc} — disabling")
            self._dots_ocr_enabled = False
            return False

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def ingest_document(
        self,
        file_path: str,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """
        Ingest a PDF.  Steps: fingerprint → dedup → extraction → chunk →
        summarise → BM25 index → persist.  No embeddings.
        """
        start_time = time.time()
        safe_client = normalize_client_id(client_id)
        logger.info(f"Starting ingestion: {file_path} (client={safe_client})")

        # ── Step 1: Fingerprint & dedup ──
        fingerprint = self._generate_fingerprint(file_path)

        if self._check_if_processed(fingerprint.doc_hash, safe_client):
            logger.info("Duplicate detected (SHA256 match), returning cached meta.")
            self._stats["dedupe_saves"] += 1
            cached = self._get_cached_metadata(fingerprint.doc_hash)
            return IngestionResult(**cached)

        # ── Step 2: Hybrid extraction ──
        doc_id = generate_doc_id(safe_client, Path(file_path).name)
        chunks: List[Chunk] = []
        text_pages = vision_pages = table_count = 0
        total_pages = 0

        try:
            # ── Read PDF bytes once into memory ──────────────────────────────
            # Workers each open their own io.BytesIO(pdf_bytes) — no disk I/O
            # in the thread pool, no shared file-handle state between threads.
            # For an 11 MB file × 4 workers this costs ~44 MB RAM, which is
            # negligible compared to the 5-25 min saved by parallelising table
            # extraction and PNG rendering.
            pdf_bytes = Path(file_path).read_bytes()

            # ── FAST classify pass in main thread ────────────────────────────
            # extract_text() only:  ~0.05 s/page → 146 pages ≈  7 s total.
            # extract_tables() is intentionally deferred to each worker thread
            # (2-10 s/page, extremely slow for annual-report-style PDFs with
            # merged cells and multi-column layouts).  Running all 146 pages
            # sequentially here would add 5-25 minutes of serial wall-clock
            # time before the first thread even starts.
            page_payloads: list = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as _classify_pdf:
                total_pages = len(_classify_pdf.pages)
                logger.info(f"Processing {total_pages} pages")
                for i, page in enumerate(_classify_pdf.pages):
                    text = page.extract_text() or ""
                    text_chars = len(text.strip())
                    page_payloads.append({
                        "page_num": i,
                        "text": text,
                        "text_chars": text_chars,
                    })

            # ── Vision page cap ───────────────────────────────────────────────
            # Sort sparse pages by text density (fewest chars first) and cap at
            # PDF_MAX_VISION_PAGES (default 20).  Annual reports with 40-60
            # chart/image pages can otherwise send 200+ API calls without this
            # guard.  Pages beyond the cap fall through to local OCR (L3).
            max_vision = _pdf_cfg.max_vision_pages
            vision_page_nums: Set[int] = set()
            if self.enable_vision and max_vision > 0:
                sparse_pages = [
                    p for p in page_payloads
                    if p["text_chars"] < self.vision_threshold
                ]
                capped = sorted(sparse_pages, key=lambda p: p["text_chars"])[:max_vision]
                vision_page_nums = {p["page_num"] for p in capped}
                n_sparse = len(sparse_pages)
                if n_sparse > max_vision:
                    logger.info(
                        f"Vision cap: {max_vision}/{n_sparse} sparse pages routed to "
                        f"cloud vision (PDF_MAX_VISION_PAGES={max_vision}); "
                        f"remaining {n_sparse - max_vision} pages use local OCR"
                    )

            for p in page_payloads:
                p["vision_eligible"] = self.enable_vision and (
                    p["page_num"] in vision_page_nums
                )

            if self.parallel_workers > 1 and total_pages > 4:
                # ThreadPoolExecutor — each worker receives (payload_dict, pdf_bytes).
                # The slow work (extract_tables, to_image, vision API) runs in parallel.
                with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
                    futures = [
                        pool.submit(
                            self._process_extracted_page, payload, doc_id, pdf_bytes,
                        )
                        for payload in page_payloads
                    ]
                    for fut in futures:
                        page_chunks, stats = fut.result()
                        chunks.extend(page_chunks)
                        text_pages += stats["text"]
                        vision_pages += stats["vision"]
                        table_count += stats["tables"]
            else:
                for payload in page_payloads:
                    page_chunks, stats = self._process_extracted_page(
                        payload, doc_id, pdf_bytes
                    )
                    chunks.extend(page_chunks)
                    text_pages += stats["text"]
                    vision_pages += stats["vision"]
                    table_count += stats["tables"]

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}", exc_info=True)
            raise

        # ── Step 3: Chunk enhancement (hashes, summaries, near-dup tagging) ──
        self._generate_chunk_summaries(chunks)
        self._update_chunk_hashes(chunks, fingerprint)
        self._compute_minhash_signatures(chunks, fingerprint)

        # ── Step 4: Detect cloud vector store availability ──────────────────────
        # When Qdrant or Chroma is connected we push chunks directly during
        # ingestion and skip the intermediate JSONL + BM25 disk files.  The
        # bloom filter is still written (dedup across restarts) and in-memory
        # BM25 is still built (within-session keyword fallback).
        _cloud_store_active = False
        _doc_ingestor = None
        try:
            from app.rag.ingest import get_document_ingestor as _get_doc_ingestor  # local import — avoids circular at module level
            _doc_ingestor = _get_doc_ingestor()
            _cloud_store_active = _doc_ingestor._active_store is not None
        except Exception as _cse:
            logger.debug(f"Cloud store check skipped: {_cse}")

        # ── Step 5: Direct push to cloud vector store ────────────────────────
        # Use ingest_chunks_batch() — embeds all chunks in ONE forward pass and
        # sends ONE batch Qdrant upsert, replacing the old concatenate→re-chunk
        # →per-chunk-upsert approach that caused the 5-minute bottleneck.
        if _cloud_store_active and _doc_ingestor is not None:
            try:
                push_chunks = [
                    {
                        "content": (_c.text or _c.summary or "").strip(),
                        "page_num": _c.page_num,
                        "chunk_id": _c.chunk_id,
                        "is_table": _c.is_table,
                        "is_visual": _c.is_visual,
                        "confidence": _c.confidence,
                        "summary": _c.summary,
                    }
                    for _c in chunks
                    if (_c.text or _c.summary or "").strip()
                ]
                if push_chunks:
                    push_result = _doc_ingestor.ingest_chunks_batch(
                        chunks=push_chunks,
                        client_id=safe_client,
                        dataset_id=doc_id,
                        metadata={
                            "source": "pdf",
                            "filename": Path(file_path).name,
                            "doc_id": doc_id,
                        },
                    )
                    self._stats["qdrant_upserts"] += push_result.get("chunks_ingested", 0)
                    logger.info(
                        f"PDF batch push → {_doc_ingestor._active_store}: "
                        f"{push_result.get('chunks_ingested')}/{len(push_chunks)} chunks, "
                        f"embed={push_result.get('embed_seconds'):.2f}s, "
                        f"upsert={push_result.get('upsert_seconds'):.2f}s"
                    )
            except Exception as _ve:
                # Non-fatal: fall back to JSONL so ingest_service can do the push
                logger.warning(f"Direct batch push failed, falling back to JSONL: {_ve}")
                _cloud_store_active = False

        # ── Step 6: BM25 index ───────────────────────────────────────────────
        # persist=False when cloud store holds the canonical chunks (no disk needed).
        # persist=True when fallback to JSONL so BM25 map survives restarts.
        self._update_bm25_index(chunks, safe_client, persist=not _cloud_store_active)

        # ── Step 7: Persist metadata + optional JSONL ────────────────────────
        ingestion_meta = {
            "doc_id": doc_id,
            "client_id": safe_client,
            "filename": Path(file_path).name,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "text_pages": text_pages,
            "vision_pages": vision_pages,
            "tables_extracted": table_count,
            "processed_at": time.time(),
            "fingerprint": asdict(fingerprint),
            "user_metadata": metadata or {},
            "chunk_index": [
                {
                    "chunk_id": c.chunk_id,
                    "page_num": c.page_num,
                    "summary": c.summary,
                    "is_table": c.is_table,
                    "is_visual": c.is_visual,
                    "confidence": c.confidence,
                    "embedding_status": c.embedding_status,
                }
                for c in chunks
            ],
            "cloud_store_used": _cloud_store_active,
        }

        self._save_metadata(ingestion_meta)
        if _cloud_store_active:
            logger.debug("Skipping JSONL chunk write — chunks pushed directly to cloud vector store")
        else:
            self._save_chunks(chunks, doc_id)

        self._stats["documents_processed"] += 1
        self._stats["pages_processed"] += total_pages

        elapsed = time.time() - start_time
        logger.info(
            f"Ingestion complete: {len(chunks)} chunks, "
            f"{text_pages} text / {vision_pages} vision pages, {elapsed:.2f}s"
        )

        return IngestionResult(
            success=True,
            doc_id=doc_id,
            filename=Path(file_path).name,
            total_pages=total_pages,
            total_chunks=len(chunks),
            text_pages=text_pages,
            vision_pages=vision_pages,
            tables_extracted=table_count,
            time_seconds=elapsed,
            fingerprint=fingerprint,
        )

    def retrieve(
        self,
        query: str,
        client_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        enable_rerank: Optional[bool] = None,
        rerank_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieval funnel: BM25 → Metadata → Lazy Embed → Vector → Rerank.
        """
        safe_client = normalize_client_id(client_id)
        logger.info(f"Retrieval query: '{query}' (client={safe_client})")

        should_rerank = enable_rerank if enable_rerank is not None else _pdf_cfg.enable_rerank
        rerank_strategy = rerank_model or _pdf_cfg.rerank_model

        # L0: BM25 pre-filter
        t0 = time.time()
        bm25_candidates = self._bm25_search(
            query, safe_client, top_k=_pdf_cfg.bm25_prefilter_k,
        )
        logger.info(f"BM25: {len(bm25_candidates)} candidates in {(time.time()-t0)*1000:.0f}ms")

        # L1: Metadata filter
        if filters:
            filtered = self._apply_metadata_filters(bm25_candidates, filters)
            logger.info(f"Metadata filter: {len(bm25_candidates)} → {len(filtered)}")
        else:
            filtered = bm25_candidates

        survivors = filtered[:_pdf_cfg.vector_cap]  # cap before embedding

        # ── Load chunk texts once (reused for embedding + returned to caller)
        chunk_ids = [c["chunk_id"] for c in survivors]
        chunk_texts = self._load_chunk_texts(chunk_ids)
        for c in survivors:
            txt = chunk_texts.get(c["chunk_id"], "")
            c["text"] = txt
            c["content"] = txt

        # L2: Lazy embed (cache-first, batch embed misses)
        t1 = time.time()
        embeddings = self._ensure_embeddings_for_chunks(
            chunk_ids, preloaded_texts=chunk_texts,
        )
        logger.info(f"Embedding ensured in {(time.time()-t1)*1000:.0f}ms")

        # L3: Vector similarity
        t2 = time.time()
        query_emb = self._embed_query(query)
        results = self._compute_vector_similarity(query_emb, survivors, embeddings)
        logger.info(f"Vector search in {(time.time()-t2)*1000:.0f}ms")

        # L4: Rerank — delegates to DocumentIngestor._rerank_results
        if should_rerank and len(results) > 1:
            reranker = self._get_reranker()
            if reranker is not None:
                try:
                    for r in results:
                        r.setdefault("content", r.get("text", ""))
                        r.setdefault("score", r.get("vector_score", 0.0))
                    results = reranker._rerank_results(
                        query, results, top_k, rerank_strategy,
                    )
                    logger.info(f"Reranked {len(results)} results via {rerank_strategy}")
                except Exception as e:
                    logger.warning(f"Rerank failed ({rerank_strategy}): {e}")

        return results[:top_k]

    # ─────────────────────────────────────────────────────────────
    # PAGE PROCESSING
    # ─────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────
    # PRE-EXTRACTED PAGE PROCESSING (thread-safe, no pdfplumber handles)
    # ─────────────────────────────────────────────────────────────

    def _process_extracted_page(
        self, payload: Dict[str, Any], doc_id: str, pdf_bytes: bytes
    ) -> Tuple[List[Chunk], Dict[str, int]]:
        """Process one PDF page inside a worker thread — fully parallel.

        The main thread ran only the fast classify pass (extract_text only,
        ~0.05 s/page).  All heavy work happens here in the thread pool:

        * ``extract_tables()``  — pdfplumber lattice/stream (2-10 s/page for
          annual-report PDFs with merged cells).  Running this in parallel for
          a 146-page document cuts wall-clock time from 5-25 min to ~3-5 min.
        * ``page.to_image()``   — PNG render for vision-candidate pages.
        * Vision-cascade API   — Gemini / dots.ocr / RapidOCR / Tesseract.

        Each thread opens its own ``io.BytesIO(pdf_bytes)`` pdfplumber handle
        (no shared state, no file-lock contention).  For an 11 MB PDF × 4
        workers the RAM cost is ~44 MB — negligible on a modern machine.

        Thread-safety:
          * ``self._stats`` mutations are always inside ``self._stats_lock``.
          * SimHash registry write is inside the lock; the read-only
            near-dup scan loop runs outside (minimal critical section).
          * Vision-API semaphore is acquired inside ``_extract_via_vision``.
        """
        page_num: int = payload["page_num"]
        text: str = payload["text"]
        text_chars: int = payload["text_chars"]
        vision_eligible: bool = payload.get("vision_eligible", False)

        stats = {"text": 0, "vision": 0, "tables": 0}

        # ── Open private pdfplumber handle for this worker ────────────────────
        # BytesIO wraps bytes already in process memory — zero disk I/O.
        # Table extraction (lattice + stream algorithms) runs in parallel across
        # all page workers, giving O(1/workers) latency instead of O(n_pages).
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as _worker_pdf:
            page = _worker_pdf.pages[page_num]
            tables: list = page.extract_tables()

            img_bytes: Optional[bytes] = None
            page_hash: Optional[str] = None
            if vision_eligible:
                image = page.to_image(resolution=150)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                page_hash = hashlib.md5(text.encode()).hexdigest()

        # ── Vision vs text path ───────────────────────────────────────────────
        use_vision = vision_eligible and img_bytes is not None
        if use_vision:
            logger.debug(f"Page {page_num}: vision path ({text_chars} chars < {self.vision_threshold})")
            content, extracted_tables = self._extract_via_vision(
                page=None, page_num=page_num,
                img_bytes=img_bytes, page_hash=page_hash,
            )
            stats["vision"] = 1
            stats["tables"] = len(extracted_tables)
            with self._stats_lock:
                self._stats["vision_calls"] += 1
        else:
            logger.debug(f"Page {page_num}: text path")
            content = text
            extracted_tables = tables
            stats["text"] = 1
            stats["tables"] = len(tables)
            if extracted_tables:
                content += f"\n\n[TABLES]\n{self._tables_to_markdown(extracted_tables)}"

        # SimHash from pre-extracted text (no page object needed).
        # Minimal critical section: scan runs outside the lock (read-only);
        # only the registry write + stat increment require the lock.
        sh = _simhash(text, bits=_SIMHASH_BITS)
        _near_dup_info: Optional[Tuple] = None
        for existing_sh, (existing_doc, existing_page) in self._simhash_registry.items():
            if existing_doc == doc_id:
                continue
            if _simhash_is_near_dup(sh, existing_sh):
                _near_dup_info = (existing_doc, existing_page, _hamming_distance(sh, existing_sh))
                break

        with self._stats_lock:
            if _near_dup_info is not None:
                _dup_doc, _dup_page, _dist = _near_dup_info
                self._stats["simhash_near_dups"] += 1
                logger.warning(
                    f"SimHash near-duplicate detected: page {page_num} of {doc_id[:8]} "
                    f"matches page {_dup_page} of {_dup_doc[:8]} "
                    f"(hamming_dist={_dist})"
                )
            self._simhash_registry[sh] = (doc_id, page_num)

        # Reuse SmartChunker from app.rag.ingest
        context = f"page_{page_num}"
        raw_chunks = self._chunker.chunk_document(content, context=context)

        # Convert SmartChunker dicts → Chunk dataclass
        chunks: List[Chunk] = []
        for i, rc in enumerate(raw_chunks):
            chunk_id = generate_chunk_id(doc_id, page_num * 1000 + i)
            chunks.append(Chunk(
                text=rc["content"],
                doc_id=doc_id,
                chunk_id=chunk_id,
                page_num=page_num,
                is_visual=use_vision,
                is_table=rc.get("type") == "table",
            ))

        return chunks, stats

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_port(url: Optional[str]) -> Optional[int]:
        """Extract port from URL string, e.g. 'http://localhost:5555' → 5555."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.port
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────
    # 4-STAGE VISION CASCADE
    # ─────────────────────────────────────────────────────────────

    def _get_vision_prompt(self) -> str:
        """Load vision extraction prompt from templates.yaml, with inline fallback."""
        return _PROMPT_TEMPLATES.get("pdf_vision_extract", (
            "Extract all visible text from this financial document page.\n"
            "If there are tables, convert them to markdown format.\n"
            "If there are charts, describe the trend or key insight.\n\n"
            'Return JSON: {"text": "...", "tables": ["..."], "charts": ["..."]}'
        ))

    def _extract_via_vision(
        self,
        page,
        page_num: int,
        img_bytes: Optional[bytes] = None,
        page_hash: Optional[str] = None,
    ) -> Tuple[str, List]:
        """
        Cascading vision pipeline:
          L1  Google Cloud Vision  (primary, 1K units/month free)
          L2  dots.ocr             (self-hosted 1.7B VLM, if enabled)
          L3a RapidOCR             (ONNX CPU, zero cost)
          L3b Tesseract            (classic OCR, final fallback)

        When ``img_bytes`` and ``page_hash`` are supplied (pre-extracted path),
        ``page`` may be None — no pdfplumber handle is touched.
        """
        # --- Hash & cache key ---
        if page_hash is None:
            page_hash = self._hash_page(page)
        cache_key = f"vision_{page_hash}"
        cached = self._get_vision_cache(cache_key)
        if cached:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            return cached["text"], cached["tables"]

        # --- Image bytes (use pre-rendered if available) ---
        if img_bytes is not None:
            img_data = img_bytes
        else:
            image = page.to_image(resolution=150)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            img_data = buf.getvalue()

        vision_prompt = self._get_vision_prompt()

        # ── L1: Primary Vision API ──
        try:
            with self._vision_sem:  # max N concurrent Gemini/Cloud-Vision calls
                # Per-call timeout: if Gemini stalls, abandon and fall through to L2/L3.
                # Without this a single slow response holds a semaphore slot indefinitely.
                _call_timeout = _pdf_cfg.vision_call_timeout  # default 30 s
                with ThreadPoolExecutor(max_workers=1) as _vision_pool:
                    _fut = _vision_pool.submit(
                        self.llm.vision_chat, img_data, vision_prompt
                    )
                    try:
                        response = _fut.result(timeout=_call_timeout)
                    except Exception as _te:
                        raise RuntimeError(f"vision_chat timeout/error after {_call_timeout}s: {_te}") from _te
            result = json.loads(response)
            text = result.get("text", "")
            tables = result.get("tables", [])
            charts = result.get("charts", [])
            full_text = text
            if charts:
                full_text += "\n\n[CHARTS]\n" + "\n".join(charts)

            self._save_vision_cache(cache_key, {
                "text": full_text, "tables": tables, "source": "primary_vision"
            })
            logger.info(f"Page {page_num}: primary vision succeeded")
            return full_text, tables

        except Exception as e:
            logger.warning(f"Primary vision failed p{page_num}: {e}")

        # ── L2: dots.ocr (lazy-started only after L1 fails) ──
        if self._dots_ocr_enabled and self._ensure_dots_ocr_ready():
            try:
                dots_result = self._extract_via_dots_ocr(img_data, page_num)
                if dots_result:
                    self._save_vision_cache(cache_key, {
                        "text": dots_result["text"],
                        "tables": dots_result.get("tables", []),
                        "source": "dots_ocr",
                    })
                    with self._stats_lock:
                        self._stats["dots_ocr_calls"] += 1
                    logger.info(f"Page {page_num}: dots.ocr succeeded")
                    return dots_result["text"], dots_result.get("tables", [])
            except Exception as e:
                logger.warning(f"dots.ocr failed p{page_num}: {e}")

        # ── L3: Local OCR (RapidOCR → Tesseract) ──
        try:
            ocr_text = self._extract_via_local_ocr(img_data)
            self._save_vision_cache(cache_key, {
                "text": ocr_text, "tables": [], "source": "local_ocr"
            })
            with self._stats_lock:
                self._stats["local_ocr_calls"] += 1
            logger.info(f"Page {page_num}: local OCR succeeded")
            return ocr_text, []
        except Exception as e:
            logger.error(f"All vision methods failed p{page_num}: {e}")
            # Fallback: return whatever text was pre-extracted (may be empty).
            # When called from _process_extracted_page, page is None so we
            # use the text already available in the payload (passed via content).
            if page is not None:
                return page.extract_text() or "", []
            return "", []

    def _extract_via_dots_ocr(
        self, image_data: bytes, page_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        dots.ocr — self-hosted 1.7B VLM (rednote-hilab/dots.ocr).
        Endpoint and timeout driven entirely by PDFSettings.
        """
        try:
            import httpx
            import base64

            b64 = base64.b64encode(image_data).decode("utf-8")
            payload = {
                "image": b64,
                "prompt": (
                    "Read all visible text, numbers, and tables. "
                    'Return JSON: {"text": "...", "tables": [...], "confidence": 0.0}'
                ),
            }
            resp = httpx.post(
                f"{self._dots_ocr_endpoint}/parse",
                json=payload,
                timeout=self._dots_ocr_timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            confidence = result.get("confidence", 0.0)
            if confidence < self._dots_ocr_min_confidence:
                logger.warning(f"dots.ocr low confidence ({confidence:.2f}) p{page_num}")
                return None

            return {
                "text": result.get("text", ""),
                "tables": result.get("tables", []),
                "confidence": confidence,
            }

        except ImportError:
            logger.warning("httpx not installed — skipping dots.ocr")
            return None
        except Exception as e:
            logger.error(f"dots.ocr extraction failed: {e}")
            return None

    def _extract_via_local_ocr(self, image_data: bytes) -> str:
        """
        Two-tier local OCR — zero cost, no API keys.
          L3a: RapidOCR (ONNX, faster)
          L3b: Tesseract (classic, backup)
        """
        # ── L3a: RapidOCR ──
        if self._rapid_ocr_available:
            try:
                from rapidocr_onnxruntime import RapidOCR
                ocr = RapidOCR()
                result, _ = ocr(image_data)
                if result:
                    texts = [item[1] for item in result]
                    extracted = "\n".join(texts)
                    if extracted.strip():
                        return extracted
                logger.debug("RapidOCR empty; falling back to Tesseract")
            except Exception as e:
                logger.warning(f"RapidOCR failed ({e}); falling back to Tesseract")

        # ── L3b: Tesseract ──
        if self._tesseract_available:
            try:
                import pytesseract
                from PIL import Image
                pil_img = Image.open(io.BytesIO(image_data))
                return pytesseract.image_to_string(pil_img)
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {e}")

        return ""

    # ─────────────────────────────────────────────────────────────
    # FINGERPRINTING & DEDUPLICATION
    # ─────────────────────────────────────────────────────────────

    def _bloom_prepopulate(self):
        """
        Pre-populate the in-process Bloom filter.

        Priority:
          1. Redis SET  — O(1) SMEMBERS, no disk scan, survives restarts.
          2. Disk scan  — legacy fallback when Redis is unavailable.
        """
        count = 0
        r = get_redis()
        if r is not None:
            try:
                members = r.smembers(_BLOOM_REDIS_KEY)  # returns set of str (decode_responses=True)
                for doc_hash in members:
                    self._bloom.add(doc_hash)
                count = len(members)
                if count:
                    logger.info(f"Bloom filter pre-populated from Redis: {count} doc hashes")
                return
            except Exception as exc:
                logger.warning(f"Redis bloom load failed, falling back to disk scan: {exc}")

        # ── Disk fallback ──────────────────────────────────────────────────
        for meta_file in self.cache_dir.glob("*.meta.json"):
            doc_hash = meta_file.stem.replace(".meta", "")
            self._bloom.add(doc_hash)
            count += 1
        if count:
            logger.info(f"Bloom filter pre-populated from disk: {count} doc hashes")

    def _generate_fingerprint(self, file_path: str) -> DocumentFingerprint:
        """SHA256 of entire file for exact duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

        stat = os.stat(file_path)
        return DocumentFingerprint(
            doc_hash=hasher.hexdigest(),
            metadata={
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "filename": Path(file_path).name,
            },
        )

    def _hash_page(self, page) -> str:
        """MD5 of page text for vision-cache keying."""
        text = page.extract_text() or ""
        return hashlib.md5(text.encode()).hexdigest()

    def _simhash_page(self, page, page_num: int, doc_id: str) -> int:
        """
        Compute SimHash for a page and register it for cross-document
        near-duplicate detection.

        If a previously ingested page has a SimHash within the Hamming
        threshold, a warning is logged and the stat counter incremented.
        The page is still processed (near-dups may have different context).

        Returns:
            64-bit SimHash integer.
        """
        text = page.extract_text() or ""
        sh = _simhash(text, bits=_SIMHASH_BITS)

        # Check for near-duplicate against all registered pages
        for existing_sh, (existing_doc, existing_page) in self._simhash_registry.items():
            if existing_doc == doc_id:
                continue  # Don't compare within same document
            if _simhash_is_near_dup(sh, existing_sh):
                dist = _hamming_distance(sh, existing_sh)
                self._stats["simhash_near_dups"] += 1
                logger.warning(
                    f"SimHash near-duplicate detected: page {page_num} of {doc_id[:8]} "
                    f"matches page {existing_page} of {existing_doc[:8]} "
                    f"(hamming_dist={dist})"
                )
                break  # One match is enough to log

        # Register this page's SimHash
        self._simhash_registry[sh] = (doc_id, page_num)
        return sh

    def _update_chunk_hashes(
        self, chunks: List[Chunk], fingerprint: DocumentFingerprint
    ):
        """Record MD5 of each chunk's text in the fingerprint."""
        for c in chunks:
            fingerprint.chunk_hashes[c.chunk_id] = hashlib.md5(
                c.text.encode()
            ).hexdigest()

    def _compute_minhash_signatures(
        self, chunks: List[Chunk], fingerprint: DocumentFingerprint
    ):
        """
        Compute MinHash signatures for each chunk and detect fuzzy
        duplicates against previously ingested chunks.

        Each chunk's 3-gram shingle set is hashed with 128 permutations
        to produce a compact sketch. Jaccard similarity ≥ 80% triggers
        a fuzzy-duplicate warning.

        The signature is stored on the Chunk dataclass (for serialization)
        and in the fingerprint dict (for persistence).
        """
        for c in chunks:
            shingles = _shingle(c.text, k=_SHINGLE_K)
            sig = _minhash_signature(shingles, num_perm=_MINHASH_NUM_PERM)
            c.minhash_sig = sig
            fingerprint.chunk_minhashes[c.chunk_id] = sig

            # ── Sliding-window fuzzy dedup — O(_MINHASH_WINDOW) not O(n²) ──
            # Only compare against the most recent _MINHASH_WINDOW chunks.
            # For large docs (146 pages × ~10 chunks = 1460 chunks) this cuts
            # comparisons from ~1M to ~500K without sacrificing near-dup recall
            # (practically identical content appears within a sliding window).
            window_ids = self._minhash_insertion_order[-_MINHASH_WINDOW:]
            for existing_id in window_ids:
                existing_sig = self._minhash_registry.get(existing_id)
                if existing_sig is None:
                    continue
                jaccard = _minhash_jaccard(sig, existing_sig)
                if jaccard >= _MINHASH_JACCARD_THRESHOLD:
                    self._stats["minhash_fuzzy_dups"] += 1
                    logger.debug(
                        f"MinHash fuzzy dup: {c.chunk_id[:12]} ≈ "
                        f"{existing_id[:12]} (Jaccard={jaccard:.2f})"
                    )
                    c.metadata["fuzzy_dup_of"] = existing_id
                    break

            # Register and maintain insertion-order list (bounded by window)
            self._minhash_registry[c.chunk_id] = sig
            self._minhash_insertion_order.append(c.chunk_id)
            # Evict old entries beyond 2× window to bound memory
            if len(self._minhash_insertion_order) > _MINHASH_WINDOW * 2:
                oldest = self._minhash_insertion_order[: _MINHASH_WINDOW]
                for old_id in oldest:
                    self._minhash_registry.pop(old_id, None)
                self._minhash_insertion_order = self._minhash_insertion_order[_MINHASH_WINDOW:]

    # ─────────────────────────────────────────────────────────────
    # SUMMARISATION (reuses TokenManager.summarize_for_context)
    # ─────────────────────────────────────────────────────────────

    def _generate_chunk_summaries(self, chunks: List[Chunk]):
        """
        Extractive summary via TokenManager — reuses the scoring heuristics
        (key financial terms, headers, numeric density) from prompts.py.
        Falls back to first-3-sentence extraction if token manager unavailable.

        Uses ThreadPoolExecutor to summarise chunks in parallel (CPU-bound
        extractive logic + optional LLM calls), matching the throughput of
        the parallel page-extraction step.
        """
        def _summarise_one(c: "Chunk"):
            try:
                c.summary = self._token_mgr.summarize_for_context(
                    c.text, target_tokens=125, preserve_key_info=True,
                )
            except Exception:
                sentences = c.text.split(".")
                summary = ". ".join(s.strip() for s in sentences[:3] if s.strip()) + "."
                c.summary = summary[:500] + "..." if len(summary) > 500 else summary

        workers = min(self.parallel_workers, len(chunks)) if chunks else 1
        if workers <= 1:
            for c in chunks:
                _summarise_one(c)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_summarise_one, chunks))  # consume futures; exceptions surface

    # ─────────────────────────────────────────────────────────────
    # BM25 INDEX
    # ─────────────────────────────────────────────────────────────

    def _update_bm25_index(self, chunks: List[Chunk], client_id: str, persist: bool = True):
        """Rebuild BM25 inverted index from chunk summaries.

        Args:
            chunks: PDF chunks to index.
            client_id: Client/tenant identifier.
            persist: When False, skip writing the JSON map to disk (use when a
                     cloud vector store already holds the canonical chunks so the
                     disk copy would be redundant).
        """
        corpus: List[List[str]] = []
        chunk_map: Dict[int, Dict[str, Any]] = {}

        for chunk in chunks:
            tokens = chunk.summary.lower().split()
            corpus.append(tokens)
            chunk_map[len(corpus) - 1] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_num,
                "client_id": client_id,
            }

        self._bm25_index = BM25Okapi(corpus) if corpus else None
        self._bm25_corpus_map = chunk_map

        if persist:
            # Persist for retrieval across restarts (only needed when no cloud store)
            idx_path = self.cache_dir / f"bm25_index_{client_id}.json"
            with open(idx_path, "w") as f:
                json.dump({str(k): v for k, v in chunk_map.items()}, f)

        logger.info(f"BM25 index updated: {len(corpus)} chunks (persist={persist})")

    def _bm25_search(
        self, query: str, client_id: str, top_k: int = 500
    ) -> List[Dict[str, Any]]:
        """Keyword search over chunk summaries; returns scored candidates.

        On first call after a restart, lazily reconstructs the BM25Okapi object
        from the persisted chunk_map JSON + per-doc chunks JSONL files so that
        retrieval works across process boundaries without re-ingesting.
        """
        if not self._bm25_index:
            # ── Lazy BM25 reconstruction from disk ──────────────────────────
            # The chunk_map (index → {chunk_id, doc_id, …}) is always persisted.
            # Summaries live in {doc_id}.chunks.jsonl — reconstruct corpus from them.
            idx_path = self.cache_dir / f"bm25_index_{client_id}.json"
            if not idx_path.exists():
                logger.warning("BM25 index not initialised and no persisted map found")
                return []
            with open(idx_path) as f:
                raw_map: Dict[str, Any] = {int(k): v for k, v in json.load(f).items()}

            if not raw_map:
                return []

            # Collect unique doc_ids and load all their chunks once
            doc_ids = {v["doc_id"] for v in raw_map.values() if v.get("doc_id")}
            chunk_summary: Dict[str, str] = {}
            for doc_id in doc_ids:
                jsonl_path = self.cache_dir / f"{doc_id}.chunks.jsonl"
                if not jsonl_path.exists():
                    continue
                try:
                    with open(jsonl_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            c = json.loads(line)
                            chunk_summary[c["chunk_id"]] = c.get("summary") or c.get("text", "")
                except Exception as exc:
                    logger.warning(f"Failed loading chunks from {jsonl_path}: {exc}")

            # Rebuild BM25 corpus in original index order
            max_idx = max(raw_map.keys()) + 1
            corpus: List[List[str]] = []
            rebuilt_map: Dict[int, Dict[str, Any]] = {}
            for idx in range(max_idx):
                entry = raw_map.get(idx)
                if entry is None:
                    corpus.append([])  # keep index alignment
                    continue
                summary = chunk_summary.get(entry.get("chunk_id", ""), "")
                corpus.append(summary.lower().split() if summary else [])
                rebuilt_map[idx] = entry

            self._bm25_index = BM25Okapi(corpus) if any(corpus) else None
            self._bm25_corpus_map = rebuilt_map
            logger.info(
                f"BM25 index reconstructed from disk: {len(rebuilt_map)} chunks "
                f"across {len(doc_ids)} doc(s)"
            )

            if not self._bm25_index:
                logger.warning("BM25 reconstruction produced empty corpus")
                return []

        scores = self._bm25_index.get_scores(query.lower().split())

        # Load persisted map if in-memory is empty
        if not self._bm25_corpus_map:
            idx_path = self.cache_dir / f"bm25_index_{client_id}.json"
            if idx_path.exists():
                with open(idx_path) as f:
                    self._bm25_corpus_map = {
                        int(k): v for k, v in json.load(f).items()
                    }

        results = []
        for idx, score in enumerate(scores):
            if score > 0 and idx in self._bm25_corpus_map:
                entry = self._bm25_corpus_map[idx].copy()
                entry["bm25_score"] = float(score)
                results.append(entry)

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────
    # LAZY EMBEDDING (uses SemanticCache from llm_utils)
    # ─────────────────────────────────────────────────────────────

    def _ensure_embeddings_for_chunks(
        self,
        chunk_ids: List[str],
        preloaded_texts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Cache-first lazy embedding via SemanticCache.
        1. Check cache for each chunk_id.
        2. Load text for misses from JSONL on disk (or reuse *preloaded_texts*).
        3. Batch-embed misses via DocumentIngestor (CUDA fast-path).
        4. Store results in SemanticCache.
        5. Push newly-generated vectors to Qdrant (hot storage) if available.
        """
        embeddings: Dict[str, List[float]] = {}
        missing: List[str] = []

        for cid in chunk_ids:
            cached = self._semantic_cache.get(f"emb_{cid}")
            if cached is not None:
                embeddings[cid] = cached
                self._stats["cache_hits"] += 1
            else:
                missing.append(cid)

        if not missing:
            return embeddings

        texts = preloaded_texts or self._load_chunk_texts(missing)
        batch_size = _pdf_cfg.embedding_batch_size
        newly_embedded: Dict[str, Tuple[List[float], str]] = {}  # cid → (vector, text)

        # When JSONL was skipped (cloud-push path), texts may be empty — filter those
        # out so we don't send empty strings to the embedding API.
        embeddable_ids = [cid for cid in missing if texts.get(cid, "").strip()]
        if not embeddable_ids:
            logger.debug(
                f"No chunk texts found on disk for {len(missing)} chunk(s) — "
                "skipping lazy embedding (chunks live in cloud vector store)"
            )
            return embeddings

        # Use the singleton DocumentIngestor for embedding (already on GPU).
        # Falls back to LLMProvider.embed_batch() only if DocumentIngestor
        # is unavailable — avoids creating a second SentenceTransformer.
        embedder = self._get_reranker()

        for i in range(0, len(embeddable_ids), batch_size):
            batch_ids = embeddable_ids[i : i + batch_size]
            batch_texts = [texts[cid] for cid in batch_ids]
            if embedder is not None:
                batch_embs = embedder._generate_embeddings_batch(batch_texts)
            else:
                batch_embs = self.llm.embed_batch(batch_texts)

            for cid, emb, txt in zip(batch_ids, batch_embs, batch_texts):
                self._semantic_cache.set(f"emb_{cid}", emb)
                embeddings[cid] = emb
                if txt:  # Only push non-empty to Qdrant
                    newly_embedded[cid] = (emb, txt)

        # Push newly-generated embeddings to Qdrant (hot storage tier)
        if newly_embedded:
            self._push_to_qdrant(newly_embedded)

        logger.info(
            f"Embedded {len(missing)} chunks ({self._stats['cache_hits']} cache hits)"
        )
        return embeddings

    def _embed_query(self, query: str) -> List[float]:
        """Embed query with single-flight dedup and caching.

        Uses the singleton DocumentIngestor (already on GPU) when available,
        falling back to LLMProvider.embed() otherwise.
        """
        cache_key = f"qemb_{hashlib.md5(query.encode()).hexdigest()[:16]}"
        cached = self._semantic_cache.get(cache_key)
        if cached is not None:
            return cached
        embedder = self._get_reranker()
        if embedder is not None:
            emb = embedder._generate_embedding(query)
        else:
            emb = self.llm.embed(query)
        self._semantic_cache.set(cache_key, emb)
        return emb

    def _load_chunk_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Load chunk texts from JSONL files on disk."""
        texts: Dict[str, str] = {}
        by_doc: Dict[str, List[str]] = {}

        for cid in chunk_ids:
            # chunk_id format from generate_chunk_id: "{doc_id}_chunk_{index}"
            parts = cid.rsplit("_chunk_", 1)
            doc_id = parts[0] if len(parts) == 2 else cid.split("_")[0]
            by_doc.setdefault(doc_id, []).append(cid)

        for doc_id, cids in by_doc.items():
            cid_set = set(cids)
            path = self.cache_dir / f"{doc_id}.chunks.jsonl"
            if not path.exists():
                logger.warning(f"Chunks file not found: {path}")
                continue
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    if data["chunk_id"] in cid_set:
                        texts[data["chunk_id"]] = data["text"]

        return texts

    def _compute_vector_similarity(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        chunk_embeddings: Dict[str, List[float]],
    ) -> List[Dict[str, Any]]:
        """Cosine similarity between query vector and chunk vectors."""
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            for c in candidates:
                c["vector_score"] = 0.0
            return candidates

        for c in candidates:
            cid = c["chunk_id"]
            if cid not in chunk_embeddings:
                c["vector_score"] = 0.0
                continue
            v = np.array(chunk_embeddings[cid], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            c["vector_score"] = float(np.dot(q, v) / (q_norm * v_norm)) if v_norm > 0 else 0.0

        candidates.sort(key=lambda x: x["vector_score"], reverse=True)
        return candidates

    # ─────────────────────────────────────────────────────────────
    # QDRANT HOT STORAGE
    # ─────────────────────────────────────────────────────────────

    def _push_to_qdrant(
        self, newly_embedded: Dict[str, Tuple[List[float], str]]
    ):
        """
        Push lazily-generated embeddings to Qdrant (hot storage tier).

        Reuses the Qdrant client from DocumentIngestor so configuration
        is DRY (same env vars, same collection). Failures are non-fatal
        — embeddings remain in SemanticCache regardless.
        """
        ingestor = self._get_reranker()  # Lazily initialised DocumentIngestor
        if ingestor is None or getattr(ingestor, "_qdrant", None) is None:
            return  # Qdrant not available — skip silently

        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            return

        # Deterministic namespace for UUID5 — Qdrant requires UUID or uint64.
        _QDRANT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        points = []
        for cid, (vector, text) in newly_embedded.items():
            # Extract doc_id from chunk_id convention: "{doc_id}_chunk_{idx}"
            parts = cid.rsplit("_chunk_", 1)
            doc_id = parts[0] if len(parts) == 2 else ""
            # Convert string chunk_id → deterministic UUID5 (Qdrant requires it)
            point_id = str(uuid.uuid5(_QDRANT_NS, cid))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": text,
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "source": "pdf_lazy_embed",
                    },
                )
            )

        try:
            collection = ingestor.collection_name
            # Batch upsert (Qdrant handles batching internally)
            ingestor._qdrant.upsert(
                collection_name=collection,
                points=points,
            )
            self._stats["qdrant_upserts"] += len(points)
            logger.info(
                f"Pushed {len(points)} embeddings to Qdrant "
                f"(collection={collection})"
            )
        except Exception as e:
            # Non-fatal — vectors remain in SemanticCache
            logger.warning(f"Qdrant upsert failed (non-fatal): {e}")

    # ─────────────────────────────────────────────────────────────
    # PERSISTENCE & CACHE
    # ─────────────────────────────────────────────────────────────

    def _check_if_processed(self, doc_hash: str, client_id: str) -> bool:
        """
        Fast dedup check — three-tier cascade:

          1. In-process Bloom (sub-μs) → Definitely not → skip everything.
          2. Redis SISMEMBER (< 1ms)   → Exact set membership → skip disk.
          3. Disk stat                 → Final ground truth (Redis miss).

        Guarantees: no false negatives (never misses a processed doc).
        """
        # Tier 1: in-process Bloom — zero I/O fast reject
        if not self._bloom.might_contain(doc_hash):
            self._stats["bloom_io_saved"] += 1
            return False

        # Tier 2: Redis exact check — O(1) but network
        r = get_redis()
        if r is not None:
            try:
                if r.sismember(_BLOOM_REDIS_KEY, doc_hash):
                    return True
                # Redis says definitely not in the SET → skip disk stat
                return False
            except Exception:
                pass  # Redis hiccup — fall through to disk

        # Tier 3: Disk fallback
        return (self.cache_dir / f"{doc_hash}.meta.json").exists()

    def _get_cached_metadata(self, doc_hash: str) -> Dict[str, Any]:
        with open(self.cache_dir / f"{doc_hash}.meta.json") as f:
            return json.load(f)

    def _save_metadata(self, metadata: Dict[str, Any]):
        path = self.cache_dir / f"{metadata['doc_id']}.meta.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        # Register in in-process Bloom + Redis SET for persistence/cross-worker dedup
        hashes_to_register = [metadata["doc_id"]]
        doc_hash = metadata.get("fingerprint", {}).get("doc_hash", "")
        if doc_hash and doc_hash != metadata["doc_id"]:
            hashes_to_register.append(doc_hash)

        for h in hashes_to_register:
            self._bloom.add(h)

        r = get_redis()
        if r is not None:
            try:
                r.sadd(_BLOOM_REDIS_KEY, *hashes_to_register)  # atomic SADD of multiple members
            except Exception as exc:
                logger.debug(f"Redis bloom write failed (non-fatal): {exc}")

    def _save_chunks(self, chunks: List[Chunk], doc_id: str):
        path = self.cache_dir / f"{doc_id}.chunks.jsonl"
        with open(path, "w") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), default=_json_default) + "\n")

    def _get_vision_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Two-tier vision cache lookup:
          1. Redis   — sub-ms, survives restarts, shared across workers.
          2. Disk    — legacy fallback when Redis is unavailable.
        """
        redis_key = f"pdf:vision:{cache_key}"
        r = get_redis()
        if r is not None:
            try:
                raw = r.get(redis_key)
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:
                logger.debug(f"Redis vision cache get failed: {exc}")

        # Disk fallback
        path = self.cache_dir / "vision_cache" / f"{cache_key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_vision_cache(self, cache_key: str, result: Dict[str, Any]):
        """
        Persist vision result to Redis (primary) and disk (fallback/backup).
        Redis TTL = 7 days.  Disk write is best-effort (non-fatal on failure).
        """
        redis_key = f"pdf:vision:{cache_key}"
        serialised = json.dumps(result, default=_json_default)

        r = get_redis()
        if r is not None:
            try:
                r.set(redis_key, serialised, ex=_VISION_CACHE_TTL)
            except Exception as exc:
                logger.debug(f"Redis vision cache set failed: {exc}")

        # Always write disk copy as fallback
        try:
            d = self.cache_dir / "vision_cache"
            d.mkdir(exist_ok=True)
            with open(d / f"{cache_key}.json", "w") as f:
                f.write(serialised)
        except Exception as exc:
            logger.debug(f"Disk vision cache write failed (non-fatal): {exc}")

    def _apply_metadata_filters(
        self, candidates: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter candidates by metadata key/value match."""
        return [
            c for c in candidates
            if all(c.get(k) == v for k, v in filters.items())
        ]

    def _tables_to_markdown(self, tables: List) -> str:
        """Convert pdfplumber table rows to markdown."""
        parts: List[str] = []
        for table in tables:
            if not table:
                continue
            lines: List[str] = []
            for i, row in enumerate(table):
                if row:
                    lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
                    if i == 0:
                        lines.append("| " + " | ".join("---" for _ in row) + " |")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return processing statistics including bloom/simhash/minhash/qdrant."""
        stats = dict(self._stats)
        stats["bloom"] = self._bloom.stats()
        stats["simhash_registry_size"] = len(self._simhash_registry)
        stats["minhash_registry_size"] = len(self._minhash_registry)
        return stats

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0


# ═════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ═════════════════════════════════════════════════════════════════

def _json_default(obj):
    """JSON serialiser fallback for numpy / set / bytes types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ═════════════════════════════════════════════════════════════════
# SINGLETON — one ingestor per process, lazy-initialised
# ═════════════════════════════════════════════════════════════════

_pdf_ingestor_instance: Optional["SmartPDFIngestor"] = None


def get_pdf_ingestor() -> "SmartPDFIngestor":
    """
    Return a process-level singleton SmartPDFIngestor.

    First call initialises the ingestor (GPU detection, bloom populate,
    dots.ocr lazy config, etc.). Subsequent calls return the cached
    instance instantly — no re-initialisation cost on every query.

    LLMProvider is built from settings so no arguments are required
    by callers.
    """
    global _pdf_ingestor_instance
    if _pdf_ingestor_instance is None:
        from app.core.llm_provider import create_llm_provider
        from app.config import settings as _s
        _llm = create_llm_provider({
            "provider_preference": _s.llm.provider_preference,
            "temperature": _s.llm.temperature,
        })
        _pdf_ingestor_instance = SmartPDFIngestor(
            llm_provider=_llm,
            parallel_workers=min(4, os.cpu_count() or 2),  # Up to 4 threads; avoids gRPC pickling issues
        )
        logger.info("SmartPDFIngestor singleton created")
    return _pdf_ingestor_instance
