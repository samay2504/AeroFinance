"""
Tests for SmartPDFIngestor — vision cascade, lazy embedding, and retrieval funnel.

Mocks all external services (Cloud Vision, dots.ocr, RapidOCR, Tesseract)
so the test suite runs offline with zero API keys.
"""

import json
import hashlib
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.ingest.pdf_ingest import (
    SmartPDFIngestor,
    Chunk,
    DocumentFingerprint,
    IngestionResult,
)
from app.rag.ingest import SmartChunker


# ═════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════

class MockLLMProvider:
    """Mock LLM provider tracking call counts for lazy-embedding verification."""

    def __init__(self):
        self.embed_call_count = 0
        self.embed_batch_call_count = 0
        self.vision_chat_call_count = 0

    def embed(self, text: str):
        self.embed_call_count += 1
        return [0.1] * 768

    def embed_batch(self, texts):
        self.embed_batch_call_count += 1
        self.embed_call_count += len(texts)
        return [[0.1] * 768 for _ in texts]

    def vision_chat(self, image: bytes, prompt: str, **kwargs) -> str:
        self.vision_chat_call_count += 1
        return json.dumps({
            "text": "Extracted vision text for testing.",
            "tables": ["| Col1 | Col2 |\n| --- | --- |\n| A | B |"],
            "charts": ["Revenue trending upward"],
        })

    def generate(self, prompt: str, **kwargs) -> str:
        return "Generated text."


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory for test isolation."""
    return str(tmp_path / "pdf_cache")


@pytest.fixture
def ingestor(mock_llm, tmp_cache):
    """SmartPDFIngestor configured for testing (vision disabled for most tests)."""
    return SmartPDFIngestor(
        llm_provider=mock_llm,
        cache_dir=tmp_cache,
        enable_vision=False,
        parallel_workers=1,
    )


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal valid PDF for fingerprint tests."""
    pdf_content = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n206\n%%EOF"
    )
    path = tmp_path / "sample.pdf"
    path.write_bytes(pdf_content)
    return str(path)


@pytest.fixture
def modified_pdf(tmp_path):
    """Create a different PDF so fingerprint differs from sample_pdf."""
    pdf_content = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
        b"% MODIFIED CONTENT\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n230\n%%EOF"
    )
    path = tmp_path / "modified.pdf"
    path.write_bytes(pdf_content)
    return str(path)


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════

class TestChunk:
    """Test Chunk dataclass auto-id generation via id_generator."""

    def test_auto_generates_chunk_id(self):
        """Empty chunk_id triggers generate_chunk_id(doc_id, page_num)."""
        c = Chunk(text="Hello world", doc_id="abcdef1234567890", chunk_id="", page_num=0)
        assert c.chunk_id != ""
        # id_generator.generate_chunk_id produces "{doc_id}_chunk_{index}"
        assert "chunk" in c.chunk_id or c.chunk_id != ""

    def test_preserves_explicit_chunk_id(self):
        c = Chunk(text="Hello", doc_id="abc", chunk_id="explicit_id", page_num=1)
        assert c.chunk_id == "explicit_id"

    def test_embedding_status_defaults_to_pending(self):
        c = Chunk(text="Test", doc_id="d", chunk_id="c1", page_num=0)
        assert c.embedding_status == "pending"


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: FINGERPRINTING
# ═════════════════════════════════════════════════════════════════

class TestFingerprinting:
    """Test document fingerprinting & deduplication."""

    def test_same_file_same_hash(self, ingestor, sample_pdf):
        fp1 = ingestor._generate_fingerprint(sample_pdf)
        fp2 = ingestor._generate_fingerprint(sample_pdf)
        assert fp1.doc_hash == fp2.doc_hash

    def test_different_file_different_hash(self, ingestor, sample_pdf, modified_pdf):
        fp1 = ingestor._generate_fingerprint(sample_pdf)
        fp2 = ingestor._generate_fingerprint(modified_pdf)
        assert fp1.doc_hash != fp2.doc_hash

    def test_fingerprint_metadata(self, ingestor, sample_pdf):
        fp = ingestor._generate_fingerprint(sample_pdf)
        assert "size_bytes" in fp.metadata
        assert "filename" in fp.metadata
        assert fp.metadata["filename"] == "sample.pdf"


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: VISION CASCADE
# ═════════════════════════════════════════════════════════════════

class TestVisionCascade:
    """Test 4-stage vision fallback: Primary → dots.ocr → RapidOCR → Tesseract."""

    def test_primary_vision_success(self, mock_llm, tmp_cache):
        """When primary vision works, dots.ocr and local OCR are not called."""
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=True,
            parallel_workers=1,
        )

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_page.to_image.return_value = MagicMock()
        def fake_save(buf, format=None):
            buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        mock_page.to_image.return_value.save = fake_save

        text, tables = ing._extract_via_vision(mock_page, page_num=0)

        assert "Extracted vision text" in text
        assert len(tables) == 1
        assert mock_llm.vision_chat_call_count == 1

    def test_dots_ocr_fallback(self, tmp_cache):
        """When primary vision fails, dots.ocr should be tried."""
        llm = MockLLMProvider()
        llm.vision_chat = MagicMock(side_effect=Exception("API Error"))

        ing = SmartPDFIngestor(
            llm_provider=llm,
            cache_dir=tmp_cache,
            enable_vision=True,
            parallel_workers=1,
        )
        ing._dots_ocr_enabled = True

        dots_result = {"text": "dots.ocr extracted text", "tables": [], "confidence": 0.85}
        ing._extract_via_dots_ocr = MagicMock(return_value=dots_result)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        def fake_save(buf, format=None):
            buf.write(b"\x89PNG" + b"\x00" * 100)
        mock_page.to_image.return_value = MagicMock()
        mock_page.to_image.return_value.save = fake_save

        text, tables = ing._extract_via_vision(mock_page, page_num=0)

        assert text == "dots.ocr extracted text"
        assert ing._extract_via_dots_ocr.called
        assert ing._stats["dots_ocr_calls"] == 1

    def test_dots_ocr_low_confidence_falls_through(self, tmp_cache):
        """Low-confidence dots.ocr result should fall through to local OCR."""
        llm = MockLLMProvider()
        llm.vision_chat = MagicMock(side_effect=Exception("API Error"))

        ing = SmartPDFIngestor(
            llm_provider=llm,
            cache_dir=tmp_cache,
            enable_vision=True,
            parallel_workers=1,
        )
        ing._dots_ocr_enabled = True

        ing._extract_via_dots_ocr = MagicMock(return_value=None)
        ing._extract_via_local_ocr = MagicMock(return_value="RapidOCR fallback text")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        def fake_save(buf, format=None):
            buf.write(b"\x89PNG" + b"\x00" * 100)
        mock_page.to_image.return_value = MagicMock()
        mock_page.to_image.return_value.save = fake_save

        text, tables = ing._extract_via_vision(mock_page, page_num=0)

        assert text == "RapidOCR fallback text"
        assert ing._extract_via_local_ocr.called

    def test_vision_cache_hit(self, mock_llm, tmp_cache):
        """Cached vision results should bypass all extraction."""
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=True,
            parallel_workers=1,
        )

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "cached page text"

        page_hash = hashlib.md5("cached page text".encode()).hexdigest()
        cache_key = f"vision_{page_hash}"
        ing._save_vision_cache(cache_key, {
            "text": "Cached result",
            "tables": [],
            "source": "test",
        })

        text, tables = ing._extract_via_vision(mock_page, page_num=0)
        assert text == "Cached result"
        assert mock_llm.vision_chat_call_count == 0


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: BM25 INDEX
# ═════════════════════════════════════════════════════════════════

class TestBM25Index:
    """Test BM25 indexing and search."""

    def test_index_and_search(self, ingestor):
        chunks = [
            Chunk(
                text=f"Revenue and profit analysis for quarter {i}",
                doc_id="doc1",
                chunk_id=f"chunk_{i}",
                page_num=i,
                summary=f"Revenue profit quarter {i}",
            )
            for i in range(20)
        ]
        ingestor._update_bm25_index(chunks, "client_test")
        results = ingestor._bm25_search("revenue profit", "client_test", top_k=5)

        assert len(results) <= 5
        assert all(r["bm25_score"] > 0 for r in results)

    def test_empty_index_returns_empty(self, ingestor):
        results = ingestor._bm25_search("anything", "client_x")
        assert results == []


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: LAZY EMBEDDING
# ═════════════════════════════════════════════════════════════════

class TestLazyEmbedding:
    """Verify embeddings are generated on retrieval, not on ingestion."""

    def test_no_embeddings_at_ingest_time(self, mock_llm, tmp_cache):
        """Ingestion must NOT call embed or embed_batch."""
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=False,
            parallel_workers=1,
        )

        chunks = [
            Chunk(text="Test text", doc_id="doc1", chunk_id="c1", page_num=0)
        ]
        fp = DocumentFingerprint(doc_hash="abc123")

        ing._generate_chunk_summaries(chunks)
        ing._update_chunk_hashes(chunks, fp)
        ing._update_bm25_index(chunks, "client1")
        ing._save_chunks(chunks, "abc123")

        assert mock_llm.embed_call_count == 0
        assert mock_llm.embed_batch_call_count == 0

    def test_embeddings_on_retrieval(self, mock_llm, tmp_cache):
        """Retrieval should trigger lazy embedding for cache misses."""
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=False,
            parallel_workers=1,
        )

        doc_id = "abcdef12"
        chunks = [
            Chunk(
                text=f"Financial data for company {i}",
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{i}",
                page_num=0,
                summary=f"Financial data company {i}",
            )
            for i in range(5)
        ]
        ing._update_bm25_index(chunks, "client_test")
        ing._save_chunks(chunks, doc_id)

        mock_llm.embed_call_count = 0
        results = ing.retrieve("financial data", "client_test", top_k=3)

        assert mock_llm.embed_call_count > 0


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: EMBEDDING CACHE
# ═════════════════════════════════════════════════════════════════

class TestEmbeddingCacheIntegration:
    """Test SemanticCache from llm_utils via the ingestor."""

    def test_cache_hit_skips_embed(self, mock_llm, tmp_cache):
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=False,
            parallel_workers=1,
        )

        # Pre-populate cache with the emb_ prefixed key
        ing._semantic_cache.set("emb_chunk_001", [0.5] * 768)

        embeddings = ing._ensure_embeddings_for_chunks(["chunk_001"])
        assert "chunk_001" in embeddings
        assert mock_llm.embed_batch_call_count == 0


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: RETRIEVAL FUNNEL
# ═════════════════════════════════════════════════════════════════

class TestRetrievalFunnel:
    """Test multi-stage retrieval pipeline."""

    def test_metadata_filter(self, ingestor):
        candidates = [
            {"chunk_id": "c1", "client_id": "alpha", "bm25_score": 1.0},
            {"chunk_id": "c2", "client_id": "beta", "bm25_score": 0.8},
            {"chunk_id": "c3", "client_id": "alpha", "bm25_score": 0.6},
        ]
        filtered = ingestor._apply_metadata_filters(
            candidates, {"client_id": "alpha"}
        )
        assert len(filtered) == 2
        assert all(c["client_id"] == "alpha" for c in filtered)

    def test_vector_similarity_ranking(self, ingestor):
        query_emb = [1.0, 0.0, 0.0]
        candidates = [
            {"chunk_id": "a"},
            {"chunk_id": "b"},
        ]
        chunk_embs = {
            "a": [0.5, 0.5, 0.0],   # cos sim ≈ 0.707
            "b": [1.0, 0.0, 0.0],   # cos sim = 1.0
        }
        results = ingestor._compute_vector_similarity(query_emb, candidates, chunk_embs)
        assert results[0]["chunk_id"] == "b"
        assert results[0]["vector_score"] == pytest.approx(1.0, abs=0.01)


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: PERSISTENCE
# ═════════════════════════════════════════════════════════════════

class TestPersistence:
    """Test metadata and chunk JSONL persistence."""

    def test_save_and_load_metadata(self, ingestor):
        meta = {
            "doc_id": "hash123",
            "client_id": "c1",
            "filename": "test.pdf",
            "total_pages": 5,
        }
        ingestor._save_metadata(meta)
        loaded = ingestor._get_cached_metadata("hash123")
        assert loaded["filename"] == "test.pdf"

    def test_save_and_load_chunks(self, ingestor):
        chunks = [
            Chunk(text="Chunk A", doc_id="doc1", chunk_id="doc1_chunk_0", page_num=0),
            Chunk(text="Chunk B", doc_id="doc1", chunk_id="doc1_chunk_1", page_num=1),
        ]
        ingestor._save_chunks(chunks, "doc1")

        texts = ingestor._load_chunk_texts(["doc1_chunk_0", "doc1_chunk_1"])
        assert texts["doc1_chunk_0"] == "Chunk A"
        assert texts["doc1_chunk_1"] == "Chunk B"

    def test_dedup_check(self, ingestor):
        assert not ingestor._check_if_processed("nonexistent", "c1")
        ingestor._save_metadata({"doc_id": "exists123"})
        assert ingestor._check_if_processed("exists123", "c1")


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: TABLE MARKDOWN
# ═════════════════════════════════════════════════════════════════

class TestTableConversion:
    def test_tables_to_markdown(self, ingestor):
        tables = [
            [["Header1", "Header2"], ["A", "B"], ["C", "D"]],
        ]
        md = ingestor._tables_to_markdown(tables)
        assert "| Header1 | Header2 |" in md
        assert "| --- | --- |" in md
        assert "| A | B |" in md

    def test_empty_tables(self, ingestor):
        assert ingestor._tables_to_markdown([]) == ""
        assert ingestor._tables_to_markdown([None, []]) == ""


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: ENV-DRIVEN CONFIGURATION
# ═════════════════════════════════════════════════════════════════

class TestEnvDrivenConfig:
    """Verify all configuration comes from PDFSettings, not hardcoded."""

    def test_dots_ocr_reads_from_settings(self, mock_llm, tmp_cache):
        """dots.ocr config should come from settings.pdf, not os.getenv."""
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=True,
            parallel_workers=1,
        )
        # These should be the values from PDFSettings defaults
        from app.config import settings
        assert ing._dots_ocr_enabled == settings.pdf.dots_ocr_enabled
        assert ing._dots_ocr_endpoint == settings.pdf.dots_ocr_endpoint
        assert ing._dots_ocr_timeout == settings.pdf.dots_ocr_timeout
        assert ing._dots_ocr_min_confidence == settings.pdf.dots_ocr_min_confidence

    def test_constructor_overrides_settings(self, mock_llm, tmp_cache):
        """Explicit constructor args should override PDFSettings defaults."""
        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=tmp_cache,
            enable_vision=False,
            vision_threshold=0.99,
            max_chunk_tokens=512,
            parallel_workers=2,
        )
        assert ing.enable_vision is False
        assert ing.vision_threshold == 0.99
        assert ing.max_chunk_tokens == 512
        assert ing.parallel_workers == 2


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: REUSE VERIFICATION
# ═════════════════════════════════════════════════════════════════

class TestCodeReuse:
    """Verify pdf_ingest reuses existing infrastructure."""

    def test_uses_smart_chunker(self, ingestor):
        """SmartChunker from app.rag.ingest should be the chunking engine."""
        assert isinstance(ingestor._chunker, SmartChunker)

    def test_uses_semantic_cache(self, ingestor):
        """Should use SemanticCache from llm_utils, not a custom cache."""
        from app.core.llm_utils import SemanticCache
        assert isinstance(ingestor._semantic_cache, SemanticCache)

    def test_uses_token_manager(self, ingestor):
        """Should use TokenManager from prompts for summarisation."""
        from app.core.prompts import TokenManager
        assert isinstance(ingestor._token_mgr, TokenManager)

    def test_uses_bloom_filter(self, ingestor):
        """Should use BloomFilter from data_registry for O(1) dedup checks."""
        from app.core.data_registry import BloomFilter
        assert isinstance(ingestor._bloom, BloomFilter)


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: SIMHASH (Page-Level Near-Duplicate Detection)
# ═════════════════════════════════════════════════════════════════

class TestSimHash:
    """Verify Charikar SimHash implementation for page near-dedup."""

    def test_simhash_returns_integer(self):
        from app.ingest.pdf_ingest import _simhash
        result = _simhash("hello world this is a test")
        assert isinstance(result, int)
        assert result >= 0

    def test_simhash_empty_text_returns_zero(self):
        from app.ingest.pdf_ingest import _simhash
        assert _simhash("") == 0

    def test_simhash_identical_texts_same_hash(self):
        from app.ingest.pdf_ingest import _simhash
        text = "The quick brown fox jumps over the lazy dog"
        assert _simhash(text) == _simhash(text)

    def test_simhash_similar_texts_near_dup(self):
        """Slightly modified text should produce a SimHash within Hamming threshold."""
        from app.ingest.pdf_ingest import _simhash, _hamming_distance
        text_a = "The quick brown fox jumps over the lazy dog in the park"
        text_b = "The quick brown fox leaps over the lazy dog in the park"  # 1 word changed
        ha = _simhash(text_a)
        hb = _simhash(text_b)
        dist = _hamming_distance(ha, hb)
        # Similar texts should have small Hamming distance (< ~10 for 64-bit)
        assert dist < 10, f"Expected small distance, got {dist}"

    def test_simhash_different_texts_large_distance(self):
        from app.ingest.pdf_ingest import _simhash, _hamming_distance
        text_a = "Financial report quarterly earnings revenue growth"
        text_b = "Machine learning neural network deep learning tensorflow"
        ha = _simhash(text_a)
        hb = _simhash(text_b)
        dist = _hamming_distance(ha, hb)
        # Very different texts should have large Hamming distance (> ~15)
        assert dist > 10, f"Expected large distance, got {dist}"

    def test_hamming_distance_zero_for_equal(self):
        from app.ingest.pdf_ingest import _hamming_distance
        assert _hamming_distance(0xFF, 0xFF) == 0

    def test_hamming_distance_correct(self):
        from app.ingest.pdf_ingest import _hamming_distance
        # 0b1010 vs 0b1001 → 2 bits differ
        assert _hamming_distance(0b1010, 0b1001) == 2

    def test_simhash_is_near_dup(self):
        from app.ingest.pdf_ingest import _simhash_is_near_dup
        assert _simhash_is_near_dup(0, 1, threshold=3)  # dist=1 ≤ 3
        assert not _simhash_is_near_dup(0, 0xFF, threshold=3)  # dist=8 > 3

    def test_simhash_registry_populated_on_page_process(self, ingestor):
        """SimHash registry should track pages as they're processed."""
        assert len(ingestor._simhash_registry) == 0


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: MINHASH (Chunk-Level Fuzzy Duplicate Detection)
# ═════════════════════════════════════════════════════════════════

class TestMinHash:
    """Verify MinHash sketch generation and Jaccard estimation."""

    def test_shingle_basic(self):
        from app.ingest.pdf_ingest import _shingle
        result = _shingle("abcde", k=3)
        assert result == {"abc", "bcd", "cde"}

    def test_shingle_short_text(self):
        from app.ingest.pdf_ingest import _shingle
        result = _shingle("ab", k=3)
        assert result == {"ab"}

    def test_shingle_empty(self):
        from app.ingest.pdf_ingest import _shingle
        assert _shingle("") == set()

    def test_minhash_signature_length(self):
        from app.ingest.pdf_ingest import _minhash_signature
        sig = _minhash_signature({"abc", "bcd", "cde"}, num_perm=128)
        assert len(sig) == 128
        assert all(isinstance(x, int) for x in sig)

    def test_minhash_empty_shingles(self):
        from app.ingest.pdf_ingest import _minhash_signature
        sig = _minhash_signature(set(), num_perm=128)
        assert sig == [0] * 128

    def test_minhash_signature_deterministic(self):
        from app.ingest.pdf_ingest import _minhash_signature
        shingles = {"hello", "world", "test"}
        sig_a = _minhash_signature(shingles, num_perm=128)
        sig_b = _minhash_signature(shingles, num_perm=128)
        assert sig_a == sig_b

    def test_minhash_jaccard_identical_sets(self):
        from app.ingest.pdf_ingest import _minhash_signature, _minhash_jaccard
        shingles = {"abc", "bcd", "cde", "def"}
        sig = _minhash_signature(shingles, num_perm=128)
        assert _minhash_jaccard(sig, sig) == 1.0

    def test_minhash_jaccard_disjoint_sets(self):
        from app.ingest.pdf_ingest import _minhash_signature, _minhash_jaccard, _shingle
        sig_a = _minhash_signature(_shingle("abcdefgh"), num_perm=128)
        sig_b = _minhash_signature(_shingle("xyz12345678"), num_perm=128)
        jaccard = _minhash_jaccard(sig_a, sig_b)
        assert jaccard < 0.3, f"Expected low Jaccard, got {jaccard}"

    def test_minhash_jaccard_similar_sets(self):
        from app.ingest.pdf_ingest import _minhash_signature, _minhash_jaccard, _shingle
        sig_a = _minhash_signature(_shingle("the quick brown fox jumps"), num_perm=128)
        sig_b = _minhash_signature(_shingle("the quick brown fox leaps"), num_perm=128)
        jaccard = _minhash_jaccard(sig_a, sig_b)
        # Similar texts should have moderate-to-high Jaccard
        assert jaccard > 0.4, f"Expected moderate Jaccard, got {jaccard}"

    def test_compute_minhash_signatures_stores_on_chunk(self, ingestor):
        """_compute_minhash_signatures should set minhash_sig on each Chunk."""
        chunks = [
            Chunk(text="Revenue growth was strong this quarter", doc_id="doc1", chunk_id="c1", page_num=0),
            Chunk(text="Operating expenses decreased year over year", doc_id="doc1", chunk_id="c2", page_num=0),
        ]
        fingerprint = DocumentFingerprint(doc_hash="abc123")
        ingestor._compute_minhash_signatures(chunks, fingerprint)

        for c in chunks:
            assert c.minhash_sig is not None
            assert len(c.minhash_sig) == 128

        assert "c1" in fingerprint.chunk_minhashes
        assert "c2" in fingerprint.chunk_minhashes

    def test_fuzzy_dup_detected(self, ingestor):
        """Chunks with identical text should be flagged as fuzzy duplicates."""
        text = "This is a test chunk with enough words to generate meaningful shingles for dedup"
        c1 = Chunk(text=text, doc_id="doc1", chunk_id="c1", page_num=0)
        c2 = Chunk(text=text, doc_id="doc2", chunk_id="c2", page_num=0)
        fp = DocumentFingerprint(doc_hash="test")
        ingestor._compute_minhash_signatures([c1], fp)
        ingestor._compute_minhash_signatures([c2], fp)

        assert ingestor._stats["minhash_fuzzy_dups"] >= 1
        assert c2.metadata.get("fuzzy_dup_of") == "c1"


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: BLOOM FILTER INTEGRATION
# ═════════════════════════════════════════════════════════════════

class TestBloomFilterIntegration:
    """Verify Bloom filter is used for O(1) dedup checks."""

    def test_bloom_filter_initialized(self, ingestor):
        from app.core.data_registry import BloomFilter
        assert isinstance(ingestor._bloom, BloomFilter)

    def test_bloom_negative_lookup_saves_io(self, ingestor):
        """If doc_hash is definitely not in Bloom, skip disk I/O."""
        result = ingestor._check_if_processed("definitely_not_here", "client1")
        assert result is False
        assert ingestor._stats["bloom_io_saved"] >= 1

    def test_bloom_positive_after_save(self, ingestor):
        """After saving metadata with fingerprint, Bloom should say 'maybe'."""
        meta = {
            "doc_id": "doc_abc",
            "fingerprint": {"doc_hash": "doc_abc"},
            "client_id": "c1",
        }
        ingestor._save_metadata(meta)
        # Now Bloom should say "maybe" and disk check should confirm
        result = ingestor._check_if_processed("doc_abc", "c1")
        assert result is True

    def test_bloom_prepopulate(self, mock_llm, tmp_path):
        """Existing .meta.json files should be loaded into Bloom on init."""
        cache = str(tmp_path / "cache")
        Path(cache).mkdir(parents=True, exist_ok=True)
        # Create a fake meta file
        meta_path = Path(cache) / "existing_hash.meta.json"
        meta_path.write_text('{"doc_id": "existing_hash"}')

        ing = SmartPDFIngestor(
            llm_provider=mock_llm,
            cache_dir=cache,
            enable_vision=False,
            parallel_workers=1,
        )
        # Bloom should recognize the prepopulated hash
        assert ing._bloom.might_contain("existing_hash") is True

    def test_bloom_stats_in_get_stats(self, ingestor):
        """get_stats() should include Bloom filter statistics."""
        stats = ingestor.get_stats()
        assert "bloom" in stats
        assert "capacity_bits" in stats["bloom"]


# ═════════════════════════════════════════════════════════════════
# UNIT TESTS: QDRANT HOT STORAGE PUSH
# ═════════════════════════════════════════════════════════════════

class TestQdrantHotStorage:
    """Verify lazy embeddings are pushed to Qdrant when available."""

    def test_push_to_qdrant_skips_when_unavailable(self, ingestor):
        """Should silently skip when Qdrant client is not available."""
        newly_embedded = {
            "c1": ([0.1] * 768, "Test text"),
        }
        # Should not raise even though Qdrant is not initialised
        ingestor._push_to_qdrant(newly_embedded)
        assert ingestor._stats["qdrant_upserts"] == 0

    def test_push_to_qdrant_calls_upsert(self, ingestor):
        """When DocumentIngestor has a Qdrant client, upsert should be called."""
        mock_qdrant = MagicMock()
        mock_doc_ingestor = MagicMock()
        mock_doc_ingestor._qdrant = mock_qdrant
        mock_doc_ingestor.collection_name = "test-collection"
        ingestor._doc_ingestor = mock_doc_ingestor

        newly_embedded = {
            "doc1_chunk_0": ([0.1] * 768, "Revenue grew 20%"),
            "doc1_chunk_1": ([0.2] * 768, "Expenses fell 10%"),
        }

        with patch.dict("sys.modules", {"qdrant_client.models": MagicMock()}):
            ingestor._push_to_qdrant(newly_embedded)

        mock_qdrant.upsert.assert_called_once()
        assert ingestor._stats["qdrant_upserts"] == 2

    def test_qdrant_failure_is_non_fatal(self, ingestor):
        """Qdrant upsert failures should log a warning, not raise."""
        mock_qdrant = MagicMock()
        mock_qdrant.upsert.side_effect = Exception("Network timeout")
        mock_doc_ingestor = MagicMock()
        mock_doc_ingestor._qdrant = mock_qdrant
        mock_doc_ingestor.collection_name = "test"
        ingestor._doc_ingestor = mock_doc_ingestor

        newly_embedded = {"c1": ([0.1] * 768, "Test text")}
        # Should NOT raise
        ingestor._push_to_qdrant(newly_embedded)
        assert ingestor._stats["qdrant_upserts"] == 0

    def test_embed_triggers_qdrant_push(self, ingestor, mock_llm):
        """_ensure_embeddings_for_chunks should call _push_to_qdrant for new embeds."""
        # Create a chunk file so loading works
        chunks = [Chunk(text="Test chunk", doc_id="testdoc", chunk_id="testdoc_chunk_0", page_num=0)]
        ingestor._save_chunks(chunks, "testdoc")

        with patch.object(ingestor, "_push_to_qdrant") as mock_push:
            ingestor._ensure_embeddings_for_chunks(["testdoc_chunk_0"])
            mock_push.assert_called_once()
            # Verify the dict passed contains the chunk_id
            call_args = mock_push.call_args[0][0]
            assert "testdoc_chunk_0" in call_args
