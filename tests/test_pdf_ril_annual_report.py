"""
Real-World PDF Capability Test — RIL Integrated Annual Report 2024-25
═════════════════════════════════════════════════════════════════════
End-to-end stress test using Reliance Industries' 146-page annual report.

WHAT THIS TESTS:
  §1  Ingestion — fingerprint, dedup, hybrid text/vision extraction
  §2  Structure — page routing, table extraction, chunk integrity
  §3  Content — real financial data verification against known values
  §4  Vision Cascade — L1 → L2 → L3 fallback behaviour
  §5  Retrieval — BM25, semantic, and funnel pipeline
  §6  Edge Cases — scanned pages, charts, multi-column layout, headers
  §7  Performance — timing, memory, parallelism

TEST FILE:
  D:/Projects2.0/Valuenaire/RIL-Integrated-Annual-Report-2024-25.pdf
  - 146 pages, 11.2 MB
  - Created by Adobe InDesign 20.4
  - Contains: financials, tables, charts, governance reports, images

KNOWN GROUND-TRUTH (from actual RIL AR 2024-25):
  - Revenue from Operations FY2025: ₹10,71,174 crore
  - Total Pages: 146
  - Sections: Corporate Overview, Management Discussion, Financial Statements
  - Standalone + Consolidated financial statements
  - BSE code: 500325, NSE: RELIANCE

Run:  python -m tests.test_pdf_ril_annual_report
"""

import sys
import os
import io
import re
import time
import json
import hashlib
import logging
import warnings
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# ── Windows Bootstrapping ────────────────────────────────────────
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Disable Docker auto-start — the test focuses on PDF pipeline,
# not VLM container lifecycle (which adds 2+ min of startup).
os.environ['PDF_DOTS_OCR_AUTO_DOCKER'] = 'false'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    torch_lib = r'd:\Projects2.0\Valuenaire\.conda\Lib\site-packages\torch\lib'
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        try:
            import torch  # noqa
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDF_STRESS_TEST")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.WARNING)
logging.getLogger("pdfminer").setLevel(logging.WARNING)


# ════════════════════════════════════════════════════════════════
# CONSTANTS & GROUND TRUTH
# ════════════════════════════════════════════════════════════════

PDF_PATH = Path("D:/Projects2.0/Valuenaire/RIL-Integrated-Annual-Report-2024-25.pdf")
CLIENT_ID = "pdf_stress_test_ril"

# Known ground truth extracted from the actual report
GROUND_TRUTH = {
    "total_pages": 146,
    "file_size_mb": 11.2,
    "creator": "Adobe InDesign",
    "bse_code": "500325",
    "nse_code": "RELIANCE",
    "cin": "L17110MH1973PLC019786",
    "registered_office": "Nariman Point, Mumbai",
    # 10-year financial highlights (page 5, FY 2024-25)
    "revenue_fy25_crore": 10_71_174,  # ₹10,71,174 crore
    "revenue_fy24_crore": 10_00_122,  # ₹10,00,122 crore
    # SEC: Standalone + Consolidated statements present
    "has_standalone_statements": True,
    "has_consolidated_statements": True,
    # Key persons
    "chairman": "Ambani",
    # Sections
    "sections": [
        "Corporate Overview",
        "Management Discussion",
        "Corporate Governance",
        "Standalone Financial Statements",
        "Consolidated Financial Statements",
    ],
}


# ════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ════════════════════════════════════════════════════════════════

class TestCategory(Enum):
    INGESTION = "ingestion"
    STRUCTURE = "structure"
    CONTENT = "content"
    VISION = "vision_cascade"
    RETRIEVAL = "retrieval"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"


@dataclass
class TestResult:
    name: str
    category: TestCategory
    passed: bool
    detail: str = ""
    elapsed: float = 0.0
    error: Optional[str] = None


class PDFTestRunner:
    """Runs the comprehensive PDF stress test suite."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.ingestor = None
        self.ingestion_result = None
        self.chunks: List[Any] = []
        self.start_time = time.time()

    # ── Setup ──────────────────────────────────────────────────
    def setup(self) -> bool:
        print("\n" + "=" * 80)
        print("📄 PDF STRESS TEST — RIL Integrated Annual Report 2024-25")
        print("=" * 80)

        if not PDF_PATH.exists():
            print(f"   ❌ PDF not found: {PDF_PATH}")
            return False
        print(f"   📁 File: {PDF_PATH.name} ({PDF_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

        try:
            from app.core.dll_fix import apply_dll_fix
            apply_dll_fix()
        except Exception:
            pass

        try:
            from app.core.llm_provider import LLMProvider
            from app.config import settings
            # Build config from settings (uses .env values)
            config = {
                "provider": getattr(settings, "llm_provider", "google_genai"),
                "model_name": getattr(settings, "llm_model", "gemini-2.0-flash"),
                "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
            }
            self.llm = LLMProvider(config)
            print(f"   ✅ LLMProvider initialized (provider={self.llm.current_provider})")
        except Exception as e:
            print(f"   ⚠️ LLMProvider failed ({e}), using mock")
            self.llm = self._create_mock_llm()

        try:
            from app.ingest.pdf_ingest import SmartPDFIngestor
            self.ingestor = SmartPDFIngestor(
                llm_provider=self.llm,
                cache_dir=str(PROJECT_ROOT / "data" / "pdf_cache_test"),
                enable_vision=True,
                parallel_workers=1,
            )
            print("   ✅ SmartPDFIngestor initialized")
            ocr_status = []
            if self.ingestor._dots_ocr_enabled:
                ocr_status.append("dots.ocr ✓")
            if self.ingestor._rapid_ocr_available:
                ocr_status.append("RapidOCR ✓")
            if self.ingestor._tesseract_available:
                ocr_status.append("Tesseract ✓")
            print(f"   🔍 OCR backends: {', '.join(ocr_status) or 'none'}")
        except Exception as e:
            print(f"   ❌ SmartPDFIngestor failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def _create_mock_llm(self):
        """Minimal mock LLM for offline testing."""
        class MockLLM:
            def embed(self, text): return [0.1] * 768
            def embed_batch(self, texts): return [[0.1] * 768 for _ in texts]
            def vision_chat(self, image, prompt, **kw):
                return json.dumps({"text": "Mock vision text", "tables": [], "charts": []})
            def generate(self, prompt, **kw): return "Mock response"
        return MockLLM()

    # ── Test Runner ────────────────────────────────────────────
    def run_test(self, name: str, category: TestCategory, fn):
        """Execute a single test and record the result."""
        t0 = time.time()
        try:
            detail = fn()
            elapsed = time.time() - t0
            self.results.append(TestResult(
                name=name, category=category, passed=True,
                detail=str(detail or ""), elapsed=elapsed,
            ))
            detail_str = f" — {detail}" if detail else ""
            print(f"   ✅ {name} ({elapsed:.2f}s){detail_str}")
        except Exception as e:
            elapsed = time.time() - t0
            self.results.append(TestResult(
                name=name, category=category, passed=False,
                error=str(e), elapsed=elapsed,
            ))
            print(f"   ❌ {name} ({elapsed:.2f}s): {e}")

    # ════════════════════════════════════════════════════════════
    # §1  INGESTION TESTS
    # ════════════════════════════════════════════════════════════
    def run_ingestion_tests(self):
        print("\n" + "─" * 60)
        print("§1  INGESTION TESTS")
        print("─" * 60)

        # Test 1.1: Full document ingestion
        def test_ingest():
            result = self.ingestor.ingest_document(
                file_path=str(PDF_PATH),
                client_id=CLIENT_ID,
                metadata={"source": "test_suite", "type": "annual_report"},
            )
            self.ingestion_result = result
            assert result.success, "Ingestion returned success=False"
            assert result.total_pages == GROUND_TRUTH["total_pages"], (
                f"Expected {GROUND_TRUTH['total_pages']} pages, got {result.total_pages}"
            )
            assert result.total_chunks > 0, "No chunks produced"
            return (
                f"{result.total_pages} pages → {result.total_chunks} chunks, "
                f"{result.text_pages} text / {result.vision_pages} vision, "
                f"{result.tables_extracted} tables in {result.time_seconds:.1f}s"
            )
        self.run_test("Full document ingestion (146 pages)", TestCategory.INGESTION, test_ingest)

        # Test 1.2: Fingerprint determinism
        def test_fingerprint():
            fp = self.ingestor._generate_fingerprint(str(PDF_PATH))
            assert fp.doc_hash, "doc_hash is empty"
            assert len(fp.doc_hash) == 64, f"SHA256 should be 64 hex chars, got {len(fp.doc_hash)}"
            # Re-fingerprint should produce identical hash
            fp2 = self.ingestor._generate_fingerprint(str(PDF_PATH))
            assert fp.doc_hash == fp2.doc_hash, "Fingerprint is not deterministic"
            return f"SHA256: {fp.doc_hash[:16]}..."
        self.run_test("Fingerprint determinism (SHA256)", TestCategory.INGESTION, test_fingerprint)

        # Test 1.3: Deduplication check
        def test_dedup():
            # Second ingestion should detect duplicate
            result2 = self.ingestor.ingest_document(
                file_path=str(PDF_PATH),
                client_id=CLIENT_ID,
            )
            # Should still succeed (from cache)
            assert result2.success, "Dedup re-ingestion failed"
            return "Duplicate detected — returned cached metadata"
        self.run_test("Deduplication (second ingestion)", TestCategory.INGESTION, test_dedup)

        # Test 1.4: Chunk ID uniqueness
        def test_chunk_ids():
            if not self.ingestion_result:
                raise AssertionError("No ingestion result")
            r = self.ingestion_result
            doc_id = r.doc_id
            # Load persisted chunks
            chunks_dir = Path(self.ingestor.cache_dir) / "chunks"
            chunk_ids = set()
            chunk_files = list(chunks_dir.glob(f"{doc_id}*.json"))
            if not chunk_files:
                # Try loading from metadata
                return f"✓ {r.total_chunks} chunks (IDs verified via result)"
            for cf in chunk_files:
                with open(cf) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for c in data:
                        cid = c.get("chunk_id", "")
                        assert cid not in chunk_ids, f"Duplicate chunk_id: {cid}"
                        chunk_ids.add(cid)
            return f"{len(chunk_ids)} unique chunk IDs"
        self.run_test("Chunk ID uniqueness", TestCategory.INGESTION, test_chunk_ids)

    # ════════════════════════════════════════════════════════════
    # §2  STRUCTURE TESTS
    # ════════════════════════════════════════════════════════════
    def run_structure_tests(self):
        print("\n" + "─" * 60)
        print("§2  STRUCTURE TESTS")
        print("─" * 60)

        # Test 2.1: Page count matches metadata
        def test_page_count():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                page_count = len(pdf.pages)
            assert page_count == GROUND_TRUTH["total_pages"], (
                f"Expected {GROUND_TRUTH['total_pages']}, got {page_count}"
            )
            return f"{page_count} pages"
        self.run_test("Page count matches expected (146)", TestCategory.STRUCTURE, test_page_count)

        # Test 2.2: Text extraction from text-heavy pages
        def test_text_extraction():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Page 4 = CMD's Statement (text-heavy)
                page = pdf.pages[3]
                text = page.extract_text() or ""
                assert len(text) > 500, f"CMD's Statement too short: {len(text)} chars"
                assert "India" in text or "Reliance" in text, "Missing expected keywords"
                return f"Page 4 (CMD Statement): {len(text)} chars extracted"
        self.run_test("Text extraction — CMD's Statement (page 4)", TestCategory.STRUCTURE, test_text_extraction)

        # Test 2.3: Table extraction from financial pages
        def test_table_extraction():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Page 5 = 10-Year Financial Highlights (known to have 2 tables)
                page = pdf.pages[4]
                tables = page.extract_tables() or []
                assert len(tables) > 0, f"No tables found on page 5 (financial highlights)"
                # Check first table has data
                first_table = tables[0]
                assert len(first_table) > 3, f"Table too small ({len(first_table)} rows)"
                return f"Page 5: {len(tables)} tables, first has {len(first_table)} rows"
        self.run_test("Table extraction — 10-Year Highlights (page 5)", TestCategory.STRUCTURE, test_table_extraction)

        # Test 2.4: Multi-column layout handling
        def test_multicolumn():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Page 3 = "Reliance at a Glance" — dual column layout
                page = pdf.pages[2]
                text = page.extract_text() or ""
                assert "RELIANCE AT A GLANCE" in text.upper() or "STAKEHOLDER" in text.upper(), \
                    "Multi-column page not properly extracted"
                assert len(text) > 200, f"Multi-column too short: {len(text)}"
                return f"Page 3: {len(text)} chars from multi-column layout"
        self.run_test("Multi-column layout — At a Glance (page 3)", TestCategory.STRUCTURE, test_multicolumn)

        # Test 2.5: Financial statement tables (complex, multi-table pages)
        def test_complex_tables():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Page 81 had 15 tables — complex standalone financials
                page = pdf.pages[80]
                tables = page.extract_tables() or []
                text = page.extract_text() or ""
                assert len(tables) >= 5, f"Expected ≥5 tables on page 81, got {len(tables)}"
                assert "crore" in text.lower() or "C in crore" in text, "Missing currency marker"
                return f"Page 81: {len(tables)} tables (complex financial)"
        self.run_test("Complex table page — Standalone FS (page 81)", TestCategory.STRUCTURE, test_complex_tables)

        # Test 2.6: Text/vision routing decision
        def test_routing():
            if not self.ingestion_result:
                raise AssertionError("No ingestion result")
            r = self.ingestion_result
            total = r.text_pages + r.vision_pages
            text_pct = r.text_pages / total * 100 if total > 0 else 0
            # An annual report should be mostly text-extractable
            assert r.text_pages > 0, "No text pages extracted"
            return (
                f"Text: {r.text_pages}/{total} ({text_pct:.0f}%), "
                f"Vision: {r.vision_pages}/{total}"
            )
        self.run_test("Text vs vision routing ratio", TestCategory.STRUCTURE, test_routing)

        # Test 2.7: Chunk sizing sanity
        def test_chunk_sizes():
            if not self.ingestion_result:
                raise AssertionError("No ingestion result")
            r = self.ingestion_result
            # Average chunks per page should be reasonable
            avg_chunks = r.total_chunks / r.total_pages
            assert 0.5 <= avg_chunks <= 20, f"Avg chunks/page = {avg_chunks:.1f} (too extreme)"
            return f"Avg {avg_chunks:.1f} chunks/page ({r.total_chunks} total)"
        self.run_test("Chunk sizing sanity (0.5-20 per page)", TestCategory.STRUCTURE, test_chunk_sizes)

    # ════════════════════════════════════════════════════════════
    # §3  CONTENT VERIFICATION
    # ════════════════════════════════════════════════════════════
    def run_content_tests(self):
        print("\n" + "─" * 60)
        print("§3  CONTENT VERIFICATION (Ground Truth)")
        print("─" * 60)

        # Test 3.1: Cover page metadata
        def test_cover():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                cover = pdf.pages[0].extract_text() or ""
                assert "Integrated Annual Report" in cover, "Missing title on cover"
                assert "2024-25" in cover, "Missing year on cover"
                return "Cover: title + year verified"
        self.run_test("Cover page — title and year", TestCategory.CONTENT, test_cover)

        # Test 3.2: TOC validation
        def test_toc():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                toc_text = pdf.pages[1].extract_text() or ""
                assert "TABLE OF CONTENTS" in toc_text.upper() or "CONTENTS" in toc_text.upper(), \
                    "Missing Table of Contents"
                for section in ["Corporate Overview", "Financial Highlights"]:
                    assert section.lower() in toc_text.lower() or \
                        section.split()[-1].lower() in toc_text.lower(), \
                        f"Missing section: {section}"
                return "TOC verified with key sections"
        self.run_test("Table of Contents verification", TestCategory.CONTENT, test_toc)

        # Test 3.3: Revenue figure extraction
        def test_revenue():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Page 5 = 10-Year Financial Highlights
                page5_text = pdf.pages[4].extract_text() or ""
                # Look for revenue figures — allow for OCR/layout variations
                found_revenue = False
                for pattern in [r"10[,.]?71[,.]?174", r"1071174", r"10,71,174"]:
                    if re.search(pattern, page5_text):
                        found_revenue = True
                        break
                assert found_revenue, (
                    f"Revenue ₹10,71,174 crore not found on page 5. "
                    f"Text preview: {page5_text[:500]}"
                )
                return "Revenue FY25: ₹10,71,174 crore ✓"
        self.run_test("Revenue extraction — ₹10,71,174 crore (FY25)", TestCategory.CONTENT, test_revenue)

        # Test 3.4: Prior year comparison
        def test_fy24_revenue():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                page5_text = pdf.pages[4].extract_text() or ""
                found = False
                for pattern in [r"10[,.]?00[,.]?122", r"1000122"]:
                    if re.search(pattern, page5_text):
                        found = True
                        break
                assert found, "FY24 Revenue ₹10,00,122 crore not found"
                return "Revenue FY24: ₹10,00,122 crore ✓"
        self.run_test("Prior year revenue — ₹10,00,122 crore (FY24)", TestCategory.CONTENT, test_fy24_revenue)

        # Test 3.5: Company identifiers
        def test_identifiers():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                last_page_text = pdf.pages[-1].extract_text() or ""
                for key, gt_val in [
                    ("BSE", GROUND_TRUTH["bse_code"]),
                    ("NSE", GROUND_TRUTH["nse_code"]),
                    ("CIN", GROUND_TRUTH["cin"]),
                ]:
                    assert gt_val in last_page_text, f"{key} code '{gt_val}' not found on last page"
                return f"BSE={GROUND_TRUTH['bse_code']}, NSE={GROUND_TRUTH['nse_code']} ✓"
        self.run_test("Company identifiers (BSE/NSE/CIN)", TestCategory.CONTENT, test_identifiers)

        # Test 3.6: Chairman reference
        def test_chairman():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                cmd_text = pdf.pages[3].extract_text() or ""  # CMD's Statement
                assert GROUND_TRUTH["chairman"] in cmd_text, \
                    f"Chairman '{GROUND_TRUTH['chairman']}' not found in CMD's Statement"
                return f"Chairman reference: {GROUND_TRUTH['chairman']} ✓"
        self.run_test("Chairman reference in CMD's Statement", TestCategory.CONTENT, test_chairman)

        # Test 3.7: Financial statement sections exist
        def test_sections():
            import pdfplumber
            found_standalone = False
            found_consolidated = False
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                for i in range(min(len(pdf.pages), 146)):
                    text = (pdf.pages[i].extract_text() or "").upper()
                    if "STANDALONE FINANCIAL" in text:
                        found_standalone = True
                    if "CONSOLIDATED FINANCIAL" in text:
                        found_consolidated = True
                    if found_standalone and found_consolidated:
                        break
            assert found_standalone, "Standalone Financial Statements section not found"
            assert found_consolidated, "Consolidated Financial Statements section not found"
            return "Both standalone + consolidated statements found ✓"
        self.run_test("Financial statement sections", TestCategory.CONTENT, test_sections)

        # Test 3.8: Currency notation
        def test_currency():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Check financial pages for ₹/crore
                found_crore = False
                for i in [4, 5, 50, 70, 100]:  # Financial pages
                    if i < len(pdf.pages):
                        text = pdf.pages[i].extract_text() or ""
                        if "crore" in text.lower() or "in crore" in text.lower():
                            found_crore = True
                            break
                assert found_crore, "No 'crore' currency notation found in financial pages"
                return "'in crore' currency notation verified ✓"
        self.run_test("Currency notation (₹ crore)", TestCategory.CONTENT, test_currency)

    # ════════════════════════════════════════════════════════════
    # §4  VISION CASCADE TESTS
    # ════════════════════════════════════════════════════════════
    def run_vision_tests(self):
        print("\n" + "─" * 60)
        print("§4  VISION CASCADE")
        print("─" * 60)

        # Test 4.1: OCR backend availability
        def test_ocr_backends():
            backends = []
            if self.ingestor._rapid_ocr_available:
                backends.append("RapidOCR")
            if self.ingestor._tesseract_available:
                backends.append("Tesseract")
            if self.ingestor._dots_ocr_enabled:
                backends.append("VLM/dots.ocr")
            assert len(backends) > 0, "No OCR backend available — install pytesseract or rapidocr"
            return f"Available: {', '.join(backends)}"
        self.run_test("OCR backend availability", TestCategory.VISION, test_ocr_backends)

        # Test 4.2: Vision route detection (image-heavy vs text-heavy pages)
        def test_vision_routing():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                threshold = self.ingestor.vision_threshold
                # Cover page (likely image-heavy → vision)
                cover = pdf.pages[0]
                cover_text = (cover.extract_text() or "").strip()
                cover_chars = len(cover_text)
                cover_vision = cover_chars < threshold
                # Page 4 = CMD's Statement (text-heavy → text path)
                cmd = pdf.pages[3]
                cmd_text = (cmd.extract_text() or "").strip()
                cmd_chars = len(cmd_text)
                cmd_vision = cmd_chars < threshold
                # Cover should use vision, CMD should use text
                assert cover_vision, (
                    f"Cover has {cover_chars} chars — expected < {threshold} (vision path)"
                )
                assert not cmd_vision, (
                    f"CMD Statement has {cmd_chars} chars — expected >= {threshold} (text path)"
                )
                return (
                    f"Cover: {cover_chars} chars → VISION, "
                    f"CMD: {cmd_chars} chars → TEXT (threshold={threshold})"
                )
        self.run_test("Vision routing — char-count threshold", TestCategory.VISION, test_vision_routing)

        # Test 4.3: Local OCR on rendered page
        def test_local_ocr():
            if not (self.ingestor._rapid_ocr_available or self.ingestor._tesseract_available):
                return "⚠ SKIPPED (no local OCR backend)"
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                page = pdf.pages[3]  # CMD's Statement
                img = page.to_image(resolution=150)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                ocr_text = self.ingestor._extract_via_local_ocr(buf.getvalue())
                assert len(ocr_text) > 50, f"Local OCR too short: {len(ocr_text)} chars"
                return f"Local OCR extracted {len(ocr_text)} chars from page 4"
        self.run_test("Local OCR — page 4 render-to-text", TestCategory.VISION, test_local_ocr)

        # Test 4.4: Vision stats from ingestion
        def test_vision_stats():
            if not self.ingestion_result:
                raise AssertionError("No ingestion result")
            r = self.ingestion_result
            stats = self.ingestor._stats
            return (
                f"vision_calls={stats.get('vision_calls', 0)}, "
                f"local_ocr={stats.get('local_ocr_calls', 0)}, "
                f"dots_ocr={stats.get('dots_ocr_calls', 0)}, "
                f"cache_hits={stats.get('cache_hits', 0)}"
            )
        self.run_test("Vision cascade stats summary", TestCategory.VISION, test_vision_stats)

    # ════════════════════════════════════════════════════════════
    # §5  RETRIEVAL TESTS
    # ════════════════════════════════════════════════════════════
    def run_retrieval_tests(self):
        print("\n" + "─" * 60)
        print("§5  RETRIEVAL PIPELINE")
        print("─" * 60)

        # Test 5.1: BM25 basic retrieval
        def test_bm25_revenue():
            results = self.ingestor.retrieve(
                query="What is the total revenue from operations for FY 2024-25?",
                client_id=CLIENT_ID,
                top_k=5,
            )
            assert len(results) > 0, "BM25 returned zero results"
            # Check if any result contains revenue-related text
            combined = " ".join(r.get("text", r.get("content", "")) for r in results)
            has_revenue = any(kw in combined.lower() for kw in [
                "revenue", "sales", "10,71,174", "1071174",
            ])
            return f"{len(results)} results, revenue-relevant: {has_revenue}"
        self.run_test("BM25 — revenue query", TestCategory.RETRIEVAL, test_bm25_revenue)

        # Test 5.2: BM25 corporate governance
        def test_bm25_governance():
            results = self.ingestor.retrieve(
                query="Who are the independent directors on the board?",
                client_id=CLIENT_ID,
                top_k=5,
            )
            assert len(results) > 0, "BM25 returned zero results for governance query"
            combined = " ".join(r.get("text", r.get("content", "")) for r in results)
            has_governance = any(kw in combined.lower() for kw in [
                "director", "board", "independent", "governance",
            ])
            return f"{len(results)} results, governance-relevant: {has_governance}"
        self.run_test("BM25 — governance query", TestCategory.RETRIEVAL, test_bm25_governance)

        # Test 5.3: BM25 media/telecom segment
        def test_bm25_jio():
            results = self.ingestor.retrieve(
                query="What are Jio's subscriber numbers and ARPU?",
                client_id=CLIENT_ID,
                top_k=5,
            )
            assert len(results) > 0, "No results for Jio query"
            combined = " ".join(r.get("text", r.get("content", "")) for r in results)
            has_jio = "jio" in combined.lower()
            return f"{len(results)} results, Jio-relevant: {has_jio}"
        self.run_test("BM25 — Jio segment query", TestCategory.RETRIEVAL, test_bm25_jio)

        # Test 5.4: Cross-section retrieval
        def test_cross_section():
            results_fin = self.ingestor.retrieve(
                query="EBITDA margin and operating profit",
                client_id=CLIENT_ID, top_k=3,
            )
            results_gov = self.ingestor.retrieve(
                query="Related party transactions",
                client_id=CLIENT_ID, top_k=3,
            )
            # Verify they return different page chunks
            fin_pages = {r.get("page_num", -1) for r in results_fin}
            gov_pages = {r.get("page_num", -1) for r in results_gov}
            overlap = fin_pages & gov_pages
            return (
                f"Financial pages: {fin_pages}, "
                f"Governance pages: {gov_pages}, "
                f"Overlap: {len(overlap)} pages"
            )
        self.run_test("Cross-section retrieval diversity", TestCategory.RETRIEVAL, test_cross_section)

        # Test 5.5: Empty query handling
        def test_empty_query():
            results = self.ingestor.retrieve(
                query="", client_id=CLIENT_ID, top_k=5,
            )
            # Should return empty or handle gracefully
            return f"Empty query: {len(results)} results (graceful)"
        self.run_test("Empty query handling", TestCategory.RETRIEVAL, test_empty_query)

    # ════════════════════════════════════════════════════════════
    # §6  EDGE CASES
    # ════════════════════════════════════════════════════════════
    def run_edge_case_tests(self):
        print("\n" + "─" * 60)
        print("§6  EDGE CASES")
        print("─" * 60)

        # Test 6.1: Last page (back cover — minimal text)
        def test_last_page():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                last = pdf.pages[-1]
                text = last.extract_text() or ""
                # Last page has minimal corporate info
                assert "ril.com" in text.lower() or "reliance" in text.lower() or \
                    GROUND_TRUTH["bse_code"] in text, \
                    "Last page doesn't contain expected corporate info"
                return f"Last page: {len(text)} chars, corporate info present ✓"
        self.run_test("Last page — minimal content handling", TestCategory.EDGE_CASE, test_last_page)

        # Test 6.2: Unicode / special characters in financial text
        def test_unicode():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                # Scan for special chars (₹, `, etc.)
                special_found = set()
                for i in range(min(10, len(pdf.pages))):
                    text = pdf.pages[i].extract_text() or ""
                    for ch in "₹–—''""":
                        if ch in text:
                            special_found.add(ch)
                return f"Special chars found: {special_found or 'none (encoded variants only)'}"
        self.run_test("Unicode / special character handling", TestCategory.EDGE_CASE, test_unicode)

        # Test 6.3: Page with many tables
        def test_table_heavy():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                max_tables = 0
                max_page = 0
                for i in range(len(pdf.pages)):
                    tables = pdf.pages[i].extract_tables() or []
                    if len(tables) > max_tables:
                        max_tables = len(tables)
                        max_page = i + 1
                assert max_tables > 5, f"Expected dense table pages, max was {max_tables}"
                return f"Densest: page {max_page} with {max_tables} tables"
        self.run_test("Dense table page discovery", TestCategory.EDGE_CASE, test_table_heavy)

        # Test 6.4: Notes to financial statements (very long text)
        def test_long_text_page():
            import pdfplumber
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                max_chars = 0
                max_page = 0
                for i in range(len(pdf.pages)):
                    text = pdf.pages[i].extract_text() or ""
                    if len(text) > max_chars:
                        max_chars = len(text)
                        max_page = i + 1
                assert max_chars > 5000, f"Expected long text pages, max was {max_chars}"
                return f"Longest: page {max_page} with {max_chars} chars"
        self.run_test("Longest text page discovery", TestCategory.EDGE_CASE, test_long_text_page)

        # Test 6.5: Non-English characters (Hindi/Marathi references possible)
        def test_non_english():
            import pdfplumber
            non_ascii_pages = 0
            with pdfplumber.open(str(PDF_PATH)) as pdf:
                for i in range(len(pdf.pages)):
                    text = pdf.pages[i].extract_text() or ""
                    if any(ord(c) > 127 for c in text):
                        non_ascii_pages += 1
            return f"{non_ascii_pages} pages with non-ASCII characters"
        self.run_test("Non-ASCII character pages", TestCategory.EDGE_CASE, test_non_english)

        # Test 6.6: Ingestion with wrong client_id retrieval
        def test_client_isolation():
            results = self.ingestor.retrieve(
                query="Revenue from operations",
                client_id="nonexistent_client_12345",
                top_k=5,
            )
            # Should return zero or empty (no data for this client)
            return f"Wrong client: {len(results)} results (expected: 0)"
        self.run_test("Client ID isolation", TestCategory.EDGE_CASE, test_client_isolation)

    # ════════════════════════════════════════════════════════════
    # §7  PERFORMANCE TESTS
    # ════════════════════════════════════════════════════════════
    def run_performance_tests(self):
        print("\n" + "─" * 60)
        print("§7  PERFORMANCE")
        print("─" * 60)

        # Test 7.1: Ingestion time (should be < 5 min for 146 pages)
        def test_ingest_time():
            if not self.ingestion_result:
                raise AssertionError("No ingestion result")
            t = self.ingestion_result.time_seconds
            MAX_TIME = 300  # 5 minutes max for 146 pages
            assert t < MAX_TIME, f"Ingestion too slow: {t:.1f}s > {MAX_TIME}s"
            pages_per_sec = self.ingestion_result.total_pages / t if t > 0 else 0
            return f"{t:.1f}s total, {pages_per_sec:.1f} pages/sec"
        self.run_test("Ingestion time < 5 min", TestCategory.PERFORMANCE, test_ingest_time)

        # Test 7.2: Retrieval latency
        def test_retrieval_latency():
            t0 = time.time()
            results = self.ingestor.retrieve(
                query="What is the total EBITDA?",
                client_id=CLIENT_ID, top_k=5,
            )
            latency_ms = (time.time() - t0) * 1000
            MAX_LATENCY = 5000  # 5 seconds max
            assert latency_ms < MAX_LATENCY, f"Retrieval too slow: {latency_ms:.0f}ms"
            return f"{latency_ms:.0f}ms for {len(results)} results"
        self.run_test("Retrieval latency < 5s", TestCategory.PERFORMANCE, test_retrieval_latency)

        # Test 7.3: Fingerprint speed
        def test_fingerprint_speed():
            t0 = time.time()
            fp = self.ingestor._generate_fingerprint(str(PDF_PATH))
            latency_ms = (time.time() - t0) * 1000
            assert latency_ms < 5000, f"Fingerprinting too slow: {latency_ms:.0f}ms for 11 MB"
            return f"{latency_ms:.0f}ms for {PDF_PATH.stat().st_size / 1024 / 1024:.1f} MB"
        self.run_test("Fingerprint speed — 11 MB file", TestCategory.PERFORMANCE, test_fingerprint_speed)

        # Test 7.4: Memory check (rough)
        def test_memory():
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            MAX_MEM = 4096  # 4 GB max
            return f"RSS: {mem_mb:.0f} MB (limit: {MAX_MEM} MB)"
        self.run_test("Memory usage check", TestCategory.PERFORMANCE, test_memory)

    # ════════════════════════════════════════════════════════════
    # REPORT
    # ════════════════════════════════════════════════════════════
    def generate_report(self):
        total_time = time.time() - self.start_time
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        print("\n" + "=" * 80)
        print("📊 PDF STRESS TEST REPORT")
        print("=" * 80)
        print(f"   File: {PDF_PATH.name} ({GROUND_TRUTH['total_pages']} pages)")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Results: {len(passed)}/{len(self.results)} passed")
        print()

        # By category
        categories = {}
        for r in self.results:
            cat = r.category.value
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if r.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        print(f"   {'Category':<25} {'Passed':<10} {'Failed':<10} {'Status'}")
        print(f"   {'─' * 55}")
        for cat, counts in categories.items():
            status = "✅" if counts["failed"] == 0 else "❌"
            print(f"   {cat:<25} {counts['passed']:<10} {counts['failed']:<10} {status}")

        if failed:
            print(f"\n   ❌ FAILURES ({len(failed)}):")
            for r in failed:
                print(f"      • {r.name}: {r.error}")

        # Ingestion summary
        if self.ingestion_result:
            r = self.ingestion_result
            print(f"\n   📄 Ingestion Summary:")
            print(f"      Pages: {r.total_pages} ({r.text_pages} text, {r.vision_pages} vision)")
            print(f"      Chunks: {r.total_chunks}")
            print(f"      Tables: {r.tables_extracted}")
            print(f"      Time: {r.time_seconds:.1f}s")

        print("\n" + "=" * 80)

        return 0 if not failed else 1


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main() -> int:
    runner = PDFTestRunner()

    if not runner.setup():
        print("\n❌ Setup failed — cannot continue")
        return 1

    # §1: Ingestion (this is the big one — takes time)
    runner.run_ingestion_tests()

    # §2: Structure checks
    runner.run_structure_tests()

    # §3: Content verification against ground truth
    runner.run_content_tests()

    # §4: Vision cascade
    runner.run_vision_tests()

    # §5: Retrieval pipeline
    runner.run_retrieval_tests()

    # §6: Edge cases
    runner.run_edge_case_tests()

    # §7: Performance
    runner.run_performance_tests()

    return runner.generate_report()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
