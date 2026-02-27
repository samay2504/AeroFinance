#!/usr/bin/env python
"""
═══════════════════════════════════════════════════════════════════════
  QA API TESTER — Enterprise-Grade Integration Test Suite
  Target: AI-CA FastAPI Server (http://localhost:9999)
  Focus:  Smart PDF Ingestion & Retrieval Pipeline
═══════════════════════════════════════════════════════════════════════

Workflow:
  1. Health check          — verify the server is alive
  2. ID creation           — generate a tracked session ID
  3. PDF upload            — ingest the RIL Integrated Annual Report (146 pages)
  4. Dataset verification  — confirm the uploaded doc appears in the registry
  5. Query (sync)          — hit the BM25 → Lazy Embed → Vector funnel
  6. Query (SSE stream)    — validate real-time token streaming
  7. Metrics               — confirm telemetry recorded the test hits

Tested PDF:
  D:/Projects2.0/Valuenaire/RIL-Integrated-Annual-Report-2024-25.pdf
  - 146 pages, ~11.2 MB, Adobe InDesign generated
  - Contains: financials, tables, charts, governance reports, images
  - Known revenue FY25: ₹10,71,174 crore

Usage:
  python qa_api_tester.py                        # default: localhost:9999
  python qa_api_tester.py --base-url http://host:port/ai-ca
  python qa_api_tester.py --pdf C:/other/report.pdf

Requirements:
  pip install requests
"""

import argparse
import io
import json
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ═════════════════════════════════════════════════════════════════
# ANSI COLOURS
# ═════════════════════════════════════════════════════════════════
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _pass(label: str, detail: str = "", latency_ms: float = 0):
    lat = f" {DIM}({latency_ms:.0f}ms){RESET}" if latency_ms else ""
    det = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {GREEN}[PASS]{RESET} {label}{lat}{det}")


def _fail(label: str, detail: str = "", latency_ms: float = 0):
    lat = f" {DIM}({latency_ms:.0f}ms){RESET}" if latency_ms else ""
    det = f"  {RED}{detail}{RESET}" if detail else ""
    print(f"  {RED}[FAIL]{RESET} {label}{lat}{det}")


def _warn(label: str, detail: str = "", latency_ms: float = 0):
    lat = f" {DIM}({latency_ms:.0f}ms){RESET}" if latency_ms else ""
    det = f"  {YELLOW}{detail}{RESET}" if detail else ""
    print(f"  {YELLOW}[WARN]{RESET} {label}{lat}{det}")


def _info(msg: str):
    print(f"  {DIM}-> {msg}{RESET}")


def _header(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


# ═════════════════════════════════════════════════════════════════
# FALLBACK DUMMY PDF GENERATOR (only if real PDF not found)
# ═════════════════════════════════════════════════════════════════

def generate_dummy_pdf(text: str = "The Q3 Revenue was $50 Million") -> bytes:
    """
    Builds a minimal valid PDF 1.4 file in memory.
    Used only as fallback when the real RIL PDF is not available.

    Structure: Catalog -> Pages -> Page -> Font (Helvetica) -> Content stream
    """
    safe_text = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )

    stream_content = (
        f"BT\n/F1 12 Tf\n72 720 Td\n({safe_text}) Tj\nET\n"
    ).encode("latin-1")

    compressed = zlib.compress(stream_content)

    objects: List[str] = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] /Contents 5 0 R "
        "/Resources << /Font << /F1 4 0 R >> >> >>\nendobj",
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
        f"5 0 obj\n<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n",
    ]

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: List[int] = []
    for i, obj_str in enumerate(objects):
        offsets.append(buf.tell())
        if i < 4:
            buf.write(obj_str.encode("latin-1"))
            buf.write(b"\n")
        else:
            buf.write(obj_str.encode("latin-1"))
            buf.write(compressed)
            buf.write(b"\nendstream\nendobj\n")

    xref_offset = buf.tell()
    buf.write(b"xref\n")
    buf.write(f"0 {len(offsets) + 1}\n".encode())
    buf.write(b"0000000000 65535 f \n")
    for off in offsets:
        buf.write(f"{off:010d} 00000 n \n".encode())

    buf.write(b"trailer\n")
    buf.write(f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n".encode())
    buf.write(b"startxref\n")
    buf.write(f"{xref_offset}\n".encode())
    buf.write(b"%%EOF\n")

    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════
# TEST RESULT TRACKING
# ═════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float = 0.0
    detail: str = ""
    error: Optional[str] = None


# ═════════════════════════════════════════════════════════════════
# QA TESTER CLASS
# ═════════════════════════════════════════════════════════════════

class QATester:
    """
    Sequential integration test runner for the AI-CA FastAPI server.

    Each test method:
      1. Sends an HTTP request to the live endpoint
      2. Measures wall-clock latency
      3. Asserts status code + response schema
      4. Records PASS/FAIL with detail

    Shared state (doc_id, session_id) flows between tests to
    simulate a realistic user workflow.
    """

    # ── Server endpoints ──────────────────────────────────────
    #    base_url = http://localhost:9999/ai-ca  (includes root_path)
    #    API routes are prefixed with /v1/api
    #    IDs routes are at /api/ids/* (mounted at root in main.py)
    EP_HEALTH   = "/v1/api/health"
    EP_IDS      = "/api/ids/create"
    EP_UPLOAD   = "/v1/api/upload"
    EP_DATASETS = "/v1/api/datasets"
    EP_QUERY    = "/v1/api/query"
    EP_STREAM   = "/v1/api/query/stream"
    EP_METRICS  = "/v1/api/metrics"

    CLIENT_ID   = "qa_tester"
    # Generous timeout — RIL PDF is 146 pages, ingestion can take minutes
    REQUEST_TIMEOUT = 600

    # Default PDF path — the real RIL Annual Report
    DEFAULT_PDF = Path("D:/Projects2.0/Valuenaire/RIL-Integrated-Annual-Report-2024-25.pdf")

    def __init__(self, base_url: str = "http://localhost:9999/ai-ca",
                 pdf_path: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.results: List[TestResult] = []
        # Workflow state carried across tests
        self.doc_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.pdf_path: Path = Path(pdf_path) if pdf_path else self.DEFAULT_PDF
        self.using_dummy_pdf = False
        self.pdf_filename = ""
        self.pdf_size_mb = 0.0

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        """Build full URL from endpoint path."""
        return f"{self.base_url}{path}"

    def _record(self, name: str, passed: bool, latency_ms: float = 0,
                detail: str = "", error: str = ""):
        """Record test result and print coloured output."""
        self.results.append(TestResult(
            name=name, passed=passed, latency_ms=latency_ms,
            detail=detail, error=error or None,
        ))
        if passed:
            _pass(name, detail, latency_ms)
        else:
            _fail(name, error or detail, latency_ms)

    def _timed_request(self, method: str, url: str, **kwargs) -> Tuple[requests.Response, float]:
        """Execute request and return (response, latency_ms)."""
        kwargs.setdefault("timeout", self.REQUEST_TIMEOUT)
        t0 = time.perf_counter()
        resp = requests.request(method, url, **kwargs)
        latency = (time.perf_counter() - t0) * 1000
        return resp, latency

    # ─────────────────────────────────────────────────────────
    # S0  PDF PREPARATION
    # ─────────────────────────────────────────────────────────

    def prepare_pdf(self):
        """
        Resolve PDF file for testing.
        Prefers the real RIL Annual Report; falls back to dummy PDF generation.
        """
        _header("S0  PDF PREPARATION")

        if self.pdf_path.exists():
            self.pdf_size_mb = self.pdf_path.stat().st_size / (1024 * 1024)
            self.pdf_filename = self.pdf_path.name
            self.using_dummy_pdf = False
            _info(f"Using real PDF: {self.pdf_path.name}")
            _info(f"Size: {self.pdf_size_mb:.1f} MB")
            self._record("PDF file located", True,
                         detail=f"{self.pdf_filename} ({self.pdf_size_mb:.1f} MB)")
        else:
            _warn(f"PDF not found: {self.pdf_path}")
            _info("Falling back to programmatically-generated dummy PDF")
            self.using_dummy_pdf = True
            self.pdf_filename = "qa_dummy_report.pdf"
            self._record("PDF file located", True,
                         detail="Using dummy PDF (real file not found)")

    # ─────────────────────────────────────────────────────────
    # S1  HEALTH CHECK
    # ─────────────────────────────────────────────────────────

    def test_health(self):
        """
        GET /v1/api/health
        Verify the server is alive and all subsystem components report status.
        If this fails, abort immediately — no point running further tests.
        """
        _header("S1  HEALTH CHECK")
        url = self._url(self.EP_HEALTH)
        _info(f"GET {url}")

        try:
            resp, lat = self._timed_request("GET", url, timeout=10)
        except requests.ConnectionError:
            _fail("Server reachable", "Connection refused — is the server running?")
            print(f"\n  {RED}{BOLD}FATAL: Server at {self.base_url} is not reachable.{RESET}")
            print(f"  {RED}Start it with:  cd Re && python serve.py{RESET}\n")
            sys.exit(1)
        except requests.Timeout:
            _fail("Server reachable", "Connection timed out after 10s")
            sys.exit(1)

        try:
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            body = resp.json()
            # Response: {"status": "ok", "data": {"api": "ok", "llm": "...", ...}, ...}
            assert body.get("status") == "ok", f"status={body.get('status')}"
            assert body.get("data") is not None, "Missing 'data' in health response"

            data = body["data"]
            # data contains subsystem keys: api, sql_engine, vector_db, llm
            comp_detail = ", ".join(
                f"{k}={v}" for k, v in sorted(data.items()) if isinstance(v, str)
            )
            self._record("Health endpoint returns 200 OK", True, lat,
                         detail=comp_detail)
        except AssertionError as e:
            self._record("Health endpoint returns 200 OK", False, lat, error=str(e))
            print(f"\n  {RED}{BOLD}FATAL: Health check failed — aborting.{RESET}\n")
            sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # S2  ID CREATION
    # ─────────────────────────────────────────────────────────

    def test_id_creation(self):
        """
        POST /api/ids/create
        Generate a tracked session ID.
        Schema: IDCreateRequest{user_id, namespace?, meta?}
        """
        _header("S2  ID CREATION")
        url = self._url(self.EP_IDS)
        payload = {
            "user_id": self.CLIENT_ID,
            "namespace": "qa_session",
            "meta": {"source": "qa_api_tester", "run_ts": time.time()},
        }
        _info(f"POST {url}")
        _info(f"Body: {json.dumps(payload, default=str)[:120]}")

        try:
            resp, lat = self._timed_request("POST", url, json=payload, timeout=15)
        except requests.ConnectionError as e:
            self._record("ID creation endpoint", False, error=f"Connection error: {e}")
            return

        if resp.status_code == 404:
            self._record("ID creation endpoint", False, lat,
                         error="404 — IDs router not mounted in main.py")
            return

        try:
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}. Body: {resp.text[:200]}"
            body = resp.json()
            assert body.get("status") == "ok", f"status={body.get('status')}"
            data = body.get("data", {})

            # Capture session ID if returned
            if isinstance(data, dict):
                self.session_id = data.get("id") or data.get("session_id")
            detail = f"session_id={self.session_id or 'N/A'}"
            self._record("ID creation returns valid response", True, lat, detail)
        except AssertionError as e:
            self._record("ID creation returns valid response", False, lat, error=str(e))

    # ─────────────────────────────────────────────────────────
    # S3  PDF UPLOAD (CRITICAL PATH)
    # ─────────────────────────────────────────────────────────

    def test_pdf_upload(self):
        """
        POST /v1/api/upload  (multipart/form-data)
        Upload the RIL Integrated Annual Report (146 pages, 11.2 MB).

        This is the CRITICAL ingestion path — it exercises:
          - File validation & PDF type detection
          - pdfplumber text extraction -> vision cascade fallback
          - Semantic chunking (SmartChunker)
          - BM25 index update
          - Fingerprinting (SHA256 + SimHash + MinHash)
          - Disk persistence (meta JSON + chunks JSONL)
        """
        _header("S3  PDF UPLOAD (Critical Path)")

        url = self._url(self.EP_UPLOAD)
        _info(f"POST {url}")

        # Build the file payload
        if self.using_dummy_pdf:
            _info("Generating dummy PDF with financial text...")
            pdf_bytes = generate_dummy_pdf(
                "The Q3 Revenue was $50 Million. "
                "Operating Profit was $12 Million. "
                "Net Income was $8 Million for the quarter ending September 2025."
            )
            _info(f"Dummy PDF size: {len(pdf_bytes)} bytes")
            file_obj = io.BytesIO(pdf_bytes)
            filename = self.pdf_filename
        else:
            _info(f"Uploading: {self.pdf_filename} ({self.pdf_size_mb:.1f} MB)")
            _info("This may take several minutes for a 146-page PDF...")
            file_obj = open(self.pdf_path, "rb")
            filename = self.pdf_filename

        files = {
            "file": (filename, file_obj, "application/pdf"),
        }
        form_data = {
            "client_id": self.CLIENT_ID,
            "ingest_all": "true",
        }
        if self.session_id:
            form_data["chat_id"] = self.session_id

        try:
            resp, lat = self._timed_request("POST", url, files=files, data=form_data)
        except requests.ConnectionError as e:
            self._record("PDF upload endpoint reachable", False, error=str(e))
            return
        except requests.Timeout:
            self._record("PDF upload completes within timeout", False,
                         error=f"Timed out after {self.REQUEST_TIMEOUT}s — try increasing timeout")
            return
        finally:
            file_obj.close()

        # ── Assert: Status 200 ──
        try:
            assert resp.status_code == 200, (
                f"Expected 200, got {resp.status_code}. "
                f"Body: {resp.text[:500]}"
            )
            self._record("Upload returns HTTP 200", True, lat)
        except AssertionError as e:
            self._record("Upload returns HTTP 200", False, lat, error=str(e))
            # Try to extract doc_id even from error responses
            try:
                body = resp.json()
                data = body.get("data", {})
                if isinstance(data, dict) and data.get("doc_id"):
                    self.doc_id = data["doc_id"]
                    _info(f"Captured doc_id despite error: {self.doc_id}")
            except Exception:
                pass
            return

        # ── Assert: Response schema (UploadResponse) ──
        try:
            body = resp.json()
            assert "data" in body, "Missing 'data' key in response"
            data = body["data"]
            assert isinstance(data, dict), f"data is {type(data).__name__}, expected dict"

            # UploadResponse required fields
            assert "success" in data, "Missing data.success"
            assert data["success"] is True, f"data.success={data['success']}, error={data.get('error')}"
            assert "doc_id" in data, "Missing data.doc_id"
            assert data["doc_id"], "doc_id is empty"

            self.doc_id = data["doc_id"]
            datasets = data.get("datasets", [])
            detail = f"doc_id={self.doc_id}, datasets={len(datasets)}"
            self._record("Upload response schema valid (doc_id captured)", True, lat, detail)
        except AssertionError as e:
            self._record("Upload response schema valid", False, lat, error=str(e))
            # Still try to capture doc_id
            try:
                self.doc_id = body.get("data", {}).get("doc_id")
            except Exception:
                pass

        # ── Assert: Ingestion latency ──
        if lat > 0:
            # For a 146-page PDF: up to 5 minutes is acceptable
            threshold_s = 300 if not self.using_dummy_pdf else 60
            if lat < threshold_s * 1000:
                self._record(
                    "Upload latency within threshold",
                    True, lat,
                    detail=f"{lat/1000:.1f}s (< {threshold_s}s threshold)"
                )
            else:
                self._record(
                    "Upload latency within threshold",
                    False, lat,
                    error=f"{lat/1000:.1f}s exceeds {threshold_s}s threshold"
                )

    # ─────────────────────────────────────────────────────────
    # S4  DATASET VERIFICATION
    # ─────────────────────────────────────────────────────────

    def test_datasets(self):
        """
        GET /v1/api/datasets?client_id=qa_tester
        Verify the uploaded PDF appears in the dataset registry.
        Query param: client_id (str, required)
        """
        _header("S4  DATASET VERIFICATION")
        url = self._url(self.EP_DATASETS)
        params = {"client_id": self.CLIENT_ID}
        _info(f"GET {url}?client_id={self.CLIENT_ID}")

        try:
            resp, lat = self._timed_request("GET", url, params=params, timeout=30)
        except requests.ConnectionError as e:
            self._record("Datasets endpoint reachable", False, error=str(e))
            return

        # ── Assert: Status 200 ──
        try:
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            self._record("Datasets returns HTTP 200", True, lat)
        except AssertionError as e:
            self._record("Datasets returns HTTP 200", False, lat, error=str(e))
            return

        # ── Assert: Response contains our doc ──
        try:
            body = resp.json()
            assert body.get("status") == "ok", f"status={body.get('status')}"
            data = body.get("data")
            assert data is not None, "Missing 'data' in response"

            # data may be a list or dict wrapping a list
            if isinstance(data, list):
                datasets = data
            elif isinstance(data, dict):
                datasets = data.get("datasets", data.get("items", [data]))
            else:
                datasets = []

            detail = f"Found {len(datasets)} dataset(s) for client '{self.CLIENT_ID}'"
            self._record("Datasets lists client data", True, lat, detail)

            # Check if our doc_id is present
            if self.doc_id and datasets:
                found = any(self.doc_id in str(ds) for ds in datasets)
                if found:
                    self._record(
                        "Uploaded doc_id present in registry",
                        True, lat,
                        detail=f"doc_id={self.doc_id}"
                    )
                else:
                    self._record(
                        "Uploaded doc_id present in registry",
                        False, lat,
                        error=f"doc_id={self.doc_id} not found in {len(datasets)} dataset(s)"
                    )
        except AssertionError as e:
            self._record("Datasets response schema valid", False, lat, error=str(e))

    # ─────────────────────────────────────────────────────────
    # S5  SYNCHRONOUS QUERY (CRITICAL PATH)
    # ─────────────────────────────────────────────────────────

    def test_query(self):
        """
        POST /v1/api/query  (application/json)
        Ask a financial question against the uploaded RIL annual report.

        This tests the full retrieval funnel:
          L0  BM25 keyword pre-filter
          L1  Metadata filter
          L2  Lazy embed (cache-first)
          L3  Cosine similarity
          L4  Rerank (Jina/Cohere/Cross-Encoder)

        Schema: QueryRequest{client, query, dataset_id?, session_id?, chat_id?, use_cache}
        """
        _header("S5  SYNCHRONOUS QUERY (Critical Path)")

        if not self.doc_id:
            _info("Skipping — no doc_id from upload step")
            self._record("Query test skipped", False, error="No doc_id — upload likely failed")
            return

        url = self._url(self.EP_QUERY)

        # Use a query that should match known RIL Annual Report content
        query_text = (
            "What is the revenue from operations?"
            if not self.using_dummy_pdf
            else "What is the revenue?"
        )

        payload = {
            "client": self.CLIENT_ID,
            "query": query_text,
            "dataset_id": self.doc_id,
            "use_cache": False,
        }
        if self.session_id:
            payload["session_id"] = self.session_id

        _info(f"POST {url}")
        _info(f"Query: \"{query_text}\"")
        _info(f"dataset_id: {self.doc_id}")

        try:
            resp, lat = self._timed_request("POST", url, json=payload)
        except requests.ConnectionError as e:
            self._record("Query endpoint reachable", False, error=str(e))
            return
        except requests.Timeout:
            self._record("Query completes within timeout", False,
                         error=f"Timed out after {self.REQUEST_TIMEOUT}s")
            return

        # ── Assert: Status 200 ──
        try:
            assert resp.status_code == 200, (
                f"Expected 200, got {resp.status_code}. "
                f"Body: {resp.text[:500]}"
            )
            self._record("Query returns HTTP 200", True, lat)
        except AssertionError as e:
            self._record("Query returns HTTP 200", False, lat, error=str(e))
            return

        # ── Assert: QueryResponse schema ──
        try:
            body = resp.json()
            data = body.get("data", {})
            assert isinstance(data, dict), f"data is {type(data).__name__}"

            # Required fields
            assert "success" in data, "Missing data.success"
            assert "result" in data, "Missing data.result"
            assert "method" in data, "Missing data.method"

            method = data.get("method", "unknown")
            query_id = data.get("query_id", "N/A")
            result_preview = str(data.get("result", ""))[:150]
            provenance = data.get("provenance", {})

            detail = f"method={method}, query_id={query_id}"
            if provenance:
                detail += f", provenance_keys={list(provenance.keys())}"
            self._record("Query response schema valid", True, lat, detail)

            if data.get("result"):
                self._record(
                    "Query returned non-empty result",
                    True, lat,
                    detail=f"preview: \"{result_preview}...\""
                )
            else:
                self._record(
                    "Query returned non-empty result",
                    False, lat,
                    error="result is empty/null"
                )
        except AssertionError as e:
            self._record("Query response schema valid", False, lat, error=str(e))

        # ── Assert: Latency ──
        if lat > 0:
            threshold_s = 60
            if lat < threshold_s * 1000:
                self._record(
                    "Query latency acceptable",
                    True, lat,
                    detail=f"{lat/1000:.1f}s (< {threshold_s}s)"
                )
            else:
                self._record(
                    "Query latency acceptable",
                    False, lat,
                    error=f"{lat/1000:.1f}s exceeds {threshold_s}s threshold"
                )

    # ─────────────────────────────────────────────────────────
    # S6  STREAMING QUERY (SSE)
    # ─────────────────────────────────────────────────────────

    def test_stream(self):
        """
        POST /v1/api/query/stream  (SSE — text/event-stream)
        Validate Server-Sent Events response.

        Expected SSE event types from stream_service.py:
          event: start    -> {"query_id", "client", "status": "processing"}
          event: route    -> {"track", "confidence"}
          event: progress -> {"step", ...}
          event: token    -> {"token": "<word>"}
          event: complete -> {"query_id", "result", "method", "success"}
          event: error    -> {"error", "query_id"}

        Schema: StreamQueryRequest{client, query, dataset_id?, chat_id?}
        """
        _header("S6  STREAMING QUERY (SSE)")

        if not self.doc_id:
            _info("Skipping — no doc_id from upload step")
            self._record("Stream test skipped", False, error="No doc_id — upload likely failed")
            return

        url = self._url(self.EP_STREAM)

        query_text = (
            "Summarize the key financial highlights"
            if not self.using_dummy_pdf
            else "What is the quarterly revenue?"
        )

        payload = {
            "client": self.CLIENT_ID,
            "query": query_text,
            "dataset_id": self.doc_id,
        }
        if self.session_id:
            payload["chat_id"] = self.session_id

        _info(f"POST {url}")
        _info(f"Query: \"{query_text}\"")

        try:
            t0 = time.perf_counter()
            resp = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.REQUEST_TIMEOUT,
                headers={"Accept": "text/event-stream"},
            )
            connect_lat = (time.perf_counter() - t0) * 1000
        except requests.ConnectionError as e:
            self._record("Stream endpoint reachable", False, error=str(e))
            return
        except requests.Timeout:
            self._record("Stream connection within timeout", False,
                         error=f"Timed out after {self.REQUEST_TIMEOUT}s")
            return

        # ── Assert: Status 200 and correct content type ──
        try:
            assert resp.status_code == 200, (
                f"Expected 200, got {resp.status_code}. "
                f"Body: {resp.text[:300]}"
            )
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type, (
                f"Expected text/event-stream, got: {content_type}"
            )
            self._record("Stream returns 200 + text/event-stream", True, connect_lat)
        except AssertionError as e:
            self._record("Stream returns 200 + text/event-stream", False, connect_lat, error=str(e))
            resp.close()
            return

        # ── Read SSE events ──
        events_received: Dict[str, int] = {}
        data_chunks: List[str] = []
        current_event = ""
        line_count = 0
        max_lines = 2000  # Safety limit for large responses

        _info("Reading SSE event stream...")

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
                line = line.strip()
                line_count += 1

                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    events_received[current_event] = events_received.get(current_event, 0) + 1
                elif line.startswith("data:"):
                    data_payload = line[5:].strip()
                    data_chunks.append(data_payload)

                if line_count >= max_lines:
                    _info(f"Line limit reached ({max_lines}), closing stream")
                    break

                # Stop after terminal event data is received
                if current_event in ("complete", "error") and line.startswith("data:"):
                    break

        except Exception as e:
            _info(f"Stream read ended: {e}")
        finally:
            resp.close()

        total_lat = (time.perf_counter() - t0) * 1000

        # ── Assert: SSE event structure ──
        _info(f"Events: {dict(events_received)}")
        _info(f"Data chunks: {len(data_chunks)}, Lines read: {line_count}")

        # 'start' event — should always be first
        if "start" in events_received:
            self._record("SSE 'start' event received", True, total_lat)
            # Validate start payload
            try:
                start_data = json.loads(data_chunks[0]) if data_chunks else {}
                has_qid = "query_id" in start_data
                has_status = "status" in start_data
                self._record(
                    "SSE 'start' payload valid",
                    has_qid and has_status,
                    detail=f"query_id={'present' if has_qid else 'MISSING'}, "
                           f"status={start_data.get('status', 'MISSING')}"
                )
            except json.JSONDecodeError:
                self._record("SSE 'start' payload valid", False,
                             error="Failed to parse start data as JSON")
        else:
            self._record(
                "SSE 'start' event received",
                len(data_chunks) > 0,
                total_lat,
                detail="No named 'start' event, but data chunks present" if data_chunks else "",
                error="No 'start' event and no data chunks" if not data_chunks else "",
            )

        # 'route' event — routing decision
        if "route" in events_received:
            self._record("SSE 'route' event received", True, total_lat,
                         detail=f"{events_received['route']} event(s)")

        # Data chunks received (the actual content)
        if data_chunks:
            self._record(
                "SSE data chunks received",
                True, total_lat,
                detail=f"{len(data_chunks)} chunk(s)"
            )
        else:
            self._record("SSE data chunks received", False, total_lat,
                         error="No data: lines received")

        # 'token' events — the AI response words
        if "token" in events_received:
            self._record(
                "SSE 'token' events received",
                True, total_lat,
                detail=f"{events_received['token']} token event(s)"
            )

        # Terminal event — 'complete' or 'error'
        has_terminal = "complete" in events_received or "error" in events_received
        if has_terminal:
            terminal_type = "complete" if "complete" in events_received else "error"
            self._record(
                f"SSE terminal event ('{terminal_type}') received",
                True, total_lat,
            )
            # Validate complete payload
            if terminal_type == "complete" and data_chunks:
                try:
                    complete_data = json.loads(data_chunks[-1])
                    has_result = "result" in complete_data
                    has_success = "success" in complete_data
                    self._record(
                        "SSE 'complete' payload valid",
                        has_result and has_success,
                        detail=f"success={complete_data.get('success')}, "
                               f"method={complete_data.get('method', 'N/A')}"
                    )
                except json.JSONDecodeError:
                    pass
        else:
            self._record(
                "SSE terminal event received",
                False, total_lat,
                error="No 'complete' or 'error' event — stream may have been truncated"
            )

        # Total stream latency
        self._record(
            "Stream total latency",
            total_lat < 120_000,
            total_lat,
            detail=f"{total_lat/1000:.1f}s total",
        )

    # ─────────────────────────────────────────────────────────
    # S7  METRICS
    # ─────────────────────────────────────────────────────────

    def test_metrics(self):
        """
        GET /v1/api/metrics
        Verify system telemetry is recording.
        After our test hits, counters should reflect activity.
        """
        _header("S7  SYSTEM METRICS")
        url = self._url(self.EP_METRICS)
        _info(f"GET {url}")

        try:
            resp, lat = self._timed_request("GET", url, timeout=15)
        except requests.ConnectionError as e:
            self._record("Metrics endpoint reachable", False, error=str(e))
            return

        # ── Assert: Status 200 ──
        try:
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            self._record("Metrics returns HTTP 200", True, lat)
        except AssertionError as e:
            self._record("Metrics returns HTTP 200", False, lat, error=str(e))
            return

        # ── Assert: Response has data ──
        try:
            body = resp.json()
            assert body.get("status") == "ok", f"status={body.get('status')}"
            data = body.get("data")
            assert data is not None, "Missing 'data' in metrics response"

            if isinstance(data, dict):
                keys = sorted(data.keys())[:12]
                detail = f"Metrics keys: {', '.join(keys)}"
                # Check for specific counters
                total_queries = data.get("total_queries", data.get("queries", 0))
                total_uploads = data.get("total_uploads", data.get("uploads", 0))
                if total_queries or total_uploads:
                    detail += f" | queries={total_queries}, uploads={total_uploads}"
            else:
                detail = f"Metrics type: {type(data).__name__}"
            self._record("Metrics response contains telemetry data", True, lat, detail)
        except AssertionError as e:
            self._record("Metrics response schema valid", False, lat, error=str(e))

    # ─────────────────────────────────────────────────────────
    # TEST RUNNER
    # ─────────────────────────────────────────────────────────

    def run_all(self) -> int:
        """
        Execute all tests in sequence and print a summary report.
        Returns 0 if all pass, 1 if any fail.
        """
        print(f"\n{'=' * 60}")
        print(f"{BOLD}  QA API TESTER — AI-CA Integration Suite{RESET}")
        print(f"{'=' * 60}")
        print(f"  {DIM}Server:    {self.base_url}{RESET}")
        print(f"  {DIM}Client:    {self.CLIENT_ID}{RESET}")
        print(f"  {DIM}PDF:       {self.pdf_path}{RESET}")
        print(f"  {DIM}Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

        suite_start = time.perf_counter()

        # S0 — Prepare the PDF file
        self.prepare_pdf()

        # S1 — Health (fatal on failure)
        self.test_health()

        # S2 — ID creation
        self.test_id_creation()

        # S3 — PDF upload (critical path)
        self.test_pdf_upload()

        # S4 — Dataset verification
        self.test_datasets()

        # S5 — Synchronous query (critical path)
        self.test_query()

        # S6 — Streaming query (SSE)
        self.test_stream()

        # S7 — Metrics
        self.test_metrics()

        # ── Summary Report ──────────────────────────────────
        suite_time = (time.perf_counter() - suite_start) * 1000
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        print(f"\n{'=' * 60}")
        print(f"{BOLD}  SUMMARY{RESET}")
        print(f"{'=' * 60}")
        print(f"  PDF tested:   {self.pdf_filename}")
        print(f"  Total tests:  {len(self.results)}")
        print(f"  {GREEN}Passed:     {len(passed)}{RESET}")

        if failed:
            print(f"  {RED}Failed:     {len(failed)}{RESET}")
        else:
            print(f"  Failed:     0")

        print(f"  Suite time:   {suite_time/1000:.1f}s")

        if self.doc_id:
            print(f"  doc_id:       {self.doc_id}")

        if failed:
            print(f"\n  {RED}{BOLD}Failed tests:{RESET}")
            for r in failed:
                err = r.error or r.detail
                lat_str = f" ({r.latency_ms:.0f}ms)" if r.latency_ms else ""
                print(f"    {RED}x {r.name}{lat_str}: {err}{RESET}")
        else:
            print(f"\n  {GREEN}{BOLD}All tests passed!{RESET}")

        print(f"{'=' * 60}\n")

        return 0 if not failed else 1


# ═════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QA API Tester — AI-CA Integration Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python qa_api_tester.py\n"
            "  python qa_api_tester.py --base-url http://10.0.0.5:9999/ai-ca\n"
            "  python qa_api_tester.py --pdf D:/path/to/report.pdf\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:9999/ai-ca",
        help="Base URL of the AI-CA server (default: http://localhost:9999/ai-ca)",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help=(
            "Path to the PDF file to upload. "
            "Default: D:/Projects2.0/Valuenaire/RIL-Integrated-Annual-Report-2024-25.pdf"
        ),
    )
    args = parser.parse_args()

    tester = QATester(base_url=args.base_url, pdf_path=args.pdf)
    exit_code = tester.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
