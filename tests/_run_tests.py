"""Comprehensive validation of all recent modifications (Phases 1-10)."""
import sys, pathlib, inspect
sys.path.insert(0, str(pathlib.Path(__file__).parent))

results = {}

# ── 1. dll_fix ──────────────────────────────────────────────────
try:
    from app.core.dll_fix import apply_dll_fix, is_dll_fix_applied
    apply_dll_fix()
    assert is_dll_fix_applied()
    results['dll_fix.apply_dll_fix'] = '[PASS]'
except Exception as e:
    results['dll_fix.apply_dll_fix'] = f'[FAIL] {e}'

# ── 2. httpx available ──────────────────────────────────────────
try:
    import httpx
    results[f'httpx {httpx.__version__}'] = '[PASS]'
except Exception as e:
    results['httpx'] = f'[FAIL] {e}'

# ── 3. gpu_utils — httpx in is_healthy, no requests ────────────
try:
    from app.core.gpu_utils import DotsOCRManager
    src = inspect.getsource(DotsOCRManager.is_healthy)
    assert 'httpx' in src, 'httpx missing from is_healthy'
    assert 'requests' not in src, 'requests still in is_healthy'
    results['gpu_utils.is_healthy httpx'] = '[PASS]'
except Exception as e:
    results['gpu_utils.is_healthy httpx'] = f'[FAIL] {e}'

# ── 4. ingest_service._JobStore full put/get/update cycle ───────
try:
    from app.services.ingest_service import _JobStore, _JobState
    s = _JobStore()
    job = _JobState(task_id='unit-test-001', filename='test.csv', client_id='test-client')
    s.put(job)
    got = s.get('unit-test-001')
    assert got is not None and got.task_id == 'unit-test-001', f'got {got}'
    s.update('unit-test-001', status='done')
    got2 = s.get('unit-test-001')
    assert got2.status == 'done', f'status={got2.status}'
    results['ingest_service._JobStore put/get/update'] = '[PASS]'
except Exception as e:
    results['ingest_service._JobStore put/get/update'] = f'[FAIL] {e}'

# ── 5. excel_ingest — ThreadPoolExecutor present (method) ───────
try:
    from app.ingest.excel_ingest import ExcelIngestor
    src = inspect.getsource(ExcelIngestor.ingest_all_sheets)
    assert 'ThreadPoolExecutor' in src, 'ThreadPoolExecutor missing from ingest_all_sheets'
    results['excel_ingest parallel (ThreadPoolExecutor)'] = '[PASS]'
except Exception as e:
    results['excel_ingest parallel (ThreadPoolExecutor)'] = f'[FAIL] {e}'

# ── 6. ingest __init__ — survives torch DLL failure ────────────
try:
    import app.ingest as pkg
    status = 'available' if pkg.SmartPDFIngestor else 'None (torch DLL — OK)'
    results[f'ingest.__init__ SmartPDFIngestor={status}'] = '[PASS]'
except Exception as e:
    results['ingest.__init__'] = f'[FAIL] {e}'

# ── 7. llm_provider — httpx + embed/embed_batch methods ────────
try:
    src = pathlib.Path('app/core/llm_provider.py').read_text(errors='ignore')
    assert 'import httpx' in src, 'httpx not found in llm_provider.py'
    assert 'REQUESTS_AVAILABLE' not in src, 'REQUESTS_AVAILABLE still present'
    assert 'def embed(self' in src, 'embed() method missing from LLMProvider'
    assert 'def embed_batch(self' in src, 'embed_batch() method missing from LLMProvider'
    assert '_generate_embeddings_batch' in src, 'delegation to DocumentIngestor missing'
    results['llm_provider httpx + embed/embed_batch'] = '[PASS]'
except Exception as e:
    results['llm_provider httpx + embed/embed_batch'] = f'[FAIL] {e}'

# ── 8. rag/ingest — httpx + Qdrant payload index ────────────────
try:
    src = pathlib.Path('app/rag/ingest.py').read_text(errors='ignore')
    bare = [l.strip() for l in src.splitlines()
            if l.strip() in ('import requests', 'import requests ')]
    checks = [
        ('httpx.get' in src, 'httpx.get missing (Ollama/Qdrant health checks)'),
        ('httpx.post' in src, 'httpx.post missing (Jina reranker)'),
        (not bare, f'bare requests import remains: {bare}'),
        ('_ensure_payload_indexes' in src, 'Qdrant payload index method missing'),
        ('PayloadSchemaType.KEYWORD' in src, 'Keyword index type missing'),
        ('client_id' in src and 'create_payload_index' in src, 'client_id index missing'),
    ]
    failures = [msg for ok, msg in checks if not ok]
    results['rag/ingest.py httpx + payload indexes'] = '[PASS]' if not failures else f'[FAIL] {failures}'
except Exception as e:
    results['rag/ingest.py httpx + payload indexes'] = f'[FAIL] {e}'

# ── 9. llm_wrapper — httpx in jina embedding ────────────────────
try:
    src = pathlib.Path('app/core/llm_wrapper.py').read_text(errors='ignore')
    bare = [l.strip() for l in src.splitlines()
            if l.strip() in ('import requests', 'import requests ')]
    assert 'import httpx' in src, 'httpx not found in llm_wrapper.py'
    assert not bare, f'bare requests import remains: {bare}'
    results['llm_wrapper httpx jina embeddings'] = '[PASS]'
except Exception as e:
    results['llm_wrapper httpx jina embeddings'] = f'[FAIL] {e}'

# ── 10. pdf_ingest — httpx in dots.ocr + error message updated ──
try:
    src = pathlib.Path('app/ingest/pdf_ingest.py').read_text(errors='ignore')
    assert 'import httpx' in src, 'httpx not in pdf_ingest'
    assert 'httpx not installed' in src, 'error message not updated'
    assert 'requests not installed' not in src, 'old error message still present'
    results['pdf_ingest httpx dots.ocr'] = '[PASS]'
except Exception as e:
    results['pdf_ingest httpx dots.ocr'] = f'[FAIL] {e}'

# ── 11. serve.py — uvloop with platform guard ───────────────────
try:
    src = pathlib.Path('serve.py').read_text(errors='ignore')
    assert 'uvloop' in src, 'uvloop missing'
    assert "sys.platform != 'win32'" in src or 'sys.platform != "win32"' in src, 'platform guard missing'
    assert 'ImportError' in src, 'ImportError fallback missing'
    results['serve.py uvloop (platform-guarded + fallback)'] = '[PASS]'
except Exception as e:
    results['serve.py uvloop (platform-guarded + fallback)'] = f'[FAIL] {e}'

# ── 12. prompts.py — dll_fix called BEFORE langchain import ─────
try:
    lines = pathlib.Path('app/core/prompts.py').read_text(errors='ignore').splitlines()
    dll_line = next((i for i, l in enumerate(lines) if 'apply_dll_fix' in l), None)
    lc_line  = next((i for i, l in enumerate(lines) if 'from langchain_core' in l), None)
    assert dll_line is not None, 'apply_dll_fix not found in prompts.py'
    assert lc_line  is not None, 'langchain_core import not found in prompts.py'
    assert dll_line < lc_line, f'dll_fix (L{dll_line}) must come before langchain (L{lc_line})'
    results['prompts.py dll_fix BEFORE langchain import'] = '[PASS]'
except Exception as e:
    results['prompts.py dll_fix BEFORE langchain import'] = f'[FAIL] {e}'

# ── 13. no bare 'import requests' anywhere in app/ ───────────────
try:
    bad = []
    for f in pathlib.Path('app').rglob('*.py'):
        for line in f.read_text(errors='ignore').splitlines():
            stripped = line.strip()
            if stripped in ('import requests', 'import requests '):
                bad.append(str(f))
    results['no bare import requests in app/'] = '[PASS]' if not bad else f'[FAIL] found in: {bad}'
except Exception as e:
    results['no bare import requests in app/'] = f'[FAIL] {e}'

# ── 14. ingest/__init__ catches Exception (not just ImportError) ─
try:
    src = pathlib.Path('app/ingest/__init__.py').read_text(errors='ignore')
    assert 'except Exception' in src, 'still using bare except ImportError'
    assert 'except ImportError' not in src, 'old ImportError handler still present'
    results['ingest/__init__ catches Exception for pdf'] = '[PASS]'
except Exception as e:
    results['ingest/__init__ catches Exception for pdf'] = f'[FAIL] {e}'

# ── 15. LLMProvider.embed_batch runtime check ────────────────────
try:
    from app.core.llm_provider import LLMProvider
    assert hasattr(LLMProvider, 'embed'), 'LLMProvider missing embed()'
    assert hasattr(LLMProvider, 'embed_batch'), 'LLMProvider missing embed_batch()'
    assert hasattr(LLMProvider, '_get_embedder'), 'LLMProvider missing _get_embedder()'
    results['LLMProvider.embed + embed_batch methods'] = '[PASS]'
except Exception as e:
    results['LLMProvider.embed + embed_batch methods'] = f'[FAIL] {e}'

# ── 16. Qdrant payload index in _ensure_qdrant_collection source ─
try:
    from app.rag.ingest import DocumentIngestor
    src = inspect.getsource(DocumentIngestor._ensure_qdrant_collection)
    assert '_ensure_payload_indexes' in src, 'collection init does not call payload indexing'
    src2 = inspect.getsource(DocumentIngestor._ensure_payload_indexes)
    assert 'client_id' in src2, 'client_id not indexed'
    assert 'dataset_id' in src2, 'dataset_id not indexed'
    assert 'KEYWORD' in src2, 'keyword type not specified'
    results['Qdrant client_id payload index'] = '[PASS]'
except Exception as e:
    results['Qdrant client_id payload index'] = f'[FAIL] {e}'

# ── 17. pdf_ingest retrieve() injects text/content into results ─
try:
    src = pathlib.Path('app/ingest/pdf_ingest.py').read_text(errors='ignore')
    assert 'chunk_texts = self._load_chunk_texts(chunk_ids)' in src, '_load_chunk_texts not called in retrieve()'
    assert 'c["text"] = txt' in src, 'text not injected into survivors'
    assert 'c["content"] = txt' in src, 'content not injected into survivors'
    results['retrieve() injects text/content'] = '[PASS]'
except Exception as e:
    results['retrieve() injects text/content'] = f'[FAIL] {e}'

# ── 18. _get_reranker uses singleton (get_document_ingestor) ─────
try:
    from app.ingest.pdf_ingest import SmartPDFIngestor
    src = inspect.getsource(SmartPDFIngestor._get_reranker)
    assert 'get_document_ingestor' in src, '_get_reranker still creates DocumentIngestor() directly'
    assert 'DocumentIngestor()' not in src, '_get_reranker still uses direct construction'
    results['_get_reranker uses singleton'] = '[PASS]'
except Exception as e:
    results['_get_reranker uses singleton'] = f'[FAIL] {e}'

# ── 19. _push_to_qdrant uses UUID5 for point IDs ────────────────
try:
    src = pathlib.Path('app/ingest/pdf_ingest.py').read_text(errors='ignore')
    assert 'uuid.uuid5' in src, 'UUID5 conversion missing from _push_to_qdrant'
    assert 'id=point_id' in src, 'PointStruct not using point_id'
    assert 'id=cid' not in src, 'PointStruct still using raw cid as ID'
    results['_push_to_qdrant UUID5 IDs'] = '[PASS]'
except Exception as e:
    results['_push_to_qdrant UUID5 IDs'] = f'[FAIL] {e}'

# ── 20. _ensure_embeddings uses DocumentIngestor directly ────────
try:
    src = inspect.getsource(SmartPDFIngestor._ensure_embeddings_for_chunks)
    assert 'self._get_reranker()' in src, '_ensure_embeddings not using _get_reranker()'
    assert '_generate_embeddings_batch' in src, 'not calling _generate_embeddings_batch directly'
    assert 'preloaded_texts' in src, 'preloaded_texts param missing'
    results['_ensure_embeddings uses singleton'] = '[PASS]'
except Exception as e:
    results['_ensure_embeddings uses singleton'] = f'[FAIL] {e}'

# ── 21. _embed_query uses DocumentIngestor directly ──────────────
try:
    src = inspect.getsource(SmartPDFIngestor._embed_query)
    assert '_get_reranker' in src, '_embed_query not using _get_reranker()'
    assert '_generate_embedding' in src, 'not calling _generate_embedding directly'
    results['_embed_query uses singleton'] = '[PASS]'
except Exception as e:
    results['_embed_query uses singleton'] = f'[FAIL] {e}'

# ── 22. redis_client localhost fallback ───────────────────────────
try:
    src = pathlib.Path('app/core/redis_client.py').read_text(errors='ignore')
    assert 'redis://localhost:6379' in src, 'localhost fallback missing from redis_client'
    assert 'local Docker Redis' in src or 'local_client' in src, 'fallback logic missing'
    results['redis_client localhost fallback'] = '[PASS]'
except Exception as e:
    results['redis_client localhost fallback'] = f'[FAIL] {e}'

# ── 23. uuid imported in pdf_ingest ──────────────────────────────
try:
    src = pathlib.Path('app/ingest/pdf_ingest.py').read_text(errors='ignore')
    assert '\nimport uuid\n' in src, 'uuid not imported in pdf_ingest.py'
    results['pdf_ingest imports uuid'] = '[PASS]'
except Exception as e:
    results['pdf_ingest imports uuid'] = f'[FAIL] {e}'

# ═══════════════════════ Phase 10 tests ═══════════════════════════

# ── 24. query_service — PDF summary intent detection ─────────────
try:
    src = pathlib.Path('app/services/query_service.py').read_text(errors='ignore')
    assert '_SUMMARY_KEYWORDS' in src, '_SUMMARY_KEYWORDS not defined in query_service'
    assert 'summarize_dataset' in src, 'summarize_dataset not called for PDF summary'
    assert '_is_summary' in src, '_is_summary detection variable missing'
    assert 'TRACK_PDF_SUMMARY' in src, 'TRACK_PDF_SUMMARY route missing'
    results['query_service PDF summary routing'] = '[PASS]'
except Exception as e:
    results['query_service PDF summary routing'] = f'[FAIL] {e}'

# ── 25. query_service — timing instrumentation ──────────────────
try:
    src = pathlib.Path('app/services/query_service.py').read_text(errors='ignore')
    assert 'import time' in src, 'time not imported'
    assert '_t0 = time.time()' in src, 'timer start missing'
    assert 'elapsed_s' in src, 'elapsed_s not in metadata'
    results['query_service timing instrumentation'] = '[PASS]'
except Exception as e:
    results['query_service timing instrumentation'] = f'[FAIL] {e}'

# ── 26. ingest_service — PDF DataFrame registration ─────────────
try:
    src = pathlib.Path('app/services/ingest_service.py').read_text(errors='ignore')
    assert 'pdf_df = pd.DataFrame(chunk_records)' in src, 'PDF DataFrame creation missing'
    assert 'agent.register_dataframe' in src, 'register_dataframe call missing for PDF'
    assert 'chunk_records' in src, 'chunk_records list missing'
    assert 'summarize_dataset' in src or 'chunk_text' in src, 'PDF chunk text type marker missing'
    results['ingest_service PDF DataFrame registration'] = '[PASS]'
except Exception as e:
    results['ingest_service PDF DataFrame registration'] = f'[FAIL] {e}'

# ── 27. ingest_service — timing logs for all types ───────────────
try:
    src = pathlib.Path('app/services/ingest_service.py').read_text(errors='ignore')
    assert 'JSON ingest completed in' in src, 'JSON timing log missing'
    assert 'Excel/CSV ingest completed in' in src, 'Excel timing log missing'
    assert 'PDF ingest completed in' in src, 'PDF timing log missing'
    results['ingest_service timing logs'] = '[PASS]'
except Exception as e:
    results['ingest_service timing logs'] = f'[FAIL] {e}'

# ── 28. json_ingest — orjson fast path ───────────────────────────
try:
    src = pathlib.Path('app/ingest/json_ingest.py').read_text(errors='ignore')
    assert 'orjson' in src, 'orjson fast-path not added'
    assert '_loads' in src, '_loads helper missing'
    assert '_loads(json_text)' in src or '_loads(raw)' in src, '_loads not used in parse_json_text'
    results['json_ingest orjson fast-path'] = '[PASS]'
except Exception as e:
    results['json_ingest orjson fast-path'] = f'[FAIL] {e}'

# ── 29. excel_ingest — calamine engine fast-path ─────────────────
try:
    src = pathlib.Path('app/ingest/excel_ingest.py').read_text(errors='ignore')
    assert '_EXCEL_ENGINE_KWARGS' in src, 'calamine engine variable missing'
    assert 'calamine' in src, 'calamine not mentioned'
    # Verify engine kwargs used in hot-path methods
    assert '**_EXCEL_ENGINE_KWARGS' in src, '_EXCEL_ENGINE_KWARGS not spread into pd.read_excel calls'
    results['excel_ingest calamine fast-path'] = '[PASS]'
except Exception as e:
    results['excel_ingest calamine fast-path'] = f'[FAIL] {e}'

# ── 30. stream_service — PDF summary override ────────────────────
try:
    src = pathlib.Path('app/services/stream_service.py').read_text(errors='ignore')
    assert '_is_pdf_summary' in src, 'PDF summary detection missing from stream_service'
    assert 'doc_' in src, 'doc_ prefix check missing from stream_service'
    assert '_SUMMARY_KW' in src, 'summary keywords missing from stream_service'
    results['stream_service PDF summary support'] = '[PASS]'
except Exception as e:
    results['stream_service PDF summary support'] = f'[FAIL] {e}'

# ── 31. json_ingest — ThreadPoolExecutor parallel tables ─────────
try:
    src = pathlib.Path('app/ingest/json_ingest.py').read_text(errors='ignore')
    assert 'ThreadPoolExecutor' in src, 'ThreadPoolExecutor missing from json_ingest'
    assert 'as_completed' in src, 'as_completed missing from json_ingest'
    assert '_process_one_table' in src, '_process_one_table helper missing'
    results['json_ingest parallel (ThreadPoolExecutor)'] = '[PASS]'
except Exception as e:
    results['json_ingest parallel (ThreadPoolExecutor)'] = f'[FAIL] {e}'

# ── 32. json_ingest — ingest_to_rag parallelism ─────────────────
try:
    from app.ingest.json_ingest import JSONIngestor
    src = inspect.getsource(JSONIngestor.ingest_to_rag)
    assert 'ThreadPoolExecutor' in src, 'ThreadPoolExecutor missing from ingest_to_rag'
    assert '_process_rag_table' in src, 'worker function missing from ingest_to_rag'
    results['json_ingest ingest_to_rag parallel'] = '[PASS]'
except Exception as e:
    results['json_ingest ingest_to_rag parallel'] = f'[FAIL] {e}'

# ── 33. json_ingest — functional multi-table parallel test ───────
try:
    from app.ingest.json_ingest import JSONIngestor
    ingestor = JSONIngestor()
    # nested_tables structure with 3 tables
    test_data = {
        "employees": [{"name": "Alice", "salary": 100000}, {"name": "Bob", "salary": 90000}],
        "departments": [{"dept": "Eng", "budget": 500000}, {"dept": "Sales", "budget": 300000}],
        "projects": [{"project": "Alpha", "cost": 50000}],
    }
    collected = []
    def _test_cb(ds_id, df, meta):
        collected.append({"ds_id": ds_id, "rows": len(df)})
    result = ingestor.ingest_json(test_data, "test_co", "client1", register_callback=_test_cb)
    assert result["success"], f'ingest failed: {result}'
    assert len(result["datasets"]) == 3, f'expected 3 tables, got {len(result["datasets"])}'
    assert len(collected) == 3, f'expected 3 callbacks, got {len(collected)}'
    results['json_ingest multi-table functional test'] = '[PASS]'
except Exception as e:
    results['json_ingest multi-table functional test'] = f'[FAIL] {e}'

# ── 34. upload_file background_tasks covers all types ────────────
try:
    src = pathlib.Path('app/services/ingest_service.py').read_text(errors='ignore')
    assert 'background_tasks.add_task' in src, 'background_tasks.add_task missing'
    assert '_run_background' in src, '_run_background worker missing'
    # Verify _run_background calls _ingest_file_content (universal dispatch)
    from app.services.ingest_service import _run_background
    fn_src = inspect.getsource(_run_background)
    assert '_ingest_file_content' in fn_src, '_run_background does not call _ingest_file_content'
    results['upload_file background for all types'] = '[PASS]'
except Exception as e:
    results['upload_file background for all types'] = f'[FAIL] {e}'

# ── Print results ────────────────────────────────────────────────
print()
for k, v in results.items():
    print(f'  {v}  {k}')
fails = [k for k, v in results.items() if '[FAIL]' in v]
print(f'\nResult: {len(results) - len(fails)}/{len(results)} passed\n')
if fails:
    print('FAILED:')
    for f in fails:
        print(f'  - {f}: {results[f]}')
