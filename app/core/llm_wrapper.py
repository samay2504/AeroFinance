"""
LLM Wrapper - Orchestrates LLM calls with caching, retry logic, structured output, and metrics.
Integrates with llm_provider for multi-provider orchestration.
Includes structured interaction logging for observability.
"""
import logging
import logging.handlers
import time
from typing import Any, Dict, List, Optional, Union, Callable
import json
import re
import hashlib
import os
import sys
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

# Prevent transformers from loading torch which causes DLL issues on Windows
# We don't use HuggingFace models, so this is safe
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, Exception)

# =============================================================================
# STRUCTURED INTERACTION LOGGING
# =============================================================================

# Sensitive column names that should be masked in logs
SENSITIVE_COLUMNS = {'name', 'email', 'phone', 'ssn', 'pan', 'aadhaar', 'address', 'dob', 'date_of_birth'}

# Initialize interaction logger with JSONL format
_interaction_logger: Optional[logging.Logger] = None


def _get_interaction_logger() -> logging.Logger:
    """Get or create the structured interaction logger."""
    global _interaction_logger
    
    if _interaction_logger is not None:
        return _interaction_logger
    
    _interaction_logger = logging.getLogger("ai_ca.interactions")
    _interaction_logger.setLevel(logging.INFO)
    _interaction_logger.propagate = False  # Don't propagate to root logger
    
    # Create logs directory if it doesn't exist
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler for JSONL logs
    log_file = log_dir / "interaction_logs.jsonl"
    try:
        handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Use simple formatter - we'll format JSON ourselves
        handler.setFormatter(logging.Formatter('%(message)s'))
        _interaction_logger.addHandler(handler)
        logger.info(f"Interaction logger initialized: {log_file}")
    except Exception as e:
        logger.warning(f"Could not initialize interaction log file: {e}")
    
    return _interaction_logger


def _mask_pii(data: Any, sample_n: int = 3) -> Any:
    """
    Mask PII in data for logging.
    
    Args:
        data: Data to mask (dict, list, or other)
        sample_n: Max number of sample rows to include
        
    Returns:
        Masked data safe for logging
    """
    if isinstance(data, dict):
        masked = {}
        for k, v in data.items():
            key_lower = str(k).lower()
            if any(sens in key_lower for sens in SENSITIVE_COLUMNS):
                masked[k] = "<MASKED>"
            else:
                masked[k] = _mask_pii(v, sample_n)
        return masked
    elif isinstance(data, list):
        return [_mask_pii(item, sample_n) for item in data[:sample_n]]
    elif isinstance(data, str) and len(data) > 500:
        return data[:500] + "...<truncated>"
    else:
        return data


def _compute_code_hash(code: str) -> Optional[str]:
    """Compute SHA256 hash of code string."""
    if not code:
        return None
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def _upload_to_s3_if_configured(key: str, data: str) -> bool:
    """
    Attempt to upload log data to S3 if configured.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Check for S3 configuration
        bucket = os.getenv("S3_LOG_BUCKET")
        if not bucket:
            return False
        
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3')
        
        # Append to existing log file or create new one
        timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        s3_key = f"logs/{timestamp}/{key}"
        
        try:
            # Try to get existing content
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            existing = response['Body'].read().decode('utf-8')
            data = existing + data
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                raise
        
        # Upload with retry
        for attempt in range(3):
            try:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=data.encode('utf-8'),
                    ContentType='application/x-ndjson'
                )
                return True
            except ClientError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        
    except ImportError:
        logger.debug("boto3 not available for S3 logging")
    except Exception as e:
        logger.debug(f"S3 upload failed (falling back to local): {e}")
    
    return False


def log_interaction(func: Callable = None, *, include_result: bool = True):
    """
    Decorator for logging user-agent interactions with structured JSON.
    
    Fields logged:
    - log_id, timestamp_utc, env, user_id, client_id, chat_id, session_id
    - query_text, routed_agent, dataset_ids, analysis_method
    - executed_code_hash, result, result_type, provenance
    - llm_provider, llm_model, tokens_used, latency_ms
    - cache_hit, error, status
    
    Usage:
        @log_interaction
        def process_query(query: str, context: dict) -> dict:
            ...
            
        @log_interaction(include_result=False)  # Don't log result
        def sensitive_operation(...):
            ...
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from app.core.id_generator import generate_short_id, get_iso_timestamp
            
            start_time = time.time()
            
            # Extract context from kwargs or args
            ctx = kwargs.get("context", {}) or {}
            if not ctx and len(args) > 1 and isinstance(args[1], dict):
                ctx = args[1]
            
            # Build base log document
            log_doc = {
                "log_id": generate_short_id("log"),
                "timestamp_utc": get_iso_timestamp(),
                "env": os.getenv("ENV", "dev"),
                "user_id": ctx.get("user_id"),
                "client_id": ctx.get("client_id"),
                "chat_id": ctx.get("chat_id"),
                "session_id": ctx.get("session_id"),
                "query_text": ctx.get("query") or ctx.get("query_text") or (args[0] if args else None),
                "routed_agent": ctx.get("agent") or ctx.get("routed_agent"),
                "dataset_ids": ctx.get("dataset_ids", []),
                "dataset_match_scores": ctx.get("match_scores"),
                "selected_dataset_id": ctx.get("dataset_id") or ctx.get("selected_dataset_id"),
                "selected_sheet_name": ctx.get("sheet_name"),
                "analysis_method": None,
                "executed_code_hash": None,
                "result": None,
                "result_type": None,
                "provenance": ctx.get("provenance", []),
                "llm_provider": ctx.get("llm_provider"),
                "llm_model": ctx.get("llm_model"),
                "tokens_used": ctx.get("tokens_used"),
                "latency_ms": None,
                "cache_hit": ctx.get("cache_hit", False),
                "error": None,
                "status": "success"
            }
            
            result = None
            try:
                result = fn(*args, **kwargs)
                
                # Extract result metadata
                elapsed_ms = int((time.time() - start_time) * 1000)
                log_doc["latency_ms"] = elapsed_ms
                
                if isinstance(result, dict):
                    log_doc["analysis_method"] = result.get("method")
                    log_doc["cache_hit"] = result.get("cache_hit", log_doc["cache_hit"])
                    
                    if include_result:
                        # Mask PII in result
                        result_for_log = result.get("result") or result.get("value")
                        log_doc["result"] = _mask_pii(result_for_log)
                    
                    # Determine result type
                    actual_result = result.get("result") or result.get("value")
                    if actual_result is not None:
                        if isinstance(actual_result, (int, float)):
                            log_doc["result_type"] = "numeric"
                        elif isinstance(actual_result, dict) and "value" in actual_result:
                            log_doc["result_type"] = "structured"
                        else:
                            log_doc["result_type"] = "text"
                    
                    # Extract code hash if code was executed
                    code = result.get("code") or result.get("executed_code")
                    if code:
                        log_doc["executed_code_hash"] = _compute_code_hash(code)
                    
                    # Extract provenance
                    if result.get("provenance"):
                        log_doc["provenance"] = _mask_pii(result["provenance"])
                        
                    # Check for fallback status
                    if result.get("fallback") or result.get("error"):
                        log_doc["status"] = "fallbacked" if result.get("result") else "failed"
                        log_doc["error"] = result.get("error")
                
                log_doc["status"] = log_doc.get("status", "success")
                
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                log_doc["latency_ms"] = elapsed_ms
                log_doc["status"] = "failed"
                log_doc["error"] = str(e)
                raise
            
            finally:
                # Write log
                try:
                    log_line = json.dumps(log_doc, default=str) + "\n"
                    
                    # Try S3 first, fallback to local
                    if not _upload_to_s3_if_configured("interaction_logs.jsonl", log_line):
                        # Local fallback
                        interaction_logger = _get_interaction_logger()
                        interaction_logger.info(log_line.rstrip())
                        
                except Exception as log_error:
                    logger.warning(f"Failed to write interaction log: {log_error}")
            
            return result
        
        return wrapper
    
    # Handle both @log_interaction and @log_interaction(include_result=False)
    if func is not None:
        return decorator(func)
    return decorator


def build_provenance(
    dataset_id: str,
    sheet_name: Optional[str] = None,
    row_index: Optional[int] = None,
    column: Optional[str] = None,
    row_sample: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Build a provenance object for tracking data lineage.
    
    Args:
        dataset_id: The dataset identifier
        sheet_name: Optional sheet name
        row_index: Optional row index
        column: Optional column name
        row_sample: Optional sample rows (will be masked for PII)
        
    Returns:
        Provenance dict suitable for logging
    """
    provenance = {
        "dataset_id": dataset_id,
    }
    if sheet_name:
        provenance["sheet"] = sheet_name
    if row_index is not None:
        provenance["row_index"] = row_index
    if column:
        provenance["column"] = column
    if row_sample:
        provenance["row_sample"] = _mask_pii(row_sample, sample_n=3)
    
    return provenance


# =============================================================================
# LOG RETENTION & TTL PURGE SYSTEM
# =============================================================================
# Tiered retention strategy for CA-grade compliance and fine-tuning data quality.
# Config-driven, no new services required.

# Default retention configuration (can be overridden via env vars)
LOG_RETENTION_CONFIG = {
    # HIGH-VALUE: Gold-quality fine-tuning data (long retention)
    "high_value": {
        "retention_days": int(os.getenv("LOG_HIGH_VALUE_RETENTION_DAYS", "365")),  # 12 months
        "fields": ["doc_id", "dataset_id", "user_query", "final_response", 
                   "sql_executed", "tools_used", "confidence", "method", "provenance"],
        "s3_glacier_transition_days": 90,
        "compress": True,
    },
    # MEDIUM-VALUE: Operational logs (short retention)
    "medium_value": {
        "retention_days": int(os.getenv("LOG_MEDIUM_VALUE_RETENTION_DAYS", "14")),  # 2 weeks
        "fields": ["router_decision", "dataset_match_scores", "prompt_version",
                   "llm_provider", "llm_model", "tokens_used", "latency_ms"],
        "compress": False,
    },
    # LOW-VALUE: Debug/ephemeral (aggressive purge)
    "low_value": {
        "retention_hours": int(os.getenv("LOG_LOW_VALUE_RETENTION_HOURS", "72")),  # 3 days
        "fields": ["intermediate_reasoning", "retry_attempts", "debug_trace"],
        "purge_on_startup": True,
    },
    # Redis hot logs (if enabled)
    "redis": {
        "ttl_seconds": int(os.getenv("LOG_REDIS_TTL_SECONDS", "3600")),  # 1 hour
    },
    # S3 upload settings
    "s3": {
        "batch_upload_interval_hours": int(os.getenv("LOG_S3_BATCH_INTERVAL_HOURS", "24")),
        "bucket": os.getenv("S3_LOG_BUCKET"),
        "prefix": os.getenv("S3_LOG_PREFIX", "logs/ai-ca"),
    }
}


def classify_log_value(log_entry: Dict[str, Any]) -> str:
    """
    Classify a log entry into high/medium/low value tier.
    
    Classification rules:
    - HIGH: Contains user query + response + provenance (fine-tuning quality)
    - MEDIUM: Contains operational metrics without full response
    - LOW: Debug/trace data
    """
    has_query = bool(log_entry.get("query_text"))
    has_result = log_entry.get("result") is not None
    has_provenance = bool(log_entry.get("provenance"))
    status = log_entry.get("status", "")
    
    # HIGH VALUE: Successful queries with results and provenance
    if has_query and has_result and has_provenance and status == "success":
        return "high_value"
    
    # HIGH VALUE: Even failed queries with SQL/code executed (learning data)
    if log_entry.get("executed_code_hash") or log_entry.get("analysis_method"):
        return "high_value"
    
    # MEDIUM VALUE: Has operational metrics
    if log_entry.get("latency_ms") or log_entry.get("tokens_used"):
        return "medium_value"
    
    # LOW VALUE: Everything else
    return "low_value"


def extract_high_value_fields(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only high-value fields for long-term archival.
    Reduces storage while preserving fine-tuning quality.
    """
    high_value_fields = LOG_RETENTION_CONFIG["high_value"]["fields"]
    
    extracted = {
        "log_id": log_entry.get("log_id"),
        "timestamp_utc": log_entry.get("timestamp_utc"),
        "client_id": log_entry.get("client_id"),
    }
    
    # Field mapping (log field -> archive field)
    field_mapping = {
        "query_text": "user_query",
        "result": "final_response", 
        "analysis_method": "method",
        "executed_code_hash": "code_hash",
    }
    
    for log_field, archive_field in field_mapping.items():
        if log_entry.get(log_field) is not None:
            extracted[archive_field] = log_entry[log_field]
    
    # Direct copy fields
    for field in ["dataset_id", "provenance", "llm_provider", "llm_model", 
                  "latency_ms", "status", "error"]:
        if log_entry.get(field) is not None:
            extracted[field] = log_entry[field]
    
    return extracted


def purge_expired_logs(
    log_dir: Optional[Path] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Time-based purge of local log files based on retention policy.
    
    This is a SCHEDULED FUNCTION - call from cron/scheduler, not on every request.
    
    Args:
        log_dir: Log directory (default: data/logs)
        dry_run: If True, report what would be deleted without deleting
        
    Returns:
        Dict with purge statistics
    """
    from datetime import timedelta
    import glob
    
    log_dir = log_dir or Path("data/logs")
    now = datetime.utcnow()
    
    stats = {
        "files_checked": 0,
        "files_purged": 0,
        "bytes_freed": 0,
        "errors": [],
        "dry_run": dry_run,
    }
    
    # Define retention thresholds
    high_value_cutoff = now - timedelta(days=LOG_RETENTION_CONFIG["high_value"]["retention_days"])
    medium_value_cutoff = now - timedelta(days=LOG_RETENTION_CONFIG["medium_value"]["retention_days"])
    low_value_cutoff = now - timedelta(hours=LOG_RETENTION_CONFIG["low_value"]["retention_hours"])
    
    # Process daily log directories (format: YYYY-MM-DD)
    for date_dir in log_dir.glob("????-??-??"):
        if not date_dir.is_dir():
            continue
        
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
        except ValueError:
            continue
        
        # Determine if entire directory can be purged based on age
        if dir_date < high_value_cutoff:
            # Very old - purge entire directory
            if not dry_run:
                import shutil
                size = sum(f.stat().st_size for f in date_dir.rglob("*") if f.is_file())
                shutil.rmtree(date_dir)
                stats["bytes_freed"] += size
            stats["files_purged"] += 1
            logger.info(f"Purged old log directory: {date_dir}")
    
    # Process individual log files
    for log_file in log_dir.glob("*.jsonl"):
        stats["files_checked"] += 1
        
        try:
            # Get file modification time
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            file_size = log_file.stat().st_size
            
            # Check file name for value tier hint
            filename = log_file.name.lower()
            
            if "debug" in filename or "trace" in filename:
                cutoff = low_value_cutoff
            elif "interaction" in filename:
                # Main interaction logs - use medium retention for local copy
                # High-value entries should already be archived to S3
                cutoff = medium_value_cutoff
            else:
                cutoff = medium_value_cutoff
            
            if mtime < cutoff:
                if not dry_run:
                    log_file.unlink()
                    stats["bytes_freed"] += file_size
                stats["files_purged"] += 1
                logger.info(f"Purged expired log: {log_file.name}")
                
        except Exception as e:
            stats["errors"].append(f"{log_file}: {str(e)}")
    
    logger.info(f"Log purge complete: {stats['files_purged']} files, {stats['bytes_freed']/1024:.1f}KB freed")
    return stats


def archive_high_value_logs_to_s3(
    log_dir: Optional[Path] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Batch upload high-value log entries to S3 for long-term archival.
    
    This function:
    1. Reads local interaction logs from previous day(s)
    2. Extracts high-value entries only
    3. Compresses and uploads to S3 with proper partitioning
    4. Marks local files as "archived" (or deletes if configured)
    
    Should be run on a schedule (daily/hourly).
    
    Args:
        log_dir: Log directory (default: data/logs)
        force: If True, process today's logs too (not recommended)
        
    Returns:
        Dict with upload statistics
    """
    import gzip
    
    s3_config = LOG_RETENTION_CONFIG["s3"]
    bucket = s3_config["bucket"]
    
    if not bucket:
        return {"status": "skipped", "reason": "S3_LOG_BUCKET not configured"}
    
    log_dir = log_dir or Path("data/logs")
    now = datetime.utcnow()
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    
    stats = {
        "entries_processed": 0,
        "high_value_archived": 0,
        "bytes_uploaded": 0,
        "s3_keys": [],
        "errors": [],
    }
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        s3_client = boto3.client('s3')
    except ImportError:
        return {"status": "error", "reason": "boto3 not installed"}
    
    # Find log files to process
    main_log = log_dir / "interaction_logs.jsonl"
    files_to_process = []
    
    if main_log.exists():
        files_to_process.append(main_log)
    
    # Also check dated directories
    for date_dir in log_dir.glob("????-??-??"):
        if date_dir.name > yesterday and not force:
            continue  # Skip today's logs
        for log_file in date_dir.glob("*.jsonl"):
            files_to_process.append(log_file)
    
    if not files_to_process:
        return {"status": "success", "message": "No logs to archive"}
    
    # Process and collect high-value entries
    high_value_entries = []
    
    for log_file in files_to_process:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        stats["entries_processed"] += 1
                        
                        # Classify and filter
                        value_tier = classify_log_value(entry)
                        if value_tier == "high_value":
                            archived_entry = extract_high_value_fields(entry)
                            high_value_entries.append(archived_entry)
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            stats["errors"].append(f"{log_file}: {str(e)}")
    
    if not high_value_entries:
        return {"status": "success", "message": "No high-value entries to archive", **stats}
    
    # Compress and upload
    stats["high_value_archived"] = len(high_value_entries)
    
    # Create JSONL content
    jsonl_content = "\n".join(json.dumps(e, default=str) for e in high_value_entries)
    compressed_content = gzip.compress(jsonl_content.encode('utf-8'))
    stats["bytes_uploaded"] = len(compressed_content)
    
    # S3 key with date partitioning (Hive-style for Athena compatibility)
    s3_key = f"{s3_config['prefix']}/year={now.year}/month={now.month:02d}/day={now.day:02d}/interactions_{now.strftime('%H%M%S')}.jsonl.gz"
    
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=compressed_content,
            ContentType='application/gzip',
            ContentEncoding='gzip',
            Metadata={
                'entry_count': str(len(high_value_entries)),
                'source': 'ai-ca-interaction-logs',
                'retention_tier': 'high_value',
            }
        )
        stats["s3_keys"].append(s3_key)
        logger.info(f"Archived {len(high_value_entries)} high-value logs to s3://{bucket}/{s3_key}")
        
    except ClientError as e:
        stats["errors"].append(f"S3 upload failed: {str(e)}")
        return {"status": "error", **stats}
    
    return {"status": "success", **stats}


def setup_s3_lifecycle_rules(bucket: Optional[str] = None) -> Dict[str, Any]:
    """
    Configure S3 lifecycle rules for log retention.
    
    Creates rules for:
    1. Transition to Glacier after 90 days
    2. Delete after retention period (configurable)
    
    This only needs to be run ONCE per bucket (or when policy changes).
    """
    bucket = bucket or LOG_RETENTION_CONFIG["s3"]["bucket"]
    
    if not bucket:
        return {"status": "error", "reason": "No S3 bucket configured"}
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        s3_client = boto3.client('s3')
    except ImportError:
        return {"status": "error", "reason": "boto3 not installed"}
    
    retention_days = LOG_RETENTION_CONFIG["high_value"]["retention_days"]
    glacier_days = LOG_RETENTION_CONFIG["high_value"]["s3_glacier_transition_days"]
    prefix = LOG_RETENTION_CONFIG["s3"]["prefix"]
    
    lifecycle_config = {
        'Rules': [
            {
                'ID': 'ai-ca-log-glacier-transition',
                'Prefix': f'{prefix}/',
                'Status': 'Enabled',
                'Transitions': [
                    {
                        'Days': glacier_days,
                        'StorageClass': 'GLACIER'
                    }
                ],
            },
            {
                'ID': 'ai-ca-log-expiration',
                'Prefix': f'{prefix}/',
                'Status': 'Enabled',
                'Expiration': {
                    'Days': retention_days
                },
            },
        ]
    }
    
    try:
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket,
            LifecycleConfiguration=lifecycle_config
        )
        logger.info(f"S3 lifecycle rules configured for bucket: {bucket}")
        return {
            "status": "success",
            "bucket": bucket,
            "glacier_transition_days": glacier_days,
            "expiration_days": retention_days,
        }
        
    except ClientError as e:
        return {"status": "error", "reason": str(e)}


def run_log_maintenance(
    purge_local: bool = True,
    archive_to_s3: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Combined maintenance function for scheduled execution.
    
    Run this from a scheduler (cron, celery, etc.) or at application startup.
    
    Args:
        purge_local: Run local log purge
        archive_to_s3: Archive high-value logs to S3
        dry_run: Don't actually delete/upload, just report
        
    Returns:
        Combined results from all operations
    """
    results = {"timestamp": datetime.utcnow().isoformat() + "Z"}
    
    if archive_to_s3:
        try:
            results["archive"] = archive_high_value_logs_to_s3()
        except Exception as e:
            results["archive"] = {"status": "error", "reason": str(e)}
    
    if purge_local:
        try:
            results["purge"] = purge_expired_logs(dry_run=dry_run)
        except Exception as e:
            results["purge"] = {"status": "error", "reason": str(e)}
    
    logger.info(f"Log maintenance completed: {json.dumps(results, default=str)[:200]}...")
    return results


# DataAnalystResult schema for structured output
DataAnalystResult = {
    "value": (int, float, str, type(None)),
    "method": str,  # sql_duckdb, pandas, llm_direct, summary
    "provenance": list,  # List of provenance dicts
    "code": (str, type(None)),  # Executed code hash

    "exec_time_ms": int
}




class LLMWrapper:
    """High-level wrapper around LLM providers with caching and structured output."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._failed_providers: set = set()
        self._call_count = 0
        self._total_latency_ms = 0
        self._cache_hits = 0

        # Initialize LLM provider
        try:
            from .llm_provider import LLMProvider
            self._llm_provider = LLMProvider(config)
            self.llm = self._llm_provider.llm
            self.provider_name = self._llm_provider.current_provider
        except Exception as e:
            logger.error(f"LLM provider initialization failed: {e}")
            self._llm_provider = None
            self.llm = None
            self.provider_name = "fallback"

        # Initialize cache (Redis or in-memory)
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = config.get("cache_enabled", True)
        self._redis_client = None
        
        try:
            if config.get("redis_enabled"):
                import redis
                redis_url = config.get("redis_url", "redis://localhost:6379/0")
                self._redis_client = redis.from_url(redis_url)
                self._redis_client.ping()
                logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory cache: {e}")

    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        hash_val = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_val}"

    def _get_cached(self, key: str) -> Optional[str]:
        """Get from cache."""
        if not self._cache_enabled:
            return None
        if self._redis_client:
            try:
                val = self._redis_client.get(key)
                return val.decode() if val else None
            except Exception:
                pass
        return self._cache.get(key)

    def _set_cached(self, key: str, value: str, ttl: int = 1800):
        """Set in cache."""
        if not self._cache_enabled:
            return
        if self._redis_client:
            try:
                self._redis_client.setex(key, ttl, value)
                return
            except Exception:
                pass
        self._cache[key] = value

    def _attempt_provider_fallback(self) -> bool:
        """Attempt to switch to fallback provider."""
        if not self._llm_provider:
            return False

        current = self.provider_name
        if current:
            base = str(current).split("_")[0]
            self._failed_providers.add(base)

        provider_preference = self.config.get(
            "provider_preference",
            ["groq", "google_genai", "ollama", "openrouter", "openai", "fallback"]
        )

        for provider_name in provider_preference:
            if provider_name in self._failed_providers:
                continue
            if current and provider_name in str(current):
                continue

            try:
                from .llm_provider import LLMProvider
                fallback_config = self.config.copy()
                fallback_config["provider_preference"] = [provider_name]
                fallback_config["strict_provider_list"] = True
                fallback_config["provider_blacklist"] = list(self._failed_providers)

                new_provider = LLMProvider(fallback_config)
                if new_provider.llm and new_provider.current_provider != "fallback":
                    self._llm_provider = new_provider
                    self.llm = new_provider.llm
                    self.provider_name = new_provider.current_provider
                    logger.info(f"✅ Switched to: {self.provider_name}")
                    return True
                else:
                    self._failed_providers.add(provider_name)
            except Exception as e:
                logger.warning(f"Fallback to {provider_name} failed: {e}")
                self._failed_providers.add(provider_name)

        return False

    def get_provider_info(self) -> Dict[str, Any]:
        """Get current provider info."""
        if self._llm_provider:
            return self._llm_provider.get_provider_info()
        return {"provider": "unavailable", "available": False, "fallback_mode": True}

    def _format_prompt(self, prompt: Union[str, PromptTemplate], **kwargs) -> str:
        """Safely format prompt with template variables."""
        if isinstance(prompt, PromptTemplate):
            try:
                return prompt.format(**kwargs)
            except (KeyError, ValueError):
                template_text = getattr(prompt, "template", str(prompt))
                for key, value in kwargs.items():
                    template_text = template_text.replace(f"{{{key}}}", str(value))
                return template_text
        return str(prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    )
    def invoke(self, prompt: Union[str, PromptTemplate], use_cache: bool = True, **kwargs) -> str:
        """Invoke LLM with prompt."""
        if not self.llm:
            return "Error: LLM provider unavailable"

        try:
            formatted_prompt = self._format_prompt(prompt, **kwargs)

            # Check cache
            if use_cache:
                cache_key = self._cache_key("llm", prompt=formatted_prompt, provider=self.provider_name)
                cached = self._get_cached(cache_key)
                if cached:
                    self._cache_hits += 1
                    return cached

            # Invoke LLM
            start_time = time.time()
            response = self.llm.invoke(formatted_prompt)
            elapsed_ms = (time.time() - start_time) * 1000

            self._call_count += 1
            self._total_latency_ms += elapsed_ms

            # Extract content
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)

            # Cache result
            if use_cache:
                self._set_cached(cache_key, result)

            logger.debug(f"LLM invoke: {elapsed_ms:.0f}ms ({self.provider_name})")
            return result

        except Exception as e:
            error_str = str(e)
            is_quota_error = any(
                x in error_str.lower()
                for x in ["429", "quota", "rate", "exhausted", "exceeded"]
            )

            if is_quota_error:
                logger.warning(f"⚠️ Rate limit: {error_str[:100]}")
                if self._attempt_provider_fallback():
                    return self.invoke(prompt, use_cache=use_cache, **kwargs)
                return "Error: Rate limit exceeded, no fallback available"

            logger.error(f"LLM invoke failed: {e}")
            return f"Error: {str(e)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    )
    def invoke_with_structured_output(
        self, prompt: Union[str, PromptTemplate], output_schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Invoke LLM expecting structured JSON output."""
        if not self.llm:
            return {"error": "LLM provider unavailable"}

        try:
            formatted_prompt = self._format_prompt(prompt, **kwargs)

            # Add JSON instruction
            def serialize_type(o):
                if isinstance(o, type):
                    return o.__name__
                return str(o)

            try:
                schema_str = json.dumps(output_schema, indent=2, default=serialize_type)
            except Exception:
                schema_str = str(output_schema)

            json_instruction = (
                f"\n\nRESPOND WITH ONLY VALID JSON matching this schema "
                f"(no markdown, no code blocks):\n{schema_str}"
            )
            if "json" not in formatted_prompt.lower():
                formatted_prompt += json_instruction

            # Invoke
            start_time = time.time()
            response = self.llm.invoke(formatted_prompt)
            elapsed_ms = (time.time() - start_time) * 1000

            self._call_count += 1
            self._total_latency_ms += elapsed_ms

            # Extract content
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Clean markdown code blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            if not content:
                return {"error": "Empty response from LLM", "fallback": True}

            # Parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

                # Try Python dict literal
                try:
                    import ast
                    result = ast.literal_eval(content)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    pass

                # Extract Python code block
                code_match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
                if code_match:
                    return {
                        "code": code_match.group(1).strip(),
                        "explanation": "Extracted from code block"
                    }

                return {"error": "Invalid JSON", "raw_content": content[:500], "fallback": True}

        except Exception as e:
            error_str = str(e)
            is_quota_error = any(
                x in error_str.lower()
                for x in ["429", "quota", "rate", "exhausted", "exceeded"]
            )

            if is_quota_error:
                if self._attempt_provider_fallback():
                    return self.invoke_with_structured_output(prompt, output_schema, **kwargs)
                return {"error": "Rate limit exceeded", "fallback": True}

            logger.error(f"Structured invoke failed: {e}")
            return {"error": str(e), "fallback": True}

    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry metrics."""
        avg_latency = self._total_latency_ms / max(1, self._call_count)
        return {
            "provider": self.provider_name,
            "total_calls": self._call_count,
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hits": self._cache_hits,
        }


    def reset_metrics(self):
        """Reset telemetry counters."""
        self._call_count = 0
        self._total_latency_ms = 0
        self._cache_hits = 0


# Try to import PandasAI's base LLM class for proper inheritance
_PandasAI_LLM_Base = None
try:
    from pandasai.llm.base import LLM as _PandasAI_LLM_Base
except ImportError:
    pass


class PandasAILLMAdapter(_PandasAI_LLM_Base if _PandasAI_LLM_Base else object):
    """
    Adapter class to make LLMWrapper compatible with PandasAI 3.0.
    
    PandasAI 3.0 expects an LLM object that inherits from pandasai.llm.base.LLM.
    This adapter wraps our multi-provider LLMWrapper to work with PandasAI.
    """
    
    def __init__(self, llm_wrapper: LLMWrapper):
        """
        Initialize adapter with existing LLMWrapper.
        
        Args:
            llm_wrapper: The existing LLMWrapper instance
        """
        # Initialize base class if it exists
        if _PandasAI_LLM_Base:
            super().__init__()
        self._wrapper = llm_wrapper
        self._model = llm_wrapper.provider_name or "gemini"
        
    @property
    def model(self) -> str:
        """Return model name for PandasAI."""
        return self._model
    
    @property
    def type(self) -> str:
        """Return LLM type for PandasAI."""
        return "custom"
    
    def __call__(self, instruction: str, context: str = "", suffix: str = "") -> str:
        """PandasAI 3.0 may call the LLM directly."""
        return self.call(instruction, context, suffix)
        
    def call(self, instruction: str, context: str = "", suffix: str = "") -> str:
        """
        PandasAI calls this method for LLM interaction.
        
        Args:
            instruction: The instruction/prompt
            context: Optional context
            suffix: Optional suffix
            
        Returns:
            LLM response as string
        """
        # Combine instruction with context
        full_prompt = instruction
        if context:
            full_prompt = f"{context}\n\n{instruction}"
        if suffix:
            full_prompt = f"{full_prompt}\n\n{suffix}"
            
        try:
            response = self._wrapper.invoke(full_prompt, use_cache=True)
            return str(response)
        except Exception as e:
            logger.warning(f"PandasAI LLM call failed: {e}")
            return f"Error: {e}"
    
    def complete(self, prompt: str) -> str:
        """
        Simple completion method for PandasAI 3.0.
        
        Args:
            prompt: The prompt to complete
            
        Returns:
            LLM completion as string
        """
        try:
            response = self._wrapper.invoke(prompt, use_cache=False)
            return str(response)
        except Exception as e:
            logger.warning(f"PandasAI complete failed: {e}")
            return f"Error: {e}"
    
    def chat_completion(self, messages: list) -> str:
        """
        Handle chat completion format used by some PandasAI versions.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            LLM response as string
        """
        # Flatten messages into a prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.insert(0, content)
            else:
                prompt_parts.append(content)
        
        full_prompt = "\n\n".join(prompt_parts)
        
        try:
            response = self._wrapper.invoke(full_prompt, use_cache=True)
            return str(response)
        except Exception as e:
            logger.warning(f"PandasAI chat completion failed: {e}")
            return f"Error: {e}"
    
    def generate_code(self, instruction: str, context: str = "") -> str:
        """
        Generate Python code - used by PandasAI for code generation.
        
        Args:
            instruction: Code generation instruction
            context: Data context
            
        Returns:
            Generated Python code
        """
        prompt = f"""Generate Python code to answer this question about the data.
        
DATA CONTEXT:
{context}

INSTRUCTION: {instruction}

Return ONLY valid Python code that works with a pandas DataFrame named 'df'.
The code should print or return the final result."""

        try:
            response = self._wrapper.invoke(prompt, use_cache=False)
            # Extract code from response if wrapped in markdown
            code = str(response)
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            logger.warning(f"PandasAI code generation failed: {e}")
            return f"# Error: {e}"


def create_pandasai_llm_adapter(llm_wrapper: LLMWrapper = None) -> PandasAILLMAdapter:
    """
    Create a PandasAI-compatible LLM adapter.
    
    Args:
        llm_wrapper: Optional LLMWrapper instance. If None, uses singleton.
        
    Returns:
        PandasAILLMAdapter instance
    """
    if llm_wrapper is None:
        llm_wrapper = get_llm_wrapper()
    return PandasAILLMAdapter(llm_wrapper)


# Singleton instance
_llm_wrapper: Optional[LLMWrapper] = None


def get_llm_wrapper(config: Optional[Dict[str, Any]] = None) -> LLMWrapper:
    """Get or create singleton LLM wrapper."""
    global _llm_wrapper

    if _llm_wrapper is None:
        if config is None:
            from app.config import settings
            config = {
                "provider_preference": settings.llm.provider_preference,
                "temperature": settings.llm.temperature,
                "max_retries": settings.llm.max_retries,
                "redis_enabled": settings.cache.redis_enabled,
                "redis_url": settings.cache.redis_url,
            }
        _llm_wrapper = LLMWrapper(config)

    return _llm_wrapper


__all__ = [
    "LLMWrapper", 
    "get_llm_wrapper", 
    "PandasAILLMAdapter", 
    "create_pandasai_llm_adapter",
    # Structured logging
    "log_interaction",
    "build_provenance",
    "DataAnalystResult",
    "_mask_pii",
    "_compute_code_hash",
    # Log retention & TTL purge
    "LOG_RETENTION_CONFIG",
    "classify_log_value",
    "extract_high_value_fields",
    "purge_expired_logs",
    "archive_high_value_logs_to_s3",
    "setup_s3_lifecycle_rules",
    "run_log_maintenance",
]



