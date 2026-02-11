"""
Data Registry - Production-grade storage abstraction with multi-backend support.

Storage Backends:
- LocalStorageBackend: Filesystem + LRU memory cache (development)
- S3StorageBackend: AWS S3 + request-scope cache (production)

Architecture follows hybrid storage best practices:
1. Abstract interface for backend-agnostic operations
2. Hot LRU cache layer for frequently accessed data
3. Redis for metadata and cross-instance state
4. Environment-driven backend selection

References:
- AWS Storage Gateway patterns for hybrid S3/local storage
- LRU cache with write-through for consistency
- Multi-tenant isolation via client_id on all paths
"""
import logging
import json
import hashlib
import os
import io
import re
import threading
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# BLOOM FILTER — Zero-Latency Probabilistic Existence Checks
# =============================================================================

class BloomFilter:
    """
    Probabilistic set for O(1) "definitely not in set" checks.

    Used to skip expensive I/O (S3 or disk) when a dataset definitely
    does not exist. False positives are possible (proceed to I/O check),
    but false negatives are impossible.

    Implementation:
      - Uses Python `int` as a bit vector (no external deps)
      - k=7 hash functions via double-hashing (SHA256 + MD5)
      - Default capacity=10000, fp_rate ≈ 1%

    Thread Safety:
      All mutations are protected by threading.Lock.

    Reference:
      Kirsch & Mitzenmacher (2006) - "Less Hashing, Same Performance"
    """

    def __init__(self, capacity: int = 10_000, fp_rate: float = 0.01):
        """
        Args:
            capacity: Expected max number of items
            fp_rate:  Target false-positive rate (0.01 = 1%)
        """
        # Optimal bit array size: m = -(n * ln(p)) / (ln(2)^2)
        self._m = max(64, int(-capacity * math.log(fp_rate) / (math.log(2) ** 2)))
        # Optimal hash count: k = (m/n) * ln(2)
        self._k = max(1, int((self._m / max(capacity, 1)) * math.log(2)))
        # Bit vector stored as Python int (arbitrary precision — no size limit)
        self._bits: int = 0
        self._count: int = 0
        self._lock = threading.Lock()

        logger.debug(
            f"BloomFilter initialized: m={self._m} bits, k={self._k} hashes, "
            f"capacity={capacity}, target_fp={fp_rate}"
        )

    def _hashes(self, key: str) -> List[int]:
        """
        Generate k hash positions using double-hashing.

        h_i(key) = (h1(key) + i * h2(key)) % m
        This avoids computing k independent hash functions.
        """
        h1 = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, key: str) -> None:
        """Add a key to the filter."""
        positions = self._hashes(key)
        with self._lock:
            for pos in positions:
                self._bits |= (1 << pos)
            self._count += 1

    def might_contain(self, key: str) -> bool:
        """
        Test if a key MIGHT be in the set.

        Returns:
            False → Definitely not in set (skip I/O, save 200-500ms)
            True  → Possibly in set (proceed to actual check)
        """
        positions = self._hashes(key)
        # Read without lock (int reads are atomic in CPython; benign races)
        bits = self._bits
        return all((bits >> pos) & 1 for pos in positions)

    def clear(self) -> None:
        """Reset the filter."""
        with self._lock:
            self._bits = 0
            self._count = 0

    def rebuild(self, keys: List[str]) -> None:
        """
        Rebuild the filter from a list of keys.
        Used after deletions (Bloom filters don't support delete).
        """
        with self._lock:
            self._bits = 0
            self._count = 0
        for key in keys:
            self.add(key)
        logger.debug(f"BloomFilter rebuilt with {len(keys)} keys")

    def stats(self) -> Dict[str, Any]:
        """Return filter statistics."""
        set_bits = bin(self._bits).count('1')
        fill_ratio = set_bits / self._m if self._m > 0 else 0
        # Estimated false-positive rate: (set_bits / m) ^ k
        estimated_fp = fill_ratio ** self._k if fill_ratio < 1 else 1.0
        return {
            "capacity_bits": self._m,
            "hash_functions": self._k,
            "items_added": self._count,
            "bits_set": set_bits,
            "fill_ratio": round(fill_ratio, 4),
            "estimated_fp_rate": round(estimated_fp, 6),
        }


# =============================================================================
# STORAGE BACKEND ABSTRACTION (Abstract Factory Pattern)
# =============================================================================

class StorageBackend(ABC):
    """
    Abstract storage backend for DataFrame persistence.
    
    Implementations must be thread-safe and handle:
    - Multi-tenant isolation via client_id
    - Hierarchical dataset organization (client/doc/sheet)
    - Parquet serialization for cross-platform compatibility
    - Graceful error handling with appropriate logging
    
    Design Principles:
    1. All operations are atomic where possible
    2. Read operations should never throw - return None on missing
    3. Write operations should return status/key
    4. Metadata is stored separately from data for efficiency
    """
    
    @abstractmethod
    def save_dataframe(
        self, 
        client_id: str, 
        dataset_id: str, 
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save DataFrame to backend storage.
        
        Args:
            client_id: Normalized tenant identifier
            dataset_id: Unique dataset identifier
            df: DataFrame to persist
            metadata: Optional metadata dict
            
        Returns:
            Storage key/path on success
            
        Raises:
            StorageError on failure (implementations should log and re-raise)
        """
        pass
    
    @abstractmethod
    def load_dataframe(
        self, 
        client_id: str, 
        dataset_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from backend storage.
        
        Args:
            client_id: Tenant identifier for ownership verification
            dataset_id: Dataset identifier
            
        Returns:
            DataFrame if found and accessible, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_dataframe(
        self, 
        client_id: str, 
        dataset_id: str
    ) -> bool:
        """
        Delete DataFrame from storage.
        
        Args:
            client_id: Tenant identifier for authorization
            dataset_id: Dataset to delete
            
        Returns:
            True if deleted or didn't exist, False on error
        """
        pass
    
    @abstractmethod
    def list_datasets(self, client_id: str) -> List[str]:
        """
        List all dataset IDs for a client.
        
        Args:
            client_id: Tenant identifier
            
        Returns:
            List of dataset IDs belonging to client
        """
        pass
    
    @abstractmethod
    def exists(self, client_id: str, dataset_id: str) -> bool:
        """Check if dataset exists."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get storage backend metrics (optional override)."""
        return {"backend": self.__class__.__name__}


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage with LRU memory cache.
    
    Structure: base_path / client_id / doc_id / sheet_name.parquet
    
    Features:
    - Memory LRU cache for hot data (configurable size)
    - Write-through caching (writes go to both memory and disk)
    - Thread-safe with RLock
    - Parquet format for efficient columnar storage
    """
    
    def __init__(
        self, 
        base_path: str = "data/dataframes",
        max_cache_size: int = 10,
        compression: str = "snappy"
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.compression = compression
        
        # Thread-safe LRU cache
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._lock = threading.RLock()
        
        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._disk_reads = 0
        self._disk_writes = 0
        
        logger.info(f"LocalStorageBackend initialized: {self.base_path} (cache_size={max_cache_size})")
    
    def _make_key(self, client_id: str, dataset_id: str) -> str:
        """Create cache key from identifiers."""
        return f"{client_id}:{dataset_id}"
    
    def _get_path(self, client_id: str, dataset_id: str) -> Path:
        """
        Get filesystem path for dataset.
        
        Handles hierarchical dataset_id formats:
        - 3-part: client:doc:sheet → client/doc/sheet.parquet
        - 2-part: doc:sheet → client_id/doc/sheet.parquet
        - 1-part: name → client_id/misc/name.parquet
        """
        # Parse dataset_id
        parts = dataset_id.split(':')
        if len(parts) >= 3:
            # Full hierarchical: use as-is (client already in dataset_id)
            safe_client = self._sanitize_path_component(parts[0])
            safe_doc = self._sanitize_path_component(parts[1])
            safe_sheet = self._sanitize_path_component(':'.join(parts[2:]))
        elif len(parts) == 2:
            safe_client = self._sanitize_path_component(client_id)
            safe_doc = self._sanitize_path_component(parts[0])
            safe_sheet = self._sanitize_path_component(parts[1])
        else:
            safe_client = self._sanitize_path_component(client_id)
            safe_doc = "misc"
            safe_sheet = self._sanitize_path_component(dataset_id)
        
        return self.base_path / safe_client / safe_doc / f"{safe_sheet}.parquet"
    
    def _sanitize_path_component(self, value: str) -> str:
        """Sanitize string for safe filesystem path."""
        if not value:
            return "unnamed"
        sanitized = re.sub(r'[\\/:*?"<>|]', '_', str(value).lower().strip())
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        return sanitized.strip('_') or "unnamed"
    
    def _evict_lru(self) -> None:
        """Evict least recently used items if cache is full."""
        while len(self._cache) > self.max_cache_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"LRU evicted: {evicted_key}")
    
    def _serialize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for Parquet serialization with comprehensive type handling.
        
        DESIGN PHILOSOPHY:
        This is a defensive serialization layer that prioritizes DATA PRESERVATION
        over perfect type fidelity. For analytical/LLM workloads, having the data
        in a queryable format is more important than preserving exact dtypes.
        
        HANDLED EDGE CASES (from real-world Excel/CSV imports):
        ┌─────────────────────────────────────────────────────────────────┐
        │ Type                        │ Conversion Strategy               │
        ├─────────────────────────────┼───────────────────────────────────┤
        │ category                    │ → string (preserves labels)       │
        │ object (mixed types)        │ → string (safe fallback)          │
        │ object (nested list/dict)   │ → JSON string                     │
        │ Sparse arrays               │ → dense then convert              │
        │ Extension arrays (Int64)    │ → native nullable int/float       │
        │ StringDtype                 │ → object string                   │
        │ datetime64[tz]              │ → datetime64[ns] (strip tz)       │
        │ timedelta64                 │ → string (ISO format)             │
        │ Interval/Period             │ → string                          │
        │ complex128                  │ → string                          │
        │ bytes                       │ → base64 string                   │
        │ Decimal                     │ → float64                         │
        │ UUID                        │ → string                          │
        │ Custom objects              │ → str() representation            │
        │ NaN variations              │ → unified pd.NA or None           │
        │ Infinity values             │ → preserved (Parquet supports)    │
        │ MultiIndex columns          │ → flattened string names          │
        │ Duplicate column names      │ → suffixed with _1, _2, etc.      │
        └─────────────────────────────┴───────────────────────────────────┘
        
        Returns:
            pd.DataFrame: Parquet-safe DataFrame with all types serializable
        """
        import numpy as np
        import base64
        from decimal import Decimal
        
        df_clean = df.copy()
        
        # === STEP 1: Handle problematic column names ===
        
        # Flatten MultiIndex columns
        if isinstance(df_clean.columns, pd.MultiIndex):
            df_clean.columns = ['_'.join(map(str, col)).strip('_') for col in df_clean.columns]
        
        # Handle duplicate column names (common in messy Excel files)
        seen = {}
        new_cols = []
        for col in df_clean.columns:
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                new_cols.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                new_cols.append(col_str)
        df_clean.columns = new_cols
        
        # === STEP 2: Handle each column by dtype priority ===
        
        for col in df_clean.columns:
            try:
                series = df_clean[col]
                dtype = series.dtype
                dtype_name = dtype.name
                
                # --- Sparse arrays: densify first ---
                if isinstance(dtype, pd.SparseDtype):
                    df_clean[col] = series.sparse.to_dense()
                    series = df_clean[col]
                    dtype = series.dtype
                    dtype_name = dtype.name
                
                # --- Category: convert to string ---
                if dtype_name == 'category':
                    df_clean[col] = series.astype(str).replace('nan', pd.NA)
                    continue
                
                # --- Nullable extension types (Int64, Float64, boolean, string) ---
                if isinstance(dtype, pd.api.types.CategoricalDtype):
                    df_clean[col] = series.astype(str)
                    continue
                    
                if dtype_name in ('Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'):
                    # Convert nullable int to float (to preserve NaN)
                    df_clean[col] = series.astype('float64')
                    continue
                    
                if dtype_name in ('Float32', 'Float64'):
                    df_clean[col] = series.astype('float64')
                    continue
                    
                if dtype_name == 'boolean':
                    # Nullable boolean -> object with True/False/None
                    df_clean[col] = series.astype(object)
                    continue
                    
                if dtype_name in ('string', 'String'):
                    df_clean[col] = series.astype(object)
                    continue
                
                # --- Datetime with timezone: strip timezone ---
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    try:
                        if hasattr(series.dt, 'tz') and series.dt.tz is not None:
                            df_clean[col] = series.dt.tz_convert('UTC').dt.tz_localize(None)
                        # Ensure it's standard datetime64[ns]
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    except Exception:
                        df_clean[col] = series.astype(str)
                    continue
                
                # --- Timedelta: convert to string ---
                if pd.api.types.is_timedelta64_dtype(dtype):
                    df_clean[col] = series.astype(str)
                    continue
                
                # --- Interval/Period: convert to string ---
                if dtype_name.startswith(('interval', 'Interval', 'period', 'Period')):
                    df_clean[col] = series.astype(str)
                    continue
                
                # --- Complex numbers: convert to string ---
                if np.issubdtype(dtype, np.complexfloating):
                    df_clean[col] = series.apply(lambda x: f"{x.real}+{x.imag}j" if pd.notna(x) else None)
                    continue
                
                # --- Object columns: the most complex case ---
                if dtype == object or dtype_name == 'object':
                    df_clean[col] = self._serialize_object_column(series)
                    continue
                    
                # --- Standard numeric types: leave as-is ---
                if np.issubdtype(dtype, np.number):
                    continue
                    
                # --- Boolean: leave as-is ---
                if dtype == bool or dtype_name == 'bool':
                    continue
                    
            except Exception as col_error:
                # Ultimate fallback: force everything to string
                logger.warning(f"Column '{col}' serialization fallback: {col_error}")
                try:
                    df_clean[col] = df_clean[col].apply(
                        lambda x: str(x) if x is not None and pd.notna(x) else None
                    )
                except Exception:
                    df_clean[col] = df_clean[col].fillna('').astype(str)
        
        return df_clean
    
    def _serialize_object_column(self, series: pd.Series) -> pd.Series:
        """
        Serialize an object-dtype column for Parquet compatibility.
        
        CRITICAL: Parquet requires homogeneous column types. If a column contains
        mixed types (str + int + float), we MUST convert everything to string.
        """
        import numpy as np
        import base64
        from decimal import Decimal
        
        # Increase sample size for better detection (was 100, now 500)
        # Still keeps performance reasonable while reducing risk of missing outliers
        sample = series.dropna().head(500).tolist()
        
        if not sample:
            return series.fillna('').astype(str)
        
        # Detect types in sample
        type_counts = {}
        has_complex = False
        
        for val in sample:
            val_type = type(val).__name__
            type_counts[val_type] = type_counts.get(val_type, 0) + 1
            
            if isinstance(val, (list, dict, set, tuple, np.ndarray, bytes, bytearray, Decimal)):
                has_complex = True
        
        # Determine if column is homogeneous primitive
        primitive_types = {'str', 'int', 'float', 'bool'}
        unique_types = set(type_counts.keys())
        is_homogeneous_primitive = (
            len(unique_types) == 1 and 
            unique_types.issubset(primitive_types) and
            not has_complex
        )
        
        # If homogeneous and just strings, do fast path
        if is_homogeneous_primitive and 'str' in unique_types:
            return series.fillna('').astype(str)
        
        # For ANYTHING mixed or complex: convert ALL to string representation
        def safe_stringify(val):
            if val is None: return None
            if isinstance(val, float) and (np.isnan(val) or pd.isna(val)): return None
            try:
                if isinstance(val, (list, tuple, dict, set, frozenset, np.ndarray)):
                    if isinstance(val, np.ndarray): val = val.tolist()
                    if isinstance(val, (set, frozenset)): val = list(val)
                    return json.dumps(val)
                if isinstance(val, (bytes, bytearray)):
                    return base64.b64encode(bytes(val)).decode('ascii')
                if isinstance(val, Decimal): return str(val)
                if isinstance(val, (pd.Timestamp, pd.Timedelta)): return str(val)
                return str(val)
            except:
                return str(val)
        
        result = series.apply(safe_stringify)
        return result.astype(object)

    def save_dataframe(
        self, 
        client_id: str, 
        dataset_id: str, 
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save DataFrame to local filesystem with LRU caching and self-healing resilience."""
        key = self._make_key(client_id, dataset_id)
        path = self._get_path(client_id, dataset_id)
        
        with self._lock:
            # Update cache (write-through)
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = df
            self._evict_lru()
            
            # Persist to disk with Retry/Fallback logic
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Attempt 1: Standard Serialization
                try:
                    df_clean = self._serialize_df(df)
                    df_clean.to_parquet(path, index=False, compression=self.compression)
                    self._disk_writes += 1
                    logger.debug(f"Saved to disk: {path}")
                    
                except Exception as e_parquet:
                    logger.warning(f"Standard persistence failed for {dataset_id}: {e_parquet}. Retrying with Brute Force cleaning...")
                    
                    # Attempt 2: Brute Force Cleaning (Convert ALL object/category cols to string)
                    # This fixes PyArrow type inference errors on mixed columns
                    df_retry = df.copy()
                    for col in df_retry.columns:
                        if df_retry[col].dtype == object or df_retry[col].dtype.name == 'category':
                            df_retry[col] = df_retry[col].astype(str)
                    
                    # Try saving again
                    df_retry.to_parquet(path, index=False, compression=self.compression)
                    self._disk_writes += 1
                    logger.info(f"Self-healed persistence for {dataset_id} using brute force stringification.")

            except Exception as e:
                logger.error(f"Failed to persist {dataset_id} after retries: {e}")
                # Data is still in cache, so operation partially succeeded (System doesn't crash)
        
        return key

    
    def load_dataframe(
        self, 
        client_id: str, 
        dataset_id: str
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache or filesystem."""
        key = self._make_key(client_id, dataset_id)
        
        with self._lock:
            # Check cache first
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache_hits += 1
                return self._cache[key]
            
            self._cache_misses += 1
            
            # Load from disk
            path = self._get_path(client_id, dataset_id)
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    self._disk_reads += 1
                    # Hydrate cache
                    self._cache[key] = df
                    self._evict_lru()
                    logger.debug(f"Loaded from disk: {path}")
                    return df
                except Exception as e:
                    logger.error(f"Failed to load {dataset_id}: {e}")
            
            return None
    
    def delete_dataframe(
        self, 
        client_id: str, 
        dataset_id: str
    ) -> bool:
        """Delete DataFrame from cache and filesystem."""
        key = self._make_key(client_id, dataset_id)
        path = self._get_path(client_id, dataset_id)
        
        with self._lock:
            # Remove from cache
            if key in self._cache:
                del self._cache[key]
            
            # Remove from disk
            if path.exists():
                try:
                    path.unlink()
                    logger.debug(f"Deleted: {path}")
                except Exception as e:
                    logger.error(f"Failed to delete {dataset_id}: {e}")
                    return False
        
        return True
    
    def list_datasets(self, client_id: str) -> List[str]:
        """List all datasets for client by scanning filesystem."""
        safe_client = self._sanitize_path_component(client_id)
        client_path = self.base_path / safe_client
        
        datasets = []
        if client_path.exists():
            for doc_path in client_path.iterdir():
                if doc_path.is_dir():
                    for parquet_file in doc_path.glob("*.parquet"):
                        sheet_name = parquet_file.stem
                        dataset_id = f"{safe_client}:{doc_path.name}:{sheet_name}"
                        datasets.append(dataset_id)
        
        return datasets
    
    def exists(self, client_id: str, dataset_id: str) -> bool:
        """Check if dataset exists in cache or filesystem."""
        key = self._make_key(client_id, dataset_id)
        with self._lock:
            if key in self._cache:
                return True
        return self._get_path(client_id, dataset_id).exists()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get storage backend metrics."""
        with self._lock:
            cache_size = len(self._cache)
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / max(1, total_requests)
        
        return {
            "backend": "LocalStorageBackend",
            "base_path": str(self.base_path),
            "cache_entries": cache_size,
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": round(hit_rate, 3),
            "disk_reads": self._disk_reads,
            "disk_writes": self._disk_writes,
        }


class S3StorageBackend(StorageBackend):
    """
    AWS S3 storage backend for production deployment.
    
    Structure: s3://bucket/prefix/client_id/doc_id/sheet_name.parquet
    
    Features:
    - Request-scope ephemeral cache (cleared per Lambda invocation)
    - Lazy S3 client initialization
    - Parquet format with configurable compression
    - Thread-safe for concurrent access
    
    Environment Variables:
    - AWS_S3_BUCKET: Target bucket name
    - AWS_S3_PREFIX: Key prefix for DataFrames
    - AWS_REGION: AWS region
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "dataframes/",
        region: Optional[str] = None,
        enable_local_cache: bool = True,
        max_cache_size: int = 5,
        compression: str = "snappy"
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'
        # Read region from env if not specified
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
        self.compression = compression
        
        # Request-scope cache (ephemeral)
        self._enable_cache = enable_local_cache
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._max_cache_size = max_cache_size
        self._lock = threading.RLock()
        
        # Lazy S3 client
        self._s3 = None
        self._s3_resource = None
        
        # Metrics
        self._s3_reads = 0
        self._s3_writes = 0
        self._cache_hits = 0
        
        logger.info(f"S3StorageBackend initialized: s3://{bucket}/{prefix}")
    
    def _get_s3_client(self):
        """Lazy-initialize S3 client."""
        if self._s3 is None:
            try:
                import boto3
                self._s3 = boto3.client('s3', region_name=self.region)
                self._s3_resource = boto3.resource('s3', region_name=self.region)
            except ImportError:
                raise RuntimeError("boto3 required for S3StorageBackend. Install with: pip install boto3")
        return self._s3
    
    def _make_key(self, client_id: str, dataset_id: str) -> str:
        """Create S3 object key."""
        # Sanitize and build hierarchical path
        safe_client = self._sanitize_component(client_id)
        
        parts = dataset_id.split(':')
        if len(parts) >= 3:
            safe_doc = self._sanitize_component(parts[1])
            safe_sheet = self._sanitize_component(':'.join(parts[2:]))
        elif len(parts) == 2:
            safe_doc = self._sanitize_component(parts[0])
            safe_sheet = self._sanitize_component(parts[1])
        else:
            safe_doc = "misc"
            safe_sheet = self._sanitize_component(dataset_id)
        
        return f"{self.prefix}{safe_client}/{safe_doc}/{safe_sheet}.parquet"
    
    def _make_cache_key(self, client_id: str, dataset_id: str) -> str:
        """Create cache key."""
        return f"{client_id}:{dataset_id}"
    
    def _sanitize_component(self, value: str) -> str:
        """Sanitize string for S3 key."""
        if not value:
            return "unnamed"
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', str(value).lower().strip())
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        return sanitized.strip('_') or "unnamed"
    
    def _evict_cache(self) -> None:
        """Evict LRU items from request cache."""
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
    
    def save_dataframe(
        self, 
        client_id: str, 
        dataset_id: str, 
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save DataFrame to S3."""
        s3_key = self._make_key(client_id, dataset_id)
        cache_key = self._make_cache_key(client_id, dataset_id)
        
        # Serialize to Parquet bytes
        buffer = io.BytesIO()
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                df_clean[col] = df_clean[col].astype(str)
        df_clean.to_parquet(buffer, index=False, compression=self.compression)
        buffer.seek(0)
        
        # Upload to S3
        try:
            s3 = self._get_s3_client()
            s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream',
                Metadata={
                    'client_id': client_id,
                    'dataset_id': dataset_id,
                    'rows': str(len(df)),
                    'columns': str(len(df.columns)),
                    'created_at': datetime.utcnow().isoformat()
                }
            )
            self._s3_writes += 1
            logger.debug(f"Saved to S3: s3://{self.bucket}/{s3_key}")
            
            # Update request cache
            if self._enable_cache:
                with self._lock:
                    self._cache[cache_key] = df
                    self._evict_cache()
            
            return s3_key
            
        except Exception as e:
            logger.error(f"S3 upload failed for {dataset_id}: {e}")
            raise
    
    def load_dataframe(
        self, 
        client_id: str, 
        dataset_id: str
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from S3 with request-scope caching."""
        cache_key = self._make_cache_key(client_id, dataset_id)
        
        # Check request cache
        if self._enable_cache:
            with self._lock:
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    self._cache_hits += 1
                    return self._cache[cache_key]
        
        # Load from S3
        s3_key = self._make_key(client_id, dataset_id)
        
        try:
            s3 = self._get_s3_client()
            response = s3.get_object(Bucket=self.bucket, Key=s3_key)
            buffer = io.BytesIO(response['Body'].read())
            df = pd.read_parquet(buffer)
            self._s3_reads += 1
            
            # Hydrate cache
            if self._enable_cache:
                with self._lock:
                    self._cache[cache_key] = df
                    self._evict_cache()
            
            logger.debug(f"Loaded from S3: s3://{self.bucket}/{s3_key}")
            return df
            
        except self._get_s3_client().exceptions.NoSuchKey:
            return None
        except Exception as e:
            # Log but don't crash - try alternate methods
            logger.warning(f"S3 load failed for {dataset_id}: {e}")
            return None
    
    def delete_dataframe(
        self, 
        client_id: str, 
        dataset_id: str
    ) -> bool:
        """Delete DataFrame from S3."""
        s3_key = self._make_key(client_id, dataset_id)
        cache_key = self._make_cache_key(client_id, dataset_id)
        
        # Remove from cache
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
        
        # Delete from S3
        try:
            s3 = self._get_s3_client()
            s3.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.debug(f"Deleted from S3: s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"S3 delete failed for {dataset_id}: {e}")
            return False
    
    def list_datasets(self, client_id: str) -> List[str]:
        """List all datasets for client in S3."""
        safe_client = self._sanitize_component(client_id)
        prefix = f"{self.prefix}{safe_client}/"
        
        datasets = []
        try:
            s3 = self._get_s3_client()
            paginator = s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.parquet'):
                        # Parse key back to dataset_id
                        rel_path = key[len(self.prefix):]
                        parts = rel_path.rstrip('.parquet').split('/')
                        if len(parts) >= 3:
                            dataset_id = ':'.join(parts)
                            datasets.append(dataset_id)
        except Exception as e:
            logger.error(f"S3 list failed for {client_id}: {e}")
        
        return datasets
    
    def exists(self, client_id: str, dataset_id: str) -> bool:
        """Check if dataset exists in S3."""
        cache_key = self._make_cache_key(client_id, dataset_id)
        
        # Check cache first
        with self._lock:
            if cache_key in self._cache:
                return True
        
        # Check S3
        s3_key = self._make_key(client_id, dataset_id)
        try:
            s3 = self._get_s3_client()
            s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except:
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get storage backend metrics."""
        with self._lock:
            cache_size = len(self._cache)
        
        return {
            "backend": "S3StorageBackend",
            "bucket": self.bucket,
            "prefix": self.prefix,
            "cache_entries": cache_size,
            "s3_reads": self._s3_reads,
            "s3_writes": self._s3_writes,
            "cache_hits": self._cache_hits,
        }


def get_storage_backend() -> StorageBackend:
    """
    Factory function to get appropriate storage backend based on environment.
    
    Environment Selection:
    - If DEPLOYMENT_ENV='aws' or AWS_S3_BUCKET is set: S3StorageBackend
    - Otherwise: LocalStorageBackend
    
    This function is idempotent and caches the backend instance.
    """
    global _storage_backend
    
    if _storage_backend is not None:
        return _storage_backend
    
    try:
        from app.config import settings
        deployment = settings.deployment
        storage = settings.storage
        
        if deployment.is_aws:
            bucket = deployment.aws_s3_bucket
            if not bucket:
                logger.warning("AWS mode but no bucket configured, falling back to local")
            else:
                _storage_backend = S3StorageBackend(
                    bucket=bucket,
                    prefix=deployment.aws_s3_prefix,
                    region=deployment.aws_region,
                    enable_local_cache=deployment.enable_local_cache,
                    max_cache_size=storage.max_lru_dataframes,
                    compression=storage.parquet_compression
                )
                return _storage_backend
        
        # Local mode (default)
        _storage_backend = LocalStorageBackend(
            base_path=storage.dataframe_cache_path,
            max_cache_size=storage.max_lru_dataframes,
            compression=storage.parquet_compression
        )
        
    except Exception as e:
        logger.warning(f"Storage backend config failed, using defaults: {e}")
        _storage_backend = LocalStorageBackend()
    
    return _storage_backend


# Singleton backend instance
_storage_backend: Optional[StorageBackend] = None




class DataRegistry:
    """
    Thread-safe registry for DataFrames with:
    - LRU cache for in-memory DataFrames (capped size)
    - Parquet persistence to disk
    - Redis metadata storage (optional)
    - Bloom filter for zero-latency existence checks
    - Multi-tenant isolation via client_id
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_lru_size: int = 10,
        redis_url: Optional[str] = None
    ):
        from app.config import CACHE_DIR, settings
        import os
        
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_lru_size = max_lru_size or settings.storage.max_lru_dataframes
        
        # LRU cache: OrderedDict preserves insertion order
        self._dataframes: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Bloom filter: zero-latency "definitely not here" check
        self._bloom = BloomFilter(
            capacity=settings.storage.bloom_capacity,
            fp_rate=settings.storage.bloom_fp_rate
        )
        self._bloom_check_saves: int = 0  # observability counter
        
        # Redis client (optional)
        self._redis = self._initialize_redis(redis_url)
        
        # Load existing metadata from disk (also populates bloom filter)
        self._load_disk_metadata()

    def _initialize_redis(self, redis_url: Optional[str]):
        """Initialize Redis with Upstash support and Docker fallback."""
        import os
        
        # 1. Resolve URL (Argument -> Env -> Setting)
        url = redis_url or os.getenv("REDIS_URL")
        token = os.getenv("REDIS_TOKEN")
        
        client = None
        
        # 2. Try Upstash (HTTP)
        if url and (url.startswith("http://") or url.startswith("https://")):
            try:
                from upstash_redis import Redis as UpstashRedis
                logger.info(f"Connecting to Upstash Redis: {url}")
                # Upstash client needs url and token
                auth_token = token or os.getenv("UPSTASH_REDIS_REST_TOKEN")
                client = UpstashRedis(url=url, token=auth_token)
                # Verify
                client.get("test_connection")
                logger.info("[OK] Connected to Upstash Redis")
                return client
            except Exception as e:
                logger.warning(f"Upstash connection failed: {e}")
                client = None

        # 3. Try Standard Redis (TCP)
        if url and not client and not (url.startswith("http://") or url.startswith("https://")):
            try:
                import redis
                logger.info(f"Connecting to standard Redis: {url}")
                client = redis.from_url(url, decode_responses=True)
                client.ping()
                logger.info("[OK] Connected to standard Redis")
                return client
            except Exception as e:
                logger.warning(f"Standard Redis failed: {e}")
                client = None
        elif url and (url.startswith("http://") or url.startswith("https://")) and not client:
            logger.warning("HTTP(S) Redis URL detected but Upstash REST connection failed; skipping standard Redis client.")

        # 4. Fallback: Local Docker Redis
        if not client:
            return self._start_local_docker_redis()
            
        return client

    def _start_local_docker_redis(self):
        """Start local Redis container if needed."""
        import subprocess
        import redis
        import time
        
        logger.warning("Attempting to start local Redis (Docker fallback)...")
        
        # Container name from env or default
        container_name = os.getenv("REDIS_DOCKER_CONTAINER", "ai-ca-redis")
        port = int(os.getenv("REDIS_PORT", "6379"))
        
        try:
            # Check if running
            check = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={container_name}"],
                capture_output=True, text=True
            )
            
            if not check.stdout.strip():
                # Check if exists but stopped
                exists = subprocess.run(
                    ["docker", "ps", "-aq", "-f", f"name={container_name}"],
                    capture_output=True, text=True
                )
                if exists.stdout.strip():
                    logger.info("Starting existing Redis container...")
                    subprocess.run(["docker", "start", container_name], check=True)
                else:
                    logger.info("Pulling and running Redis container...")
                    subprocess.run(["docker", "pull", "redis:alpine"], check=False) # Pull
                    subprocess.run(
                        ["docker", "run", "-d", "--name", container_name, "-p", f"{port}:6379", "redis:alpine"],
                        check=True
                    )
                
                # Wait for startup
                time.sleep(2)
            
            # Connect
            local_url = f"redis://localhost:{port}/0"
            client = redis.from_url(local_url, decode_responses=True)
            client.ping()
            logger.info("[OK] Connected to Local Docker Redis")
            return client
            
        except Exception as e:
            logger.error(f"Failed to start/connect to local Redis: {e}")
            return None

    def _load_disk_metadata(self):
        """Load metadata from disk cache and populate bloom filter."""
        meta_file = self.cache_dir / "_metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    self._metadata = json.load(f)
                # Populate bloom filter with all known dataset IDs
                for dataset_id in self._metadata:
                    self._bloom.add(dataset_id)
                logger.info(
                    f"Loaded {len(self._metadata)} dataset metadata entries "
                    f"(bloom filter populated)"
                )
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    def _save_disk_metadata(self):
        """Persist metadata to disk."""
        meta_file = self.cache_dir / "_metadata.json"
        try:
            with open(meta_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _evict_lru(self):
        """Evict least recently used DataFrames if over limit."""
        while len(self._dataframes) > self.max_lru_size:
            oldest_key = next(iter(self._dataframes))
            del self._dataframes[oldest_key]
            logger.debug(f"Evicted LRU DataFrame: {oldest_key}")

    def _parquet_path(self, dataset_id: str, client_id: Optional[str] = None) -> Path:
        """
        Get parquet file path for dataset for tiered storage.
        
        Structure: cache_dir / client_id / doc_id / sheet_name.parquet
        This mimics S3 object key structure for easy migration.
        
        Handles various dataset_id formats:
        - 3-part: "client:doc:sheet" → client/doc/sheet.parquet
        - 2-part: "doc:sheet" → client_id/doc/sheet.parquet (uses provided client_id)
        - 1-part: "sheet" → client_id/misc/sheet.parquet
        
        Args:
            dataset_id: The dataset identifier
            client_id: Optional client ID (used for 2-part legacy IDs)
        """
        import re
        from app.core.id_generator import parse_dataset_id
        
        # Parse components
        try:
            parts = parse_dataset_id(dataset_id)
            parsed_client = parts.get("client_id")
            parsed_doc = parts.get("doc_id")
            parsed_sheet = parts.get("sheet_name")
        except Exception:
            parsed_client = None
            parsed_doc = None
            parsed_sheet = dataset_id
        
        # Determine actual values with proper fallbacks
        # For 2-part IDs like "filename:sheetname", use filename as doc_id
        if parsed_doc is None and parsed_client is not None:
            # 2-part ID: client_id is actually the filename, use it as doc_id
            final_client = client_id or "default_client"
            final_doc = parsed_client  # filename becomes doc_id
            final_sheet = parsed_sheet or "data"
        elif parsed_client is None and parsed_doc is None:
            # 1-part ID: just sheet name
            final_client = client_id or "default_client"
            final_doc = "misc"
            final_sheet = parsed_sheet or dataset_id or "data"
        else:
            # 3-part ID: full hierarchical structure
            final_client = parsed_client or client_id or "default_client"
            final_doc = parsed_doc or "misc"
            final_sheet = parsed_sheet or "data"

        # Normalize client_id for VectorDB compatibility
        try:
            from app.core.id_generator import normalize_client_id
            final_client = normalize_client_id(final_client)
        except ImportError:
            pass

        # Sanitize components for filesystem/S3 safety
        safe_client = re.sub(r'[\\/:*?"<>|]', '_', str(final_client))
        safe_doc = re.sub(r'[\\/:*?"<>|]', '_', str(final_doc))
        safe_sheet = re.sub(r'[\\/:*?"<>|]', '_', str(final_sheet))
        
        # Create hierarchical directory structure
        target_dir = self.cache_dir / safe_client / safe_doc
        target_dir.mkdir(parents=True, exist_ok=True)
        
        return target_dir / f"{safe_sheet}.parquet"

    def _normalize_client_id(self, client_id: Optional[str]) -> Optional[str]:
        """Normalize client ID for consistent filtering."""
        if not client_id:
            return None
        try:
            from app.core.id_generator import normalize_client_id
            return normalize_client_id(client_id)
        except ImportError:
            return client_id.lower().replace(' ', '_').replace(':', '_')

    def register(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None
    ) -> bool:
        """
        Register a DataFrame with the registry.
        
        Args:
            dataset_id: Unique identifier for the dataset
            df: Pandas DataFrame to register
            metadata: Optional metadata dict
            client_id: Client/tenant ID for multi-tenancy
            
        Returns:
            True if successful (in-memory registration always succeeds)
        """
        if df is None or df.empty:
            logger.warning(f"Attempted to register empty DataFrame: {dataset_id}")
            return False

        # Normalize client_id for consistent filtering
        safe_client_id = self._normalize_client_id(client_id)

        # Build metadata first
        meta = metadata or {}
        meta.update({
            "dataset_id": dataset_id,
            "client_id": safe_client_id,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "registered_at": datetime.utcnow().isoformat(),
        })

        # Store in LRU cache first (always succeeds)
        if dataset_id in self._dataframes:
            self._dataframes.move_to_end(dataset_id)
        self._dataframes[dataset_id] = df
        self._evict_lru()

        # Store metadata in memory
        self._metadata[dataset_id] = meta

        # Try to persist to Parquet (non-blocking - log errors but don't fail)
        parquet_path = self._parquet_path(dataset_id, safe_client_id)
        try:
            # Convert problematic columns to string to avoid Parquet type issues
            df_clean = df.copy()
            for col in df_clean.columns:
                col_dtype = df_clean[col].dtype
                # Handle object, category, and mixed-type columns
                if col_dtype == object or str(col_dtype) == 'category':
                    df_clean[col] = df_clean[col].astype(str)
                # Handle nullable integer types with mixed data
                elif 'Int' in str(col_dtype) or 'int' in str(col_dtype).lower():
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except Exception:
                        df_clean[col] = df_clean[col].astype(str)
            df_clean.to_parquet(parquet_path, index=False)
            meta["parquet_path"] = str(parquet_path)
        except Exception as e:
            logger.warning(f"Parquet persistence failed for {dataset_id} (using in-memory only): {e}")
            meta["parquet_path"] = None
            meta["memory_only"] = True

        # Save metadata to disk
        self._save_disk_metadata()

        # Redis metadata (optional)
        if self._redis:
            try:
                redis_key = f"dataset:{safe_client_id}:{dataset_id}"
                self._redis.setex(redis_key, 86400, json.dumps(meta, default=str))
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        logger.info(f"Registered dataset: {dataset_id} ({len(df)} rows)")

        # Update bloom filter
        self._bloom.add(dataset_id)

        return True

    def get(self, dataset_id: str, client_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame by ID with optional client ownership check.
        
        Uses Bloom filter for zero-latency negative lookups:
        - If bloom says "no" → skip I/O entirely (200-500ms saved)
        - If bloom says "maybe" → proceed to LRU / disk check
        
        Args:
            dataset_id: Dataset identifier
            client_id: If provided, verify ownership
            
        Returns:
            DataFrame or None if not found/unauthorized
        """
        # ═══ Bloom Filter Fast-Reject ═══
        if not self._bloom.might_contain(dataset_id):
            self._bloom_check_saves += 1
            logger.debug(
                f"Bloom filter: '{dataset_id}' definitely not in registry "
                f"(saved I/O #{self._bloom_check_saves})"
            )
            return None

        # Normalize client_id and check ownership
        if client_id:
            safe_client_id = self._normalize_client_id(client_id)
            meta = self._metadata.get(dataset_id)
            if meta and meta.get("client_id") != safe_client_id:
                logger.warning(f"Client {client_id} unauthorized for {dataset_id}")
                return None

        # Check LRU cache first
        if dataset_id in self._dataframes:
            self._dataframes.move_to_end(dataset_id)
            return self._dataframes[dataset_id]

        # Load from disk
        safe_client_id = self._normalize_client_id(client_id) if client_id else None
        parquet_path = self._parquet_path(dataset_id, safe_client_id)
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                # Add to LRU cache
                self._dataframes[dataset_id] = df
                self._evict_lru()
                logger.debug(f"Loaded {dataset_id} from disk")
                return df
            except Exception as e:
                logger.error(f"Failed to load {dataset_id}: {e}")

        return None

    def get_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        return self._metadata.get(dataset_id)

    def list_for_client(self, client_id: str) -> List[Dict[str, Any]]:
        """
        List all datasets for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            List of dataset metadata dicts
        """
        results = []
        # Normalize client_id for matching
        safe_client_id = self._normalize_client_id(client_id)
        for dataset_id, meta in self._metadata.items():
            if meta.get("client_id") == safe_client_id:
                results.append(meta)
        return results

    def list_all(self) -> List[str]:
        """List all dataset IDs."""
        return list(self._metadata.keys())

    def delete(self, dataset_id: str, client_id: Optional[str] = None) -> bool:
        """Delete a dataset with ownership check."""
        if client_id:
            safe_client_id = self._normalize_client_id(client_id)
            meta = self._metadata.get(dataset_id)
            if meta and meta.get("client_id") != safe_client_id:
                logger.warning(f"Unauthorized delete attempt: {dataset_id}")
                return False

        # Remove from LRU
        if dataset_id in self._dataframes:
            del self._dataframes[dataset_id]

        # Remove from metadata
        if dataset_id in self._metadata:
            del self._metadata[dataset_id]
            self._save_disk_metadata()

        # Remove parquet file
        safe_client_id = self._normalize_client_id(client_id) if client_id else None
        parquet_path = self._parquet_path(dataset_id, safe_client_id)
        if parquet_path.exists():
            parquet_path.unlink()

        logger.info(f"Deleted dataset: {dataset_id}")

        # Rebuild bloom filter (Bloom filters don't support deletion)
        self._bloom.rebuild(list(self._metadata.keys()))

        return True

    def get_schema_info(self, dataset_id: str) -> str:
        """Get schema information for SQL generation."""
        meta = self._metadata.get(dataset_id)
        if not meta:
            return "No schema available"

        columns = meta.get("columns", [])
        dtypes = meta.get("dtypes", {})
        
        schema_lines = [f"Table: {dataset_id}"]
        for col in columns:
            dtype = dtypes.get(col, "unknown")
            schema_lines.append(f"  - {col}: {dtype}")
        
        return "\n".join(schema_lines)


    def bloom_stats(self) -> Dict[str, Any]:
        """
        Get Bloom filter statistics for observability.
        
        Returns:
            Dict with filter stats and I/O savings count
        """
        stats = self._bloom.stats()
        stats["io_checks_saved"] = self._bloom_check_saves
        return stats


# Singleton instance
_registry: Optional[DataRegistry] = None


def get_data_registry() -> DataRegistry:
    """Get or create singleton data registry."""
    global _registry
    
    if _registry is None:
        # Use sensible defaults
        max_lru = 10
        redis_url = None
        
        try:
            from app.config import settings
            max_lru = settings.storage.max_lru_dataframes
            redis_url = settings.cache.redis_url if settings.cache.redis_enabled else None
        except Exception:
            # Use defaults if settings fail
            pass
        
        _registry = DataRegistry(
            max_lru_size=max_lru,
            redis_url=redis_url
        )
    
    return _registry


__all__ = [
    # Storage Backend Abstraction
    "StorageBackend",
    "LocalStorageBackend", 
    "S3StorageBackend",
    "get_storage_backend",
    # Bloom Filter
    "BloomFilter",
    # Data Registry
    "DataRegistry", 
    "get_data_registry",
]
