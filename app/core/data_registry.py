"""
Data Registry - In-process LRU registry with Redis persistence for dataset metadata.
Enforces multi-tenancy via client_id on all operations.
"""
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DataRegistry:
    """
    Thread-safe registry for DataFrames with:
    - LRU cache for in-memory DataFrames (capped size)
    - Parquet persistence to disk
    - Redis metadata storage (optional)
    - Multi-tenant isolation via client_id
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_lru_size: int = 10,
        redis_url: Optional[str] = None
    ):
        from app.config import CACHE_DIR, settings
        
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_lru_size = max_lru_size or settings.storage.max_lru_dataframes
        
        # LRU cache: OrderedDict preserves insertion order
        self._dataframes: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Redis client (optional)
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
                self._redis.ping()
                logger.info("Data registry connected to Redis")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")
        
        # Load existing metadata from disk
        self._load_disk_metadata()

    def _load_disk_metadata(self):
        """Load metadata from disk cache."""
        meta_file = self.cache_dir / "_metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    self._metadata = json.load(f)
                logger.info(f"Loaded {len(self._metadata)} dataset metadata entries")
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

    def _parquet_path(self, dataset_id: str) -> Path:
        """
        Get parquet file path for dataset for tiered storage.
        
        Structure: cache_dir / client_id / doc_id / sheet_name.parquet
        This mimics S3 object key structure for easy migration.
        """
        import re
        from app.core.id_generator import parse_dataset_id
        
        # Parse components
        try:
            parts = parse_dataset_id(dataset_id)
            client_id = parts.get("client_id", "unknown_client")
            doc_id = parts.get("doc_id", "unknown_doc")
            sheet_name = parts.get("sheet_name", dataset_id)
        except Exception:
            # Fallback for legacy IDs
            client_id = "misc"
            doc_id = "misc"
            sheet_name = dataset_id

        # Sanitize components for filesystem/S3 safety
        safe_client = re.sub(r'[\\/:*?"<>|]', '_', client_id)
        safe_doc = re.sub(r'[\\/:*?"<>|]', '_', doc_id)
        safe_sheet = re.sub(r'[\\/:*?"<>|]', '_', sheet_name)
        
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
        parquet_path = self._parquet_path(dataset_id)
        try:
            # Convert object columns to string to avoid Parquet type issues
            df_clean = df.copy()
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
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
        return True

    def get(self, dataset_id: str, client_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame by ID with optional client ownership check.
        
        Args:
            dataset_id: Dataset identifier
            client_id: If provided, verify ownership
            
        Returns:
            DataFrame or None if not found/unauthorized
        """
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
        parquet_path = self._parquet_path(dataset_id)
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
        parquet_path = self._parquet_path(dataset_id)
        if parquet_path.exists():
            parquet_path.unlink()

        logger.info(f"Deleted dataset: {dataset_id}")
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


__all__ = ["DataRegistry", "get_data_registry"]
