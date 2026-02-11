"""
SQL Engine - DuckDB helper for analytical queries with Pandas fallback.
Handles DataFrame registration, safe parameterized queries, type coercion,
and production-grade security validation (SQL injection, path traversal,
code sandbox, client ID normalization).
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ═══ DLL Fix: Must run before loading native C extensions (DuckDB) ═══
try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available, using Pandas-only mode")

try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# SECURITY: Compiled patterns for production-grade input validation
# ═══════════════════════════════════════════════════════════════════

# Dangerous SQL patterns that should never appear in generated queries
_SQL_DANGEROUS_PATTERNS = [
    re.compile(r'\b(DROP|ALTER|TRUNCATE|DELETE|INSERT|UPDATE|CREATE|REPLACE|GRANT|REVOKE)\b\s', re.IGNORECASE),
    re.compile(r';\s*(DROP|ALTER|TRUNCATE|DELETE|INSERT|UPDATE|CREATE)', re.IGNORECASE),
    re.compile(r'--\s*.*$', re.MULTILINE),   # SQL comments (injection vector)
    re.compile(r'/\*.*?\*/', re.DOTALL),      # Block comments
    re.compile(r'\bEXEC(UTE)?\b', re.IGNORECASE),
    re.compile(r'\bxp_\w+', re.IGNORECASE),   # SQL Server extended procs
    re.compile(r'\bSYSTEM\b\s*\(', re.IGNORECASE),
    re.compile(r'\bLOAD_FILE\b', re.IGNORECASE),
    re.compile(r'\bINTO\s+OUTFILE\b', re.IGNORECASE),
    re.compile(r'\bINTO\s+DUMPFILE\b', re.IGNORECASE),
    re.compile(r'\bUNION\b.*\bSELECT\b.*\bFROM\b.*\binformation_schema\b', re.IGNORECASE | re.DOTALL),
]

# Whitelist: only these statement types are allowed
_SQL_ALLOWED_STATEMENTS = re.compile(
    r'^\s*(SELECT|WITH|EXPLAIN)\b',
    re.IGNORECASE
)

# Path traversal attack patterns
_PATH_TRAVERSAL_PATTERNS = [
    re.compile(r'\.\.[\\//]'),           # ../
    re.compile(r'[\\//]\.\.'),           # /..
    re.compile(r'^~'),                    # Home directory expansion
    re.compile(r'[\x00-\x1f]'),           # Control characters
    re.compile(r'[<>"|?*]'),              # Windows special chars
]

# Dangerous Python patterns for sandbox code validation
_PYTHON_DANGEROUS_PATTERNS = [
    re.compile(r'\b(os\.(system|exec|popen|remove|rmdir|unlink|rename))\b'),
    re.compile(r'\b(subprocess|shutil)\.\w+'),
    re.compile(r'\b(eval|exec|compile)\s*\('),
    re.compile(r'\b(__import__|importlib)\b'),
    re.compile(r'\bopen\s*\(.*["\']w["\']'),  # open(..., 'w')
    re.compile(r'\b(requests|urllib|socket|http)\.\w+'),
    re.compile(r'\b(pickle|marshal|shelve)\.(load|loads)\b'),
    re.compile(r'\bglobals\s*\(\s*\)'),
    re.compile(r'\b__builtins__\b'),
]

# Client ID format enforcement
_CLIENT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}$')


class SQLEngine:
    """
    DuckDB SQL engine with:
    - DataFrame registration as views
    - Safe parameterized queries
    - Automatic type coercion and cleaning
    - Pandas fallback when DuckDB fails
    - S3 Parquet integration via httpfs extension
    - Performance tuning (object cache, parallel CSV)
    """

    def __init__(
        self,
        memory_limit: str = "2GB",
        threads: int = 4,
        s3_config: Optional[Dict[str, Any]] = None,
        perf_config: Optional[Dict[str, Any]] = None,
    ):
        self._connection: Optional["duckdb.DuckDBPyConnection"] = None
        self._registered_tables: Dict[str, pd.DataFrame] = {}
        self.memory_limit = memory_limit
        self.threads = threads
        self._s3_config = s3_config or {}
        self._perf_config = perf_config or {}
        self._s3_initialized = False
        self._s3_views: Dict[str, str] = {}  # view_name → S3 URI

        if DUCKDB_AVAILABLE:
            self._init_duckdb()

    def _init_duckdb(self):
        """Initialize DuckDB connection with performance tuning and optional S3."""
        try:
            self._connection = duckdb.connect(":memory:")
            self._connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self._connection.execute(f"SET threads={self.threads}")

            # ═══ Performance Tuning PRAGMAs ═══
            perf = self._perf_config

            # Object cache: keeps Parquet metadata in memory for faster repeated scans
            if perf.get("enable_object_cache", True):
                self._connection.execute("SET enable_object_cache=true")
                cache_size = perf.get("object_cache_size", "256MB")
                try:
                    self._connection.execute(f"SET object_cache_max_size='{cache_size}'")
                except Exception:
                    pass  # Older DuckDB versions may not support this pragma
            else:
                self._connection.execute("SET enable_object_cache=false")

            # Insertion order: disabling gives ~15% speedup on wide table scans
            preserve_order = perf.get("preserve_insertion_order", False)
            self._connection.execute(
                f"SET preserve_insertion_order={'true' if preserve_order else 'false'}"
            )

            logger.info(
                f"DuckDB initialized (memory={self.memory_limit}, "
                f"threads={self.threads}, "
                f"object_cache={'on' if perf.get('enable_object_cache', True) else 'off'}, "
                f"preserve_order={'on' if preserve_order else 'off'})"
            )

            # ═══ S3 Integration (lazy — only if enabled) ═══
            if self._s3_config.get("enable_s3", False):
                self._configure_s3()

        except Exception as e:
            logger.error(f"DuckDB initialization failed: {e}")
            self._connection = None

    # ═══════════════════════════════════════════════════════════════
    # S3 INTEGRATION — httpfs extension + credential management
    # ═══════════════════════════════════════════════════════════════

    def _configure_s3(self) -> None:
        """
        Install and configure DuckDB's httpfs extension for S3 access.

        Supports:
        - Standard AWS S3 (auto-detects credentials from env/IAM role)
        - Custom S3-compatible endpoints (MinIO, LocalStack)
        - Region override

        Environment Variables Used:
        - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (explicit)
        - AWS_DEFAULT_REGION (region fallback)
        """
        if not self._connection or self._s3_initialized:
            return

        try:
            # Install httpfs extension (downloads once, cached afterwards)
            self._connection.execute("INSTALL httpfs")
            self._connection.execute("LOAD httpfs")

            # Region: explicit config → env → deployment region → default
            region = (
                self._s3_config.get("s3_region")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-1"
            )
            self._connection.execute(f"SET s3_region='{region}'")

            # Credentials: from env (standard AWS SDK convention)
            access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
            secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            if access_key and secret_key:
                self._connection.execute(f"SET s3_access_key_id='{access_key}'")
                self._connection.execute(f"SET s3_secret_access_key='{secret_key}'")
                logger.info("DuckDB S3: Configured with explicit AWS credentials")
            else:
                # Fall back to IAM role / instance metadata (EC2/ECS/Lambda)
                logger.info("DuckDB S3: No explicit credentials — using IAM role/instance metadata")

            # Session token (STS temporary credentials)
            session_token = os.environ.get("AWS_SESSION_TOKEN", "")
            if session_token:
                self._connection.execute(f"SET s3_session_token='{session_token}'")

            # Custom S3 endpoint (MinIO, LocalStack, etc.)
            endpoint = self._s3_config.get("s3_endpoint")
            if endpoint:
                self._connection.execute(f"SET s3_endpoint='{endpoint}'")
                self._connection.execute("SET s3_url_style='path'")
                self._connection.execute("SET s3_use_ssl=false")
                logger.info(f"DuckDB S3: Custom endpoint configured — {endpoint}")

            self._s3_initialized = True
            logger.info(f"DuckDB S3: httpfs loaded (region={region})")

        except Exception as e:
            logger.warning(f"DuckDB S3 configuration failed (non-fatal): {e}")
            self._s3_initialized = False

    def ensure_s3(self) -> bool:
        """
        Ensure S3 is configured. Lazily initializes if not already done.
        Returns True if S3 is available.
        """
        if self._s3_initialized:
            return True
        if not self._connection:
            return False
        # Only configure S3 if explicitly enabled via config
        if not self._s3_config.get("enable_s3", False):
            return False
        self._configure_s3()
        return self._s3_initialized

    def create_s3_view(
        self,
        view_name: str,
        s3_uri: str,
        format: str = "parquet",
    ) -> bool:
        """
        Create a DuckDB VIEW backed by an S3 Parquet file.

        This enables zero-copy SQL queries against cloud data:
            SELECT * FROM my_view WHERE revenue > 1000

        Args:
            view_name: SQL-safe name for the view
            s3_uri: Full S3 URI (s3://bucket/path/file.parquet)
            format: File format ('parquet' or 'csv')

        Returns:
            True if view created successfully
        """
        if not self.ensure_s3():
            logger.warning(f"Cannot create S3 view '{view_name}': S3 not configured")
            return False

        safe_name = self._sanitize_column_name(view_name)

        try:
            if format.lower() == "csv":
                reader = f"read_csv_auto('{s3_uri}')"
            else:
                reader = f"read_parquet('{s3_uri}')"

            self._connection.execute(
                f"CREATE OR REPLACE VIEW \"{safe_name}\" AS SELECT * FROM {reader}"
            )
            self._s3_views[safe_name] = s3_uri
            logger.info(f"Created S3 view '{safe_name}' → {s3_uri}")
            return True

        except Exception as e:
            logger.error(f"Failed to create S3 view '{safe_name}': {e}")
            return False

    def query_s3_parquet(
        self,
        s3_uri: str,
        sql_template: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Execute an ad-hoc SQL query directly against an S3 Parquet file.

        Args:
            s3_uri: Full S3 URI (s3://bucket/path/file.parquet)
            sql_template: Optional SQL with {table} placeholder.
                          Defaults to "SELECT * FROM {table}"

        Returns:
            Result DataFrame

        Example:
            df = engine.query_s3_parquet(
                "s3://my-bucket/data/sales.parquet",
                "SELECT product, SUM(revenue) FROM {table} GROUP BY product"
            )
        """
        if not self.ensure_s3():
            raise RuntimeError(
                "S3 not configured. Set DUCKDB_ENABLE_S3=true and provide AWS credentials."
            )

        reader = f"read_parquet('{s3_uri}')"
        sql = (sql_template or "SELECT * FROM {table}").replace("{table}", reader)

        # Validate the generated SQL
        is_valid, error = self.validate_sql(sql)
        if not is_valid:
            raise ValueError(f"Invalid SQL for S3 query: {error}")

        return self._connection.execute(sql).fetchdf()

    def list_s3_views(self) -> Dict[str, str]:
        """List all S3-backed views and their URIs."""
        return dict(self._s3_views)

    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for SQL compatibility."""
        # Replace special characters with underscore
        sanitized = re.sub(r"[^\w]", "_", str(name).strip().lower())
        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"col_{sanitized}"
        return sanitized or "unnamed"

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame for DuckDB registration (Production-Grade).
        
        Handles:
        1. Sanitize column names
        2. Serialize complex objects (numpy arrays, lists) to JSON
        3. Handle mixed types
        """
        import json
        import numpy as np
        
        df = df.copy()

        # 1. Sanitize column names
        new_columns = {}
        for col in df.columns:
            new_name = self._sanitize_column_name(col)
            # Handle duplicates
            original_new_name = new_name
            i = 1
            while new_name in new_columns.values():
                new_name = f"{original_new_name}_{i}"
                i += 1
            new_columns[col] = new_name
        df.columns = [new_columns[c] for c in df.columns]

        # 2. Robust Type Cleaning
        for col in df.columns:
            dtype = df[col].dtype
            
            # --- Handle Object Type Columns ---
            if dtype == object:
                # SAFETY FIRST: Sampling was causing issues (missed numpy arrays deep in file).
                # New approach: Check a larger sample (up to 1000) or just default to safe serialization
                # for columns that are likely to contain mixed data.
                
                # Check first 500 non-nulls (good balance for performance/safety)
                sample = df[col].dropna().head(500).tolist()
                
                has_complex = False
                if not sample:
                    # Empirically, empty object cols are often best treated as string
                    pass
                else:
                    # Check for complex types
                    has_complex = any(isinstance(x, (np.ndarray, list, dict, set, bytes)) for x in sample)
                
                if has_complex:
                    # Serialize complex objects to JSON string
                    def _safe_serialize(val):
                        if val is None: return None
                        if hasattr(val, 'tolist'): return json.dumps(val.tolist()) # Numpy
                        if isinstance(val, (list, dict, set)): return json.dumps(val)
                        if isinstance(val, bytes): return val.decode('ascii', errors='ignore')
                        return str(val)
                    
                    df[col] = df[col].apply(_safe_serialize)
                else:
                    # Even if no complex types found in sample, we should be careful with mixed str/numeric
                    try:
                        # Clean common financial formatting
                        series_clean = (
                            df[col].astype(str)
                            .str.replace(r'[\$,]', '', regex=True)
                            .str.replace(r'^\((.*)\)$', r'-\1', regex=True)
                            .str.replace('%', '', regex=False)
                        )
                        pd.to_numeric(series_clean, errors='raise')
                        df[col] = pd.to_numeric(series_clean, errors='coerce')
                    except Exception:
                        pass

            # --- Handle Category ---
            elif dtype.name == 'category':
                df[col] = df[col].astype(str).replace('nan', None)

            # --- Handle Datetimes ---
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                 if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                     df[col] = df[col].dt.tz_localize(None)

        return df



    def register_dataframe(
        self,
        table_name: str,
        df: pd.DataFrame,
        clean: bool = True
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Register DataFrame as DuckDB table/view.
        
        Args:
            table_name: Name for the table
            df: DataFrame to register
            clean: Whether to sanitize the DataFrame
            
        Returns:
            Tuple of (success, cleaned_dataframe)
        """
        if clean:
            df = self._clean_dataframe(df)

        # Sanitize table name
        safe_name = self._sanitize_column_name(table_name)
        self._registered_tables[safe_name] = df
        
        # Store mapping from original name to safe name
        if not hasattr(self, '_table_name_map'):
            self._table_name_map = {}
        self._table_name_map[table_name] = safe_name
        self._table_name_map[safe_name] = safe_name  # Identity mapping for safe names

        if self._connection:
            try:
                # Register as view
                self._connection.register(safe_name, df)
                logger.info(f"Registered table '{safe_name}' with {len(df)} rows")
                return True, df
            except Exception as e:
                logger.warning(f"DuckDB registration failed: {e}")

        logger.info(f"Pandas fallback: stored '{safe_name}' in memory")
        return True, df

    def get_safe_table_name(self, table_name: str) -> str:
        """Get the sanitized table name for a given original name."""
        if not hasattr(self, '_table_name_map'):
            return self._sanitize_column_name(table_name)
        return self._table_name_map.get(table_name, self._sanitize_column_name(table_name))

    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety using production-grade pattern matching.

        Checks:
        1. Whitelist: only SELECT / WITH / EXPLAIN allowed
        2. Blacklist: compiled dangerous-pattern regex scan
        3. Anti-hallucination: referenced tables must exist
        4. Optional sqlparse structural validation

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        sql_stripped = sql.strip()

        # ── Whitelist: only SELECT / WITH / EXPLAIN ──
        if not _SQL_ALLOWED_STATEMENTS.match(sql_stripped):
            first_word = sql_stripped.split()[0] if sql_stripped.split() else "EMPTY"
            logger.warning(f"SQL blocked: non-SELECT statement: {first_word}")
            return False, f"Only SELECT queries allowed, got: {first_word}"

        # ── Blacklist: compiled dangerous patterns ──
        for pattern in _SQL_DANGEROUS_PATTERNS:
            match = pattern.search(sql_stripped)
            if match:
                logger.warning(f"SQL injection attempt blocked: {match.group()}")
                return False, f"Dangerous SQL pattern detected: {match.group()}"

        # ── Anti-hallucination: verify referenced tables exist ──
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'

        referenced_tables = set()
        sql_keywords = ('select', 'where', 'group', 'order', 'having', 'limit',
                        'as', 'on', 'and', 'or', 'not', 'in', 'is', 'null',
                        'case', 'when', 'then', 'else', 'end')
        # DuckDB table-function calls (read_parquet, read_csv_auto, etc.) are not real tables
        duckdb_table_functions = (
            'read_parquet', 'read_csv', 'read_csv_auto', 'read_json',
            'read_json_auto', 'read_ndjson', 'generate_series', 'range',
            'glob', 'parquet_scan', 'csv_scan',
        )
        for match in re.finditer(from_pattern, sql_stripped, re.IGNORECASE):
            table_name = match.group(1).lower()
            if table_name not in sql_keywords and table_name not in duckdb_table_functions:
                referenced_tables.add(table_name)

        for match in re.finditer(join_pattern, sql_stripped, re.IGNORECASE):
            table_name = match.group(1).lower()
            if table_name not in sql_keywords and table_name not in duckdb_table_functions:
                referenced_tables.add(table_name)

        if referenced_tables:
            existing_tables = set(t.lower() for t in self._registered_tables.keys())
            missing_tables = referenced_tables - existing_tables
            if missing_tables:
                available = ', '.join(sorted(existing_tables)[:5])
                return False, f"Table not found: {', '.join(missing_tables)}. Available: {available}"

        # ── Optional sqlparse structural validation ──
        if SQLPARSE_AVAILABLE:
            try:
                parsed = sqlparse.parse(sql_stripped)
                if not parsed:
                    return False, "Failed to parse SQL"
                stmt = parsed[0]
                if stmt.get_type() != "SELECT":
                    return False, f"Only SELECT allowed, got: {stmt.get_type()}"
            except Exception as e:
                return False, f"SQL parse error: {e}"

        return True, ""

    def execute_df(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute SQL and return DataFrame result.
        
        Args:
            sql: SQL query string
            params: Optional parameters for parameterized queries
            
        Returns:
            Result DataFrame
        """
        is_valid, error = self.validate_sql(sql)
        if not is_valid:
            raise ValueError(f"Invalid SQL: {error}")

        # Try DuckDB first
        if self._connection:
            current_sql = sql
            max_retries = 3
            
            for attempt in range(max_retries + 1):
                try:
                    if params:
                        result = self._connection.execute(current_sql, params).fetchdf()
                    else:
                        result = self._connection.execute(current_sql).fetchdf()
                    return result
                except Exception as e:
                    # If this was the last attempt, log and break to fallback
                    if attempt == max_retries:
                        logger.warning(f"DuckDB execution failed after {max_retries} auto-fixes: {e}, trying Pandas")
                        break

                    # AUTO-FIX: Check for common column naming mismatch (numeric columns)
                    error_str = str(e)
                    fixed_sql = None
                    
                    if 'Referenced column' in error_str:
                        try:
                            # 1. Extract missing column
                            missing_match = re.search(r'Referenced column "([^"]+)" not found', error_str)
                            if missing_match:
                                missing = missing_match.group(1)
                                target_candidate = None
                                
                                # 2. Extract potential candidates from error
                                candidates_match = re.search(r'Candidate bindings: (.*)', error_str)
                                if candidates_match:
                                    # Parse list of candidates: "col_0", "col_1"
                                    candidate_list = re.findall(r'"([^"]+)"', candidates_match.group(1))
                                    
                                    # Search for a good match in candidates
                                    for cand in candidate_list:
                                        if cand == f"col_{missing}" or missing in cand or cand.endswith(missing):
                                            target_candidate = cand
                                            break
                                
                                # 3. FALLBACK: If matching numeric pattern (e.g. "1") and no generic candidate found,
                                # force 'col_1' pattern which is our standard sanitization
                                if not target_candidate and missing.isdigit():
                                    target_candidate = f"col_{missing}"

                                if target_candidate:
                                    logger.warning(f"Auto-fixing SQL column mismatch ({attempt+1}/{max_retries}): '{missing}' -> '{target_candidate}'")
                                    fixed_sql = current_sql.replace(f'"{missing}"', f'"{target_candidate}"')
                                    # Also fix unquoted variants
                                    fixed_sql = re.sub(rf'\b{re.escape(missing)}\b', target_candidate, fixed_sql)
                        except Exception as e_fix:
                            logger.warning(f"Auto-fix logic failed: {e_fix}")
                            pass
                    
                    if fixed_sql and fixed_sql != current_sql:
                        current_sql = fixed_sql
                        continue # Retry with new SQL
                    else:
                        # Cannot fix, break to fallback
                        logger.warning(f"DuckDB execution failed: {e}, trying Pandas")
                        break

        # Pandas fallback
        return self._execute_pandas_fallback(sql)

    def execute_scalar(self, sql: str) -> Any:
        """Execute SQL and return single scalar value."""
        df = self.execute_df(sql)
        if df.empty:
            return None
        return df.iloc[0, 0]

    def _execute_pandas_fallback(self, sql: str) -> pd.DataFrame:
        """Execute SQL-like query using Pandas."""
        logger.info("Using Pandas fallback for SQL execution")

        # Simple SQL parser for basic queries
        sql_upper = sql.upper()
        
        # Extract table name
        from_match = re.search(r"FROM\s+(\w+)", sql_upper)
        if not from_match:
            raise ValueError("Could not parse table name from SQL")
        
        table_name = from_match.group(1).lower()
        if table_name not in self._registered_tables:
            raise ValueError(f"Table not found: {table_name}")

        df = self._registered_tables[table_name].copy()

        # Simple SUM extraction
        sum_match = re.search(r"SUM\s*\(\s*[\"']?(\w+)[\"']?\s*\)", sql, re.IGNORECASE)
        if sum_match:
            col_name = sum_match.group(1).lower()
            matching_cols = [c for c in df.columns if col_name in c.lower()]
            if matching_cols:
                result = df[matching_cols[0]].sum()
                return pd.DataFrame({"result": [result]})

        # Simple AVG extraction
        avg_match = re.search(r"AVG\s*\(\s*[\"']?(\w+)[\"']?\s*\)", sql, re.IGNORECASE)
        if avg_match:
            col_name = avg_match.group(1).lower()
            matching_cols = [c for c in df.columns if col_name in c.lower()]
            if matching_cols:
                result = df[matching_cols[0]].mean()
                return pd.DataFrame({"result": [result]})

        # Return full table for SELECT *
        if "SELECT *" in sql_upper or "SELECT * FROM" in sql_upper:
            return df

        logger.warning("Could not parse SQL for Pandas fallback")
        return pd.DataFrame()

    def get_table_schema(self, table_name: str) -> str:
        """Get schema information for a registered table."""
        safe_name = self._sanitize_column_name(table_name)
        
        if safe_name not in self._registered_tables:
            return f"Table '{table_name}' not found"

        df = self._registered_tables[safe_name]
        lines = [f"Table: {safe_name} ({len(df)} rows)"]
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(s)[:20] for s in sample)
            lines.append(f"  - {col}: {dtype} (e.g., {sample_str})")

        return "\n".join(lines)

    def list_tables(self) -> List[str]:
        """List all registered tables."""
        return list(self._registered_tables.keys())

    def close(self):
        """Close DuckDB connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# Singleton instance
_sql_engine: Optional[SQLEngine] = None


def get_sql_engine() -> SQLEngine:
    """Get or create singleton SQL engine with full config."""
    global _sql_engine
    
    if _sql_engine is None:
        # Use sensible defaults - don't depend on settings to avoid parsing errors
        memory_limit = "2GB"
        threads = 4
        s3_config: Dict[str, Any] = {}
        perf_config: Dict[str, Any] = {}
        
        try:
            from app.config import settings
            duck = settings.duckdb
            memory_limit = duck.memory_limit
            threads = duck.threads
            
            # S3 config from settings
            s3_config = {
                "enable_s3": duck.enable_s3,
                "s3_region": duck.s3_region,
                "s3_endpoint": duck.s3_endpoint,
            }
            
            # Performance config from settings
            perf_config = {
                "enable_object_cache": duck.enable_object_cache,
                "object_cache_size": duck.object_cache_size,
                "preserve_insertion_order": duck.preserve_insertion_order,
                "enable_parallel_csv": duck.enable_parallel_csv,
            }
        except Exception:
            # Use defaults if settings fail
            pass
        
        _sql_engine = SQLEngine(
            memory_limit=memory_limit,
            threads=threads,
            s3_config=s3_config,
            perf_config=perf_config,
        )
    
    return _sql_engine


# ═══════════════════════════════════════════════════════════════════
# STANDALONE SECURITY UTILITIES
# Callable without an SQLEngine instance.
# ═══════════════════════════════════════════════════════════════════

def sanitize_sql(sql: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and sanitize LLM-generated SQL (standalone, no engine needed).

    Args:
        sql: Raw SQL string from LLM

    Returns:
        Tuple of (is_safe, sanitized_sql, error_message)
    """
    if not sql or not sql.strip():
        return False, "", "Empty SQL"

    sql = sql.strip()

    # Whitelist check
    if not _SQL_ALLOWED_STATEMENTS.match(sql):
        first_word = sql.split()[0] if sql.split() else "EMPTY"
        logger.warning(f"SQL blocked: non-SELECT statement: {first_word}")
        return False, "", f"Only SELECT queries allowed, got: {first_word}"

    # Blacklist check
    for pattern in _SQL_DANGEROUS_PATTERNS:
        match = pattern.search(sql)
        if match:
            logger.warning(f"SQL injection attempt blocked: {match.group()}")
            return False, "", f"Dangerous SQL pattern detected: {match.group()}"

    # Strip trailing semicolons (prevent multi-statement injection)
    sql = sql.rstrip(";").strip()

    # Remove SQL comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    sql = sql.strip()

    return True, sql, None


def validate_file_path(
    path: str,
    allowed_base_dirs: Optional[List[str]] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a file path against traversal attacks.

    Args:
        path: File path to validate
        allowed_base_dirs: Optional whitelist of base directories

    Returns:
        Tuple of (is_safe, resolved_path, error_message)
    """
    if not path or not path.strip():
        return False, "", "Empty path"

    for pattern in _PATH_TRAVERSAL_PATTERNS:
        if pattern.search(path):
            logger.warning(f"Path traversal attempt: {path}")
            return False, "", f"Path traversal detected: {path}"

    try:
        resolved = os.path.realpath(os.path.abspath(path))
    except Exception as e:
        return False, "", f"Path resolution failed: {e}"

    if allowed_base_dirs:
        in_allowed = False
        for base_dir in allowed_base_dirs:
            base_resolved = os.path.realpath(os.path.abspath(base_dir))
            if resolved.startswith(base_resolved):
                in_allowed = True
                break
        if not in_allowed:
            logger.warning(f"Path outside allowed dirs: {resolved}")
            return False, "", "Path outside allowed directories"

    return True, resolved, None


def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate LLM-generated Python code for sandbox execution.

    Args:
        code: Python code string

    Returns:
        Tuple of (is_safe, error_message)
    """
    if not code or not code.strip():
        return False, "Empty code"

    for pattern in _PYTHON_DANGEROUS_PATTERNS:
        match = pattern.search(code)
        if match:
            logger.warning(f"Dangerous Python pattern blocked: {match.group()}")
            return False, f"Unsafe code pattern: {match.group()}"

    return True, None


def validate_client_id(client_id: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and normalize a client ID.

    Args:
        client_id: Raw client ID

    Returns:
        Tuple of (is_valid, normalized_id, error_message)
    """
    if not client_id or not client_id.strip():
        return False, "", "Empty client ID"

    normalized = client_id.strip().lower()

    if not _CLIENT_ID_PATTERN.match(normalized):
        return False, "", f"Invalid client ID format: {client_id}"

    return True, normalized, None


__all__ = [
    "SQLEngine",
    "get_sql_engine",
    "sanitize_sql",
    "validate_file_path",
    "validate_python_code",
    "validate_client_id",
]
