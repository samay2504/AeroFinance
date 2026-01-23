"""
SQL Engine - DuckDB helper for analytical queries with Pandas fallback.
Handles DataFrame registration, safe parameterized queries, and type coercion.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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


class SQLEngine:
    """
    DuckDB SQL engine with:
    - DataFrame registration as views
    - Safe parameterized queries
    - Automatic type coercion and cleaning
    - Pandas fallback when DuckDB fails
    """

    def __init__(self, memory_limit: str = "2GB", threads: int = 4):
        self._connection: Optional["duckdb.DuckDBPyConnection"] = None
        self._registered_tables: Dict[str, pd.DataFrame] = {}
        self.memory_limit = memory_limit
        self.threads = threads

        if DUCKDB_AVAILABLE:
            self._init_duckdb()

    def _init_duckdb(self):
        """Initialize DuckDB connection with configuration."""
        try:
            self._connection = duckdb.connect(":memory:")
            self._connection.execute(f"SET memory_limit='{self.memory_limit}'")
            self._connection.execute(f"SET threads={self.threads}")
            logger.info(f"DuckDB initialized (memory={self.memory_limit}, threads={self.threads})")
        except Exception as e:
            logger.error(f"DuckDB initialization failed: {e}")
            self._connection = None

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
        Validate SQL query for safety.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        sql_upper = sql.strip().upper()

        # Only allow SELECT statements
        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT statements allowed"

        # Block dangerous keywords
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "EXEC"]
        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False, f"Dangerous keyword not allowed: {keyword}"

        # ANTI-HALLUCINATION: Check that referenced tables exist
        # Extract table names from FROM and JOIN clauses
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        referenced_tables = set()
        for match in re.finditer(from_pattern, sql, re.IGNORECASE):
            table_name = match.group(1).lower()
            # Skip common SQL keywords that might be matched
            if table_name not in ('select', 'where', 'group', 'order', 'having', 'limit'):
                referenced_tables.add(table_name)
        
        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            table_name = match.group(1).lower()
            referenced_tables.add(table_name)
        
        # Check if all referenced tables exist
        if referenced_tables:
            existing_tables = set(t.lower() for t in self._registered_tables.keys())
            missing_tables = referenced_tables - existing_tables
            
            if missing_tables:
                # Provide helpful error message with available tables
                available = ', '.join(sorted(existing_tables)[:5])
                return False, f"Table not found: {', '.join(missing_tables)}. Available: {available}"

        # Parse with sqlparse if available
        if SQLPARSE_AVAILABLE:
            try:
                parsed = sqlparse.parse(sql)
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
    """Get or create singleton SQL engine."""
    global _sql_engine
    
    if _sql_engine is None:
        # Use sensible defaults - don't depend on settings to avoid parsing errors
        memory_limit = "2GB"
        threads = 4
        
        try:
            from app.config import settings
            memory_limit = settings.duckdb.memory_limit
            threads = settings.duckdb.threads
        except Exception:
            # Use defaults if settings fail
            pass
        
        _sql_engine = SQLEngine(
            memory_limit=memory_limit,
            threads=threads
        )
    
    return _sql_engine


__all__ = ["SQLEngine", "get_sql_engine"]
