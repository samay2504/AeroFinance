"""
Excel Ingest Engine - Robust multi-sheet XLSX/CSV ingestion with:
- Auto-detection of best sheet by data density
- Multi-sheet ingestion option
- Column normalization and type coercion
- Period column detection (FY21, 9MFY22, Q1FY23, etc.)
"""

import logging
import re
import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ExcelIngestor:
    """
    Production-grade Excel/CSV ingestion engine.
    Handles multi-sheet detection, column normalization, and type coercion.
    """

    # Period patterns for financial data
    PERIOD_PATTERNS = [
        (r"^fy\d{2}$", "fiscal_year"),  # FY21, FY22
        (r"^\d{1,2}mfy\d{2}$", "period_fy"),  # 9MFY22, 3MFY21
        (r"^q\d[_-]?fy\d{2}$", "quarter"),  # Q1FY22, Q2-FY23
        (r"^q\d[_-]?\d{2}$", "quarter"),  # Q121, Q2-22
        (r"^h\d[_-]?fy\d{2}$", "half_year"),  # H1FY22
        (r"^\d{4}$", "year"),  # 2021, 2022
        (
            r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[_-]?\d{2,4}$",
            "month_year",
        ),
    ]

    # NA-like values to replace
    NA_VALUES = [
        "Na/p",
        "N/A",
        "n/a",
        "NA",
        "-",
        "--",
        "None",
        "none",
        "NULL",
        "null",
        "N.A.",
        "n.a.",
    ]

    def __init__(self):
        self._last_sheets_info: List[Dict[str, Any]] = []

    def _score_sheet(self, df: pd.DataFrame) -> float:
        """
        Score a sheet by data quality for auto-selection.
        Higher score = better quality data sheet.
        """
        if df is None or df.empty:
            return 0.0

        total_cells = df.size
        non_null = df.count().sum()

        # Base score: non-null density
        density = non_null / max(total_cells, 1)

        # Bonus for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        numeric_bonus = min(numeric_cols / max(df.shape[1], 1) * 0.2, 0.2)

        # Bonus for reasonable row count
        row_bonus = min(len(df) / 1000, 0.1)

        # Penalty for too many unnamed columns
        unnamed_ratio = sum(
            1 for c in df.columns if str(c).startswith("Unnamed")
        ) / max(len(df.columns), 1)
        unnamed_penalty = unnamed_ratio * 0.3

        return density + numeric_bonus + row_bonus - unnamed_penalty

    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name to consistent format."""
        name = str(name).strip().lower()
        # Replace special chars with underscore
        name = re.sub(r"[^\w\s]", "_", name)
        # Replace whitespace with underscore
        name = re.sub(r"\s+", "_", name)
        # Remove multiple underscores
        name = re.sub(r"_+", "_", name)
        # Strip leading/trailing underscores
        name = name.strip("_")
        return name or "unnamed"

    def _detect_period_column(self, col_name: str) -> Optional[str]:
        """Detect if column is a period column and return type."""
        name_lower = col_name.lower().strip()
        for pattern, period_type in self.PERIOD_PATTERNS:
            if re.match(pattern, name_lower):
                return period_type
        return None

    def _coerce_numeric_column(self, series: pd.Series) -> pd.Series:
        """Attempt to coerce string column to numeric."""
        if series.dtype != object:
            return series

        # Replace NA-like values
        series = series.replace(self.NA_VALUES, np.nan)

        # Try conversion
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return series

        try:
            # Clean common patterns
            cleaned = (
                series.astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(r"^\(([\d.]+)\)$", r"-\1", regex=True)  # (100) -> -100
                .str.replace(r"^₹\s*", "", regex=True)  # Remove INR symbol
                .str.replace(r"^\$\s*", "", regex=True)  # Remove USD symbol
            )

            # Test if conversion works on sample
            test = pd.to_numeric(cleaned.dropna().head(20), errors="raise")

            # Apply full conversion
            return pd.to_numeric(cleaned, errors="coerce")
        except (ValueError, TypeError):
            return series

    def _preprocess_dataframe(
        self, df: pd.DataFrame, sheet_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess DataFrame with normalization and type coercion.

        Returns:
            Tuple of (cleaned_dataframe, preprocessing_report)
        """
        report = {
            "original_shape": df.shape,
            "columns_normalized": [],
            "numeric_coerced": [],
            "period_columns": [],
            "na_replaced": 0,
        }

        df = df.copy()

        # Drop completely empty rows/columns
        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)

        # Normalize column names
        new_columns = {}
        for col in df.columns:
            new_name = self._normalize_column_name(col)
            # Handle duplicates
            if new_name in new_columns.values():
                i = 1
                while f"{new_name}_{i}" in new_columns.values():
                    i += 1
                new_name = f"{new_name}_{i}"
            new_columns[col] = new_name
            if str(col) != new_name:
                report["columns_normalized"].append(
                    {"original": str(col), "normalized": new_name}
                )

        df.columns = [new_columns[c] for c in df.columns]

        # Detect period columns
        for col in df.columns:
            period_type = self._detect_period_column(col)
            if period_type:
                report["period_columns"].append({"column": col, "type": period_type})

        # Coerce numeric columns and replace NA values
        for col in df.columns:
            original_na_count = df[col].isna().sum()
            df[col] = self._coerce_numeric_column(df[col])
            new_na_count = df[col].isna().sum()

            if new_na_count > original_na_count:
                report["na_replaced"] += new_na_count - original_na_count

            if df[col].dtype in [np.float64, np.int64] and col not in [
                r["column"] for r in report.get("period_columns", [])
            ]:
                report["numeric_coerced"].append(col)

        report["final_shape"] = df.shape
        report["sheet_name"] = sheet_name

        return df, report

    def read_sheets_info(
        self, file_content: bytes, filename: str
    ) -> List[Dict[str, Any]]:
        """
        Read and score all sheets in an Excel file.

        Returns:
            List of sheet info dicts with name, rows, columns, score
        """
        sheets_info = []

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(file_content))
                score = self._score_sheet(df)
                sheets_info.append(
                    {
                        "name": "csv_data",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "score": score,
                        "column_names": list(df.columns)[:10],
                    }
                )
            else:
                xl = pd.ExcelFile(io.BytesIO(file_content))
                for sheet_name in xl.sheet_names:
                    try:
                        df = xl.parse(sheet_name, nrows=500)  # Read sample for scoring
                        score = self._score_sheet(df)
                        sheets_info.append(
                            {
                                "name": sheet_name,
                                "rows": len(df),
                                "columns": len(df.columns),
                                "score": score,
                                "column_names": list(df.columns)[:10],
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read sheet {sheet_name}: {e}")
                        sheets_info.append(
                            {
                                "name": sheet_name,
                                "rows": 0,
                                "columns": 0,
                                "score": 0.0,
                                "error": str(e),
                            }
                        )
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            return []

        # Sort by score
        sheets_info.sort(key=lambda x: x.get("score", 0), reverse=True)
        self._last_sheets_info = sheets_info
        return sheets_info

    def read_sheets_info_from_path(
        self, file_path: Path, filename: str
    ) -> List[Dict[str, Any]]:
        """
        Read and score all sheets from a local file path.
        """
        sheets_info: List[Dict[str, Any]] = []

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path)
                score = self._score_sheet(df)
                sheets_info.append(
                    {
                        "name": "csv_data",
                        "rows": len(df),
                        "columns": len(df.columns),
                        "score": score,
                        "column_names": list(df.columns)[:10],
                    }
                )
            else:
                xl = pd.ExcelFile(file_path)
                for sheet_name in xl.sheet_names:
                    try:
                        df = xl.parse(sheet_name, nrows=500)
                        score = self._score_sheet(df)
                        sheets_info.append(
                            {
                                "name": sheet_name,
                                "rows": len(df),
                                "columns": len(df.columns),
                                "score": score,
                                "column_names": list(df.columns)[:10],
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read sheet {sheet_name}: {e}")
                        sheets_info.append(
                            {
                                "name": sheet_name,
                                "rows": 0,
                                "columns": 0,
                                "score": 0.0,
                                "error": str(e),
                            }
                        )
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            return []

        sheets_info.sort(key=lambda x: x.get("score", 0), reverse=True)
        self._last_sheets_info = sheets_info
        return sheets_info

    def ingest_best_sheet(
        self,
        file_content: bytes,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest the best (highest scoring) sheet from the file.

        Returns:
            Ingestion result with dataset_id and metadata
        """
        sheets_info = self.read_sheets_info(file_content, filename)

        if not sheets_info:
            return {"error": "No readable sheets found", "success": False}

        best_sheet = sheets_info[0]
        sheet_name = best_sheet["name"]

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)

            df, report = self._preprocess_dataframe(df, sheet_name)

            # Generate dataset ID
            safe_filename = re.sub(r"[^\w]", "_", Path(filename).stem.lower())
            safe_sheet = re.sub(r"[^\w]", "_", sheet_name.lower())
            dataset_id = f"{client_id}:{safe_filename}:{safe_sheet}"

            metadata = {
                "filename": filename,
                "sheet_name": sheet_name,
                "preprocessing": report,
                "client_id": client_id,
            }

            if register_callback:
                register_callback(dataset_id, df, metadata)

            return {
                "success": True,
                "dataset_id": dataset_id,
                "rows": len(df),
                "columns": list(df.columns),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Failed to ingest {sheet_name}: {e}")
            return {"error": str(e), "success": False}

    def ingest_best_sheet_from_path(
        self,
        file_path: Path,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest the best sheet from a local file path.
        """
        sheets_info = self.read_sheets_info_from_path(file_path, filename)

        if not sheets_info:
            return {"error": "No readable sheets found", "success": False}

        best_sheet = sheets_info[0]
        sheet_name = best_sheet["name"]

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            df, report = self._preprocess_dataframe(df, sheet_name)

            safe_filename = re.sub(r"[^\w]", "_", Path(filename).stem.lower())
            safe_sheet = re.sub(r"[^\w]", "_", sheet_name.lower())
            dataset_id = f"{client_id}:{safe_filename}:{safe_sheet}"

            metadata = {
                "filename": filename,
                "sheet_name": sheet_name,
                "preprocessing": report,
                "client_id": client_id,
            }

            if register_callback:
                register_callback(dataset_id, df, metadata)

            return {
                "success": True,
                "dataset_id": dataset_id,
                "rows": len(df),
                "columns": list(df.columns),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Failed to ingest {sheet_name}: {e}")
            return {"error": str(e), "success": False}

    def ingest_all_sheets(
        self,
        file_content: bytes,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ingest all sheets from the file.

        Returns:
            List of ingestion results for each sheet
        """
        results = []
        sheets_info = self.read_sheets_info(file_content, filename)

        if not sheets_info:
            return [{"error": "No readable sheets found", "success": False}]

        for sheet_info in sheets_info:
            sheet_name = sheet_info["name"]

            if sheet_info.get("error"):
                results.append(
                    {
                        "sheet_name": sheet_name,
                        "success": False,
                        "error": sheet_info["error"],
                    }
                )
                continue

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(file_content))
                else:
                    df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)

                if df.empty or df.shape[0] < 2:
                    results.append(
                        {
                            "sheet_name": sheet_name,
                            "success": False,
                            "error": "Empty or minimal data",
                        }
                    )
                    continue

                df, report = self._preprocess_dataframe(df, sheet_name)

                # Generate dataset ID
                safe_filename = re.sub(r"[^\w]", "_", Path(filename).stem.lower())
                safe_sheet = re.sub(r"[^\w]", "_", sheet_name.lower())
                dataset_id = f"{client_id}:{safe_filename}:{safe_sheet}"

                metadata = {
                    "filename": filename,
                    "sheet_name": sheet_name,
                    "preprocessing": report,
                    "client_id": client_id,
                }

                if register_callback:
                    register_callback(dataset_id, df, metadata)

                results.append(
                    {
                        "success": True,
                        "dataset_id": dataset_id,
                        "sheet_name": sheet_name,
                        "rows": len(df),
                        "columns": list(df.columns),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to ingest {sheet_name}: {e}")
                results.append(
                    {"sheet_name": sheet_name, "success": False, "error": str(e)}
                )

        return results

    def ingest_all_sheets_from_path(
        self,
        file_path: Path,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ingest all sheets from a local file path.
        """
        results: List[Dict[str, Any]] = []
        sheets_info = self.read_sheets_info_from_path(file_path, filename)

        if not sheets_info:
            return [{"error": "No readable sheets found", "success": False}]

        for sheet_info in sheets_info:
            sheet_name = sheet_info["name"]

            if sheet_info.get("error"):
                results.append(
                    {
                        "sheet_name": sheet_name,
                        "success": False,
                        "error": sheet_info["error"],
                    }
                )
                continue

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)

                if df.empty or df.shape[0] < 2:
                    results.append(
                        {
                            "sheet_name": sheet_name,
                            "success": False,
                            "error": "Empty or minimal data",
                        }
                    )
                    continue

                df, report = self._preprocess_dataframe(df, sheet_name)

                safe_filename = re.sub(r"[^\w]", "_", Path(filename).stem.lower())
                safe_sheet = re.sub(r"[^\w]", "_", sheet_name.lower())
                dataset_id = f"{client_id}:{safe_filename}:{safe_sheet}"

                metadata = {
                    "filename": filename,
                    "sheet_name": sheet_name,
                    "preprocessing": report,
                    "client_id": client_id,
                }

                if register_callback:
                    register_callback(dataset_id, df, metadata)

                results.append(
                    {
                        "success": True,
                        "dataset_id": dataset_id,
                        "sheet_name": sheet_name,
                        "rows": len(df),
                        "columns": list(df.columns),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to ingest {sheet_name}: {e}")
                results.append(
                    {"sheet_name": sheet_name, "success": False, "error": str(e)}
                )

        return results

    def get_last_sheets_info(self) -> List[Dict[str, Any]]:
        """Get info from last file read."""
        return self._last_sheets_info


# ═══════════════════════════════════════════════════════════════════
# DataFrame Token Compression + Intelligent EDA
# (PRD: Performance Optimization · Data Understanding · Audit-Ready)
#
# Reduces LLM context usage by 60-80% via structural representation
# while providing production-grade preprocessing intelligence:
#   • Smart column classification (metric / dimension / temporal / ID)
#   • Data quality scoring (completeness, consistency, outlier %)
#   • IQR-based outlier flagging (audit red-flags)
#   • Distribution profiling (skewness, kurtosis)
#   • Correlation insights (top correlated pairs)
#   • Temporal growth trend detection (YoY / QoQ signals)
#   • Financial ratio & pattern recognition
#
# All features are READ-ONLY — zero data loss, zero mutation.
# ═══════════════════════════════════════════════════════════════════

# ── Column Classification — Single Source of Truth ────────────────
# Keywords are dynamically derived from FinancialNER.METRIC_CATEGORIES
# in app/core/semantic_understanding.py.
# Import is lazy + cached; a complete fallback dictionary is provided
# in case of circular-import or missing-dependency failures.

_KEYWORD_CACHE: Optional[Dict[str, frozenset]] = None

# ── Canonical fallback sets (used ONLY when FinancialNER is unavailable) ──
_FALLBACK_TEMPORAL = frozenset({
    "fy", "quarter", "q1", "q2", "q3", "q4", "period", "date",
    "time", "year", "month", "day", "week", "half", "h1", "h2",
    "ytd", "mtd", "qtd", "9mfy", "3mfy", "6mfy",
})
_FALLBACK_ID = frozenset({
    "id", "key", "code", "ref", "number", "no", "sr", "index",
})
_FALLBACK_RATIO = frozenset({
    "margin", "ratio", "pct", "percent", "%", "growth", "yield",
    "return", "roe", "roa", "roce", "eps", "pe", "pb",
    "debt_equity", "current_ratio", "quick_ratio", "leverage",
    "ebitda_margin", "pat_margin", "npm", "opm",
})
_FALLBACK_METRIC = frozenset({
    "revenue", "sales", "income", "profit", "loss", "expense",
    "cost", "ebitda", "ebit", "pat", "pbt", "depreciation",
    "amortization", "tax", "interest", "dividend", "capex",
    "cash", "debt", "equity", "asset", "liability", "reserve",
    "turnover", "provision", "impairment", "goodwill",
    "receivable", "payable", "inventory", "working_capital",
})

# Compiled once — strips ALL non-alphanumeric characters from regex patterns
# so  r'operating\s*expense'  →  'operating expense'  →  ['operating', 'expense']
_RE_ALPHA_ONLY = re.compile(r"[^a-z0-9]+")


def _extract_words_from_patterns(patterns: List[str], min_len: int = 2) -> set:
    """
    Extract plain English words from a list of regex pattern strings.

    Strategy: replace every non-alphanumeric character with a space,
    split on whitespace, and keep tokens ≥ *min_len*.
    This cleanly handles patterns like  r'net\\s*sales' → {'net', 'sales'}.
    """
    words: set = set()
    for pat in patterns:
        cleaned = _RE_ALPHA_ONLY.sub(" ", pat.lower())
        for tok in cleaned.split():
            if len(tok) >= min_len:
                words.add(tok)
    return words


def _build_keyword_sets() -> Dict[str, frozenset]:
    """
    Lazily build column-classification keyword sets from the canonical
    ``FinancialNER.METRIC_CATEGORIES`` (single source of truth).

    Keyword sets are cached after the first call.  If the import fails
    (e.g. circular import during early bootup), a complete set of
    hardcoded fallback keywords is used instead.
    """
    global _KEYWORD_CACHE
    if _KEYWORD_CACHE is not None:
        return _KEYWORD_CACHE

    try:
        from app.core.semantic_understanding import FinancialNER
        cats = FinancialNER.METRIC_CATEGORIES

        # ── Ratio / Performance keywords ──────────────────────────
        ratios = _extract_words_from_patterns(
            cats.get("ratio", []) + cats.get("performance", [])
        )
        # Supplement with common financial acronyms handled by
        # FinancialNER._expand_financial_acronyms but not present
        # as regex patterns in METRIC_CATEGORIES.
        ratios.update({
            "roe", "roa", "roce", "eps", "pe", "pb", "npm", "opm",
            "margin", "pct", "percent", "%", "growth", "yield",
            "leverage", "debt_equity", "current_ratio", "quick_ratio",
        })

        # ── Financial metric keywords ─────────────────────────────
        metric_cats = ("revenue", "expense", "profit", "asset",
                       "liability", "equity", "cashflow", "volume")
        metrics: set = set()
        for cat_key in metric_cats:
            metrics |= _extract_words_from_patterns(cats.get(cat_key, []))

        logger.debug(
            "Column keywords derived from FinancialNER (%d ratio, %d metric)",
            len(ratios), len(metrics),
        )

        _KEYWORD_CACHE = {
            "temporal": _FALLBACK_TEMPORAL,  # temporal/id not in METRIC_CATEGORIES
            "id":       _FALLBACK_ID,
            "ratio":    frozenset(ratios),
            "metric":   frozenset(metrics),
        }

    except Exception as exc:
        logger.debug(
            "FinancialNER unavailable (%s); using fallback keyword sets", exc
        )
        _KEYWORD_CACHE = {
            "temporal": _FALLBACK_TEMPORAL,
            "id":       _FALLBACK_ID,
            "ratio":    _FALLBACK_RATIO,
            "metric":   _FALLBACK_METRIC,
        }

    return _KEYWORD_CACHE



def _classify_column(col: str, series: pd.Series, n_rows: int) -> str:
    """
    Classify a column into one of: temporal, identifier, ratio,
    financial_metric, dimension, or metric.

    Uses both name heuristics AND data characteristics (uniqueness ratio,
    dtype, value range) for robust, adaptive classification.
    """
    col_lower = col.lower().replace(" ", "_")
    tokens = set(re.split(r"[_\s\-\.]+", col_lower))

    kw = _build_keyword_sets()

    # 1. Temporal — name-based
    if tokens & kw["temporal"]:
        return "temporal"

    # 2. Identifier — name-based + high cardinality
    if tokens & kw["id"]:
        return "identifier"

    # 3. Ratio/percentage — name or value-range based
    if tokens & kw["ratio"]:
        return "ratio"
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            # Values in [0, 1] or [0, 100] with small range → likely ratio/pct
            vmin, vmax = clean.min(), clean.max()
            if 0 <= vmin and vmax <= 1.0 and (vmax - vmin) < 1.0:
                return "ratio"
            if 0 <= vmin and vmax <= 100 and "%" in col:
                return "ratio"

    # 4. Financial metric — name-based
    if tokens & kw["metric"]:
        return "financial_metric"

    # 5. Dimension vs Metric — data-driven
    if series.dtype == "object" or str(series.dtype) == "category":
        nunique = series.nunique()
        # Low cardinality relative to rows → dimension
        if nunique < max(n_rows * 0.5, 20):
            return "dimension"
        return "identifier"

    if pd.api.types.is_numeric_dtype(series):
        return "metric"

    return "metric"


def _column_quality_score(series: pd.Series) -> dict:
    """
    Compute a per-column data quality score (0–100).

    Components:
      completeness  — % of non-null values
      consistency   — % of values that match the dominant dtype
      outlier_free  — % of values inside 1.5×IQR (numeric only)

    Returns dict with individual scores and weighted aggregate.
    """
    n = len(series)
    if n == 0:
        return {"completeness": 0, "consistency": 0, "outlier_free": 100, "quality": 0}

    # Completeness
    completeness = round(series.notna().sum() / n * 100, 1)

    # Consistency — for object cols, check if coercion to numeric works uniformly
    consistency = 100.0
    if series.dtype == "object":
        non_null = series.dropna()
        if len(non_null) > 0:
            numeric_coerced = pd.to_numeric(non_null, errors="coerce")
            numeric_frac = numeric_coerced.notna().sum() / len(non_null)
            # If most values are numeric-like or most are strings → consistent
            consistency = round(max(numeric_frac, 1 - numeric_frac) * 100, 1)

    # Outlier-free (numeric only, using IQR)
    outlier_free = 100.0
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) >= 4:
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_outliers = ((clean < lower) | (clean > upper)).sum()
                outlier_free = round((1 - n_outliers / len(clean)) * 100, 1)

    # Weighted aggregate (completeness matters most for data trust)
    quality = round(
        completeness * 0.50 + consistency * 0.25 + outlier_free * 0.25, 1
    )
    return {
        "completeness": completeness,
        "consistency": consistency,
        "outlier_free": outlier_free,
        "quality": quality,
    }


def _detect_outliers_iqr(series: pd.Series, col_name: str) -> List[dict]:
    """
    Detect outliers using the IQR method.  Returns list of
    {index, value, direction} for audit flagging.
    Capped at 5 outliers to keep output concise.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return []
    clean = series.dropna()
    if len(clean) < 4:
        return []
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = []
    for idx, val in clean.items():
        if val < lower:
            outliers.append({"row": idx, "value": round(float(val), 2), "dir": "LOW"})
        elif val > upper:
            outliers.append({"row": idx, "value": round(float(val), 2), "dir": "HIGH"})
        if len(outliers) >= 5:
            break
    return outliers


def _distribution_profile(series: pd.Series) -> Optional[dict]:
    """
    Compute distribution shape metrics for a numeric column.
    Returns skewness + kurtosis + interpretation.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None
    clean = series.dropna()
    if len(clean) < 8:  # Need enough samples for meaningful stats
        return None
    try:
        skew = float(clean.skew())
        kurt = float(clean.kurtosis())
    except (ValueError, TypeError):
        return None

    # Human-readable interpretation
    if abs(skew) < 0.5:
        shape = "symmetric"
    elif skew > 0:
        shape = "right-skewed (long tail of high values)"
    else:
        shape = "left-skewed (long tail of low values)"

    if kurt > 3:
        tail = "heavy-tailed (leptokurtic — extreme values likely)"
    elif kurt < -1:
        tail = "light-tailed (platykurtic — values concentrated near mean)"
    else:
        tail = "normal-tailed (mesokurtic)"

    return {
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "shape": shape,
        "tail": tail,
    }


def _top_correlations(df: pd.DataFrame, columns: List[str], top_n: int = 5) -> List[dict]:
    """
    Find the top N most correlated column pairs.
    Only considers numeric columns. Returns pairs with |r| > 0.5.
    """
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return []
    try:
        corr = df[numeric_cols].corr()
    except Exception:
        return []

    pairs = []
    seen = set()
    for i, c1 in enumerate(numeric_cols):
        for j, c2 in enumerate(numeric_cols):
            if i >= j:
                continue
            key = (c1, c2)
            if key in seen:
                continue
            seen.add(key)
            r = corr.loc[c1, c2]
            if pd.notna(r) and abs(r) > 0.5:
                pairs.append({
                    "col_a": c1,
                    "col_b": c2,
                    "r": round(float(r), 3),
                    "strength": (
                        "strong" if abs(r) > 0.8 else "moderate"
                    ),
                })
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pairs[:top_n]


def _temporal_growth_trends(
    df: pd.DataFrame, columns: List[str], period_cols: List[str],
) -> List[str]:
    """
    Detect growth trends across period-indexed data.

    If the DataFrame has period columns as headers (transposed financial
    statements), compute period-over-period growth for metric rows.
    Returns human-readable trend descriptions.
    """
    trends: List[str] = []
    if not period_cols:
        return trends

    # Case 1: Period columns are actual DataFrame columns (transposed layout)
    numeric_period_cols = [
        c for c in period_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_period_cols) >= 2:
        # Compare last two period columns for each row
        col_prev, col_curr = numeric_period_cols[-2], numeric_period_cols[-1]
        # Find a label column (first non-numeric column)
        label_col = None
        for c in df.columns:
            if df[c].dtype == "object":
                label_col = c
                break

        if label_col:
            for _, row in df.head(15).iterrows():
                prev_val = row.get(col_prev)
                curr_val = row.get(col_curr)
                label = str(row.get(label_col, ""))[:40]
                if (
                    pd.notna(prev_val) and pd.notna(curr_val)
                    and isinstance(prev_val, (int, float))
                    and isinstance(curr_val, (int, float))
                    and prev_val != 0
                ):
                    growth = (curr_val - prev_val) / abs(prev_val) * 100
                    if abs(growth) > 5:  # Only report meaningful changes
                        direction = "↑" if growth > 0 else "↓"
                        trends.append(
                            f"{label}: {direction} {abs(growth):.1f}% "
                            f"({col_prev} → {col_curr})"
                        )
                if len(trends) >= 8:
                    break

    return trends


def _financial_pattern_hints(columns: List[str], df: pd.DataFrame) -> List[str]:
    """
    Detect financial patterns and provide hints for the LLM.
    Recognises common financial statement layouts, ratios, and metrics.
    """
    hints: List[str] = []
    col_set = {c.lower().replace(" ", "_") for c in columns}

    # Detect statement type
    if {"revenue", "sales", "turnover"} & col_set or any(
        "revenue" in c.lower() or "sales" in c.lower() for c in columns
    ):
        hints.append("Layout: likely Income Statement / P&L data")
    if {"asset", "liability", "equity"} & col_set or any(
        "balance" in c.lower() for c in columns
    ):
        hints.append("Layout: likely Balance Sheet data")
    if any("cash" in c.lower() and "flow" in c.lower() for c in columns):
        hints.append("Layout: likely Cash Flow Statement data")

    # Detect if data has Indian financial conventions
    for col in columns:
        cl = col.lower()
        if any(p in cl for p in ["crore", "lakh", "inr", "₹", "rupee"]):
            hints.append("Currency: Indian Rupees (likely in Crores/Lakhs)")
            break

    # Check for transposed layout (metrics as rows, periods as columns)
    if df.shape[1] > df.shape[0] and df.shape[0] > 3:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(obj_cols) <= 2 and len(num_cols) >= 3:
            hints.append(
                f"Structure: transposed layout (metrics as rows, "
                f"{len(num_cols)} period columns)"
            )

    return hints


def compress_dataframe_for_llm(
    df: pd.DataFrame,
    max_sample_rows: int = 5,
    max_columns: int = 50,
    include_stats: bool = True,
    include_eda: bool = True,
) -> str:
    """
    Create a token-efficient, intelligence-enriched structural representation
    of a DataFrame for LLM context.

    Combines structural compression (60-80% token reduction) with adaptive
    EDA intelligence so the LLM understands data *quality*, *shape*, and
    *relationships* — not just schema and samples.

    Layers (all read-only, zero data mutation):
      1. Column Schema       — names, types, null%, smart classification
      2. Data Quality Score   — per-column + aggregate quality grade
      3. Statistical Summary  — min/max/mean + distribution profile
      4. Outlier Flags        — IQR-based audit red-flags
      5. Correlation Insights — top correlated column pairs
      6. Temporal Trends      — YoY/QoQ growth detection
      7. Financial Patterns   — statement type, currency, layout hints
      8. Representative Sample

    Args:
        df: Source DataFrame (never mutated)
        max_sample_rows: Number of sample rows to include
        max_columns: Maximum columns to describe
        include_stats: Include statistical summaries (min/max/mean/distribution)
        include_eda: Include EDA intelligence (quality, outliers, correlations,
                     trends, financial patterns). Set False for minimal output.

    Returns:
        LLM-ready string representation with embedded data intelligence
    """
    if df is None or df.empty:
        return "DataFrame is empty (0 rows × 0 columns)"

    lines: List[str] = []
    n_rows = len(df)
    lines.append(f"DataFrame Shape: {n_rows} rows × {df.shape[1]} columns")
    lines.append("")

    columns = list(df.columns)[:max_columns]
    if len(df.columns) > max_columns:
        lines.append(f"⚠ Showing {max_columns} of {df.shape[1]} columns")
        lines.append("")

    # ── 1. Column Schema with Smart Classification ────────────────
    lines.append("═══ COLUMN SCHEMA ═══")

    classifications: Dict[str, str] = {}
    quality_scores: Dict[str, dict] = {}
    period_cols: List[str] = []
    date_cols: List[str] = []
    id_cols: List[str] = []

    for col in columns:
        series = df[col]
        dtype = series.dtype
        non_null = series.notna().sum()
        null_pct = round((1 - non_null / max(n_rows, 1)) * 100, 1)

        # Classify column
        col_class = _classify_column(col, series, n_rows)
        classifications[col] = col_class

        # Track pattern columns
        if col_class == "temporal":
            col_lower = col.lower()
            if any(p in col_lower for p in ["fy", "quarter", "q1", "q2", "q3", "q4", "period"]):
                period_cols.append(col)
            else:
                date_cols.append(col)
        elif col_class == "identifier":
            id_cols.append(col)

        # Quality score (when EDA enabled)
        if include_eda:
            quality_scores[col] = _column_quality_score(series)

        # Build column description line
        class_tag = f"[{col_class}]"
        q_tag = ""
        if include_eda and quality_scores.get(col, {}).get("quality", 100) < 80:
            q_tag = f" ⚠Q:{quality_scores[col]['quality']}"

        if pd.api.types.is_numeric_dtype(series) and include_stats:
            stats = series.describe()
            mean_val = stats.get("mean", 0)
            # Safe formatting — avoid crash on NaN/inf
            try:
                mean_str = f"{mean_val:.2f}"
            except (ValueError, TypeError):
                mean_str = str(mean_val)
            lines.append(
                f"• {col} ({dtype}) {class_tag}{q_tag} | "
                f"{non_null} values, {null_pct}% null | "
                f"min={stats.get('min', 'N/A')}, "
                f"max={stats.get('max', 'N/A')}, "
                f"mean={mean_str}"
            )
        elif series.dtype == "object":
            nunique = series.nunique()
            top_vals = series.dropna().value_counts().head(3).index.tolist()
            top_str = ", ".join(str(v)[:30] for v in top_vals)
            lines.append(
                f"• {col} ({dtype}) {class_tag}{q_tag} | "
                f"{non_null} values, {null_pct}% null | "
                f"{nunique} unique | Top: [{top_str}]"
            )
        else:
            lines.append(
                f"• {col} ({dtype}) {class_tag}{q_tag} | "
                f"{non_null} values, {null_pct}% null"
            )

    # ── 2. Aggregate Data Quality Score ──────────────────────────
    if include_eda and quality_scores:
        lines.append("")
        lines.append("═══ DATA QUALITY ASSESSMENT ═══")
        avg_quality = round(
            sum(q["quality"] for q in quality_scores.values()) / len(quality_scores), 1
        )
        avg_completeness = round(
            sum(q["completeness"] for q in quality_scores.values()) / len(quality_scores), 1
        )
        # Letter grade
        if avg_quality >= 90:
            grade = "A"
        elif avg_quality >= 75:
            grade = "B"
        elif avg_quality >= 60:
            grade = "C"
        else:
            grade = "D"
        lines.append(
            f"Overall Quality: {avg_quality}/100 (Grade {grade}) | "
            f"Completeness: {avg_completeness}%"
        )

        # Flag low-quality columns
        low_quality = [
            f"{col} ({q['quality']})"
            for col, q in quality_scores.items()
            if q["quality"] < 70
        ]
        if low_quality:
            lines.append(f"⚠ Low quality columns: {', '.join(low_quality[:8])}")

    # ── 3. Distribution Profiles (for key numeric columns) ───────
    if include_stats and include_eda:
        dist_lines = []
        numeric_cols = [
            c for c in columns
            if pd.api.types.is_numeric_dtype(df[c])
            and classifications.get(c) in ("metric", "financial_metric", "ratio")
        ]
        for col in numeric_cols[:8]:  # Cap at 8 to save tokens
            profile = _distribution_profile(df[col])
            if profile:
                dist_lines.append(
                    f"  {col}: {profile['shape']}, {profile['tail']} "
                    f"(skew={profile['skewness']}, kurt={profile['kurtosis']})"
                )
        if dist_lines:
            lines.append("")
            lines.append("═══ DISTRIBUTION PROFILES ═══")
            lines.extend(dist_lines)

    # ── 4. Outlier Detection ─────────────────────────────────────
    if include_eda:
        outlier_lines = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                outliers = _detect_outliers_iqr(df[col], col)
                if outliers:
                    sample_outliers = outliers[:3]
                    vals_str = ", ".join(
                        f"row {o['row']}={o['value']} ({o['dir']})"
                        for o in sample_outliers
                    )
                    outlier_lines.append(
                        f"  ⚠ {col}: {len(outliers)} outlier(s) — {vals_str}"
                    )
        if outlier_lines:
            lines.append("")
            lines.append("═══ OUTLIER FLAGS (IQR Method) ═══")
            lines.extend(outlier_lines[:10])  # Cap to keep concise

    # ── 5. Correlation Insights ──────────────────────────────────
    if include_eda and n_rows >= 5:
        corr_pairs = _top_correlations(df, columns, top_n=5)
        if corr_pairs:
            lines.append("")
            lines.append("═══ CORRELATION INSIGHTS ═══")
            for p in corr_pairs:
                direction = "positive" if p["r"] > 0 else "negative"
                lines.append(
                    f"  {p['col_a']} ↔ {p['col_b']}: "
                    f"r={p['r']} ({p['strength']} {direction})"
                )

    # ── 6. Temporal Growth Trends ────────────────────────────────
    if include_eda and period_cols:
        trends = _temporal_growth_trends(df, columns, period_cols)
        if trends:
            lines.append("")
            lines.append("═══ GROWTH TRENDS (Period-over-Period) ═══")
            for t in trends:
                lines.append(f"  {t}")

    # ── 7. Financial Pattern Hints ───────────────────────────────
    if include_eda:
        hints = _financial_pattern_hints(columns, df)
        if hints:
            lines.append("")
            lines.append("═══ FINANCIAL INTELLIGENCE ═══")
            for h in hints:
                lines.append(f"  {h}")

    # ── 8. Detected Patterns (enhanced) ──────────────────────────
    # Merge with pattern detection from classification
    metric_cols = [c for c, cl in classifications.items() if cl == "financial_metric"]
    ratio_cols = [c for c, cl in classifications.items() if cl == "ratio"]

    has_patterns = period_cols or date_cols or id_cols or metric_cols or ratio_cols
    if has_patterns:
        lines.append("")
        lines.append("═══ DETECTED PATTERNS ═══")
        if period_cols:
            lines.append(f"  Period columns: {period_cols}")
        if date_cols:
            lines.append(f"  Date columns: {date_cols}")
        if id_cols:
            lines.append(f"  ID columns: {id_cols}")
        if metric_cols:
            lines.append(f"  Financial metrics: {metric_cols[:10]}")
        if ratio_cols:
            lines.append(f"  Ratios/percentages: {ratio_cols[:10]}")

    # ── 9. Sample Data ───────────────────────────────────────────
    lines.append("")
    actual_sample = min(max_sample_rows, n_rows)
    lines.append(f"═══ SAMPLE DATA (first {actual_sample} rows) ═══")
    sample_df = df[columns].head(max_sample_rows)
    lines.append(sample_df.to_string(max_colwidth=30, index=True))

    return "\n".join(lines)


__all__ = ["ExcelIngestor", "compress_dataframe_for_llm"]
