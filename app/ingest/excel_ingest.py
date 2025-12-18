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
        (r"^fy\d{2}$", "fiscal_year"),        # FY21, FY22
        (r"^\d{1,2}mfy\d{2}$", "period_fy"),   # 9MFY22, 3MFY21
        (r"^q\d[_-]?fy\d{2}$", "quarter"),     # Q1FY22, Q2-FY23
        (r"^q\d[_-]?\d{2}$", "quarter"),       # Q121, Q2-22
        (r"^h\d[_-]?fy\d{2}$", "half_year"),   # H1FY22
        (r"^\d{4}$", "year"),                  # 2021, 2022
        (r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[_-]?\d{2,4}$", "month_year"),
    ]

    # NA-like values to replace
    NA_VALUES = ["Na/p", "N/A", "n/a", "NA", "-", "--", "None", "none", "NULL", "null", "N.A.", "n.a."]

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
        unnamed_ratio = sum(1 for c in df.columns if str(c).startswith("Unnamed")) / max(len(df.columns), 1)
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

    def _preprocess_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
                report["columns_normalized"].append({"original": str(col), "normalized": new_name})

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
                report["na_replaced"] += (new_na_count - original_na_count)
            
            if df[col].dtype in [np.float64, np.int64] and col not in [r["column"] for r in report.get("period_columns", [])]:
                report["numeric_coerced"].append(col)

        report["final_shape"] = df.shape
        report["sheet_name"] = sheet_name

        return df, report

    def read_sheets_info(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
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
                sheets_info.append({
                    "name": "csv_data",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "score": score,
                    "column_names": list(df.columns)[:10],
                })
            else:
                xl = pd.ExcelFile(io.BytesIO(file_content))
                for sheet_name in xl.sheet_names:
                    try:
                        df = xl.parse(sheet_name, nrows=500)  # Read sample for scoring
                        score = self._score_sheet(df)
                        sheets_info.append({
                            "name": sheet_name,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "score": score,
                            "column_names": list(df.columns)[:10],
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read sheet {sheet_name}: {e}")
                        sheets_info.append({
                            "name": sheet_name,
                            "rows": 0,
                            "columns": 0,
                            "score": 0.0,
                            "error": str(e),
                        })
        except Exception as e:
            logger.error(f"Failed to read file {filename}: {e}")
            return []

        # Sort by score
        sheets_info.sort(key=lambda x: x.get("score", 0), reverse=True)
        self._last_sheets_info = sheets_info
        return sheets_info

    def ingest_best_sheet(
        self,
        file_content: bytes,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None
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

    def ingest_all_sheets(
        self,
        file_content: bytes,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None
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
                results.append({
                    "sheet_name": sheet_name,
                    "success": False,
                    "error": sheet_info["error"]
                })
                continue

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(file_content))
                else:
                    df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)

                if df.empty or df.shape[0] < 2:
                    results.append({
                        "sheet_name": sheet_name,
                        "success": False,
                        "error": "Empty or minimal data"
                    })
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

                results.append({
                    "success": True,
                    "dataset_id": dataset_id,
                    "sheet_name": sheet_name,
                    "rows": len(df),
                    "columns": list(df.columns),
                })

            except Exception as e:
                logger.error(f"Failed to ingest {sheet_name}: {e}")
                results.append({
                    "sheet_name": sheet_name,
                    "success": False,
                    "error": str(e)
                })

        return results

    def get_last_sheets_info(self) -> List[Dict[str, Any]]:
        """Get info from last file read."""
        return self._last_sheets_info


__all__ = ["ExcelIngestor"]
