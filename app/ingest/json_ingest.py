"""
JSON Ingest Engine - Robust JSON ingestion with:
- Auto-detection of data structure (list of records, nested tables, single record)
- Column normalization and type coercion
- Support for nested JSON with multiple tables
- Direct text JSON parsing (for pasted content)
"""
import logging
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class JSONIngestor:
    """
    Production-grade JSON ingestion engine.
    Handles various JSON structures and converts them to DataFrames.
    """
    
    def __init__(self):
        self._last_ingest_info: Dict[str, Any] = {}
    
    def _detect_structure(self, data: Any) -> str:
        """
        Detect the structure of JSON data.
        
        Returns:
            Structure type: 'records', 'nested_tables', 'columnar', 'single_record', 'unknown'
        """
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return "records"  # List of record dicts
            return "unknown"
        
        if isinstance(data, dict):
            values = list(data.values())
            if not values:
                return "unknown"
            
            # Check if all values are lists (columnar format or nested tables)
            if all(isinstance(v, list) for v in values):
                # Check if nested tables (list of dicts) or columnar (list of primitives)
                if values and len(values[0]) > 0 and isinstance(values[0][0], dict):
                    return "nested_tables"
                return "columnar"
            
            # Check if some values are lists (mixed - treat as nested tables)
            if any(isinstance(v, list) for v in values):
                return "nested_tables"
            
            # All primitive values - single record
            return "single_record"
        
        return "unknown"
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to consistent format."""
        new_columns = []
        seen = set()
        
        for i, col in enumerate(df.columns):
            # Convert to string and clean
            col_str = str(col).strip()
            col_str = col_str.replace(' ', '_').replace('-', '_')
            col_str = ''.join(c if c.isalnum() or c == '_' else '' for c in col_str)
            
            if not col_str or col_str[0].isdigit():
                col_str = f"col_{i}"
            
            # Handle duplicates
            base = col_str
            counter = 1
            while col_str.lower() in seen:
                col_str = f"{base}_{counter}"
                counter += 1
            seen.add(col_str.lower())
            new_columns.append(col_str)
        
        df.columns = new_columns
        return df
    
    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to coerce columns to appropriate types."""
        for col in df.columns:
            # Try numeric conversion
            if df[col].dtype == object:
                try:
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if numeric.notna().sum() > len(df) * 0.5:
                        df[col] = numeric
                except:
                    pass
        return df
    
    def parse_json_text(self, json_text: str) -> Tuple[Any, Optional[str]]:
        """
        Parse JSON text and return data.
        
        Returns:
            Tuple of (parsed_data, error_message)
        """
        try:
            data = json.loads(json_text)
            return data, None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"
    
    def ingest_json(
        self,
        data: Any,
        source_name: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Ingest JSON data into DataFrames.
        
        Args:
            data: Parsed JSON data (dict or list)
            source_name: Name for the source (used in dataset_id)
            client_id: Client identifier
            register_callback: Optional callback to register each DataFrame
            
        Returns:
            Dict with ingestion results
        """
        results = {
            "success": True,
            "datasets": [],
            "structure": "unknown",
            "error": None
        }
        
        try:
            structure = self._detect_structure(data)
            results["structure"] = structure
            
            safe_name = source_name.lower().replace(' ', '_').replace('-', '_')
            
            if structure == "records":
                # List of record dicts -> single DataFrame
                df = pd.DataFrame(data)
                df = self._normalize_column_names(df)
                df = self._coerce_types(df)
                
                dataset_id = f"{client_id}:{safe_name}"
                
                if register_callback:
                    register_callback(dataset_id, df, {"structure": structure})
                
                results["datasets"].append({
                    "dataset_id": dataset_id,
                    "rows": len(df),
                    "columns": len(df.columns)
                })
                
            elif structure == "columnar":
                # Dict where values are lists (columns)
                df = pd.DataFrame(data)
                df = self._normalize_column_names(df)
                df = self._coerce_types(df)
                
                dataset_id = f"{client_id}:{safe_name}"
                
                if register_callback:
                    register_callback(dataset_id, df, {"structure": structure})
                
                results["datasets"].append({
                    "dataset_id": dataset_id,
                    "rows": len(df),
                    "columns": len(df.columns)
                })
                
            elif structure == "nested_tables":
                # Dict with nested arrays/objects -> multiple DataFrames
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict):
                            df = pd.DataFrame(value)
                        else:
                            df = pd.DataFrame({key: value})
                        
                        df = self._normalize_column_names(df)
                        df = self._coerce_types(df)
                        
                        safe_key = key.lower().replace(' ', '_').replace('-', '_')
                        dataset_id = f"{client_id}:{safe_name}:{safe_key}"
                        
                        if register_callback:
                            register_callback(dataset_id, df, {"structure": structure, "table": key})
                        
                        results["datasets"].append({
                            "dataset_id": dataset_id,
                            "table_name": key,
                            "rows": len(df),
                            "columns": len(df.columns)
                        })
                
            elif structure == "single_record":
                # Single dict -> DataFrame with one row
                df = pd.DataFrame([data])
                df = self._normalize_column_names(df)
                df = self._coerce_types(df)
                
                dataset_id = f"{client_id}:{safe_name}"
                
                if register_callback:
                    register_callback(dataset_id, df, {"structure": structure})
                
                results["datasets"].append({
                    "dataset_id": dataset_id,
                    "rows": 1,
                    "columns": len(df.columns)
                })
                
            else:
                results["success"] = False
                results["error"] = "Unknown or unsupported JSON structure"
                
            self._last_ingest_info = results
            
        except Exception as e:
            logger.error(f"JSON ingestion failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def ingest_json_file(
        self,
        file_content: bytes,
        filename: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Ingest JSON from file content.
        
        Args:
            file_content: Raw bytes of JSON file
            filename: Original filename
            client_id: Client identifier
            register_callback: Optional callback to register each DataFrame
            
        Returns:
            Dict with ingestion results
        """
        # Decode bytes to string
        try:
            json_text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                json_text = file_content.decode('latin-1')
            except:
                return {"success": False, "error": "Could not decode file content"}
        
        # Parse JSON
        data, error = self.parse_json_text(json_text)
        if error:
            return {"success": False, "error": error}
        
        # Extract source name from filename
        source_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        return self.ingest_json(data, source_name, client_id, register_callback)
    
    def ingest_json_text(
        self,
        json_text: str,
        source_name: str,
        client_id: str,
        register_callback: Optional[Callable[[str, pd.DataFrame, Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Ingest JSON from raw text (e.g., pasted content).
        
        Args:
            json_text: Raw JSON text
            source_name: Name for the source
            client_id: Client identifier
            register_callback: Optional callback to register each DataFrame
            
        Returns:
            Dict with ingestion results
        """
        # Parse JSON
        data, error = self.parse_json_text(json_text)
        if error:
            return {"success": False, "error": error}
        
        return self.ingest_json(data, source_name, client_id, register_callback)
    
    def get_last_ingest_info(self) -> Dict[str, Any]:
        """Get info from last ingestion."""
        return self._last_ingest_info


__all__ = ["JSONIngestor"]
