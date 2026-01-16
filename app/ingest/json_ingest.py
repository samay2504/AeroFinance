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
    
    Features:
    - Deep flattening for nested objects
    - Recursive table extraction
    - Smart key path handling for deeply nested data
    - Corporate financial data optimized
    """
    
    def __init__(self, max_depth: int = 10):
        self._last_ingest_info: Dict[str, Any] = {}
        self._max_depth = max_depth
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Recursively flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Prefix for keys
            sep: Separator between key levels
            
        Returns:
            Flattened dictionary with composite keys
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                # Recursively flatten nested dicts
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # Handle lists specially - keep reference for table extraction
                if len(v) > 0 and isinstance(v[0], dict):
                    # List of dicts -> will be extracted as separate table
                    items.append((new_key, f"[{len(v)} records]"))
                elif len(v) > 0:
                    # Simple list -> join as string
                    items.append((new_key, ", ".join(str(x) for x in v[:10])))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _extract_tables_deep(
        self, 
        data: Any, 
        path: str = '', 
        depth: int = 0
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        Recursively extract all table-like structures from nested JSON.
        
        Returns:
            List of (table_name, DataFrame) tuples
        """
        tables = []
        
        if depth > self._max_depth:
            return tables
        
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # List of dicts -> convert to DataFrame
                try:
                    # First flatten each record
                    flattened_records = []
                    for record in data:
                        if isinstance(record, dict):
                            flattened_records.append(self._flatten_dict(record))
                        else:
                            flattened_records.append({'value': record})
                    
                    df = pd.DataFrame(flattened_records)
                    if len(df) > 0:
                        table_name = path or 'root'
                        tables.append((table_name, df))
                except Exception as e:
                    logger.debug(f"Could not create table from {path}: {e}")
                    
        elif isinstance(data, dict):
            # For dicts, check if it has any list values (potential tables)
            has_primitive = any(not isinstance(v, (dict, list)) for v in data.values())
            has_nested = any(isinstance(v, (dict, list)) for v in data.values())
            
            # If mix of primitives and nested, create a root table with flattened primitives
            if has_primitive:
                flattened = self._flatten_dict(data)
                # Filter out the "[X records]" placeholders
                flattened = {k: v for k, v in flattened.items() 
                            if not (isinstance(v, str) and v.startswith('[') and 'records]' in v)}
                if flattened:
                    df = pd.DataFrame([flattened])
                    table_name = path or 'overview'
                    tables.append((table_name, df))
            
            # Recurse into nested structures
            for key, value in data.items():
                child_path = f"{path}_{key}" if path else key
                child_tables = self._extract_tables_deep(value, child_path, depth + 1)
                tables.extend(child_tables)
        
        return tables
    
    def _detect_structure(self, data: Any) -> str:
        """
        Detect the structure of JSON data.
        
        Returns:
            Structure type: 'records', 'nested_tables', 'columnar', 'single_record', 'complex_nested', 'unknown'
        """
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return "records"  # List of record dicts
            return "unknown"
        
        if isinstance(data, dict):
            values = list(data.values())
            if not values:
                return "unknown"
            
            # Check for deeply nested structure (corporate data pattern)
            has_nested_objects = any(isinstance(v, dict) and len(v) > 0 for v in values)
            has_nested_arrays = any(isinstance(v, list) and len(v) > 0 and isinstance(v[0] if v else None, dict) for v in values)
            
            if has_nested_objects or has_nested_arrays:
                # Check depth - if deeply nested, use complex handler
                def check_depth(d, current_depth=0):
                    if current_depth > 3:
                        return True
                    if isinstance(d, dict):
                        for v in d.values():
                            if check_depth(v, current_depth + 1):
                                return True
                    elif isinstance(d, list) and d and isinstance(d[0], dict):
                        return check_depth(d[0], current_depth + 1)
                    return False
                
                if check_depth(data):
                    return "complex_nested"
            
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
            
            elif structure == "complex_nested":
                # Complex nested structure (like corporate financial data)
                # Use deep extraction to get all tables
                tables = self._extract_tables_deep(data)
                
                for table_name, df in tables:
                    df = self._normalize_column_names(df)
                    df = self._coerce_types(df)
                    
                    # Clean table name for dataset_id
                    safe_table = table_name.lower().replace(' ', '_').replace('-', '_')
                    safe_table = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe_table)
                    dataset_id = f"{client_id}:{safe_name}:{safe_table}"
                    
                    if register_callback:
                        register_callback(dataset_id, df, {"structure": structure, "table": table_name})
                    
                    results["datasets"].append({
                        "dataset_id": dataset_id,
                        "table_name": table_name,
                        "rows": len(df),
                        "columns": len(df.columns)
                    })
                
                # If no tables extracted, fallback to flattened single record
                if not results["datasets"]:
                    flattened = self._flatten_dict(data)
                    df = pd.DataFrame([flattened])
                    df = self._normalize_column_names(df)
                    df = self._coerce_types(df)
                    
                    dataset_id = f"{client_id}:{safe_name}:overview"
                    
                    if register_callback:
                        register_callback(dataset_id, df, {"structure": "flattened"})
                    
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
    
    def generate_text_representation(self, df: pd.DataFrame, table_name: str = "") -> str:
        """
        Generate a rich text representation of a DataFrame for RAG embedding.
        Creates natural language descriptions of the data.
        """
        lines = []
        
        # Table header
        if table_name:
            lines.append(f"Table: {table_name}")
        
        # Column summary
        columns = list(df.columns)
        lines.append(f"Columns: {', '.join(str(c) for c in columns)}")
        lines.append(f"Rows: {len(df)}")
        
        # Generate natural language for each row
        for idx, row in df.iterrows():
            row_parts = []
            for col in columns:
                val = row[col]
                if pd.notna(val):
                    row_parts.append(f"{col}: {val}")
            if row_parts:
                lines.append(f"Row {idx + 1}: {'; '.join(row_parts)}")
        
        # Generate summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            lines.append("\nSummary Statistics:")
            for col in numeric_cols:
                total = df[col].sum()
                mean = df[col].mean()
                lines.append(f"  {col}: Total={total:,.2f}, Average={mean:,.2f}")
        
        return "\n".join(lines)
    
    def ingest_to_rag(
        self,
        data: Any,
        source_name: str,
        client_id: str,
        rag_pipeline = None
    ) -> Dict[str, Any]:
        """
        Ingest JSON data into both DataFrames AND RAG vector storage.
        
        Args:
            data: Parsed JSON data
            source_name: Name for the source
            client_id: Client identifier
            rag_pipeline: RAG pipeline instance for vector storage
        
        Returns:
            Dict with ingestion results including RAG status
        """
        results = {
            "success": True,
            "datasets": [],
            "rag_chunks": 0,
            "error": None
        }
        
        try:
            structure = self._detect_structure(data)
            safe_name = source_name.lower().replace(' ', '_').replace('-', '_')
            
            dataframes = []  # Collect (name, df) tuples
            
            if structure == "records":
                df = pd.DataFrame(data)
                df = self._normalize_column_names(df)
                df = self._coerce_types(df)
                dataframes.append((safe_name, df))
                
            elif structure == "columnar":
                df = pd.DataFrame(data)
                df = self._normalize_column_names(df)
                df = self._coerce_types(df)
                dataframes.append((safe_name, df))
                
            elif structure == "nested_tables":
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict):
                            df = pd.DataFrame(value)
                        else:
                            df = pd.DataFrame({key: value})
                        df = self._normalize_column_names(df)
                        df = self._coerce_types(df)
                        safe_key = key.lower().replace(' ', '_').replace('-', '_')
                        dataframes.append((f"{safe_name}:{safe_key}", df))
                        
            elif structure == "single_record":
                df = pd.DataFrame([data])
                df = self._normalize_column_names(df)
                df = self._coerce_types(df)
                dataframes.append((safe_name, df))
            
            # Process each DataFrame
            for table_name, df in dataframes:
                dataset_id = f"{client_id}:{table_name}"
                
                results["datasets"].append({
                    "dataset_id": dataset_id,
                    "rows": len(df),
                    "columns": len(df.columns)
                })
                
                # Ingest into RAG if available
                if rag_pipeline and rag_pipeline.is_available:
                    text_repr = self.generate_text_representation(df, table_name)
                    
                    # Create metadata for RAG (use primitive types only)
                    metadata = {
                        "source_type": "json",
                        "table_name": table_name,
                        "structure": structure,
                        "rows": len(df),
                        "columns": ", ".join(str(c) for c in df.columns)  # String, not list
                    }
                    
                    try:
                        rag_result = rag_pipeline.ingest_document(
                            text=text_repr,
                            client_id=client_id,
                            doc_id=dataset_id,
                            metadata=metadata
                        )
                        if rag_result.get("success"):
                            results["rag_chunks"] += rag_result.get("chunks_created", 1)
                    except Exception as e:
                        logger.warning(f"RAG ingestion failed for {dataset_id}: {e}")
            
            self._last_ingest_info = results
            
        except Exception as e:
            logger.error(f"JSON RAG ingestion failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results


__all__ = ["JSONIngestor"]
