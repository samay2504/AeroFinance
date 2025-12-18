"""
Data Analyst Agent - SQL-first analytical agent with LLM-driven semantic understanding.
Handles numeric queries, growth calculations, and period-based analysis dynamically.
Uses LLM to understand data structure rather than hardcoded patterns.
"""
import logging
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of data analysis."""
    success: bool
    result: Any = None
    value: Optional[float] = None
    method: str = "unknown"
    explanation: str = ""
    error: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAnalystAgent:
    """
    LLM-driven data analyst with:
    - Semantic schema understanding (no hardcoded patterns)
    - DuckDB SQL execution with intelligent prompts
    - LLM-generated SQL with data context
    - Sandboxed Python code execution
    - Iterative refinement on errors
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._sql_engine = None
        self._data_registry = None
        self._template_engine = None
        self._sandbox = None
        self._schema_analyzer = None
        self._schema_cache: Dict[str, Any] = {}  # Cache analyzed schemas
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self._init_components()

    def _init_components(self):
        """Initialize component dependencies."""
        try:
            from app.sql_engine import get_sql_engine
            self._sql_engine = get_sql_engine()
        except Exception as e:
            logger.warning(f"SQL engine unavailable: {e}")

        try:
            from app.core.data_registry import get_data_registry
            self._data_registry = get_data_registry()
        except Exception as e:
            logger.warning(f"Data registry unavailable: {e}")

        try:
            from app.sql_templates import get_template_engine
            self._template_engine = get_template_engine()
        except Exception as e:
            logger.warning(f"Template engine unavailable: {e}")

        try:
            from app.sandbox.sandbox_executor import SandboxExecutor
            self._sandbox = SandboxExecutor()
        except Exception as e:
            logger.warning(f"Sandbox unavailable: {e}")

        try:
            from app.core.schema_analyzer import SchemaAnalyzer
            self._schema_analyzer = SchemaAnalyzer(self._llm)
        except Exception as e:
            logger.warning(f"Schema analyzer unavailable: {e}")

    def register_dataframe(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        preprocessing_report: Optional[Dict] = None,
        client_id: Optional[str] = None
    ) -> bool:
        """Register DataFrame for analysis."""
        if df is None or df.empty:
            return False

        # Store locally
        self.dataframes[dataset_id] = df

        # Register with SQL engine
        if self._sql_engine:
            self._sql_engine.register_dataframe(dataset_id, df)

        # Register with data registry
        if self._data_registry:
            metadata = {"preprocessing": preprocessing_report} if preprocessing_report else {}
            self._data_registry.register(dataset_id, df, metadata, client_id)

        logger.info(f"Registered dataset: {dataset_id} ({len(df)} rows)")
        return True

    def list_datasets_for_client(self, client_id: str) -> List[Dict[str, Any]]:
        """List datasets for a client."""
        if self._data_registry:
            return self._data_registry.list_for_client(client_id)
        
        # Fallback to local dataframes
        return [
            {"dataset_id": k, "rows": len(v), "columns": list(v.columns)}
            for k, v in self.dataframes.items()
            if client_id in k
        ]

    def match_dataset_by_query(
        self,
        query: str,
        datasets: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Match query to best dataset based on content keywords."""
        if not datasets:
            return None

        query_lower = query.lower()
        
        # Keyword mapping to sheet types
        keyword_map = {
            "income statement": ["income", "statement", "p_l", "pnl"],
            "balance sheet": ["balance", "sheet", "assets"],
            "cashflow": ["cashflow", "cash_flow", "cash"],
            "cashburn": ["cashburn", "cash_burn", "burn"],
            "unit economics": ["unit", "economics", "per_order", "per order", "cost_per"],
            "cost per order": ["unit", "economics", "per_order"],  # Maps to unit economics
            "digital marketing": ["unit", "economics", "marketing"],  # Maps to unit economics
            "marketing cost": ["unit", "economics", "marketing"],
            "gmv": ["gmv", "merchandise", "value", "cashburn"],
            "december": ["cashburn", "monthly", "cash_burn"],  # Monthly data
            "revenue": ["income", "revenue", "statement"],
            "collection": ["collection", "prepaid", "recorded", "fy22"],
            "variance": ["variance", "actual", "budget", "forecasting", "comp"],
            "actual": ["comp", "variance", "model"],
            "model": ["comp", "variance", "project"],
        }

        best_match = None
        best_score = 0

        for dataset in datasets:
            dataset_id = dataset.get("dataset_id", "")
            columns = dataset.get("columns", [])
            
            score = 0
            dataset_lower = dataset_id.lower()
            columns_lower = " ".join(str(c).lower() for c in columns)

            # Check keywords
            for keyword, patterns in keyword_map.items():
                if keyword in query_lower:
                    for pattern in patterns:
                        if pattern in dataset_lower or pattern in columns_lower:
                            score += 10

            # Direct column match
            for word in query_lower.split():
                if len(word) > 3:
                    if word in columns_lower:
                        score += 5
                    if word in dataset_lower:
                        score += 3

            # Period match
            periods = re.findall(r"fy\d{2}|9mfy\d{2}|q\d", query_lower)
            for period in periods:
                if period in columns_lower:
                    score += 8

            if score > best_score:
                best_score = score
                best_match = dataset_id

        # If no good match, return first dataset
        if best_match is None and datasets:
            best_match = datasets[0].get("dataset_id")

        return best_match

    def _get_dataframe(self, dataset_id: str, client_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get DataFrame by ID."""
        if dataset_id in self.dataframes:
            return self.dataframes[dataset_id]
        
        if self._data_registry:
            return self._data_registry.get(dataset_id, client_id)
        
        return None

    def _get_schema_info(self, df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        """Get schema information for a DataFrame."""
        return {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "sample": df.head(3).to_dict() if len(df) > 0 else {}
        }

    def execute_sql_query(
        self,
        query: str,
        df_id: str,
        user_id: str = "default",
        client_id: Optional[str] = None,
        use_cache: bool = True
    ) -> AnalysisResult:
        """
        Execute analytical query using SQL-first approach.
        
        Pipeline:
        1. Try deterministic SQL templates
        2. Try LLM SQL generation
        3. Fall back to LLM SQL/Python
        """
        df = self._get_dataframe(df_id, client_id)
        if df is None:
            return AnalysisResult(
                success=False,
                error=f"Dataset not found: {df_id}",
                method="error"
            )

        schema = self._get_schema_info(df, df_id)

        # Check for metadata queries first
        meta_result = self._handle_metadata_query(query, df, df_id, client_id)
        if meta_result:
            return meta_result

        # Step 1: Try deterministic template
        template_result = self._try_template_sql(query, df, df_id, schema)
        if template_result and template_result.success:
            return template_result

        # Step 2: Try semantic Pandas (fast, deterministic, schema-based)
        # This is more reliable than LLM for structured patterns
        heuristic_result = self._try_heuristic_pandas(query, df, df_id)
        if heuristic_result and heuristic_result.success:
            return heuristic_result

        # Step 3: Try LLM SQL generation
        if self._llm:
            llm_result = self._try_llm_sql(query, df, df_id, schema)
            if llm_result and llm_result.success:
                # Filter out "Data not found" type results
                result_str = str(llm_result.result).lower()
                if "not found" not in result_str and "no data" not in result_str:
                    return llm_result

        # Step 4: Try LLM Python code generation
        if self._llm and self._sandbox:
            python_result = self._try_llm_python(query, df, df_id, schema)
            if python_result and python_result.success:
                # Filter out "Data not found" type results
                result_str = str(python_result.result).lower()
                if "not found" not in result_str and "no data" not in result_str:
                    return python_result

        return AnalysisResult(
            success=False,
            error="Could not process query with any available method",
            method="exhausted",
            explanation="Tried: template SQL, semantic Pandas, LLM SQL, LLM Python"
        )

    def _handle_metadata_query(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        client_id: Optional[str]
    ) -> Optional[AnalysisResult]:
        """Handle metadata queries about sheets/structure."""
        query_lower = query.lower()

        # Sheet count query
        if "how many sheets" in query_lower or "number of sheets" in query_lower:
            datasets = self.list_datasets_for_client(client_id) if client_id else list(self.dataframes.keys())
            count = len(datasets) if isinstance(datasets, list) else len(datasets)
            
            return AnalysisResult(
                success=True,
                result=f"There are {count} sheets available.",
                value=float(count),
                method="metadata",
                explanation="Counted registered datasets"
            )

        # List sheets query
        if "list" in query_lower and "sheet" in query_lower:
            datasets = self.list_datasets_for_client(client_id) if client_id else []
            if isinstance(datasets, list):
                sheet_names = [d.get("dataset_id", d) if isinstance(d, dict) else str(d) for d in datasets]
            else:
                sheet_names = list(self.dataframes.keys())
            
            return AnalysisResult(
                success=True,
                result=f"Available sheets: {', '.join(sheet_names)}",
                method="metadata",
                explanation="Listed all registered sheets"
            )

        return None

    def _try_template_sql(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        schema: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Try deterministic SQL template matching."""
        if not self._template_engine or not self._sql_engine:
            return None

        try:
            result = self._template_engine.generate_deterministic_sql(query, schema, df_id)
            if not result:
                return None

            sql, method = result
            logger.info(f"Template SQL: {sql[:100]}")

            # Ensure DataFrame is registered
            self._sql_engine.register_dataframe(df_id, df)

            # Execute SQL
            result_df = self._sql_engine.execute_df(sql)
            
            if result_df.empty:
                return None

            # Extract scalar value if single result
            if result_df.shape == (1, 1):
                value = float(result_df.iloc[0, 0])
                return AnalysisResult(
                    success=True,
                    result=value,
                    value=value,
                    method=f"sql_duckdb:{method}",
                    explanation=f"Executed SQL: {sql[:100]}..."
                )
            else:
                return AnalysisResult(
                    success=True,
                    result=result_df.to_dict(),
                    method=f"sql_duckdb:{method}",
                    explanation=f"Returned {len(result_df)} rows"
                )

        except Exception as e:
            logger.warning(f"Template SQL failed: {e}")
            return None

    def _try_llm_sql(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        schema: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """
        Try LLM-generated SQL with semantic understanding.
        Uses schema analyzer to understand data structure and provides
        actual data samples to the LLM for context.
        """
        if not self._llm or not self._sql_engine:
            return None

        try:
            from app.core.prompts import get_data_analyst_sql_prompt

            # Get sanitized table name for SQL
            safe_table_name = self._sql_engine.get_safe_table_name(df_id)
            
            # Analyze schema semantically using LLM
            semantic_info = ""
            if self._schema_analyzer:
                data_schema = self._schema_analyzer.analyze(df, df_id, context=df_id)
                
                # Build semantic info string for prompt
                semantic_parts = []
                if data_schema.label_column:
                    semantic_parts.append(f"- Label/Metric column: {data_schema.label_column}")
                if data_schema.period_columns:
                    period_str = ", ".join([f"{p} -> {c}" for p, c in data_schema.period_columns.items()])
                    semantic_parts.append(f"- Period columns: {period_str}")
                if data_schema.summary:
                    semantic_parts.append(f"- Data description: {data_schema.summary}")
                if data_schema.data_start_row > 0:
                    semantic_parts.append(f"- Data starts at row {data_schema.data_start_row}")
                semantic_info = "\n".join(semantic_parts)
            
            # Build schema string with sanitized table name
            schema_str = f"Table: {safe_table_name}\nShape: {df.shape[0]} rows x {df.shape[1]} columns\nColumns:\n"
            for col in schema["columns"]:
                dtype = schema["dtypes"].get(col, "unknown")
                # Include sample values for each column
                sample_vals = df[col].dropna().head(3).tolist()
                sample_str = str(sample_vals)[:50] if sample_vals else "empty"
                schema_str += f"  - {col}: {dtype} (e.g. {sample_str})\n"

            # Build data sample - show first few rows as context
            sample_rows = min(5, len(df))
            data_sample = df.head(sample_rows).to_string(max_colwidth=30)

            # Build columns list
            available_columns = ", ".join(schema["columns"])

            prompt = get_data_analyst_sql_prompt(
                schema_info=schema_str,
                query=query,
                available_columns=available_columns,
                data_sample=data_sample,
                semantic_info=semantic_info
            )
            
            # Try SQL generation with up to 2 refinement attempts
            max_attempts = 2
            last_error = None
            
            for attempt in range(max_attempts):
                response = self._llm.invoke_with_structured_output(
                    prompt,
                    output_schema={"sql": str, "columns_used": list, "explanation": str}
                )

                if "error" in response or "sql" not in response:
                    last_error = response.get("error", "No SQL in response")
                    continue

                sql = response["sql"]
                
                # Validate SQL
                is_valid, error = self._sql_engine.validate_sql(sql)
                if not is_valid:
                    logger.warning(f"LLM SQL attempt {attempt+1} invalid: {error}")
                    # Add error context to prompt for next attempt
                    prompt += f"\n\nPREVIOUS SQL FAILED: {sql}\nERROR: {error}\nPlease fix the SQL."
                    last_error = error
                    continue

                # Ensure DataFrame is registered
                self._sql_engine.register_dataframe(df_id, df)

                # Try to execute
                try:
                    result_df = self._sql_engine.execute_df(sql)
                    
                    if result_df.empty:
                        logger.warning(f"LLM SQL returned empty result")
                        prompt += f"\n\nPREVIOUS SQL RETURNED EMPTY: {sql}\nPlease revise to return data."
                        continue

                    if result_df.shape == (1, 1):
                        val = result_df.iloc[0, 0]
                        # Handle potential non-numeric result
                        try:
                            value = float(val) if pd.notna(val) else None
                        except (ValueError, TypeError):
                            value = None
                            
                        return AnalysisResult(
                            success=True,
                            result=val if value is None else value,
                            value=value,
                            method="sql_duckdb:llm_semantic",
                            explanation=response.get("explanation", f"SQL: {sql[:80]}")
                        )
                    else:
                        return AnalysisResult(
                            success=True,
                            result=result_df.to_dict(),
                            method="sql_duckdb:llm_semantic",
                            explanation=response.get("explanation", f"Returned {len(result_df)} rows")
                        )
                        
                except Exception as exec_error:
                    logger.warning(f"LLM SQL execution failed: {exec_error}")
                    # Add execution error to prompt for refinement
                    prompt += f"\n\nSQL EXECUTION FAILED: {sql}\nERROR: {str(exec_error)}\nPlease fix the SQL."
                    last_error = str(exec_error)

            if last_error:
                logger.warning(f"LLM SQL failed after {max_attempts} attempts: {last_error}")

        except Exception as e:
            logger.warning(f"LLM SQL failed: {e}")
        
        return None

    def _try_llm_python(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str,
        schema: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Try LLM-generated Python code in sandbox."""
        if not self._llm or not self._sandbox:
            return None

        try:
            from app.core.prompts import get_data_analyst_python_prompt

            schema_str = f"DataFrame: {df_id}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            sample = df.head(3).to_string() if len(df) > 0 else ""
            
            prompt = get_data_analyst_python_prompt(schema_str, query, sample)
            
            response = self._llm.invoke_with_structured_output(
                prompt,
                output_schema={"code": str, "explanation": str}
            )

            if "error" in response or "code" not in response:
                return None

            code = response["code"]
            
            # Execute in sandbox
            result = self._sandbox.execute(code, df)
            
            if result.get("success"):
                value = result.get("result")
                if isinstance(value, (int, float)):
                    return AnalysisResult(
                        success=True,
                        result=value,
                        value=float(value),
                        method="pandas:llm_sandbox",
                        explanation=response.get("explanation", "Python code executed")
                    )
                else:
                    return AnalysisResult(
                        success=True,
                        result=value,
                        method="pandas:llm_sandbox",
                        explanation=response.get("explanation", "Python code executed")
                    )

        except Exception as e:
            logger.warning(f"LLM Python failed: {e}")
        
        return None

    def _try_heuristic_pandas(
        self,
        query: str,
        df: pd.DataFrame,
        df_id: str = ""
    ) -> Optional[AnalysisResult]:
        """
        Semantic Pandas operations using schema analyzer.
        Uses LLM-discovered data structure rather than hardcoded patterns.
        """
        query_lower = query.lower()

        try:
            # Use schema analyzer for semantic understanding
            schema = None
            if self._schema_analyzer and df_id:
                schema = self._schema_analyzer.analyze(df, df_id, context=df_id)
            
            # If no schema analyzer, build basic period map from data
            period_col_map = {}
            label_col = df.columns[0] if len(df.columns) > 0 else None
            
            if schema:
                period_col_map = schema.period_columns
                label_col = schema.label_column or label_col
            else:
                # Fallback: scan for periods
                for row_idx in range(min(5, len(df))):
                    for col_idx, val in enumerate(df.iloc[row_idx]):
                        val_str = str(val).lower().strip()
                        if re.match(r"^(fy\d{2}|9mfy\d{2}|\d+mfy\d{2}|q\d\s*fy\d{2})$", val_str):
                            period_col_map[val_str] = df.columns[col_idx]

            # Extract periods mentioned in query
            query_periods = re.findall(r"(fy\d{2}|9mfy\d{2}|\d+mfy\d{2})", query_lower)
            
            # GROWTH CALCULATION
            if "growth" in query_lower and len(query_periods) >= 2:
                period1, period2 = query_periods[0], query_periods[1]
                col1 = period_col_map.get(period1)
                col2 = period_col_map.get(period2)
                
                if col1 and col2 and label_col:
                    # Use LLM or semantic search to find the metric row
                    metric_row = None
                    if self._schema_analyzer:
                        metric_row = self._schema_analyzer.find_row_by_metric(df, schema, query)
                    
                    if metric_row is None:
                        # Fallback: search for keywords
                        for idx, row in df.iterrows():
                            row_label = str(row[label_col]).lower()
                            if "revenue" in row_label and "operations" in row_label:
                                metric_row = idx
                                break
                            if "revenue" in row_label and metric_row is None:
                                metric_row = idx
                    
                    if metric_row is not None:
                        val1 = pd.to_numeric(df.loc[metric_row, col1], errors='coerce')
                        val2 = pd.to_numeric(df.loc[metric_row, col2], errors='coerce')
                        if pd.notna(val1) and pd.notna(val2):
                            growth = float(val2) - float(val1)
                            metric_name = str(df.loc[metric_row, label_col])
                            return AnalysisResult(
                                success=True,
                                result=round(growth, 2),
                                value=round(growth, 2),
                                method="pandas:semantic_growth",
                                explanation=f"Growth in '{metric_name}' from {period1} ({val1}) to {period2} ({val2})"
                            )

            # SINGLE PERIOD VALUE LOOKUP
            if len(query_periods) == 1:
                period = query_periods[0]
                target_col = period_col_map.get(period)
                
                if target_col and label_col:
                    # Find the metric row using LLM
                    metric_row = None
                    if self._schema_analyzer and schema:
                        metric_row = self._schema_analyzer.find_row_by_metric(df, schema, query)
                    
                    if metric_row is not None:
                        val = pd.to_numeric(df.loc[metric_row, target_col], errors='coerce')
                        if pd.notna(val):
                            metric_name = str(df.loc[metric_row, label_col])
                            return AnalysisResult(
                                success=True,
                                result=round(float(val), 2),
                                value=round(float(val), 2),
                                method="pandas:semantic_lookup",
                                explanation=f"Found '{metric_name}' for {period}: {val}"
                            )

            # DATE/MONTH LOOKUP (e.g., "December 2020")
            month_match = re.search(
                r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?",
                query_lower
            )
            if month_match:
                month_name = month_match.group(1)
                year = month_match.group(2) or ""
                month_abbr = month_name[:3]
                
                # Try to find column from schema period_columns first
                # (schema now includes date columns with normalized keys like "dec_2020")
                target_col = None
                
                # Try various key formats that might be in period_col_map
                possible_keys = [
                    f"{month_abbr}_{year}",        # dec_2020
                    f"{month_abbr} {year}",        # dec 2020
                    f"{month_name}_{year}",        # december_2020
                    f"{month_name} {year}",        # december 2020
                ]
                
                for key in possible_keys:
                    if key in period_col_map:
                        target_col = period_col_map[key]
                        break
                
                # Fallback: scan rows for date patterns
                if not target_col:
                    for row_idx in range(min(5, len(df))):
                        for col_idx, val in enumerate(df.iloc[row_idx]):
                            val_str = str(val).lower()
                            # Check for YYYY-MM-DD format with matching month
                            date_match = re.match(r'^(\d{4})-(\d{2})-', val_str)
                            if date_match:
                                if year and date_match.group(1) == year:
                                    month_num = int(date_match.group(2))
                                    expected_month = {'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                                     'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                                     'september': 9, 'october': 10, 'november': 11, 'december': 12}
                                    if month_num == expected_month.get(month_name, 0):
                                        target_col = df.columns[col_idx]
                                        break
                            # Check for text match
                            if month_abbr in val_str:
                                if not year or year in val_str:
                                    target_col = df.columns[col_idx]
                                    break
                
                if target_col and label_col:
                    # Find the metric row
                    metric_row = None
                    if self._schema_analyzer and schema:
                        metric_row = self._schema_analyzer.find_row_by_metric(df, schema, query)
                    else:
                        # Fallback for GMV
                        for idx, row in df.iterrows():
                            row_label = str(row[label_col]).lower()
                            if "gmv" in row_label:
                                metric_row = idx
                                break
                    
                    if metric_row is not None:
                        val = pd.to_numeric(df.loc[metric_row, target_col], errors='coerce')
                        if pd.notna(val):
                            metric_name = str(df.loc[metric_row, label_col])
                            return AnalysisResult(
                                success=True,
                                result=round(float(val), 2),
                                value=round(float(val), 2),
                                method="pandas:semantic_date_lookup",
                                explanation=f"Found '{metric_name}' for {month_name} {year}: {val}"
                            )

            # ROW SUM (total across periods)
            if "total" in query_lower or "sum" in query_lower:
                metric_row = None
                if self._schema_analyzer and schema:
                    metric_row = self._schema_analyzer.find_row_by_metric(df, schema, query)
                
                if metric_row is not None and label_col:
                    values = []
                    for col in df.columns:
                        if col != label_col:
                            val = pd.to_numeric(df.loc[metric_row, col], errors='coerce')
                            if pd.notna(val) and val != 0:
                                values.append(val)
                    
                    if values:
                        total = sum(values)
                        metric_name = str(df.loc[metric_row, label_col])
                        return AnalysisResult(
                            success=True,
                            result=round(float(total), 2),
                            value=round(float(total), 2),
                            method="pandas:semantic_row_sum",
                            explanation=f"Summed '{metric_name}' across {len(values)} periods"
                        )

        except Exception as e:
            logger.warning(f"Semantic Pandas failed: {e}")

        return None


# Singleton instance
_data_analyst: Optional[DataAnalystAgent] = None


def get_data_analyst_agent(llm_wrapper=None) -> DataAnalystAgent:
    """Get or create singleton data analyst agent."""
    global _data_analyst
    if _data_analyst is None:
        _data_analyst = DataAnalystAgent(llm_wrapper)
    elif llm_wrapper and _data_analyst._llm is None:
        _data_analyst._llm = llm_wrapper
    return _data_analyst


__all__ = ["DataAnalystAgent", "AnalysisResult", "get_data_analyst_agent"]
