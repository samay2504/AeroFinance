"""
Data Analyst Agent - SQL-first analytical agent with LLM-driven semantic understanding.
Handles numeric queries, growth calculations, and period-based analysis dynamically.
Uses LLM and NLP techniques to understand data structure rather than hardcoded patterns.
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
    Production-grade LLM-driven data analyst with:
    - Semantic data understanding (NER, similarity matching)
    - Dynamic structure detection (no hardcoded patterns)
    - DuckDB SQL execution with intelligent prompts
    - LLM-generated SQL with full data context
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
        self._semantic_matcher = None
        self._structure_detector = None
        self._query_understanding = None
        self._schema_cache: Dict[str, Any] = {}  # Cache analyzed schemas
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        # Auto-initialize LLM from environment if not provided
        if self._llm is None:
            self._llm = self._init_llm_from_env()
        
        self._init_components()
    
    def _init_llm_from_env(self):
        """
        Initialize LLM wrapper from environment variables.
        Returns an LLMWrapper which provides invoke_with_structured_output.
        """
        try:
            from app.core.llm_wrapper import LLMWrapper
            import os
            
            # Load environment variables
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            
            # Build config from environment with sensible defaults
            # This avoids depending on pydantic settings parsing
            provider_pref_str = os.getenv(
                "LLM_PROVIDER_PREFERENCE", 
                "google_genai,groq,ollama,openrouter,openai"
            )
            provider_preference = [p.strip() for p in provider_pref_str.split(",")]
            
            config = {
                "provider_preference": provider_preference,
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("LLM_RETRY_DELAY", "1.0")),
                "ollama_enabled": os.getenv("OLLAMA_ENABLED", "false").lower() == "true",
                "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                "openrouter_enabled": os.getenv("OPENROUTER_ENABLED", "false").lower() == "true",
                "cache_enabled": True,  # Enable caching
            }
            
            logger.debug(f"LLM config: providers={config['provider_preference']}")
            
            # Return LLMWrapper which has invoke_with_structured_output
            wrapper = LLMWrapper(config)
            
            if wrapper.llm:
                logger.info(f"Auto-initialized LLM wrapper: {wrapper.provider_name}")
                return wrapper
            else:
                logger.warning("LLM wrapper initialization returned no LLM - running without LLM")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to auto-initialize LLM from environment: {e}")
            import traceback
            traceback.print_exc()
            return None

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

        # Initialize semantic understanding modules
        try:
            from app.core.semantic_understanding import (
                get_semantic_matcher,
                get_structure_detector,
                get_query_understanding,
                get_financial_ner,
            )
            self._semantic_matcher = get_semantic_matcher()
            self._structure_detector = get_structure_detector()
            self._query_understanding = get_query_understanding()
            self._ner = get_financial_ner()  # For dynamic metric detection
        except Exception as e:
            logger.warning(f"Semantic understanding unavailable: {e}")
            self._ner = None

        # Initialize tool orchestrator for integrated tool execution
        try:
            from app.tools.orchestrator import get_tool_orchestrator
            self._tool_orchestrator = get_tool_orchestrator(self._llm)
            logger.info(f"Tool orchestrator initialized with {len(self._tool_orchestrator._tools)} tools")
        except Exception as e:
            logger.warning(f"Tool orchestrator unavailable: {e}")
            self._tool_orchestrator = None

        # Initialize prompt compressor for token-efficient LLM queries
        self._prompt_compressor = None
        try:
            from app.core.prompt_compression import get_prompt_compressor
            self._prompt_compressor = get_prompt_compressor(max_tokens=3000)
            logger.debug("Prompt compressor initialized")
        except Exception as e:
            logger.debug(f"Prompt compressor unavailable: {e}")

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
        """
        Match query to best dataset using semantic understanding.
        Prioritizes explicit sheet/file name mentions in the query.
        """
        if not datasets:
            return None

        query_lower = query.lower()
        
        # 1. FIRST: Check for explicit sheet/file name mentions
        # Patterns like "from the X sheet", "X report", "X file" should take priority
        explicit_patterns = [
            r'from\s+(?:the\s+)?(\w+)\s+(?:sheet|report|file|data)',
            r'(?:sheet|report|file)(?:\s+called)?\s+(\w+)',
            r'in\s+(?:the\s+)?(\w+)\s+(?:sheet|report|file)',
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, query_lower)
            if match:
                explicit_name = match.group(1)
                # Find datasets that contain this name
                matching_datasets = [
                    ds for ds in datasets 
                    if explicit_name in ds.get("dataset_id", "").lower()
                ]
                
                if len(matching_datasets) == 1:
                    return matching_datasets[0].get("dataset_id")
                elif len(matching_datasets) > 1:
                    # Multiple matches - use NER to pick based on query context
                    # This is dynamic - uses patterns from FinancialNER
                    query_entities = []
                    if self._ner:
                        query_entities = self._ner.extract_entities(query)
                    
                    # Find the best match based on entity overlap with dataset names
                    best_ds = None
                    best_entity_score = 0
                    
                    for ds in matching_datasets:
                        ds_id = ds.get("dataset_id", "").lower()
                        ds_sheet = ds_id.split(':')[-1]  # Get sheet name part
                        entity_score = 0
                        
                        # Check if query entities match dataset name
                        # Uses SHEET_TYPE_CATEGORIES from FinancialNER (centralized)
                        for entity in query_entities:
                            if entity.entity_type == 'metric':
                                category = entity.metadata.get('category', '')
                                if self._ner:
                                    sheet_categories = getattr(self._ner, 'SHEET_TYPE_CATEGORIES', {})
                                    for sheet_term, categories in sheet_categories.items():
                                        if sheet_term in ds_sheet and category in categories:
                                            entity_score += 10
                                            break
                        
                        # Use semantic matcher for similarity
                        if self._semantic_matcher:
                            sim_score = self._semantic_matcher.calculate_similarity(query, ds_sheet)
                            entity_score += sim_score * 5
                        
                        if entity_score > best_entity_score:
                            best_entity_score = entity_score
                            best_ds = ds.get("dataset_id")
                    
                    if best_ds:
                        return best_ds
                    # Fallback to first match
                    return matching_datasets[0].get("dataset_id")
        
        # 2. Use FULLY DYNAMIC semantic matching for implicit references
        best_match = None
        best_score = 0
        
        # Extract query entities using NER
        query_entities = []
        if self._ner:
            query_entities = self._ner.extract_entities(query)
        
        # Expand query keywords using semantic matcher
        expanded_keywords = set()
        if self._semantic_matcher:
            expanded_keywords = self._semantic_matcher.expand_query(query)

        for dataset in datasets:
            dataset_id = dataset.get("dataset_id", "")
            columns = dataset.get("columns", [])
            
            score = 0.0
            dataset_lower = dataset_id.lower()
            dataset_sheet = dataset_lower.split(':')[-1]
            columns_lower = " ".join(str(c).lower() for c in columns)
            
            # 1. Semantic similarity (DYNAMIC - uses synonym expansion)
            if self._semantic_matcher:
                semantic_score = self._semantic_matcher.calculate_similarity(query, dataset_id)
                score += semantic_score * 15
            
            # 2. Expanded keyword matches (DYNAMIC - uses synonyms)
            for kw in expanded_keywords:
                if len(kw) >= 3:
                    if kw in dataset_lower:
                        score += 5
                    if kw in columns_lower:
                        score += 3
            
            # 3. Period entity matches (DYNAMIC - uses NER patterns)
            for entity in query_entities:
                if entity.entity_type == 'period':
                    period_normalized = entity.normalized
                    if period_normalized in columns_lower or period_normalized in dataset_lower:
                        score += 10
                elif entity.entity_type == 'metric':
                    # Check if metric category relates to dataset type
                    # Uses SHEET_TYPE_CATEGORIES from FinancialNER (centralized, extensible)
                    category = entity.metadata.get('category', '')
                    if self._ner:
                        sheet_categories = getattr(self._ner, 'SHEET_TYPE_CATEGORIES', {})
                        for sheet_term, categories in sheet_categories.items():
                            if sheet_term in dataset_sheet and category in categories:
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

            # Build data sample - ENHANCED: targeted context when strategy needs row+column search
            data_sample = ""
            sample_rows = min(5, len(df))
            
            # Parse query to get strategy
            query_info = None
            if self._query_understanding:
                query_info = self._query_understanding.parse_query(query)
            
            strategy = query_info.get('strategy', {}) if query_info else {}
            
            # If strategy requires both row and column search, try to find and show targeted sample
            if strategy.get('requires_row_search') and strategy.get('requires_column_search'):
                # Try to find the target row
                label_col = None
                if self._structure_detector:
                    label_col_idx = self._structure_detector.detect_label_column(df)
                    if label_col_idx < len(df.columns):
                        label_col = df.columns[label_col_idx]
                
                if label_col:
                    # Find the row that matches the query using semantic matching
                    keywords = query_info.get('keywords', set()) if query_info else set()
                    target_row_idx = self._find_metric_row_semantic(df, label_col, query, keywords)
                    
                    if target_row_idx is not None:
                        # Show 2 rows before and after the target
                        start_idx = max(0, target_row_idx - 2)
                        end_idx = min(len(df), target_row_idx + 3)
                        
                        data_sample = f"TARGETED SAMPLE (around matched row {target_row_idx}):\n"
                        data_sample += df.iloc[start_idx:end_idx].to_string(max_colwidth=35)
                        data_sample += f"\n\nNOTE: Row {target_row_idx} likely contains the target metric: '{df.iloc[target_row_idx][label_col]}'"
                        
                        # Also show first 2 rows for header context
                        if start_idx > 2:
                            data_sample = f"HEADER ROWS:\n{df.head(2).to_string(max_colwidth=35)}\n\n" + data_sample
            
            # Fallback to simple head if no targeted sample
            if not data_sample:
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
                
                # Fix common LLM errors: wrong table name (missing double underscore)
                # Replace single underscore variants with the correct sanitized name
                if safe_table_name not in sql:
                    # Try to find and fix table name variations
                    wrong_name = safe_table_name.replace("__", "_")
                    if wrong_name in sql:
                        sql = sql.replace(wrong_name, safe_table_name)
                    # Remove backticks which DuckDB doesn't use
                    sql = sql.replace("`", "")
                
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
        """Try LLM-generated Python code in sandbox with enhanced context."""
        if not self._llm or not self._sandbox:
            return None

        try:
            from app.core.prompts import get_data_analyst_python_prompt

            schema_str = f"DataFrame: {df_id}\nShape: {df.shape}\nColumns: {list(df.columns)}"
            
            # Enhanced: Targeted sample data
            sample = ""
            
            # Parse query to get strategy
            query_info = None
            if self._query_understanding:
                query_info = self._query_understanding.parse_query(query)
            
            strategy = query_info.get('strategy', {}) if query_info else {}
            
            # If strategy requires both row and column search, provide targeted sample
            if strategy.get('requires_row_search') and strategy.get('requires_column_search'):
                label_col = None
                if self._structure_detector:
                    label_col_idx = self._structure_detector.detect_label_column(df)
                    if label_col_idx < len(df.columns):
                        label_col = df.columns[label_col_idx]
                
                if label_col:
                    keywords = query_info.get('keywords', set()) if query_info else set()
                    target_row_idx = self._find_metric_row_semantic(df, label_col, query, keywords)
                    
                    if target_row_idx is not None:
                        start_idx = max(0, target_row_idx - 2)
                        end_idx = min(len(df), target_row_idx + 3)
                        
                        sample = f"TARGETED SAMPLE (row {target_row_idx} likely matches query):\n"
                        sample += df.iloc[start_idx:end_idx].to_string(max_colwidth=40)
                        sample += f"\n\nLabel column: '{label_col}', Target row label: '{df.iloc[target_row_idx][label_col]}'"
            
            # Fallback to simple head
            if not sample and len(df) > 0:
                sample = df.head(5).to_string()
            
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
        Production-grade semantic Pandas operations.
        Uses NER, semantic matching, and structure detection - no hardcoding.
        Designed to work with any financial data format.
        """
        try:
            # Parse query semantically
            query_info = None
            if self._query_understanding:
                query_info = self._query_understanding.parse_query(query)
            
            query_lower = query.lower()
            
            # ==== SUMMARY/OVERVIEW CHECK FIRST ====
            # Check at the beginning to prioritize summary requests
            strategy = query_info.get('strategy', {}) if query_info else {}
            if strategy.get('lookup_type') == 'summary' or (
                not strategy and any(kw in query_lower for kw in ['summary', 'summarize', 'overview', 'tell me about', 'describe'])
            ):
                # Detect label column
                label_col = None
                if self._structure_detector:
                    label_col_idx = self._structure_detector.detect_label_column(df)
                    if label_col_idx < len(df.columns):
                        label_col = df.columns[label_col_idx]
                
                if not label_col and len(df.columns) > 0:
                    label_col = df.columns[0]
                
                # Generate summary
                summary_parts = []
                summary_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
                
                # Use NER to identify key financial metrics
                if self._ner and label_col:
                    found_metrics = 0
                    for idx, row in df.iterrows():
                        label = str(row[label_col])
                        if label.lower() in ('nan', 'none', '', 'na'):
                            continue
                        
                        entities = self._ner.extract_entities(label)
                        if any(e.entity_type == 'metric' for e in entities):
                            # Get the last non-null numeric value
                            for col in reversed(list(df.columns)):
                                if col != label_col:
                                    val = pd.to_numeric(row[col], errors='coerce')
                                    if pd.notna(val) and val != 0:
                                        summary_parts.append(f"- {row[label_col]}: {round(float(val), 2)}")
                                        found_metrics += 1
                                        break
                            if found_metrics >= 6:
                                break
                
                if len(summary_parts) > 1:
                    summary = "\n".join(summary_parts)
                    return AnalysisResult(
                        success=True,
                        result=summary,
                        method="pandas:heuristic_summary",
                        explanation=f"Generated overview of {df_id or 'dataset'}"
                    )
            
            # Analyze data structure using semantic understanding
            period_col_map = {}
            label_col = None
            structure_label_col = None
            
            if self._structure_detector:
                label_col_idx = self._structure_detector.detect_label_column(df)
                header_row = self._structure_detector.detect_header_row(df)
                period_col_map = self._structure_detector.extract_period_columns(df, header_row)
                structure_label_col = df.columns[label_col_idx] if label_col_idx < len(df.columns) else None
                label_col = structure_label_col
            
            # Add schema analyzer periods (but don't override label col if structure detector found better one)
            schema = None
            if self._schema_analyzer and df_id:
                schema = self._schema_analyzer.analyze(df, df_id, context=df_id)
                if schema:
                    # Merge period columns
                    period_col_map.update(schema.period_columns)
                    # Only use schema label if structure detector didn't find one
                    if not structure_label_col and schema.label_column:
                        label_col = schema.label_column
            
            # Last resort: first column
            if not label_col and len(df.columns) > 0:
                label_col = df.columns[0]

            # Extract periods from query using NER
            periods_in_query = []
            if query_info and query_info.get('periods'):
                periods_in_query = [p['normalized'] for p in query_info['periods']]
            else:
                # Fallback: regex extraction
                periods_in_query = re.findall(r'(fy\d{2}|9mfy\d{2}|\d+mfy\d{2})', query.lower())
            
            # Get query intent
            intents = query_info.get('intent', []) if query_info else []
            query_keywords = query_info.get('keywords', set()) if query_info else set()
            query_lower = query.lower()

            # ==== GROWTH/COMPARISON CALCULATION ====
            if 'comparison' in intents or 'growth' in query_lower or 'change' in query_lower:
                if len(periods_in_query) >= 2:
                    period1, period2 = periods_in_query[0], periods_in_query[1]
                    col1 = period_col_map.get(period1)
                    col2 = period_col_map.get(period2)
                    
                    if col1 and col2 and label_col:
                        metric_row = self._find_metric_row_semantic(
                            df, label_col, query, query_keywords
                        )
                        
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

            # ==== SINGLE PERIOD LOOKUP ====
            if len(periods_in_query) == 1:
                period = periods_in_query[0]
                target_col = period_col_map.get(period)
                
                if target_col and label_col:
                    metric_row = self._find_metric_row_semantic(
                        df, label_col, query, query_keywords
                    )
                    
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

            # ==== DATE/MONTH LOOKUP ====
            # Use NER to extract month references
            month_patterns = [
                (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?', 'full'),
                (r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s\-\']*(\d{2,4})?', 'abbrev'),
            ]
            
            month_name = None
            year = None
            for pattern, _ in month_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    month_name = match.group(1)
                    year = match.group(2) or ""
                    break
            
            if month_name:
                month_abbr = month_name[:3]
                
                # Try normalized keys from period_col_map
                possible_keys = [
                    f"{month_abbr}_{year}",
                    f"{month_abbr} {year}",
                    f"{month_name}_{year}",
                    f"{month_name} {year}",
                ]
                
                target_col = None
                for key in possible_keys:
                    if key in period_col_map:
                        target_col = period_col_map[key]
                        break
                
                # Fallback: scan for dates dynamically
                if not target_col:
                    target_col = self._find_date_column(df, month_name, year)
                
                if target_col and label_col:
                    metric_row = self._find_metric_row_semantic(
                        df, label_col, query, query_keywords
                    )
                    
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

            # ==== AGGREGATION (TOTAL/SUM) ====
            if 'aggregation' in intents or any(w in query_lower for w in ['total', 'sum', 'entire', 'aggregate']):
                metric_row = self._find_metric_row_semantic(
                    df, label_col, query, query_keywords
                )
                
                if metric_row is not None and label_col:
                    # Check if there's a specific period for the total
                    if periods_in_query:
                        period = periods_in_query[0]
                        if period in period_col_map:
                            target_col = period_col_map[period]
                            val = pd.to_numeric(df.loc[metric_row, target_col], errors='coerce')
                            if pd.notna(val):
                                metric_name = str(df.loc[metric_row, label_col])
                                return AnalysisResult(
                                    success=True,
                                    result=round(float(val), 2),
                                    value=round(float(val), 2),
                                    method="pandas:semantic_period_lookup",
                                    explanation=f"Found '{metric_name}' for {period}: {val}"
                                )
                    
                    # Otherwise sum all numeric values in the row
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

            # ==== PERCENTAGE/VARIANCE CALCULATION ====
            if 'percentage' in intents or 'variance' in query_lower or '%' in query:
                # This is complex - delegate to LLM for now
                pass
            
            # ==== SUMMARY/OVERVIEW GENERATION (NO LLM FALLBACK) ====
            # Uses query strategy detection instead of hardcoded keywords
            strategy = query_info.get('strategy', {}) if query_info else {}
            if strategy.get('lookup_type') == 'summary' or (
                not strategy and any(kw in query_lower for kw in ['summary', 'summarize', 'overview'])
            ):
                # Generate a heuristic summary of the data
                summary_parts = []
                
                # 1. Dataset info
                summary_parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
                
                # 2. Use NER to identify key financial metrics dynamically
                if self._ner and label_col:
                    for idx, row in df.iterrows():
                        label = str(row[label_col])
                        if label.lower() in ('nan', 'none', '', 'na'):
                            continue
                        
                        # Use NER to detect if this is a financial metric
                        entities = self._ner.extract_entities(label)
                        if any(e.entity_type == 'metric' for e in entities):
                            # Get the last non-null numeric value
                            for col in reversed(list(df.columns)):
                                if col != label_col:
                                    val = pd.to_numeric(row[col], errors='coerce')
                                    if pd.notna(val) and val != 0:
                                        summary_parts.append(f"- {row[label_col]}: {round(float(val), 2)}")
                                        break
                            if len(summary_parts) >= 7:
                                break
                
                if len(summary_parts) > 1:
                    summary = "\n".join(summary_parts)
                    return AnalysisResult(
                        success=True,
                        result=summary,
                        method="pandas:heuristic_summary",
                        explanation=f"Generated overview of {df_id or 'dataset'}"
                    )

        except Exception as e:
            logger.warning(f"Semantic Pandas failed: {e}")

        return None

    def _find_metric_row_semantic(
        self,
        df: pd.DataFrame,
        label_col: str,
        query: str,
        keywords: set
    ) -> Optional[int]:
        """
        Find the row containing the queried metric using semantic matching.
        Uses the semantic matcher and NER for dynamic term recognition.
        No hardcoded patterns - works with any financial data.
        """
        if not label_col or label_col not in df.columns:
            return None
        
        # Use NER to extract significant terms from query
        query_entities = []
        if self._ner:
            query_entities = self._ner.extract_entities(query)
        
        # Use semantic matcher to expand query keywords
        expanded_keywords = keywords.copy() if keywords else set()
        if self._semantic_matcher:
            expanded_keywords = self._semantic_matcher.expand_query(query)
        
        best_row = None
        best_score = 0.0
        
        for idx, row in df.iterrows():
            label = str(row[label_col])
            
            # Skip empty or purely numeric labels
            if label.lower() in ('nan', 'none', '', 'na'):
                continue
            try:
                float(label.replace(',', ''))
                continue  # Skip numeric values
            except ValueError:
                pass
            
            label_lower = label.lower()
            score = 0.0
            
            # 1. Use semantic matcher for similarity (primary method)
            if self._semantic_matcher:
                semantic_score = self._semantic_matcher.calculate_similarity(query, label)
                score = semantic_score * 10  # Scale appropriately
            
            # 2. Boost for exact keyword matches from expanded keywords
            for kw in expanded_keywords:
                if len(kw) >= 3:  # Only meaningful keywords
                    if re.search(rf'\b{re.escape(kw)}\b', label_lower):
                        score += 5  # Exact word boundary match
                    elif kw in label_lower:
                        score += 2  # Substring match
            
            # 3. Boost for NER entity matches
            if query_entities:
                for entity in query_entities:
                    entity_text = entity.text.lower()
                    if entity_text in label_lower:
                        score += 8  # Strong match for recognized entities
            
            # 4. Check if label itself contains financial metrics (using NER)
            if self._ner and score > 0:
                label_entities = self._ner.extract_entities(label)
                if any(e.entity_type == 'metric' for e in label_entities):
                    score *= 1.2  # Boost for recognized financial metrics
            
            if score > best_score:
                best_score = score
                best_row = idx
        
        # Return if we have a reasonable match
        return best_row if best_score >= 2 else None

    def _find_date_column(
        self,
        df: pd.DataFrame,
        month_name: str,
        year: str
    ) -> Optional[str]:
        """
        Dynamically find the column containing a specific date.
        Works with any date format - searches both column names and header rows.
        """
        month_abbr = month_name[:3].lower()
        month_num_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month_num = month_num_map.get(month_abbr, 0)
        
        # Handle 2-digit year (20 -> 2020)
        year_full = None
        year_short = None
        if year:
            if len(year) == 2:
                year_short = year
                year_full = f"20{year}" if int(year) < 50 else f"19{year}"
            else:
                year_full = year
                year_short = year[-2:]
        
        # 1. First check column names/headers directly
        for col in df.columns:
            col_str = str(col)
            
            # Check datetime objects
            if hasattr(col, 'month') and hasattr(col, 'year'):
                # It's a datetime column
                if col.month == month_num:
                    if not year or str(col.year) == year_full or str(col.year)[-2:] == year_short:
                        return col
            
            # Check string representation
            col_lower = col_str.lower()
            
            # YYYY-MM-DD format (e.g., "2020-12-31 00:00:00")
            date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', col_str)
            if date_match:
                y = date_match.group(1)
                m = int(date_match.group(2))
                if m == month_num:
                    if not year or y == year_full or y[-2:] == year_short:
                        return col
            
            # Text-based patterns (e.g., "Dec-20", "december 2020")
            if month_abbr in col_lower:
                if not year or (year_full and year_full in col_str) or (year_short and year_short in col_lower):
                    return col
        
        # 2. Check header rows (first 5 rows) for date patterns
        for row_idx in range(min(5, len(df))):
            for col_idx, val in enumerate(df.iloc[row_idx]):
                val_str = str(val)
                
                # Skip NaN or empty
                if pd.isna(val) or val_str.lower() in ('nan', ''):
                    continue
                
                # Check datetime objects in data
                if hasattr(val, 'month') and hasattr(val, 'year'):
                    if val.month == month_num:
                        if not year or str(val.year) == year_full or str(val.year)[-2:] == year_short:
                            return df.columns[col_idx]
                
                # YYYY-MM-DD format in cell values
                date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', val_str)
                if date_match:
                    y = date_match.group(1)
                    m = int(date_match.group(2))
                    if m == month_num:
                        if not year or y == year_full or y[-2:] == year_short:
                            return df.columns[col_idx]
                
                # Text-based dates
                val_lower = val_str.lower()
                if month_abbr in val_lower:
                    if not year or (year_full and year_full in val_str) or (year_short and year_short in val_lower):
                        return df.columns[col_idx]
        
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
