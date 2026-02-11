"""
Schema Analyzer - LLM-driven semantic data understanding.
Analyzes DataFrame structure, identifies columns, periods, and metrics dynamically.
No hardcoding - uses LLM reasoning to understand data context.
"""
import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Semantic information about a column."""
    name: str
    index: int
    dtype: str
    semantic_type: str = "unknown"  # period, metric_label, value, date, identifier
    description: str = ""
    sample_values: List[Any] = field(default_factory=list)
    period_value: Optional[str] = None  # If it's a period column, what period


@dataclass
class DataSchema:
    """Semantic schema understanding of a DataFrame."""
    label_column: Optional[str] = None  # Column containing row labels/metric names
    period_columns: Dict[str, str] = field(default_factory=dict)  # period_name -> column_name
    value_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    header_row_index: int = 0  # Row containing the actual headers
    data_start_row: int = 0  # Row where actual data starts
    columns: List[ColumnInfo] = field(default_factory=list)
    summary: str = ""


class SchemaAnalyzer:
    """
    LLM-driven schema analyzer for understanding complex Excel/DataFrame structures.
    Delegates semantic matching to SemanticMatcher and period detection to FinancialNER
    from semantic_understanding.py to avoid logic duplication.
    """

    def __init__(self, llm_wrapper=None):
        self._llm = llm_wrapper
        self._cache: Dict[str, DataSchema] = {}
        self._semantic_matcher = None  # Lazy-loaded
        self._financial_ner = None     # Lazy-loaded

    def _get_matcher(self):
        """Lazy-load SemanticMatcher to avoid circular imports."""
        if self._semantic_matcher is None:
            try:
                from app.core.semantic_understanding import get_semantic_matcher
                self._semantic_matcher = get_semantic_matcher()
            except ImportError:
                logger.warning("SemanticMatcher unavailable, using basic matching")
        return self._semantic_matcher

    def _get_ner(self):
        """Lazy-load FinancialNER to avoid circular imports."""
        if self._financial_ner is None:
            try:
                from app.core.semantic_understanding import get_financial_ner
                self._financial_ner = get_financial_ner()
            except ImportError:
                logger.warning("FinancialNER unavailable, using basic period detection")
        return self._financial_ner

    def analyze(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        context: str = ""
    ) -> DataSchema:
        """
        Analyze DataFrame structure using LLM for semantic understanding.
        
        Args:
            df: DataFrame to analyze
            dataset_id: Unique identifier for caching
            context: Additional context (e.g., filename, sheet name)
            
        Returns:
            DataSchema with semantic understanding of the data
        """
        # Check cache
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        schema = DataSchema()
        
        # Step 1: Basic structure analysis
        schema = self._analyze_basic_structure(df, schema)
        
        # Step 2: If LLM available, get semantic understanding
        if self._llm:
            schema = self._llm_semantic_analysis(df, schema, context)
        else:
            # Fallback to heuristic analysis
            schema = self._heuristic_analysis(df, schema)

        # Cache the result
        self._cache[dataset_id] = schema
        
        return schema

    def _analyze_basic_structure(self, df: pd.DataFrame, schema: DataSchema) -> DataSchema:
        """Analyze basic DataFrame structure without LLM."""
        for idx, col in enumerate(df.columns):
            sample_vals = df[col].dropna().head(5).tolist()
            col_info = ColumnInfo(
                name=str(col),
                index=idx,
                dtype=str(df[col].dtype),
                sample_values=sample_vals
            )
            schema.columns.append(col_info)

        # Identify potential header rows by checking first few rows for text patterns
        for row_idx in range(min(5, len(df))):
            row_data = df.iloc[row_idx].astype(str).tolist()
            text_count = sum(1 for v in row_data if not self._is_numeric_like(v) and v.lower() not in ['nan', 'none', ''])
            if text_count > len(row_data) * 0.5:
                # This row has mostly text - could be a header row
                schema.header_row_index = max(schema.header_row_index, row_idx)

        schema.data_start_row = schema.header_row_index + 1
        
        return schema

    def _is_numeric_like(self, val: str) -> bool:
        """Check if a string looks like a number."""
        try:
            cleaned = val.replace(',', '').replace('%', '').replace('$', '').replace('₹', '').strip()
            if cleaned in ['nan', 'none', '-', '']:
                return False
            float(cleaned)
            return True
        except (ValueError, TypeError):
            return False

    def _heuristic_analysis(self, df: pd.DataFrame, schema: DataSchema) -> DataSchema:
        """Fallback heuristic analysis when LLM is unavailable.
        Delegates period detection to FinancialNER from semantic_understanding."""
        # First column is usually the label column
        if len(df.columns) > 0:
            schema.label_column = df.columns[0]
            schema.columns[0].semantic_type = "metric_label"

        ner = self._get_ner()
        month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        for row_idx in range(min(5, len(df))):
            for col_idx, val in enumerate(df.iloc[row_idx]):
                val_str = str(val).strip()
                col_name = df.columns[col_idx]

                # Delegate period/date detection to FinancialNER
                if ner:
                    entities = ner.extract_entities(val_str)
                    period_entities = [e for e in entities if e.entity_type == 'period']
                    if period_entities:
                        normalized = period_entities[0].normalized
                        schema.period_columns[normalized] = col_name
                        ptype = period_entities[0].metadata.get('period_type', 'period')
                        if col_idx < len(schema.columns):
                            schema.columns[col_idx].semantic_type = ptype
                            schema.columns[col_idx].period_value = normalized
                        if ptype == 'date':
                            schema.date_columns.append(col_name)
                        continue

                # Basic fallback if NER unavailable
                val_lower = val_str.lower()
                date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})', val_lower)
                if date_match:
                    year = date_match.group(1)
                    month = int(date_match.group(2))
                    month_name = month_names[month - 1]
                    date_key = f"{month_name}_{year}"
                    schema.period_columns[date_key] = col_name
                    schema.period_columns[f"{month_name} {year}"] = col_name
                    schema.date_columns.append(col_name)
                    if col_idx < len(schema.columns):
                        schema.columns[col_idx].semantic_type = "date"
                        schema.columns[col_idx].period_value = date_key

        return schema

    def _llm_semantic_analysis(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        context: str
    ) -> DataSchema:
        """Use LLM to semantically understand the data structure."""
        
        # Prepare data sample for LLM
        sample_rows = min(8, len(df))
        sample_cols = min(10, len(df.columns))
        
        data_sample = []
        for row_idx in range(sample_rows):
            row_data = []
            for col_idx in range(sample_cols):
                val = df.iloc[row_idx, col_idx]
                row_data.append(str(val)[:50])  # Truncate long values
            data_sample.append(row_data)

        column_names = [str(c) for c in df.columns[:sample_cols]]

        prompt = f"""You are a data analyst examining a financial spreadsheet.

CONTEXT: {context or "Financial data sheet"}

COLUMN NAMES: {json.dumps(column_names)}

DATA SAMPLE (first {sample_rows} rows):
{json.dumps(data_sample, indent=2)}

TASK: Analyze this data structure and identify:
1. Which column (by index) contains the row labels/metric names (e.g., "Revenue", "GMV")?
2. Which row index contains the period headers (e.g., "FY21", "9MFY22", "Dec-20")?
3. For each column that represents a time period, what is the period value?

Be precise about row/column indices (0-based).

OUTPUT (JSON only, no explanation):
{{
    "label_column_index": <int or null>,
    "header_row_index": <int>,
    "data_start_row": <int>,
    "period_columns": {{
        "<period_value>": <column_index>,
        ...
    }},
    "summary": "<brief description of what this data represents>"
}}"""

        try:
            response = self._llm.invoke_with_structured_output(
                prompt,
                output_schema={
                    "label_column_index": int,
                    "header_row_index": int,
                    "data_start_row": int,
                    "period_columns": dict,
                    "summary": str
                }
            )

            if "error" not in response:
                # Apply LLM insights to schema
                label_idx = response.get("label_column_index")
                if label_idx is not None and 0 <= label_idx < len(df.columns):
                    schema.label_column = df.columns[label_idx]
                    schema.columns[label_idx].semantic_type = "metric_label"

                # Handle None returns explicitly - LLM may return null/None instead of int
                header_idx = response.get("header_row_index")
                schema.header_row_index = int(header_idx) if header_idx is not None else 0
                
                data_start = response.get("data_start_row")
                schema.data_start_row = int(data_start) if data_start is not None else (schema.header_row_index + 1)
                
                schema.summary = response.get("summary", "") or ""

                # Map period columns
                period_cols = response.get("period_columns", {})
                for period_val, col_idx in period_cols.items():
                    if isinstance(col_idx, int) and 0 <= col_idx < len(df.columns):
                        col_name = df.columns[col_idx]
                        schema.period_columns[period_val.lower()] = col_name
                        if col_idx < len(schema.columns):
                            schema.columns[col_idx].semantic_type = "period"
                            schema.columns[col_idx].period_value = period_val.lower()

                logger.info(f"LLM schema analysis: {len(schema.period_columns)} periods found")

        except Exception as e:
            logger.warning(f"LLM schema analysis failed: {e}, using heuristics")
            schema = self._heuristic_analysis(df, schema)

        return schema

    def find_row_by_metric(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        metric_query: str
    ) -> Optional[int]:
        """
        Find the row index containing a specific metric.
        Delegates similarity scoring to SemanticMatcher from semantic_understanding.
        
        Args:
            df: DataFrame
            schema: Pre-analyzed schema
            metric_query: What metric to look for (e.g., "revenue from operations")
            
        Returns:
            Row index or None
        """
        if not schema.label_column:
            return None

        matcher = self._get_matcher()
        best_match = None
        best_score = 0.0

        for idx, row in df.iterrows():
            label = str(row[schema.label_column])
            if label.lower() in ('nan', 'none', ''):
                continue

            if matcher:
                score = matcher.calculate_similarity(metric_query, label)
            else:
                # Basic fallback: word overlap
                q_words = set(metric_query.lower().split())
                l_words = set(label.lower().split())
                score = len(q_words & l_words) / max(len(q_words | l_words), 1)

            if score > best_score:
                best_score = score
                best_match = idx

        # If low confidence and LLM available, ask LLM
        if best_score < 0.3 and self._llm:
            return self._llm_find_row(df, schema, metric_query)

        return best_match if best_score > 0.1 else None

    def _llm_find_row(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        metric_query: str
    ) -> Optional[int]:
        """Use LLM to find the row containing a specific metric."""
        if not schema.label_column:
            return None

        # Get all row labels
        labels = []
        for idx, row in df.iterrows():
            label = str(row[schema.label_column])
            if label.lower() not in ['nan', 'none', '']:
                labels.append({"index": idx, "label": label[:100]})

        if not labels:
            return None

        prompt = f"""Given these row labels from a financial spreadsheet:
{json.dumps(labels[:30], indent=2)}

Which row (by index) best matches the query: "{metric_query}"?

Return ONLY the numeric index, nothing else. If no match, return -1."""

        try:
            response = self._llm.invoke(prompt)
            match = re.search(r'-?\d+', response.strip())
            if match:
                idx = int(match.group())
                if idx >= 0 and idx < len(df):
                    return idx
        except Exception as e:
            logger.warning(f"LLM row matching failed: {e}")

        return None

    def get_value_at(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        metric: str,
        period: str
    ) -> Optional[float]:
        """
        Get a specific value by metric name and period.
        
        Args:
            df: DataFrame
            schema: Pre-analyzed schema
            metric: Metric name (e.g., "revenue from operations")
            period: Period name (e.g., "fy21", "9mfy22")
            
        Returns:
            Numeric value or None
        """
        period_lower = period.lower()
        
        # Find the column for this period
        if period_lower not in schema.period_columns:
            logger.debug(f"Period '{period}' not found in schema. Available: {list(schema.period_columns.keys())}")
            return None

        col = schema.period_columns[period_lower]
        
        # Find the row for this metric
        row_idx = self.find_row_by_metric(df, schema, metric)
        if row_idx is None:
            logger.debug(f"Metric '{metric}' not found in data")
            return None

        # Get the value
        try:
            val = df.loc[row_idx, col]
            return pd.to_numeric(val, errors='coerce')
        except Exception as e:
            logger.warning(f"Failed to get value at [{row_idx}, {col}]: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════
    # INTELLIGENT SCHEMA FILTERING (PRD: Enhancement 3)
    # Reduces token usage by 70-96% for large DataFrames (100+ columns)
    # Delegates similarity scoring to SemanticMatcher (semantic_understanding)
    # to avoid duplicating synonym/fuzzy-matching logic.
    # ═══════════════════════════════════════════════════════════════════

    # Key column patterns: always include temporal, ID, and label columns
    _KEY_COLUMN_PATTERNS = [
        "date", "time", "period",   # Temporal
        "id", "key", "code",        # Identifiers
        "name", "description",      # Labels
        "fy", "quarter", "year",    # Financial periods
    ]

    def get_relevant_schema(
        self,
        df: pd.DataFrame,
        user_query: str,
        max_columns: int = 20,
        include_all_if_small: bool = True,
        similarity_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Extract only columns relevant to the user's query.
        Reduces token usage by 70-96% on large DataFrames.

        Delegates similarity scoring to SemanticMatcher from
        semantic_understanding.py (synonym expansion, fuzzy matching,
        spaCy vectors, financial acronym expansion).

        Args:
            df: Source DataFrame
            user_query: User's question
            max_columns: Maximum columns to return
            include_all_if_small: If df has <=30 columns, include all
            similarity_threshold: Min similarity score to include a column (0-1)

        Returns:
            {
                "relevant_columns": List[str],
                "column_types": Dict[str, str],
                "sample_values": Dict[str, List],
                "schema_string": str,  # LLM-ready description
                "compression_ratio": float,
                "total_columns": int,
                "filtered_columns": int,
            }
        """
        all_columns = list(df.columns)
        total_columns = len(all_columns)

        # Small DataFrames: include everything
        if include_all_if_small and total_columns <= 30:
            logger.info(f"DataFrame has {total_columns} columns - including all")
            return self._build_full_filtered_schema(df)

        # Large DataFrames: intelligent filtering
        logger.info(f"DataFrame has {total_columns} columns - applying smart filtering")

        matcher = self._get_matcher()
        scored_columns: List[Tuple[str, float]] = []

        # ══════ Strategy 1: Semantic similarity scoring (via SemanticMatcher) ══════
        # SemanticMatcher handles: exact match, substring, acronym expansion,
        # spaCy vectors, synonym Jaccard — all in one call.
        for col in all_columns:
            if matcher:
                # Convert underscored col names to readable form for matching
                col_readable = col.replace('_', ' ')
                score = matcher.calculate_similarity(user_query, col_readable)
            else:
                # Basic fallback: keyword overlap
                query_words = set(user_query.lower().split())
                col_words = set(col.lower().replace('_', ' ').split())
                overlap = query_words & col_words
                score = len(overlap) / max(len(query_words | col_words), 1)

            scored_columns.append((col, score))

        # ══════ Strategy 2: Key Columns (always include temporal/ID cols) ══════
        key_columns: set = set()
        for col in all_columns:
            col_lower = col.lower()
            for pattern in self._KEY_COLUMN_PATTERNS:
                if pattern in col_lower:
                    key_columns.add(col)
                    break

        # ══════ Combine: scored columns above threshold + key columns ══════
        relevant_columns: set = key_columns.copy()
        # Sort by score descending
        scored_columns.sort(key=lambda x: x[1], reverse=True)
        for col, score in scored_columns:
            if score >= similarity_threshold:
                relevant_columns.add(col)

        # ══════ Limit & Fallback ══════
        relevant_list = list(relevant_columns)

        if len(relevant_list) > max_columns:
            # Keep key columns + top scoring non-key columns
            non_key_scored = [(c, s) for c, s in scored_columns if c not in key_columns]
            non_key_scored.sort(key=lambda x: x[1], reverse=True)
            relevant_list = list(key_columns) + [c for c, _ in non_key_scored[:max_columns - len(key_columns)]]
            logger.warning(f"Truncated to {len(relevant_list)} columns")

        if len(relevant_list) < 5:
            logger.warning(
                f"Only {len(relevant_list)} relevant columns found - adding first 10 as fallback"
            )
            fallback = [c for c in all_columns if c not in relevant_columns]
            relevant_list.extend(fallback[:10])

        # Build filtered schema
        schema_result = {
            "relevant_columns": relevant_list,
            "column_types": {col: str(df[col].dtype) for col in relevant_list},
            "sample_values": {
                col: df[col].dropna().head(3).tolist() for col in relevant_list
            },
            "schema_string": self._format_filtered_schema_string(df, relevant_list),
            "compression_ratio": round(len(relevant_list) / total_columns, 3),
            "total_columns": total_columns,
            "filtered_columns": len(relevant_list),
        }

        logger.info(
            f"Schema compression: {total_columns} → {len(relevant_list)} columns "
            f"({schema_result['compression_ratio']:.1%} retained)"
        )
        return schema_result

    def _format_filtered_schema_string(self, df: pd.DataFrame, columns: List[str]) -> str:
        """Format filtered schema for LLM consumption."""
        schema_lines = ["**Available Columns:**\n"]
        for col in columns:
            dtype = df[col].dtype
            sample_vals = df[col].dropna().head(2).tolist()
            if sample_vals:
                sample_str = ", ".join(str(v) for v in sample_vals)
                schema_lines.append(f"- `{col}` ({dtype}): Examples: [{sample_str}]")
            else:
                schema_lines.append(f"- `{col}` ({dtype})")
        return "\n".join(schema_lines)

    def _build_full_filtered_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return complete schema for small DataFrames."""
        all_columns = list(df.columns)
        return {
            "relevant_columns": all_columns,
            "column_types": {col: str(df[col].dtype) for col in all_columns},
            "sample_values": {
                col: df[col].dropna().head(3).tolist() for col in all_columns
            },
            "schema_string": self._format_filtered_schema_string(df, all_columns),
            "compression_ratio": 1.0,
            "total_columns": len(all_columns),
            "filtered_columns": len(all_columns),
        }


# Singleton
_analyzer: Optional[SchemaAnalyzer] = None


def get_schema_analyzer(llm_wrapper=None) -> SchemaAnalyzer:
    """Get or create singleton schema analyzer."""
    global _analyzer
    if _analyzer is None or (llm_wrapper and _analyzer._llm is None):
        _analyzer = SchemaAnalyzer(llm_wrapper)
    return _analyzer


__all__ = ["SchemaAnalyzer", "DataSchema", "ColumnInfo", "get_schema_analyzer"]
