"""
SQL Templates - Deterministic SQL templates for common CA calculations.
Template-first approach to reduce LLM calls and ensure consistency.
"""
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SQLTemplate:
    """SQL template definition."""
    name: str
    pattern: str  # Regex pattern to match queries
    template: str  # SQL template with placeholders
    required_columns: List[str]  # Column name patterns required
    description: str


# Column synonym mappings for flexible matching
COLUMN_SYNONYMS = {
    "revenue": ["revenue", "sales", "income", "turnover", "revenue_from_operations"],
    "cost": ["cost", "expense", "expenditure", "cogs", "cost_of_goods"],
    "profit": ["profit", "earnings", "pbt", "pat", "ebit", "ebitda", "net_income"],
    "growth": ["growth", "change", "variance", "delta"],
    "date": ["date", "period", "month", "year", "quarter", "fy"],
    "amount": ["amount", "value", "total", "sum"],
    "gmv": ["gmv", "gross_merchandise_value", "gross_value"],
    "order": ["order", "orders", "transaction", "transactions"],
}


# Template library
TEMPLATES: Dict[str, SQLTemplate] = {
    "sum_column": SQLTemplate(
        name="sum_column",
        pattern=r"(total|sum|aggregate)\s+(?:of\s+)?['\"]?(\w+)['\"]?",
        template="SELECT SUM(CAST({column} AS DOUBLE)) as total FROM {table}",
        required_columns=["numeric"],
        description="Sum a numeric column",
    ),
    
    "sum_by_period": SQLTemplate(
        name="sum_by_period",
        pattern=r"total\s+.+\s+for\s+.*(fy\d{2}|q\d|h\d|\d{4})",
        template="""
            SELECT {period_col}, SUM(CAST({value_col} AS DOUBLE)) as total 
            FROM {table} 
            WHERE LOWER({period_col}) = LOWER('{period_value}')
            GROUP BY {period_col}
        """,
        required_columns=["period", "numeric"],
        description="Sum by fiscal period",
    ),
    
    "growth_absolute": SQLTemplate(
        name="growth_absolute",
        pattern=r"(absolute\s+)?growth.*(from|between).*(fy\d{2}|q\d).*(to|and).*(fy\d{2}|q\d)",
        template="""
            WITH periods AS (
                SELECT 
                    MAX(CASE WHEN LOWER(column_name) LIKE '%{period1}%' THEN {value_col} END) as period1_val,
                    MAX(CASE WHEN LOWER(column_name) LIKE '%{period2}%' THEN {value_col} END) as period2_val
                FROM {table}
                WHERE LOWER({row_col}) LIKE '%{metric}%'
            )
            SELECT period2_val - period1_val as absolute_growth FROM periods
        """,
        required_columns=["metric_row", "period_columns"],
        description="Calculate absolute growth between periods",
    ),
    
    "value_at_period": SQLTemplate(
        name="value_at_period",
        pattern=r"(what\s+was|value\s+of|get).+?(for|in|at).*(fy\d{2}|q\d|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)",
        template="""
            SELECT {value_col} as value
            FROM {table}
            WHERE LOWER({row_col}) LIKE '%{metric}%'
        """,
        required_columns=["metric_row", "period_column"],
        description="Get value at specific period",
    ),
    
    "variance": SQLTemplate(
        name="variance",
        pattern=r"variance.*(actual|budget|forecast|projected|model)",
        template="""
            SELECT 
                {actual_col} as actual,
                {budget_col} as budget,
                ({actual_col} - {budget_col}) as variance,
                ROUND((({actual_col} - {budget_col}) / NULLIF({budget_col}, 0)) * 100, 2) as variance_pct
            FROM {table}
            WHERE LOWER({row_col}) LIKE '%{metric}%'
        """,
        required_columns=["actual", "budget", "metric_row"],
        description="Calculate variance between actual and budget",
    ),
    
    "average_column": SQLTemplate(
        name="average_column",
        pattern=r"(average|avg|mean)\s+(?:of\s+)?['\"]?(\w+)['\"]?",
        template="SELECT AVG(CAST({column} AS DOUBLE)) as average FROM {table}",
        required_columns=["numeric"],
        description="Average of a numeric column",
    ),
    
    "count_rows": SQLTemplate(
        name="count_rows",
        pattern=r"(how\s+many|count|number\s+of)\s+(rows|records|entries)",
        template="SELECT COUNT(*) as count FROM {table}",
        required_columns=[],
        description="Count rows in table",
    ),

    "specific_cell": SQLTemplate(
        name="specific_cell",
        pattern=r"(what|get|show).+(for|of).+",
        template="""
            SELECT {column} as value
            FROM {table}
            WHERE {row_condition}
        """,
        required_columns=["target_column", "row_filter"],
        description="Get specific cell value",
    ),
}


class SQLTemplateEngine:
    """
    Template engine for deterministic SQL generation.
    Matches queries to templates before falling back to LLM.
    """

    def __init__(self, templates: Optional[Dict[str, SQLTemplate]] = None):
        self.templates = templates or TEMPLATES
        self.column_synonyms = COLUMN_SYNONYMS

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return re.sub(r"\s+", " ", text.lower().strip())

    def _find_matching_column(
        self,
        search_term: str,
        available_columns: List[str],
        column_type: Optional[str] = None
    ) -> Optional[str]:
        """Find best matching column from available columns."""
        search_lower = str(search_term).lower()
        
        # Direct match
        for col in available_columns:
            col_str = str(col).lower()
            if search_lower == col_str:
                return col

        # Partial match
        for col in available_columns:
            col_str = str(col).lower()
            if search_lower in col_str or col_str in search_lower:
                return col

        # Synonym match
        for synonym_key, synonyms in self.column_synonyms.items():
            if search_lower in synonyms or any(s in search_lower for s in synonyms):
                for col in available_columns:
                    col_str = str(col).lower()
                    if any(s in col_str for s in synonyms):
                        return col

        return None

    def _extract_periods(self, query: str) -> List[str]:
        """Extract period references from query."""
        periods = []
        
        # FY patterns
        fy_matches = re.findall(r"fy\d{2}", query.lower())
        periods.extend(fy_matches)
        
        # Month+FY patterns
        mfy_matches = re.findall(r"\d{1,2}mfy\d{2}", query.lower())
        periods.extend(mfy_matches)
        
        # Quarter patterns
        q_matches = re.findall(r"q\d(?:fy\d{2})?", query.lower())
        periods.extend(q_matches)
        
        # Month year patterns
        month_matches = re.findall(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{2,4}",
            query.lower()
        )
        periods.extend(month_matches)

        return periods

    def match_template(
        self,
        query: str,
        available_columns: List[str],
        table_name: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Match query to a template and extract parameters.
        
        Returns:
            Tuple of (sql_query, params_used) or None if no match
        """
        query_normalized = self._normalize_text(query)

        for template_name, template in self.templates.items():
            if re.search(template.pattern, query_normalized, re.IGNORECASE):
                logger.debug(f"Template match: {template_name}")
                
                # Try to fill template parameters
                params = self._extract_template_params(
                    query, template, available_columns, table_name
                )
                
                if params:
                    try:
                        sql = template.template.format(**params)
                        return sql.strip(), {"template": template_name, **params}
                    except KeyError as e:
                        logger.warning(f"Template {template_name} missing param: {e}")
                        continue

        return None

    def _extract_template_params(
        self,
        query: str,
        template: SQLTemplate,
        available_columns: List[str],
        table_name: str
    ) -> Optional[Dict[str, Any]]:
        """Extract parameters for a template from the query."""
        params = {"table": table_name}
        query_lower = query.lower()

        # Extract periods
        periods = self._extract_periods(query)

        if template.name == "sum_column":
            # Find numeric column mentioned in query
            for word in query_lower.split():
                col = self._find_matching_column(word, available_columns)
                if col:
                    params["column"] = col
                    return params

        elif template.name == "growth_absolute":
            if len(periods) >= 2:
                params["period1"] = periods[0]
                params["period2"] = periods[1]
                
                # Find metric row
                for synonym_key, synonyms in self.column_synonyms.items():
                    for syn in synonyms:
                        if syn in query_lower:
                            params["metric"] = syn
                            break
                
                # Find value column (usually first period column)
                for col in available_columns:
                    if periods[0] in col.lower():
                        params["value_col"] = col
                        break
                
                # Find row identifier column
                row_cols = [c for c in available_columns if "name" in c.lower() or "item" in c.lower() or "particular" in c.lower()]
                if row_cols:
                    params["row_col"] = row_cols[0]
                    return params

        elif template.name == "value_at_period":
            if periods:
                # Find column matching period
                for col in available_columns:
                    if periods[0] in col.lower():
                        params["value_col"] = col
                        break
                
                # Extract metric
                for synonym_key, synonyms in self.column_synonyms.items():
                    for syn in synonyms:
                        if syn in query_lower:
                            params["metric"] = syn
                            break
                
                # Row identifier
                row_cols = [c for c in available_columns if any(x in c.lower() for x in ["name", "item", "particular", "description"])]
                if row_cols:
                    params["row_col"] = row_cols[0]
                    if "metric" in params and "value_col" in params:
                        return params

        elif template.name in ["sum_by_period", "average_column", "count_rows"]:
            # Simple templates - just need table and column
            if template.name == "count_rows":
                return params
            
            for word in query_lower.split():
                col = self._find_matching_column(word, available_columns)
                if col:
                    params["column"] = col
                    if periods:
                        params["period_col"] = periods[0]
                        params["period_value"] = periods[0]
                        params["value_col"] = col
                    return params

        return None

    def generate_deterministic_sql(
        self,
        query: str,
        schema_info: Dict[str, Any],
        table_name: str
    ) -> Optional[Tuple[str, str]]:
        """
        Attempt deterministic SQL generation without LLM.
        
        Returns:
            Tuple of (sql, method_description) or None
        """
        columns = schema_info.get("columns", [])
        
        result = self.match_template(query, columns, table_name)
        if result:
            sql, params = result
            return sql, f"template:{params.get('template', 'unknown')}"
        
        return None


# Singleton instance
_template_engine: Optional[SQLTemplateEngine] = None


def get_template_engine() -> SQLTemplateEngine:
    """Get or create singleton template engine."""
    global _template_engine
    if _template_engine is None:
        _template_engine = SQLTemplateEngine()
    return _template_engine


__all__ = ["SQLTemplateEngine", "SQLTemplate", "get_template_engine", "COLUMN_SYNONYMS"]
