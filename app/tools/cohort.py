"""
Cohort Analysis Tool - Build cohort retention analysis.
Generates SQL or Pandas code for cohort analysis.
"""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from langchain.tools import tool
except ImportError:
    def tool(name: str = None, return_direct: bool = False):
        def decorator(func):
            func.name = name or func.__name__
            return func
        return decorator


@tool("cohort_analysis", return_direct=True)
def cohort_analysis(
    dataset_id: str,
    date_column: str,
    user_column: str,
    value_column: Optional[str] = None,
    cohort_period: str = "month"
) -> Dict[str, Any]:
    """
    Build cohort retention analysis from transaction data.
    
    Args:
        dataset_id: Dataset ID containing transaction data
        date_column: Column name for transaction date
        user_column: Column name for user/customer ID
        value_column: Optional column for value aggregation
        cohort_period: Period for cohort grouping ('month', 'week', 'quarter')
        
    Returns:
        Dict with cohort matrix, retention rates, and SQL/code
    """
    try:
        from app.core.data_registry import get_data_registry
        registry = get_data_registry()

        df = registry.get(dataset_id)
        if df is None:
            return {
                "result": "error",
                "explain": f"Dataset not found: {dataset_id}"
            }

        # Find matching columns
        date_col = _find_column(df, date_column)
        user_col = _find_column(df, user_column)
        val_col = _find_column(df, value_column) if value_column else None

        if not date_col:
            return {
                "result": "error",
                "explain": f"Date column not found: {date_column}",
                "available_columns": list(df.columns)
            }

        if not user_col:
            return {
                "result": "error",
                "explain": f"User column not found: {user_column}",
                "available_columns": list(df.columns)
            }

        # Perform cohort analysis
        return _build_cohort_analysis(df, date_col, user_col, val_col, cohort_period)

    except Exception as e:
        logger.error(f"Cohort analysis failed: {e}")
        return {
            "result": "error",
            "explain": f"Analysis failed: {str(e)}"
        }


def _find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    """Find matching column in DataFrame."""
    if not target:
        return None
    for col in df.columns:
        if target.lower() == col.lower():
            return col
        if target.lower() in col.lower():
            return col
    return None


def _build_cohort_analysis(
    df: pd.DataFrame,
    date_col: str,
    user_col: str,
    value_col: Optional[str],
    period: str
) -> Dict[str, Any]:
    """Build cohort retention matrix."""
    
    df = df.copy()
    
    # Parse dates
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        return {
            "result": "error",
            "explain": f"Could not parse dates in {date_col}: {e}"
        }

    # Determine cohort period
    if period == "week":
        df["cohort"] = df[date_col].dt.to_period("W")
        df["period"] = df[date_col].dt.to_period("W")
    elif period == "quarter":
        df["cohort"] = df[date_col].dt.to_period("Q")
        df["period"] = df[date_col].dt.to_period("Q")
    else:  # month
        df["cohort"] = df[date_col].dt.to_period("M")
        df["period"] = df[date_col].dt.to_period("M")

    # Get first transaction cohort for each user
    user_cohorts = df.groupby(user_col)["cohort"].min().reset_index()
    user_cohorts.columns = [user_col, "first_cohort"]
    
    df = df.merge(user_cohorts, on=user_col)

    # Calculate period offset
    df["period_offset"] = (df["period"].astype(str).astype("datetime64[ns]") - 
                           df["first_cohort"].astype(str).astype("datetime64[ns]")).dt.days
    
    if period == "week":
        df["period_offset"] = df["period_offset"] // 7
    elif period == "quarter":
        df["period_offset"] = df["period_offset"] // 90
    else:
        df["period_offset"] = df["period_offset"] // 30

    # Build cohort matrix
    cohort_data = df.groupby(["first_cohort", "period_offset"])[user_col].nunique().reset_index()
    cohort_data.columns = ["cohort", "period", "users"]

    # Pivot to matrix
    cohort_matrix = cohort_data.pivot(index="cohort", columns="period", values="users")
    
    # Calculate retention rates
    cohort_sizes = cohort_matrix.iloc[:, 0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
    retention_matrix = retention_matrix.round(2)

    # Generate SQL for reproducibility
    sql_code = _generate_cohort_sql(date_col, user_col, period)

    # Summary statistics
    avg_retention = {}
    for col in retention_matrix.columns:
        if col > 0:
            avg_retention[f"period_{col}"] = round(retention_matrix[col].mean(), 2)

    return {
        "result": "success",
        "cohort_matrix": cohort_matrix.to_dict(),
        "retention_matrix": retention_matrix.to_dict(),
        "cohort_sizes": cohort_sizes.to_dict(),
        "average_retention": avg_retention,
        "num_cohorts": len(cohort_matrix),
        "periods_tracked": len(cohort_matrix.columns),
        "sql_code": sql_code,
        "explain": f"Built {len(cohort_matrix)} cohorts over {len(cohort_matrix.columns)} periods"
    }


def _generate_cohort_sql(date_col: str, user_col: str, period: str) -> str:
    """Generate SQL for cohort analysis."""
    period_func = {
        "month": "DATE_TRUNC('month', {date_col})",
        "week": "DATE_TRUNC('week', {date_col})",
        "quarter": "DATE_TRUNC('quarter', {date_col})"
    }.get(period, "DATE_TRUNC('month', {date_col})")
    
    return f"""
-- Cohort Analysis SQL (DuckDB compatible)
WITH user_cohorts AS (
    SELECT 
        {user_col},
        MIN({period_func.format(date_col=date_col)}) as first_cohort
    FROM data
    GROUP BY {user_col}
),
cohort_periods AS (
    SELECT 
        uc.first_cohort,
        {period_func.format(date_col=date_col)} as activity_period,
        COUNT(DISTINCT d.{user_col}) as users
    FROM data d
    JOIN user_cohorts uc ON d.{user_col} = uc.{user_col}
    GROUP BY uc.first_cohort, {period_func.format(date_col=date_col)}
)
SELECT 
    first_cohort,
    DATE_DIFF('month', first_cohort, activity_period) as period_offset,
    users
FROM cohort_periods
ORDER BY first_cohort, period_offset
"""


# Standalone function
def run_cohort_analysis(
    df: pd.DataFrame,
    date_col: str,
    user_col: str,
    period: str = "month"
) -> Dict[str, Any]:
    """Run cohort analysis on DataFrame directly."""
    return _build_cohort_analysis(df, date_col, user_col, None, period)


__all__ = ["cohort_analysis", "run_cohort_analysis"]
