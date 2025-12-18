"""
Three-Way Match Tool - Compare Invoice, PO, and Ledger for discrepancies.
Standard CA reconciliation procedure.
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


@tool("three_way_match", return_direct=True)
def three_way_match(
    invoice_dataset_id: str,
    po_dataset_id: str,
    ledger_dataset_id: str,
    match_columns: Optional[Dict[str, str]] = None,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Perform 3-way match between Invoice, Purchase Order, and Ledger.
    
    Args:
        invoice_dataset_id: Dataset ID for invoice data
        po_dataset_id: Dataset ID for purchase order data
        ledger_dataset_id: Dataset ID for ledger/GRN data
        match_columns: Mapping of column names for matching
        tolerance: Amount tolerance for matching (default 1%)
        
    Returns:
        Dict with match status, discrepancies, and explanation
    """
    try:
        from app.core.data_registry import get_data_registry
        registry = get_data_registry()

        # Load datasets
        invoice_df = registry.get(invoice_dataset_id)
        po_df = registry.get(po_dataset_id)
        ledger_df = registry.get(ledger_dataset_id)

        missing = []
        if invoice_df is None:
            missing.append(f"invoice: {invoice_dataset_id}")
        if po_df is None:
            missing.append(f"po: {po_dataset_id}")
        if ledger_df is None:
            missing.append(f"ledger: {ledger_dataset_id}")

        if missing:
            return {
                "result": "error",
                "explain": f"Datasets not found: {', '.join(missing)}"
            }

        # Default column mapping
        if match_columns is None:
            match_columns = {
                "invoice_id": "invoice_id",
                "po_id": "po_id",
                "amount": "amount",
                "quantity": "quantity"
            }

        return _perform_match(
            invoice_df, po_df, ledger_df,
            match_columns, tolerance
        )

    except Exception as e:
        logger.error(f"Three-way match failed: {e}")
        return {
            "result": "error",
            "explain": f"Match failed: {str(e)}"
        }


def _perform_match(
    invoice_df: pd.DataFrame,
    po_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    columns: Dict[str, str],
    tolerance: float
) -> Dict[str, Any]:
    """Perform the actual matching logic."""
    
    discrepancies = []
    matched = 0
    unmatched = 0
    
    # Find common identifier column
    id_col = columns.get("invoice_id", "invoice_id")
    amount_col = columns.get("amount", "amount")
    
    # Try to find matching columns in each DataFrame
    def find_column(df: pd.DataFrame, target: str) -> Optional[str]:
        for col in df.columns:
            if target.lower() in col.lower():
                return col
        return None

    inv_id = find_column(invoice_df, id_col) or find_column(invoice_df, "id")
    inv_amt = find_column(invoice_df, amount_col) or find_column(invoice_df, "total")
    
    po_id = find_column(po_df, "po") or find_column(po_df, id_col)
    po_amt = find_column(po_df, amount_col) or find_column(po_df, "total")
    
    ledger_id = find_column(ledger_df, id_col) or find_column(ledger_df, "ref")
    ledger_amt = find_column(ledger_df, amount_col) or find_column(ledger_df, "debit")

    # If we can't find proper columns, return guidance
    if not all([inv_id, inv_amt]):
        return {
            "result": "error",
            "explain": f"Could not identify invoice columns. Available: {list(invoice_df.columns)}",
            "guidance": "Please specify match_columns parameter"
        }

    # Perform matching
    for idx, inv_row in invoice_df.iterrows():
        inv_id_val = inv_row.get(inv_id)
        inv_amt_val = inv_row.get(inv_amt)

        if pd.isna(inv_id_val) or pd.isna(inv_amt_val):
            continue

        try:
            inv_amt_val = float(inv_amt_val)
        except (ValueError, TypeError):
            continue

        # Find matching PO
        po_match = None
        po_amt_val = None
        if po_id and po_amt:
            po_matches = po_df[po_df[po_id].astype(str) == str(inv_id_val)]
            if not po_matches.empty:
                po_match = po_matches.iloc[0]
                try:
                    po_amt_val = float(po_match[po_amt])
                except (ValueError, TypeError):
                    po_amt_val = None

        # Find matching ledger entry
        ledger_match = None
        ledger_amt_val = None
        if ledger_id and ledger_amt:
            ledger_matches = ledger_df[ledger_df[ledger_id].astype(str) == str(inv_id_val)]
            if not ledger_matches.empty:
                ledger_match = ledger_matches.iloc[0]
                try:
                    ledger_amt_val = float(ledger_match[ledger_amt])
                except (ValueError, TypeError):
                    ledger_amt_val = None

        # Check for discrepancies
        discrepancy = None
        
        if po_amt_val is not None:
            diff = abs(inv_amt_val - po_amt_val)
            if diff > inv_amt_val * tolerance:
                discrepancy = {
                    "type": "invoice_po_mismatch",
                    "id": str(inv_id_val),
                    "invoice_amount": inv_amt_val,
                    "po_amount": po_amt_val,
                    "difference": round(diff, 2)
                }

        if ledger_amt_val is not None:
            diff = abs(inv_amt_val - ledger_amt_val)
            if diff > inv_amt_val * tolerance:
                if discrepancy:
                    discrepancy["ledger_amount"] = ledger_amt_val
                    discrepancy["type"] = "three_way_mismatch"
                else:
                    discrepancy = {
                        "type": "invoice_ledger_mismatch",
                        "id": str(inv_id_val),
                        "invoice_amount": inv_amt_val,
                        "ledger_amount": ledger_amt_val,
                        "difference": round(diff, 2)
                    }

        if discrepancy:
            discrepancies.append(discrepancy)
            unmatched += 1
        else:
            matched += 1

    # Determine overall result
    total = matched + unmatched
    match_rate = matched / max(total, 1)

    if match_rate >= 0.95:
        result = "pass"
        explain = f"High match rate: {matched}/{total} ({match_rate*100:.1f}%)"
    elif match_rate >= 0.8:
        result = "warning"
        explain = f"Moderate match rate: {matched}/{total} ({match_rate*100:.1f}%). Review discrepancies."
    else:
        result = "fail"
        explain = f"Low match rate: {matched}/{total} ({match_rate*100:.1f}%). Significant reconciliation required."

    return {
        "result": result,
        "matched": matched,
        "unmatched": unmatched,
        "total": total,
        "match_rate": round(match_rate * 100, 2),
        "discrepancies": discrepancies[:20],  # Limit output
        "explain": explain
    }


# Standalone function
def run_three_way_match(
    invoice_df: pd.DataFrame,
    po_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    id_col: str = "id",
    amount_col: str = "amount",
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """Run 3-way match on DataFrames directly."""
    return _perform_match(
        invoice_df, po_df, ledger_df,
        {"invoice_id": id_col, "amount": amount_col},
        tolerance
    )


__all__ = ["three_way_match", "run_three_way_match"]
