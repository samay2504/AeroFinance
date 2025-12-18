"""
Benford's Law Test Tool - Detect anomalies in numeric data.
First-digit analysis for fraud detection.
"""
import logging
from typing import Dict, Any, List, Optional
import math

logger = logging.getLogger(__name__)

try:
    from langchain.tools import tool
except ImportError:
    def tool(name: str = None, return_direct: bool = False):
        def decorator(func):
            func.name = name or func.__name__
            return func
        return decorator

# Benford's Law expected distribution
BENFORD_EXPECTED = {
    1: 0.301,
    2: 0.176,
    3: 0.125,
    4: 0.097,
    5: 0.079,
    6: 0.067,
    7: 0.058,
    8: 0.051,
    9: 0.046,
}


def _get_first_digit(value: float) -> Optional[int]:
    """Extract first significant digit from a number."""
    if value == 0:
        return None
    value = abs(value)
    while value < 1:
        value *= 10
    while value >= 10:
        value /= 10
    return int(value)


def _calculate_benford_distribution(values: List[float]) -> Dict[int, float]:
    """Calculate first-digit distribution of values."""
    digit_counts = {d: 0 for d in range(1, 10)}
    total = 0

    for val in values:
        try:
            digit = _get_first_digit(float(val))
            if digit and 1 <= digit <= 9:
                digit_counts[digit] += 1
                total += 1
        except (ValueError, TypeError):
            continue

    if total == 0:
        return {d: 0.0 for d in range(1, 10)}

    return {d: count / total for d, count in digit_counts.items()}


def _chi_square_test(observed: Dict[int, float], expected: Dict[int, float], n: int) -> float:
    """Calculate chi-square statistic."""
    chi_sq = 0.0
    for digit in range(1, 10):
        obs = observed.get(digit, 0) * n
        exp = expected.get(digit, 0) * n
        if exp > 0:
            chi_sq += ((obs - exp) ** 2) / exp
    return chi_sq


@tool("benford_test", return_direct=True)
def benford_test(
    dataset_id: str,
    column_name: str,
    threshold: float = 0.02
) -> Dict[str, Any]:
    """
    Run Benford's Law test on a numeric column to detect anomalies.
    
    Args:
        dataset_id: ID of the dataset to analyze
        column_name: Name of numeric column to test
        threshold: Deviation threshold for flagging (default 0.02 = 2%)
        
    Returns:
        Dict with result (pass/fail), deviations, and explanation
    """
    try:
        # Get data registry
        from app.core.data_registry import get_data_registry
        registry = get_data_registry()
        
        df = registry.get(dataset_id)
        if df is None:
            return {
                "result": "error",
                "explain": f"Dataset not found: {dataset_id}"
            }

        if column_name not in df.columns:
            # Try fuzzy match
            matches = [c for c in df.columns if column_name.lower() in c.lower()]
            if matches:
                column_name = matches[0]
            else:
                return {
                    "result": "error",
                    "explain": f"Column not found: {column_name}"
                }

        # Extract numeric values
        values = df[column_name].dropna().tolist()
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue

        if len(numeric_values) < 100:
            return {
                "result": "warning",
                "explain": f"Insufficient data for Benford test ({len(numeric_values)} values, need 100+)",
                "sample_size": len(numeric_values)
            }

        # Calculate observed distribution
        observed = _calculate_benford_distribution(numeric_values)
        
        # Calculate deviations
        deviations = []
        for digit in range(1, 10):
            obs = observed.get(digit, 0)
            exp = BENFORD_EXPECTED.get(digit, 0)
            deviation = obs - exp
            
            if abs(deviation) > threshold:
                deviations.append({
                    "digit": digit,
                    "observed": round(obs * 100, 2),
                    "expected": round(exp * 100, 2),
                    "deviation_pct": round(deviation * 100, 2)
                })

        # Chi-square test
        chi_sq = _chi_square_test(observed, BENFORD_EXPECTED, len(numeric_values))
        # Critical value for 8 degrees of freedom at 95% confidence
        critical_value = 15.51

        # Determine result
        if chi_sq > critical_value or len(deviations) > 2:
            result = "fail"
            explain = f"Significant deviation from Benford's Law detected. Chi-square: {chi_sq:.2f} (critical: {critical_value})"
        elif deviations:
            result = "warning"
            explain = f"Minor deviations detected. Chi-square: {chi_sq:.2f}"
        else:
            result = "pass"
            explain = f"Data follows Benford's Law. Chi-square: {chi_sq:.2f}"

        return {
            "result": result,
            "chi_square": round(chi_sq, 2),
            "critical_value": critical_value,
            "deviations": deviations,
            "observed_distribution": {str(k): round(v * 100, 2) for k, v in observed.items()},
            "sample_size": len(numeric_values),
            "explain": explain
        }

    except Exception as e:
        logger.error(f"Benford test failed: {e}")
        return {
            "result": "error",
            "explain": f"Test failed: {str(e)}"
        }


# Standalone function for non-LangChain usage
def run_benford_test(values: List[float], threshold: float = 0.02) -> Dict[str, Any]:
    """Run Benford test on a list of values."""
    if len(values) < 100:
        return {
            "result": "warning",
            "explain": f"Insufficient data ({len(values)} values)",
            "sample_size": len(values)
        }

    observed = _calculate_benford_distribution(values)
    chi_sq = _chi_square_test(observed, BENFORD_EXPECTED, len(values))
    
    deviations = []
    for digit in range(1, 10):
        dev = observed.get(digit, 0) - BENFORD_EXPECTED.get(digit, 0)
        if abs(dev) > threshold:
            deviations.append({
                "digit": digit,
                "deviation": round(dev * 100, 2)
            })

    return {
        "result": "fail" if chi_sq > 15.51 else "pass",
        "chi_square": round(chi_sq, 2),
        "deviations": deviations,
        "sample_size": len(values)
    }


__all__ = ["benford_test", "run_benford_test"]
