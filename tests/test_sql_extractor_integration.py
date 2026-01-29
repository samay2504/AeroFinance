"""
Quick Integration Test: LLM + SQLCodeExtractor
Tests that the new SQLCodeExtractor works with real LLM responses.
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply DLL fix first
from app.core.dll_fix import apply_dll_fix
apply_dll_fix()

from app.core.llm_wrapper import get_llm_wrapper
from app.core.llm_utils import SQLCodeExtractor

def test_sql_extraction():
    """Test SQL extraction from LLM response."""
    print("\n--- TEST 1: SQL Generation ---")
    
    llm = get_llm_wrapper()
    print(f"LLM Provider: {llm.provider_name}")
    
    sql_prompt = """Generate a DuckDB SQL query to find all rows where col_0 contains 'revenue'.
Table name is: test_table
Columns are: col_0 (text), col_1 (numeric), col_2 (numeric)

Return ONLY the SQL query, no explanation."""

    sql_response = llm.invoke(sql_prompt)
    content = sql_response.content if hasattr(sql_response, 'content') else str(sql_response)
    print(f"Raw LLM response (first 200 chars): {content[:200]}")

    # Test SQLCodeExtractor
    sql_result = SQLCodeExtractor.extract_sql(content)
    print(f"SQL Extraction Success: {sql_result['success']}")
    
    if sql_result['success']:
        print(f"Extracted SQL: {sql_result['sql'][:100]}...")
        print(f"Extraction Method: {sql_result['extraction_method']}")
    
    return sql_result['success']


def test_python_extraction():
    """Test Python code extraction from LLM response."""
    print("\n--- TEST 2: Python Code Generation ---")
    
    llm = get_llm_wrapper()
    
    python_prompt = """Generate a Python function to calculate the sum of column col_1.
The function should be named 'run' and take a pandas DataFrame 'df' as input.
Return ONLY the Python code."""

    py_response = llm.invoke(python_prompt)
    content = py_response.content if hasattr(py_response, 'content') else str(py_response)
    print(f"Raw LLM response (first 200 chars): {content[:200]}")

    code_result = SQLCodeExtractor.extract_python(content)
    print(f"Python Extraction Success: {code_result['success']}")
    
    if code_result['success']:
        has_run = "def run" in code_result['code']
        print(f"Extracted code contains 'def run': {has_run}")
        print(f"Extraction Method: {code_result['extraction_method']}")
    
    return code_result['success']


def test_structured_output_enhancement():
    """Test structured output with SQL enhancement."""
    print("\n--- TEST 3: Structured Output with Enhancement ---")
    
    llm = get_llm_wrapper()
    
    struct_prompt = """Generate SQL to find total revenue. 
Table: sales_data
Columns: product, revenue, quantity."""

    struct_response = llm.invoke_with_structured_output(
        struct_prompt,
        output_schema={"sql": str, "explanation": str}
    )
    
    print(f"Structured response keys: {list(struct_response.keys())}")
    
    has_sql = "sql" in struct_response and struct_response.get("sql")
    has_error = "error" in struct_response
    
    print(f"Has SQL: {has_sql}")
    print(f"Has Error: {has_error}")
    
    if has_sql:
        sql = struct_response["sql"]
        print(f"SQL: {sql[:80]}..." if len(sql) > 80 else f"SQL: {sql}")
    
    return has_sql or not has_error


if __name__ == "__main__":
    print("=" * 60)
    print("LLM + SQLCodeExtractor Integration Test")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("SQL Extraction", test_sql_extraction()))
    except Exception as e:
        print(f"SQL Extraction failed with error: {e}")
        results.append(("SQL Extraction", False))
    
    try:
        results.append(("Python Extraction", test_python_extraction()))
    except Exception as e:
        print(f"Python Extraction failed with error: {e}")
        results.append(("Python Extraction", False))
    
    try:
        results.append(("Structured Output", test_structured_output_enhancement()))
    except Exception as e:
        print(f"Structured Output failed with error: {e}")
        results.append(("Structured Output", False))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
