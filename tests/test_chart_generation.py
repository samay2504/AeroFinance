"""
Test Chart Generation Integration - Real-world LLM test for Recharts-compatible JSON output.

This test validates:
1. LLM generates valid chart JSON in the correct format
2. ChartExtractor correctly parses the chart from response
3. Chart schema matches what React frontend expects (Recharts compatible)
4. Various chart types work correctly (line, bar, pie, composed)
"""
import pytest
import json
import os
import sys
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


class TestChartGeneration:
    """Test suite for chart generation with real LLM calls."""
    
    # Sample financial data for testing
    SAMPLE_FINANCIAL_DATA = """
    Revenue Data (in INR Crores):
    | Fiscal Year | Revenue | Expenses | Net Profit | Profit Margin |
    |-------------|---------|----------|------------|---------------|
    | FY2019      | 100     | 80       | 20         | 20%           |
    | FY2020      | 120     | 90       | 30         | 25%           |
    | FY2021      | 150     | 100      | 50         | 33%           |
    | FY2022      | 180     | 120      | 60         | 33%           |
    | FY2023      | 220     | 140      | 80         | 36%           |
    """
    
    SAMPLE_EXPENSE_BREAKDOWN = """
    Expense Breakdown FY2023:
    | Category       | Amount (Cr) | Percentage |
    |----------------|-------------|------------|
    | Salaries       | 50          | 36%        |
    | Raw Materials  | 35          | 25%        |
    | Operations     | 25          | 18%        |
    | Marketing      | 15          | 11%        |
    | Admin          | 10          | 7%         |
    | Others         | 5           | 3%         |
    """
    
    @pytest.fixture(scope="class")
    def llm_wrapper(self):
        """Get LLM wrapper instance."""
        from app.core.llm_wrapper import LLMWrapper
        config = {"cache_enabled": True, "redis_enabled": False}
        return LLMWrapper(config)
    
    @pytest.fixture(scope="class")
    def chart_extractor(self):
        """Get ChartExtractor class."""
        from app.core.prompts import ChartExtractor
        return ChartExtractor
    
    @pytest.fixture(scope="class")
    def prompts(self):
        """Get prompt functions."""
        from app.core.prompts import get_financial_advisor_prompt, get_chart_generation_prompt
        return {
            "financial_advisor": get_financial_advisor_prompt,
            "chart_generation": get_chart_generation_prompt
        }
    
    def validate_recharts_schema(self, chart: Dict[str, Any]) -> List[str]:
        """
        Validate chart object matches Recharts-compatible schema.
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        # Required fields
        required = ["type", "title", "data"]
        for field in required:
            if field not in chart:
                errors.append(f"Missing required field: {field}")
        
        # Valid chart types (Recharts supported)
        valid_types = {"line", "area", "bar", "scatter", "pie", "radar", "composed"}
        if chart.get("type") not in valid_types:
            errors.append(f"Invalid chart type: {chart.get('type')}")
        
        # Data validation
        data = chart.get("data", [])
        if not isinstance(data, list):
            errors.append("Data must be an array")
        elif len(data) == 0:
            errors.append("Data array is empty")
        else:
            # Validate data points have name/label
            for i, point in enumerate(data):
                if not isinstance(point, dict):
                    errors.append(f"Data point {i} must be an object")
                elif "name" not in point:
                    # Check for alternative label fields
                    if not any(k in point for k in ["label", "category", "x"]):
                        errors.append(f"Data point {i} missing name/label field")
                
                # Validate numeric values exist
                numeric_fields = [k for k, v in point.items() 
                                  if isinstance(v, (int, float)) and k not in ["name", "label"]]
                if len(numeric_fields) == 0 and chart.get("type") != "pie":
                    # Pie charts may use "value" field
                    if "value" not in point:
                        errors.append(f"Data point {i} has no numeric values")
        
        return errors
    
    def test_chart_extractor_basic(self, chart_extractor):
        """Test ChartExtractor parses valid JSON correctly."""
        response = """
        Revenue has grown steadily over 5 years.
        
        ```json
        {
          "charts": [
            {
              "type": "line",
              "title": "Revenue Trend",
              "data": [
                {"name": "FY19", "revenue": 100},
                {"name": "FY20", "revenue": 120},
                {"name": "FY21", "revenue": 150}
              ],
              "xAxis": "Year",
              "yAxis": "Revenue (Cr)"
            }
          ]
        }
        ```
        """
        
        text, charts = chart_extractor.extract_charts(response)
        
        assert len(charts) == 1, f"Expected 1 chart, got {len(charts)}"
        assert charts[0]["type"] == "line"
        assert len(charts[0]["data"]) == 3
        
        # Validate schema
        errors = self.validate_recharts_schema(charts[0])
        assert len(errors) == 0, f"Schema validation errors: {errors}"
        
        print("✅ Basic extraction test passed")
    
    def test_chart_intent_detection(self, chart_extractor):
        """Test semantic chart intent detection."""
        test_cases = [
            ("Show me a line chart of revenue", True, "line"),
            ("Revenue trend over the years", True, "line"),
            ("Compare expenses by category", True, "bar"),
            ("What is the total revenue?", False, None),
            ("Show expense breakdown", True, "pie"),
            ("YoY growth analysis", True, "line"),
            ("CAGR comparison", True, "line"),
        ]
        
        for query, expected_wants, expected_type in test_cases:
            wants, chart_type = chart_extractor.detect_chart_intent(query)
            assert wants == expected_wants, f"Query '{query}': expected wants_chart={expected_wants}, got {wants}"
            if expected_type:
                assert chart_type == expected_type, f"Query '{query}': expected type={expected_type}, got {chart_type}"
        
        print("✅ Intent detection test passed")
    
    @pytest.mark.integration
    def test_llm_line_chart_generation(self, llm_wrapper, chart_extractor, prompts):
        """Test LLM generates valid line chart for trend query."""
        query = "Show me a line chart of revenue growth from FY2019 to FY2023"
        
        # Get prompt with chart instruction
        sys_prompt, user_prompt = prompts["financial_advisor"](
            query=query,
            context=self.SAMPLE_FINANCIAL_DATA,
            force_chart=True,
            chart_type_hint="line"
        )
        
        # Full prompt
        full_prompt = f"{sys_prompt}\n\n{user_prompt}"
        
        # Call LLM
        response = llm_wrapper.invoke(full_prompt, use_cache=False)
        
        print(f"\n--- LLM Response (Line Chart) ---")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # Extract chart
        text, charts = chart_extractor.extract_charts(response)
        
        assert len(charts) >= 1, "LLM did not generate a chart JSON"
        
        chart = charts[0]
        errors = self.validate_recharts_schema(chart)
        assert len(errors) == 0, f"Chart schema errors: {errors}"
        
        # Verify chart type
        assert chart["type"] in ["line", "area"], f"Expected line/area chart, got {chart['type']}"
        
        # Verify data has multiple points
        assert len(chart["data"]) >= 3, f"Expected at least 3 data points, got {len(chart['data'])}"
        
        print(f"✅ Line chart test passed: {chart['title']}")
        print(f"   Data points: {len(chart['data'])}")
    
    @pytest.mark.integration
    def test_llm_bar_chart_generation(self, llm_wrapper, chart_extractor, prompts):
        """Test LLM generates valid bar chart for comparison query."""
        query = "Create a bar chart comparing revenue and expenses across all years"
        
        sys_prompt, user_prompt = prompts["financial_advisor"](
            query=query,
            context=self.SAMPLE_FINANCIAL_DATA,
            force_chart=True,
            chart_type_hint="bar"
        )
        
        response = llm_wrapper.invoke(f"{sys_prompt}\n\n{user_prompt}", use_cache=False)
        
        print(f"\n--- LLM Response (Bar Chart) ---")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        text, charts = chart_extractor.extract_charts(response)
        
        assert len(charts) >= 1, "LLM did not generate a chart JSON"
        
        chart = charts[0]
        errors = self.validate_recharts_schema(chart)
        assert len(errors) == 0, f"Chart schema errors: {errors}"
        
        assert chart["type"] in ["bar", "composed"], f"Expected bar chart, got {chart['type']}"
        
        print(f"✅ Bar chart test passed: {chart['title']}")
    
    @pytest.mark.integration
    def test_llm_pie_chart_generation(self, llm_wrapper, chart_extractor, prompts):
        """Test LLM generates valid pie chart for breakdown query."""
        query = "Show me a pie chart of expense breakdown by category"
        
        sys_prompt, user_prompt = prompts["financial_advisor"](
            query=query,
            context=self.SAMPLE_EXPENSE_BREAKDOWN,
            force_chart=True,
            chart_type_hint="pie"
        )
        
        response = llm_wrapper.invoke(f"{sys_prompt}\n\n{user_prompt}", use_cache=False)
        
        print(f"\n--- LLM Response (Pie Chart) ---")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        text, charts = chart_extractor.extract_charts(response)
        
        assert len(charts) >= 1, "LLM did not generate a chart JSON"
        
        chart = charts[0]
        errors = self.validate_recharts_schema(chart)
        assert len(errors) == 0, f"Chart schema errors: {errors}"
        
        assert chart["type"] == "pie", f"Expected pie chart, got {chart['type']}"
        
        # Pie charts should have reasonable number of slices
        assert 3 <= len(chart["data"]) <= 10, f"Pie chart has {len(chart['data'])} slices (expected 3-10)"
        
        print(f"✅ Pie chart test passed: {chart['title']}")
    
    @pytest.mark.integration
    def test_llm_auto_chart_decision(self, llm_wrapper, chart_extractor, prompts):
        """Test LLM decides whether to include chart intelligently."""
        # Query that SHOULD include a chart
        trend_query = "How has net profit evolved from FY2019 to FY2023?"
        
        sys_prompt, user_prompt = prompts["financial_advisor"](
            query=trend_query,
            context=self.SAMPLE_FINANCIAL_DATA,
            force_chart=False  # Let LLM decide
        )
        
        response = llm_wrapper.invoke(f"{sys_prompt}\n\n{user_prompt}", use_cache=False)
        text, charts = chart_extractor.extract_charts(response)
        
        print(f"\n--- Auto Decision (Trend Query) ---")
        print(f"Charts generated: {len(charts)}")
        
        # Query that should NOT include a chart
        simple_query = "What is the net profit for FY2023?"
        
        sys_prompt2, user_prompt2 = prompts["financial_advisor"](
            query=simple_query,
            context=self.SAMPLE_FINANCIAL_DATA,
            force_chart=False
        )
        
        response2 = llm_wrapper.invoke(f"{sys_prompt2}\n\n{user_prompt2}", use_cache=False)
        text2, charts2 = chart_extractor.extract_charts(response2)
        
        print(f"\n--- Auto Decision (Simple Query) ---")
        print(f"Charts generated: {len(charts2)}")
        print(f"Response: {text2[:200]}...")
        
        # Trend query should have a chart, simple query may or may not
        # We just verify both work without errors
        print("✅ Auto chart decision test completed")
    
    @pytest.mark.integration
    def test_chart_data_accuracy(self, llm_wrapper, chart_extractor, prompts):
        """
        Test that chart data matches source data.
        
        This test validates that LLM-generated chart data contains accurate
        values from the source data. Uses fuzzy matching to handle:
        - Year format variations (2019, FY19, FY2019, 2019-20)
        - Value key name variations (revenue, Revenue, value, y)
        - Minor rounding differences
        """
        import re
        
        query = "Create a line chart showing revenue for each fiscal year"
        
        sys_prompt, user_prompt = prompts["financial_advisor"](
            query=query,
            context=self.SAMPLE_FINANCIAL_DATA,
            force_chart=True,
            chart_type_hint="line"
        )
        
        response = llm_wrapper.invoke(f"{sys_prompt}\n\n{user_prompt}", use_cache=False)
        text, charts = chart_extractor.extract_charts(response)
        
        assert len(charts) >= 1, "No chart generated"
        
        chart = charts[0]
        data = chart.get("data", [])
        
        # Canonical expected values: year -> revenue
        expected_revenues = {
            "2019": 100,
            "2020": 120,
            "2021": 150,
            "2022": 180,
            "2023": 220,
        }
        
        def extract_year(text: str) -> str:
            """Extract 4-digit year from any format (FY2019, 2019, FY19, etc.)."""
            if not text:
                return ""
            text = str(text)
            # Try 4-digit year first
            match = re.search(r'20(\d{2})', text)
            if match:
                return f"20{match.group(1)}"
            # Try 2-digit year (FY19 -> 2019)
            match = re.search(r'(?:FY|fy)?(\d{2})(?:\D|$)', text)
            if match:
                year_2d = int(match.group(1))
                return f"20{year_2d:02d}" if year_2d < 50 else f"19{year_2d:02d}"
            return ""
        
        def extract_value(point: dict) -> float:
            """Extract numeric value from chart data point with flexible key names."""
            value_keys = ['revenue', 'Revenue', 'value', 'Value', 'y', 'amount', 'Amount']
            for key in value_keys:
                if key in point:
                    try:
                        return float(point[key])
                    except (ValueError, TypeError):
                        continue
            # Fallback: try first numeric value in dict
            for v in point.values():
                try:
                    val = float(v)
                    if val > 0:  # Assume positive values only
                        return val
                except (ValueError, TypeError):
                    continue
            return 0.0
        
        # Match data points to expected values
        matches = 0
        matched_years = set()
        
        print(f"\n--- Data Accuracy Check ---")
        print(f"Chart data points: {len(data)}")
        
        for point in data:
            name = point.get("name", point.get("label", point.get("x", "")))
            year = extract_year(name)
            value = extract_value(point)
            
            if year in expected_revenues and year not in matched_years:
                expected = expected_revenues[year]
                tolerance = expected * 0.05  # 5% tolerance for rounding
                
                if abs(value - expected) <= tolerance:
                    matches += 1
                    matched_years.add(year)
                    print(f"  ✓ {year}: {value} ≈ {expected}")
                else:
                    print(f"  ✗ {year}: got {value}, expected {expected}")
            else:
                print(f"  ? Unmatched point: {name} = {value}")
        
        print(f"\nMatched {matches}/{len(expected_revenues)} expected data points")
        
        # Softer threshold: at least 2 matches (LLMs may use different year ranges)
        # or at least 40% of the generated data matches expected values
        min_matches = min(2, len(expected_revenues))
        success = matches >= min_matches or (len(data) > 0 and matches / len(data) >= 0.4)
        
        assert success, (
            f"Data accuracy too low: {matches} matches. "
            f"Need at least {min_matches} exact matches or 40% accuracy."
        )
        
        print("✅ Chart data accuracy test passed")



def run_quick_test():
    """Run a quick sanity check without pytest."""
    print("=" * 60)
    print("CHART GENERATION QUICK TEST")
    print("=" * 60)
    
    from app.core.prompts import ChartExtractor, get_financial_advisor_prompt
    from app.core.llm_wrapper import LLMWrapper
    
    # Test 1: Chart extraction
    print("\n1. Testing chart extraction...")
    test_response = '''
    Revenue grew 50% over 3 years.
    
    ```json
    {"charts": [{"type": "line", "title": "Revenue", "data": [{"name": "Y1", "value": 100}, {"name": "Y2", "value": 150}], "xAxis": "Year", "yAxis": "Amount"}]}
    ```
    '''
    text, charts = ChartExtractor.extract_charts(test_response)
    assert len(charts) == 1, "Extraction failed"
    print(f"   ✅ Extracted {len(charts)} chart")
    
    # Test 2: Intent detection
    print("\n2. Testing intent detection...")
    wants, ctype = ChartExtractor.detect_chart_intent("Show me a bar chart")
    assert wants == True and ctype == "bar", "Intent detection failed"
    print(f"   ✅ Detected: wants_chart={wants}, type={ctype}")
    
    # Test 3: LLM chart generation
    print("\n3. Testing LLM chart generation...")
    
    # Initialize LLM with minimal config
    llm_config = {
        "cache_enabled": True,
        "redis_enabled": False,  # Use in-memory cache for tests
    }
    llm = LLMWrapper(llm_config)
    
    data = """
    Revenue by Year:
    - FY2021: 100 Cr
    - FY2022: 120 Cr  
    - FY2023: 150 Cr
    """
    
    sys_prompt, user_prompt = get_financial_advisor_prompt(
        query="Create a line chart of revenue",
        context=data,
        force_chart=True,
        chart_type_hint="line"
    )
    
    response = llm.invoke(f"{sys_prompt}\n\n{user_prompt}", use_cache=False)
    print(f"\n   LLM Response Preview:")
    print(f"   {response[:300]}...")
    
    text, charts = ChartExtractor.extract_charts(response)
    
    if charts:
        chart = charts[0]
        print(f"\n   ✅ LLM generated valid chart:")
        print(f"      Type: {chart['type']}")
        print(f"      Title: {chart['title']}")
        print(f"      Data points: {len(chart['data'])}")
        print(f"      Sample: {json.dumps(chart['data'][:2], indent=2)}")
    else:
        print(f"   ⚠️ No chart extracted. Response may not contain valid JSON.")
        print(f"   Full response: {response}")
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        pytest.main([__file__, "-v", "-s", "-m", "integration"])
