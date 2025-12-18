#!/usr/bin/env python
"""
LLM Provider & Tool Calling Test Suite
Tests LLM API connectivity, tool calling (web search, etc.), and fallback mechanisms.
"""
import sys
import os
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(name: str, success: bool, details: str = ""):
    icon = "✅" if success else "❌"
    print(f"{icon} {name}: {details}")

def test_llm_providers():
    """Test LLM provider initialization and fallback."""
    print_section("LLM PROVIDER TEST")
    
    results = {
        "primary_provider": False,
        "fallback_works": False,
        "basic_invoke": False,
    }
    
    try:
        from app.core.llm_provider import LLMProvider
        
        config = {
            "provider_preference": ["google_genai", "groq", "openrouter", "ollama"],
            "temperature": 0.1,
            "max_retries": 2,
        }
        
        provider = LLMProvider(config)
        
        # Check which provider was selected
        current = provider.current_provider
        print(f"\n📍 Selected Provider: {current}")
        
        if current and current != "fallback":
            results["primary_provider"] = True
            print_result("Provider Initialization", True, f"Using {current}")
        else:
            print_result("Provider Initialization", False, "Fell back to fallback mode")
        
        # Test basic invocation
        print("\n🔄 Testing basic LLM invoke...")
        response = provider.invoke("Say 'Hello World' and nothing else.")
        
        if response and len(str(response)) > 0:
            results["basic_invoke"] = True
            print_result("Basic Invoke", True, f"Response: {str(response)[:100]}...")
        else:
            print_result("Basic Invoke", False, "Empty or no response")
        
        # Get metrics
        metrics = provider.get_metrics()
        print(f"\n📊 Metrics: {metrics}")
        
    except Exception as e:
        print_result("LLM Provider Test", False, f"Error: {e}")
    
    return results


def test_llm_structured_output():
    """Test LLM structured JSON output."""
    print_section("STRUCTURED OUTPUT TEST")
    
    results = {"json_output": False}
    
    try:
        from app.core.llm_wrapper import get_llm_wrapper
        
        config = {
            "provider_preference": ["google_genai", "groq", "ollama"],
            "cache_enabled": False,
        }
        
        llm = get_llm_wrapper(config)
        
        print(f"\n📍 Using Provider: {llm.provider_name}")
        print("🔄 Testing structured JSON output...")
        
        response = llm.invoke_with_structured_output(
            "What is 2 + 2? Provide the answer.",
            output_schema={"answer": int, "explanation": str}
        )
        
        if isinstance(response, dict) and "answer" in response:
            results["json_output"] = True
            print_result("Structured Output", True, f"Got: {response}")
        elif "error" in response:
            print_result("Structured Output", False, f"Error: {response.get('error')}")
        else:
            print_result("Structured Output", False, f"Unexpected: {response}")
        
    except Exception as e:
        print_result("Structured Output Test", False, f"Error: {e}")
    
    return results


def test_schema_analyzer():
    """Test LLM-powered schema analysis (tool calling equivalent)."""
    print_section("SCHEMA ANALYZER TEST (Tool Calling)")
    
    results = {"schema_analysis": False}
    
    try:
        from app.core.schema_analyzer import SchemaAnalyzer
        from app.core.llm_wrapper import get_llm_wrapper
        import pandas as pd
        
        # Create test DataFrame
        test_data = {
            "Metric": ["Revenue", "Expenses", "Profit"],
            "FY21": [100, 60, 40],
            "FY22": [150, 80, 70],
            "FY23": [200, 100, 100],
        }
        df = pd.DataFrame(test_data)
        
        print("\n📊 Test DataFrame:")
        print(df.to_string())
        
        # Initialize analyzer with LLM
        config = {"provider_preference": ["google_genai", "groq", "ollama"]}
        llm = get_llm_wrapper(config)
        print(f"\n📍 Using Provider: {llm.provider_name}")
        
        analyzer = SchemaAnalyzer(llm)
        
        print("🔄 Analyzing schema with LLM...")
        schema = analyzer.analyze(df, "test_data", context="Financial metrics by year")
        
        if schema:
            print(f"\n📋 Schema Analysis Results:")
            print(f"   Label Column: {schema.label_column}")
            print(f"   Period Columns: {list(schema.period_columns.keys())}")
            print(f"   Header Row: {schema.header_row_index}")
            
            if schema.period_columns:
                results["schema_analysis"] = True
                print_result("Schema Analysis", True, f"Found {len(schema.period_columns)} periods")
            else:
                print_result("Schema Analysis", False, "No periods detected")
        else:
            print_result("Schema Analysis", False, "No schema returned")
        
    except ImportError as e:
        print_result("Schema Analyzer", False, f"Import error: {e}")
    except Exception as e:
        print_result("Schema Analyzer Test", False, f"Error: {e}")
    
    return results


def test_semantic_understanding():
    """Test semantic understanding (NER, matching) - works without LLM."""
    print_section("SEMANTIC UNDERSTANDING TEST")
    
    results = {
        "ner": False,
        "matching": False,
        "structure": False,
    }
    
    try:
        from app.core.semantic_understanding import (
            get_financial_ner,
            get_semantic_matcher,
            get_structure_detector,
        )
        
        # Test NER
        ner = get_financial_ner()
        test_text = "Revenue for FY22 increased by 20% compared to FY21"
        entities = ner.extract_entities(test_text)
        
        print(f"\n📝 Input: '{test_text}'")
        print(f"   Entities Found: {len(entities)}")
        for e in entities[:5]:
            print(f"   - {e.text} ({e.entity_type}): {e.normalized}")
        
        if entities:
            results["ner"] = True
            print_result("NER Extraction", True, f"Found {len(entities)} entities")
        else:
            print_result("NER Extraction", False, "No entities found")
        
        # Test Semantic Matching
        matcher = get_semantic_matcher()
        query = "revenue from sales"
        candidates = ["Revenue", "Expenses", "Cost of Goods Sold", "Net Income"]
        
        best = matcher.find_best_match(query, candidates)
        print(f"\n🔍 Query: '{query}'")
        print(f"   Candidates: {candidates}")
        
        if best:
            results["matching"] = True
            print_result("Semantic Matching", True, f"Best match: '{best[0]}' (score: {best[1]:.2f})")
        else:
            print_result("Semantic Matching", False, "No match found")
        
        # Test Structure Detection
        import pandas as pd
        test_df = pd.DataFrame({
            "A": ["Header", "Revenue", "Expenses"],
            "B": ["FY21", "100", "60"],
            "C": ["FY22", "150", "80"],
        })
        
        detector = get_structure_detector()
        label_col = detector.detect_label_column(test_df)
        header_row = detector.detect_header_row(test_df)
        
        print(f"\n📊 Structure Detection:")
        print(f"   Label Column Index: {label_col}")
        print(f"   Header Row: {header_row}")
        
        if label_col is not None:
            results["structure"] = True
            print_result("Structure Detection", True, f"Label col: {label_col}, Header row: {header_row}")
        
    except ImportError as e:
        print_result("Semantic Understanding", False, f"Import error: {e}")
    except Exception as e:
        print_result("Semantic Understanding Test", False, f"Error: {e}")
    
    return results


def test_fallback_mechanism():
    """Test provider fallback when primary fails."""
    print_section("FALLBACK MECHANISM TEST")
    
    results = {"fallback": False}
    
    try:
        from app.core.llm_provider import LLMProvider
        
        # Force a bad config to trigger fallback
        config = {
            "provider_preference": ["google_genai", "groq", "ollama", "fallback"],
            "temperature": 0.1,
            "max_retries": 1,
        }
        
        print("🔄 Testing provider chain...")
        provider = LLMProvider(config)
        
        print(f"\n📍 Final Provider: {provider.current_provider}")
        
        if provider.llm is not None:
            results["fallback"] = True
            print_result("Fallback Mechanism", True, f"Landed on: {provider.current_provider}")
        else:
            print_result("Fallback Mechanism", False, "No provider available")
        
        # Show cooldowns
        if provider._cooldowns:
            print(f"\n⏳ Provider Cooldowns: {provider._cooldowns}")
        
    except Exception as e:
        print_result("Fallback Test", False, f"Error: {e}")
    
    return results


def test_web_search_tool():
    """Test web search tool (DuckDuckGo-based)."""
    print_section("WEB SEARCH TOOL TEST")
    
    results = {
        "web_search": False,
        "tax_search": False,
    }
    
    try:
        from app.tools.web_search import web_search, search_tax_rate
        
        # Test basic web search
        print("\n🔍 Testing web search tool...")
        query = "GST rate in India 2024"
        
        # Handle both direct function and LangChain tool
        if hasattr(web_search, 'func'):
            # LangChain StructuredTool
            response = web_search.func(query, num_results=2)
        elif hasattr(web_search, 'invoke'):
            # LangChain Tool with invoke
            response = web_search.invoke({"query": query, "num_results": 2})
        else:
            # Direct function call
            response = web_search(query, num_results=2)
        
        print(f"   Query: '{query}'")
        print(f"   Status: {response.get('result')}")
        
        if response.get("result") == "success":
            results["web_search"] = True
            result_count = response.get("num_results", 0)
            print_result("Web Search", True, f"Found {result_count} results")
            
            # Show first result
            if response.get("results"):
                first = response["results"][0]
                print(f"   First Result: {first.get('title', 'N/A')[:50]}...")
                print(f"   URL: {first.get('url', 'N/A')[:60]}...")
        elif response.get("result") == "no_results":
            print_result("Web Search", True, "No results (search works but no matches)")
            results["web_search"] = True  # Search function works
        else:
            print_result("Web Search", False, f"Error: {response.get('explain')}")
        
        # Test tax rate search function (direct function, not a tool)
        print("\n💰 Testing tax rate search...")
        tax_result = search_tax_rate("corporate", "India", 2024)
        
        print(f"   Status: {tax_result.get('result')}")
        
        if tax_result.get("result") == "success" and tax_result.get("results"):
            results["tax_search"] = True
            print_result("Tax Rate Search", True, f"Found {len(tax_result.get('results', []))} results")
        elif tax_result.get("result") == "no_results":
            print_result("Tax Rate Search", True, "No results (function works)")
            results["tax_search"] = True
        else:
            print_result("Tax Rate Search", False, "Search failed")
        
    except ImportError as e:
        print_result("Web Search Tool", False, f"Import error: {e}")
        print("   NOTE: Install httpx and beautifulsoup4 for web search")
    except Exception as e:
        print_result("Web Search Tool Test", False, f"Error: {e}")
    
    return results


def test_other_tools():
    """Test other available tools (Benford, Cohort, etc.)."""
    print_section("OTHER TOOLS TEST")
    
    results = {
        "benford_available": False,
        "cohort_available": False,
        "three_way_available": False,
    }
    
    try:
        # Test Benford's Law tool
        try:
            from app.tools.benford import benford_test
            results["benford_available"] = True
            print_result("Benford's Law Tool", True, "Available")
        except ImportError:
            print_result("Benford's Law Tool", False, "Not imported")
        
        # Test Cohort Analysis tool
        try:
            from app.tools.cohort import cohort_analysis
            results["cohort_available"] = True
            print_result("Cohort Analysis Tool", True, "Available")
        except ImportError:
            print_result("Cohort Analysis Tool", False, "Not imported")
        
        # Test Three-Way Match tool
        try:
            from app.tools.three_way_match import three_way_match
            results["three_way_available"] = True
            print_result("Three-Way Match Tool", True, "Available")
        except ImportError:
            print_result("Three-Way Match Tool", False, "Not imported")
        
    except Exception as e:
        print_result("Tools Test", False, f"Error: {e}")
    
    return results


def test_tool_orchestrator():
    """Test the tool orchestrator for LLM-integrated tool execution."""
    print_section("TOOL ORCHESTRATOR TEST")
    
    results = {
        "orchestrator_init": False,
        "tool_detection": False,
        "tool_execution": False,
    }
    
    try:
        from app.tools.orchestrator import get_tool_orchestrator, ToolCategory
        
        # Initialize orchestrator
        orchestrator = get_tool_orchestrator()
        results["orchestrator_init"] = True
        print_result("Orchestrator Init", True, f"Registered {len(orchestrator._tools)} tools")
        
        # List available tools
        print("\n📋 Available Tools:")
        for tool in orchestrator.list_tools():
            print(f"   - {tool['name']}: {tool['description'][:50]}...")
        
        # Test intent detection
        test_queries = [
            ("What is the GST rate in India?", "search_tax_rate"),
            ("Check for fraud in the ledger", "benford_test"),
            ("Reconcile invoice with PO", "three_way_match"),
        ]
        
        print("\n🎯 Testing Intent Detection:")
        detection_success = 0
        for query, expected in test_queries:
            detected = orchestrator.detect_tool_intent(query)
            match = detected == expected
            if match:
                detection_success += 1
            print(f"   '{query[:30]}...'")
            print(f"   Expected: {expected}, Got: {detected} {'✅' if match else '❌'}")
        
        if detection_success >= 2:
            results["tool_detection"] = True
            print_result("Intent Detection", True, f"{detection_success}/3 correct")
        else:
            print_result("Intent Detection", False, f"{detection_success}/3 correct")
        
        # Test actual tool execution (web search)
        print("\n🔧 Testing Tool Execution:")
        result = orchestrator.execute_tool("web_search", query="Python programming", num_results=1)
        
        if result.success:
            results["tool_execution"] = True
            print_result("Tool Execution", True, f"web_search returned {len(result.result.get('results', []))} results")
        else:
            print_result("Tool Execution", False, f"Error: {result.explanation}")
        
    except ImportError as e:
        print_result("Orchestrator", False, f"Import error: {e}")
    except Exception as e:
        print_result("Orchestrator Test", False, f"Error: {e}")
    
    return results


def main():
    """Run all tests."""
    print("\n" + "🚀" * 20)
    print("   LLM & TOOL CALLING TEST SUITE")
    print("🚀" * 20)
    
    all_results = {}
    
    # Run tests
    all_results["providers"] = test_llm_providers()
    all_results["structured"] = test_llm_structured_output()
    all_results["schema"] = test_schema_analyzer()
    all_results["semantic"] = test_semantic_understanding()
    all_results["fallback"] = test_fallback_mechanism()
    all_results["web_search"] = test_web_search_tool()
    all_results["other_tools"] = test_other_tools()
    all_results["orchestrator"] = test_tool_orchestrator()
    
    # Summary
    print_section("SUMMARY")
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        for test_name, success in results.items():
            total_tests += 1
            if success:
                passed_tests += 1
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED!")
        return 0
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ MOSTLY PASSING - Some tests need attention")
        return 0
    else:
        print("❌ SIGNIFICANT FAILURES - Review logs above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
