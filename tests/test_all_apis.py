"""
Comprehensive API Testing Script for AI-CA System
Tests all services with Innovist_MIS_July-2025.xlsx file
"""

import requests
import json
import time

BASE_URL = "http://localhost:9999/v1/api"
CLIENT_ID = "test_innovist"

print("\n" + "="*70)
print("AI-CA API COMPREHENSIVE TESTING")
print("="*70 + "\n")

# Test 1: Health Check
print("TEST 1: Health Check")
print("-" * 70)
try:
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"✓ Status: {r.status_code}")
    data = r.json()["data"]
    print(f"✓ API: {data['api']}")
    print(f"✓ SQL Engine: {data['sql_engine']}")
    print(f"✓ Vector DB: {data['vector_db']}")
    print(f"✓ LLM: {data['llm']}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: List Datasets
print("\nTEST 2: List Datasets")
print("-" * 70)
try:
    r = requests.get(f"{BASE_URL}/datasets?client_id={CLIENT_ID}", timeout=10)
    print(f"✓ Status: {r.status_code}")
    datasets = r.json()["data"]["datasets"]
    print(f"✓ Total datasets: {len(datasets)}")
    for i, ds in enumerate(datasets[:3], 1):
        print(f"  {i}. {ds['dataset_id'].split(':')[-1]}: {ds['rows']} rows, {len(ds['columns'])} cols")
    if len(datasets) > 3:
        print(f"  ... and {len(datasets)-3} more")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Query - Total Revenue
print("\nTEST 3: Query - Total Revenue")
print("-" * 70)
try:
    payload = {
        "client": CLIENT_ID,
        "query": "What is the total revenue for July 2025?",
        "use_cache": True
    }
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=45)
    duration = time.time() - t1
    
    print(f"✓ Status: {r.status_code}")
    print(f"✓ Duration: {duration:.2f}s")
    result = r.json().get("data", {})
    print(f"✓ Success: {result.get('success')}")
    print(f"✓ Method: {result.get('method')}")
    print(f"✓ Query ID: {result.get('query_id')}")
    print(f"✓ Result: {result.get('result', '')[:250]}")
    if result.get('provenance'):
        prov = result['provenance']
        print(f"✓ Provenance: dataset={prov.get('dataset_id', '').split(':')[-1]}, row={prov.get('row_label')}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Query - Expense Analysis
print("\nTEST 4: Query - Expense Analysis")
print("-" * 70)
try:
    payload = {
        "client": CLIENT_ID,
        "query": "What are the major expense categories?",
        "use_cache": True
    }
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=45)
    duration = time.time() - t1
    
    print(f"✓ Status: {r.status_code}")
    print(f"✓ Duration: {duration:.2f}s")
    result = r.json().get("data", {})
    print(f"✓ Success: {result.get('success')}")
    print(f"✓ Method: {result.get('method')}")
    print(f"✓ Result: {result.get('result', '')[:250]}")
    if result.get('provenance'):
        prov = result['provenance']
        print(f"✓ Provenance: {json.dumps(prov, indent=2)}")
    else:
        print(f"✗ No provenance data")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Query - Profit Calculation
print("\nTEST 5: Query - Profit/Loss Calculation")
print("-" * 70)
try:
    payload = {
        "client": CLIENT_ID,
        "query": "Calculate the net profit or loss for this period",
        "use_cache": True
    }
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=45)
    duration = time.time() - t1
    
    print(f"✓ Status: {r.status_code}")
    print(f"✓ Duration: {duration:.2f}s")
    result = r.json().get("data", {})
    print(f"✓ Success: {result.get('success')}")
    print(f"✓ Method: {result.get('method')}")
    print(f"✓ Result: {result.get('result', '')[:250]}")
    if result.get('provenance'):
        prov = result['provenance']
        print(f"✓ Provenance: {json.dumps(prov, indent=2)}")
    else:
        print(f"✗ No provenance data")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Query - Specific Value Lookup
print("\nTEST 6: Query - Specific Account Balance")
print("-" * 70)
try:
    payload = {
        "client": CLIENT_ID,
        "query": "What is the cash balance in the trial balance?",
        "use_cache": True
    }
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=45)
    duration = time.time() - t1
    
    print(f"✓ Status: {r.status_code}")
    print(f"✓ Duration: {duration:.2f}s")
    result = r.json().get("data", {})
    print(f"✓ Success: {result.get('success')}")
    print(f"✓ Method: {result.get('method')}")
    print(f"✓ Result: {result.get('result', '')[:250]}")
    if result.get('provenance'):
        prov = result['provenance']
        print(f"✓ Provenance: {json.dumps(prov, indent=2)}")
    else:
        print(f"✗ No provenance data")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 7: Query - Summary Request
print("\nTEST 7: Query - Dataset Summary")
print("-" * 70)
try:
    payload = {
        "client": CLIENT_ID,
        "query": "Give me a summary of the financial data",
        "use_cache": True
    }
    t1 = time.time()
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=45)
    duration = time.time() - t1
    
    print(f"✓ Status: {r.status_code}")
    print(f"✓ Duration: {duration:.2f}s")
    result = r.json().get("data", {})
    print(f"✓ Success: {result.get('success')}")
    print(f"✓ Method: {result.get('method')}")
    print(f"✓ Result: {result.get('result', '')[:250]}")
    if result.get('provenance'):
        prov = result['provenance']
        print(f"✓ Provenance: {json.dumps(prov, indent=2)}")
    else:
        print(f"✗ No provenance data")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 8: Metrics Endpoint
print("\nTEST 8: System Metrics")
print("-" * 70)
try:
    r = requests.get(f"{BASE_URL}/metrics", timeout=10)
    print(f"✓ Status: {r.status_code}")
    data = r.json()["data"]
    print(f"✓ Total Queries: {data.get('query_stats', {}).get('total_queries', 0)}")
    print(f"✓ Cache Hit Rate: {data.get('llm', {}).get('cache_hit_rate', 0)*100:.1f}%")
    print(f"✓ Avg Response Time: {data.get('query_stats', {}).get('avg_response_time', 0):.3f}s")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 9: Cache Verification (Two Similar Queries)
print("\nTEST 9: Cache Isolation Test")
print("-" * 70)
try:
    # Query 1
    payload1 = {
        "client": CLIENT_ID,
        "query": "What is the total revenue?",
        "use_cache": True
    }
    t1 = time.time()
    r1 = requests.post(f"{BASE_URL}/query", json=payload1, timeout=45)
    dur1 = time.time() - t1
    res1 = r1.json().get("data", {})
    
    print(f"Query 1: 'What is the total revenue?'")
    print(f"  Duration: {dur1:.2f}s")
    print(f"  Result: {res1.get('result', '')[:100]}...")
    print(f"  Query ID: {res1.get('query_id')}")
    if res1.get('provenance'):
        print(f"  Provenance: {json.dumps(res1['provenance'], indent=4)}")
    
    time.sleep(1)
    
    # Query 2 - Different question
    payload2 = {
        "client": CLIENT_ID,
        "query": "What is the total expense?",
        "use_cache": True
    }
    t2 = time.time()
    r2 = requests.post(f"{BASE_URL}/query", json=payload2, timeout=45)
    dur2 = time.time() - t2
    res2 = r2.json().get("data", {})
    
    print(f"\nQuery 2: 'What is the total expense?'")
    print(f"  Duration: {dur2:.2f}s")
    print(f"  Result: {res2.get('result', '')[:100]}...")
    print(f"  Query ID: {res2.get('query_id')}")
    if res2.get('provenance'):
        print(f"  Provenance: {json.dumps(res2['provenance'], indent=4)}")
    
    # Verify different results
    if res1.get('result') != res2.get('result'):
        print(f"\n✓ CACHE ISOLATION: Different queries returned different results")
    else:
        print(f"\n✗ WARNING: Same result returned for different queries")
        
    if res1.get('query_id') != res2.get('query_id'):
        print(f"✓ UNIQUE QUERY IDS: Each query got unique ID")
    else:
        print(f"✗ WARNING: Same query ID used")
        
except Exception as e:
    print(f"✗ Error: {e}")

# Summary
print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print("\nAll services tested:")
print("  ✓ Health Check Service")
print("  ✓ Datasets Service")
print("  ✓ Query Service (Multiple query types)")
print("  ✓ Metrics Service")
print("  ✓ Cache Isolation Verification")
print("\n")
