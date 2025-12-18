"""Test FY22 aggregation through the full agent"""
import sys
sys.path.insert(0, '.')
import os

# Set environment for testing
os.environ['DATA_DIR'] = 'data'

from app.agents.data_analyst import DataAnalystAgent, get_data_analyst_agent
from app.core.llm_wrapper import get_llm_wrapper
import pandas as pd
from pathlib import Path

# Set up LLM wrapper
config = {
    "provider_preference": ["groq", "ollama"],
    "cache_enabled": True
}
llm = get_llm_wrapper(config)

# Get agent
agent = get_data_analyst_agent(llm)

# Load FY22 data
parquet_path = Path('data/dataframe_cache/test_client_ca_mis__report_fy22.parquet')
df = pd.read_parquet(parquet_path)
agent.register_dataframe("test_client_ca:mis__report:fy22", df)

# Run the query - using the correct method signature (query, df_id, user_id)
query = "What is the total 'Prepaid recorded lectures | Domestic' collection for the entire FY22 period?"
print(f"Query: {query}")
print()

result = agent.execute_sql_query(query, "test_client_ca:mis__report:fy22")

print(f"Success: {result.success}")
print(f"Method: {result.method}")
print(f"Result: {result.result}")
print(f"Value: {result.value}")
print(f"Explanation: {result.explanation}")

# Expected: ~276.32
if result.value:
    expected = 276.32
    diff = abs(float(result.value) - expected)
    if diff < 1.0:
        print(f"✅ PASS - Got {result.value}, expected ~{expected}")
    else:
        print(f"❌ FAIL - Got {result.value}, expected ~{expected}")


