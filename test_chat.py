"""
Interactive Chat Client for the Re AI Chartered Accountant System
Allows real-world testing of the data analysis pipeline.
"""
import sys
import os

# Set environment for Windows DLL loading (MUST BE BEFORE ANY OTHER IMPORTS)
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Add torch DLL directory to PATH before importing torch
    torch_lib = r'd:\Projects2.0\Valuenaire\.conda\Lib\site-packages\torch\lib'
    if os.path.exists(torch_lib):
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
    
    # Pre-import torch to ensure DLLs load correctly
    try:
        import torch  # noqa
    except Exception:
        pass

from pathlib import Path
import json
import time
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
try:
    from app.agents.data_analyst import DataAnalystAgent
    from app.core.llm_wrapper import LLMWrapper, get_llm_wrapper
    from app.agents.router import RouterAgent, get_router_agent, TRACK_DATA, TRACK_DOC, TRACK_WEB
    from app.rag.ingest import DocumentIngestor, get_document_ingestor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Re directory")
    sys.exit(1)

# Global state
loaded_files: Dict[str, Any] = {}
data_analyst: Optional[DataAnalystAgent] = None
router: Optional[RouterAgent] = None
doc_ingestor: Optional[DocumentIngestor] = None


def initialize_components():
    """Initialize all required components."""
    global data_analyst, router, doc_ingestor
    
    print("\n⏳ Initializing AI components...")
    
    try:
        # Initialize LLM
        llm = get_llm_wrapper()
        print(f"  ✅ LLM initialized ({llm.provider_name})")
        
        # Initialize Router
        router = get_router_agent(llm)
        print("  ✅ Router initialized")
        
        # Initialize Data Analyst with LLM
        data_analyst = DataAnalystAgent(llm_wrapper=llm)
        print("  ✅ Data Analyst initialized")
        
        # Initialize RAG (optional)
        try:
            doc_ingestor = get_document_ingestor()
            print("  ✅ Document Ingestor initialized")
        except Exception as e:
            print(f"  ⚠️ Document Ingestor unavailable: {e}")
            doc_ingestor = None
        
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        return False


def load_file(file_path: str) -> bool:
    """Load an Excel or CSV file for analysis."""
    global loaded_files, data_analyst
    
    import pandas as pd
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\n📂 Loading: {path.name}")
    
    try:
        if path.suffix.lower() in ['.xlsx', '.xls']:
            # Excel file - load all sheets
            xl = pd.ExcelFile(path)
            sheet_names = xl.sheet_names
            print(f"  📄 Found {len(sheet_names)} sheets: {', '.join(sheet_names[:5])}{'...' if len(sheet_names) > 5 else ''}")
            
            for sheet_name in sheet_names:
                df = pd.read_excel(path, sheet_name=sheet_name, header=None)
                if df.empty:
                    continue
                    
                # Generate df_id
                safe_name = path.stem.replace(' ', '_').replace('-', '_').lower()
                safe_sheet = sheet_name.replace(' ', '_').replace('-', '_').lower()
                df_id = f"{safe_name}:{safe_sheet}"
                
                # Register with data analyst
                data_analyst.register_dataframe(df_id, df)
                loaded_files[df_id] = {
                    'path': str(path),
                    'sheet': sheet_name,
                    'rows': len(df),
                    'cols': len(df.columns)
                }
                print(f"    ✅ Loaded sheet '{sheet_name}' as '{df_id}' ({len(df)} rows)")
                
        elif path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
            df_id = path.stem.replace(' ', '_').replace('-', '_').lower()
            data_analyst.register_dataframe(df_id, df)
            loaded_files[df_id] = {
                'path': str(path),
                'rows': len(df),
                'cols': len(df.columns)
            }
            print(f"  ✅ Loaded '{df_id}' ({len(df)} rows, {len(df.columns)} columns)")
        else:
            print(f"❌ Unsupported file type: {path.suffix}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False


def ask_question(query: str) -> str:
    """Process a user question through the AI pipeline."""
    global data_analyst, router, loaded_files
    
    if not query.strip():
        return ""
    
    start_time = time.time()
    
    try:
        # Route the query
        route_result = router.route(query)
        track = route_result.get('track', TRACK_DATA)
        confidence = route_result.get('confidence', 0.5)
        
        track_display = {
            TRACK_DATA: "📊 Data Analysis",
            TRACK_DOC: "📄 Document Search",
            TRACK_WEB: "🌐 Web Search"
        }.get(track, track)
        
        print(f"\n  🎯 Routed to: {track_display} (confidence: {confidence:.0%})")
        
        if track == TRACK_DATA:
            # Try each loaded dataset
            if not loaded_files:
                return "⚠️ No data files loaded. Use 'load <filepath>' to load data first."
            
            # Try to find the best matching dataset
            best_result = None
            best_df_id = None
            
            for df_id in loaded_files.keys():
                result = data_analyst.execute_sql_query(query, df_id)
                if result.success:
                    if best_result is None or (hasattr(result, 'value') and result.value is not None):
                        best_result = result
                        best_df_id = df_id
            
            if best_result and best_result.success:
                elapsed = time.time() - start_time
                
                # Format the result
                result_str = str(best_result.result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                
                response = f"📊 {result_str}"
                response += f"\n\n  📁 Source: {best_df_id}"
                response += f"\n  🔧 Method: {best_result.method}"
                if best_result.explanation:
                    response += f"\n  💡 {best_result.explanation}"
                response += f"\n  ⏱️ Time: {elapsed:.2f}s"
                return response
            else:
                return f"⚠️ Could not process query. Tried {len(loaded_files)} datasets."
                
        elif track == TRACK_WEB:
            # Web search (placeholder - would need web search implementation)
            return "🌐 Web search not implemented in this test client. Use the full API for web queries."
            
        elif track == TRACK_DOC:
            # RAG search
            if doc_ingestor:
                return "📄 Document search not implemented in this test client. Use the full API for document queries."
            else:
                return "⚠️ RAG is not available. Load documents first."
        
        return "⚠️ Unknown route track"
        
    except Exception as e:
        return f"❌ Error: {e}"


def show_help():
    """Display help information."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                    🤖 Re AI-CA Test Chat                           ║
╠════════════════════════════════════════════════════════════════════╣
║ COMMANDS:                                                          ║
║   load <filepath>   - Load an Excel/CSV file for analysis          ║
║   list              - Show loaded datasets                         ║
║   clear             - Clear loaded datasets                        ║
║   help              - Show this help message                       ║
║   exit / quit       - Exit the chat                                ║
║                                                                    ║
║ EXAMPLES:                                                          ║
║   load "D:\\Data\\MIS- report.xlsx"                                ║
║   What is the total revenue for FY22?                              ║
║   Calculate the growth from FY21 to FY22                           ║
║   Show me the digital marketing expenses                           ║
╚════════════════════════════════════════════════════════════════════╝
""")


def list_datasets():
    """List all loaded datasets."""
    if not loaded_files:
        print("\n📂 No datasets loaded. Use 'load <filepath>' to load data.")
        return
    
    print(f"\n📂 Loaded Datasets ({len(loaded_files)}):")
    print("─" * 60)
    for df_id, info in loaded_files.items():
        sheet = info.get('sheet', '')
        rows = info.get('rows', 0)
        cols = info.get('cols', 0)
        print(f"  • {df_id}")
        if sheet:
            print(f"    Sheet: {sheet}")
        print(f"    Size: {rows} rows × {cols} columns")
    print("─" * 60)


def chat():
    """Main interactive chat loop."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║             🤖 Re AI Chartered Accountant - Test Chat              ║
║        Interactive testing for the data analysis pipeline          ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    if not initialize_components():
        return
    
    show_help()
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            cmd_lower = user_input.lower()
            
            if cmd_lower in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif cmd_lower == 'help':
                show_help()
                continue
            
            elif cmd_lower == 'list':
                list_datasets()
                continue
            
            elif cmd_lower == 'clear':
                loaded_files.clear()
                data_analyst._registered_datasets.clear()
                router.clear_cache()
                print("\n🗑️ All datasets cleared.")
                continue
            
            elif cmd_lower.startswith('load '):
                file_path = user_input[5:].strip().strip('"').strip("'")
                load_file(file_path)
                continue
            
            # Regular question
            print("\n⏳ AI is thinking...")
            response = ask_question(user_input)
            print(f"\n🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def quick_test():
    """Run a quick test with sample data."""
    print("\n🧪 Running quick test...")
    
    if not initialize_components():
        return False
    
    # Try to load sample file if exists
    sample_files = [
        r"D:\Projects2.0\Valuenaire\MIS- report.xlsx",
        r"D:\Projects2.0\Valuenaire\Innovist MIS July 2025.xlsx",
    ]
    
    for sample_path in sample_files:
        if Path(sample_path).exists():
            load_file(sample_path)
    
    if not loaded_files:
        print("⚠️ No sample files found for testing.")
        return False
    
    # Test queries
    test_queries = [
        "What is the total revenue for FY22?",
        "How many sheets does this file have?",
        "Show me the gross margin",
    ]
    
    print("\n" + "=" * 60)
    print("Running test queries...")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        response = ask_question(query)
        print(f"💬 Response: {response[:200]}...")
    
    print("\n✅ Quick test complete!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re AI-CA Test Chat")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with sample data")
    parser.add_argument("--load", type=str, help="Load a file at startup")
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        if args.load:
            if initialize_components():
                load_file(args.load)
        chat()
