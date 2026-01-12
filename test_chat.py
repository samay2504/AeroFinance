"""
Interactive Chat Client for the Re AI Chartered Accountant System
Allows real-world testing of the data analysis pipeline.
"""
import sys
import os

# Set environment for Windows DLL loading (MUST BE BEFORE ANY OTHER IMPORTS)
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Windows DLL path fix for torch/spacy
try:
    from app.core.dll_fix import apply_dll_fix
    apply_dll_fix()
except ImportError:
    pass


from pathlib import Path
import json
import time
import logging
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
try:
    from app.agents.data_analyst import DataAnalystAgent
    from app.core.llm_wrapper import LLMWrapper, get_llm_wrapper
    from app.agents.router import RouterAgent, get_router_agent, TRACK_DATA, TRACK_DOC, TRACK_WEB
    from app.rag.ingest import DocumentIngestor, get_document_ingestor
    from app.core.id_generator import normalize_client_id, generate_session_id
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Re directory")
    sys.exit(1)


# ==============================================================================
# HUMAN-LIKE RESPONSE FORMATTER - GitHub Flavored Markdown
# ==============================================================================
def format_human_response(
    query: str,
    result: Any,
    value: Optional[float] = None,
    source: str = "",
    method: str = "",
    explanation: str = "",
    elapsed: float = 0.0
) -> str:
    """
    Generate human-like, conversational responses in GitHub Flavored Markdown.
    
    Transforms raw values into natural language sentences with proper formatting.
    """
    import re
    
    query_lower = query.lower()
    result_str = str(result)
    
    # Detect query intent for response styling
    is_count_query = any(kw in query_lower for kw in ['how many', 'count', 'number of'])
    is_value_query = any(kw in query_lower for kw in ['what is', 'what are', 'what was', 'show', 'find', 'get'])
    is_total_query = any(kw in query_lower for kw in ['total', 'sum', 'aggregate'])
    is_growth_query = any(kw in query_lower for kw in ['growth', 'change', 'increase', 'decrease', '%'])
    is_list_query = any(kw in query_lower for kw in ['list', 'all', 'names', 'sheets'])
    is_comparison_query = any(kw in query_lower for kw in ['compare', 'vs', 'versus', 'difference'])
    
    # Format numeric values nicely
    def format_number(val):
        if val is None:
            return None
        try:
            f = float(val)
            if abs(f) >= 1e9:
                return f"{f/1e9:,.2f} Billion"
            elif abs(f) >= 1e6:
                return f"{f/1e6:,.2f} Million"
            elif abs(f) >= 1e3:
                return f"{f:,.2f}"
            elif abs(f) < 0.01 and f != 0:
                return f"{f:.4f}"
            else:
                return f"{f:,.2f}"
        except (ValueError, TypeError):
            return str(val)
    
    # Build the response
    response_parts = []
    
    # Main answer with context
    if value is not None:
        formatted_value = format_number(value)
        
        if is_growth_query:
            if '%' not in formatted_value:
                response_parts.append(f"📊 **The growth rate is {formatted_value}%**")
            else:
                response_parts.append(f"📊 **The growth rate is {formatted_value}**")
        elif is_total_query:
            response_parts.append(f"📊 **The total is {formatted_value}**")
        elif is_count_query:
            response_parts.append(f"📊 **There are {formatted_value} items**")
        else:
            response_parts.append(f"📊 **The value is {formatted_value}**")
    elif is_list_query and isinstance(result, (list, str)):
        if isinstance(result, list):
            items = result[:20]  # Limit to 20 items
            if len(items) > 5:
                response_parts.append(f"📋 **Found {len(result)} items:**\n")
                response_parts.append("| # | Item |")
                response_parts.append("|---|------|")
                for i, item in enumerate(items, 1):
                    response_parts.append(f"| {i} | {item} |")
                if len(result) > 20:
                    response_parts.append(f"\n*...and {len(result) - 20} more*")
            else:
                response_parts.append(f"📋 **Found {len(result)} items:** {', '.join(str(x) for x in items)}")
        else:
            response_parts.append(f"📋 {result_str}")
    else:
        # Generic result formatting
        if len(result_str) > 500:
            # Long result - format as code block
            response_parts.append(f"📊 **Analysis Result:**\n\n```\n{result_str[:1500]}\n```")
            if len(result_str) > 1500:
                response_parts.append("\n*...output truncated*")
        else:
            response_parts.append(f"📊 {result_str}")
    
    # Add contextual explanation
    if explanation and explanation not in result_str:
        response_parts.append(f"\n\n💡 *{explanation}*")
    
    # Add metadata in collapsed details (GFM compatible)
    response_parts.append("\n\n---")
    response_parts.append(f"\n<details><summary>📁 Source Details</summary>\n")
    response_parts.append(f"\n- **Dataset:** `{source}`")
    response_parts.append(f"\n- **Method:** `{method}`")
    response_parts.append(f"\n- **Response time:** {elapsed:.2f}s")
    response_parts.append(f"\n</details>")
    
    return "\n".join(response_parts)


# ==============================================================================
# CONVERSATION MEMORY - Sliding window with session management
# ==============================================================================
class ConversationMemory:
    """
    Production-grade conversation memory with:
    - Sliding window (keeps last K messages)
    - Session/document ID isolation
    - LRU cache for memory efficiency
    - Context summarization for long conversations
    """
    def __init__(self, window_size: int = 10, max_sessions: int = 100):
        self.window_size = window_size
        self.max_sessions = max_sessions
        self._sessions: Dict[str, list] = {}
        self._access_order: list = []  # LRU tracking
        
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._access_order.append(session_id)
        
        # LRU eviction if too many sessions
        if len(self._sessions) > self.max_sessions:
            oldest = self._access_order.pop(0)
            del self._sessions[oldest]
        
        # Update access order
        if session_id in self._access_order:
            self._access_order.remove(session_id)
        self._access_order.append(session_id)
        
        # Add message with sliding window
        self._sessions[session_id].append({
            "role": role,
            "content": content[:1000],  # Truncate to save memory
            "timestamp": time.time()
        })
        
        # Keep only last window_size messages
        if len(self._sessions[session_id]) > self.window_size:
            self._sessions[session_id] = self._sessions[session_id][-self.window_size:]
    
    def get_history(self, session_id: str, last_n: int = None) -> list:
        """Get conversation history for a session."""
        if session_id not in self._sessions:
            return []
        history = self._sessions[session_id]
        if last_n:
            return history[-last_n:]
        return history
    
    def get_context_string(self, session_id: str, last_n: int = 5) -> str:
        """Get conversation context as formatted string for LLM."""
        history = self.get_history(session_id, last_n)
        if not history:
            return ""
        
        context_parts = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content'][:300]}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._access_order.remove(session_id)
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get session statistics."""
        if session_id not in self._sessions:
            return {"exists": False}
        return {
            "exists": True,
            "message_count": len(self._sessions[session_id]),
            "window_size": self.window_size
        }

# ==============================================================================
# GLOBAL STATE
# ==============================================================================
loaded_files: Dict[str, Any] = {}
data_analyst: Optional[DataAnalystAgent] = None
router: Optional[RouterAgent] = None
doc_ingestor: Optional[DocumentIngestor] = None
conversation_memory: Optional[ConversationMemory] = None
current_session_id: str = "default"
current_client_id: str = "test_client"


def initialize_components():
    """Initialize all required components."""
    global data_analyst, router, doc_ingestor, conversation_memory, current_session_id, current_client_id
    
    print("\n⏳ Initializing AI components...")
    
    try:
        # Generate session and normalize client
        current_session_id = generate_session_id()
        current_client_id = normalize_client_id("Test Client")
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
        
        # Initialize Conversation Memory
        conversation_memory = ConversationMemory(window_size=10, max_sessions=100)
        print("  ✅ Conversation Memory initialized (window=10)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        return False


def load_file(file_path: str) -> bool:
    """Load an Excel, CSV, or JSON file for analysis. Also handles pasted JSON text."""
    global loaded_files, data_analyst
    
    # Check if this is JSON text pasted directly (not a file path)
    stripped = file_path.strip()
    if stripped.startswith('{') or stripped.startswith('['):
        return load_json_text(stripped)
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # INGESTION CACHING: Check if file already loaded
    file_key = str(path.resolve())
    already_loaded = [k for k, v in loaded_files.items() if v.get('path') == file_key]
    if already_loaded:
        print(f"✅ File already loaded ({len(already_loaded)} sheets). Use 'list' to see datasets.")
        return True
    
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
                
                # Detect and apply header row
                df = _detect_and_apply_header(df)
                    
                # Generate df_id
                safe_name = path.stem.replace(' ', '_').replace('-', '_').lower()
                safe_sheet = sheet_name.replace(' ', '_').replace('-', '_').lower()
                df_id = f"{safe_name}:{safe_sheet}"
                
                # Register with data analyst
                # Register with data analyst
                data_analyst.register_dataframe(df_id, df, client_id=current_client_id)
                loaded_files[df_id] = {
                    'path': file_key,
                    'sheet': sheet_name,
                    'rows': len(df),
                    'cols': len(df.columns)
                }
                print(f"    ✅ Loaded sheet '{sheet_name}' as '{df_id}' ({len(df)} rows)")
                
        elif path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
            df_id = path.stem.replace(' ', '_').replace('-', '_').lower()
            data_analyst.register_dataframe(df_id, df, client_id=current_client_id)
            loaded_files[df_id] = {
                'path': file_key,
                'rows': len(df),
                'cols': len(df.columns)
            }
            print(f"  ✅ Loaded '{df_id}' ({len(df)} rows, {len(df.columns)} columns)")
            
        elif path.suffix.lower() == '.json':
            # JSON file support
            with open(path, 'r', encoding='utf-8') as f:
                return load_json_text(f.read(), source_name=path.stem)
        else:
            print(f"❌ Unsupported file type: {path.suffix}")
            return False
        
        # Inform router that data is now available with context
        if router and loaded_files:
            datasets_info = ", ".join([f.split(':')[-1] for f in loaded_files.keys()])
            router.set_data_context(True, f"Loaded datasets: {datasets_info}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False


def load_json_text(json_text: str, source_name: str = "json_data") -> bool:
    """Load JSON text (pasted or from file) as DataFrame with RAG support."""
    global loaded_files, data_analyst, doc_ingestor
    
    try:
        from app.ingest.json_ingest import JSONIngestor
        from app.rag.ingest import get_rag_pipeline
        
        # Initialize JSON ingestor
        json_ingestor = JSONIngestor()
        
        # Parse JSON first
        data, error = json_ingestor.parse_json_text(json_text)
        if error:
            print(f"❌ Error parsing JSON: {error}")
            return False
        
        # Get RAG pipeline (may be None if unavailable)
        rag_pipeline = None
        try:
            rag_pipeline = get_rag_pipeline()
        except:
            pass
        
        # Define callback to register DataFrames with data analyst
        def register_df(dataset_id, df, metadata):
            data_analyst.register_dataframe(dataset_id, df, client_id=current_client_id)
            loaded_files[dataset_id] = {
                'rows': len(df),
                'cols': len(df.columns),
                'source': 'json'
            }
            print(f"  ✅ Loaded JSON table '{metadata.get('table', dataset_id.split(':')[-1])}' as '{dataset_id}' ({len(df)} rows)")
        
        # Use ingest_to_rag if RAG available, otherwise regular ingest
        if rag_pipeline and rag_pipeline.is_available:
            result = json_ingestor.ingest_to_rag(
                data=data,
                source_name=source_name,
                client_id=current_client_id,
                rag_pipeline=rag_pipeline
            )
            
            # Also register DataFrames for SQL queries
            regular_result = json_ingestor.ingest_json(
                data=data,
                source_name=source_name,
                client_id=current_client_id,
                register_callback=register_df
            )
            
            if result.get("rag_chunks", 0) > 0:
                print(f"  🔍 Indexed {result['rag_chunks']} chunks in vector storage")
        else:
            # Regular ingestion without RAG
            result = json_ingestor.ingest_json(
                data=data,
                source_name=source_name,
                client_id=current_client_id,
                register_callback=register_df
            )
        
        if result.get("success") and result.get("datasets"):
            if router:
                router.set_data_context(True)
            return True
        else:
            print(f"❌ JSON ingestion failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False


def _detect_and_apply_header(df: pd.DataFrame) -> pd.DataFrame:
    """Detect header row in DataFrame and apply it."""
    if df.empty or len(df) < 2:
        return df
    
    # Check first few rows for header patterns
    for row_idx in range(min(5, len(df))):
        row = df.iloc[row_idx]
        # Count non-null string values
        str_count = sum(1 for v in row if isinstance(v, str) and len(str(v).strip()) > 1)
        # Count 'Unnamed' or empty
        unnamed_count = sum(1 for v in row if pd.isna(v) or 'unnamed' in str(v).lower())
        
        if str_count > len(row) * 0.5 and unnamed_count < len(row) * 0.3:
            # This looks like a header row
            new_df = df.iloc[row_idx + 1:].copy()
            new_df.columns = [str(v) if pd.notna(v) else f"Col_{i}" for i, v in enumerate(row)]
            new_df.reset_index(drop=True, inplace=True)
            return new_df
    
    return df


def ingest_document(file_path: str) -> bool:
    """Ingest a document for RAG search."""
    global doc_ingestor
    
    if not doc_ingestor:
        print("❌ Document ingestor not available.")
        return False
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\n📄 Ingesting document: {path.name}")
    
    try:
        # Read file content based on type
        text = ""
        
        if path.suffix.lower() == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        elif path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except ImportError:
                print("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
                return False
        
        elif path.suffix.lower() in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(path)
                text = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                print("❌ python-docx not installed. Install with: pip install python-docx")
                return False
        
        else:
            print(f"❌ Unsupported file type: {path.suffix}")
            return False
        
        if not text.strip():
            print("❌ No text content found in document")
            return False
        
        # Ingest into RAG
        doc_id = path.stem.replace(' ', '_').replace('-', '_').lower()
        
        result = doc_ingestor.ingest_text(
            text=text,
            client_id=current_client_id,
            dataset_id=doc_id,
            metadata={"filename": path.name, "path": str(path)}
        )
        
        if result.get("success"):
            chunks = result.get("chunks_created", 0)
            print(f"  ✅ Ingested '{doc_id}' ({chunks} chunks created)")
            print(f"  💡 You can now ask questions about this document!")
            return True
        else:
            error = result.get("error", "Unknown error")
            print(f"  ❌ Ingestion failed: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Error ingesting document: {e}")
        return False


def ask_question(query: str, force_track: str = None) -> str:
    """Process a user question through the AI pipeline with conversation context."""
    global data_analyst, router, loaded_files, doc_ingestor, conversation_memory, current_session_id
    
    if not query.strip():
        return ""
    
    start_time = time.time()
    
    # Add user query to conversation memory
    if conversation_memory:
        conversation_memory.add_message(current_session_id, "user", query)
    
    try:
        query_lower = query.lower()
        
        # ==================================================================
        # HANDLE METADATA QUERIES DIRECTLY (sheet names, column info, etc.)
        # ==================================================================
        if loaded_files:
            # Sheet names query
            if any(kw in query_lower for kw in ['sheet name', 'sheet names', 'names of sheet', 'what are the sheets']):
                sheet_names = [info.get('sheet', df_id.split(':')[-1]) for df_id, info in loaded_files.items()]
                response = f"📋 There are {len(loaded_files)} sheets:\n"
                for i, name in enumerate(sheet_names, 1):
                    response += f"  {i}. {name}\n"
                if conversation_memory:
                    conversation_memory.add_message(current_session_id, "assistant", response)
                return response + f"\n  ⏱️ Time: {time.time() - start_time:.2f}s"
        
        # ==================================================================
        # SMART ROUTING WITH DATA PRIORITY
        # ==================================================================
        if force_track:
            track = force_track
            confidence = 1.0
            print(f"  🎯 Forced routing to: {force_track}")
        else:
            # Data-related keywords - route to DATA track
            data_keywords = [
                'sheet', 'column', 'row', 'data', 'file', 'excel', 'csv', 'table',
                'value', 'total', 'sum', 'average', 'count', 'max', 'min',
                'nifty', 'stock', 'price', 'volume', 'gain', 'loss', 'market',
                'revenue', 'profit', 'expense', 'cost', 'growth', 'percent',
                'this', 'loaded', 'show', 'list', 'top', 'bottom', 'first', 'last',
                'many', 'how many', 'what is', 'what are', 'which', 'where'
            ]
            
            has_data_keyword = any(kw in query_lower for kw in data_keywords)
            
            route_result = router.route(query)
            track = route_result.get('track', TRACK_DATA)
            confidence = route_result.get('confidence', 0.5)
            
            # Override to DATA if data is loaded and query seems data-related
            if loaded_files and has_data_keyword and track != TRACK_DATA:
                track = TRACK_DATA
                confidence = 0.85
                print(f"  📝 Redirected to data analysis (data keywords detected)")
            
            # Also override for context references
            refers_to_loaded_data = any(phrase in query_lower for phrase in [
                'this data', 'this file', 'this document', 'this sheet', 'this excel',
                'loaded data', 'loaded file', 'the data', 'the file', 'my data',
                'the sheet', 'these sheets', 'this table'
            ])
            
            if track == TRACK_DOC and loaded_files and refers_to_loaded_data:
                track = TRACK_DATA
                confidence = 0.85
        
        track_display = {
            TRACK_DATA: "📊 Data Analysis",
            TRACK_DOC: "📄 Document Search",
            TRACK_WEB: "🌐 Web Search"
        }.get(track, track)
        
        if not force_track:
            print(f"\n  🎯 Routed to: {track_display} (confidence: {confidence:.0%})")
        
        # Get conversation context for LLM-based methods
        context_str = ""
        if conversation_memory:
            context_str = conversation_memory.get_context_string(current_session_id, last_n=3)
        
        if track == TRACK_DATA:
            # Data analysis track
            if not loaded_files:
                return "⚠️ No data files loaded. Use 'load <filepath>' to load data first."
            
            # KEYWORD-BASED DATASET MATCHING (Priority)
            # Check if query explicitly mentions a dataset name
            query_lower = query.lower()
            keyword_match = None
            
            keyword_map = {
                'balance sheet': 'balance_sheet',
                'balance': 'balance_sheet',
                'assets': 'balance_sheet',
                'liabilities': 'balance_sheet',
                'equity': 'balance_sheet',
                'income statement': 'income_statement',
                'income': 'income_statement',
                'revenue': 'income_statement',
                'profit': 'income_statement',
                'net income': 'income_statement',
                'cogs': 'income_statement',
                'company': 'company_meta',
                'company name': 'company_meta',
                'ticker': 'company_meta',
                'employees': 'company_meta',
            }
            
            for keyword, table_suffix in keyword_map.items():
                if keyword in query_lower:
                    # Find a loaded dataset that ends with this suffix
                    for df_id in loaded_files.keys():
                        if df_id.endswith(table_suffix):
                            keyword_match = df_id
                            break
                    if keyword_match:
                        break
            
            # Use keyword match if found, otherwise use smart dataset matching
            datasets = [{"dataset_id": df_id, **info} for df_id, info in loaded_files.items()]
            matched_df_id = keyword_match or data_analyst.match_dataset_by_query(query, datasets)
            
            # If no smart match, try all datasets to find best result
            best_result = None
            best_df_id = None
            
            if matched_df_id:
                print(f"  📂 Matched dataset: {matched_df_id}")
                result = data_analyst.execute_sql_query(query, matched_df_id)
                if result.success:
                    best_result = result
                    best_df_id = matched_df_id
            
            # If matched dataset failed or no match, try all datasets
            if not best_result or not best_result.success:
                print(f"  🔄 Trying all {len(loaded_files)} datasets...")
                for df_id in loaded_files.keys():
                    if df_id == matched_df_id:
                        continue  # Already tried
                    result = data_analyst.execute_sql_query(query, df_id)
                    if result.success:
                        # Prefer results with actual values
                        if best_result is None:
                            best_result = result
                            best_df_id = df_id
                        elif hasattr(result, 'value') and result.value is not None:
                            if not hasattr(best_result, 'value') or best_result.value is None:
                                best_result = result
                                best_df_id = df_id
                        # Prefer non-error methods
                        elif 'error' not in result.method.lower():
                            if 'error' in best_result.method.lower():
                                best_result = result
                                best_df_id = df_id
            
            elapsed = time.time() - start_time
            
            if best_result and best_result.success:
                # Generate human-like markdown response
                response = format_human_response(
                    query=query,
                    result=best_result.result,
                    value=getattr(best_result, 'value', None),
                    source=best_df_id,
                    method=best_result.method,
                    explanation=best_result.explanation,
                    elapsed=elapsed
                )
                return response
            else:
                # ================================================================
                # MULTI-STRATEGY FALLBACK: Try different approaches
                # ================================================================
                
                # STRATEGY 1: RAG Semantic Search (for conceptual/descriptive queries)
                try:
                    from app.rag.ingest import get_rag_pipeline
                    rag = get_rag_pipeline()
                    if rag and rag.is_available:
                        print("  🔍 Trying RAG semantic search...")
                        rag_results = rag._ingestor.search(
                            query=query,
                            client_id=current_client_id,
                            top_k=3,
                            score_threshold=0.3
                        )
                        if rag_results and len(rag_results) > 0:
                            # Use LLM to synthesize answer from RAG results
                            from app.core.llm_wrapper import get_llm_wrapper
                            llm = get_llm_wrapper()
                            
                            context = "\n\n".join([
                                f"Source: {r.get('metadata', {}).get('table_name', 'data')}\n{r.get('content', '')[:800]}"
                                for r in rag_results[:3]
                            ])
                            
                            prompt = f"""Based on this data context, answer the user's question.

DATA CONTEXT:
{context}

QUESTION: {query}

Provide a clear, accurate answer. If the data contains numbers, include them. 
If asked for totals/sums, calculate them from the provided data."""
                            
                            answer = llm.invoke(prompt)
                            elapsed = time.time() - start_time
                            return f"📊 {answer}\n\n  📁 Source: RAG semantic search\n  🔧 Method: rag_llm\n  ⏱️ Time: {elapsed:.2f}s"
                except Exception as rag_error:
                    logger.debug(f"RAG fallback failed: {rag_error}")
                
                # STRATEGY 2: Direct DataFrame Analysis with LLM
                try:
                    from app.core.llm_wrapper import get_llm_wrapper
                    llm = get_llm_wrapper()
                    
                    # Collect data from ALL relevant datasets
                    all_data_context = []
                    for df_id in list(loaded_files.keys())[:5]:  # Limit to 5
                        df = data_analyst._get_dataframe(df_id)
                        if df is not None:
                            # Create rich context with stats
                            table_name = df_id.split(':')[-1]
                            context_parts = [f"\n--- TABLE: {table_name} ---"]
                            context_parts.append(f"Columns: {list(df.columns)}")
                            context_parts.append(f"Data:\n{df.to_string()}")
                            
                            # Add numeric summaries
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                context_parts.append("\nNumeric Summaries:")
                                for col in numeric_cols:
                                    total = df[col].sum()
                                    avg = df[col].mean()
                                    context_parts.append(f"  {col}: Total={total:,.2f}, Avg={avg:,.2f}")
                            
                            all_data_context.append("\n".join(context_parts))
                    
                    if all_data_context:
                        full_context = "\n".join(all_data_context)[:6000]
                        prompt = f"""You are a data analyst. Analyze ALL the data below and answer the question accurately.

{full_context}

QUESTION: {query}

Instructions:
- If asking for totals/sums, calculate from the data
- If asking about specific metrics, find and report the exact values
- If asking for a summary, describe key insights from the data
- Be specific and include numbers where relevant
- If the question asks about a specific table (balance sheet, income statement, etc.), focus on that data"""
                        
                        answer = llm.invoke(prompt)
                        elapsed = time.time() - start_time
                        return f"📊 {answer}\n\n  📁 Source: Multi-table analysis\n  🔧 Method: llm_comprehensive\n  ⏱️ Time: {elapsed:.2f}s"
                except Exception as llm_error:
                    logger.debug(f"LLM fallback failed: {llm_error}")
                
                error_msg = best_result.error if best_result else "No results from any dataset"
                return f"⚠️ Query failed: {error_msg}\n  📂 Tried {len(loaded_files)} datasets\n  ⏱️ Time: {elapsed:.2f}s"
                
        elif track == TRACK_WEB:
            # Web search track
            try:
                # Import implementation directly, not the @tool decorated version
                from app.tools.web_search import _web_search_impl as web_search_impl
                from app.core.llm_wrapper import get_llm_wrapper
                
                print("  🔍 Searching the web...")
                search_result = web_search_impl(query, num_results=3)
                
                if search_result.get("result") == "success":
                    results = search_result.get("results", [])
                    
                    # Synthesize answer using LLM
                    llm = get_llm_wrapper()
                    snippets = "\n\n".join([
                        f"**{r['title']}**\n{r['snippet']}\nSource: {r['url']}"
                        for r in results[:3]
                    ])
                    
                    prompt = f"""Based on the following web search results, answer this question: {query}

Search Results:
{snippets}

Provide a concise, accurate answer based on the search results."""
                    
                    answer = llm.invoke(prompt)
                    elapsed = time.time() - start_time
                    
                    response = f"🌐 {answer}"
                    response += f"\n\n  📚 Sources:"
                    for r in results[:3]:
                        response += f"\n    • {r['title']}: {r['url']}"
                    response += f"\n  ⏱️ Time: {elapsed:.2f}s"
                    return response
                else:
                    return "⚠️ Web search returned no results."
                    
            except ImportError:
                return "⚠️ Web search not available. Install required dependencies."
            except Exception as e:
                return f"❌ Web search error: {e}"
            
        elif track == TRACK_DOC:
            # Document RAG track
            if not doc_ingestor:
                return "⚠️ Document ingestor not available."
            
            try:
                from app.core.llm_wrapper import get_llm_wrapper
                
                print("  🔍 Searching documents...")
                
                # Search for relevant documents
                # Use first loaded file's client_id or default
                client_id = "test_client"
                docs = doc_ingestor.search(query, client_id, top_k=5, score_threshold=0.3)
                
                if not docs:
                    return "📄 No relevant documents found. Try loading some documents first."
                
                # Synthesize answer using LLM
                llm = get_llm_wrapper()
                context = "\n\n".join([
                    f"Document {i+1}:\n{d.get('content', '')[:500]}"
                    for i, d in enumerate(docs[:3])
                ])
                
                prompt = f"""Based on the following document excerpts, answer this question: {query}

Context:
{context}

Provide a comprehensive answer based on the documents."""
                
                answer = llm.invoke(prompt)
                elapsed = time.time() - start_time
                
                response = f"📄 {answer}"
                response += f"\n\n  📚 Found {len(docs)} relevant documents"
                response += f"\n  ⏱️ Time: {elapsed:.2f}s"
                return response
                
            except Exception as e:
                return f"❌ Document search error: {e}"
        
        return "⚠️ Unknown route track"
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error: {e}\n\nDetails:\n{error_details[:500]}"


def show_help():
    """Display help information."""
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    🤖 Re AI-CA Test Chat - Full Pipeline               ║
╠════════════════════════════════════════════════════════════════════════╣
║ DATA COMMANDS:                                                         ║
║   load <filepath>       - Load an Excel/CSV file for analysis          ║
║   list                  - Show loaded datasets                         ║
║   summary               - Show summary of all loaded data              ║
║   sample <dataset>      - Show sample rows from a dataset              ║
║   columns <dataset>     - Show columns and types of a dataset          ║
║   clear                 - Clear all loaded datasets                    ║
║                                                                        ║
║ DOCUMENT COMMANDS:                                                     ║
║   ingest <filepath>     - Ingest PDF/DOCX/TXT for RAG search           ║
║                                                                        ║
║ MEMORY COMMANDS:                                                       ║
║   history               - Show conversation history                    ║
║   clearhistory          - Clear conversation history                   ║
║                                                                        ║
║ FORCE ROUTING (prefix your query with):                                ║
║   @data <query>         - Force data analysis track                    ║
║   @web <query>          - Force web search track                       ║
║   @doc <query>          - Force document search track                  ║
║                                                                        ║
║ EXAMPLES:                                                              ║
║   load "D:\\Data\\NSE-DATA.xlsx"                                       ║
║   summary                                                              ║
║   sample nifty                                                         ║
║   What are the top 5 volume gainers?                                   ║
║   What are the sheet names?                                            ║
║   @web What is the current repo rate in India?                         ║
╚════════════════════════════════════════════════════════════════════════╝
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
                data_analyst.dataframes.clear()
                router.clear_cache()
                router.set_data_context(False)  # No data loaded anymore
                print("\n🗑️ All datasets cleared.")
                continue
            
            elif cmd_lower.startswith('load '):
                file_path = user_input[5:].strip().strip('"').strip("'")
                load_file(file_path)
                continue

            elif user_input.startswith(('{', '[')):
                # Auto-detect pasted JSON data - handle multi-line input
                print("  📋 Detected JSON data, collecting...")
                json_buffer = user_input
                
                # Count braces to detect complete JSON
                def is_json_complete(s):
                    opens = s.count('{') + s.count('[')
                    closes = s.count('}') + s.count(']')
                    return opens > 0 and opens == closes
                
                # Keep reading until JSON is complete
                while not is_json_complete(json_buffer):
                    try:
                        more = input("  ... ")
                        json_buffer += more
                    except EOFError:
                        break
                
                print(f"  📋 Collected {len(json_buffer)} characters of JSON")
                load_file(json_buffer)
                continue
            
            elif cmd_lower.startswith('ingest '):
                file_path = user_input[7:].strip().strip('"').strip("'")
                ingest_document(file_path)
                continue
            
            elif cmd_lower == 'summary':
                # Show summary of all loaded data
                if not loaded_files:
                    print("\n📂 No data loaded. Use 'load <filepath>' first.")
                    continue
                print("\n📊 Data Summary:")
                for df_id, info in loaded_files.items():
                    df = data_analyst._get_dataframe(df_id)
                    if df is not None:
                        print(f"\n  📁 {df_id}")
                        print(f"     Rows: {len(df)}, Columns: {len(df.columns)}")
                        
                        # Check if columns are integer indices (no header) or named
                        first_col = df.columns[0] if len(df.columns) > 0 else None
                        if isinstance(first_col, int):
                            # Try to detect header from first row
                            first_row = df.iloc[0].tolist() if len(df) > 0 else []
                            header_preview = [str(v)[:20] for v in first_row[:5] if pd.notna(v)]
                            print(f"     Header (row 0): {header_preview}{'...' if len(first_row) > 5 else ''}")
                        else:
                            col_names = [str(c)[:20] for c in list(df.columns[:7])]
                            print(f"     Columns: {col_names}{'...' if len(df.columns) > 7 else ''}")
                        
                        # Show numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        print(f"     Numeric columns: {len(numeric_cols)}")
                continue
            
            elif cmd_lower.startswith('sample '):
                # Show sample of specific dataset
                dataset_part = user_input[7:].strip()
                found = False
                for df_id in loaded_files.keys():
                    if dataset_part.lower() in df_id.lower():
                        df = data_analyst._get_dataframe(df_id)
                        if df is not None:
                            print(f"\n📊 Sample from {df_id}:")
                            print(df.head(10).to_string())
                            found = True
                            break
                if not found:
                    print(f"⚠️ Dataset matching '{dataset_part}' not found. Use 'list' to see available datasets.")
                continue
            
            elif cmd_lower.startswith('columns '):
                # Show columns of specific dataset
                dataset_part = user_input[8:].strip()
                found = False
                for df_id in loaded_files.keys():
                    if dataset_part.lower() in df_id.lower():
                        df = data_analyst._get_dataframe(df_id)
                        if df is not None:
                            print(f"\n📋 Columns in {df_id}:")
                            for i, col in enumerate(df.columns):
                                dtype = df[col].dtype
                                non_null = df[col].count()
                                print(f"  {i+1}. {col} ({dtype}, {non_null} values)")
                            found = True
                            break
                if not found:
                    print(f"⚠️ Dataset matching '{dataset_part}' not found.")
                continue
            
            elif cmd_lower == 'history':
                # Show conversation history
                if conversation_memory:
                    history = conversation_memory.get_history(current_session_id)
                    if history:
                        print(f"\n📜 Conversation History ({len(history)} messages):")
                        print("─" * 50)
                        for msg in history:
                            role = "👤 You" if msg["role"] == "user" else "🤖 AI"
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"{role}: {content}")
                        print("─" * 50)
                    else:
                        print("\n📜 No conversation history yet.")
                else:
                    print("\n⚠️ Conversation memory not initialized.")
                continue
            
            elif cmd_lower == 'clearhistory':
                # Clear conversation history
                if conversation_memory:
                    conversation_memory.clear_session(current_session_id)
                    print("\n🗑️ Conversation history cleared.")
                continue
            
            # Check for force-routing prefixes
            force_track = None
            actual_query = user_input
            if user_input.lower().startswith('@data '):
                force_track = TRACK_DATA
                actual_query = user_input[6:].strip()
            elif user_input.lower().startswith('@web '):
                force_track = TRACK_WEB
                actual_query = user_input[5:].strip()
            elif user_input.lower().startswith('@doc '):
                force_track = TRACK_DOC
                actual_query = user_input[5:].strip()
            
            # Regular question
            print("\n⏳ AI is thinking...")
            response = ask_question(actual_query, force_track=force_track)
            
            # Save response to conversation memory
            if conversation_memory and response:
                conversation_memory.add_message(current_session_id, "assistant", response)
            
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
