# Re AI-CA Test Chat - Full Feature Guide

## Overview
The `test_chat.py` provides a comprehensive interactive testing interface for the Re AI Chartered Accountant system with full pipeline access.

## Features

### 1. **Data Analysis** 📊
- Load Excel/CSV files with automatic sheet detection
- Smart dataset matching based on query context
- SQL-first analytical pipeline with LLM fallback
- Support for:
  - Numeric queries (revenue, growth, totals)
  - Period-based analysis (FY21, 9MFY22, etc.)
  - Metadata queries (sheet count, column names)
  - Complex calculations (variance, growth rates)

### 2. **Document Search (RAG)** 📄
- Ingest PDF, DOCX, and TXT documents
- Semantic search using vector embeddings
- LLM-powered answer synthesis from document context
- Multi-tenant support with client isolation

### 3. **Web Search** 🌐
- Real-time web search integration
- LLM synthesis of search results
- Source attribution and citations

### 4. **Intelligent Routing**
- Automatic query classification
- Confidence-based routing to appropriate track
- Fallback mechanisms for edge cases

## Commands

```
load <filepath>     - Load Excel/CSV for analysis
ingest <filepath>   - Ingest document for RAG
list                - Show loaded datasets
clear               - Clear all data
help                - Show help
exit/quit           - Exit chat
```

## Usage Examples

### Data Analysis
```
💬 You: load "D:\Data\MIS-report.xlsx"
💬 You: What is the total revenue for FY22?
💬 You: Calculate growth from FY21 to 9MFY22
💬 You: How many sheets are in this file?
```

### Document Search
```
💬 You: ingest "D:\Docs\annual_report.pdf"
💬 You: What is the company's accounting policy?
💬 You: Explain the revenue recognition method
```

### Web Search
```
💬 You: What is the current GST rate in India?
💬 You: Latest RBI monetary policy updates
```

## Technical Details

### Pipeline Flow
1. **Query Input** → Router analyzes query intent
2. **Route Selection** → TRACK_DATA, TRACK_DOC, or TRACK_WEB
3. **Execution** → Appropriate agent processes query
4. **Response** → Formatted result with metadata

### Data Analysis Methods
- Template SQL (deterministic patterns)
- Semantic Pandas (heuristic matching)
- LLM SQL generation
- LLM Python code generation
- PandasAI (natural language)

### Fallback Chain
```
Template → Heuristic → LLM SQL → LLM Python → PandasAI
```

## Configuration

### Environment Variables
```
GROQ_API_KEY=your_key_here
PYTHONIOENCODING=utf-8
KMP_DUPLICATE_LIB_OK=TRUE
```

### LLM Providers
- Primary: Groq (fast, free tier available)
- Fallback: Ollama (local), OpenAI, Anthropic

## Testing Workflow

### 1. Basic Test
```bash
python test_chat.py
```

### 2. Quick Test (Automated)
```bash
python test_chat.py --quick-test
```

### 3. Load Specific File
```bash
python test_chat.py --load "path/to/file.xlsx"
```

## Sample Test Session

```
🤖 Re AI Chartered Accountant - Test Chat

💬 You: load "NSE-DATA-SCRAP.xlsx"
📂 Loading: NSE-DATA-SCRAP.xlsx
  📄 Found 6 sheets
    ✅ Loaded 'niftygainers' (2813 rows)
    ✅ Loaded 'volumegainer' (751 rows)

💬 You: how many sheets are there?
⏳ AI is thinking...
  🎯 Routed to: 📊 Data Analysis (confidence: 70%)
  📂 Using dataset: nse_data_scrap:volumegainer

🤖 AI: 📊 There are 6 sheets available.
  📁 Source: nse_data_scrap:volumegainer
  🔧 Method: metadata
  💡 Counted registered datasets
  ⏱️ Time: 0.01s

💬 You: what are the top 5 volume gainers?
⏳ AI is thinking...
  🎯 Routed to: 📊 Data Analysis (confidence: 85%)
  📂 Using dataset: nse_data_scrap:volumegainer

🤖 AI: 📊 [Results showing top 5 stocks...]
  📁 Source: nse_data_scrap:volumegainer
  🔧 Method: sql_duckdb:llm_semantic
  💡 Filtered and sorted by volume
  ⏱️ Time: 2.34s
```

## Advanced Features

### Smart Dataset Matching
- Analyzes query for sheet/file name mentions
- Semantic similarity matching
- Automatic fallback to best dataset

### Error Handling
- Graceful degradation on failures
- Detailed error messages with traceback
- Automatic retry with fallback methods

### Performance Optimization
- Query caching
- Lazy component initialization
- Efficient dataset registration

## Troubleshooting

### Common Issues

**1. DLL Loading Errors (Windows)**
- Already handled in test_chat.py
- Pre-imports torch with proper DLL paths

**2. Document Ingestor Unavailable**
- Qdrant not running: Start with `docker run -p 6333:6333 qdrant/qdrant`
- Falls back to Chroma automatically

**3. Web Search Not Working**
- Check internet connection
- Verify API keys if using paid services

## Performance Metrics

- **Average Query Time**: 0.5-3s (depending on complexity)
- **Dataset Loading**: ~0.1s per sheet
- **Document Ingestion**: ~1-5s per document
- **Web Search**: ~2-4s including synthesis

## Future Enhancements

- [ ] Batch query processing
- [ ] Export results to Excel/PDF
- [ ] Query history and replay
- [ ] Custom prompt templates
- [ ] Multi-language support

## Support

For issues or questions:
1. Check error messages and traceback
2. Review logs in console output
3. Verify environment variables
4. Test with simple queries first

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-23  
**Status**: Production Ready ✅
