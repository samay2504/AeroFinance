# AI-CA API Documentation

## Overview

AI-CA (AI Chartered Accountant) is a production-grade Agentic AI system for financial data analysis with:
- Multi-format data ingestion (Excel, CSV, JSON)
- SQL-first query execution with LLM fallback
- RAG (Retrieval-Augmented Generation) for semantic search
- Multi-tenant isolation with unique ID system

---

## Base URL

```
HTTP API:   http://localhost:8000
ZMQ IPC:    ipc:///tmp/ai_ca.sock  (for Node.js integration)
```

---

## Authentication & Identification

All requests require a `client_id` which is automatically normalized for consistency:

```javascript
// Input variations all normalize to the same ID
"Test Client:123"  → "test_client_123"
"test-client 123"  → "test_client_123"
```

### ID Schema

| ID Type | Format | Example | Purpose |
|---------|--------|---------|---------|
| `client_id` | Normalized string | `test_client_123` | Tenant isolation |
| `doc_id` | `doc_{timestamp}_{hash}` | `doc_24624287_7a8e58` | Document tracking |
| `dataset_id` | `{client}:{doc}:{sheet}` | `client:doc_xxx:income` | Dataset reference |
| `query_id` | `qry_{timestamp}_{random}` | `qry_24624288_0a9e8945` | Audit trail |
| `session_id` | `ses_{timestamp}_{random}` | `ses_24624287_994709a8` | Conversation tracking |

---

## REST API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "provider": "groq",
  "components": {
    "api": "ok",
    "sql_engine": "ok",
    "vector_db": "chroma",
    "llm": "groq"
  }
}
```

---

### 2. Upload File

Upload Excel, CSV, or JSON files for analysis.

```http
POST /v1/ai-ca/upload
Content-Type: multipart/form-data
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Excel (.xlsx, .xls), CSV (.csv), or JSON (.json) |
| `client_id` | String | Yes | Client/tenant identifier |
| `ingest_all` | Boolean | No | Ingest all sheets (default: true) |

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/ai-ca/upload \
  -F "file=@financial_report.xlsx" \
  -F "client_id=acme_corp" \
  -F "ingest_all=true"
```

**Response:**
```json
{
  "success": true,
  "doc_id": "doc_24624287_7a8e58",
  "datasets": [
    {
      "dataset_id": "acme_corp:doc_24624287_7a8e58:income_statement",
      "sheet": "Income Statement",
      "rows": 25,
      "columns": 10,
      "success": true
    },
    {
      "dataset_id": "acme_corp:doc_24624287_7a8e58:balance_sheet",
      "sheet": "Balance Sheet",
      "rows": 30,
      "columns": 8,
      "success": true
    }
  ]
}
```

---

### 3. Ingest JSON Text

Ingest raw JSON data directly (for pasted content).

```http
POST /v1/ai-ca/ingest-json
Content-Type: application/json
```

**Request Body:**
```json
{
  "json_text": "{\"income\": [{\"metric\": \"Revenue\", \"2023\": 1000000}]}",
  "client_id": "acme_corp",
  "source_name": "quarterly_report"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/ai-ca/ingest-json \
  -H "Content-Type: application/json" \
  -d '{
    "json_text": "{\"income\": [{\"metric\": \"Revenue\", \"2023\": 1000000}]}",
    "client_id": "acme_corp",
    "source_name": "quarterly_report"
  }'
```

**Response:**
```json
{
  "success": true,
  "doc_id": "",
  "datasets": [
    {
      "dataset_id": "acme_corp:quarterly_report:income",
      "table": "income",
      "rows": 1,
      "success": true
    }
  ]
}
```

---

### 4. Query Data

Execute analytical queries on uploaded data.

```http
POST /v1/ai-ca/query
Content-Type: application/json
```

**Request Body:**
```json
{
  "client": "acme_corp",
  "query": "What is the total revenue for 2023?",
  "dataset_id": null,
  "session_id": "ses_123456_abcdef",
  "use_cache": true
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `client` | String | Yes | Client/tenant identifier |
| `query` | String | Yes | Natural language query |
| `dataset_id` | String | No | Specific dataset to query (auto-detected if null) |
| `session_id` | String | No | Session ID for conversation tracking |
| `use_cache` | Boolean | No | Use cached results (default: true) |

**cURL Example:**
```bash
curl -X POST http://localhost:8000/v1/ai-ca/query \
  -H "Content-Type: application/json" \
  -d '{
    "client": "acme_corp",
    "query": "What is the total revenue for 2023?"
  }'
```

**Response:**
```json
{
  "success": true,
  "result": "The total revenue for 2023 is $15,000,000",
  "method": "sql_duckdb:llm_semantic",
  "explanation": "Executed SQL query on income_statement dataset",
  "error": null,
  "query_id": "qry_24624288_0a9e8945",
  "metadata": {
    "dataset_id": "acme_corp:doc_xxx:income_statement",
    "route": "DATA"
  }
}
```

**Query Methods:**
| Method | Description |
|--------|-------------|
| `sql_duckdb:direct` | Direct SQL execution |
| `sql_duckdb:llm_semantic` | LLM-generated SQL |
| `sql_duckdb:llm_analytical` | Complex analytical query |
| `rag:semantic` | RAG vector search |
| `llm:comprehensive` | Full LLM analysis |
| `web:search` | Web search results |

---

### 5. List Datasets

Get all datasets for a client.

```http
GET /v1/ai-ca/datasets?client_id=acme_corp
```

**Response:**
```json
{
  "success": true,
  "client_id": "acme_corp",
  "datasets": [
    {
      "dataset_id": "acme_corp:doc_xxx:income_statement",
      "rows": 25,
      "columns": ["metric", "2022", "2023", "2024"],
      "registered_at": "2026-01-12T13:30:24.288148Z"
    }
  ],
  "count": 1
}
```

---

### 6. System Metrics

Get system performance metrics.

```http
GET /v1/ai-ca/metrics
```

**Response:**
```json
{
  "llm": {
    "total_requests": 150,
    "total_tokens": 45000,
    "avg_latency_ms": 250
  },
  "datasets": 5,
  "provider": "groq"
}
```

---

## ZeroMQ IPC Integration

For Node.js/Electron integration, use ZMQ socket:

### Configuration

```yaml
# config.yaml
zmq:
  enabled: true
  socket_path: "ipc:///tmp/ai_ca.sock"
```

### Node.js Example

```javascript
const zmq = require('zeromq');

async function queryAI() {
  const socket = new zmq.Request();
  await socket.connect('ipc:///tmp/ai_ca.sock');
  
  const request = {
    action: 'query',
    request_id: 'req_' + Date.now(),
    payload: {
      query: 'What is total revenue?',
      dataset_id: 'acme_corp:doc_xxx:income',
      client_id: 'acme_corp'
    }
  };
  
  await socket.send(JSON.stringify(request));
  const [response] = await socket.receive();
  return JSON.parse(response.toString());
}
```

### ZMQ Actions

| Action | Payload | Description |
|--------|---------|-------------|
| `query` | `{query, dataset_id, client_id}` | Execute SQL query |
| `list_datasets` | `{client_id}` | List available datasets |

### ZMQ Response Format

```json
{
  "success": true,
  "request_id": "req_123456",
  "action": "query",
  "result": {
    "success": true,
    "result": "Total revenue: $15,000,000",
    "method": "sql_duckdb:llm_semantic",
    "query_id": "qry_24624288_0a9e8945",
    "client_id": "acme_corp"
  }
}
```

---

## Postman Collection

### Import Variables

```json
{
  "baseUrl": "http://localhost:8000",
  "clientId": "test_client"
}
```

### Request Examples

| Name | Method | URL |
|------|--------|-----|
| Health Check | GET | `{{baseUrl}}/health` |
| Upload File | POST | `{{baseUrl}}/v1/ai-ca/upload` |
| Ingest JSON | POST | `{{baseUrl}}/v1/ai-ca/ingest-json` |
| Query Data | POST | `{{baseUrl}}/v1/ai-ca/query` |
| List Datasets | GET | `{{baseUrl}}/v1/ai-ca/datasets?client_id={{clientId}}` |
| Get Metrics | GET | `{{baseUrl}}/v1/ai-ca/metrics` |

---

## Frontend Integration Guide

### React/Next.js Example

```typescript
// api/ai-ca.ts
const API_BASE = process.env.NEXT_PUBLIC_AI_CA_URL || 'http://localhost:8000';

interface QueryRequest {
  client: string;
  query: string;
  datasetId?: string;
  sessionId?: string;
}

interface QueryResponse {
  success: boolean;
  result: string;
  method: string;
  queryId: string;
  error?: string;
}

export async function queryAICA(req: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/v1/ai-ca/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      client: req.client,
      query: req.query,
      dataset_id: req.datasetId,
      session_id: req.sessionId,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  const data = await response.json();
  return {
    success: data.success,
    result: data.result,
    method: data.method,
    queryId: data.query_id,
    error: data.error,
  };
}

export async function uploadFile(
  file: File,
  clientId: string
): Promise<{ docId: string; datasets: any[] }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('client_id', clientId);
  formData.append('ingest_all', 'true');
  
  const response = await fetch(`${API_BASE}/v1/ai-ca/upload`, {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  return {
    docId: data.doc_id,
    datasets: data.datasets,
  };
}
```

### Session Management

```typescript
// Generate session ID on client
function generateSessionId(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 10);
  return `ses_${timestamp}_${random}`;
}

// Use in chat component
const [sessionId] = useState(() => generateSessionId());

async function sendMessage(query: string) {
  const response = await queryAICA({
    client: userId,
    query,
    sessionId,
  });
  // Store query_id for audit trail
  console.log('Query ID:', response.queryId);
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid file type, missing params) |
| 500 | Server error |

### Error Response Format

```json
{
  "success": false,
  "error": "No datasets found for client. Please upload data first.",
  "query_id": "qry_24624288_0a9e8945"
}
```

---

## ID Generator Utilities

Import from `app.core.id_generator`:

```python
from app.core.id_generator import (
    normalize_client_id,      # Normalize for VectorDB
    generate_doc_id,          # Generate document ID
    generate_dataset_id,      # Generate dataset ID
    generate_query_id,        # Generate query ID
    generate_session_id,      # Generate session ID
    parse_dataset_id,         # Parse dataset ID components
)
```

---

## Starting the Server

```bash
# Development with hot reload
cd Re
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Environment Variables

Create `.env` file in project root (not tracked in git):

```env
# LLM Provider Configuration
OPENAI_API_KEY=sk-xxx
GROQ_API_KEY=gsk_xxx
GOOGLE_API_KEY=AIzaSyXXX
HUGGINGFACEHUB_API_TOKEN=hf_xxx

# Local LLM (Ollama)
OLLAMA_ENABLED=true
OLLAMA_MODEL=dolphin-llama3

# OpenRouter (optional)
OPENROUTER_ENABLED=true
OPENROUTER_API_KEY=sk-or-xxx

# Provider preference order
LLM_PROVIDER_PREFERENCE=google_genai,groq,ollama,openrouter,openai,huggingface

# FastAPI & Server
FASTAPI_ENV=development
FASTAPI_DEBUG=true
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Vector DB Configuration
VECTOR_DB_TYPE=faiss  # 'faiss' (local) or 'qdrant' (cloud)
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=xxx
QDRANT_COLLECTION=AI-CA
FAISS_INDEX_PATH=./data/faiss_index

# File Storage
FILE_STORE_PATH=./data/uploads
DOCUMENT_STORE_PATH=./data/documents
MAX_UPLOAD_SIZE_MB=50
ALLOWED_FILE_TYPES=["pdf","docx","xlsx","csv","json"]

# Multi-tenancy & Security
JWT_SECRET_KEY=your-secure-jwt-secret
JWT_ALGORITHM=HS256
ENFORCE_TENANT_FILTERING=true

# Data Analyst Sandbox
SANDBOX_TIMEOUT_SECONDS=10
SANDBOX_MAX_MEMORY_MB=512
SANDBOX_ALLOWED_MODULES=["pandas","numpy","math","json","re","datetime"]

# ZeroMQ IPC
ZMQ_IPC_ENDPOINT=ipc:///tmp/ai_ca.sock
ZMQ_TIMEOUT_MS=5000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Development Settings
ENABLE_SWAGGER=true
ENABLE_CORS=true
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

---

## Infrastructure & Deployment

### File Storage Structure (S3 Compatible)

The system uses a hierarchical directory structure for file storage, which maps 1:1 to S3 object keys for easy migration and partitioning.

```
data/dataframe_cache/
├── {client_id}/               # Normalized Client ID (S3 Prefix)
│   ├── {doc_id}/              # Unique Document ID
│   │   ├── {sheet_name}.parquet  # Actual Data File
│   │   └── metadata.json         # (Optional) Sidecar metadata
```

**S3 Migration Strategy:**
1. Mount S3 bucket using `s3fs` or EFS to `data/dataframe_cache`
2. Configure `FILE_STORE_PATH` to point to the mount
3. System automatically partitions data by client for security and performance

### Vector Database Architecture

| Environment | Recommended Setup |
|-------------|-------------------|
| **Dev/Test** | ChromaDB (Local persistent) |
| **Production** | Qdrant Cloud (Managed Cluster) |

**Scaling Qdrant:**
- Enable `ENFORCE_TENANT_FILTERING=true`
- Use `client_id` as the payload filter key (automatically handled by ID generator)
- Shard collection by `client_id` for massive scale

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-01-12 | Added S3-compatible hierarchical storage & hierarchical ID system |
| 1.0.0 | 2026-01-12 | Initial release with unique ID system |
