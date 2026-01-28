# 🧪 Postman Testing Guide - Step by Step

## Phase 1: Server Setup

### Step 1.1: Start the Server

```powershell
# Navigate to project
cd d:\Projects2.0\Valuenaire\Re

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Time:** ~5-10 seconds

---

### Step 1.2: Verify Server Health

**In Postman:**
1. Method: `GET`
2. URL: `http://localhost:8000/health`
3. Click **Send**

**Expected Response (200 OK):**
```json
{
    "status": "ok",
    "request_id": "req_...",
    "data": {
        "version": "1.0.0",
        "provider": "groq",
        "components": {
            "api": "ok",
            "sql_engine": "ok",
            "vector_db": "qdrant",
            "llm": "groq"
        },
        "uptime_s": 0
    },
    "error": null
}
```

**If not 200:** ❌ Server not responding. Check terminal for errors.

---

## Phase 2: File Upload

### Step 2.1: Prepare Test File

Create a simple Excel file or use an existing one from your data folder.

**Example files:**
- `data/documents/sample.xlsx`
- `data/documents/financial_data.csv`

### Step 2.2: Create Upload Request

**In Postman:**

1. **Create New Request**
   - Click **+** tab
   - Name: "Upload File"

2. **Method & URL**
   - Method: `POST`
   - URL: `http://localhost:8000/v1/ai-ca/upload`

3. **Body Settings**
   - Click **Body** tab
   - Click **form-data** button (IMPORTANT: NOT raw/JSON)

4. **Add Form Fields**
   - Row 1:
     - Key: `file`
     - Type: **File** (select from dropdown)
     - Value: Click "Select Files" and choose your Excel/CSV file
   
   - Row 2:
     - Key: `client_id`
     - Type: **Text**
     - Value: `test_user`
   
   - Row 3:
     - Key: `ingest_all`
     - Type: **Text**
     - Value: `true`

5. **Headers** (should auto-populate)
   - Should see `Content-Type: multipart/form-data` (auto)

6. **Click Send**

**Expected Response (200 OK):**
```json
{
    "status": "ok",
    "request_id": "req_...",
    "data": {
        "doc_id": "doc_...",
        "sheet_names": ["Sheet1", "Sheet2"],
        "sheets_count": 2,
        "dataset_ids": [
            "test_user:doc_...:sheet1",
            "test_user:doc_...:sheet2"
        ],
        "datasets": [
            {
                "success": true,
                "sheet_name": "Sheet1",
                "rows": 100,
                "columns": 10
            }
        ]
    },
    "error": null
}
```

**⚠️ If 422 Error:**
- ❌ Check: Is Body type set to **form-data**? (Not raw/JSON)
- ❌ Check: Are field names exactly `file`, `client_id`, `ingest_all`?
- ❌ Check: Is `file` type set to **File**? (Not Text)

**✅ Save this for later:** Copy the `dataset_ids` value (you'll need it for queries)

---

## Phase 3: List Datasets

### Step 3.1: Create List Request

**In Postman:**

1. **Create New Request**
   - Click **+** tab
   - Name: "List Datasets"

2. **Method & URL**
   - Method: `GET`
   - URL: `http://localhost:8000/v1/ai-ca/datasets?client_id=test_user`

3. **Click Send**

**Expected Response (200 OK):**
```json
{
    "status": "ok",
    "request_id": "req_...",
    "data": {
        "client_id": "test_user",
        "datasets": [
            {
                "dataset_id": "test_user:doc_...:sheet1",
                "rows": 100,
                "columns": 10,
                "created_at": "2026-01-24T..."
            }
        ],
        "count": 1
    },
    "error": null
}
```

**✅ Verify:** You should see the dataset you just uploaded

---

## Phase 4: Query Data

### Step 4.1: Create Query Request

**In Postman:**

1. **Create New Request**
   - Click **+** tab
   - Name: "Query Data"

2. **Method & URL**
   - Method: `POST`
   - URL: `http://localhost:8000/v1/ai-ca/query`

3. **Headers** 
   - Add (if not auto-set):
     - Key: `Content-Type`
     - Value: `application/json`

4. **Body Settings**
   - Click **Body** tab
   - Click **raw** button (IMPORTANT: NOT form-data)
   - Select **JSON** from dropdown (right side)

5. **Paste Request Body**
```json
{
    "client": "test_user",
    "query": "What is the total value?",
    "dataset_id": null,
    "session_id": null,
    "use_cache": true
}
```

6. **Click Send**

**Expected Response (200 OK):**
```json
{
    "success": true,
    "result": "Based on the data analysis, the total value is...",
    "method": "sql_duckdb",
    "explanation": "Executed SQL aggregation",
    "error": null,
    "query_id": "qry_...",
    "metadata": {
        "dataset_id": "test_user:doc_...:sheet1",
        "route": "TRACK_DATA"
    }
}
```

**⚠️ If 422 Error:**
- ❌ Check: Is Body type set to **raw** (not form-data)?
- ❌ Check: Is JSON format selected (dropdown)?
- ❌ Check: Field name is `client` (not `client_id`)?
- ❌ Check: Is `use_cache` a boolean `true` (not string `"true"`)?

---

## Phase 5: Advanced Queries

### Step 5.1: Query with Specific Dataset

If you have multiple sheets, specify which one:

```json
{
    "client": "test_user",
    "query": "Calculate the average revenue",
    "dataset_id": "test_user:doc_...:sheet2",
    "session_id": null,
    "use_cache": true
}
```

### Step 5.2: Query with Session (Multi-turn)

For conversation-style interaction:

```json
{
    "client": "test_user",
    "query": "What was last year's profit?",
    "dataset_id": null,
    "session_id": "session_user_001",
    "use_cache": true
}
```

Then follow up with:

```json
{
    "client": "test_user",
    "query": "What about this year?",
    "dataset_id": null,
    "session_id": "session_user_001",
    "use_cache": true
}
```

---

## Testing Checklist

| Step | Test | Expected | Status |
|------|------|----------|--------|
| 1 | Server health check | 200 OK, status "ok" | ☐ |
| 2 | File upload | 200 OK, doc_id received | ☐ |
| 3 | List datasets | 200 OK, dataset visible | ☐ |
| 4 | Query data | 200 OK, result returned | ☐ |
| 5 | Multiple queries | All return 200 OK | ☐ |
| 6 | Error handling | 422 error gone | ☐ |

---

## Troubleshooting During Testing

### Upload Returns 422

**Checklist:**
- [ ] Body is **form-data** (check button at top of Body tab)
- [ ] Field 1: `file` type is **File** (dropdown)
- [ ] Field 2: `client_id` type is **Text**
- [ ] Field 3: `ingest_all` type is **Text**
- [ ] Field names exactly match (case-sensitive)
- [ ] File is selected (not empty)

**Fix:**
```
1. Delete all form fields
2. Start fresh with 3 fields
3. Check each type carefully
4. Resend
```

### Query Returns 422

**Checklist:**
- [ ] Body is **raw** (check button at top of Body tab)
- [ ] JSON format selected (right dropdown)
- [ ] Field `client` (not `client_id`)
- [ ] Field `use_cache` is boolean `true` (not `"true"`)
- [ ] JSON is valid (not missing commas)

**Fix:**
```
1. Clear body
2. Copy exact JSON from this guide
3. Change only "test_user" and query text
4. Resend
```

### Connection Refused

**Fix:**
```
1. Check terminal - is server still running?
2. If not, restart: uvicorn app.main:app --reload
3. Wait 5 seconds
4. Resend
```

### 404 Not Found

**Fix:**
```
1. Check URL exactly:
   - Upload:  http://localhost:8000/v1/ai-ca/upload
   - Query:   http://localhost:8000/v1/ai-ca/query
2. Verify no typos
3. Verify port is 8000 (not 3000 or 5000)
```

---

## Success Indicators

✅ **Upload Success:**
- 200 OK response
- Contains `doc_id`
- Contains `dataset_ids` array
- `success` is `true` for at least one sheet

✅ **Query Success:**
- 200 OK response
- Contains `result` with actual answer
- `success` is `true`
- `method` is one of: `sql_duckdb`, `pandas`, `llm_direct`, `rag:semantic`, `web:search`

✅ **Overall Success:**
- No 422 errors
- All responses are 200 OK
- Data flows from upload → list → query without issues

---

## Performance Expectations

| Operation | Expected Time | Notes |
|-----------|----------------|-------|
| Health check | < 500ms | Should be instant |
| File upload | 2-10 seconds | Depends on file size |
| List datasets | < 500ms | Instant lookup |
| Simple query | 1-3 seconds | SQL execution |
| Complex query | 3-10 seconds | Multiple aggregations |
| LLM query | 2-5 seconds | Depends on LLM |

---

## Next Steps After Testing

1. ✅ All tests passing?
2. ✅ No more 422 errors?
3. ✅ Uploading and querying working?

**Then you can:**
- Create more test cases with different queries
- Upload multiple files for testing
- Test the semantic router with various query types
- Validate the AI-CA functionality end-to-end

---

**Last Updated:** January 24, 2026  
**Status:** Ready to Use ✅
