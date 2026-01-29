# 🎯 Postman Quick Reference Card

## ⚡ Fix 422 Errors Checklist

### Upload Endpoint (POST /v1/ai-ca/upload)
- [ ] Method is `POST`
- [ ] URL is `http://localhost:8000/v1/ai-ca/upload`
- [ ] **Body type is `form-data`** (NOT raw/JSON) ⚠️
- [ ] Field 1: `file` (Type: File) - Select your Excel/CSV
- [ ] Field 2: `client_id` (Type: Text) - Value: `test_user`
- [ ] Field 3: `ingest_all` (Type: Text) - Value: `true`
- [ ] Click **Send**
- [ ] Expect: 200 OK with `doc_id` in response

### Query Endpoint (POST /v1/ai-ca/query)
- [ ] Method is `POST`
- [ ] URL is `http://localhost:8000/v1/ai-ca/query`
- [ ] **Body type is `raw`** (NOT form-data) ⚠️
- [ ] Format is `JSON` (dropdown in Postman)
- [ ] Field `client` (NOT `client_id`) ⚠️
- [ ] Field `query` contains your question
- [ ] Field `dataset_id`: `null`
- [ ] Field `session_id`: `null`
- [ ] Field `use_cache`: `true` (boolean, not string)
- [ ] Click **Send**
- [ ] Expect: 200 OK with `success: true` and result

---

## 🔄 Complete Workflow

```
1. START SERVER
   └─ uvicorn app.main:app --reload

2. HEALTH CHECK (GET)
   └─ http://localhost:8000/health
   └─ Expect: status "ok"

3. UPLOAD FILE (POST, form-data)
   └─ http://localhost:8000/v1/ai-ca/upload
   └─ file + client_id + ingest_all
   └─ Save: doc_id from response

4. LIST DATASETS (GET)
   └─ http://localhost:8000/v1/ai-ca/datasets?client_id=test_user
   └─ Verify: dataset_ids available

5. QUERY DATA (POST, JSON)
   └─ http://localhost:8000/v1/ai-ca/query
   └─ client + query + dataset_id + session_id + use_cache
   └─ Expect: success + result
```

---

## 📋 Copy-Paste Examples

### Upload JSON (COPY EXACTLY)
```
Method: POST
URL: http://localhost:8000/v1/ai-ca/upload
Body Type: form-data

Fields:
- file: [SELECT YOUR FILE]
- client_id: test_user
- ingest_all: true
```

### Query JSON (COPY EXACTLY)
```json
{
    "client": "test_user",
    "query": "What was the total revenue?",
    "dataset_id": null,
    "session_id": null,
    "use_cache": true
}
```

---

## ❌ Common Mistakes → ✅ Fixes

| Mistake | Error | Fix |
|---------|-------|-----|
| Upload with JSON body | 422 | Change Body to **form-data** |
| Query with form-data | 422 | Change Body to **raw JSON** |
| Query field `client_id` | 422 | Change to field name `client` |
| `use_cache: "true"` | 422 | Change to `use_cache: true` (no quotes) |
| Missing `file` field | 422 | Select file in form-data |
| Wrong URL path | 404 | Check `/v1/ai-ca/` prefix |
| Server not running | Connection refused | Run `uvicorn` command first |

---

## 🎮 Postman Settings

### Pre-request Script (Optional)
```javascript
// Auto-generate X-Request-ID header
if (!pm.request.headers.has("X-Request-ID")) {
    pm.request.headers.add({
        key: "X-Request-ID",
        value: "req_" + Date.now() + "_" + Math.random().toString(36).substr(2, 8)
    });
}
```

### Collection Variables
| Variable | Value |
|----------|-------|
| `base_url` | `http://localhost:8000` |
| `client_id` | `test_user` |

---

## 🔍 Debug in Postman

**To see actual request being sent:**
1. Click **Console** (Bottom-left corner)
2. Send request
3. Expand request in console
4. Check "Request Body" section
5. Compare with examples above
6. Look for field name typos or wrong types

---

## 📞 Quick Support

| Issue | Solution |
|-------|----------|
| "422 Unprocessable Entity" | See ❌→✅ table above |
| "404 Not Found" | Check URL path in address bar |
| "Connection refused" | Server not running - run `uvicorn` |
| "500 Internal Error" | Check server console for error message |
| Unsure about format | Check "Copy-Paste Examples" section |

---

**Last Updated:** January 24, 2026  
**Status:** Ready for Testing ✅
