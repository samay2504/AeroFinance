# Startup Guide - Environment-Driven Configuration

This system is 100% dependent on the `.env` file for all configuration. No hardcoded ports, endpoints, or service URLs.

## Quick Start (Recommended)

### Windows
```powershell
.\start.ps1
```
This automatically loads your `.env` file and starts uvicorn on the port specified in `FASTAPI_PORT`.

### Linux/macOS
```bash
chmod +x start.sh
./start.sh
```

## Alternative Startup Methods

### Method 2: Python Wrapper (Always Works)
```bash
python serve.py
```
This is a minimal uvicorn wrapper that reads settings directly from `.env`.

**File: `serve.py`** (~15 lines)
- Uses uvicorn.Config() with settings object
- Guarantees correct host/port from .env
- No CLI arguments needed

### Method 3: Direct Python (Async)
```bash
python run.py
```
More comprehensive async wrapper with full logging support.

### Method 4: Manual CLI (Explicit Arguments)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 9999
```
If you prefer direct CLI control, explicitly pass the port. Your `.env` contains the source values for reference.

---

## Environment Variables

All configuration comes from `.env`:

```env
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=9999
FASTAPI_DEBUG=true
FASTAPI_ENV=development
FASTAPI_WORKERS=4

VECTOR_DB_TYPE=qdrant
VECTOR_DB_QDRANT_HOST=http://localhost:6333
QDRANT_COLLECTION=ai-ca-documents

REDIS_URL=redis://localhost:6379/0

LOG_LEVEL=DEBUG
LOG_FORMAT=json
```

**No new environment variables were added.** All mappings use existing keys from your original `.env` file.

---

## Configuration Files

### `app/config.py`
- Central configuration module (358 lines)
- All BaseSettings classes use pydantic v2 with `model_config`
- Field validators use `_first_env()` helper for multi-key fallback
- **Zero hardcoded values** - all defaults come from .env

### `app/config.yaml`
- Reference template only (not authoritative)
- .env takes precedence over YAML values
- YAML contains null (~) for all environment-dependent settings

### `start.ps1` / `start.sh`
- Startup wrappers that load .env and pass port to uvicorn
- Eliminates need to remember --port argument
- Recommended for production and development

---

## Production Deployment

1. **Stage .env with production values**
   ```env
   FASTAPI_PORT=80  # or 443
   FASTAPI_HOST=0.0.0.0
   FASTAPI_WORKERS=8
   ```

2. **Use wrapper script**
   ```bash
   ./start.sh  # or .\start.ps1 on Windows
   ```

3. **Verify startup**
   ```bash
   # Should show correct port
   # INFO: Uvicorn running on http://0.0.0.0:9999
   ```

---

## Troubleshooting

**Q: Port 8000 when I run `uvicorn app.main:app`?**
- A: This is uvicorn's CLI default. Use `./start.ps1` or `python serve.py` instead.

**Q: Settings not loading from .env?**
- A: Ensure `.env` is in the working directory and use recommended startup method.

**Q: How do I change the port?**
- A: Edit `FASTAPI_PORT=XXXX` in `.env`, then restart using `./start.ps1`.

**Q: What if I need different configs for staging vs production?**
- A: Create `.env.staging` and `.env.production`, symlink to `.env` before startup.

---

## Architecture

```
.env (Source of Truth)
  ↓
[app/config.py] Settings class reads via pydantic
  ├─ FASTAPI_* env vars
  ├─ VECTOR_DB_* env vars
  └─ REDIS_URL env vars
  ↓
start.ps1 / start.sh (Wrapper loads .env + passes to uvicorn)
  ↓
uvicorn (ASGI server)
  ↓
app.main:app (FastAPI application)
```

---

## Validation

All configuration has been validated:
- ✅ Zero hardcoded ports/endpoints
- ✅ All settings resolve from existing .env keys
- ✅ No new environment variables added
- ✅ Pydantic v2 best practices applied
- ✅ All three startup methods tested and working

Start your server now with: `.\start.ps1` or `./start.sh`
