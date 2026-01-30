# PowerShell startup script - respects FASTAPI_PORT from .env
# Usage: .\start.ps1

# Load .env and set env variables
$envFile = Join-Path (Get-Location) ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Read FASTAPI_PORT from env
$port = [System.Environment]::GetEnvironmentVariable("FASTAPI_PORT", "Process")
$host = [System.Environment]::GetEnvironmentVariable("FASTAPI_HOST", "Process")

if (-not $port) { $port = "9999" }
if (-not $host) { $host = "0.0.0.0" }

Write-Host "Starting AI-CA on ${host}:${port}" -ForegroundColor Green

# Start uvicorn with env-derived port
& uvicorn app.main:app --host $host --port $port --reload
