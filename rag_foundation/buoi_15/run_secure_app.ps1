$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path (Split-Path -Parent $ProjectRoot) "buoi_14\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Không tìm thấy .venv của Buổi 15 hoặc Buổi 14" }
Set-Location -LiteralPath $ProjectRoot
$LocalModelCache = Join-Path $ProjectRoot "cache\huggingface"
$PreviousModelCache = Join-Path (Split-Path -Parent $ProjectRoot) "buoi_14\cache\huggingface"
$env:HF_HOME = if (Test-Path -LiteralPath $LocalModelCache) { $LocalModelCache } else { $PreviousModelCache }
Write-Host "Secure RBAC App: http://localhost:8503"
Write-Host "Giữ terminal mở; nhấn Ctrl+C để dừng."
& $Python -m streamlit run app_secure.py --server.port 8503
