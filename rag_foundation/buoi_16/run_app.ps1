$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Không tìm thấy Python tại $Python. Hãy tạo .venv và cài requirements.txt trước."
}

Set-Location -LiteralPath $ProjectRoot
$env:HF_HOME = Join-Path $ProjectRoot "cache\huggingface"
Write-Host "Streamlit: http://localhost:8502"
Write-Host "Giữ terminal này mở. Nhấn Ctrl+C để dừng."
& $Python -m streamlit run app.py
