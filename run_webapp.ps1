# PowerShell One-Click Runner for GISxKMUTNB Web Application
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  [*] Launching GISxKMUTNB Web Application (FastAPI + GPU)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Green

$PYTHON_EXE = "C:\Users\kittisak\anaconda3\envs\gis_kmutnb\python.exe"

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "[!] Python environment not found at $PYTHON_EXE. Using default python." -ForegroundColor Yellow
    $PYTHON_EXE = "python"
}

Write-Host "[*] Using Python: $PYTHON_EXE" -ForegroundColor White
Write-Host "[*] Access the web application at: http://localhost:8080" -ForegroundColor Green
Write-Host "[*] Access Admin Panel at        : http://localhost:8080/admin" -ForegroundColor Yellow
Write-Host "[*] Interactive API Swagger Docs : http://localhost:8080/docs" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Green

# Ensure project root is in PYTHONPATH
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"

& $PYTHON_EXE scripts/run_server.py

