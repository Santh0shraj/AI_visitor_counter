Write-Output "=== Step 1: Fix Execution Policy ==="
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
Write-Output "Execution Policy Updated Successfully."

Write-Output "`n=== Step 2 & 4: Creating / Activating venv ==="
if (-Not (Test-Path "venv")) {
    Write-Output "Creating fresh venv..."
    python -m venv venv
}
Write-Output "Activating venv..."
& .\venv\Scripts\Activate.ps1

Write-Output "`n=== Step 5: Install Dependencies ==="
pip install opencv-python insightface onnxruntime numpy scipy flask ultralytics

Write-Output "`n=== Step 6: Confirm Activation ==="
where.exe python

Write-Output "`n=== Step 7: Run main.py ==="
python main.py
