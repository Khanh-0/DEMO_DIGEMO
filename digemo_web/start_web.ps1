# Run this file from any directory. Keep the PowerShell window open while using the website.
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\digemo_web\app.py --checkpoint checkpoints/best_model_MELD_260.pth --features-dir MELD_pro/features --host 127.0.0.1 --port 8000
