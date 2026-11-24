Set-Location (Split-Path $PSScriptRoot -Parent)
$env:DIGEMO_OVERLAP_ASR = "0"   # bỏ dòng này nếu không dùng tách giọng chồng
python .\digemo_web\app.py --checkpoint checkpoints/digemo_meld.pth --host 127.0.0.1 --port 8000