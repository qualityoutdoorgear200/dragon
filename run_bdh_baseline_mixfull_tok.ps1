[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:HF_HOME = 'D:\cache\.cache'
$env:HF_DATASETS_CACHE = 'D:\cache\.cache'
Set-Location D:\dragon
& .\.venv\Scripts\Activate.ps1
New-Item -ItemType Directory -Path 'runs/bdh_baseline_mixfull_tok' -ErrorAction SilentlyContinue | Out-Null
Write-Host '[FullMix] Training bdh_baseline_mixfull_tok'
py -3.12 -m dragonwave.train --config configs/bdh_baseline_mixfull_tok.yaml 2>&1 | Tee-Object -FilePath 'runs/bdh_baseline_mixfull_tok/train_stdout.log'
