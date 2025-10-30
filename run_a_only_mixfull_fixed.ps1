[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:HF_HOME = 'D:\cache\.cache'
$env:HF_DATASETS_CACHE = 'D:\cache\.cache'
Set-Location D:\dragon
& .\.venv\Scripts\Activate.ps1
New-Item -ItemType Directory -Path 'runs/a_only_mixfull_fixed' -ErrorAction SilentlyContinue | Out-Null
Write-Host '[FullMix] Training a_only_mixfull_fixed'
python -m dragonwave.train --config configs/a_only_mixfull_fixed.yaml 2>&1 | Tee-Object -FilePath 'runs/a_only_mixfull_fixed/train_stdout.log'
