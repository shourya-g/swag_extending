$ErrorActionPreference = "Continue"

Write-Host "========================================"
Write-Host " ViT-Base SWAG / LoRA-SWAG Pipeline"
Write-Host "========================================"

& ".\.venv\Scripts\Activate.ps1"

New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\figures" | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`nTimestamp: $timestamp"

python -u -W ignore::DeprecationWarning -W ignore::UserWarning `
  -m src.experiments.vit_base_swag_pipeline `
  --batch-size 2 `
  --num-workers 0 `
  --full-epochs 1 `
  --head-epochs 4 `
  --lora-epochs 4 `
  --ood-head-epochs 4 `
  --ood-lora-epochs 4 `
  --max-rank 20 `
  --num-samples 15 `
  --sample-scale 1.0 `
  --swag-start-epoch 0 `
  --save-freq 1 `
  --lora-rank 8 `
  --lora-alpha 16 `
  --lora-dropout 0.1 `
  --amp `
  2>&1 | Tee-Object "outputs\logs\vit_base_pipeline_$timestamp.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: ViT-Base pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n========================================"
Write-Host " Pipeline Finished"
Write-Host "========================================"

Write-Host "`nClassification summary:"
Get-Content "outputs\metrics\vit_base_pipeline_classification_summary.csv"

Write-Host "`nOOD summary:"
Get-Content "outputs\metrics\vit_base_pipeline_ood_summary.csv"

Write-Host "`nCost analysis:"
Get-Content "outputs\metrics\vit_base_swag_cost_analysis.csv"

Write-Host "`nTiming summary:"
Get-Content "outputs\metrics\vit_base_pipeline_timing_summary.csv"