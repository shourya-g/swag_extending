# ================================
# OOD Detection Pipeline for SWAG
# ================================

$ErrorActionPreference = "Continue"

Write-Host "========================================"
Write-Host " CIFAR-10 5-Class OOD SWAG Pipeline"
Write-Host "========================================"

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Create required folders
New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\figures" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\checkpoints" | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "`nTimestamp: $timestamp"

# --------------------------------
# STEP 1: Train 5-class SGD baseline
# --------------------------------
Write-Host "`n========================================"
Write-Host " STEP 1: Train 5-class SGD baseline"
Write-Host "========================================"

python -u -W ignore::DeprecationWarning -W ignore::UserWarning `
  -m src.train `
  --config configs/baseline_ood.yaml `
  2>&1 | Tee-Object "outputs\logs\ood_baseline_$timestamp.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Baseline training failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nBaseline training finished."

# Check baseline checkpoint exists
$baselineCkpt = "outputs\checkpoints\resnet18_cifar10_5class_sgd.pt"

if (!(Test-Path $baselineCkpt)) {
    Write-Host "`nERROR: Expected baseline checkpoint not found:"
    Write-Host $baselineCkpt
    exit 1
}

Write-Host "Found baseline checkpoint: $baselineCkpt"

# --------------------------------
# STEP 2: Train unified SWA + SWAG
# --------------------------------
Write-Host "`n========================================"
Write-Host " STEP 2: Train unified SWA + SWAG"
Write-Host "========================================"

python -u -W ignore::DeprecationWarning -W ignore::UserWarning `
  -m src.train_swag `
  --config configs/swag_ood.yaml `
  2>&1 | Tee-Object "outputs\logs\ood_swag_$timestamp.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: SWAG training failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nUnified SWA + SWAG training finished."

# Check expected SWA checkpoint exists
$swaCkpt = "outputs\checkpoints\resnet18_cifar10_5class_swa.pt"

if (!(Test-Path $swaCkpt)) {
    Write-Host "`nWARNING: Expected SWA checkpoint not found:"
    Write-Host $swaCkpt
    Write-Host "OOD evaluation may fail if this file is required."
}
else {
    Write-Host "Found SWA checkpoint: $swaCkpt"
}

# Check expected SWAG posterior exists
$swagPosterior = "outputs\checkpoints\resnet18_cifar10_5class_swag_posterior.pt"

if (!(Test-Path $swagPosterior)) {
    Write-Host "`nWARNING: Expected SWAG posterior not found:"
    Write-Host $swagPosterior
    Write-Host "OOD evaluation may fail if this file is required."
}
else {
    Write-Host "Found SWAG posterior: $swagPosterior"
}

# --------------------------------
# STEP 3: Evaluate OOD entropy
# --------------------------------
Write-Host "`n========================================"
Write-Host " STEP 3: Evaluate OOD entropy"
Write-Host "========================================"

python -u -W ignore::DeprecationWarning -W ignore::UserWarning `
  -m src.evaluation.ood_entropy `
  --config configs/swag_ood.yaml `
  2>&1 | Tee-Object "outputs\logs\ood_entropy_$timestamp.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: OOD entropy evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nOOD entropy evaluation finished."

# Check summary exists
$oodSummary = "outputs\metrics\ood_entropy_summary.csv"

if (!(Test-Path $oodSummary)) {
    Write-Host "`nERROR: Expected OOD summary not found:"
    Write-Host $oodSummary
    exit 1
}

Write-Host "Found OOD summary: $oodSummary"

# --------------------------------
# STEP 4: Plot OOD entropy results
# --------------------------------
Write-Host "`n========================================"
Write-Host " STEP 4: Plot OOD entropy results"
Write-Host "========================================"

python -u -W ignore::DeprecationWarning -W ignore::UserWarning `
  -m src.visualization.plot_ood_entropy `
  2>&1 | Tee-Object "outputs\logs\ood_plots_$timestamp.log"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: OOD plotting failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nOOD plots generated."

# --------------------------------
# STEP 5: Show final summary
# --------------------------------
Write-Host "`n========================================"
Write-Host " STEP 5: OOD Summary"
Write-Host "========================================"

Write-Host "`nContents of outputs\metrics\ood_entropy_summary.csv:"
Get-Content "outputs\metrics\ood_entropy_summary.csv"

# --------------------------------
# STEP 6: Git status
# --------------------------------
Write-Host "`n========================================"
Write-Host " STEP 6: Git status"
Write-Host "========================================"

git status

Write-Host "`n========================================"
Write-Host " OOD Pipeline Finished Successfully"
Write-Host "========================================"

Write-Host "`nGenerated files:"
Write-Host " - outputs\metrics\ood_entropy_data.pt"
Write-Host " - outputs\metrics\ood_entropy_summary.csv"
Write-Host " - outputs\figures\ood_entropy_histograms.png"
Write-Host " - outputs\figures\ood_entropy_summary.png"
Write-Host " - outputs\logs\ood_baseline_$timestamp.log"
Write-Host " - outputs\logs\ood_swag_$timestamp.log"
Write-Host " - outputs\logs\ood_entropy_$timestamp.log"
Write-Host " - outputs\logs\ood_plots_$timestamp.log"