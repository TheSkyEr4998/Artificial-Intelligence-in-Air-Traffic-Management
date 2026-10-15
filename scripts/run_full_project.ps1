param(
    [Parameter(Mandatory=$true)][string]$FeatureDataDir,
    [Parameter(Mandatory=$true)][string]$CleanedDataDir,
    [Parameter(Mandatory=$true)][string]$SnappedDataDir,
    [Parameter(Mandatory=$true)][string]$ProgramsDir,
    [string]$PythonExecutable,
    [string]$RunName = 'complete-from-monthly-features'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$python = if ($PythonExecutable) { $PythonExecutable } elseif (Test-Path (Join-Path $repo '.venv\Scripts\python.exe')) { Join-Path $repo '.venv\Scripts\python.exe' } else { 'python' }
$runRoot = Join-Path $repo (Join-Path 'artifacts\runs' $RunName)
if (Test-Path -LiteralPath $runRoot) { throw "Run folder already exists. Preserve it and choose another -RunName: $runRoot" }

$monthNames = @('jan','feb','mar','apr','may','june','july','aug','sept','oct','nov','dec')
$featureFiles = Get-ChildItem -LiteralPath $FeatureDataDir -File
foreach ($month in 1..12) {
    $number = '{0:D2}' -f $month
    $featureName = "features_$($monthNames[$month - 1])"
    if (-not (Test-Path -LiteralPath (Join-Path $CleanedDataDir "cleanedx_$number.csv") -PathType Leaf)) { throw "Missing cleaned source file cleanedx_$number.csv" }
    if (-not (Test-Path -LiteralPath (Join-Path $SnappedDataDir "snapped$month.csv") -PathType Leaf)) { throw "Missing snapped source file snapped$month.csv" }
    if (-not ($featureFiles | Where-Object { $_.BaseName.ToLowerInvariant() -eq $featureName })) { throw "Missing monthly feature file $featureName.csv" }
}
foreach ($name in @('cleaning_data.exe', 'snap.exe', 'feature_extract.exe', 'man_map.txt')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProgramsDir $name) -PathType Leaf)) { throw "Missing source provenance file $name" }
}

New-Item -ItemType Directory -Path $runRoot | Out-Null
function Invoke-Stage {
    param([string[]]$Arguments)
    Write-Host "`n> $python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $python -B @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed with exit code $LASTEXITCODE. Keep the partial run for inspection and use a new RunName for a retry." }
}

$identity = Join-Path $runRoot 'identity_audit'
$step01 = Join-Path $runRoot 'step01'
$validation = Join-Path $runRoot 'source_validation'
$cohort = Join-Path $runRoot 'cohort'
$step02 = Join-Path $runRoot 'step02'
$step03 = Join-Path $runRoot 'step03'
$step04 = Join-Path $runRoot 'step04'
$step05 = Join-Path $runRoot 'step05'
$step06 = Join-Path $runRoot 'step06'
$prediction = Join-Path $runRoot 'predictions\example_predictions.csv'

Invoke-Stage @((Join-Path $repo 'src\identity_audit.py'), '--raw-dir', $CleanedDataDir, '--feature-dir', $FeatureDataDir, '--output-dir', $identity)
Invoke-Stage @((Join-Path $repo 'src\data_preparation.py'), '--data-dir', $FeatureDataDir, '--identity-audit', (Join-Path $identity 'identity_audit.json'), '--output-dir', $step01)
Invoke-Stage @((Join-Path $repo 'src\validate_sessions.py'), '--project-dir', $repo, '--prepared-file', (Join-Path $step01 'features_prepared.csv'), '--cleaned-dir', $CleanedDataDir, '--snapped-dir', $SnappedDataDir, '--programs-dir', $ProgramsDir, '--output-dir', $validation)
Invoke-Stage @((Join-Path $repo 'src\build_supported_cohort.py'), '--prepared', (Join-Path $step01 'features_prepared.csv'), '--validation', (Join-Path $validation 'record_validation.csv'), '--output-dir', $cohort)
Invoke-Stage @((Join-Path $repo 'src\create_splits.py'), '--input', (Join-Path $cohort 'features_supported.csv'), '--output-dir', $step02)
Invoke-Stage @((Join-Path $repo 'src\preprocessing.py'), '--split-dir', $step02, '--output-dir', $step03)
Invoke-Stage @((Join-Path $repo 'src\train_models.py'), '--split-dir', $step02, '--features-dir', $step03, '--output-dir', $step04)
Invoke-Stage @((Join-Path $repo 'src\evaluate.py'), '--split-dir', $step02, '--features-dir', $step03, '--model-dir', $step04, '--prepared-input', (Join-Path $cohort 'features_supported.csv'), '--output-dir', $step05)
Invoke-Stage @((Join-Path $repo 'src\quality_audit.py'), '--project-dir', $repo, '--artifacts-dir', $runRoot, '--feature-dir', $FeatureDataDir, '--snapped-dir', $SnappedDataDir, '--prepared-input', (Join-Path $cohort 'features_supported.csv'), '--output-dir', $step06)
Invoke-Stage @((Join-Path $repo 'src\predict.py'), '--model-dir', $step04, '--input', (Join-Path $repo 'examples\prediction_input.csv'), '--output', $prediction, '--support-profile', (Join-Path $step06 'training_support.json'))

$results = Get-Content -Raw -LiteralPath (Join-Path $step05 'test_results.json') | ConvertFrom-Json
$cohortSummary = Get-Content -Raw -LiteralPath (Join-Path $cohort 'cohort_summary.json') | ConvertFrom-Json
$split = Get-Content -Raw -LiteralPath (Join-Path $step02 'split_manifest.json') | ConvertFrom-Json
$selection = Get-Content -Raw -LiteralPath (Join-Path $step03 'selected_features.json') | ConvertFrom-Json
$run = [ordered]@{
    run_name = $RunName
    start_point = '12 monthly feature CSV files'
    source_session_screen = 'UTC and Europe/London clock hypotheses; source provenance gap policy; no label edits'
    prediction_task = 'retrospective completed-journey modelling'
    target = 'inferred taxi_time minutes'
    evaluation_status = 'repeated development evaluation; test dates were previously inspected in the earlier project run'
    test_is_fresh_independent_evidence = $false
    winner_selected_on_validation = $results.winner_selected_on_validation
    winner_candidate = $results.winner_candidate
    test_rows = $results.test_rows
    supported_rows = $cohortSummary.supported_rows
    records_held_for_source_review = $cohortSummary.excluded_for_source_review_rows
    training_rows = $split.partitions.train.rows
    validation_rows = $split.partitions.validation.rows
    test_source_dates = @($split.partitions.test.first_date, $split.partitions.test.last_date)
    selected_features = $selection.selected_factors
    pca_components_retained = $selection.pca_retained_components
    pca_variance_retained = $selection.pca_retained_route_variance
    test_metrics = $results.metrics
    notes = @('No coursework report is generated.', 'Model outputs are predictions of inferred labels, not official airport ground truth.', 'No predictions are clipped and no duration-based row filter is used.')
}
(Join-Path $runRoot 'run_summary.json') | ForEach-Object { $run | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $_ -Encoding utf8 }
Invoke-Stage @((Join-Path $repo 'src\export_run_figures.py'), '--run-dir', $runRoot, '--output-dir', (Join-Path $repo (Join-Path 'publication_figures' $RunName)))
Write-Host "`nEnd-to-end run complete: $runRoot" -ForegroundColor Green
Write-Host 'The run starts from monthly features, audits source-session support, then performs PCA-guided selection, ANFIS/MLP/polynomial training, evaluation and example prediction.'
Write-Host 'Evaluation is explicitly marked as repeated development evidence; it is not a fresh test.'
