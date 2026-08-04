# Phase 2 depth-anchor real-data integration test runner.
#
# Each test loads the full Krea-2-Raw model + a Depth-Anything-V2 perceptor
# (~tens of GB). The transformers/accelerate framework retains model references
# that gc.collect() cannot break between tests in ONE pytest process, so a
# single-process batch run OOMs after ~2 tests. The accepted Phase 2 evidence
# was itself generated with one `run.py` process per scenario; this runner
# mirrors that: each test runs in its OWN python process so GPU memory starts
# at ~0 every time.
#
# Usage (from <repo>):
#   powershell -ExecutionPolicy Bypass -File testing\integration\run_phase2_depth_tests.ps1
#
# Optional env knobs (passed through):
#   AI_TOOLKIT_TEST_DATASET, AI_TOOLKIT_DEPTH_TEST_STEPS, AI_TOOLKIT_DEPTH_TEST_OUTPUT
#
# Pass -SkipSmoke to drop the Layer C smoke test, -Extended to add the Layer D matrix.

[CmdletBinding()]
param(
    [switch]$SkipSmoke,
    [switch]$Extended
)

$ErrorActionPreference = "Continue"
$Repo = "<repo>"
$PyExe = "python"
$Module = "testing/integration/test_depth_consistency_real_data.py"
$TestFile = Join-Path $Repo $Module

if (-not (Test-Path $TestFile)) { Write-Error "Test file not found: $TestFile"; exit 2 }

$env:PYTHONPATH = $Repo
$env:AI_TOOLKIT_RUN_KREA_DEPTH_INTEGRATION = "1"
if (-not $env:AI_TOOLKIT_TEST_DATASET) {
    $env:AI_TOOLKIT_TEST_DATASET = Join-Path $Repo "datasets"
}
if (-not $env:AI_TOOLKIT_DEPTH_TEST_STEPS) { $env:AI_TOOLKIT_DEPTH_TEST_STEPS = "12" }

# Layer B (strict) + Layer C (smoke). Layer D (extended) is opt-in.
$tests = @(
    "test_depth_disabled_baseline",
    "test_depth_cache_generation_and_reuse",
    "test_strict_depth_only_lora_update",
    "test_depth_gradient_trace_and_alternation",
    "test_dataset_only_depth_activation",
    "test_preview_only_mode"
)
if (-not $SkipSmoke) { $tests += "test_operational_smoke" }
if ($Extended) {
    $env:AI_TOOLKIT_RUN_DEPTH_EXTENDED_MATRIX = "1"
    $tests += "test_extended_da2_model_comparison"
}

Write-Host "=== Phase 2 depth integration suite (process-isolated) ===" -ForegroundColor Cyan
Write-Host ("Python: " + $PyExe)
Write-Host ("Dataset: " + $env:AI_TOOLKIT_TEST_DATASET)
Write-Host ("Tests: " + ($tests -join ", "))
Write-Host ""

$results = @()
$overallStart = Get-Date
foreach ($t in $tests) {
    Write-Host "--- $t ---" -ForegroundColor Yellow
    $out = & $PyExe -m pytest "$Module::$t" -m "depth" -v --tb=short 2>&1
    $tail = ($out | Select-Object -Last 4) -join "`n"
    Write-Host $tail
    # Pytest's final summary line is wrapped in "===" borders, e.g.
    # "===== 1 passed, 74 warnings in 54s =====". Match the digit+passed token
    # anywhere, and treat an explicit "failed"/"error" count as a failure.
    $passed = $out | Select-String -Pattern "\d+\s+passed"
    $failed = $out | Select-String -Pattern "\d+\s+(failed|error)"
    $status = if ($passed -and -not $failed) { "PASS" } else { "FAIL" }
    $results += [pscustomobject]@{ Test = $t; Status = $status }
    Write-Host ""
}
$elapsed = ((Get-Date) - $overallStart).TotalSeconds

Write-Host "=== SUMMARY ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize
$npass = ($results | Where-Object Status -eq "PASS").Count
$nfail = ($results | Where-Object Status -eq "FAIL").Count
Write-Host ("{0}/{1} passed in {2:N0}s" -f $npass, $results.Count, $elapsed)
if ($nfail -gt 0) { exit 1 } else { exit 0 }
