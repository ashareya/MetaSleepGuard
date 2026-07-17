param(
    [ValidateSet(
        "status", "prepare", "train", "evaluate", "cross", "audit", "replay", "realtime", "brainstim",
        "real-openbci-report", "openbci-file-replay", "submission-pack", "metrics-export", "demo-assets",
        "public-sleep-baseline", "public-sleep-eval", "cross-dataset-eval", "public-sleep-download", "public-sleep-real-baseline",
        "isruc-download", "decision-evidence",
        "metabci-integration-test", "metabci-sleep-smoke", "report", "test"
    )]
    [string]$Task = "test",
    [ValidateSet("sleep-edf", "isruc")]
    [string]$Dataset = "sleep-edf",
    [ValidateSet("3class", "4class", "5class")]
    [string]$ClassificationTask = "5class",
    [string]$DataRoot,
    [string]$SleepEdfRoot,
    [string]$IsrucRoot,
    [string]$InputPath,
    [string]$Model,
    [string]$SerialPort,
    [switch]$Synthetic,
    [int]$MaxSubjects = 15,
    [double]$DurationSec = 60,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$sourceRoot = $PSScriptRoot

function Find-CondaEnvironmentPython {
    param([string]$EnvironmentName)
    $candidates = @()
    if ($env:CONDA_PREFIX -and (Split-Path -Leaf $env:CONDA_PREFIX) -eq $EnvironmentName) {
        $candidates += Join-Path $env:CONDA_PREFIX "python.exe"
    }
    if ($env:CONDA_EXE) {
        $condaRoot = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
        $candidates += Join-Path $condaRoot "envs\$EnvironmentName\python.exe"
    }
    if ($env:USERPROFILE) {
        $candidates += Join-Path $env:USERPROFILE "anaconda3\envs\$EnvironmentName\python.exe"
        $candidates += Join-Path $env:USERPROFILE "miniconda3\envs\$EnvironmentName\python.exe"
        $candidates += Join-Path $env:USERPROFILE ".conda\envs\$EnvironmentName\python.exe"
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        try {
            $environmentList = (& $conda.Source env list --json | ConvertFrom-Json).envs
            foreach ($environment in $environmentList) {
                if ((Split-Path -Leaf $environment) -eq $EnvironmentName) {
                    $candidate = Join-Path $environment "python.exe"
                    if (Test-Path -LiteralPath $candidate) { return $candidate }
                }
            }
        }
        catch { }
    }
    return $null
}

$analysisPython = Find-CondaEnvironmentPython "metabci"
$stimPython = Find-CondaEnvironmentPython "metabci_stim"

function Test-PythonImport {
    param([string]$Executable, [string]$Module)
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable)) { return $false }
    & $Executable -c "import $Module" 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not $Python) {
    if ($Task -eq "brainstim") {
        if (Test-PythonImport $stimPython "metabci.brainstim") { $Python = $stimPython }
        else { throw "Brainstim requires the metabci_stim environment. Pass -Python with a Python that imports metabci.brainstim." }
    }
    elseif (Test-PythonImport $analysisPython "metabci") {
        $Python = $analysisPython
    }
    else {
        throw "MetaSleep-Guard requires the metabci environment. Pass -Python, for example: .\run.ps1 -Task $Task -Python `"$analysisPython`""
    }
}
elseif (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$requiredModule = if ($Task -eq "brainstim") { "metabci.brainstim" } else { "metabci" }
if (-not (Test-PythonImport $Python $requiredModule)) {
    throw "Selected Python cannot import $requiredModule`: $Python"
}

Write-Host "python=$Python"
$packageParent = Split-Path -Parent $sourceRoot
$previousPythonPath = $env:PYTHONPATH
$sourcePaths = @($sourceRoot, $packageParent)
if ($previousPythonPath) {
    $sourcePaths += $previousPythonPath
}
$env:PYTHONPATH = $sourcePaths -join [IO.Path]::PathSeparator
Push-Location $sourceRoot
try {
    switch ($Task) {
        "status" { & $Python -m MetaSleepGuard.experiments.run_metabci_status }
        "prepare" {
            $args = @("-m", "MetaSleepGuard.experiments.run_prepare_data", "--dataset", $Dataset)
            if ($DataRoot) { $args += @("--root", $DataRoot) }
            & $Python @args
        }
        "train" {
            $args = @("-m", "MetaSleepGuard.experiments.run_train", "--dataset", $Dataset, "--task", $ClassificationTask)
            if ($DataRoot) { $args += @("--root", $DataRoot) }
            & $Python @args
        }
        "evaluate" {
            $args = @("-m", "MetaSleepGuard.experiments.run_eval", "--dataset", $Dataset, "--task", $ClassificationTask)
            if ($DataRoot) { $args += @("--root", $DataRoot) }
            if ($Model) { $args += @("--model", $Model) }
            & $Python @args
        }
        "cross" {
            $args = @("-m", "MetaSleepGuard.experiments.run_cross_dataset", "--task", $ClassificationTask)
            if ($SleepEdfRoot) { $args += @("--sleep-edf-root", $SleepEdfRoot) }
            if ($IsrucRoot) { $args += @("--isruc-root", $IsrucRoot) }
            & $Python @args
        }
        "audit" {
            if (-not $InputPath) { throw "-InputPath must point to the BDF/FIF directory" }
            & $Python -m MetaSleepGuard.experiments.run_bdf_fif_audit --input-dir $InputPath
        }
        "replay" {
            if (-not $InputPath) { throw "-InputPath must point to a replay file" }
            $args = @("-m", "MetaSleepGuard.experiments.run_openbci_file_replay", "--file", $InputPath)
            if ($Model) { $args += @("--model", $Model) }
            & $Python @args
        }
        "realtime" {
            $args = @("-m", "MetaSleepGuard.experiments.run_openbci_realtime", "--duration-sec", $DurationSec)
            if ($Synthetic) { $args += "--synthetic" }
            if ($SerialPort) { $args += @("--serial-port", $SerialPort) }
            if ($Model) { $args += @("--model", $Model) }
            & $Python @args
        }
        "brainstim" {
            $args = @("-m", "MetaSleepGuard.experiments.run_brainstim_calibration")
            if ($Synthetic) { $args += @("--dry-run", "--no-psychopy") }
            & $Python @args
        }
        "real-openbci-report" {
            $args = @("-m", "MetaSleepGuard.experiments.run_real_openbci_reports")
            if ($InputPath) { $args += @("--logs-root", $InputPath) }
            & $Python @args
        }
        "openbci-file-replay" {
            $args = @("-m", "MetaSleepGuard.experiments.run_real_openbci_reports")
            if ($InputPath) { $args += @("--logs-root", $InputPath) }
            & $Python @args
        }
        "submission-pack" { & $Python -m MetaSleepGuard.experiments.run_submission_pack --section all }
        "metrics-export" { & $Python -m MetaSleepGuard.experiments.run_submission_pack --section metrics }
        "demo-assets" { & $Python -m MetaSleepGuard.experiments.run_submission_pack --section demo }
        "public-sleep-baseline" {
            if (-not $DataRoot) {
                throw "-DataRoot must point to a complete Sleep-EDF or ISRUC directory. Synthetic fallback is disabled for this task."
            }
            foreach ($classTask in @("3class", "4class", "5class")) {
                & $Python -m MetaSleepGuard.experiments.run_train --dataset $Dataset --root $DataRoot --task $classTask --require-real-data
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
        }
        "public-sleep-eval" {
            if (-not $DataRoot) {
                throw "-DataRoot must point to a complete Sleep-EDF or ISRUC directory. Synthetic fallback is disabled for this task."
            }
            foreach ($classTask in @("3class", "4class", "5class")) {
                & $Python -m MetaSleepGuard.experiments.run_eval --dataset $Dataset --root $DataRoot --task $classTask --require-real-data
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
        }
        "cross-dataset-eval" {
            if (-not $SleepEdfRoot -or -not $IsrucRoot) {
                throw "-SleepEdfRoot and -IsrucRoot must point to complete public datasets. Synthetic fallback is disabled for this task."
            }
            foreach ($classTask in @("3class", "4class", "5class")) {
                & $Python -m MetaSleepGuard.experiments.run_cross_dataset --sleep-edf-root $SleepEdfRoot --isruc-root $IsrucRoot --task $classTask --require-real-data
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
        }
        "public-sleep-download" {
            if (-not $DataRoot) {
                throw "-DataRoot must point to the Sleep-EDF download directory."
            }
            & $Python -m MetaSleepGuard.experiments.download_sleep_edf_subset --data-root $DataRoot --subjects $MaxSubjects
        }
        "isruc-download" {
            $args = @("-m", "MetaSleepGuard.experiments.download_isruc", "--subjects", $MaxSubjects)
            if ($DataRoot) { $args += @("--data-root", $DataRoot) }
            & $Python @args
        }
        "decision-evidence" {
            $args = @("-m", "MetaSleepGuard.experiments.run_evidence_boost", "--subjects", $MaxSubjects)
            if ($DataRoot) { $args += @("--data-root", $DataRoot) }
            & $Python @args
        }
        "public-sleep-real-baseline" {
            $args = @("-m", "MetaSleepGuard.experiments.run_public_sleep_real_baseline", "--subjects", $MaxSubjects)
            if ($DataRoot) { $args += @("--data-root", $DataRoot) }
            & $Python @args
        }
        "metabci-integration-test" { & $Python -m MetaSleepGuard.experiments.run_metabci_integration_test }
        "metabci-sleep-smoke" { & $Python -m MetaSleepGuard.experiments.run_metabci_sleep_extension_smoke }
        "report" { & $Python -m MetaSleepGuard.experiments.run_generate_report }
        "test" { & $Python -m MetaSleepGuard.tests.run_all_tests }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
