param([string]$Output = (Join-Path $PSScriptRoot 'runs_distill'))

$ErrorActionPreference = 'Stop'
$py08 = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$out08 = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $out08) { throw "Output already exists: $out08. Choose a new path with -Output." }
New-Item -ItemType Directory -Path $out08 | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'cache\matplotlib'
$cpu08 = Get-CimInstance Win32_Processor
$os08 = Get-CimInstance Win32_OperatingSystem
[ordered]@{
    os = $os08.Caption
    build = $os08.BuildNumber
    cpu = $cpu08.Name
    cores = $cpu08.NumberOfCores
    logical = $cpu08.NumberOfLogicalProcessors
    installed_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    started = (Get-Date).ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out08 'hardware.json') -Encoding utf8
$phases08 = @()
foreach ($mode08 in 'quick','teacher','all','ablate','scale') {
    $args08 = @((Join-Path $PSScriptRoot 'distill_lm.py'),'--mode',$mode08,'--output',$out08)
    if ($mode08 -eq 'teacher') { $args08 += '--retrain' }
    $timer08 = [Diagnostics.Stopwatch]::StartNew()
    & $py08 -u @args08 2>&1 | Tee-Object -FilePath (Join-Path $out08 "$mode08.log")
    $exit08 = $LASTEXITCODE
    $timer08.Stop()
    $phases08 += [ordered]@{mode=$mode08; seconds=$timer08.Elapsed.TotalSeconds; exit_code=$exit08}
    ConvertTo-Json -InputObject @($phases08) | Set-Content -LiteralPath (Join-Path $out08 'phases.json') -Encoding utf8
    if ($exit08 -ne 0) { exit $exit08 }
}
& $py08 -u (Join-Path $PSScriptRoot 'run_extensions.py') --output $out08 2>&1 |
    Tee-Object -FilePath (Join-Path $out08 'extensions.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py08 -m pip freeze | Set-Content -LiteralPath (Join-Path $out08 'requirements.lock.txt') -Encoding utf8
