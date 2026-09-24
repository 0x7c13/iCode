# Add Windows Defender real-time-scan exclusions for local chrys development.
#
# The test suite spawns thousands of git/python subprocesses and writes
# thousands of small temp files, and Defender's real-time scan taxes every
# one of them on a stock developer machine. (CI needs no equivalent step:
# hosted runner images ship with real-time scanning already disabled.)
# Exclusions are PERSISTENT: run this once from an elevated PowerShell and
# every later local test run benefits — nothing hooks into pytest and no
# per-run action is needed.
#
# Usage (elevated PowerShell):
#   .\scripts\setup_defender_exclusions.ps1          # add exclusions
#   .\scripts\setup_defender_exclusions.ps1 -Undo    # remove them again
#
# Excluded paths: the repository root, %TEMP% (pytest tmp roots live there),
# and the uv cache (%LOCALAPPDATA%\uv) when present.

#Requires -RunAsAdministrator

param(
    [switch]$Undo
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$paths = @($repoRoot, $env:TEMP)
$uvCache = Join-Path $env:LOCALAPPDATA "uv"
if (Test-Path $uvCache) {
    $paths += $uvCache
}

foreach ($path in $paths) {
    if ($Undo) {
        Remove-MpPreference -ExclusionPath $path
        Write-Host "Removed Defender exclusion: $path"
    } else {
        Add-MpPreference -ExclusionPath $path
        Write-Host "Added Defender exclusion: $path"
    }
}
