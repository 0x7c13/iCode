# Build a self-contained Python distribution with chrys and every runtime
# dependency pre-installed, ready to be embedded into a PyApp binary with
# PYAPP_SKIP_INSTALL=true.
#
# The resulting archive turns PyApp's first run into a pure unpack: no pip,
# no PyPI, no network.
#
# Usage:
#   .\scripts\build_offline_dist.ps1 -DistUrl URL [options]
#   .\scripts\build_offline_dist.ps1 -DistArchive PATH [options]
#
# Required (exactly one):
#   -DistUrl URL        python-build-standalone archive to start from
#   -DistArchive PATH   the same, already downloaded
#
# Options:
#   -Output PATH   where to write the archive (default: dist\chrys-offline-dist.tar.gz)
#   -Wheel PATH    install this prebuilt chrys wheel instead of building from the checkout
#   -Extras LIST   comma-separated extras (default: tui,doc_converter,observability)
#   -NoPrune       keep pip/idlelib/tkinter and duplicate ripgrep binaries
#
# The archive's layout mirrors python-build-standalone's install_only archives —
# a single top-level "python\" directory — so the PyApp build only needs
# PYAPP_DISTRIBUTION_PATH_PREFIX=python.

param(
    [string]$DistUrl,
    [string]$DistArchive,
    [string]$Output,
    [string]$Wheel,
    [string]$Extras = "tui,doc_converter,observability",
    [switch]$NoPrune
)

$ErrorActionPreference = "Stop"
# Windows PowerShell 5 renders a progress bar per received chunk, which slows
# the multi-MB Invoke-WebRequest download by an order of magnitude.
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ($DistUrl -and $DistArchive) {
    throw "Pass either -DistUrl or -DistArchive, not both"
}
if (-not $DistUrl -and -not $DistArchive) {
    throw "One of -DistUrl or -DistArchive is required"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
if (-not $Output) {
    $Output = Join-Path $ProjectRoot "dist\chrys-offline-dist.tar.gz"
}

# Push rather than Set: the working directory is shared with whoever dot-sourced
# or invoked this script, and build.ps1 calls it from inside the PyApp checkout.
Push-Location $ProjectRoot

$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) "chrys-offline-dist-$(Get-Random)"
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

try {
    # ── Obtain the base Python distribution ───────────────────────────
    if ($DistArchive) {
        Write-Host "==> Using local Python distribution: $DistArchive"
        tar xzf $DistArchive -C $WorkDir
    } else {
        # python-build-standalone URLs carry a percent-encoded '+' in the version.
        $DistUrl = $DistUrl.Replace("%2B", "+")
        Write-Host "==> Downloading Python distribution..."
        Write-Host "    $DistUrl"
        $Downloaded = Join-Path $WorkDir "python-dist.tar.gz"
        Invoke-WebRequest -Uri $DistUrl -OutFile $Downloaded
        tar xzf $Downloaded -C $WorkDir
        Remove-Item $Downloaded
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to unpack the Python distribution" }

    $DistRoot = Join-Path $WorkDir "python"
    if (-not (Test-Path $DistRoot -PathType Container)) {
        throw "Archive does not contain a top-level 'python\' directory. " +
              "Only python-build-standalone 'install_only' archives are supported."
    }

    $Py = Join-Path $DistRoot "python.exe"
    $Stdlib = Join-Path $DistRoot "Lib"
    $SitePackages = Join-Path $Stdlib "site-packages"
    if (-not (Test-Path $Py -PathType Leaf)) {
        throw "No interpreter found in the distribution ($Py)"
    }

    $PyVersion = & $Py -c "import sys; print('%d.%d' % sys.version_info[:2])"
    Write-Host "==> Base distribution: CPython $PyVersion"

    # ── Export the locked production dependency set ───────────────────
    # Driving the install from uv.lock (rather than letting a resolver run) is
    # what makes every shipped binary contain exactly the audited dependency
    # set.  --locked fails the build on a stale lock instead of silently
    # bundling a resolution nobody reviewed.
    $Requirements = Join-Path $WorkDir "requirements.txt"
    $ExportArgs = @("export", "--locked", "--no-dev", "--no-emit-project",
                    "--format", "requirements-txt", "-o", $Requirements, "--quiet")
    foreach ($extra in $Extras.Split(",")) {
        if ($extra) { $ExportArgs += @("--extra", $extra) }
    }

    Write-Host "==> Exporting locked dependencies (extras: $Extras)..."
    & uv @ExportArgs
    if ($LASTEXITCODE -ne 0) { throw "uv export failed" }
    $DepCount = (Select-String -Path $Requirements -Pattern '^[A-Za-z0-9]').Count
    Write-Host "    $DepCount locked distributions"

    # ── Install into the distribution ─────────────────────────────────
    # --compile-bytecode moves the byte-compilation of all packages to build
    # time: complete, deterministic pyc coverage instead of whatever the first
    # import graph happens to touch on the user's machine.
    Write-Host "==> Installing dependencies..."
    & uv pip install --python $Py --require-hashes --compile-bytecode -r $Requirements --quiet
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }

    Write-Host "==> Installing chrys..."
    $ChrysSource = if ($Wheel) { $Wheel } else { $ProjectRoot }
    & uv pip install --python $Py --no-deps --compile-bytecode $ChrysSource --quiet
    if ($LASTEXITCODE -ne 0) { throw "uv pip install (chrys) failed" }

    # ── Prune ─────────────────────────────────────────────────────────
    # Pruning runs before verification so the verify step vouches for exactly
    # what ships.  With PYAPP_SKIP_INSTALL nothing in the distribution ever
    # installs packages, so pip is dead weight; chrys resolves interpreters
    # for hooks and skills from PATH, never from its own runtime.
    if (-not $NoPrune) {
        Write-Host "==> Pruning unused components..."
        $Before = (Get-ChildItem $DistRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $Patterns = @(
            "$SitePackages\pip", "$SitePackages\pip-*.dist-info",
            "$SitePackages\setuptools", "$SitePackages\setuptools-*.dist-info",
            "$SitePackages\pkg_resources", "$SitePackages\_distutils_hack",
            "$SitePackages\wheel", "$SitePackages\wheel-*.dist-info",
            "$Stdlib\idlelib", "$Stdlib\tkinter", "$Stdlib\ensurepip",
            "$Stdlib\turtledemo", "$Stdlib\turtle.py",
            "$DistRoot\tcl", "$DistRoot\DLLs\_tkinter.pyd",
            "$DistRoot\DLLs\tcl*.dll", "$DistRoot\DLLs\tk*.dll",
            "$DistRoot\Scripts\pip*.exe", "$DistRoot\Scripts\idle*"
        )
        foreach ($pattern in $Patterns) {
            Get-Item $pattern -ErrorAction SilentlyContinue |
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        }

        # A CD wheel built with fetch_rg --all carries one ripgrep per
        # platform, but this distribution only ever runs on the platform it is
        # built on.  Ask find_rg's own platform table which binary that is, so
        # the two can never drift apart, and drop the rest.
        $KeepRg = & $Py -c @'
import platform
from chrys.foundation.vendor import _TRIPLE_MAP
names = _TRIPLE_MAP.get((platform.system(), platform.machine()))
print(names[0] if names else "")
'@
        $RgDir = Join-Path $SitePackages "chrys\foundation\vendor\ripgrep"
        if ($KeepRg -and (Test-Path $RgDir)) {
            Get-ChildItem (Join-Path $RgDir "rg-*") -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -ne $KeepRg } |
                Remove-Item -Force
            # The generic name is a duplicate once the platform-specific one exists.
            if (Test-Path (Join-Path $RgDir $KeepRg)) {
                Remove-Item (Join-Path $RgDir "rg"), (Join-Path $RgDir "rg.exe") `
                    -Force -ErrorAction SilentlyContinue
            }
        }

        $After = (Get-ChildItem $DistRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
        Write-Host ("    freed {0:N0} MB" -f (($Before - $After) / 1MB))
    }

    # ── Verify ────────────────────────────────────────────────────────
    # Runs after pruning so a prune mistake fails the build here instead of on
    # a user's machine.  Core probes always run (PIL.Image loads C extensions);
    # per-extra probes match the selected extras: the TUI subtree for tui,
    # lxml.etree (C extension via python-docx/pptx) for doc_converter,
    # opentelemetry.sdk for observability.
    Write-Host "==> Verifying the bundled installation..."
    $VerifyScript = Join-Path $WorkDir "verify.py"
    Set-Content -Path $VerifyScript -Encoding utf8 -Value @'
import importlib.metadata
import os
import subprocess
import sys

version = importlib.metadata.version("chrys")
extras = {e.strip() for e in os.environ.get("CHRYS_VERIFY_EXTRAS", "").split(",") if e.strip()}

from chrys.foundation.vendor import find_rg

rg = find_rg()
if rg is None or "site-packages" not in rg:
    sys.exit(f"bundled ripgrep not found in the distribution (resolved to {rg!r})")
subprocess.run([rg, "--version"], check=True, capture_output=True)

import chrys.app.cli.app  # noqa: F401  - the entry point every flavor boots through
import anthropic, certifi, mcp, openai  # noqa: F401,E401
import PIL.Image  # noqa: F401

if "tui" in extras:
    import chrys.app.tui.app  # noqa: F401  - heaviest import subtree
    import textual  # noqa: F401
if "doc_converter" in extras:
    import lxml.etree  # noqa: F401
if "observability" in extras:
    import opentelemetry.sdk  # noqa: F401

print(f"    chrys {version}")
print(f"    rg     {rg}")
print(f"    ca     {certifi.where()}")
'@
    $env:CHRYS_VERIFY_EXTRAS = $Extras
    try {
        & $Py $VerifyScript
        if ($LASTEXITCODE -ne 0) { throw "the bundled installation is not usable" }
    } finally {
        $env:CHRYS_VERIFY_EXTRAS = $null
    }
    Remove-Item $VerifyScript

    # ── Pack ──────────────────────────────────────────────────────────
    # tar preserves mtimes, so the bundled __pycache__ stays valid after PyApp
    # unpacks it — that is what keeps startup at tens of milliseconds.
    Write-Host "==> Packing..."
    $OutputDir = Split-Path -Parent $Output
    if ($OutputDir -and -not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }
    if (Test-Path $Output) { Remove-Item $Output -Force }
    tar czf $Output -C $WorkDir python
    if ($LASTEXITCODE -ne 0) { throw "failed to pack the distribution" }

    $Unpacked = (Get-ChildItem $DistRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $Archive = (Get-Item $Output).Length
    Write-Host "==> Done: $Output"
    Write-Host ("    archive {0:N0} MB  (unpacks to {1:N0} MB)" -f ($Archive / 1MB), ($Unpacked / 1MB))
} finally {
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
    Pop-Location
}
