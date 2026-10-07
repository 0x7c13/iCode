#!/usr/bin/env bash
# Build a self-contained Python distribution with chrys and every runtime
# dependency pre-installed, ready to be embedded into a PyApp binary with
# PYAPP_SKIP_INSTALL=true.
#
# The resulting archive turns PyApp's first run into a pure unpack: no pip,
# no PyPI, no network.
#
# Usage:
#   ./scripts/build_offline_dist.sh --dist-url URL [options]
#   ./scripts/build_offline_dist.sh --dist-archive PATH [options]
#
# Required (exactly one):
#   --dist-url URL        python-build-standalone archive to start from
#   --dist-archive PATH   the same, already downloaded
#
# Options:
#   --output PATH    where to write the archive (default: dist/chrys-offline-dist.tar.gz)
#   --wheel PATH     install this prebuilt chrys wheel instead of building from the checkout
#   --extras LIST    comma-separated extras (default: tui,doc_converter,observability)
#   --no-prune       keep pip/idlelib/tkinter and duplicate ripgrep binaries
#
# Environment variables:
#   OFFLINE_ONLY_BINARY   1 to require wheels for every dependency, 0 to allow
#                         source builds.  Defaults to 1 on Linux (a source build
#                         there would link against the build host's glibc and
#                         raise the runtime glibc floor) and 0 elsewhere.
#   OFFLINE_GLIBC_FLOOR   Linux only: the glibc version the shipped bundle may
#                         require, both selecting wheels (--python-platform
#                         manylinux; floors below 2.28 pin at 2_17 — uv's enum
#                         has nothing in between) and failing the build if any
#                         ELF object ends up above it.  Default 2.17 (CentOS 7,
#                         Ubuntu 14.04) — matches python-build-standalone's
#                         own interpreter, so the deps are never the limiting
#                         factor.  CD sets 2.18 on aarch64, where ripgrep has
#                         no static musl build and its gnu binary needs 2.18.
#                         At the 2_17 pin, locked versions PyPI publishes no
#                         2.17 wheel for come from offline_wheel_overrides.txt.
#
# The archive's layout mirrors python-build-standalone's ``install_only``
# archives — a single top-level ``python/`` directory — so the PyApp build
# only needs PYAPP_DISTRIBUTION_PATH_PREFIX=python.

set -euo pipefail

DIST_URL=""
DIST_ARCHIVE=""
OUTPUT=""
WHEEL=""
EXTRAS="tui,doc_converter,observability"
PRUNE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dist-url)     DIST_URL="$2"; shift 2 ;;
        --dist-archive) DIST_ARCHIVE="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --wheel)        WHEEL="$2"; shift 2 ;;
        --extras)       EXTRAS="$2"; shift 2 ;;
        --no-prune)     PRUNE=false; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 (--dist-url URL | --dist-archive PATH) [--output PATH] [--wheel PATH] [--extras LIST] [--no-prune]" >&2
            exit 1
            ;;
    esac
done

if [ -n "$DIST_URL" ] && [ -n "$DIST_ARCHIVE" ]; then
    echo "Error: pass either --dist-url or --dist-archive, not both" >&2
    exit 1
fi
if [ -z "$DIST_URL" ] && [ -z "$DIST_ARCHIVE" ]; then
    echo "Error: one of --dist-url or --dist-archive is required" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/dist/chrys-offline-dist.tar.gz}"

cd "$PROJECT_ROOT"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# Under Git Bash `uv` is a native Windows executable and cannot read the POSIX
# paths this script works in, so translate everything handed to it.
if command -v cygpath >/dev/null 2>&1; then
    native_path() { cygpath -w "$1"; }
else
    native_path() { printf '%s' "$1"; }
fi

# ── Obtain the base Python distribution ───────────────────────────────
if [ -n "$DIST_ARCHIVE" ]; then
    echo "==> Using local Python distribution: $DIST_ARCHIVE"
    tar xzf "$DIST_ARCHIVE" -C "$WORK_DIR"
else
    # python-build-standalone URLs carry a percent-encoded '+' in the version.
    DIST_URL="${DIST_URL//%2B/+}"
    echo "==> Downloading Python distribution..."
    echo "    $DIST_URL"
    curl -fsSL "$DIST_URL" | tar xz -C "$WORK_DIR"
fi

DIST_ROOT="$WORK_DIR/python"
if [ ! -d "$DIST_ROOT" ]; then
    echo "Error: archive does not contain a top-level 'python/' directory." >&2
    echo "       Only python-build-standalone 'install_only' archives are supported." >&2
    exit 1
fi

# Layout differs between Windows and POSIX distributions; probe rather than
# branching on the host OS so the script also works under Git Bash.
if [ -x "$DIST_ROOT/python.exe" ] || [ -f "$DIST_ROOT/python.exe" ]; then
    PY="$DIST_ROOT/python.exe"
    SITE_PACKAGES="$DIST_ROOT/Lib/site-packages"
    STDLIB="$DIST_ROOT/Lib"
else
    PY="$DIST_ROOT/bin/python3"
    STDLIB="$(echo "$DIST_ROOT"/lib/python*)"
    SITE_PACKAGES="$STDLIB/site-packages"
fi

if [ ! -f "$PY" ]; then
    echo "Error: no interpreter found in the distribution ($PY)." >&2
    exit 1
fi

PY_VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "==> Base distribution: CPython $PY_VERSION"

# ── Export the locked production dependency set ───────────────────────
# Driving the install from uv.lock (rather than letting a resolver run) is what
# makes every shipped binary contain exactly the audited dependency set.
# --locked fails the build on a stale lock instead of silently bundling a
# resolution nobody reviewed.
REQUIREMENTS="$WORK_DIR/requirements.txt"
EXPORT_ARGS=(--locked --no-dev --no-emit-project --format requirements-txt)
IFS=',' read -ra EXTRA_LIST <<< "$EXTRAS"
for extra in "${EXTRA_LIST[@]}"; do
    if [ -n "$extra" ]; then
        EXPORT_ARGS+=(--extra "$extra")
    fi
done

echo "==> Exporting locked dependencies (extras: $EXTRAS)..."
uv export "${EXPORT_ARGS[@]}" -o "$(native_path "$REQUIREMENTS")" --quiet
DEP_COUNT=$(grep -c '^[A-Za-z0-9]' "$REQUIREMENTS" || true)
echo "    $DEP_COUNT locked distributions"

# ── Install into the distribution ─────────────────────────────────────
if [ -n "${OFFLINE_ONLY_BINARY:-}" ]; then
    ONLY_BINARY="$OFFLINE_ONLY_BINARY"
elif [ "$(uname -s)" = "Linux" ]; then
    ONLY_BINARY=1
else
    ONLY_BINARY=0
fi
GLIBC_FLOOR="${OFFLINE_GLIBC_FLOOR:-2.17}"

# --compile-bytecode moves the byte-compilation of all 85 packages to build
# time: complete, deterministic pyc coverage instead of whatever the first
# import graph happens to touch on the user's machine.
INSTALL_ARGS=(--python "$(native_path "$PY")" --require-hashes --compile-bytecode)
if [ "$ONLY_BINARY" = "1" ]; then
    # A source build on Linux would compile against this host's glibc and
    # silently raise the floor for everyone running the released binary.
    INSTALL_ARGS+=(--only-binary :all:)
    if [ "$(uname -s)" = "Linux" ]; then
        # Without this pin uv selects wheels for the *build host's* glibc
        # (ubuntu-latest is 2.39), silently shipping e.g. a manylinux_2_34
        # cryptography.  Pinning makes wheel selection match the floor the
        # release notes promise, independent of the runner image.
        #
        # uv's --python-platform enum jumps from manylinux_2_17 straight to
        # manylinux_2_28, so a floor in between (e.g. 2.18, forced on aarch64
        # by the vendored ripgrep) pins wheels at 2_17 — a lower pin only
        # makes wheels more portable, never less.  The ELF scan below still
        # enforces the real floor.
        WHEEL_GLIBC="$GLIBC_FLOOR"
        if [ "${GLIBC_FLOOR%%.*}" = "2" ] && [ "${GLIBC_FLOOR#2.}" -lt 28 ]; then
            WHEEL_GLIBC="2.17"
        fi
        INSTALL_ARGS+=(--python-platform "$(uname -m)-manylinux_${WHEEL_GLIBC//./_}")

        # Some locked versions publish no manylinux_2_17 wheel (Pillow from
        # 12.3.0 on).  offline_wheel_overrides.txt names a wheel built for the
        # floor for each; its hash replaces the requirement's PyPI hashes, so
        # it is the only artifact the install accepts, and the ELF scan below
        # still checks what it contains.
        if [ "$WHEEL_GLIBC" = "2.17" ]; then
            OVERRIDE_DIR="$WORK_DIR/override-wheels"
            OVERRIDE_PLAN="$WORK_DIR/override-wheels.tsv"
            mkdir -p "$OVERRIDE_DIR"
            "$PY" - "$SCRIPT_DIR/offline_wheel_overrides.txt" "$REQUIREMENTS" \
                "$(uname -m)" "$PY_VERSION" "$OVERRIDE_PLAN" <<'PYCODE'
import re
import sys
from pathlib import Path

manifest, requirements, machine, py_version, plan = sys.argv[1:]
abi_tag = "-cp" + py_version.replace(".", "") + "-"


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


overrides: dict[str, tuple[str, str, str]] = {}
for raw in Path(manifest).read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line:
        continue
    project, version, arch, sha256, url = line.split()
    if arch == machine:
        overrides[canonical(project)] = (version, sha256, url)

lines = Path(requirements).read_text(encoding="utf-8").splitlines()
rewritten: list[str] = []
rows: list[str] = []
index = 0
while index < len(lines):
    line = lines[index]
    index += 1
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line)
    if match is None or canonical(match.group(1)) not in overrides:
        rewritten.append(line)
        continue
    version, sha256, url = overrides[canonical(match.group(1))]
    if match.group(2) != version:
        sys.exit(
            f"uv.lock pins {match.group(1)}=={match.group(2)} but "
            f"offline_wheel_overrides.txt has {version} for {machine}; "
            "rebuild the wheel and update both together."
        )
    filename = url.rsplit("/", 1)[1]
    if abi_tag not in filename:
        sys.exit(f"{filename} is not built for CPython {py_version}.")
    rewritten.append(line.split("\\", 1)[0].rstrip() + " \\")
    rewritten.append(f"    --hash=sha256:{sha256}")
    while index < len(lines) and lines[index].lstrip().startswith("--hash="):
        index += 1
    rows.append(f"{filename}\t{sha256}\t{url}\n")

Path(requirements).write_text("\n".join(rewritten) + "\n", encoding="utf-8")
Path(plan).write_text("".join(rows), encoding="utf-8")
PYCODE
            while IFS=$'\t' read -r filename sha256 url; do
                echo "    $filename (built for glibc 2.17)"
                curl -fsSL --retry 3 -o "$OVERRIDE_DIR/$filename" "$url"
                echo "$sha256  $OVERRIDE_DIR/$filename" | sha256sum -c --quiet -
            done < "$OVERRIDE_PLAN"
            INSTALL_ARGS+=(--find-links "$OVERRIDE_DIR")
        fi
    fi
    echo "==> Installing dependencies (wheels only)..."
else
    # Some dependencies legitimately have no wheel for every interpreter and
    # platform (watchdog publishes no CPython 3.14 macOS wheel, so its
    # _watchdog_fsevents extension is built here).
    echo "==> Installing dependencies (source builds permitted)..."
fi
uv pip install "${INSTALL_ARGS[@]}" -r "$(native_path "$REQUIREMENTS")" --quiet

echo "==> Installing chrys..."
CHRYS_SOURCE="${WHEEL:-$PROJECT_ROOT}"
uv pip install --python "$(native_path "$PY")" --no-deps --compile-bytecode \
    "$(native_path "$CHRYS_SOURCE")" --quiet

# ── Prune ─────────────────────────────────────────────────────────────
# Pruning runs before verification so the verify step vouches for exactly what
# ships.  With PYAPP_SKIP_INSTALL nothing in the distribution ever installs
# packages, so pip is dead weight; chrys resolves interpreters for hooks and
# skills from PATH, never from its own runtime.
if [ "$PRUNE" = "true" ]; then
    echo "==> Pruning unused components..."
    BEFORE=$(du -sk "$DIST_ROOT" | cut -f1)
    for path in \
        "$SITE_PACKAGES"/pip "$SITE_PACKAGES"/pip-*.dist-info \
        "$SITE_PACKAGES"/setuptools "$SITE_PACKAGES"/setuptools-*.dist-info \
        "$SITE_PACKAGES"/pkg_resources "$SITE_PACKAGES"/_distutils_hack \
        "$SITE_PACKAGES"/wheel "$SITE_PACKAGES"/wheel-*.dist-info \
        "$STDLIB"/idlelib "$STDLIB"/tkinter "$STDLIB"/ensurepip "$STDLIB"/turtledemo \
        "$STDLIB"/turtle.py "$STDLIB"/lib-dynload/_tkinter*.so \
        "$DIST_ROOT"/lib/libtcl*.so* "$DIST_ROOT"/lib/libtk*.so* \
        "$DIST_ROOT"/lib/libtcl*.dylib "$DIST_ROOT"/lib/libtk*.dylib \
        "$DIST_ROOT"/lib/tcl[0-9]* "$DIST_ROOT"/lib/tk[0-9]* \
        "$DIST_ROOT"/lib/itcl[0-9]* "$DIST_ROOT"/lib/thread[0-9]* \
        "$DIST_ROOT"/tcl "$DIST_ROOT"/DLLs/_tkinter.pyd \
        "$DIST_ROOT"/DLLs/tcl*.dll "$DIST_ROOT"/DLLs/tk*.dll \
        "$DIST_ROOT"/bin/pip "$DIST_ROOT"/bin/pip3 "$DIST_ROOT"/bin/pip3.* \
        "$DIST_ROOT"/bin/idle3 "$DIST_ROOT"/bin/idle3.* \
        "$DIST_ROOT"/Scripts/pip.exe "$DIST_ROOT"/Scripts/pip3.exe "$DIST_ROOT"/Scripts/pip3.*.exe
    do
        if [ -e "$path" ]; then
            rm -rf "$path"
        fi
    done

    # A CD wheel built with fetch_rg.sh --all carries one ripgrep per platform,
    # but this distribution only ever runs on the platform it is built on.
    # Ask find_rg's own platform table which binary that is, so the two can
    # never drift apart, and drop the rest.
    KEEP_RG="$("$PY" -c '
import platform
from chrys.foundation.vendor import _TRIPLE_MAP
names = _TRIPLE_MAP.get((platform.system(), platform.machine()))
print(names[0] if names else "")
')"
    RG_DIR="$SITE_PACKAGES/chrys/foundation/vendor/ripgrep"
    if [ -n "$KEEP_RG" ] && [ -d "$RG_DIR" ]; then
        for rg_bin in "$RG_DIR"/rg-*; do
            if [ -e "$rg_bin" ] && [ "$(basename "$rg_bin")" != "$KEEP_RG" ]; then
                rm -f "$rg_bin"
            fi
        done
        # The generic name is a duplicate once the platform-specific one exists.
        if [ -f "$RG_DIR/$KEEP_RG" ]; then
            rm -f "$RG_DIR/rg" "$RG_DIR/rg.exe"
        fi
    fi

    AFTER=$(du -sk "$DIST_ROOT" | cut -f1)
    echo "    freed $(( (BEFORE - AFTER) / 1024 )) MB"
fi

# ── Verify ────────────────────────────────────────────────────────────
# Runs after pruning so a prune mistake fails the build here instead of on a
# user's machine.  Core probes always run (PIL.Image loads C extensions);
# per-extra probes match the selected extras: the TUI subtree for tui,
# lxml.etree (C extension via python-docx/pptx) for doc_converter,
# opentelemetry.sdk for observability.
echo "==> Verifying the bundled installation..."
CHRYS_VERIFY_EXTRAS="$EXTRAS" "$PY" - <<'PYCODE'
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

import chrys.app.cli.app  # noqa: F401  — the entry point every flavor boots through
import anthropic, certifi, mcp, openai  # noqa: F401,E401
import PIL.Image  # noqa: F401

if "tui" in extras:
    import chrys.app.tui.app  # noqa: F401  — heaviest import subtree
    import textual  # noqa: F401
if "doc_converter" in extras:
    import lxml.etree  # noqa: F401
if "observability" in extras:
    import opentelemetry.sdk  # noqa: F401

print(f"    chrys {version}")
print(f"    rg     {rg}")
print(f"    ca     {certifi.where()}")
PYCODE

# ── Enforce the glibc floor ───────────────────────────────────────────
# The distribution is only as portable as its most demanding ELF object.  The
# --python-platform pin above constrains which wheels uv *selects*; this scan
# verifies what actually landed, so a dependency that starts shipping
# tighter-than-tagged binaries (or a source build sneaking in via
# OFFLINE_ONLY_BINARY=0) fails the build instead of a user's machine.
# python-build-standalone's own interpreter sits at 2.17 (CentOS 7, 2014).
if [ "$(uname -s)" = "Linux" ]; then
    echo "==> Enforcing glibc floor ${GLIBC_FLOOR}..."
    "$PY" - "$DIST_ROOT" "$GLIBC_FLOOR" <<'PYCODE'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
floor = tuple(int(part) for part in sys.argv[2].split("."))
worst_version: tuple[int, ...] = ()
worst_path = None
offenders: list[tuple[str, str]] = []
for path in root.rglob("*"):
    if not path.is_file() or path.is_symlink():
        continue
    with path.open("rb") as handle:
        if handle.read(4) != b"\x7fELF":
            continue
        handle.seek(0)
        blob = handle.read()
    versions = {
        tuple(int(part) for part in match.decode().split("_", 1)[1].split("."))
        for match in re.findall(rb"GLIBC_\d+\.\d+(?:\.\d+)?", blob)
    }
    if not versions:
        continue
    top = max(versions)
    if top > worst_version:
        worst_version, worst_path = top, path.relative_to(root)
    if top > floor:
        offenders.append((".".join(map(str, top)), str(path.relative_to(root))))

if worst_path is None:
    print("    glibc floor: none — no ELF object references a versioned glibc symbol")
else:
    print(f"    glibc floor: {'.'.join(map(str, worst_version))}  ({worst_path})")
if offenders:
    print(f"ERROR: {len(offenders)} object(s) exceed the supported glibc "
          f"{'.'.join(map(str, floor))}:", file=sys.stderr)
    for version, rel in sorted(offenders):
        print(f"  GLIBC_{version}  {rel}", file=sys.stderr)
    print("Raise OFFLINE_GLIBC_FLOOR only together with the README/release-note "
          "portability claim.", file=sys.stderr)
    sys.exit(1)
PYCODE
fi

# ── Pack ──────────────────────────────────────────────────────────────
# tar preserves mtimes, so the bundled __pycache__ stays valid after PyApp
# unpacks it — that is what keeps startup at tens of milliseconds.
echo "==> Packing..."
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"
tar czf "$OUTPUT" -C "$WORK_DIR" python

UNPACKED=$(du -sh "$DIST_ROOT" | cut -f1)
ARCHIVE=$(du -h "$OUTPUT" | cut -f1)
echo "==> Done: $OUTPUT"
echo "    archive $ARCHIVE  (unpacks to $UNPACKED)"
