#!/bin/bash
# tools/build_upx.sh — build the DECLARED packer (UPX 5.2.0) from source, with
# NO root, NO package manager, NO cmake, and NO environment mutation.
#
# WHY THIS EXISTS (Hutter-Prize committee ruling, "Tightening up
# compression program build script requirements"):
#   "If a build procedure requires manual steps that could be in a build script,
#    the contestants be required to incorporate it into the build script."
#   "...will be rejected until it no longer appears in the README and its
#    functionality is properly incorporated into the build script -- meaning no
#    root dependencies such as apt install."
# The named-as-rejectable example is the UPX *installer* other entries ship: it
# runs `sudo apt-get install`, and it exports variables the build script then
# relies on. This script is the compliant replacement — it is CALLED BY
# construct_ship.sh (not by the evaluator), it never needs root, and it exports
# nothing: it prints the path of the packer it built and exits.
#
# WHY UPX 5.x SPECIFICALLY (measured):
#   upx 4.2.2 CANNOT pack this engine at all — `CantPackException: xspan
#   unexpected NULL pointer`, exit 1, file untouched. Measured on two hosts both
#   with and without the section-header strip, so the 5.x floor is a property of
#   the Zig-emitted ELF, not of our own -392 B strip step. There is no
#   "older upx is fine" fallback; the version below is a hard requirement.
#
# WHY NOT CMAKE: upstream's Makefile is a cmake wrapper and needs cmake >= 3.13,
# which is absent on EVERY box we control (six-box probe). A judge
# machine may equally lack it, and "if a build script doesn't build, the
# contestants will be required to fix it." This script needs only a C/C++
# compiler. Verified byte-equivalent to the cmake build (§3 of the report).
#
# Usage:  tools/build_upx.sh [OUTDIR]        -> prints the built upx path
# Env:    UPX_SRC_TARBALL  pre-downloaded upx-5.2.0-src.tar.xz (skips network)
#         CC / CXX / JOBS
set -euo pipefail

UPX_VERSION="5.2.0"
UPX_SRC_SHA256="af99e526d5759de94412aea1104d5e4ca406cb725295f8633ecc9e843dc1ce1c"
UPX_SRC_URL="https://github.com/upx/upx/releases/download/v${UPX_VERSION}/upx-${UPX_VERSION}-src.tar.xz"

OUTDIR="${1:-${PWD}/.upx-build}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
BIN="$OUTDIR/upx"

# Already built and correct? Reuse it (idempotent; keeps repeat ships cheap).
# ⚠ PIPE-FREE ON PURPOSE.  `cmd | head -1 | grep -q ...` under `set -o pipefail`
#   is a SIGPIPE RACE: grep -q exits on the first match, head and the writer die of
#   SIGPIPE, and the pipeline reports 141 — a FALSE FAILURE on a CORRECT tool.  That
#   exact shape produced a false `objcopy lacks --strip-section-headers` FATAL that
#   killed a judge-harness run.  A command substitution has no pipe to race on.
if [[ -x "$BIN" ]] && _v="$("$BIN" --version 2>/dev/null)" && [[ "${_v%%$'\n'*}" == *"upx $UPX_VERSION"* ]]; then
  echo "$BIN"; exit 0
fi

# ---- 1) source: local tarball if given, else fetch + verify -----------------
TARBALL="${UPX_SRC_TARBALL:-$OUTDIR/upx-${UPX_VERSION}-src.tar.xz}"
if [[ ! -f "$TARBALL" ]]; then
  command -v curl >/dev/null 2>&1 || {
    echo "FATAL: need curl to fetch $UPX_SRC_URL, or set UPX_SRC_TARBALL to a local copy." >&2
    echo "       (Deliberately NOT installing it — installing packages needs root," >&2
    echo "        which the prize committee's build-script ruling forbids inside a build script.)" >&2
    exit 1; }
  echo "== fetching upx ${UPX_VERSION} source ==" >&2
  curl -L --fail -o "$TARBALL.part" "$UPX_SRC_URL"
  mv "$TARBALL.part" "$TARBALL"
fi
GOT=$(sha256sum "$TARBALL" | cut -d' ' -f1)
[[ "$GOT" == "$UPX_SRC_SHA256" ]] || {
  echo "FATAL: upx source sha256 mismatch" >&2
  echo "  want $UPX_SRC_SHA256" >&2
  echo "  got  $GOT  ($TARBALL)" >&2
  exit 1; }

SRC="$OUTDIR/upx-${UPX_VERSION}-src"
if [[ ! -f "$SRC/src/main.cpp" ]]; then
  rm -rf "$SRC"; mkdir -p "$SRC"
  tar xf "$TARBALL" -C "$SRC" --strip-components=1
fi

# ---- 2) compile: flags lifted verbatim from upstream's own cmake config -----
# (build/rel/CMakeFiles/{upx,upx_vendor_ucl,upx_vendor_zlib}.dir/flags.make of a
# reference cmake configure; -w replaces -Wall since we are not developing upx.)
CXXFLAGS="-O2 -DNDEBUG -std=gnu++17 -fno-delete-null-pointer-checks -fno-lifetime-dse -fno-strict-aliasing -fno-strict-overflow -funsigned-char -fno-tree-vectorize -w -DUSE_UTIMENSAT=1 -I$SRC/vendor"
CFLAGS_UCL="-O2 -DNDEBUG -std=gnu11 -fno-delete-null-pointer-checks -fno-strict-aliasing -fno-strict-overflow -funsigned-char -fno-tree-vectorize -w -I$SRC/vendor/ucl/include -I$SRC/vendor/ucl"
CFLAGS_ZLIB="-O2 -DNDEBUG -std=gnu11 -fno-delete-null-pointer-checks -fno-strict-aliasing -fno-strict-overflow -funsigned-char -fno-tree-vectorize -w -DHAVE_UNISTD_H=1 -DHAVE_VSNPRINTF=1 -I$SRC/vendor/zlib"

OBJ="$OUTDIR/obj"
rm -rf "$OBJ"; mkdir -p "$OBJ"
echo "== building upx ${UPX_VERSION} from source (no root, no cmake) ==" >&2

fail=0
n=0
compile() { # $1=compiler  $2=flags  $3=source (relative to $SRC)
  local o="$OBJ/$(echo "$3" | tr '/' '_').o"
  ( cd "$SRC" && $1 $2 -c "$3" -o "$o" ) || { echo "COMPILE FAILED: $3" >&2; fail=1; } &
  n=$((n + 1)); (( n % JOBS )) || wait
}
# src/stub/ holds prebuilt loader stubs (checked into the tarball) — not compiled.
while read -r f; do compile "$CXX" "$CXXFLAGS" "$f"; done < <(cd "$SRC" && find src -name '*.cpp' -not -path 'src/stub/*' | sort)
while read -r f; do compile "$CC" "$CFLAGS_UCL" "$f"; done < <(cd "$SRC" && ls vendor/ucl/src/*.c | sort)
while read -r f; do compile "$CC" "$CFLAGS_ZLIB" "$f"; done < <(cd "$SRC" && ls vendor/zlib/*.c | sort)
wait
(( fail == 0 )) || { echo "FATAL: upx source compile failed (see above)" >&2; exit 1; }

$CXX -O2 -DNDEBUG -o "$BIN" "$OBJ"/*.o
rm -rf "$OBJ"

# ---- 3) prove it is the declared packer -------------------------------------
# ⚠ PIPE-FREE ON PURPOSE.  `cmd | head -1 | grep -q ...` under `set -o pipefail`
#   is a SIGPIPE RACE: grep -q exits on the first match, head and the writer die of
#   SIGPIPE, and the pipeline reports 141 — a FALSE FAILURE on a CORRECT tool.  That
#   exact shape produced a false `objcopy lacks --strip-section-headers` FATAL that
#   killed a judge-harness run.  A command substitution has no pipe to race on.
#   THIS SITE IS ON THE JUDGE'S BUILD PATH UNCONDITIONALLY (the container has no upx,
#   so build.sh always builds it), and it FATALs — the worst of the three.
_v="$("$BIN" --version 2>/dev/null || true)"   # `|| true`: keep the FATAL below, not a bare set -e exit
[[ "${_v%%$'\n'*}" == *"upx $UPX_VERSION"* ]] || {
  echo "FATAL: built binary does not report upx $UPX_VERSION:" >&2
  "$BIN" --version 2>&1 | head -3 >&2
  exit 1; }
echo "$BIN"
