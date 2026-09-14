#!/bin/bash
# tools/provision_zig.sh — obtain the DECLARED Zig toolchain (EXACTLY 0.15.1) with
# NO root, NO package manager, NO compiler, and NO environment mutation.
#
# WHY THIS EXISTS (Hutter-Prize committee ruling, "Tightening up
# compression program build script requirements"):
#   "If a build procedure requires manual steps that could be in a build script,
#    the contestants be required to incorporate it into the build script."
#   "If a build script doesn't build, the contestants will be required to fix it."
# Before this script, construct_ship.sh hardcoded
#   export PATH="$HOME/.local/zig-0.15.1:..."
# — a path that exists on our boxes and on no evaluator's machine. Obtaining the
# compiler was therefore a MANUAL STEP that a build script could perform, and on
# the committee's "test machine or well-specified cloud instance" the script would
# either fall through to whatever `zig` happened to be on PATH or fail outright.
# Both halves of the ruling, in one line. This script is the compliant fix: it is
# CALLED BY construct_ship.sh (never by the evaluator), it needs no root, and it
# exports nothing — it prints the path of the `zig` it resolved and exits.
#
# WHY THE PIN IS EXACT, NOT A FLOOR ("0.15.1", never "0.15.1+"):
#   The compiler is not a packaging detail here — it is part of the CODEC. A
#   different Zig emits different codegen, and different codegen can change the
#   arithmetic-coder's bit decisions, hence archive9, hence S. The committee
#   REBUILDS comp9 from source and books S off ITS OWN rebuild (it did exactly
#   that to cmix-lex, +10,931 B vs the submitted artifact), so the version the
#   evaluator's machine happens to carry would silently become the version that
#   is scored. A floor ("or newer") is therefore not a weaker specification of
#   the same thing — it is the wrong specification. construct_ship.sh asserts
#   `zig version` equals the pin exactly and aborts otherwise.
#
# WHY NO BUILD STEP (unlike tools/build_upx.sh): Zig ships a self-contained,
# relocatable x86_64-linux tarball — compiler + std lib + linker, no system
# dependencies, nothing to compile, nothing to install. Fetch, verify, untar,
# use in place.
#
# VERIFIED : the tarball pinned below unpacks to a `zig` binary whose
# sha256 is 0ee27482eb2e7b19fad58579c107364aa93fbfb54be2d2b9c52cef5f955d6225 and a
# lib/ tree byte-identical (diff -rq, zero differences) to ~/.local/zig-0.15.1 —
# the toolchain that produced every banked zmix artifact. Provisioning from this
# pin reproduces the exact compiler the measurements were made with.
#
# Usage:  tools/provision_zig.sh [OUTDIR]      -> prints the resolved `zig` path
# Env:    ZIG_SRC_TARBALL  pre-downloaded zig-x86_64-linux-0.15.1.tar.xz
#                          (skips the network, for an offline machine)
set -euo pipefail

ZIG_VERSION="0.15.1"
ZIG_TARBALL_SHA256="c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05"
ZIG_TARBALL_NAME="zig-x86_64-linux-${ZIG_VERSION}.tar.xz"
ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_TARBALL_NAME}"

OUTDIR="${1:-${PWD}/.zig-toolchain}"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
PREFIX="$OUTDIR/zig-${ZIG_VERSION}"
BIN="$PREFIX/zig"

zig_is_pinned() { [[ -x "$1" ]] && [[ "$("$1" version 2>/dev/null)" == "$ZIG_VERSION" ]]; }

# Already provisioned? Reuse it (idempotent; keeps repeat ships cheap).
if zig_is_pinned "$BIN"; then echo "$BIN"; exit 0; fi

# ---- 0) host check ----------------------------------------------------------
# The pin is a concrete x86_64-linux tarball. On any other host the honest answer
# is a loud failure with the escape hatch named, not a silently different compiler.
HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
HOST_OS="$(uname -s 2>/dev/null || echo unknown)"
if [[ -z "${ZIG_SRC_TARBALL:-}" && ( "$HOST_ARCH" != "x86_64" || "$HOST_OS" != "Linux" ) ]]; then
  echo "FATAL: the pinned Zig toolchain is x86_64-linux; this host is $HOST_ARCH/$HOST_OS." >&2
  echo "       Set ZIG_SRC_TARBALL= to a Zig ${ZIG_VERSION} tarball for this host, or put a" >&2
  echo "       zig ${ZIG_VERSION} on PATH. Note the shipping bitstream is only guaranteed" >&2
  echo "       reproducible on x86_64-linux." >&2
  exit 1
fi

# ---- 1) tarball: local copy if given, else fetch ----------------------------
TARBALL="${ZIG_SRC_TARBALL:-$OUTDIR/$ZIG_TARBALL_NAME}"
if [[ ! -f "$TARBALL" ]]; then
  command -v curl >/dev/null 2>&1 || {
    echo "FATAL: need curl to fetch $ZIG_URL, or set ZIG_SRC_TARBALL to a local copy." >&2
    echo "       (Deliberately NOT installing it — installing packages needs root," >&2
    echo "        which the prize committee's build-script ruling forbids inside a build script.)" >&2
    exit 1; }
  echo "== fetching zig ${ZIG_VERSION} (${ZIG_TARBALL_NAME}, ~51 MiB) ==" >&2
  curl -L --fail -o "$TARBALL.part" "$ZIG_URL"
  mv "$TARBALL.part" "$TARBALL"
fi

# ---- 2) verify the pin BEFORE unpacking -------------------------------------
GOT=$(sha256sum "$TARBALL" | cut -d' ' -f1)
[[ "$GOT" == "$ZIG_TARBALL_SHA256" ]] || {
  echo "FATAL: zig tarball sha256 mismatch — refusing to build the submission with it." >&2
  echo "  want $ZIG_TARBALL_SHA256" >&2
  echo "  got  $GOT  ($TARBALL)" >&2
  echo "  A different compiler emits different codegen and therefore a different archive9." >&2
  exit 1; }

# ---- 3) unpack (self-contained; nothing is installed system-wide) -----------
echo "== unpacking zig ${ZIG_VERSION} into $PREFIX ==" >&2
rm -rf "$PREFIX.part" "$PREFIX"
mkdir -p "$PREFIX.part"
tar xf "$TARBALL" -C "$PREFIX.part" --strip-components=1
mv "$PREFIX.part" "$PREFIX"

# ---- 4) prove it is the declared toolchain ----------------------------------
zig_is_pinned "$BIN" || {
  echo "FATAL: provisioned binary does not report zig $ZIG_VERSION:" >&2
  "$BIN" version 2>&1 | head -3 >&2
  exit 1; }
echo "$BIN"
