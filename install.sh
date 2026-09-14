#!/bin/bash
# submission/hpja/install.sh — the HPJA dependency phase for the zmix Form-1 entry.
#
# CONTRACT (ENTRANT_INSTRUCTIONS.md §4, quoted verbatim):
#   "This is the only entrant-controlled root/network phase. It runs while an
#    isolated dependency image is built. It receives no source tree and must not
#    build the entry. Use it only to install system dependencies. It must be
#    noninteractive and repeatable. After it finishes, no entrant stage receives
#    network access or root privileges."
#
# It is COPYed alone into a `docker build` context (the orchestrator writes
# `.dockerignore` = "*\n!install.sh"), so THERE IS NO SOURCE TREE HERE: this
# script cannot call tools/provision_zig.sh or tools/build_upx.sh, because those
# files do not exist in this stage. What it can do — and does — is STAGE the two
# declared toolchain tarballs into /opt/toolchain-src, where the offline
# build.sh points the in-tree provisioners' own ZIG_SRC_TARBALL / UPX_SRC_TARBALL
# escape hatches at them. The sha256 pins stay in the source package
# (tools/provision_zig.sh, tools/build_upx.sh) and are enforced THERE; pinning
# them a second time here would create two constants that can drift apart.
#
# ⛔ ON `sudo apt-get`: the judge REJECTS build scripts that shell out to
# `sudo apt-get` (Google-Group thread "Tightening up compression
# program build script requirements"; his pasted bad example is a UPX installer
# of exactly that shape). That rejection is about BUILD scripts acquiring root.
# This file is the harness's OWN sanctioned root phase, it runs as root by
# construction, and it therefore uses plain `apt-get` with NO `sudo` anywhere.
# The zmix build proper (build.sh -> construct_ship.sh) needs no root and no
# package manager at all: the compiler is an unpacked upstream tarball
# (tools/provision_zig.sh) and the packer is compiled from source without cmake
# (tools/build_upx.sh).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# The judging base image replaces /tmp with a symlink to /work/run/tmp, which is
# torn down again by the dependency Dockerfile. Anchor every temp write in a
# directory that exists for the whole of this stage.
export TMPDIR=/var/tmp

# ---- 1) system dependencies -------------------------------------------------
# Each package is here because a named build step needs it. ENTRANT_INSTRUCTIONS
# does not promise any of them, and the base image (ubuntu:22.04 + curl,
# ca-certificates, libarchive-tools, time, util-linux) supplies almost none.
#   gcc, g++, make  tools/build_upx.sh compiles upx 5.2.0 from C++ source
#   xz-utils        both toolchain tarballs are .tar.xz
#   curl, ca-certs  the two fetches below (already in the base image; named so
#                   this script does not depend on that staying true)
#   texinfo,        GNU binutils 2.42 configure/build prerequisites
#   bison, flex
# coreutils, tar, findutils, grep, sed and bash are Ubuntu `Essential:yes` and
# are not requested. `nproc` (used by tools/build_upx.sh) is in coreutils.
apt-get update
apt-get install --yes --no-install-recommends \
  gcc g++ make xz-utils curl ca-certificates texinfo bison flex
rm -rf /var/lib/apt/lists/*

# ---- 2) GNU binutils 2.42 ---------------------------------------------------
# BOTH scored prefixes go through `objcopy --strip-section-headers`, which
# binutils gained in 2.41. The judging image is ubuntu:22.04 = binutils 2.38, so
# the stock toolchain CANNOT build this entry: construct_ship.sh §2 and
# tools/build_armc_prefix.sh step 4 would both fail.
# 2.42 is one of the two binutils versions the recipe's byte-identity has been
# verified against (the other is 2.46.1 on the development box; the two produce
# byte-identical stripped engines and byte-identical packed prefixes).
# /usr/local/bin precedes /usr/bin on the default PATH, so this becomes the
# strip/objcopy/objdump the build resolves.
BINUTILS_VERSION=2.42
BINUTILS_SHA=f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800
curl -sSL --fail -o "$TMPDIR/binutils.tar.xz" \
  "https://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.xz"
echo "$BINUTILS_SHA  $TMPDIR/binutils.tar.xz" | sha256sum -c -
mkdir -p "$TMPDIR/bu" "$TMPDIR/bu-build"
tar xf "$TMPDIR/binutils.tar.xz" -C "$TMPDIR/bu" --strip-components=1
cd "$TMPDIR/bu-build"
"$TMPDIR/bu/configure" --prefix=/usr/local --disable-nls --disable-gdb \
  --disable-werror --disable-multilib >/dev/null
make -j"$(nproc)" all-binutils >/dev/null
make install-binutils >/dev/null
cd /
rm -rf "$TMPDIR/bu" "$TMPDIR/bu-build" "$TMPDIR/binutils.tar.xz"
hash -r

# Assert the three binutils programs the build actually invokes, by capability
# rather than by version string. `objdump` is not optional decoration: it is
# construct_ship.sh's §2b determinism gate, the check that keeps a transcendental
# from binding to the judge's libm instead of the vendored one.
objcopy --version | head -1
[[ "$(objcopy --help 2>&1)" == *--strip-section-headers* ]] \
  || { echo "FATAL: objcopy still lacks --strip-section-headers" >&2; exit 1; }
command -v strip   >/dev/null || { echo "FATAL: strip missing" >&2; exit 1; }
command -v objdump >/dev/null || { echo "FATAL: objdump missing (determinism gate)" >&2; exit 1; }

# ---- 3) stage the two DECLARED toolchain tarballs ---------------------------
# build.sh runs offline, so the bytes must be here before this stage ends. They
# are NOT verified here on purpose: tools/provision_zig.sh and
# tools/build_upx.sh carry the authoritative sha256 pins
#   zig 0.15.1  c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05
#   upx 5.2.0   af99e526d5759de94412aea1104d5e4ca406cb725295f8633ecc9e843dc1ce1c
# and refuse to proceed on a mismatch. A duplicate pin here would be a second
# constant that can silently disagree with the one that is actually enforced.
mkdir -p /opt/toolchain-src
curl -sSL --fail -o /opt/toolchain-src/zig-x86_64-linux-0.15.1.tar.xz \
  https://ziglang.org/download/0.15.1/zig-x86_64-linux-0.15.1.tar.xz
curl -sSL --fail -o /opt/toolchain-src/upx-5.2.0-src.tar.xz \
  https://github.com/upx/upx/releases/download/v5.2.0/upx-5.2.0-src.tar.xz
chmod -R a+rX /opt/toolchain-src

echo "install.sh complete: binutils $(objcopy --version | head -1 | awk '{print $NF}'), toolchain tarballs staged in /opt/toolchain-src"
ls -l /opt/toolchain-src
