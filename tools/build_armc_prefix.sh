#!/bin/bash
# tools/build_armc_prefix.sh — build the SUBMITTED Form-1 prefix ("arm C", the
# minimal no-CLI dual-role program) from the checked-in recipe, with NO root, NO
# package manager and NO environment mutation.
#
# WHY THIS EXISTS:
#   The artifact of record does NOT ship the prefix that `zig build ship`
#   produces. It ships `zig build form1 ... -Dform1-malloc=true` put through a
#   five-step strip/objcopy/objcopy/upx chain — and until this script existed
#   that chain lived ONLY AS PROSE in a checklist
#   step A3. `construct_form1.sh --minimal-prefix` accepts the packed prefix as
#   an externally supplied PATH; it has never built one.
#   Measured consequence of following the documented scripted procedure instead:
#     plain ship prefix   144,588 B      arm-C prefix   137,028 B
#     plain comp9         384,545 B      comp9_min      376,985 B
#   decomp_bin is charged TWICE under Form-1, so the miss is
#   2 x (144,588 - 137,028) = +15,120 B of S — with every gate green and nothing
#   anywhere warning. That is exactly the class of manual step the
#   committee ruling ("Tightening up compression program build script
#   requirements") says must be incorporated into the build script:
#     "If a build procedure requires manual steps that could be in a build
#      script, the contestants be required to incorporate it into the build
#      script."
#   Same compliant-provisioner shape as tools/build_upx.sh: it is CALLED BY
#   construct_ship.sh (not by the evaluator), it never needs root, it exports
#   nothing, and it prints the path of the artifact it built and exits.
#
# WHY THE CHAIN IS WHAT IT IS (do not "improve" it — every step is scored):
#   1. `zig build form1 $ZMIX_ENGINE_FLAGS -Dform1-malloc=true $ABI_FLAG`
#        src/form1.zig is the dual-role program with NO command-line interface:
#        it self-extracts with no arguments (RULES.md rule 2) and encodes with
#        no arguments, selecting on its own 12-byte trailer. -Dform1-malloc lives
#        in form1's OWN options module (never in the shared model_opts)
#        and swaps the GPA for a raw-malloc allocator, which is what makes the
#        CLI-free image this small.
#   2. `strip -s`                       symbol table.
#   3. `objcopy -R .comment -R .note.gnu.property -R .note.ABI-tag
#               -R .eh_frame -R .eh_frame_hdr`
#        metadata + unwind sections that survive `strip -s`. They sit inside
#        PT_LOAD, so UPX compresses them; removing them shrinks the PACKED
#        prefix (~252 B measured on the ship engine), byte-identically.
#   4. `objcopy --strip-section-headers`
#        the Linux loader does not consult section headers and UPX does not need
#        them: ~392 B of linker metadata, doubled to ~784 B of S.
#   5. `upx $ZMIX_UPX_MODE`             the DECLARED packer, exact version.
#        upx 4.x cannot pack this engine at all (CantPackException). The mode is
#        a literal flag, never "auto": --lzma packs +232 B = +464 B of S, and the
#        committee books S off its OWN rebuild, so a host-dependent recipe is a
#        scoring hazard.
#
# A PREFIX SIZE IS TREE-DEPENDENT, so this script asserts no size of its
# own — it prints what it built. Pin an expectation with ZMIX_ARMC_SHA256 when
# reproducing a specific artifact of record.
#
# Usage:  tools/build_armc_prefix.sh [OUTDIR]   -> prints the packed prefix path
# Env:    ZIG / UPX          explicit, version-asserted tool paths (construct_ship.sh
#                            passes the ones it already resolved)
#         ZIG_SRC_TARBALL / UPX_SRC_TARBALL   offline provisioning, see the helpers
#         ZMIX_ARMC_SHA256   if set, the packed prefix must match it exactly
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 0) recipe: the SAME single source of truth construct_ship.sh reads ------
[[ -f SHIP_RECIPE.env ]] || { echo "FATAL: SHIP_RECIPE.env not found" >&2; exit 1; }
# shellcheck disable=SC1091
source ./SHIP_RECIPE.env
: "${ZMIX_ENGINE_FLAGS:?SHIP_RECIPE.env must set ZMIX_ENGINE_FLAGS}"
: "${ZMIX_ABI:?SHIP_RECIPE.env must set ZMIX_ABI}"
: "${ZMIX_UPX_MODE:?SHIP_RECIPE.env must set ZMIX_UPX_MODE}"

# src/form1.zig rejects -Dr1v2=true at compile time (the minimal prefix is R1v1
# only, and --minimal-prefix refuses `embedded`). Fail here with the reason
# rather than inside the Zig compiler.
if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dr1v2=true "* ]]; then
  echo "FATAL: SHIP_RECIPE.env carries -Dr1v2=true; the arm-C minimal prefix is R1v1-only." >&2
  echo "       An r1v2 recipe must ship the full-CLI prefix with --archive-order embedded." >&2
  exit 1
fi

case "$ZMIX_ABI" in
  glibc)  ABI_FLAG="-Dglibc=true" ;;
  static) ABI_FLAG="" ;;
  *) echo "FATAL: ZMIX_ABI='$ZMIX_ABI' (want glibc|static)" >&2; exit 1 ;;
esac

OUTDIR="${1:-${PWD}/run}"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
BIN="$OUTDIR/armc_prefix"

# ---- 0a) resolve the DECLARED compiler (same ladder as construct_ship.sh) ----
WANT_ZIG="${ZMIX_ZIG_VERSION:-0.15.1}"
zig_is_pinned() { [[ -x "$1" ]] && [[ "$("$1" version 2>/dev/null)" == "$WANT_ZIG" ]]; }
if [[ -n "${ZIG:-}" ]]; then
  zig_is_pinned "$ZIG" || {
    echo "FATAL: \$ZIG=$ZIG reports '$("$ZIG" version 2>&1 | head -1)', not zig $WANT_ZIG" >&2
    exit 1; }
elif zig_is_pinned "$(command -v zig 2>/dev/null || echo /nonexistent)"; then
  ZIG="$(command -v zig)"
elif zig_is_pinned "$HOME/.local/zig-$WANT_ZIG/zig"; then
  ZIG="$HOME/.local/zig-$WANT_ZIG/zig"
elif zig_is_pinned "$PWD/.zig-toolchain/zig-$WANT_ZIG/zig"; then
  ZIG="$PWD/.zig-toolchain/zig-$WANT_ZIG/zig"
else
  echo "== no zig $WANT_ZIG on this host — provisioning the declared toolchain ==" >&2
  ZIG="$(./tools/provision_zig.sh "$PWD/.zig-toolchain")" || {
    echo "FATAL: could not provision zig $WANT_ZIG (see tools/provision_zig.sh)" >&2; exit 1; }
fi
GOT_ZIG="$("$ZIG" version 2>/dev/null || true)"
[[ "$GOT_ZIG" == "$WANT_ZIG" ]] || {
  echo "FATAL: resolved compiler is zig '$GOT_ZIG', want EXACTLY '$WANT_ZIG' ($ZIG)" >&2; exit 1; }

# ---- 0b) resolve the DECLARED packer (same ladder as construct_ship.sh) ------
# ⚠ upx reads the environment variable `UPX` as a DEFAULT-OPTIONS STRING. Passing
# the packer's PATH in a variable of that name — which is exactly what
# construct_ship.sh does, and exactly what README.md documents as the offline
# override `UPX=/path/to/upx` — makes upx abort at pack time with
#   "invalid string '/usr/bin/upx' in environment variable 'UPX'".
# It does NOT abort on --version, so a version assertion passes and the failure
# surfaces only when real work starts. Every invocation therefore goes through
# `env -u UPX`, which makes the packer immune to the caller's environment.
WANT_UPX="${ZMIX_UPX_VERSION:-5.2.0}"
upx() { env -u UPX "$UPX" "$@"; }
# ⚠ PIPE-FREE ON PURPOSE.  `cmd | head -1 | grep -q ...` under `set -o pipefail`
#   is a SIGPIPE RACE: grep -q exits on the first match, head and the writer die of
#   SIGPIPE, and the pipeline reports 141 — a FALSE FAILURE on a CORRECT tool.  That
#   exact shape produced a false `objcopy lacks --strip-section-headers` FATAL that
#   killed a judge-harness run.  A command substitution has no pipe to race on.
upx_is_pinned() { local _v; [[ -x "$1" ]] && _v="$(env -u UPX "$1" --version 2>/dev/null)" \
                   && [[ "${_v%%$'\n'*}" == *"upx $WANT_UPX"* ]]; }
if [[ -n "${UPX:-}" ]]; then
  upx_is_pinned "$UPX" || { echo "FATAL: \$UPX=$UPX is not upx $WANT_UPX" >&2; exit 1; }
elif upx_is_pinned "$(command -v upx 2>/dev/null || echo /nonexistent)"; then
  UPX="$(command -v upx)"
else
  echo "== no upx $WANT_UPX on PATH — building the declared packer from source ==" >&2
  UPX="$(./tools/build_upx.sh "$PWD/.upx-build")" || {
    echo "FATAL: could not provision upx $WANT_UPX (see tools/build_upx.sh)" >&2; exit 1; }
fi

for t in strip objcopy; do
  command -v "$t" >/dev/null 2>&1 || {
    echo "FATAL: '$t' (GNU binutils) not found — the prefix chain needs strip + objcopy." >&2
    echo "       Deliberately NOT installing it: a build script may not require root" >&2
    echo "       (the prize committee's ruling on build-script requirements)." >&2
    exit 1; }
done

echo "== arm-C minimal Form-1 prefix ==" >&2
echo "zig: $ZIG (version $GOT_ZIG, pinned)   upx: $(upx --version | head -1)" >&2

# ---- 1) build the CLI-free dual-role program --------------------------------
# shellcheck disable=SC2086  # word-splitting of the recipe flags is intended
echo "== [1/5] zig build form1 \$ZMIX_ENGINE_FLAGS -Dform1-malloc=true $ABI_FLAG ==" >&2
"$ZIG" build form1 $ZMIX_ENGINE_FLAGS -Dform1-malloc=true $ABI_FLAG 1>&2

[[ -f zig-out/bin/zmix_form1 ]] || {
  echo "FATAL: zig build form1 produced no zig-out/bin/zmix_form1" >&2; exit 1; }
# Work on a copy so zig-out stays a clean build product (stripping in place makes
# a later incremental build's freshness check meaningless).
cp zig-out/bin/zmix_form1 "$BIN"
chmod u+w "$BIN"

# ---- 2) strip -s ------------------------------------------------------------
echo "== [2/5] strip -s ==" >&2
strip -s "$BIN"

# ---- 3) drop metadata + unwind sections (inside PT_LOAD; UPX compresses them) -
echo "== [3/5] objcopy -R .comment .note.gnu.property .note.ABI-tag .eh_frame[_hdr] ==" >&2
objcopy -R .comment -R .note.gnu.property -R .note.ABI-tag -R .eh_frame -R .eh_frame_hdr "$BIN"

# ---- 3b) RECIPROCAL-ESTIMATE GATE ------------------------------------------
# Reciprocal-estimate determinism gate.
# The libm determinism gate in construct_ship.sh checks UNDEFINED SYMBOLS, so it
# is structurally blind to this: VRCPPS/VRSQRTPS are inline instructions LLVM
# emits under `@setFloatMode(.optimized)`, not libcalls. Their low mantissa bits
# come from a vendor-defined on-die table, so an encode on our silicon and a
# decode on the judge's AMD part desynchronise the arithmetic coder. This is the
# documented cause of the fx2-cmix-transformer July 2026 submission failure.
# Run HERE, BEFORE the section-header strip and before UPX, because after either
# one `objdump -d` returns a FALSE CLEAN.
echo "== [3b/5] reciprocal-estimate gate ==" >&2
if [[ -x "$(dirname "$0")/recip_gate.sh" ]]; then
    if ! "$(dirname "$0")/recip_gate.sh" "$BIN" >&2; then
        echo "FATAL: the prefix contains a reciprocal-ESTIMATE opcode — it would desync the" >&2
        echo "  coder on a different-vendor judge machine. Route the division/sqrt through" >&2
        echo "  src/strict_fp.zig. This gate is NOT advisory." >&2
        exit 1
    fi
else
    echo "FATAL: tools/recip_gate.sh missing — refusing to build an ungated prefix." >&2
    exit 1
fi

# ---- 3c) C++ EXCEPTION/UNWIND/DEMANGLE GATE (transformer-era) -------------------------
# `-Dtransformer` links libc++. If an exception-built archive member is ever
# extracted it drags in `cxa_default_handlers.o`, whose translation unit IS the
# entire Itanium demangler — 30,868 B of PACKED prefix = 61,736 B of S, with
# every other gate green. `src/cxx_local.cpp` keeps those members out by
# mangled name, which a Zig/libc++ bump can silently break. Same placement rule
# as the reciprocal gate above: BEFORE the section-header strip and BEFORE UPX,
# or the check returns a FALSE CLEAN. Inert on LSTM-era (nothing to find).
echo "== [3c/5] C++ unwind/demangle gate ==" >&2
if [[ -x "$(dirname "$0")/check_no_cxx_unwind.sh" ]]; then
    if ! "$(dirname "$0")/check_no_cxx_unwind.sh" "$BIN" >&2; then
        echo "FATAL: libc++abi exception/unwind/demangle machinery was linked into the" >&2
        echo "  prefix. See src/cxx_local.cpp — a libc++ symbol it supplies has been" >&2
        echo "  renamed, so the reference fell back to the archive. This gate is NOT" >&2
        echo "  advisory: decomp_bin is charged TWICE." >&2
        exit 1
    fi
else
    echo "FATAL: tools/check_no_cxx_unwind.sh missing — refusing to build an ungated prefix." >&2
    exit 1
fi

# ---- 3d) LIBM IMPORT GATE (transformer-era) ------------------------------------------
# The reciprocal gate above greps OPCODES and is structurally blind to an
# imported SYMBOL. `model_opt.cpp`'s RoPE beyond-table fallback called
# std::sin/std::cos, which GCC contracts into `sincosf@GLIBC_2.2.5` — a
# coder-path value taken from the judging machine's libm, present in earlier
# transformer-era prefixes and caught by no gate that then existed
# Same placement rule as
# 3b/3c: BEFORE the section-header strip and BEFORE UPX, or `nm -D` prints
# "no symbols" and the gate returns a FALSE CLEAN — which it refuses to do.
echo "== [3d/5] libm import gate ==" >&2
if [[ -x "$(dirname "$0")/libm_gate.sh" ]]; then
    if ! "$(dirname "$0")/libm_gate.sh" "$BIN" >&2; then
        echo "FATAL: the prefix IMPORTS a math function from the system libm. archive9.exe" >&2
        echo "  is encoded on one machine and self-extracts on another; two libm versions" >&2
        echo "  that differ in one mantissa bit desynchronise the arithmetic decoder." >&2
        echo "  Vendor it (src/vendored_math.zig, third_party/fx2_transformer/rope_trig.h)." >&2
        echo "  This gate is NOT advisory." >&2
        exit 1
    fi
else
    echo "FATAL: tools/libm_gate.sh missing — refusing to build an ungated prefix." >&2
    exit 1
fi

# ---- 4) drop section headers (loader never reads them; UPX does not need them) -
echo "== [4/5] objcopy --strip-section-headers ==" >&2
objcopy --strip-section-headers "$BIN"

RAW=$(wc -c < "$BIN")
RAW_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
echo "arm-C prefix (stripped, pre-UPX): $RAW bytes  sha256=$RAW_SHA" >&2

# ---- 5) pack with the DECLARED packer ---------------------------------------
echo "== [5/5] upx $ZMIX_UPX_MODE ==" >&2
case "$ZMIX_UPX_MODE" in
  auto)
    echo "WARNING: ZMIX_UPX_MODE=auto is host-dependent (--ultra-brute vs --lzma differ" >&2
    echo "         by 232 B = 464 B of S). Pin a literal mode before shipping." >&2
    UPX_USED=--ultra-brute ;;
  *) UPX_USED="$ZMIX_UPX_MODE" ;;
esac
# shellcheck disable=SC2086
upx $UPX_USED -q "$BIN" >/dev/null 2>"$OUTDIR/armc_upx.err" || {
  if [[ "$ZMIX_UPX_MODE" == auto ]]; then
    echo "upx --ultra-brute failed; falling back to --lzma" >&2
    UPX_USED=--lzma
    upx $UPX_USED -q "$BIN" >/dev/null 2>"$OUTDIR/armc_upx.err" || {
      echo "FATAL: upx --lzma failed too:" >&2; cat "$OUTDIR/armc_upx.err" >&2; exit 1; }
  else
    echo "FATAL: upx $UPX_USED failed:" >&2; cat "$OUTDIR/armc_upx.err" >&2; exit 1
  fi
}
chmod 0755 "$BIN"

# ---- 6) checklist A3's own PASS criterion -----------------------------------
upx -t -q "$BIN" >/dev/null 2>&1 || {
  echo "FATAL: 'upx -t' does not verify the packed prefix — refusing to emit it." >&2; exit 1; }

PACKED=$(wc -c < "$BIN")
PACKED_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
echo "arm-C prefix (packed, = submitted decomp_bin): $PACKED bytes  sha256=$PACKED_SHA  (upx -t OK)" >&2
echo "  NOTE: decomp_bin is charged TWICE under Form-1 — this prefix is worth $((2 * PACKED)) B of S." >&2

if [[ -n "${ZMIX_ARMC_SHA256:-}" ]]; then
  [[ "$PACKED_SHA" == "$ZMIX_ARMC_SHA256" ]] || {
    echo "FATAL: packed prefix sha256 mismatch against ZMIX_ARMC_SHA256" >&2
    echo "  want $ZMIX_ARMC_SHA256" >&2
    echo "  got  $PACKED_SHA  ($PACKED bytes)" >&2
    echo "  A prefix is TREE-dependent as well as recipe-dependent: check the commit," >&2
    echo "  SHIP_RECIPE.env, and the zig/upx pins before assuming a code change." >&2
    exit 1; }
  echo "packed prefix matches ZMIX_ARMC_SHA256 exactly" >&2
fi

echo "$BIN"
