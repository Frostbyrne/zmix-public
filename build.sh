#!/bin/bash
# submission/hpja/build.sh — the HPJA offline build phase for the zmix Form-1 entry.
#
# CONTRACT (ENTRANT_INSTRUCTIONS.md §5, quoted verbatim):
#   "build.sh runs offline as UID/GID 65532 in a fresh container. The unpacked
#    source tree is mounted read-only at /entry; the current directory /work is
#    writable. It must write the executable basename(s) declared by COMPRESSOR
#    and, for the Relaxations form, DECOMPRESSOR, directly into /work.
#    Only those declared files are returned to the host orchestrator. A larger
#    helper left elsewhere in /work cannot be reached by the formal execution
#    container."
#
# ★★★ THE ONE THING THIS FILE EXISTS TO GET RIGHT ★★★
# HPJA scores the `comp9` THIS SCRIPT EMITS (judging_assistance.sh:530,545:
#   compressor_bytes=$(stat --format='%s' "$run_results/generated/$HP_COMPRESSOR")
#   formal_total_bytes=$((compressor_bytes + generated_archive_bytes
#                         + command_line_bytes + ...)) ).
# `construct_ship.sh` emits run/comp9 = the FULL-CLI HELPER, 384,545 B. The
# artifact of record's scored compressor is `comp9_min` = 376,985 B, and it is a
# CONCATENATION that until now happened only inside construct_form1.sh — which
# additionally wants enwik9, a file this container does not have and must not
# have. So this script performs the concatenation itself, from the components
# construct_ship.sh just built.
# Under Form-1 the difference is 2 x (144,588 - 137,028) = +15,120 B of S,
# because decomp_bin is charged TWICE (once inside comp9, once inside
# archive9.exe), and EVERY GATE STAYS GREEN EITHER WAY. Emitting run/comp9 here
# would silently cost 15,120 bytes with nothing anywhere warning.
set -euo pipefail

say() { printf '\n=== %s ===\n' "$*"; }

# ---- 0) environment ---------------------------------------------------------
# HOME IS UNSET in this container (--user 65532:65532, /nonexistent home). Zig
# falls back to $HOME/.cache/zig for BOTH of its caches and dies without it, so
# both are redirected under the one writable mount before anything else runs.
# /tmp is a symlink to /work/run/tmp in the judging base image and /work/run does
# not exist in the build container, so TMPDIR is redirected too.
export HOME=/work
export TMPDIR=/work/tmp
export ZIG_GLOBAL_CACHE_DIR=/work/.zig-global
export ZIG_LOCAL_CACHE_DIR=/work/.zig-local
mkdir -p "$ZIG_GLOBAL_CACHE_DIR" "$ZIG_LOCAL_CACHE_DIR" "$TMPDIR" /work/src

# /entry is READ-ONLY and the build writes into its own tree (zig-out/, run/,
# and PPMd's ppm.<pid>.temp arena), so work on a copy.
cp -a /entry/. /work/src/
cd /work/src

say "provenance of the source tree being built"
if [[ -f PACKAGE_PROVENANCE.txt ]]; then cat PACKAGE_PROVENANCE.txt; else
  echo "(PACKAGE_PROVENANCE.txt absent — package not built by submission/hpja/make_package.sh)"
fi
# ✅ — EXPECT "engine source pin OK: <digest>" FROM construct_ship.sh.
# SHIP_RECIPE.env's pin is the git-free ZMIX_SRC_DIGEST (every file under src/ and
# third_party/, plus build.zig and build.zig.zon), so it is computable HERE, inside
# a tarball with no .git, and ZMIX_TREE_PIN_MODE is `strict` (it is assigned inside
# SHIP_RECIPE.env, which construct_ship.sh SOURCES, so the environment cannot change
# it in either direction): a tree that does not match the recipe now ABORTS rc=3
# instead of warning and continuing.
#   HISTORICAL: this used to read "EXPECT A LOUD TREE PIN MISMATCH ... building
#   unknown", because the pin was the commit sha ZMIX_TREE_SHA=874514e checked with
#   `git rev-parse HEAD`, which can only read "unknown" in a source package — the
#   exact reason strict was unusable and the guard sat disarmed. ZMIX_TREE_SHA is
#   retained as informational provenance and is no longer asserted anywhere.
# The pin is ALSO discharged on the ENTRANT side — make_package.sh asserts the same
# digest on the STAGED tree — and this script still adds a strictly STRONGER,
# git-free check at the end: the sha256 of the emitted comp9, which covers the
# source tree, the compiler, the packer and the recipe at once.

# ---- 1) resolve the DECLARED toolchain, offline -----------------------------
# install.sh staged both tarballs; these are the in-tree provisioners' own
# offline escape hatches, so THEIR sha256 pins are what gets enforced.
export ZIG_SRC_TARBALL=/opt/toolchain-src/zig-x86_64-linux-0.15.1.tar.xz
export UPX_SRC_TARBALL=/opt/toolchain-src/upx-5.2.0-src.tar.xz
for t in "$ZIG_SRC_TARBALL" "$UPX_SRC_TARBALL"; do
  [[ -f "$t" ]] || { echo "FATAL: install.sh did not stage $t" >&2; exit 1; }
done

say "provisioning zig 0.15.1 (declared compiler pin)"
ZIGBIN="$(bash ./tools/provision_zig.sh /work/.zig-toolchain)"
say "building upx 5.2.0 from source (no root, no cmake, no package manager)"
UPXBIN="$(bash ./tools/build_upx.sh /work/.upx-build)"

# Resolve BOTH through PATH and export NEITHER by name.
# ⚠ `upx` parses the environment variable UPX as a default-OPTIONS STRING, not as
# a path: exporting UPX=/path/to/upx makes every real pack die with
# "invalid string '/path/to/upx' in environment variable 'UPX'" — AFTER the
# version assertion has passed, because `UPX=... upx --version` returns rc=0.
# Both in-tree scripts now invoke the packer through `env -u UPX`, but the
# cheapest defence is not to set the variable at all.
export PATH="$(dirname "$ZIGBIN"):$(dirname "$UPXBIN"):$PATH"
unset ZIG UPX
say "declared toolchain"
zig version
env -u UPX upx --version | head -1
for t in strip objcopy objdump; do
  command -v "$t" >/dev/null || { echo "FATAL: '$t' (GNU binutils) missing — install.sh should have built binutils 2.42" >&2; exit 1; }
done
[[ "$(objcopy --help 2>&1)" == *--strip-section-headers* ]] \
  || { echo "FATAL: objcopy lacks --strip-section-headers (needs binutils >= 2.41)" >&2; exit 1; }

# ---- 2) build the engine, both prefixes and the coded assets ----------------
# construct_ship.sh does, in one pass: build the engine from SHIP_RECIPE.env's
# 38-flag line; strip + fingerprint it; pack the full-CLI helper prefix; call
# tools/build_armc_prefix.sh for the SUBMITTED arm-C prefix; self-compress the
# recipe blob and the s7 article order with the engine's own no-dict codec; mint
# the 12-byte header; assemble the helper run/comp9; and run the MANDATORY
# --extract-assets round-trip that proves the packaged tail decodes back to the
# checked-in originals byte-for-byte. It needs no root, no network and no
# package manager. Budget ~30-60 min: the asset self-compression runs the full
# multi-GB predictor twice (once to compress, once to verify).
say "construct_ship.sh"
bash ./construct_ship.sh

# ---- 3) EMIT THE SCORED comp9 (comp9_min), not the helper -------------------
# comp9_min = armc_prefix ++ comp_<recipe|dict>.<fp8> ++ comp_order.<fp8> ++ header.dat
# The asset cache filenames are keyed on the first 8 hex of the ENGINE
# FINGERPRINT (construct_ship.sh:250-251: FP=sha256(run/engine); FP8=${FP:0:8}),
# so it is derived here the same way rather than globbed or hardcoded.
say "assembling the SCORED comp9 (comp9_min)"
[[ -f run/engine ]] || { echo "FATAL: run/engine absent — construct_ship.sh did not complete" >&2; exit 1; }
FP8="$(sha256sum run/engine | cut -c1-8)"

# The dict slot holds the Arm E' RECIPE blob under -Dderivdict=true (the
# dictionary is re-derived from enwik9 at encode time and therefore charged ONCE
# instead of twice) and english.dic.comp otherwise. construct_ship.sh selects it
# from the recipe; select it the same way, from the same single source of truth.
# shellcheck disable=SC1091
ENGINE_FLAGS="$(bash -c 'source ./SHIP_RECIPE.env; printf "%s" "$ZMIX_ENGINE_FLAGS"')"
if [[ " $ENGINE_FLAGS " == *" -Dderivdict=true "* ]]; then
  SLOT_KIND=recipe
else
  SLOT_KIND=dict
fi
ARMC=run/armc_prefix
CD="run/comp_${SLOT_KIND}.${FP8}"
CO="run/comp_order.${FP8}"
HDR=run/header.dat
# ⛔⛔ transformer-era: comp9 CARRIES THE TRANSFORMER WEIGHTS BLOB, and omitting it emits a
# compressor that cannot compress. construct_ship.sh:648 assembles
#   comp9 = decomp_bin ++ comp_<slot> ++ comp_order ++ comp_tfweights ++ header
# and `predictor_lex` loads the weights from the running artifact's OWN TAIL
# (Transformer.embedded_blob, set by ship.zig). This script concatenated only
# four components until which on a transformer-era pin is fatal TWICE OVER:
# the pin's own component identity would abort the judge's build, and if anyone
# "repaired" the pin to match the four-part sum the judge would be handed a
# 417,584 B comp9 with an EMPTY weights segment — a valid-looking executable
# that dies at encode, with S booked off it. Unset on an LSTM-era recipe, where the
# file does not exist and the four-part formula is correct.
# The blob is copied VERBATIM (not self-compressed), so it has no ${FP8} key.
# GEN7 is derived from the RECIPE, not from which files happen to exist, because
# the recipe is the thing the pin describes: a missing comp_tfweights next to a
# transformer-era recipe is a BUILD FAILURE, and a present one next to an LSTM-era recipe is a
# stale artifact that must not be concatenated. Both are caught below.
GEN7=0
if [[ " $ENGINE_FLAGS " == *" -Dtransformer=true "* ]]; then GEN7=1; fi
TFW=run/comp_tfweights
TFW_ARGS=()
if [[ -f "$TFW" ]]; then
  (( GEN7 == 1 )) || { echo "FATAL: $TFW exists but the recipe is not transformer-era (-Dtransformer absent)." >&2
    echo "  A stale weights blob from an earlier build would be concatenated into comp9," >&2
    echo "  inflating S by megabytes against a pin that cannot describe it. Refusing." >&2
    exit 1; }
  TFW_ARGS=("$TFW")
elif (( GEN7 == 1 )); then
  echo "FATAL: the recipe is transformer-era (-Dtransformer=true) but $TFW is absent." >&2
  echo "  comp9 would be assembled with NO weights segment: it links, runs, and" >&2
  echo "  cannot encode, because the predictor reads its weights from comp9's own" >&2
  echo "  tail. Refusing to emit a compressor that cannot compress." >&2
  exit 1
fi

if [[ ! -f "$ARMC" ]]; then
  echo "FATAL: run/armc_prefix is missing." >&2
  echo "  The SUBMITTED decomp_bin is the arm-C minimal prefix, not the prefix that" >&2
  echo "  'zig build ship' produces. Emitting run/comp9 instead costs +15,120 B of S" >&2
  echo "  (decomp_bin is charged twice under Form-1) with every gate still green." >&2
  echo "  Refusing to emit a silently 15,120-byte-worse entry." >&2
  exit 1
fi
for f in "$CD" "$CO" "$HDR"; do
  [[ -f "$f" ]] || { echo "FATAL: expected component missing: $f" >&2; ls -l run/ >&2; exit 1; }
done

cat "$ARMC" "$CD" "$CO" ${TFW_ARGS[0]+"${TFW_ARGS[@]}"} "$HDR" > /work/comp9
chmod 0755 /work/comp9

A=$(wc -c < "$ARMC"); D=$(wc -c < "$CD"); O=$(wc -c < "$CO"); H=$(wc -c < "$HDR")
T=0; [[ ${#TFW_ARGS[@]} -eq 0 ]] || T=$(wc -c < "$TFW")
COMP9=$(wc -c < /work/comp9)
COMP9_SHA=$(sha256sum /work/comp9 | cut -d' ' -f1)
SUM=$((A + D + O + T + H))
[[ "$SUM" == "$COMP9" ]] || { echo "FATAL: component sum $SUM != |comp9| $COMP9 — assembly bug" >&2; exit 1; }

# Cross-check against construct_ship.sh's OWN arithmetic, which it printed into
# run/MANIFEST.txt from the same two prefixes it had just built. Two independent
# derivations of the same number must agree.
MAN_LINE="$(grep -m1 '|comp9_min| =' run/MANIFEST.txt || true)"
MAN_TOTAL="$(printf '%s' "$MAN_LINE" | sed -n 's/.*= \([0-9]\+\) bytes.*/\1/p')"
if [[ -n "$MAN_TOTAL" ]]; then
  [[ "$MAN_TOTAL" == "$COMP9" ]] || {
    echo "FATAL: run/MANIFEST.txt says comp9_min is $MAN_TOTAL bytes, assembled $COMP9" >&2; exit 1; }
  echo "cross-check OK against run/MANIFEST.txt: $MAN_LINE"
else
  echo "WARNING: run/MANIFEST.txt carries no |comp9_min| line to cross-check against" >&2
fi

# ---- 4) report against the artifact of record -------------------------------
# ⚠ THESE FIVE CONSTANTS ARE TREE-DEPENDENT AND MOVE TOGETHER (a
# recipe change moves decomp_bin AND comp_order AND comp_recipe, because the
# assets are RE-MINTED by the final engine).  They therefore live in exactly ONE
# place — ARTIFACT_PIN.env, staged at the package root by make_package.sh — so a
# new artifact is a single edit.  The inline values below are a FALLBACK for
# packages cut before that file existed; if it is present it always wins, and the
# build says which source it used.
ARTIFACT_LABEL="zmix"
ARTIFACT_TAG="zmix-1.0"
EXPECT_ARMC=160884
EXPECT_CD=65993
EXPECT_CO=173640
EXPECT_TFW=2815630
EXPECT_HDR=16
EXPECT_COMP9=3216163
EXPECT_COMP9_SHA=0fbc9cd2f4414785d21457d9e2cc2f844e2132f9e5693cbc0288039911eda9c3
# CWD is /work/src (step 0), so this is the package's own copy — relative on
# purpose: an absolute build-machine path in a shipped binary or build script is
# a submission defect that has bitten this project twice.
PIN_SOURCE="inline fallback in build.sh"
if [[ -f ./ARTIFACT_PIN.env ]]; then
  # shellcheck disable=SC1091
  source ./ARTIFACT_PIN.env
  PIN_SOURCE="ARTIFACT_PIN.env (packaged)"
fi
say "artifact pin ($PIN_SOURCE)"
echo "  artifact:            ${ARTIFACT_LABEL:-<unset>} (${ARTIFACT_TAG:-<unset>})"
echo "  runtime_exec_policy: ${RUNTIME_EXEC_POLICY:-strict} (ENTRANT_INSTRUCTIONS.md 7; harness default)"
echo "  hpja_pin:            ${HPJA_PIN:-<unset>}"
for v in EXPECT_ARMC EXPECT_CD EXPECT_CO EXPECT_COMP9 EXPECT_COMP9_SHA; do
  [[ -n "${!v:-}" ]] || { echo "FATAL: $v is unset — ARTIFACT_PIN.env is incomplete" >&2; exit 1; }
done
# EXPECT_TFW is transformer-era-only and defaults to 0, so an LSTM-era pin needs no new key —
# but a transformer-era pin that OMITS it would silently self-validate against a 4-component
# identity that cannot describe the artifact (it would agree with itself while
# describing nothing). Require it exactly when the recipe says there is a weights
# segment, and forbid a non-zero one when it says there is not.
if (( GEN7 )); then
  [[ -n "${EXPECT_TFW:-}" && "${EXPECT_TFW}" != 0 ]] || {
    echo "FATAL: this is a transformer-era recipe (-Dtransformer=true) but ARTIFACT_PIN.env carries no" >&2
    echo "  non-zero EXPECT_TFW. The pin would validate a 4-component comp9 against a" >&2
    echo "  5-component artifact and agree with itself while describing nothing." >&2
    exit 1; }
else
  [[ -z "${EXPECT_TFW:-}" || "${EXPECT_TFW}" == 0 ]] || {
    echo "FATAL: ARTIFACT_PIN.env sets EXPECT_TFW=$EXPECT_TFW but the recipe is not transformer-era" >&2; exit 1; }
fi
# The component identity must hold for the pin itself, or the pin is internally
# inconsistent and every comparison below is against a number nothing produced.
# EXPECT_TFW defaults to 0 and EXPECT_HDR to 12 so an LSTM-era pin validates
# unchanged. ⚠ BOTH move on transformer-era: comp9 carries the weights blob, and
# HeaderInfo grows a FOURTH field (tfweights_size), so header.dat is 16 bytes,
# not 12 (construct_ship.sh:645, "transformer-era adds the fourth field"). Hardcoding 12
# made this check reject a CORRECT transformer-era pin by exactly 4 bytes.
[[ $((EXPECT_ARMC + EXPECT_CD + EXPECT_CO + ${EXPECT_TFW:-0} + ${EXPECT_HDR:-12})) == "$EXPECT_COMP9" ]] || {
  echo "FATAL: ARTIFACT_PIN.env is self-inconsistent: $EXPECT_ARMC + $EXPECT_CD + $EXPECT_CO + ${EXPECT_TFW:-0} + ${EXPECT_HDR:-12} != $EXPECT_COMP9" >&2
  echo "  On a transformer-era (-Dtransformer) pin this usually means EXPECT_TFW is missing" >&2
  echo "  (comp9 carries the weights blob) or EXPECT_HDR is not 16." >&2
  exit 1; }
# ⛔ The trailer width is a SHAPE invariant, not decoration, and it is checked
# against the RECIPE as well as against the pin. LSTM-era writes 12 bytes (dict,order,payload),
# transformer-era writes 16 (+tfweights_size). A 12-byte read of a 16-byte trailer
# mis-slices EVERY segment, and construct_form1.sh:45 records that the failure
# mode is a plausible-looking WRONG NUMBER rather than an error — so a warning is
# not enough here: it is exactly the class that ships silently.
WANT_HDR=12; (( GEN7 == 0 )) || WANT_HDR=16
[[ "$H" == "$WANT_HDR" ]] || {
  echo "FATAL: the recipe implies a $WANT_HDR-byte trailer (transformer=$GEN7) but run/header.dat is $H bytes." >&2
  echo "  The engine and this script disagree about the container shape; refusing to emit." >&2
  exit 1; }
[[ "${EXPECT_HDR:-12}" == "$WANT_HDR" ]] || {
  echo "FATAL: ARTIFACT_PIN.env declares EXPECT_HDR=${EXPECT_HDR:-12} but the recipe implies $WANT_HDR." >&2
  echo "  The pin and the recipe describe different container shapes; one of them is stale." >&2
  exit 1; }

say "SCORED COMPRESSOR"
cat <<EOF
  armc_prefix (decomp_bin, charged 2x)  $A bytes   (artifact of record: $EXPECT_ARMC)
  comp_$SLOT_KIND                          $D bytes   (artifact of record: $EXPECT_CD)
  comp_order                            $O bytes   (artifact of record: $EXPECT_CO)
  comp_tfweights                        $T bytes   (artifact of record: ${EXPECT_TFW:-0})
  header                              $H bytes
  ------------------------------------------------------------------
  /work/comp9  (COMPRESSOR, scored)     $COMP9 bytes   (artifact of record: $EXPECT_COMP9)
  sha256=$COMP9_SHA
  want  =$EXPECT_COMP9_SHA
  helper run/comp9 (NOT emitted, NOT scored): $(wc -c < run/comp9) bytes
  premium avoided by emitting comp9_min:      $(( 2 * ($(wc -c < run/decomp_bin) - A) )) bytes of S
EOF

# DELIBERATELY A WARNING, NOT A FAILURE. The committee books S off ITS OWN
# rebuild (cmix-lex's rebuild moved S by +21,592 B), and HPJA handles a rebuild
# that differs from the submitted archive by simply running a second decode. A
# divergent-but-valid rebuild is therefore SCORED; aborting here would convert a
# scoring difference into a total entry failure. Set ZMIX_EXPECT_STRICT=1 for an
# entrant-side rehearsal, where a mismatch IS the finding.
if [[ "$COMP9_SHA" == "$EXPECT_COMP9_SHA" && "$COMP9" == "$EXPECT_COMP9" ]]; then
  echo "REBUILD IS BYTE-IDENTICAL to the $ARTIFACT_LABEL artifact of record."
else
  echo "WARNING: this rebuild is NOT byte-identical to the $ARTIFACT_LABEL artifact of record." >&2
  echo "         S will be booked off THIS rebuild. Check the zig/upx pins, binutils," >&2
  echo "         SHIP_RECIPE.env and the source tree before treating it as expected." >&2
  [[ "${ZMIX_EXPECT_STRICT:-0}" != "1" ]] || { echo "ZMIX_EXPECT_STRICT=1 -> ABORT" >&2; exit 1; }
fi

say "build.sh complete"
ls -l /work/comp9
