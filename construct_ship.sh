#!/bin/bash
set -euo pipefail
#
# construct_ship.sh — build the Form-1 `comp9` self-extracting compressor
# from the checked-in recipe (SHIP_RECIPE.env) and VERIFY it end-to-end.
#
# PORT of the record's build_and_construct_comp.sh (lines 51-81): strip+UPX the
# lean code binary (== decomp_bin / `cmix_orig`), self-compress english.dic +
# new_article_order with the binary's OWN no-dict `-c` codec, write a 12-byte
# HeaderInfo, and concatenate
#   comp9 = decomp_bin ++ comp_dict ++ comp_order ++ header  (construct.sh:80)
# `construct_form1.sh enwik9` then runs `comp9 -e`, producing the physical
# no-argument self-extracting `archive9.exe` and verifying the Form-1 pair.
# The packager requires an explicit archive-order layout: legacy/r1v1 omits
# order_comp; r1v2 must embed it because its payload restore derives from order.
#
# THE RECIPE (SHIP_RECIPE.env, sourced below) is the single source of truth:
# every banked model lever is a comptime build option that defaults to STOCK
# cmix-lex (so the golden gates stay green), and the hermetic ship binary reads
# no env — a bare `zig build ship` is stock, NOT the ship. This script is the
# only supported way to produce the shipping artifact.
#
# ASSET CACHE: the two self-compressed assets live at run/comp_dict.<fp8>
# (carried) or run/comp_recipe.<fp8> (-Dderivdict) / run/comp_order.<fp8>. They
# are compressed BY the engine, so ANY engine change (code or recipe)
# invalidates them — a stale pairing would desync or hard-abort a later
# `-e`/`-D` run. The engine fingerprint in the FILENAME is only part of the
# key: validity is decided by the ASSET_KEY stamp written beside each artifact
# (section 4-key), which also covers the weights blob and the asset SOURCES.
# Missing-or-stale-for-this-key => rebuild; FORCE_REBUILD=1 forces
# re-compression regardless.
#
# VERIFICATION (mandatory, script fails loudly): the assembled distribution
# must round-trip its own tail — `--extract-assets` output is byte-compared
# against the original english.dic / new_article_order. A build that cannot
# reproduce its assets is a broken ship, whatever its size.
#
# UPX:
#   * The packer is a DECLARED TOOLCHAIN COMPONENT, pinned in SHIP_RECIPE.env as
#     ZMIX_UPX_VERSION + ZMIX_UPX_MODE. It is resolved below, and if the host has
#     no suitable upx this script BUILDS IT FROM SOURCE via tools/build_upx.sh —
#     no root, no package manager, no cmake, nothing for the evaluator to run by
#     hand. That is what the committee ruling ("Tightening up
#     compression program build script requirements") requires: an installer
#     script the README tells the evaluator to run is REJECTABLE; a build script
#     that provisions its own tools without root is not.
#   * upx 4.x CANNOT pack this engine at all (`CantPackException: xspan
#     unexpected NULL pointer`, measured on upx 4.2.2, with and without the
#     section-header strip) — the 5.x floor is a hard requirement, not a taste.
#   * ZMIX_UPX_MODE MUST be a literal flag, not "auto". "auto" silently changes
#     the artifact if --ultra-brute happens to fail on the evaluator's host
#     (--lzma packs this engine to 135,588 vs --ultra-brute's 135,356 = +232 B,
#     doubled to +464 B of S under Form-1) — and the committee books S off its
#     OWN rebuild, so a host-dependent recipe is a scoring hazard. "auto" is
#     still honoured for back-compat, and warns.
#
# ZIG:
#   * The compiler is a DECLARED TOOLCHAIN COMPONENT too, pinned in
#     SHIP_RECIPE.env as ZMIX_ZIG_VERSION. It is resolved below, and if the host
#     has no matching zig this script PROVISIONS IT via tools/provision_zig.sh
#     (sha-pinned upstream tarball, no root, no package manager, nothing to
#     compile). Same clause of the same ruling as the packer: this
#     script used to `export PATH="$HOME/.local/zig-0.15.1:..."`, a path that
#     exists on our boxes and on no evaluator's machine.
#   * The pin is EXACT, not a floor — MEASURED, not assumed. Off-pin arm with zig
#     0.15.2 (which "0.15.1+" permits), same source: the stripped ship engine
#     moves 325,024 -> 324,960 B and the packed prefix 135,356 -> 135,332 B, so S
#     moves by 48 B (Form-1 charges the prefix twice). The stock 50k archives were
#     byte-IDENTICAL, i.e. the coded stream did NOT move at that tier — reported
#     as measured; it says little about e9. The committee rebuilds comp9 from
#     source and books S off ITS OWN rebuild, so a silent version mismatch would
#     be scored. `zig version` is asserted to equal the pin exactly; abort
#     otherwise.
#
# Env overrides:
#   DICT           path to english.dic       (default: src/models/fxcm26/goldens/english.dic)
#   ORDER          path to new_article_order (default: src/prepr/new_article_order_asset)
#   RECIPE         path to the Arm E' recipe blob, used INSTEAD of DICT in comp9's
#                  first tail slot under -Dderivdict=true
#                  (default: src/models/fxcm26/goldens/derivdict_recipe_E2.bin;
#                   re-derive with `zig build derivdict-emit`)
#   ZMIX_TFWEIGHTS path to the fx2 transformer weights blob, used ONLY on a transformer-era
#                  (-Dtransformer=true) recipe
#                  (default: assets/6m-q4-fp32.tfwc2, shipped with the source).
#                  ⛔ Keep it repo-relative: it is passed to `engine -c` as an
#                  explicit argument and copied verbatim into comp9's tail; an
#                  absolute path here is a rebuild-on-one-machine-only artifact.
#   FORCE_REBUILD  =1 re-compress assets even if cached for this fingerprint
#   UPX            explicit path to the packer   (skips resolution below)
#   ZIG            explicit path to the compiler (skips resolution below)
#   ZIG_SRC_TARBALL  pre-downloaded zig tarball for an offline machine
#
# NOTE: self-compressing an asset runs the full multi-GB predictor and takes
# ~2-13 min per asset (box-dependent); the mandatory --extract-assets check
# decompresses both again (about the same cost). Budget ~10-40 min end-to-end
# on a cold cache.

cd "$(dirname "$0")"

# ---- 0) recipe -------------------------------------------------------------
[[ -f SHIP_RECIPE.env ]] || { echo "FATAL: SHIP_RECIPE.env not found" >&2; exit 1; }
# shellcheck disable=SC1091
source ./SHIP_RECIPE.env
: "${ZMIX_ENGINE_FLAGS:?SHIP_RECIPE.env must set ZMIX_ENGINE_FLAGS}"
: "${ZMIX_ABI:?SHIP_RECIPE.env must set ZMIX_ABI}"
: "${ZMIX_UPX_MODE:?SHIP_RECIPE.env must set ZMIX_UPX_MODE}"

# ---- 0-legality) Form-1 recipe gate ----------------------------------------
# -Dr1v2 makes the DECODER derive the payload from the raw article order, so a
# Form-1 archive9.exe must then carry a SECOND copy of comp_order (layout
# `embedded`). That is legal but costs +44,966 B of S versus r1v1, and it has
# never been this entry's target. The dangerous case is the other one: r1v2 with
# the `omitted` layout this entry actually ships builds CLEANLY and produces a
# pair that simply cannot self-extract — a silent RULES.md rule-2 violation with
# no error anywhere. Refuse by default so choosing that branch is a conscious act.
# ---- 0-legality-b) the canonical-205 NEGATIVE CONTROL may never ship --------
# -Dtransformer-map-fault deliberately corrupts the token map so the submission
# map guard can be proven to fire. A build carrying it is a control, not an
# artifact: it produces a VALID, LOSSLESS, much larger archive with nothing
# anywhere complaining (that is the whole point of the guard). Refuse by name,
# the same way the r1v2 Form-1 gate does.
if [[ " $ZMIX_ENGINE_FLAGS " == *"-Dtransformer-map-fault="* \
      && " $ZMIX_ENGINE_FLAGS " != *"-Dtransformer-map-fault=0"* ]]; then
  echo "FATAL: SHIP_RECIPE.env carries a nonzero -Dtransformer-map-fault." >&2
  echo "  That flag DELIBERATELY CORRUPTS the canonical-205 token map. It exists" >&2
  echo "  only as the negative control that proves the map guard aborts; a build" >&2
  echo "  carrying it round-trips losslessly and codes far worse, silently." >&2
  echo "  See src/mixer/transformer.zig 'map_fault'. Refusing to construct." >&2
  exit 1
fi

if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dr1v2=true "* && "${ZMIX_ALLOW_R1V2_EMBEDDED:-0}" != "1" ]]; then
  echo "FATAL: SHIP_RECIPE.env carries -Dr1v2=true." >&2
  echo "  Under Form-1 that recipe is legal ONLY with archive-order 'embedded'," >&2
  echo "  i.e. archive9.exe must carry a second copy of comp_order (+44,966 B of S)." >&2
  echo "  With the shipped 'omitted' layout the build SUCCEEDS and the pair cannot" >&2
  echo "  self-extract (RULES.md rule 2) — silently. See the P0 block in" >&2
  echo "  SHIP_RECIPE.env." >&2
  echo "  If you really want branch A, re-run with ZMIX_ALLOW_R1V2_EMBEDDED=1 and" >&2
  echo "  pass --archive-order embedded to construct_form1.sh." >&2
  exit 1
fi

# ---- 0-pin) TREE-PIN ASSERTION (added ; rebased to the CONTENT) --
# The manifest RECORDS git_head after the fact but asserted nothing, so a rebuild
# from the wrong source was silent. The pin is asserted HERE — before the toolchain
# is resolved and before anything is built — so a wrong tree costs seconds, not a
# whole construct, and so the strict mode is cheap enough to leave armed. See
# SHIP_RECIPE.env for why an engine-fingerprint check cannot substitute (ship.zig
# does not compile form1.zig, so the engine sha is identical across the defect).
#
# ⚠ THE PIN IS ON THE ENGINE SOURCE DIGEST, NOT ON HEAD (rebased for
# the pin is on the engine-source digest, not on HEAD. A commit pin cannot work here,
# for two independent reasons:
#   (1) SELF-REFERENCE. Pinning HEAD means every commit invalidates the pin, and
#       the edit that fixes the pin changes HEAD again. There is no fixed point,
#       which is why no single ref could ever carry both the engine AND the
#       packaging scripts, and why this check has sat in `warn` since it landed.
#   (2) NO .git ON THE JUDGE PATH. The submitted source package ships without a
#       repository, so `git rev-parse HEAD` yields `unknown` and a strict pin
#       would abort EVERY judge build — converting a scoring difference into a
#       total failure.
# The digest below is git-free, deterministic, and STRICTLY TIGHTER than a commit
# sha in the dimension that matters: it fires on a dirty source tree, which an
# identical HEAD does not. It covers exactly what the compiler reads, so
# packaging, docs and reports move freely without touching it — and, crucially,
# it does NOT cover SHIP_RECIPE.env, so writing the digest INTO the recipe cannot
# change the digest. That is how the circularity is resolved.
#
# ★ THE transformer-era FILE SET IS WIDER THAN THE LSTM-ERA ONE, AND THE DIFFERENCE IS NOT
# COSMETIC. Digesting only `src/**/*.zig` + `build.zig` would leave the
# following OUTSIDE the pin while they fully determine the binary:
#   * third_party/fx2_transformer/*.cpp,*.h — the 9 vendored -O3 translation units
#     plus weights_io_compressed.cpp and 13 headers (build.zig:1446-1466);
#   * src/cxx_local.cpp — our libc++ replacement, ~70 KB of a prefix charged 2x;
#   * src/prepr/new_article_order_asset — @embedFile'd by article_reorder.zig:348
#     and s7_order.zig:288, i.e. compiled INTO the engine, and also the shipped
#     comp_order source;
#   * src/models/fxcm26/goldens/{english.dic,derivdict_recipe_E2.bin} — the
#     shipped dict/recipe assets, which live under src/ (see DICT/ORDER/RECIPE).
# So the set is every file under src/ and third_party/, plus build.zig and
# build.zig.zon. Nothing in the ship chain writes into either directory, so the
# digest is stable across the build; make_package.sh's staging touches neither
# (it dereferences the one top-level symlink and drops non-shipped directories),
# so the staged-tree digest equals this one by construction.
# ⇒ UPDATE ZMIX_SRC_DIGEST IN THE SAME EDIT AS ANY src/, third_party/ OR build.zig
#   CHANGE. make_package.sh recomputes this identical number on the STAGED tree;
#   if the two definitions ever diverge, packaging fails closed.
zmix_src_digest() {
  { find src third_party -type f -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 -r sha256sum
    sha256sum build.zig build.zig.zon 2>/dev/null; } | sha256sum | cut -c1-16
}
if [ -n "${ZMIX_SRC_DIGEST:-}" ]; then
  _have="$(zmix_src_digest)"
  _want="$(printf '%s' "$ZMIX_SRC_DIGEST" | cut -c1-16)"
  if [ "$_have" != "$_want" ]; then
    echo "!! ENGINE SOURCE PIN MISMATCH: src digest $_have but SHIP_RECIPE.env describes $_want" >&2
    echo "!! A recipe describing a different tree than it is built against produces a WRONG" >&2
    echo "!! artifact SILENTLY — this is exactly the 178c3d7-vs-98f543c defect (a smaller," >&2
    echo "!! cheaper-looking prefix that dies R1LengthMismatch on e9)." >&2
    if [ "${ZMIX_TREE_PIN_MODE:-warn}" = "strict" ]; then
      echo "!! ZMIX_TREE_PIN_MODE=strict -> ABORT" >&2; exit 3
    fi
    echo "!! ZMIX_TREE_PIN_MODE=warn -> continuing; the manifest will record what was ACTUALLY built." >&2
  else
    echo "engine source pin OK: $_have matches SHIP_RECIPE.env's ZMIX_SRC_DIGEST"
  fi
else
  echo "!! WARNING: SHIP_RECIPE.env carries no ZMIX_SRC_DIGEST — the tree pin is NOT asserted." >&2
fi

# ---- 0-pin-b) SHIPPED WEIGHTS BLOB PIN  -------------------
# The digest above covers exactly what the COMPILER reads. The transformer-era transformer
# weight blob is NOT a compiler input — it is an asset copied verbatim into
# comp9's tail and into archive9.exe — so it is outside that digest by design,
# and until this check existed the ship chain asserted only its 8-byte MAGIC and
# its pairing with -Dtf-weights-v3 (section 3 below). That is far weaker than it
# looks: a blob with the right magic and the WRONG CONTENT builds, roundtrips
# and ships SILENTLY, because encode and decode read the same wrong weights,
# agree perfectly, and emit a valid archive that is merely far larger. A
# lossless roundtrip cannot catch it. The blob is 87.2 % of comp9 and Form-1
# charges it TWICE, so it is the largest single unpinned input to S; only the
# EXPERIMENT DRIVERS asserted its sha, and a judge rebuild never runs one.
# Asserted HERE, next to the source pin and under the same ZMIX_TREE_PIN_MODE
# knob, before the toolchain is resolved and before anything is built — so a
# wrong blob costs a second, not a whole construct, and the negative control is
# cheap enough to actually run.
# ⚠ THE PATH AND THE SHA MOVE TOGETHER, IN ONE CONDITIONAL, for the same reason
#   section 3 resolves the path from the flag: pinning one sha unconditionally
#   would abort the v2 path for the wrong reason. This block resolves
#   ZMIX_TFWEIGHTS once; section 3's `${ZMIX_TFWEIGHTS:-...}` default is then a
#   no-op and its magic/pairing checks remain as defence in depth.
TFW_SHA_HAVE=""
TFW_SHA_WANT=""
TFW_SHA_VAR=""
if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dtransformer=true "* ]]; then
  if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dtf-weights-v3=true "* ]]; then
    ZMIX_TFWEIGHTS="${ZMIX_TFWEIGHTS:-assets/6m-q4-fp32.tfwc3}"
    TFW_SHA_WANT="${ZMIX_TFWEIGHTS_SHA256:-}"
    TFW_SHA_VAR="ZMIX_TFWEIGHTS_SHA256"
  else
    ZMIX_TFWEIGHTS="${ZMIX_TFWEIGHTS:-assets/6m-q4-fp32.tfwc2}"
    TFW_SHA_WANT="${ZMIX_TFWEIGHTS_V2_SHA256:-}"
    TFW_SHA_VAR="ZMIX_TFWEIGHTS_V2_SHA256"
  fi
  [[ -s "$ZMIX_TFWEIGHTS" ]] || {
    echo "FATAL: transformer weights '$ZMIX_TFWEIGHTS' missing or empty." >&2
    echo "       The blob ships in-tree under assets/; a checkout should have it." >&2
    echo "       Override with ZMIX_TFWEIGHTS=<path> if you moved it." >&2
    exit 1; }
  TFW_SHA_HAVE=$(sha256sum "$ZMIX_TFWEIGHTS" | cut -d' ' -f1)
  if [ -n "$TFW_SHA_WANT" ]; then
    if [ "$TFW_SHA_HAVE" != "$TFW_SHA_WANT" ]; then
      echo "!! WEIGHTS BLOB PIN MISMATCH: $ZMIX_TFWEIGHTS is" >&2
      echo "!!   $TFW_SHA_HAVE" >&2
      echo "!! but SHIP_RECIPE.env's $TFW_SHA_VAR describes" >&2
      echo "!!   $TFW_SHA_WANT" >&2
      echo "!! The MAGIC check in section 3 cannot see this: a blob with the right magic and" >&2
      echo "!! the wrong content builds, roundtrips and ships SILENTLY (encode and decode read" >&2
      echo "!! the same wrong weights, so the coder agrees with itself). comp_tfweights is 87 %" >&2
      echo "!! of comp9 and Form-1 charges it TWICE." >&2
      if [ "${ZMIX_TREE_PIN_MODE:-warn}" = "strict" ]; then
        echo "!! ZMIX_TREE_PIN_MODE=strict -> ABORT" >&2; exit 3
      fi
      echo "!! ZMIX_TREE_PIN_MODE=warn -> continuing; the manifest will record what was ACTUALLY shipped." >&2
    else
      echo "weights blob pin OK: $TFW_SHA_HAVE matches SHIP_RECIPE.env's $TFW_SHA_VAR ($ZMIX_TFWEIGHTS)"
    fi
  else
    echo "!! WARNING: SHIP_RECIPE.env carries no $TFW_SHA_VAR — the weights blob is NOT pinned." >&2
  fi
fi

# ---- 0a) resolve the DECLARED compiler --------------------------------------
# Order: $ZIG, then a PATH zig of the pinned version, then a previously
# provisioned one, then a fresh provision. Never an installer, never root, never
# "whatever zig this machine happens to have".
WANT_ZIG="${ZMIX_ZIG_VERSION:-0.15.1}"
zig_is_pinned() { [[ -x "$1" ]] && [[ "$("$1" version 2>/dev/null)" == "$WANT_ZIG" ]]; }
if [[ -n "${ZIG:-}" ]]; then
  zig_is_pinned "$ZIG" || {
    echo "FATAL: \$ZIG=$ZIG reports '$("$ZIG" version 2>&1 | head -1)', not zig $WANT_ZIG" >&2
    echo "       The compiler version changes the compressed stream — refusing to guess." >&2
    exit 1; }
elif zig_is_pinned "$(command -v zig 2>/dev/null || echo /nonexistent)"; then
  ZIG="$(command -v zig)"
elif zig_is_pinned "$HOME/.local/zig-$WANT_ZIG/zig"; then
  ZIG="$HOME/.local/zig-$WANT_ZIG/zig"
elif zig_is_pinned "$PWD/.zig-toolchain/zig-$WANT_ZIG/zig"; then
  ZIG="$PWD/.zig-toolchain/zig-$WANT_ZIG/zig"
else
  echo "== no zig $WANT_ZIG on this host — provisioning the declared toolchain =="
  ZIG="$(./tools/provision_zig.sh "$PWD/.zig-toolchain")" || {
    echo "FATAL: could not provision zig $WANT_ZIG (see tools/provision_zig.sh)" >&2; exit 1; }
fi
# Belt and braces: whatever branch we came through, the version is asserted here.
GOT_ZIG="$("$ZIG" version 2>/dev/null || true)"
[[ "$GOT_ZIG" == "$WANT_ZIG" ]] || {
  echo "FATAL: resolved compiler is zig '$GOT_ZIG', want EXACTLY '$WANT_ZIG' ($ZIG)" >&2
  echo "       Measured: zig 0.15.2 moves the packed prefix by 24 B = 48 B of S, and the" >&2
  echo "       committee scores its OWN rebuild from source. Aborting." >&2
  exit 1; }
export PATH="$(dirname "$ZIG"):$PATH"
echo "zig: $ZIG  (version $GOT_ZIG, pinned)"

DICT="${DICT:-src/models/fxcm26/goldens/english.dic}"
ORDER="${ORDER:-src/prepr/new_article_order_asset}"
RECIPE="${RECIPE:-src/models/fxcm26/goldens/derivdict_recipe_E2.bin}"
DIR=run
mkdir -p "$DIR"

# ---- Arm E' (-Dderivdict): comp9's FIRST TAIL SLOT IS THE RECIPE ------------
# Form-1 charges a CARRIED asset twice and a DERIVED one once, so under
# -Dderivdict comp9 ships the ~65.8 KB recipe blob where it otherwise ships the
# 100.5 KB english.dic.comp, and `-e` rebuilds the dictionary from enwik9 at
# encode time (src/ship.zig, src/derivdict.zig).
#
# ⛔ REGRESSION FIXED HERE. This script used to compress $DICT into
# that slot UNCONDITIONALLY while the manifest declared "dictionary mode:
# derived", so the LSTM-era comp9 carried english.dic.comp and told packaging it
# carried a recipe. `comp9 -e enwik9` then fed english.dic to
# derivdict.rebuildShipDict AS THE RECIPE, censused all of enwik9 (97 s), and
# died with error.RecipeCorrupt. The step-7 --extract-assets gate that exists to
# catch exactly this compared the extracted slot against $DICT — the wrong
# golden — so it PASSED and certified the broken artifact. Both halves are fixed:
# the slot source is selected here, and the gate below compares against it.
if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dderivdict=true "* ]]; then
  FORM1_DICT_MODE=derived
  DICT_SLOT_KIND=recipe
  DICT_SLOT_SRC="$RECIPE"
  [[ -s "$DICT_SLOT_SRC" ]] || {
    echo "FATAL: -Dderivdict=true but the Arm E' recipe blob is missing or empty: $DICT_SLOT_SRC" >&2
    echo "       Re-derive it with: zig build derivdict-emit && \\" >&2
    echo "         zig-out/bin/derivdict_emit <enwik9> $DICT $DICT_SLOT_SRC" >&2
    exit 1; }
else
  FORM1_DICT_MODE=carried
  DICT_SLOT_KIND=dict
  DICT_SLOT_SRC="$DICT"
fi

ABI_FLAG=""
case "$ZMIX_ABI" in
  glibc)  ABI_FLAG="-Dglibc=true" ;;
  static) ABI_FLAG="" ;;
  *) echo "FATAL: ZMIX_ABI='$ZMIX_ABI' (want glibc|static)" >&2; exit 1 ;;
esac

# ---- 1) build the lean recipe engine (no embedded assets) ------------------
# The levers are comptime-baked; the hermetic ship consults no env at run time.
# shellcheck disable=SC2086  # word-splitting of the recipe flags is intended
echo "== zig build ship $ZMIX_ENGINE_FLAGS $ABI_FLAG =="
"$ZIG" build ship $ZMIX_ENGINE_FLAGS $ABI_FLAG

# ---- 2) strip + fingerprint ------------------------------------------------
# run/engine = stripped, PRE-UPX engine. Its sha256 is the ENGINE FINGERPRINT:
# it keys the asset cache and goes in the manifest. (UPX output can vary by upx
# version/mode; the stripped image is the stable identity of the codec.)
cp zig-out/bin/zmix_ship "$DIR/engine"
strip -s "$DIR/engine"
# Drop metadata + unwind sections that survive `strip -s` but serve no purpose in
# a hermetic Hutter binary. These live inside PT_LOAD, so UPX DOES compress them —
# removing shrinks the packed decomp_bin by ~252 B (measured 129,408 -> 129,156 on
# the glibc build), byte-identical, still runs (verified -c/-d/-e round-trip):
#   .comment            clang/LLD version strings (pure metadata)
#   .note.gnu.property   x86 CET/SHSTK feature markers (binary uses no CET)
#   .note.ABI-tag        advisory min-ABI note (loader does not require it)
#   .eh_frame[_hdr]      residual unwind info; the code never unwinds
#                        (simple_panic, no exceptions — matches the record's
#                        C++ -fno-unwind-tables). unwind_tables=.none already
#                        emptied it to ~52 B; this removes the remainder.
objcopy -R .comment -R .note.gnu.property -R .note.ABI-tag -R .eh_frame -R .eh_frame_hdr "$DIR/engine"

# ---- 2b) DETERMINISM GATE (implements the check detm_math.zig:16-19 promises) --
# The coder's numerics must be bit-identical across machines: a submission is
# compressed on one box and self-decompresses on the judge's. detm_math.zig exports
# STRONG local exp/log/exp2/log2/tanhf/... so every LLVM-emitted transcendental
# libcall binds inside the binary, never to the host glibc libm (which is IFUNC-
# resolved per-CPU and drifts across glibc versions — a proven ~40h-in desync/crash,
# detm_math.zig:6-7). If a future change emits a libcall detm_math does NOT define
# (exp2f/logf/log2f/powf/...), it is left UNDEFINED and would resolve to the judge's
# libm at load. Catch it at build time instead of 40 h into the judged decode.
UND_MATH=$(objdump -T "$DIR/engine" 2>/dev/null | awk '/UND/{
  n=$NF
  if (n ~ /^(exp|expf|exp2|exp2f|exp10|exp10f|log|logf|log2|log2f|log10|log10f|pow|powf|tanh|tanhf|sinh|sinhf|cosh|coshf|sin|sinf|cos|cosf|tan|tanf|atan|atanf|atan2|atan2f|cbrt|cbrtf)$/) print n
}')
if [[ -n "$UND_MATH" ]]; then
  echo "FATAL: engine imports transcendental(s) from the host libm — cross-machine desync risk:" >&2
  echo "$UND_MATH" | sed 's/^/  UND /' >&2
  echo "  detm_math.zig must export a strong local definition for each (a new libcall slipped through)." >&2
  exit 1
fi
echo "determinism gate OK: no undefined transcendentals (libm pinned by detm_math)"

# ---- 2c) RECIPROCAL-ESTIMATE GATE (the OTHER half of determinism) -----------
# Reciprocal-estimate determinism gate.
# The gate above catches libm DIVERGENCE BY SYMBOL. It cannot catch divergence by
# INSTRUCTION: under `@setFloatMode(.optimized)` LLVM lowers `a/b` and
# `1/sqrt(x)` to VRCPPS/VRSQRTPS *estimates*, whose low mantissa bits come from a
# vendor-defined on-die lookup table. The previous entry shipped six such instructions on the
# prediction path (`lstm_layer.adam`, `LstmLayer.forwardNeuron`,
# `PredictorLex.perceive`). Encoder and decoder on different-vendor silicon then
# compute different probabilities and the arithmetic coder desynchronises — the
# documented cause of the fx2-cmix-transformer July 2026 submission failure on the
# committee's own machine. Fix: src/strict_fp.zig.
# Runs BEFORE --strip-section-headers and before UPX: after either, `objdump -d`
# returns a FALSE CLEAN (3 lines) and the gate would silently pass.
if [[ -x ./tools/recip_gate.sh ]]; then
  if ! ./tools/recip_gate.sh "$DIR/engine" >&2; then
    echo "FATAL: engine contains a reciprocal-ESTIMATE opcode — cross-vendor coder desync." >&2
    exit 1
  fi
  echo "reciprocal gate OK: no rcpps/rsqrtps estimates, nothing beyond x86-64-v3"
else
  echo "FATAL: tools/recip_gate.sh missing — refusing to package an ungated engine." >&2
  exit 1
fi

# ---- 2d) C++ EXCEPTION/UNWIND/DEMANGLE GATE (transformer-era) -------------------------
# See tools/check_no_cxx_unwind.sh. Must run BEFORE --strip-section-headers and
# before UPX, for the same reason the reciprocal gate does. Inert on LSTM-era.
if [[ -x ./tools/check_no_cxx_unwind.sh ]]; then
  if ! ./tools/check_no_cxx_unwind.sh "$DIR/engine" >&2; then
    echo "FATAL: engine carries libc++abi exception/unwind/demangle machinery (see src/cxx_local.cpp)." >&2
    exit 1
  fi
else
  echo "FATAL: tools/check_no_cxx_unwind.sh missing — refusing to package an ungated engine." >&2
  exit 1
fi

# Section headers are not consulted by the Linux loader and UPX does not need
# them.  Remove them only AFTER objdump's determinism gate above (objdump needs
# .dynsym/.dynstr section metadata), but before fingerprinting and packing.
# This preserves every PT_LOAD byte while avoiding 1,440 bytes of linker
# metadata.  Fourth-E decomp_bin: 146,004 -> 145,612 bytes; Form-1 carries the
# prefix in both comp9 and archive9.exe, so the score saves 784 B.
objcopy --strip-section-headers "$DIR/engine"

FP=$(sha256sum "$DIR/engine" | cut -d' ' -f1)
FP8=${FP:0:8}
RAW=$(wc -c < "$DIR/engine")
echo "engine (stripped, pre-UPX): $RAW bytes  fingerprint=$FP8 ($FP)"

# ---- 3) UPX-pack decomp_bin — this IS the record's `cmix_orig` --------------
# 3a) resolve the DECLARED packer. Order: $UPX, then a PATH upx of the pinned
# version, then a from-source build (root-free, cmake-free). Never an installer.
# ⚠ upx reads the environment variable `UPX` as a DEFAULT-OPTIONS STRING, so the
# documented offline override `UPX=/path/to/upx ./construct_ship.sh` used to make
# the pack below die with "invalid string '/path/to/upx' in environment variable
# 'UPX'" — AFTER the version assertion passed, because `--version` does not parse
# it. Every upx invocation therefore goes through `env -u UPX`, which makes the
# packer immune to the caller's environment while leaving $UPX usable as a path.
WANT_UPX="${ZMIX_UPX_VERSION:-5.2.0}"
upx_run() { env -u UPX "$UPX" "$@"; }
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
  echo "== no upx $WANT_UPX on PATH — building the declared packer from source =="
  UPX="$(./tools/build_upx.sh "$PWD/.upx-build")" || {
    echo "FATAL: could not provision upx $WANT_UPX (see tools/build_upx.sh)" >&2; exit 1; }
fi
echo "upx: $UPX  ($(upx_run --version | head -1))"

pack() { # $1 = upx flags; packs a fresh copy of the engine into decomp_bin
  cp "$DIR/engine" "$DIR/decomp_bin"
  # shellcheck disable=SC2086
  if upx_run $1 -q "$DIR/decomp_bin" >/dev/null 2>"$DIR/upx.err"; then
    UPX_USED="$1"; return 0
  fi
  return 1
}
UPX_USED=""
case "$ZMIX_UPX_MODE" in
  auto)
    echo "WARNING: ZMIX_UPX_MODE=auto is host-dependent (--ultra-brute vs --lzma" >&2
    echo "         differ by 232 B = 464 B of S). Pin a literal mode before shipping." >&2
    pack --ultra-brute || {
      echo "upx --ultra-brute failed ($(tail -1 "$DIR/upx.err" 2>/dev/null)); falling back to --lzma"
      pack --lzma || { echo "FATAL: upx --lzma failed too:" >&2; cat "$DIR/upx.err" >&2; exit 1; }
    } ;;
  *)
    pack "$ZMIX_UPX_MODE" || { echo "FATAL: upx $ZMIX_UPX_MODE failed:" >&2; cat "$DIR/upx.err" >&2; exit 1; } ;;
esac
BIN=$(wc -c < "$DIR/decomp_bin")
echo "decomp_bin (strip+upx $UPX_USED): $BIN bytes"

# ---- 3b) the SUBMITTED arm-C prefix (the artifact of record's decomp_bin) ----
# The prefix packed above belongs to the FULL-CLI helper `zmix_ship`. What the
# submission actually ships is the smaller CLI-free dual-role program built by
# `zig build form1 ... -Dform1-malloc=true` and put through the same
# strip/objcopy/objcopy/upx chain. Until that chain existed ONLY AS
# PROSE, and `construct_form1.sh --minimal-prefix` takes the packed prefix as an
# EXTERNALLY SUPPLIED path — it never built one. So a rebuild that followed this
# script and README.md literally produced comp9 384,545 B instead of comp9_min
# 376,985 B: a silent +15,120 B of S, because decomp_bin is charged TWICE, with
# every gate green and nothing anywhere warning.
# It is built HERE, reusing the compiler and packer already resolved above, so
# there is genuinely no manual step between the two scripts: hand run/armc_prefix
# to `construct_form1.sh --minimal-prefix`.
# ZMIX_SKIP_ARMC=1 skips it (development shortcut only — run/comp9 on its own is
# NOT the submittable artifact).
ARMC=""
ARMC_BIN=0
ARMC_SHA=""
if [[ "${ZMIX_SKIP_ARMC:-0}" == "1" ]]; then
  echo "WARNING: ZMIX_SKIP_ARMC=1 — the submitted arm-C prefix was NOT built." >&2
  echo "         run/comp9 alone costs +15,120 B of S versus the arm-C comp9_min." >&2
else
  # transformer-era gap CLOSED (the WARNING that stood here is gone, not
  # silenced): `src/form1.zig` now carries the 4-field container — the
  # comp_tfweights term in `segmentSizes`, the blob written between payload and
  # trailer in `writeArchive`, and `Transformer.embedded_blob` set from its own
  # tail in BOTH roles. `construct_form1.sh` reads the widened trailer and the
  # comp_tfweights segment. Proof on the artifact, not the source:
  ARMC="$(ZIG="$ZIG" UPX="$UPX" ./tools/build_armc_prefix.sh "$PWD/$DIR")" || {
    echo "FATAL: could not build the arm-C minimal prefix (tools/build_armc_prefix.sh)" >&2
    exit 1; }
  ARMC_BIN=$(wc -c < "$ARMC")
  ARMC_SHA=$(sha256sum "$ARMC" | cut -d' ' -f1)
  echo "armc_prefix (the SUBMITTED decomp_bin, charged 2x): $ARMC_BIN bytes"
fi

# ---- 3c) transformer-era ONLY: resolve the int4 transformer weights blob --------------
# fx2-cmix-transformer appends comp_tfweights between comp_order and the trailer
# and widens the header to FOUR fields (their build_and_construct_comp.sh:117-120:
#   cmix_orig -h <dict> <order> 0 <tfweights>
#   cat cmix_orig comp_dict comp_order comp_tfweights header.dat > cmix).
# Our src/prepr/self_extract.zig mirrors that layout under -Dtransformer.
# Empty and inert on LSTM-era, where the trailer stays 12 bytes.
#
# ⛔ RESOLVED HERE, BEFORE STEP 4, AND IT MUST STAY THAT WAY. Step 4 self-compresses
# english.dic and the order asset with `engine -c`, and under -Dtransformer that runs
# the FULL predictor — transformer included — on the BARE packed engine, which has no
# tail to read the weights from. Until this block sat AFTER step 4, so the
# only source `-c` had was the build-time `-Dtransformer-weights` string, which was an
# ABSOLUTE dev-box path: a committee rebuild from source died at the first `-c` with
#   weights_io: cannot open <a path on the build machine>
# (submission-blocker class: a shipped program must depend on nothing outside its
# own bytes).
# The path is now (a) DEFAULTED to the in-tree asset shipped with the source and
# (b) handed to every `-c` as an EXPLICIT ARGUMENT, so neither the build machine nor
# the caller's cwd can change the answer.
CTW=""
TW=0
TFW_ARG=()   # expands to nothing on LSTM-era, so those invocations are unchanged
if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dtransformer=true "* ]]; then
  # Shipped with the source, like src/models/fxcm26/goldens/english.dic and
  # src/prepr/new_article_order_asset. Overridable for development, never required.
  # ⛔ THE DEFAULT MUST TRACK -Dtf-weights-v3. The flag teaches the loader the
  # FX2TFWC3 container; the blob is what actually banks the -90,235 B (charged
  # TWICE). Shipping the .tfwc2 blob with the flag ON is a coherent build that
  # pays the +3,792 B decoder fee and banks nothing, and nothing would warn --
  # so the two move together here, in one conditional.
  if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dtf-weights-v3=true "* ]]; then
    ZMIX_TFWEIGHTS="${ZMIX_TFWEIGHTS:-assets/6m-q4-fp32.tfwc3}"
    TFW_WANT="assets/6m-q4-fp32.tfwc3 (2,840,417 B, magic FX2TFWC3)"
  else
    ZMIX_TFWEIGHTS="${ZMIX_TFWEIGHTS:-assets/6m-q4-fp32.tfwc2}"
    TFW_WANT="assets/6m-q4-fp32.tfwc2 (2,930,652 B, magic FX2TFWC2)"
  fi
  [[ -s "$ZMIX_TFWEIGHTS" ]] || {
    echo "FATAL: transformer weights '$ZMIX_TFWEIGHTS' missing or empty." >&2
    echo "       The blob ships in-tree at $TFW_WANT;" >&2
    echo "       a checkout should have it. Override with ZMIX_TFWEIGHTS=<path> if you moved it." >&2
    exit 1; }
  # ⚠ THE PATH IS ALREADY RESOLVED AND ITS CONTENT ALREADY PINNED in section
  # 0-pin-b (ZMIX_TFWEIGHTS_SHA256 / ZMIX_TFWEIGHTS_V2_SHA256), so the default
  # above is a no-op on the ship path. The checks below are defence in depth and
  # are NOT the content gate: magic alone cannot distinguish the right blob from
  # a wrong one carrying the same 8 bytes.
  # Verify it really is a weights blob before baking it into an artifact: the
  # first 8 bytes are the FX2TFWC1/FX2TFWC2 magic, or FX2TFWC3 (which needs
  # -Dtf-weights-v3).
  MAGIC=$(head -c 8 "$ZMIX_TFWEIGHTS")
  case "$MAGIC" in
    FX2TFWC1|FX2TFWC2) ;;
    FX2TFWC3)
      [[ " $ZMIX_ENGINE_FLAGS " == *" -Dtf-weights-v3=true "* ]] || {
        echo "FATAL: '$ZMIX_TFWEIGHTS' is FX2TFWC3 but the recipe has no -Dtf-weights-v3=true;" >&2
        echo "       the engine would die 'bad magic' on the first -c." >&2; exit 1; } ;;
    *) echo "FATAL: '$ZMIX_TFWEIGHTS' is not a weights blob (magic '$MAGIC', want FX2TFWC1/2/3)" >&2; exit 1 ;;
  esac
  CTW="$DIR/comp_tfweights"
  cp "$ZMIX_TFWEIGHTS" "$CTW"
  TW=$(wc -c < "$CTW")
  TFW_ARG=("$ZMIX_TFWEIGHTS")
  echo "comp_tfweights: $TW bytes (magic $MAGIC) <- $ZMIX_TFWEIGHTS"
fi

# ---- 4) self-compress the assets with our OWN no-dict codec ----------------
# (construct.sh:67-72). The cached assets live at run/comp_*.<fp8>; the filename
# also carries the SLOT KIND, because a carried-dict and a derived-recipe build
# of the same engine would otherwise share one path and silently ship the wrong
# asset (the regression above was a content mix-up; a stale cache is
# the same bug with a longer fuse). ⚠ THE FILENAME IS NOT THE KEY — see 4-key.
# ---- 4-key) THE CACHE KEY MUST COVER EVERY INPUT THE MINTED ASSETS DEPEND ON.
# $FP8 is the sha256 of the stripped pre-UPX engine, so it covers the code and
# the recipe FLAGS (every lever is comptime). It does NOT cover:
#   * THE TRANSFORMER WEIGHTS BLOB. It is handed to `-c` as a RUNTIME argument
#     ($TFW_ARG below), not compiled in, so swapping it changes every prediction
#     the asset compressor makes while $FP8 stands still. Observed when a new
#     weights blob was adopted: the construct took three
#     `cache hit` lines and died at VERIFY with `error: MissingTrailingNewline`
#     — the packaged comp9 could not read assets its own decoder had not
#     written. Fail-CLOSED, but only at the last gate, after the whole build.
#   * THE CONTENT OF THE ASSET SOURCES. $DICT_SLOT_KIND tells a recipe from a
#     dict but not one recipe from another, and both $ORDER and $RECIPE are
#     overridable paths: a re-derived derivdict recipe or a new article order
#     reuses the stale entry under the identical filename.
# ASSET_KEY digests all of them and is stamped beside each cached artifact as
# `<artifact>.key`, written only AFTER that artifact is successfully minted.
# ⛔ THE ARTIFACT FILENAMES ARE DELIBERATELY UNCHANGED: submission/hpja/build.sh
# — THE JUDGE'S OWN BUILD — resolves run/comp_<slot>.<fp8> and
# run/comp_order.<fp8> by those exact names, so renaming them would break the
# entry. A cache minted by an older script carries no .key file and therefore
# MISSES: there is no format in which a stale entry can be read as fresh.
# ★ The judge path is unaffected either way — a sandbox rebuild has no cache.
# This is a dev-box incremental-construct defect, and this is its fix.
_ak_sha() { sha256sum "$1" | cut -d' ' -f1; }
if [[ "$TW" -gt 0 ]]; then AK_TFW="$(_ak_sha "$ZMIX_TFWEIGHTS")"; else AK_TFW=none; fi
ASSET_KEY=$(printf 'engine=%s\nslot=%s:%s\norder=%s\ntfweights=%s\n' \
  "$FP" "$DICT_SLOT_KIND" "$(_ak_sha "$DICT_SLOT_SRC")" "$(_ak_sha "$ORDER")" \
  "$AK_TFW" | sha256sum | cut -d' ' -f1)
# A cache hit requires the artifact to exist AND its stamp to record THIS key.
_ak_hit()   { [[ "${FORCE_REBUILD:-0}" != "1" && -s "$1" && -r "$1.key" \
                 && "$(cat "$1.key" 2>/dev/null)" == "$ASSET_KEY" ]]; }
_ak_stamp() { printf '%s\n' "$ASSET_KEY" > "$1.key"; }
echo "asset-cache key: ${ASSET_KEY:0:16} (engine $FP8, $DICT_SLOT_KIND+order sources, tfweights ${AK_TFW:0:8})"

CD="$DIR/comp_$DICT_SLOT_KIND.$FP8"
CO="$DIR/comp_order.$FP8"
# -Ds7order: the order asset ships as its s7_runlabel re-encoding. Mint it with
# the engine's OWN --s7-encode (which hard-asserts the shipped inverse
# reproduces the raw order byte-exactly), then self-compress THAT. The
# mandatory --extract-assets check below still compares against the RAW
# $ORDER — under -Ds7order the ship binary s7-decodes on extraction, so the
# byte-compare verifies the whole segment -> -d -> s7⁻¹ chain.
ORDER_SRC="$ORDER"
if [[ " $ZMIX_ENGINE_FLAGS " == *" -Ds7order=true "* ]]; then
  ORDER_SRC="$DIR/order_s7.$FP8"
  if _ak_hit "$ORDER_SRC"; then
    echo "order_s7 cache hit for key ${ASSET_KEY:0:16}"
  else
    echo "== s7-encoding order asset with engine $FP8 =="
    "./$DIR/engine" --s7-encode "$ORDER" "$ORDER_SRC"
    _ak_stamp "$ORDER_SRC"
  fi
fi
if _ak_hit "$CD"; then
  echo "comp_$DICT_SLOT_KIND cache hit for key ${ASSET_KEY:0:16}"
else
  echo "== self-compressing $DICT_SLOT_KIND ($DICT_SLOT_SRC) with engine $FP8 (~2-4 min) =="
  "./$DIR/engine" -c "$DICT_SLOT_SRC" "$CD" ${TFW_ARG[@]+"${TFW_ARG[@]}"}
  _ak_stamp "$CD"
fi
if _ak_hit "$CO"; then
  echo "comp_order cache hit for key ${ASSET_KEY:0:16}"
else
  echo "== self-compressing order with engine $FP8 (~4-13 min) =="
  "./$DIR/engine" -c "$ORDER_SRC" "$CO" ${TFW_ARG[@]+"${TFW_ARG[@]}"}
  _ak_stamp "$CO"
fi
DS=$(wc -c < "$CD")
OS=$(wc -c < "$CO")
echo "comp_$DICT_SLOT_KIND: $DS bytes   comp_order: $OS bytes"

# ---- 5) HeaderInfo{dict_size, new_article_order_size, decomp_input_size=0[, tfweights_size]} --
# (construct.sh:77; transformer-era adds the fourth field)
if [[ "$TW" -gt 0 ]]; then
  "./$DIR/engine" -h "$DS" "$OS" 0 "$TW" "$DIR/header.dat"
else
  "./$DIR/engine" -h "$DS" "$OS" 0 "$DIR/header.dat"
fi

# ---- 6) assemble the physical Form-1 comp9 (construct.sh:80) ----------------
# $CTW is unquoted-empty on LSTM-era so the member list is unchanged there.
cat "$DIR/decomp_bin" "$CD" "$CO" ${CTW:+"$CTW"} "$DIR/header.dat" > "$DIR/comp9"
chmod +x "$DIR/comp9"
# Compatibility copy for existing fleet/report recipes. `run/comp9` is the
# canonical Form-1 artifact; both files are byte-identical.
cp "$DIR/comp9" "$DIR/zmix_ship_dist"
chmod +x "$DIR/zmix_ship_dist"
COMP9=$(wc -c < "$DIR/comp9")

# ---- 7) MANDATORY verification: comp9 must reproduce its own assets --------
# ⚠⚠ NEVER /tmp HERE — MEASURED IN THE REAL HPJA BUILD CONTAINER,
# where this line FAILED THE ENTRY OUTRIGHT. In the judging image /tmp is a
# SYMLINK to /work/run/tmp (Dockerfile:75), and the harness creates that path
# with `mkdir -p "$active_work_dir/run/tmp"` chmodding ONLY THE LEAF to 1777
# (build-compressor.sh:128-129) — so the intermediate `run` inherits the
# CALLER'S UMASK. A judge whose root umask is 077 (what
# `sudo` hands a non-interactive session — the normal judge path, since
# judging_assistance.sh re-execs itself under sudo) gets /work/run mode 0700;
# UID 65532 cannot traverse it, /tmp is unreachable, and this gate dies with a
# bare `error: AccessDenied`, rejecting the entry at stage `compressor_build`.
# ★ LSTM-era cleared the IDENTICAL gate on 08-20 only because its run happened to
# have umask 022 — the entry's build outcome was a property of the JUDGE'S
# SHELL, not of the entry. Neither the Dockerfile symlink nor the harness block
# changed between 24f01f8 and ba817a2.
# Prefer TMPDIR (build.sh points it at /work/tmp, inside the one writable
# mount); fall back to this build's own output directory, writable by
# construction because comp9 was just written there. NEVER an absolute path
# outside the tree — the same defect class as the fp_prior absolute-path
# submission blocker, one level up the stack. Zero effect on emitted bytes.
XT="${TMPDIR:-$DIR}"
mkdir -p "$XT"
XD="$XT/zmix-xa-d.$$"
XO="$XT/zmix-xa-o.$$"
trap 'rm -f "$XD" "$XO"' EXIT
echo "== VERIFY: comp9 --extract-assets round-trip (~2-13 min) =="
"./$DIR/comp9" --extract-assets "$XD" "$XO" || {
  echo "FATAL: --extract-assets FAILED (exit $?) — the packaged comp9 cannot read its own tail" >&2
  exit 1
}
# ⚠ Compare against the DECLARED SLOT SOURCE, never against $DICT
# unconditionally. Comparing the recipe slot against english.dic is what let the
# derivdict regression through this gate: the artifact really did
# hold english.dic, so the check passed, and the ONE mandatory gate on this
# script certified an artifact that could not encode.
cmp "$XD" "$DICT_SLOT_SRC" || {
  echo "FATAL: extracted $DICT_SLOT_KIND != $DICT_SLOT_SRC — packaging/codec MISMATCH, dist is broken" >&2
  echo "       (dictionary mode is '$FORM1_DICT_MODE'; comp9's first tail slot must hold the $DICT_SLOT_KIND)" >&2
  exit 1
}
cmp "$XO" "$ORDER" || {
  echo "FATAL: extracted order != $ORDER — packaging/codec MISMATCH, dist is broken" >&2
  exit 1
}
echo "VERIFY OK: extracted $DICT_SLOT_KIND+order byte-identical to originals"

# ---- 8) sizes, hashes, manifest --------------------------------------------
# Form-1 counts this complete compressor once. archive9.exe, produced by
# construct_form1.sh, separately repeats decomp_bin+comp_dict and adds payload.
# transformer-era adds comp_tfweights and a 4-byte header field; both terms are 0 on LSTM-era.
HDR_SIZE=12
[[ "$TW" -gt 0 ]] && HDR_SIZE=16
COMP9_EXPECTED=$((BIN + DS + OS + TW + HDR_SIZE))
[[ "$COMP9_EXPECTED" == "$COMP9" ]] || { echo "FATAL: component sum ($COMP9_EXPECTED) != comp9 size ($COMP9) — assembly bug" >&2; exit 1; }
if [[ " $ZMIX_ENGINE_FLAGS " == *" -Dr1v2=true "* ]]; then
  FORM1_ARCHIVE_ORDER=embedded
else
  FORM1_ARCHIVE_ORDER=omitted
fi
# Arm E': under -Dderivdict comp9's first tail slot is the RECIPE, not
# english.dic's compressed form, so the two submitted files no longer share that
# segment. Packaging cannot infer this from the bytes (the trailer carries three
# sizes and no kinds), so the BUILD declares it (FORM1_DICT_MODE, set with
# DICT_SLOT_SRC above so the declaration and the bytes are chosen ONCE) and
# construct_form1.sh refuses to disagree with the declaration.

SHA_BIN=$(sha256sum "$DIR/decomp_bin"     | cut -d' ' -f1)
SHA_CD=$(sha256sum "$CD"                  | cut -d' ' -f1)
SHA_CO=$(sha256sum "$CO"                  | cut -d' ' -f1)
SHA_CTW=""
if [[ "$TW" -gt 0 ]]; then SHA_CTW=$(sha256sum "$CTW" | cut -d' ' -f1); fi
SHA_COMP9=$(sha256sum "$DIR/comp9"          | cut -d' ' -f1)
SHA_DIST=$(sha256sum "$DIR/zmix_ship_dist" | cut -d' ' -f1)
[[ "$SHA_COMP9" == "$SHA_DIST" ]] || { echo "FATAL: compatibility copy differs from comp9" >&2; exit 1; }

# (The tree pin used to be asserted HERE, after the whole build. It now runs in
# section 0-pin, before the toolchain is even resolved: a strict pin that only
# fires after a multi-minute build is a strict pin nobody leaves armed, and the
# negative control is unaffordable to run. Nothing it reads is produced by the
# build, so moving it earlier is free. It is recorded in the manifest below.)

MANIFEST="$DIR/MANIFEST.txt"
{
  echo "zmix ship manifest"
  echo "generated:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head:           $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet HEAD 2>/dev/null || echo ' (dirty)')"
  echo "engine_src_pin:     ${ZMIX_SRC_DIGEST:-<none>} (asserted, mode=${ZMIX_TREE_PIN_MODE:-warn})"
  echo "tree_pin:           ${ZMIX_TREE_SHA:-<none>} (informational, NOT asserted)"
  echo "zig:                $GOT_ZIG (pinned $WANT_ZIG)  path=$ZIG"
  echo "upx:                $(upx_run --version | head -1)  mode=$ZMIX_UPX_MODE  flags_used=$UPX_USED  path=$UPX"
  echo
  echo "recipe (SHIP_RECIPE.env):"
  echo "  ZMIX_ENGINE_FLAGS=$ZMIX_ENGINE_FLAGS"
  echo "  ZMIX_ABI=$ZMIX_ABI"
  echo "  ZMIX_ZIG_VERSION=$WANT_ZIG"
  echo "  ZMIX_UPX_MODE=$ZMIX_UPX_MODE"
  echo "  DICT=$DICT"
  echo "  ORDER=$ORDER"
  if [[ "$DICT_SLOT_KIND" == recipe ]]; then
    # The blob comp9 ACTUALLY carries, by content. construct_form1.sh checks its
    # --recipe against this line, so packaging cannot verify one blob while the
    # artifact carries another.
    echo "  RECIPE=$RECIPE   ($(wc -c < "$RECIPE") bytes, Arm E' recipe in comp9's dict slot)"
    echo "  recipe_sha256: $(sha256sum "$RECIPE" | cut -d' ' -f1)"
  fi
  if [[ "$ORDER_SRC" != "$ORDER" ]]; then
    echo "  ORDER_S7=$ORDER_SRC   ($(wc -c < "$ORDER_SRC") bytes, raw s7_runlabel container fed to -c)"
  fi
  if [[ "$TW" -gt 0 ]]; then
    # The transformer-era weights blob is an ASSET, not a compiler input, so it is outside
    # ZMIX_SRC_DIGEST by design and is pinned by its own sha (section 0-pin-b).
    # Recorded here so the pin that was asserted lands in every construct log.
    echo "  ZMIX_TFWEIGHTS=$ZMIX_TFWEIGHTS   ($TW bytes, magic $MAGIC)"
    echo "  tfweights_sha256: ${TFW_SHA_HAVE:-<unmeasured>}"
    if [[ -z "${TFW_SHA_WANT:-}" ]]; then
      echo "  tfweights_pin:    <none> — the weights blob was NOT asserted"
    elif [[ "${TFW_SHA_HAVE:-}" == "$TFW_SHA_WANT" ]]; then
      echo "  tfweights_pin:    $TFW_SHA_VAR MATCHES (asserted, mode=${ZMIX_TREE_PIN_MODE:-warn})"
    else
      echo "  tfweights_pin:    $TFW_SHA_VAR MISMATCH — recipe wants $TFW_SHA_WANT (mode=${ZMIX_TREE_PIN_MODE:-warn})"
    fi
  fi
  echo
  echo "engine_fingerprint: $FP   (sha256 of stripped PRE-UPX engine; asset-cache key .$FP8)"
  echo "engine_stripped:    $RAW bytes"
  echo
  if [[ "$TW" -gt 0 ]]; then
    echo "components (comp9 = decomp_bin ++ comp_$DICT_SLOT_KIND ++ comp_order ++ comp_tfweights ++ header):"
  else
    echo "components (comp9 = decomp_bin ++ comp_$DICT_SLOT_KIND ++ comp_order ++ header):"
  fi
  echo "  decomp_bin:  $BIN bytes   sha256=$SHA_BIN   (strip+upx $UPX_USED)"
  printf '  comp_%-7s %s bytes   sha256=%s   (compressed %s)\n' "$DICT_SLOT_KIND:" "$DS" "$SHA_CD" "$DICT_SLOT_SRC"
  echo "  comp_order:  $OS bytes   sha256=$SHA_CO"
  if [[ "$TW" -gt 0 ]]; then
    echo "  comp_tfweights: $TW bytes   sha256=$SHA_CTW   (transformer-era int4 blob, verbatim copy of $ZMIX_TFWEIGHTS)"
  fi
  echo "  header:      $HDR_SIZE bytes"
  echo
  echo "comp9:              $COMP9 bytes   sha256=$SHA_COMP9"
  echo "zmix_ship_dist:     $COMP9 bytes   sha256=$SHA_DIST   (compatibility copy)"
  if [[ "$TW" -gt 0 ]]; then
    echo "|comp9| = decomp_bin($BIN) + comp_$DICT_SLOT_KIND($DS) + comp_order($OS) + comp_tfweights($TW) + header($HDR_SIZE) = $COMP9_EXPECTED bytes"
  else
    echo "|comp9| = decomp_bin($BIN) + comp_$DICT_SLOT_KIND($DS) + comp_order($OS) + header($HDR_SIZE) = $COMP9_EXPECTED bytes"
  fi
  echo
  if [[ -n "$ARMC" ]]; then
    # The SUBMITTED prefix. run/comp9 above stays the full-CLI asset/header
    # helper; it is not the file that is scored.
    echo "submitted arm-C prefix (tools/build_armc_prefix.sh):"
    echo "  armc_prefix: $ARMC_BIN bytes   sha256=$ARMC_SHA   path=$ARMC"
    if [[ "$TW" -gt 0 ]]; then
      echo "  |comp9_min| = armc_prefix($ARMC_BIN) + comp_$DICT_SLOT_KIND($DS) + comp_order($OS) + comp_tfweights($TW) + header($HDR_SIZE) = $((ARMC_BIN + DS + OS + TW + HDR_SIZE)) bytes"
    else
      echo "  |comp9_min| = armc_prefix($ARMC_BIN) + comp_$DICT_SLOT_KIND($DS) + comp_order($OS) + header($HDR_SIZE) = $((ARMC_BIN + DS + OS + HDR_SIZE)) bytes"
    fi
    echo "  helper-prefix premium if the arm-C prefix is NOT used: $(( 2 * (BIN - ARMC_BIN) )) bytes of S (decomp_bin is charged twice)"
    echo "  next: ./construct_form1.sh --archive-order $FORM1_ARCHIVE_ORDER --minimal-prefix $ARMC --comp9 $DIR/comp9 ..."
  else
    echo "submitted arm-C prefix: NOT BUILT (ZMIX_SKIP_ARMC=1) — run/comp9 is not submittable as-is"
  fi
  echo "Form-1 archive-order layout required by recipe: $FORM1_ARCHIVE_ORDER"
  echo "Form-1 dictionary mode required by recipe: $FORM1_DICT_MODE"
  echo "Form-1 total S is emitted only after construct_form1.sh --archive-order $FORM1_ARCHIVE_ORDER creates archive9.exe."
  echo
  echo "verification:       --extract-assets round-trip PASS ($DICT_SLOT_KIND+order byte-identical to $DICT_SLOT_SRC / $ORDER)"
} > "$MANIFEST"

echo
cat "$MANIFEST"
