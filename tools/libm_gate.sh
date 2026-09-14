#!/usr/bin/env bash
# tools/libm_gate.sh <artifact> [<artifact> ...]
#
# SUBMISSION GATE — refuses any artifact that IMPORTS a math function from the
# system libm.
#
# WHY:
# `archive9.exe` is encoded on OUR machine and must self-extract on the judge's.
# Every value on the coder path must therefore come from inside the binary: two
# hosts whose `sinf`/`expf`/`powf` differ in the last mantissa bit desynchronise
# the arithmetic decoder, silently, and the stream decodes to garbage. It is the
# same failure the reciprocal-estimate P0 covers (tools/recip_gate.sh) one layer
# out — that gate greps OPCODES and is structurally blind to an imported SYMBOL.
# The instance that motivated this file: `model_opt.cpp`'s RoPE beyond-table
# fallback called std::sin/std::cos, which GCC contracted into a single
# `sincosf@GLIBC_2.2.5` — the shipped decoder's only libm math import, present
# in earlier transformer-era artifacts of this entry's line, and invisible to
# every gate that then existed.
#
#   exit 0 = clean    exit 1 = LIBM MATH IMPORT FOUND    exit 2 = gate could not run
#
# ⚠ THE TRAP THIS GATE IS BUILT AROUND, and it is the same one recip_gate.sh
# names: `nm -D` on a UPX-packed or section-header-stripped image prints
# "no symbols" and exits NONZERO — which, read carelessly, is a FALSE CLEAN.
# So a "clean" verdict only counts after a POSITIVE CONTROL: the scan must see a
# plausible number of dynamic imports AND a symbol we know this engine needs.
# Scan the UNPACKED image (upx -d first); this gate refuses a packed one.
set -uo pipefail

# Everything libm exports that is a MATH function. Deliberately a whitelist-free
# blacklist by name shape: any of these, with or without an f/l suffix, plus the
# __*_finite aliases glibc emits under -ffast-math.
MATH_RE='^(__)?(a?(cos|sin|tan)h?|atan2|cbrt|ceil|copysign|cosh|drem|erfc?|exp2?|exp10|expm1|fabs|fdim|floor|fma|fmax|fmin|fmod|frexp|gamma|hypot|ilogb|j[01n]|ldexp|lgamma|llrint|llround|log|log10|log1p|log2|logb|lrint|lround|modf|nan|nearbyint|nextafter|nexttoward|pow|pow10|remainder|remquo|rint|round|scalb|scalbln|scalbn|significant|sincos|sinh|sqrt|tanh|tgamma|trunc|y[01n])(f|l|f32|f64|f32x|f64x|f128)?(_finite)?$'
# ⚠ NOT in the list on purpose: printf/scanf family (they format doubles but are
# libc, not libm, and are unavoidable), and `abs`/`labs` (integer, in libc).

rc=0
for art in "$@"; do
    [[ -r "$art" ]] || { echo "libm-gate: cannot read $art" >&2; exit 2; }

    # --- read the dynamic symbol table --------------------------------------
    # nm exits 1 with "no symbols" on a packed/stripped image; capture rather
    # than pipe, so the exit status is the one we mean to read. (
    # `| grep -q` under `pipefail` is a SIGPIPE race — do not reintroduce it.)
    syms="$(nm -D --undefined-only "$art" 2>/dev/null)"
    nm_rc=$?
    n_und=0
    [[ -n "$syms" ]] && n_und=$(printf '%s\n' "$syms" | grep -c .)

    # --- POSITIVE CONTROL: the scan must have actually seen something --------
    if (( nm_rc != 0 )) || (( n_und < 10 )); then
        echo "libm-gate: REFUSING TO PASS $art — nm -D returned $n_und undefined symbol(s)" >&2
        echo "  (rc=$nm_rc). This is what a UPX-packed or section-header-stripped image" >&2
        echo "  looks like, and it is a FALSE CLEAN, not a clean binary. Run the gate on" >&2
        echo "  the UNPACKED ELF (upx -d first; for a composite, carve the leading" >&2
        echo "  decoder as tools/recip_gate.sh does)." >&2
        exit 2
    fi
    names="$(printf '%s\n' "$syms" | sed 's/.*[[:space:]]//; s/@.*//' | sed '/^$/d')"
    if ! printf '%s\n' "$names" | grep -qx 'memcpy\|memset\|malloc\|mmap64\|read\|write'; then
        echo "libm-gate: REFUSING TO PASS $art — the control symbols (memcpy/memset/" >&2
        echo "  malloc/mmap64/read/write) are all absent from $n_und imports; this does" >&2
        echo "  not look like the engine, so a clean verdict would not mean anything." >&2
        exit 2
    fi

    # --- the gate ------------------------------------------------------------
    hits="$(printf '%s\n' "$names" | grep -E "$MATH_RE" | sort -u)"
    if [[ -n "$hits" ]]; then
        echo "!! LIBM-GATE FAIL [$(basename "$art")]: math function(s) imported from the system libm:" >&2
        printf '     %s\n' $hits >&2
        echo "   A coder-path value from outside the binary desynchronises the arithmetic" >&2
        echo "   decoder between the encoding host and the judging host. Vendor the" >&2
        echo "   implementation (src/vendored_math.zig, third_party/fx2_transformer/rope_trig.h)." >&2
        rc=1
    else
        needed="$(readelf -d "$art" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | tr '\n' ' ')"
        echo "  ok [$(basename "$art")]: 0 libm math imports of $n_und dynamic imports; NEEDED = ${needed:-<none>}"
    fi
done
if (( rc == 0 )); then echo "LIBM-GATE: PASS"; else echo "LIBM-GATE: FAIL"; fi
exit $rc
