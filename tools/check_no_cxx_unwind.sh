#!/usr/bin/env bash
# tools/check_no_cxx_unwind.sh <elf> — assert that no libc++abi exception /
# unwind / demangle machinery was linked into a transformer-era binary.
#
# WHY THIS IS A GATE AND NOT A COMMENT :
#   `-Dtransformer` sets `link_libcpp = true`. If ANY member of libc++.a or
#   libc++abi.a that was compiled WITH exceptions gets extracted, it brings a
#   `.eh_frame` personality with it, and the pull chain ends in
#   `cxa_default_handlers.o` — the translation unit that contains the ENTIRE
#   Itanium demangler, present only to pretty-print the type name of an
#   exception this binary cannot throw. Measured cost on the transformer-era arm-C prefix:
#       581 itanium_demangle symbols   69,581 B
#       libunwind                      16,101 B
#       __gxx_personality_v0 + cxa_*    ~3,400 B
#   = 30,868 B of PACKED prefix, and Form-1 charges decomp_bin TWICE, so
#   61,736 B of S.
#
#   `src/cxx_local.cpp` + `src/cxx_noexcept.zig` keep those members out by
#   defining every libc++ symbol the vendored objects need. That defence is
#   MANGLED-NAME-BASED: a Zig/libc++ bump can rename `__throw_length_error`'s
#   ABI tag or move `__next_prime`, the reference silently falls back to the
#   archive, and the prefix quietly grows by 30 KB with every other gate green.
#   Exactly the "silent, favourable-looking regression" class this project's
#   own rules exist to catch. Hence a gate.
#
# ⚠ RUN IT ON THE UNPACKED IMAGE, BEFORE `objcopy --strip-section-headers` and
#   BEFORE UPX. Same trap as the reciprocal gate: after
#   either step the symbol/section view is gone and the check returns a FALSE
#   CLEAN. This script refuses to pass a file it cannot actually read.
#
# Exit 0 = clean, 1 = machinery found, 2 = usage/unreadable.
set -uo pipefail

BIN="${1:-}"
[[ -n "$BIN" && -f "$BIN" ]] || { echo "usage: $0 <elf>   (unpacked, pre-strip-section-headers)" >&2; exit 2; }

# Refuse a file we cannot inspect — a UPX-packed or header-stripped image gives
# zero hits for every pattern and would report a clean bill of health.
#
# ⚠⚠ THE REFUSALS BELOW MUST NOT USE `cmd | grep -q`. Under `set -o pipefail`
# (line 32) `grep -q` exits on the FIRST match, `strings` dies of SIGPIPE, and
# the pipeline reports 141 — so `if <pipeline>; then refuse; fi` is FALSE
# EXACTLY WHEN THE PATTERN MATCHES. Measured this gate returned
# `rc=0, cxx-unwind gate OK` on a UPX-packed image carrying the whole demangler.
# That is the FALSE CLEAN this block exists to prevent, produced by the block
# itself. Every check here therefore captures output first, then tests it.
if ! objdump -h "$BIN" >/dev/null 2>&1; then
    echo "FATAL: $BIN has no readable section headers (UPX-packed or --strip-section-headers'd)." >&2
    echo "       This gate MUST run on the unpacked image or it returns a false clean." >&2
    exit 2
fi
# UPX leaves a STRUCTURALLY VALID but EMPTY section table, so the objdump probe
# above succeeds and reports zero sections. A real unpacked engine has ~12.
NSEC=$(objdump -h "$BIN" 2>/dev/null | awk '/^ *[0-9]+ /{n++} END{print n+0}')
if [[ "${NSEC:-0}" -eq 0 ]]; then
    echo "FATAL: $BIN has ZERO sections — it is packed or header-stripped, and every" >&2
    echo "       pattern below would return zero hits (a false clean). Run this gate" >&2
    echo "       on the UNPACKED image, before objcopy --strip-section-headers." >&2
    exit 2
fi
UPXMAGIC=$(strings -a "$BIN" | grep -c '^UPX!')
if [[ "${UPXMAGIC:-0}" -gt 0 ]]; then
    echo "FATAL: $BIN looks UPX-packed ($UPXMAGIC UPX! markers). Run this gate before packing." >&2
    exit 2
fi

FOUND=0
report() { echo "FORBIDDEN: $2 ($1 hits) — $3" >&2; FOUND=1; }

# The demangler and the unwinder both leave unmistakable strings even in a
# fully stripped image (its own diagnostic text and type names).
for pat in itanium_demangle libunwind _Unwind_ __cxa_ __gxx_personality_v0; do
    n=$(strings -a "$BIN" | grep -c -- "$pat")
    [[ "$n" -gt 0 ]] && report "$n" "$pat" "a libc++abi/libunwind archive member was extracted"
done

# Section-level tells: exception tables and the libc++ override marker only
# appear when an exception-built member came along.
for sec in .gcc_except_table .eh_frame; do
    if objdump -h "$BIN" 2>/dev/null | awk -v s="$sec" '$2==s{f=1} END{exit !f}'; then
        sz=$(objdump -h "$BIN" 2>/dev/null | awk -v s="$sec" '$2==s{print strtonum("0x"$3)}')
        # .eh_frame is stripped later in the prefix chain, so a nonzero one here
        # is informational rather than fatal; .gcc_except_table is not.
        if [[ "$sec" == ".gcc_except_table" && "${sz:-0}" -gt 0 ]]; then
            report "$sz B" "$sec" "C++ exception tables are present"
        fi
    fi
done

if [[ "$FOUND" -ne 0 ]]; then
    echo "  Fix: src/cxx_local.cpp / src/cxx_noexcept.zig must define every libc++" >&2
    echo "  symbol the vendored objects need. Re-derive the list with:" >&2
    echo "    nm -u <the fx2_transformer .o files> | sort -u" >&2
    exit 1
fi

echo "cxx-unwind gate OK: no demangler, no libunwind, no exception tables in $BIN"
exit 0
