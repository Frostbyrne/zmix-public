#!/usr/bin/env bash
# tools/recip_gate.sh <artifact> [<artifact> ...]
#
# SUBMISSION GATE — refuses any artifact containing a reciprocal-ESTIMATE opcode
# or an instruction beyond the declared x86-64-v3 baseline.
#
# WHY:
# RCPPS/RCPSS/RSQRTPS/RSQRTSS are *approximations* whose low mantissa bits come
# from an implementation-defined on-die table. Intel and AMD tables differ. In an
# arithmetic coder one differing bit desynchronises the stream, so an encode on
# our silicon and a decode on the judge's silicon produce different output. The
# `fx2-cmix-transformer` lineage lost its July 2026 submission to exactly this.
# Zig's `@setFloatMode(.optimized)` sets LLVM's `arcp`+`afn`, which licenses the
# substitution; `src/strict_fp.zig` is our layer-1 fix and THIS is our layer-3.
#
#   exit 0 = clean    exit 1 = FORBIDDEN OPCODE FOUND    exit 2 = gate could not run
#
# ⚠ THE TRAP THIS GATE IS BUILT AROUND: `objdump -d` on a UPX-packed or
# section-header-stripped image returns ~3 lines and a FALSE CLEAN. A gate that
# silently passes is worse than no gate. So every scan must clear a POSITIVE
# CONTROL (a minimum instruction count AND the presence of opcodes we know are
# in this engine) before its "clean" verdict is allowed to count.
#
# Accepts, and handles differently:
#   * a plain ELF (pre-UPX `zmix_ship` / `runner_lex`) -> objdump -d
#   * a UPX-packed ELF                                 -> upx -d, then raw disasm
#   * a composite (`comp9`, `archive9.exe` = decomp_bin+assets+payload)
#     -> carve the leading packed decoder, upx -d, then raw disasm
set -uo pipefail

FORBIDDEN='\b(v?rcpps|v?rcpss|v?rsqrtps|v?rsqrtss|vrcp14[ps][sd]|vrsqrt14[ps][sd]|vrcp28[ps][sd]|vrsqrt28[ps][sd])\b'
# Beyond x86-64-v3: AVX-512 shows up as %zmm operands or {%kN} mask operands.
# ⚠ IF THIS EVER FIRES, DIAGNOSE BEFORE DISABLING. Two very different causes:
#   (a) OUR code got AVX-512 codegen  -> a real portability break, fix the build.
#   (b) a statically linked libc pulled in its IFUNC-dispatched AVX-512 memcpy/
#       strlen -> NOT a determinism hazard (those are bit-exact and only execute
#       on hardware that has them), but still an x86-64-v3 declaration question.
#   Measured a `gcc -static` hello-world carries 516 %zmm from glibc,
#   while our own `-mcpu x86_64_v3` ship artifact carries ZERO. So a nonzero count
#   on our artifact means something changed in the build, not in libc's nature.
# ⚠⚠ THE REGISTER-SHAPE TEST ALONE IS NOT ENOUGH — IT READS CLEAN ON A BINARY THAT
#    SIGILLs. Post-v3 ISA extensions
#    that are VEX-encoded use ordinary %ymm/%xmm operands and NO mask register, so
#    `{vex} vpdpwssd` (AVX-VNNI) passes the shape test and then faults on any CPU
#    without VNNI — e.g. it SIGILLs on Tiger Lake while this gate says clean.
#    ⇒ match the MNEMONICS too, not just the register file. Each alternative below
#    is anchored and specific: `vpmadd52` not `vpmadd` (vpmaddwd is SSE2 and legal),
#    `sha1|sha256` not `sha`, `vpdp` for the VNNI/VNNI-INT family.
BEYOND_V3='(%zmm[0-9]|\{%k[0-7]\}'\
'|\bvpdp[bw][su]s?[sd]?\b'\
'|\bvpmadd52[lh]uq\b'\
'|\bvdpbf16ps\b|\bvcvtneps2bf16\b|\bvcvtne[eo](ph|bf16)2ps\b'\
'|\b(tileloadd|tilestored|tilezero|tilerelease|ldtilecfg|sttilecfg|tdpb[a-z0-9]+)\b'\
')'
# Deliberately NOT matched: vaes*/vpclmulqdq/gf2p8*/sha*. They are post-v3 in the
# strict sense but are crypto/CRC paths a STATIC libc can legitimately pull in
# (the same class as the %zmm memcpy note above), and this engine emits none of
# them from its own math. Including them would make the gate cry wolf on a static
# build — and a gate that cries wolf gets switched off, which is the failure mode
# this whole script exists to avoid. The list above is confined to extensions a
# compiler could plausibly emit for OUR floating-point/integer kernels.
# Positive control: opcodes this engine provably contains. If NONE appear, the
# disassembly is not real and "clean" is not a verdict.
CONTROL='\b(vfmadd[0-9]+[ps][sd]|vmulps|vaddps|vmovups)\b'
MIN_INSN=2000

fail=0
scanned_any=0

scan() { # $1=file  $2=label  $3=raw|elf
    local f="$1" label="$2" mode="$3" dis n_insn n_ctl hits beyond
    if [[ "$mode" == raw ]]; then
        dis=$(objdump -D -b binary -m i386:x86-64 "$f" 2>/dev/null)
    else
        dis=$(objdump -d "$f" 2>/dev/null)
    fi
    n_insn=$(printf '%s\n' "$dis" | grep -cE $'^\s+[0-9a-f]+:\t')
    n_ctl=$(printf '%s\n' "$dis" | grep -cE "$CONTROL")

    if (( n_insn < MIN_INSN )) || (( n_ctl < 10 )); then
        echo "GATE-FAIL [$label]: disassembly not credible (${n_insn} insns, ${n_ctl} control hits)."
        echo "  A low count here is the UPX/stripped-section FALSE CLEAN. Refusing to pass."
        fail=1; return
    fi

    hits=$(printf '%s\n' "$dis" | grep -nEi "$FORBIDDEN" || true)
    beyond=$(printf '%s\n' "$dis" | grep -nE "$BEYOND_V3" || true)
    scanned_any=1

    if [[ -n "$hits" ]]; then
        echo "GATE-FAIL [$label]: RECIPROCAL-ESTIMATE OPCODE PRESENT — NOT SUBMITTABLE"
        printf '%s\n' "$hits" | head -40 | sed 's/^/    /'
        echo "    ($(printf '%s\n' "$hits" | wc -l) site(s)). Fix: route the division/sqrt through src/strict_fp.zig."
        fail=1
    fi
    if [[ -n "$beyond" ]]; then
        echo "GATE-FAIL [$label]: instruction beyond x86-64-v3 (AVX-512) present"
        printf '%s\n' "$beyond" | head -20 | sed 's/^/    /'
        fail=1
    fi
    [[ -n "$hits$beyond" ]] || echo "  ok [$label]: ${n_insn} insns scanned, 0 forbidden (control ${n_ctl})"
}

TMP=$(mktemp -d "${TMPDIR:-/var/tmp}/recipgate.XXXXXX") || exit 2
trap 'rm -rf "$TMP"' EXIT

for art in "$@"; do
    [[ -r "$art" ]] || { echo "GATE-FAIL: unreadable: $art"; fail=1; continue; }
    echo "== $art ($(stat -c%s "$art") B)"

    # 1. UPX-packed anywhere in the first 4 KiB? (plain packed ELF or composite)
    if head -c 4096 "$art" | grep -qa 'UPX!'; then
        # Try a straight unpack first (plain packed ELF).
        cp "$art" "$TMP/a.bin"
        if upx -d -qq -o "$TMP/a.un" "$TMP/a.bin" >/dev/null 2>&1 && [[ -s "$TMP/a.un" ]]; then
            scan "$TMP/a.un" "$(basename "$art") [upx-unpacked]" raw
            continue
        fi
        # Composite (comp9 / archive9.exe = decomp_bin + assets + payload).
        # `upx -d` on the whole file fails ("Unpacked 0 files") because of the
        # trailing assets, so the leading packed decoder must be carved out.
        #
        # Its length is DERIVED, not guessed: a UPX-packed ELF ends 36 bytes past
        # the last "UPX!" magic of its own trailer. Verified on the previous entry's
        # decomp_bin (last UPX! 136,992 -> length 137,028) and on a 384,545 B
        # comp9 (144,552 + 36 = 144,588 unpacks cleanly). Every UPX! offset in
        # the file is tried, longest first; a chance "UPX!" byte sequence in the
        # payload cannot mislead the gate because a candidate is accepted only if
        # it ACTUALLY unpacks. The literals afterwards are a fallback for
        # historical artifacts.
        got=0
        CANDS=$(grep -abo 'UPX!' "$art" 2>/dev/null | cut -d: -f1 \
                | awk '{print $1+36}' | sort -rn | head -8)
        for n in $CANDS 137028 143964 144588 142564 135356 135516 134540 145612; do
            (( n <= $(stat -c%s "$art") )) || continue
            head -c "$n" "$art" > "$TMP/p.bin"
            if upx -d -qq -o "$TMP/p.un" "$TMP/p.bin" >/dev/null 2>&1 && [[ -s "$TMP/p.un" ]]; then
                scan "$TMP/p.un" "$(basename "$art") [carved ${n}B, unpacked]" raw
                got=1; break
            fi
            rm -f "$TMP/p.un"
        done
        if (( got == 0 )); then
            echo "GATE-FAIL: $art looks UPX-packed but no known prefix length unpacked."
            echo "  Add this artifact's decomp_bin length to the probe list in $0."
            fail=1
        fi
        continue
    fi

    # 2. Unpacked ELF. Prefer section-based disassembly; fall back to raw if the
    #    section headers were stripped (which is exactly the false-clean case).
    if head -c 4 "$art" | grep -qa $'\x7fELF'; then
        n=$(objdump -d "$art" 2>/dev/null | grep -cE $'^\s+[0-9a-f]+:\t')
        if (( n >= MIN_INSN )); then scan "$art" "$(basename "$art")" elf
        else
            echo "  note: objdump -d yielded ${n} insns (sections stripped) — using raw mode"
            scan "$art" "$(basename "$art") [raw]" raw
        fi
        continue
    fi

    scan "$art" "$(basename "$art") [raw]" raw
done

if (( scanned_any == 0 )) && (( fail == 0 )); then
    echo "GATE-FAIL: nothing was actually scanned."; exit 2
fi
if (( fail )); then
    echo
    echo "RECIP-GATE: FAIL"
    exit 1
fi
echo
echo "RECIP-GATE: PASS"
