//! slot_loss — LAB-ONLY, output-neutral measurement of the byte-mixer LSTM's
//! OWN slot loss, i.e. the member quality of the ensemble slot that the
//! `shipped-transformer` program proposes to replace.
//!
//! Motivation: the value of replacing the LSTM slot rests on
//! `d(ensemble bits)/d(LSTM-slot bits)`, whose DENOMINATOR has to be measured
//! on the same substrate as the numerator to mean anything. The
//! `fx2-cmix-transformer` artifact quotes its slot in nats/byte (LSTM 1.23 ->
//! transformer 0.911); this module produces exactly that statistic for our own
//! engine, on the stream the coder actually codes.
//!
//! What is measured, per coded byte of `prep.temp`, at the top of
//! `ByteMixer.byteUpdate` (BEFORE the LSTM is trained on the byte, so it is a
//! strictly causal prequential loss, and it is the same distribution the
//! ByteModel binary-searched to produce the mixer input):
//!   * `slot`  = -ln q_lstm(x)   with q_lstm = `base.probs` = the LSTM softmax
//!               AFTER the PRL fold (-Dlstm-prior-form/-rho), i.e. exactly what
//!               the ensemble receives in the slot.
//!   * `ppmd`  = -ln p_ppmd(x)   with p_ppmd = `inputs` BEFORE the x2 scale =
//!               the auxiliary byte-model distribution the LSTM is handed as
//!               input (num_models = 1 => this is PPMd's compacted byte
//!               distribution). Free, and it separates "the LSTM's slot is good
//!               because the LSTM is good" from "...because its input is good".
//!
//! ⚠ COMPTIME-DEAD AT DEFAULT. `on` is false unless `-Dslot-loss=true`, so every
//! body below DCEs away and stock archives stay bit-identical. The knob lives in
//! its OWN options module (`slotloss_options`), never `model_opts` —
//! the own-module rule: a shared-`model_opts` entry costs ~40 B of `decomp_bin`, charged TWICE
//! under Form-1, even when dead. An own-module knob costs S exactly zero.
//!
//! ⚠ LAB-ONLY. Nothing in the coding path reads these counters; the accumulate
//! is pure. `byte_mixer.zig` lives only in the `root` module graph, so there is
//! exactly one instance of this state (contrast `ram_census.zig`'s 8.4x
//! under-read, which came from a counter shared across four module graphs).

const std = @import("std");
const opts = @import("slotloss_options");

pub const on: bool = opts.slot_loss;

/// Emit a cumulative line every `report_every` coded bytes. Every arm codes the
/// same stream, so all arms print at IDENTICAL byte counts and the comparison is
/// exact at a common N.
///
/// ⚠ THE PERIODIC LINE ALONE IS NOT ENOUGH, and the first version of this file
/// learned it the expensive way (S1 §3.1): the last periodic line
/// covers the stream only up to the last multiple of `report_every`, while the
/// ARCHIVE it is compared against covers all of it. At `e8_1m` that left the slot
/// total understated by up to 11 %, which turned the report's central ratio from
/// a point into a band. `dump("final")` from `ByteMixer.destroy` closes it —
/// exactly the same class as an end-of-stream checkpoint rule, where a
/// periodic mint alone strands the deepest pack.
const report_every: u64 = 1 << 14;

const State = struct {
    slot_nats: f64 = 0,
    ppmd_nats: f64 = 0,
    n: u64 = 0,
};

var st: State = .{};

/// `q` = LSTM slot probability of the byte just coded; `p` = the auxiliary
/// (PPMd) probability of the same byte. Both may be 0 (PPMd assigns zero mass
/// outside its context set); clamp at 2^-40 so a zero cannot poison the sum —
/// it is charged 40 bits, which is above any real coded cost and therefore
/// pessimistic for whichever arm hits it.
pub inline fn note(q: f32, p: f32) void {
    if (comptime !on) return;
    const floor: f64 = 9.094947017729282e-13; // 2^-40
    st.slot_nats += -@log(@max(@as(f64, q), floor));
    st.ppmd_nats += -@log(@max(@as(f64, p), floor));
    st.n += 1;
    if (st.n % report_every == 0) dump("tick");
}

pub fn dump(tag: []const u8) void {
    if (comptime !on) return;
    if (st.n == 0) return;
    const n: f64 = @floatFromInt(st.n);
    const ln2: f64 = 0.6931471805599453;
    // nats/byte, bits/byte, and the cumulative slot cost in BYTES — the last is
    // the denominator of the pass-through derivative and is quoted directly so
    // no arithmetic is redone downstream.
    std.debug.print(
        "SLOTLOSS {s} n={d} slot_nats_per_B={d:.6} slot_bits_per_B={d:.6} ppmd_nats_per_B={d:.6} ppmd_bits_per_B={d:.6} slot_cost_B={d:.1} ppmd_cost_B={d:.1}\n",
        .{
            tag,
            st.n,
            st.slot_nats / n,
            st.slot_nats / n / ln2,
            st.ppmd_nats / n,
            st.ppmd_nats / n / ln2,
            st.slot_nats / ln2 / 8.0,
            st.ppmd_nats / ln2 / 8.0,
        },
    );
}
