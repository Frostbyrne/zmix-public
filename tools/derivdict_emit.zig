//! `zig build derivdict-emit -- <corpus> <english.dic> <out.bin>`
//!
//! THE PROVENANCE of `src/models/fxcm26/goldens/derivdict_recipe_E2.bin`, the
//! asset comp9 carries in place of `dict.comp` under `-Dderivdict` (Form-1
//! charges a CARRIED asset twice and a DERIVED one once; only the encoder has
//! enwik9 to derive from).
//!
//! This tool replaces the lab's `make_derivdict_recipe.py`. That script was a
//! SECOND implementation of the vocabulary — a Python `[a-z]+` regex over a
//! case-folded corpus, "fused from the jul-31 two-stage build_vocab.py +
//! make_e2.py" — and a second implementation of a format is a drift bug with a
//! delay fuse. This one links `src/derivdict.zig` and calls the SHIPPED
//! decoder's own `buildVocab` / `readWordSet` / `buildDictFromVocab`, so the
//! producer and the consumer are the same code by construction.
//!
//! Determinism (the property the judge-rebuild argument rests on): the fold is
//! a byte-level ASCII table, tokenisation is a `[a-z]+` byte scan, the
//! vocabulary order is (count desc, token asc) — a TOTAL order — the membership
//! bitmap is built by OR-ing bits, and every other sequence comes from
//! english.dic's own byte order or from a sort of byte strings. No hash-map
//! iteration order, no float, no seed reaches the output.
//!
//! `emitRecipe` refuses to emit a blob that does not rebuild the dictionary
//! byte-for-byte through the shipped reader; this tool additionally re-runs the
//! full `buildDict(corpus, blob)` entry point (the one `-e` calls) so the
//! printed result is a statement about the shipped path, not about a helper.
//!

const std = @import("std");
// Wired to src/derivdict.zig by build.zig's `derivdict_mod` — the SAME file the
// shipped decoder compiles, not a copy.
const derivdict = @import("derivdict");

fn readFile(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    const st = try f.stat();
    const buf = try gpa.alloc(u8, st.size);
    errdefer gpa.free(buf);
    var off: usize = 0;
    while (off < buf.len) {
        const n = try f.read(buf[off..]);
        if (n == 0) return error.UnexpectedEof;
        off += n;
    }
    return buf;
}

pub fn main() !u8 {
    var gpa_state: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);
    if (args.len != 4) {
        std.debug.print(
            "usage: derivdict_emit <corpus> <english.dic> <out.bin>\n" ++
                "  corpus     the 1,000,000,000-byte enwik9 the recipe selects from\n" ++
                "  english.dic the dictionary to reproduce (the ship golden)\n" ++
                "  out.bin    the recipe blob to write\n",
            .{},
        );
        return 2;
    }

    const corpus = try readFile(gpa, args[1]);
    defer gpa.free(corpus);
    const dict = try readFile(gpa, args[2]);
    defer gpa.free(dict);

    // The vocabulary the SHIPPED decoder derives (buildVocab folds in place).
    const entries = try derivdict.buildVocab(gpa, corpus);
    defer gpa.free(entries);
    std.debug.print(
        "vocab: corpus={d} B  distinct [a-z]+ tokens={d}  pool K={d}\n",
        .{ corpus.len, entries.len, derivdict.K },
    );

    const blob = try derivdict.emitRecipe(gpa, entries, dict);
    defer gpa.free(blob);

    // End-to-end re-check through the exact entry point `-e` calls. `corpus` is
    // already folded, and the fold is idempotent, so this is a faithful rerun.
    const rebuilt = try derivdict.buildDict(gpa, corpus, blob);
    defer gpa.free(rebuilt);
    if (!std.mem.eql(u8, rebuilt, dict)) {
        std.debug.print("FATAL: buildDict(corpus, blob) != dictionary — refusing to emit\n", .{});
        return 1;
    }

    var sha: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(blob, &sha, .{});

    const out = try std.fs.cwd().createFile(args[3], .{ .truncate = true });
    defer out.close();
    try out.writeAll(blob);

    std.debug.print(
        "recipe {d} B  sha256 {x}\n  rebuilds {s} ({d} B) BIT-EXACTLY through src/derivdict.zig's own buildDict (verified in-process before emit)\n",
        .{ blob.len, &sha, args[2], rebuilt.len },
    );
    return 0;
}
