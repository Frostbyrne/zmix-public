//! cxx_noexcept.zig — the ABI-TAGGED half of the transformer-era libc++ decoupling.
//! `src/cxx_local.cpp` is the other half and carries the full rationale; read it
//! first. Together they supply every libc++ symbol the vendored transformer
//! objects need, so that NO member of libc++.a / libc++abi.a / libunwind.a is
//! ever extracted into the Form-1 prefix.
//!
//! WHY THIS FILE EXISTS SEPARATELY: libc++ tags `std::__1::__throw_*` with
//! `[[abi_tag("ne200100")]]` / `("nn200100")`. A C++ definition written from
//! scratch mangles to a DIFFERENT symbol and links against nothing — silently,
//! with the archive member coming back and the prefix growing again. Exporting
//! the exact mangled strings from Zig is the one form that cannot drift by
//! accident.
//!
//! WHAT IT IS WORTH (measured on the transformer-era arm-C prefix, the
//! artifact Form-1 charges TWICE): the baseline carried 581 `itanium_demangle`
//! symbols (69,581 B) + libunwind (16,372 B) + `__gxx_personality_v0` and the
//! `__cxa_*` machinery (~3,400 B) inside a binary compiled `-fno-exceptions
//! -fno-rtti` that can never throw. Removing the whole chain:
//!     packed prefix 198,712 -> 167,844 B   = -30,868 B, = -61,736 B of S.
//!
//! ⚠ Referenced ONLY under `-Dtransformer=true` (`mixer/transformer.zig`), so a
//! stock LSTM-era build links exactly what it linked before — VERIFIED, not
//! assumed: its packed prefix is byte-identical across this change
//! (137,940 B, sha 6732f54f…).
//! ⚠ A Zig/libc++ bump can rename any of these. `tools/check_no_cxx_unwind.sh`
//! is the gate that catches the silent fallback; it runs in
//! `tools/build_armc_prefix.sh` and `construct_ship.sh`.

const builtin = @import("builtin");

/// The C++ throw sites are unreachable by construction (see the header note).
/// `abort` matches libc++'s own `_LIBCPP_HAS_NO_EXCEPTIONS` behaviour and
/// keeps a real, debuggable signal rather than a bare `ud2`.
inline fn die() noreturn {
    if (builtin.link_libc) {
        @extern(*const fn () callconv(.c) noreturn, .{ .name = "abort" })();
    } else {
        @trap();
    }
}

fn throwCStr(_: ?[*:0]const u8) callconv(.c) noreturn {
    die();
}

fn throwVoid() callconv(.c) noreturn {
    die();
}

comptime {
    // std::__1::__throw_length_error / __throw_out_of_range / __throw_overflow_error
    // — the out-of-line hooks std::vector, std::string and std::__split_buffer
    // call. Both ABI tags are emitted by the bundled libc++ (`ne` = the
    // `[[noreturn]]` externally-visible form, `nn` = the internal one) and both
    // are referenced by the transformer's TUs.
    @export(&throwCStr, .{ .name = "_ZNSt3__120__throw_length_errorB8ne200100EPKc", .linkage = .strong });
    @export(&throwCStr, .{ .name = "_ZNSt3__120__throw_length_errorB8nn200100EPKc", .linkage = .strong });
    @export(&throwCStr, .{ .name = "_ZNSt3__122__throw_overflow_errorB8ne200100EPKc", .linkage = .strong });
    @export(&throwCStr, .{ .name = "_ZNSt3__122__throw_overflow_errorB8nn200100EPKc", .linkage = .strong });
    @export(&throwCStr, .{ .name = "_ZNSt3__120__throw_out_of_rangeB8ne200100EPKc", .linkage = .strong });
    @export(&throwCStr, .{ .name = "_ZNSt3__120__throw_out_of_rangeB8nn200100EPKc", .linkage = .strong });
    // std::__throw_bad_array_new_length— emitted by `new T[n]`.
    @export(&throwVoid, .{ .name = "_ZSt28__throw_bad_array_new_lengthB8ne200100v", .linkage = .strong });
    @export(&throwVoid, .{ .name = "_ZSt28__throw_bad_array_new_lengthB8nn200100v", .linkage = .strong });
    // std::__1::__throw_bad_alloc— `operator new` failure.
    @export(&throwVoid, .{ .name = "_ZNSt3__117__throw_bad_allocB8ne200100Ev", .linkage = .strong });
    @export(&throwVoid, .{ .name = "_ZNSt3__117__throw_bad_allocB8nn200100Ev", .linkage = .strong });
}
