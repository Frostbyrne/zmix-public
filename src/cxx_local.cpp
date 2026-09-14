// cxx_local.cpp — supply, LOCALLY and WITHOUT EXCEPTIONS, every libc++ symbol
// the vendored transformer needs, so that no member of libc++.a / libc++abi.a /
// libunwind.a is ever extracted into the transformer-era Form-1 prefix.
//
// WHY (measured):
//   A symbol census of the transformer-era arm-C prefix found, inside a binary compiled
//   `-fno-exceptions -fno-rtti`:
//       581 `itanium_demangle::*` symbols   69,581 B
//       libunwind                           16,101 B
//       __gxx_personality_v0 + cxa_*         ~3,400 B
//   None of it is ours and none of it can ever run. The pull chain is entirely
//   mechanical, and `nm -u` on the nine vendored objects shows how short the
//   root is — the ONLY libc++ symbols they need are:
//       std::string::append(const char*)      \  all three live in libc++.a's
//       std::string::insert(size_t, const char*)  > `string.o`, which is built
//       std::to_string(int)                   /   WITH exceptions
//       std::__next_prime(size_t)                 (libc++.a `hash.o`, ditto)
//       std::__libcpp_verbose_abort(const char*, ...)
//       std::nothrow, operator new/delete         (libc++abi `stdlib_new_delete.o`)
//   Extracting ANY of those members brings in `.eh_frame` with a personality
//   routine, and then:
//       __gxx_personality_v0  (libc++abi cxa_personality.o)
//         -> __cxxabiv1::call_terminate -> std::terminate  (cxa_handlers.o)
//         -> the `__cxa_terminate_handler` global          (cxa_default_handlers.o)
//         -> demangling_terminate_handler— whose OWN translation unit is
//            the entire Itanium demangler, there to pretty-print the type name
//            of an exception this binary cannot throw.
//   `-fno-exceptions` on our TUs does not help: it changes how the HEADERS call
//   `__throw_*`, not how the prebuilt archive members were compiled.
//
// THE FIX is the ordinary static-link one. Our objects are searched before the
// archives, so defining these symbols ourselves means the throwing members are
// never extracted and everything below them leaves with them. This file is
// compiled with the SAME `-fno-exceptions -fno-rtti` flags as the rest of the
// port, so the string code it instantiates aborts where libc++'s would throw —
// which is exactly what libc++ itself does under `_LIBCPP_HAS_NO_EXCEPTIONS`.
//
// ⚠ Attached ONLY under `-Dtransformer=true` (build.zig `attachTransformer`),
// so a stock LSTM-era build links exactly what it linked before.
// ⚠ `tools/check_no_cxx_unwind.sh` is the gate: if a Zig/libc++ bump renames a
// symbol, the archive member silently comes back and the prefix grows again.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

// ---- 1. std::string, instantiated HERE ------------------------------------
// `<string>` carries `extern template` declarations for these members, which is
// what sends the references to libc++.a. An explicit instantiation DEFINITION
// in this TU overrides that and emits them locally, exception-free.
template class std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >;

namespace std {
inline namespace __1 {

// ---- 2. std::to_string(int) ------------------------------------------------
// Same contract as libc++'s (`string.cpp`): the shortest decimal rendering.
string to_string(int __val) {
  char __b[16];
  int __n = ::snprintf(__b, sizeof(__b), "%d", __val);
  return string(__b, __n < 0 ? 0u : static_cast<unsigned>(__n));
}

// ---- 3. std::__libcpp_verbose_abort ----------------------------------------
// libc++'s hardening/assert hook. Contract: print, then do not return.
void __libcpp_verbose_abort(const char* __format, ...) _LIBCPP_VERBOSE_ABORT_NOEXCEPT {
  va_list __ap;
  va_start(__ap, __format);
  ::vfprintf(stderr, __format, __ap);
  va_end(__ap);
  ::abort();
}

// ---- 4. std::__next_prime --------------------------------------------------
// `unordered_map`'s bucket-count hook, out-of-line in libc++.a's `hash.o`.
// Contract (libcxx/src/hash.cpp:67-69, verbatim): "If n == 0, returns 0. Else
// returns the lowest prime number that is greater than or equal to n." Their
// implementation is a 210-wheel; this is plain trial division. Same function,
// so bucket counts — and therefore the map's layout and iteration order — are
// identical. Their overflow branch needs n > 2^64 - 59 and is dropped.
size_t __next_prime(size_t __n) {
  if (__n == 0) return 0;
  if (__n <= 2) return 2;
  if (__n <= 3) return 3;
  size_t __c = __n | 1; // odd, and still >= __n
  for (;; __c += 2) {
    if (__c % 3 == 0) continue;
    bool __prime = true;
    for (size_t __d = 5; __d <= __c / __d; __d += 6) {
      if (__c % __d == 0 || __c % (__d + 2) == 0) { __prime = false; break; }
    }
    if (__prime) return __c;
  }
}

} // namespace __1
} // namespace std

// ---- 5. std::nothrow and the replaceable allocation functions --------------
// libc++abi's `stdlib_new_delete.o` defines these; its `operator new` throws
// `std::bad_alloc` and consults `std::get_new_handler`, which is what reaches
// `cxa_default_handlers.o` (the demangler's TU) even when nothing else does.
// malloc/free + abort matches this engine's existing discipline: every Zig
// allocation site is already `catch unreachable`, and the transformer's arena
// sizes come from a weights header `weights_io.cpp` has already validated.
namespace std {
const nothrow_t nothrow{};
}

static void* zmix_cxx_alloc(std::size_t __n) {
  void* __p = ::malloc(__n ? __n : 1); // operator new(0) must return non-null
  if (!__p) ::abort();
  return __p;
}

static void* zmix_cxx_alloc_aligned(std::size_t __n, std::size_t __a) {
  if (__a < sizeof(void*)) __a = sizeof(void*);
  std::size_t __sz = __n ? ((__n + __a - 1) / __a) * __a : __a;
  void* __p = ::aligned_alloc(__a, __sz);
  if (!__p) ::abort();
  return __p;
}

void* operator new(std::size_t n) { return zmix_cxx_alloc(n); }
void* operator new[](std::size_t n) { return zmix_cxx_alloc(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  return zmix_cxx_alloc_aligned(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return zmix_cxx_alloc_aligned(n, static_cast<std::size_t>(a));
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return ::malloc(n ? n : 1); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return ::malloc(n ? n : 1); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  if (static_cast<std::size_t>(a) < sizeof(void*)) a = static_cast<std::align_val_t>(sizeof(void*));
  std::size_t al = static_cast<std::size_t>(a);
  return ::aligned_alloc(al, n ? ((n + al - 1) / al) * al : al);
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t& t) noexcept {
  return ::operator new(n, a, t);
}

void operator delete(void* p) noexcept { ::free(p); }
void operator delete[](void* p) noexcept { ::free(p); }
void operator delete(void* p, std::size_t) noexcept { ::free(p); }
void operator delete[](void* p, std::size_t) noexcept { ::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { ::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { ::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { ::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { ::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { ::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { ::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { ::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { ::free(p); }

// ---- 6. the `__throw_*` hooks ---------------------------------------------
// Deliberately NOT here: libc++ tags them `[[abi_tag("ne200100")]]`, and a
// definition written from scratch would silently mangle to a different symbol
// and link against nothing. `src/cxx_noexcept.zig` exports the exact tagged
// names instead — see that file.
