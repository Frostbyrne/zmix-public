# zmix — building and running the entry

*Submission document. Written against the assembled HPJA bundle and the shipped
`entry.env`, not from memory.*

---

## 1. What the entry consists of

| file | what it is |
|---|---|
| `comp9` | the compressor. UPX-packed, takes **no arguments** (`comp9.args` is empty ⇒ **0 bytes of S** for command-line options) |
| `archive9.exe` | the **self-extracting** archive. Running it with **no arguments** writes `data9` |
| `zmix-src.tar.gz` | complete source, for the judge's rebuild |
| `entry.env` | declares the above to the harness |

Declared in `entry.env`: `ENTRY_FORMAT=self-extracting`, `EXECUTION_PLATFORM=linux-x86_64`,
`COMPRESSOR_FORMAT=upx-overlay`, `ARCHIVE_FORMAT=upx-overlay`, `DECOMPRESSED_OUTPUT=data9`.

**Score.** `S = |comp9| + |archive9.exe|`. The decompressor and the dictionary appear in **both**
files, so every byte of them is charged **twice** — this is the form every prize winner has used and
is why the entry's design treats binary size as a first-class cost.

---

## 2. Running it

```sh
./comp9                         # compress   — NO ARGUMENTS: reads ./enwik9, writes ./archive9.exe
./archive9.exe                  # decompress — NO ARGUMENTS: writes ./data9
sha256sum data9                 # must equal enwik9's
```

**Both steps take no arguments by design**, which is why `comp9.args` is empty and the
command-line term of the score is 0. `comp9` reads a file named `enwik9` in the current directory
and writes `archive9.exe` beside it; `archive9.exe` writes `data9`. The two are the *same* packed
program image, which selects its role from its own trailing 16-byte record — there is nothing to
configure and nothing outside the binary is read. `src/form1.zig`'s header comment states the same
contract: *"Running comp9 reads fixed `enwik9` and writes fixed `archive9.exe`. Running
archive9.exe writes fixed `data9`."*

> The `.exe` suffix on a Linux executable is a naming mistake. This package is the submitted
> version, `zmix 1.0`, and writes `archive9.exe`; tag `zmix-1.0.1` in the repository writes
> `archive9` and differs in nothing else. See `README.md`.

---

## 3. Rebuilding from source

The judge's rebuild is what the score is booked from, so it is the path that matters:

```sh
tar xzf zmix-src.tar.gz && cd zmix-src
./install.sh                    # harness-sanctioned root phase (toolchain fetch)
./build.sh                      # produces comp9
```

**Every build input is pinned by hash**, including the Zig toolchain itself. The C++ components
compile through Zig's bundled clang, so no host compiler is consulted and `CC`/`CXX` are not read.
The build targets `x86_64-v3` explicitly rather than `-march=native`.

⚠ **The source package is byte-reproducible** — the packaging normalises umask, git's `tar.umask`,
tag objects and the packager's checkout, verified to a single digest across ten environments.

---

## 4. Resource envelope (measured, not projected)

| resource | limit | this entry |
|---|---|---|
| **RAM** | 10 GiB peak RSS | **9.58 GB** measured on a live e9 encode = **0.913×**; passes the strict 10⁹-byte reading too (0.980×) |
| **Temp disk** | 100 GB | **≤ ~15 GB** — a sparse 14.68 GB PPMd arena (8.4 GB resident on disk mid-run); **no other temp file** |
| **Wall clock** | `70'000 / T` per program | **37.845 h at `T = 1825` ⇒ `W = 69,068` = 0.987×** on the judging harness's own formal run; see `README-SUBMISSION.md` §7 |
| **GPU** | not permitted at judging | **none used.** The transformer weights are trained offline and shipped; judging runs CPU-only |

⚠ **The harness's formal `enwik9` run is `--cold-cache` and serial, and both are MANDATORY, not
options.** At the judging-assistant commit this entry is pinned to
(`0bf7580ebe358b80ffc679f7863feaf1ebef88ea`, "Enforce cold-cache formal runs")
`judging_assistance.sh` refuses a one-billion-byte run without `--cold-cache`, and refuses to
combine cache eviction with parallel execution — so the invocation is

    ./judging_assistance.sh --cold-cache --serial --work-root <>=100GB> Entries/zmix ./enwik9

Consequently first-read I/O falls **inside** the timed window. The *bare* wall figures on record
were measured warm-cache; a cold-cache run in a real judge container is in progress. This is
flagged rather than hidden.

---

## 5. Requirements

- Linux x86_64 with **AVX2** (the target is `x86_64-v3`).
- ~16 GiB RAM, no swap.
- ~30 GB free disk for the temp arena plus output, on the volume the program runs from.
- No network access is needed at build or run time; the toolchain is fetched by the sanctioned
  install phase and everything else is in the source package.

---

## 6. Third-party components

See **`THIRD-PARTY-NOTICES.md`**. In short: the transformer inference library and its trained weights
come, with attribution, from the entry `fx2-cmix-transformer` (Vladimir Ivanov, GPL-3); zmix is a port
of `cmix` (Byron Knoll) carrying `fxcm_v26` (Kaido Orav). The prize-division statement is in
`README-SUBMISSION.md` §10.
