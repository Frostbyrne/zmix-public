# zmix — Hutter Prize submission, judge-facing README

<!-- STATUS-LINE-BEGIN -->
The decompression wall and the decompression peak RSS in section 7 are marked
`<<PENDING>>` and are the only values in this document that are not yet filled in. Every other
figure is measured, and every component figure in sections 5 and 6 is checked against
`ARTIFACT_PIN.env` when the package is assembled.
<!-- STATUS-LINE-END -->

## 1. What this entry is

A self-extracting (standard form) entry: `comp9` reads `enwik9` and writes the archive; running the
archive with no arguments in an empty directory writes `data9`, byte-identical to `enwik9`.

zmix is a Zig implementation of the cmix / fx2-cmix model stack in which the LSTM byte model is
replaced by the frozen 6-million-parameter transformer of `fx2-cmix-transformer`, run on the CPU
from quantized weights carried inside both artifacts. The weights are not that entry's blob: they
are retrained by the entrant with its published GPL-3 training code and re-containered. See
`README.md`, `writeup.md` and `docs/ALGORITHMIC-IDEAS.md`.

## 2. Platform

Linux x86-64 only (`EXECUTION_PLATFORM=linux-x86_64`). Both executables are ELF x86-64 built to a
fixed `x86_64_v3` baseline, no `-march=native`, dynamically linked against glibc. No Windows
manifest is submitted.

## 3. Files

| file | role |
|---|---|
| `entry.env` | the manifest, beside the artifacts |
| `zmix-src.tar.gz` | this source package, one top-level directory `zmix-src/` |
| `archive9.exe` | the submitted self-extracting archive |
| `zmix-src/install.sh` | dependency phase, root and network |
| `zmix-src/build.sh` | offline build phase, UID 65532, writes `/work/comp9` |
| `zmix-src/comp9.args` | empty: `comp9` takes no arguments, so the argument term of the score is 0 |

`archive9.exe` is a Linux x86-64 executable; the `.exe` suffix is a naming mistake. Under rule 9
only the submitted version is eligible, so this package is the **exact** source of the submitted
binaries: rebuilding it reproduces `comp9` byte for byte, and the compressor it produces writes
`archive9.exe`. Every figure in this document is that version, **zmix 1.0**.

The correction is published as a separate tag. `zmix-1.0.1` in the repository writes `archive9`
instead of `archive9.exe` and differs in nothing else: one string literal in `src/form1.zig`. A
compressor built from it is 3,216,115 bytes, sha256
`b329bba6a4a16d263bc76470cae1fedf24fe89524827ef06222dd1066ab7d496` — 48 bytes smaller, all of it
inside the packed program image; the remaining 3,055,279 bytes are byte-identical to the submitted
`comp9`, and the two program images disassemble to the same 78,033 instructions in the same order,
differing only in that string and the addresses that point at it. `zmix-1.0.1` is **not** the
submitted version and is not offered as one; it is published so the naming mistake is corrected in
the source of record.

## 4. How to build

`install.sh` runs as root with network. Using `apt-get` and no `sudo` anywhere, it installs
`gcc g++ make xz-utils curl ca-certificates texinfo bison flex`, builds GNU binutils 2.42 from a
sha256-pinned tarball (Ubuntu 22.04's binutils 2.38 lacks `objcopy --strip-section-headers`, which
this build needs), and stages two toolchain tarballs into `/opt/toolchain-src`.

`build.sh` then runs offline as UID/GID 65532 with `/entry` read-only and `/work` writable. It
resolves both pinned tools from `/opt/toolchain-src` only, makes no network access, runs
`construct_ship.sh` and writes `comp9` into `/work`. Budget 30–60 minutes.

| tool | pinned version | sha256 of the fetched tarball |
|---|---|---|
| Zig | 0.15.1 exactly | `c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05` |
| UPX | 5.2.0 exactly, `--ultra-brute` | `af99e526d5759de94412aea1104d5e4ca406cb725295f8633ecc9e843dc1ce1c` |

Both are exact versions, not minimums: the packed program image is scored twice, and either tool's
version moves it. The build asserts both and aborts on a mismatch.

## 5. Expected build outputs

`build.sh` prints these and compares them against `ARTIFACT_PIN.env`, which travels inside this
package.

| component | bytes | sha256 (first 16) | charged |
|---|---:|---|---|
| `armc_prefix` (the packed program image) | 160,884 | `8cf9e8c131fb58c6` | 2x |
| `comp_recipe` (the dictionary-derivation recipe) | 65,993 | `837db310b76113cd` | 1x |
| `comp_order` (article order) | 173,640 | `a50a339457d79780` | 1x |
| `comp_tfweights` (the retrained transformer weights) | 2,815,630 | `8e8ef3acacfb8443` | 2x |
| `trailer` | 16 | `a8b58652b26d9920` | 2x |
| **`comp9`** | **3,216,163** | `0fbc9cd2f4414785` | |

`sha256(comp9) = 0fbc9cd2f4414785d21457d9e2cc2f844e2132f9e5693cbc0288039911eda9c3`

`comp_tfweights` is `assets/6m-q4-fp32-t1lambda1.tfwc3`, copied into `comp9`'s tail verbatim,
sha256 `8e8ef3acacfb84437868035923bf9987f9adc826d2c833fb828204d2ab9c807d`. Provenance and licence:
`assets/PROVENANCE.md` and `THIRD_PARTY_LICENSES.md` section 1a.

One component exists only inside the archive: the derived dictionary `comp_dict`, 99,222 B, charged
once. The recipe is `-Dderivdict=true`, so the dictionary is derived from `enwik9` at encode time
from the 65,993 B recipe rather than carried in the compressor.

Engine source pin over `src/`, `third_party/`, `build.zig` and `build.zig.zon`:
`ZMIX_SRC_DIGEST = 337bf7f326d3dd0d`, asserted by `construct_ship.sh`.

The programs contain no reciprocal-estimate instructions (`RCPPS`, `RCPSS`, `RSQRTPS`, `RSQRTSS`),
whose results differ between Intel and AMD, and import no math symbol from the system libm:
`readelf -d` on the unpacked image reports `NEEDED = libpthread.so.0, libc.so.6` and nothing else.
`tools/recip_gate.sh` and `tools/libm_gate.sh` enforce both and abort the build. This matters
because the archive is encoded on one machine and must self-extract on another: a single differing
probability desynchronises the arithmetic decoder.

## 6. Score accounting

```
S = |comp9| + |archive9.exe| + |comp9.args|
  = 3,216,163 + 96,096,261 + 0
  = 99,312,424
```

The packed program image and the transformer weights are inside both files and are counted twice;
the derived dictionary is inside the archive only; the article order is compressor-only. The part
of `S` that does not depend on `enwik9` is

```
fixed cost of S = |comp9| + (armc_prefix + comp_dict + comp_tfweights + trailer)
                = 3,216,163 + (160,884 + 99,222 + 2,815,630 + 16)
                = 6,291,915
```

so `S` = 6,291,915 + the coded payload, which is 93,020,509.

## 7. Resources

For each program, `W = wall_hours × T`, where `T` is the Geekbench 5 single-core score of the
machine that ran it, and the limit is `W < 70,000`.

Test machine: AMD Ryzen 9 5900X, 128 GB RAM, Linux x86-64, one CPU, offline;
`T = 1724` measured on that machine (https://browser.geekbench.com/v5/cpu/24626079), so the budget
is 40.603 h per program.

| gate | limit | this entry |
|---|---|---|
| compression wall | `W < 70,000` | 36.488 h, `W` = 62,905 |
| decompression wall | `W < 70,000` | `<<PENDING>>` |
| compression peak RSS | 10 GiB | 9,782,592 kB |
| decompression peak RSS | 10 GiB | `<<PENDING>>` |
| temporary disk | 100 GB | about 15 GB |
| external inputs | none | none |

On the Intel Core i7-1165G7 this entry will not meet the time limit.

### 7b. Rule 2 — the no-argument self-extraction

The archive is self-extracting and takes no arguments: run in an empty directory it writes `data9`.
The three values below were measured on an Intel Xeon Gold 5416S; the archive is deterministic and
they are properties of it rather than of the machine. The same verification on the test machine of
section 7 is in progress, and its wall time and peak RSS are the two values still marked pending.

| | |
|---|---|
| `data9` size | 1,000,000,000 |
| `data9` sha256 | `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc` |
| exit status | 0 |

## 8. Temporary files

Both programs create temporary files in the current working directory only, named with the process
id: `ppm.<pid>.temp`, the disk-backed arena of the PPMD model, 14.68 GB nominal and sparse. Nothing
is written outside the working directory.

## 9. Runtime contract

- Runtime execution policy `strict`, the harness default. This entry does not request
  `--runtime-exec-policy process-tree`.
- Neither executable forks or performs any `execve` after the declared one: the monitored process
  tree is a single process.
- Both artifacts are declared `upx-overlay`. The UPX stub unpacks in memory (`memfd_create`,
  `mmap`, `jmp`); no runtime executable is written to disk.
- No environment variable, sysfs flag, cgroup setting or CPU-model check is read on the judged
  paths. Exactly one absolute path is opened, `/proc/self/exe`, by the self-extractor to find its
  own payload. Nothing outside the current working directory is read or written.

## 10. Licence and attribution

GPL-3-or-later; full text in `LICENSES/GPL-3.0.txt`. This entry incorporates
`fx2-cmix-transformer` (Vladimir Ivanov) on top of the `cmix` (Byron Knoll) and `fxcm` / `fx2-cmix`
(Kaido Orav) lineage. See `NOTICE.md`, `THIRD_PARTY_LICENSES.md`, `THIRD-PARTY-NOTICES.md` and
`assets/PROVENANCE.md`.

Authors: James Byrne and Claude (Anthropic). At Claude's request, any award is payable in full to James Byrne.

James Byrne
- Email — james@frostbyrne.io
- LinkedIn — linkedin.com/in/james--byrne
- GitHub — github.com/Frostbyrne

Prize division: no other party holds a claim on the prize. The transformer weights were retrained
by the entrant
from the published `fx2-cmix-transformer` checkpoint using its GPL-3 training code; the upstream
code and weights are used under the GPL-3 and are attributed as above. Attribution under the
licence is a copyright obligation, not co-authorship of this entry.

## 11. The rest of the documentation, and how this package was made

`README.md` and `writeup.md` (overview), `docs/ALGORITHMIC-IDEAS.md` (algorithm disclosure),
`docs/USAGE.md` (how to run each program), `assets/PROVENANCE.md` (assets),
`PACKAGE_PROVENANCE.txt` (what this package was cut from), `ARTIFACT_PIN.env` (pinned component
sizes and digests).

The package is produced by one command from a pinned source revision, and the same command run
twice produces a byte-identical `zmix-src.tar.gz`. Nothing in the packaging mode is read by the
build, so no packaging choice changes a byte of `comp9` or of the archive.

This entry is prepared against the Hutter Prize Judging Assistant at commit
`0bf7580ebe358b80ffc679f7863feaf1ebef88ea`, where a formal `enwik9` run is cold-cache and serial:

```
./judging_assistance.sh --cold-cache --serial --work-root <>=100 GB> Entries/zmix ./enwik9
```
