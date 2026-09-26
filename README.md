# zmix

## TL;DR

This is a submission for the Hutter Prize. zmix is a Zig implementation of the model stack of
Kaido Orav and Byron Knoll's fx2-cmix and Vladimir Ivanov's fx2-cmix-transformer, with retrained
transformer weights and a smaller decompressor. `S = comp9 3,216,163 + archive9.exe 96,096,261 =
99,312,424` bytes, 1.108% below fx2-cmix-transformer's 100,424,672. More details in `writeup.md`.

## Compiling

1. Run `./install.sh` once as root to install requirements. It fetches the two pinned toolchains,
   Zig 0.15.1 and the UPX 5.2.0 sources, into `/opt/toolchain-src`.
2. Run `./build.sh` to compile. It builds UPX from those sources, runs offline, and produces
   `comp9`, the enwik9-specialized compressor: 3,216,163 bytes, sha256
   `0fbc9cd2f4414785d21457d9e2cc2f844e2132f9e5693cbc0288039911eda9c3` — byte-identical to the
   submitted `comp9`.
3. Do not change the build flags in `SHIP_RECIPE.env`. In particular the binary must not contain
   the `RCPPS`/`RSQRTPS` reciprocal approximation instructions, whose results differ on Intel and
   AMD, so that an archive made on one vendor's CPU decompresses to garbage on the other's.
   `tools/recip_gate.sh` checks the compiled binary for them and refuses to build if any remain.

## Compressing

Create an empty directory and place `enwik9` and the `comp9` compressor that compiling produced in
it. Cd into this directory and run:

```bash
./comp9
```

This produces a self-extracting archive `archive9.exe`. Ignore all the other files that it
produces. It writes about 15 GB of temporary files.

## Decompressing

Create an empty directory, place the `archive9.exe` that compression produced in it, and run:

```bash
./archive9.exe
```

This produces file `data9` identical to `enwik9`. Ignore all the other files that it produces.

`archive9.exe` is a Linux x86-64 executable; the `.exe` suffix is a naming mistake. The submitted
version is **zmix 1.0**, and this package is its exact source: a rebuild reproduces the submitted
`comp9` byte for byte, and it writes `archive9.exe`.

Tag `zmix-1.0.1` in the repository corrects the output filename to `archive9` and differs only in
that string. A compressor built from it is 3,216,115 bytes, sha256
`b329bba6a4a16d263bc76470cae1fedf24fe89524827ef06222dd1066ab7d496` — 48 bytes smaller, all of it
inside the packed program image, byte-identical over the remaining 3,055,279 bytes, and the same
78,033 instructions in the same order. **It is not the submitted version**; all timings, memory
figures, the `data9` verification and the score were measured with `zmix 1.0`.

## Test machine

AMD Ryzen 9 5900X, 128 GB RAM, Linux x86-64, Geekbench 5 single core
[`T = 1724`](https://browser.geekbench.com/v5/cpu/24626079).

| | time | `W = h x T` | peak RSS |
| --- | --- | --- | --- |
| compression | 36.49 h | 62,905 | 9,782,592 kB |
| decompression | 36.44 h | 62,822 | 9,370,308 kB |

`data9` sha256 `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc`.
The entry was tested on the AMD Ryzen 9 5900X. On the Intel Core i7-1165G7 it will not meet the
time limit.

## Training the Transformer

The weights are retrained on enwik9 using the training code published with fx2-cmix-transformer.
The architecture, the 205-value byte alphabet and the PPMD input are unchanged; the objective adds
a term for the compressed size of the weights, which ship in a smaller container. The training
scripts are not part of this package.

## Licence

GPL-3-or-later, full text in `LICENSES/GPL-3.0.txt`. zmix is derived from `cmix` (Byron Knoll),
`fx2-cmix` (Kaido Orav) and `fx2-cmix-transformer` (Vladimir Ivanov); attribution and per-file
modification notices are in `NOTICE.md`, `THIRD_PARTY_LICENSES.md` and `THIRD-PARTY-NOTICES.md`.
Authors: James Byrne and Claude (Anthropic). At Claude's request, any award is payable in full to James Byrne.

James Byrne
- Email — james@frostbyrne.io
- LinkedIn — linkedin.com/in/james--byrne
- GitHub — github.com/Frostbyrne

zmix 1.0, September 2026.
