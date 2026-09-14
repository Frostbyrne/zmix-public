# Third-party notices

zmix incorporates work from prior Hutter Prize entries and their upstreams. This file records what
is incorporated, from whom, and under which licence. It is **attribution and declaration**, which
the entrant treats as an explicit condition of building on a prior entry's published work.

⚠ It is a **package-root** file on purpose: a licence file under `src/` or `third_party/` would
move `ZMIX_SRC_DIGEST` and abort the rebuild.

---

## 1. `fx2-cmix-transformer` — Vladimir Ivanov

The pending Hutter Prize entry `fx2-cmix-transformer` (submitted 2026-07-24, S = 100,424,672).

**Licence: GPL-3.** Incorporated as follows — classified by `cmp` against the upstream tree: of
the 23 kernel files, 12 are VERBATIM, 9 are MODIFIED and 2 are zmix-authored around verbatim
excerpts. GPL-3 §5(a) requires a modified file to carry a prominent notice that it was changed;
each one does. The per-file classification is in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) §1b; each modified file states its change in
its own header comment.

| what | where in this tree | detail |
|---|---|---|
| AVX2 transformer inference library | `third_party/fx2_transformer/` | **23 files** from their `cpp_infer/src/` — 12 verbatim, 9 modified, 2 derived (`fx2_shim.{cpp,h}`) |
| Trained transformer weights | `assets/6m-q4-fp32.tfwc2` | 2,930,652 B, sha256 `7f4db6c8c843a7e6264b6a48ed4805e9e431f543df7a9a0ecb37a35a5e4b8860`, magic `FX2TFWC2`, from `models/6m-q4-fp32.tfwc2` |
| The same weights, re-containered | `assets/6m-q4-fp32.tfwc3` | our `FX2TFWC3` container; **the trained values are Ivanov's**, the container is ours |

Per-asset provenance, including shas and the reasoning for shipping the weights inside the source
tree, is in **`assets/PROVENANCE.md`**.

**Consequence of combination**: the combined work is **GPL-3-or-later**, and `LICENSE` says so.
A derivative of GPL-3 code may not be redistributed under GPLv2, and `cmix`, `fx2-cmix` and
`fx2-cmix-transformer` each ship the identical FSF **GPLv3** text (md5
`1ebbd3e34237af26da5dc08a4e440464`). The full GPLv3 text is in the package at
`LICENSES/GPL-3.0.txt`. The submitted entry is open-sourced in full, which the prize rules require
independently.

---

## 2. `cmix` — Byron Knoll · `fxcm_v26` — Kaido Orav (`kaitz`)

zmix is a **Zig port of cmix** using the **`fxcm_v26`** model. Both upstream projects are licensed
under the GNU General Public License. This is already recorded in `LICENSE`; it is restated here so
that a single file answers "what is in this entry that is not ours".

The `cmix-lex` lineage (Ibrahim Marcouch, Kaido Orav, Byron Knoll) is the accepted prior record that
zmix's Form-1 packaging and temp-file usage follow as precedent.

---

## 3. zmix

Everything not listed above. Licence: **GPL-3-or-later**, see `LICENSE` and
`LICENSES/GPL-3.0.txt`.

---

## 4. ⛔ Prize division — OWED, and it requires a third party

The prize rules state, verbatim:

> *"If a decompressor has multiple authors, then a submission must include instructions for dividing
> the prize money. **All authors must agree** on this distribution before any money can be awarded."*

**This entry incorporates a third party's code and trained weights**, so a prize-division statement
is a submission requirement. It is given in `README-SUBMISSION.md` §10: this entry has a single
author, James Byrne, to whom any award is payable in full. The transformer weights shipped here
were retrained by the entrant from the published `fx2-cmix-transformer` checkpoint using that
project's own GPL-3 training code; the upstream code and weights are used under the GPL-3 and are
attributed above. Attribution under the licence is a copyright obligation, not co-authorship of
this entry.

⚠ It is worth noting that the currently-pending `cmix-lex` entry is reported to be held up by an
**open dispute over apportionment** — i.e. this exact clause is the live failure mode in this
competition right now, not a hypothetical one.
