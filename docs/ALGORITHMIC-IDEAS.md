# zmix — algorithmic ideas

*Submission document for Hutter Prize participation item 8: a description of the ideas in this entry.*

This describes **what the entry actually ships and why it is built that way**.
Where an idea was evaluated and *not* taken, it is labelled as such. The exact shipped
configuration is `SHIP_RECIPE.env`'s `ZMIX_ENGINE_FLAGS`, which is a build input rather than
documentation, and is the authority if this text and it disagree.

---

## 1. What zmix is

zmix is a **Zig port of `cmix`** (Byron Knoll) carrying the **`fxcm_v26`** text model (Kaido Orav) —
the lineage the Hutter Prize record has been set in. The port is not a translation exercise: it exists
so that the whole compressor is one statically-analysable program with a single build system, which is
what makes the packaging and determinism work of §7 possible.

The compressor is a **context-mixing** design. Many models each predict the next bit; a learned mixer
combines them; an arithmetic coder emits the result. Decompression re-runs the identical model set on
the data recovered so far, so no model state is transmitted.

What the shipped ensemble contains, in one sentence: the `fxcm_v26` context-model bank, a PPMd
order-20 byte model with a disk-backed arena, bracket / direct / match / indirect context models, and —
in the slot the lineage gives to an online-trained LSTM byte-mixer — a **frozen pre-trained
transformer**, all combined by a two-layer logistic mixer and an SSE stage into a binary arithmetic
coder. Ahead of the model set sits the inherited preprocessing chain: article reordering, `phda9`, and
the WRT dictionary transform.

---

## 2. The central idea of this generation: a frozen pre-trained transformer as a mixer input

The largest single change is **deleting** the online-trained LSTM byte-mixer and putting a **frozen,
pre-trained transformer** in the slot it occupied. The LSTM is not disabled at run time; it is compiled
out, and none of its state or code is in the shipped binary.

The architecture, the AVX2 CPU inference kernels, the training code, the quantizer and the
weight-container format come, with attribution, from the GPL-3 project **`fx2-cmix-transformer`**
(Vladimir Ivanov). See `NOTICE.md` and `THIRD_PARTY_LICENSES.md`; the weights themselves are a
retrained derivative, which is §3.

**The shipped model.** 12 layers, **5,923,228 parameters**, over the 205-value byte alphabet of the
post-transform stream. Weights are quantized to **4 bits per value with per-row bfloat16 scales**; the
remaining trained tensors are carried as bfloat16 and widened at load. It runs on one CPU core, from a
**2,815,630-byte** blob carried inside the artifacts themselves — nothing is downloaded and nothing is
read from outside the program.

**Why a frozen model is not obviously a good idea, and why it wins anyway.** A frozen model cannot
adapt to the file being compressed, which is the entire advantage of an online model. It wins because
its predictions are *much* better on ordinary English than anything an online model reaches within
1 GB, and because the mixer can learn how far to trust it, adaptively, per context.

**What it costs.** The weights must be shipped, and under the prize's self-extracting form they are
**charged twice** — once in `comp9`, once inside `archive9.exe`. At 2,815,630 B per copy that is
**5,631,260 bytes of the score**, about **89 %** of everything in the score that does not depend on
`enwik9`. That fee is what makes model size a *design* variable rather than a free choice, and it is
the reason §3 and §4 exist.

---

## 3. The weights are retrained, and their own coded size is part of the training objective

The blob this entry ships is **not** the blob `fx2-cmix-transformer` published. It is a retrained
derivative: the same 12-layer architecture, the same upstream GPL-3 training code and recipes, the same
post-transform token stream, and the same quantizer and container writer — with one change to the loss,

```
L_train = CE + multiplier · λ · soft_bits_per_weight
```

where the added term is a differentiable estimate of **the weights' own compressed size**.

The point is specific to this contest rather than to language modelling. Because the weight blob is
charged twice in the score, a model that is *equally good and cheaper to describe* is strictly better,
and training is the only place that can be bought. Measured one-variable against the published weights
on the identical engine: **−10,318 B** of coded output at the 20 MB decision tier, and a container
**24,787 B smaller per copy** ⇒ **−49,574 B** of the fixed part of the score. Every digest in the
derivation chain, and how to regenerate the blob, is in `assets/PROVENANCE.md`.

---

## 4. A measured property that shaped the rest: **the model-size axis is nearly flat**

Measured, not assumed: **loss falls with model size and the doubled weight fee rises with it, and
across the feasible range they very nearly cancel** — a depth-8 model and the 12-layer original price
out **byte-equal to within ±50 KB**, by two independent routes.

⇒ **Changing model depth moves the wall-clock budget, not the compressed size.** A shallower model was
priced at length for that reason, and a depth-8 line — with retraining from scratch to go with it — was
prepared and explored. **The entry ships the 12 layers.** Once the shipped configuration measured
inside the time limit on a judging-class machine, a shallower model had nothing left to buy: it would
have spent a retraining run to move a quantity that cancels.

⇒ It also means **bytes can only come from a better model at the same fee**, which is exactly the lever
§3 takes, and it is why width and weight-sharing tricks were measured and rejected (§6): they trade
quality for fee along the axis that already cancels.

---

## 5. Buying wall-clock time without spending bytes

The time limit and the score pull against each other — the entry is closer to the time limit than to
any RAM or disk limit — so levers that return time at little or no cost in bytes are worth more than
their size suggests.

- **Deleting context-model slots the transformer has made redundant — available, and deliberately
  *not* taken in this configuration.** With a much stronger model in the ensemble, some inherited
  context-model slots no longer earn their cycles, and the engine can delete a chosen subset of them
  **at compile time** (`-Dfxcm-slotmask` / `-Dfxcm-slotdelete`), so their code, their tables and the
  mixer inputs they fed are all gone rather than skipped. **This configuration ships unmasked**: it
  keeps every slot, pays the wall-clock time and takes the bytes. Sibling configurations of the same
  engine take the deletion and trade in the other direction; which one is submitted is a wall-clock
  decision, not a modelling one.
- **An int8 attention cache.** Holding the transformer's attention keys and values as int8 rather than
  fp32 removes an fp32 mirror of the cache from the memory traffic, which is where those layers are
  bound. Measured at **zero** cost in bytes: the archives are byte-identical with it on and off.
- **Cold code compiled for size.** The weight-loading path runs once and is charged twice, so it is
  built for size rather than speed (`-Dtf-load-opt=oz`), as are the packed int4 weight arenas the
  kernels read.
- **The submitted program has no command line.** The scored `comp9`/`archive9.exe` prefix is a
  dual-role program that compresses or self-extracts on **no arguments**, selecting on its own 12-byte
  trailer, and uses a plain allocator. Removing the argument parsing and the custom allocator is
  **−18,096 bytes of the score**, measured on the packed artifact, because that program is charged
  twice. `comp9.args` is a zero-byte file, so the rules' command-line term is 0.
- **Tuning the PPMd arena's eviction cadence** trades peak RSS — of which there is headroom — for
  wall-clock time, of which there is not.

---

## 6. Ideas that were measured and rejected

Recording these is more useful than recording successes, because each is a plausible idea with a
measured reason for dying:

- **Cross-layer weight sharing / low-rank factorization.** It dies on an exchange rate rather than on a
  similarity argument: composing the measured weight fee per parameter, the measured loss slope in
  parameters, and the measured bytes-per-nat of the coder pins the whole family at a ratio of ≈ 1.0,
  degrading monotonically as the sharing factor rises. Sharing also *bets against* longer training,
  which is a lever that does work.
- **Sub-4-bit weights.** The 4-bit width is already a size-versus-loss optimum chosen against this same
  objective, on this architecture and corpus, by the model's original authors — not an inference-speed
  artifact. Narrower loses more quality than the container saves.
- **Entropy-coding the quantized weights.** The container already entropy-codes the level stream with
  an adaptive 15-symbol bit tree and runs within about **1 %** of the order-0 entropy of its own
  alphabet, so the remaining headroom is tens of kilobytes, not megabytes. What did pay was a *lossless
  re-container* of the same values (§3's derivation), not a better model of them.
- **A long-range / retrieval channel.** The transformer's attention window leaves **43.7 % of the
  file's coded cost** beyond its reach — a real structural blind spot — yet exploiting it is worth only
  tens of kilobytes, because the compressor is already nearly free exactly where an exact long-range
  match would be right.
- **A next-word language-model channel.** The existing ensemble's top-1 next-token accuracy already
  exceeds published GPT-2 figures on comparable text; knowing the next word does not pay when the cost
  is concentrated where it is *not* knowable.
- **A better mixer.** Bounded by online-learning regret at **~1 byte over the whole 10⁹-byte file.**

---

## 7. Engineering ideas that matter for reproducibility

The prize is judged by rebuilding from source, so determinism is a first-class design goal:

- **Reciprocal-estimate instructions are removed.** `RCPPS`/`RSQRTPS`-class instructions return
  vendor-specific bits, so an archive built on one CPU vendor can fail to decode on another — this has
  already broken one real submission. The engine uses true divides, and a build-time gate disassembles
  the shipped binary and fails on any of the forbidden opcodes. The same gate checks that no libm math
  symbol is imported, so the C library's transcendental functions cannot vary underneath the model.
- **Every build input is pinned by hash, including the compiler and the packer.** Zig 0.15.1 and UPX
  5.2.0 are pinned exactly, not as minimums, and the build asserts both and aborts on a mismatch: the
  program is charged twice, so either tool's version moves the score. The effect is measurable — the
  same recipe under a different Zig patch release changes the packed program.
- **The rebuild is checked in the judge's own container, not only on ours.** The entry has been rebuilt
  inside a clean offline container matching the judging harness and reproduces `comp9`
  **byte-identically**, twice.
- **The source package is a deterministic tar stream** (sorted names, fixed mtime, numeric owner 0,
  `gzip -n`), so repeated packaging of the same tree yields the same bytes.
- **The dictionary is derived, not shipped twice.** A compact recipe regenerates it and verifies the
  result against a golden hash before use.
- **Assertions where correctness rests on a coincidence.** The 205-value token alphabet our front end
  produces has to agree, rank for rank, with the one the frozen model was trained on; nothing in either
  program enforces it, and a mismatch is invisible to a lossless round-trip because both directions
  would share the same wrong map. The entry therefore asserts the map **at compile time** — both anchor
  ranks and the full 15-byte article-separator token, so the binary cannot be built with a wrong map —
  and re-checks the separator count against the stream at the end of every compression.
- **Nothing outside the binary.** No environment variables, no absolute paths, no sysfs, no network.
  Temporary files are created in the working directory and named with the process id.

---

## 8. Honest limitations

- The transformer is **frozen**: it does not adapt to the file, and a substantial part of the file's
  cost lies outside the span it can see.
- The entry is **wall-clock-bound before it is compression-bound** — the configuration that would
  compress best does not fit the time limit, and what ships is chosen at that boundary.
- The weight fee is irreducible with current methods. The only attack on it that paid was making the
  weights cheaper to describe *while training them* (§3); every attack on the encoding of a fixed set
  of weights was measured and failed.
