# GLSL function costs on VC4 (VideoCore IV, V3D 2.1)

Read from Mesa `mesa-26.2.2` — the version the Pi Zero runs — and checked
against `engine/probes/packunpack_probe.cpp`.  Every number here is either a
citation of the source or a measurement; what is neither is marked as unknown.

The short version: on VC4 almost everything is a handful of QPU instructions
with a 1–2 cycle latency, **except** what Mesa lowers out of the hardware
(divide, mod, pow, sqrt, sin, int32 multiply) and **except six functions the
backend does not implement at all and aborts on**.

## 1. Which compiler runs

VC4 does **not** use the shared Broadcom compiler that `nir_to_vir.c` implements
for V3D 3.3+ (Pi 4/5).  It has its own NIR→QIR emitter:

| stage | file |
|---|---|
| NIR → QIR | `src/gallium/drivers/vc4/vc4_program.c` (`emit_alu`, `ntq_*`) |
| QIR → QPU machine code | `src/gallium/drivers/vc4/vc4_qpu_emit.c` |
| scheduling, add/mul pairing, stalls | `src/gallium/drivers/vc4/vc4_qpu_schedule.c` |
| instruction encoding | `src/broadcom/qpu/qpu_instr.c` |

`vc4_program.c:emit_alu()` ends with

```c
default:
        mesa_loge("unknown NIR ALU inst: %s", nir_instr_as_str(&instr->instr, NULL));
        abort();
```

so any NIR op the driver does not handle **kills the process at draw time**, not
at compile time.  That is what `packUnorm2x16` does on this board (measured).

## 2. Latency model (`vc4_qpu_schedule.c:711 waddr_latency`)

| write destination | cycles |
|---|---:|
| register file (`waddr < 32`) | 2 |
| SFU: `RECIP`, `RECIPSQRT`, `EXP`, `LOG` | 3 |
| TMU0/TMU1 setup → load | 100 |
| anything else | 1 |

The scheduler inserts the stalls; a kernel that chains dependent ALU results
pays 2 cycles per link, and a texture fetch that is consumed immediately pays
the 100.

## 3. What Mesa lowers before the backend sees it

`vc4_program.c:2135 nir_options` asks NIR to rewrite these, so they never reach
`emit_alu`:

| GLSL construct | lowered to | result |
|---|---|---|
| `a / b` | `frcp` + `fmul` | 2 instructions |
| `a % b` (`fmod`) | divide + floor + multiply + subtract | ~7 |
| `pow(a,b)` | `exp2(log2(a)*b)` | 3 (2 SFU + 1 mul) |
| `sqrt(x)` | `frsq` + `frcp` | 2 SFU |
| `-x`, `-i` | `0 - x` | 1 |
| `min/max` on ints | compare + select | 2 |
| `clamp(x)` (fsat) | saturate modifier | 0 — folded into the consumer |
| byte/word extract, insert | shifts and masks | 1–2 |
| 64-bit integers | split to 32-bit pairs | ×2 or worse |

## 4. Cost per operation (source-derived, one position, one component)

"QIR" is the instruction count `emit_alu` emits; QPU instructions are the same
unless the scheduler pairs an add with a mul (it does: one instruction issues
one add-slot and one mul-slot operation).

| GLSL | NIR | QIR | notes |
|---|---|---:|---|
| `a+b`, `a-b` | `fadd`/`fsub` | 1 | add slot |
| `a*b` | `fmul` | 1 | mul slot — **shares an instruction with an add** |
| `a*b+c` | `fmul`+`fadd` | 2 → **1** | the classic FMAD pairing |
| `min`, `max` | `fmin`/`fmax` | 1 | |
| `abs` | `fabs` | 1 | `qir_FMAXABS(x,x)` |
| `dot` (2/3/4) | mul + adds | 2–4 → **2** | pairs into 2 instructions for vec4 |
| `floor` | `ffloor` | ~5 | FTOI truncates, so floor needs the compare-and-fix |
| `fract` | `ffract` | **6** | FTOI, ITOF, FSUB, SF, conditional FADD, MOV |
| `trunc` | `ftrunc` | 2 | FTOI then ITOF |
| `sign` | `fsign` | **5** | SF + three conditional MOVs + MOV |
| `sin`, `cos` | `fsin`/`fcos` | **~12** | a 5-term polynomial on 2πx, not an SFU op |
| `1/x` | `frcp` | 1 | SFU, 3-cycle latency |
| `inversesqrt` | `frsq` | 1 | SFU |
| `exp2`, `log2` | `fexp2`/`flog2` | 1 | SFU |
| `float(int)` | `i2f32` | 1 | ITOF |
| `int(float)` | `f2i32` | 1 | FTOI, truncates toward zero |
| `int * int` (32-bit) | `imul` | **7** | `ntq_umul`: 3× MUL24 + 2× SHR + SHL + ADD |
| `int + - * & \| ^ << >>` | 1 each | 1 | the QPU has integer add/shift/logic |
| `bitfieldExtract` 8/16-bit | `ibfe`/`ubfe` | 1 | `qir_UNPACK_8_I/16_I` — **asserts** the width is exactly 8 or 16 and the offset is a uniform |
| `texture2D` | `tex` | 1 + setup | 100-cycle class latency, ~1 sample per quad per cycle |
| `packUnorm*/packSnorm*/packHalf*` and their unpacks | — | — | **aborts** (see §5) |
| `roundEven` | `fround_even` | — | **aborts** |

The `bitfieldExtract` row is the dangerous one: with a non-uniform offset or a
width that is not 8/16, the function's two `assert()`s are the only guard, and
in a release build they compile out — the operation then silently returns a
value assembled from the wrong bytes.  Measured on the Pi: extracting bits
24..32 of `0x3ebf0000` returned `0x3ebf`, i.e. bits 16..32.

## 5. What does not exist: measured, with the exact failure

`engine/probes/packunpack_probe.cpp` renders each function into a 1×1 FBO with a
**dynamic** operand and reads the result back.  Pi Zero, Mesa 26.2.2, ESSL 3.1:

| case | 300 es | 310 es |
|---|---|---|
| shifts / and (what `Q4_SHIFT` uses) | ok | ok |
| `floatBitsToInt`, `intBitsToFloat`, `uintBitsToFloat` | ok | ok |
| `bitfieldExtract` | compile error: no such function | ok, **but wrong for non-8/16-bit** |
| `packUnorm2x16` | **abort** | **abort** |
| `unpackUnorm2x16` | abort | abort |
| `packSnorm2x16` | abort | abort |
| `unpackSnorm2x16` | abort | abort |
| `packHalf2x16` | abort | abort |
| `unpackHalf2x16` | abort | abort |
| `roundEven` | abort | abort |
| `floor` (control) | ok | ok |

The abort message names the NIR op:

```
MESA: error: unknown NIR ALU inst: 32    %2 = pack_unorm_2x16 %1.xz
MESA: error: unknown NIR ALU inst: 32    %4 = fround_even %3
```

**Run this probe with dynamic inputs.**  The same battery with literal operands
reports every one of these as "ok", because the driver's constant folding
evaluates them on the CPU before the backend ever sees them.  That is how an
earlier version of `docs/engine-internals.md` came to claim half precision was
reachable through `packHalf2x16`; it is not.

**A sampler passed as a function parameter is not a reliable reader.**  A
one-fetch decode helper (`float x_i16_of2(sampler2D T, float u, int odd)`)
returned the right value on a one-row probe -- and in the engine, on the same
board, the same build produced three different replies in ten identical runs
(2026-11-25; the previous text, which writes the fetch inside each ternary
arm, was 10/10 stable in the same session).  The engine's decoders are macros
partly for this reason: a sampler is program uniform state, and a callee that
receives it is not dependable on this driver.  `probes/codec_probe` prices the
win the helper would buy (13.3 -> 7.5 ns a value); it has to be had by
hoisting the fetch at the call site instead.

## 6. What this means for this engine's kernels

- The inner loops are `mul + add` chains: the pairing is what makes a matvec
  fragment cost one instruction per multiply-add per texel.  Anything that
  breaks the pairing (a conditional, a `floor`, a `sign`) doubles it.
- `floor`/`fract`/`sign` are not free: a q4 or q8 decode that uses them pays
  5–6 instructions where a saturation modifier would have cost 0.  That is the
  measured difference between the q4 decode variants in `docs/benchmarks.md`.
- Never put `sin`/`cos`/`pow`/`mod` in a per-fragment path.
- `int * int` is 7 instructions: the q8 scale arithmetic is written in floats on
  purpose.
- **Bit-cast a sign into a float instead of computing it.**  The t3 weight
  decode built `mag*(1-2*sign)` (a float mul + sub on the sign); bit-casting
  `uintBitsToFloat(0x3F800000u|(sign<<31))*float(mag)` replaces those two float
  ops with integer shift/or (add slot) that pair with the final mul (mul slot).
  Measured -13% on the isolated decode (`probes/t3dec_probe`, 3.01 vs 3.46
  ns/MAC), byte-identical.  `uintBitsToFloat` is one of the bit-casts that DOES
  exist (§5); a vec4 LUT indexed by the 2-bit code is +22% (dynamic component
  index spills), so the algebraic bit-cast beats both the float-sign form and
  the table.
- **The float domain beats the integer domain for ternary, even "multiply-free".**
  The SASORI-style integer dot (activation kept as raw int8, weight decoded to a
  sign/mag mask, `ivec4` accumulate, one ITOF at the end) measured **+13%
  single-row and +27% pair** (`t3dec_probe` int_acc).  Two reasons: float is
  sign-magnitude, so negation is a FREE sign-bit flip, while two's-complement
  integer negation costs `(a^mask)-mask` (two ops); and the per-component integer
  mask loses the vec4 dot's FMAD pairing.  Do not "optimize" the t3 decode into
  the integer domain.  Two more dead micro-opts measured there: omitting the
  sign's `&1` before `<<31` is free but the compiler already does it (no gain),
  and zeroing by masking the float bits (`& (0-mag)`) is +4% vs ITOF+fmul.
- **"Compute without decoding" (T-MAC's LUT) is 2.4x SLOWER on VC4.**  A
  per-group activation LUT in a texture, indexed by the weight code, replaces
  the decode with a table fetch -- but the packed t3 format's advantage is
  1 weight fetch per 16 weights, and the LUT adds 4 random-access fetches per
  16 weights (5x traffic).  `probes/t3lut_probe`: 5.63 vs 2.33 ns/MAC at
  1024x1024.  The TMU cannot sustain the random access; on Xe (fast texture,
  big cache) the LUT wins, which is why T-MAC is CPU-only.  The decode is the
  irreducible cost of reading a packed format on hardware with no unpack/dot
  instruction: extracting the weight information costs either ALU (decode,
  cheap) or fetches (LUT, expensive here).
- **The unused t3 code is a don't-care, but exploiting it does not beat signbit.**
  The mag/sign encoding leaves code 2 (mag0 sign1, "negative zero") unused; it
  can be assigned any value.  Setting it to -2 makes the weight `bit0 - 2*bit1`
  (integer subtract + one ITOF, no bit-cast, no float multiply -- frees the mul
  slot for the dot's FMAD).  Measured (`t3dec_probe` subtract/sub_vec): slower
  single-row (3.13-3.57 vs signbit 3.01) and tied on the pair kernel (2.02 vs
  2.03, within noise).  The ITOF + int-subtract costs what the bit-cast + fmul
  costs; freeing the mul slot buys nothing.  The other theoretical use of the
  wasted code space -- radix-3 packing, 5 trits/byte (3^5=243<256), 20% denser --
  is counterproductive here: it needs div/mod by 3 (expensive) and the matvec is
  decode-bound, not fetch-bound, so denser packing saves the cheap part and adds
  to the expensive part.  The unused code is the price of the shift/mask-friendly
  2-bit layout, and that price is worth paying on VC4.
- **More rows a fragment does not make a q8 matvec faster; independent fetches do.**
  Measured 2026-11-21 (`probes/qkv_rows_probe`).  The engine's qkv kernel pairs
  two output rows a fragment and its FFN gate four, and the gate runs at 1.79
  ns/MAC against qkv's 2.87 -- which looks like qkv leaving a 2x on the table.
  It is not: at the real shape (1280x768) a 4-row qkv kernel costs **+15% per
  stored row** (2.700 against 2.351 ms), and a 4-row kernel with two activation
  pairs an iteration is inside noise (2.307, -2%).  The FFN wins because its
  four rows are two *independent* matrices (w1 and w3), so one iteration carries
  four independent weight fetches and the scheduler has work to overlap; qkv's
  four rows are four rows of one matrix sharing one activation pair, and the
  decode they share was never the bottleneck.  `ACT=i8` had said the same thing
  from the other side: halving the activation fetches (quads 2.80M -> 2.33M)
  left the FFN's `gpu=` and `exec=` unchanged.
  One trap in reading the probe: its 4-row arm runs the whole column loop in
  every fragment and discards half the sums, so per *MAC executed* it reads 1.37
  ns/MAC against qkv's 2.39 -- a 43% "win" that is the arm doing twice the work
  for the same output.  Price a kernel per output row, never per MAC the grid
  happened to execute.

- **The GL pipeline has one colour attachment.  The hardware does not.**  On
  the GL path `GL_MAX_DRAW_BUFFERS` and `GL_MAX_COLOR_ATTACHMENTS` both read
  **1**; a two-attachment FBO reports COMPLETE but `glDrawBuffers(2, ...)`
  returns `GL_INVALID_OPERATION`, and `gl_FragData[1]` fails to compile with
  "array index must be < 1".  `GL_EXT_draw_buffers` *is* in the extension string
  and both entry points resolve, so the query is the only way to know.
  That is the **render pipeline's** shape, not the QPU's: the architecture guide
  describes a single tile colour buffer ("only the tile color buffer is written
  out"), and a QPU program that bypasses the pipeline -- [V3DLib](https://github.com/axizo-pi/V3DLib),
  which writes through VPM and DMA/TMU to any main-memory address -- has no
  attachment count at all.  So "no MRT" is an interface limit, and the honest
  statement is that **a fragment shader can store one RGBA8 texel, i.e. two q16
  values**.
  **Bypassing GL does not pay.**  V3DLib's own benchmark
  (`docs/profiling.ods` in that repository; Rot3D, 1,024,000 vertices, ~18 flop
  each) reaches **0.268 GFLOP/s** with 4 QPUs on a Pi3 -- **0.9%** of the
  28.8 GFLOP/s peak -- against this engine's GL path at **0.754 GFLOP/s (2.6%)**
  on the board.  A hand-written QPU kernel has to manage VPM transfer, register
  allocation and instruction scheduling itself, and those costs more than eat
  the freedom.  The same table also shows TMU write-back beating DMA by 1.7x
  (0.069 against 0.082 s), which is the path this engine already uses.
  Consequence measured the same day (`probes/qkv_rows_probe`): every shape that
  works around it loses.  Four rows a fragment either has the grid run both
  halves (**+15%**, twice the MACs for the same stored rows) or deepens the
  column loop (**-2%**, inside noise, and texture quads +50%).  So a q8 kernel
  amortises its activation decode over at most the two rows one texel can hold,
  and the FFN gate's 1.79 ns/MAC is not a technique to copy: its four fetches
  share one decode for 16 MACs where qkv's two share it for 8, which is what
  gating two matrices together buys and a plain projection cannot.

- Anything that needs half precision must be assembled by hand from float
  operations; there is no `packHalf2x16`, and there are no half-float textures
  either (`OES_texture_half_float` is not exposed).
- **A loop the compiler does not unroll loses conditional index updates.**
  Measured 2026-11-15 (`ARGMAX_DUMP`, `docs/benchmarks.md`): in a scan that
  keeps a running (max, index) pair, the float max stays correct but the
  index sticks at its first-iteration value — with the index carried as int
  or float, in the single-branch or the two-branch form.  A 96-iteration loop
  showed it; a 16-iteration loop with a runtime-bound `break` showed it too;
  the 16-iteration literal-bound top2 loop (which Mesa unrolls) is exact.
  Reduction kernels therefore keep every loop at the unrollable length and
  avoid runtime breaks: out-of-range elements are sentinel texels, not a
  `break`.  Xe compiles all of these forms correctly, so only a board run
  catches this.
- **Flattening a nested group loop is shape-specific, not a free win.**
  Measured 2026-11-16 (`docs/benchmarks.md`): rewriting the q4 FFN *gate*
  from a nested group loop to a flat 8-wide loop (per-group state via
  conditional float updates) was byte-identical and **-43%** — the gate is a
  *pair* kernel, eight weight fetches and four accumulators per inner
  iteration, enough independent work that the flat loop pipelines and the
  rolled nested loop schedules badly.  The same rewrite on the *single-row*
  qkv and w2/wo kernels (two fetches, one accumulator per inner iteration)
  was byte-identical but **~12.5 ms/token slower**: there the nested loop
  already schedules fine and the flat form's per-group conditionals, resets
  and flushes are pure overhead.  Before flattening another kernel, check
  how much independent work its inner body carries; the gate's win came from
  its wide body, not from "flat beats nested".

## 7. Measuring against the hardware counters

Sections 2 and 4 are source-derived; this is the other direction.  The board
exposes `GL_AMD_performance_monitor` (one group, "V3D counters", 30 counters,
16 selectable at once, `eglinfo -s` lists it) and no timer query, so the
counters are the only way to attribute a draw's wall time to the QPU.
`probes/perfcnt.h` wraps them and `PERFCNT=1` enables them per probe; the
numbers are in [`benchmarks.md`](benchmarks.md), "What a draw's time is made
of".

What the counters say about the latency model:

- **`waiting-TMU` is small on loading kernels and large on some fast ones.**
  A 256-tap q8 matvec sits at 20% waiting-TMU with 0.8% text-cache misses,
  i.e. the stall is TMU *issue*, not DRAM -- which is the hardware reason the
  `t3lut_probe` LUT loses 2.4x (it turns one fetch per 16 weights into five
  random ones) while denser packing wins.
- **`executing-valid-instr` tracks fragment cycles closely** on ALU-bound
  kernels (77% of fragment cycles in the 256-tap test), so the cost model's
  instruction count is the right first-order term; the interest is in what the
  remaining 20% is.
- **A cheap draw does not spend its time on the QPU at all.**  The per-draw
  overhead a probe measures at 0.4-0.5 ms is 0.004-0.03 ms of fragment shading,
  so an optimization that removes a draw is worth the submit, not the shading --
  and one that removes fragment work is worth the counters' 94-100% and no more.

Treat a counter table as evidence about *this* driver: the record layout
(four words a counter), the QPU divisor (12 here) and the clock (300 MHz) are
all measured facts that a different board would have to re-measure.
