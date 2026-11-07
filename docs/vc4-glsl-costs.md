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

## 6. What this means for this engine's kernels

- The inner loops are `mul + add` chains: the pairing is what makes a matvec
  fragment cost one instruction per multiply-add per texel.  Anything that
  breaks the pairing (a conditional, a `floor`, a `sign`) doubles it.
- `floor`/`fract`/`sign` are not free: a q4 or q8 decode that uses them pays
  5–6 instructions where a saturation modifier would have cost 0.  That is the
  measured difference between the q4 decode variants in `docs/performance.md`.
- Never put `sin`/`cos`/`pow`/`mod` in a per-fragment path.
- `int * int` is 7 instructions: the q8 scale arithmetic is written in floats on
  purpose.
- Anything that needs half precision must be assembled by hand from float
  operations; there is no `packHalf2x16`, and there are no half-float textures
  either (`OES_texture_half_float` is not exposed).
