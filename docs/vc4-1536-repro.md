# Reproducing the 1536 and vocabulary-grid questions on VC4

Two claims are open on the board and cannot be settled on the laptop:

1. **Is a single 1536-fragment draw wrong on VC4?**  `docs/engine-internals.md`
   §2 and the "do not retry" table in `docs/future-optimization.md` say yes
   (measured 2026-11-03: garbage tokens).  The width sweep in `30b2882`
   (2026-11-04) measured 768..2040 as byte-identical and only **2048** as wrong.
   The two have never been reconciled; the code keeps both workarounds.
2. **What does changing `kLogitsGrid` cost on VC4?**  On Xe a repack at 1024 gave
   the same greedy text and PPL 270.3170 -> 270.5287 (+0.08%), two more logits
   draws per token.  Nothing about that is VC4-specific except the strip
   behaviour and the 2048 texture ceiling.

Everything below is a same-binary A/B through environment switches, per the
measurement rules in `AGENTS.md`.  Nothing needs a rebuild except the deploy.

## Boxes and paths

| box | path | notes |
|---|---|---|
| laptop | this checkout | packing, transfers, comparison |
| `vlab` | `~/vc4-egl`, `~/raspi-root4` | cross build; `arm-linux-gnueabihf-g++-12`, 2 cores |
| `raspi` | `/home/david/vc4-llm-runtime` | no git checkout; `bin/`, `models/minimind3/`, `engine/data/ppl_test.txt` |

State at the time of writing (2026-11-08): Pi has 3.8 G free, `CmaFree` 81 MB of
256 MB, glibc 2.41, `engine_mm3_pi` from 11-08 11:20 (older than the current
tree, so it has no `--dump-shaders`), `minimind3.q8.mmcf` and `.q4.mmcf` both
packed at stride 1536.  `vlab`'s checkout is behind; its
`receive.denyCurrentBranch` is unset, so **pull** there, do not push to it.

## 0. Move the source

`vlab` builds from its own checkout, so it has to see the commits:

```bash
# laptop (origin is git.ustc.edu.cn; the vlab remote is a non-bare checkout)
git push origin "$(git branch --show-current)"

# vlab
cd ~/vc4-egl && git pull --ff-only
git log --oneline -1            # must match the laptop
```

## 1. Cross-build and deploy

```bash
# vlab
cd ~/vc4-egl
make -C engine -B pi CROSS_SYSROOT=$HOME/raspi-root4
make -C engine -B probes/draw_probe_pi CROSS_SYSROOT=$HOME/raspi-root4
readelf -A engine/engine_mm3_pi | grep -E 'Tag_CPU_arch|Tag_FP_arch'   # v6 / VFPv2
readelf -A engine/probes/draw_probe_pi | grep -E 'Tag_CPU_arch|Tag_FP_arch'
ls -l engine/engine_mm3_pi engine/probes/draw_probe_pi
```

`vlab` and the Pi cannot see each other; the laptop is the relay:

```bash
# laptop: two hops, vlab -> laptop -> raspi
ssh_download vlab /home/ubuntu/vc4-egl/engine/engine_mm3_pi        ./engine_mm3_pi.new
ssh_download vlab /home/ubuntu/vc4-egl/engine/probes/draw_probe_pi ./draw_probe_pi.new
ssh_upload   raspi ./engine_mm3_pi.new  /home/david/vc4-llm-runtime/bin/engine_mm3_pi.new
ssh_upload   raspi ./draw_probe_pi.new  /home/david/vc4-llm-runtime/bin/draw_probe_pi.new
```

(Those are the DSH tool names; plain `scp -3` via the laptop works the same.)

```bash
# raspi
cd /home/david/vc4-llm-runtime/bin
cp -p engine_mm3_pi engine_mm3_pi.bak        # keep the old one
mv engine_mm3_pi.new engine_mm3_pi
mv draw_probe_pi.new draw_probe_pi
chmod +x engine_mm3_pi draw_probe_pi
./engine_mm3_pi --version
./engine_mm3_pi --help | grep dump-shaders   # must print the line
```

## 2. GLSL identity: the Pi build must emit the laptop's text

No model, no GPU.  This proves the cross build did not change any shader.

```bash
# raspi
./engine_mm3_pi --dump-shaders > /tmp/pi.hash 2>/tmp/pi.err
wc -l /tmp/pi.hash                                # 117

# transfer /tmp/pi.hash to the laptop, then:
grep -v '^#' engine/scripts/shader_hashes.txt > /tmp/golden
diff -u /tmp/golden pi.hash && echo "GLSL identical"
```

A difference here invalidates everything below: it means the cross build emits
different text (compiler/float formatting), not that VC4 behaves differently.

## 3. The draw boundary, without a model

`probes/draw_probe` draws a q8 matvec at 768/1216/1536/2048 rows and compares
each against two half-width draws of the same weights: a row's arithmetic does
not depend on the row count, so the bytes must match exactly.  `MATVEC_STRIP`
overrides `draw()`'s 1024-fragment cap, which is what makes a *single* wide draw
possible.

```bash
# raspi
cd /home/david/vc4-llm-runtime/bin
./draw_probe_pi               2>&1 | grep '^MV'    # cap 1024: every shape is split
MATVEC_STRIP=1536 ./draw_probe_pi 2>&1 | grep '^MV'   # 1536 drawn in ONE pass
MATVEC_STRIP=2048 ./draw_probe_pi 2>&1 | grep '^MV'   # 2048 drawn in ONE pass
```

Read the `halves identical (diff N/M)` field, and the MB/s:

| observation | conclusion |
|---|---|
| `MATVEC_STRIP=1536` -> `1536x768 ... identical`, MB/s ~270 | the command is fine, but this probe **cannot** clear 1536: it compares after `glReadPixels`, and the engine's failure only appears when the next pass samples the target.  Run step 4 before concluding anything |
| `MATVEC_STRIP=1536` -> `DIFFER` | 1536 is bad even in readback; the 11-03 claim is confirmed twice over |
| `MATVEC_STRIP=2048` -> `DIFFER` with ~5000/8192 bytes and a much higher MB/s | reproduces `30b2882`; this is the expected control |
| `MATVEC_STRIP=2048` -> identical | the 2048 finding does not reproduce on this driver/build; stop and record it |

The probe also prints `NULL`/`RMS`/`CPU` timings; they are not the point here.

## 4. The model-level test (q8, no repack)

`QKV_PAIR=0` forces the unpaired qkv matvec, i.e. a real 1536-row draw through
`draw()` (the paired path is 768 fragments wide and never hits the cap).  With
the default cap that is two strips (1024+512); with `MATVEC_STRIP=1536` it is
one.

```bash
# raspi
cd /home/david/vc4-llm-runtime
export MODEL=models/minimind3/minimind3.q8.mmcf GPU_LAYER=1 KV_PACK=16 SEQ=768
SYS="You are a helpful assistant."
P="用一句话介绍你自己。"

# baseline: paired qkv, default cap (the acceptance pin)
./bin/engine_mm3_pi --chat --system "$SYS" --prompt "$P" --gen 32 --greedy

# A: unpaired qkv, default cap -> two strips
QKV_PAIR=0 ./bin/engine_mm3_pi --chat --system "$SYS" --prompt "$P" --gen 32 --greedy
QKV_PAIR=0 ./bin/engine_mm3_pi --chat --system "$SYS" --prompt "$P" --gen 32 --greedy  # twice: determinism

# B: unpaired qkv, one 1536-wide draw
QKV_PAIR=0 MATVEC_STRIP=1536 ./bin/engine_mm3_pi --chat --system "$SYS" --prompt "$P" --gen 32 --greedy
```

Pinned Pi text for the baseline (`AGENTS.local.md`):
`Assistant: 我是一个AI助手，没有自我意识，也没有身体，所以我无法`.
A is the reference for B: if 1536 is fine they match the pin, and if the 11-03
claim holds B prints garbage (`编写一个编写一个...`) while A does not.  If A does
not match the pin either, the pairing switch itself is the confound - record it
and compare B against A, not against the pin.

PPL for the same three configurations:

```bash
./bin/engine_mm3_pi --ppl engine/data/ppl_test.txt                                  # expect 271.2855
QKV_PAIR=0 ./bin/engine_mm3_pi --ppl engine/data/ppl_test.txt
QKV_PAIR=0 MATVEC_STRIP=1536 ./bin/engine_mm3_pi --ppl engine/data/ppl_test.txt
```

Any configuration that produces garbage text will also move PPL by a lot; PPL on
its own is the sharper instrument for a one-token change.

## 5. Optional: what changing the grid costs (needs a repack)

Only worth doing if 1536 is going to be reconsidered.  The container stride and
the engine constant must move together - `Model::load` refuses a mismatch, by
design.

```bash
# laptop: pack at 2048 (4 tiles: 3 x 2048 + 256)
cp engine/tools/mmcf_grid.py /tmp/mmcf_grid.py.bak
sed -i 's/LOGITS_GRID = 1536/LOGITS_GRID = 2048/' engine/tools/mmcf_grid.py
python3 engine/tools/mmcf_pack.py --in models/minimind3 --quant q8 --seq 256 \
  --tok models/minimind3/minimind3.tok \
  --out models/minimind3/minimind3.q8.g2048.mmcf \
  --name minimind3 --template minimind3 --chat \
  --chat-template minimind3/chat_template.jinja
cp /tmp/mmcf_grid.py.bak engine/tools/mmcf_grid.py          # restore the tree
python3 engine/tools/mmcf_dump.py --tensors models/minimind3/minimind3.q8.g2048.mmcf | grep logits_tile
# expect 4 tiles, 2048/2048/2048/256 rows
```

Build a second Pi binary with the other constant (keep the 1536 one):

```bash
# laptop: a throwaway branch, because vlab builds from its checkout
git switch -c grid2048
sed -i 's/constexpr int kLogitsGrid = 1536;/constexpr int kLogitsGrid = 2048;/' engine/src/logits_grid.h
git commit -am "temp: grid 2048 for the Pi A/B"
git push origin grid2048

# vlab
cd ~/vc4-egl && git fetch origin && git switch grid2048
make -C engine -B pi CROSS_SYSROOT=$HOME/raspi-root4
cp engine/engine_mm3_pi engine/engine_mm3_pi_g2048
git switch -            # back to the main branch

# laptop: download engine_mm3_pi_g2048, upload it and the container (69 MB)
ssh_upload raspi models/minimind3/minimind3.q8.g2048.mmcf \
  /home/david/vc4-llm-runtime/models/minimind3/minimind3.q8.g2048.mmcf
ssh_download vlab /home/ubuntu/vc4-egl/engine/engine_mm3_pi_g2048 ./engine_mm3_pi_g2048
ssh_upload raspi ./engine_mm3_pi_g2048 /home/david/vc4-llm-runtime/bin/engine_mm3_pi_g2048

# afterwards
git switch - && git branch -D grid2048
```

```bash
# raspi
df -h /home/david | tail -1                     # need ~70 MB
chmod +x bin/engine_mm3_pi_g2048
MODEL=models/minimind3/minimind3.q8.g2048.mmcf ./bin/engine_mm3_pi_g2048 \
  --chat --system "$SYS" --prompt "$P" --gen 32 --greedy
```

Compare with the baseline: same text, PPL within ~0.2%, and one fewer logits
draw (`DRAWCOUNT=1`).  On Xe the text matched and PPL moved +0.08%.  Delete the
container and the extra binary afterwards.

## 6. Rules that decide whether a number means anything

- **One GPU client, under the board's lock.**  Stop the server and any
  benchmark; CMA is single-client.  Every run on the Pi takes
  `flock -w 300 /tmp/vc4-pi.lock <command>`: that file is the convention the
  tuning side uses, and nothing in the runtime takes it for you.  Check
  `grep CmaFree /proc/meminfo` before a run (88 MB free when this was
  measured).
- **Interleave A/B.**  Same binary, one env switch, alternate the runs; take the
  best of two and note the median.  `bench_ab.sh` compares two *binaries*, so
  for an env switch run the loop by hand.
- **Never compare across `SEQ`**, and never compare a Pi number with an Xe one.
- **A garbage string is the signal, not the exception.**  If B prints the pinned
  text, that is the result; do not "fix" the test to make it fail.
- **Check the load average** the Pi runs a Minecraft server sometimes; > 1.5
  makes timings noisy (correctness still stands).
- **Record the build**: `git rev-parse HEAD`, `readelf -A` output, and the
  binary's `--version` line in whatever note the result lands in.
- **The Pi has no repository.**  Transfer files, do not try to build there
  unless you accept an 8-10 minute native ARMv6 build with
  `-march=armv6 -mfpu=vfp -mfloat-abi=hard`.

## 7. What is already known, so it is not re-measured

| fact | where it came from |
|---|---|
| q8 logits are packed, 768 fragments per tile; 5 tiles at stride 1536 | `logits_pass`, `read_logits_full` |
| grid 1024: same greedy text, PPL 270.3170 -> 270.5287, 7 tiles | repack on Xe, 2026-11-08 |
| grid 1300: token ids off by 236, yet greedy and TOPK=8 output looked correct | repack on Xe, 2026-11-08 |
| a 2048-fragment draw is wrong; 2040 is exact | width sweep, `30b2882`, Pi Zero |
| a single 1536 draw is wrong | `future-optimization.md`, 2026-11-03; reproduced 2026-11-08, see below |

## Results, 2026-11-08 (Pi Zero, Mesa 26.2.2, commit `3c1da83`)

Cross build: `Tag_CPU_arch v6`, `Tag_FP_arch VFPv2`.  `--dump-shaders` on the Pi:
117 programs, 6609 bytes of digests, **byte-identical** to
`scripts/shader_hashes.txt` (`diff` clean) - the cross build emits the laptop's
GLSL.

`draw_probe_pi` (halves comparison, MB/s in the last column of the run):

| `MATVEC_STRIP` | 1536x768 | 2048x768 |
|---|---|---|
| default (1024) | split, identical | split, identical |
| 1536 | one draw, **identical** 279 MB/s | one draw, identical |
| 2048 | one draw, identical | one draw, **DIFFER 5030/8192**, 776 MB/s |

Engine, q8, `SEQ=768`, greedy:

| run | output |
|---|---|
| paired qkv, cap 1024 | `我是一个AI助手，没有自我意识，也没有身体，所以我无法…` (pin) |
| `QKV_PAIR=0`, cap 1024 | pin |
| `QKV_PAIR=0 MATVEC_STRIP=1535` | pin (split 1535+1) |
| `QKV_PAIR=0 MATVEC_STRIP=1536` | **garbage, different every run**: `..nnnnnnnn…`, `听今天今天…`, `佳译译译…`, `儿火火火…` |

`FORCE_GLFINISH=1` does not change the garbage.  PPL on the same corpus:

| configuration | PPL |
|---|---:|
| paired qkv, cap 1024 | **271.2855** (matches the pin) |
| `QKV_PAIR=0 MATVEC_STRIP=1536` | **71669.1013** |

Conclusion: a single 1536-wide draw is exact in a readback probe and corrupt in
the engine's pipeline, nondeterministically; 2048 is corrupt in readback too.
The 1024-fragment cap covers both.  The 2026-11-04 sweep's "768..2040 identical"
is a readback statement and must not be read as clearing the engine path.
