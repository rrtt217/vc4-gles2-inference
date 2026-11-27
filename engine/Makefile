CXX ?= g++
# -std is not in CXXFLAGS: `CXXFLAGS ?=` let a caller's `make CXXFLAGS='-g'`
# silently drop the standard (and the optimisation) with it.  Warnings are on by
# default, because a fresh clone that builds without them is how a dead lambda sat
# in main() unnoticed (2026-11-14).
CXXSTD := -std=c++17
OPT ?= -O2
WARN ?= -Wall -Wextra
# Compile parallelism for the forced clean build in warncheck: capped so a
# many-core host does not start more compilers than its memory can hold.
JOBS ?= $(shell n=$$(nproc 2>/dev/null || echo 2); [ $$n -gt 8 ] && n=8; echo $$n)
CXXFLAGS ?=
ALL_CXXFLAGS := $(CXXSTD) $(OPT) $(CXXFLAGS) $(WARN)
LDLIBS ?= -lEGL -lGLESv2 -lm

SRC := src
TP  := third_party
# Jinja chat templates (vendored from llama.cpp; see third_party/jinja/README.md).
# json.cpp is the only translation unit that includes nlohmann, so the compile
# cost stays in one place.
JINJA := $(TP)/jinja/lexer.cpp $(TP)/jinja/parser.cpp $(TP)/jinja/runtime.cpp \
         $(TP)/jinja/string.cpp $(TP)/jinja/value.cpp $(TP)/jinja/json.cpp \
         $(TP)/jinja/unicode.cpp
# The GPU side without main(): what a probe links against.  It has to be a
# wildcard because the refactor split gpu.cpp into gpu_*.cpp, and every probe
# target that still listed the old file failed to link (2026-11-15).  The same
# rule now covers the shader builders: one TU per weight format plus the shared
# prelude, so a new shader file is linked without every probe being edited.
SHADERLIB := $(wildcard $(SRC)/shaders*.cpp)
GPULIB := $(wildcard $(SRC)/gpu*.cpp) $(SHADERLIB) $(SRC)/glsl_io.cpp \
          $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp
CORE := $(SRC)/engine.cpp $(SRC)/common.cpp $(SRC)/xformat.cpp $(SRC)/switches.cpp \
        $(SHADERLIB) $(SRC)/dump_shaders.cpp $(SRC)/glsl_io.cpp \
        $(wildcard $(SRC)/gpu*.cpp) \
        $(SRC)/tokenizer.cpp $(SRC)/model.cpp $(SRC)/session.cpp $(SRC)/prefix_cache.cpp $(SRC)/lineedit.cpp $(SRC)/lineedit_layout.cpp \
        $(SRC)/sampling.cpp $(SRC)/grammar.cpp $(SRC)/gbnf.cpp $(SRC)/gbnf_full.cpp $(SRC)/schema.cpp $(SRC)/constrain.cpp $(SRC)/tools.cpp $(SRC)/ppl.cpp $(SRC)/cli.cpp $(SRC)/chat_format.cpp $(SRC)/prefix_build.cpp $(SRC)/app.cpp $(SRC)/worker.cpp $(SRC)/chat.cpp $(SRC)/chattmpl.cpp $(JINJA)
HEADERS := $(wildcard $(SRC)/*.h)
# Incremental build: one object per translation unit, dependencies from -MMD.
# Each toolchain gets its own tree -- a native, a cross and an armv6 object
# differ in ABI and flags, and reusing one is a wrong binary that links.  -B
# still means "rebuild everything": it is the baseline command in AGENTS.md and
# what warncheck runs.  make cannot see a CXXFLAGS or sysroot change, so after
# one, use -B or remove that object tree.
HOST_OBJ  := $(patsubst %.cpp,build/host/%.o,$(CORE))
PI_OBJ    := $(patsubst %.cpp,build/pi/%.o,$(CORE))
ARMV6_OBJ := $(patsubst %.cpp,build/armv6/%.o,$(CORE))

build/host/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(ALL_CXXFLAGS) $(SECTIONS) $(INCS) -MMD -MP -c $< -o $@

# The cross tree is only valid for the sysroot and compiler it was built with.
# The stamp is rewritten only when one of them changes, and every pi object
# depends on it, so a sysroot switch cannot reuse host-ABI objects.
PI_STAMP := build/pi/.toolchain
$(PI_STAMP): force-stamp
	@mkdir -p $(dir $@)
	@printf '%s\n%s\n' '$(CROSS_CXX)' '$(CROSS_SYSROOT)' | cmp -s - $@ || \
	  printf '%s\n%s\n' '$(CROSS_CXX)' '$(CROSS_SYSROOT)' > $@

force-stamp:
	@:

build/pi/%.o: %.cpp $(PI_STAMP) | check-cross
	@mkdir -p $(dir $@)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) $(WARN) $(SECTIONS) $(INCS) -MMD -MP -c $< -o $@

build/armv6/%.o: %.cpp | armv6-check
	@mkdir -p $(dir $@)
	$(CXX) $(ARMV6_CXXFLAGS) $(WARN) $(SECTIONS) $(INCS) -MMD -MP -c $< -o $@

-include $(HOST_OBJ:.o=.d) $(PI_OBJ:.o=.d) $(ARMV6_OBJ:.o=.d)
# Probe-only headers (not part of the engine build).  Only the probes that
# include one depend on it; the others are untouched.
PROBE_HEADERS := $(wildcard probes/*.h)
# third_party is -isystem: it is vendored code (llama.cpp's Jinja engine) and its
# own warnings are not this repository's to fix, but they must not drown ours --
# before this, a full build printed 11 warnings from jinja headers and 1 from src/.
INCS := -I$(SRC) -isystem $(TP)
# The Jinja engine is large and only one path uses it; let the linker drop what
# this build never reaches (-947 KB -> -1.19 MB stripped, measured on Xe).
SECTIONS := -ffunction-sections -fdata-sections
LDFLAGS ?= -Wl,--gc-sections

engine_mm3: $(HOST_OBJ)
	$(CXX) $(ALL_CXXFLAGS) $(SECTIONS) $(INCS) -o $@ $(HOST_OBJ) $(LDLIBS) $(LDFLAGS)


# ARMv6 cross build for the Pi Zero.  Two independent installs meet here: the
# host's compiler driver and the sysroot's runtime, and their major versions
# legitimately differ (a g++-12 driver against a gcc-14 runtime is what built the
# v6 binaries this repository ships).  Neither is hardcoded, for one measured
# reason: gcc SILENTLY IGNORES a -B directory that does not exist and then links
# the host's ARMv7 startfiles.  The link succeeds, the binary runs on the build
# host, and it faults on the ARM1176 with `Tag_CPU_arch: v7` - v6/VFPv2 with the
# right -B, v7/VFPv3-D16 with a wrong one, exit 0 and no diagnostic either way.
# So check-cross validates the discovery before the link, and the recipe
# validates the artifact after it.  docs/cross-build.md has the sysroot and
# toolchain recipe.
CROSS_SYSROOT ?=
# First match wins: the unversioned name, else the newest versioned one that is
# installed.  Lazy (`?=` plus $(shell)) so `make engine_mm3` on a host without a
# cross toolchain never runs this lookup.
CROSS_CXX ?= $(shell for v in "" -20 -19 -18 -17 -16 -15 -14 -13 -12 -11 -10; do \
                 c=arm-linux-gnueabihf-g++$$v; \
                 command -v $$c >/dev/null 2>&1 && { echo $$c; break; }; done)
# readelf reads any ELF target, so the host's copy is enough; the prefixed one is
# preferred when the cross binutils are installed.  Two details: derive the name
# from the driver's *basename* (a `ccache <driver>` wrapper must not become the
# readelf candidate), and use $(subst) because a patsubst pattern like `g++%'
# does not match the middle of `arm-linux-gnueabihf-g++-12' and would hand back
# the compiler itself.
CROSS_READELF ?= $(shell for r in $(subst g++,readelf,$(notdir $(CROSS_CXX))) \
                                  arm-linux-gnueabihf-readelf readelf; do \
                   command -v $$r >/dev/null 2>&1 && { echo $$r; break; }; done)
# Prefer the runtime matching the driver, else the highest present (a glob sorts
# ascending, so lastword is the newest).  Both `$(if ...)` guards matter: with no
# driver the major is empty, and `.../arm-linux-gnueabihf//` would match the
# parent directory and pass the check with a meaningless -B.  `?=` keeps an
# environment or command-line override working; an immediate `:=` ignores both.
CROSS_DRIVER_MAJOR = $(if $(CROSS_CXX),$(shell $(CROSS_CXX) -dumpversion 2>/dev/null | cut -d. -f1))
CROSS_GCC_DIRS = $(if $(CROSS_DRIVER_MAJOR),$(wildcard $(CROSS_SYSROOT)/usr/lib/gcc/arm-linux-gnueabihf/*/))
CROSS_GCC_DIR ?= $(strip $(if $(and $(CROSS_DRIVER_MAJOR),$(wildcard $(CROSS_SYSROOT)/usr/lib/gcc/arm-linux-gnueabihf/$(CROSS_DRIVER_MAJOR)/)),\
                   $(CROSS_SYSROOT)/usr/lib/gcc/arm-linux-gnueabihf/$(CROSS_DRIVER_MAJOR),\
                   $(patsubst %/,%,$(lastword $(CROSS_GCC_DIRS)))))
CROSS_TARGETLIB := $(CROSS_SYSROOT)/usr/lib/arm-linux-gnueabihf
CROSS_CXXFLAGS ?= --sysroot=$(CROSS_SYSROOT) \
                 -B$(CROSS_GCC_DIR)/ -B$(CROSS_TARGETLIB)/ \
                 -I$(CROSS_SYSROOT)/usr/include \
                 -O2 -marm -march=armv6 -mfpu=vfp -mfloat-abi=hard \
                 -ffast-math -std=c++17
CROSS_LDLIBS ?= -L$(CROSS_TARGETLIB) -lEGL -lGLESv2 -lm

.PHONY: pi armv6 armv6-check check-cross clean probes check warncheck test force-stamp

pi: engine_mm3_pi

# Every cross recipe depends on this.  Each check guards a condition that used to
# fail silently and leave an ARMv7 binary behind.
check-cross:
	@test -n "$(CROSS_SYSROOT)" || { \
	  echo "CROSS_SYSROOT is not set: pass the ARMv6 Raspbian root filesystem." >&2; \
	  echo "  make -C engine -B pi CROSS_SYSROOT=/path/to/pi-rootfs" >&2; \
	  echo "  how to build that rootfs and the cross compiler: docs/cross-build.md" >&2; \
	  echo "  (there is no default: every default is wrong on some host)" >&2; \
	  exit 2; }
	@test -d "$(CROSS_TARGETLIB)" || { \
	  echo "CROSS_SYSROOT=$(CROSS_SYSROOT) has no $(CROSS_TARGETLIB)" >&2; \
	  echo "  it must be a Raspbian root filesystem: Debian's armhf is ARMv7 and" >&2; \
	  echo "  cannot run on a Pi Zero.  docs/cross-build.md" >&2; \
	  exit 2; }
	@command -v "$(CROSS_CXX)" >/dev/null 2>&1 || { \
	  echo "no usable cross compiler: CROSS_CXX='$(CROSS_CXX)'" >&2; \
	  echo "  install it (g++-arm-linux-gnueabihf) or set CROSS_CXX to the driver." >&2; \
	  echo "  docs/cross-build.md" >&2; \
	  exit 2; }
	@test -n "$(CROSS_GCC_DIR)" && test -d "$(CROSS_GCC_DIR)" || { \
	  echo "no gcc runtime under $(CROSS_SYSROOT)/usr/lib/gcc/arm-linux-gnueabihf/" >&2; \
	  echo "  found: $(if $(CROSS_GCC_DIRS),$(CROSS_GCC_DIRS),(nothing))" >&2; \
	  echo "  without it -B is ignored and the link uses the host's ARMv7 startfiles;" >&2; \
	  echo "  the result builds and then faults on the ARM1176 (no diagnostic)" >&2; \
	  echo "  override with CROSS_GCC_DIR=<dir> if the glob picked wrong" >&2; \
	  exit 2; }
	@printf 'cross: %s (%s)\n' "$(CROSS_CXX)" "$$($(CROSS_CXX) -dumpversion)"
	@printf '  sysroot %s\n  runtime %s\n  libs    %s\n' \
	  "$(CROSS_SYSROOT)" "$(CROSS_GCC_DIR)" "$(CROSS_TARGETLIB)"

# The arch attributes are the only proof that the -B discovery matched reality:
# -march=armv6 on our own objects does not stop ARMv7 startfiles from raising
# Tag_CPU_arch.  The .comment line names the runtime that was actually linked.
engine_mm3_pi: $(PI_OBJ) | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) $(WARN) $(SECTIONS) $(INCS) -o $@ $(PI_OBJ) $(CROSS_LDLIBS) $(LDFLAGS)
	@if [ -n "$(CROSS_READELF)" ]; then \
	  attrs=$$($(CROSS_READELF) -A $@); \
	  echo "$$attrs" | grep -q 'Tag_CPU_arch: v6' || { \
	    echo "FAIL: $@ is not ARMv6 - it will fault on the ARM1176" >&2; exit 1; }; \
	  echo "$$attrs" | grep -q 'Tag_FP_arch: VFPv2' || { \
	    echo "FAIL: $@ is not VFPv2" >&2; exit 1; }; \
	  echo "  $@: ARMv6/VFPv2, $$($(CROSS_READELF) -p .comment $@ 2>/dev/null | sed -n 's/.*\(GCC: ([^)]*)\).*/\1/p' | head -1)"; \
	else echo "FAIL: no readelf found, so $@ is unverified; set CROSS_READELF" >&2; \
	     echo "  (any host readelf reads ARM ELF; a build that cannot be checked is not shipped)" >&2; \
	     exit 1; fi

# Build on the Pi itself: the host compiler already targets ARMv6, so there is
# no sysroot and no -B.  This target exists so the READMEs stop hand-copying the
# flag list (they had lost -marm, which the cross target carries).
ARMV6_CXXFLAGS ?= -O2 -marm -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17
armv6: $(ARMV6_OBJ) | armv6-check
	$(CXX) $(ARMV6_CXXFLAGS) $(WARN) $(SECTIONS) $(INCS) -o engine_mm3 $(ARMV6_OBJ) $(LDLIBS) $(LDFLAGS)
	@echo "  built engine_mm3 for this host (a Pi Zero reports ARMv6/VFPv2)"

# The host has to be ARM before an ARMv6 object is built; a host object under
# build/armv6 would be a v7 binary with a v6 name.
armv6-check:
	@$(CXX) -dumpmachine 2>/dev/null | grep -q '^arm' || { \
	  echo "armv6 builds on the Pi itself; this host targets $$($(CXX) -dumpmachine)." >&2; \
	  echo "  To build the Pi binary from here: make -C engine -B pi CROSS_SYSROOT=<sysroot>" >&2; \
	  exit 2; }

# Does the summation order matter on this driver, at the magnitudes a small
# model's residual reaches?  See the header of the source for the case.
probes/sumorder_probe: probes/sumorder_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

probes/sumorder_probe_pi: probes/sumorder_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/vc4_probe: probes/vc4_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

probes/vc4_ffn_probe: probes/vc4_ffn_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

probes/vc4_topk_probe: probes/vc4_topk_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

# No GL, no model: the logits grid is a text and source question.  The attention
# TU calls exp_poly_source(), which lives in shaders_common.cpp, so this links
# the common seam and the switch registry it reads.
probes/logits_grid_probe: probes/logits_grid_probe.cpp $(SRC)/shaders_attn.cpp \
                          $(SRC)/shaders_common.cpp $(SRC)/shaders.h $(SRC)/xformat.cpp \
                          $(SRC)/xformat.h $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/logits_grid_probe.cpp \
	  $(SRC)/shaders_attn.cpp $(SRC)/shaders_common.cpp $(SRC)/xformat.cpp \
	  $(SRC)/common.cpp $(SRC)/switches.cpp

probes/matvec_probe: probes/matvec_probe.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(SRC)/common.h $(HEADERS) $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/matvec_probe.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)

# The board build.  matvec_probe prices the q8 decode, and the decode cost is a
# VC4 question: a laptop run answers a different driver's question.
probes/matvec_probe_pi: probes/matvec_probe.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(SRC)/common.h | check-cross $(HEADERS) $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/matvec_probe.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(CROSS_LDLIBS)

probes/texsize_probe: probes/texsize_probe.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(SRC)/common.h $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/texsize_probe.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)

# Does a per-row q8 scale plane reach the shader?  Links the engine's own
# shader builders, so the fragment program under test is the shipped one.
probes/wrow_probe: probes/wrow_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/wrow_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)

# Do the conv block's fetches work on this driver?  Links the engine's own
# shader builders, so sections B and C run the shipped text.
probes/conv_probe: probes/conv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/conv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)

probes/conv_probe_pi: probes/conv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/conv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(CROSS_LDLIBS)

# What does the v5 LayerNorm's (mean, invstd) q16 pair cost as the residual
# spread moves, and does (mean, sd) + a per-row 2^-k hold.  Links the shipped
# ln_stats_q16/ln_apply_q16 so section A is the engine's own text.
probes/lnstats_probe: probes/lnstats_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/lnstats_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)

probes/rwkv_probe: probes/rwkv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/rwkv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)

probes/rwkv_probe_pi: probes/rwkv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/rwkv_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(CROSS_LDLIBS)

probes/wrow_probe_pi: probes/wrow_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/wrow_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(CROSS_LDLIBS)

# Standalone VC4 format probes: no engine sources, cross-built for the Pi.
probes/q4fmt_probe_pi: probes/q4fmt_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# 4-bit-per-channel texture support.  Standalone (EGL + GLES2, no engine
# objects), so it still builds while engine/src is being split, and it is the
# probe that answers this on the board rather than from the driver's format
# table.  The board has gcc and the Mesa headers, so the native recipe is the
# one usually used: g++ -O2 -std=c++17 -o rgba4444_probe ... -lEGL -lGLESv2.
# The ternary matvec pricing probe: 2-bit weights against the q8 baseline, with
# the ES3 uint decode Q4_SHIFT already uses.  EGL + GLES2 only, so it builds and
# runs on the board natively (g++ + the Mesa headers) as well as here.
probes/att_batch_probe: probes/att_batch_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/att_batch_probe_pi: probes/att_batch_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# The output-shape follow-up: does the query-batching win survive writing every
# score (one texel a fragment), at engine fragment counts, and is MRT there?
probes/att_out_probe: probes/att_out_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/att_out_probe_pi: probes/att_out_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# Does the qkv matvec's 2-row pairing leave the 2x on the table that the FFN's
# 4-row swiglu kernel banks?  Same decode, same weights, only the rows a
# fragment amortises it over change.
# Why is an engine draw 0.48 ms when a pipelined null draw is 0.008?  Chain
# shape, finish policy and dependency, on the same shader and target size.
# The minimum reproducer for "a full-target-width draw is dropped": w ==
# target_w against the engine's w-1 control, several widths, and a repeat test.
probes/vc4_draw_drop_probe: probes/vc4_draw_drop_probe.cpp probes/egl_surfaceless.h $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/vc4_draw_drop_probe_pi: probes/vc4_draw_drop_probe.cpp probes/egl_surfaceless.h | check-cross $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/draw_cost_probe: probes/draw_cost_probe.cpp probes/egl_surfaceless.h $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/draw_cost_probe_pi: probes/draw_cost_probe.cpp probes/egl_surfaceless.h | check-cross $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# What each XFormat's encode and decode cost per value, from the engine's own
# emitted codecs, with an exact round-trip check.  ES3 for the bit-cast arms.
probes/codec_probe: probes/codec_probe.cpp probes/egl_surfaceless.h $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS) $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/codec_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(LDLIBS)
probes/codec_probe_pi: probes/codec_probe.cpp probes/egl_surfaceless.h $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp | check-cross $(HEADERS) $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/codec_probe.cpp $(SHADERLIB) $(SRC)/xformat.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(CROSS_LDLIBS)

# What the q16 activation decode costs, and whether the sign fix can be made
# cheaper without changing a byte.  Runs inside the real matvec loop.
probes/q16dec_probe: probes/q16dec_probe.cpp probes/egl_surfaceless.h $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/q16dec_probe_pi: probes/q16dec_probe.cpp probes/egl_surfaceless.h | check-cross $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/qkv_rows_probe: probes/qkv_rows_probe.cpp probes/egl_surfaceless.h $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/qkv_rows_probe_pi: probes/qkv_rows_probe.cpp probes/egl_surfaceless.h | check-cross $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/t3mv_probe: probes/t3mv_probe.cpp probes/egl_surfaceless.h $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/t3mv_probe_pi: probes/t3mv_probe.cpp probes/egl_surfaceless.h | check-cross $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# The T3 decode microbenchmark: layout/fetch/accumulation fixed, only dec() varies.
probes/t3dec_probe: probes/t3dec_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/t3dec_probe_pi: probes/t3dec_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# T-MAC-style texture LUT vs decode: can the dot skip per-weight decode?
probes/t3lut_probe: probes/t3lut_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/t3lut_probe_pi: probes/t3lut_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# Activation format (I8_4 vs I16_2) effect on the t3 pair matvec.
probes/t3act_probe: probes/t3act_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/t3act_probe_pi: probes/t3act_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# rmsnorm normalize draw cost: what folding it into the consumer matvec saves.
probes/rmsfuse_probe: probes/rmsfuse_probe.cpp probes/egl_surfaceless.h $(PROBE_HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/rmsfuse_probe_pi: probes/rmsfuse_probe.cpp probes/egl_surfaceless.h | check-cross $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/rgba4444_probe: probes/rgba4444_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/rgba4444_probe_pi: probes/rgba4444_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)
probes/progbinary_probe_pi: probes/progbinary_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)
probes/load_probe_pi: probes/load_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# The half of the gate that needs no GPU and no model: what a contributor can run
# before pushing.  This is the only list -- .github/workflows/gate.yml runs this
# target rather than repeating the steps, because the two copies drifted once
# (check_q16_decode.sh was missing from the workflow until 2026-11-17).
check: test warncheck
	sh gate/docs_check.sh
	sh gate/shader_hashes.sh
	sh gate/check_q16_decode.sh
	sh gate/check_draw_cap.sh
	sh gate/check_target_reset.sh
	python3 gate/check_weight_formats.py
	python3 gate/test_mmcf_format.py
	python3 ../server/selftest.py

# The compiled tests, built and run without warncheck's forced engine rebuild:
# `make test` is the fast loop (a few seconds), and `check` runs it first so a
# broken unit fails before the minute-long build.
test: test_lineedit test_tokenizer test_xformat test_grammar test_schema test_gbnf_full test_constraint test_weight_format
	./test_lineedit
	./test_tokenizer
	./test_xformat
	./test_grammar
	./test_schema
	./test_gbnf_full
	./test_constraint
	./test_weight_format

# Warnings from src/ are a defect: -Wall -Wextra is on by default and this makes a
# clean build an assertion rather than a hope.  Warnings from third_party/ are
# the vendored Jinja engine's own (eight static helpers instantiated in four
# translation units) and are filtered here instead of silenced globally, so the
# flag keeps working for our code.
warncheck:
	@log=$$(mktemp); \
	if $(MAKE) -B -j$(JOBS) engine_mm3 > $$log 2>&1; then :; else cat $$log; rm -f $$log; exit 1; fi; \
	if grep -qE '^src/[^:]+:[0-9]+:[0-9]+: warning' $$log; then \
	    echo "warnings from src/ (a clean build is required):"; \
	    grep -E '^src/[^:]+:[0-9]+:[0-9]+: warning' $$log; rm -f $$log; exit 1; \
	fi; \
	n=$$(grep -c 'warning:' $$log || true); echo "warncheck: 0 warnings from src/ ($${n:-0} from vendored third_party/ only)"; \
	rm -f $$log

# The line editor's layout arithmetic, its paste sanitiser, and the editor itself
# driven through a pty: no GPU, no model, no terminal.  It is part of `check`
# because where the cursor lands is arithmetic no build warning can catch -- the
# old relative `ESC[nD` move was wrong on every wrapped line and looked fine.
test_lineedit: tests/test_lineedit.cpp $(SRC)/lineedit.cpp $(SRC)/lineedit_layout.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_lineedit.cpp $(SRC)/lineedit.cpp $(SRC)/lineedit_layout.cpp -lutil -pthread

# The tokenizer's .tok blob, BPE rank, special-token atomicity, the user/template
# split and decode: a synthetic blob, so no model and no GPU.  It is a unit test
# rather than a golden string because a tokenizer that splits a marker still
# answers plausibly -- the failure is an id that means something else.
test_tokenizer: tests/test_tokenizer.cpp tests/checks.h $(SRC)/tokenizer.cpp \
                $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_tokenizer.cpp $(SRC)/tokenizer.cpp \
	  $(SRC)/common.cpp $(SRC)/switches.cpp -lm

# The CPU half of the X formats: x_pack's bytes are what the GLSL decoders undo,
# so a change to one without the other is a numerics bug the shader digest cannot
# see.  No GPU, no model.
test_xformat: tests/test_xformat.cpp tests/checks.h $(SRC)/xformat.cpp \
              $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_xformat.cpp $(SRC)/xformat.cpp \
	  $(SRC)/common.cpp $(SRC)/switches.cpp -lm

# The JSON constraint's byte automaton: valid/invalid/prefix JSON, tokens that
# span construct boundaries, and the EOS-only-when-done rule.  A synthetic byte
# vocabulary, so no model and no GPU.  It is a unit test because a grammar that
# admits one illegal byte still answers plausibly -- the failure is a downstream
# JSON parse error the engine would never see.
test_grammar: tests/test_grammar.cpp tests/checks.h $(SRC)/grammar.cpp \
              $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_grammar.cpp $(SRC)/grammar.cpp \
	  $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp -lm

# The generic grammar matcher and the JSON-Schema compiler: the JSON grammar must
# reproduce test_grammar's corpus, and compiled schemas must accept/reject the
# right instances.  Synthetic byte vocabulary, no model, no GPU.  Links the
# vendored json.cpp for common_json (schema parsing).
test_schema: tests/test_schema.cpp tests/checks.h $(SRC)/gbnf.cpp $(SRC)/schema.cpp \
             $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(TP)/jinja/json.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_schema.cpp $(SRC)/gbnf.cpp $(SRC)/schema.cpp \
	  $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(TP)/jinja/json.cpp -lm

# Full GBNF: the general multi-stack matcher and text parser, on ambiguous and
# repeated grammars the LL(1) matcher cannot handle.  Synthetic byte vocab, no
# model, no GPU.
test_gbnf_full: tests/test_gbnf_full.cpp tests/checks.h $(SRC)/gbnf_full.cpp \
                $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_gbnf_full.cpp $(SRC)/gbnf_full.cpp \
	  $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp -lm

test_constraint: tests/test_constraint.cpp tests/checks.h $(SRC)/constrain.cpp \
                 $(SRC)/gbnf.cpp $(SRC)/gbnf_full.cpp $(SRC)/schema.cpp $(SRC)/grammar.cpp \
                 $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp \
                 $(TP)/jinja/json.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_constraint.cpp $(SRC)/constrain.cpp \
	  $(SRC)/gbnf.cpp $(SRC)/gbnf_full.cpp $(SRC)/schema.cpp $(SRC)/grammar.cpp \
	  $(SRC)/tokenizer.cpp $(SRC)/common.cpp $(SRC)/switches.cpp $(TP)/jinja/json.cpp -lm

# The weight-format contract: row order, caps, slice alignment, the wire dtype
# mapping and mat_bytes' arithmetic.  Header-only, so no GPU and no model.
test_weight_format: tests/test_weight_format.cpp tests/checks.h $(SRC)/weight_format.h \
                    $(SRC)/common.h $(SRC)/xformat.h
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ tests/test_weight_format.cpp

clean:
	rm -rf build
	rm -f engine engine_mm3 engine_mm3_pi test_lineedit test_tokenizer test_xformat test_grammar test_schema test_gbnf_full test_constraint test_weight_format
	rm -f probes/vc4_probe probes/vc4_ffn_probe probes/vc4_topk_probe probes/load_probe_pi
	rm -f probes/matvec_probe probes/texsize_probe probes/q4fmt_probe_pi probes/glsl_probe_pi probes/q4mv_probe_pi probes/draw_probe_pi probes/packunpack_probe probes/packunpack_probe_pi probes/logits_grid_probe probes/wrow_probe probes/wrow_probe_pi probes/conv_probe probes/conv_probe_pi probes/t3mv_probe probes/t3mv_probe_pi probes/t3dec_probe probes/t3dec_probe_pi probes/t3lut_probe probes/t3lut_probe_pi probes/t3act_probe probes/t3act_probe_pi probes/rmsfuse_probe probes/rmsfuse_probe_pi probes/att_batch_probe probes/att_batch_probe_pi probes/att_out_probe probes/att_out_probe_pi \
	rm -f probes/sumorder_probe probes/sumorder_probe_pi probes/q4mv_probe probes/progbinary_probe_pi probes/ffn2_probe_pi probes/rgba4444_probe probes/rgba4444_probe_pi

probes/glsl_probe_pi: probes/glsl_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# Which GLSL ES pack/unpack builtins actually run on this driver (VC4 needs the
# answer more than Xe does; see the probe's header).
probes/packunpack_probe_pi: probes/packunpack_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/q4mv_probe_pi: probes/q4mv_probe.cpp $(GPULIB) $(HEADERS) | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/q4mv_probe.cpp $(GPULIB) $(CROSS_LDLIBS)

# Native build of the same probe: the q4 decode A/B (Q4_NOUNPACK, Q4_BFAST,
# Q4_SHIFT=0) is a per-texel difference of a few ALU ops, and a whole-token
# decode average on Xe is too noisy to resolve it (15/23/44 ms/token for the
# same arm in one session), so the kernel needs its own instrument.
probes/q4mv_probe: probes/q4mv_probe.cpp $(GPULIB) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(INCS) -o $@ probes/q4mv_probe.cpp $(GPULIB) $(LDLIBS)

# Prices a draw, an rmsnorm and the q8 matvec shapes on the real driver
# (WORKLOG item 7: the cost model behind prefill batching).
probes/draw_probe_pi: probes/draw_probe.cpp $(GPULIB) | check-cross $(HEADERS) $(PROBE_HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/draw_probe.cpp $(GPULIB) $(CROSS_LDLIBS)

# Prices 2-position packing: one weight fetch feeding two positions, and the
# P2 layout it writes (WORKLOG item 7b).
probes/ffn2_probe_pi: probes/ffn2_probe.cpp $(GPULIB) | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/ffn2_probe.cpp $(GPULIB) $(CROSS_LDLIBS)
