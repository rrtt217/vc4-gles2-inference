CXX ?= g++
# -std is not in CXXFLAGS: `CXXFLAGS ?=` let a caller's `make CXXFLAGS='-g'`
# silently drop the standard (and the optimisation) with it.  Warnings are on by
# default, because a fresh clone that builds without them is how a dead lambda sat
# in main() unnoticed (2026-11-14).
CXXSTD := -std=c++17
OPT ?= -O2
WARN ?= -Wall -Wextra
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
# target that still listed the old file failed to link (2026-11-15).
GPULIB := $(wildcard $(SRC)/gpu*.cpp) $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp \
          $(SRC)/xformat.cpp $(SRC)/common.cpp
CORE := $(SRC)/engine.cpp $(SRC)/common.cpp $(SRC)/xformat.cpp \
        $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp $(SRC)/dump_shaders.cpp \
        $(SRC)/gpu.cpp $(SRC)/gpu_state.cpp $(SRC)/gpu_conv.cpp $(SRC)/gpu_diag.cpp $(SRC)/gpu_prog.cpp $(SRC)/gpu_attn.cpp $(SRC)/gpu_ffn.cpp $(SRC)/gpu_matvec.cpp $(SRC)/gpu_upload.cpp $(SRC)/gpu_gl.cpp \
        $(SRC)/tokenizer.cpp $(SRC)/model.cpp $(SRC)/session.cpp $(SRC)/prefix_cache.cpp $(SRC)/lineedit.cpp \
        $(SRC)/sampling.cpp $(SRC)/tools.cpp $(SRC)/ppl.cpp $(SRC)/cli.cpp $(SRC)/chat_format.cpp $(SRC)/prefix_build.cpp $(SRC)/app.cpp $(SRC)/worker.cpp $(SRC)/chat.cpp $(SRC)/chattmpl.cpp $(JINJA)
HEADERS := $(wildcard $(SRC)/*.h)
# third_party is -isystem: it is vendored code (llama.cpp's Jinja engine) and its
# own warnings are not this repository's to fix, but they must not drown ours --
# before this, a full build printed 11 warnings from jinja headers and 1 from src/.
INCS := -I$(SRC) -isystem $(TP)
# The Jinja engine is large and only one path uses it; let the linker drop what
# this build never reaches (-947 KB -> -1.19 MB stripped, measured on Xe).
SECTIONS := -ffunction-sections -fdata-sections
LDFLAGS ?= -Wl,--gc-sections

engine_mm3: $(CORE) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) $(SECTIONS) $(INCS) -o $@ $(CORE) $(LDLIBS) $(LDFLAGS)


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

.PHONY: pi armv6 check-cross clean probes check warncheck

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
engine_mm3_pi: $(CORE) $(HEADERS) | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) $(WARN) $(SECTIONS) $(INCS) -o $@ $(CORE) $(CROSS_LDLIBS) $(LDFLAGS)
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
armv6: $(CORE) $(HEADERS)
	@$(CXX) -dumpmachine 2>/dev/null | grep -q '^arm' || { \
	  echo "armv6 builds on the Pi itself; this host targets $$($(CXX) -dumpmachine)." >&2; \
	  echo "  To build the Pi binary from here: make -C engine -B pi CROSS_SYSROOT=<sysroot>" >&2; \
	  exit 2; }
	$(CXX) $(ARMV6_CXXFLAGS) $(WARN) $(SECTIONS) $(INCS) -o engine_mm3 $(CORE) $(LDLIBS) $(LDFLAGS)
	@echo "  built engine_mm3 for this host (a Pi Zero reports ARMv6/VFPv2)"

probes/vc4_probe: probes/vc4_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

probes/vc4_ffn_probe: probes/vc4_ffn_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

probes/vc4_topk_probe: probes/vc4_topk_probe.cpp
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)

# No GL, no model: the logits grid is a text and source question.
probes/logits_grid_probe: probes/logits_grid_probe.cpp $(SRC)/shaders_attn.cpp \
                          $(SRC)/shaders.h $(SRC)/xformat.cpp $(SRC)/xformat.h $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/logits_grid_probe.cpp $(SRC)/shaders_attn.cpp $(SRC)/xformat.cpp

probes/matvec_probe: probes/matvec_probe.cpp $(SRC)/common.cpp $(SRC)/common.h $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/matvec_probe.cpp $(SRC)/common.cpp $(LDLIBS)

probes/texsize_probe: probes/texsize_probe.cpp $(SRC)/common.cpp $(SRC)/common.h $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/texsize_probe.cpp $(SRC)/common.cpp $(LDLIBS)

# Does a per-row q8 scale plane reach the shader?  Links the engine's own
# shader builders, so the fragment program under test is the shipped one.
probes/wrow_probe: probes/wrow_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/wrow_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp $(LDLIBS)

# Do the conv block's fetches work on this driver?  Links the engine's own
# shader builders, so sections B and C run the shipped text.
probes/conv_probe: probes/conv_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ probes/conv_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp $(LDLIBS)

probes/conv_probe_pi: probes/conv_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/conv_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp $(CROSS_LDLIBS)

probes/wrow_probe_pi: probes/wrow_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ probes/wrow_probe.cpp $(SRC)/shaders.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp $(CROSS_LDLIBS)

# Standalone VC4 format probes: no engine sources, cross-built for the Pi.
probes/q4fmt_probe_pi: probes/q4fmt_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# 4-bit-per-channel texture support.  Standalone (EGL + GLES2, no engine
# objects), so it still builds while engine/src is being split, and it is the
# probe that answers this on the board rather than from the driver's format
# table.  The board has gcc and the Mesa headers, so the native recipe is the
# one usually used: g++ -O2 -std=c++17 -o rgba4444_probe ... -lEGL -lGLESv2.
probes/rgba4444_probe: probes/rgba4444_probe.cpp probes/egl_surfaceless.h
	$(CXX) $(ALL_CXXFLAGS) -o $@ $< $(LDLIBS)
probes/rgba4444_probe_pi: probes/rgba4444_probe.cpp probes/egl_surfaceless.h | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)
probes/progbinary_probe_pi: probes/progbinary_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)
probes/load_probe_pi: probes/load_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# The half of the gate that needs no GPU and no model: what a contributor can run
# before pushing, and what .github/workflows/gate.yml runs on every push.
check: warncheck
	sh scripts/docs_check.sh
	sh scripts/shader_hashes.sh
	sh scripts/check_q16_decode.sh
	sh scripts/check_draw_cap.sh
	sh scripts/check_target_reset.sh
	python3 scripts/test_mmcf_format.py
	python3 ../server/selftest.py

# Warnings from src/ are a defect: -Wall -Wextra is on by default and this makes a
# clean build an assertion rather than a hope.  Warnings from third_party/ are
# the vendored Jinja engine's own (eight static helpers instantiated in four
# translation units) and are filtered here instead of silenced globally, so the
# flag keeps working for our code.
warncheck:
	@log=$$(mktemp); \
	if $(MAKE) -B engine_mm3 > $$log 2>&1; then :; else cat $$log; rm -f $$log; exit 1; fi; \
	if grep -qE '^src/[^:]+:[0-9]+:[0-9]+: warning' $$log; then \
	    echo "warnings from src/ (a clean build is required):"; \
	    grep -E '^src/[^:]+:[0-9]+:[0-9]+: warning' $$log; rm -f $$log; exit 1; \
	fi; \
	n=$$(grep -c 'warning:' $$log || true); echo "warncheck: 0 warnings from src/ ($${n:-0} from vendored third_party/ only)"; \
	rm -f $$log

clean:
	rm -f engine engine_mm3 engine_mm3_pi
	rm -f probes/vc4_probe probes/vc4_ffn_probe probes/vc4_topk_probe probes/load_probe_pi
	rm -f probes/matvec_probe probes/texsize_probe probes/q4fmt_probe_pi probes/glsl_probe_pi probes/q4mv_probe_pi probes/draw_probe_pi probes/packunpack_probe probes/packunpack_probe_pi probes/logits_grid_probe probes/wrow_probe probes/wrow_probe_pi probes/conv_probe probes/conv_probe_pi \
	rm -f probes/q4mv_probe probes/progbinary_probe_pi probes/ffn2_probe_pi probes/rgba4444_probe probes/rgba4444_probe_pi

probes/glsl_probe_pi: probes/glsl_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# Which GLSL ES pack/unpack builtins actually run on this driver (VC4 needs the
# answer more than Xe does; see the probe's header).
probes/packunpack_probe_pi: probes/packunpack_probe.cpp | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/q4mv_probe_pi: probes/q4mv_probe.cpp $(GPULIB) $(HEADERS) | check-cross
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ $^ $(CROSS_LDLIBS)

# Native build of the same probe: the q4 decode A/B (Q4_NOUNPACK, Q4_BFAST,
# Q4_SHIFT=0) is a per-texel difference of a few ALU ops, and a whole-token
# decode average on Xe is too noisy to resolve it (15/23/44 ms/token for the
# same arm in one session), so the kernel needs its own instrument.
probes/q4mv_probe: probes/q4mv_probe.cpp $(GPULIB) $(HEADERS)
	$(CXX) $(ALL_CXXFLAGS) -I$(SRC) -o $@ $^ $(LDLIBS)

# Prices a draw, an rmsnorm and the q8 matvec shapes on the real driver
# (WORKLOG item 7: the cost model behind prefill batching).
probes/draw_probe_pi: probes/draw_probe.cpp $(GPULIB) | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ $^ $(CROSS_LDLIBS)

# Prices 2-position packing: one weight fetch feeding two positions, and the
# P2 layout it writes (WORKLOG item 7b).
probes/ffn2_probe_pi: probes/ffn2_probe.cpp $(GPULIB) | check-cross $(HEADERS)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ $^ $(CROSS_LDLIBS)
