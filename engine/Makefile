CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
LDLIBS ?= -lEGL -lGLESv2 -lm

SRC := src
TP  := third_party
# Jinja chat templates (vendored from llama.cpp; see third_party/jinja/README.md).
# json.cpp is the only translation unit that includes nlohmann, so the compile
# cost stays in one place.
JINJA := $(TP)/jinja/lexer.cpp $(TP)/jinja/parser.cpp $(TP)/jinja/runtime.cpp \
         $(TP)/jinja/string.cpp $(TP)/jinja/value.cpp $(TP)/jinja/json.cpp \
         $(TP)/jinja/unicode.cpp
CORE := $(SRC)/engine.cpp $(SRC)/common.cpp $(SRC)/xformat.cpp \
        $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp $(SRC)/gpu.cpp \
        $(SRC)/tokenizer.cpp $(SRC)/model.cpp $(SRC)/session.cpp \
        $(SRC)/sampling.cpp $(SRC)/tools.cpp $(SRC)/chattmpl.cpp $(JINJA)
HEADERS := $(wildcard $(SRC)/*.h)
INCS := -I$(SRC) -I$(TP)
# The Jinja engine is large and only one path uses it; let the linker drop what
# this build never reaches (-947 KB -> -1.19 MB stripped, measured on Xe).
SECTIONS := -ffunction-sections -fdata-sections
LDFLAGS ?= -Wl,--gc-sections

engine: $(CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SECTIONS) $(INCS) -o $@ $(CORE) $(LDLIBS) $(LDFLAGS)

engine_mm3: $(CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SECTIONS) $(INCS) -o $@ $(CORE) $(LDLIBS) $(LDFLAGS)


# Raspbian/ARMv6 cross build for Pi Zero.  CROSS_SYSROOT must be a full
# Raspbian root filesystem; the compiler's own startfiles are overridden with
# -B so the linked binary really uses ARMv6 glibc/libstdc++.
CROSS_CXX ?= arm-linux-gnueabihf-g++-12
CROSS_SYSROOT ?= $(HOME)/raspi-root4
CROSS_GCCLIB := $(CROSS_SYSROOT)/usr/lib/gcc/arm-linux-gnueabihf/14
CROSS_TARGETLIB := $(CROSS_SYSROOT)/usr/lib/arm-linux-gnueabihf
CROSS_CXXFLAGS ?= --sysroot=$(CROSS_SYSROOT) \
                 -B$(CROSS_GCCLIB)/ -B$(CROSS_TARGETLIB)/ \
                 -I$(CROSS_SYSROOT)/usr/include \
                 -O2 -marm -march=armv6 -mfpu=vfp -mfloat-abi=hard \
                 -ffast-math -std=c++17
CROSS_LDLIBS ?= -L$(CROSS_TARGETLIB) -lEGL -lGLESv2 -lm

pi: engine_mm3_pi

engine_mm3_pi: $(CORE) $(HEADERS)
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) $(SECTIONS) $(INCS) -o $@ $(CORE) $(CROSS_LDLIBS) $(LDFLAGS)

probes/vc4_probe: probes/vc4_probe.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

probes/vc4_ffn_probe: probes/vc4_ffn_probe.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

probes/vc4_topk_probe: probes/vc4_topk_probe.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

probes/matvec_probe: probes/matvec_probe.cpp $(SRC)/common.cpp $(SRC)/common.h
	$(CXX) $(CXXFLAGS) -I$(SRC) -o $@ probes/matvec_probe.cpp $(SRC)/common.cpp $(LDLIBS)

probes/texsize_probe: probes/texsize_probe.cpp $(SRC)/common.cpp $(SRC)/common.h
	$(CXX) $(CXXFLAGS) -I$(SRC) -o $@ probes/texsize_probe.cpp $(SRC)/common.cpp $(LDLIBS)

# Standalone VC4 format probes: no engine sources, cross-built for the Pi.
probes/q4fmt_probe_pi: probes/q4fmt_probe.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)
probes/progbinary_probe_pi: probes/progbinary_probe.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)
probes/load_probe_pi: probes/load_probe.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

clean:
	rm -f engine engine_mm3 engine_mm3_pi
	rm -f probes/vc4_probe probes/vc4_ffn_probe probes/vc4_topk_probe probes/load_probe_pi
	rm -f probes/matvec_probe probes/texsize_probe probes/q4fmt_probe_pi probes/glsl_probe_pi probes/q4mv_probe_pi probes/draw_probe_pi probes/packunpack_probe probes/packunpack_probe_pi

probes/glsl_probe_pi: probes/glsl_probe.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

# Which GLSL ES pack/unpack builtins actually run on this driver (VC4 needs the
# answer more than Xe does; see the probe's header).
probes/packunpack_probe_pi: probes/packunpack_probe.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $< $(CROSS_LDLIBS)

probes/q4mv_probe_pi: probes/q4mv_probe.cpp $(SRC)/gpu.cpp $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ $^ $(CROSS_LDLIBS)

# Prices a draw, an rmsnorm and the q8 matvec shapes on the real driver
# (WORKLOG item 7: the cost model behind prefill batching).
probes/draw_probe_pi: probes/draw_probe.cpp $(SRC)/gpu.cpp $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ $^ $(CROSS_LDLIBS)

# Prices 2-position packing: one weight fetch feeding two positions, and the
# P2 layout it writes (WORKLOG item 7b).
probes/ffn2_probe_pi: probes/ffn2_probe.cpp $(SRC)/gpu.cpp $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp $(SRC)/xformat.cpp $(SRC)/common.cpp
	@test -d "$(CROSS_SYSROOT)" || { echo "missing CROSS_SYSROOT=$(CROSS_SYSROOT)" >&2; exit 2; }
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(SRC) -o $@ $^ $(CROSS_LDLIBS)
