CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
LDLIBS ?= -lEGL -lGLESv2 -lm

SRC := src
CORE := $(SRC)/engine.cpp $(SRC)/common.cpp $(SRC)/xformat.cpp \
        $(SRC)/shaders.cpp $(SRC)/shaders_attn.cpp $(SRC)/gpu.cpp \
        $(SRC)/tokenizer.cpp $(SRC)/model.cpp $(SRC)/session.cpp \
        $(SRC)/sampling.cpp $(SRC)/tools.cpp
HEADERS := $(wildcard $(SRC)/*.h)

engine: $(CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(CORE) $(LDLIBS)

engine_mm3: $(CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(CORE) $(LDLIBS)


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
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -o $@ $(CORE) $(CROSS_LDLIBS)

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
	rm -f probes/matvec_probe probes/texsize_probe probes/q4fmt_probe_pi
