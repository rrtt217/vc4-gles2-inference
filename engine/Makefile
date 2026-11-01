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

clean:
	rm -f engine engine_mm3
	rm -f probes/vc4_probe probes/vc4_ffn_probe probes/vc4_topk_probe
	rm -f probes/matvec_probe probes/texsize_probe
