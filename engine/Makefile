CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
LDLIBS ?= -lEGL -lGLESv2 -lm

engine: engine.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

engine_mm3: engine.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f engine engine_mm3
