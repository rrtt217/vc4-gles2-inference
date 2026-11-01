#!/usr/bin/env python3
import sys
PATH = "engine.cpp"
src = open(PATH).read()

def sub(old, new, label, count):
    global src
    n = src.count(old)
    if n != count:
        print(f"FAIL {label}: matched {n}, expected {count}"); sys.exit(1)
    src = src.replace(old, new)
    print(f"ok   {label} ({count})")

sub('float dA(vec4 c){float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);',
    'float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;', 'activation dA', 6)
sub('float dB(vec4 c){float u=floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5);',
    'float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;', 'activation dB', 6)
sub('vec4 w=floor(c*255.0+0.5)-128.0;', 'vec4 w=c*255.0-128.0;', 'q8 weights', 2)
sub('float u0=floor(c.g*255.0+0.5)*256.0+floor(c.r*255.0+0.5);',
    'float u0=c.g*65280.0+c.r*255.0;', 'q16 weight u0', 2)
sub('float u1=floor(c.a*255.0+0.5)*256.0+floor(c.b*255.0+0.5);',
    'float u1=c.a*65280.0+c.b*255.0;', 'q16 weight u1', 2)
sub('float d16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u;}',
    'float d16(vec2 c){float u=c.x*65280.0+c.y*255.0;if(u>=32768.0)u-=65536.0;return u;}', 'q4 scale/min', 2)
sub('float b0=floor(w.r*255.0+0.5);float b1=floor(w.g*255.0+0.5);float b2=floor(w.b*255.0+0.5);float b3=floor(w.a*255.0+0.5);',
    'float b0=w.r*255.0;float b1=w.g*255.0;float b2=w.b*255.0;float b3=w.a*255.0;', 'q4 nibbles', 2)
sub('float dr(vec4 c){float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}',
    'float dr(vec4 c){float u=c.r*65280.0+c.g*255.0;if(u>=32768.0)u-=65536.0;return u/256.0;}', 'residual', 2)
sub('float dec16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}',
    'float dec16(vec2 c){float u=c.x*65280.0+c.y*255.0;if(u>=32768.0)u-=65536.0;return u/256.0;}', 'rmsnorm input', 3)
open(PATH, "w").write(src)
print("wrote", PATH)
