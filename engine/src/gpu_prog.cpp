// The program layer: one compiled program per kernel, cached by key *and* text.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  Attention, FFN
// and matvec all ask this layer for a program, which is why it moves before them:
// `x_prog(key, src)` is where the engine's two oldest invariants live -- a key is
// not a text (the same key yields different GLSL for another model, so the text is
// hashed on every hit) and the Q4_SHIFT probe decides at link time whether the
// driver's ESSL 3.00 compiler can be used for the q4 nibble unpack.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Q4_INLINE selects the INT4 activation-decode form per kernel group, so the
// generic and the inline build can be A/B-ed from one binary (the program
// cache key carries the choice).  Unset = all inline; Q4_INLINE=0 = all
// generic; otherwise a comma list of gate,w2,qkv enables only those.
bool q4_inline(const char* name){
    static const char* env = getenv("Q4_INLINE");
    if (!env) return true;
    static const std::string s = env;
    if (s.empty() || s == "0") return false;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find(",", p);
        const std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (tok == name || tok == "all") return true;
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return false;
}

bool GPU::shader_time_on(){
        static const bool on = env_set("SHADER_TIME");
        return on;
    }

double GPU::shader_time_ms(){
        return run_time_ms();
    }

// The key of the program currently being built.  A std::string, not a
// const char*: every caller passes a temporary ("xq8_" + to_string(rows)), and
// the pointer dangles as soon as the expression ends - which is exactly when a
// diagnostic or a uniform lookup still wants to name the program.
string g_prog_key = "?";
static Clock::time_point g_run_t0;

void gpu_set_run_t0(Clock::time_point t){ g_run_t0 = t; }

double run_time_ms(){
        return elapsed_ms(g_run_t0, Clock::now());
    }

GLuint GPU::x_prog(const string& key, const string& src){
        // SHADER_SRC=<substr> prints the generated GLSL for matching program
        // keys.  The alternative is reading the builder and guessing which
        // branch a format took, which is how a wrong decode survives a review.
        if (const char* f = getenv("SHADER_SRC"))
                if (key.find(f) != string::npos)
                        fprintf(stderr, "==== %s ====\n%s==== end %s ====\n",
                                key.c_str(), src.c_str(), key.c_str());
        // Two keys can build byte-identical text (xq8pl and xqkvp both do): a
        // program is a pure function of its shaders, so share the object and
        // skip the second compile.  64-bit hash; a collision would need two
        // different texts and is not a practical risk here.
        const size_t h = std::hash<string>{}(src);
        auto it = programs_.find(key);
        if (it != programs_.end()) {
            // Only a hit when the key still stands for this text.  A model
            // swap reuses keys (the builders name programs by kernel, not by
            // shape) while the GLSL changes with the head width, the qk-norm
            // presence and the activation format, so a key hit on its own
            // would run the previous model's kernel -- which is silent garbage
            // rather than a compile error.
            auto kh = prog_key_hash_.find(key);
            if (kh != prog_key_hash_.end() && kh->second == h) return it->second;
        }
        auto t = text_programs_.find(h);
        if (t != text_programs_.end()) {
            programs_[key] = t->second;
            prog_key_hash_[key] = h;
            if (shader_time_on())
                fprintf(stderr, "SHADER_TIME reuse key %s (identical text)\n", key.c_str());
            return t->second;
        }
        g_prog_key = key;
        GLuint p = link_program(src);
        check_program_runs(p, key);
        programs_[key] = p;
        prog_key_hash_[key] = h;
        text_programs_[h] = p;
        return p;
    }

bool g_q4_shift_ok = false;
bool g_es3_ok = false;

bool gpu_es3_ok(){ return g_es3_ok; }


static string es3ify(const string& src, bool vertex){
        if (src.find("#version") != string::npos) return src;
        if (vertex) return "#version 300 es\n#define attribute in\n" + src;
        // Three rules from the ESSL 3.00 specification, none of them optional on
        // a compiler that is not Mesa's:
        //
        //  * `gl_FragColor` is *removed* in ESSL 3.00, and `gl_`-prefixed macro
        //    names are reserved, so aliasing it with `#define` -- which this
        //    function did until 2026-11-18 -- is undefined behaviour that Mesa
        //    happens to allow.  The identifier is renamed in the text instead,
        //    which needs no reserved name.
        //  * `texture2D`/`textureCube` are removed in favour of `texture`.  The
        //    aliases below are legal macros (no `gl_` prefix) and keep every
        //    builder's text unchanged; Mesa accepts the removed spellings, a
        //    vendor compiler does not.
        //  * A fragment shader has no default float precision in ESSL 3.00, so
        //    the statement is injected rather than assumed.
        string body = src;
        for (size_t at = body.find("gl_FragColor"); at != string::npos;
             at = body.find("gl_FragColor", at + 12))
                body.replace(at, 12, "frag_color");
        return "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
               "#define texture2D texture\n#define textureCube texture\n" + body;
    }

bool q4_shift_wanted(){
        const char* e = getenv("Q4_SHIFT");
        return !(e && strcmp(e, "0") == 0);
    }

bool gpu_q4_shift(){ return g_q4_shift_ok; }

void gpu_force_q4_shift(bool on){ g_q4_shift_ok = on; }

GLuint GPU::compile_shader(GLenum type, const string& src){
        GLuint s = glCreateShader(type);
        const char* p = src.c_str();
        glShaderSource(s, 1, &p, nullptr);
        glCompileShader(s);
        GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[8192]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            // The source, not just the log: the log names a line number and
            // without the text that number is unreadable, which is how one of
            // these cost an hour.  SHADER_SRC=0 turns it off.
            cerr << "shader compile failed:\n" << log << "\n";
            if (!getenv("SHADER_SRC") || atoi(getenv("SHADER_SRC")) != 0) {
                    int n = 1;
                    cerr << "--- source ---\n";
                    for (size_t i = 0; i < src.size(); ++i) {
                            if (i == 0 || src[i - 1] == '\n') cerr << n++ << "| ";
                            cerr << src[i];
                    }
                    if (!src.empty() && src.back() != '\n') cerr << "\n";
                    cerr << "--- end ---\n";
            }
            exit(1);
        }
        return s;
    }


GLuint GPU::link_program(const string& fs_in){
        // Two kernel families need the ESSL 3.00 compiler: the q4 nibble split
        // ("&15", opt-in via Q4_SHIFT) and the RWKV bit-cast f32 state kernels
        // (uintBitsToFloat / floatBitsToUint, opt-in via RWKV_GPU).  Both gate on
        // the driver actually offering ES3 (g_es3_ok); the q4 one additionally on
        // Q4_SHIFT being wanted.  es3ify adds the #version line and the vertex
        // stage is upgraded to match.
        const bool q4_es3 = g_q4_shift_ok && fs_in.find("&15") != string::npos;
        const bool bc_es3 = g_es3_ok && (fs_in.find("uintBitsToFloat") != string::npos ||
                                         fs_in.find("floatBitsToUint") != string::npos);
        // The 2-bit ternary matvec (dtype 7) decodes with shifts and masks, so
        // it is ESSL 3.00 only and says so with a marker.  There is no ES2
        // fallback in the kernel: a driver without ES3 runs the same container
        // through T3_EXPAND=1 (the q8 expansion) instead.
        const bool t3_es3 = g_es3_ok && fs_in.find("//T3") != string::npos;
        if (!g_es3_ok && fs_in.find("//T3") != string::npos) {
                cerr << "this kernel is the 2-bit ternary matvec and needs an ESSL 3.00 "
                        "context;\n  run the container with T3_EXPAND=1 to use the q8 "
                        "expansion (4x the RAM), or give the driver ES3\n";
                exit(2);
        }
        const bool needs_es3 = q4_es3 || bc_es3 || t3_es3;
        const string fs = needs_es3 ? es3ify(fs_in, false) : fs_in;
        string vs = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }";
        // The vertex shader must match the fragment stage's language when that
        // stage was rewritten above; every other program stays ESSL 1.00.
        if (needs_es3) vs = es3ify(vs, true);
        // The vertex shader is the same text for every program, so compile it
        // once per language and attach the shared object: 19 links otherwise
        // run the same full-screen-quad vertex shader through the compiler 19
        // times.  (The ESSL 3.00 form exists only for the q4 shift kernels.)
        static GLuint vs_obj[2] = {0, 0};
        GLuint& vs_use = vs_obj[needs_es3 ? 1 : 0];
        if (!vs_use) vs_use = compile_shader(GL_VERTEX_SHADER, vs);
        const bool tm = shader_time_on();
        const double t0 = tm ? shader_time_ms() : 0.0;
        GLuint p = glCreateProgram();
        glAttachShader(p, vs_use);
        glAttachShader(p, compile_shader(GL_FRAGMENT_SHADER, fs));
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[8192]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
            cerr << "program link failed:\n" << log << "\n"; exit(1);
        }
        if (tm) {
            static int count = 0;
            static double total = 0.0;
            const double ms = shader_time_ms() - t0;
            ++count; total += ms;
            fprintf(stderr, "SHADER_TIME #%02d at %8.1f ms  link %6.1f ms  src %5zu B  h=%08zx  key=%s  cum %7.1f ms\n",
                    count, shader_time_ms(), ms, fs.size(),
                    (size_t)std::hash<string>{}(fs), g_prog_key.c_str(), total);
        }
        // The progress bar's side count.  Links are lazy: they happen inside the
        // first forward pass, so the only honest figure is how many have been
        // made (the set is whatever that pass touches, which no table states).
        // On the board this is a second and a half of the first token's prefill,
        // and it was invisible -- the bar sat at 0% while it ran.
        g_progress.side_bump("shaders");
        return p;
    }
