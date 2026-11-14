// `--ppl`: perplexity of a text file, one sliding window at a time.
//
// Moved out of `main()` (2026-11-14, P2-1 of the 2026-11-13 audit).  It is the one
// number the gate pins (270.3170 on Xe, 271.2855 on the Pi Zero), so it is also the
// best-instrumented thing to extract: if the move changes anything, the gate says
// so immediately.
//
// Two rules live here and must not be "simplified":
//   - the text goes in *with* BOS (this path feeds raw text, no template);
//   - a window warms up on `warm` tokens before scoring `stride` of them, because
//     a window that starts mid-text is cold and a cold window is not a perplexity
//     (docs/performance.md, 2026-11-12).

#include "common.h"
#include "model.h"
#include "session.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int run_ppl(const std::string& ppl_path, Model& model, GPU& gpu, Tokenizer& tokenizer) {
if (!ppl_path.empty()) {
    string text;
    if (!read_text_file(ppl_path, text)) { cerr << "cannot read " << ppl_path << "\n"; return 1; }
    // BOS, because this path feeds raw text to the model and no template is
    // involved.  It used to pass false/false, which is what chat *without*
    // a BOS policy means -- and for LFM2.5 (whose template supplies
    // <|startoftext|>) it measured a prompt the model never sees: PPL 85.4
    // without the BOS against 14.7 with it, nats 4.45 against 2.69.  It is
    // also what llama.cpp's llama-perplexity does (add_special from the
    // GGUF's add_bos_token).
    auto tokens = tokenizer.encode(text, /*bos=*/true, /*dummy_prefix=*/false);
    if ((int)tokens.size() < 4) { cerr << "not enough tokens\n"; return 1; }
    // Score every token of the file, in windows of at most SEQ tokens.
    //
    // A window that starts mid-text is cold: it has no left context, and on
    // a real document that costs ~1.8 nats/token (measured 2026-11-12 on
    // this file: 3.80 nats with the document before it, 5.61 starting at
    // token 512 -- f32 numpy, so it is the model and not the engine).  So a
    // sum of cold windows is not a perplexity anyone else reports, and the
    // 2026-11-11 note that called it a window-by-window degradation was
    // reading llama.cpp's warm chunks against our cold ones.
    //
    // Each window therefore warms up on the tokens before the ones it
    // scores: it feeds `span` tokens and scores the last `stride` of them,
    // which is the sliding-window protocol (`--ppl-stride` in llama.cpp,
    // `--stride` in HuggingFace's evaluate).  Default stride is SEQ/2 so
    // every scored token has at least SEQ/2 tokens of context and each
    // window pays half its passes for context.  PPL_STRIDE=0 restores the
    // old cold-window behaviour (stride SEQ-1, no warm-up), which is the
    // only way to reproduce the pre-2026-11-12 long-file numbers.
    const int span = model.seq;
    int stride = span / 2 > 0 ? span / 2 : 1;
    if (const char* ps = getenv("PPL_STRIDE")) stride = atoi(ps);
    if (stride < 0) stride = 0;
    if (stride == 0) stride = span - 1;             // cold windows
    if (stride > span - 1) stride = span - 1;
    const int warm = span - stride;                 // context tokens per window
    FILE* ppl_dump = nullptr;
    if (const char* dp = getenv("PPL_DUMP")) ppl_dump = fopen(dp, "w");
    double nll = 0.0;
    long cnt = 0;
    int windows = 0;
    const long n = (long)tokens.size();
    long next = 1;                                  // first token still to score
    while (next < n) {
        if (g_stop) break;
        const long a = windows == 0 ? 0 : next - warm;   // first token fed
        const long len = std::min((long)span, n - a);
        if (len < 2 || a < 0) break;
        Session win(model, gpu);
        for (long j = 0; j + 1 < len; ++j) {
            if (g_stop) break;
            const long t = a + j + 1;               // token this position predicts
            const bool score = t >= next;
            vector<float> lg = win.step(tokens[a + j], (int)j, score);
            if (!score) continue;
            if ((int)lg.size() <= tokens[t]) break;
            double mx = -1e30;
            for (float v : lg) mx = max((double)v, mx);
            double se = 0.0;
            for (float v : lg) se += exp((double)v - mx);
            double logp = ((double)lg[tokens[t]] - mx) - log(se);
            nll += -logp;
            // PPL_DUMP=path writes one line per scored token.  The mean is
            // not enough to see what a gap is made of: a handful of
            // catastrophic positions and a uniform 2% error both raise PPL.
            if (FILE* pf = ppl_dump) fprintf(pf, "%ld %d %.6f\n", t, tokens[t], -logp);
            ++cnt;
            next = t + 1;
        }
        ++windows;
    }
    if (!cnt) { cerr << "no tokens scored\n"; return 1; }
    printf("PPL %.4f  nll=%.4f  tokens=%ld  windows=%d  span=%d  stride=%d\n",
           exp(nll / cnt), nll / cnt, cnt, windows, span, stride);
    return 0;
}
}
