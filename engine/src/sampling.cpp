#include "sampling.h"

#include <algorithm>
#include <cmath>

SamplingSettings g_sampling;

bool sampling_penalties_active() {
    return g_sampling.repetition_penalty != 1.0f ||
           g_sampling.frequency_penalty != 0.0f ||
           g_sampling.presence_penalty != 0.0f ||
           g_sampling.no_repeat_ngram > 1 ||
           g_sampling.min_p > 0.0f;
}

void apply_sampling_penalties(vector<float>& logits, const vector<int>& recent) {
    if (recent.empty() && g_sampling.min_p <= 0.0f) return;
    unordered_map<int, int> counts;
    for (int t : recent) ++counts[t];
    if (g_sampling.repetition_penalty != 1.0f) {
        for (const auto& kv : counts) {
            int t = kv.first;
            if (t < 0 || t >= (int)logits.size()) continue;
            if (logits[t] > 0) logits[t] /= g_sampling.repetition_penalty;
            else logits[t] *= g_sampling.repetition_penalty;
        }
    }
    if (g_sampling.frequency_penalty != 0.0f || g_sampling.presence_penalty != 0.0f) {
        for (const auto& kv : counts) {
            int t = kv.first;
            if (t < 0 || t >= (int)logits.size()) continue;
            logits[t] -= g_sampling.frequency_penalty * kv.second;
            logits[t] -= g_sampling.presence_penalty;
        }
    }
    const int n = g_sampling.no_repeat_ngram;
    if (n > 1 && (int)recent.size() >= n) {
        vector<int> prefix(recent.end() - (n - 1), recent.end());
        for (size_t i = 0; i + n < recent.size(); ++i) {
            bool match = true;
            for (int j = 0; j < n - 1; ++j) {
                if (recent[i + j] != prefix[j]) { match = false; break; }
            }
            if (match) {
                int banned = recent[i + n - 1];
                if (banned >= 0 && banned < (int)logits.size()) logits[banned] = -1e30f;
            }
        }
    }
    if (g_sampling.min_p > 0.0f) {
        float mx = -1e30f;
        for (float v : logits) mx = max(mx, v);
        float cutoff = mx + logf(g_sampling.min_p);
        for (float& v : logits) if (v < cutoff) v = -1e30f;
    }
}

int sample_token(const vector<float>& logits, float temp, float top_p, mt19937& rng) {
    vector<pair<float, int>> p(logits.size());
    float mx = -1e30f;
    for (size_t i = 0; i < logits.size(); ++i) mx = max(mx, logits[i] / temp);
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        float e = expf(logits[i] / temp - mx);
        p[i] = {e, (int)i}; sum += e;
    }
    for (auto& x : p) x.first /= sum;
    sort(p.begin(), p.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    float c = 0; int cut = 0;
    for (; cut < (int)p.size(); ++cut) { c += p[cut].first; if (c >= top_p) { ++cut; break; } }
    float r = uniform_real_distribution<float>(0, 1)(rng) * c;
    float acc = 0;
    for (int i = 0; i < cut; ++i) { acc += p[i].first; if (r <= acc) return p[i].second; }
    return p[cut - 1].second;
}

int argmax_token(const vector<float>& logits) {
    return (int)(max_element(logits.begin(), logits.end()) - logits.begin());
}

int sample_candidate_result(const CandidateResult& r, float temp, float top_p, mt19937& rng) {
    if (r.cand.empty()) return 2;
    vector<Candidate> cand = r.cand;
    if (r.strict) {
        for (auto& c : cand) c.value = expf(c.value / temp - r.logsumexp);
    } else {
        float mx = -1e30f;
        for (auto& c : cand) mx = max(mx, c.value / temp);
        float sum = 0;
        for (auto& c : cand) { c.value = expf(c.value / temp - mx); sum += c.value; }
        for (auto& c : cand) c.value /= sum;
    }
    sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.value > b.value; });
    float cdf = 0; int cut = 0;
    for (; cut < (int)cand.size(); ++cut) { cdf += cand[cut].value; if (cdf >= top_p) { ++cut; break; } }
    float x = uniform_real_distribution<float>(0, 1)(rng) * cdf;
    float acc = 0;
    for (int i = 0; i < cut; ++i) { acc += cand[i].value; if (x <= acc) return cand[i].token; }
    return cand[cut - 1].token;
}

// -----------------------------------------------------------------------------
// Tool calls: minimal JSON extraction, built-in handlers, and a safe arithmetic
// parser.  The model only emits <tool_call>{...}</tool_call>; execution and the
// <tool_response> round-trip are done here.
// -----------------------------------------------------------------------------
