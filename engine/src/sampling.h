#pragma once

#include "common.h"
#include "gpu.h"

#include <random>

struct SamplingSettings {
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int no_repeat_ngram = 0;
    float min_p = 0.0f;
};

extern SamplingSettings g_sampling;

bool sampling_penalties_active();
void apply_sampling_penalties(vector<float>& logits, const vector<int>& recent);
// A draw plus the probability it had.  `logprob` is measured in the *kept*
// distribution (temperature, top-k, top-p applied and renormalised), which is
// what a client means by "the probability of this token".
struct SampledToken {
    int id = 0;
    float logprob = 0.0f;
};

int sample_token(const vector<float>& logits, float temp, float top_p, int top_k, mt19937& rng);
SampledToken sample_token_report(const vector<float>& logits, float temp, float top_p,
                                 int top_k, mt19937& rng);
SampledToken argmax_token_report(const vector<float>& logits);
int argmax_token(const vector<float>& logits);
int sample_candidate_result(const CandidateResult& r, float temp, float top_p, int top_k, mt19937& rng);
