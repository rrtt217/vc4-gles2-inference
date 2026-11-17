#pragma once
#include <cstdio>

// The switch registry.
//
// Every environment switch the engine reads has exactly one row here, and the
// row is the only place that knows its *name*, its older spellings, its type,
// its closed set of values, whether it is permanent or a probe, and which axis
// of the engine it moves.  `--dump-switches` prints the table, `docs_check.sh`
// compares it with the source and with docs/test-hooks.md in both directions,
// and common.cpp's `env_*` helpers read through it.
//
// Why this exists (2026-11-17): the switches had grown to 126 across 55
// first-token prefixes, some of which name a module (`Q4_*`, `RWKV_*`) and some
// an axis (`ACT_*`, `KV_*`, `RES_*`), and nothing anywhere recorded a polarity,
// a value set or an owner.  The consequences were concrete: `KV_PACK` (bits) and
// `KV_FMT` (codec) were two halves of one axis with an undefined interaction,
// `ACT8` (width), `ACT_F32` (placement) and `ACT_FMT` (format) all moved the
// activation stream, and 26 `Q4_*` arms from one concluded campaign were still
// in the acceptance path.  A switch is not free: each one is a branch in a
// numerics path that someone has to re-verify, so a switch that is not written
// down with an owner and a status rots into a trap.
//
// The rules:
//   - a permanent switch keeps its name for ever and is A/B'd by its values;
//   - a probe is expected to die: it must name its campaign in the note, and
//     `docs_check.sh` fails if a retired name is read by the source again;
//   - a rename keeps the old spelling working through kRenames, with a line on
//     stderr, so a command in an old note does something correct rather than
//     something silently different.
enum class SwKind { Flag, Int, Num, Str };

enum class SwStatus {
    Permanent,  // part of the documented configuration surface
    Probe,      // a diagnostic or a pricing arm; candidates for retirement
    Retired,    // documented as gone; the source must not read it
};

// The axis, not the module: two switches that move the same thing share an axis
// whatever file reads them.  A closed set, so `--dump-switches` can be read.
enum class SwAxis {
    Dump,     // writes the engine's numbers out for an offline oracle
    Probe,    // prices a pass (draws, syncs, phase timings)
    Place,    // which path runs (GPU vs CPU, per block)
    Act,      // the activation streams' precision
    Att,      // the attention block's own fixed-point grids (score, sum)
    Kv,       // the KV cache's encoding
    Res,      // the residual stream's format
    Ffn,      // the FFN hidden stream / kernel shape
    Q4,       // the 4-bit weight kernels
    Rwkv,     // RWKV-7 (a different layer loop)
    Prefill,  // prefill batching
    Load,     // container loading, memory, upload
    Tok,      // tokenizer
    Gen,      // sampling, penalties, stop policy
    Prefix,   // the disk prefix cache
    Env,      // set by the environment or the driver, not by the engine
    Retired,  // documented as removed
};

struct SwitchDef {
    const char* name;      // canonical spelling
    const char* aliases;   // comma separated older spellings, "" for none
    SwKind kind;
    const char* values;    // "a|b|c" for a closed set, "" when free form
    SwStatus status;
    SwAxis axis;
    const char* note;      // one line: what it moves, for --dump-switches
};

// A legacy spelling and what it means now.  `value` empty means "whatever the
// legacy variable held"; `becomes` empty means "keep that value".  `name` must
// be a canonical row above, and the rewrite only happens when the canonical
// name is not set itself (the new spelling wins).
struct SwRename {
    const char* legacy;
    const char* value;
    const char* name;
    const char* becomes;
};

int sw_count();
const SwitchDef& sw_at(int i);
const SwitchDef* sw_find(const char* name);
const char* sw_axis_name(SwAxis a);
const char* sw_kind_name(SwKind k);
const char* sw_status_name(SwStatus s);

// All reads go through these.  `sw_str` returns the resolved value (the
// canonical name's, or a legacy spelling's already rewritten), or nullptr when
// the switch is unset; `sw_enum` is the same and documents the intent to read a
// closed set (validation happens for every set switch at first read, so a typo
// fails the run instead of falling back to a default).
bool sw_isset(const char* name);
bool sw_flag(const char* name);
int sw_int(const char* name, int def);
double sw_num(const char* name, double def);
const char* sw_str(const char* name);
const char* sw_enum(const char* name);

// Resolve the environment once (idempotent; called by every accessor).
void sw_init();
// The resolved table plus what is set: `--dump-switches`.
void sw_dump(FILE* out);
