// GPU counters for probes, through GL_AMD_performance_monitor.
//
// The board exposes no timer query: `eglinfo -s` on the Pi lists
// GL_AMD_performance_monitor and nothing with "timer" or "disjoint" in it, so
// "how much of this draw's wall time did the QPU spend, and on what" cannot be
// answered with glGetQueryObjectui64v.  It can be answered with the V3D counter
// group, and the answer changes how a probe's own number reads -- draw_probe's
// null draw is 15% GPU and 85% submit+finish, so its "0.6 ms a draw" is not one
// quantity.  A probe that prints a per-draw time should be able to say which.
//
// Usage:
//
//     PerfMon pm;
//     if (pm.missing()) return 0;            // prints why, once
//     pm.pick({"QPU-total-idle-clk-cycles", "waiting-TMU", ...});   // substring
//     pm.begin();
//     ...draws...
//     pm.end_and_print("MV 1216x768", draws);
//
// begin() also takes a wall-clock start and end_and_print() divides the counters
// by the QPU count and the clock, so the table lines up a GPU time against the
// wall time the probe already prints.
//
// Two limits the caller must keep in mind, both of them from the driver:
//
//   * at most 16 counters at once (GL_MAX_PERFMON_ACTIVE), and the board has 30
//     in one group -- pick the ones the question needs, do not try to take all.
//   * the counters are whole-GPU totals, summed over the 4 QPUs of a Pi Zero's
//     V3D.  A per-draw cycle count is therefore cycles/(QPUS), and the ms column
//     assumes all QPUs run at the same clock.
//
// No counter is enabled unless the caller asks: a probe's numbers must not
// change because it linked this header.  PERFCNT is the switch that asks.
#pragma once

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <string>
#include <vector>

#ifndef PERFCNT_QPUS
// The counters are summed over the V3D's QPUs, and the divisor is measured, not
// assumed: on this board a 1216x1 null draw's fragment-shading count comes out
// 12x its wall time at 300 MHz, so the total covers 12 QPU-equivalents of
// clock.  Getting this wrong is not a small error -- 4 instead of 12 made a
// matvec report 291% GPU time, which is the tell.
#define PERFCNT_QPUS 12
#endif
#ifndef PERFCNT_CLOCK_MHZ
#define PERFCNT_CLOCK_MHZ 300  // `vcgencmd get_config gpu_freq` on the board
#endif

// GL_AMD_performance_monitor has no entry points in libGLESv2 on every driver:
// the spec allows them through eglGetProcAddress, and Mesa's vc4 does that.
struct PerfMon {
    // Counters are addressed by id; the name is kept for the print.
    struct Counter {
        GLuint id = 0;
        std::string name;
        std::string tag;  // short label the caller gave
    };

    PFNGLGENPERFMONITORSAMDPROC gen_ = nullptr;
    PFNGLSELECTPERFMONITORCOUNTERSAMDPROC select_ = nullptr;
    PFNGLBEGINPERFMONITORAMDPROC begin_ = nullptr;
    PFNGLENDPERFMONITORAMDPROC end_ = nullptr;
    PFNGLGETPERFMONITORCOUNTERDATAAMDPROC data_ = nullptr;

    GLuint monitor_ = 0;
    GLuint group_ = 0;
    GLint max_active_ = 0;
    std::vector<Counter> picked_;
    bool armed_ = false;
    bool missing_ = true;
    const char* why_ = "";

    struct Sample {
        std::string tag;
        std::vector<GLuint> value;  // parallel to picked_
    };

    // One arm of a table: the counters plus the wall time the probe already
    // measured for the same draws, so the two are printed side by side.
    struct Row {
        Sample sample;
        double wall_ms = 0;  // per draw, as the probe prints it
    };

    PerfMon() {
        if (!load()) return;
        GLint ngroups = 0;
        auto groups = (PFNGLGETPERFMONITORGROUPSAMDPROC)egl_symbol("glGetPerfMonitorGroupsAMD");
        if (!groups) {
            why_ = "glGetPerfMonitorGroupsAMD is not resolvable";
            return;
        }
        groups(&ngroups, 0, nullptr);
        if (ngroups <= 0) {
            why_ = "the driver reports no counter groups";
            return;
        }
        // The board has exactly one group ("V3D counters"); a driver with more
        // would need a group argument, so say so instead of guessing.
        if (ngroups > 1) {
            printf("PERFCNT: %d groups, using the first (extend this if it matters)\n", ngroups);
        }
        GLuint gs[8] = {0};
        groups(&ngroups, 8, gs);
        group_ = gs[0];
        auto get_counters = (PFNGLGETPERFMONITORCOUNTERSAMDPROC)egl_symbol("glGetPerfMonitorCountersAMD");
        GLint ncounters = 0;
        get_counters(group_, &ncounters, &max_active_, 0, nullptr);
        missing_ = false;
    }

    bool missing() const {
        if (missing_) printf("PERFCNT: unavailable -- %s\n", why_ ? why_ : "no AMD_performance_monitor");
        return missing_;
    }

    // Every counter in the group, in driver order, for a "what is here" listing.
    void list(const char* group_filter = nullptr) {
        if (missing_) return;
        auto get_counters = (PFNGLGETPERFMONITORCOUNTERSAMDPROC)egl_symbol("glGetPerfMonitorCountersAMD");
        auto cstr = (PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC)egl_symbol("glGetPerfMonitorCounterStringAMD");
        GLint n = 0, maxact = 0;
        get_counters(group_, &n, &maxact, 0, nullptr);
        std::vector<GLuint> ids((size_t)n);
        get_counters(group_, &n, &maxact, n, ids.data());
        printf("PERFCNT: %d counters, %d active at once\n", n, maxact);
        for (GLint i = 0; i < n; ++i) {
            char name[128] = {0};
            GLsizei len = 0;
            cstr(group_, ids[(size_t)i], sizeof(name) - 1, &len, name);
            if (group_filter && !strstr(name, group_filter)) continue;
            printf("  %2d  %s\n", i, name);
        }
    }

    // Substring match on the driver's counter names, in the order given, so a
    // report's column order is the probe's and not the driver's.  A name that
    // matches nothing is printed and skipped rather than silently absent.
    bool pick(std::initializer_list<const char*> wanted) {
        if (missing_) return false;
        auto get_counters = (PFNGLGETPERFMONITORCOUNTERSAMDPROC)egl_symbol("glGetPerfMonitorCountersAMD");
        auto cstr = (PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC)egl_symbol("glGetPerfMonitorCounterStringAMD");
        GLint n = 0, maxact = 0;
        get_counters(group_, &n, &maxact, 0, nullptr);
        std::vector<GLuint> ids((size_t)n);
        get_counters(group_, &n, &maxact, n, ids.data());
        std::vector<Counter> all;
        for (GLint i = 0; i < n; ++i) {
            Counter c;
            c.id = ids[(size_t)i];
            char name[128] = {0};
            GLsizei len = 0;
            cstr(group_, c.id, sizeof(name) - 1, &len, name);
            c.name = name;
            all.push_back(c);
        }
        picked_.clear();
        for (const char* want : wanted) {
            const Counter* hit = nullptr;
            for (const Counter& c : all)
                if (c.name.find(want) != std::string::npos) { hit = &c; break; }
            if (!hit) {
                printf("PERFCNT: no counter matches \"%s\"\n", want);
                continue;
            }
            if ((GLint)picked_.size() >= max_active_) {
                printf("PERFCNT: %d is the limit, dropping \"%s\"\n", max_active_, want);
                continue;
            }
            Counter c = *hit;
            // The tag is the driver's own name, not the substring: two
            // counters can contain "cache-miss", and the table keys off the
            // tag.  pick("waiting-TMU") still selects by substring.
            c.tag = c.name;
            picked_.push_back(c);
        }
        return !picked_.empty();
    }

    void begin() {
        if (missing_ || picked_.empty()) return;
        if (!monitor_) gen_(1, &monitor_);
        std::vector<GLuint> sel;
        for (const Counter& c : picked_) sel.push_back(c.id);
        select_(monitor_, GL_TRUE, group_, (GLsizei)sel.size(), sel.data());
        wall_start_ = now_ms();
        begin_(monitor_);
        armed_ = true;
    }

    // Reads one sample.  Called by end_and_print(); exposed for a probe that
    // wants to batch several arms and print one table.
    Sample end() {
        Sample s;
        if (!armed_) return s;
        end_(monitor_);
        armed_ = false;
        GLuint avail = 0;
        GLint written = 0;
        for (int i = 0; i < 200000 && !avail; ++i)
            data_(monitor_, GL_PERFMON_RESULT_AVAILABLE_AMD, 4, &avail, &written);
        GLuint size = 0;
        data_(monitor_, GL_PERFMON_RESULT_SIZE_AMD, 4, &size, &written);
        if (!avail || size == 0) {
            printf("PERFCNT: the result never became available\n");
            return s;
        }
        // Fresh allocation every time: reusing a vector for the result buffer
        // was the shape that segfaulted on the board (2026-11-21), before the
        // size was known to be 4 words a counter.
        std::vector<GLuint> res(size / 4 + 4);
        data_(monitor_, GL_PERFMON_RESULT_AMD, size, res.data(), &written);
        // The record layout on Mesa/vc4 is four words a counter:
        // {group, counter, value, 0}.  Walking it three words at a time reads
        // the group id as a counter id and reports a plausible, wrong number;
        // `raw.cpp` in the probe directory dumped the words that show it.
        s.value.assign(picked_.size(), 0);
        for (size_t i = 0; i + 3 < res.size(); i += 4) {
            const GLuint id = res[i + 1];
            for (size_t k = 0; k < picked_.size(); ++k)
                if (picked_[k].id == id) s.value[k] += res[i + 2];
        }
        return s;
    }

    // The one-liner: begin() must have been called, `draws` is what the caller
    // issued inside the region so the columns are per draw.
    Sample end_and_print(const char* label, int draws) {
        (void)draws;
        Sample s = end();
        s.tag = label;
        if (s.value.empty()) return s;
        printf("PERFCNT %-22s", label);
        for (size_t k = 0; k < picked_.size(); ++k)
            printf(" %s=%llu", picked_[k].tag.c_str(), (unsigned long long)s.value[k]);
        printf("\n");
        return s;
    }

    // A table with the derived columns a probe actually wants to read: cycles
    // per QPU, milliseconds of QPU time, and the GPU share of the wall time.
    // `draws` is how many draws each row measured, so the columns are per draw.
    // `draws` is the number of draws EACH row's region covered (the same total
    // the caller passed to end_and_print, i.e. iters * draws_per_call).  The
    // columns are then per draw, and Row::wall_ms is the caller's per-call
    // wall time, so a row that issued several draws per call must pass
    // wall_ms_per_call / draws_per_call to line the two up.
    void table_rows(const std::vector<Row>& rows, int draws) {
        if (rows.empty() || picked_.empty()) return;
        const double d = draws > 0 ? (double)draws : 1.0;
        printf("%-24s %10s %10s %10s %10s %6s\n", "arm", "wall ms/d", "frag ms/d",
               "idle ms/d", "exec ms/d", "gpu%");
        for (const Row& r : rows) {
            long long frag = 0, exec = 0, idle = 0;
            for (size_t k = 0; k < picked_.size(); ++k) {
                const std::string& t = picked_[k].tag;
                if (t.find("clk-cycles-fragment-shading") != std::string::npos) frag = (long long)r.sample.value[k];
                else if (t.find("executing-valid-instr") != std::string::npos) exec = (long long)r.sample.value[k];
                else if (t.find("idle-clk-cycles") != std::string::npos) idle = (long long)r.sample.value[k];
            }
            const double to_ms = 1.0 / (qpus_ * PERFCNT_CLOCK_MHZ * 1000.0) / d;
            printf("%-24s %10.4f %10.4f %10.4f %10.4f %5.0f\n", r.sample.tag.c_str(),
                   r.wall_ms, (double)frag * to_ms, (double)idle * to_ms,
                   (double)exec * to_ms,
                   r.wall_ms > 0 ? 100.0 * (double)frag * to_ms / r.wall_ms : 0.0);
        }
    }

    GLint max_active() const { return max_active_; }

    // The picked counters, in pick() order.  A probe that wants one column
    // back (rather than printing the raw line) indexes these against
    // Sample::value, which is parallel to them.
    size_t picked_count() const { return picked_.size(); }
    const std::string& picked_name(size_t i) const { return picked_[i].tag; }

    // The divisor the table uses.  A fixed 12 is right on this board, but a
    // wrong constant reads as a plausible percentage, so a probe that has a
    // cheap wall-clock reference for the same region can hand it over and get
    // a measured ratio instead.  `qpu_share` is what the region's counters
    // mean over the caller's own wall time; calibrate() sets qpus_ so the two
    // agree for a region the caller knows is GPU-bound.
    double qpu_share(long long cycles, double wall_ms) const {
        if (wall_ms <= 0) return 0.0;
        return (double)cycles / (qpus_ * PERFCNT_CLOCK_MHZ * 1000.0) / wall_ms;
    }
    void set_qpus(double n) { if (n > 0) qpus_ = n; }
    double qpus() const { return qpus_; }

private:
    double qpus_ = PERFCNT_QPUS;

public:

private:
    double wall_start_ = 0;

    static double now_ms() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
    }

    static void* egl_symbol(const char* name) { return (void*)eglGetProcAddress(name); }

    bool load() {
        gen_ = (PFNGLGENPERFMONITORSAMDPROC)egl_symbol("glGenPerfMonitorsAMD");
        select_ = (PFNGLSELECTPERFMONITORCOUNTERSAMDPROC)egl_symbol("glSelectPerfMonitorCountersAMD");
        begin_ = (PFNGLBEGINPERFMONITORAMDPROC)egl_symbol("glBeginPerfMonitorAMD");
        end_ = (PFNGLENDPERFMONITORAMDPROC)egl_symbol("glEndPerfMonitorAMD");
        data_ = (PFNGLGETPERFMONITORCOUNTERDATAAMDPROC)egl_symbol("glGetPerfMonitorCounterDataAMD");
        if (!gen_ || !select_ || !begin_ || !end_ || !data_) {
            why_ = "glBegin/EndPerfMonitorAMD is not resolvable (no GL_AMD_performance_monitor)";
            return false;
        }
        return true;
    }
};

// PERFCNT=1 asks for the counters; anything else leaves a probe's numbers exactly
// as they were.  Unset is the default on every machine.
static inline bool perfcnt_enabled() {
    const char* v = getenv("PERFCNT");
    return v && v[0] && strcmp(v, "0") != 0;
}
