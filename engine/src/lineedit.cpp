// A small line editor for the chat REPL: arrows, home/end, history, the usual
// control keys, and a buffer that may hold more than one line.  No dependency,
// because the target has no guaranteed readline and the engine must still run
// from a pipe.
//
// When stdin is not a terminal this is a plain getline(): the worker protocol,
// the regression scripts and `echo hi | engine_mm3 --chat` must not see escape
// sequences or a missing echo.
//
// Rules this follows, all of them learned by measuring:
//   * Screen positions are counted in terminal *columns*, never bytes.  A CJK
//     character is two columns and three bytes; counting bytes moved the cursor
//     the wrong way over every Chinese character.
//   * Every escape sequence is parsed to its final byte, parameters and all.
//     Reading exactly two bytes after ESC left ";5C" (Ctrl-Right) and "~" (Home
//     in tmux) inside the line, where they were sent to the model.
//   * The cursor is placed by absolute position.  Moving back N columns with
//     'ESC[nD' stops at column 0 of the current row, so it cannot reach an
//     earlier row of a wrapped line -- which is why the draw used to be correct
//     only while the cursor stayed on the last screen row.
//
// The buffer may contain '\n': a paste, Ctrl-V Enter (readline's quoted-insert) or
// Alt-Enter puts one there, and Enter still means "submit the whole buffer".  Where
// a byte sits on the screen, and what Up/Down mean, is lineedit_layout.h's business.
//
// There is no clipboard here, and there must not be: the terminal owns the
// clipboard and delivers its contents as input bytes.  The editor understands the
// frame the terminal puts around a paste (ESC[200~ ... ESC[201~) because it asks
// for one with ESC[?2004h; *without* that frame a paste is byte-for-byte typing,
// and is not guessed at.  A heuristic ("this input arrived too fast to be typed")
// was considered and rejected: ssh and the terminal split a large paste across
// reads, and one wrong guess sends half a paragraph to the model as a message.
#include "lineedit.h"

#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

#include "lineedit_layout.h"

namespace {

// 32 lines is enough to recall a conversation's questions without holding the
// answers; each entry is a copy, the REPL is not allocation sensitive.
std::deque<std::string> g_history;

bool read_byte(unsigned char& c) {
    while (true) {
        const ssize_t n = ::read(STDIN_FILENO, &c, 1);
        if (n == 1) return true;
        if (n < 0 && errno == EINTR) continue;
        return false;
    }
}

int term_cols() {
    struct winsize w {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
    return 80;
}

int term_rows() {
    struct winsize w {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_row > 0) return w.ws_row;
    return 24;
}

// A line break is a word boundary for the cursor motions: Alt-b from the start of
// a line should walk back into the previous one, as in an editor.
bool is_space(char32_t cp) { return cp == ' ' || cp == '\t' || cp == '\n'; }

size_t word_left(const std::string& s, size_t i) {
    while (i > 0) {
        const size_t p = lineedit::prev_char(s, i);
        if (!is_space(lineedit::decode_rune(s, p).cp)) break;
        i = p;
    }
    while (i > 0) {
        const size_t p = lineedit::prev_char(s, i);
        if (is_space(lineedit::decode_rune(s, p).cp)) break;
        i = p;
    }
    return i;
}

size_t word_right(const std::string& s, size_t i) {
    while (i < s.size() && is_space(lineedit::decode_rune(s, i).cp))
        i = lineedit::next_char(s, i);
    while (i < s.size() && !is_space(lineedit::decode_rune(s, i).cp))
        i = lineedit::next_char(s, i);
    while (i < s.size() && is_space(lineedit::decode_rune(s, i).cp))
        i = lineedit::next_char(s, i);
    return i;
}

// True while the tail of s is an incomplete UTF-8 sequence, so a character
// arriving byte by byte is drawn once instead of once per byte.
bool utf8_pending(const std::string& s) {
    size_t i = s.size();
    int seen = 0;
    while (i > 0 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) {
        if (++seen > 3) return false;
        --i;
    }
    if (i == 0) return false;
    const unsigned char lead = (unsigned char)s[i - 1];
    int want = 1;
    if ((lead & 0xE0) == 0xC0) want = 2;
    else if ((lead & 0xF0) == 0xE0) want = 3;
    else if ((lead & 0xF8) == 0xF0) want = 4;
    return seen + 1 < want;
}

enum class Act {
    None, Left, Right, WordLeft, WordRight, Up, Down, Home, End, Delete,
};

// The final byte of a CSI sequence plus its parameters: `ESC[1;5C` is Ctrl-Right,
// `ESC[1~` is Home on terminals that do not send `ESC[H`.
Act csi_action(const std::string& p, unsigned char final) {
    int field[8] = {0};
    int n = 0, v = 0;
    bool digits = false;
    for (char ch : p) {
        if (ch >= '0' && ch <= '9') { v = v * 10 + (ch - '0'); digits = true; }
        else if (ch == ';') { if (n < 8) field[n++] = digits ? v : 0; v = 0; digits = false; }
        else return Act::None;              // private or intermediate byte
    }
    if (n < 8 && digits) field[n++] = v;
    // A lone number before C/D is a repeat count, not a modifier; only a second
    // field is the xterm modifier (1 + shift/alt/ctrl bits).
    const int mod = n >= 2 ? field[1] : 1;
    const bool word = ((mod - 1) & (2 | 4)) != 0;
    switch (final) {
        case 'A': return Act::Up;
        case 'B': return Act::Down;
        case 'C': return word ? Act::WordRight : Act::Right;
        case 'D': return word ? Act::WordLeft : Act::Left;
        case 'H': return Act::Home;
        case 'F': return Act::End;
        case '~':
            if (n == 0) return Act::None;
            if (field[0] == 1 || field[0] == 7) return Act::Home;
            if (field[0] == 4 || field[0] == 8) return Act::End;
            if (field[0] == 3) return Act::Delete;
            return Act::None;
        default: return Act::None;
    }
}

// The body of a paste: the terminal framed the clipboard with ESC[200~ ... ESC[201~
// because this editor asked for the frame (ESC[?2004h).  Everything between the
// markers is content, including bytes that would be keys outside a paste -- a
// Ctrl-C in a copied log is a byte, not a signal.  Only EOF ends this early.
//
// The end marker can arrive split across reads, and its first bytes can be real
// pasted text ("ESC[" in a log), so the longest suffix that still matches the
// marker is held back instead of being appended.
bool read_paste(std::string& out) {
    static const char kEnd[] = "\x1b[201~";
    const size_t end_len = sizeof(kEnd) - 1;
    const size_t kMax = 1u << 20;      // a paste is input, not a document
    size_t match = 0;
    while (true) {
        unsigned char c;
        if (!read_byte(c)) return false;
        if (match > 0) {
            if (c == (unsigned char)kEnd[match]) {
                if (++match == end_len) return true;
                continue;
            }
            out.append(kEnd, match);
            match = 0;
        }
        if (c == (unsigned char)kEnd[0]) { match = 1; continue; }
        if (out.size() < kMax) out.push_back((char)c);
    }
}

// Everything the redraw needs to know about the screen, kept together so the
// arithmetic stays in one place.
//
// The block is drawn by printing `prompt + buf` and letting the terminal wrap it,
// then the cursor is put where it belongs by absolute position: move to the target
// row (relative to the end of the text, which is where printing left the cursor),
// carriage return, then move right by the column.  `last_row` is the cursor's row
// *within the block*, which is all a relative move needs after a scroll.
struct Editor {
    std::string prompt;
    int prompt_w;
    int cols;
    int last_row;              // cursor row inside the block after the last draw

    explicit Editor(std::string p)
        : prompt(std::move(p)), prompt_w(lineedit::text_width(prompt)),
          cols(term_cols()), last_row(0) {}

    void draw(const std::string& buf, size_t cur) {
        cols = term_cols();
        const int screen_rows = term_rows();
        const std::vector<lineedit::Row> laid = lineedit::rows_of(buf, prompt_w, cols);
        const int height = (int)laid.size();
        const lineedit::Pos at = lineedit::pos_of(buf, cur, prompt_w, cols);
        const lineedit::Pos end = lineedit::pos_of(buf, buf.size(), prompt_w, cols);

        std::string s;
        if (height >= screen_rows) {
            // Taller than the screen: 'ESC[nA' clamps at the top row, so anchor
            // the block at the top and clear instead of moving up.  A block that
            // fits always does so after such a scroll, which is why this is the
            // only case that needs the clear.
            s += "\x1b[H\x1b[2J";
            last_row = 0;
        } else {
            if (last_row > 0) s += "\x1b[" + std::to_string(last_row) + "A";
            s += "\r";
            s += "\x1b[J";                       // first row, erase the block
        }
        s += prompt;
        s += buf;
        const int drow = at.row - end.row;       // negative: the cursor is above
        if (drow < 0) s += "\x1b[" + std::to_string(-drow) + "A";
        else if (drow > 0) s += "\x1b[" + std::to_string(drow) + "B";
        s += "\r";
        int col = at.col;
        if (col >= cols) col = cols - 1;         // pending wrap: last cell
        if (col > 0) s += "\x1b[" + std::to_string(col) + "C";
        fwrite(s.data(), 1, s.size(), stdout);
        fflush(stdout);
        last_row = at.row;
    }
};

}  // namespace

bool line_edit_is_tty() {
    return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
}

bool line_edit_read(const char* prompt, std::string& out) {
    out.clear();
    if (!line_edit_is_tty()) {
        // Pipe or file: exactly the old behaviour, prompt and all.
        fputs(prompt, stdout);
        fflush(stdout);
        int c;
        bool any = false;
        while ((c = getchar()) != EOF) {
            any = true;
            if (c == '\n') break;
            out.push_back((char)c);
        }
        if (!any && out.empty()) return false;
        while (!out.empty() && (out.back() == '\r')) out.pop_back();
        return true;
    }

    struct termios old_t, raw;
    if (tcgetattr(STDIN_FILENO, &old_t) != 0) return false;
    raw = old_t;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);   // we handle Ctrl-C ourselves
    raw.c_iflag &= ~(IXON | ICRNL);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    // Ask the terminal to frame pastes (ESC[200~ ... ESC[201~).  A terminal that
    // does not know the sequence ignores it; one that knows wraps what it is about
    // to send, which is the only in-band way to tell a pasted newline from a
    // pressed Enter.  Turned off again on the way out, or the next program would
    // inherit the markers.
    fputs("\x1b[?2004h", stdout);
    fflush(stdout);

    std::string buf;
    size_t cur = 0;
    int want_col = -1;         // column a run of vertical moves aims for; -1 = derive
    int hist_pos = -1;         // -1 = editing a fresh line
    std::string saved;         // the fresh line, kept while browsing history

    Editor ed(prompt);
    ed.draw(buf, cur);

    auto draw = [&]() { ed.draw(buf, cur); };

    auto insert = [&](const std::string& t) {
        if (t.empty()) return;
        buf.insert(cur, t);
        cur += t.size();
        want_col = -1;
        draw();
    };

    auto hist_prev = [&]() {
        if (g_history.empty()) return;
        if (hist_pos < 0) { saved = buf; hist_pos = (int)g_history.size() - 1; }
        else if (hist_pos > 0) --hist_pos;
        buf = g_history[(size_t)hist_pos];
        cur = buf.size();
        want_col = -1;
        draw();
    };
    auto hist_next = [&]() {
        if (hist_pos < 0) return;
        if (hist_pos + 1 < (int)g_history.size()) {
            ++hist_pos;
            buf = g_history[(size_t)hist_pos];
        } else {
            hist_pos = -1;
            buf = saved;
        }
        cur = buf.size();
        want_col = -1;
        draw();
    };

    // One screen row up or down *inside the buffer*; false when there is no row in
    // that direction, which is when the caller falls back to history.
    auto move_vert = [&](int dir) -> bool {
        const int cols = term_cols();
        const lineedit::Pos p = lineedit::pos_of(buf, cur, ed.prompt_w, cols);
        const std::vector<lineedit::Row> rows = lineedit::rows_of(buf, ed.prompt_w, cols);
        const int target = p.row + dir;
        if (target < 0 || (size_t)target >= rows.size()) return false;
        // The column is measured from the row's own start: a vertical move means
        // "same character offset", and row 0's prompt must not shift it.
        if (want_col < 0) want_col = p.col - rows[(size_t)p.row].col0;
        const size_t at = lineedit::row_target(buf, ed.prompt_w, cols, target, want_col);
        if (at == std::string::npos) return false;
        cur = at;
        draw();
        return true;
    };

    auto act = [&](Act a) {
        switch (a) {
            case Act::Left:
                if (cur > 0) { cur = lineedit::prev_char(buf, cur); want_col = -1; draw(); }
                break;
            case Act::Right:
                if (cur < buf.size()) { cur = lineedit::next_char(buf, cur); want_col = -1; draw(); }
                break;
            case Act::WordLeft: {
                const size_t p = word_left(buf, cur);
                if (p != cur) { cur = p; want_col = -1; draw(); }
                break;
            }
            case Act::WordRight: {
                const size_t p = word_right(buf, cur);
                if (p != cur) { cur = p; want_col = -1; draw(); }
                break;
            }
            case Act::Up:
                // Inside a multi-line buffer Up moves the cursor; only above the
                // first row does it reach for history.  This is also what keeps
                // Up from replacing a wrapped long line with an old question.
                if (!move_vert(-1)) hist_prev();
                break;
            case Act::Down:
                if (!move_vert(1)) hist_next();
                break;
            case Act::Home: {
                const size_t p = lineedit::line_start(buf, cur);
                if (p != cur) { cur = p; want_col = -1; draw(); }
                break;
            }
            case Act::End: {
                const size_t p = lineedit::line_end(buf, cur);
                if (p != cur) { cur = p; want_col = -1; draw(); }
                break;
            }
            case Act::Delete:
                if (cur < buf.size()) {
                    const size_t e = lineedit::next_char(buf, cur);
                    buf.erase(cur, e - cur);
                    want_col = -1;
                    draw();
                }
                break;
            case Act::None: break;
        }
    };

    bool eof = false, interrupted = false;
    while (true) {
        unsigned char c;
        if (!read_byte(c)) { eof = true; break; }
        // Enter submits; a newline in the buffer can only come from a paste, from
        // Ctrl-V Enter or from Alt-Enter.
        if (c == '\r' || c == '\n') break;
        if (c == 3) { interrupted = true; break; }        // Ctrl-C
        if (c == 4) {                                     // Ctrl-D
            if (buf.empty()) { eof = true; break; }
            if (cur < buf.size()) act(Act::Delete);
            continue;
        }
        if (c == 127 || c == 8) {                         // Backspace
            if (cur > 0) {
                const size_t p = lineedit::prev_char(buf, cur);
                buf.erase(p, cur - p);
                cur = p;
                want_col = -1;
                draw();
            }
            continue;
        }
        if (c == 22) {                                    // Ctrl-V: next byte is text
            unsigned char q;
            if (!read_byte(q)) { eof = true; break; }
            if (q == '\r' || q == '\n') insert("\n");
            else if (q >= 32 && q != 127) {
                // A quoted control byte is dropped: an ESC in the buffer would be
                // executed by the terminal the next time the block is redrawn.
                buf.insert(cur, 1, (char)q);
                ++cur;
                want_col = -1;
                if (!utf8_pending(buf)) draw();
            }
            continue;
        }
        if (c == 27) {                                    // ESC ...
            unsigned char a;
            if (!read_byte(a)) { eof = true; break; }
            if (a == '[' || a == 'O') {
                unsigned char final = 0;
                std::string params;
                bool ok = true;
                if (a == '[') {
                    // Parameter and intermediate bytes run until the final byte
                    // 0x40..0x7E.
                    while (true) {
                        if (!read_byte(final)) { ok = false; break; }
                        if (final >= 0x40 && final <= 0x7E) break;
                        if (params.size() < 24) params.push_back((char)final);
                    }
                } else if (!read_byte(final)) {
                    ok = false;
                }
                if (!ok) { eof = true; break; }
                if (final == '~' && params == "200") {   // the terminal framed a paste
                    std::string frameted;
                    if (!read_paste(frameted)) { eof = true; break; }
                    insert(lineedit::paste_normalise(frameted));
                    continue;
                }
                if (final == '~' && params == "201") continue;   // stray end marker
                act(csi_action(params, final));
            } else if (a == 'b') {
                act(Act::WordLeft);
            } else if (a == 'f') {
                act(Act::WordRight);
            } else if (a == '\r' || a == '\n') {         // Alt-Enter
                insert("\n");
            } else if (a == 127 || a == 8) {             // Alt-Backspace
                // One word back, but never across the start of the line: killing a
                // word is an edit of *this* line.
                size_t p = word_left(buf, cur);
                const size_t floor = lineedit::line_start(buf, cur);
                if (p < floor) p = floor;
                if (p != cur) { buf.erase(p, cur - p); cur = p; want_col = -1; draw(); }
            }
            continue;
        }
        if (c == 1) { act(Act::Home); continue; }                 // Ctrl-A
        if (c == 2) { act(Act::Left); continue; }                 // Ctrl-B
        if (c == 5) { act(Act::End); continue; }                  // Ctrl-E
        if (c == 6) { act(Act::Right); continue; }                // Ctrl-F
        if (c == 11) {                                            // Ctrl-K
            const size_t e = lineedit::line_end(buf, cur);
            if (e != cur) { buf.erase(cur, e - cur); want_col = -1; draw(); }
            continue;
        }
        if (c == 12) {                                            // Ctrl-L
            fputs("\x1b[H\x1b[2J", stdout);
            ed.last_row = 0;
            draw();
            continue;
        }
        if (c == 14) { hist_next(); continue; }                   // Ctrl-N
        if (c == 16) { hist_prev(); continue; }                   // Ctrl-P
        if (c == 21) {                                            // Ctrl-U
            const size_t b = lineedit::line_start(buf, cur);
            if (b != cur) { buf.erase(b, cur - b); cur = b; want_col = -1; draw(); }
            continue;
        }
        if (c == 23) {                                            // Ctrl-W
            size_t p = word_left(buf, cur);
            const size_t floor = lineedit::line_start(buf, cur);
            if (p < floor) p = floor;
            if (p != cur) { buf.erase(p, cur - p); cur = p; want_col = -1; draw(); }
            continue;
        }
        if (c < 32) continue;                                     // other controls
        buf.insert(cur, 1, (char)c);
        ++cur;
        want_col = -1;
        // A multi-byte character arrives one byte at a time; draw when it is
        // whole, so the width used for the cursor is the real one.
        if (!utf8_pending(buf)) draw();
    }

    fputs("\x1b[?2004l", stdout);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_t);
    fputc('\n', stdout);
    fflush(stdout);

    if (eof) return false;
    if (interrupted) { out.clear(); return true; }
    if (!buf.empty()) {
        if (g_history.empty() || g_history.back() != buf) g_history.push_back(buf);
        if (g_history.size() > 32) g_history.pop_front();
    }
    out = buf;
    return true;
}
