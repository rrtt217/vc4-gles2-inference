// A small line editor for the chat REPL: arrows, home/end, history, the usual
// control keys.  No dependency, because the target has no guaranteed readline
// and the engine must still run from a pipe.
//
// When stdin is not a terminal this is a plain getline(): the worker protocol,
// the regression scripts and `echo hi | engine_mm3 --chat` must not see escape
// sequences or a missing echo.
//
// Two rules the implementation follows, both learned by measuring:
//   * Screen positions are counted in terminal *columns*, never bytes.  A CJK
//     character is two columns and three bytes; counting bytes moved the cursor
//     the wrong way over every Chinese character.
//   * Every escape sequence is parsed to its final byte, parameters and all.
//     Reading exactly two bytes after ESC left ";5C" (Ctrl-Right) and "~" (Home
//     in tmux) inside the line, where they were sent to the model.
//
// Column widths come from the table below rather than wcwidth(), so the editor
// behaves the same under LANG=C: with a C locale wcwidth() reports every CJK
// character as unprintable, and every library we tested (readline, libedit)
// then either splits or drops the character.
#include "lineedit.h"

#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <deque>
#include <string>

namespace {

// 32 lines is enough to recall a conversation's questions without holding the
// answers; each entry is a copy, the REPL is not allocation sensitive.
std::deque<std::string> g_history;

bool read_byte(unsigned char& c){
        while (true) {
                ssize_t n = ::read(STDIN_FILENO, &c, 1);
                if (n == 1) return true;
                if (n < 0 && errno == EINTR) continue;
                return false;
        }
}

struct Rune {
        char32_t cp;
        int len;
};

Rune decode(const std::string& s, size_t i){
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) return {c, 1};
        int len;
        char32_t cp;
        if ((c & 0xE0) == 0xC0)      { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        else return {0xFFFD, 1};
        if (i + len > s.size()) return {0xFFFD, 1};
        for (int k = 1; k < len; ++k) {
                const unsigned char cc = (unsigned char)s[i + k];
                if ((cc & 0xC0) != 0x80) return {0xFFFD, 1};
                cp = (cp << 6) | (cc & 0x3F);
        }
        return {cp, len};
}

// Column width of one code point, without a locale.  Same ranges wcwidth()
// would use; combining marks are zero and East Asian wide/fullwidth are two.
int rune_width(char32_t cp){
        if (cp == 0) return 0;
        if (cp < 32 || (cp >= 0x7F && cp < 0xA0)) return 0;
        if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x0483 && cp <= 0x0489) ||
            (cp >= 0x0591 && cp <= 0x05BD) || (cp >= 0x0610 && cp <= 0x061A) ||
            (cp >= 0x064B && cp <= 0x065F) || (cp >= 0x0E31 && cp <= 0x0E3A) ||
            (cp >= 0x1AB0 && cp <= 0x1AFF) || (cp >= 0x1DC0 && cp <= 0x1DFF) ||
            (cp >= 0x20D0 && cp <= 0x20FF) || (cp >= 0xFE00 && cp <= 0xFE0F) ||
            (cp >= 0xFE20 && cp <= 0xFE2F))
                return 0;
        if ((cp >= 0x1100 && cp <= 0x115F) || cp == 0x2329 || cp == 0x232A ||
            (cp >= 0x2E80 && cp <= 0x303E) || (cp >= 0x3041 && cp <= 0x33FF) ||
            (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
            (cp >= 0xA000 && cp <= 0xA4CF) || (cp >= 0xA960 && cp <= 0xA97F) ||
            (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0xF900 && cp <= 0xFAFF) ||
            (cp >= 0xFE10 && cp <= 0xFE19) || (cp >= 0xFE30 && cp <= 0xFE6F) ||
            (cp >= 0xFF00 && cp <= 0xFF60) || (cp >= 0xFFE0 && cp <= 0xFFE6) ||
            (cp >= 0x1F300 && cp <= 0x1F64F) || (cp >= 0x1F900 && cp <= 0x1F9FF) ||
            (cp >= 0x20000 && cp <= 0x3FFFD))
                return 2;
        return 1;
}

// Columns in s[0, end).
int width_upto(const std::string& s, size_t end){
        int w = 0;
        for (size_t i = 0; i < end && i < s.size(); ) {
                const Rune r = decode(s, i);
                w += rune_width(r.cp);
                i += r.len;
        }
        return w;
}

int text_width(const std::string& s){ return width_upto(s, s.size()); }

// A UTF-8 continuation byte belongs to the character before it; deleting one
// byte of a multi-byte character would otherwise leave a broken sequence in the
// prompt sent to the model.
size_t prev_char(const std::string& s, size_t i){
        if (i == 0) return 0;
        --i;
        while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) --i;
        return i;
}

size_t next_char(const std::string& s, size_t i){
        if (i >= s.size()) return s.size();
        ++i;
        while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) ++i;
        return i;
}

bool is_space(char32_t cp){ return cp == ' ' || cp == '\t'; }

size_t word_left(const std::string& s, size_t i){
        while (i > 0) {
                const size_t p = prev_char(s, i);
                if (!is_space(decode(s, p).cp)) break;
                i = p;
        }
        while (i > 0) {
                const size_t p = prev_char(s, i);
                if (is_space(decode(s, p).cp)) break;
                i = p;
        }
        return i;
}

size_t word_right(const std::string& s, size_t i){
        while (i < s.size() && is_space(decode(s, i).cp)) i = next_char(s, i);
        while (i < s.size() && !is_space(decode(s, i).cp)) i = next_char(s, i);
        while (i < s.size() && is_space(decode(s, i).cp)) i = next_char(s, i);
        return i;
}

// True while the tail of s is an incomplete UTF-8 sequence, so a character
// arriving byte by byte is drawn once instead of once per byte.
bool utf8_pending(const std::string& s){
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

int term_cols(){
        struct winsize w {};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
        return 80;
}

enum class Act {
        None, Left, Right, WordLeft, WordRight, Up, Down, Home, End, Delete,
};

// The final byte of a CSI sequence plus its parameters: `ESC[1;5C` is
// Ctrl-Right, `ESC[1~` is Home on terminals that do not send `ESC[H`.
Act csi_action(const std::string& p, unsigned char final){
        int field[8] = {0};
        int n = 0, v = 0;
        bool digits = false;
        for (char ch : p) {
                if (ch >= '0' && ch <= '9') { v = v * 10 + (ch - '0'); digits = true; }
                else if (ch == ';') { if (n < 8) field[n++] = digits ? v : 0; v = 0; digits = false; }
                else return Act::None;              // private or intermediate byte
        }
        if (n < 8 && digits) field[n++] = v;
        // A lone number before C/D is a repeat count, not a modifier; only a
        // second field is the xterm modifier (1 + shift/alt/ctrl bits).
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

// Everything the redraw needs to know about the screen, kept together so the
// arithmetic stays in one place.
struct Editor {
        std::string prompt;
        int prompt_w;
        int cols;
        int last_pos;              // cursor column position after the last draw

        explicit Editor(std::string p)
            : prompt(std::move(p)), prompt_w(text_width(prompt)), cols(term_cols()), last_pos(0) {}

        // Row the cursor sat on after printing `pos` columns.  When the text
        // exactly fills a row the terminal leaves the cursor in the last cell
        // with the wrap pending, so the row is (pos - 1) / cols, not pos / cols.
        int row_of(int pos) const { return pos <= 0 ? 0 : (pos - 1) / cols; }

        void draw(const std::string& buf, size_t cur){
                cols = term_cols();                    // survives a resize
                const int up = row_of(last_pos);
                const int total = prompt_w + text_width(buf);
                const int tgt = prompt_w + width_upto(buf, cur);
                const int visual = (total > 0 && total % cols == 0) ? total - 1 : total;
                std::string s;
                if (up > 0) s += "\x1b[" + std::to_string(up) + "A";
                s += "\r\x1b[J";                       // first row, erase the block
                s += prompt;
                s += buf;
                const int back = visual - tgt;
                if (back > 0) s += "\x1b[" + std::to_string(back) + "D";
                else if (back < 0) s += "\x1b[" + std::to_string(-back) + "C";
                fwrite(s.data(), 1, s.size(), stdout);
                fflush(stdout);
                last_pos = tgt;
        }
};

}  // namespace

bool line_edit_is_tty(){
        return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
}

bool line_edit_read(const char* prompt, std::string& out){
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

        std::string buf;
        size_t cur = 0;
        int hist_pos = -1;         // -1 = editing a fresh line
        std::string saved;         // the fresh line, kept while browsing history

        Editor ed(prompt);
        ed.draw(buf, cur);

        auto hist_prev = [&]() {
                if (g_history.empty()) return;
                if (hist_pos < 0) { saved = buf; hist_pos = (int)g_history.size() - 1; }
                else if (hist_pos > 0) --hist_pos;
                buf = g_history[hist_pos];
                cur = buf.size();
                ed.draw(buf, cur);
        };
        auto hist_next = [&]() {
                if (hist_pos < 0) return;
                if (hist_pos + 1 < (int)g_history.size()) {
                        ++hist_pos;
                        buf = g_history[hist_pos];
                } else {
                        hist_pos = -1;
                        buf = saved;
                }
                cur = buf.size();
                ed.draw(buf, cur);
        };
        auto act = [&](Act a) {
                switch (a) {
                        case Act::Left:
                                if (cur > 0) { cur = prev_char(buf, cur); ed.draw(buf, cur); }
                                break;
                        case Act::Right:
                                if (cur < buf.size()) { cur = next_char(buf, cur); ed.draw(buf, cur); }
                                break;
                        case Act::WordLeft: {
                                const size_t p = word_left(buf, cur);
                                if (p != cur) { cur = p; ed.draw(buf, cur); }
                                break;
                        }
                        case Act::WordRight: {
                                const size_t p = word_right(buf, cur);
                                if (p != cur) { cur = p; ed.draw(buf, cur); }
                                break;
                        }
                        case Act::Up: hist_prev(); break;
                        case Act::Down: hist_next(); break;
                        case Act::Home:
                                if (cur != 0) { cur = 0; ed.draw(buf, cur); }
                                break;
                        case Act::End:
                                if (cur != buf.size()) { cur = buf.size(); ed.draw(buf, cur); }
                                break;
                        case Act::Delete:
                                if (cur < buf.size()) {
                                        const size_t e = next_char(buf, cur);
                                        buf.erase(cur, e - cur);
                                        ed.draw(buf, cur);
                                }
                                break;
                        case Act::None: break;
                }
        };

        bool eof = false, interrupted = false;
        while (true) {
                unsigned char c;
                if (!read_byte(c)) { eof = true; break; }
                if (c == '\r' || c == '\n') break;
                if (c == 3) { interrupted = true; break; }        // Ctrl-C
                if (c == 4) {                                     // Ctrl-D
                        if (buf.empty()) { eof = true; break; }
                        if (cur < buf.size()) {
                                const size_t e = next_char(buf, cur);
                                buf.erase(cur, e - cur);
                                ed.draw(buf, cur);
                        }
                        continue;
                }
                if (c == 127 || c == 8) {                         // Backspace
                        if (cur > 0) {
                                const size_t p = prev_char(buf, cur);
                                buf.erase(p, cur - p);
                                cur = p;
                                ed.draw(buf, cur);
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
                                        // Parameter and intermediate bytes run
                                        // until the final byte 0x40..0x7E.
                                        while (true) {
                                                if (!read_byte(final)) { ok = false; break; }
                                                if (final >= 0x40 && final <= 0x7E) break;
                                                if (params.size() < 24) params.push_back((char)final);
                                        }
                                } else if (!read_byte(final)) {
                                        ok = false;
                                }
                                if (!ok) { eof = true; break; }
                                act(csi_action(params, final));
                        } else if (a == 'b') {
                                act(Act::WordLeft);
                        } else if (a == 'f') {
                                act(Act::WordRight);
                        } else if (a == 127 || a == 8) {          // Alt-Backspace
                                const size_t p = word_left(buf, cur);
                                if (p != cur) { buf.erase(p, cur - p); cur = p; ed.draw(buf, cur); }
                        }
                        continue;
                }
                if (c == 1) { cur = 0; ed.draw(buf, cur); continue; }            // Ctrl-A
                if (c == 2) { act(Act::Left); continue; }                        // Ctrl-B
                if (c == 5) { cur = buf.size(); ed.draw(buf, cur); continue; }   // Ctrl-E
                if (c == 6) { act(Act::Right); continue; }                       // Ctrl-F
                if (c == 11) { buf.erase(cur); ed.draw(buf, cur); continue; }    // Ctrl-K
                if (c == 12) {                                                   // Ctrl-L
                        fputs("\x1b[H\x1b[2J", stdout);
                        ed.last_pos = 0;
                        ed.draw(buf, cur);
                        continue;
                }
                if (c == 14) { hist_next(); continue; }                          // Ctrl-N
                if (c == 16) { hist_prev(); continue; }                          // Ctrl-P
                if (c == 21) { buf.erase(0, cur); cur = 0; ed.draw(buf, cur); continue; }  // Ctrl-U
                if (c == 23) {                                                   // Ctrl-W
                        size_t p = cur;
                        while (p > 0 && buf[p - 1] == ' ') --p;
                        while (p > 0 && buf[p - 1] != ' ') p = prev_char(buf, p);
                        buf.erase(p, cur - p);
                        cur = p;
                        ed.draw(buf, cur);
                        continue;
                }
                if (c < 32) continue;                                            // other controls
                buf.insert(cur, 1, (char)c);
                ++cur;
                // A multi-byte character arrives one byte at a time; draw when
                // it is whole, so the width used for the cursor is the real one.
                if (!utf8_pending(buf)) ed.draw(buf, cur);
        }

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
