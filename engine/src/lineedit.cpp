#include "lineedit.h"

#include <unistd.h>
#include <termios.h>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

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

// A UTF-8 continuation byte belongs to the character before it; deleting one
// byte of a multi-byte character would otherwise leave a broken sequence on
// screen and in the prompt sent to the model.
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

void redraw(const std::string& prompt, const std::string& buf, size_t cur){
        // \r + prompt + buffer, clear to end of line, then put the cursor back.
        std::string s = "\r" + prompt + buf + "\x1b[K";
        const size_t tail = buf.size() - cur;
        if (tail) s += "\x1b[" + std::to_string(tail) + "D";
        fwrite(s.data(), 1, s.size(), stdout);
        fflush(stdout);
}

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

        fputs(prompt, stdout);
        fflush(stdout);

        bool done = false, eof = false, interrupted = false;
        while (!done) {
                unsigned char c;
                if (!read_byte(c)) { eof = true; break; }
                if (c == '\r' || c == '\n') { done = true; break; }
                if (c == 3) { interrupted = true; break; }        // Ctrl-C
                if (c == 4) {                                     // Ctrl-D
                        if (buf.empty()) { eof = true; break; }
                        if (cur < buf.size()) {
                                size_t e = next_char(buf, cur);
                                buf.erase(cur, e - cur);
                                redraw(prompt, buf, cur);
                        }
                        continue;
                }
                if (c == 127 || c == 8) {                         // Backspace
                        if (cur > 0) {
                                size_t p = prev_char(buf, cur);
                                buf.erase(p, cur - p);
                                cur = p;
                                redraw(prompt, buf, cur);
                        }
                        continue;
                }
                if (c == 1) { cur = 0; redraw(prompt, buf, cur); continue; }            // Ctrl-A
                if (c == 5) { cur = buf.size(); redraw(prompt, buf, cur); continue; }   // Ctrl-E
                if (c == 11) { buf.erase(cur); redraw(prompt, buf, cur); continue; }    // Ctrl-K
                if (c == 21) { buf.erase(0, cur); cur = 0; redraw(prompt, buf, cur); continue; }  // Ctrl-U
                if (c == 23) {                                                          // Ctrl-W
                        size_t e = cur, p = cur;
                        while (p > 0 && buf[p - 1] == ' ') --p;
                        while (p > 0 && buf[p - 1] != ' ') p = prev_char(buf, p);
                        buf.erase(p, e - p);
                        cur = p;
                        redraw(prompt, buf, cur);
                        continue;
                }
                if (c == 12) {                                                          // Ctrl-L
                        fputs("\x1b[H\x1b[2J", stdout);
                        redraw(prompt, buf, cur);
                        continue;
                }
                if (c == 27) {                                                          // ESC [ X
                        unsigned char a, b;
                        if (!read_byte(a)) { eof = true; break; }
                        if (a != '[' && a != 'O') continue;   // Alt-<key>: ignore
                        if (!read_byte(b)) { eof = true; break; }
                        if (b == 'A') {                                                 // Up
                                if (g_history.empty()) continue;
                                if (hist_pos < 0) { saved = buf; hist_pos = (int)g_history.size() - 1; }
                                else if (hist_pos > 0) --hist_pos;
                                buf = g_history[hist_pos];
                                cur = buf.size();
                                redraw(prompt, buf, cur);
                        } else if (b == 'B') {                                          // Down
                                if (hist_pos < 0) continue;
                                if (hist_pos + 1 < (int)g_history.size()) {
                                        ++hist_pos;
                                        buf = g_history[hist_pos];
                                } else {
                                        hist_pos = -1;
                                        buf = saved;
                                }
                                cur = buf.size();
                                redraw(prompt, buf, cur);
                        } else if (b == 'C') {                                          // Right
                                if (cur < buf.size()) { cur = next_char(buf, cur); redraw(prompt, buf, cur); }
                        } else if (b == 'D') {                                          // Left
                                if (cur > 0) { cur = prev_char(buf, cur); redraw(prompt, buf, cur); }
                        } else if (b == 'H') {                                          // Home
                                cur = 0; redraw(prompt, buf, cur);
                        } else if (b == 'F') {                                          // End
                                cur = buf.size(); redraw(prompt, buf, cur);
                        } else if (b == '3') {                                          // Delete
                                unsigned char tilde;
                                if (read_byte(tilde) && tilde == '~' && cur < buf.size()) {
                                        size_t e = next_char(buf, cur);
                                        buf.erase(cur, e - cur);
                                        redraw(prompt, buf, cur);
                                }
                        }
                        continue;
                }
                if (c < 32) continue;                                                   // other controls
                buf.insert(cur, 1, (char)c);
                ++cur;
                // Echo just the new character and what follows it, so a long
                // line does not flicker; multi-byte UTF-8 arrives byte by byte
                // and simply appends.
                fputc(c, stdout);
                if (cur < buf.size()) {
                        // mid-line insert: reprint the tail and step back
                        fwrite(buf.data() + cur, 1, buf.size() - cur, stdout);
                        fputs(("\x1b[" + std::to_string(buf.size() - cur) + "D").c_str(), stdout);
                }
                fflush(stdout);
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
