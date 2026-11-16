// The line editor's half of the no-GPU gate: the layout arithmetic, the paste
// sanitiser, and the editor itself driven through a pty the way a terminal drives
// it (a winsize, the bytes a terminal sends, and the frame it puts around a
// paste).  No GPU, no model, and no tty of the caller's.
//
//   engine$ ./test_lineedit          # part of `make check`
#include <pty.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "lineedit.h"
#include "lineedit_layout.h"

namespace {

int g_fail = 0;
bool g_no_pty = false;

std::string show(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == 0x1b) o += "\\e";
        else if (c < 32 || c == 127) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            o += b;
        } else {
            o.push_back((char)c);
        }
    }
    return o;
}

void eq(const char* what, const std::string& got, const std::string& want) {
    if (got == want) return;
    ++g_fail;
    std::fprintf(stderr, "FAIL %s\n   got  \"%s\"\n   want \"%s\"\n", what,
                 show(got).c_str(), show(want).c_str());
}

void eqn(const char* what, long got, long want) {
    if (got == want) return;
    ++g_fail;
    std::fprintf(stderr, "FAIL %s: %ld, want %ld\n", what, got, want);
}

void pos_is(const char* what, const std::string& buf, size_t at, int prompt_w, int cols,
            int row, int col) {
    const lineedit::Pos p = lineedit::pos_of(buf, at, prompt_w, cols);
    if (p.row == row && p.col == col) return;
    ++g_fail;
    std::fprintf(stderr, "FAIL %s: pos (%d,%d), want (%d,%d)\n", what, p.row, p.col, row, col);
}

// ---------------------------------------------------------------- layout ----

void test_layout() {
    pos_is("ascii", "hello", 3, 0, 80, 0, 3);
    pos_is("prompt offset", "hello", 5, 6, 80, 0, 11);
    pos_is("wrap", "0123456789ab", 12, 0, 10, 1, 2);
    // Exactly filling a row leaves the cursor in its last cell: one more column
    // than the width, which the draw turns into cols - 1.
    pos_is("pending wrap", "0123456789", 10, 0, 10, 0, 10);
    pos_is("cjk", "你好", 6, 0, 80, 0, 4);
    pos_is("cjk at the margin", "123456789你", 12, 0, 10, 1, 2);
    pos_is("emoji", "\xF0\x9F\x98\x80", 4, 0, 80, 0, 2);
    pos_is("newline", "ab\ncd", 3, 0, 80, 1, 0);
    pos_is("newline end", "ab\ncd", 5, 0, 80, 1, 2);
    pos_is("trailing newline", "ab\n", 3, 0, 80, 1, 0);
    pos_is("tab stop", "\tx", 1, 0, 80, 0, 8);

    const std::string long_line(200, 'x');
    const std::vector<lineedit::Row> rows = lineedit::rows_of(long_line, 6, 80);
    eqn("rows of a wrapped line", (long)rows.size(), 3);
    eqn("second row starts after the prompt's row", (long)rows[1].begin, 74);
    eqn("third row start", (long)rows[2].begin, 154);

    const std::string two = "aa\nbbb";
    eqn("row_target past the last row", (long)lineedit::row_target(two, 0, 80, 2, 0),
        (long)std::string::npos);
    eqn("row_target clamps to the row end", (long)lineedit::row_target(two, 0, 80, 0, 5), 2);
    eqn("row_target mid row", (long)lineedit::row_target(two, 0, 80, 1, 1), 4);
    eqn("row_target row start", (long)lineedit::row_target(two, 0, 80, 1, 0), 3);
    // The first row's prompt must not shift the column a vertical move carries:
    // the third character of "bbb" is the third of "aaa", not the prompt.
    eqn("row_target is row-relative", (long)lineedit::row_target("aaa\nbbb", 6, 80, 0, 3), 3);
    eqn("row_target clamps behind a prompt",
        (long)lineedit::row_target("aa\nbbb", 6, 80, 0, 5), 2);

    eqn("line_start", (long)lineedit::line_start(two, 5), 3);
    eqn("line_end", (long)lineedit::line_end(two, 1), 2);
    eqn("line_start at 0", (long)lineedit::line_start(two, 0), 0);
    eqn("line_end at the end", (long)lineedit::line_end(two, two.size()), (long)two.size());
}

// ----------------------------------------------------------------- paste ----

void test_paste() {
    eq("crlf is one break", lineedit::paste_normalise("a\r\nb"), "a\nb");
    eq("lone cr is a break", lineedit::paste_normalise("a\rb"), "a\nb");
    eq("other control bytes are dropped", lineedit::paste_normalise("a\x01\x02" "b"), "ab");
    // Only the escape bytes go; a pasted colour code leaves its text behind
    // instead of being interpreted, because the buffer is never executed.
    eq("escape bytes dropped", lineedit::paste_normalise("\x1b[31mred\x1b[0m"), "[31mred[0m");
    eq("tab kept", lineedit::paste_normalise("a\tb"), "a\tb");
    eq("cjk kept", lineedit::paste_normalise("你好"), "你好");
}

// ---------------------------------------------------------------- editor ----

// Runs the editor against a pty: a fresh terminal per case, so `g_history` is the
// only state carried between them.
bool run_editor(const std::string& keys, std::string& out) {
    int master = -1, slave = -1;
    if (openpty(&master, &slave, nullptr, nullptr, nullptr) != 0) {
        g_no_pty = true;
        return false;
    }
    struct winsize ws {};
    ws.ws_col = 80;
    ws.ws_row = 24;
    ioctl(slave, TIOCSWINSZ, &ws);
    const int saved_in = dup(STDIN_FILENO);
    const int saved_out = dup(STDOUT_FILENO);
    if (saved_in < 0 || saved_out < 0 || dup2(slave, STDIN_FILENO) < 0 ||
        dup2(slave, STDOUT_FILENO) < 0) {
        close(master);
        close(slave);
        return false;
    }
    close(slave);

    // The keys cannot be queued before the editor starts: it enters raw mode with
    // TCSAFLUSH, which discards input that arrived earlier.  So a reader thread
    // waits for the first prompt on the master, sends the keys then, and keeps
    // draining the drawings so a large paste cannot fill the pty.
    std::thread pump([&]() {
        bool sent = false;
        std::string seen;
        char tmp[512];
        while (true) {
            const ssize_t n = read(master, tmp, sizeof tmp);
            if (n <= 0) break;
            if (sent) continue;
            seen.append(tmp, (size_t)n);
            if (seen.find("User: ") == std::string::npos) continue;
            size_t off = 0;
            while (off < keys.size()) {
                const ssize_t w = write(master, keys.data() + off, keys.size() - off);
                if (w <= 0) break;
                off += (size_t)w;
            }
            sent = true;
        }
    });

    const bool ok = line_edit_read("User: ", out);

    dup2(saved_in, STDIN_FILENO);
    dup2(saved_out, STDOUT_FILENO);
    close(saved_in);
    close(saved_out);
    pump.join();
    close(master);
    return ok;
}

void test_editor() {
    const std::string esc = "\x1b";
    const std::string up = esc + "[A";
    const std::string open = esc + "[200~";
    const std::string close = esc + "[201~";

    auto run = [&](const char* name, const std::string& keys, const std::string& want) {
        std::string got;
        if (!run_editor(keys, got)) {
            ++g_fail;
            std::fprintf(stderr, "FAIL %s: the editor returned no line\n", name);
            return;
        }
        eq(name, got, want);
    };

    run("typed line", "hi\r", "hi");
    // A pasted block is content: its newlines are part of the prompt, and only
    // the Enter after the frame submits.  The trailing newline of a copied line
    // is kept, not trimmed -- the screen shows what the model receives.
    run("pasted block keeps its newlines", open + "a\r\nb\nc" + close + "\r", "a\nb\nc");
    run("ctrl-v enter inserts a newline", "x" + std::string(1, '\x16') + "\r" + "y\r", "x\ny");
    run("alt-enter inserts a newline", "x" + esc + "\r" + "y\r", "x\ny");

    run("up moves inside the buffer", open + "aaa\nbbb" + close + up + "Z\r", "aaaZ\nbbb");

    // Above the first row Up is history again, which needs an entry to reach.
    run("seed history", "old\r", "old");
    run("up at the first row reaches history", open + "aaa\nbbb" + close + up + up + "\r", "old");

    // 200 columns with a 6-column prompt: rows of 74, 80, 46.  Up twice keeps the
    // column (46) instead of jumping to the end of the shorter first row.
    const std::string wide(200, 'x');
    run("up keeps the column across a wrapped line", open + wide + close + up + up + "Z\r",
        std::string(46, 'x') + "Z" + std::string(154, 'x'));

    // A shorter row above clamps the column: the cursor lands at the end of
    // "0123456789", not 40 columns into nothing.
    run("up clamps to a shorter row", open + "0123456789\n" + wide + close + up + up + up + "Z\r",
        "0123456789Z\n" + wide);
}

}  // namespace

int main() {
    {
        int master = -1, slave = -1;
        if (openpty(&master, &slave, nullptr, nullptr, nullptr) != 0) {
            g_no_pty = true;
            std::fprintf(stderr, "test_lineedit: no pty (%s), skipping the editor cases\n",
                         std::strerror(errno));
        } else {
            close(master);
            close(slave);
        }
    }
    test_layout();
    test_paste();
    if (!g_no_pty) test_editor();
    if (g_fail) {
        std::fprintf(stderr, "test_lineedit: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("test_lineedit: layout, paste and the pty-driven editor pass\n");
    return 0;
}
