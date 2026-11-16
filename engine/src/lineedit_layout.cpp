#include "lineedit_layout.h"

#include <algorithm>

namespace lineedit {

namespace {

// How far the cursor advances when this character is placed at column `col` of a
// row `cols` wide.  Tabs go to the terminal's 8-column stops; a tab with no stop
// left before the margin clamps to the last column, which is what xterm does.
int advance(char32_t cp, int col, int cols) {
    if (cp != '\t') return rune_width(cp);
    int next = (col / 8 + 1) * 8;
    if (next > cols - 1) next = cols - 1;
    if (next <= col) return 0;
    return next - col;
}

}  // namespace

Rune decode_rune(const std::string& s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) return {c, 1};
    int len;
    char32_t cp;
    if ((c & 0xE0) == 0xC0)      { len = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
    else return {0xFFFD, 1};
    if (i + (size_t)len > s.size()) return {0xFFFD, 1};
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = (unsigned char)s[i + (size_t)k];
        if ((cc & 0xC0) != 0x80) return {0xFFFD, 1};
        cp = (cp << 6) | (cc & 0x3F);
    }
    return {cp, len};
}

// Column width of one code point, without a locale.  Same ranges wcwidth() would
// use; combining marks are zero and East Asian wide/fullwidth are two.
int rune_width(char32_t cp) {
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

int text_width(const std::string& s) {
    int w = 0;
    for (size_t i = 0; i < s.size(); ) {
        const Rune r = decode_rune(s, i);
        w += rune_width(r.cp);
        i += (size_t)r.len;
    }
    return w;
}

// A UTF-8 continuation byte belongs to the character before it; deleting one byte
// of a multi-byte character would otherwise leave a broken sequence in the prompt
// sent to the model.
size_t prev_char(const std::string& s, size_t i) {
    if (i == 0) return 0;
    --i;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) --i;
    return i;
}

size_t next_char(const std::string& s, size_t i) {
    if (i >= s.size()) return s.size();
    ++i;
    while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) ++i;
    return i;
}

std::vector<Row> rows_of(const std::string& buf, int prompt_w, int cols) {
    if (cols <= 0) cols = 80;
    std::vector<Row> rows;
    Row r;
    r.begin = 0;
    r.col0 = prompt_w;
    int col = prompt_w;
    size_t i = 0;
    while (i < buf.size()) {
        const Rune u = decode_rune(buf, i);
        if (u.cp == '\n') {
            r.end = i;
            rows.push_back(r);
            i += (size_t)u.len;
            r.begin = i;
            r.col0 = 0;
            col = 0;
            continue;
        }
        const int w = advance(u.cp, col, cols);
        if (w > 0 && col + w > cols) {
            r.end = i;
            rows.push_back(r);
            r.begin = i;
            r.col0 = 0;
            col = 0;
            continue;   // the same character is placed again on the new row
        }
        col += w;
        i += (size_t)u.len;
    }
    r.end = buf.size();
    rows.push_back(r);
    return rows;
}

Pos pos_of(const std::string& buf, size_t at, int prompt_w, int cols) {
    if (cols <= 0) cols = 80;
    Pos p;
    p.row = 0;
    p.col = prompt_w;
    const size_t n = std::min(at, buf.size());
    for (size_t i = 0; i < n; ) {
        const Rune u = decode_rune(buf, i);
        if (u.cp == '\n') {
            ++p.row;
            p.col = 0;
            i += (size_t)u.len;
            continue;
        }
        const int w = advance(u.cp, p.col, cols);
        if (w > 0 && p.col + w > cols) {
            ++p.row;
            p.col = 0;
        }
        p.col += w;
        i += (size_t)u.len;
    }
    return p;
}

size_t row_target(const std::string& buf, int prompt_w, int cols, int row, int col) {
    if (cols <= 0) cols = 80;
    const std::vector<Row> rows = rows_of(buf, prompt_w, cols);
    if (row < 0 || (size_t)row >= rows.size()) return std::string::npos;
    const Row& r = rows[(size_t)row];
    const int target = r.col0 + col;
    int c = r.col0;
    for (size_t i = r.begin; i < r.end; ) {
        const Rune u = decode_rune(buf, i);
        const int w = advance(u.cp, c, cols);
        if (c + w > target) return i;    // the cursor sits before this character
        c += w;
        i += (size_t)u.len;
    }
    return r.end;                    // the row is shorter than the column asked for
}

size_t line_start(const std::string& buf, size_t at) {
    size_t i = std::min(at, buf.size());
    while (i > 0 && buf[i - 1] != '\n') --i;
    return i;
}

size_t line_end(const std::string& buf, size_t at) {
    size_t i = std::min(at, buf.size());
    while (i < buf.size() && buf[i] != '\n') ++i;
    return i;
}

std::string paste_normalise(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ) {
        const unsigned char c = (unsigned char)raw[i];
        if (c == '\r') {
            out.push_back('\n');
            ++i;
            if (i < raw.size() && raw[i] == '\n') ++i;   // CRLF is one break
            continue;
        }
        if (c == '\n' || c == '\t') {
            out.push_back((char)c);
            ++i;
            continue;
        }
        if (c < 0x20 || c == 0x7F) {   // other C0 and DEL are not text
            ++i;
            continue;
        }
        out.push_back((char)c);
        ++i;
    }
    return out;
}

}  // namespace lineedit
