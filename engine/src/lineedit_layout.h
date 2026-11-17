#pragma once
// Where a byte of the input buffer sits on the screen, and what a vertical move
// means, for the chat REPL's line editor.
//
// The editor used to be single-line by construction: Enter ended the line before
// any key was dispatched, so the buffer could never contain '\n', Up/Down were
// bound to history, and the cursor was moved with relative CUB ('ESC[nD'), which
// stops at column 0 of the current row and therefore cannot reach an earlier row
// of a wrapped line.  Two things a user sees: pressing Up on a pasted multi-line
// block replaces it with a history entry instead of moving the cursor, and a long
// wrapped line cannot be edited above its last screen row.
//
// Everything that decides *where* a byte sits is a pure function here, over
// (buffer, offset, prompt width, terminal width), so the arithmetic has a test
// that needs no terminal: tests/test_lineedit.cpp, run by `make check`.
//
// The conventions are the terminal's, because the editor prints prompt + buffer
// and lets the terminal do the wrapping:
//   * row 0 begins at column prompt_w; every later row begins at column 0;
//   * '\n' ends a row: the next byte starts the next row at column 0;
//   * '\t' advances to the next multiple of 8 columns (the tty's own stops);
//   * a character that does not fit on the rest of the row starts the next row;
//   * a row may end with the cursor in its last cell and the wrap pending: that
//     is Pos::col == cols, and the screen column to address is cols - 1.

#include <cstddef>
#include <string>
#include <vector>

namespace lineedit {

struct Pos {
    int row = 0;
    int col = 0;
};

// One row of the display: [begin, end) are byte offsets and col0 is the column of
// the first byte on the row.  `end` is the byte that starts the next row (a '\n',
// or the character that wrapped), so it is not part of this row's content.
struct Row {
    size_t begin = 0;
    size_t end = 0;
    int col0 = 0;
};

std::vector<Row> rows_of(const std::string& buf, int prompt_w, int cols);

Pos pos_of(const std::string& buf, size_t at, int prompt_w, int cols);

// The byte offset on `row` closest to column `col`, measured from the *start of
// that row's text* (so the first row's prompt does not shift it), clamped to the
// row's end.  std::string::npos when the row does not exist.
//
// Row-relative rather than absolute is what a vertical move means: from the third
// character of "bbb" (row 1, absolute column 3) Up must reach the third character
// of "aaa" (row 0, absolute column 9, because the prompt occupies 0..5), not the
// prompt.
size_t row_target(const std::string& buf, int prompt_w, int cols, int row, int col);

// Edits that are line-relative rather than buffer-relative: Home/End, Ctrl-U,
// Ctrl-K.  On a one-line buffer these are the old 0 and size().
size_t line_start(const std::string& buf, size_t at);
size_t line_end(const std::string& buf, size_t at);

// A UTF-8 character is the unit of every edit, never a byte.
struct Rune {
    char32_t cp = 0;
    int len = 1;
};
Rune decode_rune(const std::string& s, size_t i);
int rune_width(char32_t cp);
int text_width(const std::string& s);
size_t prev_char(const std::string& s, size_t i);
size_t next_char(const std::string& s, size_t i);

// What the terminal framed as a paste, made safe for the buffer: the clipboard's
// line breaks are '\n' here (a bare '\r' would otherwise be sent to the model),
// and the other C0 bytes are not text.  Nothing else is touched -- the buffer
// holds exactly what was pasted, so the screen is what the model receives.
std::string paste_normalise(const std::string& raw);

}  // namespace lineedit
