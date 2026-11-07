// A small line editor for the chat REPL: arrows, home/end, history, the usual
// control keys.  No dependency, because the target has no guaranteed readline
// and the engine must still run from a pipe.
//
// When stdin is not a terminal this is a plain getline(): the worker protocol,
// the regression scripts and `echo hi | engine_mm3 --chat` must not see escape
// sequences or a missing echo.
#ifndef LINEEDIT_H
#define LINEEDIT_H

#include <string>

// Reads one line from stdin, with editing when stdin is a tty.  Returns false on
// EOF (Ctrl-D on an empty line, or a closed pipe).  `prompt` is printed to
// stdout before the line.
bool line_edit_read(const char* prompt, std::string& out);

// True when stdin is a terminal, i.e. line_edit_read() is doing real editing.
bool line_edit_is_tty();

#endif
