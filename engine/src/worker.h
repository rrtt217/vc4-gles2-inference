#pragma once
// The worker protocol (see worker.cpp).

#include <string>

#include "app.h"
#include "session.h"

// The tool list is not a parameter: the worker's request carries it (column 11).
int run_worker(App& app, Session& session, bool enc_bos, bool chat_dump);
