#!/usr/bin/env python3
"""Self-test for the deterministic tool router and the server routes.

Runs without an engine: a StubWorker stands in for the GLES2 worker, so every
assertion here is about routing, not about inference.  Exit code 0 = all pass.

    python3 server/selftest.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as srv  # noqa: E402
import localtools as local_tools  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# --------------------------------------------------------------------------- tools
print("== tools: calculate_math ==")
cases = [
    ("256*37", 9472),
    ("12+30*2", 72),
    ("sqrt(2)", 1.4142135624),
    ("pi", 3.1415926536),
    ("2**10", 1024),
    ("100//7", 14),
    ("100%7", 2),
    ("(3+5)*2", 16),
    ("-4*3", -12),
    ("max(3,7,5)", 7),
    ("round(2.567,2)", 2.57),
]
for expr, expected in cases:
    got = local_tools.run_tool("calculate_math", {"expression": expr})
    ok = got.get("ok") and abs(float(got["result"]["result"]) - float(expected)) < 1e-6
    check(f"math {expr!r} == {expected}", bool(ok), str(got))

print("== tools: calculate_math rejects ==")
rejects = [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "(1).__class__",
    "2**9999",
    "1/0",
    "lambda: 1",
    "[1,2,3]",
    "x+1",
    "",
    "a=1",
]
for expr in rejects:
    got = local_tools.run_tool("calculate_math", {"expression": expr})
    check(f"math rejects {expr!r}", got.get("ok") is False, str(got))

print("== tools: names and aliases ==")
check("alias get_status -> system_status",
      local_tools.resolve_tool_name("get_status") == "system_status")
check("alias calc -> calculate_math",
      local_tools.resolve_tool_name("calc") == "calculate_math")
check("unknown tool rejected",
      local_tools.run_tool("rm_rf", {}).get("ok") is False)

print("== tools: status and time ==")
st = local_tools.run_tool("system_status", {})
check("system_status ok", st.get("ok") is True, str(st))
check("system_status has host", bool(st.get("result", {}).get("host")), str(st))
tm = local_tools.run_tool("get_current_time", {"timezone": "UTC"})
check("time utc ok", tm.get("ok") is True and tm["result"]["iso"].endswith("Z"), str(tm))
tm_local = local_tools.run_tool("get_current_time", {})
check("time local ok", tm_local.get("ok") is True, str(tm_local))

print("== tools: network and disk ==")
net = local_tools.run_tool("network_info", {})
check("network_info ok", net.get("ok") is True, str(net)[:200])
check("network_info reports host", bool(net.get("result", {}).get("host")), str(net)[:200])
disk = local_tools.run_tool("disk_usage", {})
check("disk_usage / ok", disk.get("ok") is True and disk["result"]["total_mb"] > 0, str(disk)[:200])
check("disk_usage rejects relative path", local_tools.run_tool("disk_usage", {"path": "tmp"}).get("ok") is False)
check("disk_usage rejects ..", local_tools.run_tool("disk_usage", {"path": "/var/../etc"}).get("ok") is False)

print("== tools: argument handling ==")
check("string arguments parsed as JSON",
      local_tools.run_tool("calculate_math", '{"expression": "1+1"}').get("ok") is True)
check("bad JSON arguments reported",
      local_tools.run_tool("calculate_math", "{oops").get("ok") is False)
check("extra args ignored but surfaced",
      local_tools.run_tool("calculate_math",
                           {"expression": "1+1", "bogus": 1}).get("ignored_args") == ["bogus"])


# --------------------------------------------------------------------------- server
class TinyContextWorker:
    """Raises ContextFull whenever the prompt carries a marker, records prompts."""

    def __init__(self, marker="[[TOO_LONG]]", fail_always=False, seq=64):
        self.marker = marker
        self.fail_always = fail_always
        self.seq = seq
        self.prefix_cache_tokens = 0
        self.alive = True
        self.restarts = 0
        self.prompts = []

    def ensure_alive(self):
        pass

    def generate(self, prompt, max_tokens, temperature, top_p, seed, on_token=None, penalties=None):
        self.prompts.append(prompt)
        if self.marker in prompt and (self.fail_always or len(self.prompts) == 1):
            raise srv.ContextFull("[context full]")
        return "ok", "length", {"prompt_tokens": 1, "completion_tokens": 1}


class StubWorker:
    """Records calls; never touches a GPU."""

    def __init__(self) -> None:
        self.prefix_cache_tokens = 0
        self.calls = 0
        self.last_prompt = ""
        self.total_tokens = 0
        self.restarts = 0
        self.alive = True
        self.last_penalties = None

    def ensure_alive(self) -> None:
        if not self.alive:
            raise srv.WorkerError("engine worker exited (auto-restart disabled or exhausted)")

    def generate(self, prompt, max_tokens, temperature, top_p, seed, on_token=None, penalties=None):
        self.ensure_alive()
        self.calls += 1
        self.last_penalties = penalties
        self.last_prompt = prompt
        self.total_tokens += 1
        if on_token:
            on_token("STUB")
        return "STUB", "stop", {"prompt_tokens": 1, "completion_tokens": 1,
                                "total_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0},
                                "prefill_tokens": 1}

    def close(self) -> None:
        pass


worker = StubWorker()
httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
httpd.model_id = "minimind3-64m"
httpd.worker = worker
httpd.only_tool = None
httpd.seq = 1024
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{port}"


def http(method: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip().startswith("{") else raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


print("== server: deterministic GET routes ==")
code, body = http("GET", "/status")
check("/status 200", code == 200 and body.get("ok") is True, str(body))
check("/status did not call the model", worker.calls == 0)
code, body = http("GET", "/time")
check("/time 200", code == 200 and body.get("ok") is True, str(body))
code, body = http("GET", "/tools")
names = {t.get("name") for t in body.get("data", [])}
check("/tools lists the local pack", code == 200 and {"system_status", "get_current_time",
      "calculate_math", "network_info", "disk_usage"} <= names, str(names))
code, body = http("GET", "/math?expr=256*37")
check("/math GET computes 9472", code == 200 and body["result"]["result"] == 9472, str(body))
code, body = http("GET", "/math?expr=1/0")
check("/math 400 on bad expression", code == 400 and body.get("ok") is False, str(body))

print("== server: POST tool routes ==")
code, body = http("POST", "/v1/math", {"expression": "12+30*2"})
check("/v1/math 200 -> 72", code == 200 and body["result"]["result"] == 72, str(body))
code, body = http("POST", "/v1/tool", {"name": "get_status_task", "arguments": {}})
check("/v1/tool alias resolves", code == 200 and body["tool"] == "system_status", str(body))
check("tool routes never call the model", worker.calls == 0)

print("== server: named tool_choice bypasses the model ==")
body = {
    "model": "minimind3-64m",
    "messages": [{"role": "user", "content": "256*37"}],
    "tools": [{"type": "function", "function": local_tools.TOOL_SPECS["calculate_math"]}],
    "tool_choice": {"type": "function", "function": {"name": "calculate_math"}},
}
code, resp = http("POST", "/v1/chat/completions", body)
calls = resp.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
check("named tool_choice 200", code == 200, str(resp)[:300])
check("named tool_choice ran calculate_math",
      bool(calls) and calls[0]["function"]["name"] == "calculate_math", str(resp)[:300])
check("named tool_choice result is 9472",
      resp.get("tool_result", {}).get("result", {}).get("result") == 9472, str(resp)[:400])
check("named tool_choice did not call the model", worker.calls == 0,
      f"calls={worker.calls}")

body_str_choice = dict(body, tool_choice="calculate_math", messages=[{"role": "user", "content": "12+30*2"}])
code, resp = http("POST", "/v1/chat/completions", body_str_choice)
check("string tool_choice works",
      resp.get("tool_result", {}).get("result", {}).get("result") == 72, str(resp)[:300])

print("== server: streaming deterministic route ==")
stream_body = dict(body, stream=True, stream_options={"include_usage": True})
req = urllib.request.Request(BASE + "/v1/chat/completions",
                             data=json.dumps(stream_body).encode(), method="POST",
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    raw = r.read().decode()
check("stream has [DONE]", "data: [DONE]" in raw, raw[-200:])
check("stream carries tool_calls", '"tool_calls"' in raw, raw[-300:])
check("stream carries usage", '"usage"' in raw, raw[-300:])
check("stream did not call the model", worker.calls == 0 and worker.total_tokens == 0,
      f"calls={worker.calls}")

print("== server: tool_choice required with one local tool ==")
req_body = {"model": "minimind3-64m",
            "messages": [{"role": "user", "content": "现在几点"}],
            "tools": [{"type": "function", "function": local_tools.TOOL_SPECS["get_current_time"]}],
            "tool_choice": "required"}
calls_before = worker.calls
code, resp = http("POST", "/v1/chat/completions", req_body)
check("required + single local tool -> runs it",
      resp.get("tool_result", {}).get("tool") == "get_current_time", str(resp)[:250])
check("required + single local tool -> no model", worker.calls == calls_before,
      f"calls={worker.calls}")

two_local = dict(req_body, tools=[
    {"type": "function", "function": local_tools.TOOL_SPECS["get_current_time"]},
    {"type": "function", "function": local_tools.TOOL_SPECS["system_status"]}])
code, resp = http("POST", "/v1/chat/completions", two_local)
check("required + two tools -> falls back to the model", worker.calls == calls_before + 1,
      f"calls={worker.calls}")

print("== server: auto still uses the model ==")
auto_body = dict(body)
auto_body.pop("tool_choice")
before_auto = worker.calls
code, resp = http("POST", "/v1/chat/completions", auto_body)
check("auto path calls the model", worker.calls == before_auto + 1, f"calls={worker.calls}")
check("auto path returns STUB", resp["choices"][0]["message"]["content"] == "STUB", str(resp)[:200])

print("== server: --only-tool filters the prompt ==")
httpd.only_tool = "calculate_math"
two_tools = {"model": "minimind3-64m",
             "messages": [{"role": "user", "content": "hi"}],
             "tools": [{"type": "function", "function": local_tools.TOOL_SPECS["calculate_math"]},
                       {"type": "function", "function": local_tools.TOOL_SPECS["system_status"]}]}
http("POST", "/v1/chat/completions", two_tools)
prompt = worker.last_prompt
check("only-tool prompt keeps calculate_math", '"name": "calculate_math"' in prompt, prompt[-300:])
check("only-tool prompt drops system_status", '"name": "system_status"' not in prompt, prompt[-300:])
httpd.only_tool = None

print("== server: tool_choice none skips tools ==")
code, resp = http("POST", "/v1/chat/completions", dict(auto_body, tool_choice="none"))
check("tool_choice none still answers", code == 200 and resp["choices"][0]["message"]["content"] == "STUB",
      str(resp)[:200])

print("== sampling penalties reach the worker ==")
code, resp = http("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "hi"}],
    "repetition_penalty": 1.3, "frequency_penalty": 0.5,
    "presence_penalty": -0.5, "min_p": 0.05})
check("penalties forwarded", worker.last_penalties == {
    "repetition_penalty": 1.3, "frequency_penalty": 0.5,
    "presence_penalty": -0.5, "min_p": 0.05}, str(worker.last_penalties))
code, resp = http("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "hi"}], "repetition_penalty": 9.0})
check("penalties clamped", worker.last_penalties == {"repetition_penalty": 2.0},
      str(worker.last_penalties))

print("== stop sequences ==")
from server import StopFilter, apply_stop_text  # noqa: E402

text, hit = apply_stop_text("hello world END tail", ["END"])
check("apply_stop_text truncates", text == "hello world " and hit, f"{text!r} {hit}")
text, hit = apply_stop_text("no marker here", ["END"])
check("apply_stop_text leaves text", text == "no marker here" and not hit, f"{text!r}")

collected = []
sf = StopFilter(["WORLD"], collected.append)
for piece in ["hel", "lo ", "WOR", "LD", "after"]:
    sf.feed(piece)
sf.finish()
check("StopFilter cuts across token boundaries", "".join(collected) == "hello ", repr("".join(collected)))
check("StopFilter reports the hit", sf.hit is True)

collected2 = []
sf2 = StopFilter(["ZZZ"], collected2.append)
for piece in ["abc", "def"]:
    sf2.feed(piece)
sf2.finish()
check("StopFilter passes text without stops", "".join(collected2) == "abcdef", repr("".join(collected2)))

print("== server: stop in a real response ==")
code, resp = http("POST", "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}], "stop": ["TU"]})
txt = resp["choices"][0]["message"].get("content") or ""
check("non-stream stop truncates", txt == "S", repr(txt))
check("non-stream stop sets finish_reason",
      resp["choices"][0]["finish_reason"] == "stop", resp["choices"][0]["finish_reason"])

print("== tool definitions are canonicalised for the prefix cache ==")
para = [{"type": "function", "function": {"name": "calculate_math",
         "description": "some other wording", "parameters": {"type": "object", "properties": {}}}}]
canon = [{"type": "function", "function": srv.local_tools.TOOL_SPECS["calculate_math"]}]
p_para, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}], "tools": para})
p_canon, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}], "tools": canon})
check("paraphrased tool renders identically", p_para == p_canon)
check("paraphrased wording is dropped", "some other wording" not in p_para)

reordered = [{"type": "function", "function": srv.local_tools.TOOL_SPECS[n]}
             for n in ("disk_usage", "system_status")]
in_order = [{"type": "function", "function": srv.local_tools.TOOL_SPECS[n]}
            for n in ("system_status", "disk_usage")]
r1, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}], "tools": reordered})
r2, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}], "tools": in_order})
check("tool order is normalised", r1 == r2)

custom = [{"type": "function", "function": {"name": "my_tool", "description": "x",
           "parameters": {"type": "object", "properties": {}}}}]
p_custom, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}], "tools": custom})
check("unknown tools pass through", "my_tool" in p_custom)

print("== request stats ==")
stats_path = str(Path(tempfile.mkdtemp()) / "stats.jsonl")
httpd.stats_file = stats_path
httpd.stats_lock = threading.Lock()
code, _ = http("POST", "/v1/chat/completions",
               {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
httpd.stats_file = None
lines = []
if Path(stats_path).exists():
    lines = [json.loads(l) for l in Path(stats_path).read_text().splitlines() if l.strip()]
check("stats line written", len(lines) == 1, str(lines)[:200])
check("stats records status and wall time", bool(lines) and lines[0].get("status") == 200
      and "wall_s" in lines[0], str(lines)[:200])
check("stats records token usage", bool(lines) and isinstance(lines[0].get("usage"), dict),
      str(lines)[:200])

print("== context overflow handling ==")
tiny = TinyContextWorker()
httpd.worker = tiny
httpd.seq = 1024        # big window: the engine's error must drive the retry
code, resp = http("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "old [[TOO_LONG]] turn"},
                 {"role": "user", "content": "new question"}],
    "max_tokens": 4})
check("trims history and retries", code == 200 and len(tiny.prompts) == 2, f"code={code} tries={len(tiny.prompts)}")
check("retry prompt dropped the long turn",
      "[[TOO_LONG]]" not in tiny.prompts[-1], tiny.prompts[-1][:120])

tiny2 = TinyContextWorker(fail_always=True)
httpd.worker = tiny2
code, resp = http("POST", "/v1/chat/completions", {
    "messages": [{"role": "user", "content": "[[TOO_LONG]]"}], "max_tokens": 4})
check("unrecoverable overflow is a 400",
      code == 400 and "context" in json.dumps(resp).lower(), f"code={code} {resp}")

httpd.worker = worker
httpd.seq = 1024

print("== server: dead worker ==")
worker.alive = False
code, body = http("GET", "/health")
check("/health 503 when worker is dead", code == 503 and body.get("worker_alive") is False, str(body))
code, resp = http("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
check("chat 500 when worker cannot restart", code == 500, str(resp)[:200])
worker.alive = True
code, body = http("GET", "/health")
check("/health 200 after recovery", code == 200 and body.get("worker_alive") is True, str(body))


print("== chat template comes from the checkpoint ==")
chatml, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}]})
WANT_CHATML = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
               "<think>\n\n</think>\n\n")
check("chatml renderer unchanged", chatml == WANT_CHATML, repr(chatml))
mm3, _ = srv.build_prompt({"messages": [{"role": "user", "content": "hi"}]}, "minimind3")
check("minimind3 name keeps the chatml renderer", mm3 == chatml, repr(mm3))

tbb_body = {"messages": [{"role": "system", "content": "be terse"},
                         {"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "Hello!"},
                         {"role": "user", "content": "who are you?"}]}
tbb, tbb_thinking = srv.build_prompt(tbb_body, "tinybrainbot")
WANT_TBB = "<|user|> hi<|assistant|> Hello!<|end|><|user|> who are you?<|assistant|> "
check("tinybrainbot prompt is its own template", tbb == WANT_TBB, repr(tbb))
check("tinybrainbot drops the chatml markers and the system turn",
      "<|im_start|>" not in tbb and "be terse" not in tbb, repr(tbb))
check("tinybrainbot has no thinking block", tbb_thinking is False and "<think>" not in tbb, repr(tbb))

# The worker reports the template in its ready event; the HTTP path must follow.
worker.template = "tinybrainbot"
httpd.worker = worker
code, body = http("GET", "/health")
check("/health reports the template", body.get("template") == "tinybrainbot", str(body))
code, resp = http("POST", "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
check("HTTP request is rendered with the checkpoint template",
      worker.last_prompt == "<|user|> hi<|assistant|> ", repr(worker.last_prompt))
worker.template = "chatml"
httpd.shutdown()
httpd.server_close()

print()
print(f"passed {len(PASS)}  failed {len(FAIL)}")
if FAIL:
    print("failed:", ", ".join(FAIL))
    sys.exit(1)
