#!/usr/bin/env python3
"""Deterministic local tools for the MiniMind3 server.

Why this file exists
--------------------
A 64M model cannot reliably pick a tool out of three, and it mangles arguments
(see docs/future-optimization.md, P0 item 2).  Anything deterministic therefore
must not go through the model at all: the caller names the tool, we validate the
arguments, run it, and return short JSON.

Everything here is dependency-free and safe to run on a Pi Zero:
  * no shell, no eval of arbitrary code, no file paths from the caller
  * the math evaluator walks a restricted AST and refuses anything else
  * every result is a small JSON object (the model may see it, so keep it short)
"""
from __future__ import annotations

import ast
import json
import math
import os
import platform
import time

MAX_EXPRESSION_LEN = 200
MAX_AST_NODES = 120

ALLOWED_FUNCS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
ALLOWED_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}


class ToolError(ValueError):
    """Bad name or arguments.  Never a crash of the server."""


# --------------------------------------------------------------------------- specs

TOOL_SPECS = {
    "system_status": {
        "name": "system_status",
        "description": "Report hostname, uptime, load average, free memory, CPU temperature and GPU CMA free memory of this Raspberry Pi.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_current_time": {
        "name": "get_current_time",
        "description": "Return the current local date and time in ISO 8601, plus epoch seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA name such as Asia/Shanghai. Only UTC and the local zone are resolved offline.",
                }
            },
            "required": [],
        },
    },
    "calculate_math": {
        "name": "calculate_math",
        "description": "Evaluate an arithmetic expression exactly. Supports + - * / // % ** and sqrt/sin/cos/tan/log/exp/abs/floor/ceil/round/min/max, pi and e.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. 256*37 or sqrt(2)+pi",
                }
            },
            "required": ["expression"],
        },
    },
    "network_info": {
        "name": "network_info",
        "description": "Report this machine's IPv4 interfaces (non-loopback), the default gateway and the configured DNS servers. Read-only.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "disk_usage": {
        "name": "disk_usage",
        "description": "Report total, used and free space plus usage percent for the filesystem holding a given path (default /). Read-only.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to inspect, defaults to /"}
            },
            "required": [],
        },
    },
}

DETERMINISTIC_TOOLS = tuple(TOOL_SPECS)

# Aliases the 64M model has been observed to emit, mapped onto real tools.
TOOL_ALIASES = {
    "get_status": "system_status",
    "get_status_task": "system_status",
    "system_status_task": "system_status",
    "status": "system_status",
    "time": "get_current_time",
    "current_time": "get_current_time",
    "get_time": "get_current_time",
    "math": "calculate_math",
    "calculate": "calculate_math",
    "calc": "calculate_math",
    "network": "network_info",
    "ifconfig": "network_info",
    "disk": "disk_usage",
    "df": "disk_usage",
    "storage": "disk_usage",
}


def resolve_tool_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ToolError("tool name must be a non-empty string")
    key = name.strip()
    key = TOOL_ALIASES.get(key, key)
    if key not in TOOL_SPECS:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(DETERMINISTIC_TOOLS)}")
    return key


# --------------------------------------------------------------------------- helpers

def _read_first_line(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readline().strip()
    except OSError:
        return ""


def _meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        out[key.strip()] = int(parts[0])
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _cpu_temp_c() -> float | None:
    raw = _read_first_line("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(raw) / 1000.0, 1)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- tools

def tool_system_status(args: dict) -> dict:
    mem = _meminfo()
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    uptime = None
    raw_uptime = _read_first_line("/proc/uptime")
    if raw_uptime:
        try:
            uptime = int(float(raw_uptime.split()[0]))
        except (IndexError, ValueError):
            uptime = None
    result = {
        "host": platform.node(),
        "machine": platform.machine(),
        "uptime_s": uptime,
        "load": [round(x, 2) for x in load],
        "mem_total_kb": mem.get("MemTotal"),
        "mem_available_kb": mem.get("MemAvailable"),
        "swap_free_kb": mem.get("SwapFree"),
        "cpu_temp_c": _cpu_temp_c(),
        # CmaFree is the real GPU-side occupancy signal on this board.
        "cma_free_kb": mem.get("CmaFree"),
        "cma_total_kb": mem.get("CmaTotal"),
    }
    return {k: v for k, v in result.items() if v is not None}


def tool_get_current_time(args: dict) -> dict:
    tz = args.get("timezone")
    now = time.time()
    local = time.localtime(now)
    result = {
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", local),
        "epoch_s": int(now),
        "timezone": tz or time.strftime("%Z", local) or "local",
    }
    if isinstance(tz, str) and tz.strip().upper() in {"UTC", "GMT", "Z"}:
        utc = time.gmtime(now)
        result.update({
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", utc),
            "timezone": "UTC",
        })
    return result


class _SafeEvaluator(ast.NodeVisitor):
    """Walk an arithmetic AST and evaluate it, refusing everything else."""

    def __init__(self) -> None:
        self.nodes = 0

    def visit(self, node):  # type: ignore[override]
        self.nodes += 1
        if self.nodes > MAX_AST_NODES:
            raise ToolError("expression is too complex")
        return super().visit(node)

    def evaluate(self, node):
        method = getattr(self, f"eval_{type(node).__name__}", None)
        if method is None:
            raise ToolError(f"unsupported syntax: {type(node).__name__}")
        return method(node)

    def eval_Expression(self, node):
        return self.evaluate(node.body)

    def eval_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError("only numeric constants are allowed")

    def eval_Name(self, node):
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise ToolError(f"unknown name: {node.id}")

    def eval_BinOp(self, node):
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        op = type(node.op).__name__
        if op == "Add":
            return left + right
        if op == "Sub":
            return left - right
        if op == "Mult":
            return left * right
        if op == "Div":
            if right == 0:
                raise ToolError("division by zero")
            return left / right
        if op == "FloorDiv":
            if right == 0:
                raise ToolError("division by zero")
            return left // right
        if op == "Mod":
            if right == 0:
                raise ToolError("modulo by zero")
            return left % right
        if op == "Pow":
            if abs(right) > 64:
                raise ToolError("exponent is too large")
            return left ** right
        raise ToolError(f"unsupported operator: {op}")

    def eval_UnaryOp(self, node):
        value = self.evaluate(node.operand)
        op = type(node.op).__name__
        if op == "UAdd":
            return +value
        if op == "USub":
            return -value
        raise ToolError(f"unsupported unary operator: {op}")

    def eval_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ToolError("only plain function names may be called")
        func = ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ToolError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise ToolError("keyword arguments are not allowed")
        args = [self.evaluate(a) for a in node.args]
        if not args:
            raise ToolError("function needs at least one argument")
        return func(*args)


def tool_calculate_math(args: dict) -> dict:
    expr = args.get("expression")
    if not isinstance(expr, str) or not expr.strip():
        raise ToolError("argument 'expression' must be a non-empty string")
    expr = expr.strip()
    if len(expr) > MAX_EXPRESSION_LEN:
        raise ToolError(f"expression longer than {MAX_EXPRESSION_LEN} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"cannot parse expression: {e.msg}") from e
    value = _SafeEvaluator().evaluate(tree)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolError("result is not a finite number")
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
        else:
            value = round(value, 10)
    if abs(value) > 1e15:
        raise ToolError("result is out of range")
    return {"expression": expr, "result": value}


def _ipv4_interfaces() -> list[dict]:
    """Non-loopback IPv4 addresses, without shelling out to `ip`."""
    import fcntl
    import socket
    import struct

    SIOCGIFADDR = 0x8915
    out = []
    for _index, name in socket.if_nameindex():
        addr = None
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = struct.pack("256s", name.encode("utf-8")[:15])
            addr = socket.inet_ntoa(fcntl.ioctl(s.fileno(), SIOCGIFADDR, packed)[20:24])
        except OSError:
            addr = None
        finally:
            s.close()
        if not addr or addr.startswith("127."):
            continue
        state = _read_first_line(f"/sys/class/net/{name}/operstate") or None
        out.append({"if": name, "ip": addr, "state": state})
    return out


def _default_gateway() -> str | None:
    """Gateway of the 0.0.0.0 route in /proc/net/route (little-endian hex)."""
    import socket
    import struct

    try:
        with open("/proc/net/route", "r", encoding="utf-8", errors="replace") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 3 or parts[1] != "00000000":
                    continue
                try:
                    return socket.inet_ntoa(struct.pack("<L", int(parts[2], 16)))
                except (ValueError, OSError):
                    return None
    except OSError:
        return None
    return None


def _dns_servers(limit: int = 3) -> list[str]:
    out = []
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver" and not parts[1].startswith("127."):
                    out.append(parts[1])
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out


def tool_network_info(args: dict) -> dict:
    result = {
        "host": platform.node(),
        "interfaces": _ipv4_interfaces(),
        "gateway": _default_gateway(),
        "dns": _dns_servers(),
        "wifi_ssid": None,
    }
    # Pi OS keeps the SSID here when wpa_supplicant is in use; absent is fine.
    ssid = _read_first_line("/proc/net/wireless")
    if ssid:
        result["wifi_present"] = True
    return {k: v for k, v in result.items() if v is not None}


def tool_disk_usage(args: dict) -> dict:
    import shutil

    path = args.get("path") or "/"
    if not isinstance(path, str) or not path.startswith("/"):
        raise ToolError("argument 'path' must be an absolute path")
    if ".." in path.split("/"):
        raise ToolError("argument 'path' must not contain '..'")
    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        raise ToolError(f"cannot stat {path}: {e}") from e
    total = usage.total // (1024 * 1024)
    free = usage.free // (1024 * 1024)
    used = total - free
    return {
        "path": path,
        "total_mb": total,
        "used_mb": used,
        "free_mb": free,
        "used_pct": round(100 * used / total, 1) if total else None,
    }


RUNNERS = {
    "system_status": tool_system_status,
    "get_current_time": tool_get_current_time,
    "calculate_math": tool_calculate_math,
    "network_info": tool_network_info,
    "disk_usage": tool_disk_usage,
}


# --------------------------------------------------------------------------- entry

def run_tool(name: str, args) -> dict:
    """Run a tool by name.  Returns a compact JSON-able dict, never raises."""
    try:
        key = resolve_tool_name(name)
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as e:
                return {"ok": False, "tool": key, "error": f"arguments are not valid JSON: {e.msg}"}
        if not isinstance(args, dict):
            return {"ok": False, "tool": key, "error": "arguments must be a JSON object"}
        unknown = [k for k in args if k not in TOOL_SPECS[key]["parameters"].get("properties", {})]
        if unknown:
            # Not fatal: the caller may send extra OpenAI-ish keys, but surface it.
            args = {k: v for k, v in args.items() if k not in unknown}
        result = RUNNERS[key](args)
        payload = {"ok": True, "tool": key, "result": result}
        if unknown:
            payload["ignored_args"] = unknown
        return payload
    except ToolError as e:
        return {"ok": False, "tool": name, "error": str(e)}
    except Exception as e:  # never let a tool take the server down
        return {"ok": False, "tool": name, "error": f"{type(e).__name__}: {e}"}


def tool_result_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
