# OpenAI 兼容 MiniMind3 服务器

这是 MiniMind3 GLES2 引擎外的一层无依赖 OpenAI 风格 HTTP 服务。
引擎是长生命周期 worker；Python 进程负责 HTTP、JSON、SSE 和 tool-call
schema。

```text
OpenAI 客户端 / curl
        |  HTTP + SSE
server.py（仅标准库）
        |  base64 + tab 分隔请求，JSONL 事件
engine_mm3 --worker（一个 GL context，一个模型）
```

## 启动

```bash
cd server
python3 server.py --host 127.0.0.1 --port 8000
```

默认：

- engine：`../engine/engine_mm3`
- model：`../models/minimind3/minimind3.q8`
- tokenizer：`../models/minimind3/minimind3.tok`
- model id：`minimind3-64m`
- cache：1024 tokens
- KV packing：关闭（24-bit）；可用 `--kv-pack 16` 或 `8`
- worker log：`worker.log`

使用离线 q4：

```bash
MODEL=../models/minimind3/minimind3.q4 \
python3 server.py --seq 768 --kv-pack 16
```

不要把 `Q4_WEIGHTS=1` 和 Q4M1 模型文件一起设置。

## 接口

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

非流式：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimind3-64m",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 32,
    "temperature": 0.8,
    "top_p": 0.9,
    "seed": 2
  }'
```

流式：

```bash
curl -sS -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimind3-64m",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 32,
    "stream": true
  }'
```

Thinking：

```json
{
  "messages": [{"role": "user", "content": "1+1等于几？"}],
  "chat_template_kwargs": {"open_thinking": true}
}
```

思考模式完全由请求参数决定，支持常见客户端的写法：

```json
{"thinking": true}
{"thinking": {"type": "enabled"}}
{"reasoning_effort": "low"}
{"reasoning": {"enabled": true}}
{"enable_thinking": true}
{"chat_template_kwargs": {"enable_thinking": true}}
{"extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}
```

`thinking: false`、`{"type": "disabled"}`、`reasoning_effort: "none"`
以及不传参数都表示**关闭**；没有基于文本内容猜测的规则。

## 工具调用

服务器使用标准 OpenAI 流程：返回 `tool_calls`，客户端执行工具，然后发回
`role: "tool"` 消息。

```json
{
  "model": "minimind3-64m",
  "messages": [
    {"role": "user", "content": "256 乘以 37 等于多少？"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculate_math",
        "description": "Evaluate a math expression",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string"}},
          "required": ["expression"]
        }
      }
    }
  ]
}
```

返回的 `tool_calls` 直接转发给客户端；客户端执行后，把结果作为
`role: "tool"` 加回 messages。流式 tool call 会等完整
`<tool_call>` 块生成后再一次性发出，避免半截 JSON。

## Prefix KV cache

worker 的 token/KV 复用、磁盘 `MMKV1` 格式和 packed KV 模式见
[../docs/prefix-cache.md](../docs/prefix-cache.md)。

## 限制

- 只支持内置工具；外部工具由客户端执行。
- Python 层不是瓶颈；缓存请求的 HTTP 开销约 1 ms。
- 模型/缓存必须在目标机器上匹配；不要在笔记本和 Pi 之间复制 KV cache。

完整协议和事件格式见英文 [README.md](README.md)。
