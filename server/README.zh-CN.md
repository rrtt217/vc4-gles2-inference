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

以 `{` 开头的一行是**控制行**而不是请求（请求第一列是 base64 的 prompt，base64 没有 `{`）。
目前只支持一个操作：`{"op":"load","model":"..."}`，它不重建 EGL 上下文与已编译程序地换模型
（板上省下 4.23 s 上下文 + 1.89 s 程序链接，代价是 6.12 s 读容器 + 2.99 s 上载，见
`docs/performance.md`）。回复一个 `loaded` 事件（含 `load_ms`）或一个 `error` 事件；容器读不出来
时进程直接退出，滚动前缀缓存随模型一起丢弃。验收见 `engine/scripts/swap_matrix.py`。

默认：

- engine：`../engine/engine_mm3`
- model：`../models/minimind3/minimind3.q8.mmcf`
- tokenizer：容器内置（只有旧式 `.tok` 采用 `--tokenizer`）
- model id：`minimind3-64m`
- cache：1024 tokens
- KV packing：关闭（24-bit）；可用 `--kv-pack 16` 或 `8`
- worker log：`worker.log`

使用离线 q4：

```bash
MODEL=../models/minimind3/minimind3.q4.mmcf \
python3 server.py --seq 768 --kv-pack 16
```

不要把 `Q4_WEIGHTS=1` 和 Q4M1 模型文件一起设置。

## 对话模板

**引擎是唯一的渲染器**：它用容器自带的 Jinja 模板（MMCF 元数据 key 9）渲染，
与 `llama.cpp` 的 `tokenizer.chat_template` 同一个思路。服务器只发 messages、
不再拼任何 prompt 文本。`ready` 事件里的 `"chat_template": true|false` 决定
能不能服务 `/v1/chat/completions`：为 false 时直接返回 400 并提示用
`mmcf_pack.py --chat-template <file>.jinja` 重新打包——过去那种"猜一个格式"的
回退，会把模型没见过的 ChatML 标记塞到它前面。仓库里所有对话容器都带 key 9，
只有 `stories*` 补全模型没有（它们也没有对话轮可渲染）。`GET /health` 的
`template` 字段就是容器声明的模板名。

system 轮仍是服务器的策略：Chatbox 的运行时元数据会被剥掉、客户端没给时填入中性默认值，
所以引擎收到的永远是一份具体的 messages。

TinyBrainBot 还需要 `--no-gpu-layer`：它在 VC4 上的整层 GPU 路径目前不正确
（见 docs/performance.md 的 item 12），可部署的是 CPU 路径。

## logprobs

`logprobs: true` 返回 OpenAI 形状的 `logprobs.content`：每个生成 token 一条
`{token, logprob}`，`logprob` 是在**采样器实际使用的分布**里测的（温度、top-k、top-p
过滤后按保留集重归一化）。引擎用的是它采样时的同一份 logits，所以**要 logprob 不会改变
文本**，只增加一点计算；这类请求会绕过候选读回路径（`--top-k-candidates`），因为那条路径
不带全词表。

尚未实现、并且会明确拒绝而不是静默忽略：`top_logprobs`（备选 token 需要把候选 id 解回
文本）和 `stream: true` 时的 `logprobs`。

## 一个服务器多个模型

`--model` 可以重复；每个容器都会出现在 `GET /v1/models` 里，`/health` 还会说明当前驻留的是哪个。
`--model-id` 按顺序对应 `--model`，没给就用容器文件名。请求里的 `model` 字段选择容器，
**同一时刻只驻留一个**：Pi Zero 的 CMA 只放得下一个，而引擎换模型时保留 EGL 上下文和已编译程序，
所以切换的代价是一次容器读取加上载（板上 6.12 s + 2.99 s），而不是重启进程（那还要再付 4.23 s 上下文
和 1.89 s 程序链接，见 `docs/performance.md`）。切换发生在第一个指定了另一个模型的请求上，并打印
`[model] <id> resident, load <ms> ms`；已驻留的模型不会每请求重载。容器自带的聊天模板随之切换，
所以 `/health` 的 `template` 与"无模板即 400"的判定都跟着当前模型走。未知模型名返回 400 并列出可用的 id。

## 控制行、健康检查与取消

以 `{` 开头的一行是控制行（请求第一列是 base64，不含 `{`），现有三个操作：

| op | 回复 | 用途 |
|---|---|---|
| `{"op":"load","model":PATH}` | `loaded` | 换容器，保留 EGL 上下文与已编译程序 |
| `{"op":"cancel","id":ID}` | `cancelled` | 停止正在跑的请求（引擎在 token 之间窥视 stdin，几个 token 内生效）；空闲时的 cancel 是确认，不是错误 |
| `{"op":"stats"}` | `stats` | 只有引擎知道的数：`cma_free_kb`（板上最关键，它决定还能不能放下另一个模型）、`mem_available_kb`、已编译 `programs`、`seq`、`kv_pack`、`template`、`uptime_ms` |

`GET /health` 还会给 `busy`/`inflight`（引擎单飞，所以就是队列）、`est_wait_ms`（近八次请求中位数 × 前面排队数；有排队但还没有已完成请求时返回 `-1`）、`last_request_ms`，以及 `engine`（引擎最近一次自报的数值）。

每个完成的请求会打一行带 id 的日志，客户端拿到 `chatcmpl-…` 就能在服务器输出里找到它：
`[req] chatcmpl-… model=mm3-q8 stream=0 finish=length prompt=31 completion=4 wall=0.72s`。

客户端在流式过程中断开时，服务器向引擎发 `{"op":"cancel"}` 并打印
`[cancel] <id> sent=1 (client disconnected)`；这是**尽力而为**——只有在下一块写入失败时才会察觉，所以剩余很少时可能已经跑完。worker 死掉时的 500 会附上引擎最后几行日志（`--- engine stderr tail ---`），真正的原因（CMA/GL）在那里。

## 接口

```text
GET  /health
GET  /v1/models
GET  /status            确定性，不走模型
GET  /time              确定性，不走模型
GET  /tools             列出确定性工具
GET  /math?expr=...     确定性算术
POST /v1/math           {"expression": "..."}
POST /v1/tool           {"name": "...", "arguments": {...}}
POST /v1/chat/completions
```

`/status`、`/time`、`/math`、`/tool` 四条路径完全不碰 GL worker：在 Pi Zero 上
10–30 ms 返回，而同一个问题走模型要 25 秒以上。

### 健康检查与 worker 自愈

```json
{"status": "ok", "worker_alive": true, "restarts": 1, "prefix_cache_tokens": 277}
```

引擎进程一消失，`GET /health` 立即返回 **503**（`status: degraded`，
`worker_alive: false`）；下一次 `/v1/chat/completions` 会**就地重建 worker**
（重载模型 + 前缀缓存，Pi 上约 30 秒）并正常返回，而不是一直报错。
`--no-auto-restart` 关闭该行为，`--max-restarts N`（默认 5）限制次数。

实测：杀掉 worker 后 `/health` 立刻变 503，随后一次对话请求在重建后正常返回
（`restarts: 1`，缓存重新加载）。

## 确定性工具（模型不参与）

64M 模型挑不准工具、也写不准参数（实测把 `system_status` 说成 `get_status` /
`get_status_task`，把 `256*37` 改成 `32 + 45 * 3`）。因此确定性工具支持**点名调用**。

工具在 `server/localtools.py`：

| 工具 | 参数 | 返回 |
|---|---|---|
| `system_status` | — | 主机名、架构、uptime、load、内存、CPU 温度、**CmaFree** |
| `get_current_time` | 可选 `timezone` | ISO 8601 + 时间戳，支持 UTC / 本地 |
| `calculate_math` | `expression` | 精确算术（四则、整除、取余、幂、sqrt/sin/cos/tan/log/exp/floor/ceil/round/min/max、pi、e） |
| `network_info` | — | 非回环 IPv4 接口、默认网关、DNS 服务器 |
| `disk_usage` | 可选 `path`（默认 `/`） | 该文件系统的总量/已用/剩余（MB）与使用率 |

数学求值走受限 AST：不允许名字、属性访问、白名单外的函数、超过 64 的指数、超过
200 字符的表达式。`__import__('os')`、`(1).__class__`、`lambda: 1` 都会在求值前被拒。

### 用 `tool_choice` 路由

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "256*37"}],
  "tool_choice": "calculate_math"
}'
```

支持 `"calculate_math"`、`{"type":"function","function":{"name":"calculate_math"}}`，
以及 OpenAI 的 `"auto"` / `"none"` / `"required"`。点名到本地工具时请求**不进 worker**：

- `message.content` 放 JSON 结果，普通客户端也能直接看到答案；
- `message.tool_calls` 同时报告这次调用，Agent 客户端可以继续走标准工具循环；
- 顶层 `tool_result` 放同一份结果，方便排查；
- `usage` 全 0，`finish_reason` 为 `tool_calls`。

参数优先取请求里的 `tool_arguments`，否则取最后一条 user 消息（`calculate_math` 直接
把消息当表达式，开头的 `/math` 会被去掉）。任何非法输入返回
`{"ok": false, "error": "..."}` 和 HTTP 400，而不是被执行。

`--only-tool NAME` 把 prompt 里的工具列表裁到只剩一个：模型仍然在选，但选项只有一个。

## 自测、评测、前缀缓存

```bash
python3 server/selftest.py            # 路由、SSE 与采样，不需要 GPU
python3 server/eval_suite.py --base http://127.0.0.1:8000 --out eval.json
python3 server/build_prefix_cache.py build  --tools system_status --out server/pfx.kv
python3 server/build_prefix_cache.py verify --cache server/pfx.kv
```

`selftest.py` 用 stub worker 替换引擎，1 秒内验证路由、参数校验和 SSE 分帧；
`eval_suite.py` 对活服务跑固定用例（身份、算术、工具路由、多轮、长 prompt、流式、
畸形请求）并写 JSON 便于对比；前缀缓存脚本会写 sidecar `.json`（引擎哈希、模型哈希、
seq、KV 打包、prompt 哈希、工具列表），`verify` 在元数据不匹配时拒绝加载。

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
    "top_k": 50,
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

Chatbox 会在 system prompt 后追加 `Additional info` / `## Runtime` 元数据。服务端会剥离这段元数据，并使用短默认提示 `You are a helpful assistant.`；包含 `MiniMind` 的自定义提示词会保留。

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

## 上下文超限

超长对话不再直接失败：

1. 服务器按「约 3 字符/token」估算，若 `prompt + max_tokens` 会超过 `--seq`，
   **发送前**就丢掉最旧的非 system 消息（流式一旦发出响应头就无法重试，所以
   这一步必须前置）。
2. 如果引擎仍返回 `[context full]`，非流式路径会裁剪后重试，最多四次；单条超长
   消息会被截半，并带上 `[earlier text trimmed to fit the context window]` 标记。
3. 只有当实在没有可裁剪的内容时才失败，返回 HTTP **400** 与
   `code: context_length_exceeded`。

Pi 实测：单条 2880 字符消息被裁剪后正常回答（prefill 占大头，约 190 s）；两轮
对话中第一轮是这条长消息时，丢掉第一轮后约 10 s 返回。

`max_tokens` 超过窗口时由引擎钳制，不报错。

## 限制

- 只支持内置工具；外部工具由客户端执行。
- Python 层不是瓶颈；缓存请求的 HTTP 开销约 1 ms。
- 模型/缓存必须在目标机器上匹配；不要在笔记本和 Pi 之间复制 KV cache。

完整协议和事件格式见英文 [README.md](README.md)。
