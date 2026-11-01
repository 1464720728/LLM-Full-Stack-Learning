# A02 模型协议：写得出适配器，也知道哪里不能转换

## 协议先于 SDK

HTTP 是传输请求与响应的一套机制；JSON 是数据编码格式；Chat Completions、Responses、Messages 则各自规定消息、工具、状态和流式事件怎样表达。Python SDK 包装了这些协议，但 SDK 中的方法名不是底层机制本身。`client.chat.completions.create` 的作用不是在本机运行模型，而是构造请求、发送到服务、解析返回结果。服务器可以运行闭源托管模型，也可以运行你部署的模型。

先把一次请求拆成 URL、HTTP 方法、headers、body。URL 告诉客户端访问哪个服务与资源；header 常携带鉴权和内容类型；body 描述模型与输入。响应则有 HTTP 状态码和协议内容。HTTP 200 只表明请求在 HTTP 层得到了正常响应，不保证生成内容完整，也不保证工具实际执行，更不保证任务成功。

你可以使用同一类 SDK 访问不同提供方的“兼容端点”，但兼容需要逐项验证。字段名一样不意味着支持相同模型选项；服务可能忽略温度参数、限制工具 schema、用不同停止标记，甚至需要保留额外推理字段。协议名、模型能力、服务实现版本是三个不同维度。

## 三种工具循环的结构

| 结构问题 | Chat Completions | Responses | Messages |
|---|---|---|---|
| 客户端工具请求 | assistant 的 `tool_calls` | output 中的 `function_call` 项 | assistant content 中的 `tool_use` 块 |
| 关联身份 | `tool_calls[].id` | `call_id` | `tool_use.id` |
| 工具结果 | `role=tool`，`tool_call_id` | `function_call_output`，`call_id` | user content 的 `tool_result`，`tool_use_id` |
| 历史主体 | 有序消息 | 有序输入/输出项与可选服务端状态 | 消息中的内容块 |
| 需要警惕 | 扩展字段、截断、工具参数字符串 | 不能只取第一项或只保留文字 | 内容块顺序、推理块与工具结果配对 |

这张表针对客户端函数工具的必要结构，不涵盖所有服务端工具、音视频和长任务事件。依据是各提供方一手工具文档，核查日期 2026-11-01；入口见本卷 `SOURCES.md`。实际开发仍应核对选定模型和服务的文档。

以加法为例，Chat 请求的工具结果可以是：

```json
{"role":"tool","tool_call_id":"call-1","content":"{\"value\":42}"}
```

外层 JSON 表示一条消息，`content` 里面又是一个 JSON 字符串。初学者常把这两层混为一谈。Python 中 `json.dumps({"value":42})` 得到字符串，字符串作为 `content` 的值之后，再编码整个 HTTP body。不要手工反复添加反斜杠。

Responses 的 `id` 与 `call_id` 也不要混淆：输出项本身可以有对象身份，函数结果必须关联正确调用身份。Messages 的工具结果属于 user 内容块，不意味着现实中用户亲自输入了工具输出；这是该协议表达客户端观察的方式。

## 为什么我们保留原始消息

参考实现 `from_chat` 同时生成统一的 `Turn`，又保存 `provider_message`。统一字段便于 Harness 理解“文字和工具请求”；原始消息用于同协议往返，保留提供方需要的扩展。两者用途不同。若把原始消息全部压成 `text`，可能丢失 `reasoning_content`、调用身份、签名或其他协议状态。

DeepSeek 当前思考模式文档明确要求在携带 tools 的后续请求中保留所需的 `reasoning_content`。因此本卷适配器不会擅自删除这个字段，也不会把它冒充用户可见答案。不要把某一提供方的要求推广成所有服务都相同；也不要把这些字段直接塞给另一个不接受它们的 API。[DeepSeek 思考模式文档](https://api-docs.deepseek.com/guides/thinking_mode/)

Responses 的 `responses_continuation` 接收上一轮完整 payload，把原始 output 项连同函数结果继续发送。它保留不透明 reasoning 项，是为了协议延续，不是为了读取或展示隐藏推理。Messages 的辅助函数也保留前一轮完整 content，再构造工具结果消息。**透明保存与跨协议理解不是同一件事。** 一个系统能存储某字段，并不意味着它知道怎样把字段转换给另一个服务。

## 逐块读代码

运行三个实验：

```bash
python -m labs.agents.experiments 03-chat
python -m labs.agents.experiments 04-responses
python -m labs.agents.experiments 05-messages
```

`from_chat` 先检查 `finish_reason`，再拿 `choices[0].message`，逐个解析 `tool_calls`。这里仅支持单候选响应；如果你请求多个候选，要明确选择策略而不能假装都执行。`from_responses` 遍历所有 output 项，不假设第一个就是最终文字。遇到不认识的 item 会报错，而不是悄悄丢掉。`from_messages` 遍历 content blocks，把工具的 `input` 对象编码成统一的参数字符串。

这些函数是一个**有限能力适配层**。主 Harness 的真实网络路径目前只接 Chat Completions。Responses 和 Messages 的实验提供正确的数据解析与继续调用构造，用于练习和扩展；没有伪装成三个全部在线验证过的客户端。这样的标注很重要：接口代码能导入，远不等于你已经验证某模型上的端到端行为。

## 真 API 的最小接入

读 `labs/agents/live.py`。构造函数要求完整 endpoint，而不是偷偷猜测 `/v1` 是否已包含。它检查 HTTPS，只有你显式选择本机测试时才允许 localhost 的 HTTP。它拒绝自动重定向，避免把 Authorization 随意带到另一个地址。key 从环境变量读取，输出日志不打印请求 header。单次请求有 timeout，响应体有大小上限，整个循环有调用预算。

下面是命令形式，不包含任何真实 key 或固定型号：

```bash
export COURSE_API_KEY='在自己的终端中设置，不写入仓库'
python -m labs.agents.live --allow-live \
  --endpoint 'https://api.openai.com/v1/chat/completions' \
  --model '你账户实际可用且支持工具调用的模型'
```

在 PowerShell 中环境变量赋值形式为 `$env:COURSE_API_KEY = '...'`。不要把密钥写进 notebook、截图、提交记录或命令行参数；环境变量也不是绝对保密机制，生产环境需要更正式的秘密管理。此命令会产生真实 API 请求和可能的费用，离线测试不会调用它。若提供方要求额外参数，先读文档，再用 `--extra-json` 显式设置，不要盲目照抄其他模型配置。

## 设计能力矩阵

为每个后端维护一份能力声明：支持的协议、是否允许工具、是否允许并行工具、流式事件种类、结构化 schema 子集、最大输出参数名称、思考模式配置、状态保存方式、usage 字段解释。矩阵可以由配置和探针共同生成。探针应尽量短，例如要求模型调用一次 add，观察实际请求与返回结构；不能仅靠 HTTP 200 判定。

适配层最好“默认拒绝未知能力”。例如模型不支持图像，你传图像时就报清楚错误，不应把图像删除后继续回答。遇到提供方新增字段时，日志记录原始字段类型以便调查，但要避免泄露内容。版本升级时重跑契约测试，用录制的脱敏响应作 fixture。这样协议变动会产生明确失败，不会变成用户发现的静默能力下降。

## 练习与参考解法

1. **双层 JSON。** 手写 Python 对象，正确编码工具结果为字符串，再编码整个 body。验收：外层解码后 `content` 类型是 str，内层解码后 value 为 42。参考：恰好两次编码发生在两个不同边界；同一字符串连续编码两次是常见错误。

2. **多工具结果。** 构造两次调用，把结果 ID 故意交换。验收：你自己的关联检查能发现内容与调用不一致。参考：集合检查只能证明 ID 完整，不能证明业务结果对应正确；还要在执行器中将结果直接绑定源调用对象，避免先存两个无身份列表再靠顺序 zip。

3. **截断测试。** 分别构造 `finish_reason=length` 和 Responses `status=incomplete`。验收：不能返回正常 final。参考：内容看起来像完整句子也不改变协议状态；需要重新请求、调整预算或告知不完整。

4. **新增字段。** 给 Chat 消息加一个测试扩展，确保往返保留；给 Responses 加一个未知 item，确保拒绝。参考：同协议透明保存可以宽容，语义解析对未知可执行项应保守，两种策略并不矛盾。

5. **添加在线 Responses 客户端。** 从 `live.py` 分离 HTTP transport，将输入转换、输出解析、继续调用分别实现。验收：离线 fixture 完整通过后，显式运行一次小额线上测试；记录实际 endpoint、模型、时间与结果，不记录 key。参考：保留所有 output 项、匹配 call_id、处理 incomplete/refusal，不把统一 Turn 单独用作全部重放历史。

6. **证明有损转换。** 设计一个带提供方特定推理或内建工具项的响应，写出统一文本接口会丢失哪些信息。参考：标出“无法表达”“可保存但不理解”“可精确转换”三种情况。验收不是声称完全兼容，而是能明确画出能力边界。

通过门槛：无需 SDK 自动 runner，就能手写一次真实工具往返；能解释为什么三个 API 都能调用工具，却不能仅靠重命名字段保证无损转换。
