# 源码工作簿一：亲手从四个函数走到可恢复 Harness

这是一节需要打开编辑器、反复停下来画表的长实验。建议分三次完成，每次只做到自己能不看答案复写的阶段。阅读 A01–A04 后开始，A10 后再做后半部分。最终目标是你能读懂 `runtime.py` 的每条主路径，知道哪个对象被谁改变，为什么改变，以及崩溃时会留下什么。

## 任务 1：先不使用任何类

把下面代码抄到自己的练习文件，先不要运行。预测输出顺序，再逐行解释：

```python
import json

def model(history):
    if history[-1]["role"] == "tool":
        result = json.loads(history[-1]["content"])
        return {"kind": "final", "text": str(result["value"])}
    return {"kind": "call", "id": "c1", "name": "add",
            "arguments": '{"a":17,"b":25}'}

def execute(call):
    args = json.loads(call["arguments"])
    return {"ok": True, "value": args["a"] + args["b"]}

def run(goal):
    history = [{"role": "user", "content": goal}]
    for step in range(3):
        action = model(history)
        if action["kind"] == "final":
            return action["text"], history
        history.append({"role": "assistant", "call": action})
        observation = execute(action)
        history.append({"role": "tool", "call_id": action["id"],
                        "content": json.dumps(observation)})
    raise RuntimeError("step limit")

answer, history = run("计算 17+25")
print(answer)
print(len(history))
```

输出应为 42 和 3。为什么 history 不是4？这版代码把最终答案作为返回值，却没有 append 到 history。参考 Harness 会保存最终 assistant 消息，因此正常完整消息数为4。这个小差异很适合练习：你不能只看两个程序都输出42就认为状态管理等价。历史是否保存 final，会影响恢复、审计和下一轮对话。

再检查 model 中的条件：它只观察最后一条消息角色，如果是 tool 就把工具值转成文字。这里没有自然语言理解，是确定性替身。真正模型也要在相同边界接收历史并返回动作；换模型并不会改变执行器必须自己运行工具的事实。

这段小代码故意不处理错误，不能直接当成完整产品。请在纸上列至少六个缺口，再与下表对照。

| 输入或故障 | 当前代码会怎样 | 需要补的机制 |
|---|---|---|
| arguments 不是 JSON | json.loads 抛异常，循环中断 | 解析错误变成结构化观察 |
| name 不是 add | execute 仍按加法执行 | 工具注册表与未知工具拒绝 |
| a 是字符串 | 可能拼接或 TypeError | 明确输入 schema |
| 模型重复请求 | 一直重复执行直到循环上限 | 调用身份、预算、幂等策略 |
| 工具执行后进程退出 | 内存历史丢失 | ledger 与检查点 |
| 模型返回空 final | 当正常结果返回 | 完成语义检查 |
| 结果巨大 | 全部塞历史 | 输出与上下文预算 |
| 用户取消 | 没入口 | 终态与协作取消 |

## 任务 2：把任意字典改成明确契约

打开 `contracts.py`，阅读 ToolCall、Turn、Tool 三个数据类。数据类自动生成初始化等方法，使字段清楚，但它不自动提供全面业务验证。`ToolCall.__post_init__` 只要求 id 和 name 非空；arguments 的 JSON 验证仍在执行边界完成。不要因为使用 dataclass 或类型注解就认为运行时会自动拒绝所有错误类型。

请手工构造：

```python
from labs.agents.contracts import ToolCall, Turn

call = ToolCall("c1", "add", '{"a":17,"b":25}')
turn = Turn(calls=[call])
print(call.wire())
print(turn.message())
```

`wire()` 返回的是 Chat Completions 风格工具请求结构；`message()` 生成完整 assistant 消息。传入的 Python dataclass 不会被 JSON 自动理解，需要显式转换。`asdict(call)` 得到统一内部字段，`call.wire()` 得到提供方协议字段，它们有不同用途。内部 pending 使用前者，发给模型的 messages 使用后者。

这里可以画一张映射表：

| 层 | 名称字段 | 参数字段 | 身份字段 |
|---|---|---|---|
| 内部 ToolCall | name | arguments | id |
| Chat 工具请求 | function.name | function.arguments | id |
| 内部 pending | name | arguments | id |
| Chat 工具结果 | 不必重复工具名 | content | tool_call_id |

试着把内部 ToolCall 的 name 改成 tool_name，但故意不改 wire。你的测试应该暴露属性错误。然后改成把 name 放错到外层，代码可能不会报 Python 错，提供方却会拒绝协议。这说明类型正确、JSON 正确、协议正确和业务正确是四层不同检查。

## 任务 3：注册表怎样把名字变成函数

读 `Harness.__init__`：`self.tools={tool.name:tool for tool in tools}`。这是一个普通字典推导式。假设传入三个工具，名字分别 read_file、replace_text、test_fixture，就得到名字到 Tool 对象的映射。模型输出的 name 只允许在这个字典中查找，不允许使用 `globals()[name]`、`eval(name)` 或随意导入模块。

如果两个工具同名，字典会覆盖前者。参考实现通过比较构建前后数量拒绝重复，避免工具目录看似有两项、实际只执行后一项。请在练习中创建两个同名 add，验证构造 Harness 时失败。这个错误应当在运行前发现，而不是等模型调用后才随机表现。

再读 `Tool.definition()`。它只把名字、描述与 schema 返回给模型，Python function 不会通过网络传给模型执行。这是工具调用最常见的误会之一：给模型“注册工具”只是告诉它可请求什么；真正函数对象仍在你的进程里。远程模型没有因为看见函数说明就获得了本机 Python 内存。

## 任务 4：逐步观察 State，而不是只看最终答案

在 Python 交互环境或自己的文件中运行下面代码。每调用一次 step，停下来预测下一行：

```python
from dataclasses import asdict
from labs.agents.contracts import Turn, ToolCall
from labs.agents.runtime import Harness, ScriptedModel
from labs.agents.tools import arithmetic_tool

model = ScriptedModel([
    Turn(calls=[ToolCall("c1", "add", '{"a":17,"b":25}')]),
    Turn("42"),
])
h = Harness(model, [arithmetic_tool()])
s = h.new("trace-1", "计算17+25")

def inspect(label):
    print(label, {
        "status": s.status, "version": s.version,
        "model_calls": s.model_calls, "tool_calls": s.tool_calls,
        "messages": len(s.messages), "pending": len(s.pending),
        "inflight_model": s.inflight_model,
    })

inspect("new")
h.step(s); inspect("model call")
h.step(s); inspect("tool call")
h.step(s); inspect("final")
```

标准轨迹如下。version 数值来自当前实现的保存点，不是所有 Harness 的通用规律。

| 时刻 | status | version | model_calls | tool_calls | messages | pending |
|---|---:|---:|---:|---:|---:|---:|
| new | ready | 1 | 0 | 0 | 1 | 0 |
| 第一次模型结束 | tools | 3 | 1 | 0 | 2 | 1 |
| 工具结束 | ready | 4 | 1 | 1 | 3 | 0 |
| 最终模型结束 | finished | 6 | 2 | 1 | 4 | 0 |

第一次 step 为什么从1到3？调用前先保存 model_requested 与 inflight_model=true，调用后保存 response 和 inflight_model=false。你在 inspect 中只看到请求完成后的对象，所以看不到中间版本2，但数据库曾保存过它。故障工作簿会故意停在这个间隙。

现在打开 s.messages，逐项写出来源。user 来自目标；assistant 请求来自模型；tool 来自真实执行器；assistant final 来自第二次模型输出。`pending` 则不是另外一份聊天消息，而是执行队列，它引用同一轮提出的工具请求。两份数据角色不同，更新时必须保持一致。

## 任务 5：看懂函数内每次 return 的意义

`step` 有许多提前返回。最前面检查终态，避免 finished 或 cancelled 被继续推进。随后检查未完成模型请求标记，识别恢复后的不确定窗口。然后检查 deadline，再处理 pending 分支，最后才考虑新模型请求。

这个顺序不是任意的。已有 pending 时应先处理当前轮工具，不应该再让模型在没有观察的情况下继续生成。预算已耗尽时不得先执行工具再记录超限。用户已经取消时，即使 pending 里还放着请求，也不能继续动作。你可以通过交换两个条件制造 bug，但必须在临时练习副本里做，别把主参考代码改坏后忘了恢复。

在 pending 分支，`call=ToolCall(**state.pending[0])` 使用字典解包，把内部序列化字段重新构造成对象；`_execute` 返回输出字符串；只有输出能放入预算时才 pop pending、增加 tool_calls、追加 tool 消息并保存。为什么 output budget 失败时仍保留 pending？因为动作结果可能已在 ledger 中，系统需要让操作者理解“未交给模型的观察”，不能假装动作不存在。

在模型分支，index 取调用前计数，再把计数加1。ScriptedModel 用 index 选择响应，这让重启后的正常模型步与已保存计数一致。但若请求已预留却响应没保存，仅凭增加后的 index 继续会跳过一个响应。因此 State 的 inflight_model 标记非常重要：它让恢复停止为 uncertain，而不是静默跳到下一条脚本。这里已经由 `test_model_crash_window_does_not_skip_response` 覆盖。

## 任务 6：完整追踪一次工具执行

读 `_execute`，把每一步旁边写上“纯计算”“数据库读写”或“外部效果”。签名 hash 是纯计算；Store.start 是数据库读写；parse_json 和 validate 是纯计算；tool.function 可能是外部效果；json.dumps 是序列化；Store.complete 是数据库写入。

这一标记能帮助你识别故障窗口。假设 function 创建了工单，然后 json.dumps 因返回非序列化对象而失败：业务效果已经发生，却还没有 ledger done。由于工具声明 side_effect=true，执行器应进入 uncertain，不会把异常当作“什么都没发生”。只有执行器前置解析与 schema 验证作用域中的 ContractError，才能证明业务函数尚未调用，因此可以安全返回 invalid_arguments。一旦进入 tool.function，不论它抛 RuntimeError 还是自己使用 ContractError，都不能仅凭异常名字推断效果回滚。参考实现用两个独立异常作用域区分：副作用工具调用或输出序列化阶段的异常统一进入 uncertain；纯工具内部异常记为 tool_failed。回归测试故意让工具先产生效果再抛 ContractError，确认 ledger 保持 running，未把这次动作误标为已完成的输入错误。

参考工具都是有意设计的简单操作。扩展真实工具时，应把异常类别视为接口契约的一部分：哪些保证零效果，哪些可能部分效果，哪些能查询确认。不要在任意业务异常外统一 `except Exception: return {'ok':False}` 后就宣称恢复安全。

## 任务 7：写一个输出契约包装器

输入验证保护工具函数不接收错误参数，输出验证保护调用方不接收错误实现结果。完整参考实现位于 `advanced_solutions.output_checked`，它只允许包装纯工具。请先自己写，再对照下面核心逻辑：

```python
result = tool.function(arguments)
try:
    validate(result, output_schema)
except ContractError as exc:
    raise RuntimeError("tool output contract failed") from exc
return result
```

这里使用 RuntimeError 表达工具输出实现错误；即使内部继续使用 ContractError，执行器也会依据异常发生在调用后的作用域把它归类为 tool_failed，不会告诉模型“你传错参数了”。纯工具没有外部副作用，可以作为 tool_failed 观察处理。若包装副作用工具，输出验证失败发生在效果之后，需要不确定性处理，所以参考包装器明确拒绝副作用，而不是隐藏这一复杂度。

测试方法：构造一个应返回整数的函数却返回字符串，断言错误类别；再改成返回42，断言正常结果。这个测试比“函数能调用”更接近真实接口维护：你在下游接入新版服务时，返回类型改变会被立即发现。

## 任务 8：独立实现一个小项目

项目目标是“读取两份结构化报价，检查币种相同，计算差价并生成有来源的报告”。工具只读课程临时 JSON 文件，不调用真实商业服务。分三轮：读两份报价、计算差价、最终报告。报价至少有 source、currency、amount_minor_units、timestamp。若币种不同，系统应明确拒绝直接相减，而不是偷偷做未经授权汇率换算。

参考解题步骤如下：

1. 定义 read_quote 的 path schema 与输出 schema。输出金额使用整数最小货币单位，避免初学阶段的浮点金额误差。
2. 写纯函数 compare_quotes，先检查币种和必需字段，再计算整数差。将 source 原样带到结果，不让模型自己编出处。
3. 用 ScriptedModel 建立正常路径，验证2次读取与1次计算都在 trace 中。
4. 加三个失败夹具：坏 JSON、缺 currency、币种不同。验证每种失败都有明确观察，最终报告不会声称完成正确比较。
5. 最后才换真实模型，比较它能否自行选择动作。离线测试验证工程行为，线上测试验证规划与表达。

完整验收表应包含字段类型、权限、预算、消息配对、最终事实和来源。提交时附自己的状态表、失败轨迹、测试命令和解释，不需要 GPU。通过这项任务，才说明你能把 Harness 用在一个新问题上，而不只是运行已有42示例。

## 检查自己的理解

请口头回答：为什么 Tool.function 不发给模型？为什么 `asdict` 与 `wire` 不是一回事？为什么一次模型 step 保存两次？为什么 pending 不直接等于 messages？为什么输入错误与输出契约错误分类不同？为什么 inflight_model 不能在恢复时无脑清零？每个答案都要指向具体代码和一个测试。

如果你只能复述术语，请返回最小字典程序，在纸上展开每一条列表元素。专家级理解往往来自对这些普通对象和故障时间点的精确掌握，不来自把主循环换成更复杂框架。
