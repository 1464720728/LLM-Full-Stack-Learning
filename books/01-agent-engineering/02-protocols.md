# A02 模型协议：从网络字节到可继续执行的工具回合

A01 中的脚本模型直接返回 Python 对象，所以你还没有面对真实网络边界。真实 API 不会把一个 Python 的 Turn 对象通过网络送到你的进程里。它发送字节；客户端按照 HTTP 和 JSON 的规则解释这些字节，再按提供方的模型协议识别文本、工具请求、结束状态和用量。本章把这些层逐个拆开，最后写出可测试的适配与继续调用过程。

本章限定在客户端函数工具的基础往返。真实平台还可能提供服务端搜索、计算机操作、多模态输入、后台任务和其他专用对象；这些能力不能因为“也是 JSON”就被一个最小转换函数自动支持。课程现有在线客户端使用 Chat Completions，Responses 与 Messages 提供离线解析及续接构造。本章会清楚区分已经实现的路径、协议原理和你需要在选定服务上验证的能力。

## 第一课：协议不是 SDK 方法名

先想一件与模型无关的事：两个人交换表格，必须约定每列是什么意思。如果第一列有时是姓名、有时是金额，即使文件能打开，也无法正确处理。协议就是这种可由程序遵循的约定。网络协议不仅规定字段名字，还规定允许的状态、调用顺序、错误表达和关联方式。

一个 Python SDK 往往把若干协议步骤包装成一个方法。调用方法可能完成字段验证、JSON 编码、HTTP 请求、重试、响应解析，最后返回带属性访问的对象。这使使用方便，但也容易让初学者误以为模型就在本机函数内部运行。实际上，函数只是客户端入口；模型服务可能位于远端，也可能是你在 Linux 服务器启动的推理服务。

不要把“使用相同 SDK”当作“服务行为完全相同”的证据。客户端可以把请求发送到兼容端点，但服务支持哪些字段、哪些模型允许工具、哪些参数会被忽略，仍由服务实现决定。一个参数通过 Python 类型检查，只说明 SDK 接受了这个调用形式；服务器是否接收、是否采用、是否与另一个模型同义，需要分别确认。

也不要把“都是 HTTP JSON”当成一个协议。HTTP 负责请求与响应交换，JSON 负责表示对象、数组、字符串等值，模型协议负责把这些值解释成对话、工具或任务。相同的 JSON 对象在不同协议中可能无效。例如某协议把工具结果放在 role 为 tool 的消息中，另一协议把结果放在 user 消息内的专用块中。简单改一个 role 字符串不会自动修复其余结构。

从工程上，你可以把客户端拆成三层：传输层发送和接收字节；协议层构造请求、解释响应；运行层决定下一步调用什么。这样网络超时不会与模型返回空文本混在一起，解析错误不会与工具执行错误混在一起。分层并不是为了多写类，而是为了让每种失败有可以单独测试的位置。

## 第二课：一次 HTTP 请求的四个部分

一条请求首先需要地址。URL 可以包含协议方案、主机、端口和资源路径。方案是 https 或 http，主机指向服务，端口定位该主机上的监听入口，路径选择具体 API。客户端如果需要完整的 `/chat/completions` 地址，就不应该又自动追加相同后缀；反过来，如果 SDK 只接收 base URL，也不能随便把完整资源路径塞进去。重复路径是初学者遇到 404 的常见原因。

其次是方法。模型生成通常通过 POST 发送带结构化输入的请求；方法不是模型推理模式，而是 HTTP 的操作类别。再其次是请求头，常见的内容类型告诉服务 body 应如何解释，鉴权字段让服务识别调用者。最后是 body，它包含模型选择、消息、工具定义和其他已确认支持的选项。

```python
body = {
    "model": "YOUR_AVAILABLE_TOOL_CAPABLE_MODEL",
    "messages": [{"role": "user", "content": "请调用加法工具"}],
    "tools": [],
    "stream": False,
}
```

这个 Python 字典还不是网络请求。你必须先编码成 JSON 文本，再编码成字节，传输库才能发送。编码是表示形式的变化，不是模型推理。收到响应后则反向进行：字节解码为文本，JSON 解析为 Python 值，协议适配器再提取模型回合。把这些步骤都叫“调用模型”，会让你在错误栈出现时不知道失败发生在哪个边界。

HTTP 状态码也需要分层理解。成功的 HTTP 响应不保证输出完整，不保证工具已经执行，更不保证业务目标完成。服务可能正常返回一个表示生成被截断的对象，也可能正常返回一个拒绝回答的状态。相反，非成功状态可能来自鉴权、请求格式、限流或服务故障。不能把所有非二百响应都当成可无限重试的暂时故障。

在本课程客户端中，我们为请求设置单次超时和响应大小上限。这两个限制不是总任务预算。一个循环可以执行多个单次都未超时的请求，累计却超过用户可接受时间；一个响应可以小于大小上限，但其中工具结果仍可能超出上下文预算。A06 会把不同层的预算合并成清楚的资源策略，本章先确保你知道它们并不是同一个数字。

## 第三课：JSON 的值、Python 对象与双层编码

JSON 是一种文本格式，表示对象、数组、字符串、数字、布尔值和空值。Python 的字典、列表、字符串等可以映射到这些类型，但不是任意 Python 对象都能编码成 JSON。函数、数据库连接和普通自定义类实例不能直接作为标准 JSON 值传输。工具结果必须先整理成协议允许的结构。

下面三样东西必须分清：Python 字典、包含 JSON 的 Python 字符串、包含这个字符串的外层消息。它们在打印时可能相似，但需要经过不同边界。

```python
import json

result_object = {"value": 42}
result_text = json.dumps(result_object)
message_object = {"role": "tool", "tool_call_id": "c1", "content": result_text}
request_bytes = json.dumps({"messages": [message_object]}).encode("utf-8")

outer = json.loads(request_bytes)
assert type(outer["messages"][0]["content"]) is str
inner = json.loads(outer["messages"][0]["content"])
assert inner["value"] == 42
```

这里有两次 JSON 编码，但对象不同。第一次把业务结果编码为工具 content 字符串；第二次把包含该字符串的请求对象编码为整个 body。错误写法是先把完整 body 编码成字符串，再把这个字符串当成 body 对象编码一次，导致服务器收到一个 JSON 字符串而非预期对象。你可以通过解码一次后的类型快速判断是否多编码了一层。

反斜杠并不是业务内容凭空增加了字符，而是字符串表示需要对内层引号进行转义。不要手工逐个添加反斜杠，让 JSON 编码器负责。手工拼接在遇到中文、换行、引号和反斜杠时尤其容易出错。正确方法是先建立 Python 对象，只有在明确的传输边界执行编码。

JSON 解析成功也不代表字段有效。数字一和字符串“一”可以都是合法 JSON，却不一定符合工具参数合同。某些解析器还会接受非标准常量或重复对象键，这需要更严格的边界策略。本仓库的 parse_json 会拒绝重复键和非 JSON 数值常量，A03 会解释为什么；在这里先把“能解析”和“符合业务含义”分开。

## 第四课：把同一个动作放进三种协议

三种协议都能表达“模型请求客户端执行一个函数”，但它们使用的容器和关联字段不同。下面的表只比较这一最小公共能力，不声称覆盖所有平台功能。

| 问题 | Chat Completions | Responses | Messages |
|---|---|---|---|
| 请求出现的位置 | assistant 消息的 tool_calls | output 列表中的 function_call 项 | assistant content 列表的 tool_use 块 |
| 调用关联身份 | tool call 的 id | call_id | tool_use 的 id |
| 参数表示 | arguments 字符串 | arguments 字符串 | input 对象 |
| 客户端结果 | tool 消息 | function_call_output 项 | user 消息中的 tool_result 块 |
| 结果关联字段 | tool_call_id | call_id | tool_use_id |

容器结构的差异会直接影响你的遍历代码。Chat 的第一层可能是 choices，Responses 的输出是异构 item 列表，Messages 的 content 是异构 block 列表。异构表示列表中的元素类型可以不同：一项可能是文字，另一项可能是工具请求，另一些可能是推理或专用状态。不能默认每个元素都有同一组字段。

共同的运行逻辑仍然是：读取动作，验证参数，执行工具，按原调用身份返回结果，继续模型请求。协议适配器的价值在于把共同动作映射到内部结构，同时保留无法丢弃的协议资料。一个内部 ToolCall 可以统一表达名字与参数，但不一定足以恢复完整的下一轮请求。后面会专门说明这种“统一视图”和“可重放原件”的区别。

这些基础结构已经按提供方一手工具文档核查，链接见本卷 SOURCES。当前高级内容块可能增加额外身份字段，例如特定工具集合的分派信息；本章基础适配器没有自动支持所有新块。如果实际响应包含未支持类型，应明确拒绝或进入经过测试的专用处理路径，不能当作无关字段直接删除。

## 第五课：完整走一遍 Chat Completions 工具往返

先建立用户消息与工具定义。工具定义向模型描述允许的名称、参数结构和用途；它没有把本机 Python 函数发送到模型服务器。模型服务收到的是说明数据，你的实际函数仍在客户端进程中。即使定义写得非常详细，服务也不会因此自动获得本机文件访问能力。

```python
messages = [{"role": "user", "content": "请用add计算17加25"}]
assistant_message = {
    "role": "assistant",
    "content": None,
    "tool_calls": [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a":17,"b":25}'},
    }],
}
payload = {"choices": [{"finish_reason": "tool_calls", "message": assistant_message}]}
```

响应里的 choices 是候选列表。本课程 from_chat 选择第一项，适用于单候选路径。如果你请求多个候选，必须另外决定选哪个候选继续；不能把所有候选中的写入工具都执行一遍，再挑一个最终答案。候选生成和行动执行的成本与副作用完全不同，这个选择应在执行工具前完成。

`content` 为 None 不意味着响应无效，因为有效动作在 tool_calls 中。相反，某些响应有文本也有工具请求，不能因为已经看到一句自然语言就跳过动作部分。解析器需要看完整结构，而不是搜索第一个非空字符串。`finish_reason` 也不能忽略：若服务明确报告长度截断，你不能仅因某个工具参数片段碰巧可解析就当它已完整结束。

工具执行后，把原 assistant 消息和对应 tool 结果按顺序追加到历史，再发下一次请求：

```python
import json
from labs.agents.protocols import from_chat

turn = from_chat(payload)
call = turn.calls[0]
args = json.loads(call.arguments)
value = args["a"] + args["b"]
messages.append(turn.message())
messages.append({"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps({"ok": True, "value": value})})
assert messages[-1]["tool_call_id"] == "call-1"
```

这里把 from_chat 生成的原始消息保存回历史，而不是自己只重建 role 与 content。这样可以保留提供方需要的扩展字段。单纯从展示角度看，工具请求之外的字段可能不重要；从同协议继续调用角度看，它们可能是服务验证上下文的必要信息。不要让界面上的“只显示答案”规则污染底层的协议保存规则。

在多工具情况下，先为该轮所有请求产生结果，确保每个结果关联正确身份，再继续模型。结果的实际内容应该由对应调用的执行过程直接生成并绑定身份，而不是先收集一个无身份的结果列表，再靠两个列表的位置去 zip。只检查 ID 集合相同无法发现“c1 的内容被塞到 c2”这种业务错配，因此身份绑定必须从执行处一直保留。

## 第六课：Responses 的 item 列表与 call_id

Responses 的一个响应可以包含多个输出项，而不是只给一个最终文本字段。你应当先按 type 分派，再解释该类型的字段。对于客户端函数工具，function_call 项携带函数名、参数字符串和 call_id；函数结果用 function_call_output 项返回，并携带相同的 call_id。输出项自身的对象 ID 与函数调用的关联 ID 不应混用。

```python
payload = {
    "status": "completed",
    "output": [
        {"type": "reasoning", "id": "rs_example", "summary": []},
        {"type": "function_call", "id": "fc_example", "call_id": "c1",
         "name": "add", "arguments": '{"a":17,"b":25}'},
    ],
}
```

这个例子中，rs_example 标识一个推理相关项，fc_example 标识一个函数调用项，而 c1 是工具结果需要回应的调用身份。三个字符串不能因为都叫某种 ID 就互换。一个好的适配器会在变量名字上体现作用，例如 item_id 与 call_id，而不是把所有字段都存到一个叫 id 的局部变量里。

本仓库 `from_responses` 遍历整个 output。遇到 function_call 时创建内部 ToolCall；遇到 message 时遍历其 content，提取 output_text；遇到 refusal 时拒绝把它当正常答案；遇到 reasoning 时不把它当普通展示文本；未知 item 类型则明确报错。这个函数提供运行层需要的统一视图，但并不承担完整原始响应的存档职责。

为什么需要额外的 continuation 函数？因为统一 Turn 中只有文本和工具请求，未必包含下一轮需要的所有原始项目。`responses_continuation` 接收上一轮完整 payload 与按调用 ID 映射的结果，保留原 output 项，再附加 function_call_output。保留不透明项是为了协议继续，并不要求你的程序理解或展示其内部推理内容。

```python
from labs.agents.protocols import from_responses, responses_continuation

turn = from_responses(payload)
assert turn.calls[0].id == "c1"
continuation = responses_continuation(payload, {"c1": '{"value":42}'})
assert continuation[-1] == {
    "type": "function_call_output", "call_id": "c1", "output": '{"value":42}'
}
assert continuation[0]["type"] == "reasoning"
```

这个辅助函数返回的是一个继续调用片段，不是自动包含最初用户任务、系统规则和全部过去输入的完整 HTTP 请求。如果你采用客户端维护完整历史，应把片段追加到自己维护的输入历史。如果采用提供方支持的服务端状态引用，需要遵循该状态机制的规则。两种策略不能在没有检查的情况下混用，否则可能重复历史、遗漏指令或引用错误会话。

课程版本通过集合比较要求结果键与预期 call_id 集合一致。这能抓住缺失或额外结果，但不能单独证明结果内容正确，也不是完整的重复 ID 验证。调用 ID 唯一性还要由解析或运行层检查。多个层各自验证不同性质是合理的，但你应写清每一层究竟证明了什么，不要把一个集合相等测试称为“完整协议验证”。

## 第七课：Messages 的内容块与工具结果位置

Messages 通过 content 列表承载不同内容块。普通文本块的 type 是 text，工具请求块的 type 是 tool_use，参数通常已经以 input 对象表示。适配到本课程统一 ToolCall 时，from_messages 把 input 编码成 JSON 参数字符串，以便后续工具边界使用相同解析与验证流程。

```python
payload = {
    "role": "assistant",
    "stop_reason": "tool_use",
    "content": [
        {"type": "text", "text": "我先计算。"},
        {"type": "tool_use", "id": "u1", "name": "add",
         "input": {"a": 17, "b": 25}},
    ],
}
```

这里的 input 已经是 Python 字典表示的对象。若你再次假定它是字符串并直接调用 json.loads，就会产生类型错误。相反，Chat 的 arguments 是字符串，不能像字典一样按 a 取值。三协议适配最先要解决的就是这种结构差异，而不是先写一个复杂通用基类。

工具结果在一条 user 消息中的 tool_result 块表达。这个 user 是协议容器角色，不表示真人用户亲自填写了结果。实际观察依然来自工具执行器。正确的结果必须关联 tool_use_id，并按协议要求紧接对应工具请求回合。你不能在请求与结果之间随意插入另一条普通消息，再期待服务自动猜测关联。

```python
from labs.agents.protocols import from_messages, messages_continuation

turn = from_messages(payload)
assert turn.calls[0].id == "u1"
following = messages_continuation(payload, {"u1": '{"value":42}'})
assert following[0]["role"] == "assistant"
assert following[1]["role"] == "user"
assert following[1]["content"][0]["tool_use_id"] == "u1"
```

完整内容块必须被保存，而不只是把 text 块拼成一个字符串。假如响应包含推理相关块或其他对继续调用有意义的资料，展示层可以不显示它们，存档和重放层仍需按协议保留。对于你不理解的新工具集合或专用块，应该先核查其分派字段和返回格式，再扩展适配器。基础工具调用成功，不等于计算机操作、浏览器工具或服务端工具也已经兼容。

当前课程 continuation 辅助函数构造的是基础成功结果，未给每个输出额外传递独立 is_error 元数据。错误结果可以在内容中表达，但如果你要严格利用协议的错误标记，就需要扩展输出结构与测试。识别这样的差距是源码阅读能力的一部分：不要因为真实协议支持某字段，就假定课程辅助函数已经接入了它。

## 第八课：统一数据模型能保留什么，不能保留什么

内部统一接口通常会保留最终文本、工具名、参数和调用身份，这让 Harness 不必直接知道每个平台的字段布局。但统一意味着选择共同表达能力；如果内部模型没有某种字段，转换就可能丢信息。你必须决定哪些信息用于运行决策，哪些原样存档，哪些不支持时应当拒绝。

以 Turn 为例，它能够表达一轮文本和多个工具请求。Chat 路径还有 provider_message，用于保留原始 assistant 消息。Responses 与 Messages 的解析器没有把全部原始 payload 塞进 Turn，因此调用者必须额外保存 payload，并使用对应的 continuation helper。把三者都压成 Turn 后再仅调用 Turn.message，不能保证为三个协议重建正确历史。

这不是实现小细节，而是抽象边界。一个接口如果声称“所有模型通用”，就必须说明它支持的是哪些共同能力。如果只支持文本和客户端函数调用，应该明确拒绝图像或服务端工具；如果支持不透明扩展存储，应说明扩展只能回放给相同协议与兼容服务。透明保存不是跨协议理解，能够存一个字段也不代表能够解释它的授权或执行含义。

可以把转换结果分成三类。精确可映射的部分，例如基础函数名和关联身份，可以放进统一字段。可保存但不理解的部分，例如提供方不透明状态，可以按原协议保存而不解释。无法在目标协议表达的部分，例如特定服务端动作，应当报告不支持或进入明确的降级流程。最糟糕的选择是静默删除，然后继续对用户声称能力未变。

一个有用的思想实验是把所有输出都变成纯文本。模型最终文字还能显示，但工具身份、参数结构、错误状态、未完成标记和继续调用所需资料都可能丢失。即使你把这些信息写成自然语言，目标程序也未必能可靠恢复原结构。数据结构存在的意义就是让关联和约束可以由程序验证，而不是每次重新让模型猜测。

因此，框架迁移不能只测试“能不能聊天”。至少应测试一次工具往返、同轮多工具、错误参数、截断、拒绝、未知扩展和继续调用。每个用例对应不同的协议性质。你可以先用离线夹具覆盖这些性质，再选择一个短任务做在线验收；这样真正在线失败时，已有离线证据可以帮助缩小问题范围。

## 第九课：逐行拆解三个解析器

`from_chat` 首先取 `payload['choices'][0]`，因此它假设 choices 存在且非空。代码用异常转换把缺字段、空列表和某些类型错误变成 ContractError。这是调用者可识别的协议错误。随后检查 finish_reason，拒绝 length 与 content_filter 这两种当前实现明确处理的非正常结束。你不能从这个有限检查推出所有可能结束原因都已完整分类。

它对 message 做 deepcopy，然后遍历 tool_calls，为每项取 id、函数名和参数字符串。深复制的目的是避免调用者以后修改原 payload 时，已保存的消息跟着变化。若只是复制外层字典，内部的 tool_calls 列表与 function 字典仍可能共享。基础卷的别名知识在这里直接决定历史保存是否可靠。

返回 Turn 时，content 为空值会被转换为空字符串，而原 message 继续保留在 provider_message。这个统一视图方便循环判断是否有文本；原始视图方便同协议回放。注意二者可能在表示上不同，例如统一 text 是空字符串，原 content 仍是 null。你不应该为了让它们“看起来一致”就擅自改写原始协议资料。

`from_responses` 先检查顶层 status，只接受当前实现支持的完成状态或缺省情况，再按 item type 分派。普通 message 还需要第二层遍历，因为 message 的 content 自己也是列表。两层循环分别对应两种容器，不是多余复杂度。把 output 的第一项直接当最终文本，会在前面插入 reasoning 或 function_call 时立刻出错。

`from_messages` 按 content block type 分派，text 累积到文本列表，tool_use 转成 ToolCall，已知推理块不进入普通文本，其他类型报错。这个函数对 input 执行 json.dumps，是为了统一参数表示。由于它不是完整 JSON Schema 验证器，它不会在这里检查业务参数是否缺失或超范围；那是工具执行边界的工作。

三个函数都不是完整平台 SDK 的替代品，也没有覆盖所有异常形状、全部状态或所有媒体类型。读者的任务是学会为有限支持范围写清合同，并用反例验证，而不是看到几十行解析代码就宣布“已实现通用模型网关”。如果要扩大范围，应先增加输入夹具与预期行为，再扩展分派逻辑，避免把未知数据变成静默成功。

## 第十课：传输客户端逐行看，不让网络成为黑盒

仓库 `ChatModel` 接收 endpoint、model、key、extra 和 timeout。endpoint 是完整资源 URL，model 是当前账户实际可用并支持相关能力的模型标识，key 是鉴权秘密，extra 是已核查的提供方附加字段。把这些参数分开，是为了不把环境配置硬编码进教材或测试。

构造函数先解析 URL，检查主机、用户信息和 fragment 等组成。它要求 HTTPS，只有显式选择本机测试时才允许 localhost 的 HTTP。这个规则服务于教学客户端的明确范围：真实远端密钥请求应经过加密传输，本地开发可以使用回环地址测试。它不是一个通用网络访问控制系统，也不验证远端业务是否值得信任。

接着它检查 extra 必须是字典，并禁止 extra 覆盖 messages、tools、model 与 stream。这样调用者不能一边传入经过 Harness 管理的消息，一边通过附加配置悄悄替换历史。允许扩展并不意味着允许任意覆盖核心合同。这个做法也让调试更简单：最终请求的关键来源是唯一的。

`complete` 构造 body，编码 JSON，再建立 POST 请求。Authorization 头携带 key，Content-Type 说明 JSON 内容。`NoRedirect` 拒绝自动重定向，避免客户端在未明确检查的新地址上继续请求。它读取最多上限加一字节，用是否超过上限判断响应过大；这个技巧比先无限读取再检查长度更早限制内存使用。

最后将响应体解析为 JSON，交给 from_chat。网络库错误、JSON 错误和协议错误可能发生在不同位置；Harness 会把模型请求失败记录成错误事件。本课程客户端没有自动重试，不代表真实系统永远不应重试，而是把重试策略留在可以明确讨论失败类别和预算的位置。A06 会说明为什么不能在 SDK、传输层和 Harness 各自偷偷重试三次。

真实接入时，先用 `python -m labs.agents.live --help` 阅读参数，不需要任何密钥。只有明确添加 allow-live 才发网络请求。环境变量中的 key 仍需谨慎管理，但比把 key 写进教材、命令行参数或 Git 文件更容易避免误提交。操作记录保存端点类别、模型、时间与结果即可，不保存 Authorization 值。

## 第十一课：用能力矩阵代替“兼容”两个字

为一个后端记录能力时，不要只有一个 compatible 布尔值。你至少需要知道它支持什么输入、什么工具结构、什么流式事件、什么结束状态、什么附加参数，以及下一轮如何携带状态。一个后端可能支持普通文本与串行工具，却不支持并行工具、某类 schema 或某种推理字段。这些差异会改变系统行为。

能力矩阵可以从小表开始，每格写已验证、未验证或不支持。已验证还应附带证据，例如一个脱敏夹具和一个短在线探针的结果。未验证不是支持；没有报错也不是参数真正生效。特别是温度、采样、思考强度等选项，提供方可能接受字段但忽略其值，或者按模型映射到不同范围。不要把请求 JSON 中出现某参数当成采用了该参数的证据。

| 能力 | 最小验证 | 不能据此推出 |
|---|---|---|
| 普通文本 | 一次短回答且状态完整 | 支持工具或长上下文 |
| 单工具 | 请求、执行、结果、下一轮完整闭环 | 支持并行或复杂 schema |
| 多工具 | 同轮两个 ID 正确往返 | 结果可任意乱序或可重复 |
| 扩展字段 | 原件保存与同协议续接成功 | 可跨提供方无损转换 |
| 流式 | 分帧、终止、完整参数验收 | 断线后可自动恢复 |
| 用量 | 能解释已返回字段 | 缺字段代表零消耗 |

用量尤其容易误读。请求没有返回 usage，表示你没有得到该项数据，不等于消耗为零。缓存输入、输出、推理相关用量等字段可能有包含关系或独立口径，不能把所有数字简单相加。课程中的字符预算只是教学上容易观测的限制，不是某个提供方精确的 token 计费器。你应把估计与服务返回的观测分开记录。

对于 DeepSeek 的思考字段，当前官方文档说明带 tools 的后续请求需要按要求保留 reasoning_content；不带 tools 的路径规则不同。这个事实支持我们保存原始消息的设计，但不支持“所有模型都必须返回相同字段”的推断。选定具体模型后，再核对其参数行为和继续调用要求。[核查来源](https://api-docs.deepseek.com/guides/thinking_mode/)

升级 SDK 或换服务前，先重放已有夹具，确认本地解析行为没有变化；再在明确预算下运行最小在线探针。把实际模型标识、服务版本或可用 revision、访问日期和结果保存下来。这样以后发生差异时，你能比较是哪一层变了，而不是只记得“上个月能用”。

## 第十二课：四个协议故障现场

现场一：第一次请求能得到工具调用，第二次请求返回格式错误。先检查历史有没有保存原 assistant 工具请求，再检查工具结果是否放在对应协议要求的位置。若你把 Messages 的结果写成 role 为 tool 的消息，或者把 Responses 的 item ID 当 call_id，都是结构错误。不要先修改模型提示词；提示无法修复客户端发送的错误 JSON 结构。

现场二：模型返回了完整一句话，但适配器拒绝成功。检查结束状态是否是截断或拒绝。人读起来完整不等于协议完整，例如工具参数后半部分被截掉，前面的解释文字仍然通顺。正确处理是保留不完整状态并按策略决定是否重新请求，不能把 text 非空作为覆盖所有状态的通行证。

现场三：更换兼容端点后，第一轮正常，第二轮失败。检查是否丢掉了提供方要求保留的扩展字段；检查服务是否真正支持相同工具定义；检查是否把 base URL 和完整资源路径混用。为了定位，不要一次同时改 SDK、模型、提示词和历史格式。先固定一个变量，使用已经保存的脱敏请求结构进行比较。

现场四：同轮两次查询都成功，但答案把两座城市的天气交换。HTTP 状态与 ID 集合都可能正常，错误发生在执行器把内容绑定到身份的阶段。最小复现是让两个工具返回不同的明显值，并故意颠倒完成顺序。修复方法是让每个异步任务携带原调用对象，完成后返回调用 ID 与结果组成的整体，不能把结果收集到一个无身份列表里再按顺序匹配。

这四种故障分别属于请求结构、结束状态、兼容边界和关联语义。它们可能都在用户界面表现为“模型出错”，但修复位置完全不同。写故障报告时，保留经过脱敏的最小 payload、预期行为、实际错误与责任层，比贴一整页混杂网络和业务日志更有效。

## 第十三课：十二道协议练习与完整答案

### 练习一：证明双层 JSON 没有多编码

构造一个结果对象，把它放进 Chat 工具消息，再编码完整请求。解码外层后，content 必须仍是字符串；再解码 content 才得到业务对象。第三课给出的程序就是完整参考解。反例是对已经编码的完整请求再执行一次 dumps，此时第一次 loads 返回字符串而非字典。验收时检查类型，不要仅对比打印效果，因为转义后的字符串经常让人误判。

### 练习二：空 choices 不是空答案

调用 `from_chat({"choices": []})` 应抛出 ContractError。空列表意味着缺少本适配器需要的候选响应，而不是模型正常返回空文本。完整测试如下：

```python
from labs.agents.protocols import from_chat
from labs.agents.contracts import ContractError
try:
    from_chat({"choices": []})
except ContractError:
    pass
else:
    raise AssertionError("missing choice must be rejected")
```

这里测试的是协议结构失败。不要修改预期为返回一个空 Turn，然后让后面的循环猜测发生了什么；越接近错误来源的边界给出明确分类，后面的代码越容易保持简单。

### 练习三：保留原件但隔离别名

创建一个含 tool_calls 的 Chat payload，解析后修改原 payload 中的函数名。Turn 保存的 provider_message 不应随之改变。参考测试如下，既验证扩展保留，也验证深复制，而不是只检查外层对象是否不同。

```python
from labs.agents.protocols import from_chat
p = {"choices": [{"message": {"role": "assistant", "content": None,
     "custom_field": {"version": 1}, "tool_calls": [{"id": "c1",
     "function": {"name": "add", "arguments": "{}"}}]}}]}
t = from_chat(p)
p["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "changed"
assert t.provider_message["tool_calls"][0]["function"]["name"] == "add"
assert t.provider_message["custom_field"]["version"] == 1
```

这个测试只能证明解析时保存了一份独立原件。Turn.message 当前返回原件的外层字典副本，嵌套结构的后续所有权仍应由调用方管理。不要把某一次 deepcopy 夸大成整个系统永远不可能发生别名修改；应明确在哪个边界复制、哪些对象之后只读。

### 练习四：截断必须可见

为 Chat 构造 finish_reason 为 length，为 Responses 构造 status 为 incomplete，为 Messages 构造 stop_reason 为 max_tokens。三者都不应作为正常完成回合继续执行。完整测试如下：

```python
from labs.agents.protocols import from_chat, from_responses, from_messages
from labs.agents.contracts import ContractError
cases = [
    (from_chat, {"choices": [{"finish_reason": "length", "message": {"content": "看起来完整"}}]}),
    (from_responses, {"status": "incomplete", "output": []}),
    (from_messages, {"stop_reason": "max_tokens", "content": []}),
]
for parser, payload in cases:
    try:
        parser(payload)
    except ContractError:
        continue
    raise AssertionError("incomplete response was accepted")
```

测试还应保留一个正常完成对照，以免你写出“无条件抛错”的解析器仍通过所有坏输入测试。有效的验收既需要证明拒绝坏输入，也需要证明接受支持范围内的好输入。

### 练习五：区分两种 ID

Responses 项同时包含 id 为 item-9、call_id 为 call-2。工具结果必须使用 call-2。若你使用 item-9，continuation 的集合检查应拒绝。这个题的答案不是记住哪个字段更长，而是理解两个身份的对象不同：一个标识输出项，一个标识函数调用关联。给变量起不同名字，再在结果构造函数只接收 call_id，能从接口层减少混淆。

### 练习六：缺结果与多结果都要拒绝

若上一轮请求 c1、c2，结果映射只有 c1，属于缺失；映射包含 c1、c2、c3，属于额外结果。两个 continuation helper 都应拒绝这两种情况。正确结果映射恰好包含 c1、c2。集合相等是必要条件，但不是充分业务验证：你仍需确认每个内容由对应调用产生。这个区别要求你同时写结构测试与执行绑定测试。

### 练习七：Messages 中的角色含义

工具结果为什么放在 user 消息里，能不能因此认为它来自真实用户？答案是不能。角色字段是协议安排，真实来源由工具执行记录与宿主身份决定。你应把网页或工具输出保留在专用结果块中，不能为方便拼接而升级成系统规则。这个练习连接 A07 的来源与信任问题：数据的内容可以包含命令语气，但不会因此变成授权指令。

### 练习八：未知类型不能悄悄丢弃

构造 Responses output 中的 `{"type":"future_action"}`，from_responses 应拒绝。完整参考测试如下。若新增类型只是可选元数据，未来可以有经过核查的透明保存策略；但在尚不知道它是否代表动作、状态或必要上下文之前，删除后继续运行不可靠。

```python
from labs.agents.protocols import from_responses
from labs.agents.contracts import ContractError
try:
    from_responses({"status": "completed", "output": [{"type": "future_action"}]})
except ContractError:
    pass
else:
    raise AssertionError("unknown item must not disappear silently")
```

### 练习九：写一个响应概况器

概况器只记录结构和计数，不记录工具参数或敏感文本。完整参考实现如下。它适合排查“收到什么类型”，不适合证明业务结果正确。

```python
def responses_shape(payload):
    items = payload.get("output", [])
    counts = {}
    for item in items:
        kind = item.get("type", "missing")
        counts[kind] = counts.get(kind, 0) + 1
    return {"status": payload.get("status"), "items": len(items), "types": counts}

assert responses_shape({"status": "completed", "output": [
    {"type": "function_call", "arguments": "private"}
]}) == {"status": "completed", "items": 1, "types": {"function_call": 1}}
```

这里没有把原 payload 直接写进日志，是因为调试所需信息通常比全部内容少。若需要保存脱敏夹具，应有独立的脱敏和人工检查步骤，不能指望日志系统自动判断每个字段是否私密。

### 练习十：实现完整的离线 Responses 往返

下面把解析、工具执行和续接串起来，保留初始输入与全部 output 项。它使用传输替身，整个例子不会联网。把 transport 换成真实 HTTP 前，还需要完成鉴权、超时、错误分类和显式费用授权等边界；本例不冒充已经在线验证。

```python
import json
from copy import deepcopy
from labs.agents.protocols import from_responses, responses_continuation

class FixtureTransport:
    def __init__(self):
        self.requests = []
    def __call__(self, body):
        self.requests.append(deepcopy(body))
        if len(self.requests) == 1:
            return {"status": "completed", "output": [
                {"type": "reasoning", "id": "r1", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "add",
                 "arguments": '{"a":17,"b":25}'},
            ]}
        return {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "42"}]}
        ]}

def offline_responses_loop(transport):
    history = [{"role": "user", "content": "请计算17加25"}]
    for _ in range(3):
        payload = transport({"model": "fixture", "input": deepcopy(history)})
        turn = from_responses(payload)
        if not turn.calls:
            if not turn.text.strip():
                raise ValueError("empty completion")
            return turn.text
        outputs = {}
        for call in turn.calls:
            if call.name != "add" or call.id in outputs:
                raise ValueError("unsupported or duplicate call")
            args = json.loads(call.arguments)
            if set(args) != {"a", "b"} or any(type(args[k]) is not int for k in args):
                raise ValueError("bad arguments")
            outputs[call.id] = json.dumps({"value": args["a"] + args["b"]})
        history.extend(responses_continuation(payload, outputs))
    raise RuntimeError("model round budget")

transport = FixtureTransport()
assert offline_responses_loop(transport) == "42"
second_input = transport.requests[1]["input"]
assert second_input[0]["role"] == "user"
assert second_input[1]["type"] == "reasoning"
assert second_input[-1]["call_id"] == "c1"
```

参考实现最重要的一行是 history.extend，而不是替换 history。它保存初始任务，再添加完整输出和结果。如果你只把 continuation 当作下一轮全部输入，初始任务可能丢失。另一方面，如果你同时使用服务端会话状态，又无条件重发全部历史，也可能重复内容。历史所有权必须在一种策略下定义清楚。

### 练习十一：构造有损转换报告

选择一个包含普通文字、函数请求和不透明推理项的响应，转换成只含 text 的字符串。报告至少列出丢失的调用身份、参数结构、项目顺序和不透明续接资料。参考结论应是“文本展示可以完成，原协议继续调用无法仅从该字符串可靠恢复”，不能笼统写“转换失败”或“完全兼容”。把损失精确落到字段与行为上，才有助于决定是否可以接受降级。

### 练习十二：制定最小在线验收表

表中记录请求模型、明确端点、调用时间、是否启用工具、是否流式、预期调用数、实际结果和结束状态。密钥不进入表。先用离线 fixture 跑过解析、续接、截断与多工具用例，再明确启动一次加法请求。参考通过条件是工具请求结构正确、客户端确实执行、结果正确关联、下一轮完整结束；仅收到 HTTP 成功状态不够。若真实服务拒绝参数，先按其官方文档调整能力声明，不能在客户端静默删字段后继续声称测试相同能力。

## 本章项目：建立一个可审查的协议夹具目录

为三种协议各保存正常文本、单工具、多工具、截断、拒绝和未知类型六类合成夹具。文件名表达协议与场景，文件内容不包含真实 key 或私人对话。为每个夹具写预期统一视图和继续调用片段；对不支持的场景写明确异常类型。运行同一组测试时，输出应该能说明哪个协议性质失败，而不只是返回一个总失败数字。

然后故意制造三次迁移错误：把 call_id 换成 item ID，删除原始推理项，把 Messages 结果放进错误角色。每次先预测哪个检查会失败，再运行。最后把错误改回并保留回归用例。这样你得到的不只是几个 JSON 样例，而是一份可以随 SDK、端点和模型升级反复执行的协议合同。

本章通过门槛是：不借助 SDK 自动 runner，你能解释一次请求从 Python 对象到字节再到 Turn 的每一层，能正确构造三种基础工具续接，能指出统一接口无法表达的资料，并能用最小反例证明转换边界。下一章进入工具合同，把“协议上有一个参数对象”继续细化成“这个动作在业务上允许执行”。

## 补充小课：请求身份、会话身份、调用身份和业务身份

一个真实系统经常同时出现四种 ID。HTTP 请求身份标识一次网络尝试，便于与服务日志关联；会话或运行身份把多个轮次归在同一个任务下；工具调用身份关联某次模型提出的动作与返回结果；业务身份标识订单、工单或文档等现实对象。它们有不同生命周期，不能因为都使用字符串就复用同一个字段。一次业务更新可能经历多次网络尝试，一个 Agent 运行可能处理多个业务对象，一轮模型响应又可能包含多个工具调用。

假设运行 run-A 要更新工单 ticket-7，模型提出调用 call-3，HTTP 请求 request-9 在返回前断线。重试的网络请求可能获得 request-10，但仍然是在处理 ticket-7 的同一个业务意图。若你把 request-10 当成新的业务幂等键，工具可能再次更新；若你把 ticket-7 永远当调用 ID，又会阻止同一工单后续合法的不同操作。正确设计通常需要把运行、调用、业务操作键和网络尝试分别记录，关联起来，而不是寻找一个万能 ID。

在脱敏日志中，你可以记录这些身份之间的映射，而不记录完整参数。例如某事件说明运行 A 的调用三通过网络尝试九访问业务对象七，结果未知；恢复后通过一次只读对账确认业务版本已增加。这样的轨迹能够帮助你回答“这次重试是不是同一件事”，也能把协议问题连接到 A10 的持久化问题。反过来，一条只有“请求失败”的日志无法告诉你哪些任务受到影响，也无法判断有没有重复副作用。

对于多租户系统，身份还必须带作用域。两个用户都可能拥有名为 call-1 的调用，不能把它们放在一个全局去重集合里互相冲突。本课程 ledger 使用 run_id 与 call_id 组成身份，体现了最小作用域；生产系统可能还需要租户、工具版本和业务操作类型。你应先写清哪个范围内必须唯一，再选择 UUID、计数器或其他生成方式。随机字符串并不会替你决定正确的唯一性范围。
