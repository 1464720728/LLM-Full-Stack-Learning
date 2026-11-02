# A05 流式输出：把字节、事件与动作分别做对

本章以你会写循环、字典、函数和异常处理为起点。先不要连接任何模型服务。我们把一段离线数据切成任意大小的字节块，逐层恢复它表达的两个工具调用；随后故意删除字节、重复片段、改变顺序、取消消费，观察故障在哪里发生。最后得到的能力不是让文字逐字出现，而是知道哪些内容仅供展示，哪些内容已构成可以验证的完整请求，以及哪些失败必须使整轮生成失效。

本章使用 `labs/agents/streaming.py` 的两个教学组件：`sse_events` 与 `ToolDeltaAssembler`。前者实现明确受限的 SSE 数据帧读取，后者组装已经由提供方适配器解释过的工具增量。它们没有网络连接、重连、鉴权和提供方完整事件解释器。`labs/agents/live.py` 当前仍是非流式 Chat 请求；不能因为本章通过了离线测试，就声称仓库具有全部厂商的实时流式客户端。

## 第一课：从网络块还原 Unicode 字符

### 一次读取为什么不等于一条消息

想象寄一篇文章。作者把文章分段，印刷厂把段落排成页，邮寄人员把纸张装进信封，收件人一次拿起几个信封。这些分界由不同人决定。一封信里可能装半段，也可能装很多段。网络读取块的意义类似“一次拿到多少东西”，它不负责保持应用消息的边界。读取到的块很整齐，仅说明某次实验碰巧如此，不能把这种现象当作接口保证。

在我们的问题里至少有六种边界。模型内部产生的 token 是生成单位；Unicode 字符是程序理解文字的单位；UTF-8 字节是传输的编码单位；HTTP 客户端一次交付的 bytes 是读取单位；SSE 空行标记事件边界；具体模型协议再在事件中表达文字片段、工具参数片段或完成通知。工具参数本身又是一段 JSON 字符串，它的结束和所在事件的结束也不一致。只有清楚地分层，才能解释为什么一个事件是合法 JSON，里面的参数却仍然不是合法 JSON。

下面的表不是术语记忆题。每一行都对应一类真实错误和一个可以单独测试的函数。

| 层次 | 输入 | 输出 | 完成条件 | 典型错误 |
|---|---|---|---|---|
| 字节解码 | 任意 bytes 块 | 完整字符片段 | 编码状态足够 | 汉字被切开后丢失 |
| SSE 分帧 | 字符片段 | 完整 data 字符串 | 空行 | 两个事件粘成一个 |
| 事件解释 | data 字符串 | 有类型的协议事件 | JSON 与事件契约成立 | 把心跳当参数 |
| 调用组装 | 某调用的增量 | 完整工具请求 | 正式完成且字段齐全 | 两个调用参数串线 |
| 请求验证 | 工具请求 | 可执行参数 | schema、权限、预算通过 | 部分参数提前执行 |
| 业务执行 | 已授权参数 | 结果或不确定状态 | 业务提交和结果确认 | 重连后重复创建 |

初学时常想“把错误全部包在一个 try 里就不会崩”。程序不崩和程序正确不同。如果 UTF-8 解码失败后继续尝试 JSON，你可能把数据损坏伪装成模型参数错误；如果缺少完成事件却返回已有文本，界面可能展示一份并未写完的操作建议。异常应保留发生层次，最外层再决定如何给用户解释。内部可以区分 `decode_error`、`truncated_frame`、`incomplete_generation`，用户界面不需要暴露所有实现细节，但日志必须足够复现。


### 亲手看见一个汉字跨越三个块

Python 的 `str` 表示文本，`bytes` 表示整数序列。把文本变成 bytes 叫编码，把 bytes 恢复为文本叫解码。同一个字符串用不同编码可能得到不同字节序列；本章只使用 UTF-8。ASCII 范围字符通常占一个字节，一个常用汉字通常占三个 UTF-8 字节。注意“通常”不等于所有字符；某些符号占四个字节，屏幕上的一个视觉字符还可能由多个 Unicode 码点构成。因此 `len(text)` 不是显示宽度，更不是网络字节数。

```python
import codecs

raw = "鲸".encode("utf-8")
assert list(raw) == [233, 178, 184]
decoder = codecs.getincrementaldecoder("utf-8")()
observed = [decoder.decode(raw[:1]), decoder.decode(raw[1:2]), decoder.decode(raw[2:])]
assert observed == ["", "", "鲸"]
assert decoder.decode(b"", final=True) == ""
```

第一行导入标准库，第二段把文字转换成三个确定的字节。`getincrementaldecoder` 返回一个解码器类，末尾括号创建有内部状态的实例。第一次收到前导字节时，它知道字符还没结束，所以输出空字符串，同时保留这个字节。第二次仍未足够，也输出空字符串。第三次收齐后才输出一个完整汉字。空输出不表示丢包，也不表示流结束，只表示这次输入还不能形成一个可交付的字符。

`final=True` 是一条额外信息：“以后不会再有字节了”。如果解码器仍保存残缺字符，必须在这里报错。没有这一步，程序可能一直把最后两个字节当作等待中的内容，随后安静地结束，造成尾部字符丢失。严格结束检查是流式程序里经常遗漏的一半逻辑：正常循环只处理已到达的数据，结束分支处理永远不会到达的数据。

这里不能用每块单独的 `.decode()` 代替增量解码器。单独调用不共享残留状态；第一块只有 `233` 时会触发 `UnicodeDecodeError`。也不要用 `errors="ignore"` 或替换字符来“修复”协议载荷。用户界面的容错显示和动作参数的忠实读取是不同任务。工具名称或路径中丢掉一个字符，可能从“不可识别请求”变成另一个恰好存在的资源。损坏输入应该失败，不能凭空拼出看似可用的动作。

你可以为调试记录字节长度和错误偏移，但不要默认把整段原始响应写入日志。模型响应可能包含用户输入、检索内容或工具参数。一个足够定位 UTF-8 问题的最小日志可以只包含流编号、读取块序号、字节数、异常类型和经过脱敏的短十六进制片段。日志和完整重放样本的保留权限应分别设计。

## 第二课：SSE 子集与分帧器逐行实现

### SSE 数据帧的边界与教学子集

SSE 是一种文本事件流格式。我们关心 `data:` 行和空行：一个事件可以有多条数据行，空行结束当前事件，数据行之间按换行连接。`data: hello` 和 `data:hello` 都能表达 `hello`；冒号后可移除至多一个空格，不能任意 `.strip()`，因为内容开头和末尾的空格可能属于数据。JSON 对象外的空白通常无关，但普通文本事件的空白可能直接影响显示与代码缩进。

```python
from labs.agents.streaming import sse_events

raw = "data: 第一行\r\ndata:  第二行\r\n\r\ndata:[DONE]\n\n".encode("utf-8")
events = list(sse_events([raw[:7], raw[7:19], raw[19:]]))
assert events == ["第一行\n 第二行", "[DONE]"]
```

第二条数据行在冒号后有两个空格，解析器只移除一个，所以输出仍以一个空格开头。`\r\n` 表示 CR 和 LF 两个控制字符；本实现先以 `\n` 拆行，再去掉末尾 `\r`。即使网络块正好在这两个字符之间切开，也不会凭空产生额外空行，因为没有看到 `\n` 时仍保留缓冲内容。

必须把教学子集写清楚。WHATWG 的事件流规则还涉及事件类型、事件编号、重连时间、不同换行形式以及浏览器 EventSource 的连接行为。我们的函数忽略非 `data:` 行，不提供事件类型与最后事件编号，不实现浏览器重连；只以换行字符 `\n` 分行，因此不支持只有 `\r` 的行结束；没有冒号的 `data` 字段也不会按完整标准解释。课程还选择把末尾未结束的帧报为错误。这是主动采用的严格输入契约，不应描述成完整标准对所有客户端的要求。

为什么不直接把实现补到标准的全部范围？本章目标是让你掌握层次和状态，不是重新造一个浏览器。生产系统可以选经过测试的流式库，但选择库之后仍要知道它怎样处理残缺事件、注释、重连和事件编号。如果产品需要自定义事件类型，当前函数只返回字符串就不够，应返回例如 `Event(data, event_type, id)` 的结构，且对空字段、重复字段和编号范围补上测试。增加字段不是多写两个字典键，而是扩展接口契约和恢复逻辑。

还有一个容易混淆的点：`[DONE]` 不是 SSE 的固定保留词。它在某些应用协议里表示结束，在另一些协议里完成通知可能是带有类型和状态的 JSON 事件。SSE 解析器只负责交付字符串；是否把某个字符串解释为终止，应由提供方事件解释器决定。将所有提供方流都硬编码为“等待 `[DONE]`”，会把合法完成误判成断线，也可能忽略真正的错误事件。


### 逐行读分帧器，找到每一份暂存状态

先看核心结构，省略的只有容量检查，下一课再单独解释。这里的函数用于说明，与仓库实现同样保留三份状态：尚未分行的字符、当前事件的数据行、解码器内部残缺字节。

```python
def explain_frames(chunks):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    data_lines = []
    for chunk in chunks:
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            if line == "":
                if data_lines:
                    yield "\n".join(data_lines)
                data_lines = []
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
    buffer += decoder.decode(b"", final=True)
    if buffer or data_lines:
        raise ValueError("incomplete final frame")

assert list(explain_frames([b"data: a\n", b"data: b\n\n"])) == ["a\nb"]
```

`buffer += ...` 只追加已解码字符，残缺字节留在 decoder 中，所以不能把这两份缓冲合成同一个字符串。`while` 而非 `if` 很重要：一块网络数据可能包含上百个完整行，只处理一行就会把其余完整事件滞留到下一次读取，若下一次读取一直不来，用户便觉得程序卡住。`split(..., 1)` 只拿出最左边一行，右侧残余仍可能包含许多行，循环负责继续处理。

去掉末尾回车后才判断空行，使 CRLF 形式的空行也成为空字符串。空行里有一个空格就不是空行，本实现不会把它当作帧结束；这属于字面协议边界，不能用“看起来空白”代替严格规则。`if data_lines` 判断的是列表里有没有数据字段，不是数据内容是否非空；一条空的 `data:` 行会产生包含空字符串的列表，因此应该交付空字符串事件。代码必须用 `event is not None` 判断有没有事件，不能用 `if event` 把空事件吞掉。

`yield` 让这个函数成为生成器。调用函数时不会立刻把所有输入读完；调用方每次请求下一项，它才推进到下一个事件。这有利于控制消费速度，但并不自动使整条网络链路有界。底层 HTTP 库可能提前读取，调用方也可能立刻用 `list(...)` 把全部结果收集到内存。生成器只是一个局部按需接口，容量保证必须检查每一层。

结束时有两个独立检查。decoder 的最终解码检查编码是否完整；随后 `buffer or data_lines` 检查应用帧是否完整。例如 `b"data: ok\n"` 的编码和单行都完整，却缺少结束事件的空行，因此第二个检查失败。反过来，输入最后一个 UTF-8 字符只有一半，即使我们暂时看不到数据行，也应在编码层失败。把两个检查合并成“有没有最后一个换行”会漏掉错误。

| 输入到达后 | decoder 残留 | buffer | data_lines | 已交付 |
|---|---|---|---|---|
| `data: a` | 空 | `data: a` | 空 | 无 |
| `\n` | 空 | 空 | `[a]` | 无 |
| `data: b\n` | 空 | 空 | `[a,b]` | 无 |
| `\n` | 空 | 空 | 空 | `a\nb` |

手算这张表时不要跳步。每一次输入先解码，再分行，再解释行，最后才可能交付事件。能把状态按顺序写出来，你就能自己定位“数据明明已经到了，为何界面没更新”的问题：是没有构成字符、没有构成行、没有构成事件，还是事件已经交给下游但没有刷新界面。

## 第三课：容量上限到底限制了什么

仓库分帧器同时检查单行字符长度和事件累计数据字符长度。单行限制保护一个永远不发送换行的输入；事件限制保护不断发送短数据行却永远不发送空行的输入。两者不能互相替代。一条百万字符的行可能尚未进入 `data_lines`，仅检查事件大小会漏掉它；十万条短行分别不超限，却能让当前事件无限增长，仅检查单行也会漏掉它。

这里的单位是 Python 字符数量，不是 UTF-8 字节，也不是模型 token。`size` 累加各条数据内容的长度，没有把连接这些行时插入的换行算入，所以它是“数据字符累计上限”而不是最终 `join` 字符串的严格大小上限。另外单行检查发生在去掉回车之前，且包含 `data:` 前缀，边界测试要按真实实现解释。如果业务明确要求最终事件最多多少字节，应该在输出前检查编码后的总长度，或者维护精确字节预算，不能给当前参数换个名字就宣称完成了要求。

一个读取块可能很大，里面却全是短小合法事件。以整个 chunk 长度判断单事件过大会误拒绝正常输入。相反，底层一次给出巨大的 chunk，即便我们随后逐行消费，也已经把这个 chunk 放入内存，因此应用分帧限制并不能代替 transport 的合理读取大小与总体响应限制。生产约束可以分别写成：每次读取最多若干字节，未完成行最多若干字符，单事件最多若干字节，整轮最多若干事件和总字节，工具参数最多若干字符。各项限制对应不同资源，不能只保留一个含糊的 `max_size`。

失败后的行为同样重要。当累计参数超过上限时，当前 assembler 可能已经追加了超限片段再抛异常；不能捕获后删掉一点继续复用它，除非接口明确提供回滚或恢复语义。本课程采用“本轮组装失败即废弃实例”的原则：错误对象不能继续成为可执行结果。这样每个实例对应一轮生成，成功或失败都清晰结束，下一次尝试建立新实例，避免残留状态泄漏。

容量上限还应进入可观测记录。只记录“解析失败”看不出究竟是用户合法大请求超过产品限制，还是服务端异常输出。记录限制名称、配置值、实际已见大小和请求标识即可，不必记录被拒绝的全部内容。对用户的提示应说明可以缩小输入或重新请求，而不是让模型用更小参数偷偷绕过安全边界。

## 第四课：工具调用有两个身份，参数有自己的缓冲

假设同一轮模型决定同时查询北京和上海天气。流式协议可能给它们分配位置编号零和一，每个位置又有调用编号 `call_bj`、`call_sh`。位置编号用于在本轮生成中找到相应累加器，调用编号用于以后把工具结果关联回请求。前者通常只在一轮输出内部有意义，后者需要原样保存在对话中。把位置零当作永久业务身份，会在下一轮也出现位置零时发生冲突；拿调用编号去做数组下标，又会遇到它尚未到达的早期片段。

仓库 `ToolDeltaAssembler` 用 `parts[index]` 保存三个字段：`id`、`name`、`arguments`。其中参数是字符串，而不是已经解析的字典。每个增量先根据 index 找到自己的槽位，然后追加到自己的字符串。不同调用之间可以交错，同一个调用内部的参数片段必须保持协议定义的顺序。网络中的字节顺序保持了，并不意味着应用层随意把事件交给并行任务仍能保持处理顺序。

```python
from labs.agents.streaming import ToolDeltaAssembler
from labs.agents.contracts import parse_json, ContractError

assembler = ToolDeltaAssembler()
assembler.feed(1, call_id="call_sh", name="weather", arguments='{"city":')
assembler.feed(0, call_id="call_bj", name="weather", arguments='{"city":"北')
assembler.feed(1, arguments='"上海"}')
assembler.feed(0, arguments='京"}')
calls = assembler.finish()
assert [c.id for c in calls] == ["call_bj", "call_sh"]
assert [parse_json(c.arguments) for c in calls] == [{"city":"北京"}, {"city":"上海"}]
```

注意第一条到达的是位置一，但 `finish()` 按位置排序，所以结果先北京后上海。这种排序不意味着实际工具一定按此顺序执行，只是生成一个稳定的请求列表；A09 再讨论工具执行顺序和结果回填。若模型协议提供的是不同类型的条目编号，而不是这里假设的工具位置，适配器必须先映射，不能直接把任何叫 `index` 的字段塞进来。

逐行看 `feed` 的关键语义：它首先要求 `type(index) is int` 且非负，所以 `True` 被拒绝，尽管 Python 里布尔值属于整数的子类。随后仅在新位置出现时检查调用数量上限，已有位置继续追加不会增加调用数。`setdefault` 在不存在时建立槽位，已经存在则返回原槽位。元数据只有非 `None` 才更新；重复给出相同名称或编号可以接受，重复给出不同值则失败。最后参数追加并检查长度。这一顺序解释了为何不能在异常后继续信任这个实例：前面字段可能已经更新。

这里的 id 和 name 按“完整元数据”处理，不支持把工具名称分成 `wea` 和 `ther` 两次追加。真实提供方的流式字段可能要求不同组装方式，因此必须在协议适配层核查。课程不能把通用类名当作兼容声明。参数是增量追加，名称是完整值验证，这两个字段即使都属于字符串，合并规则也不同。你写数据结构时应把这种规则写在字段文档和测试中。

`finish()` 首先检查每个调用具有非空编号和名称，再把完整参数交给严格 JSON 解析器。参数语法正确不代表符合工具 schema；它只完成了 A03 的第一道门。随后检查所有调用编号唯一，避免两个不同槽位都叫 `call_1` 导致工具结果无法明确关联。这个检查最好在任何工具执行之前完成，否则第一个调用已经产生副作用，才发现第二个编号重复，整轮就难以安全恢复。

| 到达事件 | 位置零参数 | 位置一参数 | 允许执行 |
|---|---|---|---|
| 上海第一片 | 不存在 | `{"city":` | 否 |
| 北京第一片 | `{"city":"北` | `{"city":` | 否 |
| 上海第二片 | 未完成 | `{"city":"上海"}` | 仍需正式完成与验证 |
| 北京第二片 | `{"city":"北京"}` | 已完整 | 仍需正式完成与验证 |
| 合法完成事件 | 完整并验证 | 完整并验证 | 按权限与预算决定 |

## 第五课：看起来完整的 JSON 为什么仍不能执行

设参数前缀已经是 `{"amount":100}`。能成功调用 `json.loads`，只证明目前字符串是一个合法 JSON 文档。它没有证明服务器已经承诺这就是最终请求，也没有证明后面没有协议错误、拒绝事件或未完成状态。若工具涉及付款，你不能让“解析器碰巧成功”代替生成协议的终态。应用必须等到为这个调用或整轮响应定义的完成条件成立，再使用最终组装结果。

这里有两种不同的提前工作。可以提前把草稿展示给用户，也可以预先读取不会改变业务状态的公开资料；但真正的动作提交必须绑定最终参数和当前权限。即使预读取没有副作用，它也可能泄漏信息或消耗预算，所以仍需要权限与资源控制。低延迟优化不意味着绕开边界，而是明确哪一部分可重复、可丢弃、不会改变结果。

审批尤其容易受到流式草稿影响。用户在页面看到“创建一个任务”并点击允许后，后续参数又增加了另一个收件人，旧批准不能自动覆盖新内容。正确做法是先形成稳定的动作对象，计算摘要，展示具体目标与影响，再把批准绑定到该摘要。参数、工具版本或受影响资源发生变化，需要重新判断批准是否仍有效。A11 会给出这种摘要绑定的完整状态流。

对文本答案也要区分“正在展示”与“最终认可”。流式文本在生成途中可能自我修正，引用尚未输出完，甚至最终因长度限制中止。界面可以持续显示，但应保留生成中、完成、中断等状态。不能因为页面已经有一段看似完整的结论，就把整轮评测记为成功。评测需要最终状态和业务验收，显示字数只是过程观测。

实践中可以把流处理分成三份独立产物：供界面展示的临时片段，供诊断保留的受控事件记录，供后续模型和工具使用的最终消息。临时片段允许逐步更新，最终消息只在终止条件通过后提交。这样网络中断时，用户仍能看到已有内容，同时系统不会把半条工具请求写成已完成对话。持久化草稿也可以，但必须带 `incomplete` 标记，恢复时不能与正常 assistant 消息混用。

## 第六课：结束、断线、拒绝和截断是四个不同结果

先定义一个教学状态机：`receiving` 表示仍接收事件；`completed` 表示协议确认正常结束且所有组装检查通过；`failed` 表示收到明确错误或本地解析失败；`incomplete` 表示连接结束但没有得到所需终止事实。后两者可能都需要告诉用户重试，却不能在日志中合并，因为重试风险和定位方式不同。明确错误可以包含服务端拒绝原因；不完整则意味着没有足够信息判断末尾发生了什么。

一个字节流读到 EOF，仅表示这个连接不再提供字节。可能是服务端正常关闭，也可能是代理中断或客户端取消。只有协议层的完成通知能告诉你生成是否正常结束。反过来，收到完成通知之后是否立即停止读取，也必须遵守提供方协议：有的协议还会发使用量或清理事件。不要在不知道事件顺序契约时用一个通用的 `break` 丢掉尾部信息。

长度截断也不同于正常完成。若输出预算耗尽，已有文字可能仍有参考价值，但工具参数可能缺字段或缺右括号。本章不做“自动补齐括号然后执行”。补齐只能制造语法完整，无法恢复模型未生成的语义。正确处理是标记未完成，必要时向用户展示已生成的文字，并根据成本和安全策略决定是否发起新的完整请求。重试生成与继续同一个生成，是两个需要分别设计的行为。

拒绝事件同样不应被拼入普通工具参数。某些提供方会在内容中给出拒绝信息或特殊停止原因。协议适配器应识别这些情况，把它们映射成明确结果；否则通用 JSON 代码可能把拒绝描述当成不合法参数，再错误地提示模型“修复 JSON”。这会把真正的语义结果误诊为格式问题。A02 的非流式适配已经区分部分拒绝与截断，流式适配也必须补上等价规则。

下面的终态表要求你同时看数据和证据，而不是只看文本是否存在。

| 已收到内容 | 完成证据 | 解析状态 | 应记录结果 |
|---|---|---|---|
| 完整文本 | 正常完成 | 通过 | completed |
| 部分文本 | 无，连接结束 | 已收片段可显示 | incomplete |
| 完整参数字符串 | 无 | JSON 通过 | incomplete，禁止执行 |
| 任意内容 | 明确拒绝 | 协议有效 | refused 或业务定义的失败 |
| 参数片段 | 长度截断 | JSON 失败或未完成 | incomplete |
| 完整文本 | 正常完成 | 尾部协议冲突 | failed，需诊断 |

## 第七课：重放、去重与自定义序列号

追加式增量不是天然幂等操作。对同一个片段执行两次 `buffer += piece`，结果不同于执行一次。假设原本片段是 `{"x":` 与 `1}`，第一片重复后变成 `{"x":{"x":1}`，通常会解析失败；更危险的是数字片段重复，原来 `1` 变成 `11`，可能仍然合法，甚至通过范围较宽的 schema。不能用“最终 JSON 能解析”证明没有重放。

去重需要稳定事件身份。你自己控制两端时，可以给每个增量分配连续序号，并规定同一序号必须对应完全相同的载荷。接收方保存已见序号的规范化内容；重收到相同序号和相同内容时忽略，内容不同则报冲突，跳过下一个期望序号则报缺口。这样协议错误会在组装前暴露，而不是等业务结果奇怪时才发现。`advanced_solutions.SequencedAssembler` 就是这一自定义实验的参考实现。

```python
import json

class OrderedDeltaReceiver:
    def __init__(self):
        self.next_seq = 0
        self.seen = {}
        self.inner = ToolDeltaAssembler()

    def feed(self, seq, payload):
        if type(seq) is not int or seq < 0:
            raise ContractError("invalid sequence")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if seq in self.seen:
            if self.seen[seq] != canonical:
                raise ContractError("sequence payload changed")
            return
        if seq != self.next_seq:
            raise ContractError("sequence gap")
        self.inner.feed(**payload)
        self.seen[seq] = canonical
        self.next_seq += 1

    def finish(self):
        return self.inner.finish()

receiver = OrderedDeltaReceiver()
first = {"index":0, "call_id":"c1", "name":"count", "arguments":'{"n":'}
receiver.feed(0, first)
receiver.feed(0, first)
receiver.feed(1, {"index":0, "arguments":"1}"})
assert parse_json(receiver.finish()[0].arguments) == {"n":1}
```

规范化 JSON 的目的是让键顺序不同的字典表达同一个载荷，不是证明它们在所有业务语义上相等。这里参数片段本身仍是字符串，里面空格和字符顺序全部保留；不能为了去重把片段先解析再排序，因为片段往往还不是完整 JSON。`allow_nan=False` 避免不标准数值悄悄进入自定义协议。这个类也没有独立验证 payload 的全部字段类型，调用方必须满足 `inner.feed` 的输入契约，生产边界仍需严格 schema。

这段实现把“成功应用增量”和“记录序号”放在一个同步函数里。若 `inner.feed` 抛异常，序号不会推进，但 inner 可能已发生局部修改，所以同样应废弃本轮接收器。若你需要在进程崩溃后恢复，内存字典当然不够；需要把增量记录与组装状态的提交协调起来，或者只持久化原始事件并从头确定性重放。序号去重解决的是重复输入，不会自动解决持久化原子性。

真实厂商是否提供可恢复序列号、事件编号是否跨连接稳定、服务端是否支持从指定位置继续，必须查看该协议的官方定义。本实验不能据此声称“所有 SSE 都可断点续传”。浏览器 SSE 的事件编号机制和模型请求的计费、生成状态也不是同一件事。没有服务端恢复契约时，重新打开请求可能创建一轮全新生成；新旧片段不能继续拼到同一个工具调用里。

## 第八课：背压让慢消费者把压力传回上游

如果每秒产生一百个事件，消费者只能处理十个，无界队列会每秒增加九十个元素。起初只是界面落后，随后内存增长，再随后进程被终止。给队列设置上限后，生产者在队列满时等待，这种等待使上游暂时减少读取，称为背压。它不是让下游变快，而是让系统在可承受容量内表达“现在处理不过来”。

```python
import asyncio

async def bounded_demo():
    queue = asyncio.Queue(maxsize=2)
    consumed = []
    peak = 0
    sentinel = object()

    async def producer():
        nonlocal peak
        for number in range(8):
            await queue.put(number)
            peak = max(peak, queue.qsize())
        await queue.put(sentinel)

    async def consumer():
        while True:
            item = await queue.get()
            try:
                if item is sentinel:
                    return
                consumed.append(item)
                await asyncio.sleep(0)
            finally:
                queue.task_done()

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    try:
        await asyncio.gather(producer_task, consumer_task)
        await queue.join()
    finally:
        for task in (producer_task, consumer_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
    return consumed, peak

consumed, peak = asyncio.run(bounded_demo())
assert consumed == list(range(8)) and peak <= 2
```

`await queue.put` 在队列满时让出控制权，并不阻塞整个事件循环。消费者 `get` 取走一个对象，`task_done` 表示这项处理已经结束；用 `finally` 保证遇到哨兵或异常也配对。哨兵是一个仅用于本次运行的对象，通过 `is` 比较，避免把普通数据值误当结束。`queue.join` 等待所有已放入项目完成处理，和等待生产者不再生产不是同一条件。

清理逻辑必须同时照顾两端。如果消费者抛异常，生产者可能正等队列腾空间；只等待生产者自然结束会永久挂住。`finally` 取消尚未完成的任务，再 `gather(..., return_exceptions=True)` 回收它们，避免留下后台任务。这里的 `sleep(0)` 只是显式让出调度，不用实际毫秒时长证明正确性，所以测试不依赖机器速度。A09 会进一步解释任务取消为何需要被等待。

背压有传播范围。应用队列满了，HTTP 客户端内部可能仍缓存一段数据；操作系统也有接收缓冲，服务端也可能继续生成。因此“队列最多两个元素”只证明这一层的容量，不证明服务端立即停止产生费用。用户取消时应尽快关闭响应、停止新动作、清理任务，并把已产生的成本按能获得的使用量记录。不能把前端停止渲染当作后端请求已经取消。

## 第九课：把故障变成可重复的输入

先研究四个小事故。调试的第一步不是增加打印，而是构造最短失败样本；第二步是预测应在哪一层失败；第三步才运行。这样你修复的是被准确命名的错误，而不是碰巧让原始大输入通过。

事故一：中文偶尔消失，英文始终正常。某实现对每块 bytes 调用 `.decode(errors="ignore")`，测试数据全部是英文。英文单字节掩盖了分块错误，中文跨块时两边都被丢弃。修复是一个连接使用同一个增量解码器，并在结束时做最终检查。验收不是只测“鲸”这一个字符，而是把包含中英文、换行和四字节符号的短样本在每个字节位置切开，输出必须与一次输入完全一致。若只测试字符边界，你仍没有覆盖真正的故障。

事故二：两个工具各自的参数都很简单，组装结果却出现北京和上海混杂。开发者把所有 `arguments` 片段写进全局字符串，认为同一条 HTTP 连接只会处理一个调用。修复是按协议提供的调用位置分别维护状态，再在完成时检查调用编号唯一。验收还应故意先发送位置一，再发送位置零；否则只测试按位置连续输出，错误实现可能侥幸通过。

事故三：工具在断线后仍执行了。开发者每收到一个片段就尝试解析 JSON，成功便执行。测试只覆盖最后片段恰好使 JSON 完整的路径，没有覆盖“参数完整但完成通知丢失”。修复是把正式完成作为必要条件，解析成功只是其中一项。验收给出一个完整参数事件但不发结束事件，函数必须抛出不完整错误，工具调用计数保持零。这里最重要的断言是副作用没有发生，而不只是返回了某个异常。

事故四：重连后把数量一变成十一。重放片段恰好是一位数字，重复追加后的 JSON 仍然合法。schema 允许一到一百，所以也通过。修复不是加强 JSON 解析，而是明确重连协议：使用有稳定身份的事件去重，或废弃旧一轮从头生成，并以业务幂等标识保护后续动作。验收需要一个重复片段仍可形成合法 JSON 的反例，这比只测重复左大括号更有说服力。

这些事故分别发生在编码、调用关联、终态与输入身份四个层面。把它们统统叫“流式不稳定”无法指导修复。故障报告至少应写输入、期望输出、实际输出、最早违反的契约和最小回归测试。真实服务的问题可以先保存脱敏后的事件样本，再在离线环境重放；这样不需要反复付费，也不依赖服务端每次恰好生成同样片段。

## 第十课：十二道有完整解答的递进练习

练习一，穷举两个块的所有切点。给定短 SSE 数据，要求任何字节切分都不改变结果，包括零长度第一块和零长度第二块。答案如下。`range(len(raw)+1)` 比 `range(len(raw))` 多覆盖最后一个边界；空块也是合法输入，不应被误认为 EOF，因为 EOF 由迭代结束表达。

```python
sample = "data: 鲸🙂\r\n\r\ndata: end\n\n".encode("utf-8")
expected = ["鲸🙂", "end"]
for cut in range(len(sample) + 1):
    assert list(sse_events([sample[:cut], sample[cut:]])) == expected
assert list(sse_events([bytes([byte]) for byte in sample])) == expected
```

练习二，穷举三个块的切点。目的不是追求随机数量，而是覆盖一个字符、一个回车换行、一个事件分隔分别跨不同边界的组合。短样本的二次循环足够便宜。答案要求 `left <= right`，因此允许中间块为空；不限制严格小于，避免漏掉空读取片段。大样本不宜做所有组合，可在确定边界集上组合，再补固定种子抽样。

```python
small = "data: 中\n\n".encode("utf-8")
for left in range(len(small) + 1):
    for right in range(left, len(small) + 1):
        chunks = [small[:left], small[left:right], small[right:]]
        assert list(sse_events(chunks)) == ["中"]
```

练习三，证明空数据事件不能用真值判断丢弃。答案使用 `data:` 后立刻换行，随后空行结束。得到的列表长度必须是一，而不是零。解释：没有事件用 `None` 表示，空字符串仍然是一个事件载荷。把“内容为空”和“没有内容项”分开，是后面处理空工具结果、空检索集合时同样需要的习惯。

```python
assert list(sse_events([b"data:\n\n"])) == [""]
assert list(sse_events([b": heartbeat\n\n"])) == []
assert list(sse_events([b"data:  x  \n\n"])) == [" x  "]
```

练习四，让三类截断在正确层失败。第一种残缺 UTF-8 应产生编码异常，第二种缺少帧结束空行应产生合同错误，第三种 SSE 完整但工具参数不完整应在组装结束时报合同错误。答案没有用一个宽泛的 `except Exception` 将三种结果混在一起；分类断言能防止以后修改代码时把错误移动到错误的层次。

```python
def expect_error(kind, operation):
    try:
        operation()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")

expect_error(UnicodeDecodeError, lambda: list(sse_events([b"data: \xe9"])))
expect_error(ContractError, lambda: list(sse_events([b"data: ok\n"])))
broken = ToolDeltaAssembler()
broken.feed(0, call_id="c", name="t", arguments='{"x":')
expect_error(ContractError, broken.finish)
```

练习五，设计三个调用的交错到达，保持每个调用内部顺序。答案故意采用位置二、零、一的第一片顺序，随后换另一种顺序补齐。验收比较解析后的业务参数，另外比较输出调用编号的稳定顺序。只比较调用数量不能发现串线；只比较 JSON 可解析也不能发现值被放错位置。

```python
three = ToolDeltaAssembler()
for index in (2, 0, 1):
    three.feed(index, call_id=f"c{index}", name="echo", arguments='{"n":')
for index in (1, 2, 0):
    three.feed(index, arguments=f"{index}" + "}")
finished = three.finish()
assert [item.id for item in finished] == ["c0", "c1", "c2"]
assert [parse_json(item.arguments)["n"] for item in finished] == [0, 1, 2]
```

练习六，检查元数据冲突和调用编号冲突。前者是同一个槽位的名称中途变化，后者是两个槽位最终拥有相同编号。这两种问题发生时间不同：前者 feed 时即可拒绝，后者在 finish 统一检查。答案分别建立新实例，避免一个失败实例残留状态影响后续断言。测试中重用损坏对象，会使第二次失败看似覆盖了新问题，实际仍然来自第一次损坏。

```python
metadata = ToolDeltaAssembler()
metadata.feed(0, call_id="a", name="read", arguments="{}")
expect_error(ContractError, lambda: metadata.feed(0, name="write"))
duplicate = ToolDeltaAssembler()
duplicate.feed(0, call_id="same", name="read", arguments="{}")
duplicate.feed(1, call_id="same", name="read", arguments="{}")
expect_error(ContractError, duplicate.finish)
```

练习七，写出“重放后仍是合法 JSON”的反例。答案让数字片段 `1` 重复到达，最终数量从一变十一。这个反例提醒你，语法校验、schema 校验与传输去重各自解决不同问题。即便 schema 设置最大值一百也无法发现这里的重复，因为十一确实属于允许范围。

```python
replayed = ToolDeltaAssembler()
replayed.feed(0, call_id="r", name="buy", arguments='{"quantity":')
replayed.feed(0, arguments="1")
replayed.feed(0, arguments="1")
replayed.feed(0, arguments="}")
assert parse_json(replayed.finish()[0].arguments) == {"quantity":11}
```

练习八，为自定义序号接收器证明三种行为：相同事件可重复，不同载荷复用编号失败，序号缺口失败。答案复用本章的 `OrderedDeltaReceiver`。这里没有实现乱序缓存，因为本协议要求按序交付；如果你增加缓存，就必须另设缓存容量、等待缺口的超时以及结束时缺口检查，不能只把报错改成字典暂存。

```python
ordered = OrderedDeltaReceiver()
piece = {"index":0, "call_id":"s", "name":"echo", "arguments":"{}"}
ordered.feed(0, piece)
ordered.feed(0, dict(piece))
expect_error(ContractError, lambda: ordered.feed(0, dict(piece, name="other")))
expect_error(ContractError, lambda: ordered.feed(2, {"index":0, "arguments":""}))
assert ordered.next_seq == 1
assert len(ordered.finish()) == 1
```

练习九，分别触发调用数量上限和参数长度上限。答案使用很小的配置，让边界清楚可见。上限等于当前大小时允许，再多一个才拒绝。注意调用数量按不同槽位计数，已有槽位多次 feed 不增加数量。超限后不再使用该实例，符合本章失败即废弃的约定。

```python
limited = ToolDeltaAssembler(max_calls=1)
limited.feed(0, call_id="only", name="x", arguments="{}")
expect_error(ContractError, lambda: limited.feed(1, call_id="extra", name="x", arguments="{}"))
short = ToolDeltaAssembler(max_argument_chars=2)
short.feed(0, call_id="short", name="x", arguments="{}")
assert short.finish()[0].arguments == "{}"
expect_error(ContractError, lambda: short.feed(0, arguments=" "))
```

练习十，给单行与多行事件各写一个上限反例。答案选择上限十；第一种有十一位数据加前缀，单行超限，第二种每行自身不超限，但三条四字符数据累计十二超限。这个练习验证你知道限制检查的对象，而不是只找到某个能抛异常的大字符串。测试上限通常应包含恰好等于、刚好超过和多块累积三个位置。

```python
expect_error(ContractError, lambda: list(sse_events([b"data: 12345678901\n\n"], max_event_chars=10)))
expect_error(ContractError, lambda: list(sse_events([b"data: 1234\ndata: 5678\ndata: 9012\n\n"], max_event_chars=10)))
assert list(sse_events([b"data: 1234\ndata: 5678\n\n"], max_event_chars=10)) == ["1234\n5678"]
```

练习十一，区分显示片段和最终提交。要求显示过程可以看到 `{"x":1}`，但没有完成事件时提交集合仍为空。下面使用一个纯内存对象表达两条路径，不模拟任何提供方。答案把完成作为显式方法；仅追加文本永远不触发提交。生产代码还需在 commit 前加入 schema、授权和预算，不能把这个最小类直接接到副作用工具上。

```python
class DraftView:
    def __init__(self):
        self.display = ""
        self.committed = []
        self.closed = False
    def append(self, text):
        if self.closed:
            raise ValueError("already closed")
        self.display += text
    def complete(self):
        if self.closed:
            raise ValueError("already closed")
        value = parse_json(self.display)
        self.committed.append(value)
        self.closed = True

draft = DraftView()
draft.append('{"x":1}')
assert draft.display and draft.committed == []
draft.complete()
assert draft.committed == [{"x":1}]
```

练习十二，测试有界队列消费者异常后两端都被回收。答案让消费者读到第二项时失败；生产者可能仍在等待放入。`finally` 同时取消并等待两者，最后确认任务全部 done。这里不要用“等待一秒看是否退出”作为主要证明；任务状态与异常传播是更直接的断言。真实程序还要关闭网络响应对象，这个离线例子只证明异步任务清理。

```python
async def failure_cleanup():
    queue = asyncio.Queue(maxsize=1)
    async def producer():
        for n in range(100):
            await queue.put(n)
    async def consumer():
        assert await queue.get() == 0
        assert await queue.get() == 1
        raise RuntimeError("consumer failed")
    tasks = [asyncio.create_task(producer()), asyncio.create_task(consumer())]
    try:
        try:
            await asyncio.gather(*tasks)
        except RuntimeError as exc:
            assert str(exc) == "consumer failed"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert all(task.done() for task in tasks)

asyncio.run(failure_cleanup())
```

这些练习的共同验收不是“代码运行没有红字”，而是你能解释每个断言对应哪条合同。特别是故障测试，应先确认被测试操作确实执行到了预期层。例如为了测试工具参数 JSON 截断，SSE 必须本身完整；如果 SSE 已经坏了，参数组装根本没发生，得到异常也不能证明参数检查有效。构造测试时一次只破坏一个边界，随后再写跨层组合测试。

## 第十一课：完成一个离线流式工具前端

小项目采用我们自己定义的事件格式，而非冒充任何厂商协议。每个 SSE data 是一个 JSON 对象，`kind=delta` 携带 seq 与工具片段，`kind=complete` 宣告本轮结束。事件结构使用 A03 的有限 schema 验证，参数完成后再验证工具 schema。项目只返回经过验证的调用，不执行业务函数；这样无论在哪一层失败，都不会产生提前副作用。

```python
from labs.agents.contracts import validate, object_schema

PAYLOAD = object_schema({
    "index":{"type":"integer", "minimum":0},
    "call_id":{"type":"string", "minLength":1},
    "name":{"type":"string", "minLength":1},
    "arguments":{"type":"string"},
}, required=["index", "arguments"])
DELTA = object_schema({
    "kind":{"type":"string", "enum":["delta"]},
    "seq":{"type":"integer", "minimum":0},
    "payload":PAYLOAD,
})
COMPLETE = object_schema({"kind":{"type":"string", "enum":["complete"]}})
ADD_ARGS = object_schema({
    "a":{"type":"integer", "minimum":-1000, "maximum":1000},
    "b":{"type":"integer", "minimum":-1000, "maximum":1000},
})

def receive_custom_stream(chunks):
    receiver = OrderedDeltaReceiver()
    completed = False
    for text in sse_events(chunks):
        event = parse_json(text)
        if type(event) is not dict:
            raise ContractError("event must be object")
        if completed:
            raise ContractError("event after completion")
        if event.get("kind") == "delta":
            validate(event, DELTA)
            receiver.feed(event["seq"], event["payload"])
        elif event.get("kind") == "complete":
            validate(event, COMPLETE)
            completed = True
        else:
            raise ContractError("unknown event kind")
    if not completed:
        raise ContractError("generation incomplete")
    calls = receiver.finish()
    if not calls:
        raise ContractError("expected at least one call")
    for call in calls:
        if call.name != "add":
            raise ContractError("tool not allowed")
        validate(parse_json(call.arguments), ADD_ARGS)
    return calls

def encode_events(events):
    return "".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events).encode("utf-8")

custom_events = [
    {"kind":"delta", "seq":0, "payload":{"index":0,"call_id":"sum","name":"add","arguments":'{"a":17,'}},
    {"kind":"delta", "seq":1, "payload":{"index":0,"arguments":'"b":25}'}},
    {"kind":"complete"},
]
wire = encode_events(custom_events)
validated = receive_custom_stream([bytes([b]) for b in wire])
assert parse_json(validated[0].arguments) == {"a":17, "b":25}
expect_error(ContractError, lambda: receive_custom_stream([encode_events(custom_events[:-1])]))
expect_error(ContractError, lambda: receive_custom_stream([encode_events(custom_events + [{"kind":"complete"}])]))
```

先读事件 schema。payload 的属性表允许编号、名称、位置和参数，但 required 只有位置与参数，因此后续片段可以不重复元数据；`additionalProperties=False` 由 `object_schema` 提供，防止拼错字段被静默忽略。完整调用所需的编号与名称由 `finish()` 统一检查。事件 schema 的工作是验证每个增量的形状，组装器的工作是验证跨事件的关系，两者不能互相替代。

`completed` 是单轮布尔状态。收到完成后并不立即 return，而是继续消费输入，若后面还有数据事件就拒绝，符合这个自定义协议“一次完成且最后出现”的约定。请不要把此规则套到所有提供方；有的提供方允许完成之后还有使用量事件，需要在其适配器中明确接受。这里选择严格小协议，是为了让结束行为可以完整测试。

随后检查至少有一个调用，这是本项目“工具前端”的业务目标。通用模型适配器当然可以正常返回纯文本，因此不应把这个约束塞进通用 SSE 函数。名称白名单再确认只允许 `add`，参数 schema 则拒绝缺字段、额外字段、布尔数值或超范围整数。完成后返回 ToolCall，而不是直接算出和，使调用方仍能在真正执行前加上预算、权限和日志。接口设计的价值在于让后续边界有地方插入。

项目的完整状态变化是：建立空接收器，收到零号事件建立槽位，收到一号事件补齐参数，收到完成事件关闭生成，读完流确认没有额外事件，检查所有调用完整性，再验证业务参数。任何一步失败都没有调用工具函数。若你要加入执行，应该先验证所有调用，再根据业务依赖决定顺序或并发；不能边遍历验证边执行，否则第二个请求验证失败时，第一个已经提交。

## 第十二课：从离线项目走向真实适配器的检查表

现在可以规划真正的流式适配器，但仍不需要发起付费请求。选择一个提供方的一个明确端点，把官方事件类型列成表：哪种事件创建输出条目，哪种携带文字增量，哪种携带参数增量，哪种宣布某调用完成，哪种宣布整轮完成，哪种报告使用量，哪种表示错误。为每种事件准备最小脱敏 fixture，优先用官方示例和你合法获得的响应，标明文档读取日期。不要让同名字段掩盖协议差异。

适配器输出应该是课程内部明确的事件结构，而不是把提供方对象原样交给所有业务组件。例如可以输出 TextDelta、CallStarted、ArgumentsDelta、CallFinished、ResponseFinished 和 UsageObserved。内部类型不必与任何厂商字段一一同名，它的目标是为上层提供稳定语义。不过不能为了统一而丢掉继续对话所需的原始内容，例如推理相关条目或协议要求的签名块；统一展示视图和可续接原始消息应分别保存。

网络层还需要至少三种时间约束。连接超时限制建立连接的等待；读取空闲超时限制多久没有任何新字节；总 deadline 限制整轮从开始到结束的时间。持续收到心跳可能避免空闲超时，却仍不产生有效进展，因此某些应用还需要“有效事件停滞”监测。它应依据协议语义计算，不能把任意字节都当成任务进展。A06 将继续分析这些时钟为什么不能只用一个 timeout 参数替代。

观测指标也要明确时间点。请求开始到首字节，首字节到首个可显示文字，首个文字到最后文字，全部工具参数完整到执行开始，是不同延迟。用户感觉慢可能来自建立连接，也可能来自模型先生成工具调用而没有文本，或界面刷新批次太大。测量时应记录单调时钟差值，不拿系统日期相减判断短时延迟；系统时间调整可能使结果出现负数。

最后考虑取消与费用。取消本地任务通常只能证明本地停止等待，不能保证远端尚未产生额外 token。若使用量尾部事件没有收到，成本记录应标记未知或估算，而不是填零。若工具动作已经开始，流式请求取消也不会撤销已完成的业务提交。取消、生成中断、工具回滚是三个不同契约，必须在界面和状态里分清楚。

通过本章的门槛是：你能拿一段任意切块的 UTF-8 SSE 样本，画出每层残留状态；能在调用交错、事件重复、缺口、截断和取消时预测准确结果；能说明为什么参数完整仍可能禁止执行；能独立写出一个有终态、容量限制和完整失败测试的离线事件前端。做到这些以后再接真实接口，网络问题才不会和你自己的组装错误混在一起。

本章标准与编码依据：WHATWG《Server-sent events》以及 Python 标准库 `codecs` 的增量编解码接口，核查日期为 2026-11-01。课程实现的严格 EOF 行为和受限字段支持在正文已单独声明，不能当作完整标准实现。参考链接：[SSE 标准](https://html.spec.whatwg.org/multipage/server-sent-events.html)、[Python codecs](https://docs.python.org/3/library/codecs.html)。

## 第十三课：算法复杂度也是流式正确性的一部分

你可能已经发现示例频繁执行字符串追加。Python 字符串不可变，追加通常需要构造新字符串。当一个很长参数被拆成许多小片段时，反复复制已有前缀可能产生远大于最终长度的工作量。若最终长度为一万，每次只追加一个字符，累计复制量近似一加二直到一万，而不是仅复制一万次字符。教学代码通过较小参数上限控制规模，生产实现应结合实际长度和分块分布评估，不能因为功能测试通过就认定适合无限长响应。

一种改进是把参数片段放入列表，同时维护累计字符数量，在完成时一次 `"".join(parts)`。这避免每次追加都复制整个前缀，但引入另一种资源开销：每个列表元素都有对象和引用开销。一个字符一个片段的恶意输入即便总字符数有限，也可能创建大量小对象。因此除了字符预算，还可以限制片段数，或在积累到适当大小后合并。选择策略应依据总数据量与事件频率，而不是笼统地认为列表永远更省内存。

分行缓冲也有类似问题。当前实现每次 `split` 都创建残余字符串，超大块里包含大量行时可能反复复制长尾。完整工程可以维护读取位置、按块扫描或使用经过验证的流解析库。优化之前要保留分块不变性测试：无论输入怎样切块，事件结果必须一致。很多“提高速度”的修改会在 `\r` 与 `\n` 分属两个块时漏掉回车，或者把空行误计为两个结束事件，因此性能优化仍必须维护协议合同。

更细的性能测试应分别测总字节固定、块数增加，以及块数固定、单事件长度增加。第一组更容易暴露每块固定开销，第二组更容易暴露前缀复制问题。结果应报告输入大小、事件数、块数和机器环境，不能只给出一个“每秒多少 token”，因为这个解析器根本不知道模型 token 边界。离线 microbenchmark 也不等于端到端 API 延迟；网络和模型生成可能占主要部分。

如果你希望做一个可维护的优化任务，先固定三项不变量：内容完全相同，错误类别不被静默吞掉，容量限制仍然在相同或更严格的边界触发。然后替换参数累加方式，运行全部切点与重放反例，再比较不同片段长度的耗时和峰值内存。只要在课程上限内性能已经足够，不必为了漂亮数字引入难以解释的共享缓冲。工程能力包括知道何时值得优化，也包括知道哪些简单实现因明确边界而足够可靠。

## 第十四课：重放文件怎样保存才真正有诊断价值

保存一个最终拼好的回答，无法重现流式问题。要重现 UTF-8 切分错误，必须保存原始字节以及块边界；要重现事件重复，必须保存事件顺序与身份；要重现取消竞争，还需要记录取消发生在什么事件之前或之后。因此“记录响应”不是一个充分规格，你需要先说明想重放哪一层，再选择最小但足够的数据。

对于编码和分帧测试，可以保存脱敏后的原始 bytes，另保存每块长度列表。恢复时按长度切开，不使用真实网络时延即可复现大多数边界错误。对于协议适配测试，可以保存完整 SSE data 字符串列表，再在测试中重新编码并穷举切分。对于工具执行恢复测试，应该保存完整调用与账本状态，而不是依赖重新生成同样的 token。三个层次的 fixture 可以互相连接，但不要混成一个含义不清的日志文件。

脱敏也会改变测试性质。若把所有中文替换成 ASCII，原先的多字节边界故障可能消失；若把不同调用编号都替换成相同占位符，又会制造本不存在的重复编号错误。好的脱敏保留需要测试的结构：字符字节宽度、字段是否为空、编号相等关系、片段顺序和长度分布。你可以重新构造完全人工的内容来保留这些属性，不需要把私人原文保存在公共仓库。

一个 fixture 还应带预期结果和来源说明。预期结果不能由被测实现现场计算，否则实现错了，测试也跟着错。可以人工写出两条期望调用，或使用独立的非流式规范对象作为对照。来源说明写清楚是课程自定义协议、官方公开示例还是脱敏实测，避免后来读者把人工事件当作真实厂商保证。若官方协议发生变化，旧 fixture 仍有价值，但应归属于对应版本和兼容范围。

下面是最小重放包的建议字段及其作用。它不要求你立即建立大型数据平台，而是让一个普通 JSON 文件足够自解释。

| 字段 | 内容 | 解决的问题 |
|---|---|---|
| fixture_schema | 重放包版本 | 后续读取器知道怎样迁移 |
| protocol | 自定义或具体端点名称 | 防止错误适配器读取 |
| encoding | 本例固定 UTF-8 | 明确字节解释方式 |
| chunk_lengths | 原始分块长度 | 复现跨块边界 |
| expected_status | 完成或具体失败类别 | 测试不只检查文本 |
| expected_calls | 最终编号、名称、参数 | 检查跨调用关联 |
| provenance | 人工或公开来源说明 | 避免夸大兼容性证据 |

重放包不应默认保存 API 密钥、Authorization 头、公司服务地址或完整用户身份。大多数解析测试不需要这些信息。把它们删掉不会降低测试质量，反而让 fixture 可以安全进入版本控制和持续集成。如果确实需要私有诊断样本，应把访问控制与保留期限放在专门流程中，不能混入公开教材的示例目录。

## 第十五课：流式状态怎样与持久化衔接

A04 已经区分内存状态和已提交快照。流式场景会把这个差距放大：页面可能显示了几百个片段，而持久化快照仍停留在“模型请求进行中”。进程此时崩溃，恢复后不能因为浏览器曾显示过文字就假设完整 assistant 消息存在。浏览器展示、服务端内存组装和数据库提交是三份状态，需要明确哪一份是恢复依据。

最简单的安全方案是只持久化请求开始与最终完成，中间片段只用于展示。优点是写入量小、最终消息保持原子；代价是崩溃后丢失中间草稿，且模型请求结果可能不确定。课程 Harness 的 `inflight_model` 正是在请求前保存一个意图标志，恢复时遇到未完成请求会停止为 uncertain，而不是默默重发。要支持流式，并不是把每个片段直接塞入正式 messages；应该先设计独立的草稿状态与完成提交。

另一种方案是持久化原始事件，再在恢复时重放组装。它允许保留更多诊断信息，但要求事件有稳定顺序和完整性标记。若日志末尾只有半条记录，读取器需要识别并拒绝或截去未提交记录；若完成事件已经记录但最终消息未写入，可以确定性重建消息。这里的关键是明确定义提交点，而不是“每来一片就写文件”这么简单。文件缓冲、刷新和进程崩溃之间也存在窗口。

恢复后是否能继续原来的远端生成，是另外一个问题。你可以完整重建本地已收片段，但服务器未必允许从断点继续。没有提供方恢复支持时，已收事件只用于诊断和展示，不应让本地等待一个永远不会继续的连接。重新请求需要建立新生成身份，把旧草稿标记为中断；如果旧生成可能已经发出完整工具请求，仍要通过调用账本和业务幂等规则防止重复动作。

本地持久化还会影响背压。如果每个片段都同步写盘，磁盘成为消费者的一部分，队列可能经常满；如果异步批量写入，则界面已显示和日志已持久化之间存在更大窗口。两种选择都可以合理，但产品必须接受相应恢复语义。你可以规定最多丢失最后一批草稿，同时保证任何工具执行前所需的最终请求已经持久化。这样优化的是显示过程，动作边界仍保持严格。

最后做一个口头验收题：用户看到了完整参数，数据库里只有请求开始记录，服务端连接已经断开，是否可以直接执行该参数？答案是否定的。界面草稿没有证明协议完成，也没有证明它与数据库恢复状态一致。应把本轮标记为不完整或不确定，按照明确策略重新获取完整请求，再重新验证与授权。这个结论连接了本章的流边界、A04 的状态权威以及 A10 的恢复事务，是你以后读真实 Agent 框架时最值得追踪的一条主线。

再检查一个取消竞争：完成事件已进入内存队列，用户同时点击取消。系统必须由同一个运行拥有者决定先处理哪一个事实，并记录决定。若已经提交 completed，迟到取消应保持终态；若取消先被接受且尚未提交最终消息，应停止后续动作，必要时保存只读草稿。不能让界面线程写 cancelled、消费线程又写 completed，最后由写入速度决定结果。状态版本检查可以发现冲突，但不能代替产品对先后关系的定义。课程运行时已经保证迟到或重复 cancel 不覆盖既有终态，真实流适配器接入时应保留这项不变量，并增加“完成前取消”和“完成后取消”两条独立回归。测试应通过可控事件门闩安排顺序，不靠短暂 sleep 猜测哪条协程先运行。

因此，展示层可以尽快响应，动作层必须等待完整证据；两层使用不同提交规则，并通过明确状态连接。
