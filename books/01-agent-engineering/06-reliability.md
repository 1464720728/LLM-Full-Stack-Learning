# A06 可靠性：失败之后，先判断事实，再决定行动

本章把“失败了重试一下”拆成可实现、可审计的决策。你将从一个两次失败第三次成功的普通函数开始，逐步处理等待时间、任务截止时间、次数和费用预算、取消、程序崩溃以及副作用结果未知。每个例子都在 CPU 上离线运行，不需要真实 API，不需要 GPU。学习目标是你能对一次失败回答五个问题：发生在哪个阶段，已经确认了什么，仍然不知道什么，下一步是否安全，还有没有资源执行下一步。

这里的可靠性并不等于成功率百分之百。外部服务可能不可用，用户可能取消，任务可能超出预算。可靠系统应在可完成时完成，在不可完成时诚实停止，在结果未知时保留证据和对账入口，而不是用重复请求制造表面上的成功。本章反复使用 A04 的状态和 A03 的工具合同，请在开始前确认你能区分“工具尚未调用”“工具已经返回”“结果已经记账”三种时点。

## 第一课：异常证据与重试的三道门

### 异常是观测，不是业务事实的完整描述

Python 异常告诉你某段控制流没有正常返回，但不自动告诉你外部世界是否改变。例如函数先创建工单，再把结果转换为 JSON；如果转换失败，调用方看见异常，但工单已经存在。另一函数在检查输入时就失败，工单从未创建。两者可能都抛出 ValueError，然而重试风险完全不同。异常名称不足以判断是否回滚，必须结合异常发生阶段与工具副作用声明。

把一次调用分为准备、发送、远端处理、接收、解析、提交六个阶段会更清楚。准备阶段包括参数 schema 和本地权限检查，失败通常能证明还未调用工具。发送阶段可能只发送了部分请求，客户端未必知道远端是否接受。远端处理结束后，响应仍可能在网络中丢失。接收完整响应后，本地解析和持久化仍可能失败。只有把阶段记清楚，才能理解“超时”为什么有时可以重试，有时必须对账。

| 失败位置 | 已知事实 | 不能推断的事 | 通常下一步 |
|---|---|---|---|
| 参数验证前后 | 工具尚未开始 | 模型下次一定会修好 | 返回结构化输入错误 |
| 请求发送过程中 | 本地没有正常结果 | 远端一定没收到 | 按幂等与查询能力判断 |
| 工具提交后响应丢失 | 本地结果未知 | 业务没有发生 | 查询业务操作编号 |
| 输出序列化失败 | 工具函数已执行 | 执行结果可安全重复 | 副作用工具进入不确定 |
| 本地快照保存失败 | 内存可能已推进 | 数据库已有同样状态 | 重新读取持久状态 |
| 用户取消等待 | 不再愿意继续 | 已有动作已撤销 | 停止新动作并记录已知效果 |

永久输入错误表示同一输入原样重试通常不会改变结果，例如缺少必填字段。权限错误表示当前身份不能执行，重复请求也不会自动获得权限。临时错误表示条件可能随时间改变，例如某项短暂服务不可用，但它不等于“允许无条件重试”。结果未知表示缺少足够证据判断成功与否；它可以与临时网络错误同时出现。程序缺陷则要求修复实现，重复运行可能只会重复损坏数据。

分类最好使用结构化证据：操作阶段、明确状态码、提供方错误类型、工具副作用属性和业务查询结果。不要仅搜索异常字符串里是否包含 timeout 或 retry。错误消息面向人类，可能改语言、改格式，也可能包含用户提供的文本。把它直接作为自动行为依据，既脆弱又可能产生错误动作。


### 值得重试、允许重试、能够重试是三道门

第一次门是“有希望改变结果吗”。完全相同的格式错误反复提交通常没有希望；模型自行修正参数是一轮新的推理与请求，不是网络层重试。第二道门是“重复执行是否安全”。读取公共天气一般可以重复，创建收费订单需要稳定业务幂等键或确认此前未执行。第三道门是“剩余时间与额度是否允许”。一个安全且可能成功的请求，也可能因为任务 deadline 已到而必须停止。

可以把决策写成逻辑与：错误属于可恢复类别，操作具备重复安全性，尝试次数未耗尽，剩余 deadline 足够，预算已经预留，取消尚未生效，才进入下一次尝试。其中任何条件不满足，都应该返回清楚原因。这样做比在通用函数里捕获所有 Exception 再循环更容易审查，因为每条必要条件都可以单独写测试。

还要区分“重试同一次操作”和“重新规划一个任务”。网络层重试应保持相同业务意图与幂等身份；模型重新规划可能改变参数、调用另一工具或拆分任务。若外层只看见一个字符串错误，然后让模型随意换参数重试，旧动作可能已经成功，新动作又创建一次。对结果未知的副作用，应该先完成对账，不能把它作为普通失败交给模型自由发挥。

例如用户要求创建标题为“整理报告”的工单。第一次请求业务键为 `task-42`，响应超时。第二次若仍用 `task-42` 且服务端保证同键同意图返回原结果，重复可能安全；第二次若生成新键 `task-43`，就可能创建两张工单。即使标题一样，也不能依靠标题去重，因为用户确实可能需要两张同名工单。业务身份需要来自明确任务意图，而不是随重试次数变化。

本卷 `TransientFailure` 与 `PermanentFailure` 是教学分类，不是所有 HTTP 库都会自动抛出的异常。实际适配器必须根据官方接口合同做映射，还要保留原始错误码与请求编号用于诊断。遇到未知类别，默认向上报告比盲目视为临时更可控。后续增加新类别，应同时说明可重试条件和停止条件，而不只是给异常换个名字。

## 第二课：重试函数、退避与放大

### 从三次尝试读懂重试函数

仓库 `labs/agents/reliability.py` 的核心很短，适合逐行阅读。以下复现其主要逻辑，强调调用者负责幂等、预算和真实等待。教学实现默认 sleep 是不等待的函数，方便离线实验；如果拿它接真实服务却忘记注入 `time.sleep`，所谓退避只会算出数字而不会真正降低请求频率。

```python
import random
from labs.agents.reliability import TransientFailure, PermanentFailure, retry

attempts = []
delays = []
def eventually():
    attempts.append(len(attempts) + 1)
    if len(attempts) < 3:
        raise TransientFailure("temporary")
    return "ok"

assert retry(eventually, max_attempts=3, sleep=delays.append, rng=random.Random(7)) == "ok"
assert attempts == [1, 2, 3]
assert len(delays) == 2
assert 0 <= delays[0] <= 0.1 and 0 <= delays[1] <= 0.2
```

`operation` 是一个不带参数的可调用对象，需要的输入可通过闭包绑定。`max_attempts=3` 包括第一次调用，总共最多三次，不是第一次加三次重试。循环中的 attempt 从零开始，所以第一次失败时指数为零，等待上界为 base_delay；第二次上界翻倍。只有捕获到显式 TransientFailure 才进入重试分支，其他异常直接传播。

最后一次失败不再等待，因为已经没有下一次请求。这个细节影响用户等待时间，也能用尝试三次、等待两次的断言直接证明。成功后立即 return，不继续消耗剩余尝试次数。若第一次就成功，delays 应为空；若第一次抛 PermanentFailure，调用次数应为一，delays 也为空。这些简单边界比只测试“第三次成功”更能说明合同。

`rng.uniform(0, cap)` 选择零到上界之间的随机等待，称为全范围抖动。注入固定种子的随机源，使测试可复现；注入 `delays.append`，把本应等待的时长记入列表，使测试不必真的睡眠。这里的依赖注入不是高级框架功能，只是把会变化或产生外部效果的依赖当参数传入。它同时让时间逻辑更容易手算。

课程这个短函数没有完整验证 max_attempts 的精确类型、延迟是否有限非负，也没有 deadline 或取消支持。不要把它描述为生产策略全集。后面的小项目会给出更严格的完整版本，但保留这个短实现作为理解控制流的起点。实际工程应在入口验证配置，不能让 `True` 被当作一次尝试、负延迟传给 sleep，或 NaN 使比较失去预期行为。


### 手算指数退避与重试放大

设基础等待为零点二秒，上限为一点零秒，最多五次尝试。四次失败之后的等待上界依次为零点二、零点四、零点八、一点零。若随机样本在区间中的相对位置依次为二分之一、四分之一、四分之三、十分之一，实际等待就是零点一、零点一、零点六、零点一，总等待零点九秒。不要把上界之和误写成实际等待，也不要忘记每次请求本身还会花时间。

| 真实尝试号 | 循环索引 | 失败后上界 | 示例随机比例 | 示例等待 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.2 | 0.5 | 0.1 |
| 2 | 1 | 0.4 | 0.25 | 0.1 |
| 3 | 2 | 0.8 | 0.75 | 0.6 |
| 4 | 3 | 1.0 | 0.1 | 0.1 |
| 5 | 4 | 无下一次 | 不取随机数 | 不等待 |

为什么需要抖动？如果大量客户端同时受到同一故障影响，并固定等待一秒，它们会在一秒后一起冲击服务，形成新的波峰。随机等待把这些请求分散在时间范围内，但不能凭空增加服务容量，也不保证每个任务都成功。抖动是负载协调的一部分，仍需要最大并发、全局速率限制和重试总预算。

最危险的放大常来自多层嵌套。HTTP 客户端最多三次，模型包装器最多三次，任务队列又把整个任务最多执行三次，最坏可产生二十七次底层请求。若任务每次先执行两个成功工具再在第三步失败，外层重跑还可能重复前面两个动作，实际副作用次数不只是一条请求的乘法。给每层命名并画调用树，才能找到重试所有者。

选择唯一重试所有者并不意味着其他层永远不能恢复。网络层可以负责同一个幂等请求的短暂失败，任务层可以负责从持久状态继续未完成步骤；关键是它们不能同时把同一段已完成工作当成全新工作。每次真实尝试都应有 attempt_id，每个逻辑操作有稳定 operation_id，整个任务有 run_id。日志只记录一个“请求编号”，会把三种层次混在一起。

服务端可能建议等待时长。是否遵守、怎样解释单位、是否受最大等待和任务 deadline 限制，要依据该服务的合同。本章不硬编码所有状态码或头部语义。即使建议等待十分钟，用户任务只剩五秒，也应返回受限结果，而不是越过任务期限。拒绝继续是可靠性的一部分，并不代表重试策略失败。

## 第三课：时间必须分层，deadline 必须是同一个基准

连接超时限制建立连接的等待；读取空闲超时限制两次数据到达之间的间隔；单次调用 timeout 限制一次操作；工具执行超时限制工具工作；整个任务 deadline 限制多轮模型与工具的总时间。这些计时范围不同。十次各不超过三十秒的调用，仍可能让任务持续五分钟；不断收到心跳的流也可能永远不触发空闲超时，却没有任何业务进展。

deadline 通常保存为某个单调时钟下的绝对截止值，例如启动时 `clock()+30`。每次安排操作前计算 `remaining=deadline-clock()`，把剩余额度传给下层，不能每次重试都重新获得完整三十秒。单调时钟用于测量经过时间，不受系统日期校准影响；墙上时钟用于人类阅读时间戳。跨进程持久化单调时钟值通常不能直接在另一进程或机器继续使用，因此恢复时还要定义持久期限与本地剩余预算的转换。

```python
from labs.agents.runtime import Harness, ScriptedModel, Budget
from labs.agents.contracts import Turn

class FakeClock:
    def __init__(self, now=0.0):
        self.now = now
    def __call__(self):
        return self.now
    def advance(self, amount):
        self.now += amount

clock = FakeClock(10.0)
harness = Harness(ScriptedModel([Turn("should not run")]), [], deadline=10.0, clock=clock)
state = harness.new("deadline-equal", "test")
harness.step(state)
assert state.status == "limit" and state.model_calls == 0
```

恰好等于 deadline 时停止，因此比较使用大于等于。假时钟让你不用真的等十秒，也不用担心机器忙时断言抖动。它还能把时间安排在每个关键窗口：请求前、请求返回后、准备重试前、等待结束后。可靠的时间测试应控制事实顺序，而不是用很短 sleep 希望线程恰好切换。

课程 Harness 只在每次状态转移开始前检查 deadline。同步工具一旦开始，外层检查不能中途打断它；工具耗时超过剩余时间后仍可能返回。这个限制必须写清楚。给外层加一个时间比较，不等于获得强制终止能力。线程上的等待超时往往也只是等待方停止等，工作线程仍可能继续写入。进程可以被终止，但它已经提交到外部服务的动作仍不会自动撤销。

因此硬超时需要与执行方式配套：可协作函数定期检查取消标记，异步网络调用支持取消与关闭响应，子进程需要终止并等待回收，远端任务需要提供取消接口与状态查询。无论哪种方式，业务副作用仍需明确提交点。你应把“停止继续工作”“停止等待结果”“保证没有发生效果”分别写成三个能力，而不是统一叫 cancel。

## 第四课：预算单位与运行时扣减

### 预算是一组单位明确的账本

一个 Agent 可以只调用模型两次却花费很多，也可以调用十次而很便宜。次数预算适合阻止无限循环，但不是费用预算。输入 token、输出 token、工具调用、文件字节、数据库查询、总时间和并发槽位分别消耗不同资源。给它们统一一个 `budget=100`，会让调用方不知道一百究竟是什么，也难以解释为何停止。

课程 Budget 有 model_calls、tool_calls、tool_output_chars 和 max_argument_chars。前两个是累计计数，第三个是累计写入对话的工具结果字符，第四个是单次参数字符串上限。字符不是 token，参数上限也不是总输出预算。所有值要求精确非负整数，零表示没有额度，不表示无限。布尔值虽然能参与整数运算，但不属于合法配置；浮点数一点五次调用也没有清楚语义。

```python
for bad in (True, -1, 1.5, float("nan")):
    try:
        Budget(model_calls=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid budget accepted")
assert Budget(model_calls=0).model_calls == 0
```

估算费用时先把单位写出来。若输入单价为每百万 token 二元、输出单价为每百万 token 八元，某轮输入两千、输出五百，则费用为二千乘二除以一百万，加五百乘八除以一百万，总计零点零零八元。这个数字只是课程假设，不是任何厂商当前报价。三轮输入分别一千、两千、四千，累计输入七千，不能只拿最后一轮四千计算，因为前面的请求也消耗资源。

缓存输入、推理相关计量、图像和厂商工具可能有不同计价项，必须按所选服务当前合同读取。为了让账本可扩展，可以保存每轮原始 usage 的受控字段、估算方式、价格版本和结算状态。未知用量应标记 unknown，不能填零；估算值应标记 estimated，不能装成账单确认值。A13 会用这些区别做成本评测。

预算还有预留与结算两阶段。操作开始前按保守上限预留，操作完成后按真实使用量结算并退回未用部分。若并发任务都先检查余额再各自扣款，就可能一起通过并透支，所以检查和预留必须原子。同步单拥有者循环容易实现，异步并发时需要锁或数据库条件更新。A09 将给出完整的并发预算例子，本章先把账本语义建立起来。


### 逐行追踪运行时在哪里消耗额度

模型请求分支先检查当前 model_calls 是否已经达到上限，再把计数加一、设置 inflight_model、记录 model_requested 并保存快照，最后才调用外部模型。为什么失败也算一次？因为请求可能已经消耗服务端资源，且恢复时不能把它当作从未发生。把扣次数放在成功响应之后，会使失败重试绕过预算，甚至让一个永远失败的任务无限请求。

下面是与源码对应的简化时序，注意保存发生在外部调用之前。这里不用执行伪代码，而是用表说明每个字段何时变化。

| 时点 | model_calls | inflight_model | 已保存事实 | 外部请求 |
|---|---:|---|---|---|
| 准备第一次请求 | 0 | false | 初始任务 | 尚未开始 |
| 预留并保存后 | 1 | true | 第一次请求可能发出 | 即将开始 |
| 正常响应进入内存 | 1 | false | 仍可能是旧快照 | 已完成 |
| 完成快照保存后 | 1 | false | 响应与待办已保存 | 已完成 |
| 崩溃后只见预留快照 | 1 | true | 结果缺失 | 不能断言没执行 |

工具分支则先检查 tool_calls 上限，执行 `_execute` 得到结果，再检查结果字符预算。结果长度只有拿到序列化输出后才知道，因此输出预算失败时工具可能已经完成，ledger 中也已经有结果。运行时进入 limit，并保留 pending，而不是截断 JSON 字符串后继续发送。截断的 JSON 会破坏下轮模型协议，也可能误导模型以为工具返回了另一种结果。

这一实现的 tool_calls 计数在结果接受进对话时增加，因此它更接近“已提交工具结果次数”，不完全等于底层函数真实进入次数。输出超限时底层可能已经执行一次，计数仍未增加。你不能拿这个字段独自做外部工具账单；应结合 ledger 和事件，或者在生产设计中增加 started、completed、delivered 等独立计数。解释真实语义比给字段一个听起来全面的名字更重要。

预算不足后如何向用户返回结果？最稳妥的课程方案是确定性摘要：列出已完成动作、尚未完成事项和停止原因，不再调用模型。如果产品希望让模型写一段总结，需要预先为总结预留额度，且限制它不能再调用工具。否则“为了说明已超预算”又花一次不受控请求，会让预算规则自相矛盾。

## 第五课：副作用异常为什么必须分两个 try

阅读 `_execute` 时最重要的结构不是哈希，也不是 JSON 编码，而是两个异常作用域。第一个 try 包括工具查找、参数长度、JSON 解析和 schema 验证。这里的 ContractError 可以证明工具函数尚未调用，所以返回 `invalid_arguments` 合理。第二个 try 包括函数调用及结果序列化；无论抛出什么异常，只要工具声明 side_effect，就进入 UncertainExecution。

如果把两个 try 合成一个，再用 `except ContractError` 一律归类为输入错误，就会出问题。工具函数自己也可能抛 ContractError，而且可能在写入之后才抛。异常类同名不证明发生阶段相同。结果序列化失败也一样：工具返回一个不可编码对象，业务写入却已经成功。正确分类依赖控制流边界，而不是开发者给异常起了什么名字。

```python
from labs.agents.contracts import Tool, ToolCall, object_schema, ContractError

created = []
def writes_then_fails(args):
    created.append(args["title"])
    raise ContractError("failure after write")

tool = Tool("create", "teaching side effect", object_schema({"title":{"type":"string"}}), writes_then_fails, True)
h = Harness(ScriptedModel([Turn(calls=[ToolCall("c1", "create", '{"title":"report"}')])]), [tool])
s = h.new("effect-failure", "create report")
h.run(s)
assert s.status == "uncertain" and created == ["report"]
before = len(created)
h.run(s)
assert len(created) == before
```

这里第一次状态转移获取工具请求，第二次进入函数，列表写入发生后抛出异常。运行时把状态设为 uncertain，后续 run 看见终态就停止，不会再次写入。这个实验只证明当前运行时没有盲目重试，并不证明列表写入具有持久性；列表只是可观察副作用的最小模型。A10 会用数据库和业务键演示真正的恢复窗口。

若纯工具抛异常，运行时返回结构化 `tool_failed`，模型可以看到工具失败。但“纯”是工具注册者的承诺，不是函数名叫 read 就自动成立。读取接口也可能产生审计记录、收费或缓存写入；你要明确哪些效果属于需要避免重复的业务效果。把有副作用的函数错误标成纯工具，会绕过不确定性保护，这是注册阶段必须审查的合同。

还有一种失败发生在 ledger.complete。工具正常返回了，但本地记录结果时数据库错误；此时外部效果与本地账本可能不一致。当前 `_execute` 只捕获部分执行阶段异常，数据库操作异常可能传播给调用者。生产外层必须保留这一失败并重新读取状态，不能把未知异常自动映射为“工具肯定没执行”。课程的 ledger 是帮助识别窗口的教学模型，不是任意外部系统上的恰好一次执行保证。

## 第六课：不确定状态需要对账，不需要猜测

对账的目标是把未知事实变成有证据的状态。设业务操作编号为 request-17，可以向业务系统查询该编号：若找到同一意图的已完成记录，取回结果并更新本地账本；若系统明确保证查询未找到意味着请求从未提交，并且没有仍在处理的可能，才考虑重新执行；若查询本身也不确定，就继续保持未知并通知用户。不能把一次暂时查不到当作从未发生，尤其当服务采用最终一致查询或异步处理时。

对账也要比较意图摘要。同一个业务编号若对应不同标题或不同目标，应报冲突，而不是随便返回现有结果。这能发现调用者错误复用编号，防止把另一项操作误认为当前请求成功。摘要应覆盖所有影响业务含义的字段，必要时包括工具版本、资源身份和操作者；展示文案或无关追踪字段是否纳入，则由合同决定。

对账不是让模型自由猜“应该已经成功了”。模型可以帮助解释日志或建议排查路径，但确认事实应来自业务查询、数据库记录、明确回执或受控人工核验。最终状态中保留证据来源和确认时间，便于以后审计。若只能人工确认，系统应明确进入等待核验，而不是把未完成任务当作失败后重新安排。

下面是三个常见的对账结果，注意“未找到”没有自动等于允许重试。

| 查询结果 | 还需检查 | 下一状态 |
|---|---|---|
| 找到同键同意图成功记录 | 结果是否完整、是否属于当前租户 | 可确认完成并回填 |
| 找到同键不同意图记录 | 编号复用原因 | 冲突，停止自动执行 |
| 未找到 | 查询一致性、是否仍在处理、服务保证 | 根据合同继续等待或安全重试 |
| 查询失败 | 没有新证据 | 保持不确定 |

补偿也不是回滚的同义词。已经创建的工单可以再关闭，已经发送的邮件可能只能追加更正，已经被外部用户看到的信息无法撤回。补偿本身是新的业务动作，有权限、失败、重试和幂等问题。不要在通用异常处理器里自动调用一个叫 undo 的函数，就宣称事务完整；必须针对业务定义补偿可达到的效果与剩余影响。

## 第七课：取消要保持终态，且不能假装撤销过去

课程 `cancel` 只在 ready 或 tools 状态接受取消。已经 finished、error、limit、uncertain 或 cancelled 时直接返回原状态，不新增事件，也不增加快照版本。这个规则解决迟到取消覆盖成功结果的问题：用户点击按钮的网络请求可能晚于任务实际完成，若无条件改成 cancelled，历史会丢失真实结果。重复取消也应幂等，不能每次点击都创建一条新的终止记录。

```python
from copy import deepcopy

completed = Harness(ScriptedModel([Turn("done")]), [])
finished = completed.run(completed.new("late-cancel", "finish"))
snapshot = deepcopy(finished)
completed.cancel(finished)
assert finished == snapshot and completed.resume("late-cancel") == snapshot

waiting = Harness(ScriptedModel([Turn("unused")]), [])
cancelled = waiting.new("repeat-cancel", "stop")
waiting.cancel(cancelled)
snapshot = deepcopy(cancelled)
waiting.cancel(cancelled)
assert cancelled == snapshot and cancelled.model_calls == 0
```

这段测试比较整个状态而不只比较 status，因此同时覆盖事件数、版本和最终文本是否被修改。对终态不变量，完整比较比单字段断言更有价值。当前 Harness 是同步、单运行拥有者的教学循环，cancel 不能神奇地中断正在运行的同步函数；调用者只能在转移边界调用它，或通过更复杂的执行器协调正在执行的动作。

真实异步系统要定义取消的线性化点：在哪个明确时刻把取消视为已接受。接受之后停止安排新动作，但已经进入外部系统的动作仍需等待结果或对账。若完成提交先发生，迟到取消保持完成；若取消先被接受，后续返回结果可以记录为已观察效果，却不应继续自动规划新步骤。两个线程各自覆盖 status 会形成竞争，单靠最后写入获胜无法表达产品语义。

取消传播还涉及任务树。父任务取消后，子任务、网络流、子进程和等待队列应收到信号并被回收；只把父级 Future 标记为取消，子任务可能继续占资源。清理本身也可能失败，需要有限等待和日志。对于公司服务器，停止发起新的 GPU 工作和终止自己启动的受控进程是合理边界，不能为了清理本任务而广泛杀死同机其他进程。

把取消视为协议后，用户说明也会更准确：可以报告已停止后续步骤、某项已提交动作仍在确认、已保存哪些结果。不要写“已取消，什么都没有发生”，除非所有可能产生效果的操作都确实尚未开始且有证据。诚实说明已完成部分，比一个含糊的成功或取消图标更有助于用户决定下一步。

## 第八课：四个故障复盘与修复依据

故障一：上限三次，却观测到九次 HTTP 请求。内层 SDK 重试三次，外层包装器又重试三次，每次内层耗尽才让外层看到异常。修复先确定谁拥有同一逻辑请求的重试权，再让另一层透传结果；如果保留两层，则必须共享总尝试预算并记录真实请求数。验收用一个永远抛 TransientFailure 的底层计数函数，断言精确次数，不能只看最终异常类型。最终异常相同并不意味着成本相同。

故障二：任务三十秒截止，运行了两分钟。每次重试都把 timeout 重置为三十秒，等待退避又没有算入任务时间。修复建立唯一 deadline，每次请求前取剩余时间，等待前也检查预计等待是否越界。验收用假时钟让每次操作消耗十秒，再设置退避；确认到期后没有新的调用。若底层同步操作不支持中断，文档必须承认它可能超过 deadline，不能拿“开始前检查”冒充硬截止。

故障三：创建工单失败后自动重试，出现两张同名工单。服务端第一次已提交，响应在返回途中丢失，客户端只看到超时。修复引入稳定业务键、服务端去重或查询接口，并把未知结果转入对账。验收应模拟“写入后抛异常”，而不是“写入前抛异常”；后者无法覆盖真正的重复风险。测试必须检查业务记录数，不仅检查本地状态名称。

故障四：工具成功后被显示为“预算不足，未执行”。工具返回巨大的结果，运行时在输出字符检查时停止，但业务动作已发生。修复把结果存在、结果已进入对话、任务最终完成分为不同事实；大结果改为受控 artifact 引用，或提前声明输出上限并预留。验收查 ledger 中保留的完整结果，同时确认 messages 没有损坏的截断 JSON。用户提示应说明动作已完成但结果交付受限，而不是把所有 limit 都解释为没做事。

这四个复盘分别关注次数、时间、副作用和输出交付。它们说明可靠性不是在一个装饰器里解决所有错误，而是让每层拥有明确责任。写复盘时应记录触发条件、已发生效果、缺失证据、修复后的不变量和覆盖测试。若修复只是“把重试从三改成五”，却没有解释为什么重复安全，通常还没有解决核心问题。

## 第九课：十二道完整练习与答案

练习一，永远失败的操作最多尝试三次，应等待几次？答案是调用三次、等待两次。下面直接运行并检查，捕获最后的 TransientFailure 后仍检查计数。若测试只断言抛异常，错误实现尝试一次或一百次都可能通过。

```python
calls_seen = []
waits_seen = []
def always_transient():
    calls_seen.append(1)
    raise TransientFailure("still unavailable")
try:
    retry(always_transient, max_attempts=3, sleep=waits_seen.append, rng=random.Random(1))
except TransientFailure:
    pass
else:
    raise AssertionError("expected final failure")
assert len(calls_seen) == 3 and len(waits_seen) == 2
```

练习二，永久错误是否进入等待？答案不进入。相同输入与身份没有改变时，重复请求不具备新的成功条件。下面保持重试上限为五，证明分类优先于上限；不能因为还有额度就继续做确定无效的工作。

```python
permanent_calls = []
permanent_waits = []
def denied():
    permanent_calls.append(1)
    raise PermanentFailure("permission denied")
try:
    retry(denied, max_attempts=5, sleep=permanent_waits.append)
except PermanentFailure:
    pass
assert len(permanent_calls) == 1 and permanent_waits == []
```

练习三，证明嵌套重试的乘法。答案建立内外各三次的调用，底层总是失败。这里等待函数不真实睡眠，所以测试很快；这不表示生产里可以不等待。将内层 max_attempts 改为一后，真实调用降为三次，说明修复的依据是责任归属，不是隐藏异常。

```python
bottom_calls = []
def bottom():
    bottom_calls.append(1)
    raise TransientFailure("planned")
def middle():
    return retry(bottom, max_attempts=3)
try:
    retry(middle, max_attempts=3)
except TransientFailure:
    pass
assert len(bottom_calls) == 9
bottom_calls.clear()
try:
    retry(lambda: retry(bottom, max_attempts=1), max_attempts=3)
except TransientFailure:
    pass
assert len(bottom_calls) == 3
```

练习四，用假时钟证明转移边界 deadline 的能力范围。工具开始时尚未超时，函数内部把时间推进到截止之后，当前同步调用仍会完成；下一次转移才停止。答案明确展示这个限制，不把测试改成符合愿望的结果。理解限制后，才能决定是否需要异步取消或隔离进程。

```python
from labs.agents.tools import arithmetic_tool

clock = FakeClock()
def slow_pure(args):
    clock.advance(20)
    return {"answer":42}
slow = Tool("slow", "advance fake time", object_schema({}), slow_pure)
h = Harness(ScriptedModel([Turn(calls=[ToolCall("s", "slow", "{}")]), Turn("unused")]), [slow], deadline=10, clock=clock)
s = h.new("slow-boundary", "test")
h.step(s)
h.step(s)
assert s.tool_calls == 1 and s.status == "ready" and clock.now == 20
h.step(s)
assert s.status == "limit" and s.model_calls == 1
```

练习五，模型请求失败是否消耗次数？答案消耗。课程运行时先预留并保存，再调用模型。即使适配器抛 RuntimeError，model_calls 仍为一，并进入 error。注意这与进程直接崩溃不同：捕获到异常的路径会保存明确错误，崩溃只留下 inflight 标志时则需要 uncertain。

```python
h = Harness(ScriptedModel([RuntimeError("transport failed")]), [], budget=Budget(model_calls=1))
s = h.run(h.new("failed-request-count", "test"))
assert s.status == "error" and s.model_calls == 1
assert any(e["kind"] == "model_error" for e in s.events)
assert h.resume(s.run_id).model_calls == 1
```

练习六，零调用预算是否还会发请求？答案不会。零是明确禁止调用，不是默认值或无限。下面的模型脚本即使包含可用答案，也不应被读取；最终状态 limit，计数零。设计配置时不要写 `budget or default_budget` 来替换数值零，这会把用户明确限制变成默认额度。

```python
h = Harness(ScriptedModel([Turn("unused")]), [], budget=Budget(model_calls=0))
s = h.run(h.new("zero-budget", "test"))
assert s.status == "limit" and s.model_calls == 0
assert not any(e["kind"] == "model_requested" for e in s.events)
```

练习七，输出超限后结果在哪里？答案在 ledger，尚未进入对话。下例函数返回一百个字符，上限为十；序列化包裹还会增加长度，所以必然超限。使用与运行时相同的签名规则查账本，确认 status 为 done。不能只检查 pending 还在就推断工具没执行。

```python
import hashlib
large = Tool("large", "large result", object_schema({}), lambda args: "x" * 100)
h = Harness(ScriptedModel([Turn(calls=[ToolCall("big", "large", "{}")])]), [large], budget=Budget(tool_output_chars=10))
s = h.run(h.new("large-output", "test"))
signature = hashlib.sha256("large\0{}".encode()).hexdigest()
row = h.store.lookup(s.run_id, "big", signature)
assert s.status == "limit" and row["status"] == "done"
assert not any(m["role"] == "tool" for m in s.messages)
assert s.pending and s.tool_calls == 0
```

练习八，副作用函数返回不能编码的对象，是否属于输入错误？答案不是。函数已经进入且可能完成效果，序列化异常也属于调用后窗口。下面记录一次效果，返回 set，使 JSON 编码失败，状态必须 uncertain。这个测试防止把异常作用域扩大后误分类。

```python
serialization_effects = []
def bad_output(args):
    serialization_effects.append("created")
    return {1, 2}
t = Tool("bad_output", "side effect with invalid result", object_schema({}), bad_output, True)
h = Harness(ScriptedModel([Turn(calls=[ToolCall("bad", "bad_output", "{}")])]), [t])
s = h.run(h.new("serialize-unknown", "test"))
assert s.status == "uncertain" and serialization_effects == ["created"]
```

练习九，同样工具若在 schema 阶段失败，是否有副作用？答案没有，前提是验证确实发生在函数调用之前。下面缺少 title，记录列表保持为空，工具消息给出 invalid_arguments，模型随后可以结束。这一对照与上一题共同证明分类按阶段，而不按异常名称猜测。

```python
never_called = []
t = Tool("needs_title", "required field", object_schema({"title":{"type":"string"}}), lambda args: never_called.append(args), True)
h = Harness(ScriptedModel([Turn(calls=[ToolCall("missing", "needs_title", "{}")]), Turn("need title")]), [t])
s = h.run(h.new("validation-before-effect", "test"))
import json
result = json.loads(next(m["content"] for m in s.messages if m["role"] == "tool"))
assert result["error"]["code"] == "invalid_arguments" and never_called == []
```

练习十，最后一次允许的转移恰好完成，能否覆盖成 limit？答案不能。循环次数耗尽只说明没有更多转移机会，不表示最后一次没完成。下面 max_transitions 为一，模型直接返回最终答案，状态必须 finished；同时检查没有额外 run_stopped 事件。这个边界已进入仓库回归测试。

```python
h = Harness(ScriptedModel([Turn("finished on boundary")]), [])
s = h.run(h.new("last-transition", "test"), max_transitions=1)
assert s.status == "finished"
assert not any(e["kind"] == "run_stopped" for e in s.events)
```

练习十一，手算并实现三轮费用。使用 Decimal 避免金额展示中的二进制小数误差，假设输入每百万二元、输出每百万八元。三轮输入一千、两千、四千，输出一百、二百、三百，总费用为零点零一八八元。真实服务还要按自身计价项结算，这里只检验单位与累加。

```python
from decimal import Decimal
usage = [(1000,100), (2000,200), (4000,300)]
price_in = Decimal("2")
price_out = Decimal("8")
cost = sum((Decimal(i)*price_in + Decimal(o)*price_out)/Decimal(1_000_000) for i,o in usage)
assert cost == Decimal("0.0188")
```

练习十二，取消后再次 run 与再次 cancel 是否改变状态？答案都不改变。下面在任何模型请求前取消，随后运行一百次允许转移，也不会发请求。完整状态比较同时约束版本和事件，防止重复取消悄悄污染审计记录。

```python
h = Harness(ScriptedModel([Turn("unused")]), [])
s = h.new("cancel-before-request", "test")
h.cancel(s)
before = deepcopy(s)
h.run(s, max_transitions=100)
h.cancel(s)
assert s == before and h.resume(s.run_id) == before
```

完成这些题后，请把每题的安全结论和范围各写一句。例如“输出超限不会把坏 JSON 送给模型”是已验证结论，“输出超限意味着工具未执行”则是错误推断。可靠性学习最容易出现的问题，是把一个局部测试的结论扩大成整个系统的保证。每个保证都应带前提、观测范围和失败后的状态。

## 第十课：实现带 deadline 的可测试重试执行器

下面的小项目补齐配置验证、明确重复安全性、时间限制、取消检查和事件记录。它仍然同步，不能中断 operation 内部；operation 接受剩余时间，由实际适配器把它传给底层超时机制。这个函数不隐式判断副作用安全，调用者必须显式提供 repeat_safe。它也不把未知错误改名成临时错误，只捕获 TransientFailure。

```python
import math
from dataclasses import dataclass

class RetryStopped(RuntimeError):
    pass

def finite_config_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base: float = 0.1
    cap: float = 2.0
    def __post_init__(self):
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("attempts must be positive integer")
        for value in (self.base, self.cap):
            if not finite_config_number(value) or value < 0:
                raise ValueError("delays must be finite nonnegative numbers")
        if self.cap < self.base:
            raise ValueError("cap must be at least base")

def controlled_retry(operation, *, repeat_safe, policy, deadline, clock, sleep, rng, cancelled, events):
    if type(repeat_safe) is not bool:
        raise ValueError("repeat_safe must be boolean")
    if not finite_config_number(deadline):
        raise ValueError("finite deadline required")
    delay_cap = policy.base
    for attempt in range(1, policy.attempts + 1):
        if cancelled():
            raise RetryStopped("cancelled")
        remaining = deadline - clock()
        if remaining <= 0:
            raise RetryStopped("deadline")
        events.append({"kind":"attempt_started", "attempt":attempt, "remaining":remaining})
        try:
            value = operation(remaining)
        except TransientFailure:
            events.append({"kind":"attempt_failed", "attempt":attempt})
            if not repeat_safe or attempt == policy.attempts:
                raise
            if cancelled():
                raise RetryStopped("cancelled")
            wait = rng.uniform(0, delay_cap)
            if wait >= deadline - clock():
                raise RetryStopped("insufficient time for retry")
            events.append({"kind":"retry_wait", "seconds":wait})
            sleep(wait)
            delay_cap = min(policy.cap, delay_cap * 2)
        else:
            events.append({"kind":"attempt_completed", "attempt":attempt})
            return value

clock = FakeClock()
trace = []
count = []
def operation(remaining):
    assert remaining > 0
    count.append(1)
    clock.advance(0.2)
    if len(count) < 3:
        raise TransientFailure("planned")
    return 42
answer = controlled_retry(operation, repeat_safe=True, policy=RetryPolicy(), deadline=5.0,
    clock=clock, sleep=clock.advance, rng=random.Random(3), cancelled=lambda:False, events=trace)
assert answer == 42 and len(count) == 3
assert sum(e["kind"] == "retry_wait" for e in trace) == 2
```

逐行看配置对象：frozen 禁止普通字段重赋值，使一次运行的策略不被中途改写。attempts 要求精确正整数，延迟要求有限非负数，cap 不低于 base。finite_config_number 捕获巨大整数转换浮点时的 OverflowError，返回不合法，再由入口统一抛 ValueError。这与 A03 接受任意精度业务整数的合同不同：这里配置要交给以浮点秒为单位的时钟与等待函数，因此主动拒绝不可表示的巨大值。检查工具参数与检查计时配置可以有不同边界，前提是把原因和异常语义说清楚。

主函数每次尝试之前先检查取消，再计算剩余时间，再记录开始事件。这一顺序保证到期后没有新的 operation 调用。失败后先记录，再判断重复安全与次数，随后检查取消和等待是否越过期限。等待结束后没有直接调用 operation，而是回到循环顶部重新检查，因为时间可能变化、用户也可能在等待中取消。仅在等待前检查一次，会漏掉这个窗口。

成功后返回结果，即使 operation 内部消耗了超出预期的时间，也不会假装结果不存在。是否把这种“晚到成功”交付给用户，应由上层政策决定；副作用已经成功时尤其不能因迟到而自动重试。当前执行器记录完成，但不提供强制中断，也不把晚到结果当作取消。这个取舍保持了已知事实，避免把时间策略与业务效果混为一谈。

事件列表只记录尝试号、剩余时间和等待，不记录密钥或全部业务参数。生产日志可以再加 run_id、operation_id、attempt_id 与错误分类，形成可关联轨迹。不要把完整异常字符串直接写入公共日志；异常可能携带 URL 查询参数、请求体或身份信息。结构化记录比事后用正则表达式清理更可靠。

## 第十一课：故障注入应覆盖每一个提交窗口

只让 operation 在入口抛异常，覆盖不了恢复中最危险的窗口。一个完整的故障注入计划应把失败插在参数验证后、请求预留保存后、远端提交后、响应接收后、结果账本保存后和对话快照保存后。每个窗口的预期不同：有的能证明没执行，有的只能证明可能执行，有的已有可复用结果。把它们都期望为 error，会抹掉可靠性设计最重要的区分。

以工具调用为例，先在 ledger 写 running，随后调用业务函数，成功后写 done 与 output，最后更新 messages 和 pending。若在 running 后崩溃，纯工具可以根据声明重新执行，副作用工具需要对账；若在 done 后、messages 保存前崩溃，恢复可以取账本缓存结果回填，不应再调用业务函数。这个“效果已发生但对话没看见”的窗口，是为什么工具结果不能仅存在 Python 返回值里。

你可以用包装对象注入故障。例如给 Store.save 包一层计数，在第三次保存时抛 OSError；或让工具在业务记录写入后抛自定义异常。测试中不要直接修改内部 state 伪装故障，除非你正在测试不变量验证器。真正的恢复测试应从合法初始状态推进到指定窗口，让真实代码写出相应记录，再模拟进程中断并用新对象加载。这样才能证明恢复逻辑与实际写入顺序一致。

模拟进程崩溃与普通 Exception 也不同。普通异常可能被运行时捕获并保存一个明确 error；真实崩溃没有机会执行 except 或 finally。测试可以用一个不被普通 Exception 捕获的专用控制信号，在外层测试框架捕获它，再重新加载持久状态；但不要在生产代码里使用这种技巧表达业务错误。课程 A10 会把这些窗口和 SQLite 事务放在一起完整演示。

故障注入的验收应同时检查外部效果、账本、对话和状态版本。例如某次恢复最终回答正确，但业务记录出现两份，不能算成功；又如没有重复业务写入，但 pending 被错误清空，模型永远不知道工具结果，也没有完成任务。可靠性是这些事实之间保持一致关系，不是单个 status 字符串变成 finished。

另外要检查再次恢复。第一次恢复成功后，再次加载并运行应保持终态，不新增工具效果，不重复追加 tool 消息。恢复代码本身也可能在写入结果后崩溃，所以不能把“恢复只会发生一次”作为隐含前提。一个好的测试会在恢复路径上再次注入失败，观察账本是否仍然足够区分已完成和未完成工作。

## 第十二课：熔断、限流和重试分别解决什么问题

重试处理一次逻辑操作遇到的可恢复失败；限流控制一段时间内允许发出的请求速度；并发限制控制同时在途的请求数量；熔断在观察到持续故障时暂时停止向某个依赖发送请求。这四者相关但不同。把并发设为十，不代表每秒最多十次，因为每个请求若只花一毫秒，吞吐可以很高；每秒限制十次，也不代表同时只有十个请求，因为慢请求可能持续堆积。

熔断的直觉是：一个依赖已经连续失败，继续让所有任务轮流重试会占满资源，还可能妨碍依赖恢复。可以维护 closed、open、half-open 三种状态。closed 正常发送并记录结果；错误达到策略阈值后进入 open，快速返回不可用；冷却后允许少量探测进入 half-open；探测成功恢复 closed，失败重新 open。名字只是约定，关键是状态转移条件和探测并发必须清楚。

不要按任意错误率熔断所有业务。用户输入错误、权限不足和模型拒绝不一定说明依赖不健康；把它们算作服务故障，会让一个恶意或错误请求影响其他正常用户。应按依赖级别和错误类别统计，必要时按租户、端点或资源隔离。熔断器也不应该隐藏结果未知的副作用，已经发出的操作仍需对账，open 只能阻止新的发送。

限流与重试结合时，重试也应消耗真实请求配额，不能享有无限优先权。否则故障时重试流量会挤掉新任务，形成饥饿。可以为重试设置独立的小预算，或用公平队列让新请求与重试按政策分享资源。这里不要求你立刻实现复杂调度器，但必须在设计中回答“失败任务是否可以无限占住名额”。

一个实用的小系统可以先使用单依赖并发上限、总请求预算和指数抖动，不一定需要完整熔断。只有观察到持续依赖故障导致资源堆积，且能清楚定义恢复探测时，再引入熔断。增加可靠性组件也会增加状态和故障模式；例如熔断器状态被多进程各自维护，可能仍同时发出很多探测，需要共享协调或接受明确的近似范围。

测试这些组件时应采用假时钟和可控服务脚本。脚本前三次失败、随后成功，检查何时打开、冷却前是否没有真实请求、冷却后是否只有允许的探测。不要用真实服务故障来“证明熔断有效”，因为外部条件不可控，费用和副作用也难以界定。离线状态轨迹先正确，再用受限集成验证 transport 映射，是更清楚的学习顺序。

## 第十三课：把预算预留写成可解释的账本

设一个任务允许最多一千个单位的资源。开始第一项操作前预留四百，实际用了二百五十，结算后释放一百五十；第二项操作预留六百时，剩余额度足够。若第一项结果未知，不能直接释放全部四百，因为它可能已经消耗资源。账本应保留 reserved、settled 或 unknown 等状态，直到获得实际用量或按明确规则计入保守估算。

| 操作 | 预留 | 确认使用 | 未决预留 | 可用余额 |
|---|---:|---:|---:|---:|
| 初始 | 0 | 0 | 0 | 1000 |
| 操作甲开始 | 400 | 0 | 400 | 600 |
| 操作甲结算 | 400 | 250 | 0 | 750 |
| 操作乙开始 | 600 | 250 | 600 | 150 |
| 操作乙响应丢失 | 600 | 250 | 600 | 150 |

这个表中的可用余额等于总额减已确认使用减未决预留。若把“未收到结果”当作“未消耗”，最后一行会错误地变回七百五十，从而允许继续发起超预算请求。现实账单可能延迟到达，所以本地预算需要接受保守性：暂时阻止一个本来可以支付的请求，比错误承诺不会超支更诚实。用户若愿意追加预算，可以通过明确配置变更继续，而不是由重试器偷偷扩大额度。

预留标识也需要幂等。相同 operation_id 重复请求预留时，应返回原预留，不再次扣款；同一个标识对应不同金额或意图时应冲突。结算也应幂等，同一结算重复到达不能重复消费。A10 的业务幂等和 inbox 去重不仅适用于工单，也适用于预算记账，因为它们都是“消息可能重复但业务事实只能记录一次”的问题。

预算与权限不要混为一谈。有额度不代表有权限操作某个资源，有权限也不代表有无限额度。正确顺序通常是先识别操作和身份，验证权限及输入，预留必要资源，再提交动作。某些服务调用本身就是验证手段，可能已经消耗费用；这种特殊情况应明确记账，而不是假装所有验证免费。设计 API 时把授权、预算、业务执行作为独立责任，便于分别测试和审计。

你可以为课程项目增加一个内存预算账本，先不涉及并发：reserve 接受 operation_id 和 amount，settle 接受实际用量，unknown 保留预留，release 只允许证明未执行的操作。每次操作后断言余额非负、同一编号不重复结算、已确认使用单调增加。等这些合同稳定，再在 A09 用锁保护检查加扣减，在 A10 用事务保护进程崩溃后的恢复。不要一开始就把并发、数据库和业务规则揉成一个难以验证的大类。

## 第十四课：错误输出如何帮助模型修复，又不诱导危险重试

工具返回错误不只是给人看的文字，它会成为下一轮模型输入，因此错误结构会影响模型行为。一个有效的错误可以包含 code、stage、retryability、safe_to_repeat 和可操作的字段路径，但这些字段必须由宿主根据事实设置，不能让模型自己声明。输入错误可以指出 `$.count` 超范围，并给出允许范围；权限错误应说明当前不能执行，不应泄漏其他用户资源是否存在；未知副作用应要求对账，不能写成“请换参数再试”。

“可重试”最好不要只用一个布尔值。可以区分原样重试、修正输入后新请求、等待用户授权、等待依赖恢复、先查询操作状态和禁止自动继续。这样模型能看到真正的下一步，而不是把所有失败都理解为继续调用同一工具。对结果未知，宿主甚至可以直接把运行状态转为 uncertain，阻止模型继续规划，直到有新的可信证据进入系统。

错误内容也可能带有不可信数据。数据库错误里可能包含用户提交的文本，网页抓取失败里可能包含远端页面标题，子进程 stderr 可能含有恶意指令。把这些内容放入 tool 消息不代表它们获得指令权限。应保留错误来源、限制长度、脱敏敏感字段，并在上下文装配中把它们视为观察数据。A07 会把这种信任边界进一步展开。

对模型的反馈应足够修复，但不要暴露内部堆栈和密钥。开发日志可以包含异常类型和受控堆栈，模型消息只需要业务相关信息。例如参数字段不合法应提供字段路径；数据库连接串错误不应把完整连接串回传给模型。若信息不足以安全修复，可以停止并请求人工处理，而不是让模型猜测系统环境。

评测错误反馈时，可以准备同一组输入错误，用不同错误结构做配对实验，比较修正成功率、额外调用次数和是否重复副作用。不能只看最后回答语言是否自然。一个系统可能用很漂亮的解释掩盖任务失败，另一个系统则简短说明无法继续但保留正确状态。可靠性的评分应围绕业务事实和边界，而不是措辞流畅度。

## 第十五课：完成本章的综合验收

把前面的组件组合成一个小型“可靠工单助手”设计报告。输入是创建工单的意图和稳定业务键；工具先验证标题、权限与预算，再调用受控业务函数；函数可能在提交前失败、提交后丢失响应、返回成功但本地保存失败。你的报告必须逐一画出状态与账本变化，说明每个窗口是否允许自动重试，重试是否保持同一业务键，以及用户取消后还能保证什么。

最低验收包含六条执行轨迹：首次成功；两次临时纯读取失败后成功；永久参数错误立即停止；截止时间恰好到达不再请求；副作用提交后响应丢失进入不确定；最终成功之后迟到取消保持原状态。每条轨迹都记录模型请求数、工具真实进入数、已接受工具结果数、业务记录数和终态。把这些计数并列，能防止你把不同层次的“调用次数”误当同一个指标。

进一步的验收是恢复两次。先在工具结果写入账本后中断，再加载旧对话快照，通过缓存结果回填；回填成功后再次加载并运行，应保持 finished，不重复业务动作。这个任务将在 A10 使用现有 Store 完整实现，因此本章报告应为后续代码准备精确预期，而不是只写“支持断点恢复”。你现在已经具备判断该功能是真是假所需的概念与测试方法。

最后检查你的解释是否包含能力边界：同步 deadline 无法强制中断函数，取消不自动回滚，客户端 ledger 不等于外部系统恰好一次，次数预算不等于费用预算，异常名称不证明执行阶段，未知用量不能记零。它们不是消极免责声明，而是决定下一步怎样实现的具体前提。真正能独立构建 Agent 的人，既能让正常流程跑通，也能在这些边界上给出准确、可验证的行为。

综合验收还应包含一份面向运维的诊断顺序。先根据 run_id 找到最后已提交状态，确认是否存在 inflight_model 或 pending；再查对应调用账本，区分没有记录、running 和 done；随后查询业务系统的稳定操作编号；最后才决定是否需要新请求。这个顺序从本地确定事实走向外部证据，避免一看到用户说“没有回复”就立刻重跑整个任务。没有回复可能只是交付失败，业务已经成功。

若账本为 done 而对话没有 tool 消息，优先回填缓存结果。若账本为 running 且工具有副作用，优先对账。若没有账本记录但模型请求仍在途，先处理模型结果未知；不要直接构造一个新的调用编号代替。若状态已经 finished，检查用户是否只是没有收到展示通知，重新展示已有最终结果通常不需要新的模型调用。这些分支都能减少不必要费用，也降低重复动作风险。

通知交付本身也是一个外部效果。假设任务完成后需要把结果推送到界面或消息系统，推送失败不应把整个任务重置为 ready。可以把任务完成与结果通知分开记录，通知采用自己的幂等键和重试策略。否则一次界面断连就触发完整 Agent 重跑，前面所有工具都会受到影响。A10 的 outbox 模式正适合处理“业务提交成功，但后续通知还没发出”的窗口。

在多用户服务中，可靠性还需要隔离。某个用户的长时间重试不应占满所有工作槽位；某个依赖持续失败不应使其他独立工具一起停止。可以按租户或依赖设置配额，并将等待中的重试从执行槽位释放，等下一次真正尝试时再获取。若一直持有并发槽位睡眠，系统虽然限制了在途工作，却让没有实际工作的等待占住资源，正常任务可能被饿死。

不过释放槽位也有公平性取舍：重试重新排队可能延迟很久，任务 deadline 需要继续计时，不能因为排队就暂停。队列等待、退避等待和实际执行时间都属于用户等待。指标应分别记录这些时间，才能判断慢在哪里。只记录工具函数内部耗时会掩盖调度问题，让你误以为模型很快而用户体验仍差却找不到原因。

本章的小项目可以再加一项静态审查：逐个寻找所有 `except Exception`，写出它覆盖的代码阶段，以及捕获后是否可能重试或丢失证据。宽泛捕获本身不一定错误，顶层服务需要把异常转换为可理解状态；问题在于是否把不同事实混成同一种安全结论。若捕获发生在副作用之后，不能无依据宣称未执行；若捕获后只打印并继续，可能让损坏状态进入下一轮。这个审查方法简单，却能在真实框架源码中发现很多比“超时值设多少”更重要的可靠性问题。

还有一个配置变更边界：任务因预算停止后，用户追加额度，不应直接篡改历史计数为零。历史请求仍然发生过，应保留原计数与原停止记录，再通过明确的继续操作建立新的允许范围。若新配置改变了模型、工具定义或权限，也要记录配置指纹，确认旧 pending 是否仍适用。继续一个已停止任务是受控状态迁移，不是随便把 status 字符串改回 ready。这个原则让你能够解释总共用了多少资源、何时获得追加额度，以及哪些动作属于原授权范围；后续版本管理和持久化设计都应保留这份因果关系。

配置变更也应有测试：旧计数保持，新增额度明确，已确认效果不重复，未知效果先对账，再决定是否继续。
