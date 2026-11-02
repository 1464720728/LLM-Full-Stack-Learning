# A15 真实协议与框架：把自写 Agent 接出去，并能沿源码修改回来

本章把前十四章连成一条可交付的工程路线。你已经写过循环、工具、检查点、状态图、评测和编码助手，现在的问题变成：外部团队提供一个工具服务时怎样接入；另一个 Agent 接受长期任务时怎样跟踪；换成真实框架后哪些责任被框架承担，哪些仍然由你承担；协议或依赖升级后怎样证明旧任务没有被悄悄改写。我们不以“安装成功”作为终点，而以一次完整调用的身份、权限、状态、故障与验证都能解释作为终点。

先修要求是能读懂函数参数、字典、异常、上下文管理器、异步函数与 JSON。忘记 `async with` 时，把它理解为会等待的资源进入和退出：进入时打开连接或启动服务，退出时即使出错也要回收。不要把 `await` 理解成后台自动执行；它表示当前协程等待一个可等待对象，具体是否并行要看是否创建了其他任务。本章基础项目仍只需要标准库和 CPU；真实 SDK 实验使用单独虚拟环境，不调用任何付费模型，不使用 GPU。

## 第一课：画清三种接口，才能决定该接什么

一个 Agent 系统至少存在三种不同接口。模型接口把消息和工具定义交给推理服务，返回文字、工具调用或协议事件。工具接口把一个具体能力交给外部实现，例如查询课程章节。任务接口把一个目标交给另一个具有自己循环的应用，例如“检查这份报告的所有证据”。它们都可能经过 HTTP，都可能返回 JSON，因此只看网络格式很容易误判。正确分类方法是问：调用方交付的是一组确定参数，还是一个需要对方规划并持续推进的目标；返回的是本次执行结果，还是可继续查询的任务身份。

例如你的助手要回答“本课程哪里讲到副作用未知”。模型接口负责决定调用搜索。工具接口负责执行关键词查询并给出命中文档。假如你又委派另一个审查 Agent 检查三十处引用，任务接口负责记录审查任务，允许等待、补充信息、取消与获取产物。一个搜索工具内部也可能使用模型，但这不改变外部契约：如果调用方只看到一次确定参数调用，它仍可以作为工具。一个任务服务内部也可能只是固定工作流，协议不要求它拥有神秘的自主性。

| 边界 | 调用方交出的对象 | 关键身份 | 调用方主要验收对象 |
|---|---|---|---|
| 模型 | 消息、工具定义、生成配置 | 模型请求和工具调用身份 | 结构、预算、终止原因 |
| 工具 | 工具名、参数、权限上下文 | 请求身份和业务幂等身份 | 参数、结果、实际副作用 |
| Agent 任务 | 目标、输入材料、约束 | 上下文身份和任务身份 | 状态、产物、独立 verifier |

MCP 的宿主、客户端、服务端是职责划分。宿主组织模型、用户和权限；客户端负责某条协议连接或请求路径；服务端提供能力。一个宿主可以持有多个客户端，并过滤各个服务端暴露的工具。这里的“服务端”不必在公网，标准输入输出连接的本地子进程同样可能扮演这一角色。反过来，同进程内的普通函数无需为了时髦强行套网络协议，只有进程边界、语言边界、组织边界或生态兼容需求真正出现时，协议成本才有明确收益。

A2A 面向独立 Agent 应用之间的任务协作。对方可以保持内部实现不透明，调用者不必读取其全部工具和内部状态，仍能通过任务与产物判断进展。这里的不透明不等于免于验收。一个远程 Agent 声称完成证据审查，你仍应检查其报告中的文档版本和引用位置。A12 的证据合并、A13 的外部判定器，在跨应用任务中仍然适用。标准化的是交换方式，任务正确性仍要由业务事实支撑。

你应画出自己的责任表：模型选择什么，由谁验证；工具来自哪里，由谁授权；远程任务超时后，由谁确认是否已经完成。这个练习比背三种缩写更重要。很多集成事故来自责任空白：客户端认为服务端检查权限，服务端认为宿主已经批准，最终没有任何一层真正把主体、动作和资源绑定。协议文档提供字段，工程设计必须说明字段的可信来源与检查位置。

## 第二课：把版本当作输入，不把今天的网页当作永恒接口

本章核查日期为 2026-11-01。MCP 的 2026-07-28 版本采用新的无状态核心，每次请求携带协议版本、客户端信息与能力元数据；旧版本的初始化交换与会话头不应被机械套用到新版本核心。可选的发现操作承担另一部分能力查询职责。这个变化来自[官方发布说明](https://blog.modelcontextprotocol.io/posts/2026-07-28/)，而不是本课程根据名称猜测。看见旧教程中的初始化代码时，应先判定它对应哪一版协议，再决定使用历史适配器还是迁移，不能直接宣布旧教程所有部分都错了。

规范版本、SDK 版本、运行服务版本、教程日期是四个不同量。规范描述双方应遵守的语义；SDK 是某次软件发布，可能同时支持多个规范版本，也可能尚未实现刚发布的规范；服务器可能长期固定旧 SDK；教程可能在几个月前验证过一个组合。安装名为 `mcp` 的包，只能证明你拿到了这个发行包，不能证明它恰好支持你正在阅读的规范。类似地，仓库默认分支上的示例不一定能在昨天安装的包里导入。

最低可复现记录应包括 Python 版本、发行包版本、锁定依赖文件、运行入口、所选协议版本、成功与失败夹具，以及源代码位置。若读取的是 Git 检出，应记录实际提交；若只安装 wheel，就记录发行版本和实际导入文件，不凭空编造提交哈希。文档链接的 `latest` 只是访问入口，不是固定版本证据。保存一个请求响应夹具能够帮助你定位升级破坏了字段、错误分类还是生命周期。

下面的版本选择器是课程自定义入口策略，不是 MCP 的完整协商实现。它只允许明确配置的当前方法层和显式标注的历史入口，不做偷偷补字段。把“拒绝”写成可测试行为，远比依靠默认分支碰运气稳定。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Compatibility:
    protocol: str
    adapter: str
    allow_legacy: bool = False

def select_adapter(config):
    if type(config) is not Compatibility:
        raise TypeError("Compatibility required")
    if config.protocol == "2026-07-28" and config.adapter == "course-methods":
        return "current-method-fixture"
    if config.protocol == "historical-fixture" and config.allow_legacy is True:
        return "isolated-legacy-fixture"
    raise ValueError("unsupported explicit compatibility pair")

assert select_adapter(Compatibility("2026-07-28", "course-methods")) == "current-method-fixture"
try:
    select_adapter(Compatibility("2025-03-26", "course-methods"))
except ValueError:
    pass
else:
    raise AssertionError("old input must not silently become current")
```

`frozen=True` 阻止这个配置对象在正常属性赋值中被意外改变；它不意味着嵌套对象都不可变，本例字段只是简单值。第一个条件同时检查协议和适配器，避免仅修改版本字符串就让旧解析器假装支持新协议。第二个条件使用 `is True`，不接受字符串形式的同意。最后的异常让部署启动阶段就失败，而不是等到用户第一次调用时才出现不明原因的错误。

真实迁移时应制作差异表：新增必需字段、删除字段、错误位置、结果形状、连接生命周期、认证要求、缓存键分别是否改变。若两版本的同名操作语义不同，仅重命名字段不能完成转换。你要把不能保存的语义明确标为不支持，并让测试断言拒绝。版本管理的价值是保存可解释的行为证据，不是让依赖字符串永远显得新。

## 第三课：逐行读离线 RPC 方法层，找到它真正实现的边界

本仓库 `labs/agents/integrations.py` 的 `ToolRPC` 是一个方法分派练习。它接受 Python 字典，返回 Python 字典，没有监听端口，没有身份认证，没有真正的传输断开，也没有完整 MCP 兼容承诺。当前[工具规范](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)要求请求携带必要元数据，即使文档中的简化示例省略了这些字段；课程请求构造器显式加入它们。完成结果的类型也在返回中显式保留。分页、缓存、多轮请求、资源、提示模板与通知都不在这个小实现内。

先运行一个完整调用，而不是只看类名。工具函数接收一个参数字典，这与真实 SDK 从函数签名生成 schema 的方式不同。课程调用链是字典验证后再传给函数，SDK 常把字典转换成具名参数。不要把两个调用约定混着使用。

```python
from labs.agents.contracts import Tool, object_schema
from labs.agents.integrations import ToolRPC, request_message

add_schema = object_schema({"a": {"type": "integer"}, "b": {"type": "integer"}})
add_tool = Tool("add", "Add two exact integers", add_schema,
                lambda args: args["a"] + args["b"])
rpc = ToolRPC([add_tool])
listing = rpc.request(request_message("list-1", "tools/list"))
answer = rpc.request(request_message("rpc-1", "tools/call", {
    "name": "add", "arguments": {"a": 17, "b": 25}}))
assert listing["result"]["tools"][0]["inputSchema"] == add_schema
assert answer["id"] == "rpc-1"
assert answer["result"]["content"] == [{"type": "text", "text": "42"}]
assert answer["result"]["isError"] is False
```

构造函数把工具列表变成以名称为键的字典，查找变快了，但重复名称会覆盖前一个。这个实现没有自动拒绝重名，因此宿主接入层必须补检查。`request` 的第一条分支要求输入恰好为字典，然后提取请求身份；内部 `error` 小函数让错误保持相同信封形状。它接受字符串或整数身份，不接受布尔值，也不支持省略身份的通知。接着检查 `params` 和 `_meta` 的容器类型，再核对当前版本与必需元数据键。

注意“键存在”不等于“值已经完整验证”。课程检查客户端信息与能力字段存在，并没有完整验证其中每个子字段。这是方法练习的明确范围。若要实现对外服务，必须按所选规范验证完整请求，而不是把这段代码复制到公网后声称拥有标准实现。输入经过 Python 对象层，也意味着原始 JSON 的重复键、字节长度和编码错误尚未在这里处理；这些属于更外层的解析边界。

`tools/list` 分支把本地工具名称、描述和输入 schema 映射成协议结果。`tools/call` 分支先检查名称类型，再查注册表，再拒绝带副作用标记的工具。只有经过这些检查，才会验证参数并调用函数。成功结果用 `str(value)` 形成文本，所以返回字典时得到的是 Python 字典表示，不是承诺可重新解析的 JSON。需要结构化输出时，应设计独立结构字段与 schema，不能让消费者对任意文本做猜测解析。

最后两个异常分支也值得认真读。课程把 `ContractError` 映射为参数类错误，把其他普通异常映射为工具执行失败结果。由于参数验证和函数执行放在同一个 `try` 中，函数内部自己抛出的 `ContractError` 也会归入参数类错误。这种分类粒度不适合复杂业务；本实现仅允许纯工具，因此没有把一个已经发生的写操作误认为安全重试的风险。若扩展有副作用工具，应像 A03 与 A06 一样把调用前验证和调用后异常分开，调用后未知结果不能靠异常类名变回“没有执行”。

| 阶段 | 本例数据 | 能确定的事实 | 尚不能确定的事实 |
|---|---|---|---|
| 列表请求 | `list-1` | 宿主请求能力列表 | 当前用户有权使用每项能力 |
| 方法检查 | `tools/call` | 选择了已实现方法 | 参数符合业务要求 |
| 参数验证 | 两个精确整数 | 形状符合课程 schema | 任意外部系统可接受此数值 |
| 函数返回 | 整数四十二 | 本地计算完成 | 网络客户端收到返回 |
| 结果消费 | 文本四十二 | 宿主看到了结果 | 最终任务已经通过 verifier |

## 第四课：身份映射是一张关系表，不能压缩成一个万能 id

A02 区分过模型工具调用身份，现在再增加三层。JSON-RPC 请求身份用于关联一次请求与响应；传输尝试身份区分重试或连接尝试；业务操作身份用于识别同一个真实动作。一个模型调用可能因为网络读取失败而对应两次传输尝试，但仍然是同一个业务请求。反过来，模型在下一轮重新提出相同参数，也不一定表示同一个业务意图。相同参数可以是用户真的要买第二份东西，不能仅凭参数摘要永久去重。

不要把模型生成的 `call_id` 直接当作全局数据库主键。它的唯一性范围取决于协议和运行上下文，至少应与运行身份联合使用。也不要把 HTTP 请求身份当作业务幂等键，因为重试常生成新的传输身份。完整记录应该能够从最终证据反查模型动作、RPC 请求和业务结果，也能从一次失败网络尝试找到它所属的动作。关联不是为了日志漂亮，而是为了发生未知结果时知道该查询哪个真实对象。

```python
from copy import deepcopy

class IdentityMap:
    def __init__(self):
        self.actions = {}
    def bind(self, run_id, call_id, operation_id, server, tool):
        key = (run_id, call_id)
        record = {"operation_id": operation_id, "server": server,
                  "tool": tool, "attempts": []}
        if key in self.actions:
            old = self.actions[key]
            if any(old[k] != record[k] for k in ("operation_id", "server", "tool")):
                raise ValueError("identity rebound")
            return
        self.actions[key] = record
    def attempt(self, run_id, call_id, rpc_id, transport_id):
        row = self.actions[(run_id, call_id)]
        item = {"rpc_id": rpc_id, "transport_id": transport_id}
        if item in row["attempts"]:
            return
        row["attempts"].append(item)
    def get(self, run_id, call_id):
        return deepcopy(self.actions[(run_id, call_id)])

ids = IdentityMap()
ids.bind("run-a", "call-a", "operation-a", "course", "add")
ids.attempt("run-a", "call-a", "rpc-1", "http-1")
ids.attempt("run-a", "call-a", "rpc-2", "http-2")
assert ids.get("run-a", "call-a")["operation_id"] == "operation-a"
assert len(ids.get("run-a", "call-a")["attempts"]) == 2
```

`bind` 只允许同一个动作身份保持原有业务身份、服务端与工具，不允许后来悄悄改绑。`attempt` 记录每次网络尝试，不执行任何重试。`get` 返回深拷贝，防止观察者修改内部审计记录。这个类是内存教学表，没有数据库事务和并发控制；生产中它需要进入 A10 的持久记录并受到唯一约束保护。它也没有替你决定哪些请求可以重试。身份稳定只是重试正确性的前提之一，服务端支持幂等、动作可重复、结果可核验仍是另外的条件。

思考一个反例：服务端成功创建工单后，响应在网络中丢失。客户端生成新业务键重发，服务端会合理地创建第二张工单，因为它看到的是新请求。正确处理是保留原业务键并查询或按服务端契约重放。再看另一个反例：两个用户碰巧都说“创建同标题工单”，如果你只按标题去重，就会把两个合法意图合并。身份必须来自业务操作定义，而不是从字符串相似性偷推。

## 第五课：宿主工具目录需要命名空间、版本和权限闭包

外部服务端返回工具清单，宿主不能原样把所有内容交给模型。首先需要明确服务身份；两个服务都叫 `search` 时，模型可见名称必须无歧义。其次需要限制工具类型和 schema 子集；若当前宿主不支持图片结果、长任务或某种 schema 关键字，应在接入时拒绝或选择专用适配器。最后需要把权限放在宿主可信配置中，不能让服务端描述中的“此工具已获管理员批准”成为授权凭据。

下面的目录只接收我们控制的纯函数 fixture，因此本地保存 `Tool` 对象；真实网络目录应保存远程定义以及服务身份，再由单独客户端调用。服务名和工具名使用受限字符，并保留双下划线 `__` 作为两者之间的分隔串，组件自身不得包含双下划线。否则 `alpha__beta` 服务的 `search` 与 `alpha` 服务的 `beta__search` 会得到相同公开名，第二个合法身份无法注册。该限制是本章宿主目录的约定，不是所有远程工具的通用命名规则。`version` 参与宿主目录快照，升级时由可信配置显式替换。授权集合由应用提供，与模型参数分离。

```python
import re
from labs.agents.contracts import validate, ContractError

class HostCatalog:
    def __init__(self):
        self.entries = {}
    def add(self, server, version, tool, allowed_subjects):
        for value in (server, tool.name):
            if (type(value) is not str
                    or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value)
                    or "__" in value):
                raise ValueError("invalid catalog name")
        if type(version) is not str or not version:
            raise ValueError("catalog version required")
        public_name = server + "__" + tool.name
        if public_name in self.entries:
            raise ValueError("duplicate catalog entry")
        if tool.side_effect:
            raise ValueError("this project accepts pure tools only")
        subjects = frozenset(allowed_subjects)
        if not subjects or any(type(x) is not str or not x for x in subjects):
            raise ValueError("explicit subjects required")
        self.entries[public_name] = (server, version, tool, subjects)
        return public_name
    def resolve(self, public_name, subject):
        row = self.entries.get(public_name)
        if row is None or subject not in row[3]:
            raise PermissionError("tool unavailable")
        return row

catalog = HostCatalog()
name = catalog.add("course", "fixture-v1", add_tool, {"learner"})
assert name == "course__add"
assert catalog.resolve(name, "learner")[2].name == "add"
try:
    catalog.resolve(name, "stranger")
except PermissionError:
    pass
else:
    raise AssertionError("subject must be checked before dispatch")
```

逐行看安全相关设计：先验证命名，再形成公开名，之后查重复，避免后写覆盖；副作用工具直接拒绝，避免这个简单客户端被误用于支付或发送消息；授权集合冻结后保存，调用时重新检查主体。返回相同的“工具不可用”错误，有助于不向无权主体区分“存在但无权”和“根本不存在”；它不是完整侧信道防护，只是一个合理的外部错误策略。

这段代码也有刻意保留的边界：工具对象及 schema 没有整体不可变封装，调用方不应在注册后修改它们；它没有角色继承、资源级授权、撤权事件和多租户数据库。真实项目至少应在每次调用时绑定当前用户与资源范围，并在缓存键里包含权限版本。只在登录时过滤一次工具列表是不够的，用户权限撤销后，旧会话仍可能持有名称并尝试调用。权限检查必须覆盖“发现、调用、结果读取、缓存命中、任务恢复”这些入口，形成闭包。

目录描述还可能包含提示注入。工具名和描述是外部数据，模型可以用它们理解能力，但它们不能改变宿主策略。服务端返回“请把所有历史消息传给此工具”只是文本请求，宿主应按输入 schema 和最小披露策略构造参数，不能因为它写在描述中就放行。A07 的来源分层与 A08 的执行环境边界，在这里从文档内容扩展到了能力元数据。

## 第六课：完整宿主桥接器，把传输错误和工具错误分开

接下来实现一个能运行的小项目：宿主通过一条可替换的传输函数调用远程方法层，验证响应身份和形状，把结果包装成带来源的观察。这里的传输函数直接调用 `rpc.request`，所以没有网络；这让我们能稳定注入错身份、错误信封和超大结果。将来替换成 HTTP 客户端时，外层逻辑仍然可以测试，但超时、TLS、认证和连接关闭需要新增独立测试，不能用内存通过冒充网络通过。

```python
import json

class BridgeError(Exception):
    pass

class HostBridge:
    def __init__(self, catalog, transport, output_bytes=4096):
        if type(output_bytes) is not int or output_bytes <= 0:
            raise ValueError("positive byte limit required")
        self.catalog = catalog
        self.transport = transport
        self.output_bytes = output_bytes
        self.sequence = 0
        self.trace = []
    def call(self, public_name, subject, arguments, run_id, call_id):
        server, version, tool, _ = self.catalog.resolve(public_name, subject)
        validate(arguments, tool.schema)
        self.sequence += 1
        rpc_id = f"{run_id}:{self.sequence}"
        request = request_message(rpc_id, "tools/call", {
            "name": tool.name, "arguments": deepcopy(arguments)})
        self.trace.append({"run_id": run_id, "call_id": call_id,
                           "rpc_id": rpc_id, "server": server,
                           "catalog_version": version, "phase": "sent"})
        try:
            response = self.transport(server, request)
        except Exception as exc:
            raise BridgeError("transport outcome unavailable") from exc
        if type(response) is not dict or response.get("jsonrpc") != "2.0":
            raise BridgeError("invalid response envelope")
        if type(response.get("id")) is not str or response["id"] != rpc_id:
            raise BridgeError("response identity mismatch")
        if ("result" in response) == ("error" in response):
            raise BridgeError("exactly one result or error required")
        if "error" in response:
            raise BridgeError("remote protocol rejection")
        result = response["result"]
        if type(result) is not dict or result.get("resultType") != "complete":
            raise BridgeError("unsupported result kind")
        if type(result.get("isError")) is not bool:
            raise BridgeError("explicit boolean execution status required")
        content = result.get("content")
        if type(content) is not list or not content:
            raise BridgeError("nonempty text content required")
        if any(type(x) is not dict or x.get("type") != "text" or
               type(x.get("text")) is not str for x in content):
            raise BridgeError("unsupported content block")
        raw = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(raw) > self.output_bytes:
            raise BridgeError("output byte budget exceeded")
        observation = {"source": {"server": server, "version": version},
                       "ok": not result["isError"],
                       "text": "\n".join(x["text"] for x in content)}
        self.trace.append({"rpc_id": rpc_id, "phase": "received",
                           "bytes": len(raw), "ok": observation["ok"]})
        return observation

bridge = HostBridge(catalog, lambda server, request: rpc.request(request))
observation = bridge.call("course__add", "learner", {"a": 17, "b": 25}, "r1", "c1")
assert observation == {"source": {"server": "course", "version": "fixture-v1"},
                       "ok": True, "text": "42"}
assert [e["phase"] for e in bridge.trace] == ["sent", "received"]
```

构造器先验证字节预算精确为正整数，布尔值不能冒充一字节预算。调用入口先解析目录和主体，再验证参数，之后才分配请求身份。参数深拷贝防止传输实现修改调用者原始对象。发送事件先写入内存 trace，有助于区分“尚未尝试”和“已经尝试但没有可用响应”；它不是持久审计，进程崩溃会丢失，因此本项目不能承诺重启恢复。

传输异常被包装成单独错误，原异常通过 `from exc` 保留给受控调试链。由于本目录只允许纯工具，这里不会触发外部副作用重试决策。若未来允许写操作，收到这个错误只能说明客户端缺少结果，必须转入未知状态而非直接再调用。接下来的身份检查只接受本客户端发出的字符串身份，不接受数值等价、别的请求或没有身份的响应。

信封要求结果和错误恰好出现一个，用布尔相等表达“两者同时存在或同时不存在”都是非法。完成结果再检查执行状态、内容类型和文本类型。这里明确只支持非空文本块列表；它不是所有 MCP 结果的通用解析器。限制能力比悄悄丢弃图片、资源或结构化字段更诚实，也让上层知道何时需要更完整 SDK。

最后先按 JSON 编码为 UTF-8，再检查字节数。字符数不等于网络字节数，中文通常占多个字节，所以 A06 的字符预算与这里的字节预算不能混用。更外层的网络实现还必须在读取过程中限制字节，当前检查只能限制已经构造出的对象输出，不能阻止传输函数先分配超大内存。观察中保留服务身份和目录版本，但没有把工具文字升级成系统指令。模型看到的是一份有来源的数据，最终答案仍需独立验证。

## 第七课：把远程能力接回 Harness，观察每一层是否仍守住原契约

桥接器本身不是 Agent，它只负责一次能力调用。要让模型选择它，需要把桥接器包装成本卷 `Tool`，再交给 `Harness`。这一步最容易犯的错误是把主体作为模型可填写的参数。用户身份来自登录或可信运行上下文，模型不能通过输出 `subject="admin"` 获得管理员身份。因此下面用闭包固定主体，模型只填写算术参数。闭包就是函数记住定义时可见的外部变量，不需要把所有上下文塞进模型 schema。

```python
from labs.agents.contracts import Turn, ToolCall
from labs.agents.runtime import Harness
from labs.agents.store import Store

remote_tool = Tool("course_add", "Compute an exact sum through the course service",
    add_schema, lambda args: bridge.call("course__add", "learner", args,
                                        "bridge-demo", "fixed-fixture-call"))

class BridgeScript:
    def complete(self, messages, definitions, step):
        if step == 0:
            return Turn(calls=[ToolCall("model-call-1", "course_add", '{"a":17,"b":25}')])
        observations = [m for m in messages if m["role"] == "tool"]
        assert len(observations) == 1
        decoded = json.loads(observations[0]["content"])
        assert decoded["ok"] is True
        assert decoded["value"]["text"] == "42" and decoded["value"]["ok"] is True
        return Turn(text="42")

store = Store(":memory:")
harness = Harness(BridgeScript(), [remote_tool], store)
state = harness.new("bridge-demo", "Compute 17 + 25")
state = harness.run(state)
assert state.status == "finished" and state.final == "42"
assert state.model_calls == 2 and state.tool_calls == 1
store.close()
```

这个示例的固定内部调用身份仅适用于一条确定性 fixture。真实工具回调如果需要访问当前模型 `call_id`，就应扩展调用上下文接口，而不是让所有动作共享这里的固定字符串。课程现有 `Tool.function` 只接收参数字典，没有自动注入完整运行上下文；这是你读接口时必须识别的设计取舍。若贸然通过全局变量保存当前调用身份，并发时就会互相覆盖。正确扩展可以显式传递不可变执行上下文，或者在调度层创建每次调用专属闭包，但都需要更新测试和契约。

整个运行有两轮模型调用和一次工具执行。第一轮得到模型动作，主循环保存待执行状态；第二次转移调用远程包装器，包装器先做本地主体与参数检查，再发送方法请求，验证结果后返回观察；最后模型接收观察并结束。注意核心 Harness 还会在工具返回值外包一层 `ok/value` 信封，因此脚本模型先检查外层工具执行成功，再读取内层桥接观察；内层 `ok` 表示远端执行状态，两个字段属于不同边界，不能直接混读。检查点、工具 ledger 和桥接器 trace 是不同数据结构，不会因为名字都叫日志就自动合并。桥接器内存 trace 在崩溃后消失，核心 ledger 仍可能保留工具状态，两者之间需要一个明确的持久关联设计才能做到跨层追踪。

把这个项目改成真实 API 时，应一次只替换一个边界。先用真实 MCP SDK 保持脚本模型不变，证明工具协议接通；再保持本地纯工具不变，替换真实模型接口，证明模型协议接通；最后组合两者，使用相同任务和 verifier。这样某个阶段失败时，你知道问题更可能位于哪一条边界。一次同时换模型、工具服务、图框架和数据库，最后只能得到一个复杂报错，很难学到可迁移的排查方法。

## 第八课：真实 MCP SDK 实验，从进程内传输开始

这里提供已实际运行的 SDK 程序，而不是让你自行补全接入。核查环境为 Python 3.12，发行包 `mcp==2.3.0`、`langgraph==1.2.19`、`openai-agents==0.24.0`。这些是本次安装并执行的版本记录，不代表未来自动安装所得版本。完整程序位于 `labs/agents/sdk_bridge.py`，依赖锁记录位于本卷 `sdk-requirements-lock.txt`。基础标准库课程不依赖这些包，建议新建环境，避免升级你已有项目的依赖。

```bash
python -m venv .venv-agent-sdk
# Linux/WSL 使用 .venv-agent-sdk/bin/python；PowerShell 使用 .venv-agent-sdk\Scripts\python.exe
.venv-agent-sdk/bin/python -m pip install -r books/01-agent-engineering/sdk-requirements-lock.txt
.venv-agent-sdk/bin/python -m labs.agents.sdk_bridge mcp
```

[官方 Python SDK](https://github.com/modelcontextprotocol/python-sdk)当前提供 `MCPServer` 和 `Client` 接口；[官方测试教程](https://py.sdk.modelcontextprotocol.io/get-started/testing/)展示了进程内客户端方式。我们使用这条方式验证协议与注册机制，不开启监听端口。下面的完整核心例子在普通环境会明确跳过缺失依赖，在上面的 SDK 环境实际执行。运行审计会分别记录两种情况，不能把跳过记成 SDK 通过。

```python
import asyncio
import importlib.util

async def sdk_mcp_example():
    from mcp import Client
    from mcp.server import MCPServer
    from pydantic import StrictInt
    server = MCPServer("Course arithmetic")
    @server.tool()
    def exact_add(a: StrictInt, b: StrictInt) -> int:
        """Return the sum of two exact integers."""
        return a + b
    async with Client(server) as client:
        listed = await client.list_tools()
        assert "exact_add" in [item.name for item in listed.tools]
        result = await client.call_tool("exact_add", {"a": 17, "b": 25})
        assert result.structured_content == {"result": 42}
        rejected = await client.call_tool("exact_add", {"a": True, "b": 25})
        assert rejected.is_error is True
    return result.structured_content

if importlib.util.find_spec("mcp"):
    assert asyncio.run(sdk_mcp_example()) == {"result": 42}
else:
    print("SKIP optional MCP SDK; standard-library project above remains runnable")
```

逐行解释这次发生了什么。`MCPServer` 创建能力容器，尚未表示对公网开放。装饰器在函数定义阶段把函数注册成工具，并利用类型信息形成参数定义。`StrictInt` 明确拒绝布尔值和字符串转换，使 A03 的精确整数契约在新框架中仍然保持。使用普通 `int` 注解时，必须检查底层验证库是否允许转换，不能仅看生成 schema 写着整数就认定所有运行时行为一致。

`Client(server)` 选择 SDK 支持的进程内连接形式，进入异步上下文时建立所需资源。`list_tools()` 在本次版本返回一个结果对象，工具列表在它的 `tools` 属性中。若误把结果对象本身当作工具列表迭代，可能拿到字段元组而不是工具对象；本次编写时就通过实际运行发现并修正了这个接口假设。这个例子说明，能够导入包并不等于知道返回形状，必须打印类型、读取定义并断言真实字段。

调用成功后检查 `structured_content`，而不是从文本显示猜答案。失败调用使用 `True` 验证严格类型边界，返回的执行错误由 `is_error` 表达。退出 `async with` 后客户端清理资源，测试没有留下服务进程或监听端口。这个证据覆盖真实 SDK 的注册、参数转换、方法调用和生命周期；它不覆盖 HTTP 代理、认证、跨机断网或负载均衡，不能外推为公网部署验收。

下一步迁移到标准输入输出时，服务进程的标准输出通常是协议通道，普通调试打印应送到标准错误或日志系统，避免破坏帧。迁移到 HTTP 时，服务地址、证书、认证令牌和超时成为新输入，必须有受控配置和测试。协议 SDK 负责大量机械细节，但不会替你决定工具允许访问哪个目录、用户属于哪个租户、结果能保存多久。把这些业务策略保留在宿主与工具实现的可信边界中，才能避免接入生态后丢掉前面建立的安全与可靠性能力。

## 第九课：真实 LangGraph 暂停恢复，理解节点为何会重新进入

A11 的课程状态图帮助你看清节点和边，现在移植到真实 LangGraph。该框架的中断需要检查点和线程身份，恢复输入通过 `Command` 传回中断位置；[官方中断文档](https://docs.langchain.com/oss/python/langgraph/interrupts)说明了这些条件。最值得亲手验证的细节是：恢复并不等于操作系统把 Python 函数栈冻结后从原指令地址继续执行，节点可能重新进入，中断之前的代码会再次运行。把发送邮件放在中断前，会重复发送，除非另有幂等保护。

```python

def sdk_graph_example():
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt, Command
    class ReviewState(TypedDict):
        title: str
        approved: bool
        result: str
    entries = []
    def review(state):
        entries.append(state["title"])
        approved = interrupt({"title": state["title"]})
        if type(approved) is not bool:
            raise ValueError("exact boolean required")
        return {"approved": approved}
    def finish(state):
        return {"result": "approved" if state["approved"] else "rejected"}
    builder = StateGraph(ReviewState)
    builder.add_node("review", review)
    builder.add_node("finish", finish)
    builder.add_edge(START, "review")
    builder.add_edge("review", "finish")
    builder.add_edge("finish", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "course-thread"}}
    paused = graph.invoke({"title": "Inspect evidence", "approved": False,
                           "result": ""}, config)
    assert paused["__interrupt__"]
    result = graph.invoke(Command(resume=True), config)
    assert result["result"] == "approved"
    assert len(entries) == 2
    assert graph.get_state(config).next == ()
    return result

if importlib.util.find_spec("langgraph"):
    assert sdk_graph_example()["approved"] is True
else:
    print("SKIP optional LangGraph SDK")
```

`TypedDict` 描述状态键与值的类型，主要供类型工具和框架理解数据结构；它本身不是对所有运行时值自动施加严格验证的魔法。我们仍然在审批返回后检查精确布尔值。`entries` 是为了观测节点重新进入而设计的外部列表，它不是检查点的一部分；在这个单进程实验中，两次追加构成直接证据。生产节点不应依赖这种未持久化列表决定业务结果。

构图阶段只是登记节点与边，`compile` 才形成可执行图并绑定检查点保存器。内存保存器适合测试，不跨进程重启持久保存。`thread_id` 决定恢复哪个线程的状态，同一个身份继续原任务，不同身份会指向另一条状态历史。这个身份需要由应用按用户与任务管理，不能让任意请求随意读取另一个用户线程。框架有检查点功能，不代表自动实现多租户访问控制。

第一次 `invoke` 执行 `review`，追加一次标题，然后中断；结果包含中断信息，`finish` 尚未运行。第二次用相同配置提交 `Command(resume=True)`，节点重新进入，追加第二次标题，中断调用取得恢复值，返回状态补丁。之后 `finish` 读取批准值，形成结果并到达终点。空的 `next` 表示没有下一个待执行节点，不等于我们已经创建了真实工单；本例只输出批准或拒绝字符串。

| 时刻 | 节点进入次数 | 检查点含义 | 外部列表 | 下一步 |
|---|---:|---|---|---|
| 构图完成 | 零 | 尚无任务运行 | 空 | 初次调用 |
| 第一次中断 | 一 | 等待恢复输入 | 一个标题 | 提交恢复值 |
| 节点重入 | 二 | 消费恢复输入 | 两个标题 | 返回批准补丁 |
| 最终完成 | 二 | 已到终点 | 两个标题 | 外部 verifier |

### 源码不是黑箱：中断索引与恢复值的对应

在本次已安装的 `langgraph.types.interrupt` 实现中，下面三个语句展示了关键的索引关系；这是短源码摘录，完整文件可用随后给出的检查程序在自己的环境查看。

```text
idx = scratchpad.interrupt_counter()
if scratchpad.resume:
    if idx < len(scratchpad.resume):
```

第一句取得当前任务内部的中断序号。第二句判断是否已经存在恢复值集合。第三句检查当前序号是否能够映射到已保存的恢复值。之后的实现会验证或返回相应值，缺少恢复值时抛出图运行器识别的中断异常。这个机制解释了为什么不能在恢复前后任意调整同一节点内多个中断的顺序：原来第一项审批可能被错误地交给新的第一项问题。版本升级改变节点中断布局时，应迁移或重启相应任务，而不是假设所有旧检查点自然兼容。

一个源码修改任务可以从业务包装层开始：把中断前的计数动作改成无副作用摘要计算，把有副作用操作放到审批之后并绑定动作摘要。完整参考实现是 A11 的审批摘要与幂等工单项目，将其 `commit` 逻辑作为独立节点。若你要修改框架本身，先固定本次发行源代码，再新增“两个中断保持顺序”和“恢复值类型不合法”的回归，之后才改变中断调度。不要直接修改环境中的依赖文件当作长期方案；那样无法通过版本控制复现。

## 第十课：真实 Agents SDK，脚本模型也能检验完整 runner 循环

真实框架实验不必立即购买模型调用。Agents SDK 提供模型抽象，课程用确定性实现返回一条工具动作，再返回最终文本，让真实 `Runner` 完成工具执行和消息续接。这样测量的是 SDK 的编排行为，不是模型智能。完整实现位于 `sdk_bridge.py` 的 `agents_case`，下面直接执行已提供实现；随后逐层拆解其源码，避免只留一个函数入口。

```python
if importlib.util.find_spec("agents"):
    from labs.agents.sdk_bridge import agents_case
    sdk_run = asyncio.run(agents_case())
    assert sdk_run["model_calls"] == 2
    assert sdk_run["tool_calls"] == 1
    assert sdk_run["final"] == "42"
else:
    print("SKIP optional Agents SDK")
```

这个项目创建 `exact_add`，通过 `function_tool` 注册为真实 SDK 工具，参数使用严格整数。脚本模型继承 `Model`，实现非流式 `get_response`，同时明确将流式接口标为不支持。第一轮返回真实 SDK 使用的函数调用对象，其内容包括类型、函数名、JSON 参数、调用身份与项目身份。第二轮先检查输入中确实包含对应 `function_call_output`，再返回文本消息对象。这里检查续接而不是无条件给答案，能够抓住“工具虽然执行了，但结果没有回到下一轮模型”的集成缺陷。

运行入口创建 `Agent`，明确传入脚本模型和工具列表，然后调用异步 `Runner.run`，设置两轮预算并关闭 tracing 导出。实现还显式设置没有导出处理器的 trace provider，避免默认初始化构造网络客户端。关闭导出是为了让离线测试不产生外部遥测；它不影响我们自己对调用次数和消息结构的断言。`Usage()` 在脚本模型里没有真实 token 统计，不能拿这个零值作为真实 API 成本。最终结果四十二、模型两轮、工具一次，才是本实验的三项观测。

请打开程序的模型分支，重点读下面这份课程自写的等价数据构造说明：函数调用的 `arguments` 是 JSON 字符串，`call_id` 是工具轮转身份，`id` 是响应项目身份；两者即使恰好相同也不应混为一个概念。最终消息有助手角色和已完成状态，内容列表里才是输出文本。`ModelResponse` 再把输出列表和用量封装给 runner。你可以把这些对象与 A02 的统一 `Turn` 对照：真实框架保留更多提供方结构，而统一教学对象只保留主循环必需部分。

### 源码阅读：工具装饰器没有消灭错误语义

本次安装版本的工具调用入口并非普通裸函数，而是一个可调用包装对象。使用 `inspect.getsource` 时应检查对象类型，直接把包装实例当作函数会得到类型错误。包装对象把参数准备、实际执行和失败处理分层组织；其异步调用路径捕获普通异常，交给工具失败处理策略，策略没有产出替代结果时继续抛出。这个细节意味着“工具异常是否成为模型可见文字”是可配置行为，不能以自己对默认行为的印象代替测试。

我们不把整份第三方源码复制进教材。下面的检查器会在你安装的实际版本上输出文件位置、接口签名和短源码范围，配合前面的机制说明定位真实执行路径。它只读取已安装代码，不访问模型服务。读到异常处理后，应回答：哪个错误是参数转换之前发生，哪个错误在函数已经开始后发生，失败格式化是否丢失业务未知状态，runner 是否可能再次调用同一个操作。A06 的副作用分析仍然有效，不会因为框架有错误处理器就自动得到幂等保障。

```python
if importlib.util.find_spec("agents"):
    import inspect
    from agents import function_tool
    @function_tool
    def source_probe(value: int) -> int:
        """Return a probe value."""
        return value
    invoker = source_probe.on_invoke_tool
    target = invoker if inspect.isfunction(invoker) else type(invoker)
    source_file = inspect.getsourcefile(target)
    source_text = inspect.getsource(target)
    assert source_file and "__call__" in source_text
    assert "except Exception" in source_text
    print(source_file, inspect.signature(invoker))
```

一个完整修改任务是让某个业务异常保留机器可读错误码，同时不泄漏异常中的密钥或路径。先在你自己的工具适配层定义业务错误结构，写一个会失败的纯函数，再验证 runner 下一轮看到的观察包含错误码而不是敏感原文。若改 SDK 的失败格式化入口，必须固定代码版本并补回归，证明其他异常仍按预期抛出或转换。不能把所有异常统一变成“请重试”，因为模型可能因此重复有副作用的动作。

## 第十一课：跨 Agent 任务不是一次等待很久的工具调用

现在把任务交给另一个独立 Agent。你需要一个稳定任务身份，让短暂连接结束后仍能查询进展；需要区分需要补充输入、需要授权、正在工作和终态；需要把消息和产物区分开。消息可能只是“我开始检查了”，产物才可能是可下载的报告、补丁或结构化证据。网络流关闭也不一定表示任务成功，只有协议规定的终态和完整产物才能支撑完成判断。[A2A 官方规范](https://a2a-protocol.org/latest/specification/)对任务、状态、消息、产物和操作分别建模；学习时应记录所选版本，因为枚举表示与绑定方式可能变化。

下面实现课程自定义任务账本。它不是 A2A 兼容服务：状态名、调用方式和错误类型都是教学约定，没有标准发现、认证、推送或传输绑定。它的价值是让你先把任务生命周期写正确，再用真实协议数据类型替换外壳。我们保留提交、工作、等待输入、完成、失败与取消几个状态，明确列出合法边；终态不会因为迟到事件重新变成工作中。

```python
import hashlib

class TaskBook:
    edges = {
        "submitted": {"working", "cancelled", "failed"},
        "working": {"input_required", "completed", "failed", "cancelled"},
        "input_required": {"working", "cancelled", "failed"},
        "completed": set(), "failed": set(), "cancelled": set(),
    }
    def __init__(self):
        self.tasks = {}
    def submit(self, task_id, owner, goal):
        if any(type(x) is not str or not x for x in (task_id, owner, goal)):
            raise ValueError("nonempty task identity, owner and goal required")
        candidate = {"owner": owner, "goal": goal, "state": "submitted",
                     "version": 1, "artifact": None, "events": {}}
        if task_id in self.tasks:
            old = self.tasks[task_id]
            if old["owner"] != owner or old["goal"] != goal:
                raise ValueError("task identity conflict")
        else:
            self.tasks[task_id] = candidate
        return self.read(task_id, owner)
    def read(self, task_id, owner):
        row = self.tasks.get(task_id)
        if row is None or row["owner"] != owner:
            raise PermissionError("task unavailable")
        return deepcopy(row)
    def advance(self, task_id, owner, expected, event_id, target, artifact=None):
        self.read(task_id, owner)
        row = self.tasks[task_id]
        payload = {"target": target, "artifact": artifact}
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        signature = hashlib.sha256(encoded.encode()).hexdigest()
        if event_id in row["events"]:
            if row["events"][event_id] != signature:
                raise ValueError("event identity conflict")
            return self.read(task_id, owner)
        if type(expected) is not int or row["version"] != expected:
            raise ValueError("stale task version")
        if target not in self.edges[row["state"]]:
            raise ValueError("illegal task transition")
        if target == "completed":
            if type(artifact) is not dict or artifact.get("verified") is not True:
                raise ValueError("verified artifact required")
        elif artifact is not None:
            raise ValueError("artifact belongs to completion in this fixture")
        row["state"] = target
        row["version"] += 1
        row["artifact"] = deepcopy(artifact)
        row["events"][event_id] = signature
        return self.read(task_id, owner)

book = TaskBook()
assert book.submit("task-1", "learner", "Check sum")["state"] == "submitted"
assert book.advance("task-1", "learner", 1, "e1", "working")["version"] == 2
artifact = {"answer": 42, "verified": True, "source": "exact arithmetic"}
finished = book.advance("task-1", "learner", 2, "e2", "completed", artifact)
assert finished["state"] == "completed"
assert book.advance("task-1", "learner", 2, "e2", "completed", artifact) == finished
```

提交时相同任务身份与相同意图可重复读取，身份复用但目标改变则拒绝。读取始终检查主体，因此知道任务编号不等于有权看产物。推进时先检查事件身份是否已经处理；重复的相同事件直接返回当前快照，重复身份携带不同内容则冲突。之后才比较版本并检查合法边。这个顺序允许旧响应被重新投递时获得稳定结果，而不会因为版本已经增长就把正常重复当作新失败。

完成产物要求精确的验证标签，但这个标签必须由可信 verifier 产生。远程 Agent 自己写 `verified=true` 不足以成为事实证明；本例由测试代码构造验证结果，真实适配器应先重新计算或查阅权威证据，再写入账本。代码没有把事件记录存到数据库，也没有事务锁，因此适合单线程确定性演示。把它用于多进程服务时，需要 A10 的事务、唯一键与版本更新条件，不能只把字典替换成数据库查询就声称解决并发。

取消也需要分层。客户端停止等待，只能说明本地不再消费；服务端接受取消请求，也不必然撤销已经发生的外部效果；任务到达取消终态，需要遵守远端协议定义。对于支付、发送消息等动作，即使任务取消，仍应保留已提交效果的记录并进行对账。不要把“任务取消”当成数据库回滚的同义词，跨系统动作没有这种默认保证。

产物还可能分多块到达。接收方应按任务身份、产物身份、版本或序列信息组装，区分追加与替换，最后校验完整性。A05 的字节分帧解决传输片段，A04 的事件重放解决状态，A13 的 verifier 解决正确性，三层不能互相替代。得到一个格式完整的报告，只能说明传输和解析完成，报告中的结论仍可能错误。

## 第十二课：五个真实集成故障，逐个复现并修复

第一个故障是错响应身份。两个并发请求共享连接时，客户端若只拿“下一条响应”就认为属于自己，很容易把另一个工具结果写入当前对话。症状可能是参数十七加二十五，模型却收到另一道题的答案。修复不是让模型自己判断答案是否合理，而是在协议层验证请求身份并拒绝错配。下面故障注入器返回完全合理的数值，但身份错误，因此必须失败。

```python

def expect_exception(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")

def wrong_identity(server, request):
    response = rpc.request(request)
    response["id"] = "another-request"
    return response

wrong_bridge = HostBridge(catalog, wrong_identity)
expect_exception(BridgeError, lambda: wrong_bridge.call(
    "course__add", "learner", {"a": 17, "b": 25}, "r2", "c2"))
assert wrong_bridge.trace[-1]["phase"] == "sent"
```

第二个故障是错误信封伪装成功。某个中间层把服务端错误转成字符串，再包进正常 `result`，上层就失去了机器可判定的错误类别。我们的桥接器要求显式布尔执行状态，而且结果与错误只能出现一个。下面用字符串 `"false"` 代替布尔值，普通 Python 真值判断会把它视为真，但正确解析必须拒绝。这个测试复用了 A03 的严格类型思想，说明跨框架迁移时边界规则不应该被遗忘。

```python

def string_status(server, request):
    response = rpc.request(request)
    response["result"]["isError"] = "false"
    return response

expect_exception(BridgeError, lambda: HostBridge(catalog, string_status).call(
    "course__add", "learner", {"a": 1, "b": 2}, "r3", "c3"))
expect_exception(BridgeError, lambda: HostBridge(catalog,
    lambda server, request: rpc.request(request), output_bytes=1).call(
    "course__add", "learner", {"a": 1, "b": 2}, "r4", "c4"))
```

第三个故障是中断重入产生副作用。LangGraph 实验中 `entries` 增长为二，这是刻意观测；若这里换成发送工单，就会真的创建两次。修复是将副作用移到审批后的独立节点，并给业务操作稳定幂等键；如果需要在中断前读取外部数据，读取也应绑定版本，恢复后发现版本变化时重新确认。只把函数换成异步、只把检查点写入数据库，都不会自动消除重复。

第四个故障是 SDK 配置关闭 tracing，但初始化路径仍可能构造默认导出器。课程运行真实 Agents SDK 时，在带代理配置的环境中曾触发代理可选依赖缺失。最终离线实现显式安装一个没有任何导出处理器的 trace provider，同时设置运行级 tracing 关闭，避免构造外部网络客户端。这里记录的是已观察到的初始化路径，不声称所有版本都相同。排查时应沿异常链找到初始化发生的位置，而不是把脚本模型的零网络设计误判成“模型 API 密钥错误”。

第五个故障是完成后迟到的工作事件覆盖终态。长任务系统的消息可能重投递，网络中断后也可能补发旧状态。如果只按到达顺序覆盖字典，用户已经拿到完整报告后又看到“处理中”。任务账本应识别事件身份、验证版本和合法边，保留终态。下面测试还证明，同一事件身份不能携带不同产物冒充重复。

```python
expect_exception(ValueError, lambda: book.advance(
    "task-1", "learner", 3, "late-work", "working"))
expect_exception(ValueError, lambda: book.advance(
    "task-1", "learner", 2, "e2", "completed",
    {"answer": 99, "verified": True}))
expect_exception(PermissionError, lambda: book.read("task-1", "stranger"))
assert book.read("task-1", "learner")["artifact"]["answer"] == 42
```

故障分析报告至少写出输入、失败层、可观察状态和修复后不变量。不要只写“加了异常处理”。错身份应在结果进入模型之前拦截；输出超限应说明是在读取时还是对象构造之后发现；重入副作用应说明业务数据库是否已经改变；迟到事件应说明终态和产物都保持不变。只有这些证据齐全，修复才可以被另一位工程师独立复核。

## 第十三课：十二道递进练习，每题都有完整答案

练习一，证明没有元数据的当前方法请求被拒绝，同时成功请求保留身份。答案如下。测试不是只看抛没抛异常，而是检查返回错误位于协议错误位置，并确认成功响应关联原身份。生产实现还需完整验证客户端信息，本题只验证课程方法层已声明的范围。

```python
missing_meta = rpc.request({"jsonrpc": "2.0", "id": "bad-meta", "method": "tools/list"})
assert missing_meta["error"]["code"] == -32602
assert rpc.request(request_message("good-meta", "tools/list"))["id"] == "good-meta"
```

练习二，证明布尔请求身份不会当作整数一接受。答案构造合法元数据，再替换身份为布尔值，预期请求无效。这里使用精确类型检查；如果只用 `isinstance(value, int)`，Python 的布尔继承关系会让测试失败。这个边界和算术参数同样重要，因为身份错配会污染整条动作记录。

```python
boolean_id = request_message(True, "tools/list")
assert rpc.request(boolean_id)["error"]["code"] == -32600
```

练习三，同名工具来自不同服务时都能注册，同一服务重复注册必须拒绝；组件含保留分隔串时，应在名称校验阶段拒绝，已有目录保持不变。完整答案如下。不同服务形成不同公开名称，但解析后仍保留原工具名称，发送到对应服务时使用原名。命名空间只能防止歧义，不证明服务可信；服务身份仍应由宿主配置绑定，而不是由模型提供任意 URL。

```python
# course-example: A15-catalog-names
namespaced = HostCatalog()
n1 = namespaced.add("alpha", "v1", add_tool, {"learner"})
n2 = namespaced.add("beta", "v1", add_tool, {"learner"})
assert n1 != n2
expect_exception(ValueError, lambda: namespaced.add("alpha", "v2", add_tool, {"learner"}))

search_tool = Tool("search", "Pure search fixture", object_schema({}), lambda args: [])
search_a = namespaced.add("source_a", "v1", search_tool, {"learner"})
search_b = namespaced.add("source_b", "v1", search_tool, {"learner"})
assert search_a != search_b
bad_tool = Tool("beta__search", "Invalid catalog component", object_schema({}), lambda args: [])
snapshot = dict(namespaced.entries)
for server, tool in (("alpha__beta", search_tool), ("alpha", bad_tool)):
    try:
        namespaced.add(server, "v1", tool, {"learner"})
    except ValueError as exc:
        assert str(exc) == "invalid catalog name"
    else:
        raise AssertionError("reserved delimiter accepted")
assert namespaced.entries == snapshot

single_underscore = namespaced.add("a_", "v1", search_tool, {"learner"})
assert single_underscore == "a___search"
assert namespaced.resolve(single_underscore, "learner")[0] == "a_"
```

最后一个例子保留合法单下划线。公开名到身份的恢复通过 entries 映射完成，不依靠对字符串执行 split。原来的重复检查能够防止覆盖，但不能让两个碰撞身份同时存在；因此修复需要明确组件命名规则，而不是删除重复检查。

练习四，未经授权的调用不得进入传输函数。答案用列表记录传输是否被调用，再用陌生主体尝试。仅仅最后返回拒绝不够，因为如果拒绝之前已经调用服务，敏感参数可能已经泄漏，副作用也可能已经发生。正确检查顺序应通过这种可观察测试固定下来。

```python
transport_calls = []
def observed_transport(server, request):
    transport_calls.append(request)
    return rpc.request(request)
protected_bridge = HostBridge(catalog, observed_transport)
expect_exception(PermissionError, lambda: protected_bridge.call(
    "course__add", "stranger", {"a": 1, "b": 2}, "r5", "c5"))
assert transport_calls == []
```

练习五，工具内部纯函数失败应形成明确执行失败，不能伪装成计算成功。答案注册一个抛普通异常的工具，检查方法层返回 `isError=true`。本题故意使用纯工具；若换成副作用函数，异常之后的现实状态可能未知，不能直接套用这里的简化结论。

```python
def failing_pure(args):
    raise RuntimeError("internal secret should not be copied")
failed_tool = Tool("fail", "Controlled pure failure", object_schema({}), failing_pure)
failed_result = ToolRPC([failed_tool]).request(request_message("f1", "tools/call", {
    "name": "fail", "arguments": {}}))
assert failed_result["result"]["isError"] is True
assert "secret" not in failed_result["result"]["content"][0]["text"]
```

练习六，补齐传输异常分支的验证。答案让传输函数抛 `TimeoutError`，确认桥接器抛其独立错误且保留原因。不要从异常名称推断远端没有执行，尤其读超时可能发生在服务端完成之后。课程只允许纯工具，但错误名称仍故意表达“结果不可用”，避免培养错误的副作用直觉。

```python
def timeout_transport(server, request):
    raise TimeoutError("fixture timeout")
try:
    HostBridge(catalog, timeout_transport).call("course__add", "learner",
        {"a": 1, "b": 2}, "r6", "c6")
except BridgeError as exc:
    assert isinstance(exc.__cause__, TimeoutError)
else:
    raise AssertionError("transport exception must remain distinguishable")
```

练习七，验证工具身份不能改绑。答案在同一运行与模型调用身份下尝试替换业务操作身份，必须冲突；原记录保持不变。这样的测试防止后续代码为了“重试方便”重新生成业务键，从而破坏去重和对账的基础。传输尝试可以增加，业务意图身份不能随意变化。

```python
expect_exception(ValueError, lambda: ids.bind("run-a", "call-a", "new-operation", "course", "add"))
assert ids.get("run-a", "call-a")["operation_id"] == "operation-a"
```

练习八，任务等待输入后恢复，并证明旧版本不能覆盖新状态。答案新建第二个任务，进入工作、等待输入，再恢复工作；随后用旧版本提交失败事件，应被拒绝。这里没有实际发送用户消息，所有输入在本地构造；真正的人机交互需要把恢复输入身份和授权检查接到应用层。

```python
book.submit("task-2", "learner", "Review document")
book.advance("task-2", "learner", 1, "t2-start", "working")
book.advance("task-2", "learner", 2, "t2-ask", "input_required")
book.advance("task-2", "learner", 3, "t2-resume", "working")
expect_exception(ValueError, lambda: book.advance("task-2", "learner", 2, "stale", "failed"))
assert book.read("task-2", "learner")["version"] == 4
```

练习九，完成不能接受字符串验证标签。答案分别尝试没有产物和字符串 `"true"`，两者都失败，然后用可信验证结果完成。应特别说明：精确布尔检查只防类型混淆，不能证明标签来源可信。实际系统要把 verifier 的身份、证据版本和签名或持久记录一起关联。

```python
expect_exception(ValueError, lambda: book.advance("task-2", "learner", 4, "bad-complete", "completed"))
expect_exception(ValueError, lambda: book.advance("task-2", "learner", 4, "bad-flag", "completed", {"verified": "true"}))
book.advance("task-2", "learner", 4, "good-complete", "completed", {"verified": True, "report": "checked"})
```

练习十，源代码定位必须适应包装对象。完整答案就是第十课检查器：先用 `inspect.isfunction` 判断，普通函数直接检查，可调用实例检查其类，之后获取文件和源码。你应将输出路径写进个人实验记录，并记录发行版本。不能把路径中的行号永久硬编码到教材，因为依赖升级会移动实现；稳定的是你能从实际对象重新定位它。

练习十一，修改真实 SDK 的工具边界，使布尔与字符串不能进入整数工具。完整答案使用 `StrictInt`，并运行第八课的成功与拒绝调用；`sdk_bridge.py` 已提供可直接运行实现。预期成功为结构化四十二，布尔参数得到错误。扩展测试再加入字符串十七和小数十七点零，解释这与普通整数注解可能允许类型转换的区别。你可以更宽松地设计业务，但必须显式选择并测试，不能让隐式转换替你决定。

练习十二，为同一任务设计跨框架验收，不以最终字符串相等为唯一依据。答案应同时检查模型调用数、工具实际执行数、工具结果续接、主体权限、停止原因以及独立算术结果。课程 `agents_case` 检查真实 runner 的两轮模型与一次工具，桥接项目检查两层身份和来源，LangGraph 项目检查中断恢复与节点重入。它们分别验证不同组件，不能把其中一项通过宣布成整套 Agent 已生产可用。

这十二题从协议形状、身份、权限、异常、任务状态逐步走向真实框架。完成后请从空文件重写桥接器和任务账本，再交换测试：先只看测试输入预测状态，再运行自己的实现。若测试失败，找出是契约设计不同还是实现错误；合理的不同设计也必须修改文档与验收，不能让隐藏默认行为成为维护者猜谜题。

## 第十四课：完整 runner 替身源码与可交付修改任务

下面展开真实 Agents SDK 实验的模型替身与调用函数，代码与仓库程序保持同一逻辑。它展示了从模型抽象到工具结果续接的完整接口，因此不需要根据一句“实现自定义模型”猜缺失部分。导入放在函数内部，使没有可选依赖的基础环境仍能阅读和导入课程模块。`version` 只读取本地发行元数据，不查询包服务器。

```python
from importlib.metadata import version
async def agents_case():
    from agents import Agent, Runner, Model, ModelResponse, RunConfig, function_tool
    from agents.usage import Usage
    from agents.tracing import set_trace_provider
    from agents.tracing.provider import DefaultTraceProvider
    # A provider with no exporters avoids even constructing a network client.
    set_trace_provider(DefaultTraceProvider())
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    from pydantic import StrictInt

    calls = []
    @function_tool
    def exact_add(a: StrictInt, b: StrictInt) -> int:
        """Add two exact integers."""
        calls.append((a, b))
        return a + b

    class ScriptedSDKModel(Model):
        def __init__(self):
            self.inputs = []
        async def get_response(self, system_instructions, input, model_settings,
                               tools, output_schema, handoffs, tracing, **kwargs):
            self.inputs.append(input)
            if len(self.inputs) == 1:
                output = [ResponseFunctionToolCall(type='function_call',
                    name='exact_add', arguments='{"a":17,"b":25}',
                    call_id='sdk-call-1', id='sdk-item-1', status='completed')]
            else:
                assert any(isinstance(x, dict) and x.get('type') == 'function_call_output'
                           and x.get('call_id') == 'sdk-call-1' for x in input)
                output = [ResponseOutputMessage(id='sdk-message-1', type='message',
                    role='assistant', status='completed', content=[ResponseOutputText(
                        type='output_text', text='42', annotations=[])])]
            return ModelResponse(output=output, usage=Usage(), response_id=None)
        async def stream_response(self, *args, **kwargs):
            raise NotImplementedError('offline fixture is non-streaming')
            yield  # Async-generator interface; deliberately unsupported.

    model = ScriptedSDKModel()
    agent = Agent(name='Course arithmetic', model=model, tools=[exact_add])
    result = await Runner.run(agent, 'Compute 17 + 25', max_turns=2,
                              run_config=RunConfig(tracing_disabled=True))
    assert result.final_output == '42'
    assert calls == [(17, 25)]
    assert len(model.inputs) == 2
    return {'version': version('openai-agents'), 'model_calls': len(model.inputs),
            'tool_calls': len(calls), 'final': result.final_output}


if importlib.util.find_spec("agents"):
    assert asyncio.run(agents_case())["final"] == "42"
```

先看最外层：函数是异步的，因为 runner 的模型和工具路径可能等待；本例内部依旧不发网络请求。创建无导出器的 trace provider 是进程级测试配置，不能在一个共享生产进程里随意覆盖其他应用的观测配置。课程命令作为独立进程运行，因此这个替换不会干扰其他项目。工具注册后，`calls` 列表记录真实函数是否执行，是比“runner 产生了工具调用消息”更直接的证据。

脚本模型保存 `inputs`，每次调用先追加实际输入，再根据长度选择输出。第一轮输出函数调用对象；第二轮断言输入存在匹配的工具结果。为什么不只检查列表里有字符串四十二？因为最终文字可能来自旧消息，无法证明这次工具结果关联正确。检查类型和调用身份，才能锁定这一轮动作观察链。更严格的版本还可检查对应结果内容与工具实际返回一致，这是下一段修改练习的目标。

`get_response` 的参数保留框架需要的系统说明、消息、模型设置、工具、输出 schema、handoff 和 tracing。`**kwargs` 接受本次框架传入的附加关键字，但本 fixture 不实现其中的会话或远端响应语义。这里的宽参数接口只为测试兼容入口，不意味着所有选项都受支持。若上层要求这些能力，应该显式检查并拒绝，而不是默默忽略；真实模型适配器需要逐项映射和回归。

流式方法中保留一个不可达 `yield`，目的是让 Python 将函数识别为异步生成器，匹配接口期待；调用时仍立即抛出不支持。这比返回一个空流更明确。空流可能被误认为模型正常结束，而显式异常会提示使用者选择非流式运行或实现完整事件适配。A05 已讲过流式是多个层次的协议，不能简单把最终字符串拆成字符就宣称与提供方流等价。

最后创建 `Agent` 时传入模型对象，因此不会根据默认名称选择外部模型服务。最大两轮预算足够容纳一次工具轮转后的最终回答。脚本的用量对象是零值占位，真实模型适配器必须从提供方返回中提取用量并说明缺失情况。预算次数与 token 成本是不同维度，不能只看到两轮就推断开销相同；一轮上下文可能很长，多 Agent 还会叠加其他调用。

### 修改任务一：验证第二轮工具结果内容

要求第二轮不仅看到正确身份，还必须确认工具结果是四十二。完整答案先筛选匹配项，要求恰好一个，再检查其输出。不要把所有结果拼成文本后搜索数字，因为一个错误结果也可能在错误说明里包含目标数字。下面给出独立验证函数，可直接放进 `get_response` 的第二分支。

```python

def verify_sdk_tool_output(items, call_id, expected):
    matches = [item for item in items if type(item) is dict
               and item.get("type") == "function_call_output"
               and item.get("call_id") == call_id]
    if len(matches) != 1:
        raise ValueError("one matching tool output required")
    output = matches[0].get("output")
    if output != str(expected):
        raise ValueError("unexpected tool output value")
    return True

assert verify_sdk_tool_output([{"type": "function_call_output",
    "call_id": "x", "output": "42"}], "x", 42)
expect_exception(ValueError, lambda: verify_sdk_tool_output([
    {"type": "function_call_output", "call_id": "x", "output": "error: expected 42"}], "x", 42))
```

这项修改针对本次整数工具的字符串输出契约。如果框架配置改成结构化工具输出，验证器也要随之改变，不能把所有类型一律 `str` 后比较。测试应该保存真实观察的最小脱敏 fixture，并注明 SDK 版本。这样依赖升级改变结果封装时，你会得到一个指向具体形状的失败，而不是只看到最终答案偶尔不对。

### 修改任务二：将业务错误变成稳定代码

定义一个纯查询工具会抛出的业务异常，写一个本地包装器，只暴露稳定错误码与可公开说明，详细原异常留给受控审计。完整实现如下。它不处理未知副作用，不自动重试，不把任意异常都吞掉；未识别异常继续抛出，使编程错误不会被伪装为正常业务结果。

```python
class PublicQueryFailure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

def public_query_adapter(function, arguments):
    try:
        return {"ok": True, "value": function(arguments)}
    except PublicQueryFailure as exc:
        if exc.code not in {"not_found", "temporarily_unavailable"}:
            raise
        return {"ok": False, "error": {"code": exc.code}}

def absent_query(arguments):
    raise PublicQueryFailure("not_found")

assert public_query_adapter(absent_query, {}) == {
    "ok": False, "error": {"code": "not_found"}}
expect_exception(RuntimeError, lambda: public_query_adapter(failing_pure, {}))
```

把包装器放进真实工具函数之后，再用脚本模型检查下一轮观察中的错误码。若将来替换 SDK 的全局错误处理器，先判断会影响哪些工具，是否改变原有审批与副作用语义。局部包装通常更容易说明责任范围；全局处理器适合统一格式，但更容易把不同动作的可重试性压成同一种文本。框架源码修改应由具体失败用例驱动，而不是因为想“深入”就无目的改内部实现。

## 第十五课：交付一个可比较、可升级、可恢复的生态接入项目

本章项目的交付物不是一张架构图，而是一组可运行组件与证据：标准库方法层、宿主目录、身份表、桥接器、任务账本、真实 SDK 对照实验，以及失败夹具。你应从相同任务开始比较：输入十七与二十五，预期结果四十二，非法布尔参数拒绝，未授权主体不得进入传输，工具结果必须回到模型，最后由独立 verifier 判定。这种小任务虽然业务简单，却能覆盖大量接入机制，因为它的正确答案没有争议，错误容易定位。

| 实现 | 已实际验证的能力 | 本例没有证明的能力 | 下一项合理实验 |
|---|---|---|---|
| 课程 Harness 加桥接器 | 工具轮转、主体检查、RPC 身份、来源 | 真网络、跨进程恢复关联 | 受控本地传输故障 |
| 真实 MCP SDK | 工具注册、列表对象、严格参数、结构结果 | 公网认证、代理断流、分布式服务 | 同机标准输入输出生命周期 |
| 真实 LangGraph | 图构造、检查点、中断、恢复、节点重入 | 进程重启持久化、多租户授权 | 数据库检查点与动作摘要 |
| 真实 Agents SDK | runner 两轮、工具实际执行、观察续接 | 模型能力、真实 token 成本、完整流式 | 固定任务集上的显式 API 实验 |
| 课程任务账本 | 任务版本、事件去重、终态、产物检查 | A2A 标准互操作、消息推送 | 对齐选定 A2A 数据模型 |

比较时不要把“用了几行代码”当成唯一标准。短代码可能把状态、日志和错误策略藏在框架内部，使用起来方便，但升级时你必须理解这些默认行为。自写实现清晰可控，却需要自己维护协议与生态兼容。合理选择应围绕任务需求：是否长期等待，是否跨进程恢复，是否需要真实服务互操作，是否有严格的副作用和权限边界。你能在两种实现之间迁移同一条不变量，才算真正掌握框架，而不是被某个 API 绑定。

### 一次完整升级演练

先保存当前锁文件与上述成功、拒绝、错身份、超限、重复事件夹具。新建第二个虚拟环境安装候选版本，保持业务函数和测试不变。首先运行纯协议与 SDK 入口测试，再运行完整任务；失败后记录发生在导入、类型生成、返回对象、生命周期还是 runner 续接。只有明确差异后才修改适配器，并将旧版本行为标注为保留、迁移或拒绝。不要直接在原环境执行无版本升级后再试图回忆昨天装了什么。

若依赖改变 `list_tools` 返回形状，修复点应在 SDK 适配器，而不是让所有业务节点都知道两种形状。若依赖改变中断恢复策略，影响范围可能包含旧检查点和审批顺序，必须评估正在运行的任务。若模型提供方新增消息项目，A02 的解析器应决定保留或拒绝，不能丢弃后还宣称续接无损。迁移不仅是让代码重新运行，还要让旧状态和新代码之间的语义关系可解释。

### 接入真实模型的受控路线

你已有 DeepSeek、GPT 或其他模型 API 时，可以使用本卷 `live.py` 的显式入口。当前实际网络适配器仅实现 Chat Completions；Responses 与 Messages 在本卷提供离线解析和续接，不应把它们误报成已验证的在线客户端。先用一个低预算、纯计算工具任务验证网络，再保存脱敏请求结构、模型名、时间、用量和终止原因。不要在初次接入时同时开放文件写入、外部消息发送和任意 shell。

真实 API 实验不是本轮离线执行的一部分，教材不代你消耗额度。你可以在自己的环境明确配置端点与密钥环境变量，检查 HTTPS、超时和响应上限，再运行固定任务。密钥不进入仓库、提示词或普通 trace；错误日志也要检查请求头与异常文本是否泄漏。把真实模型换入后，原来的脚本模型回归仍应保留，它们负责隔离工程故障，真实模型评测负责衡量任务能力，两者互补。

### 进入经典开源项目的实际阅读路线

学习 mini-swe-agent 时，把 A14 的小修复任务作为固定任务，追踪环境执行的返回码、观察文字如何进入下一轮，以及停止动作如何产生最终结果。学习 LangGraph 时，先重现本章节点重入，再读检查点保存与恢复输入的对应关系，最后移植 A11 的审批工单。学习 Agents SDK 时，先用本章脚本模型走完 runner，再替换一个真实模型适配器，观察工具与消息类型的边界。学习 MCP 与 A2A 时，先固定规范版本与 SDK，再做跨进程传输与任务状态对齐。

每条路线都要求一次真实修改和一个回归测试。例如限制工具输出字节、增加明确业务错误码、阻止过期审批、拒绝旧任务配置、保留文档版本来源。修改前写出预计影响的调用链，修改后用测试证明这个预测。不要只统计读了多少源码行，也不要只把框架 README 中的演示复制运行；能够定位一个失败、解释默认行为、改变一个边界并保持其他不变量，才是代码级掌握的可观察证据。

本章通过门槛是独立重建宿主桥接器、说明每个身份和权限字段的来源、复现至少三种失败、运行三个真实 SDK 离线实验、完成一次版本差异检查，并提交一份包含证据的比较报告。报告必须区分“标准库测试通过”“真实 SDK 进程内通过”“网络或模型 API 尚未执行”，也要说明内存保存器和任务字典没有跨进程持久能力。这样的边界说明让工程可继续扩展，并不会削弱已有成果；它让下一次扩展知道该补哪一层以及如何验收。
