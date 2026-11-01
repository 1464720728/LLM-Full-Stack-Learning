# A10 持久化、幂等与恢复：进程消失后，仍能解释每一次动作

本章从最普通的字典开始，逐步建立 SQLite 快照、版本条件更新、工具调用账本、业务幂等键、outbox 与 inbox，以及带证据的恢复操作。你会在准确的提交窗口注入故障，关闭并重新打开数据库，证明结果能恢复、已完成动作不会被无依据重做。全部实验使用临时本地数据库，不需要网络、模型费用或 GPU。

先定义本章的目标：恢复不是“重新运行直到成功”，而是根据最后持久事实继续未完成工作。已经完成的效果应复用，明确未开始的工作可以重新安排，结果未知的副作用必须查询或核验。这个判断贯穿 A04 状态、A06 可靠性和 A09 调度，本章把它落在数据库行与事务里。

## 第一课：持久记录、数据表与事务

### 内存状态、持久记录与外部效果是三份东西

Python 字典只存在于当前进程。程序退出后，另一个进程不会自动继承它。把字典写成 JSON 文件或数据库记录，可以让下一次启动读取，但“调用过保存函数”并不自动证明整个业务操作原子完成。外部系统可能先改变，数据库后失败；数据库可能保存了请求意图，外部请求还没发出。你需要明确每份事实属于哪里。

假设 Agent 创建工单。内存 State 记录当前节点，工具 ledger 记录调用已开始，工单系统记录真实工单。三者可能在短时间内不一致。正确恢复依赖这种不一致具有明确含义：ledger running 表示可能开始而结果未确认，不应直接等同失败；ledger done 表示本地已有结果，可以回填对话；State pending 表示对话尚未完成这项交换，不证明业务尚未执行。

| 对象 | 保存什么 | 谁是权威 | 崩溃后怎样使用 |
|---|---|---|---|
| 内存 State | 当前执行副本 | 只在本次进程内暂时有效 | 保存失败后应丢弃并重读 |
| runs 快照 | 最后已提交任务状态 | 本地恢复入口 | 决定下一步与版本 |
| tool_ledger | 调用身份、开始与结果 | 本地执行证据 | 复用已记录结果或识别未知 |
| 业务数据库 | 工单等真实对象 | 业务事实 | 按稳定业务键查询对账 |
| 对话 messages | 模型已见的请求与结果 | 协议历史 | 不能替代业务数据库 |

持久化系统设计的第一问应该是：什么时候向外宣布完成。若数据库提交后才报告成功，提交前崩溃会留下未确认状态；若先报告成功再保存，用户可能收到完成通知，但重启后找不到记录。把承诺点与提交点对齐，能减少无法解释的状态。跨系统无法完全对齐时，就需要下面的幂等与消息模式。

本章选择 SQLite，因为标准库即可运行，便于观察事务、唯一约束与条件更新。这不是说 SQLite 只能教学，也不是说它适合所有部署规模。数据库选择应基于并发、部署、备份和运维需求；先学会语义，再讨论替换实现，避免把换数据库误当作自动获得分布式正确性。


### 亲手建立一个表，理解行、主键与事务

数据库表可以先理解为结构化记录集合。列规定字段，行代表一个实体，主键要求身份唯一。SQL 的 INSERT 新增记录，SELECT 查询，UPDATE 修改，DELETE 删除。参数占位符把数据和 SQL 结构分开，避免把用户字符串拼进命令。下面只操作临时内存数据库，先建立最小心智模型。

```python
import sqlite3

connection = sqlite3.connect(":memory:")
connection.execute("CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT NOT NULL)")
with connection:
    connection.execute("INSERT INTO tasks VALUES(?, ?)", ("t1", "prepare report"))
row = connection.execute("SELECT title FROM tasks WHERE id=?", ("t1",)).fetchone()
assert row == ("prepare report",)
try:
    with connection:
        connection.execute("INSERT INTO tasks VALUES(?, ?)", ("t1", "duplicate"))
except sqlite3.IntegrityError:
    pass
else:
    raise AssertionError("duplicate primary key accepted")
assert connection.execute("SELECT count(*) FROM tasks").fetchone()[0] == 1
connection.close()
```

`:memory:` 表示当前连接的内存数据库，关闭后消失，适合快速测试但不具备跨连接持久性。要演示重启，需要使用文件路径。fetchone 返回一行元组，没有结果时为 None；不要把空结果和标题为空字符串混为一谈。主键冲突由数据库拒绝，即使应用层忘了先查，也不会插入第二条同身份记录。

`with connection` 在当前事务正常结束时提交，异常退出时回滚，但它不是“进入就必然取得写锁”的通用保证。在课程默认连接配置下，SELECT 不会自动让后续所有检查具备独占写入权；具体事务何时开始与语句、连接模式有关。后面的 outbox 会展示为什么先查再插入仍可能竞争，以及如何在本例中先执行写语句再核对内容。

事务的价值是让一组本地修改作为一个整体提交。例如扣库存与创建订单若在同一数据库事务中，第二步失败可以回滚第一步；若分开提交，崩溃可能留下只扣库存没有订单。事务范围必须覆盖需要保持的不变量，但不宜把慢网络等待放在持有数据库写锁的事务里，否则会长时间阻塞其他工作，而且也不能让远端 HTTP 自动参与本地回滚。

## 第二课：版本快照与真实重启恢复

### 快照表怎样用版本拒绝旧写入

Store 的 runs 表包含 id、version 和 payload。payload 保存 State 的 JSON，version 是每次成功保存递增的整数。第一次 expected 为零时执行 INSERT，版本变一；后续 UPDATE 带上 `WHERE id=? AND version=?`，只有数据库版本等于调用者观察版本才更新。若受影响行数不是一，就抛 Conflict。

```python
from labs.agents.store import Store, Conflict, UncertainExecution
from labs.agents.runtime import State
from dataclasses import asdict

store = Store()
state = State(run_id="version-demo")
v1 = store.save(asdict(state), expected=0)
assert v1 == 1
first = store.load("version-demo")
second = store.load("version-demo")
first["final"] = "new observation"
v2 = store.save(first, expected=1)
assert v2 == 2
try:
    store.save(second, expected=1)
except Conflict:
    pass
else:
    raise AssertionError("stale write accepted")
assert store.load("version-demo")["final"] == "new observation"
store.close()
```

first 与 second 都读到版本一。first 成功保存后数据库变二，second 再以一作为预期就失败。这叫乐观并发控制：读时不长期持锁，写时检查自己是否仍基于最新版本。它避免盲目覆盖，但不自动合并冲突。调用方应重读并判断任务语义，而不是把 expected 改成最新值强行重存旧内容。

Store.save 先构造带新版本的 payload，再用 json.dumps(allow_nan=False) 编码。不能序列化的状态在写入前失败，避免把非法 JSON 留进数据库。数据库事务回滚不回滚 Python 对象；Harness 通常先修改内存 State 再调用 save，若保存失败，内存已经推进。此时应丢弃这份副本，从持久快照重新加载并查看账本，而不是继续 step。

版本也不是任务拥有者租约。两个 worker 可能在各自保存前都执行外部动作，最后一个保存冲突虽被发现，重复效果已经发生。课程假设单个 run 有单一执行拥有者；版本保护是防盲覆盖的一道边界，不是多 worker 调度协议。A09 已讨论租约与 fencing，不能从 CAS 字样推导整个系统恰好一次。


### 关闭再打开，证明恢复不是内存幻觉

下面创建文件数据库，在模型已经提出工具请求后关闭 Store，再用新 Store 和新 Harness 加载。脚本模型游标由已保存 model_calls 决定，所以新对象从下一次响应继续；工具结果写入后，模型返回最终答案。这里模型是确定性脚本，真实模型适配器的恢复还需兼容配置和请求结果未知处理。

```python
from pathlib import Path
import tempfile
from labs.agents.runtime import Harness, ScriptedModel
from labs.agents.contracts import Turn, ToolCall
from labs.agents.tools import arithmetic_tool

with tempfile.TemporaryDirectory() as folder:
    database = Path(folder) / "runs.sqlite"
    turns = [Turn(calls=[ToolCall("add-1","add",'{"a":17,"b":25}')]), Turn("42")]
    first_store = Store(database)
    first_harness = Harness(ScriptedModel(turns), [arithmetic_tool()], store=first_store)
    first_state = first_harness.new("resume-demo", "add numbers")
    first_harness.step(first_state)
    assert first_state.status == "tools" and first_state.model_calls == 1
    first_store.close()
    second_store = Store(database)
    second_harness = Harness(ScriptedModel(turns), [arithmetic_tool()], store=second_store)
    restored = second_harness.run(second_harness.resume("resume-demo"))
    assert restored.status == "finished" and restored.final == "42"
    assert restored.tool_calls == 1 and restored.model_calls == 2
    second_store.close()
```

新 Harness 没有继承旧 Python State，只通过数据库恢复，证明关键数据确实保存了。模型脚本相同、工具定义相同、业务存储相同，是本实验前提。若恢复时把 add 工具改成乘法，旧 pending 参数仍能通过 schema，但任务语义已经变了。恢复配置兼容性必须检查，不能只凭工具同名继续。

恢复还需要相同权限上下文。昨天用户能读取某资源，不代表今天权限仍有效；旧 pending 在执行前仍应按当前授权政策核验。与此同时，已发生的历史效果不能被新权限状态抹掉，审计应保留。任务继续、历史查看和业务对账可能需要不同权限，生产界面应分别设计。

文件数据库也需要正确关闭与备份。close 释放连接资源，不等于删除数据库；TemporaryDirectory 退出才清理本例文件。真实任务若要长期保存，应放在明确持久位置并建立备份恢复流程。一个只在进程正常退出时保存的 JSON 文件，无法覆盖崩溃窗口；本章每个关键转移都保存，正是为了让故障后有可解释的最近事实。

## 第三课：工具账本与崩溃窗口

### 工具账本的身份、签名与状态

tool_ledger 的主键是 run_id 加 call_id。run_id 区分任务，call_id 区分该任务内模型提出的一次工具调用请求；一次模型响应可以包含多个不同 call_id。signature 是工具名、零字节分隔符和原始参数字符串的 SHA-256，防止同一调用编号被复用为不同动作。status 只有 running 与 done，output 在 done 时保存序列化结果。

```python
import hashlib

ledger = Store()
run_id, call_id = "r", "c"
signature = hashlib.sha256("add\0{\"a\":1,\"b\":2}".encode()).hexdigest()
assert ledger.start(run_id, call_id, signature, side_effect=False) is None
assert ledger.lookup(run_id, call_id, signature)["status"] == "running"
ledger.complete(run_id, call_id, '{"ok":true,"value":3}')
assert ledger.start(run_id, call_id, signature, side_effect=False) == '{"ok":true,"value":3}'
try:
    ledger.start(run_id, call_id, "different", side_effect=False)
except Conflict:
    pass
else:
    raise AssertionError("identity reused with different intent")
ledger.close()
```

start 返回 None 表示调用者应继续执行；返回字符串表示已有完整缓存结果，应直接复用。不能用普通真值判断，因为合法结果字符串理论上可能为空，接口应按 `is not None` 区分。当前 Harness 的输出包裹不会为空，但底层接口仍应保持明确语义。lookup 先核对签名，再返回状态，防止把别的动作结果错当当前结果。

原始参数签名是保守选择。`{"a":1,"b":2}` 与 `{"b":2,"a":1}` 在业务上可能等价，却有不同字符串摘要。课程宁可报冲突，也不擅自推断所有参数等价。若要规范化，必须严格解析并定义数值、默认值、Unicode、字段顺序和省略值的语义；简单 sort_keys 只能解决一部分表示差异，不能替代业务身份。

running 不是“正在某个线程运行”的实时监控标志，而是“已声明开始，尚未保存结果”的持久事实。进程可能已经死了，也可能业务已完成但结果没回来。done 表示本地已有结果，不保证对话已接收，更不保证整个任务完成。把这两个状态的含义说清楚，恢复逻辑才不会误判。


### 三个崩溃窗口，三种不同恢复动作

窗口一，账本尚未开始工具，进程崩溃。若持久状态只有 pending，没有对应 ledger 行，可以在重新验证权限和预算后执行。窗口二，ledger 已 running，业务效果可能发生，但本地没有结果。对纯工具可按声明重新计算，对副作用必须进入不确定并对账。窗口三，ledger 已 done，任务快照还没加入 tool 消息。恢复应取缓存结果回填，不再次进入业务函数。

| 窗口 | 快照 pending | ledger | 业务可能状态 | 恢复原则 |
|---|---|---|---|---|
| 开始前 | 有 | 无 | 未由本次调用启动 | 验证后可执行 |
| 效果与记账之间 | 有 | running | 未发生、进行中或已完成 | 副作用对账 |
| 结果与对话之间 | 有 | done | 已有本地确认结果 | 复用并回填 |
| 对话已提交 | 无对应项 | done | 已交付观察 | 不重复追加 |

这张表最重要的不是记忆状态名，而是理解本地数据库无法观察远端每个瞬间。窗口二跨越两个系统，单靠 SQLite 事务不能原子控制另一家的 HTTP 操作。客户端可以记录意图、使用幂等键、查询结果或执行补偿，但不能凭一个本地 running 标志推断远端未提交。任何宣称恰好一次的框架都应说明其效果范围和下游协议。

下面直接调用教学内部 `_execute` 演示窗口三的缓存重放。它是为了学习账本，不建议应用代码绕过 Harness 的预算和状态流程。副作用用列表追加模拟，真实业务应使用持久存储。

```python
from labs.agents.contracts import Tool, object_schema

actual_effects = []
def effect(args):
    actual_effects.append(args["value"])
    return {"saved":args["value"]}
tool = Tool("save", "teaching effect", object_schema({"value":{"type":"integer"}}), effect, True)
harness = Harness(ScriptedModel([]), [tool])
state = harness.new("cached-effect", "test")
call = ToolCall("same-call", "save", '{"value":7}')
first_output = harness._execute(state, call)
second_output = harness._execute(state, call)
assert first_output == second_output and actual_effects == [7]
harness.store.close()
```

这个测试用真实进入次数证明只执行一次，而不是只看最终文件内容。重复写同样内容可能让文件看起来没变化，却仍产生两次通知、审计或费用。效果计数是验证幂等边界的重要证据。缓存结果字符串完全相同，方便保持模型观察一致；若每次重放重新生成时间戳，就可能制造不同结果。

对 running 副作用，start 会抛 UncertainExecution，不调用函数。对 running 纯工具，start 返回 None，允许重新计算。这个差异完全依赖 side_effect 声明正确。一个偷偷写文件却标为纯的工具，会破坏恢复保证；工具元数据必须由宿主审查和测试，不能由模型参数控制。

## 第四课：调用编号与业务幂等键是两个层次

模型可能用 call-1 创建工单，后来又用 call-2 请求同一工单。调用账本把它们视为两个不同请求，因此仅按 call_id 去重不能防止重复业务对象。业务幂等键表达的是用户要完成的同一个意图，例如 request-42；两次调用若属于同一意图，就应使用同一业务键，由业务系统保证同键同意图返回同一对象。

课程 TicketBook 的 tickets 表以 request_id 为主键。create 先查已有记录，若同键同标题就返回原记录并 created=False；若同键不同标题就报冲突；否则插入 open 工单并 created=True。created 表示本次是否新建，不表示业务目标是否成功。重复请求复用已有工单通常仍是成功完成意图。

```python
from labs.agents.tools import TicketBook
from labs.agents.contracts import ContractError

book = TicketBook()
first = book.create("request-42", "prepare report")
second = book.create("request-42", "prepare report")
assert first["created"] is True and second["created"] is False
assert book.db.execute("SELECT count(*) FROM tickets").fetchone()[0] == 1
try:
    book.create("request-42", "different title")
except ContractError:
    pass
else:
    raise AssertionError("business key conflict ignored")
assert book.db.execute("SELECT title FROM tickets WHERE request_id=?", ("request-42",)).fetchone()[0] == "prepare report"
book.close()
```

同键不同意图不能静默覆盖，因为那会把“重试创建”变成“修改对象”。若用户确实想改标题，应使用明确更新操作和新的版本检查，而不是复用创建接口偷改。幂等接口通常需要把关键意图字段保存下来，重复请求时比较，避免错误键复用被掩盖为成功。

业务键也不能随每次重试随机生成。它应在逻辑动作建立时生成并持久保存，随后网络重试、进程恢复和模型重复提议都引用同一键。若模型自由生成键，可能每次换一个，业务去重失效。宿主可以为用户确认的动作分配键，并把它作为不可由模型任意改变的执行上下文。

TicketBook 的先 SELECT 再 INSERT 在单拥有者教学场景清楚易读，但多连接并发时仍可能同时查不到，再争抢唯一键。数据库主键会防止两行，但失败方可能收到 IntegrityError，需要显式重读并核对意图，或采用下面 outbox 的写入后核对策略。不要把单进程示例描述成已完整处理所有多写入者竞争。

## 第五课：外部通知与 outbox 竞争

### 本地事务不能把外部通知一起回滚

假设创建工单后要发送通知。若先提交工单，再发送通知，进程可能在两者之间崩溃，工单存在但通知永远没发。若先发通知再提交工单，提交失败时用户收到不存在工单的通知。把网络发送写在数据库 with 块里也不解决：远端发送成功后本地回滚，邮件不会自动收回。

outbox 的办法是把工单和“需要发送通知”的事件写进同一个本地事务。只要工单提交，待发送事件也存在；若事务回滚，两者都不出现。之后独立投递器读取 outbox 并发送。这样把跨系统动作分成可恢复两段，但投递可能重复，因为发送成功后、标记 sent 前仍可能崩溃。

inbox 在接收端以 event_id 去重。重复收到同一事件时不重复执行业务效果，同时核对同编号内容一致。接收端提交成功后，发送端才标记 sent。若发送端标记前崩溃，下次重发，接收端发现已处理并返回相同确认。网络交付至少一次，接收端业务处理通过幂等达到不重复效果；这不等于网络只传了一次。

```python
from labs.agents.advanced_solutions import Outbox

box = Outbox()
box.create("task-1", "only one notification")
try:
    box.deliver_one(crash_after_delivery=True)
except RuntimeError:
    pass
assert box.consumer.execute("SELECT count(*) FROM inbox").fetchone()[0] == 1
assert box.producer.execute("SELECT sent FROM outbox").fetchone()[0] == 0
assert box.deliver_one() is True
assert box.consumer.execute("SELECT count(*) FROM inbox").fetchone()[0] == 1
assert box.producer.execute("SELECT sent FROM outbox").fetchone()[0] == 1
assert box.deliver_one() is False
box.close()
```

本例生产者与消费者使用两个独立数据库连接，故意不共享事务。故障注入发生在消费者提交之后、生产者更新之前，正好覆盖重复投递窗口。若你把两张表放在同一个事务里再宣称解决分布式投递，就没有测试真正问题。这个设计让窗口可见、可重放，也让保证范围清楚。

outbox 还需要事件身份稳定。课程 event_id 是 created 加 task_id，同一任务创建事件只有一个编号。若每次投递重新生成 UUID，消费者就无法识别重复。事件内容也应固定，不能重复发送时读当前工单标题重新生成不同 payload，却仍用同一 event_id；那会造成身份冲突。更新工单应生成新的业务事件与版本。


### 逐行读 outbox 的竞争修复

create 先把 task_id 和 title 编成排序 JSON payload，再进入生产者事务。第一条 SQL 是 INSERT OR IGNORE tasks，而不是 SELECT。写语句使本例在默认连接事务模式下取得相应写入协调，随后 SELECT 读取数据库实际保存的 title 并比较。若已有同键不同标题，抛 ContractError，事务回滚，不能把被忽略的插入当作成功。

接着构造 event_id，INSERT OR IGNORE outbox，再 SELECT 实际 payload 比较。两处核对都必要：任务行可能正确但事件编号被错误复用，或事件内容损坏；只检查任务标题不能证明通知意图一致。任何冲突都应回滚本次事务，让业务状态与待发送事件保持一致。

为什么旧的“先 SELECT，不存在再 INSERT OR IGNORE”可能错？两个连接都先查到不存在，随后一个插入标题甲，另一个插入标题乙被忽略。如果失败方不再读取实际行，它会误以为自己成功创建了标题乙。唯一键保证只有一行，却没有保证这一行符合失败方的意图。写入后核对把身份唯一与内容一致两项不变量连接起来。

消费者同样先 INSERT OR IGNORE inbox，再读取实际 payload 比较。若同编号内容不同，抛冲突，生产者 sent 不能被设为一。否则系统会把错误交付当作成功，丢失真正事件。课程已有双连接回归覆盖这些窗口，修订时不得退回仅靠唯一键忽略冲突的实现。

需要注意这个策略针对当前 SQLite 实现与连接设置。换数据库时应重新检查事务隔离、冲突语法与锁行为，不能逐字复制 SQL 后假设语义完全相同。生产 outbox 还需要批量领取、投递重试、死信、保留和监控；本章先完整实现核心一致性窗口，再说明这些扩展需要什么合同。

## 第六课：恢复配置、证据与预期版本

### 恢复配置与迁移也是持久协议

保存 State 只保存了运行字段，没有自动保存完整代码、工具实现、模型协议和权限版本。恢复时如果配置不兼容，旧数据可能被错误解释。应为运行记录配置指纹，至少覆盖状态 schema 版本、工具定义版本、模型适配协议、关键策略版本与代码提交。指纹相同不是绝对正确证明，但不同应触发兼容检查，而不是静默继续。

配置指纹不应包含明文密钥。密钥可以轮换，运行记录只需保存凭证引用或授权上下文标识，并在恢复时重新获取当前允许的凭证。把密钥混进 payload 或日志，会让数据库备份成为秘密泄漏源。任务参数也可能敏感，持久化范围、加密、访问控制和保留期限应按应用需求设计，公共教材只使用人工数据。

状态 schema_version 与运行 version 不同。前者说明字段结构怎样解释，后者是并发保存次数。增加一个字段可能需要 schema 迁移，却不意味着任务刚执行了一步；任务每保存一次运行 version 增加，也不意味着字段结构改变。把两个概念混在一个整数里，会让恢复和冲突检查难以维护。

迁移应是显式函数，从已知旧结构转换到新结构，验证后再保存，并保留迁移事件。不能对未知字段随意丢弃，也不能给缺失的已完成标志默认 True。若旧版本没有足够信息证明某动作完成，新版本应保留未知或要求核验，而不是为了顺利加载制造事实。迁移测试应包含旧快照样本和无法安全迁移的反例。


### 恢复操作应该携带证据与预期版本

进入 uncertain 后，系统需要一个明确核验动作。操作者应看到 run_id、call_id、工具名、参数摘要、账本状态、业务操作编号和已知时间线。可选结果包括确认已完成并补录结果、确认未发生且允许重新执行、继续等待、执行明确补偿或终止。每种结果都需要证据来源和审计记录，不能提供一个无条件“全部重置继续”的按钮。

确认已完成时，补录结果必须与原调用身份和意图一致，且通过输出合同。确认未发生则需要业务系统保证，不能仅凭暂时查不到。恢复命令应携带预期运行版本，防止操作者基于旧页面操作已改变任务；同一核验编号重复提交应幂等，不同内容复用编号应冲突。人工操作也需要合同，不是人工就可以绕过一致性。

当前 Harness 没有通用 resolve-uncertain API，所以不能直接把任意终态改回 ready。本章后面给出一个独立的本地核验记录项目，展示如何把证据与幂等写入做完整；真正接回 Harness 时还要验证 pending、ledger 与 messages 的关系。这个边界说明现有接口范围，同时让核心核验机制有可执行实现。

恢复日志应保留谁依据什么确认、确认了哪个具体动作、使用哪个版本和产生了什么后续状态。不要只写“管理员点了继续”。若业务结果后来发现有误，需要能追溯当时证据，而不是只能看到被覆盖后的最终状态。追加审计记录比直接改数据库字段更适合这种需要解释的流程。

## 第七课：四个故障复盘

故障一，进程重启后模型跳过了丢失响应。原因是 model_calls 已保存增加，但调用返回前崩溃，恢复脚本直接使用下一个索引。课程 inflight_model 在请求前持久化，恢复看到它就进入 uncertain，而不是自动跳过。修复验收必须模拟真正未执行异常处理的中断，检查持久快照仍标记请求结果未知。

故障二，旧 worker 保存覆盖了新 worker 的结果。原因是 UPDATE 只按 run_id，不检查 version。修复增加预期版本条件与 rowcount 检查，冲突后重读。验收两份副本都读版本一，第一份保存成功，第二份失败且数据库保持第一份结果。注意这只保护保存，不撤销第二个 worker 可能已经产生的外部效果。

故障三，重复请求创建两张工单。原因是仅按 call_id 去重，模型第二次用了新编号。修复业务 request_id 唯一与同意图核对，调用账本仍保留两次请求。验收应看到 ledger 两行、tickets 一行，说明两个身份层各自正确。若只检查 tickets 一行而丢失第二次调用审计，也不完整。

故障四，outbox 消费者收到同编号不同内容，发送端仍标记成功。原因是 INSERT OR IGNORE 后未核对实际 payload。修复在消费者事务中读取并比较，冲突抛出后生产者 sent 保持零。测试必须构造预先存在的不同内容，不只是重复同内容；后者只能证明去重，不能证明冲突检测。

## 第八课：十二道完整练习与答案

练习一，内存数据库关闭后不会自动恢复。答案创建两个独立的 `:memory:` 连接，第二个没有第一份表。这个测试不是数据库故障，而是配置语义。真正恢复测试必须使用文件路径。

```python
first = sqlite3.connect(":memory:")
first.execute("CREATE TABLE x(value INTEGER)")
first.close()
second = sqlite3.connect(":memory:")
assert second.execute("SELECT count(*) FROM sqlite_master WHERE name='x'").fetchone()[0] == 0
second.close()
```

练习二，事务中第二步失败应回滚第一步。答案在同一事务插入两个同主键记录，唯一约束导致异常，表里最终零行。若把第一条单独 commit，结果会不同，因此事务边界是验收的一部分。

```python
db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE x(id TEXT PRIMARY KEY)")
try:
    with db:
        db.execute("INSERT INTO x VALUES(?)", ("same",))
        db.execute("INSERT INTO x VALUES(?)", ("same",))
except sqlite3.IntegrityError:
    pass
assert db.execute("SELECT count(*) FROM x").fetchone()[0] == 0
db.close()
```

练习三，重复 run_id 初次保存应冲突。答案两次 expected=0，第二次不能覆盖第一条。首次创建与后续更新是不同操作，不能用一个无条件 upsert 掩盖任务身份冲突。

```python
store = Store()
record = asdict(State(run_id="same-run"))
assert store.save(record, 0) == 1
try:
    store.save(record, 0)
except Conflict:
    pass
else:
    raise AssertionError("duplicate run creation accepted")
assert store.load("same-run")["version"] == 1
store.close()
```

练习四，同调用编号不同参数应冲突，即使工具是纯函数。答案先记录一个签名，再使用另一个签名。纯工具允许重复计算同一请求，不允许同身份变成另一请求，否则消息关联与审计会失真。

```python
store = Store()
store.start("r","c","signature-1",False)
try:
    store.start("r","c","signature-2",False)
except Conflict:
    pass
else:
    raise AssertionError("changed call accepted")
store.close()
```

练习五，running 副作用不自动重放，running 纯工具可重放。答案对两个调用分别测试 start 行为。这里没有实际调用函数，验证的是账本许可规则；真实 Harness 测试再检查函数进入次数。

```python
store = Store()
store.start("r","effect","s1",True)
try:
    store.start("r","effect","s1",True)
except UncertainExecution:
    pass
else:
    raise AssertionError("unknown effect replayed")
store.start("r","pure","s2",False)
assert store.start("r","pure","s2",False) is None
store.close()
```

练习六，两个 call_id 同一业务键只创建一行。答案使用 TicketBook 工具与 Harness 的执行账本，实际执行两次工具请求，但业务创建只发生一次。第二次输出 created=False，仍包含原对象。

```python
import json
book = TicketBook()
h = Harness(ScriptedModel([]), [book.tool()])
s = h.new("two-calls", "create once")
args = '{"request_id":"business-1","title":"report"}'
a = json.loads(h._execute(s, ToolCall("c1","create_ticket",args)))
b = json.loads(h._execute(s, ToolCall("c2","create_ticket",args)))
assert a["value"]["created"] is True and b["value"]["created"] is False
assert book.db.execute("SELECT count(*) FROM tickets").fetchone()[0] == 1
assert h.store.db.execute("SELECT count(*) FROM tool_ledger").fetchone()[0] == 2
h.store.close(); book.close()
```

练习七，outbox 同业务键不同标题必须拒绝，原记录和事件不变。答案保存旧 payload，冲突后比较。不能把“数据库只有一行”当作充分断言，因为那一行可能被错误覆盖。

```python
box = Outbox()
box.create("a","first")
old_payload = box.producer.execute("SELECT payload FROM outbox").fetchone()[0]
try:
    box.create("a","second")
except ContractError:
    pass
else:
    raise AssertionError("conflicting business intent accepted")
assert box.producer.execute("SELECT title FROM tasks").fetchone()[0] == "first"
assert box.producer.execute("SELECT payload FROM outbox").fetchone()[0] == old_payload
box.close()
```

练习八，消费者身份冲突不能让发送端 sent 变一。答案预先写入同事件编号的错误载荷，再投递真实事件，必须失败。消费者原记录不变，生产者仍待处理，留给核验而不是静默丢失。

```python
box = Outbox()
box.create("a","correct")
with box.consumer:
    box.consumer.execute("INSERT INTO inbox VALUES(?,?)", ("created:a", "wrong payload"))
try:
    box.deliver_one()
except ContractError:
    pass
else:
    raise AssertionError("inbox conflict accepted")
assert box.producer.execute("SELECT sent FROM outbox").fetchone()[0] == 0
assert box.consumer.execute("SELECT payload FROM inbox").fetchone()[0] == "wrong payload"
box.close()
```

练习九，关闭再打开 outbox 两端，重复投递仍一行。答案在消费者提交后模拟崩溃，关闭连接，再用相同文件重新建立对象。它比同对象重试更接近重启，证明去重不依赖内存集合。

```python
with tempfile.TemporaryDirectory() as folder:
    producer_path = Path(folder) / "producer.sqlite"
    consumer_path = Path(folder) / "consumer.sqlite"
    box = Outbox(producer_path, consumer_path)
    box.create("a","persisted")
    try:
        box.deliver_one(crash_after_delivery=True)
    except RuntimeError:
        pass
    box.close()
    box = Outbox(producer_path, consumer_path)
    assert box.deliver_one() is True
    assert box.consumer.execute("SELECT count(*) FROM inbox").fetchone()[0] == 1
    box.close()
```

练习十，模型请求开始后进程中断，恢复应 uncertain。答案用专门的 BaseException 模拟不经过普通 Exception 处理的中断，外层测试捕获后重新加载。这个信号只用于测试，不是生产业务异常。

```python
class SimulatedProcessDeath(BaseException):
    pass
class DiesDuringModel:
    def complete(self, messages, tools, index):
        raise SimulatedProcessDeath()

store = Store()
h = Harness(DiesDuringModel(), [], store=store)
s = h.new("model-death", "test")
try:
    h.step(s)
except SimulatedProcessDeath:
    pass
restored = h.resume("model-death")
assert restored.inflight_model is True and restored.model_calls == 1
h.run(restored)
assert restored.status == "uncertain"
store.close()
```

练习十一，快照保存失败后内存与数据库可能不同。答案让完成响应后的保存失败，内存已经 finished，数据库仍 inflight。重新加载后必须 uncertain，不能继续使用内存完成状态。这个测试直接展示数据库回滚不回滚 Python 对象。

```python
class FailThirdSave(Store):
    def __init__(self):
        super().__init__()
        self.saves = 0
    def save(self, state, expected):
        self.saves += 1
        if self.saves == 3:
            raise OSError("simulated persistence failure")
        return super().save(state, expected)

store = FailThirdSave()
h = Harness(ScriptedModel([Turn("answer")]), [], store=store)
s = h.new("save-failure", "test")
try:
    h.step(s)
except OSError:
    pass
assert s.status == "finished"
restored = h.resume("save-failure")
assert restored.inflight_model is True
h.run(restored)
assert restored.status == "uncertain"
store.close()
```

练习十二，配置指纹应稳定但不能包含密钥。答案对规范化配置编码，字典键顺序不影响结果，工具版本变化会改变结果。显式白名单选择配置字段，避免把整个环境字典序列化进去。

```python
def config_fingerprint(config):
    allowed = {key:config[key] for key in ("state_schema","protocol","tool_versions","policy_version")}
    encoded = json.dumps(allowed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()

config = {"state_schema":1,"protocol":"chat","tool_versions":{"add":"v1"},"policy_version":"p1","api_key":"not-persisted"}
a = config_fingerprint(config)
b = config_fingerprint(dict(config, api_key="rotated"))
c = config_fingerprint(dict(config, tool_versions={"add":"v2"}))
assert a == b and a != c
```

这些练习覆盖的是不同事实边界，不是重复演示 SQL。完成后你应能为每个失败写出数据库实际行、内存状态、业务效果和允许下一步。若某项测试只检查异常而不检查原数据未变，通常还缺少重要验收；若只检查最终成功而不检查效果次数，也可能掩盖重复执行。

## 第九课：完整核验账本项目与故障测试

### 完整小项目，带证据的本地核验账本

下面建立独立的核验服务，专门处理“某动作结果未知，后来获得明确证据”的记录。它不修改 Harness 内部状态，也不自动重试外部动作；它先把核验本身做成有版本、有幂等编号、有冲突检查的事务。这样你可以先验证人工或自动对账入口，再决定如何把确认结果接回 Agent 状态机。

项目有 operations 与 resolutions 两张表。operations 保存动作当前状态和版本，初始为 unknown；resolutions 保存每次核验的编号、动作编号、证据与结果。resolve 只接受 confirmed_done 或 confirmed_not_done，不接受模糊的 maybe；证据来源和说明必须非空。confirmed_not_done 只是一项受信任核验结论，是否足以允许重试仍由业务合同判断。

```python
class ReconciliationBook:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS operations(
            id TEXT PRIMARY KEY, intent TEXT NOT NULL, version INTEGER NOT NULL,
            status TEXT NOT NULL, result TEXT);
        CREATE TABLE IF NOT EXISTS resolutions(
            id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, payload TEXT NOT NULL,
            FOREIGN KEY(operation_id) REFERENCES operations(id));
        """)
        self.db.execute("PRAGMA foreign_keys=ON")

    def register_unknown(self, operation_id, intent):
        encoded = json.dumps(intent, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO operations VALUES(?,?,1,'unknown',NULL)", (operation_id,encoded))
            row = self.db.execute("SELECT intent FROM operations WHERE id=?", (operation_id,)).fetchone()
            if row[0] != encoded:
                raise Conflict("operation identity changed")

    def inspect(self, operation_id):
        row = self.db.execute("SELECT intent,version,status,result FROM operations WHERE id=?", (operation_id,)).fetchone()
        if row is None:
            raise KeyError(operation_id)
        return {"id":operation_id,"intent":json.loads(row[0]),"version":row[1],
                "status":row[2],"result":None if row[3] is None else json.loads(row[3])}

    def resolve(self, *, resolution_id, operation_id, expected_version, decision, evidence, result=None):
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("positive expected version required")
        if decision not in {"confirmed_done","confirmed_not_done"}:
            raise ValueError("explicit decision required")
        if type(evidence) is not dict or set(evidence) != {"source","detail"}:
            raise ValueError("evidence source and detail required")
        if any(type(evidence[key]) is not str or not evidence[key].strip() for key in evidence):
            raise ValueError("evidence must be nonempty strings")
        if decision == "confirmed_not_done" and result is not None:
            raise ValueError("not-done decision cannot carry completed result")
        payload = json.dumps({"operation_id":operation_id,"expected_version":expected_version,
            "decision":decision,"evidence":evidence,"result":result}, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result_json = None if result is None else json.dumps(result, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO resolutions VALUES(?,?,?)", (resolution_id,operation_id,payload))
            saved = self.db.execute("SELECT payload FROM resolutions WHERE id=?", (resolution_id,)).fetchone()
            if saved[0] != payload:
                raise Conflict("resolution identity changed")
            current = self.inspect(operation_id)
            if current["version"] == expected_version + 1 and current["status"] == decision:
                old_result = self.db.execute("SELECT result FROM operations WHERE id=?", (operation_id,)).fetchone()[0]
                if old_result == result_json:
                    return current
            if current["version"] != expected_version or current["status"] != "unknown":
                raise Conflict("stale or already resolved operation")
            cursor = self.db.execute("UPDATE operations SET version=version+1,status=?,result=? WHERE id=? AND version=? AND status='unknown'",
                (decision,result_json,operation_id,expected_version))
            if cursor.rowcount != 1:
                raise Conflict("concurrent resolution")
        return self.inspect(operation_id)

    def close(self):
        self.db.close()

book = ReconciliationBook()
book.register_unknown("op-1", {"type":"create_ticket","business_key":"req-1"})
request = {"resolution_id":"review-1","operation_id":"op-1","expected_version":1,
    "decision":"confirmed_done","evidence":{"source":"business-query:req-1","detail":"record exists with matching intent"},
    "result":{"ticket_id":"ticket-9"}}
resolved = book.resolve(**request)
assert resolved["status"] == "confirmed_done" and resolved["version"] == 2
assert book.resolve(**request) == resolved
assert book.db.execute("SELECT count(*) FROM resolutions").fetchone()[0] == 1
try:
    book.resolve(**dict(request, result={"ticket_id":"different"}))
except Conflict:
    pass
else:
    raise AssertionError("changed resolution accepted")
book.close()
```

先看 register_unknown。它规范化意图，写入后读取实际保存值，避免同编号不同意图被 INSERT OR IGNORE 静默吞掉。初始版本一与 unknown 是宿主确认“需要核验”的状态，不是模型随意声明。真实应用还应验证 operation_id 与当前 run、call_id、业务键的映射，并把授权身份注入服务，不允许外部文档写入核验结论。

resolve 的参数验证发生在事务之前，避免无效输入产生记录。decision 使用封闭集合，evidence 要求精确键与非空字符串，结果必须可标准 JSON 编码。这里证据文字的真实性不能由 schema 自动证明；调用方必须来自受信任的业务查询或授权操作者。结构验证保护数据形状，证据真实性依赖来源与权限，不能混为一项。

事务先写 resolutions 再读实际 payload，得到写入协调并验证幂等身份。若相同核验请求重复到达，数据库已有同 payload；随后检查动作已处于预期的下一版本和同结果，直接返回。若请求编号相同但内容变化，明确冲突。若另一个核验已经改变动作版本，旧请求应冲突，不能覆盖。

这段教学实现对“不同核验编号但完全同结论、同预期版本”会保存第二条相同事实的审计请求并复用当前结果；这可以解释为两次独立确认，而不是重复业务执行。若产品要求每个动作只能有一条核验记录，应增加唯一约束或单独的确认集合政策。幂等单位是 resolution_id，业务终态是 operation_id，两者职责不同，设计时必须明说。

为什么不在这里直接调用工具重试？因为核验事实与执行许可是两个步骤。confirmed_not_done 是否可重试，取决于业务查询一致性、当前权限、剩余预算和任务是否仍有效。将核验记录与执行解耦，可以让上层状态机在重新验证之后安排动作，而不是在数据库事务里夹带外部请求。confirmed_done 则需要把受验证结果回填到对应 ledger，再由单拥有者协调对话状态。


### 核验项目的故障测试与保证边界

核验服务也需要崩溃窗口分析。若 resolutions 插入后、operations 更新前抛异常，两者处于同一事务，应该回滚；若提交后响应丢失，重复同 resolution_id 应返回同结果；若同编号不同证据，必须冲突。不要因为这是人工恢复入口就降低测试要求，恢复过程一旦重复执行，可能比原故障造成更大混乱。

可以使用 SQLite trigger 在 UPDATE 时故意抛异常，验证前面的核验记录也回滚。下面是完整测试；触发器只存在本题内存数据库中，不影响仓库其他数据。删除触发器后同请求可正常执行，证明失败没有留下半条核验。

```python
book = ReconciliationBook()
book.register_unknown("op-fail", {"business_key":"k"})
book.db.executescript("""
CREATE TRIGGER fail_resolution BEFORE UPDATE ON operations
BEGIN SELECT RAISE(ABORT,'planned update failure'); END;
""")
request = {"resolution_id":"resolution-fail","operation_id":"op-fail","expected_version":1,
    "decision":"confirmed_not_done","evidence":{"source":"trusted-query","detail":"authoritative absence confirmed"}}
try:
    book.resolve(**request)
except sqlite3.IntegrityError:
    pass
else:
    raise AssertionError("trigger did not abort")
assert book.inspect("op-fail")["status"] == "unknown"
assert book.db.execute("SELECT count(*) FROM resolutions").fetchone()[0] == 0
book.db.execute("DROP TRIGGER fail_resolution")
assert book.resolve(**request)["status"] == "confirmed_not_done"
book.close()
```

这个测试验证本地事务原子性，不验证外部查询真的权威。证据 detail 写“authoritative”并不会自动让它成为权威；真实系统需要记录查询接口、业务键、结果版本和核验身份。教材使用人工 fixture 是为了学习事务结构，不能把字符串当安全证明。

核验记录还要避免宽泛批量操作。一次命令把所有 running 改 done，会把不同调用的未知状态混在一起，没有逐项结果，也无法恢复模型观察。正确入口应绑定具体 operation_id 与预期版本，必要时批量请求也逐项返回成功、冲突或仍未知。整体报告可以汇总，但底层证据必须可定位到每个动作。

若确认结果来自人工，界面应展示实际动作、参数摘要、可能影响和现有证据，不能只问“是否继续”。用户批准一个具体恢复决定后，后端再次核对版本，避免批准期间状态变化。A11 的审批摘要绑定与这里的核验版本控制可以结合，但审批授权与事实确认仍有不同含义：允许执行不证明已经执行，确认已执行也不自动授权新的动作。

## 第十课：事件日志、快照与重放的不同选择

快照保存当前状态，读取快但可能难以解释每一步来源；事件日志保存变化事实，便于审计和重建，但需要稳定事件 schema 与确定性 reducer。课程 State.events 存在快照内部，主要用于诊断，并不是独立追加式事件存储。不能因为有 events 列表就声称实现了完整事件溯源系统。

若采用事件溯源，应定义哪些事件是事实、哪些只是请求。ToolRequested 表示意图，ToolCompleted 表示已确认结果，TaskCancelled 表示停止继续安排，BusinessReconciled 表示后来获得证据。reducer 只根据事件更新状态，不在重放时重新调用模型或工具，否则重建历史会再次产生效果。外部动作由独立执行器响应待执行意图，结果再写成事件。

事件必须有稳定序号或版本，重复事件应幂等或明确拒绝，缺口应停止重放。事件内容也需要 schema_version，旧事件不可随新代码假设解释。快照可以作为加速点，保存覆盖到哪个事件序号；加载后只重放后续事件。若快照与事件日志不一致，应有校验和重建路径，不能静默相信较新的时间戳。

事件溯源并非所有项目必选。小型单用户 Agent 使用快照加工具账本可能更容易维护，只要故障窗口明确。完整事件系统带来更多存储、迁移、隐私和重放复杂度。选择应基于审计、恢复和协作需求，而不是因为框架宣传某个名词。你应能说明当前快照方案有哪些无法回答的问题，再判断是否值得引入事件日志。

隐私删除在事件系统中尤其需要设计。不可变日志与用户数据删除要求可能冲突，可以采用受控引用、字段加密和独立敏感数据存储等方案，但具体政策取决于应用与适用要求。本教材不提供通用合规结论，只提醒你不要把所有原始提示、密钥和私人文档永久追加进公共日志。记录最小必要证据，是工程设计而不是事后清理。

## 第十一课：投递器的重试、死信与顺序

outbox 核心解决业务状态与待投递意图一致，但生产投递器还要处理临时网络错误、永久拒绝、重复确认和长时间未送达。每条事件可以保存尝试次数、下次尝试时间、最后错误类别和投递状态；重试仍遵守 A06 的退避、预算和取消规则。不能让一个永久无效事件无限占据队首，阻塞所有后续通知。

死信表示自动投递已停止，需要核验或修复，不是删除事件。应保留业务身份、载荷摘要、失败原因和处理历史，允许有权限的操作者修复后重投或明确放弃。重投应保留原事件身份，除非业务语义确实变为新事件；随意生成新编号可能绕过接收端去重，造成重复效果。

顺序也要按业务定义。创建后更新、更新后删除可能需要同一实体内有序，但不同实体事件可以并发。全局严格顺序会降低吞吐并让一条坏事件阻塞所有任务；完全无序又可能让删除先于创建到达。可以按实体键分区或让消费者用版本拒绝过期事件。课程按 event_id 排序只是确定性演示，不代表真实业务顺序。

接收端去重记录有保留期限问题。如果 inbox 很快删除旧编号，迟到重复消息可能再次被当作新事件；如果永久保留，存储会增长。需要结合发送端重试上限、消息系统保留和业务风险制定策略，并保留必要的长期业务唯一约束。去重表清理不是纯数据库优化，它会改变系统重复处理保证的时间范围。

消费者处理也必须与 inbox 记录在同一事务中，才能保证“标记已处理”与“业务效果”一致。课程消费者只把通知放进 inbox，因此一条 INSERT 就代表模拟效果。若真实消费者还要更新本地业务表，应同事务提交；若它又调用外部 API，就再次出现跨系统窗口，需要下一层幂等或 outbox。模式可以组合，但不会凭名字消除所有边界。

## 第十二课：数据库恢复演练、备份与兼容性

### 恢复演练要验证数据库，不只验证程序返回值

很多恢复测试写成“调用 resume 后没有异常”，这只能证明某条路径跑完，不能证明业务正确。完整验收至少检查快照版本、pending 集合、工具消息编号、ledger 行数与状态、业务对象数量和最终 verifier。某个运行最终回答四十二，但工具执行了两次收费操作，仍然是失败；某个运行没有重复效果，却永远丢失结果，也没有完成目标。

为每个故障窗口建立一张预期表。横轴是注入点，纵轴是各份存储的事实。测试先运行到注入点，再关闭对象，建立新对象加载，最后对照表检查。不要在恢复前手工把数据库改成你希望的状态，这会绕过真正的写入顺序。手工构造状态适合验证不变量拒绝，不适合证明正常流程能写出可恢复状态。

故障注入可以分三类。第一类是可捕获异常，运行时有机会保存 error；第二类是模拟进程死亡，跳过普通异常清理，留下最后持久快照；第三类是提交成功但确认丢失，调用方不知道结果却数据库已经改变。三类都可能表现为“调用没返回”，但恢复结论不同。测试名称应写清楚哪一类，避免一个笼统 crash_test 混淆。

对账后也要再次注入失败。例如已确认业务成功，正在把结果补录 ledger 时失败；ledger 补录成功，正在更新 messages 时失败；messages 更新成功，通知用户时失败。恢复过程不是魔法区，它同样需要幂等与提交边界。每一段都可以重试而不重复效果，才有持续恢复能力，而不是只能从第一次故障侥幸救回。

测试数据应保持小而具体。一个工单、一条事件、两个数据库连接就能展示跨系统窗口；不需要搭建复杂消息集群才理解 outbox。先用确定性小实验验证语义，再在真实数据库和消息服务上做受限集成，确认锁、隔离、超时和连接错误映射。复杂环境越早引入，越容易把协议设计错误误认为基础设施偶发问题。


### 备份、恢复与状态兼容性

数据库文件存在不等于有可靠备份。误删、磁盘损坏、错误迁移或应用 bug 都可能影响主数据。备份策略应说明备份哪些对象、何时生成、一致性边界、保留多久以及怎样验证能恢复。对 Agent 系统，可能需要同时保存 runs、ledger、业务幂等记录、outbox、inbox 和配置版本；只备份 messages 会丢失最重要的效果证据。

备份与运行数据库并发时要使用适合数据库的机制，不能假设随手复制某个文件就一定包含一致状态。具体日志模式和备份方法应按当前数据库官方文档核查。课程不在这里配置生产备份，但可以用关闭连接后的临时文件副本演示恢复，并明确那只覆盖受控静止状态。不要把教学复制步骤包装成在线生产方案。

恢复备份还涉及时间倒退。假设业务系统保留最新工单，本地 runs 恢复到昨天，旧 pending 可能重新出现。若业务幂等键仍稳定，下游可以返回既有结果；若幂等记录也被回滚或清理，就可能重复创建。因此灾难恢复应协调各系统的恢复点，并保留足够长的业务身份约束。单个数据库恢复成功，不代表跨系统事实自动一致。

配置兼容性在备份恢复时更突出。旧快照可能来自旧工具版本，新部署已经改变参数意义。应先用兼容读取器检查 schema 与指纹，再执行迁移或保持只读核验；不能让应用启动后立刻自动跑所有 pending。恢复模式可以先只提供 inspect，再逐项启用执行，这样操作者有机会确认权限、业务存储和外部依赖状态。

演练应验证真实读取与业务核对，而不只是看到备份文件大小大于零。可以恢复到临时隔离目录，读取一组已知 run，检查版本与账本关系，运行只读一致性报告，不连接真实副作用服务。通过后再按明确运维流程切换。课程中的临时数据库习惯正是在培养这种可重复演练能力。

## 第十三课：建立一个只读一致性报告器

下面的报告器检查课程 Store 的几项局部关系：done 账本必须有输出，running 不应带已完成输出；快照里的 tool 消息应能找到同 call_id 的 done 记录；同一个快照不应出现重复工具结果编号。它不尝试证明外部业务真的成功，也不自动修复异常，只生成可审查问题列表。

```python
def inspect_store_consistency(store):
    issues = []
    rows = store.db.execute("SELECT run_id,call_id,status,output FROM tool_ledger").fetchall()
    ledger_by_key = {}
    for run_id, call_id, status, output in rows:
        ledger_by_key[(run_id,call_id)] = (status,output)
        if status == "done" and output is None:
            issues.append({"run_id":run_id,"call_id":call_id,"code":"done_without_output"})
        if status == "running" and output is not None:
            issues.append({"run_id":run_id,"call_id":call_id,"code":"running_with_output"})
    for run_id, payload in store.db.execute("SELECT id,payload FROM runs"):
        state = json.loads(payload)
        seen = set()
        for message in state["messages"]:
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if call_id in seen:
                issues.append({"run_id":run_id,"call_id":call_id,"code":"duplicate_tool_message"})
            seen.add(call_id)
            row = ledger_by_key.get((run_id,call_id))
            if row is None or row[0] != "done":
                issues.append({"run_id":run_id,"call_id":call_id,"code":"message_without_done_ledger"})
            elif row[1] != message.get("content"):
                issues.append({"run_id":run_id,"call_id":call_id,"code":"message_output_mismatch"})
    return issues

h = Harness(ScriptedModel([Turn(calls=[ToolCall("c","add",'{"a":1,"b":2}')]),Turn("3")]), [arithmetic_tool()])
s = h.run(h.new("audit-good","add"))
assert inspect_store_consistency(h.store) == []
with h.store.db:
    h.store.db.execute("UPDATE tool_ledger SET output=? WHERE run_id=? AND call_id=?", ("changed","audit-good","c"))
assert inspect_store_consistency(h.store)[0]["code"] == "message_output_mismatch"
h.store.close()
```

报告器先读 ledger 建映射，随后逐个快照检查消息。这个实现适合小型课程数据库，会把账本全部放入内存；大规模系统需要分页、索引和一致性读取边界，避免扫描期间状态变化产生误报。只读报告也可能接触敏感结果，输出只保留编号与错误码，不把完整工具内容复制到日志。

done 账本但快照没有 tool 消息不一定是错误，可能正处于合法窗口三或输出预算停止。因此报告器没有把它直接列为损坏，而应在另一个“待回填或待核验”报告中结合 pending 和状态解释。检查不变量时要区分不可能状态与暂时不一致，过度严格会把正常恢复窗口当作事故。

相反，快照已经有 tool 消息但找不到 done 账本，在课程正常流程下不应发生，因为账本完成先于对话提交。这个关系由代码顺序建立，可以作为审计不变量。若未来改变实现顺序或允许导入外部历史，需要更新检查规则和数据来源标签，不能让旧检查器误判新合法流程。

自动修复需要更高标准。看到消息与账本不一致时，不能随便选择较新的字符串覆盖另一份，因为哪份代表真实业务结果并不由时间戳决定。应保留两份证据、停止相关运行并查询业务系统。报告器的职责是定位问题，不是凭猜测重写历史；这也是恢复工具应默认只读检查、明确提交动作的原因。

## 第十四课：幂等范围、并发条件与持久成本

### 幂等性还有范围、期限与并发条件

说一个接口幂等，必须说明相同的是什么、效果范围是什么、保证多久、并发调用怎样处理。相同 URL 不一定同意图，相同参数不一定同业务动作；例如用户可能合法地购买两件完全相同商品，不能仅按参数哈希永久合并。稳定业务键由调用者明确表示“这是同一项意图”，服务端据此去重，才不会误伤合法重复业务。

幂等结果也可能包含易变字段。第一次返回 created=True，第二次返回 created=False，二者文本不完全相同，但业务对象相同，核心效果不重复。若需要完全重放同一响应，应保存原始响应；若只保证业务效果幂等，可以返回当前对象与重复标记。接口合同应明确是哪一种，调用方不能把响应字节不同自动视为重复效果。

幂等记录的有效期影响恢复窗口。若服务只保留键一段时间，超过期限再次请求可能创建新对象。客户端必须知道期限，不能在长期离线后盲重放旧请求。业务唯一约束有时可以提供更长保障，但也要考虑对象删除与键复用政策。课程 TicketBook 不清理键，因此只在本地表生命周期内展示长期去重，不代表所有外部 API 都如此。

并发重复请求也要测试。两个相同键同时到达，一个创建，另一个应等待或读取同结果，而不是一个成功、另一个无法解释的错误。唯一约束是最后防线，应用还要把冲突映射为正确重复响应或明确意图冲突。若操作还包含多个表修改，幂等记录与业务写入必须处于同一事务，避免先占键后失败导致永远无法重试，或先写业务后占键导致重复。

有些操作天然具有幂等效果，例如把某状态设置为同一值；有些操作不是，例如余额减一。即便设置操作也可能触发通知或审计，因此“最终字段值相同”不一定意味着所有效果幂等。工具作者应列出需要保护的业务效果，而不是仅根据 HTTP 方法或函数名字判断。A06 的 side_effect 声明与这里的幂等合同相互补充。


### 持久化成本与状态膨胀

每次保存完整 messages 和 events 很直观，但随着任务增长，快照越来越大，每次写入重复序列化旧内容。课程选择这种方式是为了可读性，不代表高吞吐系统应无限复制完整历史。可以把大型 artifact、消息、事件和运行索引分表或分存储，快照只保存引用和当前控制状态；但引用对象也需要版本、完整性和权限检查。

拆分存储会增加一致性关系。快照引用一个 artifact，artifact 尚未提交或已被清理，恢复就失去材料。可以先写不可变 artifact，再在事务中提交引用，未引用对象由后台安全回收；或使用存储系统提供的明确提交机制。垃圾回收必须知道哪些运行仍引用对象，不能只按文件年龄删除，否则长任务恢复可能缺失证据。

事件与工具输出也要控制大小。巨大的原始日志可以保存为受控 artifact，ledger 保存摘要、大小、内容哈希与引用；模型只接收必要片段。哈希帮助检查内容是否变化，引用帮助定位，权限控制决定谁能读取。它们各有职责，不能因为有哈希就把 artifact 公开。

压缩存储与上下文压缩不同。数据库压缩可以无损减小字节，不改变事实；模型摘要通常有损，只保留部分语义。持久层不能为了省空间把唯一原始证据替换成未经验证摘要，除非保留政策明确允许且业务接受损失。A07 的摘要是派生视图，本章强调恢复依据需要足够精确。

性能优化之前先测量。记录快照大小、序列化耗时、事务耗时、账本增长和恢复时间，使用人工数据避免泄漏。不要因为一次小实验很快就宣称支持百万任务，也不要因为看到 JSON 快照就认定一定不可用。工程判断要基于工作负载与保证范围，课程代码提供可理解基线，后续优化保持相同不变量。

## 第十五课：工单通知全流程与恢复手册

### 把整个工单通知流程按提交点手算一遍

设用户确认创建一张工单，业务键为 req-7，模型调用编号为 call-3。宿主先保存 pending，工具账本写 running；业务事务创建工单并写 outbox；工具得到结果后写 ledger done；任务快照加入 tool 消息；投递器把事件写入消费者 inbox；最后生产者标记 sent。这里至少有四个独立提交领域：任务存储、业务生产者、消费者和最终用户展示。

| 提交之后立刻崩溃 | 恢复时看到 | 允许做什么 | 不能直接做什么 |
|---|---|---|---|
| pending 快照 | 有调用，无账本 | 验证后开始调用 | 假称已经完成 |
| ledger running | 调用可能开始 | 查询业务键 | 换新业务键重建 |
| 工单与 outbox | 业务已存在，通知待发 | 同键复用并恢复投递 | 再建同意图对象 |
| ledger done | 已有结果，历史可能旧 | 回填工具结果 | 再次调用业务函数 |
| tool 消息快照 | 模型观察完整 | 继续下一轮推理 | 重复追加同结果 |
| inbox 提交 | 接收端已处理，sent 可能零 | 同事件重投并去重 | 生成新事件编号 |
| sent 标记 | 投递已确认 | 返回已知状态 | 因界面断线重跑全部任务 |

这张表可以作为你审读真实 Agent 框架的模板。找到每个外部调用前后的保存位置，问清楚哪一步可能重复、哪一步可能丢失、哪一步有业务唯一约束。框架提供 checkpoint 只覆盖其中一部分；工具服务是否幂等、通知是否可去重、用户界面是否重复提交，都需要一起分析。

如果业务工单和 Agent ledger 恰好使用同一个数据库，是否可以把它们放进同一事务缩小窗口？在受控架构中可能可以，但工具抽象通常也要支持外部服务，因此不能把本地特例当通用保证。即便合并本地事务，通知和用户展示仍跨边界。最好的设计是明确哪些操作共享事务，哪些依赖幂等协议，而不是笼统说整个 Agent 是事务性的。


### 最终验收与恢复手册

本章最终项目应交付一份可执行实验和一份恢复手册。实验使用临时文件数据库，运行正常路径，再逐个注入模型请求中断、业务提交后结果丢失、账本完成后快照失败、消费者提交后发送端失败。每次建立新对象恢复，检查业务对象和通知都没有重复，未知状态没有被猜成成功。手册则按实际数据库字段说明怎样 inspect、怎样获取证据、怎样提交核验和怎样验证恢复结果。

恢复手册不应要求操作者直接执行一条无条件 UPDATE 把状态改为 ready。它应说明具体 run、call、业务键和预期版本，列出确认完成与确认未发生所需证据，以及无法确认时怎样保持暂停。若需要补录结果，先验证结果结构和对应意图，再通过受控接口写审计。若需要补偿，作为新的业务动作单独授权、记账和验证。

验收还有一个容易遗漏的条件：重复执行恢复命令。网络超时可能让操作者不知道第一次核验是否提交，于是再次点击。同 resolution_id 同内容应返回原结果，不重复改变版本或效果；不同内容复用同编号应冲突。恢复入口自身具备幂等性，系统才不会在故障处理过程中制造第二层故障。

完成后，你应能不用运行代码就手算每个窗口的数据库状态，也能用测试证明手算正确。你应知道快照、事件、调用账本、业务键、outbox、inbox、租约和配置指纹分别解决哪一个问题，不能用其中一个名词替代其他机制。下一章会把这些能力放入显式工作流与审批中断，届时“从节点继续”必须遵守本章的效果与证据边界。

还要明确本章参考代码的集成范围。ReconciliationBook 是独立教学项目，展示核验事务与审计身份，并未自动接管现有 Harness 的 uncertain 终态；inspect_store_consistency 是只读检查器，不会修复数据。把它们接入产品时，需要一个单拥有者协调步骤，验证核验结果对应当前 pending 与 ledger 签名，再以预期版本保存后续状态。不能只导入这个类就宣称现有运行时已有完整运维恢复界面。

同样，Outbox 的消费者目前用 inbox 行表示模拟通知效果，没有真的发送邮件、Slack 或其他消息。本章所有投递实验都是本地数据库操作，零外部发送。若以后换成真实通知服务，应重新分析该服务的幂等支持、查询能力和未知结果窗口；课程中的一行 inbox 不能自动代表外部消息系统已提供相同保证。这种明确映射能力，是从教学实现走向真实框架时最重要的审查习惯。

最后一次恢复检查应保留原始证据，记录新增确认，而不是覆盖历史错误。原先为何进入不确定、后来依据何种查询确认完成，两件事都是真实发生的事实。把后者写入新事件，才能让未来审查者理解完整因果，而非只看到一个被人工改成成功的字段。
