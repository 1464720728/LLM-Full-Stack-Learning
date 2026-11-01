# 源码工作簿二：八个崩溃现场，重建一条可信执行轨迹

本工作簿对应 A04、A06、A10、A11。你将故意在不同时间点打断程序，然后根据数据库内容判断下一步。目标不是背“加 checkpoint 就能恢复”，而是对每个窗口明确：效果是否发生、结果是否保存、历史是否更新、是否允许重试。

全部实验在临时目录或内存 SQLite 中进行，不需要任何线上账号。请保留每次故障前后的 State、runs 表和 tool_ledger 表。不要真实关闭公司服务器或终止别人的进程来模拟崩溃；在代码中注入异常就足以验证这些逻辑。

## 现场一：模型请求还没发出

调用 h.new 后，state.status=ready、model_calls=0、inflight_model=false。此时关闭 Store 并重新打开，恢复得到相同状态。下一次 step 可以正常发出第一条请求，因为系统尚未预留或尝试该请求。

运行 `16-checkpoint` 会演示另一位置的恢复；先自己把它改成 new 后就退出。对照数据库：runs 有一行，tool_ledger 没有行。用 SQL 查询：

```python
print(store.db.execute("SELECT id,version FROM runs").fetchall())
print(store.db.execute("SELECT run_id,call_id,status FROM tool_ledger").fetchall())
```

不要把“模型请求还没发出”仅凭没有 response 日志判断。你现在知道这一事实，是因为测试注入点在调用之前，且 state 明确没有 inflight 标记。在真实系统中，缺日志可能只是日志没落盘，不能作为效果不存在的普遍证据。

## 现场二：请求已预留，响应未保存

构造一个模型，在 complete 中直接抛 KeyboardInterrupt：

```python
class CrashingModel:
    def complete(self, messages, tools, index):
        raise KeyboardInterrupt("simulated process interruption")
```

普通 Exception 捕获不会吞掉这个中断，因此调用前保存的状态留在数据库：model_calls=1、inflight_model=true、messages 仍只有用户任务。恢复后 step 会进入 uncertain，不会请求下一条脚本响应。

这里“服务端是否真的执行”在模拟中可以由你知道，但恢复算法不能依赖测试作者的脑内知识。它只看到持久状态。真实 HTTP 请求可能已发送并计费，也可能还没离开本机。因此最保守语义是未知。若提供方支持按请求身份检索结果，你可以查询并补录；否则可由操作者决定重新尝试并接受可能重复费用。不能悄悄把调用计数减回0。

参考测试 `test_model_crash_window_does_not_skip_response` 用同一个 Store 重建 Harness，证明恢复不跳过未保存响应。这是一个独立的重要测试：只有“工具 pending 恢复成功”并不能覆盖模型请求窗口。

## 现场三：模型已经提出工具，但工具尚未开始

第一次正常 step 返回后，messages 中有 assistant tool_calls，pending 中有 c1，ledger 仍为空。此时重启进程，恢复即可执行 c1。`16-checkpoint` 就覆盖这个位置。ScriptedModel 的调用索引已经是1，恢复先走 pending 分支，不会立刻调用模型 index=1；工具完成后才请求最终响应。

手工检查这个条件顺序。如果你把模型分支放在 pending 分支前，重启后模型会在没有工具观察的情况下继续生成，可能直接宣称完成。完整消息历史和 pending 队列必须一起恢复，不能只保存其中一个。

## 现场四：ledger 已声明 running，纯工具尚未记录结果

`Store.start` 先插入 running 记录，再执行工具。对于 add 这种纯计算，如果程序中断，重放计算没有外部副作用。恢复可以重新算一遍，随后把结果写 done。你应验证两次执行数值相同，但不要把这件事推广到所有“看起来只是查询”的工具。

查询也可能有计费、审计或访问计数，只是通常业务可接受重复。pure 的定义要针对你需要保障的效果：如果查询会触发昂贵 GPU 推理或消耗有限额度，重复虽不修改业务对象，仍有资源后果。工具元数据可以进一步区分 readonly 与 replay_safe，而不是一个布尔值承担所有语义。本课程为简化把纯工具视为可安全重放，实际扩展要写清前提。

## 现场五：副作用已发生，结果未知

创建一个工具先向列表 append，再抛 RuntimeError。这个列表只是模拟外部系统，能够让你明确观察效果发生。运行后状态应 uncertain，effects 中一项，ledger 仍为 running。

```python
effects = []
def effect_then_fail(arguments):
    effects.append({"ticket": "created"})
    raise RuntimeError("response lost after effect")
```

如果你的执行器捕获后简单返回 `ok=false` 并重试，effects 会变两项。这个反例证明“工具报错”并不等于“工具没做事”。正确恢复需要查询业务身份、判断是否已完成，或按用户允许策略处理。日志可以告诉你发生了异常，但只有目标系统状态或幂等协议才能解决效果不确定。

在真实生产系统里，尽量让工具调用具有稳定 request_id，并提供 `get_operation(request_id)`。创建请求和查询请求是两种工具能力，后者可以帮助恢复，而不是让模型反复猜测“再试一次应该没问题”。

## 现场六：结果 ledger 已保存，消息还未保存

先直接调用 `_execute` 得到结果，但故意不追加 tool 消息也不 pop pending。关闭进程后恢复，从 ledger 查到 done，返回缓存输出，不重复业务函数。随后正常更新 messages 与状态。这是本地 ledger 最有用的窗口。

参考测试 `test_ledger_replays_completed_without_effect` 用计数器证明函数只运行一次。只检查最终文件内容不够，因为同样内容写两次看起来仍一样。验证去重必须观察执行次数或独立副作用记录。

进一步思考：如果结果文件不在数据库而是外部 artifact，ledger 只保存引用，那么引用目标必须持久且可访问。不能在 ledger 中记录一个马上被清理的临时文件路径，再声称恢复完整。artifact 生命周期应至少覆盖任务恢复所需时间，且权限保持正确。

## 现场七：两个 worker 同时恢复

两个 worker 从同一个 version 读取 State，各自都认为有 pending。即使最终 Store.save 的版本检查只允许一个成功，也不能保证工具函数只执行一次。因为副作用可能发生在 save 之前。数据库 CAS 保护状态，业务幂等保护效果，任务租约保护执行所有权，它们解决不同问题。

课程明确假设一个 run 同时只有一个执行所有者。你可以实现 owner_id 与租约作为扩展，但要设计租约过期时旧 worker 仍活着的情况。仅有 `lease_until` 时间戳不够：旧 worker 可能在暂停很久后恢复，继续执行动作。常见进一步机制是 fencing token，让下游拒绝过期持有者的操作；下游是否支持必须确认。

不要为了“跑通多进程”去掉 expected_version 条件。冲突是保护在工作，不是数据库妨碍你。先查清两个 worker 为什么都拿到了同一任务，再决定单所有者、分区或幂等策略。

## 现场八：图暂停后进程重启

参考 `advanced_solutions.GraphSnapshots` 保存 GraphState、schema_version 和乐观版本。暂停状态 waiting 中必须保留 proposed_action、review_digest、当前 node 和历史。重启后加载，用户批准匹配 digest，再从同一节点重新运行。

完整参考模式：

```python
from labs.agents.advanced_solutions import GraphSnapshots
from labs.agents.workflows import Workflow, GraphState, approval_node, action_digest

# 正式学习时把 :memory: 换成自己的临时目录中的 graph.db。
store = GraphSnapshots()
graph = Workflow({"review": approval_node}, {"review": "END"})
action = {"name": "create_ticket", "title": "Example"}
state = graph.run(GraphState("review", {"proposed_action": action}))
version = store.save("g1", state, 0)
restored, version = store.load("g1")
graph.approve(restored, action_digest(action), True)
graph.run(restored)
store.save("g1", restored, version)
store.close()
```

内存版本演示保存与加载接口，真正跨进程测试使用文件路径，已由 `test_persistent_graph_resume` 覆盖。若只在同一内存数据库中存取一次，就声称已验证断电恢复，是不准确的。课程测试验证的是进程对象重建后的逻辑恢复，不宣称模拟了硬件断电、磁盘损坏和所有文件系统持久性条件。

## 综合实验：Outbox 与幂等收件箱

`advanced_solutions.Outbox` 使用两个独立 SQLite 连接，代表生产者与消费者两个事务域。生产者在同一事务里写 tasks 和 outbox；投递 worker 读未发送事件，消费者用 event_id 唯一键写 inbox；最后生产者把 sent 改为1。

故意在消费者提交后、生产者标记前抛错。恢复后 outbox 仍显示未发送，worker 会再次投递。消费者发现相同 event_id 已存在，不新增，因此 inbox 仍一行。最后生产者标记完成。这是至少一次投递加幂等消费，而不是网络只传了一次。

运行：

```bash
python -m labs.agents.advanced_solutions
```

应输出 outbox_inbox_rows=1。接着自己查询两边状态：首次崩溃后 inbox=1、sent=0；恢复后 inbox=1、sent=1。看到重复传输与唯一业务效果同时存在，你才真正理解“幂等”解决了哪一层。

如果消费者收到相同 event_id 却不同 payload，参考实现拒绝身份冲突。否则攻击或程序 bug 可以把不同事件伪装成重复而被静默丢弃。事件 ID 的稳定性必须与业务语义一致，不能每次重试随机生成新 UUID，否则消费者无法识别重复。

再加入第二条连接，才能检验“拒绝冲突”是否在并发下成立。错误顺序是 A 查到无记录，B 插入不同内容并提交，A 的 INSERT OR IGNORE 忽略冲突，然后 A 报告成功。仅用 `with connection` 包住 SELECT 和 INSERT 不能阻止这一交错，因为默认 SQLite 的只读 SELECT 尚未开启所需的写事务。当前解法把第一次 INSERT 放在查询之前：它获得写锁，随后 SELECT 读到真实赢家内容，再决定是否接受；若内容不同就抛错，由事务上下文回滚本次写入。

阅读 `tests/test_agents_advanced.py` 的 BeforeInsertConnection：它不是生产数据库封装，而是精确的故障注入器，在首个 INSERT 之前提交另一连接的写入。`test_outbox_concurrent_business_key_conflict` 验证不同标题不会被当成同一请求成功；`test_outbox_concurrent_inbox_collision_never_marks_sent` 验证消费者存在不同内容时，生产者 sent 仍为0；`test_outbox_existing_event_collision_rolls_back_new_task` 验证同一事务里已经插入的新 task 也会回滚。请手写三条时间线，并分别指出唯一约束、写锁、内容比较、事务回滚各自保护什么。去掉其中任一层，至少有一个测试应失败。

## 交付作业与详细评分

提交八个现场的故障表，每行至少有：注入点、已知效果、持久 state、ledger、允许恢复动作、禁止动作和对应测试。再完成一个新增工具：模拟向外部目录写一份报告，业务键 report_id 固定。要求在重复调用下只出现一份业务记录，参数冲突被拒绝，写入后响应丢失可以通过 report_id 查询确认。

参考实现步骤是：先把报告内容 hash 与 report_id 写进一个持久业务表；对于相同 report_id 不同 hash 拒绝；内容文件采用明确目标与原子可见写入；查询工具返回当前记录与文件 hash；恢复命令比较业务状态并补录 ledger。你还要考虑数据库记录与文件替换之间的窗口，不能声称两个存储天然事务一致。最简单教学版本可以把短报告文本直接放同一个 SQLite 表中，让业务状态与内容处在一个事务域，再把导出文件当可重建视图。

评分重点不是写了多少行恢复代码，而是有没有把无法保证的地方说清楚：模型请求计费不确定、跨服务副作用不确定、多 worker 争抢、artifact 生命周期、数据 schema 迁移。能够识别未知并安全停止，是可靠系统能力的一部分，不是“Agent 不够自主”。
