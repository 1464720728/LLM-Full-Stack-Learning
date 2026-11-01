# 卷 A：从零构建可解释、可恢复、可验证的 Agent

本卷的目标不是把某个 SDK 的方法名背熟，而是建立一套判断能力：看到任意 Agent 工程，你能指出状态在哪里、模型输入怎样形成、一次工具动作怎样得到授权、失败在哪个边界可以重试、恢复时会不会重复产生副作用，以及任务真正成功的证据是什么。然后你能独立实现这些机制，并在真实开源框架里找到对应位置。

先完成[基础卷](../00-foundations/README.md)中的 Python 函数、字典、类、异常、文件、模块、JSON、命令行、测试与进程课程。初读不需要 GPU，不需要模型 API，也不需要学完 Transformer。你应能解释 `return` 与 `print` 的区别、能阅读一个包含两个类的 Python 文件、知道进程退出后普通变量消失。还不熟时，请先完成 A01，再回基础卷补足卡住的具体语法。

## 学习方法

每章按“预测 → 运行 → 解释 → 修改 → 故障注入 → 独立复写”完成。只看参考答案或让另一个模型替你写完，不足以通过验收。保存自己的实验笔记，至少记录输入、预测、实际结果、产生差异的代码位置、修复与回归测试。代码已经提供参考实现；练习要求你修改或从空文件重建一个组件，并证明它的边界。

本卷共有 15 章正文、3 本源码工作簿，27 个编号离线实验、独立评测与进阶解法程序，2 个完整的集成项目；每章另有递进练习和参考解题过程。离线脚本模型是可控测试替身，验证 Harness 的行为，**不代表真实语言模型的能力测量**。所有命令从仓库根目录执行：

```bash
python -m labs.agents.experiments --list
python -m labs.agents.experiments 01-loop
python -m unittest tests.test_agents_core -v
```

如你的 Linux 环境只提供 `python3`，将上述 `python` 换成 `python3`。Windows PowerShell 可以原生运行这些标准库实验；推荐也在 WSL/Linux 做一遍文件与进程相关练习，比较环境差别。无需安装外部 Python 包。真实 API 入口是 `python -m labs.agents.live --help`，只有显式添加 `--allow-live` 才能执行网络请求。

| 章 | 核心能力 | 主实验 |
|---|---|---|
| [A01 最小循环](01-loop.md) | 手推模型—动作—观察状态转移 | 01-loop |
| [A02 模型协议](02-protocols.md) | 解析并保留协议语义，写适配器 | 03-chat、04-responses、05-messages |
| [A03 工具契约](03-tools.md) | 从参数验证到业务不变量 | 02-contract、08-errors |
| [A04 状态与事件](04-state-events.md) | 设计可观察且可测试的状态机 | 16-checkpoint、17-conflict |
| [A05 流式](05-streaming.md) | 从字节流还原并发工具调用 | 06-sse、07-deltas |
| [A06 容错和预算](06-reliability.md) | 区分重试、取消、未知结果 | 09-budget、10-retry、19-cancellation |
| [A07 上下文工程](07-context.md) | 保持工具配对、证据与预算 | 11-context、12-provenance |
| [A08 工具执行环境](08-workspaces.md) | 文件修改、进程与真实权限边界 | 13-paths、14-edit |
| [A09 并发](09-concurrency.md) | 控制扇出、回收任务、避免竞态 | 20-fanout |
| [A10 持久化](10-durability.md) | SQLite 检查点、幂等与恢复 | 15-idempotency、18-uncertain |
| [A11 状态图](11-workflows.md) | 条件路由、审批暂停与恢复 | 21-approval、25-ticket-project |
| [A12 多 Agent](12-multi-agent.md) | 任务分解、证据合并和争议处理 | 22-multi-agent |
| [A13 评测与观测](13-evaluation.md) | 构建任务集、分层归因、回归 | 23-metrics、24-redaction |
| [A14 编码助手](14-coding-agent.md) | 独立运行一个修复闭环 | 26-coding-project |
| [A15 MCP、A2A 与框架](15-integrations.md) | 将组件接到真实协议和开源工程 | 27-tool-rpc |

## 参考代码的地图

`contracts.py` 定义模型、工具和数据验证；`runtime.py` 是主循环；`store.py` 提供持久化；`protocols.py` 处理三种协议的必要差别；`streaming.py` 拆分网络分帧与工具参数组装；`context.py` 管理完整上下文片段；`tools.py` 演示文件工具和业务幂等；`workflows.py` 实现状态图与异步扇出；`reliability.py` 提供策略函数；`capstones.py` 把组件接起来；`live.py` 是显式启用的真实 Chat Completions 客户端。每个文件都能独立阅读，没有隐含框架魔法。

## 完成标准

基础完成：不看答案手写一个有工具验证、步数预算、错误回传的循环。工程完成：进程重启后能从工具待执行状态恢复，解释外部副作用无法天然 exactly-once 的原因，并有故障测试。进阶完成：用同一个任务集比较自己的 Harness 和一个真实开源项目，报告协议差异、成功率、成本与失败类型。深入研究入口：将[检索、记忆与知识图谱卷](../02-retrieval-memory-graphs/README.md)接入本卷工具注册表，再将模型客户端接到[推理系统卷](../04-inference-systems/README.md)的本地服务。

本卷的代码是完整教学实现，刻意保持规模可读。它没有生产级多租户鉴权、恶意代码隔离、分布式领导者选举、完整 JSON Schema、全量模型多模态协议，也不承诺跨协议无损转换。相应章节会教你定位这些缺口，给出实操任务，而不会把教学版本包装成通用生产框架。

## 实操与深入源码工作簿（必做）

- [27张实操卡与组合挑战](LAB-CARDS.md)：逐题预测、修改、故障与参考路线。
- [工作簿一：从四个函数到可恢复 Harness](WORKBOOK-1-build-and-trace.md)：展开每个对象、消息与检查点变化，逐段读主循环。
- [工作簿二：八个崩溃现场](WORKBOOK-2-crash-lab.md)：模型未知窗口、ledger、幂等、恢复、Outbox 与持久状态图。
- [工作簿三：真实模型、工具与框架迁移](WORKBOOK-3-production-bridge.md)：协议夹具、固定任务、全局预算、资源锁、流去重与真实工程验收。
- [一手来源与验证边界](SOURCES.md)。
- [跨卷综合项目](../../projects/README.md)：把本卷 Harness 接入知识、权限与证据验证。

进阶解法不是空白方向提示：`labs/agents/advanced_solutions.py` 提供状态验证、输出契约、固定任务上下文、序列去重、原子预算、资源键调度、持久图与 Outbox 的完整参考实现；对应 `tests/test_agents_advanced.py` 验证故障与边界。
