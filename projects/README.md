# 综合项目工作台

综合项目把一个机制放进完整的输入、状态、错误和交付链。以下项目都有参考实现；学习时先阅读接口和验收，用自己的分支完成，再对照实现。各卷正文中的项目还包含更细的分阶段任务。

## 分卷项目

| 项目 | 代码入口 | 主要练习 |
|---|---|---|
| 日志分析 CLI | [log_project.py](../labs/foundations/log_project.py) | 文件、解析、错误行、统计、命令行与测试 |
| 小型任务服务 | [task_service.py](../labs/foundations/task_service.py) | HTTP、SQLite、状态与持久化 |
| 可靠 Agent 与代码修复助手 | [Agent 综合实现](../labs/agents/capstones.py) | 工具执行、恢复、测试、受限改动 |
| 本地证据助手 | [assistant.py](../labs/knowledge/assistant.py) | 摄入、检索、权限、引用与拒答 |
| 长期记忆助手 | [memory_assistant.py](../labs/knowledge/memory_assistant.py) | 记忆类型、修正、版本、期限与遗忘 |
| 微型语言模型 | [tiny_lm.py](../labs/models/tiny_lm.py) | 训练损失、梯度、生成与可复现性 |
| 模型机制比较 | [compare.py](../labs/models/compare.py) | 缓存、稀疏、线性注意力、SSM 的对照 |
| 连续批处理与策略实验 | [Engine](../labs/serving/engine.py)、[实验入口](../labs/serving/experiments.py) | 页所有权、前缀、抢占、取消、可观测事件 |
| 算子正确性与性能方法 | [capstone.py](../labs/kernels/capstone.py) | CPU 参考、真实 GPU 路径、形状／误差／计时 |

每个程序的具体参数以对应章节和 `--help` 为准。不要因为看到一个文件就猜测它会启动服务或调用 API；先阅读入口说明。

## 跨卷项目 P1：可恢复的证据 Agent

阅读 [逐步项目教程](P1_EVIDENCE_AGENT.md)，运行：

```bash
python -m projects.evidence_agent
python -m unittest tests.test_projects.EvidenceProjectTest -v
```

这个项目实际连接 A 卷 Harness、SQLite 状态账本和 R 卷知识库。模型是明确的确定性测试替身，检索和状态管理是真实代码；不调用外部 API。项目把检索前权限过滤、工具参数验证、执行持久化、引用复验组合起来。

## 跨卷项目 P2：模型、调度与在线注意力

阅读 [逐步项目教程](P2_MODEL_RUNTIME.md)，运行：

```bash
python -m projects.model_runtime
python -m unittest tests.test_projects.ModelRuntimeProjectTest -v
```

这个项目把 M 卷完整小型 decoder 的真实 NumPy 前向接入 S 卷调度接口，同时用 K 卷独立实现验证在线 attention。页池仍存 token 元数据，模型重算数值上下文；时间是调度模拟值。项目让你看清“模型计算”“调度策略”“内存表示”可以分别验证，不能因为输出正确就宣称已经实现生产级分页 KV。

## 综合验收如何交付

1. 保留初版参考运行记录，写下自己要改变的行为。
2. 写一个失败测试证明问题存在。
3. 修改实现，使新测试通过且既有测试不回归。
4. 写清楚真实执行路径与教学简化。
5. 给一个陌生输入，现场预测与诊断，而不是背示例输出。

拓展任务应当进一步接入真实 API、真实语义 embedding、真实框架和真实 GPU，但都从上述稳定小基线出发。完整步骤、完成标准和参考答案在对应项目教程与分卷中，第一版不把这些任务留成一句“以后再学”。
