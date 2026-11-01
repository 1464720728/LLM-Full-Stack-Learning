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
python -m projects.evidence_workshop all
python -m unittest tests.test_projects.EvidenceProjectTest -v
python -m unittest tests.test_projects_workshop -v
```

这个项目实际连接 A 卷 Harness、SQLite 状态账本和 R 卷知识库。模型是明确的确定性测试替身，检索和状态管理是真实代码；上述命令不调用外部 API。长教程从一份文档与一次搜索开始，逐步解释权限过滤、分块身份、BM25／RRF、工具参数验证、每一步状态写入和引用复验。

`evidence_workshop` 把权限范围、检索排名、状态轨迹、文档删除／更新／清理后重建、身份伪造、引用伪造和进程重开分别做成可以观察的场景。先选择 `trace` 跟踪正常请求，再运行 `delete` 观察失效历史在下一次模型调用前被拒绝；`auth-boundary` 证明旧任务不能改绑身份、同一主体撤权后也不会继续发送旧证据。完整教程另演示模型完成后、交付前的复验，并提供真实模型接入代码与显式启动条件。

## 跨卷项目 P2：模型、调度与在线注意力

阅读 [逐步项目教程](P2_MODEL_RUNTIME.md)，运行：

```bash
python -m projects.model_runtime
python -m projects.incremental_runtime
python -m projects.incremental_runtime --online
python -m unittest tests.test_projects.ModelRuntimeProjectTest -v
python -m unittest tests.test_projects_incremental -v
```

教程从两维手算模型开始，带你逐步接通 M 卷模型、S 卷请求调度和 K 卷在线注意力。保留两个可比较的运行路径：`model_runtime` 使用完整上下文重算与页元数据调度器；`incremental_runtime` 为每个请求保存真实 NumPy K/V 数组，交错执行增量解码，并在完成或取消时释放缓存。

`--online` 将逐块在线注意力实际替换进完整 decoder 的每一层，比较完整前向、缓存前向和调度输出。教程还跟踪数值分页缓存、共享页和写时复制，但没有将连续 NumPy 缓存冒充生产 GPU 页池。调度器报告的合成时间与实际数组字节量分别标注；GPU 性能实验需要按教程在真实硬件上另行执行。

## 综合验收如何交付

1. 保留初版参考运行记录，写下自己要改变的行为。
2. 写一个失败测试证明问题存在。
3. 修改实现，使新测试通过且既有测试不回归。
4. 写清楚真实执行路径与教学简化。
5. 给一个陌生输入，现场预测与诊断，而不是背示例输出。

拓展任务应当进一步接入真实 API、真实语义 embedding、真实框架和真实 GPU，但都从上述稳定小基线出发。完整步骤、完成标准和参考答案在对应项目教程与分卷中，第一版不把这些任务留成一句“以后再学”。
