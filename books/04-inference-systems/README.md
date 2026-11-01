# 推理系统工程：从请求到框架源码与真实KV

本卷包含19章完整主课与工作坊、24个独立离线主题实验、逐章可执行例子与完整答案、轨迹回放、本地HTTP服务、真实NumPy分页K/V与online attention，以及固定vLLM源码的回归测试练习。正文按连续小课展开，每章不是一天的阅读量；应把手算、代码、故障和工程阶段分开完成。它面向已经学过基础Python/数组、希望从“会调参数”进入“能改核心组件”的学习者。

先完成[基础卷](../00-foundations/README.md)与[模型卷](../03-model-architectures/README.md)中张量、attention、自回归与KV内容。学习时每章都要运行实验、手算状态、做修改题和故障题；不要仅浏览标题。

| 顺序 | 章节 | 核心产物 |
|---|---|---|
| S01 | [请求生命周期](01-request-lifecycle.md) | 请求状态与首token时间表 |
| S02 | [调度与连续批处理](02-scheduling.md) | token预算、chunk与策略实验 |
| S03 | [分页KV](03-paged-kv.md) | 页分配/COW/引用计数 |
| S04 | [前缀缓存](04-prefix-cache.md) | 共同前缀、命名空间、淘汰 |
| S05 | [抢占与取消](05-preemption-cancellation.md) | 重算不重复、取消幂等 |
| S06 | [Runner与后端](06-runner-backends.md) | 可变长张量与接口追踪 |
| S07 | [采样与流式输出](07-sampling-streaming.md) | 独立RNG、SSE任意边界 |
| S08 | [测量与排队](08-measurement.md) | 明确计时定义的报告 |
| S09 | [vLLM/SGLang源码](09-source-reading.md) | 固定SHA调用路径 |
| S10 | [分布式并行](10-parallelism.md) | 矩阵分片、通信与所有权 |
| S11 | [PD分离](11-pd-disaggregation.md) | 状态传输、背压与失败协议 |
| S12 | [量化与确定性](12-numerics-determinism.md) | 分层精度与首处分歧定位 |
| S13 | [容量估算](13-capacity.md) | 每rank内存账与校准 |
| S14 | [诊断](14-diagnostics.md) | 最小复现与回归报告 |
| S15 | [共享H100实验](15-hardware-labs.md) | 有界单/多卡真实验证 |
| S16 | [综合工程](16-capstones.md) | 回放、服务、补丁、硬件报告 |
| S17 | [源码工作坊](17-source-workbook.md) | 真实Scheduler测试与负向控制 |
| S18 | [状态工作坊](18-state-workbook.md) | prefix/COW/preempt逐变量推演 |
| S19 | [真实数值分页KV](19-numeric-paged-kv.md) | 增量K/V与完整前向等价 |

## 怎样分段走完本卷

第一段 S01–S05 先建立请求与资源模型。每章先不看实现，手算两到三个请求的时间线或页表，再运行代码核对。此段出口是能解释“已知token、已计算token、已输出token”为什么不同，以及最后一个引用何时才允许释放页。S18 的状态工作坊可以在这一段之后插入，帮助把多个机制组合起来。

第二段 S06–S08 将状态接到数值与协议。完成 packed attention 与独跑对照、请求级随机流、任意字节切分、完整计时边界。S19 可在学过模型卷 attention 后插入，亲手验证浮点KV、分页gather、online归一化与COW；它与Engine的token元数据池有不同证据范围。

第三段 S09–S12 进入框架代码与高级执行。先读固定EngineCore/Scheduler/Runner调用链，再用S17逐轮源码工作坊检查字段与预算；随后学习并行、PD、量化和确定性。每次源码阅读都应产出输入输出、状态写入点和最小断言，不用“看过文件”替代理解。

第四段 S13–S16 完成部署工程。先算逐rank容量，再建立诊断证据，最后进入Win11/WSL、单卡与必要多卡实验。CPU阶段可以每天推进；GPU窗口前准备好代码、模型、输入和停止条件。S16把各章产物汇成可以审查的项目，而不是只保留一堆截图。

做题时先隐藏答案，写预测、执行、比较、解释。若预测错，保留错误推理与修正；若代码通过，再用一项负控制证明检查器确实能发现错误。无法解释状态变化时退回对应小课，不用堆更多随机参数掩盖理解缺口。

## 第一小时

从仓库根目录运行，依赖为Python标准库与NumPy：

```bash
python -m labs.serving.experiments --list
python -m labs.serving.experiments --lab lifecycle
python -m labs.serving.experiments --lab cow
python -m unittest discover -s tests -p 'test_serving*.py' -v
```

第一次不需要GPU、Docker、模型下载或API。先用可预测Bigram模型检查runtime，再运行TinyAttention与真实NumPy分页K/V，最后进入真实框架和GPU。

## 代码地图

- [`engine.py`](../../labs/serving/engine.py)：请求、调度、抢占、输出与不变量。KV是token元数据，时间是模拟值。
- [`kv.py`](../../labs/serving/kv.py)：事务式页申请、引用计数、COW、前缀缓存。
- [`backend.py`](../../labs/serving/backend.py)：Bigram与完整上下文NumPy单层attention。
- [`paged_attention.py`](../../labs/serving/paged_attention.py)：真正存储浮点K/V，gather参考与online分块归一化；独立CPU数值实现。
- [`sampling.py`](../../labs/serving/sampling.py)、[`streaming.py`](../../labs/serving/streaming.py)：采样与字节流边界。
- [`replay.py`](../../labs/serving/replay.py)：固定trace策略对照。
- [`mock_server.py`](../../labs/serving/mock_server.py)：本地HTTP教学服务，明确是有限接口子集。
- [`source_walk.py`](../../labs/serving/source_walk.py)：只读检查固定上游源码。
- [`install_upstream_test.py`](../../labs/serving/install_upstream_test.py)：dry-run复制新测试到独立上游checkout。
- [`collective_cpu.py`](../../labs/serving/collective_cpu.py)：有界多进程归约与身份/资源回收，不是NCCL性能。
- [`http_benchmark.py`](../../labs/serving/http_benchmark.py)、[`concurrent_probe.py`](../../labs/serving/concurrent_probe.py)：串行/有限并发SSE探测，默认dry-run、有限输入文件、协议终态校验。
- [`analysis.py`](../../labs/serving/analysis.py)：容量、通信、PD与量化的透明参考公式。
- [`h100_server.sh`](../../labs/serving/h100_server.sh)：默认dry-run，有界时长、明确设备、本地已有镜像/模型。

## 结果标签

课程离线单元测试与24主题实验已在CPU环境运行。NumPy数值分页KV通过独立测试，包括非整块、多头、真实数值COW、前缀复用和无效尾槽。真实vLLM测试文件已按固定源码接口核对、做语法检查，但没有在本次环境安装vLLM执行；H100/5070Ti、真实模型API与Docker实验未执行。任何模拟毫秒与模拟吞吐均有明确标签，不当作硬件性能证据。

深写覆盖、实际执行示例、故障修正与逐章篇幅见 [DEPTH_AUDIT.md](DEPTH_AUDIT.md)。篇幅只用于发现薄弱章节，执行与人工审读才支持相应的正确性结论。

## 参考资料

[来源与快照说明](SOURCES.md)及机器可读[sources.json](sources.json)记录核查日、固定commit与支持的结论范围。读源码与做练习均可在普通Linux CPU上进行；真正GPU任务集中在S15与需要设备数值/通信证据的阶段。
