# 推理系统工程：从请求到框架源码与真实KV

本卷包含16章主课、3个深入工作坊、24个独立离线主题实验、轨迹回放和本地HTTP服务工程、真实NumPy分页K/V与online attention，以及固定vLLM源码的回归测试练习。它面向已经学过基础Python/数组、希望从“会调参数”进入“能改核心组件”的学习者。

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
- [`h100_server.sh`](../../labs/serving/h100_server.sh)：默认dry-run，有界时长、明确设备、本地已有镜像/模型。

## 结果标签

课程离线单元测试与24主题实验已在CPU环境运行。NumPy数值分页KV通过独立测试，包括非整块、多头、真实数值COW、前缀复用和无效尾槽。真实vLLM测试文件已按固定源码接口核对、做语法检查，但没有在本次环境安装vLLM执行；H100/5070Ti、真实模型API与Docker实验未执行。任何模拟毫秒与模拟吞吐均有明确标签，不当作硬件性能证据。

## 参考资料

[来源与快照说明](SOURCES.md)及机器可读[sources.json](sources.json)记录核查日、固定commit与支持的结论范围。读源码与做练习均可在普通Linux CPU上进行；真正GPU任务集中在S15与需要设备数值/通信证据的阶段。
