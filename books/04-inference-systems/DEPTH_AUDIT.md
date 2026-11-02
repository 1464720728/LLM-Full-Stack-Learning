# 推理系统卷深度重写审计

本轮针对首版讲解跳步、每个机制篇幅不足的问题，直接扩充 S01–S19 主章。字数统计删除 Markdown 代码块后统计汉字；它只作为发现未展开章节的下限，不代替准确性、可读性和内容覆盖检查。

| 章 | 必须补齐的知识链 | 去代码汉字下限 |
|---|---|---:|
| S01 | HTTP/JSON/token→请求对象→队列→prefill/decode→采样→输出→完成和释放；逐变量追踪 | 13000 |
| S02 | 请求/序列/token三个预算；静态/连续批处理；分块、公平性、资源约束与计划提交 | 13000 |
| S03 | KV形状和字节→逻辑页/物理页→翻译→内部碎片→引用→COW→失败原子性 | 13000 |
| S04 | token身份与模型语义→整页前缀→查找→pin/unpin→LRU→租户与位置边界 | 13000 |
| S05 | 重算/换出→用户可见输出→RNG→取消→在途设备操作→延迟释放→前进证明 | 13000 |
| S06 | Work计划→扁平化/位置/slot mapping→model runner→worker→backend→kernel/graph | 13000 |
| S07 | logits→稳定softmax→过滤→RNG→停止条件→token解码→UTF8/SSE→背压 | 9000 |
| S08 | 时间边界→TTFT/ITL/TPOT/E2E→吞吐→分位数→负载模型→排队→实验设计 | 9000 |
| S09 | 固定快照→入口/调用链→类/接口→状态与所有权→测试→修改→证据范围 | 13000 |
| S10 | rank/group→矩阵切分→collective→TP/PP/DP/EP/CP→拓扑与数值 | 13000 |
| S11 | P/D工作区别→调度分离→KV传输布局→所有权交接→失败恢复→背压与收益 | 13000 |
| S12 | 浮点舍入→量化→分层误差→sampling→batch变化→首处分歧→可复现实验 | 9000 |
| S13 | 权重/KV/activation/workspace/graph/allocator→每rank账本→峰值→校准 | 9000 |
| S14 | 症状→层级→证据→最小复现→假设→负向控制→补丁→回归报告 | 9000 |
| S15 | Windows/WSL/Linux边界→单卡环境→模型身份→有界启动→负载→多卡验证 | 9000 |
| S16 | 需求→接口→实现里程碑→观察→故障注入→代码评审→项目验收 | 9000 |
| S17 | 真实Scheduler/helper/output逐段→预算手算→回归测试→mutation负向控制 | 13000 |
| S18 | prefix/COW/preempt的独立状态账→逐事件推演→复合情景→所有权不变量 | 13000 |
| S19 | 浮点K/V存储→append→地址翻译→gather→online归一化→数值COW与增量等价 | 13000 |

每章需要多轮连续小课、多个完整例子、逐行代码讲解、至少十二道分级练习与参考解答、至少三个明确故障情景。原有运行接口与实验保持可用；新增内容不得把模拟值写成GPU性能，也不得把未执行的上游测试写成实测。最终统计和检查结果在本文件补充。

## 分批执行证据

- S01–S05 首批：从正文提取并独立执行 18 个完整 Python 示例，全部通过。包括 UTF-8/JSON 往返、请求列表所有权、三输入三输出的逐轮生成、小型生命周期实现、零预算、取消幂等、预算分配、年龄计数、因果连接计数、COW 与释放、失败原子性、缓存附加/namespace 隔离、最长前缀 oracle、延迟释放状态机和后端错误隔离。
- 同批执行本卷现有 48 项 unittest，全部通过。运行接口未修改。
- 未作为独立程序运行的摘录：S02 的绑定实例方法内部片段、S05 的方法摘录。它们依赖完整 Engine 上下文；对应行为由现有测试和章节完整程序覆盖，不计作独立示例成功数。
- S01 HTTP 客户端示例需要显式启动本地服务，本批没有自动启动，尚未声称本轮完成其网络验证。GPU、Docker、上游 vLLM/SGLang 和外部模型 API 仍未执行。
- 正文明确修正首版跳步：computed 与 context 的一项差、首输出来自末 prompt、失败后状态是否改变、缓存拥有与活动拥有分离、逻辑取消与在途完成分离。S02 特别说明教学 phase 按 chunk 起点判定，重算 chunk 可以跨过 prompt 边界，避免自然语言阶段名称掩盖实际代码行为。

- S06–S08 第二批：独立执行 16 个完整 Python 示例，全部通过：可变长度打包与逆映射、packed attention 与逐请求 logits 对照、交错请求真实数值 K/V 与完整上下文对照、最后行索引、softmax/温度/平局、非法 logits/贪心 RNG、独立随机流、跨片段停止、全部二段字节切分与逐字节 SSE、有界队列、离线停止 oracle、完整时间线指标、插值百分位、FIFO 排队、系统人数面积、SLO 缺失值策略。S06 另有 1 段绑定方法源码摘录，不计独立程序。
- S06 执行时发现初稿误假定 TinyAttentionBackend 保存 width 字段；核对实际 backend.py 后统一改为 x.shape[-1]，并重跑所有相关例子通过。真实源码节选明确注明合并赋值已展开，不伪称逐字符原样。
- S08 执行 metrics CLI 及 replay.compare(DEFAULT) 九组配置，输出等价断言通过；时间继续标注 synthetic，未声称真实 GPU 速度。
- S19 由根任务协作者完成整章重写；其回报证据为 24 个 Python 块按章共享 namespace 执行通过（含后续独立审读新增反例）、tests.test_serving_tensor_kv 8 项通过、paged_attention CLI all_pages_returned=true。内容覆盖真实数值页、COW、失败原子、GQA、online softmax；未执行 GPU。最终全卷检查将合并验证。
- 当前新补硬件与网络机制为教学解释和受限实验计划，CUDA Graph、真实 vLLM/SGLang 推理、HTTP SSE 外部端点仍未在本批执行。PyTorch 官方 CUDA Graph 地址/形状机制于 2026-11-01 核查，正文链接明确来源。

- S09：重新通过 GitHub 连接器读取固定 SHA 的 EngineCore、SchedulerOutput、Scheduler、KVCacheManager、GPUModelRunner、SGLang Scheduler 与 RadixCache；前六份完整文本经 AST 解析成功，无框架导入。3 个完整 CPU 程序（提交/取消结算账本、真实 Runner 索引公式、AST 调用观察器）通过，2 个绑定源码节选不计独立程序。纠正 get_computed_blocks 三项返回、cached request 与 prefix hit 区别、computed/in-flight 乐观推进，以及真实 allocate_slots 可先清理滑窗块的失败语义。上游 pytest 仍未执行。
- S10：7 个完整 CPU 示例执行通过，包括四 rank ring reduce-scatter/all-gather 每步、列/行分片与 bias、两层 MLP、非均衡 PP 递推、MoE 路由逆映射、CP softmax 统计合并、ring 单位估算。固定 RowParallelLinear 节选不独立执行；Column/Row 源码已通过 GitHub 固定 SHA 核对，NCCL/PyTorch 官方 collective 文档已核查。
- 新增 labs/serving/collective_cpu.py：实际 spawn 2 与 4 个 CPU 进程成功，各 rank 获得相同整数归约结果；非法空组、异形、bool/float 输入前置拒绝通过。它是有界协调式通信语义教学，不是 NCCL/ring 性能实现。当前未执行 worker 故意失联/重复身份故障注入，正文将其标为学习任务；没有运行任何 GPU collective。新增模块后现有 serving 48 项测试再次通过，原 Engine/Backend/PagePool 接口保持不变。

- S11：5 个完整 CPU 程序实际通过，包括 KV 字节/尾页、真实 NumPy 数值交接与提前读/取消/旧代次、有限队列守恒、PD 成本模型、TP2→TP4 头重排。SGLang 释放就绪节选仅静态解释，不作为独立运行成功。重新读取固定 vLLM KVConnectorBase_V1 与 SGLang prefill.py/decode.py，明确查询/分配/加载/完成/异步释放契约，以及 host drain 与 device deadline fallback 的真实区别。没有声称执行 RDMA、GPU PD 或网络传输。
- S11 指标解释补充：P 直接输出首 token 时，KV 交接成本可能表现为首个 ITL，不能将 pd_break_even 的交接就绪近似 TTFT 公式无条件用于该协议。正文分别列出首输出与交接时间线。

- S12：10 个完整 Python 示例实际执行通过，涵盖 float32/64 非结合、误差报告、稳定 softmax、margin 翻转、四位手算、矩阵误差界、分组离群值、teacher forcing 与自由生成、首处分歧、配对题目统计。1 段 symmetric_quantize 内部源码节选不计独立程序。实际 TinyAttention 量化例的自由生成发生分歧，但正文不把该例泛化成所有量化必然改变输出。vLLM 官方确定性与固定 SGLang 支持矩阵仅文档核查，未运行 GPU 验证。
- S13：9 个完整 CPU 程序实际通过，包括 byte/GB/GiB、KV 字节、页尾与共享、保守 admission、逐 rank 不均衡、Capacity 支持边界、混合状态、生命周期峰值、页数残差。1 段 Capacity 方法摘录不独立运行。所有容量和观测示例明确为算术/人工数据，没有硬件测量主张。
- S17 根任务协作者回报：整章 13664 去代码汉字；20 个 Python 块共享 namespace 实际执行通过；重新核对固定 helpers 与 test_scheduler，细化真实三轮 400/400/224→400/400/224→1/1/352、computed/inflight、ID 映射、共享引用、Runner 两阶段、取消排空、SGLang batch 配对与 WAR。上游 pytest 未执行；正负控制作为明确环境中的后续实操。S09 已同步强调 long-prefill threshold 的单请求例外、adaptive 条件与 helper 的 trust_remote_code 边界。

- S14：7 个完整 CPU/本地网络示例实际通过：生命周期事件验证、真实 Engine 事件聚合、report 深层列表隔离、页拥有者守恒、临时 loopback mock 服务、累计直方图合并、同工作量配对计时。HTTP 例实际验证非法 max_tokens=65 返回 400、bad_request 增而 accepted 不增、后续合法两 token 完成、health 正常、无缓存输入全池归还，并只清理所创建子进程。它补充了首批未自动运行 HTTP 例的本轮网络证据，但仍不是 GPU 或生产协议验证。
- S15：5 个完整 CPU 程序通过，包括物理/可见设备映射、实验卡上限、内容 manifest、两线程 rendezvous 与失败保留、有界 UTF-8 输入。1 段可选 PyTorch CUDA Event 程序仅 compile 语法检查，未请求 GPU。新增 concurrent_probe 默认 dry-run、loopback、至多 8 请求/4 线程/每条 64 输出；原串行探针新增有界 prompt-file，保留旧 measure 参数兼容。两探针采用明确 completions [DONE] 终态契约，拒绝缺标记或错误事件。
- 新增 tests.test_serving_probes 的 3 项有意义回归实际通过：输入不静默截断/非法UTF8、并发身份与部分失败和线程归零、真实临时 loopback SSE 的输入原样传递/中文/usage/缺终止与错误事件。它们验证新功能边界，不计作教材深度。当前 serving 全部 51 项 unittest 通过。
- h100_server.sh 修复重复或非规范 GPU 编号可能把同一设备误当多个设备的问题。实际 bash -n 通过；以临时模型目录和明确的虚构镜像 fixture 仅执行 dry-run，1/2/8 卡有效参数成功、重复 0,0、00 和无多卡开关均前置拒绝。虚构摘要没有用于 Docker run，也未声称验证镜像或硬件。Docker、真实模型加载与 GPU 服务仍未运行。
- S18 根任务协作者完成并经独立审读修正：19 个 Python 块实际执行，345 组 COW 成功/少一页失败、9 组前进案例与 37 项管理/边界回归通过。四类状态损坏负控制、三时刻取消、RNG 对照、ConservativeResumeEngine 完整实现都有执行证据。审读发现仅判断缓存总长度整除会漏掉两半页，已加逐页满页断言与反例，重新执行通过。
- S19 独立审读后最终为 24 个执行通过的 Python 块：补充 K/V 联合置换在末 query 全可见条件下对任意输入保持 attention 不变的数学解释与随机反例，已有 RoPE 不改变此结论；因此顺序正确性另用 gather 逐行 oracle 验证，不能只看 decode attention 输出。数值COW、失败原子与全部页归还证据保持有效。

- S15 独立审读发现 urllib 默认重定向会绕过初始 loopback/allow-remote 检查，审读者用两个临时 loopback 端点实际复现。已在共享 measure 路径禁止 HTTP 重定向，并新增 302 目标未被访问的回归断言；相关 3 项 probe 测试重新通过。脚本同时加 --pull=never，明确保证只使用本地镜像；本轮仍未启动真实 Docker/GPU。S13 手算中文数字表达已按审读修正，不改变公式。

- 重定向修复经原审读者独立复验：127.0.0.1 入口返回 302 指向 127.0.0.2，当前并发 CLI 只访问入口、目标零请求，报告 successful=0 / ValueError 并退出1。拒绝路径显式关闭重定向响应；相关本地回归再次通过。独立审读对 S15、启动脚本和两探针无剩余阻断。

- S16 协作者完成并冻结：22 个 Python 块按章顺序实际执行通过，15 课、12 完整答案、3 个可执行故障与分阶段完整项目。OwnedWorker + Future 实际完成 24 并发提交/8 取消/16 完成、12 自动并发完成、8 前端桥接完成；取消幂等、active_limit 拒绝、输入复制、快照隔离、关闭中止与全部归页已验。完整模型项目覆盖 TinyDecoder/OnlineDecoder × FIFO/RR 四组合、六轮逐层 KV/full logits 对照与 step 异常原状态保持。上游测试文件 AST/hash、预算 0–1024 枚举及 total+1 负控制通过；没有执行上游 pytest、GPU、外部 API 或真实端口 HTTP 桥接。

## 最终逐章篇幅与正文执行证据

统计由 `python tools/check_depth.py --track S --check` 生成，删除 fenced code 后统计汉字，含标题、表格和完整答案。十九章全部达到本轮下限；没有检出跨章完全相同且至少一百二十汉字的重复段落。篇幅不是正确性证书，执行证据与上述审读修正分别列明。

| 章节 | 去代码汉字 | 本轮下限 | 实际执行完整 Python 块 | 执行方式 |
|---|---:|---:|---:|---|
| [S01](01-request-lifecycle.md) | 13158 | 13000 | 7 | 逐块独立 CPU |
| [S02](02-scheduling.md) | 13032 | 13000 | 3 | 逐块独立 CPU |
| [S03](03-paged-kv.md) | 13013 | 13000 | 3 | 逐块独立 CPU |
| [S04](04-prefix-cache.md) | 13027 | 13000 | 3 | 逐块独立 CPU |
| [S05](05-preemption-cancellation.md) | 13036 | 13000 | 2 | 逐块独立 CPU |
| [S06](06-runner-backends.md) | 13028 | 13000 | 4 | 逐块独立 CPU |
| [S07](07-sampling-streaming.md) | 9205 | 9000 | 7 | 逐块独立 CPU |
| [S08](08-measurement.md) | 9117 | 9000 | 5 | 逐块独立 CPU |
| [S09](09-source-reading.md) | 13129 | 13000 | 3 | 逐块独立 CPU |
| [S10](10-parallelism.md) | 13008 | 13000 | 7 | 逐块独立 CPU |
| [S11](11-pd-disaggregation.md) | 13015 | 13000 | 5 | 逐块独立 CPU |
| [S12](12-numerics-determinism.md) | 9282 | 9000 | 10 | 逐块独立 CPU |
| [S13](13-capacity.md) | 9255 | 9000 | 9 | 逐块独立 CPU |
| [S14](14-diagnostics.md) | 9084 | 9000 | 7 | CPU 与临时 loopback HTTP |
| [S15](15-hardware-labs.md) | 9266 | 9000 | 5 | CPU；另有1块GPU仅语法检查 |
| [S16](16-capstones.md) | 10977 | 9000 | 22 | 按章共享 namespace |
| [S17](17-source-workbook.md) | 13784 | 13000 | 20 | 按章共享 namespace |
| [S18](18-state-workbook.md) | 13163 | 13000 | 19 | 按章共享 namespace |
| [S19](19-numeric-paged-kv.md) | 13884 | 13000 | 24 | 按章共享 namespace |

本卷合计 **224,463** 去代码汉字；实际执行正文完整 Python 块合计 **165**。数字只按块计，单块内部可能覆盖多个输入；方法源码摘录和可选 GPU 块不计执行成功数。现有与新增本卷 **51** 项 unittest 通过，包含24主题离线实验、调度/缓存/严格边界、数值分页、真实本地SSE与并发探针回归。全仓本地Markdown链接与Python语法检查通过；本卷Python compileall与Shell语法检查通过。

最终保留的未验证边界：真实 vLLM/SGLang 安装与上游 pytest、Docker GPU 容器、H100/5070 Ti、CUDA/NCCL/PD 设备路径、真实模型服务性能、外部模型 API 和 EvalScope/Harbor 任务均没有本轮运行证据。CPU 模拟时间、数值参考、解析容量与真实墙钟分别标识；模型/框架支持以固定来源与实际环境复验为准。

## 1.1.1 正确性勘误复验（2026-11-02）

S07 修复重叠停止词的分片不变性：feed 等待可能更早开始的候选，finish 在正常结束时重新检查尾部。算法推导、手算轨迹、练习七／十四和项目验收同步更新。该章现有 8 个 Python 块按序执行通过，正文为 9,837 汉字；前文统计保留为 1.1.0 记录。

回归直接读取教材 StopFilter：二字符字母表上长度零至六的全部文本、长度一至三的单个或两个 stop 与全部非空分片组合，共 286,755 例通过；另有 119 例覆盖中文、空片段与指定重叠模式。检查累计输出前缀、尾部上界、最终正文和停止标志、重复结束以及停止后忽略输入。

serving SSE 仅移除冒号后至多一个空格。新测试在 LF／CRLF、零至三个普通空格、制表符、尾部空白、多行及空 data 下检查全部二段字节切点与逐字节输入，并与 Agent 解码器对照。完整软件结果见根目录 docs/VALIDATION.md。
