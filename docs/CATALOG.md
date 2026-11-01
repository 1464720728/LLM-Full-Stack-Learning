# 完整课程索引

由各卷 manifest.json 生成。每课的详细练习、答案和验收在正文中。先修是最低建议，不等于必须先读完所有上游章节。

## F Linux、Python与工程基础

| 章 | 标题 | 先修 | 注册实验数 |
|---|---|---|---|
| F01 | [环境与学习闭环](../books/00-foundations/01-environment-and-learning.md) | 无 | 1 |
| F02 | [文件路径与编码](../books/00-foundations/02-files-paths-and-encoding.md) | F01 | 1 |
| F03 | [Shell管道与文本](../books/00-foundations/03-shell-pipes-and-text.md) | F02 | 1 |
| F04 | [进程权限与环境](../books/00-foundations/04-processes-permissions-environment.md) | F03 | 2 |
| F05 | [网络SSH与HTTP JSON](../books/00-foundations/05-network-ssh-http-json.md) | F04 | 2 |
| F06 | [Python数据与函数](../books/00-foundations/06-python-values-control-functions.md) | F01 | 2 |
| F07 | [模块对象与迭代器](../books/00-foundations/07-modules-objects-iterators.md) | F06 | 2 |
| F08 | [异常文件与SQLite](../books/00-foundations/08-files-exceptions-sqlite.md) | F07 | 2 |
| F09 | [测试与调试](../books/00-foundations/09-testing-debugging-and-quality.md) | F08 | 1 |
| F10 | [CLI日志与接口](../books/00-foundations/10-cli-logging-and-interfaces.md) | F09 | 1 |
| F11 | [并发与asyncio](../books/00-foundations/11-concurrency-async-and-cancellation.md) | F10 | 1 |
| F12 | [Git与源码修改](../books/00-foundations/12-git-from-snapshots-to-contribution.md) | F09 | 1 |
| F13 | [Docker镜像容器与卷](../books/00-foundations/13-docker-images-containers-volumes.md) | F04, F05, F10 | 1 |
| F14 | [数学张量与数值](../books/00-foundations/14-math-numpy-and-numerics.md) | F06, F09 | 2 |
| F15 | [项目A日志分析CLI](../books/00-foundations/15-project-log-analysis-cli.md) | F10, F12, F14 | 1 |
| F16 | [项目B任务服务](../books/00-foundations/16-project-task-service.md) | F05, F08, F09, F15 | 1 |
| F17 | [工作簿A从空文件到CLI](../books/00-foundations/WORKBOOK-01-from-empty-file-to-cli.md) | F06 | 5 |
| F18 | [工作簿B从状态到HTTP](../books/00-foundations/WORKBOOK-02-from-state-to-http.md) | F08 | 3 |
| F19 | [工作簿C Linux诊断训练场](../books/00-foundations/WORKBOOK-03-linux-diagnostic-lab.md) | F05 | 0 |
| F20 | [工作簿D Python源码阅读](../books/00-foundations/WORKBOOK-04-reading-python-source.md) | F07, F09 | 0 |
| F21 | [工作簿E手算到训练](../books/00-foundations/WORKBOOK-05-math-by-hand-and-training.md) | F14 | 1 |

## A Agent 工程：执行、状态、协议与可靠性

| 章 | 标题 | 先修 | 注册实验数 |
|---|---|---|---|
| A01 | [从最小函数到 Agent 循环](../books/01-agent-engineering/01-loop.md) | F06, F09, F10 | 1 |
| A02 | [模型协议与适配器](../books/01-agent-engineering/02-protocols.md) | A01 | 3 |
| A03 | [工具契约、验证与错误语义](../books/01-agent-engineering/03-tools.md) | A02 | 2 |
| A04 | [状态、事件与检查点](../books/01-agent-engineering/04-state-events.md) | A03 | 2 |
| A05 | [流式分帧与工具增量](../books/01-agent-engineering/05-streaming.md) | A04 | 2 |
| A06 | [容错、预算、取消](../books/01-agent-engineering/06-reliability.md) | A05 | 3 |
| A07 | [上下文与证据完整性](../books/01-agent-engineering/07-context.md) | A06 | 2 |
| A08 | [文件、编辑与进程工具](../books/01-agent-engineering/08-workspaces.md) | A07 | 2 |
| A09 | [并发与资源调度](../books/01-agent-engineering/09-concurrency.md) | A08, F11 | 1 |
| A10 | [幂等、账本与恢复](../books/01-agent-engineering/10-durability.md) | A09 | 2 |
| A11 | [状态图与人工协作](../books/01-agent-engineering/11-workflows.md) | A10 | 2 |
| A12 | [多 Agent 分工与证据合并](../books/01-agent-engineering/12-multi-agent.md) | A11 | 1 |
| A13 | [评测、追踪与统计](../books/01-agent-engineering/13-evaluation.md) | A12 | 3 |
| A14 | [编码助手综合项目](../books/01-agent-engineering/14-coding-agent.md) | A13, F12 | 1 |
| A15 | [MCP、A2A 与真实框架](../books/01-agent-engineering/15-integrations.md) | A14 | 1 |
| A16 | [源码工作簿：从四个函数到可恢复 Harness](../books/01-agent-engineering/WORKBOOK-1-build-and-trace.md) | A04, A10 | 1 |
| A17 | [源码工作簿：八个崩溃现场](../books/01-agent-engineering/WORKBOOK-2-crash-lab.md) | A10, A11 | 1 |
| A18 | [源码工作簿：真实模型、工具与框架迁移](../books/01-agent-engineering/WORKBOOK-3-production-bridge.md) | A15, A16, A17 | 1 |

## R 检索增强、记忆与知识图谱工程

| 章 | 标题 | 先修 | 注册实验数 |
|---|---|---|---|
| R01 | [检索与证据对象](../books/02-retrieval-memory-graphs/01-evidence-and-retrieval.md) | F08, F14, A03 | 2 |
| R02 | [切块、身份与摄入契约](../books/02-retrieval-memory-graphs/02-chunking-and-identities.md) | R01 | 3 |
| R03 | [BM25 从计数到公式](../books/02-retrieval-memory-graphs/03-bm25-from-counts.md) | R02 | 2 |
| R04 | [向量、索引与微型双编码器](../books/02-retrieval-memory-graphs/04-vectors-and-indexes.md) | R03 | 4 |
| R05 | [混合检索、重排与评价](../books/02-retrieval-memory-graphs/05-hybrid-rerank-evaluation.md) | R04 | 5 |
| R06 | [RAG、预算与引用](../books/02-retrieval-memory-graphs/06-rag-and-citations.md) | R05 | 2 |
| R07 | [查询规划与多跳](../books/02-retrieval-memory-graphs/07-query-planning-and-multihop.md) | R06 | 1 |
| R08 | [摄入、权限、版本与删除](../books/02-retrieval-memory-graphs/08-ingestion-access-lifecycle.md) | R07 | 5 |
| R09 | [Agent 记忆生命周期](../books/02-retrieval-memory-graphs/09-memory-lifecycle.md) | R08 | 3 |
| R10 | [有证据的知识图谱](../books/02-retrieval-memory-graphs/10-knowledge-graphs.md) | R09 | 1 |
| R11 | [图检索与全局问题](../books/02-retrieval-memory-graphs/11-graph-rag-and-global-questions.md) | R10 | 1 |
| R12 | [Agentic RAG 受控行动](../books/02-retrieval-memory-graphs/12-agentic-retrieval.md) | R11 | 0 |
| R13 | [证据助手与记忆助手项目](../books/02-retrieval-memory-graphs/13-two-capstone-projects.md) | R12 | 2 |
| R14 | [开源源码与前沿研读](../books/02-retrieval-memory-graphs/14-open-source-reading-and-frontiers.md) | R13 | 0 |
| R15 | [长工作簿：一条输入穿过知识系统](../books/02-retrieval-memory-graphs/WORKBOOK-source-to-running-system.md) | R14 | 8 |
| R16 | [真实Embeddings API与可选FAISS实操](../books/02-retrieval-memory-graphs/LAB-real-embeddings-and-faiss.md) | R04, R08, M02 | 3 |

## M 模型结构：从张量到可实现的现代架构

| 章 | 标题 | 先修 | 注册实验数 |
|---|---|---|---|
| M01 | [从一个汉字到一块张量](../books/03-model-architectures/01-tokens-tensors.md) | F06, F14 | 2 |
| M02 | [损失、链式法则与反向传播](../books/03-model-architectures/02-gradients-training.md) | M01 | 3 |
| M03 | [逐项实现稳定注意力](../books/03-model-architectures/03-attention.md) | M02 | 1 |
| M04 | [原始编码器—解码器Transformer](../books/03-model-architectures/04-original-transformer.md) | M03 | 1 |
| M05 | [完整稠密解码器](../books/03-model-architectures/05-dense-decoder.md) | M04 | 2 |
| M06 | [KV Cache与MHA/GQA/MQA](../books/03-model-architectures/06-kv-gqa.md) | M05 | 2 |
| M07 | [低秩、矩阵吸收与MLA](../books/03-model-architectures/07-latent-attention.md) | M06 | 1 |
| M08 | [MoE路由、专家、容量与合并](../books/03-model-architectures/08-moe.md) | M07 | 1 |
| M09 | [局部、块稀疏与压缩注意力](../books/03-model-architectures/09-sparse-attention.md) | M08 | 2 |
| M10 | [线性注意力与Delta记忆](../books/03-model-architectures/10-linear-attention.md) | M09 | 2 |
| M11 | [状态空间与选择性递推](../books/03-model-architectures/11-state-space.md) | M10 | 2 |
| M12 | [混合架构、AttnRes与残差变体](../books/03-model-architectures/12-hybrid-residuals.md) | M11 | 6 |
| M13 | [模型数值与量化](../books/03-model-architectures/13-quantization.md) | M12 | 1 |
| M14 | [采样、SFT、DPO、组优势与多模态](../books/03-model-architectures/14-training-modalities.md) | M13 | 7 |
| M15 | [综合项目：训练并恢复微型语言模型](../books/03-model-architectures/15-tiny-language-model.md) | M14 | 2 |
| M16 | [综合项目：架构比较台与真实源码研究](../books/03-model-architectures/16-architecture-study.md) | M15 | 1 |
| M17 | [深度工作簿：两token逐数追踪与MoE增量实现](../books/03-model-architectures/WORKBOOK.md) | M16 | 3 |

## S 推理系统与框架源码工程

| 章 | 标题 | 先修 | 注册实验数 |
|---|---|---|---|
| S01 | [一次请求怎样穿过推理系统](../books/04-inference-systems/01-request-lifecycle.md) | F11, M06 | 1 |
| S02 | [调度器：把请求变成每一轮可执行的工作](../books/04-inference-systems/02-scheduling.md) | S01 | 4 |
| S03 | [分页 KV：从地址翻译到引用计数](../books/04-inference-systems/03-paged-kv.md) | S02 | 2 |
| S04 | [前缀复用：缓存命中必须先保证语义正确](../books/04-inference-systems/04-prefix-cache.md) | S03 | 3 |
| S05 | [抢占、取消与故障：让系统在坏情况中仍然前进](../books/04-inference-systems/05-preemption-cancellation.md) | S04 | 3 |
| S06 | [Model Runner：把调度计划翻译成张量与 Kernel](../books/04-inference-systems/06-runner-backends.md) | S05 | 2 |
| S07 | [采样、停止与流式输出：模型之后仍然有很多正确性工作](../books/04-inference-systems/07-sampling-streaming.md) | S06 | 2 |
| S08 | [测量与排队：让性能结论可以被复核](../books/04-inference-systems/08-measurement.md) | S07 | 1 |
| S09 | [从教学引擎进入 vLLM 与 SGLang 源码](../books/04-inference-systems/09-source-reading.md) | S08 | 1 |
| S10 | [TP、PP、DP、EP、CP：先切清楚数据，再讨论卡数](../books/04-inference-systems/10-parallelism.md) | S09, K03 | 1 |
| S11 | [Prefill/Decode 分离：搬走计算，也搬来了状态协议](../books/04-inference-systems/11-pd-disaggregation.md) | S10 | 1 |
| S12 | [数值、量化与确定性：定位第一处分歧](../books/04-inference-systems/12-numerics-determinism.md) | S11, M13 | 2 |
| S13 | [容量模型：先算出每个 rank 的账，再启动模型](../books/04-inference-systems/13-capacity.md) | S12 | 1 |
| S14 | [服务诊断：从现象到最小复现](../books/04-inference-systems/14-diagnostics.md) | S13 | 0 |
| S15 | [从个人PC到共享H100：把真实硬件用于明确问题](../books/04-inference-systems/15-hardware-labs.md) | S14 | 0 |
| S16 | [综合工程：交付一个能解释、能测试、能诊断的推理系统](../books/04-inference-systems/16-capstones.md) | S15 | 2 |
| S17 | [源码工作坊：从一条真实请求计划到一个回归测试](../books/04-inference-systems/17-source-workbook.md) | S16 | 0 |
| S18 | [状态工作坊：亲手跟踪prefix、COW与抢占](../books/04-inference-systems/18-state-workbook.md) | S17 | 3 |
| S19 | [把真实K/V放进页里：实现可验证的分页注意力](../books/04-inference-systems/19-numeric-paged-kv.md) | S18 | 1 |

## K 算子、编译器与GPU硬件

| 章 | 标题 | 先修 | 注册实验数 |
|---|---|---|---|
| K01 | [张量不是表格：从一个地址开始理解布局](../books/05-kernels-hardware/01-addresses-layout.md) | F14, M03 | 2 |
| K02 | [为读 Kernel 学 C++：值、地址、生命周期与编译](../books/05-kernels-hardware/02-cpp-memory.md) | K01 | 1 |
| K03 | [CUDA 的执行组织：从一个线程到整个 Grid](../books/05-kernels-hardware/03-cuda-execution.md) | K02 | 1 |
| K04 | [访存：合并访问、缓存、共享内存与 Bank](../books/05-kernels-hardware/04-memory-hierarchy.md) | K03 | 2 |
| K05 | [归约、Softmax、RMSNorm：第一次把数值与并行一起考虑](../books/05-kernels-hardware/05-reductions-normalization.md) | K04 | 6 |
| K06 | [GEMM：从三个循环到分块复用与 Tensor Core](../books/05-kernels-hardware/06-gemm.md) | K05 | 2 |
| K07 | [把 FlashAttention 的核心算法亲手推出来](../books/05-kernels-hardware/07-online-attention.md) | K06 | 4 |
| K08 | [Triton 的思维方式：一个 Program 处理一块数据](../books/05-kernels-hardware/08-triton-model.md) | K07 | 1 |
| K09 | [把 Softmax、RMSNorm、GEMM 和流式注意力搬到 Triton](../books/05-kernels-hardware/09-triton-kernels.md) | K08 | 1 |
| K10 | [Kernel 进入框架以后：算子契约、反向传播与编译](../books/05-kernels-hardware/10-custom-ops-autograd.md) | K09 | 1 |
| K11 | [量化算子：值的编码、缩放、打包与累加](../books/05-kernels-hardware/11-quantized-kernels.md) | K10 | 4 |
| K12 | [MoE 算子：路由之后，数据怎样交给专家](../books/05-kernels-hardware/12-moe-grouped-gemm.md) | K11 | 1 |
| K13 | [Stream、Event 与 CUDA Graph：时间关系也是程序的一部分](../books/05-kernels-hardware/13-streams-events-graphs.md) | K12 | 1 |
| K14 | [性能工程：先提出可证伪的解释，再优化](../books/05-kernels-hardware/14-performance-method.md) | K13 | 2 |
| K15 | [从 Python/CUDA/Triton 到硬件：编译链的可观察边界](../books/05-kernels-hardware/15-compilation.md) | K14 | 0 |
| K16 | [H100 与你的 5070 Ti：同样写 CUDA，不同的硬件契约](../books/05-kernels-hardware/16-hopper-blackwell.md) | K15 | 1 |
| K17 | [Warp 分工、Persistent Kernel 与 MegaKernel](../books/05-kernels-hardware/17-warp-persistent-megakernel.md) | K16 | 0 |
| K18 | [通信算子与算子工程综合项目](../books/05-kernels-hardware/18-communication-capstone.md) | K17 | 3 |
| KW1 | [工作簿 W1：从空白文件写出可靠的 Softmax](../books/05-kernels-hardware/workbook-softmax.md) | K05, K08, K09 | 1 |
| KW2 | [工作簿 W2：从三重循环到 CUDA/Triton GEMM](../books/05-kernels-hardware/workbook-gemm.md) | K06, K09 | 1 |
| KW3 | [工作簿 W3：从 Dense Attention 到在线注意力，再读真实 FlashAttention](../books/05-kernels-hardware/workbook-attention.md) | K07, K09 | 1 |
| KS1 | [固定版本真实源码研读](../books/05-kernels-hardware/source-reading.md) | KW1, KW2, KW3, K17 | 0 |
