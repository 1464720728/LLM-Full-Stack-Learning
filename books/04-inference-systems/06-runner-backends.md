# S06 Model Runner：把调度计划翻译成张量与 Kernel

先修模型卷的矩阵乘法、attention 和 KV 基础，以及 S02–S04。框架的 Scheduler 不应该理解每个 CUDA 指令，Kernel 也不应该决定 HTTP 请求何时超时。连接两者的关键层是 Model Runner：它把请求状态与执行计划变成设备可执行的数据。

## 1. 四个层次不要混为一谈

模型定义规定数学计算，例如投影、归一化、attention、MLP；算子后端选择某类计算的实现；Runner 准备输入张量、位置、页表和 attention 元数据；Executor/Worker 负责在哪个进程、哪张卡上运行，以及跨 rank 协作。

名字会随框架变化，职责却能通过接口识别。一个 `execute_model` 方法可能包含输入准备、图回放、模型前向、结果暂存，但采样可能被拆成另一个方法。不能因为“模型执行结束”就假定已经拿到文本。

## 2. 从三条请求到扁平输入

A 本轮 prefill 3 个 token `[1,2,3]`，B decode 1 个 `[7]`，C prefill 2 个 `[4,5]`。一种布局是把 token 扁平拼接为 `[1,2,3,7,4,5]`。还要有 query 起止偏移 `[0,3,4,6]`，每请求已有上下文长度、当前位置、缓存页表等。

只有拼接 token 不够。若 attention 不知道边界，A 会看到 B/C 的 token，导致跨请求数据污染。若位置编号全从 0 开始，B 的 decode RoPE 位置就错了。若把请求的逻辑序列长度误写成这一轮 chunk 长度，attention 会丢失历史 KV。

手算：A 已计算 0，B 已计算 10，C 已计算 4，则本轮 positions 分别为 `[0,1,2]`、`[10]`、`[4,5]`。query offsets 指示本轮 query 的范围；seq_lens 则描述包含历史的可见长度。这两个数组用途不同。

## 3. Backend 接口与可替换性

本卷 [`backend.py`](../../labs/serving/backend.py) 定义 `Backend.logits(token_ids)`。BigramBackend 给出简单 transition logits；TinyAttentionBackend 使用 NumPy 实际计算 embedding、Q/K/V、最后位置的 attention 和输出投影。

TinyAttentionBackend 的计算为：`x=embedding[token_ids]`，`q=x[-1]@Wq`，`k=x@Wk`，`v=x@Wv`；分数 `q@k.T/sqrt(d)`，减最大值后 softmax，得到加权值，再做残差与输出投影。由于只计算最后位置，给定的全部上下文都是它的因果过去，不需要额外未来 mask。但这不意味着完整训练 attention 不需要 mask。

运行：

```bash
python -m labs.serving.experiments --lab runner_swap
python -m labs.serving.experiments --lab batch_equivalence
```

更换 Backend 后输出会不同，但调度、资源释放、终止条件应继续工作。这说明逻辑层通过契约解耦。它不证明两种模型质量可比，也不证明 TinyAttention 实现了训练所需的完整 Transformer。

## 4. 为什么真实 Runner 比我们的接口复杂

为了效率，真实 Runner 不会为每个请求逐个重算全部上下文。它要把批次变成紧凑张量，把 K/V 写入预分配存储，用页表访问历史，还可能缓存输入缓冲区以便 CUDA Graph 回放。不同模型还有 Mamba/SSM 状态、混合 attention 层、视觉特征、MoE 路由和 LoRA 等状态。

因此接口不只是 tokens→logits，还可能包括：slot_mapping、block_tables、position_ids、序列起止数组、有效 token 数、graph padding、encoder 输入、结构化输出 mask、speculative metadata。某个字段为空不代表无用，可能只在另一个模式启用。

最有效的源码阅读方式是选一个最简单的模式：单 GPU、普通 dense decoder、无 LoRA、无多模态、无推测解码、eager。先追踪一条路径，再逐项启用功能并记录增加了哪些字段。一次打开所有开关，会把基础数据流淹没在边界条件中。

## 5. Attention backend 的契约

同样的 attention 数学可由不同 Kernel 实现。后端选择考虑设备架构、dtype、head_dim、布局、稀疏模式、滑动窗口、KV 格式、是否需要确定性，以及编译或图捕获限制。名字相同的“Flash”实现也未必支持同一组输入。

一个良好接口应明确输入布局、mask 语义、输出 dtype、累加精度、支持的 stride、是否允许别名、是否异步。数学等价不意味着 bitwise 一致；支持同一 dtype 也不意味着同一舍入路径。替换后端时要同时测数值、边界 shape 和性能。

小型验收矩阵至少包含：长度 1、非整块长度、极长但可控长度、多请求不同长度、非默认 head_dim、全 masked 行的定义、NaN 输入如何处理，以及缓存与无缓存对照。测试只用 128/256 这种整齐尺寸，会错过尾块 mask 错误。

## 6. 图捕获与静态缓冲区

CUDA Graph 减少重复主机提交开销，但捕获的执行通常对地址和 shape 有要求。框架可能把实际 batch padding 到一个捕获尺寸，再传有效 token 数或 mask。padding 的额外行不能参与输出，也不能写入真实请求页。

如果某请求取消后，输入缓冲区对应行仍有旧值，这本身不一定错误；关键是有效范围和映射是否把它排除。把缓冲区全部清零可以掩盖错误，却未必修复映射契约。测试要故意用可辨识的旧值，证明失效行绝不会影响合法输出。

动态图形、新模型分支或某些 backend 可能不适合图捕获，于是框架提供 eager 退路。读性能报告时要记录图模式，冷启动捕获时间应与稳态推理延迟分开。

## 7. 练习与答案

1. **写扁平布局器。** 输入三请求的新 token 与 computed，输出 flat_ids、positions、query_offsets。对于上文例子，必须得到 offsets `[0,3,4,6]`。再写 inverse，按 offsets 还原原输入，验证往返一致。不要用 equal-length reshape 假装支持可变长度。
2. **数值对照。** TinyAttentionBackend 固定 seed，重复同一 context，logits 应逐值相同；调换请求遍历顺序后每请求 logits 不变。此结论来自各次调用没有共享可变 state，不是 GPU 的普遍保证。
3. **实现真正的增量 K/V。** 在 CPU 参考 backend 中为每个请求保存 `K,V`，新 token 只做一次投影。参考解法的核心是历史 append 和位置一致性：完整前向最后 logits 与逐 token cache logits 应 `allclose`。先关闭位置编码写最小版，再加 RoPE。必须使用 request ID 分隔缓存，并在 finish 时清理。
4. **后端异常边界。** 给 backend 返回空 logits、全 `-inf`、含 NaN，验收请求 error，页归还，其他请求不受影响。若 sampling 抛错却没有 cleanup，改 Engine 的错误边界，不要把 NaN 强行转成 0。
5. **真实源码追踪。** 在 vLLM `GPUModelRunner.execute_model` 找到输入准备与 model 调用，再找 `sample_tokens`。列出为何它们分离、哪些状态在两次方法间保存。答案必须有固定 commit、路径、符号，不能仅引用某篇博客的示意图。
6. **Backend 替换任务。** 在独立环境中固定一个小模型和输入，比较 eager 与默认图模式。记录冷启动、首轮、预热后分布、logits/输出差异；不要求性能提升。若没 GPU，先为本地模拟 runner 实现“捕获尺寸向上取整”的 padding wrapper，并证明 padding 不改变有效输出。

掌握门槛：能独立构造可变长度批次元数据；能解释 block table、slot mapping 和 positions 的不同含义；能给替换 backend 写数值与边界验收；能够沿源码从 SchedulerOutput 走到真正模型调用。

下一章：[S07 采样与流式输出](07-sampling-streaming.md)。
