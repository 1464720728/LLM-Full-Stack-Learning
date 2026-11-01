# P2：把手写模型接到调度器，再连接到算子

先修：M 卷 decoder／KV／采样，S 卷运行时与调度，K 卷 GEMM／在线注意力。CPU 与 NumPy 即可。代码 [model_runtime.py](model_runtime.py)，测试 [test_projects.py](../tests/test_projects.py)。

## 1. 三个问题分开验证

模型的问题是“给定上下文 token，输出怎样的 logits”；调度器的问题是“哪些请求这一轮可以前进，拥有多少资源”；算子的问题是“某个数学函数如何计算并验证”。生产框架把它们紧密连接，但学习时先把契约分开，错误才容易定位。

本项目创建 `DecoderBackend`，其 `logits(token_ids)` 调用 M 卷 `TinyDecoder.forward`，返回最后一个位置的 logits。这是真实的小模型数值计算，权重固定且很小，没有语言能力承诺。S 卷 Engine 只依赖 backend 的 `vocab_size` 和 `logits` 接口，不需要知道模型内部是否使用 RMSNorm 或 RoPE。

我们另外生成一组 Q、K、V，用 M 卷 dense attention 和 K 卷 online attention 两个独立实现对照。这不是把 online attention 自动装进 decoder；它先验证内核算法与模型公式的一致性。真正替换 decoder 内部算子需要适配形状、mask、布局与 dtype，属于后面的练习。

## 2. 运行并读取报告

```bash
python -m projects.model_runtime
```

报告中有 `direct`、`scheduled`、`matches_direct` 和 `independent_attention_max_error`。初版默认输入在已记录 CPU 环境中，direct 与 scheduled 相同；注意力差值接近 float64 舍入误差。具体 token IDs 来自固定随机初始化，不代表翻译或问答质量。

`scheduler_report` 中的时间明确标记为 synthetic。它用于观察调度策略和资源预算对队列的相对影响，不能拿来回答一张 H100 的吞吐量。NumPy backend 当前为每次解码重新计算完整上下文；页池存 token 元数据，不能把它叫做数值 KV cache 实现。

## 3. 直接生成的基线

`direct_generate` 从 prompt 复制一个 context。每一步调用 logits，选 `argmax`，把选中的 token 同时追加到 context 和输出列表。这个循环体现自回归：第 t 步生成的 token 成为第 t+1 步输入的一部分。`count=4` 表示生成四个新 token，不包括 prompt。

先手工假设 logits 是 `[0.1, 0.7, 0.2]`，第一次选 token 1。下次模型输入包含刚才选的 1；不能继续用最初的 logits 反复选 1 来代替自回归。测试模型未训练，仍然遵守同样的计算接口。

这条简单路径没有多请求竞争，所以很适合作为正确性基线。它不一定高效，但容易解释。工程中常用一个慢而清楚的参考实现对照优化实现；性能优化前先建立这个基线。

## 4. 调度路径为什么应保持输出

Engine 接收两个不同长度的 prompt，最大同时活动请求数为 2，每轮 token 预算和 prefill chunk 大小都有限。FIFO、decode_first、round_robin 改变请求获得执行机会的顺序。只要每个请求的输入上下文正确、模型固定、贪心采样确定，改变交错顺序不应改变各自输出。

这不是说真实 GPU 任意 batch 调度都能位级一致。真实实现可能选择不同 Kernel、归约顺序和通信路径，引入浮点差异。本项目在 CPU 小模型上建立逻辑不变量：调度器本身不得丢 token、重复追加、串错上下文或换掉模型身份。数值非确定性需要另一个专门实验，不应拿来解释所有逻辑错误。

观察 Engine 里的 Request：prompt 是初始上下文，output 是已生成 token，computed 表示元数据调度进度，state 表示生命周期。它们有不同含义。一个 token 已经生成，不等于某个模拟页已永久属于该请求；请求完成后页应释放或按缓存策略保留，不能泄漏。

## 5. 在线 attention 的对照

普通 attention 先计算完整分数矩阵，再对每行 softmax，最后乘 V。在线算法分块读 K/V，用运行最大值 m、归一化分母 l 和加权分子积累结果。新块的最大分数改变时，旧分子与旧分母必须乘同一个缩放因子。只重标定分母会得到错误输出。

本项目使用长度 7、维度 4、block=3，故意让最后一个块不足 3 个元素。这个选择验证尾块，而不是只测试整除的漂亮形状。causal mask 要保证第 i 行只看 j≤i 的键；不能因为分块而把未来位置放进归一化。

两个实现的输出逐元素比较。绝对误差小是证据之一，还应扩展输入尺度、非连续视图、不同 block 和不同维度。对于低精度 GPU 版本，误差阈值需根据 dtype 和算法定义，不能直接复制 float64 的 1e-12。

## 6. 练习与解答路线

### 练习 1：第三个请求

加入一个长度为 1 的 prompt，生成两个 token，保持 `max_sequences=2`。验收：三个请求都完成，各自结果等于 direct 基线，结束后资源检查通过。

解法：在 inputs 中增加条目，但每请求生成数不同时不能继续把常数 4 写死。建立 `specs={id:(prompt,count)}`，让直接路径和调度路径读取同一份规格。不要在测试中复制两份不同数据导致假失败。

### 练习 2：制造上下文串线

故意让 backend 缓存“最后一次上下文”并错误地给另一个请求复用。验收：单请求测试可以通过，多请求交错测试必须失败；写出最小反例。

参考解释：跨请求共享状态必须有 request ID、模型版本或明确的缓存键。一个无参的全局 last_context 无法表达所有权。修复可以保持 backend 无状态，或者给它设计显式的每请求状态接口；不能只在测试里把并发设成 1。

### 练习 3：使用真正的数值 KV

把 TinyDecoder 的 `step` 接口接入一个新的 per-request runner。验收：每个请求的完整前向与增量解码 logits 对照；两个请求交错不串线；取消后回收状态；新请求不能继承旧缓存。

参考设计：现有 Backend 只接收完整 token_ids，没有 request ID、reset、cancel 或缓存版本，所以直接把一个全局 KV 放进该接口是错误的。先设计 `start(request_id,prompt)`、`decode(request_id,token)`、`finish(request_id)` 等接口，写适配测试，再修改 Engine 调用点。分页元数据与实际 KV 的布局还需建立映射，不是把 Python dict 命名成 paged cache 就完成了。

### 练习 4：替换注意力实现

在一个独立模型副本中，把 dense attention 换成 K 卷在线实现。验收：相同参数、输入和 mask 下每层输出接近；不等长 Q/K 的支持边界有明确错误；原始 encoder-decoder 的 cross-attention 不被错误地套成 self-attention causal mask。

参考解法：先只替换二维单 head 情况，逐 head 调用并恢复形状，保持缩放和 mask 含义一致。当前 online 参考的 causal 模式要求相等 Q/K 长度；增量解码需要绝对位置偏移，不能把 query 的局部索引 0 当成整个序列的起点。写全上下文与单步缓存对照后再扩展。

### 练习 5：采样随机性

将 temperature 改成非零，为什么原来的 greedy 基线不再适用？怎样比较调度策略？

参考答案：基线必须使用相同采样算法和每请求独立随机状态。若所有请求共用一个全局 RNG，调度顺序可能改变随机数分配，即使 logits 完全相同输出也可能不同。为每请求固定 seed，并在恢复时保存 RNG 状态。即使如此，真实 GPU logits 的微小差异仍可能改变采样边界，不能把固定 seed 当成全系统确定性证明。

### 练习 6：从报告推断性能

有人看到 `time_ms=8.35`，宣称模型在 H100 上生成速度很快。写一段纠正并设计真实实验。

参考答案：这个值由模拟成本规则累加，没有 GPU、驱动、Kernel 或通信测量。真实实验应固定模型、权重、输入／输出长度、并发、运行栈和设备；区分 warmup／编译与稳定阶段；记录 TTFT、ITL、整体延迟和吞吐，采集多次样本。数值结果与性能结果分别验收。

### 练习 7：模型身份与前缀缓存

两个请求的 token 前缀一样，一个使用不同 adapter 或不同模型权重，能否共享数值 KV？验收：用一个反例说明 token 字符串相同不足以构成缓存键。

参考答案：KV 是输入经过模型投影后的状态，依赖权重、位置处理、adapter、dtype 等条件。当前 namespace 提示你把模型／adapter／租户语义纳入缓存隔离。真正生产 key 的构成以对应框架源码和配置为准，不能仅凭文件名猜测。

### 练习 8：从教学适配器走向上游源码

沿 S 卷锁定提交找到模型 Runner 和 scheduler 的边界。列出上游比本项目多出的至少五种约束，再选择一个小变更添加回归测试。

参考方向：设备 Tensor 生命周期、异步执行、真实 KV block table、采样元数据、分布式 worker、模型加载／量化、流式输出、取消与资源释放。先选择能明确验证的改变，例如给一个边界条件增加测试或完善指标；不要第一次进入仓库就替换整个调度器。

## 7. 项目验收

你应能从一条请求的 prompt 追踪到 logits、采样、output、下一轮 context，再说明调度器怎样让多个请求交错；能指出元数据页池和数值 KV 的区别；能用一个独立参考核对在线注意力；能设计至少一个真实 GPU 实验并说清哪些结论不能从 CPU 模拟中推出。完成这些后，再扩大模型和硬件规模才有可靠基础。
