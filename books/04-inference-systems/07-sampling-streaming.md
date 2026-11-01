# S07 采样、停止与流式输出：模型之后仍然有很多正确性工作

先修概率、softmax、S01 与 Python bytes/string。即使 logits 完全正确，采样与流式处理出错也能改变结果、截断 Unicode 或重复发送内容。你需要把“一个 GPU token”“一个文本片段”“一个网络 chunk”分开。

## 1. 从 logits 到概率

logit 是未归一化分数。softmax 是 `exp(z_i)/sum(exp(z_j))`。实际实现先减最大值，避免 `exp(10000)` 溢出；减同一个常数不改变归一化结果。温度 T>0 时使用 `z_i/T`，T 小让分布更尖锐，T 大让分布更平。T=0 不是把公式除以零，而是单独定义为贪心选择。

数值例子：logits `[0,ln2,ln3]`，softmax 恰好 `[1/6,2/6,3/6]`。如果 top_k=2，留下后两项，再归一化为 `[0,2/5,3/5]`。如果先 top_p=0.7，按概率排序取 0.5，再加 1/3 越过阈值，因此保留两项。阈值穿过的那一项必须保留，而不是只保留累计仍小于阈值的项。

本参考实现明确采用“温度→top_k→top_p→重新归一化”。不同服务实现的处理顺序和约束可能有差别，对齐评测时要查实际源码与参数，不要仅认为字段名相同就等价。

## 2. 平局、mask 和非法值

贪心遇到相同最大 logit 要有稳定规则。我们选最小 token ID；top_k 排序也按分数降序、ID 升序。生产 Kernel 可能采用其他约定，跨实现比较时平局处理是可观测差异。

`-inf` 常用于禁止某 token；softmax 后概率 0。所有 token 都被 mask 时，没有有效概率分布，应报告错误，不应返回 NaN 再随便采样。NaN 和正无穷也应有明确处理。本代码拒绝它们，避免把数值故障伪装成随机输出。

结构化输出可以在采样前施加语法 mask，但如果状态机错误地屏蔽所有候选，就会产生全 `-inf`。正确调试方向是检查约束状态和 token 边界，而不是把温度改大。

## 3. 随机数应该属于谁

设共享 RNG 的序列是 `u1,u2,u3,u4...`。单独运行 A 会用 u1,u2,u3；A/B 交错时 A 可能用 u1,u3,u5。即使 logits 不变，A 也会变。这是调度造成的随机流差异，与浮点不确定性是两回事。

本引擎为每请求构造 `Random(seed)`，每生成一个实际输出才消费一次。重算不能消费，预填充中间 chunk 不能消费，被取消后的迟到结果也不应继续消费。真实系统可能使用基于 counter 的随机数生成，让 `(request,位置)` 映射到随机样本；核心目标仍是明确定义随机流归属。

运行：

```bash
python -m labs.serving.experiments --lab sampling
python -m labs.serving.experiments --lab deterministic
```

第二条应报告 equal=true；它证明参考模型在不同预算下保持每请求采样流一致，不证明所有 CUDA 后端都 batch invariant。

## 4. 停止规则与输出长度

常见停止条件包括 EOS、最大输出 token、停止字符串、工具调用结构完整、客户端取消、时间预算、上下文上限。它们发生在不同层。EOS 是 token 级；停止字符串是文本级，可能跨多个 token；超时来自运行时；工具完整性可能来自 parser 或约束解码。

如果停止字符串是 `END`，网络片段分别给出 `E`、`N`、`D`，逐片段检测会漏掉。你需要保留足够尾部字符，或使用增量匹配自动机。还要决定停止文本是否包含在返回内容中。不同 API 契约可能不同，不能靠肉眼“差不多”对齐。

本引擎只实现固定 max_new_tokens，不模拟 EOS 和聊天模板；这是刻意限制，不应被包装成完整兼容服务。拓展练习要求你在可测试范围内增加停止语义。

## 5. UTF-8、Tokenizer 与 SSE 是三层边界

UTF-8 一个汉字通常占多个字节，TCP 可以在任何字节之间切开。直接对每个 bytes chunk 调 `.decode()`，可能把半个汉字当非法序列。要使用增量 decoder，保留未完成字节。

Tokenizer token 也不一定等于一个字符或一个完整 Unicode 片段。正确的增量 detokenization 需要遵循 tokenizer 的拼接规则。有些 token 的可见文本要等后续 token 才能确定。不要把“每 token decode 再拼接”假设为所有 tokenizer 都成立。

SSE 事件通过空行分隔，事件的 `data:` 可以有多行，一个网络 chunk 可能含半个事件或多个事件。`SSEDecoder` 先增量解 UTF-8，再处理 CRLF 与空行，聚合 data 行。它有缓冲上限，避免服务端无限发一个不结束的事件占尽内存。

```bash
python -m labs.serving.experiments --lab stream_decoder
python -m unittest tests.test_serving_core.NumericalTests.test_sse_all_byte_boundaries -v
```

测试把同一条含中文事件在每一个可能字节边界切开，再断言结果相同。这样的边界穷举比“给两块正好切在换行处”强得多。

## 6. 网络 chunk 不能直接当 token

一个 SSE 事件可能带多个 token，也可能只是角色、空增量、usage 或完成标志。测 TTFT 时应定义“首个有效文本到达”，并排除空角色事件；测 ITL 需要真实 token 时间或清楚的服务器粒度。仅用客户端片段时间，应命名为 inter-chunk latency。

本卷真实 HTTP 探测脚本输出 `ttft_text_ms`、`mean_inter_chunk_ms`，不伪装为真实逐 token ITL。若响应 usage 给出 completion_tokens，才记录该数量；不能拿字符数、chunk 数或英文单词数替代 token 数。

## 7. 练习与答案

1. **手算温度。** logits 为 `[0,ln2]`，T=1 得 `[1/3,2/3]`；T=0.5 等效指数权重 `[1,4]`，得 `[1/5,4/5]`。这不是把概率直接除温度。
2. **nucleus 边界。** 原概率 `[.6,.3,.1]`，top_p=.8，保留前两项并归一化为 2/3、1/3。若 top_p=.6，第一项累计已达到阈值，只留第一项。编写明确 `>=` 的边界用例，允许浮点表示误差时使用精确构造或容差说明。
3. **随机流错误。** 将所有 Request.rng 换成共享 Random(0)，运行 deterministic。输出可能随预算改变。修复要恢复请求级流，不是把温度设零掩盖问题。
4. **停止字符串。** 实现一个纯函数接收片段列表和 stop，返回 stop 前文本。测试 stop 跨三片段、多个候选 stop、空片段、stop 根本不存在。参考：保存累计文本最简单但内存随输出增长；优化版只需保留最长 stop 长度减一的未确认尾部，确认安全前缀可以立即发送。
5. **非法 SSE。** 对截断 UTF-8、没有空行结束的事件、超长 data 写测试。答案：协议容错策略要显式定义；本 decoder final=True 时拒绝未完成事件，不能静默把尾部当完整 JSON。为 keepalive 注释行也写测试，它不应产生 data 事件。
6. **真实端点验证。** 在你自己启动的本地服务上显式执行 `http_benchmark`，最多 4 请求、每次 32 输出。报告必须写清楚 API 类型、model ID、模板、采样参数、是否收到 usage、片段粒度。没有 token 时间时，删除“ITL”标签，不进行推算补齐。

掌握门槛：能手算 top_k/top_p 后概率；能解释 seed 为什么不是全部确定性；能写不依赖网络切片边界的流解析；能区分 token、字符、SSE 事件、TCP chunk。

下一章：[S08 指标与实验设计](08-measurement.md)。
