# S10 TP、PP、DP、EP、CP：先写清每个 rank 的数据，再启动多卡

八张 H100 不会因为插在同一台服务器里就自动组成一块巨大显存。你需要决定权重、激活、KV、请求和通信分别放在哪里，再用进程组与执行顺序把它们连接起来。本章从四个数字的矩阵乘法开始，逐步构造张量并行、流水线并行、请求副本、专家路由和上下文分片，最后读真实并行线性层源码。先修矩阵乘法、S03 页表和 S06 Runner；暂时不会 NCCL 也可以完成全部 CPU 数值练习。

本章不要求一开始使用八卡。数学分片与路由在 NumPy 上验证，进程通信先用很小的本地 CPU 任务，只有检查设备通信时才使用已分配的两卡，最后按研究目标决定是否扩大到八卡。所有通信时间公式都是分析模型，不是你服务器的实测。NCCL 与 PyTorch 分布式接口说明于 2026-11-01 通过官方资料核查，实际安装版本仍需单独记录。

## 1. 第一课：进程、设备、rank 与 group 是四种对象

进程是操作系统管理的执行实体，GPU 是设备，rank 是某个分布式参与者的编号，group 是共同参加某类通信的一组 rank。教学中常用一进程对应一 GPU，但这只是常见映射，不是这些名词天然相同。一个进程可以知道多个 group，一个物理节点可以运行多个 rank，一个模型也可以同时使用 TP 与 PP 组。

world_size 通常指默认进程组的参与者数量；local_rank 常用于本节点设备映射；某个子组内部又有自己的组内 rank。设备可见列表改变后，进程看到的 cuda:0 可能对应物理机器上的另一张卡，因此日志应同时记录分配给任务的可见设备与 rank 映射。不要用 rank 数字直接猜物理 GPU 编号。

假设八个全局 rank 组成 DP=2、TP=2、PP=2。可以人为规定编号为 `((dp*pp_size)+pp)*tp_size+tp`。那么同一 DP 副本、同一 PP 阶段中的两个 TP rank 一起完成层内计算；同一副本、同一 TP 位置的两个 PP rank 传递层间激活；两个 DP 副本分别服务不同请求。这是一个明确的教学映射，真实框架编号可能不同，必须查其 group 构造。

| 全局 rank | DP 副本 | PP 阶段 | TP 位置 | 本例角色 |
|---:|---:|---:|---:|---|
| 0、1 | 0 | 0 | 0、1 | 副本零的前半层，层内二分 |
| 2、3 | 0 | 1 | 0、1 | 副本零的后半层，层内二分 |
| 4、5 | 1 | 0 | 0、1 | 副本一的前半层，层内二分 |
| 6、7 | 1 | 1 | 0、1 | 副本一的后半层，层内二分 |

同一个请求通常选择一个 DP 副本，然后经过该副本的 PP 阶段，在每个阶段内部使用 TP 组。它不会自动使用另一个副本的空闲 KV。把八卡总显存除以模型大小得到“应该能跑”，忽略了这个所有权结构。容量必须按最重 rank 计算，S13 会建立完整账本。

通信组也是协议身份。TP all-reduce 应在 TP 组里执行，不能误用默认全局组把其他副本的请求激活加进来。即使所有张量 shape 相同，错误 group 也可能得到数值看似合理但语义错误的输出。分布式正确性首先是“谁和谁在处理同一份数学问题”。

## 2. 第二课：六种通信原语各自做什么

all-reduce 将各 rank 的同形张量按某个运算归约，并让所有参与者得到归约结果。最常见求和：rank 零有 `[1,2]`，rank 一有 `[10,20]`，结果每边都是 `[11,22]`。它不是拼接，也不自动除以 rank 数；若需要平均，必须使用明确的平均语义或再除以数量。

all-gather 把各 rank 分片按组内 rank 顺序收集到所有参与者。例如零有 `[1,2]`，一有 `[3,4]`，每边得到 `[1,2,3,4]`。reduce-scatter 先对完整对应元素归约，再让每个 rank 只保留结果的一段。broadcast 从指定 root 复制同一份数据给其他 rank；reduce 只有 root 获得归约结果；all-to-all 则每个 rank 向每个目标发送各自不同的片段。

官方 NCCL 文档强调，参与同一集合操作的 rank 必须按相应契约调用兼容的数量与 dtype；all-gather 的结果顺序由 rank 决定。具体 sendcount/recvcount 含义要看每个 API，不能把所有操作的 count 当同一种总长度。[NCCL collective 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)与[参数定义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html)是核查入口。

| 操作 | 输入分布 | 输出分布 | 典型推理用途 |
|---|---|---|---|
| all-reduce | 每 rank 同形部分和 | 每 rank 完整和 | row parallel 输出 |
| all-gather | 每 rank 不同分片 | 每 rank 完整拼接 | 恢复隐藏维或词表片段 |
| reduce-scatter | 每 rank 完整部分和 | 每 rank 一段归约结果 | 保持后续激活分片 |
| broadcast | root 有完整数据 | 每 rank 相同数据 | 一致 token/控制信息 |
| all-to-all | 每 rank 有发往各目标的数据 | 每 rank 收齐发给自己的数据 | MoE token 分发 |
| send/recv | 一个发送者与接收者 | 目标获得指定数据 | PP 激活传递 |

集合操作还需要顺序一致。若 rank 零先做 all-reduce 甲再做乙，而 rank 一先做乙再做甲，系统可能等待、报错或把不对应的工作匹配到一起。多个进程组并用时也要保证全局一致的执行顺序，这是 PyTorch 官方分布式说明特别指出的要求。不能因为两次操作在不同 Python 函数里，就认为它们没有同步关系。[官方进程组说明](https://docs.pytorch.org/docs/stable/distributed.html)

CPU 参考可以把 all-reduce 写成数组求和，把 all-gather 写成 concatenate，但真实 NCCL 还包含通信器、stream、设备缓冲区与完成事件。本章先证明数组语义，再讨论设备执行；不要把 NumPy 的函数调用耗时当 NCCL 性能。


### 2.1 四个 rank 的 ring all-reduce，每一步到底传哪一块

把每个 rank 的输入切成四个等长块，编号零到三。为便于手算，每块只有一个数：rank 零输入一、二、三、四；rank 一输入十、二十、三十、四十；rank 二输入一百、二百、三百、四百；rank 三输入一千、二千、三千、四千。目标归约结果是一千一百一十一、二千二百二十二、三千三百三十三、四千四百四十四。

规定环方向为零到一到二到三再到零。reduce-scatter 第一轮，每个 rank 发送与自己编号相同的块给下一个 rank，接收前一个 rank 的对应块并加到本地同编号块。rank 零收到 rank 三的块三，得到四加四千；rank 一收到块零得到十加一；rank 二收到块一得到二百加二十；rank 三收到块二得到三千加三百。

第二轮转发上一轮刚合并的块，接收另一个块继续累加。第三轮后，每个 rank 恰好拥有一个已经汇集四份贡献的完整归约块：rank 零拥有块一，rank 一拥有块二，rank 二拥有块三，rank 三拥有块零。其余块可能仍是中间部分和，不能把整块本地数组都当作最终结果。

| reduce-scatter 轮次 | rank 0 本轮更新 | rank 1 本轮更新 | rank 2 本轮更新 | rank 3 本轮更新 |
|---:|---|---|---|---|
| 1 | 块3=4004 | 块0=11 | 块1=220 | 块2=3300 |
| 2 | 块2=3303 | 块3=4044 | 块0=111 | 块1=2220 |
| 3 | 块1=2222 | 块2=3333 | 块3=4444 | 块0=1111 |

all-gather 阶段不再相加，而是传播已经完整归约的块。rank 零先把块一发给 rank 一，同时从 rank 三收到块零；下一轮继续转发刚收到的块零并接收块三；最后接收块二。三轮后每个 rank 都有全部四个归约块。若这一阶段错误地继续加法，会重复累计贡献。

下面程序用同步数组快照模拟每一轮，避免在一个 Python 循环里先更新某 rank 后，后面的 rank 错误读到“本轮未来值”。这相当于所有发送读取轮初缓冲区，接收写到轮末缓冲区。它演示算法语义，不声称复现 NCCL 实际调度、chunk 大小或链路协议。

```python
import numpy as np

original = np.array([[1,2,3,4],[10,20,30,40],
                     [100,200,300,400],[1000,2000,3000,4000]])
state = original.copy()
ranks = len(state)
trace = []
for step in range(ranks-1):
    previous = state.copy()
    updates = []
    for rank in range(ranks):
        source = (rank-1) % ranks
        chunk = (rank-step-1) % ranks
        state[rank,chunk] = previous[rank,chunk] + previous[source,chunk]
        updates.append((rank,chunk,int(state[rank,chunk])))
    trace.append(updates)
for rank in range(ranks):
    chunk = (rank+1) % ranks
    assert state[rank,chunk] == original[:,chunk].sum()
for step in range(ranks-1):
    previous = state.copy()
    for rank in range(ranks):
        source = (rank-1) % ranks
        chunk = (rank-step) % ranks
        state[rank,chunk] = previous[source,chunk]
expected = original.sum(axis=0)
np.testing.assert_array_equal(state,np.tile(expected,(ranks,1)))
print("reduce-scatter trace:",trace)
print("all-gather result:",state.tolist())
```

每个 rank 每阶段发送三块，块大小为 N/4，所以总发送量六乘 N/4，也就是一点五倍 N。这与 `2*(p-1)/p*N` 在 p 为四时一致。这个推导让通信量公式有了具体数据路径，而不再是一串需要背诵的符号。接收量也相应存在，但不能在带宽模型里不说明口径就随意把发送与接收再加一倍。

真实实现可以将大张量继续切成更多传输 chunk，使用流水、多个通道或不同算法来利用拓扑。数学归约顺序也可能改变浮点舍入，所以两种正确 collective 不一定逐位相同。本例使用小整数精确比较，是为了先排除数值误差，把算法与索引错误暴露出来；浮点验证会在 S12 进一步展开。
## 3. 第三课：一层矩阵乘法的两种分片

设输入 X 为 `[m,k]`，权重 A 为 `[k,n]`，输出 Y=XA 为 `[m,n]`。m 可以理解为本轮 token 行数，k 是输入隐藏宽度，n 是输出宽度。数学写法与某些库保存 weight 的 `[out,in]` 轴顺序可能相反；读源码时要看 forward 使用的乘法约定，而不是只凭参数张量第一维叫“行”。

按输出列切 A，得到 A0 和 A1，各为 `[k,n/2]`。两 rank 都需要完整 X，分别算 Y0=XA0、Y1=XA1。把 Y0、Y1 沿输出维拼接即可得到完整 Y。若下一层能直接消费这两个分片，就不必立刻 all-gather；避免不必要的收集是张量并行的重要设计空间。

按输入行切 A，同时按 X 的隐藏维切成 X0、X1，则 Y=X0A0+X1A1。每个 rank 得到 `[m,n]` 部分和，需要求和归约恢复完整输出。这里两个张量不是不同输出列，而是相同输出元素的不同贡献，因此用 concatenate 会得到错误 shape，用平均会把结果缩小。

手算 X 为 `[1,2,3,4]`，A 的四行为 `[1,0]`、`[0,1]`、`[1,1]`、`[2,0]`。完整输出第一项为一加三加八等于十二，第二项为二加三等于五。输入维前半的贡献是 `[1,2]`，后半是 `[11,3]`，相加恢复 `[12,5]`。

```python
import numpy as np

x = np.array([[1.,2.,3.,4.]])
a = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,0.]])
full = x @ a
column_parts = [x @ a[:, :1], x @ a[:, 1:]]
column_result = np.concatenate(column_parts, axis=-1)
row_parts = [x[:, :2] @ a[:2, :], x[:, 2:] @ a[2:, :]]
row_result = sum(row_parts)
np.testing.assert_array_equal(full, [[12.,5.]])
np.testing.assert_array_equal(column_result, full)
np.testing.assert_array_equal(row_result, full)
bias = np.array([1.,-1.])
correct = row_result + bias
wrong = sum(part+bias for part in row_parts)
np.testing.assert_array_equal(correct, [[13.,4.]])
np.testing.assert_array_equal(wrong, [[14.,3.]])
print("column gather and row sum recover the full matmul")
```

bias 必须只加一次。若每个 rank 都在部分和上加完整 bias，再 all-reduce，bias 会被加 rank 数次。可在归约后加一次，也可只让一个 rank 在归约前加，或采用其他明确等价的分配。优化框架可能把 bias 留给后续融合操作，因此看到 forward 返回 output_bias 并不意味着忘了加，而是接口把责任交给调用者。

## 4. 第四课：两层 MLP 怎样减少中间通信

考虑 MLP 为 `Y=φ(XA)B`，A 将隐藏维 d 扩大到中间维 h，B 再从 h 投影回 d。把 A 按输出列分片，每个 rank 得到自己的一段中间激活；逐元素激活 φ 可以在本地完成，因为每个元素不依赖其他隐藏维。再把 B 按输入行对应分片，各 rank 计算部分输出，最后 all-reduce 求和。

这条路线不必在两层之间收集完整 h 维激活。因为第二层恰好消费同一分片维度，分片可以跨层传递。若你在第一层后立即 all-gather，第二层又重新切片，数学仍然正确，但产生额外通信与内存。并行设计要看相邻算子如何衔接，不能孤立地给每层加一次 gather。

激活函数必须符合所假定的局部性。普通逐元素非线性可以独立做；跨隐藏维归一化需要全局统计，不能随意只对本地分片求均值方差。gated MLP 还要求 gate 与 up 分片对应同一中间元素，再做逐元素相乘。把两组分片顺序弄错，shape 完全相同也会得到错误结果。

```python
import numpy as np

rng = np.random.default_rng(5)
x = rng.normal(size=(3,4))
a = rng.normal(size=(4,8))
b = rng.normal(size=(8,4))
relu = lambda z: np.maximum(z, 0)
full = relu(x@a)@b
partials = []
for rank in range(2):
    sl = slice(rank*4, (rank+1)*4)
    hidden_shard = relu(x@a[:,sl])
    partials.append(hidden_shard@b[sl,:])
sharded = sum(partials)
np.testing.assert_allclose(sharded, full, rtol=1e-12, atol=1e-12)
print("column-then-row MLP matches reference", sharded.shape)
```

这个例子只模拟数学，不启动两个进程。它让你检查分片轴、激活和归约位置。如果把 sum 改成 concatenate，输出宽度会错误扩大；如果对每个 partial 再做 relu 后相加，也一般不等于原式，因为非线性不能随意穿过求和。优化通信前必须确认代数等式成立。

TP 增大以后，每个 rank 的矩阵更小，单卡 GEMM 效率可能下降，同时集合通信次数仍然存在。因此单请求延迟不一定随 TP 成比例降低。大模型为容量不得不分片时，TP 是必要手段；小模型本来一张卡就能容纳时，多卡可能反而增加延迟。选择并行度应结合容量、batch 与目标，而不是“卡越多越先进”。


### 4.1 一层 Transformer 的 TP 通信账本

在一个常见的头分片思路中，各 TP rank 起初持有相同的 token 隐藏状态 X。QKV 投影按输出维或 head 组织分片，各 rank 得到自己负责的 Q/K/V；位置编码在本地对相应 head 应用，K/V 写入本 rank 的缓存；attention 在本地 head 上读取历史，产生局部 head 输出。然后输出投影按输入行分片，得到隐藏维上的部分和，归约后恢复每 rank 相同的完整隐藏状态。

之后残差与归一化按照具体实现处理，再进入 MLP 的列分片、局部非线性、行分片和归约。这个“两个大模块各有一次归约”的叙述是常见基础布局的教学账本，不是对所有模型与框架的固定次数保证。序列并行、融合、MoE、不同残差结构和后端优化都可能改变实际通信位置。

关键是每个通信点前后，张量的分布发生了什么变化。attention 输出投影前是 head 分片；输出投影后的局部结果是同一隐藏状态的部分和；all-reduce 后是复制的完整隐藏状态。若把“分片”和“部分和”都简称为 local tensor，就很容易选错 gather 或 reduce。

| 阶段 | 每 rank 数据 | 数学关系 | 下一步需要什么 |
|---|---|---|---|
| 层输入 | 完整 X 的副本 | 各 rank 相同 | 本地 QKV 投影 |
| attention 输入 | 本 rank 的 heads | head 子集 | 本地缓存与注意力 |
| 输出投影后 | 完整隐藏宽度的部分和 | 各 rank 贡献相加 | 归约 |
| attention 残差后 | 完整隐藏状态副本 | 数学结果一致 | MLP 列分片 |
| MLP 中间 | 中间宽度分片 | 各 rank 不同区间 | 本地非线性与行投影 |
| MLP 行投影后 | 隐藏宽度部分和 | 各 rank 贡献相加 | 归约与残差 |

归一化需要特别小心。若每 rank 已有完整隐藏状态，RMSNorm 或 LayerNorm 可以各自算出相同结果；若隐藏维被切开，只用本地平方和或均值就不等于全维统计，需要相应跨 rank 合并。序列维分片则每 rank 仍可拥有某些 token 的完整隐藏维，归一化可以对本地 token 独立做。分片轴不同，是否需要通信也不同。

同理，残差相加要求两条路径在同一张量分布下对齐。一个是完整副本、另一个是部分和时，不能直接加完再随意归约，否则残差会被重复累加。把分布状态写进接口说明，甚至在教学类型名中区分 replicated、sharded 与 partial，可以帮助你提前发现这种错误。
## 5. 第五课：attention、GQA 与 KV 到底怎么分

普通多头注意力可以按 head 分给不同 rank。每个 rank 计算自己的一组 Q/K/V 与 attention 输出，再通过输出投影的分片与归约恢复隐藏状态。这是直觉起点，但真实模型可能采用 grouped-query attention：Q heads 多，KV heads 少，一组 Q heads 共享一组 KV。

设 Hq 为八、Hkv 为二，TP 为二，最简单均匀方案每 rank 四个 Q head 与一个 KV head。若 TP 为四，Q 可以每 rank 两个，但两个 KV head 无法各自再按“一个完整 head”平均分给四个 rank。某些实现会复制 KV head，某些采用其他布局；必须看具体后端与模型实现，不能永远把 KV 字节除以 TP。

每 rank 持有的 KV 还与 PP 层分配、CP 序列分片和缓存 dtype 有关。统一全注意力模型在简单 TP 头分片下，每 token 每 rank KV 字节可写为二乘本 rank 层数乘本 rank KV heads 乘 head_dim 乘每元素字节。这里的二来自 K 与 V，不是两个 rank。若发生复制或混合状态，公式要按实际结构修改。

位置编码与缓存写入也必须保持一致。同一个逻辑 head 在不同 rank 上的旋转位置不能不同；复制的 KV head 应具有相同语义内容，但浮点路径与设备执行仍需验证。若只是把头维切成几段而没有相应 attention 算法，点积与 softmax 会缺少贡献，不能把任何切分都叫 head parallel。

输出词表也可以分片。每 rank 持有一部分 lm_head，得到局部 logits。贪心可以比较各 rank 局部最大值再选择全局最大，但随机采样、top-p 和 logprobs 需要相应全局概率处理，不能每 rank 独立采一个 token 再随便挑一个。分片延伸到采样层时，S07 的概率契约仍然成立。

## 6. 第六课：PP 按层切分，流水线为什么会有空档

Pipeline Parallel 把模型层放到不同 stage。一个输入先经过 stage 零的层，再把隐藏状态发给 stage 一，依次前进。不同于 TP 同时计算同一层的部分结果，PP 让不同 stage 计算模型的不同层。权重与 KV 通常按层归属，负责哪些层就持有哪些层的状态。

单个微批必须遵守层依赖，后面的 stage 不能在输入到达前工作。多个微批可以交错：stage 一处理甲时，stage 零处理乙。开始填充与最后排空时，有些 stage 没有工作，这些空档常叫流水线 bubble。若只有一个短 decode 请求，没有足够独立微批，流水线很难填满。

假设三个 stage 每个微批都花一个时间单位，处理四个微批，忽略通信，时间格如下。最后完成时间为 stage 数加微批数减一，即六。总有效计算量十二个 stage-微批单位，设备总可用时间十八，因此理想利用比例三分之二。这个公式只在等时、可流水、无额外依赖的假设下成立。

| 时间格 | stage 0 | stage 1 | stage 2 |
|---:|---|---|---|
| 1 | 甲 | 空 | 空 |
| 2 | 乙 | 甲 | 空 |
| 3 | 丙 | 乙 | 甲 |
| 4 | 丁 | 丙 | 乙 |
| 5 | 空 | 丁 | 丙 |
| 6 | 空 | 空 | 丁 |

若 stage 时间分别一、三、一，慢的中间 stage 限制吞吐，不能继续使用等时公式。embedding、lm_head、MoE 层和不同层结构可能导致计算与显存不均衡；简单按层数平均切分不一定平衡。容量判断也应看最重 stage，而不是平均每卡权重。

自回归还带来输出反馈。最后 stage 采样得到下一 token 后，前面的阶段需要知道它才能继续该请求的下一轮。真实框架通过相应消息或广播维持这个依赖。PP 可以同时推进多个请求/微批，但不能让一个请求未来 token 在尚未采样时凭空出现。S09 中非最后 rank 返回 IntermediateTensors 正是这个层级边界。


### 6.1 用递推程序处理不均衡 stage

等时流水线公式只是特殊情况。更一般地，微批 j 在 stage i 的开始时间，是“同一微批在前一 stage 完成”和“同一 stage 的前一微批完成”两者的最大值。再加本 stage 服务时间得到完成时间。这个递推同时表达数据依赖与设备一次只能处理一个微批的资源限制。

```python
def pipeline_finish(stage_times, microbatches):
    if not stage_times or microbatches <= 0 or any(t <= 0 for t in stage_times):
        raise ValueError("positive stages and microbatch count required")
    finish = [[0.0]*microbatches for _ in stage_times]
    for stage,duration in enumerate(stage_times):
        for batch in range(microbatches):
            from_previous_stage = finish[stage-1][batch] if stage else 0.0
            from_previous_batch = finish[stage][batch-1] if batch else 0.0
            finish[stage][batch] = max(from_previous_stage,
                                       from_previous_batch) + duration
    return finish

balanced = pipeline_finish([1,1,1],4)
unbalanced = pipeline_finish([1,3,1],4)
assert balanced[-1][-1] == 6
assert unbalanced[-1][-1] == 14
print("balanced:",balanced)
print("unbalanced:",unbalanced)
```

不均衡例子的第一微批需要一加三加一共五单位，后续每个微批主要被中间 stage 的三单位节奏限制，四个微批总共十四单位。增加微批可以减少填充排空相对占比，但不能把中间 stage 本身变快。若一个阶段远慢于其他阶段，应该重新分层、改变算子或资源布局，而不是只增加排队请求。

通信可以作为前后 stage 之间的额外依赖延迟加入递推，但若通信与计算争用资源，简单相加仍不够精确。更完整模拟需要分别表示计算设备与链路占用。教学阶段先明确假设，再逐项增加资源，避免一个复杂模拟器给出数字却无法解释其来源。

推理微批也受 KV 与输出长度影响。把请求分成更多微批可能提高流水填充，却增加调度与通信次数，改变每请求首 token 和间隔。训练流水线常见的微批公式不能不加条件套到自回归在线服务。你应保留每请求时间线，同时观察整体设备利用，避免只优化吞吐而让交互体验变差。

PP 的释放责任同样按层分布。请求结束时，每个持有其层 KV 的 stage 都需要收到终态或清理通知；只在最后 stage 删除请求，不会自动释放前面 stage 的缓存。某个 stage 故障后，其他 stage 的在途激活与页也需要按组清理。多卡增加的不只是容量，还增加了分布式生命周期。
## 7. 第七课：DP 是请求分流，容量边界在副本内部

推理中的常规 Data Parallel 复制模型，给不同副本分配不同请求。它提高整体吞吐和可用性，但单个请求通常只在一个副本内执行。若两个副本各有四十 GiB 空闲，一个请求需要六十 GiB 状态，不能因为总空闲八十就自动运行。需要 TP、PP、CP 或显式状态迁移等其他机制改变单请求资源边界。

负载均衡也不只是轮流分配请求数。一个短 prompt 短输出请求与长上下文长输出请求成本不同；有的副本已经保留用户前缀，另一个没有。路由策略可以考虑队列、预计 token 工作、缓存亲和与容量，但每种估计都有误差。只看当前请求数可能把大量重任务集中到一个副本。

会话亲和可以提高 prefix cache 命中，却可能造成热点；随机分流更均匀，却可能重复计算共享前缀。更复杂路由可以在亲和与负载之间取舍，但不能假定所有副本的 KV 缓存自动共享。模型权重相同，不意味着各副本已经拥有相同的请求历史状态。

某些框架把 DP attention 与 EP MoE 组合，模型并不以最朴素方式整份复制。此时 DP 是配置体系中的一个维度，必须继续看专家组和权重布局。我们先用常规模型副本建立直觉，再读具体组合；不能看到配置字段 DP=2 就断言所有参数恰好复制两遍。

DP 副本之间若共享外部队列，需要明确失败重试。一个请求已生成部分输出后转到另一副本，可能需要重算历史并恢复随机流与停止状态；重复发送完整响应会造成用户可见重复。负载均衡属于服务协议，不只是启动两份进程后放一个轮询指针。

## 8. 第八课：EP 从路由表走到通信与归并

Expert Parallel 将 MoE 专家分到不同 rank。router 为每个 token 选择若干专家和权重，然后把 token 激活发到专家所在设备，专家计算后把结果送回原 token 位置，再按路由权重求和。一次 token 可以被发送到多个专家，但这不等于把所有专家权重复制给该 token 所在设备。

设两 rank、四专家，rank 零拥有专家零和一，rank 一拥有专家二和三。三个 token 的 top-2 路由分别是零/二、二/三、一/三，权重都为二分之一。六条路由记录必须保存 token ID、expert ID、源 rank、目标 rank、权重和恢复位置。通信到达顺序可能改变，归并不能靠数组当前顺序猜原 token。

| token | 专家 | 目标 rank | 权重 | 归并槽 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.5 | 0 |
| 0 | 2 | 1 | 0.5 | 0 |
| 1 | 2 | 1 | 0.5 | 1 |
| 1 | 3 | 1 | 0.5 | 1 |
| 2 | 1 | 0 | 0.5 | 2 |
| 2 | 3 | 1 | 0.5 | 2 |

先统计每目标 rank 的发送条数，交换必要的数量元数据，再分发激活；目标端按专家分组做 GEMM，结果沿逆映射返回。若各目标数量不等，接口需要支持可变 split 或通过 padding 表达，不能假定每 rank 总是收到相同 token 数。all-to-all 是通信模式，具体框架可以使用不同实现与融合路径。

```python
routes = {
    0: [(0,0.5),(2,0.5)],
    1: [(2,0.5),(3,0.5)],
    2: [(1,0.5),(3,0.5)],
}
packets = []
for token, experts in routes.items():
    for expert, weight in experts:
        packets.append({"token":token,"expert":expert,
                        "destination":expert//2,"weight":weight})
received = {0:[],1:[]}
for packet in packets:
    received[packet["destination"]].append(packet)
result = [0.0]*3
# 模拟不同到达顺序；专家函数用 expert+token，便于手算。
for destination in (1,0):
    for packet in reversed(received[destination]):
        value = packet["expert"] + packet["token"]
        result[packet["token"]] += packet["weight"]*value
assert result == [1.0,3.5,4.0]
assert [len(received[r]) for r in (0,1)] == [2,4]
print("routed contributions restored by identity", result)
```

专家负载不均衡会让某些 rank 更慢。上例 rank 一收到四份，rank 零两份，即使权重分配均匀，计算量也不均匀。router 分布、专家容量、丢弃策略和负载均衡机制会影响质量与性能。不能只按专家数量平均分显存后就认为吞吐必然平衡。

归并还要避免权重重复应用。若发送前已经把激活乘权重，专家是非线性函数，一般不能认为等价于输出后乘权重；路由加权位置必须遵循模型定义。教学示例在专家输出后加权，真实实现可能融合操作但必须保持相同数学语义。


### 8.1 从发送计数矩阵看 all-to-all 的契约

前面的 MoE 例子只给出目标 rank，现在增加源 rank。设 token 零和一来自 rank 零，token 二来自 rank 一。rank 零的四条路由中，一条发给 rank 零、三条发给 rank 一；rank 一的两条路由中，一条发给 rank 零、一条发给 rank 一。发送计数矩阵的行代表源、列代表目的，因此是 `[[1,3],[1,1]]`。

目的 rank 零应接收两条，目的 rank 一接收四条。接收计数必须与各源的发送计数转置对应，不能各 rank 只根据自己的 token 数猜接收缓冲区大小。实际可变长 all-to-all 通常先交换数量或拥有事先计算好的 split 元数据，再分配/选择接收空间。数量不匹配可能导致越界、截断或等待。

每条路由还要保存专家内偏移。目标 rank 一有专家二与三，各自接收的 token 数可能不同；为了执行分组 GEMM，需要按专家排序或构造分段边界。计算后再经过逆排列回到路由记录，最后按 token 身份加权合并。这里至少有请求/token 顺序、通信顺序、专家分组顺序三种排列，不能用一个 index 数组含糊表示全部。

padding 到专家容量可以让形状更规则，但无效 token 必须被屏蔽，不能参与输出或负载统计。若采用容量限制丢弃路由，还要明确模型语义与质量影响，不是单纯性能优化。某些推理实现不采用训练时的同类丢弃策略，不能根据 MoE 名字推断行为。

专家权重重分布或冗余专家副本会进一步改变 expert 到 rank 的映射。路由中的 expert ID 表示逻辑专家，不必永远等于物理设备编号；映射更新时，正在执行的批次必须使用一致版本。否则同一批的发送方与接收方对目的位置理解不同，可能出现难以复现的错误。先在 CPU 模拟中给映射加版本，是理解动态专家负载均衡的好起点。
## 9. 第九课：SP 与 CP 的分片轴不同，softmax 不能直接平均

Sequence Parallel 常用于将某些激活沿 token/序列维分片，减少 TP 组内重复的存储与工作，并配合 gather 或 reduce-scatter。Context Parallel 更直接针对长上下文的 attention/KV 分布。名称在不同项目中可能有不同具体含义，因此你应问每 rank 有哪些 query、哪些 key/value、哪些 head，以及在哪一步交换数据。

如果同一个 query 的 keys 分到两个 rank，每边可以计算局部 scores，但不能各自 softmax 后把输出简单平均。局部归一化的分母不同，一边可能拥有很大的最大分数。正确合并需要局部最大值 m、指数和 l、未归一化加权值 o，再在共同最大值下重新缩放与求和。

设分片一分数为零，value 为十；分片二分数为二，value 为二十。各片单元素 softmax 都为一，简单平均输出十五。完整注意力输出为 `(10+exp(2)*20)/(1+exp(2))`，约十八点八一，明显不同。错误来自把分片先归一化后丢掉了相对权重尺度。

```python
import math
import numpy as np

scores = np.array([0.,2.,-1.])
values = np.array([[10.,0.],[20.,2.],[5.,4.]])
def local_stats(s,v):
    m = float(np.max(s))
    weights = np.exp(s-m)
    return m,float(weights.sum()),weights@v
parts = [local_stats(scores[:1],values[:1]),
         local_stats(scores[1:],values[1:])]
m = max(p[0] for p in parts)
denom = sum(math.exp(pm-m)*pl for pm,pl,po in parts)
numer = sum(math.exp(pm-m)*po for pm,pl,po in parts)
merged = numer/denom
p = np.exp(scores-scores.max()); p /= p.sum()
np.testing.assert_allclose(merged,p@values,rtol=1e-12,atol=1e-12)
print("context-sharded statistics merge correctly",merged)
```

这个统计合并与 online softmax 同源，也是 CP 正确性的基础之一。真实算法可能环形传递 KV、交换部分统计或采用其他布局，通信代价取决于 query/key 分片方式。你可以迁移数学不变量，但不能从一个公式推断所有框架都使用相同 NCCL 调用序列。

## 10. 第十课：读真实 Column/Row Parallel Linear

固定 [vLLM linear.py](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/model_executor/layers/linear.py) 中，ColumnParallelLinear 的文档明确用 Y=XA+b 解释输出列分片。forward 先调用 quant_method.apply 做本地乘法，只有 gather_output 为真且 tp_size 大于一时才调用 tensor_model_parallel_all_gather，否则保留分片输出。类名不表示一定发生 gather。

RowParallelLinear 的 input_is_parallel 决定输入是否已经沿最后维切好。若为真，直接使用；否则按 tp_size 切输入并取本 rank 那份。随后只在 rank 零且没有 skip_bias_add 时把 bias 融入本地乘法，避免归约后重复加 bias。只有 reduce_results 为真且 tp_size 大于一时才 all-reduce。

```python
# 固定 RowParallelLinear.forward 的关键节选，不是独立程序。
if self.input_is_parallel:
    input_parallel = input_
else:
    split_input = split_tensor_along_last_dim(
        input_, num_partitions=self.tp_size
    )
    input_parallel = split_input[self.tp_rank].contiguous()
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
output_parallel = self.quant_method.apply(self, input_parallel, bias_)
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

quant_method.apply 表明线性层的存储与算子实现可以由量化方法决定。数学分片语义没有因此消失，但权重加载轴、打包格式和 scale 元数据可能更复杂。读一个 quantized linear 时应同时追踪权重加载与 forward，不能只看调用名字就断言本地 tensor 是普通连续浮点矩阵。

skip_bias_add 与 return_bias 用于让调用者或融合算子处理 bias。若你改写 forward，必须保留返回结构与责任，不可一边在本地加了 bias，又把同一 bias 返回给调用者再加。S03 的资源所有权在这里变成计算责任：每个数学操作也需要清楚知道由谁执行一次。

本章静态核查了固定代码路径，没有在该环境启动真实 TP 模型。你可以先用前面的 NumPy例子模拟这些开关，再到独立开发环境验证小模型。错误的 all-reduce group、错误分片顺序和 bias 重复，往往在很小张量上就能抓到，不需要先占用八卡运行巨型模型。


### 10.1 读取权重加载代码时，检查分片是否与 forward 对应

forward 正确还不够，加载到每 rank 的权重也必须是正确分片。数学 A 按输出列切，存储 weight 若是转置的输出行布局，加载器实际切的是存储第一维。若按数学图直接对物理张量第二维切，可能得到 shape 不符，或者在方阵情况下形状恰好相同却内容错误。方阵尤其容易掩盖切错轴。

使用非方阵和可辨识权重是最有效的初始测试。令权重元素由行号与列号编码，例如十乘行号加列号，打印每 rank 子矩阵，逐项确认其逻辑范围。再用 one-hot 输入选择某一行，检查输出是否对应预期权重。随机矩阵 allclose 能发现错误，one-hot 反例则更容易解释错误发生在哪个轴。

量化权重还带有 scale、分组、零点或打包后的位布局。分片边界必须与这些元数据一致，可能受到 group size 与 kernel 支持条件限制。不能把一个 packed byte 数组随意按长度平均切成 p 份，就认为每 rank 得到了相同逻辑比例的权重。需要追加载器、量化方法和 kernel 输入契约三者的连接。

checkpoint 可以原本未分片，由加载器按 rank 取片；也可以已经保存为某种分片格式。重复切片会丢失数据，错误拼接会改变顺序。报告中记录 checkpoint 格式与加载路径，避免把“下载的模型文件相同”当成每 rank 实际权重相同的证据。模型结构、参数版本与分片规则共同决定计算身份。

验证可以分三步：先还原所有分片并与完整权重比较，再比较单层输出，最后比较完整模型 logits。第一步抓加载错误，第二步抓通信与 bias，第三步抓跨层分布衔接。只看最终生成文本相同不够，因为 argmax 可能暂时对小误差不敏感，很多层级问题尚未显现。
## 11. 第十一课：ring 模型与真实拓扑的边界

理想 ring all-reduce 可分为 reduce-scatter 与 all-gather 两阶段，每阶段大约有 p减一 个通信步。若每 rank 的输入张量为 N 字节，每步传 N/p，单 rank 总发送量约 `2*(p-1)/p*N`。延迟项约为 `2*(p-1)*α`，带宽项为发送量除有效带宽 β。它是特定算法的分析模型，不是 NCCL 永远只使用 ring 的声明。

N 为三十二 MiB、p 为八时，每 rank 总发送量五十六 MiB。若有效带宽五十 GB/s，单带宽项约一点一七四毫秒，再加十四个启动延迟。MiB 使用二的二十次方字节，GB/s 使用十的九次方字节每秒；若把 Gbit/s 当 GB/s，会差八倍。单位必须在变量名和报告中写清楚。

```python
from labs.serving.analysis import ring_allreduce

r = ring_allreduce(32*1024*1024,8,50.0,latency_us=5.0)
assert r["bytes_sent_per_rank"] == 56*1024*1024
assert abs(r["ideal_ms"]-(0.07+56*1024*1024/50e9*1000)) < 1e-12
assert ring_allreduce(1024,1,50.0)["ideal_ms"] == 0
print(r)
```

真实拓扑包括 GPU 间互联、PCIe 交换结构、CPU NUMA、网卡与跨节点路径。不同 rank pair 可能经过不同链路，某条瓶颈会限制通信。设备型号相同也不保证互联相同；H100 的具体形态、节点结构和网络配置都应记录。不能只抄宣传峰值带宽来预测应用延迟。

小消息常受启动与同步延迟影响，大消息更可能接近带宽限制。decode 每步频繁通信时，小消息延迟可能比大块带宽更重要。prefill 大矩阵与大激活又有不同平衡。因此测通信不能只选一个巨大数组跑出高 GB/s，再推断短 decode 也同样高效。

计算与通信重叠需要真实独立工作与正确依赖。把调用标为 async 只改变主机等待方式，不自动缩短临界路径；如果下一个算子立刻需要归约结果，仍然必须等。S08 的时间线应区分提交、设备执行与完成事件，避免把 Python 调用很快误认为数据已经传完。


### 11.1 多个进程组与错误传播为什么容易造成挂起

设一个任务同时有 TP 组和 PP 通信。如果某 rank 在 TP all-reduce 前因输入校验失败退出，而其他 TP rank 已经进入集合操作，剩余进程可能一直等待。异常不再只属于一个 Python 调用栈，而会影响整个通信协议。应设置明确超时、记录最早错误，并让组内其他成员按框架机制终止或恢复。

多个 group 的创建也需要一致顺序。若 rank 零先建立组甲再组乙，rank 一反过来，底层通信器身份可能无法按预期配对。官方 PyTorch 文档要求进程组使用遵守一致顺序，特别是 NCCL 多组并发时。不要把 group 初始化藏在只在部分 rank 执行的任意条件分支中，除非该 API 明确支持并且调用集合正确。

退出时也应显式清理进程组与子进程，避免残留任务继续占用设备。清理范围仅限本实验创建的进程和资源，不能用广泛 kill 或删除共享目录来“解决 NCCL 挂起”。在公司服务器上，先保存本任务 PID、端口和日志，才能精确停止自己的实验而不影响其他任务。

日志中保留 collective 序号比只打印开始/结束文本更有用。每条记录包含组成员、rank、操作类型、张量元素数、dtype 与序号。比较所有 rank 的最后一致序号，往往能快速找到某个条件分支漏调用。若只看到“卡住了”，就很难区分等待数据、等待组初始化还是某个 rank 已经退出。

不要一遇到通信问题就修改大量 NCCL 环境变量。先证明设备映射、group、调用顺序与 tensor 合约正确，再研究网络接口和拓扑。环境变量可能改变传输路径或算法，让问题暂时消失，但若根因是调用不匹配，调参只会增加不确定性。每次只改一个条件并记录其效果。


### 11.2 用一个简化延迟模型判断何时值得增加 TP

假设某层单卡计算需要八毫秒，理想均匀分片后两卡各需四毫秒，但本层新增通信和同步合计一点五毫秒，那么简化总时间为五点五毫秒，优于八毫秒。若单卡原本只需一毫秒，两卡理想计算零点五毫秒，通信仍一点五毫秒，总计两毫秒反而更慢。这个算例说明固定开销对小工作负载影响更大。

真实计算并不一定严格除以卡数。矩阵变小可能降低算子利用率，某些权重或 KV 复制部分也不会缩小；通信又可能随消息尺寸与拓扑变化。因此这个模型只能帮助提出假设，不能代替实测。把每项写成函数，例如计算时间依赖 batch 与分片宽度，通信时间依赖字节和 group 大小，再用短实验校准。

容量目标与延迟目标有时冲突。单卡八十 GiB 装不下某模型时，即使两卡单请求延迟稍高，分片仍然使任务成为可能。若模型本来很小，使用两个 DP 副本可能比 TP=2 更适合提高整体请求吞吐，但每个副本的 KV 容量和缓存亲和又不同。选择来自具体目标与工作负载，不存在一个对所有请求都最优的并行缩写。

报告因此至少给出单请求延迟、目标并发下吞吐、每 rank 峰值内存和 SLO 达标率。一个配置可能在其中两项胜出、另外两项退步；把取舍完整呈现，才是工程判断。不要只选择最能证明“八卡很强”的一个数字，而忽略用户实际等待与共享资源成本。

## 12. 第十二课：真正启动两个 CPU 进程，把通信身份跑通

前面 NumPy 程序在一个进程内模拟多个 rank，适合验证数学。现在使用 [collective_cpu.py](../../labs/serving/collective_cpu.py) 真正启动独立子进程，通过队列发送贡献，由父进程求和并把结果广播回各 rank。它是集中协调式 all-reduce 语义练习，不使用 NCCL，也不模拟 ring 的性能。进程数限定为一到四，等待有超时，finally 只清理本实验创建的子进程。

```bash
python -m labs.serving.collective_cpu --ranks 2
python -m labs.serving.collective_cpu --ranks 4
```

两进程的输入分别是一、十与二、二十，所以每个 rank 应收到三、三十。四进程则各贡献 rank 加一及其十倍，结果为十、一百，每个 rank 都相同。输出 kind 明确标记 CPU_coordinator_semantics_not_NCCL_performance，避免有人把启动时间或队列传输时间拿去比较 GPU 通信。

程序使用 spawn 创建进程，每个子进程从模块入口导入所需函数，而不是共享父进程正在执行的全部 Python 栈。`if __name__ == "__main__"` 保护避免子进程导入时再次无穷创建子进程。请通过模块命令运行，不要把创建进程的主体无保护地粘贴到交互解释器。这个结构也为 Windows 上的 CPU 学习保留了兼容的启动方式，但当前执行证据来自本课程 Linux 环境，不冒称已在你的 Win11 上测试。

_worker 收到 rank、自己的整数向量、公共 incoming 队列、专属 outgoing 队列和 timeout。它先把三元组 `("contribution",rank,values)` 放入公共队列，然后等待自己的结果通道，最后把收到的结果作为 ack 回报。专属结果通道避免某个 rank 抢走另一个 rank 的消息；公共输入通道则必须在消息里显式携带 rank。

父进程先收齐所有 contribution，检查消息类型、rank 范围和重复身份，再按元素求和。只有收齐以后才发送结果，因此贡献阶段与确认阶段不会被自己的广播打乱。随后收齐每个 rank 的 result，验证它们都与同一个 total 相同。rank 的消息到达顺序不重要，字典按身份保存，最后按 rank 顺序输出。

注意这里把贡献向量限制为普通整数，是为了让结果可以精确相等，先专注通信身份。若换成浮点，需要考虑归约顺序与容差；若换成不同 shape，需要先设计可变长协议，不能靠 zip 默默截断。程序在进入子进程之前校验相同非空长度，避免一个 rank 发送少一项而父进程悄悄得到短结果。

超时不是性能阈值，而是实验退出边界。某 rank 不回报时，父进程抛出明确错误，finally 检查仅由本次启动的 process 对象，仍活着就终止并 join。这个清理不会查找或结束机器上其他 Python 任务，也不会占用 GPU。你可以在公司服务器 CPU 上运行，不需要为理解 group 与 rank 先申请多卡。

不要从这个集中协调实现推断真实 all-reduce 会把所有张量先搬到 CPU。这里父进程是教学协调者，NCCL 的设备通信可以采用分布式算法与直接设备路径。二者共享的是“各 rank 对相同元素进行归约并获得一致结果”的契约，执行效率与底层机制完全不同。把语义和实现分开，是从参考程序进入高性能系统的必要习惯。

### 12.1 三个有意义的故障注入

第一项故障是在调用 coordinator_allreduce 前传入不同长度的两行。预期在父进程前置校验失败，不创建任何子进程。它验证 shape 合约，而不是通信超时。完整修复应明确支持等长输入，或重新设计变长操作；不能把短行补零后假装用户原来就要求这个数学问题。

第二项故障是在你自己的练习副本中，让一个 worker 把 rank 改成已有 rank。父进程应在重复身份检查失败，然后清理本实验子进程。这个测试说明“收到两条消息”不等于“两个不同 rank 都参加了”。集合操作需要完整参与者集合，消息数量与身份集合都要正确。

第三项故障是让一个 worker 在发送 contribution 后不再接收结果。父进程可能已经求出 total，但最终确认阶段或 join 会超时，不能报告所有 rank 成功。这个场景区分“协调者算出结果”与“所有参与者完成协议”。真实分布式服务同样不能只看 root 的日志就宣布整个 group 健康。

故障注入只在复制出的练习文件或测试进程内进行，保留正常参考。记录预期失败位置，确认不是 import 错误或无关异常，再恢复并重跑两个与四个 rank。每种故障都对应一个契约：形状、身份、完成。它们比盲目增加随机输入数量更能帮助你理解多进程为什么会挂起。

### 12.2 从 CPU 进程过渡到两张 GPU 的受限实验

真正测试 NCCL 前，先确认当前任务被分配的设备列表、每 rank 到设备的映射以及其他任务不会被影响。实验默认只用两张已分配设备，张量从很小的一维整数或浮点向量开始，执行固定少量 collective，设置进程组超时，结束后显式销毁组。不要在第一轮实验加载大模型，否则通信错误会与权重加载和显存不足混在一起。

第一项 GPU 验证只关心数值：rank 零填一，rank 一填二，sum all-reduce 后每个元素为三；all-gather 后按 rank 顺序拼接；reduce-scatter 用可手算的完整输入验证每 rank 得到哪一段。每个操作都记录 group、count、dtype 与输出 shape，建立与 CPU 参考一致的结果。这里没有自动执行任何 GPU 命令，本章仅给受限实验契约。

第二项才测时间。分别观察小消息与较大消息，进行固定预热，再收集有限次数的设备时间，说明同步与计时边界。不能只在异步 API 前后用主机计时就宣布传输完成；应使用框架支持的完成/同步机制。总实验设置分钟级上限，发现错误立即停止本任务，避免一组挂起进程持续占卡。

第三项接入小模型 TP，对比单卡与两卡的 logits 或固定输入输出，记录容差与采样规则，再比较端到端延迟。若模型在两卡上更慢，先分析矩阵形状和通信，不必强行寻找“应该更快”的数字。实验目标是理解分片代价与容量收益，不是证明卡数增加必然胜利。

只有需要研究拓扑、TP 扩展或 EP 路由时才扩大到八卡。八卡实验仍应有固定请求数、最大输出长度、总时限和清理步骤，先确认两卡路径正确再扩大。服务器 GPU 使用自由不等于实验无需边界；设计清楚的小实验通常更容易获得可解释证据，也更便于与你公司的日常任务共存。

### 12.3 把并行配置写成一份每 rank 账本

账本第一栏是权重：哪些层、哪些输入/输出维、哪些专家、是否复制 embedding 或 norm。第二栏是状态：哪些请求的哪些层、哪些 KV head、哪些序列范围、缓存 dtype 与 page layout。第三栏是激活：当前持有完整副本、分片还是部分和。第四栏是通信：对哪个 group 做何种操作，输入输出 shape 怎样变化。

再加一栏生命周期：什么时候创建、谁最后使用、什么事件允许释放。PP 激活发送后源缓冲区何时可重用，EP 路由包在回传前需要保留哪些 inverse mapping，TP 归约结果在下一层开始前需要哪种完成依赖，都应写在这里。显存估算与错误排查使用的是同一份所有权信息，不是两套互不相关的知识。

完成账本后，用一个只有两个 token、两个 rank 的样本逐步填数字。若某个字段只能写“框架自动处理”，继续找调用者和数据结构。你不需要手写 NCCL 内核才能成为框架开发者，但应知道自己调用的 collective 在数学上弥补了哪一部分缺失，以及调用前后谁拥有完整结果。

这份账本也能帮助阅读新技术。看到新的上下文并行、专家调度或通信融合，不必先背新的缩写，先问它改变了哪一栏：权重放置、状态放置、激活分布、通信顺序还是生命周期。新方法的收益与限制通常就藏在这些具体变化中。

## 13. 第十三课：故障定位、分级题与完整工程路线

**故障一：多卡结果总是多一个 bias。** 用本章四维手算样本比较完整乘法与分片乘法，检查每 rank 是否都加完整 bias。若误差恰好是 `(p-1)*bias`，这是很强的线索。修复后同时测试 bias 为零和非零；只用零 bias 会让错误实现也通过。

**故障二：某个 collective 永远不返回。** 记录每 rank 的 group、操作序号、操作类型、shape、dtype 和进入时间，找最后一个一致点。可能某 rank 因异常提前退出、分支条件不同、group 建立顺序不同或调用顺序不同。先用小张量和有限超时复现，不要重复启动更多进程让现场更混乱。

**故障三：MoE 输出顺序乱了但数值范围正常。** 用每个 token 可辨识的激活和简单专家函数，故意重排通信包，检查是否按原 token ID 归并。错误通常在 permutation/inverse permutation 或路由权重对应关系，不一定在专家 GEMM。只比较整体均值会漏掉这种排列错误。

**故障四：TP 增大后更慢。** 先确认输出与工作量一致，再分解 GEMM、通信和同步时间。每 rank 矩阵变小、通信次数增加、KV 复制或拓扑变差都可能造成退化。它不一定是安装错误，也不意味着 TP 没用；若单卡装不下，容量收益仍可能必要，目标应明确区分单请求延迟与整体吞吐。

**练习一，矩阵分片。** X 为 `[1,2,3,4]`，A 按本章定义，完整输出 `[12,5]`。列分片后拼接，行分片后求和。答案必须写出每个部分的 shape 与数值，不能只说“最后做通信”。通信类型由数学缺少的是拼接片段还是部分和决定。

**练习二，bias。** 加 `[1,-1]` 后正确结果 `[13,4]`；两 rank 各加再归约得到 `[14,3]`。修复是在归约后加一次，或仅一个 rank 归约前加。若 skip_bias_add 打开，还要确认调用者确实承担后续加法，没有遗漏或重复。

**练习三，MLP 中间 gather。** 第一层列分片得到 h/2 激活，逐元素非线性本地执行，第二层行分片直接消费，再 all-reduce。无需中间 gather。若中间算子需要全隐藏维统计，例如某种归一化，就必须额外处理，不能照搬逐元素局部性。

**练习四，GQA 容量。** Hq 八、Hkv 二、TP 四，不能按每 rank 半个完整 KV head 的简单头分片解释。必须查实际复制或其他分片规则。容量估算若盲目除四，可能低估每 rank KV。答案应保留“取决于实现”，并指出需要核对的具体布局字段。

**练习五，PP 时间。** 三 stage、四微批、各阶段一单位，理想完成六单位；有效工作十二，设备总时间十八，理想利用三分之二。若阶段不等时或存在通信，公式不再直接给真实时间。单请求 decode 还受下一 token 反馈依赖约束。

**练习六，DP 显存。** 两副本各四十 GiB 空闲，六十 GiB 请求不能在常规独立 DP 内自动运行。需要改变单请求分片或状态放置，而不是把总和写成八十后忽略副本边界。若服务支持迁移，也必须明确状态如何转移与何时可用。

**练习七，MoE 路由。** 上文六条路由中 rank 零收两份、rank 一收四份；专家函数 expert+token、权重二分之一，输出为一、三点五、四。即使通信到达顺序反转，按 token 身份归并仍相同。按包到达顺序拼接会错误地改变结果长度和位置。

**练习八，CP 合并。** 两片各自 softmax 后平均一般错误，因为丢失相对归一化尺度。答案应保存局部最大值、指数和与未归一化加权值，以共同最大值缩放后合并。可以用零和二两个分数手算反例，证明十五不等于完整结果约十八点八一。

**练习九，all-reduce 与 gather。** rank 零有一、二，rank 一有十、二十，sum all-reduce 得十一、二十二；all-gather 得一、二、十、二十。两者 shape 与数学意义不同。reduce-scatter 则先归约后各留一段，不能直接把输入切片当成已归约结果。

**练习十，ring 字节。** 三十二 MiB、八 rank，每 rank 五十六 MiB；这是单 rank 发送量，不是整个网络总流量。若把所有 rank 发送相加还要乘八，但共享链路与双向传输使总流量不能直接除某个单链路峰值预测时间。

**练习十一，group 错误。** DP 副本零与一处理不同请求，不应把它们的 row-parallel 部分和相加。即使 shape 一致，也属于不同数学问题。测试应让副本输入明显不同，错误全局归约才能暴露；所有副本输入相同可能让问题被误认为只是缩放差异。

**练习十二，源码开关。** ColumnParallelLinear 只有 gather_output 且 TP 大于一时 gather；RowParallelLinear 只有 reduce_results 且 TP 大于一时 reduce。答案必须看实际 forward，不根据类名推断每次通信。input_is_parallel 与 skip_bias_add 也改变调用者责任。

**练习十三，异步完成。** 主机拿到异步句柄不等于设备结果可读。下一步依赖结果时必须等待相应完成或建立 stream 依赖；源/目标缓冲区在使用结束前不能复用。仅调用 async 参数不能证明通信与计算真的重叠，更不能证明端到端更快。

**练习十四，首次实验。** 先 CPU 数组对照，再本地两个进程验证通信语义，再申请两张设备跑小张量和小模型，最后才研究八卡。每步记录 shape、dtype、group、映射、误差、超时和清理。第一次实验不要同时引入大模型、跨节点、量化与 EP，否则故障维度太多。

项目交付包括每 rank 权重/KV/激活清单、进程组映射表、每层通信序列、NumPy 数值参考、故障注入与受限硬件记录。你的个人 PC 完成数学与 CPU 进程练习；服务器 CPU 可承担源码与协议测试；真实通信只在需要时短时使用分配到的 GPU。掌握后进入 [S11 Prefill/Decode 分离](11-pd-disaggregation.md)，研究同一请求跨执行池交接状态。
