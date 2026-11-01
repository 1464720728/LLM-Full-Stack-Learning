# K18 从集合通信到算子作品集：手算、协议模拟、真实两卡与完整验收

前面的算子主要在一张设备内读写张量。本章增加一个边界：一份逻辑计算可能由多个进程、多个 GPU 共同完成，某个进程的结果需要依赖其他进程的输入。通信算子的任务不仅是搬字节，还要建立清楚的数据分片、归约与结果归属关系。

本章先修是 F11 的并发、K06 的 GEMM、K13 的异步与同步，以及 S10 的并行概念。不会要求你一开始占用八卡。多数推导与完整参考模拟都在 CPU 完成；真实实验先使用明确分配的两卡短窗口，检验接口和结果，再决定是否做更大拓扑实验。

本章同时是算子卷综合项目。你会把数学参考、实现、边界测试、源码路径与环境证据整理成别人能复现的作品。完成标准是能够从输入合同解释到数值与资源行为，不是给文件取一个叫“高性能 Kernel”的名字。所有 Python 教学块按本章出现顺序在同一个会话中执行；真实 NCCL 命令另列，必须具备设备后显式启动。

## 第一课：进程、rank、设备和通信组分别是谁

一个进程有自己的 Python 解释器、变量、地址空间与运行进度。两个进程里都叫 x 的变量，并不是同一个数组。即使它们运行同一份脚本，也需要通过明确通信才能交换数据。把多个进程启动起来，只建立了多个执行者，并没有自动让它们共享完整模型状态。

rank 是某个通信组里的逻辑编号。一个四进程组通常有 rank 零到三，编号用来规定角色、拼接次序或分片归属。rank 不是某块 GPU 在机器上的永久物理名称；同一个进程在不同子组中，也可能具有不同组内身份。阅读框架时必须看清参数属于哪个组，不能把所有 rank 字段当作同一个命名空间。

local rank 表示当前节点内的进程编号，global rank 表示整个作业的编号。单机实验中它们可能数值相同，到了多节点就不一定相同。设备选择还受可见设备列表影响：如果进程只能看见两张被分配的 GPU，本进程里的 cuda:0 是可见列表的第一张，不一定是整台机器物理编号零。

因此一份有用的启动记录至少包括节点身份的可公开替代标识、world size、global/local rank、可见设备列表与选中的设备。记录这些关系是为了诊断映射；公开报告不需要包含公司主机名或内部地址。不能只写“用 H100 跑了”，因为这句话没有说明实际启动多少参与者以及它们如何对应资源。

通信组规定哪些参与者一起执行一个 collective。某个模型可能有数据并行组、张量并行组和专家并行组，各组成员和用途不同。对错误的组做一次数值合法的 AllReduce，也可能得到完全错误的模型结果。通信 API 执行成功，只说明某次操作完成，不保证你选择了正确数学分解。

## 第二课：先区分六种结果关系

### 1. AllReduce：逐位置合并，完整结果发给每个参与者

设两个 rank 的输入分别是 `[1,2]` 与 `[10,20]`，使用求和归约。结果是每个 rank 都得到 `[11,22]`。它不是把两条向量首尾相连，而是按相同位置把不同 rank 的值相加。输出长度与每个 rank 的输入长度相同。

Reduce 与它的区别是完整归约结果只由指定 root 接收；其他 rank 的接收缓冲不能随意被当成同样有效的结果。Broadcast 则把 root 已有的一份输入复制给所有参与者，本身不对各 rank 输入求和。把 Reduce 和 Broadcast 连起来，可以建立与 AllReduce 相同的结果关系，但实际执行成本和实现路径还需要另外分析。

### 2. AllGather：保留各份数据并按身份拼接

对于同样输入，AllGather 得到 `[1,2,10,20]`，每个 rank 都收到这个拼接结果。位置不再表示“两个 rank 的同一元素相加”，而是表示原始分片来自谁、位于该分片哪个位置。rank 顺序参与数据布局，所以不能随意改变编号却仍按旧方式解释结果。

教学中先假设每个 rank 提供相同长度片段。若真实任务长度不同，需要使用适合该 API 的计数、padding 或变长通信协议；不能把长度不一致的缓冲直接交给要求匹配计数的调用，然后等待库自行猜测拼接方式。

### 3. ReduceScatter：归约后只领取自己负责的片段

设两个 rank 各有四元素输入 `[1,2,3,4]` 与 `[10,20,30,40]`。先按位置相加得到 `[11,22,33,44]`，再按组内 rank 顺序分成两片：rank 零得到 `[11,22]`，rank 一得到 `[33,44]`。它把数学归约与结果分片结合起来。

后接 AllGather 可以重建完整求和结果。这个分解很适合理解某些 AllReduce 算法，但不能据此认为所有实现一定以两个公开 API 调用组成。库可以融合、分块或采用其他协议；先分清结果关系，再观察真实执行策略。

### 4. All-to-All：每个发送者都为每个接收者准备不同片段

假设两个 rank，rank 零准备 `[给0的片段, 给1的片段]`，rank 一也准备两片。接收 rank 零获得两个发送者各自“给0”的片段，接收 rank 一获得两个发送者各自“给1”的片段。它改变的是所有者分布，不会自动把不同来源相加。

这种关系常出现在专家计算的数据分发：按目标专家或设备重新排列 token，再把各片送到负责设备。但 MoE 路由还需要 token 身份、专家身份、路由权重和回收顺序，All-to-All 只负责其中一次交换。不能把通信完成当成完整 dispatch/combine 逻辑正确。

下面的 CPU 代码只展示结果语义，没有进程或网络通信。把这些数组操作和后面的真实设备实验分开记录。

```python
import numpy as np

inputs = np.array([[1,2,3,4],[10,20,30,40]], dtype=np.float64)
allreduce_result = inputs.sum(axis=0)
allgather_result = np.concatenate([row for row in inputs])
reduce_scatter_result = np.split(allreduce_result, 2)
np.testing.assert_array_equal(allreduce_result, [11,22,33,44])
np.testing.assert_array_equal(allgather_result, [1,2,3,4,10,20,30,40])
np.testing.assert_array_equal(reduce_scatter_result[0], [11,22])
np.testing.assert_array_equal(reduce_scatter_result[1], [33,44])

# First axis: sender; second: destination; third: elements in one message.
send = np.array([[[1,2],[3,4]], [[10,20],[30,40]]])
received = [np.concatenate([send[source,destination] for source in range(2)])
            for destination in range(2)]
np.testing.assert_array_equal(received[0], [1,2,10,20])
np.testing.assert_array_equal(received[1], [3,4,30,40])
```

本节接口语义与匹配要求可对照 [NCCL collective 官方说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)，本次核查于 2026-11-01。这里的数值例子与模拟代码是独立教学构造；实际可用接口和后端行为仍需核对你安装的版本。

## 第三课：把张量并行的数学需求映射到通信

### 1. 沿输入维分解矩阵乘法

设 Y=XW，X shape 为 `[M,K]`，W 为 `[K,N]`。把 K 维拆成两段，得到 X0/X1 和 W0/W1，那么 Y 等于 `X0@W0 + X1@W1`。两个 rank 各算一个完整 `[M,N]` 部分结果，最后需要按位置相加；这正是求和归约的语义。

如果后续每个 rank 都需要完整 Y，可以使用 AllReduce 这样的结果关系；如果后续只需要 Y 的不同分片，可以考虑 ReduceScatter，前提是分片方式与下一算子合同相符。选择通信前应先写出张量等式和输出所有权，而不是先决定“这里一般用某个 API”。

```python
x = np.array([[1.0,2.0,3.0,4.0]])
w = np.array([[1.0,0.0],[0.0,1.0],[2.0,0.0],[0.0,2.0]])
partial0 = x[:,:2] @ w[:2]
partial1 = x[:,2:] @ w[2:]
np.testing.assert_array_equal(partial0, [[1.0,2.0]])
np.testing.assert_array_equal(partial1, [[6.0,8.0]])
np.testing.assert_array_equal(partial0 + partial1, x @ w)
```

### 2. 沿输出维分解时，结果不是求和

若按 N 维拆 W，每个 rank 算的是输出不同列。它们的结果应按列拼接，而不是对应位置相加。两个局部结果形状可能相同，但语义不同；仅看 shape 无法决定用归约还是 gather。

实际网络层可以让后续算子继续消费分片，避免每层都立刻拼成完整结果。为了做到这一点，下一层必须知道输入的分布方式，并选择相容的矩阵分解。通信优化常来自重新安排分片与计算边界，而不只是把同一个 AllReduce 调得更快。

### 3. 非线性通常不能随意穿过求和

如果两个 rank 的部分值为负一和二，先求和再 ReLU 得到一；分别 ReLU 后求和得到二。这说明不能任意把激活移动到归约前。你需要检查算子是否满足所需代数关系，而不是认为“都在算同一个网络，所以顺序差一点没事”。

```python
before_reduce = np.maximum(-1.0, 0.0) + np.maximum(2.0, 0.0)
after_reduce = np.maximum(-1.0 + 2.0, 0.0)
assert before_reduce == 2.0 and after_reduce == 1.0
```

这种小反例也适用于归一化、softmax 和路由权重。通信与计算融合必须保留完整数学语义。你在 K07 学到在线 softmax 需要 m/l/u，而不是局部输出平均；分布式环境中的部分归约同样不能丢掉合并所需信息。

## 第四课：手工走一遍三 rank 的 ring

### 1. 环只规定邻接，不规定最终数值含义

把三个 rank 排成环，零发给一，一发给二，二发给零。每个 rank 原始向量分成三个等长块，编号零、一、二。分块让消息可以逐步传递和累加，避免每一步都把完整向量绕一圈。这里采用一套明确教学调度，其他等价实现可能用不同块编号约定。

ReduceScatter 第 s 步，rank r 发送块 `(r-s) mod p` 给右邻居，同时从左邻居收到块 `(r-1-s) mod p` 并加到自己保存的同编号块上。每步必须先从旧状态取得所有发送数据，再统一更新接收状态；在单进程模拟中若边遍历 rank 边直接读已更新数组，就会把同一步未来信息提前使用。

三 rank 的第一步：零送块零给一，一送块一给二，二送块二给零。接收者把对方贡献加到自己的对应块。第二步发送刚形成的部分和，让第三个参与者加上自己的贡献。结束后，rank 零拥有完整块一，rank 一拥有完整块二，rank 二拥有完整块零。这个所有权偏移来自本节编号约定，不是归约语义本身要求固定偏移。

### 2. 接着把完整归约块传播给所有人

AllGather 阶段，每个 rank 从自己拥有的完整块开始，继续沿环发送；接收者保存收到的完整块，不再相加。经过 p-1 步，所有 rank 都得到全部归约块。若第二阶段还执行加法，就会重复计入已完成的贡献；两阶段虽然都传块，接收动作的数学职责不同。

选择三个 rank 而不是只测两个，是为了迫使算法处理不止一次传播。两 rank 的退化例子只有一步，许多错误块编号或更新时机可能恰好看不出来。测试还应包括一个 rank 的空通信边界和四个 rank 的更长环。

下面给出完整可运行模拟。输入 shape 为 `[rank, elements]`，元素数能被 rank 数整除。trace 记录每步发送的块号，便于与手算对照；它不模拟真实带宽、网络拥塞或线程调度。

```python
def ring_sum_reference(rank_inputs):
    x = np.asarray(rank_inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("inputs must be a nonempty [rank, elements] matrix")
    p, elements = x.shape
    if elements % p or not np.isfinite(x).all():
        raise ValueError("finite input and equal chunks required")
    width = elements // p
    chunks = x.reshape(p,p,width).copy()
    trace = []
    for step in range(p-1):
        old = chunks.copy()
        sent = [(rank-step) % p for rank in range(p)]
        for rank in range(p):
            previous = (rank-1) % p
            incoming = (rank-1-step) % p
            chunks[rank,incoming] += old[previous,incoming]
        trace.append({"phase":"reduce_scatter", "step":step, "sent_chunks":sent})
    for step in range(p-1):
        old = chunks.copy()
        sent = [(rank+1-step) % p for rank in range(p)]
        for rank in range(p):
            previous = (rank-1) % p
            incoming = (rank-step) % p
            chunks[rank,incoming] = old[previous,incoming]
        trace.append({"phase":"all_gather", "step":step, "sent_chunks":sent})
    return chunks.reshape(p,elements), trace

ring_inputs = np.array([[1,2,3],[10,20,30],[100,200,300]], dtype=np.float64)
ring_outputs, ring_trace = ring_sum_reference(ring_inputs)
np.testing.assert_array_equal(ring_outputs, np.tile([111,222,333], (3,1)))
assert ring_trace[0]["sent_chunks"] == [0,1,2]
assert ring_trace[1]["sent_chunks"] == [2,0,1]
assert [row["phase"] for row in ring_trace] == ["reduce_scatter"]*2 + ["all_gather"]*2
```

### 3. 为什么模拟要保留旧快照

模拟中的 old=chunks.copy 用来表达一个同步通信步骤开始时可发送的状态。如果去掉 copy，让 rank 零先更新后，rank 一又从这个已更新状态读取，就可能在一个逻辑步骤里穿过多条链路。这样程序也许很快得到某个和，却不再对应我们要验证的协议。

真实通信不靠每步复制整个全局数组来实现同步；每个进程只拥有自己的缓冲，发送、接收与设备事件形成实际依赖。CPU 模拟的快照是为了避免 Python 循环引入不属于协议的顺序。教学模型的执行方法与真实实现可以不同，但要明确它保持的是哪一个抽象。

## 第五课：从每步消息量推导成本模型

### 1. 先规定符号与单位

令 p 为 rank 数，S 为每个 rank 原始输入的字节数，alpha 为一个通信步骤的固定延迟，B 为有效传输带宽，单位分别为无量纲、字节、秒、字节每秒。这里 B 表示带宽，不是 batch，也不是显存总量。写清单位可以避免把 GB/s、Gb/s、MiB 和元素数直接混在同一个除法里。

前面的环分成两个阶段，每阶段 p-1 步，每步发送 S/p 字节。于是每 rank 的总发送量为 `2*(p-1)*S/p`。用发送量除以 B 得到带宽项，再加步骤数乘 alpha，得到近似 `T=2*(p-1)*alpha + 2*(p-1)/p*S/B`。模型假设每步传输行为规则、有效带宽相同，没有显式建模拥塞和协议选择。

这个公式不是在说每张卡向每张卡都发送一份完整 S，也不是把所有 rank 的发送字节简单相加后再除以单链路带宽。多个链路可以并行工作，性能模型要围绕关键路径定义。计算“全系统总传输字节”与计算“单 rank 环阶段耗时”是不同问题。

### 2. 完整手算一组数值

设 S 为六十四 MiB，即 `64*2**20=67108864` 字节，alpha 为五微秒，B 为每秒二百五十亿字节。p=2 时固定项十微秒，传输项为 S/B，合计约二点六九四毫秒。p=8 时固定项七十微秒，传输项为一点七五乘 S/B，合计约四点七六八毫秒。

这组参数是人为设置的模型输入，不是从 H100、PCIe 或某种交换网络测出的值。即使数字看起来像常见硬件量级，也不能给输出贴上具体设备实测标签。使用模型的目的是比较趋势和发现需要测量的量，而不是绕过实验获得“真实性能”。

```python
from labs.kernels.reference import ring_allreduce_seconds

size_bytes = 64 * 2**20
latency_s = 5e-6
bandwidth_bytes_s = 25e9
two = ring_allreduce_seconds(2, size_bytes, latency_s, bandwidth_bytes_s)
eight = ring_allreduce_seconds(8, size_bytes, latency_s, bandwidth_bytes_s)
np.testing.assert_allclose(two, 0.00269435456)
np.testing.assert_allclose(eight, 0.00476762048)
assert ring_allreduce_seconds(1, size_bytes, latency_s, bandwidth_bytes_s) == 0
```

p=1 时模型为零，因为没有跨 rank 传输；真实函数调用仍有检查、调度或返回开销。这个边界说明模型只覆盖它声称覆盖的成本。不要因为函数测得一个非零时间，就认为整个推导被反驳；先判断测量包含哪些额外阶段。

### 3. 小消息与大消息的交界

让固定延迟项等于带宽项，消去公共因子后得到 S=p*alpha*B。上述八 rank 参数下，交界为一百万字节左右。比它小很多时，步骤固定成本可能更突出；比它大很多时，数据传输项可能更突出。这个交界属于简化模型，不是库切换算法的真实阈值。

降低消息次数可能帮助小消息场景，但合并消息也可能让计算等待更久，增加延迟或缓冲占用。拆小消息有利于流水线重叠，也可能增加固定开销。你应把端到端目标放进分析：训练吞吐、单请求延迟、尾延迟和资源利用率不一定在同一配置下最优。

真实拓扑可能包含不同带宽的边、共享链路、节点内外层级以及并发任务。一个环的某条慢边可能影响关键路径，理想 B 不再是一个固定常数。生产库也可能选择树或分层算法。当前模型给你一套可解释的起点，后续要用拓扑与测量补上被省略的条件。

## 第六课：浮点归约为什么不能只要求比特相同

实数加法满足结合律，浮点加法因舍入不一定满足。取很大的数、其相反数和一个小数，先把大数相消再加小数，与先把小数加入大数再相消，可能得到不同结果。归约树、环顺序或消息分块改变累加顺序，因此即使数学定义相同，也可能出现末位或更明显差异。

```python
large = np.float64(1e16)
left = (large + (-large)) + np.float64(1.0)
right = large + ((-large) + np.float64(1.0))
assert left == 1.0 and right == 0.0
```

这个例子用于揭示非结合性，不是说所有训练都会出现这种极端误差。验收应根据 dtype、输入范围、归约长度和参考精度设置阈值。简单整数且能被浮点精确表示的微例子适合做严格等值检查；一般随机浮点则更适合使用有依据的误差界。

平均值还涉及额外语义。sum AllReduce 得到的是和，不会自动知道你想按多少样本平均。若每个 rank 的局部 batch 不同，先取各 rank 局部平均再简单按 rank 数平均，会让样本少的 rank 获得过大权重。正确方法通常需要总和与总样本数等信息，具体以训练目标定义为准。

例如 rank 零只有一个样本，损失为零；rank 一有三个样本，损失都为四。局部均值分别零和四，简单平均为二；全体四样本的均值为三。这个手算说明通信 API 只能执行你要求的运算，无法替你修复错误统计口径。

```python
local_sums = np.array([0.0, 12.0])
local_counts = np.array([1, 3])
wrong_mean = np.mean(local_sums / local_counts)
correct_mean = local_sums.sum() / local_counts.sum()
assert wrong_mean == 2.0 and correct_mean == 3.0
```

对梯度、指标、专家负载和 token 数都应做类似检查：被归约的量是什么单位，分母来自哪里，哪个阶段进行了平均，是否在两个地方重复除以 world size。日志出现看似合理的数字，并不保证统计意义正确。

## 第七课：collective 的调用顺序也是协议

### 1. 参与者必须对同一次操作达成一致

设两个 rank 都先做一轮梯度 AllReduce，再做一次指标 AllGather。如果某个 rank 因局部条件跳过第一轮，直接进入第二轮，其他 rank 就无法按预期配对。通信库不会阅读业务含义，把不同操作自动重新排序。类型、元素数量、参与组和调用次序都属于接口合同。

下面用纯 CPU 描述符检查一个最小协议。它只验证事先给出的操作列表相同，不是真实分布式一致性协议，也不能保证运行时没有退出或设备错误。但它能帮助你在纸上发现条件分支造成的次序差异。

```python
def check_collective_plan(plans):
    if not plans:
        raise ValueError("at least one rank is required")
    expected = list(plans[0])
    for rank, plan in enumerate(plans[1:], 1):
        if list(plan) != expected:
            raise ValueError(f"collective plan mismatch at rank {rank}")
    return len(expected)

plan = [("all_reduce", 16, "float32", "world"),
        ("all_gather", 4, "float32", "world")]
assert check_collective_plan([plan, plan]) == 2
try:
    check_collective_plan([plan, plan[1:]])
except ValueError:
    pass
else:
    raise AssertionError("mismatched order accepted")
```

### 2. 错误退出会改变其他参与者的命运

若 rank 零在参数检查阶段抛异常退出，rank 一已经进入通信等待，后者可能直到超时才知道作业无法继续。单个进程的 try/finally 有助于释放它自己创建的资源，但不能承诺修复所有分布式失败。作业管理需要观察所有参与者并终止或重启整个相关组。

排查时记录各 rank 最近完成的阶段，例如初始化、分配、进入第几次 collective、完成、校验和清理。一个进程停在“进入操作三”，另一个只到“分配失败”，已经提供了很强线索。不要第一反应就是增加超时时间；增加等待只会让同一个逻辑错误更晚出现。

### 3. 异步提交和完成要分别解释

某些接口允许异步提交，返回一个工作句柄。返回表示操作已经被安排到相应执行路径，并不意味着任意后续代码都可以无条件读取结果或复用发送缓冲。要依据该 API 的流与完成语义建立等待，再让依赖计算消费数据。

本章真实微实验采用简单阻塞调用并在检查后同步设备，目的是把初次正确性路径做清楚。若以后尝试通信计算重叠，先写出“哪个计算需要哪份通信结果”，再设置事件与等待。删除同步得到较短主机计时，不等于端到端工作真的更快。

API 与后端的具体完成语义可查 [PyTorch distributed 官方文档](https://docs.pytorch.org/docs/stable/distributed.html)，本次核查于 2026-11-01。学习记录应保存实际安装版本；文档的 stable 页面会更新，不能把本章阅读日期当成你机器上的依赖版本。

## 第八课：读懂并运行一个有边界的两卡实验

### 1. 先读入口，明确何时会执行

[nccl_check.py](../../labs/kernels/nccl_check.py) 默认不运行真实通信。只有显式传入 `--run` 才进入依赖与设备检查。它要求恰好两个 rank、两个可见 GPU，并使用 LOCAL_RANK 选择本地可见设备。这个实验针对单机两卡，不能不改协议就拿去当多节点模板。

接着它初始化 NCCL 进程组，并设置六十秒超时。每个 rank 建立十六个浮点数，rank 零填一，rank 一填二。sum AllReduce 后，每个元素都应为三。结果很小，便于人工确认，也避免在第一次通信时混入大模型、长数据加载和复杂内存压力。

程序用 torch.testing.assert_close 比较预期，随后同步当前设备并打印本 rank 的成功信息，finally 销毁本进程创建的通信组。两条成功记录都需要看到；只看 rank 零打印一行，不能推断另一个参与者也正常完成了校验与清理。

### 2. 把资源选择写进命令

先确认实际分配给你的两个设备编号。下面零、一只是示例，必须替换成被允许使用的设备。不要通过占满所有八卡来排查一个两卡微例子，也不要为了方便结束任务而清理其他人的进程。

```bash
python tools/run_experiment.py --seconds 120 --gpu 0,1 -- \
  torchrun --standalone --nproc-per-node=2 -m labs.kernels.nccl_check --run
```

外层工具给当前作业设置最长运行时间并记录结果；它不提供显存硬上限，也不是公司调度器。内层 torchrun 建立两个工作进程，脚本自己的 world size 检查防止意外扩大范围。若环境没有 PyTorch、兼容 CUDA/NCCL 或两张可见 GPU，应先解决相应前提；SKIP 不算执行成功。

个人 PC 只有一张所述显卡时，可以完成全部 CPU 协议与成本实验，但不能完成这个两设备验证。共享 H100 服务器适合用一次短两卡窗口完成它；后面真正研究张量或专家并行拓扑时，再按任务需要扩到更多卡。学习顺序依据问题需要资源，而不是依据机器拥有多少卡。

### 3. 这条命令验证什么、没有验证什么

它验证两个进程能够建立组、在相应设备上完成一项极小求和归约并得到预期值。它没有测量稳定带宽，没有验证大消息、多个通信组、故障恢复或长时间负载，也没有执行模型训练。第一次成功后应记录准确环境和命令，再按一个变量扩展实验。

若要测耗时，应区分初始化、首次调用与稳态重复，说明同步位置、warmup、消息字节、重复次数和聚合统计。不要把整个 torchrun 启动时间除以十六个元素，当成通信带宽；大部分时间可能花在进程与运行时初始化。K14 已给出性能实验方法，本章要求把同一标准应用到多参与者环境。

## 第九课：先建立一个能复现的算子作品集

### 1. 四份核心交付物各自解决什么问题

第一份是合同：输入输出 shape、dtype、布局、mask 含义、设备与是否允许别名。没有合同，别人不知道一个失败输入属于代码错误还是未支持范围。第二份是数学参考：清楚实现定义，可读性比速度重要。第三份是候选实现：可以是分块 CPU、CUDA 或 Triton，但必须说明实际走哪条路径。第四份是验证报告：记录输入、预期性质、误差、环境和未执行范围。

性能报告只有在真实测量后才加入。一个只有理论带宽公式的文件不能叫实测性能报告；一个只跑过语法检查的 GPU 程序不能标成设备通过。作品集的价值在于让另一位工程师判断你已经证明什么、还需要什么环境，而不是只看见一张没有条件的加速比图。

仓库 [capstone.py](../../labs/kernels/capstone.py) 已提供可运行基线：四组形状、两种因果设置、四种 block 大小，共三十二个组合。它比较 dense attention 与 online attention，报告最大误差和实际 Python/NumPy 版本。命令输出的 backend 明确写 CPU NumPy，不伪装 GPU 结果。

```bash
python -m labs.kernels.capstone --output build/kernel-labs/correctness.json
```

```python
from labs.kernels.capstone import correctness_report
portfolio = correctness_report()
assert portfolio["status"] == "passed"
assert portfolio["attention_cases"] == 32
assert portfolio["environment"]["backend"] == "CPU NumPy"
assert portfolio["gpu_validation"] == "not performed by this command"
```

### 2. 为什么不能只随机比较一次

随机输入能覆盖一般数值，却不一定触发具体边界。你还需要长度一、非整块尾部、block 大于序列、不同 value 维度、极端分数、全无效行、矩形 query/key 以及因果偏移。每项边界都应对应一条明确机制，不能为了增加数量把同一随机例子复制几十次。

测试还应包含负向控制。若你的测试声称验证因果性，就故意移除 mask，确认它能失败；若声称验证布局，就提供非连续视图或明确拒绝该输入。一个测试在正确代码上通过，并不能证明它会发现你最担心的错误。

## 第十课：完整实现矩形位置 mask 的参考与分块版本

### 1. 先定义 query 的逻辑位置

decode 中 query 长度可以是一，但它对应完整历史的最后位置。例如 key 有六个位置，query 对应位置五，它应该读取零到五全部历史，而不是因为 query 数组下标零就只读取 key 零。我们把 query_positions 作为显式参数，允许条件为 `key_position <= query_position`。

下面两个完整函数都采用二维 Q/K/V，Q 和 K 最后维相同，V 可以使用另一特征维。key 位置固定为零到 K 长度减一，query_positions 是每条 query 的逻辑位置。位置为负一时没有可见 key，约定输出零。它们是 CPU 教学实现，没有自动求导或真实 GPU 后端。

```python
def check_attention_inputs(q, k, v, query_positions):
    q, k, v = [np.asarray(x, dtype=np.float64) for x in (q,k,v)]
    positions = np.asarray(query_positions)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("Q/K/V must be matrices")
    if min(*q.shape, *k.shape, *v.shape) < 1:
        raise ValueError("nonempty dimensions required")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("incompatible attention dimensions")
    if positions.shape != (q.shape[0],) or positions.dtype.kind not in "iu":
        raise ValueError("one integer logical position per query required")
    if not all(np.isfinite(x).all() for x in (q,k,v)):
        raise ValueError("finite inputs required")
    return q,k,v,positions


def dense_position_attention(q,k,v,query_positions):
    q,k,v,positions = check_attention_inputs(q,k,v,query_positions)
    scores = q @ k.T / np.sqrt(q.shape[1])
    allowed = np.arange(len(k))[None,:] <= positions[:,None]
    masked = np.where(allowed, scores, -np.inf)
    maximum = masked.max(axis=1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    weights = np.exp(np.where(allowed, scores-maximum, -np.inf))
    denominator = weights.sum(axis=1, keepdims=True)
    probabilities = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator>0)
    return probabilities @ v


def online_position_attention(q,k,v,query_positions,block=3):
    q,k,v,positions = check_attention_inputs(q,k,v,query_positions)
    if type(block) is not int or block < 1:
        raise ValueError("positive integer block required")
    output = np.zeros((len(q), v.shape[1]))
    for row in range(len(q)):
        maximum, denominator = -np.inf, 0.0
        numerator = np.zeros(v.shape[1])
        for start in range(0,len(k),block):
            stop = min(start+block,len(k))
            ids = np.arange(start,stop)
            valid = ids <= positions[row]
            if not valid.any():
                continue
            selected = ids[valid]
            scores = k[selected] @ q[row] / np.sqrt(q.shape[1])
            next_max = max(maximum,float(scores.max()))
            alpha = np.exp(maximum-next_max)
            p = np.exp(scores-next_max)
            numerator = alpha*numerator + p @ v[selected]
            denominator = alpha*denominator + float(p.sum())
            maximum = next_max
        if denominator > 0:
            output[row] = numerator/denominator
    return output

rng = np.random.default_rng(7)
rect_q, rect_k, rect_v = rng.normal(size=(2,3)), rng.normal(size=(6,3)), rng.normal(size=(6,4))
positions = np.array([4,5])
for block in [1,2,4,9]:
    np.testing.assert_allclose(online_position_attention(rect_q,rect_k,rect_v,positions,block),
                               dense_position_attention(rect_q,rect_k,rect_v,positions),
                               rtol=1e-12,atol=1e-12)
np.testing.assert_array_equal(dense_position_attention(rect_q[:1],rect_k,rect_v,[-1]), np.zeros((1,4)))
```

### 2. 逐行解释边界处理

check_attention_inputs 先固定矩阵与维度关系，避免后面的广播偷偷接受错误 shape。positions 必须每 query 一个整数。这里允许超出最后 key 的位置，含义是现有 key 全部可见；也允许负位置以构造全无效行。实际系统可以进一步收紧位置合同，但应保持验证与文档一致。

dense 版本先构造 allowed，再为每行选择最大有效分数。全无效行的最大值是负无穷，我们将其稳定化基准替换为零，并通过 where 保证所有无效权重仍为零。最后 divide 的 where 防止分母零时产生 NaN，输出按约定保持零。不能只在结果末尾用 nan_to_num 把错误擦掉，却不知道哪种输入触发了它。

online 版本每次只取当前块里的有效位置。全无效块直接跳过，因此没有对空数组取 max；若一整行都无有效 key，denominator 始终零，输出保持初始化零。遇到有效块时采用 K07 的 m/l/u 更新，最后才除分母。这里用索引选取产生额外 CPU 拷贝，是为了把 mask 语义写清楚。

### 3. 加上真正检验因果性的干预实验

构造四个 query/key，K 全零使有效位置权重均匀，V 为递增值。修改最后一个 value 后，前三个 query 输出必须不变，最后一个必须变化。后半条条件很重要：如果修改没有任何影响，可能是测试数据或代码路径根本没有使用被改位置，不能用它证明 mask 正确。

```python
causal_q = np.ones((4,1))
causal_k = np.zeros((4,1))
causal_v = np.arange(4.0).reshape(4,1)
causal_positions = np.arange(4)
before = dense_position_attention(causal_q,causal_k,causal_v,causal_positions)
changed_v = causal_v.copy()
changed_v[-1,0] += 100.0
after = dense_position_attention(causal_q,causal_k,changed_v,causal_positions)
np.testing.assert_array_equal(before[:3],after[:3])
assert after[-1,0] > before[-1,0]
wrong_positions = np.full(4,3)
wrong_before = dense_position_attention(causal_q,causal_k,causal_v,wrong_positions)
wrong_after = dense_position_attention(causal_q,causal_k,changed_v,wrong_positions)
assert not np.array_equal(wrong_before[:3],wrong_after[:3])
```

最后两行构造错误的“每条 query 都能看完整历史”路径，确认同样干预会污染早期输出。这个负向控制让验收更有说服力。之后把相同输入喂给设备实现，仍应保留因果不变量，即使浮点比较需要调整容差。

## 第十一课：从独立算子走到真实框架的一条调用链

选择一个边界清楚的算子，例如 RMSNorm。先从模型 forward 找到调用处，记录输入张量与参数；再沿包装层进入算子注册或后端分派；接着找到具体 CUDA/Triton 实现；最后观察输出如何交回下一层。每次只追一条实际配置会触发的路径，避免把整个仓库所有实现同时摊开。

你需要在每个边界回答同一组问题：shape、dtype、stride、设备、参数单位、是否原地修改、是否允许别名、谁分配输出，以及异常在哪里校验。两个函数都叫 rms_norm，不代表 epsilon、累加精度、权重广播或布局支持一定相同。读真实代码时应拿具体输入核对合同，而不是根据名字填空。

本卷 [固定版本源码路线](source-reading.md) 已给出 Triton、CUTLASS 和 FlashAttention 的读取快照。推理系统卷 S09/S17 连接模型 runner 与 backend。固定提交用于让报告可复现，不表示上游未来版本仍保持同一文件布局。发现路径变化时，记录新提交并重新追踪，而不是把旧报告链接悄悄改到 main。

一个适合第一份源码作品的改动，是在隔离副本里为某个已支持形状补一条边界测试，或增加一条可解释诊断。先得到失败输入和预期行为，再改实现，最后运行相关测试。不要一开始就改生产安装包、替换全部注意力后端或同时调整模型配置，那样很难知道结果来自哪项变化。

完整报告应包含：固定提交、文件与符号、输入合同、一次数值轨迹、你的参考实现、上游实际路径、差异原因、测试命令和未验证项。只有当确实编译运行过上游路径，才能写该环境通过；读过源码和在自写 NumPy 上得到同样公式，不能代替真实集成验证。

## 第十二课：三个常见故障的分层诊断

### 故障一：两个进程启动了，但只有一个进入通信

先比对 rank 日志，不要立即怀疑网络带宽。如果一个 rank 在设备选择或分配阶段失败，另一个停在 AllReduce，根因发生在 collective 之前。检查可见设备列表、本地 rank 映射、world size 和每个进程的异常；由当前作业的管理边界结束相关进程，不干扰其他任务。

参考修复是统一前置配置与检查，保证两边进入匹配操作，并保留超时。把超时从一分钟改成一小时不修复参与者缺失，只会增加资源占用。修复后先重复十六元素微例子，再增加消息规模，避免同时引入更多变量。

### 故障二：数值正常，但报告带宽高得不合理

检查是否仅测量了异步提交的主机时间，是否漏了完成同步，是否把字节与比特混淆，或者把 MiB 当十进制 MB。还要看是否重复把 p 个 rank 的字节加到分子，却只用单 rank 的延迟作为分母，而没有说明统计定义。

参考修复是重新定义测量区间与单位，区分算法带宽、链路有效量和端到端吞吐，不把不同口径放在同一图里比较。保存原始重复值与环境，而不只给一个最佳数。异常好看的结果也需要排查，不能只对慢结果保持怀疑。

### 故障三：CPU 算子通过，接入框架后输出错误

先比较真实输入合同与独立测试：框架可能传入非连续布局、不同 dtype、不同 mask 位置或别名输出。打印最小必要的 shape/stride/参数，提取一个合成复现，不必复制公司数据。然后把同一输入交给参考与候选路径，逐段缩小差异首次出现的位置。

若差异来自 query 逻辑位置，修复应发生在 mask/position 合同；若来自权重布局，则检查转置和 stride；若来自异步复用，检查完成事件与缓冲寿命。不能一律提高容差或把输入强制复制连续后就宣称支持原接口，除非明确把复制成本和布局约束写进包装合同。

## 第十三课：十二道练习与完整参考答案

### 练习一：从结果选择 collective

三个 rank 各有一个两元素分片，你希望每个 rank 都得到六元素完整向量，应选择哪种结果关系？参考答案是 AllGather，因为要保留各片内容并按 rank 拼接；AllReduce 会按位置合并成两元素，改变含义。若希望三份同形状部分矩阵相加且每个 rank 都获得结果，才符合求和 AllReduce。答案必须说明元素身份，不能只背 API 名字。

### 练习二：手算 ReduceScatter 后再 AllGather

两个 rank 输入 `[1,2,3,4]` 和 `[10,20,30,40]`。参考中间状态是 rank 零领取 `[11,22]`，rank 一领取 `[33,44]`；再按 rank 顺序拼接恢复 `[11,22,33,44]`。若拼接顺序反过来，数值集合相同但布局错误，后续矩阵解释会出问题。可直接用第二课的 reduce_scatter_result 做拼接断言。

```python
np.testing.assert_array_equal(np.concatenate(reduce_scatter_result), allreduce_result)
assert not np.array_equal(np.concatenate(reduce_scatter_result[::-1]), allreduce_result)
```

### 练习三：验证 ring 不只适用于三 rank

对一、二、三、四个 rank，各生成可被 p 整除的输入长度，比较 ring 输出与独立逐位置总和。参考代码如下，同时检查 trace 长度为 `2*(p-1)`。这里比较的是数学结果与步骤数量，不是网络耗时。

```python
random = np.random.default_rng(123)
for ranks in [1,2,3,4]:
    sample = random.normal(size=(ranks, ranks*5))
    actual, steps = ring_sum_reference(sample)
    expected = np.broadcast_to(sample.sum(axis=0), actual.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert len(steps) == 2*(ranks-1)
```

### 练习四：解释 ring 两个阶段为何一个加、一个覆盖

参考答案：第一阶段传播部分和，每次加入尚未包含的参与者贡献；第二阶段传播已经完成的归约块，接收者只需保存副本。第二阶段继续相加会重复计数。你可以在个人副本把第二阶段赋值改成加法，用三 rank 例子证明测试失败，再恢复代码。只用全零输入无法揭露此错误，因此必须选择非零且可区分的贡献。

### 练习五：把元素数换成字节数

每 rank 有一百万个 float32 元素，S 应是四百万字节，而不是一百万。若换 float16 且仍有相同元素数，S 是两百万字节，但数值误差与支持路径也可能改变。参考成本计算必须把元素大小乘进去，再代入第五课公式；不能把两种 dtype 的吞吐比较解释成只改了“通信算法”。

### 练习六：找出错误均值

rank 零两个样本，总和六；rank 一一个样本，总和九。局部均值三和九，简单平均六；真正总均值十五除以三，等于五。参考实现是归约总和和计数，最后相除。若训练程序已经在本地按某种规则缩放梯度，还需检查全局归约是否再次平均，避免重复缩放。

### 练习七：制造调用顺序不一致

一个 rank 计划 AllReduce→AllGather，另一个计划 AllGather→AllReduce。使用第七课 check_collective_plan 应拒绝。参考解释不是“函数名字不同所以报错”这么简单，而是同一个组的第一个通信位置无法匹配相同操作。真实系统日志应记录组与序号，帮助定位第一个分歧，不能只写最后一次超时异常。

### 练习八：推导延迟与带宽的交界

从 `2*(p-1)*alpha = 2*(p-1)/p*S/B` 推出 `S=p*alpha*B`，前提 p 大于一。对 p=8、alpha 五微秒、B 二百五十亿字节每秒，结果一百万字节。p=1 时两项恒为零，不能把消去零因子后的式子解释成真实交界。这个边界提醒你，代数变形也有条件。

### 练习九：验证矩形 decode 的位置偏移

用两条 query、六条 key，位置为四和五，与错误位置零和一比较。正确第一行可读五个位置，第二行可读六个；错误版本只读一个和两个。参考代码对照第十课两个完整实现，并加入一组固定数值，使两种位置合同确实产生不同结果。

```python
position_q = np.ones((2,1))
position_k = np.zeros((6,1))
position_v = np.arange(6.0).reshape(6,1)
right = dense_position_attention(position_q,position_k,position_v,[4,5])
wrong = dense_position_attention(position_q,position_k,position_v,[0,1])
np.testing.assert_allclose(right, [[2.0],[2.5]])
np.testing.assert_allclose(wrong, [[0.0],[0.5]])
```

### 练习十：增加 batch/head 包装而不改变单头公式

输入约定为 `[batch,head,token,feature]`，Q/K/V 的 batch/head 数相同，所有头共享同一 query_positions。下面提供完整 CPU 包装，逐个 batch/head 调用已验证的二维函数。它不支持 GQA、不支持每样本不同长度 mask，也不声称是高性能批量实现；这些限制写进合同，避免外层 shape 一扩就误以为支持全部真实模型。

```python
def batched_position_reference(q,k,v,query_positions):
    q,k,v = [np.asarray(x) for x in (q,k,v)]
    if any(x.ndim != 4 for x in (q,k,v)):
        raise ValueError("expected [batch,head,token,feature]")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2] or min(q.shape[:2]) < 1:
        raise ValueError("matching nonempty batch/head axes required")
    return np.stack([
        np.stack([dense_position_attention(q[b,h],k[b,h],v[b,h],query_positions)
                  for h in range(q.shape[1])])
        for b in range(q.shape[0])])

bq = np.stack([np.stack([rect_q,rect_q*2]), np.stack([rect_q*3,rect_q*4])])
bk = np.broadcast_to(rect_k, (2,2,*rect_k.shape)).copy()
bv = np.broadcast_to(rect_v, (2,2,*rect_v.shape)).copy()
bresult = batched_position_reference(bq,bk,bv,[4,5])
assert bresult.shape == (2,2,2,4)
np.testing.assert_allclose(bresult[1,0], dense_position_attention(rect_q*3,rect_k,rect_v,[4,5]))
```

### 练习十一：给作品集加入明确的未执行字段

参考报告应区分 CPU 数值通过、GPU 代码存在、GPU 实际运行、真实多卡通信和真实性能测量。不能用一个 passed 字段把所有环境混在一起。下面是最小结构；其中布尔值来自本章实际 CPU 计算，设备字段保持未执行。你在自己的机器完成真实实验后，应另存该次环境与原始结果，而不是无证据地改一个词。

```python
import json
report = {
    "cpu_attention": portfolio["status"],
    "cpu_attention_cases": portfolio["attention_cases"],
    "ring_semantics": "passed on CPU reference",
    "rectangular_attention": "passed on CPU reference",
    "gpu_kernel": "not executed in this CPU lesson",
    "nccl_two_rank": "not executed in this CPU lesson",
    "performance": "not measured",
    "numpy": np.__version__,
}
assert json.loads(json.dumps(report))["performance"] == "not measured"
```

### 练习十二：形成一份可以审查的小改动

选择自己参考实现中的一个真实缺口，例如矩形逻辑位置、全无效行或非整块尾部。先保留失败输入与预期结果，再修改实现，最后提交前后对照和局部回归。参考交付不是一句“修复 attention”，而是写清触发条件、原错误、修复后的合同，以及没有扩大的支持范围。

如果将来向开源项目提补丁，也先在隔离副本完成这套证据。是否向上游发送、以什么身份发送，由你决定；课程不会因为提供源码任务就自动代表你或公司联系维护者。技术准备与外部沟通是两个步骤，前者应先做到可审查。

## 第十四课：把八卡资源用于有问题意识的扩展实验

两卡微例子通过后，不必立即运行更大的模型。先选一个明确问题：同样总消息大小下，不同 rank 数的固定成本怎样变化；不同设备组合是否呈现不同瓶颈；通信能否与一项独立计算重叠；专家分发的不均衡会怎样影响最慢 rank。每次只选择其中一个，写出预测和可证伪观察。

在共享八卡服务器上，先使用被分配的短窗口和最小可回答问题的参与者数量。涉及多卡拓扑时使用相应 H100 是合理的，但没有必要把一个十六元素接口检查扩成长期八卡占用。记录实际设备组合、消息尺寸、运行次数与超时，任务结束后确认自己的进程退出。

设备组合不同可能包含不同互连路径，因此不能只用“同型号 GPU”推断相同通信性能。观察工具和系统配置以实际环境为准；本章不假设你的八卡一定采用某种互连拓扑。先读取可用拓扑信息，再设计比较，避免把未确认的机器结构写进结论。

当你增加真实设备实验，应保留 CPU oracle 与协议模拟，它们仍能帮助区分数学错误、布局错误和设备执行问题。GPU 只负责提供新的执行证据，不能替代前面建立的合同。某个快配置若只在一组数据正确，就还不是可交付优化；某个慢参考若定义清楚，则仍然是有价值的验收基线。

## 本卷结业：从名字走到可解释的实现

你应能从字节地址与 stride 解释一个线程读到哪里，从归约和矩阵乘法推导输出，从 m/l/u 说明在线注意力为何等价，从事件与引用说明何时可以复用缓冲，再从分片数学解释为什么需要某种 collective。每一层都要有一个自己实现过、测试过、故意破坏过再修好的小例子。

最后整理一份作品集：基础向量与归约、Norm、GEMM、注意力、一个自定义算子接口、一条真实框架源码路径，以及本章的通信语义与受限设备实验。每个作品写清已执行范围、输入合同和一个仍然不支持的条件。能够准确解释自己的边界，才有基础继续打磨性能，而不是靠算子名称和硬件规格描述能力。

建议保留一份按实验编号排列的索引，把每个结论链接到对应命令、输入文件和原始结果。失败实验也保留，注明后来修复了哪个条件。这样几个月后重新阅读时，你能恢复当时的推理与环境，而不是只剩一段无法解释的“最佳成绩”。
