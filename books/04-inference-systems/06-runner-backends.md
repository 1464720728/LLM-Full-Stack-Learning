# S06 Model Runner：从调度计划到可验证的张量执行

本章的任务是把“框架调用模型”拆开，直到你可以自己编写一个接收可变长度请求、构造位置与边界、执行 attention、把结果正确交还请求的 CPU Runner。先修 S01 的请求生命周期、S02 的调度和 S03 的页表；矩阵乘法与注意力的数学可以边读边回查模型卷。你不需要先懂 CUDA。先把索引、形状、所有权和执行顺序写对，之后 GPU 才是在同一个契约下更换计算实现。

本章有两个不同的学习对象。本仓库现有 Engine 用统一调度器管理请求，但对各个请求分别调用 `Backend.logits(context)`，其中 TinyAttentionBackend 每次重新计算完整上下文。它能验证状态和数值，却没有把整个调度批次变成一次 GPU 执行。本章另外给出真正把多个上下文拼接后计算的 NumPy 参考程序，用来解释真实 Runner 的元数据。这个参考程序仍然会计算完整上下文，也没有声称提供真实分页 GPU attention。请始终把“调度层批次”“张量层批次”“设备上的一次或一组 kernel”分开记录。

## 1. 第一课：五个名字分别承诺什么

模型定义回答“给定输入和参数，数学上应该计算什么”。例如一个 decoder 层先对隐藏状态归一化，再投影成 Q、K、V，进行因果注意力，经过输出投影与残差，再计算 MLP。模型定义可以接受过去的 KV，也可以不接受；这要看接口，不能凭它叫 Transformer 就断言已经实现增量推理。

Model Runner 回答“这轮调度工作如何变成模型能够消费的张量”。调度器可能只说请求甲计算三个新位置、请求乙计算一个新位置。Runner 要从请求表取 token，填入输入缓冲区，计算逻辑位置，构造 query 边界、序列长度、页表、缓存写入位置和采样位置。模型输出的是张量，而 HTTP 请求需要分开的结果，因此 Runner 还要保存足够的映射把张量行还原为请求身份。

Attention Backend 回答“某个注意力计算使用哪一种实现以及哪一种布局”。数学可以相同，实现可以是简单矩阵乘法、融合 kernel 或专用分页 kernel。后端不仅提供函数名，还提供元数据的解释：块大小、head 维、KV 的轴顺序、mask 定义、累加精度等。错误的布局约定不会总是崩溃，有时会给出看起来正常但实际上属于别的位置的数值。

Worker 是执行工作的进程或设备侧工作单元。它通常拥有本 rank 的模型权重、设备上下文和缓存存储。Executor 组织一个或多个 Worker，安排远程调用、进程通信或分布式集合通信，并把结果传回上层。具体项目可能合并这些对象，也可能继续分层；读源码时应看数据归谁、方法输入输出是什么，而不是要求所有框架都有五个同名类。

Scheduler 决定“谁在这一轮获得多少资源”。它不应该通过修改 RoPE 的位置张量来调节公平性；Runner 也不应该看到输出长度较长就偷偷终止用户请求。边界清楚的好处不只是容易读：当错误发生时，你能问是调度计划已经错了，还是计划到张量的翻译错了，还是数值实现错了。三个问题需要三种完全不同的证据。

| 层次 | 典型输入 | 典型输出 | 最关键的责任 |
|---|---|---|---|
| Scheduler | 排队请求、空闲页、预算 | 每请求新计算范围 | 进度、公平性、资源预留 |
| Runner | 执行计划、请求状态、页表 | logits 或中间结果、执行事件 | 索引、布局、位置、输入生命周期 |
| 模型 | token/embedding、position、KV | hidden states 或 logits | 层级数学与参数 |
| Attention Backend | Q/K/V、缓存、边界元数据 | attention 输出 | 约定的 mask 与数值正确性 |
| Worker/Executor | 本地或分布式执行命令 | 本 rank 或聚合结果 | 设备、通信与完成状态 |

同一个方法可以承担多项责任。例如 `GPUModelRunner.execute_model` 的名字只是阅读入口，不能据此认定其返回值一定是采样后的 token。固定源码中还有 `sample_tokens`，这本身就提醒我们必须追踪跨方法保存的结果。真正读懂一个类的标准，是能把一条请求的身份从输入一直追到输出，而不是背下类名。

## 2. 第二课：执行计划是一份合同，不是一串 token

本地 Engine 的 Work 记录请求 ID、阶段、起始逻辑位置和本轮 token 序列。设甲已经计算四个位置，这轮需要计算 `[8,9]`，那么这两个 token 的逻辑位置是四和五。它们被放在批次数组的第零、第一行，并不会把逻辑位置改成零和一。物理数组偏移、序列逻辑位置和 token 的整数值，是三个互相独立的概念。

设本轮有三个请求。甲是首次 prefill，新 token 为 `[1,2,3]`，之前计算数量零；乙已经有十个位置的 KV，这轮 decode 输入为 `[7]`；丙已有四个位置，本轮追加 `[4,5]`。拼接后的 token 是 `[1,2,3,7,4,5]`。positions 应为 `[0,1,2,10,4,5]`，而不是 `[0,1,2,3,4,5]`。批次压紧只是改变物理行号，不得改变每个请求在模型中的位置。

query offsets 为 `[0,3,4,6]`。相邻两个值之间的半开区间就是一个请求本轮的 query 行。甲占 `[0,3)`，乙占 `[3,4)`，丙占 `[4,6)`。使用半开区间的便利在于长度直接等于右端减左端，最后一个右端恰好等于总 query 数，不需要为每个请求额外加减一。

这轮执行完成后，可见上下文长度分别为三、十一、六。这个数组常被叫作 sequence lengths，不能用本轮 query 数三、一、二代替。对乙而言只有一个新 query，但它应读取十一个位置的 key/value。如果错误地把序列长度填成一，模型不会看到过去十个位置，即使历史 KV 完好地躺在设备内存里也没有用。

| 请求 | 原 computed | 本轮 query 数 | query 行范围 | 逻辑 positions | 完成后的上下文长度 |
|---|---:|---:|---|---|---:|
| 甲 | 0 | 3 | 0 到 2 | 0、1、2 | 3 |
| 乙 | 10 | 1 | 3 | 10 | 11 |
| 丙 | 4 | 2 | 4 到 5 | 4、5 | 6 |

执行计划至少需要满足：请求 ID 不重复；count 为正整数；start 不超过已知上下文长度；本轮范围没有越过已知 token；资源已经足够容纳即将写入的 KV；同一请求没有两个互相冲突的在途写任务。异步系统还需要标识计划所属的请求代次以及提交编号，否则一个被取消后重新创建的同名请求可能接到旧结果。

这些校验应尽量发生在提交设备之前。设备越接近底层，错误信息越难关联到用户请求。如果在 Python 层就能指出“乙的 position 长度与 query 数不相等”，不要等到矩阵乘法报 shape mismatch，或者等到异步 kernel 在下一次同步时报告非法访问。前置校验的成本通常远小于一次错误设备执行后的排查成本。

## 3. 第三课：手写可变长度打包器，并验证逆变换

下面程序只构造元数据，不运行模型。它使用标准库，可以直接在仓库根目录运行。列表里的每个三元组是请求 ID、原 computed、新 token；返回的 owners 保存每一行所属的请求。实际框架可能用整数请求槽位代替字符串，但这个教学版本先保留可读身份。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Part:
    rid: str
    start: int
    token_ids: tuple[int, ...]

def pack(parts):
    flat, positions, owners = [], [], []
    offsets = [0]
    seen = set()
    for part in parts:
        if not part.rid or part.rid in seen:
            raise ValueError("request identity must be unique")
        if type(part.start) is not int or part.start < 0:
            raise ValueError("start must be a nonnegative integer")
        if not part.token_ids:
            raise ValueError("empty work is not schedulable")
        if any(type(t) is not int or t < 0 for t in part.token_ids):
            raise ValueError("token IDs must be nonnegative integers")
        seen.add(part.rid)
        flat.extend(part.token_ids)
        positions.extend(range(part.start, part.start + len(part.token_ids)))
        owners.extend([part.rid] * len(part.token_ids))
        offsets.append(len(flat))
    return flat, positions, owners, offsets

parts = [
    Part("A", 0, (1, 2, 3)),
    Part("B", 10, (7,)),
    Part("C", 4, (4, 5)),
]
flat, pos, owner, off = pack(parts)
assert flat == [1, 2, 3, 7, 4, 5]
assert pos == [0, 1, 2, 10, 4, 5]
assert off == [0, 3, 4, 6]
restored = [flat[off[i]:off[i+1]] for i in range(len(parts))]
assert restored == [list(p.token_ids) for p in parts]
print(flat, pos, off)
```

`frozen=True` 使 Part 的字段不能被重新赋值，token_ids 本身又是 tuple，因此计划建立后不会被普通列表追加意外改变。它不意味着整个执行系统具有线程安全性；不可变对象只是让“计划在执行中被悄悄改掉”这一类问题更容易避免。

`offsets=[0]` 建立第一个请求的左端。每处理一个请求，先扩展 flat，再把新的总长度追加到 offsets。这样最后一个 offset 一定等于 flat 长度。代码没有假定三个请求长度相等，因此不能用普通的二维 reshape 代替它。reshape 要求每行拥有相同元素数，可变长度请求必须通过 padding 或额外的边界数组描述。

`range(start,start+count)` 构造的是模型序列位置。这行代码最容易在“先写一个能跑的 demo”时被省略，然后在加入 RoPE 或分块 prefill 后暴露问题。没有位置编码的玩具模型可能对这种错误不敏感，不代表真实模型也不敏感。验证位置数组时要主动选择 start 非零的请求；所有测试都从零开始会掩盖错误。

逆变换把每一段 flat 按 offsets 切回去。如果往返不等，至少有打包或边界错误；但往返相等仍不能证明 positions、页表正确。因此这个断言只是第一层证据。完整验收还要检查每个位置属于哪个请求、缓存写到哪里、attention 允许读哪些 key，以及结果是否回到对应的请求。

空批次在这个函数里返回空 flat 与 offsets 为 `[0]`，它可以作为“没有要执行的工作”的合法元数据，但调用模型的上层必须决定是否直接返回。空请求则明确拒绝，因为没有 query 的请求不应该出现在本轮 Work 中。区分空批次与空成员，可以避免使用 `max(lengths)` 等操作时产生不明确的边界行为。

## 4. 第四课：一个张量拥有形状，还拥有步长和语义

矩阵形状告诉你每个轴有多长，不能告诉你每一行代表什么。一个 `[6,8]` 张量可能是六个 token 的八维 embedding，也可能是六个请求各自的最后隐藏状态。只看 shape 完全一样，接下来允许的操作却不同。Runner 中每个中间张量都应该能回答“第 i 行对应哪个请求的哪个逻辑位置”。

在 NumPy 中，二维 C 连续数组的最后一维通常连续存放。转置可能只改变 strides 而不移动数据。例如 `x.T` 的 shape 改变了，底层 storage 仍然可能与 x 相同。一个假定最后一维连续的自定义 kernel 如果只读取 shape 不处理 strides，就会读错数据。高层库的矩阵乘法一般替你处理多种布局，但自己写算子时必须把支持条件变成明确契约。

dtype 同样不只是显存占用。token IDs 通常需要整数，embedding 是浮点数，offsets 可能根据实现要求使用特定整数宽度。把 token IDs 存成浮点后再转换，可能在非常大的值上丢失整数精度；把页表存为错误的整数类型，可能造成 kernel 对同一段字节做不同解释。对本课程的小词表，这些错误不一定立即出现，因此要从接口上禁止不合约的数据。

本轮 query 的轴常写成 `Tq`，历史缓存的可见 token 数常写成 `Tk`。decode 时可以是 `Tq=1,Tk=4096`。如果你看到 attention 的 Q 和 K 第一维不同，不要立刻认为 shape 错了；这恰好可能是增量推理的正常情况。真正需要相等的，是乘法中被收缩的 head dimension，而不是 query 和 key 的序列长度。

多头场景中常见逻辑形状是 Q 为 `[Tq,Hq,D]`，K/V 为 `[Tk,Hkv,D]`。当采用 grouped-query attention，Hq 可以大于 Hkv，每组 query heads 共享一个 KV head。Runner 或后端必须明确映射规则，不能把 Hkv 直接补零成 Hq，也不能随意沿错误的轴 repeat。S10 会继续讨论这些 head 如何在多个 rank 上分配。

一个好的调试打印不只是 `shape=(...)`，还应包括 dtype、device、strides、有效范围、请求行映射和必要的最小最大值。实际应用中不要打印完整用户 prompt 或全部隐藏状态；为了排查索引，你通常只需要合成输入、请求匿名 ID 和数值摘要。可复现的小合成样本比一大段真实业务文本更容易定位错误，也更适合提交公开 issue。

## 5. 第五课：拼接 attention 时，边界就是隔离机制

假设把两个请求的完整上下文拼成 `[a0,a1,b0]`。如果直接对三行做普通因果 attention，最后的 b0 会看到 a0、a1 和自己，因为普通下三角 mask 只知道行号大小，不知道请求身份。这不是一个微小精度问题，而是把不同请求的上下文混在一起，既破坏模型语义，也破坏请求隔离。

正确的完整上下文拼接 mask 同时满足两个条件：query 与 key 属于同一请求；key 的逻辑位置不晚于 query。设行所属请求为 owner，逻辑位置为 pos，则允许矩阵元素满足 `owner[i]==owner[j] and pos[j]<=pos[i]`。这会得到多个互不连接的下三角块，因此常称为块对角因果 mask。

对于甲长度三、乙长度一、丙长度二的完整上下文示例，甲的第三行允许读取甲的前三行，乙的唯一一行只能读取乙自身，丙的第二行允许读取丙的两行。即使乙在 flat 中排在甲之后，它也不能读取甲。padding 行则不属于任何真实请求，其结果必须被丢弃，写缓存时更不得使用真实请求的页槽。

| Query 行 | 所属请求/位置 | 允许读取的 flat key 行 | 不允许读取的原因举例 |
|---:|---|---|---|
| 0 | 甲/0 | 0 | 甲未来位置尚不可见 |
| 2 | 甲/2 | 0、1、2 | 乙、丙属于其他请求 |
| 3 | 乙/0 | 3 | 甲虽然排在前面，仍不是乙的历史 |
| 4 | 丙/0 | 4 | 丙的下一位置是未来 |
| 5 | 丙/1 | 4、5 | 甲乙均被请求边界隔离 |

这里的表使用完整上下文，因此每个请求位置从零开始。前面增量计划里乙的 position 是十，它之前的十个 key 存在历史缓存中，不在当前 flat query 数组里。真实增量 backend 必须同时理解“当前 query 的边界”和“缓存中的 key 范围”。不能把完整上下文参考程序的 mask 原封不动套到只有新 token 的增量输入上，否则会漏掉历史。

mask 通常把不允许的 score 设为负无穷，再执行 softmax。如果一整行都是负无穷，减最大值会产生无效运算。真实有效 query 一般至少能看到自身，但 padding 行可能没有可见 key。实现必须决定跳过无效行、设置安全占位，或让 kernel 通过有效行数完全不计算它们。把 NaN 在最后替换成零，只会掩盖为什么这行本不应该被计算。

从正确性角度，先构造显式二维 mask 非常有价值，因为可以打印和手算。从性能角度，长度上万时显式分数矩阵和 mask 很昂贵，融合 attention 会通过块索引和条件判断隐式表达相同语义。优化可以改变存储形式，不能改变允许关系。你写 kernel 时应把这个小型显式版本作为数值参考。

## 6. 第六课：逐行读本仓库的模型 Backend

本地接口位于 [backend.py](../../labs/serving/backend.py)。Protocol 声明 backend 至少暴露词表大小以及 `logits(token_ids)`。这里 token_ids 是当前请求的完整上下文，而不是本轮 Work 的新 token。这个差别非常重要：Engine 的页表练习与 TinyAttentionBackend 的实际数值计算是两条关联但没有完全合并的教学路径。

BigramBackend 根据最后一个 token 给每个候选 token 一个分数。它不是真正的大语言模型，但输出确定、词表小，因此适合检查调度、取消和停止边界。TinyAttentionBackend 则持有 embedding、三个投影矩阵和输出矩阵，使用 NumPy 计算单头 attention。以下是该实现的核心计算，下面保留当前仓库的计算顺序，并把合并赋值展开便于逐行阅读：

```python
# 这是 backend.py 中 logits 的核心节选，需放在已有模型对象的方法内阅读。
np = self.np
x = self.embedding[token_ids]
q = x[-1] @ self.q
k = x @ self.k
v = x @ self.v
scores = q @ k.T / x.shape[-1] ** .5
p = np.exp(scores - scores.max())
p /= p.sum()
return ((p @ v + x[-1]) @ self.out).tolist()
```

设定 np 别名之后，第一项数值操作是 embedding 查表。token ID 是索引，不能对 ID 数值本身做线性投影来代替查表。若上下文长为 L，隐藏宽度为 d，x 的形状就是 `[L,d]`。第二行只取最后一个位置的隐藏状态作为 query，得到 `[d]`，这正是本接口只需要“预测下一 token 的最后位置 logits”的原因。

第三、四行对全部上下文计算 K 和 V，形状都是 `[L,d]`。注意每次调用都会重做这两次矩阵乘法，因此它没有增量 KV 的性能收益。第五行中 `q @ k.T` 产生长度 L 的分数，除以 `sqrt(d)` 是标准缩放点积注意力的缩放。只有最后位置的 query，所以所有给定位置都属于它的过去或自身，不需要额外屏蔽未来；若改成计算全部 query，就必须增加因果 mask。

第六、七行执行稳定 softmax。先减最大分数，使最大的指数成为一，可以减少大正数导致的指数溢出；这个平移不会改变最终概率，因为分子与分母都乘了同一常数。这里假定输入数值有限且至少有一个候选位置。更通用的生产接口还要定义无效输入和全 masked 行的行为。

最后一行先得到加权 V，形状为 `[d]`，加上最后位置的 embedding 作为残差，再乘输出矩阵，得到词表长度的 logits。logits 不是概率，也不是 token ID。采样层随后才把分数转换为概率或选择最大值。将 logits 转为 Python list 便于教学接口使用，真实 GPU 流水线若每步都这样拷贝，可能带来不必要同步和主机传输。

这个模型没有多层结构、归一化、MLP、多头、RoPE 或训练过程，因此不能用它说明真实模型质量，也不能从它的用时推断 H100 的吞吐。它的价值是每一项张量都小到可以检查，同时有真实注意力数值，不只是伪造 token。后续把完整 TinyDecoder 接入增量运行时，仍然应保留它作为极简基线。

## 7. 第七课：真正跑一次打包数值计算

下面程序复用 TinyAttentionBackend 的同一组参数，对多个完整上下文同时构造 Q/K/V，使用块对角因果 mask，然后抽取每个请求最后一行 logits。它与逐请求调用 backend 的结果应一致到浮点容差。这里“同时”指一次 NumPy 数组计算，不是多 GPU 并行，也不承诺底层只启动一个算子。

```python
import math
import numpy as np
from labs.serving.backend import TinyAttentionBackend

def packed_logits(model, contexts):
    if not contexts or any(not c for c in contexts):
        raise ValueError("nonempty batch and contexts required")
    if any(type(t) is not int or not 0 <= t < model.vocab_size
           for c in contexts for t in c):
        raise ValueError("token outside vocabulary")
    lengths = np.array([len(c) for c in contexts], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ids = np.array([t for c in contexts for t in c], dtype=np.int64)
    owner = np.repeat(np.arange(len(contexts)), lengths)
    position = np.concatenate([np.arange(n) for n in lengths])
    x = model.embedding[ids]
    q, k, v = x @ model.q, x @ model.k, x @ model.v
    scores = q @ k.T / math.sqrt(x.shape[-1])
    allowed = ((owner[:, None] == owner[None, :]) &
               (position[None, :] <= position[:, None]))
    scores = np.where(allowed, scores, -np.inf)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    all_logits = (p @ v + x) @ model.out
    return all_logits[offsets[1:] - 1]

model = TinyAttentionBackend()
contexts = [[1, 2, 3], [7], [4, 5]]
actual = packed_logits(model, contexts)
expected = np.array([model.logits(c) for c in contexts])
np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
reordered = [contexts[2], contexts[0], contexts[1]]
again = packed_logits(model, reordered)
np.testing.assert_allclose(again, expected[[2, 0, 1]],
                           rtol=1e-12, atol=1e-12)
print("packed reference matches independent requests", actual.shape)
```

`owner[:,None]` 是列向量，`owner[None,:]` 是行向量，广播比较后得到每对 query/key 是否同属一个请求。位置比较使用 key 的列位置不大于 query 的行位置。把两个条件用按位与连接，才能同时保证请求隔离与因果性。若把位置比较方向写反，最后一行通常只能看到自己，模型仍能输出数字，因此必须用数值对照发现。

`axis=1,keepdims=True` 表示每个 query 行独立减最大值并归一化，保留长度为一的轴用于广播。若漏写 axis，整个批次会共同归一化，甲的概率就会受到乙的分数影响。这样的错误不是更换 seed 能修复的，它是概率定义被改了。检查每一行 p 的和为一，是一个有用但仍不足够的断言，因为错 mask 的概率也可能和为一。

`offsets[1:]-1` 是每段最后一行的位置。对长度三、一、二，它等于二、三、五。这个映射把 flat 输出还原成“每请求一行”。如果错用 `offsets[:-1]`，你拿到的是每段第一行；长度一的请求恰好没问题，会让部分样本通过。测试必须同时包含长度一和长度大于一，不能只靠单一形状覆盖。

最后的重排实验验证请求遍历顺序变化不会改变每个请求对应的结果。这对当前 CPU 参考实现应在给定容差内成立。它不是声称所有 GPU backend 在任何 batch 下逐位一致；不同 kernel 路径或归约顺序可能带来微小差异。S18 会区分语义隔离、数值容差、采样可复现与 bitwise 确定性。

这个参考实现显式构造总长度平方的分数矩阵，即使大部分跨请求区域被 mask，也仍然消耗内存和计算。它适合几十个 token 的验证，不适合拿来服务长上下文。高效后端通过独立分段、变长 kernel 或分页索引跳过这些无效区域；先写清楚慢而正确的版本，再优化，是理解框架与算子的共同路线。


### 7.1 用两维向量手算一行 attention

为了知道程序中的每个矩阵乘法确实在算什么，暂时不使用随机参数。设隐藏宽度为二，两个 token 的 embedding 分别是 `x0=[1,0]`、`x1=[0,1]`，Q、K、V 投影都用二阶单位矩阵，输出投影也用单位矩阵。这里词表仅有两个元素，因此最后输出也是长度二。这个特殊参数只是便于手算，不代表训练出来的模型应该如此设置。

最后位置的 query 是 `[0,1]`，两个 key 是 `[1,0]` 与 `[0,1]`。点积分别为零和一，除以根号二以后，分数约为零和零点七零七一。稳定 softmax 先减去最大值，指数权重约为零点四九三一和一，归一化概率约为零点三三零二和零点六六九八。概率偏向第二个位置，但第一个位置仍然参与输出。

由于 V 就是两个 embedding，加权和为 `[0.3302,0.6698]`。加上最后位置残差 `[0,1]` 得到 `[0.3302,1.6698]`。输出投影为单位矩阵，所以这就是两个词的 logits。注意 attention 概率约为零点三三和零点六七，而最终 logits 的第二项大于一，两者不是同一种数量。若还要采样，应对这两个 logits 再做采样层定义的概率变换。

再计算第零个 query。因果性只允许它看到第零个 key，因此注意力概率是 `[1,0]`，attention 输出为 `[1,0]`，加残差后为 `[2,0]`。若错误地允许它看未来第二个 token，结果会改变。最后位置没有未来可屏蔽，并不意味着其他位置也没有；完整矩阵实现必须正确处理每一行。

| 中间量 | 最后位置的数值 | 维度 | 应检查的性质 |
|---|---|---|---|
| query | [0,1] | 二 | 只来自最后位置 |
| raw dot products | [0,1] | 上下文长二 | 每项对应一个 key |
| scaled scores | [0,约0.7071] | 上下文长二 | 除以根号隐藏宽度 |
| attention probability | [约0.3302,约0.6698] | 上下文长二 | 非负且和为一 |
| weighted value | [约0.3302,约0.6698] | 隐藏宽度二 | 是 V 的加权和 |
| residual output | [约0.3302,约1.6698] | 隐藏宽度二 | 不是概率 |

如果把两个请求各自只有一个 token 的输入拼起来，它们分别只能看到自己。即使合起来的 embedding 与上述两 token 序列相同，正确结果也变成第一请求 `[2,0]`、第二请求 `[0,2]`。这说明仅凭 flat token 数组不能决定模型语义，还需要请求边界。同一段物理数组加上不同的边界元数据，代表不同的计算问题。

## 8. 第八课：缓存地址、逻辑位置和采样位置如何一起工作

增量 Runner 接到乙的新 token 时，不需要重新投影此前十个 token，但要找到它们的 K/V。页表把逻辑块编号映射到物理块编号；slot mapping 则告诉这轮每个新 token 的 K/V 写入哪一个具体槽位。positions 进入位置编码，query offsets 描述当前批次分段，sequence lengths 限制可读历史。四者都与位置有关，但不能互相替代。

设块大小四，乙逻辑页表为 `[5,2,9]`，新 token 的逻辑位置十。逻辑块是 `10//4=2`，块内偏移是 `10%4=2`，因此要写物理页九的槽二。如果把缓存展平成“页×块大小”的槽数组，slot 为 `9*4+2=38`。position 仍是十，不是三十八；RoPE 若用三十八，会错误地让物理内存布局影响模型含义。

甲这轮计算三个 query 时，每个 query 的可见历史长度依次增加。若甲原 computed 为四，新 query 位置为四、五、六，那么第一行只能看零到四，不能看本轮后面刚写入的五、六。将所有新 K/V 一次写入缓存通常是可以的，但 attention 的因果限制仍必须逐 query 生效。“内存中已经存在”与“数学上允许读取”是两回事。

采样位置又是另一组索引。一个长 prompt 分成三个 chunk，前两个 chunk 只是在建立缓存，不应向用户生成下一 token；只有完成当前已知上下文的最后一个位置，才得到可用于下一 token 采样的 logits。本地 Engine 使用 `computed==len(context)` 判断这一点。真实 Runner 可能根据调度元数据筛选需要采样的请求行，避免把每个 prefill token 的 logits 都传回主机。

| 元数据 | 示例 | 回答的问题 | 常见误用 |
|---|---|---|---|
| position | 10 | token 在本请求的逻辑序列哪里 | 错填 flat 行号 |
| block table | [5,2,9] | 历史逻辑块存在哪些物理页 | 将页号当 token 值 |
| slot mapping | 38 | 新 K/V 写到哪个物理槽 | 对 padding 也写真实槽 |
| query offsets | [0,3,4,6] | 本轮每请求有哪些 query 行 | 当成完整上下文边界 |
| sample row | 本段最后有效行 | 哪行 logits 可以生成新 token | 每个 chunk 都采样 |

资源所有权也必须贯穿这条路径。调度器已预留页，并不自动意味着 Runner 已完成写入。异步执行期间，输入缓冲区、页表缓冲区和 KV 页都要保持有效，直到相关设备工作不再使用它们。取消请求可以阻止结果交付，但不能使在途 kernel 所读写的内存立即被另一个请求重用。S05 的 generation 和延迟释放示例正是这个边界的简化模型。

### 8.1 写一个真正保存数值 K/V 的最小 Runner

下面的完整程序把 TinyAttentionBackend 参数用于逐请求增量计算。它不是替换现有 Engine 接口，而是一个独立参考：`append` 接收某请求的新 token，投影新增 K/V 并拼到该请求自己的缓存，返回新片段最后位置的 logits。代码没有位置编码、多层结构和分页，但它确实保存了数值 K/V，能够验证“只投影新增 token”与“完整上下文重算”的等价性。

```python
import math
import numpy as np
from labs.serving.backend import TinyAttentionBackend

class IncrementalTinyRunner:
    def __init__(self, model):
        self.model = model
        self.states = {}

    def append(self, rid, token_ids):
        if not rid or not token_ids:
            raise ValueError("identity and new tokens required")
        if any(type(t) is not int or not 0 <= t < self.model.vocab_size
               for t in token_ids):
            raise ValueError("token outside vocabulary")
        x = self.model.embedding[token_ids]
        new_k, new_v = x @ self.model.k, x @ self.model.v
        previous = self.states.get(rid)
        if previous is None:
            k, v = new_k, new_v
        else:
            k = np.concatenate((previous["k"], new_k), axis=0)
            v = np.concatenate((previous["v"], new_v), axis=0)
        q = x[-1] @ self.model.q
        scores = q @ k.T / math.sqrt(x.shape[-1])
        p = np.exp(scores - np.max(scores))
        p /= p.sum()
        logits = ((p @ v + x[-1]) @ self.model.out).tolist()
        # 数值成功后才提交新状态；前面的验证失败不会部分追加。
        self.states[rid] = {"k": k, "v": v}
        return logits

    def release(self, rid):
        return self.states.pop(rid, None) is not None

model = TinyAttentionBackend()
runner = IncrementalTinyRunner(model)
a1 = runner.append("A", [1, 2])
b1 = runner.append("B", [7])
a2 = runner.append("A", [3])
b2 = runner.append("B", [4, 5])
np.testing.assert_allclose(a1, model.logits([1, 2]), atol=1e-12)
np.testing.assert_allclose(a2, model.logits([1, 2, 3]), atol=1e-12)
np.testing.assert_allclose(b1, model.logits([7]), atol=1e-12)
np.testing.assert_allclose(b2, model.logits([7, 4, 5]), atol=1e-12)
assert runner.states["A"]["k"].shape[0] == 3
assert runner.states["B"]["k"].shape[0] == 3
assert runner.release("A") and not runner.release("A")
assert "B" in runner.states
assert runner.release("B") and not runner.states
print("interleaved numeric KV matches full-context reference")
```

`states` 是请求 ID 到缓存对象的映射，所以甲乙交错执行不会共用历史。缓存中的 K/V 数组已经是投影结果，不是 token ID 列表。新的 x 只包含新增 token，因此 `x @ Wk` 和 `x @ Wv` 的行数等于本次新增数量；旧 K/V 不再投影。这是增量计算真正节省的部分。这里使用 concatenate 每次复制旧数组，存储效率仍很差，分页和预分配将解决另一层问题。

程序最后才把新缓存写回 states，意味着输入校验失败或前向中途异常不会把旧状态部分推进。但这不自动提供强并发事务：若两个线程同时为同一 rid 调用 append，都可能读到同一旧状态，后写者覆盖先写者。当前参考明确要求每个请求串行追加。真实 Runner 需要调度器保证没有冲突在途写入，或者在状态更新时比较提交代次。

为什么可以一次追加多个 token，只计算最后 query？因为这个单层参考中 K/V 直接来自 token embedding，前面的 token 不依赖本层 attention 输出来形成同一层的 K/V。对多层 Transformer，新 token 的下一层输入需要本层所有新增位置的输出，不能在第一层就只保留最后位置。你若把这个优化直接复制到多层模型，会缺少中间 token 在高层的 K/V。这是“能跑单层”与“实现完整 decoder”之间一个非常具体的差别。

最后一次 release 返回 False，用来支持重复清理时明确报告“已经不存在”，而不是二次释放其他对象。真实 GPU 系统还要考虑异步使用完成，不能因为 Python 字典删除了对象就认定设备没有引用。这个同步 CPU 程序只验证请求隔离、数值缓存和幂等清理，不验证 CUDA 生命周期；把证据边界写清楚，才不会用简单测试遮蔽复杂问题。

## 9. 第九课：执行、采样和回传为什么可以分开

模型前向通常产生隐藏状态或 logits，采样还需要每请求的 temperature、top-k、top-p、随机状态、允许 token 集合和停止策略。把所有这些逻辑混在模型层中，会使模型数学与服务策略紧密耦合。分开的接口允许模型前向复用，也允许结构化解码或推测解码在采样阶段提供不同约束。

分开不意味着中间状态可以没有身份。假设 execute 返回后，下一轮计划已经重排请求；sample 若使用新的请求顺序解释旧 logits，就会把甲的分数交给乙。中间结果应保存与执行提交一致的请求映射、有效行数和代次。一个裸 logits 张量只有 shape，没有业务身份，不能独自承担跨异步阶段的契约。

对于流水线并行，前面的 rank 可能只返回隐藏状态给下一阶段；对于张量并行，每个 rank 可能持有部分词表 logits，需要按实现约定聚合或分布式采样；对于普通单卡，采样可以直接在设备上完成，回传少量 token ID。无论走哪种路径，上层必须知道哪些请求实际产生了 token，哪些只推进了缓存，哪些被取消或出错。

本地 Engine 为教学便利，在 Python 中得到完整 logits list，再调用采样函数，并把生成 token 放进 Request.output。这个过程同步、易观察，也使异常清理容易理解。它不能代表高性能框架会把完整词表分数每步都拷到 CPU。读源码时应特别找 `.cpu()`、`.item()`、同步事件和异步拷贝，它们决定“Python 方法返回”到底意味着什么。

一个结果对象最好同时携带值与状态，例如请求 ID、提交 ID、输出 token、是否结束、错误、设备完成标记。不要把空列表同时用来表示“没有生成”“请求取消”“执行失败”和“输出 token 恰好为空”。含义过载会让调度器无法决定应继续、重试还是释放资源。明确状态通常比节省一个小字段更重要。

### 9.1 从同步返回走向异步提交：需要多出哪些状态

同步函数返回 logits 时，调用者通常可以认为本次 CPU 数值计算已完成。设备异步执行则不同：主机函数返回可能只表示命令已提交。于是接口应区分提交句柄与最终结果，句柄关联请求映射、资源引用和完成事件。上层不能把“已拿到句柄”当成“KV 已经可以被其他 stream 读取”，除非依赖关系明确保证了顺序。

一个最小的异步执行记录可包含 submit_id、request_generations、input_buffer_id、page_handles、completion_event 和 result_location。这里的 request_generations 防止旧结果进入重建请求；page_handles 防止页号复用后旧写入覆盖新所有者；input_buffer_id 防止同一输入区域在设备读取前被下一轮改写。三个代次或所有权问题相似，但作用对象不同，不能只增加一个全局布尔 busy 就以为全部解决。

双缓冲是一种常见直觉：一块输入供设备使用，另一块由 CPU 准备下一轮，交替切换。可是“两块”本身不是正确性证明；如果设备执行比 CPU 准备慢，CPU 再次绕回第一块时仍可能覆盖在途数据。每块缓冲区都需要完成状态，准备方只能取得已经不再使用的块。缓冲数量影响吞吐与内存，所有权协议决定正确性。

从 Worker 到 Executor 的结果传递也可能发生错误。若 Worker 已执行但网络响应丢失，上层重发相同命令可能重复推进请求。服务端应通过提交 ID 识别重复命令，或者把执行设计成可恢复的明确协议。纯本地教学 Engine 没有这类分布式重试，因此本章只解释所需状态，不声称当前代码已经处理远程恰好一次执行。

调试异步问题时，把日志按提交 ID 聚合，而不只是按墙上时间排序。不同进程时钟可能不完全一致，同一设备上的不同 stream 也不能仅凭主机日志先后推断依赖。你需要看到谁等待了谁的事件、哪块内存被谁持有、完成通知由哪个线程处理。没有这些信息的“偶尔出错”日志很难转化成可修复的因果链。

## 10. 第十课：编译、融合与 CUDA Graph 是三类不同工作

编译器可以把模型或算子表达转换为更高效的执行形式，融合可以把多个中间步骤合并，CUDA Graph 则记录一组设备工作及其依赖并重复提交。三者可以组合，但不能当作同义词。即使某个 kernel 已经编译得很好，很多很小 kernel 的主机提交开销仍可能明显；图回放针对的是重复提交路径中的一部分开销。

PyTorch 官方 CUDA 语义文档说明，图回放复用捕获时的 kernel 参数，指针参数意味着相同的内存地址，因此输入通常通过复制新内容到长期存活的缓冲区更新，而不是简单把 Python 变量重新绑定到新张量。图是否适合某条路径，还受形状、控制流和所调用操作的限制。这里采用的是官方机制说明，不假定某一框架的所有模式都能捕获。[官方说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)，核查日期为 2026-11-01。

考虑一个固定容量八行的输入缓冲区，本轮只有五个有效 token。Runner 可以把五个 token 写到前五行，其余三行作为 padding，再用有效范围和安全的映射确保额外行不会参与用户输出或写坏 KV。下一轮只有两行时，第三到第五行保留旧值并不自动构成错误；真正的要求是这些失效行不会再被当成有效请求执行。

为了验证这一点，测试应给 padding 放入可识别的毒值，并检查有效输出完全不变。如果你的“修复”只是每轮把整个大缓冲区清零，可能掩盖了有效范围错误，也增加额外带宽成本。正确实现可以选择清理，但清理不能替代边界契约。对自定义 kernel，padding 的 slot mapping 还必须使用后端明确支持的无效哨兵，而不是擅自假定负一安全。

多种真实批次形状可能映射到一组捕获 bucket，例如把三行映射到四行容量。bucket 越密，padding 浪费通常越少，但捕获时间和长期缓冲区占用可能增加。这个权衡属于 Runner 的资源与执行策略，不是模型结构本身。不要把冷启动的编译、图捕获、权重加载时间混在稳态单步延迟里，再得出后端快慢结论。

本章不提供未经运行的 GPU 性能数字，也不要求你立即在共享服务器捕获大模型。CPU 先完成“容量固定但有效行变化”的映射实验，S12 再安排有时间上限、单设备、可复现输入的真实框架实验。自己的 Win11 机器可先用 CPU 跑本章；真实 CUDA 路径需要独立确认对应系统、驱动和安装组合，不能仅凭显卡型号认定所有框架安装方式都支持。


### 10.1 为什么“换一个 attention backend”仍然需要模型级对照

假设两个后端都宣称计算缩放点积 attention。第一个接受连续 K/V，第二个接受分页 K/V，二者的数学目标相同，但调用方必须提供不同布局和不同的有效长度。若把连续数组按分页轴顺序解释，矩阵乘法可能仍得到形状正确的输出。只有把相同逻辑序列放进两种存储方式，再还原到同一参考问题，才能判断后端替换是否正确。

还要检查位置编码发生在哪一侧。有的路径在模型投影后先对 Q/K 应用位置旋转，再写缓存；如果另一层误以为缓存里还是未旋转的 K，又旋转一次，数值会错。接口文档应该明确“缓存存的是哪一个中间量”，而不只是写 dtype 与 shape。缓存内容的语义身份比数组类型更丰富。

量化缓存进一步增加 scale、零点或分组信息。主数据页与 scale 元数据必须按同一个请求、层、head、块或分组规则对应。只拷贝主字节而忘记对应 scale，通常不会触发越界，却会系统性改变数值。选择后端时必须把量化格式作为契约的一部分，不可仅按“都叫八位”判断兼容。

最后做模型级检查：固定小模型和输入，比较多个层的关键中间值与最后 logits，并覆盖 prefill、decode、分块和分页边界。单算子测试通过后，集成层仍可能把正确算子接错。反过来，一次生成文本相同也不能替代算子误差检查，因为 argmax 在一段数值区间内保持不变，很多错误暂时不会改变可见 token。

## 11. 第十一课：四个故障现场，从症状回到契约

**现场一：单请求正确，多请求输出突然相似。** 先固定参数与输入，把两个完全不同的短上下文分别运行，再打包运行。若独立结果正确而打包错误，优先检查 owner mask、offsets 和最后行映射。不要先怀疑模型质量。将第二个请求的输入改变后，第一个请求的 logits 若明显变化，说明隔离契约被破坏。显式打印一个六乘六 allowed 矩阵通常比看几百行框架日志更有效。

**现场二：完整 prefill 正确，chunked prefill 错误。** 将相同上下文分别一次完成、切成二加三、逐 token 处理，对比每个逻辑位置的 Q/K/V 与最终 logits。若只在第二块开始出现差异，重点检查 position 是否错误地从零重启，或者 sequence length 是否填成 chunk 长度。若位置正确而历史缺失，再检查 block table 和缓存写入。按最早出现差异的中间量排查，比只比较最后生成文本更敏感。

**现场三：取消后另一个请求偶尔得到奇怪输出。** 使用带代次的请求 ID 和物理页日志，记录提交、完成、取消和重分配的先后关系。若旧执行仍在写页，新请求已经复用该页，修复点在资源生命周期，不在 sampling。用同步执行可以暂时让问题消失，但这只能证明与并发有关，不能当作最终根因。必须明确设备完成事件何时允许页进入 free 集合。

**现场四：eager 正确，图回放总是重复第一轮结果。** 检查是否把捕获输入变量重新赋值为新张量，而没有更新图实际引用的旧缓冲区。记录 storage 地址、有效形状和复制发生的 stream 依赖。若地址没变但结果仍旧，检查数据复制是否在回放读取之前完成。不要通过每次重新捕获回避问题，那会把可复用执行路径重新变成昂贵冷启动。

这四个现场分别对应隔离、逻辑位置、资源生命周期和静态缓冲区更新，不能使用同一种“加同步、清零、换 seed”操作统一解决。排查报告应写出一个能够反驳自己假设的实验。例如怀疑 mask，就保留其他条件只替换显式参考 mask；怀疑生命周期，就延迟复用但保持计算不变。能够设计反证实验，是从会调用框架走向会修改框架的关键能力。

## 12. 第十二课：十五道分级练习与完整答案

**练习一，区分位置。** 块大小四，页表 `[7,1,6]`，本轮 token 逻辑位置九，放在 flat 第三行。写出 position、逻辑块、块内偏移、物理页和扁平物理槽。答案依次为九、二、一、六、二十五。flat 行号三只用于本轮输入索引，绝不能替代 position 九。物理槽二十五来自六乘四加一，不影响模型的位置编码。

**练习二，手算打包。** 三请求原 computed 为二、零、七，新 token 数为二、三、一。答案 offsets 为 `[0,2,5,6]`，positions 为 `[2,3,0,1,2,7]`，执行后 seq_lens 为四、三、八。最后一行索引为一、四、五。如果第二个请求只是 prompt 的中间块，虽然它有最后一行，也不一定允许采样；采样还需要知道是否完成全部已知上下文。

**练习三，构造非法计划。** 用 Part("A",0,(1,)) 和 Part("A",1,(2,)) 同时传给 pack。答案应抛 ValueError，而不是默默合并。因为函数契约是一请求一段，重复 ID 可能隐含冲突状态。若想支持同一请求多段，应设计显式连续性校验与合并规则，不能在不改变契约的情况下接受任何重复。

**练习四，为什么不能只用下三角。** 对长度二和一的两个请求，普通下三角第三行允许读取前三行；正确第三行只允许读取自身。因此输入总长三时错误有两个跨请求可见元素。答案不是“因为 batch 不同”，而是写出允许关系 `owner相同且key位置不晚于query`，说明物理排序不能建立语义上的历史关系。

**练习五，找出 softmax 轴错误。** 若对整个 scores 调用无 axis 的 max 与 sum，减一个全局最大值本身在精确算术下仍可用于各行归一化，但对所有元素共同求和会使每行概率和不再为一，且请求之间相互影响。完整修复是逐行 max、逐行 sum，保留维度进行广播。要精确地区分“共享平移常数”与“共享归一化分母”，不要把两者都笼统称为错。

**练习六，最后行抽取。** 长度为一、四、二时 offsets 为零、一、五、七，正确最后行是零、四、六。用 offsets 的左端会得到零、一、五，第一请求恰好正确，另外两个错误。参考解法为 `np.cumsum(lengths)-1`，并断言所有 lengths 大于零；存在空成员时减一可能索引到前一个请求的最后行，因此应先拒绝或单独处理空成员。

**练习七，增加一个请求不应改变另一个请求。** 保留 packed_logits 程序，把 contexts 改成只有甲，得到甲 logits；再加入乙和丙，对比第一行。答案在给定 CPU 参考容差内相同。这个测试检查请求隔离与映射，不验证所有 GPU 的 bitwise 确定性。若差异很大，先检查 mask；若差异很小且只在更换后端时出现，再研究归约顺序和容差。

**练习八，计算显式分数内存。** 总 query/key 长度一千、分数用 float32，矩阵元素一百万，单分数矩阵约四百万字节；布尔 mask 另有存储，概率和中间结果还会继续占用。总长度翻倍，单分数矩阵变成四倍。答案必须说明这只是某些中间张量的估算，不是整个程序峰值内存，也不能直接推断融合 attention 的实际分配。

**练习九，设计 padding 契约。** 固定容量八，本轮有效行三。完整答案包括：有效 token 数单独保存；请求 offsets 只描述真实三行；sample rows 只指向有效行；额外行不拥有真实请求身份；缓存写入使用后端支持的无效映射或完全跳过；输出只取有效结果；下一轮数据更新遵守执行依赖。只说“补零”不是完整答案，因为 token 零也可能是合法词表项。

**练习十，解释当前 Engine 的边界。** 它会把多个请求放入同一调度轮，并统一推进模拟时间，但 TinyAttentionBackend.logits 仍逐请求调用并重算完整上下文。所以 runner_swap 能验证接口替换和状态机继续运行，不能证明 GPU 连续批处理性能。完整答案要同时承认它真实计算 attention 数值，也承认它没有真实数值分页 KV 接入，避免把“玩具”误解为所有结果都伪造。

**练习十一，划分异常责任。** 输入 token 超出词表，应该在提交校验或 Runner 输入校验时明确拒绝；backend 返回 NaN，采样验证应报数值错误并触发请求清理；Worker 进程消失，Executor 应报告执行失败并使在途请求进入明确终态或可恢复状态。答案不能把三种异常都转成空字符串，因为上层无法据此决定重试、资源释放和用户响应。

**练习十二，写最后行索引函数。** 参考完整实现见下面。它同时拒绝 bool，因为 Python 的 bool 是 int 子类，但“长度为 True”不是本接口接受的规范输入。总长度与最后索引是两个不同结果，空批次返回空索引与总长零，调用者随后可以跳过执行。

```python
def last_rows(lengths):
    total = 0
    rows = []
    for length in lengths:
        if type(length) is not int or length <= 0:
            raise ValueError("positive integer lengths required")
        total += length
        rows.append(total - 1)
    return rows, total

assert last_rows([1, 4, 2]) == ([0, 4, 6], 7)
assert last_rows([]) == ([], 0)
for bad in ([0], [-1], [True], [1.5]):
    try:
        last_rows(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid length accepted")
print("last-row mapping checked")
```

**练习十三，解释为什么只计算最后 query 可以不加未来 mask。** 当前 backend 的输入恰好是全部已知上下文，最后 query 的逻辑位置最大，所以所有 K 都不在它之后。若输入里混入 padding、其他请求或未来训练 token，这个前提立刻失效。答案必须写明前提，不能推广成“推理永远不需要 causal mask”。

**练习十四，分析重新绑定输入。** 捕获时图引用缓冲区甲，之后 Python 执行 input=新张量乙，只改变变量所指对象，不改变图保存的甲地址。正确思路是保持甲存活，把乙的数据复制进甲，并确保复制先于回放读取。这里还要验证 shape 与 dtype 合约；不能靠复制操作自动修正不兼容形状。本题是机制推理，未宣称在本环境执行了 CUDA Graph。

**练习十五，设计完整验收矩阵。** 最低包含长度一、非整块长度、不同长度多请求、非零起始位置、请求重排、取消后槽位复用、空批次、非法 token、padding 毒值，以及独立完整前向与增量前向对照。每项应明确预期：拒绝、跳过、容差相等或资源保持。只列输入而没有判定条件的“测试计划”，无法告诉你修改是否正确。

## 13. 第十三课：把练习组织成一个能够审阅的 Runner 工程

第一阶段只做布局。将 Part、pack、last_rows 放进你自己的练习目录，写一份输入输出示例和索引表。验收时拿掉模型，要求同学或未来的自己仅凭元数据就能恢复每个请求的 token、逻辑位置和本轮范围。这个阶段没有任何 GPU 依赖，适合在个人电脑反复练习，错误也容易定位。

第二阶段接入 packed_logits。保留逐请求 TinyAttentionBackend.logits 作为独立参考，比较单请求、多请求、重排和不同长度。使用固定小权重与合成 token，先检查 logits，再检查采样 token。因为两个 logits 只要接近 top-1 边界，极小差异就可能改变 token，直接比较生成文本会失去最早的数值证据。

第三阶段加入真正增量的 K/V。每个请求维护自己的 K/V 数组，新 token 只追加新投影，最后 query 读取对应历史。请求结束时删除其状态，取消后不可再交付旧结果。完整模型的增量项目见仓库 projects 中的运行时实现与项目验收；本阶段自己的版本先保持单头、无位置编码，再加入位置编码，逐步与完整前向比较，不要一次加入所有优化。

第四阶段把连续数组缓存替换为页表访问。逻辑序列读取必须通过页表还原，与连续缓存结果一致；非整页尾部和跨页追加是重点。共享前缀时加入引用计数和写时复制，确保一个分支追加不会改变另一个分支的历史。这里才真正连接 S03 的资源模型与 attention 数值，不能只检查页数量就宣称缓存正确。

第五阶段阅读固定上游路径。vLLM 快照 `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` 的 [GPUModelRunner](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/worker/gpu_model_runner.py) 是入口。先固定单 GPU、普通 dense decoder、无多模态、无 LoRA、无推测解码的简单模式，记录 execute_model 的输入从哪里来、输入准备如何使用调度结果、模型调用产生什么、sample_tokens 如何使用保存状态。S09 与 S17 会把这个阅读任务展开成逐步工作簿。

读到陌生字段时，不要立即跳到搜索引擎找一句定义。先找它在哪里赋值、在哪里读取、shape 由哪个数量决定、默认值表示什么、错误值会破坏哪条不变量。把这些写成字段合同，比摘抄注释更有效。如果一个字段只在推测解码分支使用，先标注该分支并暂时绕开，等基本路径完整后再回来扩展，而不是把“不看所有分支”等同于“永远忽略”。

工程报告必须分开列出正确性证据与性能证据。打包参考和独立调用一致，证明在这些输入下数学映射正确；CPU 耗时降低，不证明 GPU 更快；固定 GPU 实验中吞吐提高，也不能证明取消和内存隔离正确。每项结论后写环境、输入、预期和实际结果，最后标记没有运行的路径。这样的报告才能成为以后修改真实框架的可靠起点。

### 13.1 一份代码评审应该怎样提出具体问题

评审打包器时，先要求作者指出每个数组的长度由哪个数量决定。flat_ids 长度是本轮总 query 数，offsets 长度是请求数加一，seq_lens 长度是请求数，block tables 的行数可能是请求槽位数而列数是最大逻辑块数。看到多个数组都恰好长度三，不应默认它们可以相互替代；在不等长输入下它们的区别才会显现。

评审数值路径时，要求作者说明缩放、mask、归一化与输出投影的顺序，并提供一个可以手算的两维样本。仅有随机张量 allclose 可能让两个实现共享同一个错误，例如参考和优化版都把 mask 比较方向写反。手算样本与独立实现互相补充，才能降低“两个程序一致地错”的风险。

评审缓存路径时，询问追加前后哪些数组被复用、哪些被复制、谁负责清理、异常发生后原状态是否仍可使用。对本章 IncrementalTinyRunner，concatenate 产生新数组，旧状态直到最终赋值前仍保留；对原地写入的优化实现，异常恢复就更复杂，需要已预留范围与提交进度标记。性能优化改变了写入方式，也会改变失败语义。

评审性能结论时，要求区分输入准备、主机提交、设备执行、采样和回传。若一次优化减少 CPU 数组构造却增加设备 padding，端到端是否更快取决于原瓶颈和请求分布。不能看到某个内部函数缩短就宣布整个服务吞吐提高。最有说服力的报告是保留正确性对照，同时对同一工作负载测量完整路径，并给出误差、预热与停止条件。

评审完成后，应把发现写成可执行的验收条件，而不是“代码不够优雅”。例如“长度为一、四、二时 sample rows 必须是零、四、六”“乙输入改变不应改变甲的参考 logits”“取消甲后乙缓存长度保持不变”，都能帮助作者准确修复。你将来贡献真实推理框架时，清楚描述契约和最小反例，往往比一次提交大量重构更容易被维护者审阅。

掌握门槛：你能不看答案构造不等长批次的 tokens、positions、offsets、seq_lens、block table、slot mapping 和 sample rows；能解释每个字段为什么不能被另一个字段替代；能用逐请求参考定位跨请求污染、分块位置错误和异步释放错误；能沿固定源码追踪一次执行的身份与资源。达到这个门槛后，再进入 [S07 采样与流式输出](07-sampling-streaming.md)，把数值结果安全地变成用户可见的增量输出。
