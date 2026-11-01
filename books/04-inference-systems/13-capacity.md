# S13 容量模型：把每个字节、每个拥有者与每个峰值时刻算清楚

先修 S03 的分页 KV、S10 的并行以及基础卷的字节单位。容量规划回答三个逐渐严格的问题：按架构公式估算是否可能放下；按具体框架的布局和预分配策略是否可能启动；在给定请求长度、并发与临时峰值下是否能够稳定服务。三个问题的证据不同。计算器输出 True 只回答它建模的那部分算术，不能替代加载、执行和负载验证。

本章先用纸笔建立账本，再读本仓库的估算器，接着处理分页、共享、并行不均衡、混合结构和临时峰值，最后形成有停止条件的校准项目。所有代码默认在仓库根目录运行，只用 Python 标准库；表格中的硬件预算是人为设定的示例，不是对你的 5070 Ti 或公司 H100 当前可用显存的测量。实际设备容量、空闲量和可用范围应在 S15 的检查中取得。

## 1. 第一课：先辨认单位，再辨认内存属于谁

byte 是八个 bit。模型说“十六位权重”，通常是在说明每个编码元素需要十六个 bit，即两个 byte；设备报告几 GiB，则是在说明容量。GB 使用十的九次方 byte，GiB 使用二的三十次方 byte。把两者都简写为 G 会让报告产生约百分之七的偏差，在接近容量边界时足以改变结论。

八十亿个参数，每个两 byte，裸载荷是一百六十亿 byte，换算约十四点九零 GiB。这里的“裸”很重要：它没有包含量化元数据、对齐、额外权重副本和运行时状态。也没有回答参数究竟在 CPU、GPU、磁盘还是多个 rank 上。容量表的每一行都应该带有位置、拥有者、生命周期和单位，否则同一个数字很容易重复计算或漏算。

```python
def units(nbytes):
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("nonnegative integer bytes required")
    return {"bytes": nbytes, "GB": nbytes/10**9, "GiB": nbytes/2**30}

print(units(8_000_000_000*2))
assert units(2**30)["GiB"] == 1
assert units(10**9)["GB"] == 1
```

你应看到 GB 为十六，而 GiB 约十四点九零。函数要求整数 byte，拒绝把“十六 GiB”这个数十六直接作为 byte 传入。真实容量工具最好也在输入名称中写明单位，如 runtime_gib_per_rank，而不是模糊的 runtime。函数内部统一用 byte，展示时再转换，能够减少混算。

再区分“总容量”“当前空闲”“允许本任务使用”“框架预算”。一张卡物理容量不等于本任务可独占容量；共享服务器上其他任务、驱动和已有上下文都可能占用一部分。即使当前空闲很多，也不意味着你被允许把它全部占满。课程实验使用明确的设备和限制，以一次任务的授权预算作为账本上限。

容量单位也不能与带宽单位混用。GB/s 是每秒十亿 byte，Gbit/s 是每秒十亿 bit，两者相差八倍。S11 迁移一 GiB KV 的时间需要把容量转换成 byte，再除 byte/s。把网络标称数字不换单位直接代入，会得出错误的传输时间和错误的 PD 盈亏判断。

## 2. 第二课：建立六类内存账，防止一项省下的钱在另一项花掉

第一类是权重，包括 embedding、attention 与 MLP 矩阵、norm 参数、路由器、专家、输出头等。参数是否共享会影响实际载荷；两个名字指向同一块 storage 时，不能直接按名字各算一份。相反，加载过程可能临时存在原始权重、转换后的权重和设备副本，运行时只留一份并不代表启动峰值只有一份。

第二类是请求状态，最常见的是 KV。它随有效历史长度、请求数和层结构变化，分页和前缀共享改变物理占用方式。线性注意力、状态空间模型或混合模型可能保存固定大小递归状态、卷积窗口和检查点，不能统一称为“每 token 两份 K/V”之后套同一公式。

第三类是激活与工作区。prefill 会处理多个输入位置，某些算子需要中间张量；高效 attention 可能避免完整注意力矩阵，但不意味着没有临时空间。矩阵乘法算法、排序、采样、通信重排也可能需要 workspace。chunked prefill 能约束一次处理的 token 数，却不能自动减少请求最终要保留的全部 KV。

第四类是通信缓冲区。TP、EP、PP、PD 可能需要收发、拼接、重排或暂存数据。缓冲区有时被复用，有时必须双缓冲以便重叠计算。只把权重和 KV 除以并行度，忽略通信副本，会高估可用容量。第五类是框架与执行设施，例如 CUDA context、编译缓存、图捕获使用的持久输入输出、调度器设备元数据。第六类是分配器保留、对齐与碎片，它让“活跃张量字节数”与“设备显示已占用”不完全相同。

| 对象 | 常见拥有者 | 生命周期 | 主要随什么变化 |
|---|---|---|---|
| 模型权重 | worker / model runner | 加载到卸载 | 参数与表示格式 |
| 请求 KV | 页池，外加请求或缓存引用 | 首次计算到最后引用释放 | 历史、并发、共享 |
| prefill 工作区 | runner / backend | 一轮或复用池 | 批次 token 与算法 |
| 通信缓冲 | 并行组 / connector | 一次传输或持久复用 | 消息、rank 与重叠 |
| 图输入输出 | graph runner | 捕获到销毁 | 捕获规格与缓冲设计 |
| 分配器保留 | 运行时 | 可能跨请求保持 | 历史峰值与分配模式 |

这张表的价值是让优化有明确落点。权重量化减少第一类，但可能增加反量化工作区；降低 chunk 减少某些临时激活，却可能增加轮数和调度开销；缓存热前缀减少重复计算，却可能延长 KV 的保留时间。不要使用“优化显存”这一个模糊词，而应指出优化哪项、增加哪项、在哪个时刻测量。

## 3. 第三课：从单层张量形状推导标准 KV

对某一层、某一请求、一个历史位置，K 有 Hkv 个头，每头 D 个元素，V 同样大小。每元素 b byte，则单层单位置状态为二乘 Hkv 乘 D 乘 b。若 L 层形状相同，单请求 T 个位置，总字节为二乘 L 乘 Hkv 乘 D 乘 b 乘 T。并发 C 个完全独立、相同长度的请求，再乘 C。

这不是需要死记的魔法公式。每个因子都来自张量的一个维度：二代表 K 与 V；层数代表每层独立状态；KV 头数不是 query 头数；head dimension 是每个头的通道；元素字节由存储 dtype 决定；长度只计真正保留的状态位置。如果模型使用 GQA，query 头更多但共享 KV，容量应使用 KV 头数。

手算例子取三十二层、八个 KV 头、每头一百二十八维、两个 byte。单 token 为十三万一千零七十二 byte，即一百二十八 KiB。三万二千七百六十八个位置是四 GiB；十三万一千零七十二位置是十六 GiB；一百零四万八千五百七十六位置是一百二十八 GiB。它们都是标准、完整、未分片 KV 的算术值，尚未加入权重和工作区。

```python
def kv_bytes(layers, kv_heads, head_dim, element_bytes, tokens):
    values = (layers, kv_heads, head_dim, element_bytes, tokens)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("nonnegative integers required")
    return 2*layers*kv_heads*head_dim*element_bytes*tokens

per_token = kv_bytes(32, 8, 128, 2, 1)
assert per_token == 128*1024
for length in (32768, 131072, 1048576):
    print(length, kv_bytes(32, 8, 128, 2, length)/2**30, "GiB")
assert kv_bytes(40, 4, 128, 2, 65536) == 5*2**30
```

函数允许零个 token，结果为零；真实模型的层数、头数等通常应是正整数。这里将通用乘积与配置验证分开，是为了明确容量公式本身与模型是否合法不是同一件事。实际导入模型 config 时，必须额外验证缺失字段、别名、结构类型和 head dimension 的定义。

还有一个常被忽视的边界：输出 token 已知，不等于它已经写入 KV。S01 讲过采样得到的最新 token 往往要在下一轮作为输入后才进入状态。做静态最坏容量预算时，可以按总允许上下文长度保守规划；做每步实际占用解释时，则应使用已经计算或已经预留的位置，不能把已输出数量机械地当作完整 KV 长度。

## 4. 第四课：分页把连续公式变成阶梯，引用共享再改变物理数量

设每页容纳 B 个位置，请求长度为 T，则简单独占页数是向上取整 T/B。最后一页未满的位置仍占据物理空间。两个长度分别十七、三十一的请求，在每页十六位置时，各需要两页，总四页；总 token 虽为四十八，若先相加再向上取整只得到三页，低估了一页。因为两个请求的尾部不能凭空拼在同一页中。

因此，应先逐请求算页数，再求和。只有框架明确支持更细粒度打包，或两个请求确实共享相同物理页时，才能减少。S03 的 PagePool 有清晰页表与引用计数，正适合验证“逻辑页数量”和“不同物理页数量”的差别。缓存中的引用不会复制数据，但它可能阻止物理页回到 free 集合。

```python
def ceil_div(n, d):
    if type(n) is not int or type(d) is not int or n < 0 or d <= 0:
        raise ValueError("invalid page geometry")
    return (n+d-1)//d

lengths, block = [17, 31], 16
logical_pages = [ceil_div(n, block) for n in lengths]
assert logical_pages == [2, 2]
assert sum(logical_pages) == 4
assert ceil_div(sum(lengths), block) == 3
used_tokens = sum(lengths)
reserved_tokens = sum(logical_pages)*block
print({"pages": logical_pages, "reserved_tokens": reserved_tokens,
       "tail_waste_tokens": reserved_tokens-used_tokens})
tables = {"A": [7, 2], "B": [7, 4]}
assert sum(map(len, tables.values())) == 4
assert len(set(page for table in tables.values() for page in table)) == 3
```

最后两行给出共享例子：A 与 B 都引用物理页七，尾页分别二与四，逻辑上四个页表项，物理上只有三页。要把该优势写进生产预算，还需要证明共享前缀长度、身份和实际命中率。不能默认所有并发请求都拥有完全相同前缀，再把最乐观的共享结果当容量保障。

写时复制还会产生短时峰值。两个请求共享尾页，其中一个要追加时，可能先申请新页、复制旧有效内容、切换页表、减少旧引用。旧页与新页会短暂同时存在。如果预算恰好只够稳定态，COW 就可能失败。S03 的 needed_for_append 与原子分配正是在追加前把这种需求算出来。

框架常在启动时预分配整个 KV 池。此时增加请求不会让设备已占用字节线性增加，因为物理内存早已申请；变化的是页池内部 free/used/cached 的归属。观察 nvidia-smi 曲线平坦，不能推断 KV 不占空间。正确的观测应同时包含池总页数、空闲页数、请求引用、缓存引用、预留页和在途页。

## 5. 第五课：长度预算、并发预算和调度预算是三个不同旋钮

max model length 通常限制单请求上下文范围；并发序列上限限制同时活跃多少请求；每轮 token budget 限制一次调度多少输入位置。三者不能互相替代。一个很小的每轮预算可以让长请求分很多次 prefill，但最终仍可能保留很长的 KV。一个很大的并发上限也不意味着页池足够让所有请求走到最大长度。

设页池一百页，每页十六位置。十个请求各已保留八十位置，合计五十页。若每个还允许再生成一百位置，最坏完成长度一百八十需要十二页，十个请求总一百二十页，超过池容量。此时系统可能动态抢占或拒绝新增工作，也可能提前做更保守的 admission。两种策略有不同吞吐与延迟，不能只看当前页够用就承诺未来不会停顿。

将输出长度上限写入静态 admission 会过于保守，因为大多数请求可能提前 EOS。只按实际增长分配则需要有可靠的抢占、回收和失败语义。课程 Engine 展示后一类动态过程的一部分；生产系统还需要处理截止时间、优先级和租户。容量规划应该明确自己是在预测平均需求、保证硬上限，还是为允许抢占的系统估算常见运行区间。

下面实现保守的页数 admission，仅表示“为每个请求的输入加最大输出预留足够页”。它不使用 prefix sharing，不模拟推测槽位，不保证 runtime workspace，因此结果仍是一个明确但局部的规则。

```python
def reserve_pages(prompt_tokens, max_new_tokens, block_size):
    if any(type(x) is not int or x < 0 for x in (prompt_tokens,max_new_tokens)):
        raise ValueError("nonnegative token counts required")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("positive block size required")
    return (prompt_tokens+max_new_tokens+block_size-1)//block_size

def admit(existing_reservations, candidate, total_pages):
    needed = reserve_pages(*candidate)
    free = total_pages-sum(existing_reservations)
    return {"needed_pages":needed,"free_pages":free,"admit":needed<=free}

assert reserve_pages(80,100,16) == 12
print(admit([12]*8, (80,100,16), 100))
assert not admit([12]*8, (80,100,16), 100)["admit"]
```

代码把 prompt 与 max new 相加作为保守上界，可能比真实最后一步 KV 多一个位置，这种保守性应写出来。如果项目追求精确 admission，就必须把首 token 所有权、EOS、推测生成、回滚和 padding 的协议统一建模，不能在容量章节随意减一而没有状态证明。

多租户场景还应区分全局空闲与本租户配额。一个租户即使发现池还有空页，也未必能把其他租户的保留份额全部占用。配额不是物理拥有者，它是 admission 的政策约束；真正分配后仍需要同一套页表与引用不变量。把政策账本和资源账本分开，才不会出现“配额归还了但物理页仍在传输”的虚假可用。

## 6. 第六课：逐 rank 记账，复制对象不能除以卡数

并行布局应从“对象如何分布”开始。TP 的矩阵可能按行或列切分，norm 参数可能复制；PP 把层放在不同 stage；DP 每个副本保留完整模型或完整模型分片组；EP 按专家分布，但路由器或共享专家可能另有规则。总字节除八只是一种平均值，既不能证明每卡够用，也不能描述数据怎么流动。

设四个 rank，每个都有一 GiB 复制状态，总共十二 GiB 可均分权重，以及只放在 rank 零的四 GiB 大对象。每 rank 分片权重三 GiB，前三项合起来 rank 零八 GiB，其他各四 GiB。总和二十 GiB，平均五 GiB。如果每卡预算六 GiB，总和小于二十四 GiB，却仍然无法在 rank 零放下。部署是否成功取决于最大 rank，而非平均。

```python
def rank_ledger(world_size, replicated_bytes, shards, placed_objects):
    if world_size <= 0 or len(shards) != world_size:
        raise ValueError("one shard size per rank required")
    totals = [replicated_bytes+size for size in shards]
    for rank, size in placed_objects:
        if not 0 <= rank < world_size or size < 0:
            raise ValueError("invalid placement")
        totals[rank] += size
    return totals

gib = 2**30
usage = rank_ledger(4, gib, [3*gib]*4, [(0,4*gib)])
assert [x//gib for x in usage] == [8,4,4,4]
assert sum(usage) < 4*6*gib
assert max(usage) > 6*gib
print({"per_rank_gib":[x/gib for x in usage],
       "average_gib":sum(usage)/len(usage)/gib})
```

对 KV 来说，只有当 KV 头被真实均匀切分时，才能按 TP 除。两个 KV 头配八个 TP rank，不能解释成每 rank 四分之一个头。某些实现可能复制 KV 头，另一些布局可能不支持该组合；必须读取模型实现和 backend 元数据，确认本地实际头数。估算器拒绝不整除，是保护假设，不是宣称所有框架都无法运行。

PP 也要考虑首尾不均衡。embedding 常在最前 stage，输出投影常在最后 stage；若权重绑定跨 stage，实现可能需要复制或通信。层数平均不保证参数字节平均，更不保证执行时间平均。先列 stage 的具体层与特殊对象，再分别计算权重、KV、激活和通信，最后比较最紧张的 stage。

DP 可以提升总请求容量，但不降低单副本的模型内存。prefix cache 的热度也分散到多个副本；如果路由随机，某个常见前缀未必每次命中同一缓存。容量表可以写共享的最佳和保守两种边界，却不应把跨 DP 的逻辑相同 token 误当成同一块物理 KV。

## 7. 第七课：读真实估算器，认识它明确支持和没有支持的东西

打开 `labs/serving/analysis.py` 的 `Capacity`。它使用 dataclass 保存参数、位宽、层数、KV 头数、维度、KV 元素字节、上下文、并发和 TP，再加设备预算比例、运行时固定项和权重开销比例。字段名包含 per rank 或 GiB 的地方，是为了把单位和分布假设写进接口。

核心算式如下，它直接来自仓库当前实现；片段依赖方法里的 self 和 GiB，不是独立可运行程序。

```python
weights = self.parameters*self.weight_bits/8*(1+self.weight_overhead_fraction)/self.tp
kv_per_token = 2*self.layers*(self.kv_heads//self.tp)*self.head_dim*self.kv_bytes
kv = kv_per_token*self.context*self.sequences
total = weights+kv+self.runtime_gib_per_rank*GiB
available = self.gpu_gib*GiB*self.usable_fraction
```

第一行假设全部参数使用同一位宽，额外开销按固定比例，并均匀分到 TP rank。第二行假设 KV 头整除 TP，所有层形状相同；前面的校验会拒绝不整除。第三行假设所有请求同长度、没有共享。第四行把各种未细分运行时成本合成一个固定项。第五行把物理容量乘可用比例，得到此次算术比较的预算。

这些假设适合第一轮排除明显不可能的方案，不适合精确预测 MLA、混合线性层、EP、PP 或复杂量化格式。报告主动输出 assumptions，正是为了让读者知道 True 是在什么条件下产生。weight overhead 百分之五只是输入默认值，不是对所有量化方案的经验定律；runtime 四 GiB 同样不是生产框架的普遍常数。

```python
from labs.serving.analysis import Capacity
case = Capacity(parameters=8_000_000_000, weight_bits=16,
                layers=32, kv_heads=8, head_dim=128, kv_bytes=2,
                context=32768, sequences=2, tp=2,
                gpu_gib=24, usable_fraction=.8,
                runtime_gib_per_rank=3, weight_overhead_fraction=.05)
report = case.estimate()
print(report)
assert report["kv_bytes_per_token_per_rank"] == 65536
assert report["kv_gib_per_rank"] == 4
assert report["total_gib_per_rank"] > 14
try:
    Capacity(1_000_000,16,24,2,64,2,1024,tp=8).estimate()
except ValueError as exc:
    assert "divisible" in str(exc)
else:
    raise AssertionError("unsupported KV placement must be explicit")
```

这个例子权重每 rank 约七点八二 GiB，KV 四 GiB，固定运行时三 GiB，总约十四点八二，预算十九点二，算术余量约四点三八。参数值是教学案例，未绑定某个真实模型。你应能在不运行代码时先算出数量级，再用代码检查，而不是复制一组数字后只看最后的布尔值。

注意接口校验仍是教学级的：它主要检查正数与范围，没有把所有整数型字段的类型、非有限浮点输入、真实模型配置交叉关系全部验证完。不要把一个估算器导出成外部服务后认为已具有严格输入边界。扩展项目应在外层建立 schema，明确拒绝 NaN、无穷、布尔冒充整数和未知结构，再调用内部算式。


### 7.1 反向求容量：剩余预算能装多少页

正向公式给定上下文和并发算需求，反向公式则给定预算算最多能保留多少状态。先从预算减去权重、已知固定运行时和明确工作区余量，得到可分给 KV 的字节；再除每页字节并向下取整，得到物理页数。最后根据请求长度分布、共享和预留策略决定可接纳请求数。不要直接把剩余字节除单 token 字节后再平均给请求，因为分页尾部和保留份额会改变结果。

例如 KV 预算一 GiB，每页两 MiB，共五百一十二页。每个请求最终需要三十三个位置，每页十六，所以每请求三页，最多完整容纳一百七十个这样的独占请求，余两页。若误用一 GiB 除每 token 一百二十八 KiB，再除三十三，会得到约二百四十八个请求，严重高估。差异来自大量请求的最后一页只放一个位置。

这也解释为什么页大小是设计取舍。更大页减少页表项和某些管理开销，但短请求或不齐长度会浪费更多尾部；更小页减少内部碎片，却增加页表、引用、地址转换和管理工作。容量章节能够算出碎片，不能单独决定最佳页大小，因为实际 kernel 访存和调度成本还要一起测量。

可用比例与固定余量不要重复计算。若设设备预算为总量的百分之八十，并且这百分之二十已经用于覆盖运行时与安全余量，又额外扣除同样内容，就会过度保守。相反，若百分之八十只是留给其他用户的隔离边界，预算内部的运行时仍然需要扣除。报告应写明每个预留保护的是哪个对象或哪种不确定性，避免一项成本在不同名字下出现两次。


## 8. 第八课：混合模型按层求和，固定状态与滑动窗口分别算

统一层公式方便，但模型架构并不总是统一。假设二十四层中十二层为标准 attention，另外十二层每请求保存一 MiB 固定状态。标准层使用四个 KV 头、一百二十八维、两 byte，长度六万五千五百三十六。attention 部分为一点五 GiB，固定部分十二 MiB，总计一点五一一七一八七五 GiB。并发两个且没有共享时，再乘二。

固定状态并不表示整个模型的运行时成本与长度无关。处理更长输入仍要更多计算；某些训练、回滚、推测验证或 prefix checkpoint 方案还保存额外历史状态。容量公式需要对应推理执行策略，不能只从架构名称猜测。线性注意力的状态矩阵形状、卷积窗口和 dtype 都要从具体实现得到。

滑动窗口 attention 理想上只需保留最近 W 个位置，但实际是否回收旧页、哪些层使用全局 attention、窗口边界是否需要额外对齐，取决于实现。若一半层是全局、一半层是窗口，状态应分别求和。把整个模型长度截到窗口大小，会低估全局层；把所有层都算完整长度则可能过于保守，但至少应明确保守在哪里。

```python
def layer_state_bytes(layer, length):
    kind = layer["kind"]
    if kind == "attention":
        kept = min(length, layer["window"]) if layer.get("window") else length
        return 2*layer["kv_heads"]*layer["head_dim"]*layer["element_bytes"]*kept
    if kind == "fixed":
        return layer["bytes_per_request"]
    raise ValueError("unknown state type; do not silently guess")

attention = dict(kind="attention", kv_heads=4, head_dim=128, element_bytes=2)
layers = [attention.copy() for _ in range(12)]
layers += [dict(kind="fixed",bytes_per_request=2**20) for _ in range(12)]
total = sum(layer_state_bytes(layer,65536) for layer in layers)
assert total == int(1.5*2**30)+12*2**20
print("mixed state GiB:",total/2**30)
window_layer = dict(attention,window=4096)
assert layer_state_bytes(window_layer,65536) == 2*4*128*2*4096
```

这段程序仍未做页对齐，也没有处理共享或额外 checkpoint。它的价值是把异构结构显式化：每层给出状态类型，未知类型报错，而不是继续套默认公式。下一步可以为每种类型加分配粒度和拥有者，然后按每 rank 的层集合求和。不要让一个名称为“通用容量计算”的函数偷偷假设所有模型都是标准 decoder。

MoE 还有一个独立误区：每 token 只激活少数专家，描述的是计算路径，并不自动减少常驻权重。若全部专家常驻，就要计算全部专家载荷；若 offload 或按需加载，则必须加入缓存、传输、预取与临时副本。将活跃参数量当总权重参数量，会得到极具诱惑但完全错误的容量结论。

## 9. 第九课：峰值发生在时间上，不能只把稳定态相加

容量是随时间变化的。一个请求进入时可能先 tokenization，再预留页，再运行 prefill，再产生采样工作区；图捕获或模型加载也可能有一次性峰值。若两个阶段互不重叠，可以取最大值而不是相加；若异步重叠，就必须同时计入。理解生命周期，比给所有项目随意加一个百分比更可靠。

考虑一个虚构单 rank：常驻权重八 GiB，KV 池六 GiB，基础运行时一 GiB。prefill 工作区三 GiB，decode 工作区一 GiB。如果它们严格串行且可复用同一工作区，峰值十八 GiB；如果为了重叠保留两份互不共享工作区，峰值十九 GiB。再加一个两 GiB 在途 KV 接收缓冲，重叠峰值可能二十一 GiB。哪一个数字正确，取决于执行顺序和缓冲区是否同一 storage。

```python
def timeline_peak(events):
    current, peak, records = 0, 0, []
    live = {}
    for action, name, size in events:
        if action == "alloc":
            if name in live or size < 0:
                raise ValueError("invalid allocation")
            live[name] = size
            current += size
        elif action == "free":
            if name not in live:
                raise ValueError("double free or unknown object")
            current -= live.pop(name)
        else:
            raise ValueError("unknown event")
        peak = max(peak,current)
        records.append((action,name,current))
    return peak,records,live

events = [("alloc","persistent",15),("alloc","prefill",3),
          ("free","prefill",0),("alloc","decode",1),
          ("alloc","transfer",2),("free","transfer",0)]
peak,records,live = timeline_peak(events)
assert peak == 18
print(records,"peak GiB =",peak)
overlap = [("alloc","persistent",15),("alloc","prefill",3),
           ("alloc","decode",1),("alloc","transfer",2)]
assert timeline_peak(overlap)[0] == 21
```

这里用整数 GiB 作为简化单位，代码不是设备分配器。它教的是事件账本：每个对象只有一次申请和一次释放，释放未知对象报错。真正的设备异步任务里，Python 引用消失不等于底层工作已经结束；如果流中还有读写，缓冲区要保留到完成事件。S05 与 S11 的延迟释放会直接影响峰值预算。

模型加载也有类似问题。磁盘压缩文件大小、CPU 解码数组、设备转换权重可能同时存在；加载完成后只剩最终格式，不能用最终大小解释启动 OOM。量化模型如果在加载时先建立浮点权重再转换，峰值可能比稳态大很多。是否如此应读实际 loader，不能根据文件后缀猜。

图捕获可能对多个 batch 规格保留额外资源。不同框架可能共享内存池或复用输入，但这种复用有执行互斥条件。你不能既假设所有规格可以同时运行，又假设它们共享同一工作区而没有同步。容量设计必须与调度设计一致，否则省下的只是纸面数字。

## 10. 第十课：CPU、磁盘和网络也有容量，offload 只是改变位置

把权重或 KV 放到 CPU 能降低 GPU 常驻量，但会增加主机内存需求。主机进程、数据加载、tokenizer、HTTP 请求体和日志也占内存；多个 worker 可能各加载一份，进程之间是否共享需要实际确认。容器的内存限制可能小于宿主机物理容量，不能只看整台机器 free 的数字。

传输还可能使用 pinned memory。固定页内存有助于某些异步拷贝，但不是免费的无限仓库；大量锁页会影响系统可回收内存。offload 设计应限制在途字节，而不只是限制请求数，因为一个长请求的状态可能远大于多个短请求。S11 的字节背压可以直接迁移到这里。

磁盘容量与读带宽也不同。模型文件能放进磁盘，不代表每轮按需加载专家足够快；操作系统页缓存会让冷读与热读差异巨大。若报告 offload 性能，必须说明是冷启动、热页缓存还是预取后的稳定态，并记录 CPU NUMA、设备连接与并发传输条件。没有这些边界，数字很难复现。

一次一 GiB 传输，理想有效吞吐十 GB/s，纯载荷约零点一零七秒；若每个 decode 步骤都需要这样的传输，生成速度会受到明显限制。这里使用的是人为设定有效吞吐，不是某个 PCIe 或网络规格保证。真实链路还存在建立、排队、协议与同步成本，应通过受限实验测量。

对你的使用场景，CPU 与 Docker 可以长期练习，是很适合训练所有权与容量模型的环境。先在 CPU 仿真中把字节预算、队列和取消释放做正确，再用短 GPU 实验测真实工作区与传输。没有必要为了理解一个页数公式长期占用八张 H100；当研究 TP 或 EP 分布时，多卡才提供不可替代的证据。

## 11. 第十一课：用校准实验修正公式，同时防止测量误解

第一步固定模型 revision、配置、dtype、并行布局、框架版本与启动参数。第二步记录设备总量、任务预算、启动前占用、加载后占用、图捕获后占用，以及框架报告的 KV 池大小。第三步使用短而固定的输入，逐级增加长度和并发，每次只改变一个轴，并在结束后记录逻辑资源是否归还。

若框架动态分配 KV，可以观察长度增加与占用增量；若预分配整个池，就观察已用页数和高水位。两种模式的测量曲线不同，但都可以与公式对应。不要把进程 reserved 不下降直接判为泄漏；也不能因为 reserved 平坦就忽略页池 free 持续减少。泄漏诊断需要比较所有权终态和重复实验的趋势。

假设每页十六 token，每页两 MiB，三个独立请求长度十六、十七、三十一，理论页数一、二、二，总五页十 MiB。若测到六页，要先查是否预留下一 token、推测槽位、COW 或缓存引用；若只有四页，查共享是否真实发生。残差本身是线索，不能直接归因于“框架开销”。

```python
def page_observation(lengths,block_size,page_bytes,observed_pages):
    expected = sum((n+block_size-1)//block_size for n in lengths)
    return {"expected_pages":expected,"observed_pages":observed_pages,
            "residual_pages":observed_pages-expected,
            "expected_bytes":expected*page_bytes}
row = page_observation([16,17,31],16,2*2**20,6)
assert row["expected_pages"] == 5 and row["residual_pages"] == 1
print(row)
# 人工观测值只用于诊断练习，不是设备测量。
```

在容量边界附近，不要用无限加压找 OOM。预先设定长度阶梯、最大并发、总时长、允许峰值和停止条件；出现预算越界、持续抢占或错误就结束该阶梯，保留证据。共享机器上的实验更应从已分配的一卡开始，不扫描或干扰其他用户的进程。

最终部署表应包含预测、观测、差值解释和证据范围。某次单请求成功只证明这一输入和路径能够运行，不能证明最大并发、最长输出或取消压力都可靠。容量验收至少覆盖正常完成、接近页边界、共享后追加、取消后资源归还和一次明确的资源不足失败。失败若能干净拒绝且状态不变，也是一项重要能力。

### 11.1 长度分布决定真实负载，平均值不能保护尾部

假设九十九个请求只有一百个位置，另一个请求十万个位置。平均长度约一千零九十九，但把所有请求都按平均值规划，会隐藏那个长请求对页池和 prefill 的冲击。如果系统允许长短混合，就要观察输入长度、输出长度、并发时刻和缓存命中的联合分布。分别知道平均输入与平均输出，还不足以知道同一时刻活跃请求的总状态。

到达过程也重要。相同的一千个请求，逐个串行执行与瞬间同时到达，峰值完全不同。长请求的存活时间更长，因此在任意时刻采样活跃队列，可能比从请求清单均匀抽样看到更多长请求。这是容量与排队相连的地方：服务越慢，状态保留越久，排队与抢占压力也可能增加，不能把到达率和请求持有时间当成无关量。

实用做法是保留一份公开合成负载，明确短、中、长请求比例与到达时间，先用 S02 的调度器回放得到最大活跃页数，再用真实框架验证代表性点。模拟给的是在指定规则下的逻辑容量，不是真实 GPU 耗时；如果采用模拟服务时间，应在报告中持续标记，不能把峰值预测伪装成真实生产观测。

敏感性分析比一个小数点后很多位的答案更有用。把输出预算、并发、权重开销、工作区和缓存命中分别改变一个合理范围，观察哪个变量最容易让余量变负。若总量主要受长上下文 KV 影响，继续精确估计很小的 norm 权重不会解决主要不确定性；若模型加载峰值已经超预算，优化平均输出长度也不能让进程成功启动。

最后，留出“无法承诺”的区域并给出运行政策。例如在最大并发与最长上下文同时出现时需要抢占，则服务文档应明确排队或拒绝规则，基准也要报告该组合的延迟和失败。容量模型不是为了证明所有组合都安全，而是为了清楚划定可运行范围，并让范围外的请求以可解释、可恢复的方式处理。

## 12. 第十二课：故障工作台、完整练习与容量工程

**故障一：总显存够，某个 rank OOM。** 先输出每 rank 的对象账本，检查 embedding、输出头、共享专家、通信缓冲与层分配。不要立即把所有 rank 的利用率统一调低；那可能浪费其他 rank 而没有修复不均衡。参考定位顺序是比较峰值 rank、列最大差异对象、确认复制或切分规则，再调整布局或预算。

**故障二：启动成功，长 prefill 才 OOM。** 对比加载后与 prefill 峰值，检查工作区、一次 token 数、attention 路径和图规格。降低 chunk 可能验证临时空间假设，但不能证明最终 KV 足够。若长请求随后在 decode 阶段耗尽页池，要继续处理状态容量，不能把两种失败混在一起。

**故障三：请求全部完成，第二轮更容易耗尽页。** 检查终态请求页表、缓存引用、在途传输与释放事件。缓存保留可能是设计，但必须可驱逐且有明确预算；无法解释的活引用才是泄漏线索。设备 reserved 没下降并非充分证据，页池所有权不闭合则是直接问题。

**故障四：四位模型比预期大很多。** 检查未量化层、scale 与零点、打包、对齐、浮点副本和 loader 峰值。Python int 列表或 int8 数组不自动实现四位压缩。分别报告磁盘文件、CPU 驻留、GPU 稳态与加载峰值，才能知道“大”发生在哪一层。

**练习一，单位。** 一百六十亿 byte 是十六 GB，约十四点九零 GiB。完整解法是先写分母，十的九次方对应 GB，二的三十次方对应 GiB。不要把十六作为 GiB 继续参与预算，否则误差会传播到余量与 admission。

**练习二，标准 KV。** 四十层、四个 KV 头、一百二十八维、两 byte、六万五千五百三十六位置，共五 GiB。两个完全独立请求十 GiB。每个因子都应能在张量形状中指出来源；query 头数不参与这里的状态大小。

**练习三，页尾。** 每页十六，长度十七和三十一需要四页，容量六十四位置，有效四十八，尾部浪费十六。先合并长度得到三页是错误的，因为不同请求的独占尾页不能自动合并。共享必须由物理页表证明。

**练习四，共享。** A 页表七、二，B 页表七、四，逻辑四项，物理三页。若缓存再持有七，物理仍三页但引用增加。释放 A 只能减少七的一个引用并释放独占二，不能把七直接归还空闲池。

**练习五，COW。** 两请求共享未满尾页，A 追加时先需要一个新页。若 free 为零，即使追加后总有效 token 只增加一，也可能无法执行。完整答案应解释申请、复制、切换、旧引用递减的顺序，并说明失败前不能破坏原页表。

**练习六，TP。** 两 KV 头与 TP 八不适用均匀整头分片公式。答案不是算零点二五头，而是检查本地 KV 头布局，可能复制或不支持。课程 Capacity 报错是正确保护，不能为了得到数字而删除校验。

**练习七，rank 不均衡。** 本章四 rank 示例总二十 GiB，平均五，但 rank 零八 GiB，预算六时失败。需要重分布特殊对象、改并行方案或增加该 rank 预算；仅说明总容量二十四没有帮助。

**练习八，混合状态。** 十二标准层加十二个每层一 MiB 固定状态，按本章参数为一点五 GiB 加十二 MiB。不能把二十四层全部乘长度，也不能把全部状态都当固定。若支持回滚另有 checkpoint，应增加对应对象与生命周期。

**练习九，预分配。** 请求数增加而设备占用不变，可能是 KV 池已在启动时预分配。应查看 free block、高水位和引用，不据此否认状态增长。相反，逻辑 free 回到初始值而 reserved 保持不变，可以是正常复用。

**练习十，峰值。** 常驻十五 GiB、prefill 三、decode 一、传输二，串行复用场景峰值十八，全部重叠二十一。答案必须画出事件顺序或使用 timeline_peak，而不是随意选择较小数字。异步释放必须等待真实完成。

**练习十一，输出预算。** 输入八十、最多输出一百、页长十六，保守预留十二页。八个已有同样预留占九十六页，一百页池不能再接纳一个同样请求。动态分配可以采用不同策略，但必须承担后续抢占与失败语义。

**练习十二，容量与速度。** 权重量化腾出 KV 空间，不保证更快；反量化、工作区和 kernel 支持会影响性能。一个方案“算术放得下”与“满足延迟目标”是两项验收，不能用容量表替代 S08 的计时实验。

**练习十三，残差。** 理论五页、观测六页，列出至少三个可验证原因：下一位置预留、推测槽位、缓存或 COW 引用。逐项查看实际元数据，不用“框架开销”笼统解释。若找不到拥有者，转入 S14 的资源守恒诊断。

**练习十四，未知架构。** 配置缺少 KV 头数时，不能按相近模型猜。查固定 revision 的 config、模型类与 attention 实现；明确默认继承 query 头数是否真实发生。无法确认时报告未知并停止给出精确容量承诺，这是工程诚实而不是计算失败。

工程第一阶段写单位严格的账本，输出权重、状态、工作区、通信、持久运行时与余量，并逐项标明估算依据。第二阶段加入逐请求分页与逐 rank 放置，完成本章所有手算 oracle。第三阶段加入事件时间轴，区分稳定态、加载峰值和传输重叠。第四阶段用受限真实框架实验校准，记录预分配策略与残差原因。第五阶段把结果交给 admission 和诊断模块，使资源不足能够明确拒绝且不破坏所有权。

验收不是得到一张看起来精确的表，而是能解释表中每个因子来自哪段结构、哪个拥有者持有它、什么时候释放、误差如何被观测修正。若还不能解释某个固定开销，就把它写成待测范围并给出下一项实验。下一章 [S14 服务诊断](14-diagnostics.md) 将把这些账本转化为故障定位证据。
