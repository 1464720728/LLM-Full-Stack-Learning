# S19 从真实数值缓存到分页注意力：逐行实现、验证与排错

在 S03、S04、S18 中，你已经用 token 元数据学习页表、引用计数、前缀共享和写时复制。本章把页面里保存的东西换成真正参与注意力计算的 K/V 数组。目标不是让一个模拟器打印“缓存命中”，而是证明旧计算得到的向量确实被后续 query 读取，并在分叉、释放、复用之后仍给出正确结果。

先修为 F14 的数组、shape 与矩阵乘法，M03 的注意力，S03 的页池和 S04 的引用所有权。如果你还不能解释 `a[2, :3]` 选择哪个轴，先回到 F14 做小数组练习；本章会重新解释张量各轴，但不会把切片语法当成已经自动掌握的知识。完整参考实现是 [paged_attention.py](../../labs/serving/paged_attention.py)，页池是 [kv.py](../../labs/serving/kv.py)，独立无缓存对照是 [backend.py](../../labs/serving/backend.py)。

所有示例从仓库根目录运行，只需 NumPy。正文的 Python 块按出现顺序在同一个 Python 会话中执行；部分后续块复用已定义的函数和导入。命令入口与回归入口如下：

```bash
python -m labs.serving.paged_attention
python -m unittest tests.test_serving_tensor_kv -v
```

这里没有 GPU Kernel，没有训练好的语言能力，也没有声称达到生产框架性能。本章有真实数值、有真实资源所有权、有可验证等价关系；这些正是日后改写 GPU 实现时必须保留的合同。CPU 运行正确与真实设备运行正确分别记录，不能互相替代。

## 第一课：缓存究竟替代了哪一次计算

### 1. 从一次无缓存前向开始

设已经确定的 token 序列为 `[1,2,3]`。模型先按每个 token 的编号，从 embedding 表取出一个向量，再分别乘 K、V 投影矩阵，得到三个位置的 key 和 value。最后位置的输入还会乘 Q 投影，产生当前 query。query 与全部 key 比较，经过 softmax 得到权重，再对 value 加权，形成当前注意力输出。

如果下一个 token 确定为四，无缓存实现重新输入 `[1,2,3,4]`。对这个单层、没有位置编码的教学模型，前面三个位置的 K/V 与上一轮完全相同，却又被计算了一遍。缓存的直接作用就是保存这三行，把新 token 四的 K/V 追加进去。注意力仍然要读取历史 K/V；缓存省去重复投影，不是让历史信息不再参与计算。

为什么只保存 K/V 而不必把每个历史 Q 都留下？当前下一步的输出由当前 query 去读取历史 key/value；先前 query 的职责已经完成。某些其他模型或算法可能维护不同状态，但在本章固定的注意力定义中，历史 Q 不是下一次读历史所需的输入。理解这种数据依赖，比背“KV cache 只有两个字母”更重要。

### 2. 缓存条件必须明确

“同样 token 的 K/V 一样”不是对所有模型的无限保证。在多层模型中，某层的 K/V 来自上一层隐藏状态，隐藏状态依赖前文；使用位置编码时也依赖逻辑位置；权重或 adapter 改变时，投影结果会改变。因此共享条件是相同有效计算上下文，而不是在任意两个位置看见同一个 token 就合并向量。

本章模型只有一个注意力层，embedding 后直接投影，没有 RoPE、MLP 或 LayerNorm。我们刻意缩小模型，使缓存存储与注意力合并的错误更容易隔离。完整多层模型与每请求真实缓存的组合在 [P2 项目](../../projects/P2_MODEL_RUNTIME.md)；两种实现解决相邻但不同的学习问题。

缓存还需要确定有效长度。数组里分配了一百个槽位，不表示已经计算了一百个 token。前向只能读取已完成写入且属于当前请求历史的行。未使用槽位可能是零、旧请求的数据或任何残留值；算法不能依赖它们“碰巧没影响”。后面的污染实验会专门验证这一点。

## 第二课：先把形状、单位和存储量算对

### 1. 一行 K/V 的含义

假设每个 token 有两个 KV 头，每个头有三个坐标。一个 token 的 K 是 `[2,3]` 数组：第一轴选择头，第二轴选择该头的坐标。三个 token 的 K 就是 `[3,2,3]`。把 token 轴放在最前面，是本章接口约定；其他框架可能采用不同顺序，不能只看元素总数相同就直接接起来。

分页再增加两个层次。我们预分配 P 个物理页，每页最多 B 个 token，因此完整池的 K shape 为 `[P,B,H,D]`。P 是页面数量，B 是每页槽数，H 是 KV 头数，D 是头维度。V 在本实现中与 K 同形状；有些数学定义允许 value 维度不同，但本章为了接口清楚固定二者相同。

假设 P=8、B=4、H=2、D=3，一份 K 的元素数是八乘四乘二乘三，即一百九十二。float64 每元素八字节，所以 K 占一千五百三十六字节，K+V 占三千零七十二字节。这是数组元素存储量，不包含页表、Python 对象、模型参数、临时 scores 或分配器管理开销。

```python
import numpy as np
from labs.serving.paged_attention import TensorKV, PagedTinyDecoder

pool = TensorKV(pages=8, block_size=4, heads=2, dim=3)
assert pool.keys.shape == (8, 4, 2, 3)
assert pool.keys.dtype == np.float64
assert pool.keys.nbytes == 1536
assert pool.keys.nbytes + pool.values.nbytes == 3072
```

### 2. 容量、占用和保留量是不同数字

这个池从创建起就分配了全部数组。当前只有一个请求、一个 token 时，仍然保留三千零七十二字节的 K/V 容量；有效数据只占两个头乘三个坐标乘 K/V 两份乘八字节，即九十六字节。其余空间是尚未被有效状态使用的容量，不是自动泄漏。

当请求释放后，元数据页面回到空闲列表，数组本身仍然存在。这样下一次请求可以复用页，不必每次申请一组新的大数组。真实服务也常有预分配池，因此监控中的“可复用空闲页”和“进程已向设备申请的内存”可能同时存在。看到请求结束但总显存不下降，不能立刻下结论说释放失败。

比较两种配置时，要把公式中的单位写出来。例如把每页 token 数翻倍，而物理页数不变，会把总 token 容量翻倍；如果为了保持总容量而把页数减半，元素总量可以保持不变，但尾页浪费、页表长度和分配粒度会变化。不能只说“大页更省内存”或“小页更灵活”，必须说明固定了哪个量。

### 3. 每层缓存怎样扩展这个公式

假设有 L 层，所有层采用相同 KV shape，且各层缓存独立，那么总元素存储大约再乘 L。若不同层有不同 KV 头数、维度或缓存机制，需要逐层求和，不能一概套用某个模型宽度。GQA 的缓存量由 KV 头数决定，不是 query 头数；这一点后面会在映射练习中重新验证。

dtype 也只是公式的一部分。把 float64 换成 float16，元素字节数减少，但数值范围、舍入误差、累加方式和工具链支持也随之改变。本章使用 float64 是为了建立容易诊断的 CPU 对照，不是向生产服务推荐这种精度。任何精度修改都要重新比较实际数值，而不能只计算理论节省。

## 第三课：从逻辑 token 位置找到物理数组元素

### 1. 为什么需要两次寻址

请求认为自己的历史是位置零、一、二、三……这样的连续序列。内存池却可能把这些位置分散放在物理页三、页一、页六。页表负责连接这两种视角：请求的第零个逻辑页对应哪个物理页，第一逻辑页又对应哪个物理页。物理页号的大小不代表时间先后。

给定逻辑位置 p 和页大小 B，先计算 `logical, slot = divmod(p, B)`。logical 是该 token 落在请求的第几个逻辑页，slot 是页内第几个槽。随后用 `table[logical]` 得到物理页号 bid，再读取 `keys[bid, slot, head, coordinate]`。除法负责切分历史，页表负责重定位，后两个轴负责选数值分量。

例如 p=6、B=4、table=[3,1]。六除以四商一余二，所以应读第二个逻辑页，对应物理页一、槽二。若直接读取物理页一并不是因为 `6//4` 恰好等于一，而是页表第二项确实等于一；换一张表后这个巧合就消失。测试必须故意打乱物理页，防止错误程序碰巧通过。

```python
logical, slot = divmod(6, 4)
table = [3, 1]
bid = table[logical]
assert (logical, slot, bid) == (1, 2, 1)
```

### 2. 从四维索引进一步推到字节偏移

若数组采用普通连续行优先布局，一个元素的线性索引是 `(((bid*B+slot)*H+head)*D+coordinate)`，再乘每元素字节数得到相对基址的字节偏移。对于 B=4、H=2、D=3，bid=1、slot=2、head=1、coordinate=2，线性索引为四十一，float64 偏移为三百二十八字节。

这个公式只适用于当前连续布局。如果数组是转置视图、有 padding 或其他 stride，应该使用各轴索引乘对应字节 stride 再相加。K01 已解释 shape 与 stride 的区别。分页负责逻辑页到物理页的转换；页内布局仍然需要独立描述，不能把页表当成所有地址问题的万能解释。

```python
B, H, D = 4, 2, 3
linear = (((1 * B + 2) * H + 1) * D + 2)
assert linear == 41
assert pool.keys.strides == (192, 48, 24, 8)
assert sum(i * stride for i, stride in zip((1, 2, 1, 2), pool.keys.strides)) == 328
```

### 3. 元数据不能替代数值

PagePool 的 Page.data 仍保存 token 编号，用它表示哪些槽有效、每页长度是多少。K/V 的真正浮点内容则在 TensorKV 的 keys/values 数组。两个表示必须同步：如果元数据声称已有三个 token，而对应向量只写了两个，读取第三行就会出错；如果数值写了三行而有效长度仍是二，第三行不会被正确使用。

保留 token 元数据是为了教学可检查性。生产系统可以用长度、块描述符等更紧凑结构保存状态，并不一定把所有 token 编号都复制进页对象。你应该迁移的是不变量：逻辑位置、物理位置、有效范围和实际数值相互一致，而不是要求真实框架长得与 Python 参考类一模一样。

## 第四课：亲手写入、读回，再打乱物理页面

### 1. 先用人工数值隔离模型计算

测试缓存寻址时，暂时不要调用 embedding 或投影。我们直接构造每行容易辨认的数字：位置零是 `[0,1]`，位置一是 `[2,3]`，位置二是 `[4,5]`。V 在这些数字上加十。这样一旦行号错位，可以从输出立即知道读到了哪里。

```python
kv = TensorKV(pages=4, block_size=2, heads=1, dim=2)
kv.create("a")
k = np.arange(6.0).reshape(3, 1, 2)
v = k + 10
kv.append("a", [1, 2, 3], k, v)
read_k, read_v = kv.gather("a")
np.testing.assert_array_equal(read_k, k)
np.testing.assert_array_equal(read_v, v)
assert kv.pool.tables["a"] == [0, 1]
assert [len(kv.pool.pages[b].data) for b in kv.pool.tables["a"]] == [2, 1]
kv.pool.check()
```

create 只建立一个空页表，不会为零 token 分配物理页。append 根据新增数量申请足够页面并写入。前三行跨两个页，第一页满两行，第二页只用一行。gather 必须按请求页表顺序读取，并只取有效行，得到形状 `[3,1,2]`，而不是把整池 `[4,2,1,2]` 直接拉平。

### 2. 制造非连续物理顺序

PagePool 使用最小页号优先的空闲堆，因此一个简单单请求常会得到 `[0,1,2]`。这太容易掩盖“把逻辑页号当物理页号”的错误。我们先让临时请求占住页零，再让目标请求写满页一；释放临时请求后继续追加，目标的第二逻辑页会复用页零，形成 `[1,0]`。

```python
shuffled = TensorKV(pages=3, block_size=2, heads=1, dim=2)
shuffled.create("temporary")
shuffled.append("temporary", [9], np.zeros((1,1,2)), np.zeros((1,1,2)))
shuffled.create("target")
shuffled.append("target", [1,2], k[:2], v[:2])
shuffled.release("temporary")
shuffled.append("target", [3], k[2:], v[2:])
assert shuffled.pool.tables["target"] == [1,0]
np.testing.assert_array_equal(shuffled.gather("target")[0], k)
shuffled.pool.check()
```

这一测试没有直接篡改页表，而是通过合法分配与释放形成所需布局。这样页引用、空闲列表和有效长度都保持一致。如果你直接把两个页号交换，却不同时考虑各页实际内容，制造的可能是一个本就无效的状态，不能用它证明正常接口有问题。

读出顺序必须跟随 table，而不能对物理页号排序。物理页号零可能保存较晚 token，页号一可能保存较早 token。把页号排小到大看似“整理内存”，实际破坏了接口承诺的逻辑行顺序。不过，不能据此推断“最终 decode 输出一定会改变”。检查中间合同与检查最终数值，负责的错误并不相同。

考虑一个最后位置的 query，所有有效历史都可见，且没有随数组列号另外添加的位置 bias。若 K 与 V 使用完全相同的排列 P，点积分数也按同样顺序重排，softmax 权重随之重排，最后加权求和不变。用矩阵写就是 $K'=PK$、$V'=PV$，于是 $\operatorname{softmax}(qK'^T)V'=\operatorname{softmax}(qK^T)P^TPV=\operatorname{softmax}(qK^T)V$。这是对任意输入成立的联合置换性质，只有浮点加法顺序可能带来末位差异；即使 K 已经包含 RoPE，整行一起移动也不破坏这个等式。

```python
permutation_rng = np.random.default_rng(84)
permutation_q = permutation_rng.normal(size=5)
permutation_k = permutation_rng.normal(size=(7, 5))
permutation_v = permutation_rng.normal(size=(7, 3))
permutation = np.array([6, 2, 0, 5, 1, 4, 3])

def one_query_sum(query, keys, values):
    scores = keys @ query / np.sqrt(query.size)
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()) @ values

np.testing.assert_allclose(
    one_query_sum(permutation_q, permutation_k, permutation_v),
    one_query_sum(permutation_q, permutation_k[permutation], permutation_v[permutation]),
    rtol=1e-12, atol=1e-12,
)
assert not np.array_equal(permutation_k, permutation_k[permutation])
```

因此本章直接用 gather 的逐行结果和原始 K/V 比较，抓住“逻辑位置映射错误”，不把最终 token 是否变化当作唯一验收。若后续按逻辑列添加 causal mask 或位置 bias，却没有同步其身份；或者多 query prefill 使用错误列可见范围；或者追加逻辑误把物理排序末尾当请求尾页，错误就可能影响计算。K 与 V 使用不同排列更会直接打破键和值的对应关系。先说清哪条合同被破坏，再选择能观察它的断言，才能写出有区分力的测试。

## 第五课：读懂 append 的每一个阶段

### 1. 输入先建立结构合同

append 的四个输入分别是请求 ID、待追加 token 列表、这些 token 对应的 K、对应的 V。若新增 n 个 token，则 K/V 必须具有 `[n,H,D]` 形状。token 数与向量行数一一对应，不能只检查数组“有三个维度”，也不能让广播自动填补缺失的 token 轴。

函数先把 K/V 转成池采用的 dtype，再检查形状与有限性。有限性指每个元素都不是 NaN 或正负无穷。这样能在改变页池前拒绝明显非法数据。例如你误传 `[n,D]` 而少了 head 轴，程序应在申请页面之前报错，不应先占用一页再要求调用者猜怎样回滚。

需要区分合同检查与完整安全边界。当前类是可信教学代码的内部接口，不是可以直接公开接收任意网络 payload 的服务。它没有为任意巨大输入提供资源上限，没有保证所有可能的 dtype 都适用，也没有把线程并发纳入设计。你应按示例使用浮点数数组和已验证的小规模输入；扩展接口时再明确加强相应约束。

```python
bad = TensorKV(pages=2, block_size=2, heads=1, dim=3)
bad.create("r")
free_before = tuple(bad.pool.free)
try:
    bad.append("r", [1], np.zeros((1,3)), np.zeros((1,3)))
except ValueError:
    pass
else:
    raise AssertionError("missing head axis was accepted")
assert tuple(bad.pool.free) == free_before
assert bad.pool.tokens("r") == []
```

### 2. 旧状态要在变更前记录

代码保存 old_size、old_tail 和 valid_tail。old_size 是当前请求已经有多少 token；old_tail 是旧页表的最后物理页号；valid_tail 是 `old_size % block_size`，表示尾页尚未满时有多少行有效。这些信息用于在元数据操作之后判断是否发生写时复制，并复制准确的旧有效区域。

为什么不在 append 之后再计算 old_size？因为那时长度已经包含新 token，无法直接知道哪些行属于旧前缀。为什么只记录旧尾页，而不记录所有旧页？因为本接口只做末尾追加，满页无需原地修改；只有共享且未满的最后一页可能需要复制。这个推理依赖 append-only 语义，不能搬到任意位置覆盖写入的接口。

随后调用 PagePool.append。页池先计算完成本次追加所需的全部新页数量，包括共享尾页 COW 所需的新页，再检查空闲页是否足够。只有整次资源需求满足时才改变引用和页表。这让可预见的 OutOfPages 失败在旧状态仍完整时发生。

### 3. 数值复制与新行写入

元数据 append 成功后，如果旧尾页有效但新的对应物理页号发生改变，说明请求从共享尾页分裂到了一张新页。此时必须复制 `[:valid_tail]` 的 K 和 V。只复制 K 不复制 V，或者复制到错误的页号，都会让页表表面正确而计算结果错误。

再遍历每个新 token，令 position=old_size+offset，做 divmod，查当前页表，写入新 K/V 行。一个追加可以跨多个页面，所以不能把全部新向量简单写到“当前尾页剩余切片”。若本轮新增六行而尾页只剩一格，还需要填满那一格，再写一个完整新页，再写最后不足页的部分。

这段 Python 循环清楚地表达了地址合同，但不是 GPU 性能目标。之后可以按页连续区间合并拷贝，或在设备 Kernel 中并行写入；优化前后应保持相同逻辑位置和有效范围。不能为了去掉循环就依赖物理页恰好连续，因为那会直接抹掉分页的关键条件。

### 4. 原子性的保证到哪里为止

本实现保证常见输入错误和页不足在预检阶段不会部分修改资源。它没有实现对任意后续内存写失败、进程突然终止或异步设备错误的完整事务回滚。同步 NumPy 切片赋值在已验证形状的正常执行路径下完成，正是当前教学保证成立的环境。

因此描述应是“经过预检的页不足操作保持状态”，而不是“append 面对任何故障都绝对原子”。真实框架需要考虑设备错误、复制事件、发布次序和恢复协议。精确说明保证的触发条件，是工程严谨性的一部分；把所有失败都写成一个宽泛的原子标签，只会隐藏尚未设计的情况。

## 第六课：先做慢而清楚的 gather 注意力

### 1. 把两种问题暂时拆开

gather 逐页读取有效 K/V，然后把它们拼成连续历史。它做了额外拷贝，所以不适合作为最终高性能路线；但它能让我们先独立检查寻址。确认连续历史等于原始输入后，再用普通注意力公式检查数学。如果同时写复杂页寻址和分块 softmax，一旦输出不对，很难定位责任。

当前 q shape 为 `[H,D]`，历史 k/v shape 为 `[T,H,D]`。对每个 head h、历史位置 t，分数为该头 query 与该位置 key 的点积，再除以根号 D。输出 scores shape 为 `[H,T]`。这里 H 个头形成 H 个独立分布；softmax 在 T 轴上归一化，不是在不同 head 之间竞争。

`np.einsum('hd,thd->ht', q, k)` 的字母是维度标签。h 在输入和输出中都保留，表示逐头计算；t 从历史输入保留到输出；d 在两个输入中出现却不在输出中，表示沿特征坐标求和。这与手写 `sum(q[h,d]*k[t,h,d] for d...)` 是同一运算，只是一次完成全部 h/t。

接着对每头 scores 减去该头最大值，指数化并除以该头指数和。减最大值不会改变 softmax 比例，因为分子分母同时乘了同一个缩放因子；却能防止原始大正数直接指数造成溢出。最后 `einsum('ht,thd->hd', weights, v)` 沿 t 求和，保留每头每坐标输出。

```python
q = np.array([[1.0, -0.5]])
read_k, read_v = kv.gather("a")
scores = np.einsum("hd,thd->ht", q, read_k) / np.sqrt(2.0)
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True)
manual = np.einsum("ht,thd->hd", weights, read_v)
np.testing.assert_allclose(kv.attention("a", q), manual)
np.testing.assert_allclose(weights.sum(axis=1), [1.0])
```

### 2. 为什么这里没有额外因果 mask

这个接口只计算当前最后位置的一条 query。池中已有的有效行包含它之前的历史以及当前 token，本来都允许被当前 query 读取，没有尚未生成的未来 token。因此把所有有效历史纳入当前注意力是正确的。

如果你把接口改成同时计算一批 query，它们的逻辑位置不同，允许读取的历史范围也不同。最早 query 不能读取该批后面的 K/V；仅仅把 q 轴扩成 `[Q,H,D]` 而仍让每条 query 看完整池，会泄露未来。正确扩展必须给出 query 的逻辑位置和可见关系，并测试改变未来 token 不影响早期输出。

这说明“是否需要 mask”不能只由矩阵是否方形决定。真正问题是每条 query 的语义位置与每条 key 的身份。页内槽号零不等于序列位置零；一次 decode 的唯一 query 数组下标也不等于它的绝对位置。把数组坐标和逻辑时间分开，才能正确处理缓存和分块。

### 3. 参考算法也有数值边界

输入有限不保证点积永远有限。若坐标极大，乘法或求和仍可能溢出；仅在 softmax 前减最大值不能挽回已经产生的无穷分数。本章测试采用合理范围的 float64 输入。以后改低精度时，要同时检查投影、点积、归一化和累加，而不是把所有问题都归结为 softmax 稳定化。

对比输出时也不要无条件使用严格相等。不同求和顺序可能产生末位舍入差异，尤其是把多个页分开处理之后。测试使用绝对与相对误差阈值，并保留手算例子、形状检查和不变量。容差不是允许任意错误的借口；当误差大幅超过参考数值范围时，应先找到具体阶段。

## 第七课：为什么各页结果不能直接平均

### 1. 页内归一化丢掉了跨页权重

假设第一页包含两个位置，未归一化权重为一和二，value 分别为一和四。第一页自己的 attention 输出是 `(1*1+2*4)/(1+2)=3`。第二页只有一个位置，未归一化权重为六，value 为十，因此页内输出为十。把页结果平均得到六点五。

但全局注意力应该把三个位置一起归一化，输出为 `(1*1+2*4+6*10)/(1+2+6)=69/9=23/3`，约七点六六七。第二页的总权重是第一页的两倍，不能给两个页各一半总影响。页内输出只是一个归一化后的均值，缺少该页在全局中的相对质量。

如果你恰好选择第二页权重为三，两页总权重相同，错误平均会偶然给出正确结果。这就是为什么只测试对称输入容易漏错。一个有解释力的反例需要打破错误算法暗中假设的对称性：不同页长度、不同分数范围、不同权重和不同 value 都值得分别改变。

```python
page_mean = (3.0 + 10.0) / 2.0
correct = (1.0 + 2.0*4.0 + 6.0*10.0) / 9.0
assert page_mean == 6.5
assert abs(correct - 23/3) < 1e-12
assert not np.isclose(page_mean, correct)
```

### 2. 只保存分母还不够说明数值尺度

如果直接保存每页原始指数和，可以在数学上按它们重新加权。但原始 scores 可能很大，直接指数会溢出；而每页各自减最大值后，分母处于不同缩放尺度。第一页减了十，第二页减了一百，它们得到的“指数和二点几”不能直接比较。

因此需要同时保存每页或累计状态的最大值 m、按这个最大值缩放的分母 l、按同样尺度缩放的加权和 u。m 告诉我们这些统计在哪个尺度下表示，l 表示总权重，u 表示未最终归一化的内容总和。只有三者放在一起，才能稳定地把不同页的贡献合并。

这是在线归一化的关键思想。它不是近似丢弃历史，也不是因为分块而改变 attention 定义；它保留足够的统计量，使完整求和可以按页逐步计算。浮点求和顺序可以变化，但理想实数公式保持相同。你应该先理解这个等价性，再讨论硬件如何高效实现。

## 第八课：一步步推导在线 softmax 的更新

### 1. 为已处理历史定义三个量

对于一个头，已处理分数为 s_i，对应 value 向量为 v_i。定义 m 为这些分数的最大值，l 为所有 `exp(s_i-m)` 之和，u 为所有 `exp(s_i-m)*v_i` 之和。l 是标量，u 是 D 维向量，最终输出为 u/l。每个头各自保存一套 m/l/u，不能让不同头共用一个统计量。

新页到来，设新页最大分数为 b，新共同最大值为 `m_new=max(m,b)`。旧统计原来减去的是 m，现在必须改成减去 m_new。对任一旧项，`exp(s_i-m_new)=exp(s_i-m)*exp(m-m_new)`，所以旧 l 和旧 u 都应乘同一个 alpha，即 `exp(m-m_new)`。

新页每项直接按新尺度计算 `p_j=exp(s_j-m_new)`。于是新 l 等于 alpha 乘旧 l，再加新页 p_j 之和；新 u 等于 alpha 乘旧 u，再加新页 p_j*v_j 之和。最后将 m 更新为 m_new。这里既要缩放分母，也要缩放分子，只更新其中之一会改变结果。

初始没有历史时，m 为负无穷，l 和 u 为零。第一个有限页到来，alpha 为零，旧零统计自然不贡献。这个初始化依赖至少有一个有效位置；空历史不能产生一个正常归一化输出，所以函数明确报错，而不是除以零后把 NaN 当成合法结果返回。

### 2. 手算两个页的完整轨迹

取第一页分数 `[0,ln2]`，value `[1,4]`。其最大值为 ln2，缩放权重为 `[1/2,1]`，因此 l=1.5，u=0.5*1+1*4=4.5。此时归一化输出是三，但我们还不丢弃 m/l/u，因为之后还可能出现新页。

第二页分数为 ln6，value 为十。新最大值是 ln6，alpha=exp(ln2-ln6)=1/3。旧分母变为零点五，旧分子变为一点五；新页在新尺度下权重为一，所以新分母一点五，新分子十一点五。最终输出十一点五除以一点五，即二十三除以三，约七点六六七。

这个例子中特意把第二页最大值提高，迫使旧统计重标度。若所有后续页最大值都不超过第一页面最大值，alpha 始终为一，忘记重标度的错误实现可能长期不暴露。测试应该覆盖最大值在后页上升、保持不变和只在前页出现的情况。

```python
m, denominator, numerator = -np.inf, 0.0, 0.0
trace = []
for block_scores, block_values in [
    (np.array([0.0, np.log(2.0)]), np.array([1.0, 4.0])),
    (np.array([np.log(6.0)]), np.array([10.0])),
]:
    next_m = max(m, float(block_scores.max()))
    alpha = np.exp(m - next_m)
    p = np.exp(block_scores - next_m)
    numerator = alpha * numerator + float(p @ block_values)
    denominator = alpha * denominator + float(p.sum())
    m = next_m
    trace.append((m, denominator, numerator))
np.testing.assert_allclose(trace[0][1:], [1.5, 4.5])
np.testing.assert_allclose(trace[1][1:], [1.5, 11.5])
np.testing.assert_allclose(numerator / denominator, 23/3)
```

### 3. 对照实际实现中的广播

代码 maximum shape 为 `[H]`，denominator 也是 `[H]`，numerator 为 `[H,D]`。更新 numerator 时，alpha 必须写成 `alpha[:,None]`，使每个头的标量沿该头所有坐标广播。若直接用 `[H]` 乘 `[H,D]`，在 H 恰好等于 D 时可能运行却沿错轴，在二者不同时则可能直接报错。

scores shape 为 `[H,T_page]`，因此减最大值时使用 `new_max[:,None]`。每个头只减自己的最大值，每页位置共享该头缩放。`p.sum(axis=1)` 沿页内 token 轴求和；加权 value 时再次使用 einsum，把页内位置消去，输出每头 D 维贡献。每一处广播都来自形状推导，而不是随便加一个 None 直到程序不报错。

最后输出 `numerator/denominator[:,None]`。这一除法只在全部有效页处理完后做一次，因此不会把各页误当成相同权重的独立答案。你可以在循环中打印 m/l/u，核对上面的手算；调试时保留最小两页输入，比打印成千上万行随机矩阵更有用。

## 第九课：共享前缀时，数值写时复制怎样发生

### 1. fork 共享的是物理状态

假设 A 已保存三个 token，页大小为四。A 的尾页还有一个空槽。fork(A,B) 后，B 的页表复制了 A 的物理页号，两个请求指向同一张页，引用计数从一变成二。这里没有复制 K/V 大数组；共享前缀的价值正是避免立即复制相同状态。

接着 B 要追加自己的第四个 token。若直接写共享页空槽，页的有效长度会变为四，A 也会通过同一个 Page 对象看见这个新 token，导致两个历史串在一起。因此即使原来的三个槽没有被覆盖，追加本身仍会改变共享尾页的可见状态，需要 COW。

PagePool 先为 B 申请新页，复制旧 token 元数据，将 B 页表尾项改为新页，并减少旧页引用。TensorKV 看到页号改变，再把旧三个有效 K/V 行复制到新页，最后写入 B 的第四行。A 继续持有旧页，旧页内容和有效长度均不改变。

```python
branch = TensorKV(pages=4, block_size=4, heads=1, dim=2)
branch.create("A")
prefix_k = np.arange(6.0).reshape(3,1,2)
prefix_v = prefix_k + 10
branch.append("A", [1,2,3], prefix_k, prefix_v)
branch.fork("A", "B")
shared_page = branch.pool.tables["A"][0]
assert branch.pool.pages[shared_page].refs == 2
branch.append("B", [9], np.array([[[99.0,98.0]]]), np.array([[[97.0,96.0]]]))
assert branch.pool.tables["A"][0] != branch.pool.tables["B"][0]
assert branch.pool.tokens("A") == [1,2,3]
assert branch.pool.tokens("B") == [1,2,3,9]
np.testing.assert_array_equal(branch.gather("A")[0], prefix_k)
np.testing.assert_array_equal(branch.gather("B")[0][:3], prefix_k)
np.testing.assert_array_equal(branch.gather("B")[1][-1], [[97.0,96.0]])
branch.pool.check()
```

### 2. 为什么复制旧数值时旧页仍然可读

触发 COW 时旧页至少有两个拥有者。移走 B 的一个引用后，至少还有 A 或缓存条目持有旧页，因此它不会回到 free 列表。同步执行过程中，新页复制能读取稳定的旧有效区。这个推理依赖引用计数正确，并依赖没有其他线程绕过协议改写旧页。

GPU 上“主机元数据已经改完”不代表复制已完成。若新页复制尚在设备队列中，后续 attention 可能需要等待复制事件；若旧页的最后拥有者随后释放，还必须保证设备复制不再读取旧页后才能复用。引用计数解决逻辑拥有，设备完成事件解决实际使用时序，两者缺一不可。

当前参考没有并发线程和异步设备，所以可以把这两个时序合在同步调用里理解。以后加入事件时，不要只是给 release 加一个 sleep；应明确哪个事件证明哪次读取完成，再由完成事件触发回收。等待固定毫秒数无法构成正确性保证。

### 3. 满尾页为什么通常不用复制

若 A 的最后一页已经满，B 追加时会申请新的逻辑页，不需要改写共享满页的有效长度或内容，因此可以继续共享旧页。COW 仅针对需要修改的共享部分。无条件复制全部前缀虽然可能保持正确，却会浪费内存与复制带宽，失去共享价值。

这也解释了为什么页大小影响分叉成本。较大的页可能在共享尾页中保留更多已有效行，分叉时需要复制的数值更多；较小的页可能增加页表和管理开销。不能只看一个维度就宣布某个块大小最优，应根据请求长度、共享模式、硬件访问粒度与测量结果比较。

## 第十课：一次跨多页追加，怎样证明资源不足不留下半成品

### 1. 手算需要申请多少页

设页大小四，A/B 共享三个 token 的部分尾页，B 要追加六行。完成后 B 总长九，需要三张逻辑页。B 原来有一张页表项，因此新增长度本身需要两张页；但旧部分尾页共享，还需要一张 COW 页，总计要从 free 申请三张。

如果只按 `ceil(6/4)` 计算，会得到两张，漏掉旧尾页复制；如果只看最终三张减原来一张，同样漏掉 COW。页池的 needed 同时考虑最终逻辑页数与共享尾页复制责任。先把这个算式手写出来，再看代码里的布尔 copy_tail，才知道那一项为什么不能省。

本池若总共只有三张物理页，旧共享前缀已占一张，只剩两张空闲，因此这次追加必须整体失败。不能先复制尾页、写入一部分 token，随后才发现最后一页不够。否则调用者收到失败时，B 却已经悄悄增长，重试会重复数据或丢失原边界。

### 2. 状态快照要同时覆盖元数据和数值

只断言 token 历史没变还不够。错误实现可能没有更新列表，却已经覆盖了共享 K/V；只断言数组相同也不够，因为引用计数可能已经减少或空闲页已被取走。一次失败原子性检查应保存页表、各页 token、引用数、free 列表及 K/V 数组，再逐项比较。

```python
from labs.serving.kv import OutOfPages

def metadata_snapshot(tensor_kv):
    p = tensor_kv.pool
    return {
        "tables": {rid: tuple(ids) for rid, ids in p.tables.items()},
        "pages": tuple((tuple(page.data), page.refs) for page in p.pages),
        "free": tuple(p.free),
    }

limited = TensorKV(pages=3, block_size=4, heads=1, dim=2)
limited.create("A")
limited.append("A", [1,2,3], prefix_k, prefix_v)
limited.fork("A", "B")
assert limited.pool.needed("B", 6) == 3
before_meta = metadata_snapshot(limited)
before_k, before_v = limited.keys.copy(), limited.values.copy()
try:
    limited.append("B", [4,5,6,7,8,9], np.ones((6,1,2)), np.ones((6,1,2))*2)
except OutOfPages:
    pass
else:
    raise AssertionError("append should require three free pages")
assert metadata_snapshot(limited) == before_meta
np.testing.assert_array_equal(limited.keys, before_k)
np.testing.assert_array_equal(limited.values, before_v)
limited.pool.check()
```

快照中的 `.copy()` 很重要。如果只是 `before_k=limited.keys`，两个变量指向同一个数组；错误实现把数组改了，所谓“旧快照”也随之改变，比较自然通过。测试代码本身也必须遵守对象身份与复制语义，否则会给出没有意义的安全感。

### 3. 用足够资源验证同一操作的成功路径

失败测试之外还需要成功对照。把池改成四页，旧前缀占一页，剩三页恰好够用。追加后 B 长九，A 长三；A 的数值保持原样，B 的前三行与旧前缀相同，后六行为新数据。完成后释放 A、B，所有页面应归还。

成功与失败只改变池容量这一因素，使测试能直接归因于资源边界。若同时改变块大小、头数、输入数据和请求顺序，即使某个配置成功，也很难证明你理解的是哪一条需求。控制变量不仅用于性能实验，也适用于状态机正确性实验。

```python
sufficient = TensorKV(pages=4, block_size=4, heads=1, dim=2)
sufficient.create("A")
sufficient.append("A", [1,2,3], prefix_k, prefix_v)
sufficient.fork("A", "B")
new_k = np.arange(12.0).reshape(6,1,2) + 100
new_v = new_k + 10
sufficient.append("B", [4,5,6,7,8,9], new_k, new_v)
np.testing.assert_array_equal(sufficient.gather("A")[0], prefix_k)
np.testing.assert_array_equal(sufficient.gather("B")[0], np.concatenate([prefix_k,new_k]))
assert [len(sufficient.pool.pages[p].data) for p in sufficient.pool.tables["B"]] == [4,4,1]
sufficient.release("A")
sufficient.release("B")
sufficient.pool.check()
assert len(sufficient.pool.free) == 4
```

## 第十一课：前缀缓存复用必须真的接上数值页面

### 1. 发布条目如何延长状态寿命

PrefixCache 与 TensorKV 共用同一个 PagePool。发布缓存时，缓存条目记录命名空间、token 前缀和物理页号，并为自己增加引用。这使原请求释放后，真实数值页仍有拥有者，不会被新请求分配覆盖。缓存条目只保存一个字符串而没有页引用，不足以保留可用状态。

新请求 attach 时，找到兼容前缀，把相应物理页号附加到自己的空页表并增加请求引用。这样新请求可以通过 TensorKV.gather 读到与旧请求完全相同的向量。仅把 computed 设成四，却没有页表和数值所有权，会让调度器跳过四个根本不存在的状态。

```python
from labs.serving.kv import PrefixCache

cached = TensorKV(pages=8, block_size=2, heads=1, dim=2)
cache = PrefixCache(cached.pool)
cached.create("old")
cache_k = np.arange(8.0).reshape(4,1,2)
cached.append("old", [1,2,3,4], cache_k, cache_k+2)
assert cache.publish("old", "model-v1/tenant-demo", 4) == 4
cached.release("old")
cached.create("new")
assert cache.attach("new", "model-v1/tenant-demo", (1,2,3,4,5)) == 4
np.testing.assert_array_equal(cached.gather("new")[0], cache_k)
cached.pool.check(cache.refs())
cache.clear()
np.testing.assert_array_equal(cached.gather("new")[0], cache_k)
cached.release("new")
cached.pool.check()
assert len(cached.pool.free) == 8
```

清空缓存以后，new 仍持有自己的引用，所以它继续读到有效状态。只有 new 也释放后，页面才回到 free。这条轨迹同时验证了缓存拥有者与活动请求拥有者彼此独立；淘汰缓存不能让正在执行的请求失去所需数据。

### 2. 为什么示例 prompt 比命中前缀长一项

缓存了四个位置，新 prompt 是五个位置，所以 attach 可以复用前四个，重算最后一个产生下一 token 所需 logits。当前接口只保存 K/V，不保存可直接返回的末位置 logits。因此即使整个 prompt 与缓存完全相同，也会保留至少一个 token 的重算责任，再按完整页粒度向下取整。

这是当前模型执行协议的设计，不是说所有框架在任何情况下都必须采用同一重算数量。若你同时缓存末 hidden state 或 logits，可以改变协议，但要增加对应身份、格式、寿命和内存管理。不能只删除长度减一，却不补上首个输出的计算来源。

### 3. 命名空间不会自动替你识别模型

示例字符串包含模型版本与租户，只是表明调用者应该承担这些隔离责任。PrefixCache 不会扫描权重计算正确摘要，也不会知道两个任意字符串是否代表同一模型。如果宿主把不同权重都错误命名为 model-v1，缓存仍会按相同名称复用，从而产生错误数值。

真实系统应从部署配置和经过验证的模型身份产生 namespace，必要时包含 adapter、位置编码设置、缓存 dtype/布局等影响兼容性的条件。隔离策略还可能禁止不同访问域共享状态。哪些字段参与键，应该依据“是否改变状态或允许共享关系”判断，而不是机械加入所有请求参数。

## 第十二课：有效长度如何保护请求免受旧槽位干扰

### 1. 释放不是清零

当页面引用归零时，PagePool 清空 token 元数据并归还页号，但 TensorKV 的数值数组不清零。这不是遗漏，而是刻意让有效长度承担正确职责。新请求只能读取自己写入的行；无效区域是什么数字都不应影响输出。

如果测试总在全零数组上运行，错误地多读一行可能只让输出稍微缩小，甚至在特定数据上看不出变化。我们把无效槽写成极大正负数，作为明显干扰。正确算法切片只取有效行，结果仍不变；错误算法读整页，会立刻受到污染。

```python
poisoned = TensorKV(pages=2, block_size=4, heads=1, dim=2)
poisoned.create("r")
poisoned.append("r", [1], np.ones((1,1,2)), np.array([[[2.0,3.0]]]))
page = poisoned.pool.tables["r"][0]
poisoned.keys[page,1:] = 1e20
poisoned.values[page,1:] = -1e20
np.testing.assert_allclose(poisoned.online_attention("r", np.ones((1,2))), [[2.0,3.0]])
```

这里只有一个有效 value，所以任何合法 query 的注意力输出都应等于它，构成非常强的判据。它不依赖某个复杂模型的特定随机输出，也无需猜预期小数。选择这种简单不变量，可以让测试更直接地揭露读范围错误。

### 2. 跨请求复用的完整轨迹

让 A 先写三行值九十九，然后释放；B 复用同一页，只写一行值七。B 的输出必须是七，不能混入 A 的剩余两行。这个测试比“创建一个全新池给 B”更强，因为后者没有真正覆盖资源复用场景。

```python
reuse = TensorKV(pages=1, block_size=4, heads=1, dim=2)
reuse.create("A")
reuse.append("A", [1,2,3], np.ones((3,1,2)), np.ones((3,1,2))*99)
reuse.release("A")
reuse.create("B")
reuse.append("B", [8], np.ones((1,1,2)), np.ones((1,1,2))*7)
assert reuse.pool.tokens("B") == [8]
np.testing.assert_allclose(reuse.online_attention("B", np.ones((1,2))), [[7.0,7.0]])
reuse.pool.check()
```

生产系统可以基于隔离要求增加清零策略，但清零不能代替访问边界。即使残留变成零，错误读取仍然改变归一化分母和输出；它只把某种明显异常变成较隐蔽的数值偏差。正确范围与数据清理是两项可以配合的措施，应分别说明和验证。

### 3. 引用计数检查为什么还不够

PagePool.check 可以证明页引用与请求、缓存条目计数一致，检查空闲列表没有重复，非末页都满。它不知道某个浮点向量是不是该 token 在该模型下的正确 K/V。因此所有权检查通过，不表示数值正确；数值例子恰好正确，也不表示资源没有泄漏。

本章分别使用元数据不变量、人工数值读回、完整模型 logits 对照和释放后全部页归还。这些证据互补，不能用一个“测试通过”概括成所有性质都获得同样证明。设计自己的系统测试时，也应先列出要证明的合同，再选择对应观察量。

## 第十三课：把页缓存接回一个完整可对照的小模型

### 1. 先固定两边相同的参数

PagedTinyDecoder 内部创建 TinyAttentionBackend，它们共享同一组 embedding、Q/K/V 投影和输出投影。对照不能只要求“使用同样结构”，还必须使用相同权重与输入。若一边 seed 十一、一边 seed 十二，输出不同是模型不同，不能归因于分页算法。

append 的第一步验证 token 是合法内置整数且落在词表范围。随后查 embedding 得到 x，计算 `x@k` 与 `x@v`，加上 token 轴和 head 轴后写入 TensorKV。当前模型只有一个头，所以 `[D]` 变为 `[1,1,D]`。最后计算 `x@q`，用所有已保存历史做在线注意力，取回单头结果。

注意力结果与当前 embedding x 相加，形成这个教学模型的残差，再乘输出矩阵得到词表 logits。它没有完整 Transformer 的所有组件；但从 token 输入、投影、真实缓存、页表读取、归一化到输出的数值链是完整的。你可以逐个中间值打印出来，并与无缓存路径逐项比较。

### 2. 逐位置比较比只看最终输出更有定位价值

若仅比较整个序列最后一步，错误可能在第一步就发生，却直到很久后才被注意。逐位置比较会告诉你差异首次出现在哪个长度，特别适合发现跨页边界、尾页满、分叉或新页最大分数上升时的问题。

```python
decoder = PagedTinyDecoder(pages=16, block_size=3, seed=11)
decoder.create("r")
tokens = [1,2,3,4,9,2,8]
errors = []
for index, token in enumerate(tokens):
    actual = decoder.append("r", token)
    expected = decoder.model.logits(tokens[:index+1])
    error = float(np.max(np.abs(actual - expected)))
    errors.append(error)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
assert len(errors) == len(tokens)
decoder.kv.pool.check()
```

这里块大小三，输入长度七，确保走过第一页满、第二页满以及第三页只用一行。测试不把某个环境中打印的 `1.11e-16` 写死为唯一正确结果，而是检查 float64 对照所要求的误差界。若改用其他 dtype，应重新选择有依据的阈值并记录数值范围。

还可以逐位置比较完整 logits，而不只比较 argmax token。两个错误输出向量仍可能选中同一个最大项，因此 token 一致是一种较弱证据。反过来，两个 logits 只有很小误差但最大两项接近，也可能让贪心 token 改变。正确性检查应先观察数值，再解释采样决策。

### 3. 分叉后分别接不同后缀

对共同前缀 `[1,2,3]`，创建 A 后 fork 到 B，让 A 追加四，B 追加九。两个分支应该分别等价于完整输入 `[1,2,3,4]` 和 `[1,2,3,9]`。它们共享旧计算，但新后缀独立；正确性不能仅靠 token 列表不同来证明，还要比较实际 logits。

```python
forked = PagedTinyDecoder(pages=8, block_size=4, seed=11)
forked.create("A")
for token in [1,2,3]:
    forked.append("A", token)
forked.fork("A", "B")
output_a = forked.append("A", 4)
output_b = forked.append("B", 9)
np.testing.assert_allclose(output_a, forked.model.logits([1,2,3,4]), rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(output_b, forked.model.logits([1,2,3,9]), rtol=1e-12, atol=1e-12)
forked.kv.release("A")
forked.kv.release("B")
forked.kv.pool.check()
assert len(forked.kv.pool.free) == 8
```

同样的思路能用于 beam、候选分支或共享提示的实验，但不能据此宣称已经实现完整 beam search。真正搜索还需要候选分数、选择、重排、停止与资源回收策略。本章验证的是分支状态如何保持独立，这是搜索算法可以依赖的基础。

## 第十四课：三个故障工作坊，把症状追到具体机制

### 故障一：只有分叉后的第一步数值不对

观察到不分叉时完整对照通过，一旦共享未满尾页并追加，B 的输出错误，A 可能仍正确。首先比较两个分支 token 历史和页表：若 B 已拥有新页且历史正确，说明元数据 COW 很可能完成。下一步直接 gather B 的旧前三行，与 A 对照。

如果 B 的旧行是零或其他残留值，问题是新物理页没有复制有效 K/V；若 K 对但 V 错，检查是否只复制了 keys。最小修复是在元数据 COW 后复制旧有效切片，再写新增行。回归要同时检查 A 不变、B 前缀相同、B 新行正确，不能只比较最终某个 token。

一个可控的负向实验是在自己的测试副本中，COW 后把 B 的旧 value 行改成大值，再比较 logits。预期测试应失败。不要把这段故意破坏逻辑提交到参考实现；它的作用是证明你的测试确实能发现缺少数值复制，而不是只检查页号变化。

### 故障二：完整 gather 正确，在线结果只在第二页出错

先用人工分数 `[0,ln2]` 与 `[ln6]` 复现。如果 gather 的全局结果为二十三除以三，而在线结果偏离，打印每页 m、alpha、l、u。第二页最大值提高时，alpha 应为三分之一；若 alpha 被写成一，说明旧统计没有换到新尺度。

如果分母更新正确但分子没有乘 alpha，输出仍然错误。若只有多头输入失败，重点检查广播轴：maximum 是逐头量，不能在所有头上取一个全局 max；alpha 应扩成列向量再乘 numerator。通过改变 H 与 D，让二者不相等，可揭露那些依靠相同维度掩盖的广播错误。

修复后用多个页大小与长度重新比较在线和 gather，覆盖最后不足页。不要因为手算一个输入通过就删除原来的随机小规模对照。手算提供机制解释，随机对照覆盖更多组合，两者应同时保留。

### 故障三：第一次请求正常，页复用后出现离谱值

此类症状首先让你怀疑有效长度或状态重置，而不是模型权重突然变了。记录请求 A 释放前后的页号、refs、Page.data，以及 B 追加后的有效行数。若 B 只写一行，却在 blocks 中返回整页四行，就会读取旧请求残留。

正确修复应让 blocks 按当前页有效长度切片，同时确保 release 清除元数据。仅把整个浮点页清零可能暂时消除巨大值，但错误行仍参与 softmax 分母，不能作为根因修复。加入单有效 value 恒等于输出的测试，能直接检查读取范围是否正确。

三个故障分别来自数值复制、归一化合并与有效范围。它们可能都表现为“最终 logits 不一样”，却需要不同观察量。排错能力的核心是利用阶段性证据缩小范围，而不是不断调容差、换 seed 或增大页池直到错误不再出现。

## 第十五课：十二道递进练习与完整参考解答

### 练习一：把容量换成具体字节

设十六页、每页八 token、两个 KV 头、每头四维、float32。先计算一份 K，再算 K+V，最后假设十二层完全相同。参考推导：一份 K 有 `16*8*2*4=1024` 元素，每个四字节，共四千零九十六字节；K+V 八千一百九十二字节；十二层九万八千三百零四字节。这个结果不包括任何临时激活和元数据，不能直接当作进程总内存。

### 练习二：翻译逻辑位置

页大小三，页表 `[4,1,7]`，逻辑位置七、头一、坐标二应该读哪里？七除以三商二余一，因此读物理页七、槽一、头一、坐标二。若有人回答物理页二，是把逻辑页号当作物理页号。验收再换一张表 `[0,6,2]`，位置七就应读物理页二、槽一，证明你实际使用了映射。

### 练习三：证明页顺序不能排序

使用第四课 shuffled 池，其目标页表为 `[1,0]`。正确 gather 按该顺序读取；故意按页号排序的读法会得到不同历史。完整负向验证如下。它不修改池本身，只构造错误读法用于比较，因此不会破坏后续练习需要的状态。

```python
wrong_order = []
for page in sorted(shuffled.pool.tables["target"]):
    count = len(shuffled.pool.pages[page].data)
    wrong_order.append(shuffled.keys[page,:count])
wrong_k = np.concatenate(wrong_order)
assert not np.array_equal(wrong_k, k)
np.testing.assert_array_equal(shuffled.gather("target")[0], k)
```

### 练习四：拒绝 NaN 后检查资源

创建空请求，把 K 的一个坐标设为 NaN，调用 append。应报 ValueError，且请求仍为空、free 页数不变。完整参考如下。不要只断言“抛了某个错误”；如果错误在已经分配后才发生，同样会抛异常但违反本题要求。

```python
finite = TensorKV(pages=2, block_size=2, heads=1, dim=2)
finite.create("r")
invalid_k = np.zeros((1,1,2))
invalid_k[0,0,0] = np.nan
state_before = metadata_snapshot(finite)
try:
    finite.append("r", [1], invalid_k, np.zeros((1,1,2)))
except ValueError:
    pass
else:
    raise AssertionError("NaN was accepted")
assert metadata_snapshot(finite) == state_before
```

### 练习五：在真实 TensorKV 上复现错误平均反例

令 D=1、q=1，K 三行分别为零、ln2、ln6，V 为一、四、十，页大小二。这样点积分数恰好就是指定三数，无需额外缩放。gather 和 online 都应得到二十三除以三，而页内均值的平均为六点五。这个构造把纸上反例接到了真实代码，排除了“公式对了但接口弄错”的可能。

```python
counterexample = TensorKV(pages=2, block_size=2, heads=1, dim=1)
counterexample.create("r")
ck = np.log(np.array([1.0,2.0,6.0])).reshape(3,1,1)
cv = np.array([1.0,4.0,10.0]).reshape(3,1,1)
counterexample.append("r", [0,1,2], ck, cv)
for method in (counterexample.attention, counterexample.online_attention):
    np.testing.assert_allclose(method("r", np.ones((1,1))), [[23/3]])
```

### 练习六：检验每个头有自己的稳定化尺度

两个头的分数分别集中在一千附近和负一千附近。若每头各减最大值，两者仍能得到正常分布；若全局减一千，负一千头可能全部下溢为零。下面的完整例子固定三个位置，要求 online 与 gather 均有限且相近。它测试归一化尺度，不声称覆盖点积本身溢出的极端输入。

```python
heads_test = TensorKV(pages=3, block_size=2, heads=2, dim=1)
heads_test.create("r")
hk = np.array([[[1000.0],[-1000.0]], [[1001.0],[-999.0]], [[999.0],[-1001.0]]])
hv = np.array([[[1.0],[4.0]], [[2.0],[5.0]], [[3.0],[6.0]]])
heads_test.append("r", [0,1,2], hk, hv)
hq = np.ones((2,1))
assert np.isfinite(heads_test.online_attention("r", hq)).all()
np.testing.assert_allclose(heads_test.online_attention("r", hq), heads_test.attention("r", hq), rtol=1e-12, atol=1e-12)
```

### 练习七：解释完整尾页与部分尾页的资源差别

共享尾页有三行、块大小四、追加一行时，needed 为一，因为要复制部分尾页。共享尾页已经四行、追加一行时，needed 也是一，但原因是申请新逻辑页。两者数字相同，机制不同。参考答案必须画出页表变化：前者替换尾页号并保留旧三行，后者保留旧页号并在后面追加新页。不能仅凭 needed 数相同认为两条路径等价。

### 练习八：把页不足反例扩成跨页失败

使用第十课完整 limited 例子，解释为什么需要三张 free 而不是两张，并让异常后 K、V、引用、free、页表全部不变。参考实现就是该课的 metadata_snapshot 与两个数组 copy。继续在 sufficient 例子中用四页成功完成同样追加，验证 `[4,4,1]` 有效长度。这个题的核心是两个相邻容量边界，而不是随意把池调大。

### 练习九：证明缓存淘汰不损坏活跃请求

复用第十一课发布、attach、clear、release 的完整代码。参考解释：发布后缓存持有页；旧请求退出不回收；新请求 attach 增加自己的引用；clear 只减缓存引用；新请求仍能 gather；最后 release 才归还。若在 clear 后立即断言全部页 free，反而是在要求一个会破坏活动请求的错误行为。

### 练习十：增加一个 GQA 数值参考

当前 TensorKV 要求 query 头数等于 KV 头数。先不修改它，而是在外围写一个清楚的 gather 参考：四个 query 头共享两个 KV 头，规定前两个 query 读 KV 头零，后两个读 KV 头一。下面函数完整实现该映射，作为以后写分页 GQA Kernel 的对照。它有意 gather 并用 Python 循环，目标是语义清晰。

```python
def gqa_gather_reference(tensor_kv, rid, queries):
    queries = np.asarray(queries, dtype=tensor_kv.keys.dtype)
    if queries.ndim != 2 or queries.shape[1] != tensor_kv.dim:
        raise ValueError("query shape must be [query_heads, dim]")
    query_heads = queries.shape[0]
    if query_heads == 0 or query_heads % tensor_kv.heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if not np.isfinite(queries).all():
        raise ValueError("finite queries required")
    keys, values = tensor_kv.gather(rid)
    group = query_heads // tensor_kv.heads
    output = []
    for h in range(query_heads):
        kv_head = h // group
        scores = keys[:,kv_head] @ queries[h] / np.sqrt(tensor_kv.dim)
        p = np.exp(scores - scores.max())
        p /= p.sum()
        output.append(p @ values[:,kv_head])
    return np.stack(output)

gqa_queries = np.ones((4,1))
gqa_output = gqa_gather_reference(heads_test, "r", gqa_queries)
np.testing.assert_allclose(gqa_output[:2], np.repeat(heads_test.attention("r", hq)[:1], 2, axis=0))
np.testing.assert_allclose(gqa_output[2:], np.repeat(heads_test.attention("r", hq)[1:], 2, axis=0))
```

完整验收还应让各 query 向量不同，再用显式 repeat K/V 的普通多头公式作对照。分组索引用整除，而不是 `h % kv_heads`；后者会得到零、一、零、一，改变已声明分组。两组 KV 数值必须不同，否则错误映射也可能输出相同结果。

### 练习十一：比较多组页大小与随机数值

固定随机 seed，遍历页大小一、二、四、七，长度一、三、八、十三，使用三个头、每头五维。对每组比较 online 与 gather，最后检查页池。完整参考与仓库回归采用相同性质，换 seed 时应仍然成立。不能仅挑输出看起来正常的一组。

```python
generator = np.random.default_rng(42)
for block_size in [1,2,4,7]:
    for length in [1,3,8,13]:
        candidate = TensorKV(pages=20, block_size=block_size, heads=3, dim=5)
        candidate.create("r")
        candidate.append("r", list(range(length)), generator.normal(size=(length,3,5)), generator.normal(size=(length,3,5)))
        queries = generator.normal(size=(3,5))
        np.testing.assert_allclose(candidate.online_attention("r", queries), candidate.attention("r", queries), rtol=1e-12, atol=1e-12)
        candidate.pool.check()
```

### 练习十二：写一份能区分责任的实验报告

报告至少包含四组证据：页表乱序仍正确、部分尾页 COW 保留真实前缀、页不足没有半成品、逐 token logits 与完整重算一致。每组写输入、预期性质、实际观察和它不能证明的范围。参考结论应是“当前同步 float64 CPU 合同成立”，而不是“生产 PagedAttention 已经实现并达到某性能”。

如果加入耗时，必须分别统计投影、append、gather/online 和完整端到端，并说明 NumPy、BLAS 与 Python 循环的影响。不能拿一个小数组 CPU 结果推断 GPU 带宽，也不能把为了诊断保存快照的开销当成核心算子速度。先明确问题，再选观测方法，避免为了报告好看而测一个没有解释价值的数字。

## 第十六课：把正确性基线带入框架和 GPU 实现

### 1. 分页与在线注意力分别改变什么

分页改变历史的物理存储组织，使不同长度请求可以按块分配、共享和释放；在线注意力改变归一化求和的组织，使计算不必先物化完整分数矩阵。二者可以一起使用，也可以分别存在。连续数组可以做在线注意力，分页数组也可以先 gather 再做普通注意力。

假设 H 个头、T 个历史位置，gather 路线额外生成连续 K/V，元素数量约为 `2*T*H*D`；scores 又有 H*T 个元素。在线路线每次只需要当前页的分数 H*B，以及跨页保留的 m/l/u，其中 u 大小 H*D。这里说的是这个参考算法显式创建的数组量，真实执行库还可能有临时缓冲，不能直接当作精确峰值内存。

如果 T 很小，Python 逐页循环的开销可能使 online 比 gather 更慢，即使它避免某些大临时数组。这并不否定数学上的内存组织优势，只说明当前执行平台和输入规模下有其他成本主导。性能结论必须带上实现、形状和测量条件，不能从算法名字推断实际快慢。

### 2. 为 GPU Kernel 写清输入合同

一个分页 decode Kernel 至少需要 query、K/V 池、页表、每请求有效长度以及必要的 stride/布局参数。它不能只拿一个“总 token 数”，因为不同请求的页表和有效长度不同。若批次含 A、B，线程必须知道当前计算属于哪个请求、哪个 query 头，以及该头映射到哪个 KV 头。

访问某个历史位置时，先由逻辑位置确定逻辑页与槽，再从对应请求页表读取物理页号，最后用布局计算 K/V 地址。最后一页必须根据该请求长度屏蔽无效槽。GQA 还要使用明确的头映射，量化缓存则要读取与该数据匹配的 scale 等元信息。每加一项功能，都增加一条需要测试的数据依赖。

GPU 并行会把页或 tile 分配给线程组。若多个组分别计算部分 m/l/u，最后还需要按相同尺度合并这些统计。不能把每组已归一化的输出直接平均，也不能在没有同步关系时让一个组读取另一个组尚未完成的部分和。K05、K07、K09 分别提供归约、在线注意力和 Triton 实现的详细路径。

### 3. 从本章生成可复用的测试输入

先用可辨认数值测试地址，再用随机小数组测试一般数值，再用污染无效槽验证长度。至少覆盖物理页乱序、非满尾页、页大小不同、query/KV 头数不同、请求长度不同，以及最大分数出现在后页的输入。完整无缓存模型对照应在这些基础机制稳定后再接入，以免所有错误混在最终 logits 中。

从 CPU 导出数据时，应保存数值 dtype、shape、页表和有效长度，而不是只保存某个最终输出。运行 GPU 后先比较每请求每头输出，再分析误差如何随长度和精度变化。若输出不一致，能回到原输入重现，才有继续优化的可靠基础。

某些 CPU 断言不适合原封不动用于低精度 GPU。例如 float64 的 `1e-12` 误差界不能机械要求 float16 路径满足；但页表身份、有效范围、取消后不再发布结果等逻辑性质不能因为换硬件就随意放宽。数值容差与状态正确性应该分开制定。

### 4. 接入调度时先固定请求状态协议

本章 append 以 request ID 和新 token 为输入，已有 Engine 的 Backend.logits 则接受完整 token 历史。这是两种不同协议，不能把一个函数名替换掉就假定生命周期自动一致。增量后端需要知道何时创建、追加多少、是否有可复用前缀、取消时释放什么，以及抢占后选择保留、移动还是重算状态。

[P2 的 IncrementalRunner](../../projects/incremental_runtime.py) 提供完整可运行的每请求缓存生命周期参考：先完成 prefill，再发布请求；推进时先成功计算新状态再更新输出；完成或取消时清除请求缓存；FIFO 与轮转都与完整重算结果对照。它使用连续多层 NumPy K/V，和本章的单层分页数值池各自保留清楚边界。

把二者组合时，建议先保留现有两条参考路径，增加第三条实现作对照。不要删除完整重算基线，也不要把原先只管理 token 页的接口悄悄改成包含真实张量却不更新其他调用者。先列出 create/append/fork/release/attention 的合同，再逐个用本文的故障输入证明它们，最后接到调度策略。

这项组合不要求一开始就实现 GPU、多卡和所有模型结构。你已有完整的组成机制与可运行参考，选择一次清楚的接口组合就能形成有意义的工程改动。验收重点是新增路径确实读写真实状态，并能解释其资源归属；输出某段像语言的文本并不能替代这些证据。

## 结业检查：能否从一个位置一直解释到一次释放

请不看正文，给一个逻辑位置翻译出物理页与槽位，列出该行 K/V 的 shape 和字节数；再用两页不等权重的例子推导 m/l/u 更新。随后实际完成部分尾页分叉、资源不足、旧页复用三组实验，说明每个断言分别验证了什么。

最后沿 PagedTinyDecoder.append 从 token 走到 logits，再沿 release 走到空闲页，指出数值数组为何仍然分配、哪些行已经无效、哪些引用仍可能保留状态。如果你能在陌生输入上解释这条链，并从故意破坏的复制、mask 或重标度中定位错误，才算掌握本章，而不只是记住“PagedAttention 能节省显存”。

源码研读时继续使用 S09、S17 的固定提交路线，寻找真实实现中的页表、cache layout、模型 runner 和 Kernel 参数。把真实对象对应到本章合同，同时记录那些本章没有实现的异步、并发、分布式与精度机制。对应关系帮助你阅读代码，边界记录帮助你避免把教学参考误当生产实现的完整复制品。
