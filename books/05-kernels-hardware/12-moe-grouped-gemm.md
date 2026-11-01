# K12 MoE执行器：从路由表、分桶与恢复顺序到Grouped GEMM和反向

MoE模型把不同token送到不同专家，专家内部仍可以是熟悉的线性层与激活，困难在于任务数量和数据位置不再规整。本章专门研究执行层：路由结果怎样变成专家输入，怎样为不同大小矩阵分配工作，怎样把多个专家贡献正确加回原token，以及这些步骤如何影响反向、通信和性能。

先修K01索引、K06矩阵乘法、K10梯度累加。模型卷解释路由学习与MoE结构，本章把它们落实为可运行的NumPy执行器。所有核心索引与数学实验都可在CPU完成；单卡或多卡GPU用于后续有界验证，不能把CPU模拟结果当成真实通信或Grouped GEMM性能。

## 第一课：Token、Expert与Assignment三个数量

### 1. 稠密层为什么规整

稠密线性层把T个token组成T乘D矩阵，所有行乘同一个D乘F权重，得到T乘F输出。任务规模由T、D、F决定，所有token走同一条路径。MoE则为每个token选择一个或多个专家，每个专家有自己的权重。

设专家数E，每token选择R个专家，路由表ids形状T乘R，gate同shape。每个表项对应一次assignment，即一个token向某个专家的一次计算请求。总assignment数通常为T乘R，但最终输出仍只有T行，因为同token的多个专家结果需要合并。

### 2. Assignment不是新的独立token

同一个token可以在多个专家输入中出现，甚至教学接口允许同一专家重复出现于两个slot。输入向量被多次使用，输出贡献必须逐assignment加权。把assignment数量当成最终token数量，会在恢复输出shape和性能统计中出错。

例如四个token、每个两个专家，共八个assignment。专家计数之和应为八，但某个专家可能零个，另一个可能六个。平均值不能反映最长任务，也不能保证每个专家至少有输入。执行器必须处理空专家与极端偏斜。

### 3. 先固定一个简单专家数学

本章先用线性专家fₑ(x)=xWₑ，W形状E乘D乘F。最终输出yₜ=Σᵣgₜᵣ fᵢdₛₜᵣ(xₜ)。这个公式清楚表达三个索引：token t、slot r、专家id。后面可把线性函数替换为两层MLP，路由与归并结构仍保留。

gate是否和为一由模型定义决定，执行器不应擅自归一化。有的模型使用全专家softmax后选top-k并保留原权重，有的对选中权重重新归一化。两种结果不同，必须由路由接口明确提供，而不是看到概率就自动除以和。

## 第二课：先写最慢但最透明的参考

~~~python
import numpy as np

def moe_naive(x, expert_ids, weights, gates):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    expert_ids = np.asarray(expert_ids)
    gates = np.asarray(gates, dtype=np.float64)
    if x.ndim != 2 or weights.ndim != 3 or expert_ids.ndim != 2:
        raise ValueError('X, W, IDs must have ranks 2, 3, 2')
    if expert_ids.shape != gates.shape or expert_ids.shape[0] != x.shape[0]:
        raise ValueError('routing shape mismatch')
    if x.shape[1] != weights.shape[1]:
        raise ValueError('feature width mismatch')
    if not np.issubdtype(expert_ids.dtype, np.integer):
        raise ValueError('integer expert IDs required')
    if np.any(expert_ids < 0) or np.any(expert_ids >= weights.shape[0]):
        raise ValueError('expert ID out of range')
    if not all(np.isfinite(a).all() for a in (x, weights, gates)):
        raise ValueError('finite numerical inputs required')
    output = np.zeros((x.shape[0], weights.shape[2]))
    for token in range(x.shape[0]):
        for slot in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, slot])
            value = x[token] @ weights[expert]
            output[token] += gates[token, slot] * value
    return output
~~~

### 1. 为什么用显式两层路由循环

外层token保证最终输出归属清楚，内层slot逐个处理assignment。expert从ids读取，不是循环slot本身。一个常见错误是把slot零当专家零、slot一当专家一，这只在特殊路由表上碰巧正确。

output先清零，因为每个token需要累加多个贡献。若用empty再加，会混入未初始化内容；若每次用赋值，会只保留最后一个专家。这个参考不追求速度，但每个输出贡献都能追溯到一个明确assignment。

### 2. 完整两专家手算

输入x为一、二，专家零为单位矩阵，专家一为二倍单位矩阵，gate为零点二五与零点七五。两个专家输出分别为一、二与二、四，加权后为零点二五、零点五与一点五、三，总输出一点七五、三点五。

若两个slot都指向专家一，gate为零点四与零点六，输出仍为二、四。重复assignment是合法的累加测试，不应被普通覆盖写丢掉。即使真实router保证top-k专家唯一，执行层测试允许重复输入仍有助于发现scatter逻辑缺陷；若接口明确拒绝重复，也应有拒绝测试。

## 第三课：把路由表展开成可审计记录

### 1. 每条记录保存什么

对每个assignment保存token_id、slot_id、expert_id、gate。token_id用于从X取输入并最终归并，slot_id用于对应gate和可能的路由梯度，expert_id用于选择权重，gate用于加权。只保存expert_id不足以恢复原token顺序。

将T乘R表按行优先展开，原assignment编号a对应token=a整除R、slot=a模R。这个编号可以作为稳定排序后的逆向标识。前向打包后即使数据顺序改变，仍能通过记录恢复每条贡献的来源。

### 2. 完整四token例子

设ids为[[二,零],[一,二],[零,零],[二,一]]。展开expert序列为二、零、一、二、零、零、二、一。对应token序列零、零、一、一、二、二、三、三，slot序列零、一、零、一、零、一、零、一。

按专家稳定分组后，专家零得到原assignment一、四、五；专家一得到二、七；专家二得到零、三、六。计数为三、二、三，前缀边界为零、三、五、八。专家零包含token二两次，因此后续归并必须保留两份gate贡献。

### 3. 稳定排序的意义与边界

稳定排序保证相同专家内保留原assignment顺序，便于可复现调试与逆映射。它不自动保证GPU浮点累加逐bit确定，因为后续矩阵计算或原子归并可能采用不同顺序。排序确定性与完整数值确定性是不同层次。

若性能实现使用原子计数分配桶内位置，桶内顺序可能随调度变化，只要记录完整、归并正确，数学语义仍可成立。若业务要求固定顺序或可复现，应选择相应算法并记录代价，不能假设所有分桶都稳定。

## 第四课：计数、前缀和与稳定打包

~~~python
from dataclasses import dataclass

@dataclass
class RoutingPlan:
    order: np.ndarray
    token: np.ndarray
    slot: np.ndarray
    expert: np.ndarray
    gate: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray

def make_routing_plan(expert_ids, gates, experts):
    ids = np.asarray(expert_ids)
    gates = np.asarray(gates, dtype=np.float64)
    if ids.ndim != 2 or gates.shape != ids.shape:
        raise ValueError('routing matrix and gates required')
    if experts <= 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('positive expert count and integer IDs required')
    if np.any(ids < 0) or np.any(ids >= experts) or not np.isfinite(gates).all():
        raise ValueError('invalid routing data')
    tokens, slots = ids.shape
    flat_ids = ids.reshape(-1)
    order = np.argsort(flat_ids, kind='stable')
    flat_token = np.repeat(np.arange(tokens), slots)
    flat_slot = np.tile(np.arange(slots), tokens)
    counts = np.bincount(flat_ids, minlength=experts)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return RoutingPlan(order, flat_token[order], flat_slot[order],
                       flat_ids[order], gates.reshape(-1)[order],
                       counts, offsets)

ids = np.array([[2, 0], [1, 2], [0, 0], [2, 1]])
gates = np.array([[0.7, 0.3], [0.4, 0.6],
                  [0.2, 0.8], [0.5, 0.5]])
plan = make_routing_plan(ids, gates, 3)
assert np.array_equal(plan.order, [1, 4, 5, 2, 7, 0, 3, 6])
assert np.array_equal(plan.counts, [3, 2, 3])
assert np.array_equal(plan.offsets, [0, 3, 5, 8])
~~~

### 1. 前缀边界如何定位专家片段

专家e的数据位于packed数组的offsets[e]到offsets[e加一]之前，长度等于counts[e]。空专家的两个边界相等，自然形成空切片。offsets长度E加一，最后一个值等于总assignment数，这个不变量应始终检查。

bincount统计每个专家被选多少次，不是统计不同token数。重复assignment也要计入，因为它对应独立gate贡献。若你想去重，需要先证明模型语义允许把相同专家的gate合并，而不是在计数阶段悄悄删除。

### 2. Gather输入与元数据必须同序

packed_x等于X按plan.token取行，packed_gate等于按同一order排列的gate。若只排序输入而忘记排序gate，shape仍正确，结果却把别人的权重乘到当前assignment上。使用每个slot不同gate的测试能发现这种错位。

同样，恢复输出时必须使用排序后的token映射，而不是原展开token序列。代码中两个数组长度一样，并不意味着可以互换。给变量命名为original与packed有实际价值，可以避免在复杂dispatch代码中混淆顺序。

## 第五课：Grouped执行与Scatter-Add

~~~python
def moe_grouped(x, expert_ids, weights, gates):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    # Reuse the naive checker once, then compare independent execution structure.
    reference = moe_naive(x, expert_ids, weights, gates)
    plan = make_routing_plan(expert_ids, gates, weights.shape[0])
    packed_x = x[plan.token]
    packed_output = np.zeros((plan.token.size, weights.shape[2]))
    for expert in range(weights.shape[0]):
        begin, end = plan.offsets[expert:expert + 2]
        if begin == end:
            continue
        packed_output[begin:end] = packed_x[begin:end] @ weights[expert]
    weighted = packed_output * plan.gate[:, None]
    output = np.zeros_like(reference)
    np.add.at(output, plan.token, weighted)
    return output, plan, packed_output

x = np.array([[1., 2.], [3., 4.], [-1., 1.], [2., -2.]])
weights = np.array([np.eye(2), 2 * np.eye(2),
                    np.array([[1., 1.], [0., 1.]])])
grouped_output, plan, packed_output = moe_grouped(x, ids, weights, gates)
assert np.allclose(grouped_output, moe_naive(x, ids, weights, gates))
assert np.allclose(grouped_output,
                      [[1., 2.7], [4.2, 7.4], [-1., 1.], [3., -2.]])
~~~

### 1. 完整手算四行输出

token零的一、二经过专家二得到一、三，gate零点七贡献零点七、二点一；专家零贡献零点三、零点六，合并一、二点七。token一的三、四经过专家一得到六、八，乘零点四得到二点四、三点二；专家二得到三、七，乘零点六得到一点八、四点二，总和四点二、七点四。

token二的负一、一两次进入专家零，gate零点二与零点八合并后仍是负一、一。token三的二、负二经过专家二得到二、零，乘零点五得到一、零；专家一得到四、负四，乘零点五得到二、负二，总和三、负二。每个输出都能从assignment表逐项恢复。

### 2. 为什么不能普通高级索引加等于

NumPy的out[indices]加等于values在重复索引时不等价于逐次scatter-add，因为高级索引可能先产生临时数组，再写回，重复位置贡献可能丢失。np.add.at明确按索引累加，适合教学正确参考。

GPU实现可以采用原子加、按token独占归并或先排序归约等策略。普通store只适合唯一写者。top-k大于一时，即使专家id不重复，同token仍有多个专家贡献，所以最终归并天然需要合并，不应只在重复专家测试时才考虑。

### 3. 为什么这里调用一次naive检查

示例为了复用完整输入验证并立即建立独立结果对照，先计算naive参考。正式高性能wrapper不应每次都运行慢参考，应把合同检查与参考计算拆开。教学代码明确展示这层额外工作，避免把它当作实际性能实现。

进一步工程化时，提取validate_inputs函数供两条路径调用，reference只在测试开启。不要为了避免重复代码而让生产执行器偷偷进行完整CPU计算，也不要为了速度删除必要参数验证。

### 4. 不用排序也能理解分桶：计数与游标

稳定分桶可以分两遍完成。第一遍统计每个专家的assignment数，前缀和给出每个桶的起始位置；第二遍按原assignment顺序遍历，用该专家当前游标决定写入位置，然后游标加一。因为按原顺序遍历，同专家内顺序自然稳定。

这个算法清楚分离三个数组：counts是总容量，offsets是桶边界，cursor是当前写入进度。不能把cursor原地修改后又当作原offsets使用，否则后续切片起点会变成桶末尾。保留边界与进度两份状态，是许多并行packing实现中的基本习惯。

~~~python
def stable_bucket_order(flat_ids, experts):
    flat_ids = np.asarray(flat_ids, dtype=np.int64)
    if np.any(flat_ids < 0) or np.any(flat_ids >= experts):
        raise ValueError('invalid expert')
    counts = np.bincount(flat_ids, minlength=experts)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    cursor = offsets[:-1].copy()
    order = np.empty(flat_ids.size, dtype=np.int64)
    for assignment, expert in enumerate(flat_ids):
        position = cursor[expert]
        order[position] = assignment
        cursor[expert] += 1
    assert np.array_equal(cursor, offsets[1:])
    return order, counts, offsets

bucket_order, bucket_counts, bucket_offsets = stable_bucket_order(ids.reshape(-1), 3)
assert np.array_equal(bucket_order, plan.order)
~~~

第二遍结束时，每个cursor应恰好等于对应桶末尾，证明写入数量与计数一致。order中的每个原assignment编号应恰好出现一次，证明没有重复或遗漏。两个不变量分别检查容量与排列，最好同时验证。

### 5. GPU并行分桶中的原子游标

若多个线程同时向同一专家桶写入，需要为每条assignment分配唯一位置。原子加可以返回旧游标，让每个线程获得不同槽位，但完成顺序可能不固定，因此桶内顺序未必稳定。只要后续保留正确token-slot映射，数学结果仍可正确。

不能先普通读取cursor，再普通写回cursor加一，因为两个线程可能读到相同旧值并写同一槽。这个竞争与最终scatter-add是两种不同阶段的问题：分桶阶段需要唯一分配位置，归并阶段需要合并数值贡献。两者都可能用原子，但语义不同。

如果需要稳定并行分桶，可以使用分块计数、块间前缀和与块内稳定位置等更复杂方案。先用CPU稳定参考验证最终排列关系，再研究并行实现。不要把稳定性、唯一性与性能混成一个模糊要求。

### 6. Inverse permutation能恢复什么，不能恢复什么

order是原assignment编号到packed顺序的一种排列，可以构造inverse，使inverse[order[p]]=p。它能把每个assignment输出恢复到原token-slot顺序。但T乘R个assignment恢复后，仍需要沿slot求和，才能得到T个token输出。

因此“逆排列”不等于“最终combine”。前者是一一对应的顺序恢复，后者是多对一的加权归约。把两个步骤分开实现，有助于调试；高性能代码可以融合，但必须保留同样的数学关系。

~~~python
inverse_order = np.empty_like(plan.order)
inverse_order[plan.order] = np.arange(plan.order.size)
restored_assignment_output = packed_output[inverse_order]
restored_assignment_gate = plan.gate[inverse_order]
combined_by_slots = (
    restored_assignment_output * restored_assignment_gate[:, None]
).reshape(ids.shape[0], ids.shape[1], -1).sum(axis=1)
assert np.allclose(combined_by_slots, grouped_output)
~~~

这个独立combine路径与np.add.at路径应得到相同结果。一个通过逆排列恢复后按slot求和，一个直接按token scatter-add，表达方式不同，适合互相验证。若两者不一致，先检查order、inverse与gate是否使用同一排列。


## 第六课：Top-K与Gate语义

### 1. 从logit得到选择与权重

router可以先输出每个token对E个专家的logit，再选择若干专家。选择前是否softmax、选择后是否重新归一化、是否加入噪声或约束，都由模型定义。执行器收到ids与gates后应忠实执行，不重新解释。

假设全专家概率为零点六、零点三、零点一，选前两个。保留原权重时和为零点九；重新归一化后变成三分之二与三分之一。若两个专家输出相同向量，前者输出该向量的零点九倍，后者输出原向量，因此差异可直接手算。

### 2. 稳定Top-K教学实现

~~~python
def router_topk(logits, top_k, renormalize=True):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not 1 <= top_k <= logits.shape[1]:
        raise ValueError('valid top-k for a logit matrix required')
    if not np.isfinite(logits).all():
        raise ValueError('finite logits required')
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-probabilities, axis=1, kind='stable')
    selected = order[:, :top_k]
    gates = np.take_along_axis(probabilities, selected, axis=1)
    if renormalize:
        gates = gates / gates.sum(axis=1, keepdims=True)
    return selected, gates

selected, selected_gates = router_topk(np.log([[0.6, 0.3, 0.1]]), 2)
assert np.array_equal(selected, [[0, 1]])
assert np.allclose(selected_gates, [[2/3, 1/3]])
~~~

稳定排序使相同概率时较早专家编号优先，这是本教学实现明确的平局规则。真实router可能采用不同选择机制，不能要求所有库与此逐bit一致。测试路由执行器时最好直接提供ids与gates，先把选择算法与执行算法分开验证。

### 3. 离散选择与梯度边界

top-k索引是离散选择，对索引本身不能像连续变量那样直接求普通导数。给定固定选择后，gate与专家输出的连续路径可以求导；选择边界附近的行为需要模型训练方法明确处理。不能因为gate来自softmax，就认为整个top-k过程处处光滑。

本章反向练习固定ids，计算X、W与gates的梯度，清楚限定数学问题。模型卷再讨论路由学习、辅助损失与相关训练策略。执行层不应偷偷加入新的路由梯度近似。

## 第七课：Padding、容量与有效数量

### 1. 为矩阵tile补齐专家行数

若每个专家的输入行数补齐到BM倍数，计数三、二、三在BM四时变成四、四、四，共十二行，其中八行真实、四行padding。有效比例三分之二。补齐让矩阵路径规整，却增加计算与存储，尤其小专家任务浪费明显。

padding输入可以填零，但最终归并必须只处理真实assignment。若专家包含bias，零输入的输出不一定零；若有非线性或状态，情况更复杂。因此“padding行输入零”不能替代有效数量mask，必须保留真实count。

### 2. 完整padding计划

~~~python
def padded_counts(counts, block_rows):
    counts = np.asarray(counts, dtype=np.int64)
    if block_rows <= 0 or np.any(counts < 0):
        raise ValueError('valid counts and block size required')
    padded = ((counts + block_rows - 1) // block_rows) * block_rows
    offsets = np.concatenate([[0], np.cumsum(padded)])
    return padded, offsets

padded, padded_offsets = padded_counts([3, 2, 3], 4)
assert np.array_equal(padded, [4, 4, 4])
assert np.array_equal(padded_offsets, [0, 4, 8, 12])
assert 8 / padded.sum() == 2/3
~~~

空专家计数零，补齐后仍零，通常不需要启动计算。若某硬件路径强制最小tile，需要明确额外占位与mask，不能把它计入真实token吞吐。报告性能时同时列真实assignment数与执行padding行数，避免虚高数字。

### 3. Capacity改变语义

容量上限限制每专家最多接收多少assignment。超出后丢弃、重路由或延迟处理会改变模型行为，不能由执行器为了方便静默决定。若模型定义允许丢弃，需要明确选择保留哪些assignment、gate是否重新归一化以及输出缺失贡献如何处理。

教学默认不丢弃，容量不足直接报错或扩大缓冲。这样数学参考保持清楚。研究容量策略时，应把它作为独立路由变换，先生成新的有效assignment表，再交给执行器，避免把策略隐藏在打包过程里。

### 4. Padding执行器为何必须保留有效行映射

假设某专家真实三行，补齐到四行，专家函数为x乘W加bias。第四行输入全零，但输出等于bias，不是零。如果把第四行的默认token编号零拿去scatter，token零会多收到一份bias。这个错误在无bias线性专家测试中可能完全隐藏。

正确做法是为每个padded槽保存有效标志或仅遍历真实count。padding槽的token编号可以设为哨兵负一，但不能依赖NumPy负索引自动忽略，因为负一会访问最后一个token。必须在索引之前过滤有效性。

~~~python
def moe_padded_linear_bias(x, ids, weights, gates, bias, block_rows=4):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    plan = make_routing_plan(ids, gates, weights.shape[0])
    if bias.shape != (weights.shape[0], weights.shape[2]):
        raise ValueError('one bias vector per expert required')
    output = np.zeros((x.shape[0], weights.shape[2]))
    used_rows = 0
    for expert, count in enumerate(plan.counts):
        padded_count = ((int(count) + block_rows - 1) // block_rows) * block_rows
        used_rows += padded_count
        if count == 0:
            continue
        begin, end = plan.offsets[expert:expert + 2]
        buffer = np.zeros((padded_count, x.shape[1]))
        buffer[:count] = x[plan.token[begin:end]]
        computed = buffer @ weights[expert] + bias[expert]
        valid_result = computed[:count] * plan.gate[begin:end, None]
        np.add.at(output, plan.token[begin:end], valid_result)
    return output, used_rows

expert_bias = np.array([[1., 2.], [3., 4.], [5., 6.]])
padded_result, used_rows = moe_padded_linear_bias(
    x, ids, weights, gates, expert_bias)
expected_bias_result = np.zeros_like(padded_result)
for token in range(ids.shape[0]):
    for slot in range(ids.shape[1]):
        expert = ids[token, slot]
        expected_bias_result[token] += gates[token, slot] * (
            x[token] @ weights[expert] + expert_bias[expert])
assert np.allclose(padded_result, expected_bias_result)
assert used_rows == 12
~~~

buffer真实行之外都为零，computed却全部包含bias；切片computed[:count]是保证padding不进入combine的关键。这个例子把“无效计算可以发生”与“无效结果不能参与语义”分开，适合迁移到其他分块算子。

### 5. 容量策略的可观察结果

若专家容量为二，而某专家收到三条assignment，需要明确哪一条被拒绝或转移。按原顺序保留前二、按gate保留最大二、随机保留二，都是不同策略。它们可能影响训练与推理结果，不能由底层线程竞争顺序偶然决定。

容量处理后，counts之和不再等于原T乘R，而应等于保留assignment数。记录丢弃数、每token剩余slot数与gate处理方式，才能解释输出变化。若某token全部assignment被丢弃，需要定义零输出、残差回退或其他模型行为，不能留下未初始化输出。

本章默认无丢弃执行器，容量实验作为独立变换。这样原始数学参考稳定，新增策略可以用自己的预期表验收。工程上把策略与执行拆开，也便于后来更换调度而不悄悄改变模型语义。


## 第八课：Grouped GEMM的任务表

### 1. 每个专家是不同M的矩阵乘法

专家e收到Mₑ行，计算Mₑ乘D与D乘F的乘法。D与F可能相同，Mₑ却动态变化。逐专家单独launch简单，但许多小矩阵可能受到启动与调度开销影响。Grouped GEMM把多个问题的tile放进共同任务组织，减少某些开销并改善负载分配。

它不等于把所有专家权重横向拼成一个大矩阵，让每个token计算全部专家。那会产生大量不需要的交叉结果，改变稀疏执行的目的。正确grouped调度仍确保每个assignment只使用被选择专家的权重。

### 2. 从专家计数生成tile数量

给定BM与BN，专家e的行tile数为Mₑ向上除以BM，列tile数为F向上除以BN，任务数是二者乘积。对所有专家任务数做前缀和，就能把全局任务编号映射到专家与专家内tile坐标。

~~~python
def grouped_tile_tasks(counts, output_width, bm, bn):
    if min(output_width, bm, bn) <= 0:
        raise ValueError('positive dimensions required')
    tasks = []
    column_tiles = (output_width + bn - 1) // bn
    for expert, count in enumerate(counts):
        for row_tile in range((int(count) + bm - 1) // bm):
            for column_tile in range(column_tiles):
                tasks.append((expert, row_tile, column_tile))
    return tasks

tasks = grouped_tile_tasks([3, 0, 5], 7, 2, 4)
assert len(tasks) == 10
assert all(expert != 1 for expert, _, _ in tasks)
assert tasks[-1] == (2, 2, 1)
~~~

### 3. 任务编号与输出所有权

每个任务的输出行范围属于该专家packed片段，列范围属于输出特征tile。不同任务不应重叠写同一packed输出元素。最终scatter-add是另一阶段，负责把不同assignment贡献合并到token，不能把这两种写者关系混淆。

若采用持久化worker循环领取任务，任务表数学映射不变，只改变哪个worker执行哪个tile。K17会详细讲调度与生命周期。本章先用CPU枚举全部任务，检查每个合法packed输出恰好覆盖一次，再讨论动态调度性能。

### 4. 全局tile编号的前缀解码

显式任务列表适合教学，但大型实现可能不想为每个tile存完整三元组。可以只保存每专家任务数的前缀和，用全局任务号查找所属专家，再解码专家内行列tile。查找可以采用二分或其他调度结构，正确性首先取决于边界定义。

设任务数四、零、六，前缀为零、四、四、十。任务号四应属于专家二，而不是空专家一；因此查找要处理重复边界。用右侧插入点减一可以跳过零长度区间，随后局部任务号为全局号减该专家起点。

~~~python
def decode_grouped_task(task, counts, output_width, bm, bn):
    counts = np.asarray(counts, dtype=np.int64)
    ntiles = (output_width + bn - 1) // bn
    task_counts = ((counts + bm - 1) // bm) * ntiles
    prefix = np.concatenate([[0], np.cumsum(task_counts)])
    if not 0 <= task < prefix[-1]:
        raise IndexError('task out of range')
    expert = int(np.searchsorted(prefix, task, side='right') - 1)
    local = task - int(prefix[expert])
    return expert, local // ntiles, local % ntiles

decoded_tasks = [decode_grouped_task(t, [3, 0, 5], 7, 2, 4)
                 for t in range(10)]
assert decoded_tasks == grouped_tile_tasks([3, 0, 5], 7, 2, 4)
~~~

### 5. 任务数均衡与计算量均衡并不相同

如果所有tile形状相同且K相同，每个完整tile计算量相近，但尾tile有效行数不同，实际有用工作不同。若专家维度不同、采用不同量化格式或激活，任务成本差异更大。按任务数平均分配不一定等于按执行时间均衡。

动态worker领取任务能减少某些尾部等待，但增加调度状态与可能的原子开销。静态分配更简单、可预测，却可能在偏斜分布下利用率不足。性能选择应基于实际路由直方图与任务成本，而不是只看专家总数。


## 第九课：固定路由下的完整反向

### 1. 三类梯度

对assignment输出z=xWₑ、最终贡献g z，上游对token输出的梯度为dY。对专家输出的梯度是g dY；对输入x贡献是g dY Wₑ转置；对权重贡献是x转置乘g dY；对gate梯度是dY与z的内积。

同token多个assignment的dX需要累加，同专家多个token的dW需要累加，gate梯度则对应每个token-slot位置。前向中的两种共享关系在反向产生两类归约，不能只实现其中一种。

~~~python
def moe_backward_fixed_routes(x, ids, weights, gates, grad_output):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.float64)
    ids = np.asarray(ids)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    expected = moe_naive(x, ids, weights, gates)
    if grad_output.shape != expected.shape:
        raise ValueError('output gradient shape mismatch')
    dx = np.zeros_like(x)
    dw = np.zeros_like(weights)
    dg = np.zeros_like(gates)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            value = x[token] @ weights[expert]
            upstream = gates[token, slot] * grad_output[token]
            dx[token] += upstream @ weights[expert].T
            dw[expert] += np.outer(x[token], upstream)
            dg[token, slot] = np.dot(grad_output[token], value)
    return dx, dw, dg
~~~

### 2. 重复专家的梯度也要重复贡献

若同token两个slot都指同一专家，dX与dW按两个gate贡献相加，dg的两个位置各自等于相同专家输出与上游内积。把重复assignment去重但不合并gate，会改变梯度；合并gate后若还需要原slot梯度，则必须保留逆映射。

固定ids的有限差分只扰动X、W与gates，不扰动整数索引。这样测试的是明确光滑函数。若gate被约束和为一，直接独立扰动gate测试的是执行器输入合同，而不是完整router参数化后的梯度；两层应分开理解。

### 3. 固定路由梯度的完整有限差分

~~~python
def finite_difference_moe(function, value, step=1e-5):
    value = np.asarray(value, dtype=np.float64).copy()
    result = np.empty_like(value)
    for index in np.ndindex(value.shape):
        old = value[index]
        value[index] = old + step
        plus = float(function(value))
        value[index] = old - step
        minus = float(function(value))
        value[index] = old
        result[index] = (plus - minus) / (2 * step)
    return result

upstream_moe = np.array([[1., -1.], [0.5, 2.], [-2., 1.], [3., 0.25]])
dx_moe, dw_moe, dg_moe = moe_backward_fixed_routes(
    x, ids, weights, gates, upstream_moe)
loss_moe = lambda xx, ww, gg: np.sum(
    moe_naive(xx, ids, ww, gg) * upstream_moe)
numeric_x = finite_difference_moe(lambda z: loss_moe(z, weights, gates), x)
numeric_w = finite_difference_moe(lambda z: loss_moe(x, z, gates), weights)
numeric_g = finite_difference_moe(lambda z: loss_moe(x, weights, z), gates)
assert np.allclose(dx_moe, numeric_x, rtol=1e-5, atol=1e-7)
assert np.allclose(dw_moe, numeric_w, rtol=1e-5, atol=1e-7)
assert np.allclose(dg_moe, numeric_g, rtol=1e-5, atol=1e-7)
~~~

上游包含不同正负值，避免某些梯度路径因对称而消失。ids保持固定，重复专家案例也保留，能检查dX与dW是否正确累加。对gate做独立扰动没有保持行和一，这是执行器连续输入的梯度；若要测试router参数梯度，应把softmax参数化接在外层再求导。


## 第十课：从单卡分桶到专家并行通信

### 1. 专家位置增加新的索引

当专家分布在多个设备，需要把全局expert_id映射到目标rank与本地expert_id。assignment记录还要保留源rank、源token位置与slot，返回时才能恢复。只保留目标本地专家编号会丢失全局身份，不同rank上的专家零并不是同一个专家。

发送前通常按目标rank分组，再在目标内部按本地专家分组，具体布局取决于实现。两次分组的逆映射必须组合正确。可以先在CPU用列表模拟发送桶与返回桶，验证每条assignment的源信息保持不变，再运行真实通信。

### 2. 通信与计算的依赖

专家计算只能读取已经到达并完成写入的输入块，结果返回只能读取已经完成的专家输出，源端归并只能消费已经收到的结果。可以按块流水重叠，但每条依赖都要有明确事件或通信完成条件，不能凭异步函数返回就当作数据可用。

多GPU正确性还要求各rank匹配通信调用与数据数量，错误可能表现为挂起而不是数值异常。先两卡小数据验证，再考虑八卡性能；不需要为了学习本地路由与Grouped GEMM直接占满八张H100。K18给出集合通信与有界实验的完整路线。

### 3. 负载不均会放大尾部延迟

一次MoE层可能要等待最慢专家或最慢rank完成。平均每专家token数相同，不代表最大负载相同；少数热点专家可能主导延迟。调度器可以改善任务利用，但不能凭空消除模型路由造成的数据与计算需求。

负载均衡训练损失影响未来路由分布，运行时调度处理当前分布，容量策略改变哪些任务被执行，三者职责不同。性能分析应分别记录路由直方图、padding、通信量与计算时间，避免把所有问题统称为“MoE慢”。

## 第十一课：三个故障案例

第一类是排序X后gate未同步排序。使用每个slot不同gate、专家权重明显不同的小例子，检查packed记录逐行是否保持原token-slot配对。修复应统一使用同一个order重排所有关联元数据。

第二类是scatter普通覆盖或高级索引加等于丢失重复贡献。用token同时路由两个专家或重复同专家，手算加权和，修复为真正scatter-add或独占token归并。只测试top-one会漏掉此类错误。

第三类是padding行经过带bias专家后产生非零输出并被错误归并。修复保留真实count与有效assignment映射，padding不进入最终combine。输入零不代表专家输出零，必须从完整专家函数判断。

## 第十二课：十二道练习与完整答案

### 练习一：数量关系

题目：T五、R二、E三。答案：十个assignment，计数和十，输出仍五行，不能保证每专家非空或均分。

### 练习二：前缀和

题目：counts二、零、三。答案：offsets零、二、二、五，专家一切片为空，专家二从位置二开始。

### 练习三：重复路由

题目：x三、四，专家二倍单位矩阵，两gate零点四与零点六。答案：输出六、八；覆盖写可能只留下三点六、四点八，忘gate得到十二、十六。

### 练习四：稳定分组

答案：本章四token例子的排序原assignment编号为一、四、五、二、七、零、三、六。token与slot必须用同一顺序重排。

### 练习五：Top-K归一化

答案：概率零点六、零点三、零点一，选前二后保留原权重和零点九；重归一化为三分之二与三分之一。执行器不擅自选择模式。

### 练习六：Padding比例

答案：counts三、二、三，BM四，执行十二行、真实八行，有效率三分之二。真实吞吐不应把padding当token。

### 练习七：Grouped任务数

答案：counts三、零、五，BM二，F七、BN四，任务数分别四、零、六，总十。空专家无任务。

### 练习八：Gate移动位置

答案：对单个线性专家，标量gate可在输入或输出乘，数学等价；若专家包含非线性或bias，通常不能任意移动，必须推导完整函数。

### 练习九：固定路由反向

答案：dX累加g乘dY乘W转置，dW累加x外积g乘dY，dGate为dY与专家输出内积。整数ids不在此梯度合同内。

### 练习十：非法专家ID

答案：负数、超过E减一或非整数类型应在读取权重前拒绝，不能依赖数组负索引语义意外访问最后专家。

### 练习十一：容量溢出

答案：默认执行器不静默丢弃，容量不足报错或显式扩容；若模型规定丢弃或重路由，先作为独立策略生成有效assignment表并测试。

### 练习十二：完整验收集

答案包括空专家、所有token同专家、重复assignment、非均匀gate、非整除专家行数、非法ID、不同输出宽度、固定路由梯度与通信源位置恢复。每个案例都对应明确数学或索引不变量。

## 第十三课：用CPU模拟一次专家并行往返

### 1. 给每条消息保留源身份

本地执行器只需要token与slot，多rank执行器还需要source_rank。两个rank都可能有token零，它们不是同一条输入。消息至少包含源rank、源token、slot、目标专家、gate和输入向量，返回时必须携带能恢复源位置的标识。

专家位置表把全局expert_id映射到目标rank。例如三个专家分别位于rank零、rank一、rank一。目标rank收到消息后仍需知道具体专家，而不是把所有消息都交给一个权重。通信分桶与专家分桶可以分层进行，但两层身份都不能丢。

### 2. 完整两rank合成模拟

~~~python
def expert_parallel_cpu(inputs_by_rank, ids_by_rank, gates_by_rank,
                        weights, expert_owner):
    weights = np.asarray(weights, dtype=np.float64)
    expert_owner = np.asarray(expert_owner, dtype=np.int64)
    ranks = len(inputs_by_rank)
    if expert_owner.shape != (weights.shape[0],):
        raise ValueError('one owner per expert required')
    if np.any(expert_owner < 0) or np.any(expert_owner >= ranks):
        raise ValueError('owner rank out of range')
    mailboxes = [[] for _ in range(ranks)]
    send_counts = np.zeros((ranks, ranks), dtype=np.int64)
    outputs = [np.zeros((len(x), weights.shape[2])) for x in inputs_by_rank]
    for source in range(ranks):
        x_local = np.asarray(inputs_by_rank[source], dtype=np.float64)
        ids_local = np.asarray(ids_by_rank[source])
        gates_local = np.asarray(gates_by_rank[source], dtype=np.float64)
        moe_naive(x_local, ids_local, weights, gates_local)
        for token in range(x_local.shape[0]):
            for slot in range(ids_local.shape[1]):
                expert = int(ids_local[token, slot])
                destination = int(expert_owner[expert])
                mailboxes[destination].append(
                    (source, token, slot, expert,
                     float(gates_local[token, slot]), x_local[token].copy()))
                send_counts[source, destination] += 1
    returned = [[] for _ in range(ranks)]
    for destination, messages in enumerate(mailboxes):
        for source, token, slot, expert, gate, vector in messages:
            assert expert_owner[expert] == destination
            value = gate * (vector @ weights[expert])
            returned[source].append((token, slot, value))
    for source, messages in enumerate(returned):
        for token, slot, value in messages:
            outputs[source][token] += value
    return outputs, send_counts

inputs_rank = [x[:2], x[2:]]
ids_rank = [ids[:2], ids[2:]]
gates_rank = [gates[:2], gates[2:]]
outputs_rank, count_matrix = expert_parallel_cpu(
    inputs_rank, ids_rank, gates_rank, weights, [0, 1, 1])
assert np.allclose(np.concatenate(outputs_rank), grouped_output)
assert count_matrix.sum() == ids.size
assert np.array_equal(count_matrix, [[1, 3], [2, 2]])
~~~

这段模拟没有真实网络、NCCL、异步可见性或设备内存，只验证消息身份与数学往返。mailboxes按目标rank分组，returned按源rank分组，消息中source与token共同定位最终输出。输入向量copy避免后续修改影响已发送的教学消息。

计数矩阵的行表示源rank，列表示目标rank。源rank零的四条assignment中，一条去rank零、三条去rank一；源rank一两条去各自目标。总和八等于全局assignment数。真实All-to-All变长通信需要双方对发送与接收数量达成一致，这张矩阵是理解协议的起点。

### 3. 返回顺序可以变化，身份不能变化

真实通信与专家计算可能以不同顺序完成，返回列表不一定与发送列表同序。只要每条消息保留源token与slot，并采用正确累加，结果不依赖完成顺序的数学排列。有限精度下累加顺序可能影响末位，但不应丢失或错配贡献。

可以在CPU模拟中反转每个返回桶，再比较结果，验证实现没有依赖隐含顺序。若结果发生大幅变化，检查是否用列表位置代替消息身份。这个测试能在没有多GPU环境时发现大量路由往返错误。

### 4. 通信量如何估算

每条assignment发送D个输入元素，返回F个输出元素，还要携带必要元数据。若输入输出float16，纯向量数据约为二乘assignment数乘D加F字节；真实实现可能通过共享输入、压缩元数据或不同返回布局改变流量。这个账本只描述逻辑数据，不包含协议、对齐与重复缓冲。

top-k增加assignment数量，通常也增加dispatch与combine工作。某些同rank专家路径可以避免跨设备传输，但仍有本地packing成本。统计时区分本地与远程assignment，不能把总数直接等同网络消息数。

### 5. 两卡实验的最小验收

先使用每rank两个token、三个或四个专家、每token两个slot，专家权重选单位矩阵与不同倍数，输出可以手算。验证发送计数、接收计数、每条源身份、专家选择和最终结果，再扩展随机shape。不要一开始就用巨大模型和真实请求调试通信。

真实GPU实验还要保证所有rank执行匹配协议，异常时能终止并释放资源。CPU模拟不能证明这些运行时行为，因此记录中应明确“索引模拟通过”与“真实多卡通过”两条证据。K18的有界通信实验提供后续步骤。

## 第十四课：负载、容量、专家函数与性能解释

### 1. 平均负载不足以描述执行时间

设十六条assignment分给四个专家。分布四、四、四、四与十三、一、一、一平均都为四，但最大负载差异很大。若专家计算并行且最终要等全部完成，热点专家可能决定尾部延迟。均值相同不代表吞吐或延迟相同。

可以记录最大值、最小非零值、空专家数、最大除平均以及标准差等指标。指标只是描述分布，不自动给出优化方案。某些偏斜来自真实输入语义，不能为了让图更均匀就擅自改变路由；调度与训练策略需要各自明确目标。

~~~python
def routing_statistics(counts, block_rows):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts < 0):
        raise ValueError('nonnegative expert counts required')
    padded, _ = padded_counts(counts, block_rows)
    total = int(counts.sum())
    mean = float(counts.mean())
    return {
        'assignments': total,
        'empty_experts': int(np.count_nonzero(counts == 0)),
        'maximum': int(counts.max()),
        'max_over_mean': float(counts.max() / mean) if mean else 0.0,
        'padded_rows': int(padded.sum()),
        'useful_fraction': float(total / padded.sum()) if padded.sum() else 1.0,
    }

balanced_stats = routing_statistics([4, 4, 4, 4], 4)
skewed_stats = routing_statistics([13, 1, 1, 1], 4)
assert balanced_stats['max_over_mean'] == 1
assert skewed_stats['max_over_mean'] == 3.25
assert balanced_stats['useful_fraction'] == 1
assert skewed_stats['padded_rows'] == 28
~~~

### 2. Padding与负载偏斜可能同时存在

十三行补到十六，一行各补到四，总执行二十八行，真实十六行，有效率约零点五七一。热点专家仍然长，冷专家又浪费很多padding。只优化一个平均tile大小未必同时解决两类问题，可能需要不同小任务路径或动态调度。

Grouped GEMM减少逐专家launch，并不自动消除padding，也不保证热点任务被均匀拆分。应分别记录任务数量、每任务有效比例、资源占用与调度尾部。性能诊断要对应具体机制，而不是把所有改善都归功于“grouped”这个名字。

### 3. 两层MLP与门控专家的布局

真实专家常包含第一层投影、激活或门控、第二层投影。若第一层输出宽度为H，第二层从H映射回D或其他输出宽度，中间激活可能按专家packed行存储。两次GEMM的M相同，K与N角色不同，最佳tile与量化布局也可能不同。

门控MLP可能并行计算两个投影，再逐元素组合，例如一个分支经过激活后乘另一个分支。可以融合某些中间操作减少写回，但必须保持每条assignment与专家权重对应。不要在第一层打包顺序与第二层恢复顺序之间丢失映射。

### 4. Gate为何通常在专家结果之后加权

对纯线性无bias函数，g乘(xW)等于(gx)W，所以gate可在输入侧或输出侧乘。加入bias后，g乘(xW+b)与(gx)W+b不同，后者没有缩放bias；加入非线性后，一般也不满足交换。执行器不能为了融合方便擅自移动gate位置。

用x为一、权重一、bias二、gate零点五手算。正确输出零点五乘三等于一点五，先缩放输入再加bias得到二点五，明显不同。这个极小反例应加入组合专家测试，防止某次性能优化改变数学语义。

### 5. 负载均衡损失与执行器日志

训练中的辅助目标可能鼓励专家使用更均衡，但执行器只看到当前ids与gates。它可以报告计数、容量溢出、padding和时间，为模型训练提供观察，却不应在底层偷偷调整gate以改善日志。否则前向函数与训练目标之间会出现隐藏变化。

记录分布时注意隐私与成本，只保存合成或聚合统计即可，不需要记录真实token内容。教学工程可以用固定随机路由与手工偏斜分布完成大部分实验，避免依赖公司数据才能复现。

### 6. 性能实验的对照应保持语义一致

比较逐专家GEMM、padded batched路径与Grouped路径时，使用相同ids、gates、专家函数和精度。若某路径丢弃超容量任务、另一条完整计算，速度不能直接比较。若一条包括dispatch/combine、另一条只计GEMM，也必须明确区分局部与端到端指标。

报告真实token数、assignment数、top-k、专家数、各专家count、特征维、量化格式、padding、通信拓扑与计时边界。缺少路由分布的MoE性能数字很难解释，因为同样T与E可以产生完全不同任务结构。

### 7. 一份可维护的路由计划对象

工程中可以让RoutingPlan保存原始与packed映射、counts、offsets、有效mask和版本信息，并提供validate方法检查守恒关系。执行器只消费已经验证的计划，策略层负责top-k、容量与重路由，通信层负责rank映射。清楚分层可以减少索引被重复计算且相互不一致的风险。

计划对象也可能占用显著内存，尤其top-k与token数大时。优化时可压缩索引类型或融合某些步骤，但要先证明最大索引范围与生命周期，不能为省几个字节使用会溢出的类型。正确性基线建立后，才有可靠空间讨论元数据成本。


### 8. 路由计划的守恒检查

一个正确计划应满足：counts之和等于assignment数，offsets从零开始且非递减，最后边界等于总数，每个桶内expert_id相同，order是原assignment编号的排列，token与slot仍在合法范围。任何一条失败，都应在读取专家权重前停止。

~~~python
def validate_routing_plan(plan, token_count, slots, experts):
    assignments = token_count * slots
    if plan.order.shape != (assignments,):
        raise AssertionError('wrong order length')
    if not np.array_equal(np.sort(plan.order), np.arange(assignments)):
        raise AssertionError('order is not a permutation')
    if plan.counts.shape != (experts,) or plan.offsets.shape != (experts + 1,):
        raise AssertionError('wrong count metadata shape')
    if plan.offsets[0] != 0 or plan.offsets[-1] != assignments:
        raise AssertionError('wrong prefix endpoints')
    if not np.array_equal(np.diff(plan.offsets), plan.counts):
        raise AssertionError('count/prefix mismatch')
    if np.any(plan.token < 0) or np.any(plan.token >= token_count):
        raise AssertionError('token index out of range')
    if np.any(plan.slot < 0) or np.any(plan.slot >= slots):
        raise AssertionError('slot index out of range')
    for expert in range(experts):
        begin, end = plan.offsets[expert:expert + 2]
        if not np.all(plan.expert[begin:end] == expert):
            raise AssertionError('mixed expert bucket')
    return True

assert validate_routing_plan(plan, 4, 2, 3)
~~~

这些检查只验证计划结构，不证明gate与原表对应，也不证明输入向量按正确token读取。因此还要检查plan.token等于order整除slots，plan.slot等于order模slots，以及plan.gate等于原gate扁平后按order重排。结构守恒与语义配对需要同时成立。

若执行容量丢弃策略，order不再是全部assignment的排列，而是一个无重复子集，验证器必须按新合同调整。不能直接关闭所有断言让容量路径通过，应明确哪些守恒关系改变、哪些仍必须保持，例如保留数加丢弃数等于原总数。

### 9. 空token集合与空slot的合同

T为零时，没有assignment，所有专家count为零，输出shape为零乘F。这个路径可以不启动专家计算，仍应返回明确dtype与shape。R为零则表示每个token没有任何专家贡献，数学上可定义零输出，也可以接口拒绝；本章执行公式自然得到零，但模型层通常应避免无意产生这种路由。

空专家与空token不同。空专家只是某个桶没有任务，其他专家照常执行；空token集合则整个层没有真实工作。区分两者能避免对空数组做最大值、错误分配零grid或把全局无工作误当某个专家异常。

### 10. Gate与专家输出的关联测试

如果所有专家函数都相同且每token gate和为一，MoE输出应等于该公共专家函数。若gate和为其他值，输出应按该和缩放。这个性质能检查combine与gate配对，但不能验证不同专家权重选择，因此还需专家互不相同的测试。

把某个未被选择专家的权重任意改变，输出应不变；把某个被选择专家的权重改变，只有选择它的token可能受影响。这个干预测试直接对应稀疏路由语义，可以发现错误地计算或混入其他专家输出的实现。

### 11. 从错误输出定位到具体阶段

如果输出行顺序整体错乱，优先检查逆排列与源token映射；如果只有多slot时偏小，检查scatter覆盖与gate；如果某专家相关token全部异常，检查专家权重或scale布局；如果只在非整除count时失败，检查padding与真实有效数。

把调试输出按assignment记录，而不是只按最终token打印，可以看见每个专家贡献是否正确。先比较单条专家结果，再比较加权结果，最后比较combine。分阶段证据能防止在最终输出错误时盲目修改router或GEMM。


### 12. 手算小数与机器结果的比较

本章四token例子中，零点七乘三再加零点三乘二，在二进制浮点中可能得到与直接写字面量二点七略有末位差异的结果。因此数值断言使用合理的近似比较；索引排列、count与offset等整数结构则应精确相等。把两类检查混用，会制造假失败或掩盖真错误。

稳定排序使累加顺序更可预测，但并不把十进制小数变成二进制精确数。不同合法归并顺序可能出现极小差异，应该用误差分析解释；若少掉一个assignment或gate错位，误差通常大得多，不应靠放宽容差掩盖。

调试记录应保留实际最大误差与位置，而不仅是通过或失败。若某次优化把误差从机器末位提升到明显比例，先检查顺序、精度与语义是否改变。这样既不会因正常舍入误判，也不会把真正的数据丢失包装成数值噪声。

## 第十五课：综合项目——一个可解释的MoE执行器

第一阶段完成naive参考与手算表，第二阶段完成稳定分桶、packed计算与scatter-add，第三阶段加入padding与Grouped任务表，第四阶段实现固定路由反向并做有限差分。每次改变数据顺序，都保存明确逆映射与数量不变量。

第五阶段再将专家矩阵计算搬到单GPU，保留CPU路由参考；随后用两GPU小任务验证发送与返回索引，最后按需要扩展八卡。每阶段明确代码、数值与性能证据，不把模型最终loss或生成文本当作底层索引正确性的唯一验收。

完成后应能从一个输出token追溯到全部assignment、专家权重、gate与packed位置，也能从一个Grouped tile追溯到其输出所有权。掌握这些关系，才有能力阅读并修改真实MoE后端。

最后再检查一个没有数值计算的条件：每条保留的assignment必须恰好发送一次、计算一次、返回一次并归并一次。不同阶段可以重排、分块和并行，但不能改变这四个计数。把它写进调试日志，能在复杂多卡实现中先排除任务丢失，再研究浮点误差。

任何任务去重都必须先证明数学等价。
