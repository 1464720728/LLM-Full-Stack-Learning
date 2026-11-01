# M09　局部、块稀疏与压缩注意力：连接集合、选择误差与因果边界

[本卷导航](README.md) · 前置：[注意力](03-attention.md) 与 [缓存](06-kv-gqa.md) · 下一章：[线性注意力](10-linear-attention.md)

标准因果 attention 允许当前位置读取全部过去，长度增加时合法连接数量按平方增长。本章研究三种不同的改变：只保留一部分原始连接，把连接组织成规则块，以及把一段历史压缩成新的记录再选择。它们都可能减少工作量，但改变函数的方式不同。你将先写清允许集合，再建立同一定义的稠密参考，最后实现真正只访问选中记录的路径。

这里的“稀疏”特指 token 之间的 attention 连接，不是 MoE 的专家选择，也不是权重矩阵里出现很多零。若某实现先计算全部分数再遮挡，它在数学上使用稀疏连接，但没有省掉全部被遮挡点积。区分模型定义与执行方式，是本章最重要的出发点。

## 第一课　用集合定义窗口，避免边界猜测

令当前位置为 i，历史键位置为 j。窗口大小 W 在本章包含当前位置，因此合法集合为 $A_i=\{j\mid0\le j\le i,\ j>i-W\}$。等价地，j 从最大值零与 i 减 W 加一开始，到 i 结束。W 等于一时只看自己；W 等于三时最多看自己和前两个位置。任何实现前先写出这个集合，不要只凭一个叫 window 的参数猜语义。

长度五、窗口三时，各行允许索引依次为零；零和一；零、一、二；一、二、三；二、三、四。连接数为一加二加三加三加三，等于十二。完整因果 attention 连接数为一加二加三加四加五，等于十五。窗口在序列开头被边界截短，所以不能直接用 T 乘 W 当作精确连接数。

当 T 大于等于 W，窗口连接数为前 W 行的三角数加后面每行 W，即 $W(W+1)/2+(T-W)W$。整理为 $TW-W(W-1)/2$。当 T 小于 W，窗口覆盖全部过去，连接数回到 $T(T+1)/2$。这个分段公式可以用来检查实现是否漏掉当前 token 或多读一个历史位置。

```python
import numpy as np
from labs.models.core import attention,softmax,local_global_mask,sparse_attention

def window_mask(length,window):
    if length<1 or window<1:
        raise ValueError("positive length and window required")
    i=np.arange(length)[:,None]
    j=np.arange(length)[None,:]
    return (j<=i)&(j>i-window)

def window_edges(length,window):
    effective=min(length,window)
    return effective*(effective+1)//2+(length-effective)*effective

mask5=window_mask(5,3)
assert [np.flatnonzero(row).tolist() for row in mask5]==[
    [0],[0,1],[0,1,2],[1,2,3],[2,3,4]]
assert mask5.sum()==12
for length in range(1,12):
    for window in range(1,8):
        assert window_mask(length,window).sum()==window_edges(length,window)
```

有些库把窗口参数定义成只计算过去多少个 token，不包含当前；有些接口同时提供左窗口和右窗口；还有些模型交替使用局部与全局层。迁移时必须重新核对集合，不要因为参数同名就认为等价。最容易发现 off-by-one 的测试是 W 等于一、T 小于 W，以及最后一行的允许索引。

## 第二课　局部层如何传播远处信息

单层窗口只能直接读取附近位置，但多层可以间接传播更远信息。假设每层窗口三，位置六在第一层读四、五、六；第二层读这些位置的第一层表示，而位置四的第一层又读二、三、四，因此第二层位置六可间接依赖二到六。每增加一层，最远依赖向前扩展最多 W 减一，L 层的理论感受范围最多为一加 L 乘 W 减一。

这不等于顶层能像全 attention 一样自由读取范围内每个原始 token。远处信息经过中间位置的隐藏向量压缩和非线性变换，可能被保留，也可能丢失。直接连接和多跳传播在路径长度、容量与训练难度上不同。不能只算出感受野覆盖一万 token，就断言模型具备与全注意力相同的精确检索能力。

残差让每个位置保留自身信息，但不会自动把远处信息瞬间送到当前。全局 token、周期性全局层、压缩摘要或检索支路可以缩短长程路径。它们分别引入新的连接或状态，不是窗口本身已经解决的问题。架构演进往往来自这种矛盾：局部计算便宜，但长程依赖需要额外通路。

可以用布尔可达性矩阵验证多层依赖。A 的第 i 行第 j 列为真表示一层可读；两层可达性由布尔意义上的 A 与 A 相乘得到，只要存在中间节点使两条边相连即可。普通数值矩阵乘法结果大于零可作为小规模参考，但不要把路径数量当作 attention 权重，它们只描述可能的信息流。

```python
one_layer=window_mask(7,3)
two_layers=(one_layer.astype(int)@one_layer.astype(int))>0
np.testing.assert_array_equal(np.flatnonzero(two_layers[6]),[2,3,4,5,6])
three_layers=(two_layers.astype(int)@one_layer.astype(int))>0
np.testing.assert_array_equal(np.flatnonzero(three_layers[6]),np.arange(7))
```

这个测试只证明依赖通路存在。若某层权重恰好为零，实际影响可能为零；若训练没有学会保留相关特征，通路也未必有用。结构分析给出上限与可能性，任务实验验证实际能力。将二者分开，可以避免对稀疏模式作过度承诺。

## 第三课　全局键与集合并集：重复记录会改变概率

一种简单长程补充是让每隔 G 个位置的键成为全局键，允许所有后续查询读取它。因果条件仍然必须成立：全局并不意味着可以读取未来。定义全局集合为所有不晚于 i 且位置能被 G 整除的 j，再与局部窗口取并集。并集中的同一位置只能出现一次。

长度四、窗口二、全局步长三时，位置三的局部集合为二和三，全局集合为零和三，并集是零、二、三。若直接把两个列表拼接而不去重，位置三会出现两次。在 softmax 中，重复一条相同键值相当于给它增加未归一化权重，改变输出。重复不是无害的存储细节。

取三个合法记录分数都为零，值分别零、三、六，正确并集概率各三分之一，输出三。若最后一条重复一次，四个概率各四分之一，输出为零加三加六加六除四，等于三点七五。这个最小例子说明，即使键完全相同，重复计数也会改变归一化质量。

```python
mask_global=local_global_mask(4,window=2,global_stride=3)
np.testing.assert_array_equal(np.flatnonzero(mask_global[3]),[0,2,3])
unique_output=softmax(np.zeros((1,3)))@np.array([[0.],[3.],[6.]])
duplicate_output=softmax(np.zeros((1,4)))@np.array([[0.],[3.],[6.],[6.]])
np.testing.assert_allclose(unique_output,[[3.]])
np.testing.assert_allclose(duplicate_output,[[3.75]])
```

全局键的表示是否包含你需要的远处信息，要看它自己的因果可见性和训练。位置零作为全局键，只能包含序列开头的信息，不会随着后来 token 到来自动变成整篇摘要；因果模型中旧位置状态不会被未来更新。若希望一个摘要记录持续汇总新信息，需要另一个明确的状态更新机制。把固定全局 token 想成不断更新的外部记忆，会违反因果模型的计算定义。

全局查询与全局键也不同。全局键是让许多后续位置能读它；全局查询是让某个位置能读更广范围。双向编码器中可以定义特殊全局 token 同时拥有两种方向的连接，因果解码器则还需限制未来。讨论任何全局模式时，都应分别说明谁能读谁，而不是只画一个醒目的节点。

## 第四课　同一 mask 的稠密参考与稀疏执行

稠密参考先计算全部 QK 转置，再把非法分数设为负无穷，softmax 后读 V。它易于实现与验证，但被遮挡点积仍已计算。稀疏执行则先取得合法索引，只取对应 K/V，再计算当前 query 的分数。两者在同一个 mask 下应数学等价。这个等价是实现验证，不是说稀疏模型与全历史模型等价。

```python
def dense_masked_reference(q,k,v,mask):
    scores=q@k.T/np.sqrt(q.shape[-1])
    scores=np.where(mask,scores,-np.inf)
    probabilities=softmax(scores)
    return probabilities@v,probabilities

def gather_reference(q,k,v,allowed):
    output=np.zeros((len(q),v.shape[-1]))
    visited=0
    for i,indices in enumerate(allowed):
        indices=np.asarray(indices,dtype=int)
        if len(indices)==0:
            continue
        scores=q[i]@k[indices].T/np.sqrt(q.shape[-1])
        probabilities=softmax(scores)
        output[i]=probabilities@v[indices]
        visited+=len(indices)
    return output,visited

rng=np.random.default_rng(9)
q=rng.normal(size=(7,3))
k=rng.normal(size=(7,3))
v=rng.normal(size=(7,2))
mask=local_global_mask(7,window=3,global_stride=4)
allowed=[np.flatnonzero(row) for row in mask]
dense_output,_=dense_masked_reference(q,k,v,mask)
gather_output,visited=gather_reference(q,k,v,allowed)
library_output,library_visited=sparse_attention(q,k,v,mask)
np.testing.assert_allclose(gather_output,dense_output,atol=1e-12)
np.testing.assert_allclose(library_output,dense_output,atol=1e-12)
assert visited==library_visited==int(mask.sum())
```

这里返回 visited 是为了记录实际参与点积的连接数，而不是猜测稀疏度。它仍不等于实际运行时间：Python 循环、索引 gather、临时数组和小矩阵乘法可能比一次大矩阵调用更慢。算法减少运算是一项证据，硬件加速是另一项证据。课程 CPU 参考不宣称生产性能。

空允许集合在本章定义输出零，避免全负无穷 softmax 产生 NaN。对正常包含自身的因果窗口不会出现空行，但外部 mask 或错误索引可能造成。接口应明确选择报错还是零输出，不能依赖某个库碰巧返回什么。若训练损失使用这样的空行，还要明确梯度与残差行为。

## 第五课　完整手算：稀疏归一化与删除后的变化

一个查询对四个键的分数为零、一、二、三，值分别一、二、四、八。全 attention 的未归一化权重为一、e、e 平方、e 立方，分母是四者之和。若只允许索引零和二，稀疏分布不是把全概率中另外两项设零后直接使用，而是重新在保留集合上归一化，概率为一除一加 e 平方，以及 e 平方除一加 e 平方。

对应输出为一乘第一概率加四乘第二概率，约三点六四二四。若先做全 softmax 再把非法位置概率乘零，保留权重和小于一，输出幅度被缩小；它实现的是另一个函数。正确 mask 应在 softmax 之前作用，或在删除后重新计算归一化分母。

现在假设分数最高的索引三被删除。无论剩下记录计算得多精确，都无法读取它的值八。稀疏选择的质量问题与核心 attention 的数值正确性是两个层次。kernel 可以完全正确地计算一个选择错误的稀疏模型；反过来，选择策略很好，错误的归一化也会损坏结果。

```python
scores=np.array([[0.,1.,2.,3.]])
values=np.array([[1.],[2.],[4.],[8.]])
keep=np.array([[True,False,True,False]])
correct_prob=softmax(np.where(keep,scores,-np.inf))
wrong_prob=softmax(scores)*keep
np.testing.assert_allclose(correct_prob.sum(axis=1),1.)
assert wrong_prob.sum()<1.
correct_value=correct_prob@values
np.testing.assert_allclose(correct_value,[[(1+4*np.exp(2))/(1+np.exp(2))]])
assert not np.allclose(correct_value,wrong_prob@values)
```

如果局部与全局分别计算 attention，再把输出相加或平均，也一般不等于并集 attention。因为两个子集合各有自己的分母，丢失了跨集合的相对权重。可以通过正确合并未归一化统计量、最大值和分母实现等价的分块 softmax，但那需要显式公式，不能简单把两个已经归一化的输出相加。

## 第六课　块稀疏：规则性与额外连接的取舍

GPU 常以矩形 tile 处理矩阵。若按单个连接任意稀疏，索引和访存不规则，可能难以利用高吞吐矩阵指令。块稀疏把连接按固定大小的块选择：一个块被选中，就计算块内一组分数，再用细粒度 mask 处理必要边界。这样可以牺牲部分理论稀疏度，换取规则计算与数据复用。

设块大小二，查询位置零和一属于查询块零，键位置二和三属于键块一。一个块级布尔矩阵描述哪些查询块读取哪些键块。展开到 token 级后，还必须与因果 mask 相交，否则同一块内后面的 token 会泄漏给前面的 token。选择下三角块并不自动保证块内严格因果；对角块需要额外三角屏蔽。

如果从滑动窗口 mask 出发，只要块内有一个合法连接就启用整个块，展开后可能多出原窗口之外的连接。你可以保留原细粒度 mask，使函数仍等价于原窗口，但块内无效位置的部分计算仍发生；也可以接受扩展后的连接集合，定义一个新的块稀疏模型。两种选择应分别报告“逻辑合法连接数”和“执行覆盖元素数”。

```python
def block_cover(mask,block_size):
    rows,cols=mask.shape
    block_rows=(rows+block_size-1)//block_size
    block_cols=(cols+block_size-1)//block_size
    selected=np.zeros((block_rows,block_cols),dtype=bool)
    expanded=np.zeros_like(mask)
    for bi in range(block_rows):
        for bj in range(block_cols):
            rs=slice(bi*block_size,min((bi+1)*block_size,rows))
            cs=slice(bj*block_size,min((bj+1)*block_size,cols))
            if mask[rs,cs].any():
                selected[bi,bj]=True
                expanded[rs,cs]=True
    return selected,expanded

fine=window_mask(7,2)
blocks,covered=block_cover(fine,2)
causal=np.tril(np.ones((7,7),dtype=bool))
expanded_model=covered&causal
assert np.all(expanded_model[fine])
assert expanded_model.sum()>=fine.sum()
assert not expanded_model[np.triu_indices(7,1)].any()
```

边界块可能不足完整大小，例如长度七、块大小二，最后块只有一个 token。内核仍可能按完整 tile 发射，但必须屏蔽越界加载与写入。数学 mask、内存边界 mask 和块选择是三个不同条件，代码中常常需要同时存在。算子实现时若把它们混成一个判断，很容易在最后一块出错。

## 第七课　压缩记录：它已经不是原始 token 的简单子集

删除连接保留的是原始 K/V；压缩则把多个历史位置变成一个新记录。最简单压缩是块均值：一个块的键为块内 K 的平均，值为块内 V 的平均。它易于手算，但通常不等价于对原始记录做 attention。因为 softmax 的指数非线性不能穿过均值，query 对块内不同键的偏好也被压平。

用两个键一和负一、两个值十和负十、查询一举例。原始 attention 分数为一和负一，输出为十乘两个概率之差，约七点六一五九。均值键为零、均值值为零，压缩后只有一个记录，输出零。原来 query 能偏向正键对应的正值，压缩后这组配对信息已经丢失，任何后续精确 softmax 都不能恢复。

即使块内键完全相同，简单均值记录也需要考虑块大小对归一化的影响。若一个块包含 m 个相同键，每个键对应一个值，原始 attention 中这个块的总未归一化质量是 m 乘指数分数；压缩成一个均值记录后，质量只剩一个指数分数。若想在这个特殊条件下保持块总质量，可以给压缩分数加 log m，同时用均值 value。但一般块内键不同，仍不能靠一个常数精确还原。

```python
q_compress=np.array([[1.]])
k_compress=np.array([[1.],[-1.]])
v_compress=np.array([[10.],[-10.]])
original_read=attention(q_compress,k_compress,v_compress)[0]
mean_read=attention(q_compress,k_compress.mean(axis=0,keepdims=True),
                    v_compress.mean(axis=0,keepdims=True))[0]
np.testing.assert_allclose(original_read,[[10*np.tanh(1.)]])
np.testing.assert_allclose(mean_read,[[0.]])
assert abs(original_read[0,0])>7.
```

因此压缩模型的正确性测试不能要求它等于全原始 attention。应该先定义压缩器、记录集合、位置语义和归一化规则，再建立同一压缩定义的显式参考。全 attention 可以作为质量或近似误差对照，但它不是同一函数。把预期近似差异叫作 kernel bug，和把真实实现错误叫作近似误差，都是不准确的。

真实压缩器可以有学习参数、卷积、重叠窗口、门控或其他结构。本章用均值，是为了让信息丢失和因果边界完全可见，不把它称为某个具体前沿模型的完整实现。研究新压缩架构时，应逐项核对压缩器的输入范围、输出维度、位置处理与训练方式，而不是只看到一个 compression ratio。

## 第八课　压缩块的因果性：完成时间决定可读时间

假设压缩块覆盖位置四和五。位置四的查询不能读取这个完整块均值，因为均值包含尚未允许的未来位置五。即使压缩键的代表位置被标成四，数据内容仍来自五，因果泄漏不会因为换一个位置标签而消失。合法性必须依据压缩器实际读取的原始范围，而不是摘要记录的名字。

课程压缩参考采用非重叠块，并要求整个块位于当前局部窗口之前。设当前位置 i，窗口 W，局部起点为 max(0,i-W+1)。块区间从 start 到 end，不含 end，只有 end 小于等于局部起点才可作为压缩候选。这样既保证块完全属于过去，也避免同一原始区间同时作为压缩记录和局部原始记录重复进入本例。

避免重叠是本教学定义，不是所有模型必须遵守的定律。某些架构可能有意同时提供压缩摘要和局部原始记录，让模型学习两种表示的关系。那时重复覆盖是设计的一部分，应有对应训练与归一化。你不能把本章去重规则强加给所有模型，也不能把一个无意重叠的 bug 解释成“也许模型会学会”。

完整手算边界：窗口二、压缩块大小二。位置二的局部起点一，块零到二尚未完全位于局部窗口之前，所以没有压缩块；位置三的局部起点二，块零到二合法，局部记录为二和三；位置四的局部起点三，块零到二仍合法，块二到四会与局部窗口位置三重叠，因此不合法。边界条件应使用块结束位置，不是只检查块起点。

```python
def eligible_blocks(position,window,compression):
    local_start=max(0,position-window+1)
    result=[]
    for start in range(0,local_start,compression):
        end=start+compression
        if end<=local_start:
            result.append((start,end))
    return local_start,result

assert eligible_blocks(2,2,2)==(1,[])
assert eligible_blocks(3,2,2)==(2,[(0,2)])
assert eligible_blocks(4,2,2)==(3,[(0,2)])
assert eligible_blocks(5,2,2)==(4,[(0,2),(2,4)])
```

因果性测试应改变未来 K/V 或原始输入，检查所有更早输出不变。只检查 mask 是下三角不够，因为压缩记录本身可能已经混入未来信息。这个原则也适用于池化、卷积、检索索引、预处理统计和归一化：只要中间状态读取了未来，后面再加一个因果 mask 也无法消除泄漏。

## 第九课　索引器与核心 attention：先选谁，再怎么算

压缩后仍可能有很多历史块，可以用一个较便宜的 indexer 为候选打分，选择 top-k，再由核心 attention 计算精确读取。本章 indexer 使用当前 query 与压缩键的点积，核心 attention 对选中摘要和局部原始记录一起做缩放 softmax。真实设计可以使用低维索引向量、不同头、量化或其他得分函数，但每一种都需要明确训练和成本。

选择错误意味着重要记录根本不会进入核心 attention。核心计算再精确也无法读取被排除的信息。因此评估应分成三项：indexer 是否保留有用候选，核心 attention 是否正确计算同一候选集合，整个模型的任务效果如何。只测最终答案，很难知道问题来自哪一层；只测索引召回，也不能保证模型会正确使用候选。

下面实现课程压缩规则的独立参考，返回每个查询的候选块、选中块和实际读取记录。选择使用稳定降序排序，局部记录始终保留。压缩 value 与压缩 key 必须使用同一块范围；如果排序只重排 K 而没有同步 V，输出会读错配对，shape 却可能完全正常。

```python
def compressed_reference(q,k,v,window=2,compression=2,top_k=2):
    outputs=[]
    traces=[]
    for i in range(len(q)):
        local_start,intervals=eligible_blocks(i,window,compression)
        records=[]
        for start,end in intervals:
            records.append({"start":start,"end":end,
                            "key":k[start:end].mean(axis=0),
                            "value":v[start:end].mean(axis=0)})
        if records:
            ranking=np.array([q[i]@record["key"] for record in records])
            selected=np.argsort(-ranking,kind="stable")[:top_k]
            chosen=[records[j] for j in selected]
        else:
            ranking=np.empty(0)
            chosen=[]
        keys=[record["key"] for record in chosen]
        values=[record["value"] for record in chosen]
        keys.extend(k[local_start:i+1])
        values.extend(v[local_start:i+1])
        keys=np.asarray(keys)
        values=np.asarray(values)
        row,_=attention(q[i:i+1],keys,values)
        outputs.append(row[0])
        traces.append({"eligible":intervals,
                       "chosen":[record["start"] for record in chosen],
                       "local":list(range(local_start,i+1)),
                       "records":len(keys)})
    return np.asarray(outputs),traces

from labs.models.advanced import compressed_sparse_attention
q_long=rng.normal(size=(9,3))
k_long=rng.normal(size=(9,3))
v_long=rng.normal(size=(9,2))
comp_ref,comp_trace=compressed_reference(q_long,k_long,v_long,2,2,2)
comp_lib,comp_chosen=compressed_sparse_attention(q_long,k_long,v_long,2,2,2)
np.testing.assert_allclose(comp_ref,comp_lib,atol=1e-12)
assert [trace["chosen"] for trace in comp_trace]==comp_chosen
```

这份参考每一步重新计算历史均值，效率很低，但状态语义清楚。实际增量实现可以在块完成时计算一次摘要并缓存。若每块是非重叠固定长度，维护运行和与计数即可；块完成后生成均值，再清空当前累积。注意尚未完成的块不能按本章规则提前发布给历史检索。

## 第十课　压缩、局部与选择的完整数值例子

设一维 K 为零、二、四、六、八、十，V 为一、三、五、七、九、十一，窗口二、压缩块二、top-k 一。考虑最后位置五，局部起点四，所以合法压缩块是零到二和二到四。第一个块均值键一、均值值二；第二个块均值键五、均值值六。局部原始记录为位置四和五，键八和十，值九和十一。

若 query 为零，两个压缩块的 indexer 分数都为零，稳定排序选择第一个块。核心 attention 读取三个记录：压缩块零的键一值二，以及局部键八值九、键十值十一。所有核心分数也为零，所以概率各三分之一，输出为二加九加十一除三，等于二十二除三。这个例子把选择、记录构造与归一化分开，所有数都可以手算。

若 query 为正一，indexer 选择第二个块，因为五大于一。核心分数为五、八、十，减去最大值十后指数为 e 的负五次方、e 的负二次方和一。分母为三者之和，输出为六乘第一项、九乘第二项、十一乘第三项再除分母。压缩选择影响候选，核心分数又决定候选之间的权重，二者不是同一个阶段。

```python
k_example=np.arange(0,12,2,dtype=float)[:,None]
v_example=np.arange(1,13,2,dtype=float)[:,None]
q_zero=np.zeros((6,1))
zero_out,zero_trace=compressed_reference(q_zero,k_example,v_example,2,2,1)
assert zero_trace[-1]["chosen"]==[0]
np.testing.assert_allclose(zero_out[-1],[22/3])
q_one=np.ones((6,1))
one_out,one_trace=compressed_reference(q_one,k_example,v_example,2,2,1)
assert one_trace[-1]["chosen"]==[2]
den=np.exp(-5)+np.exp(-2)+1
expected=(6*np.exp(-5)+9*np.exp(-2)+11)/den
np.testing.assert_allclose(one_out[-1],[expected])
```

现在把未来位置五的 K/V 改成极大值，位置零到四的输出应该保持不变。这个测试同时覆盖压缩候选资格与局部范围。若错误实现把块四到六在位置四时就加入，未来变化会通过块均值污染位置四，即使最后核心 attention 只读“过去摘要”也无济于事。

```python
changed_k=k_long.copy()
changed_v=v_long.copy()
changed_k[-1]+=1000
changed_v[-1]-=1000
changed_out,_=compressed_reference(q_long,changed_k,changed_v,2,2,2)
np.testing.assert_allclose(changed_out[:-1],comp_ref[:-1],atol=1e-12)
```

## 第十一课　选择的梯度、召回与近似误差

固定选择集合后，核心 attention 可以正常反向传播到选中的 Q/K/V 和压缩器参数。top-k 集合切换是离散边界，与 MoE 路由类似，普通自动微分不会提供对“换一个候选”这一动作的平滑导数。若 indexer 需要学习，可能采用额外监督、蒸馏、辅助目标或其他明确策略，不能只写 top-k 就假设所有候选得分都得到任务梯度。

均值压缩器的梯度容易推导。一个块大小 m，摘要为块内向量平均，上游摘要梯度会平均分给 m 个原始向量。若块未被选择，当前查询的核心 attention 不会通过它产生这条梯度；若多个查询选择同一摘要，梯度会累加。若摘要被缓存但在训练中 detach，梯度路径又会改变，必须说明训练分块与图保留方式。

可以用全 attention 的高权重位置作为分析参照，检查所选块是否覆盖这些位置。这是一个诊断 oracle，不是真实任务标签，也不表示全 attention 权重必然等于因果重要性。它帮助定位 indexer 的遗漏，但不能单独证明最终任务质量。报告应说明参照来自同一模型的哪一层、哪一头和什么输入。

选择质量与成本存在曲线。增大 top-k 通常增加候选覆盖与计算，但不保证所有任务指标严格单调，因为模型定义、归一化与训练分布也在变化。改变压缩率会同时改变摘要数量、信息损失与索引成本。合理实验一次改变一个因素，并记录原始记录数、摘要数、选中数和最终访问数，而不是只写一个“稀疏率”。

外部 RAG 的召回与内部稀疏 indexer 有结构类比：先筛候选，再精读。但外部检索返回文档并作为 token 输入，内部选择直接控制某层读取哪些隐藏状态。它们的训练信号、状态生命周期、权限边界和错误传播不同。用类比帮助理解可以，用同一个实现替代二者不可以。

## 第十二课　十二道练习与完整参考答案

### 练习一：画六乘六局部加全局 mask

窗口二、全局步长三，逐行允许集合为零；零、一；零、一、二；零、二、三；零、三、四；零、三、四、五。答案特别检查最后一行包含零和三两个全局键，以及四和五两个局部键。全局位置三只能被位置三及以后读取，不能泄漏给前两行。

```python
exercise_mask=local_global_mask(6,window=2,global_stride=3)
assert [np.flatnonzero(row).tolist() for row in exercise_mask]==[
    [0],[0,1],[0,1,2],[0,2,3],[0,3,4],[0,3,4,5]]
assert not exercise_mask[np.triu_indices(6,1)].any()
```

### 练习二：精确计算连接数

T 为十、W 为四，前四行连接数一、二、三、四，合计十；后六行各四，共二十四，总三十四。完整因果连接五十五。节省二十一条连接，比例二十一除五十五。不能用四十代替精确窗口数，也不能用一百作为因果基准而不说明自己比较的是未利用三角结构的执行。

```python
assert window_edges(10,4)==34
assert 10*11//2==55
np.testing.assert_allclose((55-34)/55,21/55)
```

### 练习三：证明两个独立 softmax 不能直接合并

局部分数零、值一，全局分数二、值五。并集输出为一加五乘 e 平方除一加 e 平方；分别归一化后两个输出是一与五，相加为六，平均为三，都不等于并集输出。完整参考如下。

```python
union=softmax(np.array([[0.,2.]]))@np.array([[1.],[5.]])
np.testing.assert_allclose(union,[[(1+5*np.exp(2))/(1+np.exp(2))]])
assert not np.allclose(union,[[6.]])
assert not np.allclose(union,[[3.]])
```

### 练习四：同步重排 K/V

给定合法索引二、零，使用同一索引取 K 与 V，结果应与索引零、二一致，因为 attention 对键值对的共同排列不敏感。若只重排 K 不重排 V，结果通常不同。这个测试直接检验配对，而不是只检验排序是否稳定。

```python
small_q=np.array([[1.]])
small_k=np.array([[0.],[2.],[4.]])
small_v=np.array([[1.],[3.],[9.]])
order_a=np.array([2,0])
order_b=np.array([0,2])
read_a=attention(small_q,small_k[order_a],small_v[order_a])[0]
read_b=attention(small_q,small_k[order_b],small_v[order_b])[0]
wrong_pair=attention(small_q,small_k[order_a],small_v[order_b])[0]
np.testing.assert_allclose(read_a,read_b)
assert not np.allclose(read_a,wrong_pair)
```

### 练习五：块覆盖的额外连接

使用第六课 block_cover，记录原窗口合法连接数、块覆盖元素数、与因果相交后的连接数。答案应说明三者不同：块覆盖包含可能的未来和窗口外元素，因果相交去掉未来，保留原细粒度 mask 才能严格恢复原窗口。不能把扩展后的模型与原窗口比较后，把差异误判成浮点错误。

```python
base=window_mask(8,2)
selected,cover=block_cover(base,2)
expanded=cover&np.tril(np.ones((8,8),dtype=bool))
assert int(base.sum())==15
assert int(cover.sum())>=int(expanded.sum())>=int(base.sum())
same_fine,_=dense_masked_reference(q_long[:8],k_long[:8],v_long[:8],base)
same_sparse,_=sparse_attention(q_long[:8],k_long[:8],v_long[:8],base)
np.testing.assert_allclose(same_fine,same_sparse,atol=1e-12)
```

### 练习六：均值压缩丢失配对信息

第七课的一正一负例子是完整答案。再解释为什么把两个值换成十和十时，压缩输出会恰好相同：无论概率怎样分配，相同 value 的加权和仍是十。这个退化案例不能证明均值压缩普遍等价，因此测试必须包含不同值与非均匀权重。

```python
same_values=np.array([[10.],[10.]])
original_same=attention(q_compress,k_compress,same_values)[0]
compressed_same=attention(q_compress,k_compress.mean(0,keepdims=True),
                         same_values.mean(0,keepdims=True))[0]
np.testing.assert_allclose(original_same,compressed_same)
```

### 练习七：构造并修复未来压缩泄漏

错误函数在每个位置读取包含该位置的完整两 token 块均值，因此偶数位置会读到下一个 token。下面只演示这个错误的数据依赖，随后用课程正确实现验证未来不干扰。答案应指出修复是限制压缩器输入范围或延迟摘要发布，而不是仅给摘要分数加一个下三角 mask。

```python
def leaking_block_means(values):
    rows=[]
    for i in range(len(values)):
        start=(i//2)*2
        rows.append(values[start:min(start+2,len(values))].mean(axis=0))
    return np.asarray(rows)

leak_values=np.arange(6,dtype=float)[:,None]
changed_values=leak_values.copy();changed_values[3]=100.
assert leaking_block_means(leak_values)[2,0]!=leaking_block_means(changed_values)[2,0]
safe_before,_=compressed_reference(q_one,k_example,v_example,2,2,1)
safe_values=v_example.copy();safe_values[3]+=100
safe_after,_=compressed_reference(q_one,k_example,safe_values,2,2,1)
np.testing.assert_allclose(safe_before[:3],safe_after[:3])
```

### 练习八：实现摘要的增量发布

非重叠块大小三，逐 token 累加 K/V，计数达到三才发布均值。完整答案如下。未完成块保留在 pending 状态，不出现在 published 中；这与是否已经满足某个查询的局部窗口资格仍是两步判断，发布完成不代表立即允许所有查询读取。

```python
class MeanBlockBuilder:
    def __init__(self,block_size,key_dim,value_dim):
        self.block_size=block_size
        self.key_sum=np.zeros(key_dim)
        self.value_sum=np.zeros(value_dim)
        self.count=0
        self.position=0
        self.published=[]
    def append(self,key,value):
        self.key_sum+=key
        self.value_sum+=value
        self.count+=1
        self.position+=1
        if self.count==self.block_size:
            self.published.append((self.position-self.block_size,self.position,
                self.key_sum/self.count,self.value_sum/self.count))
            self.key_sum=np.zeros_like(self.key_sum)
            self.value_sum=np.zeros_like(self.value_sum)
            self.count=0

builder=MeanBlockBuilder(3,1,1)
for i in range(5):
    builder.append(np.array([float(i)]),np.array([float(i+10)]))
assert len(builder.published)==1
assert builder.count==2
start,end,key_mean,value_mean=builder.published[0]
assert (start,end)==(0,3)
np.testing.assert_allclose(key_mean,[1.])
np.testing.assert_allclose(value_mean,[11.])
```

### 练习九：固定选择下的均值反向

块中三个 value 的均值被读取，上游梯度为向量二、负一。每个原 value 都收到三分之一的该梯度。若两个查询都读取此摘要，上游梯度先相加再平均分配。完整数值检查使用线性损失，中心差分应与手算一致。

```python
block_values=np.array([[1.,2.],[3.,4.],[5.,6.]])
upstream=np.array([2.,-1.])
expected_gradient=np.broadcast_to(upstream/3,block_values.shape)
numeric=np.zeros_like(block_values)
for index in np.ndindex(block_values.shape):
    old=block_values[index]
    block_values[index]=old+1e-6
    plus=block_values.mean(axis=0)@upstream
    block_values[index]=old-1e-6
    minus=block_values.mean(axis=0)@upstream
    block_values[index]=old
    numeric[index]=(plus-minus)/2e-6
np.testing.assert_allclose(numeric,expected_gradient,atol=1e-8)
```

### 练习十：区分三类误差

同 mask 的 gather 与 dense 不一致，是实现错误或数值误差；压缩模型与原始全 attention 不一致，可能是预期的模型差异；indexer 没选到高权重块，是选择质量问题。完整答案应为每类提供自己的基准：同定义高精度参考、原模型近似对照、候选覆盖诊断。不能用同一个阈值和同一句“误差很小”处理所有问题。

### 练习十一：设计因果回归测试

对长度一到十的随机 Q/K/V，逐个改变位置 t 的 K/V，检查零到 t 减一的输出不变。同时改变 query 只应影响该位置自己的读取，不影响其他行，因为本函数的 query 行彼此独立。这个性质测试覆盖窗口、全局与压缩边界，完整参考如下。

```python
for length in range(1,11):
    qq=rng.normal(size=(length,2))
    kk=rng.normal(size=(length,2))
    vv=rng.normal(size=(length,3))
    baseline,_=compressed_reference(qq,kk,vv,2,2,2)
    for t in range(length):
        altered_k=kk.copy();altered_v=vv.copy()
        altered_k[t]+=7.;altered_v[t]-=5.
        altered,_=compressed_reference(qq,altered_k,altered_v,2,2,2)
        np.testing.assert_allclose(altered[:t],baseline[:t],atol=1e-12)
```

### 练习十二：独立项目验收

实现局部、局部加全局、块覆盖、均值压缩加选择四种模式，分别输出允许集合或记录表。每种模式都有独立明确的参考，不把四种结果要求相同。报告连接数、记录数、临时状态、运行时间与误差，并解释每种模式改变了哪些信息通路。通过标准是你能从一个错误输出追溯到具体索引、压缩区间或归一化，而不是只会调 window 和 top-k。

## 第十三课　分块 softmax 的正确合并：把归一化统计量带回来

前面说局部与全局两个 attention 输出不能直接相加，并不意味着必须把所有分数同时放在内存里。可以分别计算每组的最大分数 m、稳定指数和 l，以及未归一化加权值和 a。对一组分数 s，定义 $m=\max s$、$l=\sum_j\exp(s_j-m)$、$a=\sum_j\exp(s_j-m)v_j$，最后输出为 a 除 l。

现在有两组，统计为 $(m_1,l_1,a_1)$ 与 $(m_2,l_2,a_2)$。取共同最大值 $m=\max(m_1,m_2)$。第一组指数需要乘 $\exp(m_1-m)$，第二组乘 $\exp(m_2-m)$，所以合并分母为这两个缩放后的 l 之和，合并分子为同样缩放后的 a 之和。这个推导只是把两组未归一化权重移到同一数值基准，因而与并集 softmax 精确等价。

手算两组各一项，分数零和值一，分数二和值五。第一组统计为零、一、一；第二组为二、一、五。共同最大值二，第一组缩放 e 的负二次方，第二组缩放一。合并分母为 e 的负二次方加一，分子为 e 的负二次方加五，输出等于前面并集答案。只保存归一化输出一和五却不保存分母与最大值，就失去了正确合并所需的信息。

```python
def softmax_summary(scores,values):
    maximum=float(np.max(scores))
    weights=np.exp(scores-maximum)
    denominator=float(weights.sum())
    numerator=weights@values
    return maximum,denominator,numerator

def merge_summaries(left,right):
    m1,l1,a1=left
    m2,l2,a2=right
    maximum=max(m1,m2)
    scale1=np.exp(m1-maximum)
    scale2=np.exp(m2-maximum)
    return maximum,scale1*l1+scale2*l2,scale1*a1+scale2*a2

summary_left=softmax_summary(np.array([0.]),np.array([[1.]]))
summary_right=softmax_summary(np.array([2.]),np.array([[5.]]))
merged=merge_summaries(summary_left,summary_right)
np.testing.assert_allclose(merged[2]/merged[1],union[0])

scores_many=rng.normal(size=11)
values_many=rng.normal(size=(11,3))
summary=None
for start in range(0,11,3):
    current=softmax_summary(scores_many[start:start+3],values_many[start:start+3])
    summary=current if summary is None else merge_summaries(summary,current)
np.testing.assert_allclose(summary[2]/summary[1],
                           softmax(scores_many)@values_many,atol=1e-12)
```

这段代码为后续在线 attention 算子提供基础，但并没有自动实现稀疏选择。你可以对合法记录分块，再用这个统计合并；非法记录不进入分组，或在块内明确屏蔽。若记录重复，合并仍会把它重复计入，因为统计量无法知道两个记录代表同一原 token。集合去重必须在更早阶段完成。

全遮挡组需要单独约定。它可以被跳过，或表示为分母零、分子零且不参与最大值更新；不能直接对空数组求最大值。实现在线归一化时，初始化和空块处理与主公式同样重要。对于窗口包含自身的正常查询，最终至少有一条合法记录，但中间某个块完全无效仍很常见。

## 第十四课　一个特殊的精确压缩情形及其局限

假设某个块的所有键完全相同，都是 k，而 value 可以不同。该块对 query 的每个分数都相同，为 s。原始 attention 中，这个块的分母贡献为 m 乘 e 的 s 次方，分子贡献为 e 的 s 次方乘所有 value 之和。若压缩 value 为均值，并把压缩分数设为 s 加 log m，则分母贡献变成 e 的 s 加 log m 次方，也就是 m 乘 e 的 s 次方；分子则是同样质量乘均值，恰好恢复原和。

这个结论说明记录数量本身携带概率质量。简单均值压缩若忽略块大小，会把一个大块与一个单 token 视为同样一条候选。是否需要补偿取决于模型定义：若目标是近似原 attention，块质量可能重要；若模型从头训练就把摘要当一种新记录，也可以学习另一套行为。不能把一种近似推导当成所有压缩模型必须遵循的规则。

下面用一个包含两条相同键的块和一个独立键验证。块内 value 为一和五，均值三；独立 value 为九。压缩后给块分数加 log 二，输出与原始三记录 attention 相同。随后改变块内一条键，等价一般失效，显示条件不是装饰性的前提。

```python
query=np.array([1.])
keys=np.array([[2.],[2.],[0.]])
vals=np.array([[1.],[5.],[9.]])
original=softmax(query@keys.T)@vals
compressed_scores=np.array([2.+np.log(2.),0.])
compressed_vals=np.array([[3.],[9.]])
exact_special=softmax(compressed_scores)@compressed_vals
np.testing.assert_allclose(original,exact_special)

keys_different=np.array([[1.],[3.],[0.]])
original_different=softmax(query@keys_different.T)@vals
assert not np.allclose(original_different,exact_special)
```

为什么不同键难以用单一均值精确表示？块的真实分母是多项指数和，作为 query 的函数通常不能写成一个固定均值键的单一指数乘常数；真实分子还把每条键的权重与其 value 配对。压缩同时损失了键分布与键值关联。学习型压缩器可以尝试保留任务相关统计，但它需要容量与训练，不能依靠线性平均的代数魔法解决所有查询。

这个例子还提醒你区分“摘要足以回答某一类问题”和“摘要对所有查询精确等价”。前者可能通过任务限制实现，例如只关心某个平均属性；后者要求更强的函数表示。模型研究经常接受任务层面的有效近似，课程实现则必须把精确条件与经验效果分开说明。

## 第十五课　渐进项目：建立稀疏模式的实验矩阵

第一阶段只做静态模式。实现窗口、固定全局键、膨胀间隔和块稀疏四种允许集合，输出每行索引与连接总数。膨胀模式可定义为读取当前位置及每隔 d 个位置的过去节点，再与局部窗口取并集。你应检查因果性、去重和边界，而不是先追求漂亮的可视化。矩阵图可以辅助，但索引列表是可直接测试的契约。

第二阶段为每个模式建立同 mask 的 dense 与 gather 对照。使用多种长度、头维度和值维度，尤其包含非方形查询键长度。测试数据既有随机值，也有可识别的确定值。随机测试覆盖广，确定值方便手算与定位。若只用全零输入，很多错误 mask 和错配 K/V 都会被掩盖。

第三阶段加入压缩器与 indexer。保留原始记录、压缩记录与局部记录的来源区间，生成 trace。对每个查询，trace 应能回答为什么某块合法、为什么被选、核心 attention 实际读了哪些记录。不要只返回最终输出，否则发生未来泄漏时很难知道是候选资格、压缩器范围还是局部窗口出了问题。

第四阶段做选择质量分析。可以定义一个诊断参照：在当前合法历史中，原始全 attention 权重最大的若干位置属于哪些块，所选块覆盖了多少。明确这只是分析指标，不是真实答案标签。再记录不同 top-k、压缩率和窗口大小下的访问数与输出差异，形成一张有条件的取舍表。

第五阶段研究性能。先固定数学定义，再替换执行方式。比较 Python gather、向量化密集 mask 和块式实现时，要记录是否构造完整分数矩阵、是否包含索引生成、是否重复计算摘要。CPU 小数据可能密集路径更快，GPU 大数据可能有不同结果。不要把一次设备上的排序解释成普遍架构结论。

### 十五点一　静态稀疏与动态选择的复杂度区别

窗口固定时，每个 query 的候选数大约 W，连接规模约 TW；固定数量全局键再增加约 TG。若全局键按固定步长不断增加，历史越长，全局数量也越多，总连接仍可能含平方项，只是常数变小。看到“局部加全局”不能直接声称线性复杂度，必须说明全局记录数量如何随 T 增长。

动态 top-k 选择也不自动消除扫描成本。若 indexer 每步仍对所有历史块打分，选择阶段随块数线性增长；核心 attention 只读 k 个块，并不意味着整个步骤只做 k 次工作。若使用层级索引、近似搜索或其他加速，索引构建、更新和查询成本也要计入。完整复杂度应把候选打分、选择、核心读取和状态维护分别列出。

压缩率 c 把历史块数从 T 降到约 T 除 c，但若每个 query 都扫描全部块，总打分仍约 T 平方除 c。它可能显著降低常数和缓存体积，却不是自动改变渐近阶。某些设计通过共享选择、跨层复用或更便宜的 indexer 进一步降低成本，这些是额外机制，需要各自分析。

### 十五点二　可复现报告的最低内容

报告应写出每种模式的集合定义、压缩区间、选择分数、top-k tie-break、核心归一化和空集合规则。附一个能手算的样例与一个随机回归测试，记录固定种子和 dtype。结果表分别列数学连接数、执行覆盖数、持久状态与临时工作区，避免把不同统计口径放在同一列比较。

若某模式改变了模型函数，应明确说输出差异是近似或结构变化，不宣称与全注意力完全一致。若某执行优化保持函数，应给出同定义参考误差。若没有实际运行 GPU，就不要填写推测吞吐；可以列出待测参数与预期瓶颈，但它们是实验计划，不是结果。

本章通过标准是你能精确回答“哪些历史信息有机会进入当前输出、它们以原始记录还是摘要存在、选择是否可能漏掉它们、归一化是否把重复和块大小处理正确”。这些问题会继续贯穿线性注意力、状态空间、检索增强与推理算子。掌握它们之后，你就能把新架构中的稀疏设计拆成可证明、可测量的组件。

### 十五点三　膨胀连接的完整答案与缓存保留问题

膨胀连接让当前位置读取距离为固定步长倍数的历史位置。下面定义距离不超过跨度 R，且距离能被步长 d 整除，再与局部窗口取并集。这里的相位跟随当前 query：位置五、步长二会读五、三、一；它不同于固定全局键只读位置零、二、四。两种模式都看起来像“隔一个读一个”，但连接集合并不相同。

```python
def dilated_local_mask(length,window,dilation,span):
    if min(length,window,dilation)<1 or span<0:
        raise ValueError("invalid sparse pattern")
    i=np.arange(length)[:,None]
    j=np.arange(length)[None,:]
    distance=i-j
    dilated=(distance>=0)&(distance<=span)&(distance%dilation==0)
    return window_mask(length,window)|dilated

dilated=dilated_local_mask(6,2,2,4)
np.testing.assert_array_equal(np.flatnonzero(dilated[5]),[1,3,4,5])
assert not dilated[np.triu_indices(6,1)].any()
```

缓存保留必须考虑未来查询可能访问哪些旧位置。纯固定窗口可以丢弃早于窗口边界的 K/V，因为任何未来 query 都不会再读它们。若加入固定全局键，这些全局位置可能需要长期保留；若加入动态检索，很多旧记录即使当前没被选中，也可能在未来被选中，不能因为本步没访问就立即删除。

压缩模型可以在一个块完成并且不再需要原始局部读取后，只保留摘要而释放原始 K/V，但这取决于未来选择是否只读取摘要。若系统还支持回到原始块做精读，原始数据必须保留在某个可访问层级，可能在显存、主存或其他存储中。减少当前 attention 访问量与减少全部状态存储不是同一件事。

一个完整状态账本应区分近期原始缓存、历史摘要、索引状态和可选的原始历史备份。某方案只把原始历史移到主存，显存减少但总内存未必减少；某方案彻底丢弃原始历史，存储更小但无法无损回看。报告时应说明这些差异，否则“压缩了多少倍”容易掩盖真正的能力变化。

### 十五点四　因果性不仅是输出测试，也是设计约束

未来不干扰测试很有用，但有限样例通过不能替代结构证明。证明时逐个检查中间状态：压缩器只读取已完成区间；indexer 只考虑合法摘要；局部支路只取不晚于当前的位置；位置编码不读取未来内容；归一化不跨未来 token 聚合。任何一个环节不满足，最后的 mask 都无法补救。

同时，因果性只是必要条件，不保证稀疏模式有足够能力。一个只看自身的模型完全因果，却无法利用历史；一个摘要只保留均值的模型也可能因果，却无法回答需要精确顺序的问题。测试应同时包含结构不变量与任务相关能力，分别说明两类证据。你应能在报告中清楚写出“实现没有未来泄漏”和“模型在某个任务上保留了足够信息”，而不是把它们合成一句含糊的正确。

索引记录最好携带来源区间而不只是一个数组下标。分页、排序或压缩率改变后，物理下标可能重新编号，但来源时间范围仍决定因果资格。调试日志同时记录逻辑位置与物理槽位，能够发现把旧页号误当新位置、把摘要编号误当原 token 位置等问题。这样的元数据不会改变注意力公式，却能让状态错误更早暴露。
