# M06　KV Cache、MHA、GQA、MQA：从因果性证明到完整缓存运行器

[本卷导航](README.md) · 前置：[稠密解码器](05-dense-decoder.md) · 下一章：[低秩与 MLA](07-latent-attention.md)

你已经能把一条 token 序列送进完整解码器。本章要解决一个具体问题：提示词有一千个 token，模型又生成一个 token，为什么不需要把前面一千个位置从头算一遍？答案不能停在“因为有缓存”。你需要证明哪些值不会变化，指出保存的位置与身份，写出完整前向、逐 token 和分块三条路径，并用同一权重验证它们的每一个位置。

本章所有核心代码可在 CPU 上执行，不需要 API，也不需要 GPU。参考实现使用 NumPy 双精度和直接数组，目的是把状态语义写清楚。高性能框架的分页、融合内核和调度可以改变物理布局，但必须遵守本章建立的因果依赖与请求隔离契约。阅读时准备纸笔，至少手画一次两层、三个 token 的依赖关系。

## 第一课　先区分输入 token、预测位置与生成步骤

语言模型接收前缀 $x_0,\ldots,x_{t-1}$，输出每个位置的词表分数。其中位置 $i$ 的分数用于预测 $x_{i+1}$。因此，把整个 prompt 送入模型后，最后一行 logits 已经能采样第一个新 token。采样本身没有执行一层神经网络；它只是根据这个分布选择一个整数 ID。要预测第二个新 token，才需要把刚选出的 ID 作为新输入送入模型。

用三个已知 token A、B、C 举例。完整前向产生三行 logits，分别预测 B、C、D。生成时通常只使用最后一行选择 D。接着执行一次 step(D)，得到预测 E 的 logits。这里必须避免两个常见错位：其一，在 prefill 之后把 C 再送一次，导致缓存里出现 A、B、C、C；其二，采样 D 后直接把预测 D 的旧 logits 再用来选择 E，根本没有让模型读到 D。

缓存长度表示已经进入模型的输入 token 数，而不是已经返回给用户的生成 token 数。prompt 长三，prefill 之后缓存长度三，尚未执行任何新 token 的 step。采样 D 并输出给用户之后，缓存仍是三；执行 step(D) 后才变成四。如果 D 恰好是终止符，服务可以直接结束，不一定有必要再把它写入缓存。日志和计费需要明确统计的是输入、输出还是实际计算位置，否则“生成了一百个 token，却只做了九十九次 decode”容易被误认为少算一步。

这一时间顺序还能解释首 token 延迟与后续 token 延迟的差异。首 token 通常要等待整个 prompt 的处理，而后续 token 只追加新位置，但要读取已有缓存。二者属于同一个模型的不同输入形状，不能把它们想成两个不同的模型。以后看到 prefill 和 decode 分开调度，你应先想起这个输入输出契约。

## 第二课　缓存正确性的因果证明

设第 $\ell$ 层位置 $i$ 的隐藏向量为 $h_i^\ell$，embedding 是 $h_i^0=E[x_i]$ 加上模型定义的位置处理。因果注意力只允许位置 $i$ 读取 $j\le i$ 的键和值。归一化与 FFN 在本模型中只沿特征轴计算，不跨 token 聚合。因此第 $\ell+1$ 层位置 $i$ 的结果只依赖第 $\ell$ 层位置零到 $i$。

证明可以做逐层归纳。第零层显然只依赖当前 token 和它的位置。假设第 $\ell$ 层的每个位置 $j$ 只依赖输入前缀零到 $j$，那么位置 $i$ 的 attention 读取 $j\le i$，每个被读取状态都只依赖零到 $i$ 之内的输入。其后的逐位置 FFN 和残差相加不会引入新的位置依赖，因此下一层也满足性质。对所有层重复，得到过去位置的隐藏状态不会因追加未来 token 而改变。

每层的键和值是这些状态的确定函数。例如 $K_i^\ell=\operatorname{RoPE}(N(h_i^\ell)W_K^\ell,i)$，$V_i^\ell=N(h_i^\ell)W_V^\ell$。如果权重、归一化规则、位置和计算模式都不变，追加未来 token 不会改变它们。于是我们可以保存已经算好的 K/V，下一次只计算新位置的 Q/K/V。这个论证说明缓存是对重复计算的消除，而不是对模型函数的近似。

证明也明确了适用条件。双向 encoder 中，旧位置可以读取新加入的 token，所以上述归纳不成立；更新模型权重后，过去投影不再对应当前函数；使用跨时间的归一化或改变位置缩放策略，也可能破坏旧状态的可复用性。训练时随机 dropout 的样本路径与自动微分图还有额外约束，不能把推理缓存直接当作普通全序列训练的替代。

为什么通常不缓存旧 Q？下一步要计算的是新位置的输出，只需要新查询与历史键比较，再读历史值。旧查询对应的旧输出已经算完，因果性又保证它不必重新计算。这里的“不需要”是针对标准自回归前向任务的结论，不是说任何算法里都不能保存 Q。训练反向、调试或特殊注意力机制可能保存不同的中间量，应按各自依赖关系判断。

## 第三课　用两个位置完整手算缓存

先暂时只看一个 attention 头，维度为一，缩放因子就是一。第一个位置的查询为一、键为二、值为三。只有一个合法键，所以分数是二，softmax 概率是一，输出为三。保存的缓存是 $K=[2]$、$V=[3]$。请注意，缓存里没有保存“输出三”作为未来的 value；value 三恰好和输出相同只是单键退化情况。

第二个位置的查询为一，新的键为零，新的值为五。追加后 $K=[2,0]$、$V=[3,5]$。新查询的分数为 $[2,0]$。减去最大值二，指数为 $[1,e^{-2}]$，分母 $1+e^{-2}$。两个概率约为 $0.880797$ 和 $0.119203$，因此新输出为 $0.880797\times3+0.119203\times5\approx3.238406$。

完整前向也会得到同样结果。其分数矩阵第一行对应第一个查询，第二行对应第二个查询。假定两个查询都为一，则未遮挡分数两行都是 $[2,0]$，因果 mask 把第一行第二列设成负无穷。因此第一行概率 $[1,0]$、输出三；第二行概率就是上面的两个数、输出约三点二三八四。缓存路径只是没有重新计算第一行。

如果错误地把第一个位置的最终 attention 输出作为缓存 V，在这个例子里暂时看不出问题，因为单键输出等于它自己的 V。加入第三个位置后，若你把第二个位置的输出三点二三八四当成第二个 value，就会用混合后的结果代替原值五，函数立即改变。因此测试不能只用一个 token，也不能只检查第一步。至少需要多个位置、非均匀分数和不同 value 才能暴露这种错误。

```python
import numpy as np
from labs.models.core import attention

q = np.array([[[1.0],[1.0]]])
k = np.array([[[2.0],[0.0]]])
v = np.array([[[3.0],[5.0]]])
full, weights = attention(q,k,v,np.tril(np.ones((2,2),dtype=bool)))
cached, cached_weights = attention(q[:,1:],k,v,np.ones((1,2),dtype=bool))
expected_second = (3.0*np.exp(2.0)+5.0)/(np.exp(2.0)+1.0)
np.testing.assert_allclose(full[0,:,0],[3.0,expected_second])
np.testing.assert_allclose(cached,full[:,1:])
np.testing.assert_allclose(cached_weights,weights[:,1:])
```

代码中最外层轴是头数，中间轴是位置，最后轴是每头特征。即便只有一个头和一个特征，也保留这两个轴，以便和后续多头实现共用同一规则。第二次调用只传最后一个查询，但 K/V 仍包含全部历史。mask 的形状是查询数乘键数，也就是一乘二，不能误用一乘一的下三角矩阵。

## 第四课　状态的 shape、身份与生命周期

在无 batch 的教学模型中，一层 K 的 shape 为 $[H_{kv},T,d]$，V 也是同样 shape。整个模型的状态是长度为 $L$ 的列表，每个元素为该层的 K/V 二元组。若加入 batch，常见布局之一是 $[B,H_{kv},T,d]$，也有框架采用不同轴顺序。布局本身没有唯一正确答案，但每个接口必须说明轴含义和步长。

逻辑状态至少携带四类身份。第一是请求身份：两个用户即使 prompt 长度相同，也不能共享未经验证的历史内容。第二是模型身份：相同结构、不同权重的 K/V 不可混用。第三是层身份：每层的投影不同，第一层的 K 不能交给第二层。第四是位置身份：数组中的第几个元素对应哪个逻辑位置，以及是否采用了滑动窗口或分页重排。

本章简单状态没有显式存这些元数据，而是由调用者和列表结构保证。实际服务应在请求对象中保存模型版本、当前位置、缓存句柄和结束状态。一个“形状匹配”检查无法防止跨请求污染，因为两个错误请求的缓存很可能拥有相同尺寸。隔离测试必须使用内容不同但长度相同的请求，并验证交错执行与分别执行一致。

生命周期通常包括创建、追加、读取、分支、回滚和释放。创建空状态意味着新请求；追加新 K/V 后有效长度增加；读取只访问有效区域；分支用于 beam search 或候选推测；回滚用于拒绝候选；释放把物理空间交还状态池。即使某个简单模型只实现创建和追加，理解其余操作仍然重要，因为服务层的很多错误来自生命周期，而不是 attention 公式。

若缓存来自一个已经结束的请求，是否能作为另一个请求的前缀复用？只有在输入 token 前缀、模型权重、位置规则和相关运行配置一致时才有数学依据。文本看起来一样并不够，tokenizer 或模板差异可能使 ID 序列不同。前缀复用还要处理访问控制与引用计数；这些属于服务系统，本章只建立复用必须满足的函数条件。

## 第五课　MHA、MQA 与 GQA 的共享轴

设隐藏维度 $D=H_qd$，查询头数为 $H_q$，每头维度为 $d$。MHA 让每个查询头有自己的键和值头，所以 $H_{kv}=H_q$。MQA 只有一个键值头，所以 $H_{kv}=1$。GQA 使用中间数量，要求本章的简单分组实现中 $H_q$ 能被 $H_{kv}$ 整除。每组查询头共享同一组 K/V，但查询向量仍然不同。

例如 $H_q=4,H_{kv}=2$，组大小 $g=2$。查询头零和一读 KV 头零，查询头二和三读 KV 头一。映射公式为 $r(h)=\lfloor h/g\rfloor$。这不是交替映射零、一、零、一。若采用另一种头排列，也可以定义其他映射，但加载权重时必须同步改变 Q/O 与 K/V 的排列。本章参考固定连续分组，因此用 repeat，而不是 tile。

为什么共享 K/V 以后仍有四个输出头？因为每个查询头都形成自己的分数 $q_hk_{r(h)}^\top/\sqrt d$，不同 q 导致不同概率，再对共享 V 做不同加权。输出 shape 仍是 $[H_q,T,d]$，合并后仍是 $[T,D]$，输出投影 O 也仍从 D 到 D。减少 KV 头改变了 K/V 投影与缓存容量，没有把整个多头输出宽度缩小。

取共享的两条键为零和一，值为二和六。查询为零的头得到均匀概率，输出四；查询为正一的头更偏向第二条键，输出 $(2+6e)/(1+e)\approx4.924234$；查询为负一的头更偏向第一条键，输出 $(2e+6)/(e+1)\approx3.075766$。三者读的是同一份 K/V，却有不同读取策略。这就是“共享记忆”与“相同输出”的区别。

```python
from labs.models.core import grouped_attention

shared_k=np.array([[[0.],[1.]]])
shared_v=np.array([[[2.],[6.]]])
different_q=np.array([[[0.]],[[1.]],[[-1.]]])
read,_=grouped_attention(different_q,shared_k,shared_v)
np.testing.assert_allclose(read[:,0,0],
    [4.,(2.+6.*np.e)/(1.+np.e),(2.*np.e+6.)/(1.+np.e)])
```

从参数量看，Q 和 O 各需要 $D^2$ 个权重，K 和 V 各需要 $D H_{kv}d$。MHA 的注意力投影总数为 $4D^2$；GQA 为 $2D^2+2DH_{kv}d$。这只计算四个无 bias 投影，不包含 FFN、归一化、embedding 或输出头。把 KV 头减半并不意味着整个模型参数减半，因为许多其他部分没有变化。

GQA 也不是任意 MHA 权重的无损开关。若两个原有 K/V 头不同，强行让它们共享一个头一般会改变函数。可以研究平均、选择或训练转换，但要重新评估质量。GQA 原论文提出从已有 MHA checkpoint 进行继续训练的转换方案，不能把这个事实缩写成“删几个头就等价”。本章先学会实现结构，再把转换实验作为有明确定义的近似研究。

## 第六课　不用复制 K/V 的分组参考实现

为了清楚，现有 grouped_attention 使用 np.repeat 把 KV 头展开成与查询头同数，再调用普通 attention。这在数学上便于验证，但如果真的把缓存永久展开，节省空间的目的就落空。一个更贴近物理共享的 CPU 参考可以逐查询头取相应 KV 头，不构造完整重复数组。它不追求速度，只证明共享读取的索引规则。

```python
def gqa_without_repeat(q,k,v,mask=None):
    q,k,v=map(lambda a:np.asarray(a,dtype=np.float64),(q,k,v))
    if q.ndim!=3 or k.ndim!=3 or v.ndim!=3:
        raise ValueError("expected [head,time,feature]")
    if k.shape[:2]!=v.shape[:2] or q.shape[-1]!=k.shape[-1]:
        raise ValueError("incompatible K/V or feature dimensions")
    if k.shape[0]==0 or q.shape[0]%k.shape[0]:
        raise ValueError("invalid head grouping")
    group=q.shape[0]//k.shape[0]
    outputs=[]
    all_weights=[]
    for h in range(q.shape[0]):
        kv_head=h//group
        out,prob=attention(q[h:h+1],k[kv_head:kv_head+1],
                           v[kv_head:kv_head+1],mask)
        outputs.append(out[0])
        all_weights.append(prob[0])
    return np.stack(outputs),np.stack(all_weights)

rng=np.random.default_rng(6)
q_test=rng.normal(size=(4,3,2))
k_test=rng.normal(size=(2,3,2))
v_test=rng.normal(size=(2,3,2))
mask_test=np.tril(np.ones((3,3),dtype=bool))
repeat_out,repeat_prob=grouped_attention(q_test,k_test,v_test,mask_test)
index_out,index_prob=gqa_without_repeat(q_test,k_test,v_test,mask_test)
np.testing.assert_allclose(index_out,repeat_out,atol=1e-12)
np.testing.assert_allclose(index_prob,repeat_prob,atol=1e-12)
```

逐行看，首先把输入转成双精度并验证三维布局，然后确认 K/V 的头数和时间轴一致、Q/K 的特征轴一致。V 的最后维度可以不同于 K，只要输出拼接和后续投影按它定义；本章完整解码器恰好取相同值维度。接着计算组大小，循环每个查询头，用整数除法定位 KV 头，保留长度一的头轴调用 attention。最后 stack 还原所有查询头的输出。

这一实现仍然会重复从内存读取共享 K/V，也会产生 Python 循环开销，所以不能据此断言 GQA 一定达到某个加速比。实际内核需要考虑线程组如何复用同一 KV 数据、缓存层级是否命中、矩阵块大小是否合适。结构减少了需要保存的独立数据量，硬件实现决定这些潜在节省有多少转化为实际性能。

## 第七课　完整模型的逐 token 等价测试

现在回到上一章的完整解码器。对一个固定 token 序列，先执行一次 full 得到所有位置 logits；再从空状态开始，逐 token 调用 step，每一步收集一行 logits。两者应在浮点容差内一致。测试要覆盖 MHA、GQA 和 MQA，因为只有一种头数时，错误映射可能碰巧退化为正确结果。

```python
from labs.models.core import TinyDecoder,DecoderConfig

tokens=[1,7,2,9,3,5]
for kv_heads in (1,2,4):
    decoder=TinyDecoder(DecoderConfig(kv_heads=kv_heads,seed=11))
    full_logits=decoder.forward(tokens)
    state=None
    rows=[]
    for position,token in enumerate(tokens):
        logits,state=decoder.step(token,state)
        rows.append(logits)
        assert len(state)==decoder.config.layers
        for keys,values in state:
            assert keys.shape==(kv_heads,position+1,4)
            assert values.shape==keys.shape
    cached_logits=np.stack(rows)
    np.testing.assert_allclose(cached_logits,full_logits,atol=1e-10,rtol=1e-10)
```

这里的头维度四来自默认隐藏维度十六除以四个查询头，不能把它误认为 KV 头数变化后也跟着改变。减少 KV 头数改变的是 K/V 总宽度，不改变每个头与查询配对的特征维度。状态长度在第 position 步后是 position 加一，因为当前 token 的 K/V 已追加进去。

比较 logits 而不是只比较 argmax，是因为不同分布可能拥有相同最大项。假设两个实现给出 $(10,0)$ 和 $(1,0)$，最大类别相同，但置信度完全不同；后续采样和训练行为都会不同。反过来，几乎并列的两项可能因微小舍入而交换最大值。结构验证应先看连续数值误差，再结合离散生成行为。

本章双精度小矩阵使用很小的容差。把相同阈值原样复制到半精度 GPU 内核并不合理；应该按数据类型、运算长度、归约顺序和应用敏感度制定阈值。也不应为了让测试通过而无条件放大容差。若误差很大，先定位第一层、第一步、第一处不同的中间张量，再判断它是否可能来自舍入。

## 第八课　分块 prefill：矩形 mask 的完整实现

完整 prefill 一次处理所有 prompt；逐 token step 一次只处理一个位置；分块 prefill 每次处理若干连续位置。三者可以统一成同一个接口：输入一段新 token 和旧 cache，返回新段所有位置的 logits 与追加后的 cache。块长度为一时就是 step，旧 cache 为空且块覆盖全部输入时就是 full。

假设已经有三个历史位置，新块长度为二。新查询的全局位置是三和四，键的位置是零到四。第一条新查询能读零、 一、二、三，不能读四；第二条能读全部。mask 是二乘五：第一行为真真真真假，第二行为全真。如果直接调用一个二乘五的普通 tril，第一行只允许第一列、第二行只允许前两列，会错误地屏蔽绝大部分历史。原因是查询行号从零重新开始，但逻辑位置没有重新开始。

正确规则是 $M_{ij}=[j\le P+i]$，其中 $P$ 是旧缓存长度，$i$ 是新块内部查询索引，$j$ 是完整键序列索引。将 P 设零就回到普通因果下三角。这个公式比记住“右下角对齐”更可靠，因为它直接使用位置语义。若键并非从零连续保存，例如滑动窗口或分页重排，则应使用显式逻辑位置数组，而不是把物理下标当成时间。

```python
from labs.models.core import rms_norm,swiglu

def chunk_forward(model,new_tokens,cache=None):
    raw=np.asarray(new_tokens)
    if raw.ndim!=1 or len(raw)==0 or not np.issubdtype(raw.dtype,np.integer):
        raise ValueError("expected nonempty integer token vector")
    if np.any(raw<0) or np.any(raw>=model.config.vocab):
        raise ValueError("token out of range")
    if cache is None:
        cache=[None]*model.config.layers
    if len(cache)!=model.config.layers:
        raise ValueError("wrong number of cache layers")
    lengths=[]
    for item in cache:
        if item is None:
            lengths.append(0)
            continue
        keys,values=item
        expected_heads=model.config.kv_heads
        expected_dim=model.config.dim//model.config.heads
        if keys.ndim!=3 or values.shape!=keys.shape:
            raise ValueError("bad layer cache shape")
        if keys.shape[0]!=expected_heads or keys.shape[2]!=expected_dim:
            raise ValueError("cache configuration mismatch")
        lengths.append(keys.shape[1])
    if len(set(lengths))!=1:
        raise ValueError("cache lengths disagree")
    positions=np.arange(lengths[0],lengths[0]+len(raw))
    x=model.embedding[raw]
    new_cache=[]
    for block,old in zip(model.blocks,cache):
        delta,state=model._attn(rms_norm(x),block,positions,old)
        x=x+delta
        x=x+swiglu(rms_norm(x),block["gate"],block["up"],block["down"])
        new_cache.append(state)
    return rms_norm(x)@model.output,new_cache

chunk_model=TinyDecoder(DecoderConfig(seed=17))
sequence=[1,3,5,7,9,2,4]
reference=chunk_model.forward(sequence)
for boundaries in ([7],[1,1,1,1,1,1,1],[2,3,2],[3,4]):
    state=None
    offset=0
    pieces=[]
    for length in boundaries:
        logits,state=chunk_forward(chunk_model,
            np.array(sequence[offset:offset+length],dtype=np.int64),state)
        pieces.append(logits)
        offset+=length
    assert offset==len(sequence)
    np.testing.assert_allclose(np.concatenate(pieces),reference,
                               atol=1e-10,rtol=1e-10)

old_length=3
new_length=2
rectangular=np.arange(old_length+new_length)[None,:] <= (
    old_length+np.arange(new_length))[:,None]
np.testing.assert_array_equal(rectangular,
    [[True,True,True,True,False],[True,True,True,True,True]])
```

这段代码复用了模型内部 attention 子函数，但没有调用完整 forward 或 step 来冒充自己的块式实现。它独立管理输入校验、旧状态长度、全局位置、层间残差和新状态列表。每一层必须读取该层旧缓存，而不是把第一层追加后的状态传给第二层；层间传递的是隐藏状态 x，层内跨时间传递的是该层 K/V。这两条数据流在代码里分别由 x 与 cache 表示。

输入校验有意拒绝浮点 token 向量。若先无条件转换成整数，像一点九这样的错误值会悄悄变成一，数据问题被掩盖。检查缓存 shape 也不等于验证身份，调用者仍需保证它属于同一模型和同一请求。分层责任应明确：函数能检查尺寸与长度，服务层负责请求句柄和模型版本。

分块方式不应改变结果，所以测试遍历完整一块、全部单 token、二三二分块和三四分块。随机选择边界可以进一步增强覆盖，但必须固定种子让失败可复现。一次只测“前两 token 加后三 token”仍可能遗漏长度为一、空旧状态和最后短块等边界。这里拒绝空新块；如果服务接口允许空块，必须另行定义返回值和状态是否保持不变。

## 第九课　RoPE 与缓存：旋转一次，位置不丢

在本模型里，K 先做线性投影，再按其逻辑位置做 RoPE，然后保存。以后新查询直接与这些已经旋转的旧 K 做点积。不能每一步再次对全部旧 K 旋转，否则位置编码被重复施加。以二维旋转为例，先旋转角度 $\theta$，再旋转同一个角度，等于旋转 $2\theta$，并不是保持原结果。

取原始键 $k=(1,0)$，位置角度 $\pi/2$。正确缓存为 $(0,1)$。如果下一步又按原位置旋转一次，变成 $(-1,0)$。新查询若为 $(0,1)$，正确点积是一，错误点积为零。这个反例不依赖大模型，直接显示重复旋转改变了注意力分数。

另一个错误是每块位置从零重新开始。假设旧前缀长度三，新 token 应位于位置三，却被当作位置零。旧 K 保留原来的位置，新 Q/K 的坐标系却突然重置，相对距离全部改变。只有把同一次 attention 中所有查询和键共同平移，固定频率 RoPE 的相对点积性质才保证不变；局部重置不满足条件。

```python
def rotate2(x,angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([c*x[0]-s*x[1],s*x[0]+c*x[1]])

original_key=np.array([1.,0.])
cached_key=rotate2(original_key,np.pi/2)
wrong_key=rotate2(cached_key,np.pi/2)
query=np.array([0.,1.])
np.testing.assert_allclose(query@cached_key,1.,atol=1e-12)
np.testing.assert_allclose(query@wrong_key,0.,atol=1e-12)
```

位置外推策略还带来更细的兼容性问题。如果某种策略根据当前总长度动态改变频率，已经缓存的旧 K 可能使用旧频率，新查询却使用新频率。是否需要重建、是否采用固定推理配置，取决于具体算法。不能仅凭名称含有 RoPE 就假设所有长度调整都与已有缓存兼容。阅读模型配置和官方实现时，应把位置函数视为缓存定义的一部分。

对于左 padding 的 batch，物理列号也未必等于有效 token 的逻辑位置。一个长度二的样本可能被补到四列，其有效 token 在列二和三，但逻辑位置希望从零和一开始。position_ids、attention mask 与 cache_position 在不同接口中可能承担不同职责，必须读清楚各自语义。教学代码没有 batch padding，因此位置等于有效长度；扩展时不能默默沿用这个简化。

## 第十课　显存账本：容量、带宽与分配浪费分开算

传统逐层 K/V 缓存的元素数量为
$$
N_{kv}=2BLTH_{kv}d.
$$
第一个二表示 K 和 V 两份数组；B 是同时保存状态的序列数；L 是层数；T 是每条序列的缓存长度，若不同长则改成对各请求长度求和；$H_{kv}$ 是 KV 头数；d 是每头特征数。再乘每元素字节数 s，得到字节容量 $M=N_{kv}s$。这个公式没有包含任何权重或临时工作区。

完整手算一个例子：B 等于二，L 等于十二，T 等于一千零二十四，KV 头数四，头维度六十四，BF16 每元素两字节。每个头每层单个 K 有 $1024\times64=65536$ 个元素；四头为 $262144$；加上 V 为 $524288$；十二层为 $6291456$；两个请求为 $12582912$；乘两字节为 $25165824$ 字节。除以 $1024^2=1048576$，得到二十四 MiB。

如果查询头数是十六而 KV 头数从四变十六，其他因素不变，缓存变成九十六 MiB。若改成单 KV 头，变成六 MiB。这是缓存这一个分量的变化，不是整个进程显存的变化。权重、激活、分配器保留、图执行缓冲、通信区和采样工作区都要另算。显存占用界面上的数字与公式不同，并不自动表示公式错了，而可能表示统计范围不同。

```python
def kv_bytes(batch,layers,length,kv_heads,head_dim,item_bytes):
    values=(batch,layers,length,kv_heads,head_dim,item_bytes)
    if any(not isinstance(v,int) or v<0 for v in values):
        raise ValueError("expected nonnegative integers")
    return 2*batch*layers*length*kv_heads*head_dim*item_bytes

assert kv_bytes(2,12,1024,4,64,2)==25165824
assert kv_bytes(2,12,1024,4,64,2)/(1024**2)==24
assert kv_bytes(2,12,1024,16,64,2)/(1024**2)==96
assert kv_bytes(2,12,1024,1,64,2)/(1024**2)==6
```

容量和带宽不是同一个问题。即使缓存放得下，每个解码步骤仍要读取历史 K/V，标准注意力单个新 token 的计算随历史长度线性增长。忽略缓存命中与融合细节，读取的数据量大致与当前缓存有效体积成正比。长上下文 decode 可能受内存带宽限制，而 prefill 的较大矩阵乘法更容易利用计算单元。这是一种性能趋势，不是任何尺寸和硬件下都成立的绝对定律，必须以测量确认瓶颈。

再算分页浪费。假设每页十六个 token，三个请求长度分别十七、三十二、三十三。需要页数分别二、二、三，总共七页，可容纳一百一十二个 token；实际有效八十二个，空闲尾部三十个。若每 token 的全部层 K/V 占 c 字节，理论有效容量八十二 c，已分配容量一百一十二 c。页表、对齐和元数据还可能增加额外成本。分页减少大块连续分配与搬迁的问题，却不会让尾部碎片完全消失。

```python
def page_budget(lengths,page_size,bytes_per_token):
    pages=sum((n+page_size-1)//page_size for n in lengths)
    used=sum(lengths)
    capacity=pages*page_size
    return {"pages":pages,"used_tokens":used,"capacity_tokens":capacity,
            "tail_tokens":capacity-used,"allocated_bytes":capacity*bytes_per_token}
budget=page_budget([17,32,33],16,100)
assert budget=={"pages":7,"used_tokens":82,"capacity_tokens":112,
               "tail_tokens":30,"allocated_bytes":11200}
```

对 MLA、线性注意力、状态空间和混合模型，不应直接套传统公式。它们保存的对象可能是低维 latent、额外位置键、固定矩阵状态或不同层的不同状态。正确做法是先列每个状态张量的 shape，再求元素数与类型字节数。知道一个公式有适用边界，比记住一个看似通用的数字更有价值。

## 第十一课　为什么 concatenate 正确却不适合长序列

教学代码每一步把旧 K 与新 K concatenate，得到一个新的连续数组。若第 t 步缓存长度为 t，复制量与 t 成正比，整个生成过程的总复制量约为 $1+2+\cdots+T=T(T+1)/2$。因此即使 attention 的数学计算已经避免重算历史投影，数组管理本身仍可能引入平方级搬运。这是数据结构问题，不是缓存概念有误。

一种替代是预分配最大容量，使用有效长度游标。每次把新 K/V 写入尚未使用的位置，再把有效前缀视图交给 attention。追加本身只写新数据，旧数据不搬家。代价是提前占用容量，并且需要处理超出容量、请求分支、并发写入和释放。下面实现一个单层状态容器，明确不允许超量追加，也不把未初始化区域暴露给读取者。

```python
class LayerKVBuffer:
    def __init__(self,heads,capacity,dim):
        if min(heads,capacity,dim)<=0:
            raise ValueError("positive dimensions required")
        self.keys=np.empty((heads,capacity,dim),dtype=np.float64)
        self.values=np.empty_like(self.keys)
        self.length=0
    def append(self,new_k,new_v):
        new_k=np.asarray(new_k,dtype=np.float64)
        new_v=np.asarray(new_v,dtype=np.float64)
        if (new_k.ndim!=3 or new_v.shape!=new_k.shape
            or new_k.shape[0]!=self.keys.shape[0]
            or new_k.shape[2]!=self.keys.shape[2]):
            raise ValueError("shape mismatch")
        end=self.length+new_k.shape[1]
        if end>self.keys.shape[1]:
            raise ValueError("cache capacity exceeded")
        self.keys[:,self.length:end]=new_k
        self.values[:,self.length:end]=new_v
        self.length=end
    def view(self):
        return self.keys[:,:self.length],self.values[:,:self.length]
    def truncate(self,new_length):
        if not 0<=new_length<=self.length:
            raise ValueError("invalid rollback")
        self.length=new_length

buffer=LayerKVBuffer(2,8,3)
first_k=np.arange(12,dtype=float).reshape(2,2,3)
first_v=-first_k
buffer.append(first_k,first_v)
buffer.append(first_k[:,:1]+100,first_v[:,:1]-100)
view_k,view_v=buffer.view()
assert view_k.shape==(2,3,3)
np.testing.assert_array_equal(view_k[:,:2],first_k)
buffer.truncate(2)
assert buffer.view()[0].shape==(2,2,3)
```

truncate 只缩小有效长度，不清零后面的内存。只要读取严格限制在有效前缀内，旧值不会参与计算；若缓存会跨安全边界复用，服务层还需要相应的隔离与数据处理策略。数学正确性依赖“不可读到无效区”，不是依赖无效区恰好全零。调试时把未使用区填成 NaN，有时更容易暴露越界读取，因为零值可能掩盖错误。

view 返回的是共享视图。调用者如果修改它，会修改内部缓存，因此这个教学容器不具备只读保护。分支搜索若让两个分支共享同一可变缓冲区并分别追加，就可能互相覆盖。可以复制有效状态，或使用只读共享页加写时复制；选择取决于性能目标。重要的是显式定义所有权，不要仅凭“Python 变量不同”就假定内存独立。

预分配、分页和压缩是不同维度的设计。预分配解决反复重分配；分页解决动态容量与共享管理；GQA 减少独立 KV 头；量化减少每元素字节；滑动窗口减少可读历史范围。它们可以组合，但每一种都需要自己的正确性论证。尤其滑动窗口改变注意力可见性，不再与全历史模型严格等价，而预分配本身通常只改变物理存储。

## 第十二课　请求交错、分支与恢复：状态属于调用者

先做一个最小的交错实验。请求 A 的 token 为一、二、三、四，请求 B 为九、八、七。模型权重可以共享，因为推理时不更新；两条请求的 cache 必须独立。调用顺序可以是 A 一步、B 一步、A 一步、B 一步，只要每次取回对应状态，结果就应与各自独立运行一致。这个实验直接检验状态是否被错误地藏在模型对象的公共成员里。

```python
shared_model=TinyDecoder(DecoderConfig(seed=23))
requests={"A":[1,2,3,4],"B":[9,8,7]}
states={"A":None,"B":None}
outputs={"A":[],"B":[]}
for tick in range(4):
    for name in ("A","B"):
        if tick<len(requests[name]):
            row,states[name]=shared_model.step(requests[name][tick],states[name])
            outputs[name].append(row)
for name in requests:
    np.testing.assert_allclose(np.stack(outputs[name]),
        shared_model.forward(requests[name]),atol=1e-10,rtol=1e-10)
```

为什么只创建两个 state 变量就足够？因为当前 TinyDecoder.step 不在内部保存请求历史，它接收显式 cache，并通过 concatenate 创建新数组。权重对象只读，局部隐藏状态每次重新计算。这个设计不等于线程安全的完整保证，但把最重要的请求状态从共享对象中剥离出来，便于测试和管理。若以后改成原地预分配，必须重新检查状态所有权，不能假设原来的分支行为自动保留。

再考虑共同前缀后分叉。输入一、二之后，一个候选追加三，另一个追加四。它们可以共享前缀的数学值，但后续状态必须各自增长。教学 concatenate 版本不修改旧数组，因此从同一旧状态调用两次 step 是安全的。预分配版本若直接在同一有效长度后写入，会让后调用覆盖前调用，必须复制、分配独立尾页或使用写时复制。

```python
prefix_logits,prefix_state=chunk_forward(shared_model,np.array([1,2]))
branch3,state3=shared_model.step(3,prefix_state)
branch4,state4=shared_model.step(4,prefix_state)
assert all(k.shape[1]==2 for k,v in prefix_state)
assert all(k.shape[1]==3 for k,v in state3)
assert all(k.shape[1]==3 for k,v in state4)
np.testing.assert_allclose(branch3,shared_model.forward([1,2,3])[-1],atol=1e-10)
np.testing.assert_allclose(branch4,shared_model.forward([1,2,4])[-1],atol=1e-10)
```

恢复状态时，不能只保存 K/V 而忘记生成历史。K/V 能继续计算模型，但重复惩罚、停止字符串检测、输出文本解码、采样随机数状态和审计日志可能还需要 token 序列或其他状态。如果目标是精确复现同一次随机采样，还要保存随机数生成器状态。模型状态与生成策略状态是两个相关但不同的对象，应在接口里分别说明。

回滚也有两种含义。一种是退回较短前缀，丢弃后面 token 的缓存，这在严格因果模型中数学上成立，因为较早位置没有依赖被删掉的未来。另一种是修改中间某个 token，却只删最后一个缓存位置，这通常错误：从被修改位置开始的所有后续层状态都可能受影响，必须从那个位置重新计算。若系统支持编辑历史，需要识别最长不变前缀，而不是简单复用旧长度。

## 第十三课　完整生成循环：先 prefill，再消费采样结果

下面写一个可运行的贪心生成器。输入 prompt 非空，先通过 chunk_forward 得到 prompt 的最后 logits 与缓存；每轮选择最大分数 token，加入输出；若达到终止符或最大长度则结束；否则把刚选出的 token 送进 step，更新下一轮 logits。为避免“随机模型生成自然语言”的错觉，本例使用小词表 ID，只验证状态与无缓存基准一致。

```python
def greedy_cached(model,prompt,max_new_tokens,eos_id=None):
    if max_new_tokens<0:
        raise ValueError("negative generation length")
    prompt=list(prompt)
    logits,state=chunk_forward(model,np.asarray(prompt,dtype=np.int64))
    next_logits=logits[-1]
    generated=[]
    for index in range(max_new_tokens):
        token=int(np.argmax(next_logits))
        generated.append(token)
        if token==eos_id or index+1==max_new_tokens:
            break
        next_logits,state=model.step(token,state)
    return generated

def greedy_uncached(model,prompt,max_new_tokens,eos_id=None):
    history=list(prompt)
    generated=[]
    for _ in range(max_new_tokens):
        token=int(np.argmax(model.forward(history)[-1]))
        generated.append(token)
        history.append(token)
        if token==eos_id:
            break
    return generated

for prompt in ([1],[1,2],[3,4,5]):
    assert greedy_cached(shared_model,prompt,8)==greedy_uncached(shared_model,prompt,8)
assert greedy_cached(shared_model,[1,2],0)==[]
```

这里最后一个生成 token 没有再送入 step，因为循环已经结束，不需要预测后继。这是一个可选择的接口约定。如果函数返回的 cache 将被后续“继续生成”使用，就必须说明缓存是否包含全部返回 token。上面的函数只返回生成列表，因此不暴露不完整状态；若扩展为返回状态，建议额外返回 pending_token 或在结束前补做最后一步，使恢复方知道当前状态的位置。

无缓存基准每次把完整 history 送进模型，是慢但清楚的正确性参考。验证缓存生成时，先固定贪心策略避免随机数消耗顺序影响比较；再加入固定 RNG 的采样，确保两条路径对每一步使用相同随机数。若两条路径采用不同的随机数调用次数，即使概率相同也可能生成不同文本，不能据此直接判断缓存错误。

停止条件也会影响实验。按 EOS ID 停止是 token 级规则；按某段字符串停止可能跨越多个 token，必须在解码后的尾部检查；遇到最大长度停止则不是模型主动结束。报告生成结果时应记录停止原因。模型结构课程的实验可以只用 EOS 与长度上限，但完整系统不能把所有结束都混为一谈。

## 第十四课　故障诊断：从第一个错误位置倒推原因

如果 full 与 step 在第一个 token 就不同，优先检查是否重复输入了 prompt 最后 token、是否在空缓存时把位置设成一、是否遗漏 final norm、是否用了不同权重或不同随机模式。第一个 token 没有复杂历史读取，很多长上下文因素尚未出现，因此它是最简单的定位点。如果第一步一致、第二步开始不同，常见原因是缓存追加、位置递增、mask 对齐或把错误层的状态复用。

如果 MHA 通过而 GQA 失败，重点检查头映射和 K/V reshape。MHA 中每个查询头对应同编号 KV 头，repeat 因子为一，错误实现可能恰好不暴露；GQA 才要求多个查询头读取同一 KV 头。若 MQA 通过、GQA 失败，也不能说明分组正确，因为只有一个 KV 头时，很多错误映射都会指向零。至少测试一个中间分组配置。

如果短序列通过而长序列失败，检查位置计数是否溢出或重置、预分配容量是否越界、页边界转换是否正确、低精度误差是否累积，以及 sliding window 的逻辑位置是否仍然保留。不要先把问题归因于“模型不擅长长文本”，因为结构等价测试与模型能力无关：同一组随机权重也应满足 full 与 cache 的数学契约。

如果只有分块 prefill 失败，优先检查矩形 mask。特别是旧长度大于零、新块长度大于一时，错误的方阵下三角最容易显现。还要检查每层是否使用同一个旧长度、块内新 K/V 是否全部追加后再做带因果 mask 的读取，以及是否误把新块最后一个位置当作所有查询的位置。一次只测试长度一无法覆盖块内未来屏蔽。

数值比较应同时记录最大绝对误差、均方误差和误差最大位置。相对误差在参考值接近零时可能很大，应与绝对误差联合使用。若错误从第一层 K 就出现，继续检查后面十层没有意义；先修复最早分叉。若 K/V 一致、注意力概率不同，检查 mask、缩放与 softmax；若概率一致而输出不同，检查 V、合并头与 O 投影。

```python
def error_report(actual,expected):
    actual=np.asarray(actual,dtype=np.float64)
    expected=np.asarray(expected,dtype=np.float64)
    if actual.shape!=expected.shape:
        return {"shape_match":False,"actual":actual.shape,"expected":expected.shape}
    diff=np.abs(actual-expected)
    location=np.unravel_index(np.argmax(diff),diff.shape)
    return {"shape_match":True,"max_abs":float(diff.max()),
            "rmse":float(np.sqrt(np.mean((actual-expected)**2))),
            "location":tuple(int(i) for i in location)}

report=error_report(np.array([[1.,2.],[3.,4.1]]),
                    np.array([[1.,2.],[3.,4.]]))
assert report["location"]==(1,1)
np.testing.assert_allclose(report["max_abs"],0.1)
```

## 第十五课　分级练习、复杂度证明与缓存实验台

### 练习一：把预测顺序写成状态表

prompt 为三个 token，计划生成两个新 token。写出 prefill、第一次采样、第一次 step、第二次采样之后的缓存长度，以及每份 logits 预测哪个位置。参考答案：prefill 后长度三，最后 logits 预测第四个 token；第一次采样不改变缓存，长度仍三；把第四个 token 执行 step 后长度四，返回 logits 预测第五个 token；第二次采样后长度仍四。如果此时结束，缓存并不包含第五个 token。这个答案的重点是把“选择一个 token”和“把它输入模型”区分开。

进一步追问：若 API 承诺返回可无缝继续的完整状态，应怎样处理？一种完整方案是在结束前把第五个 token 也送入 step，得到长度五的 cache 与预测第六个 token 的 logits；另一种方案保存长度四的 cache 加 pending_token 和对应协议。两种都可以，但不能只返回 cache 而不说明它落后于输出历史。

### 练习二：手算连续分组映射

查询头数十二，KV 头数三。组大小为四，所以查询头零到三读取 KV 零，四到七读取 KV 一，八到十一读取 KV 二。下面的完整程序生成映射并用可辨认的 value 验证。所有查询和键都为零，因此每个头只需读自己组的唯一 value；输出直接暴露映射，不依赖随机数。

```python
mapping=[h//4 for h in range(12)]
assert mapping==[0,0,0,0,1,1,1,1,2,2,2,2]
q_map=np.zeros((12,1,1))
k_map=np.zeros((3,1,1))
v_map=np.array([[[10.]],[[20.]],[[30.]]])
out_map,_=gqa_without_repeat(q_map,k_map,v_map)
np.testing.assert_array_equal(out_map[:,0,0],
    [10,10,10,10,20,20,20,20,30,30,30,30])
```

若错误地用 h 对三取余数，结果会交替十、二十、三十。shape 完全正确，输出也都有限，但语义错误。这个例子说明测试数据应有区分能力，不能让所有 KV 头都填同一个常数，否则任何映射都看起来正确。

### 练习三：推导并验证矩形 mask

旧长度四，新块三。第一行可见键零到四，第二行可见零到五，第三行可见零到六。完整答案如下。测试还验证每一行合法键数依次为五、六、七，因为当前 token 可以看到自身；如果实现的是不含自身的其他注意力定义，应另写规则，不能沿用本章答案。

```python
past,new=4,3
mask=np.arange(past+new)[None,:] <= (past+np.arange(new))[:,None]
assert mask.shape==(3,7)
np.testing.assert_array_equal(mask.sum(axis=1),[5,6,7])
assert not mask[0,5] and not mask[0,6]
assert mask[2].all()
```

### 练习四：证明缓存不能跨层交换

构造两层模型，先缓存两个 token，再交换两层状态后执行第三个 token。参考结果通常与正确路径不同。原因不是两层 shape 不匹配，而是每层的 K/V 来源于不同隐藏状态和不同投影。下面保留原状态并分别执行，避免错误实验污染正确基准。

```python
_,valid_state=chunk_forward(shared_model,np.array([1,2]))
correct_row,_=shared_model.step(3,valid_state)
swapped=[valid_state[1],valid_state[0]]
wrong_row,_=shared_model.step(3,swapped)
assert np.max(np.abs(correct_row-wrong_row))>1e-8
```

如果随机权重恰好出现退化相等，不能把它当作层交换恒等式；应换用确定的非退化参数或多个种子。数学上只有在两层状态恰好相同的特殊情形才可能不变，正常模型没有这样的保证。测试证明的是实现满足一般契约，不能靠偶然样例解释结构。

### 练习五：逐项算缓存与分页预算

两条请求长度分别一千和一千五百，层数二十四，KV 头数八，头维度六十四，每元素两字节。每 token 跨所有层的 K/V 为 $2\times24\times8\times64\times2=49152$ 字节。总有效 token 数二千五百，因此有效容量 $122880000$ 字节，除以 $1048576$ 约一百一十七点一八七五 MiB。若页大小十六，请求分别需要六十三页和九十四页，总一百五十七页，容量二千五百一十二 token，多十二 token 的尾部空间。

```python
per_token=2*24*8*64*2
assert per_token==49152
assert per_token*(1000+1500)==122880000
paged=page_budget([1000,1500],16,per_token)
assert paged["pages"]==157
assert paged["tail_tokens"]==12
np.testing.assert_allclose(122880000/(1024**2),117.1875)
```

这里没有把两个请求强行补到同样最大长度。若使用静态密集 batch 张量，每条都按一千五百分配，则容量对应三千 token，浪费五百 token。不同布局的预算不同，比较时必须保持统计范围一致。不能用动态分页的有效 token 数去解释静态张量的实际分配。

### 练习六：解释并验证回滚

先处理一、二、三、四，再把每层 K/V 截到前两个位置，接着输入九。结果应该等于完整前向一、二、九的最后一行。理由是前两个位置的状态不依赖后来被删除的三、四。下面通过切片复制，避免后续原地操作影响原缓存；在当前 concatenate 实现中复制不是数学必需，但使所有权更清晰。

```python
_,long_state=chunk_forward(shared_model,np.array([1,2,3,4]))
short_state=[(k[:,:2].copy(),v[:,:2].copy()) for k,v in long_state]
rolled_row,_=shared_model.step(9,short_state)
np.testing.assert_allclose(rolled_row,
    shared_model.forward([1,2,9])[-1],atol=1e-10,rtol=1e-10)
```

若修改的是第二个 token，第一位置仍可复用，但第二及之后必须重算。缓存截断的位置应是最长不变前缀的长度，而不是原序列长度减一。这个规则在编辑对话历史、候选验证和恢复请求时都适用。

### 练习七：确认请求污染会怎样出现

用 A 的前缀一、二建立 cache，却把 B 的下一 token 九接到这个 cache 上。得到的分布对应一、二、九，不是新请求九。完整答案应同时给出错误结果的真实语义，而不只是说“输出不同”。下面验证这两个关系。

```python
_,a_state=chunk_forward(shared_model,np.array([1,2]))
contaminated,_=shared_model.step(9,a_state)
fresh,_=shared_model.step(9,None)
np.testing.assert_allclose(contaminated,shared_model.forward([1,2,9])[-1],atol=1e-10)
np.testing.assert_allclose(fresh,shared_model.forward([9])[-1],atol=1e-10)
assert np.max(np.abs(contaminated-fresh))>1e-8
```

这类错误可能不会抛异常，也可能生成语法正常的文本，所以不能靠人工读输出发现。状态句柄应与请求身份绑定，并在测试中覆盖交错执行、取消后复用和不同长度请求。缓存安全与模型正确性在这里直接相连。

### 练习八：模拟低精度缓存并报告误差

将 K/V 对称量化到有符号八位整数，再立即反量化成双精度供现有 attention 使用。它只模拟数值近似，不节省这个 Python 运行器的实际内存，也不模拟高性能量化内核。每个张量使用单一 scale，零张量单独处理。完整答案必须说明它不再严格等价，并记录每步 logits 误差。

```python
def fake_int8(array):
    array=np.asarray(array,dtype=np.float64)
    peak=float(np.max(np.abs(array)))
    if peak==0:
        return np.zeros_like(array)
    scale=peak/127.
    quantized=np.clip(np.rint(array/scale),-127,127).astype(np.int8)
    return quantized.astype(np.float64)*scale

quantized_state=None
errors=[]
seq=[1,2,3,4,5,6,7,8]
for index,token in enumerate(seq):
    row,quantized_state=shared_model.step(token,quantized_state)
    reference_row=shared_model.forward(seq[:index+1])[-1]
    errors.append(float(np.max(np.abs(row-reference_row))))
    quantized_state=[(fake_int8(k),fake_int8(v)) for k,v in quantized_state]
assert errors[0]<1e-10
assert max(errors[1:])>0
assert np.isfinite(errors).all()
```

第一步还没有读到经过量化的历史，所以误差接近零。后面误差不必随长度严格单调，因为注意力权重、抵消与输出投影都会影响传播。应报告序列而不是只写“越长误差越大”。若希望测量一次量化后永久保存的真实格式，还需要显式保存整数与 scale，避免每步把全部历史重新量化；那是另一种误差和存储实验。

### 练习九：计算反复拼接的搬运量

每步新增一个标量，共 T 步，若每次复制整个新数组，总写入元素数为 $T(T+1)/2$。当 T 为一千，写入五十万零五百个元素，最终只有一千个有效元素；当 T 翻倍，总量约四倍。若每个 token 包含 c 个 K/V 元素，再乘 c。下面用实际长度计数验证，不必真的分配巨大数组。

```python
for length in (1,2,10,1000):
    counted=sum(range(1,length+1))
    assert counted==length*(length+1)//2
assert sum(range(1,1001))==500500
```

预分配的追加写入总量是 T 个新位置，注意力读取历史的成本仍然存在。不能把减少拼接复制说成把整个 decode 变成常数时间；你只优化了状态追加这一部分。性能分析应把投影、attention 读取、FFN、输出头和状态管理分开计时。

### 练习十：写完整分块一致性验收

前文 chunk_forward 已给出完整实现。现在要求对长度一到十二、三种 KV 头数、多个不同块边界验证。参考答案使用确定随机种子生成 token，并随机选择每块长度一到四。所有块的输出拼接后与 full 比较，每层状态的最终长度应等于输入长度。这是一个小而有意义的性质测试，覆盖不同路径组合。

```python
rng_verify=np.random.default_rng(99)
for hk in (1,2,4):
    verifier=TinyDecoder(DecoderConfig(kv_heads=hk,seed=31))
    for length in range(1,13):
        ids=rng_verify.integers(0,verifier.config.vocab,size=length)
        pieces=[]
        state=None
        start=0
        while start<length:
            count=min(int(rng_verify.integers(1,5)),length-start)
            rows,state=chunk_forward(verifier,ids[start:start+count],state)
            pieces.append(rows)
            start+=count
        np.testing.assert_allclose(np.concatenate(pieces),verifier.forward(ids),
                                   atol=1e-10,rtol=1e-10)
        assert all(k.shape[1]==length for k,v in state)
```

### 练习十一：给缓存接口写拒绝条件

至少拒绝错误层数、层间长度不一致、错误 KV 头数、错误头维度、负 token 和超词表 token。参考答案使用统一的 expect_value_error 辅助器，确保错误确实发生，而不是只把调用写在文档里。注意本章 chunk_forward 能检查的仍是结构错误，不能识别同 shape 的错误请求身份。

```python
def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

expect_value_error(lambda:chunk_forward(shared_model,np.array([-1])))
expect_value_error(lambda:chunk_forward(shared_model,np.array([shared_model.config.vocab])))
expect_value_error(lambda:chunk_forward(shared_model,np.array([1]),[None]))
bad_lengths=[(k[:,:1],v[:,:1]) if i==0 else (k,v)
             for i,(k,v) in enumerate(valid_state)]
expect_value_error(lambda:chunk_forward(shared_model,np.array([3]),bad_lengths))
bad_heads=[(k[:1],v[:1]) for k,v in valid_state]
expect_value_error(lambda:chunk_forward(shared_model,np.array([3]),bad_heads))
```

### 练习十二：从结构变化判断能否复用旧缓存

逐项判断五种操作：改变采样温度、改变输出 top-k、更新模型权重、改变 prompt 的第二个 token、只追加新 token。答案依次是：前两项通常可以复用模型 K/V，因为它们只改变从 logits 选 token 的策略；更新权重后不能沿用原缓存，因为 K/V 的函数已经改变；修改第二个 token 只能复用它之前的不变前缀；追加新 token 正是标准缓存适用情形。

还要说明边界：如果所谓温度修改实际改变了模型内部 attention 温度，而不是输出采样温度，那么隐藏状态和 K/V 可能改变，不能套用前一个答案。术语相同不代表作用位置相同。判断依据始终是“过去状态的计算函数是否发生变化”，而不是参数名字听起来像推理选项还是训练选项。

### 延伸工作坊一　把复杂度推导到底：缓存到底省了哪一项

先只看投影和逐位置 FFN。设每个位置通过全部层的这些运算成本近似为常数 C，生成过程先后处理长度一、二、三直到 T 的前缀。无缓存路径在第 t 步重新处理 t 个位置，总成本为 $C\sum_{t=1}^{T}t=CT(T+1)/2$。缓存路径每步只处理一个新位置，总成本约为 CT。这里的常数 C 实际与层数、隐藏维度和 FFN 宽度有关，但对固定模型它不随历史长度变化。

再看标准注意力。对长度 t 的完整序列，所有查询与所有键比较，矩阵元素数量约为 $t^2$，因果三角可以减少常数但不改变数量级。无缓存反复执行所有前缀，总量近似 $\sum_{t=1}^{T}t^2=T(T+1)(2T+1)/6$，随 T 的三次方增长。缓存每步只有一个新查询，与 t 个历史键比较，总量为 $\sum_{t=1}^{T}t=T(T+1)/2$，随 T 的平方增长。缓存显著减少重复，但没有把整段生成的注意力总成本降为线性。

这两个推导解释了为什么“每步只算一个 token”不等于“每步成本相同”。投影和 FFN 的新位置工作近似固定，attention 的历史读取越来越长。实际运行还包含大矩阵与小矩阵的效率差异、内核启动、输出头、采样和调度，因此墙钟时间不会严格按这些多项式变化。复杂度提供增长方向，基准测试给出具体硬件上的常数与瓶颈。

若已有长度 P 的 prompt，再生成 G 个 token，prefill 一次处理 P，后续 decode 的注意力键长度依次约为 P 加一到 P 加 G。总历史读取规模为 $GP+G(G+1)/2$，而不是仅仅 G。长 prompt 即使只生成很短回答，后续每个 token 也要面对较长缓存。相反，短 prompt 长生成会逐步积累历史。这两类负载在容量与延迟上的表现不同，服务测试应分别覆盖。

分块 prefill 没有改变所有合法 token 对的总量。若块大小为 c，第一个块处理块内三角，第二个块既读第一个块又处理自己的三角，以此类推。所有查询最终仍读取它之前的全部键，合法连接数总共是 $P(P+1)/2$。分块的主要价值在于控制单次工作量、临时内存和调度公平性，而不是凭空删除模型定义的连接。若某实现声称分块后计算量大幅减少，应检查它是否同时引入稀疏、窗口或近似机制。

### 延伸工作坊二　分块等价性的第二个证明：按层与按时间交错

完整前向通常按层推进：先计算全部 token 的第一层，再计算全部 token 的第二层。逐 token 缓存则按时间推进：先让第一个 token 通过所有层，再让第二个 token 通过所有层。两种执行顺序看起来差别很大，为什么能相同？关键在于它们都是同一张有向无环依赖图的合法拓扑顺序。

图上的节点可以写成“位置 i 的层 l 状态”。一个节点依赖前一层位置零到 i 的状态，不依赖未来位置。完整前向先完成某层所有位置，再进入下一层，显然满足依赖。缓存前向处理位置 i 时，过去位置在所有层的 K/V 已经存在；当前位置在前一层的状态刚刚算出，所以它需要的依赖也全部满足。只要不把尚未计算的未来值读进来，两种顺序就执行同样的函数。

这个解释还能帮助理解分块。块内各位置通过矩阵运算并行计算，但每层使用因果 mask 限制依赖；块间过去状态通过 cache 提供。它在“按层”和“按时间”之间选择一个中间粒度。并行并不意味着未来可见，因为并行计算只是调度方式，mask 决定信息依赖。训练时一次计算完整目标序列也同理：虽然所有 token ID 都已在内存里，当前位置仍不能读取被遮挡的未来信息。

数值上，两种拓扑顺序可能导致矩阵乘法形状不同，从而使用不同归约路径，这解释了容差比较的必要性。数学等价保证实数运算结果相同，不保证任何浮点实现逐位一致。如果使用随机 dropout，完整与分块路径还可能消费不同随机掩码，必须先切到确定的推理模式，或严格控制掩码对应关系。把随机性关掉并不是掩盖问题，而是先隔离需要验证的确定函数。

现在可以回答一个高级问题：能否让第二层提前处理某些 token，而第一层还在处理后面的 token？只要所需前缀依赖已经完成，原则上可以。这属于流水调度与执行引擎问题。模型公式本身没有强制整个序列的某层必须等待未来位置都算完。理解依赖图，会让你在后续框架卷阅读流水、连续批处理和分块执行时更容易判断哪些重排合法。

### 延伸工作坊三　训练为什么不能随意照搬推理缓存

推理时权重固定，旧 K/V 是可复用的常量。训练时权重正在更新，且损失要对产生 K/V 的参数求导。如果你在一个训练步骤中把旧 K/V detach，当作不带梯度的缓存，后续位置的损失就不能通过这些历史键和值反传到它们的来源。前向数值可能与完整序列相同，但梯度函数已经不同。是否允许这种截断，必须由训练目标明确规定。

举一个标量例子。设参数 w 同时生成历史 value $v_1=w x_1$ 和当前输出中的另一项，损失依赖当前查询读取到的 $v_1$。完整反向会给 w 一条经过历史 value 的梯度路径。若缓存 v1 后把它当常量，当前损失仍读到相同数值，却丢掉这条路径。因此“full 与 cache logits 一致”不足以证明训练梯度一致，训练验证还需要比较各参数梯度。

如果在每个 token 后更新一次权重，问题更明显。第一个 token 的 cache 由旧权重产生，第二个 token 的新投影由更新后权重产生，模型内部混用了两套参数。这样可能构成某种有意设计的在线算法，但不再等于固定参数下完整前缀的前向。普通语言模型批量训练不应无意中这样做。

可以设计保留计算图的状态传递，或者使用截断反向传播处理长序列，但必须明确保存图的内存成本、更新边界和梯度截断位置。对于循环状态模型，类似问题也存在：状态可以跨片段传递，梯度是否跨片段传递是另一个选择。不要把“能保存状态”与“能够免费保存完整训练依赖”混为一谈。

因此本章的缓存实现明确用于确定的前向验证与推理。若把它移植到自动微分框架，应先在很短序列上比较 full 与分块的参数梯度，确保没有意外 detach、原地覆盖或跨优化步复用。只有前向和梯度都符合目标定义，才能说训练路径等价。这个区分会在框架与算子卷的自动微分和内存管理中再次出现。

### 延伸工作坊四　边界条件工作坊：窗口、前缀与动态位置

滑动窗口缓存常被描述为“只保留最近 W 个 token”。如果模型原本定义全历史 attention，删除更早 K/V 就改变了函数，因此属于近似或不同模型。若模型本来就是窗口 attention，那么只保留仍可能被未来查询访问的状态可以是精确优化。判断时应从 mask 定义出发，而不是从缓存长度出发。

窗口里物理下标会循环复用，但逻辑位置不能随意循环。例如窗口大小四，当前已处理一百个 token，缓冲区第零槽可能存逻辑位置九十六。新查询的 RoPE 位置仍是一百，而不是零。mask 和位置编码需要逻辑时间，数组寻址需要物理槽，两者应通过映射连接。把环形槽号直接当位置号，会在第一次绕回后产生难以察觉的错误。

前缀共享也有类似的身份问题。两个请求具有同样的前一百个 token，可以共享这部分缓存；若一个请求增加系统提示导致位置整体改变，即使后面文本相同，也不能直接拿原来的旋转键使用。某些固定频率 RoPE 情况下可以推导坐标变换，但必须同时处理全部相关分量，不能靠“内容一样”跳过推导。工程系统通常选择更严格的缓存键以换取简单可靠的正确性。

动态 batch 的请求长度不同，不能用一个公共长度给所有请求生成 position。假设 A 已有十个 token，B 已有三个，新一步的逻辑位置分别十和三。将两条请求放在同一次矩阵调用中，不会使它们变成同一时间轴。每请求的位置、有效键长度和页表都必须独立。批处理是共享计算资源，不是共享历史内容。

如果某请求被取消，释放状态后另一个请求重用同一物理页，页中的旧数值可能仍在。正确实现只允许读取新请求已经写入的有效位置，不能因为内存有内容就认为缓存有效。有效长度与页表更新必须和写入顺序一致。即便本章不实现并发内核，也要理解状态元数据的正确性与数组数值同样重要。

### 延伸工作坊五　渐进项目：做一个可解释的缓存实验台

第一阶段保留本章 NumPy 模型，包装一个显式请求对象，字段至少包括请求编号、token 历史、cache、下一位置、最新 logits 和结束原因。实现 create、prefill、step、fork、truncate 和 close。每个操作都写清楚前置条件与返回后状态。不要一开始加入网络服务，先在单进程测试中证明生命周期一致。

第二阶段制作固定的实验矩阵。输入长度取一、八、三十二、六十四，KV 头数取一、二、四，块长度取一、三、整段。记录 full 与 chunk 最大误差、缓存实际 nbytes、理论容量、处理耗时和每层 shape。CPU 时间会受噪声影响，应重复几次报告中位数，不把一次微小差异解释成架构优势。数学等价与性能比较分开呈现。

第三阶段替换存储方式。把 concatenate 状态换成预分配缓冲，保持外部请求接口不变。先验证全套数值测试，再比较追加的数据搬运量。若做分页原型，可以用页列表加逻辑位置映射，在读取时 gather 成连续数组作为正确性基准；这还不是高性能分页 attention，但足以检验页表、尾页和回滚语义。不要因为底层仍有 gather 就声称已经实现生产级加速。

第四阶段加入可控故障开关：错误位置偏移、错误层交换、GQA 交替映射、重复旋转历史 K、跨请求复用和容量越界。实验台应给出具体失败位置，而不是只输出一个 False。每种故障写一段解释，指出它违反了哪条依赖或身份契约。能主动构造反例，是从“会调用缓存”走向“理解缓存”的重要一步。

项目报告最后回答三个问题。第一，哪些改动只改变物理存储，仍与完整模型等价？预分配、正确分页和合法执行重排属于这一类。第二，哪些改动有意改变数值或可见性？量化、窗口截断和近似压缩属于这一类。第三，哪些改动直接破坏状态身份？跨请求、跨层和跨权重复用属于这一类。三类问题需要不同的验收方法，不能都用“输出差不多”来判断。

掌握本章之后，你应能从一段推理代码中指出每一份缓存的来源、shape、位置与所有权，能独立实现 full、step 和 chunk 的等价路径，能算出显存与分配浪费，也能解释 GQA 为什么减少独立 KV 数据而保留查询头的表达。下一章将继续追问：能否连每个 token 的 K/V 特征维度也压缩，并通过代数变换避免每次完整展开？

### 项目验收补充：不要用随机文本代替数值证据

提交项目时至少保留一个可复现的小样例：固定种子、固定权重配置、固定 token 序列、明确分块边界，以及每个位置的最大误差。随机权重模型生成的 ID 没有自然语言意义，却非常适合结构验证，因为你可以严格控制全部输入。若直接拿一个大模型聊天结果当作验收，采样、模板、服务版本和截断策略会同时影响输出，很难判断变化到底来自哪里。

还应保留一个非退化的确定参数样例，例如本章两个键为二和零、两个值为三和五的手算。随机测试覆盖广，手算测试便于解释；两者互补。测试失败时，先让确定样例通过，再扩大随机尺寸。不要通过删掉失败种子或只比较最后 token 来“修复”实现。

继续阅读原始资料时，重点对应 [GQA 论文](https://arxiv.org/abs/2305.13245) 的分组定义与训练转换问题，并结合本卷 [固定源码导读](SOURCE_WALK.md) 查看实际模型如何形成查询头与键值头。论文中的质量实验回答训练配方和模型能力，本文的等价测试回答给定参数下的实现正确性。两种证据解决的问题不同，不能用一方替代另一方。生产框架的接口还可能改变缓存布局，因此移植本章代码前，应重新核对目标版本的状态契约。

最后检查有效长度与数组容量是否分别记录；二者混用会让未写入位置进入注意力。
