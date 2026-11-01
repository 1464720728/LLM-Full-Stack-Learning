# M12　混合架构与 Attention Residuals：沿时间记忆，沿深度选择

[本卷导航](README.md) · 前置：[MoE](08-moe.md)、[线性注意力](10-linear-attention.md)、[状态空间](11-state-space.md) · 下一章：[量化](13-quantization.md)

你已经实现不同的序列算子。现在有两个新的设计问题：能否在不同层使用不同序列机制；能否改变层与层之间的残差聚合，而不只更换 attention？前者形成混合架构，后者引出深度方向的连接设计。本章将把时间、特征、深度、专家和残差流五种轴分开，完整拆解已经可运行的 Block Attention Residuals 解码器，并用手算、反向与状态测试证明它究竟做了什么。

本章依据 [Attention Residuals 论文](https://arxiv.org/abs/2603.15031) 与本卷固定源码导读中的官方伪代码讨论其核心机制。微型模型使用仓库自写 GQA/RoPE 与 SwiGLU，随机权重、无训练，不宣称兼容官方 checkpoint。多流受约束混合部分是用于理解 mHC 的明确教学模型，不把一个 Sinkhorn 函数称为完整 mHC。

## 第一课　先分清五种不同的选择轴

token attention 为当前序列位置选择历史位置，softmax 通常沿键的时间轴；MoE 为当前 token 选择专家，归一化或 top-k 沿专家轴；Attention Residuals 为当前子层选择较早深度的来源，softmax 沿深度来源轴；多流残差混合在同一深度的多个残差流之间变换；通道投影则在线性层的特征坐标之间混合。它们都可能出现矩阵乘法与权重，但处理的对象不同。

假设张量 shape 为来源数 N、token 数 T、隐藏维 D。深度聚合对每个固定 token t，把 N 条来源的 D 维向量混合，输出 T 乘 D。它不会直接读取另一个 token 的位置。某条来源本身可能已经经过 token attention 吸收了历史信息，但当前深度操作的索引仍然只在同一 t 上变化。

这个区分直接关系因果性。若错误地把 N 与 T 展平成一个轴做 softmax，当前 token 就可能读到未来 token 的某个深度来源。shape 最后仍可能被整理成 T 乘 D，程序能运行，却破坏自回归。正确实现应在函数名、shape 注释和测试中明确归一化轴，不能只写一个没有 axis 的 softmax。

多种机制可以同时存在：一个层的时间算子可以是 GQA，FFN 可以是 MoE，层间聚合可以是 AttnRes；另一些层可以用固定状态记忆。这不意味着它们效果必然叠加，更不意味着名字越多越先进。每种机制增加自己的参数、状态和训练交互，需要分别理解和验证。

## 第二课　普通残差的展开与它没有告诉你的事

普通 PreNorm 子层可以写成下一状态等于当前状态加一个由当前状态产生的增量。连续展开后，最终状态等于初始 embedding 加上所有先前子层增量。这个显式加法中的系数都是一，但每个增量依赖前面已经混合的状态，所以整个网络不是若干独立函数的简单求和。

残差提供一条直接传递与梯度路径，让子层学习相对修改。它并不保证各层增量都同样重要，也不保证幅度始终相近。深层网络可能需要有选择地组合不同深度的表示。Attention Residuals 的动机之一，就是把“固定累加所有来源”改成内容相关的深度读取，而不是只在子层内部改 attention。

不能把普通残差的状态 x 当成一份可拆开的完整历史档案。虽然代数上能写成增量和，运行时若只保存 x，通常无法从和中唯一恢复每个增量。深度 attention 若要分别选择来源，需要显式保留这些来源或某种分块摘要，因此带来额外存储和读取成本。结构表达与资源开销在这里再次形成取舍。

手算一个标量例子，初始 E 为二，两个外部增量为三与负一，普通残差最终四。若把 E、三、负一当作三条独立来源做均匀深度平均，结果四除三，不是四。这说明归一化深度混合与普通残差不是仅换一种写法；它改变了聚合函数。真实增量又依赖聚合结果，差异会沿深度继续传播。

## 第三课　Full Attention Residual 的精确定义

设来源 $X_n$ 的 shape 为 T 乘 D，n 从零到 N 减一。每个来源沿最后特征轴做 RMSNorm，得到匹配用的 key；当前子层有一个学习 query q，shape 为 D。对每个 token，分数为 q 与该来源归一化向量的点积。沿 N 轴 softmax 得到权重，再用这些权重对原始、未归一化来源加权求和。

公式为
$$
r_{n,t}=\sqrt{\frac1D\sum_dX_{n,t,d}^2+\epsilon},\quad
s_{n,t}=\sum_dq_dX_{n,t,d}/r_{n,t},
$$
$$
\alpha_{n,t}=\operatorname{softmax}_{n}(s_{n,t}),\qquad
Y_{t,d}=\sum_n\alpha_{n,t}X_{n,t,d}.
$$
epsilon 控制零向量与极小向量处的分母，属于函数定义。key 归一化减少来源幅度对匹配的直接影响，value 保留原幅度承载内容。若把 value 也归一化，就是另一个模型。

```python
import numpy as np
from labs.models.core import attention_residual,rms_norm,softmax

def depth_attention_reference(sources,query,eps=1e-6):
    sources=np.asarray(sources,dtype=float)
    radius=np.sqrt(np.mean(sources*sources,axis=-1,keepdims=True)+eps)
    keys=sources/radius
    scores=np.einsum("ntd,d->nt",keys,query)
    weights=softmax(scores,axis=0)
    output=np.einsum("nt,ntd->td",weights,sources)
    return output,weights,keys

sources=np.array([[[1.,0.],[2.,1.]],
                  [[0.,2.],[1.,3.]],
                  [[3.,3.],[-1.,2.]]])
query=np.array([.3,-.2])
reference,weights,keys=depth_attention_reference(sources,query)
library,library_weights=attention_residual(sources,query)
np.testing.assert_allclose(reference,library,atol=1e-12)
np.testing.assert_allclose(weights,library_weights,atol=1e-12)
np.testing.assert_allclose(weights.sum(axis=0),np.ones(sources.shape[1]))
```

代码中的 einsum 明确保留 token 轴：ntd 与 d 收缩特征得到 nt，再用 nt 权重与 ntd 来源收缩来源轴得到 td。这里 n 是深度来源，不是 batch；如果增加 batch，应再明确一个独立轴。不能把来源数量与层数永远视为相同，因为 block 变体把多个子层增量合成一个来源。

## 第四课　完整手算：零 query、方向偏好与 value 幅度

一个 token 的三条来源分别为一、零；零、二；三、三。query 全零时，无论 RMSNorm 后的 key 是什么，三个分数都为零，softmax 权重各三分之一，输出为四除三、五除三。这个退化是重要测试：若输出变成归一化来源的均值，说明 value 被错误归一化；若权重不是均匀，说明 softmax 轴或初始化有问题。

忽略极小 epsilon 以便纸上近似，第一条 key 为根号二、零，第二条为零、根号二，第三条为一、一。query 取一、零，分数为根号二、零、一。指数分别约四点一一三三、一、二点七一八三，分母约七点八三一六，权重约零点五二五二、零点一二七七、零点三四七一。输出第一分量为第一权重加三倍第三权重，第二分量为二倍第二权重加三倍第三权重。

这不是硬选择，有限 query 仍让多个来源有贡献。把 query 放大可以让分布更尖，但也改变梯度与训练行为。不能看到最大权重属于第一来源，就说输出只使用第一来源；应计算所有加权贡献。对于高维向量，方向与幅度都可能影响最终结果。

```python
hand_sources=np.array([[[1.,0.]],[[0.,2.]],[[3.,3.]]])
zero_read,zero_weights,_=depth_attention_reference(hand_sources,np.zeros(2))
np.testing.assert_allclose(zero_read,[[4/3,5/3]])
np.testing.assert_allclose(zero_weights,np.full((3,1),1/3))
direction_read,direction_weights,_=depth_attention_reference(hand_sources,np.array([1.,0.]))
assert direction_weights[0,0]>direction_weights[2,0]>direction_weights[1,0]
np.testing.assert_allclose(direction_read[0],
    direction_weights[:,0]@hand_sources[:,0,:])
```

把第一条来源整体乘十，归一化 key 的方向近似不变，所以分数近似保持；但原始 value 变大，最终输出贡献会显著变化。epsilon 使极小向量处不完全尺度不变，因此测试不应要求所有幅度下逐位相同。这个设计区分了“匹配依据”与“被读取内容”，与 token attention 中 K/V 职责分离有相似之处。

## 第五课　完整反向：来源同时作为 key 与 value

深度聚合的来源有两条梯度路径：它直接作为 value 参与加权和，也通过 RMSNorm 影响分数和权重。上游梯度 G 为 T 乘 D。对每个权重的梯度是 G 与对应来源向量的点积；经过来源轴 softmax 后，分数梯度为 alpha 乘权重梯度减去加权平均。query 梯度是所有 token 与来源的分数梯度乘归一化 key 后求和。

来源的直接 value 梯度为 alpha 乘 G。key 路径先把分数梯度乘 query 得到 RMSNorm 输出梯度，再经过归一化导数。若半径为 r，输入 x、上游 g，RMSNorm 输入梯度为 g 除 r，减去 x 乘 g 与 x 的点积再除 D r 的三次方。最后把 key 与 value 两条来源梯度相加。

零 query 时，来源的 key 路径梯度为零，但 query 自己的梯度一般不为零。因此零初始化并不意味着深度选择永远不能学；它提供初始均匀混合，任务梯度可以推动 query 改变。若所有来源完全相同，改变权重不会改变输出，query 梯度才可能退化为零。这两个情况不能混淆。

```python
def depth_attention_backward(sources,query,upstream,eps=1e-6):
    output,weights,keys=depth_attention_reference(sources,query,eps)
    radius=np.sqrt(np.mean(sources*sources,axis=-1,keepdims=True)+eps)
    dweight=np.einsum("td,ntd->nt",upstream,sources)
    dscore=weights*(dweight-np.sum(weights*dweight,axis=0,keepdims=True))
    dquery=np.einsum("nt,ntd->d",dscore,keys)
    dkeys=dscore[:,:,None]*query[None,None,:]
    dim=sources.shape[-1]
    dsource_key=dkeys/radius-sources*np.sum(dkeys*sources,axis=-1,keepdims=True)/(dim*radius**3)
    dsource_value=weights[:,:,None]*upstream[None,:,:]
    return dsource_key+dsource_value,dquery

rng=np.random.default_rng(12)
grad_sources=rng.normal(size=(3,2,3))
grad_query=rng.normal(size=3)
grad_upstream=rng.normal(size=(2,3))
analytic=depth_attention_backward(grad_sources,grad_query,grad_upstream)
for parameter,expected in zip((grad_sources,grad_query),analytic):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=float(np.sum(depth_attention_reference(grad_sources,grad_query)[0]*grad_upstream))
        parameter[index]=old-1e-6
        minus=float(np.sum(depth_attention_reference(grad_sources,grad_query)[0]*grad_upstream))
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=2e-7,rtol=2e-7)
zero_source_grad,zero_query_grad=depth_attention_backward(
    grad_sources,np.zeros(3),grad_upstream)
np.testing.assert_allclose(zero_source_grad,
    np.broadcast_to(grad_upstream/3,grad_sources.shape),atol=1e-12)
assert np.linalg.norm(zero_query_grad)>0
```

这段代码验证了完整原语的导数，不是只训练一个脱离来源的权重向量。将它放入完整模型时，来源来自前面真实子层，梯度还会继续流回那些子层；多个后续深度聚合读取同一来源时，贡献也要累加。深度连接改变的不只是前向混合，还改变梯度在各来源间的分配。

## 第六课　为什么需要 Block：保存粒度决定资源

Full 深度 attention 可以为每个子层保留独立来源，深度增加时来源数增长，读取和存储也增长。Block 变体把一段子层增量累加成一个块来源，同时保留当前未完成块。这样减少来源数量，但块内的多个增量不再能被后续独立选择。它是表达粒度与资源之间的明确取舍。

一个完整 Transformer 层通常包含 attention 与 FFN 两个子层。本仓库 block_sublayers 按这两个子层计数，不按完整层计数，且要求正偶数、最小二。原因是当前实现沿固定官方伪代码在 attention 前检查块边界；若允许奇数而仍只在 attention 前检查，某些边界无法按声明发生。要支持任意边界，需要重写生命周期并重新测试。

必须区分三个对象：completed 是已经封存的深度来源列表；partial 是当前块内增量累加；每层 KV cache 是跨 token 的时间状态。completed 沿深度增长，KV 沿时间增长；逐 token step 时会为新 token 重新构造它自己的 completed/partial，不能把上一个 token 的深度列表接到下一个 token 后面。

初始状态采用 completed 为空、partial 等于 embedding。第一次 attention 先读取这唯一来源，再把 embedding 封存，清空 partial，使第一个 attention 增量成为新计算块起点。embedding 只计入一次。若初始化时 completed 已经含 embedding，同时 partial 也等于 embedding，第一次读取就重复计数，即使 shape 和权重和检查都可能通过。

## 第七课　完整走完三层：先读取，再封存，再开始新块

设三层模型，每层两个子层，总共六个子层，块大小四。用 E 表示 embedding，用 a0、m0、a1、m1、a2、m2 表示各层 attention 与 MLP 的实际输出增量。每个都是 T 乘 D 数组，而且依赖前面的深度混合，不是预先独立给定的常数。为了理解状态机，先把依赖暂时记成符号，再回到真实计算。

| 读取位置 | 本次来源 | 读取之后的变化 |
|---|---|---|
| attention 0 | E | 封存 E；新 partial 为 a0 |
| MLP 0 | E、a0 | partial 加 m0 |
| attention 1 | E、a0+m0 | 不封存；partial 加 a1 |
| MLP 1 | E、a0+m0+a1 | partial 加 m1 |
| attention 2 | E、a0+m0+a1+m1 | 先读旧 partial，再封存；新 partial 为 a2 |
| MLP 2 | E、已封存前四增量块、a2 | partial 加 m2 |
| 输出头 | E、已封存块、a2+m2 | 聚合后做 RMSNorm 与词表投影 |

来源数因此依次为一、二、二、二、二、三、三。最容易误写的是 attention 2：它处于块边界，但本次读取仍只有两条来源，因为封存发生在读取之后。若先封存又把旧 partial 同时保留为当前来源，就会重复读取；若先清空 partial 再读，又会丢掉旧块信息。正确顺序必须明确到语句级。

在边界后的新块，partial 应从新 attention 增量开始，不能写成旧 partial 加新增量。旧 partial 已经作为 completed 保存，再加进新 partial 会让同一历史通过两条路径重复进入后续混合。这个错误不一定改变来源数量，所以只测试计数不够，还要测试具体来源数值。

最后输出头也需要一个明确闭合规则。仓库微型模型使用额外的学习 query 读取所有 completed 与当前 partial，再 RMSNorm 和输出投影。这个额外 head query 是教学模型明确的设计约定，不能据此推断所有官方 checkpoint 的末端读出都相同。若只取最后 partial，会忽略所有 completed，属于另一个读出变体。

## 第八课　独立复写完整执行器：每一步都有来源

下面不调用原模型的内部执行器，而按上一课状态表独立写一条前向路径。它使用同一套真实 attention 与 SwiGLU 参数，便于与仓库完整模型逐项比较。函数接收当前 token 段、全局位置与每层时间 cache，返回 logits、新 cache 和深度 trace。深度状态在本次调用内构建，时间 cache 由调用者显式传入。

```python
from labs.models.core import DecoderConfig,swiglu
from labs.models.block_attnres_decoder import TinyBlockAttnResDecoder

def independent_attnres_execute(model,ids,positions,cache=None):
    ids=np.asarray(ids,dtype=int)
    if cache is None:
        cache=[None]*model.config.layers
    completed=[]
    labels=[]
    partial=model.embedding[ids].copy()
    partial_start=None
    events=[]
    next_cache=[]
    def read_depth(index,kind):
        label=("embedding" if partial_start is None
               else f"partial[{partial_start}:{index}]")
        values=np.stack(completed+[partial])
        mixed,weights,_=depth_attention_reference(values,model.depth_queries[index])
        events.append({"index":index,"kind":kind,"labels":tuple(labels+[label]),
                       "sources":values.copy(),"mixed":mixed.copy(),
                       "weights":weights.copy(),"sealed":None})
        return mixed
    for layer,block_weights in enumerate(model.blocks):
        index=2*layer
        mixed=read_depth(index,"attention")
        if index%model.block_sublayers==0:
            label=("embedding" if partial_start is None
                   else f"block[{partial_start}:{index}]")
            completed.append(partial.copy())
            labels.append(label)
            events[-1]["sealed"]=label
            partial=None
            partial_start=index
        delta,new_state=model._attn(rms_norm(mixed),block_weights,positions,cache[layer])
        partial=delta if partial is None else partial+delta
        next_cache.append(new_state)
        mixed=read_depth(index+1,"mlp")
        delta=swiglu(rms_norm(mixed),block_weights["gate"],
                    block_weights["up"],block_weights["down"])
        partial=partial+delta
    mixed=read_depth(2*model.config.layers,"head")
    logits=rms_norm(mixed)@model.output
    return logits,next_cache,events

model=TinyBlockAttnResDecoder(DecoderConfig(layers=3,seed=12),block_sublayers=4)
ids=[1,2,3,4]
independent_logits,independent_cache,independent_events=independent_attnres_execute(
    model,ids,np.arange(len(ids)))
library_logits,library_audit=model.forward(ids,return_trace=True)
np.testing.assert_allclose(independent_logits,library_logits,atol=1e-12)
assert [len(e["labels"]) for e in independent_events]==[1,2,2,2,2,3,3]
for ours,theirs in zip(independent_events,library_audit["events"]):
    assert ours["labels"]==theirs["source_labels"]
    np.testing.assert_allclose(ours["sources"],theirs["sources"],atol=1e-12)
    np.testing.assert_allclose(ours["mixed"],theirs["mixed"],atol=1e-12)
```

逐行阅读时，把 read_depth 看成只读操作：它把当前来源堆叠、计算深度权重、记录快照，不改变 partial。边界检查在读取之后，把旧 partial 复制到 completed，再将 partial 设为空。attention 产生新 delta 后，根据是否为空决定开始新块还是继续累加。MLP 之前再次深度读取，而不是直接使用 attention 的输出作为唯一输入。

这意味着一层中有两次独立深度 query：attention 前一次，MLP 前一次。它们可以学习不同的来源偏好。若只在整层开始混合一次，再把 MLP 接在 attention 后面，你实现的是另一种变体，不等于本章状态机。源代码中的二倍层数加一条 query，正对应每层两个子层加最终 head。

### 八点一　真实子层与外部增量演示的区别

仓库还保留 block_depth_aggregate，它接收外部给定 updates，便于手算深度快照，但它不计算真实 attention 或 FFN。完整 TinyBlockAttnResDecoder 则让每个增量由当前 mixed 经归一化、投影、RoPE、因果 attention 或 SwiGLU 产生。两者是不同教学层次，不能拿同名 block_size 直接比较而不核对计数单位和初始化规则。

外部增量演示适合定位生命周期，例如“封存前读到哪几条来源”；完整模型适合验证这些来源真的影响后续计算与 logits。一个完整课程需要两者，但必须说明各自证据范围。只展示外部常数更新，不能证明你已经把残差机制接进真实解码器；只给完整代码而不做小手算，又不利于初学者理解。

## 第九课　边界手算与零增量极限

考虑某个块边界，只有一个 token、二维特征，已封存 embedding 为二、零，当前旧 partial 为零、四，query 为零。本次 attention 前读取两条来源，均值为一、二。随后封存旧 partial，假设真实子层计算出的新增量暂记为三、三，则新 partial 为三、三。MLP 前三条来源均值为五除三、七除三。

这个手算强调“先读旧 partial”与“新 partial 不再包含旧块”。若错误地先把旧块封存并又保留 partial，attention 前会读 E、旧块、旧块三条，均值变二除三、八除三。若错误地清空后只读 E，结果二、零。三种输出显著不同，测试可以明确定位顺序错误。

```python
embedding_hand=np.array([[2.,0.]])
old_partial=np.array([[0.,4.]])
mixed_before,_ ,_=depth_attention_reference(
    np.stack([embedding_hand,old_partial]),np.zeros(2))
np.testing.assert_allclose(mixed_before,[[1.,2.]])
new_delta=np.array([[3.,3.]])
mixed_mlp,_,_=depth_attention_reference(
    np.stack([embedding_hand,old_partial,new_delta]),np.zeros(2))
np.testing.assert_allclose(mixed_mlp,[[5/3,7/3]])
wrong_duplicate,_,_=depth_attention_reference(
    np.stack([embedding_hand,old_partial,old_partial]),np.zeros(2))
np.testing.assert_allclose(wrong_duplicate,[[2/3,8/3]])
```

再把完整模型所有 attention 输出投影 O 与 MLP down 设零，所有计算增量都为零。三层、块大小四、零 query 时，最终来源为 E、一个零封存块、一个零 partial，所以最终 mixed 为 E 除三。输出不是直接用 E，而是 RMSNorm(E 除三) 后乘输出头。因为 epsilon 存在，不能声称它与 RMSNorm(E) 在所有输入上严格相同。

```python
import copy
zero_model=copy.deepcopy(model)
for block_weights in zero_model.blocks:
    block_weights["o"].fill(0.)
    block_weights["down"].fill(0.)
zero_logits,zero_audit=zero_model.forward(ids,return_trace=True)
expected_mixed=zero_model.embedding[ids]/3.
np.testing.assert_allclose(zero_audit["events"][-1]["mixed"],expected_mixed)
np.testing.assert_allclose(zero_logits,rms_norm(expected_mixed)@zero_model.output)
```

这个极限测试可以发现最终读出漏掉 completed、embedding 重复计数或边界累加错误。它不证明零增量模型有意义，而是利用可精确预测的退化情况验证结构。完整随机测试与退化测试应同时存在：前者覆盖丰富数值，后者提供明确解释。

## 第十课　快照、不可变性与深度状态所有权

completed 保存的是历史块的值，不是一个会继续改变的名字。若把 partial 的数组引用放进列表，之后对 partial 原地加 delta，历史来源也会被改写。Python 列表里出现两个变量名，不代表有两份独立内存。这个错误常常不报 shape 异常，最终输出也有限，却破坏了模型定义。

仓库封存使用复制并设只读，trace 也返回独立只读快照。这样后续计算或调试代码不能无意修改历史证据。教学实现宁可保守复制，先建立正确语义；高性能实现可以通过所有权、缓冲区轮换或不可变视图减少复制，但必须证明被引用数据不会再被覆盖。

```python
array=np.array([[1.,2.]])
bad_history=[array]
array+=3.
np.testing.assert_array_equal(bad_history[0],[[4.,5.]])
array2=np.array([[1.,2.]])
good_history=[array2.copy()]
array2+=3.
np.testing.assert_array_equal(good_history[0],[[1.,2.]])

saved_sources=[event["sources"].copy() for event in library_audit["events"]]
model.forward([5,6,7])
for saved,event in zip(saved_sources,library_audit["events"]):
    np.testing.assert_array_equal(saved,event["sources"])
    assert not event["sources"].flags.writeable
assert not np.shares_memory(library_audit["partial"],model.embedding)
```

训练自动微分还增加一层约束：某个张量可能被反向保存，之后原地修改会破坏梯度所需值。框架可能报版本错误，也可能在某些不受保护路径中产生错误结果。优化残差存储时不能只看前向输出，还要检查反向保存与重计算策略。深度来源的生命周期与时间缓存一样，需要明确所有权。

## 第十一课　full 与 step：深度状态不会跨 token 延续

逐 token 推理时，新 token 从自己的 embedding 开始走完整深度，在每个子层建立当前 token 的 completed/partial。跨时间传递的是每层 attention 的 K/V，而不是上一 token 的深度列表。深度聚合只读取同一 token 的不同深度来源，随后 token attention 才读取该层历史 K/V。这两种读取交替发生，职责清楚。

因此 full 与 step 可以保持等价：过去 token 在每层的状态由因果结构固定，新 token 的深度来源只依赖它在更早层的当前状态；attention 的历史读取由正确 cache 提供。若把深度聚合错误地混到时间轴，或者把上一 token 的 partial 直接传进下一 token，这个证明就不成立。

```python
for block_size in (2,4,8):
    tested=TinyBlockAttnResDecoder(DecoderConfig(layers=3,seed=21),block_size)
    tested.depth_queries[:]=rng.normal(0,.2,size=tested.depth_queries.shape)
    expected=tested.forward(ids)
    cache=None
    rows=[]
    for token in ids:
        row,cache=tested.step(token,cache)
        rows.append(row)
    np.testing.assert_allclose(np.stack(rows),expected,atol=1e-10,rtol=1e-10)
    altered=tested.forward([1,2,3,9])
    np.testing.assert_allclose(altered[:-1],expected[:-1],atol=1e-10)
```

这里特意使用非零 query。只测试零 query，会让深度选择退化为均值，某些分数计算错误可能被掩盖。块大小二、四、八覆盖每层封存、跨层封存与大于总子层数的情况。未来不干扰同时验证时间因果性，full/step 比较验证 cache 与深度生命周期组合。

step 还要检查每层 K/V 形状、头数、特征维、有限性和共同长度。深度机制不消除这些原有要求。若只有某一层 cache 长度错误，当前位置计数就不一致，RoPE 与 mask 可能同时失效。完整模型测试应保留上一章建立的状态契约，而不是只关注新增加的残差公式。

## 第十二课　多流残差与 mHC：先理解约束混合的数学

AttnRes 选择历史深度来源；多流残差则在同一深度维护多条并行残差流。设有 n 条流，每条 D 维，组成 X，shape 为 n 乘 D。一个教学层可以先用 pre 权重 a 把多流混成单个子层输入，计算 f(a 转置 X)，再用 post 权重 b 把子层输出写回多流，同时用矩阵 H 混合原残差流。形式为 $X'=HX+b\,f(a^\top X)$。

这只是说明三种职责的教学公式：pre 决定子层读哪些流，post 决定输出写到哪些流，H 决定残差本身怎样跨流传递。真实 [mHC 论文](https://arxiv.org/abs/2512.24880) 的约束、参数化、初始化与完整层结构应逐项核对，不能把本例当作全部实现。尤其不同后续变体可能改变求解与执行方式，不应凭同一个名字推断细节。

为什么约束 H？如果每层都用任意矩阵混合残差流，连续相乘可能放大或压缩某些方向。非负且行和为一的矩阵让每个输出流成为输入流的凸组合，限制某些幅度行为；若列和也为一，则所有流的总和在纯残差混合中保持。这样的双随机约束提供可分析结构，但不代表包含子层写入的整个网络自动稳定。

手算两流，X 为二、零与零、四，H 为四分之一、四分之三以及四分之三、四分之一。纯残差混合得到二分之一、三与一点五、一，两流总和仍二、四。pre 权重各二分之一，读出一、二；子层暂取乘二，输出二、四；post 权重零点二、零点八，分别写入零点四、零点八与一点六、三点二。最终两流为零点九、三点八和三点一、四点二。

```python
streams=np.array([[2.,0.],[0.,4.]])
mixing=np.array([[.25,.75],[.75,.25]])
pre=np.array([.5,.5])
post=np.array([.2,.8])
residual_mixed=mixing@streams
read=pre@streams
update=2*read
new_streams=residual_mixed+post[:,None]*update[None,:]
np.testing.assert_allclose(residual_mixed,[[.5,3.],[1.5,1.]])
np.testing.assert_allclose(residual_mixed.sum(axis=0),streams.sum(axis=0))
np.testing.assert_allclose(new_streams,[[.9,3.8],[3.1,4.2]])
assert not np.allclose(new_streams.sum(axis=0),streams.sum(axis=0))
```

最后一个断言提醒你：双随机 H 只约束纯混合支路，新增子层更新仍改变总和。不能看到约束矩阵就宣布整个网络范数不变。pre/post 的归一化也可能不同于本例，必须从公式确认。理解局部性质的作用范围，比把“稳定残差”当成口号更重要。

### 十二点一　Sinkhorn：交替归一化为何需要多次迭代

给一个正方正矩阵，先按行除以行和，每行和变一；再按列除以列和，每列和变一，但前一步的行和一般又被改变。重复交替可以在适当条件下逼近双随机矩阵。一次行列归一化不保证两个方向都精确满足，有限迭代只能报告实际误差。

双随机不意味着每个元素都等于一除 n。单位矩阵也是双随机，表示保持各流；置换矩阵也是双随机，表示交换；均匀矩阵表示平均；还有大量介于其间的混合。约束保留了选择空间，不是强迫所有流相同。

对二乘二正矩阵，最终双随机形式可以写成 p、一减 p以及一减 p、p。原矩阵元素一、二、三、四时，比例关系给出 p 为二除二加根号六，约零点四四九四九。这个例子说明归一化后不必均匀，也提供一个可独立验算的结果。

```python
from labs.models.core import sinkhorn

positive=np.array([[1.,2.],[3.,4.]])
doubly=sinkhorn(np.log(positive),iterations=100)
p=2/(2+np.sqrt(6))
np.testing.assert_allclose(doubly,[[p,1-p],[1-p,p]],atol=1e-12)
np.testing.assert_allclose(doubly.sum(axis=0),np.ones(2),atol=1e-12)
np.testing.assert_allclose(doubly.sum(axis=1),np.ones(2),atol=1e-12)
```

### 十二点二　为什么在 log-domain 归一化

如果先对极端 logits 取指数，可能上溢成无穷或下溢成整列零，再除列和产生 NaN。log-domain 直接对每行、每列减 log-sum-exp，最后才取指数。减去最大值计算 log-sum-exp 避免大指数，归一化的乘除在对数域变成加减。

以两行都为一千、零的 logits 为例，朴素 exp 一千上溢；即使先减全局最大值，第二列 exp 负一千可能全为零，列归一化仍失败。log-domain 保留负一千这个有限对数值，列归一化后能恢复两列相对质量，结果为全二分之一。这个反例说明仅做一次全局减最大值还不够。

```python
extreme=sinkhorn(np.array([[1000.,0.],[1000.,0.]]),iterations=30)
np.testing.assert_allclose(extreme,np.full((2,2),.5),atol=1e-12)
assert np.isfinite(extreme).all()
logits3=np.array([[3.,-1.,.2],[.4,2.,-.7],[-.2,.5,1.]])
errors=[]
for iterations in (1,5,100):
    matrix=sinkhorn(logits3,iterations)
    errors.append((np.max(np.abs(matrix.sum(0)-1)),
                   np.max(np.abs(matrix.sum(1)-1))))
assert errors[-1][0]<1e-10 and errors[-1][1]<1e-10
```

课程函数是理解约束工具，不是完整 mHC。真正多流架构还需要 pre/post 参数、残差矩阵生成、训练梯度、流展开与收缩、算子融合和模型级评估。把这些逐项列出来，才能知道自己已经实现什么、下一步需要验证什么。

## 第十三课　混合序列层：完整小模型与逐层状态表

混合架构可以在一些层使用全历史 attention，在另一些层使用固定状态。全历史层保留逐 token K/V，固定状态层保存有限矩阵或向量，局部卷积还可能有短缓存。整个模型的状态不是一条统一 KV 公式，而是逐层类型对应的状态列表。调度器和 checkpoint 接口都必须知道每层保存什么。

下面构造一个明确的教学混合栈，层类型为 attention、state、attention。attention 使用单头因果 softmax，无 RoPE；state 使用逐通道衰减状态，写入与读出有线性投影。每层把增量加到残差。它不是某个具名混合模型，但完整实现 full 与 step，展示异构状态怎样一起推进。

```python
from labs.models.core import attention

class TinyHybridStack:
    def __init__(self,dim=4,types=("attention","state","attention"),seed=120):
        self.dim=dim
        self.types=tuple(types)
        local_rng=np.random.default_rng(seed)
        self.layers=[]
        for kind in self.types:
            if kind=="attention":
                self.layers.append({name:local_rng.normal(0,.25,size=(dim,dim))
                                    for name in ("q","k","v","o")})
            elif kind=="state":
                self.layers.append({"decay":local_rng.uniform(.2,.9,size=dim),
                                    "write":local_rng.normal(0,.25,size=(dim,dim)),
                                    "read":local_rng.normal(0,.25,size=(dim,dim))})
            else:
                raise ValueError("unknown layer type")
    def forward(self,x):
        x=np.asarray(x,dtype=float).copy()
        states=[]
        for kind,weights in zip(self.types,self.layers):
            if kind=="attention":
                q=x@weights["q"];k=x@weights["k"];v=x@weights["v"]
                delta=attention(q,k,v,np.tril(np.ones((len(x),len(x)),dtype=bool)))[0]
                x=x+delta@weights["o"]
                states.append((k.copy(),v.copy()))
            else:
                state=np.zeros(self.dim)
                rows=[]
                for token in x:
                    state=weights["decay"]*state+token@weights["write"]
                    rows.append(state@weights["read"])
                x=x+np.stack(rows)
                states.append(state.copy())
        return x,states
    def step(self,x,states=None):
        if states is None:
            states=[None]*len(self.layers)
        x=np.asarray(x,dtype=float).copy()
        new=[]
        for kind,weights,old in zip(self.types,self.layers,states):
            if kind=="attention":
                q=(x@weights["q"])[None,:]
                k=(x@weights["k"])[None,:]
                v=(x@weights["v"])[None,:]
                if old is not None:
                    k=np.concatenate([old[0],k])
                    v=np.concatenate([old[1],v])
                delta=attention(q,k,v)[0][0]@weights["o"]
                x=x+delta
                new.append((k,v))
            else:
                previous=np.zeros(self.dim) if old is None else old
                state=weights["decay"]*previous+x@weights["write"]
                x=x+state@weights["read"]
                new.append(state.copy())
        return x,new

hybrid=TinyHybridStack()
hybrid_input=rng.normal(size=(8,4))
hybrid_full,hybrid_full_states=hybrid.forward(hybrid_input)
hybrid_states=None
hybrid_rows=[]
for token in hybrid_input:
    row,hybrid_states=hybrid.step(token,hybrid_states)
    hybrid_rows.append(row)
np.testing.assert_allclose(np.stack(hybrid_rows),hybrid_full,atol=1e-12)
assert hybrid_states[0][0].shape==(8,4)
assert hybrid_states[1].shape==(4,)
assert hybrid_states[2][0].shape==(8,4)
```

full 路径按层处理整段，state 层在该层内部递推；step 路径按 token 穿过所有层。两种拓扑顺序保持同一因果依赖，因此输出一致。注意第二个 attention 层的 K/V 来自已经经过第一 attention 与 state 层的表示，不能复用第一层 cache，尽管 shape 相同。

状态容量按层相加。这个例子两个 attention 层各保存两份 T 乘 D，加一个 D 维状态，总元素为四 T D 加 D。若把中间状态层换成完整 selective block，还要加 inner 乘 state_dim 与卷积缓存；若把 attention 改为 GQA，按 KV 头与每头维计算。层类型混合以后，模型名字不能替代实际账本。

### 十三点一　混合的动机与不能凭结构推出的结论

固定状态提供紧凑历史摘要，全 attention 提供更直接的逐记录读取。交替使用可能让模型在资源与能力之间形成新的取舍，但具体比例需要训练和评测。少量全局层是否足以恢复某类长程任务，取决于它们的位置、表示能力、训练数据与任务要求，不能从“每四层一个全局层”直接推导答案。

层顺序也重要。attention 后接 state 与 state 后接 attention 一般不交换，因为后一个模块接收不同输入。即使两种布局有相同参数数量与状态容量，函数和训练动力学仍不同。比较时应控制模型规模、训练预算与数据，明确只改变层类型排列，才能讨论结构影响。

混合还可能增加执行复杂度：不同状态需要不同内核、不同缓存分配和不同调度粒度。单一机制的理论优势在完整模型里可能被切换与同步成本抵消，也可能通过合理融合得到收益。数学课程建立正确性与资源模型，框架卷再用真实硬件测量，二者不能互相替代。

## 第十四课　十二组练习与完整参考答案

### 练习一：零 query 的均值验收

任意 N 条来源、T 个 token、D 维，query 为零时权重应为一除 N，输出是来源轴均值。答案应同时检查权重沿来源轴和为一，不能误检查时间轴。前面三来源手算已经给出具体数值，下面使用不同轴长度，避免 N 与 T 相等时掩盖轴错误。

```python
zero_fixture=rng.normal(size=(4,3,5))
zero_result,zero_alpha=attention_residual(zero_fixture,np.zeros(5))
np.testing.assert_allclose(zero_result,zero_fixture.mean(axis=0),atol=1e-12)
np.testing.assert_allclose(zero_alpha.sum(axis=0),np.ones(3))
assert zero_alpha.shape==(4,3)
```

### 练习二：value 不能被偷偷归一化

把两条来源设为一、零与零、十，零 query 输出应为二分之一、五。若 value 也做 RMSNorm，结果会接近两个单位方向的均值，幅度完全不同。完整参考如下。这个测试针对 key/value 职责，不能只用同范数来源，否则错误实现可能看不出来。

```python
amplitude_sources=np.array([[[1.,0.]],[[0.,10.]]])
amplitude_result,_=attention_residual(amplitude_sources,np.zeros(2))
np.testing.assert_allclose(amplitude_result,[[.5,5.]])
wrong_normalized_value=rms_norm(amplitude_sources).mean(axis=0)
assert not np.allclose(amplitude_result,wrong_normalized_value)
```

### 练习三：同一 token 的深度选择不应读取别的位置

改变 sources 中最后一个 token 的所有深度来源，前面 token 的深度聚合输出应不变。这个原语测试独立于完整模型的 token attention，直接验证 depth softmax 的轴。若把 N 与 T 展平再归一化，其他位置的分母会变化，测试就会失败。

```python
axis_sources=rng.normal(size=(3,4,2))
axis_query=np.array([.4,-.3])
axis_before,_=attention_residual(axis_sources,axis_query)
axis_changed=axis_sources.copy();axis_changed[:,-1]+=100.
axis_after,_=attention_residual(axis_changed,axis_query)
np.testing.assert_allclose(axis_before[:-1],axis_after[:-1],atol=1e-12)
```

### 练习四：预测不同块大小的来源数量

三层、六个子层，块大小二时来源数为一、二、二、三、三、四、四；块大小四时为一、二、二、二、二、三、三；块大小八时为一、二、二、二、二、二、二。最后一项是输出头。答案应解释每个边界先读后封存，因此新 completed 数量在下一次聚合才表现出来。

```python
expected_counts={2:[1,2,2,3,3,4,4],
                 4:[1,2,2,2,2,3,3],
                 8:[1,2,2,2,2,2,2]}
for size,counts in expected_counts.items():
    candidate=TinyBlockAttnResDecoder(DecoderConfig(layers=3),size)
    _,audit=candidate.forward(ids,return_trace=True)
    assert [len(event["source_labels"]) for event in audit["events"]]==counts
```

### 练习五：embedding 只能封存一次

检查 trace 中 completed_labels，embedding 应只出现一次，且第一次 attention 聚合来源只有 embedding。若初始化 completed 与 partial 都包含 E，会重复读取。答案不能只统计字符串，因为错误代码可能给重复来源不同名字；还应检查第一次 sources 的第一维等于一，并与 embedding 数值一致。

```python
assert library_audit["completed_labels"].count("embedding")==1
first_event=library_audit["events"][0]
assert first_event["sources"].shape[0]==1
np.testing.assert_allclose(first_event["sources"][0],model.embedding[ids])
```

### 练习六：封存快照不能被后续修改

第十课已经给出完整引用与复制反例。扩展答案应检查 trace 数组只读，并确认第二次 forward 不改变第一次 trace。若为了优化删除复制，必须提供新的所有权证明与相应测试，不能仅凭最终 logits 一次相同就认为安全。别名错误往往在复用缓冲区或并发请求时才暴露。

### 练习七：每个真实子层必须影响输出

分别把某一层 attention 的 O 或 MLP 的 down 设零，其余参数保持相同，完整模型 logits 通常应改变。每次都从原模型深复制，避免多个修改叠加。这个测试不要求所有可能参数下都必然变化，而是在固定非退化 fixture 上确认路径确实接入。

```python
baseline_logits=model.forward(ids)
for layer_index in range(model.config.layers):
    for parameter_name in ("o","down"):
        altered_model=copy.deepcopy(model)
        altered_model.blocks[layer_index][parameter_name].fill(0.)
        altered_logits=altered_model.forward(ids)
        assert np.max(np.abs(altered_logits-baseline_logits))>1e-8
```

### 练习八：零 query 为什么仍能训练

第五课证明 query 梯度一般非零。现在实际训练一个深度 query，让三条固定来源的混合接近第一条来源。只更新 query，来源保持固定，使用平方误差和完整反向。这个实验展示零初始化能离开均匀混合，不代表完整 AttnRes 语言模型已训练。

```python
learn_sources=np.array([[[2.,0.]],[[0.,2.]],[[1.,1.]]])
learn_query=np.zeros(2)
learn_target=np.array([[2.,0.]])
learn_losses=[]
for step in range(500):
    prediction,_,_=depth_attention_reference(learn_sources,learn_query)
    difference=prediction-learn_target
    learn_losses.append(float(np.mean(difference**2)))
    _,query_gradient=depth_attention_backward(
        learn_sources,learn_query,2*difference/difference.size)
    learn_query-=.1*query_gradient
assert learn_losses[-1]<learn_losses[0]*.02
learn_output,learn_weights,_=depth_attention_reference(learn_sources,learn_query)
assert learn_weights[0,0]>learn_weights[1,0]
assert learn_weights[0,0]>learn_weights[2,0]
```

如果三条来源完全相同，任何权重都得到同一输出，query 没有可利用的差异，梯度可为零。训练测试应有不同方向与内容，不应只验证参数更新了某个非零数。损失下降、权重变化与输出方向三者一起看，才有清楚的解释。

### 练习九：双随机不等于均匀

给出单位矩阵、交换矩阵和均匀矩阵，三者行列和都为一，但对多流输入行为不同。答案应分别解释保持、交换和平均。再指出 Sinkhorn 的有限迭代结果需要测量行列误差，不能因为函数名字叫归一化就假定精确满足。

```python
identity=np.eye(2)
swap=np.array([[0.,1.],[1.,0.]])
uniform=np.full((2,2),.5)
for matrix in (identity,swap,uniform):
    np.testing.assert_allclose(matrix.sum(0),[1.,1.])
    np.testing.assert_allclose(matrix.sum(1),[1.,1.])
np.testing.assert_allclose(identity@streams,streams)
np.testing.assert_allclose(swap@streams,streams[::-1])
np.testing.assert_allclose(uniform@streams,np.tile(streams.mean(0),(2,1)))
```

### 练习十：混合栈的状态容量

第十三课模型有两个单头全 attention 层、一个 D 维递推状态，长度 T，总持久元素为四 T D 加 D。T 八、D 四时是一百三十二。完整答案应从实际返回状态数组求和验证，而不是只引用公式。若增加 batch，按请求数乘；若各请求长度不同，对 attention 部分按长度求和。

```python
actual_elements=0
for kind,state in zip(hybrid.types,hybrid_states):
    actual_elements+=(state[0].size+state[1].size
                      if kind=="attention" else state.size)
assert actual_elements==4*8*4+4==132
```

### 练习十一：全局归一化错误的反例

把深度分数 N 乘 T 展平后做一个 softmax，再 reshape，虽然所有权重总和一，但每个 token 的来源权重和不再是一。用同一来源与 query 计算错误结果，比较正确输出。答案应指出这不仅改变尺度，还让某 token 的权重依赖其他 token 的分数，可能引入未来信息。

```python
_,correct_alpha,normalized_keys=depth_attention_reference(axis_sources,axis_query)
scores=np.einsum("ntd,d->nt",normalized_keys,axis_query)
wrong_alpha=softmax(scores.reshape(-1)).reshape(scores.shape)
assert not np.allclose(wrong_alpha.sum(axis=0),np.ones(scores.shape[1]))
wrong_output=np.einsum("nt,ntd->td",wrong_alpha,axis_sources)
correct_output=np.einsum("nt,ntd->td",correct_alpha,axis_sources)
assert not np.allclose(wrong_output,correct_output)
```

### 练习十二：完整模型验收与边界报告

独立复写执行器应通过来源标签与数值对照、零 query 均值、非零 query full/step、未来不干扰、块大小变化、零增量极限、快照不可变和各子层影响测试。完成后写明模型随机权重、没有训练归一化 gain、没有 dropout、最终 head query 是显式教学约定。准确报告边界不降低完成度，它防止把可验证教学实现夸大成官方权重复现。

## 第十五课　深入工作坊与渐进项目

### 十五点一　深度聚合的因果证明

对每个 token，深度聚合只读取该 token 在更早子层形成的来源。如果这些来源已经只依赖当前及过去 token，那么其加权混合也只依赖这些输入，因为 query 是模型参数，RMSNorm 和 softmax 只沿特征与深度轴操作，不引入其他时间位置。随后 causal token attention 只读取合法历史，逐位置 FFN 不跨时间，因此可以逐子层归纳整个模型因果。

这个证明说明测试应检查两类轴：深度 softmax 沿来源，token mask 沿时间。只验证 token attention 的下三角 mask 不够，因为深度聚合若混时间也会泄漏；只验证深度原语独立也不够，因为真实子层可能使用错误 cache 或位置。完整模型的未来不干扰测试把两类机制组合起来。

封存发生在深度边界，与时间因果边界不同。一个块包含若干子层增量，不表示包含若干未来 token。若把“block”误读成时间块，就会混淆缓存策略。代码变量名 block_sublayers 明确计数单位，文档和实验应保持一致。

### 十五点二　零 query 初始化的尺度效应

零 query 使每次读取对当前来源均匀平均，来源数量增加时，每条来源的直接权重变小。这不是普通残差的固定系数一，也不是把所有来源求和后简单乘一个全局常数，因为不同子层看到的来源集合和数量不同，后续增量又依赖这些混合。整个网络的尺度行为需要沿生命周期分析。

归一化可能缓解某些幅度变化，但 epsilon、输出投影与非线性意味着不能随意把平均和求和视为等价。第九课零增量极限已经展示最终 mixed 的确是 E 除来源数。即便 RMSNorm 在大幅度区域近似尺度不变，训练梯度与小幅度行为仍可能不同。比较初始化策略时应看中间状态与梯度，不只看最终 logits。

学习 query 后，来源权重随 token 内容变化，但 query 本身在同一子层通常跨 token 共享。它不是为每个 token 独立存一个可训练向量，而是用共享方向读取不同来源 key。若你设计输入相关 query，那是另一个变体，需要增加投影和梯度路径，不能混称为本章公式。

### 十五点三　Block 粒度如何限制可选择性

一旦多个增量被合并进同一个 partial，后续深度 attention 只能给整个块一个权重，不能独立放大块内某个增量、抑制另一个。块越大，来源更少、读取更便宜，但选择粒度更粗；块越小，保留更多深度细节，状态与计算增加。这与时间压缩把多个 token 合成摘要有相似的信息取舍，但压缩轴不同。

不能仅凭来源数减少就推断训练质量必然下降或上升。较粗聚合可能提供有用的归纳偏置，也可能丢失需要的深度选择；较细聚合增加自由度，也可能增加优化难度。合理实验在相同子层与训练预算下改变块大小，记录任务指标、深度权重分布、状态容量和运行时间。

如果块大小大于全部计算子层，最终只有 embedding 与当前计算增量块两条来源，并不是退化为普通残差。两条来源仍经过深度归一化混合，子层读取也不等于旧 x。反过来，块大小二意味着每完整层的两个子层增量形成一块，也不等于每个子层独立 Full 版本。计数与聚合规则要分别理解。

### 十五点四　深度 query 的梯度解释与局限

某来源的权重梯度等于上游输出梯度与该来源 value 的点积。softmax 再减去加权平均，表示相对其他来源的边际偏好。若某来源更有助于降低损失，它的分数可能被推动；但来源本身也会通过 key 与 value 两条路径更新，所以训练不是只学一个固定的层重要性排序。

不能把训练后的平均深度权重直接解释成某层“贡献百分比”。权重乘的是内容不同、幅度不同的来源，后续非线性也会放大或抵消影响。一个小权重来源可能具有很大幅度，或者在少数关键 token 上起作用；一个大权重来源也可能与其他来源高度冗余。解释需要同时看权重、value 范数和受控干预。

可以做删除来源或替换来源的消融，但它会改变归一化分母与后续状态，结果是一个干预实验，不是简单读取原权重。报告应说明干预方式，例如把 value 置零但保留 key，还是完全从候选集合删除；这两者不同。前者仍参与 softmax 竞争，后者会重新分配所有概率。

### 十五点五　多流约束的局部性质与完整网络

若 H 非负且每行和一，纯残差混合的每个输出流是输入流的凸组合，所以对任意坐标不会超过输入流该坐标的最大最小范围。若列和也一，流总和保持。这些性质可以直接测试，也有助于理解为什么约束可能抑制某些任意放大。

但是 pre/post 子层写入、非线性、归一化和多层组合仍然改变系统行为。H 双随机不保证整个 block 是等距变换，也不保证所有梯度范数都等于一。单位矩阵与均匀矩阵都双随机，前者保留流差异，后者会把流差异压平，说明约束集合内也有很不同的动态。

Sinkhorn 是一种产生近似双随机矩阵的方法，不是唯一方式。有限迭代、温度、数值精度和输入极值都会影响结果。若训练中使用有限轮归一化，推理突然改成更多轮，函数可能变化；应按模型定义保持一致，或把改变作为近似改造重新评估。不能把“更接近约束”自动等同“对已训练模型更正确”。

### 十五点六　跨时间状态与跨深度临时量的内存预算

完整 prefill 时，深度来源形状是 T 乘 D，来源数随块数量增长；逐 token decode 时，深度来源只有一乘 D，但每层时间 KV 仍随历史 T 增长。两阶段的内存主项不同。不能用逐 token 的小深度状态估计整段训练激活，也不能把 completed 当成需要永久跨请求保存的历史。

训练时反向可能需要保存来源、权重、归一化半径以及真实子层激活；使用重计算可以减少某些保存，但增加算术。教学 trace 为可解释性额外复制大量数组，不代表优化实现必须保存同样多副本。性能报告应关闭 trace 再测，并说明是否包含调试记录。

参数量也应单独计算。每个子层一个 D 维 query，加一个 head query，共二倍层数加一乘 D 个参数；这只是本教学模型新增的深度 query 参数，不包含可能的归一化 gain 或其他官方变体。新增参数很少，不意味着运行开销很小，因为读取多个 T 乘 D 来源的带宽成本可能更显著。

### 十五点七　Sinkhorn 的可微路径：约束不是停止学习

有限轮 log-domain Sinkhorn 由一串可微归一化组成。每次沿某轴把 log 值减去 log-sum-exp，形式上是 log-softmax。若该步输出的上游梯度为 g，输入梯度为 g 减去 softmax 概率乘该轴上 g 的和。注意这里不是普通 softmax 的梯度公式，因为输出是对数值；少乘或多乘一个概率都会错误。

最后从 log 矩阵取 exp，梯度先乘最终正矩阵，再逆序穿过所有行列归一化。归一化顺序正向是行、列，反向就从最后一轮的列、行开始。下面给出完整有限轮梯度并做逐元素差分。它说明约束矩阵可以被任务损失训练，但不等于已经实现整个 mHC 网络的反向。

```python
def sinkhorn_forward_saved(logits,iterations=5):
    log_values=np.asarray(logits,dtype=float).copy()
    saved=[]
    for _ in range(iterations):
        for axis in (1,0):
            maximum=log_values.max(axis=axis,keepdims=True)
            log_sum=maximum+np.log(np.exp(log_values-maximum).sum(axis=axis,keepdims=True))
            log_values=log_values-log_sum
            saved.append((axis,np.exp(log_values)))
    result=np.exp(log_values)
    return result,saved

def sinkhorn_backward(result,saved,upstream):
    gradient=upstream*result
    for axis,probabilities in reversed(saved):
        gradient=gradient-probabilities*gradient.sum(axis=axis,keepdims=True)
    return gradient

sink_logits=rng.normal(size=(3,3))
sink_upstream=rng.normal(size=(3,3))
sink_result,sink_saved=sinkhorn_forward_saved(sink_logits,5)
sink_grad=sinkhorn_backward(sink_result,sink_saved,sink_upstream)
sink_numeric=np.zeros_like(sink_logits)
for index in np.ndindex(sink_logits.shape):
    old=sink_logits[index]
    sink_logits[index]=old+1e-6
    plus=np.sum(sinkhorn_forward_saved(sink_logits,5)[0]*sink_upstream)
    sink_logits[index]=old-1e-6
    minus=np.sum(sinkhorn_forward_saved(sink_logits,5)[0]*sink_upstream)
    sink_logits[index]=old
    sink_numeric[index]=(plus-minus)/2e-6
np.testing.assert_allclose(sink_grad,sink_numeric,atol=2e-7,rtol=2e-7)
np.testing.assert_allclose(sink_result,sinkhorn(sink_logits,5),atol=1e-12)
```

这个反向对应五轮有限算法，而不是某个无限迭代极限的隐式求导。若训练实现用五轮，数值参考也应使用五轮；改成一百轮后，矩阵与梯度都可能变化。比较算法时必须把迭代次数作为函数定义的一部分。不要只看最终行列和更接近一，就忽略训练路径已经改变。

### 十五点八　因果 encoder-decoder：历史记录的来源也可以改变

原始 Transformer 的 source encoder 通常双向读取完整源序列，而自回归 decoder 通过 cross-attention 读取其输出。另一类教学设计可以让 encoder 本身也因果，然后让 decoder 读取这些因果编码状态。这样历史 K/V 不一定来自每个 decoder 层自己的过去隐藏状态，而可能来自一个独立的因果编码阶段。

本卷的 causal_encoder_decoder 原语实现两个连续的因果 attention 阶段，没有完整复刻任何具名模型的多层结构、训练或缓存共享。它的价值是让你明确 K/V 来源是架构选择，不是所有模型都必须沿用标准 decoder self-attention 的缓存假设。资源账本与 cache manager 应按真实来源关系设计。

```python
from labs.models.advanced import causal_encoder_decoder,conditional_ngram_memory

ced_x=rng.normal(size=(5,4))
ced_wq=rng.normal(0,.3,size=(4,4))
ced_wk=rng.normal(0,.3,size=(4,4))
ced_wv=rng.normal(0,.3,size=(4,4))
ced_decoded,ced_encoded=causal_encoder_decoder(ced_x,ced_wq,ced_wk,ced_wv)
ced_mask=np.tril(np.ones((5,5),dtype=bool))
manual_encoded=attention(ced_x@ced_wq,ced_x@ced_wk,ced_x@ced_wv,ced_mask)[0]
manual_decoded=attention(ced_x@ced_wq,manual_encoded@ced_wk,
                         manual_encoded@ced_wv,ced_mask)[0]
np.testing.assert_allclose(ced_encoded,manual_encoded)
np.testing.assert_allclose(ced_decoded,manual_decoded)
ced_changed=ced_x.copy();ced_changed[-1]+=50.
changed_decoded,changed_encoded=causal_encoder_decoder(
    ced_changed,ced_wq,ced_wk,ced_wv)
np.testing.assert_allclose(changed_encoded[:-1],ced_encoded[:-1],atol=1e-12)
np.testing.assert_allclose(changed_decoded[:-1],ced_decoded[:-1],atol=1e-12)
```

第一阶段因果，第二阶段也只读取不晚于当前位置的编码记录，因此组合仍因果。若第一阶段改成双向，即使第二阶段用下三角 mask，较早编码状态已经含有未来信息，仍会泄漏。这个逻辑与压缩摘要的因果边界相同：不能只检查最后一次读取 mask，还要检查被读取记录怎样生成。

是否能跨 decoder 层共享编码缓存，取决于各层是否使用同一投影、同一来源和同一位置规则。相同内容但不同 K/V 投影仍需要不同结果，除非有合法吸收或共享设计。不能从“存在 encoder”直接推断缓存一定只保存一份。读源码时应追踪实际张量生成与复用，而不是从架构名称估计。

### 十五点九　条件参数记忆与动态历史状态

还有一种机制根据当前 token 或短 n-gram 计算索引，读取一张参数表。它与 attention 的历史匹配不同：表可以是模型参数，索引由离散 token 条件生成；与 KV cache 也不同，表不一定随当前请求增长或更新。它可以提供某些模式的直接查表表示，但具体能力取决于训练、哈希与门控设计。

课程 conditional_ngram_memory 用确定的滚动哈希读取固定表，展示条件索引与碰撞。它没有学习表、没有多哈希消歧、没有完整门控，也不声称是某个具名条件记忆架构。你应先理解索引路径，再研究真实设计如何处理容量、冲突与训练。

```python
memory_table=np.arange(12,dtype=float).reshape(4,3)
memory_a,slots_a=conditional_ngram_memory([0,0],memory_table,n=2)
memory_b,slots_b=conditional_ngram_memory([1,3],memory_table,n=2)
assert slots_a[-1]==slots_b[-1]==2
np.testing.assert_array_equal(memory_a[-1],memory_b[-1])
repeat_a,repeat_slots=conditional_ngram_memory([2,1,2,1],memory_table,n=2)
assert repeat_slots[1]==repeat_slots[3]
np.testing.assert_array_equal(repeat_a[1],repeat_a[3])
```

两个不同 n-gram 落在同一槽位，只表示哈希碰撞，不表示它们语义相同。固定表返回同一向量可能是一种压缩代价，也可能由更复杂结构缓解。不能把碰撞样例解释成模型理解了同义关系。类似地，相同 n-gram 在不同上下文中读到同一静态表向量，也不意味着最终模型表示相同，因为其他上下文路径仍会参与。

“记忆”在模型讨论中至少有三种含义：训练后参数中储存的模式，当前请求的动态历史状态，以及外部检索系统的持久资料。它们的更新时机、权限和生命周期完全不同。明确分类可以避免把模型内部查表、KV cache、Agent 长期记忆和向量数据库混成一件事。

### 十五点十　独立工程的分阶段验收

第一阶段只实现深度聚合原语，包含正向、反向、零 query、value 幅度与轴隔离测试。把每个中间 shape 写在函数文档里，使用 N、T、D 不相等的样例。这个阶段完成后，你应能手算两三条来源，并解释 query 怎样从均匀读取开始学习。

第二阶段实现 Block 生命周期，不先接真实模型，用可控增量验证先读后封存、embedding 一次、新块起点与最终读出。随后接入真实 GQA/RoPE 与 SwiGLU，用本章 independent_attnres_execute 对照仓库模型。两阶段应保留不同测试，不把外部增量演示当作最终交付。

第三阶段完善状态与 trace。封存来源不可变，旧 trace 不受下一次运行影响；full 与 step 在零和非零 query 下都一致；不同块大小与不满最后块都正确；所有真实子层都能影响 logits。写一个诊断报告，列出每次读取来源标签、权重和 mixed 的摘要，便于读者沿深度追踪。

第四阶段加入可训练 query 实验，再在自动微分框架中接完整模型训练。先固定其他权重只训练 query，验证梯度；再允许子层参数更新，检查共享来源的梯度累加。不要直接跑大语料后用 loss 下降替代机制验证，因为很多错误架构也能在小数据上下降。

第五阶段研究混合与多流，分别实现逐层状态表和约束矩阵原语。每次只改变一个维度：时间算子、深度聚合、专家类型或流混合。若一次同时换掉多个机制，观察到的质量或速度变化很难归因。保留基线配置、随机种子、数据切分和训练预算，才能形成可解释比较。

### 十五点十一　版本规范：语义变化也要记版本

块大小的计数单位、封存时机、head 读出、query 初始化、key 归一化 epsilon、value 是否归一化，都是模型函数的一部分。它们可能不改变大部分权重 shape，却改变输出。checkpoint 应保存这些配置，实验报告应记录具体版本。仅凭“AttnRes 模型”这个名字不足以复现。

如果某个优化合并了来源、改变了快照存储或重排了读取顺序，应先证明数学等价，再比较数值。若它改变了最终读出或块内聚合，则应标成结构变体，不把差异藏在性能优化提交里。清楚的版本边界让后续扩展有依据，也帮助你从研究想法走向可维护工程。

混合模型的层类型列表同样需要保存。只保存参数数组而不保存某层应该调用 attention 还是 state，加载程序无法恢复原函数。状态 checkpoint 还应记录每层状态类型、dtype、逻辑长度或位置，以及对应模型权重版本。shape 相同不等于语义相同，这个原则在本章多个机制中反复出现。

### 十五点十二　错误注入比单次通过更有说服力

为每个关键不变量写一个明确故障：深度 softmax 改成时间轴，value 也做 norm，边界先清空，旧 partial 重复加入新块，embedding 重复封存，trace 保存共享引用，head 只读最后 partial，step 复用上一 token 的深度状态。每个故障应被至少一个针对性测试抓到，而不是希望随机 logits 比较碰巧发现。

例如来源计数能抓住部分重复，但抓不住旧块数值偷偷混进新 partial；零增量测试能发现输出闭合错误，但非零 query 的分数错误可能被均匀初始化掩盖；未来不干扰能发现时间混合，却不能证明深度值幅度正确。测试组合应覆盖不同故障机制，不能用一个大测试代替所有局部证据。

这种做法也能帮助你判断测试是否只是在复述实现。若参考函数与被测函数共享同一错误步骤，它们可能一起通过。加入独立手算、不变量与故障注入，能减少共同错误。模型结构学习的目标是建立独立判断能力，而不只是让两个同源代码输出相同。

### 十五点十三　哪些结论能从公式得到，哪些需要训练

从公式可以证明 softmax 权重和、归一化轴、块生命周期、因果性、缓存等价与某些约束矩阵性质；从小型数值实验可以验证实现与梯度；从训练与任务评估才能判断质量、泛化、长上下文能力或专家分工。三个证据层次都重要，但不能互相冒充。

随机权重完整模型证明机制确实接进了真实子层和缓存路径，比一个外部更新原语更完整；它仍不证明训练后能力。一个共享 query 小训练证明该参数能获得有效梯度并学习偏好；它仍不证明完整大模型的训练稳定性。准确陈述这些范围，会让后续扩展更扎实，而不是让成果显得更少。

阅读新论文时，先把已公开公式与代码对应，再看实验条件与消融。对尚未公开的结构保持未知，不用其他家族的实现补空白。你可以设计自己的假设性变体，但要明确它是实验提案。这样才能既紧跟新技术，又保持工程结论可验证。

### 十五点十四　深度 query 梯度的完整数值手算

只用一个 token、两条来源，分别为二、零与零、二，query 初始零。权重各二分之一，输出一、一。目标是二、零，损失取两个坐标平方误差的平均，得到一；上游输出梯度为负一、正一。第一权重的梯度是上游与第一来源点积，等于负二；第二为正二。

softmax 的加权平均梯度为零，所以两个分数梯度分别负一、正一。每条来源的 RMS 半径为根号二加 epsilon 的相应平方根，归一化 key 近似根号二、零与零、根号二。query 梯度因此近似负根号二、正根号二。梯度下降使 query 第一坐标增加、第二坐标减少，下一次读取会更偏向第一来源，方向与目标一致。

此时来源的 key 路径梯度为零，因为分数对 key 的导数含零 query；value 路径则让每条来源都收到二分之一乘负一、正一。也就是说，零初始化时仍可以同时学习 query 与来源，只是来源通过匹配路径的梯度暂时为零。这个细节解释了为什么“均匀初始化”不等于“冻结深度选择”。

```python
gradient_hand_sources=np.array([[[2.,0.]],[[0.,2.]]])
gradient_hand_query=np.zeros(2)
gradient_hand_output,_,_=depth_attention_reference(gradient_hand_sources,gradient_hand_query)
gradient_hand_upstream=gradient_hand_output-np.array([[2.,0.]])
source_gradient,query_gradient=depth_attention_backward(
    gradient_hand_sources,gradient_hand_query,gradient_hand_upstream)
radius=np.sqrt(2.+1e-6)
np.testing.assert_allclose(query_gradient,[-2/radius,2/radius])
np.testing.assert_allclose(source_gradient,
                           [[[-.5,.5]],[[-.5,.5]]])
```

如果把损失定义成平方误差之和而不是平均，上游梯度会多一个二的因子，后续所有梯度也相应变化。比较手写梯度时先核对损失归约与系数，不能把纯粹的尺度约定差异误判成链式法则错误。训练学习率也会受这个尺度影响。

### 十五点十五　双随机混合能证明什么范数性质

对非负行随机 H，每个输出流的某个坐标是输入坐标的凸组合。平方函数是凸的，因此这个输出坐标平方不超过输入坐标平方的同权加权和。再对所有输出流求和，若 H 的列和也为一，每个输入平方恰好累计一次，所以纯混合后的平方和不超过原平方和。对所有特征坐标求和，得到 Frobenius 范数不增。

这个证明要求 H 非负、行列和都为一；有限 Sinkhorn 只是近似满足，所以数值测试应带适当容差。它约束的是 H 乘 X 这条纯残差混合，不包括 post 写入或非线性子层。把这个局部不增性质扩展成整个深层网络必然不爆炸，是超出证明的结论。

```python
mix_logits=rng.normal(size=(3,3))
mix_matrix=sinkhorn(mix_logits,iterations=200)
mix_input=rng.normal(size=(3,5))
assert np.linalg.norm(mix_matrix@mix_input)<=np.linalg.norm(mix_input)+1e-10
np.testing.assert_allclose((mix_matrix@mix_input).sum(0),mix_input.sum(0),atol=1e-10)
```

单位矩阵使范数保持，均匀平均通常缩小流之间差异。约束矩阵并不是简单的“无损传递器”，它可以收缩某些方向。学习系统可能利用这种收缩，也可能需要其他支路保留信息。理解具体不变量与收缩方向，比笼统地说“约束改善稳定性”更能指导实验。

### 十五点十六　来源读取成本的具体估算

假设四十八层，每层两个子层，隐藏维四千零九十六，序列长度一千零二十四，块大小八个子层。最终 head 来源数为 embedding 加十二个计算块，共十三条；每条来源有四百一十九万四千三百零四个元素。若两字节存储，最终这批来源本身约一百零四 MiB，还未计算子层激活、梯度或调试副本。

这个数字不是整个训练峰值，也不是官方实现的实测，因为优化实现可能重用缓冲、分块重计算或采用不同布局。它只是根据教学来源定义的元素账本，提醒你新增 query 参数很少并不代表来源读取免费。深度增加时，每个后续子层还要读已有来源，累计带宽可能成为重要因素。

逐 token decode 时 T 等于一，同样来源只需很少元素，但每层时间 K/V 仍随已处理历史增长。若混合模型某些层改成固定状态，这部分又会改变。容量报告应分别列深度临时来源、时间持久状态和模型参数，避免把一种机制的节省用来掩盖另一种机制的开销。

### 十五点十七　结业说明应能让另一位学习者复现

提交本章项目时，附一个最小命令运行完整模型、一个命令运行测试，以及固定配置下的来源计数与误差报告。说明 Python 与 NumPy 版本、是否使用随机权重、是否训练、是否运行 GPU。若只有 CPU 结构实验，就如实写 CPU；这已经是完整机制验证，不需要借用未执行的大模型部署来增加分量。

代码应保留清楚的模块边界：深度聚合原语、块状态机、真实子层、时间缓存、混合层状态和多流约束工具。后续扩展某个模块时，其他模块的测试应继续通过。不要把全部逻辑压进一个难以定位的大函数，也不要只留下调用官方接口的包装而丢掉本章独立实现。

你现在应能把一个新架构图拆成明确的问题：它沿哪个轴选择，保存哪些状态，在哪里归一化，何时封存或释放，哪些操作可交换，哪些梯度需要累加。无论之后研究新的残差机制、混合状态层还是条件记忆，这些问题都能帮助你从名词走到可运行、可证明的工程。

### 十五点十八　一个容易误读的归一化不变量

对同一个 token 的所有深度分数加同一个常数，softmax 权重不变；但给所有来源向量加同一个偏移，不一定只让输出加这个偏移，因为 RMSNorm key 与分数也会改变。若权重固定，value 平移有简单关系；若权重由来源内容生成，就必须同时考虑匹配路径。这是解释消融实验时常见的混淆。

同样，把某来源乘负一不仅改变 value，也翻转归一化 key 方向，不能像正尺度放大那样近似认为分数不变。极小来源受 epsilon 影响，零来源 key 为零，query 再大也不会从这条 key 得到非零分数，但它仍以零 value 参与归一化竞争。若直接删除零来源，其余权重会重新分配，输出可能改变。

因此零增量块不是自动可以从 completed 列表删除的“无用数据”。在本章定义中，它仍占一个候选位置，零 query 下会改变均匀平均的分母。第九课最终 E 除三正是这个现象。若优化器想跳过全零块，需要同时调整归一化以保持原函数，或者明确把删除定义成另一个变体。数值为零不意味着在所有非线性结构中都可以无条件消去。

训练或调试时应把来源标签与权重一起记录。单独一列权重无法告诉你它对应 embedding、完整块还是当前 partial；块大小改变后，同一列编号也可能代表不同深度范围。比较两个实验时先对齐来源语义，再讨论偏好变化，避免把数组位置相同误当成模型组件相同。

来源语义必须随配置一同保存。
