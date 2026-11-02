# M07　低秩、矩阵吸收与 MLA：把历史压缩成可读取的状态

[本卷导航](README.md) · 前置：[KV Cache 与 GQA](06-kv-gqa.md) · 下一章：[混合专家](08-moe.md)

上一章通过减少 KV 头数节省缓存，但每个历史位置仍保存一组键和一组值。本章继续追问：如果键和值都由一个较小的共同向量生成，能否只保存那个向量？如果可以，读取时是否必须把全部历史重新展开？这两个问题分别对应低维状态设计和矩阵吸收。把它们混在一句“压缩 KV”里，会漏掉真正重要的数学和工程选择。

本章先建立完全可运行的无位置低秩 attention，再加入独立位置分支，最后写多头与缓存版本。它是为阅读 MLA 准备的精确教学模型，不宣称复刻某个完整旗舰模型的全部参数、归一化、训练与内核。架构概念依据 [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)；具体版本的配置应以本卷 [来源表](SOURCES.md) 和官方代码为准。

## 第一课　秩：一个矩阵究竟有多少独立方向

先看矩阵 $\bigl[\begin{smallmatrix}1&2\\2&4\end{smallmatrix}\bigr]$。第二行是第一行的两倍，第二列也是第一列的两倍，因此它虽然存了四个数，独立变化方向只有一个。它可以写成列向量 $(1,2)^\top$ 与行向量 $(1,2)$ 的外积。输入向量经过这个矩阵后，输出总落在同一条直线上；矩阵的秩为一。

再看单位矩阵。它把 $(1,0)$ 和 $(0,1)$ 分别保持为两个独立方向，不能用一个外积精确表示。一个外积的所有列都是同一个列向量的倍数，最多只有一个独立列，所以无法产生两个独立方向。这个反例说明，把中间维度减到一不是一种对任意矩阵都无损的存储技巧，而是对函数表达空间的限制。

一般地，若 $W=AB$，其中 A 的 shape 为 $[D,r]$，B 为 $[r,F]$，那么 W 的秩不超过 r。每个输入先被压到 r 个中间坐标，再由这些坐标组合出 F 个输出。参数数量从 DF 变成 $Dr+rF$。只有当 r 足够小，这个数量才更少；当 D 和 F 都为十六、r 为十二时，原矩阵二百五十六个参数，分解后却有三百八十四个，不能看到“低秩”三个字就断言参数节省。

这里有两种不同实验。第一种从模型定义开始就使用 A 和 B，训练在这个函数族中进行；第二种先有一个任意训练好的 W，再找低秩近似。前者没有“近似原矩阵”的问题，因为它本来就定义为乘积；后者通常有重建误差。后面 attention 的精确吸收建立在 K/V 原本由 latent 生成这一条件上，不意味着任何现成 KV 都能任意压缩而不损失信息。

```python
import numpy as np
left=np.array([[1.],[2.]])
right=np.array([[1.,2.]])
rank_one=left@right
np.testing.assert_array_equal(rank_one,[[1.,2.],[2.,4.]])
assert np.linalg.matrix_rank(rank_one)==1
assert np.linalg.matrix_rank(np.eye(2))==2
assert 16*12+12*16>16*16
```

秩还有一个容易误读的地方：它描述线性关系，不等同于语义信息量的一个简单整数。一个低维连续向量能以实数精度表达很多变化，但读取方式受限；一个高维矩阵也可能在实际数据上只占据低维流形。模型设计利用这种结构可能有效，却需要训练和评估证明。我们在本章能严格证明的是代数关系与存储数量，不能仅凭秩推导自然语言质量。

## 第二课　共同 latent：先定义模型，再讨论优化

设输入隐藏状态 $X$ 的 shape 为 $[T,D]$。通过下投影 $W_D\in\mathbb R^{D\times r}$ 得到压缩状态 $C=XW_D$，shape 为 $[T,r]$。每一行 $c_t$ 对应一个历史 token 的低维内容表示。接着定义键上投影 $U_K\in\mathbb R^{r\times d_k}$、值上投影 $U_V\in\mathbb R^{r\times d_v}$，得到 $K=CU_K$ 与 $V=CU_V$。

查询 Q 的 shape 为 $[S,d_k]$，S 是当前查询位置数。完整 attention 分数为 $QK^\top/\sqrt{d_k}$，shape 为 $[S,T]$，经过 mask 和逐行 softmax 得到 P，再计算 $PV$，shape 为 $[S,d_v]$。这和你已经实现的 attention 完全一样，区别只在 K/V 的来源受到共同 latent 的约束。

共同 latent 不是把 K 和 V 简单拼接后切一半。它是一个能同时生成两者的基础表示；不同上投影可以从同一个 C 读出不同特征。也不能把它理解为先计算自由 K/V 再平均，因为那样未必存在同样的线性重建关系。构建模型时必须明确 C 从哪里来、是否归一化、每个头是否共享，以及位置部分是否另存。

为什么共享 C 可能减少状态？显式 K/V 每 token 保存 $d_k+d_v$ 个数；只保存 C 则是 r 个。若 r 小于两者之和，就减少持久状态。但上投影权重仍然属于模型参数，不能从账本中消失。缓存减少的是每个请求随长度增长的部分，而不是把固定权重也一起删除。

还要区分 query 压缩和 KV 压缩。某些架构也把查询生成过程写成低秩下投影加上投影，这可能减少参数或改变计算，但查询通常只在当前步使用，不像历史 KV 那样持续积累。因此它对持久缓存容量的作用不同。阅读真实 MLA 时应分别画出 query 路径和 KV 路径，不能把所有低秩维度都叫成同一个 rank。

## 第三课　两次矩阵吸收：softmax 留在原来的位置

由 $K=CU_K$，分数的未缩放部分为
$$
QK^\top=Q(CU_K)^\top=QU_K^\top C^\top.
$$
先算 $Q'=QU_K^\top$，shape 为 $[S,r]$，再算 $Q'C^\top$，即可得到同样的 $[S,T]$ 分数。这个变换只用了转置乘积规则和矩阵乘法结合律，没有交换矩阵顺序。$QU_K^\top$ 合法，因为 Q 最后一维是 $d_k$，而 $U_K^\top$ 第一维也是 $d_k$。

得到相同分数后，使用相同 mask、缩放和 softmax，得到相同概率 P。值路径有
$$
PV=P(CU_V)=(PC)U_V.
$$
先读取 latent，得到 $Z=PC$，shape 为 $[S,r]$，再上投影成 $[S,d_v]$。因此不必为全部历史物化完整 V，只需在当前输出上使用上投影。

softmax 没有被移过矩阵。一般不存在 $\operatorname{softmax}(AB)=\operatorname{softmax}(A)\operatorname{softmax}(B)$ 这样的规则；其非线性和归一化轴都会破坏这种写法。可以重新结合的是同一线性乘法链里的矩阵，不能越过激活、归一化、裁剪或依赖数据的选择操作。以后遇到“权重吸收”时，先把中间所有非线性标出来，才能判断合法边界。

缩放仍然是 $\sqrt{d_k}$，不是 $\sqrt r$。原函数定义按原始键空间维度缩放；虽然计算过程中出现 r 维 query，代数变换没有授权修改温度。若 $d_k=8,r=2$，改用 $\sqrt2$ 会把分数放大两倍，使概率通常更尖锐。输出可能仍然有限、shape 完全正确，但已经不是同一个函数。

```python
from labs.models.core import softmax

def explicit_latent_attention(q,c,uk,uv,mask=None):
    keys=c@uk
    values=c@uv
    scores=q@keys.T/np.sqrt(q.shape[-1])
    if mask is not None:
        scores=np.where(mask,scores,-np.inf)
    probabilities=softmax(scores)
    return probabilities@values,scores,probabilities

def absorbed_latent_attention(q,c,uk,uv,mask=None):
    latent_query=q@uk.T
    scores=latent_query@c.T/np.sqrt(q.shape[-1])
    if mask is not None:
        scores=np.where(mask,scores,-np.inf)
    probabilities=softmax(scores)
    latent_read=probabilities@c
    return latent_read@uv,scores,probabilities

rng=np.random.default_rng(7)
q=rng.normal(size=(4,6))
c=rng.normal(size=(4,2))
uk=rng.normal(size=(2,6))
uv=rng.normal(size=(2,3))
causal=np.tril(np.ones((4,4),dtype=bool))
explicit,score_a,prob_a=explicit_latent_attention(q,c,uk,uv,causal)
absorbed,score_b,prob_b=absorbed_latent_attention(q,c,uk,uv,causal)
np.testing.assert_allclose(score_a,score_b,atol=1e-12)
np.testing.assert_allclose(prob_a,prob_b,atol=1e-12)
np.testing.assert_allclose(explicit,absorbed,atol=1e-12)
```

代码返回分数和概率是为了定位误差。若输出不一致，先比较分数；分数已不同，问题在键路径、缩放或 mask；分数一致而概率不同，问题在 softmax；概率一致而输出不同，问题在值路径或上投影。只比较最终输出会让多个错误互相抵消，尤其在零参数或对称输入中更容易产生假通过。

## 第四课　完整手算：从两维 latent 到四维键

选两个历史位置，latent 为 $C=\bigl[\begin{smallmatrix}1&0\\0&2\end{smallmatrix}\bigr]$。键上投影为
$$
U_K=\begin{bmatrix}1&0&1&0\\0&1&0&1\end{bmatrix},
$$
值上投影为 $U_V=\bigl[\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\bigr]$。当前查询 $q=(2,0,0,0)$，原键维度四，所以缩放分母为二。先显式展开：第一条键为 $(1,0,1,0)$，第二条键为 $(0,2,0,2)$；第一条值为 $(1,2)$，第二条值为 $(6,8)$。

查询与两条键的点积分别二和零，除以二得到分数 $(1,0)$。概率为 $(e/(e+1),1/(e+1))$，约 $(0.731059,0.268941)$。输出第一分量为 $0.731059\times1+0.268941\times6\approx2.344707$，第二分量为 $0.731059\times2+0.268941\times8\approx3.613649$。

现在走吸收路径。$qU_K^\top=(2,0)$。它与 C 两行的点积仍是二和零，除以原分母二，得到相同分数。先读取 latent：$PC=(0.731059,0.537883)$，第二分量来自第二条 latent 的二乘概率零点二六八九四一。再乘 $U_V$，第一输出 $0.731059+3\times0.537883=2.344707$，第二输出 $2\times0.731059+4\times0.537883=3.613649$。

每个中间量都有明确意义。latent_read 的第二分量不是概率，它可以超过某个概率，甚至在其他输入中为负或大于一，因为它是对一般特征的加权和。不要把 attention 输出中间向量都解释成概率；只有 softmax 的那一层满足非负和逐行和为一的约束。

```python
c_hand=np.array([[1.,0.],[0.,2.]])
uk_hand=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])
uv_hand=np.array([[1.,2.],[3.,4.]])
q_hand=np.array([[2.,0.,0.,0.]])
out_hand,scores_hand,p_hand=absorbed_latent_attention(q_hand,c_hand,uk_hand,uv_hand)
np.testing.assert_allclose(scores_hand,[[1.,0.]])
np.testing.assert_allclose(p_hand,[[np.e/(np.e+1),1/(np.e+1)]])
latent_read_hand=p_hand@c_hand
np.testing.assert_allclose(latent_read_hand,[[np.e/(np.e+1),2/(np.e+1)]])
np.testing.assert_allclose(out_hand,
    [[(np.e+6)/(np.e+1),(2*np.e+8)/(np.e+1)]])
```

若错误除以 $\sqrt2$，分数变为 $(\sqrt2,0)$，第一概率约零点八零四四，比原来的零点七三一更偏向第一条值。因此输出更靠近 $(1,2)$。这个错误不能通过简单给最终输出乘一个固定常数修复，因为 softmax 改变的是不同历史位置之间的相对权重，不是统一缩放。

## 第五课　多头吸收与输出投影：不要把不同头混在一起

多头情况下，假设所有头共享同一个 C，但每个头有自己的 $U_{K,h}$ 和 $U_{V,h}$。查询 $Q_h$ 产生各自分数 $S_h=(Q_hU_{K,h}^\top)C^\top/\sqrt{d_k}$，再得到各自概率 $P_h$。虽然 latent 相同，不同查询与上投影让读取策略不同。每头先得到 $Z_h=P_hC$，然后输出 $O_h=Z_hU_{V,h}$。

把各头输出拼接后乘整体输出矩阵 $W_O$。若将 $W_O$ 按输入头块拆成 $W_{O,h}\in\mathbb R^{d_v\times D}$，则
$$
\operatorname{Concat}(O_1,\ldots,O_H)W_O
=\sum_h O_hW_{O,h}
=\sum_h Z_h(U_{V,h}W_{O,h}).
$$
因此可以预先把每头值上投影与对应输出块相乘，得到 $[r,D]$ 的吸收权重。这里出现求和，是因为拼接向量乘一个分块矩阵等于各块贡献相加。

不能先把不同头的概率平均后只读一次 C，再期待得到同样结果。不同头后面的上投影与输出块不同，一般无法交换这种合并。只有在非常特殊的共享权重条件下才可能化简。多头吸收减少的是某些中间物化与线性链，不会自动删除头的独立读取。

```python
def multihead_latent(q,c,uk,uv,wo,absorbed_output):
    heads,queries,key_dim=q.shape
    value_dim=uv.shape[-1]
    reads=[]
    probs=[]
    for h in range(heads):
        scores=(q[h]@uk[h].T)@c.T/np.sqrt(key_dim)
        p=softmax(scores)
        probs.append(p)
        reads.append(p@c)
    if absorbed_output:
        result=np.zeros((queries,wo.shape[-1]))
        for h,z in enumerate(reads):
            block=wo[h*value_dim:(h+1)*value_dim]
            result+=z@(uv[h]@block)
    else:
        head_outputs=[z@uv[h] for h,z in enumerate(reads)]
        result=np.concatenate(head_outputs,axis=-1)@wo
    return result,np.stack(probs)

q_multi=rng.normal(size=(3,2,4))
c_multi=rng.normal(size=(5,2))
uk_multi=rng.normal(size=(3,2,4))
uv_multi=rng.normal(size=(3,2,3))
wo_multi=rng.normal(size=(9,6))
out_direct,p_direct=multihead_latent(q_multi,c_multi,uk_multi,uv_multi,wo_multi,False)
out_absorb,p_absorb=multihead_latent(q_multi,c_multi,uk_multi,uv_multi,wo_multi,True)
np.testing.assert_allclose(out_direct,out_absorb,atol=1e-12)
np.testing.assert_array_equal(p_direct,p_absorb)
```

权重预合并适合固定权重推理。如果参数仍在训练中，每次更新都改变乘积，不能把旧乘积当作永久常量。若需要对原始因子分别求梯度，也要保留它们的参数化关系。部署时将两个线性层合成一个矩阵，改变的是执行形式；训练优化两个因子与直接优化乘积矩阵，则是不同的优化问题。

## 第六课　位置为什么阻碍朴素吸收

前面的键是 $c_tU_K$，上投影对所有位置相同。如果对整个展开键施加位置相关旋转 $R_t$，键变成 $(c_tU_K)R_t$。查询与键的点积中出现 $U_KR_t$，它随历史位置 t 变化。你不能把一个固定的 $U_K^\top$ 吸收到当前查询后，就对所有 t 使用同一个变换，因为各历史位置的旋转不同。

用行向量约定写得更清楚：$q[(c_tU_K)R_t]^\top=qR_t^\top U_K^\top c_t^\top$。右侧的 query 变换包含 $R_t^\top$，依赖正在比较的历史位置。因此无法像无位置情况那样只预先算一次 $qU_K^\top$。即使查询也有自己的旋转 $R_s$，相对旋转仍依赖 s 与 t 的差，不会一般性地消失。

有些矩阵与特定旋转可能交换，但这是额外结构条件，不能默认成立。维度不同的时候，“把旋转挪到 latent 里面”甚至没有合法 shape。原键维度可能一百二十八，latent 维度五百一十二或更小；两者旋转矩阵并不是同一个对象。正确推导必须逐项检查线性映射的定义域与值域。

解决思路之一是把内容分数与位置分数分开。内容键保持可吸收的低秩形式，位置键使用独立的小分支并保存。总分数由两部分相加，softmax 仍只做一次。这样既保留内容读取的压缩机会，也保留位置关系。它不是把两个 attention 输出简单相加，因为“先加分数再 softmax”与“分别 softmax 再加输出”一般不同。

这个区分是理解真实 MLA 的核心。你应当在官方公式中分别识别内容查询、位置查询、内容键、位置键、共同 KV latent 与值上投影。不同模型可能有不同归一化和维度，不能用一个泛化图代替具体配置。本章下一课写一个完整的解耦分支教学实现，让这些名词对应实际张量。

## 第七课　独立位置分支：一个完整可比较的教学模型

定义内容查询 $Q^C\in\mathbb R^{H\times S\times d_c}$，位置查询 $Q^R\in\mathbb R^{H\times S\times d_r}$，共享内容 latent $C\in\mathbb R^{T\times r}$，共享位置键 $K^R\in\mathbb R^{T\times d_r}$。每头内容键为 $K_h^C=CU_{K,h}$。位置查询与位置键已经在各自逻辑位置施加 RoPE，本函数只接收旋转后的结果。

总分数定义为
$$
S_h=\frac{Q_h^C(K_h^C)^\top+Q_h^R(K^R)^\top}{\sqrt{d_c+d_r}}.
$$
分母使用拼接后总键维度，因为这个教学定义等价于把内容与位置部分拼接后做普通缩放点积。真实模型可能规定额外 scale 或位置缩放，加载时应使用其定义，不要擅自从本例复制常数。mask 加到总分数上，再做一次 softmax，得到 $P_h$。

值只来自内容 latent：$V_h=CU_{V,h}$。位置分支参与选择读哪里，不在本例中直接成为 value。显式路径把内容键和值全部展开；吸收路径计算 $(Q_h^CU_{K,h}^\top)C^\top$，加上相同位置分数，再通过 $(P_hC)U_{V,h}$ 形成输出。两条路径应完全等价，缓存对象则是 C 与 $K^R$。

```python
def decoupled_attention(q_content,q_position,c,k_position,uk,uv,mask=None,
                        use_absorption=True):
    heads,queries,content_dim=q_content.shape
    position_dim=q_position.shape[-1]
    scale=np.sqrt(content_dim+position_dim)
    outputs=[]
    probabilities=[]
    for h in range(heads):
        if use_absorption:
            content_scores=(q_content[h]@uk[h].T)@c.T
        else:
            content_keys=c@uk[h]
            content_scores=q_content[h]@content_keys.T
        position_scores=q_position[h]@k_position.T
        scores=(content_scores+position_scores)/scale
        if mask is not None:
            scores=np.where(mask,scores,-np.inf)
        p=softmax(scores)
        if use_absorption:
            out=(p@c)@uv[h]
        else:
            out=p@(c@uv[h])
        outputs.append(out)
        probabilities.append(p)
    return np.stack(outputs),np.stack(probabilities)

from labs.models.core import rope
head_count,length,rank,content_dim,position_dim,value_dim=2,5,3,4,2,3
positions=np.arange(length)
content=rng.normal(size=(length,rank))
q_content=rng.normal(size=(head_count,length,content_dim))
q_position=rope(rng.normal(size=(head_count,length,position_dim)),positions)
k_position=rope(rng.normal(size=(1,length,position_dim)),positions)[0]
uk_dec=rng.normal(size=(head_count,rank,content_dim))
uv_dec=rng.normal(size=(head_count,rank,value_dim))
mask_dec=np.tril(np.ones((length,length),dtype=bool))
expanded_dec,p_expanded=decoupled_attention(q_content,q_position,content,k_position,
    uk_dec,uv_dec,mask_dec,False)
absorbed_dec,p_absorbed=decoupled_attention(q_content,q_position,content,k_position,
    uk_dec,uv_dec,mask_dec,True)
np.testing.assert_allclose(expanded_dec,absorbed_dec,atol=1e-12)
np.testing.assert_allclose(p_expanded,p_absorbed,atol=1e-12)
```

逐行对应公式：content_scores 是内容相似度，position_scores 是位置分支点积，二者相加后才除以共同 scale。mask 屏蔽的是完整连接，不是只屏蔽内容项；如果只把内容项置零而保留位置项，被禁止位置仍可能获得概率。softmax 之后，概率用于读取 C，再通过值上投影产生每头输出。这里没有让位置键进入 value，因此缓存位置分支的用途非常明确。

这个实现可以被称为“带独立位置分支的低秩多头注意力教学模型”，但不能直接宣称是完整 DeepSeek 模型。完整架构还涉及查询低秩路径、归一化、具体头维度、输出投影、层间结构、训练配置和部署优化。本章已经实现核心代数和状态，不把这些额外部分凭空猜成已完成。读源码时应在已验证机制上逐项增加实际差异。

## 第八课　分支分数手算：位置如何改变读取但不改变值

构造最简单的两个历史位置。内容分数未缩放为 $(1,0)$，位置分数为 $(-1,2)$，总分数未缩放为 $(0,2)$。若总维度四，分母二，最终分数为 $(0,1)$，概率为 $(1/(1+e),e/(1+e))$。即使内容分支偏向第一个位置，位置分支也能使总读取偏向第二个位置。这不是两个独立注意力结果的投票，而是同一分布的两类证据。

令两个 value 为一和五，最终输出为 $(1+5e)/(1+e)\approx3.924234$。如果分别对内容分数和位置分数做 softmax，再平均两个输出，会得到另一个数。原因是指数作用于分数之和，相当于未归一化权重相乘后再统一归一化；分别归一化后平均不是同一运算。这个差别在实现时很容易被“分支”这个词误导。

再观察位置分支为零的退化情况。总分数仍需使用本模型定义的 $\sqrt{d_c+d_r}$ 分母，除非你同时修改模型定义。不能因为某次输入的位置分量恰好全零，就临时把分母改成 $\sqrt{d_c}$。函数的尺度规则是架构的一部分，不随某个中间向量是否为零随意改变。

```python
content_scores=np.array([[1.,0.]])
position_scores=np.array([[-1.,2.]])
combined=softmax((content_scores+position_scores)/2.)
values=np.array([[1.],[5.]])
combined_output=combined@values
separate_output=(softmax(content_scores/2.)@values
                 +softmax(position_scores/2.)@values)/2.
np.testing.assert_allclose(combined,[[1/(1+np.e),np.e/(1+np.e)]])
np.testing.assert_allclose(combined_output,[[(1+5*np.e)/(1+np.e)]])
assert np.max(np.abs(combined_output-separate_output))>1e-3
```

位置分支并不意味着模型只能使用位置而不能使用内容。训练会调整两个分支的投影，使总分数承担任务需要的选择。也不能从某一条样本的位置分数较大，就推断模型整体“忽略语义”。解释模型行为需要多个输入和更谨慎的实验，本章只要求你准确追踪分数如何合成。

## 第九课　低维缓存的逐步实现与严格等价

缓存应保存 C 和已经旋转的位置键。当前 token 产生自己的内容查询、位置查询、新 latent 和新位置键，然后追加两种历史状态。注意内容 C 不能再乘一次下投影；它已经是模型定义的 latent。位置键也不能再次 RoPE。两种状态的时间长度必须同步，否则同一列的内容与位置会指向不同 token。

下面不引入整个解码器，而构造完整注意力层的全序列与逐步对照。输入 X 通过固定权重生成 latent、内容查询、位置查询和位置键，各头上投影固定。完整路径使用因果 mask；逐步路径每次只取当前查询，并读取前缀缓存。两者对每个头每个位置的输出逐项比较。

```python
D,H,R,DC,DR,DV,T=6,2,3,4,2,3,6
x=rng.normal(size=(T,D))
wd=rng.normal(size=(D,R))
wqc=rng.normal(size=(D,H*DC))
wqr=rng.normal(size=(D,H*DR))
wkr=rng.normal(size=(D,DR))
uk_step=rng.normal(size=(H,R,DC))
uv_step=rng.normal(size=(H,R,DV))

def project_all(inputs,offset=0):
    n=len(inputs)
    local_heads=uk_step.shape[0]
    pos=np.arange(offset,offset+n)
    latent=inputs@wd
    qc=(inputs@wqc).reshape(n,local_heads,DC).transpose(1,0,2)
    qr=(inputs@wqr).reshape(n,local_heads,DR).transpose(1,0,2)
    kr=(inputs@wkr)[None,:,:]
    return latent,qc,rope(qr,pos),rope(kr,pos)[0]

all_c,all_qc,all_qr,all_kr=project_all(x)
all_out,_=decoupled_attention(all_qc,all_qr,all_c,all_kr,
    uk_step,uv_step,np.tril(np.ones((T,T),dtype=bool)),True)

cache_c=np.empty((0,R))
cache_kr=np.empty((0,DR))
step_outputs=[]
for t in range(T):
    new_c,new_qc,new_qr,new_kr=project_all(x[t:t+1],t)
    cache_c=np.concatenate([cache_c,new_c],axis=0)
    cache_kr=np.concatenate([cache_kr,new_kr],axis=0)
    step_out,_=decoupled_attention(new_qc,new_qr,cache_c,cache_kr,
                                  uk_step,uv_step,None,True)
    step_outputs.append(step_out)
np.testing.assert_allclose(np.concatenate(step_outputs,axis=1),all_out,atol=1e-10)
assert cache_c.shape==(T,R)
assert cache_kr.shape==(T,DR)
```

为什么逐步路径没有传 mask？因为缓存只包含过去与当前，没有未来列，所有已有键都合法。若一次追加多个 token，就必须恢复矩形因果 mask；不能因为使用 cache 就认为 mask 永远可以省略。这个规则与上一章完全一致，变化的只是保存对象的特征轴。

这段代码保存完整历史 latent，因此状态仍随 T 线性增长。它不是下一章线性注意力那种固定大小的递推状态。虽然 query 与 latent 的内积维度可能较小，每个新 token 仍要与历史各位置比较，softmax 仍在历史轴上归一化。把“低维缓存”称为“常数内存注意力”是不正确的。

## 第十课　成本账本：节省容量不等于所有路径更快

假设有 H 个头，每头内容键维度 dc、值维度 dv，共享位置键维度 dr，共享 latent 维度 r。若显式保存每头内容 K/V，加上共享位置键，每 token 元素数为 $H(dc+dv)+dr$。保存 latent 与位置键则为 $r+dr$。这个公式清楚显示哪些项共享，哪些项随头数增长。若某模型的位置键也按头独立保存，应相应修改，不能照抄。

取 H 为八、dc 与 dv 都为六十四、r 为一百二十八、dr 为三十二。显式状态每 token 为 $8\times128+32=1056$ 个元素；低维状态为一百六十个，元素比例六点六。长度四千零九十六、两字节类型时，单层显式状态为 $1056\times4096\times2=8650752$ 字节，低维为 $160\times4096\times2=1310720$ 字节。这里只算一层一个请求，不含页尾、scale、临时工作区或权重。

计算成本需要另列。显式路径可以先把历史 C 展开成 K/V，再做高效 attention；吸收路径每个查询头要把 query 投影到 r 维，与 C 做分数，并把读取的 r 维结果映射到输出。如果 r 大于某些原头维度，某些矩阵乘法可能变多；如果减少了大量历史展开和读取，又可能整体获益。性能取决于 query 数、历史长度、头数、硬件和融合方式，不是只由压缩比决定。

Prefill 有很多查询，常规展开后的大矩阵可能更容易利用硬件；decode 查询数少，持久缓存读取更突出，吸收路径可能更有价值。实际实现可以为两阶段选择不同计算路线，只要在同一参数下结果符合定义。一个模型拥有低维缓存，并不意味着每个阶段都必须使用完全相同的矩阵结合顺序。

```python
H,dc,dv,r,dr,T,bytes_each=8,64,64,128,32,4096,2
explicit_elements=H*(dc+dv)+dr
latent_elements=r+dr
assert explicit_elements==1056
assert latent_elements==160
assert explicit_elements*T*bytes_each==8650752
assert latent_elements*T*bytes_each==1310720
np.testing.assert_allclose(explicit_elements/latent_elements,6.6)
```

还要检查临时展开是否抵消峰值节省。如果持久 cache 只有 C，但每次 decode 都把全部历史 K/V 展开到临时大数组，峰值显存和带宽仍可能很高。它在语义上是低维缓存，在执行上却可能没有充分利用吸收。报告时应分别记录持久状态、瞬时工作区和峰值总量，不能只展示最小的那一项。

## 第十一课　低秩、量化、LoRA 与线性注意力的边界

低秩改变特征表示所需的独立方向数量；量化减少每个数的编码位数。两者可以组合，例如先设计 latent，再把 latent 以低比特保存，但误差来源不同。低秩模型若原生如此定义，吸收可保持精确函数；量化通常把连续值近似成离散等级，改变数值。验收时应先在高精度下证明代数等价，再单独评估量化误差。

LoRA 常写 $W'=W+\alpha AB/r$。它保留原矩阵 W，只训练一个低秩增量。减少的是需要更新和保存的适配参数，不自动把历史 K/V 改成共同 latent。若把 LoRA 应用于 K/V 投影，最终投影权重改变，缓存仍按照该模型的状态契约生成。不能看到两个小矩阵相乘，就把 LoRA 与 MLA 当作同一个机制。

线性注意力则通常改变 attention kernel，使历史能汇总成固定维度的矩阵与向量状态。它不再为每个历史 token 保存一行 latent，也不必对所有历史位置显式 softmax。低维 softmax attention 与线性 attention 的时间复杂度、状态大小和表达方式不同。它们都可能被宣传为长上下文优化，但理解时应回到实际公式。

事后低秩近似还需要考虑误差如何经过 softmax 放大。键重建误差导致分数误差，分数差异可能改变概率，进而改变对 value 的读取。即便 K/V 的均方重建误差很小，某些接近决策边界的查询也可能有明显输出变化。因此只报告矩阵重建误差不足以证明任务质量，应同时检查 attention 输出、logits 与任务指标。

一个严谨实验应包含三层对照：第一，原生低秩模型显式与吸收路径的数值一致性；第二，对自由矩阵做低秩近似后的重建与输出误差；第三，训练或微调后的任务质量变化。第一层是实现正确性，第二层是近似分析，第三层是模型能力。混淆三者会让“精确等价”变成一个超出证据的口号。

## 第十二课　十二道练习与可运行参考答案

### 练习一：外积与秩

用列向量二、三与行向量一、四生成矩阵，证明非零结果秩一。答案为 $\bigl[\begin{smallmatrix}2&8\\3&12\end{smallmatrix}\bigr]$，第二列是第一列四倍，所有列落在同一方向。下面同时检查重建和值域维度。若两个向量有一个全零，结果秩为零，所以“外积总是秩一”也需要非零条件。

```python
outer=np.array([[2.],[3.]])@np.array([[1.,4.]])
np.testing.assert_array_equal(outer,[[2.,8.],[3.,12.]])
assert np.linalg.matrix_rank(outer)==1
assert np.linalg.matrix_rank(np.zeros((2,2)))==0
```

### 练习二：写出所有 shape

Q 为三乘八，C 为七乘二，Uk 为二乘八，Uv 为二乘五。答案：显式 K 为七乘八，V 为七乘五；吸收 query 为三乘二；scores 为三乘七；概率为三乘七；latent_read 为三乘二；最终输出为三乘五。每次乘法都检查中间维度相等，不能只记最终 shape。尤其概率轴是历史长度七，不是 latent 维度二。

### 练习三：故意使用错误缩放

沿用第四课手算，比较正确分母二与错误分母根号二。完整参考如下。断言输出不同只说明这个反例能暴露错误，不替代一般证明；一般原因是 softmax 的相对概率随分数尺度变化。

```python
correct_out,_,_=absorbed_latent_attention(q_hand,c_hand,uk_hand,uv_hand)
bad_scores=(q_hand@uk_hand.T)@c_hand.T/np.sqrt(c_hand.shape[-1])
bad_out=(softmax(bad_scores)@c_hand)@uv_hand
assert np.max(np.abs(correct_out-bad_out))>1e-2
```

### 练习四：有 mask 与无 mask 都要等价

用不同查询数和历史长度验证，避免只测试方阵因果情形。参考选择两个查询、五个历史，第一行允许前三个，第二行允许全部。显式与吸收应在相同 mask 下相等；取消 mask 后仍相等，但两种 mask 对应的模型输出通常不同。

```python
q_rect=rng.normal(size=(2,6))
c_rect=rng.normal(size=(5,2))
mask_rect=np.array([[1,1,1,0,0],[1,1,1,1,1]],dtype=bool)
for selected_mask in (None,mask_rect):
    left_out,_,_=explicit_latent_attention(q_rect,c_rect,uk,uv,selected_mask)
    right_out,_,_=absorbed_latent_attention(q_rect,c_rect,uk,uv,selected_mask)
    np.testing.assert_allclose(left_out,right_out,atol=1e-12)
```

### 练习五：构造 softmax 不能交换的反例

取两个二乘二矩阵，比较 softmax(A B) 与 softmax(A) softmax(B)。下面每行归一化后维度合法，但数值不同，说明维度正确不等于代数规则成立。解释时指出指数与按行分母依赖整行输入，不能分别作用于两个因子后相乘。

```python
a=np.array([[2.,0.],[0.,1.]])
b=np.array([[1.,3.],[2.,0.]])
assert not np.allclose(softmax(a@b),softmax(a)@softmax(b))
```

### 练习六：把值上投影吸收到输出头

单头情况下，验证 $((PC)U_V)W_O=(PC)(U_VW_O)$。参考同时检查合并权重 shape 为 r 乘输出维度。若中间插入非线性激活，等价一般不再成立；不能把有激活的两层 MLP 直接合成一个线性层。

```python
pc=rng.normal(size=(3,2))
uv_local=rng.normal(size=(2,5))
wo_local=rng.normal(size=(5,4))
merged=uv_local@wo_local
assert merged.shape==(2,4)
np.testing.assert_allclose((pc@uv_local)@wo_local,pc@merged,atol=1e-12)
nonlinear_left=np.maximum(pc@uv_local,0)@wo_local
assert not np.allclose(nonlinear_left,pc@merged)
```

### 练习七：计算附加位置状态后的压缩比

历史长度四千零九十六，显式键维度一百二十八、值维度六十四、latent 维度三十二。无额外位置状态时显式元素 $4096\times192=786432$，latent 元素 $4096\times32=131072$，比值六。若低维路径每 token 另存十六维位置状态，元素变成一十九万六千六百零八，比值四。答案必须说明显式基准是否已经包含位置分量，比较时应保持同一模型定义。

```python
assert 4096*(128+64)==786432
assert 4096*32==131072
assert 786432/(4096*32)==6
assert 4096*(32+16)==196608
assert 786432/(4096*(32+16))==4
```

### 练习八：自由矩阵的低秩近似

生成自由矩阵，做奇异值分解，只保留一个奇异方向，再报告重建误差。它与前面的精确吸收不是同一实验。完整参考还保留全部奇异值，验证完整分解能重建原矩阵，再证明截断通常有非零误差。

```python
free=rng.normal(size=(5,4))
u,s,vt=np.linalg.svd(free,full_matrices=False)
full_rebuild=(u*s)@vt
rank1=(u[:,:1]*s[:1])@vt[:1]
np.testing.assert_allclose(full_rebuild,free,atol=1e-12)
assert np.linalg.norm(rank1-free)>1e-6
assert np.linalg.matrix_rank(rank1)==1
```

### 练习九：位置分支不能在 softmax 后随意相加

第八课已经给出完整反例。请写出原因：总分数的指数满足 $e^{a+b}=e^ae^b$，但归一化分母是所有历史位置乘积权重之和；分别归一化再平均使用的是两个不同分母。因此两种组合规则不同。若你想研究后者，它可以是另一个模型，但不能作为本章模型的等价优化。

### 练习十：逐步缓存验证

第九课给出了基本参考答案，十三点二继续验证输入长度一、三、七和位置起点零、十。验收要求比较全部头和全部位置，检查 C 与位置键的时间长度一致。本章使用固定频率 RoPE，若逐步路径从头开始统一平移所有查询与键的位置起点，即使完整路径保留原起点，输出仍应在浮点容差内一致，因为相对位置未变。要构造失败用例，可以保留已有缓存键的位置，却错误重置后续查询与新键的位置，使它们与历史键的相对位置发生变化。

### 练习十一：低秩状态对模型权重版本的依赖

判断更新 Uk、Uv、Wd 后哪些缓存还能保留。若只更新上投影 Uk/Uv，而 C 的生成权重 Wd 和输入隐藏状态都不变，单独这一层的 latent 数值可能仍可用，但输出函数已经改变；完整多层模型中上游变化可能使后续层 latent 失效。更新 Wd 直接改变 C，旧 latent 不能复用。答案应从依赖图逐层判断，不能简单说“缓存与权重无关”。

### 练习十二：独立实现一个带位置分支的吸收层

完整参考由第七课 decoupled_attention 和第九课 project_all 组成。你的版本应返回输出、概率和状态，并拒绝不同步的内容与位置长度。测试包含显式展开一致性、逐步一致性、错误缩放反例、位置重复旋转反例和多头输出吸收。不要通过在函数内调用显式版本冒充吸收实现；代码中应看得到 query 进入 latent 维度、概率读取 latent、最后再映射输出这三步。

## 第十三课　渐进项目：从代数实验到源码阅读报告

第一阶段把本章函数整理成一个单独实验文件，固定种子生成权重与输入，打印每个中间 shape 和三类误差：分数、概率、输出。实现两个开关，分别选择显式 K/V 或吸收路径，以及是否使用独立位置分支。四种组合都应有明确的数学定义。先验证它们内部应等价的配对，再观察不同模型定义之间的差异。

第二阶段加入完整状态对象，包含 latent、位置键、逻辑长度和模型配置摘要。支持分块输入，矩形 mask 由旧长度与新块长度生成。每次追加只计算新 token 的投影，历史状态保持不变。用不同分块方式与全序列比较。此时你已经拥有一个可运行的低维缓存 attention 层，而不是只有一个矩阵恒等式演示。

第三阶段测量成本。分别记录持久状态 nbytes、临时展开 K/V 的 nbytes、每次 query 投影和输出投影的矩阵尺寸。用 CPU 小规模计时只做趋势观察，不对 GPU 吞吐作推断。若迁移到 GPU，应先保留高精度参考，再用实际 dtype 检查误差。服务器 GPU 可用于短时有边界的基准，但本章的数学掌握不依赖长期占用它。

第四阶段阅读官方 MLA 相关实现，建立逐项映射表：隐藏输入对应哪个变量，KV 下投影是否带归一化，内容与位置如何切分，query 是否也低秩化，RoPE 作用在哪些维度，cache 保存哪些张量，scale 如何计算，prefill 与 decode 是否选择不同路径。每个结论应附固定版本的文件与函数位置，不要只抄模型宣传图。

报告必须区分“已经在本章实现并测试的机制”“官方模型额外采用的机制”和“尚未核对的细节”。对未知细节明确标注，比用另一个模型的配置填空更专业。你不需要先下载全部权重才能理解形状与数据流；先用微型配置复现路径，再考虑权重加载和性能优化，会更容易定位问题。

本章最终能力不是背出一个压缩比例，而是能独立判断某个矩阵重排是否合法、哪个非线性阻止吸收、位置分支为什么要独立保存、以及状态与工作区分别占多少内存。带着这些问题阅读新架构，即使名字改变，你也能从公式和代码中找出真正新增的机制。

### 十三点一　反向传播工作坊：吸收不改变函数，但梯度要回到原参数

为了真正理解低秩参数化，继续手推无位置单头情形。设输出 $Y=PCU_V$，上游梯度为 G，shape 与 Y 相同。先记 $Z=PC$。值上投影梯度为 $dU_V=Z^\top G$，latent_read 梯度为 $dZ=GU_V^\top$。由于 Z 同时依赖 P 与 C，得到 $dP=dZC^\top$，以及值路径对 C 的贡献 $dC_{\text{value}}=P^\top dZ$。

softmax 的每行梯度仍是上一章熟悉的形式：$dS=P\odot(dP-\sum_jP_jdP_j)$，其中求和沿历史键轴并保留一列维度。若某连接被 mask 禁止，它的概率为零，相应梯度也为零。分数 $S=AC^\top/\sqrt{d_k}$，其中 $A=QU_K^\top$。因此 $dA=dSC/\sqrt{d_k}$，键路径对 C 的贡献为 $dC_{\text{key}}=dS^\top A/\sqrt{d_k}$。

最后，A 来自 Q 与 Uk，所以 $dQ=dAU_K$，$dU_K=dA^\top Q$。C 被键和值两条路径共享，总梯度必须相加：$dC=dC_{\text{value}}+dC_{\text{key}}$。这一步与 embedding 重复索引、残差分支的梯度累加是同一种原则：同一个变量被多条路径使用，反向时每条路径的贡献都要收集。只保留其中一条，会让前向正确但训练错误。

如果 C 又来自 $XW_D$，继续得到 $dW_D=X^\top dC$、$dX=dCW_D^\top$。真实模型还可能让 X 同时生成 query 和位置分支，因此对 X 的梯度还有这些支路的贡献。低秩并没有让链式法则变成另一套规则，只是增加了共享与分叉，需要更仔细地累计。

下面给出完整手写反向并对 Q、C、Uk、Uv 的每个元素做中心差分。损失取输出与固定 G 的内积，这样 G 正是指定的上游梯度。测试使用小尺寸双精度，不涉及任何自动微分框架，也不把模型训练能力与导数正确性混在一起。

```python
def absorbed_backward(q,c,uk,uv,upstream):
    scale=np.sqrt(q.shape[-1])
    a=q@uk.T
    scores=a@c.T/scale
    p=softmax(scores)
    z=p@c
    duv=z.T@upstream
    dz=upstream@uv.T
    dp=dz@c.T
    dc_value=p.T@dz
    ds=p*(dp-np.sum(dp*p,axis=-1,keepdims=True))
    da=ds@c/scale
    dc_key=ds.T@a/scale
    dq=da@uk
    duk=da.T@q
    return dq,dc_value+dc_key,duk,duv

q_grad=rng.normal(size=(2,3))
c_grad=rng.normal(size=(4,2))
uk_grad=rng.normal(size=(2,3))
uv_grad=rng.normal(size=(2,2))
upstream=rng.normal(size=(2,2))
parameters=[q_grad,c_grad,uk_grad,uv_grad]
analytic=absorbed_backward(*parameters,upstream)
epsilon=1e-6
for parameter,expected in zip(parameters,analytic):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+epsilon
        plus=float(np.sum(absorbed_latent_attention(*parameters)[0]*upstream))
        parameter[index]=old-epsilon
        minus=float(np.sum(absorbed_latent_attention(*parameters)[0]*upstream))
        parameter[index]=old
        numerical[index]=(plus-minus)/(2*epsilon)
    np.testing.assert_allclose(numerical,expected,atol=2e-6,rtol=2e-6)
```

这里没有 mask，是为了先检验最核心的共享梯度。加入固定布尔 mask 后，只要在同一分数位置使用负无穷并确保全遮挡行的语义明确，合法连接的导数仍按相同规则。若 mask 本身由可学习的离散选择产生，就不能把选择步骤假装成普通可微矩阵乘法；那会涉及后面稀疏与路由章节的边界。

### 十三点二　位置起点练习的完整代码答案

第十题同时验证路径等价性和位置原点的作用。完整路径所有位置使用 origin 加本地索引；逐步路径在第 t 步使用 origin 加 t。旧位置键只保存一次，后续不重新旋转。cache 长度与位置起点是不同概念：长度从零开始计数，逻辑位置可能由外部前缀或指定偏移决定。下面把完整与逐步路径包装成函数，再分别验证同原点、统一平移和错误重置三种情况。

固定频率旋转满足 $R(a)^{\mathsf T}R(b)=R(b-a)$，所以 $(R(s+a)q)^{\mathsf T}(R(t+a)k)=q^{\mathsf T}R(t-s)k$。所有查询和键同时增加相同偏移 a 时，位置点积保持不变；本例的内容分支、value 和因果可见集合也没有改变。因此完整路径从零开始、逐步路径从十开始仍应一致。若历史键保留原来的旋转位置，后续新查询和新键却重置为零，它们与历史的相对位置就改变了，这是有效的错误对照。

```python
# course-example: M07-position-origin
def full_position_output(inputs,origin):
    cc,qqc,qqr,kkr=project_all(inputs,origin)
    return decoupled_attention(qqc,qqr,cc,kkr,uk_step,uv_step,
        np.tril(np.ones((len(inputs),len(inputs)),dtype=bool)),True)[0]

def step_position_output(inputs,positions):
    old_c=np.empty((0,R))
    old_kr=np.empty((0,DR))
    collected=[]
    assert len(positions)==len(inputs)
    for t,position in enumerate(positions):
        nc,nqc,nqr,nkr=project_all(inputs[t:t+1],position)
        old_c=np.concatenate([old_c,nc])
        old_kr=np.concatenate([old_kr,nkr])
        assert len(old_c)==len(old_kr)==t+1
        out,_=decoupled_attention(nqc,nqr,old_c,old_kr,
                                  uk_step,uv_step,None,True)
        collected.append(out)
    return np.concatenate(collected,axis=1)

position_rng=np.random.default_rng(20261102)
for test_length in (1,3,7):
    test_x=position_rng.normal(size=(test_length,D))
    for origin in (0,10):
        target=full_position_output(test_x,origin)
        stepped=step_position_output(test_x,range(origin,origin+test_length))
        np.testing.assert_allclose(stepped,target,atol=1e-10,rtol=1e-10)
    zero_origin=full_position_output(test_x,0)
    shifted=step_position_output(test_x,range(10,10+test_length))
    np.testing.assert_allclose(shifted,zero_origin,atol=1e-10,rtol=1e-10)
    reset_positions=step_position_output(test_x,[0]*test_length)
    if test_length>1:
        assert np.max(np.abs(reset_positions-zero_origin))>1e-3
```

长度一的注意力只有一个可见键，softmax 权重恒为一，所以不能要求它的错误位置对照产生输出差异。长度三、七使用固定的非零位置输入和有区别的 value，才能观察相对位置错误如何改变读取。这里的阈值用于这个固定夹具，不是所有输入都必然产生同样大的差异。若某模型另外使用绝对位置 embedding 或位置相关频率缩放，共同平移可能改变模型函数；本节验证的是当前固定频率 RoPE 教学模型，不把结论推广到所有位置机制。

### 十三点三　数值误差与条件数：等价重排为什么仍可能有差异

实数矩阵乘法满足结合律，浮点运算不保证逐位相同。显式路径先计算 CUk，吸收路径先计算 QUk 的转置方向乘积，两条路径的中间量范围和加法顺序不同。若权重尺度很大、正负项严重抵消，舍入误差可能被放大。判断时应先用适当的小尺度双精度参考验证公式，再考察目标数据类型下的误差。

例如一个点积包含一个非常大的正数、一个非常大的负数和一个小数，不同求和顺序可能先丢掉小数，也可能先抵消大数后保留小数。矩阵结合改变了这些局部求和的分组。模型通常通过归一化和训练控制数值范围，但不能据此假定所有参数下误差都相同。基准报告应记录输入与权重尺度，而不只记录 shape。

softmax 对统一平移不敏感，所以计算时先减每行最大值能避免指数溢出；它对分数差异敏感，所以键路径的小误差若改变接近并列项的相对顺序，可能影响采样。比较输出文本前先比较分数、概率和 logits，可以把数值问题与离散采样放大分开。对近似量化路径，还应测量误差分布，而不是只给一个最坏样例或一个幸运样例。

本章使用高精度严格等价测试，是建立实现基准，不是规定所有生产内核都必须采用同样精度。后续算子优化时可以降低精度、融合计算或改变布局，但必须把这些选择对应到新的误差预算。数学恒等式是起点，数值工程决定你在有限精度机器上如何可靠地实现它。

### 十三点四　阅读新架构时的三张图

第一张图只画数学依赖：输入经过哪些线性映射、哪些非线性、哪些位置操作，softmax 沿哪一轴归一化。第二张图画状态：哪些张量跨 token 保存，哪些只在当前调用存在，哪些属于固定权重。第三张图画执行：prefill 是否展开，decode 是否吸收，输出投影是否合并，内核是否物化概率矩阵。三张图可能长得不同，但必须描述同一个模型函数。

把它们混成一张图，会产生常见误解。例如数学公式显式写 K 和 V，不表示实现一定把它们永久存入 cache；实现缓存 C，也不表示公式没有 K/V 的语义；内核不物化概率矩阵，也不表示它没有计算同样的归一化读取。框架阅读能力的一部分，就是在数学对象、持久状态和执行临时量之间建立准确对应。

完成项目后，用你自己的实现回答：如果只给出 cache 文件，你还需要哪些配置才能继续推理？至少需要对应权重、各分支维度、位置起点或逻辑位置、数据类型、布局和已处理长度。若保存的是量化 latent，还需要 scale 与量化轴。若这些信息缺失，即便数组能成功加载，也不能保证恢复同一函数。这是模型结构知识进入工程可靠性的具体位置。

### 十三点五　接口检查也属于模型实现

前面的函数为了展示公式，默认输入已经满足 shape 契约。独立项目应把这些前提变成显式检查。内容 query 与位置 query 必须拥有相同头数和查询数；C 与位置键必须拥有相同历史长度；Uk 的头数、latent 维和内容维必须分别匹配；Uv 的头数与 latent 维也要匹配。否则 NumPy 可能在较晚位置报错，或者通过广播产生一个看似合法但语义错误的结果。

特别要避免把全局变量当作模型配置。教学笔记里经常先用 H 表示两个头，后面在容量例子里又把 H 改成八。若一个函数在调用时读取这个可变全局 H，它的行为就会被无关例子改变。更可靠的函数应从参数 shape 推导维度，或把不可变配置作为对象成员保存。这个问题不是低秩数学特有，却会直接影响模型代码的可复现性。

下面的检查器只验证本章支持的布局与维度。它还拒绝非有限输入，因为 NaN 或无穷通常表示上游已经失效。注意 mask 中允许使用布尔值表达屏蔽，而不是要求所有分数都有限；输入张量检查与内部负无穷 mask 是不同阶段。若你有意测试极端分数，应明确构造而不是让损坏的数据静默流入。

```python
def validate_decoupled(qc,qr,c,kr,uk,uv):
    arrays=(qc,qr,c,kr,uk,uv)
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("nonfinite input")
    if qc.ndim!=3 or qr.ndim!=3 or c.ndim!=2 or kr.ndim!=2:
        raise ValueError("invalid ranks")
    heads,queries,dc=qc.shape
    if qr.shape[:2]!=(heads,queries) or c.shape[0]!=kr.shape[0]:
        raise ValueError("sequence axes disagree")
    if uk.shape!=(heads,c.shape[1],dc):
        raise ValueError("key projection shape mismatch")
    if uv.ndim!=3 or uv.shape[:2]!=(heads,c.shape[1]):
        raise ValueError("value projection shape mismatch")
    if qr.shape[-1]!=kr.shape[-1]:
        raise ValueError("position feature mismatch")

validate_decoupled(all_qc,all_qr,all_c,all_kr,uk_step,uv_step)
try:
    validate_decoupled(all_qc,all_qr,all_c,all_kr[:-1],uk_step,uv_step)
except ValueError:
    pass
else:
    raise AssertionError("expected cache length mismatch")
```

对错误输入给出早期且具体的异常，能让后续框架集成更容易定位。例如内容缓存长度六、位置缓存长度五，不能简单裁到较短长度继续计算；那会隐藏状态丢失。正确处理应由调用者恢复一致状态，或者明确失败并重新计算。模型实现的职责不只是让正确输入得到正确输出，也包括防止错误状态被包装成似乎正常的分布。

最后用自己的话解释一次完整数据流：隐藏输入生成共享 latent 与独立位置键，当前查询分成内容和位置两部分，内容项通过吸收在 latent 空间比较，位置项单独做点积，两项合成一个概率分布，概率读取 latent 后再映射成每头值输出。能连续说清这条链，并把每一步对应到代码和 shape，就已经达到本章的核心目标。

还有一个值得独立实验的参数是 latent 维度。改变它不仅改变缓存容量，也改变内容查询进入的比较空间和值重建的中间空间，因此可能同时影响表达能力、参数量与计算成本。不能只依据缓存公式把维度不断缩小，再期待模型质量自动保持。若要做训练比较，应控制数据、训练步数、优化器和评估集合，分别记录损失、任务指标与资源消耗。

低秩因子还存在表示不唯一性。对可逆矩阵 A，可以把 C 改成 CA，把上投影改成 A 的逆乘原上投影，显式 K/V 不变。因此 latent 的某个坐标通常没有唯一的语义解释。比较两个训练模型的 latent 时，逐坐标数值差异不直接等于功能差异；更有意义的是比较重建的键值、分数和输出。这一点也提醒你，内部表示的可视化需要谨慎，不能仅凭某个维度看起来较大就赋予固定概念。

参数与缓存应分别列账，吸收后的临时矩阵也要计入峰值；只展示持久状态无法证明运行时总内存一定更低。
