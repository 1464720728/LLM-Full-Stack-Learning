# M10　线性注意力与 Delta 记忆：从历史列表到固定状态

[本卷导航](README.md) · 前置：[注意力](03-attention.md)、[缓存](06-kv-gqa.md)、[稀疏连接](09-sparse-attention.md) · 下一章：[状态空间模型](11-state-space.md)

前两章仍为历史位置保留记录，只是减少维度、连接或记录数量。本章换一个问题：能否把全部过去汇总成一个大小不随长度增长的状态，并让新查询直接读取它？答案是可以，但必须改变或限制相似度函数，并接受有限状态对记忆表达的约束。你将从显式 kernel attention 推导递推式，再学习遗忘与误差纠正式写入，最后完成状态延续、梯度和关联记忆实验。

“线性”主要描述固定特征维度下随序列长度的增长，不表示整个模型只有线性函数。特征映射、门控、归一化和投影都可以是非线性的。也不表示它与标准 softmax attention 普遍等价。本章精确等价的对象是同一个特征 kernel 的显式与递推实现；与 softmax 的比较属于不同算子的比较。

## 第一课　为什么不能直接把 softmax attention 重新结合

若没有中间非线性，矩阵结合律允许 $(QK^\top)V=Q(K^\top V)$。左边先形成查询乘历史的分数矩阵，右边先把键和值汇总成一个特征乘特征矩阵。右边可能避免长度平方的中间量。但标准 attention 是 $\operatorname{softmax}(QK^\top/\sqrt d)V$，softmax 对每一行的所有历史分数做指数和归一化，不能被删掉或任意移动。

一个两位置反例已经足够。查询对两个键分数为零和二，值为一和五，softmax 读取约四点五二三二；直接用分数乘值则为十。给结果加一个统一除数也不能普遍恢复 softmax，因为指数改变了相对权重。线性 attention 不是在不改变函数的情况下偷偷跳过这个非线性，而是选择另一种可分解的相似度。

定义 $\kappa(q,k)=\phi(q)^\top\phi(k)$，其中特征映射 $\phi$ 把原向量映射到 m 维。若特征非负，相似度非负，可以像注意力一样归一化为权重。输出定义为所有合法历史 value 的相似度加权平均。这里 m 不必等于原头维度，但课程使用相同维度以减少符号；不要把这个教学选择当作所有线性 attention 的限制。

有些方法试图用特征映射近似 softmax kernel，有些直接设计不同 kernel。本章选择 ELU 加一作为独立定义，不声称它精确等于指数点积。理解这个边界后，你就知道应做两类实验：显式 kernel 与递推路径必须相等；kernel 与 softmax 可以不同，需要任务质量或近似误差分析。

## 第二课　正特征映射：每个分量具体怎么算

课程使用逐元素映射：输入非负时输出 x 加一，输入负数时输出指数 x。这等于 ELU(x) 加一。零点两侧都得到一，函数连续；正侧导数为一，负侧导数为指数 x，在零点也衔接。对任何有限实数，数学上输出严格为正，但浮点中极负数的指数可能下溢为零。

手算输入负二、零、三，特征为 e 的负二次方、一、四，约零点一三五三三五、一、四。两个向量的特征内积是各坐标乘积之和，不是先算原始点积再取指数。比如 q 为零、零，k 为一、负一，则 phi(q) 为一、一，phi(k) 为二、e 的负一次方，相似度约二点三六七八七九；它与 exp(q 点乘 k) 等于一显然不同。

实现时不要用会同时计算两个分支的表达式对所有正数也执行 exp(x)，否则大正数可能产生无用的溢出警告。先用布尔索引分开处理正负区间，只在负区间计算指数。输入类型应转换成浮点，避免整数数组接收指数结果时被截断。有限性检查能阻止 NaN 与无穷悄悄进入持久状态。

```python
import numpy as np
from labs.models.core import positive_features,linear_attention_dense,linear_attention_recurrent
from labs.models.core import delta_recurrence

def feature_map(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("finite features required")
    result=np.empty_like(x)
    positive=x>=0
    result[positive]=x[positive]+1.
    result[~positive]=np.exp(x[~positive])
    return result

np.testing.assert_allclose(feature_map(np.array([-2.,0.,3.])),
                           [np.exp(-2),1.,4.])
np.testing.assert_allclose(feature_map(np.array([0.,0.]))@
                           feature_map(np.array([1.,-1.])),2+np.exp(-1))
np.testing.assert_allclose(feature_map(np.array([-3.,2.])),
                           positive_features(np.array([-3.,2.])))
```

非负特征保证分母不会因正负抵消而变成零，但不保证数值永远安全。特征过大可能让状态累计变大，特征过小可能下溢，长序列累加也会有舍入误差。后面会单独分析归一化与缩放，不把“正值”当成万能稳定性证明。

## 第三课　显式 kernel attention 的完整定义

设映射后的查询为 $a_t=\phi(q_t)$，键为 $b_i=\phi(k_i)$，value 为 $v_i$。当前位置包含自身，合法历史为 i 小于等于 t。权重未归一化值为 $a_t^\top b_i$，分母是这些相似度之和，输出为
$$
y_t=\frac{\sum_{i\le t}(a_t^\top b_i)v_i}
{\sum_{i\le t}a_t^\top b_i}.
$$
分子是 value 维向量，分母是标量。不能把分母误写成对每个特征坐标分别归一化，那会定义另一个算子。

显式实现先计算全部特征内积矩阵 A B 的转置，再将未来位置置零。这里置零合法，因为我们处理的是非负 kernel 权重，而不是 softmax 前的 logits；零权重就是无贡献。然后每行分子为权重矩阵乘 V，分母为行和，最后相除。不要把这个零 mask 规则机械复制回 softmax logits，后者需要负无穷屏蔽。

若某行分母为零，本章定义输出零。数学上有限输入的正特征通常使包含自身的行分母为正，但浮点下溢可能产生零。接口应明确这个边界，显式与递推实现采用同一规则。随意只在某条路径加 epsilon，会破坏等价性，也改变极小但合法分母的输出。

```python
def explicit_kernel_attention(q,k,v):
    aq,bk=feature_map(q),feature_map(k)
    weights=aq@bk.T
    weights*=np.tril(np.ones((len(q),len(k)),dtype=bool))
    denominator=weights.sum(axis=1,keepdims=True)
    numerator=weights@v
    output=np.divide(numerator,denominator,
                     out=np.zeros_like(numerator),where=denominator>0)
    return output,weights,denominator

q_small=np.array([[0.,1.],[1.,0.]])
k_small=np.array([[0.,0.],[1.,0.]])
v_small=np.array([[2.],[8.]])
explicit_small,weights_small,den_small=explicit_kernel_attention(q_small,k_small,v_small)
np.testing.assert_allclose(weights_small,[[3.,0.],[3.,5.]])
np.testing.assert_allclose(den_small,[[3.],[8.]])
np.testing.assert_allclose(explicit_small,[[2.],[5.75]])
```

逐项手算第二行：query 特征为二、一，第一键特征一、一，内积三；第二键特征二、一，内积五。分母八，分子三乘二加五乘八等于四十六，输出四十六除八等于五点七五。第一行只有自己，任何正相似度都约掉，输出等于第一 value 二。

## 第四课　固定状态从交换求和次序中出现

把分子展开，$a_t^\top b_i$ 是特征坐标乘积的和。将历史求和与特征求和交换，可写成 $a_t^\top S_t$，其中
$$
S_t=\sum_{i\le t}b_iv_i^\top,\qquad z_t=\sum_{i\le t}b_i.
$$
S 的 shape 为特征维 m 乘 value 维 dv，z 为 m 维。输出为 $a_t^\top S_t/(a_t^\top z_t)$。历史不再以一条条 K/V 保存，而被汇总成键特征与 value 的外积和，以及键特征的总和。

每来一个 token，更新 $S_t=S_{t-1}+b_tv_t^\top$，$z_t=z_{t-1}+b_t$。状态大小 m 乘 dv 加 m，与历史长度无关。每步外积更新和读取仍需要约 m 乘 dv 的工作，所以固定状态不是零成本，也不保证短序列下比普通 KV 更小或更快。

继续第三课手算。第一键特征一、一，value 二，外积列为二、二，所以 S1 为二、二，z1 为一、一。第二键特征二、一，value 八，外积为十六、八，累计 S2 为十八、十，z2 为三、二。第二 query 特征二、一，分子二乘十八加十等于四十六，分母二乘三加二等于八，得到同样五点七五。

```python
def recurrent_kernel_attention(q,k,v,state=None):
    aq,bk=feature_map(q),feature_map(k)
    feature_dim=bk.shape[1]
    value_dim=v.shape[1]
    if state is None:
        matrix=np.zeros((feature_dim,value_dim))
        normalizer=np.zeros(feature_dim)
    else:
        matrix=np.array(state[0],dtype=float,copy=True)
        normalizer=np.array(state[1],dtype=float,copy=True)
    outputs=[]
    traces=[]
    for query,key,value in zip(aq,bk,v):
        matrix+=np.outer(key,value)
        normalizer+=key
        denominator=float(query@normalizer)
        numerator=query@matrix
        outputs.append(numerator/denominator if denominator>0 else np.zeros(value_dim))
        traces.append((matrix.copy(),normalizer.copy(),denominator))
    return np.asarray(outputs),(matrix,normalizer),traces

recurrent_small,state_small,trace_small=recurrent_kernel_attention(q_small,k_small,v_small)
np.testing.assert_allclose(recurrent_small,explicit_small)
np.testing.assert_allclose(trace_small[0][0],[[2.],[2.]])
np.testing.assert_allclose(trace_small[1][0],[[18.],[10.]])
np.testing.assert_allclose(trace_small[1][1],[3.,2.])
```

这里先写入当前记录，再读取，所以包含当前位置。若先读后写，第一步会读空状态，后续每步少看当前 value。那可以定义另一种记忆机制，但不等于本章的因果 attention。时序必须从公式决定，不能凭循环写起来顺手就选择。

## 第五课　一维退化与有限状态的记忆限制

当特征维度只有一，query 特征是正标量 a，分子与分母都含 a，它会约掉。输出只由历史 key 权重和 value 决定，与当前 query 的具体大小无关。手算键特征一、二、一，value 二、四、八，状态 S 为二加八加八等于十八，z 为四，输出四点五。任何正 query 都读到相同结果。

这个退化揭示了特征维度的作用。多维状态允许 query 通过不同方向组合各行统计量；一维只有一个方向，无法根据查询区分多种关联。增加维度可以提高表示能力，但也增加状态与每步更新成本。固定状态并不意味着无限精确记忆，而是把历史压缩到有限数量的统计量中。

即使二维也不能任意恢复所有历史记录。若两组不同历史产生相同 S 和 z，后续任何 query 都得到相同输出，模型无法区分这两组历史。比如交换两条记录的时间顺序，简单加法状态保持不变，因此最终状态不包含它们的顺序。每个中间输出仍依赖前缀，但最终汇总对历史排列不敏感。加入衰减或 Delta 更新后，情况会改变。

```python
keys_1d=np.array([1.,2.,1.])
values_1d=np.array([2.,4.,8.])
matrix_1d=float(np.sum(keys_1d*values_1d))
normalizer_1d=float(keys_1d.sum())
assert matrix_1d==18 and normalizer_1d==4
for positive_query in (.1,1.,10.):
    np.testing.assert_allclose(positive_query*matrix_1d/
                               (positive_query*normalizer_1d),4.5)
```

因此评估线性记忆应包含需要区分顺序、精确复制、关联检索和抗干扰的任务，而不只看能处理多长输入。长度上限高表示系统能接收更多 token，不自动表示每个早期细节都被完整保存。架构设计通常在状态成本、训练可行性与任务需要之间取舍，不能只用“大 O 更好”替代能力分析。

## 第六课　full、step 与 chunk：状态延续的完整契约

显式 kernel、一次递推和分块递推应计算同一函数。分块时第一段返回 S/z，第二段从它们继续，不能重置任何一项。只延续 S 而重置 z，会保留分子却丢失分母，输出幅度错误；只延续 z 而重置 S，会把历史权重放在分母却没有历史 value 贡献。两者必须作为同一状态对象同步管理。

状态函数应复制调用者传入数组，或者明确声明原地更新。本章课程实现复制状态，因此分支与回滚实验不会意外修改旧快照。若为了性能改成原地更新，调用者必须管理所有权；同一个前缀分叉到两个后续序列时，需要独立尾状态或复制。固定状态小并不意味着它没有生命周期问题。

```python
rng=np.random.default_rng(10)
q_seq=rng.normal(size=(11,4))
k_seq=rng.normal(size=(11,4))
v_seq=rng.normal(size=(11,3))
dense_seq=linear_attention_dense(q_seq,k_seq,v_seq)
whole_seq,whole_state=linear_attention_recurrent(q_seq,k_seq,v_seq)
first,first_state=linear_attention_recurrent(q_seq[:4],k_seq[:4],v_seq[:4])
saved_first=(first_state[0].copy(),first_state[1].copy())
second,second_state=linear_attention_recurrent(q_seq[4:],k_seq[4:],v_seq[4:],first_state)
np.testing.assert_allclose(dense_seq,whole_seq,atol=1e-12)
np.testing.assert_allclose(np.concatenate([first,second]),whole_seq,atol=1e-12)
np.testing.assert_array_equal(first_state[0],saved_first[0])
np.testing.assert_array_equal(first_state[1],saved_first[1])
np.testing.assert_allclose(second_state[0],whole_state[0],atol=1e-12)
np.testing.assert_allclose(second_state[1],whole_state[1],atol=1e-12)
```

输入长度必须严格相同。Python 的 zip 会按最短序列结束，若 Q 长十一、K 长十、V 长十一，没有校验就可能静默丢掉最后一个 query。模型代码应在进入循环前验证 shape、非空、有限性与状态尺寸。一个能返回数组的函数不一定完成了所有输入，接口检查是数值正确性的一部分。

分块路径的误差可能来自求和顺序，但本例两段仍按同一时间顺序累加，通常非常接近。若使用并行前缀和，归约树不同，浮点末位可能变化。验证应在高精度小样例建立公式基准，再为低精度并行实现设置合理容差，不能把任何差异都归咎于“线性 attention 本来近似”。

## 第七课　分母、下溢与状态缩放：不能靠随手加 epsilon 修复

归一化分母很小，不代表输出应该很小。取一维 q 与 k 都为负三百五十，特征约为 e 的负三百五十次方，相似度约为 e 的负七百次方，仍在双精度可表示的正数范围。若 value 为一，分子与分母相同，输出应为一。若把分母强行夹到十的负三百次方，输出会被错误缩小几个数量级。

因此本章显式与递推路径对任何正分母直接相除，只有分母实际为零时返回零。这个规则不是说所有生产模型都不该使用 epsilon，而是说 epsilon 属于函数定义，必须在所有路径一致，并认识到它改变极小值行为。稳定性处理不能只在某个实现里偷偷加入，再要求与另一个公式严格等价。

```python
tiny_q=np.array([[-350.]])
tiny_k=np.array([[-350.]])
unit_value=np.array([[1.]])
tiny_dense=linear_attention_dense(tiny_q,tiny_k,unit_value)
tiny_recurrent,_=linear_attention_recurrent(tiny_q,tiny_k,unit_value)
np.testing.assert_allclose(tiny_dense,[[1.]])
np.testing.assert_allclose(tiny_recurrent,[[1.]])
underflow_q=np.array([[-1000.]])
underflow_k=np.array([[-1000.]])
underflow_dense=linear_attention_dense(underflow_q,underflow_k,unit_value)
underflow_recurrent,_=linear_attention_recurrent(underflow_q,underflow_k,unit_value)
np.testing.assert_array_equal(underflow_dense,[[0.]])
np.testing.assert_array_equal(underflow_recurrent,[[0.]])
```

状态整体缩放也要谨慎。对于某一次读取，把 S 和 z 同时乘同一个正数 c，分子分母都乘 c，输出不变。但如果缩放之后继续追加未缩放的新外积和新 key，历史与新记录的相对权重就改变了。不能为了避免状态变大，每隔几步随意把 S/z 除二，然后继续原样写入，声称函数保持不变。

要保持精确语义，需要同时维护尺度因子，使新写入也转换到相同坐标系；或者采用明确定义的衰减模型，承认旧记录权重发生变化。数值重标定与遗忘机制有时看起来都是“把状态乘一个数”，但前者应保持函数，后者有意改变历史权重。实现时必须说明是哪一种。

另一个细节是 query 整体正缩放会在归一化中约掉，而单独某个 key 的缩放不会约掉，它会改变该记录相对权重。所有历史 key 同时乘同一常数，则分子分母一起缩放，结果不变。利用这些不变量可以构造测试，但只能在映射后的特征空间施加缩放；对原始 q 乘常数再经过非线性 phi，通常不等于对 phi(q) 乘常数。

## 第八课　完整反向：分子、分母和共享状态都要有梯度

先对单个读取推导。令 a 为查询特征，S 为状态矩阵，z 为归一化向量，分子 n 等于 aS，分母 d 等于 az，输出 y 等于 n 除 d。上游梯度 g 与 y 同维。对分子梯度为 g 除 d；对分母梯度为负的 g 与 n 点积除 d 平方，也可写成负的 g 与 y 点积除 d。

于是对 S 的梯度是 a 的列向量与分子梯度的外积；对 z 的梯度是 a 乘分母梯度；对 a 的梯度包含两条路径，分别是分子梯度乘 S 转置，以及分母梯度乘 z。只对分子求导而把分母当常量，会得到错误的 query 与 key 梯度，即使前向完全正确。

对完整显式 kernel attention，先把每行归一化权重写成 P。给定输出梯度 G，value 梯度为 P 转置乘 G；未归一化权重梯度为每行的 value 相关项减去输出相关项，再除分母。更具体地，若 w 是某行 kernel 权重，y 是加权平均，某个权重的梯度为 g 点乘 v_i 减 y，再除该行分母。这个形式直观显示：增加某记录权重的影响，取决于它的 value 相对当前平均输出的差异。

下面实现显式 kernel 的完整手写反向，覆盖 Q、K、V。mask 之后的未来权重梯度置零；特征映射导数在非负区间为一，负区间为指数。用中心差分检查每个输入元素，输入避开零点附近，减少数值边界干扰。

```python
def feature_derivative(x):
    x=np.asarray(x,dtype=float)
    result=np.ones_like(x)
    negative=x<0
    result[negative]=np.exp(x[negative])
    return result

def kernel_forward_saved(q,k,v):
    aq,bk=feature_map(q),feature_map(k)
    mask=np.tril(np.ones((len(q),len(k)),dtype=bool))
    raw=(aq@bk.T)*mask
    den=raw.sum(axis=1,keepdims=True)
    p=raw/den
    out=p@v
    return out,(q,k,v,aq,bk,mask,raw,den,p,out)

def kernel_backward(saved,gradient):
    q,k,v,aq,bk,mask,raw,den,p,out=saved
    dv=p.T@gradient
    dp=gradient@v.T
    draw=(dp-np.sum(dp*p,axis=1,keepdims=True))/den
    draw*=mask
    daq=draw@bk
    dbk=draw.T@aq
    return daq*feature_derivative(q),dbk*feature_derivative(k),dv

grad_q=np.array([[.2,-.4],[.7,-.2],[.3,.8]])
grad_k=np.array([[-.6,.5],[.4,.2],[-.3,.9]])
grad_v=np.array([[1.,2.],[-1.,.5],[.2,-.7]])
grad_upstream=rng.normal(size=(3,2))
grad_out,grad_saved=kernel_forward_saved(grad_q,grad_k,grad_v)
grad_analytic=kernel_backward(grad_saved,grad_upstream)
grad_parameters=[grad_q,grad_k,grad_v]
for parameter,expected in zip(grad_parameters,grad_analytic):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=float(np.sum(kernel_forward_saved(*grad_parameters)[0]*grad_upstream))
        parameter[index]=old-1e-6
        minus=float(np.sum(kernel_forward_saved(*grad_parameters)[0]*grad_upstream))
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=2e-7,rtol=2e-7)
```

这份反向假设每行分母严格为正，不覆盖下溢为零时的分段边界。训练测试使用正常尺度，先验证主要导数；如果要支持极端数值，应明确零分母分支的导数约定。不能因为前向定义了零输出，就假设所有自动微分系统都会自动得到你想要的边界梯度。

递推训练还需要把未来损失对状态的梯度向过去传播。状态 S/z 被后续所有读取使用，反向时要累加来自未来步骤的梯度，再穿过当前写入。若每步把状态 detach，训练变成截断版本，前向可能相同，参数梯度却不同。固定状态节省推理历史存储，不代表完整训练反向可以免费丢弃全部时间依赖。

## 第九课　遗忘：给历史记录不同时间权重

简单累加对所有过去记录保持永久贡献，容易出现干扰。引入标量衰减 alpha，定义 $S_t=\alpha_tS_{t-1}+b_tv_t^\top$，$z_t=\alpha_tz_{t-1}+b_t$，输出仍用归一化读取。为了保持分子分母对应同一历史权重，二者必须使用相同衰减。只衰减 S 不衰减 z，会让输出幅度随时间额外缩小，定义已经改变。

展开三步可以看清每条记录的权重。第一条写入到第三步被乘 alpha2 与 alpha3；第二条只乘 alpha3；第三条系数一。若 alpha 恒为二分之一，第一条到第三步权重四分之一，第二条二分之一，第三条一。它形成指数时间偏好，而不是突然删除某一条记录。

一维 key 特征都为一，value 为二、四、八，alpha 为二分之一。第一步 S 二、z 一，输出二；第二步 S 五、z 一点五，输出十除三；第三步 S 十点五、z 一点七五，输出六。直接按历史权重四分之一、二分之一、一计算，分子半加二加八为十点五，分母一点七五，同样得到六。

```python
def decayed_kernel_features(q_features,k_features,values,decay):
    m,dv=k_features.shape[1],values.shape[1]
    state=np.zeros((m,dv))
    normalizer=np.zeros(m)
    outputs=[]
    for query,key,value,alpha in zip(q_features,k_features,values,decay):
        state=alpha*state+np.outer(key,value)
        normalizer=alpha*normalizer+key
        den=float(query@normalizer)
        outputs.append((query@state)/den if den>0 else np.zeros(dv))
    return np.asarray(outputs),(state,normalizer)

decayed_out,decayed_state=decayed_kernel_features(
    np.ones((3,1)),np.ones((3,1)),np.array([[2.],[4.],[8.]]),np.full(3,.5))
np.testing.assert_allclose(decayed_out[:,0],[2.,10/3,6.])
np.testing.assert_allclose(decayed_state[0],[[10.5]])
np.testing.assert_allclose(decayed_state[1],[1.75])
```

alpha 接近一保留较久，接近零更偏向新信息。若 alpha 由输入生成，它成为选择性遗忘门，但不能直接解释成模型理解了什么“重要记忆”。门的实际行为需要在具体任务上观察。逐通道衰减可以让不同状态方向有不同时间尺度，公式和并行化也相应更复杂。

还有一个容易混淆的实验：写入一次后让后续输入 value 为零，仍可能持续写入新的 key 到 z，导致输出下降既来自遗忘也来自分母增加。若想单独观察旧状态衰减，应把后续写入项完全设零，或直接测状态而不是只测归一化输出。实验设计必须控制到底哪个机制在变化。

## 第十课　Delta 规则：写入目标与当前预测之间的误差

加法外积每次都把新关联叠加进去。同一个 key 先写三再写七，普通加法记忆会累成十，而许多任务希望新值覆盖旧值。Delta 规则先用当前 key 读取已有预测，再只写入误差。状态方向仍是 key 维乘 value 维，先衰减得到临时状态，预测为 key 行向量乘临时状态，误差为目标 value 减预测。

完整更新为
$$
\widetilde S_t=\alpha_tS_{t-1},\quad
e_t=v_t-k_t^\top\widetilde S_t,\quad
S_t=\widetilde S_t+\beta_tk_te_t^\top,\quad
y_t=q_t^\top S_t.
$$
这里 k 与 q 不再自动套用前面 ELU 加一映射，课程 delta_recurrence 直接使用传入向量；也没有归一化分母 z。这是另一种记忆更新，不应把两个函数的状态或输入混用。

取单位 key 为一、零，value 是标量。初始 S 两行都零，第一次写三、beta 一：预测零，误差三，更新后第一行三、第二行零，读取同 key 得三。第二次写七：预测三，误差四，更新后第一行七，读取七。新写入纠正旧预测，而不是把整个七再叠加。

beta 为二分之一时，第二次从三走向七的一半，读到五。若第一次也使用二分之一，则第一次只写到一点五，第二次走到四点二五。讨论“第二次 beta 为二分之一得到五”时，必须明确第一步已完整写入三，不能省略初始状态。下面代码显式给出两个 beta 序列。

```python
unit_keys=np.array([[1.,0.],[1.,0.]])
unit_queries=unit_keys.copy()
write_values=np.array([[3.],[7.]])
overwrite,_=delta_recurrence(unit_queries,unit_keys,write_values,np.array([1.,1.]))
partial,_=delta_recurrence(unit_queries,unit_keys,write_values,np.array([1.,.5]))
both_partial,_=delta_recurrence(unit_queries,unit_keys,write_values,np.array([.5,.5]))
np.testing.assert_allclose(overwrite[:,0],[3.,7.])
np.testing.assert_allclose(partial[:,0],[3.,5.])
np.testing.assert_allclose(both_partial[:,0],[1.5,4.25])
```

Delta 更新可以理解为对当前 key-value 关联做一次在线误差修正，但它不是一个完整语言模型。真实 gated delta 系统还需要输入投影、key 归一化、可能的短卷积、门控生成、输出归一化与高效分块算法。本章把核心状态更新单独拆出，是为了让你能推导和验证它，再逐项加入外层组件。

### 十点一　单位范数为何决定覆盖结论

更新后再次用同一个 key 读取，结果为旧预测加 beta 乘 key 范数平方乘误差。只有 key 范数平方为一、beta 为一时，恰好补足全部误差。如果 key 为二、零，范数平方四，beta 一会把误差放大四倍，可能过冲。归一化 key、控制 beta 或修改更新公式，都是模型定义的一部分。

手算一维 key 二、初始状态零、目标三、beta 一。误差三，状态更新为二乘三等于六；重新读 key 得十二，不是三。若 beta 取四分之一，状态变一点五，读取三，才实现当前 key 上的精确匹配。这个例子比一句“注意数值稳定”更具体地说明了尺度问题。

```python
large_key=np.array([[2.,0.]])
large_value=np.array([[3.]])
overshoot,_=delta_recurrence(large_key,large_key,large_value,np.array([1.]))
corrected,_=delta_recurrence(large_key,large_key,large_value,np.array([.25]))
np.testing.assert_allclose(overshoot,[[12.]])
np.testing.assert_allclose(corrected,[[3.]])
```

多个 key 之间若不正交，更新一个关联也会改变另一个关联的读取。假设 key1 为一、零，key2 为一除根号二、一除根号二。向 key2 写入误差时，状态第一行也被修改，因此 key1 的旧读取受到影响。这是关联记忆的干扰，不是实现 bug。维度、归一化、门控和训练共同决定能保留多少关联。

### 十点二　遗忘必须发生在误差读取之前还是之后

本章先衰减旧状态，再用当前 key 读取预测，然后写误差。如果改成先从未衰减状态读取误差，再衰减并写入，得到不同函数。设标量旧状态四、key 一、目标六、alpha 二分之一、beta 一。正确顺序先得到临时状态二，误差四，最终状态六；错误顺序先算误差二，再把旧状态衰减成二并加二，最终四。

这说明代码中几行看似可交换的操作其实具有语义顺序。测试应使用 alpha 不等于一、旧状态非零的例子，否则两种顺序可能偶然相同。后续分块并行化也必须从这一确定更新式推导，不能先写一个方便的 scan 再假设它实现了原模型。

## 第十一课　递推不等于只能串行：加法前缀和与仿射扫描

简单线性 attention 的状态是外积项与 key 特征的前缀和。每个 token 的外积可以独立计算，再对时间轴做前缀和，得到所有位置的 S；z 同理。这样训练或 prefill 可以使用并行扫描，而 decode 仍用单步更新。数学上它们计算同一组前缀统计量，区别在执行顺序与中间存储。

加法具有结合律，所以可以先算每块总和，再求块前缀，再在块内补上之前块的状态。块内第一个位置不是从零开始，而是从之前全部块的累计状态开始。若只把每块独立计算再拼接，等于把历史在块边界清空，输出会改变。分块算法的核心就是正确传递块边界状态。

```python
def prefix_kernel_attention(q,k,v):
    aq,bk=feature_map(q),feature_map(k)
    outer_terms=np.einsum("ti,tj->tij",bk,v)
    states=np.cumsum(outer_terms,axis=0)
    normalizers=np.cumsum(bk,axis=0)
    numerators=np.einsum("ti,tij->tj",aq,states)
    denominators=np.einsum("ti,ti->t",aq,normalizers)
    return np.divide(numerators,denominators[:,None],
                     out=np.zeros_like(numerators),
                     where=denominators[:,None]>0)

np.testing.assert_allclose(prefix_kernel_attention(q_seq,k_seq,v_seq),
                           dense_seq,atol=1e-12)
```

这份参考把每个位置的完整状态都物化出来，训练内存随 T 乘 m 乘 dv 增长，不能把它称为固定内存实现。固定状态指逐步推理所需的持久状态；为了并行训练或反向传播，可能保存更多中间量。不同执行模式的内存账本必须分别计算。高效算法可以通过分块、重计算与融合减少物化，但需要额外实现。

带标量衰减的更新是仿射变换：状态变成 a 乘旧状态加 b。两步合并为第二步 a 乘第一步 a，再加第二步 a 乘第一步 b 与第二步 b。用二元组表示变换，组合规则具有结合律，因此也能做前缀扫描；但一般不能交换顺序。结合律允许改变括号，不允许把时间颠倒。

Delta 更新也可以写成仿射矩阵形式。由先衰减再纠错，
$$
S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
+\beta_tk_tv_t^\top.
$$
定义 A 为前面的 key 维方阵，B 为后面的外积，则仍是 $S_t=A_tS_{t-1}+B_t$。两步组合为 $(A_2A_1,A_2B_1+B_2)$。这给出并行化的数学入口，但直接构造和相乘这些方阵可能很贵，实际高效算法会利用低秩结构与分块关系。

```python
def compose_affine(second,first):
    a2,b2=second
    a1,b1=first
    return a2@a1,a2@b1+b2

def delta_affine_step(key,value,beta,decay):
    dim=len(key)
    a=decay*(np.eye(dim)-beta*np.outer(key,key))
    b=beta*np.outer(key,value)
    return a,b

key_a=np.array([1.,0.])
key_b=np.array([.6,.8])
step_a=delta_affine_step(key_a,np.array([3.]),.7,.9)
step_b=delta_affine_step(key_b,np.array([5.]),.4,.8)
initial=np.array([[1.],[2.]])
sequential=step_b[0]@(step_a[0]@initial+step_a[1])+step_b[1]
combined=compose_affine(step_b,step_a)
np.testing.assert_allclose(combined[0]@initial+combined[1],sequential)
reverse=compose_affine(step_a,step_b)
assert not np.allclose(reverse[0]@initial+reverse[1],sequential)
```

这里的 A 是 key 维乘 key 维，B 是 key 维乘 value 维。若 key 维很大，朴素矩阵组合可能比原始逐步外积更新更贵。找到结合律只是证明可重排，不等于已经得到高性能算法。阅读高效 chunk delta 实现时，要进一步看它如何避免完整矩阵链、如何利用三角结构、如何安排数值精度。

### 十一点一　一个三步仿射手算

标量状态初始一，三步变换分别二乘旧状态加三、四乘旧状态加五、二分之一乘旧状态减一。顺序执行得到第一步五，第二步二十五，第三步十一点五。前两步合并为八乘初始加十七，再与第三步合并为四乘初始加七点五，代入一得到十一点五。

换一种括号，先合并后两步，得到二乘第一步状态加一点五，再代入第一步二乘初始加三，仍为四乘初始加七点五。这说明结合律。若交换第一与第二步，结果会变成不同常数项，说明不交换。并行扫描依赖前者，不能误用成对更新排序无关。

## 第十二课　Delta 的完整反向：读、写、遗忘三条路径

为避免把 Delta 当成不可理解的循环，我们手写它的时间反向。每一步保存旧状态、衰减后状态、误差和新状态。输出 y 等于 q 乘新状态，上游输出梯度会给新状态增加一个 q 与梯度的外积；未来步骤也会传回状态梯度，两者先相加。随后穿过误差写入，再穿过误差读取，最后穿过衰减。

令新状态总梯度为 G。beta 梯度是 G 与外积 k e 的逐元素内积；误差梯度为 beta 乘 k 转置 G；写入路径对 k 的梯度为 beta 乘 G e。因为误差是 v 减 k 转置 R，value 梯度等于误差梯度，读取路径对 k 另贡献负的 R 乘误差梯度，对 R 贡献负的 k 与误差梯度外积。再加上新状态直接包含 R 的恒等路径，得到 R 的总梯度。

alpha 梯度是 R 梯度与旧状态的逐元素内积，旧状态梯度是 alpha 乘 R 梯度。这个梯度继续传给前一步。query 只影响当前输出读取，其梯度是新状态乘输出梯度。一个 key 同时参与预测读取和误差写入，所以必须把两条 key 梯度相加；漏掉任何一条都可能让代码看似训练却不对应原函数。

```python
def delta_forward_saved(q,k,v,beta,decay):
    state=np.zeros((k.shape[1],v.shape[1]))
    outputs=[]
    saved=[]
    for query,key,value,b,a in zip(q,k,v,beta,decay):
        previous=state.copy()
        decayed=a*previous
        error=value-key@decayed
        state=decayed+b*np.outer(key,error)
        outputs.append(query@state)
        saved.append((previous,decayed,error,state.copy()))
    return np.asarray(outputs),saved

def delta_backward(q,k,v,beta,decay,saved,upstream):
    dq=np.zeros_like(q);dk=np.zeros_like(k);dv=np.zeros_like(v)
    dbeta=np.zeros_like(beta);ddecay=np.zeros_like(decay)
    future=np.zeros((k.shape[1],v.shape[1]))
    for t in range(len(q)-1,-1,-1):
        previous,decayed,error,state=saved[t]
        gradient=future+np.outer(q[t],upstream[t])
        dq[t]=state@upstream[t]
        dbeta[t]=np.sum(gradient*np.outer(k[t],error))
        derror=beta[t]*(k[t]@gradient)
        dk[t]=beta[t]*(gradient@error)-decayed@derror
        dv[t]=derror
        ddecayed=gradient-np.outer(k[t],derror)
        ddecay[t]=np.sum(ddecayed*previous)
        future=decay[t]*ddecayed
    return dq,dk,dv,dbeta,ddecay

dq_input=np.array([[.2,.7],[.6,-.1],[.5,.4]])
dk_input=np.array([[.8,.6],[-.6,.8],[1.,0.]])
dv_input=np.array([[1.,-.5],[.3,.8],[-.2,.4]])
db_input=np.array([.2,.6,.4])
da_input=np.array([.9,.7,.8])
delta_upstream=rng.normal(size=(3,2))
delta_output,delta_saved=delta_forward_saved(dq_input,dk_input,dv_input,db_input,da_input)
delta_analytic=delta_backward(dq_input,dk_input,dv_input,db_input,da_input,
                              delta_saved,delta_upstream)
delta_parameters=[dq_input,dk_input,dv_input,db_input,da_input]
for parameter,expected in zip(delta_parameters,delta_analytic):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=float(np.sum(delta_forward_saved(*delta_parameters)[0]*delta_upstream))
        parameter[index]=old-1e-6
        minus=float(np.sum(delta_forward_saved(*delta_parameters)[0]*delta_upstream))
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=2e-7,rtol=2e-7)
course_delta,_=delta_recurrence(dq_input,dk_input,dv_input,db_input,da_input)
np.testing.assert_allclose(delta_output,course_delta,atol=1e-12)
```

这份代码检查 Q、K、V、beta 和 decay 的所有元素，包括跨时间的梯度。它不是只对最后状态求导，也不是把状态在每步切断。若你想实现截断反向，应显式在片段边界把 future 清零，并承认梯度目标改变。若 beta 和 alpha 来自 sigmoid 网络，还要继续乘相应导数，把梯度传回生成门的参数。

### 十二点一　门控范围与稳定性

若 key 归一化且 beta 在零到一，单步对当前 key 方向的误差通常以受控比例修正；alpha 在零到一提供衰减。这些约束有助于稳定，但不自动证明完整模型所有状态都不增长，因为 value、query、投影和多个方向交互仍会影响幅度。可以用 sigmoid 生成门，也可以采用其他参数化，具体范围属于架构定义。

若 beta 大于一，更新可能越过目标；若 alpha 大于一，旧状态可能被放大；若 key 未归一化，实际修正强度还乘 key 范数平方。调试时应记录这些量，而不是只看最终 loss。很多“循环状态爆炸”可以追溯到门控范围、key 尺度或重复写入语义，明确的数值日志比泛泛地降低学习率更有帮助。

## 第十三课　十二组练习与完整参考答案

### 练习一：一维 query 为什么消失

映射后 key 为一、二、一，value 为二、四、八。显式未归一化权重是 query 标量 a 乘一、二、一，分母四 a，分子十八 a，输出四点五。只要 a 为正且没有下溢，a 约掉。答案必须说明这是特征维一的退化，不是所有线性 attention 都忽略 query。

### 练习二：手算二维状态与两种查询

历史 key 特征分别一、零与零、一，value 分别二与八，则 S 为两行二、八，z 为一、一。query 特征三、一读到三乘二加八除四，等于三点五；query 特征一、三读到二加三乘八除四，等于六点五。不同方向能够选择不同历史统计量，展示多维状态相对一维的能力。

```python
state_matrix=np.array([[2.],[8.]])
state_normalizer=np.array([1.,1.])
for query,answer in ((np.array([3.,1.]),3.5),(np.array([1.,3.]),6.5)):
    np.testing.assert_allclose((query@state_matrix)/(query@state_normalizer),[answer])
```

### 练习三：只延续 S 而丢掉 z 会怎样

取第一段已经写入 key 一、value 二，所以 S 二、z 一；第二段写入 key 一、value 八。正确结果是十除二等于五。若只保留 S 而把 z 清零，追加后 S 十、z 一，输出十。它不是较小的舍入误差，而是分子分母代表不同历史。完整答案应同时指出状态必须成对保存。

```python
correct_matrix=2.+8.
correct_z=1.+1.
wrong_z=0.+1.
assert correct_matrix/correct_z==5.
assert correct_matrix/wrong_z==10.
```

### 练习四：构造先读后写的反例

第一步空状态，当前 value 为七。包含当前的定义输出七；先读后写只能得到空状态约定值零。第二步 value 为九，先读后写读到七，而包含当前的等 key 平均为八。这个反例明确时序，不需要大模型。你可以把两种机制都实现，但函数名与文档必须区分。

### 练习五：正分母不能任意截断

第七课负三百五十输入是完整可运行参考。解释时列出分子与分母都约为 e 的负七百次方，所以比值一。若添加 epsilon，分母变大而分子不变，输出缩小。稳定性修改需要在模型定义、训练和推理中一致，不能只为了消除警告随意加入。

### 练习六：状态大小交叉点

特征维与 value 维都为六十四，S 有四千零九十六个元素，z 有六十四，共四千一百六十。传统单头 K/V 每 token 一百二十八元素，三十二 token 为四千零九十六，三十三 token 为四千二百二十四。因此元素数交叉点在三十三附近。这个计算忽略门控状态、卷积缓存、层数、类型和工作区，不是墙钟性能交叉点。

```python
linear_state_elements=64*64+64
assert linear_state_elements==4160
assert 32*128<linear_state_elements<33*128
```

### 练习七：衰减的历史权重

alpha 固定四分之三，四个 value 的权重在最后一步依次为二十七除六十四、九除十六、四分之三和一。key 特征若都为一，输出是这些权重对 value 的加权平均。完整参考同时用递推与直接展开比较，防止把最早记录的衰减次数多算或少算一次。

```python
values_decay=np.array([[1.],[2.],[4.],[8.]])
alpha=.75
read_decay,_=decayed_kernel_features(np.ones((4,1)),np.ones((4,1)),
                                    values_decay,np.full(4,alpha))
history_weights=alpha**np.arange(3,-1,-1)
np.testing.assert_allclose(read_decay[-1],
                           history_weights@values_decay/history_weights.sum())
```

### 练习八：Delta 覆盖与非单位 key

单位 key、第一步 beta 一写三，第二步 beta 二分之一写七，得到三、五。key 改为二时，不应沿用答案，因为读写增益乘四。第十课给出完整程序。请再解释怎样选择 beta 才能在当前 key 方向精确覆盖：若 key 非零，可取 beta 为其范数平方的倒数，但实际模型未必采用这种规则，且跨 key 干扰仍存在。

### 练习九：写入顺序不交换

用两个不正交单位 key 分别写不同 value，交换顺序后最终状态一般不同。下面完整参考把 query 设置为当前 key，只用于产生输出；比较重点是最终状态。若使用正交 key 且无衰减、完全写入，可能碰巧交换，因此测试特意选非正交向量。

```python
keys_order=np.array([[1.,0.],[1/np.sqrt(2),1/np.sqrt(2)]])
values_order=np.array([[3.],[7.]])
_,state_ab=delta_recurrence(keys_order,keys_order,values_order,np.ones(2))
_,state_ba=delta_recurrence(keys_order[::-1],keys_order[::-1],values_order[::-1],np.ones(2))
assert not np.allclose(state_ab,state_ba)
```

### 练习十：关联干扰的完整手算

先用 key 一、零写入值三，状态为三、零。再用 key 为一除根号二、一除根号二写入值七。旧预测为三除根号二，误差为七减三除根号二。beta 一时两个状态坐标各增加误差除根号二。重新读取第一个 key，结果变成三加七除根号二减一点五，也就是一点五加七除根号二，约六点四四九七。旧关联三受到明显干扰。

```python
expected_old_read=1.5+7/np.sqrt(2)
np.testing.assert_allclose(np.array([1.,0.])@state_ab,[expected_old_read])
```

这个例子说明单位范数保证当前 key 的覆盖，不保证所有其他 key 的旧关联保持。只有与当前 key 正交的读取方向不受这次外积写入影响。模型需要通过表示学习、维度和门控管理干扰，不能把 Delta 理解成无限容量的字典。

### 练习十一：验证仿射组合结合律

随机生成三步 A/B，比较第三步组合第二步再组合第一步，与先组合前两步再组合第三步。结果应相同。这个测试验证数学组合实现，不保证朴素矩阵扫描比逐 token 快。完整参考如下。

```python
affine_steps=[]
for _ in range(3):
    affine_steps.append((rng.normal(size=(2,2)),rng.normal(size=(2,3))))
left=compose_affine(affine_steps[2],compose_affine(affine_steps[1],affine_steps[0]))
right=compose_affine(compose_affine(affine_steps[2],affine_steps[1]),affine_steps[0])
np.testing.assert_allclose(left[0],right[0],atol=1e-12)
np.testing.assert_allclose(left[1],right[1],atol=1e-12)
```

### 练习十二：完整梯度验收

第八课与第十二课分别给出 kernel 和 Delta 的全部手写反向及逐元素中心差分。你的独立实现应保留这些测试，并加入一个故障版本：漏掉 Delta 的 key 读取梯度，或者漏掉 kernel 分母梯度，确认差分会失败。测试不仅要在正确实现上通过，也应能抓住有代表性的错误，否则它可能只覆盖了退化输入。

把一次差分验证扩展到多个固定种子时，避免在特征映射零点、ReLU 拐点或极端下溢处误用普通平滑中心差分。先验证远离边界的导数，再为边界行为写单独约定与测试。数值验证是一种实验工具，需要理解其前提，而不是无条件按一个按钮。

## 第十四课　把记忆机制放进一个可训练实验

前面已经验证导数，现在用它实际学习一个门。任务很小但完整：输入一串标量目标，每一步都希望记忆读取等于当前目标；key 与 query 都取单位标量，decay 固定一，唯一可训练参数通过 sigmoid 生成所有步骤共享的 beta。初始 beta 较小，记忆更新缓慢；训练应该推动 beta 接近一，使当前写入更接近覆盖。

这个任务刻意简单，因为它只检验损失、跨时间反向、门参数化与优化更新是否接通。它不能证明模型具备复杂关联记忆，也不能证明某个完整架构的性能。你应先在这样的受控问题上理解每个梯度，再扩展数据与网络；直接训练大模型会把结构错误、优化问题和任务难度混在一起。

```python
gate_targets=np.array([[1.],[2.],[-1.],[3.],[.5],[-2.]])
gate_queries=np.ones((len(gate_targets),1))
gate_keys=np.ones_like(gate_queries)
gate_decay=np.ones(len(gate_targets))
gate_logit=-2.
gate_losses=[]
for training_step in range(600):
    gate_value=1/(1+np.exp(-gate_logit))
    beta_vector=np.full(len(gate_targets),gate_value)
    prediction,saved=delta_forward_saved(gate_queries,gate_keys,gate_targets,
                                         beta_vector,gate_decay)
    difference=prediction-gate_targets
    gate_losses.append(float(np.mean(difference**2)))
    gradients=delta_backward(gate_queries,gate_keys,gate_targets,beta_vector,
                              gate_decay,saved,2*difference/difference.size)
    logit_gradient=float(gradients[3].sum())*gate_value*(1-gate_value)
    gate_logit-=.1*logit_gradient
learned_beta=1/(1+np.exp(-gate_logit))
assert np.isfinite(gate_losses).all()
assert gate_losses[-1]<gate_losses[0]*.02
assert learned_beta>.9
```

逐行看，门的 sigmoid 输出保持在零到一，beta_vector 把同一个参数用于所有时间步。反向返回每一步 beta 的梯度，因为参数共享，先求和，再乘 sigmoid 导数，得到唯一 logit 参数的梯度。若只取最后一步 beta 梯度，就漏掉其他时间位置对同一参数的贡献。这与共享 embedding、共享专家参数的梯度累加是同一个原则。

训练前后可以手算极端情况。beta 接近零时，状态几乎不写入，输出接近初始零；beta 等于一时，单位 key 每步精确覆盖，输出等于当前目标，损失为零。优化器沿这个可解释目标移动，所以损失下降具有清楚含义。若训练失败，先检查 beta 梯度与参数更新，而不是怀疑数据量不够。

### 十四点一　Delta 是一次在线平方误差更新

固定当前 key 与目标 value，考虑关于状态 S 的损失二分之一乘 value 减 key 转置 S 的平方范数。其对 S 的梯度为负的 key 与误差外积。对 S 做一步梯度下降，步长 beta，正好得到 S 加 beta 乘 key 与误差外积。若先对旧状态衰减，再对这个局部误差做一步更新，就是本章 Delta 的核心形式。

这个解释帮助理解 beta 的作用：它类似当前关联写入的学习率。key 范数影响局部目标曲率，因而同一个 beta 对不同尺度 key 有不同效果。对于单位 key，当前读取误差在更新后乘一减 beta；对于一般 key，误差乘一减 beta 乘 key 范数平方。若这个因子的绝对值大于一，重复写入同一目标可能使误差放大。

例如单位 key、beta 二分之一，初始预测零、目标八，连续写同一个关联，预测依次四、六、七、七点五，误差依次四、二、一、零点五。beta 一直接一步到八；beta 一点五会在目标两侧交替但逐步靠近，因为误差乘负二分之一；beta 三使误差乘负二，震荡幅度越来越大。实际模型常通过门控范围和 key 归一化控制这种行为。

```python
repeat_keys=np.ones((4,1))
repeat_values=np.full((4,1),8.)
half_updates,_=delta_recurrence(repeat_keys,repeat_keys,repeat_values,np.full(4,.5))
np.testing.assert_allclose(half_updates[:,0],[4.,6.,7.,7.5])
oscillating,_=delta_recurrence(repeat_keys,repeat_keys,repeat_values,np.full(4,1.5))
np.testing.assert_allclose(oscillating[:,0],[12.,6.,9.,7.5])
unstable,_=delta_recurrence(repeat_keys,repeat_keys,repeat_values,np.full(4,3.))
np.testing.assert_allclose(unstable[:,0],[24.,-24.,72.,-120.])
```

这里的在线优化解释只针对状态更新，不意味着整个神经网络在推理时更新模型权重。状态是每个请求独立的动态量，模型参数通常保持固定；训练负责学习如何生成 key、value、query 和门。把“快速权重式记忆”误解为每次对全模型做梯度下降，会混淆参数与状态两个时间尺度。

### 十四点二　有限特征 kernel 的秩限制

对一组查询与键，未归一化 kernel 矩阵是特征矩阵 A 与 B 转置的乘积，其秩最多为特征维 m。若序列长度远大于 m，这个矩阵不能表示任意满秩相似度结构。逐行归一化相当于左乘一个对角缩放矩阵，在非零分母下不会提高秩。因此有限特征 attention 的权重结构受到表示维度约束。

这个结论不是说低秩一定不够用，也不是说它的最终深层网络输出只能低秩。多层非线性、门控、残差和输入相关特征可以形成复杂函数。但在一个固定层、固定输入的当前 kernel 矩阵中，秩限制是明确的。它解释了为什么扩大特征维、改变更新规则或使用混合层可能有价值。

标准 softmax 对点积分数逐元素指数再归一化，通常不能简单保持原点积矩阵的低秩结构。非线性可能产生更高秩权重矩阵。因此用有限特征内积精确表示任意 softmax attention 一般需要更强条件或更高维表示。某些方法使用随机或确定特征近似相关 kernel，但近似精度与特征维之间需要分析，不能把本章 ELU 映射说成通用精确替代。

### 十四点三　凸组合性质与 Delta 输出的差别

正 kernel 归一化 attention 的每个输出是历史 value 的非负加权平均，所以落在这些 value 的凸包中。标量 value 时，输出位于历史最小值与最大值之间；向量时，每个坐标也在对应坐标范围内，但更强的关系是整个向量属于凸包。这个性质可以作为实现测试，前提是分母正、没有额外输出投影或门控。

Delta 读取没有同样的概率归一化，输出可以超出历史 value 范围。前面的非单位 key 过冲、非正交关联干扰都展示了这一点。这不自动意味着 Delta 错误，因为它定义的是可更新的线性记忆，而不是凸组合读取。测试不能把 kernel attention 的范围不变量强加给 Delta。

同样，给 kernel 输出再乘一个可学习输出矩阵后，最终坐标可以超出原 value 范围；给 Delta 加归一化或输出门，也会改变幅度行为。讨论不变量时必须明确作用于哪个中间张量。只看最终模型输出，很容易把原本正确的局部性质解释错。

```python
bounded_values=np.array([[-2.],[1.],[5.],[3.]])
bounded_q=rng.normal(size=(4,3))
bounded_k=rng.normal(size=(4,3))
bounded_out=linear_attention_dense(bounded_q,bounded_k,bounded_values)
for t,row in enumerate(bounded_out[:,0]):
    assert bounded_values[:t+1].min()-1e-12<=row<=bounded_values[:t+1].max()+1e-12
```

### 十四点四　位置与短卷积：状态机制之外的输入处理

简单加法状态对最终历史排列不敏感，因此模型可能需要额外方式表达顺序。衰减本身引入时间权重，Delta 更新也通常不交换；输入投影前的短因果卷积可以让当前 key/value 包含最近几个 token 的局部模式；位置特征也可以改变写入和读取。它们是不同机制，加入后要重新检查状态与因果性。

短卷积的状态通常是最近若干输入向量，大小与卷积核宽度有关。即使主记忆矩阵固定，完整层还需要保存这段局部缓存。统计状态容量时不能只数 S 而遗漏卷积缓冲、门控状态或归一化缓存。不同层可能使用不同机制，完整模型账本应逐层相加。

位置旋转也不能随意套到正 kernel 上。RoPE 会产生正负分量，若直接把旋转后的特征当作非负 kernel 特征，分母可能出现抵消甚至为零，前面的概率解释不再成立。可以设计特定位置处理方案，但必须重新定义相似度与归一化；不能因为 softmax attention 使用 RoPE，就假设所有递推记忆都能直接复用相同做法。

### 十四点五　输入相关门与语言模型层的完整接口

一个实际层通常先从隐藏输入生成多组投影：query、key、value、遗忘门、写入门和输出门。它们可能共享输入，但参数不同。query 决定读什么，key 决定写到哪个方向，value 决定写入内容，遗忘门控制旧状态，写入门控制修正强度，输出门控制读取结果怎样进入残差。把这些向量都叫“门控”会丢失职责差异。

训练时所有支路对隐藏输入的梯度都要累加；推理时每个请求保存自己的状态。输出通常还经过归一化与线性投影，再加回残差。若只实现核心 S 更新，就应明确它是一个记忆算子，而不是完整 block。反过来，理解核心后再读完整 block，你可以把每一条投影对应到具体职责，避免只看见很多线性层而失去主线。

门的参数化也影响初始化。若遗忘门初始接近零，状态很快清空，模型难以利用长历史；若接近一，保留更久但干扰与数值积累可能更明显。写入门过小会学习缓慢，过大可能过冲。完整训练配方会考虑这些问题，本章不凭一个默认 sigmoid 偏置宣称适用于所有模型。

## 第十五课　渐进项目：建立一个记忆机制比较台

第一阶段实现三种核心：正 kernel 加法记忆、带一致分母衰减的 kernel 记忆、Delta 纠错记忆。每种都写明输入映射、状态 shape、更新顺序、读取规则与门范围。不要把三个函数统一到一个模糊的 state 参数后隐藏差异；公共接口可以统一，但配置与返回状态类型必须能区分。

第二阶段设计四个小任务。第一是固定 key 反复更新 value，观察覆盖与累加差异；第二是正交 key 关联存储，观察独立方向；第三是不正交 key 干扰，量化旧关联误差；第四是写入后插入无关记录再查询，观察保留与遗忘。每个任务都使用可手算的小维度，再扩展到随机维度。

第三阶段建立实现对照。kernel 使用显式矩阵、逐步递推、分块延续和前缀和四条路径；Delta 使用逐步更新和仿射组合参考。比较输出与最终状态，不只比较最后 argmax。对每条路径记录输入 shape、dtype、最大误差与是否修改调用者状态。错误示例包括遗漏 z、先读后写、错误衰减顺序和未归一化 key 的覆盖误判。

第四阶段加入训练。先运行第十四课共享 beta 学习实验，再让 beta 由输入的一个小线性层产生，反向时继续通过 sigmoid 和线性层。随后可以学习 key/query 投影，但要加入范数控制并保留梯度差分。每次只增加一层复杂度，先验证梯度，再观察学习效果，避免同时改变状态规则与优化器而无法定位原因。

第五阶段连接到完整解码器。保留 embedding、残差、FFN 与输出头，把某些 attention 层换成你实现的记忆层。所有层都用同一种机制是一种模型，交替使用全 attention 与固定状态层是另一种模型。测试应逐层确认状态类型，并验证 full 与 step 的一致性。混合层的缓存总量由两类状态共同决定，不能只报其中较小的一类。

### 十五点一　任务设计要区分读写时序

关联记忆任务常有“写入 token”和“查询 token”两种角色。若查询步骤仍自动把当前 key/value 写入状态，可能在读取前覆盖了你想测的旧关联。为了测纯读取，可以把写入门设零，或定义专门的读操作；若使用包含当前的 kernel attention，也要明确当前 value 是否参与平均。任务接口必须防止答案被当前输入直接泄漏。

例如先写 key A 对应三，再在最后查询 A，同时把 value 三也作为查询步骤输入并允许写入，模型可能无需记住历史就能答对。一个有效记忆测试应在查询时不给答案，或者关闭写入。这个问题与语言模型标签错位、RAG 答案泄漏同属实验设计问题，不能靠更复杂模型补救。

对顺序任务也应避免标签只依赖集合统计。若目标是所有 value 的平均，加法状态天然足够；它在这个任务上表现好，不能证明它记住了顺序。要测试顺序，可以要求最近一次写入、指定位置值或交换顺序后答案改变。选择任务时先写清需要保留的信息，再判断状态机制是否理论上可能表达。

### 十五点二　状态恢复、分支与历史编辑

固定状态很小，保存快照相对便宜，但无法从最终 S/z 一般性地恢复每个原始 token。对纯加法状态，如果你仍保存被删除记录的精确外积和 key 特征，可以做减法回滚；浮点误差与重复操作仍需注意。若这些原始写入项已经丢失，单靠汇总状态不能知道该减什么。

带衰减或 Delta 的回滚更复杂。某些更新矩阵可能不可逆，例如单位 key、beta 一的精确覆盖会消除旧状态在该方向的信息。不能指望从覆盖后的状态恢复被覆盖值。实用方案通常保存边界快照或从较早前缀重算，而不是假设固定状态天然支持任意撤销。

历史中间 token 被修改后，从该位置开始的状态通常都需重算。即使状态大小固定，依赖仍跨越全部后续步骤。服务层支持编辑对话时，应定位最长不变前缀并恢复对应快照，或者重新处理全部输入。固定内存改变资源成本，不改变因果依赖的基本逻辑。

### 十五点三　batch、padding 与无效 token 的写入

多请求 batch 中每个请求拥有独立 S/z 或 Delta 状态，不能在 batch 轴上相加。padding token 也不应默认写入记忆，除非模型训练明确定义如此。若为了对齐长度加入 padding，写入 mask 应阻止无效位置更新状态，读取输出也可以按损失 mask 忽略。只在最终 loss 上忽略 padding，却允许它改变后续有效状态，可能产生错误。

左 padding 与右 padding 的影响不同。右 padding 若后面没有有效 token，可能只影响无用尾状态；但如果这个状态被拿来继续生成，padding 已经污染历史。左 padding 会在所有有效 token 之前写入，影响更直接。正确做法是在状态更新处使用有效位置条件，而不是希望 padding embedding 恰好为零，因为投影 bias、特征映射和门控仍可能产生非零写入。

chunked prefill 中每个请求的有效 token 数不同，状态更新次数也不同。调度把多个请求拼成一个批次，不代表它们共享时间索引或归一化统计。完整接口应传入每请求有效长度与状态句柄，测试交错执行与独立执行一致。这与 KV cache 的隔离原则相同，只是状态张量换成固定矩阵。

### 十五点四　资源比较必须把训练与推理分开

逐 token 推理只需当前状态与当前投影，状态容量不随历史长度增长。完整反向训练可能需要保存每步状态、门与中间量，或者采用重计算和专门反向算法。不能用推理状态公式直接估计训练显存。前缀和参考物化所有外积和状态，也可能在长序列上占用大量内存，尽管它展示的是线性时间算法。

计算成本也有维度因素。普通单步 attention 主要随历史长度 T 乘头维 d 增长；矩阵状态更新随 key 维乘 value 维增长。短序列时后者可能更贵，长序列时优势可能出现。具体交叉点还受内存带宽、矩阵指令利用率、门控和投影成本影响，不能只比较大 O。

个人电脑可以完成本章全部数学、梯度与小训练实验。若进行 GPU 基准，应固定模型维度与精度，分别测试 prefill 和 decode，记录 warmup、同步方式与状态分配是否计时。服务器实验应限制长度与步数，先验证结果再扩大规模。没有运行过的设备结果应留空，而不是按理论比例填写。

### 十五点五　阅读官方实现时的检查顺序

先找核心更新对应的变量与方向：状态是 key 乘 value 还是 value 乘 key，query 在左乘还是右乘，key 是否归一化，遗忘发生在哪一步。再看输入投影与短卷积，确认它们的缓存和因果范围。接着看门参数化、输出归一化与残差。最后看 chunk 或 scan 内核如何实现同一公式。

不要把不同实现的状态直接互换。一个采用转置状态的内核可能 shape 看起来相近，但乘法方向不同；一个把门吸收到 key 的实现，输入尺度与本章显式 beta 不同；一个先归一化再投影的模型，与投影后归一化也不是同一函数。移植时应建立变量映射表，并用微型配置比较每个中间量。

本章的正 kernel 推导可与 [Linear Transformers 原论文](https://arxiv.org/abs/2006.16236) 对照。Delta 教学更新不宣称复现任何完整新模型；阅读具名架构时，必须把其新增门控、卷积、归一化、训练与内核逐项补齐。你已经具备的能力应是推导与验证核心机制，而不是给所有循环记忆套同一个名称。

### 十五点六　带有效位置 mask 的完整 batch 状态参考

下面实现一个小型 batch 版本，输入 shape 为 batch、时间、特征，状态为 batch、key 特征、value 特征与 batch、key 特征。每个请求独立更新。无效位置既不写入，也不产生有效输出；我们把其输出设零作为明确约定。循环写法便于检查，不追求速度。

```python
def batched_kernel_attention(q,k,v,valid):
    if q.ndim!=3 or k.shape!=q.shape or v.shape[:2]!=q.shape[:2]:
        raise ValueError("batch/time shape mismatch")
    if valid.shape!=q.shape[:2]:
        raise ValueError("mask shape mismatch")
    batch,time,feature_dim=q.shape
    value_dim=v.shape[-1]
    state=np.zeros((batch,feature_dim,value_dim))
    normalizer=np.zeros((batch,feature_dim))
    output=np.zeros((batch,time,value_dim))
    aq,bk=feature_map(q),feature_map(k)
    for t in range(time):
        for b in range(batch):
            if not valid[b,t]:
                continue
            state[b]+=np.outer(bk[b,t],v[b,t])
            normalizer[b]+=bk[b,t]
            denominator=float(aq[b,t]@normalizer[b])
            if denominator>0:
                output[b,t]=(aq[b,t]@state[b])/denominator
    return output,(state,normalizer)

batch_q=rng.normal(size=(2,5,3))
batch_k=rng.normal(size=(2,5,3))
batch_v=rng.normal(size=(2,5,2))
valid=np.array([[0,0,1,1,1],[1,1,1,1,1]],dtype=bool)
batch_out,batch_state=batched_kernel_attention(batch_q,batch_k,batch_v,valid)
for b in range(2):
    indices=np.flatnonzero(valid[b])
    independent,independent_state=linear_attention_recurrent(
        batch_q[b,indices],batch_k[b,indices],batch_v[b,indices])
    np.testing.assert_allclose(batch_out[b,indices],independent,atol=1e-12)
    np.testing.assert_allclose(batch_state[0][b],independent_state[0],atol=1e-12)
    np.testing.assert_allclose(batch_state[1][b],independent_state[1],atol=1e-12)
np.testing.assert_array_equal(batch_out[0,:2],np.zeros((2,2)))
```

这个测试把第一条请求的两个左 padding 完全跳过，再与删除 padding 后独立运行比较。若你只把 padding 的 value 设零，却仍把它的 key 特征加到 z，分母会增大，结果不相等。即使原始 padding key 是零，ELU 加一后也变成一，因此“零输入不会影响状态”的直觉在这里明确失效。

在某些模型中，位置索引与有效 token 数有关，跳过 padding 后还需要正确生成位置。当前 kernel 没有额外位置编码，所以测试只涉及写入 mask；加入位置机制后应扩展验证。完整模型测试必须覆盖每个实际使用的状态，而不能因为核心算子通过就忽略外围处理。

### 十五点七　用不变量设计更强的测试

除了与参考数值比较，还可以利用数学不变量。正 kernel 的 value 若全部等于同一个向量，所有有正分母的输出都应等于该向量，因为归一化权重和为一。将所有 value 同时加同一个偏移，输出也应加同一偏移；将所有 value 乘常数，输出同样乘常数。这些性质来自对 V 的线性与权重归一化，能抓住分母遗漏和广播错误。

```python
constant_value=np.tile(np.array([2.,-3.]),(len(q_seq),1))
constant_out=linear_attention_dense(q_seq,k_seq,constant_value)
np.testing.assert_allclose(constant_out,constant_value)
base_out=linear_attention_dense(q_seq,k_seq,v_seq)
shift=np.array([1.,-2.,.5])
shifted=linear_attention_dense(q_seq,k_seq,v_seq+shift)
np.testing.assert_allclose(shifted,base_out+shift,atol=1e-12)
scaled=linear_attention_dense(q_seq,k_seq,3*v_seq)
np.testing.assert_allclose(scaled,3*base_out,atol=1e-12)
```

这些不变量不应机械套到 Delta。Delta 对 value 在固定 key、query、门的条件下仍是线性递推，但加一个常量 value 不一定让输出加同一个常量，因为没有归一化概率和为一的保证。对每个机制先推导性质，再写测试，能避免把错误期望当成实现 bug。

还可以测试状态复制与分支：从同一个前缀状态分别续写两个不同后缀，旧状态保持不变，两条结果分别与完整序列一致。对固定状态机制，这个实验比 KV 更便宜，但语义同样重要。性能优化若改成原地更新，应同步修改接口和调用者，不能让原来通过的分支测试悄悄失效。

### 十五点八　把实验结论写到它真正支持的范围

本章实际运行的门训练说明一个共享 beta 参数可以通过跨时间梯度学会更强覆盖；它不说明完整语言模型中的最优 beta 总接近一。若任务需要保留旧关联、抗噪声或缓慢更新，较小 beta 可能更合适。训练目标决定行为，不能把一个合成任务的结果当成所有应用的推荐值。

同样，正交 key 实验说明在理想方向分离下可以独立写入，不说明自然语言隐藏向量天然正交。非正交干扰实验说明有限状态存在冲突，不说明所有线性记忆都无法处理长文本。正确的工程与研究写作应保留这些条件，让读者知道结论来自什么输入、什么公式与什么测试。

完成本章后，你应能区分历史列表、归一化统计状态和纠错式矩阵记忆，能从公式写出前向与反向，能解释固定状态的成本与信息限制，也能把一个新实现拆成投影、写入、读取、门控和执行算法。下一章会从连续时间系统出发，推导另一类固定状态模型，并比较它们何时等价于卷积、何时因输入选择而失去固定卷积形式。

再补一个容易误判的量：状态范数增长不一定等于归一化输出爆炸。简单 kernel 累加中，重复写入相同 key/value 会让 S 与 z 同比例增长，输出保持相同；但有限精度仍可能因巨大中间量失去精度。因此应同时监控状态范数、分母范围与输出范数，不能只凭其中一个指标判断稳定性。

相反，Delta 状态不带配套分母，状态方向与 query 尺度直接影响输出。即使状态范数不大，一个很大的 query 也可能产生大输出；即使 query 归一化，多个 value 方向仍可能累积。调试日志应记录读写两侧的尺度，而不是只给状态做裁剪后停止分析。

若使用状态裁剪作为实验措施，必须说明它改变了递推函数，并重新检查训练与推理一致性。把一个有裁剪的训练模型换成无裁剪推理，或者反过来，都可能产生长期轨迹差异。状态模型的数值处理会沿时间传播，配置版本和复现实验应把这些规则与权重一同保存。

保存状态时还要记录特征维与值维的方向约定；把转置后的矩阵误读为原布局，在维度恰好相等时不会触发 shape 异常，却会改变读取函数。
