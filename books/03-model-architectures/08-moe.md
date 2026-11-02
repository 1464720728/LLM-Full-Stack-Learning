# M08　混合专家 MoE：路由、容量、分发、合并与训练梯度

[本卷导航](README.md) · 前置：[低秩注意力](07-latent-attention.md) · 下一章：[稀疏注意力](09-sparse-attention.md)

稠密解码器的每个 token 都经过同一组 FFN 参数。若想让模型拥有更多参数，却不让每个 token 都执行全部参数，一种办法是准备多个专家，只为当前 token 选择其中少数几个。本章不把这个想法停在“稀疏激活”四个字，而是写出从 router logits 到专家结果回填的完整数据流，手算容量溢出，推导可微与不可微部分，并把 MoE 放回完整解码器。

这里的专家是神经网络子模块，通常不是一个独立聊天机器人，也不是 Agent 团队。路由是张量计算中的条件选择，不是让多个模型开会。我们使用 token-choice top-k 和 ReLU 两层专家作为清晰基准；真实模型可能使用共享专家、细粒度专家、不同打分函数、不同负载策略和分布式实现。讨论这些变体时必须保留各自定义，不能把一个简单 toy 声称成任意具名模型。

## 第一课　总参数、激活参数与真实计算量

设一个专家包含上投影 $W_{up}\in\mathbb R^{D\times F}$ 和下投影 $W_{down}\in\mathbb R^{F\times D}$，忽略 bias，参数量为 $2DF$。如果有 E 个专家，总专家参数为 $2EDF$。每个 token 只选 k 个专家，理论上执行的专家矩阵参数规模约为 $2kDF$，再加路由器和其他非专家模块。E 与 k 是不同的设计轴，前者主要扩展总容量，后者直接影响每 token 的专家计算。

“激活参数”并不等于实际浮点运算数，更不等于整个模型显存。所有专家权重可能仍需驻留或分布在设备上；路由需要计算；分发和回收需要搬运；小批量专家矩阵乘法可能利用率较低；不均衡负载可能让少数设备成为瓶颈。因此总参数更大而激活较少，说明了结构目标，不自动给出确定的速度优势。

用一个具体账本说明：隐藏维度八、中间维度十六，一个 ReLU 专家有二百五十六个参数；四个专家有一千零二十四个。top-2 每 token 执行两个专家，专家乘加约五百一十二次，另有八乘四的 router 计算。若比较一个中间维度三十二的稠密 FFN，它也有五百一十二个参数和相近的矩阵乘加，但总参数容量不同。公平比较需要明确你控制的是总参数、激活计算还是训练预算。

注意力稀疏与专家稀疏也要分开。MoE 通常改变 FFN 的参数选择，attention 仍可能是完整因果注意力；稀疏 attention 改变 token 之间的连接。一个模型可以同时采用二者，也可以只采用其一。把“稀疏模型”当成唯一机制，会让你无法判断缓存、计算和通信分别发生了什么变化。

## 第二课　路由器：从隐藏向量到专家概率

把一个 batch 的所有有效 token 展平为 $X\in\mathbb R^{N\times D}$，N 是 token 数，不一定是序列长度。路由器矩阵 $W_R\in\mathbb R^{D\times E}$ 产生 logits $Z=XW_R$，shape 为 $[N,E]$。对每个 token 沿专家轴做 softmax，得到 $p_{t,e}$。同一行概率非负、和为一，表示该 token 对各专家的相对偏好。

这里的概率不是每个专家实际收到 token 的比例。实际分配还要经过 top-k、容量限制和调度。一个专家的平均概率很高，可能被很多 token 选中；但若容量满了，实际接收数会受限。后面负载均衡会同时观察软概率与硬分配，正是因为它们描述不同阶段。

top-k 为每行选择 k 个专家编号。若分数相同，必须定义确定的 tie-break，才能复现。课程实现使用稳定排序，因此同分时较小编号先被选。这个选择本身不是模型能力的核心，但会影响小样例、负载计数和测试。使用不同库的 top-k 时，不能默认并列顺序完全一致。

选中后常见做法是把所选概率重新归一化：
$$
g_{t,e}=\frac{p_{t,e}}{\sum_{j\in S_t}p_{t,j}},\quad e\in S_t.
$$
这样选中的权重和为一。也有模型使用不同归一化、额外缩放或直接保留原概率，必须按定义实现。本章采用重新归一化，且容量丢弃之后不再次归一化；这个细节后面会完整手算。

```python
import numpy as np
from labs.models.core import softmax,moe,load_balance_loss

def route_tokens(x,router,top_k):
    logits=x@router
    probabilities=softmax(logits)
    chosen=np.argsort(-probabilities,axis=1,kind="stable")[:,:top_k]
    selected=np.take_along_axis(probabilities,chosen,axis=1)
    weights=selected/selected.sum(axis=1,keepdims=True)
    return logits,probabilities,chosen,weights

x_hand=np.eye(3)
router_hand=np.log(np.array([[4.,2.,1.],[1.,3.,2.],[2.,1.,2.]]))
z_hand,p_hand,chosen_hand,g_hand=route_tokens(x_hand,router_hand,2)
np.testing.assert_array_equal(chosen_hand,[[0,1],[1,2],[0,2]])
np.testing.assert_allclose(p_hand,[[4/7,2/7,1/7],[1/6,3/6,2/6],[2/5,1/5,2/5]])
np.testing.assert_allclose(g_hand,[[2/3,1/3],[3/5,2/5],[1/2,1/2]])
```

代码用单位矩阵作为 X，所以每个 token 恰好读取 router 的一行。这是为了让手算可追踪，不是实际模型的输入形式。取对数作为 logits，使 softmax 后得到简单整数比例：第一行指数四、二、一，除以七；第二行一、三、二，除以六；第三行二、一、二，除以五。稳定排序让第三行并列的专家零排在专家二之前。

## 第三课　专家网络与加权合并：完整手算三个 token

本章专家定义为 $f_e(x)=\operatorname{ReLU}(xW_{up,e})W_{down,e}$。它与稠密 FFN 的基本结构相同，只是有多套参数并按 token 选择。为了手算，令所有上投影为三维单位矩阵，下投影分别为二、四、八倍单位矩阵。输入是非负 one-hot，因此 ReLU 不改变结果，三个专家分别把输入放大二、四、八倍。

第一个 token 选专家零和一，权重三分之二与三分之一，输出倍率为 $2\times2/3+4\times1/3=8/3$。第二个 token 选专家一和二，权重为五分之三（`3/5`）与五分之二（`2/5`），输出倍率为 $4\times3/5+8\times2/5=28/5$。第三个 token 选专家零和二，各二分之一，输出倍率为一加四等于五。因为输入分别是三个坐标方向，最终输出是对角矩阵，对角元素为八除三、二十八除五和五。

专家结果必须回到原 token 的位置再相加。若为了高效把所有 token 按专家分组，计算结果的行顺序会改变，需要保存原 token 索引。不能把专家零的第一行输出直接当成原 batch 第一行，除非分发映射恰好如此。dispatch 与 combine 的主要任务就是管理这个对应关系。

```python
experts_hand=[(np.eye(3),scale*np.eye(3)) for scale in (2.,4.,8.)]
out_hand,audit_hand=moe(x_hand,router_hand,experts_hand,top_k=2)
np.testing.assert_allclose(out_hand,np.diag([8/3,28/5,5.]))
np.testing.assert_array_equal(audit_hand["counts"],[2,2,2])
assert not audit_hand["dropped"].any()
```

输出合并后通常作为残差增量加回隐藏状态，而不是取代整个模型状态。若一个 token 的所有专家分配都被丢弃，MoE 增量可以为零，但残差主干仍保留原输入。这不表示丢弃没有影响：模型失去了本层专家变换，训练和质量都会受到影响。不能用“还有残差”把高丢弃率视为无关紧要。

## 第四课　容量限制：必须说清按什么顺序丢弃

假设每个专家最多接收 C 个 token 分配。token-choice top-k 一共提出 $Nk$ 个分配请求，平均每专家约 $Nk/E$。常见容量公式会乘一个 capacity factor 再向上取整，但不同实现对分组、batch、训练与推理的计数范围可能不同。本章直接传整数 capacity，避免把未定义的公式藏在参数名里。

课程参考按 token 顺序遍历，再按该 token 的 top-k 槽位顺序遍历。专家已接收数达到容量后，后来的分配被丢弃。这个规则简单且确定，但具有顺序偏置：前面的 token 更容易被接收。其他实现可能按门控权重排序或采用不同策略，它们不是同一函数，应单独测试。容量限制不是一个只有数值没有语义的配置项。

把第三课的三个 token 设置每专家容量一。第一个 token 的专家零和一都被接收，输出仍为八除三乘第一坐标。第二个 token 想去专家一和二；专家一已满，被丢弃，专家二接收，保留权重五分之二（`2/5`），因此输出倍率为 $8\times2/5=16/5$。第三个 token 想去专家零和二，两者都满，全部丢弃，输出为零。

这里没有在丢弃后重新归一化。若重新归一化，第二个 token 唯一存活专家的权重会变成一，输出倍率八，而不是十六除五。两种策略都能定义，但必须明确选择。课程 audit 返回的是容量处理前的选中权重，以及单独的 dropped 布尔矩阵，读日志时要把两者结合，不能直接把 weights 当成实际全部贡献。

```python
capacity_out,capacity_audit=moe(x_hand,router_hand,experts_hand,top_k=2,capacity=1)
np.testing.assert_allclose(capacity_out,np.diag([8/3,16/5,0.]))
np.testing.assert_array_equal(capacity_audit["counts"],[1,1,1])
np.testing.assert_array_equal(capacity_audit["dropped"],
                             [[False,False],[True,False],[True,True]])
assert capacity_audit["dropped"].sum()==3
```

计数 counts 表示实际接受的分配数，不是最初被选中的次数。最初每个专家都被选两次，容量后每个只接受一次。报告负载时应至少区分 proposed、accepted 和 dropped 三种统计。仅看 accepted 全部相等可能误以为负载完美，而实际上有一半分配被丢弃，容量把不均衡或过载掩盖了。

## 第五课　dispatch 与 combine：把散乱选择变成专家批次

逐 token 调用专家最清楚，却会产生大量小矩阵乘法。更合理的执行是先收集每个专家收到的 token，形成一个批次矩阵，一次计算该专家，再根据原索引和门控权重回填。每条分配需要记录 token 编号、top-k 槽位、专家编号和门控权重；容量策略决定哪些记录保留。

下面给出完整分发实现，与课程参考保持相同的遍历顺序和容量语义。buckets 保存每个专家的 token 与槽位。计算专家后，结果按 token 索引累加。这里使用显式循环保证重复 token 的多个专家贡献全部加到同一行；如果改成向量化 fancy indexing 的加法，要注意重复索引的累加语义。

```python
def dispatched_moe(x,router,experts,top_k=2,capacity=None):
    _,probabilities,chosen,weights=route_tokens(x,router,top_k)
    buckets=[[] for _ in experts]
    dropped=np.zeros_like(chosen,dtype=bool)
    for token in range(len(x)):
        for slot,expert in enumerate(chosen[token]):
            expert=int(expert)
            if capacity is not None and len(buckets[expert])>=capacity:
                dropped[token,slot]=True
            else:
                buckets[expert].append((token,slot))
    result=np.zeros_like(x,dtype=float)
    for expert,records in enumerate(buckets):
        if not records:
            continue
        indices=np.array([token for token,slot in records])
        up,down=experts[expert]
        batch=x[indices]
        expert_outputs=np.maximum(batch@up,0)@down
        for row,(token,slot) in zip(expert_outputs,records):
            result[token]+=weights[token,slot]*row
    counts=np.array([len(records) for records in buckets])
    return result,{"chosen":chosen,"weights":weights,"probabilities":probabilities,
                   "counts":counts,"dropped":dropped,"buckets":buckets}

for cap in (None,0,1,2):
    reference,reference_audit=moe(x_hand,router_hand,experts_hand,2,cap)
    grouped,grouped_audit=dispatched_moe(x_hand,router_hand,experts_hand,2,cap)
    np.testing.assert_allclose(grouped,reference)
    np.testing.assert_array_equal(grouped_audit["dropped"],reference_audit["dropped"])
    np.testing.assert_array_equal(grouped_audit["counts"],reference_audit["counts"])
```

capacity 为零时所有分配都被丢弃，结果为零，计数也为零。这个边界应显式测试，因为空专家批次很常见：即使容量非零，也可能没有 token 选择某个专家。不能对空列表直接 stack 而让运行器崩溃，也不能为了凑 batch 给专家加入未定义的假 token 后忘记屏蔽输出。

真实高性能实现可能把分发表示成排序索引、前缀和、压缩槽位或专门的稀疏矩阵。无论布局如何，逻辑上都必须保持 token 到专家的映射，并在 combine 时恢复对应关系。先掌握这个 CPU 参考，再阅读融合 dispatch 内核，会更容易看懂那些看似复杂的索引计算究竟在维护什么。

## 第六课　路由梯度：离散选择与连续权重是两件事

设一个 token 选中的专家集合为 S，输出为 $y=\sum_{e\in S}g_e f_e(x)$。只要 logits 的小扰动没有改变 top-k 集合，S 可以在这个局部区域内视为固定。门控权重 g 仍是连续可微的，专家网络也是分段可微的。因此 task loss 可以通过门控权重训练 router，也可以通过被选中的专家训练专家参数。但选择集合发生切换的边界不具有普通平滑导数。

本章采用对选中概率重新归一化。因为 $p_e=\exp z_e/\sum_j\exp z_j$，代入后公共分母消掉，得到 $g_e=\exp z_e/\sum_{j\in S}\exp z_j$。也就是说，在固定 S 的局部区域，g 等于只在选中 logits 上做 softmax。未选专家 logits 对这个局部 task loss 的导数为零，不是因为它们的原始 softmax 概率为零，而是因为重新归一化恰好消除了公共分母。

设上游梯度为向量 u，门控权重的梯度为 $a_e=u^\top f_e(x)$。选中 logit 的梯度为
$$
\frac{\partial L}{\partial z_e}
=g_e\left(a_e-\sum_{j\in S}g_ja_j\right).
$$
这个公式与 softmax Jacobian 相同，只是归一化范围变成选中集合。对 router 矩阵，若 z 来自 xWr，则梯度为 x 的列向量与 dz 的外积。多个 token 的贡献相加，得到 $X^\top dZ$。

top-1 是一个必须认真理解的退化情形。选中后重新归一化，唯一门控权重恒等于一，所以在选择不变的局部区域内，task loss 对 router logits 的梯度为零。专家仍能训练，但 router 无法通过这个门控权重路径获得普通任务梯度。不同 MoE 配方会采用不同门控与辅助机制，不能以本章的 top-2 推导无条件解释所有 top-1 模型。

容量丢弃又增加一个离散边界。若某选中专家被丢弃，它的实际输出贡献为零；但本章不在丢弃后归一化，原门控权重仍来自所有选中专家。计算 task loss 梯度时，可以把被丢弃专家的 f 视为零，再对原 g 求导。这样被丢弃槽位仍可能通过归一化竞争影响其他存活权重，不能简单把它的 logit 梯度也强行清零。具体训练实现必须与前向定义一致。

```python
def gate_loss_and_gradient(logits,expert_values,top_k,upstream):
    chosen=np.argsort(-logits,kind="stable")[:top_k]
    g=softmax(logits[chosen][None,:])[0]
    selected=expert_values[chosen]
    output=g@selected
    gate_gradient=selected@upstream
    dz_selected=g*(gate_gradient-np.dot(g,gate_gradient))
    dz=np.zeros_like(logits)
    dz[chosen]=dz_selected
    return float(output@upstream),dz,chosen

gate_logits=np.array([2.,0.7,-1.])
expert_values=np.array([[1.,2.],[4.,-1.],[-2.,3.]])
upstream=np.array([0.5,-0.2])
loss_gate,analytic_gate,selected_gate=gate_loss_and_gradient(
    gate_logits,expert_values,2,upstream)
numerical_gate=np.zeros_like(gate_logits)
eps=1e-6
for i in range(len(gate_logits)):
    plus=gate_logits.copy();plus[i]+=eps
    minus=gate_logits.copy();minus[i]-=eps
    numerical_gate[i]=(gate_loss_and_gradient(plus,expert_values,2,upstream)[0]
                       -gate_loss_and_gradient(minus,expert_values,2,upstream)[0])/(2*eps)
np.testing.assert_allclose(analytic_gate,numerical_gate,atol=1e-8)
assert analytic_gate[2]==0.
_,top1_gradient,_=gate_loss_and_gradient(gate_logits,expert_values,1,upstream)
np.testing.assert_allclose(top1_gradient,0.,atol=1e-12)
```

测试的 logits 明确分离，微小扰动不会改变选择。若把第二与第三项设成几乎相等，中心差分可能跨越 top-k 边界，得到不代表单侧局部导数的结果。自动微分也不会神奇地为离散切换提供普通导数；它通常对当前选中路径求导。理解这一点，才能正确解释 router 训练的局限与辅助损失的作用。

## 第七课　专家反向：只更新被实际执行的路径

对一个专家，令 $A=X_eW_{up}$、$H=\operatorname{ReLU}(A)$、$Y_e=HW_{down}$。若上游专家输出梯度为 G，则 $dW_{down}=H^\top G$，$dH=GW_{down}^\top$，$dA=dH\odot[A>0]$，$dW_{up}=X_e^\top dA$，$dX_e=dAW_{up}^\top$。这些公式与普通 FFN 完全相同，MoE 增加的是分发与门控权重。

如果原 token 的 MoE 输出梯度为 u，而该分配的门控权重为 g，则送入专家输出的梯度为 gu。一个 token 被多个专家处理，它收到的输入梯度要累加多个专家路径，还要加上 router 对输入的梯度。若 x 同时通过残差直接传递，上游梯度还有一条恒等支路。只计算专家内部反向而忘记路由与残差，仍然不是完整 MoE 层的反向。

下面先给出单个专家的完整反向和数值验证，避免把所有复杂度一次混在一起。输入与参数选择使 ReLU 远离零点，中心差分不会跨过激活拐点。每个参数元素都被检查，而不是只看梯度范数非零。

```python
def expert_forward(x,up,down):
    pre=x@up
    hidden=np.maximum(pre,0)
    return hidden@down,(x,up,down,pre,hidden)

def expert_backward(saved,gradient):
    x,up,down,pre,hidden=saved
    ddown=hidden.T@gradient
    dhidden=gradient@down.T
    dpre=dhidden*(pre>0)
    dup=x.T@dpre
    dx=dpre@up.T
    return dx,dup,ddown

rng=np.random.default_rng(8)
ex_x=np.array([[1.,2.],[2.,1.]])
ex_up=np.array([[0.4,-0.7,0.2],[0.3,0.1,-0.8]])
ex_down=rng.normal(size=(3,2))
ex_gradient=rng.normal(size=(2,2))
ex_out,ex_saved=expert_forward(ex_x,ex_up,ex_down)
ex_analytic=expert_backward(ex_saved,ex_gradient)
ex_parameters=[ex_x,ex_up,ex_down]
for parameter,expected in zip(ex_parameters,ex_analytic):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=float(np.sum(expert_forward(*ex_parameters)[0]*ex_gradient))
        parameter[index]=old-1e-6
        minus=float(np.sum(expert_forward(*ex_parameters)[0]*ex_gradient))
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=1e-7,rtol=1e-7)
```

一个从未被选中的专家，普通 task loss 不会通过其输出更新参数。它可能长期没有训练信号，进一步变得不擅长任何输入，形成“越不选越学不到，越学不到越不选”的循环。负载均衡、初始化和训练策略的动机之一，就是避免这种路由塌缩。不能仅凭总参数很多就假设每个专家都得到了有效训练。

## 第八课　完整 MoE 局部反向：把所有梯度接回去

现在实现无容量丢弃、固定 top-k 局部区域下的完整反向。先执行每个 token 的选中专家，保存各专家输出和中间激活；输出加权合并。反向时分别计算门控梯度、专家参数梯度、专家输入梯度，再通过 router 计算输入和参数梯度。专家参数跨 token 共享，所以使用加法累计，不能每遇到一个 token 就覆盖一次。

```python
def moe_forward_saved(x,router,experts,top_k):
    _,probs,chosen,weights=route_tokens(x,router,top_k)
    output=np.zeros_like(x,dtype=float)
    records=[]
    for t in range(len(x)):
        token_records=[]
        for slot,e in enumerate(chosen[t]):
            value,saved=expert_forward(x[t:t+1],*experts[e])
            output[t]+=weights[t,slot]*value[0]
            token_records.append((int(e),value[0],saved))
        records.append(token_records)
    return output,(x,router,experts,chosen,weights,records)

def moe_backward_saved(saved,upstream):
    x,router,experts,chosen,weights,records=saved
    dx=np.zeros_like(x)
    dz=np.zeros((len(x),len(experts)))
    expert_grads=[(np.zeros_like(up),np.zeros_like(down)) for up,down in experts]
    for t,token_records in enumerate(records):
        dg=np.array([value@upstream[t] for e,value,record in token_records])
        selected_dz=weights[t]*(dg-np.dot(weights[t],dg))
        dz[t,chosen[t]]=selected_dz
        for slot,(e,value,record) in enumerate(token_records):
            local_dx,local_up,local_down=expert_backward(
                record,(weights[t,slot]*upstream[t])[None,:])
            dx[t]+=local_dx[0]
            expert_grads[e][0][...]+=local_up
            expert_grads[e][1][...]+=local_down
    drouter=x.T@dz
    dx+=dz@router.T
    return dx,drouter,expert_grads

train_x=np.array([[1.,0.2],[0.3,1.]])
train_router=np.array([[1.8,0.4,-1.],[0.2,1.4,-0.7]])
train_experts=[
    (np.array([[.4,.7],[.2,.3]]),rng.normal(size=(2,2)))
    for _ in range(3)]
train_upstream=rng.normal(size=(2,2))
train_out,train_saved=moe_forward_saved(train_x,train_router,train_experts,2)
dx_train,dr_train,de_train=moe_backward_saved(train_saved,train_upstream)
parameter_pairs=[(train_x,dx_train),(train_router,dr_train)]
for (up,down),(dup,ddown) in zip(train_experts,de_train):
    parameter_pairs.extend([(up,dup),(down,ddown)])
for parameter,expected in parameter_pairs:
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=float(np.sum(moe_forward_saved(train_x,train_router,train_experts,2)[0]
                          *train_upstream))
        parameter[index]=old-1e-6
        minus=float(np.sum(moe_forward_saved(train_x,train_router,train_experts,2)[0]
                           *train_upstream))
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=2e-6,rtol=2e-6)
```

这份代码没有实现容量丢弃的反向，也没有假装支持所有路由变体。它完整实现并验证了自己声明的局部函数：top-k 集合固定、选中权重重新归一化、ReLU 专家、无丢弃。要扩展容量，先按第四课的前向语义决定哪些专家输出视为零，再检查 gate 梯度与专家梯度是否对应。要扩展共享专家，则另加一条始终执行的专家支路及其梯度。

数值差分通过意味着在这些非边界点导数正确，不意味着训练必然稳定或任务质量好。训练还依赖数据、初始化、学习率、路由分布和负载策略。把实现验证与训练效果分开，是避免把一次 loss 下降当作全部正确性的关键。

## 第九课　负载均衡：软概率与硬分配的不同角色

定义每个专家的平均路由概率 $P_e=\frac1N\sum_t p_{t,e}$，硬分配比例 $f_e$ 为所有 top-k 分配中属于该专家的比例。本章简单辅助量为 $L_{balance}=E\sum_eP_ef_e$。若概率和分配都均匀，每项为一除 E 的平方，求和为一除 E，再乘 E 得一。若所有概率和分配都集中到同一专家，值可接近 E。

这个值不是“必须精确等于零”的损失，也不是任何分布下都保证某个简单全局范围的万能指标。它把软偏好与硬使用结合，鼓励避免概率和分配共同集中。不同论文有不同归一化、top-k 计数与训练策略，必须按具体公式实现。课程函数明确只是一个可解释的简单版本，不代表完整训练配方。

硬分配 f 通常作为当前选择结果的统计量，不对离散 top-k 本身求普通导数。把 f 视为常量时，辅助量对概率的梯度为 $E f_e/N$，再经过 softmax 传回 router logits。这样即便某专家未被当前 token 选中，它的 softmax 概率也可能通过辅助损失获得梯度，与前面局部 task loss 的未选梯度为零形成区别。

```python
uniform_prob=np.full((6,3),1/3)
uniform_chosen=np.array([[0],[1],[2],[0],[1],[2]])
np.testing.assert_allclose(load_balance_loss(uniform_prob,uniform_chosen),1.)
collapsed_prob=np.tile([0.98,0.01,0.01],(6,1))
collapsed_chosen=np.zeros((6,1),dtype=int)
np.testing.assert_allclose(load_balance_loss(collapsed_prob,collapsed_chosen),2.94)

def balance_gradient(logits,chosen):
    p=softmax(logits)
    n,e=p.shape
    fractions=np.bincount(chosen.ravel(),minlength=e)/chosen.size
    dp=np.broadcast_to((e/n)*fractions,p.shape)
    dz=p*(dp-np.sum(dp*p,axis=1,keepdims=True))
    loss=e*np.dot(p.mean(axis=0),fractions)
    return float(loss),dz

balance_logits=np.array([[2.,0.,-1.],[1.,0.5,-0.3]])
fixed_chosen=np.array([[0],[0]])
balance_loss,balance_dz=balance_gradient(balance_logits,fixed_chosen)
balance_numeric=np.zeros_like(balance_logits)
for index in np.ndindex(balance_logits.shape):
    old=balance_logits[index]
    balance_logits[index]=old+1e-6
    plus=balance_gradient(balance_logits,fixed_chosen)[0]
    balance_logits[index]=old-1e-6
    minus=balance_gradient(balance_logits,fixed_chosen)[0]
    balance_logits[index]=old
    balance_numeric[index]=(plus-minus)/2e-6
np.testing.assert_allclose(balance_dz,balance_numeric,atol=1e-8)
```

只优化均衡也会失败。如果每个专家被强迫同样使用，但它们没有学到有用功能，模型仍然可能质量很差。辅助项系数太大时，路由可能为了均衡牺牲任务适配；太小时，塌缩可能持续。报告应同时展示任务损失、专家负载、丢弃率、router 熵和梯度统计，不能只选一个漂亮数字。

router 熵是每个 token 概率分布的不确定性；负载均衡是跨 token 的分配分布。每个 token 都极其确定，但不同 token 分散到不同专家，整体可以均衡；每个 token 都近似均匀，top-k 的固定 tie-break 却可能把很多 token 送到同一批专家。二者相关但不等价，调试时需要分别观察。

## 第十课　把 MoE 放回完整解码器

仓库已有 TinyMoEDecoder，它继承前章完整 GQA、RoPE 和 KV cache 路径，只把 FFN 子层替换成路由专家。初始化时先建立基础模型，再删除不再使用的 gate、up、down 稠密 FFN 参数，为每层建立独立 router 和专家列表。删除旧参数很重要，否则参数账本会把未参与前向的稠密权重也算进去，给出错误的总容量。

每层前向仍是两次残差更新。先对 x 做 RMSNorm，执行 causal attention，得到 delta 并加回 x；再对更新后的 x 做 RMSNorm，送入该层 MoE，得到专家增量并加回。router 看到的是 attention 更新后的归一化状态，不是原始 embedding，也不是上一层的旧输入。改变这个读取点就改变了架构。

该完整模型没有容量丢弃，因此每个 token 的专家选择只依赖该 token 当前隐藏向量，与同批其他 token 无关。在此前提下，full 与 step 应保持等价。若引入按整批容量竞争，full 把多个 token 放在同一容量窗口，step 每次只处理一个 token，可能得到不同的接收决策；这不是浮点误差，而是你改变了容量作用范围。缓存等价测试必须把这类批次依赖考虑进去。

```python
from labs.models.moe_decoder import TinyMoEDecoder
from labs.models.core import DecoderConfig

moe_model=TinyMoEDecoder(DecoderConfig(seed=19),experts=3,top_k=2)
ids=[1,2,3,4,5]
full_logits=moe_model.forward(ids)
state=None
rows=[]
for token in ids:
    row,state=moe_model.step(token,state)
    rows.append(row)
np.testing.assert_allclose(np.stack(rows),full_logits,atol=1e-10,rtol=1e-10)
assert all(set(block)=={"q","k","v","o"} for block in moe_model.blocks)
assert len(moe_model.expert_blocks)==moe_model.config.layers
```

这个模型使用随机权重，提供完整结构和缓存路径验证，不具有经过训练的语言能力。它的专家使用 ReLU 两层 FFN，没有共享专家、专家并行或生产容量策略。你已经在本章实现了路由与反向的最小完整机制，后续可将这些机制接到可训练框架，但不能把随机前向测试写成“已经训练了一个 MoE 大模型”。

阅读 forward 时应该能逐行解释：token ID 查 embedding，位置数组为零到长度减一，逐层计算 attention 和 MoE，最后 RMSNorm 与词表投影。阅读 step 时应指出：每层 cache 仍然只保存 attention 的 K/V，普通逐 token FFN 专家没有额外跨时间状态。MoE 增加很多权重，不必然增加传统 KV cache 的头维度或历史长度；这两个成本来自不同模块。

## 第十一课　专家并行：路由表怎样变成跨设备通信

当全部专家放在同一设备时，dispatch 主要是本地索引和数据搬运。若专家分布在多张设备上，一个 token 选择的专家可能位于其他设备，需要把隐藏向量发过去，计算后再把结果送回。专家并行的基本通信因此包含“发送到拥有专家的设备”和“把专家输出返回原 token 所在位置”两个阶段。它不是把整个模型复制八份，也不是普通数据并行的另一个名字。

设四个专家分布在两张设备上，设备零保存专家零和一，设备一保存专家二和三。一个 token 选择专家一和三，就需要在两张设备分别执行。若原 token 在设备零，本地专家一无需跨设备输入，专家三需要发送一次隐藏向量并返回一次输出。多个 token 可以按目标设备打包，减少小消息开销。逻辑上的专家桶进一步聚合为设备桶，但仍必须保留 token 与槽位映射。

通信量与 token 数、隐藏维度、top-k、数据类型和本地命中比例有关。粗略地，每条远程分配至少发送一个 D 维输入并返回一个 D 维输出，约两个 D 乘每元素字节数；实际还可能有门控权重、索引、对齐、重复传输或不同的 combine 策略。这个估算只描述数据规模，不能直接换算成延迟，因为网络拓扑、带宽、消息大小和重叠执行都影响结果。

负载不均衡会放大分布式瓶颈。即使总分配数适中，如果大多数 token 都去某一张设备，它的专家计算与接收队列会拖慢整个步骤。其他设备空闲也无法自动替它执行，除非有复制专家或动态调度机制。容量限制可以控制最坏批次，但丢弃会影响函数；负载均衡训练可以改善分布，但不能保证每个运行时批次都均匀。

因此 MoE 性能报告应同时记录专家级与设备级负载。两个专家分配均匀，但若它们恰好都在同一设备，设备负载仍不均；反过来，专家负载不均也可能通过合理放置缓解设备瓶颈。模型路由、专家放置和执行调度是三个相互作用的层次。只看总 FLOPs 无法解释这些现象。

### 十一点一　一个可手算的通信账本

有四个 token，原本都在设备零，隐藏维度八，每元素两字节。选择集合依次为零和二、一和三、二和三、零和一。设备一拥有专家二和三，因此远程分配数分别一、一、二、零，总共四条。每条输入十六字节、输出十六字节，最简单模型下共一百二十八字节有效向量数据。

如果第三个 token 同时去设备一的两个专家，系统可以选择只发送一次输入，再在设备一内部复制给两个专家，返回合并后结果；也可以按分配发送两份。前者可能减少传输，但要求目标端知道两个专家及各自权重，并正确合并。这个例子说明同一个路由图可以有不同通信实现，不能把理论分配数直接当成唯一物理字节数。

若采用目标设备内先合并，返回的应是已经乘门控权重后的专家和，还是未加权的各专家输出，也需要明确。不同协议会改变哪些元数据必须一起发送。只要最终每个 token 的贡献与参考定义一致，执行可以重排；但若把不同 token 的输出错误合并，shape 也许仍对，结果却完全错。服务框架的通信测试应使用可识别的 token 与专家标记。

### 十一点二　为什么小专家批次可能降低硬件效率

一个大矩阵乘法能形成较多并行工作，复用权重与输入。MoE 把 token 分散到多个专家后，每个专家收到的行数可能很少，矩阵变得又小又不规则。即使算术总量下降，内核启动和数据整理开销的比例可能上升。对个人显卡的小 batch 生成尤其需要测量，不能仅凭激活参数少就期待线性加速。

分组矩阵乘法与融合算子可以把多个不同专家的小任务组织在一起，提高执行效率，但它们仍需处理不同权重、不同 token 数和可能的空专家。填充到统一容量便于规则计算，却可能计算不少无效槽位；紧凑打包减少无效算术，但索引更复杂。两种方案体现规则性与紧凑性的取舍，哪个更快取决于尺寸与硬件。

在 H100 服务器上做短时实验时，应先用合成数据固定 N、D、F、E、k，分别测均匀与高度偏斜路由。不要一开始就启动大型训练占用很久；小型受控基准已经能揭示分发、专家计算和回收的比例。报告中说明是否包含通信、是否预热、是否同步计时，以及是否把数据构造时间算进去。模型机制学习首先需要可解释的测量，而不是一个没有上下文的吞吐数字。

## 第十二课　架构变体：共享专家、细粒度专家与路由策略

共享专家指所有 token 都执行的一条或多条专家支路。若共享输出为 s(x)，路由专家输出为 r(x)，一种简单定义是 y=s(x)+r(x)。共享支路可以承担通用变换，路由支路承担条件化变化，但这是设计动机，实际功能分工仍需要训练与分析。不能因为名字叫共享专家，就断言它必然负责语法、事实或某种固定能力。

共享专家会增加每 token 固定计算量，也增加总参数。计算激活参数时要把它计入，不能只数 top-k 路由专家。反向时共享支路总有 task 梯度，而未选路由专家没有相应执行路径。容量丢弃导致路由增量为零时，共享支路仍可能提供变换，但这同样不意味着丢弃没有代价。

细粒度专家通常把单个专家的中间宽度做小，同时增加专家数量与选择数量。举例，原先两个宽度六十四的专家，与四个宽度三十二的专家，在简单两层 FFN 的总激活乘加上可能相近，但组合选择空间和路由行为不同。不能仅凭总宽度相等就认为函数相同，因为 ReLU 或 SwiGLU 非线性位于各专家内部，门控也分别作用。

专家内部激活也不是固定只有 ReLU。可以使用 SwiGLU，此时每个专家包含 gate、up、down 三个矩阵，参数与计算账本应改为大约三 DF，而不是两 DF。把本章 ReLU 专家直接替换成 SwiGLU 后，反向要同时处理两个输入投影和门控乘法。外层 dispatch 与 combine 的逻辑仍可复用，这体现了模块接口的价值。

token-choice 是每个 token 选专家；expert-choice 则可让每个专家选择 token。后者能直接控制每专家负载，但一个 token 可能被不同数量的专家选择，甚至没有专家选择，合并与覆盖策略需要另定义。两者都涉及 top-k，却沿不同轴做选择，不能只把数组转置一下就认为所有语义自动等价。

路由得分可以使用 softmax、sigmoid 或其他变换，也可能加入用于负载调整的偏置。选择用的分数与最终 combine 权重未必相同：某些设计可能用调整后的分数选专家，再用另一套原始分数计算贡献。阅读实现时应分别标出“选择分数”和“输出权重”，否则容易把负载控制项错误地直接乘到专家输出上。

### 十二点一　哪些概念可以从本章迁移，哪些必须重推

可以迁移的是：每条 token 到专家的映射需要被记录；专家计算结果必须正确回填；共享输入的梯度需要累加；容量与丢弃必须有明确语义；分布式通信必须保持映射。需要重推的是：具体门控归一化、top-k 的轴、辅助损失、容量优先级、专家内部导数和不同阶段的路由策略。

例如将 softmax 改成 sigmoid 后，所选分数是否重新归一化会影响梯度公式；保留原 sigmoid 权重与重新归一化的 sigmoid 权重不是同一函数。若选择使用一个不可微的负载偏置，但输出权重不用它，梯度应沿实际输出权重路径传播。不能把本章“选中 logits 上 softmax”的简化公式机械套到所有新路由器。

阅读 [Switch Transformer](https://arxiv.org/abs/2101.03961) 与 [DeepSeek-V3 报告](https://arxiv.org/abs/2412.19437) 时，应带着这些问题逐项核对。前者和后者的路由、专家设计与训练策略并不相同。本章的价值是给你一套能拆解这些差异的基本工具，而不是让你用一个统一 toy 抹平所有变体。

## 第十三课　十二组分级练习与完整参考解

### 练习一：从 router logits 手算完整输出

使用第三课的三个 token 与三个专家，写出 logits 指数、完整 softmax、top-2 编号、重新归一化权重、每个专家输出与最终合并。参考答案已经逐项给出，但提交时应独立重算：第一行概率为七分之四（`4/7`）、七分之二（`2/7`）、七分之一（`1/7`），选中权重三分之二与三分之一；第二行选中权重为五分之三（`3/5`）与五分之二（`2/5`）；第三行两个并列专家各二分之一。输出对角元素为八除三、二十八除五、五。

容易犯的错误是直接用原始概率合并，第一行会得到十六除七而不是八除三。原始 top-2 概率和只有六除七，未选专家的概率质量被丢掉；本章定义通过重新归一化把选中权重和恢复为一。两种实现都能运行，但必须与声明的门控规则一致。

### 练习二：容量一时逐条填写分发表

答案按遍历顺序为：token 零到专家零接受、到专家一接受；token 一到专家一拒绝、到专家二接受；token 二到专家零拒绝、到专家二拒绝。实际接收数各一，丢弃三条，总提出六条，分配丢弃率二分之一。至少一个专家贡献被丢弃的 token 有两个，所以 token 受影响比例三分之二；全部贡献被丢弃的 token 有一个，比例三分之一。

这三个比例描述不同问题，不能都叫“drop rate”而不给分母。分配丢弃率反映路由槽位损失，受影响 token 比例反映覆盖范围，完全丢弃比例反映本层没有路由专家变换的严重情况。实际监控应至少说明采用哪一个定义。

```python
_,drop_audit=moe(x_hand,router_hand,experts_hand,2,1)
assignment_drop=float(drop_audit["dropped"].mean())
affected_token=float(drop_audit["dropped"].any(axis=1).mean())
fully_dropped=float(drop_audit["dropped"].all(axis=1).mean())
np.testing.assert_allclose([assignment_drop,affected_token,fully_dropped],
                           [1/2,2/3,1/3])
```

### 练习三：证明容量会引入顺序依赖

把 token 顺序逆转，仍用容量一，运行后再把输出行逆回原顺序，与原结果比较。它们一般不同，因为容量由先到先得决定。完整参考如下。这个实验说明在有容量丢弃时，token 的排列可以改变模型函数，即使每个 token 的 router logits 不变。

```python
original_cap,_=moe(x_hand,router_hand,experts_hand,2,1)
reversed_cap,_=moe(x_hand[::-1],router_hand,experts_hand,2,1)
restored_order=reversed_cap[::-1]
assert not np.allclose(original_cap,restored_order)
original_no_cap,_=moe(x_hand,router_hand,experts_hand,2,None)
reversed_no_cap,_=moe(x_hand[::-1],router_hand,experts_hand,2,None)
np.testing.assert_allclose(original_no_cap,reversed_no_cap[::-1])
```

无容量时，每个 token 独立选择与计算，所以恢复排列后相同。有容量时，跨 token 竞争引入依赖。将这个结论用于 full 与 step 比较，你就能解释为什么某些训练容量策略不能直接作为推理缓存等价路径使用。

### 练习四：实现一个所有专家都执行的稠密混合基准

令 top-k 等于专家总数，门控权重等于完整 softmax。对每个专家计算全部 token，再按每行概率加权求和，应与 moe 的 top-k 全选路径一致。这个基准用于检查 dispatch/combine，并不是标准“稠密 FFN”：它仍然有多专家与门控，只是全部激活。

```python
def all_expert_mixture(x,router,experts):
    p=softmax(x@router)
    out=np.zeros_like(x,dtype=float)
    for e,(up,down) in enumerate(experts):
        out+=p[:,e:e+1]*(np.maximum(x@up,0)@down)
    return out

all_route,_=moe(x_hand,router_hand,experts_hand,3,None)
np.testing.assert_allclose(all_route,all_expert_mixture(x_hand,router_hand,experts_hand))
```

### 练习五：验证 top-1 重新归一化的局部梯度为零

第六课代码是完整参考。你还应解释为什么这不等于“router 永远不能改变”：优化器可能收到辅助损失或其他路径的梯度，输入表示也可能随上游训练改变，离散选择可能在训练过程中切换。但仅看本章这一条 task loss 到归一化单权重的局部路径，导数确实为零。结论必须限定到具体门控定义。

若把 top-1 权重保留为完整 softmax 中的选中概率，它不再恒为一，任务梯度可以通过该概率传播；但输出幅度也随选中概率变化。这是另一种前向函数，不能在解释梯度时偷偷切换。理解这个差别比记住“top-1 可微或不可微”这种过度简化标签更有用。

### 练习六：手算两个选中专家的 router 梯度

设门控权重为四分之三和四分之一，两个专家对损失的贡献导数 a 分别二和六。加权平均为三，故两个 logit 梯度为 $3/4(2-3)=-3/4$ 与 $1/4(6-3)=3/4$。梯度下降会增加第一个 logit、减少第二个，使更低损失方向的专家得到更大权重。两个梯度和为零，对应 softmax 对统一平移不敏感。

```python
g=np.array([.75,.25])
a=np.array([2.,6.])
dz=g*(a-g@a)
np.testing.assert_allclose(dz,[-.75,.75])
np.testing.assert_allclose(dz.sum(),0.)
```

若所有专家的 a 都相同，门控梯度为零，因为改变混合比例不会改变当前损失的一阶变化。不是 router 出故障，而是这条样例没有提供区分专家的任务信号。调试时应检查专家输出是否退化相同，不能只看到 router 梯度小就提高学习率。

### 练习七：统计 proposed、accepted 与 overflow

完整答案从 chosen 统计提出次数，从 counts 读取接受次数，两者之差为每专家丢弃次数。这个等式也可作为 audit 自洽检查。若统计不相等，可能是分发过程中漏记、重复计数或 capacity 语义不一致。

```python
_,stats=moe(x_hand,router_hand,experts_hand,2,1)
proposed=np.bincount(stats["chosen"].ravel(),minlength=3)
accepted=stats["counts"]
overflow=proposed-accepted
np.testing.assert_array_equal(proposed,[2,2,2])
np.testing.assert_array_equal(accepted,[1,1,1])
np.testing.assert_array_equal(overflow,[1,1,1])
assert overflow.sum()==stats["dropped"].sum()
```

### 练习八：实现共享专家支路

参考定义为共享专家输出加路由专家输出，不额外归一化。共享专家选择单位上投影和二分之一单位下投影，在非负输入上输出二分之一 x。结果应等于第三课 MoE 输出加这条确定增量。若你选择平均两条支路，那是另一个定义，应改写公式与测试。

```python
shared_up=np.eye(3)
shared_down=.5*np.eye(3)
shared_output=np.maximum(x_hand@shared_up,0)@shared_down
routed_output,_=moe(x_hand,router_hand,experts_hand,2,None)
combined_output=shared_output+routed_output
np.testing.assert_allclose(combined_output,np.diag([8/3+.5,28/5+.5,5.5]))
```

### 练习九：训练一个真正更新参数的小 MoE

使用第八课完整反向训练一个二维回归任务，目标为第一坐标乘二、第二坐标取负。数据是固定的正数输入，三个专家各有三维中间层，top-2，无容量丢弃。所有专家、router 和输入路径都按同一前向定义求导；训练只更新参数，不把输入当参数。这个实验验证“从损失到参数更新”的闭环，不代表语言建模或专家已经形成可解释分工。

```python
fit_rng=np.random.default_rng(81)
fit_x=fit_rng.uniform(.1,1.,size=(12,2))
fit_target=np.stack([2*fit_x[:,0],-fit_x[:,1]],axis=1)
fit_router=np.array([[1.2,.5,-.6],[.1,.8,-.2]],dtype=float)
fit_experts=[(fit_rng.uniform(.2,.6,size=(2,3)),
              fit_rng.normal(0,.2,size=(3,2))) for _ in range(3)]
fit_losses=[]
for step in range(350):
    prediction,saved=moe_forward_saved(fit_x,fit_router,fit_experts,2)
    difference=prediction-fit_target
    loss=float(np.mean(difference**2))
    fit_losses.append(loss)
    upstream=2*difference/difference.size
    _,router_gradient,expert_gradients=moe_backward_saved(saved,upstream)
    fit_router-=.05*router_gradient
    for (up,down),(dup,ddown) in zip(fit_experts,expert_gradients):
        up-=.05*dup
        down-=.05*ddown
assert np.isfinite(fit_losses).all()
assert fit_losses[-1]<fit_losses[0]*.2
```

为什么损失不必每一步严格下降？top-k 集合可能切换，有限学习率也可能使局部近似不够准确。这里验收看整体明显下降与有限性，并保留完整曲线供检查。若要研究路由训练质量，应再加入独立验证集、多个种子和负载统计，而不能把训练集拟合当成泛化证明。

### 练习十：构造一个未训练专家的证据

在一个固定样例中让第三个专家永远不进入 top-2，再执行第八课反向，第三个专家的参数梯度应为零。前面 train_router 的第三列明显较低，可直接用它验证。注意这只是当前 batch 的梯度，不能推出该专家在整个训练中永远没有更新；需要跨步骤统计实际选择。

```python
_,saved_zero=moe_forward_saved(train_x,train_router,train_experts,2)
_,_,zero_expert_grads=moe_backward_saved(saved_zero,train_upstream)
np.testing.assert_array_equal(zero_expert_grads[2][0],np.zeros_like(train_experts[2][0]))
np.testing.assert_array_equal(zero_expert_grads[2][1],np.zeros_like(train_experts[2][1]))
```

### 练习十一：计算总参数与激活计算

D 为十六、F 为三十二、E 为八、k 为二，ReLU 专家参数总数为 $8\times2\times16\times32=8192$，router 参数为一百二十八。每 token 的专家乘加约二千零四十八，再加 router 一百二十八。若有一个同宽共享专家，总参数再加一千零二十四，每 token 专家乘加也再加一千零二十四。

这些数字忽略 bias、激活逐元素运算、路由排序和通信，因此应称为主要矩阵乘加估算。把它写成“每 token 精确 FLOPs”会不严谨，因为一次乘加通常在 FLOPs 约定中计两次运算，且其他操作尚未计入。报告必须注明单位和统计范围。

### 练习十二：独立复写完整 MoE 解码器

参考实现是仓库 TinyMoEDecoder，但你的提交应能解释每一行的作用，而不是只继承它并改一个参数。最低要求包括每层独立 router 与专家、删除未使用稠密 FFN、attention 后再路由、残差更新、最终 norm 与输出头、full/step 一致性。使用无容量路径保证 token 独立路由，再另写实验展示容量如何改变批次语义。

完整验收同时检查参数账本和状态账本：专家权重确实被计入，总参数不包含被删除的稠密 FFN；cache 仍然由每层 attention K/V 组成；专家数变化不应偷偷改变 query/KV 头配置。若加入共享专家，应在前向、反向、参数计数和激活计算四处同步更新，不能只在图上画一条支路。

## 第十四课　再做一次全链手算：从损失到 router 与专家参数

为了把前面的矩阵公式落到具体数字，考虑只有一个 token、两个专家、标量输入的最小例子。输入 x 为二，两个专家都是线性倍率，分别为 a 等于一和 b 等于三，输出分别二和六。router 两个 logits 都为零，top-2 全选，门控权重各二分之一，所以混合输出 y 为四。目标值为一，损失定义为二分之一乘预测误差平方，因此损失为四点五，上游导数 dy 为三。

先算专家参数。y 对 a 的导数为门控权重乘输入，即一；所以损失对 a 的导数为三。对 b 也是三。若学习率零点一，两个参数分别从一降到零点七、从三降到二点七。这个方向把两个专家输出都往目标方向拉，但 router 同时也能调整混合比例，不必让所有专家完全相同。

门控权重梯度分别为 dy 乘专家输出，即六和十八。加权平均为十二，所以两个 logit 梯度为二分之一乘六减十二，得到负三；第二个为二分之一乘十八减十二，得到正三。梯度下降使第一个 logit 增加零点三，第二个减少零点三，偏向当前输出更小、更接近目标的专家。两个 logit 梯度和为零，符合 softmax 平移不变性。

若 router logits 来自输入乘权重，两个 router 权重初始均为零，则它们梯度是 x 乘 dz，即负六和正六。这里要注意：前面把 logits 当独立参数更新，与现在把 router 权重当参数更新不是同一个步长效果。权重更新后 logits 还要再乘输入二，因此变化会放大。讨论学习率时必须知道优化器实际更新哪个变量，不能把中间 logits 与模型参数混为一谈。

输入梯度也有两条来源。专家路径为门控加权的专家导数，再乘 dy，即三乘二分之一乘一加二分之一乘三，得到六。router 路径为 dz 与 router 权重的点积；因为此例 router 权重初始为零，这条贡献为零。若 router 权重非零，输入梯度就还要加上它。完整模型里这个输入梯度会继续传回归一化、attention 和更早层。

这个最小例子没有 ReLU 拐点、top-k 切换或容量竞争，因此每一步都能直接验算。你可以先把它写成标量程序，确认有限差分，再扩展到向量输入和两层专家。学习复杂模块时，从最小但不退化的例子出发，比直接在上百维随机矩阵里寻找错误更有效。

```python
scalar_x=2.
scalar_a,scalar_b=1.,3.
scalar_logits=np.zeros(2)
scalar_g=softmax(scalar_logits[None,:])[0]
scalar_values=np.array([scalar_a*scalar_x,scalar_b*scalar_x])
scalar_y=float(scalar_g@scalar_values)
scalar_dy=scalar_y-1.
scalar_da=scalar_dy*scalar_g[0]*scalar_x
scalar_db=scalar_dy*scalar_g[1]*scalar_x
scalar_dg=scalar_dy*scalar_values
scalar_dz=scalar_g*(scalar_dg-scalar_g@scalar_dg)
np.testing.assert_allclose([scalar_y,.5*scalar_dy**2,scalar_da,scalar_db],
                           [4.,4.5,3.,3.])
np.testing.assert_allclose(scalar_dz,[-3.,3.])
np.testing.assert_allclose(scalar_x*scalar_dz,[-6.,6.])
```

### 十四点一　加上容量丢弃后，门控梯度为何仍会竞争

继续使用同样两个专家，但假设第二个专家分配被丢弃，且不重新归一化。输出变成二分之一乘二等于一，若目标为零，损失二分之一，上游导数为一。两个门控权重的梯度应使用实际贡献：第一个为二，第二个为零。加权平均为一，因此 logits 梯度为正二分之一和负二分之一。

这意味着梯度下降会降低存活专家的 logit、提高被丢弃专家的 logit，因为在当前固定丢弃决策下，把权重转移到零贡献槽位能降低输出并减少这个目标损失。这个行为可能不是你希望的训练效果，却是当前前向定义的真实导数。它说明容量策略、归一化策略和损失共同决定训练动力学，不能只把丢弃看成一个无关的运行时优化。

如果丢弃后重新归一化，唯一存活专家权重变成一，输出为二，门控权重在局部又变成常量，router 的这条任务梯度为零。两种前向定义产生不同梯度，必须同步修改训练实现。把一种前向与另一种反向拼接，会得到没有对应标量目标的错误更新。

在真实训练中，容量丢弃可能通过残差、共享专家、辅助目标与不同路由策略缓解，但不存在一个脱离定义的通用答案。你应先写出实际输出公式，再对它求导，最后判断这种梯度是否符合设计目标。这是研究新路由策略时最基本的纪律。

### 十四点二　专家“专长”不能只靠名字推断

专家编号零、一、二通常只是参数组索引，没有预先规定的语义。训练后可能出现某些输入模式偏好，但这种偏好需要测量。看到某专家接收大量代码 token，不足以断言它是“代码专家”；也许它偏好的是某种标点、频率、位置或隐藏向量尺度。分析应控制这些混杂因素。

可以构造一个可解释的合成任务：输入带有明确类别标签，不同类别要求不同线性变换，然后观察路由是否与类别相关。即便如此，也要比较不同随机种子，因为专家编号可以置换，功能相同的模型未必把同一类别交给相同编号。比较时应允许专家排列对齐，而不是逐编号硬比。

对真实语言数据，记录 token 表面形式、上下文类别、路由概率、选中专家和专家输出贡献，再做统计。不要只展示几个符合叙事的例子；应报告样本量、基线分布和反例。路由频率表示使用，不直接表示某专家对最终答案的重要性；重要性还取决于门控权重、输出幅度、残差和后续层。

## 第十五课　渐进项目：可训练、可审计的 MoE 工作台

第一阶段整理一个最小模块接口。输入为二维 token 矩阵，输出为同 shape 的 MoE 增量和 audit。配置明确专家数、top-k、专家内部宽度、容量是否启用、丢弃后是否归一化、tie-break 规则。不要把这些选择写成散落在循环里的隐含常数，否则后续比较不同策略时很难知道究竟改了什么。

第二阶段保留两条前向路径：逐 token 参考与按专家批次分发。用同一参数、同一输入、同一路由决策比较结果。测试包括空专家、全部丢弃、并列 logits、top-1、top-E、不同 token 排列和重复 token。参考路径可以慢，但应该清楚到每一条贡献都能打印出来；优化路径负责减少循环与搬运。

第三阶段接入第八课反向，用小型回归任务先验证完整训练闭环。每十步记录任务损失、每专家提出与接受次数、router 熵、每专家参数梯度范数和输出范数。若加入辅助均衡损失，要分别记录任务项与辅助项，避免总损失下降只是辅助项变化。初期在 CPU 上完成即可，数据和维度都足够小。

第四阶段将 MoE 模块接到可训练解码器的 FFN 位置。先使用很小词表、很短序列和一两层模型，在固定小数据上过拟合；再增加验证数据。保持 attention、tokenizer、优化器与数据相同，对比稠密 FFN 和 MoE。比较应明确选择相同激活计算还是相同总参数，两种基准回答的问题不同，不应混在同一结论里。

第五阶段研究容量。先在推理参考中证明有容量时 full 与逐 token 可能不一致，再决定训练和推理各自采用什么策略。若推理使用不丢弃或足够容量，应在配置中明确；若必须限制容量，应把批次组成和排序纳入函数与测试。不要让服务系统为了凑 batch 悄悄改变一个请求的模型输出，而你却仍声称完全等价。

第六阶段才进入性能优化。测量 router、排序或分桶、专家矩阵乘法、combine 各阶段耗时。单卡先研究小批次与大批次；多卡再研究专家放置、通信与负载。服务器 GPU 使用应有明确实验矩阵、步数上限和退出条件，避免长时间运行却没有可解释结果。一次十分钟的受控实验通常比一夜无记录的大训练更能帮助定位实现问题。

### 十五点一　完整审计函数：让路由行为可以复查

下面的函数把课程 audit 转成明确统计。它同时计算概率熵、提出次数、接受次数和不同丢弃率。熵计算对零概率使用条件保护，避免零乘负无穷产生 NaN。这个统计不改变模型输出，只为实验提供证据。若不同实现使用不同归一化或容量范围，报告中仍需附配置说明。

```python
def audit_summary(audit):
    probabilities=audit["probabilities"]
    chosen=audit["chosen"]
    dropped=audit["dropped"]
    experts=probabilities.shape[1]
    proposed=np.bincount(chosen.ravel(),minlength=experts)
    accepted=np.asarray(audit["counts"])
    safe=np.maximum(probabilities,np.finfo(float).tiny)
    entropy=-np.sum(probabilities*np.log(safe),axis=1)
    return {
        "proposed":proposed.tolist(),
        "accepted":accepted.tolist(),
        "overflow":(proposed-accepted).tolist(),
        "assignment_drop_rate":float(dropped.mean()),
        "affected_token_rate":float(dropped.any(axis=1).mean()),
        "fully_dropped_token_rate":float(dropped.all(axis=1).mean()),
        "mean_router_entropy":float(entropy.mean()),
        "mean_probability":probabilities.mean(axis=0).tolist(),
    }

summary=audit_summary(capacity_audit)
assert summary["proposed"]==[2,2,2]
assert summary["accepted"]==[1,1,1]
assert summary["overflow"]==[1,1,1]
np.testing.assert_allclose(summary["assignment_drop_rate"],.5)
assert np.isfinite(summary["mean_router_entropy"])
```

解释报告时，先看总量自洽：提出次数之和应为 N 乘 k，接受次数加丢弃次数应等于提出次数。再看分布：是否某些专家长期为零，是否少数专家占据绝大多数。最后把统计与任务损失关联，观察均衡改善是否伴随质量变化。不能只凭平均熵高低作判断，因为 top-k、温度和专家数都会影响它的尺度。

### 十五点二　常见训练故障的具体排查顺序

如果任务损失完全不下降，先检查专家和 router 参数是否真的被优化器更新，梯度是否有限，输入与目标是否错位。用一个 token、两个专家的小样例验证手写反向或自动微分结果，再尝试固定数据过拟合。不要一开始就调整大量超参数，因为基础计算图错误不会被更复杂的训练配方可靠修复。

如果专家输出幅度不断变大，检查初始化、学习率、归一化位置、门控权重和残差尺度。选中权重是否确实按声明归一化？是否在 combine 时重复加了一次？共享专家是否被额外乘了错误系数？如果输出幅度异常仅出现在某个专家，检查它接收的样本分布与梯度范数，而不是只看全局平均。

如果 router 很快塌缩到少数专家，先区分软概率塌缩与硬选择偏置。并列或近似并列 logits 加稳定 top-k，可能让相同编号长期被选，即使 softmax 看似均匀。输入表示是否过于相似、router 初始化是否完全对称、辅助目标是否有梯度，都需要检查。不能只把 capacity 调小，因为那可能把塌缩转成大量丢弃而不解决学习问题。

如果负载看起来均匀但质量很差，检查辅助项是否压过任务目标，专家是否学到不同且有用的函数，路由是否只是随机分散。均匀分配是资源目标之一，不是语言能力本身。可以暂时固定路由或执行全部专家作为诊断对照，判断问题更接近专家训练还是选择机制；这些对照改变了模型函数，应明确标记为诊断实验。

如果 full 与 step 不一致，先关闭容量和随机路由噪声，再验证每 token 的 logits、选择编号与门控权重。无容量且专家逐 token 独立时，差异应追溯到 attention cache、归一化、路由输入或参数读取。若只有容量开启后不一致，应检查容量窗口是否随 batch 或 chunk 改变，而不是盲目放大数值容差。

如果多卡吞吐不佳，先测单卡专家计算，再测纯分发与回收，最后测完整路径。否则无法判断是矩阵太小、负载偏斜、网络通信还是同步等待。模型结构知识帮助你提出正确的分解，性能工具帮助你量化各项。不要用一个总耗时去猜所有瓶颈。

### 十五点三　版本管理与扩展规范

每次改变路由语义都应更新配置版本和测试样例。特别是丢弃后是否归一化、容量按何种范围计数、tie-break 是否稳定、选择分数是否与输出权重相同，这些变化可能不改变张量 shape，却改变模型函数。仅保存权重文件而不保存这些规则，会让后来的人无法复现输出。

新增专家类型时，保持统一的输入输出契约，并为其提供单独前向与反向检查。新增分发实现时，保留慢参考作为数值基准。新增负载策略时，先用确定手算样例说明每条分配接受或拒绝的原因，再做随机压力测试。这样项目可以逐步扩展，而不会在每次优化后失去可解释性。

最终通过标准是：你能从一个 token 的隐藏向量开始，计算 router logits、选中集合、门控权重、容量决策、专家输出和 combine 结果；能指出哪些路径可微、哪些是离散选择；能写出完整梯度并通过差分；能把模块放进完整解码器并验证缓存；能区分总参数、激活计算、状态容量与通信成本。达到这些能力后，再阅读复杂 MoE 模型时，你面对的是可拆解的机制组合，而不是一串新名词。

### 十五点四　完整解码器的参数账本不能只数专家

以仓库默认配置为例，词表三十二、隐藏维度十六、两层、四个查询头、两个 KV 头、每头维度四、专家中间宽度三十二、每层三个专家。embedding 与独立输出头各有五百一十二个参数，合计一千零二十四。每层 attention 的 Q 与 O 各二百五十六，K 与 V 各一百二十八，合计七百六十八。

每个 ReLU 专家上投影十六乘三十二、下投影三十二乘十六，共一千零二十四。三个专家合计三千零七十二，router 为十六乘三等于四十八。于是每层共有七百六十八加三千零七十二加四十八，等于三千八百八十八；两层为七千七百七十六，再加 embedding 与输出头，完整模型共八千八百个参数。课程 RMSNorm 没有可训练 gain，因此这个账本没有额外归一化参数；若移植到带 gain 的实现，应补上。

```python
def count_moe_parameters(model):
    total=model.embedding.size+model.output.size
    total+=sum(array.size for block in model.blocks for array in block.values())
    for block in model.expert_blocks:
        total+=block["router"].size
        total+=sum(up.size+down.size for up,down in block["experts"])
    return total

default_moe=TinyMoEDecoder()
assert count_moe_parameters(default_moe)==8800
```

这个默认模型并不是与默认稠密模型严格计算匹配的公平基准。稠密模型使用 SwiGLU 三个矩阵，专家模型使用 ReLU 两个矩阵且 top-2，所以每 token 的 FFN 主要矩阵乘加分别为三乘十六乘三十二与二乘二乘十六乘三十二，也就是一千五百三十六和二千零四十八。若直接比较速度或损失，必须承认激活计算不同，而不能只说“MoE 更大但计算一样”。

要设计计算匹配实验，可以调整专家中间宽度或 top-k，但整数维度、激活函数和矩阵效率仍会影响实际运行。要设计总参数匹配实验，则可能需要减少专家数或宽度，却导致每 token 计算不同。没有一个单一配置能自动控制所有因素，研究报告应先说清问题，再选择匹配标准。

### 十五点五　从模块测试走向训练复现

保存训练 checkpoint 时，除模型参数外还应保存专家排列、router 配置、优化器状态、随机数状态和数据进度。专家编号看似可以任意置换，但必须同时置换 router 输出列、专家参数列表和相关优化器状态。只交换专家权重而不交换 router 列，会改变每个编号指向的函数；只交换参数而不交换动量，也会改变后续训练轨迹。

恢复训练后，先在一个固定小 batch 上比较损失与路由 audit，再继续大量步骤。若输出一致而之后迅速分叉，检查优化器矩状态、随机路由噪声和数据顺序。若恢复第一步就不同，检查配置、参数加载顺序和专家索引。把这些检查纳入项目，可以避免训练数小时后才发现某些专家从未正确恢复。

对多 adapter 或多模型服务，也不能跨权重版本复用专家输出或 attention cache。即便专家结构相同，参数不同就意味着函数不同；即便只修改某层专家，后续层的隐藏状态和 K/V 也可能改变。状态复用仍然必须沿完整依赖图判断，这与前面缓存章节的原则完全一致。

最后再检查一个容易遗漏的事实：专家输出可以为负。即使内部使用 ReLU，下投影仍是一般线性矩阵，能够产生正负分量。因此门控权重非负且和为一，只保证输出落在选中专家输出的凸组合范围，不保证每个坐标非负，也不保证向量范数一定很小。若所有专家输出相互抵消，合并结果可能接近零；若输出方向相同且幅度大，合并仍可能很大。

这也解释了为什么仅监控路由概率不够。两个模型可以拥有完全相同的 chosen 和 weights，却因专家参数不同而产生截然不同的结果。完整审计应同时观察路由与专家贡献，必要时记录每条贡献的范数以及合并前后的关系。这样才能区分“选错专家”“专家没学好”和“合并实现错误”这三类问题。

路由索引与专家参数排列必须同步保存。
