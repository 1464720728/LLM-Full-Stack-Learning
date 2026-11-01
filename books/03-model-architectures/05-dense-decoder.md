# M05 完整稠密解码器：从输入 token 到下一 token 的每一步

这一章把你在前面学过的组件组成一个现代风格的 decoder-only 小模型。我们使用 pre-RMSNorm、GQA、RoPE 和 SwiGLU，保留完整多层前向与输出头。这里的“现代风格”是一组明确组件的组合，不代表所有当前模型都完全一样，更不代表这份随机初始化 NumPy 模型可以加载任意厂商权重。

学习目标是读懂并独立重写 `TinyDecoder`。你需要知道每张矩阵的形状、每个中间表示的身份、每次归一化的轴、位置如何进入 Q/K，以及一个 token 的信息怎样经过多个残差子层。M06 再深入缓存与 KV 头共享，本章先确保完整前向本身没有黑箱。

运行入口是 `python -m labs.models.run_lab decoder`、`python -m labs.models.run_lab rope`，以及 `python -m labs.models.trace_decoder`。CPU 足够完成所有本章数值与结构检查。随机权重只用于验证实现，不具备训练后的语言能力。

## 第一课：“稠密”到底形容什么

稠密模型通常表示每个 token 在常规层中会使用该层同一组主要参数，而不是通过路由只选择少数专家。它不意味着 attention 矩阵永远没有 mask，也不意味着权重矩阵中的每个数都非零。一个使用因果 attention 的 dense decoder 仍然是稠密模型，因为“是否稀疏读取历史”和“是否稀疏激活专家参数”是不同轴。

对一个普通 FFN，每个 token 都经过同一组上投影、激活和下投影。参数在位置之间共享，但每个 token 的激活值不同。MoE 后面会引入路由，让不同 token 选择不同专家；理解这一变化前，必须先能清楚描述 dense FFN 的完整计算路径。

decoder-only 表示没有独立的 encoder 和每层 cross-attention。提示、对话模板和生成内容通常组织成同一条因果序列。每个位置通过 self-attention 读取允许的前缀，再用 FFN 处理当前表示。它不是“原始 Transformer 只保留最后一个输出层”，而是保留一整套堆叠的因果解码层。

提示与回答在结构上可以是同一序列，但训练时不一定对所有位置使用相同损失权重。例如指令微调可能只监督助手回答位置。这个标签 mask 改变训练目标，不自动改变 attention 可见性。提示位置可能不计损失，仍然为回答提供上下文。

本章模型的输入是 token ID 序列，输出是每个位置对词表的 logits。它不负责聊天角色模板、网络 API、工具调用或生成停止规则。这些上层流程会使用模型输出，但不能把它们的责任隐藏到“模型应该自己知道”的假设里。

## 第二课：先读配置，再读 forward

课程 `DecoderConfig` 包含词表大小 vocab、模型宽度 dim、query 头数 heads、KV 头数 kv_heads、层数 layers、FFN 隐藏宽度 hidden 和随机种子 seed。读一个模型时，先把这些数写成形状表，比从第一行导入开始逐字背代码更有效。

默认配置是词表三十二、模型宽度十六、四个 query 头、两个 KV 头、两层、FFN 隐藏宽度三十二。每个 query 头维度为十六除四等于四，KV 投影总宽度为两个 KV 头乘四，等于八。Q 投影输出十六通道，K/V 各输出八通道，因此它们的矩阵形状并不全部相同。

模型宽度必须能被 query 头数整除，query 头数必须能被 KV 头数整除，本卷 RoPE 使用成对坐标，所以每头维度还必须是偶数。这些不是为了让代码少写分支而随意加的限制，它们对应本实现的布局和分组定义。其他模型可以采用不同约定，但需要相应实现。

随机种子控制初始化可复现性，不是模型知识来源。固定种子让你修改代码后能比较同一组权重下的数值；换种子会换参数，输出自然变化。若比较两个结构，必须说明参数是否共享、是否以可比方式初始化，不能仅因两者使用同一个 seed 就断言所有对应权重相同。

```python
from labs.models.core import DecoderConfig, TinyDecoder
import numpy as np

config = DecoderConfig(vocab=32,dim=16,heads=4,kv_heads=2,layers=2,hidden=32,seed=7)
model = TinyDecoder(config)
assert model.embedding.shape == (32,16)
assert model.output.shape == (16,32)
block = model.blocks[0]
assert block["q"].shape == (16,16)
assert block["k"].shape == block["v"].shape == (16,8)
assert block["gate"].shape == block["up"].shape == (16,32)
assert block["down"].shape == (32,16)
```

把这张形状表保存为实验的一部分。以后你改 heads、kv_heads 或 hidden 时，应先更新预期形状，再运行检查。一个配置能成功构造，不代表你对每张参数的用途已经清楚；你还要能解释为什么 K/V 总通道较少，以及为什么 FFN 最后必须回到模型宽度。

## 第三课：输入 embedding 与 residual stream

输入 ID 序列查表得到 $[T,D]$ 的表示，常称为 residual stream 的初始状态。这个名称强调后续各子层会对同一宽度的状态不断加入增量。它不是一个额外神秘模块，而是贯穿层间的数据对象。

设 token 数为三，模型宽度十六，embedding 输出为 $[3,16]$。每一行对应一个位置，每一列是一个可学习特征坐标。相同 token 在不同位置最初取到相同 embedding 行，但后续 RoPE 和因果上下文会让它们的计算条件不同，因此最终表示不必相同。

本卷 TinyDecoder 不在 embedding 上直接加正弦位置向量，也不乘原始 Transformer 那样的根号 D 缩放。它把位置信息放入每层 Q/K 的 RoPE。这个差异是明确的架构选择，不能同时把原始正弦相加也偷偷加入而仍称为同一参考模型。

残差宽度在所有层中保持 D，不代表每个组件内部也必须是 D。attention 拆成多头、KV 投影可能更窄，FFN 隐藏层可能更宽；它们最后通过输出投影回到 D，才能加入 residual stream。区分外部接口宽度与内部计算宽度，是理解模型结构的关键。

残差状态不是一个永远只累加“新知识”的语义账本。子层输出可以有正有负，后续处理也会重新组合特征。我们能够准确描述数组相加和依赖关系，却不能把每个坐标简单命名为某个事实。学习时先掌握可验证的数值行为，再研究更复杂的表示解释。

## 第四课：RMSNorm 从公式到数值实现

RMSNorm 的核心是用每个位置特征向量的均方根缩放它。对向量 x，均方根为 $\sqrt{\frac1D\sum_i x_i^2+\epsilon}$，输出为 x 除以这个值；实际模型还可能有逐特征可学习 gain。它与 LayerNorm 的重要区别是没有先减去均值。

取 x 为 $[1,2,2]$，平方为一、四、四，平均为三，忽略很小 epsilon 后均方根为根号三。输出约为 $[0.57735,1.15470,1.15470]$。输出均值不为零，这不是实现错误，而是因为 RMSNorm 本来不执行去均值操作。

对常量向量 $[2,2,2]$，LayerNorm 的去均值结果为零；RMSNorm 则输出接近 $[1,1,1]$。所以两者不能互换后声称只改了性能。训练好的权重依赖所用归一化定义，改变它会改变网络数值和函数。

```python
def rms_reference(x, eps=1e-5):
    x = np.asarray(x,dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature axis required")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("positive finite epsilon required")
    return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)

x_norm = np.array([[1.,2.,2.],[2.,2.,2.]])
normalized = rms_reference(x_norm)
np.testing.assert_allclose(normalized[0],x_norm[0]/np.sqrt(3+1e-5))
np.testing.assert_allclose(normalized[1],np.ones(3),atol=2e-6)
```

先转换为浮点数再平方很重要。若输入是较小范围的整数 dtype，例如十六位整数，平方可能先在整数类型中溢出，再转换成浮点时已经无法恢复。仓库参考在归一化前统一转 float64，并有针对这类问题的回归测试。数值契约不仅是输出 dtype，还包括中间运算在哪种类型中发生。

epsilon 不只是一个可以随意删除的小数。全零向量的均方根在没有 epsilon 时为零，除法会失效；近零向量的缩放也会受到 epsilon 明显影响。比较实现时，应固定 epsilon 的位置和数值：放在平方根内与放在平方根外不是同一个公式。

归一化沿最后的特征轴逐位置执行，不能沿 token 轴或整个 batch 一起计算。如果两个 token 的特征完全不同，各自有自己的缩放因子。测试应使用不等轴长和不同幅度行，避免错误归约轴恰好得到相似结果。

## 第五课：pre-norm 的两次残差更新不能合并成一句话

本卷一层的核心顺序是：
$u=x+\mathrm{Attention}(\mathrm{RMSNorm}(x))$，
$y=u+\mathrm{SwiGLU}(\mathrm{RMSNorm}(u))$。
第二次归一化的输入是已经加入 attention 增量的 u，不是最初的 x。这个连接细节决定 FFN 能看到什么信息。

若你把两次归一化都对原始 x 计算，再写 $x+\mathrm{Attention}(n(x))+\mathrm{FFN}(n(x))$，就变成并行残差形式。它可以是一种有意设计的模型，但不等于当前串行结构。shape 全部正确也无法发现这种差异，必须检查数据依赖和数值对照。

pre-norm 保留了 residual stream 的直接相加路径，归一化主要发生在进入子层前。层末状态不一定具有单位 RMS；随着增量累积，其幅度可以变化。最终输出前通常再做一次 norm。本章 TinyDecoder 正是这样：所有层结束后对 residual stream 做 RMSNorm，再乘输出头。

一个有用的极简测试是把所有子层输出投影设为零。此时每层的 attention 增量和 FFN 增量都为零，residual stream 在层间保持初始 embedding。最终 logits 仍受最后 norm 和输出矩阵影响，所以不能期待整个模型直接返回 embedding。测试要明确观察的是层状态还是最终输出。

另一个测试是只打开 attention、关闭 FFN，再只打开 FFN、关闭 attention。这样可以分别检查两个子层对状态的贡献。若同时随机启用所有矩阵，一旦结果错误很难判断哪个路径出了问题。分阶段激活是理解完整模型的有效方法。

残差相加通常产生新数组或在受控条件下复用内存。教学代码应避免不明确的原地修改，尤其当你保存中间快照用于比较时。若记录列表里存的是同一个数组对象的引用，后续原地更新会让过去的“快照”一起变化，制造错误的运行轨迹。

## 第六课：SiLU 与 SwiGLU 的逐步推导

SiLU 定义为 $x\sigma(x)$，其中 sigmoid 为 $1/(1+\exp(-x))$。它在零处输出零，在正区域近似保留较大输入，在负区域可以保留小的负值。它不是 ReLU：ReLU 会把所有负数直接置零，SiLU 则是平滑函数，梯度行为不同。

取 x 为 log 三，sigmoid 等于四分之三，所以 SiLU 输出为四分之三乘 log 三。取 x 为负 log 三，sigmoid 等于四分之一，输出为负四分之一乘 log 三。这个可手算例子说明负输入并不全部被清零，也说明“gate”不一定只是零到一之间的概率，因为 SiLU 的最终输出还乘了 x。

SwiGLU 使用三张权重：gate 投影、up 投影和 down 投影。公式为
$$
f(x)=\big[\mathrm{SiLU}(xW_g)\odot(xW_u)\big]W_d.
$$
gate 与 up 都从 D 维投影到 F 维；逐元素相乘后仍为 F 维；down 再回到 D 维。这里的逐元素乘法不是矩阵乘法，也不是对 token 做路由。

```python
def sigmoid_stable(x):
    x = np.asarray(x,dtype=float)
    return np.exp(-np.logaddexp(0.,-x))

def silu_reference(x):
    x = np.asarray(x,dtype=float)
    return x*sigmoid_stable(x)

def swiglu_reference(x,wg,wu,wd):
    gate_pre = x @ wg
    up = x @ wu
    gated = silu_reference(gate_pre)*up
    return gated @ wd

value = np.log(3.)
np.testing.assert_allclose(silu_reference([value,-value]),
                           [0.75*value,-0.25*value])
```

为什么需要两条上投影路径？up 提供被调制的特征，gate 提供与当前输入相关的调制因子，两者在隐藏坐标上相乘。它比单纯先线性后激活再线性具有不同的参数化方式。不能把 gate 理解成“选择哪个专家”，因为这里没有按 token 选择不同参数组，所有位置仍使用同一组 dense 权重。

参数量也发生变化。普通两层 FFN 主要有二 D F 个权重，SwiGLU 主要有三 D F。若想在近似相同参数预算下与隐藏宽度四 D 的普通 FFN 比较，需要重新选择 F，而不是直接保持四 D 后说只换了激活函数。真实模型还会把 F 调整到适合硬件的倍数，具体数值应查配置。

本卷参考使用无偏置 SwiGLU，某些模型或实现可能有不同偏置设置。读取 checkpoint 时，参数名和形状都应与代码一致。看见一个叫 gate_proj 的张量不够，还要确认其前后运算、激活函数与乘法位置。

## 第七课：RoPE 先从二维旋转理解

二维旋转矩阵把向量 $[a,b]$ 变成 $[a\cos\theta-b\sin\theta,\ a\sin\theta+b\cos\theta]$。它改变方向，保持长度。RoPE 把每个头的特征坐标分成若干二维对，每对使用与位置相关的角度，对 Q 和 K 做旋转。

为什么只旋转 Q/K 而不必同样旋转 V？Q/K 参与匹配分数，旋转让点积能够表达相对位置关系；V 提供被读取内容。某些架构可以采用其他设计，但本卷和所读 Llama 路径中，RoPE 作用于 Q/K，V 不进入旋转。把 V 也机械旋转会改变模型定义。

取二维 q 为 $[1,0]$，位置 p 的旋转角为 p。位置零得到 $[1,0]$，位置一得到 $[\cos1,\sin1]$。若 query 位于位置二，key 位于位置一，两者旋转后的点积为 $\cos(2-1)=\cos1$。共同把两个位置都加十，差仍为一，点积不变。

这个性质来自旋转矩阵的组合：$R(p)^\top R(q)=R(q-p)$。因此 $(R(p)u)^\top(R(q)v)=u^\top R(q-p)v$。注意结果仍依赖原始内容向量 u、v，不是只剩一个位置距离函数。RoPE 将内容匹配与相对位置结合，而不是用位置替代内容。

```python
def rotate_pair(pair,angle):
    a,b = pair
    return np.array([a*np.cos(angle)-b*np.sin(angle),
                     a*np.sin(angle)+b*np.cos(angle)])

base = np.array([1.,0.])
q2 = rotate_pair(base,2.)
k1 = rotate_pair(base,1.)
np.testing.assert_allclose(q2@k1,np.cos(1.))
np.testing.assert_allclose(np.linalg.norm(q2),np.linalg.norm(base))
np.testing.assert_allclose(rotate_pair(base,12.)@rotate_pair(base,11.),q2@k1)
```

高维 RoPE 对不同坐标对使用不同频率。常见角度形式为 $p/base^{2i/d}$，其中 d 是每头参与旋转的维度，i 是坐标对索引。本卷使用相邻偶奇坐标成对；某些真实代码使用前后两半配对。两者可以通过一致坐标置换对应，但不能在不变换相关投影权重的情况下直接互换。

“旋转保持长度”不能证明 checkpoint 兼容。还需要核对配对布局、频率、旋转维度、位置编号、base 以及可能的缩放策略。一个实现数学上确实是旋转，仍可能与目标权重训练时使用的布局不同。源码工作坊会把这个差异具体映射到固定 Llama 文件。

## 第八课：把 RoPE 写成有形状契约的代码

本卷 RoPE 接受形状 $[\ldots,T,d]$ 的 Q 或 K，最后一维必须为偶数，positions 形状为 $[T]$。前面的省略轴可以包含 head 等批量维度。每个位置的角度沿所有头共享同一位置编号，但每个坐标对有自己的频率。

```python
def rope_reference(x,positions,base=10000.):
    x = np.asarray(x,dtype=float)
    positions = np.asarray(positions,dtype=float)
    width = x.shape[-1]
    if width % 2 or positions.shape != (x.shape[-2],):
        raise ValueError("expected [...,T,even_d] and [T] positions")
    frequency = base ** (np.arange(0,width,2)/width)
    angles = positions[:,None]/frequency
    cosine,sine = np.cos(angles),np.sin(angles)
    out = np.empty_like(x)
    out[...,0::2] = x[...,0::2]*cosine-x[...,1::2]*sine
    out[...,1::2] = x[...,0::2]*sine+x[...,1::2]*cosine
    return out

r = np.random.default_rng(11)
values = r.normal(size=(2,3,4))
rotated = rope_reference(values,np.arange(3))
np.testing.assert_allclose(np.sum(rotated**2,axis=-1),
                           np.sum(values**2,axis=-1),atol=1e-12)
```

输出使用新数组，避免计算奇坐标时误读已经更新过的偶坐标。如果你原地先写偶数位置，再用同一输入数组计算奇数位置，就会混用旋转前后的值，破坏正交变换。一个只检查 shape 的测试不会发现，范数保持与逆旋转测试可以发现。

positions 不能简单使用当前数组行号代替所有情况。完整前向从零到 T 减一较直观；缓存解码中当前一个 token 的数组长度为一，但逻辑位置可能是一百。M06 会详细讨论。如果单步解码总传位置零，输出与完整前向不一致，即使 K/V 拼接看起来正确。

长上下文扩展可能改变频率或位置缩放，但没有一个“把 base 改大”就对所有模型可靠的通用配方。模型训练长度、使用的具体 RoPE 变体和框架实现都需要核对。本章只实现明确的基础旋转定义，不对未训练长度的能力作保证。

测试 RoPE 至少包括位置零恒等、范数保持、共同平移后的 QK 点积不变、配对布局反例和不同头广播。不要只使用全零向量，因为任何错误线性变换对全零输入都可能看起来正确。非对称递增或随机值更有判别力。

## 第九课：QKV 投影、分组头与 attention 输出

一层中的 attention 输入是 RMSNorm 后的 residual state。Q 投影输出 Hq 个头，K/V 投影输出 Hkv 个头，每头维度 d 相同。课程实现将 Q/K 旋转后，把 KV 按组重复，交给已经验证的 attention 函数。这种显式 repeat 便于理解，但不是生产高效 GQA 的唯一方式。

若 Hq 为四、Hkv 为二，query 头零和一读取 KV 头零，query 头二和三读取 KV 头一。这里的分组约定必须与训练权重和后端一致。不能把 repeat_interleave 与整组平铺混淆：前者得到 KV 索引零、零、一、一，后者可能得到零、一、零、一，映射不同。

每个 query 头仍有自己的 Q，因此即使共享 K/V，它们的 attention 权重也可以不同。共享 KV 不等于把所有 query 头变成同一个头。输出仍然有 Hq 个头，合并后回到 D 维，再乘输出投影。缓存存储按 KV 头数计算，而注意力输出通道按 query 头数计算。

```python
from labs.models.core import split_heads,merge_heads,grouped_attention

ids = np.array([1,2,3])
x = model.embedding[ids]
normed = rms_reference(x)
q = rope_reference(split_heads(normed@block["q"],config.heads),np.arange(3))
k = rope_reference(split_heads(normed@block["k"],config.kv_heads),np.arange(3))
v = split_heads(normed@block["v"],config.kv_heads)
allowed = np.arange(3)[None,:] <= np.arange(3)[:,None]
heads_out,weights = grouped_attention(q,k,v,allowed)
attention_delta = merge_heads(heads_out)@block["o"]
assert q.shape == (4,3,4)
assert k.shape == v.shape == (2,3,4)
assert attention_delta.shape == x.shape
```

这段代码直接展开了 TinyDecoder 的 `_attn` 主要路径。每个变量都可以打印并与预期形状比较。这里使用的 `block` 是第一层权重，不能把第二层也误用同一字典，除非你有意构造参数共享模型。层数增加通常意味着不同的参数组，而不是多次调用完全同一组权重。

因果 mask 使用 key 的逻辑位置与 query positions 比较。完整前向中 positions 是零到 T 减一，所以得到普通下三角；缓存路径中 key 包含历史，query positions 是新增位置，需要矩形条件。将这段逻辑写成基于位置的表达式，比硬编码一个方阵模板更容易扩展。

## 第十课：完整 forward 的逐行数据流

TinyDecoder 的完整前向先校验 token 序列，查 embedding，建立位置数组，再遍历各层。每层先计算 attention 增量并相加，再计算基于更新后状态的 SwiGLU 增量并相加。最后做 RMSNorm 和词表投影。这个顺序可以写成一个独立、可对照的显式函数。

```python
def explicit_forward(model,tokens,return_trace=False):
    ids = np.asarray(tokens,dtype=np.int64)
    if ids.ndim != 1 or len(ids)==0 or np.any(ids<0) or np.any(ids>=model.config.vocab):
        raise ValueError("valid nonempty token sequence required")
    x = model.embedding[ids]
    positions = np.arange(len(ids))
    trace = [{"name":"embedding","value":x.copy()}]
    for layer,weights in enumerate(model.blocks):
        n1 = rms_reference(x)
        delta,_ = model._attn(n1,weights,positions)
        x = x+delta
        trace.append({"name":f"layer{layer}.after_attention","value":x.copy()})
        n2 = rms_reference(x)
        delta = swiglu_reference(n2,weights["gate"],weights["up"],weights["down"])
        x = x+delta
        trace.append({"name":f"layer{layer}.after_ffn","value":x.copy()})
    logits = rms_reference(x)@model.output
    return (logits,trace) if return_trace else logits

manual,trace = explicit_forward(model,[1,2,3],True)
reference = model.forward([1,2,3])
np.testing.assert_allclose(manual,reference,atol=1e-12)
assert [item["name"] for item in trace] == [
    "embedding","layer0.after_attention","layer0.after_ffn",
    "layer1.after_attention","layer1.after_ffn"
]
```

trace 保存 copy，是为了保留当时状态的独立快照。若某个实现后来改成原地相加，引用同一数组的旧记录会被覆盖；独立复制可以避免这个观察工具本身制造假象。记录中每个名字说明状态已经经过哪些子层，不把输入、增量和更新后状态混在一起。

为什么最终 norm 只执行一次？因为每层的 pre-norm 主要为子层输入提供尺度，而 residual stream 本身仍在累加；最终 norm 为输出投影提供定义明确的最终表示。若你把最终 norm 删除，或者在每层末尾额外加 norm，得到的是另一种网络。不能把归一化看成“哪里数值大就随便插一个”的修补工具。

输出 logits 的形状是 $[T,V]$。每行对应当前位置对下一 token 的打分，而不是对当前位置 token 的分类解释。训练标签如何右移在 M04/M14 中明确构造；生成时通常取最后有效位置。模型 forward 不执行采样，因此相同输入和参数下，这个 CPU 参考是确定性数值函数。

## 第十一课：参数数量与内存账本

默认模型的 embedding 有三十二乘十六等于五百一十二个参数，输出头同样五百一十二。每层 Q 和 O 各十六乘十六等于二百五十六；K 和 V 各十六乘八等于一百二十八。因此 attention 投影每层合计七百六十八。

SwiGLU 的 gate 和 up 各十六乘三十二等于五百一十二，down 为三十二乘十六也等于五百一十二，合计一千五百三十六。每层主要权重总数为二千三百零四，两层为四千六百零八，加 embedding 和输出头一千零二十四，总计五千六百三十二。课程没有 norm gain 和偏置，所以它们不计入此模型。

```python
parameter_count = model.embedding.size+model.output.size
parameter_count += sum(array.size for b in model.blocks for array in b.values())
assert parameter_count == 5632
```

如果参数保存为 float64，每个元素通常八字节，仅这些数组的原始数据约四万五千零五十六字节。这个数不包含 Python 对象、数组元数据、临时张量或缓存。它只是一个精确的参数数组账本，不是进程内存测量。真实模型使用不同 dtype 和额外状态，计算方式需要相应调整。

将层数加倍时，层参数部分近似加倍，但 embedding 和输出头不随层数变化。将词表大小加倍时，embedding 和输出头增长，层内投影不直接增长。将 hidden 加倍时，SwiGLU 参数增长，attention 参数不直接增长。这样的拆项分析比记住一个总参数数字更有迁移价值。

推理缓存还与序列长度、batch、层数、KV 头数和每头维度相关。训练则需要梯度、优化器状态和反向激活。一个只有五千多参数的小模型可以轻松放 CPU，但同样的公式扩大到真实尺度后，哪个部分占主导会随场景变化。M06 与框架卷会继续做资源估算。

## 第十二课：结构正确、训练成功与 checkpoint 兼容是三件事

结构正确意味着代码遵守你声明的计算图、形状和状态契约。训练成功意味着在明确数据与目标下，参数更新确实学到有用行为。checkpoint 兼容意味着模型参数的名称、形状、布局、归一化、位置机制和 tokenizer 等与某个保存格式对应。三者相关，但不能互相代替。

本章随机权重模型可以通过前向、缓存等价和因果性测试，却不会因此回答常识问题。一个训练过的小模型可以在人工语料上降低损失，却仍不兼容某个大型模型 checkpoint。一个能成功加载张量形状的实现，也可能因 RoPE 配对布局或 norm epsilon 不同而产生错误输出。

加载真实权重时，应先列出配置和参数映射表，逐项确认缺失或多余张量，不要默认忽略所有错误。对小输入比较中间层输出，比只看最终生成文本更有定位能力。文本生成还受到采样策略、模板和停止条件影响，不能作为唯一结构对照。

本卷源码工作坊使用固定版本 Llama 实现帮助建立映射，但不把 TinyDecoder 宣传成完整 Llama 复刻。它省略可学习 norm gain、一些配置分支和生产缓存接口，并使用明确的教学布局。理解差异本身就是源码阅读能力的一部分，不应通过扩大误差容忍度把不同定义强行测成一致。

### 12.1 一个两 token、二维隐藏状态的完整数值模型

为了真正跨过“看懂模块名”到“看懂模型代码”的门槛，我们再建立一个几乎所有数值都能手算的完整模型。词表只有三项，模型宽度二，只有一个 attention 头、一个 KV 头、一层，FFN 隐藏宽度也是二。输入使用编号零和一，embedding 分别为 $[1,0]$ 与 $[0,1]$。

Q、K、V 和 attention 输出投影都取单位矩阵。gate 和 up 也取单位矩阵，down 取零点一倍单位矩阵。输出头把二维最终表示 $[u,v]$ 映射为三个 logits：u、v、负 u 减 v。这个配置不是随机挑出来求一个结果，而是特意让每个阶段都可以用短公式追踪。

设 epsilon 为十万分之一，初始两行的均方都是二分之一，所以归一化放大系数 $r=1/\sqrt{0.5+\epsilon}$。归一化后的输入为 $[r,0]$ 和 $[0,r]$。位置零的 RoPE 不改变第一行；位置一把第二行旋转为 $[-r\sin1,r\cos1]$。Q 与 K 相同，V 保持未旋转的两行。

第一条 query 受因果 mask 限制，只能读第一条 value，因此 attention 输出为 $[r,0]$。第二条 query 对第一条 key 的缩放分数为 $-r^2\sin1/\sqrt2$，对第二条 key 的缩放分数为 $r^2/\sqrt2$。令第一项权重为 a，则 $a=1/[1+\exp(r^2(1+\sin1)/\sqrt2)]$，第二项为一减 a。

第二位置的 attention 输出因此为 $[ra,r(1-a)]$。加回原始 embedding 后，两行 residual state 分别为 $[1+r,0]$ 和 $[ra,1+r(1-a)]$。注意相加的是未归一化的原 residual x，不是 RMSNorm 后的表示。把归一化输入当残差底座，会改变这两行，属于常见实现错误。

接下来对这两行各自做第二次 RMSNorm，记为 n。因为 gate 与 up 都是单位矩阵，SwiGLU 的每个隐藏坐标变成 $\mathrm{SiLU}(n_i)n_i=n_i^2\sigma(n_i)$，down 再乘零点一。因此 FFN 增量为每个坐标的 $0.1n_i^2\sigma(n_i)$。最终状态是 attention 后状态加这个增量，再经 final RMSNorm 和输出头。

```python
small_config=DecoderConfig(vocab=3,dim=2,heads=1,kv_heads=1,layers=1,hidden=2,seed=0)
small=TinyDecoder(small_config)
small.embedding=np.array([[1.,0.],[0.,1.],[-1.,-1.]])
small.output=np.array([[1.,0.,-1.],[0.,1.,-1.]])
for name in ("q","k","v","o","gate","up"):
    small.blocks[0][name]=np.eye(2)
small.blocks[0]["down"]=0.1*np.eye(2)

eps=1e-5
radius=1/np.sqrt(0.5+eps)
first_score=-radius**2*np.sin(1.)/np.sqrt(2.)
second_score=radius**2/np.sqrt(2.)
a=1/(1+np.exp(second_score-first_score))
after_attention=np.array([
    [1+radius,0.],
    [radius*a,1+radius*(1-a)],
])
second_norm=rms_reference(after_attention)
ff_delta=0.1*second_norm*second_norm*sigmoid_stable(second_norm)
after_ffn=after_attention+ff_delta
expected_logits=rms_reference(after_ffn)@small.output
actual_logits,small_trace=explicit_forward(small,[0,1],True)
np.testing.assert_allclose(actual_logits,expected_logits,atol=1e-12)
np.testing.assert_allclose(small_trace[1]["value"],after_attention,atol=1e-12)
np.testing.assert_allclose(small_trace[2]["value"],after_ffn,atol=1e-12)
```

这段检查同时覆盖 RoPE、缩放、因果权重、两次不同输入的 RMSNorm、SwiGLU 和最终输出头。它比只比较两个复杂函数的最终结果更有解释力，因为中间 attention 后状态和 FFN 后状态都有独立闭式推导。若某一步错了，可以直接定位到对应阶段。

你可以做三次单独干预。把 down 设零，FFN 增量消失，其他阶段保持不变；把 attention O 设零，第二次 norm 的输入退化为原 embedding；把位置一错误改成零，第二个 query 的匹配分数改变，但第一个位置输出仍不受未来位置影响。每次都先预测哪张表会变化，再运行确认。

不要用最终 argmax 是否相同代替数值比较。三个 logits 可以发生明显变化但最大项仍然不变，生成一个 token 时看不出错误。模型结构测试应比较完整 logits 或中间张量，并使用明确容差。生成文本只观察离散选择，信息量远少于完整数值输出。

### 12.2 RMSNorm 的几何性质与反向传播

忽略 epsilon 时，RMSNorm 把向量缩放到固定均方根，保留方向而消除整体正比例幅度。若 x 乘一个正数 c，分子与分母同样乘 c，输出不变；若乘负数，方向反转，输出符号也反转。加入 epsilon 后，这种缩放不变性只近似成立，尤其在很小输入附近会明显偏离。

这并不意味着模型完全丢掉 residual stream 的幅度信息。pre-norm 结构把归一化结果送进子层，但原始 residual x 仍沿相加路径保留。子层看到的是受控尺度的方向信息，外层状态仍可以积累幅度。把 RMSNorm 后表示直接替代 residual stream，会改变这种分工。

对单个 token，记 $r=\sqrt{\mathrm{mean}(x^2)+\epsilon}$，输出 y 等于 x 除以 r。收到上游梯度 g 后，
$$
dx=g/r-x\,\mathrm{mean}(gx)/r^3.
$$
第一项来自分子 x 的直接变化，第二项来自分母 r 依赖 x 的变化。不能只把梯度除以 r，因为改变一个坐标还会影响所有坐标共同使用的缩放。

与 LayerNorm 相比，这个公式没有减去 mean(g) 的项，因为前向没有去均值。这个差别不仅影响前向，也影响梯度路径。把 LayerNorm 的反向代码简单删掉一个前向减法，却保留原反向公式，会得到错误训练，即使推理输出看起来已经像 RMSNorm。

```python
def rms_backward(x,upstream,eps=1e-5):
    x=np.asarray(x,dtype=float)
    upstream=np.asarray(upstream,dtype=float)
    radius=np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)
    coupling=np.mean(upstream*x,axis=-1,keepdims=True)
    return upstream/radius-x*coupling/(radius**3)

xx_norm=np.array([[1.,2.,-1.]])
gg_norm=np.array([[3.,-2.,4.]])
analytic=rms_backward(xx_norm,gg_norm)
numeric=np.zeros_like(xx_norm)
for index in np.ndindex(xx_norm.shape):
    plus=xx_norm.copy();minus=xx_norm.copy()
    plus[index]+=1e-5;minus[index]-=1e-5
    numeric[index]=(np.sum(rms_reference(plus)*gg_norm)
                    -np.sum(rms_reference(minus)*gg_norm))/(2e-5)
np.testing.assert_allclose(analytic,numeric,atol=1e-7,rtol=1e-6)
```

如果有逐特征 gain，先把上游梯度乘 gain，再计算核心输入梯度；gain 梯度则是 upstream 乘归一化输出，在所有共享 token 与 batch 位置上求和。课程 TinyDecoder 没有这组参数，所以参数账本不包含它；真实模型通常需要检查具体 norm 实现，而不能根据函数名字自动加或删。

在 epsilon 为零的理想数学情形，沿 x 自身的纯尺度方向不会改变归一化输出，因此梯度与 x 的内积为零。实际使用正 epsilon 后，这个性质只近似成立。它可以作为理解几何的辅助检查，却不应作为所有输入下的严格零断言，特别是近零向量。

### 12.3 SwiGLU 的完整梯度与参数使用关系

设 $a=xW_g$、$b=xW_u$、$h=\mathrm{SiLU}(a)\odot b$、$y=hW_d$。收到输出梯度 G 后，先有 $dW_d=h^\top G$、$dh=GW_d^\top$。逐元素乘法反向给出 $db=dh\odot\mathrm{SiLU}(a)$，$da=dh\odot b\odot\mathrm{SiLU}'(a)$。

SiLU 导数为 $\sigma(a)+a\sigma(a)(1-\sigma(a))$。然后 $dW_g=x^\top da$、$dW_u=x^\top db$，输入梯度是两条路径之和 $dx=daW_g^\top+dbW_u^\top$。这里再次出现共享输入的梯度累加：x 同时影响 gate 和 up，漏掉任何一条都不是正确导数。

为什么 gate 的梯度会乘 b？因为改变 gate 只有在被调制特征 b 不为零时才影响乘积。为什么 up 的梯度会乘 SiLU(a)？因为上游内容变化被当前 gate 值缩放。这些相互依赖解释了门控结构的训练行为，但不能简单概括为“gate 关了就永远学不动”，因为不同坐标、残差和后续步骤都会改变状态。

```python
def swiglu_backward(x,wg,wu,wd,upstream):
    a=x@wg;b=x@wu
    sigmoid=sigmoid_stable(a)
    gate=a*sigmoid
    hidden=gate*b
    dwd=hidden.T@upstream
    dh=upstream@wd.T
    da=dh*b*(sigmoid+a*sigmoid*(1-sigmoid))
    db=dh*gate
    return {
        "x":da@wg.T+db@wu.T,
        "gate":x.T@da,
        "up":x.T@db,
        "down":dwd,
    }

rng_grad=np.random.default_rng(22)
input_grad=rng_grad.normal(size=(2,3))
wg_grad=rng_grad.normal(size=(3,4))
wu_grad=rng_grad.normal(size=(3,4))
wd_grad=rng_grad.normal(size=(4,3))
upstream_grad=rng_grad.normal(size=(2,3))
derivatives=swiglu_backward(input_grad,wg_grad,wu_grad,wd_grad,upstream_grad)
parameters=[input_grad,wg_grad,wu_grad,wd_grad]
names=["x","gate","up","down"]
for part,name in enumerate(names):
    numerical=np.zeros_like(parameters[part])
    for index in np.ndindex(numerical.shape):
        plus=[a.copy() for a in parameters]
        minus=[a.copy() for a in parameters]
        plus[part][index]+=1e-5;minus[part][index]-=1e-5
        lp=np.sum(swiglu_reference(*plus)*upstream_grad)
        lm=np.sum(swiglu_reference(*minus)*upstream_grad)
        numerical[index]=(lp-lm)/(2e-5)
    np.testing.assert_allclose(derivatives[name],numerical,atol=1e-7,rtol=1e-6)
```

这项完整差分检查让你能够把三张权重的角色与梯度对应起来。只有前向和最终 loss 下降还不够，因为错误梯度有时也会沿某个下降方向移动。尤其门控分支多，必须检查每张矩阵和输入的独立偏导，才能发现遗漏路径或错用激活导数。

做性能优化时，gate 与 up 投影可以合并到一次更宽的矩阵乘法，再把结果切成两半；这在参数排列正确时可以保持数学等价。合并的是执行方式，不是把两张矩阵强制相等。你应先构造拼接权重并验证输出、梯度一致，再讨论是否真的减少了启动或内存开销。

### 12.4 RoPE 的布局变换、反向和缓存约定

相邻成对布局可以写为 $[a_0,b_0,a_1,b_1]$，分半布局可以写为 $[a_0,a_1,b_0,b_1]$。它们表示同样两组二维坐标，但元素存放顺序不同。用排列 $[0,2,1,3]$ 可以把前者转成后者；旋转后再用逆排列恢复，才适合逐项比较。

如果输入特征由线性投影产生，改变输出坐标顺序意味着投影权重的列也要做一致置换。对行向量约定 $q=xW_Q$，把 q 的列重排，等价于重排 $W_Q$ 的输出列。不能只改 RoPE 函数而保留原权重排列，否则旋转会配对错误的特征。

Q 和 K 的相关布局必须一致，才能保持点积语义。V 不参与这个旋转，所以不一定需要同样的坐标置换；具体是否还有其他布局要求，要看整个头维度组织和后端接口。迁移 checkpoint 时应写一张参数与激活坐标映射表，而不是凭“都是四维”判断兼容。

RoPE 的反向相对简单：对一个正交旋转 y 等于 R x，输入梯度为 $R^\top g$，也就是按相反角度旋转上游梯度。它不会凭空改变梯度范数，但与前后投影、归一化和其他路径组合后，完整层梯度仍可能放大或缩小。局部正交性质不能替代整网稳定性分析。

```python
pair=np.array([1.2,-0.7])
up=np.array([0.3,2.])
angle=0.8
analytic_pair=rotate_pair(up,-angle)
numeric_pair=np.zeros(2)
for i in range(2):
    plus=pair.copy();minus=pair.copy()
    plus[i]+=1e-5;minus[i]-=1e-5
    numeric_pair[i]=(rotate_pair(plus,angle)@up-rotate_pair(minus,angle)@up)/(2e-5)
np.testing.assert_allclose(analytic_pair,numeric_pair,atol=1e-8)
```

缓存通常保存已经应用对应位置旋转的 K。这样历史 key 的位置固定，后续新 query 只需使用自己的新位置即可比较。若缓存保存旋转前 K，也可以设计正确系统，但每次读取需要以正确位置处理，接口和计算成本不同。混用“缓存前旋转”和“读取时旋转”两套假设，会导致重复旋转或漏旋转。

位置偏移变化可能使缓存失效。比如把同一 token 前缀放到不同位置起点，若模型的位置机制依赖绝对位置或某些缩放状态，旧 K 未必能直接复用。即使基础 RoPE 点积具有共同平移性质，完整系统还可能有其他位置依赖或不同缓存布局，不能据此无条件共享任意偏移的 KV。

### 12.5 为什么完整 decoder 保持目标因果性

可以按层证明前缀非干扰。初始位置 i 的 embedding 只依赖自己的 token；RoPE 使用位置 i 的已知编号；RMSNorm 和 FFN 逐位置计算，不读取别的位置。attention 在位置 i 只允许读取不晚于 i 的 K/V，因此其输出只依赖前缀。残差相加组合的两项都只依赖同一允许前缀，所以结果仍然因果。

把这个论证重复到下一层，前一层每个位置的表示已经只依赖对应前缀；下一层仍使用同样的因果读取规则，因此不会引入未来依赖。最终 norm 和输出头逐位置处理，也保持该性质。这个归纳说明为什么可以在完整模型上测试“改变未来 token，过去 logits 不变”。

证明依赖于若干实现条件。不能在 token 轴上做一个跨全序列统计后再用它归一化每个位置；不能让某个全局特征从未来位置汇总进来；不能在训练时把未来标签作为额外输入；不能让不同样本共享可变状态。仅仅存在一个下三角 attention mask，并不足以保证任意外层代码都因果。

这也是为什么把 LayerNorm 或 RMSNorm 沿错轴计算会严重破坏模型，而不只是改变缩放大小。如果归一化统计跨 token 轴，早期位置会通过均值或均方根间接读取未来信息。shape 可能仍然正确，甚至训练损失更低，但生成时信息条件已经不一致。

一个可靠测试应在多层完整模型上进行，并使用不同输入长度、不同未来干预和多种非退化参数。若只测试零权重模型，所有输出可能都相同，因果测试当然通过，却没有检验真实路径。测试数据需要让允许路径能产生变化，同时禁止路径保持不变。

### 12.6 从模型参数到训练工程的边界

TinyDecoder 的 forward 是完整的，但它没有为所有参数实现训练梯度。你可以在 PyTorch 中照同样结构实现并使用自动微分，也可以逐个组合本章和 M03 的手写反向。两条路线都要保留相同的输入、位置、mask、norm epsilon 和输出定义，才能比较结果。

训练一个 dense decoder 不仅要有结构，还需要语料、tokenizer、下一 token 标签、优化器和评估。M15 的可训练小模型先用较简单结构让每个梯度都可检查；可选 PyTorch Transformer 项目展示完整自回归训练。不要把“某个 NumPy 前向类没有 backward”误解为它的结构不完整，也不要把有 forward 误解为训练系统已经完整。

若实现手写反向，残差会带来两条输入梯度：直接路径的上游梯度，以及穿过子层和 norm 的梯度。对串行两子层，必须按逆序先处理 FFN 残差，再处理 attention 残差。将两个增量的梯度独立算完后随意相加，可能忽略 FFN 对 attention 更新后状态的依赖。

参数共享也需要明确。输入 embedding 与输出头可以绑定为同一矩阵的转置，但本卷默认是独立矩阵。绑定后同一参数会从输入查表路径和输出分类路径同时收到梯度，必须累加。只在 forward 中让二者数值相同、训练时却分别更新两个副本，并不等于真正权重绑定。

训练检查建议从单个 batch 过拟合开始，再加入验证集、保存恢复和生成。若一个极小固定任务都无法降低损失，优先检查标签 shift、mask、梯度和更新顺序，而不是先上更大 GPU。资源增加不会修复计算图或数据对齐错误。

### 12.7 读真实 decoder 源码时怎样定位每个责任

先找到配置类，记录 vocab、hidden size、intermediate size、层数、query/KV 头数、norm epsilon 和位置相关字段。然后找到 embedding、decoder layer、attention、MLP、final norm 和语言模型头。不要从所有导入和辅助工具开始漫游；用这张主干图限定第一轮阅读范围。

在 attention 里先追三件事：投影后 shape 怎样变化，RoPE 在 cache 更新前后哪里执行，后端函数收到什么 mask 与 scaling。接着检查输出如何合头与投影。不同 backend 可能融合很多中间步骤，但模型定义的职责仍能映射到这些边界。

在 MLP 里确认激活是固定 SiLU 还是由 config 选择，是否有 bias，gate/up/down 的形状是什么。在 norm 里确认计算 dtype、epsilon 位置、是否含 gain。某个类名叫 LlamaMLP 或 RMSNorm，并不足以替代这些具体检查；同名接口也可能随版本演化。

缓存接口可能把实际 K/V 存储交给独立对象，而不是像课程一样返回 Python 列表。读模型层时可以先把它当成“按层号保存和更新状态”的契约，再进入框架卷追物理页、分配和调度。把模型数学与缓存管理分层阅读，能避免一开始就在巨大调用链中迷失。

最后用一个固定输入建立中间张量对照表，而不是只比较最终文本。若 embedding 已不同，检查 tokenizer 与权重；若第一层 QKV 不同，检查矩阵布局；若 RoPE 后不同，检查位置与配对；若 attention 后不同，检查 mask/backend；若 FFN 后不同，检查激活与 norm。这样的定位顺序能显著缩小排查范围。

### 12.8 模型配置中哪些改变需要重新训练或转换

将 hidden 从十六改成三十二，不只是“给模型更多空间”，而是改变几乎所有参数形状，原权重不能直接原样加载。将层数减少，会丢弃部分计算路径；将 KV 头数减少，也涉及投影与表示约束。某些研究提供训练后转换或继续训练方法，但不能把任意删头、裁矩阵当成通用无损优化。

有些改变看起来不改参数形状，仍会改变函数。例如 norm epsilon、RoPE base、激活函数和残差顺序。加载器可能不报 shape 错误，输出却发生系统性变化。因此 checkpoint 兼容检查必须包含配置值与计算语义，不能只检查张量尺寸。

还有些改变属于执行实现，例如在相同定义下把两个投影合并、把 attention 换成经过验证的融合后端、调整物理缓存布局。它们的目标是保持数学契约，但浮点顺序可能带来容差内差异。正确做法是用参考实现和代表性输入建立等价证据，再测性能，而不是仅凭“优化”标签假定无影响。

对于用户可随时使用的个人 CPU 或 5070 Ti，本阶段主要目标是建立正确性和源码理解；H100 更适合后续真实框架与 GPU kernel 的受限实验。不要为了“像大模型工程”而把一个两 token 形状错误搬到八卡服务器上调试。先在小输入上把每个索引和数值说清，之后扩大规模才会更有把握。



## 第十二课续：把结构知识变成可检查的工程判断

### 十二点九　输出头究竟把隐藏状态变成了什么

一个常见误解是把隐藏向量的某个坐标当作一个词的概率。隐藏维度和词表维度是两种不同的轴。设当前隐藏向量只有两个分量，词表只有三个词，输出矩阵三列依次为“向第一坐标取证”“向第二坐标取证”和“同时反对两个坐标”。取 $h=(2,-1)$，输出矩阵为 $W=\bigl[\begin{smallmatrix}1&0&-1\\0&1&-1\end{smallmatrix}\bigr]$。矩阵乘法给出 logits $(2,-1,-1)$。第一个词的分数来自 $2\times1+(-1)\times0=2$，第二个来自 $2\times0+(-1)\times1=-1$，第三个来自 $2\times(-1)+(-1)\times(-1)=-1$。分数能够为负，因为它们还不是概率。

对三个分数减去最大值二，得到 $(0,-3,-3)$。指数为 $(1,0.049787,0.049787)$，分母为 $1.099574$，概率约为 $(0.909443,0.045279,0.045279)$。如果正确词是第二个，损失为 $-\log0.045279\approx3.094923$。这个损失较大不意味着整个隐藏状态“毫无信息”，只表示经过当前输出头后，正确词的相对概率较低。反向传播既会调整输出头，也会把误差信号送回隐藏状态，再穿过全部残差层和 embedding。只有输出头被冻结时，前面网络才被迫独自承担改变表示的任务。

继续手算输出头梯度，正确类独热向量为 $(0,1,0)$，所以 logits 梯度为 $(0.909443,-0.954721,0.045279)$。对输出矩阵求导得到外积 $h^\top g$。第一行是 $(1.818886,-1.909442,0.090558)$，第二行是 $(-0.909443,0.954721,-0.045279)$。对隐藏状态的梯度是 $gW^\top$，第一分量为 $0.909443-0.045279=0.864164$，第二分量为 $-0.954721-0.045279=-1$。这种明确的数值链条说明，模型不是把一个词直接藏在某个坐标，而是学习一个能够被输出头读取的分布式表示。

当词表很大时，输出投影和损失也可能成为重要成本。对于一个隐藏向量，朴素输出头需要约 $DV$ 次乘加；对长度为 $T$ 的全部位置，需要约 $TDV$。训练时通常需要每个非忽略位置的 logits，因为每个位置都有监督目标。生成时若只需要下一个 token，可以只取最后位置的隐藏状态再做输出投影。在缓存解码中，本来就只有一个新位置，因此这个优化自然发生。但不能为了省计算而在训练时只保留最后位置，却仍声称使用了原先的全位置语言建模目标。

### 十二点十　相同结构为什么仍然会有不同函数

比较两个实现时，首先分清数学结构、参数约定和数值执行。数学结构包括子层先后、归一化位置、注意力可见性和激活函数。参数约定包括矩阵在文件中是 $[out,in]$ 还是 $[in,out]$、头排列、RoPE 配对、词表排列和是否绑权重。数值执行包括浮点类型、归约顺序、融合内核和近似函数。三类差异会表现成不同错误，诊断时不能混在一起。

例如 NumPy 本章用 $XW$，其权重形状是输入维度乘输出维度；常见线性层把参数存成输出维度乘输入维度，执行时计算 $XW^\top$。如果两者维度恰好相等，把矩阵直接复制过去不会触发 shape 错误，却会把函数变成另一个函数。以 $W=\bigl[\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\bigr]$、$x=(1,0)$ 为例，正确右乘得到 $(1,2)$，误转置得到 $(1,3)$。这个最小测试比在完整模型末尾观察一大串乱码更能定位问题。

再如归一化的 $\epsilon$ 常被当作无关常数。当输入均方远大于它时，两种小正数影响确实可能很小；当隐藏向量接近零时，分母主要由 $\sqrt\epsilon$ 决定。取 $x=(10^{-4},0)$，均方是 $5\times10^{-9}$。若 $\epsilon=10^{-5}$，第一分量归一化后约为 $0.031615$；若 $\epsilon=10^{-6}$，约为 $0.099751$。差异超过三倍。读取 checkpoint 时必须保留配置里的归一化常数，不能因为“都是很小的数”而自行替换。

还要区分函数相等和逐位相等。两个正确的矩阵乘法内核可能使用不同的加法归约树，浮点加法不满足严格结合律，末位差异可能积累。验证时先用小尺寸双精度参考确认公式，再用目标类型制定绝对和相对容差，最后检查任务级指标。不能把所有大差异都解释成浮点误差；也不能把合理范围内的舍入差异当成结构错误。比较应尽量从第一处出现差异的中间张量开始，而不是只比较最终文本。

### 十二点十一　完整参考解：串行依赖、参数增量与轨迹

第七题的错误版本必须在同一套参数下比较。下面只改 FFN 的读取点：正确版本用 attention 更新后的状态，错误版本用更新之前的状态。这样控制变量后，输出差异才证明依赖关系不同。错误示例不是另一种架构一定不可训练，而是它没有实现本章承诺的串行架构。若你打算研究并行残差，应单独定义模型、训练配置和测试，而不能把误写当作等价改写。

```python
def wrong_parallel_forward(m, ids):
    ids = np.asarray(ids, dtype=np.int64)
    x = m.embedding[ids].copy()
    for b in m.blocks:
        previous = x.copy()
        x = x + m._attn(rms_reference(x), b, np.arange(len(ids)))[0]
        x = x + swiglu_reference(rms_reference(previous),
                                 b["gate"], b["up"], b["down"])
    return rms_reference(x) @ m.output

probe_ids = [1, 2, 3, 4]
assert np.max(np.abs(wrong_parallel_forward(model, probe_ids)
                     - model.forward(probe_ids))) > 1e-8

def all_parameter_count(m):
    return (m.embedding.size + m.output.size
            + sum(a.size for b in m.blocks for a in b.values()))

larger_config = DecoderConfig(
    vocab=config.vocab, dim=config.dim, heads=config.heads,
    kv_heads=config.kv_heads, layers=config.layers,
    hidden=48, seed=config.seed)
larger_model = TinyDecoder(larger_config)
assert all_parameter_count(larger_model) - all_parameter_count(model) == 1536
```

第十一题需要真正的不可变轨迹。这里不依赖前面示范的 trace 字段名称，而定义一个只保存数组副本的最小记录器。每层保存 attention 更新后与 FFN 更新后两个状态。修改第二层 K 的一个元素之后，第零层的两个记录严格一致，第二层的 attention 更新后开始出现差异。取副本是为了使“过去的记录”保持过去的值。如果保存的只是同一块可变缓冲区的引用，那么记录可能随之后的原地操作一起改变，调试证据就不可信。

```python
import copy

def layer_snapshots(m, ids):
    x = m.embedding[np.asarray(ids, dtype=np.int64)].copy()
    snapshots = []
    for b in m.blocks:
        x = x + m._attn(rms_reference(x), b, np.arange(len(ids)))[0]
        snapshots.append(x.copy())
        x = x + swiglu_reference(rms_reference(x),
                                 b["gate"], b["up"], b["down"])
        snapshots.append(x.copy())
    return snapshots

modified_model = copy.deepcopy(model)
modified_model.blocks[1]["k"][0, 0] += 0.4
before_trace = layer_snapshots(model, probe_ids)
after_trace = layer_snapshots(modified_model, probe_ids)
np.testing.assert_array_equal(before_trace[0], after_trace[0])
np.testing.assert_array_equal(before_trace[1], after_trace[1])
assert np.max(np.abs(before_trace[2] - after_trace[2])) > 1e-8
```

第十二题的完整参考就是前文的 explicit_forward，但独立复写时应增加接口检查。输入是一维整数 token 序列；空序列没有本章定义的预测位置；负数在 NumPy 中具有从末尾索引的特殊含义，必须先拒绝，不能让它悄悄指向最后一行 embedding；超出词表的 ID 同样应拒绝。浮点数即使看起来是整数，也最好在数据入口显式转换并校验，避免把错误数据源悄悄接受进训练管道。工程接口的清晰程度会影响你能否复现模型，而不只是影响报错好不好看。

### 十二点十二　从随机前向走向一个真实训练实验

本章随机权重模型的用途是验证结构，不是展示自然语言能力。要把结构变成可训练模型，需要为所有参与运算的参数建立梯度路径，并定义优化器、数据与评估。若使用自动微分框架，线性层参数应注册为可训练张量，不能在每次 forward 里用 NumPy 重新生成，也不能把张量转换成普通数组后再转回来；后者通常切断计算图。若手写反向，则要为 attention、RoPE、RMSNorm、SwiGLU 和每次残差相加分别实现导数，再把各分支梯度累加到共享输入。

训练最小试验可以选择固定长度的符号复制或周期预测。先固定二十个训练样本，用小模型过拟合，观察训练损失能否显著下降，再引入独立验证集。过拟合小集合是诊断工具：它能发现梯度完全断开、目标错位、更新没有生效等问题；它不能证明模型学会了通用规律。第二步应改变随机种子和样本排列，检查是否只是记住某个固定位置。第三步才扩大数据并考察未见样本，记录训练与验证曲线是否分离。

生成实验要说明采样方法。最大概率选择会放大模型的局部偏好；温度采样会增加随机性；重复惩罚会改变分布。这些外层策略都不能修复错误的因果 mask 或错位的标签。如果模型总是输出同一个词，先看训练数据频率、标签构造、损失曲线、输出 logits 范围和 embedding 更新，再尝试采样技巧。把所有退化输出都归因于温度过低，会掩盖真正的实现错误。

还要把参数量和训练状态量分开。一个三十二位参数占四字节，但 Adam 常常还需要梯度、一阶矩和二阶矩，混合精度方案可能另有主权重副本。激活也需要保存给反向传播，attention 的中间矩阵可能随序列长度平方增长。估算硬件需求时，应分别列参数、优化器状态、梯度和激活，注明类型及是否重计算。本章的小模型在 CPU 上足够进行结构验证；进入更大模型训练之前，再依据测量决定是否使用个人显卡或短时服务器 GPU。

### 十二点十三　独立项目：构造一个可读的模型检查器

项目产物不是一个只打印总参数量的脚本，而是一份与配置和实际数组一致的结构报告。第一部分记录词表、隐藏维度、查询头数、KV 头数、头维度、层数和 FFN 中间宽度。第二部分逐层列出每个矩阵的形状、元素数和数据类型。第三部分运行一个三 token 样例，保存 embedding、每层 attention 前的 norm、Q/K/V、每层残差输出和最终 logits 的摘要。摘要至少包含形状、有限性、均值、均方根和最大绝对值。

验收应包含人为故障。把一个权重转置，检查报告能否通过逐层数值比较定位；把头维度设成奇数，检查 RoPE 是否在构建阶段拒绝；把 KV 头数设成不能整除查询头数，检查分组映射是否被拒绝；把 token 改成负数，检查输入层是否报错。每个故障都应有明确的失败原因，而不是等 NumPy 在深层运算中抛出难懂异常。恢复正确配置后，同一份报告应重新通过所有结构和因果性检查。

最后写一页解释，回答为什么增加层数、增加隐藏维度和增加 FFN 宽度会改变不同的成本；为什么减少 KV 头主要影响缓存和 K/V 投影；为什么更长的上下文会增加注意力计算，却不会增加固定模型的参数数量。能把这几种增长方向区分开，就已经开始用工程方式理解模型结构，而不再只是背诵名称。下一章会把这张成本表进一步展开到 prefill、逐 token 解码和缓存生命周期。


### 十二点十四　两个容易忽略的边界：位置原点与样本边界

假设一句话在自己的请求里从位置零开始，另一句在位置一百开始。对于采用固定频率 RoPE 的单次自注意力，如果所有 token 位置整体平移同一个常数，且没有额外绝对位置机制，成对点积保持不变。这并不表示你可以随意给一个缓存里的部分 token 改编号。共同平移要求历史键和新查询遵守同一个坐标变换；只改新 token 的位置会改变相对距离，从而改变注意力分数。保存缓存时应把位置计数看成状态的一部分，不能只保存 K/V 数组。

样本边界更不能靠位置编号代替。把两个训练样本直接拼接成一条长序列，再只使用普通下三角 mask，第二个样本会看见第一个样本。这可能是有意的连续语料训练，也可能是数据管道错误。若你的目标是彼此独立的样本，需要同时检查因果关系和所属样本编号相等：一个查询只允许读取同一段中不晚于自己的键。把每段位置重新归零只是改变位置编码，不会自动屏蔽跨段连接。

以两个长度分别为二和三的样本为例，拼接后索引为零到四，段编号为 $(0,0,1,1,1)$。第三个位置虽然在全局因果顺序上能看到前两个位置，但它们段编号不同，所以正确的分块因果 mask 应禁止连接。第三行只有自身合法，第四行可读第三与第四位置，第五行可读第三到第五位置。这个例子说明，mask 是数据语义的一部分，不能只根据矩阵尺寸生成。

在评估中，类似问题会导致更隐蔽的泄漏。若把验证答案放在输入序列里而目标移位错误，模型可能直接读取正确答案；若把同一长文切成高度重叠的训练和验证窗口，验证集虽然文件不同，内容却几乎一样。架构实现正确并不能保护你免于这些数据错误。验收完整模型时，必须同时审查 token 级目标、段级可见性和数据集级隔离。

第十五题作为综合自测：写出长度二与长度三的分块因果矩阵，并证明改变第一个样本的任意 token 不影响第二个样本的所有隐藏状态。证明从 embedding 的逐位置独立性开始，对每层归纳：归一化和 FFN 不跨位置，attention 的可读键全部属于同段，残差只合并同一位置。任何一步若使用全序列均值归一化、未分段的 attention 或跨样本卷积，这个证明都需要重新检查。这样你学到的就不只是一段模型代码，而是一套判断信息能否越界的办法。


分块 mask 的可运行答案如下。第一项比较段编号，第二项比较时间索引，两者使用逻辑与连接。注意这里的 True 表示允许访问；若接入一个把 True 定义为禁止的接口，需要在边界显式取反。应把这个语义写在函数文档中，避免不同库之间迁移时把屏蔽方向颠倒。
```python
segments=np.array([0,0,1,1,1])
indices=np.arange(5)
packed_mask=(segments[:,None]==segments[None,:]) & (indices[:,None]>=indices[None,:])
np.testing.assert_array_equal(packed_mask, np.array([
    [1,0,0,0,0],[1,1,0,0,0],[0,0,1,0,0],
    [0,0,1,1,0],[0,0,1,1,1]],dtype=bool))
```

## 第十三课：十二道实现练习与完整答案

**练习一：列出每张权重形状。** 使用默认配置，答案见第二课和参数账本。重点是 K/V 输出宽度八而 Q 输出宽度十六；gate/up 输出三十二；down 回到十六。若所有矩阵都写成十六乘十六，虽然部分运算可能能跑，但已经丢掉 GQA 和隐藏扩展定义。

**练习二：RMSNorm 与 LayerNorm 反例。** 对常量向量 $[2,2,2]$，RMSNorm 输出接近全一，LayerNorm 核心输出全零。答案必须说明是否包含仿射参数和 epsilon，不能只比较函数名称。

**练习三：整数溢出。** 用 int16 输入较大整数，比较先平方后转浮点与先转 float64 后平方。参考解如下。这个故障与模型理论无关，却能破坏所有后续归一化数值。

```python
integers = np.array([300,300],dtype=np.int16)
wrong = (integers*integers).astype(np.float64)
right = integers.astype(np.float64)**2
assert not np.array_equal(wrong,right)
np.testing.assert_array_equal(right,[90000,90000])
```

**练习四：SwiGLU 逐段输出。** 固定一组二维输入和三张小矩阵，分别打印 gate_pre、SiLU gate、up、逐元素积、down 输出。参考答案的验收不是某个随机小数，而是每一步形状正确、最终与 `swiglu_reference` 一致，以及将 up 置零后输出必为零。

```python
rng = np.random.default_rng(4)
xx = rng.normal(size=(2,4))
wg,wu,wd = rng.normal(size=(4,6)),rng.normal(size=(4,6)),rng.normal(size=(6,4))
np.testing.assert_array_equal(swiglu_reference(xx,wg,np.zeros_like(wu),wd),
                              np.zeros((2,4)))
```

**练习五：RoPE 位置零。** 任意输入在位置零的旋转应保持原值。再检查非零位置范数保持。只使用全零输入不合格，因为错误实现也可能通过。

**练习六：RoPE 相对位置。** 对随机二维 q、k，比较位置 p、r 的点积与共同加偏移后的点积。参考测试固定同一频率和同一配对布局；如果改变频率策略，不能继续无条件套用原性质。

```python
u=np.array([1.,2.]);v=np.array([-1.,3.])
np.testing.assert_allclose(rotate_pair(u,2.)@rotate_pair(v,5.),
                           rotate_pair(u,12.)@rotate_pair(v,15.),atol=1e-12)
```

**练习七：残差更新顺序。** 实现错误版本，让 FFN 使用最初 x 的 norm，而不是 attention 更新后的状态。比较一个非退化随机配置，输出应不同。答案应指出差异来自串行与并行依赖，而不是浮点舍入。

**练习八：关闭子层。** 把所有 attention O 和 FFN down 设零，验证最终 logits 等于初始 embedding 经 final RMSNorm 和输出头的结果。完整参考保留独立模型，避免破坏其他练习使用的对象。

```python
zero_model=TinyDecoder(config)
for b in zero_model.blocks:
    b["o"].fill(0.)
    b["down"].fill(0.)
ids_zero=np.array([1,2,3])
expected_zero=rms_reference(zero_model.embedding[ids_zero])@zero_model.output
np.testing.assert_allclose(zero_model.forward(ids_zero),expected_zero)
```

**练习九：未来 token 非干扰。** 将序列最后一个 token 改变，前面的 logits 应保持一致。完整模型测试比只检查 attention mask 更强，因为它同时覆盖位置、残差和 FFN 的局部性。

```python
original=model.forward([1,2,3,4])
changed=model.forward([1,2,3,9])
np.testing.assert_allclose(original[:3],changed[:3],atol=1e-12)
```

**练习十：参数账本。** 将 hidden 改成四十八，推导总参数增量。每层 SwiGLU 增加 $3D(48-32)$，两层增加一千五百三十六，其他权重不变。用实际数组 size 求和验证，而不是凭模型名称估计。

**练习十一：中间轨迹定位。** 故意在第二层改变一个 K 投影元素，比较 trace。第一层结束状态应不变，第二层 attention 之后开始变化，最终 logits 通常变化。若第一层记录也跟着变，检查快照是否共享了被后续修改的数组。

**练习十二：独立复写完整 forward。** 不复制 TinyDecoder.forward，按本章形状表和公式写自己的版本。最低验收包含非法 ID、非空要求、每层顺序、最终 norm、因果性、关闭子层退化和与固定参数参考的数值一致性。不能靠在函数内部直接调用 model.forward 通过，这样没有验证自己的实现。

本章掌握门槛是你能把一条 token 序列从 embedding 一直追到词表 logits，明确每个投影、旋转、归一化和残差更新的输入输出。你还应能解释哪些组件属于模型结构，哪些属于生成与服务外层，以及为什么正确的 shape 不足以证明 checkpoint 兼容。下一章将让同一个模型从“每次重算全部前缀”变成“只计算新 token 并复用历史”，并逐项证明缓存没有改变定义。
