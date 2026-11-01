# M04 原始编码器—解码器 Transformer：把所有组件组装起来

“Transformer 就是 attention”是不完整的理解。原始 Transformer 是一个完整的编码器—解码器系统：它要把输入序列转换成可读取的上下文表示，再按自回归方式生成目标序列。注意力只是其中负责信息交换的模块；位置表示、前馈网络、残差、归一化、训练目标和解码规则共同决定了整个模型的行为。

本章以原始论文的结构思想为依据，使用本卷 `OriginalTransformer` 作为一层、无 dropout 的 NumPy 前向参考。它保留编码器自注意力、解码器因果自注意力、解码器对编码器的交叉注意力以及 post-LayerNorm 顺序；它省略完整训练配方、可学习 norm gain/bias 等内容。我们会明确拆开“论文系统”“教学参考”和“可训练扩展”，不把几十行前向代码冒充完整论文复现。

先修 M01–M03。运行入口为 `python -m labs.models.run_lab transformer`。本章主要使用 CPU 小数组；如果你只想理解原始结构，不需要下载预训练权重，也不需要占用 H100。真正需要掌握的是每条数据路径和每个 mask 的含义。

## 第一课：先用一个序列到序列任务定义输入与输出

考虑一个人工任务：输入 $[a,b,c]$，输出 $[c,b,a]$。它不是语言翻译，却具有清楚的序列到序列关系，适合观察编码器和解码器如何交互。输入与输出长度可以相同，也可以不同；原始架构没有要求两者使用同样长度。真实翻译中，一个源语言词可能对应多个目标 token，长度差异更加常见。

源序列是模型已经完整得到的信息。生成目标序列时，模型可以读取整个源序列，但不能读取尚未生成的目标 token。因此编码器自注意力通常允许源位置彼此双向读取；解码器自注意力必须遵守目标侧因果限制；交叉注意力允许当前目标位置读取所有有效源位置。三种 attention 的可见性不同，不能共用一张随意构造的下三角矩阵。

编码器输出不是最终答案，也不是一个必须压成单个向量的摘要。它通常为每个源 token 保留一个上下文化向量，形成 $[T_s,D]$ 或带 batch 的 $[B,T_s,D]$ 数组。解码器在每个目标位置用 query 从这组表示中读取信息。这种逐位置来源比把整句压成一个固定向量更灵活。

解码器的输出也不是直接的文本，而是每个目标位置对目标词表的 logits。采样或贪心选择把最后位置的 logits 转成下一个 token ID，再由 tokenizer 解码为文字。编码器、解码器、输出头和生成循环是不同层次，不能把模型一次 forward 与整个生成任务混为一谈。

人工反转任务的验收可以非常明确：给定源序列和生成规则，输出应按指定顺序结束；输入中的有效 token 不应被 padding 污染；改变未来目标 token 不应影响较早位置的预测。如果只看最终字符串是否偶尔反转成功，而不检查这些结构性质，可能漏掉标签泄漏或错误 mask。

本章的小模型使用同一词表和一张 embedding 表方便讲解，但原始序列到序列任务可以使用不同的源、目标词表。是否共享 embedding 或输出权重是明确的设计选择，不是“Transformer”这个名称自动规定的全部细节。读取实际实现时，要检查参数对象是否真的共享。

## 第二课：目标序列为什么要右移一位

假设正确目标为 $[c,b,a,\mathrm{EOS}]$。训练时解码器输入应为 $[\mathrm{BOS},c,b,a]$，预测目标则为 $[c,b,a,\mathrm{EOS}]$。第零个位置看到开始标记，学习预测 c；第一个位置看到开始标记和 c，学习预测 b；依此类推。目标和输入错开一位，是下一 token 预测的基本对齐。

如果把同一条目标序列同时作为输入和标签，并且当前位置可以读取自己的 token，模型可能学会直接复制当前位置，而不是真正预测下一个。训练损失会显著下降，却无法在没有正确答案前缀的生成阶段工作。这是典型的数据泄漏，不是模型“学得特别快”。

反过来，若数据管线已经把标签右移，模型内部损失函数又再移动一次，就会错开两位。此时每个位置被要求预测隔一个 token 的目标，训练和生成语义不一致。不能只凭变量名叫 labels 就推断是否需要 shift；要沿数据构造和损失代码查到实际发生的位置。

```python
BOS, EOS, PAD = 1, 2, 0
target = [7, 6, 5, EOS]
decoder_input = [BOS] + target[:-1]
labels = target
assert decoder_input == [BOS, 7, 6, 5]
assert labels == [7, 6, 5, EOS]
for position, (seen, expected) in enumerate(zip(decoder_input, labels)):
    print(position, "current input", seen, "next-token target", expected)
```

teacher forcing 指训练时把真实目标前缀送入解码器，而不是把模型自己上一步生成的错误 token 送回。这样可以并行计算所有目标位置，也使每个位置有明确监督。它并不表示模型在某个位置可以看到未来目标；因果 mask 仍然必须存在。

训练和生成因此存在条件分布差异：训练常看到正确前缀，生成可能看到自己先前犯错的前缀。这个现象值得认识，但本章不把它简化成“teacher forcing 是错的”。它是一个高效且广泛使用的训练组织方式；如何改善生成鲁棒性、搜索或后训练属于进一步设计，需要具体目标与证据。

EOS 也要作为预测目标学习。没有 EOS 或明确长度规则，生成循环不知道何时停止；只有最大长度限制会在固定预算后截断，不等于模型学会任务结束。BOS 的用途则是给第一个预测提供一个合法输入位置。把二者混为同一语义，会使训练对齐和停止行为难以解释。

## 第三课：词嵌入与位置表示如何进入网络

token ID 查 embedding 得到 $[T,D]$ 表示。原始结构对 embedding 乘以 $\sqrt D$，再加入位置表示。这个缩放属于具体架构设计，不是所有模型都必须采用的通用规则。后来的某些 decoder 使用不同位置机制和归一化顺序，不能直接把这个步骤复制过去就声称兼容。

没有位置机制的非因果自注意力对输入排列具有置换等变性。语言和反转任务都需要区分顺序，所以原始 Transformer 使用正弦与余弦位置编码。对于位置 p、特征对编号 i，常见公式是 $PE_{p,2i}=\sin(p/10000^{2i/D})$，$PE_{p,2i+1}=\cos(p/10000^{2i/D})$。

每一对坐标使用不同频率。低编号对变化较快，高编号对变化较慢，因此同一位置用多个尺度的周期信号表示。这里的正弦编码不是把位置转换成一个单独整数特征，也不是为每个位置保存一张任意查表；它由确定公式生成。是否能在更长长度上实际泛化，是模型训练与结构共同决定的问题，不能仅凭公式可计算就保证能力。

取 D 等于四。第零对使用分母一，第二对使用分母一百。位置零的编码为 $[0,1,0,1]$；位置一为 $[\sin1,\cos1,\sin0.01,\cos0.01]$。角度单位是弧度，不是度数。若误用角度制，程序可能仍输出有限数，但位置定义已经改变。

```python
import numpy as np

def sinusoidal_positions(length, width):
    if width <= 0 or width % 2:
        raise ValueError("positive even width required")
    positions = np.arange(length)[:, None]
    frequencies = 10000 ** (np.arange(0, width, 2) / width)
    angles = positions / frequencies
    result = np.zeros((length, width))
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles)
    return result

pe = sinusoidal_positions(3, 4)
np.testing.assert_allclose(pe[0], [0,1,0,1])
np.testing.assert_allclose(pe[1], [np.sin(1),np.cos(1),np.sin(.01),np.cos(.01)])
```

位置编码与 embedding 逐元素相加，要求两者具有相同特征宽度。这个相加并不保留两个独立的显式字段；后续投影看到的是混合后的向量，需要通过训练利用内容与位置信息。若你想拼接而不是相加，输入维度和后续参数形状都要改变，那是另一种结构。

源序列与目标序列各自从自己的位置起点编码。源位置三与目标位置三不是同一个 token，也不必具有某种直接对齐关系。交叉注意力通过内容表示学习读取关系，不是简单按相同位置编号一一复制。真实任务中的对齐可以是多对一、一对多或跨较远位置。

## 第四课：编码器自注意力如何让每个源位置得到上下文

编码器层先对源表示做多头自注意力。Q、K、V 都来自当前编码器输入，但经过各自投影。每个源位置可以读取所有有效源位置，因此不使用目标侧的因果下三角限制。若有 padding，则仍要阻止读取无效填充位置。

为什么允许双向读取？源序列已经完整给定，编码器的任务是构造对整个输入有用的表示。一个词的含义可能依赖后面的词，双向上下文有助于编码。这里不涉及生成未来源 token，所以读取完整源序列不违反当前任务的因果要求。

“每个位置上下文化”意味着输出不再只是它自己的词表向量。通过 attention，它可以包含来自其他源位置的组合信息；通过多层堆叠，还能反复交换和变换这些信息。但输出行仍保留对应源位置的索引身份，便于解码器后来按 key/value 读取。不要把上下文化理解为所有位置最后都必须相同。

编码器自注意力的输出会经过输出投影，再进入残差和归一化。注意力函数本身通常不负责这些外层步骤。很多框架把它们封装在不同模块里，因此读代码时应沿调用顺序确认，而不是凭类名推断“Attention 类肯定包含 LayerNorm”。

如果你给编码器错误地加上因果 mask，它仍然是一个可运行模型，但不再是本章定义的原始双向编码器。对于反转任务，这种限制可能降低每个源位置可获得的信息；不过解码器仍可读取所有源位置，所以不能只凭任务能否学会判断结构是否一致。正确的比较应直接检查允许依赖矩阵。

一个有力测试是改变最后一个源 token，观察较早编码器位置的输出。非因果编码器通常允许这些输出变化；因果编码器则禁止早期位置受未来影响。这个测试的结论是“允许的依赖不同”，不是“某次随机参数下必须有很大变化”；参数偶然让某条路径贡献为零时，要换非退化输入或检查中间权重。

## 第五课：残差与 LayerNorm 的顺序为什么必须写清楚

原始 Transformer 常用 post-norm 形式：先计算子层输出 f(x)，与输入相加，再做 LayerNorm，即 $y=\mathrm{LN}(x+f(x))$。后来的许多模型采用 pre-norm，即 $y=x+f(\mathrm{LN}(x))$。两者包含相同组件，但连接顺序不同，数值和梯度行为不等价。

残差相加要求形状一致。attention 的内部 value 维度可以不同，但最终输出投影通常把它映射回模型宽度 D，才能与输入相加。FFN 中间层可以更宽，最后也要投影回 D。若为了通过 shape 检查随意截断或补零，你已经改变了架构，应重新解释和验证。

LayerNorm 对每个 token 的特征轴计算均值和方差。若向量为 $[1,2,3]$，均值为二，偏差为负一、零、一，方差为三分之二。忽略 epsilon 和可学习仿射参数，归一化结果约为 $[-1.224745,0,1.224745]$。它不是对所有 token 一起求均值，也不是对 batch 轴做统计。

公式为 $y_i=\gamma_i(x_i-\mu)/\sqrt{\sigma^2+\epsilon}+\beta_i$。gamma 和 beta 是可学习的逐特征缩放与平移；epsilon 用于避免零方差时除零。课程 NumPy 参考只保留归一化核心，省略 gamma/beta，这是明确的教学简化。完整可训练模块一般需要按具体架构检查这些参数。

```python
def layer_norm_reference(x, eps=1e-5, gamma=None, beta=None):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / np.sqrt(variance + eps)
    if gamma is not None:
        normalized = normalized * gamma
    if beta is not None:
        normalized = normalized + beta
    return normalized

normalized = layer_norm_reference(np.array([[1.,2.,3.]]))
np.testing.assert_allclose(normalized.mean(axis=-1), 0., atol=1e-12)
```

有 epsilon 时，输出方差并不严格等于一，而是 $\sigma^2/(\sigma^2+\epsilon)$，再考虑 gamma 后还会变化。测试如果无条件要求方差精确一，会对常量或很小方差输入做出错误判断。严谨理解公式，可以避免把合理边界误判为实现 bug。

残差不是简单的“防止信息丢失”口号。它建立了一条直接相加路径，使子层学习某种增量，并在反向中提供额外梯度路径。但 post-norm 会对相加后的整个向量再归一化，因此恒等路径也受到后续归一化影响。要讨论训练稳定性，必须结合具体顺序，不能把所有残差结构视为完全相同。

## 第六课：逐位置前馈网络到底做了什么

编码器中的另一主要子层是前馈网络，常见形式为 $\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$。第一张矩阵把模型宽度 D 投影到较大的隐藏宽度 F，经过 ReLU 非线性，再投影回 D。原始论文使用的具体宽度比例是架构配置的一部分，本课程小模型为了方便计算使用较小尺寸。

FFN 对每个 token 独立使用同一组参数，不直接在 token 轴上混合信息。跨位置关系由 attention 提供，逐位置的特征组合和非线性变换由 FFN 提供。两者不是一个负责“理解”、另一个负责“输出”这么简单的分工，但从计算轴上可以准确区分。

设输入 $[1,-1]$，第一层权重 $[[1,2,0],[0,1,-1]]$，偏置为零。预激活为 $[1,1,1]$，ReLU 后不变。若第二层权重为 $[[1,0],[0,1],[1,-1]]$，输出为 $[2,0]$。这是一条完整的二维到三维再到二维路径，最后可以与原输入相加。

如果预激活某项为负，ReLU 会将它置零；这改变了不同输入区域对应的线性变换，使两层组合不再能普遍合并为一张固定矩阵。若删除非线性，在没有其他变化的情况下，两个线性层可以合成一个线性层，增加中间宽度也不会带来同样的函数表达能力。

```python
x_ff = np.array([[1.,-1.]])
w1 = np.array([[1.,2.,0.],[0.,1.,-1.]])
w2 = np.array([[1.,0.],[0.,1.],[1.,-1.]])
pre = x_ff @ w1
hidden = np.maximum(pre, 0.)
ff_output = hidden @ w2
np.testing.assert_array_equal(pre, [[1,1,1]])
np.testing.assert_array_equal(ff_output, [[2,0]])
```

FFN 的参数量与序列长度无关，但计算和激活存储随处理位置数量增长。相同模型处理更长输入，不会突然多出新的 W1/W2 参数，却会为更多位置计算隐藏激活。把参数量、激活量和计算量分开，是后面估算训练与推理资源的基本习惯。

## 第七课：解码器中的三条子层路径

原始解码器层通常依次执行：带因果 mask 的目标自注意力、读取编码器输出的交叉注意力、逐位置 FFN，每个子层都带残差与归一化。顺序是模型定义的一部分。把交叉注意力放在自注意力之前，或把两者并行后相加，可能构成其他合理结构，但不是这里的同一个模型。

目标自注意力让每个位置整合已知目标前缀。例如预测第三个目标 token 时，可以利用开始标记和前两个真实目标 token。训练使用 teacher forcing 时，这些前缀来自数据；生成时来自模型已经输出的结果。无论来源如何，都不能越过当前逻辑位置读取未来目标。

交叉注意力的 Q 来自当前解码器表示，K/V 来自编码器最终输出。Q 的长度是目标长度，K/V 的长度是源长度，因此分数矩阵通常是矩形。源侧 padding mask 仍然有效，但通常不需要对源位置施加目标时间的下三角约束，因为整条源序列已经可见。

交叉注意力不是把编码器输出直接加到解码器输入上。两条序列长度可能不同，位置也未必对齐；它通过 query-dependent 权重读取来源内容，再产生与目标 query 数量相同的输出行。这种机制让同一个源位置可以被多个目标位置读取，一个目标位置也可以综合多个源位置。

之后的 FFN 在每个目标位置上处理已经融合目标前缀和源上下文的表示。输出头再把最终解码器表示投影到目标词表。若堆叠多层，每层都可重新执行自注意力、交叉注意力和 FFN，逐步细化表示；不能把第一层交叉注意力的权重当成模型唯一的源目标对齐。

一个常见误读是以为解码器只能看到编码器的最后一个位置。实际上它通常读取编码器的整组位置表示。若你把编码器输出先平均成单个向量，仍可以定义一个系统，但交叉注意力的来源数量和表达能力发生了改变。这里必须按实际数组形状确认。

## 第八课：三种 mask 的完整形状与合并规则

设 batch 为二，源最大长度为五，目标最大长度为四。源有效性数组可为 $[B,T_s]$，目标有效性数组为 $[B,T_t]$。编码器 self-attention 的允许矩阵通常对每条 query 使用同一源 key 有效性，广播成 $[B,1,T_s,T_s]$ 或库要求的等价布局。

解码器 self-attention 的允许条件至少包括目标因果关系和目标 key 有效性。可表示为 $[T_t,T_t]$ 的下三角矩阵与 $[B,1,1,T_t]$ 的有效 key mask 组合。若还要求无效 query 不产生有意义输出，可在模块外或 mask 构造中额外处理，但要避免引入全掩码行的未定义行为。

交叉注意力的允许矩阵依据源 key 有效性，形状对应 $[B,1,T_t,T_s]$。目标 query 的数量与源 key 的数量不同，不能把解码器 self mask 直接拿来复用。这个错误在源、目标长度恰好相等的小测试中容易隐藏，所以必须使用不等长样本验证。

损失 mask 通常针对目标标签有效位置，形状为 $[B,T_t]$。它决定哪些位置的交叉熵进入平均，以及分母是多少。注意力 mask 控制读取，损失 mask 控制监督，两者不能相互替代。一个填充位置不计损失，不代表前面的有效位置就不会通过错误 attention 读取它。

```python
source_valid = np.array([[True,True,True,False,False],
                         [True,True,True,True,True]])
target_valid = np.array([[True,True,True,False],
                         [True,True,True,True]])
causal_target = np.arange(4)[None,:] <= np.arange(4)[:,None]
encoder_allowed = np.broadcast_to(source_valid[:,None,None,:], (2,1,5,5))
decoder_allowed = causal_target[None,None,:,:] & target_valid[:,None,None,:]
cross_allowed = np.broadcast_to(source_valid[:,None,None,:], (2,1,4,5))
assert encoder_allowed.shape == (2,1,5,5)
assert decoder_allowed.shape == (2,1,4,4)
assert cross_allowed.shape == (2,1,4,5)
```

这些数组使用 True 表示允许，是课程自定义约定。不同框架的 mask 参数可能有不同含义和形状要求。迁移时应先用一个“唯一允许 key”的例子确认接口语义，再上大模型。布尔数组可以成功传入函数，不代表它表达了你希望的可见性。

## 第九课：逐行阅读并重建 NumPy 原始结构

下面的实现与课程参考相同地采用一层 encoder 和一层 decoder，省略 batch、padding、dropout 和 norm 可学习仿射参数，专注最核心数据流。输入必须是非空合法 ID 序列。它是完整前向结构，但权重为随机初始化，不能生成有意义的翻译。

```python
from labs.models.core import attention, split_heads, merge_heads, causal_mask

class OneLayerSeq2Seq:
    def __init__(self, vocab=16, width=8, heads=2, seed=4):
        if width % heads or width % 2:
            raise ValueError("invalid width or heads")
        self.width, self.heads, self.vocab = width, heads, vocab
        rng = np.random.default_rng(seed)
        def weight(a, b):
            return rng.normal(0, 1/np.sqrt(a), (a,b))
        self.embedding = weight(vocab, width)
        self.output = weight(width, vocab)
        self.projections = [
            tuple(weight(width,width) for _ in range(4))
            for _ in range(3)
        ]
        self.feedforwards = [
            (weight(width,2*width),weight(2*width,width))
            for _ in range(2)
        ]

    def mha(self, query_input, memory_input, weights, allowed=None):
        wq, wk, wv, wo = weights
        q = split_heads(query_input @ wq, self.heads)
        k = split_heads(memory_input @ wk, self.heads)
        v = split_heads(memory_input @ wv, self.heads)
        result, _ = attention(q,k,v,allowed)
        return merge_heads(result) @ wo

    def encode(self, source):
        x = self.embedding[source] * np.sqrt(self.width)
        x = x + sinusoidal_positions(len(source),self.width)
        x = layer_norm_reference(x+self.mha(x,x,self.projections[0]))
        up, down = self.feedforwards[0]
        x = layer_norm_reference(x+np.maximum(x@up,0)@down)
        return x

    def decode(self, target_input, memory):
        x = self.embedding[target_input] * np.sqrt(self.width)
        x = x + sinusoidal_positions(len(target_input),self.width)
        self_delta = self.mha(x,x,self.projections[1],causal_mask(len(x)))
        x = layer_norm_reference(x+self_delta)
        cross_delta = self.mha(x,memory,self.projections[2])
        x = layer_norm_reference(x+cross_delta)
        up, down = self.feedforwards[1]
        x = layer_norm_reference(x+np.maximum(x@up,0)@down)
        return x @ self.output

    def forward(self, source, target_input):
        source = np.asarray(source)
        target_input = np.asarray(target_input)
        for ids in (source,target_input):
            if ids.ndim != 1 or len(ids)==0 or ids.dtype.kind not in "iu":
                raise ValueError("nonempty integer sequence required")
            if np.any(ids<0) or np.any(ids>=self.vocab):
                raise ValueError("token outside vocabulary")
        return self.decode(target_input,self.encode(source))

model = OneLayerSeq2Seq()
logits = model.forward([3,4,5],[BOS,7,6])
assert logits.shape == (3,16)
assert np.isfinite(logits).all()
```

构造函数中的三组 projections 分别对应 encoder self、decoder self 和 decoder cross。每组都含 Q、K、V、O 四张矩阵。虽然形状相同，它们是不同参数对象，承担不同任务。把列表乘三可能意外共享同一组对象，所以这里使用循环分别初始化，避免语言层别名改变模型定义。

`mha` 接受两个输入：query_input 与 memory_input。自注意力调用时二者是同一个表示，交叉注意力时二者来自不同序列。这个接口直接把“Q 来自哪里、KV 来自哪里”写进函数参数，比在函数内部依靠某个隐藏模式猜测更容易读懂和测试。

encode 先查表加位置，再执行 attention 的残差归一化，再执行 FFN 的残差归一化。decode 则多了一段 cross-attention。每一行的 x 都代表完成某个子层后的新状态，不能把某个早先归一化结果缓存后同时喂给所有子层，除非你明确改变结构。

最后输出矩阵把每个目标位置的 D 维表示映射为 vocab 个分数。此时还没有执行 softmax，因为交叉熵可以直接接收 logits，贪心生成也只需比较分数大小。把 softmax 提前放进模型再交给一个内部还会做 log-softmax 的损失函数，可能造成目标定义错误或数值问题。

## 第十课：为什么随机模型的结构测试仍然有价值

一个随机初始化模型不会翻译，不妨碍我们验证形状、因果性和依赖边界。结构测试回答“实现是否遵守定义”，能力评测回答“训练后的参数是否解决任务”。两者需要不同证据，不能因为随机输出没有意义就跳过结构检查，也不能因为结构测试通过就宣称模型具有语言能力。

最重要的测试之一是目标前缀不变性。给定相同源序列，两条目标输入只在最后一个位置不同，较早位置的 logits 应相同。因为目标 self-attention 有因果 mask，交叉注意力只读取相同源 memory，FFN 和 LayerNorm 又是逐位置操作，因此未来目标不应影响过去输出。

```python
source_ids = [3,4,5]
first = model.forward(source_ids,[BOS,7,6,5])
second = model.forward(source_ids,[BOS,7,6,9])
np.testing.assert_allclose(first[:3],second[:3],atol=1e-12)
```

另一个测试是源信息确实可以影响目标输出。改变源 token，目标 logits 通常应发生变化；但只测试一个随机配置可能出现极小影响，因此更严谨的方式是检查交叉注意力的 memory 输入和非退化数值例子。不要把“某次输出没明显变”直接解释为编码器没有作用，也不要只凭变化就证明所有 mask 都正确。

还可以分离 encode 和 decode。对相同源序列只计算一次 memory，再对不同目标前缀调用 decode；结果应与每次完整 forward 一致。这个等价性为后面缓存编码器输出提供基础。它同时验证模块之间没有依赖未记录的临时状态。

与仓库 `OriginalTransformer` 比较时，要保证随机种子、权重初始化顺序、位置编码、epsilon 和所有参数相同。两个模型都叫“一层 Transformer”但参数随机生成顺序不同，输出自然不同；这不构成算法错误。最可靠的对照是显式共享同一组参数，再比较中间张量。

## 第十一课：生成循环怎样与训练 forward 对接

生成时，源序列已经完整给定，先编码一次。目标前缀从 BOS 开始。每一轮将当前前缀送入 decoder，取最后有效位置的 logits，选择一个新 token，追加到前缀；若 token 为 EOS 则停止，否则继续，直到达到明确的最大生成步数。

```python
def greedy_decode(model, source, bos_id=1, eos_id=2, max_new_tokens=8):
    if max_new_tokens < 1:
        raise ValueError("positive generation budget required")
    source = np.asarray(source,dtype=np.int64)
    memory = model.encode(source)
    prefix = [bos_id]
    generated = []
    for _ in range(max_new_tokens):
        logits = model.decode(np.array(prefix,dtype=np.int64),memory)
        token = int(np.argmax(logits[-1]))
        generated.append(token)
        prefix.append(token)
        if token == eos_id:
            break
    return generated

generated = greedy_decode(model,[3,4,5],max_new_tokens=5)
assert 1 <= len(generated) <= 5
```

这段循环每步重算整个目标前缀，速度不高，但语义清楚，适合作为 cache 版本的正确性参考。编码器 memory 已经复用，因为源输入不变；目标 self-attention 的历史 K/V 还没有缓存。M06 会解释如何只计算新 token 的表示，并验证与这个慢版本一致。

最大步数是一项资源预算，不是模型学会结束的证据。如果随机模型没有生成 EOS，循环仍会在预算耗尽后返回。报告应区分“模型输出结束标记”和“被长度上限截断”。真实服务里还可能有取消、超时和停止字符串，不能把所有结束都写成正常完成。

贪心解码每步选择当前最高分 token，不保证得到全序列概率最高的目标。beam search 会保留多个候选前缀，但也需要处理长度偏好、结束候选和预算。采样则从分布中随机选择，用于产生多样输出。它们属于生成策略，模型 forward 的 logits 定义可以保持不变。

训练时使用真实前缀，生成时使用自身输出，所以一个早期错误会改变后续条件。调试时可以分别比较 teacher-forced loss 和自由生成结果，判断是模型仍没学会局部分布，还是错误传播、搜索或停止规则导致最终序列失败。只看一段生成文本往往不足以定位原因。

## 第十二课：完整训练需要哪些额外部分

要把前向结构变成可训练系统，需要参数的梯度、损失、优化器、数据批次和训练循环。NumPy 原始结构没有实现所有反向，因此不能对它直接调用一个不存在的 backward。你可以手写每个子层反向，也可以在 PyTorch 中保留同样的显式计算结构，让自动微分处理梯度汇总。

“用 PyTorch”并不等于“只会调用现成 Transformer”。你仍可以亲自写 QKV 投影、拆头、mask、softmax、残差、LayerNorm 和 FFN，再让自动微分追踪这些基础运算。学习目标是理解并实现结构，不是强迫自己在所有阶段重写高效张量库。M02 的手写反向则帮助你验证框架背后的规则。

一个可训练的原始 encoder-decoder 模块应显式包含三类 attention，各自独立参数；源和目标 embedding；位置编码；按具体设计存在的 norm gain/bias；输出词表头；以及必要的 dropout。训练样本构造要明确 BOS/EOS/PAD，输入与标签右移，损失忽略填充位置，并按有效 token 数归一化。

训练循环通常依次做：取一批源和目标；构造三种 attention mask 和标签 mask；前向得到 logits；计算交叉熵；清理或按约定累计梯度；反向；处理梯度裁剪；更新参数；记录损失和有效 token 数。评估时关闭训练随机行为，使用固定数据和明确解码策略。每一步都应能独立观察，而不是藏在一个无法检查的训练函数里。

原论文还包含具体优化设置，例如学习率调度、label smoothing 和 dropout。它们影响训练结果，但不是仅仅把网络连接图写对就自动获得。若你声称复现原论文，需要同时对齐数据、tokenizer、模型规模、初始化、训练步数和评价协议；本章只建立可检验的结构与小任务训练路线。

下一阶段的合理实验不是立刻用 H100 训练大型翻译模型，而是在几十个符号的小词表上学习复制或反转，先证明损失能下降、生成能结束、未见长度和 token 组合下的行为可解释。只有当小任务的数据对齐、mask 和训练循环都正确，扩大规模才有意义。

### 12.1 三个 token 的完整训练时间表

把反转任务写成一张逐位置的时间表，能消除“训练时一次输入整个答案，所以模型一定作弊”的误解。源序列是三、四、五；正确目标是五、四、三、结束。解码器输入是开始、五、四、三。虽然四个输入位置一起放进一个数组，位置零的 self-attention 只允许读取开始标记，不能读取后面的五、四、三。

位置一可以读取开始和五，监督目标是四；位置二可以读取开始、五、四，监督目标是三；位置三可以读取开始、五、四、三，监督目标是结束。每个位置的 logits 都来自不同的允许前缀，但它们共享同一组参数。并行计算只是把四个已知监督条件一起处理，不改变每个条件的信息边界。

编码器 memory 对四个位置都相同，因为源输入没有变化。但 cross-attention 的 query 不同，所以读取权重可以不同。你可以将它理解为同一份来源资料被四个不同的问题读取，而不是编码器为每个目标位置预先准备一个固定答案。注意力矩阵的四行分别对应四个读取分布。

训练损失是四个目标位置负对数似然的平均。如果某个位置的正确类别概率分别为二分之一、四分之一、八分之一、二分之一，四个损失分别为 log 二、log 四、log 八、log 二，平均为 $7\log2/4$。这项手算让你明确：序列损失不是只看最后一个位置，也不是把四个概率简单相加。

生成时没有真实目标前缀可用。第一次只输入开始，选出一个 token；第二次输入开始和刚选出的 token；依此类推。若第一步选错，后续条件就与 teacher forcing 不同。为了定位问题，你可以同时记录“给真实前缀时能否预测下一项”与“自由生成整条序列是否正确”，但不能把两种指标混成同一个准确率。

### 12.2 给原始结构加上可训练参数，保留显式 attention

下面提供一个完整的小型 PyTorch 编码器—解码器训练示例。它使用基础 Linear、Embedding、LayerNorm 和张量运算，自己实现多头 attention，没有调用 `nn.Transformer` 或现成 MultiheadAttention。为了让代码可以直接阅读，示例只训练固定长度的反转任务，不包含 padding 和 dropout；这两项在后面的升级任务中独立加入。

本次教材构建环境没有安装 PyTorch，因此这段扩展的执行状态是“提供完整代码、未在本环境训练验证”。它不会自动下载模型或数据。使用时先按课程环境说明安装与你的 CPU 或 GPU 环境匹配的 PyTorch；默认只使用 CPU，固定小词表、小 batch 和有限训练步数。不要把打印出的随机初始化输出称为训练成果。

```python
import importlib.util

def run_seq2seq_torch_demo(steps=80):
    if importlib.util.find_spec("torch") is None:
        print("SKIP: optional PyTorch not installed; seq2seq training not executed.")
        return None

    import torch
    from torch import nn
    torch.manual_seed(29)
    torch.set_num_threads(2)
    width, heads, vocab, maximum = 16, 2, 12, 8
    head_dim = width // heads

    class MHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(width,width,bias=False)
            self.k = nn.Linear(width,width,bias=False)
            self.v = nn.Linear(width,width,bias=False)
            self.o = nn.Linear(width,width,bias=False)

        def forward(self, query, memory, allowed=None):
            batch, qlen, _ = query.shape
            klen = memory.shape[1]
            q = self.q(query).view(batch,qlen,heads,head_dim).transpose(1,2)
            k = self.k(memory).view(batch,klen,heads,head_dim).transpose(1,2)
            v = self.v(memory).view(batch,klen,heads,head_dim).transpose(1,2)
            score = q @ k.transpose(-1,-2) / head_dim**0.5
            if allowed is not None:
                score = score.masked_fill(~allowed,float("-inf"))
            probability = torch.softmax(score,dim=-1)
            out = probability @ v
            out = out.transpose(1,2).contiguous().view(batch,qlen,width)
            return self.o(out)

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.source_embedding = nn.Embedding(vocab,width)
            self.target_embedding = nn.Embedding(vocab,width)
            self.encoder_attention = MHA()
            self.decoder_attention = MHA()
            self.cross_attention = MHA()
            self.encoder_ff = nn.Sequential(
                nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width))
            self.decoder_ff = nn.Sequential(
                nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width))
            self.en1 = nn.LayerNorm(width)
            self.en2 = nn.LayerNorm(width)
            self.de1 = nn.LayerNorm(width)
            self.de2 = nn.LayerNorm(width)
            self.de3 = nn.LayerNorm(width)
            self.output = nn.Linear(width,vocab)
            positions = torch.arange(maximum)[:,None]
            frequency = 10000 ** (torch.arange(0,width,2)/width)
            pe = torch.zeros(maximum,width)
            pe[:,0::2] = torch.sin(positions/frequency)
            pe[:,1::2] = torch.cos(positions/frequency)
            self.register_buffer("position",pe)

        def encode(self, source):
            x = self.source_embedding(source)*width**0.5
            x = x+self.position[:source.shape[1]]
            x = self.en1(x+self.encoder_attention(x,x))
            return self.en2(x+self.encoder_ff(x))

        def decode(self, target_input, memory):
            length = target_input.shape[1]
            x = self.target_embedding(target_input)*width**0.5
            x = x+self.position[:length]
            allowed = torch.tril(torch.ones(length,length,dtype=torch.bool))
            x = self.de1(x+self.decoder_attention(x,x,allowed))
            x = self.de2(x+self.cross_attention(x,memory))
            x = self.de3(x+self.decoder_ff(x))
            return self.output(x)

        def forward(self, source, target_input):
            return self.decode(target_input,self.encode(source))

    network = Seq2Seq()
    optimizer = torch.optim.Adam(network.parameters(),lr=0.01)
    source = torch.randint(3,vocab,(64,3))
    target = torch.cat([source.flip(1),torch.full((64,1),EOS)],dim=1)
    decoder_input = torch.cat([torch.full((64,1),BOS),target[:,:-1]],dim=1)
    trace = []
    for step in range(steps):
        network.train()
        logits = network(source,decoder_input)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1,vocab),
                                                 target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
        optimizer.step()
        trace.append(float(loss.detach()))
    network.eval()
    with torch.no_grad():
        memory = network.encode(source[:2])
        prefix = torch.full((2,1),BOS)
        for _ in range(4):
            logits = network.decode(prefix,memory)
            selected = logits[:,-1].argmax(dim=-1,keepdim=True)
            prefix = torch.cat([prefix,selected],dim=1)
    result = {
        "initial_loss":trace[0], "final_loss":trace[-1],
        "source":source[:2].tolist(), "target":target[:2].tolist(),
        "generated":prefix[:,1:].tolist(),
        "scope":"fixed tiny training batch; not a generalization benchmark",
    }
    print(result)
    return network, result

# 显式调用才训练；没有 PyTorch 时只报告未执行。
torch_demo = run_seq2seq_torch_demo()
```

MHA 的 query 和 memory 参数让三种 attention 共享同一计算代码，却拥有不同模块实例和参数。编码器调用时二者都是源表示；解码器 self 调用时二者都是目标表示并传因果 mask；cross 调用时 query 是目标状态，memory 是编码器输出。这个接口结构把数据来源写得很明确，便于逐层检查。

view 与 transpose 的顺序沿用 M03 的布局推导。最后的 contiguous 是为了让转置后的数据按后续 view 需要的布局组织；它属于具体框架的存储接口，不改变数学上的合头含义。理解时先确认元素归属，再讨论这一步是否发生复制以及能否融合，不能把内存优化放在正确性之前。

LayerNorm 在这里使用框架模块，因此包含可学习的仿射参数，与前面的无仿射 NumPy 教学核心不同。源和目标 embedding 也分别创建，不共享参数；输出头还有偏置。它保持原始 encoder-decoder 的关键连接顺序，但不声称逐项复现原论文的全部参数共享或训练配方。你应能自己列出这些配置选择。

训练数据只有一个固定小 batch，目的是先看模型能否拟合一个可控任务。这里没有独立验证集，因此不能把最后损失当作泛化结果。下一步必须使用新的源序列评估，区分记住训练组合与学会可迁移的反转规律。即使所有训练样本都生成正确，也不能直接宣称已经理解语言翻译。

示例每次生成固定四项，因为任务目标长度已知为三个符号加 EOS。更通用的生成器应逐样本记录 finished，已经输出 EOS 的样本停止追加有效 token，并保留明确最大长度。batch 中各样本结束时间不同，不能因为一条完成就把全部请求停止，也不能让已完成样本继续无条件产生新内容。

### 12.3 把固定长度训练升级为可变长度与 padding

可变长度任务首先改变数据组织。为每条源序列保存真实长度，为每条目标保存包括 EOS 的真实长度，然后分别填充到本 batch 的最大源长度与最大目标长度。源和目标最大长度不一定相同，不能只计算一个统一长度后在所有地方复用。

编码器注意力需要忽略源 padding key；解码器 self 需要因果限制与目标 padding key；cross 需要源 padding key；损失需要忽略标签 padding。四项职责应分别写清。你可以把它们封装成一个 batch 对象，但不能因为放进同一个对象，就把四个 mask 合成一个没有明确语义的布尔数组。

若用框架交叉熵的 ignore_index 忽略 PAD，要确认 reduction 的平均分母只包含未忽略标签。若手动计算逐位置损失，则应先乘有效性，再用有效位置总数归一化，并处理全部无效的非法 batch。直接对填充后的 B 乘 T 个槽位求平均，会让短序列 batch 的梯度规模受填充比例影响。

padding embedding 是否固定为零是另一个选择。即使它是零，位置编码相加后填充位置也可能不再为零；即使整个填充表示为零，未正确屏蔽的 attention 仍可能给它正权重。因此“把 pad 向量设成零”不能替代 attention mask。测试要改变填充区域数值，确认有效位置输出不受影响。

建议先只在源侧加入 padding，保持目标长度固定，验证 cross 和 encoder 的屏蔽；再在目标侧加入 padding，检查 self 和损失；最后测试同一有效样本放进不同最大长度的 batch 时，有效输出保持一致。分阶段引入边界，比一次改完后只观察总损失更容易定位问题。

这里还有一个 batch 组合不变性测试：把某条样本单独运行，再和一条更长样本一起运行，比较它有效位置的 logits。在关闭 dropout、相同参数和相同位置约定下，两者应在浮点容差内一致。如果差异很大，优先检查 padding、batch 维度混用或位置编号，而不是把它解释成模型“受到其他句子的语境影响”。

### 12.4 label smoothing、dropout 与损失报告的细节

one-hot 标签把全部目标概率质量放在一个类别上。label smoothing 则把一小部分质量分配给其他类别或按某种基准分布混合，使目标分布不再是完全尖锐的 one-hot。不同实现对“分配给所有类别”还是“只分配给非目标类别”的约定可能不同，所以必须写出公式，不能只记录一个 epsilon 数字。

例如本节定义 $q=(1-\epsilon)\,\mathrm{onehot}(y)+\epsilon/V$。词表三类、目标编号一、epsilon 为零点一时，目标分布为 $[1/30,14/15,1/30]$。平滑交叉熵为 $-\sum_i q_i\log p_i$，logits 梯度为 $p-q$。它不再只是目标位置减一，代码必须与所选定义一致。

label smoothing 会改变最优分布和损失的最低可能值，因此不同平滑设置下的训练 loss 不能不加说明直接比较。评估时若希望报告普通负对数似然，应使用明确的不平滑目标重新计算。不要把带额外正则项的训练目标当成同一种标准 perplexity 输入。

dropout 则随机屏蔽某些激活或权重贡献。它使训练前向带随机性，有助于某些条件下的正则化，但也让逐步数值对照更复杂。初次实现时把 dropout 设为零，先验证确定性数据流；结构正确后再加入，并检查 train/eval 行为。这样可以避免把随机变化误当成 mask 或布局错误。

梯度检查也应关闭 dropout，或确保两次差分前向使用同一随机掩码。否则参数加 h 和减 h 的损失差同时包含随机变化，无法代表导数。反向传播本身可以正确处理某次固定 dropout 采样，但你的数值验证必须针对同一条实际计算路径。

### 12.5 LayerNorm 的反向为何与 token 轴无关

设某个 token 的特征向量为 x，均值为 μ，标准差记为 $s=\sqrt{\mathrm{var}(x)+\epsilon}$，归一化向量为 $z=(x-\mu)/s$。暂不考虑仿射参数，收到上游梯度 g 后，输入梯度可以写为 $[g-\mathrm{mean}(g)-z\,\mathrm{mean}(gz)]/s$，其中均值沿当前 token 的特征轴计算。

这个公式包含三部分：直接缩放的梯度、减去由均值依赖产生的共同项、减去由方差依赖产生的项。LayerNorm 不是把 x 除以一个与 x 无关的常数，所以反向不能仅写成 g 除以 s。均值和方差本身也依赖每个特征，必须通过链式法则计入。

若有 gamma 和 beta，先把上游梯度乘 gamma，再代入归一化核心反向；gamma 的梯度来自 g 乘 z，在所有共享位置上求和；beta 的梯度来自 g 的求和。这里参数跨 batch 与 token 共享，但归一化统计是每个 token 独立计算。两种求和轴承担不同职责，不能混为一谈。

这也说明为什么 batch 中增加另一个样本，不应改变当前样本的 LayerNorm 前向统计。它与依赖 batch 统计的其他归一化方案不同。读模型时看到 Norm 这个词，不能自动假定统计范围和训练评估行为相同，必须查看实现所用轴和参数。

你可以用一个三维、非恒定输入和一个非对称上游梯度做有限差分；再测试常量输入时输出有限，但不要要求零 epsilon 下仍然合法。对于近常量输入，epsilon 会明显影响尺度与梯度，这是数值稳定机制的一部分，不应被解释为随机误差。

### 12.6 参数数量、激活与训练状态的完整账本

以模型宽度 D、FFN 隐藏宽度 F 为例，一组无偏置标准多头 attention 的 Q、K、V、O 投影共约四 D 平方个参数，头数改变但总投影宽度保持 D 时，这个主要参数项不直接随头数线性增长。一个 FFN 的两张权重约为二 D F；偏置和归一化参数再单独计入。

一层编码器有一组 attention 和一个 FFN；一层解码器有两组 attention 和一个 FFN，所以相同宽度下解码器层通常多出交叉注意力的参数。embedding 与输出头还与词表大小相关。如果源和目标词表不同，分别使用 $V_sD$ 和 $V_tD$ 计数；权重共享会改变实际独立参数总数。

参数量不能完整解释训练内存。训练还可能保存参数梯度、优化器一阶和二阶状态、前向激活以及临时计算空间。不同 dtype、混合精度方案和分布式策略会改变每项存储。初学时应先列清对象，再代入每元素字节数，而不是背一个固定的“每参数多少字节”常数用于所有环境。

交叉注意力的分数规模为目标长度乘源长度，与目标 self 的目标长度平方和源 self 的源长度平方不同。若源很长、目标较短，三部分相对开销会不同。把整个 encoder-decoder 的 attention 成本只写成一个不注明长度的 T 平方，会掩盖实际任务形状。

生成时编码器只需计算一次，交叉注意力 K/V 也可按层预先投影并复用；目标 self KV 随生成长度增长。训练通常一次处理全部目标位置，并需要保存更多反向信息。相同模型的训练、prefill 和 decode 是不同工作负载，不能用其中一种测量替代另外两种。

### 12.7 原始结构与现代 decoder 的比较应该怎样做

现代 decoder-only 模型通常不再保留独立的双向编码器和每层 cross-attention，而是把提示与待生成内容放在同一条因果序列中。这个变化减少了一类路径，但不表示“编码器没有用”或“decoder 永远优于 encoder-decoder”。不同任务、数据组织和部署约束会影响设计选择。

比较结构时，先固定任务的信息条件。encoder-decoder 可以在源侧双向读取完整输入，同时目标侧保持因果；普通 decoder-only 对整条序列使用因果限制时，早期提示位置不能读取后面的提示内容。最终生成位置仍能读取整个提示，但中间表示形成方式不同。仅凭两者都能回答问题，不能说内部计算等价。

归一化顺序、激活函数、位置编码和 KV 头共享也分别是独立设计轴。一个模型从 post-LayerNorm 改成 pre-RMSNorm，并从 ReLU FFN 改成 SwiGLU，再加入 RoPE，不能只用“删掉 encoder”解释全部差异。下一章会逐个拆开这些变化，保留本章作为结构基准。

做架构实验时，每次只改变一个轴最容易解释结果。例如保持数据、参数规模和训练预算近似一致，只比较 norm 顺序；再单独比较位置机制。若同时改变词表、宽度、训练数据和解码策略，最终指标差异不能归因于某一个组件。学习模型演进需要这种控制变量意识，而不是按发布时间列一串名称。


### 12.8 贪心与束搜索：同一模型为何生成不同序列

模型给出的是条件分布：在当前源序列和目标前缀下，下一个 token 的概率是多少。贪心每次取最大的局部概率，但整条序列概率是各步条件概率的乘积，或对数概率的和。局部最大并不保证全局最大，这是搜索问题与模型打分问题的区别。

用一个两步树手算。第一步 A 的概率零点六，B 的概率零点四，贪心选择 A。若 A 之后最可能的完成项概率只有零点二，而 B 之后某个完成项概率为零点九，那么两条候选的联合概率分别为零点一二和零点三六。虽然 B 第一步较低，它的完整序列概率反而更高。这个例子不涉及训练失败，只说明局部选择不能穷尽未来可能。

beam search 保留若干高分前缀，并在下一步共同扩展，再选出有限数量候选。beam 宽度一通常退化为贪心；宽度更大增加计算和存储，但仍不保证找到无限搜索空间中的全局最优。最大长度、结束处理和长度修正也会改变结果，不能只记录一个 beam 数量就认为解码协议完整。

下面给出一个小型、无长度修正的参考。每个候选保存 token 前缀、累计对数概率和是否结束。已结束候选不再调用模型扩展，而是保留原分数。函数只适合教学小词表与小 beam，真实服务会用更复杂的批处理与缓存复用，不能用这段 Python 循环推断生产性能。

```python
def beam_decode_reference(model, source, width=2, max_new_tokens=5,
                          bos_id=1, eos_id=2):
    if width < 1 or max_new_tokens < 1:
        raise ValueError("positive search budgets required")
    memory = model.encode(np.asarray(source,dtype=np.int64))
    beams = [([bos_id],0.0,False)]
    for _ in range(max_new_tokens):
        candidates = []
        for prefix, score, finished in beams:
            if finished:
                candidates.append((prefix,score,True))
                continue
            logits = model.decode(np.asarray(prefix,dtype=np.int64),memory)[-1]
            shifted = logits-logits.max()
            logprob = shifted-np.log(np.exp(shifted).sum())
            for token in range(len(logprob)):
                candidates.append((prefix+[token],score+float(logprob[token]),
                                   token==eos_id))
        candidates.sort(key=lambda item:(-item[1],item[0]))
        beams = candidates[:width]
        if all(finished for _,_,finished in beams):
            break
    return [{"tokens":prefix[1:],"log_probability":score,
             "ended_by_eos":finished} for prefix,score,finished in beams]

beams = beam_decode_reference(model,[3,4,5],width=2,max_new_tokens=3)
assert len(beams) == 2
assert all(len(item["tokens"]) <= 3 for item in beams)
```

为什么累加对数而不是直接乘概率？长序列中很多小于一的数相乘会很快变得极小，容易下溢；取对数后乘积变成求和，数值更容易表示。对数是单调函数，因此在相同定义下不改变候选概率排序。这里的稳定 log-softmax 也避免先把很小概率舍入成零再取对数。

长度偏好需要单独说明。每增加一个 token，通常会加上一个非正的对数概率，因此原始累计分数天然偏向某些较短序列。长度归一化或惩罚会改变排序目标，不是一个纯粹不影响语义的性能优化。若比较模型评测，必须固定解码规则；否则不同结果可能来自搜索而非参数能力。

还有一个状态边界：如果 beam 中只有一条候选生成 EOS，不能立即停止所有候选，除非你的停止准则已经证明其他候选不可能更好，或明确采用那种近似。参考函数等全部保留候选结束才提前停止，否则继续到预算。结束候选保持原分数，避免重复添加 EOS 让它被无意义地重新计分。

### 12.9 从“能过拟合”走到“有泛化证据”

训练一个小 batch 到低损失，是调试结构和梯度的必要阶段之一，却不是最终学习成果。一个宽度足够的小网络可能记住有限样本，并没有学会你希望的序列规律。为了区分，需要在未参与更新的数据上测试，并明确未见的是哪些方面：新组合、新长度、新 token 还是新的任务分布。

反转任务可以分层设计测试。第一层只测试训练中出现过的序列，检查实现是否有拟合能力。第二层保持词表和长度相同，但使用从未出现过的排列组合。第三层改变长度，仍保持在位置编码和模型接口支持范围内。第四层加入重复 token、极短序列和不同 padding 比例，检查边界。每一层回答不同问题，不能用一个总准确率替代所有观察。

词表外 token 是表示覆盖问题，不宜和结构泛化混为一谈。如果测试 token 的 ID 超出 embedding 范围，模型根本没有合法输入；如果使用一个从未训练过但已有参数行的 ID，输入合法，但该行可能仍接近随机状态。报告应区分这两种情况，而不是都写成“模型不理解新词”。

序列准确率与逐 token 准确率也不同。长度十的输出只错一个 token，逐 token 准确率九成，整序列准确率却为零。若任务要求完整反转，整序列正确是重要指标；若研究局部预测，逐 token 分析也有价值。EOS 提前、EOS 缺失、符号错误和顺序错误最好分别统计，帮助定位模型或生成流程问题。

训练与验证数据划分必须在产生重叠样本或重复序列之前考虑。对人工随机序列，可用集合检查训练和验证完整序列是否重复；对自然文本滑窗，应先划分原始文档或连续区间，再建立窗口，避免高度重叠的片段横跨边界。这里只背一句“不要数据泄漏”不够，必须检查实际样本身份。

保存实验时至少记录随机种子、数据生成规则、模型尺寸、训练步数、优化器、学习率、有效 token 统计和解码策略。这样当你修改一个组件后，才能判断变化来自结构还是实验条件。一个没有这些信息的漂亮输出截图很难作为可复现证据。

### 12.10 交叉注意力图能告诉你什么，不能告诉你什么

在反转任务中，你可能期待目标第一个位置主要读取源最后一个位置。训练后某些头也许会出现类似图案，这能帮助理解模型如何组织信息。但不能把这个期待硬编码为所有头必须满足的验收，因为编码器表示已经整合了上下文，一个源位置可能携带其他位置的信息，多个头和层也可以分工。

假设编码器每个位置都通过双向 attention 汇总了整句信息，解码器即使读取一个固定位置，也可能得到足够完成某些小任务的内容。此时交叉注意力图看起来没有简单反向对角线，不必然说明模型错误。相反，一张看起来非常整齐的对齐图，也不能证明最终预测正确，还要检查 value、输出投影和后续层。

更可靠的分析是做干预：改变一个源 token，观察哪些目标输出和中间表示改变；屏蔽某个源位置，测量任务损失变化；比较多个头是否有冗余或互补作用。干预也要谨慎，因为随意把激活置零可能制造训练时从未见过的状态。它提供的是特定干预条件下的证据，不是无条件的心理解释。

对初学者而言，先把注意力图当作计算中间量，不要赋予它过多拟人含义。你可以准确说“这个 query 对这些 value 使用了这些系数”，却不能仅凭系数说“模型在思考某个概念”。这种表达习惯会帮助你以后阅读可解释性研究时区分观察、假设和因果证据。

### 12.11 一个真正可审查的原始 Transformer 结业提交

最终提交不应只有一份能运行的脚本。它应包含带形状的数据流图、三类 attention 的来源表、训练样本右移示例、mask 真值表、参数数量计算、一个完整手算例子，以及至少三种故障的定位记录。代码与文字应引用同一组配置，不能文档写两个头而实际默认四个头。

测试应覆盖正常前向、源目标不等长、未来目标非干扰、padding 非干扰、编码器复用等价、保存重载等价和小 batch 训练。未安装 PyTorch时，可以完成 NumPy 结构测试，但必须把训练测试记为未执行，而不是用导入成功或语法通过替代。使用真实 GPU 后，再补设备与工具链信息，不改写过去的执行事实。

这份提交的价值在于你能解释每一处行为：为什么这里有 cross-attention、为什么这里先加残差再归一化、为什么标签比输入向前一位、为什么结束位置也要监督。能够回答这些问题，再进入现代 decoder 的变化，你就有了可靠的参照系，不会把每个新名称都当作从零出现的孤立技术。


### 12.12 为什么源序列追加和目标序列追加不能共用同一缓存假设

双向编码器允许早期源位置读取后面的源位置。因此，如果源序列从三个 token 增长到四个，原来三个位置的编码器表示也可能改变。不能把旧 memory 原样保留，只计算第四个位置，就宣称等价于重新编码完整源序列。这个差别来自允许依赖关系，不是某个框架实现得不够聪明。

因果解码器的目标历史不同：在固定参数和一致位置语义下，早先位置不能读取新追加的未来位置，所以它们先前计算出的逐层 K/V 可以继续使用。这正是目标侧增量缓存成立的关键前提。下一章的现代 decoder 和 M06 的 cache 会沿着这个前提推导，而不能把结论无条件推广到所有序列模块。

交叉注意力缓存同样依赖源 memory 不变。若用户在生成过程中修改源输入，原先投影好的 cross K/V 可能全部失效，需要重新计算并明确目标生成状态如何处理。只检查源长度没有变化还不够，因为内容、模型权重和位置配置改变也会影响缓存语义。缓存身份应包含实际计算条件，而不是只用一个字符串长度作为键。

这个例子说明，优化前先画依赖图非常重要。某个中间量能够复用，是因为它的所有输入和参数保持等价；若依赖条件改变，继续复用就是错误。学习缓存时应从这种数学与状态前提出发，再讨论页表、引用计数或设备存储，避免把缓存理解成“把算过的东西随便存下来”。

## 第十三课：十二道分级练习与完整参考判断

**练习一：三种 attention 的来源。** 写出 encoder self、decoder self、decoder cross 的 Q 和 KV 分别来自哪里。答案依次是源表示/源表示、目标表示/目标表示、目标表示/编码器最终表示。再写出是否需要目标因果限制：编码器不需要，解码器 self 需要，cross 一般读取全部有效源位置。不能只回答“三种都是 QKV”。

**练习二：目标右移。** 正确目标为 $[9,8,EOS]$，写出 decoder input 和 labels。答案分别是 $[BOS,9,8]$ 与 $[9,8,EOS]$。若目标长度包含 EOS，一条训练样本的监督位置数仍为三；BOS 是输入，不作为这条序列的下一个目标。

**练习三：位置编码手算。** D 为四，位置零和一的编码如第三课所示。把位置一误当一度会产生什么变化？答案是正弦余弦值完全不同，且违反公式使用弧度的约定。代码不报错不等于定义正确。

**练习四：LayerNorm 常量输入。** 输入 $[3,3,3]$，均值三、方差零，去均值后全零；无仿射时输出全零，有 beta 时输出 beta。不能要求方差为一，也不能删除 epsilon 让零除零产生 NaN。参考解直接运行本章函数并加入断言。

```python
constant = np.full((2,3),3.)
np.testing.assert_array_equal(layer_norm_reference(constant),np.zeros((2,3)))
beta = np.array([1.,2.,3.])
np.testing.assert_allclose(layer_norm_reference(constant,beta=beta),
                           np.broadcast_to(beta,(2,3)))
```

**练习五：post-norm 与 pre-norm。** 选一个非恒等子层 f，分别计算 LN(x+f(x)) 和 x+f(LN(x))，给出不相等反例。下面使用逐元素平方作为容易观察的子层，不宣称它是 Transformer attention。

```python
x_order = np.array([[1.,2.,4.]])
post = layer_norm_reference(x_order+x_order*x_order)
normalized_input = layer_norm_reference(x_order)
pre = x_order+normalized_input*normalized_input
assert not np.allclose(post,pre)
```

**练习六：FFN 参数共享。** 对两个相同 token 表示逐位置应用同一 FFN，输出应相同；把其中一个输入改变，另一个输出不受影响。这个测试只能验证 FFN 的局部性，不适用于已经包含 attention 的整层，因为整层允许位置之间交互。

**练习七：目标未来非干扰。** 使用本章前缀测试，再故意在 decoder self-attention 中删掉 causal mask，确认某个非退化输入可以让早期 logits 变化。参考答案要保留原实现和错误实现的对照，不只给正确版本的 PASS。

**练习八：源目标长度不同。** 源长度五，目标输入长度三，词表十六，输出应为 $[3,16]$，cross 分数为 $[H,3,5]$。如果得到 $[5,16]$，说明错误地把源位置当作输出 query 位置，或在某处交换了 Q 与 K 的来源。

```python
different_lengths = model.forward([3,4,5,6,7],[BOS,8,9])
assert different_lengths.shape == (3,16)
```

**练习九：编码器复用。** 比较 `model.forward(source,prefix)` 与先 encode 再 decode 的结果。参考代码如下。这个性质为生成时复用源 memory 提供证据，但不证明目标 self KV cache 已经实现。

```python
src = np.array([3,4,5])
prefix = np.array([BOS,7,6])
np.testing.assert_allclose(model.forward(src,prefix),
                           model.decode(prefix,model.encode(src)))
```

**练习十：参数对象独立。** 检查三组 attention 投影中相应矩阵不是同一个对象。随后有意把 decoder cross 的投影引用设成 decoder self 的投影，解释这变成参数共享设计，不能再声称与原先独立参数模型相同。值恰好相同和对象共享也要区分，后者会影响训练更新。

**练习十一：写出一次完整训练样本。** 为反转任务 $[3,4,5]\to[5,4,3]$ 添加 BOS/EOS，画出每个目标位置能读的目标前缀和全部源位置。答案第零位置输入 BOS、目标五；第一位置输入五、目标四；第二位置输入四、目标三；第三位置输入三、目标 EOS。所有四个位置都可以读源的三个有效位置。

**练习十二：结业小项目。** 从本章模块出发，完成复制或反转任务的训练与生成检查。最低交付是数据构造、明确 mask、有效 token 平均损失、固定验证集、贪心生成、EOS 与长度截断区分、保存并重载参数的等价性。参考完成顺序是先让一个极小 batch 过拟合，再加入不同长度，再加入 padding，最后测试未见组合；每次新增机制都保留前一步测试。

常见错误包括目标没有 shift 导致泄漏、cross mask 错用目标下三角、LayerNorm 沿错轴、多个层意外共享对象、生成时每轮重复添加 BOS，以及用随机权重输出解释模型能力。遇到问题先打印源、decoder input、labels 和三类允许矩阵；这些离散结构往往比某个深层浮点张量更容易直接暴露错误。

本章掌握门槛是你能不看图重画完整 encoder-decoder，并给每条边标出形状和信息来源；能独立实现三种 attention 的连接；能手算右移、mask、norm 和 FFN 的小例子；能设计一个未来目标泄漏测试。完成后再进入现代 decoder：你会看见它删去了哪些路径，替换了哪些组件，又保留了哪些根本机制。
