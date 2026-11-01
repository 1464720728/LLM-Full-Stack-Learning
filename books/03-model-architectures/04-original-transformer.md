# M04　手工拼装 2017 年的编码器—解码器 Transformer

[本卷导航](README.md) · 前置：[M03](03-attention.md) · 下一章：[现代稠密解码器](05-dense-decoder.md)

原始 Transformer 不是今天常见 decoder-only 模型的简单别名。它为序列到序列任务设计：编码器读取源序列，解码器一边读取已生成目标前缀，一边读取编码器的表示。先掌握这条结构，才能理解后来的删减、改造，以及新一代因果编码器—解码器与原始结构为什么也不等同。

## 1. 源文本和目标文本是两个时间轴

假设源序列是 `I like cats`，目标是 `我 喜欢 猫`。训练时 encoder 可读整个源序列；decoder 的第一个位置拿到起始标记，预测“我”；第二个位置拿到“我”，预测“喜欢”。它不能提前读到后面的正确目标词，否则会泄漏答案。

Encoder self-attention 的 Q/K/V 都来自源表示。Decoder self-attention 的 Q/K/V 来自目标表示，并施加因果 mask。Cross-attention 的 Q 来自目标表示，K/V 来自 encoder 输出。若源长 3、目标长 4，那么 cross-attention 分数矩阵是 `(4,3)`，没有理由必须是方阵。

这三种 attention 可以复用同一个基本函数，但使用不同参数与不同数据源。复用代码不意味着共享权重；原始网络中不同层、不同子模块通常有独立参数。

## 2. Post-LayerNorm 残差块逐行解释

本卷 `OriginalTransformer` 写出一个 encoder 层和一个 decoder 层，省略 dropout、可训练 norm 的 gain/bias 和多层堆叠，目的是使每一步可追踪。数学核心仍包括三类 attention、FFN、残差、位置编码与输出头。

Encoder 单层：

$$ U=\operatorname{LN}(X+\operatorname{MHA}(X,X,X)), $$
$$ H=\operatorname{LN}(U+\operatorname{FFN}(U)). $$

Decoder 单层：

$$ A=\operatorname{LN}(Y+\operatorname{MaskedMHA}(Y,Y,Y)), $$
$$ B=\operatorname{LN}(A+\operatorname{CrossMHA}(A,H,H)), $$
$$ Z=\operatorname{LN}(B+\operatorname{FFN}(B)). $$

这里的 PostNorm 指对子层输出与残差相加后的结果做归一化。现代模型常见 PreNorm，把 norm 移到子层输入处；两者一般不代数等价，不能只移动代码行而期望同一权重继续产生相同输出。

残差要求两条相加路径形状相同。Attention 内部可以切头、改变 value 维度，但输出投影必须回到 D。FFN 先把 D 扩展到更宽的中间维度，再回到 D。它在每个 token 位置独立执行，权重跨位置共享，因此本身不混合时间信息。

## 3. LayerNorm 的手算

对于向量 `[1,3]`，均值 2、方差 1，忽略 ε 时归一化为 `[-1,1]`。这里除的是维度数 D 而非 D-1，因为网络归一化的方差定义不是统计学无偏样本估计。加 ε 是为了常数向量也能稳定处理。

若把每个位置的 D 个通道归一化，就是 LayerNorm；若不小心对 T 轴求均值，就会让某个 token 的归一化依赖其他 token，甚至在 causal 模型中泄漏未来。对归一化轴的误解，与 mask 错误一样能造成“loss 特别好却无法真正生成”。

可训练 gain/bias 使模型能够重新缩放和偏移归一化结果。本卷简化前向省略了它们；完整训练脚本使用 `nn.LayerNorm`，包含可训练参数。比较这两个实现不能只比较名字，必须比较具体定义。

## 4. 正弦位置编码补上了什么

没有任何位置机制的 self-attention 对输入位置置换具有相应的等变结构：如果把 token 顺序和输出位置一起置换，计算本身不会知道“先后”。正弦编码为每个位置提供不同频率的坐标，再与 token embedding 相加。

偶数通道用 `sin(pos/10000^(2i/D))`，奇数通道用对应 cos。pos=0 时 sin 为 0、cos 为 1。不同频率使网络在多个尺度上区分相对与绝对位置。正弦编码不是训练出来的词义，它是提供位置线索的确定性特征。

原始方案将 embedding 乘 √D 再加位置编码。本卷前向保留了这个缩放。现代 RoPE 则主要作用于 Q/K 的成对通道，不是把一个位置向量加到 x 上；二者不能在同一 checkpoint 中任意替换。

## 5. 读取实际参考代码的顺序

打开 `labs/models/core.py`，先找 `OriginalTransformer.forward`。不要从构造函数里一堆随机矩阵开始读。先写一张张量表：源 IDs→源 embedding→encoder output；目标 IDs→目标 embedding→masked self-attn→cross-attn→FFN→logits。再回到 `__init__` 给每组参数贴标签。

`self.proj` 有三组，每组四个矩阵，对应 Q、K、V、O。`_mha(qx,kvx,p,mask)` 故意把查询输入和键值输入分成两个参数，以便 self-attention 时传同一个张量，cross-attention 时传不同张量。阅读 API 设计也能帮助理解结构。

`forward(source,target_input)` 返回每个目标位置的词表 logits。它没有执行采样，也没有自动添加 BOS 或移动标签。这些责任属于训练/生成调用方。把所有事情塞进一个 `forward` 会使教学阶段更难辨认错误边界。

运行：

```bash
python -m labs.models.run_lab transformer
python -m unittest tests.test_models_core.ModelMechanismTests.test_original_transformer_prefix -v
```

实验改变目标序列最后一个 token，前面位置 logits 应不变；另一个测试改变源序列，目标 logits 应变化。这两个测试分别证明因果遮罩起作用、cross-attention 路径真正连通。只测试输出形状无法证明这两件事。

## 6. 训练、教师强制和推理循环

Teacher forcing 是训练时使用真实目标前缀作为输入。它提高训练并行性，但推理时前缀来自模型自己的生成；一次错误可能改变后续全部输入，这称为分布偏移的一部分。不能通过训练阶段偷看完整目标来“解决”这个问题。

生成时先编码源序列，保存 encoder 输出，然后从 BOS 开始：运行 decoder、取最后位置 logits、选下一个 token、追加到前缀，直到 EOS 或长度限制。可以缓存 decoder self-attention 的历史 K/V，也可以缓存从 encoder 输出投影的 cross-attention K/V。前者随目标生成增长，后者通常对同一源输入保持固定。

源序列中有敏感数据时，encoder 状态与 cross K/V 也属于请求数据，不应跨无关请求复用。模型卷讲数学，但一旦实现缓存，就必须与推理框架卷的请求隔离和生命周期一起理解。

## 7. 练习与参考解

1. **画三个分数矩阵。** 源长 5、目标长 3、两头。答案：encoder `(2,5,5)`，decoder self `(2,3,3)`，cross `(2,3,5)`。解释各自的行列对应什么，不能只填数字。
2. **替换源输入。** 固定目标前缀，交换源 token 顺序。验收：输出改变。若完全不变，检查 cross K/V 是否误用了 decoder 自己的张量；但特别对称的人工参数也可能形成偶然不变，故测试使用固定非退化随机权重。
3. **因果泄漏故障。** 取消 decoder mask，然后改变目标最后一项。参考：前面 logits 现在通常改变。写出 failed test 的数值差，而不是声称“模型看到未来了”却无证据。
4. **实现两层 encoder。** 为每层创建独立矩阵，按层循环。验收：参数量增加、每层输入输出均为 `(T,D)`。如果把同一个字典重复放进列表，可能意外共享参数；用对象 identity 检查。
5. **加入 padding。** 给源 `[a,b,PAD,PAD]`，构造 key mask，比较与不补齐版本输出。参考：同一真实位置和相同位置编码下应一致；padding query 输出是否保留由上层决定，但不能影响真实 token。
6. **实现复制任务训练。** 在可选 PyTorch 环境把源整数序列复制到目标，使用 BOS、EOS 和 shift。验收：训练 loss 下降且新随机序列上复制正确率提高。参考首先只用短长 3–6、词表 8，再测长 7；不要把训练集正确率当泛化证明。
7. **解释新的 CED。** 参照本卷来源表中 DeepSeek-V4.1 的模型卡，指出它的 causal encoder/共享投影 KV 与本章双向源 encoder 翻译架构至少三处不同。参考答案：编码器因果性、输入/输出数据流、KV 来源与复用方式；“都叫 encoder-decoder”不足以认定相同结构。

通过标准：从零写出含 cross-attention 的前向，能隔离 decoder 因果性与 source 依赖，并能说清本章简化未覆盖哪些训练细节。
