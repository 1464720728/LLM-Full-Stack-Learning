# M12　混合架构与注意力残差：沿序列读信息，沿深度选表示

[本卷导航](README.md) · 前置：[M08](08-moe.md)–[M11](11-state-space.md) · 下一章：[数值与量化](13-quantization.md)

现代模型可以在不同层选择不同的序列算子，也可以改变跨层残差连接。理解它们的关键是区分三个轴：时间/序列、通道、深度。Token attention通常沿序列选择历史；Attention Residuals沿深度选择先前层表示；MoE路由选择专家。它们可以同时存在，不能互相替代名称。

## 1. 普通残差展开后保存了什么

PreNorm残差可抽象为 `x_(l+1)=x_l+f_l(x_l)`。展开得到输入embedding与所有先前增量的累加。每条历史增量在这个显式求和中的系数是1，但f_l本身依赖先前状态，所以不能把整个网络误认为一组互相独立的向量求和。

当层数增加，模型可能希望不同位置更有选择地使用不同深度产生的表示。注意力残差的一种设计让当前层用可学习query，对多个历史深度来源打分，softmax后混合这些来源。这改变了层间信息聚合方式，而不是直接把token上下文窗口变大。

## 2. Full Attention Residual的可执行原语

本卷输入sources形状为 `(L,T,D)`：L个深度来源，每个来源有T个token、每个token D维。对每个来源做RMSNorm作为key；与本层learned query点积，得到 `(L,T)` 分数；沿L轴softmax；values保留未归一化的原始sources。

$$ a_{l,t}=\operatorname{softmax}_{l}(q^\top\operatorname{RMSNorm}(x_{l,t})), $$
$$ y_t=\sum_l a_{l,t}x_{l,t}. $$

为什么key归一化而value不跟着归一化？key归一化控制匹配时的尺度影响；value保留输出内容与幅度。若两者都归一化，就是另一种函数。官方具体变体还包括初始化、block划分、参数共享和执行优化，必须单独核对。

手算来源为 `[1,0]`、`[0,2]`、`[3,3]`。若query全零，三个score都是0，权重各1/3，输出就是三个来源的均值。这个边界测试非常有用：如果输出不是均值，softmax轴或value处理很可能错了。

## 3. 深度attention不应混到token轴

对于每一个token位置t，聚合它在不同深度的表示，不把其他token位置加入这次深度softmax。来源张量的同一t对应相同序列位置，但它已经可能通过先前的token attention吸收其他位置的信息。

如果错误地把 `(L,T)` flatten后softmax，你既混深度又混时间，并可能泄漏未来。保持形状注释和axis命名是最便宜的预防手段。`attention_residual`使用显式einsum，把收缩和保留轴写清楚。

Full版本保留很多深度来源，可能增加存储和读取成本。Block Attention Residual把若干层内的增量合并为块，再在块来源之间选择，折中表达力与资源。本卷`block_depth_aggregate`给出可追踪的块快照演示，updates由外部给定。继续完成[完整Block AttnRes Decoder工作坊](WORKBOOK_ATTNRES.md)：`TinyBlockAttnResDecoder`实际执行GQA/RoPE attention、SwiGLU、两次层内深度聚合、partial封存，并支持full与逐token cache路径。其block大小按attention+MLP子层计数；完整实现、参考解与回归测试均已提供，随机权重模型不宣称复刻官方checkpoint。

## 4. mHC与AttnRes为什么应分开学习

Manifold-Constrained Hyper-Connections涉及多条残差流及受约束的混合；AttnRes聚焦对历史深度表示进行注意力聚合。两者都改残差，并不意味着它们是同一个公式。

为了理解约束混合，本卷提供Sinkhorn小实验：把一个正方矩阵交替按行、按列归一化，使其逐渐接近双随机矩阵。双随机指行和、列和都为1，不是“每个元素都等于1/n”。有限迭代只有近似满足，epsilon和数值实现也会影响结果。

具体代码在log-domain迭代：对每行、每列减去稳定的log-sum-exp，最后才取exp。若直接先exp再归一化，极端logits可能让整列下溢为零，随后除零得到NaN。`[[1000,0],[1000,0]]`是专门覆盖的反例，稳定实现给出每项0.5。输入要求有限方阵与正整数迭代次数。

这只是理解某类mHC约束思想的数学工具。真实多流架构还需要定义pre混合、post写回、residual混合以及训练参数化；Single-Pass mHC等后续变体更不能用一句“做了Sinkhorn”全部覆盖。读论文时把每条流及矩阵维度画出来，再去对应kernel。

## 5. 混合模型为什么需要逐层状态表

假设12层中9层用线性状态、3层用全局attention。其运行状态不是统一的KV公式：线性层有固定S等状态，全局层有随T增长的K/V。若另有局部层，其缓存可受window限制；压缩层又有压缩池与indexer状态。

因此容量估算应按层类型求和。代码中通常有layer type列表、周期性规则或配置字段。不能只看模型名字判断所有层相同。模型实现、scheduler和cache manager都要理解这些状态的生命周期。

混合也改变任务能力与性能权衡：固定状态提供低成本历史摘要，全局层提供更直接的长程读取。具体配比是否好，需要训练与评测；本卷不会从“同时用了两种技术”推导出一定优于单一结构。

## 6. 当前公开架构作为阅读对象

截至2026-11-01核对的官方资料给出以下方向，精确字段见本卷来源表。这里只摘取用于定位代码的事实，不把其他未公开细节补成猜测。

| 家族/资料对象 | 本卷应追踪的结构 | 不能做的简化 |
|---|---|---|
| DeepSeek-V3 | MLA、MoE | 不能把后续V4全按V3解释 |
| DeepSeek-V4-Flash | 压缩/稀疏注意力、mHC | 压缩注意力不等于低秩MLA |
| DeepSeek-V4.1-Flash | 因果编码器解码器、CSA2、条件记忆、修订残差 | 不能套用每decoder层独立产生全部KV的假设 |
| Kimi-K3 | KDA与Gated MLA混合、AttnRes | 深度attention不等于token attention |
| MiMo-V2.6-Flash-MOPD | 局部SWA与全局GA交替、MoE、多模态编码器 | 混合attention不自动意味着线性attention/SSM |

这些资料用作架构辨识练习，不代表本卷在本地部署了对应巨型checkpoint。小模型实现和旗舰源码阅读是两种证据层次；写清差别能防止把教学验证夸大成生产验证。

## 7. CED与条件记忆的最小实验

`causal_encoder_decoder`先计算因果encoder状态，再把这些状态作为decoder可读K/V来源，两个阶段都保持因果性。它没有完整复刻官方CED的训练与cache复用，却能测试一个核心区别：decoder读到的历史记录不必来自每一decoder层自己的历史表示。

`conditional_ngram_memory`根据最近几个token计算确定性哈希，查询一张表。它说明参数记忆可以由token条件索引，区别于对所有历史进行softmax匹配。哈希碰撞、学习到的表、门控和压缩都是独立问题；本toy只实现查表和重复n-gram的一致性。

## 8. 练习与参考解

运行 `attnres`、`block_attnres`、`hyperconnections`、`ced`、`conditional_memory` 五个原语lab，以及完整模型命令 `python -m labs.models.block_attnres_decoder`。原语的外部update计数和完整模型的attention/MLP子层计数是两个明确接口，不能不核对含义就比较相同数字的block_size。

1. **零query。** 任意L个来源，query=0，输出等于均值。验证每token深度权重和为1，而不是每层时间权重和为1。
2. **深度偏好。** 对两条正交来源调大query某一通道，观察权重向相应来源移动。答案不是输出必然等于那条来源，有限分数仍是soft mixture。
3. **保留value幅度。** 将一条来源整体乘10，比较key归一化后的分数和最终value贡献。参考：归一化可能使score近似不变，但value幅度仍变化；epsilon使极小向量处不完全尺度不变。
4. **Block边界。** block_size=2、四个updates，候选数量依次1、2、2、3。解释完成块为何形成新source，当前未完成块为何单独保留。然后改成block_size=1，检查候选增长。
5. **双随机矩阵。** 用3×3 logits跑1、5、100次Sinkhorn，分别报告行/列和误差。参考：最后一次操作决定某一轴更接近精确归一化，另一轴需迭代收敛，不应声称1轮必然双随机。
6. **CED因果性。** 改最后输入token，较早encoder与decoder输出必须不变。对照原始Transformer的源双向encoder，解释不同因果契约。
7. **条件记忆碰撞。** 表只有4行，枚举多个二元token组合，找两组落同一slot。参考：碰撞意味着共享同一存储位置，不证明它们语义相同；真实设计需要处理统计、多个哈希或学习机制。
8. **真实源码检查单。** 任选表中一个模型，写出层类型、norm、位置、FFN、残差、状态、输入模态七项，并给每项具体源码或官方字段依据。未知项保持未知。验收重在证据与形状，而非模型名字背得多。

通过标准：你能在同一图上区分序列、深度、通道与专家四种选择机制，并实现各自最小可验证例子。
