# M15　综合项目一：真正训练、保存、恢复并生成一个微型语言模型

[本卷导航](README.md) · 前置：M01–M06、M14 · 下一章：[比较与源码研究](16-architecture-study.md)

这里交付两个完整可运行项目，而不是让你复制一段loss循环。第一个用NumPy手写神经语言模型和全部梯度；第二个用PyTorch自动微分训练手写Transformer。先完成小模型，可以用精确数值检查训练机制，再让自动微分承担复杂网络的反向传播。

## 1. 项目A的契约和边界

`TinyLanguageModel`是固定上下文的字符级MLP LM：取最近3个字符，查embedding，拼接，经过tanh隐藏层，投影到下一个字符的logits。它不是Transformer，也不能记住超过上下文窗口的信息。我们明确选择它，是为了让embedding梯度、矩阵梯度、非线性与优化更新都可以手算和差分。

数据是本地人工语法，例如猫/狗、坐/跑的英文短句重复组合。没有网络下载，也没有API费用。训练结果只展示学习某种简单分布；生成像句子的文本不代表通用语义能力，验证集也与训练集共享人工语法。

运行：

```bash
python -m labs.models.tiny_lm --steps 160
python -m labs.models.tiny_lm --steps 160 --save artifacts/models/tiny-lm.npz
```

第二条命令显式写出checkpoint；第一条不写文件。训练步骤有上限，默认CPU完成，无需申请H100。

## 2. 数据流从原始字符串到batch

先建立排序后的字符表，保证同一文本产生固定ID映射。若换了alphabet顺序却直接加载旧权重，相同ID对应不同字符，输出会混乱。因此checkpoint除了权重还保存alphabet和配置。

先按原始文本位置划分训练/验证段，再各自构造窗口。长度N、context=C产生N-C个样本。第i个输入是`ids[i:i+C]`，标签为`ids[i+C]`。随机batch按样本索引抽取，固定seed控制抽样。

这里使用有放回抽样，某个样本可重复出现；它不是每轮严格遍历全数据的epoch定义。报告steps比声称“训练了几个epoch”更准确。如果你改成无放回迭代，要重新说明数据顺序与最后不足batch的处理。

## 3. 完整前向的五个参数

参数表是E、W1、b1、W2、b2。设词表V、上下文C、embedding维e、hidden维h：

- E形状`(V,e)`。
- 三个embedding拼接后输入维度为C×e。
- W1为`(C*e,h)`、b1为`(h,)`。
- W2为`(h,V)`、b2为`(V,)`。

前向：`emb=E[ids].reshape(N,C*e)`；`hidden=tanh(emb@W1+b1)`；`logits=hidden@W2+b2`。缓存ids/emb/hidden是为了反向复用，不是持久推理KV。

看代码时每行都写下形状。广播bias时，同一bias加到所有样本。若错误地使用`b1:(N,h)`，参数可能绑定训练batch大小，换batch就坏；这类bug用多个batch形状测试很容易发现。

## 4. 把反向每一项推回去

令G为CE对logits的梯度。则：

$$ dW_2=H^\top G,\quad db_2=\sum_nG_n, $$
$$ dA=(GW_2^\top)\odot(1-H^2), $$
$$ dW_1=E_{flat}^\top dA,\quad db_1=\sum_ndA_n, $$
$$ dE_{flat}=dA W_1^\top. $$

把最后一项reshape回`(N,C,e)`，按ids将梯度累加到embedding表。这里必须处理重复ID，`np.add.at`保证每次出现都贡献。你应先对极小模型V=4、C=2、e=3、h=5检查所有参数类型，再扩大尺寸。

优化器使用SGD和全局梯度范数裁剪。`norm=sqrt(sum_all_parameters(g²))`，若超阈值则所有梯度乘相同系数。这保留整体方向但缩短步长；逐参数独立clip会改变不同参数块相对比例，不是同一算法。

## 5. 损失下降的证据链

代码输出初始训练CE、最终训练CE、验证CE、steps和数据说明。单元测试要求在固定种子下160步使训练CE至少下降0.5，这是对这个具体小实验的回归条件，不是一般训练保证。

如果loss不降，诊断顺序应是：检查标签shift；检查CE梯度；检查每类参数差分；检查参数是否真的改变；检查学习率与数值；最后再怀疑模型容量。盲目增加hidden或换H100无法修复错位标签。

如果训练loss降、验证不降，检查数据分布和过拟合。如果验证异常低，检查数据泄漏。不要为了让图好看而把训练集当验证集；本项目已经说明重复语法的局限，你还应在扩展时增加未见组合测试。

## 6. 保存与恢复不是只dump权重

本项目用`np.savez`保存参数、结构配置与alphabet，读取时禁止pickle，并检查参数形状。恢复后的同一输入logits必须精确一致，单元测试覆盖这一点。

继续训练还需要优化器状态、随机数状态、当前step和数据采样位置。本项目基础checkpoint用于推理恢复，并未宣称完整断点续训。SGD本身无动量状态，但随机batch和进度仍需保存；如果换Adam，更必须保存一阶/二阶矩估计。

项目练习会让你加入resume，但核心推理保存/加载已完整实现。用明确功能边界比在文件里写一个`state_dict`就声称“训练可复现”更可靠。

## 7. 项目B：手写Transformer并让PyTorch算梯度

```bash
python -m labs.models.torch_transformer --device cpu --steps 40
```

如果未安装PyTorch，会打印明确SKIP，不假装训练成功。需要强制要求依赖时加`--require-torch`。本书生成环境只验证NumPy项目，GPU结果不作声明。安装请按官方当前环境选择，不直接把别人的CUDA wheel命令照搬到Win11或WSL。

脚本包含手写QKV投影、多头reshape、causal mask、scaled scores、softmax、value混合、output投影、PreNorm残差、MLP、位置embedding和输出层。没有用`nn.Transformer`隐藏主体结构。默认两层、dim32、四头、context24、batch8，适合CPU；个人GPU或H100单卡只需显式`--device cuda`，不自动占用多卡。

训练循环使用AdamW、梯度裁剪、固定CPU抽样generator。验证使用固定batch；生成阶段切eval、关闭梯度，每次取最后位置argmax，并裁掉超过context的输入。这个裁切也使learned absolute position从窗口起点重新编号，是本toy的生成约定，不等同无限长KVcache。

## 8. 项目验收任务与参考方案

1. **不看答案重写MLP前向。** 验收三种N下输出形状正确，用相同权重与参考逐元素比较。答案重点是embedding拼接轴，而非随意reshape凑尺寸。
2. **重写全反向。** 每种参数至少检查两个坐标，包含重复embedding ID。验收相对/绝对误差报告；避免在非平滑点比较ReLU差分，本项目tanh更易检查。
3. **加入可恢复训练。** 保存params、step、rng.bit_generator.state及配置，运行80步保存，再80步，比较连续160步。参考：保存发生在同一更新边界，加载后不额外抽batch，才能逐步一致。
4. **增加早停。** 每固定间隔测验证，patience到达后恢复最佳参数。参考必须复制数组而非保存引用，否则“最佳参数”会随后续更新继续变化。
5. **将context从3改5。** 重新初始化W1或显式迁移，不能直接加载形状不匹配checkpoint。交付容量、训练损失和未见组合表现，说明更大context未必对人工语法有帮助。
6. **Transformer因果测试。** 给两条仅最后token不同的序列，eval模式下比较更早logits。验收误差阈值按float32制定。若忘mask，训练loss可能仍下降，所以此测试不可由训练结果替代。
7. **替换一个组件。** 将LayerNorm改RMSNorm或learned position改RoPE，要求逐项解释权重/形状变化并重训。不要直接用不同架构checkpoint作为所谓兼容测试。
8. **资源报告。** CPU默认跑通后，若使用5070Ti或H100，限制一张可用GPU、40–200步、小batch；记录device、版本、dtype、峰值显存、wall time，结束进程释放资源。没有实际运行就标“未验证”，不得填估计值当测量。

项目通过标准：你能解释每一项梯度、恢复一个checkpoint、构造泄漏测试，并亲手修改一个模型组件后重新训练。生成一段文本只是验收中的一项。
