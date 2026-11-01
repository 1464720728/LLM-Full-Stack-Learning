# 模型结构卷：把公式写成代码，把代码变成可验证的模型

本卷从字符、张量与微分开始，亲手实现原始Transformer、现代稠密decoder、KV/GQA、低秩缓存、MoE、稀疏/线性注意力、SSM、注意力残差，再完成训练与架构比较项目。它与[基础卷](../00-foundations/README.md)、[推理框架卷](../04-inference-systems/README.md)、[算子硬件卷](../05-kernels-hardware/README.md)相互衔接。

请先完成基础卷的Python函数/类、数组、测试和命令行。所有核心机制与NumPy训练项目在普通CPU可运行；可选PyTorch训练默认也用CPU。只有进入真实GPU算子/性能任务才需要GPU，且明确单卡与步数限制。本卷没有大权重下载、自动API调用或持续GPU任务。

## 学习顺序

| 章 | 内容 | 命名 CLI 实验数（不含正文代码） |
|---|---|---:|
| M01 | [从一个汉字到一块张量](01-tokens-tensors.md) | 2 |
| M02 | [损失、链式法则与反向传播](02-gradients-training.md) | 3 |
| M03 | [逐项实现稳定注意力](03-attention.md) | 1 |
| M04 | [原始编码器—解码器Transformer](04-original-transformer.md) | 1 |
| M05 | [完整稠密解码器](05-dense-decoder.md) | 2 |
| M06 | [KV Cache与MHA/GQA/MQA](06-kv-gqa.md) | 2 |
| M07 | [低秩、矩阵吸收与MLA](07-latent-attention.md) | 1 |
| M08 | [MoE路由、专家、容量与合并](08-moe.md) | 1 |
| M09 | [局部、块稀疏与压缩注意力](09-sparse-attention.md) | 2 |
| M10 | [线性注意力与Delta记忆](10-linear-attention.md) | 2 |
| M11 | [状态空间与选择性递推](11-state-space.md) | 2 |
| M12 | [混合架构、AttnRes与残差变体](12-hybrid-residuals.md) | 6 |
| M13 | [模型数值与量化](13-quantization.md) | 1 |
| M14 | [采样、SFT、DPO、组优势与多模态](14-training-modalities.md) | 7 |
| M15 | [综合项目：训练并恢复微型语言模型](15-tiny-language-model.md) | 2 |
| M16 | [综合项目：架构比较台与真实源码研究](16-architecture-study.md) | 1 |
| M17 | [深度工作簿：两token逐数追踪与MoE增量实现](WORKBOOK.md) | 3 |

不要将“运行成功”当成“已经学会”。每章先手算或预测，运行后对照中间量，再做修改题与故障题；参考解在各章练习后紧随给出。M17是长工作簿，带26个中间阶段的真实trace，建议与M03–M08交叉使用，不必等到最后才看。

## 怎样使用扩写后的正文

M01–M16 是完整主课，不需要先读懂源码才开始。每章沿着具体问题、词汇与形状、逐项手算、完整参考代码、反例、至少十二道分级练习及答案、渐进项目展开。主课的正文与习题已大幅重写；M17、源码工作坊和 AttnRes 工作坊用于交叉验证，不替代主课。

一次学习只推进一到两个小课。先在纸上写出输入、输出与每个轴的意义，再计算小例子的所有中间量，最后运行代码。遇到断言失败，保留失败输入和第一个不一致的中间量；修好后补一个能够区分错误实现与正确实现的测试。答案紧随练习提供，但建议先独立写出自己的实现，再解释两种实现为什么等价。

本卷已经实际执行的正文代码、依赖跳过、手写反向验证和小训练结果见[深度与执行审计](DEPTH_AUDIT.md)。表中的 CLI 数只是实验命令的入口数量；各章另外包含大量完整 Python 代码块、故障注入和参考答案。

## 三条实际代码路线

1. **数值机制**：`core.py`与`advanced.py`提供小型完整参考，`run_lab.py`有31个命名实验。每个简化都有边界说明。
2. **训练项目**：`tiny_lm.py`手写全部NumPy梯度与checkpoint；`torch_transformer.py`手写Transformer结构并用可选PyTorch反向传播。
3. **实现与修改**：`trace_decoder.py`逐阶段对应TinyDecoder，`moe_decoder.py`完成Dense→MoE结构改造，`block_attnres_decoder.py`提供完整注意力残差模型及[逐步工作坊](WORKBOOK_ATTNRES.md)，`compare.py`提供跨架构的正确性与state计数比较。

```bash
python -m labs.models.run_lab attention
python -m labs.models.trace_decoder --active-ffn
python -m labs.models.tiny_lm --steps 160
python -m labs.models.moe_decoder
python -m labs.models.compare
python -m unittest discover -s tests -p 'test_models*.py' -v
```

## 证据和能力边界

课程NumPy测试覆盖稳定softmax、数值梯度、full/step等价、mask因果性、RoPE、GQA、latent吸收、MoE路由与capacity、稀疏/linear/SSM等价、量化误差、组优势、trace与结构变体。它们验证小型CPU实现，不证明旗舰训练质量或GPU性能。

随机权重的TinyDecoder负责结构正确性；它不具备自然语言能力。人工语法小模型展示训练闭环，不是通用能力benchmark。真实项目阅读入口、核查日期和简化对照见[一手来源与边界](SOURCES.md)。完成M17后进入[固定版本源码工作坊](SOURCE_WALK.md)，逐项核对Llama、Mamba和Attention Residuals的真实函数、维度及简化差异。所有核心参考代码完整；习题是让你独立重写/修改，不依赖隐藏答案或未实现核心。

## 本卷毕业作品

提交一台自己重写的微型decoder、一套带故障案例的测试、一份训练与checkpoint报告、一个MoE/linear等结构变体、一个真实开源模型的张量/状态/源码映射。你应能解释并定位数值与接口错误，再把reference交给框架和kernel实验使用。
