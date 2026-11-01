# 模型卷一手资料与版本边界

核查日：**2026-11-01**。本卷代码为独立教学实现，不复制下列项目实现，不宣称能直接加载其checkpoint。论文公式、模型卡事实与课程自造示例分开呈现；模型规模和性能宣传不作为本地测量结果。

| ID | 一手来源 | 用途与边界 |
|---|---|---|
| S01 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 原始encoder-decoder、多头、位置与残差结构 |
| S02 | [GQA](https://arxiv.org/abs/2305.13245) | MHA/MQA之间的KV头共享；训练转换并非任意删头 |
| S03 | [Transformers are RNNs / Linear Attention](https://arxiv.org/abs/2006.16236) | 可分解kernel与递推思想；本卷ELU+1示例独立编写 |
| S04 | [DeepSeek-V3技术报告](https://arxiv.org/abs/2412.19437)、[官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) | MLA/MoE源码导读；本卷latent toy省略真实RoPE等分支 |
| S05 | [Mamba论文](https://arxiv.org/abs/2312.00752)、[官方实现](https://github.com/state-spaces/mamba) | selective state与reference scan阅读任务；不声称scalar toy是Mamba |
| S06 | [Attention Residuals官方仓库](https://github.com/MoonshotAI/Attention-Residuals)、[论文](https://arxiv.org/abs/2603.15031) | 深度attention及Full/Block区分；课程原语不是完整官方block |
| S07 | [mHC论文](https://arxiv.org/abs/2512.24880) | 多残差流约束混合的阅读对象；Sinkhorn仅为数学辅助实验 |
| S08 | [RMSNorm论文](https://arxiv.org/abs/1910.07467) | 不去均值的归一化；实际gain/epsilon须按模型核对 |
| S09 | [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | 旋转与相对点积性质；pair布局及长上下文策略另核对 |
| S10 | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 门控FFN结构；课程SwiGLU数值例子为自造 |
| S11 | [Switch Transformers](https://arxiv.org/abs/2101.03961) | 稀疏专家、负载与容量学习；不把所有router都当softmax top-1 |
| S12 | [DPO论文](https://arxiv.org/abs/2305.18290) | 成对偏好目标；课程只检验小型标量loss与梯度方向 |
| S13 | [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 组相对优势背景；课程clipped surrogate不是完整GRPO训练器 |
| S14 | [LoRA论文](https://arxiv.org/abs/2106.09685) | 低秩增量参数化；不等同MLA缓存设计 |
| S15 | [Vision Transformer](https://arxiv.org/abs/2010.11929) | patch序列化的原始阅读；课程patch+projection不具备视觉语义能力 |
| S16 | [Transformers官方仓库](https://github.com/huggingface/transformers) | configuration/modeling/processor/cache的真实源码研究 |
| S17 | [micrograd作者仓库](https://github.com/karpathy/micrograd) | 标量autograd的教学桥梁，非生产框架推荐 |
| S18 | [nanoGPT作者仓库](https://github.com/karpathy/nanoGPT) | 训练工程阅读桥梁；README已提及后续nanochat，课程不称其最新生产标准 |
| S19 | [DeepSeek-V4-Flash官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md)、[配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | CSA/HCA、mHC作为公开架构阅读方向 |
| S20 | [DeepSeek-V4.1-Flash官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md) | CED、CSA2、Engram、修订残差的公开事实；不推导未公开实现 |
| S21 | [Kimi-K3官方仓库](https://github.com/MoonshotAI/Kimi-K3) | KDA与Gated MLA混合、AttnRes、多模态方向 |
| S22 | [MiMo-V2.6-Flash-MOPD官方模型卡](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/main/README.md) | SWA/GA混合、MoE和模态编码器；不误称为SSM |
| S23 | [PyTorch SDPA官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 真实API mask/dropout/backend核对；stable是移动链接 |

## 怎样固定你自己的研究版本

上述GitHub/Hugging Face分支链接会变化。本次实际查询commit接口并读取了下列固定版本的目标文件，详见[固定版本源码工作坊](SOURCE_WALK.md)。其余分支链接仅表示核查日的公开资料；阅读任务要求你在实际checkout后记录`git rev-parse HEAD`，把代码路径与环境版本一起写进报告。论文可选择明确arXiv版本；新增版本可能修正公式或实验，需要保留引用版本。

| 项目 | 实际核查的commit | 读取文件 |
|---|---|---|
| Attention-Residuals | `85e22310fe5ee860b4a023de312d791de8a5a5e6` | `README.md`中的Full/Block说明与伪代码 |
| Mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | `mamba_ssm/ops/selective_scan_interface.py` |
| Transformers | `ca8c871ede62b521a75a2ddfc5e1968ac9c8e097` | `src/transformers/models/llama/modeling_llama.py` |

不要把“核查日”当作“已经下载并跑通官方完整实现”。本卷实际执行证据来自课程的NumPy tests和CPU小项目。PyTorch依赖、GPU后端与官方大模型权重均需在满足条件的环境另行验证。

## 如何引用本卷的简化实现

| 文件/函数 | 已完整实现的教学内容 | 不包含 |
|---|---|---|
| `core.OriginalTransformer` | 一层postnorm encoder/decoder、三类attention、正弦位置、FFN | 完整原论文训练配方、dropout、norm gain/bias |
| `core.TinyDecoder` | 多层GQA+RoPE+SwiGLU前向与KVstep | 训练梯度、真实tokenizer、生产cache分配 |
| `core.latent_attention` | 无位置分支的共同latent吸收等价 | 完整MLA、厂商checkpoint兼容 |
| `core.delta_recurrence` | 遗忘后delta写入与读取 | 完整KDA/gated delta block、GPU chunk kernel |
| `advanced.compressed_sparse_attention` | 因果均值压缩、选择、局部支路 | 学习压缩器、真实CSA2 indexer/跨层复用 |
| `advanced.conditional_ngram_memory` | 确定性token n-gram索引查表 | Engram训练/门控/完整哈希系统 |
| `advanced.block_depth_aggregate` | 可观察的block来源快照与深度聚合 | 官方AttnRes完整子层与优化 |
| `block_attnres_decoder.TinyBlockAttnResDecoder` | 实际GQA/RoPE+SwiGLU、Block AttnRes来源生命周期、full/step、显式最终深度读出 | 官方checkpoint、训练梯度、norm gain、dropout、生产优化；初始化及head约定见工作坊 |
| `advanced.clipped_policy_objective` | 标量importance ratio裁剪与解析梯度 | rollout、KL、token mask、分布式GRPO |

这种明确边界使你能把课程代码当正确性工具，再有步骤地进入真实实现，而不会把几个简化函数误认为旗舰系统已经复刻完成。
