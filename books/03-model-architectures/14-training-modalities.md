# M14　从分布到行为：采样、SFT、偏好学习、组相对优化与多模态

[本卷导航](README.md) · 前置：[M02](02-gradients-training.md)、[M05](05-dense-decoder.md) · 下一章：[训练项目](15-tiny-language-model.md)

这一章不把训练方法当名词清单。每种方法都落到小型可运行数值目标：采样改变怎样从分布选择；SFT改变哪些标签概率被优化；DPO比较成对回答；组相对方法从一组结果构造优势；多模态改变输入怎样变成可计算表示。它们可能共享同一个Transformer骨干，却不是同一层操作。

## 1. Temperature、top-k、top-p具体改了什么

温度T>0时用 `softmax(logits/T)`。T变小通常使分布更尖，T变大更平；T=0在本实验定义为greedy argmax，不能真的在代码里除0。温度不改变模型参数，也不等同服务商的reasoning effort。

Top-k保留最高的k个候选再归一化。Top-p按概率从大到小排序，保留累计概率首次达到p的最小前缀。本实现同时开启时先top-k再在剩余分布上top-p，顺序是明确约定；不同provider组合规则需要核对。

例：概率`.6,.25,.15`，p=.8，保留前两项，因为第一项累计.6不足，第二项达到.85。若错误地只保留累计值≤.8的项，就会丢掉跨越阈值的第二项。单元测试应覆盖这种边界。

同一seed只是随机数发生器状态的约定。输入分布一旦改变，即使用同一随机数也可能选到不同token；API服务还可能不承诺跨版本、跨并发形状逐bit可复现。做实验时记录你真正控制的因素。

## 2. 预训练与SFT：相同CE，不同数据和监督位置

预训练通常在大量文本上做下一token预测。SFT用指令/对话示范，并可只对assistant回答位置计算损失。对话中user文本仍作为上下文影响输出，但它的目标位置可以被loss mask排除。

设4个位置、3类均匀logits，只有最后两个assistant位置参与loss。损失仍是ln3，因为对参与位置取平均；前两个位置对直接输出logits的梯度为0，后两行梯度是单样本CE梯度除2。**前两个输入embedding仍可能通过后续attention间接获得梯度**，不能把“该位置输出loss被mask”理解为“该位置完全不参与训练”。

运行 `loss_mask` lab打印完整梯度矩阵。它在logits层演示mask，不包含整个模型反向，因此请用上述依赖图理解间接梯度，避免从局部实验得出过强结论。

## 3. DPO的最小可计算目标

给同一prompt的chosen与rejected回答，取它们在当前policy和固定reference下的序列log probability。定义：

$$ m=\beta[(\log\pi(y_w)-\log\pi(y_l))-(\log\pi_{ref}(y_w)-\log\pi_{ref}(y_l))], $$
$$ L=-\log\sigma(m)=\operatorname{softplus}(-m). $$

当policy相对reference的偏好差等于0时，m=0，loss=ln2。若policy更偏好chosen，m增大，loss下降。梯度对chosen logprob为`-βσ(-m)`，对rejected为相反数。因此梯度下降会推高chosen、压低rejected，具体参数更新还要乘网络的logprob梯度。

例如policy和reference的logprob都为chosen=-3、rejected=-4，β=.1，loss=.6931。policy改为chosen=-2、rejected=-5，参考不变，m=.2，loss约.5981。这是目标函数检查，不是一次完整偏好训练。

序列logprob通常是回答token条件logprob之和。回答长度、EOS处理、prompt mask、reference版本和tokenizer都会影响数值。不能拿两个API返回的某个摘要分数直接替代而不核对定义。不同DPO变体还可能有长度归一化和其他损失设计，本章固定原型后再扩展。

## 4. 从奖励到组优势，再到策略梯度

给同一问题采样三条回答，奖励为0、1、2。均值1、总体标准差√(2/3)≈.8165，标准化优势约`[-1.2247,0,1.2247]`。奖励全部相同则优势全0，组内没有相对学习信号。

这一步不需要value网络，但它还不是完整GRPO。完整训练还要产生rollout、计算新旧policy token概率、处理mask、截断、reference约束、训练稳定性与分布式分组。本卷只实现一个可检查的标量clipped surrogate：

$$ L=-\operatorname{mean}\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A), $$
$$ r=\exp(\log\pi-\log\pi_{old}). $$

在r=1、未裁剪处，对logπ的梯度为`-A/N`。正优势会推动概率上升，负优势推动下降。对正优势，r已经超过1+ε后，继续提高概率的局部奖励被裁剪；负优势对应的限制方向不同，必须根据min表达式推导，不能套一个对称“所有超界梯度都是零”的口号。

`grpo` lab打印优势、损失和梯度；测试用有限差分验证未处于不可导边界的坐标，并检查正优势上界裁剪。它明确省略KL与真实token训练，因此结果应叫“组优势与clipped目标实验”，不能称“训练好了一个GRPO模型”。

## 5. RLHF、在线/离线与奖励可验证性

RLHF是以人类反馈构造训练信号的一类流程，可能训练reward model，再用策略优化；偏好数据方法和可验证奖励方法在反馈来源、在线采样成本和目标上不同。On-policy强调训练使用的轨迹与当前或近当前策略的关系，不是说必须每个step从零重新部署模型。

数学题最终答案可验证，仍可能存在格式投机、错误奖励或训练分布偏差。代码测试通过可能依赖漏洞或不充分用例。Agent任务的奖励尤其容易受工具环境、超时、网络和隐藏测试影响。学习后训练时要同时读奖励协议，而不是只读优化器公式。

## 6. LoRA是参数化方式，不是独立训练目标

LoRA增加低秩参数AB，可以用SFT、DPO或其他目标训练。低秩适配和偏好目标处在不同层级。运行 `lora` lab，验证独立adapter与合并权重前向等价。若W是量化表示，实际合并可能需要反量化、重新量化并引入误差，不应直接把数学等价当所有部署路径逐bit一致。

## 7. 图像怎样成为可被模型读取的token

对4×4 RGB图像，patch size=2，每个patch含2×2×3=12个数，总共4个patch。展平得到 `(4,12)`，乘视觉投影 `(12,D)` 得到4个D维visual tokens。真实ViT还含位置、多个编码层、归一化等；多模态LLM还可能使用resampler、projector、特殊边界或图像切片策略。

本卷 `patchify` 明确了patch顺序；`cross_modal_read`让文本query对视觉token做cross-attention。这个toy能验证图像变化是否影响输出，但没有训练视觉语义，不能识别猫狗。另一路融合方式是将视觉embedding与文本embedding拼接，统一进入因果模型，两者的mask与位置处理不同。

视频增加时间轴，音频可从波形或频谱生成帧/patch，再经encoder映射。输入序列数量、采样率、空间分辨率影响成本。用户上传一张图“看起来只有一个附件”，模型内部可能对应大量视觉token；Agent成本核算必须以实际服务规则为准。

## 8. 练习与参考解

1. **Top-p边界。** 对`.6,.25,.15`、p=.8应保留两项，重新归一化约`.7059,.2941`。写一个不依赖随机抽样频率的候选集合测试。
2. **SFT间接梯度。** 只监督最后token，解释前面输入embedding是否可能有梯度。答案：可能，经attention依赖返回；输出logits的直接梯度为零不代表输入无影响。
3. **DPO差分。** 对chosen logprob=-3用±`1e-5`差分，m=0时导数应为-β/2。rejected导数相反。改变reference后重新计算m，不能把reference项删除。
4. **组奖励平移。** 对奖励加10，优势不变；乘正数3，在ε可忽略时优势近似不变；乘负数则顺序反转。全部奖励相同输出零。验收测试这些不变量。
5. **裁剪方向。** A=-1，r=.5、ε=.2，哪个分支更小？答案：raw=-.5、clipped=-.8，取-.8，所以该区间对r局部梯度为零。A=-1、r=1.5则raw=-1.5被选中，梯度不为零。
6. **视觉patch顺序。** 单通道4×4数值0..15，左上patch应`[0,1,4,5]`，右上`[2,3,6,7]`。若得到`[0,1,2,3]`，你把整行当成patch了。
7. **模态影响测试。** 保持文本query，改变图像patch，输出应改变；保持视觉tokens只调温度，不应改变encoder表示。参考：分别定位输入建模与输出采样的责任。
8. **小型后训练项目。** 用M15的toy LM输出两种风格回答，构建少量人工偏好，先只对最后分类头执行DPO数值更新。验收包括偏好margin提高、独立样本评估、reference不更新与数据划分记录，不能只展示训练loss下降。

通过标准：能手算并编程检查几类目标的梯度方向，能区分数据、目标、参数化、采样与模态输入这五个维度。

## 9. 先画五个独立维度：为什么这些方法不能互相替代

一个语言模型实验至少有五个可以独立选择的维度。数据决定模型看到什么示范或反馈；目标决定哪些输出受到奖励或惩罚；参数化决定哪些权重能更新；输入表示决定文字、图像或声音如何进入计算；解码规则决定已训练分布怎样变成具体输出。SFT、DPO、LoRA、视觉 projector 和 top-p 分别主要位于不同维度，所以不能把它们当成一组互斥模型架构。

例如同一个 decoder 可以冻结大部分参数，用 LoRA 训练，数据是带图像的问答，目标是 assistant-only 交叉熵，推理时用温度和 top-p。这里五个选择同时存在。若效果变化，要知道这次实验改了哪个维度，否则你可能把数据清洗带来的提升错误归因于采样参数，或把减少可训练参数误认为更换了损失函数。

训练与推理也要分开观察。训练更新参数，推理通常在固定参数下计算分布并选 token。温度让输出更集中，不会把模型权重训练得更确定；SFT 损失下降，不保证使用任意解码配置都产生好结果。记录完整实验配置时应同时保存权重版本、数据协议和生成参数，使后续比较有明确前提。

本章默认使用 NumPy 小目标与合成数据，目的是把概率、梯度、掩码和形状算清。真实模型后训练还需要完整网络反向、优化器状态、数据加载、分布式与评估。拥有 H100 并不要求跳过小例子直接运行长训练；小例子能发现方向与索引错误，真实硬件阶段再验证规模与性能。

## 10. 采样的完整执行顺序与可观察候选集合

首先得到一维 logits，每个位置对应一个词表 token。温度大于零时缩放 logits，排序，按 top-k 截断，再对剩余分数 softmax。随后计算累计概率，找到第一次达到 top-p 的位置，保留包含该位置的前缀，再归一化并抽样。这是本课程 sample_token 的明确顺序，不应假定所有模型服务采用相同组合协议。

为什么 top-k 后要重新归一化再做 top-p？因为本实现把 top-p 解释为剩余候选分布中的质量阈值。若先在完整分布做 top-p，再与 top-k 取交，边界可能不同。两种规则都可以被实现，但实验必须固定一种，否则“同样参数”可能仍代表不同候选集合。

候选集合测试比大量随机采样更直接。若某 token 被过滤，概率应为零；如果它被保留但概率很低，有限次采样没出现不能证明过滤正确。先确定排序、阈值和重新归一化，再用固定 RNG 检查调用可复现。频率实验可作为统计扩展，但不能替代确定性集合验收。

```python
import numpy as np

probabilities = np.array([0.6, 0.25, 0.15])
order = np.argsort(-probabilities, kind="stable")
cumulative = np.cumsum(probabilities[order])
count = np.searchsorted(cumulative, 0.8, side="left") + 1
kept = order[:count]
filtered = probabilities[kept] / probabilities[kept].sum()
np.testing.assert_array_equal(kept, [0, 1])
np.testing.assert_allclose(filtered, [0.6 / 0.85, 0.25 / 0.85])
```

稳定排序为分数相同候选提供确定的相对次序，特别是在 top-k 截断边界出现平分时。这个次序是实现合同之一，不是数学 softmax 唯一规定。temperature=0 的特殊分支直接 argmax，不执行除零；它仍可能需要明确平分处理。采样函数的有限输入检查也应与业务入口验证配合，不能给 NaN 温度或空词表后期待有意义输出。

生成停止条件还包括 EOS、最大新 token 数和业务停止字符串。采样本身只选择下一 token，不负责让整个生成循环有界。一个可靠演示应限制步数，记录是否正常 EOS 或因上限截断。把被截断回答拿去计算偏好或奖励时，必须保留截断信息，因为它改变回答完整性。

## 11. SFT 的移位标签：每个 logit 究竟预测谁

自回归模型在当前位置看到前缀，预测下一个 token。若完整序列为 BOS、USER、A、SEP、ASSISTANT、B、EOS，训练输入通常去掉最后一个，标签去掉第一个。于是输入位置 ASSISTANT 对应目标 B，输入位置 B 对应目标 EOS。标签与输入必须错开一位，不能把模型已经看到的当前 token 当作预测目标。

assistant-only 掩码应针对目标位置，而不是根据输入位置名字随意截取。这个例子监督 B 和 EOS，对应移位标签的最后两个位置。USER 与 A 仍进入前向上下文，影响后续回答。若只把 user token 从损失中排除，却意外从 attention 中也排除，模型就无法根据问题回答，训练目标已经改变。

```python
import numpy as np
from labs.models.core import cross_entropy

BOS, USER, A, SEP, ASSISTANT, B, EOS = range(7)
sequence = np.array([BOS, USER, A, SEP, ASSISTANT, B, EOS])
inputs, targets = sequence[:-1], sequence[1:]
active = np.array([False, False, False, False, True, True])
logits = np.zeros((len(inputs), 7))
loss, selected_gradient = cross_entropy(logits[active], targets[active])
full_gradient = np.zeros_like(logits)
full_gradient[active] = selected_gradient
assert targets[4] == B and targets[5] == EOS
assert np.all(full_gradient[:4] == 0)
np.testing.assert_allclose(loss, np.log(7))
```

这是输出 logits 层的梯度分布。后面两个位置依赖前面输入时，梯度会沿 attention、投影与 embedding 传回更早位置。不能把 full_gradient 前几行零，解释为前面的参数或输入 embedding 都被冻结。冻结参数、屏蔽输出损失和禁止注意力连接是三种不同操作。

平均分母是 active 数量。若两个有效位置损失分别二和四，平均为三；把四个被 mask 位置也算进分母会得到一，使梯度随 prompt 长度变化。对话越长、用户内容越多，这种错误越可能造成训练尺度不稳定。全 mask 样本还需要明确拒绝或跳过，不能对空集合取平均产生 NaN。

## 12. 对话模板、EOS、packing 与数据泄漏

同一段自然语言可以被不同模板编码成不同 token 序列。角色标记、轮次分隔、图像占位符和 EOS 的位置都影响监督。训练时使用一种模板，推理时使用另一种，模型可能把格式当成新分布。保存训练数据时，既要保留原始结构，也要固定实际 tokenizer 与模板版本，不能只保存最终可读文本后猜分隔规则。

多轮对话可以监督所有 assistant 回合，也可以只监督最后一轮，二者样本权重不同。工具调用消息是否监督、工具返回是否作为上下文、系统提示是否计入 loss，都需要明确。若模型需要学习输出结构化工具调用，训练目标应保留正确 schema 与错误示例策略，而不是把所有消息拼成一段没有角色边界的字符串。

packing 把多条短样本放进一个长度块，提高利用率，但必须决定跨样本 attention 是否允许以及标签如何断开。若独立样本之间不应该共享上下文，就需要相应边界或块状 mask。仅插一个 EOS 不一定等于数学上完全阻断注意力，具体行为取决于训练设计。把多个问题答案混在一起还可能导致不想要的跨样本信息泄漏。

训练集与验证集应按真实依赖单位划分。相同题目的改写、同一文档切片或重复回答若分散到两边，验证成绩可能被记忆污染。去重不只是逐字文本相同，还包括数据来源与任务结构。小项目可以先用明确不同输入组合建立留出集，并记录划分规则，再逐步学习更复杂去重。

样本权重同样重要。按 token 平均让长回答贡献更多位置，按序列平均使每条回答权重更接近。哪一种符合目标取决于任务，不应由 padding 代码偶然决定。评估时也要区分格式准确率、任务正确率、长度与拒绝行为，不能只用训练 loss 下降代表全部能力改善。

## 13. 序列 log probability 的逐 token 计算

DPO 输入不是模型随口给出的“置信度”，而是同一 prompt 条件下回答 token 的条件对数概率和。每个回答 token 的概率取自预测该 token 的 logits 行，经 log-softmax 得到，再按真实 token id 选择。prompt 部分通常不进入回答分数，但提供条件上下文。EOS 是否计入必须固定，chosen 与 rejected 使用相同约定。

```python
import numpy as np

def sequence_logprob(logits, targets, active):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    active = np.asarray(active, dtype=bool)
    if logits.ndim != 2 or targets.shape != (len(logits),) or active.shape != targets.shape:
        raise ValueError("shape mismatch")
    if not active.any() or np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("invalid targets or empty answer")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    selected = log_probs[np.arange(len(targets)), targets]
    return float(selected[active].sum())

logits = np.zeros((4, 3))
assert abs(sequence_logprob(logits, [0, 1, 2, 1], [False, False, True, True]) + 2 * np.log(3)) < 1e-12
```

均匀三类时每个目标对数概率为负 ln3，两个回答位置之和为负二 ln3。若你改为平均，对不同长度回答的相对分数会改变，因此它是目标变体，不是无害优化。长度偏好、EOS 与截断需要在数据和目标层面共同检查，不能只在结果异常时临时除以长度。

policy 与 reference 必须在同一 token 序列与 mask 上打分。不同 tokenizer、不同模板或不同截断长度，会让相减的量不再对应相同事件。缓存 reference logprob 能节省重复计算，但缓存键应绑定数据版本、模型版本和分词模板；否则你可能用新回答减旧参考分数而没有形状报错。

## 14. DPO 梯度方向与一个真正更新的小项目

把 policy 的 chosen 与 rejected 分数差记为 d，把 reference 差记为 d_ref，margin 为 beta 乘二者差。损失是 softplus(-margin)，随着 margin 增大下降。对 margin 的导数为负 sigmoid(-margin)，再乘 chosen 或 rejected 对 margin 的符号，就得到两者相反梯度。参考模型固定，因此不更新它的参数。

在 margin 为零时，sigmoid 为二分之一，chosen 导数为负 beta 除二，rejected 为正 beta 除二。梯度下降让 chosen 分数上升、rejected 下降，但实际网络参数共享，更新会同时影响其他 token。只在标量层观察方向不能保证所有生成质量提升，所以还需要独立数据评估与完整模型训练。

下面用只有两个可选回答的分类策略完成有限更新。它不是语言模型训练，但确实包含固定参考、可训练 policy、目标、解析梯度和参数更新，避免只打印一个损失值就宣称训练完成。

```python
import numpy as np
from labs.models.core import dpo_loss

policy_logits = np.array([0.0, 0.0])
reference_logprob = np.log(np.array([0.5, 0.5]))
beta = 0.5
history = []
for step in range(30):
    shifted = policy_logits - policy_logits.max()
    logprob = shifted - np.log(np.exp(shifted).sum())
    margin = beta * ((logprob[0] - logprob[1]) - (reference_logprob[0] - reference_logprob[1]))
    loss = dpo_loss(logprob[0], logprob[1], *reference_logprob, beta=beta)
    slope = -beta / (1 + np.exp(margin))
    gradient = np.array([slope, -slope])
    history.append(loss)
    policy_logits -= 0.2 * gradient
assert history[-1] < history[0]
assert policy_logits[0] > policy_logits[1]
np.testing.assert_allclose(reference_logprob, np.log([0.5, 0.5]))
```

为什么这里 logits 梯度可以直接用一正一负？两个回答构成同一二类 softmax，对数概率差中的共同归一化项抵消，等于两个 logits 的差。真实序列模型没有这样简单的独立参数关系，需要继续通过 token logprob 的计算图反向传播。这个特殊简化应写清楚，否则会把两类推导误用到任意网络。

beta 同时影响参考相对差的尺度与目标梯度，不是可以随意解释为温度或学习率的同义参数。调整 beta 与调整优化器学习率会产生不同目标与动态。比较实验时一次只改一个因素，记录 margin、loss、回答长度和留出偏好，而不是仅看某次训练损失更低。

## 15. 组相对优势：组是谁，标准差如何算

一组结果必须对应同一个问题或明确共享比较基线。把不同难度问题的奖励混到一组标准化，可能使简单题与难题相互影响，改变学习信号。奖励为零、一、二时，均值一，总体方差为三项平方偏差之和除三，即三分之二；总体标准差开方。课程 group_advantages 使用总体标准差并加 epsilon，具体训练实现可能采用其他约定。

减均值让优势和接近零，表达组内相对好坏；除标准差调整尺度。给所有奖励加相同常数，中心化后不变；乘正数后，若 epsilon 不可忽略，就不是严格尺度不变，只是常见数值范围内近似不变。全部奖励相同时，中心化为零，没有相对区分信号，不能凭标准化创造学习信息。

奖励设计决定优化方向。格式正确但答案错误是否得分，超时如何处理，工具调用失败是否算模型错误，部分正确是否有连续分数，都要写成可运行协议。可验证奖励并不保证奖励函数没有漏洞；模型可能找到满足测试却偏离真实任务的行为。奖励单元测试应包含投机反例，不只包含理想正确答案。

在线采样意味着训练数据来自当前或近当前策略产生的结果，旧策略用于记录采样概率，reference 用于约束或比较，二者职责不同。不要把 old policy 与 reference 混为同一变量后随意覆盖。训练迭代中何时更新 old、何时冻结 reference，是算法合同的一部分。

## 16. 裁剪目标的四个区域逐一手算

ratio 是新策略概率相对采样时旧策略概率的比值，等于 logprob 差的指数。ratio 大于一表示该动作在新策略下更可能，小于一表示更不可能。乘优势后，正优势鼓励增加概率，负优势鼓励减少。min 与 clip 的组合限制特定方向的过度改善，不是把所有越界区域都设成零梯度。

正优势为一时，ratio 零点五：raw 为零点五，clipped 为零点八，min 选 raw，梯度仍存在，因为新策略正在降低好动作概率，需要纠正。ratio 一点五时，raw 一点五，clipped 一点二，选 clipped，继续提高好动作的局部收益被截住。负优势则方向反过来。

负优势为负一时，ratio 零点五：raw 负零点五，clipped 负零点八，min 选后者，该区间局部梯度零；ratio 一点五时，raw 负一点五更小，选择 raw，梯度不为零，仍推动降低坏动作概率。先代数算四格，再读 np.where 条件，才不会把符号逻辑记反。

```python
import numpy as np
from labs.models.advanced import clipped_policy_objective
ratios = np.array([0.5, 1.5, 0.5, 1.5])
advantages = np.array([1., 1., -1., -1.])
loss, gradient = clipped_policy_objective(np.log(ratios), np.zeros(4), advantages)
np.testing.assert_allclose(gradient, [-0.125, 0., 0., 0.375])
assert np.isfinite(loss)
```

梯度对 logprob 求导，所以 raw 区域还要乘 ratio，再除样本总数。裁剪拐点不可导或存在分支选择约定，有限差分应避开恰好边界。课程 helper 没有完整 token mask、KL、rollout 和分布式分组，四格正确只证明这个标量目标，不证明已经实现完整 GRPO。

## 17. LoRA 的形状、初始化与梯度

设基础权重 W 为 d 乘 m，低秩维 r，A 为 d 乘 r，B 为 r 乘 m。输出为 XW 加缩放后的 XAB，缩放采用 alpha/r。可训练参数从 d 乘 m 转为 d 乘 r 加 r 乘 m，但基础 W 仍需存储并参与前向。可训练参数减少不等于整套模型内存按同一比例下降，因为激活、优化器和基础权重有不同成本。

常见初始化让一个低秩因子为随机值、另一个为零，使初始增量为零。若 A 与 B 都为零，二者初始梯度都可能为零，无法开始学习；若 B 为零而 A 非零，B 可以先获得梯度，A 的第一步梯度可能为零，之后随着 B 非零而参与更新。这个现象可由矩阵链式法则直接推导。

设上游梯度 G 与输出同形，缩放为 s。dB 为 s 乘 A.T、X.T 与 G 的合适乘积，等价于 s*(X@A).T@G；dA 为 s*X.T@G@B.T。基础 W 冻结只意味着不更新 W，不意味着前向不计算 XW，也不意味着梯度不需要经过 X。理解冻结与计算图路径是 LoRA 实现的重要细节。

```python
import numpy as np
from labs.models.core import lora
X = np.array([[1., 2.]])
W = np.eye(2)
A = np.array([[1.], [0.]])
B = np.array([[2., 3.]])
alpha = 2.0
merged = W + (alpha / A.shape[1]) * (A @ B)
np.testing.assert_allclose(lora(X, W, A, B, alpha), X @ merged)
```

数学合并等价要求相同表示与缩放。量化权重、重新量化、不同累加精度或 adapter 切换实现可能引入差异，不能把这个 float64 例子扩张成所有部署逐位一致。多个 adapter 如何组合、目标模块选哪些、是否训练 bias，都需要独立配置与验收。

## 18. 图像 patch 到视觉 token：索引比名称重要

图像数组 H 乘 W 乘 C，patch 大小 P，要求本例 H 与 W 可整除 P。patchify 先拆成纵向块、块内行、横向块、块内列、通道五个轴，再转置让两个块索引相邻，最后把每块内部展平。若直接把整张图 reshape 成若干行，可能把连续整行当成 patch，形状正确但空间邻接错误。

```python
import numpy as np
from labs.models.core import patchify
image = np.arange(16).reshape(4, 4, 1)
patches = patchify(image, 2)
np.testing.assert_array_equal(patches, [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
assert patches.shape == (4, 4)
```

RGB patch 每个像素有三个通道，展平顺序还涉及通道排列。输入是 HWC 还是 CHW、颜色是 RGB 还是 BGR、数值范围是零到二百五十五还是归一化浮点，都影响模型实际看到的内容。视觉预处理不是任意把图片读成数组就结束，必须与训练时 processor 合同一致。

视觉投影把每个 patch 的 P²C 个数映射到模型维度 D，但随机投影没有学到猫狗语义。真实视觉 encoder 还通过训练、位置表示和多层变换提取结构。cross_modal_read 让文本 query 读视觉 token，只证明跨模态数据路径存在；输出随图像变化不等于模型理解图像。

拼接融合与 cross-attention 的状态成本不同。把视觉 token 放进统一 decoder 前缀，会增加上下文长度和相应缓存；单独视觉编码再交叉读取，需要维护视觉状态与跨注意力组件。视频还引入时间采样和帧顺序，音频需要采样率与帧表示。一个附件不等于一个 token，成本必须按实际模型与服务规则核对。

## 19. 十二道逐级练习与完整参考答案

### 一：greedy 与 top-k 为一

```python
import numpy as np
from labs.models.core import sample_token
logits = np.array([1., 3., 2.])
rng = np.random.default_rng(7)
assert sample_token(logits, rng, temperature=0) == 1
assert all(sample_token(logits, rng, top_k=1) == 1 for _ in range(20))
```

两种设置在这个唯一最大值样例中都选同一 token，但执行路径不同。不要用这个巧合推断任意服务中所有 greedy 与过滤选项组合都完全等价。

### 二：相同随机源与分布可重放

```python
import numpy as np
from labs.models.core import sample_token
left, right = np.random.default_rng(9), np.random.default_rng(9)
a = [sample_token([2., 1., 0.], left) for _ in range(20)]
b = [sample_token([2., 1., 0.], right) for _ in range(20)]
assert a == b
```

这里固定了本地实现、调用顺序与分布。改变 logits、插入额外随机调用或更换远端服务，都不在这份重放保证内。

### 三：监督位置的平均分母

```python
import numpy as np
from labs.models.core import cross_entropy
logits = np.zeros((4, 3))
targets = np.array([0, 1, 2, 1])
active = np.array([False, False, True, True])
loss, gradient = cross_entropy(logits[active], targets[active])
np.testing.assert_allclose(loss, np.log(3))
np.testing.assert_allclose(gradient[0], [1/6, 1/6, -1/3])
```

梯度除以两个有效位置，不是全部四个位置。若你得到一半幅度，优先检查分母与 mask，而不是怀疑 softmax 求导。

### 四：全 mask 应由适配层拒绝

```python
import numpy as np

def require_supervision(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or not mask.any():
        raise ValueError("at least one supervised position required")
    return mask
try:
    require_supervision([False, False])
except ValueError:
    pass
else:
    raise AssertionError("empty supervision must fail")
```

这个检查放在选择有效 logits 之前，避免空均值。数据管道也可以选择显式跳过，但必须记录跳过数量，不能静默改变有效批次。

### 五：DPO 的零 margin 与有限差分

```python
import numpy as np
from labs.models.core import dpo_loss
beta, h = 0.1, 1e-5
assert abs(dpo_loss(-3, -4, -3, -4, beta) - np.log(2)) < 1e-12
plus = dpo_loss(-3 + h, -4, -3, -4, beta)
minus = dpo_loss(-3 - h, -4, -3, -4, beta)
assert abs((plus - minus) / (2 * h) + beta / 2) < 1e-8
```

差分只改变 policy chosen，reference 保持不变。若同时改变两边相同分数，margin 可能不变，验证的就不是目标偏导。

### 六：reference 不能随 policy 一起更新

```python
from labs.models.core import dpo_loss
initial = dpo_loss(-3, -4, -3, -4)
improved = dpo_loss(-2, -5, -3, -4)
accidentally_moved_reference = dpo_loss(-2, -5, -2, -5)
assert improved < initial
assert abs(accidentally_moved_reference - initial) < 1e-12
```

如果每次把 reference 也改成当前 policy，偏好差相对参考又回到零。这个反例解释冻结参考的作用，而不是只把 freeze 当框架参数记忆。

### 七：奖励平移与同奖励组

```python
import numpy as np
from labs.models.advanced import group_advantages
np.testing.assert_allclose(group_advantages([0., 1., 2.]), group_advantages([10., 11., 12.]), atol=1e-12)
np.testing.assert_array_equal(group_advantages([5., 5., 5.]), [0., 0., 0.])
```

同奖励组没有相对排序信号。继续增加训练步数不能凭空从全零优势中学到偏好，需要检查采样、奖励和题目难度。

### 八：未裁剪点的梯度

```python
import numpy as np
from labs.models.advanced import clipped_policy_objective
advantages = np.array([-1., 0., 1.])
loss, gradient = clipped_policy_objective(np.zeros(3), np.zeros(3), advantages)
np.testing.assert_allclose(gradient, -advantages / 3)
assert abs(loss) < 1e-12
```

平均优势为零使当前标量 loss 为零，但梯度并非全零。只看损失数值不能判断是否存在学习信号，这个例子很适合检查训练日志解读。

### 九：LoRA 双零初始化的停滞

```python
import numpy as np
X = np.array([[1., 2.]])
G = np.array([[3., 4.]])
A = np.zeros((2, 1))
B = np.zeros((1, 2))
dA = X.T @ G @ B.T
dB = (X @ A).T @ G
assert np.all(dA == 0) and np.all(dB == 0)
A[0, 0] = 1.0
dB = (X @ A).T @ G
assert np.any(dB != 0)
```

一边非零可以给另一边提供首步梯度。这里没有调用训练框架，矩阵公式已经足以解释初始化规则。

### 十：patch 不应按整行切开

```python
import numpy as np
from labs.models.core import patchify
image = np.arange(16).reshape(4, 4, 1)
patches = patchify(image, 2)
np.testing.assert_array_equal(patches[0], [0, 1, 4, 5])
np.testing.assert_array_equal(patches[1], [2, 3, 6, 7])
assert not np.array_equal(patches, image.reshape(4, 4))
```

形状都为四乘四，但空间关系不同。只测 shape 会漏掉这种错误，位置编码样例比全零或全一图片更有辨识力。

### 十一：跨模态数据通路确实使用视觉值

```python
import numpy as np
from labs.models.advanced import cross_modal_read
queries = np.eye(2)
patches = np.array([[1., 0.], [0., 1.]])
projection = np.eye(2)
first = cross_modal_read(queries, patches, projection)
second = cross_modal_read(queries, patches + 1., projection)
assert first.shape == second.shape == (2, 2)
assert not np.allclose(first, second)
```

输出随图像值改变证明通路没有完全忽略视觉输入，但不证明语义识别能力。还应在真实训练后使用有标签任务评估，不能用这条断言替代视觉能力测试。

### 十二：低秩参数量与基础参数量分开统计

```python
d, m, rank = 6, 4, 2
base_parameters = d * m
adapter_parameters = d * rank + rank * m
assert base_parameters == 24
assert adapter_parameters == 20
assert base_parameters + adapter_parameters == 44
```

这个小尺寸的低秩增量并没有显著省参数，说明 rank 必须相对输入输出维度足够小才有意义。实际显存还包括激活与优化器，不能把参数计数直接当峰值内存。

## 20. 四个故障现场与最小复现

**SFT loss 很低但回答像复述问题。** 检查标签是否错误对齐到当前输入 token，是否把 prompt 也作为主要监督，assistant mask 是否针对移位后的目标。用七 token 小序列打印 input、target、active 三列，逐位置核对。不要先增加训练步数，因为目标错位可能让模型越来越擅长错误任务。

**DPO loss 下降但 chosen 概率也下降。** DPO 主要约束相对偏好差，不保证所有单独分数都单调上升。检查 chosen/rejected 相对 reference 的 margin、回答长度、数据质量和共享参数影响。只看 loss 或只看 chosen 一个数都不够；用固定小样例与留出集区分目标机制和泛化效果。

**组优势全为零。** 检查同组奖励是否完全相同，奖励函数是否总返回默认值，采样是否只生成同一回答，或者归一化分组是否错误。先直接打印受控三奖励样例，再接真实奖励器。把 epsilon 调小不能创造不存在的组内差异，也不能修复奖励协议错误。

**视觉输入变化但模型输出完全不变。** 检查 patch 顺序、投影权重是否全零、视觉 mask 是否全部屏蔽、拼接位置是否被截断、实际输入是否仍指向旧缓存。先用两维非对称图块观察中间 visual tokens，再看 cross-attention 输出。不要仅凭最终生成文本相同就断言 encoder 没变化，因为解码可能把不同分布映射到同一最大 token。

## 21. 一个可完成的后训练实验设计

先用两类回答策略完成前面的 DPO 更新，验证损失、margin 和固定 reference。第二阶段构造多个 prompt 的独立小参数向量，划分训练与留出 prompt，明确这种参数表没有跨 prompt 泛化能力。它能训练目标与评估流程，却不能被包装成语言理解模型。第三阶段才接 M15 的可训练模型，沿已有反向接口把序列 logprob 梯度传回参数。

每次实验保存数据版本、模板、tokenizer、mask、reference、beta、优化器、学习率、随机源和停止条件。结果至少包括训练目标、留出偏好准确率、回答长度与格式错误比例。若使用真实模型 API 产生候选，另记调用模型与时间，因为远端模型可能更新；API 生成数据与本地训练是两个阶段，成本与隐私边界分别控制。

奖励训练项目先用确定可验证的小任务，如两位数加法或有限状态操作序列。奖励器要测试正确答案、错误答案、格式投机、重复答案与超长输出。把最终答案验证与过程偏好分开，避免一个模糊总分掩盖多个目标。只有奖励器通过反例测试，才值得扩大 rollout 数量。

多模态项目先用合成图块任务：输入只有几种颜色或位置模式，标签由确定规则生成。验证训练集与测试集的组合差异，确保模型没有只记固定文件名。先冻结随机投影时可能学不到所需信息，再逐步训练 projector 或 encoder，观察哪一层改变。这个实验能建立表示学习直觉，无需一开始下载大型视觉语言权重。

## 22. 训练批次、梯度累积与资源边界

显存受参数、优化器状态、梯度与激活共同影响。LoRA 减少可训练参数相关部分，但长序列激活仍可能占主要空间；偏好训练还要处理 chosen 与 rejected 以及 reference 打分。是否缓存参考分数、是否分批前向、是否使用梯度检查点，会改变时间与内存，不能只根据 adapter 参数量预测能否装进显卡。

梯度累积把多个微批次梯度加起来再更新，需要正确缩放损失。若每个微批次有效 token 数不同，简单平均每个微批次平均损失再平均，可能不等于整个大批次按 token 平均。应该先定义目标权重，再按有效计数归一化。这个问题与 SFT mask 分母直接相连，不能当作纯框架开关。

用户的个人 PC 可用于小模型与 CPU 机制实验；需要真实训练时可选择单张合适 GPU，先小批量确认数值和显存，再扩大。八卡 H100 只有在分布式、吞吐或规模问题确实需要时使用，并设置有限步数、保存点和结束清理。代码检查或标量梯度验证不需要占用多卡资源。

本章没有在 GPU 上执行 SFT、DPO 或 GRPO，也没有调用模型 API。数值目标与小策略更新的通过证据不能扩张成真实大模型后训练完成。后续硬件实验应报告实际模型、精度、序列长度、批次与设备，明确与本章 CPU 参考的关系。

## 23. 来源、版本与掌握门槛

DPO 的原始成对目标见 [DPO v3](https://arxiv.org/abs/2305.18290v3)，组相对优化背景见 [DeepSeekMath v3](https://arxiv.org/abs/2402.03300v3)，低秩增量参数化见 [LoRA v2](https://arxiv.org/abs/2106.09685v2)，图像 patch 编码背景见 [ViT v2](https://arxiv.org/abs/2010.11929v2)。本轮核查日期为 2026-11-01；正文数值例子与训练脚本为独立教学构造，不复现论文性能结论。

通过本章时，你应能在纸上对齐 input/target/mask，手算一个序列分数、一个 DPO margin、一个组优势和四个裁剪区域，解释 LoRA 第一轮梯度，写出图像 patch 的确切顺序。随后能把这些机制分别放进可测试数据流，知道哪些只改变解码、哪些改变目标、哪些改变参数、哪些改变输入表示。

最终验收不接受只运行七个 lab 然后记录输出。你需要主动制造 mask 错位、参考模型被更新、裁剪方向颠倒和 patch 顺序错误中的至少两种，展示失败测试与修复。能定位这些错误后，再去读真实后训练框架的 rollout、loss 和数据 collator，才有能力判断代码实际优化了什么。

### 补充推导：偏好目标与策略约束为何同时需要

仅最大化奖励可能让策略远离原有语言分布，出现格式投机、重复或其他不希望行为。参考策略约束用于控制这种偏离，但约束强度、奖励尺度与优化步幅共同影响结果。DPO 把参考相对对数概率差写进成对目标，并不意味着可以忽略数据质量、策略支持范围或训练稳定性。任何目标都只优化你明确写入的信号。

KL 散度在有限离散分布上可写成按策略概率加权的对数概率比之和。它比较整个分布，不等于只对一个采样动作取 logprob 差就得到无噪声完整 KL。采样估计需要说明采样来自谁、采用什么估计量与归一化方式。组相对算法中的某个 KL 近似不能凭名字直接替换成任意两个分数相减。

奖励模型、old policy 与 reference 是三个不同角色。奖励模型评价结果，old policy 记录 rollout 的产生分布，reference 提供固定或按算法定义的比较基准。它们可能来源于相同初始权重，但训练中生命周期不同。将三个变量都命名为 model 会让源码阅读困难，研究真实框架时应先追踪它们的构造、冻结和更新位置。

离线偏好数据避免每个训练步都重新采样，但数据分布可能逐渐与当前策略不同。在线采样更贴近当前行为，却增加生成、奖励与环境交互成本，也可能引入非稳定数据。不能只用“在线更先进”或“离线更便宜”做选择；应明确任务需要的反馈、可验证性、预算与评估方式。

### 补充：奖励器本身也需要工程验收

一个数学奖励器如果只搜索输出里是否出现正确数字，模型可以把所有候选数字列出来获得奖励。一个代码奖励器如果测试过少，程序可能硬编码样例；一个工具奖励器如果只看返回文本含 success，错误或恶意工具输出也可能被误判。奖励协议应限定答案提取、执行环境和成功条件，并为投机路径写反例。

奖励失败与模型失败应分开。测试环境不可用、外部 API 超时或解析器崩溃，不应未经说明都当作模型回答错误。可以记录 outcome、reward、error_kind 和可重试状态，让统计区分真实失败与评价基础设施失败。否则训练会惩罚与任务能力无关的事件，评估也难以解释。

奖励归一化会改变不同组的权重。某组奖励差异极小，除以小标准差后可能被放大；全部相同时又成为零。epsilon、是否按问题分组、是否跨设备聚合，都影响实际梯度。真实分布式代码必须保证同一个逻辑组没有被错误拆开或与其他问题混合，这不是仅看单卡公式就能发现的错误。

### 补充：多模态掩码与位置关系

图像 token 放在文字前缀时，后续文字通常可以读取视觉信息，但是否允许视觉 token 读取未来回答取决于 mask。训练时若错误开放未来答案给视觉分支，再把其结果传给当前预测，就形成泄漏路径。因果性检查应覆盖所有模态间连接，而不只是文字主干的一张三角矩阵。

多张图像还需要图像边界与位置身份。四个 patch 来自第一张图还是第二张图，不能仅凭数值形状区分；模型需要相应位置或边界编码。视频增加时间轴后，帧顺序错误可能让静态内容看似正常而动作理解失败。用带明确空间与时间标号的合成输入，比随机噪声更容易验证排列。

裁剪、缩放和分辨率决定视觉 token 数量。固定 patch 大小时，宽高同时翻倍可能使 patch 数量约四倍；若后续全注意力处理全部视觉 token，某些计算项增长更快。真实系统可能使用切片、压缩或重采样控制长度，因此成本估算必须跟随实际 processor 和架构，不能把原始像素数直接等同最终上下文 token 数。

### 补充：把学习曲线与能力结论分开

训练 loss 下降说明模型更符合当前优化目标，不自动说明新任务泛化、事实正确性或 Agent 工具可靠性提升。偏好准确率也可能因长度偏好、格式捷径或数据重复而提高。评估应包含独立样本、不同长度、反例与任务实际成功条件，并记录生成协议，避免把解码变化当成参数学习。

对小项目，最有价值的结果不是追求高分，而是能解释一次失败：是标签错位、梯度方向、奖励漏洞、数据泄漏还是表示通路断开。把失败缩成最小输入并保留回归测试，下一次换更大的模型或框架时仍能复用。这个习惯使你从“会调训练参数”走向“知道训练代码正在计算什么”。

最后给训练数据也建立与模型相同严格的版本身份。文件名相同并不说明内容相同：清洗规则变化、模板加了一个结束符、过滤掉一条异常样本，都会改变实际优化的问题。一个可复查的检查点至少关联原始样本集合的摘要、预处理程序版本、模板版本、划分清单和训练参数。保存这些信息的目的不是装饰报告，而是让你能回答“这次变好究竟来自算法，还是因为测试题进入训练集”。恢复训练时也检查优化器步数、随机数状态和数据位置；仅恢复权重可以继续学习，却未必延续原来那条训练轨迹。课程中的小数组实验不需要大型数据平台，但应从第一天保存这条最小证据链。当实验结果反常时，先比较实际输入和标签，再比较损失实现，最后讨论更复杂的优化理论，这通常比盲目扩大训练规模更有效。
