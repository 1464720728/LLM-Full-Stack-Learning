# M16　综合项目二：架构比较台与真实开源模型源码研究

[本卷导航](README.md) · 前置：本卷M01–M15 · 深入工作簿：[逐行追踪与增量实现](WORKBOOK.md)

你已经见过很多模型名字。最后一章要把“知道名词”变成能完成的研究工作：明确比较问题、固定证据、建立oracle、测量状态和运算量、找到真实源码、写可验证修改。比较台不输出排行榜，也不把随机权重toy的结果包装成能力评测。

## 1. 每次只回答一个比较问题

“哪个架构更好”过于含糊。可以改成：同一数学函数的两个实现是否等价？在固定维度下状态随T怎样增长？稀疏策略减少了多少允许连接？某种压缩对固定任务的质量影响是什么？给定硬件与batch时哪个实现更快？这些问题需要不同实验。

本项目首先回答四个等价问题：稀疏gather与同mask稠密、kernel linear并行与递推、固定线性SSM递推与卷积、同一decoder完整前向与KVstep。它不会要求softmax与linearattention相同，因为它们本来就是不同函数。

运行：

```bash
python -m labs.models.compare
python -m unittest discover -s tests -p 'test_models*.py' -v
```

输出包含四个最大误差、连接计数和状态元素数。数值误差应很小，连接与元素数可手算。这里没有tokens/s或H100加速比，避免把未做的性能测量混进报告。

## 2. 为什么需要独立参考实现

如果优化版和reference都调用同一个错误helper，二者一致不能证明正确。多层验证组合更强：手算极小例子验证公式、独立朴素实现验证索引、随机property tests验证广泛形状、真实框架对照验证接口约定。

以attention为例，reference可以用双循环算score，优化版用matmul；两者共享稳定softmax还不够完全独立，所以再验证行和、mask零值、平移不变、V缩放等不变量。多个不同角度互相补强。

测试中的dtype也要显式。float64CPU利于排查，BF16GPU利于真实部署；二者验证目标不同。不能因GPU不能达到float64的`1e-12`阈值就判错，也不能把大误差一律推给低精度。

## 3. 参数、状态、计算、质量四张表

参数表记录各模块权重数量与dtype。状态表记录每请求/每层的KV、线性state、卷积history、位置计数及scale。计算表记录每token或prefill块的主要矩阵乘和连接数。质量表记录任务、数据划分、指标与采样协议。

一个模型总参数多，但每token激活少；状态可能小，但特殊kernel要求高；训练质量好，但某一服务框架尚无优化实现。这些都说明单个数字不足以排序。

以本比较台T=24、head_dim=8为例，标准单头KV存384个数；kernel linear状态72个数。这个差异仅涉及本定义的state，不包括投影、层数与其他临时数据。你应把说明写进结果旁边，而不是藏在报告末尾。

## 4. 开源项目的分层阅读路线

第一站可以读micrograd，目标是把本卷Value与一个成熟的小型autograd实现逐项比较；它是教学桥梁，不作为企业生产框架推荐。下一站读nanoGPT的model与training入口，重点看真实工程如何处理参数初始化、权重共享、优化器分组、checkpoint与训练循环；它同样不是大型生产服务框架的替代。

之后进入Hugging Face Transformers对应模型目录。先读configuration定义，再读modeling中的attention、block、cache接口，最后看generation与processor。不要先从数千行通用generate入口一路迷失；带着一个具体问题，例如“这个模型的KV头数在哪减少”，更容易建立调用链。

对于Mamba，先读官方reference scan与小模块，再研究CUDA/Triton路径。对于AttnRes，先读官方README与reference/pseudocode，区分Full/Block；对于旗舰模型，先下载config和少量源码，不需要下载几百GB权重才开始学习。

这些项目都作为扩展任务使用，来源表给出核实链接。项目是否适合你，不看stars一个指标，而看它是否拥有你要学习的真实组件、明确输入输出和可运行验证路径。

## 5. 固定版本的最小工作流

首次研究一个仓库，记录完整URL、访问日、实际commit SHA、文件路径和本地环境。分支main/master会移动，访问日不等于内容永久固定。不要把本书阅读时链接指向的分支当成已锁版本。

你可以在自己的研究目录执行：

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
git rev-parse HEAD
git status --short
```

上面是明确的外部下载拓展任务，不在本卷离线测试中自动执行。运行后把实际SHA记入研究记录，再基于该commit定位符号。若源码布局已经变化，用`rg`找类/函数，先确认语义，禁止为了匹配书里的行号而强行修改代码。

我们本卷来源记录按2026-11-01核对，未取得稳定SHA的项目明确标为移动链接，不编造commit。后续课程更新应保留原证据与变更理由；用户学习中的修改另开分支，便于比较前后行为。

## 6. 最新模型研究的具体产物

以官方模型卡/配置为入口，交付以下内容：一页结构图；每个子层的输入输出形状；状态张量表；关键前向调用链；至少一个小维度reference实现；与已有实现的差异；三类测试；未核实项。

例如研究MiMo混合attention，应该找出SWA与GA层调度、两类Q/KV头数、window定义和state容量，而不是因它叫“混合”就添加SSM章节。研究DeepSeek-V4.1，则要区分CED产生的共享状态、压缩/索引复用与token条件记忆，不把所有东西塞进传统层独立KV公式。研究Kimi-K3，区分KDA序列递推与AttnRes深度聚合。

闭源模型若没有公开具体结构，可以研究公开API行为、能力评测与系统论文，但架构表中必须写未知。性能或回答风格不是反向证明内部用了某种专家/注意力结构的可靠证据。

## 7. 实验报告模板

每次提交报告至少包含：问题与假设；输入/参数/seed/dtype；参考与待测实现；正确性指标与阈值依据；状态/计算计数；实际硬件和软件版本；观测结果；失败案例；结论适用范围。

若做性能测试，还要加入预热、同步、测量次数、分位数、是否包含编译/数据传输。若做模型质量，还要加入样本量、prompt/模板、生成参数、重复次数与统计不确定性。把两种报告混合但不写协议，通常无法得出可靠结论。

## 8. 毕业任务与参考路线

1. **扩展比较台。** 加入分块linear state continuation误差项与非零初态SSM项。参考：后者必须补初态响应，不可直接拿零初态卷积作oracle。
2. **增加一个可控故障。** 选RoPE offset、Moe scatter、SFT mask其中一个，引入bug；先展示失败测试，再修复。验收包括失败证据和修复解释，不能只提交正确最终代码。
3. **读取真实小decoder。** 输出配置→模块→forward→cache完整调用链，至少标出五处本卷简化。参考重点：norm参数、bias、位置策略、weight tying、cache布局、mask API与dtype。
4. **实现一个结构变体。** 将TinyDecoder的FFN换成MoE，先不训练，只要求shape、因果性和full/step一致。之后可在PyTorch项目中训练相同变体，分别报告正确性与学习结果。
5. **复现一个官方reference子算子。** 选Mamba scan或AttnRes，用同一小张量逐项对照。验收输入契约、输出与state、误差和边界情况；不接受只跑官方demo后截图。
6. **连接推理框架卷。** 找服务端model runner怎样调用你研究的model forward，怎样传位置和cache。写出一次请求从token到某层算子的路径；区分scheduler决策与模型数学。
7. **连接算子卷。** 将自己的RMSNorm或attention reference作为oracle，实现一个Triton/CUDA版，先正确再优化。必须使用kernel卷的设备/依赖检查和有限GPU预算，不把公司的空闲卡当无限benchmark资源。
8. **设计质量实验。** 选择长度泛化或关联回忆toy，比较全attention、local与linear。固定参数预算或明确不固定，报告训练协议。不要用随机权重比较“理解能力”；先让每种模型真正训练，再讨论表现。

本卷最终验收不是记住所有架构，而是能读一个新结构、写出它的关键计算、设计正确性实验、定位实现差异，并知道还缺哪些证据才能宣布它适用于真实部署。

## 9. 第一轮研究课：把大问题改写成可证伪的命题

先把“比较”拆成两个动作：规定哪些东西保持相同，再规定观察哪一个差异。假如你把稠密注意力换成线性注意力，又把维度减半、训练步数翻倍、数据集换掉，最后损失降低，你只知道整套方案变化后得到了不同结果，不知道是哪一处变化造成的。研究不是拒绝同时修改多项，而是要给结论分配相应的范围。产品评估可以比较两个完整方案；机制研究常常需要把其余条件固定。二者都合理，不能拿一种实验回答另一种问题。

本章建议给每项工作写一句有失败条件的话。例如“对于任意非空、每行至少有一个可见键的布尔掩码，逐行收集实现与同掩码稠密实现，在本次有限形状和双精度输入上误差不超过指定阈值”。句子里的限定并不多余：非空限制避免没有定义的归一化；同掩码限制保证数学任务相同；有限形状限制诚实说明测试不是形式证明；双精度说明误差背景。如果测试发现一行全被屏蔽，首先补上输入契约，而不是突然把异常输出当成优化失败。

另一个命题是“线性注意力把长度为九的序列分为四个位置和五个位置两段计算，并传递第一段状态，其输出与一次处理九个位置一致”。这个命题不仅比较输出，还检验状态能否承载前缀信息。只检查最后一个输出可能漏掉中间位置错误；只检查输出不检查传入状态是否被原地修改，又可能漏掉会破坏多分支解码的别名问题。因此同一个命题可以拆为数值等价、输入状态不变、输出状态尺寸正确三项验收。

把“为什么失败”也提前写下来。数值不等可能来自不同公式、坐标布局、初始状态、位置偏移、归一化分母、精度或真的实现错误。只有排除前面这些合理差别，才有资格把问题归因于某段代码。将两个都叫attention的函数放进`allclose`并不构成研究设计；你必须先写清它们是否在计算同一个函数。

一个研究目录至少保存问题说明、输入生成方式、运行脚本和结果。原始结果与解释分开：结果文件里保留实际数组形状、元素数、最大误差和环境；说明文件里解释这些数字支持什么。以后你的理解变化，可以修改解释而不改写历史数据。相反，若把结果直接手抄进文章而没有对应运行入口，连一次抄错小数点都很难追溯。

这里的研究对象是你自己的小实现。你完全可以先不安装外部项目，先证明自己理解状态和轴，再映射真实源码。这样阅读外部函数时，每个变量都有已经算过的对应物。等到你确实需要验证库的具体行为，再在隔离环境中安装固定版本，记录这一步新增了什么证据。

## 10. 独立参考课：从三行手算到双循环oracle

参考答案通常叫reference或oracle。它的职责是清楚地表达你相信的定义，优先级是可检查，其次才是速度。如果待测实现和参考都由相同的矩阵乘、相同的掩码辅助函数组成，它们可能一起继承同一个错误。独立性不是要求完全不用共同库，而是让关键索引和公式经过另一条推导路径。

先手算一个没有缩放歧义的标量例子。取三个查询和三个键都为零，值依次为一、三、八，使用因果可见性。任意可见点积都是零，所以每行在可见位置上均匀分配概率。第一行只见第一个值，输出一；第二行见一和三，输出二；第三行见全部，输出四。这个例子同时检查了因果方向、对当前词元的包含关系和归一化范围。如果输出四、五点五、八，掩码方向可能反了；如果第一行未定义，可能错误地排除了当前位置。

下面写一个只支持二维有限数组、每行至少一个可见键的参考。限制是有意的：先明确一个小契约，再按需求扩充。它不复用课程的softmax，不调用课程的mask构造，不用整块`q @ k.T`，因此能独立检查核心索引。`math.fsum`尽量减少标量求和的舍入误差，但并不意味着结果具有无限精度。

```python
import math
import numpy as np
from labs.models.core import attention

def loop_attention(q, k, v, allowed):
    q, k, v = [np.asarray(x, dtype=float) for x in (q, k, v)]
    allowed = np.asarray(allowed, dtype=bool)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('二维输入必需')
    if q.shape[1] != k.shape[1] or len(k) != len(v) or q.shape[1] == 0:
        raise ValueError('特征或键值长度不匹配')
    if allowed.shape != (len(q), len(k)) or not allowed.any(axis=1).all():
        raise ValueError('掩码形状错误或存在完全不可见行')
    if not all(np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('输入必须有限')
    result = np.zeros((len(q), v.shape[1]))
    for i in range(len(q)):
        indices = [j for j in range(len(k)) if allowed[i, j]]
        scores = [math.fsum(float(q[i, d] * k[j, d])
                           for d in range(q.shape[1])) / math.sqrt(q.shape[1])
                  for j in indices]
        largest = max(scores)
        numerators = [math.exp(s - largest) for s in scores]
        denominator = math.fsum(numerators)
        for f in range(v.shape[1]):
            result[i, f] = math.fsum(
                w * float(v[j, f]) for w, j in zip(numerators, indices)
            ) / denominator
    return result

q = np.zeros((3, 1)); k = np.zeros((3, 1))
v = np.array([[1.], [3.], [8.]])
allowed = np.array([[True, False, False],
                    [True, True, False],
                    [True, True, True]])
expected = np.array([[1.], [2.], [4.]])
np.testing.assert_allclose(loop_attention(q, k, v, allowed), expected)
np.testing.assert_allclose(attention(q, k, v, allowed)[0], expected)
print(expected.ravel())
```

逐行理解循环比看最终打印更重要。外层`i`固定一个查询位置；`indices`收集这个位置能访问的键；内层`d`求一个查询和一个键的点积；分母在所有可见键之间求和；最后`f`遍历值向量的输出坐标。查询特征维和输出值维可以不同，所以代码分别用`q.shape[1]`和`v.shape[1]`。很多实现错误来自默认二者永远相等，导致方阵例子通过、长方形例子失败。

稳定softmax先减去当前行最大分数，原因是指数可能非常大；所有分数同时平移不会改变最终比值。这里最大值只从可见分数中取出。若先把被屏蔽位置设成一个普通很小的常数，再在极端输入中让真实分数更小，被屏蔽位置仍可能得到概率。布尔可见集合的手算参考避免了这种含糊，适合检查实现约定。

再加随机输入时，首先使用小整数，方便失败后检查。维度选查询长度二、键长度四、查询特征三、值特征五，不要全取同一个数。方阵会掩盖转置错误；不同维度会尽早让错误暴露。加入随机数的目的不是让实验看起来复杂，而是让人工例子没有覆盖的索引组合出现。每次失败保存具体输入，不能只保存随机种子，因为不同随机生成调用顺序会改变种子对应的实际数组。

## 11. 误差课：绝对误差、相对误差与不变量分别回答什么

设参考结果是`r`，待测结果是`x`。绝对误差是`|x-r|`，单位与输出相同；相对误差想回答相对于结果规模偏离多少，但参考接近零时直接相除会放大噪声。常用的逐元素通过条件是`|x-r| <= atol + rtol*|r|`。这里`atol`照顾零附近，`rtol`照顾大数附近，两个容忍值都应该与数据范围和精度有关。

例如参考为一百万，差零点零零一可能相对很小；参考为零，差零点零零一就不一定可接受。只写最大相对误差而不说明零分母处理，会让报告充满无穷大；只写平均误差则可能掩盖少数位置的严重错误。本章至少记录最大绝对误差、最差元素位置和相应参考值。对会返回多个张量的函数，分别记录输出和状态，不要把它们拼成一个难以解释的大平均数。

```python
import numpy as np

def error_report(actual, expected, atol=1e-10, rtol=1e-8):
    actual = np.asarray(actual); expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError('需要相同且非空的形状')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('非有限结果不能靠误差阈值掩盖')
    difference = np.abs(actual - expected)
    location = np.unravel_index(np.argmax(difference), difference.shape)
    bound = atol + rtol * np.abs(expected)
    return {'max_absolute': float(difference[location]),
            'location': tuple(int(i) for i in location),
            'expected_at_location': float(expected[location]),
            'passed': bool(np.all(difference <= bound))}

print(error_report(np.array([0., 1_000_000.001]),
                   np.array([0., 1_000_000.])))
assert error_report([1., 2.], [1., 2.])['passed']
assert not error_report([0.01], [0.], atol=1e-5)['passed']
```

这里先检查非有限值，是因为`NaN`不是一个可排序的普通实数。若代码错误地把非有限结果替换成零，然后再计算平均误差，报告反而可能变得漂亮。替换操作改变了待检验对象。除非算法明确规定了这样的清洗行为，否则应把它作为失败信号并保留最小输入。

容忍阈值也不能在看到失败后不断放宽直到通过。合理做法是先用高精度和很小输入确认公式，然后按运算长度、累积方式和目标精度建立误差基线。切到低精度后，比较误差随长度和幅度的变化，而不是把所有误差都归到“浮点数本来就不准”。如果长度翻倍时某个位置突然出现量级很大的跳变，优先怀疑索引、溢出或掩码，而不是先改变阈值。

不变量提供另一种证据。注意力每个非空可见行的权重和应为一；被遮挡位置权重应为零；所有值向量都相同，输出也应为那个向量；给每行分数加同一个常数，概率不变。这些性质不需要知道每一个随机输入的标准输出，因此能揭示两个实现共同犯下的错误。但不变量也不充分：始终输出可见值的平均数也满足若干性质，却不是一般softmax注意力。所以手算、独立参考、性质测试要互相补充。

## 12. 状态课：把“常数内存”写成真实的张量账本

比较内存前先说清计量范围。参数是模型跨请求共享的权重；状态是为了延续某个请求而保存的历史；临时张量是一次运算中出现、之后可释放的中间结果；运行时还有分配器、工作空间和其他元数据。一个方法减少了请求状态，不代表它的总显存按相同比例下降。反过来，即使参数占大头，长上下文并发时状态仍可能成为限制。

本比较台的标准单头缓存包含长度为`T`的K和V，各形状为`[T,d]`，元素数为`2*T*d`。这里假设键和值维度同为`d`。若值维不同，应写成`T*(d_k+d_v)`。元素数再乘每元素字节数才能得到理论张量字节；二字节和四字节只是存储类型假设，不能据此断言实际框架分配精确等于这个数字。

课程kernel线性注意力维护`S`和`z`。`S`是所有历史键特征与值的外积和，形状为`[d_k,d_v]`；`z`是历史键特征之和，形状为`[d_k]`。其元素数是`d_k*d_v+d_k`，与序列长度无关，但随特征维度增长。所谓常数内存，是固定维度、固定层数、固定批量时相对于长度的常数，不是与任何条件都无关的一个小数。

按`T=24,d_k=d_v=8`手算，K/V需要三百八十四个元素，线性状态需要七十二个元素。长度为二时，K/V只需要三十二个元素，反而小于七十二。令`2*T*d=d*d+d`，可得交叉长度`T=(d+1)/2`。这只是这两套状态定义的元素数交叉点，不是速度交叉点，更不是模型质量交叉点。

把头数、层数和批量加回来时，要检查哪些轴真的独立。标准多头KV通常按KV头数，而不是query头数计量；GQA的多个查询头共享同一KV头。若已经按KV头数计算，又额外除一次共享比例，就会重复折扣。层间共享、滑窗截断、压缩缓存等会改变公式，必须从实际实现的状态对象重新数，不能把任何叫KV的东西都套进同一表达式。

```python
import numpy as np
from labs.models.core import linear_attention_recurrent, TinyDecoder

q = np.ones((24, 8)); k = np.ones((24, 8)); v = np.ones((24, 8))
_, state = linear_attention_recurrent(q, k, v)
assert [x.shape for x in state] == [(8, 8), (8,)]
print('linear elements', sum(x.size for x in state))
print('linear float64 bytes', sum(x.nbytes for x in state))
model = TinyDecoder()
cache = None
for token in [1, 2, 3]:
    _, cache = model.step(token, cache)
for layer, (keys, values) in enumerate(cache):
    print('layer', layer, 'K', keys.shape, 'V', values.shape,
          'elements', keys.size + values.size)
expected = (model.config.layers * 2 * 3 * model.config.kv_heads
            * (model.config.dim // model.config.heads))
assert sum(k.size + v.size for k, v in cache) == expected
```

`array.size`是元素个数，`array.nbytes`是这个数组表示的元素字节数。对共享底层存储的多个视图简单相加，可能重复计算同一物理内存；对框架分配器只看数组字节，又会漏掉缓存池和对齐。本例缓存数组独立，足够检查数学状态账本。进入生产框架时再区分逻辑占用、实际分配和峰值，不要让三个不同数字使用同一个模糊的“显存”标题。

分块递推还要检查状态的生命周期。前缀状态可能被两个候选后续共同使用，如果函数直接修改它，第一个后续会污染第二个。课程线性递推复制传入状态，因此调用者可以保留前缀快照；你的新实现若改成原地更新，必须改变接口约定并相应管理所有权。状态可以继续计算，不等于状态可以安全共享。

## 13. 计算课：连接数、浮点操作与实际速度为何不同

长度为`T`的因果全连接，第一行访问一个键，第二行访问两个，最后一行访问`T`个，总连接数是一到`T`的和，即`T*(T+1)/2`。这个数字描述允许的query-key对。某个朴素稠密实现仍然先计算完整`T*T`分数，再遮挡未来位置；逻辑连接数量和实际执行的矩阵乘数量因此不同。报告里要指出你数的是哪一种。

局部窗口包含当前位置，窗口为四时，第六个位置最多访问自己和前三个历史位置。若再加入每八个位置一个全局键，这些键与局部集合要取并集，不能把两部分数量直接相加，否则重复键被计两次。课程`local_global_mask`构造的正是并集；`sparse_attention`逐行访问这个集合，并返回真实收集的键数量`visits`。它减少评分连接，但Python循环和每行索引分配可能很慢。

一个点积大约涉及`d`次乘法和`d-1`次加法。常见浮点操作计数把一次乘加视为两次操作，于是可粗略写成每对约`2*d`；但这个约定需要注明，指数、除法、归约和内存搬运不能简单当作同样廉价。注意力还有权重乘V，投影也有矩阵乘。只数评分部分而宣称整个模型计算减少同样比例，通常夸大收益。

线性递推没有显式的`T*T`分数矩阵，却需要每步更新`d_k*d_v`状态，并读取它与查询相乘。固定`d_k,d_v`时它随长度线性增长；如果维度很大，常数项仍可能显著。对于短序列，高度优化的大矩阵乘也可能比大量小递推更快。复杂度是增长规律，具体硬件性能是另一层实验，两者互相解释但不能彼此替代。

```python
from labs.models.core import local_global_mask

for length in [1, 4, 8, 24]:
    mask = local_global_mask(length, window=4, global_stride=8)
    actual = int(mask.sum())
    hand = 0
    for i in range(length):
        allowed = {j for j in range(i + 1)
                   if j > i - 4 or j % 8 == 0}
        hand += len(allowed)
    assert actual == hand
    print(length, 'causal pairs', length * (length + 1) // 2,
          'local-global pairs', actual)
```

这段集合推导不调用掩码数组，因而是连接计数的独立核对。注意`range(i+1)`已经限制了因果方向，而条件中的`j>i-4`精确对应包含当前位置的四项窗口。把它换成`>=`会多保留一个历史位置。研究论文或代码中的window可能采用其他约定；你应以集合表达式说明本次采用哪一种，不能只在表里写“窗口四”。

如果要做速度测试，先把正确性门禁放在计时之外，计时区域只包含你要测的操作。CPU实验应记录进程环境、线程配置、数组形状、数据类型和预热次数，使用重复测量的中位数及分位数，而不是挑最快一轮。GPU实验还必须处理异步执行和同步，明确是否包含编译、传输和缓存分配。计时程序结束得快，可能只是把任务排进了设备队列。

本章没有给任何H100或5070Ti吞吐数字。你以后可以按算子卷的有限预算协议测量一个设备、几个固定形状，先确认没有公司任务冲突，再扩大范围。八张GPU的存在不是正确性研究的前提；绝大多数本章问题在CPU小数组上能更便宜、更透明地回答。

## 14. 源码阅读课：用一个词元走完真实调用链

读源码时，先选择一项足够小、又能穿过主要模块的任务，例如“已有三个历史词元，现在输入第四个，最后一层如何得到新位置的logits”。不要同时研究训练、量化、分布式、图捕获和全部后端。先选一个明确路径，通常是最直观的eager实现；理解它以后，其他路径才有可比较的语义基准。

从配置开始并不是因为配置简单，而是因为它决定后面数字的含义。记录词表大小、隐藏维度、层数、查询头数、KV头数、中间层宽度、位置编码与归一化参数。遇到配置没有给出的字段，追默认值和构造函数，不要凭同类模型的经验补上。一个模型类支持多种配置，并不说明你研究的那个checkpoint使用了全部功能。

接着找模型构造函数，确定参数在哪里创建。把权重的形状写出来：查询投影、键投影、值投影、输出投影，以及前馈网络各投影。检查是否有偏置、是否共享嵌入与输出权重、norm是否有可学习缩放。课程TinyDecoder没有可学习norm缩放，且输入嵌入和输出权重独立；外部模型若不同，你应把差异列明，不能仅凭同样写着RMSNorm就认为参数表完全一致。

进入forward之后，把每行操作翻译成“谁的什么轴发生了什么变化”。`view`和`reshape`改变如何解释元素布局，`transpose`改变轴的顺序，矩阵乘消去一个相同长度的轴并产生新的输出轴。不要只记“这里分头”。应能写出`q_raw[b,t,h*d+j]`对应哪个多头张量位置，再用一组递增整数确认。形状相同不代表每个元素仍处在正确的语义位置。

遇到缓存对象时，暂停追数学，先画所有权。谁创建缓存，谁持有它，更新函数返回新对象还是修改原对象，层号如何定位，位置从序列长度还是外部参数取得，多个请求是否共享存储。一次单请求前向正确，不能证明批量调度、前缀复用和请求取消都正确。这些属于推理系统卷，但本章必须先提供清晰的模型接口，才能让系统层正确调用。

再回到数学：确认位置编码在缓存更新之前还是之后；确认softmax的归一化轴；确认掩码以布尔还是加性偏置表达；确认缩放常数从哪个维度得到；确认不同query头怎样映射共享KV头。每个判断都应指向具体符号或运行记录。若看到后端分发，就写下当前分支和未读分支，不把一个分支的细节推断成全部后端的行为。

最后沿残差、前馈、最终归一化和输出头返回logits。logits是未归一化分数，不是概率，更不是生成出来的文本。采样器、停止条件和tokenizer解码属于后续调用路径。把模型forward与生成循环分开研究，你才能判断一个错误来自张量计算，还是来自重复追加词元、错误停止标记或随机采样状态。

本仓库的[固定版本源码工作坊](SOURCE_WALK.md)已经为Llama、Mamba与Attention Residuals保存了读取过的commit链接。使用它们时先把本章的小数组实验映射过去。比如Llama RoPE分半配对与本课程相邻配对存在坐标约定差异；Mamba所读reference的输入写入离散化与课程exact ZOH示范不同。这些是需要解释的真实差异，不能通过放大误差阈值抹平。

你的阅读笔记至少应出现一个“原先猜测错误”的修正。发现自己误把KV头数当查询头数，或者把深度来源轴当时间轴，说明你开始用证据约束理解。漂亮的结构图没有错误并不代表研究完整；关键是每条箭头都能还原成数组形状、参数和状态变化。

## 15. 十二道递进练习与完整参考解

以下答案都可以从仓库根目录运行，使用CPU和NumPy，不下载模型。先独立实现，再对照。对照时除了输出，还要说明你的版本使用了哪些输入契约；一个没有契约、恰好跑通给定例子的函数，还不能成为可靠参考。

### 题一：手算可见集合，而不调用掩码函数

长度六、局部窗口三、全局间隔四。逐行列出可见键，再用课程函数验证。第零个键总会满足全局间隔条件，但仍受到因果限制；第四个键只有从第四行开始才可见。参考解使用集合表达式，和原实现的数组广播路径不同，适合检查窗口边界。

```python
import numpy as np
from labs.models.core import local_global_mask
expected = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3],
            [0, 2, 3, 4], [0, 3, 4, 5]]
mask = local_global_mask(6, window=3, global_stride=4)
actual = [np.flatnonzero(row).tolist() for row in mask]
assert actual == expected
assert int(mask.sum()) == sum(map(len, expected))
print(actual)
```

第三行访问四个键并不违反“局部窗口三”：其中一、二、三是局部位置，零是额外全局位置。若你预期每行永远不超过三个键，说明没有把混合策略的并集算进去。这道题训练的是集合定义，而不是记忆某个函数输出。

### 题二：证明同掩码稀疏和稠密相同

使用不同的查询特征维和值特征维，避免方阵掩盖错误。除了输出误差，还检查返回访问量确实等于掩码真值数量。参考解限定每行至少可见自身，因此不会触发全屏蔽行的约定差异。

```python
import numpy as np
from labs.models.core import attention, sparse_attention, local_global_mask
r = np.random.default_rng(12)
q = r.normal(size=(7, 3)); k = r.normal(size=(7, 3)); v = r.normal(size=(7, 5))
mask = local_global_mask(7, 2, 3)
dense, weights = attention(q, k, v, mask)
sparse, visits = sparse_attention(q, k, v, mask)
np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(weights.sum(axis=1), np.ones(7))
assert np.all(weights[~mask] == 0)
assert visits == int(mask.sum())
print('visits', visits, 'maximum error', np.max(np.abs(sparse - dense)))
```

若只比较稀疏输出与全因果稠密输出，这个测试应该失败，因为可见集合不同。正确性比较要求同一函数；质量实验才会研究删掉连接是否影响任务表现。二者不能使用同一种“误差接近零”的验收条件。

### 题三：分块延续线性注意力状态

把九个位置在第四个位置之后切开。分别比较完整输出、最终状态，并确认第一段状态没有被第二次调用修改。此题把数值契约和所有权契约放在一起，适用于后来做前缀分支的场景。

```python
import numpy as np
from labs.models.core import linear_attention_recurrent
r = np.random.default_rng(3)
q = r.normal(size=(9, 3)); k = r.normal(size=(9, 3)); v = r.normal(size=(9, 2))
whole, final = linear_attention_recurrent(q, k, v)
first, prefix = linear_attention_recurrent(q[:4], k[:4], v[:4])
snapshot = tuple(a.copy() for a in prefix)
second, continued = linear_attention_recurrent(q[4:], k[4:], v[4:], prefix)
np.testing.assert_allclose(np.concatenate([first, second]), whole)
for a, b in zip(continued, final): np.testing.assert_allclose(a, b)
for a, b in zip(prefix, snapshot): np.testing.assert_array_equal(a, b)
print('split continuation passed')
```

这里`prefix`是二元组，元组不可改并不意味着内部数组不可变；真正保护快照的是实现复制数组。若你的答案只说“传的是元组所以安全”，就还没有理解Python容器和数组存储之间的区别。

### 题四：给SSM卷积补上非零初态响应

原卷积函数假设初态零。现在给定非零`h_initial`，推导额外项。递推第一步已经乘一次`a`，所以位置零的初态响应是`sum(c*a*h_initial)`，位置`t`是`sum(c*a**(t+1)*h_initial)`。把指数写成`t`会整体错一步。

```python
import numpy as np
from labs.models.core import diagonal_ssm, ssm_convolution
u = np.array([1., -1., 2., 0.])
a = np.array([.5, .8]); b = np.array([1., .3]); c = np.array([2., -1.])
h_initial = np.array([3., -2.])
recurrent, _ = diagonal_ssm(u, a, b, c, initial=h_initial)
zero_state = ssm_convolution(u, a, b, c)
response = np.array([np.sum(c * a ** (t + 1) * h_initial)
                     for t in range(len(u))])
np.testing.assert_allclose(recurrent, zero_state + response)
assert not np.allclose(recurrent, zero_state)
print('initial-state response', response)
```

此处可以叠加输入响应和初态响应，是因为这套固定系数SSM是线性的。输入依赖的选择性系统一般不能直接使用同一个固定卷积核；看到递推形式相似不代表所有推导条件仍然成立。

### 题五：验证decoder完整前向、单步缓存与因果性

先比较同一个模型实例的两种执行方式，再修改未来位置，要求过去位置的输出保持不变。使用同一实例避免无意中比较不同随机权重。参考解检查全部位置，最后一个词元相同还不足以证明之前各步正确。

```python
import numpy as np
from labs.models.core import TinyDecoder
model = TinyDecoder()
tokens = np.array([1, 3, 5, 7, 9])
full = model.forward(tokens)
cache = None; step_outputs = []
for token in tokens:
    logits, cache = model.step(int(token), cache)
    step_outputs.append(logits)
np.testing.assert_allclose(np.stack(step_outputs), full, rtol=1e-10, atol=1e-10)
changed = tokens.copy(); changed[3:] = [11, 13]
np.testing.assert_allclose(model.forward(changed)[:3], full[:3])
assert all(k.shape[1] == len(tokens) for k, v in cache)
print('full/step and prefix causality passed')
```

因果性测试说明未来输入没有影响已检查的前缀输出；它不能证明模型有语言能力，也不能证明每个当前位置输出有用。随机权重足以验证结构连接，却不适合评价理解和推理质量。

### 题六：独立计算GQA缓存元素数

取三层、隐藏维度二十四、六个查询头、两个KV头，输入五个词元。每头维度四，每层K/V各含`2*5*4`个元素，总计二百四十。参考解把手算公式与实际数组核对，不能用数组结果反过来当手算过程。

```python
from labs.models.core import DecoderConfig, TinyDecoder
config = DecoderConfig(dim=24, heads=6, kv_heads=2, layers=3, hidden=48)
model = TinyDecoder(config); cache = None
for token in [1, 2, 3, 4, 5]: _, cache = model.step(token, cache)
expected = 3 * 2 * 2 * 5 * 4
actual = sum(k.size + v.size for k, v in cache)
assert expected == actual == 240
print(actual)
```

如果误用六个查询头，答案会变成七百二十。计算时先写每个因子的单位：层、K与V两份、KV头、位置、每头特征。带单位展开能迅速发现重复乘头数或漏掉两份缓存的问题。

### 题七：确认输入词嵌入与输出头是否共享参数

阅读课程构造函数后，用对象和数值实验确认结论。两个矩阵转置后形状能匹配，不代表底层共享存储。参考解只修改自己的新模型实例，不影响其他练习。

```python
import numpy as np
from labs.models.core import TinyDecoder
model = TinyDecoder()
old_output = model.output.copy()
assert model.embedding.shape == model.output.T.shape
assert not np.shares_memory(model.embedding, model.output)
model.embedding[0, 0] += 10
np.testing.assert_array_equal(model.output, old_output)
print('teaching decoder uses independent embedding and output weights')
```

真实工程可能通过赋值、参数绑定或加载逻辑实现权重共享；必须追实际对象关系。报告参数数量时，若同一存储被两个模块引用，应说明按唯一参数计数还是按模块槽位计数，否则不同报告的总数无法比较。

### 题八：写一个参数账本并与数组总量核对

课程无bias、无可学习norm参数。每层查询与输出各`d*d`，键和值各`d*kv`，三个前馈矩阵总计`3*d*h`，嵌入与输出头合计`2*vocab*d`。这个公式只针对本课程配置，不得直接套到真实Llama或其他checkpoint。

```python
from labs.models.core import TinyDecoder
model = TinyDecoder(); c = model.config
kv = c.kv_heads * (c.dim // c.heads)
per_layer = 2 * c.dim * c.dim + 2 * c.dim * kv + 3 * c.dim * c.hidden
expected = 2 * c.vocab * c.dim + c.layers * per_layer
actual = model.embedding.size + model.output.size
actual += sum(weight.size for block in model.blocks for weight in block.values())
assert expected == actual
print('parameters', actual, 'per layer', per_layer)
```

这个练习的重点是解释漏项和适用范围。以后加入bias、norm缩放、共享权重或专家网络，都应先修改账本再修改报告，而不是把旧公式继续用于新的结构。

### 题九：用输入干预检查注意力的值通道

将所有值向量加同一个向量`delta`，输出也应加同一个向量，因为每行注意力权重和为一。保持Q、K和掩码不变，才能把变化归因于V。此题是一种变形测试，不需要显式计算每行softmax的正确数值。

```python
import numpy as np
from labs.models.core import attention, causal_mask
r = np.random.default_rng(4)
q = r.normal(size=(5, 3)); k = r.normal(size=(5, 3)); v = r.normal(size=(5, 2))
delta = np.array([2., -3.])
first, weights = attention(q, k, v, causal_mask(5))
second, _ = attention(q, k, v + delta, causal_mask(5))
np.testing.assert_allclose(second, first + delta, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(weights.sum(axis=-1), 1.)
print('value translation property passed')
```

若输出没有平移，可能是权重没有正确归一化、值聚合轴错误，或者代码额外对V做了别的变换。性质失败提供的是定位线索，不会自动告诉你具体错误行；随后要缩小到单行、单输出坐标继续检查。

### 题十：演示快照别名错误并修复

模拟每一步封存状态。错误版本反复保存同一个数组引用，后续修改使所有历史一起变化；修复版本保存副本。必须同时展示错误结果与正确结果，才能证明你知道故障机制，而不是偶然使用了`copy`。

```python
import numpy as np
state = np.zeros(2); wrong = []; correct = []
for value in [1., 2., 3.]:
    state += value
    wrong.append(state)
    correct.append(state.copy())
assert [x.tolist() for x in wrong] == [[6., 6.], [6., 6.], [6., 6.]]
assert [x.tolist() for x in correct] == [[1., 1.], [3., 3.], [6., 6.]]
assert wrong[0] is wrong[2]
assert correct[0] is not correct[2]
print('wrong', wrong, 'correct', correct)
```

复制会增加开销，因此生产实现未必每一步复制全部状态。它可能采用不可变块、引用计数或写时复制。但优化必须保留明确的生命周期契约；“不复制更快”不能成为污染历史的理由。

### 题十一：用内容摘要检查研究输入是否相同

相同文件名不代表相同内容。下面用标准库JSON序列化和SHA-256给小实验输入生成摘要。排序键和固定分隔符让相同字典内容产生相同文本，但浮点序列化约定也属于协议，跨语言研究需要更严格的数据格式约束。

```python
import hashlib
import json

def digest(record):
    payload = json.dumps(record, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

first = {'tokens': [1, 2, 3], 'seed': 7, 'dtype': 'float64'}
reordered = {'dtype': 'float64', 'seed': 7, 'tokens': [1, 2, 3]}
changed = {'tokens': [1, 2, 4], 'seed': 7, 'dtype': 'float64'}
assert digest(first) == digest(reordered)
assert digest(first) != digest(changed)
print(digest(first))
```

摘要说明两个序列化输入是否相同，不能证明数据来自可信来源，也不能证明没有数据泄漏。将摘要和来源、划分方式一起记录，才形成可解释的输入身份。

### 题十二：为现有比较台建立结果契约

运行两个长度，检查误差字段有限、四项等价关系通过、计数公式正确。不要在测试中断言实际耗时，否则机器繁忙会使数学正确的代码随机失败。性能另设报告，不混进离线正确性门禁。

```python
import math
from labs.models.compare import compare

for length in [1, 9]:
    report = compare(length=length, dim=4, seed=21)
    errors = {k: v for k, v in report.items() if k.startswith('max_error_')}
    assert len(errors) == 4
    assert all(math.isfinite(value) and value < 1e-10 for value in errors.values())
    assert report['dense_causal_score_pairs'] == length * (length + 1) // 2
    assert report['one_head_dense_kv_state_elements'] == 2 * length * 4
    assert report['one_head_kernel_linear_state_elements'] == 4 * 4 + 4
    assert 1 <= report['sparse_score_pairs'] <= report['dense_causal_score_pairs']
    print(length, errors)
```

长度一是重要边界：所有合法因果方法都只能访问当前位置，很多复杂状态更新会在这里暴露初始化问题。长度九跨过全局键间隔，也能检查局部和全局连接发生重叠时的计数。

## 16. 三个故障诊断室：先缩小差异，再决定修改哪里

**故障一：稀疏版和稠密版的误差一直很大。** 第一轮不要改softmax，先打印同一个查询位置的可见键编号。假设局部实现允许最近三个位置，稠密参考却用了完整因果掩码；二者的分数可能完全正确，只是读取了不同历史。最小复现取查询和键全零、值为位置编号，此时输出就是可见值的平均数，能够排除点积和数值稳定性的干扰。修复比较协议，让两边使用同一个集合。然后故意保留不同集合，记录输出确实不同，说明测试能够区分两个问题。若可见集合已经一致，再比较每个允许位置的分数、权重和最终值聚合，沿数据流逐段缩小。

**故障二：单步解码第一步一致，第二步开始偏离。** 这个时间特征优先指向状态或位置。检查第二步cache长度是否为一，位置是否为一，过去K是否被错误旋转第二次，各层是否使用自己的缓存。再打印本步新增K与完整前向中同一位置K的对应值。不要首先怀疑采样：比较的是logits，完全可以输入固定词元序列，移除生成随机性。若重新创建模型实例，保证权重完全相同；否则你可能在调试“不同参数产生不同结果”这个正常现象。修复后至少保留长度一、长度二和多层完整序列测试，防止初始化正确但累计过程再次出错。

**故障三：分段SSM只在第一段正确，第二段少了历史影响。** 先检查第二段是否传入第一段最终状态，再核对初态定义是进入本段前还是处理第一个输入后。将第二段输入全部设为零，可以让输入响应消失，只观察历史状态按`a`衰减的过程。若你拿零初态卷积作参考，自然看不到这段响应；应使用题四推导的初态项。这个故障的修复不是给误差增加一个常数，而是把状态边界写清楚。对于输入依赖系数，进一步检查第二段参数是否按原始全局输入生成，不能因为切块而重新定义时间或条件。

三个故障都有共同方法：先选择能隔离一个机制的输入。全零Q/K隔离可见集合，全零后续输入隔离初态，固定词元隔离采样。随机大输入适合发现问题，小而有结构的输入适合解释问题。定位完后再把这个反例保存成长期回归测试，让未来优化必须保留它所验证的性质。

## 17. 分阶段综合项目：交付一份可以重做的架构研究

**第一阶段，正确性比较台。** 以`labs.models.compare`为基线，在自己的研究脚本加入题三的分块延续与题四的非零初态两项。保留原来四个误差字段，新增字段明确命名。报告输入长度、维度、seed和dtype，失败时同时输出最大误差位置与前后数组切片。验收不是打印“通过”，而是另一位读者只需仓库和命令，就能得到同一类结果并理解各字段含义。此阶段只需CPU，无需下载任何权重。

**第二阶段，受控故障与修复。** 从三个诊断室选择一个，复制一份最小独立故障函数，不破坏课程公共实现。先运行应当失败的断言并解释失败原因，再给修复函数与通过证据。把失败输入压缩到尽可能小的形状，例如两步、一个头、两维状态。能把复杂错误缩成四个数的计算，是深入掌握的标志；只展示修复后没有报错，无法证明你定位了真实原因。

**第三阶段，真实源码映射。** 从固定版本工作坊选择一条路线，交付配置表、参数表、轴变换表和状态生命周期表。每张表至少五条具体记录，每条都带符号名称或固定源码位置。运行本地NumPy参考说明其中一个关键机制；如果没有安装上游库，就明确证据是“读取源码并运行独立参考”，不要写“复现上游运行”。若选择安装，只在单独环境执行上游的小测试，记录依赖、结果和未覆盖分支，不自动下载大权重。

**第四阶段，结构变体。** 选择一个范围有限的变化，例如把局部窗口从三改为五，或者在线性递推中增加明确的状态衰减。先写新公式、旧公式和预期差异，再改实现。衰减让旧信息权重随时间变化，原来的无衰减并行公式不再是正确oracle；你必须同步推导带衰减的参考。验收既包含新实现内部等价，也包含至少一个输入证明新旧定义确实不同。这样才能避免“改了代码但实际上没有改变任何行为”的假实验。

**第五阶段，质量与性能分别立项。** 若要讨论模型学得如何，接入M15的小训练任务，固定训练和评估划分、参数预算、优化器、训练步数及生成协议。至少保留独立测试集和几种失败样本。若要讨论运行快慢，先冻结已通过正确性的版本，再按照计时协议测量，保留机器和软件信息。你可以先完成其中一个方向；报告必须明确另一个方向尚无证据，而不是将低损失自动解释为高吞吐，或将小状态自动解释为更强推理能力。

项目的最终目录不要求复杂。一个问题说明、一份可运行脚本、一份机器可读结果、一份失败案例和一份解释报告已经足够。复杂的仪表盘不能补偿不清楚的输入契约；反之，一个简洁目录只要证据完整，也可以成为后续学习推理框架和算子优化的可靠起点。

## 18. 毕业口试：用证据回答，而不是用术语回答

请拿出一个你没有预先背过的小结构，独立回答：它保留什么历史，状态每一步怎样更新，哪些轴用于时间混合，哪些轴用于深度混合，哪部分参数跨位置共享，哪个参考实现可以验证它，哪种输入最容易暴露错误。回答时必须写至少一个两步状态表和一个完整数值例子。若只能说“这是高效注意力”“用了缓存”“复杂度更低”，说明还没有达到本章要求。

再解释一个合理的不相等：不同掩码、不同RoPE坐标布局、不同离散化或不同初态都可以。能够说明为什么不该相等，与能够修复本该相等的两个实现同样重要。最后指出你目前证据的边界：CPU小数组的正确性、真实源码的阅读、上游测试的执行、GPU性能的测量和真实任务质量，是不同层次的事实。把它们分清之后，你就能持续阅读新模型，而不需要每出现一个新名字就从头建立全部理解。
