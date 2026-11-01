# K05 归约与归一化：从一串加法推到Softmax、RMSNorm、LayerNorm和反向传播

本章把三个层次同时讲清楚：数学上要得到什么结果，有限精度会怎样改变结果，许多线程怎样合作完成它。归约看起来只是“把很多数合成一个数”，却贯穿注意力、归一化、损失、采样和通信。如果只知道函数名，不理解中间状态与误差，就很难判断一个自写算子究竟错在公式、并行顺序还是数值范围。

先完成K01至K04，并能阅读基础NumPy与简单循环。本章所有数学推导先用小数组，主要参考实验可以在CPU完成：

~~~bash
python -m labs.kernels.run --lab 6
python -m labs.kernels.run --lab 7
python -m labs.kernels.run --lab 8
python -m labs.kernels.run --lab 9
python -m labs.kernels.run --lab 10
python -m labs.kernels.run --lab 16
~~~

读完后应能从空文件写稳定softmax和RMSNorm前后向，解释全掩码行的合同，用有限差分检查梯度，并为GPU实现设计正确的阶段与测试。这里没有把CPU数值通过等同于GPU编译或性能通过。

## 第一课：归约的输入、输出和轴

### 1. 从求和开始定义

输入是一串数，求和输出一个数。最大值归约也输入一串数，但输出其中最大的值。若输入是矩阵，可以沿行中的列轴归约，得到每行一个结果；也可以沿行轴归约，得到每列一个结果。轴决定哪些元素被合并，不能只说“对矩阵做sum”而不解释范围。

设矩阵两行三列，第一行为一、二、三，第二行为十、二十、三十。沿最后一维求和得到六和六十，形状可以是长度二，也可以保留归约维成为二行一列。沿第一维求和得到十一、二十二、三十三。两个结果都叫sum，但业务含义与后续广播方式不同。

keepdims保留长度一的归约轴，使结果更容易与原数组广播组合。例如每行最大值保持二乘一，减到二乘三矩阵时，同一行的三个元素使用该行最大值。若丢掉轴后形状不巧仍能广播，可能把本该按行的数作用到列，产生静默语义错误。

### 2. 中性元素与空输入

求和的中性元素是零：任何数加零不变。求乘积的中性元素是一。最大值在适当扩展数域中可以用负无穷作为初始状态。中性元素让尾块无效槽位可以参与统一归约结构，而不改变合法输入结果。

空输入的行为要由接口定义。求和通常可以自然返回零；空轴softmax没有一个正常概率分布可输出，教学实现明确拒绝；最大值对空集合也需要特别合同。不要因为某个库选择了一种行为，就在自己的算子里无意识继承另一种行为。

GPU归约经常把输入补齐到某个方便并行的长度。补的是物理计算槽位，不是新增有效数据。你必须选择与目标归约匹配的中性值：最大值补零会改变全负输入的结果，求和补负无穷则完全错误。

## 第二课：串行顺序与树形顺序

### 1. 数学上的结合律为何重要

实数精确算术中，加法满足结合律，先算前两个再加第三个，或先算后两个再加第一个，结果相同。这使我们能够把长求和拆成多个部分并行计算，再合并部分结果。最大值也具有适合分组合并的性质。

串行求和形成一条长依赖链：每一步都等待上一步总和。树形求和先成对相加，再合并成对结果，依赖深度随层数增长更慢。对八个元素，串行需要七次依赖更新，平衡树有三轮合并。总加法次数仍是七，改变的是并行结构。

### 2. 完整八元素树

输入一到八，第一轮得到三、七、十一、十五；第二轮得到十和二十六；第三轮得到三十六。每个叶子元素进入且只进入一个分支，最终所有分支合并，因此每个输入贡献恰好一次。

若输入长度不是二的幂，可以在合适阶段补中性值，或采用能处理不规则长度的算法。仓库tree_sum在奇数长度时补一个零，再进行相邻配对。补零不改变数学和，但会改变有限精度下的具体运算路径，因此后面仍要讨论误差。

~~~python
import numpy as np

def tree_sum_reference(values):
    a = np.asarray(values).copy().ravel()
    if a.size == 0:
        return 0.0
    while a.size > 1:
        if a.size % 2:
            a = np.concatenate([a, np.zeros(1, dtype=a.dtype)])
        a = a[::2] + a[1::2]
    return a[0]

assert tree_sum_reference(np.arange(1, 9, dtype=np.int64)) == 36
~~~

copy防止修改调用者输入，ravel把逻辑元素变成一维工作序列，切片偶数位置与奇数位置建立成对集合。每轮数组长度约减半，直到一个值。它是清晰的CPU参考，不是高性能设备归约实现。

### 3. 多阶段GPU归约

一个block可以先得到一份partial，多个block产生多个partial，再由第二阶段归约。这样全局问题被分成局部协作和全局合并两层。每层都要记录输入长度、输出长度、谁写输出以及阶段顺序。

不要在普通Kernel里让所有block用一个自制全局等待屏障强行合并；K03已经解释资源驻留可能导致死锁。多Kernel阶段是更容易证明正确的起点。之后再研究特定协作启动、原子合并或更高级策略，各自有不同合同。

## 第三课：浮点数为什么不满足你习惯的精确加法

### 1. 有限有效位意味着舍入

浮点格式只能保存有限数量的有效二进制位。数值很大时，相邻可表示数之间的间隔也变大。把一个很小的数加到一个很大的数上，结果可能舍入回原来的大数。这个现象不是程序随机坏掉，而是表示精度的直接结果。

在常见float32中，一亿加一可能仍表示为一亿。于是数组一亿、一、负一亿、一的串行求和，先丢掉第一个一，再与负一亿抵消，最后加一，得到一；相邻配对的树形求和可能把两个一分别丢掉，最后得到零；更高精度参考得到二。

~~~python
import numpy as np
x = np.array([1e8, 1, -1e8, 1], dtype=np.float32)
serial = np.float32(0)
for value in x:
    serial = np.float32(serial + value)
tree = tree_sum_reference(x)
wide = x.astype(np.float64).sum()
assert float(serial) == 1.0
assert float(tree) == 0.0
assert float(wide) == 2.0
~~~

这不是说树形求和总比串行差。在很多输入上，平衡合并有更好的误差特性；这里只用一个反例证明不同顺序可以不同。讨论数值算法时，不应该用单个例子推出永远优劣，而应分析输入范围、误差模型和实际目标。

### 2. 确定性与准确性是两条轴

每次都按同一错误顺序得到零，可以是确定的，却不接近精确答案二；每次都很接近二但末位偶尔变化，可以数值准确却不逐bit确定。你在量化和模型评测中关心的稳定性，需要同时区分这两种目标。

并行原子浮点累加可能因完成顺序不同而改变舍入路径。固定归约树可以约束顺序，但跨架构、编译器和指令选择仍可能影响结果。不能因为设置了随机seed，就认为所有浮点归约顺序都被固定。

### 3. 提升累加精度与补偿求和

把输入先转换为更宽类型再累加，能够减少某些舍入问题；在窄类型里已经算完再转换，则不能恢复丢失信息。GPU常见混合精度算术也正是在输入格式、乘法方式和累加格式之间做取舍。

补偿求和试图记录被舍入丢失的一部分误差，但它增加状态和运算，未必适合所有高吞吐Kernel。学习它的意义是知道“求和”可以有多个数值实现，而不是把某一个库sum当成无条件精确答案。参考oracle也要说明精度与范围。

## 第四课：Softmax先是一个概率映射，再是一个Kernel

### 1. 从权重归一化理解公式

给定一行实数logits，先对每个数取指数，得到正权重，再除以所有权重之和。这样输出非负，并且正常有限输入下总和为一。指数使较大logit获得更大权重，但logit本身不是概率，也不要求位于零到一之间。

如果logits为零、ln二、ln三，指数权重为一、二、三，总和六，概率分别六分之一、三分之一、二分之一。这个例子可以不用近似小数完成全部手算，非常适合检查公式和代码。

softmax通常沿某个指定轴独立作用。注意力中每个query对可见key位置归一化，词表输出中每个token位置对词表项归一化。归一化轴改变，数学任务就改变，即使最终shape相同也不能混用。

### 2. 为什么直接exp可能溢出

logits加上一个很大常数后，概率在数学上不应改变，但直接指数可能超过浮点范围。例如一千、一千零一、一千零二的指数非常大，在普通浮点格式中可能变成无穷。再做无穷除以无穷，会得到无效结果。

因此稳定实现利用平移不变性：所有logits减去同一个常数，指数分子和分母都乘以相同因子，比例不变。选择本行最大值后，所有差值不大于零，最大指数等于一，避免正方向指数爆炸。

### 3. 严格推导平移不变性

原概率为exp(x_i)除以所有exp(x_j)之和。把每个x减去m后，分子变成exp(x_i)乘exp(-m)，分母每一项也都乘exp(-m)。这个公共正因子约去，所以结果相同。选择最大值只是为了数值方便，不是改变模型定义。

减最大值不保证任何输入都绝对没有数值问题。极小权重可能下溢到零，输入NaN或无穷需要额外合同，后续累加仍有舍入。稳定通常表示避免主要的可预见溢出路径，而不是宣称无限范围精确计算。

## 第五课：逐行实现稳定Softmax

~~~python
import numpy as np

def softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('nonempty row dimension required')
    if not np.isfinite(x).all():
        raise ValueError('finite logits required')
    maximum = x.max(axis=1, keepdims=True)
    shifted = x - maximum
    weights = np.exp(shifted)
    denominator = weights.sum(axis=1, keepdims=True)
    return weights / denominator

x = np.array([[0.0, np.log(2), np.log(3)],
              [1000.0, 1000.0 + np.log(2), 1000.0 + np.log(3)]])
p = softmax_rows(x)
assert np.allclose(p[0], [1/6, 1/3, 1/2])
assert np.allclose(p[0], p[1])
assert np.allclose(p.sum(axis=1), 1)
~~~

asarray明确参考计算精度，维数检查限制本函数处理二维行集合。检查有限值避免把未定义输入悄悄送进指数和归约。maximum的shape是行数乘一，因此减法会把每行自己的最大值广播到该行所有列。若错写axis零，得到的是每列最大值，程序可能仍能运行，但数学含义完全不同。

weights与输入shape相同；denominator每行一个值。正常非空有限行至少有一个减后值为零，因此至少一个指数为一，分母严格为正，不需要随意加一个极小常数来防止零除。如果你的实现出现零分母，应先检查mask、输入和状态合同，而不是自动加epsilon掩盖问题。

测试同时验证手算答案、平移不变性和行和为一。这三项捕捉不同错误。一个总是输出均匀分布的假实现也能让行和为一，却不能通过第一项，所以不能只用归一化性质验收。

## 第六课：Mask不是把概率事后乘零那么简单

### 1. 掩码必须进入归一化范围

设logits为零、ln二、ln三，只允许第一项和第三项。正确权重是一和三，分母四，输出四分之一、零、四分之三。如果先对三项做softmax，再把第二项乘零，得到六分之一、零、二分之一，总和只有三分之二，不是同一个分布。

因此无效位置应在求分母前排除。常见表达是把无效logit视为负无穷，使其指数贡献零，但实现仍要处理全无效行与无效算术。使用一个很大的负有限数只是近似屏蔽，不能在所有数值范围里等同于严格排除。

### 2. 全掩码行为什么需要单独合同

若一行没有任何有效位置，数学上的“在可见项上归一化成概率分布”没有普通定义。不同系统可以选择报错、返回全零或采用任务特定行为。本卷CPU参考明确选择全零输出，并把它当作接口合同，不声称这是一个和为一的概率分布。

如果直接把所有分数设为负无穷，再减本行最大值负无穷，就会出现负无穷减负无穷，产生NaN。稳定实现必须识别没有有效最大值的情况，用安全参考值参与中间计算，并让无效权重始终为零。

### 3. 完整带mask参考

~~~python
def masked_softmax_rows(x, mask):
    x = np.asarray(x, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('nonempty matrix rows required')
    if not np.isfinite(x[mask]).all():
        raise ValueError('visible logits must be finite')
    scores = np.where(mask, x, -np.inf)
    maximum = scores.max(axis=1, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    weights = np.where(mask, np.exp(scores-safe_maximum), 0.0)
    denominator = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, denominator,
                     out=np.zeros_like(weights),
                     where=denominator > 0)

x = [[0, np.log(2), np.log(3)], [2, 3, 4]]
mask = [[True, False, True], [False, False, False]]
p = masked_softmax_rows(x, mask)
assert np.allclose(p[0], [0.25, 0, 0.75])
assert np.array_equal(p[1], [0, 0, 0])
~~~

np.divide的out提供默认零结果，where只在分母正时执行相应除法。这里不能简单写where条件包住一个已经计算好的weights除denominator，因为某些表达式求值会先产生无效除法。正确使用带where语义的运算，把无效路径从实际除法中排除。

## 第七课：Softmax反向传播从两元素推到通用公式

### 1. 一个logit会影响全部概率

提高某个logit，不仅提高它自己的指数权重，也增大所有项共同的分母。因此自己的概率上升，其他概率下降。softmax的导数不是各元素互不相关的逐元素导数，它包含整行耦合。

设输出概率为p，上游梯度为g，g_i表示最终损失对p_i的变化敏感度。我们需要求损失对每个输入logit的梯度。通过链式法则，把所有输出概率对该输入的影响加起来，得到一个可以高效计算的形式：输入梯度等于p逐元素乘以g减去整行p乘g的和。

### 2. 完整手算两元素

令p为四分之一、四分之三，上游梯度g为二、负一。先计算加权和：四分之一乘二，加四分之三乘负一，等于负四分之一。第一项梯度为四分之一乘二加四分之一，等于九除十六；第二项为四分之三乘负一加四分之一，等于负九除十六。

两项梯度相加为零。这与平移不变性一致：如果所有logit一起增加相同小量，输出不变，因此沿全一方向的导数为零。这个不变量是很好的测试，但同样不能单独证明所有梯度都正确。

~~~python
def softmax_backward(p, grad):
    p = np.asarray(p, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    if p.shape != grad.shape:
        raise ValueError('shape mismatch')
    dot = (p * grad).sum(axis=-1, keepdims=True)
    return p * (grad - dot)

dx = softmax_backward([0.25, 0.75], [2.0, -1.0])
assert np.allclose(dx, [9/16, -9/16])
assert abs(dx.sum()) < 1e-12
~~~

这个实现不需要显式创建完整的概率雅可比矩阵，只需要一次行归约和逐元素计算。数学推导让你知道为什么可以避免二次大小的中间矩阵，而不是凭源码看见一个sum就猜它在做什么。

## 第八课：RMSNorm从均方根开始

### 1. “均方根”三个字按顺序执行

输入一行x，先把每个元素平方，再求平均，最后开平方，得到均方根。RMSNorm用这个尺度归一化原输入，通常还乘以每个特征自己的可学习权重。它不先减均值，因此与LayerNorm不同。

设x为三和四，平方为九和十六，平均十二点五，平方根约三点五三五五。忽略epsilon作手算时，归一化结果约零点八四八五和一点一三一四。注意第二个数大于一并不错误：这里不是概率归一化，要求的不是每项位于零到一，也不是总和为一。

若权重gamma为一和二，第二个归一化值再乘二，得到约二点二六二七。权重负责每个特征的可学习缩放，不能把它当成求均方根时的一部分，除非你有意定义了不同算子。

### 2. epsilon放在哪里具有数学意义

常见公式把epsilon加在均方值之后、开平方之前，避免接近零时分母过小。把epsilon加在平方根之后，得到的是另一种函数，其梯度也不同。不能因为两种写法都“防止除零”就互换。

epsilon也会破坏某些理想化尺度不变性。没有epsilon且乘正比例时，归一化可以抵消整体尺度；有epsilon时，特别在输入很小时，输出会受到绝对尺度影响。测试应根据实际公式设计，而不是把忽略epsilon的手算性质硬套到所有范围。

### 3. 完整前向

~~~python
def rmsnorm_reference(x, weight, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError('nonempty feature dimension required')
    if weight.shape != (x.shape[-1],):
        raise ValueError('weight must match feature dimension')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('positive finite epsilon required')
    mean_square = (x*x).mean(axis=-1, keepdims=True)
    inverse_rms = 1.0 / np.sqrt(mean_square + eps)
    return x * inverse_rms * weight
~~~

类型转换发生在平方之前，避免整数输入在较窄整数类型里先平方溢出。这个参考仍有有限数值范围，极端巨大浮点数的平方可能溢出；若接口需要覆盖那种范围，应采用额外缩放策略并写明合同，不能仅凭输入有限就声称中间所有运算必然有限。

## 第九课：RMSNorm反向逐项推导

### 1. 把依赖拆成直接路径与尺度路径

输出y_i等于x_i乘r再乘gamma_i，其中r是整行均方值加epsilon的负二分之一次方。损失对x的影响有两条路：x直接进入自己的输出；x还通过平方平均改变r，从而影响整行所有输出。遗漏第二条路会得到形状正确但数学错误的梯度。

先把上游梯度g与gamma逐元素相乘，记为u。直接路径给出r乘u。尺度路径通过均方值的导数与r的导数，最终得到减去x乘r的三次方，再乘整行u乘x的平均。这里的平均包含特征数分母，不能误写成未除维度的sum。

### 2. 数值例子

取x为三、四，gamma为一、一，上游g为一、零，手算时令epsilon为零。r约零点二八二八四三，u乘x的平均是一点五。第一维梯度约零点一八一零一九，第二维约负零点一三五七六五。虽然第二个输出的上游梯度为零，第二个输入梯度却不为零，因为它改变了共同归一化尺度。

这是检验是否理解耦合的关键例子。若你写的反向只是g乘gamma乘r，就会错误地给第二维零梯度。有限差分也能抓住这个问题，但先用手算解释原因，才能避免把数值检查当成黑箱评分器。

~~~python
def rmsnorm_backward_reference(x, weight, grad, eps=1e-5):
    x, weight, grad = [np.asarray(a, dtype=np.float64)
                       for a in (x, weight, grad)]
    inverse = 1.0 / np.sqrt((x*x).mean(axis=-1, keepdims=True) + eps)
    u = grad * weight
    dx = inverse*u - x*(inverse**3)*(u*x).mean(axis=-1, keepdims=True)
    axes = tuple(range(x.ndim-1))
    dw = (grad*x*inverse).sum(axis=axes) if axes else grad*x*inverse
    return dx, dw
~~~

gamma在不同batch或token行之间共享，所以weight梯度要沿所有非特征轴累加。x梯度则保持每行自身归一化依赖，不跨无关行混合。正确选择归约轴，是反向实现最常见的难点之一。

## 第十课：LayerNorm为何还要减均值

### 1. 完整前向区别

LayerNorm先求一行均值，再用每个元素减均值，得到中心化值；计算这些中心化值的平方平均作为方差，再除以平方根尺度，通常还带可学习缩放与偏置。RMSNorm直接对原值求均方根，没有中心化步骤。

输入一和三，均值二，中心化为负一和一，方差一，忽略epsilon得到负一和一。输入整体加十变成十一和十三，中心化仍为负一和一，因此LayerNorm对整体平移具有相应不变性。RMSNorm则不具备同样平移不变性。

### 2. 方差的两种公式在有限精度下不同

数学上，均方减均值平方等于中心化平方的平均。但当所有数都很大且彼此很接近时，前者是在两个巨大且接近的数之间做减法，容易损失有效位。直接中心化后平方，或使用Welford等稳定更新方法，常能更好处理这类情况。

这里再次出现“数学等价不等于浮点路径相同”。你不能因为两条公式在纸上相等，就只测试普通小随机数而忽略大偏置、小方差输入。归一化恰好对尺度与差值敏感，数值测试应覆盖这些情境。

### 3. 完整参考实现

~~~python
def layernorm_reference(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] == 0:
        raise ValueError('empty feature axis')
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = (centered*centered).mean(axis=-1, keepdims=True)
    return centered / np.sqrt(variance + eps)

assert np.allclose(layernorm_reference([[1.0, 3.0]]),
                   layernorm_reference([[11.0, 13.0]]))
~~~

这份参考没有加入gamma与beta，目的是先隔离归一化核心。生产LayerNorm接口若包含这些参数，数学合同与梯度还要继续扩展。省略参数必须明确写出，不能把一个核心函数宣称为所有框架接口的直接替代。

## 第十一课：Welford让均值与方差可以逐步更新

### 1. 保存三个状态

Welford维护已见元素数量n、当前均值mean和中心平方差总和M2。来一个新值x，先把数量加一，计算x与旧均值的差delta，再用delta除新数量更新均值，最后用delta乘x减新均值更新M2。

为什么需要旧均值与新均值两个差？因为新增元素不仅带来自己的偏差，也改变了已有元素的中心位置。这个更新式把两种影响组合起来，避免每次重新扫描全部历史，也避免直接用两个巨大平方量相减。

### 2. 手算一、二、三

初始n零、mean零、M2零。读一后，delta一，n一，mean一，x减新mean为零，所以M2仍零。读二时，旧mean一，delta一，n二，新mean一点五，新差零点五，M2增加零点五。

读三时，旧mean一点五，delta一点五，n三，新mean二，新差一，M2增加一点五，最终M2为二。总体方差是M2除n，得到三分之二；若某统计任务需要无偏样本方差，则除n减一得到一。LayerNorm通常使用其定义中的总体形式，不要把统计估计中的分母习惯随意换进模型算子。

~~~python
def welford(values):
    n = 0
    mean = 0.0
    m2 = 0.0
    for value in values:
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
    return n, mean, m2

assert np.allclose(welford([1.0, 2.0, 3.0]), [3, 2, 2])
~~~

### 3. 并行合并两个状态

两个子集分别有数量、均值和M2。合并时，新数量相加，新均值按数量加权；M2除了两边自身平方差，还要加入两边均值差导致的跨组修正项。这个修正与均值差平方、两边数量乘积及总数量有关。

例如一、二一组，三、四一组，两边数量都二，均值一点五与三点五，M2都零点五。均值差二，跨组修正为二的平方乘二乘二再除四，等于四；总M2为五，总体方差一点二五，与完整四元素计算一致。

### 4. 为什么Welford更新式中有两个不同的差值

设已经处理n个样本，旧均值为μ，旧平方离差和为M₂。新样本为x，更新后的均值是μ′=μ+(x−μ)/(n+1)。这里的除数必须是更新后的样本个数；若还除以n，第一个样本就会遇到零除，而且之后的状态也不再是样本均值。

记δ=x−μ，δ₂=x−μ′。新的平方离差和不是旧M₂简单加δ²，因为所有旧样本的中心也从μ移动到了μ′。把旧样本对新中心的平方展开，交叉项包含旧离差之和，而旧离差之和为零。旧样本的整体修正是n(μ−μ′)²，新样本贡献是(x−μ′)²。将μ′−μ=δ/(n+1)代入，二者合计得到δ²n/(n+1)，也就是δδ₂。

因此算法写成先算δ，再更新μ，最后用更新后的μ计算δ₂，并累加δδ₂。若把两个差值都定义为旧均值下的δ，累计的就是另一个量。看上去只差一行代码的执行顺序，实际改变了统计量定义。这个推导也解释为何不能把M₂误叫成“方差”：它还没有除以样本数。

以样本十、十二、十四为例。第一步n为一，均值十，M₂为零。第二步δ为二，新均值十一，δ₂为一，M₂增加二。第三步δ为三，新均值十二，δ₂为二，M₂再增加六，总计八。直接围绕最终均值十二计算平方离差，得到四加零加四，仍然是八。总体方差为八除以三；样本无偏估计是八除以二。LayerNorm通常使用前者，不应误套统计课程里默认的样本无偏方差。

### 5. 可合并状态的完整实现

~~~python
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('finite sample required')
        n = self.count + 1
        delta = value - self.mean
        mean = self.mean + delta / n
        return Moments(n, mean, self.m2 + delta * (value - mean))

    def merge(self, other):
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        n = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / n
        correction = delta * delta * self.count * other.count / n
        return Moments(n, mean, self.m2 + other.m2 + correction)

    def population_variance(self):
        if self.count == 0:
            raise ValueError('empty state has no variance')
        return self.m2 / self.count

def moments_of(values):
    state = Moments()
    for value in values:
        state = state.add(value)
    return state

left = moments_of([1, 2])
right = moments_of([3, 4])
joined = left.merge(right)
assert joined.count == 4
assert joined.mean == 2.5
assert joined.m2 == 5.0
assert joined.population_variance() == 1.25
assert Moments().merge(joined) == joined
~~~

frozen让这个教学状态不可原地修改，每一步产生新状态，便于保存轨迹与避免误用更新后的字段。实际GPU实现通常使用寄存器中的可变标量，思想相同，不要求照搬Python对象分配。count记录有效样本数，所以尾块无效槽应贡献空状态，而不是额外插入零样本；插入零样本会改变均值和方差。

合并时先处理空状态，让它成为合并的中性元素。两个非空组的总均值是样本数加权平均，不是两个均值简单取平均。只有两个组样本数相等时，简单平均才恰好正确。用一个样本的一组和三个样本的一组作测试，比两个等长组更容易暴露这个错误。

平方离差的修正项来自两组中心之间的距离。假设左组内部全部是零，右组内部全部是十；各组M₂都是零，但合并后显然存在变化。没有δ²n₁n₂/n这一项，就会错误报告零方差。这个反例比随机输入更能直接指出缺失的数学项。

合并在实数算术中满足适合树形归约的结合关系，有限精度仍会受合并次序影响。将顺序版本、分两组版本、分四组版本与float64中心化参考比较时，应检查合理误差，并记录分组方式。不能把“Welford稳定”解释成任意数据范围和任意合并顺序下完全相同。

### 6. 统计量、归一化、可微分性之间的边界

Welford是一种计算均值与方差的方法，不会改变LayerNorm的数学函数。你可以前向用Welford得到均值与逆标准差，反向仍使用从最终数学表达式推导的公式。没有必要把统计更新循环的每一步都当作必须保存的自动微分节点；自定义算子的价值之一，就是用紧凑的足够状态描述反向所需信息。

但这不意味着前向实现细节绝对无关。如果前向在某处做了截断、量化、条件分支或特殊NaN处理，反向合同必须说明这些操作如何处理。教学中先限定有限输入和正epsilon，可以把重点放在可微的普通路径。之后引入低精度与异常值处理时，应分别讨论数学参考、实际离散计算和所采用的梯度近似。

保存均值与逆标准差通常比保存所有归约中间值更节省内存。对R行N列输入，统计量只有每行几个标量，而不是每行N个数。若反向仍需要输入或归一化后的值，就还要考虑保存输入的引用、重算成本及生命周期；“只保存两个统计量”不自动代表整个反向不再需要输入数据。


## 第十二课：从参考公式变成GPU阶段

### 1. Softmax的三项工作

一行softmax需要最大值归约、指数和归约、归一化写出。一个block或program可以承担一行，线程或向量元素分担列位置。中间值是保留在寄存器、写到shared还是重新读取global，会影响资源与流量。

若一行很短，一个大block可能有很多无效线程；若一行很长，单个block或program需要的状态可能过大。生产实现会按尺寸选择策略，教学实现可以先限制最大行宽并明确检查。限制形状不等于算法错误，但不能假装支持任意长行。

### 2. 归约中性值必须与每一阶段匹配

求最大值阶段，无效列应贡献负无穷；求指数和阶段，无效列贡献零；归一化输出阶段，无效列不写回。把同一个填充值从第一阶段一路机械沿用，可能使分母包含无效位置或造成全无效行NaN。

RMSNorm的平方和阶段无效列贡献零，但平均值分母应使用真实特征数N，而不是补齐后的BLOCK大小。若N为三而程序补到四，除以四会改变尺度；只有无效槽位参与的算术被中性化，统计定义中的元素数不能随padding改变。

### 3. 反向也需要归约，不只是逐元素乘法

Softmax反向先算整行p乘g的和，再逐元素应用公式。RMSNorm反向需要整行u乘x的平均，weight梯度还可能跨多行归约。前向每行独立，不代表所有参数梯度都能逐行独立写回同一个位置。

多个行对同一weight梯度有贡献时，需要分阶段合并、原子或其他正确归约策略。普通store会覆盖其他行贡献。K10会把这种梯度所有权与框架autograd接口连接起来。

### 4. 一个八线程Softmax的完整执行账本

现在把一行十三个logit交给八个线程。线程t先处理列t，如果t加八仍小于十三，再处理列t加八。因此线程零至四各处理两个数，线程五至七各处理一个数。每一列的线程归属是列号模八，线程内部轮次是列号整除八。这和K03的grid-stride思想一致，只是范围局限在当前行。

设输入为零、零、零、零、零、零、零、零、ln二、ln二、ln二、ln二、ln二。线程零至四的局部最大值为ln二，线程五至七的局部最大值为零。八个局部最大值写入shared槽零至七，所有线程到达屏障，才允许开始跨线程读取。第一轮四个线程分别比较槽零与四、一与五、二与六、三与七；第二轮两个线程合并零与二、一与三；第三轮线程零合并零与一。最终shared零中得到全行最大值ln二。

这一步的局部最大值初始值是负无穷。若线程因为行很短完全没有合法列，它仍要把负无穷写入自己的shared槽并参加屏障。让这类线程提前return，会使参与屏障的线程集合不一致。输入mask和同步控制是两种不同机制，不能把“没有数据”误解成“不再参与协作”。

所有线程读取最终最大值后，shared数组即将被用于下一次求和。这里还需要明确防止覆盖的顺序：读取最大值到每个线程的私有寄存器，再建立全block屏障，然后才可以写入自己的局部指数和。如果某线程已经覆盖shared零，而另一个线程尚未读到最大值，后者会把局部和误当成最大值。前一轮末尾的屏障保证归约结果已写好，却不能自动保证下一阶段每个线程已经把它读走。

减去最大值后，前八个元素的指数均为二分之一，后五个元素的指数均为一。线程零至四的局部指数和为一点五，线程五至七的局部指数和为零点五。八个局部和再次进入树形求和，总和是九。最终前八个概率为十八分之一，后五个概率为九分之一。检查总概率为八除以十八加五除以九，恰好等于一。

如果实现先把指数临时写进输出数组，再读取输出完成除法，那么同一行会写两次输出：第一次写未归一化权重，第二次覆盖为概率。它能减少重新计算exp，但增加设备内存流量。另一个版本可以重算exp，少存中间值。比较这两种策略时，不能只数浮点操作；还要看行宽、缓存、寄存器、占用和实际指数指令成本。

### 5. 一个可以逐行读懂的CUDA行Softmax

下面的代码限定每行有限浮点输入、列数大于零、输入输出不重叠。每个block负责一行，block线程数必须是编译时常量T，且T为二的幂。它刻意使用显式shared归约，便于学习；高性能库可能采用warp归约、向量读取或更复杂调度。

~~~cpp
#include <cuda_runtime.h>
#include <cmath>

template<int T>
__global__ void teaching_softmax(const float* x, float* y,
                                int rows, int cols) {
    static_assert(T > 0 && (T & (T - 1)) == 0,
                  "T must be a power of two");
    __shared__ float scratch[T];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += T)
        local_max = fmaxf(local_max, x[row * cols + col]);
    scratch[tid] = local_max;
    __syncthreads();
    for (int step = T / 2; step > 0; step /= 2) {
        if (tid < step)
            scratch[tid] = fmaxf(scratch[tid],
                               scratch[tid + step]);
        __syncthreads();
    }
    float maximum = scratch[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += T) {
        float weight = expf(x[row * cols + col] - maximum);
        y[row * cols + col] = weight;
        local_sum += weight;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int step = T / 2; step > 0; step /= 2) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    float denominator = scratch[0];
    for (int col = tid; col < cols; col += T)
        y[row * cols + col] /= denominator;
}
~~~

static_assert检查的是编译期T，而不是运行时launch是否真的用了T个线程。调用者必须用teaching_softmax<128>配合128线程，不能模板写128却启动64或256线程。前者会留下未初始化shared槽，后者会越界写shared。接口参数与启动配置共同构成算子合同，模板本身不能替你验证所有调用错误。

row由block索引决定，整个block共享同一个row，所以row超界时所有线程一起return，不会形成同一block内有的线程参与屏障、有的线程不参与的情况。相比之下，按col超界提前return就不安全，因为同一block中线程的列索引不同。这里把列边界放在for循环条件里，无数据线程保留中性值并完整参与归约。

row乘cols是行的起始元素偏移，随后加col定位元素。教学接口用int适合小实验，生产实现若可能超过整型范围，应使用足够宽的索引类型并验证乘积，不应直接把这个示例推广到任意大小张量。这里的指针加法以float元素为单位，不要再乘四；字节大小只在分配和复制接口中计算。

两轮归约都在每一步末尾同步，因为下一步读取的槽可能由别的线程刚更新。step从T除二开始，每轮减半，活动线程数逐渐减少，但全部线程仍执行屏障。把屏障放入if(tid<step)会让非活动线程跳过屏障，破坏协作合同。条件写数据和条件参加屏障必须分开考虑。

最后除法不需要再用shared做跨线程通信；每个线程读取同一个已经完成的分母，然后只更新自己负责的列。函数结束本身不是对host的同步，host比较结果前还要通过适当的设备同步或阻塞复制观察完成。本仓库没有在当前无GPU环境执行此代码，学习者应先用CPU轨迹核对，再按有界GPU实验流程编译运行。

### 6. LayerNorm反向为什么多出一个“去均值”项

记一行输入x，均值μ，中心化值c=x−μ，逆标准差r=(mean(c²)+ε)的负二分之一次方，输出y=γcr+β。将上游梯度g先乘γ得到u。反向有三条影响路径：x直接影响中心化值，x通过均值影响整行中心化值，x通过方差影响共同尺度。只保留第一条，就会得到错误的逐元素梯度。

对中心化过程，微小变化dc=dx−mean(dx)。这说明所有输入一起加同一个微小常数时，中心化值不变。因此正确的输入梯度沿全一向量方向应没有分量，整行梯度和接近零。这个性质是LayerNorm的重要测试；RMSNorm没有减均值，通常不具备相同的平移不变性。

方差v=mean(c²)，其微分为二乘mean(c·dc)。因为mean(c)等于零，mean(c·dc)与mean(c·dx)相同。逆标准差的微分dr=−r³ mean(c·dx)。将输出微分代入损失微分，再收集每个dx的系数，得到输入梯度为r乘以u减u的均值，再减去c乘r的平方乘mean(u·c)。等价写法是r[u−mean(u)−xhat·mean(u·xhat)]，其中xhat=cr。

这个简洁公式仍包含epsilon的影响，因为r来自带epsilon的方差；推导没有假设epsilon为零。若你直接假设mean(xhat²)严格等于一，再据此简化其他式子，在epsilon非零时可能引入错误。对于常量行，c全零，方差为零，但r为一除以epsilon平方根，前向输出β，输入梯度仍可能非零，因为对单个元素的扰动会打破常量行。

参数梯度中，γ的梯度是g乘xhat并沿所有共享该参数的行相加，β的梯度是g沿这些行相加。若输入shape是batch乘time乘hidden，而γ只有hidden维，则前两个轴都要归约。只沿batch求和会留下time轴，可能在后续隐式广播中产生错误，甚至被某些框架累积逻辑暂时掩盖。

~~~python
def layernorm_backward_reference(x, gamma, grad, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    if x.shape != grad.shape or gamma.shape != (x.shape[-1],):
        raise ValueError('shape mismatch')
    if x.shape[-1] == 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError('positive eps and nonempty features required')
    center = x - x.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt(np.mean(center * center, axis=-1,
                              keepdims=True) + eps)
    normalized = center * inv
    u = grad * gamma
    dx = inv * (u - u.mean(axis=-1, keepdims=True)
                - normalized * np.mean(u * normalized,
                                      axis=-1, keepdims=True))
    axes = tuple(range(x.ndim - 1))
    dgamma = np.sum(grad * normalized, axis=axes)
    dbeta = np.sum(grad, axis=axes)
    return dx, dgamma, dbeta

x = np.array([[1.0, 2.0, 4.0]])
gamma = np.array([1.0, 2.0, 0.5])
grad = np.array([[2.0, -1.0, 3.0]])
dx, dgamma, dbeta = layernorm_backward_reference(x, gamma, grad)
assert np.allclose(dx.sum(axis=-1), 0, atol=1e-12)
assert np.array_equal(dbeta, grad[0])
~~~

axes可能是空元组，这表示一维输入没有需要合并的前导轴，参数梯度应保持逐特征形状。不要把空元组错误替换成None；NumPy中的axis=None意味着归约所有轴，会把参数梯度变成单个数。这是一个很适合初学者练习的API语义细节：数学上的“没有归约轴”和库参数中的“未指定归约轴”往往不同。

### 7. 用标量损失检查整个向量反向

有限差分最容易核对标量函数的梯度。若算子输出是向量，可以固定一个上游权重g，构造标量损失L=sum(y·g)。对每个输入元素做正负微小扰动，计算两次损失之差再除以两倍步长，就得到该输入位置的数值梯度。这样检查的是任意向量雅可比积，而不必显式存放完整雅可比矩阵。

g不要总取全一。softmax各输出和恒为一，取全一时损失是常量，正确输入梯度全零；一个错误地永远返回零梯度的实现会轻易通过。选择包含正、负、不同大小的固定g，能够检验更完整的耦合路径。对于归一化参数梯度，也分别扰动γ和β，不能只检查x。

步长并不是越小越好。过大时中央差分的截断误差明显，过小时两个相近损失相减会损失有效数字。教学时用float64，先试一乘十的负四次方、一乘十的负五次方、一乘十的负六次方，观察误差是否存在合理稳定区间。若所有步长都差很多，优先怀疑公式或实现；若只有极小步长变差，可能是数值消减。

~~~python
def central_difference(function, value, step=1e-5):
    value = np.asarray(value, dtype=np.float64).copy()
    result = np.empty_like(value)
    for index in np.ndindex(value.shape):
        original = value[index]
        value[index] = original + step
        positive = float(function(value))
        value[index] = original - step
        negative = float(function(value))
        value[index] = original
        result[index] = (positive - negative) / (2 * step)
    return result

def layernorm_loss(x, gamma, beta, upstream, eps):
    centered = x - x.mean(axis=-1, keepdims=True)
    inv = 1 / np.sqrt(np.mean(centered**2, axis=-1,
                             keepdims=True) + eps)
    y = centered * inv * gamma + beta
    return np.sum(y * upstream)

beta = np.array([0.2, -0.1, 0.4])
numeric_x = central_difference(
    lambda value: layernorm_loss(value, gamma, beta, grad, 1e-5), x)
numeric_gamma = central_difference(
    lambda value: layernorm_loss(x, value, beta, grad, 1e-5), gamma)
numeric_beta = central_difference(
    lambda value: layernorm_loss(x, gamma, value, grad, 1e-5), beta)
assert np.allclose(dx, numeric_x, rtol=1e-5, atol=1e-7)
assert np.allclose(dgamma, numeric_gamma, rtol=1e-5, atol=1e-7)
assert np.allclose(dbeta, numeric_beta, rtol=1e-5, atol=1e-7)
~~~

这里每次扰动后恢复原值，否则下一位置的差分会建立在已经改变的输入上。copy防止检查器修改调用者的数据，np.ndindex保证对任意维数逐元素遍历。函数参数使用独立lambda固定其他变量，从而清楚地区分输入梯度、缩放参数梯度与偏置参数梯度。

在有mask的softmax中，mask是离散选择，通常不对布尔mask求梯度。保持mask不变，对有效logit扰动检查；无效logit不影响输出，梯度应为零。全mask行按本章零输出合同，也应给出零输入梯度。若某接口选择对全mask行报错，则对应检查应验证异常，而不是沿用零梯度预期。

### 8. 哪些测试能把错误定位到某一层

先检查输出shape与dtype，再检查有限性，再检查逐元素误差，最后检查结构不变量。这个顺序能避免NaN污染误差统计，或用错误广播把本来不同形状的结果强行比较。np.allclose允许广播，因此正式检查应先断言shape完全一致，再调用近似比较。

对softmax，保存每行最大值、指数总和和最终行和。最大值错误通常指向轴、mask或最大值中性值；最大值正确而指数总和错误，检查减最大值、有效元素计数和求和归约；两者都正确而概率错误，检查输出地址或分母广播。对于RMSNorm，保存平方和、真实N、逆均方根以及γ索引，能够把常见尺度偏差逐层拆开。

对梯度，先在长度二或三的小向量上打印u、归约内积和最终dx。softmax需要sum(p·g)，RMSNorm需要mean(u·x)，LayerNorm需要mean(u)与mean(u·xhat)。这些量看起来相似，却有不同的权重、是否取均值及中心化定义。把一套归约机械复用到另一算子，是常见的“代码非常像、结果就是不对”来源。

误差报告至少给出最大绝对误差、出现位置、参考值、计算值和相对误差的分母规则。参考接近零时，相对误差会非常大，不能单独用它判断失败；大数值输出则只看绝对误差也不合理。常见判据是绝对误差不超过绝对容差加相对容差乘参考绝对值，但容差应由计算类型和误差要求解释，而不是不断调大直到测试变绿。

### 9. 从一个block扩展到长行时要重新设计什么

当一行很长时，一个block可通过线程内循环处理所有元素，但每个线程可能需要很长的串行工作，或者为了保存中间值消耗大量寄存器。让多个block共同处理一行，能够增加并行度，却引入跨block统计量合并与额外阶段。这个选择不是简单把grid变大；输出概率需要所有分块共享同一个最终最大值和分母。

一个易验证的长行softmax方案分三次launch。第一阶段每个行块得到局部最大值，第二阶段合并得到每行最大值并计算或安排局部指数和，后续阶段合并分母并归一化。具体可以合并部分工作，但每一次合并都要画清状态依赖。若局部指数采用各自局部最大值，不能直接把局部指数和相加，必须按统一最大值重新缩放，这正是下一章在线注意力状态合并的准备知识。

长行LayerNorm同样可以每块产生Welford状态，再合并得到每行均值与M₂。随后各块读取最终统计量完成归一化。这里的count必须是每块真实有效数量，最后一块通常较短；若误用统一块长，均值和方差都会受padding影响。把输入全设为同一个非零数，可以快速发现错误计数，因为正确方差应接近零。

参数梯度跨许多行归约时，临时缓冲可以按行组保存部分dγ和dβ，再二次归约。直接对同一参数做浮点原子加更简单，但可能改变确定性和性能特征。教学首先实现两阶段版本，并在报告中写出临时缓冲shape、元素数和生命周期，再以此为基线讨论原子或持久化策略。

### 10. 本章微型工程的文件与验收顺序

把数学参考、轨迹模拟、数值检查和GPU入口分开。参考文件只依赖NumPy，测试导入参考不应触发CUDA初始化；GPU入口在显式运行时才检查设备和编译环境。这样个人电脑暂时没有匹配软件栈时仍可完成大部分推导与调试，也不会让普通CPU测试因为缺少GPU而失败。

第一份提交实现三种前向并写明合同；第二份提交加入三个反向和中央差分；第三份提交加入归约树与Welford合并轨迹；第四份提交才加入一个受限GPU前向。每次提交都保留之前的可运行状态，出现错误时能通过版本差异定位。不要把“代码最终跑通”作为唯一学习记录，保留一次真实失败的输入和原因更有复习价值。

验收时选择一行普通输入、一行大公共偏置、一行常量、一行非整除长度和一个带全mask行的案例。逐个解释参考结果为什么如此，再展示测试结果。一个能够解释五个边界例子、知道实现合同的人，比只展示随机大矩阵速度数字的人更接近独立编写可靠算子的目标。


### 11. 补偿求和与成对求和怎样选择

前面用浮点反例说明求和顺序会改变结果，现在再实现一个保存误差补偿的顺序算法。它并不让浮点数变成实数，也不保证每种排列下都得到最佳结果；它的教学价值是把“累加器只有一个数字”的限制显式展开成“主和加补偿状态”。在性能敏感的GPU路径里，多一个状态和几次运算可能改变寄存器压力与吞吐，所以必须测量而不是凭名称选择。

~~~python
def compensated_sum(values):
    total = 0.0
    correction = 0.0
    for value in values:
        adjusted = float(value) - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    return total

small = [1.0] + [1e-10] * 10000
ordinary = sum(small)
compensated = compensated_sum(small)
exact_target = 1.0 + 10000 * 1e-10
assert abs(compensated - exact_target) <= abs(ordinary - exact_target)
~~~

adjusted先把上一次记录的舍入偏差从新输入中扣除，updated执行真正的浮点加法。表达式updated减total尝试重构这次实际进入主和的增量，再减adjusted得到被舍入影响的差值。下一轮用它修正新输入。变量命名比压成一行更重要，因为任何一次错误重排都可能破坏补偿的含义。

这个例子的比较只证明当前输入上的表现，不构成对所有浮点输入的优劣定理。尤其当输入数量极大、动态范围极宽或者出现溢出时，补偿仍然有局限。更高精度、分组顺序、缩放和问题本身的条件数都可能影响结果。写教材中的“稳定”算法时，要让读者知道它解决了哪一种失真，而不是给它贴上万能标签。

对于GPU归约，成对树往往自然适合并行结构；补偿求和则可能在线程内部的短序列上使用，再合并线程局部结果。混合方案需要重新定义合并状态，不能把每个线程的补偿值随意丢弃后宣称获得完整补偿效果。先建立误差目标，再讨论是否值得增加计算和状态，这是合理的设计顺序。

### 12. 最大值与位置一起归约时，平局也是合同

只求最大值时，相等元素之间没有选择问题；求argmax时还需要返回位置。输入三、七、七、二的最大值为七，但位置可以是第一处七或第二处七。接口必须规定平局规则，否则不同归约树可能返回不同位置，而每个结果看上去都“是最大值”。

一个可组合状态可以写成数值与索引的二元组。比较时先选择数值更大的状态，数值相等时选择索引更小的状态。这样“最早最大值”的规则被编码进每一次合并，而不是希望最终某条线程恰好先完成。无效槽使用负无穷与足够大的哨兵索引，但如果合法输入允许负无穷，还要确保平局规则不会误选哨兵。

~~~python
def combine_argmax(left, right):
    lv, li = left
    rv, ri = right
    if rv > lv or (rv == lv and ri < li):
        return right
    return left

def first_argmax(values):
    values = list(values)
    if not values:
        raise ValueError('empty input')
    if not all(np.isfinite(values)):
        raise ValueError('finite input contract')
    state = (float(values[0]), 0)
    for index, value in enumerate(values[1:], 1):
        state = combine_argmax(state, (float(value), index))
    return state

assert first_argmax([3, 7, 7, 2]) == (7.0, 1)
~~~

这个实现限定有限输入，避免NaN比较规则使二元比较出现不直观结果。若需要支持NaN，应明确传播、忽略或报错策略，并为所有合并路径实现相同规则。不能让CPU参考忽略NaN、GPU实现传播NaN，再靠放宽容差解释不一致。

argmax练习连接后面的路由、top-k和采样。那些算子常常同时产生数值和索引，索引的确定性、平局顺序以及有效数量直接影响后续数据搬运。一个看似只是“最大值附带位置”的扩展，实际上要求你把数学选择规则和并行合并操作保持一致。

### 13. 阅读一个陌生归约Kernel的六次停顿

第一次停在输入地址，写出每个线程拿到哪些元素，检查覆盖与重复。第二次停在局部状态初始化，确认中性值与归约类型一致。第三次停在线程间合并，画出每一轮谁读谁写，检查同步是否覆盖生产与消费。第四次停在最终缩放，确认分母是真实样本数、有效权重和还是别的统计量。

第五次停在输出所有权，确认每个结果由唯一线程或定义良好的原子协议写入。第六次停在误差测试，确认参考采用什么精度、比较哪些边界、是否检查全部输出。不要一开始就逐行猜每个优化变量的含义；先用这六个问题建立整个数据流，再回头理解向量化、warp shuffle或寄存器展开。

把这六次停顿写成自己的阅读笔记，而不是只在脑中想一遍。每次记录一个具体输入、一条线程轨迹和一个可能失败的反例。这样阅读开源实现时，你能判断自己已经理解哪一层，也能带着可验证的问题查官方文档，而不是在大量宏、模板和底层指令之间迷失。


## 第十三课：十二道练习与完整参考答案

### 练习一：写出归约轴

对矩阵第一行一、二、三，第二行十、二十、三十，按行求和得到六、六十；按列得到十一、二十二、三十三。保留维度时前者shape为二乘一，后者一乘三。答案需要指出后续广播各自会作用到哪些原元素。

### 练习二：手算树形归约

输入一到五，先补零成一到五加零，第一轮得到三、七、五；再补零，第二轮十、五；第三轮十五。每次补的是求和中性值。若任务改成最大值且输入全负，不能继续补零，应使用适当负无穷或有效性标记。

### 练习三：复现浮点非结合

运行一亿、一、负一亿、一的float32例子，记录串行、树形和float64结果，并逐步说明每个一在哪个阶段被舍入丢失。只写“浮点有误差”不够，必须关联到具体运算顺序。

### 练习四：手算稳定softmax

logits为零、ln二、ln三，减最大ln三后指数为三分之一、三分之二、一，总和二。除以二得到六分之一、三分之一、二分之一。与直接权重一、二、三归一化一致，展示公共尺度被约去。

### 练习五：错误mask的反例

先对三项求softmax再把第二项乘零，得到总和三分之二；正确可见集合归一化得到四分之一、零、四分之三。完整答案说明mask影响分母，不能只在输出上删除概率质量。

### 练习六：全mask合同

采用本卷全零合同，输入任意有限logits、mask全部假，输出全零且无NaN。验收不要求行和为一，因为该行没有概率分布。若项目选择报错合同，则测试应期待错误，而不能与本卷实现混用。

### 练习七：手算softmax反向

p为四分之一、四分之三，g为二、负一，答案九除十六、负九除十六。先计算加权和负四分之一，再逐元素计算。最后验证梯度和零，以对应平移不变性。

### 练习八：区分RMSNorm与LayerNorm

输入三、四，RMSNorm按均方根缩放，两个输出都为正；LayerNorm先减均值三点五，得到负零点五与零点五，再按方差归一化，输出约负一和一。结果符号不同不是误差，而是定义不同。

### 练习九：检查padding分母

真实N为三，输入一、二、二，平方和九，真实均方三。若补到四后误除四，均方变二点二五，归一化尺度错误。参考解要求无效列平方贡献零，但仍除真实N三。

### 练习十：有限差分梯度检查

~~~python
def finite_difference_scalar(fn, x, step=1e-5):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for index in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[index] += step
        minus[index] -= step
        out[index] = (fn(plus) - fn(minus)) / (2*step)
    return out

x = np.array([[0.2, -0.7, 1.1]])
g = np.array([[1.0, 2.0, -1.0]])
p = softmax_rows(x)
analytic = softmax_backward(p, g)
numeric = finite_difference_scalar(lambda z: float((softmax_rows(z)*g).sum()), x)
assert np.allclose(analytic, numeric, atol=1e-8, rtol=1e-6)
~~~

把向量输出与固定上游g做点积，变成标量函数，再检查每个输入分量的导数。这样验证的是实际反向需要的向量雅可比乘积，不必显式构造巨大雅可比矩阵。

### 练习十一：合并Welford状态

对一、二与三、四两组使用前文数值，合并数量四、均值二点五、M2五、总体方差一点二五。删除跨组修正项会得到M2一，明显错误。这个负向例子证明两组内部方差不能直接相加代替完整方差。

### 练习十二：设计归一化测试矩阵

至少包含普通随机输入、大公共偏置、小尺度、全零、单元素行、非整除行宽、部分mask、全mask和参数shape错误。前向检查手算值与不变量，反向检查有限差分和梯度shape。不能只用一个随机矩阵与框架结果接近就宣称覆盖全部边界。

## 第十四课：三个故障案例与修复过程

第一类故障是softmax大logits产生NaN。先检查是否直接exp，再用平移等价的两组输入对比；修复减最大值后，继续检查全mask行，避免另一个负无穷减负无穷路径。不要简单把所有NaN替换成零，因为那会隐藏正常行计算失败。

第二类故障是RMSNorm在非二的幂行宽上整体尺度偏差。检查平方和是否正确，再检查除数是N还是补齐BLOCK。用一、二、二这个三元素例子，可以把差异精确定位到均方定义，而不是浮点容差。

第三类故障是反向只有第一维接近参考，其他维差异很大。检查是否遗漏共同归一化尺度的路径，是否把mean写成sum，以及weight梯度归约轴是否正确。先用两元素手算，再用有限差分，最后才扩大随机形状。

## 第十五课：综合项目——写一个可解释的归一化算子包

第一阶段只提供CPU参考，包含稳定softmax、带mask版本、RMSNorm、LayerNorm核心和对应反向。每个接口有明确shape、dtype、mask与空输入合同。第二阶段增加逐步trace，输出最大值、指数和、均方值、逆尺度和梯度归约量，让错误能定位到中间状态。

第三阶段把其中一个前向搬到Triton或CUDA，使用同一CPU参考验收，覆盖真实N与补齐BLOCK不同的情况。第四阶段再接框架autograd，验证梯度与参数共享归约。每阶段保留可运行基线，不在一次修改里同时更换公式、精度和并行策略。

完成标准是你能从公式推到代码，又能从一条错误输出反推可能出错的中间量。性能优化建立在这些证据上：先知道计算了什么，再问怎样少搬数据、怎样并行归约、怎样减少同步和保留必要状态。

## 阅读实现时的版本与证据

本章数学参考以有限实数输入和明确的空输入合同为起点。阅读框架中的归一化算子时，先查当前固定版本的接口说明，确认默认epsilon、方差定义、参数shape和累加精度，再进入具体设备实现。名称相同并不保证全部边界行为相同；把接口差异误当成Kernel错误，会浪费大量调试时间。

可以从仓库固定来源中的Triton fused softmax教程开始，依次标记行号到program的映射、列向量的生成、尾部mask、减最大值、求和以及写回。然后对照本章显式shared版本，回答哪些同步由编译器根据张量运算生成，哪些资源选择仍由程序员通过block大小与启动参数影响。阅读目标是建立两个抽象层之间的对应关系，而不是记住一份代码的写法。

最后保留一份实验记录，写明执行的是CPU参考、编译检查还是实际GPU运行，输入范围与误差是多少。若只完成前两者，就如实注明设备路径尚未验收。这个区分能让之后的性能结果有可信基线，也使你在换到公司服务器时知道还有哪些检查必须补做。

掌握本章的标志，是能解释一个错误梯度来自哪条依赖路径，能说清一条屏障保护哪些读写，并能为任何padding位置选择正确的中性状态。只背出softmax或归一化公式，还不足以独立修改并行实现。
