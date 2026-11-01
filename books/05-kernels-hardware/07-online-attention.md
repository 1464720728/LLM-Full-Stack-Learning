# K07 在线注意力：从完整分数矩阵推导分块前向、反向重算与掩码边界

本章回答一个非常具体的问题：注意力需要对所有key计算概率，为什么还能不保存完整概率矩阵，逐块读取key与value得到同样的数学结果？答案不是“用了某种更快的softmax”这么简单，而是一组可以合并、重新缩放并保持不变量的状态。理解这组状态后，你才能读懂FlashAttention类实现的主循环，也能判断自己的mask、反向或分块修改是否正确。

先修K05的稳定softmax与K06的矩阵乘法。主要参考实验只依赖NumPy，可在CPU执行；本章给出的教学实现不声称达到真实FlashAttention的性能，也不覆盖所有框架接口。GPU实现的正确性、编译与性能需要在对应环境中分别验证。

## 第一课：Q、K、V与四个关键维度

### 1. Query在问什么，Key和Value分别是什么

将一次注意力看成带权检索：query与每个key比较，得到相关性分数；分数归一化成权重；最后用权重对value加权求和。key决定匹配程度，value决定被聚合的信息。两者经常来自同一组token的不同线性投影，但数学上不是同一个矩阵。

设Q有Nq行D列，K有Nk行D列，V有Nk行Dv列。Q与K的特征宽度必须相同，才能逐行做点积；K与V的行数必须相同，因为每个key对应一个value。输出O有Nq行Dv列。D与Dv可以不同，Nq与Nk也可以不同，这两个自由度在测试中很有价值。

若默认所有矩阵都是N乘D，许多错误会被方形shape隐藏。例如把value宽度误写成query宽度，只在Dv不等于D时暴露；把key长度误写成query长度，只在交叉注意力或带缓存解码时暴露。因此本章从一开始就保留四个独立符号，而不是先写一个只对方阵工作的公式再补边界。

### 2. 标准缩放点积注意力

分数S=QK的转置乘缩放系数s，通常s为一除以D平方根。对每个query行，在可见key位置做softmax得到P，再计算O=PV。这里softmax沿key轴归一化，不能沿query轴。每个输出行是同一组value的不同加权组合。

缩放的直觉是：若点积各项具有一定独立性与相近方差，D增长会让未缩放点积的典型幅度增大，导致softmax更尖锐。这个解释依赖输入分布假设，不是任何模型状态下的绝对定理。实现上应把scale作为明确参数或按D计算，并在前向与反向保持一致。

### 3. 一个完全能手算的注意力

取一个query，令其与三个key的缩放分数为零、ln二、ln三；value分别为[一,零]、[零,二]、[三,一]。概率是六分之一、三分之一、二分之一。输出第一维为六分之一加零加一点五，得到五除以三；第二维为零加三分之二加二分之一，得到七除以六。

这里不必纠结如何构造Q与K产生这些分数，可以先把分数当成已算出的输入，专注于归一化与value聚合。随后令D为一、Q为一、K为上述三个分数、scale为一，就得到完整QKV实例。小例子的价值在于所有状态都可以写成精确分数。

若只允许第一与第三个key，概率变成四分之一、零、四分之三，输出为[二点五,零点七五]。这个结果不能通过把原输出中“第二个key的贡献减掉”后保持原分母得到，因为mask改变了归一化分母。前向、在线状态与反向都必须使用同一可见集合。

## 第二课：完整矩阵参考是最重要的对照

### 1. 明确接口合同

本章参考接受二维有限浮点矩阵，允许不同query与key长度，允许不同query特征与value特征宽度。空query返回空输出；空key时本章选择零输出；mask必须能明确广播或与分数shape一致。为了让实现更容易审计，下面限定mask恰好为Nq乘Nk。

全mask行没有普通概率分布定义，本章沿用K05的零输出合同。这个选择适合教学和边界测试，不表示所有框架都采用相同策略。若你的生产接口要求报错，应在wrapper中明确检查，而不是在Kernel内部偶然产生NaN后再处理。

~~~python
import numpy as np

def attention_inputs(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('two-dimensional inputs required')
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError('incompatible attention shapes')
    if q.shape[1] == 0:
        raise ValueError('positive query/key feature width required')
    if not all(np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('finite inputs required')
    return q, k, v

def attention_dense(q, k, v, mask=None, scale=None):
    q, k, v = attention_inputs(q, k, v)
    nq, d = q.shape
    nk, dv = v.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    if not np.isfinite(scale):
        raise ValueError('finite scale required')
    allowed = np.ones((nq, nk), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if allowed.shape != (nq, nk):
        raise ValueError('mask shape mismatch')
    if nk == 0:
        return np.zeros((nq, dv)), np.zeros((nq, 0))
    scores = (q @ k.T) * scale
    masked = np.where(allowed, scores, -np.inf)
    maximum = masked.max(axis=1, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    weights = np.where(allowed, np.exp(masked - safe_maximum), 0.0)
    denominator = weights.sum(axis=1, keepdims=True)
    probability = np.divide(weights, denominator,
                            out=np.zeros_like(weights), where=denominator > 0)
    return probability @ v, probability
~~~

### 2. 每个中间shape都应该写下来

q乘k的转置得到Nq乘Nk分数，maximum与denominator是Nq乘一，probability仍为Nq乘Nk，最后乘V得到Nq乘Dv。keepdims保留归约轴，确保每行用自己的最大值和分母。若误丢维度，在某些方形输入上仍可能广播成功，却把行统计量错配到列。

np.where并不保证像程序分支一样只计算被选中的一侧，因此不能把危险算术简单藏在未选分支中就认为安全。这里先将全无效行的最大值替换为零，保证不会出现负无穷减负无穷；无效分数仍为负无穷，指数为零。后续除法使用where与预置零输出，避免零分母。

有限输入不等于所有中间乘积都一定有限。极端大值可能使点积溢出，即使输入本身有限。教学实验限定合理幅度，正式接口如果需要支持更宽范围，应增加缩放策略、异常检查或明确拒绝。稳定softmax解决指数的主要溢出路径，不会自动修复此前点积已经产生的无穷。

### 3. 内存问题来自哪里

完整分数矩阵与概率矩阵都随Nq乘Nk增长。自注意力中Nq与Nk都等于序列长度N，所以中间矩阵按N平方增长。Q、K、V与输出则大致按序列长度乘特征宽度增长。长序列时，保存二次规模中间张量会占用大量显存并产生大量读写。

在线注意力的目标是在保持普通精确注意力数学定义的前提下，不把完整分数或概率矩阵写到设备全局内存。它仍需要计算所有被允许的query-key配对，普通稠密情形的计算量并没有因此变成线性。减少中间存储与改变渐近计算量是两件不同的事。

## 第三课：先从单行在线Softmax开始

### 1. 为什么局部softmax不能直接拼接

把三个分数分成第一组零、ln二和第二组ln三。第一组局部softmax为三分之一、三分之二，第二组只有一个元素，局部softmax为一。把它们拼起来总和为二，不是全局分布。每组都失去了“这组在全体权重中占多少”的尺度信息。

正确合并需要保留未完全归一化的统计量。稳定表示可以保存局部最大值m与相对该最大值的指数和l。对于某组分数，l等于所有exp(score−m)之和。只要知道m和l，就能把该组重新表达在另一个公共最大值下。

### 2. 两组状态怎样合并

设旧组状态为m₁、l₁，新组为m₂、l₂，合并最大值m=max(m₁,m₂)。旧组指数和需要乘exp(m₁−m)，新组指数和需要乘exp(m₂−m)，然后相加。原因是每个旧权重exp(score−m₁)乘exp(m₁−m)，恰好变成exp(score−m)。

这就是重新缩放，不是经验修正。共同参考点改变后，所有以旧参考点表示的量都必须按同一因子转换。若只更新最大值、不缩放旧指数和，结果就把不同尺度的数字直接相加，数学上不成立。

### 3. 手算最大值发生变化的一轮

第一组分数零与ln二，最大值ln二，指数和为二分之一加一，等于一点五。第二组分数ln三，最大值ln三，指数和一。合并最大值为ln三，旧缩放系数为二除以三，新缩放系数一，合并指数和为一点五乘二除以三加一，得到二。

最终三个相对最大值权重为三分之一、三分之二、一，总和二，概率仍为六分之一、三分之一、二分之一。这个例子中的旧缩放系数不等于一，能够检验最关键的状态转换。若测试分数总按递减顺序排列，最大值可能从不增加，漏掉缩放的错误就不容易被发现。

## 第四课：把Value聚合加入同一状态

### 1. 再保存一个未归一化分子向量

除了m和l，再保存向量u，定义为所有exp(score−m)乘对应value的和。u长度为Dv，l是标量。最终输出为u除以l。这样不用保存每个概率，只保存对最终输出足够的聚合状态。

当公共最大值从旧m变成新m时，u与l必须乘同一个旧缩放因子。因为二者都由相同的指数权重构成，只是分子还乘了value。只缩放l而不缩放u，是在线注意力最常见且最容易用手算发现的错误之一。

### 2. 完整两块数值轨迹

沿用第一课value。第一块包含分数零、ln二。m为ln二，权重为二分之一、一，l为一点五，u为二分之一乘[一,零]加一乘[零,二]，得到[零点五,二]。

第二块包含分数ln三与value[三,一]。新m为ln三，旧缩放系数二除以三。旧u缩放后为[三分之一,四除以三]，加入新value后为[十除以三,七除以三]。新l为二，因此输出为[五除以三,七除以六]，与完整矩阵参考一致。

这里应把每个分数、权重、最大值、分母和分子都写进表格或日志。只打印最终输出无法区分错误来自QK点积、状态缩放、V索引还是最终除法。一个好的教学实现会提供trace，但生产Kernel通常不保留这些调试中间量。

### 3. 状态不变量

处理完任意前缀key集合后，m等于该集合有效分数的最大值；l等于该集合全部exp(score−m)之和；u等于相同权重乘value的和。最终u/l因此等价于对整个可见集合做softmax再聚合。

证明采用归纳法：空集合状态作为起点，每次合并一个新块时，用共同最大值重新表达旧集合和新集合，再相加。关键是每个key恰好属于一个块，且mask定义在每一块保持一致。重叠块会重复计入，遗漏块会少计贡献，状态公式本身无法修复错误分块。

### 4. 三块而非两块：逐轮验证分子与分母

为了避免只记住“两块合并公式”，再用四个key分成三块。分数依次为ln二、零、ln四、ln二，value为一、三、二、五。原始指数权重是二、一、四、二，总和九，加权分子为二加三加八加十，等于二十三，最终输出二十三除以九。

第一块只包含首项，m为ln二，l为一，u为一。第二块包含中间两项，块内最大值ln四，公共最大值提高到ln四。旧因子二分之一，新块相对权重为四分之一、一，所以新l为二分之一加四分之一加一，得到七除以四；新u为二分之一加三乘四分之一加二，得到十三除以四。

第三块分数ln二小于当前最大值ln四，因此旧因子为一，新权重二分之一。新l为九除以四，新u为十三除以四加五乘二分之一，得到二十三除以四。二者相除仍为二十三除以九。这个轨迹同时包含最大值上升与保持不变两种分支，适合检查程序中的条件和缩放。

如果把key顺序重新排列，精确实数结果不变，但每轮最大值与中间状态可能完全不同。有限精度下最终结果允许有细小差异。你应分别检查数学顺序不变性与数值误差边界，而不是把不同trace误认为实现不一致。状态表示可以不同，只要始终满足相同不变量。

### 5. 保存归一化输出的另一种合法状态

有些实现保存当前归一化输出o=u/l，而不是未归一化分子u。合并新块时，不能直接套本章u的更新式。若旧分母为l旧，新公共最大值下旧缩放系数为α，新块权重和为l块，新分母为l新=αl旧+l块，则新输出为(αl旧o旧+u块)/l新。

这说明旧输出需要乘αl旧除以l新，新块分子需要除以l新。若只写o新=αo旧+u块，就把归一化与未归一化状态混在一起。阅读开源代码时，看见acc或out变量不能立即判断其数学意义，必须追踪它是否在每轮被除过分母。

以第一块m为ln二、l为一点五、u为[零点五,二]为例，旧o为[三分之一,四除以三]。合并第三个key后，αl旧等于一，新l为二，所以旧o的系数为二分之一；新value[三,一]的系数也为二分之一。得到[五除以三,七除以六]。两种状态写法数学等价，却有不同的操作顺序和舍入路径。

保存未归一化u的优点是更新形式直接，最后只除一次；保存归一化o可能适合某些算法组织或状态合并。选择应基于数值、资源和指令安排，而不是变量名字是否熟悉。最好的阅读习惯是在每个循环边界写出变量的精确定义。

### 6. 在线状态可以树形合并，而不只顺序扫描

将每个key块独立计算成状态m、l、u，再用同一公共最大值规则合并，可以形成树形归约。这样不同块先并行计算，之后合并状态，适合某些长序列或解码分区策略。代价是保存每块部分输出与统计量，并执行额外合并。

空状态必须是合并中性元素。若一块对某query全无效，它的l为零、u为零、m为负无穷；合并时应直接返回另一有效状态，或使用避免无效算术的安全公式。两个空状态合并仍为空。把空块当作含一个零分数的块，会凭空增加一个value贡献。

~~~python
def merge_attention_states(left, right):
    m1, l1, u1 = left
    m2, l2, u2 = right
    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)
    if u1.shape != u2.shape:
        raise ValueError('value widths must match')
    if l1 == 0:
        return float(m2), float(l2), u2.copy()
    if l2 == 0:
        return float(m1), float(l1), u1.copy()
    maximum = max(m1, m2)
    a = np.exp(m1 - maximum)
    b = np.exp(m2 - maximum)
    return maximum, a * l1 + b * l2, a * u1 + b * u2

first = (np.log(2.0), 1.5, np.array([0.5, 2.0]))
second = (np.log(3.0), 1.0, np.array([3.0, 1.0]))
m_join, l_join, u_join = merge_attention_states(first, second)
assert np.allclose(u_join / l_join, [5/3, 7/6])
~~~

这个函数假定非空状态由合法有限分数构成，l严格正且u维度一致。它没有把任意三元组都验证成合法统计量，调用者不能随便传负分母或与m不一致的u。教学时可以增加状态构造器，让状态只能通过有效块产生，从而减少接口误用。

树形合并的数学结合性来自集合并集与共同尺度表示，有限精度仍受合并树影响。比较顺序、二叉树和多分区版本时，应固定输入并记录合并顺序，使用误差容差而不是强行要求所有末位一致。这个结论与K05的归约讨论一脉相承。


## 第五课：完整在线前向参考

~~~python
def attention_online(q, k, v, block_size=2, mask=None, scale=None,
                     return_trace=False):
    q, k, v = attention_inputs(q, k, v)
    if block_size <= 0:
        raise ValueError('positive key block size required')
    nq, d = q.shape
    nk, dv = v.shape
    scale = 1.0 / np.sqrt(d) if scale is None else float(scale)
    if not np.isfinite(scale):
        raise ValueError('finite scale required')
    allowed = np.ones((nq, nk), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if allowed.shape != (nq, nk):
        raise ValueError('mask shape mismatch')
    maximum = np.full(nq, -np.inf)
    denominator = np.zeros(nq)
    numerator = np.zeros((nq, dv))
    trace = []
    for begin in range(0, nk, block_size):
        end = min(begin + block_size, nk)
        scores = (q @ k[begin:end].T) * scale
        valid = allowed[:, begin:end]
        masked = np.where(valid, scores, -np.inf)
        block_maximum = masked.max(axis=1)
        updated_maximum = np.maximum(maximum, block_maximum)
        safe_maximum = np.where(np.isfinite(updated_maximum),
                                updated_maximum, 0.0)
        old_factor = np.exp(maximum - safe_maximum)
        weights = np.where(valid,
                           np.exp(masked - safe_maximum[:, None]), 0.0)
        denominator = old_factor * denominator + weights.sum(axis=1)
        numerator = old_factor[:, None] * numerator + weights @ v[begin:end]
        maximum = updated_maximum
        if return_trace:
            trace.append((begin, end, maximum.copy(),
                          denominator.copy(), numerator.copy()))
    output = np.divide(numerator, denominator[:, None],
                       out=np.zeros_like(numerator),
                       where=denominator[:, None] > 0)
    return (output, trace) if return_trace else output

q = np.array([[1.0]])
k = np.array([[0.0], [np.log(2.0)], [np.log(3.0)]])
v = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
output, trace = attention_online(q, k, v, 2, scale=1.0,
                                 return_trace=True)
assert np.allclose(output, [[5/3, 7/6]])
assert np.allclose(output, attention_dense(q, k, v, scale=1.0)[0])
~~~

### 1. 为什么初始最大值是负无穷

空集合没有普通有限最大值，用负无穷作为最大值归约的中性状态。初始l与u为零。当第一个有效块到来时，旧缩放因子exp(负无穷减有限数)为零，旧状态自然不贡献任何值。若初始m设为零，对全负分数仍可能得到数学等价表示，但会失去“m就是已见最大值”的清晰不变量，也可能使数值范围更差。

当一行在当前与过去所有块中都无有效key，updated_maximum仍为负无穷。直接计算旧m减新m会得到NaN，所以safe_maximum将这种情况替换为零；旧因子成为零，无效权重仍为零，l与u保持零。后续一旦出现有效key，状态可正常开始累计。

### 2. 每个广播的方向

old_factor长度为Nq，乘denominator直接逐行缩放；乘numerator时增加末尾长度一轴，才能让同一query的所有Dv分量使用相同因子。safe_maximum增加末尾轴后从每个query的所有当前key分数中减去。两个广播都沿行共享，却作用于不同宽度的矩阵。

weights乘当前V块，是一个Nq乘块长与块长乘Dv的矩阵乘法。这里的归约轴是当前key块，不是特征D。前一个QK乘法归约D，后一个PV乘法归约key块长，二者虽然都叫矩阵乘法，数据角色完全不同。

### 3. Trace为什么必须copy

maximum、denominator与numerator会在后续轮次更新。如果trace只保存可变数组引用，某些写法可能让所有记录最后都指向同一份最终状态，失去历史意义。copy为教学调试保存独立快照，代价是额外CPU内存；实际高性能路径不应默认启用完整trace。

本函数一次处理所有query，主要用于展示key方向在线合并。真实GPU实现通常再对query分块，使当前Q块、输出累加器和统计量能适应片上资源。数学状态不变，资源组织改变，下一课将把这层映射明确展开。

## 第六课：Query块、Key块与GPU数据流

### 1. 一个program负责一组query

设query块高为BQ，key块宽为BK，特征宽为D，value宽为Dv。当前Q块形状BQ乘D，当前K块BK乘D，分数块BQ乘BK，V块BK乘Dv，输出累加器BQ乘Dv。每个query行各有一个m与l，因此统计量长度为BQ。

把完整Nq乘Nk分数矩阵变成每轮BQ乘BK临时块，能显著减少需要保存的中间数据。Q块可在多个key轮次中复用，输出累加器在整个key循环期间保持活跃。K与V块依次装入，消费完成后可复用对应缓冲。

### 2. 一个具体资源账本

假设BQ为十六、BK为三十二、D为六十四、Dv为六十四。Q块有一千零二十四个输入元素，K与V各有二千零四十八个元素，分数块五百一十二个元素，输出累加器一千零二十四个标量。若累加器使用float32，仅输出累加状态就需要四千零九十六字节的逻辑存储。

这些逻辑元素不一定全部同时以独立数组存在，编译器可能分布在寄存器、shared或指令片段中；也可能因资源不足而spill。资源账本的作用是预测压力和比较方案，不是凭源代码数组shape就宣布实际寄存器占用。最终仍要看编译结果与设备测量。

BQ增加可以让同一K/V块服务更多query，但输出累加器与分数临时块也变大。BK增加减少循环轮次，却增加当前块数据和临时状态。最佳选择取决于形状、精度、硬件和实现方式，不是单纯越大越好。

### 3. 两个矩阵乘法之间还有归约与缩放

每轮先计算QK转置，加入scale与mask，得到新最大值；随后计算指数权重、分母更新与旧输出缩放；最后执行权重乘V并累加。这个流程混合矩阵指令、逐元素运算和行归约，因此不是简单把两个独立GEMM拼接就能获得理想性能。

高性能实现可能改变部分步骤顺序或保存规范化输出而不是未规范化分子，但必须能恢复同一不变量。阅读不同代码时，不要仅凭变量名m、l、acc判断含义；确认acc存的是u、u/l还是带其他缩放的量，再推导更新式是否匹配。

### 4. 一个二维tile的地址与mask完整轨迹

取Nq为三、Nk为五、D为二、Dv为三，query块高二、key块宽三。第一个query块覆盖全局query零、一，第二个query块覆盖二、三，其中三是padding。第一个key块覆盖零、一、二，第二个覆盖三、四、五，其中五是padding。

Q连续行优先，query二的两项地址为四、五。K中key四的两项地址为八、九。V中key四的三项地址为十二、十三、十四。输出query二的三项地址为六、七、八。D与Dv不同，因此K与V行stride不同；把二者都写成D会在这个例子中立即读错value。

在第二query块、第二key块里，逻辑分数tile为二乘三。只有第一行对应真实query二，只有前两列对应真实key三、四。若再使用方形位置式因果关系j≤i，则这两个真实key也都在query二之后，整块对真实query没有有效贡献。不能因为物理索引有效就忘记因果条件。

如果query二代表绝对位置四，则key三、四都可见，情况又不同。这说明“尾块mask”和“语义mask”要分开推导：前者保护内存边界，后者定义注意力范围。两者最后可以合并成布尔条件，但调试时应分别打印，避免把数学错误误诊为越界。

无效query行可以仍参与向量化计算，但不能写越界输出，也不能让其统计量污染有效行。每行归约应独立，若错误把整个二维score tile做全局最大值，padding行中的数值或另一query的最大值就会改变有效行状态。用两个query分数幅度差异很大的测试可定位这种轴错误。

### 5. 分数块为何通常不写到全局内存

当前score tile只在计算最大值、指数和以及PV局部乘法期间有用。消费完成后，下一key块会替代它。因此理想实现让它停留在寄存器、shared或编译器组织的片上片段中，不把每个tile都写成全局矩阵再读回。

但“逻辑上不保存完整矩阵”不保证编译器一定没有spill。如果tile过大，寄存器压力可能导致部分临时状态进入local memory，实际仍产生设备内存访问。检查性能时要结合编译资源与profiler，而不是仅凭Python代码没有创建大tensor就宣称所有中间值都在片上。

同样，完整概率矩阵没有显式输出，并不代表工作内存完全为零。可能存在分区输出、统计量、调度信息、梯度partial或调试缓冲。报告显存节省时，应列出实际分配与保存状态，而不是只比较一个理论上的N平方矩阵。

### 6. Q驻留与KV遍历的选择

常见组织是一个query块遍历全部key块，保持Q与输出累加器活跃。这样dV方向的前向不需要跨query合并，因为每个输出行由一个任务完整负责。另一种组织可以让多个任务分别处理key分区，最后合并在线状态，增加并行度但增加partial输出。

在解码中Nq可能很小，而Nk很长，按query块划分的任务数可能不足。这时key分区能提供更多并行工作，合并成本则相对输出宽度和分区数增长。是否值得需要看具体shape与设备，不能因为训练长序列实现采用一种调度，就认为单token解码也应完全照搬。

当batch或head数量很多时，即使每个query很少，总任务数也可能足够。因此评估并行度时要把batch、head与query块一起考虑，而不是只看一个注意力矩阵。K17的持久化与调度会进一步讨论这种任务数量与硬件资源的关系。


## 第七课：因果Mask与矩形注意力

### 1. 方形因果注意力

自回归训练中，位置i通常只允许看位置j不大于i的key。对方形N乘N注意力，mask为j≤i，含对角线。每个query至少能看自身，所以正常非空序列不会出现全mask行。若改成严格小于，则第一行全mask，需要合同处理。

将mask放在softmax前，使不可见分数不参与最大值、分母或分子。不能先计算普通attention再把输出某些位置清零，因为一个输出已经混合许多key贡献，无法用单个后处理恢复正确可见集合。

### 2. 矩形时“causal=True”不够明确

当Nq与Nk不同，query行号i未必等于它在完整序列中的绝对位置。带KV cache的解码中，一个新query可能处于长序列末尾；若仍使用j≤i且i从零开始，就只允许看第一个key，显然错误。必须说明query对应的绝对位置或对齐方式。

本章推荐直接传query_positions与key_positions，以key位置不大于query位置定义因果mask。这样交叉长度、缓存前缀与分块query都能明确表达。若选择上左对齐或下右对齐的特殊约定，应写在接口合同中，不要把某个框架版本的默认行为默认为普遍标准。

~~~python
def causal_mask_from_positions(query_positions, key_positions):
    qp = np.asarray(query_positions, dtype=np.int64)
    kp = np.asarray(key_positions, dtype=np.int64)
    if qp.ndim != 1 or kp.ndim != 1:
        raise ValueError('one-dimensional positions required')
    return kp[None, :] <= qp[:, None]

mask = causal_mask_from_positions([3, 4], [0, 1, 2, 3, 4])
assert np.array_equal(mask,
    [[True, True, True, True, False],
     [True, True, True, True, True]])
~~~

### 3. Padding与因果条件需要同时满足

有效key条件通常包括物理索引小于Nk、样本真实长度范围、因果位置关系，以及可能的窗口或自定义稀疏条件。这些条件取逻辑与。无效query输出位置则在最终store时屏蔽，不能只靠key mask处理。

例如batch中某样本真实长度为三，但张量padding到五，query位置二只能看key零、一、二。即使因果条件允许某些padding位置，长度mask仍必须排除。把padding的K或V设零不能替代mask，因为零key可能得到非零softmax权重，从而改变分母。

### 4. 一个受限但完整可读的CUDA在线前向

下面让一个线程负责一个query，顺序遍历所有key。它不使用Tensor Core，也没有跨线程复用，因此不是高性能实现；它的作用是把在线状态直接翻译成设备代码，帮助你验证数学状态、地址、类型与边界。限定Dv不超过模板常量MAX_DV，输入有限，所有矩阵连续行优先，query与key的位置由显式数组给出。

~~~cpp
template<int MAX_DV>
__global__ void online_attention_scalar(
    const float* q, const float* k, const float* v,
    const int* query_position, const int* key_position,
    float* out, int nq, int nk, int d, int dv,
    float scale, bool causal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nq) return;
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    float numerator[MAX_DV];
    for (int r = 0; r < MAX_DV; ++r) numerator[r] = 0.0f;
    for (int key = 0; key < nk; ++key) {
        if (causal && key_position[key] > query_position[row])
            continue;
        float score = 0.0f;
        for (int feature = 0; feature < d; ++feature)
            score += q[row * d + feature] * k[key * d + feature];
        score *= scale;
        float updated = fmaxf(maximum, score);
        float old_factor = expf(maximum - updated);
        float new_weight = expf(score - updated);
        denominator = old_factor * denominator + new_weight;
        for (int r = 0; r < dv; ++r)
            numerator[r] = old_factor * numerator[r]
                         + new_weight * v[key * dv + r];
        maximum = updated;
    }
    for (int r = 0; r < dv; ++r)
        out[row * dv + r] = denominator > 0.0f
                           ? numerator[r] / denominator : 0.0f;
}
~~~

wrapper必须在launch前验证零小于D、零不大于Dv且Dv不超过MAX_DV，验证分配大小、dtype和设备一致。模板数组容量不会自动限制运行时dv，若dv更大，循环会越界写线程局部存储。这个错误可能表现为结果异常而不是立即报清晰异常，所以接口检查不可省略。

这里每次只加入一个有效key，当前score有限时updated也有限，因此第一次有效key的old_factor是零；无效key直接continue，不改变状态。若所有key都被跳过，denominator保持零，最后返回零。与向量化块版本相比，它通过控制流避免构造全mask块的负无穷最大值，二者安全处理路径不同但合同相同。

每个线程私有numerator数组逻辑上长度MAX_DV，编译器可能将其放在寄存器，也可能因动态索引与资源压力产生local memory访问。不能把源代码中“局部数组”直接等同于高速寄存器。这个版本的目的在于可读性，真正优化时需要检查编译报告并改变数据组织。

query_position和key_position只有causal为真时才会读取，但wrapper仍应明确是否允许非因果路径传空指针。为了减少初学者接口分支，最简单做法是始终分配合法位置数组。更成熟的API可以专门处理可选参数，并通过编译期特化消除不需要的路径。

没有block内屏障，因此超界query线程提前返回和不同线程跳过不同key都是安全的。若以后改成一个block协作一个query，不能继续随意让线程按不同条件退出；需要重新分析所有集体操作。这是从标量正确版本向并行优化版本迁移时必须重新证明的部分。

### 5. 这个CUDA版本如何验收而不冒充性能实现

先使用三个query、五个key、D为二、Dv为三的小输入，比较全部九个输出。然后测试非因果、显式位置因果、全不可见query与空key。每次调用后检查launch错误，并在读取结果前等待设备完成。只编译通过说明语法与部分静态约束成立，不说明地址、mask与数值正确。

计时前不要用这个版本与高度优化库比速度后得出在线算法无效。它故意重复读取K与V，没有矩阵指令、没有共享复用，也没有流水线。它回答“在线状态能否在设备上正确执行”，高性能版本还要回答“怎样把同一数学计算映射到硬件”。这两个阶段都重要，但证据含义不同。

一旦这个版本通过，可以每次只增加一种复杂度：先让多个query组成tile复用KV，再引入块内矩阵乘法，再优化流水线。每步保留CPU与标量GPU基线，出现错误就用相同输入比较中间状态。不要在一次改动中同时更换mask语义、精度与线程分工。


## 第八课：前向存什么，反向为什么要重算

### 1. 反向需要概率，但不必保存全部概率

普通自动微分实现可能保存完整P用于反向。在线前向不保存Nq乘Nk矩阵，因此反向可以重新计算分数块，并用前向保存的归一化统计量恢复概率。这样增加一些计算，换取减少中间显存与读写。

可保存每行log-sum-exp，记L=m+log(l)。对有效位置，Pᵢⱼ=exp(Sᵢⱼ−Lᵢ)。保存L只需要每query一个标量。全mask行l为零，L可定义为负无穷，但重建时必须用有效行与mask控制，避免负无穷相减。接口也可以保存m和l两份统计量，代价略不同。

### 2. 重算与checkpoint的关系

重算是一种计算换内存的策略。这里重算的是局部分数与概率，而不是盲目重跑整个模型。它利用注意力结构知道反向真正需要哪些状态，因此可以精细控制临时数据生命周期。理解这个例子有助于之后理解激活checkpoint，但两者的实现粒度与接口不完全相同。

重算必须使用与前向一致的输入、scale、mask和数值策略。若前向使用某种低精度乘法、反向重算采用另一种策略，概率可能出现差异，影响梯度。训练实现需要定义可接受误差，并确认保存输入的生命周期与版本没有被原地修改破坏。

### 3. Dropout会增加状态要求

若注意力概率还经过dropout，反向必须知道前向保留了哪些位置以及缩放规则。可以保存mask，也可以使用可重现的随机数状态与索引映射重建。仅保存一个随机seed而不固定计数器、块划分和消耗顺序，未必足以在不同执行组织下恢复同一mask。

本章完整代码不包含dropout，以便把在线归一化与梯度推导讲清楚。扩展时应把随机掩码当作独立状态合同，测试前反向使用相同mask，并区分结构性attention mask与随机dropout mask。两者都叫mask，但前者改变归一化可见集合，后者通常作用于归一化后的概率。

## 第九课：反向传播完整推导

### 1. 从O=PV得到dV与dP

设上游梯度G与输出O同shape，为Nq乘Dv。对value，dV=P的转置乘G，shape为Nk乘Dv。对概率，dP=G乘V的转置，shape为Nq乘Nk。它们都是矩阵乘法，但归约轴分别是query和value特征。

看单个位置，O[i,r]=ΣⱼP[i,j]V[j,r]。因此V[j,r]的梯度是所有query的G[i,r]P[i,j]之和；P[i,j]的梯度是所有value特征r的G[i,r]V[j,r]之和。这个逐项推导能帮助你在分块反向中判断哪些输出需要跨query累加。

### 2. 通过Softmax得到dS

K05已经推导每行softmax反向：dS=P·(dP−ΣⱼP·dP)，其中乘法逐元素，归约结果按行广播。记Dᵢ为ΣⱼPᵢⱼdPᵢⱼ，则dSᵢⱼ=Pᵢⱼ(dPᵢⱼ−Dᵢ)。

D还有一个非常有用的等价表达：Dᵢ=ΣᵣGᵢᵣOᵢᵣ。将dP=GV转置代入，再交换求和，内部ΣⱼPᵢⱼVⱼᵣ恰好是Oᵢᵣ。这样D可以只用前向输出和上游梯度计算，每行一个标量，不必先构造完整dP。

### 3. 通过缩放点积得到dQ与dK

S=sQK转置，所以dQ=s dS K，dK=s dS的转置 Q。scale必须同时出现在两个梯度中。若前向已把Q预缩放，反向仍要追踪对原始Q的链式关系，不能因为代码中S没有显式乘scale就忘记梯度缩放。

无效mask位置P为零，所以dS也为零。全mask行按零输出合同，其输入相关梯度为零。value梯度只来自有效概率位置，不会因为无效value恰好是零就自动满足正确mask逻辑；测试应使用非零且各不相同的value。

### 4. 两key单query手算

取P为四分之一、四分之三，V为一、三，输出为二点五，上游G为二。dP为二、六，D等于G乘O为五。dS分别为四分之一乘负三与四分之三乘一，得到负零点七五、正零点七五，和为零。

若D特征为一，Q为一，K为零与ln三，scale为一，dQ为负零点七五乘零加零点七五乘ln三，即零点七五ln三。dK为负零点七五与正零点七五。dV为P乘G，得到零点五与一点五。这个例子可完整核对所有三个输入梯度。

### 4. 用微分符号重新推一遍反向

将矩阵内积定义为所有对应元素乘积之和，损失变化可写为dL=〈G,dO〉。因为O=PV，变化dO=dP·V+P·dV。利用矩阵内积的转置关系，第一项变成〈GV转置,dP〉，第二项变成〈P转置G,dV〉，于是得到前面的dP与dV公式。

softmax的微分对每行满足dpⱼ=pⱼ(dsⱼ−Σₖpₖdsₖ)。代入损失变化Σⱼgⱼdpⱼ，交换第二项求和，得到每个dsⱼ的系数pⱼ(gⱼ−Σₖpₖgₖ)。这个推导没有构造整个雅可比矩阵，直接得到向量雅可比积，解释为什么实际反向只需要一个行归约。

分数S=sQK转置的变化是s(dQ·K转置+Q·dK转置)。再次把损失变化中的矩阵内积整理到dQ与dK前，得到s dS K与s dS转置 Q。这样前向三个运算的反向被完整连接，任何scale、transpose或归约轴都能追溯到一个明确的微分步骤。

理解这种推导方法后，你可以独立处理加性attention bias。若S=sQK转置+B，则对完整B的梯度就是dS；若B按某些轴广播，其梯度还需沿这些广播轴归约。不能直接把dS复制给一个只有一维的bias参数，shape与共享关系决定最终归约。

### 5. 前向输出误差如何进入反向D项

D=sum(O·G)使用实际前向输出。如果前向O有小误差，D也会受到相应扰动，随后每个dS位置通过P乘该扰动传播。这个关系说明前向和反向的误差并非完全独立，也解释为什么高精度参考与低精度设备实现比较时，需要明确前向保存值采用何种精度。

测试可以分两组：第一组用高精度参考O计算反向，检查公式与索引；第二组用目标前向产生的O计算反向，检查完整链路。若第一组通过而第二组误差明显增大，可能需要调查前向精度、输出类型或统计量保存，而不是只盯反向矩阵乘法。

同时不要为了让反向测试更好看，悄悄换成与实际训练不同的O来源。真实框架集成的验收应使用实际前向返回并保存的状态，额外高精度组仅用于定位。报告中明确区分这两类实验，才能让结果具有可复现含义。


## 第十课：分块重算反向的完整CPU实现

~~~python
def attention_backward_recompute(q, k, v, output, grad,
                                 block_size=2, mask=None, scale=None):
    q, k, v = attention_inputs(q, k, v)
    output = np.asarray(output, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    nq, d = q.shape
    nk, dv = v.shape
    if output.shape != (nq, dv) or grad.shape != output.shape:
        raise ValueError('output or gradient shape mismatch')
    if block_size <= 0:
        raise ValueError('positive block size required')
    scale = 1 / np.sqrt(d) if scale is None else float(scale)
    allowed = np.ones((nq, nk), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if allowed.shape != (nq, nk):
        raise ValueError('mask shape mismatch')
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv_result = np.zeros_like(v)
    if nk == 0:
        return dq, dk, dv_result
    # Teaching version obtains row statistics without storing full P.
    maximum = np.full(nq, -np.inf)
    denominator = np.zeros(nq)
    for begin in range(0, nk, block_size):
        end = min(begin + block_size, nk)
        scores = (q @ k[begin:end].T) * scale
        valid = allowed[:, begin:end]
        masked = np.where(valid, scores, -np.inf)
        updated = np.maximum(maximum, masked.max(axis=1))
        safe = np.where(np.isfinite(updated), updated, 0.0)
        factor = np.exp(maximum - safe)
        weights = np.where(valid, np.exp(masked - safe[:, None]), 0.0)
        denominator = factor * denominator + weights.sum(axis=1)
        maximum = updated
    row_valid = denominator > 0
    logsumexp = np.zeros(nq)
    logsumexp[row_valid] = maximum[row_valid] + np.log(denominator[row_valid])
    row_dot = np.sum(output * grad, axis=1)
    for begin in range(0, nk, block_size):
        end = min(begin + block_size, nk)
        scores = (q @ k[begin:end].T) * scale
        valid = allowed[:, begin:end] & row_valid[:, None]
        masked = np.where(valid, scores, -np.inf)
        probability = np.exp(masked - logsumexp[:, None])
        dp = grad @ v[begin:end].T
        ds = probability * (dp - row_dot[:, None])
        dq += scale * (ds @ k[begin:end])
        dk[begin:end] += scale * (ds.T @ q)
        dv_result[begin:end] += probability.T @ grad
    return dq, dk, dv_result
~~~

### 1. 为什么教学版先重建统计量

实际前向可以保存L，反向直接读取，不必再执行第一遍统计量循环。这里为了让函数独立、输入合同清楚，先用在线过程重新获得m与l，然后在第二遍重建每个概率块。它展示的是数学与数据流，不是最优训练实现。

进一步工程化时，可以让前向返回output与logsumexp，并在autograd上下文保存必要输入引用和统计量。这样反向减少一遍分数计算。保存哪些张量、是否允许原地修改、何时释放，都属于K10的框架集成问题。

### 2. dQ为何累加，dK与dV为何按块写

当前代码一次处理全部query，循环key块。每个key块只覆盖一部分对dQ的贡献，所以dq必须跨块累加。当前dk切片与dv切片对应当前key范围，已包含全部query的贡献，因此不同key块输出互不重叠。

真实GPU可能同时对query分块，此时同一个key的dK与dV还会收到多个query块贡献，需要额外归约或不同任务分配。不能把CPU中“切片互不重叠”的结论原封不动搬到二维并行网格。输出梯度所有权必须重新画清楚。

### 3. row_dot使用前向输出的前提

D=sum(O·G)依赖O与当前QKV、mask和scale一致。若调用者传入错误输出、前向输出被修改，或者前反向精度策略差异过大，这个等价关系会失效或出现额外误差。测试应使用同一次前向的输出，并在框架层保护必要状态。

这里不通过保存完整P来检查反向，而是与中央差分以及独立dense反向对照。两个实现都使用同一个错误公式时，相互比较可能通过；手算、数值差分和独立表达式三种证据结合，能更有效避免共同错误。

### 4. 分块反向的两种任务分配

一种方案让每个query块完整遍历key块，独占自己的dQ输出。它自然积累dQ，但对dK与dV产生跨query块的部分贡献，需要原子或二阶段归约。另一种方案让每个key块完整遍历query块，独占dK与dV；此时dQ贡献分散，需要另一路处理或额外归约。

还可以分别运行两个Kernel：一个按query计算dQ，一个按key计算dK与dV。这样会重算一些分数与概率，但减少原子冲突与复杂同步。是否更快取决于shape、精度、带宽和计算资源。增加计算并不必然更慢，减少同步与内存写入可能抵消重算成本。

在设计任务分配时，先写每个输出梯度的求和轴。dQ对key归约，dK与dV对query归约。谁在任务内部覆盖完整归约轴，谁就能成为该输出的唯一写者。这个规则比记住某个开源Kernel的grid排列更通用。

### 5. 一个partial梯度缓冲的容量例子

设query分成四块，Nk为一百二十八，D为六十四，Dv为三十二。若每个query块各存一份dK与dV partial，float32元素数为四乘一百二十八乘九十六，共四万九千一百五十二，字节数为十九万六千六百零八。之后还要读取这些partial并归约。

这个数量对小实验可能不大，但随query分块数与key长度增长。高性能实现会在并行度、临时显存、原子竞争和重算之间权衡。若只说“不保存N平方概率矩阵”，却忽略大量梯度partial，可能高估实际显存节省。

二阶段归约的初始化与生命周期也要明确。每个partial必须完整写入，不能读取未初始化缓冲；最终归约只处理真实有效块数，不能把分配容量当成有效数量。使用全零上游梯度时所有输出梯度应为零，是检查未初始化partial污染的一个有效测试。


## 第十一课：有限差分与结构性检查

### 1. 构造标量损失

固定上游梯度G，定义损失为输出与G逐元素相乘再求和。对Q、K、V各元素分别做中央差分，得到数值梯度。使用float64小矩阵、合理步长和不同正负权重，避免把精度噪声或特殊常量损失误认为实现正确。

~~~python
def finite_difference(function, value, step=1e-5):
    value = np.asarray(value, dtype=np.float64).copy()
    gradient = np.empty_like(value)
    for index in np.ndindex(value.shape):
        old = value[index]
        value[index] = old + step
        positive = float(function(value))
        value[index] = old - step
        negative = float(function(value))
        value[index] = old
        gradient[index] = (positive - negative) / (2 * step)
    return gradient

rng = np.random.default_rng(7)
q_test = rng.normal(size=(2, 3))
k_test = rng.normal(size=(4, 3))
v_test = rng.normal(size=(4, 2))
g_test = rng.normal(size=(2, 2))
mask_test = np.array([[True, False, True, True],
                      [False, True, True, False]])
o_test = attention_online(q_test, k_test, v_test, 3, mask_test)
dq, dk, dv = attention_backward_recompute(
    q_test, k_test, v_test, o_test, g_test, 3, mask_test)
loss = lambda qv, kv, vv: np.sum(
    attention_dense(qv, kv, vv, mask_test)[0] * g_test)
numeric_q = finite_difference(lambda z: loss(z, k_test, v_test), q_test)
numeric_k = finite_difference(lambda z: loss(q_test, z, v_test), k_test)
numeric_v = finite_difference(lambda z: loss(q_test, k_test, z), v_test)
assert np.allclose(dq, numeric_q, rtol=1e-5, atol=1e-7)
assert np.allclose(dk, numeric_k, rtol=1e-5, atol=1e-7)
assert np.allclose(dv, numeric_v, rtol=1e-5, atol=1e-7)
~~~

### 2. 不变量能检查哪些错误

每个正常概率行和为一，全mask行和为零；无效位置概率与分数梯度为零；softmax分数梯度每行和接近零；把同一常数加到某行全部有效分数，输出应不变。不同key块大小得到的结果应在合理误差内一致。

但这些性质不能独立证明完整正确。一个总输出零的实现可以满足许多“无效位置为零”检查；一个错误排列value的实现可能仍满足概率行和。必须同时比较手算结果、独立参考和边界不变量，让不同测试覆盖不同失效模式。

## 第十二课：四个常见故障的具体定位

第一类故障是只更新分母、不缩放旧分子。使用第一块最大值小、第二块最大值大的手算例子，检查第二轮u是否乘旧缩放系数。若l正确而输出偏大，往往可以直接从trace发现u仍处于旧尺度。

第二类故障是全mask块导致NaN。检查是否出现负无穷减负无穷，尤其是第一块对某些query完全不可见、后面才有有效key的情况。不能只测试方形因果且首块总含可见key的路径，否则特殊假设会掩盖通用接口错误。

第三类故障是矩形causal错位。用一个query代表绝对位置四、五个key代表零至四。正确允许全部五个key；若实现只允许key零，说明把局部query行号误当绝对位置。修复需要明确位置合同，不能简单放宽某个mask常数。

第四类故障是反向少乘scale或错用归约轴。用D不等于一的输入，分别比较dQ、dK、dV。若dV正确而dQ和dK按相近比例偏差，优先检查scale；若只有某些key块错误，检查块边界、切片与跨query累加所有权。

### 4. GQA与MQA增加的是头映射合同

多头注意力为每个head执行相似计算。GQA允许多个query head共享一个KV head，MQA是KV head数量很少的特殊组织。计算公式本身不需要改成新softmax，关键是query head如何映射到对应K与V的头，以及地址stride如何计算。

例如八个query head、两个KV head，常见分组映射让query head零至三使用KV head零，四至七使用KV head一。这个映射需要接口明确，不能把head号直接对KV head数取模而得到交错分组，除非模型布局确实采用那种规则。两种映射shape都可能合法，却会产生不同模型结果。

反向中共享KV head收到多个query head的梯度贡献，因此dK与dV还要沿共享head组累加。前向仅通过重复视图或索引复用就能表达，反向却增加归约责任。这个例子再次说明读取别名与写入累加不是同一个问题。

### 5. 窗口、加性偏置与块稀疏的区别

滑动窗口mask限制query只看位置附近的key，可以在score计算后屏蔽，也可以在调度层跳过完全不可见的key块。后者减少计算，但必须证明跳过的块确实对该query块所有行都不可见；对边界块仍可能需要逐元素mask。

加性偏置改变可见位置的分数，例如相对位置偏置。它应在最大值与指数之前加入，因为它属于logit定义。把偏置加到最终输出或概率上不是等价操作。若偏置包含负无穷来表示屏蔽，要与有限输入合同及全mask处理协调。

块稀疏调度把可见关系组织为块列表，可能避免遍历所有key块。在线状态仍可按列表顺序合并，但列表不能重复同一个key块，否则重复计入权重；也不能遗漏应可见块。验证稀疏列表时，先在小尺寸展开成完整布尔mask，与dense参考逐项比较。

### 6. 内存复杂度与计算复杂度分开写

对单头普通稠密注意力，分数计算和value聚合都随query数乘key数乘特征宽度增长。在线算法主要减少需要写入全局内存的二次规模中间张量，并通过分块提高数据利用。不能把“不存N平方矩阵”误写成“计算只需线性时间”。

如果引入窗口或块稀疏，允许配对数量减少，计算量才可能相应下降；如果引入线性注意力，则数学表达式或近似假设发生变化。将这些方法放在同一张对比表时，应分别列出数学定义是否改变、配对数量、保存状态和误差来源，而不是统一称为“快速注意力”。


## 第十三课：十二道练习与完整答案

### 练习一：列出全部shape

题目：Q为三乘四，K为五乘四，V为五乘二。答案：分数与概率为三乘五，输出三乘二，每行最大值与分母为三乘一。前向两个乘法归约维分别为四与五。

### 练习二：手算两块状态

题目：分数零、ln二、ln三，value一、二、四，先处理前两项。答案：第一块m为ln二，l为一点五，u为二点五；合并第三项时旧因子二除以三，新l为二，新u为十七除以三，最终输出十七除以六。直接概率加权也得到相同值。

### 练习三：证明局部概率不能拼接

答案：每个非空局部softmax各自和为一，两块拼接总和二，缺少两组在全体中的权重比例。保存m与l后可恢复组间比例；只保存局部归一化概率通常丢失必要尺度信息。

### 练习四：构造全mask再有效的情况

~~~python
q_case = np.ones((1, 1))
k_case = np.array([[1.], [2.], [3.]])
v_case = np.array([[10.], [20.], [30.]])
mask_case = np.array([[False, False, True]])
actual = attention_online(q_case, k_case, v_case, 2, mask_case, 1.0)
assert np.array_equal(actual, [[30.]])
~~~

第一块对该query没有任何有效key，状态应保持空；第二块只有一个有效key，最终概率为一。若产生NaN，优先检查safe_maximum而不是调小输入。

### 练习五：矩形因果mask

题目：query绝对位置为二、三，key位置零至四。答案：第一行允许零、一、二，第二行允许零、一、二、三，第五个key对两行都不可见。使用本章位置函数可以完整生成并断言该布尔矩阵。

### 练习六：块大小不变性

~~~python
reference = attention_dense(q_test, k_test, v_test, mask_test)[0]
for block in (1, 2, 3, 7):
    actual = attention_online(q_test, k_test, v_test, block, mask_test)
    assert actual.shape == reference.shape
    assert np.allclose(actual, reference, rtol=1e-12, atol=1e-12)
~~~

块大小七大于key数，测试单块路径；块三产生尾块；块一让最大值更新发生最频繁。三类路径结合比只测一个整除块大小更有效。

### 练习七：解释D等于输出与上游梯度内积

答案：把dP中对value特征的求和代入ΣP·dP，交换key与value特征求和顺序，内部ΣP·V恰好是输出O。这个恒等式使反向每行的softmax修正项可在不保存完整P的情况下计算。

### 练习八：检查两key梯度

题目：P为四分之一、四分之三，V一、三，G二。答案：O二点五，dP二、六，D五，dS负零点七五、正零点七五，dV零点五、一点五。dS和为零是附加检查，不代替逐项结果。

### 练习九：value全部相同会怎样

若所有有效value都等于同一向量v，正常行输出应为v，与分数分布无关。因此对Q与K的梯度应接近零，value梯度仍由概率分配。全mask行按本章合同输出零，需单独区分。

### 练习十：空key合同

~~~python
empty_k = np.zeros((0, 3))
empty_v = np.zeros((0, 2))
empty_output = attention_online(q_test, empty_k, empty_v, 2)
assert np.array_equal(empty_output, np.zeros((2, 2)))
~~~

这验证循环零次执行时，零分母路径仍返回明确定义的零输出。若生产接口拒绝空key，应将测试改为验证异常，而不是保留两套互相冲突的行为。

### 练习十一：反向输出所有权

题目：两个query块都遍历同一key块，能否普通store各自dK。答案：不能，各query块产生的是同一dK的部分和。可以独立partial再归约、原子累加或按key分配任务并在内部遍历query。每种方案都有不同资源、确定性与性能代价。

### 练习十二：完整误差测试矩阵

答案应包含Nq不等于Nk、D不等于Dv、块大小大于与不整除Nk、部分mask、全mask、首块全mask后块有效、位置明确的矩形causal、全相同value、普通随机与大公共分数偏置。前向比较dense参考，反向比较中央差分，所有比较先检查shape与有限性。

## 第十四课：指数底数、随机状态与可复现审计

### 1. 自然指数与二为底指数不能混用尺度

数学softmax通常写exp，也就是以自然常数为底的指数。某些底层实现为了适应指令或编译优化，使用以二为底的指数函数。二者可以精确转换：exp(x)=二的x除以ln二次方，也就是exp2(x乘log₂e)。如果分数先乘了log₂e，后续最大值与归一化统计量也必须处于相同单位。

假设原始分数为零与ln二，自然指数权重是一与二。若直接把它们送入exp2，第二个权重变成二的ln二次方，并不等于二。只有先把ln二乘log₂e，变成一，再用exp2，才恢复正确权重。这个两元素例子可以快速识别指数底数转换遗漏。

保存的log-sum-exp也要注明底数。如果前向保存的是以二为底的对数分母，反向却用自然指数直接计算score减该值，就会混合单位。阅读底层实现时，变量名lse未必足以说明它采用自然对数还是二进制对数，必须看生成与消费的位置。

使用不同指数近似指令可能改变数值误差与性能。教学参考采用NumPy float64自然指数，GPU目标可能采用float32及不同近似策略，比较时允许合理误差，但不能用“近似”解释数量级错误。先用精确可手算的零与ln二例子核对单位，再讨论小幅数值差异。

### 2. 一份能发现共同错误的测试设计

如果dense参考与online实现都调用同一个错误mask生成函数，它们可能完全一致却都违反任务语义。因此mask本身需要独立手写小矩阵预期。例如query位置三、四，key位置零至四，第一行最后一项必须不可见，第二行全部可见。这个预期不应由待测函数再次生成。

同理，如果前向与反向都遗漏scale，有限差分可能仍通过，因为它检查的是当前错误前向的梯度。要另外用已知scale的手算或独立参考确认前向数学定义。梯度检查证明“反向匹配这个前向”，不自动证明“这个前向符合目标模型”。

还可以做因果干预测试：保持query与可见key/value不变，只大幅修改未来不可见位置，当前输出应不变。若输出发生变化，说明未来信息泄漏。这个测试直接对应因果语义，比只检查一个布尔mask数组更接近最终行为，但仍应配合地址和shape检查。

~~~python
q_intervene = np.array([[0.2, -0.3]])
k_intervene = np.array([[1., 0.], [0., 1.], [2., 2.]])
v_intervene = np.array([[1., 2.], [3., 4.], [5., 6.]])
visibility = causal_mask_from_positions([1], [0, 1, 2])
before = attention_online(q_intervene, k_intervene, v_intervene,
                          2, visibility)
k_changed = k_intervene.copy()
v_changed = v_intervene.copy()
k_changed[2] = [1000., -1000.]
v_changed[2] = [-999., 777.]
after = attention_online(q_intervene, k_changed, v_changed,
                         2, visibility)
assert np.array_equal(before, after)
~~~

这个例子修改的只是被mask排除的位置，因此正确实现的所有有效运算完全相同，适合要求精确一致。若修改的是可见位置，则输出一般应改变，但某些对称或相同value输入可能让变化抵消，所以负向控制需要选择非退化数据。测试设计也要考虑反例是否真的能区分正确与错误实现。

### 3. 统计量保存与输入版本

训练框架保存Q、K、V与输出时，通常保存的是张量对象或必要数据引用，而不一定复制全部内容。若用户在反向前原地修改这些张量，重算得到的分数就不再对应前向。成熟框架会通过版本追踪等机制检测部分情况，自定义算子必须遵循框架的保存与mutation合同。

即使没有用户显式修改，复用临时缓冲也可能让保存的统计量被后续操作覆盖。前向返回L时，应保证它拥有合适的生命周期；不能返回指向已经释放或即将重用的工作区的视图。这个问题与CUDA异步执行叠加后尤其隐蔽，K13会从stream与event角度继续讲资源释放时机。

因此在线注意力的正确性不仅包括公式，还包括状态在反向开始时仍然代表同一次前向。记录shape、dtype、scale、mask版本和随机状态，是为了让这份依赖可追溯。一个数值公式完全正确但保存状态失效的实现，仍然不是可靠的训练算子。

### 4. 阅读真实实现的路线

先找接口检查，确认支持的head维、dtype、设备、布局与mask模式；再找前向任务映射，写出一个block负责哪些query与head；然后追踪最大值、分母和输出累加器的单位。不要一开始就逐条阅读汇编或所有模板参数，否则容易失去整个算法状态。

进入反向时，先定位每行D项的计算，再找概率重建所用统计量，最后查看dQ、dK、dV分别由哪些任务写入。凡是多个任务贡献同一输出的位置，都应能找到原子、临时缓冲或独占归约策略。若找不到，就继续追踪，不能凭变量名假定框架会自动合并。

最后检查测试与基准的支持范围。某个实现可能只支持方形因果、自注意力、特定head宽度或特定精度；这些限制不使它失去价值，但你修改接口时必须重新补齐被限制排除的边界。把限制看成证明的一部分，才能安全地扩展代码，而不是把它们当成可以随手删除的断言。


### 5. 用逆向问题检验理解

给你最终输出和一份错误trace，先判断是哪一轮首次偏离不变量，而不是从最后一行代码开始猜。若最大值正确、分母错误，检查权重计算与有效集合；若两者正确、分子错误，检查旧分子缩放和value地址；若所有状态正确、输出错误，检查最终除法、广播与写回位置。

反向同样可以逆向定位。若数值差分与解析梯度只有value路径一致，优先检查softmax修正项、缩放和转置；若只有一个query块异常，检查局部到全局坐标；若结果随块大小大幅改变，检查状态合并与尾块。让每个故障假设都对应一个最小输入，调试就会从盲目改代码变成有证据的排除过程。

## 第十五课：逐步完成一个可解释注意力项目

第一阶段实现dense参考与手算trace，明确mask和空输入合同。第二阶段实现在线前向，至少用四种块大小验证，并保存一次最大值增加时的完整状态表。第三阶段实现重算反向，分别检查Q、K、V梯度，不能只测试一个输入。

第四阶段再把一个受限前向搬到GPU，先限制小特征宽度、单头与有限长度，逐项扩展而不是一次支持全部模型接口。第五阶段阅读固定版本FlashAttention源码，寻找前向统计量、输出缩放、反向D项、概率重建和梯度写者，逐个与本章公式对应。

项目完成意味着你能从一行错误输出追到某个状态或地址，能解释不保存概率为什么仍可反向，能说清自己的实现支持哪些mask和布局。真实高性能实现还包含更精细的流水线、指令布局和调度，本章为阅读与修改这些代码提供可以验证的数学基础。

最后应保存每个失败案例的原始输入、预期结果和修复原因，使它成为回归测试。优化后的实现必须重新经过这些案例；一次性能改动不能让已经明确的因果、空行或梯度合同悄悄退化。
