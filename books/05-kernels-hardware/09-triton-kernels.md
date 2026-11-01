# K09 六类算子实战：逐元素、归约、Softmax、RMSNorm、GEMM与在线注意力

本章把前面分别学过的数学、地址与Triton语义组合成一个真正可维护的算子小库。我们不是把六个函数名字列在一起，而是对每类算子回答同一组具体问题：输入输出合同是什么，哪个program拥有哪个输出，中间状态的shape与dtype是什么，尾部如何处理，哪些测试能区分公式错误和地址错误，以及怎样逐步优化而不丢失正确性。

主要实现位于labs/kernels/triton_ops.py，CPU参考位于labs/kernels/reference.py。当前设备路径限定连续CUDA float32张量，部分形状刻意设上界，不提供自动梯度。限制是为了让第一版能完整审计；扩展支持范围时必须同步修改wrapper、测试和说明。当前交付环境没有Triton与GPU，因此本章展示与逐行解释设备代码，实际GPU编译、结果和性能仍待目标环境执行。

## 第一课：先建立六类算子的共同合同

### 1. 什么叫一个可靠算子

一个算子不只是核心Kernel。它还包括输入检查、输出分配、启动配置、错误传播、数值参考、边界测试和明确的支持范围。若Kernel公式正确，但wrapper把输出分配到错误设备或错误dtype，整个算子仍然错误。若结果正确但偷偷修改输入，也可能破坏后续模型。

本卷的六类教学算子统一要求输入位于同一CUDA设备，dtype为float32，布局连续。这样可以用简单行优先地址公式，把学习重点放在各类数据流。非连续输入不是“不重要”，而是需要单独设计的接口能力；K08已经给出stride路径，后续扩展时可以逐项移植。

### 2. 类型和分配必须显式对应

输出应继承目标合同中的device与dtype，不能依赖全局默认类型。比如torch.empty((m,n),device=a.device,dtype=a.dtype)明确表示与输入相同类型；省略dtype后，若用户改变默认类型，输出可能与Kernel预期不同。这样的错误往往只在某些环境出现，难以从单次成功推断可靠性。

临时partial缓冲同样需要明确dtype。归约内部使用float32时，partial应保存float32，不能因为输入未来可能扩展成半精度就自动继承窄类型而增加一次不必要舍入。当前库限制float32，所以继承输入类型与累加合同一致；扩展类型时必须重新审查。

### 3. 统一检查函数

~~~python
def check_teaching_inputs(*tensors):
    import torch
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError('CUDA tensor required')
        if tensor.dtype != torch.float32:
            raise ValueError('float32 teaching path required')
        if not tensor.is_contiguous():
            raise ValueError('contiguous tensor required')
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError('all tensors must use the same device')
~~~

这个函数只验证共同条件，各算子仍要检查自己的shape与标量参数。softmax要求非空归约维，RMSNorm要求weight长度匹配且epsilon为正，GEMM要求内维相等，attention要求受支持的长度与特征宽度。把所有条件塞进一个庞大通用检查器反而会降低可读性。

输入检查不应吞掉所有异常后返回空结果。用户传错shape应得到明确错误，缺少GPU依赖应报告未执行，设备运行失败应保留错误信息。这三种情况不能统一标成“通过”或“自动回退成功”，否则学习者无法知道自己实际验证了哪条路径。

## 第二课：逐元素加法的完整数据流

### 1. 最小Kernel与wrapper

~~~python
import triton
import triton.language as tl

@triton.jit
def add_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    a = tl.load(A + offsets, mask=valid, other=0.0)
    b = tl.load(B + offsets, mask=valid, other=0.0)
    tl.store(C + offsets, a + b, mask=valid)

def add(a, b):
    import torch
    check_teaching_inputs(a, b)
    if a.shape != b.shape:
        raise ValueError('equal shapes required')
    output = torch.empty_like(a)
    if a.numel() != 0:
        grid = (triton.cdiv(a.numel(), 256),)
        add_kernel[grid](a, b, output, a.numel(), 256)
    return output
~~~

连续张量可以按扁平元素序列处理，输出shape仍由empty_like保留。这里不支持广播加法，a与b必须shape完全相同。若用户想加一个长度hidden的bias，不能仅依靠指针读取碰巧重复，而应明确写出广播索引或使用另一个专用算子。

N被标为constexpr会让不同长度可能生成不同特化，这是当前教学实现的选择，不是向量加法必须如此。后续可以研究把N作为运行时值，以减少特化数量，再比较编译缓存与稳态性能。任何改动都应保留同样的尾部合同。

### 2. 十个元素、四个槽的轨迹

program零读取输入零至三并写输出零至三，program一处理四至七，program二只读取和写八、九。无效槽十、十一的a与b取零，算术结果也为零，但最终不写。这个例子中无效槽没有参与跨元素归约，所以替代值不会影响合法输出。

如果输出与输入完全别名，逐元素加法可能可以设计为安全原地更新，但当前wrapper独立分配输出，避免把别名扩展混进基础合同。若未来支持重叠切片，例如输出起点比输入后移一项，就可能覆盖其他program尚未读取的数据，不能直接继承安全结论。

### 3. 它是内存流量练习

每个元素逻辑上读取两个float32并写一个float32，总计十二字节，执行一次加法。对大数组，算术工作相对少，通常应先分析内存访问与固定开销，而不是寻找复杂数学优化。实际设备流量可能受缓存影响，十二字节是逻辑张量账本，不是实测HBM事务保证。

融合后续逐元素操作可能减少中间张量写回。例如先加再乘标量，可以在同一Kernel中完成，避免单独中间输出。但融合更多输入也会增加读取与寄存器状态，需要计算完整数据流。不能把“一个Kernel”自动等同于“最少流量”或“最快”。

### 4. 把一条加法拆成地址轨迹、值轨迹和所有权轨迹

设输入A为一、二、三、四、五，B为十、二十、三十、四十、五十，BLOCK为四。第一个program的地址轨迹是A与B各读取偏移零至三，C写零至三；值轨迹是对应相加得到十一、二十二、三十三、四十四；所有权轨迹是这四个C位置只由program零负责。

第二个program的逻辑offset为四、五、六、七。地址轨迹只允许读取偏移四，其他槽没有实际内存读取；值轨迹在合法槽得到五加五十等于五十五，无效槽用零加零；所有权轨迹只写C四。三条轨迹分别回答“从哪里取”“怎么算”“谁能写”，不能用其中一条代替另外两条。

如果把program乘BLOCK写成program加BLOCK，第一个program从偏移四开始，第二个从五开始，零至三遗漏且后面大量越界。数值公式a加b仍然完全正确，但地址与覆盖错误。这个例子说明阅读算子时，不能看到核心算术正确就停止审查。

如果两个program都写C四，且恰好写同一个五十五，最终结果比较可能通过，但结构上存在重复写者。后续改成累加或输入不同步时可能失败。因此调试阶段的所有权模拟有价值，尤其是在任务重排、持久化循环和多维grid中。

### 5. 融合加法与缩放的完整成本推导

考虑Y=(A+B)乘s。分开执行时，第一个算子读A、B并写临时T，第二个读T并写Y。float32每元素逻辑流量为十二加八，共二十字节。融合后读A、B并写Y，共十二字节，标量s可在program内共享。理想减少八字节，但真实收益还受缓存、launch与计算影响。

若再加入另一个张量R做残差，融合版本还需要读取R，每元素增加四字节。不能继续沿用十二字节估计。若后续激活需要复杂指数或查表，算术成本可能变得显著。成本账本应随实际数据流更新，而不是给“融合”贴固定收益标签。

融合也改变中间结果可观察性。如果外部调用者还需要T，直接消除T会改变接口；如果autograd反向需要T或其他状态，则可能要保存或重算。推理前向中的一次流量优化，不能不加分析地搬进训练路径。K10会把这种保存与重算选择落实到框架接口。


## 第三课：归约必须明确阶段边界

### 1. 第一阶段产生partial

~~~python
@triton.jit
def partial_sum_kernel(X, PARTIAL, N: tl.constexpr,
                       BLOCK: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N,
                     other=0.0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(PARTIAL + program, total)

def reduce_sum(x):
    import torch
    check_teaching_inputs(x)
    if x.numel() == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    count = triton.cdiv(x.numel(), 256)
    partial = torch.empty((count,), device=x.device, dtype=x.dtype)
    partial_sum_kernel[(count,)](x, partial, x.numel(), 256)
    return partial.sum()
~~~

每个program只归约自己负责的BLOCK槽，得到一个partial。第二阶段明确使用框架sum，因此这个实现不是“全部融合进一个Triton Kernel”。保留这样的清晰阶段有利于学习与验收，后续再决定是否值得手写第二阶段。

空输入返回零维标量零，dtype与device明确。非空输入无论原shape如何，都按连续扁平元素求和。这个接口没有axis参数，不能把它误当成支持任意维度归约的通用sum。

### 2. 一千零三个元素的分组

BLOCK二百五十六时需要四个program。前三个各处理二百五十六项，第四个处理二百三十五项，其余二十一个槽填零。partial长度四，第二阶段把四个数相加。真实元素总数是三乘二百五十六加二百三十五，恰好一千零三。

如果将最后块的无效槽留作未定义值，局部sum会把它们加进去；如果所有program都写PARTIAL零位置，其他槽未初始化且存在写冲突。正确性需要同时验证加载中性值、partial地址和第二阶段有效长度。

### 3. 浮点误差与阶段顺序

多阶段归约与串行sum可能采用不同加法顺序，末位差异属于预期数值现象。测试应使用float64参考与合理容差，并包含K05中的非结合反例，让学习者知道“不是逐bit相同”不自动代表错误。

但数量级偏差、遗漏尾块或NaN不能轻易归因于舍入。先检查每个partial与CPU对应分段和是否一致，再检查第二阶段。这个分层比较能把错误定位到局部读取、局部归约或全局合并，而不是只得到一个最终失败数字。

### 4. 归约partial逐项核对的完整例子

用输入一到十，BLOCK四。第一个partial为一加二加三加四等于十，第二个为五加六加七加八等于二十六，第三个为九加十加零加零等于十九。最终五十五。先核对三个partial，再核对总和，比直接比较五十五更能定位错误。

若第三个partial为三十，可能是尾槽没有清零并重复读到某些旧值；若partial数组为十、二十六、未定义，可能是grid少启动一块；若只有最后一个partial有值，可能所有program写同一地址。每种症状都对应不同层次，不能统一归因于浮点误差。

在GPU调试中，可以临时让wrapper返回partial与最终结果，用小输入复制回CPU检查。这个调试接口不必成为正式公共API，但应以清楚的开关或独立函数实现，避免性能路径无意保留额外同步与主机复制。调试可观察性与生产效率可以分开管理。

### 5. 第二阶段什么时候值得手写

当partial数量很少，框架sum可能已经足够，手写第二阶段的收益主要取决于调用与调度开销。若partial很多，第二阶段也需要分块，可能形成多层归约。选择之前先测量完整算子，而不是只看第一阶段Kernel时间。

一个受限教学版本可以要求partial数量不超过某个BLOCK，用单program归约全部partial。wrapper需要验证这个上界，或者递归产生更短partial。不能假设任何输入规模都只产生几百个partial，然后在更大输入上越界或漏算。

递归归约的终止条件是当前元素数不超过目标块容量，最终输出一个标量。每层长度按向上整除缩小，空输入单独返回零。临时缓冲的生命周期持续到下一层读取完成；异步执行下不能提前释放或复用，框架分配器与stream语义需要正确配合。


## 第四课：行Softmax的完整实现与边界

### 1. 设备代码

~~~python
@triton.jit
def softmax_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns,
                     mask=columns < N,
                     other=-float('inf')).to(tl.float32)
    maximum = tl.max(values, axis=0)
    weights = tl.exp(values - maximum)
    probability = weights / tl.sum(weights, axis=0)
    tl.store(Y + row * N + columns, probability, mask=columns < N)
~~~

每个program拥有一整行输出，因此没有跨program归约。BLOCK选不小于N的二次幂，尾槽填负无穷，指数贡献零。最大值与分母均为当前行标量，广播回整行。输入限定有限普通数值，空行拒绝，不处理任意attention mask。

### 2. Wrapper与资源上界

~~~python
def softmax(x):
    import torch
    check_teaching_inputs(x)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 4096:
        raise ValueError('matrix with 1..4096 columns required')
    output = torch.empty_like(x)
    if x.shape[0] > 0:
        block = triton.next_power_of_2(x.shape[1])
        softmax_kernel[(x.shape[0],)](x, output, x.shape[1], block)
    return output
~~~

四千零九十六的上界是教学接口选择，不是宣称所有设备在该范围都达到最佳性能，也不是Triton的普遍理论极限。它限制单program逻辑状态规模，使实验可控。扩大上界需要检查编译资源、性能与长行策略，不能只改断言。

行数为零时输出空矩阵并跳过launch，列数为零则拒绝，因为空归一化轴没有本接口定义的概率分布。两个零维情况数学含义不同，不应统一处理。测试应分别验证空行集合与空特征维。

### 3. 手算与故障定位

输入零、ln二、ln三，BLOCK四，加载后第四槽为负无穷。最大值ln三，指数为三分之一、三分之二、一、零，分母二，前三个概率为六分之一、三分之一、二分之一。第四槽不写回。

若错把padding填零，这个例子中最大值仍可能正确，但padding指数会变成三分之一并进入分母，所有合法概率都会变小。与K08讨论的“最大值填零”不同，这里整个softmax尾槽必须在指数阶段为零贡献，因此不能只说最大值结果没变就认为安全。

### 4. Softmax中的padding如何沿整条链传播

仍用N三、BLOCK四，真实logits为负三、负二、负一。正确padding负无穷，最大值负一，指数为e的负二次方、e的负一次方、一、零。分母约一点五零三二，合法概率和为一。

若padding错误填零，最大值变成零，指数为e的负三次方、e的负二次方、e的负一次方、一。即使最后不写第四项，前三项的和也只有约零点三五六，错误已经进入分母。输出mask只能保护存储边界，不能撤销padding对数学归约的影响。

若padding仍为负无穷，但把sum除数误写成N，softmax也会错误，因为softmax不是求指数的平均值后直接当分母；正确分母就是指数和。归一化与均值虽然都包含除法，却有不同定义。不要把RMSNorm的除N习惯迁移到softmax。

### 5. 大公共偏置与极端差值是两种不同测试

给所有logit加一千，稳定softmax应保持结果，因为减最大值抵消公共偏置。这个测试检验是否采用稳定平移。另一种测试是同一行分数差距很大，例如零与负一千；此时小概率可能下溢到零，这是有限精度表示的结果，不同于直接exp一千产生溢出。

如果业务需要保留极小概率，可能应使用log-softmax或直接在对数域计算损失，而不是期望普通float32概率保存任意小正数。算子接口应服务实际数学目标，不能通过随意加epsilon让所有概率非零，因为那会改变分布。

softmax前向的行和检查也应采用容差。浮点求和不保证严格等于一，但若偏差达到明显百分比，通常不是末位舍入。打印分母、最大值与合法元素数，能迅速区分稳定化、padding和归约轴问题。


## 第五课：RMSNorm的数值合同

### 1. 完整Kernel

~~~python
@triton.jit
def rmsnorm_kernel(X, W, Y, N: tl.constexpr,
                   EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns,
                     mask=columns < N, other=0.0).to(tl.float32)
    weight = tl.load(W + columns,
                     mask=columns < N, other=0.0).to(tl.float32)
    square_sum = tl.sum(values * values, axis=0)
    inverse = tl.rsqrt(square_sum / N + EPS)
    output = values * inverse * weight
    tl.store(Y + row * N + columns, output, mask=columns < N)
~~~

N是真实特征数，BLOCK是物理逻辑槽数。平方和中padding零不贡献，但均值必须除N。若N三、BLOCK四，除四会让均方根错误变小，输出整体偏大。这个错误在N恰好为二次幂时完全隐藏，因此非整除宽度是必要测试。

values先转float32再平方，避免未来窄输入路径在平方阶段过早溢出。当前wrapper已限定float32，转换仍明确内部合同。rsqrt表示平方根倒数，epsilon放在均方加法后、开根之前，与把epsilon加到根之后不是同一函数。

### 2. 完整Wrapper

~~~python
def rmsnorm(x, weight, eps=1e-5):
    import math
    import torch
    check_teaching_inputs(x, weight)
    if x.ndim != 2:
        raise ValueError('matrix input required')
    if weight.shape != (x.shape[1],):
        raise ValueError('one weight per feature required')
    if not 1 <= x.shape[1] <= 4096:
        raise ValueError('feature width outside teaching bounds')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('finite positive epsilon required')
    output = torch.empty_like(x)
    if x.shape[0] > 0:
        block = triton.next_power_of_2(x.shape[1])
        rmsnorm_kernel[(x.shape[0],)](
            x, weight, output, x.shape[1], eps, block)
    return output
~~~

epsilon检查不能只写小于等于零，因为NaN与普通比较的结果可能让它绕过检查。显式isfinite同时拒绝NaN与无穷，保证数值合同清楚。weight长度匹配不代表数据值有限，当前教学实验使用有限输入；若生产接口要主动拒绝异常值，需决定检查成本与位置。

### 3. 两个手算案例

输入三、四，weight一、二，忽略epsilon时均方为十二点五，逆均方根约零点二八二八四三，输出约零点八四八五二八与二点二六二七四二。加入epsilon后结果略有变化，参考与目标必须使用相同epsilon，不能拿忽略epsilon的手算值要求机器逐bit一致。

输入全零时平方和零，inverse为epsilon平方根的倒数，但最终values仍为零，所以输出零。这个测试能检查零输入路径，却不能发现忘乘weight，因为零乘任何weight都为零。需要再加一个非零输入和各不相同的weight，才能覆盖参数索引。

### 4. RMSNorm的weight共享与梯度准备

weight长度N，在所有行共享，因此前向每行都读取同一组参数。地址为W加column，不包含row。若误写row乘N加column，第一行可能正确，后续行越界或读取其他数据。测试至少两行且输入不同，才能发现这种共享参数地址错误。

将来注册反向时，输入梯度每行独立，但weight梯度需要沿所有行累加。前向共享读取很简单，反向共享写入却需要归约。理解这个非对称性，有助于解释为什么某些前向很短的算子，其反向实现复杂得多。

如果希望融合残差，先明确归一化对象是X加R，还是归一化X后再加R。两种表达式数学不同。前者平方和针对组合值，反向同时影响两个输入；后者残差路径直接传递上游梯度。不要只因代码都出现add与rmsnorm就把它们视为可任意交换。

### 5. Epsilon与尺度的具体反例

取单元素x为很小正数，weight为一。RMSNorm输出是x除以平方根(x平方加epsilon)。当x平方远小于epsilon时，输出近似x除以epsilon平方根，而不是恒等于一。若把epsilon放在平方根外，近似结果会不同，尤其在小输入区域差异明显。

取x为零，输出零；取x为负数，输出保持负号。把归一化误写成绝对值除均方根，会在正输入上通过，却在负输入上失败。因此测试应包含正负混合，不能只用随机均匀正数。

weight可以为负或零，输出应按对应特征缩放。某个特征weight零时该输出为零，但输入仍参与整行均方，因此改变该输入仍可能影响其他特征输出。这个依赖说明RMSNorm不是各特征独立缩放，反向必须包含共同尺度路径。


## 第六课：GEMM的二维program映射

### 1. 完整Kernel

~~~python
@triton.jit
def matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    local_k = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(0, tl.cdiv(K, BK)):
        kk = block * BK + local_k
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    mask=(rows[:, None] < M) & (kk[None, :] < K),
                    other=0.0)
        b = tl.load(B + kk[:, None] * N + columns[None, :],
                    mask=(kk[:, None] < K) & (columns[None, :] < N),
                    other=0.0)
        accumulator += tl.dot(a, b, input_precision='ieee')
    tl.store(C + rows[:, None] * N + columns[None, :],
             accumulator,
             mask=(rows[:, None] < M) & (columns[None, :] < N))
~~~

a形状BM乘BK，b形状BK乘BN，accumulator形状BM乘BN。每轮kk表示当前全局归约位置，两个load分别用各自矩阵行宽K与N。input_precision参数明确当前float32点积的目标策略，实际支持与生成路径依赖安装版本和目标，必须在设备环境验证。

当前教学配置BM十六、BN十六、BK三十二，不宣称它对所有shape最佳。选择这些值是为了给出有限而可读的起点。K06已经解释tile资源与复用，K14会讨论怎样有证据地调参。

### 2. Wrapper与独立输出类型

~~~python
def matmul(a, b):
    import torch
    check_teaching_inputs(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('compatible matrices required')
    if min(a.shape + b.shape) <= 0:
        raise ValueError('nonempty matrices required')
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    matmul_kernel[grid](a, b, output, m, n, k, 16, 16, 32)
    return output
~~~

此接口选择拒绝零维矩阵，与K06的CPU参考可支持空和不同。这不是数学矛盾，而是受限设备路径的合同；调用者需要在更高层处理空输入，或扩展wrapper实现一致行为。文档与测试必须明确这种差异，不能暗示所有参考边界都已在GPU实现支持。

### 3. 十九乘二十三乘十七的地址轨迹

M十九、K二十三、N十七，grid为二乘二，K只一轮BK三十二。右下program的rows为十六至三十一，columns为十六至三十一，只有前三行和第一列输出有效。A加载的行条件保留十六、十七、十八，K条件保留零至二十二；B加载的列条件只保留十六。

若把BK改成十六，则K有两轮，第二轮只七个k有效。两种配置数学结果应一致，有限精度顺序可能略有变化。这个对照能检查accumulator跨轮保留与尾部零填充，而不仅是二维输出mask。

### 4. GEMM中每个program到底保存多久

accumulator在K循环外创建，生命周期覆盖当前输出tile的全部K块。a与b每轮加载后用于一次dot更新，之后可被下一轮替代。rows与columns在循环外固定，kk随轮次变化。这三种生命周期对应不同资源压力，优化时应尽量避免让不必要中间值跨越过长范围。

若为了调试保存每轮a、b或accumulator到全局数组，会显著改变内存流量与资源，不能把调试版本的时间当作正式Kernel性能。正确性trace可以使用小尺寸单独运行，性能测试则关闭额外观察路径。

tl.dot并不意味着整个GEMM自动拥有最佳流水线。当前代码只描述分块乘加，实际编译器如何安排加载、矩阵指令与布局转换，要看目标和配置。理解源码不应停在“调用了dot所以用了某个硬件单元”的猜测。

### 5. 输入精度策略与基线公平性

当前float32路径显式选择一种精度策略，是为了让教学数值合同更容易解释。框架矩阵乘法基线可能受全局精度设置影响，使用不同内部乘法方式。若两者速度与误差不同，应先核对策略，而不是直接判断自写Kernel优劣。

比较时至少记录输入dtype、累加dtype、输出dtype与矩阵乘法精度配置。若使用半精度输入加单精度累加，应基于已舍入输入构建参考。若改变策略以提高性能，必须同时报告误差变化和适用范围，不能把数值放宽伪装成纯调度优化。

小矩阵与大矩阵还可能让框架选择不同算法。固定API调用并不保证底层Kernel相同，因此性能曲线出现突变时应检查调度与算法选择。自写算子的优势或劣势通常有形状范围，不能用一个尺寸推广到全部模型层。


## 第七课：在线注意力的受限Triton实现

### 1. 为什么先限定自注意力

当前库要求Q与K shape相同，V行数相同，长度一到五百一十二，D与Dv一到六十四。每个program负责一个query，顺序遍历key块，重点展示在线状态。它没有多query共享KV块的高性能组织，也没有反向或dropout。

方形因果模式使用key索引不大于query索引，含对角线。第一个key块包含key零，因此每个query在第一块至少有一个有效key。这一事实使当前实现可以直接计算有限的新最大值；若扩展矩形位置mask或任意mask，就必须加入K07的全mask安全处理，不能只改shape检查。

### 2. 完整Kernel

~~~python
@triton.jit
def attention_kernel(Q, K, V, O,
                     N: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                     CAUSAL: tl.constexpr, BK: tl.constexpr,
                     BD: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0)
    feature = tl.arange(0, BD)
    value_feature = tl.arange(0, BV)
    query = tl.load(Q + row * D + feature,
                    mask=feature < D, other=0.0).to(tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.full((BV,), 0.0, tl.float32)
    for block in range(0, tl.cdiv(N, BK)):
        token = block * BK + tl.arange(0, BK)
        keys = tl.load(K + token[:, None] * D + feature[None, :],
                       mask=(token[:, None] < N) & (feature[None, :] < D),
                       other=0.0).to(tl.float32)
        score = tl.sum(keys * query[None, :], axis=1)
        score *= tl.rsqrt(float(D))
        valid = token < N
        if CAUSAL:
            valid = valid & (token <= row)
        score = tl.where(valid, score, -float('inf'))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        old_factor = tl.exp(maximum - updated)
        probability_numerator = tl.exp(score - updated)
        values = tl.load(
            V + token[:, None] * DV + value_feature[None, :],
            mask=(token[:, None] < N) & (value_feature[None, :] < DV),
            other=0.0).to(tl.float32)
        numerator = numerator * old_factor + tl.sum(
            probability_numerator[:, None] * values, axis=0)
        denominator = denominator * old_factor + tl.sum(
            probability_numerator, axis=0)
        maximum = updated
    tl.store(O + row * DV + value_feature,
             numerator / denominator, mask=value_feature < DV)
~~~

keys形状BK乘BD，query扩成一乘BD后广播，沿feature轴一求和得到BK个分数。values形状BK乘BV，权重扩成BK乘一，沿token轴零求和得到BV维分子贡献。两个tl.sum轴不同，分别对应点积特征与key聚合，不能机械复用同一个axis。

BD与BV分别是D和Dv向上取二次幂后的逻辑宽度。QK点积padding填零，V特征padding也填零；score的token无效位置则设负无穷，使指数权重零。不同维度的padding需要不同处理，正是前面课程反复强调中性值的原因。

### 3. 完整Wrapper

~~~python
def attention(q, k, v, causal=False):
    import torch
    check_teaching_inputs(q, k, v)
    if q.ndim != 2 or k.shape != q.shape:
        raise ValueError('Q/K self-attention shapes required')
    if v.ndim != 2 or v.shape[0] != q.shape[0]:
        raise ValueError('V rows must match sequence length')
    n, d = q.shape
    dv = v.shape[1]
    if not 1 <= n <= 512 or not 1 <= d <= 64 or not 1 <= dv <= 64:
        raise ValueError('bounded teaching attention shape')
    output = torch.empty((n, dv), device=q.device, dtype=q.dtype)
    attention_kernel[(n,)](
        q, k, v, output, n, d, dv, causal, 32,
        triton.next_power_of_2(d), triton.next_power_of_2(dv))
    return output
~~~

输出dtype显式与Q一致，避免默认类型影响指针与存储。causal是编译期布尔条件，不同模式可以生成不同特化。代码没有检查Q、K、V是否包含NaN或无穷，实验合同限定有限合理范围；正式产品应根据需求决定异常处理策略，而不是默认所有数值都可稳定运行。

### 4. Attention一轮的全部shape与数值单位

设D五、Dv三、BK四，BD八、BV四。query是长度八向量，后三项零；keys是四乘八，特征尾部零；逐元素相乘后沿八个特征求和，得到四个未缩放点积。乘一除以五平方根后才是softmax分数，不能除以BD八平方根。

values是四乘四，最后特征槽零；权重是四个标量，扩成四乘一后与values相乘，沿token归约得到长度四分子。最终只写前三个value特征。这里D与BD、Dv与BV各有真实尺寸与逻辑块尺寸之分，四个符号都不能混用。

最大值与分母是当前query的标量，numerator是当前query的value向量。若把maximum初始化为长度BK向量，或把分母保留为每token值，就不再是同一在线状态。检查中间shape能在数值比较前发现很多公式翻译错误。

### 5. 因果模式中后续全mask块为什么仍安全

对较早query，第一块之后的许多key块可能全部在未来，score全负无穷。此时maximum已经由第一块的可见key变成有限值，updated保持旧maximum，old_factor为一，当前权重全零，所以状态不变。这个路径安全依赖旧状态已经有效。

如果第一块本身全mask，旧maximum仍负无穷，updated也负无穷，就会触发无效差值。两种“当前块全mask”看起来相似，但历史状态不同，必须区分。当前受限方形因果接口保证首块有效，通用接口则要处理空状态。

这也是阅读高性能实现时不能随意删除shape限制的原因。某个简短表达式可能依赖接口排除了一类边界；扩大接口后，原数学公式需要增加状态分支或安全表示。限制与实现共同构成正确性证明。


## 第八课：六类算子的统一CPU参考与检查

### 1. 参考精度与实际输入一致

将目标实际输入值转换到float64再计算参考，可以减少参考自身误差。若输入先经历半精度或量化编码，参考也应基于编码后的实际值，而不是编码前理想值。当前路径float32，因此先生成float32输入，再转float64参考是清晰做法。

~~~python
import numpy as np

def softmax_reference(x):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)

def rmsnorm_reference(x, weight, eps):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight

def attention_reference(q, k, v, causal=False):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = q @ k.T / np.sqrt(q.shape[1])
    if causal:
        rows = np.arange(q.shape[0])[:, None]
        cols = np.arange(k.shape[0])[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)
    return softmax_reference(scores) @ v
~~~

这个attention参考只用于当前方形受限合同，第一行含自身，因此不会全mask。通用矩形与全mask实现已在K07给出，不要把这里简化函数当作覆盖所有注意力语义的通用库。

### 2. 误差检查的顺序

先检查输出shape、dtype与device，再确认有限性，然后比较数值。若比较函数允许广播，错误shape可能被自动扩展后看起来接近，所以shape必须独立断言。若输出出现NaN，先定位原因，不要用忽略NaN的统计量让测试继续显示小误差。

每次失败报告最大绝对误差位置、参考值、实际值与输入shape。对于矩阵或注意力，位置模式往往能指向错误轴、尾块或head映射。仅报告“allclose=False”会浪费大量调试时间，尤其在多个算子共享同一测试框架时。

### 3. 六类不变量并不相同

加法可检查交换两个输入结果不变；总和可检查分段合并与空和；softmax可检查行和、平移不变性与非负；RMSNorm可检查零输入与weight作用；GEMM可检查小型手算、单位矩阵和非方阵；attention可检查常量value、因果干预与块大小一致性。

这些性质各自只能覆盖一部分错误。单位矩阵GEMM不能替代一般随机矩阵，因为许多索引错误在单位结构下被隐藏；softmax行和一不能发现概率排列错误；attention常量value不能发现分数错误。测试集需要不同性质互补，而不是重复十次同一弱断言。

## 第九课：三个典型失败案例的完整修复

### 1. RMSNorm非二次幂宽度整体偏大

先用N三、输入一、二、二、weight全一。平方和九，正确均方三，忽略epsilon的逆尺度是一除以三平方根。若错除BLOCK四，均方变二点二五，逆尺度三分之二，输出整体偏大。通过打印平方和与除数，可以直接定位，而不必怀疑rsqrt近似。

修复除数为真实N后，再测试N一、三、三十一、三十二、三十三。若只有尾部输出错误，继续检查weight加载与store mask。一个修复可能解决尺度问题，却不自动证明地址边界也正确。

### 2. GEMM只在第二个K块失败

将K设为三十三、BK三十二，让前三十二项全零、最后一项非零。正确结果完全由尾块贡献，能检验最后一轮地址。再反过来让最后一项零、前段非零，检验accumulator是否跨轮保留。两组输入分别针对“尾块遗漏”和“累加器重置”。

修复时检查kk等于block乘BK加local_k，A行宽K、B行宽N，以及两个独立mask。不要只把某个索引加一减一试到通过；重新推导全局坐标才能确保其他shape不再失败。

### 3. Attention扩展任意mask后出现NaN

原实现依赖每个query在第一块看见key零。若新增mask让某行首块全不可见，maximum与updated都为负无穷，旧因子计算出现无效差值。修复需要K07的safe最大值与零分母合同，而不是把NaN输出事后替换零。

同时检查后续块可能出现有效key的情况。首块全mask、第二块有效时，输出应等于有效集合的正常注意力；如果简单检测首块空就提前返回零，会丢失后面的合法贡献。边界修复必须覆盖完整状态生命周期。

## 第十课：十二道分级练习与完整答案

### 练习一：加法尾部所有权

题目：N一千零三、BLOCK二百五十六，最后program编号及有效数。答案：编号三，有效二百三十五，offset七百六十八至一千零二。最后二十一个逻辑槽不读不写。

### 练习二：归约partial参考

~~~python
x = np.arange(1003, dtype=np.float32)
partials = [x[start:start + 256].astype(np.float64).sum()
            for start in range(0, x.size, 256)]
assert len(partials) == 4
assert sum(partials) == x.astype(np.float64).sum()
~~~

### 练习三：Softmax手算

答案：输入零、ln二、ln三得到六分之一、三分之一、二分之一。加同一常数一千后结果应在误差内一致，测试同时检查行和与逐项值。

### 练习四：RMSNorm真实分母

答案：输入一、二、二，平方和九，真实N三，所以均方三。BLOCK四只是padding槽数，不进入定义。epsilon应在均方上相加，再取平方根倒数。

### 练习五：GEMM三种mask

答案：A检查row小于M且k小于K；B检查k小于K且col小于N；C检查row小于M且col小于N。三者不同，不能用一个输出mask替代所有加载边界。

### 练习六：Attention两个归约轴

答案：keys乘query后形状BK乘BD，沿轴一归约特征，得到BK分数；权重乘values后形状BK乘BV，沿轴零归约token，得到BV分子。D与Dv不同时更容易发现混淆。

### 练习七：零输入能发现什么

答案：零输入能检查初始化、epsilon和部分输出覆盖，但不能发现忘乘weight、错误概率分布或许多地址排列错误。必须配合非零且位置可辨识的输入。

### 练习八：构造位置编码矩阵

~~~python
a = np.array([[10 * i + k for k in range(3)] for i in range(2)],
             dtype=np.float64)
b = np.array([[100 * k + j for j in range(4)] for k in range(3)],
             dtype=np.float64)
expected = np.zeros((2, 4))
for i in range(2):
    for j in range(4):
        expected[i, j] = sum(a[i, k] * b[k, j] for k in range(3))
assert np.array_equal(expected, a @ b)
~~~

### 练习九：Attention常量value

~~~python
rng = np.random.default_rng(9)
q = rng.normal(size=(5, 3))
k = rng.normal(size=(5, 3))
v = np.tile([2.0, -1.0], (5, 1))
assert np.allclose(attention_reference(q, k, v), v)
assert np.allclose(attention_reference(q, k, v, True), v)
~~~

### 练习十：明确未实现的梯度

答案：当前六个Triton wrapper没有注册autograd，不能因为输入requires_grad为真就假定梯度正确。训练使用必须在K10增加明确反向或自定义算子注册，并通过梯度检查。

### 练习十一：设计非法输入测试

答案包括CPU张量、不同设备、非float32、非连续视图、shape不匹配、空softmax列、非正或非有限epsilon、GEMM内维不等、attention超出长度或特征上界。每个测试应检查明确异常，不把拒绝合同外输入算成Kernel错误。

### 练习十二：给出完整验收顺序

答案依次是CPU数学参考、地址与写者模拟、设备编译、小尺寸GPU结果、边界与负向测试、合理数值范围、最后性能测量。每步保留证据，不能通过最后一次速度数字反推前面全部已经正确。

## 第十一课：建立真正能定位问题的验证矩阵

### 1. 形状维度不是随便挑几个数字

对向量算子，围绕块边界选择长度；对行算子，分别改变行数与列数；对GEMM，独立改变M、N、K；对attention，独立改变序列长、query特征宽和value特征宽。每个维度对应不同索引或归约责任，不能只把所有维度同时设成同一个数。

GEMM至少需要一组所有维度都不整除tile的输入，一组只有K有尾部的输入，一组只有输出行或列有尾部的输入。这样失败模式能对应到A加载、B加载、累加或C写回。若只测试一个完全不规则形状，虽然覆盖广，定位时仍需要更细分案例。

attention当前方形合同下，选择N三十五、D五、Dv七可以同时触发key尾块、query特征padding与value特征padding。再分别用D八、Dv七和D五、Dv八隔离两个特征维。数字的选择有明确目的，不是为了显得测试很多。

### 2. 数据分布也要分层

位置编码输入适合地址检查，例如矩阵元素为一百乘行号加列号；小整数适合精确手算；正负混合随机数适合一般数值验证；大公共偏置适合softmax稳定性；接近零输入适合epsilon行为；大动态范围适合累加误差研究。

不要让所有算子共享一种随机正态分布然后结束。正态随机通常很难触发全零、常量行、严格重复索引、尾部特殊值或极端公共偏置。每种结构性输入都应附一条解释，说明它能暴露哪类错误，从而让测试成为教学材料的一部分。

随机seed只保证生成过程在相同条件下可重复，不保证设备执行完全确定，也不保证未来库版本生成完全相同的所有细节。遇到重要失败时保存实际小输入数组与预期值，比只记录seed更稳妥。不要保存公司真实数据，合成最小复现通常更清楚。

### 3. 误差阈值需要与运算结构对应

向量加法只有一次浮点加法，误差通常容易解释；长归约与GEMM包含很多累加，误差可能随长度与数值分布变化；softmax与归一化还包含指数、平方根和除法。统一给所有算子同一个极宽容差，会掩盖局部错误；统一给极严容差，又可能把正常舍入当成失败。

一个合理流程是先用float64参考计算目标，再查看误差绝对值与相对值分布，结合输入和累加类型设定阈值。阈值应在测试前或基于明确数值分析确定，而不是看到失败后不断放大。若某个边界需要不同阈值，应解释原因并保留更严格的普通路径测试。

对参考接近零的结果，绝对误差更重要；对大幅度结果，相对误差更能体现比例偏差。常用混合条件同时考虑二者。还要检查输出全部有限，否则NaN可能让某些比较逻辑失效，或者被只统计有限项的代码忽略。

### 4. 一个统一CPU验收器

~~~python
def numerical_report(actual, expected, atol=1e-6, rtol=1e-5):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError('shape mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('non-finite result')
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))
    if np.any(error > allowed):
        index = np.unravel_index(int(np.argmax(error)), actual.shape)
        raise AssertionError(
            f'max error at {index}: {error[index]}, '
            f'actual={actual[index]}, expected={expected[index]}')
    return {'max_abs': float(error.max()) if error.size else 0.0,
            'elements': int(error.size)}

assert numerical_report([1, 2], [1, 2])['max_abs'] == 0
~~~

这个函数不检查GPU的device与dtype，因为输入已经转换为NumPy；实际GPU验收应在复制前先检查这些属性。转换到CPU可能同步设备，适合正确性验证，不应混进纯Kernel计时。把正确性与性能两个路径分开，避免测量结果被隐含复制影响。

验收器本身应有负向控制：故意改变一个元素、传错shape、插入NaN，确认它会拒绝。否则一个写错比较条件的测试框架可能让所有算子“通过”。测试代码也会出错，不能因为它位于tests目录就自动可信。

### 5. 不变量测试与独立oracle如何配合

独立oracle给出逐项预期，不变量检查给出结构约束。二者最好来自不同表达方式，例如GEMM目标用分块dot，参考用显式三循环；attention目标用在线状态，参考用完整概率矩阵。若两边复制相同索引错误，相互比较可能失去意义。

不变量也应有负向控制。例如把softmax输出列顺序打乱，行和仍为一，说明行和检查不能发现排列错误；把attention的V全部设相同，很多错误分数仍得到相同输出，说明常量value测试不能单独验收。明确每个测试的盲区，比堆积断言更能提高质量。

当oracle与不变量矛盾时，先缩小输入手算，不要自动相信更长或更复杂的实现。一个短参考也可能错误，一个高性能库也可能采用不同合同。明确接口语义后再比较，才能把实现错误与合同差异区分开。

### 6. 变换输入后，哪些结果关系必须保持

除了逐项oracle，还可以利用数学变换建立关联测试。加法交换两个输入结果不变；GEMM若将A某行乘二，输出对应行也应乘二；若将B某列乘负一，输出对应列应变号。这样的测试不依赖重复实现全部算法，能发现某些行列映射错误。

Softmax对每行加同一常数保持不变，但对每列加不同常数通常会改变分布。RMSNorm在epsilon为正时并不严格尺度不变，所以不能把忽略epsilon的数学直觉写成无条件断言。关联测试必须从实际合同推导，不能凭“归一化应该差不多”猜性质。

Attention若对所有value加同一向量，正常非空概率行的输出也应加该向量，因为概率和为一。当前受限自注意力每行都至少有一个有效key，所以这个关系成立；通用全mask零输出合同下，全mask行不应加该向量，需要单独处理。相同测试在不同接口中的适用范围可能不同。

~~~python
rng_properties = np.random.default_rng(91)
a_property = rng_properties.normal(size=(3, 5))
b_property = rng_properties.normal(size=(5, 4))
base_product = a_property @ b_property
changed_a = a_property.copy()
changed_a[1] *= 2
expected_product = base_product.copy()
expected_product[1] *= 2
assert np.allclose(changed_a @ b_property, expected_product)

q_property = rng_properties.normal(size=(4, 3))
k_property = rng_properties.normal(size=(4, 3))
v_property = rng_properties.normal(size=(4, 2))
shift = np.array([2.0, -3.0])
base_attention = attention_reference(q_property, k_property, v_property, True)
shifted_attention = attention_reference(
    q_property, k_property, v_property + shift, True)
assert np.allclose(shifted_attention, base_attention + shift)
~~~

这些CPU例子验证测试性质本身，搬到GPU验收时把目标调用替换为待测算子，并保留独立预期变换。不要两边都调用同一个目标函数后只比较完全相同的表达式，那样可能没有增加任何检验能力。关联测试应让两次输入经历不同但数学可预测的路径。

### 7. 错误注入如何训练调试能力

为每类算子保留一个只在CPU模拟中的故障变体，例如向量grid向下取整、归约padding填一、softmax沿错误轴、RMSNorm除BLOCK、GEMM交换B索引、attention不缩放旧分子。要求测试集对每个故障至少有一个明确失败案例，并记录是哪条断言抓住它。

错误注入不应进入正式设备路径或生产代码，也不需要为了测试故意运行可能越界的GPU Kernel。CPU模拟足以验证测试设计是否有区分力，真正设备代码保持正确与安全边界。这样既能学习故障模式，又避免用非法访问制造无意义风险。

如果某个故障变体仍全部通过，说明测试有盲区，而不是故障合理。例如全零输入无法发现大部分索引排列错误，方阵无法可靠发现行宽混用，相同weight无法发现参数索引颠倒。调整输入结构，让每个逻辑位置具有可辨识贡献，通常比单纯增加随机样本数更有效。

### 8. 从测试失败形成一条可复用知识记录

每次修复写四件事：最小输入是什么、预期中间状态是什么、哪一个索引或数值合同被违反、增加了哪条回归断言。不要只写“修复精度问题”或“修复边界问题”，这种笼统描述无法帮助未来复习，也不利于代码审查。

例如RMSNorm的记录可以明确写：宽度三补齐四，平方和九，错误除四导致均方二点二五，修复为除真实宽度三；回归输入一、二、二并用不同weight。这样一条记录同时连接数学、实现和测试，后续学习新的归一化变体时可以直接迁移方法。


## 第十二课：从六类前向扩展到训练与组合算子

### 1. 前向正确不代表自动梯度存在

普通Triton Kernel是设备计算，框架未必知道它内部的数学关系。输入requires_grad为真并不会自动让任意自写内存读写进入autograd图。训练使用必须注册明确反向或通过受支持的自定义算子机制接入，下一章将完整实现。

最危险的情况不是立即报错，而是某条路径静默断开梯度，模型仍能运行却不按预期更新。验证训练集成时，应检查每个需要梯度的输入和参数是否得到非空、shape正确且数值合理的梯度，再做有限差分。不能只看loss下降一次就认为算子反向正确。

### 2. 哪些前向状态值得保存

加法反向通常不需要保存输入值，主要需要shape与广播关系；RMSNorm反向需要输入、weight和逆均方根等信息；softmax可保存输出概率；attention可保存输入、输出与每行归一化统计量，重算概率块。保存策略因数学结构不同，不应统一“把所有中间张量都存下来”。

保存更多状态减少重算但增加显存，保存更少增加计算与实现复杂度。决定之前先写出反向公式依赖哪些量，再判断哪些可从现有状态重建。不要为了少占显存而丢弃无法恢复的信息，也不要无意识保留巨大中间矩阵破坏在线算法的主要收益。

### 3. 融合RMSNorm与残差的两种合同

若输出为RMSNorm(X+R)，先将X与R逐元素相加，再对和计算均方。若输出为RMSNorm(X)+R，R不参与归一化统计量。两者都可以融合成一个Kernel，但数学行为与反向完全不同。接口名应明确顺序，测试用非零且不同的X与R验证。

前者反向对X与R的梯度相同地接收归一化输入梯度；后者R梯度直接等于上游G，X梯度经过归一化反向。用这个差异可以设计梯度测试，防止实现只在前向某些特殊输入上看起来接近。

若还要返回未归一化残差和供后续层使用，算子变成多输出，别名与保存策略又发生变化。融合并不只是把几行算术放在一起，还包括外部可观察输出、生命周期与梯度合同。学习组合算子时应先画数据依赖，再决定Kernel边界。

### 4. Bias、激活与GEMM的收尾

GEMM后加按列bias再激活，适合在accumulator写回前完成，因为每个输出元素所需信息局部可得。若激活需要保存某种中间值供反向，可以保存最终输出或预激活值，具体取决于函数。不能一概说“融合后不需要任何中间状态”。

若激活包含跨行或跨列归约，就不再是简单epilogue。当前一个program只拥有BM乘BN输出tile，未必覆盖完整归约轴。直接在tile内做softmax会得到局部归一化，数学上不同于整行softmax。融合范围必须与输出所有权和归约范围匹配。

这个例子说明性能设计的核心不是尽量少Kernel，而是选择能保持正确数据依赖并有效利用资源的边界。有时两个清晰阶段比一个复杂全局协作Kernel更可靠且更快，尤其当临时数据量不大、跨block同步困难时。

### 5. 扩展非连续布局的分步路线

先支持一种明确模式，例如输入转置视图但输出连续，再支持一般正stride，最后才考虑负stride或重叠视图。每一步都需要独立地址公式与边界测试。一次宣称“支持任意tensor”会引入大量未证明情形，远超删除contiguous断言的工作量。

对GEMM，传入A与B的两个stride，可以表达常见转置；对行归一化，列stride不为一会改变访问合并与资源表现；对attention，多head布局还涉及batch/head stride与共享KV映射。接口越一般，测试矩阵越复杂，性能专用路径也可能越多。

可以保留连续快速路径与通用正确路径，由wrapper根据合同选择。选择逻辑必须可观察和可测试，性能报告说明走了哪条路径。通用路径慢并不一定是失败，首先提供正确覆盖，再针对真实常见布局优化，是合理工程顺序。

## 第十三课：安全而有信息量的GPU实验

### 1. 编译、暖机与正确性分别记录

第一次调用可能包含JIT编译，不能拿它代表稳态Kernel时间。先完成一次小输入编译与正确性比较，再进行若干暖机调用，最后测量稳定执行。编译失败、运行失败和数值失败应分别记录，方便判断是环境、目标支持还是实现问题。

暖机不需要无限长时间，也不应在共享H100上持续占用。为每个算子设有限shape集合、有限重复次数和总时长预算，完成后退出。学习目标是获得可解释证据，几十秒内的小实验通常比长时间无目标跑分更有价值。

### 2. 一个完整设备验收脚本的结构

~~~python
def gpu_smoke_checks():
    import torch
    from labs.kernels import triton_ops as ops
    if not torch.cuda.is_available():
        raise RuntimeError('GPU execution not available')
    torch.manual_seed(9)
    device = torch.device('cuda')
    a = torch.arange(1003, device=device, dtype=torch.float32)
    b = torch.ones_like(a) * 3
    torch.testing.assert_close(ops.add(a, b), a + b)
    torch.testing.assert_close(ops.reduce_sum(a), a.sum())
    x = torch.randn((3, 131), device=device, dtype=torch.float32)
    torch.testing.assert_close(ops.softmax(x), torch.softmax(x, dim=-1),
                               rtol=1e-4, atol=1e-6)
    weight = torch.linspace(0.5, 1.5, 131, device=device)
    expected_norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(ops.rmsnorm(x, weight), expected_norm,
                               rtol=1e-4, atol=1e-5)
    left = torch.randn((19, 23), device=device)
    right = torch.randn((23, 17), device=device)
    torch.testing.assert_close(ops.matmul(left, right), left @ right,
                               rtol=1e-4, atol=1e-4)
    q = torch.randn((35, 5), device=device)
    k = torch.randn((35, 5), device=device)
    v = torch.randn((35, 7), device=device)
    for causal in (False, True):
        score = q @ k.T / (5 ** 0.5)
        if causal:
            allowed = torch.arange(35, device=device)[None, :] <= torch.arange(35, device=device)[:, None]
            score = score.masked_fill(~allowed, -float('inf'))
        expected_attention = torch.softmax(score, dim=-1) @ v
        torch.testing.assert_close(ops.attention(q, k, v, causal),
                                   expected_attention,
                                   rtol=1e-4, atol=1e-4)
    torch.cuda.synchronize()
~~~

这个函数定义了可执行验收流程，但当前环境未运行。实际使用时还要核对框架矩阵精度设置，避免目标与基线策略不同；必要时将输入复制到CPU用float64参考比较。示例容差是小规模float32起点，不是所有设备、形状与数值范围的永久保证。

最后同步保证函数返回前设备工作完成，也让异步错误更容易归属到本次测试。正式性能测量不应把全部正确性断言与CPU复制放进计时区间。测试脚本可以先验证，再单独计时，两个结果同时保存。

### 3. 共享服务器上的有界任务

先用单张允许使用的GPU，限制序列长度、矩阵尺寸与重复次数，不启动无期限循环。记录开始与结束时间、设备选择和最大显存分配。若公司任务需要资源，应能快速停止当前实验并从已有记录恢复，而不是依赖长时间持续占用才能完成学习。

H100适合验证较复杂的矩阵指令、低精度与多GPU通信，但向量地址、归约公式与多数数值检查可以先在CPU完成。个人PC可以承担日常编辑、参考测试与适配后的单卡实验，服务器用于确实需要目标硬件特性的有限验证。资源分配应由实验问题决定。

不要在教材中硬编码公司机器的真实路径、主机名或访问凭据。实验命令使用环境变量或本地配置，并避免把密钥、驱动日志中的敏感信息提交仓库。这里的目的只是让教学工程可复现，不需要任何公司数据。

### 4. 为什么性能测试也要负向控制

如果计时结果接近零，可能只测到异步提交；如果每次都包含编译，可能把编译延迟当成Kernel；如果输出从未被使用，某些测试组织可能没有测到预期工作。通过增加输入规模、加入明确同步或比较事件计时与端到端计时，可以检查测量是否符合预期。

性能结果还应满足粗略量级常识。向量加法逻辑流量为十二乘元素数，计算有效带宽时单位要统一；GEMM使用约二乘M乘N乘K计算FLOP，时间从毫秒转秒再求吞吐。单位错误可以制造看似惊人的性能，必须先检查量纲。

### 5. 失败后如何保留可继续学习的状态

将失败输入、配置、错误输出与环境写进小型记录，保留最近一个正确版本。不要在同一个未提交工作区连续修改五个参数后只记得“后来好了”。每次修复都应新增或保留对应回归案例，让过去踩过的边界成为工程资产。

如果某个GPU配置无法编译，继续完成CPU推导与其他受支持配置，不必让整个学习计划停住；但要明确该路径未验证。持久推进与诚实边界可以同时做到，不能用“以后再说”掩盖已承诺核心内容，也不能伪造设备成功记录。

## 第十四课：把实现读成一份可维护的代码审查

### 1. 审查输入与输出所有权

先列每个参数是只读输入、可写输出还是临时缓冲，再列是否允许别名。对每个输出写地址，写出由哪个program和哪个逻辑槽负责。若两个任务可能写同一位置，必须找到明确的原子或归约协议，否则就是需要修复的风险。

只读输入也可能被错误store修改，因此测试应保存输入副本并比较调用前后。一个算子输出正确但破坏输入，可能在后续层才暴露，定位成本很高。输入不变性属于接口合同，不是可选的额外检查。

### 2. 审查每个归约的范围与中性值

逐个找到tl.sum、tl.max和tl.dot，写出输入shape、归约轴、真实有效数量与padding值。注意dot内部也包含归约维，尾部A/B必须填零。对均值再检查除数，对softmax再检查mask进入分母前的时机。

如果归约结果广播回原张量，明确插入哪个长度一轴。很多静默错误来自方形shape下的错误广播，因此审查时应主动换成非方形例子手算。不要只看编译器是否接受shape，语言合法与数学正确是两条检查。

### 3. 审查数值链条

在每个load后标注类型，在平方、乘法、指数、累加与store前标注转换。检查epsilon是否有限正值、scale是否使用真实特征数、指数是否稳定化、全mask或空输入是否定义。对低精度扩展，尤其检查转换发生在运算前还是后。

数值链条审查应与错误测试相连。比如担心窄类型平方溢出，就构造输入幅度验证；担心scale用BD而非D，就选非二次幂D；担心权重索引错误，就让weight每项不同。每条审查意见都能落到一个具体证据，才不会变成空泛清单。

### 4. 审查编译与运行假设

源码中的constexpr、tile与精度选项构成编译合同，wrapper中的shape上界构成运行合同。若修改其中一边，另一边与测试都要跟着检查。例如扩大attention长度可能增加编译或运行成本，扩大特征宽度可能改变资源占用，不能只把数字改大。

对于版本相关API，保存固定来源与实际安装版本。教材中的解释强调稳定概念，具体工具调用仍需按版本验证。遇到API变化时，先保留数学与测试不变，再适配语法，这样可以知道变化是否仅是接口迁移还是算法行为改变。

### 5. 最终小库的完成标准

六个算子都应拥有可独立说明的数学合同、完整wrapper、参考实现、边界与负向测试、设备验证状态和性能记录入口。尚未执行GPU时，明确写未执行；完成后补真实证据。不要把一个算子成功推广成六个全部成功，也不要把CPU参考通过当成设备路径通过。

当你能不用照抄教程，从公式写出索引、mask、归约和wrapper，再用最小反例定位错误，就已经建立了独立编写算子的核心能力。后续学习框架接入、量化、MoE、异步与硬件特化，都是在这套可解释工作流上增加新的合同与资源约束。


### 6. 公开接口与内部Kernel参数分开维护

用户调用matmul时应提供矩阵与必要数学参数，不需要理解每个内部tile配置。wrapper可以选择默认配置并允许高级实验显式覆盖，但内部参数改变不应悄悄改变输出数学定义。把性能参数与语义参数分开，能够减少学习者误用，也便于后续自动调优。

相反，epsilon、causal和scale属于数学语义，必须在接口与实验记录中明确。将它们隐藏成未记录的全局变量，会让同一调用在不同环境下产生不同结果。版本更新若改变默认语义，应有清楚迁移说明与回归测试，而不是只修改Kernel常量。

最后为每个接口写一个最小完整调用，包含输入构造、结果预期与错误案例。函数签名只能告诉读者传什么参数，完整调用才能展示这些参数如何共同定义一个可验证任务。这也是从“知道算子名字”走向“能够独立使用和修改实现”的必要一步。

## 第十五课：从六个算子形成一个小库

按算子分类组织文件，保留统一输入检查与误差报告，但不要把各算子的数学合同隐藏在过度抽象中。每个Kernel旁边应有shape说明、索引说明和最小手算案例。每个wrapper旁边应有支持范围与空输入行为。

CPU测试默认运行，不导入GPU模块；GPU测试显式启用并在缺少依赖时清楚报告未执行。持续集成只通过CPU测试时，发布说明应保留设备验证缺口。这样学习者在个人PC、CPU服务器与H100之间切换时，知道哪些成果可以复用，哪些仍需补验。

性能优化采用独立提交：先保存正确基线，再修改tile、warp数、融合或流水线，每次重新运行相同边界集。最终交付包括代码、完整参考、测试结果、失败案例和环境记录，而不是只有一个能够在某台机器上运行的脚本。

复习时任选一个算子，遮住实现，只根据合同重新推导地址和归约，再与代码逐行比较；能够解释每一处差异，才算真正掌握。
