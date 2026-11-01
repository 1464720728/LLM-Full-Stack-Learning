# K08 Triton编程模型：把一块数据写清楚，再理解它如何落到GPU

Triton让程序员以一块张量为单位描述计算，再由编译器将这块工作分配到GPU线程、寄存器与指令。它降低了手写标量线程代码的负担，但没有消除地址、边界、数值与并行正确性问题。本章从普通Python函数和数组开始，逐步写出可解释的Triton程序，并明确哪些事实可以由源码推导，哪些必须查看编译产物或实际测量。

先完成K01至K07。尤其要能区分元素偏移与字节偏移、逻辑shape与物理stride、归约中性值与输出mask。没有GPU环境时，可以完成本章全部地址模拟、广播推导和CPU验收；设备代码需要安装匹配的框架、Triton与驱动后再编译运行，不能把跳过执行记为成功。

## 第一课：普通Python、张量框架与设备Kernel是三层代码

### 1. Python负责组织，Kernel负责设备计算

普通Python函数在主机上执行，可以检查参数、分配张量、选择配置并启动设备任务。Triton Kernel函数虽然使用Python语法书写，却会经过编译生成设备代码；它不是由Python解释器在GPU上逐条执行。两者可以出现在同一个文件中，但执行位置和可用操作不同。

例如wrapper读取x.numel得到元素数，创建输出张量，决定一共启动多少个program；Kernel内部用program编号计算自己负责的元素偏移，读取数据并写结果。wrapper中的列表、字典、异常与打印不应自动当作设备代码可用功能。反过来，tl.load与tl.program_id也不是普通NumPy操作。

这种分层意味着很多错误应在host尽早拒绝。输入设备不同、shape不匹配、dtype不支持、stride不符合合同，都不应等到GPU越界或编译失败才暴露。一个清晰的wrapper是算子实现的一部分，不是可以省略的“样板代码”。

### 2. 装饰器与JIT到底做什么

Python装饰器会在函数定义时把函数交给另一个对象处理。Triton的jit装饰器标记函数可被编译为设备程序，真正生成某个具体版本通常发生在第一次按特定参数与配置启动时。这个过程涉及类型、常量与目标设备等信息，不能把函数定义成功等同于Kernel已成功编译。

第一次调用可能包含编译开销，后续相同特化可能复用缓存。因此计时要区分冷启动与稳态执行。若每次都改变影响特化的参数，可能不断生成新版本；这既影响启动延迟，也影响缓存与实验可比性。K15会完整追踪编译链，本章先建立调用边界。

### 3. 一个程序处理一块元素

在CUDA入门中，我们常描述“一个线程处理一个元素”；Triton常描述“一个program处理一个元素块”。program内部的张量表达式同时表示许多逻辑元素，编译器决定这些元素如何分布到线程和寄存器。不要把program编号误当成单个线程编号。

例如长度十的向量，每个program处理四个逻辑槽，需要三个program。编号零负责零至三，编号一负责四至七，编号二负责八至十一，其中十、十一越界。program数是三，逻辑槽数是十二，真实元素数是十，底层线程数则由启动配置与编译实现决定，四者不是同一个数字。

## 第二课：从完整向量加法开始

### 1. 设备函数

~~~python
import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    x = tl.load(X + offsets, mask=valid, other=0.0)
    y = tl.load(Y + offsets, mask=valid, other=0.0)
    result = x + y
    tl.store(Z + offsets, result, mask=valid)
~~~

X、Y、Z表示设备内存起点指针，N是有效元素数。BLOCK用于构造固定形状的逻辑块，因此标注为编译期常量。program是当前实例沿grid第一轴的编号。arange产生从零到BLOCK减一的逻辑偏移向量，乘加后变成当前program的全局元素位置。

X加offsets不是把一个数组值加到另一个数组值，而是构造一组指针。tl.load根据这些指针读取值，得到逻辑张量x；只有到这一步才取得内存中的数值。把指针表达式与数据表达式分开看，可以避免把地址计算中的stride误当成数值运算。

valid是逐槽布尔张量。加载时mask为假不会访问对应非法地址，other给这些槽提供确定的替代值。写回同样使用mask，避免把无效槽写出分配范围。即使无效槽不会写回，也应为后续计算选择适当other，尤其当它们参与归约时。

### 2. 完整主机wrapper

~~~python
import torch

def add(x, y):
    if not x.is_cuda or not y.is_cuda:
        raise ValueError('CUDA tensors required')
    if x.device != y.device:
        raise ValueError('same device required')
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError('matching shape and dtype required')
    if x.dtype != torch.float32:
        raise ValueError('teaching path supports float32')
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError('contiguous tensors required')
    output = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return output
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    add_kernel[grid](x, y, output, n, BLOCK=256, num_warps=4)
    return output
~~~

输出使用empty_like继承输入shape、dtype与device，但内容未初始化，正确Kernel必须覆盖每个合法位置。空输入直接返回空输出，避免启动零工作网格。grid是一个根据编译元参数计算任务数量的函数，当前BLOCK为二百五十六，任务数是N向上除以该值。

num_warps影响一个program使用的warp数量等编译与启动组织，不是“每个program有num_warps个元素”。BLOCK也不是CUDA blockDim的直接同义词。逻辑块元素可以多于或少于底层线程数，每线程可能负责多个值，具体映射需要看编译结果与布局。

wrapper返回时，设备工作通常只是排入相应执行流，不代表已经完成。后续同流操作可通过流顺序使用结果，但主机计时或读取结果需要正确同步。K13会详细展开；本章只要求不要用普通Python调用前后时间差当作完整GPU执行时长。

### 3. 为什么明确限制float32

教学入口先限制一种类型，便于固定累加与误差预期。若允许float16、bfloat16、整数等多种输入，算术提升、输出转换、溢出与比较容差都要逐个定义。函数能编译某种类型，不代表已经完成该类型的数值验收。

后续扩展时应从接口合同出发：输入类型是什么，内部计算用什么类型，输出用什么类型，是否允许混合输入。不要仅删除dtype断言就宣称支持全部类型。每种新增类型至少需要独立边界测试和明确误差标准。

## 第三课：用CPU模拟program、offset与mask

### 1. 完整十元素轨迹

当N为十、BLOCK为四，program零的offsets为零、一、二、三，mask全真；program一为四、五、六、七，mask全真；program二为八、九、十、十一，mask为真、真、假、假。输出所有权是这三个集合中合法位置的并集，恰好覆盖零至九且不重复。

如果grid使用N整除BLOCK，只启动两个program，最后两个元素永远不写。由于output来自empty，结果可能是任意旧内存内容。若测试只检查前几个元素，错误会被遗漏。边界问题应通过完整输出与专门非整除长度测试发现。

~~~python
import numpy as np

def program_trace(n, block):
    if n < 0 or block <= 0:
        raise ValueError('invalid sizes')
    records = []
    for program in range((n + block - 1) // block):
        offsets = program * block + np.arange(block)
        records.append((program, offsets, offsets < n))
    return records

def add_program_simulator(x, y, block=4):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError('matching vectors required')
    result = np.empty_like(x + y)
    writes = np.zeros(x.size, dtype=np.int64)
    for program, offsets, valid in program_trace(x.size, block):
        xv = np.zeros(block, dtype=x.dtype)
        yv = np.zeros(block, dtype=y.dtype)
        xv[valid] = x[offsets[valid]]
        yv[valid] = y[offsets[valid]]
        values = xv + yv
        result[offsets[valid]] = values[valid]
        writes[offsets[valid]] += 1
    assert np.all(writes == 1)
    return result

x_cpu = np.arange(10, dtype=np.float32)
y_cpu = 10 * np.arange(10, dtype=np.float32)
assert np.array_equal(add_program_simulator(x_cpu, y_cpu), x_cpu + y_cpu)
~~~

这个模拟刻意先构造有效位置，再索引原数组。不能先执行x[offsets]再把越界位置替换成零，因为非法读取已经发生。这个顺序对应GPU中mask必须作用在load本身，而不是读取之后的数值选择。

写者计数writes只验证覆盖与唯一写入，不验证算术正确，所以还要比较结果。反过来，某些重复写入相同数值的错误可能让结果比较通过，却在其他算法中形成竞争；写者计数能补充发现这种结构问题。

### 2. BLOCK的特殊要求与选择

Triton许多张量构造与归约路径偏好或要求特定静态形状，常见教学写法使用二的幂BLOCK。不要把任意运行时N直接当作arange结束值，期待编译器自动生成任意动态长度向量。通常先选择合适的编译期块大小，再通过mask处理真实长度。

选择BLOCK时先保证正确覆盖，再考虑资源与性能。较大块减少program数量，却可能增加每个program寄存器与工作量；较小块增加调度任务和边界比例。对于简单逐元素算子，多个配置都可能正确，性能优劣需要在相同输入和计时边界下测量。

### 3. 零长度、单元素与尾部测试

至少测试N为零、一、BLOCK减一、BLOCK、BLOCK加一，以及几个更大非整除值。零长度验证wrapper是否合理跳过launch；单元素验证大多数逻辑槽无效时仍正确；相邻边界值能捕捉向上取整与mask符号错误。

把mask写成offsets不大于N，会让offset恰好等于N的位置越界。把它写成offsets小于N减一，会漏掉最后一个元素。使用N为一时，这两类错误特别直观，因此小测试不是“太简单没意义”，而是定位边界定义的高价值工具。

## 第四课：二维广播生成地址矩阵

### 1. 一维向量如何组成二维网格

设rows是长度BM的行索引，cols是长度BN的列索引。rows[:,None]形状为BM乘一，cols[None,:]形状为一乘BN。二者相加时广播成BM乘BN网格，每个位置同时获得对应行与列信息。

对行优先矩阵，地址偏移为rows[:,None]乘行stride，加cols[None,:]乘列stride。这个表达式没有实际复制整张输入矩阵，只构造当前逻辑tile的指针集合。随后tl.load根据该集合读取对应元素。

例如rows为二、三，cols为一、二、三，行stride五、列stride一，地址矩阵第一行为十一、十二、十三，第二行为十六、十七、十八。每行跨度与原矩阵物理行宽一致，tile自身宽度三不会改变global行stride五。

### 2. 完整CPU地址模拟

~~~python
def tile_offsets(row_start, col_start, bm, bn, stride_row, stride_col):
    rows = row_start + np.arange(bm)
    cols = col_start + np.arange(bn)
    return rows[:, None] * stride_row + cols[None, :] * stride_col

offsets = tile_offsets(2, 1, 2, 3, 5, 1)
assert np.array_equal(offsets, [[11, 12, 13], [16, 17, 18]])
~~~

如果忘记None轴，两个同长度一维向量相加只得到逐项配对，而不是外积网格；若长度不同，可能直接广播失败。这个错误很常见，因为代码短而且某些方形尺寸下能运行。先写每个中间shape，再看表达式，能够避免依赖试错理解广播。

mask也用同样的二维广播方式构造：行有效条件变成BM乘一，列有效条件变成一乘BN，取逻辑与得到整个tile有效区域。不能用Python的and连接张量条件，应使用适合逐元素布尔运算的表达式并加括号，避免运算优先级误解。

### 3. 转置读取与转置写回不是同一公式

读取一个矩阵tile时，输入地址可按row乘输入行stride加col。若输出是转置矩阵，输出逻辑行变成原col，输出逻辑列变成原row，因此输出地址应使用col乘输出行stride加row。输出行stride取决于转置后形状，不是继续使用输入列数。

设输入三乘五，转置输出五乘三。输入[二,四]偏移十四，输出[四,二]偏移十四，单个角落恰好相同，可能误导你认为公式无须改变。换输入[一,三]，输入偏移八，输出[三,一]偏移十，差异才显现。测试多个位置而不是只看角落很重要。

## 第五课：tl.load、tl.store与数值选择的边界

### 1. Mask保护的是访问，不是整个表达式

tl.load的mask决定哪些指针位置会被实际读取。tl.store的mask决定哪些位置写回。它们不能自动保证指针表达式中的整数运算不溢出，也不能让其他已经发生的非法访问变安全。接口仍需要正确索引类型和合法storage范围。

tl.where根据条件选择数值，但不应当作短路控制流来保护危险load。错误写法是先对全部指针执行load，再用where把越界槽替换零。正确写法把mask直接传给load，并提供合适other。这个区别与NumPy中的先索引再选择类似。

### 2. Other取什么取决于后续计算

逐元素加法尾部不写回，other零通常方便；求和归约无效槽应为零；最大值归约通常用负无穷；最小值用正无穷；乘积用一。不存在“所有mask都填零”的通用规则。

例如真实输入全为负数，归约最大值时把padding填零，会让结果错误变成零。softmax随后减去错误最大值在数学上可能仍得到同样归一化比例，但数值范围和其他最大值输出已改变；若后续还有argmax或门控，影响更直接。选择中性值应从算子定义推导，而不是凭常见代码习惯。

### 3. 内存mask与数学mask

内存mask保护访问在分配范围内，数学mask决定某个合法元素是否参与计算。注意力中未来key地址可能完全合法，但因果语义要求排除；向量尾部则地址本身不合法。两个mask可以合并传给load或分数处理，但它们的理由不同。

有时数学mask作用于计算后的score，而不是输入K或V。把不可见K加载成零并不等于排除该key，因为零点积仍有指数权重。必须在归一化前把不可见score排除。这说明“已经给load加mask”不代表整个算子的数学mask已经正确。

## 第六课：归约轴与广播回原形状

### 1. 行归约的shape变化

对形状BM乘BN的逻辑张量，沿axis一求和得到长度BM的向量，每行一个结果；沿axis零求和得到长度BN，每列一个结果。归约输出轴消失后，若要广播回原矩阵，需要明确补上长度一轴。

例如每行最大值m长度BM，计算x减m[:,None]。若写x减m而BM恰好等于BN，广播可能沿列应用m，程序仍能运行但语义错误。使用非方形tile与每行明显不同的偏置，是捕捉这类错误的有效方法。

### 2. 完整行Softmax Kernel

~~~python
@triton.jit
def row_softmax_kernel(X, Y, M, N,
                       STRIDE_X_ROW: tl.constexpr,
                       STRIDE_Y_ROW: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(X + row * STRIDE_X_ROW + cols,
                mask=valid, other=-float('inf')).to(tl.float32)
    maximum = tl.max(x, axis=0)
    weights = tl.exp(x - maximum)
    denominator = tl.sum(weights, axis=0)
    probability = weights / denominator
    tl.store(Y + row * STRIDE_Y_ROW + cols,
             probability, mask=valid)
~~~

每个program负责一行，因此这里的逻辑x是一维BLOCK向量，axis零就是列归约。row不是张量行向量，而是当前program的标量行号。不要把二维tile例子的axis一机械套到这个一维版本，否则会产生无效轴或错误理解。

wrapper必须保证N大于零、BLOCK至少覆盖N且适合编译要求，grid恰好M行。若M为零则跳过launch。输入每行列stride被限定为一，行stride可以单独传入；若要支持列stride不为一，还要在地址中乘对应stride，不能只传行stride就宣称支持全部非连续布局。

输入转float32后再做exp和求和，明确累加精度。输出store会按输出指针类型转换，因此输出dtype也是合同的一部分。若输入float16而输出float16，内部单精度并不阻止最后一次舍入；测试应使用与目标精度匹配的容差。

### 3. 归约不是普通Python循环

tl.sum描述逻辑张量的归约，编译器选择线程间与线程内的实现方式。你不必手写每一条shared屏障，但仍要理解归约范围、逻辑布局和资源需求。错误轴、错误中性值或过大tile，编译器不会自动替你纠正数学定义。

编译器生成的归约可能使用寄存器交换、shared或其他目标相关机制。源码中没有显式__syncthreads，不代表设备上完全没有同步与通信成本。性能分析需要查看生成代码与计数，不能仅按Python行数估算工作量。

## 第七课：编译期常量与运行时值

### 1. 为什么shape通常需要静态信息

编译器要为逻辑张量安排寄存器、线程分布与指令，因此许多块shape必须在编译时已知。BLOCK、BM、BN、BK通常标为tl.constexpr。N、M、K可以作为运行时尺寸参与边界判断和循环，具体是否特化取决于接口与编译策略。

把每个输入值都标成constexpr可能生成大量版本，增加编译与缓存开销；把必须决定张量shape的值留作普通运行时参数则可能无法编译。合理做法是区分“决定程序结构的元参数”和“当前问题的数据或范围”，并为常见形状选择有限配置集合。

### 2. 常量分支与数据分支

如果分支条件由constexpr决定，编译器可以为不同特化生成不同代码路径，未选分支可能被消除。若条件来自设备数据，运行时需要执行相应控制流或选择逻辑，可能影响分歧与资源。两者都写成if，并不意味着执行成本相同。

例如是否使用bias可作为编译期布尔参数，生成带或不带bias读取的版本；每个元素是否大于零则是数据条件，通常用逐元素最大值或where表达激活。不要为了让所有条件都静态化，把每个数据值传回主机决定分支，那会破坏并行与异步执行。

### 3. 循环展开与代码体积

固定小次数循环可以展开，减少循环控制并暴露优化机会；过度展开则可能增大代码、寄存器活跃范围和编译时间。沿很长K维完全展开通常不是简单的好事。编译器指令与API可能随版本变化，学习时先理解展开的代价，再查当前固定版本支持的写法。

同一数学算法可以使用运行时循环、编译期静态循环或软件流水线循环。选择不同形式会改变调度与生成代码，但不应改变输入覆盖与归约不变量。先在CPU模拟每轮索引，再查看编译器是否按预期组织，而不是把某个语法当作自动加速开关。

## 第八课：布局不是NumPy stride的另一个名字

### 1. 两种布局分别描述什么

张量的物理stride描述逻辑索引如何映射到全局内存地址。编译器内部布局描述逻辑块中的值如何分布到线程、lane、warp和寄存器等执行资源。它们相关，但不是同一个概念。一个连续输入可以被读取成多种线程分布，同一种逻辑块分布也可以从不同stride输入加载。

在Triton普通语言层，很多布局选择由编译器推导。更底层或实验性接口可能允许更显式控制，但不能把某个版本的内部布局语法当成永远稳定的用户API。阅读编译IR时应记录版本与目标，区分语言语义与当前实现策略。

### 2. 一个抽象分布例子

设逻辑向量有八个元素，四个参与者每人持有两个值。一种分布是参与者零持零、一，参与者一持二、三；另一种是参与者零持零、四，参与者一持一、五。两种都覆盖相同向量，但相邻参与者同一条load访问的地址关系不同，可能影响合并访存。

如果要沿某个维度归约，数据分布还决定需要多少跨线程交换。某些布局让一个线程持有一段归约元素，先局部求和再通信；另一些布局让相关元素分散。编译器选择布局时需要兼顾load、dot、reduce与store，而不只是让某一步单独最好。

### 3. 布局转换意味着真实工作

当某个操作偏好的分布与下一个操作不同，编译器可能插入布局转换。转换可能只是重新解释寄存器，也可能涉及寄存器交换、shared读写或其他指令。不能因为源代码只写了一个reshape或transpose，就假定完全免费；也不能认为每个逻辑转置都一定搬运全局内存。

例如GEMM输入片段、行归约结果与输出store可能需要不同组织。阅读IR时，如果看到转换，应问它连接哪两个操作、搬运哪些逻辑值、作用范围是什么。之后再用低层代码与性能数据确认代价，避免只凭操作名猜测。

### 4. 完整二维复制Kernel把广播知识连起来

~~~python
@triton.jit
def copy_matrix_kernel(X, Y, M, N,
                       SX0, SX1, SY0, SY1,
                       BM: tl.constexpr, BN: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * BM + tl.arange(0, BM)
    cols = col_block * BN + tl.arange(0, BN)
    offsets_x = rows[:, None] * SX0 + cols[None, :] * SX1
    offsets_y = rows[:, None] * SY0 + cols[None, :] * SY1
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    values = tl.load(X + offsets_x, mask=valid, other=0.0)
    tl.store(Y + offsets_y, values, mask=valid)
~~~

grid应为M向上除以BM、N向上除以BN。两组stride独立，使输入和输出可以采用不同物理布局。每个program处理一个逻辑矩形，边界mask同时考虑行与列。这个算子没有算术复杂度，特别适合先验证地址与写者，再将同一映射扩展为bias、激活或归一化前处理。

设输入是连续三乘五，输出是同一逻辑shape但物理每行八槽，BM二、BN四。右上program的row_block为零、col_block为一，rows为零、一，cols为四、五、六、七。合法位置只有两行的第一列，输入偏移四、九，输出偏移四、十二。其余槽必须完全不写，不能因为输出storage容量足够就写入逻辑padding。

输出stride若为零，会让同一行多个逻辑列映射到同一地址。这个Kernel并没有定义重复写者的顺序，因此wrapper应拒绝重叠输出布局。输入零stride读取广播则可以合法，只要storage范围有效。相同stride数值出现在输入与输出时，合法性可能不同，原因是读共享与写冲突的区别。

### 5. Alias不是靠mask就能解决

假设Y与X完全相同storage且相同逻辑布局，纯复制可能每个位置读后写自身，通常可以设计成安全的原地操作；但若Y是X的转置视图，某个program可能先覆盖另一个program尚未读取的输入。即使所有地址都在范围内、mask也正确，仍存在跨任务依赖。

因此支持原地与支持任意别名是不同承诺。最简单的复制或变换接口可以要求输出独立分配，明确拒绝重叠输入输出。若希望支持某种原地变换，需要专门算法与证明，不能靠把store放在load后面就认为全局读写顺序安全。

单个program内部先load完整tile再store，也只能保护该tile已经读入的部分，不能保证其他program的数据已读完。Triton程序之间没有普通全局屏障，调度顺序不可依赖。这个问题与CUDA相同，语言层次提高并不会自动消除算法级别的别名危险。

### 6. 对齐提示与整除提示是承诺

某些优化接口允许告诉编译器地址对齐、数值倍数关系或连续性信息。这些提示用于生成更高效的访问与指令，前提是你提供的事实在所有受支持输入上成立。它们不是运行时修复，也不应拿来“试试看能不能更快”。

例如声称偏移总是某个向量宽度的倍数，必须结合起始指针、stride、program起点与子视图偏移一起证明。基础分配可能对齐，但从第一个元素之后的切片起点未必仍满足更严格对齐。只检查主张量而忽略子视图offset，会让提示失真。

如果提示错误，编译器可能根据不成立的前提优化，结果不再可靠。调试时应先去掉非必要提示建立正确基线，再逐个加入并验证其前提。性能优化中的“我认为应该连续”必须变成明确地址公式，而不是经验猜测。

### 7. 逻辑转置与布局转换的三种情况

第一种是只改变指针网格，让load从转置视图读取；此时全局访问模式改变，可能影响合并。第二种是已经加载的逻辑tile做维度置换，编译器可能需要改变值在执行资源中的分布。第三种是把结果写成物理转置矩阵，涉及完整输出地址变化。三个操作都可能在代码中出现transpose一词，但成本与作用位置不同。

阅读性能报告时，应问测量的是哪一种。如果只改变视图元数据，没有实际Kernel执行，就不能与物理转置带宽直接比较；如果编译器内部做片段重排，也不能把它说成复制整个输入矩阵。用数据生命周期定位操作所在层次，比依赖函数名更准确。

对一个具体tile，可以把每个逻辑元素标成其原始全局坐标，再跟踪经过转置后属于哪个逻辑位置、由哪个输出地址保存。即使暂时不知道底层lane映射，这份逻辑轨迹也能验证数学正确性；随后再查看IR补充执行布局与成本。

### 8. 为什么读懂Triton还需要CUDA基础

Triton替你生成许多线程级细节，但性能仍受寄存器、shared、warp调度、内存合并与同步影响。若不了解这些资源，看到编译报告中的spill、共享内存占用或归约通信时就无法解释。K01至K07并不是使用Triton之前的繁琐门槛，而是理解其生成代码的语言。

反过来，也不应把手写CUDA经验机械映射到Triton源码。例如一个逻辑块不是一个线程，一个tl.sum不等于你熟悉的某段固定shared归约代码。正确做法是先尊重高层语义，再用硬件知识解释编译结果；既不忽略底层，也不凭旧经验猜内部布局。

当你能在这两个层次之间往返，才开始具备独立修改算子的能力：高层公式保证数学与边界，低层证据解释资源与性能。单纯复制一个教程参数可以跑起来，但无法告诉你换shape、换dtype或换设备后为什么失效。


## 第九课：从Python源码到IR与设备代码

### 1. 编译阶段的职责

Triton前端把Kernel表达式转成带张量操作与类型信息的中间表示，随后进行布局、并行映射和目标相关降低，最终生成设备可执行代码。不同版本的IR名称、缓存结构与调试接口可能变化，因此本章不要求死记某个输出文件路径。

阅读时先在较高层IR确认程序做了什么：指针偏移、mask、load、加法、归约和store是否存在；再在较低层看如何分配到线程与指令；最后才分析目标汇编。直接从机器指令猜原始数学含义，难度远高于分层追踪。

### 2. 一个向量加法应保留哪些语义

无论优化如何进行，都应能找到当前program负责的全局偏移、对N的边界判断、两个输入读取、逐元素加法和一个输出写回。某些表达式可能被融合或重排，变量名也可能消失，但数据依赖应保持。

若编译器把常量BLOCK折叠进地址计算，这是正常优化；若生成向量化load，需要满足对应对齐与有效性条件；若尾部使用谓词控制，应该与源码mask一致。你可以用不同N与BLOCK观察生成变化，但每次仍要跑正确性测试，不能仅看汇编更短就认为更好。

### 3. 调试信息与可复现记录

保存Kernel源码版本、Triton版本、框架版本、GPU型号、驱动信息、输入shape/stride/dtype和编译元参数。若报告编译错误，附最小可复现调用与完整错误位置，而不是只贴最后一行。编译缓存中的某个文件不能脱离对应源码与参数单独解释。

本仓库当前环境没有Triton与GPU，因此本章设备代码没有在交付环境编译执行。CPU模拟验证的是索引与数学结构，不能替代编译检查；编译检查也不能替代设备结果与性能。把三层证据分开记录，是后续修改复杂算子时的重要习惯。

## 第十课：四类错误及其修复

第一类是program数少一块。N为十、BLOCK四却只启动两个program，结果最后两项未写。修复grid向上取整，并用写者计数验证每个合法输出恰好一次。不要通过把output预置零掩盖漏写，因为某些正确结果也可能恰好为零。

第二类是load后才mask。先读取X加全部offsets，再用where选择合法值，越界访问已经发生。修复把valid传给load，store同样加mask。测试N为BLOCK加一，最后program只有一个合法槽，最容易观察问题。

第三类是广播轴错误。二维行归约结果长度BM，直接减到BM乘BN张量，若BM等于BN可能静默按列广播。修复显式增加末尾轴，并用BM不等于BN、每行偏置不同的输入检查。代码能编译不代表广播语义符合目标。

第四类是dtype与输出分配不匹配。wrapper默认创建某种dtype输出，Kernel却按另一种数值合同计算。修复输出显式继承或指定合同dtype，检查所有输入同设备同类型，并分别记录内部累加与最终store转换。不能只修改误差容差绕过类型错误。

## 第十一课：十二道递进练习与完整答案

### 练习一：列出program的职责

题目：N十七、BLOCK八，共多少program，最后负责哪些位置。答案：三个program，最后逻辑偏移十六至二十三，只有十六合法。底层线程数不能仅由这两个数字确定。

### 练习二：模拟完整覆盖

~~~python
for n in (0, 1, 7, 8, 9, 17):
    trace = program_trace(n, 8)
    covered = [int(i) for _, offsets, valid in trace for i in offsets[valid]]
    assert covered == list(range(n))
~~~

这个答案验证连续一维映射的覆盖顺序。若采用重排调度，覆盖列表顺序可能不同，应比较排序后的集合并额外检查重复次数。

### 练习三：写二维地址

题目：rows二、四，cols一、三，行stride七、列stride二。答案地址矩阵为十六、二十；三十、三十四。注意逻辑列间距二表示非连续列视图，不应再次乘元素字节数。

### 练习四：选择归约填充值

答案：求和零，最大值负无穷，最小值正无穷，乘积一。若求平均，除数必须是真实有效数量，不能直接除补齐BLOCK；若某些mask并非尾部而是任意稀疏有效集，数量还需单独归约。

### 练习五：区分运行时与编译期

题目：BLOCK与N分别为何不同。答案：BLOCK决定逻辑块shape与编译组织，常作为constexpr；N决定当前有效元素范围，可运行时参与mask。把N也特化可能有收益或成本，但不是正确性所必需的统一规则。

### 练习六：完整二维tile mask模拟

~~~python
rows = 2 + np.arange(2)
cols = 3 + np.arange(4)
valid = (rows[:, None] < 3) & (cols[None, :] < 5)
assert np.array_equal(valid,
    [[True, True, False, False],
     [False, False, False, False]])
~~~

第一行是真实行二，只有列三、四有效；第二行对应越界行三，全部无效。这个mask同时保护行与列，不能只检查扁平偏移小于总元素数，因为越界列可能落入下一行合法storage。

### 练习七：解释扁平边界不足

题目：三乘五矩阵访问逻辑位置[一,六]，偏移十一小于十五，是否有效。答案：无效，列六超出逻辑列范围，偏移十一实际上指向另一行的元素。物理storage范围与逻辑索引范围必须同时满足。

### 练习八：实现CPU行softmax块模拟

~~~python
def row_softmax_simulator(x, block):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not 0 < x.shape[1] <= block:
        raise ValueError('positive row width within block required')
    out = np.empty_like(x)
    for row in range(x.shape[0]):
        values = np.full(block, -np.inf)
        values[:x.shape[1]] = x[row]
        weights = np.exp(values - values.max())
        out[row] = (weights / weights.sum())[:x.shape[1]]
    return out

case = np.array([[-3., -2., -1.], [1000., 1001., 1002.]])
result = row_softmax_simulator(case, 4)
assert np.allclose(result[0], result[1])
assert np.allclose(result.sum(axis=1), 1)
~~~

### 练习九：找出布局与stride的区别

答案：stride决定逻辑元素在全局内存中的地址；编译布局决定逻辑块元素由哪些执行资源持有。修改输入stride可能改变load地址，修改内部布局可能改变线程间通信，两者不能互相替代。

### 练习十：正确计时的必要条件

答案：先完成编译与暖机，使用适当GPU事件或经过验证的基准工具测量设备工作，明确是否包含分配和传输，重复多次并报告统计量。普通主机函数返回时间不能直接代表异步Kernel耗时。

### 练习十一：给wrapper增加非连续支持

完整方向是传入每个输入的实际stride，并在地址中逐轴使用；同时验证输出布局与别名合同。若只想支持连续路径，保持明确拒绝也是正确设计。不能在wrapper调用contiguous后不记录复制成本，却与原地支持任意stride的基线比较性能。

### 练习十二：设计编译与运行证据表

答案分三行：CPU地址与数学模拟，记录输入与断言；设备编译，记录版本、目标与配置；GPU实际运行，记录设备、shape、误差与同步方式。三行分别填完成、失败或未执行，不能把任一行的通过复制到其他行。

## 第十二课：把地址、数值和写者三条轨迹同时写清楚

### 1. 按行加偏置的完整例子

考虑输入X为三行五列，bias长度为三，每行加自己的偏置。这里与K06按列bias不同，必须先说明广播轴。选择一个program负责一行，逻辑列块BLOCK为八。program一读取第二行五个元素，给它们都加bias[一]，其余三个槽不读取、不写回。

~~~python
@triton.jit
def row_bias_kernel(X, BIAS, Y, M, N,
                    STRIDE_X_ROW, STRIDE_X_COL,
                    STRIDE_Y_ROW, STRIDE_Y_COL,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = (row < M) & (cols < N)
    values = tl.load(X + row * STRIDE_X_ROW + cols * STRIDE_X_COL,
                     mask=valid, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + row, mask=row < M, other=0.0).to(tl.float32)
    result = values + bias
    tl.store(Y + row * STRIDE_Y_ROW + cols * STRIDE_Y_COL,
             result, mask=valid)
~~~

这一版把stride作为运行时参数，表达更一般的地址关系；是否进一步将常见stride特化，应由实际编译与性能需求决定。输入列stride可不为一，因此可以读取隔列视图。输出stride也显式传入，但wrapper必须保证合法输出元素不重叠，否则多个槽可能写同一地址。

bias是标量load，不需要为每个列槽重复构造同一个指针向量。加到values时，标量广播到全部逻辑列。无效列即使得到bias数值也不会写回，因此本例other零不会影响合法结果；若后续又对整块求和，无效列中的bias就会错误参与，需要再次mask或改变计算组织。

这揭示了一个容易忽略的原则：在load时填中性值，并不保证经过任意后续变换后仍是中性值。零加bias不再是零，负无穷乘零可能变NaN。每次将逐元素结果送入归约前，都应重新检查无效槽的数值含义。

### 2. 完整非连续地址手算

假设X的物理storage包含三行，每行八个槽，逻辑只取每行前五项，则行stride为八、列stride为一。program二的合法地址是十六至二十，而不是十至十四。若bias为十、二十、三十，第二行与第三行的偏置地址分别是一和二，不能乘X的行stride。

再把X改成每行隔一个元素取值，行stride十二、列stride二，逻辑五列对应偏移零、二、四、六、八。program一读取十二、十四、十六、十八、二十。这里的“列数五”只决定合法范围，实际地址间距由stride二决定。BLOCK八只决定逻辑计算槽，不应进入global行stride公式。

若输出Y是连续三乘五，输出行stride五、列stride一。输入和输出stride不同完全合法，只要每个逻辑元素读取正确位置并写到对应输出。把一个STRIDE参数同时用于X和Y会在这种情况下出错，因此接口中分别命名具有实际意义。

### 3. CPU对照实现与哨兵保护

~~~python
def row_bias_storage_reference(storage, rows, cols, sr, sc, bias):
    storage = np.asarray(storage, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (rows,):
        raise ValueError('one bias per row required')
    result = np.empty((rows, cols), dtype=np.float64)
    touched = []
    for row in range(rows):
        for col in range(cols):
            address = row * sr + col * sc
            if not 0 <= address < storage.size:
                raise ValueError('address outside storage')
            result[row, col] = storage[address] + bias[row]
            touched.append(address)
    return result, touched

storage = np.full(36, -999.0)
for row in range(3):
    for col in range(5):
        storage[row * 12 + col * 2] = 10 * row + col
biased, touched = row_bias_storage_reference(
    storage, 3, 5, 12, 2, [100, 200, 300])
assert np.array_equal(biased,
    [[100, 101, 102, 103, 104],
     [210, 211, 212, 213, 214],
     [320, 321, 322, 323, 324]])
assert all(storage[index] != -999 for index in touched)
~~~

哨兵值放在不属于逻辑矩阵的storage槽中。如果错误stride读到padding，结果会出现明显异常。正式测试还可以在输出前后放保护区域，检查没有被越界写改动；但在GPU上仅靠哨兵未改变不能证明没有越界读取，仍需正确mask与工具检查。

这个参考返回touched地址列表，让你能把数值错误与地址错误分开。若地址集合正确但数值错误，检查bias、类型与算术；若地址集合错误，先修索引，不要调容差。把所有错误都归结为“GPU精度不同”会掩盖最常见的程序缺陷。

### 4. Gather读取为何与普通连续读取不同

Gather根据索引数组从输入中选取元素。逻辑输出位置连续，但输入地址由数据决定。先读取索引，再根据索引构造第二次load指针，因此存在数据依赖。索引必须满足零到输入长度减一，输出尾部mask与索引合法mask是两层不同条件。

例如输入十、二十、三十、四十，索引三、零、三，输出四十、十、四十。重复读取同一输入地址是合法的，不需要原子；不同输出槽各自拥有独立写者。若把这个逻辑改成scatter，重复索引则会让多个槽写同一输出地址，必须定义覆盖或累加语义。

Triton中不能假设数据索引总合法而只检查输出长度。若索引来自外部输入，应在接口或设备路径上定义越界行为，例如报错、填零或拒绝；教学可以先用CPU验证索引范围。性能优化不能默默删除必要语义检查而改变合同。

### 5. Scatter与原子操作的语义

设把值一、二、三按索引零、零、一累加到长度二输出，正确结果为三、三。两个逻辑槽都写输出零，普通store无法表达累加合同，最后结果取决于写入顺序，甚至存在数据竞争。可以使用原子加，也可以先排序或分组归约后唯一写回。

原子保证某个更新按其定义不可分割地作用于目标，不自动保证整个算法的全局顺序。浮点原子累加的完成顺序可能变化，导致末位差异。若需要确定性，可能选择固定分组与归约树，代价是额外工作。K12的MoE分桶会再次遇到索引、重复与恢复顺序问题。

输出缓冲必须先初始化为累加中性值。使用empty后直接atomic_add，会把未初始化内容也加入结果。初始化与累加若在不同Kernel中，应通过正确流顺序建立依赖；不能让一个program负责清零、其他program同时累加，并假定清零一定先完成。

### 6. 多program归约为何不能直接复用tl.sum

tl.sum归约的是当前program中的逻辑张量，不会自动汇总其他program的结果。若长度很大，多个program各算部分和，需要把partial写到独立位置，再由第二次launch或其他明确机制合并。把每个program的局部sum都store到同一标量地址，只会产生竞争。

~~~python
@triton.jit
def partial_sum_kernel(X, PARTIAL, N, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N,
                     other=0.0).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(PARTIAL + program, total)
~~~

每个program写自己的partial位置，因此第一阶段无写冲突。第二阶段可以用框架sum作为清晰起点，也可以再写一个受限Triton归约。若partial长度仍很大，可以递归分阶段，但每层都要记录真实长度与缓冲大小。

不要在普通Kernel中自制“所有program等齐”的全局barrier。部分任务可能尚未获得硬件资源，已经驻留的任务若一直等待未驻留任务，就可能死锁。K03讨论过同样问题，Triton抽象不会改变GPU资源调度的基本约束。

## 第十三课：类型、数值提升与接口验证的完整工作流

### 1. 指针类型决定load和store的元素解释

同一个地址位模式，按float16、float32或整数读取会得到不同解释，指针元素类型因此是Kernel类型系统的重要部分。Triton通常从传入张量获得类型信息，但这不表示可以把任意原始地址强制解释为另一类型而不考虑对齐、大小和编码。

指针加一移动一个该指针元素，不是固定一个字节。若输入float16、输出float32，相同元素偏移对应不同字节距离。代码中的逻辑索引仍按元素计算，分配与低层字节接口才需要乘各自元素大小。把输入与输出的字节stride混用会造成严重地址错误。

### 2. 整数索引与浮点数值不要混淆

索引计算应保持整数语义，尤其是整除、取模和边界比较。把索引转换成float再参与大地址计算，可能因有效位不足失去整数精确性。即使小数组测试正常，尺寸变大也可能出现地址跳跃或重复。

对非常大张量，三十二位索引的乘法可能溢出，因此wrapper与Kernel要明确使用足够宽的类型。并不是每个教学算子都必须支持极端规模，但应写清受限范围。一个小规模实现的正确性证据不能自动覆盖任意大shape。

### 3. 提升必须发生在危险运算之前

若输入为半精度，先平方再转float32与先转float32再平方不同。前者可能已经在窄类型中溢出或舍入，后者保留更宽的中间范围。RMSNorm、方差和点积都需要认真检查转换位置，而不是只看最终变量声明。

同样，整数矩阵乘法若先在int8数组上执行点积，再把结果转int32，不能恢复已经溢出的累加。K11会给出具体编码例子；本章要求你在每个算术节点上标注输入类型、结果类型与转换时机，避免被简短表达式掩盖。

### 4. 常量的类型也可能影响表达式

零、一、epsilon等字面量参与类型推导，具体规则取决于语言与版本。教学代码在关键位置显式转换到目标累加类型，减少依赖隐含提升的歧义。尤其是整数除法与浮点除法、负无穷常量与窄类型转换，都应通过小例子确认。

不要为追求“代码更短”删除所有to转换。显式类型既是数值合同，也是阅读提示。另一方面，重复无意义转换可能增加指令或阻碍优化，所以应在理解数据流后保留必要转换，而不是机械地每行都cast。

### 5. Wrapper不是只检查shape

完整wrapper需要检查设备一致、dtype支持、维数、逻辑shape、stride、storage与别名限制、标量参数范围，以及空输入行为。对epsilon应检查有限且为正，对块大小应检查合法，对输出维度乘积应考虑溢出或不合理分配。

如果不支持非连续输入，可以拒绝或显式复制。选择复制时，返回行为与计时边界必须说明：用户看到的完整算子成本包括复制，不能在性能表中只统计复制后的Kernel而声称是通用stride实现的速度。正确接口与诚实测量是同一工程问题的两面。

### 6. 一个可复用的CPU验证函数

~~~python
def compare_arrays(actual, expected, atol=1e-6, rtol=1e-5):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f'shape: {actual.shape} != {expected.shape}')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('non-finite value in comparison')
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    threshold = atol + rtol * np.abs(expected.astype(np.float64))
    failing = absolute > threshold
    if failing.any():
        flat = int(np.argmax(absolute))
        index = np.unravel_index(flat, actual.shape)
        raise AssertionError(
            f'largest absolute error at {index}: '
            f'actual={actual[index]}, expected={expected[index]}, '
            f'abs={absolute[index]}, allowed={threshold[index]}')
    return float(absolute.max()) if absolute.size else 0.0

assert compare_arrays([1, 2, 3], [1, 2, 3]) == 0.0
~~~

先检查shape是为了防止近似比较函数自动广播，掩盖错误输出维度。先检查有限性是为了避免NaN让误差统计变得不直观。将误差计算转float64减少比较器自身的舍入影响，但参考精度与目标误差来源仍需独立解释。

这里报告最大绝对误差位置，便于定位大偏差；正式数值分析还可以同时报告最大相对误差、均方误差和误差分布。参考接近零时相对误差不稳定，因此不能只用相对误差阈值。容差不是“调到通过”的旋钮，应与输入/累加/输出精度和运算长度相关。

### 7. 负向测试要确保真的会失败

验证器自身也需要一个负向控制：故意改坏一个输出，确认比较器报错；故意传错shape，确认不会被广播接受。否则你可能写了一个从不触发失败的测试框架，却把所有绿色结果当成可靠证据。

~~~python
def must_fail(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError('negative control unexpectedly passed')

must_fail(lambda: compare_arrays([1, 2], [[1, 2]]))
must_fail(lambda: compare_arrays([1, 99], [1, 2]))
~~~

负向控制不是为了增加形式上的测试数量，而是验证测试确实能区分正确与错误行为。对GPU算子还可以故意用错误mask或漏写尾部的CPU模拟作为对照，确认所选输入足以暴露问题，再把同一测试用于真实实现。

### 8. 可重复随机测试与确定性构造

固定随机seed便于复现某次失败，但随机输入并不能替代结构性构造。全零、全负、单元素、非整除、非方形、不同stride和大公共偏置分别针对不同错误。每种测试都应写明它试图暴露哪一类缺陷。

随机矩阵适合覆盖一般数值组合，手工编码矩阵适合定位地址，极端但有限值适合检查数值范围。把三者混合成小而有意义的测试集，比盲目生成大量同分布随机数组更有效。失败时保存具体输入，避免只保存seed却在未来依赖变化后无法完全重现生成序列。

## 第十四课：启动配置、编译诊断与实验记录

### 1. Grid的维数与program_id轴

一维grid常用于向量块或逐行算子，二维grid可分别对应输出行块与列块，第三维可用于batch或head。program_id的轴编号必须与grid定义一致。若使用一维线性任务编号再解码二维tile，应写出除法与取模规则，并验证边界。

例如M为五、N为七，BM为二、BN为四，则输出tile网格为三行两列，共六个任务。线性program号四若按行优先解码，行块号为四整除二等于二，列块号为四模二等于零，对应真实行四和padding行五、列零至三。每个坐标都应能从公式还原。

若使用二维grid直接返回行块数与列块数，Kernel中哪个轴对应行、哪个对应列没有唯一强制习惯，但必须一致。变量命名pid_m与pid_n比笼统pid0、pid1更能减少混淆，尤其在后续加入batch/head维时。

### 2. num_warps影响什么、不直接决定什么

num_warps描述一个program的某些执行资源配置，编译器据此安排逻辑块。它会影响可用线程、寄存器分布、归约实现与占用等，但不能单独推出具体每个lane拥有哪些元素。不同操作、shape与目标可能生成不同布局。

例如BLOCK为二百五十六，num_warps为四时，直觉上平均每个底层线程处理若干逻辑元素，但实际映射可能受向量化与布局影响，不能把平均数当作逐lane合同。编写正确Triton代码应依赖逻辑张量语义，而不是未声明的内部映射。

调参时改变num_warps后，先重新检查正确性，再记录资源与性能。某些配置可能不被支持或资源超限，失败应被明确记录，而不是静默跳过后只展示成功最快的一项。完整实验表有助于理解限制来自哪里。

### 3. num_stages与流水线不是同义于循环次数

某些路径通过阶段参数控制软件流水线与缓冲安排。阶段数增加可能提高加载与计算重叠，也可能增加shared或寄存器占用。它不是输入被分成多少块，也不是程序执行多少轮；实际K循环轮数由问题大小与BK决定。

将阶段参数从二改成三，不会自动减少数学运算量。若性能改善，可能是延迟隐藏更好；若下降，可能是资源压力或小问题预热收尾占比增加。应结合K06双缓冲状态理解，而不是把更大阶段数当作普遍更高级的配置。

### 4. 编译错误的四层分类

第一层是Python组织错误，例如函数名未定义、参数位置不匹配、模块未安装；这发生在进入设备编译之前。第二层是语言语义错误，例如张量shape不兼容、归约轴不存在、需要编译常量却传运行时值。第三层是目标降低或资源问题，例如某种dot配置不受目标支持。第四层是设备运行错误，例如越界访问或非法地址。

不同层次需要不同排查方式。Python错误先检查普通调用；shape错误先列中间维度；目标问题先核对支持类型与配置；运行错误先缩小输入并检查地址、mask和同步。不要把所有失败都归结为“驱动问题”，也不要在没有证据时反复重装环境。

### 5. 最小复现应保留哪些信息

最小复现包含一个Kernel、一个wrapper、一个确定输入和一条失败断言。删去无关模型、数据加载与网络调用，保留实际触发问题的shape、stride、dtype和配置。若错误只在非连续输入出现，不能为了简化把输入改成连续后仍宣称复现了同一问题。

编译失败时保留完整错误栈与生成配置，运行错误时保留同步点前后的调用顺序。GPU异步可能让错误在后续操作才报告，因此报错行不一定是根因所在。可以临时在小测试中增加同步缩小范围，但不要把调试同步永久留进性能路径而不说明。

### 6. 从IR寻找一个具体索引

以向量加法为例，先找program编号如何进入偏移，再找BLOCK常量乘法与arange组合，最后追踪比较N的谓词如何连接load与store。即使变量名被重命名，数据依赖图仍应能建立对应。读IR时每次只追踪一个输出槽，比一次理解全部优化更有效。

对于二维tile，重点追踪两个扩展维度与stride乘法，确认广播形成外积网格。对于归约，确认轴与输出shape。对于dot，确认操作数布局、输入精度与累加类型。每类算子有不同的关键节点，不能用同一份“看有没有load/store”的浅层检查代替全部理解。

### 7. 编译缓存与实验隔离

同一源码在不同元参数、dtype或目标设备下可能生成不同产物。缓存命中可以减少启动成本，但也意味着你必须确认正在测量的是预期版本。修改代码后若结果似乎没有变化，应先核对调用路径与配置，而不是立即认为编译器忽略修改。

不要随意删除整个系统缓存作为常规调试步骤。更好的做法是记录版本、使用独立实验目录、明确冷启动与热启动测试，并依照当前工具文档选择受支持的缓存控制方式。缓存目录与调试开关属于版本相关接口，实际操作前应查对应安装版本。

### 8. 解释器模式与真实GPU的边界

某些开发工具提供解释或模拟模式，能帮助检查逻辑表达式与中间值，但它们不等同于真实GPU并行执行。竞争、资源限制、目标指令、内存层级与性能可能完全不同。通过模拟是有价值的早期证据，不能作为设备路径已验收的替代。

本章自写NumPy模拟器更明确地只验证索引与数学关系，不试图模仿Triton编译器全部语义。遇到类型提升、目标支持或线程分布问题，必须回到实际编译与运行环境。知道模拟器不覆盖什么，才能正确使用它缩小错误范围。

### 9. 一份逐步实验记录的示例格式

记录第一项是目标函数与合同，例如连续float32向量加法，空输入返回空，输出不与输入别名。第二项是输入集合，例如零、一、二百五十五、二百五十六、二百五十七与一千零三个元素。第三项是配置，例如BLOCK二百五十六、num_warps四。第四项是证据：CPU模拟通过、设备编译未执行、GPU结果未执行。

到服务器后，只更新实际完成的行，补上设备与版本、最大误差、执行命令和退出状态。不要把原来的“未执行”直接删除而不保留证据来源。一个能追踪验证阶段的记录，比一张没有环境信息的成功截图更有长期价值。


### 10. 从一条错误结果反推最小复现

假设向量加法在长度二百五十七时只有最后一个值错误，而长度二百五十六正常。先检查尾块是否启动，再检查offset恰好二百五十六时mask是否为真，最后检查输出是否写回。这个症状强烈指向覆盖与边界，通常不需要先研究浮点舍入或底层指令。

假设二维算子每行第一个值正确，其余列按固定间隔错位，优先检查列stride与广播轴。若整行整体偏移，检查行stride与program到行号的映射。若只有转置视图失败，检查wrapper是否仍套连续公式。把错误按空间模式观察，往往比只看最大误差数字更有诊断力。

假设所有结果都按同一个比例偏大，检查归约除数、缩放参数与padding数量；假设仅大幅度输入出现NaN，检查类型转换位置、指数稳定化与中间溢出；假设相同输入偶尔不同，检查重复写者、未初始化输出或异步依赖。每种假设都应通过一个专门构造的小输入验证，避免一口气修改多个变量。

### 11. 小项目如何形成可维护的工程边界

把CPU参考放在不导入GPU库的模块，设备实现放在显式可选模块，wrapper集中定义支持范围，测试按CPU与GPU分组。这样普通学习与持续集成不需要初始化GPU，设备验证又能清楚报告缺失依赖或真实失败。跳过原因应具体，例如未安装Triton或没有可用设备，而不是把所有异常吞掉后显示通过。

每次新增一个参数，补充一个解释它作用的例子和一个拒绝非法值的检查。每次改变布局或精度，更新测试矩阵与接口说明。每次性能优化，保留之前正确版本或可追溯提交。工程规范的目的不是增加文书，而是让你几周后仍能判断某个速度数字对应哪份代码、哪些输入和哪些已验证合同。

## 第十五课：小项目——可解释的Triton入门算子集

第一阶段完成CPU program模拟器，支持一维向量、二维tile、stride与mask，所有输出附写者计数。第二阶段实现向量加法与行softmax的wrapper，明确shape、dtype、device和空输入合同。第三阶段在GPU环境编译运行，逐项比较边界测试。

第四阶段选择两个BLOCK和两个num_warps配置，确认结果一致，再观察编译资源与性能差异。每次实验只改变一个因素，记录冷启动与稳态区别。第五阶段阅读固定来源官方教程，把每一行与本章的地址、广播、归约、特化或启动职责对应。

最终你应该能解释一条Triton表达式代表什么shape、哪些地址、哪些有效槽与什么数值类型；也能承认源码不能直接告诉你的线程分布和实际性能，需要进一步查看编译产物与测量。掌握这条边界后，再进入K09的六类完整算子实现。

完成时再做一次独立口述：不看代码，说明一个尾部槽从索引生成、合法性判断、替代值、算术处理到最终不写回的全过程。若其中某一步只能回答“编译器会处理”，就回到相应小课补齐自己负责的合同；编译器负责实现合法语义，程序员负责先把语义写对。
