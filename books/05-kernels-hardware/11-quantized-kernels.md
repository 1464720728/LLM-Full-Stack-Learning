# K11 低精度算子：从位编码、量化尺度到打包、整数GEMM与分组反量化

“模型用了四位”这句话远远不够描述一个可运行算子。四位可能表示有符号整数、无符号整数、某种浮点编码或查表代码；数值还可能乘按行、按组或多层尺度；物理字节可能经过重排，供特定矩阵指令消费。本章把这些层次逐个拆开，让你能自己编码、解码、量化、打包并实现可检查的量化矩阵乘法。

先修K01的字节与地址、K05的浮点误差、K06的GEMM。主要实验全部可用NumPy在CPU完成。我们实现明确命名的教学INT8、INT4与E2M1缩放参考，不冒称完整MXFP4或NVFP4标准实现，也不把CPU解包模拟当成某种GPU原生低位算术。

## 第一课：位模式、数值类型与量化方案

### 1. 同一串位可以有不同含义

四位代码二进制一一一一，按无符号整数解释是十五，按四位二补码有符号整数解释是负一，按本章E2M1浮点表解释是负六。位模式本身不携带完整语义，解码规则决定数值。存储文件必须同时记录格式或由接口明确规定。

量化方案又在编码之外增加尺度与可能的零点。代码负一乘尺度零点二，代表近似值负零点二；同一代码乘尺度三，代表负三。因此一个量化张量通常由代码数组、尺度数组、零点或其他元数据、逻辑shape与布局共同组成。

### 2. 存储精度与计算精度分开

权重以四位保存，Kernel可能先解包并转成半精度，再用半精度矩阵指令与单精度累加；也可能使用目标硬件支持的低位矩阵路径。两者都可以被笼统叫“四位权重推理”，但指令、流量、误差与性能不同。

输入类型、乘法方式、累加类型和输出类型应分别记录。把int8点积结果最后转int32，并不等于在int32中累加；把float16平方后转float32，也不能恢复先前溢出。转换位置是数值算法的一部分。

### 3. 压缩比要把元数据算进去

每个权重四位只描述代码本体。若每三十二个权重还存一个十六位尺度，平均增加半位，总计至少四点五位每元素，尚未计入padding、零点、全局尺度或布局对齐。小矩阵中元数据与padding占比可能更高。

因此“从十六位变四位就是严格四倍节省”只在忽略其他开销时成立。真实模型显存还包括激活、缓存、临时缓冲与框架开销。学习算子时先算代码与元数据，再讨论完整模型，而不是把单项压缩率直接当作总显存收益。

### 4. 浮点数为何不是均匀网格

普通二进制浮点把数分成符号、指数和尾数。对正规数，数值可以写成符号因子乘二的指数次方，再乘一加尾数字段对应的小数。指数扩大可表示范围，尾数决定同一数量级内的间隔。因此越大的数，相邻可表示值的绝对间距通常越大。

以标准binary32为例，符号一位、指数八位、显式尾数二十三位。指数采用偏置表示，普通正规数的真实指数等于字段值减一百二十七。隐含的最高有效一让正规数获得额外一位有效精度，但零与次正规数采用不同规则，不能对所有代码都强加这个一。

binary16使用五位指数与十位显式尾数，bfloat16使用八位指数与七位显式尾数。二者都是十六位存储，却在范围与精度之间做不同取舍。bfloat16与binary32具有相近的指数范围结构，但尾数更短；binary16在一附近网格更细，却更容易因较小指数范围出现溢出。

### 5. 用Python拆开float32位字段

~~~python
import struct
import numpy as np

def float32_fields(value):
    bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 255
    fraction = bits & ((1 << 23) - 1)
    return bits, sign, exponent, fraction

def decode_binary32_fields(sign, exponent, fraction):
    if exponent == 255:
        return (-1.0 if sign else 1.0) * np.inf if fraction == 0 else np.nan
    if exponent == 0:
        magnitude = (fraction / (1 << 23)) * 2.0**(-126)
    else:
        magnitude = (1.0 + fraction / (1 << 23)) * 2.0**(exponent - 127)
    return -magnitude if sign else magnitude

bits, sign, exponent, fraction = float32_fields(1.5)
assert bits == 0x3FC00000
assert (sign, exponent, fraction) == (0, 127, 1 << 22)
assert decode_binary32_fields(sign, exponent, fraction) == 1.5
~~~

struct.pack把Python浮点按float32编码成四个字节，unpack按无符号三十二位整数读取同一位模式。这里明确使用小端格式只是为了稳定解释字节序，数学字段提取随后通过位移与掩码完成。不要把数值转换成整数与重新解释位模式混淆：int(一点五)得到一，完全不是上述位编码。

一点五等于一点一的二进制形式，即一加二分之一，真实指数零。字段指数因此是一百二十七，尾数最高位为一，其余零。这个手算与代码中的十六进制结果对应，说明浮点编码可以逐字段理解，不是不可读的黑盒。

负二点五等于负的一点二五乘二，符号一，真实指数一，字段指数一百二十八，尾数为四分之一。用同一函数拆解并恢复，可以练习符号、指数与尾数各自作用。之后阅读低精度格式时，仍按这三层提问，再核对其特殊编码规则。

### 6. 正规数、次正规数与特殊值

指数全零时，标准IEEE二进制格式使用次正规规则，隐含最高位不再是一。这样最小正规数附近可以逐步靠近零，而不是突然出现巨大空档。代价是有效精度逐渐下降。不同硬件与运算模式对次正规值的处理可能不同，实际数值实验应记录目标行为。

指数全一通常用于无穷与NaN，具体由尾数字段区分。某些低位浮点格式为了扩大有限范围，可能重新分配这些代码，因此不能把binary32特殊值规则直接套到所有FP8或FP4格式。格式名字中的指数尾数位数只是起点，特殊值表同样必要。

binary16最大有限值为六万五千五百零四，最小正规正数为二的负十四次方，最小正次正规为二的负二十四次方。知道这些量级可以解释为什么半精度平方、指数或长累加容易出问题，但实际模型还需考虑缩放、混合精度和输入分布。

### 7. 一附近的间隔与舍入

在一到二之间，binary32相邻正规值间隔为二的负二十三次方，binary16为二的负十次方，bfloat16为二的负七次方。更短尾数意味着更粗的相对分辨率。把一加一个小于半间隔的增量存入相应格式，可能仍舍入回一。

这个现象与INT8的固定尺度网格类似，但浮点间隔随指数变化。低精度评估不能只用一个固定绝对误差阈值覆盖所有数量级；同样的相对舍入在大值上产生更大的绝对误差。归约与矩阵乘法又会进一步累积这些误差。

### 8. 一个明确的bfloat16舍入模拟

~~~python
def round_float32_to_bfloat16_reference(value):
    bits, sign, exponent, fraction = float32_fields(value)
    if exponent == 255:
        raise ValueError('finite float32 input required')
    upper_low_bit = (bits >> 16) & 1
    rounded_bits = (bits + 0x7FFF + upper_low_bit) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', rounded_bits))[0]

assert round_float32_to_bfloat16_reference(1.00390625) == 1.0
assert round_float32_to_bfloat16_reference(1.01171875) == 1.015625
~~~

这段教学代码对有限float32输入演示保留高十六位并按最近偶数舍入。加的偏置与保留位最低位共同决定中点时向偶数结果靠拢。它返回float32容器中的bfloat16可表示数值，不是一个真正的bfloat16存储数组，也不声称覆盖所有异常值转换合同。

第一个输入恰好位于一与一点零零七八一二五的中点，选择尾数偶数的一。第二个输入位于下一对相邻值中点，选择一点零一五六二五。中点测试比普通随机数更能检查舍入规则，格式转换应包含这样的边界，而不仅是平均误差。

极大有限输入舍入后可能变成无穷，这属于目标格式范围与舍入规则问题。本函数只演示有限输入的位操作，不自动保证输出有限。生产转换若要求有限饱和，需要按目标格式合同实现，不能把截断、最近舍入与饱和混为一谈。


## 第二课：从二补码理解INT4与INT8

### 1. 有符号整数的范围

n位二补码通常表示从负二的n减一次方到正二的n减一次方减一。四位是负八到七，八位是负一百二十八到一百二十七。正负范围不完全对称，因为零占用了一个非负编码。

本章对称INT8量化主动使用负一百二十七到一百二十七，舍弃最负的一个代码，使正负缩放更对称。这个选择是量化合同，不是int8存储类型只能表示这些值。读取他人格式时必须确认实际使用范围。

### 2. 四位符号扩展

四位代码零到七表示非负数，八到十五表示负数，解码时减十六。代码八得到负八，代码十五得到负一。若直接把四位代码八放进int8字节并期待自动变负八，会得到正八，因为符号位位置已经不同。

符号扩展的本质是把窄有符号值转换到更宽有符号表示，保持数值不变。解包时先提取四位，再按规则修正，不能仅靠数组dtype名称自动完成。位运算与数值转换是两个明确步骤。

### 3. 手算一个字节

把第一个INT4值负一放低四位，第二个值二放高四位。负一编码十五，二编码二，所以字节为二乘十六加十五，等于四十七，十六进制为二F。解包低四位十五减十六得到负一，高四位二直接得到二。

这里约定低半字节先对应逻辑第一个元素。另一种格式可能相反，因此字节完全相同也可能对应不同元素顺序。不要把半字节顺序简单等同于整个机器的大小端概念；这是格式自己定义的打包顺序，需要明确记录。

## 第三课：对称INT8量化的完整实现

### 1. 公式中的每个量

给定一行x，取绝对最大值a，尺度s=a除以一百二十七；量化代码q为x除s后按最近规则舍入，再裁剪到负一百二十七至一百二十七；反量化近似值为s乘q。尺度越小，网格越细，但可表示范围也越小。

全零行不能使用零尺度，因为除零无定义。教学选择s为一，q全零，反量化仍全零。这个尺度不是唯一选择，只要有限正值并保持零行正确即可；固定为一使元数据与测试简单。

~~~python
import numpy as np

def quantize_rows_int8(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('matrix with nonempty feature axis required')
    if not np.isfinite(x).all():
        raise ValueError('finite input required')
    peak = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.where(peak > 0, peak / 127.0, 1.0)
    codes = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return codes, scale

def dequantize_rows_int8(codes, scale):
    codes = np.asarray(codes, dtype=np.int8)
    scale = np.asarray(scale, dtype=np.float64)
    if codes.ndim != 2 or scale.shape != (codes.shape[0], 1):
        raise ValueError('code/scale shape mismatch')
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('positive finite scales required')
    return codes.astype(np.float64) * scale

codes, scale = quantize_rows_int8([[-1, 0, 0.5, 1], [0, 0, 0, 0]])
assert np.array_equal(codes, [[-127, 0, 64, 127], [0, 0, 0, 0]])
assert np.array_equal(scale[1], [1])
reconstructed = dequantize_rows_int8(codes, scale)
assert np.max(np.abs(reconstructed[0] - [-1, 0, 0.5, 1])) <= scale[0, 0] / 2
~~~

### 2. 舍入规则不能省略

NumPy的rint采用其定义的最近舍入规则，中点通常按偶数处理。零点五除以一除一百二十七等于六十三点五，舍入到六十四。其他库或硬件转换可能提供不同舍入模式，格式互通时必须核对。

裁剪发生在转换为int8之前。若先astype到int8，超范围值可能已经按类型转换规则产生错误结果，再clip无法恢复。正确顺序是宽类型计算、舍入、范围裁剪、最后编码到窄类型。

### 3. 单元素误差界

在未裁剪且最近舍入的条件下，反量化误差绝对值不超过半个尺度，因为每个实数选择最近网格点。若输入超出所设范围并被裁剪，误差还包含超范围距离，不再受同一半尺度界约束。

这个界只针对单个输入元素，不直接保证矩阵乘法输出误差或模型质量。长点积会累积多项误差，相消结果还可能出现很大相对误差。需要继续分析下游运算，而不是从局部网格误差跳到“模型无损”。

## 第四课：非对称量化与零点

### 1. 为什么需要零点

对主要位于非负范围的数据，使用对称正负网格可能浪费一半代码。非对称形式用x约等于s乘(q减z)，其中q可取零到二百五十五，z是表示实数零的整数代码。尺度决定间隔，零点决定网格平移。

设范围负一到二，使用二百五十六个代码，尺度为三除以二百五十五，即一除以八十五，零点为八十五。实数负一对应代码零，零对应八十五，二对应二百五十五，三个值都能精确落在网格上。

### 2. 完整有限范围参考

~~~python
def quantize_affine_u8(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('nonempty finite input required')
    low = min(float(x.min()), 0.0)
    high = max(float(x.max()), 0.0)
    scale = (high - low) / 255.0 if high > low else 1.0
    zero_point = int(np.clip(np.rint(-low / scale), 0, 255))
    codes = np.clip(np.rint(x / scale) + zero_point, 0, 255).astype(np.uint8)
    return codes, scale, zero_point

codes_u8, scale_u8, zero_u8 = quantize_affine_u8([-1, 0, 2])
assert zero_u8 == 85
assert np.array_equal(codes_u8, [0, 85, 255])
assert np.allclose(scale_u8 * (codes_u8.astype(np.int32) - zero_u8), [-1, 0, 2])
~~~

范围主动包含零，保证零点有合理意义。减zero_point前先转宽有符号整数，否则uint8减法可能发生无符号回绕。这个类型问题与量化误差无关，属于编码算术错误，必须先修正。

### 3. 零点如何进入点积

若a约等于sa乘(qa减za)，b约等于sb乘(qb减zb)，点积展开为sa乘sb乘以：qa与qb点积，减zb乘qa求和，减za乘qb求和，再加K乘za乘zb。零点不为零时，不能只做原始整数dot再乘尺度。

某些实现通过预计算权重行和或列和减少校正成本，某些采用对称权重使一部分项消失。优化之前先完整展开公式，才能知道哪些项可以省略。把非对称代码直接送进对称INT8 Kernel，会产生系统性偏移。

## 第五课：INT8矩阵乘法与累加范围

### 1. 按A行、B列量化

A为M乘K，按每行一个尺度sa；B为K乘N，按每列一个尺度sb。对固定输出i、j，sa[i]与sb[j]不随k变化，因此可以从点积求和中提出。先计算整数和，再乘两个尺度，得到量化输入对应的近似矩阵乘法。

~~~python
def int8_gemm_reference(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('compatible matrices required')
    qa, sa = quantize_rows_int8(a)
    qbt, sbt = quantize_rows_int8(b.T)
    accumulator = qa.astype(np.int64) @ qbt.astype(np.int64).T
    result = accumulator.astype(np.float64) * sa * sbt.T
    return result, accumulator, (qa, sa, qbt.T, sbt.T)

a = np.array([[1., 2., -1.], [0.5, -0.25, 1.]])
b = np.array([[1., 2.], [-1., 0.5], [0.25, -2.]])
quantized_result, integer_sum, metadata = int8_gemm_reference(a, b)
qa, sa, qb, sb = metadata
decoded_a = qa.astype(np.float64) * sa
decoded_b = qb.astype(np.float64) * sb
assert np.allclose(quantized_result, decoded_a @ decoded_b)
~~~

这里CPU参考用int64累加，便于避免教学大范围输入的int32溢出；真实设备路径常有明确的int32累加合同，需要验证K与输入范围。仓库原始int8_matmul使用int32并适用于有界小实验，不能把它推广到任意长K而不检查范围。

### 2. 为什么必须先cast再dot

十六项全为一百二十七的点积，每项乘积一万六千一百二十九，总和二十五万八千零六十四。这个结果远超int8范围。若NumPy按窄整数运算产生窄结果，再转int32，已经发生的溢出不会被修复。

即使累加用int32，也不是无限安全。对称范围最大绝对乘积一万六千一百二十九，最坏同号和随K线性增长。可以用int32最大值除该乘积得到保守K上限；实际输入可能更小，但接口若要保证所有允许值安全，就应按最坏情况考虑。

~~~python
maximum_k = np.iinfo(np.int32).max // (127 * 127)
assert maximum_k == 133144
wide_dot = np.full(16, 127, dtype=np.int8).astype(np.int32) @ np.full(16, 127, dtype=np.int8).astype(np.int32)
assert int(wide_dot) == 258064
~~~

### 3. 分离编码误差与计算误差

量化结果与原始a乘b的差异包含编码误差；量化Kernel与decoded_a乘decoded_b的差异主要反映执行与累加误差。先检查后者，确认Kernel正确实现所声明的量化数值，再评估前者是否满足应用需求。

如果两种误差混在一起，可能把打包索引错误误称为量化损失，也可能把合理量化误差当作Kernel错误。测试报告应同时列出参考基于原始值还是反量化值，避免读者看到一个误差数字却不知道它回答什么问题。

## 第六课：沿K分组时，尺度不能随意提出

### 1. 分组公式

将K分成若干组，每组拥有自己的A行尺度与B列尺度。输出是各组整数点积乘该组尺度乘积后再相加。由于尺度随组变化，不能先把所有整数点积合并，再乘一个平均尺度。

用两个组手算：第一组整数和十，尺度乘积零点一，贡献一；第二组整数和二十，尺度乘积零点零一，贡献零点二，总结果一点二。若先合并三十，再乘平均尺度零点零五五，得到一点六五，明显错误。

### 2. 完整分组参考

~~~python
def grouped_int8_gemm(a, b, group_size):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('compatible matrices required')
    if group_size <= 0 or a.shape[1] == 0:
        raise ValueError('positive group and reduction dimension required')
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    traces = []
    for begin in range(0, a.shape[1], group_size):
        end = min(begin + group_size, a.shape[1])
        part, integer, meta = int8_gemm_reference(a[:, begin:end], b[begin:end])
        result += part
        traces.append((begin, end, integer, meta))
    return result, traces

grouped, group_trace = grouped_int8_gemm(a, b, 2)
assert len(group_trace) == 2
assert group_trace[-1][:2] == (2, 3)
~~~

最后一组可能短于group_size，量化范围与整数点积只针对真实元素。若物理打包补齐，padding代码需要表示实数零，并且尺度索引不能越界。非对称量化中实数零对应zero_point，不一定是代码零，这是尾部处理的重要区别。

### 3. Scale布局也需要地址公式

若A尺度shape为M乘组数，地址可为row乘组数加group；B尺度shape为组数乘N，地址为group乘N加col。两者轴顺序不同，与权重代码布局也可能不同。Kernel必须分别计算，不能因为都叫scale就共用同一stride。

为了匹配矩阵指令，真实后端可能重排尺度块。读取模型文件后不能仅reshape代码与scale就假定可直接使用。格式转换需要同时验证代码顺序、尺度顺序与逻辑分组，一处错位会让整组数值系统性偏差。

## 第七课：INT4打包与解包的完整实现

~~~python
def pack_signed_int4(values):
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('integer values required')
    if np.any(values < -8) or np.any(values > 7):
        raise ValueError('INT4 range is [-8, 7]')
    flat = values.reshape(-1).astype(np.int16)
    codes = (flat & 15).astype(np.uint8)
    count = codes.size
    if count % 2:
        codes = np.concatenate([codes, np.zeros(1, dtype=np.uint8)])
    packed = codes[0::2] | (codes[1::2] << 4)
    return packed, count

def unpack_signed_int4(packed, count):
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if count < 0 or count > 2 * packed.size:
        raise ValueError('invalid logical count')
    codes = np.empty(2 * packed.size, dtype=np.int16)
    codes[0::2] = packed & 15
    codes[1::2] = packed >> 4
    codes = codes[:count]
    return np.where(codes >= 8, codes - 16, codes).astype(np.int8)

all_values = np.arange(-8, 8, dtype=np.int8)
packed, count = pack_signed_int4(all_values)
assert np.array_equal(unpack_signed_int4(packed, count), all_values)
packed_odd, count_odd = pack_signed_int4(np.array([-1, 2, -8]))
assert np.array_equal(packed_odd, [47, 8])
assert np.array_equal(unpack_signed_int4(packed_odd, count_odd), [-1, 2, -8])
~~~

### 1. 每个操作对应哪一层

与十五做按位与保留低四位，实现二补码代码提取；左移四位把第二个代码放到高半字节；按位或合并互不重叠的两组位。解包时先提取无符号代码，再根据最高位做符号恢复。astype在这里不是随意装饰，而是控制位运算与数值范围。

count保存逻辑元素数，避免奇数长度补齐的最后半字节被当作真实数据。原shape也需要另外保存，才能把一维解包结果恢复成矩阵。只有packed字节而没有逻辑shape与count，通常不足以恢复完整张量语义。

### 2. 行边界打包与全局连续打包

如果每行长度为奇数，可以选择每行单独补齐到整字节，也可以把所有行扁平后连续打包。两种布局的行起点不同。前者更容易按行寻址但增加padding，后者更紧凑却可能让某行从高半字节开始。

例如两行各三项，逐行打包需要四字节，整体打包只需三字节。Kernel如果按逐行布局读取整体打包数据，第二行起点会错。格式文档必须说明打包沿哪个轴、行stride以字节还是元素计、是否每行对齐。

## 第八课：E2M1浮点编码与整数网格的区别

### 1. 一个符号位、两位指数、一位尾数

E2M1名称描述字段数量，但完整格式还需要指数偏置、特殊值与次正规规则。本章使用有限E2M1值表，正幅值依次为零、零点五、一、一点五、二、三、四、六，最高位表示符号。它不是均匀间隔整数网格，幅度越大间隔通常越大。

代码七对应正六，代码十五对应负六；INT4中的同样代码分别是七与负一。四位存储宽度相同不代表可互换。解码器必须知道格式，不能把所有四位权重都当二补码整数处理。

### 2. 完整值表解码

~~~python
E2M1_MAGNITUDES = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 6.])

def decode_e2m1(codes):
    codes = np.asarray(codes, dtype=np.int64)
    if np.any(codes < 0) or np.any(codes > 15):
        raise ValueError('four-bit codes required')
    sign = np.where((codes & 8) != 0, -1.0, 1.0)
    return E2M1_MAGNITUDES[codes & 7] * sign

decoded = decode_e2m1(np.arange(16))
assert decoded[7] == 6
assert decoded[15] == -6
assert np.signbit(decoded[8])
~~~

负零与正零数值比较相等，但符号位可不同。是否保留负零、如何舍入中点、超范围如何饱和，都属于编码合同。当前解码完整列出十六个代码，不把它扩张成任意FP4格式通用解释。

### 3. 教学最近值编码

~~~python
def encode_e2m1_nearest(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('finite values required')
    distances = np.abs(np.abs(values)[..., None] - E2M1_MAGNITUDES)
    magnitude_code = np.argmin(distances, axis=-1).astype(np.uint8)
    sign_code = np.where(np.signbit(values), 8, 0).astype(np.uint8)
    return magnitude_code | sign_code

assert np.array_equal(decode_e2m1(encode_e2m1_nearest(decoded)), decoded)
assert decode_e2m1(encode_e2m1_nearest([100.0]))[0] == 6.0
~~~

argmin在距离相同情况下选择第一个位置，因此本教学编码采用明确的“较早幅值代码”平局规则，不宣称与所有硬件最近偶数转换逐bit一致。超范围值选择最大幅值，相当于有限饱和。要与具体库互通，必须替换为其精确舍入与特殊值规则并逐代码测试。

## 第九课：分块浮点尺度与格式名称的边界

### 1. 一个教学分块E2M1张量

可以把每组实数除以尺度，使最大绝对值落到六附近，再编码为E2M1，反量化时乘回尺度。这个过程帮助理解低位浮点加共享尺度的思想，但只要尺度类型、组大小或布局与标准不同，就不应称为完整MXFP4或NVFP4实现。

~~~python
def quantize_e2m1_groups(values, group_size=4):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if group_size <= 0 or not np.isfinite(values).all():
        raise ValueError('positive group and finite values required')
    codes = np.empty(values.size, dtype=np.uint8)
    scales = []
    for begin in range(0, values.size, group_size):
        end = min(begin + group_size, values.size)
        peak = float(np.max(np.abs(values[begin:end])))
        scale = peak / 6.0 if peak > 0 else 1.0
        codes[begin:end] = encode_e2m1_nearest(values[begin:end] / scale)
        scales.append(scale)
    return codes, np.asarray(scales), values.size

def dequantize_e2m1_groups(codes, scales, group_size):
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.size != (codes.size + group_size - 1) // group_size:
        raise ValueError('scale count mismatch')
    result = decode_e2m1(codes)
    for group, scale in enumerate(scales):
        begin = group * group_size
        result[begin:begin + group_size] *= scale
    return result

codes4, scales4, _ = quantize_e2m1_groups([0, 1, 2, 6, -3], 4)
restored4 = dequantize_e2m1_groups(codes4, scales4, 4)
assert np.allclose(restored4, [0, 1, 2, 6, -3])
~~~

这个例子尺度用float64，既不是节省元数据的标准格式，也不是硬件直接消费布局。它的名字刻意保留“教学分组E2M1”，让数学实验与格式兼容性分开。后续若实现标准，应先写字段清单，再逐项替换并验证。

### 2. 需要核对的格式字段

至少核对元素代码、尺度编码、分组轴与大小、是否另有全局尺度、零与异常值处理、舍入与饱和、padding、打包顺序以及硬件矩阵接口要求的布局。只知道“低四位加一个scale”远不足以互操作。

不同GPU可以存储相同字节，也可以用普通算术模拟解码，但不一定拥有相同原生矩阵指令。硬件能力、编译器支持、框架Kernel与模型格式匹配是四个独立层次。K16会进一步比较目标设备，不能只根据一个架构名字推断全部低精度能力。

## 第十课：误差、离群值与校准

### 1. 离群值如何降低分辨率

一行大多数值在负一到一之间，却有一个一百，按行最大值量化时尺度由一百决定，小值网格变粗，许多小权重可能映射到相同代码或零。按更小组量化可以限制离群值影响范围，但增加尺度元数据与Kernel复杂度。

裁剪离群值可以减小尺度、改善多数小值的分辨率，却对被裁剪值引入更大误差。最佳选择取决于数据分布与下游敏感性，不能只看最大误差或平均误差一个指标。教学先在合成数据上画出重建值与误差，再连接模型质量评估。

### 2. 点积误差展开

若量化后A为A加ΔA，B为B加ΔB，则乘积误差为AΔB加ΔAB加ΔAΔB。对单个输出取绝对值并用三角不等式，可得到各项绝对值乘积之和的上界。它解释了输入幅度、归约长度与误差相关性为何都影响输出。

相消点积的真实结果可能接近零，即使绝对误差不大，相对误差也会很大。因此量化评估应同时报告绝对误差、相对误差与任务相关指标，不应把相对误差无限放大直接当作模型不可用，也不能只看平均误差忽略关键输出。

### 3. 静态与动态量化的成本

静态尺度从代表性数据估计，运行时可省去部分统计；动态尺度根据当前激活计算，更适应变化但增加归约与量化成本。权重通常可离线预处理并反复使用，激活往往每次调用变化。性能比较必须说明是否计入这些步骤。

如果只计整数GEMM，不计激活量化、权重解包、尺度读取与输出反量化，可能高估端到端收益。对小batch解码，权重带宽与解包成本可能重要；对大batch矩阵计算，矩阵指令吞吐与累加策略可能更突出。具体结论必须测量，不能凭位数单独判断。

### 4. 非对称点积校正的完整数值例子

设qa为十、二十、三十，qb为四、五、六，za为十，zb为四，两个尺度都为一。反量化后的a为零、十、二十，b为零、一、二，正确点积为五十。原始整数点积却为三百二十，说明零点校正不是微小修饰。

展开式中qa求和六十，qb求和十五，K为三。三百二十减四乘六十，减十乘十五，再加三乘十乘四，得到五十。每一项都能对应到乘法展开，删除任何一项都会改变结果。若某个零点恰好为零，对应项才自然消失。

~~~python
qa_affine = np.array([10, 20, 30], dtype=np.int64)
qb_affine = np.array([4, 5, 6], dtype=np.int64)
za, zb = 10, 4
corrected = (qa_affine @ qb_affine
             - zb * qa_affine.sum()
             - za * qb_affine.sum()
             + qa_affine.size * za * zb)
assert corrected == 50
assert corrected == (qa_affine - za) @ (qb_affine - zb)
~~~

如果激活零点按行变化、权重零点按列变化，校正项也要按输出行列组合，不能用一个全局标量替代。若零点按K组变化，则每组都要单独校正。与尺度一样，参数是否随归约索引变化决定能否移到最终epilogue。

### 5. 用误差上界检查推导而不是承诺质量

~~~python
a_original = np.array([[1.0, -2.0, 0.5]])
b_original = np.array([[2.0], [1.0], [-3.0]])
a_approx = a_original + np.array([[0.01, -0.02, 0.03]])
b_approx = b_original + np.array([[0.02], [0.01], [-0.01]])
delta_a = a_approx - a_original
delta_b = b_approx - b_original
bound = (np.abs(a_original) @ np.abs(delta_b)
         + np.abs(delta_a) @ np.abs(b_original)
         + np.abs(delta_a) @ np.abs(delta_b))
actual_error = np.abs(a_approx @ b_approx - a_original @ b_original)
assert np.all(actual_error <= bound + 1e-15)
~~~

这个上界通常偏保守，因为它把不同项的符号相消全部忽略。它适合检查数学关系与理解误差来源，不适合直接预测模型质量。真实误差可能远小于上界，也可能在某些敏感方向对下游产生较大影响，因此还需要任务层评估。

若量化只作用于权重，激活保持原精度，则ΔA为零，误差表达式简化为AΔB。这个简化帮助解释不同激活行对相同权重量化误差的敏感性不同：激活幅度与方向决定误差如何投影到输出。单看权重均方误差不能完整描述每个输入的输出误差。

### 6. Padding代码必须表示数学零

对称整数格式中，代码零通常对应实数零；非对称格式中，代码zero_point才对应零。若尾部padding直接填字节零，却使用非零zero_point反量化，padding会变成负的尺度乘zero_point，污染点积。这个错误即使所有真实元素编码正确，也会在非整除K时出现。

浮点分组格式通常也有零代码，但符号零、尺度与异常值规则仍需核对。padding的职责是让无效计算贡献中性值，而不是随意填一个看起来空的位模式。设计低精度Kernel时，应从解码后的数值检查padding，再看其物理字节编码。

### 7. Scale自身也有舍入误差

如果参考使用float64尺度，而实际格式把尺度存成float16或更低精度，重建值还会受到尺度编码误差。测试应先把尺度按真实格式舍入，再计算参考反量化，才能公平检查Kernel。否则你可能把尺度格式的必然误差误认为设备乘法错误。

尺度通常共享一组元素，其误差会对整组产生相关缩放偏差，不像每个元素独立的舍入噪声。观察输出中整组或整列按固定比例偏差时，除了索引错位，也应检查尺度dtype与转换位置。元数据虽然数量少，数值作用却覆盖许多元素。


## 第十一课：三个故障案例

第一类是先int8点积再cast，输出出现看似随机的窄范围值。用十六项全一百二十七的点积核对二十五万八千零六十四，修复为乘加前转宽类型，并检查更长K的累加上限。

第二类是INT4负值解包成正八到十五。检查是否完成代码大于等于八时减十六的符号恢复。遍历全部十六个值比随机抽几个更直接，编码空间很小，应穷举验收。

第三类是分组尺度错位。让每组数值幅度相差十倍，分别打印组整数和与尺度乘积；若某组结果按固定比例偏差，检查scale的行列与group索引。不能通过整体乘一个校正系数掩盖布局错误。

## 第十二课：十二道练习与完整答案

### 练习一：解释代码十五

答案：无符号四位为十五，二补码INT4为负一，本章E2M1为负六。相同位宽不决定数值含义。

### 练习二：手算打包

题目：INT4负八、七，低位先放第一项。答案代码八与七，字节一百二十，即十六进制七八。解包后恢复负八、七。

### 练习三：奇数长度

答案：三个INT4需要两个字节，最后高半字节为padding，必须保存逻辑count三，不能解码成四个真实元素。

### 练习四：对称尺度

题目：行负二、零、一、二。答案尺度二除以一百二十七，代码负一百二十七、零、六十四、一百二十七，最近中点规则需明确。

### 练习五：零行

答案：选有限正尺度一，代码全零，反量化全零。零尺度会造成除零，不能作为普通路径。

### 练习六：非对称零点

题目：范围负一到二，uint8。答案尺度一除以八十五，零点八十五，负一、零、二对应零、八十五、二百五十五。

### 练习七：累加范围

答案：十六项一百二十七相乘求和为二十五万八千零六十四；必须在足够宽类型中乘加，事后cast无效。

### 练习八：分组缩放

题目：两组整数和十、二十，尺度乘积零点一、零点零一。答案一点二；先合并再乘平均尺度得到一点六五，错误。

### 练习九：元数据开销

答案：每三十二个四位元素加十六位尺度，平均四点五位每元素，未计padding与其他元数据。

### 练习十：E2M1间隔

答案：一与一点五间隔零点五，四与六间隔二，网格不均匀，因此不是INT4乘固定尺度就能表示相同代码表。

### 练习十一：两种参考

答案：与反量化输入乘积比较检查Kernel数值正确；与原始输入乘积比较评估总量化误差。二者必须分开报告。

### 练习十二：完整格式审计

答案列出代码含义、尺度类型与shape、分组轴、零点、舍入、饱和、padding、打包顺序、累加类型和目标硬件路径，并用小矩阵逐字段往返验证。

## 第十三课：从打包字节到矩阵输出的完整数据通路

### 1. 定义一个可审计的INT4权重布局

设权重W逻辑shape为K乘N，按每列、沿K分组量化。为了让示例容易实现，我们采用每列独立存储代码序列，列内低半字节先放较早k，每列末尾若K为奇数则补一个半字节。尺度shape为组数乘N，组号由k整除group_size得到。

这个布局不一定是高性能硬件最喜欢的布局，但它完整且可解释。物理字节地址是列号乘每列字节数，加k整除二；半字节选择由k模二决定。尺度地址是组号乘N加列号。代码与尺度的两个地址公式共同决定一个权重值。

若K为五、N为三，每列需要三个字节，总代码九字节。逻辑权重十五个，每个四位理想是七点五字节，逐列补齐使实际需要九字节。若group_size为三，则每列两个尺度，共六个尺度；尺度类型若为float32，元数据二十四字节，比代码本体还大。小例子能提醒你不要忽略元数据开销。

### 2. 完整打包对象与访问函数

~~~python
from dataclasses import dataclass

@dataclass
class PackedInt4Columns:
    packed: np.ndarray
    scales: np.ndarray
    k_size: int
    n_size: int
    group_size: int

    def value(self, k, col):
        if not 0 <= k < self.k_size or not 0 <= col < self.n_size:
            raise IndexError('logical weight index out of range')
        byte = int(self.packed[col, k // 2])
        code = (byte >> (4 * (k % 2))) & 15
        signed = code - 16 if code >= 8 else code
        return signed * self.scales[k // self.group_size, col]

def pack_weight_int4_columns(weight, group_size):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2 or min(weight.shape) <= 0 or group_size <= 0:
        raise ValueError('nonempty matrix and positive group required')
    if not np.isfinite(weight).all():
        raise ValueError('finite weight required')
    k_size, n_size = weight.shape
    groups = (k_size + group_size - 1) // group_size
    scales = np.empty((groups, n_size))
    codes = np.empty((k_size, n_size), dtype=np.int8)
    for group in range(groups):
        begin = group * group_size
        end = min(begin + group_size, k_size)
        peak = np.max(np.abs(weight[begin:end]), axis=0)
        scale = np.where(peak > 0, peak / 7.0, 1.0)
        scales[group] = scale
        codes[begin:end] = np.clip(
            np.rint(weight[begin:end] / scale), -7, 7).astype(np.int8)
    packed = np.empty((n_size, (k_size + 1) // 2), dtype=np.uint8)
    for col in range(n_size):
        packed[col], _ = pack_signed_int4(codes[:, col])
    return PackedInt4Columns(packed, scales, k_size, n_size, group_size)

weight_case = np.array([[1., -2.], [3., 4.], [-5., 6.],
                        [7., -1.], [2., 0.5]])
packed_weight = pack_weight_int4_columns(weight_case, 3)
decoded_weight = np.array([
    [packed_weight.value(k, col) for col in range(2)]
    for k in range(5)])
assert packed_weight.packed.shape == (2, 3)
assert packed_weight.scales.shape == (2, 2)
~~~

量化代码主动使用负七到七，而打包容器支持负八到七。两者不矛盾：存储格式范围更宽，当前对称量化策略没有使用最负代码。格式标识需要同时描述编码和量化策略，不能只写INT4后让读者猜尺度除数是七还是八。

value方法先检查逻辑索引，再解码字节与半字节，最后乘正确组尺度。它非常适合做慢速oracle，因为每一步都能打印。生产Kernel会批量加载与解包，而不是对每个元素调用Python方法，但应与这个清楚参考在小矩阵上完全对应。

### 3. 直接消费打包权重的CPU矩阵乘法

~~~python
def matmul_packed_weight_reference(activation, weight):
    activation = np.asarray(activation, dtype=np.float64)
    if activation.ndim != 2 or activation.shape[1] != weight.k_size:
        raise ValueError('activation shape mismatch')
    output = np.zeros((activation.shape[0], weight.n_size))
    for row in range(activation.shape[0]):
        for col in range(weight.n_size):
            total = 0.0
            for k in range(weight.k_size):
                total += activation[row, k] * weight.value(k, col)
            output[row, col] = total
    return output

activation_case = np.array([[1., 2., 3., 4., 5.],
                            [-1., 0., 1., 0., -1.]])
packed_output = matmul_packed_weight_reference(activation_case, packed_weight)
assert np.allclose(packed_output, activation_case @ decoded_weight)
~~~

这个实现不先构造完整反量化权重再计算，而是按需解码每个访问，展示“融合反量化与乘法”的数据路径。它在Python中很慢，目的不是性能，而是提供一个能追踪字节、尺度与输出贡献的基线。

真实GPU可以每次加载包含多个代码的字节或字，使用位移掩码批量解包，再转换成乘法输入。若权重被多个输出行复用，解码结果也可能在寄存器或shared中复用。选择取决于batch大小、tile与硬件路径，不能只数压缩后字节就判断速度。

### 4. 手动追踪一个权重元素

取K五、group_size三，访问k四、列一。每列字节数三，列一的第三个字节包含k四的低半字节与padding高半字节。组号为四整除三等于一，使用scales[一,一]。如果误用k整除二作为组号，就会读第三组而越界或错位。

这个元素参与每个输出行的对应点积。对activation第一行，贡献是五乘该反量化权重；第二行贡献是负一乘该权重。解码一次后可以在多行间复用，但尺度仍属于该权重组，不会随activation行改变，除非采用不同的激活量化方案。

### 5. 序列化至少保存哪些字段

保存packed字节之外，还要保存逻辑K与N、group_size、尺度dtype与shape、代码有符号解释、半字节顺序、每列padding规则和量化范围。版本号用于未来改变布局时明确迁移，不能让新解码器默默按旧布局猜测。

读取文件时先验证元数据一致性，例如packed shape应为N乘K向上除以二，scale组数应为K向上除以group_size，尺度必须有限正值。文件字节长度正确不代表逻辑格式正确，元数据检查可以在真正GPU计算前发现许多兼容错误。

## 第十四课：误差实验、性能账本与格式扩展

### 1. 用离群值做可解释实验

构造一行三十一个小值与一个大值，分别采用整行尺度和四元素分组尺度。观察小值中有多少被量化成零、最大绝对误差与均方误差如何变化。分组通常能局部隔离离群值影响，但具体结果取决于离群值落在哪组，不能只给一个平均数字。

再把大值裁剪到较小阈值，比较被裁剪位置误差与其余位置误差。这样可以看见分辨率改善与饱和损失的交换。教材不需要直接推荐某个通用阈值，因为真实模型的敏感性与数据分布不同；学习目标是能设计并解释这类实验。

### 2. 完整误差分解报告

对量化矩阵乘法，先计算原始高精度输出，再计算反量化输入的高精度输出，最后计算目标Kernel输出。第一与第二之差是编码引入的近似；第二与第三之差是执行路径误差；第一与第三之差是总误差。三者不必在绝对值上简单相加，但这种分层能定位来源。

如果执行误差很大，先检查打包、尺度与累加，不要继续调校准数据。若执行误差小而编码误差大，才研究粒度、裁剪、范围或更高位宽。把工程正确性与算法近似分开，可以避免在错误Kernel上做无意义的模型调优。

### 3. 吞吐与端到端延迟的账本

一次低精度线性层可能包含激活统计、激活量化、权重代码读取、尺度读取、解包、矩阵乘法、反量化、bias与输出转换。预打包权重的离线成本可以单独记录，运行时每次发生的步骤必须计入端到端延迟。

若只展示矩阵指令主循环时间，需明确它是局部指标。模型服务关心完整请求或层的延迟，不能把局部加速倍数直接当作整体加速。对于小任务，启动与解包可能占主导；对于大任务，计算与带宽关系又不同。

### 4. 读一个真实后端的顺序

先找加载模型权重时的格式转换，确认原始shape如何变成packed shape；再找scale与zero point的存储布局；然后找Kernel中解包与尺度索引；最后确认矩阵乘法输入与累加类型。四处证据缺一，就不能仅凭模型配置名断言实际实现。

选择一个极小矩阵，手算其中两个权重的字节位置和尺度，再与后端输出比较。若后端要求特殊对齐尺寸，可以用padding构造小逻辑矩阵，但要记录真实有效范围。不要直接从巨大模型的最终文本输出推断格式互通是否正确。

### 5. 新格式进入工程的验收顺序

先穷举单元素代码解码，再验证舍入中点与饱和，再验证打包往返，再验证尺度分组，再验证小矩阵乘法，最后才测目标GPU性能。小位宽代码空间有限，穷举成本很低，应充分利用，不必只依赖随机测试。

格式扩展应新增明确标识与版本，不覆盖旧解码器的含义。保留旧测试与已知字节样本，防止后续布局优化破坏兼容。一个可以可靠读取旧权重并解释新布局的工程，比只追逐最新位宽名称更有长期价值。


### 6. 格式文档与代码证据怎样互相核对

阅读官方转换接口时，分别找到输入类型、目标解释、舍入模式和越界行为；阅读矩阵指令时，再找操作数打包、尺度布局和累加类型。转换函数存在不代表对应矩阵路径在你的设备可用，矩阵路径存在也不代表框架已经为当前模型格式实现适配。

本章E2M1值表与转换边界参考NVIDIA的PTX与CUDA数学接口文档，具体网址见本卷来源表。教学最近值编码明确采用自己的平局规则，因此不宣称逐位复现官方转换。若要建立兼容性，必须在实际工具链上逐个边界值对照，并保存版本与输出代码。

最后检查格式名称是否包含了没有实现的承诺。只完成代码表和共享尺度，就称为教学分块浮点；完成某标准的全部字段、舍入与布局并验证互通后，才使用该标准名称。准确命名能防止后续读者把原理实验误接到真实模型权重上。

## 第十五课：小项目——可审计的低精度矩阵包

第一阶段完成INT8量化与反量化，保存代码、尺度、shape和格式标识；第二阶段完成INT4穷举打包回读与奇数行布局；第三阶段实现按行列和按K分组的量化GEMM，分别报告编码误差与执行误差。

第四阶段加入教学E2M1值表与分组尺度，明确它不是完整标准格式。第五阶段选择一个真实固定版本后端，追踪权重打包、尺度布局和累加类型，用一个小矩阵验证互通。没有目标GPU时完成CPU编解码与静态阅读，设备性能保持明确未验证。

完成标准是你能看着一段字节和元数据恢复逻辑数值，能推导尺度放在归约内还是外，能解释误差来自编码、累加还是布局，并能为新增格式写穷举或边界测试。

每次格式变更都应保留旧字节样本，确保兼容性检查可以重复执行。
