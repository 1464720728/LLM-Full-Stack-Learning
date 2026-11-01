# M13 数值与量化：从一个整数码到完整模型误差实验

[本卷导航](README.md) · 前置：[梯度与数值](02-gradients-training.md)、[稠密解码器](05-dense-decoder.md) · 下一章：[训练与多模态](14-training-modalities.md)

量化不能只理解成“把模型从十六位改成四位”。你需要知道哪些数被近似、整数码代表什么、尺度在哪个轴上共享、如何打包、计算时怎样还原，以及误差为什么可能改变模型行为。本章从一个标量开始，逐步实现编码、解码、分组、四位打包、矩阵误差分析和完整 TinyDecoder 的逐前缀对照。

所有代码使用 CPU NumPy，不下载权重、不调用模型 API。Python 块按本章顺序在同一会话运行；代码只验证定义与行为，不把随机初始化模型的数值误差写成真实语言能力结论。既有参考函数位于 [core.symmetric_quantize](../../labs/models/core.py)，入口为：

```bash
python -m labs.models.run_lab quantize
python -m unittest tests.test_models_core -v
```

## 第一课：位数、数值范围与表示间距

### 1. 一个存储单元能表达多少种状态

一位可以区分两种状态，四位最多区分十六种状态。但这十六种状态究竟代表哪些数，需要编码规则决定。可以代表零到十五的整数，也可以代表带正负号的数，或者代表一张非均匀查找表中的十六个值。只知道“四位”不能推导其数值含义。

本章先采用对称整数编码：四位数值规则只使用负七到七，共十五个码值，留出一个位模式不用。这样正负范围对称，零有准确表示。常见有符号四位整数的完整范围可以包含负八，但本章没有使用这个额外负端点；阅读其他实现时要核对范围，不能把它们混为同一合同。

浮点数采用符号、指数和有效数字组织数值，间距随数值量级变化；整数均匀量化则用一个尺度把等间隔整数映射回实数。INT4、某种 FP4 和带组尺度的低精度格式并不因为都叫四位就相同。本章的整数实验不能声称复现所有低位浮点编码，相关硬件与位格式在算子卷进一步学习。

### 2. 为什么需要 scale

假设整数码 q=7，若 scale s=0.1，它表示零点七；若 s=10，它表示七十。码和尺度必须一起解释。反量化公式是 `x_hat=s*q`，其中 x_hat 表示由有限码恢复的近似值，通常不等于原始 x。只有码而丢失尺度，就像保存一个数字却忘了它的单位。

如果选择可用码范围负 L 到 L，给定 s，编码规则为先计算 x/s，再舍入到整数，最后裁剪到允许范围。反量化再乘 s。舍入决定落在哪个网格点，裁剪处理超出范围的数；它们造成的误差不同，不能合起来只叫“精度损失”而不分析来源。

### 3. 误差界成立的条件

当 x 位于可表示范围内、采用最近舍入且没有其他算术误差时，相邻格点距离 s，最近格点距 x 不超过 s/2。若 x 超出 L*s 的范围，裁剪到端点，误差可以远大于 s/2。因此写测试时要先判断是否发生饱和，再选择适用的误差界。

例如 s=0.1、L=7，最大可恢复值零点七。输入三会被裁剪为七，再恢复成零点七，误差二点三，显然超过零点零五。这不是量化公式自相矛盾，而是你把无裁剪条件下的界用于范围之外。这个区别在使用校准范围处理新输入时非常重要。

## 第二课：完整手算一次对称四位量化

### 1. 从最大绝对值确定尺度

给定 `[-1,-0.3,0,0.8]`，最大绝对值为一。四位对称范围 L=7，取 s=1/7。分别除尺度得到负七、负二点一、零、五点六；最近舍入得到 `[-7,-2,0,6]`。所有码都在范围内，所以本次不需要额外裁剪。

反量化结果为 `[-1,-2/7,0,6/7]`，即负一、约负零点二八五七、零、约零点八五七一。误差 x_hat-x 分别是零、约零点零一四三、零、约零点零五七一，绝对值均不超过 s/2，约零点零七一四。这里把误差符号保留下来，是为了知道近似值变大还是变小，而不仅看一个最大绝对值。

```python
import numpy as np
from labs.models.core import symmetric_quantize, TinyDecoder, DecoderConfig, softmax, cross_entropy

values = np.array([-1.0,-0.3,0.0,0.8])
q, scale, restored = symmetric_quantize(values, bits=4)
np.testing.assert_array_equal(q, [-7,-2,0,6])
np.testing.assert_allclose(scale, [1/7])
np.testing.assert_allclose(restored, [-1,-2/7,0,6/7])
assert np.all(np.abs(restored-values) <= scale/2 + 1e-12)
```

### 2. 一半位置究竟向哪边舍入

NumPy 的 rint 使用最近偶数规则处理精确半整数。例如一点五变二，二点五也变二；负一点五变负二，负二点五也变负二。不能默认所有语言或量化 Kernel 都采用“点五远离零”的规则。两个实现若在中点不同，码就可能不同，之后的数值对照也会变化。

```python
np.testing.assert_array_equal(np.rint([0.5,1.5,2.5,-0.5,-1.5,-2.5]), [0,2,2,0,-2,-2])
midpoints = np.array([-2.0,-1.0,0.0,1.0,2.0])
mid_q, mid_s, mid_restored = symmetric_quantize(midpoints, 4)
np.testing.assert_array_equal(mid_q, [-7,-4,0,4,7])
np.testing.assert_allclose(mid_restored, [-2,-8/7,0,8/7,2])
```

真实二进制浮点有时不能精确表示你写出的十进制中点，因此构造边界测试时要确认实际商。不要只根据打印保留两位小数就认定内部值恰好是二点五。可以选择能被二进制精确表示的比例，或明确比较接近中点两侧的输入，检查实现的定义。

### 3. 全零组为何把尺度设为一

一组全零数最大绝对值为零，若直接取 s=0，再计算 x/s 会除零。参考函数约定全零组使用 s=1，码全部零，恢复仍全部零。这个一只是有效编码的约定，不表示原组含有值一，也不影响恢复正确性。

```python
zero_q, zero_s, zero_y = symmetric_quantize(np.zeros((2,3)), 4, axis=1)
np.testing.assert_array_equal(zero_q, np.zeros((2,3),dtype=np.int8))
np.testing.assert_array_equal(zero_s, np.ones((2,1)))
np.testing.assert_array_equal(zero_y, np.zeros((2,3)))
```

你应该验证反量化值与元数据格式，而不是无理由要求所有实现的零组 scale 必须相同。不同实现可以选其他正尺度，只要码、还原与下游计算一致。当然，若文件格式明确规定零组编码，读写双方就必须遵循该格式，不能各自选择。

## 第三课：逐行读懂参考量化函数

函数先检查 bits 在二到八之间，然后把输入转换为浮点数组。limit 通过 `2**(bits-1)-1` 得到对称正端点。bits=8 时 limit 为一百二十七，bits=4 时为七。当前 q 使用 int8 容器保存，因此这个函数没有把四位码真正压成半字节。

`np.max(np.abs(x), axis=axis, keepdims=True)` 找到每组最大绝对值。keepdims 保留被归约轴的长度一，使 scale 后续能按预期广播。若把 keepdims 去掉，某些形状仍能运行却沿错误轴广播，尤其非方阵更容易揭露问题。分组定义由 axis 决定，下一课会逐个算它。

scale 由 maximum/limit 生成，零组用一替代。`np.rint(x/scale)` 把每个数映射到最近格点，clip 限制码范围，astype(int8) 才变成整数容器。最后 `q.astype(float)*scale` 返回反量化值，便于直接比较。三个返回量分别承担编码、解释尺度、数值近似的职责，不应随便只保留其中一个。

本函数是小型内部参考，不是完整不可信数据解析器。它没有替你定义空张量、任意非有限输入、极小尺度下溢或所有 axis 类型的业务合同。以下练习使用非空有限、合理量级浮点输入；若将量化器做成公开库，需要明确拒绝或支持这些边界，并增加相应测试。

浮点 cast 与量化也不能混淆。把 x 直接转 int8 会截断数值并按转换规则处理范围，它没有为当前张量选择 scale，也无法按照本章公式恢复。把 float64 转 float16 则是浮点格式舍入，不是对称整数网格。区分这些操作，才能正确解读模型加载参数究竟改变了什么。

## 第四课：逐张量、逐通道与逐组尺度

### 1. 一个离群值怎样改变所有小值

对 `[0.01,0.02,0.03,10]` 共用四位 scale，最大值十使 s=10/7，远大于前面三个小值。它们除尺度后都非常接近零，舍入后成为零。大值能够表示，小值的相对信息却丢失。这不是码数变少的抽象口号，而是具体网格间距覆盖不了两个相差很大的量级。

若前三个小值单独成组，最大值零点零三，对应尺度约零点零零四二九，小值就能获得非零码。代价是保存更多尺度，以及让计算实现知道每个码属于哪组。更细粒度不自动意味端到端更快，因为元数据加载、解码和 Kernel 支持也有成本。

```python
outlier = np.array([0.01,0.02,0.03,10.0])
whole_q, whole_s, whole_y = symmetric_quantize(outlier, 4)
small_q, small_s, small_y = symmetric_quantize(outlier[:3], 4)
np.testing.assert_array_equal(whole_q[:3], [0,0,0])
assert np.all(small_q != 0)
assert np.mean((small_y-outlier[:3])**2) < np.mean((whole_y[:3]-outlier[:3])**2)
```

### 2. axis 数字必须结合权重布局解释

本卷线性层采用 `Y=X@W`，W shape 为 `[input_features,output_features]`。若每个输出通道独立尺度，应沿输入轴求最大，即 axis=0，得到 `[1,output_features]`。axis=1 则每个输入行共享尺度，得到 `[input_features,1]`，代表另一种分组。

有些框架把线性权重保存为 `[output_features,input_features]` 并在计算时转置，因此同样“每输出通道”可能使用 axis=1。只写 per-channel 不足以描述可复现实验，必须同时记录权重布局和归约轴。两个代码都用 axis=1，却可能量化了完全不同的语义维度。

```python
weight = np.array([[0.1,10.0,1.0],[0.2,20.0,2.0]])
row_q, row_s, row_y = symmetric_quantize(weight, 4, axis=1)
col_q, col_s, col_y = symmetric_quantize(weight, 4, axis=0)
assert row_s.shape == (2,1)
assert col_s.shape == (1,3)
np.testing.assert_allclose(col_s, [[0.2/7,20/7,2/7]])
```

### 3. 实现完整的逐行分组量化

下面把二维矩阵每行沿最后一轴切成固定大小的组。末组不足时按实际长度处理，不把 padding 的数值混进统计。每组保存一项 scale，输出码与原矩阵同形状，另返回恢复矩阵。这个布局适合学习元数据归属，不等同于任何特定模型文件格式。

```python
def group_quantize(matrix, bits=4, group_size=4):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("nonempty finite matrix required")
    if type(group_size) is not int or group_size < 1:
        raise ValueError("positive integer group size required")
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer in 2..8")
    rows, columns = matrix.shape
    groups = (columns + group_size - 1) // group_size
    codes = np.empty(matrix.shape, dtype=np.int8)
    scales = np.empty((rows,groups), dtype=np.float64)
    restored = np.empty_like(matrix)
    for row in range(rows):
        for group in range(groups):
            start = group * group_size
            stop = min(start + group_size, columns)
            q,s,y = symmetric_quantize(matrix[row,start:stop], bits)
            codes[row,start:stop] = q
            scales[row,group] = s.item()
            restored[row,start:stop] = y
    return codes,scales,restored

group_input = np.array([[0.01,0.02,0.03,10.0,0.5], [0.0,0.0,0.0,0.0,0.0]])
group_q, group_s, group_y = group_quantize(group_input, 4, 3)
assert group_s.shape == (2,2)
np.testing.assert_array_equal(group_y[1], np.zeros(5))
assert np.all(group_q[0,:3] != 0)
```

注意 group_size 三时，第一组三个小值共用一个尺度，第二组十与零点五共用另一个尺度。改变分组边界会改变误差，不只是改变元数据数量。若为了对齐把两个逻辑通道混在一组，也要明确这种布局选择，而不是只报告一个抽象组大小。

## 第五课：校准范围与裁剪误差

### 1. 为什么尺度不一定永远由当前最大值决定

权重在一次部署中相对固定，可以预先统计；激活随输入变化，可能需要离线校准范围或运行时动态尺度。若每次都按当前最大值选尺度，当前范围通常能被覆盖，但要计算统计并传递元数据。若固定使用校准得到的尺度，执行协议可能更稳定，却要面对新输入超出已观察范围的问题。

校准集应代表实际使用中可能出现的输入分布。它不是最终评测集，也不是越大就一定越好；关键是覆盖任务、长度和数值行为。用同一批样本选择尺度并报告效果，会高估对未见输入的表现。即使模型固定，输入分布变化也可能改变合适的范围。

### 2. 裁剪是在两类误差之间取舍

把最大绝对值 A 作为范围，尺度为 A/L。A 变大可容纳更多离群值，但格点变稀；A 变小让常见小值更精细，却裁剪更多大值。不能只统计范围内舍入误差而忽略饱和项，否则会错误偏好极小范围。

下面固定范围而不让函数重新按当前最大值选尺度，完整观察这个取舍。clip_rate 统计输入超出范围的比例，mse 统计全部位置恢复误差；两者回答不同问题。有限合理量级是本示例前提，不处理浮点范围极限。

```python
def quantize_with_bound(x, bound, bits=4):
    x = np.asarray(x,dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("nonempty finite input required")
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("integer bits in 2..8 required")
    if not np.isscalar(bound) or not np.isfinite(bound) or bound <= 0:
        raise ValueError("positive finite bound required")
    limit = 2**(bits-1)-1
    scale = float(bound)/limit
    q = np.clip(np.rint(x/scale), -limit, limit).astype(np.int8)
    y = q.astype(np.float64)*scale
    return q,scale,y

calibration = np.concatenate([np.linspace(-0.2,0.2,101), [3.0]])
calibration_rows = []
for bound in [0.2,0.5,1.0,3.0]:
    cq,cs,cy = quantize_with_bound(calibration,bound)
    calibration_rows.append({"bound":bound,"clip_rate":float(np.mean(np.abs(calibration)>bound)),
                             "mse":float(np.mean((cy-calibration)**2))})
assert calibration_rows[-1]["clip_rate"] == 0.0
assert calibration_rows[0]["clip_rate"] > 0.0
```

这里没有根据结果宣布某个 bound 对所有任务最佳。即使某个范围让权重 MSE 最小，下一层输出或最终任务也可能更在意另一种误差方向。选择指标时，要知道它近似的是哪一项实际目标，不能因为一个数字容易计算就让它代表全部质量。

### 3. 静态与动态描述的是何时决定参数

静态量化通常在部署前确定相关尺度或范围；动态方案会在运行时根据当前数据计算某些参数。两者都可能与不同位宽、分组和 Kernel 组合。不要把“动态”自动解释为精度更高或速度更慢，具体成本取决于统计、融合和硬件实现。

本章可以用同一个输入分别固定旧尺度与重新估计尺度，观察裁剪和格点变化，但没有实现完整生产动态量化 Kernel。把新尺度作为另一个数组传给 NumPy，只证明数值规则可表达，不证明设备上可以无额外代价执行。

## 第六课：加入零点的仿射量化

### 1. 非对称范围为什么可能更合适

如果数据大多落在零到三，使用负七到七的对称码会把约一半格点留给不存在的负值。另一种方法使用零到十五的无符号码，并设置 scale 与 zero point。反量化为 `x_hat=s*(q-z)`，其中 z 是表示真实零的整数码。编码相应为 `clip(round(x/s)+z,qmin,qmax)`。

以范围零到三为例，s=3/15=0.2，z=0，输入一得到码五，输入二得到码十。这组数据可以精确落在格点上。若范围同时含负正值，z 会移动到内部。零点不只是数学平移：整数矩阵乘法需要处理减零点的修正项，存储和计算接口都要知道它。

### 2. 端点、零与舍入不能同时随意保证

常见构造先选包含零的范围 a 到 b，令 s=(b-a)/(qmax-qmin)，再把 qmin-a/s 舍入得到整数 z。因为 z 必须是整数，实际恢复端点可能与原 a/b 略有偏差；不能声称任意端点都绝对精确且零也精确。完整格式还要规定裁剪、舍入和参数类型。

下面实现只用于小规模有限数据，主动把范围扩到包含零，常量全零时使用合法尺度。它展示仿射编码的完整闭环，不假定与某个框架的序列化格式逐字节兼容。

```python
def affine_quantize(x, bits=4):
    x = np.asarray(x,dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("nonempty finite input required")
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("integer bits in 2..8 required")
    qmin,qmax = 0,2**bits-1
    lower,upper = min(0.0,float(x.min())),max(0.0,float(x.max()))
    scale = (upper-lower)/(qmax-qmin) if upper>lower else 1.0
    zero = int(np.clip(np.rint(qmin-lower/scale),qmin,qmax))
    codes = np.clip(np.rint(x/scale)+zero,qmin,qmax).astype(np.uint8)
    restored = scale*(codes.astype(np.float64)-zero)
    return codes,scale,zero,restored

affine_q,affine_s,affine_z,affine_y = affine_quantize([0.0,1.0,2.0,3.0])
np.testing.assert_array_equal(affine_q,[0,5,10,15])
np.testing.assert_allclose(affine_y,[0,1,2,3])
assert affine_z == 0
```

### 3. 整数乘法中的修正项

若 x=sx*(qx-zx)，w=sw*(qw-zw)，乘积就是 sx*sw 乘 `(qx*qw - zx*qw - zw*qx + zx*zw)`。矩阵乘法沿输入维求和时，各修正项也要正确归约。对称零点为零可以简化其中一些项，但不代表所有非对称方案都不值得使用。

这解释了为什么“多一个 zero point 参数”可能影响 Kernel 设计。一个只支持对称权重的解码路径，不能收到带零点权重后直接忽略 z；输出 shape 也许仍正确，数值却系统偏移。文件格式、模型加载器和计算后端必须共同遵守编码合同。

## 第七课：真正把两个四位码放进一个字节

### 1. 数值位宽与容器位宽

symmetric_quantize 返回 int8 数组。bits=4 时，每个元素仍占一个字节，只是可用值范围更小；它没有自动把存储量减半。要获得实际紧凑表示，需要打包。打包只改变码的存储形式，理想情况下不再引入新的数值误差，因为整数码应精确往返。

本章约定将有符号码 q 映射成 nibble=q+8，范围一到十五，零位模式不用于有效码。一个字节的低四位保存第一个码，高四位保存第二个码。若元素数为奇数，最后一个高四位填表示零码的八，并单独保存原始有效长度。没有长度就无法区分真实末尾零与 padding。

例如 q 为负七和七，映射到一和十五，打包值为 `1 | (15<<4)=241`。解包时低四位与十五按位与，高四位右移四位，再各减八恢复。注意按位操作针对编码，不是把实数直接做位移；scale 仍要另外保存。

```python
def pack_symmetric_int4(codes):
    codes = np.asarray(codes)
    if codes.ndim != 1 or codes.dtype.kind not in "iu":
        raise ValueError("a one-dimensional integer code array is required")
    if np.any(codes < -7) or np.any(codes > 7):
        raise ValueError("this format uses codes -7..7")
    count = len(codes)
    nibbles = (codes.astype(np.int16)+8).astype(np.uint8)
    if count % 2:
        nibbles = np.concatenate([nibbles,np.array([8],dtype=np.uint8)])
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed,count


def unpack_symmetric_int4(packed,count):
    packed = np.asarray(packed)
    if packed.ndim != 1 or packed.dtype != np.uint8:
        raise ValueError("one-dimensional uint8 packed storage required")
    if type(count) is not int or count < 0 or len(packed) != (count+1)//2:
        raise ValueError("invalid logical length")
    nibbles = np.empty(2*len(packed),dtype=np.uint8)
    nibbles[0::2] = packed & 15
    nibbles[1::2] = packed >> 4
    nibbles = nibbles[:count]
    if np.any(nibbles == 0):
        raise ValueError("reserved nibble in valid data")
    return (nibbles.astype(np.int16)-8).astype(np.int8)

example_codes = np.array([-7,7,0,-2,6],dtype=np.int8)
packed,count = pack_symmetric_int4(example_codes)
assert packed[0] == 241
assert packed.nbytes == 3 and example_codes.nbytes == 5
np.testing.assert_array_equal(unpack_symmetric_int4(packed,count),example_codes)
```

### 2. 解包必须恢复原来分组关系

把码恢复成一个一维数组还不够。你还需要原 shape、分组方向、group size、scale 的排列方式，以及是否有零点。若把每行 group 的尺度错接到另一行，即使每个 nibble 精确解包，反量化值仍然错误。文件头和元数据不是附带装饰，而是数值解释的一部分。

对于刚才 group_input，可以把 group_q 拉平成一维打包，解包后按原 shape 恢复，再根据行与组取 scale。下面给出完整解码，证明编码压缩没有改变已确定的反量化值。这里记录的 group_size 必须与编码时相同。

```python
group_packed, group_count = pack_symmetric_int4(group_q.ravel())
recovered_codes = unpack_symmetric_int4(group_packed,group_count).reshape(group_q.shape)
recovered_values = np.empty_like(group_input)
for row in range(group_input.shape[0]):
    for column in range(group_input.shape[1]):
        recovered_values[row,column] = recovered_codes[row,column]*group_s[row,column//3]
np.testing.assert_array_equal(recovered_codes,group_q)
np.testing.assert_allclose(recovered_values,group_y)
```

### 3. 完整存储量怎样计算

N 个四位码理论有效载荷约 N/2 字节，奇数时向上取整；若每 G 个码配一个 float32 尺度，尺度开销约四乘 N/G 字节，还可能有零点、对齐、索引与未量化模块。不能拿整个模型参数数乘零点五，就宣称这是最终显存或文件大小。

本例 group_s 是 float64，因此每项八字节；生产格式可能选择其他元数据精度，但那会引入新的表示误差或兼容要求。若为了计算压缩率假定 float32，而实际文件仍保存 float64，报告就是两套不同对象。统计时应优先直接测实际容器字节，再解释理想公式与真实差异。

把码解包到完整 float64 权重后再做矩阵乘法，运行时仍会持有大浮点矩阵。它可以证明量化数值影响，却不体现压缩推理的全部内存收益。真实低位 Kernel 往往要按 tile 解码或用相应低精度计算路径，相关执行机制由 K11 接续。

## 第八课：误差怎样通过线性层与离散选择

### 1. 权重误差不是输出误差

设量化恢复权重为 W_hat=W+DeltaW，输入暂时保持 X。输出差值严格等于 `X@DeltaW`。因此同样的权重误差对不同输入影响不同：某个输入方向会强调误差，另一个方向可能使误差抵消。仅报告权重 MSE，不能保证所有实际输入的输出都同样接近。

若输入也近似为 X+DeltaX，则输出差分包含 `X@DeltaW + DeltaX@W + DeltaX@DeltaW`。最后一项是两边误差相乘，不应在精确恒等式里凭空删除；若做一阶近似忽略它，必须说明近似条件。这样才能把“误差小”从口号变成可检查的推导。

```python
x_original = np.array([[1.0,2.0],[-1.0,0.5]])
w_original = np.array([[0.3,-0.7],[1.1,0.2]])
_,_,w_hat = symmetric_quantize(w_original,4,axis=0)
_,_,x_hat = symmetric_quantize(x_original,4,axis=1)
dx,dw = x_hat-x_original,w_hat-w_original
np.testing.assert_allclose(x_original@w_hat-x_original@w_original,x_original@dw,atol=1e-15)
np.testing.assert_allclose(x_hat@w_hat-x_original@w_original,
                           x_original@dw+dx@w_original+dx@dw,atol=1e-15)
```

### 2. 激活统计提供了什么额外信息

对一批校准输入 X，输出误差平方和为 `||X@DeltaW||_F^2`。展开后会出现 `X.T@X`，它描述输入方向与相关性如何加权权重误差。即使两个候选量化权重拥有相同普通 MSE，若误差落在经常被放大的输入方向，输出误差仍可能更大。

这解释了为什么某些量化方法参考激活或二阶信息，而不是只逐元素找最近码。它们在尝试更贴近层输出或模型目标的代价函数。当前推导只是动机与一个可计算目标，不等于已经实现某篇论文的全部优化、分块更新和数值稳定措施。

### 3. 很小连续误差也可能改变离散选择

若两个 logits 分别为一和一点零零零一，第二项略大；给第一项加零点零零零二后，第一项变大，argmax 改变。MoE top-k 路由也有类似边界：分数微变可能选中另一个专家，之后执行的函数不再相同。

```python
close_logits = np.array([1.0,1.0001])
perturbed = close_logits + np.array([0.0002,0.0])
assert np.argmax(close_logits) == 1
assert np.argmax(perturbed) == 0
assert np.max(np.abs(perturbed-close_logits)) < 0.001
```

这不自动意味着整体任务质量严重下降，也不能用 top-1 一致率唯一评价所有误差。需要同时观察连续分布、离散分支和实际任务。自由生成一旦某步 token 不同，后续输入也不同，两个模型的输出差异就包含了上下文分叉；逐固定前缀比较能隔离最初数值变化，下一课会实现这条路径。

## 第九课：把量化放进完整 TinyDecoder，逐层比较影响

### 1. 先固定基线，再只改变权重表示

创建一个 TinyDecoder，并深复制为候选模型。不能简单赋值 `candidate=reference`，否则两变量指向同一个对象，替换候选权重也会改变基线。两边必须保留相同结构与初始参数，只在明确选择的矩阵上做量化再反量化。

本卷权重采用 X@W 方向，因此 q/k/v/o、gate/up/down 和输出矩阵按 axis=0 设置每输出通道尺度；embedding 则按每个 token 行使用 axis=1。这个选择不是唯一正确方案，但必须把它与布局一同记录。当前 RMSNorm 没有可学习 gain，所以没有额外 norm 权重需要量化；真实模型可能有不同参数。

以下函数可分别选择 attention、FFN、embedding、output 四组矩阵。候选执行仍采用恢复后的浮点矩阵，属于数值影响实验，没有压缩执行或低位 GEMM 加速。每个矩阵的码和 scale 可用于存储分析，但这里不把它们直接接入硬件 Kernel。

```python
import copy

def quantized_decoder_copy(reference, bits=4, groups=("attention","ffn","embedding","output")):
    allowed = {"attention","ffn","embedding","output"}
    if not set(groups) <= allowed:
        raise ValueError("unknown parameter group")
    candidate = copy.deepcopy(reference)
    records = []
    def convert(name,matrix,axis):
        codes,scales,restored = symmetric_quantize(matrix,bits,axis=axis)
        records.append({"name":name,"shape":list(matrix.shape),"axis":axis,
                        "code_bytes":codes.nbytes,"scale_bytes":scales.nbytes,
                        "max_abs_error":float(np.max(np.abs(restored-matrix))),
                        "mse":float(np.mean((restored-matrix)**2))})
        return restored
    if "embedding" in groups:
        candidate.embedding = convert("embedding",reference.embedding,1)
    if "output" in groups:
        candidate.output = convert("output",reference.output,0)
    for index,(source,target) in enumerate(zip(reference.blocks,candidate.blocks)):
        for key in ["q","k","v","o"]:
            if "attention" in groups:
                target[key] = convert(f"block{index}.{key}",source[key],0)
        for key in ["gate","up","down"]:
            if "ffn" in groups:
                target[key] = convert(f"block{index}.{key}",source[key],0)
    return candidate,records

reference_model = TinyDecoder(DecoderConfig(seed=7))
original_embedding = reference_model.embedding.copy()
quantized_model,weight_records = quantized_decoder_copy(reference_model,4)
np.testing.assert_array_equal(reference_model.embedding,original_embedding)
assert quantized_model is not reference_model
assert len(weight_records) == 16
```

### 2. 固定前缀对照怎样避免生成分叉

构造十条固定 token 序列，每条把前若干 token 作为输入，后移一位作为目标。两模型收到完全相同的前缀，因此每位置 logits 差异来自参数近似及其传播，而不是一边前一步生成了不同 token。这个比较常称 teacher-forced 对照，强调输入轨迹固定。

我们记录 logits 最大绝对误差、均方误差、top-1 一致率、参考分布到候选分布的 KL，以及两边对同一目标的交叉熵。随机权重和随机 token 的 CE 只用于验证数值评估链，不能称作语言困惑度基准或模型能力。真正评测需要合适语料、tokenizer、训练权重与协议。

KL 使用稳定 log-softmax 表示，避免先计算很小概率再取 log 造成额外问题。reference KL candidate 与反方向通常不同；报告必须说明方向。有限小模型中 KL 应接近非负，舍入可能出现极小负值，应结合数值量级判断，而不是把任何末位负号解释成新理论。

```python
def log_probabilities(logits):
    shifted = logits-logits.max(axis=-1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))


def compare_fixed_sequences(reference,candidate,sequences):
    reference_logits,candidate_logits,targets = [],[],[]
    for sequence in sequences:
        if len(sequence) < 2:
            raise ValueError("at least two tokens required")
        reference_logits.append(reference.forward(sequence[:-1]))
        candidate_logits.append(candidate.forward(sequence[:-1]))
        targets.extend(sequence[1:])
    a,b = np.concatenate(reference_logits),np.concatenate(candidate_logits)
    targets = np.asarray(targets,dtype=np.int64)
    log_a,log_b = log_probabilities(a),log_probabilities(b)
    return {"positions":len(targets),"max_abs_logits":float(np.max(np.abs(a-b))),
            "mse_logits":float(np.mean((a-b)**2)),
            "top1_agreement":float(np.mean(np.argmax(a,axis=-1)==np.argmax(b,axis=-1))),
            "kl_reference_to_candidate":float(np.mean(np.sum(np.exp(log_a)*(log_a-log_b),axis=-1))),
            "reference_ce":cross_entropy(a,targets)[0],
            "candidate_ce":cross_entropy(b,targets)[0]}

sequence_rng = np.random.default_rng(123)
fixed_sequences = [sequence_rng.integers(0,reference_model.config.vocab,size=7).tolist() for _ in range(10)]
model_comparison = compare_fixed_sequences(reference_model,quantized_model,fixed_sequences)
assert model_comparison["positions"] == 60
assert all(np.isfinite(value) for value in model_comparison.values())
assert 0 <= model_comparison["top1_agreement"] <= 1
assert model_comparison["kl_reference_to_candidate"] >= -1e-12
```

### 3. 做参数组消融，而不是立刻量化一切

先只量化 attention，再只量化 FFN，再加入 embedding 与输出层，比较同一套固定输入。不同组可能对中间状态与 logits 有不同影响。单组影响与全部组影响不必简单相加，因为层间有归一化、门控、残差和非线性，误差会相互作用。

```python
ablation = {}
for group in ["attention","ffn","embedding","output"]:
    changed,records = quantized_decoder_copy(reference_model,4,(group,))
    ablation[group] = compare_fixed_sequences(reference_model,changed,fixed_sequences)
assert set(ablation) == {"attention","ffn","embedding","output"}
```

这个实验只说明当前随机模型和输入下哪些参数组更敏感。要迁移到真实模型，必须重新采集输入与评测；不能把这里某组误差较大写成所有模型都必须保留该组高精度。消融的价值在于建立诊断方法，让你知道怎样寻找问题，而不是制造一个无条件经验规则。

### 4. 再比较自由生成，观察误差反馈

固定同一 prompt，分别做有限步贪心生成。一旦某步 token 不同，后续上下文也不同，之后 logits 差异包含参数近似与输入分叉两个因素。因此应记录首次分叉位置，并回到该步的共同前缀比较 margin 和概率，而不是只看最后两条序列差多少个字符。

```python
def greedy_tokens(model,prompt,steps=6):
    tokens = list(prompt)
    if not tokens or type(steps) is not int or steps < 0:
        raise ValueError("nonempty prompt and nonnegative integer steps required")
    generated = []
    for _ in range(steps):
        token = int(np.argmax(model.forward(tokens)[-1]))
        generated.append(token)
        tokens.append(token)
    return generated

free_reference = greedy_tokens(reference_model,[1,2,3])
free_quantized = greedy_tokens(quantized_model,[1,2,3])
first_difference = next((i for i,(a,b) in enumerate(zip(free_reference,free_quantized)) if a != b),None)
assert len(free_reference) == len(free_quantized) == 6
```

本题不要求一定发生分叉，也不要求所有位宽的自由生成结果都相同。若两边恰好相同，继续看 logits 与 margin；若不同，先定位第一处差异。把“没分叉”写成“无损”，或把“一次分叉”写成“模型不可用”，都超出了当前证据。

## 第十课：理解经典量化方法各自在优化哪一件事

### 1. GPTQ：把层输出误差作为重要线索

[GPTQ 原论文](https://arxiv.org/abs/2210.17323) 使用近似二阶信息进行训练后权重量化。阅读时可从第八课的 `X.T@X` 与层输出重建目标进入：权重误差并非每个方向等价。真正算法还包含量化顺序、误差补偿、分块与数值处理，本章逐元素最近舍入并不是 GPTQ 的复刻。

阅读任务是定位论文目标、输入激活如何参与、一个权重量化后怎样影响剩余权重，再选极小矩阵按作者定义复现。先证明你理解其优化问题，再阅读大规模工程加速。论文中的速度与精度结果属于特定实验，不能直接写到本章 NumPy 的报告里。

### 2. AWQ：依据激活识别应保护的通道

[AWQ 原论文](https://arxiv.org/abs/2306.00978) 研究利用激活信息识别重要权重通道，并采用等价尺度变换改善低位权重表示。对学习者，关键问题是“哪些方向更影响实际输出”，而不是把少量权重数值最大的元素机械保留高精度。

通道重缩放在量化前可以保持原始线性函数等价，但量化以后误差分布会改变。你应分别验证变换前后的浮点等价和量化后的误差变化。这两个实验不能合并：等价变换正确，不保证某个尺度选择在所有输入上都改善量化。

### 3. SmoothQuant：在权重与激活之间迁移范围难度

[SmoothQuant 原论文](https://arxiv.org/abs/2211.10438) 通过等价变换将激活离群值带来的量化困难部分转移到权重，面向权重与激活低精度计算。最小代数是选正对角尺度 D，令 `X'=X D^-1`、`W'=D W`，则浮点下 X'W'=XW。

下面独立例子只证明这个恒等式，不能称作完成 SmoothQuant。真实方法需要选择尺度、采集统计、处理模型结构并接到相应 Kernel。把数学原语和完整方法分开，能帮助你知道下一步源码需要补齐什么。

```python
channel_scale = np.array([2.0,0.5])
x_scaled = x_original/channel_scale[None,:]
w_scaled = channel_scale[:,None]*w_original
np.testing.assert_allclose(x_scaled@w_scaled,x_original@w_original,atol=1e-15)
```

### 4. QLoRA：量化基座与可训练适配参数分工

[QLoRA 原论文](https://arxiv.org/abs/2305.14314) 将冻结的低位基座与可训练低秩适配器结合，并讨论 NF4、量化常数压缩和内存管理。冻结基座不等于整个网络不参与反向计算；梯度仍需经过相关计算传向适配参数。NF4 也不是本章均匀对称 INT4 的别名。

学习时把三个对象分开：权重的存储格式、前向/反向的计算精度、哪些参数真正更新。只看到“4bit finetuning”不能推断所有张量和梯度都以四位保存。LoRA 的矩阵参数化与训练示例在 M14，本章先建立读这些方法所需的数值与存储语言。

上述四篇原始资料于 2026-11-01 核查。它们帮助建立方法脉络，不代表截至该日所有量化研究的穷尽清单。扩展到新的格式或方法时，仍按范围、尺度、分组、打包、计算与评测六个问题逐项检查，避免只记一个新的缩写。

## 第十一课：三类真实排错思路

### 故障一：bits 从八改四，数组字节却没减少

先看 q.dtype 与 q.nbytes。当前参考把两种码都放在 int8 中，每元素同样一字节，所以容器大小不变完全符合实现。若你比较的是 restored 浮点矩阵，它仍按浮点 dtype 分配，更不会自动缩小。应把数值规则、打包格式和执行存储分别检查。

参考修复不是把 nbytes 报告除以二，而是实际执行第七课打包，再统计 packed、scale 和其他元数据的字节。若运行时仍完整解包为浮点权重，应把这部分也纳入峰值内存报告。只有对象真正改变后，测量才能支持压缩结论。

### 故障二：量化矩阵看起来正常，输出却明显偏移

先确认权重在前向中采用 X@W 还是 X@W.T，再核对 scale 轴、广播形状、zero point 与打包顺序。若把按输出通道生成的尺度错误地沿输入行使用，码范围依然合法，恢复数组却不是原先定义的近似。只检查 min/max 不足以发现这种错误。

选择非方阵和不同量级的列，打印每组最大值、scale、一个手算元素与对应恢复值。若只测试所有列数值相同的方阵，轴错误很容易被对称性掩盖。修复后同时比较元素恢复与 X@W 输出，确保格式与计算两层都一致。

### 故障三：校准时良好，上线输入出现饱和

检查新输入超出校准范围的比例，以及超限位置属于哪些层、通道或输入类型。不要立即把整个模型换回高精度，也不要只看平均 MSE。某些少量离群值可能对任务重要，也可能只是容易被平均指标掩盖的统计尾部，需要具体输入分析。

参考排查包括固定同一输入比较旧尺度与重新估计尺度、检查校准/评测数据是否混用、记录长度与任务类别，并尝试有依据的分组或保留部分高精度路径。任何修正都应在独立评测样本上验证，不能用原来挑尺度的数据反复证明自己成功。

## 第十二课：十二道练习与完整参考答案

### 练习一：重新手算四位编码

对 `[-2,-1,0,1,2]` 使用本章对称范围。参考尺度为二除以七，除尺度后为负七、负三点五、零、三点五、七。最近偶数舍入得到负七、负四、零、四、七，还原为负二、负八除以七、零、八除以七、二。请写出中间商与误差界，不要只抄最终整数码。

本题还要求解释为什么负八没有被使用：这是当前对称编码合同，而不是四位位模式数量不够。若改用负八到七的范围，需要重新定义尺度和端点处理，不能只把 clip 的下界换掉却继续声称其他性质完全相同。

### 练习二：构造误差超过半步长的输入

固定 bound 为零点七、bits=4，输入三。尺度零点一，码饱和为七，恢复零点七，误差二点三。参考代码如下。误差超过半步长是范围外裁剪的预期行为；若测试仍强制半步长上界，错的是测试前提。

```python
clip_q,clip_s,clip_y = quantize_with_bound(np.array([3.0]),0.7,4)
np.testing.assert_array_equal(clip_q,[7])
np.testing.assert_allclose(clip_y,[0.7])
assert abs(clip_y[0]-3.0) > clip_s/2
```

### 练习三：说明每个 scale 控制哪些元素

使用第四课二乘三矩阵，分别画出 axis=0 和 axis=1 的组。参考答案是 axis=0 每列一组，共三项尺度；axis=1 每行一组，共两项尺度。还要说明在 X@W 布局中，输出通道对应列。若把 W 转置并改成 X@W.T，输出通道就对应新矩阵的行，axis 数字也随之改变。

这道题的验收物是一张坐标到 scale 的映射，而不是只打印 shape。shape `[1,3]` 告诉你广播方向，映射表说明业务含义。真实文件格式往往依赖这种对应关系，不能只靠数组尺寸推测。

### 练习四：完整验证逐组解码

使用 group_size 三，五列矩阵最后一组只有两项。参考编码、打包、解包与反量化已经在第四、七课给出；本题要求改变最后一列值并再次完成精确码往返。验收包括恢复 shape 不变、零行保持零、每元素使用 `column//group_size` 对应尺度。

如果错误地用 `column % group_size` 选择 scale，五列会索引到不存在的组或错接参数。这个反例说明组号与组内偏移不同，就像分页缓存中的逻辑页号和页内槽不同。跨章节复用相同索引思想，有助于真正理解布局。

### 练习五：遍历全部合法四位码组合

对所有负七到七的码两两组合，打包再解包，应精确相同。再测试零长度、一长度和奇数长度。参考代码如下；该测试检验位存储，不采用浮点容差，因为整数码往返应精确。

```python
all_codes = np.arange(-7,8,dtype=np.int8)
pairs = np.array([(int(a),int(b)) for a in all_codes for b in all_codes],dtype=np.int8).ravel()
pair_bytes,pair_count = pack_symmetric_int4(pairs)
np.testing.assert_array_equal(unpack_symmetric_int4(pair_bytes,pair_count),pairs)
for length in [0,1,7,15]:
    original = all_codes[:length]
    storage,valid = pack_symmetric_int4(original)
    np.testing.assert_array_equal(unpack_symmetric_int4(storage,valid),original)
    assert storage.nbytes == (length+1)//2
```

### 练习六：拒绝保留码与错误长度

本章 nibble 零不是合法有效码。给 unpack 一个零字节、有效长度一，应拒绝；给一个字节却声明长度三，也应拒绝，因为容量不够。完整参考如下。不能让损坏元数据静默产生另一组看似合理的权重。

```python
for raw,valid in [(np.array([0],dtype=np.uint8),1),
                  (np.array([0x88],dtype=np.uint8),3)]:
    try:
        unpack_symmetric_int4(raw,valid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid encoding accepted")
```

### 练习七：证明较小权重误差未必给较小输出误差

设输入 X=[100,1]。候选一的权重误差向量为 `[0.01,0]`，候选二为 `[0,0.1]`。第一项权重误差范数更小，却产生输出误差一；第二项产生零点一。参考推导直接用 X@DeltaW。这个例子解释为什么要参考激活，而不是宣布某种单一误差指标永远正确。

```python
sensitive_x = np.array([[100.0,1.0]])
error_a = np.array([[0.01],[0.0]])
error_b = np.array([[0.0],[0.1]])
assert np.linalg.norm(error_a) < np.linalg.norm(error_b)
assert np.linalg.norm(sensitive_x@error_a) > np.linalg.norm(sensitive_x@error_b)
```

### 练习八：比较四位与八位，但不预设模型指标排序

分别生成四位和八位候选，对同一 fixed_sequences 运行 compare_fixed_sequences。记录权重与 logits 误差、KL、top-1 一致率和 CE。参考任务不是写一个“八位所有指标必须更好”的武断断言，而是解释更密网格通常提供更精细表示，但完整网络、目标与离散决策会带来不同观察。

```python
eight_bit_model,eight_bit_records = quantized_decoder_copy(reference_model,8)
eight_bit_comparison = compare_fixed_sequences(reference_model,eight_bit_model,fixed_sequences)
assert eight_bit_comparison["positions"] == model_comparison["positions"]
assert all(np.isfinite(value) for value in eight_bit_comparison.values())
```

要做公平比较，两边保持相同分组、量化参数组与输入；若同时换 group size 和评测序列，就无法把差异单独归因于位宽。把控制条件写进报告，是比一个胜负结论更重要的训练。

### 练习九：比较相对误差与绝对误差

参考值接近零时，一个很小绝对误差可能产生很大相对误差。若 reference=0、candidate=0.001，绝对误差零点零零一；使用极小 eps 的相对误差会很大。反过来，参考值一千的绝对误差一，相对误差只有千分之一。报告应同时给尺度背景，不能只挑看起来小的指标。

对于模型分布，还要考虑 softmax 的平移不变性：所有 logits 同加一个常数，概率不变，但 logits 最大绝对差非零。因此 logits 误差与 KL 互补，不能直接把其中一个解释成最终预测差异的全部。

```python
base_logits = np.array([[1.0,2.0,3.0]])
shifted_logits = base_logits + 7.0
np.testing.assert_allclose(softmax(base_logits),softmax(shifted_logits))
assert np.max(np.abs(base_logits-shifted_logits)) == 7.0
```

### 练习十：证明等价缩放先于量化是另一项实验

第十课的浮点变换 X'=X/D、W'=D*W 保持输出相同。分别量化原 W 与缩放后 W'，比较两种路径的输出误差。参考做法是先验证浮点恒等式，再记录量化结果；不要求任意 D 都改善误差。若你随便选尺度却报告它总能提升，就是把一个代数性质误当成优化保证。

```python
_,_,plain_weight = symmetric_quantize(w_original,4,axis=0)
_,_,scaled_weight = symmetric_quantize(w_scaled,4,axis=0)
plain_output = x_original@plain_weight
scaled_output = x_scaled@scaled_weight
plain_error = float(np.mean((plain_output-x_original@w_original)**2))
scaled_error = float(np.mean((scaled_output-x_original@w_original)**2))
assert np.isfinite(plain_error) and np.isfinite(scaled_error)
```

### 练习十一：复现三个故障并留下最小反例

分别构造未打包 int8 字节不变、scale 轴错接、固定校准范围饱和。参考证据应包括实际 dtype/nbytes、一个非方阵元素的手算恢复、一个超界输入的 clip 结果。修复后保留这些反例作为回归，不要只把运行日志中的 warning 隐藏掉。

每个反例应尽量只涉及一种失败。若同时改 dtype、scale、zero point 和模型结构，结果变好也无法说明是哪项修复有效。小而可解释的反例能长期保留在测试中，是从教材代码走向可维护工程的重要桥梁。

### 练习十二：提交一份完整模型量化实验

参考交付包括基线配置与 seed、量化参数组、位宽、分组轴、码与 scale dtype、是否实际打包、固定前缀指标、自由生成首次分叉、存储统计和未执行硬件路径。完整实现由第九课函数、第七课打包器和本节实验组成，不能只提交“误差很小”的一句判断。

报告明确写随机模型仅验证方法。若改用真实开放权重，应另外锁定模型 revision、tokenizer、校准集与评测集，记录框架和后端选择，并用相同任务协议比较。不要把原论文的性能数字复制成自己的实验结果，也不要把成功加载模型等同于所有低精度路径已正确执行。

## 第十三课：从本章走到真实低精度模型与算子

### 1. 读配置时建立六项对照

首先确认格式：是均匀整数、某种浮点格式还是查表编码；其次确认分组：逐张量、逐通道还是固定 group，以及轴的实际含义；第三确认参数：scale、zero point、可能的额外缩放与它们的 dtype；第四确认 packing：位顺序、排列、padding 和元数据；第五确认计算：激活、累加器和输出精度；第六确认范围：哪些层和缓存真正量化。

同一个“四位模型”标签可能只描述大部分线性权重，embedding、router、norm、输出层或 KV cache 另有精度。权重体积、峰值显存和运行速度不是同一个数字。加载时临时还原、工作区、批次与上下文长度都会影响显存；Kernel 是否能利用压缩格式也会影响实际速度。

### 2. 个人 GPU 与 H100 分别适合回答什么问题

先在 CPU 完成编码、packing、误差与小模型对照，再到个人 GPU 检查实际支持的低精度后端和小规模吞吐。需要特定指令、较大模型或多卡时，再使用分配的 H100 窗口。不要仅根据显卡名字推断它支持所有相似名称格式；应检查所用库、编译目标、驱动与具体 Kernel。

第一次真实实验可以只选一个线性层、固定输入和有限形状，比较高精度参考与实际低精度输出，观察加载、解码、计算与同步开销。确认算子合同后再扩大到整模型。若整模型立即出现差异，小算子基线能帮助你定位是格式加载、层选择、数值容差还是生成分叉。

本章没有运行 GPU 或真实开放权重量化任务，提供的是完整 CPU 数值与存储工作台。K11 连接量化矩阵计算与硬件路径，S12 连接数值确定性，S13 连接容量估算。按照这些接口继续，你可以逐步把本章的码、尺度和误差报告对应到真实框架，而不是重新从参数名字猜起。

### 3. KV cache 量化为何需要单独验收

权重量化改变生成 K/V 的函数，缓存量化则直接近似已经生成的历史向量，两者不能只用同一个“模型四位”标签概括。key 误差会改变 query 与历史位置的分数，进而改变 softmax 权重；value 误差会改变被加权读取的内容。相同最大误差落在 K 或 V 上，传播路径不同，应分别做消融。

对固定 query，分数误差包含 `q@DeltaK.T/sqrt(D)`，因此 query 的方向和大小仍然重要。若只量化 V 且保持注意力权重 P 不变，输出误差是 `P@DeltaV`；同时量化 K 时，P 也改变，不能继续使用这个简化式代表全部误差。先固定一项再改变另一项，可以把两个来源拆开观察。

还要记录量化发生频率：新行写入时量化一次，与每次扩展缓存都把旧行反量化再重新量化，是两种不同过程。后者可能重复引入舍入与尺度变化，也可能移动旧行所属分组。共享前缀的 scale 和有效长度必须随缓存一起保留；只共享码数组而遗漏元数据，无法恢复相同历史。S19 的数值页池与本章编码器可以作为这类实验的独立基线。

## 结业检查

不看答案解释一个整数码如何恢复成数，手算最近舍入与裁剪的差别；从 W 的前向方向判断 scale 轴；精确完成奇数长度四位打包往返；用 X@DeltaW 解释输入敏感性；对完整模型给出固定前缀与自由生成两种报告。能完成这些动作，才具备阅读和修改真实量化实现的基础。

下一步的提高来自更准确的目标、校准、数据格式和设备实现，而不是反复降低 bits 看是否还能输出文本。保留本章完整参考与失败样例，让每次新优化都有可比较的数值起点。你以后看到新的量化名字，也能先拆成已有机制，再识别它真正新增了哪一项方法。
