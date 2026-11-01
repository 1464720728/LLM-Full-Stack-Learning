# K09　把 Softmax、RMSNorm、GEMM 和流式注意力搬到 Triton

本章把 K05–K08 的算法连接起来。代码入口是 [triton_ops.py](../../labs/kernels/triton_ops.py)，可选验证是 [gpu_check.py](../../labs/kernels/gpu_check.py)。所有这些实现都是完整可执行教学代码，但本次交付环境没有 GPU，因此只做 Python 语法检查；运行正确性须在你的 GPU 上完成。

## 1. 一行归约的映射

Softmax 让一个 Program 负责一行。`row=program_id(0)`，`col=arange(0,BLOCK)`，地址为 `X+row*N+col`。BLOCK 是不小于 N 的二次幂，尾部填负无穷。先求最大值，再指数，再求和，最后归一化。真正的行长度是 N，执行布局长度是 BLOCK，二者在非二次幂时不同。

RMSNorm 同样一行一个 Program，尾部 x 和 w 填零。平方和除以 N，然后 `rsqrt`，再乘 x 和 w。因为整个行需要同时保留参与计算的值，行过宽会增加资源压力。本例限制 N≤4096，不声称这个限制是硬件或算法的理论上限。更宽行可以分阶段归约或采用其他布局，但要重新设计中间统计量与同步。

归约 sum 实现两阶段：Triton 第一阶段输出每块的 partial sum，第二阶段调用 PyTorch `partial.sum()`。这是一种明确分工，不是假装“整个 sum 已融合成一个 Kernel”。练习可以再把第二阶段改为 Triton，并对大 partial 数继续递归分层。

## 2. GEMM 的三组索引

矩阵乘法 Program 使用二维 Grid，分别负责 C 的 BM×BN Tile。`rows` 与 `cols` 选择输出子块，`kk` 选择 K 归约块。A 的地址广播成 `[BM,BK]`，B 的地址广播成 `[BK,BN]`。`tl.dot` 返回 `[BM,BN]` 的块乘加结果，累加器保留在循环状态中。

三个方向的 mask 都不可遗漏：M 的尾部、N 的尾部、K 的尾部。输入无效值填零，输出无效位置不写。代码对 float32 dot 指定 `input_precision='ieee'`，目的是让教学比较更容易解释；这可能放弃一些更高吞吐的输入精度模式。追求性能时可以实验其他模式，但须把精度契约和容差一起变化，不能只展示更快的数字。

## 3. 一个可读的流式注意力 Kernel

本例 `_attention` 一次处理一个 query，按 BK 个 key 遍历。Q 的 D 维补齐到 BD，V 的 Dv 维补齐到 BV。它用元素乘法与归约计算点积，而不是构建高吞吐的 Q Tile Tensor Core 路径。这样可以逐行对应 K07 的 m/l/u 公式：`maximum`、`denom`、`numerator`。

每块先根据 token<N 和因果条件屏蔽分数，再求新最大值、重缩放旧分子分母、加入新 V。最后写 `numerator/denom`。因为它只接受等长自注意力，第一块包含所有 query 都可见的 key 0，所以更新后最大值有限。若扩展任意 mask 或矩形位置，这一条件可能不成立，必须加入全无效状态处理。

限制 N≤512、D/Dv≤64 是实验保护和简化，不是 FlashAttention 的能力限制。输入没有 dropout，没有分页 KV，没有多头 batch；你应先完整理解这个基础版本，再逐项加入维度，而不是一次复制上千行生产实现。

## 4. 验证矩阵不能只选“漂亮尺寸”

GPU 检查覆盖向量长度 1、31、257、1003，行宽 1、31、257，GEMM 的 19×23×17 与 33×65×31，注意力长度 17、65 和非二次幂 head dim。原因是这些尺寸主动撞击补齐、尾部和块边界。

容差不是越宽越好。先用高精度 CPU oracle 建立误差范围，再考虑输入/累加 dtype 和计算顺序。对于随机 float32 小规模实验，`1e-4` 量级的阈值便于初步核查；真实业务还需要绝对/相对误差分布、极端值、长序列和下游行为。某些输出接近零时相对误差可能巨大，单看相对误差也会误导。

## 5. 从正确到可维护

把算子代码分成三层：纯数学参考、设备实现、测试与基准。参考应简单清楚，避免复制设备实现的相同索引错误；测试应检查不同形状和语义属性；基准才负责速度。不能为了通过测试，直接把 oracle 改成与错误 Kernel 一样的代码。

运行流程：

```bash
python -m unittest tests.test_kernels_reference -v
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.gpu_check --run
```

如果 CPU 通过、GPU 失败，先对最小失败形状逐项核对地址、mask、归约轴、dtype。若 GPU 结果正确但慢，保留正确版本作为基线，在另一个分支做单项优化。不要在调性能时失去可比较的参考。

## 6. 练习与参考答案

1. **Softmax 修改。** 支持列数 4097，需要只改限制吗？答案不一定；BLOCK 变为 8192 可能增加资源压力，先验证编译与占用，再考虑分阶段。语义正确与资源可执行是两道门槛。
2. **sum 完成。** 用同一个 `_partial_sum` 递归处理 partial，直到剩一个值。参考：每一层大小向上取整；空输入返回零；记录每层启动次数，不宣称单 Kernel。
3. **GEMM 变体。** 增加 alpha/beta：`C=alpha AB+beta C0`。参考：入口检查 C0 的形状与别名，epilogue 读取 C0 后写新输出；测试 alpha=0 与 beta=0，不能为 beta=0 无条件读取无效 C0。
4. **Attention 扩展。** 添加 batch/head 外层索引，但暂时保持连续布局。参考：Grid 增加 `(batch*head, query)`，基础偏移为各张量的单 head 大小；每个 head 的 m/l/u 独立。验收与显式逐 head CPU 结果比较。
5. **故障诊断。** Attention 的 V 地址把 DV 写成 D，什么数据会暴露？答案 D≠Dv；默认测试已有 7 与 5、16 与 9 的组合。只用 D=Dv 永远无法抓住这个错。
6. **综合任务。** 为六个算子建立一份支持矩阵：dtype、布局、空输入、最大实验尺寸、梯度、GPU验证状态。参考：本例 add 支持空输入；softmax 不支持空列；所有 Triton 包装尚未注册 autograd；不要用“支持 CUDA”概括所有条件。

掌握门槛：六个算子至少亲手重写三个；给每个实现加一项新能力并用新测试证明；能解释本例注意力与高性能 FlashAttention 的差距在执行组织而不在核心数学。
