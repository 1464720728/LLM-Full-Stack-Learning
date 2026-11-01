# K18　通信算子与算子工程综合项目

先修 GEMM、异步执行、基础卷进程与推理系统卷并行概念。目标：区分集合通信的数学语义与物理算法，完成一个有参考、实现、测试、环境记录和源码报告的算子作品集。

## 1. Collective 定义的是结果关系

AllReduce 对每个 rank 的输入做归约，并把完整结果交给每个 rank。AllGather 把各 rank 的片段按约定顺序拼接，并给每个 rank 完整拼接结果。ReduceScatter 先归约，再把结果分片交给各 rank。All-to-All 则按目标 rank 交换不同片段，常用于某些专家并行数据分发。

rank 是通信组中的逻辑身份，不一定等于主机上的 GPU 物理编号。`CUDA_VISIBLE_DEVICES` 会改变进程看到的设备映射，跨节点还有 global rank 与 local rank。错把 rank 当 GPU 序号，可能把多个进程放在同一设备或错误路由数据。

每个 rank 需要以匹配的次序、计数和类型参与相应操作。一个 rank 走到 AllReduce，另一个走到 AllGather，不能指望库自动识别并修复程序逻辑。挂起通常需要从各 rank 的调用顺序与失败路径分析。

## 2. Ring 的简化成本模型

理想 ring AllReduce 可看成 ReduceScatter 加 AllGather。每个 rank 的原始数据量 S、rank 数 p、每阶段延迟 alpha、链路有效带宽 beta 时，一种常用近似是：

\[
T\approx 2(p-1)\alpha+2\frac{p-1}{p}\frac{S}{\beta}.
\]

第一项随步骤数增长，第二项体现每 rank 传输量。它忽略拓扑、路由、链路共享、协议选择、并发负载与实现细节，所以只用于判断小消息延迟主导、大消息带宽主导等趋势。实际 NCCL 可以选择不同算法，不能把 ring 公式当所有集合通信的执行保证。

```bash
python -m labs.kernels.run --lab 24
python -m labs.kernels.run --lab 25
```

这些是 CPU 数学/成本实验，不产生真实通信。把它们的输出标题写成“H100 AllReduce 性能”就是错误报告。

## 3. 一个短小的真实两卡验证

[nccl_check.py](../../labs/kernels/nccl_check.py) 每个 rank 创建十六个 float32，rank0 全为1，rank1 全为2，AllReduce 后每项应为3。只允许两个 rank，有进程组超时，finally 销毁自己创建的组。

在已获得两卡使用窗口、环境就绪后：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m labs.kernels.nccl_check --run
```

GPU 编号仍要改成实际获分配的设备。本实验不测带宽、不占用大显存、不启动长期任务。通过后再逐步改变消息大小，并按性能方法记录。八卡并非正确性入门的必要条件；涉及真实 TP/EP 拓扑研究时再扩展。

## 4. 项目 A：可证明的注意力算子作品集

项目起点是 `capstone.py`。它对四组 shape、两种因果设置、四种 block 大小，共 32 个案例比较在线注意力与 dense oracle，并产生 JSON 报告。报告包含 Python/NumPy 版本、误差、后端和 GPU 是否验证，不虚构性能。

```bash
python -m labs.kernels.capstone --output build/kernel-labs/correctness.json
```

你的任务是扩展到 batch/head、矩形解码位置偏移和自定义 mask，并为每个新能力新增独立测试。然后在个人 GPU 上运行 Triton 基线，记录支持矩阵和误差。最后只选择一个优化方向，例如 query Tile 或输入精度模式，保留未优化版本并比较。

交付必须有四部分：契约文档、数学参考、设备实现、验证报告。可选性能报告是第五部分。通过标准不是“用了 Triton”，而是别人可以根据你的文档复现正确性、知道限制并理解关键公式。

## 5. 项目 B：从框架请求追到一个 Kernel

选择推理系统卷固定版本中的一个真实算子调用，例如 RMSNorm 或 attention backend。沿 Python 模块→算子接口→后端选择→GPU实现→返回张量，记录文件和符号。不要一开始选择最复杂的分布式 MoE；先选边界清晰的 Norm，再做 Attention。

对一个微小输入，写出 shape、dtype、stride、设备、别名、epsilon、mask、累加精度与输出。将该算子的数学行为与你的 CPU oracle 对照。若能够在隔离分支替换一个小形状路径，先添加单元测试，再运行局部验证。不要直接替换公司生产环境的安装包。

最终报告回答：这个算子解决什么数学问题；实际后端何时被选中；支持什么布局；错误在哪里校验；如何测试前后向；为什么某个硬件有不同实现。能回答这些，你已经从“听说过算子名字”进入代码级理解。

## 6. 练习与参考答案

1. **Collective 手算。** 两 rank 输入 `[1,2]`、`[10,20]`，sum AllReduce 结果？答案每 rank 均为 `[11,22]`；AllGather 则为 `[1,2,10,20]`，语义不同。
2. **分片。** 四 rank ReduceScatter 的输出元素数为什么比输入少？答案归约后的完整向量按 rank 分片，片段所有权不同；后接 AllGather 可恢复完整 AllReduce 结果关系。
3. **失败路径。** 一个 rank 校验失败退出，另一个在 collective 等待，如何避免无限等待？参考：尽可能在进入前统一检查、配置超时、记录各 rank 阶段，并管理整个作业退出；单个 finally 不能修复所有分布式失败。
4. **成本模型。** p=1 时 ring 公式为何为零？答案没有跨 rank 通信；不表示函数调用本身零开销，模型只包含通信部分。
5. **项目验收。** 为 attention capstone 增加“未来信息不得影响过去输出”的属性测试。参考：固定 Q，修改未来 K/V，比较早期行；同时确认最后一行确实变化，避免测试数据无效。
6. **贡献准备。** 将一个真实项目的小 bug 形成最小复现：失败输入、预期、当前输出、环境、测试修复。验收先本地完成可审查补丁；是否向上游提交由你决定，不自动代表公司或用户发言。

本卷总掌握门槛：能手写并测试基本向量/归约/Norm/GEMM/注意力；能读懂真实后端的一条路径；能记录数值与性能边界；能有依据地使用个人 GPU 与 H100，而不是凭硬件名称猜测。
