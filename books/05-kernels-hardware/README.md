# 卷 K：算子、编译器与 GPU 硬件

本卷从地址与程序开始，走到能手写、验证并阅读真实算子。内容为18章，每章按小课递进展开；另有3个逐步工作簿、固定版本源码研读，配套28个编号CPU实验、3个故障追踪实验、C++/CUDA/Triton实现与测试。目标是扎实的实现能力，不承诺读完就获得所有硬件上的SOTA优化能力。

## 从这里开始

1. 先完成[基础卷](../00-foundations/README.md)的Python、NumPy、Linux与调试；模型公式与[模型结构卷](../03-model-architectures/README.md)并行复习。
2. 顺序学习K01–K09，每一章先手算，再写自己的版本，最后对照参考。
3. 完成三个工作簿；这是本卷最重要的动手验收，不能只阅读结论。
4. 学习K10–K18，把正确性、编译、硬件、异步与性能连接起来。
5. 用固定commit阅读真实Triton、CUTLASS、FlashAttention；与[推理系统卷](../04-inference-systems/README.md)联合做调用链项目。

## 章节导航

| 阶段 | 章节 | 实际能力 |
|---|---|---|
| 地址与执行 | [K01 布局](01-addresses-layout.md)、[K02 C++](02-cpp-memory.md)、[K03 CUDA](03-cuda-execution.md)、[K04 访存](04-memory-hierarchy.md) | 手算地址、构建程序、线程覆盖、读写协作 |
| 核心算法 | [K05 归约与Norm](05-reductions-normalization.md)、[K06 GEMM](06-gemm.md)、[K07 在线注意力](07-online-attention.md) | 稳定公式、分块累加、前后向与等价性 |
| 设备实现 | [K08 Triton模型](08-triton-model.md)、[K09 Triton算子](09-triton-kernels.md)、[K10 框架集成](10-custom-ops-autograd.md) | mask、块张量、契约、autograd与集成 |
| 低精度与条件计算 | [K11 量化](11-quantized-kernels.md)、[K12 MoE](12-moe-grouped-gemm.md) | 编码/scale/累加、dispatch/combine |
| 执行与硬件 | [K13 异步与Graph](13-streams-events-graphs.md)、[K14 性能](14-performance-method.md)、[K15 编译](15-compilation.md)、[K16 Hopper/Blackwell](16-hopper-blackwell.md)、[K17 Warp/MegaKernel](17-warp-persistent-megakernel.md) | 依赖、计时、产物、架构约束与流水线 |
| 综合验收 | [K18 通信与项目](18-communication-capstone.md) | 注意力作品集、真实框架到Kernel追踪 |

必做工作簿：[W1 Softmax](workbook-softmax.md)、[W2 GEMM](workbook-gemm.md)、[W3 Attention](workbook-attention.md)。每本都有从契约、手算、CPU参考、故障注入到CUDA/Triton与验证的完整路径。

拓展：[固定源码研读](source-reading.md)、[一手资料](SOURCES.md)、[实验索引](../../labs/kernels/README.md)。修订依据见[统一深写标准](../../docs/DEPTH_STANDARD.md)，逐章覆盖、实际执行与未验证范围见[本卷深度审计](DEPTH_AUDIT.md)。

## 快速验证

在仓库根目录运行，CPU核心只依赖NumPy：

```bash
python -m labs.kernels.run --all
python -m labs.kernels.workbook all
python -m unittest tests.test_kernels_reference tests.test_kernels_workbooks -v
```

GPU始终显式启用：

```bash
python -m labs.kernels.inspect_environment
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.gpu_check --run
```

设备编号改成你获分配的GPU。没有torch/triton/GPU时输出SKIP，表示没有完成GPU验证，不是测试通过。本轮已实际执行210个正文CPU Python代码块、27个单元测试、28个编号CPU实验和3个工作簿；K02的7个完整主机C++程序与K15编译链实验也已编译运行。另29个PyTorch/Triton代码块只做Python语法检查，没有执行框架或GPU路径；CUDA、PTX与设备二进制命令没有执行。无需为了CPU阶段占用H100。

## 正文代码的执行方式

Python示例按章建立独立文件或Notebook，从前往后执行；后面的答案可能复用前文定义与输入，不要只复制函数调用。每个Python代码块在验证时单独进行语法检查，所有CPU块还按章共享命名空间执行。PyTorch/Triton块需要对应依赖与明确设备授权，不能把缺依赖跳过算作验证通过。

C++中每个带main的完整程序单独保存、单独编译，辅助定义按正文放在相应main之前；三文件链接实验按指定文件名拆开。CUDA代码要区分完整主机程序、仅设备内核与教学摘录，不能将摘录当成可直接构建的整个工程。正文明确注明的错误代码用于故障定位，先预测错误，再执行指定修复。

## 学习证据

每章至少十二道带完整答案的递进任务；建议先隐藏答案完成，再用自己的语言解释错误根因。保存自己的实现、失败案例、测试与环境，不要只保存PASS截图。完成一章意味着你能处理一个未见过的小变体，并知道哪些能力尚未实现；阅读完章节本身不等于验收通过。
