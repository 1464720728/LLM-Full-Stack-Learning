# 一手资料与核查边界

核查日期：2026-11-01。下面的页面/文件在本次编写中实际检索或读取。课程中的推导、数值例子和代码为独立教学实现；不把官方教程的特定性能结果推广到本课程。动态文档可能更新，源码研读使用固定commit。

| 资料 | 用途与边界 |
|---|---|
| [NVIDIA CUDA GPU能力表](https://developer.nvidia.com/cuda/gpus) | 核对H100 9.0、B200 10.0、B300 10.3、RTX5070Ti 12.0；不据此推断所有指令继承关系 |
| [CUDA编程指南能力附录](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) | 能力与架构特性边界 |
| [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) | Windows驱动与WSL内Toolkit/驱动分工；安装前以当前文档为准 |
| [Microsoft WSL CUDA说明](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl) | Windows/WSL环境路线 |
| [NVIDIA驱动、Toolkit与架构矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) | 工具链兼容检查 |
| [NVCC文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) | 编译目标与设备代码产物 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 虚拟指令层、目标与指令语义；不将PTX直接等同SASS |
| [Blackwell兼容指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) | 通用与架构专用目标的兼容边界 |
| [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) | 分块层级与Hopper Warp分工的研读入口 |
| [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) | Collective与调度策略的结构入口 |
| [Triton向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | Program、地址和mask基本语义 |
| [Triton Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | 块内归约与融合研读 |
| [Triton GEMM](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 多维地址、Tile与调参研读 |
| [Triton教程目录](https://triton-lang.org/main/getting-started/tutorials/) | 进一步选择LayerNorm、Attention、grouped/persistent等任务 |
| [PyTorch自定义Python算子](https://docs.pytorch.org/tutorials/advanced/python_custom_ops) | schema/alias/fake/autograd/opcheck边界；实际API以安装版本为准 |
| [NCCL Collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | 集合通信结果关系与各rank匹配要求 |
| [FlashAttention项目](https://github.com/Dao-AILab/flash-attention) | 真实实现和支持边界；不把教学Kernel宣称为性能等价替代 |

实际读取的源码快照和符号见[源码研读](source-reading.md)。SHA仅代表阅读快照；未在交付环境安装、编译或基准测试这些上游项目。

本次环境有NumPy与C++编译器，没有PyTorch、Triton、nvcc或GPU。27个本卷unittest和CPU实验已通过，C++布局程序已编译运行；CUDA/Triton/PyTorch/NCCL代码是可选实操交付，不能将SKIP当成功运行。后续执行记录应包含实际版本与设备，更新本声明时保留原始证据。

## 深写版新增核查

以下资料用于新增的代码级讲解，核查日期同为2026-11-01。正文在具体讨论处保留对应链接；版本化PyTorch链接记录此次读取的页面，不要求读者盲目更换本机版本。

| 资料 | 对应内容 |
|---|---|
| [Triton语言API](https://triton-lang.org/main/python-api/triton.language.html) | K08广播、指针、mask、归约与类型 |
| [Triton编程模型](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) | Program与块级表示 |
| [PyTorch扩展自动求导](https://docs.pytorch.org/docs/2.15/notes/extending.html) | K10 Function、保存状态、反向合同 |
| [PyTorch library](https://docs.pytorch.org/docs/2.15/library.html) | custom_op、fake、autograd与opcheck |
| [CUDA FP4转换API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html) | K11编码、转换、舍入和饱和边界；教学E2M1取舍规则单独定义 |
| [NVIDIA异步执行指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | K13 stream/event与异步依赖 |
| [PyTorch CUDA语义](https://docs.pytorch.org/docs/2.15/notes/cuda.html) | 跨流使用、record_stream、图捕获与重放 |
| [Nsight Systems用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | K14时间线、采集范围与干扰 |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 指标口径、性能计数与解释边界 |
| [PyTorch CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html) | 设备事件计时的完成与同步条件 |
| [PyTorch benchmark教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | 预热、重复样本与测量合同 |
| [LLVM语言参考](https://llvm.org/docs/LangRef.html) | K15 SSA、类型、指令与优化假设 |
| [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | cubin、cuobjdump、nvdisasm输入与产物区别 |
| [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | K16共享存储、资源与异步能力边界 |
| [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 数据中心和消费级能力差异、资源约束 |
| [CUTLASS Blackwell功能](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) | SM100相关机制，不能直接外推为SM120同等支持 |
| [CUTLASS tcgen05指南](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html) | 新矩阵指令的参与、布局与完成协议 |
| [Triton Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | K17持久工作分配研读入口 |
| [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) | K18集合通信接口与进程组 |

深写版最终验证详见[DEPTH_AUDIT](DEPTH_AUDIT.md)：210个正文CPU Python块执行通过，29个PyTorch/Triton块仅语法检查，主机C++完整程序实际构建运行。官方文档阅读、源码静态研读、CPU教学模型、主机编译与真实GPU实测是不同证据层级，本轮没有执行CUDA/Triton/NCCL设备路径。
