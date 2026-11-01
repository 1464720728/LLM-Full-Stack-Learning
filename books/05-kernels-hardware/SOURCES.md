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
