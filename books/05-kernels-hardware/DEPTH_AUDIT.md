# 算子卷深度扩写与验证审计

完成日期：2026-11-01。K01–K18主章均已整体重写，保留原章节路径、课程ID与已有实验接口。内容按先修词义、可手算轨迹、公式和硬件映射、完整代码、故障定位、至少十二道有完整答案的练习与项目推进；不再保留旧短章后追加长补充的双套结构。

篇幅使用仓库统一工具去除反引号或波浪线代码围栏后统计中文汉字，包含标题、表格和练习答案；它只是防止再次退化成几句概述的最低检查，不替代数学与代码审读。标准见[DEPTH_STANDARD](../../docs/DEPTH_STANDARD.md)。

## 逐章完成与实际执行

每章Python代码在独立命名空间中按出现顺序执行；PyTorch/Triton块只编译Python语法，不导入或运行缺失依赖。CPU执行包括答案、边界与负向案例中的断言，不只调用既有labs函数。

| 章 | 去代码汉字 | 本轮最低 | 已补齐的具体链条 | CPU Python执行 | 仅Python语法 |
|---|---:|---:|---|---:|---:|
| [K01](01-addresses-layout.md) | 14,497 | 14,000 | 字节/元素、shape/stride/offset、别名、广播、负步长、完整TensorView与BTHD布局 | 18 | 0 |
| [K02](02-cpp-memory.md) | 14,425 | 14,000 | C++语法、指针引用、生命周期、vector、RAII、模板、移动语义与多文件构建 | 0 | 0 |
| [K03](03-cuda-execution.md) | 14,052 | 14,000 | host/device、grid/thread/block/warp/SM、覆盖性、同步、grid stride与调度模型 | 4 | 0 |
| [K04](04-memory-hierarchy.md) | 14,046 | 14,000 | 逐指令sector、bank/broadcast、cache/spill、完整transpose、尾部与XOR布局 | 5 | 0 |
| [K05](05-reductions-normalization.md) | 14,007 | 14,000 | 归约树、浮点顺序、softmax/RMSNorm/LayerNorm前后向、Welford与有限差分 | 15 | 0 |
| [K06](06-gemm.md) | 14,029 | 14,000 | 标量/分块/寄存器GEMM、完整CUDA、双缓冲、Tensor Core合同、Split-K与反向 | 12 | 0 |
| [K07](07-online-attention.md) | 14,017 | 14,000 | 矩形注意力、在线状态推导、完整前反向重算、因果位置、GQA、稀疏与干预测试 | 10 | 0 |
| [K08](08-triton-model.md) | 14,019 | 14,000 | Program、广播、mask、指针、constexpr、布局、六类代码轨迹与IR边界 | 8 | 6 |
| [K09](09-triton-kernels.md) | 14,026 | 14,000 | 六类完整Triton算子、包装合同、尾部、误差矩阵、性质测试与设备验收入口 | 6 | 12 |
| [K10](10-custom-ops-autograd.md) | 14,164 | 14,000 | 标量/张量AD、广播梯度、版本检查、custom_op/fake/dispatch、高阶边界与训练步 | 8 | 7 |
| [K11](11-quantized-kernels.md) | 10,024 | 10,000 | IEEE/低精度编码、INT8/INT4/E2M1、舍入、pack、group scale与量化GEMM | 15 | 0 |
| [K12](12-moe-grouped-gemm.md) | 10,010 | 10,000 | 稳定路由、prefix/scatter、grouped GEMM、combine、容量、梯度与两rank模拟 | 15 | 0 |
| [K13](13-streams-events-graphs.md) | 10,068 | 10,000 | 异步提交、ready/free双依赖、事件、存储生命周期、图捕获与缓冲重放 | 9 | 4 |
| [K14](14-performance-method.md) | 16,280 | 10,000 | 工作量合同、Roofline、缓存流量、计时与统计、时间线/指标、六shape性能项目 | 18 | 0 |
| [K15](15-compilation.md) | 15,017 | 14,000 | 完整解析器/解释器/SSA/CSE/DCE/JIT缓存、浮点优化、真实主机编译、PTX/ABI | 21 | 0 |
| [K16](16-hopper-blackwell.md) | 10,199 | 10,000 | 硬件身份/兼容层、真实资源约束、tile模型、TMA描述器、字节完成与block scale | 14 | 0 |
| [K17](17-warp-persistent-megakernel.md) | 10,008 | 10,000 | 代际stage池、150合法交错、persistent调度、DAG执行/环检查、融合与排空 | 16 | 0 |
| [K18](18-communication-capstone.md) | 10,047 | 10,000 | 任意rank ring两阶段、collective合同、成本、完整注意力综合项目与负向控制 | 16 | 0 |

合计去代码正文 **232,935 汉字**；18章全部达到本轮篇幅下限。统一检测未发现跨章完全相同且达到120汉字的段落。正文Python共239块，其中 **210块实际执行通过，29块仅语法检查**。语法检查不能证明框架API、设备编译、数值、同步或性能正确。

## 主机C++与编译链的真实证据

- K02：六个带main的独立完整程序均以g++的C++17、Wall、Wextra、pedantic选项编译并实际运行，退出码均为0；math_ops.hpp、math_ops.cpp、main.cpp三文件示例也完成编译、链接与运行。同行复审在临时目录复现了正文保存路径和命令。零散函数/模板摘录不单独宣称为可执行程序。
- K15：affine.cpp实际完成预处理、O0与O2汇编生成、目标文件生成、链接与执行，输出35并退出0；优化汇编里实际存在affine符号。本机产物大小分别为预处理906374字节、O0汇编3499字节、O2汇编2523字节、目标文件2792字节，仅作为当次观察，不能推广为其他编译器的固定输出。
- 两章合计八个完整主机C++程序/构建案例实际执行。K15设备内核、PTX摘录、LLVM IR示例及nvcc/cuobjdump/nvdisasm命令均未编译或运行；普通主机C++成功不构成CUDA编译证据。

## 现有实验与回归检查

在仓库根目录实际执行：

~~~bash
python tools/check_depth.py --track K
python -m unittest tests.test_kernels_reference tests.test_kernels_workbooks -v
python -m labs.kernels.run --all
python -m labs.kernels.workbook all
python -m labs.kernels.inspect_environment
python -m labs.kernels.inspect_environment --gpu
python -m labs.kernels.gpu_check
python -m labs.kernels.benchmark
~~~

- 27个unittest通过；28个编号CPU实验全部通过；三个工作簿通过，包含故障注入与参考结果。K28注意力综合验收32例通过，最坏绝对误差约6.66e-16，属于float64 CPU参考范围，无性能主张。
- K14的18个Python块包含工作量、采样原始列表适配、单位错误、别名副作用、尾部mask和分派回退；samples_ms按原始毫秒样本处理，samples仍表示计数。GPU基准与Nsight采集没有执行。
- K17实际执行代际缓冲与150组合法交错检查；K18实际执行任意rank环形通信模拟及因果干预。它们验证CPU模型的协议逻辑，不等同NCCL通信实测。
- inspect_environment确认Python 3.12.14、NumPy 2.3.5、g++存在，torch、triton、nvcc和nvidia-smi不存在。默认探测未访问GPU；显式--gpu输出torch缺失的SKIP。gpu_check和benchmark在没有--run时均明确不运行。这些门控结果不计为GPU测试通过。

## 实际发现并修复的问题

- K01将掌握检查移动到全章末尾，保持小课单一顺序；完善全部代码在前置定义建立后的执行路径。
- K02明确每个main程序单独保存、辅助定义与头文件的关系，统一仓库根目录命令并补齐三文件链接步骤，避免初学者把整章多个main拼进同一文件。
- K06的Split-K答案曾提前引用后面才定义的a_test，现改用当前已经定义的a/b，并完整重跑本章12块。
- K12手算期望里的2.7涉及浮点表示，曾用array_equal作不合适的数值断言；改为allclose并解释整数路由结构精确比较与浮点结果容差比较的区别，15块重跑通过。
- K10为affine_backward明确逐元素乘加、weight与bias同形状、x无需广播的受限合同，避免把简化反向误当任意广播实现；训练步对比函数在正文块末显式调用。该PyTorch块仅语法检查，没有伪报训练结果。
- K15区分PTX文本阅读与二进制反汇编，PTX教学摘录明确省略项与32位索引假设；完整Python编译器通过32种表达式/形状/步长差分案例，并保留错误乘零、错误结果缓存、错误字节地址三种负向控制。

## 硬件、来源与后续维护边界

本轮没有安装PyTorch/Triton，没有执行任何GPU内核、CUDA编译、NCCL集体通信或性能采集。CUDA/Triton实现的完整性与CPU参考的正确性不能代替目标设备验证；用户在个人5070Ti或获分配H100上执行时，记录设备、驱动、工具链、dtype、shape、配置、数值和有界计时。

官方文档与固定源码快照见[SOURCES](SOURCES.md)及[source-reading](source-reading.md)。K16具体资源数值按Hopper与Blackwell官方指南核查；K17重新核查固定CUTLASS主循环的producer acquire/barrier/TMA、consumer wait/release与tail，没有虚构该主循环中不存在的显式producer_commit。

后续增加语法、dtype、布局、量化格式或硬件特性时，先更新输入输出合同与反例，再修改实现和完整答案，最后重跑相关正文与既有实验。若只完成语法检查、编译或CPU模型验证，记录对应层级；不得把未执行路径改写成已验证。
