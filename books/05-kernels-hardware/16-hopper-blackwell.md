# K16 H100 与 RTX 5070 Ti：从硬件能力到能验证的算子路径

你已经有一台随时可用的 Win11 / RTX 5070 Ti 电脑，以及可以安排实验的 Linux / 八卡 H100 服务器。这个条件适合把学习分成两条互相支持的路线：大量算法、布局、归约、普通矩阵算子在个人设备反复练习；需要 Hopper 专用指令、较大容量或多卡通信的实验，再使用相应 H100。决定在哪台机器运行的依据是实验问题与硬件契约，不是题目听起来是否高级。

本章先修 K01 的地址、K03 的线程组织、K04 的存储层次、K06 的分块 GEMM，以及 K15 的编译基本路线。你会学会区分产品、架构、编译目标和软件包；计算一份 tile 到底需要多少片上存储；用 CPU 模型理解张量搬运、参与线程与完成协议；最后做一份有界的双环境实验记录。Hopper 专用程序的深入流水线在 K17 继续展开。

资料核查日为 2026-11-01。正文链接指向 NVIDIA、CUDA、CUTLASS 的一手资料；动态文档使用时仍需核对安装版本。这里的 CPU 算例已执行，GPU、CUDA 编译与设备性能未在本次编写环境实测。下面不会把示例数字冒充你两台机器的测量结果。

本章的 Python 代码块按出现顺序放在同一个练习文件里执行，后文会复用前文导入的 `dataclass`、NumPy 和辅助函数。每完成一课，先运行到当前文件末尾，确认断言通过，再追加下一课；不要把依赖前文定义的片段直接当作独立程序。PowerShell、Bash 与 CUDA 命令则在标明的环境分别执行。

## 第一课：同一台 GPU 有几种不同的身份

“H100”“Hopper”“9.0”“sm_90a”“CUDA Toolkit”回答的是不同问题。H100 是产品名称，Hopper 是架构家族，Compute Capability 描述设备编程能力类别，编译目标决定程序允许使用哪些指令和生成哪类设备代码，Toolkit 是开发工具集合。显存容量和互连还与具体 SKU、整机和运行配置有关，不能从一个词推导出全部信息。

[NVIDIA 官方能力表](https://developer.nvidia.com/cuda/gpus) 列出 H100/H200 为 9.0、B200 为 10.0、B300 为 10.3、RTX 5070 Ti 为 12.0。这张表适合回答“我的设备属于哪个能力目标”，不适合回答“哪张卡在所有任务上更快”。尤其不能把 12.0 读成 CUDA Toolkit 12.0，也不能把编号大小当作所有专用指令严格包含关系。

| 名称或记录 | 实际回答的问题 | 还不能回答的问题 |
|---|---|---|
| H100 或 RTX 5070 Ti | 使用哪个产品系列 | 具体可用显存、驱动、编译包 |
| Hopper 或 Blackwell | 属于哪个架构家族 | 同家族所有设备是否支持同一 kernel |
| CC 9.0 或 12.0 | 设备能力类别 | 当前 Python 包是否包含合适二进制 |
| `sm_90a` | 生成特定架构加速路径 | 换成另一个数字是否仍然合法 |
| `torch.version.cuda` | 该 PyTorch 构建的 CUDA 版本信息 | 系统是否装了同版本 nvcc |
| `nvcc --version` | 当前调用的编译工具版本 | 当前驱动是否能运行所有生成代码 |

学习者常遇到一种困惑：`nvidia-smi` 显示一个 CUDA 版本，Python 又显示另一个，命令行 nvcc 还是第三个。它们不一定互相矛盾，因为观察的是驱动支持、框架构建依赖与本地编译器三个层面。判断是否兼容要看官方兼容矩阵和实际程序，而不是要求三个字符串机械相等。

先写一个纯数据分类练习。显式区分字段，比在日志里只保存一个 `cuda_version` 更可靠。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DeviceRecord:
    product: str
    capability: tuple[int, int]
    memory_bytes: int | None
    toolkit: str | None
    framework_build: str | None

example_hopper = DeviceRecord("H100 (teaching label)", (9, 0), None, None, None)
example_pc = DeviceRecord("RTX 5070 Ti (teaching label)", (12, 0), None, None, None)
assert example_hopper.memory_bytes is None
assert example_pc.capability != (1, 2)
```

这里故意不填写显存容量和软件版本，因为没有查询你的机器。`None` 表示未知，不是零容量。报告中的未知值应该保留下来，直到实际测量或可靠来源填补；把常见规格先填进去，会让后面容量计算看似精确，实际基于未经确认的输入。

## 第二课：从 Windows 到 WSL，再到 Python 环境

WSL2 让你在 Windows 上使用 Linux 用户空间，适合本课程的 shell、Python、C++ 和许多 GPU 工具链练习。[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 明确区分 Windows 主机驱动与 WSL 内的 CUDA 开发工具：主机安装支持 WSL 的驱动，WSL 通过映射接口使用它，不应再把 Linux 显示驱动覆盖进 WSL 环境。

先在 Windows 终端检查 WSL 发行版与版本，再进入自己的 Linux 环境。下面是检查命令，不会替你执行整套系统安装。

```powershell
wsl --list --verbose
nvidia-smi
```

进入 WSL 后，从课程仓库根目录执行：

```bash
uname -a
python --version
python -m labs.kernels.inspect_environment
```

环境探测入口默认只检查包和命令是否存在，不启动 GPU 工作。输出中的 `torch=true` 只表示能找到包，不表示 CUDA backend 一定可用；`nvcc=false` 只表示当前 PATH 找不到编译器，不表示预编译 PyTorch 一定不能运行。每一步只回答它实际检查的问题，别把一个绿色布尔值当成整个软件栈的证明。

需要 GPU 设备信息时，再显式运行：

```bash
python -m labs.kernels.inspect_environment --gpu
```

它会通过 PyTorch 查询当前可见设备的名称、capability 与显存，记录框架及其 CUDA 构建信息。共享服务器上应先按分配结果设置 `CUDA_VISIBLE_DEVICES`，让本进程只看见你获分配的设备。逻辑 `cuda:0` 指可见列表中的第一张，并不总等于整机物理编号零；不要仅凭一个数字判断是否选对卡。

在 WSL 里缺 nvcc 时，先问自己正在做什么。如果只是执行包含所需运行依赖的预编译框架，可能暂时不需要系统编译器；如果要编译自己的 `.cu` 文件或某个本地扩展，就需要匹配的 Toolkit。安装前查当前官方 WSL 安装说明，识别软件包是否连带驱动，不要把原生 Linux 的驱动安装命令照搬进来。

Python 虚拟环境也不能混用。Windows 的 Python 与 WSL 的 Python 是不同环境，某个包在 Windows 安装成功，不意味着 WSL 能 import。使用 `python -m pip` 可以让安装目标明确落在当前解释器；课程的基础卷已经展开解释器、PATH 和虚拟环境，这里把它们与实际 GPU 排错连接起来。

## 第三课：能运行、能编译、数值正确、性能合适是四个检查

第一层是加载兼容：进程能找到合适设备代码。第二层是编译兼容：工具链能理解源码、目标与所用指令。第三层是语义正确：输出满足公式、掩码和同步合同。第四层是性能适合：在给定 shape、dtype 和设备状态下达到你的目标。某一层通过不能替代后面的层次。

[Blackwell 兼容指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html) 区分 cubin 与 PTX。前者是目标设备机器代码，后者可以在运行时进一步编译；一般 PTX 的向前兼容规则有架构专用目标例外，例如 `compute_90a` 不能被当作通用于 Blackwell 的 PTX。一个包若既没有可用 cubin，也没有适用 PTX，就可能在启动 kernel 时失败。

这解释了为什么“只是把 sm_90a 改成 sm_120”不是可靠迁移方法。源码可能包含只在某类目标合法的操作数布局、参与线程约束或同步指令；改变标签不会自动把算法改写成另一套实现。框架通常需要选择不同 backend 或 kernel 路径，而不是假定所有目标共享一个实现。

写一个保守的教学分派器，只选择我们明确声明的路径。它不是 NVIDIA 完整指令数据库，也不根据编号大小猜测未知设备支持。

```python
def choose_teaching_path(capability, request_hopper=False):
    if request_hopper:
        if capability == (9, 0):
            return "hopper-specialized: verify toolchain and kernel contract"
        raise ValueError("this lesson's Hopper path requires verified CC 9.0")
    if capability in {(9, 0), (12, 0)}:
        return "portable-reference: still verify installed backend"
    return "unknown-target: inspect documentation before selection"

assert choose_teaching_path((12, 0)).startswith("portable")
assert choose_teaching_path((9, 0), True).startswith("hopper")
try:
    choose_teaching_path((12, 0), True)
except ValueError:
    pass
else:
    raise AssertionError("specialized path selected by numeric ordering")
```

真实软件的 dispatch 条件还应包含 dtype、shape、stride、对齐、共享内存、编译宏与包版本。设备能力匹配只是一个条件。即使 H100 支持某指令，你给它一个不满足布局要求的输入，kernel 仍可能错误；即使编译成功，错误的线程参与也可能触发未定义行为。

把错误按层次记录会显著缩短排查。`no kernel image` 一类错误先查二进制与目标；不认识 `sm_120` 一类编译错误先查实际 nvcc；数值错误先查数据合同与同步；稳定但很慢再查计时、布局和资源。不要看到任何 CUDA 字样就重装整个系统，那会改变太多变量，也会抹掉原始证据。

## 第四课：同一份 tile 为什么在两种设备上有不同资源压力

GPU 的全局显存容量很大，但一个 thread block 能直接协作使用的 shared memory 小得多。一个模型能放进显存，不代表它的某个 tile 配置能合法启动。相反，矩阵很小也可能因为每 block 资源过大而失败。资源约束是按层级施加的，必须明确是在设备、SM、block 还是线程层面计算。

官方 [Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) 与 [Blackwell 调优指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) 给出的相关上限不同：CC 9.0 的每 SM shared memory 为 228 KB、每 block 最大 227 KB；CC 12.0 分别为 128 KB 和 99 KB。这些是文档中的架构上限，实际 kernel 还受配置、预留和动态 shared memory opt-in 等条件影响。

考虑教学 GEMM tile：$B_M=B_N=128$、$B_K=32$，A/B 元素各两字节。A tile 是 $128\times32\times2=8192$ 字节，B tile 同样八千一百九十二字节，一套输入 stage 共十六 KiB。三 stage 就是四十八 KiB。如果 epilogue 另外同时占用一个 $128\times128$ 的 FP32 缓冲区，就是再加六十四 KiB，总计一百一十二 KiB。

```python
def tile_storage(bm, bn, bk, stages, input_bytes=2, output_bytes=4,
                 epilogue_overlaps=True, extra_bytes=0):
    if any(type(x) is not int or x <= 0 for x in
           (bm, bn, bk, stages, input_bytes, output_bytes)):
        raise ValueError("positive integer dimensions required")
    if type(extra_bytes) is not int or extra_bytes < 0:
        raise ValueError("invalid extra storage")
    inputs = stages * (bm * bk + bk * bn) * input_bytes
    epilogue = bm * bn * output_bytes
    peak = inputs + epilogue if epilogue_overlaps else max(inputs, epilogue)
    return {"input_bytes": inputs, "epilogue_bytes": epilogue,
            "peak_bytes": peak + extra_bytes}

large_tile = tile_storage(128, 128, 32, 3)
assert large_tile == {"input_bytes": 49152, "epilogue_bytes": 65536, "peak_bytes": 114688}
assert large_tile["peak_bytes"] > 99 * 1024
assert large_tile["peak_bytes"] < 227 * 1024
small_tile = tile_storage(64, 128, 32, 2, extra_bytes=2048)
assert small_tile["peak_bytes"] == 59392
print(large_tile, small_tile)
```

这里假设输入 stage 与 epilogue 缓冲区同时存活。如果代码能够证明二者生命周期不重叠并合法复用同一区域，峰值应取最大值而不是相加；但不能只为了让数字装得下就把选项改成 False。必须追踪最后一次读取和首次覆盖的时间，确认异步矩阵操作已经结束。资源计算与同步证明在这里直接相连。

再考虑寄存器。一个 $128\times128$ 输出 tile 有一万六千三百八十四个累加值，如果简化地平均分给一百二十八线程，每线程仅累加器就需一百二十八个 FP32 值，还没算地址、临时量与循环状态。这只是数量估计，不是硬件 fragment 布局，但足以提醒你：扩大 tile 可能减少全局访存，却同时提高寄存器压力。

所以迁移时要记录至少五个量：block 线程数、每线程寄存器、静态 shared memory、动态 shared memory、stage 数。编译器报告提供实际分配证据，CPU 算例提供设计前预估。两者不一致时，继续查对齐、padding、临时缓冲和编译器 spill，而不是随意把某个文档容量改成自己希望的数字。

## 第五课：occupancy 是资源约束结果，不是性能评分

Occupancy 通常描述实际驻留 warp 数占架构允许最大驻留 warp 数的比例。驻留意味着资源已经分配，可以被调度，并不表示所有 warp 此刻都在发射有用指令。低 occupancy 可能缺少隐藏等待的机会，也可能是高效 Tensor Core kernel 为大 tile 付出的合理代价；高 occupancy 同样可能被访存或依赖链限制。

下面使用明确标为虚构的硬件参数，学习如何取多个资源上限的最小值。不要把函数输出直接当成 CUDA occupancy API 的结果，因为真实分配粒度、保留资源、cluster 等因素没有纳入。

```python
from math import ceil

def occupancy_bound(threads, regs_per_thread, shared_per_block, limits):
    if threads <= 0 or regs_per_thread <= 0 or shared_per_block <= 0:
        raise ValueError("positive resource requests required")
    if threads > limits["max_threads_per_block"]:
        return {"blocks": 0, "warps": 0, "fraction": 0.0}
    warps_per_block = ceil(threads / 32)
    blocks = min(limits["blocks"], limits["threads"] // threads,
                 limits["registers"] // (threads * regs_per_thread),
                 limits["shared"] // shared_per_block,
                 limits["warps"] // warps_per_block)
    warps = blocks * warps_per_block
    return {"blocks": blocks, "warps": warps, "fraction": warps / limits["warps"]}

fictional = {"blocks": 16, "threads": 2048, "registers": 65536,
             "shared": 128 * 1024, "warps": 64, "max_threads_per_block": 1024}
case_a = occupancy_bound(256, 32, 32 * 1024, fictional)
case_b = occupancy_bound(256, 96, 32 * 1024, fictional)
assert case_a == {"blocks": 4, "warps": 32, "fraction": 0.5}
assert case_b == {"blocks": 2, "warps": 16, "fraction": 0.25}
```

第一例受 shared memory 限制，最多四个 block；第二例寄存器需求提高，每 block 需要二万四千五百七十六个寄存器，最多驻留两个。把 shared memory 减半也不自动救回第二例，因为瓶颈已经转到寄存器。这就是为什么调参要追踪哪个约束正在起作用，而不是只把所有参数往“小”方向调。

还有一个离散效应：资源稍微变化，驻留 block 数可能突然从二变一。性能曲线因此不一定平滑。你可以枚举 stage、tile 与寄存器估计，观察阶跃；然后只挑少量合理候选上 GPU 验证。CPU 模型能缩小搜索范围，却无法决定真正最快配置，因为它没有模拟指令吞吐、缓存命中和数据依赖。

在真实 cluster kernel 上，应使用相应官方 occupancy 查询与启动约束，而不是只用普通 block 上限除法。cluster 要求一组 block 协作驻留，调度粒度发生变化；剩余 SM 数量不能总是任意拼出下一组。理解资源是按组分配的，后面看 persistent 或 cluster launch control 才不容易把“更多 block”当作必然更快。

## 第六课：TMA 先理解“搬哪块数据”，再理解“谁发起搬运”

普通 tiled GEMM 通常由很多线程各自计算地址、加载元素、写入 shared memory，再通过同步让消费者读取。TMA，即 Tensor Memory Accelerator，为特定张量搬运提供硬件机制，能够用张量描述和坐标表达搬运任务。它减少某些搬运路径中显式地址工作和寄存器中转，但不会替你决定正确的逻辑矩阵，也不会取消完成同步。

先把真实指令放一边，用一个二维张量描述器说明最基本的信息：底层存储、逻辑 shape、行列 stride、起始偏移。搬一个 tile 时，每个逻辑坐标都要映射到正确位置；越界位置如何处理也要明确。下面模型选择越界填零，负 stride 不支持，单位统一为元素，故意不冒充 CUDA TensorMap 的二进制格式。

```python
import numpy as np

@dataclass(frozen=True)
class TeachingTensorMap:
    shape: tuple[int, int]
    strides: tuple[int, int]
    offset: int = 0
    def validate(self, storage_size):
        if any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("positive shape required")
        if any(type(s) is not int or s <= 0 for s in self.strides):
            raise ValueError("positive strides required")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("invalid offset")
        last = self.offset + sum((n - 1) * s for n, s in zip(self.shape, self.strides))
        if last >= storage_size:
            raise ValueError("descriptor exceeds storage")
    def tile(self, storage, origin, tile_shape):
        self.validate(len(storage))
        if any(type(n) is not int or n <= 0 for n in tile_shape):
            raise ValueError("positive tile shape required")
        out = np.zeros(tile_shape, dtype=storage.dtype)
        for i in range(tile_shape[0]):
            for j in range(tile_shape[1]):
                row, col = origin[0] + i, origin[1] + j
                if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                    address = self.offset + row * self.strides[0] + col * self.strides[1]
                    out[i, j] = storage[address]
        return out

storage = np.arange(24, dtype=np.float32)
row_major = TeachingTensorMap((4, 6), (6, 1))
np.testing.assert_array_equal(row_major.tile(storage, (2, 4), (3, 3)),
                              [[16, 17, 0], [22, 23, 0], [0, 0, 0]])
transposed = TeachingTensorMap((6, 4), (1, 6))
np.testing.assert_array_equal(transposed.tile(storage, (1, 2), (2, 2)),
                              [[13, 19], [14, 20]])
```

同一块二十四元素的存储，第一份描述把它看成四行六列，第二份看成六行四列的转置视图；数据没有实际复制。真实搬运描述也需要尊重逻辑形状与物理步长。若你只把 M/N 交换却没改 stride，得到的是另一种错误地址解释，而不是转置优化。

从元素地址换成字节地址时，还必须乘以元素字节数。例如第一份描述中逻辑坐标二、四对应元素十六，FP32 字节偏移为六十四。描述器接口若要求字节 stride，而你传入元素 stride 六，行距就会错四倍。许多“偶尔正确”的 kernel 正是因为小 shape 或特殊 dtype 暂时掩盖了单位混淆。

真正 TMA 还有对齐、维度、swizzle、描述符编码以及具体指令方向的限制，必须根据实际 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 和所用高级封装核对。上面 CPU 模型支持的任意正 stride，不代表硬件描述符都接受；模型的零填充选择也不能推导所有指令的边界行为。它的价值是让你先能算对“想搬什么”，再检查“硬件是否允许这样表达”。

学习 TMA 的正确入口是已有数值正确的 GEMM。保留普通搬运基线，只替换搬运路径，验证非整除边界、不同 stride、不同 stage 与同步。若同时改 tile、layout、精度和 epilogue，结果变快或变错都很难归因。优化实验一次改变一组有明确机制联系的因素，才能留下可复用知识。

## 第七课：发起传输不等于数据可读，数据可读也不等于缓冲区可复用

异步搬运引入至少三个时刻：发起、搬运完成、最后一个消费者完成。生产者发起后可以继续做别的事，但消费者不能提前读取；消费者开始读取后，生产者也不能马上覆盖该 stage。一个名为 ready 的布尔值通常无法同时表达这三个状态，尤其当同一缓冲区被循环用于多个代际时。

真实 TMA 与 barrier 协议包含到达计数、事务完成等机制，具体语义要跟实际 API 核对。这里构造一个原创的字节完成教学模型，分别记录“参与方到达”和“预期字节完成”，两者都满足后才允许消费。它不是 PTX mbarrier 的逐字段实现，仅用于暴露两类条件不能互相替代。

```python
class TransferStage:
    def __init__(self, generation, participants, expected_bytes):
        if participants <= 0 or expected_bytes <= 0:
            raise ValueError("positive counts required")
        self.generation = generation
        self.participants = participants
        self.expected_bytes = expected_bytes
        self.arrived = set()
        self.completed_bytes = 0
        self.consumed = False
    def arrive(self, participant):
        if not 0 <= participant < self.participants or participant in self.arrived:
            raise ValueError("invalid or duplicate participant")
        self.arrived.add(participant)
    def complete(self, generation, byte_count):
        if generation != self.generation:
            raise ValueError("stale generation")
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError("invalid completion size")
        if self.completed_bytes + byte_count > self.expected_bytes:
            raise ValueError("too many completed bytes")
        self.completed_bytes += byte_count
    @property
    def readable(self):
        return len(self.arrived) == self.participants and self.completed_bytes == self.expected_bytes
    def consume(self):
        if not self.readable or self.consumed:
            raise RuntimeError("stage not ready or already consumed")
        self.consumed = True
    @property
    def reusable(self):
        return self.consumed

stage = TransferStage(generation=5, participants=2, expected_bytes=16384)
stage.arrive(0)
stage.arrive(1)
assert not stage.readable
stage.complete(5, 8192)
assert not stage.readable
stage.complete(5, 8192)
assert stage.readable and not stage.reusable
stage.consume()
assert stage.reusable
```

这里 `consume()` 把消费者工作简化成同步完成，所以调用后可复用。若消费者是异步 WGMMA，就必须把“开始消费”和“消费完成”再拆开，不能直接照搬这个函数。K17 的完整 stage 状态机正是继续处理这个问题，并包含多代际、合法交错和尾部排空。

手算一下错误预期字节的后果。若 A/B 各搬八千一百九十二字节，真实总量一万六千三百八十四，但你只登记一半，模型会在第一份完成后错误地认为字节齐全；若登记两倍，则即使真实数据都到齐，等待仍无法结束。死锁可能来自计数定义错误，而不是机器慢，因此排错要打印预期、已完成和代际，不能只增加超时。

代际也是必要信息。同一个物理 stage 先存第零块，再存第二块，旧完成通知不能让新块提前 ready。教学模型显式传入 generation 并拒绝不匹配；真实协议可能用 phase 位、状态索引和更强的顺序条件实现。一个 phase 位只有两个值，为什么仍能工作，依赖循环复用与未完成工作数量约束，不能把它当作无限唯一版本号。

## 第八课：WGMMA 的参与单位与普通线程分工不同

Warp 是一组线程的执行组织，Hopper WGMMA 使用 warpgroup 级矩阵乘加协作，一个 warpgroup 包含四个 warp，也就是一百二十八线程。PTX 对相应指令的目标、所有线程参与和一致条件有明确要求；当前文档的 `wgmma.mma_async` 目标为 `sm_90a`。它不是让任意几个线程分别算出一小块再自动拼起来。

一个常见错误说法是“一个 warp 搬数据，剩下三个 warp 用 WGMMA 算矩阵”。角色分工本身可以成立，但若消费者只剩三个 warp，就不能直接满足一个完整 warpgroup 的 WGMMA 参与要求。真实 Hopper kernel 常组织独立生产者与消费者 warpgroup，实际发起搬运的线程可以更少；不能从“有一个搬运角色”推导 block 总共只需要四个 warp。

用集合检查一个简化参与合同：起点按一百二十八对齐，必须恰好包含完整的一百二十八个线程。这个模型不验证实际寄存器 fragment、描述符或指令编码，只负责让参与范围可视化。

```python
def require_full_warpgroup(thread_ids, start=0):
    if start < 0 or start % 128:
        raise ValueError("warpgroup start must be aligned")
    expected = set(range(start, start + 128))
    if set(thread_ids) != expected or len(thread_ids) != 128:
        raise ValueError("incomplete or duplicated warpgroup participation")
    return True

assert require_full_warpgroup(list(range(128)))
assert require_full_warpgroup(list(range(128, 256)), 128)
try:
    require_full_warpgroup(list(range(32, 128)))
except ValueError:
    pass
else:
    raise AssertionError("three warps incorrectly accepted")
```

边界 tile 的无效元素应该通过合法的数据掩码、零填充或相应路径处理，而不是让部分参与线程绕过要求集体执行的指令。数据有效性和控制参与是不同维度：某线程负责的数学元素可能无效，但它仍可能必须参加协作指令。K03 的 barrier 分歧问题在这里以更复杂的形式重新出现。

异步矩阵操作还需要遵守寄存器、共享内存与完成次序。PTX 中 fence、commit group、wait group 的职责不能合并成一个模糊的“同步一下”。例如等到搬运结束只证明输入已到达，不能证明矩阵操作已经读完；看到矩阵指令已发射也不能立即复用输入 stage。读成熟实现时按数据最后一次被使用的时间追踪，会比只记函数名更有效。

不要在这一课直接手写几行 inline PTX 后宣布完成 Hopper GEMM。合法实现还需要描述符、布局、累加器分布和完整同步。下一章沿固定 CUTLASS 文件追生产者、消费者与释放索引，提供了进入这些细节的连续路线；本章先把参与单位和资源约束建立扎实。

## 第九课：Blackwell 的新矩阵路径为什么不能按家族名套用

Blackwell 同时包含不同计算能力路线。数据中心 SM100 的 CUTLASS 文档介绍 `tcgen05.mma`、相应的 TMEM 累加器组织和低精度 block scale 路径；消费级 SM120 有自己的矩阵指令和布局支持。你的 5070 Ti 属于 CC 12.0，不应仅因也叫 Blackwell 就选择一个明确针对 SM100 的示例。

TMEM 可以先理解为特定矩阵路径使用的专门片上存储机制，而不是普通全局 tensor 的另一个名字。阅读 [tcgen05 编程指南](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html) 时，要分别找到分配、片段布局绑定、矩阵操作、读出与释放。TMEM 的列与逻辑输出列未必一一对应，不能直接把它当作普通 C 连续二维数组索引。

你不需要因为暂时没有 B200 就停止理解这些机制。可以在 CPU 上推导数学与资源拥有关系，阅读固定源码，并明确未做设备验证。真正要验证 SM100 专用指令时需要匹配设备；H100 和 5070 Ti 都不能替这个目标提供成功证据。硬件条件不同不妨碍学机制，但限制了能够声称完成的实测范围。

低精度 block scale 也需要把数学与硬件布局拆开。假设 A 的每行沿 K 每两项共享一个 scale，B 的每列同样每两项共享一个 scale，输出是对反缩放后的元素做乘加。不同 K 组的 scale 不同，不能在整个点积结束后只乘一个统一系数。下面是完整 CPU 参考，不声称实现 NVFP4 或 MXFP4 的实际编码。

```python
def block_scaled_matmul(a_codes, b_codes, a_scales, b_scales, group):
    a_codes = np.asarray(a_codes, dtype=np.float64)
    b_codes = np.asarray(b_codes, dtype=np.float64)
    if a_codes.ndim != 2 or b_codes.ndim != 2 or a_codes.shape[1] != b_codes.shape[0]:
        raise ValueError("invalid matrix shapes")
    if type(group) is not int or group <= 0:
        raise ValueError("positive group required")
    m, k = a_codes.shape
    n = b_codes.shape[1]
    groups = (k + group - 1) // group
    a_scales = np.asarray(a_scales, dtype=np.float64)
    b_scales = np.asarray(b_scales, dtype=np.float64)
    if a_scales.shape != (m, groups) or b_scales.shape != (groups, n):
        raise ValueError("scale shapes mismatch")
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            for p in range(k):
                g = p // group
                out[i, j] += (a_codes[i, p] * a_scales[i, g]) * (b_codes[p, j] * b_scales[g, j])
    return out

a_codes = np.array([[1, 2, 3, 4]], dtype=np.float64)
b_codes = np.array([[2], [1], [1], [2]], dtype=np.float64)
a_scales = np.array([[0.5, 2.0]])
b_scales = np.array([[2.0], [0.25]])
scaled = block_scaled_matmul(a_codes, b_codes, a_scales, b_scales, 2)
assert scaled[0, 0] == 9.5
expanded_a = a_codes * np.repeat(a_scales, 2, axis=1)
expanded_b = b_codes * np.repeat(b_scales, 2, axis=0)
np.testing.assert_array_equal(scaled, expanded_a @ expanded_b)
```

前两项代码点积为四，scale 乘积为一，贡献四；后两项点积为十一，scale 乘积为二分之一，贡献五点五；总和九点五。若误用第一组 scale 到所有项，结果会变十五。这个手算足以说明为什么 scale 轴、组大小和广播布局是算子合同，而不是量化文件旁边随便附上的元数据。

真实低精度格式还包含编码、舍入、饱和、scale 类型、分组长度和打包要求。[CUTLASS SM100 GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) 按类型、布局、对齐列出支持组合。阅读时挑一行具体组合，把每一列对应到你的输入，而不是只看标题里有 FP4 就假定所有四比特数据都能直接进入 Tensor Core。K11 与 M13 分别从算子和模型误差角度继续训练这部分能力。

## 第十课：先做成本手算，再选择值得上设备验证的配置

硬件特性不是免费的加速按钮。TMA 可能减少搬运指令与寄存器压力，但描述符、同步和流水线组织也有成本；更大 tile 可能增加复用，却提高寄存器与 shared memory 占用；更多 stage 可以隐藏等待，也会挤掉驻留 block。理解这些取舍，需要把“快”拆成工作量、搬运量、并行重叠和启动成本。

先用 GEMM 的简化下界估计。对 $M\times K$ 与 $K\times N$ 相乘，按一次乘加算两次浮点操作，工作量约为 $2MKN$。理想情况下 A/B 各读一次、C 写一次，字节量为输入元素数乘输入字节数再加输出字节数。真实 kernel 可能重复加载、使用中间结果或受缓存影响，所以这是分析起点，不是实测流量。

```python
def gemm_cost(m, n, k, input_bytes=2, output_bytes=4):
    flops = 2 * m * n * k
    traffic = (m * k + k * n) * input_bytes + m * n * output_bytes
    return flops, traffic, flops / traffic

flops, traffic, intensity = gemm_cost(128, 128, 128)
assert flops == 4194304
assert traffic == 131072
assert intensity == 32.0
fictional_flops_per_second = 100e12
fictional_bytes_per_second = 2e12
compute_lower_s = flops / fictional_flops_per_second
memory_lower_s = traffic / fictional_bytes_per_second
assert memory_lower_s > compute_lower_s
print({"fictional_compute_us": compute_lower_s * 1e6,
       "fictional_memory_us": memory_lower_s * 1e6})
```

算术强度是三十二 FLOP/byte。在这组明确虚构的峰值假设下，搬运下界比计算下界大，但两者都非常短，小矩阵还可能主要受启动开销限制。不能把两个下界中的最大值直接写成实际运行时间，更不能由此宣布你的 H100 能达到某个毫秒结果。它只帮助提出假设：优先减少重复搬运或融合启动是否值得测试。

再看流水线的理想模型。八个 tile，每个加载三微秒、计算五微秒，完全串行需六十四微秒；若加载与计算能在两个独立资源上完美重叠，首尾开销加稳态得到四十三微秒。这里假设足够缓冲、没有额外同步和带宽竞争，真实结果可能更慢。

```python
def two_stage_ideal(tiles, load_us, compute_us):
    if type(tiles) is not int or tiles <= 0 or min(load_us, compute_us) < 0:
        raise ValueError("invalid pipeline parameters")
    serial = tiles * (load_us + compute_us)
    pipelined = load_us + compute_us + (tiles - 1) * max(load_us, compute_us)
    return serial, pipelined

assert two_stage_ideal(8, 3, 5) == (64, 43)
assert two_stage_ideal(1, 3, 5) == (8, 8)
assert two_stage_ideal(8, 0, 5) == (40, 40)
```

第三个断言说明加载已经近乎零成本时，再增加搬运重叠不会带来收益。第二个说明只有一个 tile 时没有稳态可以重叠。高级流水线通常更适合有足够重复工作且瓶颈明确的场景；给极小输入套上复杂机制，可能只增加固定开销。学习时把这种反例也保留，才能摆脱“新技术总是更快”的直觉。

## 第十一课：在个人 GPU 上完成一份有界的正确性与计时记录

先完成本卷 CPU 参考和边界测试，再使用现有 GPU 入口。它们不下载模型，不启动常驻服务，默认不会执行设备工作；需要显式参数才运行。个人设备第一次运行可以选择短小的向量、归约、softmax 和小矩阵，重点观察编译、数值与错误信息，而不是立即追逐最大吞吐。

```bash
python -m labs.kernels.gpu_check
python -m labs.kernels.gpu_check --run
python -m labs.kernels.benchmark --run --op matmul --size 256 --repeats 10
```

第一条解释如何显式运行；第二条在依赖和设备存在时执行多种边界 shape 的正确性检查；第三条只对小矩阵做有界重复。缺 Torch、Triton 或 CUDA 设备时会显示 SKIP，这表示本机未完成 GPU 验证，不是算子通过。把这个状态原样写进记录，等到正确设备上再补真实结果。

`gpu_check` 对向量长度一、三十一、二百五十七等非整齐规模做检查，对 attention 还覆盖因果与非因果模式。这些输入是在验证 mask、尾块和维度，不是为了选最好看的性能数字。真实算子能通过规整大矩阵，却在最后一个不满 block 的 tile 写越界，仍然是不正确实现。

计时程序同时记录首次调用的主机时间与预热后的 CUDA event 区间。首次调用可能包含 JIT 编译和初始化，不应混入稳态 kernel 指标。event 区间也有明确范围：围绕包装调用记录，包含区间内设备工作，不包含输入从主机搬到设备的完整服务过程。比较时必须让两条路径使用相同范围，否则数字没有共同含义。

设备操作通常异步。若只用主机时钟包住一次 launch，可能测到提交开销，而不是完成时间；如果把一次同步放在整个大循环外，又可能测到批量吞吐而非单次延迟。两种测量都可能有用，但要用正确名称。现有 benchmark 在每个结束 event 等待完成，适合观察有限重复的区间分布，仍没有控制共享设备竞争。

保存原始样本、shape、dtype、误差、版本和设备信息，再给出中位数、最小值、最大值。下面用合成数据练习汇总；这些不是任何 GPU 的实测。

仓库 `labs.kernels.benchmark` 的每组结果中，`samples` 是样本数量，`samples_ms` 是逐次测量的毫秒数组。保存整份 JSON，才能重新计算分位数、观察偶发尖峰；只有中位数无法还原原始分布。脚本已经提供这些输出字段，但本次编写环境没有执行其 GPU 计时路径。

```python
import statistics

def summarize_ms(samples):
    if not samples or any(not np.isfinite(x) or x < 0 for x in samples):
        raise ValueError("finite nonnegative samples required")
    return {"count": len(samples), "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples)}

synthetic = [0.11, 0.10, 0.12, 0.10, 1.20]
summary = summarize_ms(synthetic)
assert summary["median_ms"] == 0.11
assert summary["max_ms"] == 1.20
assert statistics.mean(synthetic) > summary["median_ms"]
```

一个离群点不应该被悄悄删除。它可能来自编译、其他进程、调频或系统干扰，也可能来自算法某个分支。先保留样本，再记录是否能解释；必要时重新做有界实验。不要因为想得到漂亮曲线就选择性报告最小值，也不要把所有波动都归因于框架坏了。

## 第十二课：H100 实验的每一次升级都要有明确问题

你可以自由使用服务器 CPU 和 Docker，因此源码、测试、数值参考、调度模拟都可以在那里长期推进。真正需要 H100 时就使用：验证 Hopper 专用 WGMMA/TMA、较大模型容量或多 GPU 通信，都是合理问题。关键是让规模与问题一致，避免一个小型地址修复也启动八卡长任务。

单卡实验先根据分配结果设置可见设备。例如分配的是某一物理编号，就把那个编号写入当前进程环境，再运行探测确认名称和 capability；下面不替你猜一个“空闲编号”。只看瞬时利用率低也不能推断该卡没有被安排后续公司任务，应遵守已有资源分配方式。

实验卡可以直接写成如下结构：问题是“这份固定 CUTLASS Hopper 路径在指定小 shape 上是否数值正确”；设备为已分配的一张 H100；版本为实际查询值；输入规模有上限；预热和重复次数有上限；终止条件为完成结果或出现错误；输出为日志、误差和源码身份。先有这张卡，再开始构建或运行，就不会在过程中无意扩大范围。

两卡实验适合验证 AllReduce、点对点传输或 tensor parallel 合同。八卡实验应当回答扩展效率、拓扑或更大并行布局的问题。不要把八张显存容量简单相加，当作任意单卡程序可用的统一地址空间；模型必须通过分片、通信和相应运行时使用这些资源。K18 与 S10 已有从 CPU 通信模型到显式多进程实验的连续路径。

Docker 隔离用户空间依赖，但不能凭空提供宿主机没有的 GPU 驱动能力。容器内工具链、宿主驱动、可见设备与框架二进制仍要兼容。选择已有认可镜像并记录身份，比每次启动时临时下载最新版更容易复现；也不要为解决一个权限错误就添加 privileged 或挂载整个宿主文件系统。

结束时只清理自己创建的进程、容器和临时输出。CUDA 内存通常随相关进程退出释放，但持久服务、子进程和异步任务可能仍在运行，因此要核对自己的实验生命周期。不要使用终止所有 Python 或清空全部容器的命令。这里的目的不是增加繁琐流程，而是让实验可以重复，也不破坏同一服务器上的其他工作。

## 第十三课：三个设备问题的完整诊断过程

故障一：WSL 可以运行 `nvidia-smi`，但 `nvcc --version` 提示不存在。已知证据是驱动接口至少可被查询、当前 PATH 找不到 nvcc；未知的是 Toolkit 是否安装在别处，以及你的任务是否需要编译。先运行课程环境探测，检查当前 shell 与解释器；若只是运行预构建框架，测试其 CUDA 可用性；若确实编译 `.cu`，按 WSL 官方路线准备 Toolkit。重装 Windows 驱动不是从现有证据直接推出的修复。

故障二：H100 上正常的配置搬到 5070 Ti 后报告 shared memory 资源不足。先打印或查看编译器报告的每 block 静态与动态资源，再用第四课公式逐项核算 stage 与 epilogue。若理论峰值一百一十二 KiB 已超过该目标的每 block 上限，减少矩阵尺寸但不改 tile 可能完全无效。应该选择较小 tile、较少 stage，或在证明生命周期允许后复用缓冲，再重新验证正确性。

用同一算例比较三种设计，避免只口头说“减小 block”。

```python
choices = {
    "large_three_stage": tile_storage(128, 128, 32, 3),
    "smaller_tile": tile_storage(64, 128, 32, 2, extra_bytes=2048),
    "proven_nonoverlap_only": tile_storage(128, 128, 32, 3, epilogue_overlaps=False),
}
fits_teaching_block_limit = {name: value["peak_bytes"] <= 99 * 1024
                            for name, value in choices.items()}
assert fits_teaching_block_limit == {
    "large_three_stage": False,
    "smaller_tile": True,
    "proven_nonoverlap_only": True,
}
```

这只是容量筛选，不能证明两个 True 配置真的合法。真实实现还可能有额外 padding、barrier 存储和 opt-in 要求；第三种尤其需要同步证明。修复报告应写出改变了哪项资源及相应风险，不要把“能启动了”当作数值正确或更快。

故障三：异步搬运程序偶尔读到上一块数据，或者一直等不到 ready。先把输入 tile 设置成容易区分的模式，例如第零块全一、第一块全二，记录 stage、generation、预期字节、完成字节与消费者状态。若错误随 stage 复用出现，检查代际与覆盖时机；若从第一块就等待，检查参与计数、传输大小和是否真的提交。

```python
stale_case = TransferStage(8, 1, 16)
stale_case.arrive(0)
try:
    stale_case.complete(7, 16)
except ValueError as exc:
    assert "stale" in str(exc)
else:
    raise AssertionError("stale completion accepted")
assert not stale_case.readable
stale_case.complete(8, 16)
assert stale_case.readable
```

这个反例在 CPU 模型里确定地重现旧代际完成错误；真实 GPU 上需要相应工具和日志支持，不能通过加一条全局同步后错误消失，就断言已经理解了根因。全局同步可能暂时掩盖多个不同竞争。正确修复应建立所需的最小依赖，并有能再次触发错误路径的回归案例。

### 把一次排错变成可以交给别人的最小报告

一份有效报告不需要粘贴整台服务器的配置，也不应只有截图上的最后一行错误。先写最小触发命令和源码提交，再列出输入 shape、dtype、stride、编译目标与实际设备能力。随后区分失败阶段：编译器尚未生成设备代码、运行时无法加载代码、kernel 启动失败、结果比较失败，或仅性能不符合预期。阶段不同，需要补充的证据也不同。

编译失败时保存第一条有意义的诊断及相关模板实例，而不是只保留最后的构建失败汇总；加载失败时记录产物包含的目标与运行设备；数值失败时保存最小输入、期望值、实际值和误差位置；同步问题保留 stage 与代际轨迹。若为了缩小问题临时关闭某项优化，说明关闭后是否恢复，以及还没有证明什么。

最后写一项下一步实验，例如仅改变一个 tile 配置、仅切换普通搬运基线，或仅增加一个有依据的完成等待。不要一次换驱动、换框架、换精度又换矩阵尺寸；即使这样偶然成功，也无法知道是哪项改变解决了问题。可解释的排错过程本身就是工程能力，后续遇到新架构时仍然可以使用。

## 第十四课：练习、参考答案与自己的硬件档案

### 题一：解释六个字段而不混用版本

给出产品名、CC、驱动、Toolkit、框架版本、框架 CUDA 构建版本。参考答案：产品名识别设备系列；CC 是编程能力类别；驱动负责设备访问及相关兼容；Toolkit 包含编译等开发工具；框架版本识别软件接口与实现；框架 CUDA 构建版本说明其构建依赖。三种软件版本不要求字符串完全相同，是否兼容要依据支持矩阵与实际验证。任何未知字段保留未知，不从另一字段猜测。

### 题二：为什么不能用 CC 大小替代指令支持表

答案：一般能力编号提供类别信息，但架构加速路径有明确目标和兼容例外。`compute_90a` 不是任意更新架构通用代码；SM100 的某条 tcgen05 路径也不能因十二大于十就套给 SM120。第三课分派器采用显式已验证集合，未知目标要求继续检查。生产 dispatch 还要加入 dtype、布局、软件版本和资源条件。

### 题三：计算四 stage 的输入存储与同时存活输出缓冲

取 BM=BN=128、BK=32、输入两字节、输出四字节。每 stage 输入十六 KiB，四 stage 为六十四 KiB；输出缓冲六十四 KiB，若同时存活，总计一百二十八 KiB，还没加入额外元数据。这个值已经超过九十九 KiB 的每 block 上限。可运行 `tile_storage(128,128,32,4)` 核对，不能拿每 SM 的一百二十八 KiB 取代每 block 限制。

### 题四：把 stage 从三减到二，一定更快吗

答案：不一定。资源占用减少可能增加驻留，也可能减少预取深度、暴露搬运等待。先计算容量是否从非法变合法，再看 occupancy 哪个约束生效，最后对相同 shape、dtype 与计时范围实测。第十课的理想流水线也不能决定真实最优 stage，因为它没有缓存竞争、指令开销和实际占用模型。

### 题五：二维描述器的 stride 单位错了会怎样

答案：FP32 行 stride 为六元素时，字节 stride 应为二十四。若接口需要字节而传六，下一行地址偏移缩小四倍，会读取错误位置，且可能违反对齐约束。第六课的 CPU TeachingTensorMap 统一用元素单位；转换到真实 API 前必须明确乘元素字节数，并检查接口的维度顺序。用非方阵和转置视图可以更容易发现错误。

### 题六：传输已完成，是否可以立刻覆盖 shared memory

答案：还要确认所有消费者不再读取旧数据。TMA 完成解决生产到消费的依赖，WGMMA 等异步消费者完成解决消费到复用的依赖。第七课 consume 被刻意简化为同步，真实异步实现需要单独的完成阶段。若把两个条件混成一个 ready 位，就可能在计算还没读完时覆盖输入。

### 题七：四个 warp 中留一个生产者，三个消费者能组成 WGMMA 吗

答案：不能把三个消费者直接当完整 warpgroup。相关协作指令要求完整一百二十八线程参与，并遵守一致控制条件。正确设计应按实际 kernel 的角色和参与要求分配线程，例如独立生产与消费组；边界数据通过合法掩码或填充处理，不能随意让部分线程跳过集体指令。第八课集合检查提供完整可运行反例。

### 题八：block scale 的结果为什么是九点五而不是十五

答案：前一组代码点积四、scale 乘积一，贡献四；后一组代码点积十一、scale 乘积二分之一，贡献五点五。统一使用第一组 scale 才会得到十五。说明 scale 必须按 K 组在乘加内部对应应用，不能无条件移到整个归约之外。真实低精度实现还要处理编码与硬件 scale 布局，CPU 实数参考只证明数学关系。

### 题九：怎样证明转置描述器没有悄悄改数据

答案：对比逻辑访问与 NumPy 转置视图，并检查原始存储保持不变。下面给出完整核验，使用全部坐标而不是只看一个角落。

```python
storage_before = storage.copy()
logical_transpose = transposed.tile(storage, (0, 0), (6, 4))
np.testing.assert_array_equal(logical_transpose, storage.reshape(4, 6).T)
np.testing.assert_array_equal(storage, storage_before)
```

这验证教学地址映射与只读行为，不证明实际 TMA 描述符已编码正确。真实设备实验仍需检查硬件允许的 stride、对齐和边界，再用同样的逻辑参考比较数值。

### 题十：首次调用一秒、之后十次各零点几毫秒，该报告哪个

答案：两者都记录但名称不同。首次调用主机时间可能含编译和初始化，稳态 event 区间反映另一个范围。若用户关心首次请求延迟，就不能把首轮丢掉；若比较 kernel 稳态，则预热后单独报告。不能把 CPU launch 提交时间称作 GPU 完成时间，也不能把 event kernel 区间称作整个服务端到端延迟。

### 题十一：把所有八张 H100 显存相加，能否让普通单卡模型直接使用

答案：不能。需要模型或运行时明确分片，分配到不同设备，建立通信与同步；单个设备的普通分配不会自动跨八张卡。多卡还涉及拓扑、collective 匹配与每卡临时峰值。学习时先用 S10/K18 的小规模通信合同证明数据分布和结果正确，再扩大模型与卡数，而不是只看总容量数字。

### 题十二：为自己的两台环境各写一张完整实验卡

参考卡必须填写实际查询结果：环境位置用“个人 WSL”或“服务器学习环境”等非敏感标签；设备产品和 CC；Python、框架、驱动与 Toolkit；源码提交；测试命令；shape/dtype/stride；正确性误差；计时范围；样本摘要；尚未执行项目。公司主机名、内部路径、设备调度凭据和其他人的进程信息不需要进入公开仓库。

如果个人 GPU 的普通 Triton 通过，而 H100 专用路径尚未编译，卡片应分别写“普通算子通过”和“Hopper 路径未验证”。若使用 CPU 模拟理解 SM100 TMEM，则写“数学与生命周期模型完成，SM100 设备验证未执行”。精确描述证据能帮助你安排下一步，也避免几个月后把一次模拟结果误当成硬件经验。

## 掌握门槛与下一步

面对一个“不支持此 GPU”的错误，你应该能依次区分设备能力、代码目标、工具链、后端支持、输入布局和资源要求；面对一个“偶尔数值错”的错误，你应该能追踪地址、参与线程、代际和读写完成，而不是只改精度或加同步；面对一个“没有预期快”的结果，你应该能核对计时范围和瓶颈假设。

下一章 [K17 角色分工、持久执行与大型融合](17-warp-persistent-megakernel.md) 将本章的资源、代际和完成协议组合成完整流水线，并沿固定 CUTLASS 源码观察真正的生产者与消费者。完成后再把自己的数值算子接入 K18 项目，让硬件知识服务于可运行、可测试、可解释的实现。
