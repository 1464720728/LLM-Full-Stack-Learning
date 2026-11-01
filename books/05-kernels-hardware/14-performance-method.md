# K14　性能工程：从一个可复现的时间到一个能解释的优化

先修是本卷的线程、内存、分块矩阵乘法与数值误差；不要求你已经读过性能分析器。这里的“性能”指完成规定工作所需的资源与时间，不等于显卡利用率高，也不等于某次运行的数字小。学完这一章，你应当能回答：这个数字量的是什么，为什么这个算子慢，改动是否真的有效，在哪些输入上应该回退，以及整个应用究竟受益多少。正文先在 CPU 上完成推导、统计和故障实验，再给出有明确入口的单卡实测路线。所有写着“合成”的数据都是教学构造，不是对你的 RTX 5070 Ti 或 H100 的测量。

## 小课一：把“快一点”改写成可以检验的问题

假设你写了一个向量加法，运行后打印零点零一毫秒。这个数字本身几乎没有信息。向量有多少个元素？元素是四字节浮点数还是两字节浮点数？输入原本就在显存，还是每次都从主机复制？输出需要传回主机吗？程序第一次运行是否现场编译？这些问题改变的是被测工作，不能等出现奇怪结果才补写。我们先把一次实验写成一份契约：输入形状、输入类型、布局与步幅、要求的输出、允许的误差、计时开始点、计时结束点、运行环境和重复方法。

“延迟”是一个任务从指定开始点到指定结束点经过的时间。“吞吐”是单位时间完成的工作量。例如一次批处理完成一百个请求，共用五十毫秒，批处理吞吐是每秒两千个请求；这不意味着每个请求只等待半毫秒。如果请求必须等整批结束才能拿到答案，每个请求的执行等待可能接近五十毫秒，另外还有组成这一批之前的排队。平均吞吐的倒数不是任意请求的实际延迟。这一区别会决定你应不应该通过增加 batch 提高矩阵乘法的利用率。

“基线”是比较对象，通常是同一输入上的可靠库实现；“候选”是你写的优化版本。“加速比”定义为基线时间除以候选时间。两者分别为十毫秒和八毫秒，加速比是一点二五倍，而耗时减少百分之二十。不能把这两种百分比混写成同一个结论。候选时间必须来自与基线一致的语义范围。只测候选 kernel，却把基线的数据拷贝也算进去，相当于让一名选手从起跑线出发，另一名先走到终点附近，结果没有比较价值。

实验中还有“控制变量”：你暂时固定、避免干扰解释的条件。例如同样的形状、随机输入、精度和执行路径。它不是要求现实世界永不变化，而是使一次局部实验能够回答一个明确问题。若同时更换数据类型、Tile、融合方式、输入布局和 batch，即使整体快了，也无法知道哪一个因素贡献多少。真实工程最终可以联合优化，但学习与定位时先做单因素实验，再验证组合效果，推理会可靠得多。

先在纸上写一句可被推翻的假设：“这个向量操作受显存搬运约束；减少一个中间张量的写回和再读取，将使大输入的耗时下降，小输入的收益可能被启动开销遮住。”它给出了原因、改动、预测和例外。与“融合一般比较快”相比，我们已经知道应测哪些形状、统计哪些字节，以及怎样的结果会使我们重新考虑解释。性能工程的进步来自这种可被反证的具体句子。

## 小课二：先算工作量，再看显卡百分比

一个 FLOP 是一次浮点算术操作的计数单位，FLOP/s 是每秒完成的浮点操作数。乘加 `a*b+c` 在常见性能口径中按一次乘法加一次加法计为两个 FLOP，即使硬件用一条 FMA 指令完成。指令条数与 FLOP 数不是同一个量。比较性能前要约定计数规则，尤其是矩阵乘法、稀疏计算和低精度计算；不能把执行了一个稀疏指令的等效稠密工作量与另一侧实际执行的非零工作量混在一起。

对长度为 N 的逐元素加法 `c[i]=a[i]+b[i]`，每个元素执行一次浮点加法，所以 F=N。若三个数组都是 float32，一个元素占四字节，算法至少要读两个输入、写一个输出，理想有效载荷 B=12N 字节。这里的“有效”指根据算法接口计算的数据量；它还不是已经测得的显存总线流量。缓存命中、写事务、内存合并和重复读取都会改变某一层级看到的真实流量。先保留这一区别，后面会用它解释“带宽超过峰值”的假象。

对矩阵乘法 C=AB，令 A 的形状为 M×K，B 的形状为 K×N，C 的形状为 M×N。一个输出元素需要 K 次乘法与 K−1 次加法，严格计数为 MN(2K−1)。大规模性能报告常用近似 F=2MNK，差别是省略每个输出少一次加法的边界项。下面始终使用这个常见近似，同时明确不包含偏置、激活或旧 C 的累加。若运算是 `alpha*A@B+beta*C`，读取旧 C 与额外计算也应计入自己的模型。

设每个元素占 s 字节，并假设每份输入从被研究的内存层级只读一次，输出只写一次，理想数据量为 B=s(MK+KN+MN)。这个式子描述充分复用的理想情况，不是朴素三重循环的逐条 load 总数。实际分块乘法可能让同一输入被多个输出 Tile 读取；这些重复读取可能被缓存吸收，也可能继续到达显存。你应同时保留“算法最低数据量”和“实现实际数据量”两个概念，不要用前者证明某个未测实现已经接近硬件上限。

手算一个小例子：M=2、K=3、N=4，元素为 float32。近似运算量为 2×2×3×4=48 FLOP。输入 A 有六个元素，B 有十二个元素，输出有八个元素，共二十六个元素，所以最低有效载荷是一百零四字节，算术强度约零点四六二 FLOP/byte。这里规模很小，启动成本很可能主导实际时间，但工作量计算仍然正确。模型正确不意味着它已经包含所有成本。

```python
from math import isclose

def matmul_work(m, n, k, item_bytes=4):
    if min(m, n, k, item_bytes) <= 0:
        raise ValueError("positive dimensions and item size required")
    flops = 2 * m * n * k
    payload_bytes = item_bytes * (m*k + k*n + m*n)
    return flops, payload_bytes, flops / payload_bytes

f, b, intensity = matmul_work(2, 4, 3)
assert (f, b) == (48, 104)
assert isclose(intensity, 48/104)
print(f, b, round(intensity, 6))
```

这段代码的参数顺序是 m、n、k，与矩阵 A 的 m×k、B 的 k×n 约定对应。先做正数检查，防止零尺寸导致强度分母为零；对于真实框架允许的空张量，应另外定义“零工作”分支，不要为了画图把它变成无限吞吐。返回值分别是操作数、字节数和两者的比值。断言检查一个能够手算的输入，避免只在大规模随机数上相信公式。输出中的四十八和一百零四应当能够逐项在上段找到来源。

单位也是程序的一部分。一毫秒等于千分之一秒，一微秒等于百万分之一秒；一个 GB 按十的九次方字节，一个 GiB 按二的三十次方字节。若 128 MiB 数据通过 64 GB/s 的路径，时间是 `128*2**20/(64*10**9)` 秒，约二点零九七毫秒。不能直接把一百二十八除以六十四当作毫秒，因为分子分母采用了不同量级和进制。把单位写在变量名或表头里，会减少很多比复杂优化更致命的错误。

## 小课三：从两个资源约束推导 Roofline

现在令 F 为需要完成的 FLOP 数，B 为需要跨越某个指定内存层级的字节数，P 为该计算类型的可用计算吞吐，单位 FLOP/s，W 为该层级的可用带宽，单位 byte/s。仅完成计算至少需要 F/P 秒，仅搬完这些数据至少需要 B/W 秒。两项工作可能部分或充分重叠，因此一个乐观的下界是 T≥max(F/P,B/W)。我们取最大值是因为同一次执行必须同时满足两个约束，不是因为计算与搬运必然完全重叠。

若依赖关系迫使两阶段完全串行，模型时间可能更接近两者之和；若流水线能充分重叠，可能接近最大值；实际还有启动、同步、地址计算和尾部不均衡等成本。下界是一条“不能更快”的约束，不是一份“保证达到”的承诺。看到真实时间比下界大十倍，并不能直接说编译器浪费了九成性能，需要检查模型是否漏掉成本、P 与 W 是否用对、B 是否低估，以及该输入是否足够大。

把两边改写成吞吐形式更容易画图。定义算术强度 I=F/B，单位 FLOP/byte；实际算术吞吐为 R=F/T。由时间下界得到 R≤min(P,W×I)。低强度区域的上界随 I 线性增长，高强度区域最多达到 P。两段相交的位置 I*=P/W，叫转折强度：若任务强度比它小，带宽约束更紧；比它大，计算约束更紧。这不是给算法永远贴上“计算密集”或“访存密集”的标签，同一算法在不同硬件、数据类型与输入形状下可以处于不同区域。

继续使用人为设备 P=10¹² FLOP/s、W=10¹¹ byte/s，不借用任何实际显卡宣传值。转折强度为十 FLOP/byte。任务 F=2×10⁹、B=10⁹，所以 I=2，计算时间为两毫秒，搬运时间为十毫秒，下界为十毫秒。此时把 P 翻倍，计算下界变为一毫秒，总下界仍是十毫秒；把 B 减半，搬运下界降为五毫秒，整体下界才降为五毫秒。这个简单手算告诉我们，优化目标应当对准更紧的约束。

再做一个容易误解的变化：通过额外计算压缩数据，把 F 从二十亿提高到三十亿，同时把 B 从十亿降到四亿。计算下界三毫秒，搬运下界四毫秒，总下界四毫秒。操作数变多却可能变快，因为减少了更昂贵的搬运。这里还没有算压缩元数据、分支与解压代价；真正实现必须补上这些。这个例子解释了为什么“算得少”不是性能改善的充分条件，以及为什么重计算有时能换取更低时间。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Bound:
    compute_ms: float
    memory_ms: float
    lower_ms: float
    intensity: float
    ridge: float

def roofline(flops, byte_count, peak_flops_s, bandwidth_bytes_s):
    if min(flops, byte_count, peak_flops_s, bandwidth_bytes_s) <= 0:
        raise ValueError("roofline inputs must be positive")
    c = flops / peak_flops_s * 1000
    m = byte_count / bandwidth_bytes_s * 1000
    return Bound(c, m, max(c, m), flops/byte_count,
                 peak_flops_s/bandwidth_bytes_s)

base = roofline(2e9, 1e9, 1e12, 1e11)
faster_compute = roofline(2e9, 1e9, 2e12, 1e11)
less_traffic = roofline(3e9, 4e8, 1e12, 1e11)
assert (base.compute_ms, base.memory_ms, base.lower_ms) == (2, 10, 10)
assert faster_compute.lower_ms == 10
assert less_traffic.lower_ms == 4
print(base, less_traffic)
```

冻结的数据类表示一次模型计算的结果，不允许后续随手改某个字段而留下互相矛盾的数字。函数内部先使用秒，再乘一千得到毫秒；名称已经包含单位。两个反事实实验只改变明确写出的参数，最后的断言检查预测。仓库现成入口 `python -m labs.kernels.run --lab 23` 给出的强度也是二，下界也是十毫秒，且输出标明 synthetic。你可以先对照两个实现，再自行改变一项参数，观察判断何时跨过转折点。

P 与 W 应当来自与你的问题一致的证据。使用双精度计算时不能拿低精度 Tensor Core 峰值；没有满足结构化稀疏条件时不能使用稀疏峰值；跨主机传输也不能套用显存内部带宽。更稳妥的方式是同时记录理论上限和同机器、同环境下相关基准达到的可持续值，并把两条界线分别画出。可持续值仍受工作集、并发和时钟影响，所以它是实验条件下的参考，不是永恒硬件常数。

## 小课四：区分有效字节、真实流量和缓存状态

前一课把 B 当作已知量，现实中它常常是最大的不确定项。以逐元素加法为例，B_eff=12N 是根据接口计算的有效字节；若输入刚被另一个算子写入并仍留在缓存中，这些读取未必全部走到显存。如果你用 B_eff 除以很短的热缓存时间，得到超过显存标称带宽的数字，这不必然说明计时器坏了，也不表示突破了物理极限。你测量的可能是缓存供数条件下的有效吞吐，却错误地把它命名成显存带宽。

在另一方向，步幅访问可能让一个内存事务只带回少量有用元素；重复加载可能使同一元素经过多个事务；写回的粒度也不一定恰好等于程序要写的单个元素。此时真实搬运量 B_actual 大于有效字节。有效带宽 B_eff/T 适合比较相同语义的实现，但不足以判断总线是否饱和。若要建立某一层级的实际 Roofline，需要该层级的计数信息，并清楚区分读、写、缓存命中和事务口径。

可以把内存系统看成多个有不同容量与带宽的层级，但不要想象成所有访问都依次付出各层完整代价。数据命中哪一级、请求能否合并、多少请求同时在途，会共同影响延迟与吞吐。高带宽不代表单次依赖加载延迟很低；一条只能等前一条结果才能发起的指针链，即使每次只取很少字节，也可能无法积累足够在途请求来使用整条数据通路。反过来，大量独立访问能够隐藏延迟，却可能最终受总带宽限制。

工作集是一次或一段计算实际反复使用的数据集合。反复测同一对很小的向量，工作集容易保持热状态；每轮换一批远大于缓存的数据，条件就不同。二者都可以是合理实验：前者贴近紧密融合流水线中的局部重复使用，后者更像大规模流式遍历。问题是必须命名条件，不能用热缓存微基准预测一整套需要搬运大量新数据的模型服务。也不要随意写一个清空缓存的大循环，随后宣称它完全模拟真实冷启动；这类干预本身会影响功耗、时钟和其他资源。

下面的数字仅用于检验口径：有效载荷一百二十万字节，时间两微秒，有效带宽是六千亿字节每秒；假设 profiler 在指定层级统计只有三十万字节，真实层级带宽则是一千五百亿字节每秒。两者都由相同时间计算，却在回答不同问题。记录时把分子来源写进字段名，就不会在报告里无意替换概念。

```python
payload = 1_200_000
observed_layer_bytes = 300_000   # 合成的层级流量，不是硬件采样
elapsed_s = 2e-6
rates = {
    "effective_GB_s": payload / elapsed_s / 1e9,
    "layer_GB_s": observed_layer_bytes / elapsed_s / 1e9,
}
assert rates == {"effective_GB_s": 600.0, "layer_GB_s": 150.0}
print(rates)
```

一个重要调试顺序由此形成：发现不合理带宽，先核对毫秒与秒，再核对 N 与数组大小，再核对计时是否同步，最后检查缓存与流量定义。不要一开始就修改 kernel。输入规模读错四倍、把毫秒直接当秒、把一次读写误算为三次，都会制造远大于真实优化收益的误差。性能模型与程序一样需要单元级的数量校验。

## 小课五：异步执行为什么让一个正确时钟给出错误结论

Python 调用 GPU 运算时，主机经常只负责把工作提交到队列，然后继续运行。假设提交耗时十微秒，GPU 实际计算一毫秒。在调用前后使用主机时钟而不等待完成，可能只得到十微秒。这不是时钟不准确，而是开始点与结束点包围了提交行为，没有包围任务完成。先理解这一点，比死记“计时要同步”更有用，因为多 Stream、事件依赖和数据拷贝会让“同步在哪里”成为实质问题。

主机墙钟计时若希望测一段 GPU 工作的完成延迟，常见做法是在开始前排除既有工作，在结束前等待本次工作完成。这样得到的时间包含主机调用、相关等待与 GPU 工作；它仍未必包含请求进入队列前的等待、网络与后处理。设备事件则在指定设备 Stream 的执行顺序中标记时间点，用开始事件到结束事件之间的设备时间差来描述区间。两种计时器解决不同问题，不能把一个称作“真实时间”而否定另一个。

举一个离散事件模拟。主机在零微秒提交开始事件，五微秒提交 kernel，十微秒提交结束事件，随后继续做别的事。GPU 在一百微秒才轮到这段队列，kernel 执行到一千一百微秒。事件间隔约一千微秒；围绕整段提交的主机差约十微秒；如果主机从零等到所有工作结束，墙钟约一千一百微秒。已有队列等待、主机提交时间与设备执行时间解释了三个数字为什么不同。真实事件可能还有调度与粒度影响，这个模型只说明因果关系。

第一次调用又有额外状态变化：导入库、初始化上下文、加载设备代码、JIT 编译、分配器建立缓存、自动选择算法，都可能出现在冷启动路径。稳定重复调用可能跳过这些步骤。若应用是一次性脚本，冷启动就是用户成本；若是持续运行的服务，稳定态与偶发新形状编译也都重要。正确做法是分别报告，而不是从样本里偷偷删掉首次高值，也不是拿第一次调用代表所有后续请求。

极短 kernel 还会遇到测量装置的相对开销。可以在同一计时区间中重复执行若干次，再除以次数，以降低计时粒度与主机发射间隔的相对影响；但这种做法改变了缓存、分配器和队列状态，也可能把“有依赖的一连串执行”误写成“独立调用”。当输出会反馈到下一次输入时，重复方式甚至改变算法。要同时说明重复次数、是否复用输入、是否保留输出以及是否存在跨次依赖，不能只给除完后的漂亮数字。

```python
# 离散时间模拟：不使用 GPU，也不调用任何真实计时器。
host_before_us = 0
host_after_submit_us = 10
gpu_start_us = 100
gpu_finish_us = 1100
naive_host_us = host_after_submit_us - host_before_us
event_interval_us = gpu_finish_us - gpu_start_us
completion_wall_us = gpu_finish_us - host_before_us
assert (naive_host_us, event_interval_us, completion_wall_us) == (10, 1000, 1100)
print("submit/event/completion:", naive_host_us, event_interval_us, completion_wall_us)
```

这个故障实验并非模拟完整 CUDA 调度器，它只把时间点显式化，让错误计时范围变得可见。实际使用多 Stream 时，结束事件只证明它所在 Stream 的顺序约束已经满足；没有建立依赖的其他 Stream 工作不会自动被纳入区间。初学阶段先在单 Stream 上验证口径，等你能画出依赖关系，再测并行流水线。把测量范围写清，比把多个互不相同的时间强行汇总成一个“耗时”更重要。

## 小课六：逐行读懂仓库的基准，而不是只复制运行命令

现在打开 [benchmark.py](../../labs/kernels/benchmark.py)。它提供 add 与 matmul 两个受限入口，默认不启动 GPU，必须显式传入 `--run`。先在任何只有 CPU 的环境执行下面三条命令：环境检查列出解释器、包和命令是否存在；默认基准打印未运行提示；实验二十三只计算合成 Roofline。三者都不会安装包、下载模型或占用服务器 GPU。

```bash
python -m labs.kernels.inspect_environment
python -m labs.kernels.benchmark --op add --size 64 --repeats 5
python -m labs.kernels.run --lab 23
```

命令行的 `--size` 不同于统一的元素数：add 分支实际生成长度 `size*256` 的向量，所以六十四对应一万六千三百八十四个元素；matmul 分支生成两个 `size×size` 方阵。当前入口把 size 限制在一到二千零四十八，把 repeats 限制在一到一百。范围检查在依赖检查之前进行，因此错误参数即使在无 GPU 环境也会被拒绝。实验规模受限的目的是让第一次实测可控，不是声称这些形状代表全部 LLM 工作负载。

随后程序检查 torch 与 triton 是否能找到，再检查 CUDA 是否可用，缺少条件就清楚打印 SKIP。导入实际 Triton 算子发生在这些检查之后。随机种子设为四，使输入生成过程可复现；它并不能固定硬件调度与计时时间。add 的候选调用 `ops.add(a,b)`，基线调用 `a+b`。matmul 的候选调用 `ops.matmul(a,b)`，基线调用 `a@b`。输入先生成在设备上，因此后面的测量不含主机到设备的数据上传。

接着用主机时钟包住候选的第一次调用，再执行设备同步，把这个冷路径时间写入 `first_call_wall_s`。它可能包括编译、输出分配、提交和完成等待。注意单位是秒，后面的事件时间是毫秒。程序随后计算参考输出，用 `torch.testing.assert_close` 检查相对容差 `1e-4` 与绝对容差 `1e-3`，再提取最大绝对误差。只有这一步通过才会进入稳定态测量；错误输出不会因为速度快而继续被当作候选成绩。

预热循环让候选与基线各运行三次，然后同步。三次是这个简洁实验入口选的固定值，不是保证所有系统都稳定的普适常数。某些形状可能触发额外编译，某些库可能需要更复杂的热身，温度与时钟也可能继续变化。实验报告必须如实记录三次，后续若修改预热策略，应保存修改后的代码版本，不要口头把它描述成“充分预热”而没有可检查的条件。

内部 `measure(call)` 对每次重复创建开始与结束事件，在两者之间调用包装函数，等待结束事件完成，再读取毫秒差。每次的输出赋给局部变量 `out`，保证调用确实被执行，测量时没有把返回值转到 CPU。最后返回中位数、最小值、最大值、样本数量 `samples`，以及保留每次事件毫秒差的列表 `samples_ms`。注意两个名字只差后缀，却分别是计数和数据；对前者计算分位数没有意义。现在可以用原始列表计算四分位数或画分布，但不能凭两组分别采集的列表声称已经进行了交错配对实验。下课先用明确列出的合成样本验证分析程序，再说明怎样读取真实 `samples_ms`；合成数据不能被误标为设备实测。

这个区间包住了会分配输出的包装函数，并包含事件之间实际排队的设备工作；它不是只对某条内部 kernel 指令序列计时，也不是完整的主机墙钟。输入上传不在区间内，CPU 侧分配逻辑的全部开销也不能简单与事件差画等号。源码的 `scope` 字段已经写出范围，报告应保留它。两组测量先完整测候选，再完整测基线，未做交错配对；若中途发生温度漂移或外部负载变化，这种顺序可能影响结论。

在 GPU 可用且已选择一个空闲设备后，才执行下面的单卡命令。这里的设备零指 `CUDA_VISIBLE_DEVICES` 选出的可见设备；在 Linux shell 中把它替换成分配给你的卡号。不要因为服务器有八卡就同时启动八份基准。Windows 原生 shell 的环境变量语法不同，建议沿用本卷前面已经建立的 WSL 或 Linux 环境。下面的 GPU 命令在本次交付环境没有执行。

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 64 --repeats 20
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op matmul --size 127 --repeats 20
```

候选矩阵乘法的源码使用 `input_precision='ieee'`，但基线的实际计算路径仍取决于安装的 PyTorch、设备与精度配置。不能仅凭输入都是 float32 就宣称两侧底层乘法精度完全相同。第一次比较应记录全局精度设置、检查误差，再决定是否建立另一个允许较低乘法精度的实验组。不同精度合同可以分别比较，不应悄悄混进同一张“同精度加速”表。

## 小课七：一个中位数不够，先理解样本如何产生

假设基线的五次时间是九、十、十、十一、五十毫秒，候选是八、八、九、九、十毫秒。基线中位数十，候选中位数九，加速比约一点一一；若比较平均数，基线十八、候选八点八，加速比约二点零五。差别来自那次五十毫秒。它可能是共享 GPU 干扰，也可能是算法真实偶发慢路径，不能因为影响结论就删除。先保留原始样本，再依据运行日志与实验目标决定如何分类。

中位数是排序后居中的值，适合描述典型水平，受单个极端值影响较小；它不能替代尾延迟。分位数表示按某个约定，有多少比例的样本位于该位置之前。不同软件对有限样本的插值约定不同，因此小样本计算时应写明方法。下面采用相邻顺序统计量之间的线性插值：对排序后的 n 个值，位置为 q(n−1)，取左右整数位置的值做线性组合。五个样本的百分之九十五分位数不具备高精度尾部保证，数字保留很多小数也不能创造证据。

离散程度也重要。四分位距是第三四分位数减第一四分位数；MAD 是样本到中位数的绝对距离的中位数。它们用于描述分布变化，不会自动证明两个实现的优劣具有统计显著性。某个候选的中位数快百分之一，但样本波动百分之二十，应该先解决测量不稳定，而不是把一次小差异当作确定收益。硬件计时常有相关性，连续样本并非总是独立抽样，机械套用统计公式可能给出过度乐观的区间。

```python
import statistics
import math

def quantile_linear(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("nonempty values and q in [0, 1] required")
    data = sorted(float(v) for v in values)
    if any(not math.isfinite(v) or v <= 0 for v in data):
        raise ValueError("timings must be finite and positive")
    position = (len(data)-1)*q
    left = math.floor(position)
    right = math.ceil(position)
    weight = position-left
    return data[left]*(1-weight) + data[right]*weight

def timing_summary(values):
    median = quantile_linear(values, .5)
    return {
        "n": len(values), "median_ms": median,
        "p95_ms": quantile_linear(values, .95),
        "iqr_ms": quantile_linear(values, .75)-quantile_linear(values, .25),
        "mad_ms": statistics.median(abs(v-median) for v in values),
    }

baseline_ms = [9, 10, 10, 11, 50]  # 合成
candidate_ms = [8, 8, 9, 9, 10]    # 合成
s0, s1 = timing_summary(baseline_ms), timing_summary(candidate_ms)
assert s0["median_ms"] == 10 and s0["iqr_ms"] == 1
assert s1["median_ms"] == 9
assert math.isclose(s0["p95_ms"], 42.2)
print(s0, s1, "median speedup", s0["median_ms"]/s1["median_ms"])
```

实际基准 JSON 的 `triton` 与 `torch_baseline` 各包含一份测量对象。下面的适配函数直接读取当前接口，不把 `samples` 这个整数误当成时间数组。为了保证本章离线可运行，示例对象仍由前面的合成样本构成；将它换成真实 JSON 对象中的对应子字段，计算流程保持不变。校验样本数量与摘要中位数，能够提前发现文件拼接、字段错配或手工编辑造成的不一致。这里没有重新测量，也没有通过统计处理消除共享 GPU 干扰。

```python
def summarize_measure(measure):
    raw = measure["samples_ms"]
    if len(raw) != measure["samples"]:
        raise ValueError("sample count mismatch")
    result = timing_summary(raw)
    if not math.isclose(result["median_ms"], measure["median_ms"],
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("median does not match raw samples")
    return result

synthetic_measure = {
    "samples": len(baseline_ms), "samples_ms": baseline_ms,
    "median_ms": statistics.median(baseline_ms),
    "min_ms": min(baseline_ms), "max_ms": max(baseline_ms),
}
assert summarize_measure(synthetic_measure) == s0
try:
    summarize_measure(dict(synthetic_measure, samples=999))
except ValueError as exc:
    assert str(exc) == "sample count mismatch"
else:
    raise AssertionError("mismatched count was accepted")
print("raw-sample adapter and count fault passed")
```

逐行看：函数先拒绝空集合与非法分位数，再把值排序并检查正数、有限性；这一步阻止零时间、无穷大与 NaN 悄悄进入比值。位置三点八落在第四个值十一与第五个值五十之间，按零点二与零点八加权，得到四十二点二。摘要保留样本数量，因为五次与五千次的尾部认识能力不同。MAD 对基线为一，能够反映典型偏离，但那次五十毫秒仍必须作为原始样本保留，不能被摘要抹去。

运行顺序是另一个变量。先连续测完 A 再测 B，如果机器逐渐升温，后者可能吃亏；如果频率逐渐上升，后者可能占便宜。交错 AB、BA 或随机化顺序有助于减少这种系统偏差，但不会自动消除邻近实验的缓存影响。配对比较应尽量让一对 A、B 使用相同输入并靠近执行，同时记录先后顺序。对于需要独立冷缓存的实验，还必须设计一致的状态准备；不存在一个适用于所有研究问题的万能测量排列。

所谓“重复很多次”也有两层。内层重复是在同一次进程、同一输入条件下执行；外层重复是重新启动进程或在不同时间窗口重新执行。前者观察短时波动，后者可以暴露冷状态、调度、温度与外部负载差异。若你宣称某优化在日常服务中稳定有益，只有一次进程内的一千轮热缓存结果通常不足。学习项目可以先完成有限次数的内层测量，再用少量外层复核，不必为追求统计外观长时间占用公司 GPU。

## 小课八：局部加速怎样传递到整个模型

假设一次请求总时间为一百毫秒，待优化算子占十毫秒，其余九十毫秒保持不变。算子加速两倍变成五毫秒，总时间九十五毫秒，端到端加速为一百除以九十五，约一点零五三倍。即使算子无限快，总时间也至少九十毫秒，最大加速约一点一一一倍。这个推导称为 Amdahl 约束：若原始占比为 p，局部加速比为 s，总加速比为 `1/((1-p)+p/s)`。

式中的 p 必须来自与你的端到端时间范围一致的占比。若 kernel 与拷贝重叠，不能把所有 kernel 时间简单相加后除以请求时间，得到一个超过百分之百的占比，再代入串行模型。此时要分析关键路径：只有缩短决定完成时刻的路径才会直接缩短延迟。一个与其他工作充分重叠的算子即使自身快了一半，可能只增加空闲时间而不改变整体完成点。图上每段都变快与应用更快之间存在依赖关系。

吞吐优化也会改变排队。若请求到达速度接近服务能力，小幅提高处理能力可能显著减少排队；反过来，增加 batch 虽提高每秒 token 数，却可能让低负载单请求等待凑批。LLM 场景至少分清首次 token 延迟、后续 token 间隔、每请求总时间和总吞吐。prefill 常处理较多 token，矩阵乘法能复用权重；decode 每步新增少量 token，权重和 KV 数据路径可能更突出。不能只用一个平均算术强度覆盖所有阶段。

MoE 带来另一个变量：token 分到多个专家后，每个专家的矩阵规模可能很小且不均衡。总 token 数没有减少，但单个专家内部复用下降，还可能增加路由、重排和通信。优化一个大稠密 GEMM 的 Tile，不保证对许多小专家同样有效。先根据实际专家计数构造形状分布，再评估 grouped 或融合调度，才能判断收益来自数学计算还是减少了调度与重排成本。

```python
def amdahl(fraction, local_speedup):
    if (not math.isfinite(fraction) or not math.isfinite(local_speedup)
        or not 0 <= fraction <= 1 or local_speedup <= 0):
        raise ValueError("invalid fraction or speedup")
    return 1 / ((1-fraction) + fraction/local_speedup)

assert math.isclose(amdahl(.1, 2), 1/.95)
assert math.isclose(amdahl(.8, 2), 1/.6)
# 合成关键路径：计算与传输同时开始，二者均完成后才能返回。
original_finish_ms = max(8, 10)
compute_optimized_finish_ms = max(4, 10)
assert original_finish_ms == compute_optimized_finish_ms == 10
print(amdahl(.1, 2), "overlapped finish", compute_optimized_finish_ms)
```

代码中的最大值例子只有两个能够完全并行的阶段，不能代替真实调度器。它的作用是提供一个反例：局部速度提高不一定改善端到端延迟。你应在自己的应用里记录阶段依赖、重叠与输入规模，然后再决定优先优化哪一部分。若序列化、Python 调度和数据整理占主要时间，继续微调一个已经很快的设备算子可能只是让最显眼的数字更漂亮。

## 小课九：先读时间线，再决定深入哪个 Kernel

时间线把事件放在同一时间坐标上，回答“谁在什么时候做了什么”。初读时不要马上打开所有计数器。先定位一个完整请求或一次迭代，标记输入准备、设备拷贝、计算、输出处理及同步位置，再观察设备是否连续工作。长空隙可能来自 CPU 尚未准备数据、同步等待、首次编译或其他进程竞争。一个持续时间很短但调用上万次的 kernel，也可能比单次最长的 kernel 更值得关注。

NVIDIA Nsight Systems 的官方用户指南说明了进程 CUDA 活动的时间线以及聚焦采集范围的方法；本章用它观察提交、拷贝、kernel 与空隙。Nsight Compute 的官方指南说明了按集合或章节采集 kernel 指标以及 replay 的成本；本章用它研究已经选定的候选瓶颈。两者的具体界面和指标随版本变化，所以先运行本机帮助与版本命令，再对照文档。不要将截图里的颜色、菜单位置或某个版本的指标名字记成永久接口。

下面给出 Linux 上的受限采集路线：先检查工具存在及帮助，随后只对本课程的短小进程执行一次时间线采集。输出到自己的目录，输入尺寸与重复数保持小。该命令不会授权你采集其他人的作业，更不需要改成全系统跟踪。采集的第一次调用包含编译与初始化，在时间线里应与后续稳定态调用分开观察。当前交付环境没有这些工具和 GPU，因此下面只作为待实测命令交付。

```bash
nsys --version
nsys profile --help
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx --sample=none --output=k14-add-profile python -m labs.kernels.benchmark --run --op add --size 64 --repeats 5
ncu --version
ncu --list-sets
ncu --list-sections
```

想进一步使用 ncu 时，先从已验证的 kernel 名称和一次稳定态调用确定筛选条件，再根据本机 `ncu --help` 设置目标与次数上限；不要对整个长训练直接使用最全面采集。不同指标可能需要多轮重放，重放会改变并发与状态，采样本身也会增加成本。因此普通计时与 profiler 下的执行时间分别保存，不拿后者直接覆盖前者。若性能计数器访问受限制，就保留无法采集的事实，使用可获得的时间线与可控实验继续定位；课程不要求改服务器全局权限。

手工练一个时间线故障。一次迭代从零到二十毫秒，CPU 准备占零到十二，拷贝占十二到十五，设备计算占十五到十九，结果处理占十九到二十。kernel 即使从四毫秒缩短到两毫秒，总时间也只从二十到十八。相反，如果能让下一批的 CPU 准备与本批计算重叠，稳态吞吐的改善可能更大，但这需要真正独立的数据缓冲和依赖管理。先写出事件边界，再讨论流水线，避免把不存在的重叠写进收益预测。

另一个典型现象是每个 kernel 前都有小空隙。可能是 Python 逐个提交小操作，可能是隐式读取标量导致同步，也可能是每轮重新构造张量。定位步骤是先找空隙对应的主机调用，再做一次局部删除或替换实验，观察空隙是否随之消失。只有看到相关因果变化，才能说“启动或主机调度是主要原因”。单看低 GPU 利用率不能区分缺少工作、等待依赖、工作过小或指标采样粒度过粗。

## 小课十：用资源与指标解释瓶颈，但不崇拜一个百分比

Occupancy 通常描述某个 SM 上活跃驻留 Warp 数相对于可支持上限的比例。驻留表示资源已经分配、能够参与调度，不表示每个周期都在执行有效指令。许多 Warp 同时等待同一个长依赖，occupancy 可以很高，吞吐仍很低。反过来，一个复用充分、寄存器中保留大量中间结果的 kernel，occupancy 较低也可能很快。它是一条线索，不是需要单调最大化的最终目标。

驻留受多种资源同时限制：每个 block 的线程数、每线程寄存器需求、每 block 的 Shared Memory，以及硬件对 block 和 Warp 数的上限。用一个明确标为合成的 SM 练习：寄存器池六万五千五百三十六个，Shared Memory 容量六万五千五百三十六字节，最多驻留六十四个 Warp；每个 block 二百五十六线程即八个 Warp，每线程三十二寄存器，每 block 三万二千七百六十八字节共享内存。寄存器允许八个 block，共享内存只允许两个，所以主要上限是两个 block，也就是十六个 Warp，占比四分之一。真实分配还有粒度与架构限制，不能把这个简化除法当成设备计算器。

若把共享内存减到一万六千三百八十四字节，允许四个 block，占比可能提高到二分之一。但减少共享内存可能迫使更小 Tile，使全局内存重复读取增加。若为了提高 occupancy 把寄存器压得过低，编译器可能发生 spill，把部分原本应在寄存器中的值放到其他存储路径，引入额外访问。你需要同时检查资源报告、运行时间与访存线索，不能只庆祝 occupancy 数字上升。

```python
def toy_residency(threads=256, registers_per_thread=32,
                  shared_per_block=32768, registers_per_sm=65536,
                  shared_per_sm=65536, max_warps=64, max_blocks=16):
    if threads <= 0 or threads % 32 or min(registers_per_thread,
        shared_per_block, registers_per_sm, shared_per_sm, max_warps, max_blocks) <= 0:
        raise ValueError("positive resources and warp-aligned threads required")
    warps_per_block = threads // 32
    by_registers = registers_per_sm // (threads*registers_per_thread)
    by_shared = shared_per_sm // shared_per_block
    by_warps = max_warps // warps_per_block
    blocks = min(by_registers, by_shared, by_warps, max_blocks)
    return blocks, blocks*warps_per_block/max_warps

assert toy_residency() == (2, .25)
assert toy_residency(shared_per_block=16384) == (4, .5)
print("toy residency:", toy_residency(), toy_residency(shared_per_block=16384))
```

这段模型没有寄存器分配粒度、每 block 的独立上限以及架构相关资源规则，目的是训练你对多个约束取最小值。代码里 `by_registers` 是一个资源池能容纳的 block 数，`by_shared` 对应另一资源池；最后还要受 Warp 与 block 数上限约束。它与 Roofline 中取最大时间并不矛盾：一个是容量容纳数量，最小容量最先限制；另一个是完成工作所需时间，最大时间最先限制。

对访存问题，先问访问是否合并、是否有重复、缓存能否复用、是否足够多请求在途；对计算问题，先问使用了什么指令路径、是否存在长依赖、向量化或 Tensor Core 条件是否满足、工作量是否足够覆盖设备；对同步问题，检查 barrier 是否必要、线程工作是否不均衡、原子操作是否集中争用。每个问题都应对应一次可控实验。例如把连续访问改成大步幅访问，若有效带宽显著下降且事务效率随之变化，才支持访存合并的解释。

Shared Memory 的 bank 冲突也应这样研究。转置教学中把行宽从三十二改为三十三，意图改变列读取落到 bank 的映射；先证明输出不变，再在固定输入下测时间，然后在可用时观察相关计数。若时间没有改善，可能原冲突并非瓶颈，也可能新布局增加了资源成本。实验失败仍有教学价值，因为它否定的是一个具体预测，而不是笼统证明“padding 没用”。

## 小课十一：小 Shape 为什么需要自己的策略

对于大矩阵，增加 Tile 可以提高输入复用，让同一份加载数据参与更多乘加；对小矩阵，同一个大 Tile 可能只覆盖少量有效元素，其余位置被 mask 掉或用零填充。虽然不会写出错误结果，但计算资源、寄存器和调度开销仍可能被浪费。正确性上的“尾部处理”与性能上的“尾部利用率”是两个问题。解决前者后，才有资格评价后者。

以输出二维 Tile 为例，M=N=33，Tile 边长三十二，需要各维两个 Tile，共四块，覆盖六十四乘六十四的逻辑位置，实际输出只有一千零八十九个元素，利用率约百分之二十六点六。若边长十六，需各维三块，共九块，覆盖四十八乘四十八的位置，利用率约百分之四十七点三。较小 Tile 的覆盖效率更高，但 block 数和调度次数增加，输入复用也可能改变，所以不能仅凭覆盖效率就宣布一定更快。

```python
def tile_coverage(m, n, tile_m, tile_n):
    if min(m, n, tile_m, tile_n) <= 0:
        raise ValueError("positive dimensions required")
    grid_m = (m+tile_m-1)//tile_m
    grid_n = (n+tile_n-1)//tile_n
    covered = grid_m*tile_m*grid_n*tile_n
    return {"blocks": grid_m*grid_n, "positions": covered,
            "useful_fraction": m*n/covered}

large_tile = tile_coverage(33, 33, 32, 32)
small_tile = tile_coverage(33, 33, 16, 16)
assert large_tile["blocks"] == 4 and small_tile["blocks"] == 9
assert small_tile["useful_fraction"] > large_tile["useful_fraction"]
print(large_tile, small_tile)
```

矩阵乘法还有归约维 K 的尾部。即使输出块完全填满，K 不是计算块或指令要求的倍数时，最后一轮也可能包含无效工作。若为了满足某个快速路径而显式 padding，必须把 padding 的分配、填充和最终裁剪纳入对应范围。服务中可以提前复用填充权重，但动态激活仍可能每次需要处理；报告应说明预处理是否一次性、是否摊销、摊销了多少次。把只做一次的准备成本永久忽略，与把每次都发生的工作藏在计时区间外，是两种不同问题。

小输入常被启动、调度和框架包装成本主导。此时把算术指令再减少百分之十，整体可能几乎不变；把连续的加法、激活与缩放合成一次 kernel，减少中间张量与启动次数，可能更有效。融合也会延长活跃值寿命，增加寄存器需求，某些情况下导致 spill 或降低可调度性。因此先写出融合前后读写次数、启动次数、临时张量大小，再测代表形状。不能把“融合”本身当作不需要实验的优点。

对 `y=(a+b)*scale`，假设 scale 是一个已准备好的标量且忽略其极少量流量，两个独立 float32 kernel 的理想大数组流量为：加法读 a、b 并写临时 t，共十二 N 字节；缩放读 t 并写 y，共八 N 字节，总二十 N。融合后读 a、b，直接写 y，共十二 N，减少四成有效字节，启动次数从二变一。这只是数据路径模型，实际编译器可能已经做了融合，缓存也会改变真实显存流量；先检查执行图再归功于自己的改动。

```python
def fusion_payload(n, item_bytes=4):
    if n <= 0 or item_bytes <= 0:
        raise ValueError("positive sizes required")
    unfused = (3+2)*n*item_bytes
    fused = 3*n*item_bytes
    return unfused, fused, 1-fused/unfused

assert fusion_payload(1000)[:2] == (20000, 12000)
assert math.isclose(fusion_payload(1000)[2], .4)
print("synthetic payload model:", fusion_payload(1000))
```

另一种策略是分派：根据形状、布局、类型与已验证条件选择不同实现。分派不是作弊，而是承认一个实现不覆盖所有资源条件。小向量走简单路径，中等形状走一种 Tile，大形状走另一种或可靠库函数，都可以是合理设计。关键在于选择规则不能偷看输出答案，不能依赖测试集的随机偶然波动，更不能把未验证布局送入只支持连续内存的实现。保持回退路径比强行在所有输入上使用自己的 kernel 更有工程价值。

一个常见陷阱是在六个形状里挑出候选获胜的三个，然后只报告这三个，最后宣称整体更快。正确过程是先记录完整工作负载分布，制定候选与回退规则，再在独立的代表输入上验证整条分派路径。分派本身的条件判断、张量检查和可能的预处理也要算入应用成本。对于新架构或新软件版本，原有阈值可能失效，因此分派策略应绑定验证环境，并允许禁用候选，不把某次机器上的阈值当成普遍定律。

## 小课十二：故障实验——把不可靠的性能结论逐个拆开

第一种故障是单位错误。某脚本得到事件时间零点零二毫秒，用一百万字节除以零点零二，打印五千万并标注 byte/s。实际分母应为二乘十的负五次方秒，正确结果是五百亿 byte/s，整整相差一千倍。定位时不要立即怀疑硬件，先在报告里加入原始毫秒与换算后的秒，并对一个手算样例做断言。单位转换应只发生一次，避免读取 JSON 时转一次、绘图时又转一次。

第二种故障是非整除输入。一个向量 kernel 在 N=1024 正确，N=1031 偶发错。先根据 block 大小列出最后一块的索引，检查 load 与 store 是否各自有正确 mask。只保护写入并不能防止越界读取；只保护读取也不能防止多写。矩阵乘法还需要分别保护 M、N 和归约 K，不能把一个方向的 mask 复制到另一个方向而不核对广播形状。在所有正确性检查通过前，停止对错误版本做性能排名。

第三种故障是错误的精度对齐。基线使用更宽的累计类型或更严格乘法精度，候选使用低精度近似；候选速度更高，但误差合同已经改变。修复不是一概禁止低精度，而是明确两组实验：同精度比较与精度换速度比较。后者要给最大误差、相对误差、代表性输入、极值与任务指标，不能仅说“肉眼看差不多”。元素接近零时相对误差容易放大，绝对与相对容差需要同时解释。

第四种故障是只测正常分布。一个 reduction 在随机小数上误差很小，在大量正负数相消时明显偏离。求和顺序改变会影响浮点结果；并行归约与串行参考不保证逐位一致。测试至少包含全零、常数、交替符号、不同量级、非整除长度以及必要的特殊值合同。哪些输入应拒绝、哪些应传播 NaN、哪些要求有限输出，应由算子定义决定。不能靠放大容差掩盖越界或漏项。

第五种故障是优化改变输出所有权。为了少分配，候选直接覆盖输入，而基线返回新张量；如果调用方之后还要使用输入，候选已经破坏程序。性能合同除了数值结果，还包括是否原地修改、返回别名、支持梯度以及布局条件。一个只验证返回值相等的测试会漏掉副作用。下面用纯 Python 列表演示这个错误，目的在于展示合同检查，不是在模拟 GPU 速度。

```python
def reference_add(a, b):
    if len(a) != len(b):
        raise ValueError("shape mismatch")
    return [x+y for x, y in zip(a, b)]

def broken_inplace_add(a, b):
    for i in range(len(a)):
        a[i] += b[i]
    return a

a, b = [1., 2.], [3., 4.]
before = a.copy()
y = broken_inplace_add(a, b)
assert y == [4., 6.]          # 只查数值会误判为成功
assert a != before           # 合同失败已被复现
fixed_a = before.copy()
fixed_y = reference_add(fixed_a, b)
assert fixed_y == y and fixed_a == before and fixed_y is not fixed_a
print("mutation reproduced; nonmutating reference passes")
```

第六种故障是把共享资源变化当作优化。上午基线跑十毫秒，下午候选跑七毫秒，但两次不同设备、不同负载甚至不同功率限制。你没有一个可解释的受控比较。修复方法是记录环境、在相近时间用同一设备运行两者、保留波动并说明共享情况。若服务器正忙，先完成 CPU 正确性、模型推导和短小环境检查，等待合适窗口再测。没有性能结果是一种明确状态；用不具备可比性的数字补满表格，会妨碍后续判断。

## 小课十三：从六行原始数据生成一份不夸大的报告

报告应该能追溯到输入与计时范围，而不是只有一个加速倍数。最小字段包括实验标识、代码版本、设备与软件、操作名、形状、类型、布局、数值合同、预热次数、重复次数、计时范围、`samples_ms` 原始样本及摘要、误差和运行状态。运行状态至少区分已执行通过、正确性失败、依赖缺失而跳过、尚未执行。跳过不等于耗时为零，也不应该参加平均值或加速比计算。

下面用六个**合成** shape 展示结果处理。它们不是现成 add CLI 的输入集合：CLI 的 add 长度必须为 size 的二百五十六倍，这里为了展示边界策略，直接使用独立的分析记录。若要实际测这些精确长度，需要用本卷已有 `triton_ops.add` 在满足其合同的 GPU 脚本中构造对应张量；不能把 CLI 的 size=257 误写成长度二百五十七。后面项目先提供能够直接用现成 CLI 执行的六个规模。

```python
synthetic_rows = [
    {"n": 256,   "base_ms": .008, "candidate_ms": .012, "correct": True},
    {"n": 257,   "base_ms": .009, "candidate_ms": .013, "correct": True},
    {"n": 4096,  "base_ms": .012, "candidate_ms": .011, "correct": True},
    {"n": 4097,  "base_ms": .014, "candidate_ms": .012, "correct": True},
    {"n": 65536, "base_ms": .030, "candidate_ms": .020, "correct": True},
    {"n": 65537, "base_ms": .031, "candidate_ms": .021, "correct": True},
]

def analyze_row(row, minimum_speedup=1.10):
    b, c = row["base_ms"], row["candidate_ms"]
    if not all(math.isfinite(x) and x > 0 for x in (b, c)):
        raise ValueError("invalid measured time")
    speedup = b/c
    accepted = row["correct"] and speedup >= minimum_speedup
    return {**row, "speedup": speedup,
            "route": "candidate" if accepted else "baseline",
            "selected_ms": c if accepted else b}

report_rows = [analyze_row(row) for row in synthetic_rows]
assert [r["route"] for r in report_rows] == ["baseline"]*3 + ["candidate"]*3
allowed_shapes = frozenset(r["n"] for r in report_rows if r["route"] == "candidate")

def route(n, dtype="float32", contiguous=True):
    if dtype != "float32" or not contiguous:
        return "baseline"
    return "candidate" if n in allowed_shapes else "baseline"

assert route(4097) == "candidate"
assert route(4097, contiguous=False) == "baseline"
assert route(1000000) == "baseline"   # 未验证形状不能凭空泛化
print([(r["n"], r["route"], round(r["speedup"], 3)) for r in report_rows])
```

这个参考实现刻意采用已验证形状白名单，而不是从三行合成获胜数据发明一个普遍阈值。`minimum_speedup` 是教学决策余量，不是统计显著性检验，也不能自动覆盖测量噪声。`correct` 必须来自真实正确性门槛，不能手工改成真来让候选被选中。字典展开 `{**row,...}` 保留原始字段并增加派生字段，使报告仍可追溯到输入数据。对于实际实验，应附上原始文件名和摘要算法版本。

如果六种 shape 在业务里出现频率不同，就不能简单平均六个加速比。假设小 shape 占九成，候选只在大 shape 快，整体收益可能很小。应该根据明确的工作负载权重计算总时间，再取基线总时间与选中路径总时间之比；或者真实重放同一输入序列。几何平均加速比可以用于某种汇总视角，但不能冒充业务总耗时改善。报告应保留逐 shape 表，使读者能看出胜负分布。

```python
weights = [40, 30, 15, 5, 5, 5]  # 合成的一百次请求分布
baseline_total = sum(w*r["base_ms"] for w, r in zip(weights, report_rows))
selected_total = sum(w*r["selected_ms"] for w, r in zip(weights, report_rows))
assert selected_total <= baseline_total
print("synthetic weighted speedup", baseline_total/selected_total)
```

不要把这个断言理解为“分派永远不会变慢”。这里选中路径没有计入分派开销，而且数据与决策来自同一组合成样本，所以断言只验证这张表的计算关系。实际使用需要独立验证集、分派成本和环境复核。明确一个结果的证据边界，能防止教学小程序在迁移到生产时被误用。优化报告最有价值的部分往往是这些边界，而不是最后一个小数点。

## 小课十四：十二道分级练习与完整参考答案

### 练习一：亲手检查单位与量级

题目：一百二十八 MiB 数据经过每秒六十四 GB 的理想路径，耗时多少毫秒？若误把 MiB 当成十进制 MB，会得到什么值，误差方向如何？

答案：正确字节数为一百二十八乘二的二十次方，即一亿三千四百二十一万七千七百二十八。除以六百四十亿字节每秒再乘一千，得到二点零九七一五二毫秒。误算为一亿二千八百万字节时得到两毫秒，低估约百分之四点六三。低估比例以正确值为分母，不是以错误值为分母。下面代码同时检查两种口径，输出应为正确时间与负的相对误差。

```python
correct_ms = 128*2**20/(64*10**9)*1000
wrong_ms = 128*10**6/(64*10**9)*1000
relative_error = (wrong_ms-correct_ms)/correct_ms
assert math.isclose(correct_ms, 2.097152)
assert wrong_ms == 2 and relative_error < 0
print(correct_ms, relative_error)
```

### 练习二：判断真正限制 Roofline 的参数

题目：计算下界四毫秒、搬运下界七毫秒。有人说最优时间为十一毫秒，另一个人说一定达到七毫秒，谁正确？若只把计算吞吐翻倍，下界是多少？

答案：在允许重叠的乐观模型中下界为七毫秒，十一毫秒对应另一种完全串行的成本模型，不能不加条件地称为最优下界。七毫秒也只是约束，不保证达到。计算吞吐翻倍后计算下界变两毫秒，最大值仍七毫秒。若存在每次不可重叠的三毫秒启动与准备开销，可以在更具体模型中得到至少十毫秒，但必须解释这三毫秒为何与其余阶段串行。答案的关键是写明依赖和范围，而不是在七与十一之间猜一个数字。

### 练习三：把矩阵形状变化转成强度变化

题目：固定 K=N=1024，float32，分别计算 M=1 与 M=128 的理想 FLOP、有效字节和强度，解释为什么增加 M 可能改善权重复用。

答案：M=1 时 F=2,097,152，B=4,202,496，强度约零点四九九；M=128 时 F=268,435,456，B=5,242,880，强度为五十一点二。第二个输入的运算量增加一百二十八倍，但同一份 K×N 权重可以服务更多行，理想字节量没有同比增长。它不能证明延迟降低，因为总工作量也增加了；能支持的是计算与搬运比例发生变化。完整计算直接复用第二课经过手算校验的函数。

```python
one = matmul_work(1, 1024, 1024)
many = matmul_work(128, 1024, 1024)
assert one[:2] == (2097152, 4202496)
assert many[:2] == (268435456, 5242880)
assert math.isclose(many[2], 51.2)
assert many[2] > one[2]
print("M=1", one, "M=128", many)
```

### 练习四：定位一次“超过峰值”的带宽

题目：有效字节除以事件时间得到的数值超过显存公开带宽，就能宣布计时无效吗？列出有顺序的检查，并指出怎样修复报告。

答案：不能直接宣布。先确认事件已完成、区间覆盖了目标工作，再检查毫秒换算、数组元素数与每元素字节。接着确认分子是有效载荷还是真实显存事务，观察工作集是否可能热缓存命中。如果分子是有效载荷，就将指标明确命名为有效带宽，并写明缓存条件；若研究显存瓶颈，另采集对应层级流量或采用明确的大工作集流式实验。第四课的一百二十万字节、两微秒、实际层级三十万字节例子给出六百与一百五十 GB/s，完整展示两个数能够同时成立，不需要假设硬件突破上限。

### 练习五：审查现有 benchmark 的计时范围

题目：运行 `--op add --size 64 --repeats 20` 后获得 JSON。能否说“测了六十四个元素的纯 kernel 时间，并从 JSON 算出百分之九十五分位数”？

答案：前两部分不对，最后一部分在读取正确字段后可以完成。长度为六十四乘二百五十六，即一万六千三百八十四；事件区间包住分配输出的调用包装器，源码明确不是完整服务延迟，不能直接等同某一个内部 kernel 的独立执行时间。当前 JSON 的 `samples` 是计数，`samples_ms` 才是原始列表；使用第七课 `summarize_measure` 验证计数与列表一致后，能够按明确的线性插值约定计算百分之九十五分位数。正确报告保留 shape、scope、冷启动秒数与稳定态毫秒摘要，不从最小值、中位数、最大值这三个摘要反推不存在的样本。二十次样本的尾分位数仍是有限样本描述，不能承诺精确的服务尾延迟。

### 练习六：检查异常值是否改变结论

题目：基线九、十、十、十一、五十，候选八、八、九、九、十，分别用中位数与平均数求加速比。能否删除五十？

答案：中位数比为十除九，约一点一一一；平均数比为十八除八点八，约二点零四五。只有经过独立证据确认并预先制定分类规则，才能把某次运行标为环境异常并分别报告；若五十来自算法真实慢路径，删除会掩盖尾部问题。第七课的完整摘要代码会保留它，下面再次直接核对两种聚合，避免把“平均加速比”与“平均时间之比”混淆。

```python
median_ratio = statistics.median(baseline_ms)/statistics.median(candidate_ms)
mean_ratio = statistics.mean(baseline_ms)/statistics.mean(candidate_ms)
assert math.isclose(median_ratio, 10/9)
assert math.isclose(mean_ratio, 18/8.8)
print("median ratio", median_ratio, "mean ratio", mean_ratio)
```

### 练习七：计算端到端收益上限

题目：kernel 占请求时间百分之十，加速两倍。请求总加速与无限加速上限分别是多少？若它原本完全隐藏在另一条十毫秒路径后面，还能直接套这个百分之十吗？

答案：串行占比模型给一点零五二六倍，无限加速上限为一除零点九，约一点一一一倍。若阶段互相重叠，原始 kernel 时间不能直接当作串行贡献，必须找关键路径。第八课八毫秒计算与十毫秒传输同时开始的例子中，把计算缩短为四毫秒，完成点仍为十毫秒；局部加速对该请求延迟没有贡献。答案既要给数值，也要指出公式使用条件，否则容易把一个精确公式应用到错误问题。

### 练习八：解释 occupancy 翻倍却变慢

题目：你把一个 Tile 减半，occupancy 从四分之一提高到二分之一，时间从十微秒增加到十二微秒。请给出两个可检验解释与对应实验。

答案：第一种解释是更小 Tile 降低数据复用，使同一输入被更多 block 加载；固定输入和精度，比较相关层级流量、有效带宽与 block 数，检查新增流量是否与模型一致。第二种解释是每块有效工作变少而调度、地址计算或同步占比增加；比较指令与启动分布，并在较大 shape 上重复观察，若大 shape 收益不同，支持规模相关的解释。也要检查寄存器与 spill，不能只从 occupancy 猜原因。第十课的资源模型只说明驻留约束变化，不承诺速度变化，这正是该反例应得出的结论。

### 练习九：修复只在整除尺寸正确的尾部

题目：block 大小二百五十六，N=1031。最后一块有哪些有效索引，哪些必须被 mask？写一个 CPU 模型验证每个有效索引恰好覆盖一次。

答案：需要五个 block，最后一块从一千零二十四到一千二百七十九；一千零二十四至一千零三十共七个索引有效，其余二百四十九个无效。下面逐块枚举并检查覆盖，既能发现漏项，也能发现重复。GPU 实现应分别在 load 与 store 上应用对应边界，而非仅在主机侧算对 block 数。

```python
def covered_indices(n, block):
    if n < 0 or block <= 0:
        raise ValueError("invalid launch geometry")
    seen = []
    for program in range((n+block-1)//block):
        for lane in range(block):
            index = program*block+lane
            if index < n:
                seen.append(index)
    return seen

seen = covered_indices(1031, 256)
assert seen == list(range(1031))
assert len(seen) == len(set(seen))
last_valid = [i for i in range(1024, 1280) if i < 1031]
assert last_valid == list(range(1024, 1031))
assert covered_indices(0, 256) == []
print("last block valid lanes", len(last_valid))
```

### 练习十：判断融合收益的来源与限制

题目：对 float32 的 `y=(a+b)*scale`，两阶段与融合的理想大数组有效字节分别多少？减少比例是多少？列出一个融合可能变慢的原因。

答案：忽略已准备标量的少量读取，前者二十 N 字节，后者十二 N 字节，减少四成。十一课 `fusion_payload` 给出完整可运行计算。潜在变慢原因是融合使中间活跃值变多、寄存器需求上升，导致 spill 或降低并发；也可能编译器原本已经融合，手写版本没有减少实际启动。验证应同时检查执行图、资源报告、输出和不同规模的时间，不能只根据公式计算的字节减少宣布实测收益。

### 练习十一：实现带安全回退的选择规则

题目：基于十三课的六条合成记录，候选至少快百分之十且正确才允许选用；未见形状、非连续布局与其他类型必须回退。再注入一次正确性失败，检查候选不会被选中。

答案：十三课 `analyze_row` 与 `route` 已构成完整参考实现，前三个形状回退，后三个进入白名单；`route(1000000)` 回退说明没有把未验证区域外推。下面通过复制一条原本获胜记录并修改正确性状态验证失败门槛。这个门槛是程序合同的一部分，性能再好也不能越过。

```python
failed = dict(synthetic_rows[-1], correct=False)
failed_report = analyze_row(failed)
assert failed_report["speedup"] > 1.1
assert failed_report["route"] == "baseline"
assert route(65536, dtype="float16") == "baseline"
assert route(65536, contiguous=False) == "baseline"
print("correctness failure always falls back:", failed_report["route"])
```

### 练习十二：完成一份能够被别人复核的实验结论

题目：有人提交一句“我的 matmul 快两倍”。把它改写成合格报告；没有 GPU 时怎样填写？

答案：完整结论需要具体形状、类型、布局、数值容差、设备与软件版本、代码标识、预热和重复数、计时范围、基线与候选分布，以及不支持的输入。例如可以写“在记录的环境与形状集合上，正确性检查通过；稳定态设备事件摘要如下，冷启动另列；分派只对已验证区域启用”。如果没有 GPU，不能填入人为时间或声称快两倍，应写“CPU 数学与结果分析已验证；GPU 性能待测；以下为合成报告格式演示”。本章所有合成数字都必须保留 synthetic 标记，真实数据文件应带实际采集时间与环境记录。十三课的分析器负责计算，下一课的分阶段项目负责形成可交付目录与运行证据，两者合起来就是参考答案的工程入口。

## 小课十五：把本章做成自己的算子实验台

第一阶段先不接触 GPU。在仓库根目录执行环境检查、默认基准和实验二十三，再把本章 Python 代码块按顺序放到一个练习脚本中运行。脚本应验证单位、矩阵工作量、Roofline、采样摘要、Amdahl、资源限制、Tile 覆盖、输入副作用、结果分析与尾部覆盖。此阶段的验收不是“程序没报错”这一句话，而是能解释每个断言对应的数学事实或合同：例如断言白名单外形状回退，验证的是策略边界，不是该形状上的 GPU 正确性。

第二阶段在个人 PC 或获得空闲窗口的服务器单卡运行现成基准。add 选择 size 为一、二、十六、十七、二百五十六、二百五十七，对应长度二百五十六、五百一十二、四千零九十六、四千三百五十二、六万五千五百三十六、六万五千七百九十二。这些形状覆盖较小与较大工作，但全部仍是二百五十六的倍数；它们不能替代任意尾部测试。matmul 可以选择三十一、三十二、三十三、一百二十七、一百二十八、一百二十九，通过非整齐方阵观察输出与归约尾部。每条命令最多二十次重复，先完成小尺寸，再决定是否需要更大规模。

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 1 --repeats 20
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 2 --repeats 20
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 16 --repeats 20
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 17 --repeats 20
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 256 --repeats 20
CUDA_VISIBLE_DEVICES=0 python -m labs.kernels.benchmark --run --op add --size 257 --repeats 20
```

这些命令的完整参考实现就是已经逐行解释过的 `labs.kernels.benchmark.main`，候选是 `labs.kernels.triton_ops.add` 或 `matmul`，基线是相同输入上的 PyTorch 运算。每次 JSON 保存为自己的运行记录，保留原始 scope 与 note。若准备比较两次代码改动，使用相同形状集合与随机种子，并在结果表中写出各自代码标识；不要覆盖旧记录后凭记忆描述收益。依赖缺失打印 SKIP 时，记录状态与原因即可，不要进入分派表的速度计算。

第三阶段提出一个局部假设，并只改变相应实现参数。可以研究 Tile 边长、向量 block 大小、融合前后的中间写回，或转置 padding。先列预测：哪些 shape 应受益、哪些可能退步、哪些指标或流量应变化。然后完成正确性矩阵，再做短小测量。若出现错误，恢复到可比较的正确版本；若没有改善，写出反证与下一步解释。一次没有加速的实验，只要排除了一个原因，也是一份合格作业。

第四阶段接入第十三课的结果分析与分派框架。训练选择规则的数据与验证规则的数据分开，保留未见形状回退。报告目录可以包含环境 JSON、原始基准 JSON、逐 shape 汇总、正确性记录和一页结论。结论逐项说明哪些路径真的测过、哪些仅有 CPU 模型、哪些需要 GPU 复核。不要为了完成目录而写空文件；最小可交付是六个形状的明确状态、一个已复现故障及修复证据、一个有预测和结果的优化实验，以及保持正确性的回退策略。

完成以上项目后，再把同样方法迁移到 RMSNorm、Softmax、Attention 或专家矩阵乘法。先重新定义工作量与流量，不能照抄 add 的十二 N 字节；先判断是否需要中间结果、归约同步和多阶段调度，不能照抄一个事件区间就宣称覆盖全部算法。你已经学会的核心能力，是从输入、状态、资源、依赖和测量范围建立解释，再用可复现结果修正解释。这种能力比记住某个显卡上的最佳 block 数更容易迁移。

### 官方核查入口与本章执行边界

以下一手页面于 2026-11-01 实际打开。这里只将它们用于对应工具的能力和计时语义核查，正文数值、推导与合成实验为独立教学内容。页面可能随版本更新，安装与实测时应记录本机版本，不把网页显示版本视为本仓库已经安装的版本。

| 一手资料 | 在本章中的用途 |
|---|---|
| [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | 聚焦采集、CUDA 时间线、目标进程范围与命令帮助 |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 指标集合、采集开销与重放对实验的影响 |
| [PyTorch CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html) | 事件记录、完成等待与毫秒时间差语义；核查时 stable 重定向到版本化页面 |
| [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | 预热、同步与具有代表性的运行设置 |

本章的 CPU 公式、合成数据分析与故障复现可离线执行；实际执行证据由本卷深度审计汇总。现有 GPU 基准、Nsight 采集、真实显卡吞吐和候选分派的硬件性能都没有在交付环境执行。正文没有给 H100 或 RTX 5070 Ti 编造性能值，也不把合成设备的参数当成硬件规格。完成硬件实验后应把“待测”替换为真实记录，并保留原始计时范围与环境条件。
