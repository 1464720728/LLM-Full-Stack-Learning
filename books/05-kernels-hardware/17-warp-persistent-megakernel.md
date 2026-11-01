# K17 Warp 分工、持久执行与大范围融合：从缓冲协议到任务运行时

本章前置 K06 的分块 GEMM、K13 的流与完成事件、K16 的架构边界。你已经能够写一个算子，现在要回答更大的问题：谁负责准备下一块数据，谁正在消费当前块，缓冲何时可以复用，一组长期工作的执行者怎样领取新任务，多个算子之间的依赖又由谁管理。

这些问题不能靠把函数写长来解决。本章先建立可执行的 CPU 状态模型，用明确反例检查代际、依赖、资源和终止，再对照固定 CUTLASS 源码中的真实异步协议。模型用于验证高层逻辑，不证明 GPU 内存序或硬件性能；真实设备实验按对应硬件条件另行执行。

正文 Python 块按出现顺序在同一会话执行，只依赖标准库与 NumPy。不要因为例子里出现 worker、warp 或 stage 就以为 CPU 程序正在模拟硬件时钟。它们分别帮助你理解任务身份、角色与资源生命周期，参数中的时间单位也是明确的教学假设。

## 第一课：三个名字分别改变了哪个维度

### 1. Warp specialization 改变角色分工

同一个 thread block 内，不同 warp 或更大协作单位可以承担不同职责，例如一组发起数据搬运，另一组执行矩阵计算，再由适当角色处理输出。它改变的是谁执行哪类工作，并不自动要求任务数量很大，也不自动意味着整个模型融合进一个 Kernel。

普通分块 GEMM 中，线程可能轮流参与加载、同步与计算。角色分工可以让某些执行者维持更专门的指令流，并配合异步硬件形成重叠；代价是角色也占线程和资源，负载不平衡会让某一组等待。不能只根据“有专职搬运者”就推断一定更快。

### 2. Persistent 改变工作获取与生命周期

传统映射常把许多任务分成很多 block，由设备调度执行。持久执行则让相对固定的一组 worker 在一个 Kernel 生命周期内循环领取或遍历多个任务。它可以使用动态队列，也可以使用静态 grid-stride 任务序列；“持久”不等于必须有复杂原子队列。

这种设计可能复用状态、降低重复启动或改善不规则任务分配，但 worker 数量、资源占用和任务领取开销都需要考虑。一个长循环并不自动保留你想要的数据 locality；若每个任务访问完全不同大块数据，所谓复用可能并不存在。

### 3. 大范围融合改变计算边界

MegaKernel 这个名字在不同项目中范围并不完全统一，通常强调把较大的计算子图组织在一个设备程序里，可能涉及多个算子、层或任务类型。读论文或源码时应直接描述融合了什么、任务由谁调度、哪些中间结果仍写回全局内存，而不是用一个标签代替机制。

三者可以组合，也可以独立存在。一个具有专门生产者/消费者的 GEMM，融合范围仍只是一个矩阵乘法；一个持久向量处理器可以只执行同一种简单任务；一个融合 Norm 与逐元素操作的 Kernel 也可以没有动态任务队列。先分清角色、生命周期和融合范围，后面才容易比较方案。

## 第二课：四个时间点不能合并成一个 ready

### 1. 发起搬运不等于数据已经可读

生产者收到一个空缓冲，发起异步加载，随后可能继续做别的事。加载命令已经提交，只证明请求进入执行路径，不证明所有字节已经到达消费者可用的位置。若立即把缓冲标成 READY，消费者可能读到上一轮残留或部分新数据。

因此我们区分 EMPTY、LOADING、READY、COMPUTING 四个高层状态。EMPTY 表示没有当前消费者占用且可由生产者领取；LOADING 表示数据仍在准备；READY 表示相应完成条件已经满足；COMPUTING 表示消费者正在使用，不能覆盖。状态名称是教学抽象，真实硬件通常通过屏障、计数和事件编码这些关系。

### 2. 发起计算不等于操作数已经不再被读取

消费者提交异步矩阵操作后，设备可能仍继续读取共享缓冲。若立刻把缓冲还给生产者，下一块写入可能覆盖尚未被旧操作消费完的数据。这与加载尚未完成就读取是相反方向的风险：一个是读得太早，一个是写得太早。

正确释放条件应对应最后一次仍可能读取该缓冲的操作完成。真实实现可能允许多批矩阵工作在途，并在证明某个旧 stage 已经消费完后逐步释放，而不是每次都等待全部工作清空。优化发生在精确完成关系上，不能简单删除等待来获得较短主机代码。

### 3. 可见性、顺序和所有权各自承担责任

所有权回答谁有权写或复用缓冲；顺序回答哪个操作必须先于另一个；可见性回答消费者是否能观察到生产者完成的写入。一个普通布尔变量不一定同时提供三者。跨线程、跨 warp、跨 block 时，还需要符合所用平台的内存与同步语义。

CPU 状态模型能检查“逻辑上在计算中仍尝试覆盖”这种错误，却不能自动验证某条 GPU barrier 是否覆盖了正确线程、字节范围和内存作用域。把抽象状态映射到真实 API 时，应对每条边分别列出实现机制与假设。

## 第三课：循环缓冲为什么必须区分代际

### 1. 同一物理槽会先后保存不同任务

设两张共享缓冲，任务零用槽零，任务一用槽一，任务二又用槽零。若只用槽号标识某次加载，任务零的迟到完成通知可能在槽零已经用于任务二时到达。代码看到“槽零完成”就把新一轮标成 READY，会把旧事件误认成当前数据完成。

我们为句柄增加 generation。任务 t 使用槽 `t % stages`，在该槽上的代际为 `t // stages`。句柄包含槽号、代际与任务 ID。释放一代后代际加一，旧句柄即使槽号相同，也无法通过当前身份检查。这与 Agent 恢复中的调用身份、知识库删除重建版本有共同原则：资源位置可以复用，事件身份不能混淆。

### 2. 一个完整而严格的状态机

下面代码使用同步方法调用模拟合法转移。reserve 表示领取可写槽；publish 表示写入已完成；consume 表示开始读取；release 表示读取已经完成。真实设备版本必须把 publish/release 绑定到实际完成条件，不能仅因为调用了同名方法就认为硬件已完成。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Handle:
    slot: int
    generation: int
    task: int

@dataclass
class Slot:
    generation: int = 0
    state: str = "EMPTY"
    task: int | None = None
    value: int | None = None

class StagePool:
    def __init__(self,stages=2):
        if type(stages) is not int or stages < 1:
            raise ValueError("positive integer stage count required")
        self.slots = [Slot() for _ in range(stages)]
    def reserve(self,task):
        if type(task) is not int or task < 0:
            raise ValueError("nonnegative task ID required")
        index = task % len(self.slots)
        slot = self.slots[index]
        generation = task // len(self.slots)
        if slot.state != "EMPTY" or slot.generation != generation:
            raise RuntimeError("slot unavailable or wrong generation")
        slot.state,slot.task = "LOADING",task
        return Handle(index,generation,task)
    def current(self,handle,expected):
        slot = self.slots[handle.slot]
        if (slot.generation,slot.task,slot.state) != (handle.generation,handle.task,expected):
            raise RuntimeError("stale handle or illegal transition")
        return slot
    def publish(self,handle,value):
        slot = self.current(handle,"LOADING")
        slot.value,slot.state = value,"READY"
    def consume(self,handle):
        slot = self.current(handle,"READY")
        slot.state = "COMPUTING"
        return slot.value
    def release(self,handle):
        slot = self.current(handle,"COMPUTING")
        slot.state,slot.task,slot.value = "EMPTY",None,None
        slot.generation += 1

stages = StagePool(2)
h0,h1 = stages.reserve(0),stages.reserve(1)
stages.publish(h1,10)  # Independent loads may complete in another order.
stages.publish(h0,0)
assert stages.consume(h0) == 0
stages.release(h0)
h2 = stages.reserve(2)
assert h2.slot == h0.slot and h2.generation != h0.generation
try:
    stages.publish(h0,999)
except RuntimeError:
    pass
else:
    raise AssertionError("stale completion was accepted")
stages.publish(h2,20)
assert stages.consume(h1) == 10
stages.release(h1)
assert stages.consume(h2) == 20
stages.release(h2)
```

### 3. 逐行检查状态不变量

reserve 先检查 EMPTY 与预期代际，才改变状态，避免重复领取或跨代复用。current 同时比较代际、任务和状态，因此同一任务重复 publish、未 READY 就 consume、未 COMPUTING 就 release 都会失败。身份与状态共同限定某次操作是否适用，不能只检查其中一项。

release 清空 task/value，但保留并递增 generation。把整个槽重新初始化为代际零会破坏迟到事件检测；资源清理不等于忘记识别旧操作所需的信息。句柄不可变也有帮助，调用者不能在正常代码里随意把旧句柄字段改成新身份来混用。

这仍不是一个并发安全队列。多个 CPU 线程同时 reserve 时需要锁或原子协议，GPU 上还需要相应可见性；当前方法仅在单线程受控调度下执行，用于检验高层状态。明确限制后，它仍是很好的设计工具，因为很多死锁和错代问题在这里就能被暴露。

## 第四课：让两个缓冲处理五个任务

### 1. 合法交错比固定演示更有检验价值

一个固定顺序能展示正常路径，却可能遗漏加载先后变化、消费者变慢或重复完成。我们构造一个有限随机调度器：每次从当前合法动作中随机选一项，动作包括领取下一任务、完成某次加载、开始最早待消费任务、结束当前计算。只允许一个计算引擎，两个槽可以有不同加载进度。

程序保存每个任务的句柄，发布值为 task*10，消费时检查身份与数值一致，完成后释放。随机种子固定便于复现；测试多个 seed 只是探索有限交错，不是对所有并发执行的形式证明。每次执行都有最大步数限制，避免错误协议无限循环。

```python
import random

def exercise_pipeline(tasks=5,stage_count=2,seed=0):
    pool = StagePool(stage_count)
    rng = random.Random(seed)
    handles,finished,events = {},[],[]
    next_load,next_compute = 0,0
    computing = None
    for _ in range(max(1,tasks*20)):
        if len(finished) == tasks:
            assert all(s.state == "EMPTY" for s in pool.slots)
            return events,finished
        actions = []
        if next_load < tasks:
            slot = pool.slots[next_load % stage_count]
            if slot.state == "EMPTY" and slot.generation == next_load // stage_count:
                actions.append(("reserve",next_load))
        for task,handle in handles.items():
            slot = pool.slots[handle.slot]
            if slot.task == task and slot.generation == handle.generation and slot.state == "LOADING":
                actions.append(("publish",task))
        if computing is None and next_compute in handles:
            h = handles[next_compute]
            slot = pool.slots[h.slot]
            if slot.task == next_compute and slot.state == "READY":
                actions.append(("consume",next_compute))
        if computing is not None:
            actions.append(("release",computing))
        if not actions:
            raise RuntimeError("protocol made no progress")
        action,task = rng.choice(actions)
        if action == "reserve":
            handles[task] = pool.reserve(task)
            next_load += 1
        elif action == "publish":
            pool.publish(handles[task],task*10)
        elif action == "consume":
            assert pool.consume(handles[task]) == task*10
            computing = task
            next_compute += 1
        else:
            pool.release(handles[task])
            finished.append(task)
            computing = None
        events.append((action,task))
    raise RuntimeError("step budget exceeded")

for seed in range(20):
    events,finished = exercise_pipeline(seed=seed)
    assert finished == list(range(5))
    assert len(events) == 20
```

### 2. 为什么结束不能只检查加载队列为空

当最后任务已经领取，next_load 等于 tasks，但它可能仍在 LOADING、READY 或 COMPUTING。此时退出会遗漏结果或留下还在被设备读取的缓冲。正确结束条件至少需要所有任务完成，且资源回到允许退出的状态；如果系统还有输出搬运或通信，也要包含它们的完成责任。

例子里 finished 只在 release 后追加，因此它代表计算消费完成并归还资源，而不是“已领取”。四个动作各执行一次，所以五任务正常轨迹有二十个状态转移。这个计数是协议结构推导，不是硬件指令数量，也不是设备时间。

### 3. 如何把状态模型变成故障试验

在个人副本中让 publish 提前发生，再让消费者读尚未赋值的数据；或者让 release 在 consume 前发生，观察 current 的状态检查拒绝。再保留旧句柄，在新代际上重复发布。每个错误都应被具体不变量击穿，而不是最终只看到一个无法解释的错误输出。

若你删除所有检查，某些 seed 仍可能成功，因为特定交错没有触发风险。不能用一次随机运行成功证明协议正确。测试既要覆盖多种合法交错，也要直接构造已知非法转移，分别验证正常活性与错误拒绝。

## 第五课：流水线收益从哪里来，什么时候消失

### 1. 手算串行与理想重叠

设每块加载 L 个时间单位，计算 C 个时间单位，共 N 块。串行执行需要 N*(L+C)。理想两阶段流水线在首块加载和首块计算后，后续每块间隔由较慢阶段 max(L,C) 决定，近似总时长为 `L+C+(N-1)*max(L,C)`。

例如三块、加载四、计算八，串行三十六；理想流水线为四加八再加两次八，得到二十八。第一块必须先加载，最后一块也必须完成计算，因此不能直接写成 N*max(L,C) 而忽略填充与排空。N 很小时，这些边缘阶段占比尤其明显。

```python
from labs.kernels.reference import pipeline_time
assert pipeline_time(3,4,8) == 28
assert 3*(4+8) == 36
assert pipeline_time(0,4,8) == 0
assert pipeline_time(1,4,8) == 12
```

### 2. 假设不满足时公式不会替你兜底

这个模型假设加载与计算能独立重叠、缓冲足够、同步无额外代价、各块工作量相同。若两阶段争用同一瓶颈资源，或者共享内存不足导致驻留减少，真实结果可能更差。模型中的重叠空间只是机会，不能当作已经获得的加速比。

增加 stage 数可以容纳更多在途工作，但也增加共享缓冲占用。若一个 stage 保存 A/B 两块，各自大小已知，总共享空间随 stage 数增加。超过某个资源阈值后可驻留 block 数可能下降，减少隐藏其他延迟的能力。不能只画更深流水线就认定吞吐改善。

### 3. 角色数量要跟瓶颈匹配

当计算阶段远慢于加载，增加生产者数量可能没有帮助；生产者只会更早填满有限缓冲，然后等待。若加载阶段慢，额外消费者也可能空转。真实 warp 分配还受矩阵指令协作粒度、寄存器与共享内存限制，不能把角色数量当成任意连续可调比例。

一个可靠实验先记录 L/C 或相关 stall 线索，再只改变一项角色或 stage 配置，检查正确性、资源用量和端到端时间。若同时改 tile、精度、warp 数、stage 数和输入布局，观察到变化也很难解释原因。K14 的控制变量原则在这里仍然适用。

## 第六课：持久 worker 怎样取得任务

### 1. 静态分配也可以持久执行

假设有 N 个独立输出 tile、W 个 worker。worker r 可以依次处理 r、r+W、r+2W，直到越过 N。这就是任务层面的 grid-stride 遍历，不需要全局原子队列。它容易证明每个任务恰好覆盖一次，但任务耗时差异较大时，某个 worker 可能拖到最后。

```python
def static_assignments(task_count,workers):
    if type(task_count) is not int or task_count < 0 or type(workers) is not int or workers < 1:
        raise ValueError("valid task and worker counts required")
    return [list(range(worker,task_count,workers)) for worker in range(workers)]

assignments = static_assignments(7,3)
assert assignments == [[0,3,6],[1,4],[2,5]]
assert sorted(task for group in assignments for task in group) == list(range(7))
```

这个覆盖证明与真实线程地址计算不同，但思路相同：先定义逻辑任务，再把每个任务唯一映射到执行者。若输出 tile 在 M/N 平面上，领取一维 task ID 后还需要把它翻译成 tile 行列，并处理边缘尺寸。任务分配正确不自动证明 tile 内地址正确。

### 2. 动态领取改善某些不规则负载

动态方案让空闲 worker 领取下一个尚未分配任务，可能减少静态分配导致的不平衡。真实设备上可以使用原子计数器等机制，但领取操作本身有竞争和成本。任务很小或队列高度争用时，管理开销可能超过节省的空闲时间。

下面用离散时间模型比较两 worker 处理 `[9,1,1,1,1,1]` 六个任务。静态交错分配使第一 worker 总工作十一，第二只有三，尾部由十一决定；动态分配中一个 worker 处理长任务九，另一个处理五个短任务，总完成时间九。模型没有领取开销，也没有硬件争用，结果只用于解释负载均衡机会。

```python
import heapq

def dynamic_independent_schedule(durations,workers):
    if type(workers) is not int or workers < 1 or any(d <= 0 for d in durations):
        raise ValueError("positive workers and task durations required")
    available = [(0,worker) for worker in range(workers)]
    heapq.heapify(available)
    trace = []
    for task,duration in enumerate(durations):
        start,worker = heapq.heappop(available)
        end = start+duration
        trace.append({"task":task,"worker":worker,"start":start,"end":end})
        heapq.heappush(available,(end,worker))
    return trace,max((row["end"] for row in trace),default=0)

durations = [9,1,1,1,1,1]
static_makespan = max(sum(durations[i] for i in group) for group in static_assignments(6,2))
dynamic_trace,dynamic_makespan = dynamic_independent_schedule(durations,2)
assert static_makespan == 11 and dynamic_makespan == 9
```

### 3. 领取不等于完成

原子计数器递增通常只表示某个 worker 获得任务身份，不表示输出已经写好。如果下游根据“已领取数量”判断前置完成，就可能提前读取未完成数据。应分别保存待领取、执行中和已完成等必要状态，完成发布必须与数据可见性对应。

还要考虑异常或退出。如果任务已领取而 worker 无法完成，简单计数器不会自动把任务放回队列。某些 GPU Kernel 选择整个作业失败并重新执行，某些运行时可能有其他恢复策略；无论采用哪种，都要说明完成条件与失败处理，不能假定所有领取最终一定成功。

## 第七课：有依赖的任务不能只靠一个递增计数器

### 1. 从独立 tile 进入计算图

假设 A 准备输入，B 和 C 分别消费 A 的结果，D 合并 B/C。B 与 C 可以并行，但它们都必须等 A；D 必须等两者。一个普通 FIFO 队列如果先把 B、C 交给所有驻留 worker，让它们占着资源等待 A，而 A 还没执行，就可能没有 worker 能产生它们所等的数据。

设备上还有更严格的资源条件：如果所有可驻留 block 都在自旋等待尚未被调度的 block，后者可能因资源被占而无法驻留，形成循环等待。不能把 block 间全局屏障简单写成“大家等某计数达到目标”，除非执行方式与同步机制明确保证参与者能前进。

### 2. 只把依赖已满足的任务放入 ready 队列

一种清楚的调度结构是记录每个任务尚未完成的前置数量。前置全部完成后，任务进入 ready；worker 只从 ready 领取，不占着执行资源等待尚未满足的依赖。任务完成时更新后继计数，刚降为零的后继被放行。这个机制仍需正确原子性与内存序，下面先用 CPU 离散事件模型验证逻辑。

```python
def schedule_dag(spec,workers=2):
    if type(workers) is not int or workers < 1:
        raise ValueError("positive worker count required")
    remaining,successors = {},{name:[] for name in spec}
    for name,item in spec.items():
        dependencies = tuple(item["deps"])
        if len(set(dependencies)) != len(dependencies) or any(dep not in spec for dep in dependencies):
            raise ValueError("invalid dependency list")
        if item["duration"] <= 0:
            raise ValueError("positive duration required")
        remaining[name] = len(dependencies)
        for dep in dependencies:
            successors[dep].append(name)
    ready = [name for name,count in remaining.items() if count == 0]
    heapq.heapify(ready)
    free = list(range(workers))
    heapq.heapify(free)
    running,finished,trace = [],set(),[]
    now = 0
    while len(finished) < len(spec):
        while ready and free:
            name,worker = heapq.heappop(ready),heapq.heappop(free)
            end = now+spec[name]["duration"]
            heapq.heappush(running,(end,worker,name))
            trace.append({"task":name,"worker":worker,"start":now,"end":end})
        if not running:
            raise ValueError("cycle or unsatisfied dependency prevents progress")
        now = running[0][0]
        completed = []
        while running and running[0][0] == now:
            end,worker,name = heapq.heappop(running)
            heapq.heappush(free,worker)
            finished.add(name)
            completed.append(name)
        for name in completed:
            for successor in successors[name]:
                remaining[successor] -= 1
                if remaining[successor] == 0:
                    heapq.heappush(ready,successor)
    return trace,now

spec = {"A":{"deps":(),"duration":3},
        "B":{"deps":("A",),"duration":5},
        "C":{"deps":("A",),"duration":2},
        "D":{"deps":("B","C"),"duration":1}}
dag_trace,dag_end = schedule_dag(spec,2)
by_task = {row["task"]:row for row in dag_trace}
assert by_task["B"]["start"] == by_task["C"]["start"] == 3
assert by_task["D"]["start"] == 8 and dag_end == 9
for name,item in spec.items():
    assert all(by_task[name]["start"] >= by_task[dep]["end"] for dep in item["deps"])
```

### 3. 同时完成与终止条件

模拟先收集同一时间点全部完成任务，再更新后继，避免 Python 堆的弹出顺序人为决定一个后继是否多等一轮。真实设备中事件并不总按整齐时间点聚合，但逻辑目标一样：后继只有在全部必要完成条件满足时进入 ready，而且只能进入一次。

终止条件是所有任务完成，不是 ready 队列暂时为空。ready 为空但仍有 running，是正常等待；ready 与 running 都空却还有未完成任务，说明依赖循环或缺失条件阻止前进。把这两种状态混淆，可能过早退出，也可能永远等待一个不存在的生产者。

这个 CPU 模型在任务层完成依赖调度，没有模拟 GPU 原子计数竞争、写入发布的 release/acquire 或跨设备通信。读真实 MegaKernel 时，你要继续寻找这些下层机制，不能因为图调度伪代码正确就跳过并发实现。

## 第八课：融合的收益要从真实数据流推导

### 1. 一个最小的融合链

考虑 `Y=ReLU(X@W+b)`。分成三个 Kernel 时，GEMM 写出中间矩阵 Z，bias Kernel 读 Z 并写 U，ReLU Kernel 读 U 并写 Y。融合 epilogue 可以在一个输出 tile 的累加器仍可用时加 bias、做 ReLU，再只写最终 Y。

若输出有 M*N 个元素，中间 Z/U 各经历一次写和一次读，粗略可减少四倍 M*N 个元素的全局中间流量。这个账本没有包含输入 X/W、最终 Y、cache 命中或其他工作区，所以不能直接等同于实测 HBM 流量降低。它用于明确优化企图消除的是哪些数据往返。

下面给出完整分块 CPU 数值参考。每次只建立一个输出 tile，完成矩阵乘法、bias 与激活后写回。它展示融合边界，不把 NumPy 调用当成一个真实 GPU Kernel，也没有验证设备寄存器驻留。

```python
import numpy as np

def fused_tile_reference(x,w,bias,tile_m=2,tile_n=3):
    x,w,bias = [np.asarray(value,dtype=np.float64) for value in (x,w,bias)]
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0] or bias.shape != (w.shape[1],):
        raise ValueError("incompatible linear layer inputs")
    if type(tile_m) is not int or type(tile_n) is not int or min(tile_m,tile_n) < 1:
        raise ValueError("positive tile sizes required")
    output = np.empty((x.shape[0],w.shape[1]))
    for row in range(0,len(x),tile_m):
        for column in range(0,w.shape[1],tile_n):
            end_row,end_column = min(row+tile_m,len(x)),min(column+tile_n,w.shape[1])
            accumulator = x[row:end_row] @ w[:,column:end_column]
            accumulator += bias[column:end_column]
            output[row:end_row,column:end_column] = np.maximum(accumulator,0)
    return output

rng = np.random.default_rng(9)
x,w,bias = rng.normal(size=(5,7)),rng.normal(size=(7,8)),rng.normal(size=8)
np.testing.assert_allclose(fused_tile_reference(x,w,bias),np.maximum(x@w+bias,0),rtol=1e-12,atol=1e-12)
assert fused_tile_reference(x,w,bias).shape == (5,8)
```

### 2. 融合范围受依赖约束

逐元素 bias 与 ReLU 只需要当前输出元素，适合在局部结果可用后处理。行归一化需要整行统计，attention 需要跨 key 归约，后续层矩阵乘法又可能需要多个前序 tile。融合这些操作时，必须保存或合并足够统计，并建立更大范围依赖，不能按同一个局部 epilogue 模板处理一切。

如果中间结果有多个消费者，提前覆盖它也会影响其他路径。残差、分支、MoE dispatch 和多模态拼接都可能引入额外使用者。每次增加融合范围，先画出谁还需要旧值、需要哪些坐标、何时完成，再决定缓冲能否复用。

### 3. 减少启动不代表每个组成算子都更高效

融合可能增加寄存器存活范围、共享内存、代码体积与分支。一个原本能调用高度优化 GEMM 的路径，改成大融合程序后可能牺牲矩阵单元效率；减少的启动和中间流量未必抵消这种损失。小 batch 与大 batch、短序列与长序列也可能表现不同。

因此报告应同时给出正确性、支持形状、编译与首次运行成本、稳态时间、资源使用和端到端收益。不要只比较某个 tiny shape 的最佳延迟，就声称替代完整推理框架。框架还承担请求、缓存、调度、取消、模型兼容等职责，融合 Kernel 通常只解决其中一段执行路径。

## 第九课：对照固定 CUTLASS 源码理解真实协议

本章重新读取了 [固定提交的 Hopper 主循环](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp)，提交 SHA 与本卷来源表一致。下面只说明实际可定位的责任，不把单文件阅读当成已经编译或跑通整套 CUTLASS。

### 1. 生产者先获得可写 stage，再发起搬运

在 load 主循环中，生产者调用 producer_acquire，随后取得当前 stage 对应 barrier，并使用带该 barrier 的 TMA copy 发起 A/B 搬运。stage 的索引来自流水线写状态，迭代后推进写状态。这里要分别理解可写权、数据搬运与完成通知，不是看到一个 copy 函数就假设消费者已经可读。

这个文件的该段没有一个名为 producer_commit 的显式调用，因此不能照某张通用示意图硬填“acquire→copy→commit”。真实完成协议要沿所用 pipeline 和 TMA barrier 继续查。源码导读的目标是对应真实机制，方法名列表只是入口，不能反过来规定代码必须长成你预想的样子。

### 2. 消费者的读取进度与释放进度可以不同

消费者先通过 consumer_try_wait/consumer_wait 等待所需 stage，然后发起矩阵工作并提交批次。代码同时维护读取状态与释放状态，允许一定数量矩阵工作仍在途。它在满足相应 warpgroup_wait 条件后释放更早的 stage，而不一定释放刚刚提交的那个。

这正是“开始计算”和“可以复用”不能合并的实际例子。若你把两套状态索引统一成一个，可能在设备仍读取旧操作数时归还缓冲。理解这种滞后关系，需要跟踪具体代际和在途数量，不能只说“这里用了双缓冲”。

### 3. 尾部排空是算法的一部分

mma_tail 等待剩余矩阵工作完成，再释放尚未归还的阶段。主循环结束意味着没有新的 K tile 要提交，不代表所有已提交工作结束。若省略尾部，短输入或最后几块可能读写不完整，资源也可能处在错误状态。

源码中的 prologue、稳态循环和 tail 分别对应填充、持续重叠与排空。阅读时应为一个很短 K tile 数量手工走完三段，尤其是 tile 数少于常规流水深度时。只看中间长循环，很容易漏掉初始累加、最后等待或阶段数量边界。

### 4. 进一步阅读的可控范围

[官方 Efficient GEMM 说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) 提供分块层级背景，[Triton persistent matmul 教程](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) 提供另一种持久矩阵乘法阅读入口，本次于 2026-11-01 核查。动态教程可能包含多种架构专用路径，运行前必须按目标分支检查前提，不能默认个人显卡和 H100 都走相同代码。

先选择一个已确认适用的路径，记录输入 shape、设备、编译目标、program 数与任务遍历，再比较你的 CPU 分配模型和真实调度。不要把整份带大型 benchmark 的上游脚本未经阅读直接启动在共享服务器上；先缩到一个明确可回答问题的小实验。

## 第十课：资源账本决定哪些重叠真的可能发生

### 1. 每个 stage 都有存储成本

假设一个 GEMM stage 保存 A tile `[BM,BK]` 和 B tile `[BK,BN]`，每元素 e 字节，那么原始载荷为 `BK*(BM+BN)*e`。若 BM=64、BN=64、BK=32、e=2，一个 stage 是八千一百九十二字节；三 stage 是二万四千五百七十六字节，还未计 padding、barrier、描述符或其他缓冲。

这份计算应在改变 stage 数之前完成。增加 stage 不会凭空增加设备共享内存容量；同时驻留的其他 block 也要占相同类型资源。真实编译器还可能加入布局调整，因此公式是设计估算，编译输出与运行时信息是另一个证据层。

```python
bm,bn,bk,element_bytes = 64,64,32,2
stage_bytes = bk*(bm+bn)*element_bytes
assert stage_bytes == 8192
assert 3*stage_bytes == 24576
```

### 2. 一个虚构设备上的驻留上界

为了练习资源约束，假设某设备每 SM 有六万五千五百三十六个寄存器、二千零四十八个线程额度、一百二十八 KiB 共享内存，另设最大十六个 block。这些是人为教学参数，不是对 H100 或个人显卡规格的声明。

若每 block 一百二十八线程、每线程六十四寄存器、每 block 四十 KiB shared，则寄存器最多允许八个 block，线程额度允许十六个，shared 只允许三个，所以简单上界是三个。若 shared 增到八十 KiB，上界变一个。真实硬件还有分配粒度和其他限制，不能直接把这个最小值当作编译器保证。

```python
def residency_upper_bound(sm_registers,sm_shared,sm_threads,max_blocks,
                          threads_per_block,registers_per_thread,shared_per_block):
    if min(sm_registers,sm_shared,sm_threads,max_blocks,threads_per_block,
           registers_per_thread,shared_per_block) <= 0:
        raise ValueError("positive model quantities required")
    return min(max_blocks,sm_registers//(threads_per_block*registers_per_thread),
               sm_shared//shared_per_block,sm_threads//threads_per_block)

assert residency_upper_bound(65536,128*1024,2048,16,128,64,40*1024) == 3
assert residency_upper_bound(65536,128*1024,2048,16,128,64,80*1024) == 1
```

### 3. occupancy 是约束线索，不是唯一目标

较高驻留可能帮助隐藏延迟，但不意味着越高越快。一个使用更多寄存器的 tile 可能减少重复加载、提高矩阵效率，即使驻留下降仍更快；反过来，为了提高 occupancy 过度缩小 tile，可能增加总工作量。性能应根据真实瓶颈解释，不以单一指标排名。

持久 worker 的数量也受这个账本限制。如果你设计了必须同时活跃的角色或全局协作，却没有确认它们能同时获得资源，协议可能根本无法前进。资源分析因此不仅服务性能，还可能是正确性前提。把这个前提写进设计，比在挂起后随机改 block 数更可靠。

## 第十一课：三个故障工作坊

### 故障一：缓冲重用几轮之后偶尔出现旧值

先记录槽号、代际、任务 ID 和状态转移。若只在循环复用后出现，检查迟到的加载或计算完成通知是否仍能作用于同一槽的新任务。一个布尔 ready 无法区分旧任务零和新任务二，尤其在异步时序变化后容易误认。

最小复现使用第三课 h0 与 h2：释放 h0 后重用同一槽，再用 h0 触发 publish，必须被拒绝。修复要把事件身份与当前代际绑定，而不是只在某处增加一次清零。清零不能阻止旧完成事件在未来再次到达。

GPU 版本还要核对 barrier 的阶段位或计数与索引推进是否一致。CPU 句柄检查只能说明应有哪种逻辑约束，不能替你证明一个硬件 phase 的翻转时机正确。需要沿真实 API 和代码逐次跟踪。

### 故障二：所有 worker 都在等待，设备没有继续执行

先画出等待关系与占用资源。如果两个驻留 worker 分别运行 B、C，它们都等待尚未执行的 A，而 A 排在未调度任务中，就形成资源与依赖循环。只提高队列容量没有帮助，因为可执行生产者仍无法获得 worker。

参考修复是在任务满足依赖前不占用计算 worker，使用 ready 队列；或者采用经过证明的协作启动与同步协议。不能笼统说“GPU 最终会调度所有 block”，因为占着资源自旋的 block 可能阻止所需生产者驻留。测试应构造这种具体等待图，而不是只尝试增加随机延迟。

第七课 schedule_dag 对循环依赖明确报错，对正常 A→B/C→D 图正常完成。这是设计层的负向控制，不需要真的在共享 GPU 上启动一个可能挂死的程序。危险时序可以先用有界模型揭露，再在适当环境验证底层机制。

### 故障三：融合后数值正确，速度反而下降

先比较实际执行路径，而不是只数 Kernel 数量。融合可能增加寄存器、shared、编译复杂度，或者使原本高效的大 GEMM 变成较低效的通用路径。还要区分首次编译与稳态执行，确认计时同步、输入范围和基线配置公平。

参考调查顺序是：确认相同数学工作；记录输入 shape/dtype/layout；比较启动、中间流量和资源；找到主要变化；只调整一个因素重测。若没有 GPU 实测，本章只能提出这些待验证解释，不能凭 CPU 模型选一个最顺耳的原因当结论。

成功的优化报告也应说明退化区间。某个小 batch 改进而大 batch 下降，可能符合启动与矩阵效率的权衡；清楚给出适用范围，比只展示最有利点更有工程价值。

## 第十二课：十二道练习与完整参考答案

### 练习一：分类三种机制

一个 Kernel 中三组 warp 分别搬运、计算、输出，且只处理一个 GEMM；另一个 Kernel 中固定 worker 循环处理独立向量；第三个把多层计算和依赖队列融合。参考答案分别突出角色分工、持久任务执行和大范围融合。它们可以组合，但不能仅凭名字推导其他两项。答题应描述执行机制，而不是争论术语标签。

### 练习二：算槽与代际

三 stage，任务七使用哪个槽和代际？七除以三商二余一，因此槽一、代际二。任务一也使用槽一但代际零，不能接受任务一的旧完成通知。再计算任务十，得到槽一、代际三，说明物理位置复用与逻辑身份变化是独立维度。

### 练习三：拒绝四种非法转移

对同一个句柄，未发布就 consume、未消费就 release、重复 publish、释放后再次 release 都应拒绝。完整参考如下。每次失败之后仍继续完成合法路径，证明拒绝没有破坏当前可恢复状态。

```python
def expect_runtime_error(action):
    try:
        action()
    except RuntimeError:
        return
    raise AssertionError("illegal transition was accepted")

checked_pool = StagePool(1)
checked_handle = checked_pool.reserve(0)
expect_runtime_error(lambda: checked_pool.consume(checked_handle))
expect_runtime_error(lambda: checked_pool.release(checked_handle))
checked_pool.publish(checked_handle,7)
expect_runtime_error(lambda: checked_pool.publish(checked_handle,8))
assert checked_pool.consume(checked_handle) == 7
checked_pool.release(checked_handle)
expect_runtime_error(lambda: checked_pool.release(checked_handle))
```

### 练习四：扩大合法交错覆盖

把任务数改成零、一、二、五、九，stage 数改成一、二、三，各测试十个种子。参考要求每项恰好完成一次、结束时无占用槽、动作总数为四倍任务数。它覆盖短输入、单缓冲与多轮复用，但仍是受控单线程模型，不能宣称穷尽真实设备所有交错。

```python
for tasks in [0,1,2,5,9]:
    for stage_count in [1,2,3]:
        for seed in range(10):
            events,finished = exercise_pipeline(tasks,stage_count,seed)
            assert finished == list(range(tasks))
            assert len(events) == 4*tasks
```

### 练习五：反过来让加载更慢

五块、加载八、计算四，串行六十，理想流水线为八加四再加四次八，等于四十四。参考解释是瓶颈从计算换成加载，稳态间隔仍由较慢阶段决定。若只有一块，无论哪边慢，都没有后续块可重叠，总时长十二；这一边界检查能抓住漏掉填充/排空的公式。

```python
assert pipeline_time(5,8,4) == 44
assert pipeline_time(1,8,4) == 12
```

### 练习六：比较动态与静态分配，但加入假设

使用第六课六任务例子，得到十一与九。参考报告必须注明没有队列领取开销、任务独立、两个 worker 能并行且互不争用。若每次动态领取新增较大固定成本，优势可能消失。不能把离散模型九比十一的比例写成 GPU 实测加速比。

### 练习七：检测依赖循环

定义 A 依赖 B、B 依赖 A，没有初始 ready。schedule_dag 应有限时间内抛 ValueError，而不是返回空结果假装完成。再移除 A 的前置，图应完成。完整参考如下。

```python
cycle = {"A":{"deps":("B",),"duration":1},"B":{"deps":("A",),"duration":1}}
try:
    schedule_dag(cycle)
except ValueError:
    pass
else:
    raise AssertionError("cycle accepted")
acyclic = {"A":{"deps":(),"duration":1},"B":{"deps":("A",),"duration":1}}
assert schedule_dag(acyclic)[1] == 2
```

### 练习八：验证每个后继只放行一次

在 A→B/C→D 中，B/C 可以同一时间完成。把二者 duration 都设为二，要求 D 只出现一次，开始时间为五，总时间六。参考代码按任务名字计数，防止每个前置完成时都直接把 D 入队而忽略剩余依赖数量。

```python
same_end = {name:{"deps":item["deps"],"duration":item["duration"]} for name,item in spec.items()}
same_end["B"]["duration"] = same_end["C"]["duration"] = 2
same_trace,same_time = schedule_dag(same_end)
assert sum(row["task"] == "D" for row in same_trace) == 1
assert next(row for row in same_trace if row["task"] == "D")["start"] == 5
assert same_time == 6
```

### 练习九：给融合 tile 加非整除测试

改变 M/N/K 为三、五、七，tile 设二、三。参考输出与 `maximum(X@W+b,0)` 相同，右下角不足 tile 也正确。完整函数在第八课；本题重点是边界切片，不是重新复制一份同样的矩阵乘法公式作唯一判据。

```python
edge_x,edge_w,edge_bias = rng.normal(size=(3,7)),rng.normal(size=(7,5)),rng.normal(size=5)
np.testing.assert_allclose(fused_tile_reference(edge_x,edge_w,edge_bias,2,3),
                           np.maximum(edge_x@edge_w+edge_bias,0),rtol=1e-12,atol=1e-12)
```

### 练习十：算出融合可能减少的中间流量

输出一千乘一千、float32，Z/U 各一次写一次读，理想可省四百万个元素，即一千六百万字节。参考答案同时说明仍需写最终 Y、读权重与输入，还可能有 cache 和其他临时存储。若报告直接写“总显存流量减一千六百万”，缺少实际路径依据；应该称为当前数据流账本中被消除的中间访问量。

### 练习十一：对照源码中的读与释放指针

在固定 CUTLASS 文件中找到 smem_pipe_read 与 smem_pipe_release，解释为什么二者可能滞后。参考答案是新矩阵批次已提交时，旧阶段只有在适当完成条件满足后才可释放；tail 还要等待剩余在途工作。验收需跟踪一轮索引与代际，而不是只列出三个方法名。

### 练习十二：写一份有边界的方案比较

比较普通多 Kernel、局部融合、持久同类任务和更大计算图融合，分别写需要保存的状态、同步范围、潜在减少的开销、新增资源成本与适用输入。参考结论不是选一个普遍赢家，而是提出一项可实测假设，例如短请求中启动占比高时，某种局部融合可能有收益。然后写清哪组测量可以推翻这个假设。

## 第十三课：综合小项目——用真实数组执行一个依赖图

前面的 DAG 模拟只安排时间，不实际计算数组。现在把任务节点绑定到函数，只有前置结果存在时才执行。参考图为输入 X 经两个线性分支得到 B/C，再由 D 相加并做 ReLU。这个执行器是同步 CPU，帮助你把任务身份、数据依赖和输出结果连起来，不是设备端 MegaKernel。

```python
def execute_ready_graph(definitions):
    names = set(definitions)
    for name,item in definitions.items():
        if any(dep not in names for dep in item["deps"]):
            raise ValueError("unknown dependency")
    pending,values,order = set(names),{},[]
    while pending:
        ready = sorted(name for name in pending if all(dep in values for dep in definitions[name]["deps"]))
        if not ready:
            raise ValueError("graph has no ready task")
        for name in ready:
            item = definitions[name]
            arguments = [values[dep] for dep in item["deps"]]
            result = item["function"](*arguments)
            values[name] = result
            order.append(name)
            pending.remove(name)
    return values,order

project_x = np.array([[1.0,2.0],[-1.0,3.0]])
left_weight = np.array([[1.0,0.0],[0.0,2.0]])
right_weight = np.array([[0.0,1.0],[1.0,0.0]])
definitions = {
    "A":{"deps":(),"function":lambda:project_x.copy()},
    "B":{"deps":("A",),"function":lambda a:a@left_weight},
    "C":{"deps":("A",),"function":lambda a:a@right_weight},
    "D":{"deps":("B","C"),"function":lambda b,c:np.maximum(b+c,0)},
}
values,execution_order = execute_ready_graph(definitions)
np.testing.assert_allclose(values["D"],np.maximum(project_x@left_weight+project_x@right_weight,0))
assert execution_order.index("A") < execution_order.index("B") < execution_order.index("D")
assert execution_order.index("A") < execution_order.index("C") < execution_order.index("D")
```

这个完整版本保留所有中间 values，便于审查，但不优化内存。若要在最后消费者完成后释放中间结果，应为每个值计算剩余消费者数量，并在消费完成时递减；不能在第一个消费者读取后立即删除 A，因为另一个分支还需要它。输出或调试保留也可能延长寿命，需要明确哪些值是外部结果。

你还可以把两个线性分支代数合并成 `X@(W_left+W_right)` 再 ReLU，并验证当前特定图的等价性。但若两个分支中间各有不同非线性，就不能直接合并。图优化要依靠具体代数条件与浮点边界，而不是看到两个 matmul 就自动相加权重。

项目交付包括状态图、完整 CPU 执行器、直接数学对照、依赖循环反例、缓冲生命周期说明，以及一份映射到真实设备时尚需补齐的机制表。表中至少包括任务领取原子性、数据发布可见性、跨 worker 完成、资源驻留和终止。它们是具体的下一层工程责任，不能缩成一句“以后优化性能”。

## 第十四课：取消、错误与有界排空也属于运行时

### 1. 停止领取与停止设备访问不同

持久 worker 收到停止信号后，可以先不再领取新任务，但已经提交的搬运、矩阵操作或输出写入可能仍在执行。此时立即释放所有缓冲，会把“逻辑上不想继续”误当成“设备已经不再访问”。取消协议应说明哪些在途工作必须排空，哪些结果应被丢弃，以及什么时候才能归还资源。

一种清楚的教学分层是 admission_closed、draining、stopped：先关闭新任务进入，再等待已接受任务达到定义的安全边界，最后确认没有仍被使用的缓冲。它不要求所有业务结果都必须继续发布；任务可能完成底层计算但因请求已取消而不再交付。资源完成与结果可见性是两个决策。

如果取消发生在生产者已领取槽但尚未发布时，还要决定如何处理该代际。不能简单跳过 release 并永久占槽，也不能在写入尚未结束时把槽交给下一任务。真实协议需要让完成或失败事件最终关闭这次拥有关系，迟到事件仍由代际身份防止作用于新任务。

### 2. 部分角色提前退出可能阻断其他角色

假设消费者遇到一个错误直接 return，而生产者仍在等待它归还 stage，后者可能永远无法前进。反过来，生产者停止发布后，消费者可能仍在等待一个不会到来的 READY。错误路径必须传播终止信息，并定义等待者如何退出，不能只在正常循环末尾考虑清理。

CPU 模型可以用一个共同的失败状态拒绝新 reserve，并把未完成任务报告为失败；但 GPU 实现中，错误传播还要遵守实际同步原语的参与规则。不是随意让某个 warp 跳过 barrier 就能“优雅退出”。参与计数、完成事件与退出路径必须一起设计，尤其是角色分工后每组可能处于不同阶段。

### 3. 队列容量与背压

持久运行并不意味着任务描述符、ready 队列和结果缓存可以无限增长。若生产者生成任务的速度持续超过消费者，有限队列必须让上游等待、拒绝或分批提交。背压把资源约束传播回任务来源，避免最后变成不可解释的内存耗尽。

还要分别限制任务数量与任务载荷字节。十个任务可能各自引用一个巨大输入，十万个小描述符也可能只有有限载荷；仅看 queue length 不能完整描述内存。输出队列同样需要边界，否则计算端很快而主机消费慢时，结果缓存仍会增长。

设计报告应为每类资源列出谁申请、谁拥有、谁释放、何时允许阻塞、失败后如何收尾。这个表能连接本章的 stage、DAG、worker 与真实框架请求生命周期。一个只测正常吞吐的 Kernel 演示，尚不能证明这些运行时边界成立；把它们明确列出，才知道需要增加哪些有针对性的验证。

## 第十五课：把模型变成真实设备实验时保留证据边界

本章的 Python 状态机、调度与数组例子可以在个人 PC 或服务器 CPU 上完成，已经足以让你手工判断许多错代、依赖与资源问题。使用 H100 的时机是需要核对 Hopper 特定矩阵与搬运协议，或者实际测量某个适用路径；不需要为了执行这些逻辑模型占用 GPU。

真实实验首先固定源码提交和可用设备，选择一组小形状，验证数值与完成协议，再逐步改变 tile、stage 或角色配置。使用本仓库有界实验工具限制单次运行，记录编译与稳态阶段。若一个配置需要架构专用能力，应在不满足时明确跳过，不把另一张卡运行相似算法当成该路径已验证。

阅读新项目时，先找 task descriptor、worker 循环、ready/依赖结构、缓冲句柄和终止条件，再深入具体 GEMM/Attention 实现。遇到“融合整个模型”的描述，逐项问哪些算子实际支持、动态 shape 如何处理、哪些状态仍回到主机、通信是否包含，以及性能测量覆盖哪些工作负载。用这些具体问题代替对宣传标签的猜测。

本章的完成标准是：你能给出一个分工会更慢的反例、一个错误释放缓冲的反例、一个依赖等待造成无法前进的反例，并用完整小程序修复它们。做到这一步，再继续打磨硬件实现，才能知道每一次优化保留了什么正确性，而不只是看到一次运行变快。
