# K13 Stream、Event与CUDA Graph：把时间关系写成可验证的程序

一个数组地址正确、公式正确的GPU程序，仍可能因为读取过早、覆盖过早或释放过早而出错。原因是主机提交任务与设备完成任务不是同一时刻，不同执行流之间也不会仅凭Python调用先后自动形成全部依赖。本章把时间当作程序的一部分，从最小队列模型推导stream顺序、event边界、缓冲区生命周期与图重放。

先修K02资源生命周期、K03主机设备执行、K06双缓冲。本章CPU模拟验证依赖关系与算法时间线，真实PyTorch/CUDA代码需要目标环境运行。当前环境没有GPU与PyTorch，设备片段未执行；不能把模拟中的时间单位称为毫秒或实测加速。

## 第一课：提交、开始、完成是三个事件

### 1. 主机函数返回代表什么

主机调用Kernel启动接口后，通常可以继续执行后续语句，而设备任务可能尚未开始或尚未完成。返回的Tensor对象描述输出存储与元信息，不保证其中数值已经可由CPU读取。把“拿到对象”理解成“计算结束”，会导致计时和生命周期错误。

可以把任务分成提交时刻、设备开始时刻与设备完成时刻。提交到开始之间可能有排队、依赖等待或调度开销，开始到完成才包含设备工作。用户端到端等待可能包含全部阶段，纯Kernel计时则关注特定设备区间，两者回答不同问题。

### 2. 同一stream提供顺序

普通同一stream中的操作按队列顺序建立依赖，适合表达复制输入、运行Kernel、复制输出。程序员不必在每两步之间让CPU等待，只要相关操作放在正确stream并遵守API合同，设备可以按序执行。

这种顺序不代表主机阻塞。CPU可以连续提交三项工作后继续做其他任务，设备稍后按顺序完成。异步主机与有序设备队列可以同时成立，不能把“异步”误解为“所有操作随机乱序”。

### 3. 不同stream只提供潜在并行

不同stream中的独立工作可能重叠，但是否真正并行取决于硬件资源、Kernel占用、复制引擎、依赖与运行时条件。创建更多stream不会增加GPU总算力，也不保证两个满负载Kernel同时运行更快。

如果两个stream访问同一数据且存在生产消费关系，必须建立显式依赖或使用框架明确保证的机制。CPU先调用生产者、后调用消费者，只说明提交顺序，不足以证明不同stream上的设备完成顺序。

## 第二课：用CPU依赖图理解stream

### 1. 每个任务保存前驱

将任务表示为名字、所属stream、持续时间与前驱集合。同stream前一个任务自动成为前驱，跨stream依赖通过额外边加入。这个图描述必须先完成什么，不描述真实硬件所有调度细节。

~~~python
from dataclasses import dataclass, field

@dataclass
class Task:
    name: str
    stream: str
    duration: float
    dependencies: set = field(default_factory=set)

class StreamPlan:
    def __init__(self):
        self.tasks = {}
        self.last = {}

    def submit(self, name, stream, duration, wait_for=()):
        if name in self.tasks or duration < 0:
            raise ValueError('unique name and nonnegative duration required')
        dependencies = set(wait_for)
        if stream in self.last:
            dependencies.add(self.last[stream])
        if not dependencies.issubset(self.tasks):
            raise ValueError('dependency must already exist')
        self.tasks[name] = Task(name, stream, float(duration), dependencies)
        self.last[stream] = name
        return name

    def event(self, name, stream):
        return self.submit(name, stream, 0.0)

    def earliest_times(self):
        finish = {}
        timeline = {}
        for name, task in self.tasks.items():
            start = max((finish[d] for d in task.dependencies), default=0.0)
            finish[name] = start + task.duration
            timeline[name] = (start, finish[name])
        return timeline

plan = StreamPlan()
copy_done = plan.submit('copy', 'transfer', 4)
ready = plan.event('ready', 'transfer')
plan.submit('compute', 'compute', 8, wait_for=(ready,))
assert plan.earliest_times()['compute'] == (4.0, 12.0)
~~~

这个模型按添加顺序要求依赖已存在，所以天然可以按该顺序计算最早完成时间。真实调度可能还有资源冲突与延迟，最早时间是理想依赖下界，不是实际GPU预测。后续增加资源模型时，要保留这层依赖正确性。

### 2. 删除依赖会发生什么

如果compute不等待ready，两条stream都没有跨流前驱，模型允许compute在时间零开始，而copy到时间四才完成。即使某次实际调度碰巧先复制后计算，也不能证明程序正确，因为存在合法的错误执行顺序。

正确性要求对所有允许的调度都成立，不是“我跑了几次没出错”。增加event依赖将错误顺序从允许集合中排除，这比增加随机延迟等待更可靠。固定sleep既不能保证设备完成，又会浪费时间。

### 3. 依赖图也能检查释放时机

如果buffer由copy写、compute读，释放或复用必须发生在compute完成之后，而不只是copy完成之后。数据就绪与存储空闲是两个不同边界。许多双缓冲错误恰好只记录前者，忘记后者。

把free或reuse也当作图节点，就能明确要求它依赖最后使用者。即使真实内存释放由框架分配器管理，理解这条关系仍有助于正确使用record_stream或其他生命周期接口。

## 第三课：Event记录的是队列中的边界

### 1. Event不是一个随意的布尔变量

在stream中记录event，表示一个完成标记排在此前工作之后。另一个stream等待该event，可以让其后续操作在生产者到达该边界后执行。主机也可以查询或等待event，但两种等待的影响范围不同。

设备stream等待通常把依赖排入设备执行，不要求CPU一直阻塞；主机同步等待则让调用线程等待完成。选择哪种方式取决于谁需要结果。若只是GPU消费者需要数据，没必要先让CPU等待整个设备再提交消费者。

### 2. Event应紧贴真正生产者

假设stream A先生成buffer，再执行一个与buffer无关的长任务。若event记录在生成buffer之后、长任务之前，消费者可以尽早开始；若记录在长任务之后，依赖虽然安全却过强，消费者无谓等待。

优化同步的原则不是尽量删除等待，而是保留必要依赖、去掉不必要依赖。先证明数据安全，再把event放到最早合法边界。没有依赖的并行可能错误，过度依赖的串行则浪费机会，两者都需要具体数据流分析。

### 3. 不同同步范围

设备级同步等待此前相关设备工作全部完成，适合最终验收或粗粒度调试，但会阻止很多重叠。stream同步只等待一个流，event同步只等待某个边界。范围越小，越容易保留独立工作，但也要求你准确知道需要等待什么。

默认stream还有模式与库交互细节，不能用“默认流总会帮我同步”的模糊印象写程序。初学阶段显式画出生产者、消费者与stream，并按实际API文档建立边，最容易获得可解释正确性。

## 第四课：完整PyTorch跨stream例子

~~~python
def stream_dependency_example():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    current = torch.cuda.current_stream()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    produced = torch.empty(16, device='cuda', dtype=torch.float32)
    result = torch.empty_like(produced)
    producer.wait_stream(current)
    consumer.wait_stream(current)
    ready = torch.cuda.Event()
    finished = torch.cuda.Event()
    with torch.cuda.stream(producer):
        produced.fill_(3.0)
        ready.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        torch.mul(produced, 2.0, out=result)
        finished.record()
    produced.record_stream(consumer)
    result.record_stream(consumer)
    finished.synchronize()
    assert torch.equal(result.cpu(), torch.full((16,), 6.0))
    return result
~~~

### 1. 为什么新stream先等待当前stream

张量分配与此前使用可能关联当前stream及缓存分配器状态。让新stream等待当前已有工作，建立明确起点，避免把“刚分配所以绝对没有历史依赖”当成无条件事实。具体框架分配器语义应以官方CUDA说明为准。

producer填充produced后记录ready，consumer先等待ready再读取，保证数值生产完成。finished记录在乘法后，主机等待它再复制结果到CPU检查。每条等待都有具体责任，不是随意插入同步碰碰运气。

### 2. Wait与record_stream解决不同问题

wait_event建立执行依赖，保证读取不会早于生产。record_stream向框架说明某张量在该stream上还有使用，帮助管理存储复用生命周期。后者不能替代前者：即使内存不被释放，消费者仍可能读到尚未写好的值。

同样，只等待生产完成不保证最后消费者完成前存储不会被错误复用。执行顺序与资源生命周期需要同时正确。本例保持对象引用并在结束同步，使学习路径清楚；复杂异步程序还要按框架规则管理跨流使用。

### 3. 这个例子没有承诺加速

两个stream存在明确依赖，消费者必须等生产者，本例主要验证正确性，不是并行性能演示。增加stream数量不能消除真实数据依赖。要获得重叠，需要同时存在其他独立工作或不同批次流水，并且硬件资源允许。

实际运行时记录设备、框架版本与输出检查。当前教材环境未执行此函数，因此只提供完整可审计代码与CPU依赖模型，设备结果留给目标环境验证。

## 第五课：主机到设备复制与Pinned Memory

### 1. 普通主机内存与页锁定内存

主机内存管理可能移动或换页，异步设备传输需要满足相应运行时与硬件条件。页锁定内存固定相关页面，常用于允许更直接的异步传输与复制计算重叠。它是有限系统资源，不应无节制锁定大量内存。

调用名带Async或参数写non_blocking，并不保证任何源内存、传输方向与环境下都完全异步或与计算重叠。必须核对API条件、源目标位置、内存类型和设备能力。先理解合同，再测量实际时间线。

### 2. 复制源也有生命周期

主机到设备复制尚未完成时，CPU不能修改或释放源缓冲，否则设备可能读到混合数据。设备到主机复制尚未完成时，CPU不能把目标内容当成已就绪结果。异步不仅影响设备数组，也影响主机buffer的使用时机。

一个常见错误是循环中反复填同一个pinned数组并立即提交复制，却没有等待上一轮复制完成。主机填充速度可能超过传输，导致后面的数据覆盖前一批源内容。双缓冲需要同时管理主机槽与设备槽，不只是多分配两个GPU数组。

### 3. 有界复制与计算示例

~~~python
def pinned_copy_example():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    host_input = torch.arange(1024, dtype=torch.float32, pin_memory=True)
    device_input = torch.empty(1024, device='cuda', dtype=torch.float32)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    copy_stream.wait_stream(current)
    compute_stream.wait_stream(current)
    copied = torch.cuda.Event()
    completed = torch.cuda.Event()
    with torch.cuda.stream(copy_stream):
        device_input.copy_(host_input, non_blocking=True)
        copied.record()
    with torch.cuda.stream(compute_stream):
        compute_stream.wait_event(copied)
        output = device_input.square()
        completed.record()
    device_input.record_stream(compute_stream)
    completed.synchronize()
    expected = torch.arange(1024, dtype=torch.float32).square()
    torch.testing.assert_close(output.cpu(), expected)
    return output
~~~

host_input一直保持引用且未修改，直到completed同步结束，因此也覆盖了更早复制完成的要求。若要更早复用主机源，只需等待copied边界，但设备输入仍要等compute完成后才能覆盖。两种资源的最后使用者不同，不能用一个含糊“done”代替全部状态。

## 第六课：双缓冲有两条方向相反的依赖

### 1. 数据就绪与槽位可复用

生产者加载第q块到槽q模二，消费者读取该槽计算。消费者必须等待加载完成；下一次生产者重用同槽时，必须等待旧消费者完成。第一条依赖从生产到消费，第二条从消费回到下一代生产，缺一不可。

如果只记录ready，第三块可能覆盖第一块尚在计算的槽零；如果只记录free，消费者可能在当前加载完成前读取。状态机至少区分空闲、装入中、就绪、消费中，或者用两个事件链表达同样关系。

### 2. 完整理想时间线

设加载耗时四、计算耗时八，三块任务。加载零在零到四，计算零在四到十二；加载一可在四到八；计算一在十二到二十；槽零要等计算零完成后才能加载二，因此加载二在十二到十六，计算二在二十到二十八。

总时间二十八，而完全串行是三乘十二等于三十六。不能直接写三乘max四八等于二十四，因为第一块加载与最后一块完成的填充排空成本没有消失。理想流水公式是L+C+(N减一)乘max(L,C)，前提包括可重叠资源与足够缓冲。

### 3. 完整双缓冲依赖计划

~~~python
def double_buffer_plan(tiles, load_time, compute_time):
    plan = StreamPlan()
    free_event = [None, None]
    for tile in range(tiles):
        slot = tile % 2
        waits = () if free_event[slot] is None else (free_event[slot],)
        loaded = plan.submit(f'load_{tile}', 'transfer', load_time, waits)
        done = plan.submit(f'compute_{tile}', 'compute', compute_time, (loaded,))
        free_event[slot] = done
    return plan

pipeline = double_buffer_plan(3, 4, 8)
timeline = pipeline.earliest_times()
assert timeline['load_2'] == (12.0, 16.0)
assert timeline['compute_2'] == (20.0, 28.0)
~~~

free_event保存该槽上一代最后消费者，下一代load等待它。每一代都使用独立任务名字，避免把复用事件的代际语义隐藏起来。真实实现可复用事件对象，但必须遵守API关于记录与等待的具体规则；初学先用清晰代际模型证明正确性。

## 第七课：CPU模拟合法交错与竞争

### 1. 不是每种调用顺序都安全

两个stream各有自己的顺序，跨stream没有边时，多种交错都合法。为了验证一个算法，应考虑其中是否存在读旧值、覆盖未读值或提前释放的交错。单次成功运行只展示一种调度，不能排除其他合法失败顺序。

~~~python
def all_topological_orders(tasks):
    names = tuple(tasks)
    results = []
    def visit(done, order):
        if len(order) == len(names):
            results.append(tuple(order))
            return
        for name in names:
            if name not in done and tasks[name].dependencies <= done:
                visit(done | {name}, order + [name])
    visit(set(), [])
    return results

unsafe = StreamPlan()
unsafe.submit('write', 'producer', 1)
unsafe.submit('read', 'consumer', 1)
orders_unsafe = all_topological_orders(unsafe.tasks)
assert ('read', 'write') in orders_unsafe

safe = StreamPlan()
write = safe.submit('write', 'producer', 1)
safe.submit('read', 'consumer', 1, (write,))
orders_safe = all_topological_orders(safe.tasks)
assert orders_safe == [('write', 'read')]
~~~

这个枚举只适合小图，任务多时组合数量爆炸。它不模拟真实warp或内存模型，只验证任务级依赖允许哪些顺序。即便如此，也足以发现许多stream级错误，并训练你用“所有允许交错”而不是“我希望的顺序”推理。

### 2. 给任务附上真实数组动作

~~~python
def execute_write_read(order):
    state = {'buffer': 0, 'observed': None}
    for name in order:
        if name == 'write':
            state['buffer'] = 7
        elif name == 'read':
            state['observed'] = state['buffer']
    return state['observed']

assert {execute_write_read(order) for order in orders_unsafe} == {0, 7}
assert {execute_write_read(order) for order in orders_safe} == {7}
~~~

错误计划允许零与七两种结果，正确计划只允许七。这个负向控制证明测试确实能发现缺失依赖，而不只是验证某个固定顺序。后续可把buffer替换成小NumPy数组，模拟复制、计算与复用，仍保持同样思路。

### 4. 用所有合法交错验证三块双缓冲

前面的时间线假设每项任务按理想最早时刻开始，但正确性不应依赖具体耗时。我们可以忽略时间，只枚举满足依赖的所有顺序，并给每次load与compute附上实际槽位读写。这样能验证即使某阶段突然变慢，数据代际仍不会混淆。

~~~python
def buffer_dependency_plan(include_reuse_edge):
    plan = StreamPlan()
    previous_consumer = [None, None]
    for tile in range(3):
        slot = tile % 2
        waits = ()
        if include_reuse_edge and previous_consumer[slot] is not None:
            waits = (previous_consumer[slot],)
        loaded = plan.submit(f'load{tile}', 'load', 1, waits)
        computed = plan.submit(f'compute{tile}', 'compute', 1, (loaded,))
        previous_consumer[slot] = computed
    return plan

def execute_buffer_order(order):
    slots = [None, None]
    output = []
    for name in order:
        if name.startswith('load'):
            tile = int(name[4:])
            slots[tile % 2] = tile
        else:
            tile = int(name[7:])
            output.append((tile, slots[tile % 2]))
    return output

bad_buffer_plan = buffer_dependency_plan(False)
bad_results = [execute_buffer_order(order)
               for order in all_topological_orders(bad_buffer_plan.tasks)]
assert any(any(expected != observed for expected, observed in result)
           for result in bad_results)
good_buffer_plan = buffer_dependency_plan(True)
good_results = [execute_buffer_order(order)
                for order in all_topological_orders(good_buffer_plan.tasks)]
assert all(all(expected == observed for expected, observed in result)
           for result in good_results)
~~~

错误计划允许load零、load一、load二先后完成，再compute零读取槽零，此时看到的是第二代数据二。正确计划要求load二等待compute零，所以这种覆盖顺序被排除。模型不需要猜测GPU实际怎样调度，只需发现是否存在合法错误交错。

这里每个槽只存一个代际编号，真实数组可以更复杂，但代际已经足以检查所有权。给调试buffer附上generation标签是一种很有效的思维工具：读者期待第几批数据，存储现在属于第几批，是否有证据保证一致。

### 5. 事件对象复用与代际标识

真实代码可能复用同一个event对象记录多轮完成。等待语义与记录时机必须按API规则理解，不能把event当成永远代表“这一槽已经空闲”的无时间布尔值。若等待对应了错误一轮，可能在旧信号上误判新数据已完成。

教学阶段使用每代独立事件或明确的任务ID最容易审计，确认正确后再减少事件对象数量。优化对象分配不应先于证明依赖。尤其多线程主机同时提交时，记录与等待的相对顺序更需要显式设计。

### 6. 缓冲区容量限制并发深度

双缓冲允许最多两代占用槽位，生产者不能无限向前提交覆盖。更多缓冲可以容纳更深流水与不同阶段波动，但增加内存，并不保证吞吐持续提升。若最慢阶段服务速度固定，超过足够隐藏延迟的深度后，新增缓冲主要增加排队与延迟。

把缓冲槽看成有限令牌：生产者取得空闲令牌，填充后交给消费者，最后消费者归还。这个模型能连接基础卷的队列与背压，也能解释为什么“CPU提交越快越好”并不总成立。没有背压的提交可能只让队列和内存占用不断增长。


## 第八课：计时边界必须对应问题

### 1. 主机提交时间

用CPU时钟包住异步launch，可能主要测到参数准备与提交，而不是设备完成。这个数字可以用于研究主机开销，但不能标成Kernel执行时间。若你关心端到端延迟，需要在结束前等待所测工作完成，并明确包含哪些排队与传输。

同步位置也会改变测量范围。开始前同步排除之前遗留任务，结束后同步包含当前完成；若只结束同步，可能把旧队列工作也算进去。测量不是在代码两边随便放计时函数，而是先定义要包含的因果区间。

### 2. 设备Event计时

~~~python
def event_timing_example(iterations=20):
    import torch
    if iterations <= 0 or not torch.cuda.is_available():
        raise ValueError('positive iterations and CUDA required')
    x = torch.randn(1 << 16, device='cuda')
    output = torch.empty_like(x)
    for _ in range(3):
        torch.mul(x, 2.0, out=output)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.mul(x, 2.0, out=output)
    stop.record()
    stop.synchronize()
    milliseconds = start.elapsed_time(stop) / iterations
    torch.testing.assert_close(output, x * 2)
    return milliseconds
~~~

start与stop记录在相关当前stream，包围多次操作以减少单次测量噪声。stop同步保证时间戳已经可用。返回单位是毫秒，换算秒或吞吐时必须正确转换。这个例子不包含输入分配与初始随机生成，报告应明确它是稳态设备区间。

### 3. 冷启动与稳态都可能重要

JIT编译、上下文建立、库初始化与缓存暖机可能使第一次调用明显不同。稳态比较可以先暖机，但服务冷启动也是用户真实体验，不能永远丢弃不报告。分别记录冷启动与稳态，才能回答不同部署问题。

共享服务器上的其他任务也可能影响计时。保持实验短、重复次数有界，记录设备负载条件，不把一次异常值当作永久硬件结论。K14进一步讲统计量、roofline与profiler，本章先保证时间关系正确。

### 4. 异步错误为何可能在后面才报告

Kernel启动接口可以发现某些配置错误，但设备执行中的非法访问可能在后续同步或其他CUDA调用时才被观察到。因此报错的Python行不一定是产生错误的Kernel。调试时应缩小任务链，并在关键边界临时同步以定位。

临时同步会改变执行时间线，可能掩盖某些竞争，所以不能仅因加同步后错误消失就宣布修复。应回到依赖图，找出原本缺失的边或生命周期条件，再用正确事件关系替代粗同步。定位工具与最终修复是不同步骤。

### 5. 取消任务不等于设备工作已经停止

主机决定不再需要某个结果时，已经提交的设备任务可能仍在执行。不能立刻把其输入输出缓冲交给无关任务复用，除非有明确机制保证旧工作结束或安全取消。教学流水的取消策略可以是停止提交新任务，再等待已提交任务排空。

错误路径也要释放资源，但释放顺序必须尊重未完成使用。普通主机RAII能管理对象销毁，不自动证明设备已不再访问对应存储。框架与运行时通常提供相应管理机制，外部扩展必须正确接入，不能只依赖函数作用域结束。

### 6. 主机线程并发与设备stream并发

多个CPU线程可以向不同或同一设备提交任务，但主机线程同步与GPU stream同步是不同系统。CPU互斥锁保护主机数据结构，不自动让某个GPU Kernel完成；CUDA event建立设备依赖，也不自动保护两个CPU线程同时修改同一Python外部缓冲。

若一个线程填充pinned输入、另一个线程提交复制，需要主机层明确交接所有权；复制完成后再把缓冲交回填充线程，还需要设备完成通知。两层协议共同保证正确。把“线程安全”当成一个笼统属性，容易遗漏跨层依赖。


## 第九课：CUDA Graph缓存什么

### 1. 记录任务与依赖

CUDA Graph把一组设备操作及其依赖组织成可重放结构，减少反复从CPU逐项提交的开销。它不自动把所有操作融合成一个Kernel，图中仍可包含多个Kernel、复制或其他支持节点。图重放与Kernel融合解决的问题不同。

基础capture/replay常要求重放使用捕获时的指针与结构，因此需要稳定输入输出缓冲。高级图更新、条件节点等能力有各自API与限制，不能把基础示例的静态假设说成所有CUDA Graph永远不支持动态行为。本章先学习最清楚的固定shape重放路径。

### 2. 重新绑定变量不等于更新图输入

捕获时Kernel参数指向static_input的存储。之后写static_input=new_tensor只是让Python变量指向新对象，捕获图仍可能使用旧地址。正确基础做法是把新数据copy到原static_input缓冲，再replay。

输出也通常复用固定存储。若用户要保留每次重放结果，需要在下一次重放覆盖前复制或消费完成。把同一个static_output引用追加到列表，可能得到多个都指向最后结果的条目，类似K07保存trace时忘copy的问题。

### 3. 完整有界图重放例子

~~~python
def cuda_graph_example():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    static_input = torch.zeros(16, device='cuda')
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            temporary = static_input * 2.0 + 1.0
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = static_input * 2.0 + 1.0
    snapshots = []
    for value in (3.0, 5.0):
        static_input.fill_(value)
        graph.replay()
        snapshots.append(static_output.clone())
    torch.cuda.synchronize()
    torch.testing.assert_close(snapshots[0], torch.full((16,), 7.0, device='cuda'))
    torch.testing.assert_close(snapshots[1], torch.full((16,), 11.0, device='cuda'))
    return snapshots
~~~

warmup在side stream进行，并与当前stream建立前后依赖，避免把初始化工作混入capture。静态输入与输出保持引用，重放前只修改输入内容。clone在同流顺序中保存每次结果，防止下一轮覆盖同一输出storage。

这个代码需要兼容的PyTorch与CUDA环境，当前未执行。capture期间哪些操作可用、分配器如何使用图内存池、随机数与多流如何处理，都应查实际版本官方文档；不能简单宣称“capture完全不能分配内存”或“任何Python代码都会被录进图”。

## 第十课：图重放、融合与流水线如何组合

### 1. 三种优化改变不同层次

融合改变Kernel边界与中间数据存储，可能减少全局内存读写；Graph减少重复提交与依赖设置成本；流水线尝试让不同阶段重叠。三者可以组合，但每次组合都要保持同样数学与生命周期合同。

例如两个小逐元素算子先融合成一个，再把每轮复制与融合Kernel捕获为图，最后使用多个稳定缓冲处理流水。每一步的收益来自不同机制，应该分别测量，避免把所有加速归因于某一个术语。

### 2. Shape bucket的代价

固定shape图可按常见尺寸建立多个bucket，实际请求padding到某个bucket再执行。这样增加复用机会，但padding增加计算与内存，bucket数量增加捕获时间与图缓存占用。选择应根据真实请求分布与预算，不是越多越好。

语义mask仍必须排除padding，尤其attention与归一化等归约算子。仅把输入padding成零不保证数学等价，K05与K07已解释分母污染。Graph复用不能绕过算子本身的真实长度合同。

### 3. 缓存键与失效

图缓存键可能需要包含shape、dtype、device、布局、模型版本与相关执行配置。只按batch大小缓存，可能把不同序列长或不同缓冲布局误用同一图。输入指针稳定性与生命周期也必须由缓存对象管理。

模型权重原地更新可能保留地址但改变内容，某些图可继续使用；重新分配权重可能改变地址，需要重新捕获或受支持更新。不能仅凭Python对象名字相同判断图仍有效，必须看实际资源合同。

## 第十一课：三个故障案例

第一类是跨stream先读后写。症状可能偶尔正确、偶尔旧值。修复在生产者完成点记录event，消费者等待该event，并验证所有合法CPU交错都满足读取顺序。不要用固定sleep或无关全局同步长期掩盖。

第二类是双缓冲只等待ready，没有等待旧消费者完成。第三块覆盖第一块的槽，结果可能混合两代数据。修复增加槽位free依赖，明确每代buffer从加载到最后读取的生命周期。

第三类是Graph重放前只重新绑定输入变量，输出总像旧输入。修复copy到捕获时的稳定buffer，并检查输出是否被下一次重放覆盖。数值公式没错，错误来自地址与时间关系。

## 第十二课：十二道练习与完整答案

### 练习一：区分三个时刻

答案：提交是主机把任务排入执行，开始是设备实际运行，完成是结果可按依赖使用。函数返回不自动等于完成。

### 练习二：同流与跨流

答案：普通同流顺序可表达生产消费；不同流需要明确依赖，CPU调用先后不足以证明设备完成先后。

### 练习三：Event位置

答案：应记录在数据生产完成后、无关长任务之前，使消费者尽早合法开始。放更后安全但增加等待。

### 练习四：Wait与record_stream

答案：前者建立执行顺序，后者帮助框架管理跨流存储生命周期。二者不能互相替代。

### 练习五：三块流水

答案：加载四、计算八、三块理想总时间二十八，串行三十六。包含填充与排空，不能只算三乘八。

### 练习六：搬运较慢

答案：加载八、计算四、三块同样二十八，瓶颈变为搬运。消费者更常等待输入，但公式假设仍需满足。

### 练习七：主机源复用

答案：异步H2D完成前不能改写源pinned buffer；等待复制完成即可复用源，设备槽则要等最后消费者完成。

### 练习八：输出不重叠

答案：两个stream写严格不重叠区域且没有其他依赖时可独立，但共同storage释放必须等两者完成。

### 练习九：计时单位

答案：Event返回毫秒时，求每秒吞吐要除以一千转换为秒；同时说明计时包含哪些操作与暖机。

### 练习十：图输入更新

答案：基础重放使用固定地址，应copy新内容到static_input，不是只给Python变量绑定新tensor。

### 练习十一：图输出保存

答案：每次重放可能覆盖同一static_output，需在覆盖前消费或clone，并建立正确流顺序。

### 练习十二：Graph与Fusion

答案：Graph复用提交与依赖结构，Fusion改变Kernel与中间数据组织。图中仍可有多个Kernel，二者可结合但不等同。

## 第十三课：用CPU对象模拟固定地址重放

### 1. 捕获的是对象关系，不是变量名字

为了理解图重放的地址问题，构造一个教学对象，保存输入与输出NumPy数组引用，并在replay时执行固定运算。它不是CUDA Graph实现，只模拟“重放仍访问捕获时资源”的关键合同。变量后来重新绑定，不会改变对象内部保存的引用。

~~~python
import numpy as np

class FixedBufferReplay:
    def __init__(self, shape, dtype=np.float64):
        self.input = np.zeros(shape, dtype=dtype)
        self.output = np.empty(shape, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def copy_input(self, value):
        value = np.asarray(value)
        if value.shape != self.shape or value.dtype != self.dtype:
            raise ValueError('shape and dtype must match captured buffers')
        np.copyto(self.input, value)

    def replay(self):
        np.multiply(self.input, 2.0, out=self.output)
        np.add(self.output, 1.0, out=self.output)
        return self.output

replay = FixedBufferReplay((3,))
external_input = replay.input
external_input = np.array([3., 3., 3.])
assert np.array_equal(replay.replay(), [1, 1, 1])
replay.copy_input(external_input)
first_snapshot = replay.replay().copy()
replay.copy_input(np.array([5., 5., 5.]))
second_snapshot = replay.replay().copy()
assert np.array_equal(first_snapshot, [7, 7, 7])
assert np.array_equal(second_snapshot, [11, 11, 11])
~~~

external_input最初与replay.input引用同一数组，重新赋值后只改变外部名字，内部数组仍全零。因此首次输出一，清楚模拟了“换变量不换捕获地址”。copy_input修改原数组内容，才让后续重放看到新数据。

output也始终同一数组。如果first_snapshot不调用copy，第二次replay会覆盖它看到的内容。这个例子与真实图输出复用非常接近，但没有异步与设备内存，不能据此推断CUDA性能或完整capture限制。

### 2. 输入输出别名的额外风险

如果调用者把replay.output作为下一轮输入，copy与计算顺序需要明确。若输入与输出部分重叠，某些逐元素运算可以安全原地执行，复杂算子却可能读到已覆盖值。固定地址不意味着任意别名都安全，仍要遵守算子合同。

真实图缓存通常持有长期缓冲，多个并发请求不能无保护地共享同一输入输出槽。可以为并发实例分配独立图与缓冲，或使用明确队列串行复用。一个图对象可重复调用不代表其静态buffer天然支持同时处理多个请求。

### 3. Bucket缓存键的完整定义

一个教学键可以包含设备标识、输入shape、dtype、stride、算子版本与配置。某个字段变化可能改变Kernel参数、地址解释或数学行为，不能继续使用旧图。只按元素总数缓存会把不同shape与stride混在一起，尤其归约轴不同的算子会产生错误。

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class ReplayKey:
    device: str
    shape: tuple
    dtype: str
    strides: tuple
    operation_version: int
    configuration: tuple

key_a = ReplayKey('cpu-simulation', (2, 3), 'float64', (3, 1), 1, ('axis', 1))
key_b = ReplayKey('cpu-simulation', (3, 2), 'float64', (2, 1), 1, ('axis', 1))
assert key_a != key_b
~~~

这里键不包含实际输入数值，因为基础重放允许内容变化；但若某个Python标量在capture时被固化为Kernel参数，改变它可能需要更新或重新捕获。应逐个区分“通过稳定buffer传入的数据”与“捕获时固定的执行配置”。

### 4. 图缓存需要容量与失效策略

每个bucket可能保留输入输出、工作区与图对象，缓存无限增长会占用显存。应记录每项大致内存、最近使用与模型版本，采用有限容量策略。淘汰时还要确认没有在途重放使用相关缓冲，不能仅从字典删除就认为设备已不再访问。

权重版本变化时，若地址保持且语义允许，可以更新内容；若布局、dtype或Kernel配置变化，通常需要重新验证图。实际可用更新机制依API与版本，教材不把所有变化都简化成“必须重建”或“都能原地更新”。先明确变化影响哪一层合同。

### 5. 数据依赖控制流的边界

普通Python根据GPU数据item结果选择分支，会涉及主机读取与同步，也不一定能按基础capture方式重放。高级框架与CUDA图可能提供条件节点或结构化控制流，但需要专门API和支持范围。不能把任意Python if自动视为图中动态分支。

学习时先把固定结构子图捕获，动态决策留在明确边界外，再研究受支持的条件机制。这样可以分别验证静态计算与动态调度，而不是把所有服务逻辑塞进一次capture后难以解释失败。

## 第十四课：延迟、吞吐、开销与真实可重叠性

### 1. 单块延迟与流水吞吐不同

加载四、计算八时，第一块从输入开始到输出完成需要十二个时间单位。稳定流水中相邻完成结果间隔可能接近八，但并不意味着每个请求延迟都变成八。排队、等待槽位与前序任务会影响每块实际等待。

三块总时间二十八，平均每块九点三三；一百块理想总时间八百零四，平均八点零四，逐渐接近最慢阶段八。这个趋势说明填充排空成本被更多任务摊薄，但服务系统不一定总有足够批量，也不能无限排队来追求吞吐。

~~~python
def ideal_pipeline_time(items, load, compute):
    if items < 0 or load < 0 or compute < 0:
        raise ValueError('nonnegative values required')
    return 0.0 if items == 0 else load + compute + (items - 1) * max(load, compute)

assert ideal_pipeline_time(3, 4, 8) == 28
assert ideal_pipeline_time(100, 4, 8) == 804
assert ideal_pipeline_time(0, 4, 8) == 0
~~~

### 2. 两个stream为何可能更慢

两个Kernel可能争用同一显存带宽、缓存或计算资源，重叠后各自变慢；额外事件与提交也有成本。若单个Kernel已经充分利用设备，增加并发未必有剩余资源可用。正确结果加“没有加速”仍是有效实验结论。

复制与计算重叠也需要硬件与内存条件，且可能共享带宽资源。理想模型把两个阶段视为独立服务台，真实设备不一定完全独立。测量时应比较串行与流水的完整时间线，检查等待与重叠，而不是只凭stream数量宣布并行。

### 3. Graph捕获成本何时能摊薄

设捕获与初始化额外成本为C，普通每次执行主机开销为H，重放开销为h，设备工作相同。重复R次时，图方案相对节省约R乘(H减h)减C。只有重复次数足够大且每次节省为正，捕获才可能在这份简化账本上有收益。

例如纯教学假设捕获成本三十毫秒，每次减少十微秒，需要约三千次重放才能摊薄，不是一次调用就自动更快。若图方案还增加输入复制或padding，必须把这些成本加入每次执行，临界次数可能更高甚至永远不划算。

这些数字只是量纲演示，不是任何GPU或框架实测。真正报告应测量捕获、首次重放、稳态重放与完整请求路径，并说明缓存命中率。只展示最好一次热重放，会遗漏服务首次请求与新shape的成本。

### 4. 过度同步怎样破坏流水

如果每次提交load后立即device synchronize，再提交compute后又同步，就把原本可重叠的不同批次强行串行化。同步可以用于定位错误，但正式流水应让设备依赖表达顺序，主机只在需要结果或资源复用时等待。

反过来，完全删除同步也不对。最终CPU读取输出、复用pinned源、淘汰图缓存等操作都需要完成证据。正确目标是最小必要依赖，不是同步数量越少越好。每条同步都应能回答保护哪一次读写或生命周期。

### 5. 时间线中的空白也需要解释

GPU时间线上没有Kernel运行的空白，可能来自CPU提交不足、依赖等待、内存分配、同步、编译或数据准备。不能看到空白就立刻增加stream。先查空白前后事件与主机活动，确定瓶颈属于哪一层。

若设备持续忙但请求延迟仍高，可能是队列积压或单任务太大；若设备间歇空闲，可能是主机或传输跟不上。吞吐、利用率与延迟相互关联，却不是同一个指标。K14会用更系统的性能方法继续分析。

### 6. 有界实验与可停止设计

实验脚本应有明确输入大小、迭代次数和超时或中断策略。停止提交新任务后，等待已提交工作完成，再释放缓冲和事件。不要通过强行复用仍在途的资源来“快速停止”，那会把控制逻辑变成数据竞争。

在共享H100上先运行单设备、小规模、短时实验，确认正确性后再扩大到真正需要的规模。CPU模拟与公式推导已经完成大部分依赖验证，GPU阶段应带着具体问题执行，例如是否有复制计算重叠、图重放减少多少提交开销，而不是长时间盲跑。

### 7. 最终审查的四张表

第一张表列每个buffer的生产者、消费者与最后使用者；第二张表列每个event记录在哪个stream的哪条任务之后；第三张表列每个缓存对象的shape、地址与生命周期；第四张表列计时包含的操作。四张表互相对应，能把异步程序从“看起来会按顺序”变成明确合同。

如果某个buffer找不到释放前的最后完成证据，或某个等待找不到对应生产者，就继续追踪，不要靠经验猜测。能完成这份审查，通常已经具备阅读复杂推理运行时的关键基础。


### 8. 多stream计时需要一个明确的汇合点

如果两条独立stream同时工作，只在其中一条记录stop，不能证明另一条已经结束。要测整批完成时间，可以让一个汇合stream等待两条完成event，再记录最终stop；或者主机等待两条完成后测端到端时间。计时终点必须覆盖你声称包含的全部工作。

同样，开始event应与两条分支建立共同起点，否则某个分支可能在计时开始前已执行一部分。多stream计时不是把单stream的两条event随便放到不同流就结束，需要明确共同开始与共同完成的依赖图。

~~~python
joined_plan = StreamPlan()
start_marker = joined_plan.event('start', 'join')
left_done = joined_plan.submit('left', 'left-stream', 5, (start_marker,))
right_done = joined_plan.submit('right', 'right-stream', 8, (start_marker,))
stop_marker = joined_plan.submit('stop', 'join', 0, (left_done, right_done))
joined_times = joined_plan.earliest_times()
assert joined_times[stop_marker] == (8.0, 8.0)
~~~

这个理想模型中两分支独立资源并行，整批时间为最大值八，不是总和十三。若真实硬件不能重叠，实测可能接近十三或更高；依赖图只说明允许并行，不承诺资源足够。把模型与测量放在一起，才能解释差异来自语义还是硬件。

### 9. 同一存储的不同切片仍共享生命周期

两个stream处理数组不同切片，数据访问可以互不冲突，但底层分配通常仍是一块storage。释放或整体复用必须等所有切片消费者完成。不能因为某个切片对象离开作用域，就认为对应物理区域已经独立释放。

如果下一轮只覆盖其中一个已完成切片，可以设计更细粒度的区域所有权，但需要明确范围与依赖。框架分配器通常按分配对象管理，不一定理解你自定义的子区域协议。学习阶段先保持整buffer生命周期清楚，再考虑区域级复用优化。

### 10. Event查询与忙等待

查询event是否完成可以让主机继续处理其他工作，但在紧密循环中不断查询会消耗CPU，并可能影响提交线程。应根据任务粒度选择合理轮询、阻塞等待或事件驱动组织。固定长sleep也不理想，因为会增加不必要延迟且仍不是完成证明。

查询结果未完成时，不能读取输出或复用缓冲；完成后才能推进相应状态。把查询当作性能提示、同时无条件消费数据，会让协议失去意义。控制逻辑应像状态机一样明确每个分支允许哪些操作。

### 11. 结果正确也要检查资源是否泄漏

反复创建stream、event、图对象或pinned缓冲而不释放，短测试可能正确，长运行却逐渐耗尽资源。资源缓存可以减少创建成本，但必须有容量与生命周期策略。性能优化不应靠无限保留对象换取一次漂亮计时。

在有界实验中记录开始与结束的对象数量或显存占用，确认重复运行不会持续增长。若框架缓存导致内存保留，区分可复用缓存与真正泄漏，按官方接口观察，不要直接把所有保留显存都判为泄漏。理解资源所有者与最后使用时刻，才能做出准确判断。


## 第十五课：综合项目——可证明的两阶段异步流水线

先实现CPU依赖计划与错误交错，证明数据ready和buffer free两条边都必要；再实现两槽加载计算流水，输出每块开始结束时间与槽位代际。第三阶段在GPU上用小数组验证跨流数值与生命周期，保持有限重复与显存预算。

第四阶段把固定算子链捕获为图，比较eager与重放结果，分别报告提交、设备与端到端时间。第五阶段加入两个shape bucket，测量padding与缓存成本。每次优化先保持正确性，再讨论速度。

完成标准是你能回答任何一个buffer在某时刻由谁拥有、谁正在读写、何时可复用，以及某次计时究竟包含哪些依赖。掌握这些问题，才具备修改真实异步推理与训练运行时的基础。

本章API边界核查自[NVIDIA异步执行指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)与[PyTorch CUDA语义](https://docs.pytorch.org/docs/2.15/notes/cuda.html)。实际运行时记录安装版本，特别核对默认stream、跨流存储使用与图捕获限制；这里的CPU任务模型只验证依赖逻辑，不替代设备内存模型、驱动行为或真实性能测量。

本章代码按出现顺序执行，后面的流水线实验复用前面定义的任务与依赖计划类。若拆成独立文件，应显式导入这些定义，并保留有意遗漏复用依赖的负向案例；只运行正确计划无法证明检查器真的能够发现竞态。
