# F11 并发、线程、进程与 asyncio：让任务重叠推进，同时保持状态可解释

> 先修 F04、F07–F10。离线实验：`python -m labs.foundations.concurrency`。本章先用事件和状态表理解调度，不用毫秒级计时断言制造“快了几倍”的结论。asyncio 的版本细节以实际 Python 与官方文档为准，代码采用常见 Python 3.11 及以上接口，仓库实测环境另见验证记录。

## 1. 先拆开运行、等待和完成

一个任务不一定始终占用 CPU 计算。发出网络请求后，它可能等待数据；读取磁盘时可能等待设备；等待锁时可能等待别的任务。串行程序会先完成一个任务再开始下一个，即使第一个正在等待，其他任务也未必得到机会。并发让多个任务的生命周期重叠推进，使等待期间可以做别的有用工作。

并行描述某一时刻有多个执行资源真正同时工作；并发描述多个任务交错或重叠地取得进展。一个事件循环线程可以并发管理很多等待，却不会因此让两段纯 Python 计算在同一个核上同时执行。多个 CPU 核或 GPU 执行单元可以提供并行能力，但如何使用还取决于进程、线程、运行时和底层库。

手算两个任务，各等待四个时间单位再计算一个单位。串行总计十。若两次等待都在起点开始，四单位后依次做两个计算，总计六。这里假设一个计算资源、等待完全重叠且没有额外开销；不能直接写总计五，因为两个计算仍要占用同一个资源。真实耗时还包括调度、网络和数据处理，手算是理解机制，不是性能实测。

任务开始、任务完成与结果消费也不同。某任务可能先完成，但调用者为了保持输入顺序稍后才处理它；另一个任务可能还在排队，根本没开始远端请求。日志应区分 queued、started、completed，而不是只记录一个笼统 timestamp。后续分析 Agent 工具延迟时，这些阶段会帮助区分等待队列和实际执行。

## 2. 线程、进程和协程各保存什么状态

进程拥有相对独立的地址空间与资源视图。两个 Python 进程各自的普通列表不会因为变量同名就共享，通信需要管道、队列、共享内存、文件或网络。进程隔离较强，启动与数据传输也有成本。把大型对象传给子进程可能涉及序列化和复制，不能只看到多核就忽略传输开销。

线程在同一进程中共享许多内存对象，因此交流方便，也更容易发生共享状态错误。一个线程修改列表，另一个可能看到变化；多个线程读取后再写入同一计数，可能丢更新。锁用于协调某段操作的互斥，不会自动让所有对象访问都正确。程序必须明确哪些状态受哪个锁保护，并保持一致纪律。

协程是能暂停和恢复的计算过程。asyncio 通常在事件循环线程里协作调度 Task，只有任务到达可挂起点并实际让出控制，其他任务才有机会推进。普通长循环没有 await，可能阻塞整个循环。把函数声明成 async 并不自动把它送到后台线程，也不自动让内部 CPU 工作并行。

常见默认 CPython 构建中的 GIL 限制多个线程同时执行 Python 字节码，但原生扩展可能释放它，也存在不同构建与解释器实现。不要把“线程永远不并行”当普遍定理。选并发方式时先判断主要工作是 Python 计算、原生库计算、网络等待还是外部进程，再按实际环境验证。

## 3. async def、协程对象与 await

调用普通函数通常立即执行函数体直到返回；调用 async def 定义的函数会得到协程对象，函数体需要被等待或调度才推进。忘记 await 常导致“我调用了函数却没有结果”，甚至出现协程从未等待的警告。协程对象不是最终返回值，也不应该直接被当成数字或 JSON 数据。

```python
import asyncio

async def value():
    return 3

async def main():
    result = await value()
    assert result == 3

asyncio.run(main())
```

asyncio.run 是脚本入口：建立并运行事件循环，等待主协程结束，完成相应清理。已经有运行中事件循环的环境，如某些交互笔记本，不应再随意嵌套 run；应使用该环境支持的等待方式。教程中的脚本示例默认从普通 Python 进程运行，环境前提必须明确。

await 的意思是等待一个可等待对象完成，并允许在它尚未就绪时挂起当前协程。不是每个 await 都必然切换任务：如果被等待结果已就绪，可能继续执行。因此不能靠“代码里有 await”猜测确切交错顺序。要设计可重复实验，应使用 Event、Queue 等明确同步点控制进展。

```python
import asyncio

async def main():
    events = []
    async def step(name):
        events.append(name + ":start")
        await asyncio.sleep(0)
        events.append(name + ":end")
        return name
    first = await step("A")
    second = await step("B")
    assert [first, second] == ["A", "B"]
    assert events == ["A:start", "A:end", "B:start", "B:end"]

asyncio.run(main())
```

这里仍然串行，因为第二个 step 要等第一个 await 完成后才开始。sleep(0) 允许当前任务让出执行权，但没有第二个任务被创建，当然不会凭空产生并发。要让两个协程同时处于可推进状态，需要 Task 或 gather 等调度方式。

## 4. Task、gather 与顺序合同

Task 把协程交给事件循环管理，记录其执行状态、结果或异常。create_task 建立一个任务后，调用者仍需保存引用并在合适位置等待它；不要发出任务后忘记谁负责观察错误和结束。所谓后台任务也有生命周期责任，否则主流程结束时可能留下未处理异常或未完成副作用。

```python
import asyncio

async def main():
    gate = asyncio.Event()
    started = []
    async def work(name):
        started.append(name)
        await gate.wait()
        return name.upper()
    tasks = [asyncio.create_task(work("a")), asyncio.create_task(work("b"))]
    await asyncio.sleep(0)
    assert set(started) == {"a", "b"}
    gate.set()
    assert await asyncio.gather(*tasks) == ["A", "B"]

asyncio.run(main())
```

两个任务都到达 gate 等待，主任务观察已开始后打开门。gather 收集结果，返回列表按传入等待对象顺序排列，而不是按完成顺序。星号把 tasks 列表展开为多个参数；它不是把列表内容复制成多个线程。Task 的并发推进仍由同一个事件循环管理。

gather 默认在某个任务出现异常时向等待者传播异常，但不等于自动取消所有其他任务。调用者必须理解剩余任务的生命周期，必要时显式取消并等待清理，或使用具有不同失败管理合同的 TaskGroup。不能从“外层收到异常”推断全部子任务已经停止，更不能立即释放它们还在使用的共享资源。

输出顺序必须与业务标识绑定。批量请求中若按完成顺序收集，却按输入顺序解释答案，可能把答案配错题而不产生任何类型错误。携带 index 或 request_id 能避免这种错配。仓库 squares 为每项返回索引和值，再排序恢复输入顺序；虽然 gather 已保留输入顺序，这个显式标识仍用于教学强调关联关系。

## 5. 逐行追踪仓库信号量实验

squares 首先验证 concurrency 必须为正整数，拒绝 bool 与非正数。随后建立 Semaphore 和 Meter。Meter 保存当前进入受控区域的任务数 active，以及历史峰值 peak。内层 one 接收输入位置和数值，在 `async with semaphore` 内取得许可，增加 active，更新 peak，然后暂停一次并计算平方。

`async with` 允许进入或退出协议包含等待。信号量没有许可时，任务等待而不忙循环；离开区域时释放许可。finally 中 active 减一，保证正常返回、异常或取消都能更新观察计数。计数只是教学监测，不是信号量内部许可本身；两者必须在相同生命周期边界保持一致。

| 阶段 | A | B | C | active |
|---|---|---|---|---:|
| 取得两个许可 | 进入 | 进入 | 等待许可 | 2 |
| A 完成 | 释放 | 尚未释放 | 可获许可 | 不超过 2 |
| C 进入 | 已结束 | 运行或结束 | 进入 | 不超过 2 |
| 全部结束 | 完成 | 完成 | 完成 | 0 |

具体 A、B、C 谁先恢复不应作为公开合同，表格表达的是许可上限与结束状态。测试应断言 peak 不超过 concurrency，active 最终为零，结果与输入一一对应。不要断言某个任务必须在恰好某个微秒完成；操作系统负载变化不该让功能测试随机失败。

```python
import asyncio
from labs.foundations.concurrency import squares

async def main():
    for limit in [1, 2, 4]:
        result, meter = await squares(range(6), limit)
        assert result == [0, 1, 4, 9, 16, 25]
        assert 0 <= meter.peak <= limit
        assert meter.active == 0

asyncio.run(main())
```

信号量限制的是它包围区域内的同时活动数。如果你在进入前就发出网络请求，它不能限制请求并发；如果在退出后仍持有大量结果，它不能限制结果内存。控制范围必须与真正想限制的资源一致。读源码时画出许可取得和释放位置，比只看到 Semaphore 名字更有用。

## 6. 一个线程也会有竞态：读、等待、写

竞态不是只有多核才会发生。两个协程共享一个计数器，各自先读旧值，再 await，最后根据旧值写回。它们都可能读到零，各自写一，最终只增加一次。问题来自逻辑操作被拆成多个阶段，中间允许其他任务观察和修改状态。

```python
import asyncio

async def main():
    state = {"count": 0}
    both_read = asyncio.Event()
    readers = 0
    async def increment():
        nonlocal readers
        old = state["count"]
        readers += 1
        if readers == 2:
            both_read.set()
        await both_read.wait()
        state["count"] = old + 1
    await asyncio.gather(increment(), increment())
    assert state["count"] == 1

asyncio.run(main())
```

Event 保证两个任务都已读取后才允许写入，使丢更新稳定复现。nonlocal 表示修改外层函数中的 readers 绑定，不是在内层创建同名局部变量。这个例子没有依赖真实网络和随机时序，所以适合教学与回归测试。它证明交错机制，不是证明所有普通自增在所有解释器中必然这样交错。

修复可以用 asyncio.Lock 包住完整的读改写区间，或让一个唯一任务拥有状态并通过队列接收更新。锁只对遵守同一锁的协程有效，不能自动阻止另一个线程或进程绕过它。不要在持锁期间做无关长等待，否则其他任务都被阻塞；但若业务原子操作确实跨越等待，就需要重新设计状态与提交边界。

```python
import asyncio

async def main():
    state = {"count": 0}
    lock = asyncio.Lock()
    async def increment():
        async with lock:
            old = state["count"]
            await asyncio.sleep(0)
            state["count"] = old + 1
    await asyncio.gather(increment(), increment())
    assert state["count"] == 2

asyncio.run(main())
```

这个修复故意保留 await，让互斥作用可见。真实计数加一通常不需要在临界区等待；教学代码用它展示锁保护的范围。若只锁写入而把读取放在锁外，两个任务仍可能使用同一个旧值，因此“加了锁”不是充分说明，必须说明锁包围哪段不变量维护过程。

## 7. 有界队列与背压：限制的不只是正在执行

仓库 squares 为所有输入建立等待任务，再用信号量限制活动区域。输入一百万项时，虽然只有两个任务在区域内，等待的任务对象仍可能很多。完整内存上限还要控制待处理队列、结果缓存、连接池和输入本身。并发上限与队列上限是不同维度。

有界 Queue 在满时让生产者等待，消费者处理后腾出空间，上游才能继续。这叫背压：下游处理不过来时，上游减速，而不是无限堆积。固定数量 worker 从队列取任务，使任务对象数量与 worker 数相关，而不是与全部输入数量相关。若最终仍把所有结果保存在列表，总结果内存仍随输入增长，需要诚实说明。

```python
import asyncio

async def queued_squares(values, workers=2, capacity=4):
    if type(workers) is not int or workers <= 0 or type(capacity) is not int or capacity <= 0:
        raise ValueError("positive workers and capacity required")
    queue = asyncio.Queue(maxsize=capacity)
    stop = object()
    results = {}
    peak_queue = 0
    async def worker():
        while True:
            item = await queue.get()
            try:
                if item is stop:
                    return
                index, value = item
                await asyncio.sleep(0)
                results[index] = value * value
            finally:
                queue.task_done()
    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    try:
        for index, value in enumerate(values):
            await queue.put((index, value))
            peak_queue = max(peak_queue, queue.qsize())
        for _ in range(workers):
            await queue.put(stop)
        await queue.join()
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return [results[index] for index in sorted(results)], peak_queue

async def main():
    result, peak = await queued_squares(range(20))
    assert result == [value * value for value in range(20)]
    assert peak <= 4

asyncio.run(main())
```

这份基本队列演示的输入约定是可平方数值，工作阶段不会主动制造失败。若 worker 意外失败，而生产者仍在等待满队列，就可能需要更强监督才能及时结束；finally 只能在控制流离开 try 时清理，不能自动让一个正卡住的生产者退出。后面的结构化并发版本解决这个故障传播问题，不能把当前演示直接称为任意失败下完整可靠的队列系统。

每次 get 都必须对应一次 task_done，包括哨兵。join 等待未完成计数归零，不是等待队列当前长度为零；worker 已取走但尚未完成的任务仍计入未完成数。哨兵个数与 worker 数匹配，每个 worker 收到一个便结束。只放一个哨兵可能只结束一个 worker，其他仍永久等待。

## 8. 取消是请求停止，不是时光倒流

task.cancel 请求向任务注入取消，通常在可挂起点体现为 CancelledError。它不等于立即杀死任意正在执行的 Python 指令，也不自动撤销已经写入文件或发出的网络请求。取消处理应释放本地资源，并通常继续传播取消，使父任务知道操作未正常完成。

```python
import asyncio

async def main():
    started = asyncio.Event()
    cleaned = asyncio.Event()
    async def work():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    task = asyncio.create_task(work())
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert cleaned.is_set()
    assert task.cancelled()

asyncio.run(main())
```

started 确认任务确实进入主体后再取消，避免测试只取消了一个尚未开始的任务，因而没有执行预期清理。await task 负责等清理结束并观察取消结果；只调用 cancel 后立刻检查状态，可能在清理尚未运行时得出错误结论。取消是一个过程，需要明确等待终点。

不要用过宽捕获吞掉取消后继续无限循环。某些 asyncio 高层机制依赖取消完成结构化清理，吞掉它会使超时和任务组行为出乎意料。若业务确实需要屏蔽某段操作，应先理解 shield 等工具的精确合同，以及外层取消后谁继续等待内部任务；不要把屏蔽当作让程序“永远不失败”的开关。

外部副作用尤其需要独立恢复设计。请求可能已在远端完成，只是本地等待被取消；重试创建任务可能造成重复。稳定幂等键、查询原请求结果和明确状态机，才能处理这种不确定性。asyncio 负责本地协作调度，不会让两个系统突然共享一个事务。

## 9. 超时、截止时间与重试预算

单步超时限制一次等待，整体截止时间限制整个任务。两步各允许八秒，不代表整体十秒预算自动成立；第一步用了七秒，第二步最多只能使用剩余三秒。用单调时钟计算剩余预算，取单步上限与剩余时间的较小值，才能避免每一层重新获得完整预算。

wait_for 在超时时通常取消被等待操作，并等待其取消处理；因此实际返回时间可能超过名义超时，尤其当清理需要时间或任务不及时响应取消。超时不是操作系统强制杀进程。CPU 长循环没有让出执行权时，事件循环也无法及时执行超时回调，需要把计算放到适当执行资源或加入可控分段。

重试还要有最大次数、总截止与错误分类。非法 JSON 或缺字段通常不是等待一下就会修好；短暂连接失败可能适合有限重试；带写入的请求必须考虑幂等。每次重试都重新生成幂等键会失去去重意义，因为远端看到的是不同请求。键应绑定同一个逻辑操作，而不是某一次传输尝试。

并发数、速率与 token 预算不同。并发四表示最多四个在途操作，短请求仍可能在一秒内发出很多次；速率限制控制时间窗口内启动数量；token 预算控制内容规模与费用。一个信号量不能同时证明三个约束都满足。后续 Agent 的调度应分别维护这些资源，避免只加 concurrency 参数就宣称成本可控。

## 10. 结构化并发与失败监督

结构化并发让子任务的生命周期属于一个明确父作用域：离开作用域前等待子任务完成，失败时按规定取消兄弟任务并收集异常。Python 的 TaskGroup 提供这类机制，适合避免启动许多任务后忘记处理剩余生命周期。它与 gather 的失败语义不同，选择时必须知道你要的是全部独立结果还是共同成功的一组子任务。

```python
import asyncio

async def main():
    cleaned = asyncio.Event()
    started = asyncio.Event()
    async def waiting():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    async def failing():
        await started.wait()
        raise ValueError("planned failure")
    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(waiting())
            group.create_task(failing())
    except* ValueError:
        pass
    assert cleaned.is_set()

asyncio.run(main())
```

except* 用于处理异常组中的匹配异常，不是普通 except 的装饰写法。任务组可能汇总多个失败，因此保留组结构能避免只看到第一个错误。初学时先掌握这个小例子：一个任务失败，另一个等待任务被取消，finally 执行，作用域退出后清理已完成。不要在还不理解时把复杂异常组处理散布整个项目。

对队列系统，可以把生产者和固定 worker 一起放进 TaskGroup。任一 worker 失败会触发组内其他任务取消，包含可能正在 queue.put 等待的生产者，从而避免前面基本版本的挂起窗口。队列未完成计数与业务失败记录仍要认真设计，任务组只解决生命周期监督，不自动提供重试、持久化或 exactly-once 处理。

## 11. 四个故障诊断

**async 函数里所有任务仍像串行。** 检查是否逐个 await 后才创建下一个，或内部使用了阻塞 sleep、同步网络请求和长 CPU 循环。记录任务开始与挂起事件，区别没有调度与调度后无法让出。不要只增加更多 Task，因为阻塞事件循环的核心原因不会因此消失。

**活动计数永远不归零。** 检查减计数是否只放在正常返回路径，取消或异常时是否绕过。把资源释放与计数恢复放进 finally 或上下文协议，并等待所有任务清理完成后再观察。观察太早也可能误判，所以测试必须有明确结束边界。

**queue.join 永远等不到。** 检查每个 get 是否恰好对应一次 task_done，哨兵是否也确认完成，worker 是否提前异常退出。队列长度零不代表没有未完成任务。用小输入和事件日志列出取出、处理、确认三步，通常可以定位缺失确认或重复确认。

**取消后远端仍创建了任务。** 这是外部副作用不受本地取消回滚的正常可能性。检查请求是否已发送、是否有稳定幂等键、是否能查询结果。修复协议与恢复流程，而不是简单认为 cancel 失效或增加更多盲目重试。

## 12. 十二道练习与完整参考答案

### 一：串行 await 的顺序

```python
import asyncio
async def main():
    events = []
    async def one(name):
        events.append(name)
        await asyncio.sleep(0)
        return name
    assert await one("a") == "a"
    assert await one("b") == "b"
    assert events == ["a", "b"]
asyncio.run(main())
```

第二次调用尚未创建前，第一次必须完成，所以这里没有两个并发工作任务。async 语法本身不改变调用依赖关系。

### 二：并发上限不改变结果

```python
import asyncio
from labs.foundations.concurrency import squares
async def main():
    for concurrency in [1, 2, 4]:
        values, meter = await squares(range(10), concurrency)
        assert values == [x * x for x in range(10)]
        assert meter.peak <= concurrency
        assert meter.active == 0
asyncio.run(main())
```

测试检查资源不变量而非速度倍数。即使共享机器很忙，只要调度最终完成，正确结果与上限仍应成立。

### 三：非法并发必须拒绝

```python
import asyncio
from labs.foundations.concurrency import squares
async def main():
    for value in [0, -1, True, 1.5]:
        try:
            await squares([1], value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
asyncio.run(main())
```

检查发生在协程执行时，所以必须 await。只创建协程对象就期待同步异常，会把验证时间理解错。

### 四：gather 按输入顺序返回

```python
import asyncio
async def main():
    release_first = asyncio.Event()
    completed = []
    async def first():
        await release_first.wait()
        completed.append("first")
        return 1
    async def second():
        completed.append("second")
        release_first.set()
        return 2
    assert await asyncio.gather(first(), second()) == [1, 2]
    assert completed == ["second", "first"]
asyncio.run(main())
```

事件让完成顺序确定为第二个先，但结果列表仍按参数顺序。这个对照能直接区分两种顺序概念。

### 五：锁保护完整更新

```python
import asyncio
async def main():
    total = 0
    lock = asyncio.Lock()
    async def add():
        nonlocal total
        async with lock:
            old = total
            await asyncio.sleep(0)
            total = old + 1
    await asyncio.gather(*(add() for _ in range(20)))
    assert total == 20
asyncio.run(main())
```

读和写都在同一个临界区。把读取移到锁外会改变合同，即使写入仍互斥，也可能覆盖其他更新。

### 六：取消之后等待清理

```python
import asyncio
async def main():
    started = asyncio.Event()
    events = []
    async def work():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            events.append("closed")
    task = asyncio.create_task(work())
    await started.wait()
    task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert events == ["closed"]
asyncio.run(main())
```

return_exceptions=True 在这里用于收集已知取消结果，不代表应该把所有业务错误静默忽略。调用者仍检查结果类型与清理事件。

### 七：队列确认与结束

```python
import asyncio
async def main():
    queue = asyncio.Queue(maxsize=2)
    await queue.put("one")
    item = await queue.get()
    assert item == "one"
    queue.task_done()
    await queue.join()
    assert queue.empty()
asyncio.run(main())
```

这个最小例子只证明取出与确认的配对。真实 worker 应把 task_done 放在 finally，保证处理异常时未完成计数也得到正确管理。

### 八：用唯一状态所有者累计

```python
import asyncio
async def main():
    queue = asyncio.Queue()
    stop = object()
    async def owner():
        total = 0
        while True:
            item = await queue.get()
            try:
                if item is stop:
                    return total
                total += item
            finally:
                queue.task_done()
    task = asyncio.create_task(owner())
    for value in [1, 2, 3]:
        await queue.put(value)
    await queue.put(stop)
    assert await task == 6
    await queue.join()
asyncio.run(main())
```

只有 owner 修改 total，因此无需多个任务争同一计数器锁。队列成为消息边界，但若生产者数量很大，仍应设置容量。

### 九：总截止时间的剩余预算

```python
def remaining_step_budget(now, deadline, per_step_limit):
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError("deadline reached")
    return min(remaining, per_step_limit)
assert remaining_step_budget(0.0, 1.0, 0.8) == 0.8
assert abs(remaining_step_budget(0.7, 1.0, 0.8) - 0.3) < 1e-12
```

纯函数先验证预算计算，不依赖真实等待。真正异步调用再把这个结果传入超时机制，并使用同一单调时钟体系。

### 十：失败前后的本地副作用

```python
import asyncio
async def main():
    records = []
    async def work():
        records.append("sent")
        raise ValueError("response lost")
    try:
        await work()
    except ValueError:
        pass
    assert records == ["sent"]
asyncio.run(main())
```

异常没有撤销已经发生的 append。网络发送也是类似原则，只是远端结果更难观察；恢复协议必须承认这种不确定性。

### 十一：不要阻塞事件循环

```python
import asyncio

def blocking_calculation(value):
    return sum(range(value))

async def main():
    result = await asyncio.to_thread(blocking_calculation, 100)
    assert result == 4950

asyncio.run(main())
```

to_thread 把同步函数交给线程执行，能避免该调用直接占住事件循环；它不承诺纯 Python CPU 运算因此获得多核加速。取消等待也不保证已经运行的线程立刻停止，真实任务仍要设计协作停止。

### 十二：失败任务要被观察

```python
import asyncio
async def fail():
    raise ValueError("planned")
async def main():
    task = asyncio.create_task(fail())
    try:
        await task
    except ValueError as error:
        assert str(error) == "planned"
    else:
        raise AssertionError("failure must propagate")
asyncio.run(main())
```

保存并等待 Task 让异常回到明确调用者。不能只创建任务后丢弃引用，再希望日志偶尔出现的未检索异常足以承担错误处理。

## 13. 把事件循环想清楚：谁在等待，谁能继续

事件循环维护可运行任务与等待事件的关系。一个协程等待未设置的 Event 时，它的执行状态被保留，但不会不断占用 CPU 检查标志；事件设置后，等待者获得再次被调度的机会。这里的“机会”不等于立刻执行，也不保证它先于所有其他就绪任务。程序应依赖事件已满足的事实，而不是依赖某种未经承诺的绝对调度次序。

可以把一个 Task 的粗略状态写成未完成、已完成、已取消，并进一步观察未完成任务当前在运行还是等待。应用不应通过读取内部状态来强行安排业务顺序，而应使用公开同步机制。Task 完成后有结果或异常，调用者需要取得它；如果异常无人观察，即使事件循环最终打印警告，也已经失去清楚的恢复位置。

await 形成依赖关系：父任务暂时不能继续，因为它需要某个结果。create_task 则建立新的可推进工作，但不自动让父任务等待完成。把二者混淆会产生两类相反错误：只 await 导致意外串行，只有 create_task 导致主流程过早结束。阅读异步代码时，在纸上分别画“创建边”和“等待边”，就能看见哪些任务真正属于当前工作流。

以两个模型请求为例，先创建 A，再创建 B，最后 gather，二者可以重叠等待；先 await A 得到结果，再创建 B，B 的开始依赖 A 完成。若 B 的提示确实需要 A 的输出，串行是业务必需，不是性能缺陷。不能为了并发而在输入尚未确定时提前请求，最后又把无关结果当作正确推理链的一部分。

事件循环中的阻塞调用像把唯一调度员困在一项工作里。同步 time.sleep 不会告诉 asyncio “我等待期间可以处理其他任务”，它阻塞当前线程；大规模纯 Python 循环同样如此。await asyncio.sleep 则把等待交给事件循环。两者都可能看起来是“等一秒”，但对其他任务的影响完全不同，这正是异步编程的核心边界之一。

异步库需要整条调用链配合。如果外层 async 函数内部调用同步 HTTP 客户端，网络等待仍可能阻塞循环。可以选择真正异步客户端，或在明确边界把同步调用送到线程；后者还要考虑线程池容量、取消与客户端线程安全。不要只给现有函数加 async 关键字就宣布已经完成异步改造。

## 14. 线程基础：共享内存、锁与线程池

线程适合把某些阻塞 IO 与事件循环分开，也常用于依赖内部释放 GIL 的原生库工作。线程创建有成本，所以线程池复用有限数量线程。提交任务返回 Future，Future 表示尚未完成或已完成的结果；它与普通返回值不同，需要 result 或相应等待机制取得。提交很多任务不等于所有任务立刻执行，池内也可能有等待队列。

```python
from concurrent.futures import ThreadPoolExecutor

def square(value):
    return value * value

with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(square, value) for value in range(4)]
    result = [future.result() for future in futures]
assert result == [0, 1, 4, 9]
```

with 管理线程池生命周期，退出时按其合同等待并清理。max_workers 限制执行线程数量，不自动限制你提交多少 Future；如果一次提交巨量输入，等待任务仍占内存。这个问题与信号量版本相似：活动上限不能替代排队上限。真正的大数据流水线应按容量逐步提交或使用有界生产消费机制。

线程共享普通对象，因此需要线程锁而不是 asyncio.Lock 来协调跨线程临界区。asyncio 的同步原语主要用于事件循环任务，不应当作通用线程锁。反过来，在事件循环线程里长时间等待 threading.Lock，会阻塞整个循环。不同并发体系的同步工具有不同调度机制，名字都叫 Lock 不代表可以任意互换。

锁保护的是协议，不是对象自带魔法。约定对余额的所有读改写都持同一把锁，才能维护一致性；如果某条代码绕过锁，其他路径再严格也没用。读操作是否需要锁取决于你要求看到什么一致状态。例如两个字段必须共同变化，读取者若只读其中一半更新，就可能看到业务上不可能的组合。

不要在持锁时调用不可控外部代码，因为它可能很慢、再次请求同一锁或调用另一个组件形成循环等待。可以在锁内复制必要状态，释放后执行外部工作，再在锁内验证状态是否仍适用并提交；这需要版本或条件检查，不能简单把所有锁移除。设计目标是缩短临界区，同时保留不变量。

线程任务取消具有边界。Future.cancel 通常只能阻止尚未开始的任务，已经运行的线程不能靠一个普通取消调用安全地任意终止。长任务应定期检查停止事件，或者将工作分块，允许在边界退出。强行终止可能让锁与资源处于不一致状态，因此协作停止比“把线程杀掉”更适合普通应用设计。

## 15. 进程基础：隔离、序列化与入口保护

进程池可以让不同进程使用多个 CPU 核执行 Python 计算，但函数和参数往往需要能被序列化并在子进程中重建。局部嵌套函数、打开的连接、锁和复杂客户端对象不一定适合直接传递。设计任务边界时尽量传简单数据或文件位置，返回明确结果，避免把整个运行中的应用对象图发送给子进程。

不同系统和 Python 配置使用的进程启动方式可能不同，不能假设 Linux 上某次行为在 Windows 完全相同。模块顶层启动进程池可能在子进程导入模块时再次启动，形成递归创建。把启动入口放在 `if __name__ == "__main__":` 内，确保子进程导入函数定义时不自动重复执行主流程，这是入口保护的实际用途。

本章不要求在所有平台真实执行进程池，因此不要把下面的设计说明当作 Win11 实测。你的本地练习应保存为独立文件，从新终端运行，记录启动方式、解释器和结果。共享服务器上使用有限 worker 数与小任务，避免为了验证基础机制占满全部 CPU。资源无限可用不等于练习必须无界扩张。

进程隔离意味着全局列表修改不会自动返回父进程。子进程 append 到自己的普通列表，父进程仍持有原来的对象副本或独立状态。必须通过返回值、队列或共享内存传递结果。初学者常把同名全局变量误认成跨进程共享，这是把线程经验错误套用到进程上。

序列化与传输可能抵消并行收益。一个只做几次加法的小任务，启动和传参成本可能比计算大；大型数组反复复制也会消耗内存带宽。先用操作规模与传输量估算，再做代表性实验。批量合并小任务可以减少调度开销，但批次太大又增加延迟与失败重做成本，需要按实际目标权衡。

进程崩溃后的恢复也不同于协程异常。内存中的未返回结果可能丢失，外部文件却可能已经部分写入。父进程需要知道哪些任务已完成、哪些结果已提交以及重试是否幂等。将“执行完成”和“结果可靠保存”分成明确阶段，是从简单并行脚本走向可靠任务系统的重要一步。

## 16. 锁、死锁与单一所有者

死锁的典型例子是任务甲持有锁 A 等锁 B，任务乙持有锁 B 等锁 A，双方都不能继续。仅给锁等待加超时可以让程序最终失败，但没有自动修复资源顺序。更好的设计是统一获取顺序、避免嵌套持锁，或重新组织状态所有权，让需要共同修改的数据由一个组件维护。

同一协程重复获取一把非重入锁也可能把自己锁住。你看到的不是另一个任务太慢，而是当前任务等待自己尚未释放的许可。阅读代码时追踪锁的取得和释放路径，尤其注意函数内部是否再次调用会获取同一锁的函数。把隐藏锁逻辑封装得太深，会让调用者无法看见死锁风险。

单一所有者模式让一个任务独占某份可变状态，其他任务通过消息请求更新。它减少共享修改，但并不免费：所有更新可能串行经过该任务，消息队列需要容量与失败策略，所有者崩溃时也要恢复。选择它的理由是状态规则更容易集中，而不是它能自动提供无限吞吐或持久化。

读多写少的场景还可以发布不可变快照。写者构造完整新状态后替换引用，读者使用自己取得的版本；这样避免在读取过程中看到半更新。但嵌套容器必须真正按约定不再修改，否则所谓快照仍会变化。F07 的浅拷贝与深层别名知识在这里直接影响并发正确性。

无论使用锁还是消息，先写不变量。例如活动数不能负、预算不能超限、每个任务最终只能有一个已提交结果。随后列出哪些操作可能同时发生，逐个证明它们如何维护不变量。仅凭“这段很短”或“Python 有 GIL”不能替代业务原子性分析，因为等待、原生调用和未来重构都可能改变交错点。

## 17. 故障可控的队列小项目：生产者与 worker 共同受监督

这一阶段把生产者和 worker 放进同一个 TaskGroup。输入只能是有限可迭代对象，worker 数与队列容量为正整数；结果按输入位置返回，工作失败则整个调用失败并清理其他任务。例子仍将所有结果保存在字典，因而结果内存随输入增长，不能宣称全流程常数内存。

```python
import asyncio

async def supervised_map(values, operation, workers=2, capacity=4):
    if type(workers) is not int or workers <= 0 or type(capacity) is not int or capacity <= 0:
        raise ValueError("positive bounds required")
    queue = asyncio.Queue(maxsize=capacity)
    stop = object()
    results = {}
    peak = 0

    async def produce():
        nonlocal peak
        for index, value in enumerate(values):
            await queue.put((index, value))
            peak = max(peak, queue.qsize())
        for _ in range(workers):
            await queue.put(stop)

    async def consume():
        while True:
            item = await queue.get()
            try:
                if item is stop:
                    return
                index, value = item
                results[index] = await operation(value)
            finally:
                queue.task_done()

    async with asyncio.TaskGroup() as group:
        group.create_task(produce())
        for _ in range(workers):
            group.create_task(consume())
    await queue.join()
    return [results[index] for index in sorted(results)], peak

async def main():
    async def square(value):
        await asyncio.sleep(0)
        return value * value
    result, peak = await supervised_map(range(20), square)
    assert result == [value * value for value in range(20)]
    assert peak <= 4

    async def fail_on_three(value):
        await asyncio.sleep(0)
        if value == 3:
            raise ValueError("planned")
        return value
    failed = False
    try:
        await supervised_map(range(100), fail_on_three, workers=2, capacity=1)
    except* ValueError:
        failed = True
    assert failed

asyncio.run(main())
```

生产者逐项 put，队列满就等待；worker get 后执行 operation，finally 确认取出的项目已经结束处理。成功路径中生产者为每个 worker 放一个哨兵，所有 worker 正常退出，任务组退出后再 join 可以确认未完成计数为零。失败路径中任务组传播异常，不会继续执行成功路径后的 join，因此不会因为尚未消费的排队项永久等待。

失败并不意味着之前的 operation 都没执行。若 operation 写过外部系统，已经发生的结果不会被 TaskGroup 回滚。本例用纯平方与故意异常，便于验证生命周期；真正 Agent 工具需要记录每项开始、完成与提交状态。整体返回失败只代表未获得完整成功结果，不代表零副作用。

这个实现也假定 values 的同步迭代不会长时间阻塞事件循环。若输入来自慢速同步 IO，应使用合适的读取边界或异步源；若 operation 不响应取消，任务组退出仍可能等待。接口合同应说明这些前提，而不是因为使用了高层工具就把全部责任交给运行时。

项目按五步验收：空输入返回空结果；单元素正确；多元素保持顺序；队列峰值不超过容量；某一项失败时整个调用及时结束且其他任务完成清理。额外记录未验证项，包括真实网络断连、进程强杀和外部写入恢复。把这些边界写清，下一版才知道该扩展什么，而不是重复已有纯计算验证。

## 18. 速率、并发和容量的手算

设平均每个请求占用连接一秒，并发上限四。在稳定且没有其他瓶颈的理想模型中，最多大约每秒完成四个；若平均占用变成十分之一秒，同样四个并发可能每秒发出几十次。因此并发控制不能替代每分钟请求限额。限额通常按提供商定义的窗口或令牌机制计算，真实接口必须阅读对应合同。

再设每个请求最多输入两千 token、输出一千 token，四个在途请求可能预留一万二千 token 的内容预算。实际输出可能更少，但如果只在结束后扣费，就可能在同时启动时超出总预算。一个简单办法是启动前预留上限，完成后按实际结算并释放多余预留；取消与失败时如何结算还要按服务计费语义处理。

队列容量控制等待项数量，连接池控制同时可用连接，信号量控制某段业务活动，速率器控制启动节奏，总预算控制累计成本。它们各自有不同单位和生命周期。把这些资源画成表，列出取得、释放和失败处理位置，可以发现重复计数或遗忘释放。不要用一个 max_workers 参数声称同时解决所有限制。

排队时间也属于用户延迟。一个请求处理只需一秒，但前面排了很多任务，用户可能等一分钟。若设置超时只覆盖真正请求阶段，排队时间可能无限延长。总截止时间应从用户提交时或明确约定的起点计算，排队后先检查剩余预算，不够则拒绝启动，而不是继续浪费资源完成一个调用者已经放弃的工作。

背压还涉及公平性。某个巨大任务不断提交项目，可能挤占小任务的队列；按用户、任务或优先级分配容量需要额外调度策略。基础版先采用简单 FIFO 并明确限制每次提交规模，足以练习机制。不要提前堆复杂调度框架，但要知道简单队列没有自动实现公平或优先级保证。

## 19. 取消之后的资源与状态审计

每种资源都应回答三个问题：谁取得，谁释放，释放发生在什么失败路径。信号量许可由上下文管理，活动计数在 finally 减少，文件在 with 退出关闭，数据库事务按提交或回滚结束，后台任务由父作用域等待。把所有资源列出来，往往能发现某个看似无害的缓存或计数从未在取消时恢复。

清理本身也可能需要等待。例如异步连接关闭需要发送协议结束消息，清理期间再次取消又会带来复杂性。初学项目保持清理短小可控，必要时用明确上限等待，并记录清理失败。不要在 finally 中无限重试，因为调用者正在要求停止，清理却可能让任务永远无法结束。

预算释放与实际成本不同。释放本地“在途预留”不代表远端没有收费；请求发送后即使响应丢失，也可能产生真实消耗。账本应区分已知实际、预留与未知结果，不能把所有失败统一记成零成本。Agent 系统尤其需要保留这种不确定状态，以便查询或人工复核，而不是靠重试次数掩盖。

如果结果写入文件，先完成临时内容再原子发布，可以避免取消时留下半份最终文件；如果写数据库，将相关状态放进事务，可以避免部分更新。但多个外部系统之间仍没有自动共同回滚。这里把 F08 的持久化边界与本章取消边界连接起来，你会发现可靠性需要多个机制协作，而不是单个关键词。

## 20. 异步调试与测试的最后一轮练习

异步堆栈与任务关系有时比同步调用链更难读，因为创建位置与等待位置不完全相同。给任务起有意义的名字、保留请求标识和阶段事件，有助于定位哪个任务在等待什么。调试输出应受控，避免在高并发循环中打印大量文本本身改变调度和耗时。

使用事件控制顺序比依赖固定 sleep 更可靠。若测试要在任务开始后取消，等待 started 事件；若要验证两个任务都读到旧值，建立共同屏障；若要验证先后完成，使用一个任务释放另一个的门。sleep(0) 适合让出执行权，但不应该承担所有精确同步要求，因为任务内部可能有不同数量的挂起点。

测试超时主要用于防止挂死，不能把刚好没有超时当作性能证明。对一个应在本地事件驱动下很快结束的测试，可以设置宽松整体上限，失败时保存任务状态；对真实性能应另做基准。共享机器负载变化时，功能测试仍应尽量稳定，性能报告则应保留环境和分布。

最后对你的 supervised_map 增加一个带 finally 记录的 operation。让其中一个输入失败，检查每个已经开始的操作都执行了清理；尚未开始的排队项不必执行 operation 的 finally，因为它根本没有取得那些资源。这个区别能防止你错误要求“所有输入都清理”，从而在测试里把未启动和已启动状态混为一谈。

毕业门槛是能把一个并发程序还原成状态转换：每项何时排队，何时取得资源，在哪些 await 可能暂停，失败如何传播，取消何时完成，结果与输入如何对应。你不需要一开始写高性能调度器，但必须能解释这些边界。之后面对 Agent 并行工具调用、异步检索与模型服务批处理，才有能力判断框架默认行为是否符合自己的业务合同。

### 补充：Event、Lock、Semaphore 与 Queue 的选择题

Event 表达某个条件已经发生，设置后多个等待者可以继续；它不保存每一次事件的次数。如果生产者连续 set 三次，消费者不一定能区分发生了三次。若每条消息都必须处理，应使用队列而不是把 Event 当消息计数器。clear 又会改变后续等待者的观察，因此多方重复设置与清除需要清楚协议，否则容易错过状态变化。

Lock 表达同一时刻只有一个任务进入临界区。它没有携带业务数据，也不会帮你决定锁内操作是否正确。Semaphore 表达有限个许可，适合控制同时在途数量；它允许多个任务进入，所以不能直接替代保护单份余额更新的互斥锁。Queue 同时保存数据和协调等待，适合生产消费关系。工具选择从你需要表达的关系开始，而不是见到并发就全部加锁。

Condition 将等待条件与锁协调起来，适合“共享状态满足某条件才继续”的模式。等待者通常要在循环中重新检查条件，而不是醒来一次就假定条件永久成立，因为其他任务可能先改变状态。当前初学项目优先用 Event 与 Queue 表达简单关系；遇到复杂条件时，先写清状态与唤醒协议，再引入更精细原语。

一个重要区别是通知与事实。通知告诉你“可能有新情况”，事实保存在共享状态或队列里。若把通知当作唯一事实来源，某次通知在等待者注册之前发生，程序就可能永久等待。队列保留消息，Event 保留设置状态，而瞬间回调或某些条件通知可能不保留全部历史。理解这一区别能帮助你设计不会丢信号的任务协调。

### 补充：用四格状态表审查重试

对一次外部写入，可以区分尚未发送、已发送但结果未知、已确认成功、已确认失败。输入验证错误通常停在尚未发送，明确业务拒绝属于已确认失败，连接在发送后断开可能属于结果未知。不同状态需要不同动作：尚未发送可安全重新准备，明确失败按错误类别处理，未知结果应查询或使用同一幂等键恢复。

如果你只用一个 success 布尔值，未知结果会被挤进 False，随后重试逻辑可能错误地当作“肯定没发生”。因此任务状态机需要表达不确定性。日志也应记录请求标识、尝试编号与逻辑操作标识的区别：多次尝试属于同一个逻辑操作，不能每次都生成新业务身份后再声称幂等。

取消可能发生在四个状态中的任何位置。尚未发送时停止通常没有外部副作用；已发送未知时本地停止等待不能确定远端结果；成功后取消返回处理也不能抹掉成功事实。恢复程序应依据持久化证据和远端查询，而不是只看最后异常类型。这个原则在 Agent 调用支付、写文件、提交任务或修改数据库工具时尤其重要。

重试等待也占用总体预算。指数退避可以降低瞬间反复冲击，但若没有最大次数与截止时间，仍可能持续很久；加入随机抖动可以减少多个客户端同时重试，但它又引入可复现性与测试需求。基础测试可以注入等待函数和随机源，检查计算出的延迟序列，而不真的等待几十秒。

### 补充：把并发实验的观察量分开

至少区分到达数量、排队数量、活动数量、完成数量、失败数量和取消数量。某一时刻活动数量下降可能意味着成功，也可能意味着失败或取消；仅看 active 归零不能证明所有任务成功。测试必须同时检查结果数量、结果关联与失败记录，监测指标也应避免把各种结束都算作成功吞吐。

平均延迟不能描述全部等待体验。几个很慢请求可能被大量快请求掩盖，队列增长也可能让尾部延迟恶化。F15 将讲分位数统计；这里先记录每个阶段耗时，避免把排队时间和执行时间混成一个无法解释的数字。资源瓶颈判断需要阶段数据，而不是只看整体平均值升高就增加 worker。

增加并发可能提高吞吐，也可能降低效率。更多任务争用同一数据库写锁、网络连接或内存带宽时，排队与切换成本增加；远端限流还可能产生更多重试。应从小并发逐步观察，在明确输入与预算下测量，而不是默认把并发调到机器核心数或 GPU 数就最优。不同资源没有简单的一一对应关系。

有界系统在过载时必须选择行为：等待、拒绝、丢弃低优先级工作或返回可重试错误。无限等待会让调用者失去响应，无界排队会耗尽内存。初学版本可以采用有界等待后明确失败，这样状态容易解释。以后再增加优先级、公平调度和持久队列时，每项都应有新的合同与验收，而不是隐式改变已有行为。

### 补充：手算一次失败监督

设容量为一，两个 worker，输入零到九，处理三时失败。生产者可能已把若干项放进队列，也可能正等待下一个 put。失败触发 TaskGroup 取消兄弟任务：正在 await operation 的 worker 收到取消，生产者在 put 处退出，已取得项目的 worker 运行 finally 确认完成。尚在队列里的项目没有被执行，不能声称每项都处理了一次。

任务组退出时调用者收到异常组，成功返回值没有形成。已经放进 results 的部分平方仍存在于函数局部，但函数不会把它作为完整结果返回。若业务需要部分结果，应显式定义返回结构和失败策略，而不是在 except 中随手返回当前字典，让调用者误以为处理完毕。完整成功、部分成功与整体失败是不同接口。

把这个时间线写完后，再对照源码每一处 await，判断取消可能在哪里注入。你会发现可靠性来自清楚生命周期与状态语义，而不是某个库名字本身。能做这种手算后，你已经能够审查不少异步框架封装：它是否保存输入身份、是否限制等待队列、是否传播失败、是否真正等待清理，都有了具体检查方法。

完成练习后保留一份任务清单：哪些任务由哪个父作用域创建，正常结束和异常结束分别由谁等待。若某个任务找不到负责人，就先修复生命周期，再讨论更高吞吐。并发程序最难排查的缺陷往往来自无人观察的后台工作；把所有权写清楚，可以在规模尚小时消除这种隐患，也使后续 Agent 工具执行器更容易测试与恢复。
