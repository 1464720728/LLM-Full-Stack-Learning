# A09 并发与调度：把独立工作同时推进，把共享事实守住

本章从一个普通函数和一个协程函数的区别开始。你会运行多个离线任务，观察它们怎样在 await 处交错；再加入并发上限、调用身份、共享预算、资源键锁、取消和部分成功。所有例子使用事件或显式让出控制权安排顺序，不靠真实网络速度，也不用 GPU。目标是你能证明调度器没有超额、串线或留下后台任务，而不是只会在函数前写 async。

主 Harness 仍按顺序执行 pending 工具。并发组件放在 `workflows.bounded_fanout` 与 `advanced_solutions.resource_fanout`，这是有意保持教学边界：先理解单运行状态和持久化，再将并发结果接回主循环。不能因为某个独立函数支持并发，就声称主 Harness 已经具有完整并行工具恢复能力。

## 第一课：异步、并发与任务对象

### 并发、并行与异步不是同一个词

并发表示多个任务的生命周期交错进行，并行表示多个工作在同一时刻实际执行。一个线程的事件循环可以在任务甲等待网络时运行任务乙，因此有并发；它不会因为使用 async 就自动把 CPU 密集计算分配给多个核心。异步描述调用与等待结果的方式，是否并行取决于底层执行器、线程、进程、设备和库。

设三个互不依赖查询各需等待一段外部响应。顺序执行时总等待近似相加，并发时可能接近最慢的一项加协调开销。但若第二个查询参数依赖第一个结果，它们就不能同时开始。若三个查询共享一个只能串行访问的资源，并发也未必更快。决定并发的首先是依赖和资源合同，不是函数名字。

Python `async def` 定义协程函数，调用它得到协程对象；对象尚未完整执行，需要 await 或交给事件循环调度。普通函数调用通常立即运行到 return，协程则可以在 await 处暂停，等待某个可等待对象完成。暂停不是丢失局部变量，它们保存在协程状态中，恢复后从原位置继续。

```python
import asyncio

async def twice(number):
    await asyncio.sleep(0)
    return number * 2

async def first_demo():
    coroutine = twice(3)
    assert asyncio.iscoroutine(coroutine)
    value = await coroutine
    return value

assert asyncio.run(first_demo()) == 6
```

这里 sleep(0) 只让出一次调度机会，不表示真实业务需要等待零秒。asyncio.run 建立事件循环并运行顶层协程，完成后关闭。不要在已经运行的事件循环内部再次调用 asyncio.run；在那种环境中应 await 顶层协程。课程代码块作为独立脚本顺序执行，因此可以在每个实验入口使用 asyncio.run。

如果协程内部直接运行长时间同步循环而不 await，事件循环无法切换到其他任务。把阻塞 HTTP 库调用包进 async def 也不会自动变成非阻塞。需要使用真正异步接口、受控线程执行器或进程执行器，并理解它们的取消与资源语义。尤其线程等待取消后，底层函数可能仍继续运行，A06 的不确定性仍然适用。


### 任务对象让多个协程获得调度机会

逐个 await 两个协程通常仍然顺序执行，因为第二个要等第一个返回才开始。create_task 把协程注册为可调度任务，使多个任务都能在事件循环中推进。任务对象保存完成状态、结果或异常，也提供取消请求。创建后不能随便丢弃引用，否则上层难以管理失败和清理。

```python
async def interleaving_demo():
    trace = []
    async def worker(name):
        trace.append((name, "start"))
        await asyncio.sleep(0)
        trace.append((name, "finish"))
        return name
    tasks = [asyncio.create_task(worker(name)) for name in ("A", "B")]
    values = await asyncio.gather(*tasks)
    return trace, values

trace, values = asyncio.run(interleaving_demo())
assert values == ["A", "B"]
assert trace.index(("B", "start")) < trace.index(("A", "finish"))
```

这个实验通过显式 await 让任务甲在结束前让任务乙启动，证明生命周期交错。它没有测量加速倍数，也没有证明 CPU 并行。日志里的开始与结束顺序是逻辑证据，比在繁忙机器上断言几毫秒更稳定。性能测试可以另外做，但正确性测试应先控制顺序与状态。

gather 等待多个任务并按输入顺序返回结果，即使完成顺序不同。这个性质方便调用方使用，但业务仍应携带 task_id 或 call_id。若以后改为按完成顺序流式交付，数组位置就不再稳定；若某项失败或被取消，纯位置关联也容易混乱。身份与顺序是两个字段，不应让一个代替另一个。

任务生命周期可以简化为已创建、等待资源、运行或等待外部结果、完成、失败或取消。这里“运行”并不表示一直占 CPU，协程可能大部分时间在等待。调度指标应区分排队时间和执行时间，否则你可能误以为工具很快，却不知道用户在队列里等了很久。

## 第二课：模型同时提出的工具，不一定可以同时执行

模型一轮输出两个工具调用，只说明它在同一响应中提出两项请求。它没有证明它们互相独立。一个写配置、一个读配置可能有顺序依赖；两个修改同一文件可能覆盖；两个扣款共享余额约束；两个检索虽然只读，也可能受同一服务速率限制。调度器必须依赖工具作者提供的资源语义和宿主政策，而不能只相信模型说“可以并行”。

可以给任务描述增加依赖集合、资源键、效果类型、可重复性和最大并发。依赖集合说明必须等哪些任务完成，资源键说明哪些任务会冲突，效果类型说明失败与取消后的处理，最大并发说明工具自身或服务限制。数据结构不必一开始很复杂，但每项语义都应明确，避免靠函数名猜测。

| 任务甲 | 任务乙 | 是否可直接并发 | 原因 |
|---|---|---|---|
| 读取公开文档甲 | 读取公开文档乙 | 通常可以 | 无业务写入且资源独立 |
| 修改 config | 读取修改后的 config | 不可以 | 乙依赖甲的新状态 |
| 修改文件甲 | 修改文件甲 | 需要串行或事务 | 同一资源写冲突 |
| 查询账户 | 查询另一个独立账户 | 视服务限制 | 可能共享速率配额 |
| 创建工单 | 创建另一工单 | 视业务合同 | 幂等、配额和顺序可能相关 |

如果依赖是有向无环图，可以在所有前置任务完成后调度节点；若有循环，需要明确循环终止条件和迭代状态，A11 会讨论。不要用“同时创建所有任务，让它们自己等”替代依赖设计，因为大量等待任务仍消耗内存，错误依赖还可能死锁。先画最小依赖图，再决定调度结构。

工具调用结果进入模型历史时，也要保持协议完整。即使 c2 先完成、c1 后完成，最终仍应按调用编号正确关联，并在所选协议要求的完整交换边界进入下一轮。并发完成不等于可以随时让模型对半组结果继续推理；若要支持增量结果驱动规划，需要另定义协议和状态机。

## 第三课：信号量限制在途工作，不限制任务总数

Semaphore 可以想成有限通行证。limit 为二时，最多两个任务进入受保护的 worker 主体，其余等待。async with 在进入时获取，离开时释放，即使 worker 抛异常也会释放。这个结构比手动 acquire 后忘记 release 更可靠，但仍需要外层管理任务取消。

```python
from labs.agents.workflows import bounded_fanout

async def measure_peak(limit):
    active = 0
    peak = 0
    async def worker(number):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return number * number
        finally:
            active -= 1
    values = await bounded_fanout(range(6), worker, limit=limit)
    return values, peak, active

values, peak, active = asyncio.run(measure_peak(2))
assert values == [0,1,4,9,16,25] and peak <= 2 and active == 0
```

计数器增加到 await 之前，在单事件循环中这段同步代码不会被其他协程插入；await 后其他任务可能运行，峰值因此反映受保护区域的重叠。finally 确保正常、异常或取消退出都减少 active。测试检查最终 active 为零，证明没有留下一项被计为运行的任务。

源码先验证 limit 为精确正整数，再创建信号量与任务。布尔值、分数、NaN、无穷大和字符串都不是合法槽位数。底层同步原语并不替你理解业务配置；错误数值可能产生意外计数行为。因此入口验证必须发生在任何 worker 开始之前，错误配置不能顺便发出几个请求。

bounded_fanout 会为所有 items 创建任务，再用信号量限制进入 worker。这限制了在途主体，却没有限制任务对象总数。若输入有一百万项，仍会创建大量等待任务，内存可能成为问题。大规模任务应使用有界队列和固定数量消费者，或分批生产。A05 的背压与本章调度在这里相接：并发上限和排队容量是两项不同限制。

## 第四课：失败后必须取消并等待其他任务

默认 gather 遇到异常时，上层可能很快收到异常，但其他已创建任务不一定已经停止。课程 bounded_fanout 显式捕获 BaseException，向所有任务请求取消，再用 gather(return_exceptions=True) 等待回收，最后重新抛出原异常。取消不是只调用 cancel 就结束，任务还需要在 await 点响应并执行 finally。

捕获 BaseException 的目的包括传播上层取消时也清理子任务，不是吞掉所有异常。清理后必须 raise，把原控制流交还调用方。若改成只打印错误并返回空列表，调用方会把失败误当成没有结果；若只取消不等待，函数返回后后台任务仍可能占资源或产生效果。生命周期管理应该随任务创建一起设计。

```python
async def cancellation_demo():
    entered = asyncio.Event()
    cleaned = []
    async def worker(number):
        if number == 0:
            await entered.wait()
            raise RuntimeError("planned failure")
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleaned.append(number)
    try:
        await bounded_fanout([0,1], worker, limit=2)
    except RuntimeError as exc:
        assert str(exc) == "planned failure"
    else:
        raise AssertionError("expected failure")
    assert cleaned == [1]

asyncio.run(cancellation_demo())
```

事件 entered 保证任务一已经进入等待区，任务零才失败，因此清理断言确实覆盖正在运行的兄弟任务。若只用 sleep 很短时间，慢机器上任务一可能尚未开始，测试虽然通过却没覆盖目标窗口。可控事件是并发测试的重要工具，它让你安排因果顺序，而不是猜调度时机。

取消无法撤销已提交副作用。一个任务正在等待远端创建工单，取消本地等待后，远端可能仍完成。调度器应保存操作身份和已知阶段，把这类结果交给 A06 的对账逻辑。不能因为所有 asyncio.Task 都显示 cancelled，就宣称所有业务动作都没有发生。程序任务状态与业务效果状态是两套事实。

## 第五课：单线程异步也会发生业务竞争

假设预算只剩一次，两个 worker 都先读取余额，然后 await，恢复后各自扣减并发请求。它们读到的都是一，因此都会认为获准。最后共享变量可能变成零甚至负数，实际请求却已发两次。单线程不代表跨 await 的逻辑原子性；业务检查与更新之间只要让出控制权，就可能被另一个协程插入。

```python
async def reproduce_budget_race():
    remaining = 1
    observed = 0
    both_read = asyncio.Event()
    sent = []
    async def worker(name):
        nonlocal remaining, observed
        local = remaining
        observed += 1
        if observed == 2:
            both_read.set()
        await both_read.wait()
        if local > 0:
            remaining = local - 1
            sent.append(name)
    await asyncio.gather(worker("A"), worker("B"))
    return remaining, sent

remaining, sent = asyncio.run(reproduce_budget_race())
assert remaining == 0 and len(sent) == 2
```

这个反例尤其危险，因为最终余额看起来没有负数。两个协程都把本地旧值一减一写回零，账本表面正常，真实请求已经超发。只断言 remaining 非负不能发现问题，必须检查真实进入外部调用的数量。A06 已强调预算账本与真实尝试应关联，这里给出稳定复现。

修复是把检查和预留放在同一个锁区域，锁里不做网络等待。成功预留后释放锁，再执行慢操作。这样既保证额度不被同时穿透，又不把所有网络调用串行化。课程 AsyncBudget 使用 asyncio.Lock，reserve 在锁内检查 used+amount 是否超 total，未超则增加 used 并返回 True。

```python
from labs.agents.advanced_solutions import AsyncBudget

async def fixed_budget():
    budget = AsyncBudget(1)
    sent = []
    async def worker(name):
        if await budget.reserve():
            await asyncio.sleep(0)
            sent.append(name)
    await asyncio.gather(worker("A"), worker("B"))
    return budget.used, sent

used, sent = asyncio.run(fixed_budget())
assert used == 1 and len(sent) == 1
```

这个账本只做预留，不提供退款、真实费用结算或跨进程持久化。若操作失败但可能已经消耗资源，不能随意 used 减一；是否释放取决于 A06 的已知事实。若要跨进程共享，需要数据库事务或原子服务，asyncio.Lock 只协调同一事件循环中使用这个对象的协程。把对象放进两个进程不会自动共享锁状态。

## 第六课：相同资源串行，不同资源仍可并发

全局 limit 为三只能限制总量，不能防止两个任务同时修改同一文件。资源键锁为每个冲突资源建立一把锁：同 key 的任务串行，不同 key 的任务可以并发。资源键应来自宿主规范化后的业务身份，例如租户加文件真实标识，而不是模型随意提供的标签。若同一个文件用两种路径表达成两个 key，锁就失效。

课程 resource_fanout 先获取资源锁，再获取全局信号量。这个顺序很重要：若先占全局槽位再等待同一资源，多个冲突任务可能把所有槽位占满，其他独立资源任务无法推进。先等资源再占全局槽位，使等待冲突的任务不浪费全局并发额度。锁顺序本身也应保持一致，避免多资源任务形成环形等待。

```python
from labs.agents.advanced_solutions import resource_fanout

async def resource_demo():
    active = set()
    peak = 0
    async def worker(job):
        nonlocal peak
        key = job["resource_key"]
        assert key not in active
        active.add(key)
        peak = max(peak, len(active))
        try:
            await asyncio.sleep(0)
            return job["id"]
        finally:
            active.remove(key)
    jobs = [{"id":"a1","resource_key":"A"}, {"id":"a2","resource_key":"A"},
            {"id":"b1","resource_key":"B"}]
    result = await resource_fanout(jobs, worker, limit=2)
    return result, peak, active

result, peak, active = asyncio.run(resource_demo())
assert result == ["a1","a2","b1"] and peak <= 2 and active == set()
```

当前函数为本批 jobs 的 key 建立锁字典，函数结束后释放引用，因此不同批次之间不共享资源锁。若两个调用者同时启动两批任务并修改同一资源，这个函数本身不能协调它们。生产调度器需要更长生命周期的锁管理器，或用数据库、文件锁与业务服务实现跨批次约束。文档必须把锁的范围写清楚，不能只说“按资源串行”而隐去仅限单批。

多个资源的任务更复杂。例如转账同时涉及账户 A 和 B，两项任务若分别先锁 A 再锁 B、先锁 B 再锁 A，可能互相等待。可以按规范化资源键的全局顺序获取全部锁，或交给数据库事务处理。锁顺序解决死锁风险，不自动解决业务一致性；余额检查和写入仍要处于合适事务内。

## 第七课：全失败即停与收集部分结果是不同产品语义

有的任务必须全部成功，例如生成一个需要三项完整检查的发布结论；任何一项失败都应停止并回收其余工作。有的任务部分结果仍有用，例如搜索三个资料库，其中一个不可用仍可以展示另外两个。调度器应明确选择策略，而不是让异常处理偶然决定。返回空列表、吞掉异常或把部分成功标成全部成功都不可取。

完整的 collect-all 可以为每个任务返回带 id 的结构：ok、value 或 error_type。它捕获普通业务 Exception，但让取消继续传播，这样用户取消整个批次时仍能回收任务。不要把 CancelledError 包装成普通失败后继续工作，否则上层取消失去效果。若需要特定任务独立取消，也应单独表达状态。

```python
async def collect_all(jobs, worker, limit=3):
    async def wrapped(job):
        try:
            value = await worker(job)
        except Exception as exc:
            return {"id":job["id"], "ok":False, "error_type":type(exc).__name__}
        return {"id":job["id"], "ok":True, "value":value}
    return await bounded_fanout(jobs, wrapped, limit=limit)

async def partial_demo():
    async def worker(job):
        await asyncio.sleep(0)
        if job["id"] == "bad":
            raise ValueError("planned")
        return job["id"].upper()
    return await collect_all([{"id":"good"},{"id":"bad"}], worker, limit=2)

partial = asyncio.run(partial_demo())
assert partial[0] == {"id":"good","ok":True,"value":"GOOD"}
assert partial[1] == {"id":"bad","ok":False,"error_type":"ValueError"}
```

这个包装器不把完整异常字符串传给模型，避免泄漏敏感上下文；真实服务可以增加经过批准的错误码和可操作信息。最终批次状态应为 partial，而不是看到至少一个 ok 就标记 success。验收条件由任务定义：可能要求全部必要来源成功，也可能允许某些可选来源失败。把 required 标记放进任务描述，可以让聚合器确定性判断。

部分结果与副作用仍需谨慎。一个任务失败可能已经产生效果，ok=False 不代表没有写入。若 worker 是有副作用的工具，应返回或抛出明确不确定类别，并让聚合状态保留 uncertain，而不是笼统 partial。并发会让多个未知状态同时存在，宿主需要逐项操作身份和对账，而不是一个总布尔值。

## 第八课：四个调度故障与可复现修复

故障一，配置 limit 为小数后并发失控。根因是把业务配置直接交给信号量，未要求精确正整数。修复在创建任务前验证，测试所有非法值下 worker 调用次数为零。这个测试不仅检查抛异常，也检查没有提前效果，防止代码先创建任务再验证。

故障二，一个 worker 失败后其他 worker 继续在后台运行。根因是只等待 gather 抛异常，未取消并回收兄弟任务。修复在异常路径统一 cancel，再 await 全部任务完成清理。测试用 Event 确保兄弟任务已经进入受控等待，再触发失败，最后检查 finally 记录。只检查函数抛异常不足以证明没有后台工作。

故障三，预算剩一却发出两次请求。根因是检查与扣减之间存在 await。修复锁内预留，锁外执行，测试直接统计真实请求入口。最终 used 没超限并不自动证明真实调用没超限，必须把账本与行为关联。若跨进程调度，则需要共享事务，单个 asyncio.Lock 无效。

故障四，大量同资源任务占满全局槽位，独立任务饥饿。根因是先拿全局信号量再等待资源锁。修复调整获取顺序，等待冲突资源时不占全局槽位；若任务需要多个资源，统一锁顺序。验收用事件控制一个资源长期占用，同时观察另一个资源任务能否进入，而不是仅看总耗时。

这些故障说明并发正确性不仅是数据最终值正确，还包括资源峰值、取消完成、等待公平和效果数量。测试应围绕不变量和可控调度点设计。随机压力测试可以补充发现问题，但不能代替一个能稳定复现根因的最小测试。

## 第九课：十二道完整练习与答案

练习一，比较顺序 await 与同时创建任务。答案记录 starts，在第一个任务等待时检查第二个是否已启动。并发版本两个 start 都先于任意 finish；顺序版本则第一个 finish 早于第二个 start。这里用 sleep(0) 形成明确调度点，不比较机器耗时。

```python
async def order_demo(concurrent):
    events = []
    async def work(i):
        events.append((i,"start"))
        await asyncio.sleep(0)
        events.append((i,"finish"))
    if concurrent:
        await asyncio.gather(work(0),work(1))
    else:
        await work(0)
        await work(1)
    return events

sequential = asyncio.run(order_demo(False))
concurrent = asyncio.run(order_demo(True))
assert sequential.index((0,"finish")) < sequential.index((1,"start"))
assert concurrent.index((1,"start")) < concurrent.index((0,"finish"))
```

练习二，分别设置一、二、四个槽位，结果一致且峰值不超限。答案复用 measure_peak，并要求最终 active 为零。不要断言并发四一定比并发二快，因为真实依赖可能有速率限制，离线测试也没有模拟真实延迟。

```python
for slots in (1,2,4):
    values, peak, active = asyncio.run(measure_peak(slots))
    assert values == [0,1,4,9,16,25]
    assert 1 <= peak <= slots and active == 0
```

练习三，非法并发配置不能启动 worker。答案逐个测试零、负数、布尔、小数、NaN、无穷和字符串，计数列表始终为空。输入校验在资源创建与效果之前，是 A03 合同思想在调度配置上的应用。

```python
async def reject_bad_limit(value):
    called = []
    async def worker(item):
        called.append(item)
    try:
        await bounded_fanout([1], worker, limit=value)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid limit accepted")
    assert called == []

for value in (0,-1,True,1.5,float("nan"),float("inf"),"2"):
    asyncio.run(reject_bad_limit(value))
```

练习四，完成顺序与 gather 返回顺序不同。答案让任务零等待任务一发信号，因此完成顺序必定一在前；返回列表仍按输入零、一。真实工具结果应携带编号，不能只依赖列表位置。

```python
async def reverse_completion():
    gate = asyncio.Event()
    completed = []
    async def worker(i):
        if i == 0:
            await gate.wait()
        else:
            completed.append(i)
            gate.set()
            return i
        completed.append(i)
        return i
    values = await bounded_fanout([0,1], worker, limit=2)
    return values, completed

values, completed = asyncio.run(reverse_completion())
assert values == [0,1] and completed == [1,0]
```

练习五，空批次返回空结果，不启动任务。答案也验证 limit 仍必须合法，不能因为没有 items 就接受错误配置；配置合同与数据是否为空是不同问题。

```python
async def empty_batch():
    called = []
    async def worker(item):
        called.append(item)
    result = await bounded_fanout([], worker, limit=1)
    assert result == [] and called == []
asyncio.run(empty_batch())
```

练习六，父任务取消应回收全部子任务。答案等待两个 worker 都启动，再取消父级 fanout，捕获 CancelledError 后检查两个 finally 均已执行。没有真实睡眠，事件保证取消发生在目标窗口。

```python
async def parent_cancel_demo():
    started = 0
    all_started = asyncio.Event()
    cleaned = []
    async def worker(i):
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(i)
    parent = asyncio.create_task(bounded_fanout([0,1], worker, limit=2))
    await all_started.wait()
    parent.cancel()
    try:
        await parent
    except asyncio.CancelledError:
        pass
    assert sorted(cleaned) == [0,1]
asyncio.run(parent_cancel_demo())
```

练习七，十个 worker 争三个预算，恰好三个预留成功。答案使用 AsyncBudget，检查成功次数和 used 一致。失败预留不增加 used，额度为零也不允许一次“试试看”的请求。

```python
async def reserve_many():
    budget = AsyncBudget(3)
    accepted = await asyncio.gather(*(budget.reserve() for _ in range(10)))
    assert sum(accepted) == 3 and budget.used == 3
    assert await budget.reserve() is False
asyncio.run(reserve_many())
```

练习八，预留金额类型必须正确。答案拒绝零、负数、布尔和小数；total 则可以为零，因为“没有预算”具有明确语义。不要把槽位正整数与总预算非负整数混用。

```python
async def invalid_reservation():
    budget = AsyncBudget(0)
    assert await budget.reserve() is False
    for amount in (0,-1,True,1.5):
        try:
            await budget.reserve(amount)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reservation accepted")
    assert budget.used == 0
asyncio.run(invalid_reservation())
```

练习九，相同资源不得重叠。答案复用 resource_demo 的 active 集合断言；再增加一批全同 key 的任务，峰值应为一，即使全局 limit 为四。资源锁比全局槽位更严格时，正确调度应服从资源约束。

```python
async def same_resource_only():
    active = 0
    peak = 0
    async def worker(job):
        nonlocal active, peak
        active += 1
        peak = max(peak,active)
        try:
            await asyncio.sleep(0)
            return job["id"]
        finally:
            active -= 1
    jobs = [{"id":i,"resource_key":"same"} for i in range(5)]
    values = await resource_fanout(jobs, worker, limit=4)
    assert values == list(range(5)) and peak == 1 and active == 0
asyncio.run(same_resource_only())
```

练习十，部分成功不能标为全部成功。答案根据 collect_all 的结构做确定性聚合：全成功为 finished，全失败为 failed，其余为 partial。空任务在本题定义为 finished，因为没有未满足工作；真实业务若要求至少一个来源，应另加合同。

```python
def batch_status(results):
    successes = sum(item["ok"] is True for item in results)
    if successes == len(results):
        return "finished"
    if successes == 0:
        return "failed"
    return "partial"
assert batch_status(partial) == "partial"
assert batch_status([]) == "finished"
assert batch_status([{"ok":False}]) == "failed"
```

练习十一，结果编号应与输入编号一一对应。答案在调度前拒绝重复 id，调度后比较集合与数量。单靠集合相等会漏掉重复结果，因此两个条件都检查。输入 id 应为宿主生成或验证的稳定标识，不是完成位置。

```python
def validate_job_ids(jobs):
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate job id")
    return ids
jobs = [{"id":"A"},{"id":"B"}]
ids = validate_job_ids(jobs)
results = [{"id":"B","ok":True},{"id":"A","ok":True}]
assert len(results) == len(ids) and {x["id"] for x in results} == set(ids)
try:
    validate_job_ids([{"id":"A"},{"id":"A"}])
except ValueError:
    pass
else:
    raise AssertionError("duplicate id accepted")
```

练习十二，锁内不要等待外部操作。答案用事件证明两个已获预算的任务都能进入网络等待，虽然预算预留本身串行。若把网络等待放在预算锁内，第二个任务无法进入，all_started 永远不会触发。此例采用正确实现，所以不会挂住；错误版本应在受控超时测试中运行。

```python
async def reservation_not_execution_lock():
    budget = AsyncBudget(2)
    started = 0
    all_started = asyncio.Event()
    release = asyncio.Event()
    async def worker():
        nonlocal started
        assert await budget.reserve()
        started += 1
        if started == 2:
            all_started.set()
        await release.wait()
    tasks = [asyncio.create_task(worker()) for _ in range(2)]
    await all_started.wait()
    release.set()
    await asyncio.gather(*tasks)
    assert budget.used == 2
asyncio.run(reservation_not_execution_lock())
```

## 第十课：完整小项目，固定消费者数量的有界队列

前面的 fanout 为每个输入创建一个任务，适合小批量。现在实现一个固定消费者池：一个 producer 按顺序把工作放进容量有限的队列，若队列满就等待；固定数量 consumer 从队列取任务执行；结果按输入位置保存，同时携带业务 id。这样等待任务对象数量不随输入量线性增长，但本项目仍把 jobs 和 results 列表放在内存，所以不是无限数据流系统。

```python
async def queued_collect(jobs, worker, *, workers=3, capacity=6):
    for value in (workers, capacity):
        if type(value) is not int or value < 1:
            raise ValueError("workers and capacity must be positive integers")
    validate_job_ids(jobs)
    queue = asyncio.Queue(maxsize=capacity)
    sentinel = object()
    results = [None] * len(jobs)
    peak_queue = 0

    async def producer():
        nonlocal peak_queue
        for index, job in enumerate(jobs):
            await queue.put((index, job))
            peak_queue = max(peak_queue, queue.qsize())
        for _ in range(workers):
            await queue.put(sentinel)

    async def consumer():
        while True:
            item = await queue.get()
            try:
                if item is sentinel:
                    return
                index, job = item
                try:
                    value = await worker(job)
                except Exception as exc:
                    results[index] = {"id":job["id"],"ok":False,"error_type":type(exc).__name__}
                else:
                    results[index] = {"id":job["id"],"ok":True,"value":value}
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(producer())]
    tasks += [asyncio.create_task(consumer()) for _ in range(workers)]
    try:
        await asyncio.gather(*tasks)
        await queue.join()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if any(item is None for item in results):
        raise RuntimeError("missing result")
    return results, {"peak_queue":peak_queue,"workers":workers}

async def queue_project():
    active = 0
    peak_active = 0
    async def worker(job):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0)
            if job["id"] == 4:
                raise ValueError("planned")
            return job["id"] ** 2
        finally:
            active -= 1
    jobs = [{"id":i} for i in range(10)]
    results, audit = await queued_collect(jobs, worker, workers=2, capacity=3)
    assert [x["id"] for x in results] == list(range(10))
    assert results[4]["ok"] is False and results[9]["value"] == 81
    assert audit["peak_queue"] <= 3 and peak_active <= 2 and active == 0
    return results

queue_results = asyncio.run(queue_project())
assert batch_status(queue_results) == "partial"
```

先读入口合同：workers 与 capacity 都是精确正整数，重复业务 id 在创建任务前拒绝。队列元素是输入索引加 job，索引用于把结果写回稳定位置，id 用于业务关联。两者同时保留，避免把排序位置当成永久身份。sentinel 是本次运行独有对象，使用 is 判断，不会与普通 job 混淆。

producer 每放入一项都可能等待，因此队列容量会把慢消费者的压力传回生产端。末尾放入与消费者数量相同的哨兵，保证每个消费者都能退出；只放一个哨兵会让其他消费者永远等下一项。哨兵也计入 queue 的未完成数量，所以 consumer 的 finally 必须调用 task_done，即使它准备 return。

consumer 对普通业务异常产生失败结果并继续下一项，体现 collect-all 政策；CancelledError 不被普通 Exception 捕获，父任务取消能传播。最外层 finally 取消并等待所有未完成任务，使 producer 若正堵在 put、consumer 若正堵在 get 或 worker 中，都有机会清理。结果完整性检查发生在正常收尾后，防止某个分支忘记写 results 却静默返回。

这个项目没有持久队列，进程崩溃后内存 jobs、队列和结果都会丢失。它也没有跨进程资源锁或业务幂等。把它用于只读离线查询很合适；若要执行副作用，必须在 worker 中接入 A10 的持久操作记录，并在恢复时区分尚未领取、已开始、已完成和结果未交付。固定消费者池解决容量，不会自动解决恢复。

## 第十一课：排队、公平与多层限制

全局并发上限保护整个进程，但不能保证不同用户公平。一个任务提交十万项工作，可能让后来的短任务一直排队。可以增加每任务、每租户、每依赖服务的配额，或用轮询队列让不同任务轮流取工作。最简单策略未必完美，但应该明确：先来先服务、按优先级、按租户公平或按 deadline 调度，各有不同取舍。

优先级也可能造成饥饿。若高优先级任务不断到来，低优先级永远不执行，可以加入等待时间提升或预留份额。但不能让模型自己给所有任务标记最高优先级；优先级来源应由用户授权、服务政策和任务类型决定。调度元数据与工具权限一样，需要宿主控制。

多层信号量要注意获取顺序。假设先拿全局槽位再等某个提供方槽位，可能让一个拥堵提供方占满全局容量，其他提供方受影响。可以按资源类别设计独立队列，或先获得更具体的资源许可再占全局执行槽位。没有一个对所有系统都最优的顺序，关键是画出等待关系，检查是否会占着稀缺资源等待另一资源。

排队时间也消耗任务 deadline。任务在队列里等了二十九秒，用户总期限三十秒，真正开始时只剩一秒，不能重新获得完整三十秒。消费者取任务后应检查取消和剩余时间，已过期任务可以直接标记未执行。若业务要求一定执行而不是用户交互式 deadline，则应使用不同任务类型与持久调度政策，避免混用语义。

过载时需要明确反馈。无限接受任务会把拒绝隐藏成无限等待，用户不知道是否还在进行。可以限制队列长度并返回忙碌、允许用户稍后重试，或建立可查询的持久排队状态。对学习项目，一个有界队列加明确错误已经能建立正确思想；生产系统再根据服务目标选择优先级和公平策略。

## 第十二课：进程崩溃后，谁拥有这项工作

内存任务对象只在当前进程有效。要让多个 worker 从持久队列领取任务，需要领取记录或租约。租约通常包含 owner_id、到期时间和版本，worker 定期续约，其他 worker 只有在合同允许时才能接管。接管并不证明旧 worker 已经停止；它可能只是网络隔离或暂停，随后恢复继续执行，因此业务幂等仍然不可缺少。

这就是租约和执行恰好一次之间的差别。租约减少同时拥有者的可能性，却无法在任意故障下凭空知道旧进程是否仍在远端提交动作。可以使用 fencing token，让下游资源拒绝旧拥有者的过期版本；前提是下游支持并验证这个令牌。若下游只是任意外部 API，不能靠本地数据库租约宣称完全避免重复效果。

课程 Store 明确假设单个 run 只有一个执行拥有者。快照版本 CAS 防止盲目覆盖，ledger 记录调用状态，但它没有完整多 worker 领取、续租和 fencing。把 SQLite 换成另一数据库，也不会自动补齐这些协议。数据库能力是实现材料，业务状态机和故障合同仍要设计。

跨进程调度还涉及时间来源。租约到期通常需要共享可信时间或数据库时间，不能直接比较不同机器的本地单调时钟值。时钟偏差会影响接管判断，网络延迟会影响续约。你可以在课程中先用一个数据库和假时钟模拟，再分析边界；不要一开始就上多机而失去可复现性。

恢复测试应包含旧拥有者迟到返回。新 worker 已接管并完成后，旧 worker 的结果如果无版本检查直接写入，可能覆盖更新状态。结果提交需要带任务版本或拥有者令牌，宿主拒绝过期提交，同时保留诊断信息。拒绝旧结果不等于撤销它可能产生的外部效果，因此操作身份和对账仍需保留。

## 第十三课：把并发结果提交回 Agent 状态

假设模型发出 c1、c2 两项只读调用，调度器并发执行。每项完成后先以 call_id 记录结果，再由单个状态拥有者把完整结果集合加入对话。这样 worker 不直接竞争修改 messages、pending 和 version，减少共享状态复杂度。worker 负责产出带身份的结果，协调者负责验证和提交，这是一种清晰的职责分离。

若 c2 先完成，可以把它的结果持久化但暂不启动下一轮模型。等 c1 完成后，协调者检查请求集合与结果集合一致、没有重复编号、结果 schema 合法，再按协议要求构造工具消息。最终展示可以实时告诉用户哪项已完成，但模型对话提交保持完整交换边界。A05 的展示与提交分离在并发工具中再次出现。

若 c1 失败，政策决定是把结构化失败结果与 c2 成功结果一并交给模型，还是停止整个任务。对纯读取，部分结果通常容易处理；对副作用未知，协调者应进入 uncertain，阻止模型根据不完整事实继续安排动作。不能让一个通用 gather 包装器替业务做这个决定。

预算也应在调度前原子预留。模型提出十项调用但只剩三次工具额度，系统应明确选择拒绝整批、执行允许的前三项并报告剩余、或重新规划。课程顺序 Harness 会逐项检查，达到上限停止；并发扩展不能先启动十项再在结果返回时发现超额。预留策略需要与任务原子性一致，不能随意截取会破坏依赖的子集。

结果顺序不应决定业务含义。可以按原始调用顺序输出消息以便阅读，但必须用 call_id 关联。若某个结果重复到达，先比较同编号的内容摘要与调用签名；相同结果可幂等忽略，不同结果应冲突。重放一个完成事件不能再次增加工具计数或重复追加消息，这与 A10 的 inbox 去重有相同结构。

## 第十四课：从资源图识别死锁与饥饿

死锁是任务互相等待，形成无法自行解除的循环；饥饿是某任务虽然理论上可执行，却长期得不到资源。两个任务各持一把锁再等对方，是死锁；大量短任务不断插队使长任务一直排不上，是饥饿。两者症状都可能是“任务一直不完成”，但修复不同。

分析死锁可以画等待图：节点是任务或资源，边表示持有或等待。若获取多把锁，统一顺序可以消除一类环；若等待外部事件时仍持锁，检查该事件的产生者是否也需要这把锁。常见异步错误是在锁内 await 一个回调，而回调内部再次请求同一锁，造成自我等待。不要因为代码都在一个线程就忽视锁图。

超时可以让等待最终失败，但不是死锁设计的根本修复。若每次都等到超时再释放资源，系统吞吐和用户体验仍然很差，且副作用可能已经部分发生。应先消除不必要的嵌套锁和锁内外部调用，再用超时作为故障边界。清理路径也需要检查，异常后未释放锁会使后续任务全部堵住。

公平性测试不宜只跑一批相同长度任务。可以构造一个长期占用资源 A 的任务、一组等待 A 的任务和一个资源 B 的短任务，观察 B 是否被无关等待阻塞。再构造持续到达的新任务，检查早期任务是否仍有推进机会。课程不要求实现完整公平调度器，但你应能描述当前策略可能让谁等待多久，以及是否有明确上限。

## 第十五课：本章综合验收与后续接入

综合验收准备六类 job：独立只读、同资源写入模拟、显式依赖、临时失败、永久失败和取消等待。每项带稳定 id、资源键与效果声明。先在纯内存中运行，记录开始、结束、取消、预算预留和结果提交事件；验证同资源不重叠、总峰值不超限、真实进入次数不超预算、取消后没有未回收任务、结果编号不丢不重。

接着把工具副作用替换为受控临时数据库写入，再注入提交后响应丢失，确认调度器不会因为 Task cancelled 就自动重做。最后把结果提交协调者接到 Agent 消息历史，运行 turn_groups 检查完整配对。每一步都增加一个复杂维度，不要同时引入网络、多进程和数据库后再试图猜哪里出了问题。

通过本章的门槛是：你能先判断哪些动作独立，画出依赖与资源冲突，选择明确失败政策，并用事件控制写出稳定测试。你还应能解释当前实现的范围：信号量限制主体不限制所有任务对象，asyncio.Lock 不跨进程，resource_fanout 的锁只覆盖单批，取消不撤销副作用，单运行 CAS 不等于分布式任务领取。掌握这些以后，A10 的持久化与 A11 的工作流就有了可靠的调度基础。

再把资源获取顺序用一张手算表展开。全局有两个槽位，任务 A1 持有资源 A 并等待释放，A2 也需要 A，B1 只需要 B。若 A2 先拿全局槽位再等 A，两个槽位都被 A1 与 A2 占住，B1 无法开始；若 A2 先等资源锁，它不占全局槽位，B1 就能进入。这不是提高峰值上限，而是在同样上限下避免无效占用。

| 时点 | 先全局后资源 | 先资源后全局 |
|---|---|---|
| A1 开始 | 占一个槽位与资源 A | 占一个槽位与资源 A |
| A2 到达 | 占第二槽位，等待 A | 等待 A，不占槽位 |
| B1 到达 | 等待全局槽位 | 获得 B 与剩余槽位 |
| A1 释放 | A2 才推进，B1 继续竞争 | A2 推进，B1 可已完成 |

完整的确定性验收如下。A1 等待一个 release 事件，B1 启动时设置 b_started；测试必须在释放 A1 之前看到 B1 启动。正确实现可以完成，若把锁顺序改错，内部等待将无法推进，所以测试使用外层 wait_for 作为失败保护，而不把其耗时作为性能结论。

```python
async def resource_order_check():
    a_started = asyncio.Event()
    b_started = asyncio.Event()
    release = asyncio.Event()
    async def worker(job):
        if job["id"] == "A1":
            a_started.set()
            await release.wait()
        elif job["id"] == "B1":
            b_started.set()
        return job["id"]
    jobs = [{"id":"A1","resource_key":"A"}, {"id":"A2","resource_key":"A"},
            {"id":"B1","resource_key":"B"}]
    parent = asyncio.create_task(resource_fanout(jobs, worker, limit=2))
    try:
        await a_started.wait()
        await asyncio.wait_for(b_started.wait(), timeout=2)
        release.set()
        assert await parent == ["A1","A2","B1"]
    finally:
        if not parent.done():
            parent.cancel()
        await asyncio.gather(parent, return_exceptions=True)
asyncio.run(resource_order_check())
```

这里的两秒只是防止错误实现让测试永久挂住，正确性依据仍是事件先后。机器极端繁忙时任何现实超时都可能误报，因此持续集成可以适当放宽保护值；不要把它解释成 B1 必须在两秒以内完成的产品性能指标。逻辑测试与服务延迟目标应分别管理。

另一个需要检查的细节是输入可迭代对象。bounded_fanout 能遍历 items 创建任务，但 resource_fanout 先遍历 jobs 建锁字典，再遍历创建任务，因此调用方应提供可重复遍历的列表等集合。如果传入一次性生成器，第一遍可能已经消耗全部元素，第二遍没有任务。当前课程示例统一使用列表；生产接口应明确要求 Sequence，或在入口物化一次并设置总量上限。类型注解和文档应与实际迭代次数一致。

对于 queued_collect，jobs 同样需要长度和索引位置，因此是有限列表接口。若想真正处理无限或超大流，应把结果逐项写入持久存储或下游有界队列，不保留与输入同样长的 results。那时结果顺序可能需要单独排序缓冲，缓冲也要有上限；坚持严格输入顺序交付可能被一个慢任务阻塞后面所有已完成结果。这是顺序交付与低延迟交付的明确取舍，不存在一个无成本同时满足所有要求的默认选项。

你可以用任务编号保留关联，同时允许完成顺序交付，最终报告再按输入顺序整理。界面即时显示哪个子任务完成，持久账本按 id 存储，聚合器在必要任务齐备后提交整体状态。这三种视图分别服务响应速度、恢复和业务验收，不必强迫它们使用同一种列表顺序。只要身份稳定、终态明确，系统就能兼顾交互与正确性。

并发的延迟收益也应按任务完成条件计算。如果批次必须等全部子任务，总延迟由最慢必要项主导；增加并发可以减少顺序等待，却无法消除慢尾。若允许先返回部分结果，用户看到首个结果更快，但最终完成时间可能不变。指标应分别记录首结果时间、必要结果齐备时间和全部清理结束时间，不能只挑最漂亮的一项宣称系统加速。

超时策略可以按子任务与批次分别设置。一个可选资料源超时，可以记录局部失败并继续；一个必要审批服务超时，整个任务可能必须停止。把所有子任务统一设为很短 timeout，虽然降低平均等待，却可能系统性丢掉慢但重要的证据。调度参数应服务业务验收，而不是为了让计时数字好看。

实际限流还可能按 token 或资源权重计费。两个请求分别包含一百和十万 token，简单把它们都算一个槽位只是一种近似。可以为任务估计权重，使用加权预算或分队列处理大请求；但估计误差、未知输出和动态重试都要进入账本。初学阶段先把固定槽位和次数预算做正确，再逐步增加权重，避免一开始实现复杂调度却无法解释基本超发。

最后检查日志因果关系：每个任务应有一个开始事件和一个明确终结事件，终结可以是成功、失败、取消或不确定；不能既记录成功又被迟到取消覆盖。批次聚合应引用子任务终态，而不是根据某个共享计数猜测完成。A04 的终态保护和 A06 的迟到取消规则在并发系统中仍然成立，只是现在需要对每个子任务与父批次分别实施。

若父批次已取消而某个远端结果随后到达，可以把它作为迟到观察写入受控记录，用于确认实际效果；这不意味着重新启动父批次规划。是否向用户追加通知取决于产品政策，但事实不能因本地不再等待就被丢弃。把“接收事实”和“继续执行”分成两个入口，有助于在取消、恢复和人工对账之间保持清楚边界，也避免迟到回调直接修改已经终结的状态。

记录迟到结果时仍需校验调用身份与内容摘要，重复观察应保持幂等，冲突观察应进入人工核验。
