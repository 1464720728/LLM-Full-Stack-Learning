# S16 综合工程：把请求、线程、状态、数值和证据接成一条可验收的链

你已经分别学习调度、页池、前缀共享、抢占、模型执行和观测。综合工程的困难在于：这些部件各自正确，并不保证接起来也正确。接收请求的线程可能在后台线程计算时改动页表；报告可能只统计成功请求，把最慢的失败请求丢掉；数值模型可能确实输出了 token，却每一步都重新计算历史，与你声称的真实缓存不同。本章围绕这些接缝建立一套完整项目，不把最关键部分留成没有答案的待办事项。

先修是 S01—S05 的请求与缓存状态、S06 的观测、F07 的对象引用、F09 的异常与测试、F14 的数组，以及 M03、M06 的注意力与缓存。线程与 Future 在本章重新解释。所有 Python 代码块按顺序在同一个解释器执行；从仓库根目录运行，依赖标准库与 NumPy。正文包含完整回放报告、单工作线程实现、并发提交与取消验证、多层真实 K/V 对照、负向控制和十二道完整题解。没有自动下载模型、联网调用 API、启动公网服务或占用 GPU。

## 第一课：先把一个工程问题翻译成可检验的合同

假设你要给 Agent 提供模型服务。它可能在工具返回后提交一个更长提示，也可能因为用户取消而放弃请求。最简单的成功演示是返回四个 token，但这远远不够。你至少要回答：这些 token 属于哪个请求；请求何时进入调度；谁能够修改它的状态；结束后谁释放缓存；服务同时接收多个请求时输出是否串线；模型异常是否会让其他请求永远等下去。

“合同”在这里不是一种额外框架，而是明确写下允许的输入、状态变化、输出和失败行为。例如 submit 的输入是不可变提示副本和非负输出上限；它成功后产生唯一请求身份；失败不能留下活动页。cancel 的输入是请求身份；首次取消活跃任务返回真，重复取消返回假；取消结果到达后不能再发布新输出。snapshot 返回副本，调用者改变它不能修改引擎。

把这些要求放到一个表中，能防止最后只剩一张运行截图。表中的证据不是可互换的：输出一致不能证明无内存泄漏；引用计数正确不能证明矩阵乘法正确；线程没有崩溃不能证明只有一个线程修改 Engine。

| 合同 | 直接观察量 | 最小反例 |
|---|---|---|
| 请求隔离 | 每请求输出与独立运行相同 | 同时提交不同末 token 的提示 |
| 预算守恒 | 每批 count 之和不超过 budget | 长提示与短解码混合 |
| 资源归属 | 活动页表、refs、free、缓存引用 | 半页分叉后取消一支 |
| 取消终态 | 终态唯一，取消后无新输出 | 重复取消及结束后取消 |
| 数值缓存 | 每层 K/V 长度与完整 logits 对照 | 长短请求交错推进 |
| 报告诚实 | 成功、失败、取消与未定义指标分列 | 零输出请求、单输出请求 |
| 所有权 | 所有 mutation 的线程身份一致 | 多接收线程同时提交 |

第一版工程应该把问题规模控制得足够小，让每个合同都可解释。小规模不意味着只写空壳：本章会完整实现所有权、返回结果和取消。规模限制只是让你先不处理异步 GPU 内核、分布式进程和网络流控等另外几层复杂性。后续升级时，每增加一个状态持有者，就增加相应的合同与证据。

## 第二课：回放输入也是数据集，先保证它可以比较

回放不是“发一些请求看看快不快”。它固定到达时间、提示 token 数、输出上限、模型与随机种子，再对比配置。若配置甲用一组短请求，配置乙用另一组长请求，延迟差无法归因于调度。若在一次实验里同时改变 token 预算、页大小、模型精度和采样参数，也很难知道改善来自哪里。

仓库的 [replay.py](../../labs/serving/replay.py) 已有完整 compare。它使用显式 token ID，避免把 tokenizer 版本和聊天模板变化混进长度。每项含 id、prompt、output、arrival。arrival 的单位是模拟毫秒，不是电脑墙钟；output 是最多产生多少新 token，不是总上下文长度。每个请求的状态仍由 Request 和 Engine 校验，前端还可以在进入实验前检查整个 trace 的重复身份。

下面的 validate_trace 复制输入，不让调用者后来修改原列表改变已经计划的实验。它拒绝 bool 作为整数，因为 Python 中 bool 是 int 的子类，而这里 True 并不表示合法的输出预算一。所有条目先验证，再开始任何引擎执行；这让坏输入与运行中的资源失败分开。

```python
import copy
import hashlib
import json
import math
from collections import Counter
from labs.serving.engine import Engine, Request
from labs.serving.replay import compare, DEFAULT
from labs.serving.analysis import percentile

def validate_trace(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("trace requires 1..1000 rows")
    clean, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be an object")
        rid, prompt = row.get("id"), row.get("prompt")
        count, arrival = row.get("output"), row.get("arrival", 0)
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("request IDs must be unique nonempty strings")
        if not isinstance(prompt, list) or not 1 <= len(prompt) <= 256:
            raise ValueError("prompt requires 1..256 token IDs")
        if any(type(t) is not int or not 0 <= t < 16 for t in prompt):
            raise ValueError("this replay uses the 16-token Bigram backend")
        if type(count) is not int or not 0 <= count <= 64:
            raise ValueError("output must be an integer in 0..64")
        if type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival must be finite nonnegative milliseconds")
        seen.add(rid)
        clean.append({"id": rid, "prompt": list(prompt),
                      "output": count, "arrival": float(arrival)})
    return clean

trace = validate_trace(copy.deepcopy(DEFAULT))
fingerprint = hashlib.sha256(
    json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
comparison = compare(trace, budgets=(2, 8), policies=("fifo", "round_robin"))
assert comparison["output_equivalence"]
assert len(comparison["runs"]) == 4
assert len(fingerprint) == 64
```

指纹不是证明数据真实，也不是隐私清理。它只是帮助确认两份报告用的是完全相同的序列化输入。若将来输入来自公司流量，先生成不含敏感内容、经过允许的匿名长度或合成数据，再计算指纹。不能以“做了哈希”为理由把原始私有提示一起放进公开仓库。

上面四次运行固定同一 trace，比较两个预算与两种策略。chunk 被 compare 设为 min(budget,4)，因此预算二与预算八同时对应 chunk 二与四。报告必须承认这个联动；如果研究问题要求只改变预算，就自己固定 chunk。控制变量不是说绝不能改变多个参数，而是准确说明你比较的是单一因素还是一个组合配置。

## 第三课：把百分位、吞吐和 goodput 算到每个分母

TTFT 是首输出时间减到达时间；输出为空时没有首输出，所以 TTFT 未定义。ITL 是相邻输出时间差；只有一个输出时没有相邻对，所以平均 ITL 也未定义。把未定义填成零会奖励零输出或失败请求，使报告看起来更快。保留 None，并同时报告有效样本数量，才能让读者理解百分位到底覆盖了谁。

手算三个 TTFT 为二、四、十毫秒。课程 percentile 使用排序后线性插值：位置索引是样本数减一再乘分位数。中位数索引一，结果四；百分之九十五索引一点九，位于四与十之间，结果四加零点九乘六，等于九点四。这不是声称所有统计库都采用同一分位数约定，因此报告要固定实现。

吞吐也必须写清单位与窗口。如果二十四个输出 token 在三十毫秒模拟时钟内产生，则输出吞吐为二十四除以零点零三秒，等于每秒八百 token。它并不等于每秒八百请求，也不等于 H100 能做到的速度。若模拟时间包括空闲等待，吞吐分母也包括等待；换成只统计忙时会得到另一种指标，不能偷偷替换。

Goodput 表示满足指定服务目标的有效工作速率。本章按请求计：请求必须成功结束、至少输出一个 token，且 TTFT 和端到端时间都满足阈值，才计为一个 good 请求。其速率是 good 请求数除以整个运行秒数；达标比例另用所有提交请求作分母。取消请求是否算未达标属于实验定义，本章明确把它列入总提交但不计 good，避免把困难请求取消后报告自动变好。

```python
assert math.isclose(percentile([2.0, 4.0, 10.0], .95), 9.4)

def summarize(report, ttft_limit=6.0, e2e_limit=30.0):
    rows = report["requests"]
    states = Counter(row["state"] for row in rows)
    observed = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
    good = sum(
        row["state"] == "finished" and row["output_tokens"] > 0
        and row["ttft_ms"] <= ttft_limit and row["e2e_ms"] <= e2e_limit
        for row in rows
    )
    seconds = report["time_ms"] / 1000.0
    return {
        "offered_requests": len(rows), "states": dict(states),
        "ttft_sample_count": len(observed),
        "p95_ttft_ms": percentile(observed, .95) if observed else None,
        "delivered_tokens": sum(row["output_tokens"] for row in rows),
        "good_requests": good, "good_fraction": good / len(rows) if rows else None,
        "good_requests_per_second": good / seconds if seconds > 0 else None,
        "time_kind": report["time_kind"],
    }

for run in comparison["runs"]:
    summary = summarize(run["detail"])
    assert summary["offered_requests"] == 4
    assert summary["states"] == {"finished": 4}
    assert summary["ttft_sample_count"] == 4
    assert 0 <= summary["good_fraction"] <= 1

zero_engine = Engine()
zero_engine.submit(Request("zero", (1,), 0))
zero_engine.run()
zero_summary = summarize(zero_engine.report())
assert zero_summary["ttft_sample_count"] == 0
assert zero_summary["p95_ttft_ms"] is None
assert zero_summary["good_requests_per_second"] is None
```

零输出请求直接结束，但模拟时钟可能仍为零。此时 goodput 速率也未定义，而不是在数学上除以零。它的成功状态仍可统计。这样的边界例子比额外添加十个普通请求更能检查报告是否混淆“任务成功”和“生成了可测的首 token”。

不要在同一张表中只展示最好的一行。至少保留每种配置的成功率、样本数、首延迟、总时间、缓存命中和抢占次数。若一个配置首延迟较低但总工作量更高，你需要解释收益与代价，而不是把第二个数字删掉。百分位是压缩信息，逐请求细节仍应保留以便追查异常样本。

## 第四课：加入取消之后，回放必须说明事件边界

真实取消可能发生在任意墙钟时刻。本章引擎的 step 是同步的，无法在其中一行矩阵计算之间插入另一个线程事件。最清楚的教学协议是“在第几次外部 step 之前处理取消”。这个定义可重复、无需 sleep，也不会伪装成精确的网络到达时间。改变策略后，同一个 step 号对应的模拟毫秒可能不同，所以它适合状态回归，不适合直接比较取消延迟。

下面完整回放器接受取消计划，如第二次 step 前取消 long。它先取消，再推进，最后验证所有请求终态。取消计划中不存在的身份会被拒绝；取消同一个已终止请求返回假。循环有明确步数上限，若算法不再前进，测试会报错，不会一直等待。终态后检查页表空，再清缓存检查全部物理页归还。

```python
def replay_with_cancellation(rows, cancel_before_step=None, **configuration):
    rows = validate_trace(rows)
    plan = cancel_before_step or {}
    ids = {row["id"] for row in rows}
    if any(type(step) is not int or step < 1 for step in plan):
        raise ValueError("cancellation step numbers must be positive integers")
    if any(rid not in ids for group in plan.values() for rid in group):
        raise ValueError("cancellation references an unknown request")
    engine = Engine(prefix_entries=0, **configuration)
    for row in rows:
        engine.submit(Request(row["id"], tuple(row["prompt"]), row["output"], row["arrival"]))
    cancelled = []
    for step in range(1, 1001):
        for rid in plan.get(step, ()):
            cancelled.append((step, rid, engine.cancel(rid)))
        progressed = engine.step()
        if not progressed:
            break
    else:
        raise RuntimeError("replay exceeded its explicit iteration limit")
    report = engine.report()
    for event in engine.events:
        if event["kind"] == "batch":
            assert event["scheduled_tokens"] == sum(w["count"] for w in event["work"])
            assert event["scheduled_tokens"] <= engine.token_budget
    assert not engine.pool.tables
    engine.cache.clear()
    engine.check()
    assert len(engine.pool.free) == len(engine.pool.pages)
    return {"report": report, "cancelled": cancelled,
            "events": copy.deepcopy(engine.events)}

cancel_trace = [
    {"id": "short", "prompt": [1, 2], "output": 3, "arrival": 0},
    {"id": "long", "prompt": [3] * 12, "output": 5, "arrival": 0},
]
cancel_run = replay_with_cancellation(
    cancel_trace, {2: ["long"], 3: ["long"]},
    token_budget=2, chunk_size=2, max_sequences=2, pages=16,
)
assert cancel_run["cancelled"] == [(2, "long", True), (3, "long", False)]
assert cancel_run["report"]["outputs"]["short"] == [3, 4, 5]
assert summarize(cancel_run["report"])["states"] == {"finished": 1, "cancelled": 1}
```

这里短请求与长请求使用不同末 token，让串线更容易暴露。长请求在第二步前被取消时尚无完整提示，因此没有输出。短请求继续完成，说明取消只影响目标身份，不是调用清空整个引擎。所有页面最终归还说明资源链也闭合。我们同时保留事件、输出和页池证据，因为其中任何一种单独通过都不足以说明整条路径正确。

如果取消事件位于未来到达请求之前，当前 Engine 会将其到达记账调整到取消时刻，避免产生负结束时长。做真实负载报告时，你可能希望保留原计划到达与实际取消两个字段；那是数据模型扩展，不应该默默把模拟器的行为当成通用定义。最初的状态回归可以把所有请求到达设为零，先隔离取消与资源释放。

## 第五课：从一个锁到唯一拥有者，理解线程到底共享什么

[mock_server.py](../../labs/serving/mock_server.py) 的完整基础服务让多个 HTTP 接收线程共用一个 Engine，并用锁包住 submit 与 run。这个版本容易理解且避免同时改页表，但长请求执行期间锁一直被占用，其他请求只能等待获得锁。去掉锁不会自动得到连续批处理；它只会让两个线程可能同时看见相同 free 页并各自修改复合状态。

唯一拥有者方案改变的是职责。接收线程负责校验格式、复制输入并发送命令；工作线程独占 Engine、Request、页池和调度事件；结果通过 Future 返回。Future 是一个未来才会完成的容器：调用 result 可以等待值或异常。它不是模型缓存，也不是后台任务本身；改变 Future 的状态不会自动取消 Engine 请求。

线程安全 Queue 提供跨线程传递命令的边界。对队列的 put/get 是受协调的，但命令里如果装着一个仍会被外部修改的列表，线程安全并不能替你冻结那个列表。因此提交时把 prompt 转成 tuple，结果中把 output 转成 tuple。观察快照另外构造新字典，避免把原始 requests 或 pages 引用交给接收线程。

本章的 Ticket 包含唯一 ID 和结果 Future。ID 由服务生成，调用者不重复使用旧身份，减少迟到取消误伤新任务的风险。取消也进入相同命令队列，工作线程按顺序处理。一个请求如果刚好已经完成，取消返回假，这是可解释的竞争结果；不能规定任意晚到的取消都必须倒转已经完成的响应。

为了既能做真实并发测试又能确定性教学，OwnedWorker 提供 automatic 两种模式。自动模式在每批命令后推进一个引擎 step；手动模式只响应显式 advance 命令。手动模式不是“假的线程”，仍有真实独立工作线程和并发提交，只是把模拟时间推进交给测试，从而不靠电脑快慢猜测取消是否来得及。

## 第六课：完整实现 OwnedWorker 的提交、完成、取消与关闭

下面实现只用标准库。所有 Engine 方法都在 _loop 线程执行，包括构造、提交、取消、快照和释放。接收端只触碰命令队列、关闭标志与 Future。命令队列上限一百二十八，活跃任务上限六十四，提示上限二百五十六，输出上限六十四；这些是教学配置，用来保证每次演示有界，不是生产容量建议。

command_budget 限制每轮处理八条外部命令，然后让自动模式推进一步。若无限排空队列再计算，持续到来的命令可能让模型永远得不到执行；如果一次 run 到全部完成才看队列，取消又无法及时处理。交替推进提供一个清楚的公平边界。它不保证墙钟延迟，因为一次后端计算本身仍可能很慢，但不会在两个职责之间完全饿死其中一个。

Future 在返回之前标记为运行中，因此调用者直接 future.cancel 不会跳过服务协议；正确取消入口是 worker.cancel(ticket.rid)。这一选择避免未来完成时遇到 InvalidStateError，也迫使资源释放经过拥有者。结果回调若由使用者注册，会在设置结果的线程触发，因此本实现规定回调不得阻塞、不得在回调里等待同一个 worker 的另一个 Future；实际接口层应只做轻量通知。

```python
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Thread, Lock, Event, get_ident
import uuid

@dataclass(frozen=True)
class Ticket:
    rid: str
    result: Future

@dataclass(frozen=True)
class Completed:
    rid: str
    state: str
    token_ids: tuple
    ttft_ms: float | None
    error: str | None

def running_future():
    future = Future()
    assert future.set_running_or_notify_cancel()
    return future

class OwnedWorker:
    def __init__(self, automatic=True, active_limit=64, queue_limit=128):
        if type(active_limit) is not int or active_limit < 1:
            raise ValueError("positive active limit required")
        if type(queue_limit) is not int or queue_limit < 1:
            raise ValueError("positive queue limit required")
        self.automatic = automatic
        self.active_limit = active_limit
        self.commands = Queue(maxsize=queue_limit)
        self.lock = Lock()
        self.closed = False
        self.ready = Event()
        self.thread = Thread(target=self._loop, name="course-engine-owner")
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("worker startup did not complete")

    def _send(self, kind, payload):
        future = running_future()
        with self.lock:
            if self.closed:
                raise RuntimeError("worker is closed")
            try:
                self.commands.put_nowait((kind, payload, future))
            except Full:
                future.set_exception(RuntimeError("command queue is full"))
        return future

    def submit(self, prompt, max_tokens=4, seed=0):
        rid = uuid.uuid4().hex
        frozen_prompt = tuple(prompt)
        future = self._send("submit", (rid, frozen_prompt, max_tokens, seed))
        return Ticket(rid, future)

    def cancel(self, rid):
        return self._send("cancel", rid)

    def advance(self, count=1):
        return self._send("advance", count)

    def snapshot(self):
        return self._send("snapshot", None)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stopped = running_future()
        # Worker never acquires self.lock, so bounded queue backpressure is safe.
        self.commands.put(("stop", None, self.stopped))
        result = self.stopped.result(timeout=10)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("owner thread did not stop")
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _loop(self):
        engine = Engine(pages=128, block_size=4, token_budget=8,
                        chunk_size=4, max_sequences=4, prefix_entries=0)
        pending = {}
        owner = get_ident()
        mutations = set()
        self.ready.set()

        def observed():
            return {"owner_thread": owner, "mutation_threads": tuple(sorted(mutations)),
                    "pending": len(pending), "request_history": len(engine.requests),
                    "free_pages": len(engine.pool.free), "total_pages": len(engine.pool.pages),
                    "time_ms": engine.time_ms, "running": tuple(engine.running)}

        def publish_terminal():
            for rid, request in list(engine.requests.items()):
                if request.state in {"finished", "cancelled", "error"}:
                    future = pending.pop(rid)
                    result = Completed(rid, request.state, tuple(request.output),
                                       request.metrics()["ttft_ms"], request.error)
                    del engine.requests[rid]
                    future.set_result(result)
            # This service returns bounded final results, not an unbounded event archive.
            engine.events.clear()

        def one_step():
            mutations.add(get_ident())
            engine.step()
            publish_terminal()
            engine.check()

        def handle(command):
            kind, payload, future = command
            mutations.add(get_ident())
            try:
                if kind == "submit":
                    rid, prompt, count, seed = payload
                    if len(pending) >= self.active_limit:
                        raise ValueError("active request limit reached")
                    if not 1 <= len(prompt) <= 256:
                        raise ValueError("prompt length must be 1..256")
                    if type(count) is not int or not 0 <= count <= 64:
                        raise ValueError("max_tokens must be 0..64")
                    request = Request(rid, prompt, count, engine.time_ms, seed=seed)
                    engine.submit(request)
                    pending[rid] = future
                elif kind == "cancel":
                    changed = engine.cancel(payload) if payload in pending else False
                    publish_terminal()
                    future.set_result(changed)
                elif kind == "advance":
                    if type(payload) is not int or not 1 <= payload <= 128:
                        raise ValueError("advance count must be 1..128")
                    for _ in range(payload):
                        one_step()
                    future.set_result(observed())
                elif kind == "snapshot":
                    future.set_result(observed())
                elif kind == "stop":
                    for rid in list(pending):
                        engine.cancel(rid)
                    publish_terminal()
                    engine.cache.clear()
                    engine.check()
                    future.set_result(observed())
                    return False
                else:
                    raise ValueError("unknown command")
            except Exception as error:
                future.set_exception(error)
            return True

        while True:
            try:
                first = self.commands.get(timeout=.01 if self.automatic and pending else None)
            except Empty:
                first = None
            if first is not None:
                if not handle(first):
                    return
                for _ in range(7):
                    try:
                        command = self.commands.get_nowait()
                    except Empty:
                        break
                    if not handle(command):
                        return
            if self.automatic and pending:
                one_step()
```

逐段看其责任。初始化启动线程后等待 ready，避免调用者把“线程对象已创建”误认为 Engine 已准备好。_send 在锁下检查 closed 与入队，因此 close 开始后不会再接受新命令。队列满时返回带异常的 Future，调用者能够感知背压，不会得到一个永远不完成的请求。

submit 命令真正执行时才构造 Request，所以随机状态与引擎记录都属于拥有者。pending 把请求 ID 映射到需要兑现的 Future。终态发布先构造只读结果、删去内部历史，再设置 Future，外部不持有 Request 本身。prefix_entries 为零使这个服务练习聚焦活动请求资源；如果启用缓存，结束后 free 不一定等于总页数，关闭时仍需释放缓存拥有者。

close 采用中止未完成任务的明确策略，而不是默默等待全部长请求自然结束。它先禁止新命令，再将 stop 排到先前命令后面，已有提交均被处理；stop 取消全部 pending、发布结果、释放缓存并验证。这里没有使用守护线程掩盖未关闭资源。主程序必须 close 或使用 with，正常路径与异常离开路径都执行关闭。

这个实现的异常保证也有边界。输入验证与后端预测异常已经由课程接口处理；任意 Python 进程终止、内部代码缺陷、内存耗尽或使用者阻塞回调不属于“保证线程继续健康”的范围。不要把普通 try/except 当成完整进程监管。生产服务还需要进程健康、超时、队列水位和错误隔离，但本章已经完整实现所声明的同步 CPU 请求生命周期。

## 第七课：并发测试先控制时间，再放开自动推进

并发测试最容易出现一种假通过：开了线程，却每次只等待一个任务完成再提交下一个。另一种问题是用 sleep 猜测内部进度，在快电脑上请求先结束，在慢电脑上取消先到，断言偶尔失败。我们先使用手动推进，让二十四个接收线程任务真正并发入队，但在取消测试完成前不推进模拟模型。

ThreadPoolExecutor 的工作线程只是客户端，不持有 Engine。pool.map 返回 Ticket 的顺序与输入顺序一致，但实际命令入队顺序可能不同，因此测试不要求内部请求顺序固定。每个结果按自己的 Ticket 对应原 prompt，不能把“第一个完成的结果”当成“第一个提交的请求”。

先取消索引为三倍数的八项，第一次应成功，第二次应失败。然后有限次数推进其余十六项。每个结果都与原始末 token 逐步加一的 Bigram 规则比较。最后检查唯一修改线程、历史清理和全部页归还；这比只检查二十四个 Future 都结束强得多。

```python
with OwnedWorker(automatic=False) as worker:
    prompts = [[1, 2, 3 + i % 8] for i in range(24)]
    with ThreadPoolExecutor(max_workers=6) as clients:
        tickets = list(clients.map(lambda p: worker.submit(p, 4), prompts))
    barrier = worker.snapshot().result(timeout=5)
    assert barrier["pending"] == 24
    assert barrier["owner_thread"] != get_ident()
    assert len({ticket.rid for ticket in tickets}) == 24

    cancelled_indexes = set(range(0, 24, 3))
    with ThreadPoolExecutor(max_workers=4) as clients:
        changes = list(clients.map(
            lambda i: worker.cancel(tickets[i].rid).result(timeout=5),
            sorted(cancelled_indexes)))
    assert all(changes)
    assert worker.cancel(tickets[0].rid).result(timeout=5) is False

    for _ in range(100):
        snapshot = worker.advance().result(timeout=5)
        if snapshot["pending"] == 0:
            break
    else:
        raise AssertionError("manual worker did not finish")
    for i, ticket in enumerate(tickets):
        result = ticket.result.result(timeout=5)
        if i in cancelled_indexes:
            assert result.state == "cancelled" and result.token_ids == ()
        else:
            assert result.state == "finished"
            assert result.token_ids == tuple(range(prompts[i][-1] + 1, prompts[i][-1] + 5))
    assert snapshot["mutation_threads"] == (snapshot["owner_thread"],)
    assert snapshot["request_history"] == 0
    assert snapshot["free_pages"] == snapshot["total_pages"]
```

这里 snapshot 作为队列屏障：它前面的提交命令已经处理，才轮到快照。因此不需要等待任意毫秒。快照后的新命令当然不包含在这份观察里；它是某个确定队列边界的副本，不能被当成随时自动更新的实时对象。理解这个边界，有助于以后阅读异步引擎中的事件与 epoch。

再打开自动模式，让多个客户端各自等待自己的 Future。测试仍设置超时，防止代码缺陷把练习无限挂住，但超时只用于约束等待，不作为吞吐或延迟结论。验证输入复制时，先提交列表，再立刻改原列表；模型应使用提交时的 tuple。验证错误隔离时，坏 token 的 Future 应得到 ValueError，而同一 worker 随后的合法请求仍能正常完成。

```python
with OwnedWorker(automatic=True) as automatic_worker:
    def call_client(i):
        ticket = automatic_worker.submit((1, 2, i + 3), 3)
        return i, ticket.result.result(timeout=5)
    with ThreadPoolExecutor(max_workers=4) as clients:
        automatic_results = list(clients.map(call_client, range(12)))
    for i, result in automatic_results:
        assert result.token_ids == tuple((i + offset) % 16 for offset in (4, 5, 6))
    bad_ticket = automatic_worker.submit([True], 2)
    try:
        bad_ticket.result.result(timeout=5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid token accepted")
    assert automatic_worker.submit([4], 2).result.result(timeout=5).token_ids == (5, 6)

with OwnedWorker(automatic=False) as copied_worker:
    mutable_prompt = [1, 2, 3]
    copied_ticket = copied_worker.submit(mutable_prompt, 1)
    mutable_prompt[-1] = 20
    copied_worker.advance().result(timeout=5)
    assert copied_ticket.result.result(timeout=5).token_ids == (4,)
    snapshot = copied_worker.snapshot().result(timeout=5)
    snapshot["pending"] = 999
    assert copied_worker.snapshot().result(timeout=5)["pending"] == 0
```

这两段分别验证线程交错与对象别名，不应混为一个“并发没问题”的笼统结论。第一段在实际客户端线程之间竞争入队，第二段构造明确的外部修改。无论 GIL 如何安排线程，冻结输入都保持语义；不能因为一次运行没有撞上改列表的时刻，就认为共享可变对象安全。

## 第八课：关闭、背压与前端异常必须有完整返回路径

如果服务停止时只结束线程，不兑现还在等待的 Future，客户端就会永远等待一个不会再出现的结果。因此关闭协议必须决定：排空请求，还是中止请求。本实现选择中止。对未完成项返回 cancelled，释放页，再结束线程；已经完成的结果不撤回，之后再提交直接抛出 worker is closed。

“队列上限”和“活跃任务上限”处理不同资源。前者限制尚未处理命令占用的内存；后者限制已进入引擎的请求数量。队列满时提交尚未进入引擎，所以 Future 返回异常即可；活跃数量满时命令已被拥有者接收，但仍在 submit 前拒绝。无论哪一种拒绝，都不能生成一个孤立请求留在 Engine 里。

下面用活跃上限一制造确定性的背压，不靠线程速度把队列挤满。第一项合法等待，第二项得到拒绝；关闭兑现第一项取消并归还全部页。被拒绝的请求不计作已接受任务，报告应单独统计 admission_rejected，而不是伪装成模型计算失败。若做端到端服务可用率，它仍应进入外部提交总数。

```python
limited_worker = OwnedWorker(automatic=False, active_limit=1)
first_ticket = limited_worker.submit([1, 2, 3], 64)
second_ticket = limited_worker.submit([4], 1)
try:
    second_ticket.result.result(timeout=5)
except ValueError as error:
    assert "active request limit" in str(error)
else:
    raise AssertionError("admission limit ignored")
assert first_ticket.result.cancel() is False
closed_snapshot = limited_worker.close()
assert first_ticket.result.result(timeout=5).state == "cancelled"
assert closed_snapshot["pending"] == 0
assert closed_snapshot["request_history"] == 0
assert closed_snapshot["free_pages"] == closed_snapshot["total_pages"]
assert not limited_worker.thread.is_alive()
assert limited_worker.close() is None
try:
    limited_worker.submit([1], 1)
except RuntimeError:
    pass
else:
    raise AssertionError("closed service accepted new work")
```

把已有 mock_server 的输入合同接到新 worker，不必重新发明一个看似兼容的网络协议。下面的 call_frontend 是完整的接收端函数：先调用现有校验器，再转交不可变参数，等待结果；等待超时后明确发送取消，而不是误以为 Future.result 的超时会替你释放页。这里沿用同步返回方式，适合先验证职责分离。

validate_payload 返回一个尚未提交的临时 Request，前端只取它的已验证参数，真正引擎请求仍由拥有者构造；临时对象不会被两个引擎共用。服务结果标明 synthetic 时间。此函数可被多个接收线程并发调用，正文已经实际验证，不依赖读者以后自己补齐核心桥接。

```python
from concurrent.futures import TimeoutError as FutureTimeout
from labs.serving.mock_server import validate_payload
from dataclasses import asdict

def call_frontend(worker, body, timeout=5.0):
    parsed = validate_payload(body)
    ticket = worker.submit(parsed.prompt, parsed.max_new_tokens, parsed.seed)
    try:
        completed = ticket.result.result(timeout=timeout)
    except FutureTimeout:
        worker.cancel(ticket.rid).result(timeout=5)
        raise
    return {**asdict(completed), "time_kind": "synthetic_ms",
            "protocol": "course_token_id_completion"}

with OwnedWorker() as frontend_worker:
    payloads = [{"prompt_token_ids": [1, i + 2], "max_tokens": 2}
                for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as clients:
        replies = list(clients.map(lambda body: call_frontend(frontend_worker, body), payloads))
    for i, reply in enumerate(replies):
        assert reply["token_ids"] == (i + 3, i + 4)
        assert reply["state"] == "finished"
    try:
        call_frontend(frontend_worker, {"prompt_token_ids": [], "max_tokens": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("frontend validation missing")
```

这仍不是完整流式 HTTP 服务。Future 完成意味着整条任务终止，不能拿它冒充每个 token 的网络发送时刻。网络连接断开也不天然等于请求取消，宿主需要明确检测与发送取消事件。现有 mock_server 可用于本机 HTTP 协议练习，而本章给出的 worker 与前端函数完整解决线程所有权；两者的证据范围分别记录，不能因为接口路径叫 completions 就声称兼容任何厂商全部 API。

如果进一步测真实服务，S15 的 bounded probe 已提供有请求数、并发数和输出上限的客户端，以及部分失败记录。先检查 dry-run 报告，再在得到允许的设备窗口执行。探针报告中的墙钟首事件时间、引擎内部模拟 TTFT 和模型 prefill 内核耗时不是同一指标，本章不会把它们画在同一根没有单位的柱子上。

## 第九课：数值集成首先要问，缓存里究竟放了什么

Engine 的 PagePool 保存 token 元数据，Backend.logits 接收完整上下文。即使后端换成真正的 TinyDecoder，Engine 仍可能每一步重新做完整前向。因此你可以准确说“真实数值模型接进调度”，却不能据此说“每请求真实 K/V 已被复用”。这一区别决定你该去哪里寻找正确性与性能证据。

[projects.model_runtime](../../projects/model_runtime.py) 完成前一种集成：随机权重 TinyDecoder 产生真实 logits，调度策略改变工作组织，最终输出与独立完整生成相同。[projects.incremental_runtime](../../projects/incremental_runtime.py) 完成第二种集成：每个 RunningRequest 持有每层 K/V 和 next_logits，start 先完成 prefill 再发布；advance 先计算新缓存成功再更新输出；最后一个输出产生后清除缓存。

为什么最后一次不需要再把刚采样的 token 输入模型？请求的输出上限已经满足，不再需要预测它后面的 token。此时保留最后 token 的 K/V 是多余工作。若目标改成持续会话缓存，就要明确另一个协议：究竟存储到哪一个已知位置，下一轮从哪里继续。不能只看到 output 长度，就推断缓存应该同样长。

下面先比较完整重算调度与独立生成，再比较两种真实缓存调度顺序。输出等价是第一层；每个请求终态 cache 为零是第二层。完整模型仍是小型随机模型，能验证数组与控制流，不能验证自然语言质量或生产吞吐。

```python
import numpy as np
from projects.model_runtime import run as full_context_run, DecoderBackend, direct_generate
from projects.incremental_runtime import IncrementalRunner, OnlineDecoder
from labs.models.core import TinyDecoder

for policy in ("fifo", "round_robin"):
    recomputed = full_context_run(policy)
    assert recomputed["matches_direct"]
    assert recomputed["independent_attention_max_error"] < 1e-12

prompts_for_model = {"small": (1, 2), "large": (2, 3, 4, 5, 6)}
numeric_runs = {}
for model_class in (TinyDecoder, OnlineDecoder):
    for policy in ("fifo", "round_robin"):
        model = model_class()
        runner = IncrementalRunner(model)
        expected = {rid: direct_generate(DecoderBackend(model), prompt, 4)
                    for rid, prompt in prompts_for_model.items()}
        for rid, prompt in prompts_for_model.items():
            runner.start(rid, prompt, 4)
        initial = runner.snapshot()
        assert initial["small"]["cache_lengths"] == [2] * model.config.layers
        assert initial["large"]["cache_lengths"] == [5] * model.config.layers
        events = runner.run(policy)
        actual = {rid: request.output for rid, request in runner.requests.items()}
        assert actual == expected
        assert len(events) == 8
        assert all(state["cache_bytes"] == 0 for state in runner.snapshot().values())
        numeric_runs[(model_class.__name__, policy)] = actual
assert len({json.dumps(value, sort_keys=True) for value in numeric_runs.values()}) == 1
```

这里 OnlineDecoder 替换每层注意力计算，而不是只在模型旁边额外跑一次算子然后宣布“已集成”。它保留 Q/K/V 投影、RoPE、GQA 与输出投影的模型语义，把头内 attention 交给在线参考实现。支持范围是完整因果 prefill 或单个末位置 query，不是任意分块矩形 mask；这一边界在适配器中主动拒绝，避免悄悄漏掉未来遮罩。

比较所有最终 token 仍不够，因为错误 logits 可能恰好选中同一 argmax。下一课直接观察 next_logits 与每层状态。把集成拆成“请求映射正确”“数值输出正确”“资源生命周期正确”三个层次，才能在失败时知道该去查服务层、模型层还是缓存层。

## 第十课：逐步审计真实多层缓存，并验证失败前不提交

对每个 running 请求，next_logits 表示当前已消费上下文的最后位置预测。初始已消费整个 prompt；第一次 advance 选择一个 token，若请求尚未完成，就把它送入 step，形成下一次预测。于是第 j 次非最终推进后，各层缓存长度应该是 prompt 长度加已经输出数量。终态清空缓存是另一条规则，不能继续套 running 的长度等式。

下面每轮只推进一个请求，并在推进前用完整前向重算 next_logits 对照。短请求与长请求交错，让“使用某个全局共享缓存”这种错误更容易暴露。每层 K/V 都检查头数、时间轴和字节数；结构正确、数值正确和最终输出正确分别有断言。

```python
audited_model = OnlineDecoder()
audited_runner = IncrementalRunner(audited_model)
for rid, prompt in prompts_for_model.items():
    audited_runner.start(rid, prompt, 3)
numeric_checks = 0
for _ in range(3):
    for rid, prompt in prompts_for_model.items():
        state = audited_runner.requests[rid]
        context = np.asarray(tuple(prompt) + tuple(state.output), dtype=np.int64)
        full_logits = audited_model.forward(context)[-1]
        np.testing.assert_allclose(state.next_logits, full_logits, rtol=1e-10, atol=1e-10)
        for keys, values in state.cache:
            assert keys.shape == values.shape
            assert keys.shape[0] == audited_model.config.kv_heads
            assert keys.shape[1] == len(context)
        numeric_checks += 1
        audited_runner.advance(rid)
assert numeric_checks == 6
assert all(row["status"] == "finished" and row["cache_bytes"] == 0
           for row in audited_runner.snapshot().values())
```

再检查失败原子性。训练之外的推理服务同样可能发生模型错误；若先把 token 加到 output，再尝试计算下一缓存，计算失败后就会出现输出已经提交而状态仍旧的半完成请求。当前 IncrementalRunner 先得到新的 logits/cache，再追加输出，因此可以用一个明确抛错的模型验证旧状态保持。

下面的 FailingDecoder 只在 step 被调用时抛错，不改变底层数组。我们先正常完成 prefill，保存输出、next_logits 和每层 K/V 副本，再开启错误并推进。异常后必须全部不变；取消仍能释放这些旧状态。这验证的是同步 step 在发布前失败，不代表任何会原地写 GPU 缓存的模型都自动具备同一性质。

```python
class FailingDecoder(TinyDecoder):
    fail_step = False
    def step(self, token, cache=None):
        if self.fail_step:
            raise RuntimeError("injected model failure")
        return super().step(token, cache)

failing_model = FailingDecoder()
failing_runner = IncrementalRunner(failing_model)
failing_runner.start("fault", (1, 2, 3), 4)
old_state = failing_runner.requests["fault"]
old_output = list(old_state.output)
old_logits = old_state.next_logits.copy()
old_cache = [(k.copy(), v.copy()) for k, v in old_state.cache]
failing_model.fail_step = True
try:
    failing_runner.advance("fault")
except RuntimeError:
    pass
else:
    raise AssertionError("injected failure disappeared")
assert old_state.output == old_output
np.testing.assert_array_equal(old_state.next_logits, old_logits)
for (actual_k, actual_v), (expected_k, expected_v) in zip(old_state.cache, old_cache):
    np.testing.assert_array_equal(actual_k, expected_k)
    np.testing.assert_array_equal(actual_v, expected_v)
assert failing_runner.cancel("fault")
assert not failing_runner.cancel("fault")
assert failing_runner.snapshot()["fault"]["cache_bytes"] == 0
```

这个实验不要只写“抛错成功”。抛错之后仍能取消、状态仍一致，才说明故障边界得到检验。若接口改成 in-place 缓存写入，你需要预留新页、日志回滚或失败后终止请求等更强协议；不能沿用上面的证明而不检查它的非原地前提。S18、S19 的资源预检与 COW 正好提供另一个边界的具体实现。


## 第十一课：真实源码回归要保留三个结果，不能只有一张绿图

综合工程的源码部分不是“打开某函数看一遍”。我们选一个可判断的行为：长提示分块阈值与总 token 预算同时约束第一轮安排。S17 已提供固定版本的真实上游测试文件与安装辅助脚本，本章负责把它接进工程交付。课程自身离线测试、上游导入成功、真实 Scheduler 测试执行、GPU 模型前向是不同证据，不允许互相代填。

先手算：三个请求各需要八百位置，每个长提示本轮最多四百，总预算一千零二十四。第一个安排 min(800,400,1024)=400，剩余六百二十四；第二个安排 min(800,400,624)=400，剩余二百二十四；第三个安排 min(800,400,224)=224，剩余零。输出 count 列表为四百、四百、二百二十四，总和一千零二十四。总预算五百一十二时，结果变成四百、一百一十二，第三个不进入本轮。

下面先用独立十几行实现表达合同，再做变异：只把 total 字段加一，要求验证器拒绝。这不是替代真实上游测试，而是让你先知道测试究竟在检查什么。边界断言必须同时验证 total 与各项之和、各项上限、总预算；只检查 count 个数会漏掉预算超发。

```python
def reference_plan(lengths, budget, chunk):
    if any(type(x) is not int or x < 0 for x in lengths):
        raise ValueError("lengths must be nonnegative integers")
    if type(budget) is not int or budget < 0 or type(chunk) is not int or chunk < 1:
        raise ValueError("invalid budget or chunk")
    remaining, counts = budget, []
    for length in lengths:
        amount = min(length, chunk, remaining)
        if amount:
            counts.append(amount)
            remaining -= amount
    return {"counts": counts, "total": sum(counts)}

def verify_plan(plan, budget, chunk):
    assert plan["total"] == sum(plan["counts"])
    assert 0 <= plan["total"] <= budget
    assert all(type(n) is int and 0 < n <= chunk for n in plan["counts"])
    return True

correct_plan = reference_plan([800, 800, 800], 1024, 400)
assert correct_plan == {"counts": [400, 400, 224], "total": 1024}
assert verify_plan(correct_plan, 1024, 400)
mutated_plan = copy.deepcopy(correct_plan)
mutated_plan["total"] += 1
try:
    verify_plan(mutated_plan, 1024, 400)
except AssertionError:
    mutation_detected = True
else:
    raise AssertionError("negative control did not fail")
for budget in range(1025):
    assert verify_plan(reference_plan([800] * 3, budget, 400), budget, 400)
```

真实测试文件是 [test_course_scheduler_contract.py](../../labs/serving/upstream_tests/test_course_scheduler_contract.py)，它导入上游 Scheduler 与构造助手，构造 ModelRunnerOutput 并回写，另外验证运行请求取消幂等和块归还。不是把课程 Engine 换个类名当成 vLLM。它的可选变异通过 pytest monkeypatch 只改变测试进程中的 schedule 输出，不修改正在服务的安装文件。

安装辅助默认 dry-run，核对独立源码目录的固定提交与目标文件不存在，再在显式 apply 时复制一份新测试。它不会替你安装依赖或准备模型配置。按 S17 准备好独立上游环境后，完整执行顺序如下；这些命令本次没有在未配置环境中冒充通过。

```bash
python -m labs.serving.install_upstream_test --root upstream/vllm
python -m labs.serving.install_upstream_test --root upstream/vllm --apply
# 以下三条在独立 upstream/vllm 根目录执行：
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -q
COURSE_MUTATE_SCHEDULER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -k partial_prefill -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -q
```

中间一次预期失败，前后两次预期通过。若中间一次在 import torch 时失败，它没有到达关注的断言，不能算变异被检测。若第一次就失败，要先定位版本、依赖、配置与接口，不应通过调整 expected 数字让测试变绿。修复后再回到同一工作负载与同一断言，才能说明真正解决了问题。

没有上游环境时仍能执行一个诚实的静态检查：读取测试源文件、解析 AST、确认目标测试函数存在，并计算内容指纹。它证明交付文件存在且语法可解析，不证明导入、执行、取消和资源行为。下面把这种状态明确写进 evidence，而不是留一个容易被误解的空白“通过”。

```python
import ast
from pathlib import Path
upstream_test_path = Path("labs/serving/upstream_tests/test_course_scheduler_contract.py")
upstream_source = upstream_test_path.read_text()
upstream_tree = ast.parse(upstream_source)
test_names = {node.name for node in ast.walk(upstream_tree)
              if isinstance(node, ast.FunctionDef) and node.name.startswith("test_course_")}
assert test_names == {
    "test_course_partial_prefill_obeys_both_budgets",
    "test_course_abort_running_request_is_idempotent",
}
source_evidence = {
    "path": str(upstream_test_path),
    "sha256": hashlib.sha256(upstream_source.encode()).hexdigest(),
    "syntax_parsed": True,
    "upstream_pytest_executed": False,
    "gpu_forward_executed": False,
    "local_plan_negative_control_detected": mutation_detected,
}
```

未来自己提交补丁时，描述应围绕具体触发条件与改变后的行为，例如“有三个长提示且总预算不足以给第三个完整 chunk 时，回归要求安排数量之和严格等于公开 total”。不要写“优化了调度器、提高了鲁棒性”而没有输入输出。一个可评审的补丁通常只修改必要范围，附最小反例、正常边界与资源收尾，不需要先重写整个框架。

## 第十二课：三个故障工作坊，分别隔离报告、身份与状态提交

### 故障一：所有请求都成功，百分之九十五延迟却不可信

症状是报告突然显著变快，但输出数量下降。检查发现程序把没有首 token 的 TTFT 写成零，再与真正完成的请求一起排序。零不是“缺少测量”的合理默认值，因为它在排序中表示最快。修复应保留 None，报告样本数、终态数量和生成数量；不能只换一个更好看的分位数。

下面构造两个成功样本和八个没有首输出的取消样本。错误平均只有正确条件均值的五分之一。完整反例让问题直接可见，不依赖复杂调度。它也提醒你条件均值本身没有代表所有提交，必须与两成的观测覆盖率一起报告。

```python
observations = [8.0, 12.0] + [None] * 8
wrong_mean = sum(0.0 if x is None else x for x in observations) / len(observations)
valid_observations = [x for x in observations if x is not None]
conditional_mean = sum(valid_observations) / len(valid_observations)
coverage = len(valid_observations) / len(observations)
assert wrong_mean == 2.0
assert conditional_mean == 10.0
assert coverage == .2
assert wrong_mean != conditional_mean
```

### 故障二：每个 Future 都结束，但不同用户拿到了别人的结果

症状是单请求正确，并发以后偶发串线。错误实现按 Future 完成顺序拼接结果，再 zip 原始请求顺序。完成顺序通常由长度与调度决定，并不等于提交顺序。修复是保存唯一请求 ID 与结果容器的对应，或者让结果自身携带 ID，再按 ID 连接。

这个故障不需要真的制造线程随机性才能验证。把结果顺序显式反转，正确路由仍应恢复相同映射；错误 zip 必须失败。测试不把合法重复 token 当重复请求，也不使用输出文本猜身份，因为两个用户完全可能得到相同文本。

```python
submitted_ids = ["short", "long"]
completed_rows = [
    {"rid": "long", "tokens": (8, 9)},
    {"rid": "short", "tokens": (3, 4)},
]
correct_routing = {row["rid"]: row["tokens"] for row in completed_rows}
wrong_routing = dict(zip(submitted_ids, [row["tokens"] for row in completed_rows]))
assert correct_routing == {"short": (3, 4), "long": (8, 9)}
assert wrong_routing != correct_routing
```

### 故障三：模型抛错以后，重试多出一个 token

症状是失败前客户端没有收到完整响应，重试却从更长的 output 继续，或者 cache 长度比已输出序列落后。原因常是先修改 output，再计算下一缓存；异常打断了这次提交。第十课已用真实 TinyDecoder 的注入异常验证正确实现，本节用最小字典把错误顺序单独展示。

参考修复不是在 except 里盲目清空所有请求，而是在新的状态完整可用之后才发布。若底层函数原地写入，则需要另一种事务设计；这里的演示只使用返回新值的计算。代码先保留错误例子的失败状态，再比较正确实现保持原样，证明断言确实能区分两种顺序。

```python
def failing_compute():
    raise RuntimeError("computation failed before commit")

wrong_state = {"output": [], "cache_version": 0}
try:
    wrong_state["output"].append(7)
    wrong_state["cache_version"] = failing_compute()
except RuntimeError:
    pass
assert wrong_state == {"output": [7], "cache_version": 0}

right_state = {"output": [], "cache_version": 0}
before_state = copy.deepcopy(right_state)
try:
    new_version = failing_compute()
    right_state["output"].append(7)
    right_state["cache_version"] = new_version
except RuntimeError:
    pass
assert right_state == before_state
```

三类故障分别属于统计定义、请求身份和提交边界。它们最终都可能表现为“服务结果怪怪的”，但调容差、加线程、加页数不能替代定位。每份故障报告都写出第一个违约位置、最小输入、错误实现为什么会产生该状态、修复后什么断言会防止回归。只有“重启以后好了”没有解释力，也无法成为后续工程知识。

## 第十三课：十二道递进练习与完整参考答案

### 题一：预算八百，三项需求八百，chunk 四百，第一轮是什么

答案是四百、四百，第三项不入本轮，总和八百。第一项之后剩四百，第二项之后剩零，因此第三项取 min 时为零。reference_plan([800]*3,800,400) 已给完整算法；不要在 counts 中添加一个零工作项，否则后续执行器可能误以为它对应真实计算。验收还应测试预算零，返回空 counts 与 total 零。

```python
assert reference_plan([800] * 3, 800, 400) == {"counts": [400, 400], "total": 800}
assert reference_plan([800] * 3, 0, 400) == {"counts": [], "total": 0}
```

### 题二：十个提交里只有两个生成首 token，能否只写平均 TTFT 十毫秒

答案是不够。十毫秒是两个有效样本的条件均值，必须同时写观测数二、总提交十，以及其余八项的状态。否则读者可能以为十项都在十毫秒内完成。第十二课给出的 coverage 为零点二；若八项被取消，是否进入服务目标分母需要预先定义。本章 good_fraction 使用全部提交作为分母。

### 题三：一个输出 token 的请求，平均 ITL 应填什么

答案是 None，因为没有两次输出之间的间隔。一个输出仍可以有 TTFT 和端到端时间。零毫秒意味着实际观察到两个输出同时发生，与没有任何间隔样本不是一回事。下面用 Request 的真实 metrics 接口验证，不自己编一个恰好满足答案的字典。

```python
one = Engine()
one.submit(Request("one", (1, 2), 1))
one.run()
one_metrics = one.requests["one"].metrics()
assert one_metrics["output_tokens"] == 1
assert one_metrics["ttft_ms"] is not None
assert one_metrics["mean_itl_ms"] is None
```

### 题四：为什么 cancel 返回假并不一定是错误

答案是请求可能已经完成、已经取消，或者该身份不属于当前 worker 的活跃集合。唯一拥有者按命令处理顺序决定结果，不能倒转已经发布的终态。本章重复取消实验明确要求第二次为假。如果业务希望区分 unknown 与 already_finished，需要保存有界终态索引或返回更细状态；这是一项接口设计，不应偷偷改成“所有取消都返回真”。

### 题五：Future.result 超时以后，页会自动释放吗

答案是不会。超时只停止当前调用者的等待，Engine 可能仍在工作。本章 call_frontend 在超时分支显式发送取消命令并等待处理；关闭也会统一中止剩余项。若使用者直接丢弃 Ticket，又不执行取消或关闭，任务可能继续完成。资源生命周期必须跟随明确协议，不能依赖某个 Python 局部变量离开作用域。

### 题六：让关闭时仍有工作，验证所有结果有终态

完整答案是第八课 limited_worker，也可以用下面三个请求验证。手动模式确保关闭前尚未推进，取消结果为空输出；关闭后的快照应无活跃请求、无历史、全部页归还，线程不再存活。不要让测试先 run 完所有任务再 close，那无法覆盖关闭中止路径。

```python
stopping_worker = OwnedWorker(automatic=False)
unfinished = [stopping_worker.submit([1, 2, 3], 10) for _ in range(3)]
shutdown = stopping_worker.close()
assert all(t.result.result(timeout=5).state == "cancelled" for t in unfinished)
assert shutdown["pending"] == 0
assert shutdown["free_pages"] == shutdown["total_pages"]
assert not stopping_worker.thread.is_alive()
```

### 题七：为什么不能把 requests 字典直接返回给 HTTP 线程

答案是字典里的 Request、output 列表与缓存对象仍由引擎使用，返回它们会泄露修改能力。即使调用者承诺只读，另一个线程也可能在遍历中遇到状态变化。快照应复制所需字段；本章输出是不可变 Completed，状态观察是重新构造的字典与 tuple。复制快照的代价要承认，但不能为省几行代码把所有权打破。

### 题八：active_limit 为一，第二个提交失败，是否需要取消第一个

答案是不需要。拒绝发生在第二个任务进入 Engine 之前，第一个任务没有违约，应该继续运行或由自己的拥有者取消。全局清空会把局部容量问题扩大成其他请求丢失。第八课已经验证第二个 Future 得到 ValueError，而第一个只有在 close 时才变 cancelled。错误隔离要检查其他请求仍可进展。

### 题九：三 token prompt，输出上限三，第一次 advance 后缓存多长

答案是四，因为第一次输出尚不是终态，采样后还要消费这个新 token，以便得到第二次预测。第二次 advance 后长度五；第三次产生最后一个输出并清空缓存，因此终态不是长度六。下面实际验证每层长度与最终释放，注意使用请求的 status 区分两条规则。

```python
length_runner = IncrementalRunner()
length_runner.start("length", (1, 2, 3), 3)
for expected_length in (4, 5):
    length_runner.advance("length")
    assert length_runner.snapshot()["length"]["cache_lengths"] == [
        expected_length] * length_runner.model.config.layers
length_runner.advance("length")
assert length_runner.snapshot()["length"]["cache_lengths"] == []
assert length_runner.snapshot()["length"]["status"] == "finished"
```

### 题十：完整输出 token 相同，是否说明在线注意力集成正确

答案是否定的。错误 logits 仍可能具有相同 argmax，因此 token 相同是较弱证据。第十课比较每轮 next_logits 与完整前向，并检查每层缓存长度。进一步可以比较中间 attention 输出、不同位置与不同 KV 头配置。容差应来自 dtype 与计算顺序，不应不断增大到错误也能通过。

### 题十一：真实上游变异测试报 ModuleNotFoundError，算检测成功吗

答案是不算。它没有进入 schedule，更没有到达 total 与 counts 的断言。应该将结果标为环境阻塞，保留异常类型与缺失依赖；准备好固定版本环境后重新执行原始通过、变异失败、恢复通过三步。静态 AST 解析成功也不能代替这些步骤。source_evidence 中明确把两种状态分开记录。

### 题十二：写一个完整的交付摘要，避免把模拟器叫成真实 GPU 基准

参考摘要如下：本项目在标准库与 NumPy CPU 环境验证回放预算守恒、二十四项并发提交中的八项取消、唯一 Engine 修改线程、关闭全部归还页、两种调度顺序下真实多层 K/V 输出等价，以及模型异常发生前不提交输出。延迟来自 Engine 的模拟代价；上游 pytest 与 GPU 前向未执行。正文完整代码、固定输入与故障注入允许读者复验这些结论。摘要中的每一项都能对应本章一个实际断言，不能添加没有测过的性能提升。

## 第十四课：把四个阶段的结果打包成可复查交付物

第一阶段交付固定输入与回放报告，包含 trace 指纹、配置、逐请求指标、成功取消与失败分母。第二阶段交付 OwnedWorker、前端函数与并发生命周期测试，尤其保留关闭时仍有请求的案例。第三阶段交付真实数值集成，保存每层状态与 full/step 对照。第四阶段交付源码测试文件身份、静态结果和真实执行状态，已经执行的写通过，未执行的明确写未执行。

这四个阶段在本章都有完整实现和验收，并不是等待未来填空。你独立练习时应先不看参考重写一个阶段，再用同一断言比较，而不是直接复制所有代码后只换文件名。若新实现采用不同内部结构，允许它不同，只要公开合同与证据成立；课程不是要求背诵某个类的行数。

下面把已经产生的实际结果组装为 JSON，并进行文件往返验证。临时目录只是保证示例运行后自动清理，不把结果意外写到公司路径。自己的项目报告应保存在版本管理之外的实验输出目录，提交课程练习时只提交允许公开的合成输入与摘要，保留代码和配置身份。

```python
import tempfile

capstone_evidence = {
    "kind": "CPU_correctness_and_synthetic_replay",
    "trace_sha256": fingerprint,
    "replay_runs": [
        {"policy": row["policy"], "budget": row["budget"],
         "summary": summarize(row["detail"])}
        for row in comparison["runs"]
    ],
    "concurrent_manual": {"submitted": 24, "cancelled": 8, "finished": 16},
    "concurrent_auto_finished": len(automatic_results),
    "frontend_finished": len(replies),
    "numeric_logits_checks": numeric_checks,
    "numeric_model_policy_combinations": len(numeric_runs),
    "source": source_evidence,
    "gpu_executed": False,
    "api_called": False,
}
with tempfile.TemporaryDirectory() as temporary_directory:
    destination = Path(temporary_directory) / "capstone-evidence.json"
    destination.write_text(json.dumps(capstone_evidence, ensure_ascii=False, indent=2))
    restored_evidence = json.loads(destination.read_text())
    assert restored_evidence == capstone_evidence
assert capstone_evidence["concurrent_manual"]["submitted"] == (
    capstone_evidence["concurrent_manual"]["cancelled"]
    + capstone_evidence["concurrent_manual"]["finished"])
```

报告中的并发数量来自当前固定测试配置。若你改变为三十项请求，却保留二十四这个字段，就会让报告与实验脱节。更完整的工程可以把客户端结果自动汇总；本章已经通过结果枚举逐项断言固定数量，并将固定数值写在同一章，便于初学者核对。性能报告则不能只写计数，必须增加环境、时间窗口、分布和误差解释。

需要硬件时，再把一个明确假设交给 S15 的有界实验。例如“在相同总预算下，较小 prefill chunk 是否降低短请求首延迟”。先固定模型与输入，再记录设备、精度、版本、上下文、并发与样本数。个人 Win11 电脑适合随时运行本章 CPU 练习；Linux 服务器 CPU 与 Docker 可以承担依赖、源码和服务实验。H100 只在真正研究 GPU 前向、带宽或多卡通信时使用，先单卡短窗口，再根据问题增加规模。

在上卡前还可以手算一个缓存账本。二十四层、四个 KV 头、每头六十四维、每元素两字节，一条 token 的全部层 K/V 为二乘二十四乘四乘六十四乘二，等于二万四千五百七十六字节。四个请求各一千零二十四位置，总计一亿零六十六万三千二百九十六字节，也就是九十六 MiB。这个数不含权重、激活、临时工作区与管理开销，因此只是缓存项，不是所需显存总数。

```python
kv_bytes_per_token = 2 * 24 * 4 * 64 * 2
kv_total = kv_bytes_per_token * 1024 * 4
assert kv_bytes_per_token == 24576
assert kv_total == 100663296
assert kv_total / 2**20 == 96
```

如果实测显存高于九十六 MiB，不应立刻归因于泄漏。先核对权重、预分配池、工作区、图缓存、dtype 与单位，再检查活动引用。如果服务结束后预分配数组仍保留，但 free 页全部可复用，这和引用没有归还不同。S18、S19 已分别给出元数据与真实数值的释放工作坊，综合报告应沿同一合同解释差异。

## 第十五课：最终答辩，把每个承诺指向一个证据

最终答辩不要求背诵框架名词。给定一个请求，你应该能从前端参数副本走到队列命令，从拥有者线程走到 Request，从调度预算走到页表，从模型 next_logits 走到输出 Future，再从取消或完成走到资源归还。每跨一个边界，都指出身份、长度、状态与谁有修改权。解释不清的地方通常就是隐藏的竞态或计数混淆。

现场可以依次改变五个条件：提示变长、输出为零、提交数量超过上限、取消重复到达、模型计算抛错。你的系统应该给出有定义的结果，或者明确拒绝，而不是靠调整 seed 直到输出正常。随后把同一数值模型换成在线注意力实现，说明你比较了哪些中间结果以及哪些范围仍不支持。

完成本章之后，你拥有的是可复现的工程基线与定位方法。它还不是完整生产服务，但核心代码、答案和跨层验证已经在这里实现。下一步进入 [S17](17-source-workbook.md) 深追固定真实源码，再结合 [S18](18-state-workbook.md) 的资源审计和 [S19](19-numeric-paged-kv.md) 的数值分页。回到 Agent 项目时，可以用同一回放器观察工具结果让提示变长后的调度影响，用同一取消协议处理被用户中止的任务，用同一证据纪律区分模型质量、系统延迟与缓存正确性。
