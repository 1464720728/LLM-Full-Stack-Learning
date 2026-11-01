# S14 服务诊断：从一个症状追到第一处错误，再证明修复有效

先修基础卷的进程、文件、网络和 Docker，以及 S01–S13。诊断不是积累一张命令清单，而是不断缩小“错误最早可能发生在哪里”。你需要把用户看到的现象、框架内部状态、设备执行与资源所有权连接起来。最有价值的结果不是服务偶然恢复，而是你能用最小输入稳定复现、指出错误边界、做一个有针对性的修改，并证明正确性与资源终态都恢复。

本章包含可以独立执行的事件分析、资源守恒、指标聚合和本地 HTTP 故障实验。CPU 模拟器的时间仍是 synthetic；本地 HTTP 墙钟时间也不是 GPU 推理速度。真实 vLLM、SGLang、CUDA 与公司部署需要在 S15 的有界环境中验证，不能因为课程代码通过就声称已排除生产问题。

## 1. 第一课：把“卡住了”改写成可观察的故障定义

“不能用”包含许多互不相同的现象：连接被拒绝、连接成功但没有响应头、HTTP 返回四百、首 token 很慢、首 token 正常但之后停顿、输出提前结束、输出乱码、某个请求结果混入另一请求内容、进程退出、设备显存耗尽。先给现象一个边界，才能选择第一项检查。

一个合格的故障陈述至少包含触发、预期、实际和范围。例子：“固定两个请求同时进入，一长一短，短请求应该在有限轮内得到首 token；实际连续一百轮只有长请求获得预算，短请求一直 waiting；单独执行短请求正常。”这比“调度器性能不好”更有信息，因为它立即指向并发和公平性，而不是模型加载或 tokenizer。

再区分一次事件与趋势。单个请求超时可能来自偶发网络或长输入；每次请求完成后 free pages 少一页，且多轮单调下降，则更像资源归还问题。记录最近正常版本、首次异常时间和最小变化集，但不要把“某提交之后发生”直接当作因果证明；模型 revision、依赖、负载和缓存状态也可能同时改变。

| 观察 | 初始调查边界 | 还不能直接推出 |
|---|---|---|
| connection refused | 地址、监听、端口映射、网络拒绝 | 模型数值错误 |
| HTTP 404 | 已响应的 HTTP 服务及路径 | 已到达正确模型服务 |
| HTTP 400 | 请求 schema、长度、参数 | 所有服务状态都健康 |
| 首输出慢 | 排队、prefill、传输、前端缓冲 | decode kernel 一定慢 |
| 中途停顿 | decode 调度、背压、通信、超时 | 客户端一定断线 |
| free pages 持续减少 | 请求、缓存、在途引用 | GPU 分配器必然泄漏 |

状态码是路标，不是最终诊断。一个代理也可以返回四零四；四零零可能由网关校验而非模型服务发出；五百只说明服务端处理失败，不足以定位到 CUDA。要结合请求 ID、服务标识、响应体和对应日志，确认自己观察的到底是哪一层。

## 2. 第二课：画出请求经过的边界，时间戳必须带语义

最短的诊断链可以是客户端、HTTP 前端、引擎队列、runner、后端执行、输出流。更复杂的 Agent 还有 harness、工具进程、检索、环境容器和 verifier。先给每条边写输入输出与身份：请求 ID、attempt、模型 revision、token 数、开始结束事件。不要先抓所有日志，再期望从大量无关联文本中猜出调用顺序。

时间戳应说明来自哪个时钟。单进程内部可以用单调时钟计算持续时间；不同机器的墙钟即使都显示 UTC，也可能有偏差。跨机直接相减之前需要同步误差范围，或者使用各进程内部 duration 与消息因果关系。一个负的网络延迟通常首先提示时钟问题，而不是数据提前到达。

一次重试也必须有独立 attempt。用户逻辑请求可能只发起一次，但客户端、代理或 harness 重试后服务端处理了两次。若所有日志只保留同一个请求 ID，容易把两条执行轨迹拼成一个“不可能状态”。资源拥有者通常应绑定具体执行代次，不能让旧 attempt 的完成回调释放新 attempt 的页。

建议最小事件字段如下：schema version、request ID、attempt、iteration 或 step、component、event、monotonic duration 或本地 timestamp、输入输出计数、错误类别。不要默认记录完整 prompt；诊断大多数调度问题只需要 token 长度、公开合成输入与摘要。身份需要足以关联，内容需要尽量少而可控。

下面用人工事件展示一个合法生命周期检查器。它只验证这份明确 schema 的顺序，不是通用 trace 标准；遇到缺失事件会报告不完整，不能假装知道被截断日志之前发生了什么。

```python
def validate_lifecycle(events):
    states, errors = {}, []
    for index, e in enumerate(events):
        key = (e["rid"], e["attempt"])
        old = states.get(key, "absent")
        kind = e["kind"]
        allowed = {
            "submitted": {"absent"},
            "started": {"submitted"},
            "token": {"started", "streaming"},
            "finished": {"started", "streaming"},
            "cancelled": {"submitted", "started", "streaming"},
        }
        if kind not in allowed or old not in allowed[kind]:
            errors.append((index,key,old,kind))
            continue
        states[key] = "streaming" if kind == "token" else kind
    incomplete = [key for key,state in states.items()
                  if state not in {"finished","cancelled"}]
    return {"states":states,"errors":errors,"incomplete":incomplete}

events = [dict(rid="A",attempt=1,kind=k)
          for k in ("submitted","started","token","finished")]
assert not validate_lifecycle(events)["errors"]
bad = events+[dict(rid="A",attempt=1,kind="token")]
assert validate_lifecycle(bad)["errors"][0][3] == "token"
assert validate_lifecycle(events[:-1])["incomplete"] == [("A",1)]
print(validate_lifecycle(bad))
```

这里 finished 后再 token 是明确错误，因为本 schema 把 finished 定义为该 attempt 的输出终态。但生产日志中的“模型计算完成”可能先于网络发送完毕，如果它也叫 finished，就不能直接套该检查器。诊断前先统一事件语义，正是避免看似严谨的工具制造错误结论。

## 3. 第三课：从当前 Engine 的真实事件恢复调度事实

打开 `labs/serving/engine.py`。`event` 每次追加 iteration、time_ms、kind 与额外字段。`submit` 记录 submitted，`_ingest` 记录 enqueued，`_admit` 记录 admitted，调度后记录 batch，采样后记录 token，`_finish` 记录终态。allocation_blocked、cache_evicted、preempted 则解释没有直接输出时发生了什么。

batch 事件包含 work 列表，每项有 rid、start、count、phase。start 是本轮追加前的 computed，count 是这次安排的位置数，phase 依据本轮起点相对 prompt 判断。S02 已说明重算 chunk 可以跨 prompt 边界，所以不能只用 phase 名字推断每个位置都属于原始 prompt。精确分析应使用 start、count 和该请求的历史边界。

```python
from collections import Counter
from labs.serving.engine import Engine, Request

e = Engine(token_budget=4,chunk_size=3,max_sequences=2,pages=12,block_size=2)
e.submit(Request("long",(1,2,3,4,5,6),max_new_tokens=3))
e.submit(Request("short",(2,3),max_new_tokens=2))
report = e.run()
kinds = Counter(event["kind"] for event in e.events)
per_request = {rid:{"scheduled":0,"batches":0} for rid in e.requests}
for event in e.events:
    if event["kind"] == "batch":
        assert event["scheduled_tokens"] == sum(w["count"] for w in event["work"])
        assert event["scheduled_tokens"] <= e.token_budget
        for work in event["work"]:
            per_request[work["rid"]]["scheduled"] += work["count"]
            per_request[work["rid"]]["batches"] += 1
assert kinds["token"] == 5
assert kinds["finished"] == 2
assert not e.running and not e.waiting and not e.pool.tables
assert report["time_kind"] == "synthetic_ms_not_GPU_measurement"
print({"events":dict(kinds),"work":per_request,"report":report})
```

这段程序把事件作为事实来源，但仍需知道它的局限。Engine 的 simulated time 在 batch 执行时按固定成本增加，不能表示 CPU Python 开销或 GPU kernel 时间；admitted 事件不等于真实设备已经开始；token 事件代表教学引擎输出，不等于客户端已收到。把这些字段直接导入性能仪表盘而不标注语义，会制造漂亮但误导的图。

`report` 返回 outputs 时为每个请求创建新的 list。这是重要的观察隔离：调用者修改报告，不应修改引擎内部输出。相反，直接把内部可变字典引用交给诊断器，可能让“只读观察”改变系统。测试日志、指标与报告工具时也要考虑对象别名，不只是模型计算正确性。

```python
from labs.serving.engine import Engine, Request
e = Engine()
e.submit(Request("A",(1,2),max_new_tokens=2))
first = e.run()
expected = list(e.requests["A"].output)
first["outputs"]["A"].append(999)
assert e.requests["A"].output == expected
assert e.report()["outputs"]["A"] == expected
e.check()
print("report copies preserve engine state:",expected)
```

`check` 不是性能指标，而是状态不变量检查。它要求 running 无重复、running 请求确为 running、页池 token 等于 context 的 computed 前缀、页表拥有者集合等于 running 集合、输出与时间戳数量一致。它能在错误靠近发生点时失败，往往比最后出现乱码更容易定位。但生产中是否每轮执行所有昂贵检查，需要按开销设计抽样或调试模式。

## 4. 第四课：资源守恒比“显存看起来高”更能定位泄漏

一个物理页可以有多个引用，但它只能处于一套明确的物理生命周期中。最简模型中，free 与 allocated 是互斥集合；allocated 内再记录请求引用、缓存引用和在途引用。把所有引用数相加并不等于物理页数，因为共享页可能被多个拥有者持有。诊断泄漏要问“哪个对象仍拥有它”，而不是只问“占用了多少”。

下面用小型账本检查孤儿页、错误空闲页和重复引用归属。它不替代 PagePool.check，而是帮助你理解检查器应该寻找哪类矛盾。每个 owner 对同一页只能记录一次拥有；如果真实系统允许一个对象持有多份引用，需要把 multiplicity 显式建模。

```python
def audit_pages(total_pages, free, owners):
    all_pages = set(range(total_pages))
    free = set(free)
    refs = {page:[] for page in all_pages}
    errors = []
    if not free <= all_pages:
        errors.append(("unknown_free_page",sorted(free-all_pages)))
    for owner,pages in owners.items():
        if len(pages) != len(set(pages)):
            errors.append(("duplicate_owner_reference",owner))
        for page in pages:
            if page not in refs:
                errors.append(("unknown_owned_page",owner,page))
            else:
                refs[page].append(owner)
    for page,names in refs.items():
        if page in free and names:
            errors.append(("free_but_owned",page,names))
        if page not in free and not names:
            errors.append(("orphan",page))
    return {"errors":errors,"references":refs}

ok = audit_pages(4,{3},{"request:A":[0,1],"cache:P":[0],"transfer:T":[2]})
assert not ok["errors"]
bad = audit_pages(4,{2,3},{"request:A":[0],"cache:P":[0]})
assert any(e[0]=="orphan" and e[1]==1 for e in bad["errors"])
bad2 = audit_pages(4,{1,2,3},{"request:A":[0,1]})
assert any(e[0]=="free_but_owned" for e in bad2["errors"])
print(ok,bad,bad2)
```

正常例子里页零有两个引用，但只有一份物理载荷；页二由传输拥有，所以没有请求引用也不能回收。第二个例子页一不在 free，也没有任何拥有者，是孤儿。第三个例子页一既 free 又被请求持有，可能导致重新分配后两个请求写同一地址。后者往往表现为内容串扰，比单纯 OOM 更严重。

取消后的检查必须等待正确的阶段。用户取消一到，逻辑状态可以立即变 cancelled，但在途设备操作尚未完成时，页仍应由 draining 或 transfer 对象拥有。若诊断器强制所有 cancelled 请求的页立刻 free，会把正确的延迟释放误报为泄漏，甚至诱导开发者引入 use-after-free。

同样，prefix cache 保留页是设计行为，但必须有容量、驱逐和引用证明。压力升高时，驱逐一个缓存条目若物理页仍被活动请求持有，free 可能不增加。这不是驱逐失败；诊断应同时记录删除了哪个缓存引用、物理页剩余哪些引用，以及是否存在可回收候选。

## 5. 第五课：网络与容器先逐跳验证，不要在模型里修端口

客户端访问 URL 时，需要解析主机、建立连接、发送 HTTP、等待响应。连接被拒绝时，先确认目标主机和端口，以及服务是否监听；已经拿到 HTTP 响应时，再检查路径、方法和协议。把路径错误误认为模型问题，常见结果是反复重装框架却没有改变真正故障。

Linux 的 `ss -ltnp` 查看监听 TCP socket。l 表示 listening，t 表示 TCP，n 保留数字地址，p 请求进程信息；权限可能限制你看到其他用户进程名称，但监听地址与端口仍有价值。只查看和操作自己的实验服务，不批量终止别人进程。若端口已被占用，选择自己的新端口并同步客户端配置。

```bash
ss -ltnp
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
docker inspect --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}' course-serving-demo
docker logs --tail 100 course-serving-demo
```

这里的容器名是后续实验约定，尚未创建时 inspect 会报找不到对象，这是正常的信息而不是模型错误。docker ps 的 Ports 描述宿主发布与容器端口的映射；inspect 的 OOMKilled 是容器运行状态证据之一，不能把任意非零退出码都解释成 OOM。日志读取限制一百行，避免无意输出长时间业务内容。

容器中的 127.0.0.1 指容器自身，宿主机中的 127.0.0.1 指宿主机。服务在容器内只绑定 loopback，通常不能通过普通端口发布到容器外。常见课程配置是在容器内监听 0.0.0.0:8000，宿主只发布 `127.0.0.1:18080:8000`，让宿主本地访问，再通过已有授权的 SSH 隧道供你的 PC 使用。

SSH 本地转发的两个 localhost 也要分清：浏览器访问的是本机转发端口，SSH 连接远端后再连接远端 loopback 的服务端口。若容器服务健康但 PC 访问失败，逐步验证容器内、远端宿主、SSH 转发、本机客户端，哪一跳首次失败就查哪一层。不要先把监听地址改成公网可见来掩盖理解不足。

## 6. 第六课：实际启动本地 mock 服务，验证错误隔离与计数

本仓库 `mock_server.py` 仅实现 health、metrics 和一个 prompt_token_ids 的 completions 子集。它不是完整 OpenAI 协议，也没有真实 tokenizer 或 SSE。ThreadingHTTPServer 可接收多个连接，但引擎由一个 lock 串行拥有，因此“HTTP 有线程”不能证明模型连续批处理已接入。

`validate_payload` 先检查 JSON 对象、prompt 列表长度一到二百五十六、max_tokens 为零到六十四的严格整数，再构造 Request；Request 继续验证 token 类型与 seed 等。服务在锁内 submit 和 run，成功后 accepted 增加；输入错误增加 bad_request。输出返回后清除已终态请求历史与事件，避免这个长期 demo 无界积累每次请求记录。

下面程序启动自己的子进程，使用系统分配的空闲端口，先发非法请求再发合法请求，最后只终止自己创建的子进程。启动读取通过线程与队列设五秒上限，每次 HTTP 请求设两秒 timeout。该实验验证本机协议与资源行为，不衡量真实推理吞吐。

```python
import json,queue,re,subprocess,sys,threading,urllib.request,urllib.error

proc = subprocess.Popen(
    [sys.executable,"-u","-m","labs.serving.mock_server","--port","0"],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
lines = queue.Queue()
reader = threading.Thread(target=lambda:lines.put(proc.stdout.readline()),daemon=True)
reader.start()
def get(url):
    with urllib.request.urlopen(url,timeout=2) as response:
        return response.status,json.load(response)
def post(url,payload):
    request = urllib.request.Request(url,data=json.dumps(payload).encode(),
                                     headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(request,timeout=2) as response:
            return response.status,json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code,json.load(exc)
try:
    line = lines.get(timeout=5)
    match = re.search(r"http://127\.0\.0\.1:\d+",line)
    if not match:
        raise RuntimeError("server did not publish its loopback URL")
    base = match.group()
    assert get(base+"/health")[0] == 200
    status,error = post(base+"/v1/completions",
                      {"prompt_token_ids":[1,2,3],"max_tokens":65})
    assert status == 400
    before = get(base+"/metrics")[1]
    assert before["accepted"] == 0 and before["bad_request"] == 1
    status,result = post(base+"/v1/completions",
                       {"prompt_token_ids":[1,2,3],"max_tokens":2})
    assert status == 200 and len(result["token_ids"]) == 2
    after = get(base+"/metrics")[1]
    assert after["accepted"] == 1 and after["bad_request"] == 1
    assert after["active"] == 0 and after["free_pages"] == 256
    assert get(base+"/health")[0] == 200
    print({"error":error,"result":result,"before":before,"after":after})
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    proc.stdout.close()
    proc.stderr.close()
```

检查 free pages 为二百五十六之所以成立，是因为这个例子 prompt 三个 token，小于四位置整页发布边界，没有被 prefix cache 保留的完整 prompt 页。若改为更长 prompt，完成后 free 不一定回到全池，需要把缓存引用纳入解释。一个好的验收断言必须与输入机制对应，不能无条件复制到所有案例。

这个服务还有明确限制：线程数和慢请求体未做生产级 admission，锁内 run 阻止其他请求真正共享一轮调度，错误处理覆盖范围有限，metrics 也没有完整延迟分布。不要把它直接作为公司公开服务。它的用途是让你在很小的代码里看清校验、锁、引擎调用、计数和错误隔离，然后在项目中有意识地改造。

## 7. 第七课：计数器、仪表和直方图各回答不同问题

计数器记录累计事件，如 accepted、completed、cancelled、bad_request；通常单调增加，进程重启可能归零。仪表表示当前状态，如 waiting、running、free pages、inflight bytes，可增可减。直方图记录延迟分布，回答多少请求落在不同区间。把累计 accepted 当当前并发，或把 free pages 当累计释放次数，都会产生错误仪表盘。

请求 ID 不适合作为长期指标标签。每个请求产生一个新标签，会让时序数量无限增长。模型名、固定错误类别、有限状态通常更适合聚合；具体请求 ID 放到有保存期限的 trace 或日志中。指标负责发现范围，日志与 trace 负责解释具体事件，两者通过有限的样本链接，而不是把全部细节塞进标签。

直方图可以采用累计桶：延迟小于等于某阈值的请求数。桶越粗，对尾部分位数的估计越不精确；原始样本才能精确使用 S08 的指定插值方法。不要把每台机器的 p95 简单平均成全局 p95，因为分位数不是线性量。合并原始样本或兼容的桶，再计算整体分布。

```python
def cumulative_histogram(values,bounds):
    if bounds != sorted(set(bounds)):
        raise ValueError("strictly increasing bounds required")
    if any(v < 0 for v in values):
        raise ValueError("nonnegative durations required")
    buckets = [(bound,sum(v<=bound for v in values)) for bound in bounds]
    return {"buckets":buckets,"count":len(values),"sum":sum(values)}

a = cumulative_histogram([1,2,8],[2,5,10])
b = cumulative_histogram([3,9],[2,5,10])
merged = [(x[0],x[1]+y[1]) for x,y in zip(a["buckets"],b["buckets"])]
assert merged == [(2,2),(5,3),(10,5)]
assert all(x[1] <= y[1] for x,y in zip(merged,merged[1:]))
print({"a":a,"b":b,"merged":merged})
```

这个教学函数没有处理 NaN、无穷、单位标签和分布式重置，生产接口需要补齐。它清楚展示可相加的是相同边界桶计数，而不是 p95 本身。无论用哪种监控系统，都必须先统一延迟起止、单位、成功失败是否纳入，以及无输出请求如何处理。

accepted 与 completed 之差也不一定等于当前 active。如果包含取消、错误、等待或正在传输，需要完整守恒式。例如 accepted 等于 waiting 加 running 加 draining 加 finished 加 cancelled 加 failed，前提是每个请求只在一个互斥逻辑状态中计数，且统计窗口一致。累计终态与当前状态混用时，还要考虑窗口开始前已存在的请求。

## 8. 第八课：超时是多层协议，重试可能把负载放大

客户端 connect timeout、read timeout、代理 idle timeout、服务 deadline、Agent 单步预算、任务总预算和 verifier 预算是不同计时器。read timeout 往往限制等待下一段数据的间隔，不一定限制整个流的总时长；一个持续缓慢输出的服务可能不断刷新读取等待，但总任务仍然很久。具体行为需要读取客户端库或 harness 实现。

超时之后谁还在运行，是最重要的问题。客户端停止等待，不代表服务已收到取消；服务标记取消，不代表 GPU 在途工作已结束；工具超时，不代表其创建的子进程全部退出。诊断要沿调用链记录取消传播和资源终态，不能只把上层异常捕获后写“已回收”。

重试还会放大压力。假设服务处理一个请求需要五秒，客户端两秒就重试三次，服务端若没有幂等去重或及时取消，可能同时处理多个相同工作。用户看到的是超时，系统看到的是负载增加，排队更长又导致更多超时。简单增加重试次数可能使故障恶化。

幂等键也不是万能。模型采样、工具副作用和流式输出都有独立语义；某次请求已向客户端输出一半后重试，不能悄悄把另一条随机轨迹拼接在后面。服务应明确 attempt、已交付偏移和重试策略。涉及有副作用的工具时，需要 Agent 卷中的执行确认与幂等记录，推理服务不能替工具保证事务。

固定 EvalScope 来源展示了为什么要追参数落点：外层评测调度、独立 agent、环境执行与 verifier 可能使用不同模型接口和超时配置。某个参数名看似通用，并不保证作用于所有层。诊断报告应写实际调用路径与配置传递结果，不能凭项目之间的集成关系推断语义相同。


### 8.1 线程还活着不等于系统还能前进

服务进程存在、健康端点返回二百，仍可能没有请求能获得执行。健康检查如果只回答“进程活着”，不能证明调度器、worker 与输出链可用。可以把 liveness 与 readiness 分开：前者关注进程是否需要重启，后者关注是否准备接收工作；更深的功能探针则执行一个受限的合成请求。但探针本身也消耗资源，不能频繁提交长生成把共享服务压垮。

死锁、活锁和饥饿也需要分开。死锁通常是互相等待且没有状态变化；活锁有持续事件但没有有效进展，例如反复抢占并立即重新接纳相同请求；饥饿则是系统总体在工作，某个请求长期得不到服务。仅看 GPU 利用率或日志是否持续输出，无法区分这些情况。

诊断可以定义进展向量：已完成请求数、已交付 token 数、某个最老请求的 computed、可回收资源数量。观察一个有限窗口内哪些分量变化，并结合系统目标判断。重算可能让 computed 增加却没有新输出，所以“computed 变化”只是局部进展；若总是回退再重算，用户仍然没有进展。S05 与 S18 的前进证明比单纯每隔几秒打印心跳更可靠。

线程栈也应按拥有者解释。若 HTTP 线程等待 engine lock，而引擎线程等待输出队列空间，客户端又因某个协议条件不读取，就可能形成背压闭环。修复不是把所有队列改为无限，而是明确谁负责继续消费、谁可以取消，以及取消如何解除等待。无界缓冲只是把等待换成内存增长，最终可能以 OOM 结束。

对于 GPU worker，调度进程等待 future 并不自动意味着 GPU 卡死；worker 可能仍在编译、加载、通信或等待输入。先关联任务 ID 与阶段，再抓有限窗口的栈或执行状态。缺少关联信息时，随机终止进程只能恢复表面可用性，不能解释根因，也容易丢掉最有价值的现场证据。


## 9. 第九课：把 OOM 按阶段和对象分类，再决定最小实验

启动时 OOM，优先看权重加载、转换副本、并行放置、运行时初始化与图捕获。首个请求才 OOM，优先看首次 shape 触发的编译、工作区、KV 配置和懒初始化。并发增加才 OOM，重点看活跃状态、临时批次与在途缓冲。长时间后单调增长，则检查未清理请求、输出缓冲、缓存边界和日志保留。

每类都需要 S13 的账本支持。记录设备预算、模型与 rank 布局、加载后峰值、KV 池总块、当前活动页、图规格和请求长度。不要只截图一个 nvidia-smi 数字。它能说明设备层占用，但无法独自告诉你哪一层对象持有内存，也不能解释框架预分配池内部的可用情况。

共享卡出现其他任务占用时，先确认本次允许的设备与预算。不能通过杀掉别人的进程或把利用率调到接近全部容量来“修复”。缩小自己模型、长度或并发，或者换到已分配资源，是合法的实验控制。课程要求清理的是自己创建的进程和容器，名称、PID 与生命周期应可追踪。

CUDA 错误也不全是可恢复的普通内存异常。非法地址访问、通信组失败、设备异常与普通输入校验不同；某些错误后上下文或 worker 已不可靠。真实框架的恢复策略需要查实现和日志，不能把所有异常放进 while True 重试。先保存最小复现，再以明确的进程重建路径恢复，避免在损坏状态上制造更多噪声。

OOM 负控制不需要真的把共享 GPU 撑爆。可以先在 CPU 页池中把预算缩到确定不足，检查分配失败是否原子、请求是否得到明确错误、其他请求是否继续。真实 GPU 容量边界实验也应设置保守停止条件，用已观察的峰值接近阈值来停止，而不是把设备错误当唯一探测手段。

## 10. 第十课：性能回归要层层剥离，快了也要检查输出

从小到大建立四个基线：固定算子的数值与耗时、固定模型 forward、固定服务请求、完整 Agent 任务。若算子没变而 HTTP TTFT 增大，调查排队、序列化和前端；若单请求稳定而并发恶化，调查预算、资源、通信和公平性；若模型服务稳定而 Agent 变慢，调查工具轮数、检索、上下文增长和重试。

比较必须控制实际工作量。输出从一百 token 变成二十 token，完成更快不一定是 runtime 优化；丢掉最后几段或提前 EOS 也可能让吞吐看起来好。输入长度、输出长度、完成原因、错误率、缓存命中、重算 token 和逻辑任务数都应同时报告。先通过正确性，再解释性能。

GPU 提交是异步的，Python 调用返回可能只表示工作已入队。计时要使用适合的同步边界或设备事件，并说明是否包含准备、传输与同步。同步本身会影响重叠，所以端到端和局部 kernel 时间应分别报告。CPU 仿真时间不能用设备计时修正后直接当真实框架性能，它没有相同执行路径。

Profiler 应在已定位的短窗口使用。全量长 trace 会增加 CPU、磁盘和记录开销，甚至改变被观察系统。先用轻量计数确定是哪一轮、哪一类请求异常，再采一小段执行。保存 profiler 类型、版本、捕获范围与开销对照；若开启 profiler 后现象消失，也是一条调度或时序敏感线索。

下面演示配对比较，要求相同请求 ID、相同输入输出工作量和相同成功状态。程序只计算人工延迟的变化，不推断统计显著，也不把不同长度混在一起。

```python
def paired_latency(before,after):
    if set(before) != set(after):
        raise ValueError("same request cohort required")
    rows = []
    for rid in sorted(before):
        a,b = before[rid],after[rid]
        for key in ("input_tokens","output_tokens","ok"):
            if a[key] != b[key]:
                raise ValueError(f"{rid}: work or success changed")
        if not a["ok"]:
            continue
        rows.append({"rid":rid,"delta_ms":b["e2e_ms"]-a["e2e_ms"],
                     "ratio":b["e2e_ms"]/a["e2e_ms"]})
    return rows

base = {"A":dict(input_tokens=8,output_tokens=4,ok=True,e2e_ms=20)}
new = {"A":dict(input_tokens=8,output_tokens=4,ok=True,e2e_ms=24)}
assert paired_latency(base,new)[0]["ratio"] == 1.2
wrong = {"A":dict(input_tokens=8,output_tokens=2,ok=True,e2e_ms=10)}
try:
    paired_latency(base,wrong)
except ValueError:
    pass
else:
    raise AssertionError("shorter output is a different workload")
print(paired_latency(base,new))
```

真实基准还需处理零时长、缺失值、非有限输入与重复试验。这里重点是把“同一工作量”变成程序前提，而不是靠报告作者口头保证。若优化的目标本来就是减少输出长度或工具轮数，应把它定义为产品策略变化，另外评价任务质量，不能混写成算子加速。

### 10.1 配置差异本身也是证据，需要保存有效配置

启动命令不是全部有效配置。环境变量、配置文件、模型默认值和框架自动选择都可能改变运行路径。两次看似相同的命令，可能选择了不同 attention backend、不同 dtype 或不同图规格。报告应保存框架最终解析后的关键配置，以及哪些值来自默认、哪些值由显式参数覆盖。

依赖版本也要适当分层。Python 包版本描述分发版本，Git commit 描述源码快照，模型 revision 描述权重与配置，容器 digest 描述镜像内容。只记“用了最新版”无法复现；只记仓库 commit 也不能保证已安装 wheel 来自该 commit。应在同一运行环境中核对导入模块位置与版本，防止在一个目录编辑源码，却运行另一个虚拟环境的已安装副本。

做版本二分时先确认故障可重复，并固定外部条件。如果故障只在高并发偶发，某个中间版本一次没失败不能直接标为 good。可以采用固定种子与合成到达序列重复有限次，把“不确定”保留为第三种状态，而不是强迫所有结果变成好坏。二分得到的是候选引入区间，最终仍要读差异并验证因果机制。

配置变更应逐项归因。先从已知正常基线只改变一个轴，再用小型组合实验确认交互。若同时更新框架、模型、量化、并行和缓存，出现回归后难以判断哪个变化负责。紧急恢复可以回到整套已知正常配置，但后续修复需要更小的比较，不能把一次回滚成功当作对某一行代码的证明。


## 11. 第十一课：最小复现、假设表和负控制组成一套闭环

最小复现的目标是保留导致故障的最小条件。先减少请求数量，再减少长度和模型规模；每次缩减后重新确认故障仍存在。若原问题依赖两个请求共享前缀，把第二请求删掉就不再复现，那么两请求就是必要条件，不能为了“更小”把关键机制删掉。

建立一张假设表，每行写观察、可能解释、区分实验、预期结果和当前状态。例如观察是输出串扰；假设一是 packed attention mask 漏掉 owner 边界，实验是独跑与混批 logits 对比；假设二是页表错位，实验是给每页填不同数值并检查地址映射。两个假设可能造成同一文本症状，但验证手段不同。

负控制是故意破坏一个已知条件，确认检查器会失败。若你把页表顺序调换，数值对照仍通过，可能 oracle 与被测实现共享了同一个错误映射；若丢一个输出 token，基准仍报告成功，说明验收只看 HTTP 状态。负控制应只发生在独立副本或专门故障开关中，结束后恢复并验证，不能污染正常源码。

修复要尽量落在第一处契约被破坏的位置。输出错位若由 request ID 映射错误导致，不能在最终文本层按经验重新排序；那可能掩盖更复杂并发下的问题。最小补丁之后保留一个能在旧版本失败、在新版本通过的回归。不要给每行实现机械写一个镜像测试，而要验证真正的状态边界和外部行为。

提交报告时把观察与解释分开。观察是“第六轮 A 的页表引用物理页二，但页二已经分给 B”；解释是“取消后提前回收”；证据是事件顺序和引用账本；补丁是延迟到完成事件释放；验证是正常、取消、重复完成、旧代次与新请求复用场景。这样别人不需要相信作者直觉，也能审查因果链。

### 11.1 一份可审查报告怎样写到无需猜测

报告开头用两三句说明触发与影响：“固定前缀的两个请求共享尾页，取消第一个请求后，第二个请求下一轮输出与独跑不一致；没有共享时正常。”随后给出最小公开输入与命令，说明预期结果来自哪个独立参考。不要先粘贴几千行日志，让读者自行寻找真正的问题。

环境部分只保留能影响复现的内容：版本、模型身份、dtype、页大小、并行、相关设备与启动配置。密钥、公司端点、用户名、私有路径和业务 prompt 不属于公开复现材料。用合成 token 或公开文本复现后，原业务数据通常无需出现在报告里。日志中的错误栈也可能包含路径或请求内容，分享前应检查。

证据部分按照因果顺序列最短片段：共享建立、取消、错误释放、下一次分配、数值分歧。每条事件有一致身份和代次，读者可以核对资源账本。长 trace 可以作为独立附件，但正文应给出定位窗口与字段解释。没有上下文的最后一行异常，往往只是损坏传播后的结果。

补丁部分解释恢复了哪个契约，以及为什么不会破坏其他路径。验证部分列旧版本失败、新版本通过、负控制仍能被发现、相关现有回归通过，并明确未运行的 GPU 或上游路径。把这些限制写清楚，不会削弱报告；它让维护者知道还需要哪一步证据，避免把 CPU 参考通过误当成所有后端已验证。

最后保留一个复现脚本的退出状态约定：成功验证返回零，发现预期故障或验收失败返回非零，依赖缺失明确标记为未运行。不要在脚本里捕获所有异常后仍返回零，也不要把 skip 打印成 pass。自动化系统依赖这个简单约定，才能让持续集成结果与人类报告保持一致。

## 12. 第十二课：四个故障工作台与完整练习

**工作台一，非法输入污染服务。** 使用本章 HTTP 程序发 max_tokens 六十五，预期四百，bad_request 增加而 accepted 不变；随后合法请求成功，active 回零。若 accepted 提前增加，检查计数发生在校验前还是执行后；若服务退出，检查异常是否跨越请求边界。答案由实际计数和后续健康请求共同支持，不只看第一次响应。

**工作台二，取消后页泄漏。** 在独立页账本中增加一个 cancelled 请求但不删除其 owner，观察孤儿或多余引用。修复应区分没有在途工作可立即释放，与有在途工作必须转 draining。最后所有请求终态、缓存清空、传输完成后才要求全池归还。不能用取消瞬间的 free 数断言泄漏。

**工作台三，性能提升来自丢输出。** 把 after 的 output_tokens 改小，paired_latency 必须拒绝比较。进一步检查真实基准是否把失败和超时从分母删除，造成幸存者偏差。修复报告需同时给成功率、完整工作量和延迟；正确性未通过的速度不作为优化验收。

**工作台四，观察器改变系统。** 获取 report 后修改 outputs，内部输出应不变；若直接引用内部列表，则诊断工具可能制造额外 token。修复返回副本或不可变快照，并检查事件和嵌套对象是否也有别名。只复制最外层字典不足以隔离内部 list。

**练习一，四零四。** 它证明有 HTTP 响应者，但不证明是正确模型服务。完整检查包括主机端口、响应标识、路径、方法和代理链。不要先改 dtype、TP 或模型权重。

**练习二，两个 localhost。** 容器 loopback 属于容器，宿主 loopback 属于宿主。普通端口映射需要容器服务对容器网络接口监听；宿主可以仅发布 loopback，再用 SSH 转发。答案要逐跳说明，不用一个“开放端口”替代网络模型。

**练习三，事件终态。** 本章 schema 中 finished 后 token 非法，截断到 token 则 incomplete。若生产 finished 指模型完成而非网络完成，应改 schema 名称与允许状态。检查器正确与否依赖事件定义。

**练习四，共享页。** 页零被请求与缓存共同持有，引用二、物理页一。不能用引用和当占用页数。驱逐缓存后请求仍持有，free 不增加是合法结果。

**练习五，孤儿。** 页不在 free 且没有任何合法 owner，是可疑孤儿；页既 free 又有 owner，是更危险的重复分配风险。完整答案应列出请求、缓存、在途与保留对象，避免漏掉合法拥有者后误报。

**练习六，累计与当前。** accepted 是累计，active 是当前；两者不能直接相等。构造 accepted 三、finished 一、cancelled 一、running 一的状态，守恒成立。若还有 waiting 或 draining，应加入对应互斥项。

**练习七，分位数合并。** 两台机器的 p95 不能简单平均。合并原始样本或同边界直方图，重新计算整体分位数。桶只提供范围估计，不能冒充原始精确样本。

**练习八，标签。** 请求 ID 放长期指标标签会产生无界基数。固定错误类别与模型配置适合有限聚合，具体 ID 放 trace。保留期限和采样需要明确，不能无限积累每个 prompt。

**练习九，read timeout。** 它未必是整个请求 deadline。需查实际客户端语义，区分连接等待、相邻数据等待和总任务时长。客户端超时之后还要验证服务取消与资源排空。

**练习十，重试。** 五秒工作被两秒超时重复提交，若服务不去重或取消，会放大负载。答案需列逻辑请求与 attempt，说明部分输出和随机轨迹不能随意拼接，不能只建议无限增大重试次数。

**练习十一，预分配。** reserved 不下降但逻辑 free 恢复，可以是正常池复用；逻辑 free 逐轮下降且无 owner 解释，才应追泄漏。结合对象账本与生命周期，不只看设备总占用。

**练习十二，负控制。** 调换独立副本页表或删除一个输出，应使数值或协议验收失败。若仍通过，先修检查器。负控制要恢复现场，保留故障开关或临时对象，不能把损坏提交为正常实现。

**练习十三，GPU 计时。** Python 返回可能只代表提交。使用明确同步边界或设备事件，同时说明包含哪些阶段；同步会影响重叠，因此局部与端到端分别记录。CPU synthetic 时间不能转述成真实 GPU 毫秒。

**练习十四，最小补丁。** ID 错位应修结果映射，而不是最终文本排序。回归应包含批次重排、取消后旧结果和不同长度，证明映射契约恢复。用一个固定顺序 happy path 不能覆盖问题本质。

项目第一阶段运行本章 CPU 与本地 HTTP 实验，保存事件、计数和资源终态。第二阶段给 Engine 添加独立分析脚本，输出每请求排队轮数、调度 token、抢占与终态，不修改引擎内部对象。第三阶段构造一项已知故障并完成负控制，写出最小复现和补丁解释。第四阶段在真实框架短实验中采集同类证据，清楚标记哪些字段只是近似映射。第五阶段让另一位读者仅凭报告复现，不能靠口头补充关键环境。

掌握门槛是能从一个具体症状选出有理由的第一项检查，能分清事实、假设和证据边界，能用状态与资源守恒定位错误，并能证明修复没有靠丢工作“变快”。下一章 [S15 硬件实验](15-hardware-labs.md) 把这些能力带到你的 Windows PC 与共享 H100 环境。
