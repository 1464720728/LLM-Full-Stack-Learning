# S17 源码工作坊：把一条真实请求追踪到可检测错误的回归测试

这一章的目标很具体：给你三个各有八百个输入 token 的请求，你能在固定版本的 vLLM 中指出它们从哪个函数进入调度器，手算每轮安排的工作，解释请求对象何时改变，说明结果怎样按请求身份回写，并写出一个故意制造错误就会失败的测试。最后再把这套方法迁移到 SGLang。读完之后应当留下程序、表格和测试记录，不能只留下“看过 Scheduler”的印象。

这里需要 F07 的对象与引用、F09 的断言和测试、F11 的并发基本词义，以及 S01—S06 的请求和页池。若你还分不清列表下标与字典键，请先完成第一课的小实验；若你还不能解释 prefill 为什么会没有新输出，先回到 M06。你不需要先会编写 CUDA kernel。这次主要研究 CPU 端如何安排与管理工作，真正的 GPU 前向属于另一个执行层。

本章有两类程序。带完整定义的教学程序只使用 Python 标准库，可以按正文顺序放进一个文件执行；后面的代码块会复用前文定义。上游测试则导入固定 vLLM，必须在单独准备的开发环境运行。本次编写实际执行了教学程序，但没有安装上游框架并运行其 pytest，更没有用 H100 做性能测试。代码语法检查、接口阅读、CPU 语义实验和真实框架执行是四种不同证据，下面会分别说明。

## 第一课：先决定要回答什么，再决定要读哪段源码

初学者常从仓库第一层目录开始，把每个文件名看一遍，几个小时以后知道有很多模块，却仍不能解释一条请求。这不是记忆力问题，而是没有给阅读设定可终止的问题。本章把问题缩小成“第一轮为什么安排四百、四百、二百二十四”。这个问题已经限定了输入、输出和一个可核对的数字，因此能够反过来筛选需要阅读的函数。

“源码”不只包括实现文件。测试写出了作者如何构造对象以及期望哪些结果；配置对象决定分支是否启用；输出类型说明调用方实际拿到什么；辅助函数可能偷偷改变默认值。只读 `schedule` 而不读构造它的配置，类似只看一道菜的烹饪步骤却不知道原料。相同的函数，前缀命中、异步调度或不同预算都会走不同路径。

先认识本章反复出现的五个词。请求是服务方持续管理的一项任务；计划是某一轮为若干请求安排的工作描述；执行器把计划交给实际计算；结果是计算完成后带身份的数据；回写把结果合并回仍然存活的请求。请求会跨越很多轮，计划只描述某一轮，因此它们不能简单视为同一个对象。

下面用十行数据观察身份与位置。模型结果数组的位置由本批次决定，不能拿请求名字猜数组下标。

```python
request_ids = ["A", "B", "C"]
result_order = ["C", "A", "B"]
sampled = [[31], [11], [21]]
index = {rid: i for i, rid in enumerate(result_order)}
routed = {rid: sampled[index[rid]] for rid in request_ids}
assert routed == {"A": [11], "B": [21], "C": [31]}
wrong = dict(zip(request_ids, sampled))
assert wrong != routed
print(routed)
```

`enumerate` 同时提供下标和元素；字典保存“请求 ID 到当前数组行”的映射。最后一个断言刻意保留错误答案作为反例。如果测试数据一直让请求顺序等于结果顺序，错误的 `zip` 也会通过。这就是为什么后面测试要改变排列：不是增加随机性显得高级，而是让一个真实错误无法藏在巧合里。

建立一份阅读笔记，每个函数写六项：本次输入值、读取的状态、修改的状态、返回的对象、可能提前离开的条件、执行结束后的保证。注意“输入是 SchedulerOutput”还不够，应该写“字典为 A→400、B→400、C→224”。类型帮助理解结构，具体值帮助检验分支，两者都要有。

先运行课程已经准备的入口，确认环境和基本词义：

```bash
python -m labs.serving.experiments --lab token_budget
python -m labs.serving.source_walk --project vllm --show-pin
```

第一条执行本课程的参考实现；第二条只打印固定源码版本，不安装也不启动 vLLM。若第一条失败，先修本课程环境，不要同时引入大型框架依赖，否则一个错误会叠上多层不确定性。

## 第二课：固定版本、定位符号、保留能够复查的证据

本章使用 vLLM 提交 `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`，SGLang 提交 `fa5580f6d0e969a3d44c603d215e2139886e8db2`。文件链接和阅读边界见 [本卷来源表](SOURCES.md)。提交标识是一份内容快照的身份；分支名 `main` 则会移动。今天同名函数里的第一个分支，几周后可能已经迁移到辅助对象，所以教材以提交和符号为主要定位，行号只作辅助。

在自己的学习目录准备独立上游 checkout，并遵循该项目对应版本的开发安装文档。这里不把大型安装命令混入阅读程序，因为安装涉及你的 Python、编译工具、驱动和平台。阅读代码不要求先成功编译整个 GPU 框架；运行上游测试时才需要把依赖证据补齐。这个分层能让你先解决逻辑问题，再解决环境问题，而不是把两者混成一次漫长排错。

已有 checkout 时，从课程根目录执行：

```bash
python -m labs.serving.source_walk --project vllm --root upstream/vllm
python -m labs.serving.source_walk --project sglang --root upstream/sglang
```

工具会给出实际 HEAD、预期 HEAD、是否完全一致和若干符号位置。`exact_snapshot=false` 不是说明你下载了“错误项目”，而是说明本章断言还没有迁移到你的版本。先比较符号、参数和关键状态更新，再决定使用固定快照还是写一份版本迁移记录。不能把失效行号直接解释为作者删掉了整个机制。

这里的 `exact_snapshot` 只比较提交 ID，不检查工作区是否有未提交修改。在上游仓库目录再执行 `git status --short`；没有输出才说明 Git 没有发现普通已跟踪改动或未跟踪文件。若存在修改，先读 `git diff` 和 `git diff --cached`，把相关差异记录进实验日志。HEAD 相同但调度器文件已被本地改过时，不能声称正在验证原始固定快照；被忽略的构建产物也需要另行记录。

除了命令行 `rg`，你还可以用 Python 的抽象语法树找到函数。抽象语法树把源代码解析为语法结构，例如类定义、函数定义和调用表达式；解析不等于执行，因此不会触发被读文件里的 import 或模型初始化。下面是完整的定位器，先在一段自包含文本上验证，再用于自己的文件。

```python
import ast

def symbol_inventory(source):
    records = []
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
        def visit_ClassDef(self, node):
            records.append((".".join(self.scope + [node.name]), node.lineno, "class"))
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
        def visit_FunctionDef(self, node):
            records.append((".".join(self.scope + [node.name]), node.lineno, "function"))
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()
        visit_AsyncFunctionDef = visit_FunctionDef
    Visitor().visit(ast.parse(source))
    return records

sample_source = "class Core:\n    def step(self):\n        return self.scheduler.schedule()\n"
assert symbol_inventory(sample_source) == [
    ("Core", 1, "class"), ("Core.step", 2, "function")
]
print(symbol_inventory(sample_source))
```

`scope` 是当前嵌套路径，所以类里的 `step` 会记录为 `Core.step`。进入类时压入名字，离开时弹出，正好对应树的深度优先遍历。这个程序只定位定义，不会证明 `self.scheduler` 运行时是哪一种类。Python 可以注入替身、继承覆盖甚至动态修改方法；静态看到调用表达式，与确认真实调用目标是两项工作。

把定位器应用于文件时，使用 `Path(...).read_text(encoding="utf-8")` 读取文本，再传入函数。若 `ast.parse` 报语法错误，可能是本地 Python 不支持上游使用的新语法，也可能文件下载不完整。不要为了继续就删掉报错那行；先核对版本与文件内容，因为读错输入会让后面所有结论失去基础。

实际阅读顺序是：`tests/v1/core/utils.py` 的构造助手；`tests/v1/core/test_scheduler.py` 的部分 prefill 测试；`vllm/v1/core/sched/scheduler.py` 的调度与回写；`vllm/v1/core/sched/output.py` 的计划类型；`vllm/v1/engine/core.py` 的执行边界；最后进入 Runner。这样的顺序从具体行为逐步展开依赖，不要求先通读数千行。

## 第三课：配置不是背景信息，而是实验的一部分

固定快照的 `create_scheduler` 会构造 ModelConfig、SchedulerConfig、CacheConfig、ParallelConfig、KVCacheConfig 等对象，然后创建真实 Scheduler。它没有执行模型前向，但可能读取模型配置，导入的模块也有开发依赖。因此“CPU 测试”表达的是被测计算路径，不能被翻译为“任意机器、无安装、绝不访问模型资料”。

本次锁定三请求、每请求八百 token、总预算一千零二十四、长 prefill 阈值四百、上下文上限二千零四十八、关闭 prefix cache、关闭 speculative decoding、关闭异步调度、没有多模态和 KV connector。页面给得足够，暂时不引入容量抢占。每个关闭项都对应一条被排除的因果路径：否则命中缓存会减少待算位置，推测解码会引入拒绝回滚，connector 会带来外部加载等待。

| 配置 | 本次值 | 它让什么问题变得可控 |
|---|---:|---|
| `max_num_batched_tokens` | 1024 或 512 | 一轮最多安排多少 token 工作 |
| `long_prefill_token_threshold` | 400 | 在有竞争请求时限制长 prefill chunk |
| `long_prefill_token_threshold_adaptive` | False | 不按公平份额抬高有效阈值 |
| `enable_prefix_caching` | False | 不用已缓存前缀改变起始位置 |
| `max_model_len` | 2048 | 八百 token 提示在合法范围内 |
| `num_blocks` | helper 默认充足数量 | 暂时消除页池不足这一变量 |
| `async_scheduling` | False | 先观察一轮计划、一轮回写的顺序 |

尤其注意长 prefill 阈值的限定。固定快照的测试明确覆盖：只有一个长请求时可以忽略这个阈值，因为没有别的请求被它饿住；开启 adaptive 时，有效阈值还可能被预算的公平份额抬高。因此不能把“这次每个请求不超过四百”写成所有配置下永远成立的全局定理。测试断言总是依附于前提。

helper 把 watermark 设为零来固定准入与抢占行为，并给出很小的 FullAttentionSpec。这里的“一头、一维”是管理逻辑测试所需的合法结构，不是把真实 OPT 权重变成了微型模型。缓存块的数量和描述可以模拟，真实张量和模型 forward 不在这项测试里。你需要在笔记上把“对象真实”和“计算真实”分开标注。

另一个容易漏掉的配置是 helper 内部的 `trust_remote_code=True`。本章选用已核对的官方 OPT 配置，不意味着可以把任何陌生仓库名传进去。模型配置路径也是输入的一部分，运行前要知道其来源与内容。学习测试时可以用已准备的兼容本地配置目录，但不能假设任意模型配置都满足这些测试助手的隐含条件。

`create_requests` 的默认 ID 来自序号字符串，也允许显式传入 ID；提示 token 通常是按请求编号重复构造的列表；采样参数由 helper 建立，block hasher 也在那里初始化。测试应尽量从返回对象读 `request_id`，这样以后 helper 改命名策略时，真正关心预算的断言不会因为无关字符串变化而破坏。

现在写一条实验假设：“在上述配置与充分缓存资源下，三个八百 token 的等待请求按队列顺序在第一轮获得四百、四百、二百二十四。”这句话比“vLLM 会 chunk prefill”强得多，因为别人可以构造输入并证明它成立或不成立。如果你改变任何关键配置，先修改假设，再修改期望值。

## 第四课：把预算分配算到每一个中间值

调度开始时，总预算为一千零二十四。第一个请求尚有八百位置，但受到竞争时的四百阈值，先取四百，剩余预算六百二十四。第二个同样取四百，剩余二百二十四。第三个虽然也想取四百，但只能获得剩下的二百二十四。所有数量相加恰好一千零二十四，三个请求都部分进入运行。

预算改为五百一十二时，第一个仍取四百，第二个只拿一百一十二，第三个这轮没有条目。没有条目与条目值为零不是自动等价：真实输出循环通常要求被调度请求数量严格为正，零工作请求还可能留在等待队列。不要为了让表格整齐就向生产输出字典塞入零值。

下面写一个只负责这道算术题的参考函数。它模拟队列顺序、竞争阈值与 adaptive 公平份额，不模拟页池、抢占、LoRA、编码器、connector 或生产框架的全部准入策略。因此它用于检查手算，不能替代真实 Scheduler。

```python
def plan_counts(remaining, budget, threshold=400, adaptive=False):
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if type(threshold) is not int or threshold < 0:
        raise ValueError("threshold must be a nonnegative integer")
    if any(type(n) is not int or n < 0 for n in remaining.values()):
        raise ValueError("remaining counts must be nonnegative integers")
    active = sum(n > 0 for n in remaining.values())
    cap = threshold
    if adaptive and active > 1 and threshold > 0:
        cap = max(threshold, budget // active)
    left = budget
    result = {}
    for rid, need in remaining.items():
        if left == 0:
            break
        if need == 0:
            continue
        allowed = min(need, cap) if active > 1 and cap > 0 else need
        count = min(allowed, left)
        result[rid] = count
        left -= count
    assert sum(result.values()) <= budget
    return result

needs = {"A": 800, "B": 800, "C": 800}
assert plan_counts(needs, 1024) == {"A": 400, "B": 400, "C": 224}
assert plan_counts(needs, 512) == {"A": 400, "B": 112}
assert plan_counts({"A": 2000}, 1024) == {"A": 1024}
assert plan_counts({"A": 2000, "B": 10}, 1024, 100, True) == {"A": 512, "B": 10}
print(plan_counts(needs, 1024))
```

函数用输入字典的迭代顺序表达既定队列顺序，`active` 只计算本教学模型中有剩余工作的请求。真实框架如何定义竞争者必须去读其队列逻辑，不能把这个统计细节机械移植过去。`left` 每次只在安排成功后减少；如果加上资源申请，就要把减少放在资源成功的相应分支之后，失败处理取决于具体管理器契约。

最后一个例子很适合解释“总预算是上限”。A 得五百一十二，B 只需十，总量五百二十二，余下五百零二没有被强行使用。浪费预算不是从这个数字本身就能判定的：也许没有更多合法工作，也许阈值保护了后续请求，也许模型没有足够页面。性能分析必须先解释剩余原因，再判断策略是否值得改。

去真实 `schedule` 中寻找两个预算扣减位置，一个服务已经运行的请求，另一个服务等待队列准入。它们都做减法，但前面的条件不同。记录各自在何时获得缓存、何时改变队列、何时构造新请求数据。这样的逐分支阅读，比在文件里搜索一次 `token_budget` 后就宣布理解，提供了更多可检查的认识。

## 第五课：计划位置、在途数量与真正完成为什么不同

变量 `num_computed_tokens` 很容易让人直接翻译为“硬件已经算完多少 token”。但固定快照的 `_update_after_schedule` 在生成计划后就增加这个字段，同时增加 `num_in_flight_tokens`。此时 GPU 结果可能还没有回来。它首先承担调度账本的职责：下一轮不能再次安排刚刚已经提交的位置。

考虑 A 的初始状态为 computed=0、in_flight=0。计划记录起点零、长度四百，随后请求对象变成 computed=400、in_flight=400。执行器要处理的仍然是逻辑位置零到三百九十九，而不是四百到七百九十九。如果它错误地从已经推进的请求对象重新读取起点，就会跳过第一段输入。所以计划必须保存安排当时的必要信息，不能事后猜测。

“在途”表示已经交出而尚未完成回写的工作量。它不是独立的硬件计时器；一个设备操作已经结束但结果尚未被调度线程处理时，在途账本仍可能没有减少。更复杂路径里还会有推测 token、占位符和抢占后旧结果，不能简单把 computed 减 in_flight 当作所有模式下的最终有效长度公式。本章先关闭这些分支，再建立基础直觉。

下面建立可运行的算术账本。它只允许同时有一份待回写计划，以免把基础轨迹和异步协议混在一起。`Plan` 是冻结的数据类，内容使用元组，因此以后请求状态变化不会改变旧计划；`ArithmeticRequest` 则是持续变化的请求对象。

```python
from dataclasses import dataclass, field

@dataclass
class ArithmeticRequest:
    prompt: int
    computed: int = 0
    in_flight: int = 0
    outputs: list[int] = field(default_factory=list)
    @property
    def known(self):
        return self.prompt + len(self.outputs)

@dataclass(frozen=True)
class Plan:
    number: int
    entries: tuple[tuple[str, int, int], ...]
    @property
    def total(self):
        return sum(count for _, _, count in self.entries)

class RoundBook:
    def __init__(self, prompts):
        if any(type(n) is not int or n <= 0 for n in prompts.values()):
            raise ValueError("prompts must have positive integer lengths")
        self.requests = {rid: ArithmeticRequest(n) for rid, n in prompts.items()}
        self.pending = None
        self.next_number = 0
    def schedule(self, budget, threshold=400):
        if self.pending is not None:
            raise RuntimeError("finish the previous round first")
        needs = {rid: r.known - r.computed for rid, r in self.requests.items()}
        counts = plan_counts(needs, budget, threshold)
        entries = tuple((rid, self.requests[rid].computed, n)
                        for rid, n in counts.items())
        plan = Plan(self.next_number, entries)
        for rid, _, n in entries:
            self.requests[rid].computed += n
            self.requests[rid].in_flight += n
        self.next_number += 1
        self.pending = plan
        return plan
    def finish(self, plan, sampled):
        if self.pending is not plan:
            raise ValueError("unknown or already consumed plan")
        expected = {rid for rid, _, _ in plan.entries}
        if set(sampled) != expected:
            raise ValueError("result identities do not match the plan")
        for rid, _, n in plan.entries:
            tokens = sampled[rid]
            r = self.requests[rid]
            if not isinstance(tokens, list) or len(tokens) > 1:
                raise ValueError("this reference allows zero or one sampled token")
            if any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("sampled tokens must be nonnegative integers")
            if tokens and r.computed < r.prompt:
                raise ValueError("partial prefill cannot emit a token here")
            if r.in_flight != n:
                raise ValueError("in-flight accounting mismatch")
        for rid, _, n in plan.entries:
            r = self.requests[rid]
            r.in_flight -= n
            r.outputs.extend(sampled[rid])
        self.pending = None
    def snapshot(self):
        return {rid: (r.known, r.computed, r.in_flight, tuple(r.outputs))
                for rid, r in self.requests.items()}

book = RoundBook({"A": 800, "B": 800, "C": 800})
first = book.schedule(1024)
assert first.entries == (("A", 0, 400), ("B", 0, 400), ("C", 0, 224))
assert book.snapshot()["A"] == (800, 400, 400, ())
book.finish(first, {"A": [], "B": [], "C": []})
assert book.snapshot()["A"] == (800, 400, 0, ())
```

逐段看代码的责任。`known` 是当前已经知道的 token 序列长度，包括输入和已经采样返回的输出；`computed` 是已安排到的位置；两者的差给出下一轮还有哪些已知位置需要计算。`outputs` 用 `default_factory=list`，让每个请求拥有自己的列表。如果把一个共享列表直接用作所有请求的默认状态，一个请求生成 token 就可能污染另一个请求。

`schedule` 先形成不可变的 entries，再推进请求。这对应“保存旧位置，再改变账本”的顺序。`finish` 先检查全部结果，再进入修改循环，因此第二个请求结果非法时，不会留下第一个已经合并、第二个没合并的半状态。这里的原子性只针对这个单线程内存教学函数；它没有数据库事务，也没有跨进程崩溃恢复保证。

`self.pending is plan` 使用对象身份，要求回写当前实际交出的那份计划。这是本地程序防止重复消费的简单办法，不是分布式协议：经过序列化的等值对象不再是同一个 Python 对象，跨进程系统需要稳定批次 ID、代际或序列号。先理解这个边界，才不会把一个本地技巧当成远程幂等方案。

## 第六课：完整追踪三轮，而不是只解释第一轮

第一轮三者都没处理完提示，所以结果列表都是空。空列表表示本轮没有可发出的新 token，不表示请求失败。第二轮开始时，A/B 各剩四百提示位置，C 剩五百七十六；相同竞争条件下再次安排四百、四百、二百二十四。第二轮结束，A/B 恰好处理到提示末端，因此可以各采样一个 token；C 仍只处理了四百四十八位置。

用编号十一、二十一表示 A/B 的第一输出只是为了让身份可辨认，并不声称真实 OPT 会生成这些 token。此时 A 的已知长度变为八百零一，而已安排位置仍为八百。新 token 已经知道身份，却还没有作为下一步输入生成自己的 K/V；下一轮需要计算它的那个位置。

```python
second = book.schedule(1024)
assert second.entries == (("A", 400, 400), ("B", 400, 400), ("C", 224, 224))
book.finish(second, {"C": [], "B": [21], "A": [11]})
assert book.snapshot() == {
    "A": (801, 800, 0, (11,)),
    "B": (801, 800, 0, (21,)),
    "C": (800, 448, 0, ()),
}
third = book.schedule(1024)
assert third.entries == (("A", 800, 1), ("B", 800, 1), ("C", 448, 352))
assert third.total == 354
book.finish(third, {"A": [12], "B": [22], "C": [31]})
assert book.snapshot()["C"] == (801, 800, 0, (31,))
assert first.entries[0] == ("A", 0, 400)
print(book.snapshot())
```

第三轮的 C 只剩三百五十二位置，低于四百阈值，所以全部安排。总量是一加一加三百五十二，等于三百五十四，不是一千零二十四。这一点很重要：计划总量不必填满预算。A/B 各只有一个已经知道但未算的位置，不能提前凭空生成几百个未来 token 来填空；未来 token 取决于当前结果，除非另有显式的推测机制。

| 轮次 | A 的计划起点/数量 | B 的计划起点/数量 | C 的计划起点/数量 | 本轮模拟采样 |
|---|---|---|---|---|
| 1 | 0 / 400 | 0 / 400 | 0 / 224 | 三者都为空 |
| 2 | 400 / 400 | 400 / 400 | 224 / 224 | A→11，B→21 |
| 3 | 800 / 1 | 800 / 1 | 448 / 352 | A→12，B→22，C→31 |

固定上游测试 `test_schedule_concurrent_partial_requests` 覆盖的就是这条基本轨迹，第二轮给 A/B 非空模拟结果，第三轮断言一、一、三百五十二。本章表格额外把旧起点、在途账本和输出身份展开。你可以先遮住答案手算，再通过课程程序核对，最后去上游测试确认映射，而不是倒过来抄日志。

这里还能分清三种“长度”：提示长度、已知序列长度、已安排或已完成的 KV 位置长度。它们在某些时间点相等，但不是同一概念。A 第二轮回写后，提示长度八百、已知长度八百零一、已安排位置八百。若你把输出 token 数错误地加进已计算位置，下一轮就会认为没有工作，生成停在第一 token。

练习时故意把第二轮的 A 输出改成空列表。第三轮 A 就没有新的已知位置可算，因此不会像正常生成一样获得一个 decode token。这个实验说明 mock 结果是测试输入的一部分：它必须模拟你声称的场景。错误的 mock 可能让调度器表现“异常”，实际问题却是测试没有给出下一步所需的信息。

## 第七课：对象所有权决定一份计划会不会在背后改变

生产调度器会维护诸如 finished request IDs 的集合，并把信息交给下一层。固定源码在 `_update_after_schedule` 结尾给这些成员重新赋一个空集合，而不是直接调用原集合的 `clear()`。原因来自 Python 引用语义：如果计划保存了旧集合的引用，清空旧集合会同时清空计划看到的内容。

先用最小实验感受这个差别。无需线程，一个共享对象就足以制造错误。

```python
finished = {"A"}
plan_view = {"finished": finished}
finished.clear()
assert plan_view["finished"] == set()

finished = {"A"}
plan_view = {"finished": finished}
finished = set()
assert plan_view["finished"] == {"A"}
assert finished == set()
```

第一段中，两个名字指向同一个集合，`clear()` 修改集合本身。第二段中，赋值只让调度器的名字指向一个新集合，旧集合仍被计划引用。若还不熟悉这个区别，请画两个对象框和指向它们的名字箭头：框代表可变对象，箭头代表引用。赋值通常改箭头，集合方法通常改框里的内容。

“那每次 deepcopy 不就好了”也是不完整的答案。深复制可能遍历很大结构，复制张量或不该复制的资源句柄；某些对象本来就应该共享；而且复制操作本身不自动建立 GPU 同步关系。设计首先要回答谁拥有数据、谁会修改、谁负责生命周期，然后选择不可变值、浅拷贝、深拷贝或转移所有权。

浅拷贝只复制外层容器。下面的批次元数据例子说明外层列表不再相同，内部请求对象仍然共享。不要据此推断真实 SGLang 的 `batch.copy()` 等同于 Python 的这一行；真实方法复制哪些字段需要阅读其实现。本例只是拆开“复制”这个含糊词。

```python
from copy import copy
request = {"id": "A", "tokens": [1, 2]}
batch = [request]
outer_copy = copy(batch)
assert outer_copy is not batch
assert outer_copy[0] is batch[0]
request["tokens"].append(3)
assert outer_copy[0]["tokens"] == [1, 2, 3]
immutable_snapshot = tuple((r["id"], tuple(r["tokens"])) for r in batch)
request["tokens"].append(4)
assert immutable_snapshot == (("A", (1, 2, 3)),)
```

不可变快照使日志和测试更容易复查，但它只记录被选中的字段。没有写进快照的引用计数、缓存身份或取消代际仍可能变化。因此“有快照”不是泛泛的可靠性保证，要列清快照覆盖了哪些不变量。对本章三轮账本而言，ID、起点和数量足够解释预算；对真实页复用而言，还必须纳入页表和资源生命周期。

源码阅读时给每个跨函数对象加一个标记：只读值、借用可变对象、独占对象、共享资源。若某函数返回一个内部列表，调用者是否可以改它？若输出类型只是 dataclass，它的字段是否仍可变？这些问题比记住 dataclass 的语法更接近实际框架开发，因为很多难查错误来自跨模块默契没有写成合同。

## 第八课：沿 EngineCore 穿过执行和采样的边界

固定版本的 `EngineCore.step` 先检查是否还有请求，包括已经结束但仍需要从 worker 批次移除的情况；再调用 scheduler；把计划交给 `model_executor.execute_model(..., non_block=True)`；准备 grammar 数据；等待 future；如果返回值为 None，则继续调用 `sample_tokens`。最后先处理执行期间到来的 abort，再把模型结果交给调度器回写。

这里的 future 是“以后能取得结果的对象”。创建 future 不表示工作已经结束，调用 `.result()` 可能等待。`non_block=True` 描述接口提交方式，不能单凭这一参数推断所有下层操作都绝不阻塞。实际 overlap 能否发生，还依赖执行器、通信、设备 stream 和调用顺序。

为什么模型执行可能返回 None？固定 GPUModelRunner 的某条路径会先保存 ExecuteModelState，包含 logits 等临时结果，再由 `sample_tokens` 消费它。这把模型前向与采样阶段分开，允许结构化输出等数据在合适时机参与。不要把 None 翻译成“模型失败”：它在这里是一个有后续动作的协议分支。

用一个完整的两阶段 Runner 练习这个顺序。它只在 Python 列表上工作，不模拟真实网络、GPU 或推测解码；它刻意要求每次 execute 后必须 sample，便于观察状态约束。

```python
class SplitRunner:
    def __init__(self):
        self.pending_scores = None
    def execute(self, scores):
        if self.pending_scores is not None:
            raise RuntimeError("sample must consume previous execution")
        rows = [tuple(row) for row in scores]
        if any(not row for row in rows):
            raise ValueError("score rows must not be empty")
        self.pending_scores = rows
        return None
    def sample(self, allowed=None):
        if self.pending_scores is None:
            raise RuntimeError("no pending execution")
        rows = self.pending_scores
        if allowed is None:
            allowed = [set(range(len(row))) for row in rows]
        if len(allowed) != len(rows):
            raise ValueError("mask row count mismatch")
        for row, choices in zip(rows, allowed):
            if not choices or any(type(i) is not int or not 0 <= i < len(row)
                                  for i in choices):
                raise ValueError("invalid allowed token set")
        output = [max(sorted(choices), key=lambda i: row[i])
                  for row, choices in zip(rows, allowed)]
        self.pending_scores = None
        return output

runner = SplitRunner()
assert runner.execute([[1.0, 3.0, 2.0], [5.0, 2.0, 1.0]]) is None
try:
    runner.execute([[0.0, 1.0]])
except RuntimeError as exc:
    assert "consume" in str(exc)
else:
    raise AssertionError("missing phase check")
assert runner.sample([{0, 2}, {1, 2}]) == [2, 1]
assert runner.pending_scores is None
```

第一行分数最大项本来是 token 一，但允许集合只有零和二，因此结果是二。这个例子把“算出 logits”和“根据允许 token 集合采样”拆开，同时展示第二次 execute 为什么要拒绝：如果直接覆盖临时状态，第一份结果会悄悄丢失。`sorted` 给分数相同的情况一个确定的较小 ID 优先规则，这是教学程序显式选择的行为。

对照真实 GPUModelRunner：入口检测未消费的 `execute_model_state`；某条正常路径保存状态并返回 None；`sample_tokens` 解包并清理临时状态。真实 `sample_tokens` 在状态为 None 时还有 connector、流水线等特殊处理，不等同于本例一律报错。读源码不能只找相似部分，还必须写出不相同的分支，才能判断教学模型在哪些地方需要扩展。

本课交付一条带条件的调用链：普通 step 的 schedule → execute 提交 → future.result → 必要时 sample → abort 处理 → update。再标注哪些数据来自计划、哪些来自模型、哪些可能在执行期间变化。调用图只有箭头没有这些条件时，很容易掩盖取消和异步场景，后面我们就用取消把它补完整。

## 第九课：把结果路由、计数守恒和一次消费写成合同

`SchedulerOutput` 同时携带按请求分开的数量和总数量。既然总数量可以求和得到，为什么还保存一份？调用方可能频繁需要总量来判断是否有工作、分配缓冲区或选择执行路径，冗余字段能减少重复处理；代价是两份信息必须一致。读到冗余状态时，首先想到的测试应当是守恒关系，而不是只检查某个字段非零。

本章的基础合同包括：所有被安排的数量大于零；总量等于各请求数量之和；总量不超过本轮预算；每个请求起点和数量覆盖合法已知位置；回写结果身份与计划匹配；同一计划不能成功回写两次。这些断言中有的是通用结构，有的是教学配置特有。比如“所有数量不超过四百”只在前面锁定的竞争与阈值条件下成立。

注意，`scheduled_cached_reqs` 的 cached 并不直接等于跨请求 prefix cache 命中。已有请求再次送往 worker，可能使用增量更新结构，因为 worker 已有它的基本信息。本章关闭 prefix cache，第二轮仍然会出现 cached request 数据。读类型应问“这个结构让接收方更新什么”，不能仅凭一个英语单词替它认定用途。

给算术账本增加一个独立的计划验证函数。独立验证的意义是从外部可观察合同出发，而不是把实现里同一段判断复制到测试里；如果实现和测试共享了同一个错误 helper，两者可能一起给出错误答案。

```python
def verify_plan(plan, budget, known_before):
    ids = [rid for rid, _, _ in plan.entries]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate request in plan")
    for rid, start, count in plan.entries:
        if rid not in known_before:
            raise AssertionError("unknown request")
        if type(start) is not int or type(count) is not int:
            raise AssertionError("non-integer range")
        if start < 0 or count <= 0 or start + count > known_before[rid]:
            raise AssertionError("invalid request range")
    if not 0 <= plan.total <= budget:
        raise AssertionError("budget exceeded")
    return True

assert verify_plan(first, 1024, {"A": 800, "B": 800, "C": 800})
assert verify_plan(third, 1024, {"A": 801, "B": 801, "C": 800})
try:
    verify_plan(Plan(9, (("A", 790, 20),)), 1024, {"A": 800})
except AssertionError as exc:
    assert "range" in str(exc)
else:
    raise AssertionError("out-of-range plan passed")
```

这里的 `known_before` 要在调度前或依据明确的旧状态取得，而不是读取已经追加新输出的未来状态。测试拿错时间点也会放过越界：一个本轮不该读的位置，在下一轮可能已经合法。状态机测试不仅要检查值，还要检查值属于哪个时间切片。

再验证重复回写确实被拒绝，而且没有追加第二遍输出。保存快照、触发非法操作、比较快照，是一类很有用的错误路径测试。

```python
before = book.snapshot()
try:
    book.finish(third, {"A": [12], "B": [22], "C": [31]})
except ValueError as exc:
    assert "consumed" in str(exc)
else:
    raise AssertionError("duplicate completion accepted")
assert book.snapshot() == before
```

将这些检查迁移到真实框架时，要先识别合法重复与非法重复的边界。有些外部 API 的取消本来就应幂等，有些内部完成通知则必须唯一，有些流式消息允许重发但要求序列号去重。不能把“所有重复都报错”或“所有重复都忽略”当作统一答案；各接口要明确调用方可能怎样重试以及错误如何被观察。

## 第十课：取消与迟到结果，要同时处理逻辑状态和资源寿命

假设 GPU 已经开始读取 A 的 KV 页面，用户随后取消 A。服务可以立即停止向用户发正常输出，但不能仅因为逻辑状态变为取消就让另一个请求覆盖仍被设备读取的页面。停止输出和回收资源是两件事，可能发生在不同时间。理解这一点，是从同步教学引擎进入真实异步框架的关键。

固定源码的 `update_from_output` 先按已安排数量减少仍存在请求的 in-flight 账本，并处理 stale 份额，再检查 KV 加载失败、请求缺失或已经结束等情况。缺失可能表示请求已被移除；已结束但对象仍保留，可能是异步传输等流程还需要它。不能把“字典里还找得到”当作允许继续向用户生成的充分条件。

抢占又不完全等同于取消。抢占可能把请求放回等待队列并重算，旧轮次结果如何处理会影响 speculative acceptance 等逻辑。固定快照有 stale 计数和 drop 模式：某些旧结果仍参与特定输出流程，但不得错误修改重置后的计数；另一些路径直接丢弃。教学模型如果一律丢弃旧结果，必须明确这是自己的简化，不能声称复制了上游全部规则。

下面写一个更小的取消资源模型，只解决“取消后不发布迟到结果，所有旧工作结束后才可释放”这一个合同。它不支持抢占恢复、不模拟 CUDA event，也不决定生产 allocator 何时实际复用页面。

```python
class CancelFence:
    def __init__(self):
        self.cancelled = False
        self.next_ticket = 0
        self.outstanding = set()
        self.visible = []
    def launch(self):
        if self.cancelled:
            raise RuntimeError("request cancelled")
        ticket = self.next_ticket
        self.next_ticket += 1
        self.outstanding.add(ticket)
        return ticket
    def cancel(self):
        changed = not self.cancelled
        self.cancelled = True
        return changed
    def complete(self, ticket, token):
        if ticket not in self.outstanding:
            raise ValueError("unknown or duplicate completion")
        self.outstanding.remove(ticket)
        if not self.cancelled:
            self.visible.append(token)
    @property
    def releasable(self):
        return self.cancelled and not self.outstanding

fence = CancelFence()
t0, t1 = fence.launch(), fence.launch()
assert fence.cancel() is True
assert fence.cancel() is False
assert not fence.releasable
fence.complete(t1, 99)
assert not fence.releasable and fence.visible == []
fence.complete(t0, 98)
assert fence.releasable and fence.visible == []
try:
    fence.complete(t0, 98)
except ValueError:
    pass
else:
    raise AssertionError("duplicate result accepted")
```

两个 ticket 的完成顺序故意反过来，说明可释放条件依赖所有旧工作被排空，而不是依赖某一个编号恰好回来。`cancel()` 第一次返回 True，第二次返回 False，表达是否发生新状态变化；完成通知重复则报错，因为这个内部教学接口不允许重复消费。二者的策略不同是有意设计，并不矛盾。

为什么不能设置一个固定超时，时间到就假定设备不再访问？超时是主机观测不到进展的事实，不是设备完成的证明。真实系统可能需要 event、stream 顺序、执行代际、延迟释放或进程级终止等机制，具体保证要看实际调用链。教材中的计数集合只建立逻辑条件，不能给 GPU 资源的安全释放背书。

做源码笔记时把取消拆成四个问题：何时接受取消；何时停止对外输出；谁仍拥有在途引用；哪个事件允许最终回收。再去找各自的实现位置。如果只写“finish_requests 释放请求”，没有回答剩下三个问题，遇到并发结果就仍会不知所措。S18 的页状态工作坊会继续把引用和重算展开，S19 则验证实际 K/V 数值没有被错误覆盖。

## 第十一课：把课程测试放进真实上游，而不是测试自己的替身

课程提供完整文件 [`test_course_scheduler_contract.py`](../../labs/serving/upstream_tests/test_course_scheduler_contract.py)。它导入真实 Scheduler、ModelRunnerOutput、RequestStatus 和上游构造助手，测试预算边界与运行请求取消幂等。前面的 RoundBook 是解释机制的辅助程序；这一文件才是进入真实框架测试体系的桥梁，二者的证据不能互换。

先只预览复制目标，再写入自己的学习 checkout：

```bash
python -m labs.serving.install_upstream_test --root upstream/vllm
python -m labs.serving.install_upstream_test --root upstream/vllm --apply
```

安装脚本核对固定 SHA，拒绝覆盖已有同名文件，只增加这一个课程测试文件。它不安装依赖、不下载权重、不运行测试，也不改生产实现。第二条命令只适合你拥有的独立学习 checkout；如果第一条显示目标路径不是预期位置，先修路径，不要为了继续而删掉版本检查。

从上游根目录执行测试时，启用离线模型资料模式：

```bash
cd upstream/vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -q
```

这些环境变量要求所需资料已在本地。缺少 OPT 配置或开发依赖时，应该如实记录环境未就绪；先在允许联网的准备阶段按官方说明准备配置，或使用已经核验兼容的本地模型配置目录设置 `COURSE_VLLM_TEST_MODEL`。测试没有执行权重前向，但导入完整框架仍可能要求正确的 PyTorch/vLLM 安装，不能因为测试被标记为 CPU 就跳过环境核对。

打开测试文件，先读 `empty_result`。它从计划字典取请求 ID，再建立 ID 到行号的映射，为每个请求提供一个空输出列表。返回 `[[], [], []]` 与返回 `[]` 在结构上不同：前者保留三个请求槽位，后者没有槽位。实际 ModelRunnerOutput 还含 logprobs、pooler 等字段；本场景关闭这些功能，因此填合法的空值，而不是省略必需结构。

预算测试参数化两组输入：预算五百一十二期望四百和一百一十二；预算一千零二十四期望四百、四百、二百二十四。它检查列表具体值、数量总和、冗余 total 字段和每请求上限。检查具体值验证这项配置的队列策略；检查守恒验证输出结构；两者都需要，因为总量正确不代表分配正确，分配字典正确也不代表 total 字段正确。

调用 `update_from_output` 后清理全部请求，确认未完成计数归零。清理不只是礼貌，它让测试本身遵守生命周期，避免共享 fixture 或缓存资源改变后续测试结果。一个测试单独通过、和别的测试一起失败，常常来自未清理状态、全局配置泄漏或随机种子依赖，不能立即归咎于被测算法。

取消测试先记录空闲页数，加入一百二十八 token 提示，用六十四预算安排部分工作，确认页面减少。回写空结果后，第一次 abort 返回一个有效请求，第二次返回空；空闲页回到原值，请求字典与下一轮计划都没有该 ID。这里特意关闭 prefix cache，因为开启缓存时，合法的缓存拥有引用可能让“空闲页恢复原值”不再是正确断言。

本次仓库环境没有执行上述上游 pytest，状态保持为“接口按固定源码核对，执行待上游环境验证”。你在自己的服务器 CPU 环境完成后，应记录 Python、框架提交、依赖、模型配置路径身份、命令与失败或通过位置。不要把课程自身离线测试的绿色结果填到上游测试这一栏。

## 第十二课：用负向控制证明测试真的能抓住关注的错误

测试通过只说明当前输入没有触发失败。若要确认断言对某类缺陷敏感，可以主动制造该缺陷，要求测试在正确位置失败。这里的负向控制使用 pytest monkeypatch，只在测试进程包装 `Scheduler.schedule`，把 total 字段加一，其他内容保持不变，不修改磁盘上的生产源码。

```bash
COURSE_MUTATE_SCHEDULER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -k partial_prefill -q
```

预期失败在 total 等于各请求数量之和的断言。如果失败发生在 import、配置读取或设备初始化，还没有到达被测合同，不能记作“成功检测调度错误”。记录失败堆栈时至少保留失败测试名、断言两侧值与触发输入，让别人能够判断到底是哪一层失败。

先在标准库小模型里完整演练这一方法，不依赖上游安装。下面用独立的输出字典表达冗余字段，正向和负向都执行。

```python
def check_output_contract(output, budget):
    counts = output["counts"]
    if any(type(n) is not int or n <= 0 for n in counts.values()):
        raise AssertionError("nonpositive work")
    if output["total"] != sum(counts.values()):
        raise AssertionError("redundant total mismatch")
    if output["total"] > budget:
        raise AssertionError("budget exceeded")
    return True

good = {"counts": {"A": 400, "B": 112}, "total": 512}
assert check_output_contract(good, 512)
bad = {"counts": good["counts"].copy(), "total": 513}
try:
    check_output_contract(bad, 512)
except AssertionError as exc:
    assert str(exc) == "redundant total mismatch"
else:
    raise AssertionError("negative control was not detected")
assert good["total"] == 512
```

为什么先检查守恒再检查预算？这不是唯一合法顺序，但对当前变异，我们希望明确观察到冗余字段不一致，而不是泛化成超预算。测试错误信息也是诊断接口的一部分。好的信息能缩小范围，含糊的“结果不对”则会让后来维护者重新理解整个案例。

第二类负向控制是身份错配。第一课的错误 `zip` 在顺序相同的输入上通过，在 C/A/B 顺序上失败。第三类是重复完成：移除 pending 检查会让同一输出追加两次。第四类是资源提前回收：把 `releasable` 改为只看 cancelled，就会在两个 ticket 未结束时返回 True。每个变异都应有明确的预期失败断言，不能只追求“随便哪里红了”。

变异测试也有范围。你不需要为每行代码制造一个错误版本，更不能把变异永久混入正常运行入口。选择与当前修复或学习目标直接相关的合同，保存正向通过、负向预期失败、关闭变异恢复通过的三段记录即可。对于真实上游，课程环境变量默认关闭，只有显式开启才包装方法，这减少了后续忘记恢复造成的混淆。

## 第十三课：三个完整的故障工作台

第一个故障是部分 prefill 提前发出 token。触发输入为一百 token 提示、首轮预算四十、模拟结果却包含一个新 token。如果回写接受它，已知长度会错误增加，而生成模型实际上还没有看到完整提示。外部症状可能是后续起点不一致，但根因位于 mock 或回写合同，而不是预算减法。

用新对象隔离实验，保存失败前的状态，确认拒绝后仍可用正确结果完成同一轮。

```python
broken_case = RoundBook({"A": 100})
partial = broken_case.schedule(40)
saved = broken_case.snapshot()
try:
    broken_case.finish(partial, {"A": [7]})
except ValueError as exc:
    assert "partial prefill" in str(exc)
else:
    raise AssertionError("premature sample accepted")
assert broken_case.snapshot() == saved
broken_case.finish(partial, {"A": []})
assert broken_case.snapshot()["A"] == (100, 40, 0, ())
```

排错顺序是先确认本轮安排是否达到提示末端，再确认结果结构是否声称产生了输出，最后检查回写为何允许它。修复不是盲目把所有空结果改成非空，或者把预算改大让错误消失；修复应恢复“只有满足阶段条件才输出”的合同。真实框架可能有不同 runner 类型，迁移这项断言时还要限定为普通自回归文本生成。

第二个故障是结果漏掉一个请求。两个请求已经占用计划位置，但结果只包含 A，若程序直接遍历结果字典，B 的 in-flight 永远不会排空。随着更多轮次发生，资源看似泄漏，实际是身份集合不完整。不要把这种问题优先归类为显存碎片；先检查计划与结果的身份集合。

```python
missing_case = RoundBook({"A": 10, "B": 10})
pair_plan = missing_case.schedule(20)
saved = missing_case.snapshot()
try:
    missing_case.finish(pair_plan, {"A": [1]})
except ValueError as exc:
    assert "identities" in str(exc)
else:
    raise AssertionError("missing result row accepted")
assert missing_case.snapshot() == saved
missing_case.finish(pair_plan, {"B": [2], "A": [1]})
assert all(r.in_flight == 0 for r in missing_case.requests.values())
```

正确修复保留按 ID 路由，并在消费前检查完整集合。结果顺序改变应通过，缺失、额外或重复身份应被明确处理。这里用字典无法表达重复键，因为后写值会覆盖前值；若外部协议以列表传入，必须在转换成字典前检测重复。否则你检查的是已经丢失信息的数据，无法知道曾经发生过冲突。

第三个故障是跨轮引用被修改。第七课已经展示 `clear()` 会清空旧计划集合。现在把它改成一个诊断函数：要求每轮结束事件被消费一次，第二轮清理不影响第一轮快照。修复可以使用不可变快照，也可以转移旧集合并创建新集合，关键是明确所有权。

```python
def publish_finished(current):
    published = frozenset(current)
    current.clear()
    return published

current_finished = {"A", "B"}
first_finished = publish_finished(current_finished)
current_finished.add("C")
second_finished = publish_finished(current_finished)
assert first_finished == frozenset({"A", "B"})
assert second_finished == frozenset({"C"})
assert not current_finished
```

这段实现复制了一份不可变 ID 集合，适合小型教学元数据，不表示所有生产路径都应该这样改。若集合很大，或接收方需要其他共享对象，代价与接口都会不同。你应当把回归测试固定在“旧计划不被后续清理改变”，而不是固定在“必须使用 frozenset”，这样实现仍然有合理改进空间。

三个工作台都遵循同一排错过程，但错误机制不同：第一种是阶段不合法，第二种是身份不完整，第三种是所有权不清。只有把症状追到机制，才知道应该在哪个层级加断言。单纯在外层捕获全部异常并继续，可能让表面服务更安静，却把状态损坏藏得更深。

## 第十四课：从 vLLM 迁移到 SGLang，要迁移不变量而不是变量名

固定 SGLang 的 overlap 循环先接收请求并选择下一批，再根据当前批次情况决定是否提前处理上一批结果；启动当前批次后建立 WAR 相关屏障，把批次副本和结果成对加入 `result_queue`；随后处理上一批，必要时启动当前批采样。这里的 WAR 指写后续状态之前，需要考虑先前读取仍未结束的依赖，不能仅按 Python 语句已经返回来判断数据可以修改。

请在源码里确认具体顺序：`run_batch`，`_apply_war_barrier`，队列追加 `(batch.copy(), batch_result)`，以及 `pop_and_process` 最终调用 `process_batch_result`。当前批和上一批同时出现在循环里，变量名相近，很容易把“现在启动的工作”和“现在处理的结果”误配。这也是为什么结果队列要把批次与结果放在同一个元素中。

用标准库模拟这个配对问题。我们不模拟 GPU 同步，只证明队列记录必须保留提交时的身份。当前批次列表后来改变，已经提交的记录仍然要属于旧请求。

```python
from collections import deque
queue = deque()
current_batch = ["A", "B"]
queue.append((tuple(current_batch), {"A": 11, "B": 21}))
current_batch[:] = ["C"]
queue.append((tuple(current_batch), {"C": 31}))
processed = []
while queue:
    batch_ids, result = queue.popleft()
    assert set(batch_ids) == set(result)
    processed.append(tuple((rid, result[rid]) for rid in batch_ids))
assert processed == [(("A", 11), ("B", 21)), (("C", 31),)]
```

如果队列保存的是可变 `current_batch` 本身，第二轮切片赋值会改变第一条记录的身份，结果中 A/B 和批次中的 C 不再对应。真实 SGLang 的 ScheduleBatch 比这个列表复杂得多，`copy()` 的字段策略必须继续向定义处追踪；这里只说明为什么“提交批次身份稳定”是值得检查的不变量。

再把依赖顺序写成事件约束，而不是凭文字猜测。一项设备读取可以在主机提交后继续进行，某个后续状态写入必须在读取完成或有效屏障之后发生。下面的函数只是验证给定轨迹，不能执行或证明真实 CUDA 屏障。

```python
def verify_order(events, dependencies):
    if len(events) != len(set(events)):
        raise ValueError("event labels must be unique in this exercise")
    positions = {event: i for i, event in enumerate(events)}
    for before, after in dependencies:
        if before not in positions or after not in positions:
            raise AssertionError("missing event")
        if positions[before] >= positions[after]:
            raise AssertionError(f"dependency violated: {before} -> {after}")
    return True

dependencies = [("read_launch", "read_done"), ("read_done", "shared_write")]
assert verify_order(["read_launch", "cpu_prepare", "read_done", "shared_write"], dependencies)
try:
    verify_order(["read_launch", "shared_write", "read_done"], dependencies)
except AssertionError as exc:
    assert "dependency" in str(exc)
else:
    raise AssertionError("hazard was not detected")
```

阅读真实屏障实现时，继续问它约束的是哪个 stream、哪个共享缓冲区、哪个阶段的读写，以及屏障记录和等待发生在哪一侧。一个名叫 barrier 的函数不自动解决所有对象的竞争；一个数据副本也不自动解决原缓冲区还被设备读取的问题。身份稳定和内存寿命是两个相互关联但不同的合同。

vLLM 的计划账本与 SGLang 的批次对象组织方式不同，但你可以用相同的问题比较它们：谁拥有请求真相；谁保存提交时位置；如何识别迟到结果；何时允许复用缓存；取消如何影响对外输出；测试在哪一层构造这些情况。把答案写成对照表，会比把一个项目的变量名翻译成另一个项目的变量名更有迁移价值。

## 第十五课：把阅读结果变成一个小而完整的补丁

第一次上游练习建议从测试增补开始。选择一个具体触发条件，例如预算刚好跨过分块边界、取消重复调用、结果顺序变化或输出总计字段不一致。先写失败案例，再解释期望合同，然后决定是否需要改实现。不要因为读到一个函数很长，就把“整体重构调度器”作为第一个任务；那会同时改变太多尚未理解的行为。

创建自己的学习分支后，先看 `git status` 和 diff，确认改动只涉及预期文件。版本固定、依赖记录与测试输入构成可复现起点。修改前如果测试因环境问题失败，先解决环境；否则后面很难分辨补丁是否引入了新的错误。没有真实运行条件时，可以提交本地学习记录，但描述必须停留在接口审查或语法检查，不能写“所有上游测试通过”。

补丁说明应先写行为：在什么输入和状态下，原来的检查遗漏了什么；新增断言如何观察问题；变异版本为什么失败；正确版本为什么通过。接着写边界：同步或异步，prefix cache 开或关，是否涉及真实 forward，哪些组合没有验证。这样的说明让审阅者能够判断维护价值，而不需要读你的学习过程日记。

若要加入运行时断言，先决定开销和适用环境。遍历上千请求求和可能在测试里很合理，在热路径每轮执行则需要成本证据。也不要直接用普通 `assert` 承担必须存在的外部输入验证，因为优化模式可能移除断言。测试里的断言用来表达预期；生产验证是否应抛显式异常，要看失败后果和接口约定。

性能修改需要另外的实验设计。纯测试补丁可以明确不改变运行时行为；改变队列顺序、缓存申请或同步位置则可能影响延迟与吞吐。先完成 CPU 合同与短时单卡正确性，再决定是否值得申请更大窗口。你的八卡 H100 适合验证需要多设备的机制，不应成为每次改一个断言就启动的默认步骤。

迁移到新版本时，不要盲目把旧测试文件复制过去。比较构造助手签名、输出类型字段、计划账本推进时机、取消返回值和缓存拥有语义。如果变了，先写出旧合同与新合同的差别，再更新测试。测试迁移本身也是源码阅读训练，它要求你解释变化，而不是只把报错参数删到程序能跑。

## 第十六课：分级练习与完整核对答案

以下题目先独立完成，再看答案。算术题必须写中间值，代码题必须运行边界输入，源码题必须记录固定提交和具体符号。若答案只是一个术语，例如“加锁”“幂等”“异步”，还没有完成本章要求。

### 题一：预算六百、三个八百提示，第一轮怎样分配

答案：在竞争阈值四百且 adaptive 关闭时，A 先拿四百，剩二百；B 拿二百，剩零；C 没有计划条目。结果字典是 A→400、B→200，总量六百。把预算平均除以三得到每人二百，是另一种策略，不能拿来替换固定队列路径的预期。可以直接运行 `plan_counts(needs, 600)` 核对。

### 题二：只剩一个两千 token 长请求，阈值四百、预算一千零二十四

答案：固定快照相关测试允许唯一长请求忽略竞争阈值，因此本次可安排一千零二十四。前提仍包括足够缓存、合法上下文与其他限制不生效。此题用于打破“阈值永远是硬上限”的误解；真实系统还有其他准入条件，不能从这个例子推出单请求一定用满所有预算。

### 题三：为什么第二轮结束后 A 的 known 是八百零一，computed 是八百

答案：前八百提示位置已经安排并完成本轮回写，模型利用最后位置的状态采样出一个新 token。这个 token 的身份已经加入已知序列，但它尚未作为下一次输入计算自己的 K/V，所以 known 多一、computed 不变。下一轮处理这个新位置，再采样下一个输出；混淆两者会造成跳过 decode 或重复计算。

### 题四：把第一轮安排字典反向迭代，结果回写应该变吗

答案：若改变的是调度队列本身，分配可能变；若只是模型结果数组顺序改变，而 ID 到行号映射同步正确，归属不应该变。要先区分改变发生在哪个接口。第一课的 C/A/B 结果排列仍应把十一交给 A、二十一交给 B、三十一交给 C。下面给出一个可独立复用的列表协议解码器，在转成字典前拒绝重复身份。

```python
def decode_rows(ids, rows):
    if len(ids) != len(rows):
        raise ValueError("row count mismatch")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate identity")
    return {rid: list(row) for rid, row in zip(ids, rows)}

assert decode_rows(["C", "A"], [[31], [11]]) == {"C": [31], "A": [11]}
try:
    decode_rows(["A", "A"], [[11], [12]])
except ValueError as exc:
    assert "duplicate" in str(exc)
else:
    raise AssertionError("duplicate identities lost during conversion")
```

### 题五：设计一个能抓住“total 多一”但不依赖请求名称的测试

答案：从输出的数量字典取 values 求和，与 total 比较，不写死 A/B/C；另断言总和不超过输入预算。正确输出通过，把 total 单独加一后必须在守恒断言失败。第十二课的 `check_output_contract` 是完整答案。若只断言 total 小于等于预算，在原本没有用满预算的轮次里，多一仍可能通过，因此不能替代守恒检查。

### 题六：取消后收到两份迟到结果，可否收到第一份就释放

答案：不能，只要第二份仍可能访问资源，就必须保留所需寿命。第十课的 CancelFence 用 outstanding 集合表达所有在途工作；两个 ticket 都结束后才满足 releasable。对真实设备，还需要证明完成通知与相关读取结束之间的关系，集合归零只是逻辑模型。重复 cancel 可以是幂等，重复内部 completion 则按本例合同拒绝。

### 题七：测试在导入 vLLM 时失败，是否证明负向控制有效

答案：没有。预期变异在调度完成后改变 total，必须运行到对应断言才能说明测试敏感。导入失败只证明执行环境不足。报告应分别记录环境状态、正常版本结果和变异版本结果；若前者未完成，后两项写未验证，不把一个红色退出码当作目标错误被发现的证据。

### 题八：写出预算从零到一千零二十四的性质检查

答案不仅检查两个手选预算，还遍历整个小范围，要求总量不超过预算、每个条目正数、同样输入结果稳定。固定三个长请求且阈值四百时，每个条目不超过四百也成立。下面是完整程序，使用确定范围而不是依赖随机碰巧抽到边界。

```python
for budget in range(1025):
    counts = plan_counts({"A": 800, "B": 800, "C": 800}, budget)
    assert sum(counts.values()) <= budget
    assert all(0 < n <= 400 for n in counts.values())
    assert counts == plan_counts({"A": 800, "B": 800, "C": 800}, budget)
    assert sum(counts.values()) == budget
assert plan_counts({"A": 5, "B": 7}, 100) == {"A": 5, "B": 7}
```

最后一个例子专门反驳“所有输入都必须用满预算”。前三个请求的总可安排上限大于遍历范围，因此那里总和等于预算；两个短请求总需求只有十二，预算一百也只能安排十二。性质测试的前提必须随输入域说明，否则一个正确算法可能被错误性质判成失败。

### 题九：一个无 GPU 的 Scheduler 测试为什么仍可能缺模型资料

答案：构造 ModelConfig 需要模型结构配置，采样参数、tokenizer 或其他导入链也可能访问相应资料。没有执行权重前向，只排除了其中一种依赖。准备时记录配置来源与本地缓存，离线运行时缺资料就明确失败；不要偷偷取消离线限制后把下载成功误认成代码修复，也不要把未知模型配置当成等价替代。

### 题十：为什么把 finished 集合 clear 掉可能破坏旧计划

答案：计划和调度器成员若引用同一个可变集合，clear 修改的是对象，因此两边同时看见空集合。解决可以是把旧集合交给计划后让调度器成员指向新集合，也可以发布不可变副本。第七课与第十三课有两种完整实现。测试应要求发布内容不被后续清理改变，而不是强迫所有实现采用某一种复制函数。

### 题十一：写一个不会因第二行非法而半完成回写的检查

答案：先遍历全部结果做身份、类型、阶段和在途校验，所有检查通过后再遍历一次修改状态。第五课 RoundBook.finish 就采用这个结构。验证时让 A 合法、B 非法，保存操作前快照，失败后要求整个快照相等，再传正确结果应能完成。下面实际执行这个场景。

```python
atomic_case = RoundBook({"A": 4, "B": 4})
atomic_plan = atomic_case.schedule(8)
saved = atomic_case.snapshot()
try:
    atomic_case.finish(atomic_plan, {"A": [1], "B": [-1]})
except ValueError:
    pass
else:
    raise AssertionError("invalid token accepted")
assert atomic_case.snapshot() == saved
atomic_case.finish(atomic_plan, {"A": [1], "B": [2]})
assert atomic_case.snapshot()["A"] == (5, 4, 0, (1,))
```

这证明了本地函数对这些可预期验证错误的行为，没有证明进程在修改循环中被杀死时仍可恢复。跨进程持久事务需要 A10 的日志与事务机制。把保证限定清楚，能避免因为一个测试名叫 atomic 就误以为获得了所有层级的原子性。

### 题十二：为上游新增测试写一段可评审说明

完整参考说明可以是：“当三个八百 token 提示在关闭前缀缓存和自适应阈值的配置下竞争一轮预算时，新增测试覆盖五百一十二与一千零二十四两个边界，并要求输出总量等于逐请求数量之和。进程内变异将 total 增一，应触发守恒断言。测试不执行模型前向；上游依赖与配置已准备时才记录实际结果。取消案例另外验证重复 abort 不重复释放活动页面。”实际提交时把未执行状态替换成你亲自获得的证据，不能照抄成已经通过。

这段说明包含触发条件、行为、检测方法和证据范围，没有夸大成“全面验证调度器”。若还修改了生产实现，继续补充具体前后行为与受影响分支。一个范围小但能复现的补丁，比一个没有失败案例的大范围重构更适合建立代码级理解。

## 结课交付：把所有证据放回同一条请求

交付五项材料：三轮状态表；带固定版本和条件的调用路径；课程上游测试文件及你实际做过的改动；正常与变异测试记录；一份小补丁说明。状态表应包含等待/运行身份、计划起点与数量、computed/in-flight、输出列表以及资源相关观测。对于本章没有真实执行的 GPU、上游 pytest 或 API，记录未执行而不是留一个容易误解的空格。

最后进行口头自检：不用打开答案，解释为什么第一轮是四百、四百、二百二十四；为什么第三轮只有三百五十四工作；为什么 scheduled cached request 不意味着 prefix hit；为什么取消与资源回收不是同一时刻；为什么 batch copy 不自动提供设备同步。解释时如果只能说术语，就回到对应工作台重新运行、修改一个输入并预测结果。

通过这些检查后，你已经完成从“知道框架有调度器”到“能用具体输入定位、推演、验证一段真实调度行为”的训练。接下来进入 [S18 页表与抢占的逐状态推演](18-state-workbook.md)，把本章的请求账本与实际页面拥有关系连接起来，再到 S19 验证页面里真正存放的 K/V 数值。
