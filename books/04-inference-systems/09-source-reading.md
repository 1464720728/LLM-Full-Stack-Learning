# S09 从教学引擎进入 vLLM 与 SGLang：逐字段、逐调用、逐状态读源码

本章的目标不是记住两个框架的架构图，而是能够拿着一条具体请求，解释它怎样从调度计划变成输入张量，又怎样从模型结果回到请求状态和资源释放。你将读固定版本中的真实代码，手算计划字段，区分提交与完成，编写不依赖 GPU 的源码观察程序，并准备一个可审阅的小补丁。先修 S01–S08；遇到 Python dataclass、类型标注、上下文管理器和 Future 时，本章也会解释它们在这条路径中的作用。

本章源码于 2026-11-01 重新核查。vLLM 固定提交为 `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`，SGLang 为 `fa5580f6d0e969a3d44c603d215e2139886e8db2`。这些 SHA 是可复现的阅读坐标，不是推荐你把某个主分支状态直接部署到公司服务。当前课程在 CPU 上执行教学示例与静态检查，没有声称已经安装并运行这些上游完整框架或真实 GPU 模型。

## 1. 第一课：先建立一个不会影响部署的阅读工作台

源码是文本，阅读它不需要先加载模型权重。独立学习 checkout 的价值是把实验与已有部署隔离，并保留版本关系。不要直接修改正在使用的 site-packages；那样即使改对了，也很难知道它对应哪个 commit、哪些修改属于你、怎样恢复和怎样提交评审。

下面命令仅作为你在自己的学习目录准备源码的步骤。目标目录应尚不存在；若已经有自己的 checkout，先检查状态，不要覆盖。clone 需要网络，但不会自动安装模型或启动 GPU。detach checkout 表示工作区停在固定提交，没有跟随分支继续移动；之后创建自己的学习分支即可保存修改。

```bash
mkdir -p upstream
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git upstream/vllm
git -C upstream/vllm checkout --detach e0d00e4b8eb0c89ec30921d0695cfdd409e9c211
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git upstream/sglang
git -C upstream/sglang checkout --detach fa5580f6d0e969a3d44c603d215e2139886e8db2
python -m labs.serving.source_walk --project vllm --root upstream/vllm
python -m labs.serving.source_walk --project sglang --root upstream/sglang
```

[source_walk.py](../../labs/serving/source_walk.py) 不会替你 clone 或安装依赖。它运行 git rev-parse HEAD，读取固定路径，查找 class/def 定义，输出实际 SHA、期望 SHA 与符号所在行。exact_snapshot 为 false 时，它不是说代码一定不能读，而是提醒本书行号与接口需要重新核对。不要修改期望 SHA 来让警告消失，那只是删除证据。

浏览器固定 blob 链接也是合法阅读方式。没有完整开发环境时，先完成字段与调用链笔记，再准备运行测试的环境；不要把“无法马上 import vllm”误认为“今天什么都学不了”。反过来，静态读过代码也不等于测试运行通过，两类证据始终分开记录。

为每个函数准备六栏：具体输入、读取状态、修改状态、返回值、失败/等待点、资源责任。输入不要只写 SchedulerOutput 类型，要写本次甲安排多少 token、computed 是多少、页表有几页。类型让你知道可能有什么，具体值让你知道这一轮实际发生什么。

## 2. 第二课：把问题缩小到一条基础执行路径

第一条路径固定为普通生成任务、单 GPU、普通 dense decoder、无多模态、无 LoRA、无推测解码、无 KV connector，并采用容易理解的执行模式。这不是说其他功能不重要，而是先建立一条完整闭环，再逐项增加分支。你若同时打开所有功能，任何一个计数都可能同时受多个机制影响，很难判断自己是否理解。

配置不是背景信息，而是控制流的输入。例如 speculative_config 为 None 时，不需要把 draft token 纳入基础算例；没有 encoder 输入时，不需要消耗 encoder compute budget；没有 connector 时，不会走外部 KV 加载失败恢复。每个暂时绕过的分支都写出关闭条件，而不是把它从脑中删掉。

读源码时先找入口与出口。入口包括请求如何进入 scheduler，出口包括结果如何交付以及请求何时释放。只从 schedule 第一行顺着读几百行，容易把所有注意力放在准入和分配，却不知道哪些状态在下一轮被修正。完整闭环应该先读薄的一圈，再逐层加深，不必从文件开头按行号阅读全部导入和辅助代码。

本章使用两类代码：固定上游的短节选与课程自己编写的参考程序。节选保留真实字段和关键语句，省略处会说明；参考程序用于手算和故障注入，不冒充上游实现。源码里的名字可能相近但语义不同，例如 cached request 与 prefix cache hit，后面会专门拆开。

| 阅读对象 | 首先回答的问题 | 不应立即推断的事 |
|---|---|---|
| EngineCore.step | 一轮谁调用谁、哪里等待 | 方法返回一定是文本 |
| Scheduler.schedule | 本轮工作与资源怎样决定 | computed 等于设备已完成 |
| SchedulerOutput | 哪些字段跨进程传递 | cached 表示 KV 命中 |
| KVCacheManager | 逻辑块怎样查找、申请、归还 | allocate 每次都 cudaMalloc |
| GPUModelRunner | 计划怎样映射到张量行 | 每个输入位置都需要采样 |
| SGLang overlap loop | 前后批结果怎样配对 | overlap 只等于多一个线程 |


### 2.1 用三个搜索动作建立字段的生命史

第一种搜索找定义，例如 `rg -n 'class SchedulerOutput|def schedule' upstream/vllm/vllm/v1`。定义告诉你参数和返回结构，但还没有告诉你谁负责更新。第二种搜索找写入，例如 num_computed_tokens 的赋值、加法和减法。第三种搜索找读取，尤其是决定输入起点、资源释放和停止条件的地方。三种结果放在一起，才构成字段的生命史。

不要只用一个精确字符串搜索 `num_computed_tokens +=` 后就宣布找全写入。代码可能通过辅助方法、属性 setter、重建对象或赋值为表达式更新它。可以先用字段名广搜，再按上下文分类；找到一个别名时继续追踪。阅读任务没有必要一次证明整个动态语言程序的所有行为，但应诚实记录已覆盖的写入形式与剩余不确定点。

在笔记里给每个写入点标注时相：请求创建、缓存命中、计划提交、结果回写、抢占、恢复、终止。相同的加一操作在不同阶段含义可能不同。特别是 speculative rejection 的回滚，不是“模型倒退”，而是撤销此前乐观计入但最终没有接受的工作；如果不知道时相，会把正常纠正误读为计数 bug。

跨版本迁移时，先比较函数签名和结果结构，再比较调用者，最后看测试。函数名字没变不代表参数语义没变；路径移动也不代表功能消失。建立迁移表，列旧路径/符号、新路径/符号、字段变化和需要重做的断言。保留旧 SHA 的阅读证据，不要直接把旧笔记全局替换为新行号。

Git 的 diff 能帮助你聚焦变化，但差异本身不是完整上下文。一个新增条件可能依赖别处新字段，一个删除分支可能被移动到 helper。看完 diff 后回到新版本完整函数，重新走最小请求，再决定旧结论是否仍成立。这样维护教材和学习笔记，才不会在框架快速演进后只剩下过时的名字。
## 3. 第三课：逐行走过 EngineCore.step

固定源码入口是 [EngineCore.step](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/engine/core.py#L685)。以下节选保留本次阅读需要的调用顺序，省略统计与错误上下文管理器，不是可独立运行的函数。原文件以 Apache-2.0 标注授权，版权属于 vLLM 项目贡献者。

```python
# 来自固定 EngineCore.step 的关键语句；省略外围上下文。
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

第一项检查不只是“有没有等待生成的请求”。源码注释还包含已经完成但尚未从 batch 清除的请求，因此即使没有新 token 工作，也可能仍有状态需要传给 worker。返回的空字典与 false 分别表示本轮没有输出与没有执行模型工作，不能把空输出自动解释为服务故障。

schedule 返回计划对象，描述本轮要做什么。execute_model 接收这份计划并以 non_block=True 提交，返回 Future。Future 是未来结果的句柄，不是结果本身；result() 是取结果的等待边界。提交接口非阻塞，不意味着调用者可以在结果未完成时读取 logits，也不意味着后续任何一步都完全异步。

在等待结果之前准备 grammar bitmask，说明某些 CPU 工作可以与模型执行重叠。这里的 grammar 属于结构化输出约束，后续 sampling 可能使用它屏蔽不合法 token。没有结构化请求时该对象可以为空，但调用链仍保留一致接口。不要看到一个字段为空就从结构上删掉它，它可能属于其他已支持模式。

model_output 为 None 时调用 sample_tokens，不表示模型前向失败。真实 Runner 可以先执行前向并保存临时状态，随后单独采样。必须结合方法契约解释 None，不能套用“没有返回值就是错误”的常识。相同 Python 值在不同接口中有不同语义，源码阅读的任务正是找到这个语义。

处理 abort 队列发生在结果回写之前。请求可能在模型执行期间被取消，因此结果已经算出来也未必还能作为正常输出提交。update_from_output 同时需要原计划与实际结果：计划告诉它本轮为谁安排了多少工作，结果告诉它实际产生了什么。最后附加统计和权重版本信息后返回；这条路径把执行、取消和状态提交连接成一个闭环。

## 4. 第四课：SchedulerOutput 是跨层协议，不是一个随意字典

固定定义位于 [sched/output.py](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/sched/output.py#L232)。其中 scheduled_new_reqs 保存首次发给 worker 的完整请求数据；scheduled_cached_reqs 保存 worker 已经知道的请求所需增量。这里 cached 指 worker 缓存了请求元数据，不能读成“本轮 prefix KV 命中”。

num_scheduled_tokens 是 request ID 到本轮 token 数的映射，total_num_scheduled_tokens 是总和。两者冗余存在并非随意：不同层可能需要快速获得总长度，也需要按请求恢复分段。它们必须满足总量守恒，后续工作簿会用负向控制证明测试能够抓住不一致。

NewRequestData 包含 req_id、prompt_token_ids、sampling_params、block_ids 与 num_computed_tokens 等。block_ids 是按 KV cache group 组织的多组块列表，不一定是一个扁平数组。初学时可以先选择单一 attention group，但读类型时应保留多组结构的含义，不能把生产接口简化成“所有层共用同一个页表”而不注明条件。

CachedRequestData 中 resumed_req_ids 决定 new_block_ids 应追加还是替换。普通继续运行请求通常追加新块；恢复请求可能需要替换旧映射，否则被抢占后重建的页表会被错误拼在旧页表后面。num_computed_tokens 和 num_output_tokens 分开传递，再次说明计算进度与用户输出不是同一个计数。

finished_req_ids 通知 worker 清理它保存的请求状态。Scheduler 自己删除一个 Request 对象，不会自动删除另一个进程的缓存对象，所以需要显式消息。这个字段可能出现在没有新计算 token 的轮次里；如果你为了优化“空批次”而直接跳过全部处理，可能造成 worker 状态泄漏。

| 字段 | 基础例子 | 所属含义 | 关键不变量 |
|---|---|---|---|
| scheduled_new_reqs | 首次进入的甲、乙 | 初始化 worker 请求状态 | 身份与完整元数据一致 |
| scheduled_cached_reqs | 上轮已知的丙 | 更新 worker 已存状态 | 恢复与继续的页表语义不同 |
| num_scheduled_tokens | 甲400、乙112 | 本轮每请求工作量 | 正数、对应有效请求 |
| total_num_scheduled_tokens | 512 | 本轮输入总长度 | 等于映射值之和 |
| finished_req_ids | 丁 | 清理跨轮状态 | 终态不能继续正常生成 |
| scheduled_spec_decode_tokens | 基础模式为空 | 推测候选信息 | 不能混作已接受输出 |

你应该把这张表扩展成“谁写、谁读、什么时候有效”。例如 total 由 Scheduler 建立，被 Runner 用于输入有效长度；req_id_to_index 则来自模型结果，回写时通过它找到对应输出行。计划中的请求顺序与结果中的数组顺序不应靠偶然相同来关联，显式身份映射才能承受重排与过滤。


### 4.1 为什么跨进程状态更新必须有明确身份

假设 worker 已保存甲的 prompt、采样参数和页表。下一轮 Scheduler 只发送“甲新增一页、再计算一个 token”，可以减少重复传输；但 worker 必须确定这是同一个甲，而不是后来复用了相同字符串 ID 的新请求。真实接口通常通过请求生命周期和唯一身份保证这一点，学习实现若允许 ID 重用，就需要显式代次或先确认旧状态已经清理。

增量消息还依赖顺序。如果先应用“追加页九”，再应用一个迟到的“恢复并替换为页三”，新映射可能被旧消息覆盖。因此 cached 更新不是任意可交换的字典合并。你要知道发送顺序、接收顺序和在途批次之间的依赖；若协议允许乱序，就必须增加版本或提交序号来判断过期更新。

finished_req_ids 的意义也可以从这里理解。它是发送给另一个状态拥有者的清理通知，不只是日志。通知丢失、重复或延迟时，系统分别面临泄漏、重复清理和旧状态残留。幂等清理可以处理重复，但不能单独解决丢失；可靠队列或重建协议处理的是另一类问题。不要把所有可靠性要求压到一个 set 类型上。

资源身份不只请求 ID。页号会复用，tensor buffer 会复用，批次槽位也会复用。每个层次需要相应有效期，不能把“请求还在”当作所有底层地址都仍属于它的证明。本课程 S05 的 generation 句柄是这种原则的极简表达，读真实框架时则要找对应的在途计数、序号、事件或 barrier。
## 5. 第五课：真实调度计数怎样追赶已知位置

固定 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/sched/scheduler.py#L698) 的开头解释了一种统一视角：调度器让 num_computed_tokens 追赶 num_tokens_with_spec，而不是强制把所有逻辑都拆成互不相关的 prefill 与 decode 两套状态机。基础模式中没有推测 token，已知位置就是 prompt 加已生成输出；启用推测后还要考虑草稿与占位。

真实代码还分别维护 token_budget 与 input_budget，并可能为 drafting 预留槽位。因此本课程 S02 的单一预算模型只是入门，不应声称覆盖全部生产配置。读到两个看起来相似的预算时，要分别追踪初值、扣减点与用途，不能因为普通模式中数值相同就合并它们。

运行请求路径先计算希望推进的数量，再受长 prefill 阈值、剩余预算、输入预算和最大模型长度约束。固定版本还有一个值得注意的条件：只有多个可调度请求时才使用长 prefill 阈值；若只有一个请求，没有其他请求会被它饿死，可以让它使用更大预算。若开启自适应阈值，又会结合输入预算与请求数量调整下限。

这说明“chunk_size 永远等于某个配置数”不是可靠概括。配置通常参与条件分支，而不是机械决定所有轮次的固定工作量。你必须记录具体模式和队列状态，才能从参数推出计划。S17 的三请求八百 token 算例会锁定这些条件，避免一个隐含默认值改变预期。

考虑三个八百 token prompt，预算一千零二十四，长 prefill 阈值四百，关闭 adaptive 阈值，没有缓存命中、推测、多模态或资源压力。按基础顺序，甲先拿四百，剩六百二十四；乙再拿四百，剩二百二十四；丙拿剩余二百二十四。它不是自动平均分成三份。预算改为五百一十二时，得到四百、一百一十二，第三请求本轮没有工作。

| 候选 | 初始待计算 | 阈值后需求 | 进入时剩余预算 | 本轮安排 | 剩余预算 |
|---|---:|---:|---:|---:|---:|
| 甲 | 800 | 400 | 1024 | 400 | 624 |
| 乙 | 800 | 400 | 624 | 400 | 224 |
| 丙 | 800 | 400 | 224 | 224 | 0 |

但候选数量算出来还不等于计划已经提交。KV 申请失败时，调度器要走抢占、跳过或等待路径，不能先把预算扣掉再假装工作会执行。真实运行请求与等待请求路径的资源处理有所不同，应分别读申请成功后的提交代码。只看 min 公式就宣布“我已读懂调度器”，会漏掉内存约束与失败语义。

## 6. 第六课：computed 在计划后推进，为什么不是设备完成证据

固定源码的 `_update_after_schedule` 明确在 schedule 后推进请求账本。以下短节选来自同一 Scheduler 文件，省略后续结构化输出与前缀集合维护。它是生产协议中的关键细节，不能被变量名字误导。

```python
# 固定快照 _update_after_schedule 的关键更新。
for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    request.num_in_flight_tokens += num_scheduled_token
```

第一项推进允许下一轮知道已经安排了多少位置，不重复提交同一段；第二项记录这些位置仍在途。两个数字同时增长，说明 num_computed_tokens 在这个上下文中包含乐观的已调度进度。它不能独自证明 GPU 已完成对应 K/V 写入。若后续推测 token 被拒绝，回写还会调整 computed；若请求被抢占或取消，也有额外处理。

假设甲原 computed 为四百，本轮安排一百，计划建立时保存的起点仍应是四百，Runner 才能取位置四百到四百九十九。计划完成后 Scheduler 账本推进到五百，in_flight 增加一百。如果错误地用推进后的五百作为本轮输入起点，会跳过一整段。保存计划时刻的值与更新长期状态的时序因此非常重要。

| 时刻 | 计划保存起点 | Scheduler computed | in_flight | 可以宣称什么 |
|---|---:|---:|---:|---|
| 调度前 | 尚未建立 | 400 | 0 | 前四百位置已处于既有状态 |
| 计划建立 | 400 | 400 | 0 | 将执行一百位置 |
| 调度账本推进 | 400 | 500 | 100 | 一百位置已安排，在途 |
| 正常结果回写后 | 400 | 500 | 0 | 这份工作已被结果处理 |

真实系统可以有多个在途批次，表格只是一个无推测、无取消的最小场景。in_flight 的减少发生在结果处理路径，而不是你看到 Python 调用 execute_model 返回时随意减掉。异步资源释放还需要与实际设备/传输完成机制配合，不能只凭这个 CPU 整数直接回收所有页。

这也解释为什么“变量名含 computed”并不足以理解源码。你应找所有赋值和增减位置，记录它在各阶段代表的含义，尤其是乐观推进与回滚。成熟框架为了重叠调度与执行，常常需要比教学同步循环更复杂的账本；学习目标是理解这些账本为何存在，而不是批评它们“不够直观”后删掉。


### 6.1 用独立账本演示“取消也必须结算”

下面程序只演示账本协议，不替代真实 Scheduler。submit 返回不可变的提交记录，保存本轮起点；长期 computed 立即推进，并登记 in_flight。cancel 只阻止用户结果提交，complete 仍然结算对应工作。完成同一提交两次会明确拒绝，避免在途计数变负。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Ticket:
    serial: int
    start: int
    count: int

class Ledger:
    def __init__(self, computed=0):
        self.computed = computed
        self.in_flight = 0
        self.cancelled = False
        self.pending = {}
        self.output = []
        self.next_serial = 0

    def submit(self, count):
        if self.cancelled or type(count) is not int or count <= 0:
            raise ValueError("request cannot accept this work")
        ticket = Ticket(self.next_serial, self.computed, count)
        self.next_serial += 1
        self.pending[ticket.serial] = ticket
        self.computed += count
        self.in_flight += count
        return ticket

    def cancel(self):
        self.cancelled = True

    def complete(self, ticket, tokens):
        actual = self.pending.get(ticket.serial)
        if actual != ticket:
            raise ValueError("unknown or already completed ticket")
        del self.pending[ticket.serial]
        self.in_flight -= ticket.count
        if not self.cancelled:
            self.output.extend(tokens)
        assert self.in_flight == sum(t.count for t in self.pending.values())

ledger = Ledger(computed=400)
ticket = ledger.submit(100)
assert ticket.start == 400
assert (ledger.computed, ledger.in_flight) == (500, 100)
ledger.cancel()
ledger.complete(ticket, [7])
assert ledger.output == [] and ledger.in_flight == 0
try:
    ledger.complete(ticket, [7])
except ValueError:
    pass
else:
    raise AssertionError("duplicate completion accepted")
print("cancelled result discarded, submitted work settled exactly once")
```

pending 保存提交身份，computed 保存乐观推进后的边界，in_flight 保存尚未结算数量，output 保存已经允许对用户提交的事实。这四类状态各有职责。取消后 output 不增长，但 in_flight 仍减少；若把它们合成一个“进度”整数，你就无法同时表达“算完了但不应该显示”这件事。

该程序没有实际线程、设备事件或乱序依赖，只检查单对象账本。真实多批次执行还要保证后一个计算能看到前一个 KV，不能因为允许多个 Ticket 就假定所有工作可以并行。你可以在此基础上增加 predecessor 字段，让完成前必须满足前驱条件，再研究哪些独立请求可以重排。

它也不支持推测拒绝后的 computed 回滚，故意保持最小范围。扩展时应分别保存安排量、接受量与需重算范围，写出每个事件如何变化，避免简单把 count 减回去导致覆盖其他在途工作。这个例子的价值是让你在读生产回滚代码前，先看清提交、取消和结算三个动作不能互相替代。
## 7. 第七课：KVCacheManager 的返回值与失败路径

固定 [KVCacheManager](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/kv_cache_manager.py) 中，get_computed_blocks 在本快照返回三项：命中的块结构、新命中 token 数，以及 shared_prefix_boundary。第三项与稀疏保留组的共享前缀边界有关，基础全注意力模式可以先记为零，但不能照旧教程只用两个变量解包后声称接口没变。

该函数首先检查是否允许 prefix cache lookup，关闭或请求不适用时返回空块与零计数。命中块必须满足相应完整块/对齐条件。全部 prompt 已缓存时仍需重算最后位置获取 logits；因为分配接口的对齐限制，实际重算可能是一整块尾部，而不是恰好一个 token。这个机制与本课程 S04 的最后位置限制相呼应，但真实多组缓存还存在更多约束。

allocate_slots 的参数需要逐个理解。num_new_tokens 是本轮新增计算位置；num_new_computed_tokens 是本地前缀新命中的位置；num_external_computed_tokens 是 connector 提供的外部缓存位置；num_lookahead_tokens 为推测等路径预留；delay_cache_blocks 表示传输未完成时暂不把块当作可正常缓存的完成状态。它们不是五种拼写不同的长度。

返回 KVCacheBlocks 表示相应逻辑块安排成功，返回 None 表示资源条件不满足。不要把它理解为“每次都分配一个新的 GPU tensor”。大块设备缓存通常在初始化阶段准备，调度期间管理的是块身份、引用、页表和使用范围。要确认物理分配，继续追到实际 tensor 创建，而不是凭 allocate 这个动词猜测。

更微妙的是失败是否完全无副作用。S03 的教学 PagePool.append 承诺失败原子性，但真实 allocate_slots 在容量检查路径中可能先移除已经不再需要的滑动窗口块，源码明确说明即使本次无法调度也可以做这件事。因此不能把教学原子性直接强加到生产接口。正确阅读方式是识别哪些安全清理允许发生、哪些新分配必须成功后提交。

free(request) 委托 coordinator 按 request ID 归还块；缓存开启时归还活动引用不一定意味着底层显存释放，也不一定意味着块内容立刻清零。源码说明释放顺序倾向从尾部开始，以影响缓存驱逐顺序。你应分别记录活动所有权、缓存可复用身份和设备 tensor 生命周期，不能看到 free 函数返回就要求 nvidia-smi 数值下降。


### 7.1 读 SGLang RadixCache：查找也可能改变树结构

固定 [RadixCache](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/mem_cache/radix_cache.py#L369) 的 match_prefix 接收 MatchPrefixParams，内部从 params.key 取 RadixKey，再根据配置转换视图与做 page alignment，最后调用树匹配 helper，返回 MatchResult。它不是直接接收任意字符串并返回一串 K/V 数字；key 是 token 身份，value 通常是设备缓存索引。

RadixKey 在该快照包含 token_ids、extra_key、cache_salt、is_bigram 和 limit。extra_key 与 cache_salt 参与命名空间隔离，limit 允许对 token 数组形成有界视图而不总是复制。基础普通 token 模式可先忽略 bigram 视图，但不能把 limit 当成真实数组已经被截断；逻辑长度与底层存储长度可能不同。

match_prefix 的文档明确说，查找可能在匹配结束于节点片段内部时拆分节点，并刷新访问元数据。所以“查询一定只读”在这个接口上不成立。拆分树节点不等于重新计算或复制全部 KV；它主要把已有 token 范围与索引片段重新组织成更精确的边界。若你在并发环境调用它，必须尊重框架的串行访问或同步协议，不能当作无状态纯函数随意并行。

手算一个压缩路径节点保存 token `[1,2,3,4]`，查询 `[1,2,9]`。公共前缀为前两个 token，树可以把原节点拆成前缀 `[1,2]` 与后缀 `[3,4]`。插入新路径时再增加后缀 `[9]`。旧路径对应的设备索引仍可按片段引用，模型数值不因树结构拆分而改变。真正发生复制与否，要继续看 value 切片和 allocator，而不能从“split”名字直接判断。

evict 从可驱逐叶集合构建优先队列，释放叶子对应的缓存段，删除叶子；若父节点不再有孩子且 lock_ref 为零，父节点也可能进入候选。这个过程解释了为什么只看访问时间还不够：正在被请求保护的路径不能驱逐。inc_lock_ref 沿祖先链增加保护引用，并在零到一变化时把大小从 evictable 转到 protected；多一个引用不意味着每次都重复移动同一份容量。

把它与 S04 的 OrderedDict 参考比较：二者都需要身份、命中、保护和驱逐，但数据结构复杂度与共享方式不同。字典参考通过完整 tuple key 表达前缀，Radix 把公共路径显式共享。不要强行找“一条字典记录等于一个树节点”的映射；真正可迁移的是所有权不变量和同一逻辑前缀必须产生同一缓存语义。

练习时先画节点 token 片段、父子关系、设备索引范围和 lock_ref 四列，执行查找、拆分、插入、保护、释放保护与驱逐。每一步都检查剩余请求的逻辑前缀仍能恢复。只检查树节点数变化，会漏掉索引范围重叠、释放仍被保护的祖先和命名空间串用等错误。
## 8. 第八课：在 GPUModelRunner 中追一组真实索引

固定 [GPUModelRunner._prepare_inputs](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/worker/gpu_model_runner.py#L1924) 首先要求本轮总 token 数与请求数都大于零，提交 block table 的拷贝，再构造请求行索引、局部 query 位置、逻辑 positions 和 token 索引。这里可以把 S06 的打包器直接与真实代码对照。

代码对每请求安排数使用 np.repeat。若数量为二、五、三，req_indices 就是零、零、一、一、一、一、一、二、二、二。它不是 token ID，而是每个 flat 行来自哪个请求槽。局部 query_pos 则为零、一、零、一、二、三、四、零、一、二。两者长度都为十，但语义完全不同。

positions 等于每个请求的原 computed 加局部 query_pos。设原 computed 为四、十、零，则 positions 为四、五、十、十一、十二、十三、十四、零、一、二。这样长请求继续 decode 或分块 prefill 时仍使用正确逻辑位置。不要因为 flat 行号从零到九，就把位置也设置成零到九。

token_ids_cpu 是按请求槽与最大序列容量存储的二维表。把它 flatten 后，某个请求的第 p 个位置对应 `p + req_index * row_width`。源码随后用 index_select 收集当前需要的 token 到紧凑 input_ids 缓冲区。这个公式是普通二维行主序地址计算，与 S03 的页表地址翻译不是同一层，但都要求区分逻辑索引与物理位置。

```python
import numpy as np

counts = np.array([2, 5, 3])
computed = np.array([4, 10, 0])
req_indices = np.repeat(np.arange(3), counts)
local = np.concatenate([np.arange(n) for n in counts])
positions = computed[req_indices] + local
row_width = 16
table = np.arange(3*row_width).reshape(3, row_width)
indices = positions + req_indices*row_width
packed = table.reshape(-1)[indices]
assert positions.tolist() == [4,5,10,11,12,13,14,0,1,2]
assert packed.tolist() == [4,5,26,27,28,29,30,32,33,34]
assert len(packed) == int(counts.sum()) == 10
print({"request_slots":req_indices.tolist(),
       "positions":positions.tolist(),"input_values":packed.tolist()})
```

本例 table 用可辨识整数表示每个槽位，不是真实词表数据，所以二十六之类数字只是帮助追踪来源。实际框架收集的是合法 token IDs。选择可辨识的合成数据能立即发现跨请求行错位；若全部填零，很多错误索引也会返回零，测试就失去敏感性。

execute_model 在 _prepare_inputs 后确定图执行模式与 padding，构造 attention metadata，准备 input_ids、positions 与模型参数，设置 forward context，再调用 _model_forward。普通最后 pipeline rank 从 hidden_states 的 logits_indices 选择采样行，然后 compute_logits。非最后 rank 可能返回 IntermediateTensors，pooling 任务也有自己的结果路径，所以返回类型比一个 logits list 丰富得多。

## 9. 第九课：execute_model 与 sample_tokens 之间有一份临时所有权

execute_model 入口检查 execute_model_state 是否为空。如果上一次返回 None 后没有调用 sample_tokens，下一次 execute_model 会报状态错误。这是一个明确的阶段协议：前向结果暂存在 Runner 内，必须按顺序消费，不能把同一个对象当作无状态纯函数无限调用。

基础生成路径会保存 SchedulerOutput、logits、推测元数据、hidden states、采样行隐藏状态、connector 输出、图统计与 slot mappings 等，随后返回 None。sample_tokens 取出这些临时状态并清空保存位置，然后应用结构化输出 mask 与采样逻辑。清空意味着这份临时状态被转移到当前处理流程，避免下一次执行误用旧结果。

这个设计并不意味着所有模式都返回 None。非最后 PP rank、pooling 或无 token 工作可能提前返回其他结果；sample_tokens 在没有 execute_model_state 时也有 connector/PP 相关处理。因此阅读时应建立分支表，写出当前模式走哪条路，不要把基础路径的结论推广到所有配置。

一个很常见的初学错误是只看函数最后的 return，然后说“这个函数永远返回 None”。正确方法是搜索整个方法中的全部 return，按条件归类。类型标注给出联合类型只是提示，实际控制流才告诉你各类型何时出现。S10 会进一步解释为什么流水线中间阶段不应该生成完整文本。

跨阶段保存的 SchedulerOutput 保证 logits 与当时请求映射一起存在。如果你为了节省内存只保留 logits，却在采样时使用最新 batch 的顺序，就可能把甲的结果分配给乙。结果数值正确而身份错误，比显式异常更危险。每个跨异步边界的对象都应该能回答“它属于哪一轮、哪些请求、哪些代次”。

## 10. 第十课：结果回写先处理在途账本，再判断能否提交输出

固定 [update_from_output](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/sched/scheduler.py#L2252) 取出采样 token、logprobs、connector 结果等，再遍历计划中的请求。若请求对象仍存在，先减少该计划对应的 num_in_flight_tokens，并处理 stale output 计数，然后检查外部 KV 加载失败、请求缺失或已结束等情况。

为什么先减少在途再跳过 finished？请求虽然已取消，之前提交的工作仍然可能完成；这份完成通知仍需从资源账本中结算。若简单在循环第一行看到 finished 就 continue，可能永远留下 in_flight，阻止后续安全释放。逻辑输出被丢弃与执行工作已结束，是两个必须分别记录的事实。

请求不存在或已结束时，不应再追加正常输出。某些异步 KV 传输场景会暂时保留已结束 Request 对象，以完成交接清理，所以不能只用“字典里是否还存在”判断活跃。源码同时检查 is_finished，说明对象存在与请求仍可交付之间有明确区别。

正常请求通过 req_id_to_index 找到结果数组中的位置，再读取 generated_token_ids。这个映射让结果顺序与计划顺序不必靠偶然相同保持一致。若启用推测解码，还要根据接受数量修正 computed 与占位；陈旧结果不能再次回滚已经被抢占重置的状态。基础路径没有这些草稿，但你已经能理解为什么计划计数和实际接受输出可能不同。

把这个过程画成事务更容易理解：计划提交时登记预期工作，执行结果回来时结算在途责任，检查请求代次与状态，再提交允许的用户可见输出，最后根据停止条件释放或继续。它不是数据库事务的完整实现，但这个思维能防止把“计算结束”与“结果可见”合并成一个不加判断的 append。

## 11. 第十一课：SGLang normal 与 overlap 的具体时间关系

固定 [SGLang Scheduler](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/managers/scheduler.py#L2090) 的 normal loop 先 ingest_requests，再获取计划，将 plan.running_batch 保存为运行集合，从 plan.batch_to_run 得到本轮 batch，调用 run_batch，随后立即 process_batch_result。没有 batch 时进入 idle 处理，最后更新 last_batch。

注意 plan.running_batch 与 batch_to_run 不是同一个概念：前者是跨轮保存的运行集合，后者是这一轮实际执行对象。一个请求属于运行集合，不保证每轮都被执行；一个 prefill 批次也可能在完成后与运行 decode 集合合并。读 get_next_batch_to_run 时应分别追踪这两个结果，不能把所有变量都口头简称为“batch”。

overlap loop 创建 result_queue，元素是 batch 与 result 的配对。正常重叠路径先启动当前批，再把 `(batch.copy(),batch_result)` 放入队列，随后处理上一批结果。保存配对是为了让结果按当时的 batch 身份解释；copy 的具体深浅仍要看实现，不能仅凭方法名宣称所有底层数组都已深拷贝。

| 轮次 | 启动的新批 | 入队后结果队列 | 本轮处理的旧结果 | 轮末 last_batch |
|---|---|---|---|---|
| 第一轮 | 甲批 | 甲批结果 | 没有上一批 | 甲批 |
| 第二轮 | 乙批 | 甲批、乙批 | 甲批，处理后剩乙批 | 乙批 |
| 第三轮无新批 | 无 | 乙批 | 乙批，处理后为空 | 空 |

这个表只描述允许 overlap 的基础路径。源码还有 disable_overlap_for_batch，某些条件下会在启动当前批之前处理上一批。你必须检查触发条件，不能用一张固定时间图概括所有模式。当前快照还在启动后调用 WAR barrier 相关方法，注释说明结果处理需要受当前 forward 共享读取的约束；这与 S05 的在途读写所有权直接相关。

当前批采样安排在上一批结果处理之后，因为它可能依赖更新后的 grammar 等状态。于是 overlap 不只是“同时跑两个线程”，而是精心安排哪些阶段可以重叠、哪些状态依赖必须串行。真正优化异步循环时，要先画出依赖关系，再考虑提前或延后某一步；随意移动 process_batch_result 可能破坏结果身份或共享状态。


### 11.1 比较两个框架时，先对齐不变量再比较对象

vLLM 的 SchedulerOutput 与 SGLang 的 ScheduleBatch 并不是两个可以逐字段改名的结构。前者清楚承载一次计划和 worker 状态增量，后者在其调度与执行体系中拥有不同职责。比较时先列共同问题：本轮谁执行、每请求计算哪些位置、缓存由谁保护、结果属于哪个请求、取消后怎样结算。只有回答这些问题后，类与字段的差异才有意义。

同样，vLLM 的块管理与 SGLang 的 radix 树也不是相互替代的单个算法。分页解决物理存储与增长，前缀索引解决如何找到可复用历史，引用与锁保护解决什么时候可以回收。一个框架可以同时需要这些层次，只是组织边界不同。若把“一个用分页、一个用树”当成完整比较，就把不同维度放进同一选择题。

性能比较需要实际工作负载和安装环境。本章核查代码只能支持“这个版本具有这条路径、这里存在这个字段或状态协议”，不能据此宣布哪一个总是更快。即使某个源码循环看起来更短，设备 kernel、通信、缓存命中和模型结构都可能决定整体耗时。静态代码长度不是性能指标。

你可以为两个框架各画同一个三请求情景，记录计划数量、缓存引用、结果队列和取消事件。若它们采取不同调度顺序，只要各自满足明确的正确性与公平性契约，就不必强求完全相同。比较的重点是理解选择带来的成本与约束，再用 S08 方法测量你关心的目标。

最后保留每项比较的版本与模式。启用 overlap、推测解码或不同 attention backend 后，之前的同步图可能不再适用。把模式写在图旁边，未来读者才不会把一个方便教学的基础路径当成整个框架在所有配置下的唯一行为。

## 12. 第十二课：写一个不导入框架的源码观察器

阅读工具可以使用 Python AST 解析源码而不执行导入。AST 是语法树，能告诉你类、函数、调用表达式和行号。下面程序在课程 Engine 上演示寻找一个方法中的属性调用。它不会加载模型，不调用函数，也不证明动态运行时一定走过所有列出的分支。

```python
import ast
from pathlib import Path

def attribute_calls(source, class_name, method_name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == method_name)
    rows = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            rows.append((node.lineno, ast.unparse(node.func)))
    return sorted(rows)

source = Path("labs/serving/engine.py").read_text()
rows = attribute_calls(source, "Engine", "step")
names = [name for _,name in rows]
assert "self._ingest" in names and "self._admit" in names
assert "self.pool.append" in names and "self.backend.logits" in names
assert "self._finish" in names
print(rows)
```

先定位顶层类，再定位类内方法，避免用全文正则误匹配同名函数。ast.walk 会遍历方法体内所有语法节点，找到 Call 且函数表达式是 Attribute 的节点；ast.unparse 把 self.pool.append 这类表达式还原为可读文本。最后按行号排序，得到静态出现顺序，而不是实际执行轨迹。

它会列出互斥分支中的调用，也可能包含嵌套函数里的调用；循环中的一个调用只出现一次，却可能运行很多次。因此它适合生成阅读索引，不适合宣称精确调用次数或性能热度。要了解运行时行为，需要 trace、测试或调试器；静态分析与动态证据互相补充。

把同一函数应用到固定上游文件时，只需换 source、类名和方法名。先确认 SHA，再解析 EngineCore.step 或 GPUModelRunner.execute_model，记录重要调用的行号。若源码语法超出你当前 Python 版本支持，AST 解析会失败；这属于工具环境限制，不说明上游代码本身错误。不要为了让工具运行而随意改上游语法。

进一步练习可以提取一个字段的赋值位置，区分普通赋值、增量赋值与属性读取。即使这样，也无法自动理解别名、动态 setattr 或跨进程消息，所以输出应叫“候选写入点”。工具帮助你缩小阅读范围，理解仍需要回到控制流和具体状态。


### 12.1 从真实测试 helper 看见“模型配置”与“模型执行”的区别

本课程提供 [test_course_scheduler_contract.py](../../labs/serving/upstream_tests/test_course_scheduler_contract.py)，用于复制进固定上游测试目录。它导入真实 Scheduler、ModelRunnerOutput、RequestStatus，以及上游 create_scheduler/create_requests。测试不是把教学 Engine 改名为 vLLM，而是使用真实管理器接口构造请求与计划。

create_scheduler 仍会构造 ModelConfig、SchedulerConfig、CacheConfig 与并行/KV 配置。即使测试不做 GPU 前向，导入链和配置读取也可能需要正确开发依赖与本地模型配置。固定 helper 的 ModelConfig 还设置了 trust_remote_code=True，因此课程只使用已核验的官方 OPT 配置或自行审阅过的本地兼容资料，不应随意替换成未知模型目录。配置读取也可能进入代码执行路径，不能把它等同于读取一份永远被动的 JSON。因此 cpu_test 标记表示测试目标不需要真实模型 GPU 计算，不等于任何干净 Python 环境都能直接运行，也不保证完全不触及缓存资料。

empty_result 为每个已调度请求建立 req_id_to_index，并返回对应的空 sampled_token_ids。部分 prefill 还没有到达 prompt 末尾，所以空输出是正常的执行结果，不是失败。若你偷懒返回全局空列表而不保留请求槽，后续映射可能不符合接口；若每个 chunk 都伪造一个 token，又会改变被测状态机。

预算测试同时断言具体分配、总预算、total 字段守恒和单请求阈值。这些断言关注不同性质：具体分配抓策略变化，总预算抓超用或空耗，total 守恒抓跨层协议不一致，单请求阈值抓长 prompt 限制。不能因为四条都涉及数字，就认为它们是没有意义的重复。

取消测试先记录 free_before，调度部分 prompt 后确认页数减少，回写空结果，再取消两次，要求第一次返回有效请求、第二次为空，空闲页恢复，下一轮不再安排该 ID。它关闭 prefix cache，是为了让“活动释放后全部空闲”成为明确预期；开启缓存时，合法缓存持有会改变这一断言，必须写更细的所有权检查。

安装脚本默认 dry-run，核对固定 SHA，拒绝覆盖同名目标，只复制课程测试文件，不改生产实现、不安装依赖、不执行测试。进入自己的上游开发环境后再运行 pytest。课程当前只核查源码接口与 Python 语法，尚未在本环境执行完整上游依赖测试；这个状态不能被本仓库 unittest 通过所替代。
## 13. 第十三课：四种源码误读与可验证的纠正

**误读一：cached request 就是前缀命中。** 纠正证据来自 SchedulerOutput 注释与 New/Cached 数据结构。cached 表示 worker 已有请求元数据，目的是只发送增量；前缀命中由 KV 管理器和 computed blocks 表达。构造一个完全不共享 prompt、连续 decode 的请求，它第二轮仍会成为 cached request，却没有跨请求 prefix hit。

**误读二：computed 增长说明 GPU 完成。** 纠正证据是 _update_after_schedule 同时增加 computed 与 in_flight，且调用发生在执行结果回来之前。用一百位置的计划表跟踪提交与完成，能看到 computed 已推进而 in_flight 仍非零。若要释放页，需要正确完成协议，而不是检查 computed 是否达到某个数字。

**误读三：allocate 返回 None 保证没有任何状态变化。** 教学 PagePool 承诺失败原子，但真实 KV 管理器允许先释放不再需要的滑动窗口块。读失败返回前的代码，写明允许的清理与禁止的部分新提交。接口保证必须来自具体实现和测试，不能因为函数名相似就迁移全部假设。

**误读四：导入失败就是回归测试抓到了 bug。** 上游测试可能因依赖缺失、配置缓存缺失、版本不符而失败。这些失败尚未执行到目标断言，不算证明调度错误。负向控制应该让故意破坏的计数在预期断言失败，而正确版本通过；保存 traceback 的具体位置，区分环境、测试和被测实现问题。

排查这些误读时，不需要先读遍整个项目。每个问题都可以找到一个定义、一个写入点、一个读取点和一个最小情景。把四者连成证据链，比收集很多博客架构图更有用。等基础链条完整，再扩展到 connector、推测、多模态和分布式，复杂分支才有可依附的结构。

## 14. 第十四课：分级练习、完整答案与源码修改项目

**练习一，解释 Future。** execute_model(non_block=True) 返回的是未来结果句柄，future.result() 可能等待。非阻塞提交允许中间准备 grammar 等工作，但不允许在结果未就绪时读取模型输出。答案必须指出提交与完成两个时刻，不能只说“它是异步所以更快”。

**练习二，计划与结果。** SchedulerOutput 描述每请求安排数量、元数据更新和清理通知；ModelRunnerOutput 描述实际采样等结果。update_from_output 需要二者，才能结算在途工作并把结果映射回请求。若只保留结果 token 列表，会失去没有生成 token 的部分 prefill 工作与资源进度。

**练习三，字段守恒。** 计划甲三百、乙一百二十、丙九十二，总数应五百一十二。若 total 写五百一十三，输入缓冲有效长度与每请求分段可能不一致。参考断言是 total 等于 sum(mapping.values())，并分别校验每项为正整数；只检查 total 不超过预算不能抓住这类内部不一致。

**练习四，cached 的含义。** 一个请求第一次进入 worker 需要完整元数据，后续只发增量，因此进入 scheduled_cached_reqs。即使 prefix cache 完全关闭也可发生。答案要把“worker 元数据缓存”与“模型前缀 KV 复用”写成两个独立概念，并指出各自数据结构。

**练习五，恢复页表。** resumed_req_ids 中的请求使用 new_block_ids 替换原页表，普通继续请求则按契约追加。错误追加会保留已经失效的旧页映射。测试应先构造抢占恢复，再检查页表内容，而不是只看总页数，因为同样长度也可能包含错误页号。

**练习六，预算手算。** 三个八百 token 请求，阈值四百、预算五百一十二，基础无资源压力路径安排四百、一百一十二、零。它不是平均分配。若只剩一个请求，固定版本可能不再应用长 prefill 阈值，因此不能无条件期待四百；必须写出可调度请求数与相关配置。

**练习七，computed 与在途。** 原 computed 四百、in_flight 零，安排一百后为五百与一百；结果结算后 in_flight 回零。计划输入起点仍是四百。答案若写新输入从五百开始，就把更新后的长期账本误用为本轮计划快照。

**练习八，真实 KV 命中返回。** 本固定版本 get_computed_blocks 返回三项，第三项是 shared_prefix_boundary。基础模式可以为零，但调用方仍应按当前签名处理。跨版本阅读时先看定义与调用者，不能直接复制旧教程的二项解包代码。

**练习九，None 的多种语义。** allocate_slots 返回 None 表示分配条件失败；execute_model 在基础两阶段路径返回 None 表示前向状态已保存，接下来需要 sample_tokens。它们不能统一处理成“请求失败”。答案必须引用各自方法契约，而不是对 Python None 下一个全局解释。

**练习十，索引公式。** row_width 十六，请求槽二、逻辑位置五，flatten 索引为三十七。position 仍是五，req_index 仍是二。该索引用于 CPU token 表收集，不是 KV 物理 slot；后者还需要页表和块内偏移。两个地址问题都叫 index，但对象不同。

**练习十一，取消后的回写。** 请求已 finished，结果仍返回。应先结算存在对象的在途计数，再按状态丢弃不应交付的输出，并继续相应清理协议。若直接追加 token，用户会在取消后收到新内容；若只跳过不结算，资源可能长期不能释放。

**练习十二，overlap 队列。** 第一轮启动甲，队列有甲；第二轮启动乙后队列甲乙，随后处理甲，剩乙；第三轮无新批时处理乙并清空。答案要注明这是允许 overlap 的基础路径，disable_overlap 分支会改变处理时序。batch 与 result 必须配对，不能拿 current batch 解释 previous result。

**练习十三，静态观察器局限。** AST 得到调用出现位置，不知道运行次数、分支条件是否成立、动态派发目标或设备耗时。它可用于源码导航和检查符号存在，不能替代运行 trace。若一个函数有十个分支，观察器列出十条调用不表示每个请求都会执行十条。

**练习十四，设计负向控制。** 在测试进程包装 schedule，把 total 增加一而保持每请求映射不变。正确测试应在总量守恒断言失败；恢复后通过。若失败在 import 或模型配置读取，负向控制尚未验证目标性质。不要修改公司部署，也不要把故意破坏的补丁提交到生产分支。

**练习十五，第一次上游补丁范围。** 选择一个具体可复现问题，例如取消后重复清理或计划计数不一致，先写最小输入与预期，再复用真实测试 helper。说明受影响路径与未覆盖模式，运行相关测试，记录环境。不要第一天重写调度策略，因为没有足够证据区分行为变化与回归。

第一阶段项目是阅读账本：为 EngineCore、SchedulerOutput、KVCacheManager、Runner 各写输入/输出/状态表，用本章具体数字走一轮。第二阶段运行课程 AST 观察器，核对静态调用顺序，再在固定上游文本上重做，保存 SHA 与文件路径。第三阶段进入 [S17 源码工作簿](17-source-workbook.md)，使用真实上游 helper 构造计划与取消测试。

第四阶段在独立学习分支做小修改。先看 git status，确保没有无关文件；修改应围绕一个可描述的触发条件，附带正向与负向验证。第五阶段才对涉及热路径的更改做短时真实性能检查，使用 S08 的负载与指标规范。测试文件增补不需要假装提高吞吐，运行时修改也不能只凭单元测试就宣称性能无回归。


### 14.1 一个完整的小补丁说明应该长什么样

以下设计以“调度总数与每请求数量不一致时尽早定位”为主题。触发条件是某个策略或后续变更写错 total_num_scheduled_tokens，导致 Runner 按错误总长度准备输入。原行为可能在更深层出现索引或 shape 错误，难以关联到生成计划；期望行为是在调试检查中指出具体轮次与两个计数，不改变正常计划内容。

实现位置应选择计划刚构造完成、交给执行层之前，并使用项目现有调试机制决定是否启用。检查只读 counts 与 total，不调用可能修改缓存的 helper，不为了打印日志复制完整 prompt。错误信息包含匿名请求数量、总数和分配摘要即可，避免泄露用户内容。热路径成本要明确：求和是按请求数线性，不应无条件加入昂贵序列化。

验收分三层。第一，正确计划开启检查后通过，关闭检查行为不变。第二，测试进程注入 total 加一变异，断言在预期检查失败，恢复后通过。第三，部分 prefill、decode、取消清理与空计算通知都按契约处理，避免检查错误地要求每轮都有正总数。若涉及推测或 connector，需说明这些模式是否仍满足相同总量定义。

评审描述要写清楚问题、触发、修改后的行为与已运行检查，不按“我先看了哪个文件、后来又想到什么”的时间顺序叙述。若没有复现真实上游 bug，就将它定位为教学或调试验证练习，不伪造 issue。真正向上游贡献时，应先确认维护者是否已有同类检查或更合适接口，再提出小范围补丁。

提交前检查 diff，确保没有模型文件、缓存、日志密钥或无关格式化。记录固定基础 SHA 与补丁 commit，让未来可以重新应用或比较。依赖安装失败时写明失败位置，不把结果总结为“测试不通过所以补丁错误”；设备实验未执行时单独标注。一个范围小但证据清楚的补丁，比大量未经验证的重构更适合作为框架学习的第一份成果。

### 14.2 从一个问题扩展到下一层问题

完成计数校验后，下一步可以追踪一个具体资源问题，例如取消后某页仍被持有。先用 Request 身份找到计划，再用 block IDs 找管理器引用，最后用 Runner/connector 完成事件解释释放时机。每向下一层扩展，只增加一个新的状态维度，并保留之前的输入与断言，这样故障不会再次变成不可观察的“大框架偶发问题”。

再下一步才研究性能。例如怀疑输入准备成为瓶颈，先用 S08 的阶段计时确认，再检查 _prepare_inputs 中的数组构造、拷贝与同步。优化方案必须保持请求映射、位置与有效长度完全相同。你已经拥有本章的 NumPy 索引 oracle，可以在不运行大模型时验证翻译层，再到短时 GPU 实验检验实际收益。

学习路线因此是定义契约、手算、参考程序、真实源码、目标测试、小补丁、性能验证。每一步产物都能被下一步复用，而不是读完一篇架构介绍又从零猜测。遇到新框架或新模型时，类名会变化，这套围绕身份、状态、资源和证据的阅读方法仍然适用。
掌握门槛：你能够从固定请求计划一路追到实际输入索引、模型调用、采样状态、结果回写与释放；能解释 cached、computed、in_flight、None 在各自接口的具体含义；能区分源码核查、静态检查、上游测试和 GPU 实测；能提出一个有最小反例和验收条件的小补丁。下一章进入 [S10 分布式并行与通信](10-parallelism.md)，把这条数据路径扩展到多个 rank。
