# S17 源码工作坊：从一条真实请求计划到一个回归测试

本工作坊要求你同时打开本书、固定vLLM源码和一个笔记文件。它不是阅读清单。你将手算真实Scheduler的一轮计划，沿EngineCore进入Runner，解释状态为什么提前推进，随后把一个新测试放进上游测试体系，做正向与负向验证。GPU不是本工作坊的前提，但运行上游测试需要完整的开发依赖。

## 工作台0：把“读懂源码”变成可验收动作

读懂一个函数至少要能回答六个问题：调用者提供了什么；哪些数据是借用、哪些会被修改；返回结构保证什么；哪一步可能失败或等待；成功路径之外资源如何回收；它依赖哪些未写在类型里的前提。仅能解释函数名，不算读懂。

为每个函数建立四栏笔记：输入值、关键分支、状态变化、后置条件。输入值不要只写类型；本次具体使用三请求A/B/C，提示各800token，每轮预算1024，长prefill阈值400，前缀缓存关闭，无推测解码，无多模态，无KV connector。每个不相关分支都写“本次条件关闭”，而不是假装代码不存在。

本案例来自固定快照中已经存在的`test_schedule_concurrent_partial_requests`所覆盖的场景；我们写的新测试加入另一个预算边界和计数守恒检查，不复制上游整个测试。源码位置：`tests/v1/core/test_scheduler.py`约1553行。行号只对S09固定SHA成立。

先运行课程自己的参考引擎，建立词义：

```bash
python -m labs.serving.experiments --lab token_budget
python -m labs.serving.source_walk --project vllm --show-pin
```

然后在独立checkout执行source_walk。若SHA不同，先停在“定位迁移”阶段，不把本书行号当当前事实。

## 工作台1：从真实构造助手了解测试配置

打开上游`tests/v1/core/utils.py`。`create_scheduler`并不加载一个模型进行前向，它构造ModelConfig、SchedulerConfig、CacheConfig、ParallelConfig和KVCacheConfig，然后创建Scheduler。这里仍可能读取模型配置，因此“没有GPU前向”不等于“完全没有外部依赖”。

逐项分析参数：`max_num_batched_tokens`控制本轮预算；`long_prefill_token_threshold`限制长提示分块；`enable_prefix_caching=False`排除命中影响；`max_model_len=2048`使800token提示合法；`num_blocks`提供充分页面，让这一题先聚焦预算，而不是同时混入内存压力。`watermark=0`在该helper里帮助固定准入/抢占行为。

helper构造的FullAttentionSpec使用很小的头与维度，因为这里测试管理逻辑，不需要真实模型K/V。你应该把它理解为“给管理器一个合法结构”，不是把facebook/opt模型真的改成一个头。MockKV等辅助对象同样是替身，而不是生产传输器。

阅读`create_requests`时，确认request_id如何生成、token序列长度如何指定、sampling_params怎样设置以及block hasher何时初始化。测试不应该手工假设请求ID永远是某个固定字符串；使用返回对象的`request_id`更稳妥。我们的测试主要断言values和包含关系，避免把无关标识写死。

这个工作台的产物是一张配置表：每个参数为什么存在、排除了什么变量、是否需要模型权重。若你不能解释一个参数，就不要把它当“网上推荐配置”原样带进性能实验。

## 工作台2：第一次schedule前的状态

调用add_request后，A/B/C在waiting中，running为空，requests字典能按ID找到三者。它们还没完成任何输入位置。没有prefix hit，所以本轮每个请求从0开始。请暂停在schedule入口，写下：token_budget=1024、每请求known=800、computed=0。

沿等待请求路径，候选A希望追赶800，但受到长prefill阈值400和总预算1024约束，先安排400。申请页成功后，计划记录A→400，预算减到624，状态转RUNNING。B同理安排400，预算变224。C最多取剩余224。结果为`[400,400,224]`，总计1024。

现在把budget改为512，其余条件相同。A安排400，B只拿112，C这轮不被安排。请注意这个结果不是`[171,171,170]`：该快照默认路径不是自动按三个请求平均分配预算。若开启另一种公平/自适应策略，结果可能不同，测试必须锁定相应配置。

申请缓存发生在预算提交之前非常重要。若失败，应进入特定抢占或等待路径，而不是消耗不存在的计算。去找运行请求路径约1001行和等待请求路径约1569行的预算减少位置，比较它们前面分别有哪些成功条件。两个位置都减预算，但它们服务的队列身份与状态更新不同。

## 工作台3：一个容易误读的名字——num_computed_tokens

本卷简化Engine在追加元数据时推进computed；没有真实异步GPU，所以该数容易被直觉理解为已经算完。真实vLLM的计数更精细：该快照在`_update_after_schedule`中推进`num_computed_tokens`，并增加`num_in_flight_tokens`，此时GPU结果可能尚未返回。

这是一种调度账本：计划已经预定这些位置，下一个调度步需要避免再次安排同一位置。真正完成与可能的speculative拒绝，会在输出回写时调整。你不能看见`num_computed_tokens=400`就直接推断GPU已经完成了400位置的所有数据写入。

源码注释给出关键理由：SchedulerOutput必须带原始位置供Runner确定输入；之后推进请求对象，才能允许紧接着的调度继续处理下一个chunk；如果后续某些token被拒绝，update_from_output会修正。这体现计划状态、在途状态、已确认结果三者的区分。

拿A做一个时间表：调度前computed0；计划保存start0/count400；请求账本推进到400，in_flight增加400；Runner根据计划而不是已经推进后的请求对象准备输入0到399；结果回写后对应在途工作被确认。若Runner错误读取新的400作为start，本轮会跳过前400token。这就是为什么“同一字段到处传引用”可能出错。

本工作台验收：你应在笔记里写出“哪个对象保存旧值、哪个对象已更新、谁负责纠正失败”。如果只说“computed就是已计算”，说明仍然停留在高层概念，没有进入并发实现语义。

## 工作台4：SchedulerOutput是一份可检查的合同

打开`vllm/v1/core/sched/output.py`中的SchedulerOutput。它是dataclass，包含每请求调度token数与总调度token数，以及新请求、缓存请求、结束ID、编码器/connector等元数据。字典总和与总计数应一致，这是我们新增测试检查的基本守恒。

为什么同时保留总数和字典？总数可快速用于缓冲区大小、执行是否为空和某些性能路径；字典用于每请求映射。冗余信息提高效率，也带来一致性风险。因此冗余字段值得建立测试，而不是认为“其中一个反正可以推出来”就不检查。

`scheduled_new_reqs`与`scheduled_cached_reqs`的“cached”不要简单等同于prefix cache hit。它也用于已有请求在Worker侧的增量状态更新，具体语义要读类型和构造函数。一个请求第二轮进入cached request结构，可能只是因为Worker已经知道它，而不是命中了跨请求前缀。

这类命名误解很常见。学习者看到cached就联想到显存缓存，看到block就联想到CUDA thread block，看到worker就联想到一个线程。解决办法是写出本类型属于哪一层、生命周期有多长、谁创建谁消费，而不是仅翻译英文。

## 工作台5：沿EngineCore执行边界

回到EngineCore.step。它把SchedulerOutput交给model_executor.execute_model，采用非阻塞接口，返回future。随后准备grammar输出，再等待future。对本次无结构化输出案例，grammar路径可以被视为关闭，但接口仍保留。

model_output如果为None，EngineCore调用sample_tokens。这表明某些执行路径只先完成模型计算，把采样留到后续阶段。为什么这样做？设计上可以让计算、grammar准备、异步调度或其他输出处理分离，具体动机应结合该版本调用者与状态对象分析，不能假定每条路径完全相同。

在GPUModelRunner.execute_model开头，若先前的execute_model_state尚未正确消耗，会抛状态错误，要求按契约完成sample_tokens。这是代码级的顺序约束：不是任意连续调用execute_model都合法。把这个检查与我们Backend简单一次调用的接口比较，你会看到生产实现为何需要更复杂状态。

本次案例的prefill还没到提示末尾，所以mock ModelRunnerOutput返回每请求空的sampled_token_ids。空列表不是失败，也不是模型拒答，它表示本轮只推进部分输入，还没有可发出的新token。若测试在每个prefill chunk都伪造一个输出，就会掩盖或制造错误。

结果回写前EngineCore处理abort队列。然后update_from_output结合计划与真实结果更新请求。请列出此函数中哪些字段来自计划、哪些来自模型输出，以及请求可能已经取消或被抢占时如何处理。这一题需要你读具体分支，不能用“结果回来就追加”概括所有情况。

## 工作台6：追踪第二轮与第三轮

第一轮A/B各完成计划400，C计划224；没有新输出。第二轮，如果条件不变且没有新请求或资源变化，A/B各还有400提示位置，C还有576；仍可能得到400、400、224。A/B这轮到达提示末尾，可产生第一输出；C仍然只完成448提示位置。

第三轮A/B通常各需要处理一个刚生成token，C需要继续prefill。你要用实际源码计算该轮预算如何分配，尤其注意长prefill阈值和运行队列顺序。不要只写“decode优先所以一定是1、1、1022”，因为C自己的chunk限制可能仍约束它；剩余预算也不一定全部消耗。

上游已有测试正是通过ModelRunnerOutput控制哪些请求产生输出，从而验证这个变化。我们的新测试聚焦第一轮边界，后续拓展任务要求你增加第二、三轮，并用真实helper输出结构回写。这样从一个简单、可验证的断言逐步扩展，比一开始构造整个巨大模型更有效。

每轮笔记至少包含：requests状态、running顺序、waiting顺序、计划字典、总token数、computed账本、输出列表、空闲页数。缺少其中一类时，很容易在后续取消题中不知道哪个数据过时。

## 工作台7：把新测试接入真实上游

课程提供[`test_course_scheduler_contract.py`](../../labs/serving/upstream_tests/test_course_scheduler_contract.py)，不是独立假调度器测试。它导入真实Scheduler、ModelRunnerOutput、RequestStatus与上游create_scheduler/create_requests。它包含两个预算参数化用例和一个运行请求取消幂等用例。

先只检查复制目标：

```bash
python -m labs.serving.install_upstream_test --root upstream/vllm
```

确认是自己的学习checkout，再执行：

```bash
python -m labs.serving.install_upstream_test --root upstream/vllm --apply
```

辅助脚本要求HEAD等于固定SHA，拒绝覆盖现有同名文件，只增加一个测试文件，不修改生产实现。它不安装依赖、不下载模型、不执行测试。在上游开发环境准备完成后，从该源码根运行：

```bash
cd upstream/vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -q
```

这条命令强制模型资料走本地缓存。如果所需OPT配置未准备，测试应报缺资料；先在允许联网的准备阶段缓存官方小模型config/tokenizer，或把`COURSE_VLLM_TEST_MODEL`设为已核验本地兼容配置目录，再重试。测试不需要下载权重做GPU前向，但依赖导入可能仍需正确安装的vLLM/PyTorch开发环境。

本仓库发布时只对这份文件做Python语法检查，没有实际安装上游依赖运行它，因此状态是“源码接口核对、上游执行待本地环境验证”。这个标签应保留到你亲自跑出结果。不要把本课程离线单元测试通过误写成上游测试也通过。

## 工作台8：逐段解释新测试

`empty_result`根据计划中的请求ID建立req_id_to_index，并为每个请求返回空sampled_token_ids。这避免把列表位置和请求ID混淆。为什么不返回一个全局空列表？因为结果结构需要与已调度请求数量对应，不能丢掉每请求槽位。

预算测试先创建充分页面，锁定阈值400与上下文2048，加入3个800token请求，再调用schedule。它同时检查各请求数量、总量等于budget、total字段等于字典总和、单请求不超过400。四类断言分别抓策略分配变化、预算空耗或超用、冗余字段不一致、chunk上限失效。

随后用空结果回写，再把请求标为FINISHED_ABORTED并确认未完成计数归零。清理不是为了让pytest“好看”，而是测试自己也应遵守生命周期；尤其当helper或后续fixture复用全局资源时，未清理会制造顺序依赖。

取消测试先记录free_before，加入128token请求，用64budget只计算部分提示，确认占用了页面；回写空结果后第一次abort应返回一个有效请求，第二次返回空；空闲页应恢复到原值；requests中没有该ID；下一次schedule也不再包含它。

为什么关闭prefix cache？因为开启后，合法缓存保留可能让free_after不等于free_before。测试若忽略缓存所有权，会把正确的复用行为误报为泄漏。想测试开启缓存的取消，需要写更细的断言：活动引用归零，但缓存引用是否保留按契约检查。

## 工作台9：负向控制证明测试会抓错

这份测试带一个显式环境变量`COURSE_MUTATE_SCHEDULER=1`。开启时，pytest monkeypatch只在本次测试进程里包装Scheduler.schedule，让返回的total_num_scheduled_tokens多1，而每请求字典不变。它不会改磁盘源码。

```bash
COURSE_MUTATE_SCHEDULER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/v1/core/test_course_scheduler_contract.py -k partial_prefill -q
```

预期这次失败，因为冗余总数不等于真实计划总和。关闭环境变量后应恢复通过。一个测试在正确版本和故意错误版本上都通过，就没有证明它能检测所关注问题。负向控制不是为了追求失败数量，而是验证测试敏感性。

第二个变异任务由你完成：仅在测试进程把第二次finish_requests错误地重复返回同一请求，观察幂等断言是否失败。不要修改生产函数来污染其他课程任务。之后再设计一个资源未释放的变异，验证free页断言能抓住它。

如果测试只因import失败或模型config缺失而红，并不算成功抓到调度bug。必须看到失败落在预期断言上，并保存相关traceback。测试环境错误、测试逻辑错误和被测实现错误要分开记录。

## 工作台10：从测试进入自己的小补丁

先创建分支并确认`git status`只包含自己的测试。选一个范围小、可解释的变动，例如新增内部计数校验用于debug模式，或给某个取消边界补测试。不要第一天就重写整个Scheduler。

设计说明先写触发条件：什么请求集合、在哪一轮、原来会出现什么不一致、修改后如何保证。列出可能受影响的路径：同步调度、异步调度、prefix命中、抢占、connector。你不必立刻支持所有模式，但不能让未覆盖路径悄悄进入错误保证。

执行相关上游测试后，再跑课程参考引擎的同构用例。二者都通过仍然不代表真实GPU性能无回归；若修改涉及热路径，再进行短时性能验证。纯测试增补则可以明确“不改变运行时行为”，把重点放在测试是否稳定、是否依赖网络、是否假设过多内部细节。

准备代码评审时解释问题与行为，不按写代码的时间顺序讲故事。说明已验证环境、未执行GPU部分，以及为什么选择这个断言层级。不要把“我学习时觉得这样更清楚”当成上游接受更改的唯一理由；真实项目需要维护价值。

## 工作台11：对照SGLang而不做机械翻译

回到SGLang normal loop，找出ingest、plan、run、process对应职责。然后进入overlap，标出result_queue中的每个元素带着哪个batch对象。比较vLLM计划账本与SGLang批次状态，列出相同的不变量和不同的实现边界。

例如两者都必须避免取消后再正常发出不该发的结果，都必须保证KV在计算使用期间不被错误复用；但队列类型、forward batch结构、缓存索引方式不同。你应把“所有权与时间顺序”迁移过去，而不是把vLLM变量名替换成SGLang名字。

最后选SGLang RadixCache的一个操作，用纸画节点、token范围、锁引用与device indices。比较本课程字典式最长前缀的时间复杂度，解释Radix分裂为什么只改变索引结构而不必复制所有KV。这个练习把数据结构课程、运行时和框架源码真正接在一起。

## 交付物与答案核对

完成后提交五项：一张三请求三轮状态表；一条固定源码调用路径；新增上游测试文件；正向/负向测试记录；一篇不超过两页的补丁设计说明。表中的具体数字必须来自手算与实际断言一致，而不是从日志复制后不理解。

常见错误答案包括：把computed账本当硬件完成事件；把cached request误读为prefix hit；把CPU标记当无依赖；把import失败当测试成功抓bug；把mock结果当真实模型输出；把main链接当永久证据。逐条排除后，你已经具备进行更复杂框架修改的工作方式。

下一工作坊：[S18 页表与抢占的逐状态推演](18-state-workbook.md)。
