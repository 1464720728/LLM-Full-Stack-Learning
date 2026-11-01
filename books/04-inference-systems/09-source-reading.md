# S09 从教学引擎进入 vLLM 与 SGLang 源码

先修 S01–S08。现在你已经能解释调度、资源和输出的基本闭环。本章把同一组问题带到真实框架，要求你沿着实际数据结构走，不再停留在架构图。源码引用核查于 2026-11-01；固定 SHA 是阅读坐标，不表示推荐把当天主分支直接部署到公司服务。

## 1. 固定一个可以回到原处的版本

本卷实际读取过以下快照：

| 项目 | 固定提交 |
|---|---|
| vLLM | `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` |
| SGLang | `fa5580f6d0e969a3d44c603d215e2139886e8db2` |
| EvalScope 边界核查 | `8df4fd2b9d160495d01def86ab88b222dce6601d` |

你可以只在浏览器读固定链接，也可以在个人学习目录准备独立副本。以下命令只操作新建的 upstream 目录，不改已有部署：

```bash
mkdir -p upstream
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git upstream/vllm
git -C upstream/vllm checkout --detach e0d00e4b8eb0c89ec30921d0695cfdd409e9c211
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git upstream/sglang
git -C upstream/sglang checkout --detach fa5580f6d0e969a3d44c603d215e2139886e8db2
python -m labs.serving.source_walk --project vllm --root upstream/vllm
python -m labs.serving.source_walk --project sglang --root upstream/sglang
```

这需要下载源码，但不加载模型、不占 GPU。若学校/公司网络有代理，先在基础卷完成网络环境学习，不要把凭据写进仓库。`source_walk` 只读本地文件与 `git rev-parse`，不会 clone、安装或修改内容。它输出 actual 与 expected SHA；不同则明确标为非精确快照。

## 2. vLLM 的第一条闭环：EngineCore.step

打开固定链接：[EngineCore](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/engine/core.py#L685)。在该快照，`EngineCore.step` 从 685 行开始。逐条记录它调用什么，而不是急着读整个三千行文件。

核心次序是：判断是否有请求；让 scheduler 产生计划；向 executor 提交非阻塞 model execution；准备 grammar 输出；等待 model future；必要时单独执行 sampling；处理执行期间到来的 abort；把结果送回 scheduler；附加迭代与权重版本信息；返回结果。

这里有三个值得动手的问题。第一，非阻塞提交后为什么仍然调用 future.result？因为非阻塞描述提交接口，不代表结果可在尚未完成时读取。第二，为什么 model_output 为 None 时要调用 sample_tokens？因为模型计算与采样可以在架构中分阶段执行。第三，为什么 abort 在 update_from_output 之前处理？因为迟到取消应影响结果是否还可作为正常输出提交。

在纸上为每一步写“输入、输出、可能等待、所有者”。如果写不出来，就沿类型定义跳转；不要只追函数名。课程里的 Work 对应的是一种计划思想，不能声称它和完整 SchedulerOutput 一一同构。

## 3. Scheduler：从配置到计划

打开 [Scheduler](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/sched/scheduler.py#L698)。`schedule` 在此快照从 698 行开始，`update_from_output` 从 2252 行开始。

阅读顺序建议是：构造函数中保存哪些 config；schedule 开头建立哪些本轮容器；token budget 来自哪个字段；运行请求如何获得 token 数；等待请求何时进入；如何申请缓存；如何记录被抢占请求；最后如何打包输出。接着立刻读 update_from_output，找到完成判断、输出 token 接收和资源清理。只读 schedule 的前半段，会误以为调度器永远只增长不清理。

该版本的注释说明，它用已计算位置追赶已知位置的方式组织调度，能够覆盖 chunked prefill、缓存与推测相关情况。本卷 S02 的 gap 模型就是帮助你理解这一抽象，但我们的实现不含真实 speculative/encoder/connector 等分支。读到不熟悉的分支时标出触发条件，暂时绕过；回到基本路径后再逐一加入。

## 4. KV 管理器：成功、失败和释放都要读

打开 [KVCacheManager](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/kv_cache_manager.py)。已核实符号：`get_computed_blocks` 266 行、`allocate_slots` 379 行、`free` 628 行。

不要只看 allocate_slots 的名字就猜“它分配 GPU 内存”。追它返回的是页块结构还是实际张量；再找它调用的 coordinator 与 block pool。记录申请量是否包含已经命中的块、新计算位置、lookahead 或 connector 需求；资源不足时返回什么；caller 是否会抢占或跳过。

读 get_computed_blocks 时特别标出全提示命中仍需重算最后位置的限制，以及不同 cache group/状态类型如何影响可复用边界。读 free 时追引用是回到缓存可复用状态，还是物理设备内存释放。S03 的所有权账本此时应能帮助你理解复杂代码。

## 5. Runner：跟到真实模型调用

打开 [GPUModelRunner](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/worker/gpu_model_runner.py)。该快照 `execute_model` 从 3906 行、`sample_tokens` 从 4260 行开始。

先用 `rg` 找两个符号，再从入参向前追输入准备、缓存元数据、graph/eager 分支和 model 调用。对每个张量记录 shape、dtype、device、是否是预分配缓冲区、哪些位置有效。看到 `.cpu()`、`.item()`、等待 future 或 event 时，问它会不会引入同步；不要一律判断为错误，有时这是语义必须的边界。

源码阅读任务的产物应包含一张自己的数据表，例如 SchedulerOutput 的 token 计数怎样影响 input_ids 的有效长度，而不是摘录几百行代码。你需要能从一次具体请求推出数组内容，才算完成阅读。

## 6. SGLang 的另一种组织方式

打开 [Scheduler](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/managers/scheduler.py)。已核实 `event_loop_normal` 2090 行、`event_loop_overlap` 2126 行、`get_next_batch_to_run` 4166 行、`run_batch` 4970 行、`process_batch_result` 5453 行。

normal loop 易于入门：摄入请求，获得下一批计划，运行 batch，处理结果，更新 last_batch；空闲时执行相应检查。overlap loop 则持有结果队列，把上一批结果处理与当前批 GPU 计算交错。请先用两轮画出“当前 batch”“上一次 result”各自属于谁；如果只看变量名 batch，很容易把时间上的两个批次混淆。

然后读 [RadixCache](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/mem_cache/radix_cache.py)：`match_prefix` 369、`insert` 433、`evict` 557 行。最后进入 [ModelRunner](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/model_executor/model_runner.py#L1834) 的 `forward`。两框架的组件名称和边界不同，要比较职责和不变量，不应强行逐类对照。

## 7. 一个真正的源码修改练习

在独立源码副本建立自己的分支：

```bash
git -C upstream/vllm switch -c course/scheduler-observation
rg -n 'def schedule|num_scheduled_tokens|allocate_slots' upstream/vllm/vllm/v1/core/sched/scheduler.py
rg --files upstream/vllm/tests | rg 'scheduler|kv_cache'
```

第一次任务不要修改调度政策。先增加一个仅在测试开启的统计：每轮计划总 token 数是否等于各请求数之和。复用框架现有测试构造输入。验收是：不启用时零行为差异；启用时可定位违规迭代；既有目标测试通过。不要直接编辑 site-packages，因为那会丢失版本关系，也可能影响公司任务。

第二次任务选择一个明确小问题，例如为零输出或取消边界增加测试。先写会在故意破坏版本下失败的测试，再恢复代码证明通过。第三次才尝试策略修改，并用固定工作负载比较输出、资源账本和延迟。

## 8. 练习与解答

1. **五节点数据路径。** 从 EngineCore.step 列出 schedule→executor→future→abort→update 的数据与等待关系。答案必须区分 plan 与 result：前者描述要算什么，后者描述实际发生什么，两者不能随便互换。
2. **读失败返回。** 找 allocate_slots 的资源不足路径，再找 caller 如何处理。参考产物是一段自己写的控制流说明和精确代码链接；如果仅贴函数定义，没有回答失败语义，则未通过。
3. **追一个计数。** 选择 num_computed_tokens，从创建、命中、调度、回写、抢占五处追踪。答案应指出它不是 output length，也不是本轮 token count；在重算和 prefix hit 下区别尤其明显。
4. **对照两种循环。** 解释 SGLang overlap 与 normal 的区别。参考：不仅是多一个线程，而是前后批结果生命周期错开；需要保证对应 batch 的采样和缓存状态不串线。用两个批次 A/B 画结果队列变化。
5. **跨版本定位。** 把 source_walk 用在另一个本地版本。exact_snapshot=false 后先通过 rg 重新定位符号，再记录迁移表；不要修改 manifest 的 expected SHA 使警告消失。
6. **评测边界。** 读取固定 EvalScope [Terminal-Bench adapter](https://github.com/modelscope/evalscope/blob/8df4fd2b9d160495d01def86ab88b222dce6601d/evalscope/benchmarks/terminal_bench/terminal_bench_adapter.py)。核查 agent_name 路径和模型接口归属。该快照的配置说明把 terminus-2 与独立 CLI agents 区分开；不能推断所有 harness 都自动使用 EvalScope 的 model 参数。你要把参数最终送到哪一层写出来，再做一个单任务验证。

掌握门槛：能在固定版本中从请求计划走到模型调用，再走回输出与释放；能提交一个小型、有测试的源码改动；能说明每项结论适用的 commit 与执行模式。

下一章：[S10 分布式并行与通信](10-parallelism.md)。
