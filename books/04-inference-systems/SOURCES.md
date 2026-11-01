# 本卷来源、版本与证据边界

核查日：2026-11-01。上游源码通过仓库接口逐文件阅读；行号仅适用于下面固定commit。教材给出的解释与教学实现是本课程原创，框架源码本身继续适用各自上游许可证。这里没有把上游大型源码复制进本仓库。

## 固定源码

| 项目 | 固定commit | 用途 |
|---|---|---|
| vLLM | `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211` | EngineCore/Scheduler/KV/Runner与上游测试接口 |
| SGLang | `fa5580f6d0e969a3d44c603d215e2139886e8db2` | 调度事件循环、RadixCache、ModelRunner与确定性文档 |
| EvalScope | `8df4fd2b9d160495d01def86ab88b222dce6601d` | Terminal-Bench adapter调用和配置边界 |
| Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 单卡实验模型身份与结构配置 |

“固定源码”表示读到了相应版本的代码，不表示在本环境成功安装或执行了该项目。课程的上游pytest工作坊仍标为未执行；本课程CPU测试通过不能代替框架兼容性测试。

## 来源表

| ID | 原始资料 | 支持范围 |
|---|---|---|
| S-SRC-01 | [vLLM EngineCore](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/engine/core.py) | EngineCore.step: schedule, execute, sample, abort queue, update; S09/S17 |
| S-SRC-02 | [vLLM Scheduler](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/sched/scheduler.py) | token budget, running/waiting, computed/in-flight bookkeeping and output update; S02/S05/S09/S17 |
| S-SRC-03 | [vLLM KVCacheManager](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/kv_cache_manager.py) | computed prefix blocks, allocate_slots, free; last prompt token recomputation; S03/S04/S09 |
| S-SRC-04 | [vLLM GPUModelRunner](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/worker/gpu_model_runner.py) | execute_model and sample_tokens phase protocol; no assertion of every model/backend support |
| S-SRC-05 | [vLLM scheduler tests](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/tests/v1/core/test_scheduler.py) | partial prefill 400/400/224, finish/cancel, in-flight preemption tests; course test API basis |
| S-SRC-06 | [vLLM scheduler test helpers](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/tests/v1/core/utils.py) | create_scheduler and create_requests signatures/configuration; optional upstream test imports |
| S-SRC-07 | [vLLM SchedulerOutput](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/v1/core/sched/output.py) | total_num_scheduled_tokens field and per-request work contracts |
| S-SRC-08 | [SGLang Scheduler](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/managers/scheduler.py) | normal/overlap event loops; get_next_batch_to_run, run_batch, process_batch_result |
| S-SRC-09 | [SGLang RadixCache](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/mem_cache/radix_cache.py) | RadixKey/TreeNode, namespace, prefix matching, insert, eviction; not equivalent to course exact-prefix map |
| S-SRC-10 | [SGLang ModelRunner](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/model_executor/model_runner.py) | ModelRunner.forward source navigation |
| S-SRC-11 | [SGLang deterministic inference](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/docs/docs/advanced_features/deterministic_inference.mdx) | snapshot-specific flag and backend/radix constraints; no universal reproducibility guarantee |
| S-SRC-12 | [EvalScope Terminal-Bench adapter](https://github.com/modelscope/evalscope/blob/8df4fd2b9d160495d01def86ab88b222dce6601d/evalscope/benchmarks/terminal_bench/terminal_bench_adapter.py) | Agent/harness/model configuration boundaries, separate agent/verifier timeout parameters and Docker CLI requirement; no Harbor version pin claimed |
| S-SRC-13 | [vLLM reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/) | Official rolling documentation: seed alone does not guarantee online reproducibility; environment and scheduling constraints apply |
| S-SRC-14 | [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) | Official rolling documentation on batch-invariant mechanisms and restrictions; do not assume applicable to arbitrary installed version |
| S-SRC-15 | [PagedAttention paper](https://arxiv.org/abs/2309.06180) | Virtual-memory-inspired KV block management and sharing; no paper benchmark numbers transferred to course simulator |
| S-SRC-16 | [Sarathi-Serve paper](https://arxiv.org/abs/2403.02310) | Chunked prefill and scheduling research context; experiment results apply to paper configuration |
| S-SRC-17 | [DistServe paper](https://arxiv.org/abs/2401.09670) | Prefill/decode disaggregation research context; course transfer model is an explicitly simplified analytical model |
| S-SRC-18 | [Qwen2.5-0.5B-Instruct pinned config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json) | Concrete single-GPU study model identity: Qwen2ForCausalLM, 24 layers, hidden 896, 14 Q heads, 2 KV heads; not an assertion that any chosen image supports this model |

## 怎样复核，而不是只收集链接

S09先从EngineCore.step出发，把调度、执行、采样、取消、回写逐项标到源码。S17再围绕800-token提示和1024预算，读真实测试和helper，计算400/400/224的分配，并在独立checkout运行课程回归测试。S18用自己的可运行引擎验证共享引用与重算状态，S19把数值K/V接上独立的分页attention。每一层的证据范围不同，不能相互冒充。

vLLM reproducibility和batch invariance链接是滚动文档，`commit=null`是有意保留的缺口。使用其他框架版本时，应记录那一版官方文档、实际backend与硬件；不能把2026-11-01读到的支持条件套到任意旧版本。SGLang确定性文档则链接到固定commit，但也没有替代实际模型/硬件验证。

EvalScope/Harbor内容用于训练评测调用链的阅读能力。已固定并核查的是EvalScope adapter；本卷没有伪造Harbor commit，也没有声称执行过Terminal-Bench任务。独立CLI agent可能有自己的模型与密钥配置，adapter接收一个model参数并不保证所有harness使用它。

论文用来解释设计动机与比较方法。本卷所有毫秒曲线、带宽代入与容量估算都清楚标记为模拟或解析值，没有把论文中的性能结果移植成你的H100结果。真实性能只接受S15实验卡所记录的实测。

机器可读版本见[sources.json](sources.json)。验证状态见[VALIDATION.md](VALIDATION.md)。
