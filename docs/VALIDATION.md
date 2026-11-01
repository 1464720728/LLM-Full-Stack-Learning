# 第一版验证报告

验证时间（UTC）：2026-11-01T07:28:13.924246+00:00。课程版本见 VERSION。

## 实际执行环境

Python 3.12.14，NumPy 2.3.5，Linux x86_64。本次验证进程隐藏 CUDA 设备，只执行 CPU 路径。

可选依赖安装状态：torch=未安装，triton=未安装，faiss=未安装。

## 实际结果

- 自动测试：374 项，失败 0，错误 0，跳过 1。
- 注册实验去重命令：CPU／离线通过 203 条，失败 0 条，依赖 GPU／联网或缺少可选依赖而未执行 6 条。
- 课程路径／先修与本地 Markdown 链接／Python 语法问题：0 项。
- 教材规模：6 卷，113 个有稳定 ID 的单元，209 个注册实验入口。

测试覆盖机制包括状态机与恢复、流式分片、权限与文档版本、引用失效、模型等价性、梯度与数值稳定、分页与 COW、取消与抢占、布局与在线归一化，以及跨卷集成。测试数量不等于覆盖了全部生产故障。

## 没有宣称执行的部分

Win11 原生／WSL2、本地 RTX 5070 Ti、H100、CUDA／Triton、NCCL 多卡以及真实模型 API 均未在本次构建环境执行。对应教学实现与操作步骤已提供；Python 语法检查不替代 GPU 编译或运行。真实硬件的误差与性能需要在指定环境按章节验收。

GitHub Actions 配置存在不代表其远端运行成功；远端状态以仓库 Actions 页面为准。本报告是本地执行证据。

## 复现

```bash
python -m pip install -r requirements-cpu.txt
python tools/verify_release.py
```

此命令会更新本报告与 validation.json，并执行仓库内声明为离线 CPU 的实验。不会自动运行 GPU／联网实验。部分 unittest 内部还验证多个输入或多个微实验，数字按 unittest 测试方法计。

## 注册实验执行明细

| 实验 | 状态 | 命令 |
|---|---|---|
| F01-L1 | passed | `python -m labs.foundations.environment` |
| F02-L1 | passed | `python -m labs.foundations.paths` |
| F03-L1 | passed | `python -m labs.foundations.text_pipeline` |
| F04-L1 | passed | `python -m labs.foundations.permissions` |
| F04-L2 | passed | `python -m labs.foundations.process_runner` |
| F05-L1 | passed | `python -m labs.foundations.http_message` |
| F05-L2 | passed | `python -m labs.foundations.json_contract` |
| F06-L1 | passed | `python -m labs.foundations.python_values` |
| F06-L2 | passed | `python -m labs.foundations.functions` |
| F07-L1 | passed | `python -m labs.foundations.objects` |
| F07-L2 | passed | `python -m labs.foundations.iterators` |
| F08-L1 | passed | `python -m labs.foundations.file_io` |
| F08-L2 | passed | `python -m labs.foundations.sqlite_lab` |
| F09-L1 | passed | `python -m labs.foundations.debugging` |
| F10-L1 | passed | `python -m labs.foundations.cli` |
| F11-L1 | passed | `python -m labs.foundations.concurrency` |
| F12-L1 | passed | `python -m labs.foundations.git_model` |
| F13-L1 | passed | `python -m labs.foundations.container_model` |
| F14-L1 | passed | `python -m labs.foundations.tensors` |
| F14-L2 | passed | `python -m labs.foundations.numerics` |
| F15-L1 | passed | `python -m labs.foundations.log_project` |
| F16-L1 | passed | `python -m labs.foundations.task_service` |
| F17-L1 | passed | `python -m labs.foundations.workshop.step01_loop` |
| F17-L2 | passed | `python -m labs.foundations.workshop.step02_function` |
| F17-L3 | passed | `python -m labs.foundations.workshop.step03_json` |
| F17-L4 | passed | `python -m labs.foundations.workshop.step04_pipeline` |
| F17-L5 | passed | `python -m labs.foundations.workshop.step05_cli` |
| F18-L1 | passed | `python -m labs.foundations.workshop.step06_memory_tasks` |
| F18-L2 | passed | `python -m labs.foundations.workshop.step07_persistence` |
| F18-L3 | passed | `python -m labs.foundations.workshop.step08_routing` |
| F21-L1 | passed | `python -m labs.foundations.linear_regression` |
| A01-L1 | passed | `python -m labs.agents.experiments 01-loop` |
| A02-L1 | passed | `python -m labs.agents.experiments 03-chat` |
| A02-L2 | passed | `python -m labs.agents.experiments 04-responses` |
| A02-L3 | passed | `python -m labs.agents.experiments 05-messages` |
| A03-L1 | passed | `python -m labs.agents.experiments 02-contract` |
| A03-L2 | passed | `python -m labs.agents.experiments 08-errors` |
| A04-L1 | passed | `python -m labs.agents.experiments 16-checkpoint` |
| A04-L2 | passed | `python -m labs.agents.experiments 17-conflict` |
| A05-L1 | passed | `python -m labs.agents.experiments 06-sse` |
| A05-L2 | passed | `python -m labs.agents.experiments 07-deltas` |
| A06-L1 | passed | `python -m labs.agents.experiments 09-budget` |
| A06-L2 | passed | `python -m labs.agents.experiments 10-retry` |
| A06-L3 | passed | `python -m labs.agents.experiments 19-cancellation` |
| A07-L1 | passed | `python -m labs.agents.experiments 11-context` |
| A07-L2 | passed | `python -m labs.agents.experiments 12-provenance` |
| A08-L1 | passed | `python -m labs.agents.experiments 13-paths` |
| A08-L2 | passed | `python -m labs.agents.experiments 14-edit` |
| A09-L1 | passed | `python -m labs.agents.experiments 20-fanout` |
| A10-L1 | passed | `python -m labs.agents.experiments 15-idempotency` |
| A10-L2 | passed | `python -m labs.agents.experiments 18-uncertain` |
| A11-L1 | passed | `python -m labs.agents.experiments 21-approval` |
| A11-L2 | passed | `python -m labs.agents.experiments 25-ticket-project` |
| A12-L1 | passed | `python -m labs.agents.experiments 22-multi-agent` |
| A13-L1 | passed | `python -m labs.agents.experiments 23-metrics` |
| A13-L2 | passed | `python -m labs.agents.experiments 24-redaction` |
| A13-L3 | passed | `python -m labs.agents.evaluation` |
| A14-L1 | passed | `python -m labs.agents.experiments 26-coding-project` |
| A15-L1 | passed | `python -m labs.agents.experiments 27-tool-rpc` |
| A16-L1 | passed | `python -m unittest tests.test_agents_core -v` |
| A17-L1 | passed | `python -m labs.agents.advanced_solutions` |
| A18-L1 | passed | `python -m unittest tests.test_agents_advanced tests.test_agents_evaluation -v` |
| R01-L1 | passed | `python -m labs.knowledge.lab 19-rag` |
| R01-L2 | passed | `python -m labs.knowledge.lab 20-refusal` |
| R02-L1 | passed | `python -m labs.knowledge.lab 01-tokenize` |
| R02-L2 | passed | `python -m labs.knowledge.lab 02-chunks` |
| R02-L3 | passed | `python -m labs.knowledge.lab 03-identity` |
| R03-L1 | passed | `python -m labs.knowledge.lab 04-bm25` |
| R03-L2 | passed | `python -m labs.knowledge.lab 05-bm25-parameters` |
| R04-L1 | passed | `python -m labs.knowledge.lab 06-vectors` |
| R04-L2 | passed | `python -m labs.knowledge.lab 07-tfidf` |
| R04-L3 | passed | `python -m labs.knowledge.lab 08-ann` |
| R04-L4 | passed | `python -m labs.knowledge.neural_encoder` |
| R05-L1 | passed | `python -m labs.knowledge.lab 09-fusion` |
| R05-L2 | passed | `python -m labs.knowledge.lab 10-rerank` |
| R05-L3 | passed | `python -m labs.knowledge.lab 11-metrics` |
| R05-L4 | passed | `python -m labs.knowledge.lab 12-compare` |
| R05-L5 | passed | `python -m labs.knowledge.lab 27-diversity` |
| R06-L1 | passed | `python -m labs.knowledge.lab 17-packing` |
| R06-L2 | passed | `python -m labs.knowledge.lab 18-citation` |
| R07-L1 | passed | `python -m labs.knowledge.lab 21-multihop` |
| R08-L1 | passed | `python -m labs.knowledge.lab 13-acl` |
| R08-L2 | passed | `python -m labs.knowledge.lab 14-revision` |
| R08-L3 | passed | `python -m labs.knowledge.lab 15-idempotency` |
| R08-L4 | passed | `python -m labs.knowledge.lab 16-deletion` |
| R08-L5 | passed | `python -m labs.knowledge.lab 28-persistence` |
| R09-L1 | passed | `python -m labs.knowledge.lab 22-memory` |
| R09-L2 | passed | `python -m labs.knowledge.lab 23-ttl` |
| R09-L3 | passed | `python -m labs.knowledge.lab 24-forget` |
| R10-L1 | passed | `python -m labs.knowledge.lab 25-graph` |
| R11-L1 | passed | `python -m labs.knowledge.lab 26-graph-stale` |
| R13-L1 | passed | `python -m labs.knowledge.assistant ask "KV cache" --demo` |
| R13-L2 | passed | `python -m labs.knowledge.memory_assistant demo` |
| R15-L1 | passed | `python -m labs.knowledge.walkthrough --stage 1` |
| R15-L2 | passed | `python -m labs.knowledge.walkthrough --stage 2` |
| R15-L3 | passed | `python -m labs.knowledge.walkthrough --stage 3` |
| R15-L4 | passed | `python -m labs.knowledge.walkthrough --stage 4` |
| R15-L5 | passed | `python -m labs.knowledge.walkthrough --stage 5` |
| R15-L6 | passed | `python -m labs.knowledge.walkthrough --stage 6` |
| R15-L7 | passed | `python -m labs.knowledge.walkthrough --stage 7` |
| R15-L8 | passed | `python -m labs.knowledge.walkthrough --stage 8` |
| R16-L1 | passed | `python -m labs.knowledge.embedding_api` |
| R16-L2 | passed | `python -m labs.knowledge.faiss_lab` |
| R16-L3 | not_executed | `python -m labs.knowledge.embedding_api --live --batch-size 2 --max-requests 2` |
| M01-L1 | passed | `python -m labs.models.run_lab tokenization` |
| M01-L2 | passed | `python -m labs.models.run_lab tensor_shapes` |
| M02-L1 | passed | `python -m labs.models.run_lab autograd` |
| M02-L2 | passed | `python -m labs.models.run_lab softmax` |
| M02-L3 | passed | `python -m labs.models.gradient_trace` |
| M03-L1 | passed | `python -m labs.models.run_lab attention` |
| M04-L1 | passed | `python -m labs.models.run_lab transformer` |
| M05-L1 | passed | `python -m labs.models.run_lab decoder` |
| M05-L2 | passed | `python -m labs.models.run_lab rope` |
| M06-L1 | passed | `python -m labs.models.run_lab kv_cache` |
| M06-L2 | passed | `python -m labs.models.run_lab gqa` |
| M07-L1 | passed | `python -m labs.models.run_lab mla` |
| M08-L1 | passed | `python -m labs.models.run_lab moe` |
| M09-L1 | passed | `python -m labs.models.run_lab sparse` |
| M09-L2 | passed | `python -m labs.models.run_lab compressed` |
| M10-L1 | passed | `python -m labs.models.run_lab linear` |
| M10-L2 | passed | `python -m labs.models.run_lab delta` |
| M11-L1 | passed | `python -m labs.models.run_lab ssm` |
| M11-L2 | passed | `python -m labs.models.run_lab selective_ssm` |
| M12-L1 | passed | `python -m labs.models.run_lab attnres` |
| M12-L2 | passed | `python -m labs.models.run_lab block_attnres` |
| M12-L3 | passed | `python -m labs.models.run_lab hyperconnections` |
| M12-L4 | passed | `python -m labs.models.run_lab ced` |
| M12-L5 | passed | `python -m labs.models.run_lab conditional_memory` |
| M12-L6 | passed | `python -m labs.models.block_attnres_decoder` |
| M13-L1 | passed | `python -m labs.models.run_lab quantize` |
| M14-L1 | passed | `python -m labs.models.run_lab sampling` |
| M14-L2 | passed | `python -m labs.models.run_lab loss_mask` |
| M14-L3 | passed | `python -m labs.models.run_lab dpo` |
| M14-L4 | passed | `python -m labs.models.run_lab grpo` |
| M14-L5 | passed | `python -m labs.models.run_lab lora` |
| M14-L6 | passed | `python -m labs.models.run_lab multimodal` |
| M14-L7 | passed | `python -m labs.models.run_lab crossmodal` |
| M15-L1 | passed | `python -m labs.models.tiny_lm` |
| M15-L2 | not_executed | `python -m labs.models.torch_transformer --device cpu --steps 40` |
| M16-L1 | passed | `python -m labs.models.compare` |
| M17-L1 | passed | `python -m labs.models.trace_decoder` |
| M17-L2 | passed | `python -m labs.models.trace_decoder --active-ffn` |
| M17-L3 | passed | `python -m labs.models.moe_decoder` |
| S01-L1 | passed | `python -m labs.serving.experiments --lab lifecycle` |
| S02-L1 | passed | `python -m labs.serving.experiments --lab token_budget` |
| S02-L2 | passed | `python -m labs.serving.experiments --lab chunks` |
| S02-L3 | passed | `python -m labs.serving.experiments --lab continuous` |
| S02-L4 | passed | `python -m labs.serving.experiments --lab fairness` |
| S03-L1 | passed | `python -m labs.serving.experiments --lab cow` |
| S03-L2 | passed | `python -m labs.serving.experiments --lab allocation_atomic` |
| S04-L1 | passed | `python -m labs.serving.experiments --lab prefix` |
| S04-L2 | passed | `python -m labs.serving.experiments --lab namespace` |
| S04-L3 | passed | `python -m labs.serving.experiments --lab cache_eviction` |
| S05-L1 | passed | `python -m labs.serving.experiments --lab cancel` |
| S05-L2 | passed | `python -m labs.serving.experiments --lab preemption` |
| S05-L3 | passed | `python -m labs.serving.experiments --lab fault_injection` |
| S06-L1 | passed | `python -m labs.serving.experiments --lab runner_swap` |
| S06-L2 | passed | `python -m labs.serving.experiments --lab batch_equivalence` |
| S07-L1 | passed | `python -m labs.serving.experiments --lab sampling` |
| S07-L2 | passed | `python -m labs.serving.experiments --lab stream_decoder` |
| S08-L1 | passed | `python -m labs.serving.experiments --lab metrics` |
| S09-L1 | passed | `python -m labs.serving.source_walk --project vllm --show-pin` |
| S10-L1 | passed | `python -m labs.serving.experiments --lab communication` |
| S11-L1 | passed | `python -m labs.serving.experiments --lab pd_transfer` |
| S12-L1 | passed | `python -m labs.serving.experiments --lab deterministic` |
| S12-L2 | passed | `python -m labs.serving.experiments --lab quantization` |
| S13-L1 | passed | `python -m labs.serving.experiments --lab memory_capacity` |
| S16-L1 | passed | `python -m labs.serving.experiments --lab trace_replay` |
| S16-L2 | passed | `python -m labs.serving.replay` |
| S18-L1 | passed | `python -m labs.serving.workbook_trace --case prefix` |
| S18-L2 | passed | `python -m labs.serving.workbook_trace --case cow` |
| S18-L3 | passed | `python -m labs.serving.workbook_trace --case preempt` |
| S19-L1 | passed | `python -m labs.serving.paged_attention` |
| K01-L1 | passed | `python -m labs.kernels.run --lab 1` |
| K01-L2 | passed | `python -m labs.kernels.run --lab 2` |
| K02-L1 | passed | `python -m labs.kernels.inspect_environment` |
| K03-L1 | passed | `python -m labs.kernels.run --lab 3` |
| K04-L1 | passed | `python -m labs.kernels.run --lab 4` |
| K04-L2 | passed | `python -m labs.kernels.run --lab 5` |
| K05-L1 | passed | `python -m labs.kernels.run --lab 6` |
| K05-L2 | passed | `python -m labs.kernels.run --lab 7` |
| K05-L3 | passed | `python -m labs.kernels.run --lab 8` |
| K05-L4 | passed | `python -m labs.kernels.run --lab 9` |
| K05-L5 | passed | `python -m labs.kernels.run --lab 10` |
| K05-L6 | passed | `python -m labs.kernels.run --lab 16` |
| K06-L1 | passed | `python -m labs.kernels.run --lab 11` |
| K06-L2 | passed | `python -m labs.kernels.run --lab 12` |
| K07-L1 | passed | `python -m labs.kernels.run --lab 13` |
| K07-L2 | passed | `python -m labs.kernels.run --lab 14` |
| K07-L3 | passed | `python -m labs.kernels.run --lab 26` |
| K07-L4 | passed | `python -m labs.kernels.run --lab 27` |
| K08-L1 | passed | `python -m labs.kernels.run --lab 15` |
| K09-G1 | not_executed | `python -m labs.kernels.gpu_check --run` |
| K10-L1 | not_executed | `python -m labs.kernels.custom_autograd` |
| K11-L1 | passed | `python -m labs.kernels.run --lab 17` |
| K11-L2 | passed | `python -m labs.kernels.run --lab 18` |
| K11-L3 | passed | `python -m labs.kernels.run --lab 19` |
| K11-L4 | passed | `python -m labs.kernels.run --lab 20` |
| K12-L1 | passed | `python -m labs.kernels.run --lab 21` |
| K13-L1 | passed | `python -m labs.kernels.run --lab 22` |
| K14-L1 | passed | `python -m labs.kernels.run --lab 23` |
| K14-G1 | not_executed | `python -m labs.kernels.benchmark --run --op matmul --size 256 --repeats 20` |
| K16-G1 | not_executed | `python -m labs.kernels.inspect_environment --gpu` |
| K18-L1 | passed | `python -m labs.kernels.run --lab 24` |
| K18-L2 | passed | `python -m labs.kernels.run --lab 25` |
| K18-L3 | passed | `python -m labs.kernels.run --lab 28` |
| KW1-L1 | passed | `python -m labs.kernels.workbook softmax` |
| KW2-L1 | passed | `python -m labs.kernels.workbook gemm` |
| KW3-L1 | passed | `python -m labs.kernels.workbook attention` |
