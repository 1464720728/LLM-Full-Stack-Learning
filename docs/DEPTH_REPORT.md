# 逐章深写对照

由 `python tools/check_depth.py --write-report` 生成。统计去掉 fenced code block 后的中文汉字，含标题、表格和习题答案。篇幅是审稿线索，不是质量证明；教学标准见 [DEPTH_STANDARD](DEPTH_STANDARD.md)。

初版快照：`0f6a672e50715a724da3e4d57c474ec66c627758`。目标版本：1.1.0。

统计 100 篇；当前合计 1,228,216 汉字；未达到本轮最低篇幅 0 篇。

| 章节 | 初版 | 当前 | 本轮下限 | 篇幅检查 |
|---|---:|---:|---:|---|
| [books/00-foundations/01-environment-and-learning.md](../books/00-foundations/01-environment-and-learning.md) | 2097 | 8477 | 8000 | 达到 |
| [books/00-foundations/02-files-paths-and-encoding.md](../books/00-foundations/02-files-paths-and-encoding.md) | 1873 | 12073 | 12000 | 达到 |
| [books/00-foundations/03-shell-pipes-and-text.md](../books/00-foundations/03-shell-pipes-and-text.md) | 1598 | 12026 | 12000 | 达到 |
| [books/00-foundations/04-processes-permissions-environment.md](../books/00-foundations/04-processes-permissions-environment.md) | 1781 | 12047 | 12000 | 达到 |
| [books/00-foundations/05-network-ssh-http-json.md](../books/00-foundations/05-network-ssh-http-json.md) | 1889 | 12029 | 12000 | 达到 |
| [books/00-foundations/06-python-values-control-functions.md](../books/00-foundations/06-python-values-control-functions.md) | 1864 | 12127 | 12000 | 达到 |
| [books/00-foundations/07-modules-objects-iterators.md](../books/00-foundations/07-modules-objects-iterators.md) | 1948 | 12071 | 12000 | 达到 |
| [books/00-foundations/08-files-exceptions-sqlite.md](../books/00-foundations/08-files-exceptions-sqlite.md) | 1990 | 12046 | 12000 | 达到 |
| [books/00-foundations/09-testing-debugging-and-quality.md](../books/00-foundations/09-testing-debugging-and-quality.md) | 1929 | 12015 | 12000 | 达到 |
| [books/00-foundations/10-cli-logging-and-interfaces.md](../books/00-foundations/10-cli-logging-and-interfaces.md) | 1728 | 8001 | 8000 | 达到 |
| [books/00-foundations/11-concurrency-async-and-cancellation.md](../books/00-foundations/11-concurrency-async-and-cancellation.md) | 1847 | 12054 | 12000 | 达到 |
| [books/00-foundations/12-git-from-snapshots-to-contribution.md](../books/00-foundations/12-git-from-snapshots-to-contribution.md) | 2006 | 12049 | 12000 | 达到 |
| [books/00-foundations/13-docker-images-containers-volumes.md](../books/00-foundations/13-docker-images-containers-volumes.md) | 2122 | 12022 | 12000 | 达到 |
| [books/00-foundations/14-math-numpy-and-numerics.md](../books/00-foundations/14-math-numpy-and-numerics.md) | 2043 | 12019 | 12000 | 达到 |
| [books/00-foundations/15-project-log-analysis-cli.md](../books/00-foundations/15-project-log-analysis-cli.md) | 2178 | 8025 | 8000 | 达到 |
| [books/00-foundations/16-project-task-service.md](../books/00-foundations/16-project-task-service.md) | 2514 | 8011 | 8000 | 达到 |
| [books/01-agent-engineering/01-loop.md](../books/01-agent-engineering/01-loop.md) | 2299 | 10756 | 10000 | 达到 |
| [books/01-agent-engineering/02-protocols.md](../books/01-agent-engineering/02-protocols.md) | 2195 | 10086 | 10000 | 达到 |
| [books/01-agent-engineering/03-tools.md](../books/01-agent-engineering/03-tools.md) | 2173 | 14327 | 14000 | 达到 |
| [books/01-agent-engineering/04-state-events.md](../books/01-agent-engineering/04-state-events.md) | 2399 | 14075 | 14000 | 达到 |
| [books/01-agent-engineering/05-streaming.md](../books/01-agent-engineering/05-streaming.md) | 1986 | 14030 | 14000 | 达到 |
| [books/01-agent-engineering/06-reliability.md](../books/01-agent-engineering/06-reliability.md) | 2235 | 14024 | 14000 | 达到 |
| [books/01-agent-engineering/07-context.md](../books/01-agent-engineering/07-context.md) | 2170 | 14182 | 14000 | 达到 |
| [books/01-agent-engineering/08-workspaces.md](../books/01-agent-engineering/08-workspaces.md) | 2148 | 10011 | 10000 | 达到 |
| [books/01-agent-engineering/09-concurrency.md](../books/01-agent-engineering/09-concurrency.md) | 2071 | 10025 | 10000 | 达到 |
| [books/01-agent-engineering/10-durability.md](../books/01-agent-engineering/10-durability.md) | 2569 | 14090 | 14000 | 达到 |
| [books/01-agent-engineering/11-workflows.md](../books/01-agent-engineering/11-workflows.md) | 2083 | 14074 | 14000 | 达到 |
| [books/01-agent-engineering/12-multi-agent.md](../books/01-agent-engineering/12-multi-agent.md) | 2028 | 10005 | 10000 | 达到 |
| [books/01-agent-engineering/13-evaluation.md](../books/01-agent-engineering/13-evaluation.md) | 2156 | 14004 | 14000 | 达到 |
| [books/01-agent-engineering/14-coding-agent.md](../books/01-agent-engineering/14-coding-agent.md) | 2141 | 10004 | 10000 | 达到 |
| [books/01-agent-engineering/15-integrations.md](../books/01-agent-engineering/15-integrations.md) | 2224 | 15100 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/01-evidence-and-retrieval.md](../books/02-retrieval-memory-graphs/01-evidence-and-retrieval.md) | 2480 | 10019 | 10000 | 达到 |
| [books/02-retrieval-memory-graphs/02-chunking-and-identities.md](../books/02-retrieval-memory-graphs/02-chunking-and-identities.md) | 2192 | 10023 | 10000 | 达到 |
| [books/02-retrieval-memory-graphs/03-bm25-from-counts.md](../books/02-retrieval-memory-graphs/03-bm25-from-counts.md) | 1928 | 14008 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/04-vectors-and-indexes.md](../books/02-retrieval-memory-graphs/04-vectors-and-indexes.md) | 3899 | 14001 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/05-hybrid-rerank-evaluation.md](../books/02-retrieval-memory-graphs/05-hybrid-rerank-evaluation.md) | 2073 | 14006 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/06-rag-and-citations.md](../books/02-retrieval-memory-graphs/06-rag-and-citations.md) | 2202 | 14023 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/07-query-planning-and-multihop.md](../books/02-retrieval-memory-graphs/07-query-planning-and-multihop.md) | 1999 | 10012 | 10000 | 达到 |
| [books/02-retrieval-memory-graphs/08-ingestion-access-lifecycle.md](../books/02-retrieval-memory-graphs/08-ingestion-access-lifecycle.md) | 2528 | 14029 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/09-memory-lifecycle.md](../books/02-retrieval-memory-graphs/09-memory-lifecycle.md) | 2394 | 14008 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/10-knowledge-graphs.md](../books/02-retrieval-memory-graphs/10-knowledge-graphs.md) | 1780 | 14030 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/11-graph-rag-and-global-questions.md](../books/02-retrieval-memory-graphs/11-graph-rag-and-global-questions.md) | 2117 | 14032 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/12-agentic-retrieval.md](../books/02-retrieval-memory-graphs/12-agentic-retrieval.md) | 2012 | 14013 | 14000 | 达到 |
| [books/02-retrieval-memory-graphs/13-two-capstone-projects.md](../books/02-retrieval-memory-graphs/13-two-capstone-projects.md) | 2141 | 10016 | 10000 | 达到 |
| [books/02-retrieval-memory-graphs/14-open-source-reading-and-frontiers.md](../books/02-retrieval-memory-graphs/14-open-source-reading-and-frontiers.md) | 2646 | 10565 | 10000 | 达到 |
| [books/03-model-architectures/01-tokens-tensors.md](../books/03-model-architectures/01-tokens-tensors.md) | 2090 | 10274 | 10000 | 达到 |
| [books/03-model-architectures/02-gradients-training.md](../books/03-model-architectures/02-gradients-training.md) | 1822 | 14098 | 14000 | 达到 |
| [books/03-model-architectures/03-attention.md](../books/03-model-architectures/03-attention.md) | 1816 | 14038 | 14000 | 达到 |
| [books/03-model-architectures/04-original-transformer.md](../books/03-model-architectures/04-original-transformer.md) | 1898 | 14080 | 14000 | 达到 |
| [books/03-model-architectures/05-dense-decoder.md](../books/03-model-architectures/05-dense-decoder.md) | 1864 | 14003 | 14000 | 达到 |
| [books/03-model-architectures/06-kv-gqa.md](../books/03-model-architectures/06-kv-gqa.md) | 1741 | 14041 | 14000 | 达到 |
| [books/03-model-architectures/07-latent-attention.md](../books/03-model-architectures/07-latent-attention.md) | 1704 | 10011 | 10000 | 达到 |
| [books/03-model-architectures/08-moe.md](../books/03-model-architectures/08-moe.md) | 1727 | 14016 | 14000 | 达到 |
| [books/03-model-architectures/09-sparse-attention.md](../books/03-model-architectures/09-sparse-attention.md) | 1912 | 10025 | 10000 | 达到 |
| [books/03-model-architectures/10-linear-attention.md](../books/03-model-architectures/10-linear-attention.md) | 1731 | 14010 | 14000 | 达到 |
| [books/03-model-architectures/11-state-space.md](../books/03-model-architectures/11-state-space.md) | 1765 | 14028 | 14000 | 达到 |
| [books/03-model-architectures/12-hybrid-residuals.md](../books/03-model-architectures/12-hybrid-residuals.md) | 2175 | 14011 | 14000 | 达到 |
| [books/03-model-architectures/13-quantization.md](../books/03-model-architectures/13-quantization.md) | 1750 | 10054 | 10000 | 达到 |
| [books/03-model-architectures/14-training-modalities.md](../books/03-model-architectures/14-training-modalities.md) | 2029 | 10126 | 10000 | 达到 |
| [books/03-model-architectures/15-tiny-language-model.md](../books/03-model-architectures/15-tiny-language-model.md) | 1845 | 10018 | 10000 | 达到 |
| [books/03-model-architectures/16-architecture-study.md](../books/03-model-architectures/16-architecture-study.md) | 2141 | 11209 | 10000 | 达到 |
| [books/04-inference-systems/01-request-lifecycle.md](../books/04-inference-systems/01-request-lifecycle.md) | 2069 | 13158 | 13000 | 达到 |
| [books/04-inference-systems/02-scheduling.md](../books/04-inference-systems/02-scheduling.md) | 2112 | 13032 | 13000 | 达到 |
| [books/04-inference-systems/03-paged-kv.md](../books/04-inference-systems/03-paged-kv.md) | 2045 | 13013 | 13000 | 达到 |
| [books/04-inference-systems/04-prefix-cache.md](../books/04-inference-systems/04-prefix-cache.md) | 1908 | 13027 | 13000 | 达到 |
| [books/04-inference-systems/05-preemption-cancellation.md](../books/04-inference-systems/05-preemption-cancellation.md) | 2025 | 13036 | 13000 | 达到 |
| [books/04-inference-systems/06-runner-backends.md](../books/04-inference-systems/06-runner-backends.md) | 1652 | 13028 | 13000 | 达到 |
| [books/04-inference-systems/07-sampling-streaming.md](../books/04-inference-systems/07-sampling-streaming.md) | 1667 | 9205 | 9000 | 达到 |
| [books/04-inference-systems/08-measurement.md](../books/04-inference-systems/08-measurement.md) | 1729 | 9117 | 9000 | 达到 |
| [books/04-inference-systems/09-source-reading.md](../books/04-inference-systems/09-source-reading.md) | 1916 | 13129 | 13000 | 达到 |
| [books/04-inference-systems/10-parallelism.md](../books/04-inference-systems/10-parallelism.md) | 1699 | 13008 | 13000 | 达到 |
| [books/04-inference-systems/11-pd-disaggregation.md](../books/04-inference-systems/11-pd-disaggregation.md) | 1668 | 13015 | 13000 | 达到 |
| [books/04-inference-systems/12-numerics-determinism.md](../books/04-inference-systems/12-numerics-determinism.md) | 1839 | 9282 | 9000 | 达到 |
| [books/04-inference-systems/13-capacity.md](../books/04-inference-systems/13-capacity.md) | 1518 | 9255 | 9000 | 达到 |
| [books/04-inference-systems/14-diagnostics.md](../books/04-inference-systems/14-diagnostics.md) | 1731 | 9084 | 9000 | 达到 |
| [books/04-inference-systems/15-hardware-labs.md](../books/04-inference-systems/15-hardware-labs.md) | 2055 | 9266 | 9000 | 达到 |
| [books/04-inference-systems/16-capstones.md](../books/04-inference-systems/16-capstones.md) | 2099 | 10977 | 9000 | 达到 |
| [books/04-inference-systems/17-source-workbook.md](../books/04-inference-systems/17-source-workbook.md) | 4365 | 13784 | 13000 | 达到 |
| [books/04-inference-systems/18-state-workbook.md](../books/04-inference-systems/18-state-workbook.md) | 4486 | 13163 | 13000 | 达到 |
| [books/04-inference-systems/19-numeric-paged-kv.md](../books/04-inference-systems/19-numeric-paged-kv.md) | 2823 | 13884 | 13000 | 达到 |
| [books/05-kernels-hardware/01-addresses-layout.md](../books/05-kernels-hardware/01-addresses-layout.md) | 1570 | 14497 | 14000 | 达到 |
| [books/05-kernels-hardware/02-cpp-memory.md](../books/05-kernels-hardware/02-cpp-memory.md) | 1743 | 14425 | 14000 | 达到 |
| [books/05-kernels-hardware/03-cuda-execution.md](../books/05-kernels-hardware/03-cuda-execution.md) | 1410 | 14052 | 14000 | 达到 |
| [books/05-kernels-hardware/04-memory-hierarchy.md](../books/05-kernels-hardware/04-memory-hierarchy.md) | 1332 | 14046 | 14000 | 达到 |
| [books/05-kernels-hardware/05-reductions-normalization.md](../books/05-kernels-hardware/05-reductions-normalization.md) | 1221 | 14007 | 14000 | 达到 |
| [books/05-kernels-hardware/06-gemm.md](../books/05-kernels-hardware/06-gemm.md) | 1318 | 14029 | 14000 | 达到 |
| [books/05-kernels-hardware/07-online-attention.md](../books/05-kernels-hardware/07-online-attention.md) | 1486 | 14017 | 14000 | 达到 |
| [books/05-kernels-hardware/08-triton-model.md](../books/05-kernels-hardware/08-triton-model.md) | 1248 | 14019 | 14000 | 达到 |
| [books/05-kernels-hardware/09-triton-kernels.md](../books/05-kernels-hardware/09-triton-kernels.md) | 1388 | 14026 | 14000 | 达到 |
| [books/05-kernels-hardware/10-custom-ops-autograd.md](../books/05-kernels-hardware/10-custom-ops-autograd.md) | 1451 | 14164 | 14000 | 达到 |
| [books/05-kernels-hardware/11-quantized-kernels.md](../books/05-kernels-hardware/11-quantized-kernels.md) | 1338 | 10024 | 10000 | 达到 |
| [books/05-kernels-hardware/12-moe-grouped-gemm.md](../books/05-kernels-hardware/12-moe-grouped-gemm.md) | 1271 | 10010 | 10000 | 达到 |
| [books/05-kernels-hardware/13-streams-events-graphs.md](../books/05-kernels-hardware/13-streams-events-graphs.md) | 1416 | 10068 | 10000 | 达到 |
| [books/05-kernels-hardware/14-performance-method.md](../books/05-kernels-hardware/14-performance-method.md) | 1385 | 16280 | 10000 | 达到 |
| [books/05-kernels-hardware/15-compilation.md](../books/05-kernels-hardware/15-compilation.md) | 1546 | 15017 | 14000 | 达到 |
| [books/05-kernels-hardware/16-hopper-blackwell.md](../books/05-kernels-hardware/16-hopper-blackwell.md) | 1403 | 10199 | 10000 | 达到 |
| [books/05-kernels-hardware/17-warp-persistent-megakernel.md](../books/05-kernels-hardware/17-warp-persistent-megakernel.md) | 1558 | 10008 | 10000 | 达到 |
| [books/05-kernels-hardware/18-communication-capstone.md](../books/05-kernels-hardware/18-communication-capstone.md) | 1443 | 10047 | 10000 | 达到 |
| [projects/P1_EVIDENCE_AGENT.md](../projects/P1_EVIDENCE_AGENT.md) | 未纳入初版主章统计 | 16799 | 12000 | 达到 |
| [projects/P2_MODEL_RUNTIME.md](../projects/P2_MODEL_RUNTIME.md) | 未纳入初版主章统计 | 13307 | 12000 | 达到 |

## 重复段落审稿线索

仅列跨文档完全相同且至少含 120 个汉字的段落。共同定义可能需要重复，须结合上下文判断。近似重复和语义空泛仍需人工检查。

未检出满足此规则的跨文档完全重复段落。
