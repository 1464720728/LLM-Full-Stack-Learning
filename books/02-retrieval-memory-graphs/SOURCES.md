# 来源与版本说明

核查日期：2026-11-01。正文机制讲解、虚构数据、手算例题与课程代码为独立教学设计。以下采用原作者论文、官方文档与项目源代码；没有把第三方转载当作官方实现。动态 main 和在线文档不是锁定提交，进行扩展实验时必须记录实际 `git rev-parse HEAD` 与依赖版本。此处不伪造未经 checkout 验证的 SHA。

| 主题 | 第一手来源 | 本卷使用范围 |
|---|---|---|
| 信息检索基础 | [Stanford IR book](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) | tf/df/IDF、排名与评价的学习入口 |
| RAG 历史 | [Lewis et al., 2005.11401](https://arxiv.org/abs/2005.11401) | 原始论文与现代工程通称的区分 |
| RRF | [Cormack et al. 原文](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) | 基于名次融合的来源 |
| 双编码与重排 | [Sentence Transformers](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) | 编码器/重排器职责、真实扩展入口 |
| Embeddings API | [OpenAI 官方接口](https://platform.openai.com/docs/api-reference/embeddings/create) | 请求/响应协议、index 与维数校验 |
| 向量索引 | [Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) | 精确与近似索引比较入口 |
| SQLite 外键 | [官方说明](https://www.sqlite.org/foreignkeys.html) | 开启外键、级联与存在性约束 |
| SQLite 事务 | [官方说明](https://www.sqlite.org/lang_transaction.html) | 事务、写入并发与 rollback |
| Haystack | [BM25 源码](https://github.com/deepset-ai/haystack/blob/main/haystack/components/retrievers/in_memory/bm25_retriever.py) | 检索器与 store 的代码阅读任务 |
| Haystack store | [官方概念](https://docs.haystack.deepset.ai/docs/document-store) | 存储与 Retriever 的接口 |
| LlamaIndex | [IngestionPipeline 源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/ingestion/pipeline.py) | 变换、缓存、文档生命周期阅读 |
| LlamaIndex 摄入 | [官方仓库文档](https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/loading/ingestion_pipeline/index.md) | 文档管理与扩展实验入口 |
| LangGraph persistence | [官方文档](https://docs.langchain.com/oss/python/langgraph/persistence) | thread checkpoint 与长期 store 的区分 |
| LangGraph memory | [官方文档](https://docs.langchain.com/oss/python/concepts/memory) | 短期/长期与记忆类型概念 |
| GraphRAG 论文 | [Edge et al., 2404.16130](https://arxiv.org/abs/2404.16130) | 全局问题、图组织与社区摘要的来源 |
| GraphRAG 实现 | [官方项目文档](https://microsoft.github.io/graphrag/) | 索引与查询工作流的研读入口 |

## 验证范围

- 标准库知识系统：离线 CPU 执行，覆盖词元/切块、BM25数值、排名、ACL、版本、撤权、删除、引用、记忆、图以及全部演示。
- NumPy 微型双编码器：离线 CPU 执行，使用有限差分检查解析梯度，验证小训练集 loss 下降。不是对真实语义检索质量的测试。
- 真实外部框架：核查文档与源码入口，提供明确任务与验收；本次未安装并执行其完整环境，不能声称它们在用户机器上已通过。
- API 与 GPU：默认零调用、零占用。真实模型与大规模索引实验由学习者按章节契约执行后填写记录。

## 更新本卷时必须同时更新什么

如果修改 tokenizer，检查切块指纹与索引缓存版本；如果修改 Chunk 或 Context，运行全部系统和综合项目测试；如果修改引用语义，更新报告状态名与练习答案；如果采用新模型或向量索引，记录模型/索引版本对并重新评测。不要只更新 README 中的“最新”字样而保留旧实现假设。

新增验证：Embeddings 适配器使用注入传输做离线测试，覆盖乱序、缺失、重复、维数、非有限值、预算和授权出口；未执行真实 API。FAISS 实验为可选依赖，当前缺失时测试明确 SKIP。
