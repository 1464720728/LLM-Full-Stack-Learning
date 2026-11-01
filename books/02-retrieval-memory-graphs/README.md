# R 卷：检索增强、记忆与知识图谱工程

本卷把“给模型一些资料”展开成可以实现、解释、测试和维护的系统。核心能力包括可追踪的资料摄入、BM25、向量表示、混合检索、评价、引用验证、多跳查询、权限与版本、长期记忆，以及有证据的图检索。与 [Agent 工程卷](../01-agent-engineering/README.md) 配合，形成能开发知识型 Agent 的完整路线。

**学习方式：** 每章先手算一个例子，再读指定函数，再运行实验，然后完成修改、失败定位和项目任务。只看正文而不重写关键算法，不算完成。本卷包含 14 个主章、R15 源码长工作簿（8 千汉字以上）、R16 真实 Embeddings 与索引工作簿、31 个微实验入口（28 个标准库实验＋NumPy 微型学习器＋Embeddings 适配器＋可选 FAISS）、8 个源码检查站、两套完整 CLI 项目，以及独立测试。不是所有实验都需要显卡；这里全部使用 CPU，真实框架扩展也先从小规模开始。

本次深写已将14个主章扩为 **176,235 个去代码中文汉字**，每章直接进入递进讲解、手算与实作。正文122个Python代码块已按章实际执行通过；知识测试76通过、1项可选FAISS跳过。逐章覆盖、故障实验与未执行范围见 [深度审计](DEPTH_AUDIT.md)，原始执行记录见 [DEPTH_EXECUTION.json](DEPTH_EXECUTION.json)。

两个工作簿也注册到课程清单：R15 为八站代码执行跟读，R16 为真实向量接口和可选索引实验。

## 从哪里开始

先完成基础卷中的 Python、文件与 JSON、SQLite、测试和命令行部分。数学需要能读分数、对数、向量点积；不熟悉时跟着手算例题补。然后依次读 R01–R08，完成本地证据助手，再读记忆与图部分。进入 R12 时结合 Agent 卷的工具执行、预算和状态管理。

| 章 | 主题 | 核心交付 |
|---|---|---|
| [R01](01-evidence-and-retrieval.md) | 检索与证据对象 | 错误定位与引用层级 |
| [R02](02-chunking-and-identities.md) | 切块、身份与摄入契约 | 不丢字、不死循环的切块器 |
| [R03](03-bm25-from-counts.md) | BM25 从计数到公式 | 可逐词解释的排名 |
| [R04](04-vectors-and-indexes.md) | 向量、ANN、微型双编码器 | 精确基准与可验证梯度 |
| [R05](05-hybrid-rerank-evaluation.md) | RRF、重排、MMR 与评价 | 配对消融与排名指标 |
| [R06](06-rag-and-citations.md) | RAG、预算与引用 | 可验证的抽取式回答 |
| [R07](07-query-planning-and-multihop.md) | 查询规划与多跳 | 有停止条件的研究轨迹 |
| [R08](08-ingestion-access-lifecycle.md) | 版本、权限、删除、事务 | 更新与撤权回归测试 |
| [R09](09-memory-lifecycle.md) | 长期记忆生命周期 | 修正、CAS、TTL、遗忘 |
| [R10](10-knowledge-graphs.md) | 有证据的知识图谱 | 带来源的定向路径 |
| [R11](11-graph-rag-and-global-questions.md) | 局部与全局 GraphRAG | 图与摘要依赖评估 |
| [R12](12-agentic-retrieval.md) | Agentic RAG | 受控工具、状态与预算 |
| [R13](13-two-capstone-projects.md) | 两套完整项目 | 证据助手、记忆助手 |
| [R14](14-open-source-reading-and-frontiers.md) | 开源源码与前沿研读 | 真实调用链、测试与小改动 |

每读完 R03、R06、R09、R10，分别到 [长工作簿](WORKBOOK-source-to-running-system.md) 完成相应检查站。它将同一份具体输入从 SQLite、Counter、评分、引用一路追踪到记忆与图，不要求你凭抽象名词自己补全代码执行过程。

完整的 [真实 Embeddings API 与 FAISS 实操](LAB-real-embeddings-and-faiss.md) 提供可选真实模型接口与离线 mock 测试。`--live` 是唯一联网入口，默认不联网。

## 一次跑通

```bash
python -m labs.knowledge.lab all
python -m labs.knowledge.walkthrough --stage all
python -m labs.knowledge.neural_encoder
python -m labs.knowledge.embedding_api
python -m labs.knowledge.faiss_lab
python -m unittest discover -s tests -p "test_knowledge*.py" -v
python -m labs.knowledge.assistant ask "KV cache" --demo
python -m labs.knowledge.memory_assistant demo
```

前三条只有 `neural_encoder` 需要 NumPy，其他核心实验只需标准库。测试在缺少 NumPy 时明确跳过学习器测试，不把跳过说成执行成功。所有默认示例使用内存数据库或临时目录，不访问网络，不消耗 API。

## 代码地图

| 文件 | 负责什么 | 刻意不负责什么 |
|---|---|---|
| [text.py](../../labs/knowledge/text.py) | 词法 tokenizer、精确字符切片、身份哈希 | 模型 tokenizer、PDF/OCR |
| [store.py](../../labs/knowledge/store.py) | SQLite 版本、ACL、摄入、删除 | 网络认证、跨库事务、取证擦除 |
| [retrieval.py](../../labs/knowledge/retrieval.py) | BM25、TF-IDF、RRF、MMR | 生产 ANN、预训练语义模型 |
| [neural_encoder.py](../../labs/knowledge/neural_encoder.py) | 从零学习微型双编码器、显式梯度 | 通用语义能力与真实质量报告 |
| [embedding_api.py](../../labs/knowledge/embedding_api.py) | 显式联网适配器、响应校验、真实向量检索 | 自动消费API、保证供应商兼容 |
| [faiss_lab.py](../../labs/knowledge/faiss_lab.py) | 可选 Flat/HNSW 同场对照 | 缺依赖时伪装已测 |
| [evaluation.py](../../labs/knowledge/evaluation.py) | 精确指标与逐题对比 | 自动判定所有事实真假 |
| [rag.py](../../labs/knowledge/rag.py) | 预算、严格抽取、引用复核 | 自由生成式蕴含判断 |
| [memory.py](../../labs/knowledge/memory.py) | 记忆隔离、修正、过期、遗忘 | 自动推断真实用户偏好 |
| [graph.py](../../labs/knowledge/graph.py) | 有来源的关系与有界 BFS | 完整 Microsoft GraphRAG |
| [assistant.py](../../labs/knowledge/assistant.py) | 本地证据 CLI 项目 | 多用户公网服务 |
| [memory_assistant.py](../../labs/knowledge/memory_assistant.py) | 本地记忆 CLI 项目 | 认证授权产品 |

## 学习成果如何验收

基础关：独立重写 tokenizer、切块器、BM25、余弦、RRF 和 BFS 中至少四项，通过手算与边界测试。系统关：从空数据库完成摄入、查询、版本更新、旧引用失效、权限隔离和删除。Agent 关：把检索工具接入执行器，身份由运行时注入，记录预算与停止原因。记忆关：有证据写入、纠正冲突、TTL 和删除全部版本。开源关：固定一个真实项目提交，读一条完整调用链，写一个失败测试并做一个小改动。

测试通过只说明课程参考实现的相关行为在当前 CPU 环境验证通过。本卷没有宣称执行过外部模型 API、真实大规模 embedding、生产向量库或 H100 性能实验。最新外部项目资料见 [来源与版本说明](SOURCES.md)；扩展任务要求你记录实际环境和验证结果。
