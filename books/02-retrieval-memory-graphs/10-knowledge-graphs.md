# R10 知识图谱：从三元组到有证据的关系数据库

先修：字典、集合、SQLite、R08。你将实现一个有方向、有关系类型、有证据与版本约束的图，而不把任何连线图都称为知识图谱。

## 1. 三种常见的图不要混在一起

工作流图描述下一步执行哪个任务：节点是操作，边是控制或数据流。模型计算图描述张量运算依赖：节点是算子或数值，边传递张量。知识图谱描述对象之间的关系：节点可以是项目、模块、概念、人物或文档，边表达 uses、depends_on、part_of 等事实或断言。

例如 Agent→Retrieval→Index 可以是知识图上的依赖路径，也可以恰好对应某个工作流，但两者语义不相同。工作流边意味着运行时转移；知识关系边只表示某个来源声称这些对象有关，未必每次请求都经过它。

图的好处是让关系显式可查询。它不自动解决抽取准确性、实体消歧、数据更新或推理可靠性。错误边一旦进入图，可能在多跳路径中传播并变得更难察觉。

## 2. 一个关系至少需要哪些字段

最简单三元组是 `(subject,predicate,object)`。但在可信系统里还要有关系 ID、租户、来源 chunk、原文引文、版本、时间范围和可能的置信状态。例如 `(agent,uses,retrieval)` 加上来源“Agent uses retrieval.”，才知道这条边来自何处。

subject/object 应使用稳定实体 ID，而不是只用展示名称。同名公司、同名模块、同一个缩写在不同项目中含义不同。alias 别名映射应保留来源与范围，不要把“名字像”直接当成实体相等。

predicate 最好来自受控集合。本卷允许 uses、depends_on、implemented_by、part_of、supersedes、supports。限制集合让查询与测试明确，但不保证模型正确选择关系。`part_of` 与 `uses` 不可任意互换；方向也必须固定，否则路径查询会悄悄反过来。

## 3. 结构校验与语义校验是两回事

模型可以输出合法 JSON，subject/object 非空、predicate 在白名单、source_id 存在，却仍抽错关系。例如来源“系统不依赖缓存”，模型抽出 depends_on(cache)。schema 校验通过不代表事实正确。

本卷 `EvidenceGraph.add` 检查来源当前可见且引文在片段中出现。它证明关系有一个可追踪的引用位置，但**不自动证明引文蕴含关系**。课程样例由手工确认关系，避免用未经验证的生成抽取制造知识。真实抽取流程需要标注集、抽样复核、否定与条件识别，以及可撤销的关系状态。

一条关系也可能只是某作者的观点。把它存为“来源 A 断言 X”与存为“X 是系统真理”有不同语义。对于争议或时态信息，可以保留多个来源断言，再由应用层决定合并策略；不应在摄入时无声抹平冲突。

## 4. 手工建立一个小图

语料片段包含：“Agent uses retrieval. Retrieval uses index. Index uses corpus.” 建立三个实体边：agent uses retrieval，retrieval uses index，index uses corpus。查询 agent 到 index 的路径应返回两条有序边，并能展示每条边的 chunk_id 和 quote。

这里的边是有向的。agent 到 index 有路径，不表示 index 到 agent 有同样路径。若需要反向查“谁使用这个索引”，可以建立反向邻接结构或在查询时按 object 搜索，不能把所有关系当无向边，否则用途和被使用关系会混淆。

图包含环并不一定是坏数据。模块 A 与 B 可能相互调用；知识库也可能有互相引用。遍历时必须有 visited 集合与预算，避免无限循环。语义上的不合理环，比如某严格包含关系中的自包含，应该由关系类型约束检测，而不是把所有环都禁止。

## 5. BFS 为什么能找到最短跳数路径

广度优先搜索使用队列，先处理长度 0 的起点，再处理长度 1 的邻居，再处理长度 2 的邻居。对于每条边代价相同的图，首次到达目标得到最少边数路径。这里最短表示跳数最少，不表示证据最强、事实最可信或语义最合理。

课程队列元素是 `(node,path)`，path 是到当前节点的有序边列表。出队后检查深度，遍历允许边，构造 candidate path，命中目标就返回。visited 以节点去重，使一个节点只被扩展一次。这个选择适合找一条最短路径；若任务需要所有路径或按多种关系约束比较，需要更丰富的状态，例如 `(node,relation_state)`，不能直接照搬。

`max_depth` 控制路径长度，`max_expansions` 控制出队次数。两者都需要，因为一个深度很小但分支极多的图也可能昂贵。真正大型图还要限制候选边数、时间和输出大小。

## 6. 更新与权限会改变图

图边引用具体 chunk。来源更新后旧 chunk 不再当前可见，旧边就不参与查询；软删除后边仍保留数据库里，但 visible_edges 会过滤它；purge 则通过外键级联删除它。这个设计将证据生命周期传播到关系层。

课程 visible_edges 先取可见 chunk ID，再过滤租户内边。对小语料易读；大图应将权限与当前版本条件下推到索引或 SQL join，避免加载无关边。过滤必须在把图交给模型前完成，不能先生成跨租户路径再隐藏其中几个节点，因为路径本身可能泄露不可见关系。

图派生摘要也要跟踪来源边集合。若撤销一条关键关系，而社区摘要仍说它存在，图数据库清理正确也不能保证回答正确。R11 将讨论这种派生依赖。

## 7. 实操与练习答案

```bash
python -m labs.knowledge.lab 25-graph
python -m labs.knowledge.lab 26-graph-stale
```

**练习 1：写邻接表。** 完整参考解：

```python
from collections import defaultdict

def adjacency(edges, reverse=False):
    result = defaultdict(list)
    for edge in edges:
        key = edge.object if reverse else edge.subject
        result[key].append(edge)
    return {key: sorted(value, key=lambda e: e.edge_id)
            for key, value in result.items()}
```

验收：反向表以 object 为 key，但边本身仍保留原方向；不要交换 subject/object 后假装谓词不变。

**练习 2：边数与节点数。** agent→retrieval→index 路径包含 2 条边、3 个节点。若 max_depth=1，应找不到目标；max_depth=2 可以找到。这是常见 off-by-one 测试。

**练习 3：否定关系。** 来源“模块不依赖远程数据库”却抽出 depends_on(remote_database)。即使 quote 存在也不能通过语义验收。完整参考处理是将抽取标为待审核或拒绝，并增加否定样例，不是简单在 schema 加一个 confidence 就解决。

**练习 4：同名实体。** 两个项目都有 Scheduler。参考 ID 可用 `projectA:Scheduler` 与 `projectB:Scheduler`，展示名仍是 Scheduler。跨项目合并需要明确等价证据，不能靠 lowercase 相等。

**练习 5：来源撤销。** 先建立两跳路径，删除第二条边的来源文档，再查询。预期路径消失；若仍出现，检查是否缓存了未经版本约束的邻接表。测试必须覆盖在线读取与缓存读取两种入口。

**掌握门槛：** 能实现有方向 BFS、打印来源支持的路径、处理环与预算，并解释图关系为何不自动具有逻辑传递性。本实现是通用教学证据图，不是对某生产 GraphRAG 项目的完整复刻。
