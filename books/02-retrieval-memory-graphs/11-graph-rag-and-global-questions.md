# R11 GraphRAG 与全局问题：组织证据、社区与可撤销摘要

先修 R05 的评价、R06 的引用、R10 的证据图。本章要解决的问题不是“如何画更大的图”，而是“怎样从许多文档中得到有覆盖、有出处且可更新的综合回答”。先用完全离线的透明实现理解组织与聚合，再阅读 Microsoft GraphRAG 的论文与官方实现。课程 BFS 只是一种局部证据遍历，不能被称为完整 GraphRAG。

## 小课一：局部事实查询与全局综合查询

“错误码 E204 表示什么”通常有一个明确位置，精确检索找到相关段落即可。“整个项目最主要的风险与例外有哪些”需要跨许多文档归纳，答案不一定在任何一页中直接出现。两种问题需要不同证据结构。局部问题重视找到特定支持，全局问题还重视覆盖、代表性、重复来源与少数例外。

普通 top-k 检索按与查询的相关程度排序。若十篇缓存风险文档都很像“主要风险”，它们可能挤满前十，调度与权限风险一篇都没进上下文。最终答案写得很长，却只代表局部主题。把 k 加大可以改善部分覆盖，但上下文和费用增长，重复文档仍可能占据大量空间。全局问题需要一个明确的组织与聚合策略。

图可以通过实体、关系与社区提供组织结构，但不是唯一方案。文档主题分组、层级摘要、按元数据分层抽样、结构化统计也可能有效。先定义需要覆盖哪些内容以及如何验收，再决定图的额外成本是否值得。对于一个只有十二份文档的练习，手工主题分组能作为非常强的透明基线，不能因为没有复杂算法就忽略它。

Microsoft GraphRAG 的具体方法包含从文本提取实体关系、构建图、社区组织与社区报告，并利用这些结构服务不同查询方式。它比泛称“图增强 RAG”更具体。阅读时应区分方法论文、当前官方文档和安装版本，不能把课程的三跳 BFS、任意知识图查询与某版本完整实现画等号。本章实现只承担所声明的教学子问题。

## 小课二：先建立能手算的全局语料

我们用十二份虚构资料，分为缓存、调度、权限三组。每组四条：一个主要风险、一个缓解措施、一个例外、一个重复来源。风险与措施不能混成同一事实，例外不能被多数票吞掉，重复来源不能算独立支持。所有内容都短到你可以逐条核对，适合观察组织策略改变了什么。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Evidence:
    source_id: str
    topic: str
    kind: str
    text: str
    original_id: str
    revision: int = 1

corpus = [
    Evidence("c1", "cache", "risk", "Long contexts increase cache demand.", "cache-report"),
    Evidence("c2", "cache", "mitigation", "Capacity checks can delay admission.", "cache-design"),
    Evidence("c3", "cache", "exception", "Short isolated requests may avoid cache pressure.", "cache-test"),
    Evidence("c4", "cache", "risk", "Long contexts increase cache demand.", "cache-report"),
    Evidence("s1", "scheduler", "risk", "Large batches can increase waiting time.", "schedule-report"),
    Evidence("s2", "scheduler", "mitigation", "Queue limits bound pending work.", "schedule-design"),
    Evidence("s3", "scheduler", "exception", "Light load may have no queue delay.", "schedule-test"),
    Evidence("s4", "scheduler", "risk", "Large batches can increase waiting time.", "schedule-report"),
    Evidence("p1", "access", "risk", "Stale permissions can expose restricted evidence.", "access-report"),
    Evidence("p2", "access", "mitigation", "Recheck permissions before output.", "access-design"),
    Evidence("p3", "access", "exception", "Public-only corpora have no private source boundary.", "access-test"),
    Evidence("p4", "access", "risk", "Stale permissions can expose restricted evidence.", "access-report"),
]
assert len(corpus) == 12
assert len({item.original_id for item in corpus}) == 9
```

source_id 表示本次资料条目，original_id 表示独立原始来源。c1 与 c4 是两份传播副本，但来自同一原始报告，因此独立来源只有九个。revision 绑定快照，后面更新或撤权会使摘要依赖失效。topic 与 kind 在此由人工提供，作为金标组织结构；真实系统自动得到这些字段时必须另测分类误差，不能把人工完美分组的效果归功于模型。

全局答案的验收清单可以是九个槽位：每个主题的风险、措施、例外。若用户只问风险与例外，则六个槽位。答案长度不是指标，主题名出现也不等于对应事实被支持。每个槽位要有具体断言与来源，任何新增没有依据的风险都算错误。这个标注使你能够比较不同组织方法，而不是凭感觉选更像报告的一段。

## 小课三：局部图扩展怎样增加相关材料

局部流程可以是文本检索种子、识别种子实体、沿允许关系扩展、取回边来源、合并原文、回答。图的作用是找到词面可能不同但关系上相关的材料。例如用户问 Jade 的更新依赖，文本先找到 Jade，图发现 Quartz 与 Cedar，再去取 Cedar 的版本发布机制。最终陈述仍需要原文，不能只根据实体路径生成能力结论。

扩展参数要明确：哪些谓词、向前还是反向、最多几跳、扫描多少边、最多返回多少来源。深度二不等于成本小，若第一层是高度枢纽，第二层可能覆盖大半图。图扩展也会引入噪声，尤其“系统”“模型”等泛化实体。应该在固定问题集上比较零跳、一跳、两跳，观察新增正确证据与新增无关证据，而不是假设越深越好。

```python
def expand_visible(graph, context, seeds, predicates, max_edges=20):
    if type(max_edges) is not int or max_edges < 0:
        raise ValueError("invalid edge budget")
    seed_set = set(seeds)
    edges = [edge for edge in graph.visible_edges(context)
             if edge.subject in seed_set and edge.predicate in predicates]
    return sorted(edges, key=lambda edge: edge.edge_id)[:max_edges]
```

这个函数先获取当前可见边，再做种子与谓词筛选，所以不会先构造私有路径再隐藏某几个名字。它只扩展一轮，没有自动循环，也没有语义排序；多轮需要全局已访问状态与总预算。排序按边 ID 保证确定性，不声称最相关。把这些简化写清楚，才能正确解释实验结果。

## 小课四：社区是什么，以及它不是什么

图社区通常指内部连接相对密集、与外部连接相对稀疏的节点集合。不同算法优化不同目标，结果受边权、分辨率、随机种子与图结构影响。社区不必等于业务主题，也不必有清楚人类名称。一个高度公共节点可能把多个主题连起来，抽取错误也可能制造虚假桥梁。社区标签应由内容解释与验证，不能当自动发现的客观真理。

连通分量只要求节点之间存在路径，比社区弱得多。两团密集节点之间只要一条桥，就属于同一连通分量，但社区算法可能仍把它们分开。课程下面的连通分量代码用于理解组织基础，明确不是 Leiden 或 Louvain 的替代实现，也不代表 Microsoft GraphRAG 社区检测的全部过程。

```python
def connected_components(nodes, undirected_edges):
    adjacency = {node: set() for node in nodes}
    for a, b in undirected_edges:
        if a not in adjacency or b not in adjacency:
            raise ValueError("unknown node")
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining, components = set(nodes), []
    while remaining:
        start = min(remaining)
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        remaining -= component
        components.append(tuple(sorted(component)))
    return components

assert connected_components({"a", "b", "c", "d"}, [("a", "b"), ("c", "d")]) == [("a", "b"), ("c", "d")]
assert connected_components({"a", "b", "c", "d"}, [("a", "b"), ("b", "c"), ("c", "d")]) == [("a", "b", "c", "d")]
```

算法用栈遍历每个尚未处理的节点集合，所有相连节点归为一个分量。边被当无向，仅用于分组结构；原知识关系仍有方向，不能因为社区检测使用无向投影就把 uses 改成对称事实。图的某种分析视图可以丢方向，但原始证据层不能丢。

## 小课五：模块度的每个符号与一个小例子

常见社区目标之一是模块度。对无向图，可以把每个社区内部真实边权与随机连接基线比较：Q 等于各社区的内部边数除以总边数，减去社区总度数除以两倍总边数的平方，然后求和。这里总边数 m 每条无向边计一次，社区内部边数 l 也计一次，社区总度数 d 是其中所有节点度数之和，跨社区边会贡献一个端点的度。

写成公式为 $Q=\sum_c[l_c/m-(d_c/(2m))^2]$。第一项奖励内部连接，第二项扣除在同样度数分布下随机连接可能产生的内部比例。它不是“社区语义正确率”，只是结构目标。度数高的节点在随机基线中本来就更可能连接，因此要有扣除项。若 m 为零，公式分母为零，需要定义特殊处理，不能直接计算。

手算两个三角形，中间有一条桥。总边数七，每个三角形内部三条边，总度数七。分成两个社区时，每项是三除七减去二分之一平方，约零点一七八六，总和约零点三五七一。全部节点放一个社区时，内部七除七为一，总度数十四除十四平方也是一，Q 为零。这个目标倾向保留两个团，但不意味着所有图都能用同样粒度得到人类想要的主题。

```python
def modularity(nodes, edges, communities):
    edges = {tuple(sorted(edge)) for edge in edges}
    if not edges:
        return 0.0
    memberships = [node for community in communities for node in community]
    if set(memberships) != set(nodes) or len(memberships) != len(set(memberships)):
        raise ValueError("communities must partition nodes")
    degree = {node: 0 for node in nodes}
    for a, b in edges:
        if a == b:
            raise ValueError("self loops excluded in this teaching formula")
        degree[a] += 1
        degree[b] += 1
    m = len(edges)
    score = 0.0
    for community in communities:
        group = set(community)
        internal = sum(a in group and b in group for a, b in edges)
        total_degree = sum(degree[node] for node in group)
        score += internal / m - (total_degree / (2 * m)) ** 2
    return score

nodes = set("abcdef")
edges = {("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f"), ("c", "d")}
assert abs(modularity(nodes, edges, [set("abc"), set("def")]) - 5/14) < 1e-12
assert abs(modularity(nodes, edges, [nodes])) < 1e-12
```

这段函数只计算给定划分的目标值，不搜索最优划分。实际社区算法要在巨大组合空间中优化，并处理层级、边权与稳定性。模块度存在分辨率等局限，参数与目标可能把小社区合并。学习者应先理解目标含义，再阅读算法的移动、聚合与细化步骤，不能把一个社区库调用当作已经理解机制。

## 小课六：社区报告是派生证据，不是新的权威来源

社区报告把某组节点、边与文档压缩成可用于查询的描述。它至少需要内容依赖与生成依赖两套来源。内容依赖包括哪些 chunk、关系和版本；生成依赖包括模型、提示模板、schema、参数与运行标识。资料更新会使内容依赖失效，生成规则升级会改变报告行为，两类变化不能混为一谈。

摘要会丢信息，尤其少数例外、否定和条件。一个社区十篇资料说容量风险，一篇说明短请求例外，摘要若只保留多数说法，会把有限条件变成普遍规律。全局问答需要保留风险、缓解、例外、争议与覆盖缺口，而不是只输出“主要主题”。报告结构应服务这些任务槽位。

先做严格抽取式报告，按主题与 kind 保留一条独立来源，避免生成模型带来的语义误差。这个版本不具备自然语言综合能力，但可以验证组织、去重、来源与失效。以后把抽取换成模型摘要时，同样的依赖与验收继续使用。

```python
@dataclass(frozen=True)
class CommunityReport:
    report_id: str
    topic: str
    statements: tuple[tuple[str, str, str], ...]
    dependencies: tuple[tuple[str, int], ...]
    generator_version: str


def build_reports(items):
    topics = sorted({item.topic for item in items})
    reports = []
    for topic in topics:
        selected, seen_originals = [], set()
        for item in sorted(items, key=lambda row: row.source_id):
            if item.topic != topic or item.original_id in seen_originals:
                continue
            seen_originals.add(item.original_id)
            selected.append(item)
        reports.append(CommunityReport(
            "report:" + topic, topic,
            tuple((item.kind, item.text, item.source_id) for item in selected),
            tuple((item.source_id, item.revision) for item in selected),
            "strict-extract-v1"))
    return reports

reports = build_reports(corpus)
assert len(reports) == 3
assert all(len(report.statements) == 3 for report in reports)
assert all(len(report.dependencies) == 3 for report in reports)
```

seen_originals 在每个主题内重置，表示同一主题重复传播不增加独立支持；若一个原始来源确实包含多个不同 kind，简单按 original_id 去重可能过度删除。本例每个独立来源只有一个事实，所以成立。真实项目应按断言与来源组合去重，不能机械照搬这个简化。明确数据假设是理解代码的一部分。

## 小课七：地图—归约流程到底聚合了什么

map 阶段让每个社区报告针对问题产生局部候选回答，reduce 阶段合并这些候选、去重、保留支持与例外。地图—归约只是分治模式，不自动保证全面。若某社区没被查询，reduce 无法凭空恢复；若 map 丢了例外，reduce 可能永远看不到；若 reduce 按最高分截断，少数重要主题仍可能消失。

本章先用固定槽位 map：问题要求 risk 与 exception，每个报告返回相应抽取语句。reduce 按 topic 与 kind 合并，保留来源。最后检查所需槽位是否全部覆盖。这样你可以手算每一步，确认覆盖来自组织结构，而不是模型背景知识。

```python
def map_report(report, required_kinds):
    return [{"topic": report.topic, "kind": kind, "text": text,
             "source_id": source_id, "report_id": report.report_id}
            for kind, text, source_id in report.statements if kind in required_kinds]


def reduce_maps(mapped, required_topics, required_kinds):
    by_slot = {}
    for rows in mapped:
        for row in rows:
            slot = (row["topic"], row["kind"])
            by_slot.setdefault(slot, []).append(row)
    required = {(topic, kind) for topic in required_topics for kind in required_kinds}
    return {"answers": by_slot, "missing": sorted(required - set(by_slot)),
            "complete": required <= set(by_slot)}

mapped = [map_report(report, {"risk", "exception"}) for report in reports]
global_answer = reduce_maps(mapped, {"cache", "scheduler", "access"}, {"risk", "exception"})
assert global_answer["complete"]
assert len(global_answer["answers"]) == 6
```

reduce 只合并结构，没有把多个句子改写成一篇流畅文章。真实生成层可以在这些槽位上组织语言，但必须保留每个断言的来源与条件。若用户要求“最主要三项风险”，还需要定义重要性依据，是出现频率、影响程度、专家标注还是任务相关性；不能把 map 的模型自评分无条件当客观严重程度。

## 小课八：摘要依赖与失效，先写一个严格判定器

一个摘要 S 依赖 a 第一版、b 第二版、c 第一版。只要 b 更新到三，S 就不能被当作当前全量摘要；若 b 被撤权，对该用户更不能显示 S 中由 b 推出的内容。只从引用列表删掉 b 而保留正文，会把不可见事实洗成没有来源的公开结论。摘要是依赖的产物，不能比它的必要来源拥有更宽权限。

有两种基本策略。保守策略要求全部依赖当前可见且版本匹配，否则整份摘要失效；精细策略给每个断言记录依赖，只移除或重建受影响断言。前者容易实现与验证，可能重算更多；后者利用率高，但需要可靠断言级来源映射。初学项目先实现保守策略，把行为跑对，再逐步细化。

```python
def report_is_current(report, visible_revisions, generator_version):
    if report.generator_version != generator_version:
        return False
    return all(visible_revisions.get(source_id) == revision
               for source_id, revision in report.dependencies)

visible = {item.source_id: item.revision for item in corpus}
assert all(report_is_current(report, visible, "strict-extract-v1") for report in reports)
changed = dict(visible)
changed["c2"] = 2
cache_report = next(report for report in reports if report.topic == "cache")
assert not report_is_current(cache_report, changed, "strict-extract-v1")
revoked = dict(visible)
del revoked["p1"]
access_report = next(report for report in reports if report.topic == "access")
assert not report_is_current(access_report, revoked, "strict-extract-v1")
```

visible_revisions 必须由当前可信授权视图构造，不能由模型提供。这里字典同时表达存在、权限与版本匹配，缺失统一视为不可用。它不证明摘要内容语义正确，只证明依赖资格仍成立；生成时的支持检查与当前依赖复核属于两道不同检查。generator_version 变化也让旧报告失效，避免同一个集合混用不同摘要规则而无法重现。

更新不一定只影响一个社区。新边可能改变社区划分，把节点移动到另一组，引起多个报告重建。稳定报告 ID 与社区身份也需设计：若社区编号每次随机重排，报告缓存不能仅凭编号认为对象相同。可以记录成员与边的确定性指纹，或维护明确社区代际。图算法输出的整数标签不是永久业务 ID。

## 小课九：权限混合摘要为什么是一个独立风险

假设公开文档说项目使用索引，私有文档说该索引存在未修复故障。把两者放进同一社区生成“项目索引存在风险”，然后将社区报告标为公开，会泄露私有关系，即使没有原样复制私有句子。语义概括仍然可能携带受限信息，不能只检查是否包含原文子串。

一种方案按授权范围生成报告，但用户与组组合可能很多，缓存与重建成本高。另一种方案保存报告完整依赖，读取时要求用户能访问全部来源；这样安全保守，但公开信息可能因混入私有依赖而一起不可用。更细方案在断言层保存依赖并针对可见断言重新组织答案。每种方案都需要明确权衡，没有一个通用“摘要自动继承公开节点权限”的规则。

社区结构本身也可能泄露信息。若普通用户看到一个公开实体与某个未公开项目高度关联，即使边正文隐藏，也可能推断关系。最稳妥的教学流程是在授权可见图上生成当前用户使用的子图与报告；大规模系统可以优化，但必须证明等价或明确边界。先全图聚类再删私有节点，不一定等同于只在公开图聚类。

不能把外部图数据库账号当成业务权限的全部实现。一个服务账号可能能读取全部图，应用仍要按 tenant、actor、groups 和来源版本过滤。把服务账号凭证安全保存，只解决凭证问题，不解决返回内容的授权。课程 Context 入口应保留到所有图扩展、报告选择、map 与最终报告校验中。

## 小课十：全局回答评价需要覆盖与支持两条轴

覆盖率衡量所需主题或事实槽位中多少被正确回答。支持率衡量输出断言中多少有合格证据。两者不同：只输出一条正确风险，支持率可能一百但覆盖很低；输出所有主题加十条猜测，覆盖可能高但支持率低。全局系统必须同时报告，不能用答案长度或主题词数量替代。

对本章六槽位任务，若答案包含缓存风险、调度风险、权限风险与缓存例外，覆盖四除六约零点六六七。若其中权限风险引用错误来源，正确支持槽位只有三，严格覆盖应三除六。若另外添加一条不存在风险，断言支持精确率进一步下降。你应在标注规则里明确“提到”和“正确支持”不同，避免数字好看但答案不可靠。

```python
def evaluate_global(answer_rows, gold_slots, valid_supports):
    supported = []
    for row in answer_rows:
        slot = (row["topic"], row["kind"])
        if row["source_id"] in valid_supports.get(slot, set()):
            supported.append(row)
    covered = {(row["topic"], row["kind"]) for row in supported}
    return {"coverage": len(covered & set(gold_slots)) / len(gold_slots) if gold_slots else 0.0,
            "support_precision": len(supported) / len(answer_rows) if answer_rows else 0.0,
            "missing": sorted(set(gold_slots) - covered)}

wanted = {(topic, kind) for topic in ("cache", "scheduler", "access") for kind in ("risk", "exception")}
supports = {}
for item in corpus:
    supports.setdefault((item.topic, item.kind), set()).add(item.source_id)
flat_answer = [row for rows in global_answer["answers"].values() for row in rows]
assert evaluate_global(flat_answer, wanted, supports)["coverage"] == 1.0
bad_answer = flat_answer + [{"topic": "network", "kind": "risk", "source_id": "invented"}]
assert evaluate_global(bad_answer, wanted, supports)["support_precision"] == 6/7
```

这个评价器依赖人工 gold_slots 与 valid_supports，只验证来源归属，不理解自由改写文本是否蕴含。严格抽取版本中 text 来自原句，因此可作为组织实验；真实生成版本还要做断言级支持评价。不要把这个函数命名为 answer_is_true，它测的是一个明确受限契约。

此外记录重复来源比例、例外保留率、冲突保留率、越权断言数、过期引用数、在线延迟、离线索引费用与更新费用。一个方法可能提高覆盖却显著增加重建成本，是否值得取决于资料更新频率与查询频率。没有成本分解，就无法判断图组织是在帮助实际任务还是只增加离线复杂度。

## 小课十一：从错误传播链定位该修哪一层

图增强问答的错误可能来自原文解析、实体提取、实体对齐、关系抽取、社区划分、报告摘要、查询路由、map 选择、reduce 合并或最终引用。只看最终答案低分，很难知道该修哪里。应保存每层产物，构造逐层 oracle 实验：用人工正确实体替换自动实体，看下游改善多少；用人工报告替换模型摘要，看覆盖是否恢复。

如果正确边根本没进入图，调遍历深度没有用；如果边存在但实体链接错，增加社区报告长度可能更稳定地总结错误；如果报告完整但 reduce 丢掉例外，应改合并与预算策略；如果答案文本正确但引用错位，应修来源映射。分层对照能避免把所有问题都归结为模型能力不足。

错误还可能被层级摘要放大。底层一条含糊关系在社区报告中被概括成确定事实，上一层报告再引用这个确定表述，最终失去原始不确定性。每一层都流畅，却逐步增强了结论。报告应保留条件、争议与置信状态，并允许最终回答回查原始来源，而不是只引用摘要再引用摘要形成证据循环。

最有力的故障实验之一是删除唯一支持某风险的原文。正确系统的相关报告应失效或重建，最终答案不再无条件包含该风险。若答案完全不变，检查是否使用失效报告、模型背景知识或硬编码模板。再只删除重复转载，独立事实应仍被支持，但证据数量变化；这能检验系统是否正确区分独立来源与副本。

## 小课十二：十二道练习与完整答案

**练习一，方法选择。** 错误码含义优先精确检索；实体依赖适合局部图与原文；全语料主题需要分层组织与覆盖检查。完整答案必须说明图不是必需选项，文档分组与地图归约也是合理基线。选择依据是证据结构与成本，不是方法名称新旧。

**练习二，独立来源数。** 十二条资料中三条是重复副本，独立来源九个。按 source_id 计数会得到十二，按 original_id 去重得到九。若同一原始文档包含多个不同事实，不应把所有事实合并成一条；去重单位要同时考虑断言与原始来源。

**练习三，连通分量与社区。** 两个三角形加一条桥，连通分量是一，合理社区可能二。connected_components 与 modularity 的完整代码分别展示两种问题。不能把前者输出直接称为 Leiden 社区，也不能把模块度高解释为语义正确。

**练习四，模块度手算。** 总边七，每团内部三，总度七，Q 为两倍的三除七减四分之一，等于五除十四。全部一团为零。代码断言给出精确值。若去掉桥，总边六，每团内部三、总度六，Q 为二倍的二分之一减四分之一，等于二分之一。

**练习五，摘要依赖。** 缓存报告依赖 c1、c2、c3，c2 更新后整份报告按保守策略失效。不能只把依赖版本改成二而不重算正文，因为正文仍来自旧内容。完整函数 report_is_current 已提供，修改 visible 字典即可复现。

**练习六，权限撤回。** p1 不再可见，access 报告不能原样展示。删除引用标签但保留风险句仍泄露语义。完整处理是拒用该报告并在当前可见集合重建，或按断言依赖移除对应内容。两种方案都必须保留当前授权检查。

**练习七，map 丢例外。** 若 map 只取 risk，reduce 无法恢复 exception，missing 应包含三个例外槽位。运行 `map_report(report,{"risk"})` 后调用 reduce_maps，complete 为 False。模型生成层不能因为句子更完整就把缺失槽位标成功。

```python
risk_only = [map_report(report, {"risk"}) for report in reports]
partial = reduce_maps(risk_only, {"cache", "scheduler", "access"}, {"risk", "exception"})
assert partial["complete"] is False
assert len(partial["missing"]) == 3
assert all(kind == "exception" for topic, kind in partial["missing"])
```

**练习八，新增猜测。** 六条正确断言加一条无来源风险，覆盖仍一，支持精确率六除七。小课十的 bad_answer 断言是完整答案。这证明覆盖与支持不能只取一个指标，答案写得更多可能反而变差。

**练习九，图消融。** 固定语料、模型、总上下文预算，比较文本、文本加一跳、文本加两跳、社区报告。分问题类型记录覆盖、支持、例外保留、费用与时延。若只让图方案使用十倍上下文，无法判断改善来自图还是更多输入，必须控制或明确报告预算差异。

**练习十，更新成本。** 一条新桥边导致社区重新划分，可能影响多个报告。不能只重建新边所属原社区，因为成员变化跨组。完整答案是比较新旧成员与依赖指纹，标记所有变化报告，发布前检查完整性。小规模项目可全量重建作为正确基线，再优化增量。

**练习十一，重要性分数。** map 模型给九十分不表示风险有百分之九十概率或影响九十分。它是某提示规则下的排序信号，需要标注与校准。若用户要求严重程度，应依据影响、发生条件与证据，而不是直接把模型自评分显示成客观风险值。

**练习十二，术语边界。** 课程 BFS 证据图、连通分量分组、模块度计算、抽取式报告与完整 Microsoft GraphRAG 是不同能力集合。完整答案应逐项列出本章实现了什么、没有实现什么，并提供实际运行证据。不能把任意一条图路径查询宣传成完整生产 GraphRAG。

## 小课十三：从离线透明实验到模型摘要

第一步保持语料、主题与槽位标签固定，用本章抽取式 build_reports 和 map/reduce 跑完，得到覆盖与支持的上限基线。第二步只替换社区报告生成，map/reduce 仍按结构化槽位工作，观察摘要是否遗漏例外。第三步再替换 map 的问题相关选择，检查局部遗漏。第四步才让 reduce 生成自然语言，检查合并是否新增无来源结论。

每次只替换一层，才能定位变化。若一口气接入抽取、社区、报告、map、reduce 五次模型调用，最终回答好坏很难解释。你可以保留固定离线 transport 或录制的虚构响应作为回归输入，真实 API 运行单独记录模型与费用。离线测试证明控制流，真实评测证明特定模型配置在特定数据上的表现，两者不能互相冒充。

报告生成提示应要求保留主题、主要事实、条件、例外、冲突与来源 ID。结构化输出有利于验证，但语义仍需检查。来源 ID 必须来自输入集合，文本不得把推断伪装成直接证据。若允许外部常识，应该与语料支持部分分开标注；否则全局报告可能被模型先验填满，而用户以为它总结了自己的资料。

## 小课十四：真实项目的安装与阅读路线

官方文档在本次核查日期二零二六年十一月一日描述了索引流程、社区报告与局部和全局查询。具体接口会变化，安装前从官方入门与 CLI 文档确认当前版本要求，固定你实际安装的包版本和配置。不要把本章教学函数名当官方 API，也不要在未知版本上直接复制旧配置覆盖已有项目。

阅读入口为 [GraphRAG 官方主页](https://microsoft.github.io/graphrag/)、[索引概览](https://microsoft.github.io/graphrag/index/overview/) 与 [全局查询](https://microsoft.github.io/graphrag/query/global_search/)。官方全局查询采用社区报告上的地图归约组织，细粒度报告通常增加覆盖机会与模型工作量；这是方法说明，不是对你语料必然提高质量的保证。本章的代码和实验设计是独立教学实现。

安装实验使用独立虚拟环境和独立数据目录，只放十二份虚构资料。先运行帮助命令、记录版本、初始化配置，再检查模型与 embedding 设置、输入目录、输出目录和并发预算。真实索引可能调用模型多次，应在开始前明确允许的请求与费用上限，不把默认配置当无费用演示。当前教材验证没有替你运行这一付费链，因此实验状态应记录未执行，直到你实际完成并保存日志。

跑通后不要只看最终答案。检查文本单元、实体、关系、社区与报告等中间产物，抽样追踪一条回答如何回到原文。对照本章数据契约，记录官方实现在哪些地方提供字段、哪些来源与权限逻辑需要应用补齐。先做最小语料，再扩大规模，避免在还不知道错误来自哪一层时生成大量昂贵索引。

## 小课十五：完整项目的验收文件

项目至少输出五类文件：源资料与标签、索引构建清单、社区与报告依赖、查询轨迹、评价报告。清单包括语料指纹、抽取与摘要配置、模型标识、社区参数、构建时间和实际执行状态。查询轨迹包括选了哪些报告、map 产物、reduce 输入、最终引用与缺失槽位。没有这些中间证据，就很难比较两次运行为什么不同。

验收场景包括主题偏斜、重复转载、少数例外、冲突资料、权限混合、来源更新、社区划分变化和不可回答问题。每个场景都有固定预期，不要求自然语言逐字相同，但要求支持与覆盖契约相同。对模型输出可以采用人工断言标签和自动结构检查结合，明确哪些部分仍需人工判断。

最终报告允许得出“这个小语料不需要图”的结论。如果按主题分组已经取得同等覆盖且更便宜，选择简单方案是有效工程判断。图的价值应在关系连接、组织复杂性或全局覆盖中被实验展示，而不是因为你花了很多时间建图就必须证明它优越。

### 社区优化的直觉：移动、聚合与细化

理解社区算法，不必从数百行库代码开始。先设每个节点独立成组，尝试把一个节点移到邻居所在社区，计算目标值是否改善；反复移动直到局部没有明显改善；再把每个社区压缩成一个超级节点，社区之间的边成为超级节点之间的边，继续在更粗层次优化。这说明为什么算法可以产生层级，也说明结果可能受移动顺序与初始划分影响。

聚合时内部边不能凭空消失。超级节点的自环或内部权重代表原社区内部连接，跨社区边权应合并。若实现聚合只保留一条布尔边，会改变目标函数，后续优化不再等价于原图。权重与度数的定义必须沿各层一致，这也是阅读社区库源码时应追踪的数据契约。

局部优化可能陷入不理想划分，某些社区甚至内部连接质量较差。不同算法增加细化与连通性处理来改善这些问题。这里不把几十行贪心移动示例命名为完整 Leiden，因为真实算法的细化、随机性与保证需要严格实现。学习顺序应是理解目标、验证单次移动增益、理解聚合保持什么，再阅读完整算法，而不是只记住一个函数名。

你可以对六节点双三角形枚举若干候选划分，直接调用 modularity 比较，观察把桥两端节点合并是否有利。这个小实验不是搜索所有划分的高效算法，却能帮助理解结构目标的行为。再增加一条跨团边，重新计算，观察目标如何变化。通过手算，你会发现社区不是贴在节点上的永恒标签，而是相对于当前图结构与目标的结果。

### 分辨率、层级与摘要粒度的三方关系

分辨率控制倾向形成更小还是更大社区，层级反映从细到粗的组织，摘要长度决定每个社区保留多少信息。三者共同影响全局问答。很多小社区能保存局部例外，但报告数量多，map 调用更多；少数大社区便宜，却可能在摘要阶段压掉细节。仅调整最终上下文长度无法恢复底层摘要已经丢失的信息。

对本章十二份语料，可以比较三个粒度：全体一组、按三个主题分组、每个独立来源一组。全体一组容易形成短总览，但例外与主题平衡需要额外约束；主题组自然对应六槽位任务；逐来源组保真高但 reduce 工作更多。这个实验让你看到组织粒度与问题类型的匹配，而不是假设某个社区层级总是最优。

同一问题也可能先用粗报告定位主题，再下钻细报告与原文。这样降低初始输入，但必须防止粗报告遗漏主题导致永不下钻。可以保留覆盖抽样或多路线检索作为补充。动态选择粒度属于查询策略，应该记录选择依据与未访问范围，不能在只读了少量报告后声称总结了全部语料。

层级报告的来源链要能展开到原文。上层依赖下层报告，下层依赖 chunk；任一下层失效，上层也需失效或重建。若上层只保存下层文字，不保存版本依赖，更新传播就断了。把报告依赖画成有向无环图，可以用拓扑顺序重建，并检测意外循环引用，避免摘要互相引用而没有原始证据根。

### 文档、实体与社区之间的多重映射

一份文档可以提到多个实体，实体可以出现在多份文档，一个社区可以包含多个实体，同一文档可能关联多个社区。因此文档到社区通常不是简单一对一。若按文档数量统计主题覆盖，同一长文可能在多个社区重复计数；若按实体数量统计，命名密集的主题可能被过度代表。需要根据任务定义统计单位。

本章人工 topic 每条只有一个值，是为了让第一套实验可手算。真实系统应允许文档片段关联多个实体与主题，报告时保留映射。对于跨主题例外，例如权限问题影响缓存摘要展示，强行只归一个社区可能让另一个主题缺失重要条件。可以让相关断言出现在多个报告，但独立来源计数与最终合并要去重。

去重也不是把相同字符串全部删除。两份独立实测得到相同结论，是多个观察；三篇转载同一公告，是一个来源的多次传播；同一文档不同段落重复强调，可能是同一断言。应保存来源谱系与断言身份，最终按需要聚合。简单文本相似度只能提供重复候选，不能决定证据独立性。

### 完整覆盖实验：为什么前六条资料可能仍然不全面

将 corpus 按主题排序，假设一个局部排名返回 c1、c2、c3、c4、s1、s2。六条材料覆盖缓存风险、措施、例外与调度风险、措施，却完全没有权限主题，也没有调度例外。若任务是三主题风险与例外，正确支持槽位只有缓存风险、缓存例外、调度风险，共三除六。文档数量六与槽位覆盖一半不是同一个量。

主题分层策略从每组取风险与例外各一条，只需六条就覆盖全部六槽位。这个改善来自按任务组织，未必需要知识图。图社区可能在没有人工主题标签时提供组织候选，但它的质量要与人工分组上限和无图聚类基线比较。否则你可能把“使用了分层抽样”的收益全归给“使用了图”。

```python
def rows_from_sources(source_ids, items):
    selected = set(source_ids)
    return [{"topic": item.topic, "kind": item.kind, "source_id": item.source_id,
             "text": item.text} for item in items if item.source_id in selected]

local_rows = rows_from_sources(["c1", "c2", "c3", "c4", "s1", "s2"], corpus)
local_metrics = evaluate_global(local_rows, wanted, supports)
assert local_metrics["coverage"] == 0.5
stratified_rows = rows_from_sources(["c1", "c3", "s1", "s3", "p1", "p3"], corpus)
assert evaluate_global(stratified_rows, wanted, supports)["coverage"] == 1.0
```

这里 evaluate_global 的 support_precision 对措施条目也可能视为有来源支持，虽然它们不在用户要求的 gold_slots 中。这说明支持精确率与任务相关性还需分开：有依据但不相关的内容不算幻觉，却会浪费预算。真实报告可增加任务相关断言比例，或只对所需槽位计算相关支持。指标定义必须与问题匹配，不能因为一个函数输出多个小数就认为评价完整。

### 少数例外为什么容易在摘要里消失

摘要模型往往倾向提取频繁或显眼主题。如果九份资料重复一种风险，一份资料说明例外，普通总结可能只保留多数内容。但用户问的是条件与边界时，那一份例外可能最重要。频率是语料分布信号，不等于业务重要性，也不等于真实性。重复转载尤其会放大这种偏差。

可以在报告 schema 中强制设置 exceptions、conflicts、limitations 字段，要求没有时明确空数组。这样提高可观察性，但模型仍可能漏填，因此需要例外金标与删除对照。给同一语料分别保留和移除例外，检查报告是否相应变化；若完全不变，说明生成器忽略了关键来源。再把例外放在不同位置，测顺序与截断影响。

有些例外只限制某个范围，不应被改写成全面反驳。缓存风险与短请求可能不受影响可以同时成立。reduce 应保留“长上下文下”“在轻负载时”等条件，而不是用一句模糊“存在争议”糊过去。真正冲突是同一对象、属性、条件与时间下不同断言，条件不同的事实应并列说明。

### 全局问题中的“主要”需要操作性定义

用户问主要风险时，主要可能指提及频率高、影响严重、覆盖组件多、与当前任务最相关或近期最紧急。这些排序标准不同。语料提及次数高可能只是某团队写文档多，不代表风险更严重；图中心度高可能表示公共组件，不代表它最容易失败；模型评分高可能反映提示偏好。

项目应在输出中说明排序依据，或根据任务选择明确规则。例如课堂可以按三个预先标注主题各保留一项，避免伪造严重性排序；真实风险分析可以使用影响范围与证据支持的条件表，但需要领域定义。不要把纯文本汇总包装成未经验证的风险量化评估。

如果用户没有给出排序标准，可以提供按主题组织的总结，并说明当前资料不足以可靠比较严重程度。这不是拒绝完成任务，而是避免把资料分布误当现实概率。GraphRAG 的组织能力不能自动补充语料中没有的业务权重。

### 图与文本的组合不只有一种路线

一种路线是文本先行：检索片段识别实体，再图扩展。另一种是实体先行：从问题解析实体，直接查图与相关原文。第三种是社区先行：用主题报告定位范围，再回到实体与原文。第四种是并行路线：文本、图与报告分别提供候选，按来源身份融合。不同问题适合不同组合，应该在路由评测中比较。

文本先行对实体名不明确的自然语言问题较友好，但种子错了会带偏图；实体先行精确且便宜，但依赖实体链接；社区先行适合整体主题，却可能漏掉少数具体事实；并行路线覆盖广，但费用与去重复杂。没有一种组合天然取代所有其他路线。保留简单基线能帮助你判断新增模块是否值得。

融合时不能直接相加 BM25 分数、路径长度和社区重要性评分。可以先统一成排名或任务槽位，再用 RRF 或明确规则选择；最终都转换成带来源的证据对象。一个社区报告与一段原文有不同支持范围，不能只因为都返回 text 就视作同等证据。报告可以指引检索，关键事实仍应尽可能回到原文验证。

### 图更新的成本账本

离线构建成本包括解析、实体关系抽取、对齐、社区检测、报告生成、embedding 与存储。在线成本包括实体链接、检索、图扩展、报告选择、map 调用、reduce 调用与引用复核。更新成本取决于变更影响范围，可能只重算一个片段，也可能改变社区结构导致多个报告重建。把所有成本只写成一次 query 平均价格，会隐藏资料更新频繁时的真实负担。

可以用符号估算。若有 n 个文本单元，每个抽取平均一次调用，社区报告数为 c，每个一次，在线 map 批次数为 b，最终 reduce 一次，那么一次初建约 n 加 c 次模型调用，一次全局查询约 b 加一。实际流程可能有重试、多轮抽取与不同阶段调用，因此这只是预算上界设计的起点，不是产品固定收费公式。每个未知项都要在真实日志中测量。

对固定语料大量重复查询，离线报告成本可以摊薄；对每天大幅更新而查询很少的语料，重建成本可能主导。图方案是否划算取决于使用模式。个人学习项目应先用十二份文本把每阶段调用数记录清楚，再逐步增加到一百份，而不是一开始导入大量资料后才发现无法解释费用。

### 并发、重试与部分失败的处理

map 阶段天然可并行，但并行不是无限请求。需要并发上限、总调用预算、单请求超时与重试策略。若十个社区中两个失败，最终结果应标记覆盖缺口，不能把八个成功报告当全部语料。重试只针对可重试错误，格式错误或权限失败不应无限重试。取消任务后已经发出的调用仍可能计费，费用记录不能因取消而归零。

每个 map 任务绑定 report_id、report_revision、question_id 与配置指纹，结果按这些身份合并，不依赖返回顺序。异步响应先后可能不同，按列表位置 zip 会把某社区结果归给另一个社区。和 embedding 响应 index 一样，身份关联是并发正确性的基础。缺失结果应有明确状态，不能用空字符串假装成功。

reduce 只接受通过结构与来源检查的 map 结果。若某个 map 输出伪造 source_id，拒绝该结果并记录错误；若内容重复，按断言与来源合并；若同一槽位有冲突，保留冲突而不是按到达顺序覆盖。最终报告包含已覆盖与未覆盖范围，让使用者知道综合结论的实际边界。

### 一个可重放的 map 批次账本

可以把每个任务状态记录为 pending、running、succeeded、failed，保存输入报告版本、尝试次数、结果身份与错误类别。恢复时只重跑未完成任务，但先确认输入报告仍当前有效。若源资料变化，旧任务结果不能混入新构建，应启动新查询快照或明确重建受影响部分。状态持久化解决的是执行恢复，不是自动解决数据时效。

下面的离线函数演示部分失败如何进入最终覆盖报告。它不并行、不调用 API，只把失败与空结果分开。真实服务可以替换 execute，但不能改变账本语义。

```python
def map_with_ledger(reports, execute):
    ledger, outputs = [], []
    for report in reports:
        try:
            result = execute(report)
        except Exception as error:
            ledger.append({"report_id": report.report_id, "status": "failed",
                           "error_type": type(error).__name__})
        else:
            ledger.append({"report_id": report.report_id, "status": "succeeded",
                           "rows": len(result)})
            outputs.append(result)
    return ledger, outputs


def simulated_map(report):
    if report.topic == "access":
        raise TimeoutError("fictional timeout")
    return map_report(report, {"risk", "exception"})

ledger, outputs = map_with_ledger(reports, simulated_map)
partial_global = reduce_maps(outputs, {"cache", "scheduler", "access"}, {"risk", "exception"})
assert sum(row["status"] == "failed" for row in ledger) == 1
assert partial_global["missing"] == [("access", "exception"), ("access", "risk")]
```

异常只记录类型，避免将可能含内部地址或原始响应的错误全文传播到用户报告。这个捕获策略用于课堂可观察失败，真实代码应区分取消与系统退出等控制信号，并保留受控内部日志。成功但返回空列表与失败不同：前者表示执行器认为该报告无相关点，后者表示没有得到可靠判断。

### 摘要缓存与查询缓存应该分开

社区报告通常与问题无关，可被多个查询复用；map 结果与具体问题相关；最终答案还依赖输出要求与权限快照。把三者用一个 query 字符串缓存，会混淆生命周期。报告缓存键应绑定内容依赖与生成配置，map 缓存再加问题与选择规则，答案缓存再加最终渲染与授权语义。

缓存命中仍需要复核当前依赖。旧报告文本完全相同但某来源被撤权，不能因为 hash 没变就继续使用；同一用户权限组变化，也可能使此前答案不可用。可采用保守的全依赖检查，或维护撤权版本与依赖索引。性能优化的前提是知道缓存复用意味着哪些条件仍成立。

错误缓存尤其会放大系统缺陷。一份漏例外报告被复用一千次，比一次生成错误影响更大。缓存发布前应通过结构与来源检查，关键回归样本可以保存为评测集。不能因为报告是离线生成就降低验证标准，离线错误通常有更长生命周期。

### 真实官方实现的产物阅读顺序

官方索引概览说明流水线会产生结构化表，并把 embedding 写入配置的向量存储。学习时先找文本单元表，确认源文档与片段身份；再找实体与关系，检查提及如何聚合；再看社区成员与报告，追踪报告依赖；最后看查询上下文构造如何选取这些产物。不同版本字段和文件组织可能变化，所以应以实际安装版本为准，不在教材中猜测固定列名。

每读一张表，挑一条记录回到原文。实体描述是否把多个来源合并，关系方向是否正确，社区报告是否有无法定位的总结句，报告引用能否映射回当前资料。这个逐项追踪比只运行官方 notebook 更能建立代码级理解。你可以在副本上删除一条关系，观察哪个下游步骤需要重建，但不要修改唯一原始实验记录。

全局查询源码中重点找上下文批次、map 提示、局部结果结构、过滤排序、reduce 输入与最终引用处理。局部查询则重点找实体候选、邻域材料与文本单元选择。不要把不同模式混成同一条调用链，否则调参数时会改错层。记录函数名与 commit 后，才能让未来复习定位到相同实现。

### 公平比较的实验表应该怎样写

至少比较四个方案：相同总预算的文本 top-k、按人工主题分层的抽取基线、图局部扩展、社区报告全局聚合。人工主题不是可部署自动方案，但它提供组织上限，帮助判断图社区质量；文本方案是成本较低的现实基线；局部与全局图方案服务不同问题。把所有问题平均在一起可能掩盖差异，因此按精确查找、关系连接、全局主题、不可回答分组。

固定原文、权限、问题与答案评价规则。若模型不同、上下文预算不同、允许外部常识不同，要明确列出，不能把差异全归给检索架构。费用与延迟至少重复若干次观察分布，但课堂小样本不应报告过度精确百分比。人工评价最好隐藏方案标签，减少“复杂系统应该更好”的预期偏差。

失败样本比单一平均分更有价值。挑出图方案改善的两题与退化的两题，逐层解释原因。改善可能来自关系连接，退化可能来自错误实体合并；覆盖提升可能来自分层组织，支持下降可能来自摘要泛化。这样的报告能指导下一步修复，而不是只宣布某方法胜出。

### 四个故障工作坊的修复答案

第一类是主题垄断。缓存文档重复多，top-k 与社区重要性都偏向缓存，权限主题消失。修复先按 original_id 去重，再增加按主题覆盖的选择约束，最后用六槽位评价确认。若图社区把所有主题混在一起，检查高频泛实体与错误桥边，不要只加长最终答案。

第二类是例外丢失。社区报告将“长上下文有风险，短请求可能无压力”压成“所有请求都有缓存压力”。修复报告 schema 与支持检查，保留条件与例外；回归同时包含风险和例外原文，要求最终措辞不扩大范围。把例外仅放在引用列表而正文仍绝对化，不算修复。

第三类是权限洗白。私有关系进入公开社区摘要，之后普通用户通过摘要看到。修复依赖授权检查或按可见范围重建，撤销旧报告缓存。验收普通用户在所有入口都看不到私有断言，包括报告搜索、图视图与最终回答，而管理员仍能按策略使用。不能通过删除全部报告而不报告功能损失来假装完成精细授权。

第四类是更新断链。原文已改，图边已更新，社区报告仍旧。修复建立 chunk 到 edge 到 community report 到 answer cache 的依赖索引，更新时标记受影响产物。验收删除唯一支持来源后相关断言消失，恢复新来源后以新版本出现。旧来源 ID 不能在最终报告中继续有效。

### 自己实现与阅读开源之间的连接任务

将本章 CommunityReport 的每个字段映射到真实实现中的对应信息。如果某个字段没有直接对应，例如完整权限依赖或原始来源独立性，写清需要应用补齐，不要假定框架隐藏实现了。再把本章 map_with_ledger 的失败状态映射到真实任务日志，检查部分失败是否会影响最终输出完整性。

选一个小改动做源码任务：增加一个报告输入统计、在最终结果中暴露已处理社区数、或为例外字段增加结构校验。改动应有明确触发与前后行为，用十二份语料测试即可。不要一开始修改社区算法核心并同时更换模型，这会让故障原因难以定位。代码阅读能力来自能预测一个小改动如何流经系统。

进一步任务是实现报告依赖失效的小服务。输入来源变更事件，输出需要重建的报告 ID；重复事件应幂等；旧事件不能覆盖新代际；撤权事件优先阻止使用旧报告。这个任务连接 R08 生命周期与 R11 聚合层，价值比只换一个更大的摘要模型更基础。

### 对“全局理解”的谨慎解释

系统回答覆盖了全部社区，不等于理解了全部资料。社区可能漏抽实体，报告可能压缩细节，map 可能误选，reduce 可能泛化。全局在这里描述组织范围，不是全知保证。报告应说明语料范围、时间、处理成功率与缺失主题，让用户判断结论适用边界。

同样，引用很多来源不等于每个断言都被支持。一个长段落末尾堆十个引用，读者仍不知道哪个来源支撑哪句话。更好的输出把断言与来源就近关联，条件与例外也有对应证据。对于综合推断，说明它由哪些事实共同支持以及仍有哪些不确定性，而不是把推断伪装成原文直接结论。

本章最终能力是设计一个可以拆开检查的全局问答系统：知道资料怎样组织、摘要怎样生成、依赖怎样失效、查询怎样覆盖、错误怎样定位。学会这些以后，才有基础判断某个新的 GraphRAG 变体改善了哪一层，以及它的论文指标是否与你的任务一致。

### 综合演练：发布一个新社区报告版本

假设缓存例外从“短请求可能避免压力”改成“只有隔离队列中的短请求才可能避免压力”。首先源文档产生新 revision，旧 chunk 失去当前资格；其次依赖旧 chunk 的关系与缓存报告失效；然后在固定新快照上重建报告，确保新增条件保留；最后发布新报告代际，查询缓存依据旧报告版本失效。每一步都有对象身份，不能只把报告文本搜索替换几个字就跳过依赖更新。

如果重建模型超时，旧报告不应继续冒充当前有效。可以暂时回退原文检索，在答案中明确当前综合报告未更新，或者返回部分结果。回退策略应保持授权与引用检查，不因为故障就扩大数据范围。恢复后再原子切换报告指针，避免用户读到一半旧报告一半新报告。

验收用同一个问题查询更新前后。前者允许旧条件，后者必须包含隔离队列限定；旧来源引用应失败；其他两个主题结果不应无故改变。这个局部变化测试比全量重建后看一篇新报告更敏感，能发现依赖粒度过粗、缓存失效不足或随机生成导致无关漂移。

### 综合演练：一条新边改变两个社区

原图有缓存和调度两个社区，新资料明确说明调度器的准入策略直接依赖缓存容量。新增边可能使算法调整边界，或在粗层级合并两个社区。系统不能把旧社区编号当永久主题 ID，因为编号相同未必成员相同，编号不同也未必主题完全不同。应比较成员与来源指纹，生成新的社区版本。

报告重建时保留跨主题关系，但不要让合并抹掉各自例外。缓存例外与调度例外仍属于不同条件，综合报告应解释二者连接，而不是把它们平均成一条泛泛的性能风险。查询评价仍使用原有槽位与新增跨主题关系槽位，这样能看到结构变化是否真正增加答案能力。

如果新边后来被撤销，应能恢复相应结构与报告。无需保证社区编号回到原数值，但规范来源支持与任务结果应符合当前图。测试应比较语义成员与证据集合，而不是只比较算法标签字符串。随机算法的稳定性也需记录种子与配置，必要时比较多个运行的分组一致性。

### 本章提交前的最后检查

请打开最终报告，随机挑三个结论，分别向后追踪到 reduce 输入、map 结果、社区报告、关系或文本单元、原文。任何一层找不到身份或版本，就记录为依赖缺口。再选一个例外，检查它没有在某层被删除或改成普遍结论。这个反向追踪训练能把抽象的 provenance 变成真实操作。

最后检查实验结论的措辞。若只运行十二份虚构资料，应写课堂数据上的行为；若没有真实模型调用，应写组织与状态逻辑验证；若只计算模块度，应写目标函数演示；若没有部署权限服务，应写本地 Context 模拟。准确范围不会削弱成果，反而让后续扩展知道哪些能力已经有证据、哪些仍需要新的实验。

提交时附一个失败样本的完整反向追踪，比只提交成功总览更有说服力。指出错误首先出现在哪个产物、为什么后续检查没有拦住、修复后哪个断言发生变化。这样未来更换模型或社区算法时，你仍有一组具体行为可以比较，而不必重新凭印象判断系统是否更聪明。

请把每次实验中真正执行的模型调用与离线替身明确分栏，保证以后比较结果时不会把控制流验证误当成生成质量验证。
