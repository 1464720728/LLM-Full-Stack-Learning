# R10 知识图谱：从一句原文到可追溯的关系路径

先修 R01 的证据对象、R08 的生命周期，以及 Python 字典、集合、队列。本章从手工建模开始，依次完成实体、关系、出处、抽取、对齐、存储、查询与遍历。所有必做实验都在 CPU 上运行。课程实现是有证据的有向关系图，不是 Microsoft GraphRAG 的完整复刻，也不是自动逻辑证明器。

## 小课一：图的箭头先有含义，才有算法

工作流图的节点是任务，箭头表示执行依赖；计算图的节点是张量或运算，箭头表示数值依赖；知识图的节点是世界中的对象或概念，箭头表示某种关系。三个图都能画成圆圈和连线，但查询含义不同。Agent 指向 Retrieval，在工作流中可能表示下一步调用检索，在知识图中可能表示某个系统使用检索组件。后者不证明每次请求都会调用它。

图的数学表示通常写作 G 等于 V 与 E 的组合。V 是节点集合，E 是边集合。有向图的边从一个起点指向终点；有类型关系还包含 predicate。课程边是 subject、predicate、object 加来源等元数据。同一对实体可以有 uses 和 depends_on 两种不同边，也可以有同一关系的多个来源，因此它更接近有向多重关系图，而不是只有零一邻接矩阵的简单图。

先想一个虚构系统：Jade 服务使用 Quartz 插件；Quartz 插件使用 Cedar 索引；Cedar 索引读取文档仓库。用户问 Jade 的知识更新依赖哪些组件。文本检索可能找到每句话，图把连接显式保存，方便沿着实体追踪。但如果第一条关系抽错，后面的正确索引文档仍会被接到错误服务上。图提升的是关系组织与查询能力，不自动提高来源真实性。

建立图前先写问题清单。如果只需要查错误码解释，精确文本索引可能足够；如果需要问谁依赖谁、某组件变化影响哪些上游、哪些路径连接两个对象，图更有用。为了“用上知识图谱”而把所有名词都变成节点，常得到一张很大却没有明确查询语义的图。模型设计应由需要回答的问题反推。

## 小课二：实体身份、名称、提及与别名

实体是我们想持续指认的对象，例如项目 Jade 的调度器。名称是展示标签，可能叫 Scheduler；提及是文本中一次出现的具体字符串及位置；别名是同一实体的其他表达。四者不能混用。两份文档都出现 Scheduler，不代表它们指同一模块；同一实体在不同文档中叫调度器或 request scheduler，也不代表是两个对象。

实体 ID 应稳定且包含必要命名空间，如 `jade:component:scheduler` 与 `quartz:component:scheduler`。展示名称可以相同。版本是否进入实体 ID 取决于问题：如果需要比较组件各版本结构，可以把版本作为属性或独立版本节点；如果把所有版本合并成一个实体，关系必须带有效版本条件。不能一边合并所有版本，一边把旧关系当当前事实。

提及应该保存文档版本、chunk ID、字符区间和表面字符串，便于重新检查抽取。实体链接把提及映射到某个实体 ID；实体消歧解决一个提及可能指多个对象；实体对齐判断不同来源中的对象是否同一。名字规范化只处理表示差异，如大小写与空白，不等于语义对齐。把所有 lowercase 相同名称合并，会把不同项目的同名模块连成假路径。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    entity_id: str
    namespace: str
    kind: str
    name: str
    aliases: tuple[str, ...] = ()

entities = [
    Entity("jade:component:scheduler", "jade", "component", "Scheduler", ("调度器",)),
    Entity("quartz:component:scheduler", "quartz", "component", "Scheduler", ("调度器",)),
]

def resolve_name(name, namespace, catalog):
    normalized = name.strip().casefold()
    candidates = [entity for entity in catalog
                  if entity.namespace == namespace
                  and normalized in {entity.name.casefold(), *(alias.casefold() for alias in entity.aliases)}]
    if len(candidates) != 1:
        return None
    return candidates[0].entity_id

assert resolve_name(" Scheduler ", "jade", entities) == "jade:component:scheduler"
assert resolve_name("Scheduler", "unknown", entities) is None
```

函数在候选不是唯一时返回 None，而不是取第一项。空结果与歧义结果在这个简化接口里合并，真实系统可以返回不同状态与候选列表。`casefold` 只用于名称匹配，不修改实体 ID，也不证明两个别名在任何领域都等价。别名目录需要来源与管理策略，不能让模型随意创建 alias 后自动合并整张图。

## 小课三：谓词是一个契约，不是随便写个动词

uses 表示使用关系，depends_on 表示依赖，part_of 表示组成，supports 表示支持某能力，supersedes 表示替代旧对象，implemented_by 表示由某实现承担。课程把谓词限制在受控集合，便于查询与测试。受控词表不是形式逻辑公理，也不保证自然语言抽取正确；它只是让关系类型有一致名字与方向。

每个谓词最好定义 domain 与 range，即允许的起点类型与终点类型。例如 component uses component，component part_of system，implementation supports capability。若人物节点 part_of 一个数值，可能是抽取或类型建模错误。类型约束不能发现所有语义错误，但能过滤一批明显不合理结构。课程 EvidenceGraph 没有实体类型表，所以类型检查由本章示例层演示，不能声称底层已经强制执行。

方向必须固定。A uses B 的反向查询是“谁使用 B”，不是把边反过来仍叫 uses。若需要显式逆谓词，可以定义 used_by 并说明它与 uses 的对应；否则反向邻接表只改变索引键，边本身仍保留原方向。把所有关系当无向边会让依赖传播产生荒谬结论，例如库使用应用。

传递性也应逐谓词定义。严格包含关系在特定模型中可能可传递，uses 通常不能直接推出同样语义的远程使用；supports 更不能沿 uses 自动传递。A 使用 B，B 支持 GPU，不证明 A 的当前部署支持 GPU。图路径只说明边序列存在，是否能推导某个新断言，需要额外规则与适用条件。

```python
SCHEMA = {
    "uses": ({"component", "system"}, {"component", "service"}),
    "part_of": ({"component"}, {"system"}),
    "supports": ({"component", "system"}, {"capability"}),
}

def validate_relation_types(subject, predicate, obj):
    if predicate not in SCHEMA:
        return False
    domains, ranges = SCHEMA[predicate]
    return subject.kind in domains and obj.kind in ranges

system = Entity("jade:system", "jade", "system", "Jade")
component = entities[0]
capability = Entity("cap:gpu", "global", "capability", "GPU execution")
assert validate_relation_types(system, "uses", component)
assert validate_relation_types(component, "supports", capability)
assert not validate_relation_types(capability, "part_of", system)
```

## 小课四：事实、断言与证据必须能分开保存

最简单三元组只有主语、谓词、宾语，但可信系统至少还需要断言身份、来源片段、引文、时间或版本、状态与可能的置信标记。两个来源说同一关系，可以作为两个证据断言支持同一规范关系；一个来源否认关系，应作为冲突而非直接覆盖。把所有来源压成一条无出处边，会丢掉更正与删除能力。

课程 Edge 保存 edge_id、tenant、subject、predicate、object、chunk_id、quote。它没有独立有效时间与真假状态，因此样例只手工添加当前来源中确认的肯定关系。`EvidenceGraph.add` 检查引文是当前可见片段的非空子串，这证明来源位置存在，却不证明该引文蕴含所填三元组。来源“Jade 不使用 Quartz”也包含两个实体名，结构验证不能理解否定。

因此可以把流程分成 candidate assertion 与 accepted assertion。模型抽出的关系先进入候选，经过 schema、实体链接、来源区间、否定条件与抽样语义审查后才进入可查询图。对高要求领域，可能需要人工确认；对探索性检索，可以允许未确认边帮助找资料，但不能把它当已证实事实写入最终答案。状态与用途必须明确。

同一关系的多个来源也不一定独立。三篇转载同一公告，不能简单算三票。可以记录 original_source_id 或引用链，把重复传播与独立观察区分开。置信度聚合若假设来源独立，转载会让数值虚高。图中边数更多不必然意味着证据更强，可能只是重复文本更多。

## 小课五：从一份原文建立第一张有出处的图

下面用课程真实 API 建立三条边。先摄入三份独立文档，每份一句关系，使删除某一来源时可以清楚观察哪条边失效。使用公开授权，但仅在虚构租户 graph-demo 内公开。实体名用明确命名空间，避免后续同名串线。

```python
from labs.knowledge.store import CorpusStore, Context
from labs.knowledge.graph import EvidenceGraph

with CorpusStore() as store:
    texts = {
        "jade-plugin": "Jade uses Quartz.",
        "plugin-index": "Quartz uses Cedar.",
        "index-corpus": "Cedar uses Archive.",
    }
    for doc_id, text in texts.items():
        store.ingest("graph-demo", doc_id, doc_id, text, allowed=("*",))
    ctx = Context("graph-demo", "learner")
    chunks = {chunk.doc_id: chunk for chunk in store.visible(ctx)}
    graph = EvidenceGraph(store)
    graph.add(ctx, "jade:service", "uses", "jade:quartz", chunks["jade-plugin"].chunk_id, texts["jade-plugin"])
    graph.add(ctx, "jade:quartz", "uses", "jade:cedar", chunks["plugin-index"].chunk_id, texts["plugin-index"])
    graph.add(ctx, "jade:cedar", "uses", "jade:archive", chunks["index-corpus"].chunk_id, texts["index-corpus"])
    path = graph.path(ctx, "jade:service", "jade:cedar", max_depth=2)
    assert path is not None and len(path) == 2
    assert [edge.object for edge in path] == ["jade:quartz", "jade:cedar"]
    assert graph.path(ctx, "jade:cedar", "jade:service", max_depth=3) is None
```

逐行看状态：摄入产生版本一与片段；chunks 字典按 doc_id 找来源；graph.add 先通过 get_visible_chunk 检查当前授权，再计算边 ID 并 INSERT OR IGNORE。相同关系、同来源、同引文重复添加会得到相同 ID，不产生第二行；同关系不同来源则可有另一条边。这是断言级身份，不是仅按三元组去重。

`path` 返回有序 Edge 列表，而不是节点名字字符串。每条边仍带 chunk_id 与 quote，最终展示可以逐边复查。起点到自身返回空列表，表示零跳路径；找不到返回 None。两者都在 Python 布尔上下文中为假，所以调用者应使用 `is None` 区分。这个小细节会影响“已经在目标”和“不可达”的报告。

## 小课六：邻接表、反向表与关系索引

邻接表把每个起点映射到出边列表。查询一个节点的下一步时，只需要读它的列表，而不扫描全部边。反向表把终点映射到入边列表，回答谁指向它。若经常按关系类型查，可以使用 `(subject,predicate)` 作为索引键。不同索引服务不同查询，没有一个结构自动适合所有问题。

```python
from collections import defaultdict

def adjacency(edges, reverse=False):
    result = defaultdict(list)
    for edge in edges:
        key = edge.object if reverse else edge.subject
        result[key].append(edge)
    return {key: sorted(value, key=lambda edge: edge.edge_id)
            for key, value in result.items()}
```

函数不修改边，反向时只是把 object 用作键。排序让遍历顺序稳定，避免同样图因数据库返回顺序不同而选到不同的等长路径。稳定顺序有助于测试，但不表示按 edge_id 排序在语义上最好；若要优先更可信来源，需要显式权重和对应路径算法。

如果用邻接矩阵，n 个节点需要 n 乘 n 个位置，稀疏大图会浪费空间；邻接表空间与节点和边数量大致线性。课程图通常很稀疏，邻接表更直观。多关系、多来源需要保留多条边，不能把矩阵一个布尔值当完整知识表示。矩阵适合某些数值算法，证据账本仍需要详细记录。

## 小课七：BFS 的队列状态逐步手算

广度优先搜索从起点出发，先处理零跳，再处理一跳，再处理两跳。所有边代价相同且只求最少边数时，首次到达目标得到最短跳数路径。最短不是最可信，也不是最相关，更不是最终答案最正确。你必须先知道优化目标，才能选择算法。

设边为 A 到 B、A 到 C、B 到 D、C 到 D、D 到 E。初始队列只有 A 的空路径，visited 为 A。弹出 A，把 B 与 C 入队，visited 加 B、C；弹出 B，发现 D，入队路径 A-B-D；弹出 C，D 已访问，不重复入队；弹出 D，发现 E，返回 A-B-D-E。若邻接顺序先 C 后 B，则得到另一条同长路径，长度仍三。

visited 在入队时标记，可以防止同一节点从多个父节点重复入队；若在出队时才标记，队列可能包含多个重复状态。对普通无权最短路，第一次发现节点时路径长度已最短，所以这样安全。若状态还包含已用关系模式、剩余资源或时间条件，只按节点去重可能错误丢弃另一种可行状态，需要把完整状态纳入 visited。

```python
from collections import deque

def bfs_trace(adjacency_map, start, target, max_depth=3, max_expansions=20):
    if max_depth < 0 or max_expansions <= 0:
        raise ValueError("invalid budget")
    if start == target:
        return [], [{"event": "already_target"}]
    queue = deque([(start, [])])
    visited = {start}
    trace = []
    expansions = 0
    while queue and expansions < max_expansions:
        node, path = queue.popleft()
        expansions += 1
        trace.append({"event": "expand", "node": node, "depth": len(path), "queue_size": len(queue)})
        if len(path) >= max_depth:
            continue
        for neighbor in sorted(adjacency_map.get(node, [])):
            candidate = path + [(node, neighbor)]
            if neighbor == target:
                return candidate, trace
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, candidate))
    trace.append({"event": "stop", "reason": "budget" if queue else "exhausted"})
    return None, trace

small_graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": ["E"], "E": ["A"]}
route, events = bfs_trace(small_graph, "A", "E", 3)
assert route == [("A", "B"), ("B", "D"), ("D", "E")]
assert bfs_trace(small_graph, "A", "E", 2)[0] is None
assert bfs_trace(small_graph, "A", "A", 0)[0] == []
```

每次出队算一次 expansion，不是每条扫描边算一次。因此一个高度节点即使只出队一次，也可能扫描百万邻居；max_expansions 不能单独限制全部工作量。生产图遍历还需要边扫描数、候选数、时间与输出大小上限。课程 path 的 max_depth 与 max_expansions 是基础预算，不是完整资源隔离。

## 小课八：为什么加权路径与受约束路径需要不同状态

如果每条边有代价，BFS 不保证总代价最小。A 直接到 E 的边代价十，A-B-E 两条边各一，BFS 会先找到一跳总代价十，但最小代价路径是二。非负权重最短路可用 Dijkstra，按累计代价优先出队；存在负权重则需要其他算法与环检查。把置信度简单取负当代价可能引入负边，不能随便套 Dijkstra。

若边权是可信度，路径可信度如何合成也不是纯算法问题。乘积假设各边正确性近似独立，但共同来源会相关；最小边权表示瓶颈保守估计，也不是严格概率。课程不提供“路径真假概率”，只返回有出处的边序列。你可以比较启发式评分，但报告应说明定义与假设，避免一个零点八七数字造成过度信任。

受关系模式约束时，状态还要保存模式进度。例如查询要求先 uses 再 part_of。到达同一节点时，如果一个路径已经完成 uses，另一个没有，它们的未来允许边不同。只用节点 visited 会把其中一种状态丢掉。正确状态可以是 `(node, pattern_position)`。这是图算法与 Agent 状态设计的共同主题：状态要包含决定下一步合法动作所需的信息。

### 带关系模式的完整参考实现

下面使用简单四元组 `(edge_id, subject, predicate, object)`，把状态扩展成节点与模式位置。它只接受恰好按给定谓词序列行走，不做任意正则路径，也不处理来源权限；真实使用时必须先传入经过 EvidenceGraph.visible_edges 过滤的边。示例的目的在于证明为什么 visited 的键不能永远只是实体名。

```python
def pattern_path(edges, start, target, pattern):
    adjacency_map = {}
    for edge in edges:
        adjacency_map.setdefault(edge[1], []).append(edge)
    queue = deque([(start, 0, [])])
    seen = {(start, 0)}
    while queue:
        node, position, path = queue.popleft()
        if position == len(pattern):
            if node == target:
                return path
            continue
        for edge in sorted(adjacency_map.get(node, [])):
            if edge[2] != pattern[position]:
                continue
            state = (edge[3], position + 1)
            if state not in seen:
                seen.add(state)
                queue.append((edge[3], position + 1, path + [edge]))
    return None

pattern_edges = [
    ("e1", "app", "uses", "plugin"),
    ("e2", "plugin", "part_of", "platform"),
    ("e3", "app", "part_of", "platform"),
]
assert [edge[0] for edge in pattern_path(pattern_edges, "app", "platform", ["uses", "part_of"])] == ["e1", "e2"]
assert pattern_path(pattern_edges, "app", "platform", ["uses", "uses"]) is None
```

位置从零开始，表示下一条应该匹配 pattern 的第零项；走一条合法边后加一；到达长度时才检查目标。这样“已经到目标但关系模式没走完”不算成功，“模式走完但到错节点”也不算成功。这个程序展示约束进入状态后的行为，比在 BFS 最后对路径做一次过滤更可靠，因为后者可能先找到不合格最短路就错误停止。

## 小课九：从原始文本抽取关系，先做一个能解释的基线

关系抽取至少包含提及识别、实体链接、谓词判断、方向判断、条件与否定识别、证据定位。模型一次输出三元组可以把这些步骤藏在内部，但工程仍要分别评估。若 subject 错了，可能是提及识别；名字对而实体错，是链接；实体都对但 uses 与 part_of 混淆，是谓词；来源存在却否定没识别，是语义范围。不同错误需要不同修复。

先使用受控句式，例如“Jade uses Quartz.”。规则抽取器可以精确匹配这类句子，保存原文区间，把名字通过实体目录解析成 ID。它的召回很有限，遇到“Jade relies on Quartz for retrieval”就不抽取，这是诚实的边界。不要为了看起来全覆盖，把任何同句出现的两个实体都连 uses；那会提高边数却严重降低关系精确率。

```python
import re

RELATION_PATTERN = re.compile(r"(?P<subject>[A-Za-z][A-Za-z0-9_]*) uses (?P<object>[A-Za-z][A-Za-z0-9_]*)\.")

def extract_controlled(text, aliases):
    proposals = []
    for match in RELATION_PATTERN.finditer(text):
        subject_name, object_name = match.group("subject"), match.group("object")
        if subject_name not in aliases or object_name not in aliases:
            continue
        proposals.append({"subject": aliases[subject_name], "predicate": "uses",
                          "object": aliases[object_name], "start": match.start(),
                          "end": match.end(), "quote": match.group(0)})
    return proposals

aliases = {"Jade": "jade:service", "Quartz": "jade:quartz", "Cedar": "jade:cedar"}
proposals = extract_controlled("Jade uses Quartz. Quartz uses Cedar.", aliases)
assert len(proposals) == 2
assert proposals[0]["quote"] == "Jade uses Quartz."
assert extract_controlled("Jade does not use Quartz.", aliases) == []
```

正则只匹配严格句式，因此否定例子不匹配，但这不构成完整否定理解。“It is false that Jade uses Quartz.” 仍包含可匹配子串，规则会错误抽取。真实摄入若接受任意文本，必须增加句子级范围分析或把抽取候选交给审查。这个反例很重要：一个通过简单否定测试的规则，仍可能被更复杂语法击穿，不能从少数测试推出通用语义能力。

字典 aliases 在示例中是人工确认的单一命名空间。若同一字符串可能对应多个实体，应返回候选而不是固定映射。抽取结果保留 start 与 end，便于检查 text[start:end] 等于 quote；课程图 API 最终保存 quote 与 chunk_id，没有保存区间，因此若同一句重复出现只能证明子串存在。需要精确定位时可扩展边证据结构，但应同步更新身份与验证契约。

## 小课十：模型抽取的 JSON 合法性只是第一道门

一个模型提案可以包含 subject_id、predicate、object_id、source_id、quote、polarity、condition。执行器首先限制字段集合与长度，检查谓词词表、实体 ID、来源当前可见与引文范围。之后检查 polarity 是否肯定、condition 是否适用于当前图。如果模型自己把不确定条件删掉，schema 不会发现，因此还需要带标注的语义抽取评估与抽样复核。

不要允许模型在实体不存在时直接创建任意 ID 并立刻连入已验证图。可以先创建待解析提及，再经目录匹配或管理确认建立实体。否则同一个实体会因不同拼写生成许多节点，或不同实体因同一个熟悉名字被错误合并。开放世界抽取确实需要新增实体能力，但新增与合并是高影响操作，应该有来源与可撤销记录。

模型输出置信度可以作为待审查排序信号，不能直接当真实性概率。它可能对错误否定关系非常自信，也可能对专业术语正确关系保守。校准需要独立标签，按类别检查。关系抽取评价应分别记录主语、谓词、宾语、方向、条件、来源支持是否正确，不能只用三元组字符串相似度给一个总分。

把抽取作为可重放批处理也很重要。记录源版本、模型标识、提示模板版本、schema 版本与提案。升级抽取规则后，可以比较同一原文哪些边变化。若只保存最终图，无法知道新边来自资料更新还是模型行为变化。图是派生产物，抽取运行本身也需要版本身份。

## 小课十一：实体对齐，从精确匹配到候选审查

最简单对齐是规范 ID 相同；其次是受控别名完全匹配；再往后可以使用属性相似、上下文相似、向量候选或外部标识。候选生成追求不漏可能对象，最终合并追求高精确率。两者应分开。把向量相似度高于某阈值直接当 same_as，会把经常一起出现但不同的组件合并，例如缓存管理器与缓存分配器。

对齐证据可以包括项目命名空间、实体类型、版本、仓库路径、稳定外部 ID、维护者与上下文关系。相同名字但不同 namespace 通常先视为不同；不同名字但同一稳定外部 ID 更可能是别名。仍要考虑对象更名、拆分、合并与继承：一个新项目取代旧项目，不一定是同一实体，可能应该用 supersedes，而不是 same_as。

一旦合并错误，影响可能扩散到所有入边和出边，因此需要可逆映射。不要直接删除原始实体与提及，只保留合并后的名字。可以保存 canonical_id 与 alias assignment 的版本、证据和决策者，撤销时重新解析派生图。小型项目先人工确认少量别名，远比自动合并几千节点后无法解释更适合学习。

实体评估可以用成对标签：两次提及是否同一对象。准确率在大量负例下会虚高，应关注错误合并和漏合并。对于知识问答，错误合并常比漏合并更危险，因为它制造不存在的路径；漏合并则可能降低召回，但较容易通过候选扩展补救。这个不对称性应影响阈值与人工复核策略。

## 小课十二：图查询如何返回证据，而非只返回名字

局部查询“Jade 使用什么”可以查 subject=Jade、predicate=uses 的边，返回 object 与每条来源。反向查询“谁使用 Cedar”按 object 查入边。两跳查询返回路径中每条边，不能只返回终点后声称“Jade 直接使用 Cedar”。聚合查询如“哪些组件被最多服务依赖”要定义去重单位：按边数、独立来源数、不同上游实体数，结果可能不同。

如果同一关系有三份来源，度数按边数统计会把它算三次。对结构分析，应先把规范三元组去重；对证据审计，应保留三份来源；对可信度估计，还要区分来源是否独立。数据视图不同，统计意义不同。不要直接对 evidence edges 做 degree 后把高值节点称为核心组件，而不检查重复来源影响。

图路径的报告应包含起点、终点、关系序列、每条边的原文、来源版本、约束与未证明事项。例如“Jade uses Quartz；Quartz uses Cedar”可以报告为两跳依赖线索，而不是自动改写为“Jade 完全继承 Cedar 的所有更新能力”。若最终回答需要某能力传递，还要找到配置或实现证据。

```python
def explain_path(path):
    if path is None:
        return {"status": "not_found", "steps": []}
    return {"status": "zero_hop" if not path else "found",
            "steps": [{"subject": edge.subject, "predicate": edge.predicate,
                       "object": edge.object, "source_id": edge.chunk_id,
                       "quote": edge.quote} for edge in path],
            "limitation": "A path is not an automatic transitive fact."}
```

这个函数只格式化已经获得的 Edge，不重新验证来源。真实报告生成前应使用当前 Context 重新检查每个 chunk_id，防止遍历后资料被更新或撤权。若一条必要边失效，整条路径不能继续作为完整证据链；应返回失效或重新搜索，而不是只把那条引用从显示中删掉。

## 小课十三：图的更新与删除是依赖维护问题

来源更新后旧 chunk 不再当前可见，因此 EvidenceGraph.visible_edges 会过滤对应旧边。旧边行可能仍存数据库，因为历史版本也保留；purge 则通过外键级联删除。这个设计让来源生命周期自然影响图读取，但它不是预先物理删除所有陈旧边。理解逻辑不可见与物理清理的区别，才能正确解释 counts 与查询结果。

如果你缓存邻接表，缓存也必须绑定可见范围与来源版本。否则在线 visible_edges 已正确过滤，缓存却仍返回旧关系。一个简单课堂方案每次重建可见邻接表，容易理解但不适合大规模；更高效方案可以使用变更序列、依赖失效或版本化图快照。性能优化不能把 Context 从接口中拿掉。

节点可以由多个来源提及，某个来源删除不一定应删除节点本身；但依赖该来源的断言必须失效。一个公开实体名可能仍存在，某条私有关系不可见。把权限只挂在节点上会过度隐藏或泄露，因此证据边层级的权限非常重要。对社区摘要和派生结论，还要继续追踪到下一层，R11 会展开。

```python
with CorpusStore() as store:
    store.ingest("stale-graph", "a", "A", "Jade uses Quartz.", allowed=("*",))
    store.ingest("stale-graph", "b", "B", "Quartz uses Cedar.", allowed=("*",))
    ctx = Context("stale-graph", "learner")
    by_doc = {chunk.doc_id: chunk for chunk in store.visible(ctx)}
    graph = EvidenceGraph(store)
    graph.add(ctx, "Jade", "uses", "Quartz", by_doc["a"].chunk_id, by_doc["a"].text)
    graph.add(ctx, "Quartz", "uses", "Cedar", by_doc["b"].chunk_id, by_doc["b"].text)
    assert graph.path(ctx, "Jade", "Cedar") is not None
    store.delete("stale-graph", "b", expected_revision=1)
    assert graph.path(ctx, "Jade", "Cedar") is None
    assert store.counts()["edges"] == 2
    store.purge("stale-graph", "b")
    assert store.counts()["edges"] == 1
```

## 小课十四：十二道练习与完整解答

**练习一，三种图。** “检索后调用回答器”是工作流边；“矩阵乘法输出进入 softmax”是计算依赖；“Jade 使用 Quartz”是知识断言。完整答案必须解释边的语义与可执行性不同，不能因为都能 BFS 就当作同一种模型。

**练习二，同名实体。** 两个项目都有 Scheduler，应使用两个 namespace 下的实体 ID。`resolve_name` 必须结合范围，缺少范围时返回歧义候选或未解析，不取第一项。展示名相同不影响 ID 不同，别名合并需要证据。

**练习三，反向查询。** 输入 A uses B，反向表键为 B，值仍是原边 A uses B。不能改成 B uses A。调用 adjacency(edges, reverse=True) 即完整代码参考，检查返回 Edge.subject 仍为 A。

**练习四，路径长度。** A-B-C 有三节点两边。max_depth 一不能到 C，二可以。起点等于终点返回空路径而非不存在。小课七 bfs_trace 的三个断言完整覆盖这些边界，课程 EvidenceGraph.path 使用相同深度语义。

**练习五，环。** 加 C 到 A，不应无限循环。visited 阻止重复扩展，预算提供额外上限。环并不总是坏数据，互相调用可以合法；若关系是严格包含，环可能违反模型约束，应在该关系子图单独检查，而不是禁止全部图环。

**练习六，最短路不等于最好证据。** 一跳边来自含糊摘要，三跳边每条有清楚来源。BFS只优化跳数，可能选一跳；不能把结果称为最可信。完整答案应提出先做边质量过滤或明确加权目标，再用匹配算法，而不是修改 BFS 队列顺序后宣称概率最优。

**练习七，否定。** 原文“It is false that Jade uses Quartz.” 会触发受控正则中的子串。应把该抽取标为候选并通过句子范围审核，或限制输入必须完全是受控句式。完整修复基线可以使用 fullmatch，只接受整句精确模板，牺牲召回换清楚边界。

```python
def extract_one_controlled_sentence(sentence, aliases):
    match = RELATION_PATTERN.fullmatch(sentence.strip())
    if match is None:
        return None
    a, b = match.group("subject"), match.group("object")
    if a not in aliases or b not in aliases:
        return None
    return aliases[a], "uses", aliases[b]

assert extract_one_controlled_sentence("Jade uses Quartz.", aliases) == ("jade:service", "uses", "jade:quartz")
assert extract_one_controlled_sentence("It is false that Jade uses Quartz.", aliases) is None
```

**练习八，关系模式。** 要求 uses 后 part_of，不能接受直接 part_of。完整代码为 pattern_path，状态包含节点与模式位置。若只按节点 visited，可能把不同模式进度合并而漏解；因此需要解释状态足以决定后继，而不是机械记一个算法模板。

**练习九，来源重复。** 三条边同三元组不同转载来源，结构度数应按规范三元组去重，证据账本仍保留来源。若统计独立支持，需要 original_source_id 去重。完整答案要区分结构分析与证据计数两个目标。

**练习十，来源撤销。** 删除第二条边来源后两跳路径消失，软删除时边行仍在，purge 后边行减少。小课十三完整程序给出断言。若缓存邻接表仍返回旧路径，错误在缓存依赖，不是 SQL 外键没工作。

**练习十一，合法 JSON 的错误关系。** subject、predicate、object、quote 全合法，但原文否定该关系。结构验证通过，语义验证失败。解决需要否定、条件与来源支持标签，不能只增加一个 confidence 字段或让模型再说一次“确定”。

**练习十二，评价抽取。** 手工标注十句，包含肯定、否定、条件、同名、跨句指代各两句。统计正确三元组、漏掉关系、错误新增，并单独检查来源与实体链接。精确率是正确预测除以全部预测，召回率是正确预测除以全部应抽取关系；空预测的处理应预先定义，不能为了好看把无结果样本删除。

## 小课十五：完成一个小型图查询工程

第一阶段手工确认十二条关系，分属两个虚构项目，包含同名组件、环、两条同长路径、一个不可达节点与一条受限关系。每条关系独立来源，方便更新和删除。先写期望查询表：直接邻居、反向依赖、两跳路径、权限差异、深度不足与不存在目标。运行课程 EvidenceGraph，不需要外部图数据库。

第二阶段加入受控抽取器，只接受完整模板句。保存未匹配句子及原因，人工检查漏召回。不要通过悄悄放宽规则让所有句子都输出边。比较手工金标与自动抽取，统计实体错误、方向错误、谓词错误和否定错误。你会看到图质量瓶颈往往在建图，而不是 BFS 算法。

第三阶段加入实体目录与别名。故意给两个项目相同展示名，验证 namespace 防串线；增加一个明确别名，验证两个提及映射同一实体；撤销错误别名，重建受影响边。保留原始提及，使对齐决定可逆。若直接删除原节点，就无法可靠回滚错误合并。

第四阶段接入文本检索。先检索种子片段，从已确认提及得到实体，再在允许谓词与预算内扩展，最后返回每条边原文。比较 text-only 与 text-plus-graph，在需要关系连接的问题上测证据覆盖；对错误码精确查找题可能没有收益，应如实报告。图不是所有检索问题的通用升级。

第五阶段做生命周期验收：更新来源、撤权、软删除、purge、同 ID 重建。旧边和旧引用不能复活，跨租户路径不能出现，缓存应失效。把每次查询的图快照、来源版本与停止原因写入报告。完成后，你得到的是一个可解释的关系检索工程，而不是一张只能截图展示的漂亮网络图。

### 从自然语言到图模式：一个订单例子说明二元边的限制

不是所有事实都适合直接压成二元关系。句子“服务 A 在版本二通过插件 B 于时间 T 读取仓库 C”至少包含主体、动作、中介、对象、版本与时间。若只保存 A uses C，会丢掉通过谁、哪个版本、何时成立。若把所有名词两两连边，又会制造原文没有的关系。建模时要决定哪些条件作为边属性，哪些需要单独事件节点。

事件节点是一种常见选择：建立一次读取配置或部署事件，连接执行者 A、插件 B、仓库 C，并附版本与时间。这样多个参与者属于同一个事件，不会被误组合。代价是查询多走几条边、模式更复杂。属性图可以把简单条件放在边属性中；RDF 风格三元组若需要给断言再附出处，通常需要额外表示机制。具体表示不同，核心需求都是保留关系发生的条件与身份。

课堂例子可以先做一个三元关系：Alice 在项目 Jade 使用 Python，Bob 在项目 Quartz 使用 Rust。如果只建 Alice uses Python 与 Alice works_on Jade，未来查询可能把另一个项目的语言误连接过来。事件或作用域节点能明确“在项目中使用”的组合。不要为了让图简单而删除决定语义的重要维度，简单错误的模式会让所有后续算法高效地产生错误结论。

设计模式时先列你希望支持与禁止的推断。希望查询某项目实际语言，应能沿项目范围找到对应记录；禁止把个人练习语言当工作项目语言。然后构造最小反例，看模式是否能区分。模式设计不是选几个流行谓词，而是确保不同现实情况在数据中有不同表示，否则任何算法都无法恢复你已经丢掉的信息。

### 属性图、三元组与关系表：先比较契约，再比较产品

属性图通常允许节点与边带属性、类型和标签，适合把版本、来源、权重等附在关系上。三元组模型把事实表达为主体、谓词、对象，统一且便于语义词表；复杂限定需要进一步建模。关系数据库用实体表、关系表、来源表与索引也能实现许多图查询。选择图数据库不是使用图思维的前提，课程 SQLite 就能保存边并进行有限遍历。

三种表示的比较应基于实际查询：是否频繁多跳、是否需要模式约束、数据规模多大、更新是否频繁、事务与权限怎么做、查询语言是否容易维护。一个只有几千边的课堂项目，用 Python 邻接表可能最透明；一个需要复杂交互遍历的大图可能受益于专门引擎；一个主要做统计联接的业务表，关系数据库可能已经足够。不要把技术名称当成成熟度阶梯。

无论引擎如何，实体身份与来源契约不能消失。把 Edge 迁移到图数据库时，仍需保留 tenant、chunk_id、quote、当前版本过滤与删除依赖。数据库自己的登录用户不等于业务租户授权，索引高效也不等于关系真实。迁移验收应该重跑同一组权限、更新与路径测试，而不是只检查连接成功与节点数相同。

查询语言也需要参数化。模型生成任意查询字符串直接执行，可能扩大范围或产生昂贵遍历。更稳妥的工具接口提供受控参数，如起点、目标、允许谓词、深度与边数上限，由服务构建查询。若允许模型生成查询，应进行解析、允许模式与资源限制，不能只靠提示词说“请安全查询”。这与本卷所有工具的原则一致：提案来自模型，权限与执行契约来自运行器。

### 加权最短路的完整代码与手算

为了真正理解 BFS 的边界，亲手实现 Dijkstra。它维护到每个节点的当前最小代价，并用最小堆优先扩展代价最小的候选。因为所有边权非负，一旦取出的状态是当前最小且到达目标，就不会有未来绕更长路径得到更小代价。若允许负权，这个论证失效，算法可能提前返回错误结果。

```python
import heapq
import math


def dijkstra(adjacency_map, start, target):
    for edges in adjacency_map.values():
        for neighbor, cost in edges:
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("finite nonnegative costs required")
    best = {start: 0.0}
    queue = [(0.0, start, [])]
    while queue:
        total, node, path = heapq.heappop(queue)
        if total != best.get(node):
            continue
        if node == target:
            return total, path
        for neighbor, cost in adjacency_map.get(node, []):
            candidate = total + cost
            if candidate < best.get(neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor, path + [(node, neighbor)]))
    return None

weighted = {"A": [("E", 10.0), ("B", 1.0)], "B": [("E", 1.0)]}
assert dijkstra(weighted, "A", "E") == (2.0, [("A", "B"), ("B", "E")])
```

初始堆只有零代价 A。扩展 A 后，E 的候选代价十，B 的候选代价一。先弹出 B，发现到 E 的代价二，小于原十，更新 best 并把二代价 E 入堆。随后弹出二代价 E，返回。十代价 E 仍可能留在堆里，但如果没有提前返回，`total != best.get(node)` 会跳过过期候选。这是常见的懒删除技巧，不需要在堆中原地查找并修改旧项。

该实现把完整 path 复制进堆，容易理解但大图时内存较多。更高效实现保存 predecessor，最终从目标逆向恢复路径。节点字符串用于堆的同代价平局比较，因此示例假设节点可排序；任意对象可以额外加递增序号。细节不改变算法原理，但决定代码能否在不同数据类型上可靠运行。

把边权设为负对数置信度时，需要置信度严格大于零且不大于一，乘积最大等价于负对数和最小。但这个数学等价不证明输入置信度已校准，也不证明各边独立。你可以把它当启发式路径排序，并保留所有来源让人复核；不能仅凭算法名称给路径赋予事实保证。

### 全路径、最短路径与可达性不是同一个任务

可达性只问是否存在路径；一条最短路径返回一个最少跳数解释；所有最短路径可能有很多；所有简单路径在最坏情况下数量指数增长。用户问“为什么 A 受 B 影响”可能需要几条不同机制路径，而不是所有可能路线。接口应明确返回上限与排序，防止一句模糊“找路径”触发不可控枚举。

visited 的使用也随任务变化。普通可达性可以全局标记节点；枚举所有简单路径需要当前路径内的 visited，允许同一节点出现在不同候选路径中；带模式约束需要节点与模式状态；带预算约束可能需要节点与资源状态。机械复制一个 visited=set 会让算法漏解或爆炸，必须先定义状态等价关系。

环的处理同样依赖语义。uses 图存在环可能反映相互调用，不能一概删除；严格 part_of 层级中出现环可能违反建模约束；supersedes 链若回到旧版可能表示错误版本关系。可以分别对每种谓词子图检查环，再把结果交给领域规则判断。一个通用“图有环所以坏了”的提示不够准确。

大图查询还要考虑高连接度枢纽。节点“系统”“数据”“模型”若与几乎所有实体相连，两跳扩展会爆炸。它们可能是过于泛化的实体，也可能是真实公共组件。可以按实体类型、谓词、来源范围与任务约束限制扩展，或者单独处理高频概念。不能只删除所有高度节点，因为有些核心依赖确实重要；需要通过任务评测判断。

### 多来源支持的规范化视图

证据边与规范关系应分两层。证据边保留每个来源的断言，便于撤销与审计；规范关系把相同 subject、predicate、object 的当前有效证据聚合起来，用于结构查询。聚合结果应包含支持来源集合、冲突来源集合与状态，不应该只剩一个布尔真值。来源变化时重新计算聚合，避免旧多数票永久保留。

下面的函数只做肯定证据按三元组分组，没有实现冲突裁决。它清楚展示结构边数与证据边数不同。输入应已通过当前授权过滤，否则聚合的来源数量本身也可能泄露私有证据存在。

```python
def group_assertions(edges):
    grouped = {}
    for edge in edges:
        triple = (edge.subject, edge.predicate, edge.object)
        grouped.setdefault(triple, []).append({"edge_id": edge.edge_id,
                                              "source_id": edge.chunk_id,
                                              "quote": edge.quote})
    return {triple: sorted(supports, key=lambda row: row["edge_id"])
            for triple, supports in grouped.items()}
```

若两个来源互相矛盾，不要把否定转换成另一条肯定边后混进同一个集合。可以给断言增加 polarity 或把反对证据单独记录，保留各自有效时间与范围。很多表面冲突其实来自不同版本或部署条件，先对齐条件再裁决。只看三元组不看限定会把可共存事实误判为矛盾，也会把真正冲突悄悄平均掉。

图聚合还要注意来源可信度与独立性。官方配置、实际运行日志、个人笔记与模型总结的角色不同；某些任务以运行事实为准，某些任务问的是官方设计。一个全局 source_rank 未必适用所有问题。更好的报告保留来源类型与具体支持范围，让最终回答明确是在描述设计、配置还是观测。

### 抽取评价的完整计算示例

金标关系有五条，系统预测六条，其中四条完全正确、一条主客体颠倒、一条凭空新增。严格三元组精确率四除六约零点六六七，召回率四除五等于零点八，F1 是两者调和平均约零点七二七。若用实体名词面相似给颠倒关系算半对，会掩盖方向错误；严格评价应把它当错误预测与一个漏掉的正确关系。

```python
def triple_scores(predicted, gold):
    predicted, gold = set(predicted), set(gold)
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "false_positive": sorted(predicted - gold), "false_negative": sorted(gold - predicted)}

gold = {("A", "uses", "B"), ("B", "uses", "C"), ("C", "uses", "D"),
        ("D", "uses", "E"), ("E", "uses", "F")}
predicted = (gold - {("E", "uses", "F")}) | {("F", "uses", "E"), ("A", "uses", "Z")}
scores = triple_scores(predicted, gold)
assert abs(scores["precision"] - 4/6) < 1e-12
assert abs(scores["recall"] - 4/5) < 1e-12
assert abs(scores["f1"] - 8/11) < 1e-12
```

三元组正确还不够。系统可能碰巧预测正确关系，却引用错误来源。需要单独检查 source_id、引文位置与支持范围。一个评价单元可以是三元组加来源版本与条件，严格匹配用于高精度回归；人工语义评价处理不同表达但同义的情况。两类评价应分别报告，不能让宽松语义评分掩盖引用失真。

实体链接错误会影响多条关系，所以要记录错误传播。例如 Scheduler 被合并到错误项目后，五条边都错，根因只有一个链接决定。按错误边计数能反映影响，按根因分类能指导修复。报告应同时有组件指标与端到端路径结果，不然你可能一直调关系抽取提示词，却真正需要修实体目录。

### 三个确定性故障实验

第一个实验是同名串线。建立 Jade:Scheduler 与 Quartz:Scheduler，各自依赖不同索引。错误实现按展示名合并，查询 Jade 到 Quartz 索引会出现不存在路径。修复使用 namespace 后该路径消失，原来各自合法路径仍存在。验收不只看错误路径没了，还要保证没有通过删除全部边“修复”问题。

第二个实验是否定抽取。原文“Jade does not use Quartz”与“Jade uses Quartz”各一条，错误规则只检测两个名字共现，产生同样边。修复受控句式与语义审查后只接受肯定来源，否定进入冲突或待审核状态。不要把负例直接从数据集删掉，否则测试只剩容易句子，指标会虚高。

第三个实验是陈旧邻接缓存。先建立两跳路径并缓存邻接表，再删除第二条来源。实时图查询返回无路径，缓存查询仍返回旧路径。修复为缓存键绑定图代际与权限范围，或每次输出前复核边来源。验收必须真的走缓存入口；只重跑实时查询不能证明缓存问题解决。

每个故障报告包含输入、错误路径、第一次偏离、修复与回归。特别要保存错误路径的每条边出处，这使你能区分“图里有错边”与“算法错误使用正确边”。如果没有出处，调试容易退化成对着可视化猜测。

### 证据路径的逻辑审查工作表

拿到一条路径后，先逐边翻译成完整自然语言句子，保留主语、谓词、宾语与条件。再检查相邻边共享实体是否真的同一对象，版本与时间是否相容。最后写出你想在答案中表达的结论，问它是否只是重述路径，还是新增了传递推断。新增推断必须有规则或额外证据，不能因为连线连续就自动成立。

例如 A uses B，B depends_on C，可以说“资料显示 A 使用 B，而 B 依赖 C”，这只是报告两条边。若说“A 的任何请求都必须访问 C”，则新增运行时必经路径断言，需要调用条件与部署证据。若说“C 故障一定导致 A 失败”，还需要容错、缓存、降级与使用频率等信息。图路径帮助定位要查什么，不替代这些机制事实。

时间相容也必须检查。A 在版本一使用 B，B 在版本二才使用 C，不能无条件拼成 A 版本一到 C 的链。可以把关系限定版本存为属性或版本节点，遍历时检查区间交集。若没有这些字段，报告应该限制为“文档中存在相关关系线索，尚未确认同一部署版本”，而不是制造精确因果链。

### 图中的数字与单位也要有实体语义

如果关系对象是数值，必须保存单位和条件。`capacity=7` 不说明七个请求、七 GB 还是七页；不同文档可能一个写 MiB、一个写 MB。把数值都当字符串节点会让查询难以比较，也可能把不同单位误对齐。可使用属性值结构包含数值、单位、测量条件与来源，必要时单独建测量事件。

数值抽取评价应允许明确的单位换算，但不能忽略条件。例如同一模型 batch 一与 batch 八的吞吐不能直接当同一属性冲突。图模式要保留测试配置或测量事件，查询才知道比较对象。图谱不是只处理名词关系，实际工程中数值、版本、时间与条件往往决定答案是否正确。

课程基础 Edge 的 object 是字符串，适合实体关系演示，不提供通用单位系统。若你把复杂测量塞进 object 字符串，仍可存储但无法可靠计算。扩展时应先写数据结构与转换测试，再加入查询，不要靠自然语言模型在每次回答时临时猜单位。

### 图可视化应该服务诊断，而不是代替数据模型

可视化适合查看局部路径、同名实体、异常高度节点与删除影响。全图几千节点挤成一团，通常不能说明系统正确。优先展示当前问题的授权子图，每条边可回到来源与版本，节点展示 namespace 与类型。不同关系用明确标签，不能只靠颜色暗示语义，避免黑白或无障碍场景丢失信息。

图布局距离不等于语义距离。力导向布局把连接多的节点拉近，是可视化算法结果，不证明两个实体在业务上更相似。社区颜色也只是某个分组结果，不能当客观主题真理。报告中应解释布局与分组的依据，特别是面向非技术读者时，漂亮图容易制造过度确定感。

调试时可以导出一个小 Mermaid 路径图，但最终引用仍需文本来源。图中一条箭头只能显示短标签，容不下所有时间、权限与证据条件，所以旁边的路径表不可省略。可视化是帮助理解结构的视图，不是来源账本的替代品。

### 扩展到图数据库前的迁移验收

先把课程十二条关系与全部负例作为固定数据导入目标引擎。检查实体 ID、边方向、重复来源、空路径、不可达、深度限制、谓词限制、授权与删除。比较两边输出的规范三元组与来源集合，不要求内部存储相同。若数据库默认去重关系或不允许多重边，需要显式保存 assertion 节点，否则会丢掉多来源语义。

然后测试查询计划与资源上限。一个深度二查询遇到百万度节点仍可能昂贵，所以要限制扫描边或候选数量，并在报告中标注截断。数据库返回部分结果时不能写完整路径枚举完成。超时与确实无路径也应区分，前者是未知，后者是在已搜索范围内未找到；混为一谈会让系统错误断言不存在依赖。

最后做更新传播。来源版本改变后，旧断言是否立即被过滤；授权变化后，图缓存是否失效；硬删除后，派生摘要是否重建；同 ID 重建是否复用旧关系身份。任何一项失败，都不能因为数据库查询速度更快就称迁移成功。性能是正确语义之上的优化目标。

### 开源源码阅读的第一条调用链

从课程 `EvidenceGraph.add` 进入，跟踪 get_visible_chunk 如何回到 CorpusStore.visible，再看 edge_id 的 digest 输入，最后看 SQL INSERT OR IGNORE。写下这条调用链保护什么：当前可见来源、非空引文、受控谓词、确定性边身份。再写它没有保护什么：实体真实存在、类型正确、引文蕴含关系、跨文档时间相容。把两列并排，你就不会把局部验证误解成完整知识保证。

然后从 path 进入，跟踪 visible_edges、邻接表、队列、visited、深度与扩展预算。手工把一个环、一条同长替代路径和一个不可达目标喂进去，逐步比较状态。最后看更新删除如何通过来源可见性改变图。这个顺序比从大型图框架的目录开始迷路更适合初学者，因为每个组件都有可手算对象。

运行既有入口完成基础确认：

```bash
python -m labs.knowledge.lab 25-graph
python -m labs.knowledge.lab 26-graph-stale
python -m unittest discover -s tests -p 'test_knowledge*.py' -v
```

实验成功后，不要只记录“图能查询”。记录找到哪条路径、用了几条边、每条来自哪个片段、删除哪个来源后为何消失、权限不同为何结果不同。真正的掌握是能解释图的每条线从哪里来、允许推出什么、什么时候不再可用，以及查询为什么在某处停止。

### 一次完整的建图审查：从五句话到两条可用关系

准备五句话。第一句“Jade uses Quartz.”，目录中两个实体均已确认。第二句“Quartz may use Cedar when remote mode is enabled.”，带条件与不确定词。第三句“Jade does not use Archive directly.”，明确否定直接使用。第四句“Scheduler uses Cedar.”，缺少项目范围且存在两个同名调度器。第五句“Quartz uses Cedar.”，来自已撤权文档。错误系统可能生成五条肯定边；可靠流程应逐句说明为何接受、保留条件、拒绝或待消歧。

第一句可以进入已确认肯定图，来源当前可见且关系由人工标签确认。第二句不能无条件加入 uses，应保存条件或标待审核；若当前图模式无法表示条件，就不要丢掉条件后硬塞。第三句是负向断言，不能生成肯定 uses；它可能用于反驳另一条直接关系。第四句在实体链接阶段停下，返回两个候选，不因 Cedar 名字熟悉就选择某个项目。第五句即使语义完全清楚，也不能进入当前调用者可见图，因为来源资格不满足。

如果另外提供明确项目的第四句与授权有效的第五句，图才可能得到新的可用关系。这个审查顺序说明“抽取到关系”与“允许加入当前图”是两种成功。指标应该分别记录候选抽取能力与可发布断言质量，不把所有拒绝都当模型能力差。拒绝缺少条件或权限的关系，是系统保持语义边界的正确行为。

把审查结果存成表格字段：sentence_id、entity_status、relation_status、source_status、decision、reason、accepted_edge_id。理由用稳定标签，便于统计最常见瓶颈。若多数失败是 ambiguous_entity，就改实体目录与上下文；若多数是 unsupported_relation，就改抽取与语义审核；若多数是 stale_source，就查摄入与任务时序。只有分层记录，优化才不会盲目。

### 不存在的关系能否由“没有边”证明

图中没有 A uses B，不代表现实中 A 不使用 B。可能来源没摄入、抽取漏掉、实体未对齐、权限不足、版本过滤或遍历预算截断。知识图通常是不完整观察，缺失与否定必须区分。若要表达明确否定，需要来源支持的负断言及其范围；不能把每一条不存在的边都解释为假。

这也影响查询报告。`path` 返回 None 表示在当前可见图、给定谓词与预算下未找到路径，不是证明世界上不存在任何依赖。若 max_expansions 提前耗尽，甚至不能证明整个当前图不可达。课程 API 只返回 None，没有细分停止原因，所以高级项目应增加 trace 或结果状态，避免上层把预算不足误译成不存在。

封闭世界假设在某些受控系统中可以成立，例如一份完整声明所有依赖的锁文件，缺少某条直接依赖可支持特定结论。但这种完整性保证来自数据契约，不来自图结构本身。你应记录语料是否完整、对哪个版本完整、允许推断的范围。开放网页资料与完整配置清单不能使用相同缺失语义。

### 关系推理规则如何进入系统

如果领域确实需要推理，可以显式定义规则，而不是让遍历算法隐含承担。例如 part_of 在特定无环层级中传递，A part_of B 且 B part_of C 可以推出 A part_of C。派生边应标记 rule_id 与依赖边集合，不伪装成原文直接陈述。任何依赖边撤销，派生边也应失效；规则版本变化同样需要重新计算。

规则还需要避免无限生成。传递闭包可以用集合去重，直到没有新三元组；若图有环，可能推出自包含关系，需根据模型决定拒绝、标冲突或允许。计算复杂度随节点与边增长，不能无限运行。小型教学可以先在三节点上手算闭包，再增加一个环观察结果，理解“逻辑规则正确”与“领域数据合理”仍是两层问题。

多条规则组合可能产生意外结论。uses 与 supports 不应随意组合成 supports，除非规则明确包含启用配置与能力传递条件。最好把每个规则配正例、反例与边界条件，像测试普通函数一样测试。推理系统不是给所有谓词加“可传递”选项，而是把允许的结论关系形式化并保留证明依赖。

### 最后一次综合练习的完整验收答案

任务是解释虚构 Jade 服务为什么可能受 Cedar 索引更新影响。合格输出先列 Jade uses Quartz 与 Quartz uses Cedar 两条当前授权关系，附各自来源；再检索 Cedar 更新机制原文；最后明确“这是一条组件依赖线索，实际影响还取决于 Quartz 的调用方式、缓存与部署配置”。若缺少运行配置，不能升级为“更新必然导致 Jade 停机”。

程序应先解析 Jade 的唯一实体，限定项目 namespace，使用允许谓词 uses，深度最多二，取回路径后复核来源，再从终点实体关联文本检索更新说明。每一步都有输入和输出：实体 ID、边序列、来源片段、机制证据、最终断言。任何一步缺失都返回对应缺口，而不是从模型背景知识补齐。

验收负例包括同名 Jade 项目、被撤权中间边、旧版本关系、只有一跳预算、错误方向、带否定来源、来源删除后缓存路径、模型伪造 chunk_id。正确系统分别拒绝串线、移除不可见链、排除旧边、报告预算内未找到、保持方向、拒绝肯定抽取、失效缓存、拒绝伪造引用。把这些反例跑通，比增加一百个只展示成功路径的样例更能证明你的图工程可靠。

完成这项综合练习后，再进入 R11 的社区与全局问答。你已经拥有实体身份、边来源、路径预算和生命周期基础，接下来要研究的是如何组织大量关系与文档来回答整体主题问题。那是新的聚合与摘要层，不能因为本章已经会 BFS，就把后面所有 GraphRAG 机制当作同一个算法的换名。

还有一个工程细节是实体合并后的引用稳定性。展示层可以把旧实体名重定向到规范实体，但历史断言应保留当时使用的提及与对齐版本。否则一次目录修正会让旧报告看起来像当时就已经知道新的身份，审计难以解释。可以把当前规范视图与历史原始断言分开，查询时明确使用哪个视图；纠错后重新生成报告，而不是原地改写过去证据。

测试实体合并时，先建立两个相似但不同实体，分别有独立路径；再执行错误合并，观察新增的假路径；撤销合并后，假路径必须消失且原合法路径保留。这个实验要求保存原始提及与边归属，能直接检验对齐操作是否可逆。如果只能通过清空整张图重建恢复，应把这种恢复成本写进系统设计，不能把合并按钮视为无代价动作。

图的质量报告还应区分覆盖率与密度。覆盖率是相对于标注实体、关系或任务证据集的比例；密度是图自身边数相对可能边数的统计。不断增加猜测边会提高密度，却可能降低精确率和任务可靠性。任何“图更丰富了”的结论都应回答丰富的是正确来源支持，还是仅仅多了连线。保留少量高质量可追溯边，是本章从零训练时更好的起点。

每次发布图的新版本，都保存一份小型固定查询结果作为回归基线；基线随来源变化可以更新，但更新理由必须能追溯到具体资料或建模规则。
