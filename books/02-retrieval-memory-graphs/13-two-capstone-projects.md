# R13 两个完整项目：从空数据库到可验证的知识与记忆助手

先修 R01–R12。本章把前面机制连成两个可运行工程：项目 A 是带权限、版本和引用的知识助手，项目 B 是可更正、可过期、可遗忘的记忆助手。参考实现已经存在，默认全部离线、CPU、虚构数据。目标不是做一个只会回复的界面，而是让你能预测数据如何变化、解释失败在哪里、独立替换一个组件且不破坏其他契约。

## 小课一：先写交付标准，再开始敲代码

项目 A 接收资料与问题，输出带当前来源的严格抽取答案。必须回答六个工程问题：资料是否完整摄入，当前身份是否能读，检索为何选中这些片段，回答用了哪个版本，更新删除后旧引用如何失效，没有证据时如何停止。项目 B 接收显式记忆事件，输出当前适用记录，必须解释归属、来源、范围、版本、期限与删除范围。

这两个项目共享许多机制，但对象不同。知识文档有逻辑 ID、正文、片段与授权；记忆有 owner、key、kind、value 与过期时间。不要为了复用代码把两者强行塞进同一张表，造成文档版本与个人偏好状态混乱。可以共享哈希、JSON、事务与测试方法，保留各自数据契约。

交付物包括代码、固定虚构数据、命令行入口、测试、实验记录与已知限制。界面可以以后加，第一版先保证 CLI 能完整演示生命周期。一个陌生读者按照命令能重现结果，比一张漂亮截图更有价值。所有路径从仓库根目录运行，Windows 可以用原生 Python 或 WSL，Linux 服务器 CPU 足够，不需要 H100。

## 小课二：建立项目目录与环境意识

本卷模块位于 labs/knowledge，教材位于 books/02-retrieval-memory-graphs。运行 `python -m labs.knowledge.assistant` 时，Python 按包路径加载模块，因此应在仓库根目录执行。若在 labs/knowledge 内直接运行 assistant.py，相对导入可能失败；这不是依赖没装，而是启动方式改变了包上下文。先理解错误信息再安装东西，避免把路径问题误当环境问题。

实验数据放在独立工作目录，数据库不提交公开仓库。临时演示使用内存库，进程结束即消失；持久化使用文件路径，父目录必须存在。不要把自己的真实公司资料作为第一次测试，虚构语料更容易验证权限与预期，也避免意外导出。API 密钥只在明确联网实验时通过环境变量提供，不写进源码或报告。

```bash
python -m labs.knowledge.assistant --help
python -m labs.knowledge.memory_assistant --help
python -m labs.knowledge.assistant ask "KV cache" --demo
python -m labs.knowledge.memory_assistant demo
```

先观察帮助与输出字段，不急着修改。知识助手应报告 strict_extractive、来源身份与是否 abstain；记忆演示应显示保存、更正、当前值与遗忘后历史。命令成功退出只是第一层，接下来需要逐字段验收。若结果与你预期不同，记录输入、预期、实际与第一处差异，不要只说程序坏了。

## 小课三：项目 A 的模块责任与调用链

text.py 负责分词、精确切块与确定性 digest；store.py 负责文档快照、权限、事务、删除与墓碑；retrieval.py 负责 BM25、教学 TF-IDF、融合与选择；rag.py 负责上下文预算、引用验证与严格抽取报告；assistant.py 负责命令行解析和把这些模块接起来。模型生成不是默认依赖，因此你能在 CPU 上观察完整证据链。

一次 ask 的顺序是取得 Context，读取当前可见 chunks，构建检索统计，得到 Hit，打包证据，生成抽取 Claim，最终重新验证来源并输出报告。Context 不是 query 的一部分，Hit.score 不是事实概率，Claim 与 Citation 分开。每个对象的含义在前面章节已经手算，本项目要求你在真实调用链中再次识别它们。

源码阅读从数据类开始，再看入口参数，最后跟一个问题进入 hybrid 与 answer_report。不要一开始从 SQL 所有语句顺序读到末尾。你需要先知道函数接受什么、返回什么、负责哪条不变量，然后细看实现。调用链笔记可以只写五六个节点，但每个节点要附实际输入输出样本。

## 小课四：从零创建一套自己的虚构语料

下面不用默认 fixtures，建立一个更小的项目数据集。公开资料说明更新机制，受限资料说明维护窗口，另一个租户有同名文档。你可以手算 learner 与 operator 的可见集合，之后所有检索与引用都建立在这个集合上。

```python
from labs.knowledge.store import CorpusStore, Context, ConflictError
from labs.knowledge.retrieval import hybrid
from labs.knowledge.rag import answer_report, Citation, Claim, validate_claim


def seed_project(store):
    rows = [
        ("project-a", "update", "Update", "Jade builds a new generation before switching.", ("*",)),
        ("project-a", "recovery", "Recovery", "Jade retains the previous generation for recovery.", ("*",)),
        ("project-a", "maintenance", "Maintenance", "Restricted maintenance marker ORCHID.", ("group:ops",)),
        ("project-b", "update", "Other update", "Other tenant marker CEDAR.", ("*",)),
    ]
    return [store.ingest(tenant, doc, title, body, allowed=allowed)
            for tenant, doc, title, body, allowed in rows]

with CorpusStore() as store:
    assert seed_project(store) == [1, 1, 1, 1]
    learner = Context("project-a", "learner")
    operator = Context("project-a", "operator", ("ops",))
    assert {chunk.doc_id for chunk in store.visible(learner)} == {"update", "recovery"}
    assert {chunk.doc_id for chunk in store.visible(operator)} == {"update", "recovery", "maintenance"}
    assert hybrid(store, learner, "ORCHID") == []
    assert hybrid(store, learner, "CEDAR") == []
```

seed_project 返回每次摄入的 revision 列表。两个租户同名 update 不冲突，因为主键包含 tenant。受限标记词被设计为唯一，便于检验权限，不是因为真实权限系统应依赖关键词隐藏。即使 query 猜中标记，learner 仍没有候选；operator 可以读取，说明过滤是身份语义而非简单删除敏感词。

再次调用 seed_project 应返回同样版本，内容表数量不增加。只修改维护权限则该文档增版，普通用户可见性改变。把这些操作逐项写成断言，你就拥有一个比“能搜到一句话”更完整的摄入验收。

## 小课五：手工检查一次真实检索

查询 Jade generation 时，两个公开文档都相关。先打印 tokenize 的词元，再查看 BM25 与 TF-IDF 结果，最后看 RRF 排序。课程 TF-IDF 是词法稀疏向量，不是真实语义 embedding；不能把两路融合称为已经实现语义混合检索。真实 embedding 替换在后面阶段单独进行。

排序分数不一定随文档更新单调变化，因为 N、df、长度与候选集合都会改变。不要断言某个分数必须永远等于初次演示值，除非测试固定语料与算法配置。项目验收更重视标注相关来源是否进入候选、权限是否正确与同分顺序可复现。数值手算测试放在 R03 的小语料里，工程测试则验证数据流。

```python
from labs.knowledge.retrieval import BM25, TfidfVectors, rrf
from labs.knowledge.text import tokenize

with CorpusStore() as store:
    seed_project(store)
    ctx = Context("project-a", "learner")
    visible = store.visible(ctx)
    terms = tokenize("Jade generation")
    lexical = BM25(visible).search("Jade generation", k=3)
    sparse = TfidfVectors(visible).search("Jade generation", k=3)
    fused = rrf([lexical, sparse], k=3)
    assert terms == ["jade", "generation"]
    assert {hit.chunk.doc_id for hit in fused} == {"update", "recovery"}
```

如果某一步为空，按层检查：可见集合是否有资料，词元是否匹配，分数是否正，top_k 是否合理。不要在还没有模型的系统里用“温度太高”解释检索错误。这个练习训练的是明确因果链，后面接入复杂框架也应保留这种诊断方式。

## 小课六：把候选变成可复查答案

answer_report 的实际接口需要按源码使用，它负责调用严格抽取链并复核引用。项目中每条最终证据应包含 doc_id、revision、chunk_id 与原文。严格抽取限制生成自由，方便先验证来源，不代表已经具备自然语言综合推理。遇到否定或复杂条件，原句仍可能被截取失当，R06 已说明子串存在不等于语义完整。

先用手工 Claim 演练引用生命周期，比直接让模型生成更容易定位。取当前片段全文作为 quote 与 text，验证通过；更新文档后旧 chunk 不再当前可见，验证失败；重新取当前片段，新的引用通过。这个实验把摄入版本与答案发布连接起来。

```python
with CorpusStore() as store:
    seed_project(store)
    ctx = Context("project-a", "learner")
    old = next(chunk for chunk in store.visible(ctx) if chunk.doc_id == "update")
    claim = Claim(old.text, Citation(old.chunk_id, old.text))
    assert validate_claim(store, ctx, claim)[0]
    new_revision = store.ingest("project-a", "update", "Update",
                                "Jade validates a new generation before switching.",
                                allowed=("*",), expected_revision=old.revision)
    assert new_revision > old.revision
    assert not validate_claim(store, ctx, claim)[0]
    current = next(chunk for chunk in store.visible(ctx) if chunk.doc_id == "update")
    assert validate_claim(store, ctx, Claim(current.text, Citation(current.chunk_id, current.text)))[0]
```

回答前资料变化不是罕见异常，而是动态知识系统正常情况。产品可以重新检索、提示资料已更新或返回部分结果，但不能继续把旧句标为最新事实。若用户明确要求历史版本，应该使用单独历史查询契约，而不是绕过当前可见性检查随意读旧表。

## 小课七：持久化必须跨进程验证

下面命令使用你创建的实验目录。第一次 seed 写入文件，第二次 ask 是新进程，不再自动 seed。数据库仍能回答才证明持久化。示例不使用绝对个人路径，适用于 Windows 与 Linux 的仓库工作目录。

```bash
python -c "from pathlib import Path; Path('local-work').mkdir(exist_ok=True)"
python -m labs.knowledge.assistant --db local-work/knowledge.sqlite seed
python -m labs.knowledge.assistant --db local-work/knowledge.sqlite ask "KV cache"
```

全局参数 --db 在子命令前。若重复 seed，相同指纹不应增加版本。不要把 local-work 数据库提交仓库，里面虽然本次是虚构资料，未来你可能放入自己的材料。实验记录保存命令与必要输出，避免无条件导出全部正文到公共日志。

更强的持久化测试包括软删除后重开仍不可见，purge 后重开再建版本不复用，旧引用仍失效。文件存在不是数据正确，必须查询实际状态。备份恢复另有代际倒退问题，本项目只验证声明的文件重开语义，不夸大为完整灾备系统。

## 小课八：项目 B 的事件输入与写入策略

记忆助手接受显式 remember、recall、forget 与 demo。它不会自动从真实对话推断偏好。先用虚构事件清单：明确长期偏好、单次限制、项目事实、临时目标、更正、撤回与删除。手工标注哪些进入长期账本，哪些只在当前会话有效，哪些需要冲突处理。写入数量不是成功指标，错误永久化应被明确惩罚。

项目 B 的核心模块是 memory.py 与 memory_assistant.py。MemoryStore 负责 JSON 值、版本、状态、TTL 与墓碑；select_memory 是透明词法选择基线；CLI 负责显式操作。owner 参数在本地由调用者提供，不是网络认证。服务化时必须从可信会话注入，模型不能随意选择另一个人的 owner。

```python
from labs.knowledge.memory import MemoryStore, select_memory

with MemoryStore() as memory:
    first = memory.remember("learning-demo", "mira", "learning.style", "numeric examples",
                            "fictional-message-1", expected_revision=0)
    second = memory.remember("learning-demo", "mira", "learning.style", "complete derivations",
                             "fictional-correction-2", expected_revision=first)
    memory.remember("learning-demo", "mira", "temporary.goal", "finish graph exercise",
                    "fictional-message-3", expires_at=100)
    active = memory.recall("learning-demo", "mira", now=99)
    assert len(active) == 2
    assert len(memory.recall("learning-demo", "mira", now=100)) == 1
    assert memory.recall("learning-demo", "leo", now=99) == []
    history = memory.history("learning-demo", "mira", "learning.style")
    assert [row["state"] for row in history] == ["superseded", "active"]
    assert second == first + 1
```

这里 now 是显式测试时刻，避免等待与时钟波动。长期偏好没有 expires_at，不代表永远正确，用户可以更正或撤回；临时目标到期不再召回，但历史仍存在。value 可以是结构化 JSON，evidence 应指向明确事件而非“模型认为”。这些字段在本项目中都承担具体行为，不是装饰元数据。

## 小课九：记忆更正与遗忘的完整命令行轨迹

用新的实验数据库运行以下序列。第一条 expected 零断言当前没有该 key 的历史；第二次写入基于实际返回的 revision。这里全新库返回一，因此示例使用一。若你重复使用旧数据库，版本可能更高，应读取实际值，不能把教材数字当永远正确。forget 后再建虽然 expected 零，返回版本仍继续递增。

```bash
python -m labs.knowledge.memory_assistant --db local-work/memory.sqlite remember study.style "numeric examples" --evidence "fictional-message-1" --expected 0
python -m labs.knowledge.memory_assistant --db local-work/memory.sqlite recall
python -m labs.knowledge.memory_assistant --db local-work/memory.sqlite remember study.style "complete derivations" --evidence "fictional-correction-2" --expected 1
python -m labs.knowledge.memory_assistant --db local-work/memory.sqlite forget study.style
```

故意重复第二次 remember 的 expected 一，应得到冲突而不是覆盖。这说明并发保护生效。错误恢复是读取当前状态、比较来源再决定下一步，不是删掉 expected 参数。遗忘只保证课程内容表记录删除，防 ABA 哈希计数仍在；外部摘要、日志与备份需要独立策略，不能报告所有地方完全抹除。

CLI 的输出可以保存到自己的实验记录，但不要无条件把真实记忆 value 打印到公共日志。课程用虚构内容允许完整观察，未来个人项目应区分调试与业务权限。记忆属于谁、谁能管理历史、谁能 forget，都需要服务层认证与授权，本地参数只是教学模拟。

## 小课十：两个项目之间的连接点

知识助手提供文档证据，记忆助手提供用户明确偏好与项目背景。它们进入模型上下文时应分栏，保留不同来源类型。文档说明 API 机制，偏好说明讲解顺序，历史事件说明某次尝试结果，不能互相覆盖。用户偏好简短不允许省略关键证据，旧项目记忆不允许覆盖当前文档版本。

可以先实现一个纯函数 context_builder：输入当前授权证据与当前适用记忆，输出结构化 JSON。预算按最终序列化计算，不能只加正文长度。记忆被截断时可能丢“仅本次”等条件，所以宁可少选完整记录，不随意切句。输出给模型的身份字段来自运行器，模型只能引用，不能修改。

```python
import json


def combined_context(evidence_rows, memory_rows, max_chars=2000):
    payload = {"evidence": [], "memory": []}
    for section, rows in (("evidence", evidence_rows), ("memory", memory_rows)):
        for row in rows:
            candidate = {key: list(value) for key, value in payload.items()}
            candidate[section].append(row)
            encoded = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= max_chars:
                payload = candidate
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

packed = combined_context([{"source_id": "d1", "text": "fictional evidence"}],
                          [{"key": "style", "revision": 1, "value": "examples first"}], 500)
assert len(json.loads(packed)["evidence"]) == 1
assert len(json.loads(packed)["memory"]) == 1
```

这个函数按 evidence 后 memory 贪心填充，说明默认优先保证任务证据。它没有验证 row 内容、归属或来源，输入必须已经通过各自模块过滤。max_chars 若小于空包装长度，当前实现仍返回空包装且可能超预算，因此项目练习应加最低预算检查。公开指出这个边界，能让你把预算契约真正落实，而不是复制代码后假设所有输入都处理完了。

## 小课十一：十二道项目练习与完整答案

**练习一，摄入幂等。** 对 seed_project 连续执行两次，版本均一，revisions 数四。完整断言如下。若失败，检查指纹中的授权排序、来源和切块配置，而不是直接删除旧库重跑掩盖问题。

```python
with CorpusStore() as store:
    first, second = seed_project(store), seed_project(store)
    assert first == second == [1, 1, 1, 1]
    assert store.counts()["revisions"] == 4
```

**练习二，权限对照。** learner 搜 ORCHID 无结果，operator 可以得到 maintenance；另一个租户的 CEDAR 不进入 project-a。小课四代码覆盖前者，下面补管理员对照，防止通过把全部资料隐藏来假装隔离正确。

```python
with CorpusStore() as store:
    seed_project(store)
    hits = hybrid(store, Context("project-a", "operator", ("ops",)), "ORCHID")
    assert [hit.chunk.doc_id for hit in hits] == ["maintenance"]
```

**练习三，旧引用。** 更新 update 文档，旧 Claim 验证失败，新 Claim 成功。小课六提供完整实现。只断言旧失败不够，还需确认新路径仍可用，避免修复退化为永远拒答。

**练习四，删除重建。** 保存 old chunk_id，purge，再以相同正文和 expected 零创建，版本更高且新 ID 不同。完整答案如下，证明内容一样不等于对象同一代际。

```python
with CorpusStore() as store:
    seed_project(store)
    ctx = Context("project-a", "learner")
    old = next(chunk for chunk in store.visible(ctx) if chunk.doc_id == "update")
    store.purge("project-a", "update")
    new_revision = store.ingest("project-a", "update", "Update", old.text,
                                allowed=("*",), expected_revision=0)
    new = next(chunk for chunk in store.visible(ctx) if chunk.doc_id == "update")
    assert new_revision > old.revision and new.chunk_id != old.chunk_id
```

**练习五，记忆纠正。** 旧值 superseded，新值唯一 active，过期目标到时不召回。小课八完整代码可直接运行。再换 owner 查询必须为空，证明不是全局共享偏好表。

**练习六，记忆 ABA。** forget 后重建新版本，旧 expected 写入冲突。R09 的跨文件重开代码是完整参考；项目中把 key 换为 study.style，保留相同断言。不要只检查 history 空，因为那没有证明重建身份安全。

**练习七，安全导出。** 只导出当前可见片段，使用独占创建避免无意覆盖已有文件。下面函数返回条数，目标父目录由调用者明确准备，导出文件自身权限仍由实际环境管理。

```python
from pathlib import Path
from tempfile import TemporaryDirectory


def export_visible(store, context, destination):
    rows = [{"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
             "revision": chunk.revision, "start": chunk.start, "end": chunk.end,
             "text": chunk.text} for chunk in store.visible(context)]
    with Path(destination).open("x", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    return len(rows)

with TemporaryDirectory() as directory, CorpusStore() as store:
    seed_project(store)
    destination = Path(directory) / "visible.json"
    assert export_visible(store, Context("project-a", "learner"), destination) == 2
    exported = destination.read_text(encoding="utf-8")
    assert "ORCHID" not in exported and "CEDAR" not in exported
```

**练习八，组合预算下限。** 空包装 `{"evidence":[],"memory":[]}` 也占字符。完整修复先计算空包装长度，预算不足抛 ValueError，再调用原函数。真实 token 预算仍需另外计算。

```python
def checked_combined_context(evidence_rows, memory_rows, max_chars=2000):
    minimum = len(json.dumps({"evidence": [], "memory": []}, separators=(",", ":")))
    if type(max_chars) is not int or max_chars < minimum:
        raise ValueError("budget cannot hold empty envelope")
    return combined_context(evidence_rows, memory_rows, max_chars)

try:
    checked_combined_context([], [], 1)
except ValueError:
    pass
else:
    raise AssertionError("invalid envelope budget accepted")
```

**练习九，无答案。** 查询唯一未知标记，answer_report 应 abstain=True，claims 空。不能让模型背景知识填出看似相关内容。完整断言如下。

```python
with CorpusStore() as store:
    seed_project(store)
    report = answer_report(store, Context("project-a", "learner"), "UNSEEN_ZEBRA_987")
    assert report["abstain"] is True and report["claims"] == []
```

**练习十，故障回滚。** 在独立事务演示中插入部分行后抛异常，所有写入回滚。R08 给出完整 SQLite 代码。项目验收再检查活动文档版本不变，不能只看异常被捕获。外部向量库不在同一事务内，需要准备与发布状态机。

**练习十一，替换检索器。** 保留 Context、Chunk、Hit 与引用契约，只把候选排序替换为真实 dense_search。必须保持授权前置、模型配置与维度一致、响应 index 恢复、输出前版本复核。质量评价用标注相关来源，不用 ANN 近邻召回冒充语义效果。

**练习十二，完整演示。** 从新库开始执行摄入、查询、越权拒绝、更新、旧引用失效、关闭重开、记忆纠正、过期、遗忘重建。交付命令、预期与实际输出、一条故障及修复、未执行项。参考工程入口为 assistant、memory_assistant 与 projects.evidence_workshop，不接受只有功能愿望清单。

## 小课十二：真实 embedding 的完整替换链

课程 `labs.knowledge.embedding_api` 提供显式联网适配器与离线响应替身。先运行其帮助与默认离线模式，理解输入批次、响应 index、维度、模型标识、有限数值与 usage 完整性。离线合成向量只验证协议与数值行为，不代表真实语义。真实调用需要主动启用 live，并配置所选服务的 API 地址、模型与密钥。

真实链从当前授权 chunks 开始，按批发送正文与查询，恢复服务返回顺序，验证所有向量属于一致模型与维度，再用稳定 cosine_topk 排序，返回原 Chunk 的 Hit。课程 dense_search 每次重新编码当前小语料，透明但不适合大规模生产缓存。若加入缓存，键必须绑定 chunk revision、编码器身份、输入模板与权限语义，不能只用正文字符串。

```bash
python -m labs.knowledge.embedding_api --help
python -m labs.knowledge.embedding_api
python -m labs.knowledge.faiss_lab
```

FAISS 未安装时应明确 SKIP，不把跳过当通过真实性能实验。安装可选依赖应在独立环境进行，固定实际版本，再使用相同向量比较 Flat 精确近邻与近似索引。余弦相似需要文档与查询都规范化，ANN Recall@k 只测近似算法保留精确近邻多少，不等于这些近邻能回答问题。R04 已提供完整公式与索引实验解释。

外部服务使用不能默认把全部数据库发送出去。只发送当前授权范围且任务允许的资料，等待期间资料更新或撤权后应重新检查。API 返回 usage 缺失时记录 unknown，不写零；模型别名未固定时记录限制，不伪造版本。所有真实调用都应有小规模预算与明确输出位置，首次仅用本章虚构资料。

## 小课十三：从严格抽取升级到生成式回答

首先保持检索、权限、版本与来源不变，只增加一个生成适配器，输入已授权证据 JSON，输出结构化 claims 与引用。运行器验证 schema、来源 ID、引文与当前版本。生成式重述不再要求 claim.text 完全等于 quote，因此必须增加语义支持评价，不能简单删掉严格检查后宣布能力升级。

可以从受限模板开始：每条回答包含一个槽位、一个来源与一段原句，再逐步允许改写、跨句合并和多来源综合。每增加自由度，都加入否定、单位、条件、版本与冲突负例。模型提出的 source_id 必须来自输入，不能生成不存在的引用。没有证据时允许 abstain，不能为了完整回答补常识冒充资料。

如果要加入 Agent 查询规划，使用 R12 的状态、预算与工具闭包，或直接学习综合项目 P1。不要同时更换模型、检索器、存储和框架，否则任何质量变化都难以归因。先把一个接点替换清楚，保留回滚与基线，再扩展下一层。

## 小课十四：项目测试应该覆盖状态变化

最有价值的测试不是函数返回类型，而是关键状态转移。摄入前后版本与切片、授权变化前后可见集合、更新前后引用、删除重建前后身份、记忆更正前后 active 状态、TTL 边界、跨进程恢复与旧 CAS。每个测试用最小数据，预期可手算，失败时指出哪条契约破坏。

测试套件默认离线，不下载模型、不调用 API、不要求 GPU，不用随机时间断言。可选真实集成单独标记，记录依赖、版本与实际结果。一个测试跳过说明环境不具备该能力，不能在汇总里把它算成已执行成功。相同原则适用于你自己的服务器实验。

```bash
python -m unittest discover -s tests -p 'test_knowledge*.py' -v
python -m projects.evidence_workshop all
python -m unittest tests.test_projects_workshop -v
```

项目工作坊把范围、排名、轨迹、删除、更新、清理重建、身份伪造、引用伪造与进程重开做成独立场景。先跑一个场景理解，再跑 all 回归。不要因为全套绿色就停止人工检查，测试只覆盖写出的情况；新增功能必须增加对应边界与质量评价。

## 小课十五：代码审查与交付报告

请让另一个学习者沿一个问题读你的调用链，不解释先看他能否找到身份、来源、版本与预算。若这些信息藏在全局变量或模糊字符串中，改数据结构与命名。可读性不是美化代码，而是降低未来维护者误用接口的概率。特别是管理员写接口与用户读接口，应有清楚边界。

交付报告先写最终行为，再写实现与验证。列出真实执行命令与结果、未执行 API 或可选依赖、一个失败案例与修复。不要把“支持 GraphRAG、长期记忆、Agent”作为没有细节的功能标签；说明具体实现了哪种图查询、哪种记忆生命周期与哪种策略控制。

最终能力检查是独立重写一个小版本。关闭参考实现，用新的虚构主题完成同样契约，再对照测试。允许查 Python 语法，不允许用复制整项目代替理解。能解释为什么某个旧引用必须失效、为什么删除后版本不能复用、为什么未知 usage 不等于零，你已经把前面章节连成真实工程能力。

### 项目 A 的第一小时：先验证数据，再验证答案

第一次实现时，先让 seed_project 写入四份资料并打印 counts，不启动任何模型。接着分别读取 learner、operator 与另一租户的可见集合，和手工表比较。再检查每个 chunk 的 start、end 与原文切片一致。最后才执行检索。这个顺序防止你在资料根本没摄入或权限错了时，花时间调排名与生成参数。

给每个阶段一个明确通过标准。摄入通过是逻辑文档与版本完整，不是文件读取没有异常；权限通过是该看见的看见、不该看见的没有进入候选，不是最终界面隐藏了一段文字；检索通过是标注相关来源进入候选，不是分数很高；回答通过是断言与来源对应，不是语言流畅。每个标准都能转换成一两条断言。

如果发现文档正文为空，先检查输入与解析，不要自动当删除；若片段区间错，回 R02；若权限集合错，回 R08；若相关来源排名低，回 R03–R05；若检索正确但答案引用错误，回 R06。项目不是把所有问题混成一个大任务，而是训练你沿模块边界定位。

### 项目 A 的第二小时：观察真实数据库行

可以在自己的实验副本中使用 SQLite 查询 documents、revisions、chunks、grants 与 edges。先看 documents 的 current_revision，再看对应 revisions 正文，最后看 chunks 偏移。更新后旧 revision 仍在，但当前指针改变，visible 只选择新版。亲眼看见这组关系，比只背“版本化存储”更容易理解。

观察 grants 与 revision 绑定。权限改变时新版本拥有新的授权，旧授权仍属于旧快照；当前读取不会因为旧版曾公开就继续公开新版。若你把权限只保存在 documents 当前行，也能设计另一种系统，但需要重新定义历史访问与缓存失效语义。不要随意改表而不更新契约与测试。

再查看 document_revision_counters。purge 内容后，五张内容表中的目标对象消失，墓碑仍保留哈希与最后版本。这个表不是业务正文，却承担身份安全。测试与报告应明确它的存在，不能因为 counts 没包括它就说所有元数据已经删除。稳定哈希的隐私边界也要如实说明。

### 项目 A 的第三小时：建立可解释排名报告

为一个查询保存词元、候选文档、各路分数、融合名次与最终上下文。BM25 分数来自 tf、df、长度和参数，TF-IDF 来自词法向量，RRF 来自排名；不要放在一列统称 confidence。可以使用 JSON 字段 score_type，方便后续换检索器仍保留含义。

报告同时显示没有进入上下文的相关候选及原因，例如预算不足、重复来源、版本过期或权限变化。这样你能区分“检索没找到”和“找到但打包没选”。如果只保存最终答案，许多失败会被误归到模型。上下文预算应计算最终序列化载荷，来源 ID 与元数据也占空间。

把查询中一个词改成未知词，观察候选变化；把一份文档重复添加为不同逻辑来源，观察 df 与融合变化；增加一份受限文档，普通用户的可见统计应按课程语义保持不受它影响。这些控制变量实验让你知道系统实际在优化什么。

### 项目 A 的第四小时：故意制造旧答案

先缓存一个 answer_report，然后更新源文档。直接打印缓存会看到旧内容，这是预期故障；通过当前引用验证应拒绝旧 chunk。修复缓存需要记录来源依赖与版本，命中时复核或按语料代际失效。仅把缓存过期时间改短无法保证权限撤销立即生效，也不能证明版本一致。

再让渲染器使用旧数字模板覆盖新 claim，观察中间 JSON 正确而最终显示错误。验收应检查最终用户产物，不只检查内部对象。这个故障在真实应用中很常见：业务模板、前端缓存或字符串拼接可能重新引入旧值。把渲染层纳入测试，才算端到端。

最后删除唯一支持某答案的文档。正确系统拒答或部分回答；若仍输出同样结论，检查模型背景知识、硬编码答案、旧缓存与会话历史。这个证据消融实验是检索项目最有力的质量检查之一，因为预期变化非常明确。

### 项目 B 的第一小时：写入决定与持久化分开

先用纯函数判断候选是否保存，输出 save、session_only、review 或 reject，不操作数据库。对十二条虚构事件手工标注，再比较策略输出。这样你可以独立调写入规则，不会在每次测试中留下大量错误记忆。通过策略后再调用 MemoryStore，持久化失败与策略拒绝分别记录。

事件清单应包含引用别人的偏好、假设句、否定保存、仅本次要求、明确长期要求、项目范围差异、纠正与撤回。简单关键词“记住”不够，例如“不要记住”也包含它。模型候选抽取需要原文支持与范围判断，不能把 JSON 合法当语义正确。

对于项目事实，证据最好是具体工具执行或文档版本；对于个人偏好，证据是明确消息；对于程序经验，证据是可重现任务结果。不要把所有 evidence 写成一个固定字符串“user”，这样虽然接口通过，来源链却没有实际信息。证据身份是为了以后能纠正与解释，不是为了填满必填参数。

### 项目 B 的第二小时：用时间表理解状态

画一张时间表：时刻一保存偏好旧值，时刻二更正，时刻三临时目标到期，时刻四撤回偏好，时刻五重新明确保存，时刻六遗忘。每个时刻写下 recall 应返回什么、history 应包含什么、最新 revision 是多少。然后用固定 now 与 expected_revision 跑代码对照。

注意过期不会自动将 state 改成 retracted，它只是查询条件排除；retract 保留历史与版本；forget 删除内容历史但保留计数；重新写入都会获得更高版本。四种行为如果在你的表中都写成空列表，就说明只观察了召回，没有理解底层状态。必须同时检查历史与版本。

再加两个 owner 与两个 tenant，重复同一 key。当前实现应完全隔离。若未来共享项目记忆，需要显式共享身份与授权层，不要通过去掉 owner 条件来实现。共享能力是新需求，需要新契约，而不是简单删一行过滤。

### 项目 B 的第三小时：召回为什么不是越多越好

给同一任务注入四种上下文：无记忆、正确偏好、无关事件、错误或过期偏好。比较回答是否改善、是否被误导、是否暴露不必要信息。正确偏好可能让讲解顺序更合适，无关事件可能浪费 token，错误偏好可能强迫错误风格。记忆收益应通过任务表现评价，而不是条数。

select_memory 的词法重叠基线容易解释，也有明显局限。中文英文同义词可能匹配不到，项目范围也不由它理解。因此先做硬范围过滤，再使用相似度；真实 embedding 可以提高召回，但不会自动解决归属、TTL 与适用条件。把更强检索与更可靠治理分开改进。

记忆展示应让用户能纠正。可解释记录包含 key、当前值、来源与更新时间或版本，不应形成无法查看的隐藏画像。课程 CLI 是最小管理界面，真实产品可以更友好，但不能牺牲更正与删除能力。长期记忆的可信度来自可控生命周期，而非永不遗忘。

### 项目 B 的第四小时：防止遗忘后复活

先保存偏好，再 forget，再从旧会话摘要自动抽取同一偏好。若系统立刻重新保存，说明删除没有影响重新摄入路径。墓碑只防旧版本号复用，不阻止同内容被新请求写入；产品需要明确区分用户重新授权保存与旧来源自动复活。来源撤销状态、摘要重建或身份代际都可能参与解决。

在课堂项目中可以最简单地不做自动历史抽取，只接受新的显式 remember，这样遗忘语义清楚。若扩展自动记忆，必须增加旧来源撤销测试。不能因为自动抽取很方便，就把用户的删除要求变成暂时清空一次数据库。

检查点也可能保存旧记忆副本。恢复 Agent 时要复核记忆 ID 与 revision，而不是直接继续使用冻结的提示词。这个实验把项目 B 与 Agent 恢复连接起来，证明长期状态变化能够影响正在进行的任务。

### 两个项目的接口适配练习

选择一个外部框架时，先映射 Context、Chunk、Hit、Claim、Memory 与其对象。找不到版本、权限或来源字段时，通过 metadata 或适配器显式保存，不要因为框架示例没展示就丢弃。返回对象再转换回课程契约，继续运行旧引用、撤权与删除测试。

适配器应该尽量薄，只做字段与调用转换，不偷偷改变语义。例如框架 top_k 默认值不同要显式传入；相似度方向可能越小越好，需要标记与转换；文档 ID 与节点 ID 不能混用；embedding 模型维度一致但版本不同不能混搜。每个差异写进对照表与测试。

如果框架没有提供 CAS 或完整删除传播，不代表它不好，而是应用要补齐这些需求。工程能力是知道组件承诺什么、自己负责什么。把所有功能标签都归给框架会让系统在边界处失控。

### 一个实际的版本升级演练

假设把切块参数从每片二百四十字符、重叠三十二改为更短窗口。相同正文会产生新指纹、新 revision 与新 chunk IDs，旧引用失效，向量与图边需要重新构建。你应先在小语料上比较片段覆盖、检索质量、上下文成本与引用完整性，再发布，不是只把参数改小观察片段数量增加。

升级 embedding 模型则需要新的坐标空间与索引代际。旧文档向量不能与新查询向量混用，缓存键要包含模型配置。先并行构建新索引，使用固定标签比较，再切换活动版本；回滚回完整旧代际。即使模型名称只是服务别名，也要记录无法完全固定的限制。

升级摘要或记忆提取提示同样是行为变更。保存模板版本与回归结果，检查新规则是否把临时偏好永久化、是否丢例外、是否改变来源映射。配置不是无风险文本，它会改变系统输出，需要与代码一样接受可复现验证。

### 故障报告的完整写法

一份有用报告先写触发与实际后果，例如“普通用户重复查询命中了管理员答案缓存，返回受限片段”；然后写预期边界、最小复现、第一次错误状态、修复与验证。不要只写“优化缓存安全性”，那无法让审查者理解问题。

最小复现只保留必要数据：两个用户、一份公开文档、一份受限文档、同一查询。修复后同时验证管理员仍能读取、普通用户不能读取、权限变化后旧缓存失效。这样证明修复没有通过禁用全部功能来回避问题。若某外部缓存未在测试范围内，明确写未验证。

性能报告也要同样诚实。小语料一次运行更快可能只是计时噪声，不能宣称提升百分之多少；可选 FAISS 缺失就写跳过；真实 API 没调用就写协议替身验证。实际执行范围越清楚，后续扩展越容易接续。

### 项目验收中的人工部分

自动测试擅长身份、版本、状态与精确引用，较难完整判断自然语言支持、例外与任务充分性。人工审查应集中在这些语义边界，不重复检查机器已经精确验证的字段。可以逐条列出 claim、quote、条件、结论是否扩大范围，让审查有明确标准。

对于生成式答案，至少抽查否定、数字单位、时间版本、比较对象与跨句推断。一个数字正确但单位错，仍是错误；引用存在但不支持结论，仍是错误；来源过时但未标历史，仍是错误。不要让流畅文风影响事实判断。

把人工发现转成可运行回归，能自动化的部分就自动化。例如某次遗漏版本条件，可以增加结构字段；某次伪造引用，增加来源集合检查；复杂语义反例则保留人工标签或受控判别评测。测试资产应随着项目成长，而不是只在第一次提交时写一次。

### 最终演示脚本的顺序

从全新临时目录开始，先展示当前没有数据库；摄入虚构资料；普通用户查询公开事实；尝试私有标记并拒绝；管理员读取同一私有资料；更新公开文档并展示旧引用失效；关闭重开确认持久化；创建记忆、更正、过期、遗忘与重建；最后运行测试汇总。每一步有预期与实际输出，不需要漂亮前端也能展示完整能力。

演示中故意保留一个可控失败，例如旧 expected_revision 写入冲突。解释为什么这是正确保护，而不是程序坏了。再展示错误恢复：重新读取当前版本、比较来源、明确提交新更正。让观众看到系统怎样处理变化，比只展示一条成功路径更能说明工程质量。

最后给一个陌生问题，先预测它需要哪些证据，再运行检索与报告。如果结果不符合预期，沿中间产物定位，不立即修改 prompt 祈求答案变化。独立诊断能力才是这一卷的核心学习成果，也是继续深入 Agent 与框架源码的基础。

### 两份交付报告的参考骨架

知识助手报告的第一段应说明最终行为：当前授权文档能够被检索并输出版本化原文引用，未知查询拒答，更新与删除使旧引用失效。第二段说明模块边界：身份由调用者可信注入，存储与检索分离，默认严格抽取，语义真实性仍需评价。第三段列实际测试与未执行集成。不要开头写“实现了先进智能系统”，让读者猜它到底能做什么。

记忆助手报告先说明显式保存、当前召回、CAS 更正、TTL、撤回与遗忘范围。随后说明数据来源是虚构事件，未启用自动真实画像，墓碑保留防 ABA。再列跨 owner、跨租户、边界时间与重启测试。这样的报告让审查者能够据事实判断，不被功能名称误导。

两份报告都应附一个最小失败与修复。例如知识缓存遗漏权限版本，修复键与输出复核；记忆旧总结在 forget 后复活，修复写入来源策略。写出具体触发、前后行为和测试，而不是只写“增强鲁棒性”。如果你能用一页讲清这些变化，说明你已经能把代码细节组织成工程判断。

### 如何安排一次真正独立的复做

隔一天换一个虚构主题，例如植物园设备手册，而不是继续使用 Jade。重新定义两租户、两权限组、更新与恢复文档、临时记忆和纠正事件。先写预期表，再实现，不看旧答案。语法可以查，接口可以看文档，但关键状态变化应自己推导。

完成后用参考实现交叉比较：相同输入的可见集合、版本变化、引用失效与记忆状态是否一致。排名分数若因语料不同而变，不是错误；权限或 CAS 语义不同则需要解释。把差异分成有意设计与实现缺陷，训练你不把所有不一致都归为坏事，也不把所有差异都当自由发挥。

最后请同伴随机修改一份资料或撤销一个授权，不告诉你具体位置，让你从失败结果沿日志定位。能在可见集合、索引、上下文与引用层快速找到第一处变化，说明你拥有调试模型；若只能重新运行所有脚本碰运气，应回到相应章节补状态追踪。这种复做比继续收藏更多框架链接更能巩固能力。

项目结束后保留一个可运行的最小版本作为长期基线。未来加入新模型、图数据库或网页界面时，先跑基线合同测试，确认没有丢失身份、版本、预算与删除语义，再评价新增功能。这样每次扩展都有可比较起点，不会因为工程越来越大而失去最初已经验证的可靠行为。

你的代码可以与参考实现不同，只要契约清楚且证据充分。真正需要保留的是可追溯数据流、可解释状态转移和诚实的验证范围；这些能力比记住某个函数名字更能支持后续深入开源框架。

每次复做都保留一个新反例，让测试集随你的理解一起增长，而不是只重复验证最初那一条成功路径。
