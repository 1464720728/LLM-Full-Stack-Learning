# 长工作簿：一条输入怎样穿过整个知识系统

这份工作簿供你把 R01–R14 的知识变成读代码能力。它刻意放慢速度：不从“RAG 框架组件”开始，而从几个 Python 值、几行 SQL 和一个能手算的查询开始。先预测，再运行，再解释差异。每一站都有独立命令，修改失败后不需要重新搭整套环境。

所有命令从仓库根目录运行。使用 Python 3，标准库足够，不需要安装数据库服务器、GPU 驱动或模型客户端。代码在 [walkthrough.py](../../labs/knowledge/walkthrough.py)，它调用真正的课程模块，没有复制一套只为演示而不被主项目使用的算法。下面的文档 ID、词句、身份与偏好都是虚构教材数据。

开始前执行：

```bash
python -m labs.knowledge.walkthrough --stage all
python -m unittest tests.test_knowledge_system -v
```

第一条显示八个检查点，第二条检查实现的正确性。此时你不需要读懂所有输出；保留它作为后面逐站比较的基准。不要把测试通过理解成“每个生产问题都解决了”：测试只证明列出的契约在相应样例与边界下成立。

## 第零站：先认识你正在运行的 Python 程序

`python -m labs.knowledge.walkthrough` 的 `-m` 表示按模块名运行。Python 从当前仓库寻找 labs 包，再进入 knowledge，再找到 walkthrough.py。点号表达包层级，不是磁盘文件名里的点号。这样模块里的相对导入 `.store` 才能按包关系正确解析。

文件末尾的 `if __name__ == '__main__': main()` 表示：作为程序入口运行时调用 main；被其他模块 import 时只定义函数，不自动执行整个实验。这个设计使测试可以导入 `stage1` 而不在导入瞬间创建数据库、打印大量输出或调用网络。

`argparse` 把命令行 `--stage 2` 转成 `args.stage == '2'`。STAGES 字典把字符串 `'2'` 映射到 stage2 函数。注意字典里保存的是函数对象，构建时没有加括号，所以没有立即执行。等调用 `STAGES[args.stage]()` 时才运行该实验。

输出用 `json.dumps(..., ensure_ascii=False, indent=2)`。ensure_ascii=False 让中文直接显示，indent=2 只影响排版，不改变数据。JSON 没有 Python tuple 类型，因此 tuple 会显示成数组；看到输出是数组，不表示内存里的所有对象原本都用 list 表示。

把这段入口读懂，你已经获得一个可迁移的读源码技巧：先找程序入口、参数解析、函数映射、数据返回与副作用位置，再追算法。复杂框架也可以用同样方法拆开，只是函数调用层数更多。

**零站检查：** 解释导入一个模块为何不应自动启动外部 API 调用；解释函数名和函数调用的区别；运行 `--help` 并说出 allowed choices 的作用。若这些仍模糊，回到基础卷的模块与 CLI 章节，解决后再继续。

## 第一站：四份资料怎样变成三份可见资料

运行：

```bash
python -m labs.knowledge.walkthrough --stage 1
```

你应看到 documents、revisions、chunks、grants 都是 4，edges 为 0；visible_docs 是 d0、d1、d2，缺少 private。这不是资料没存进去，而是当前调用者不属于管理员组。

`build_case()` 创建 CorpusStore，再逐条调用 ingest。四份正文为：d0=`cat cat dog`，d1=`cat fish`，d2=`dog fish`，private=`cat canary secret`。前三份 allowed=(' * ',) 的实际写法没有空格，即 `('*',)`；最后一份 allowed 为 `('group:admins',)`。括号里一个字符串后面的逗号很重要：`('x',)` 是单元素 tuple，`('x')` 只是带括号的字符串。

Context 是 `Context('workshop','learner',('readers',))`。它代表 workshop 租户的 learner 用户，属于 readers 组。`principals` 产生 `'*'`、`'group:readers'`、`'user:learner'`。这里 wildcard 在候选身份集合中用于匹配“租户内公开”的授权记录，而不是让调用者变成全系统管理员。SQL 的 tenant 条件仍独立存在。

阅读 `CorpusStore.__init__`。`sqlite3.connect(':memory:')` 建立只在本连接生命期存在的数据库；给一个文件路径则持久化在磁盘。`row_factory=sqlite3.Row` 使查询结果除了按位置，还能按列名访问，例如 row['title']。这比到处写 row[4] 更便于理解和防止字段顺序错误。

SQLite 表可以先理解成“带约束的记录集合”。documents 每行记录一个逻辑文档；其复合主键 `(tenant,doc_id)` 表示同一租户不能有两个相同 doc_id，而不同租户可以使用同一 doc_id。revisions 的主键多了 revision，允许一个文档存在多个版本。chunks 的 chunk_id 是单列主键，同时通过外键指回一个合法 revision。

外键就是“这里引用的对象必须存在”的约束。只写了 FOREIGN KEY 声明还要在本连接开启 `PRAGMA foreign_keys=ON`，因此初始化立即执行它。否则代码可能写入找不到正文来源的孤儿 chunk。外键不负责用户身份，也不会判断一段原文是否支持关系；它只维护关系结构的存在性。

进入 `ingest` 后，先检查 tenant/doc_id/title/source 非空，再检查授权字符串形状。随后切块并计算 fingerprint。直到纯计算完成才 `BEGIN IMMEDIATE` 开启写事务。事务里查旧文档，没有则 current=0；再读 document_revision_counters 中该身份最后分配的版本。对从未出现的身份首个版本为 1，插入 documents/revisions/grants/chunks 并更新计数器，最后一起 commit。purge 后重建则沿用计数器继续递增，防止同一正文产生旧 chunk_id。

如果写到一半出现异常，except 执行 rollback 并重新抛出异常。`raise` 不会吞掉错误，所以调用者和测试可以看到问题。重新抛出比打印“失败”然后返回正常对象更可靠，因为后者容易让上层误以为摄入成功。

再看 `visible` 的 SQL。它从 chunks 开始，join documents 得到当前指针，join revisions 得到标题和来源；WHERE 同时要求租户相同、文档没删、chunk 版本等于当前版本；EXISTS 要求至少一条 grants 与 Context.principals 匹配。EXISTS 的意思是“存在一条即可”，不是把同一 chunk 因匹配多个 grants 而重复返回。

`IN (?, ?, ?)` 的问号由参数列表绑定。程序根据 principals 数量只构造问号数量，不把用户文本直接拼进 SQL。最后参数是 `(ctx.tenant, *ctx.principals)`。星号在这里展开 tuple；它不是 SQL 的 wildcard。不同语言层里的相似符号可能含义完全不同，读代码时必须逐层区分。

最后 `Chunk(**{...})` 做两件事：字典推导从行里选出 dataclass 所需字段，双星号把字典展开成命名参数。返回的是稳定字段的对象列表，而不是任意数据库行，这构成存储层与检索层的接口。

**第一站动手：** 将 learner 的 groups 改成 ('admins',)，预测 private 将出现，文档总数保持 4；将 tenant 改成 another，预测全部不可见；把 d0 的 allowed 改为空 tuple，预测 d0 不可见而非公开。每次只改一项，并解释是哪条 WHERE 条件改变结果。

## 第二站：Counter 如何变成 BM25 的每一个数

运行：

```bash
python -m labs.knowledge.walkthrough --stage 2
```

输出 tf 应为 `{'cat':2,'dog':1}`、`{'cat':1,'fish':1}`、`{'dog':1,'fish':1}`。df 中三个词都为 2。这里 private 文档完全没有进入统计，所以 canary 和 secret 不存在，cat 的 df 也不是 3。这一细节证明权限过滤发生在统计之前，而不是最后删结果。

`Counter(tokenize(c.text))` 接收词元列表并计数。`Counter` 的键是词，值是次数。迭代 Counter 本身得到不同的键；迭代 `.values()` 得到次数。因此 `sum(t.values())` 算文档长度，`for t in counts` 则给 df 提供每文档不同词。两行看起来都在遍历 Counter，数学意义却完全不同。

df 的生成器 `Counter(t for counts in self.tf for t in counts)` 可以展开成等价的三层操作：先建立空计数器，逐个文档取其不同词，对每词加一。把紧凑表达式展开写在纸上，能有效避免“看懂语法但没看懂统计”。

长度数组是 `[3,2,2]`，总长 7，N=3，avgdl≈2.3333333333333335。计算使用浮点除法，不是向下取整。若你误用 `//`，avgdl=2，所有长度归一化都变了。程序仍可运行，但排名分数不同，这属于静默数值错误。

`idf('cat')` 中 df=2、N=3。`math.log1p(x)` 计算 ln(1+x)，这里 x=0.6，所以约0.470003629。log1p 不仅名字方便，在 x 很小时也通常具有较好的数值精度。不要把自然对数与常用对数混用后还期待数值完全相同。

`contributions('cat',0)` 先 tokenize 并 set 去重查询，查 f=2，长度项约1.457142857，得0.5981864372218453。文档1得到0.4991762683023675。文档2因为 f=0 走 continue，不向字典加入 cat；它的贡献字典是空字典，sum(empty.values()) 按 Python 约定为0。

`search` 为所有文档生成 score，然后 `_rank` 丢掉不为正或非有限的分数，再稳定排序。math.isfinite 用于排除 NaN 与无穷值；NaN 的比较行为不符合普通大小直觉，让它进入排序会使问题难定位。`k=0` 应返回空列表；负 k 应抛错，不能把负切片当成合法 top-k。

**第二站动手：** 在一个新文件中只 import Chunk 与 BM25，自己构造三个文本，不使用 build_case；逐项打印 N、df、长度、归一化项与最终分数。然后加入一个空文档，观察 N 与平均长度改变。空文档有存在身份但没有词，这个选择会影响统计，真实系统可在摄入时明确拒绝空文档或允许它，不能不知情地混用策略。

**第二站失败定位：** 如果 cat 排名反了，先检查 tf 是否真为2和1，再检查长度分母；如果所有结果空，检查 tokenizer 是否输出英文单词而非单字符；如果 secret 出现在普通用户 df 中，问题在权限候选构造而非公式。用这棵小诊断树定位，比随机改 k1 更可靠。

## 第三站：同样候选为什么出现两种分数

运行：

```bash
python -m labs.knowledge.walkthrough --stage 3
```

你应看到 BM25 的 d0≈0.598186、d1≈0.499176；TF-IDF 余弦的 d0≈0.861037、d1≈0.707107；RRF 的 d0≈0.0327869、d1≈0.0322581。三组分数都把 d0 放前面，但不在同一个量纲，也不应解释成置信概率。

TF-IDF 的查询向量是 `{'cat':1.2876820724517808}`。这里平滑 IDF 使用另一条公式 `ln((1+N)/(1+df))+1`，所以不是前面的0.4700。不同算法可以共享 df，但权重公式并不相同。

d0 的 cat 频率2经过 `1+ln2≈1.693147`，dog 频率1权重因子为1。二者再乘相同平滑 IDF。与只有 cat 维度的查询做余弦时，共同缩放会约掉，最终 d0 相似度约0.861037。d1 的 cat 与 fish 权重相同，所以与 cat 轴夹角45度，余弦1/sqrt2≈0.707107。这是一个可以不用代码独立确认的几何解释。

`cosine` 用稀疏字典表示向量。点积遍历 a 的每个非零维度，用 b.get(term,0) 取得对应数，不存在则为0。范数由平方和求根。如果分母为0，按课程契约返回0。这里不是“所有未知语义都相似度0”的现实结论，只是词法坐标里没重叠或没有向量。

RRF 对两个排行榜分别从1开始枚举。d0 在两个列表均第一，总分2/61；d1均第二，总分2/62。原始0.598和0.861没有被直接相加。seen 集合使同一列表重复的 ID 只贡献一次。chunks 字典保留 ID 到对象映射，totals Counter 保留 ID 到融合分数，最后再还原 Hit 对象。

如果把第二个榜单完全反转，两候选各有第一与第二，融合分数相同。代码将按 chunk_id 做 tie-break。这个结果不代表检索器共同证明两条文档同样相关，只代表当前融合公式无法区分；你应在日志中识别这种情况，而非为稳定排序赋予虚构语义。

**第三站动手：** 人工构造三个 Hit，让两个榜单对某个候选有一致支持，然后验证它排名升高。再把 BM25 的全部分数乘100，确认 RRF 完全不变。最后写一个直接加分的版本，对比为何其排序会受这次缩放影响。这是理解排名融合与分数融合的最小实验。

## 第四站：候选如何进入上下文并成为可检查的答案

运行：

```bash
python -m labs.knowledge.walkthrough --stage 4
```

cat 查询返回d0、d1的原文，unknown_abstains与restricted_abstains都为true。后两者表面上都是拒答，原因却不同：elephant在可见语料里不存在；canary在系统数据库中存在，但当前用户不可见。面向用户的结果不必泄露后者的存在，内部诊断可以在有权限的审计环境中区分。

进入 `pack_context`。函数从空列表开始，为每个 Hit 创建一个字典，包含source_id、doc_id、revision、text。它尝试把“已有条目+新条目”序列化成JSON，检查整个字符串长度是否小于max_chars。只有通过才真正append。这个先构造候选再检查再提交的模式，避免加入后再猜应该删哪里。

预算中包含键名、引号、逗号等包装字符。若只统计text长度，可能正文看似没超限，但真正发送的消息超限。我们仍强调这只是字符演示；换成目标tokenizer后，还需统计message和tool格式的真实开销。

当一个块放不下时，本实现继续尝试后续更小块，而不是立即终止，也不是自动截断。这样可能出现低排名短块被纳入、高排名长块被跳过，属于明确的贪心装箱策略。是否符合任务需求需要评估，不能把它误当最优上下文选择。你可以设计“保证首个证据必须保留”或“按价值/长度选择”的替代策略，但测试要说明代价。

`extractive_answer` 把每个上下文条目的text直接作为Claim.text，并把相同text作为Citation.quote。这里没有一个隐藏LLM，所以输出不会改写词句。它能保证严格抽取形式，却不会像真正助手一样回答“为什么”或综合两个条件。教学上先把这一条可验证链建立起来，再增加生成能力。

`validate_claim` 通过store.get_visible_chunk再次找来源。查不到时统一返回source_missing_stale_or_forbidden，这避免向普通调用者区分“确实有一个你无权读的ID”。查得到后检查quote非空且是原文子串，再要求claim.text==quote。这最后一个条件是极严格的抽取限制，不是通用自然语言蕴含判断。

报告中记录doc_id、revision、source和chunk_id。source字段来自摄入时的来源标识，不会自动访问外部URL。若资料带了恶意链接，单纯展示来源也不意味着允许程序去访问或下载。

**第四站动手：** 将Claim.text改为“cat一定比dog好”，保留合法quote，观察校验拒绝；将quote改为不存在的文本，观察另一个拒绝原因；将source_id换成private的真实ID，普通Context仍拒绝。三个失败对应不同错误层，不应统一叫“模型幻觉”。

## 第五站：更新发生后，旧引用为什么不能继续冒充当前事实

运行：

```bash
python -m labs.knowledge.walkthrough --stage 5
```

before应为true，revision变成2，old_after_update为false，stale_write_rejected为true。仔细区分两种旧东西：旧引用属于revision1；旧写入意图也是基于revision1，但它们通过不同检查被拒绝。

我们先取得d0的Chunk，构造旧Claim。此时它与数据库当前状态一致。随后ingest正文`cat dog updated`并携带expected_revision=1。事务读取当前版本也是1，所以允许生成2并切换current_revision。旧revision行仍在数据库，旧Chunk对象也仍在Python内存里，但读取当前可见集合时不会返回它。

因此“我的程序还有旧对象”不意味着“它现在仍然可被用来回答”。对象快照和当前授权状态不是同一个概念。get_visible_chunk用新的查询复核，是为了让输出遵守当前规则。这也说明不能仅以chunk_id哈希稳定就跳过所有权限与版本判断。

接下来第二个更新仍携带expected_revision=1，但数据库已到2，函数抛ConflictError。try/except在工作簿中捕获它并把conflict=True。真正编辑器应该让用户看到两份修改并决定合并，或者依据明确业务规则重算。对模型自动改文档，也应该先重新取当前版本，而不是偷偷去掉CAS检查。

如果发生异常，rollback把本事务尚未提交的修改撤销。这里没有启动新事务写任何日志，所以不会出现“错误处理又提交了半份数据”的复杂性。将来加入审计日志时，可以选择同事务记录成功事件，失败事件另开事务记录，但要清楚哪些事实已经提交。

**第五站动手：** 在两次ingest之间打印documents、revisions和chunks计数。更新后documents仍是4，revisions和chunks增加；visible_docs仍只有一个d0。然后重新摄入完全一样的revision2输入，确认幂等返回2，而不增加3。注意参数也必须一致，不能忘记max_chars/overlap或source变化会影响fingerprint。

**第五站故障挑战：** 临时在插入revisions后、插入chunks前加入`raise RuntimeError('planned failure')`。重新运行一个独立测试，确认current_revision仍为旧值、revisions行数没有增加。完成后恢复代码并跑测试，不要把故障注入留在主实现。

## 第六站：一条记忆如何被纠正，而不是复制两份互相打架

运行：

```bash
python -m labs.knowledge.walkthrough --stage 6
```

active只含`step by step`，history_states为`['superseded','active']`，after_forget为空。这里使用的是MemoryStore，不是CorpusStore，因为我们跟踪的对象不同：资料按文档和片段引用；记忆按tenant/owner/key描述一个可纠正的知识项。

记忆表的复合主键是tenant、owner、key、revision。value先通过json.dumps转换为JSON字符串，allow_nan=False拒绝非标准JSON数值NaN或Infinity。这样以后读取时能可靠json.loads。保存任意Python对象并不是合法持久化契约，尤其函数、连接和文件句柄不能当普通记忆值。

写入时查历史最大的revision，不仅查active。即使上一版已retracted，新版本也应继续递增，而不是假装历史不存在。expected_revision与这个当前最大版本比较。然后把旧active标为superseded，再插入新active，最后提交。两条SQL处于同一事务，所以外部读者不会在已提交状态看到两个active版本。

recall先按tenant/owner过滤，再检查state='active'与expires_at。如果expires_at为NULL，表示没有设到期；若有值则必须大于now。SQL中的NULL不能按普通等号比较，因此用了IS NULL。这个细节是初学者读SQL时常见的坑。

`now`可以由测试传入。使用可注入时钟，使我们不必sleep等待到期，也不会因为运行机器时间不同而随机失败。真实服务可以统一使用UTC时间戳，但还要处理时钟偏差和业务有效期语义，课程只演示一个明确的比较规则。

history是管理审计入口，它会返回旧值；它不应自动拼进模型上下文。否则你虽然把旧active改成superseded，生成模型仍从history看到旧偏好并误用。数据状态正确与调用路径正确必须同时成立。

forget执行SQL DELETE删除该key的所有版本。删除后history空，不代表其他日志、备份或已经发给外部模型的数据自动撤回。这种范围说明应作为接口契约，而不只是合规段落；它直接决定测试到底能证明什么。

还有一张 memory_revision_counters 不参与 history：它保留哈希身份与最高版本，防止删除再创建时复用旧 revision。假设第一次为1，forget后 expected_revision=0 可以创建，返回2；旧线程仍拿 expected_revision=1 来写必须失败。这里0是“现在不存在”的条件，不是“下一个版本必为1”的承诺。哈希身份仍是可关联元数据，正文删除与完全匿名化要分开表述。

**第六站动手：** 新增一个kind='episodic'的记忆，value用字典保存task、action、outcome；再用kind过滤只取语义记忆。给另一个owner存相同key，确认互不覆盖。最后执行retract而非forget，观察当前recall为空但history还在，并解释两个操作的用途区别。

## 第七站：图路径是怎样一条边一条边找出来的

运行：

```bash
python -m labs.knowledge.walkthrough --stage 7
```

path应包含agent→retrieval与retrieval→index两条uses边，每条都有自己的quote；depth_one为null，表示深度1找不到两跳目标。JSON null对应Python None，不等于空列表。课程中起点等于终点返回空路径[]，找不到返回None，这个区分有助于调用者理解结果。

先看add。它检查实体字符串非空、predicate属于白名单、chunk当前可见、quote存在于原文。然后用这些字段计算edge_id，执行INSERT OR IGNORE。重复插入完全相同关系与证据不会产生两条边；换来源或quote会有不同edge_id，允许记录多来源断言。

请特别注意add没有真正理解quote含义。你可以给“Agent uses retrieval.”错误地配subject='moon'、object='cheese'，只要字段形式合法，结构检查仍可能通过。这不是隐藏功能，而是明确的语义审核边界。真实模型抽取不能只靠schema与substring就认为正确。

path首先取得visible_edges，再建立adjacency：同一个subject对应一个边列表。队列初始只有起点和空路径。每次popleft取最早入队项，所以搜索按跳数推进。expansions计数的是出队节点，不是返回边数；max_depth计数的是path长度。两个预算不要混淆。

处理agent时，看到uses→retrieval，形成一条边路径并入队；处理retrieval时，看到uses→index，形成两条边路径并立即返回。visited避免重复扩展已经到过的节点，但不禁止数据里存在环。对找一条最短跳数路径足够；对寻找所有可行证据链，这种状态压缩可能丢失不同路径，需要重新设计。

如果你把一条边反向保存，路径可能找不到。不要立即怀疑BFS，应先打印邻接表并检查subject/predicate/object的含义。方向错误常常来自抽取或ID映射，而不是遍历算法。

**第七站动手：** 加一条index uses agent形成环，确认查找不存在目标会在预算内结束；再加agent uses index直连边，确认BFS返回一跳路径；限制predicates为depends_on时应找不到uses路径。每个改动都对应图算法的一个具体假设。

## 第八站：删除怎样沿来源关系传播

运行：

```bash
python -m labs.knowledge.walkthrough --stage 8
```

新增relations文档后，五张主要表中documents/revisions/chunks/grants都是5，edges为1。soft delete后，这些数量不变，但visible_edges_after_soft_delete为0。purge后文档相关四表降回4，edges降到0。

软删除只修改documents.deleted标记。visible通过这个标记排除chunk，visible_edges又只保留可见chunk支持的边，因此路径立即不可用。注意不是数据库真的删除了边，而是读路径的可见性改变了。这种区别允许保留审计历史，同时保证在线查询不采用已撤销资料。

purge真正删除documents行，外键ON DELETE CASCADE依次清理revisions，再清理chunks和grants，edges因为引用chunk也被删除。级联删除关系必须在schema中建立，不能仅靠命名相似自动发生。用counts可以直观看到传播结果。

counts的五张表是内容表；document_revision_counters中的版本墓碑仍保留。否则旧引用在同doc_id、同内容重建时可能错误复活。这一保留范围与记忆计数器相同，必须在接口说明里公开，不能把五张表的0行解释为所有关联元数据都已消失。

如果你的实现缓存了adjacency，数据库删除后缓存仍可能返回旧边。课程每次重建可见边以保持简单，不提供大型图缓存。实际系统可以使用语料版本键、失效事件或读取时来源复核，但必须证明删除传播及时。不能只测试数据库表为空，还要测试面向用户的最终查询入口。

记忆和图也可能相互派生。例如某事件记忆由一个文档总结而来，如果记忆只存正文不存来源ID，就无法自动判断该文档撤销后哪些记忆过期。数据依赖必须在创建派生对象时保留，事后很难从自然语言猜回完整来源。

**第八站动手：** 先查询并缓存一条路径，删除来源后把缓存路径送给一个没有复核的显示函数，观察它仍能打印旧关系。然后为显示函数加当前来源检查，验证它拒绝。这个小实验说明“缓存提升性能”为什么也会改变正确性责任。

## 贯穿练习 A：读一次真实堆栈，而不是只看最后一句报错

在编写本卷时，一类典型错误是SQL占位符数与表列数不一致。chunks表有7列，却写了8个问号，会出现`sqlite3.OperationalError: table chunks has 7 columns but 8 values were supplied`。这个错误发生在store.ingest，但测试可能显示很多实验一起失败，因为所有实验都依赖同一个seed。

面对这种情况，先找第一条独立失败的完整堆栈。最上层告诉你测试或CLI入口，中间层说明调用链，最底层给出抛错位置和类型。不要因为有几十条失败就开始改几十个实验；找公共依赖中的第一个错误。修正一个SQL占位符后，所有依赖它的测试可能一起恢复。

具体排查顺序：读CREATE TABLE列定义，数列；读INSERT语句占位符，数问号；读传入tuple，数值；确认三者一一对应。更稳健的实现可以显式列名，例如`INSERT INTO chunks(chunk_id,tenant,doc_id,revision,start,end,text) VALUES(?,?,?,?,?,?,?)`，减少未来schema新增列对插入语句的影响。

不要用捕获OperationalError然后继续来“让测试过”。那会留下未摄入资料，而上层只看到空结果，错误从显式异常变成静默数据缺失。正确修复应恢复数据契约，再保留回滚和失败测试。

## 贯穿练习 B：自己写一条集成测试

下面是一条完整可运行测试函数，连接权限、查询、更新和引用。它不复制底层算法，而检查跨模块行为：

```python
from labs.knowledge.store import CorpusStore, Context
from labs.knowledge.retrieval import hybrid
from labs.knowledge.rag import Claim, Citation, validate_claim

def exercise_integration():
    ordinary = Context("demo", "learner")
    with CorpusStore() as store:
        store.ingest("demo", "guide", "Guide", "cache stores keys", ("*",))
        store.ingest("demo", "hidden", "Hidden", "private canary", ("group:admins",))
        assert hybrid(store, ordinary, "canary") == []
        hit = hybrid(store, ordinary, "cache")[0]
        claim = Claim(hit.chunk.text, Citation(hit.chunk.chunk_id, hit.chunk.text))
        assert validate_claim(store, ordinary, claim)[0]
        store.ingest("demo", "guide", "Guide", "new cache policy", ("*",), expected_revision=1)
        assert not validate_claim(store, ordinary, claim)[0]
        assert hybrid(store, ordinary, "new")[0].chunk.revision == 2

exercise_integration()
```

把它改写为unittest.TestCase并放在你自己的练习文件中。再增加以下情况：同一tenant不同owner记忆隔离、删除来源后路径失效、未知词拒答、写入失败回滚。不要依赖真实时间和网络，这样失败才更容易归因到你的代码。

## 贯穿练习 C：把读源码变成自己的实现能力

第一轮照着工作簿解释每个中间值；第二轮关掉参考文件，只保留接口和测试，重写tokenize、BM25或BFS其中一个；第三轮故意加入一个错误，用失败测试与日志定位；第四轮选择一个真实框架，对比它如何实现同一契约。这个循环比反复阅读同一段代码更接近实际开发。

你可以在练习分支中逐项实现，不需要一口气完成全部模块。每次提交只改变一个机制，例如“加入文档版本”“拒绝旧引用”“新增TTL边界测试”。提交说明写具体触发条件、原行为、新行为与验证方法，而不是“优化系统”“加强鲁棒性”。这样你半年后回看仍能理解自己的设计过程。

本工作簿最终验收不是背出函数名，而是以下能力：看到一个查询能预测候选与分数；看到一个错误能沿调用链定位；修改存储或检索组件后能保持接口契约；知道哪些结论由测试支持，哪些只是设计意图；能把真实模型接进来而不把身份、权限和数据生命周期交给模型自由决定。

## 进阶检查站：从函数行为扩展到系统不变量

你已经走完八站，现在把目光从“这个样例输出对了”移到“哪些条件无论输入怎样变化都必须成立”。不变量不是一个高深术语，它只是你承诺始终成立的关系。例如一个可见chunk必须属于当前tenant、未删除文档、当前revision，并至少有一条匹配grant。只要任何一项不成立，后续评分再精确也没有意义。

把不变量写成测试时，避免只测试固定正例。可以系统地改变一个条件：tenant相同/不同，deleted为0/1，revision当前/过期，grant匹配/不匹配。四个二元条件组合有16种，只有四项全部满足时应可见。这种小型组合测试比再添加十条随机正常文档更能暴露逻辑遗漏。

查询排名也有不变量：返回数量不超过k，ID不重复，分数有限，返回对象来自候选集合，排序遵守降分与稳定tie-break。任何实现优化，例如倒排索引、缓存或并行计算，都应继续满足它们。性能测试之前先通过这些语义测试，否则你可能只是在测一个更快的错误程序。

记忆系统的不变量包括每个tenant/owner/key最多一个active版本、revision按写入历史递增、过期记录不进入recall、forget后该key所有SQL版本为空。注意本实现没有数据库部分唯一索引直接强制“最多一个active”，而是通过单写事务中的更新与插入维持。若未来开放其他写入口或人工SQL修改，应加强数据库约束或限制写权限，不能假设所有人都会走同一个函数。

图的不变量是每条在线采用的边都有当前可见来源；路径的前一条object等于后一条subject；边数不超过max_depth；遍历预算可终止。对多来源边，删除其中一个来源后是否仍保留关系，需要你在数据模型里区分“同一事实有多个证据”和“多条独立断言”。当前实现每条边含一个chunk，因此行为简单且可解释。

### 如何写独立于实现的测试

假设你写了BM25，测试又复制完全相同的循环计算expected。两份代码可能共享同一个错误，例如都把df误算成词频。更强的测试来自手算数据、代数性质和边界：f=0贡献为0；b=0时文档长度不影响归一化；查询词重复在本契约下不改变结果；相同可见语料输入顺序变化不应改变分数，只可能改变你未固定的tie-break。

对余弦，可以测试同方向缩放不变、正交为0、维度交换一致、零向量按约定处理；对BFS，可以用有已知最短路径的小图而不是让另一个同样BFS函数给expected；对事务，可以故意在中间失败并查询提交状态，而不是只assert函数抛出了异常。异常发生了但半份数据已经提交，仍是严重失败。

对来源校验，应把“来源不存在”“来源被撤权”“来源过期”“引文不匹配”“陈述非抽取”分别测试。尽管有些对外返回同一原因，内部测试能确保每条条件存在。再测试一个全部满足的正例，防止你写出永远返回False的函数也通过所有拒绝测试。

### 当所有单元测试通过，集成仍可能失败

存储层可能正确过滤权限，检索层也能正确排序，但一个API层若把用户传来的group直接放进Context，就绕过了原本的身份信任边界。单元测试各自正确并不能证明组合后的认证正确。这就是根目录Agent+证据综合项目要测试context spoof的原因。

类似地，检索时引用有效，输出校验也正确，但如果调用链没有真的调用校验器，用户仍收到旧答案。集成测试应构造“工具完成后删除文档，再提交最终答案”的时间顺序，证明发布路径实际执行了复核。测试设计需要贴近真实事件顺序，而不是只覆盖函数名字。

又如MemoryStore的forget正确删除数据库记录，但对话线程的摘要里仍保存同一信息。数据库测试不会发现摘要残留，因为摘要属于另一个系统。应在数据流图上标出所有派生对象，再定义删除范围与验证入口。系统工程里最常见的问题之一，就是把一个局部正确性声明无意扩展成全局保证。

### 用一个小型变更训练代码审查

任务：给assistant增加“最多显示两个来源”的参数。一个草率实现只在最终print前做列表切片，检索、打包和模型仍处理大量材料。更完整的实现需要明确该参数是结果展示上限、检索候选上限还是模型上下文上限；三者不必相同。命名为display_limit、candidate_k、context_budget比含糊的limit更能帮助维护。

实现后测试0、1、2、大于候选数和非法负数。如果允许0，应返回无证据而不是悄悄默认3；如果不允许0，参数解析时明确拒绝。再检查来源数量限制是否破坏一个需要两段共同支持的答案。简短输出不是自动更正确，参数背后有业务语义。

审查另一个变更：给记忆加入importance分数。先问它的来源、范围、是否可校准、如何影响召回、能否被模型自我强化、是否需要衰减。若只在表中增加一列并按大数排序，并没有解决这些问题。代码结构改动很小，不代表产品语义简单。

### 最后一次闭卷验收

在全新分支上，只看接口定义，写一个自己的最小系统：三篇公开文档、一篇受限文档；BM25搜索；带原文引用的回答；一次更新使旧引用失效；一条记忆被纠正；一条依赖边随来源删除失效。你可以用本卷测试作为验收，但不要直接复制实现。

完成后解释三个真实失败：一个输入契约错误、一个状态生命周期错误、一个数值或排序错误。对每个错误给出触发输入、失败现象、根因、修复与防回归测试。能把这些讲清楚，说明你已从“知道RAG有哪些组件”进到“能构建并诊断知识系统”的阶段。
