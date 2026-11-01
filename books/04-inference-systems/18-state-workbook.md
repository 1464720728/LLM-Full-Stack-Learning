# S18 状态工作坊：亲手跟踪prefix、COW与抢占

本工作坊先修 S03—S05 的分页、共享和抢占，以及 F07 的引用与 F09 的断言。你还需要区分 prompt、output 和当前已知 context；若这些长度的关系不清楚，先完成 S17 的三轮账本。这里把这些概念展开成变量级执行。你要在每次操作前预测页表、数据、引用数、computed、output和free-list，再运行代码核对。预测错了不是失败，而是暴露了具体的理解缺口；如果只运行然后看“结果正常”，你会错过最重要的学习过程。

## 1. 工具与记录规则

执行以下命令，每条都是CPU离线实验：

```bash
python -m labs.serving.workbook_trace --case prefix
python -m labs.serving.workbook_trace --case cow
python -m labs.serving.workbook_trace --case preempt
```

输出是每个边界的完整JSON快照。页payload是token ID元数据，不是真实K/V浮点张量；time_ms是模拟代价。请求的context是prompt+output；computed说明本参考引擎已经写入页表的上下文前缀长度。finished请求会保留历史计数用于报告，但它已没有活动页表，所以不要拿终态computed推断还占显存。

记录时采用以下约定：P0表示物理页0；`A:[0,1]`表示A的逻辑页0/1映射到物理0/1；`refs(P0)=2`表示有两个持有引用，不表示两个token；`free=[2,3]`是空闲物理页ID，而不是它们的字节数。若一个术语不标单位，先补单位再计算。

### 1.1 先把每一种状态放在正确的对象里

请求对象保存用户层任务，包括原始提示、已经生成的输出、采样 RNG、状态和时间。页表保存逻辑页到物理页的映射；物理页保存已计算位置的教学 payload 与引用计数；前缀缓存保存可复用身份到页 ID 的引用。四者联系紧密，却不能合并成一个“缓存字典”。否则请求结束后仍存活的前缀、抢占后仍保留的输出都会难以解释。

一个最小例子是 prompt 长五、已经输出一个 token、computed 为五。这时 context 长六，页中只有前五个位置；输出 token 已知，但还没有作为下一步输入写入本教学页池。下一轮写入第六位置再生成下一个 token。最后一个输出一旦满足终止条件，就不必再为它计算下一步状态，所以正常结束时 computed 可以比 context 少一。

先构造对象，观察 submit 与真正安排计算之间的差别。后面的 Python 块按章顺序共享变量；可以把所有块依次保存到一个练习文件。每个独立案例会创建自己的池或引擎，避免上一题残留状态影响下一题。

```python
from collections import Counter
from copy import deepcopy
from labs.serving.engine import Engine, Request
from labs.serving.kv import PagePool, PrefixCache, OutOfPages
from labs.serving.workbook_trace import prefix_trace, cow_trace, preempt_trace

intro = Engine(pages=8, block_size=2, token_budget=4, chunk_size=4)
intro.submit(Request("intro", (1, 2, 3, 4, 5), 2))
assert intro.requests["intro"].state == "pending"
assert intro.pool.tables == {}
assert len(intro.pool.free) == 8
intro.step()
assert intro.requests["intro"].computed == 4
assert intro.pool.tokens("intro") == [1, 2, 3, 4]
assert intro.requests["intro"].output == []
```

注意最后两行检查了两个不同表示：页 payload 和请求输出。若只看 output 为空，无法知道是否已经完成四个输入位置；若只看页中四个值，无法知道是否错误地提前发过输出。状态审计的力量来自交叉验证独立表示，而不是重复打印同一个字段。

### 1.2 完整快照为什么要包含空闲页

现有 workbook_trace 为便于阅读，只列仍有引用的页面。这个展示很清楚，但如果要证明“空闲页没有残留 payload”，就必须读取全部物理页；只看活跃列表没有足够证据。下面写一个完整快照，复制每一层列表，避免之后修改原池时快照也跟着改变。

```python
def full_pool_snapshot(pool, cache=None):
    return {
        "block_size": pool.block_size,
        "tables": {rid: list(bids) for rid, bids in pool.tables.items()},
        "pages": [{"id": bid, "tokens": list(page.data), "refs": page.refs}
                  for bid, page in enumerate(pool.pages)],
        "free": sorted(pool.free),
        "cache": [] if cache is None else [
            {"namespace": key[0], "prefix": list(key[1]), "pages": list(bids)}
            for key, bids in cache.entries.items()],
    }

frozen_intro = full_pool_snapshot(intro.pool, intro.cache)
intro.step()
assert frozen_intro["tables"] == {"intro": [0, 1]}
assert frozen_intro["pages"][2]["tokens"] == []
assert intro.pool.pages[2].data == [5]
```

复制快照不是给系统增加第二份真相，而是记录某个边界的证据。生产引擎持续变化，快照应当稳定。反过来，快照过期后不能拿它决定当前页是否可写；诊断数据与运行时所有权必须分清。我们只在同步操作边界拍照，真实并发系统还需要一致性快照或事件序列，不能任意读几个字段就假设它们属于同一时刻。

### 1.3 从拥有者重建引用，而不是相信 refs 自己

引用计数的含义是有多少个拥有者持有这张物理页。一个活动请求的表出现一次，贡献一；一个缓存条目出现一次，也贡献一。它与页中 token 数量无关，也与引用来自几个 Python 变量无关。审计器要从表和缓存条目重新计数，再比较页面记录的 refs，才能发现多 retain 或少 unpin。

下面的完整审计器检查页 ID、页表唯一性、满页前缀、引用守恒、free 集合和空闲 payload。它不调用 PagePool.check，也不复用它的内部计数结果；使用输出快照作为输入，便于故意修改一份快照而不破坏正在运行的引擎。

```python
def audit_pool(snapshot):
    pages = snapshot["pages"]
    block = snapshot["block_size"]
    if [p["id"] for p in pages] != list(range(len(pages))):
        raise AssertionError("physical page IDs are incomplete")
    free = snapshot["free"]
    if len(free) != len(set(free)) or any(type(i) is not int or not 0 <= i < len(pages) for i in free):
        raise AssertionError("invalid free list")
    owners = Counter()
    for owner, ids in list(snapshot["tables"].items()) + [
        (f"cache:{i}", entry["pages"]) for i, entry in enumerate(snapshot["cache"])
    ]:
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate page in {owner}")
        if any(type(i) is not int or not 0 <= i < len(pages) for i in ids):
            raise AssertionError(f"invalid page in {owner}")
        owners.update(ids)
    for rid, ids in snapshot["tables"].items():
        if any(len(pages[i]["tokens"]) != block for i in ids[:-1]):
            raise AssertionError(f"non-full interior page: {rid}")
    for bid, page in enumerate(pages):
        if page["refs"] != owners[bid]:
            raise AssertionError(f"reference mismatch page {bid}: {page['refs']} != {owners[bid]}")
        if (bid in free) != (owners[bid] == 0):
            raise AssertionError(f"free ownership mismatch page {bid}")
        if len(page["tokens"]) > block or (bid in free and page["tokens"]):
            raise AssertionError(f"payload mismatch page {bid}")
    for entry in snapshot["cache"]:
        if any(len(pages[bid]["tokens"]) != block for bid in entry["pages"]):
            raise AssertionError("cache contains a non-full page")
        data = [token for bid in entry["pages"] for token in pages[bid]["tokens"]]
        if data != entry["prefix"] or len(data) % block:
            raise AssertionError("cache identity differs from payload")
    return True

assert audit_pool(frozen_intro)
assert audit_pool(full_pool_snapshot(intro.pool, intro.cache))

two_half_pages = {
    "block_size": 2, "tables": {}, "free": [],
    "pages": [{"id": 0, "tokens": [1], "refs": 1},
              {"id": 1, "tokens": [2], "refs": 1}],
    "cache": [{"namespace": "m", "prefix": [1, 2], "pages": [0, 1]}],
}
try:
    audit_pool(two_half_pages)
except AssertionError as exc:
    assert str(exc) == "cache contains a non-full page"
else:
    raise AssertionError("two partial pages were accepted as a full-page prefix")
```

这个审计器还比较缓存键中的 prefix 与对应 payload。因为本章 payload 恰好使用 token ID，这样可以直接验证身份；换成真实浮点 K/V 后，两者不是同一种数据，不能继续做列表相等。你需要保存 token 身份与数值状态之间的关联，并用模型前向或独立数值 oracle 验证，这正是 S19 的职责。

整页合同必须逐页核验。两张容量为二、却各只写一个 token 的页，合计长度也是二，拼接内容也可能恰好等于 prefix；只检查总长度能被 block 整除会漏掉这个坏状态。上面的反例保持引用计数与 free 列表完全一致，只破坏页内填充规则，因此能单独检验审计器是否覆盖了这一条。独立核验程序也可能漏检，不能因为它叫 audit 就免于反例检查。

审计成功说明快照满足列出的合同，不能证明所有实现行为正确。例如它还没有检查请求状态与表集合一致，也没有检查一次生成只发一次事件；后面的事件审计会补这两层。不要写一个叫 check 的函数，就认为所有可能错误都已被覆盖。

## 2. Prefix案例的初始条件

共有8物理页，每页2个token槽；预算4、prefill chunk4；缓存最多16条。A的prompt是`[1,2,3,4,5]`，要输出2个token。Bigram模型使输出可预测为6、7。初始requests里A处于pending，computed0、output空；没有页表，8页全free。

submit只登记请求。为什么还没分配页？因为未来到达的请求可能排队很久，提前为所有提交请求分配会放大资源占用。真实系统也可能在前端做预处理，但那不等于已经获得GPU KV。

第一轮step先ingest、再admit，为A创建空表；前缀缓存为空，所以computed仍0。count=min(gap5,budget4,chunk4)=4。`needed`算追加4位置需2页，free足够；分配P0/P1，各refs1，数据分别`[1,2]`、`[3,4]`。A表为`[0,1]`，computed4。

此时没有输出，因为computed4小于context长度5。也没有发布前缀：参考Engine选择提示至少完成后再publish，尽管前4个位置已经算过。生产实现可以采用更细粒度发布，但那需要另一套并发可见性契约。

## 3. 第二轮：发布的同时拥有者增加

A还差prompt最后一个token5，本轮count1。现有两页都满，需要P2，写入`[5]`，A表变`[0,1,2]`，computed5。现在已经完成prompt，cache.publish按整页对齐保留前4token，也就是P0/P1。

publish调用retain，使P0/P1的refs从1变2。两个拥有者分别是A活动表与缓存条目。P2只有A持有，refs1。此时free剩P3到P7，共5页。不要把cache条目也当一份完整复制：缓存只是增加对已有物理页的引用，不是再复制两页数据。

模型看到完整context`[1,2,3,4,5]`，预测6；output变`[6]`，context逻辑上变六个位置，但computed仍5。P2仍只有`[5]`，没有6。这个瞬间最能检验你是否分清“生成token”和“计算该token的KV”。

模拟时钟第一轮为2ms，第二轮增加1.25ms，到3.25ms出现首token。输出中的精确值来自本模型公式，若拿它和真实H100对比，应首先承认它不是硬件计时。

## 4. 第三轮：请求结束不等于缓存消失

处理已生成的6作为新输入。P2未满且refs1，可以原地追加，得到`[5,6]`，无需新页。computed变6。publish发现相同prefix键已存在，仅更新LRU访问顺序，不重复增加引用。若每轮重复retain而不检查已有键，会产生永久泄漏。

模型预测7，output变`[6,7]`，达到输出上限，A进入finished。`_finish`移除A活动表并释放它对P0/P1/P2的引用。P0/P1从2降1，因缓存仍持有而留下；P2从1降0，数据清空并进入free。

此时表为空，缓存有一条，物理使用2页、空闲6页。A的context长度7、computed6、output长度2仍保留在Request报告对象中，但这些是历史元数据，不能当活动页引用。你必须分别检查requests历史与pool.tables。

验证不变量：所有活动表引用数为0，缓存引用总数2，所以页面总refs和为2；空闲页与非空闲页共8；P0/P1的数据未变。若free不是6，先检查重复retain或未释放P2，而不要先清空整个pool掩盖错误。

## 5. B到来：复用前四位置，再计算不同suffix

B的prompt为`[1,2,3,4,6]`，输出2。它在A完成时到达。`attach`先求最大可命中长度：prompt5，至少重算最后位置，block2，所以limit4。缓存前缀恰好前4一致，B采用P0/P1并retain，refs再次从1到2，computed直接设4。

B最后一个prompt token6不同于A的5，所以不能共享P2；而P2早已free。B本轮分配最小空闲页P2，写`[6]`，computed5，模型预测7，output`[7]`。这证明缓存复用不要求整个提示完全相同，只要求被复用那一段的计算语义相同。

下一轮将7写入P2，得到`[6,7]`，预测8后结束。B释放活动引用，P0/P1仍由缓存持有，P2再free。最后cache.clear解除最后两份引用，8页全部free。

整个过程中，物理P2先承载A的`[5,6]`，后来承载B的`[6,7]`。页ID本身不代表永久数据身份。正确性来自引用生命周期和写入完成，而不是“这个页以前属于谁”。如果旧请求保存了一个已释放物理地址并继续读，就会读到新请求数据，这正是use-after-free的危险。

### 5.1 把文字轨迹逐步变成可执行断言

现在独立重做 A/B 案例，不直接信任日志摘要。每一步先写预期页 ID，再调用 step，最后检查 refs 和 payload。物理页由本课程的小根堆分配器优先选择最小空闲 ID，因此这里的具体页号可以确定；换一个 allocator 时，数值内容与拥有关系仍应成立，但页号可能不同。

```python
prefix_engine = Engine(token_budget=4, chunk_size=4, pages=8, block_size=2)
prefix_engine.submit(Request("A", (1, 2, 3, 4, 5), 2))
prefix_engine.step()
assert prefix_engine.pool.tables["A"] == [0, 1]
assert [p.refs for p in prefix_engine.pool.pages[:3]] == [1, 1, 0]
prefix_engine.step()
assert prefix_engine.requests["A"].output == [6]
assert [p.refs for p in prefix_engine.pool.pages[:3]] == [2, 2, 1]
assert prefix_engine.pool.pages[2].data == [5]
assert audit_pool(full_pool_snapshot(prefix_engine.pool, prefix_engine.cache))
prefix_engine.step()
assert prefix_engine.requests["A"].state == "finished"
assert prefix_engine.requests["A"].output == [6, 7]
assert prefix_engine.pool.tables == {}
assert [p.refs for p in prefix_engine.pool.pages[:3]] == [1, 1, 0]
assert len(prefix_engine.pool.free) == 6
prefix_engine.submit(Request("B", (1, 2, 3, 4, 6), 2,
                             arrival_ms=prefix_engine.time_ms))
prefix_engine.step()
assert prefix_engine.requests["B"].cache_hit_tokens == 4
assert prefix_engine.pool.pages[2].data == [6]
assert prefix_engine.requests["B"].output == [7]
prefix_engine.step()
assert prefix_engine.requests["B"].output == [7, 8]
prefix_engine.cache.clear()
assert len(prefix_engine.pool.free) == 8
assert audit_pool(full_pool_snapshot(prefix_engine.pool, prefix_engine.cache))
```

观察这段程序的三个时间边界。A 第二步后已经发出六，但 P2 中还是五；A 第三步后 P2 已释放；B 第一步重新获得 P2 时，其中只应该有六，不能残留 A 的旧尾部。最后两者输出报告都存在，活动页表却为空。这证明“请求历史仍保留”和“资源仍占用”不是同一个判断。

本例共享完整前四位置，只因为 namespace 和 token 前缀都一致。本课程缓存用完整元组作键，避免哈希碰撞问题；生产系统可能用链式 block hash、adapter 身份、模型版本或多模态标识构造键。无论表示如何变化，复用必须意味着计算语义一致，不能拿文本语义相似或 embedding 接近替代精确状态身份。

### 5.2 缓存条目数、拥有引用数、物理页数分别计算

假设同一请求分别发布长度二与长度四的前缀。两个条目都持有 P0，较长条目还持有 P1。此时缓存条目数是二，缓存拥有引用数是三，缓存涉及不同物理页数是二。如果监控把这三项混成一个 cache size，删除一个条目后就可能出现“数量下降但显存没下降”的误解。

```python
multi_pool = PagePool(4, 2)
multi_cache = PrefixCache(multi_pool)
multi_pool.create("owner")
multi_pool.append("owner", [1, 2, 3, 4])
multi_cache.publish("owner", "m", 2)
multi_cache.publish("owner", "m", 4)
assert len(multi_cache.entries) == 2
assert sum(multi_cache.refs().values()) == 3
assert len(multi_cache.refs()) == 2
assert [p.refs for p in multi_pool.pages[:2]] == [3, 2]
multi_pool.release("owner")
assert [p.refs for p in multi_pool.pages[:2]] == [2, 1]
free_before = len(multi_pool.free)
assert multi_cache.evict_one()
assert len(multi_pool.free) == free_before
assert [p.refs for p in multi_pool.pages[:2]] == [1, 1]
multi_cache.clear()
assert len(multi_pool.free) == 4
assert audit_pool(full_pool_snapshot(multi_pool, multi_cache))
```

第一次淘汰较短条目，没有释放任何物理页，因为较长条目仍持有同一 P0。这是正确行为，不是泄漏。泄漏要依据所有合法拥有者已经消失而 refs 仍非零来判断。反过来，条目存在却 refs 已归零，则是悬空引用，下一次 attach 可能读到被其他请求覆盖的数据。

同样，重复 publish 已有相同键只更新 LRU，不应再增加引用。可以连续发布十次后比较 refs 不变；若它每次都 retain，最终 clear 只 unpin 一次，就留下九份找不到拥有者的计数。我们的独立 audit_pool 会立刻把 refs 与表、条目重建结果对不上，而不是等运行很久后才从容量下降猜泄漏。

## 6. Prefix案例的独立修改任务

把B改成`[1,2,9,4,6]`。共同前缀只有2个token，因此computed起点2，复用只应P0。即使缓存条目包含P0/P1，也不能把P1整页一起adopt。`attach`会按匹配长度切bids列表，再retain对应页。

把B改成完全相同的四token提示`[1,2,3,4]`。虽然缓存有四位置，本契约必须留下最后位置以获得logits，且只允许整页命中，因此只命中2，不是4，也不是3。请观察重新计算多少位置，并解释其与block_size的关系。

把namespace改成另一个adapter身份。即使token完全一样，命中应0。当前Bigram没有真正adapter数值，所以输出仍可能相同；验收应检查命中隔离，而不是只比较输出。这个反例告诉你，测试要直接覆盖契约，不要依赖某个简化后端恰好表现出差异。

### 6.1 完整的命中变体程序

为每个候选创建新的空表，因为 attach 的合同是采用一份前缀到空表，并不是向已有表追加。测试完释放该候选，再做下一项，可以隔离引用变化。下面同时验证长提示、短提示、完全相同四 token 提示和 namespace 隔离。

```python
variant_pool = PagePool(8, 2)
variant_cache = PrefixCache(variant_pool)
variant_pool.create("source")
variant_pool.append("source", [1, 2, 3, 4])
variant_cache.publish("source", "model-a", 4)
variant_pool.release("source")
variants = [
    ("same-prefix", "model-a", (1, 2, 3, 4, 6), 4),
    ("short-common", "model-a", (1, 2, 9, 4, 6), 2),
    ("same-four", "model-a", (1, 2, 3, 4), 2),
    ("one-token", "model-a", (1,), 0),
    ("other-model", "model-b", (1, 2, 3, 4, 6), 0),
]
for rid, namespace, prompt, expected in variants:
    variant_pool.create(rid)
    hit = variant_cache.attach(rid, namespace, prompt)
    assert hit == expected, (rid, hit, expected)
    assert variant_pool.tokens(rid) == list(prompt[:expected])
    assert audit_pool(full_pool_snapshot(variant_pool, variant_cache))
    variant_pool.release(rid)
variant_cache.clear()
assert len(variant_pool.free) == 8
```

这里最后一个 token 重算是本实现获得下一 token logits 的约定，不是说那个位置的 K/V 数学上不能缓存。如果另外保存了足够的末端隐藏状态或 logits，并有相应一致性协议，系统可以设计不同策略；不能把本教材约定当作所有框架的物理必然。读真实 manager 时要确认它到底保存哪些状态，而不是只看到“最后 token”几个字就套用公式。

块大小也会放大重算。长度四、块大小二时，要保留最后位置，整页可命中上限是二；因此重算两个位置，而不是只重算一个。这是完整页粒度与末端 logits 需求共同造成的结果。若希望减少浪费，可以研究部分页缓存，但必须重新处理可见性、COW 和不同有效长度，复杂度会增加。

## 7. COW案例：同一尾页的两条未来

独立案例用4页、block4。A写`[1,2,3]`，占P0、refs1。fork B时，新建B表并复制页ID列表，不复制data；两表都指P0，refs2。此时A/B看到的token相同，但它们的未来可以分叉。

B要追加9。`needed`先计算：当前总长度3，追加后长度4，所需总页仍1；新增尾页数是0。但尾页未满且refs2，必须COW，所以总需要1页。若只使用`ceil((3+1)/4)-1=0`，会漏掉复制需求，导致共享数据污染。

reserve取P1并设refs1。复制P0.data到P1；把B表末项换成1；unpin旧P0，把refs2降1；最后向P1追加9。结果A表`[0]`、P0=`[1,2,3]`；B表`[1]`、P1=`[1,2,3,9]`。不同页里的前三个值相同，但物理所有权已分离。

随后A追加4。P0未满但refs1，无需复制，直接得到`[1,2,3,4]`。A/B的第四个值各自独立。分别release后P0/P1都归free，池恢复初态。

请特别注意操作顺序：如果先unpin P0再复制，而A已经不持有它，P0可能归零被清空，复制就丢了数据。当前案例A仍持有，所以错误可能不立刻显现；健壮实现应按正确顺序设计，而不是依赖某个测试恰好还有另一拥有者。

## 8. 失败原子性的四步证明

把COW案例池缩成只有1页。A/B共享P0，B追加9需要1新页，但free为0。`append`在修改前抛OutOfPages。你要证明四件事：A/B页表仍指P0；P0.data仍`[1,2,3]`；refs仍2；free仍空。

这比“最终输出没变”强，因为函数可能破坏计数但短期还读得到数据。测试应比较结构快照或逐项断言。不能在异常后偷偷重建整个池，因为那会丢失其他请求合法状态。

再构造一个更难边界：block4、共享尾页长度3，B追加6个token。追加后长度9，需要三逻辑页，当前一页，所以新增2；加尾页COW1，总共3物理新页。只剩2页时必须完全失败；有3页时才能成功。用这个例子检验`needed`公式不是只在追加1个token时正确。

若你把真实KV存储换成GPU张量，reserve成功后copy本身还可能失败。此时事务边界更复杂：需要回收已分配页并恢复映射，或把请求标为失败并释放整条状态。基础实现只保证资源预检失败原子性，不声称所有设备错误都可无损回滚。

### 8.1 对照 append 的每一步，而不是只记“复制一份”

打开 `labs/serving/kv.py`，从 `append` 的第一行看起。先验证所有输入 token 都是非 bool 的非负整数；接着调用 needed 计算一次追加必须预留的全部新页；若空闲页不够，在修改任何页之前抛 OutOfPages；成功才从 free heap 取出全部 reserved 页并设置其引用。这个顺序把可预测容量失败放到提交之前。

随后处理共享未满尾页：取得旧页 ID，取一个预留新页，复制旧有效 payload，把当前请求的尾映射换成新页，再释放当前请求对旧页的那一份引用。只减一份，因为其他请求或缓存仍拥有旧页。最后逐 token 填入尾页，满了才从剩余预留页中拿下一页。函数末尾检查预留迭代器已经耗尽，验证 needed 算术与实际使用一致。

为什么先修改表再 unpin 旧页？从拥有关系看，当前请求正在把自己的那份所有权从旧页移到新页。若旧页是最后引用，unpin 会清空它，所以复制必须先完成。当前正常 COW 条件是旧页 refs 大于一，但按正确顺序书写仍然重要：以后增加其他引用操作或改变并发边界时，不能依赖某个偶然剩余拥有者保护错误代码。

下面实际验证一个需要三张新页的追加。起始长度三、块大小四、共享尾页，追加六项后长度九；两张新逻辑页加一张 COW 页，总共三张。池只有三张总页时，旧页已占一张，空闲两张，因此应该整体失败；池有四张总页时才足够。

```python
def make_shared_pool(total_pages, block, initial, owner_count):
    pool = PagePool(total_pages, block)
    pool.create("A")
    pool.append("A", list(initial))
    for index in range(1, owner_count):
        pool.fork("A", f"peer-{index}")
    return pool

failed_pool = make_shared_pool(3, 4, [1, 2, 3], 2)
assert failed_pool.needed("A", 6) == 3
saved = full_pool_snapshot(failed_pool)
try:
    failed_pool.append("A", [4, 5, 6, 7, 8, 9])
except OutOfPages:
    pass
else:
    raise AssertionError("capacity failure was not reported")
assert full_pool_snapshot(failed_pool) == saved
assert audit_pool(saved)

success_pool = make_shared_pool(4, 4, [1, 2, 3], 2)
success_pool.append("A", [4, 5, 6, 7, 8, 9])
assert success_pool.tokens("A") == list(range(1, 10))
assert success_pool.tokens("peer-1") == [1, 2, 3]
assert len(success_pool.free) == 0
assert audit_pool(full_pool_snapshot(success_pool))
```

失败快照同时包含页表、所有页 payload、refs 和 free 集合，所以“没有改变”有明确范围。我们没有声称 Python 内存分配失败、进程被杀死或真实设备复制出错也能原样回滚。教学 append 的强保证针对输入验证与预检容量不足；更广泛事务需要额外日志、资源回滚或失败终止策略。

### 8.2 用独立公式枚举边界，不只测追加一个 token

令原有效长度为 $L$、追加数量为 $a$、块大小为 $b$。正常新增逻辑页是 $\lceil(L+a)/b\rceil-\lceil L/b\rceil$。若追加非空、原尾页未满且有多个拥有者，再加一张 COW 页。空追加不写任何内容，即使尾页共享，也不应该为了什么都不做而复制。

下面独立写出公式，不调用 needed 内部实现。枚举块大小一、二、四，初始长度零到两块，拥有者一到三，追加零到两块。每种组合比较预估、实际新增物理页和其他拥有者内容；需要新页时，再构造少一页的池验证整体失败。这个测试覆盖满页、空表、共享尾与跨多页追加的交叉关系。

```python
def independent_needed(length, append_count, block, owners):
    if append_count == 0:
        return 0
    before = (length + block - 1) // block
    after = (length + append_count + block - 1) // block
    cow = int(length % block != 0 and owners > 1)
    return after - before + cow

boundary_cases = 0
for block in (1, 2, 4):
    for length in range(2 * block + 1):
        initial = list(range(length))
        for owner_count in (1, 2, 3):
            for count in range(2 * block + 1):
                expected = independent_needed(length, count, block, owner_count)
                pool = make_shared_pool(40, block, initial, owner_count)
                before_free = len(pool.free)
                assert pool.needed("A", count) == expected
                added = list(range(100, 100 + count))
                pool.append("A", added)
                assert before_free - len(pool.free) == expected
                assert pool.tokens("A") == initial + added
                for index in range(1, owner_count):
                    assert pool.tokens(f"peer-{index}") == initial
                assert audit_pool(full_pool_snapshot(pool))
                if expected:
                    used = (length + block - 1) // block
                    total = used + expected - 1
                    if total >= 1:
                        tight = make_shared_pool(total, block, initial, owner_count)
                        old = full_pool_snapshot(tight)
                        try:
                            tight.append("A", added)
                        except OutOfPages:
                            assert full_pool_snapshot(tight) == old
                        else:
                            raise AssertionError("one-page-short append succeeded")
                boundary_cases += 1
assert boundary_cases == 345
```

这个枚举的规模很小，普通 CPU 很快就能完成，却比盲目跑很大的随机输入更容易定位。它也保留一个明确例外：当构造“少一页”会得到总页数零时，PagePool 本身不允许零容量，因此不构造那个非法池，而不是为了测试篡改构造合同。

不要把测试中的页数四十理解为推荐运行配置，它只是保证成功组有充足余量。用于学习的输入应该让你知道每一项为什么存在：成功组验证计算公式，失败组验证预检原子性，peer 内容验证 COW 隔离，audit 验证拥有关系。每个断言负责不同错误，删掉某一项就会失去对应覆盖。

### 8.3 边界验证也需要先验证全部输入

`retain([0, 99])` 如果先增加页零引用，再发现九十九越界，就会留下没有拥有者的额外 refs。`unpin([0, 1, 1])` 如果页一只持有一次，不能先释放页零再报错。当前实现先把 ID 转为计数，验证全部操作合法，再统一修改。这个设计把重复 ID 需要释放几份也表达清楚。

```python
atomic_pool = PagePool(3, 2)
atomic_pool.create("A")
atomic_pool.append("A", [1, 2])
atomic_pool.create("B")
atomic_pool.append("B", [3, 4])
old = full_pool_snapshot(atomic_pool)
for operation in (lambda: atomic_pool.retain([0, 99]),
                  lambda: atomic_pool.unpin([0, 1, 1])):
    try:
        operation()
    except (ValueError, AssertionError):
        assert full_pool_snapshot(atomic_pool) == old
    else:
        raise AssertionError("invalid reference operation accepted")
```

这些操作是页池内部合同，实际生产多线程环境还必须处理并发修改和同步。当前课程 PagePool 是单线程同步参考，不能因为有引用计数就拿它在多个线程上无锁共享。引用计数解决拥有数量，线程安全解决并发访问的顺序，两者属于不同层次。

## 9. 抢占案例的第一、二轮

现在池只有4页、block2、budget4、chunk2、活动上限2、prefix缓存关闭。A提示`[1,2,3,4]`，B提示`[5,6,7,8]`，各输出4。单请求最多缓存7位置，4页能容纳；两请求同时增长则不够。

第一轮A/B各处理2位置，占P0和P1，computed都2，没有输出。第二轮round-robin轮转候选顺序，B先获得P2完成prompt，A后获得P3；因此A表`[0,3]`，B表`[1,2]`，而不是按字面顺序想象A必然拿P2。物理页分配次序是调度结果的一部分，但不应改变输出语义。

第二轮结束，两者computed4，各刚生成第一输出：A output`[5]`，B output`[9]`。四页全满，每页refs1。下一步两请求都要把刚生成token写到新页，然而free0。因此一轮候选遍历可能全部allocation_blocked。

这时系统不能永远等待：没有其他会结束的请求，因为结束也需要继续生成；必须抢占或采取其他资源策略。我们的引擎选择列表尾部B作为victim，而不是声称这是最优策略。

## 10. 第三次调用内部的重试为何迭代号跳跃

一次外部`step()`调用先尝试形成计划，发现没有工作；它释放B的P1/P2，将B置waiting、computed0，保留output`[9]`与RNG。随后临时降低活动上限，在同一次调用中递归重试，使A先推进。

因此workbook_trace在外部调用边界可能从iteration2直接看到iteration4：iteration3是无计算计划与抢占事件，iteration4是重试产生的实际工作。JSON的events仍保留allocation_blocked与preempted。不要把序号跳过误判为漏掉GPU工作；先看这个序号在代码里代表“调度尝试”还是“实际模型执行”。

A在重试中把5写入P1，表`[0,3,1]`，computed5，生成6，output`[5,6]`。P2空闲，B等待。此时A的已生成6还未计算，所以P1只有`[5]`。

这个案例揭示一个真实工程问题：指标名必须准确。若把iteration数直接当GPU forward次数，空计划和重试会污染统计。生产框架可能有自己的step、batch、forward、graph replay计数，你应检查其增量位置。

## 11. B恢复后为什么不立刻重复输出9

下一外部step允许B重新admit。B从computed0重算prompt，但output仍有9。context长度5。它先算前2位置，computed2；A同时处理6，computed6、输出7。现在A占3页，B占1页，池又满。

再下一轮A需要新页写7才能生成最后输出8，B也需要新页继续prefill，所有分配阻塞。B再次被抢占，释放其唯一页，A取得新页、computed7，生成8后完成，释放全部活动页。此时B仍只输出过一次9，虽然它的prompt已经被重算过一部分两次。

B随后独占池：先算前2，接着算到4。注意computed4等于prompt长度，但context长度5，因为以前已经输出过9。所以此刻**不能再采样**。只有下一轮把9作为输入算完，computed5等于context5，才预测新token10。

这个判断是整个抢占恢复正确性的核心。若你把条件写成computed>=prompt长度，B在每次重算完提示时都会再发9，造成重复输出与错误RNG消费。某些测试只检查最终列表长度，可能仍通过；必须检查具体输出序列与每个位置只发一次。

## 12. 用守恒关系解释整个抢占过程

请求已知token数量会随着新输出增长，computed可因抢占降低；输出列表不应因重算缩短或重复；物理页占用可上可下；可用页+被引用页始终等于总页数。每种量的单调性不同，不能用同一个“应该一直增加”检查所有字段。

最终A输出`[5,6,7,8]`，B输出`[9,10,11,12]`，都finished，缓存关闭，4页全free。B发生两次抢占，这是本策略在此工作负载下的真实行为。你可以设计更好的准入策略减少反复重算，但先保持输出和所有权不变量。

一个可能改进是为最老活动请求保留足够下一页，或在恢复B前评估其minimum working set；另一种是降低活动上限直到A结束。它们可能牺牲并发，但减少重算。不能只优化“抢占次数少”，还应看TTFT、总完成时间与公平性。

### 12.1 为什么这个参考引擎最终能够前进

“内存不足就等一会”不是完整调度算法。在本例里，所有活动请求都需要下一张页才能继续，没有任何独立工作会自动释放页；如果只反复重试同一集合，状态永远不变。要恢复前进，必须改变资源关系，例如淘汰可释放缓存、抢占某请求、降低并发或选择其他有空间的工作。等待只有在确实存在外部完成事件时才有理由。

本 Engine 先尝试淘汰缓存，再检查活动请求能否组成非空计划。如果计划仍空，且活动请求至少两个，就释放队尾请求的活动页，让它回到等待。递归重试临时把活动上限缩到当前 running 数量，因此刚释放的请求不会立刻又抢回同一资源。每次这种内部重试都严格减少活动表数量，所以最多经过有限次抢占就会留下一个活动请求。

剩下一个请求时为什么不必继续无限抢占？submit 已经拒绝单请求最大所需上下文超过整个池容量的输入。对于生成上限 $g>0$，最多需要缓存提示长度加 $g-1$ 个输出位置，因为生成最后一个输出后就结束，不再计算它的下一步状态。当缓存条目逐步淘汰、其他活动表释放后，这个请求的工作集合能够装入整个池，因而可以推进。这是代码中的准入检查与抢占策略共同提供的保证，不是单看某个 while 循环就能证明的性质。

还要区分累计事件计数与唯一内容数量。当前 Request.cache_hit_tokens 在每次 admit 时累加本次命中的位置数；如果同一请求被抢占后多次恢复，它可能反复命中同一段前缀。累计命中八个位置，不意味着原提示里有八个不同位置被永久复用，也不能直接除以五个提示 token 就报告百分之一百六十的“缓存命中率”。应明确分母是各次查询可复用工作、总安排工作，还是唯一输入位置，再设计相应指标。

同理，scheduled_work 把重算也计入，所以它可以超过最终请求上下文的总长度。这正是比较抢占成本所需要的信息，不是计数错误。最终输出数量、唯一输入位置、实际安排计算位置和累计缓存命中各自回答不同问题；报告应保留这些名称，不把它们都缩写成 tokens。给每个计数写出增加位置和单位，是理解调度器性能日志的必要练习。

这个推理有明确前提：同步元数据页池，没有设备在途延迟释放；每个请求输出数量有界；后端调用能返回或抛错；新任务集合和循环预算按实验限制；缓存淘汰遵守引用合同。真实系统加入异步传输、外部 connector 或无限流式输入后，需要增加相应等待与故障终止逻辑，不能照搬本证明。

再用多种预算和活动上限运行一个小工作负载，验证最终输出数量、资源回收与审计。它不是形式化穷尽证明，但能检查上述推理覆盖的多个组合，尤其是预算小于活动请求数的情况。

```python
progress_cases = 0
for budget in (1, 2, 4):
    for active_limit in (1, 2, 4):
        engine = Engine(pages=4, block_size=2, token_budget=budget,
                        chunk_size=2, max_sequences=active_limit, prefix_entries=0)
        for i, start in enumerate((0, 3, 6, 9)):
            engine.submit(Request(f"job-{i}", tuple(range(start, start + 4)), 3))
        engine.run(max_iterations=500)
        assert all(r.state == "finished" and len(r.output) == 3
                   for r in engine.requests.values())
        assert len(engine.pool.free) == 4
        assert audit_pool(full_pool_snapshot(engine.pool, engine.cache))
        progress_cases += 1
assert progress_cases == 9
```

这里暂时只调用已经定义的页池审计；下一节定义完整事件审计后，可以把它也应用到这些引擎。这个安排提醒你：教程代码有明确依赖顺序，不能在函数定义之前期待它存在。你独立整理成模块时，应把可复用定义放在入口之前，再在 main 或测试中构造场景。

前进保证还不等于公平保证。系统可以总让某些请求先结束，其他请求长期等待；有限的固定工作负载最终完成，也不能说明持续新请求到来时没有饥饿。评估调度器时至少区分无死锁、有限任务完成、等待上界和服务公平性。你可以先保留本实现的正确性，再设计更明确的恢复优先级，逐项记录改变了哪一种保证。

递归重试是为了让这个小参考实现易读，不是生产推荐的无限深调用结构。活动请求规模很大时，更适合显式循环与严格的尝试上限，并记录每次资源变化。若改写成循环，回归必须保留“每次失败重试减少活动集合”这一进度量，否则只是把递归换成 while，仍可能制造没有状态变化的忙循环。

## 13. 写出独立审计器，再用故障样本证明它有效

页池审计解决物理拥有关系，事件审计解决外部可观察行为。两者不能互相替代：页面完全释放的引擎仍可能重复发送一个 token；输出正确的引擎也可能漏释放页面。我们先用整个 Engine 事件流重建每个请求的输出，再与请求对象中的最终列表比较，并检查终态只出现一次和 batch 预算守恒。

事件中的 iteration 是调度尝试编号，time_ms 是模拟时间，token 是输出值。一个 token 值重复出现不一定错误，例如语言模型可能合法生成两个相同 token；错误的是一次输出位置被重复交付。因此下面比较有序输出序列与终态事件，而不是用 set 去重 token 值。把输出转集合会丢掉顺序和合法重复，无法作为生成验收。

```python
def audit_events(events, outputs, states, budget):
    rebuilt = {rid: [] for rid in outputs}
    terminal = Counter()
    last_time = -1.0
    for event in events:
        if event["time_ms"] < last_time:
            raise AssertionError("event time moved backwards")
        last_time = event["time_ms"]
        kind = event["kind"]
        if kind == "batch":
            work = event["work"]
            if any(w["count"] <= 0 for w in work):
                raise AssertionError("nonpositive scheduled work")
            total = sum(w["count"] for w in work)
            if total != event["scheduled_tokens"] or total > budget:
                raise AssertionError("batch budget or total mismatch")
            if len({w["rid"] for w in work}) != len(work):
                raise AssertionError("duplicate request in batch")
        elif kind == "token":
            rid = event["rid"]
            if rid not in rebuilt or terminal[rid]:
                raise AssertionError("token for unknown or terminal request")
            rebuilt[rid].append(event["token"])
        elif kind in {"finished", "cancelled", "error"}:
            rid = event["rid"]
            if rid not in states or states[rid] != kind:
                raise AssertionError("terminal state mismatch")
            terminal[rid] += 1
            if terminal[rid] > 1:
                raise AssertionError("duplicate terminal event")
    if rebuilt != outputs:
        raise AssertionError("token events differ from output history")
    for rid, state in states.items():
        expected = int(state in {"finished", "cancelled", "error"})
        if terminal[rid] != expected:
            raise AssertionError(f"missing terminal event: {rid}")
    return True

def audit_engine(engine):
    assert audit_pool(full_pool_snapshot(engine.pool, engine.cache))
    running_states = {rid for rid, r in engine.requests.items() if r.state == "running"}
    if running_states != set(engine.running) or running_states != set(engine.pool.tables):
        raise AssertionError("running state/table mismatch")
    for rid in running_states:
        r = engine.requests[rid]
        payload = [token for bid in engine.pool.tables[rid]
                   for token in engine.pool.pages[bid].data]
        if payload != r.context[:r.computed]:
            raise AssertionError(f"computed prefix mismatch: {rid}")
    return audit_events(engine.events,
                        {rid: list(r.output) for rid, r in engine.requests.items()},
                        {rid: r.state for rid, r in engine.requests.items()},
                        engine.token_budget)

preempt_engine = Engine(pages=4, block_size=2, token_budget=4,
                        chunk_size=2, max_sequences=2, prefix_entries=0)
preempt_engine.submit(Request("A", (1, 2, 3, 4), 4))
preempt_engine.submit(Request("B", (5, 6, 7, 8), 4))
for _ in range(100):
    progressed = preempt_engine.step()
    assert audit_engine(preempt_engine)
    if not progressed:
        break
else:
    raise AssertionError("bounded run did not finish")
assert preempt_engine.report()["outputs"] == {"A": [5, 6, 7, 8], "B": [9, 10, 11, 12]}
assert preempt_engine.requests["B"].preemptions == 2
assert len(preempt_engine.pool.free) == 4
```

这段程序在每次外部 step 边界审计，而不是等全部结束。若某一轮已经多保留一个引用，后续操作偶然又把计数减回正确值，只看终态可能漏掉中间损坏。边界检查更容易把错误缩小到最近一次操作；真实大系统为了开销可能只在 debug 或采样模式启用，但学习阶段应该充分使用。

不过它仍有清楚限制：不会在 step 内部每一行拍照；不证明后端 logits 公式正确；不模拟异步设备仍在访问的页；不验证所有真实框架状态类型。独立审计也必须说明观察范围。把它作为一组明确合同来使用，远比宣称“完整证明系统正确”可靠。

### 13.1 注入四类损坏，检查失败位置

使用 deepcopy 修改快照或事件副本，不修改正常引擎。第一类把某页 refs 加一，应该触发引用不守恒；第二类把仍被拥有的页加入 free，应该触发所有权冲突；第三类复制一个 token 事件，应该触发输出序列不一致；第四类删掉 finished 事件，应该触发终态缺失。测试应检查预期错误信息，而不是接受任意异常。

```python
def expect_audit_failure(action, message_part):
    try:
        action()
    except AssertionError as exc:
        assert message_part in str(exc), (message_part, str(exc))
    else:
        raise AssertionError("corruption was not detected")

active_pool = make_shared_pool(4, 2, [1, 2, 3], 2)
healthy = full_pool_snapshot(active_pool)
wrong_refs = deepcopy(healthy)
wrong_refs["pages"][0]["refs"] += 1
expect_audit_failure(lambda: audit_pool(wrong_refs), "reference mismatch")
wrong_free = deepcopy(healthy)
wrong_free["free"].append(0)
expect_audit_failure(lambda: audit_pool(wrong_free), "free ownership")
outputs = preempt_engine.report()["outputs"]
states = {rid: r.state for rid, r in preempt_engine.requests.items()}
duplicate_token = deepcopy(preempt_engine.events)
position = next(i for i, event in enumerate(duplicate_token) if event["kind"] == "token")
duplicate_token.insert(position, deepcopy(duplicate_token[position]))
expect_audit_failure(lambda: audit_events(duplicate_token, outputs, states, 4), "output history")
missing_finish = [deepcopy(event) for event in preempt_engine.events
                  if not (event["kind"] == "finished" and event["rid"] == "B")]
expect_audit_failure(lambda: audit_events(missing_finish, outputs, states, 4), "missing terminal")
assert audit_engine(preempt_engine)
```

第二类快照的 free 列表追加页零后不再排序，但审计不要求 free 展示顺序固定，因为顺序不是拥有不变量；它要求无重复、ID 合法、与零引用严格对应。真实分配器内部可能使用堆、队列或栈，审计应该针对合同而不是某个容器恰好打印出的排列。

第三类重复事件即使 token 数值合法也必须被发现，因为外部会收到比请求历史多的位置。如果系统允许网络消息重试，需要给输出位置附序号并在消费者处去重，那是另一份协议；当前 Engine 的事件流约定每次新生成记录一次，因此重复就是错误。不要把不同层的重试容忍混为一谈。

### 13.2 抢占不能重新消费随机数

温度为零的贪心例子能检查重复输出，却不一定揭露 RNG 重置或额外抽样。我们换成固定均匀 logits 后端、非零温度、每请求独立 seed，比较紧张页池与充足页池。后端没有神经网络数值噪声，生成差异就能更直接指向采样次数或状态管理。

```python
class UniformBackend:
    vocab_size = 16
    def logits(self, context):
        return [0.0] * self.vocab_size

def sampled_workload(pages, engine_type=Engine):
    engine = engine_type(pages=pages, block_size=2, token_budget=4,
                         chunk_size=2, max_sequences=2, prefix_entries=0,
                         backend=UniformBackend())
    engine.submit(Request("A", (1, 2, 3, 4), 4, seed=11, temperature=0.8))
    engine.submit(Request("B", (5, 6, 7, 8), 4, seed=29, temperature=0.8))
    engine.run(max_iterations=100)
    assert audit_engine(engine)
    return engine

scarce = sampled_workload(4)
roomy = sampled_workload(16)
assert scarce.report()["outputs"] == roomy.report()["outputs"]
assert any(r.preemptions for r in scarce.requests.values())
for rid in ("A", "B"):
    assert scarce.requests[rid].rng.getstate() == roomy.requests[rid].rng.getstate()
```

比较 RNG 完整状态比只比较四个输出更强：不同抽样轨迹可能偶然得到相同短序列，但内部随机状态已偏移，后续生成会不同。这里的保证依赖本参考后端和每请求 RNG，不可直接推广为真实 GPU 框架跨 batch、跨 kernel 或跨精度位级一致。真实数值与调度再现性在 S12 单独讨论。

### 13.3 完整实现一个保守恢复策略，并说明付出的代价

原参考引擎在有活动名额时会尽早重新 admit 被抢占请求，这个小例子导致 B 两次重算。一个可解释改进是：只要仍有活动请求，并且等待队列里有被抢占请求，暂时不准入新请求，让当前活动集合先推进；活动集合空后再恢复。这个策略比较保守，会延迟其他等待任务，不能宣称普遍最优。

下面只在学习子类覆盖准入，不修改原参考答案。首次没有活动请求时仍按原逻辑同时准入 A/B；第一次抢占后 B 留在等待，直到 A 结束，再独立运行。我们观察抢占次数与实际安排 token 总量，并保持输出不变。

```python
class ConservativeResumeEngine(Engine):
    def _admit(self):
        recovering = any(self.requests[rid].preemptions > 0 for rid in self.waiting)
        if self.running and recovering:
            return
        super()._admit()

conservative = sampled_workload(4, ConservativeResumeEngine)
assert conservative.report()["outputs"] == scarce.report()["outputs"]
assert conservative.requests["B"].preemptions == 1

def scheduled_work(engine):
    return sum(event["scheduled_tokens"] for event in engine.events if event["kind"] == "batch")

assert scheduled_work(conservative) < scheduled_work(scarce)
assert len(conservative.pool.free) == 4
print({"original_work": scheduled_work(scarce),
       "conservative_work": scheduled_work(conservative),
       "original_time_synthetic_ms": scarce.time_ms,
       "conservative_time_synthetic_ms": conservative.time_ms})
```

这项修改减少重算，却没有让 B 一直更早得到下一个 token。它可能把等待集中在 A 完成之前，也可能在其他负载下损失并发。完整比较应看每请求 TTFT、相邻输出间隔、完成时间和总安排工作，而不是只挑一项下降的指标。这里打印的 time_ms 来自引擎公开成本公式，是策略实验的模拟量，不是 H100 性能。

再思考流量持续到来的情况：这个简单恢复闸门可能让后来请求等待很久，需要更细的优先级、保留容量或等待上界。作为第一份策略补丁，它的价值是规则清楚、反例明确、代码完整，能够让你学会以正确性不变量约束性能改动，而不是一上来构造难以解释的复杂调度器。

## 14. 取消、数值缓存与异步生命周期怎样接起来

### 14.1 在三个不同阶段取消同一类请求

取消发生的时刻会改变资源结果。尚未进入运行的请求没有活动页；部分 prefill 请求可能已经拥有私有页，但尚未发布前缀；提示完成后取消，活动引用释放了，缓存引用仍可能保留。三者最终状态都叫 cancelled，却不能期待 free 页数相同。

完整案例按这三个时刻分别构造新引擎。每个请求的输出上限给得较大，确保第一输出后尚未自然结束。检查第一次取消返回 True、第二次 False，并确认重复操作没有新增终态事件或修改页池。

```python
def cancellation_case(steps):
    engine = Engine(pages=8, block_size=2, token_budget=4, chunk_size=4)
    engine.submit(Request("cancel-me", (1, 2, 3, 4, 5), 8))
    for _ in range(steps):
        engine.step()
    output_before = list(engine.requests["cancel-me"].output)
    assert engine.cancel("cancel-me") is True
    after = full_pool_snapshot(engine.pool, engine.cache)
    event_count = len(engine.events)
    assert engine.cancel("cancel-me") is False
    assert full_pool_snapshot(engine.pool, engine.cache) == after
    assert len(engine.events) == event_count
    assert engine.requests["cancel-me"].output == output_before
    assert engine.requests["cancel-me"].state == "cancelled"
    assert engine.pool.tables == {}
    assert audit_engine(engine)
    return engine

cancel_pending = cancellation_case(0)
cancel_partial = cancellation_case(1)
cancel_after_prompt = cancellation_case(2)
assert len(cancel_pending.pool.free) == 8
assert len(cancel_partial.pool.free) == 8
assert len(cancel_after_prompt.pool.free) == 6
assert len(cancel_after_prompt.cache.entries) == 1
cancel_after_prompt.cache.clear()
assert len(cancel_after_prompt.pool.free) == 8
```

第三个结果剩六页是合法缓存保留，不是取消失败。若业务要求某类数据撤销后连缓存也不可复用，就需要额外失效策略及 namespace/version 设计，不能只调用请求取消。资源释放与数据生命周期策略有关，但并非同一个 API 的所有责任。R08 的删除传播与这里的缓存拥有关系可以用同样的“谁还持有可见版本”问题连接。

当前 Engine 是同步教学系统：step 返回时本轮模拟工作已经完成，cancel 不会与真实设备读取交错。因此它可以直接释放活动页。这不是生产异步系统的充分实现。把逻辑取消移植到 GPU 时，必须另问在途 kernel、传输或采样是否仍引用这些页面，不能只看 Request.state。

### 14.2 逻辑页表保持不变，payload 从整数变成什么

S18 的 `Page.data` 是 token ID 元数据，它让我们容易检查“computed 前缀是否正确”。真实 KV cache 保存每层投影后的 key/value，至少要说明 token 位置、KV head、head dimension、层与 dtype。一个 token ID 是整数标识，不能直接当作一行 K/V；模型计算把标识变成隐藏状态和投影，页池只管理存放位置。

例如单层、两 KV head、每头四维、每页两位置、K/V 都用 FP32，一页有效数值容量是 $2\times2\times2\times4\times4=128$ 字节。这里五个因子依次是 K/V 两份、每页位置数、head 数、每头维度与元素字节数。若有三层且每层相同布局，就再乘三。把 query head 数误用为 KV head 数，会在 GQA 配置下高估缓存。

下面只核算形状与字节，不伪造真实模型结果。数组是零初始化的数值存储；它证明布局与容量，不证明 attention 正确。

```python
import numpy as np
physical_pages, block_size, kv_heads, head_dim = 4, 2, 2, 4
keys = np.zeros((physical_pages, block_size, kv_heads, head_dim), dtype=np.float32)
values = np.zeros_like(keys)
bytes_per_page = (keys.nbytes + values.nbytes) // physical_pages
assert bytes_per_page == 128
logical_table = [3, 1]
position = 2
logical_page, offset = divmod(position, block_size)
physical_page = logical_table[logical_page]
assert (physical_page, offset) == (1, 0)
keys[physical_page, offset, 1, 2] = 7.0
assert keys[1, 0, 1, 2] == 7.0
```

逻辑位置二落在第二逻辑页、页内零号位置，再经表映射到物理页一。这套寻址与整数 payload 时相同，但现在还有 head 和 feature 轴。COW 也要复制有效 K/V 行，而不是仅复制 token 列表；复制之后，旧请求和新请求各自的 attention 应与独立完整前向一致，才能说明数值隔离正确。

课程已经在 [S19](19-numeric-paged-kv.md) 提供真实 NumPy K/V 页池、非连续物理页、COW、在线 softmax 和 TinyDecoder 对照，不把这部分留作没有答案的未来工程。你现在进入 S19 时，应携带本章的引用和事务不变量，再增加数值等价、掩码、旧槽污染与每层状态的验证。两章合起来覆盖管理与计算，不能只选其中一种证据。

### 14.3 资源“无拥有者”与“无人正在使用”并不相同

异步系统可以在主机上取消请求，但设备仍在执行已经提交的读写。若主机把 refs 减零后立即把页放入 free，新请求可能得到同一物理页，随后旧 kernel 又写回，污染新请求。此时逻辑引用计数完全可能看起来正确，所以单靠本章 audit_pool 发现不了时间上的 use-after-free。

需要扩展资源状态：逻辑拥有引用之外，记录尚未结束的执行使用或相关 fence；逻辑引用归零时先进入延迟回收队列；只有所需设备完成条件成立，才进入真正可分配的 free 集合。完成事件必须对应实际访问，单纯过了一段主机时间不是证明。S17 的 CancelFence 教学模型提供逻辑排空例子，K13/K17 继续解释设备执行顺序。

这也改变了容量监控的解释。一个页可能没有活动请求拥有、没有缓存条目拥有，但仍在等待设备完成，暂时不可分配。生产报告应区分活动、缓存、延迟回收和真正空闲，而不是把所有非活动页都叫“泄漏”。诊断时列出每类的归属和解除条件，才能知道该等事件、淘汰缓存还是修引用。

### 14.4 用请求状态与页状态的交叉表检查设计

请求状态与页状态不是同一个状态机。running 请求可能等待加载；finished 请求可能仍有历史对象；缓存条目可能跨多个请求存活；延迟回收页可能在没有活动请求时仍存在。设计文档若只画一条 pending→running→finished 的线，会遗漏资源这条独立生命周期。

| 请求观察 | 活动表 | 缓存引用 | 可能的资源解释 |
|---|---|---|---|
| pending | 无 | 可有其他请求留下的缓存 | 当前请求尚未获得活动资源 |
| running | 有，可能暂时为空 | 可有 | 已准入，仍需追踪实际计算位置 |
| waiting 且被抢占 | 本教学实现无 | 可有 | 输出历史保留，计算状态需要重建 |
| finished/cancelled | 本教学实现无 | 可有 | 历史仍在，缓存可以合法保留 |
| 异步请求已终止 | 取决于实现 | 可有 | 还应核对在途引用与延迟回收 |

表里的最后一行明确超出本同步引擎，不用一条假断言把真实框架压成教学状态。读新系统时先问它在哪一列增加了状态或责任，再扩充审计器。这样你的知识可以迁移到混合 attention/SSM 缓存、跨节点 connector 和新的调度实现，而不是只记住本仓库几个类名。

## 15. 分级练习与完整参考答案

先做预测，再看答案；代码题可以使用本章完整定义，但必须解释每个断言在检查什么。不要把“运行通过”当成能够解释中间状态的替代，也不要因为自己写得与参考不同就立即认定错误，先比较合同与边界。

### 题一：A 第二轮后，三张页各由谁拥有

答案：P0/P1 各有两个拥有者，分别是 A 活动页表和一个前缀缓存条目，因此 refs 为二；P2 只有 A 活动表拥有，refs 为一。数据依次为一二、三四、五；输出六在 Request.output 中，还不在页 payload 中。free 有五张页。若回答“缓存保存了五个 token，所以占五页”，说明混淆了 token、逻辑页和物理页。

### 题二：A 结束后 computed 仍为六，为什么不是六个活动位置

答案：Request 对象保留历史 computed 方便报告，但活动页表已经移除。此时只有缓存拥有前四个位置对应的两张页，尾页 P2 已清空释放。判断占用必须从活动表、缓存和其他合法资源引用重建，不能仅把历史计数换算成页面。第五节的完整程序同时断言终态输出、空表和 refs，提供可执行核对。

### 题三：缓存四 token、块大小二，长度四的相同提示为何只命中二

答案：本实现保留至少最后一个位置以重新得到 logits，允许命中上限先变成三，再向下对齐完整块，得到二。它不是语义不相同，而是状态保存范围与块粒度共同决定的策略。若提示改成长五且前四一致，上限四，便可命中四。第六节的 variants 程序给出所有输入与期望，不需要另外猜测试数据。

### 题四：同样 token、不同 namespace，输出一样是否说明隔离失效

答案：不能。BigramBackend 对这两个 namespace 没有真实 adapter 数值差别，结果可能恰好相同；应检查 attach 命中为零、没有采用其他 namespace 的页引用。隔离测试要直接观察被隔离的资源身份。真实系统还要确保 namespace 来自可信模型/租户上下文，而不是让不可信请求任意伪造身份字符串。

### 题五：共享满页追加一项，与共享未满尾页追加一项有什么区别

答案：满页不会被改写，追加只需一张新逻辑尾页，原满页继续共享；未满共享尾页将被写入，必须先 COW，哪怕追加后仍只占一个逻辑页。若只按逻辑页数量差计算，第二种会误以为零张新页。第八节 independent_needed 把两部分相加，并对满页、半页和空追加完整枚举。

### 题六：写出空追加不改变状态的验收

答案：对空表、私有尾页、共享尾页都保存完整快照，调用 append 空列表后要求完全相同。空追加既不增加逻辑位置，也不触发 COW 或引用变化。下面是完整共享尾页例子，其余情形已包含在三百四十五组枚举中。

```python
empty_append_pool = make_shared_pool(3, 4, [1, 2, 3], 2)
old = full_pool_snapshot(empty_append_pool)
assert empty_append_pool.needed("A", 0) == 0
empty_append_pool.append("A", [])
assert full_pool_snapshot(empty_append_pool) == old
```

### 题七：失败只检查 output 不变够不够

答案：不够。资源预检错误可能破坏 free、refs 或映射，而暂时还没影响输出。要比较完整池快照，并检查其他拥有者数据不变；失败后再进行一个合法操作，确认结构仍可用。第八节少一页测试覆盖预检失败，第十三节故障快照验证审计器能发现不可见的计数损坏。不能用重建整个池掩盖局部操作已经伤害其他请求。

### 题八：抢占后 B 重算到 prompt 长度四，为什么不能再次输出九

答案：B 以前已经输出九，context 长度五，computed 四仍落后当前已知上下文。必须先把九对应的输入位置补算到 computed 五，再采样十。判断条件是追上当前 context，而不是仅追上原 prompt。若每次重算提示都再采样，会重复输出并多消费随机数；非零温度与 RNG 状态对照可检测这个边界。

### 题九：外部 step 后 iteration 从二跳到四，是不是漏了一次模型输出

答案：没有必然关系。第三次调度尝试可能全部 allocation_blocked，随后抢占 B，内部递归重试产生第四次计划；外部只看到最终边界。事件流中应有阻塞与抢占记录，实际 batch 事件才说明有安排工作。不要把 iteration 直接当 forward 次数。检查事件类型与计数增量位置，才能正确解释指标。

### 题十：实现并比较一个减少重复抢占的策略

完整答案是第十三节 ConservativeResumeEngine：有活动请求且等待队列存在恢复请求时暂停准入，活动集合空后再调用原准入逻辑。相同均匀后端与 seed 下，它把 B 的抢占从二次降到一次，安排工作从二十 token 降到十八，输出与 RNG 轨迹保持合同。模拟总时间由约十四点一降到十三点六毫秒，但这不是设备测量，也不能推广到其他工作负载。你的报告还需比较各请求等待和完成时间，讨论保守闸门可能损失并发的代价。

### 题十一：缓存淘汰一次，没有新 free 页，该检查什么

答案：先列出被淘汰条目持有的页，再重建每页其他拥有者。第五节两个长度前缀共享 P0 的例子里，淘汰短条目只把 refs 从二减一，长条目仍持有，所以没有页释放是正确结果。若所有活动与缓存拥有者都不存在，refs 仍非零才值得查漏 unpin；若 refs 为零但仍在表中，则是危险的悬空引用。不能以“每次淘汰必须释放一页”作为通用断言。

### 题十二：取消后还剩两张缓存页，是否应该强行清空所有页

答案：不应该。取消释放当前请求活动所有权，其他请求或缓存条目可能合法持有同页。第十四节第三个取消案例留下两张前缀缓存页，clear 对应缓存后才全部释放。如果业务要求撤销该内容，设计有范围的缓存失效；不能清空整个池破坏其他请求，也不能直接把 refs 置零跳过所有权关系。

### 题十三：让审计器检测一个空闲页残留 payload

完整答案是在健康快照副本里选一张 free 页，填入一个 token，调用 audit_pool，要求失败信息指向 payload。不要修改真实引擎来做这个审计器单元练习，也不要选择非空闲页，否则测试关注的条件变了。

```python
payload_corruption = deepcopy(healthy)
free_id = payload_corruption["free"][0]
payload_corruption["pages"][free_id]["tokens"] = [999]
expect_audit_failure(lambda: audit_pool(payload_corruption), "payload mismatch")
assert audit_pool(healthy)
```

这证明审计器确实读取全部页，而不仅是活跃页展示。对于真实 GPU 张量，释放内存不一定立即物理清零，allocator 合同可能选择有效长度与访问隔离而非清零；迁移审计时要使用真实契约，不能直接把本教学 PagePool 的清空规则强加给所有后端。

### 题十四：把同一个 Request 对象交给两个 Engine 为什么应拒绝

答案：Request 不只是不可变输入，它还包含输出列表、computed、state、时间和 RNG。两个引擎同时改变同一对象，会让状态与各自页池无法一致，比较两种策略时也可能让第二次实验从第一次终态开始。当前 submit 要求新鲜 pending 对象并记录已提交标志；正确比较方式是用相同输入参数分别构造新对象。下面验证拒绝不会往第二个引擎留下半登记。

```python
shared_request = Request("single-owner", (1, 2), 2)
owner_engine, other_engine = Engine(), Engine()
owner_engine.submit(shared_request)
try:
    other_engine.submit(shared_request)
except ValueError:
    pass
else:
    raise AssertionError("mutable request accepted by two owners")
assert other_engine.requests == {}
owner_engine.cancel("single-owner")
assert audit_engine(owner_engine)
```

## 15.1 综合交付：把状态表、程序和解释放在一起

第一份交付是 prefix 全过程表，逐行列出 A/B 状态、页表、payload、refs、缓存拥有者和 free。第二份是 COW 边界枚举结果，包含成功与少一页失败，并解释公式的两个部分。第三份是抢占轨迹与独立审计器，附四类故障注入的预期失败位置。第四份是保守策略对照，保留输出、RNG、工作量和每请求指标，不只截取一个更小的时间。

第五份交付连接 S19：选一组非连续物理页和一个 COW 分叉，用真实 K/V attention 对照完整前向。已有完整参考实现和测试，任务是自己先预测、再实现或修改一个小部分、最后用错误页表负向控制验证敏感性。不要只把参考代码复制到新文件就算完成；解释每层、每头、每位置的数据如何对应，才说明你把管理与数值真正接起来。

## 16. 完成检查

你应能在看到一个refs异常时追到谁多retain或少unpin；在重复token时追到重算与sampling边界；在表面显存不下降时区分活动引用和缓存引用；在迭代号跳跃时区分空计划与实际执行。这些能力比记住某个框架的几十个启动参数更接近源码级掌握。

最后把自己的状态表、审计器、变异测试和一个改进策略提交到学习分支。保留原参考实现，避免覆盖答案后无法比较。你可以继续使用这些不变量分析任何新的cache manager、混合状态模型或跨节点connector，而不必每次从架构名词重新开始。

## 接口边界也必须可回归

状态机的正常路径正确，还不足以让它成为可复用组件。本版本把`token_budget`、`chunk_size`、活动序列数、页数和块大小限制为非bool的正整数；缓存条目数和发布token上限允许0但不允许负数或浮点数。Python的`True`是`int`的子类，所以只检查`isinstance(x,int)`会漏掉这个边界。

`submit`只接受没有运行历史的pending请求，同一个Request对象不能交给两个Engine；跨策略比较应根据同一输入重新构造Request。把一个已经finished的对象重新submit，会让“终态有输出与完成时间”的不变量在入口失效。报告中的输出列表是副本，调用者修改报告不应改坏引擎。

引用计数操作先验证整个输入，再修改：例如adopt需要页0和页1，而页0有效、页1已经free，失败后页0的引用计数也必须保持原值。fork的父请求不存在时，不应留下空的child页表。这些约束由`tests/test_serving_boundaries.py`的9个回归测试覆盖。练习时先临时移除一个预验证，运行最小反例，再恢复；验收不仅是“报错”，还要比较异常前后的页表、refs和free集合完全相同。
