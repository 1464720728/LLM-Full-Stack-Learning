# S07 采样、停止与流式输出：把 logits 变成可靠的可见结果

本章接着 S06：Runner 已经得到了最后有效位置的 logits，但请求还没有完成。你需要把分数变成一个 token，把 token 变成文本，把文本封装为事件，再把事件送到读取速度不受你控制的客户端。任何一层错误都可能让模型算对了却输出错了。本章从概率与字符串的基本概念开始，逐步实现采样、跨片段停止匹配、增量 UTF-8/SSE 解码和有界输出队列。

先修只需要 Python 列表、字典、函数、异常，以及“一个请求有自己的状态”这个概念。异步部分会解释为什么队列会阻塞，不要求你预先熟悉高性能网络编程。所有完整示例都使用标准库或本仓库 CPU 模块，不调用付费 API，不需要占用服务器 GPU。现有 Engine 只实现固定输出 token 数，文本停止和网络背压由本章独立参考程序教学；不要把新增讲解误认为现有 mock_server 已经拥有完整聊天流协议。

## 1. 第一课：六种“输出单位”不能混用

logit 是每个候选 token 的未归一化分数，长度通常等于词表大小。token ID 是词表中的一个整数索引。文本字符是 Unicode 里的字符，编码为 UTF-8 后又变成若干字节。SSE 事件是应用协议记录，TCP 或 HTTP 库返回的 chunk 是本次收到的一段字节。它们之间没有固定的一对一关系。

一个 token 可能对应一个英文单词片段、几个标点、一个汉字的一部分字节表示，或某种特殊符号。具体规律取决于 tokenizer。一个可见汉字可能需要多个字节，网络可以在任意字节之间切开；一个 SSE 事件可能同时带多个 token，也可能只是角色信息、用量统计或完成标志。不能用“收到五块”推断“生成五个 token”。

| 层次 | 例子 | 边界由谁决定 | 可以直接计作模型 token 吗 |
|---|---|---|---|
| logits | [1.2,-0.3,2.1] | 模型词表与输出投影 | 不可以，是分数向量 |
| token ID | 42 | tokenizer 词表 | 可以，但需真实 token 语义 |
| 文本字符 | 学 | Unicode 文本 | 不可以 |
| UTF-8 bytes | 一个字符的多字节编码 | 编码规则 | 不可以 |
| SSE event | data 行与空行 | 应用协议 | 不一定 |
| 网络 chunk | 本次 read 得到的字节 | 传输与缓冲时机 | 不可以 |

这一区分直接影响性能指标。客户端第一次收到 HTTP header 不是模型首 token；收到只有角色名的事件也不是首段正文。若你只能观察文本片段到达时间，就把指标命名为首文本延迟和片段间延迟。除非服务端提供真实逐 token 时间，否则不要把片段间延迟改名成 ITL，让读者误以为每个点都是一个模型 token。

它也影响停止规则。EOS 是 token 级事件，停止字符串是文本级事件，客户端断开是连接级事件，GPU 完成是执行级事件。把所有事情都放在同一个 `if output.endswith(...)` 中处理，会漏掉许多状态。我们会先分别建立清楚契约，再讨论它们如何在同一个请求生命周期里协作。

## 2. 第二课：用能手算的数字理解 softmax

设三个候选 token 的 logits 为零、二的自然对数、三的自然对数。指数权重分别为一、二、三，所以概率是六分之一、六分之二、六分之三。logits 可以为负数，可以大于一，和也不需要等于一；概率必须非负并且总和为一。这是“模型给分数”与“采样需要分布”的根本区别。

对所有 logits 加同一个常数不会改变 softmax。因为每个指数都乘上同一个因子，分母也乘同一因子，归一化后抵消。因此实际计算会先减去最大值，使最大的指数等于一，避免直接对极大正数求指数。稳定 softmax 不是修改模型偏好，而是用数值更安全的方式计算同一个分布。

当温度 T 大于零时，先把 logits 除以 T 再计算概率。若两个 logits 为零和二的自然对数，温度一时权重一比二，温度二时权重一比根号二，温度二分之一时权重一比四。温度改变相对概率的尖锐程度，不是把最终概率直接除以一个数后就结束；后者不能保证归一化。

温度零在本课程中单独定义为贪心选择，即选最大 logit 对应 token。它不是把公式中的分母真的设为零。最大值并列时还需要确定平局规则，本地实现使用最小 token ID。这个规则使 CPU 参考可复现，但不同服务实现可能采用不同约定，因此跨后端比较时应确认平局行为，而不只看 temperature 字段。

```python
import math
from labs.serving.sampling import probabilities

p = probabilities([0.0, math.log(2), math.log(3)], temperature=1.0)
assert all(abs(a-b) < 1e-12 for a,b in zip(p, [1/6, 2/6, 3/6]))
shifted = probabilities([1000.0, 1000.0+math.log(2),
                         1000.0+math.log(3)], temperature=1.0)
assert all(abs(a-b) < 1e-12 for a,b in zip(p, shifted))
cold = probabilities([0.0, math.log(2)], temperature=0.5)
assert abs(cold[1] - 0.8) < 1e-12
assert probabilities([2.0, 2.0, 1.0], temperature=0.0) == [1.0, 0.0, 0.0]
print("softmax, temperature, and tie rule checked")
```

浮点计算仍然有有限精度，所以上面的非整数结果使用容差比较，而不是要求 Python 小数完全相等。选择容差时要结合 dtype 与计算路径，不能用一个极大的阈值把明显错误也算通过。本例只涉及几个双精度数，容差可以很小；大规模混合精度模型的容差应通过专门数值实验确定。

## 3. 第三课：top-k 与 top-p 实际删掉了什么

top-k 保留分数最高的 k 个候选，其余概率设为零，再对保留集合重新归一化。这里 k 为零表示不启用 top-k，而不是“保留零个 token”。这个约定来自本地接口，阅读其他服务时仍应确认参数语义。若 k 大于词表大小，Python 切片自然保留全部候选，本实现不会因此增加虚构 token。

top-p 也叫 nucleus sampling。先按概率从大到小排序，依次纳入候选，直到累计概率达到或超过阈值 p。跨过阈值的那一个候选必须保留，否则可能只剩下累计不足 p 的集合。保留集合最后重新归一化，采样只在这个集合中进行。它保留的候选数量随分布变化，不是固定保留 p 百分比的词表。

仍用原概率 `[1/6,2/6,3/6]`。按概率排序是 token 二、token 一、token 零。top-p 为零点七时，先取二得到零点五，不够；再取一达到约零点八三三，停止。保留的一和二归一化为零点四和零点六。若 top-p 为零点四，第一项已经达到阈值，只保留 token 二，最终分布变成确定选择。

本实现顺序是先按分数排序并应用 top-k，在剩余集合上根据温度计算权重，再应用 top-p 并归一化。对于正温度，除温度不会改变排序，所以这与“先温度再 top-k”在选取集合上等价；但 top-p 使用的是 top-k 后归一化的分布。若换成先 top-p 再 top-k，某些输入下最终集合会不同，参数名字相同并不足以保证服务行为一致。

| 原分布 | 配置 | 截断前累计过程 | 最终分布 |
|---|---|---|---|
| 1/6、2/6、3/6 | top-k=2 | 保留 token 2 与 1 | 0、0.4、0.6 |
| 0.6、0.3、0.1 | top-p=0.8 | 0.6，再到0.9 | 2/3、1/3、0 |
| 0.6、0.3、0.1 | top-p=0.5 | 第一项已足够 | 1、0、0 |
| 0.4、0.35、0.25 | top-k=2,top-p=0.6 | 截断后最大项约0.533，需第二项 | 约0.533、约0.467、0 |

最后一行提醒你：top-p 的阈值针对当前归一化分布，不是原始 logits 的和，也不是未归一化指数权重的和。很多“看起来能运行”的手写实现会直接累加 logits，得到完全没有概率意义的结果。先做小数表格，再写循环，可以尽早发现这种概念错误。

## 4. 第四课：逐行读采样实现，理解错误怎样暴露

当前完整实现位于 [sampling.py](../../labs/serving/sampling.py)。第一组校验拒绝空 logits、NaN 和正无穷，并要求至少一个有限值。负无穷被允许，因为它常用来屏蔽禁止的 token。全部负无穷意味着不存在合法候选，应该报错，而不是让 softmax 产生 NaN 后随便挑一个 token。

排序键 `(-logits[i],i)` 有两部分。先按分数降序，再按 token ID 升序；这使 top-k 在边界分数相同时拥有明确规则。温度零分支直接返回 one-hot 概率，不消耗随机数。温度为正时，保留集合的最大值用于稳定指数，weights 字典只包含候选，返回时没有保留的 token 得到零概率。

top-p 循环先把当前候选加入 kept，再更新累计概率，最后判断是否达到阈值。顺序不能交换；如果先判断“加上当前项会不会超过阈值”，就可能把必须保留的跨阈值项排除。最后重新计算 total，是因为候选集合已经缩小，旧分母不再适用于新的概率。

sample 函数从 RNG 取得区间零到一中的一个随机阈值，按 token ID 顺序累计概率，阈值落在哪个区间就选哪个 token。这个遍历顺序不必与 top-p 排序一致，只要分布正确且随机变量均匀，得到的概率仍然正确。若你要跨实现逐 token 重现相同随机数，遍历顺序又会成为必须对齐的细节。

末尾的 last_nonzero 处理浮点累计没有恰好到一的情况。它不是对任意非法分布的修复，只是在已经通过验证和归一化后提供一个舍入边界退路。如果前面的概率全为零、含 NaN 或和远离一，应该在构造概率时失败，不能靠末尾返回一个默认 token 把错误藏起来。

```python
import math
import random
from labs.serving.sampling import probabilities, sample

assert probabilities([-math.inf, 2.0], 1.0) == [0.0, 1.0]
for bad in ([], [math.nan, 0.0], [math.inf, 0.0],
            [-math.inf, -math.inf]):
    try:
        probabilities(bad, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid distribution accepted")
rng = random.Random(4)
before = rng.getstate()
assert sample([1.0, 2.0], rng, temperature=0.0) == 1
assert before == rng.getstate()
print("invalid logits rejected; greedy sampling consumes no RNG")
```

一个采样错误需要沿请求生命周期清理。当前 Engine 捕获 backend/采样异常，将请求置为 error 并释放所属页，不把无效 token 塞进 output。对服务端来说，返回明确错误比返回看似正常的乱码更容易诊断。若错误来自结构化解码把全部 token 屏蔽，应该检查语法状态与 tokenizer 边界，而不是调高温度。

## 5. 第五课：随机数流是请求状态的一部分

设一个共享 RNG 依次产生 u1、u2、u3、u4。甲单独运行时使用前三个；甲乙交错运行时，甲可能使用 u1、u3、u5。即使甲每一步 logits 完全相同，输出也会变化。这个现象来自随机数分配方式，不是矩阵乘法精度，不应该用“GPU 不确定性”一笔带过。

本地 Request 在创建时构造自己的 `random.Random(seed)`。甲的随机流由甲消费，乙的随机流由乙消费，因此调度顺序不会直接改变甲使用哪个随机样本。重算已有历史、执行中间 prefill chunk、处理取消后的迟到结果，都不应再消耗一次有效输出的随机数。否则“相同 seed”仍然无法重现同一请求。

seed 只是随机流的初始条件，不是整个服务确定性的充分条件。模型版本、prompt token、模板、logits 数值、采样实现、候选 mask、随机数消费次数都会影响结果。两个不同 RNG 算法使用同一个整数 seed，也不保证产生相同序列。做可复现报告时，应记录这些条件，而不只写一句 seed=42。

```python
import random
from labs.serving.sampling import sample

logits = [0.2, 0.4, 0.8]
def generate(order):
    streams = {"A": random.Random(12), "B": random.Random(29)}
    out = {"A": [], "B": []}
    for rid in order:
        out[rid].append(sample(logits, streams[rid], temperature=1.0))
    return out

one = generate("AAABBB")
two = generate("ABABAB")
assert one == two
# 这个断言验证的是每请求消费相同数量的同一随机流。
print(one, "independent of this scheduling permutation")
```

某些系统使用计数式随机方案，把请求身份与输出位置映射到随机样本。它可以减少对可变 RNG 对象的依赖，但仍需定义身份、重试、分支、推测采样和接受拒绝过程如何分配计数。不能把“使用 counter”当成自动解决所有复现问题的魔法。本课程先通过独立 Random 对象理解归属，再在 S12 讨论更广的确定性边界。

## 6. 第六课：停止意味着停止哪一层

输出长度上限是 token 级预算。EOS 是模型词表中的结束标记。停止字符串是在解码文本中寻找某段内容。客户端取消意味着不再交付结果并开始释放资源。时间预算可能在排队、计算、读取或整请求级触发。它们可以同时发生，服务必须定义优先关系与最终 finish reason，而不是把所有结束都记为 success。

假设模型刚生成一个 EOS token。API 可以约定不把 EOS 对应特殊文本返回给用户，但生成计数、缓存进度和终止状态仍然要正确处理。若最后一个生成 token 不会再输入模型，就没有必要为它额外计算一次 KV。S01 已经说明了“已生成输出数”与“已计算位置数”的差一关系；停止逻辑不能为了让数字看起来相等而多跑一步。

停止字符串可能跨 token、跨 detokenizer 片段、跨 SSE 事件。若 stop 为 END，输入片段依次是 E、N、D，单独在每个片段中查找永远找不到。更棘手的是，如果前两片段已发送给用户，第三片段到来后你无法通过普通追加式流把先前的 EN 撤回。因此停止检测必须在对外发送之前保留尚不能确认安全的尾部。

文本停止还要定义是否包含 stop 本身、多个 stop 同时出现选哪个、空 stop 是否合法、结束时未完成前缀是否输出。本章采用明确契约：stop 非空；返回最早出现的完整 stop 之前内容；stop 本身不返回；一旦停止后后续片段忽略；正常结束时尚未形成完整 stop 的尾部作为普通文本返回。其他 API 可以不同，但不能没有定义。

工具调用完成也不是“看见右花括号就停止”。JSON 字符串内部可以有括号，转义规则会改变其含义，工具参数可能跨片段。需要真正 parser 或约束状态机判定结构完整。即使结构合法，还要验证 schema 与业务权限。Agent 卷会继续讨论工具执行；本章只负责把输出边界处理正确，不把文本解析直接等同于安全执行许可。

## 7. 第七课：完整实现跨片段停止过滤器

最简单的离线实现是把所有片段拼接后查找 stop，但这样无法及时流式输出。下面程序只保留最多“最长 stop 长度减一”的待确认尾部。没有命中时，前面的部分不可能再与未来输入组成一个跨边界 stop，所以可以安全发送。这里长度以 Python 字符串的 Unicode 码点计，不是 UTF-8 字节数。

```python
class StopFilter:
    def __init__(self, stops):
        if not stops or any(not isinstance(s, str) or not s for s in stops):
            raise ValueError("nonempty stop strings required")
        self.stops = tuple(stops)
        self.tail = ""
        self.hold = max(map(len, stops)) - 1
        self.stopped = False

    def feed(self, text):
        if self.stopped:
            return ""
        self.tail += text
        matches = [self.tail.find(s) for s in self.stops]
        matches = [i for i in matches if i >= 0]
        safe = max(0, len(self.tail) - self.hold)
        if matches and min(matches) <= safe:
            cut = min(matches)
            visible = self.tail[:cut]
            self.tail = ""
            self.stopped = True
            return visible
        visible, self.tail = self.tail[:safe], self.tail[safe:]
        return visible

    def finish(self):
        if self.stopped:
            return ""
        matches = [self.tail.find(s) for s in self.stops]
        matches = [i for i in matches if i >= 0]
        if matches:
            visible = self.tail[:min(matches)]
            self.stopped = True
        else:
            visible = self.tail
        self.tail = ""
        return visible

def run(parts, stops):
    f = StopFilter(stops)
    visible = "".join(f.feed(p) for p in parts) + f.finish()
    return visible, f.stopped

assert run(["你好E", "N", "D后面的内容"], ["END"]) == ("你好", True)
assert run(["a", "b", "c"], ["END"]) == ("abc", False)
assert run(["xx", "STOP", "yy"], ["END", "STOP"]) == ("xx", True)
assert run(["x", "E", "N"], ["END"]) == ("xEN", False)
assert run(["", "学", "习!"], ["!"]) == ("学习", True)
assert run(["abcd"], ["abcd", "bc"]) == ("", True)
assert run(["abc", "d"], ["abcd", "bc"]) == ("", True)
assert run(["abc"], ["abcd", "bc"]) == ("a", True)
print("cross-fragment stop filtering checked")
```

feed 先把新文本加入 tail，搜索完整 stop，再计算 safe。发现完整匹配不一定能立即停止：它前面可能还有一个尚未完整到达的更长 stop。只有最早完整匹配的起点不晚于 safe 时，才确认停止并返回它之前的正文；否则先输出 safe 之前的确定前缀，保留剩余候选。若两个 stop 同起点，选择其中任何一个都不改变正文边界，但仍要先排除更早起点的候选。此实现采用保守等待，不保证停止确认延迟最小。

设最长 stop 长度为 L，当前 tail 长度为 n，则 hold 等于 L减一，safe 等于 n减L加一与零的较大值。任何还需要未来字符才能完成的匹配，其起点最早为 n减L加一，因此 safe 之前可以安全输出。已有完整匹配的起点小于或等于 safe 时，未来不可能出现起点更早的匹配；即使未来出现同起点的长 stop，正文截止位置也相同，所以条件使用小于或等于。未确认停止时输出安全前缀后，最多保留 L减一字符。这个推理依赖 stop 列表固定，运行中若突然增加更长 stop，之前已经发出的文本无法追回，因此配置必须在请求开始前确定。

手动追踪重叠例子：stop 为 abcd 与 bc，最长长度四。收到 abc 时，safe 为零，虽然 bc 已从索引一完整出现，但索引零的 abcd 还可能完成，因此先不输出。随后收到 d，最早匹配变成索引零的 abcd，停止后正文为空。若输入在 abc 处正常结束，未来不会再补齐 abcd，finish 应选择已经完整的 bc，返回 a 并把 stopped 设为真。这两种结束方式正是状态机需要区分的情形。

finish 只在输入正常结束时调用，此时先重新检查尾部：若存在完整 stop，就返回最早匹配之前的内容并设置 stopped；若没有，才放行全部尾部。尾部可能保存了已经完整但尚未确认的候选，不能无条件输出。连接异常或请求取消时是否交付尾部，要由服务契约决定，不能自动当作正常完成。本程序的 finish 没有网络语义，所以调用者必须区分正常结束与错误退出。它也没有自动生成 finish reason，只返回过滤结果和 stopped 标志，便于上层明确组合状态。

该实现每次搜索所有 stop，适合短 stop 列表与小规模教学。若 stop 很多或文本很长，可以使用前缀函数、自动机或多模式匹配降低重复扫描成本。但优化前必须保留以上边界测试，尤其是跨三片段、重叠 stop、同起点长短前缀、长度一 stop、正常结束时的完整候选、无命中尾部和空片段。算法更快不能以改变可见输出语义为代价。

## 8. 第八课：增量 UTF-8 与 SSE 解码的两层缓冲

Python 的 str 是文本，bytes 是字节。一个中文字符编码后可能有多个字节，网络读取可以只给前一两个字节。如果每个 chunk 都独立调用 decode，半个字符会被当成非法序列；使用 errors="ignore" 虽然不报错，却会永久丢字。正确方法是增量 decoder 保留不完整字节，等下一块补齐。

UTF-8 解码成功后，还没有得到完整 SSE 事件。SSE 以空行分隔事件，一条 data 可以跨多个 data 行，事件本身又可以跨多个网络 chunk。因此有两个不同的缓冲：编码器内部保存未完成字符的字节，SSEDecoder.buffer 保存已经解码但还没有形成完整事件的文本。把这两个状态混成一个普通字符串，很容易在边界处丢失信息。

解析 data 字段时，冒号后至多移除一个普通空格 U+0020，其余字符属于数据。`data:x` 与 `data: x` 都产生 x，`data:  x` 则产生带一个前导空格的 `" x"`。代码使用 removeprefix 移除至多一个空格；lstrip 会删除所有前导空格，造成缩进丢失。制表符和尾部空格也应保留，多行 data 按原有内容用换行连接。这与 A05 的逐行解析规则相同。

本地 [streaming.py](../../labs/serving/streaming.py) 是明确范围的 SSE 子集参考：处理 UTF-8、LF/CRLF、空行分隔、data 行合并和未完成事件缓冲上限。它不实现完整浏览器 EventSource 的重连、last-event-id、retry 字段等全部语义，也不把任何字符串自动当 JSON。收到 data 后是否解析 JSON、是否把某个完成标记特殊处理，属于上层协议。

代码先执行增量 decode，然后将 CRLF 规范成 LF。若当前 chunk 恰好以 CR 结束，暂时保留它；下一块 LF 到来后才形成完整 CRLF。之后循环寻找两个连续 LF，拆出完整 block，将以 data: 开头的行取出并用换行连接。注释行不产生 data 事件。一个 chunk 包含多个事件时，循环会全部返回；没有完整事件时返回空列表。

```python
import json
from labs.serving.streaming import SSEDecoder

wire = ('data: {"text":"学习"}\r\n\r\n'
        ': keepalive\r\n\r\n'
        'data: first\r\ndata: second\r\n\r\n').encode("utf-8")
expected = ['{"text":"学习"}', "first\nsecond"]
for cut in range(len(wire) + 1):
    decoder = SSEDecoder()
    result = decoder.feed(wire[:cut])
    result += decoder.feed(wire[cut:], final=True)
    assert result == expected, (cut, result)
decoder = SSEDecoder()
one_byte = []
for byte in wire:
    one_byte += decoder.feed(bytes([byte]))
one_byte += decoder.feed(b"", final=True)
assert one_byte == expected
assert json.loads(expected[0])["text"] == "学习"
print("all two-piece cuts and one-byte chunks preserve events")
```

这个测试遍历每一个可能的二段切分点，还逐字节输入，因此覆盖汉字中间、CR 与 LF 中间、data 前缀中间以及事件边界。比只测试“正好在空行处切成两块”更强，因为后者没有触及最容易失败的位置。它仍不证明完整 SSE 标准兼容，只证明本参考承诺的这组输入与分块方式正确。

final=True 告诉 UTF-8 decoder 不会再有后续字节，截断字符应报错；如果文本 buffer 还有非空的未完成事件，本实现报 truncated SSE event。缓冲上限限制的是尚未完成的尾部，不是整个连接累计收到的字节数，也不是单次 feed 内所有已完成事件的总量。上层还需要限制请求体、单次读取和总输出；不能用一个缓冲参数代替所有资源限制。

### 8.1 从事件字符串到应用消息，还需要一个小状态机

SSEDecoder 返回字符串以后，上层通常才调用 JSON 解析。一个合法 JSON 对象不一定包含正文：它可能表示角色、工具参数增量、统计信息或错误。客户端应根据明确字段识别消息类型，分别更新状态，不能把对象转成字符串就展示给用户。尤其是 usage 事件中的数字，不应该被误认为模型生成的文字，也不能因为 delta 为空就认定整条流已经结束。

可以把客户端状态分为等待响应、读取事件、已收到应用结束、网络结束和错误。收到应用结束标志后，可能还要读取最终 usage；网络正常关闭但没有应用结束标志，应根据服务契约判断是否截断。反过来，已经收到应用结束而随后连接异常，也不能随意丢弃先前成功结果。状态机要保存“发生过什么”，而不是只在最后看一个连接布尔值。

去重也需要身份。若传输重试重复某个事件，简单把文本追加两次就会重复输出。支持重连的协议应提供稳定事件 ID 或可验证的位置，并定义客户端从哪个位置继续。若服务没有提供这类契约，客户端不能凭文本相同就删除重复片段，因为模型本来可能连续生成两个相同词。内容相同不等于同一个事件。

对工具参数流尤其要保留原始顺序与索引。同一响应可能同时有多个工具调用，每个调用的参数片段归属于不同 call ID；把所有参数字符串放进一个 buffer，会拼成非法甚至语义错误的对象。正确做法是按调用身份分别累计，完成后逐个解析与校验，随后才交给 Agent 的权限与执行层。这个边界把“模型提出调用”与“系统实际执行调用”分开。

## 9. 第九课：慢客户端怎样把输出变成资源问题

模型可能每秒产生很多 token，而客户端正在弱网络上缓慢读取。若服务每生成一个片段就无条件放入无限队列，队列会越积越大。即使模型已经完成，服务仍持有全部未发送结果。很多并发慢客户端叠加时，CPU 内存会成为瓶颈，甚至影响本来健康的请求。

有界队列把积压限制在明确容量内。生产者发现队列已满时，需要等待、暂停调度、降低优先级或取消慢请求，具体策略由服务目标决定。等待并非总是最佳选择：如果等待时仍持有大量 KV，会占用设备资源；如果继续计算又只缓存输出，会占用主机内存。因此背压必须跨越网络、请求状态和调度层，而不只是给 queue 增加一个数字。

下面程序用标准库 asyncio 演示容量为二的队列。没有真实网络，也没有 GPU。consumer 的一次 `await asyncio.sleep(0)` 只主动让出事件循环，不模拟某个可信网络延迟。示例检查顺序与容量，不能拿运行用时报告服务性能。

```python
import asyncio

async def demo():
    queue = asyncio.Queue(maxsize=2)
    received = []
    observed = []
    async def producer():
        for index in range(6):
            await queue.put(("text", index))
            observed.append(queue.qsize())
        await queue.put(("done", None))
    async def consumer():
        while True:
            kind, value = await queue.get()
            try:
                if kind == "done":
                    return
                received.append(value)
                await asyncio.sleep(0)
            finally:
                queue.task_done()
    await asyncio.gather(producer(), consumer())
    await queue.join()
    assert received == list(range(6))
    assert max(observed) <= 2
    return received, max(observed)

print(asyncio.run(demo()))
```

`await queue.put` 在队列满时会暂停当前生产协程，让消费协程获得机会；它不会阻塞整个操作系统线程去忙等。`task_done` 表示已取出的项目处理结束，`join` 等待所有入队项完成处理。done 也是一个队列项，所以即使 consumer 在分支中 return，finally 仍要执行 task_done。这个细节决定 join 会不会永远等待。

真实服务还要处理 consumer 因断开退出后 producer 仍等待 put 的情况。需要一个共享取消事件或任务取消机制，使生产任务退出并释放请求资源；仅仅关闭 socket 不保证调度器知道请求已不再需要。反过来，producer 出错也要通知 consumer，不能让它永久等待一个不会到来的 done。上层可以用结构化并发或明确的任务管理实现双方命运关联。

输出缓冲区所有权也要明确。若队列里放的是指向可复用数组的 view，而生产者下一轮原地覆盖该数组，消费者会读到变过的内容。不可变字符串在本示例中避免了这个问题；真实张量或字节池需要复制、引用计数或延迟复用协议。背压不仅是速度问题，也是“谁还在使用这块输出”的生命周期问题。


### 9.1 四种超时分别保护什么

连接超时限制建立连接所花的时间；读取超时限制一段时间没有收到新数据；请求总超时限制从发起到完成的总时长；应用空闲超时可以限制多久没有收到有效正文。四者不等价：服务每隔几秒发 keepalive，可能一直满足读取超时，却永远没有生成正文，因此仍需要总预算或应用进度规则。

计时应使用适合测持续时间的单调时钟，避免系统时间调整使耗时变成负值。收到任意字节、收到完整事件、收到有效文本，是三个不同的“进度”事件。你要先定义哪个事件重置哪一种超时，再写代码；否则一个无意义注释包可能意外让请求永久存活。

超时触发后，客户端应停止读取并通知服务取消；服务收到取消后阻止新输出、回收可安全释放的资源，并等待在途设备工作达到释放条件。客户端本地抛异常并不证明服务器已经停止计算，因此基准报告应记录取消是否得到确认。对共享 H100 实验，这一点也影响资源占用：短客户端超时如果没有传播，后台仍可能运行很久。

这些规则不要求你一开始实现复杂网关，但要求你在小工程中明确写出边界。一个只有几十行的客户端，如果能正确区分完成、截断、应用错误和取消，比一个不断吞掉异常并重试的长脚本更适合成为后续 Agent 系统的可靠依赖。

## 10. 第十课：四个故障现场与定位顺序

**现场一：中文偶尔少字或出现替代符号。** 保存合成输入的原始 bytes，按每个切分点喂给解码器。若独立 decode 每块失败，而增量 decoder 正常，根因是编码边界。不要用 ignore 或 replace 掩盖问题，除非 API 明确要求容错并记录了损失。随后检查 tokenizer 的增量解码是否也需要保留状态，避免只修网络层却仍逐 token 错误拼接。

**现场二：同 seed 在不同并发下输出不同。** 先固定 logits，用本章独立 RNG 程序测试调度重排。若固定 logits 仍不同，检查随机流归属和消费次数；若固定 logits 相同而真实模型不同，再比较 logits、backend、batch shape 和数值路径。这样能把采样状态问题与浮点差异分开，避免在错误层面反复设置随机种子。

**现场三：stop 字符串仍然显示给用户。** 检查 stop 检测发生在发送之前还是之后。若已把 E、N 发出才看到 D，即使最后检测到 END，也无法撤回已有流。使用 StopFilter 保留可能形成 stop 的尾部，再测试全部分块方式。如果 stop 包含 tokenizer 特殊符号，应先明确它是 token 级规则还是可见文本规则，两者不能混为一个字符串搜索。

**现场四：客户端断开后内存仍持续增长。** 记录请求状态、输出队列长度、producer/consumer 任务状态和 KV 页所有权。若 consumer 已结束而 producer 仍生成，说明取消没有传播；若请求已终止但队列仍被全局对象引用，说明结果生命周期没有结束。不要只定期清空所有队列，因为那会丢失其他正常请求的数据，也没有修复所有权错误。

这些故障都可以先用合成输入在 CPU 上复现。只有当 CPU 协议与状态正确、差异确实出现在设备数值或真实网络行为时，才需要扩大实验环境。对初学者而言，“先构造最小反例”不是绕开真实系统，而是在真实系统复杂到无法观察之前建立一个可以验证的解释。

## 11. 第十一课：十六道练习与完整答案

**练习一，手算温度。** logits 为零和三的自然对数，温度一时概率为四分之一和四分之三；温度二分之一时指数权重一和九，概率十分之一和十分之九。答案必须从 logits 除温度开始，而不是把四分之三直接除以二分之一。温度零时按最大 logit 贪心选第二项，不计算除零。

**练习二，平局规则。** logits 为二、二、一，top-k 为一，温度一。本地排序按 token ID 升序破平局，因此只保留 token 零，概率为一、零、零。若你实现的排序没有稳定规则，不同设备或并行归约可能选择另一项。验收要写出期望 ID，不能只断言“选到一个最大值”后又声称输出完全可复现。

**练习三，nucleus 阈值。** 分布为零点六、零点三、零点一，top-p 为零点八，答案保留前两项并归一化为三分之二和三分之一。若错误排除跨阈值项，只留下第一项，分布会过度尖锐。用精确构造的 log 概率做程序验证时仍考虑浮点累计误差，不要把十进制边界的表示误差当成算法定义变化。

**练习四，全部候选被屏蔽。** 正确行为是明确报错并清理请求。返回 token 零、把负无穷替换成零、自动取消语法约束都改变了任务契约。定位应检查约束状态、允许 token 集合和词表映射。若业务允许回退，回退必须是显式策略并记录原因，不能藏在 softmax 的异常处理中。

**练习五，计算 RNG 消费。** 一个请求有三段 prefill，最后生成四个 token，中间被抢占重算一次。在本地正温度采样契约下，有效输出采样应消费四次随机数，prefill 中间块和重算历史消费零次。若最终某次模型结果因取消被丢弃，是否提前消费随机数要由提交/完成协议定义，至少不能让它污染另一个请求的随机流。

**练习六，区分统计单位。** 收到四个 SSE 事件，其中两个有正文，一个只有角色，一个是 usage。不能据此得出生成两个 token；最多能说观察到两个正文事件。若 usage 提供 completion_tokens=7，可以记录服务报告输出 token 七个，但仍没有七个逐 token 时间。答案应使用片段延迟而非伪造 ITL。

**练习七，停止尾部长度。** stop 列表为 END 与 STOP，最长长度四，所以未确认停止时最多保留三个字符。输入 abcE 后可发送 a、保留 bcE；后续输入 ND 时形成完整 END，匹配起点二等于 safe 二，可以停止并返回 bc，总正文为 abc。再把 stop 改成 abcd 与 bc：输入 abc 时保留整个尾部，不立即接受 bc；随后输入 d 时返回空正文并停止，若此时直接正常结束则由 finish 返回 a 并停止。这个延迟是保守策略带来的，正确但不一定最小；更精细自动机可以只保留真正可能匹配的后缀。

**练习八，正常结束与取消。** 输入 xEN，stop 为 END，正常结束应返回 xEN，因为没有完整 stop。若客户端取消，是否再尝试发送 EN 不由 StopFilter 决定，应由连接与取消契约决定。完整答案要说明不能把所有 finally 都调用 finish 并写 socket，否则取消后可能继续发送或覆盖原始错误。

**练习九，逐字节 SSE。** 对含中文且使用 CRLF 的 wire 逐字节调用 feed，最后 final=True，应与一次性输入得到相同事件。参考程序已给出完整答案。若改成每字节 decode，中文中间字节会失败；若改成每 chunk split 空行，跨 chunk 的空行分隔会丢失。两个错误属于不同层，必须分别保存状态。

**练习十，多行 data。** 一个事件含 data: first 与 data: second，答案是一个字符串 `first\nsecond`，不是两个独立事件。若第二行改为冒号后有两个空格的 `data:  second`，答案应为 `"first\n second"`，保留一个数据空格。事件边界是空行而不是每个 data 行。注释 keepalive 不产生 data。上层 JSON parser 若不能接受多行数据，应报告协议内容不符合其预期，而不擅自把两行拆成两个 JSON 事件。

**练习十一，未完成事件。** 输入 `data: x` 然后 final=True，本实现报截断事件；输入无效 UTF-8 则由增量 decoder 报编码错误。两者错误类型与原因不同。完整答案还应说明本课程采用严格尾部策略，不宣称与所有 EventSource 客户端的 EOF 处理完全相同。

**练习十二，有界队列。** producer 每次入队一项，容量二，consumer 暂时不取。第三次 put 应等待，而不是继续无限增长。若 consumer 永远退出且没有取消传播，producer 会一直等待。答案要同时给出容量控制与任务终止策略；只增加 maxsize 可以限制内存，却不能保证系统最终完成。

**练习十三，为什么使用不可变输出。** 队列中若存字符串，每个片段的内容不会被后续原地改写；若存复用 bytearray 的 view，消费者可能看到下一轮内容。修复可选择复制成 bytes，或让缓冲池直到消费完成才回收。答案应解释成本：复制增加带宽，引用计数增加生命周期复杂度，不能只说“永远零拷贝更快”。

**练习十四，写离线停止参考。** 把所有片段拼接，查找每个 stop 的最早位置，取非负最小值；没有命中则返回全文。这个离线版本是 StopFilter 的独立 oracle，适合穷举分块测试。它不适合实时输出，但简单可审阅，正好用来验证更复杂的增量版本。

```python
def offline_stop(parts, stops):
    text = "".join(parts)
    hits = [text.find(stop) for stop in stops]
    hits = [i for i in hits if i >= 0]
    return text[:min(hits)] if hits else text

assert offline_stop(["abE", "N", "Dcd"], ["END"]) == "ab"
assert offline_stop(["学习"], ["END"]) == "学习"
assert offline_stop(["aSTOPbEND"], ["END", "STOP"]) == "a"
print("offline stop oracle checked")
```

继续用独立 oracle 验证重叠输入的全部非空分片组合，再插入空片段。每次 feed 后都检查累计输出仍是最终正文的前缀，这能发现“已经发错、后来才发现 stop”的错误；最后同时检查正文和停止标志。完整参考如下，依赖第七课的 StopFilter 与本题的 offline_stop。

```python
# course-example: S07-overlapping-stops
def all_partitions(text):
    if not text:
        yield [""]
        return
    for mask in range(1 << (len(text)-1)):
        cuts=[0]+[i for i in range(1,len(text)) if mask & (1 << (i-1))]+[len(text)]
        yield [text[a:b] for a,b in zip(cuts,cuts[1:])]

for text,stops in (("abcd",["abcd","bc"]), ("abc",["abcd","bc"]),
                   ("xabcdy",["abcd","bc"]), ("xa",["a","abcd"]),
                   ("学习结束以后",["学习结束","习结"]), ("",["!"])):
    expected=offline_stop([text],stops)
    for parts in all_partitions(text):
        stream=StopFilter(stops)
        visible=""
        for part in [""]+[piece for p in parts for piece in (p,"")]:
            visible+=stream.feed(part)
            assert expected.startswith(visible)
            assert len(stream.tail)<=max(map(len,stops))-1
        visible+=stream.finish()
        assert visible==expected
        assert stream.stopped==any(stop in text for stop in stops)
        assert stream.finish()==""
        if stream.stopped:
            assert stream.feed("ignored")==""
```

**练习十五，后端空 logits。** 应在概率构造时拒绝，并让请求进入 error。不能把空列表解释为 EOS，因为 EOS 是词表中有定义的 token，而空 logits 表示模型/接口没有给出可采样分布。若模型确实不需要生成，例如 max_new_tokens 为零，应在调度前正常结束，根本不调用采样。

**练习十六，协议完成不等于业务成功。** SSE 流正常结束可能带有模型错误或工具参数不完整；HTTP 连接断开也可能发生在服务已生成全部 token 之后。报告应分别记录网络完成、应用 finish reason、输出长度和错误。不能只用 HTTP 200 判断整次生成成功，因为错误可能在响应头发出后才出现在流中。

## 12. 第十二课：把输出管线做成可审阅的小工程

第一阶段保留 sampling.py 作为参考，独立实现一个同契约采样器。先通过手算例子、平局、mask、非法输入与温度零不消费 RNG，再增加随机频率检验。频率检验只能说明大量样本近似分布，不能替代精确边界测试；样本数量有限时自然有波动，验收不应要求每个候选出现次数恰好等于理论概率乘样本数。

第二阶段把 StopFilter 与离线 oracle 结合。对固定短文本枚举全部分片组合，覆盖重叠 stop、同起点长短前缀、空片段和正常 EOF，再比较最终正文与停止标志。额外记录每次 feed 已发出的内容，断言它始终是最终正文的前缀，不能先发错再靠最后拼接“抵消”。练习十四给出了完整参考；进一步可在二字符字母表上穷举短文本和短 stop 集合，检查尾部长度与停止后忽略输入。这个前缀性质才是流式停止正确性的重要部分。

第三阶段组合增量 UTF-8、SSEDecoder、JSON 解析和正文提取。每层只处理自己的契约，并保留分层错误：编码错误、事件截断、JSON 错误、字段缺失、应用错误事件。不要用一个宽泛 except 把所有错误变成空片段，否则你会把真实故障当成模型没有输出。合成 fixture 足以覆盖这些路径，无需调用外部 API。

第四阶段接入有界队列和取消。给生产者、消费者各设置明确终止方式；模拟消费者提前退出，证明生产者不再永久阻塞，请求资源最终释放。再模拟生产者出错，证明消费者收到错误或被取消，而不是等待永远不会出现的结束标记。验收记录队列最大长度和任务是否全部终止，不把 sleep 的毫秒数当成真实服务性能。

第五阶段才对你自己启动的真实模型端点做小规模验证。使用显式参数，最多四个请求、每请求最多三十二个输出 token，设置连接、读取和总时限；记录 model ID、模板、采样参数、是否收到 usage，以及测量的是文本片段还是 token。这里不自动调用用户 API，也不假定 mock_server 支持流式聊天。S15 会给出硬件实验的具体资源边界。

工程最终交付应包含一张从 logits 到 socket 的状态表、一组可重复合成输入、完整参考实现、失败场景及预期清理行为。你应能指出每个缓冲区保存的是字节、文本、事件还是 token，谁可以修改它，什么时候释放，以及错误如何传播。做到这些之后，输出管线才不再是一段“收到就 print”的脚本，而是可以支持真实 Agent 与模型服务的组件。

掌握门槛：能手算温度与截断后的概率；能解释 seed、随机流和数值确定性的区别；能实现不依赖网络切片边界的解码与停止；能证明慢客户端不会导致无限队列或孤儿任务；能明确区分 token、字符、事件与网络 chunk。下一章进入 [S08 指标与实验设计](08-measurement.md)，把这些清楚的边界变成可信的测量。
