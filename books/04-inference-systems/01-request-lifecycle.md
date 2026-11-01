# S01 一次请求怎样穿过推理系统：从一个字符串到可释放的运行状态

本章不把“请求进入框架，然后模型生成答案”当成解释。我们要把这句话拆到能够自己写程序的程度：输入首先是什么对象，哪个函数检查它，什么时候变成整数，谁把它放进队列，哪一轮计算产生第一个新 token，哪些状态必须保留到下一轮，客户端收到文字后为什么服务器可能仍在工作，以及任务结束后哪些资源归还、哪些资源可以继续缓存。

你只需要先掌握基础卷中的列表、字典、函数、异常与简单类，并知道矩阵乘法把输入向量变成输出向量。还不熟悉 HTTP、JSON 或 token 的读者，可以在本章第一、二课补齐本次需要的含义；更完整的网络操作在 F05。这里没有假设你已经运行过 vLLM，也不会要求你先买 GPU。所有手工追踪都能在纸上完成，参考程序在 CPU 上运行。

本章主实验是：

```bash
python -m labs.serving.experiments --lab lifecycle
python -m unittest tests.test_serving_core -v
```

运行命令前先完成第六课的手算。实验不是让你看一段漂亮输出，而是让你拿预测与真实事件逐项核对。你最后要能不看书回答：一个请求为什么同时有 prompt、output、context、computed 四种长度？第一个输出 token 为什么不必等一次单独的 decode？为什么“收到取消”与“GPU 不再读取这个请求的内存”不是同一个时刻？

## 第一课：把容易混在一起的四件东西分开

### 1.1 用户任务、网络请求、推理请求、计算工作

用户任务是一个目标，例如“解释这段代码”。网络请求是为了传递该目标而发送的一份协议消息。推理请求是服务器内部用于跟踪模型生成过程的对象。一次计算工作则是某一轮真正送给模型处理的一小段 token。它们有联系，但数量不必相等。

例如一个聊天客户端遇到超时后重试，可能为同一个用户任务发送两次 HTTP 请求。服务器如果没有识别重试，会建立两个内部推理请求，各自计算一遍。另一方面，一个内部推理请求若包含很长的输入，调度器可能把它拆成十段 prefill 工作；随后输出二十个 token，又需要若干轮 decode。于是一个用户看到的一次提问，可以对应多个网络尝试、多个内部对象和几十份模型工作。

这个区分影响日志设计。只有“用户输入是什么”的日志，无法判断重复计算是否来自客户端重试；只有“第几轮 batch”的日志，又无法还原哪一轮属于哪位用户。最小关联关系应当至少保留请求身份和工作所属身份。更完整的生产系统可能另有会话身份、追踪身份、尝试次数和上游调用身份；不能因为这些值都长得像字符串，就把它们当成同一个字段。

为了手算，本卷给每个内部请求一个短名字，如 A、B。A 不代表 token，不代表某张 GPU，也不代表某次 HTTP 连接。A 只表示“这一份需要持续更新的生成状态”。后来看到 `Work(rid="A", start=0, ...)`，你应立即读成“这份工作属于请求 A，并从它的第零个上下文位置开始”，而不是“启动 A 号进程”。

### 1.2 模型、推理框架、API 服务与客户端

模型定义计算规则和参数。给定 token 序列，它计算下一位置各词表项的分数。推理框架负责让许多请求共享计算和内存资源，同时维护每个请求的独立状态。API 服务负责接收网络输入、验证协议、把结果编码回网络。客户端负责组织问题、发送请求、读取响应和向用户展示。

这些职责可以存在于同一进程，也可以分散在多个进程或机器。进程边界不是职责定义本身。一个 Python 文件既能收 HTTP 又能跑矩阵乘法，仍然同时承担两种职责；把文件拆成两个服务也不会自动解决状态一致性。学习时先分清职责，再研究它们如何部署，能避免“只要多启动一个服务就算架构更合理”的误解。

模型并不知道浏览器里有没有打字动画。它通常也不直接接触 HTTP header。模型输出的是数值张量，采样器把张量转成 token，解码器再把 token 转成文字，网络层把文字包装成响应事件。每一步都有可能出错。因此看到网页空白，不能立刻断定模型没有算出结果；可能是采样成功而流式协议解析失败。

同样，API 返回成功状态也不自动等于用户任务成功。HTTP 成功可以只表示服务器接受并完成了一次协议请求，模型答案仍可能不正确。S 卷主要研究推理系统是否按约定计算、保存、传递和释放状态；答案是否满足复杂业务目标，需要与 Agent 卷的任务验收区分。

### 1.3 控制流与数据流

数据流回答“哪些值从哪里来到哪里去”。例如用户文本变成 token ID，token ID 查 embedding，hidden state 经过层，logits 进入采样器。控制流回答“什么时候做、谁先做、是否继续、失败怎么办”。例如请求先等待、再被选中、预算不足时暂停、输出足够时结束。

同一个模型数据流可以被不同控制流驱动。一个最简单的脚本把 A 全部生成完再处理 B；连续批处理系统可以在相邻轮次交替推进 A 与 B。只要缓存、位置、随机数和停止条件管理正确，它们仍可以计算相同的请求语义。反过来，模型代码完全没改，调度器若误把 B 的缓存交给 A，结果就已经错误。

你可以把数据流当成运算图，把控制流当成决定何时推进哪些运算的状态机。这只是帮助建立直觉的区分；真实实现里两者紧密交互。调度器必须知道请求还缺多少 token 的计算，runner 必须知道调度器安排了哪些位置，页管理器又必须知道这些位置需要多少存储。

## 第二课：从文字、字节、JSON 到模型 token

### 2.1 字符串并不是网络线上直接传递的对象

在 Python 中，`"你好"` 是字符串对象；网络传输的是字节。UTF-8 是把字符编码成字节的一种约定。一个汉字通常不止一个字节，因此一次网络读取拿到的字节块可能停在一个字符的中间。客户端不能假设每次读取到的块都能独立解码成完整文字。

先看一个不用网络的例子：

```python
text = "你好"
raw = text.encode("utf-8")
assert len(text) == 2
assert len(raw) == 6
assert raw.decode("utf-8") == text
```

第一行建立两个字符的字符串。第二行进行编码，得到字节序列。第三、四行比较的是不同计量单位：字符数和字节数。最后一行验证编码与解码在这份完整数据上往返一致。这里的六个字节也不是六个模型 token；tokenizer 有自己的切分规则。把字符、字节、token 三个长度当成一样，会同时破坏输入限制、网络缓存限制和显存估算。

在生产排障中，一个“输入太长”错误必须追问：限制的是 HTTP body 的字节数、消息文本的字符数，还是模型上下文的 token 数？这三个限制可以同时存在，阈值也不同。错误报告如果只说“超过长度”，读者无法决定应该压缩上传文件、缩短文本，还是更换上下文配置。

### 2.2 JSON 是序列化格式，不是 Python 字典本身

客户端可能在 Python 中建立一个字典，然后把它序列化为 JSON 文本，再编码成字节发送。服务器收到字节，先按协议读完消息体，再解析 JSON，得到本语言中的数据对象。两个进程没有直接共享同一个 Python 字典。

```python
import json

payload = {"model": "teaching-model", "prompt": "你好", "max_tokens": 3}
wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
restored = json.loads(wire.decode("utf-8"))
assert restored == payload
assert restored is not payload
```

`dumps` 的返回值是文本，`encode` 的返回值才是字节。`loads` 从文本中重建值。两个字典内容相等，不是同一对象。序列化边界非常重要：Python 里的函数、文件句柄、随机数生成器等对象不能仅靠普通 JSON 自动传给另一个进程。服务器必须明确知道哪些字段属于外部协议，哪些属于内部运行状态。

接着考虑 `"max_tokens": "3"`。JSON 合法不表示业务类型合法：这里的值是字符串。服务器可以选择明确转换，也可以选择拒绝，但不能在各处用不同规则悄悄处理。本课程内部 `max_new_tokens` 要求真正的非负整数，连 Python 中可参与整数运算的 `True` 也拒绝。这样的约束让边界行为可预测，避免布尔开关意外变成一个 token 的预算。

### 2.3 messages、聊天模板与 prompt token

聊天接口经常传递多条带角色的消息，例如 system、user、assistant、tool。模型最终并不是直接读取一个 Python 消息列表；服务端通常需要用模型约定的聊天模板把结构化消息转换为输入序列，再做 tokenization。模板可能插入角色标记、分隔符、结束标记或其他控制 token。

所以“用户文本一样”并不总意味着模型输入 token 一样。system 消息、模板版本、是否追加 assistant 起始标记、工具定义的编码方式，都可能改变实际输入。前缀缓存的匹配必须建立在真正送入模型的序列及其相关语义上，而不是只比较网页输入框里的文字。

本章不把某一模型的聊天模板硬编码为通用规则。你应在具体实验中保存 tokenizer 身份、模板或相关配置，以及格式化后的 token ID。若要比较两个推理框架结果，第一项核对通常不是“都用了相同温度没有”，而是它们实际处理的 token 序列是否一致。两个不同的提示编码产生不同答案，不能作为数值不确定性的证据。

### 2.4 prompt token 与生成 token 的含义

prompt 是开始生成前已知的输入 token 序列。output 是模型在本次请求中已经生成的 token 序列。context 是二者拼接而成的当前上下文。它们不是三份互不相关的数据，而是同一个生成过程的不同观察窗口。

设 prompt 为 `[2, 4, 7]`，模型已经生成 `[8, 9]`，那么 context 是 `[2, 4, 7, 8, 9]`。此时输入长度为三，输出长度为二，上下文长度为五。下一 token 的分布应以这五个 token 为条件；如果下一轮只把原始 prompt 送给一个无状态后端，模型会反复计算第一步分布，生成过程就断了。

本课程 `Request.context` 每次返回一个新列表：`list(self.prompt) + self.output`。这里 prompt 被存成元组，表达它在一次请求内固定不变；output 是会持续增长的列表。返回新列表还能避免调用方直接修改内部 prompt 容器，但它并不意味着整个系统所有状态都是不可变的。

## 第三课：亲手读懂 Request 对象

### 3.1 为什么不用一个字符串贯穿全系统

一个字符串只能告诉我们输入内容，不能告诉我们谁拥有它、何时到达、已经生成多少、缓存算到哪里、是否取消、使用哪套随机数状态。随着任务推进，这些信息都要更新。如果散放在几个没有明确关系的全局字典里，程序很容易把同名键、缺失字段和过期状态混在一起。

`Request` 把一个请求的相关状态集中起来。它不是模型本体，而是运行中的记录。你可以把它理解成一次任务的账本：固定输入写在一栏，已发生结果写在另一栏，资源与进度另列。账本本身不会执行模型；`Engine` 读取、更新这些字段并协调其他组件。

先打开 `labs/serving/engine.py`，阅读 `Request` 的字段定义，而不是直接跳到复杂循环。把字段分成四组：用户配置、输入与身份、运行进度、观察指标。分类不是装饰，它能帮助你判断某个字段应在提交前确定，还是只能由引擎运行时填写。

| 字段 | 本章含义 | 谁应当修改 |
|---|---|---|
| rid | 一次内部请求的稳定身份 | 创建时指定，运行中保持 |
| prompt | 已知输入 token | 创建时指定，运行中保持 |
| max_new_tokens | 本次允许生成的数量上限 | 提交前指定 |
| seed、temperature | 采样配置 | 提交前指定 |
| output | 已生成、已进入上下文的 token | 引擎推进时追加 |
| computed | 本课程已经为上下文前缀建立元数据的长度 | 调度／恢复逻辑 |
| state | pending、waiting、running 或终态 | 生命周期操作 |
| emitted_ms | 每个输出事件对应的模拟时刻 | 输出时追加 |
| finish_ms | 本请求结束的模拟时刻 | 终结路径 |
| rng | 请求自己的随机数生成器 | 采样器消费其状态 |

阅读表格后要回答一个具体问题：`seed` 与 `rng` 为什么同时存在？seed 是初始化参数；rng 是经过若干次抽样后发生变化的状态对象。只保存初始 seed 不等于保存了运行中随机数的位置。恢复请求时，如果从 seed 重新开始又不重放已发生的随机抽样，后续输出可能偏离原来的运行轨迹。

### 3.2 dataclass 与 default_factory 的细节

`@dataclass` 根据字段声明帮助生成初始化等样板代码。它没有替你证明业务逻辑正确，也没有让所有字段自动只读。`field(default_factory=list)` 表示每次建立一个新实例时，调用 `list` 创建新的列表。它的目的不是让列表更快，而是避免多个请求共享同一份默认列表。

假如你把 A 和 B 的 output 指向同一个列表，A 生成 token 后，B 的 context 也会悄悄变长。这个错误可能不会立即抛异常，因为类型和长度看起来都合法。它属于所有权错误：一个请求的可变输出不应归另一个请求共同所有。

你可以运行下面的身份检查，而不仅比较值：

```python
from labs.serving.engine import Request

a = Request("A", (1, 2))
b = Request("B", (3, 4))
assert a.output == b.output == []
assert a.output is not b.output
a.output.append(5)
assert b.output == []
```

两个空列表相等，只说明它们当前的元素序列相同。`is not` 才说明不是同一对象。最后两行制造一次写入，验证修改不会跨请求传播。这是很多系统测试的基本模式：先构造可能共享的状态，再改变其中一份，检查另一份是否受到不应有的影响。

### 3.3 __post_init__ 在保护什么

dataclass 生成初始化流程后会调用 `__post_init__`。本课程把 prompt 转为元组，把传入的输出和时间列表复制为新列表，并验证输入合同。验证 request ID 非空，是为了让日志和字典索引有有效身份；验证 prompt 非空，是因为本课程后端不定义空上下文如何生成。

验证 token 必须为非负整数，还不等于确认 token 在词表范围内。词表大小属于具体 backend，所以 `Engine.submit` 再检查最大 token 是否小于 backend 的 vocab_size。两个检查发生在不同层不是重复无意义劳动：请求对象检查一般形式，具体引擎检查与已选择后端的兼容性。

时间必须有限且非负，温度必须有限且非负，是为了排除 NaN、正无穷或负时间等会污染比较和排序的值。NaN 尤其危险，因为它不服从普通实数的比较直觉；让 NaN 进入等待队列后，排序、到达判断和指标计算都可能出现难解释行为。边界验证应当发生在状态被提交之前。

### 3.4 “新请求”是一种完整契约

`submit` 不接受带着旧 output、旧 finish_ms 或 finished 状态的对象。原因不是“程序不喜欢复用”，而是一次新运行与一次恢复需要不同协议。如果把完成了一半的对象当作新对象提交，页表、随机数、computed 与输出可能彼此矛盾。

恢复功能不能只靠给 Request 填几项旧值来假装完成。真正恢复必须同时恢复输入身份、生成历史、采样状态、缓存状态或者明确重算策略，并处理外部是否已经收到输出。教学引擎目前定义的是新请求提交与进程内抢占，不声称支持任意进程崩溃后的持久恢复。

因此你会看到 `_submitted` 这样的内部标记。它防止同一个对象被重复提交到不同生命周期。对象身份、请求 ID、请求是否提交过，是三个相关但不同的问题：换一个新的对象却沿用已存在的 rid 仍然被拒绝；同一个对象改个 rid 也不应悄悄绕过其运行历史。

## 第四课：状态机不是一串随意的字符串

### 4.1 pending、waiting 与 running 的区别

pending 表示请求已经登记，但在本课程的模拟时间中尚未到达可接收时刻；waiting 表示它已经到达、等待运行名额；running 表示它已被接纳并拥有活动页表。三者都还没有结束，但资源责任不同。

假设 A 在时间零到达，B 在时间五到达，系统时间目前为二。B 可以已经存在于 `requests` 字典中，同时保持 pending。它不应参与当前轮调度，也不应因为“对象已经创建”就提前占用运行序列名额。仿真器这样设计是为了方便预先输入一整份到达轨迹。

waiting 与 running 也不能合并。一个请求已经来到服务端，并不意味着此刻一定有足够的并发名额或资源。waiting 队列表达排队事实；running 集合表达引擎正在管理它的活动计算状态。没有这个区分，你无法准确测量排队时间，也无法解释为什么 GPU 忙时新请求迟迟没有首 token。

### 4.2 终态的意义

finished 表示请求按本课程的停止条件正常结束；cancelled 表示收到并接受取消；error 表示执行中发生了被引擎捕获的错误。它们都应当停止新生成，但不能在报告里混为一个“done”。

一个正常生成零个新 token 的请求也可以 finished。它与模型报错后没有输出完全不同。一个取消前已经输出两个 token 的请求，可以有非空 output 和 cancelled 状态。用户是否把这些部分结果用于业务，需要上层决定；推理层必须忠实记录发生过什么。

终态还意味着某些操作应当幂等。对已结束的请求重复 cancel，不能再次释放同一页，不能把已有 finish_ms 改到新的时间，也不能再增加一个虚假的取消事件。本课程 cancel 在检查到终态后返回 False，表达此次没有产生新的状态改变。

### 4.3 状态转换与资源转换要同步设计

当 waiting 请求进入 running，引擎创建页表、尝试附加可复用前缀、更新 computed、加入 running 列表。只改字符串而不处理页表，会得到“显示在运行，但没有资源入口”的对象。只创建页表却不改变状态，又可能让资源永远不被调度器清理。

同样，finish 路径不仅写 `state="finished"`，还要移除 waiting 或 running 中的身份、释放请求页引用、填写 finish_ms 和 error。把这些动作集中到 `_finish`，可以降低每条终结分支各自漏一步的风险。集中并不自动保证正确，所以仍然需要测试结束后活动表与运行集合一致。

本课程 `check` 包含 `set(pool.tables) == set(running)`。它不是通用框架的普遍定义，而是这个同步教学设计明确采用的不变量：每个运行请求恰好对应一个活动页表，非运行请求不保留活动页表。生产框架如果有异步在途状态，可能还需要 retired 或 deferred-free 集合；那时不变量必须随设计调整，不能机械复制。

## 第五课：prefill 与 decode 到底在算什么

### 5.1 “一次处理很多输入”与“产生很多输出”不是同义词

对一个长度为三的 prompt，模型可以在一次前向中处理三个已知 token。因果注意力保证第零位置不看未来，第一位置只看前两项，第二位置看全部三项。最后位置的 logits 用于预测第一个新 token。处理三个输入位置，并不表示已经一次生成三个未知输出。

原因在于自回归依赖。第二个输出 token 的条件里包含第一个输出 token；在第一个输出尚未选定前，普通自回归解码不能假装已经知道它。本卷先研究没有投机解码的路径：每次确定一个新 token，再把它纳入下一轮上下文。

prefill 是处理一段已知上下文并建立所需状态的阶段；decode 是在已有历史基础上推进生成的阶段。这个定义解释了为何长 prompt 可以分块处理，也解释了为什么重算生成历史时可能仍然在处理已知 token，而不是重新向用户输出那些 token。

### 5.2 一个二维注意力手算

为了看清输入位置与输出位置，考虑单头二维向量。取最后位置 query 为 `q=[1,0]`，两个已有 key 为 `k0=[0,1]`、`k1=[1,0]`，对应 value 为 `v0=[2,0]`、`v1=[0,4]`。不计缩放时，点积得分为零和一；稳定 softmax 会先减去最大值一，得到指数权重 `[exp(-1),1]`。

归一化分母约为一点三六七九，概率约为零点二六八九和零点七三一一。加权输出约为 `[0.5378,2.9244]`。如果再接一个输出投影，它才会得到词表维度的 logits；注意力输出本身不是 token ID。

这个例子中 q 只有当前最后位置的一行，而 K/V 有两行历史。下一步生成新 token 后，只需为新位置计算新的 q/k/v，并把新 K/V 加到历史。过去的 K/V 可以继续复用，因为它们已经由过去 token 和当时的模型状态确定。真正的缓存是否可复用还依赖位置、模型参数和其他语义，S03、S04 会逐步展开。

### 5.3 computed 为什么往往比 context 少一

本课程在一轮工作完成后采样并追加一个新 token。这个 token 已经属于 output，所以也已经属于 context；但它自己的 K/V 尚未经过下一轮计算。于是生成之后常见关系是 `len(context) = computed + 1`。

设 prompt 长度三。prefill 完成时 computed 为三，context 仍长三；采样得到 token 8 后，output 长一，context 长四，computed 仍为三。下一轮处理这个已知的 token 8，computed 变为四，再采样 token 9，context 变为五。这个“最后一个已选 token 尚未处理”的缝隙，是理解 decode 调度最关键的细节之一。

因此最后一次采样满足输出长度上限时，请求可以立即结束，无需再为最后输出 token 建立缓存。因为任务已经不需要预测下一个 token。S01 的 submit 容量预检使用 `prompt_length + max(0, max_new_tokens - 1)`，而不是一律加上全部输出上限；这正对应本教学实现的执行边界，不能脱离具体生成协议随意套用到所有系统。

### 5.4 教学后端与真实缓存后端的边界

当前 `Engine` 的 PagePool 存储 token 元数据，`Backend.logits(context)` 仍然使用完整上下文计算。这样做把“控制状态是否正确”与“数值缓存是否正确”分开，便于新手观察。S19 会另行实现真正存储浮点 K/V 的分页数值缓存。

所以 computed 在 Engine 中表达的是这个教学控制模型已经推进到哪里，而不意味着相同数量的真实 GPU K/V 张量已经存在。本章所有事件时间也来自透明的模拟公式。把模拟器输出的毫秒和吞吐当作你自己的 H100 性能，是错误的证据跨越。

你仍然可以用它验证很多真实问题：请求是否提前输出、重算是否重复发 token、取消后是否继续调度、预算是否超出、页引用是否泄漏。它不能验证具体 GPU kernel 的性能、异步依赖、通信速度和框架安装兼容性。学习时明确每个实验能排除什么错误，比追求一个什么都声称证明的大脚本更可靠。

## 第六课：完整手算一次三输入、三输出请求

我们采用单请求 A，prompt 为 `(1,2,3)`，最多生成三个 token，chunk_size 为二，token_budget 为二，max_sequences 为一，prefix_entries 为零。后端为 BigramBackend，温度为零。它把最后 token 的下一整数作为得分最高的词表项，词表大小为十六，因此本例输出容易预测为四、五、六。

### 6.1 提交之后还没有发生模型计算

`Engine.submit(A)` 验证 A 是新请求、ID 不重复、token 在词表范围内、单请求不会超过整个页池容量。随后把 A 放进 requests，并记录 submitted 事件。此时 state 仍是 pending，computed 为零，output 为空，pool.tables 也还没有 A 的页表。

如果你的观察程序只在 submit 后打印 output，得到空列表是正确的。提交是登记工作，不是同步完成工作。很多服务端接口也会区分“请求已接受”和“结果已完成”，只是可能通过异步 future、队列或流式响应表达这个区别。

### 6.2 第一轮只处理前两个 prompt token

第一次 `step` 先执行 ingest：A 到达时间为零，当前时间也为零，因此 pending 变成 waiting。admit 看到有一个运行名额，为 A 创建空页表，前缀缓存关闭所以 computed 仍为零，然后设为 running。

调度阶段计算 gap 为 `len(context)-computed = 3-0 = 3`。因为 computed 小于 prompt 长度，phase 是 prefill。实际 count 取 gap、剩余总预算和 chunk_size 的最小值，即二。values 是 context 的零到二位置，得到 `[1,2]`。append 成功后 computed 变成二，预算减为零。

这一轮模拟代价为固定一毫秒加上两个 prefill token 各零点二五毫秒，总计一点五毫秒。之后检查 computed 是否等于 context 长度：二不等于三，因此不能采样第一个输出。A 仍在 running，输出仍为空。把“有一轮 batch 完成”误认为“必有一个输出 token”，会在分块 prefill 场景直接出错。

### 6.3 第二轮补完 prompt，产生第一个输出

第二次 step 不需要重新接纳 A。gap 变为 `3-2=1`，phase 仍是 prefill，count 为一。它处理 token 3，computed 从二变成三，模拟时钟增加一点二五毫秒，从一点五来到二点七五。

此时 computed 等于 context 长度三，后端计算当前上下文下一项的分数。BigramBackend 对 token 4 给出最高得分；温度为零时 sample 选中 4。引擎把 4 追加到 output，把当前时间二点七五追加到 emitted_ms。现在 output 长一，context 长四，computed 仍为三。

这说明第一个新 token 是在补完 prompt 的同一轮输出的，不需要额外再安排一次“空 decode”才能产生。实际框架内部可能把 forward 和 sampling 拆成不同函数、不同阶段，但语义上最后 prompt 位置的分数已经足以预测第一个新 token。

### 6.4 第三、四轮推进 decode

第三次 step 时 computed 已不小于 prompt 长度，所以 phase 为 decode。gap 为 `4-3=1`，count 为一，处理刚才生成的 4。模拟时钟增加一点一毫秒，来到三点八五。computed 变为四，随后采样 token 5，context 长度又变为五。

第四次 step 同理处理 5，时钟来到四点九五，computed 变为五，采样 token 6。output 长度达到三，调用 _finish。请求页表被释放，A 从 running 移除，state 变成 finished，finish_ms 记录四点九五。最终 output 是 `[4,5,6]`，最后 token 6 没有再进入一次模型输入计算。

| 时点 | phase/count | computed | output | context长度 | 状态 |
|---|---|---:|---|---:|---|
| 提交 | 尚无工作 | 0 | [] | 3 | pending |
| 第一次工作后 | prefill/2 | 2 | [] | 3 | running |
| 第二次工作后 | prefill/1 | 3 | [4] | 4 | running |
| 第三次工作后 | decode/1 | 4 | [4,5] | 5 | running |
| 第四次工作后 | decode/1 | 5 | [4,5,6] | 6 | finished |

表中的时间只是教学公式推演，不是测量。真正需要你掌握的是每次状态变化的原因。先把 output 那一列遮住，只看 computed 和 context 长度，判断哪一轮有资格采样；再把 computed 那一列遮住，根据处理的 token 位置恢复它。这种双向推导比背表更能检验理解。

### 6.5 在代码中验证手算

```python
from labs.serving.engine import Engine, Request

engine = Engine(token_budget=2, chunk_size=2, max_sequences=1,
                pages=8, block_size=2, prefix_entries=0)
request = Request("A", (1, 2, 3), max_new_tokens=3, temperature=0.0)
engine.submit(request)

while engine.step():
    print(engine.iteration, request.state, request.computed,
          list(request.output), engine.time_ms)

assert request.output == [4, 5, 6]
assert request.state == "finished"
assert request.computed == 5
assert not engine.running
assert "A" not in engine.pool.tables
assert len(engine.pool.free) == 8
```

`while engine.step()` 会在最后一个完成工作的 step 返回 True 后再调用一次 step；当没有等待、运行或未来请求时返回 False。这个额外的“检查还有没有工作”调用不是又生成了一个 token。读循环时必须区分一次函数调用与一次有实际工作内容的迭代。

## 第七课：把 Engine.step 拆成可以逐行解释的阶段

### 7.1 到达与接纳

`_ingest` 按到达时间和 rid 对新到达请求排序，避免相同时间下的遍历顺序不明确。它处理 max_new_tokens 等于零的特殊情况，直接正常结束请求。其他请求转成 waiting 并加入双端队列。

`_admit` 在 waiting 非空且 running 数量低于上限时取出队首。`popleft` 表示从左端移除，符合队列先进先出的基础语义。这里的“接纳顺序”与后面的“每轮计算顺序”是两个不同决定：你可以先按到达顺序接纳，再让已接纳请求轮转推进。

接纳时创建页表，再附加已有前缀。附加返回可复用的 token 数，赋给 computed。若命中四个 prompt token，就表示前四个位置的相关状态可以复用，后续调度从位置四开始。若你只更新 computed 而没有附加实际页引用，计数看起来前进了，资源却没有对应数据，这类缺陷在 S19 用数值测试检查。

### 7.2 调度计划是一份明确的合同

`plan` 是本轮 Work 对象的列表。每份 Work 保存 rid、start、token_ids、phase。start 是请求内部位置，不是物理页号，也不是本轮扁平 batch 的偏移。token_ids 是这次实际处理的 token，不是整个请求历史。phase 是本教学框架用于区分 prefill 与 decode 的标签。

冻结的 dataclass Work 表达“计划建立后不再随手改字段”的意图。它能防止 runner 读取计划时，另一个代码分支悄悄把 start 或 phase 改掉。但内部元素若是可变容器，冻结字段也不一定阻止深层修改；本课程使用 token 元组，使这份工作记录更容易保持稳定。

生产框架的计划通常还包含 block table、position、采样元数据、结构化输出约束、已完成和被抢占请求列表等。不要因为教学 Work 只有四项，就认为真实 runner 可以只靠四个字段完成全部工作。我们先固定最少的合同，再逐层扩展，是为了让每个新增字段都有明确必要性。

### 7.3 先检查资源，再修改进度

调度器用 `pool.needed(rid,count)` 计算本次追加至少需要多少自由页。如果不够，先尝试淘汰可淘汰的前缀缓存。如果仍然不足，这个请求本轮暂不形成工作。只有 append 成功后才增加 computed 并扣减 token 预算。

这个顺序保护了一个重要关系：computed 所声称的上下文前缀，应当能在页表中找到对应元数据。先增加 computed、再尝试分配，分配失败后就必须精确回滚；否则调度器下一轮会跳过未处理的 token。把容易失败的检查尽量放在状态改变前，可以让代码更容易证明正确。

不过，真实 GPU 执行仍可能在分配和计划建立之后失败。生产系统因此需要进一步区分 scheduled、in-flight、completed 等进度，而不能把“计划已送出”当作“设备已成功完成”。当前同步教学模型有意压缩了这部分状态；S05、S06 和 S17 会解释扩展时必须新增哪些责任。

### 7.4 完成计算与产生输出

遍历 plan 后，引擎知道本轮有哪些 prefill 和 decode 工作，推进模拟时钟，再逐请求处理结果。只有 `r.computed == len(r.context)` 时，才允许对这个当前完整上下文采样。这个条件不仅处理分块 prompt，也处理抢占之后重新计算已知生成历史。

如果后端抛异常，引擎把请求终结为 error，并继续处理同一批里的其他请求。这里选择的是每请求错误隔离，而不是整个 batch 直接失控退出。实际 GPU kernel 错误是否能局部隔离，要依错误类型和运行时状态决定；不能把 Python 函数异常隔离的结论直接扩展到所有设备错误。

采样成功后，output 与 emitted_ms 一起追加。它们长度必须相等，因为每个已发出的 token 应有对应的发出时刻。若只追加 output 却漏掉时间，指标报告会低估或无法计算间隔；若时间多于 token，则记录了不存在的输出事件。

## 第八课：结果出来以后还要做什么

### 8.1 token、文字与流式事件并不一一对应

一个 token 可能解码为空白、一个单词片段或多个字符。多个 token 也可能共同组成完整字符或需要解码器结合上下文处理的片段。网络层的一次事件可以包含一个或多个增量片段，一次 TCP 读取又可能包含多个事件或半个事件。

因此不能用“收到几次网络读取”当作输出 token 数。也不能用“网页刷新了几次”推算 decode 迭代数。正确的统计需要使用明确层级的计数：模型输出 token、协议事件、字节读取、用户界面更新分别记录。

本课程 Engine 直接输出 token ID，streaming.py 另行验证 UTF-8 与 SSE 边界。这种拆分让你能独立检查两类错误：一类是模型生成错了 token，另一类是正确 token 对应的文字在网络编码中被截断。把它们混成一个端到端字符串比较，虽然能发现失败，却往往无法定位原因。

### 8.2 正常停止、EOS 与长度停止

本教学引擎按 max_new_tokens 结束，没有实现所有生产接口的 EOS、stop string 和结构化约束。它提供一个最清楚的起点：生成计数达到上限就停止。进入真实框架时，你要列出所有可能的结束原因，并确认它们如何编码到最终响应。

EOS 是模型词表中的特殊 token 语义；长度停止是外部预算；stop string 是解码后的文本匹配规则。三者作用层级不同。stop string 还可能跨越两个流式片段，因此仅检查当前片段是否包含停止字符串并不完整。S07 会给出具体跨块例子和缓冲策略。

正常结束后是否把终止标记展示给用户，也由协议和 tokenizer 规则决定。不能在没有看配置的情况下断言“生成了 EOS 就一定多显示一个字符”。推理系统工程需要明确区分内部 token 序列和用户可见文本。

### 8.3 释放请求不等于清空一切缓存

请求结束时，活动页引用需要释放。但如果前缀缓存仍持有某些页，它们可以继续存在，用于未来请求。此时 free_pages 小于总页数不一定意味着内存泄漏。判断泄漏必须看所有权账：每个未释放页是否仍有合法拥有者，拥有者是否还可能使用它。

本章例子关闭 prefix cache，所以最后所有页归还。这个设置是为了建立一个容易核对的基线。之后开启缓存，再把“请求已结束但缓存页仍被固定”的现象作为有意设计观察，才不会把正确的复用误判为泄漏。

真实 GPU 分配器也可能保留已经空闲的显存块供以后复用，因此外部工具显示进程仍占显存，并不等于每个请求的逻辑缓存还活着。逻辑资源释放、框架缓存池释放、设备分配器归还内存、操作系统观察到的占用，是不同层次的问题。

## 第九课：三类典型故障怎样追到具体责任

### 9.1 请求接受成功，却没有首 token

先区分它停在哪里。如果 state 仍是 pending，检查到达时间是否在未来；如果是 waiting，检查运行名额和前面请求是否释放；如果 running 且 computed 持续增加，可能只是长 prompt 尚未处理完；如果 computed 不增加，检查资源分配和计划是否为空。

不要只打印 output。一个有用快照至少包含 state、prompt 长度、context 长度、computed、waiting/running、free_pages 和最近一条 batch 事件。你可以据此把“没有输出”拆成“尚未开始”“正在完成已知输入”“不能分配资源”“模型执行失败但错误未传出”等不同问题。

构造一个长度为五、chunk_size 为二的 prompt，可以稳定演示前两轮没有首 token。把这种正常等待与程序死锁分开，是建立诊断直觉的第一步。真正的死锁需要说明所有可前进条件为何互相等待，而不是仅凭“等了几秒”判断。

### 9.2 输出重复，或者抢占后从头再来

检查 output 是否在恢复时被清空，rng 是否被重新初始化，computed 是否从零重算时又把每个已知位置当成新生成位置。重算已知历史的目标是恢复状态，不是再次向用户宣布同样的 token。

在本教学引擎中，输出历史被保留；computed 可以回到零。随着重算推进，只有追上当前 context 全长度时才产生一个新输出。这个门槛让“恢复缓存”与“扩展上下文”有清楚边界。删掉它，改成每完成一小块就 sample，会立刻改变生成序列。

诊断时保存两份序列：内部 output 与客户端实际收到的增量。内部没有重复而客户端重复，可能是网络重试或事件消费问题；内部本身重复推进，才需要追调度和恢复。两个层级都只保留最终拼接文本，会使这类问题难以拆分。

### 9.3 请求结束了，页数却回不来

先关闭前缀缓存做基线，再检查 _finish 是否经过统一释放路径。异常分支、取消分支、零输出预算分支都应纳入测试，不能只检查普通长度结束。很多资源泄漏只存在于冷门终结路径。

接着核对页引用总数，而不是仅看 free 列表长度。如果同一物理页被两个请求共享，结束一个请求只应该减少一个引用。若直接把页塞回 free，会产生使用后释放风险；若忘记减少引用，则页无法归还。S03 会把这两种错误写成可以手算的所有权变化。

最后检查报告对象是否只是观察快照。当前 report 对 outputs 使用 `list(r.output)` 复制，避免调用方修改报告时反向改变引擎内部输出。这是一个容易忽略的接口边界：读报告不应该授予修改正在运行请求的能力。

## 第十课：从空白文件写一个有界生命周期练习

下面的程序不实现分页与真实模型，它只让你亲手掌握“已知输入处理到哪里、何时允许生成、何时结束”的控制核心。它使用后端函数把最后一个整数加一。先复制到自己的练习文件运行，再关闭本书独立重写。

```python
from dataclasses import dataclass, field

@dataclass
class MiniRequest:
    prompt: tuple[int, ...]
    limit: int
    computed: int = 0
    output: list[int] = field(default_factory=list)
    state: str = "running"

    @property
    def context(self):
        return list(self.prompt) + self.output

def advance(r: MiniRequest, chunk: int):
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    if r.state != "running":
        return []
    if r.limit == 0:
        r.state = "finished"
        return []
    context = r.context
    gap = len(context) - r.computed
    count = min(gap, chunk if r.computed < len(r.prompt) else 1)
    if count <= 0:
        raise RuntimeError("no progress")
    work = context[r.computed:r.computed + count]
    r.computed += count
    if r.computed == len(r.context):
        r.output.append((r.context[-1] + 1) % 16)
        if len(r.output) >= r.limit:
            r.state = "finished"
    return work

r = MiniRequest((1, 2, 3), 3)
for _ in range(20):
    if r.state == "finished":
        break
    print(advance(r, 2), r.computed, r.output, r.state)
else:
    raise RuntimeError("iteration budget exceeded")
assert r.output == [4, 5, 6]
```

`advance` 先拒绝非法 chunk，再判断请求是否仍活动。把终态判断放在前面，意味着重复调用不会继续输出。零预算分支在任何 token 工作之前结束，避免无意义计算。接着从 context 长度减去 computed，得到尚未处理的位置数量；这个差值不是剩余可生成数量。

`count` 在 prefill 阶段可以多于一，在 decode 阶段限制为一。工作切片取的是旧 computed 到旧 computed 加 count 的位置，然后才更新 computed。顺序不能颠倒，否则切片会跳过该处理的位置。更新后再次读取 r.context，在尚未生成时它与旧 context 相同。

采样之后 context 增长，但 computed 不随之增长。这一行没有出现“自动补一”，正是我们希望保留的状态差。最后用一个有上限的 for 循环驱动它；如果没有在预算内结束，明确报错，而不是留下无法解释的无限循环。

这个小程序缺少完整输入验证、到达队列、资源管理、异常隔离与指标，因此不能当作生产服务。它作为学习台阶的价值在于：只用二十多行核心控制逻辑，就能独立观察本章最重要的状态不变量。下一步给它增加队列时，你应保留已经通过的单请求测试，而不是全部推翻重写。

## 第十一课：十二道分级练习与完整参考判断

### 练习一：区分四种数量

一个用户任务首次请求超时，客户端重试一次；服务端两次都接收并各建立一个推理请求。每个请求把六 token prompt 分成两块，再生成四 token，普通 decode 每次一个位置。问用户任务、HTTP 尝试、内部请求各有多少，是否能把其中任何一个直接叫“步数”？

参考答案：用户任务一个，HTTP 尝试两个，内部请求两个。每个请求至少有两次 prefill 工作，之后为后续三个输出推进三次 decode，因此在这个简化配置下有五次有模型工作的推进。第一个输出来自最后 prefill。步数必须说明计量单位；如果把一次无工作结束检查也计作函数调用，调用次数又不同。

### 练习二：字符、字节、token

文本“你好”有几个字符、几个 UTF-8 字节、几个模型 token？参考答案：本例两个字符、六个 UTF-8 字节；token 数不能仅从这两个值推出，必须选择具体 tokenizer 并实际编码。正确程序可以断言前两个值，不能把未指定 tokenizer 的 token 数写成固定常量。

### 练习三：预测下一次工作切片

prompt 为 `(2,3,4,5)`，output 为 `[6,7]`，computed 为三，chunk_size 为二。参考答案：context 为六项，phase 仍是 prefill，因为三小于原 prompt 长度四。count 为二时会处理 `[5,6]`；本教学引擎按起点判断 phase，不在 prompt 边界强行再次切段。处理后 computed 为五，尚未追上 context 六，所以不采样。这个例子提醒你，phase 标签的具体含义来自实现，不能凭自然语言想当然。

### 练习四：最后输出为什么不必进入缓存

prompt 长二，输出上限三。正常无中断生成时，最多有多少个上下文位置需要在本教学控制模型中被处理？参考答案是四：两个 prompt，加上前两个输出。第三个输出一旦被选出就触发结束，不再需要用它预测下一项。若任务改为继续生成第四个输出，才需要处理第三个输出的位置。

### 练习五：零输出预算

创建 `Request("zero",(1,),max_new_tokens=0)` 并运行引擎。应输出空列表、状态 finished、没有 token 事件。参考代码如下；不要把零预算当错误，也不要偷偷生成一个 token 后再截掉。

```python
from labs.serving.engine import Engine, Request
e = Engine(prefix_entries=0)
r = Request("zero", (1,), max_new_tokens=0)
e.submit(r)
e.run()
assert r.state == "finished" and r.output == []
assert not any(x["kind"] == "token" for x in e.events)
```

### 练习六：重复提交与新对象

先提交 A，再把同一个对象提交一次；再建立新对象但使用同一个 rid。两次都应被拒绝，但理由不同：前者不是全新运行状态，后者身份已存在。完整参考验证可以分别用 try/except 捕获 ValueError，并在每次失败后确认 requests 仍只有一个对象。不要只验证“抛异常”，还要检查没有留下部分写入。

### 练习七：列表别名

写出两个 Request 的 output 列表身份断言，向其中一个列表写入后检查另一个不变。参考代码在第三课已给出。进一步说明：虽然这样测试人工修改了运行状态，它是在构造所有权反例；正常使用应由引擎追加输出，不应在运行中从外部随意改 Request.output。

### 练习八：从日志定位等待

给出一个快照：state 为 running、prompt 长八、computed 为四、output 为空、最近事件是 prefill count 二。能否判断死锁？参考答案：不能；这与正常分块 prefill 完全相容。需要观察后续 computed 是否持续推进、是否出现 allocation_blocked、模型工作是否完成。单个瞬时快照不足以证明长期无进展。

### 练习九：取消幂等

运行 A 一步，调用两次 cancel。第一次返回 True，第二次返回 False；第二次不能改变 finish_ms，不能增加一个新的 cancelled 事件。参考实现应保存第一次后的 finish_ms 和事件数，再比较。这样比只断言最终状态 cancelled 更强，因为它能发现重复释放和重复记账。

```python
from labs.serving.engine import Engine, Request
e = Engine(prefix_entries=0, chunk_size=1)
e.submit(Request("A", (1, 2, 3), max_new_tokens=2))
e.step()
assert e.cancel("A") is True
t = e.requests["A"].finish_ms
n = len(e.events)
assert e.cancel("A") is False
assert e.requests["A"].finish_ms == t
assert len(e.events) == n
```

### 练习十：报告能不能修改引擎

运行一个请求后取 report，向 report 中某请求的 output 列表追加一个数字，检查引擎原始 output 不变。参考答案：当前实现返回输出列表副本，所以应不变。如果把 report 中的 `list(r.output)` 改成 `r.output`，测试应失败。这是一项有明确负向控制的接口测试，而不是永远成功的展示。

### 练习十一：构造提前采样错误

把 MiniRequest 练习中的完整上下文门槛改成“只要本轮处理了任意 token 就采样”，使用长度五、chunk 二的输入。参考答案：第一轮就会在 prompt 尚未处理完时追加输出，输出时机和分布条件都违反原合同。你应记录第一处分歧出现在 computed 为二、prompt 长五的位置，而不是只说最后答案不同。

### 练习十二：独立交付生命周期观察器

编写一个函数接收 Engine，每次 step 后输出请求身份、状态、prompt 长度、output 长度、context 长度、computed、自由页数。它必须只观察，不修改运行状态。参考设计是把这些值复制到新字典列表，再打印或保存 JSON；不要把内部列表直接交给外部消费者。

验收要求包括一个普通请求、一个零预算请求、一个未来到达请求、一个取消请求。普通请求的 output 逐次增长；零预算没有 token 事件；未来请求不会提前接纳；取消之后不再新增输出。最后每个请求都有可解释的终态，活动页表与 running 集合一致。完成这些检查，才算把“生命周期”落实成可检验的软件行为。

## 第十二课：把同一生命周期放到本地 HTTP 服务里

本章前面的程序在一个 Python 进程里直接调用 Engine。现在增加一个真正的协议边界，观察哪些部分必须改变、哪些状态可以保持。课程提供的 `mock_server.py` 仅是本地教学服务，接收已经编码好的 token ID，并没有实现完整 OpenAI 接口、聊天模板或真实 tokenizer。使用这个有限接口，是为了让你可以把网络问题与模型分词问题分开实验。

在一个终端运行：

```bash
python -m labs.serving.mock_server --port 18081
```

它只绑定本机回环地址。另开同一台机器的终端，用 Python 标准库发送一份 JSON；下面的请求没有任何付费模型调用：

```python
import json
from urllib.request import Request, urlopen

payload = {"prompt_token_ids": [1, 2, 3], "max_tokens": 3, "seed": 0}
wire = json.dumps(payload).encode("utf-8")
request = Request(
    "http://127.0.0.1:18081/v1/completions",
    data=wire,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=5) as response:
    result = json.load(response)
print(result)
assert result["token_ids"] == [4, 5, 6]
assert result["time_kind"] == "synthetic_ms"
```

这里导入的 Request 来自 urllib，表示客户端的 HTTP 请求对象，与本卷 engine.Request 不是同一个类。同一个英文名字在不同模块里可以承担完全不同的职责，所以阅读源码必须同时看导入路径。客户端对象携带 URL、方法、header 和字节 body；引擎对象携带 token、生成状态和缓存进度。

`urlopen` 返回一个响应对象，`with` 在退出作用域时关闭对应资源。timeout 给网络操作一个上限，避免服务没有启动时无限等待。`json.load` 从响应流读取 JSON；它与 `json.loads` 接收一个已经存在的字符串有所不同。不要仅凭函数名相似就随意互换参数。

回到服务端，`do_POST` 先检查路径，再读取 Content-Length，然后按声明长度读取 body。这里限制 body 大小，是网络输入边界；`validate_payload` 再限制 prompt token 数和最大输出数，是业务边界。两种限制互补：一个巨大 JSON 字段即使 token 列表不长，也可能先耗掉大量解析内存。

`validate_payload` 用 `uuid.uuid4().hex` 产生内部请求身份，再构造 engine.Request。客户端本例没有直接控制内部 rid，因此重试会产生新的内部请求；这也印证第一课中“一个用户任务可能变成两个推理请求”的问题。如果要实现业务重试去重，应另外设计可验证的幂等键与结果缓存，而不是把随机身份误称为自动幂等。

服务端使用一把锁保护 Engine。锁的含义是同一时刻只允许一段受保护代码修改这个引擎状态。虽然 HTTPServer 可以为不同连接启动处理线程，但核心 e.submit、e.run 以及计数更新被串行化。于是“网络层能同时接受多个线程”不等于“这些请求已经进入连续批处理”。这是本地服务明确保留的教学限制。

你可以构造两个并发客户端，观察它们都能得到结果，但不能据此宣称测得了真实批处理吞吐。要让多个网络请求在同一个连续批处理循环里交错，通常需要独立的接收队列与引擎循环：网络线程负责登记并等待对应结果，引擎循环集中拥有调度状态。S16 会把这项修改拆成项目里程碑。

请求结束后，服务把输出 token、指标和时间类型复制到响应 payload，然后删除这份终态请求的历史记录并清空教学事件列表。这是为了让演示服务长时间运行时不无限积累历史。注意它没有每次关闭整个引擎，也没有清空所有前缀缓存；服务级资源与请求级资源仍然分开。

现在故意发送 `"max_tokens": "3"`。JSON 解析仍会成功，但业务验证会返回错误响应。再把 URL 路径改错，会走找不到路径的分支。再完全停止服务，客户端会得到连接层错误。三个现象都可能被用户概括成“请求失败”，但它们发生在三个不同层：协议路由、业务输入验证、连接建立。诊断必须保留这些层次。

最后读 `/health` 与 `/metrics` 的差别。health 只说明这个教学前端还能响应，并不证明某个复杂模型已经加载成功；metrics 返回请求计数、活动数和自由页数，是状态观察。真实生产系统还可能区分存活检查与就绪检查：进程活着不代表已经具备接受模型请求的能力。本服务没有模拟这些完整状态，因此不要把它的 health 规则直接当作大型部署的标准。

完成实验后，在服务终端按 Ctrl+C 让服务退出。你应该能用自己的话说明：哪些代码属于 HTTP 适配，哪些属于输入合同，哪些属于 Engine 生命周期，为什么锁的存在改变了并发语义，以及模拟毫秒为什么即使通过真实网络返回也仍然不是 GPU 实测毫秒。

## 第十三课：进入后续章节之前的自检

关掉书，从空白纸上画出 bytes、JSON、messages/template、token IDs、Request、waiting、running、Work、backend logits、sampling、output、decoder、stream 这条链。不是要求背出唯一目录结构，而是要求说清每一个箭头转换了什么，以及错误由哪层报告。

然后在源码中为 Request、Work、Engine 各写一句准确职责：Request 保存一次生成的配置与可变状态；Work 描述本轮分配给某请求的输入位置；Engine 组织到达、接纳、调度、结果处理与终结。若你的描述只剩“请求类”“工作类”“引擎类”，说明尚未理解职责，只记住了名字。

本章完成后，S02 会研究多个请求如何公平地共享一轮预算；S03 会把页管理从名字展开为地址翻译与引用计数；S06 会解释 Work 怎样变成真实张量与 kernel 输入。你已经具备辨认这些模块输入输出的基础，不必在每看到一个新类时从头猜测它在系统里扮演什么角色。

本章来源边界：运行逻辑以本仓库 `labs/serving/engine.py`、`backend.py` 为直接依据；真实框架的分层与调用链使用本卷 [固定来源](SOURCES.md) 中的 EngineCore、Scheduler 和 ModelRunner 快照。手算时间均为本课程模拟公式，既不来自论文基准，也不代表 RTX 或 H100 实测。
