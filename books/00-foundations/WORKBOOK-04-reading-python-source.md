# 工作簿 D：读懂真实Python源码的十次逐行演练

本工作簿专门解决“概念知道，但打开工程看不懂”的问题。你不需要从Python语言规范第一页读到最后一页，而是沿一条真实调用链，逐个解释控制流、数据流、所有权和失败路径。每次演练都对应本仓库已经能运行的代码。

准备一张四列表：位置、输入、状态变化、输出/异常。读到不认识的语法，先写出它在此处的职责；如果它影响主线，再查定义。不要被每个陌生导入都拖进无限阅读。先理解一次具体输入怎样走完，再扩大到其他路径。

## D1 从入口逆向找到真正业务函数

打开`labs/foundations/cli.py`，从底部开始看`if __name__ == '__main__'`。运行模块时，这个条件成立；导入模块时通常不成立。里面调用main，并把返回值交给SystemExit。

main第一行创建解析器并parse_args，得到Namespace；下一行把args.count和args.batch_size交给range与batches；再json.dumps打印结果；最后return0。真正分批算法并不在这个文件，而是从.functions导入。

因此阅读路径应为入口→main→parser合同→batches→返回表示。没有必要先把json标准库源码通读一遍，因为此时你只需知道dumps把可序列化对象转成字符串。只有输出编码或序列化边界出错时，才深入它。

**练习：** 给输入`--count 5 --batch-size 2`写完整数据流：argv字符串列表→count整数5与size整数2→range对象产生0..4→batches列表→JSON字符串→stdout字节。每次表示变化都写出来。这样以后读模型API客户端时，你会自然追踪messages→JSON→HTTP字节→响应事件。

## D2 为什么同一个名字可能指向不同对象

读取`python_values.aliases`。original、shallow、deep是三个局部名字，但名字不同不保证内部对象独立。画出外层列表和内层列表：original与shallow外层不同，第一内层相同；deep对应内层不同。

然后阅读`functions.batches`，观察`current=[]`重新绑定。对比如果写`current.clear()`：result里已经保存了对同一列表的引用，清空会影响已保存的批次。

这不是一个孤立Python小技巧。Agent轨迹快照如果保存同一messages列表引用，后续append会让历史快照一起改变；调度器如果多个请求共享错误的状态对象，可能出现跨请求污染；测试fixture如果复用列表，会产生顺序依赖。

**代码审阅任务：** 在任何`append(object)`附近问：这个object之后还会被修改吗？这里需要共享还是快照？若需要快照，复制深度多少？你必须按字段分析，不能一律deepcopy，因为连接、锁、GPU对象的复制语义不同。

## D3 `self.step()`的定义到底藏在哪里

先看一个完整小例子：

```python
class BaseAgent:
    def step(self):
        return "base"

    def run(self):
        return self.step()

class SpecialAgent(BaseAgent):
    def step(self):
        return "special"

agent = SpecialAgent()
print(agent.run())
```

输出special，不是base。run虽然定义在BaseAgent，运行时self仍是SpecialAgent实例，因此方法查找找到覆盖后的step。这就是为什么只看调用所在文件不够：你还要知道实例的实际类型。

若子类里调用`super().step()`，它不是简单“调用文件上面的函数”，而是按方法解析顺序继续查找。单继承时常表现为调用父类实现，多继承时要查看MRO。初学阶段避免自己设计复杂多继承；但读工程时看到super，应查清执行顺序。

有些框架还会把函数动态赋给属性、通过注册表取工具、用`__getattr__`代理。此时搜索`def step`只是起点，不一定足够。先打印`type(agent)`、查看类继承，再追构造函数和注册位置。不要因为方法名常见就断言它来自某个特定Agent库。

## D4 dataclass生成了什么，又没有生成什么

读取Budget类。字段limit参与初始化，used由field设为init=False并默认0。dataclass替你生成常见样板方法，实例仍通过普通属性保存状态。

`__post_init__`验证limit；`remaining`是property，每次计算limit-used；`consume`维护状态。你能把它改写成普通类：手写`__init__(self,limit)`，赋self.limit与self.used，再做校验。完成一次改写有助于去掉装饰器带来的神秘感。

可变默认值必须特别注意。假设一个Session有messages列表，正确字段通常使用`field(default_factory=list)`，让每个实例得到新列表；共享默认列表会让不同会话互相影响。你应写两个实例的测试证明隔离，而不只是看单实例工作正常。

frozen dataclass约束字段赋值，但若字段指向可变列表，列表内容仍可能修改。`frozen=True`不等于递归不可变。协议消息、任务快照和缓存key需要的不可变性要按实际结构判断。

## D5 看见装饰器，先问它是否改变调用路径

最小装饰器可以这样写：

```python
from functools import wraps

def announce(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        print("calling", function.__name__)
        result = function(*args, **kwargs)
        return result
    return wrapped

@announce
def add(a, b):
    return a + b

print(add(2, 3))
```

执行def之后，add名字被绑定到包装函数；原始函数保存在闭包里。调用add先打印，再调用原函数，再返回结果。若包装器忘记return result，外部得到None，这与工作簿A的return/print问题同源。

`*args`收集位置参数，`**kwargs`收集关键字参数；转发时把它们展开。wraps保留部分元数据，方便调试和文档，但不自动让所有签名行为与原函数完全相同。

框架中的装饰器可能注册工具、缓存结果、编译Kernel、追踪日志或改成远程调用。它们都使用@语法，却可能改变不同层面。遇到`@triton.jit`时应查编译与运行语义；遇到`@tool`时应查schema与注册；不能凭语法外观把它们都当“普通函数别名”。

## D6 生成器的执行时间线

读取`iterators.stream_batches`，按下面代码手算：

```python
source = iter([1, 2, 3, 4])
chunks = stream_batches(source, 3)
first = next(chunks)
second = next(chunks)
```

创建chunks时，函数体尚未开始消费source。第一次next先检查size，取1、2、3，yield第一个列表后暂停；第二次next从yield之后恢复，建立新batch，取4，输入结束后yield尾批。第三次next才抛StopIteration。

因此你在生成器函数开头写的参数校验，可能不是调用时立即发生，而是第一次迭代才发生。若API要求构造时就拒绝参数，可以在外层普通函数先校验，再返回内部生成器。这是可观察的接口行为，不只是实现细节。

流式API响应也有类似特点：得到一个迭代对象不代表网络响应已全部成功，错误可能在消费到第十个事件时出现。调用方必须把异常处理覆盖整个迭代过程，而不是只包创建迭代器那一行。

## D7 `with`在异常发生时做什么

读取file_io.atomic_json。进入NamedTemporaryFile上下文后获得文件对象；退出时关闭文件。外层finally负责清理临时路径。两层生命周期不同：文件句柄要关闭，文件名也可能要删除。

再读取sqlite_lab.transaction_demo。`with db`内部第一条update成功，第二条insert违反约束，异常离开with，触发事务回滚；外面的except捕获IntegrityError。最终value仍10。

如果你把except移到with内部并吞掉异常，with看到的是正常离开，可能提交前面已经成功的update。这是一个特别值得亲手试的小变化：异常捕获位置能改变事务结果。请在新的内存数据库实验，不修改公司数据库。

上下文管理器本质上约定进入与退出动作，并接收异常信息。退出方法还可能抑制异常，所以读自定义context manager时不能只看with里的代码，也要看退出逻辑。框架中的锁、计时、缓存事务和GPU stream上下文都可能使用同一语法。

## D8 异常分类决定系统会不会悄悄吞bug

比较工作簿阶段parse抛ValueError与F15专门InvalidRecord。小函数内的ValueError大多是输入错误；工程扩大后，某个内部算法也可能因为bug抛ValueError。若分析器捕获所有ValueError并把它记作坏行，你会把程序缺陷误归因于输入。

专门业务异常让边界更清晰。CLI可以捕获InvalidRecord、OSError、UnicodeError这些预期用户可处理失败；意外bug保留traceback供开发者发现。生产系统外层可能仍有统一错误响应，但应记录内部原因并避免把它报告成成功。

错误链也很重要：`raise InvalidRecord("invalid JSON") from e`保留底层解析原因。面向用户的摘要可以简洁，面向调试的证据应该能追到原异常。你以后做Agent工具错误分类，需要区分可重试、参数错误、权限错误、外部超时和内部bug，而不是让模型猜一段自由文本。

## D9 asyncio中的一次切换具体发生在哪里

读取concurrency.squares。每个one先等待semaphore，进入后active加1，更新peak，然后await sleep(0)。sleep(0)主动让出执行机会；其他任务可能获得许可，active继续增长，直到并发上限。之后某个任务恢复，计算平方，finally将active减1，退出async with释放许可。

用并发2、输入0、1、2手算一种可能排程：任务0进入active1并让出；任务1进入active2并让出；任务2等待许可；任务0恢复返回，active1并释放；任务2才可能进入。具体顺序可能变化，但peak<=2与结束active=0应保持。

这说明测试应检查不变量，而不是锁定某一种事件排序。若你把active减1放在return之后，那一行永远不会执行；若放在try正常路径末尾，异常或取消可能跳过。finally表达的是“无论怎样离开都恢复资源计数”。

信号量只限制区域内任务数，外部已经创建的协程对象仍占内存。阅读真实Agent批量工具调用时，检查它是否为十万项一次性创建十万个task，是否有有界队列，是否保留完成结果，是否能取消整个任务树。不要只看一个concurrency参数就认为资源都受控。

## D10 从测试反推源码合同

打开`tests/test_foundations_projects.py`。先看测试名，再看断言，而不是先读全部实现。`test_create_retry_and_conflict`告诉你同键同请求是200、不同title是409、最终只有一条；`test_sql_metacharacters_are_literal`说明参数化SQL合同；`test_persistence`说明关闭重开后要保留记录。

接着回到TaskStore对应方法，圈出支持每条合同的代码。若某条测试无法对应到明确实现路径，可能你还没理解；若某个实现分支从来没有测试覆盖，想一想它是否重要。

测试也可能不完整或错误，因此不能把测试当绝对规范。最稳妥的是需求、测试和实现三者互相核对。开源项目中还可看issue与提交历史，理解某个奇怪条件为何存在；但不要在不了解历史时随手“简化”边界检查。

## D11 阅读笔记的最小格式

每次源码阅读只记录一条主线：入口是什么，输入有哪些字段/shape，谁拥有状态，调用了哪些核心组件，在哪些地方可能等待，怎样返回结果，失败如何清理。把文件位置写成仓库路径，版本写成实际commit或已记录版本，不写“最新版”。

例如日志项目的阅读笔记可以是：main解析Path→文件逐行输入analyze→parse_record建立Record→seen去重→latency排序统计→atomic_json保存→stdout序列化。失败分成输入IO、记录合同与输出保存。内存限制是保存所有有效记录。

这样的笔记能迁移到vLLM：HTTP入口→请求规范化→tokenizer→调度队列→KV分配→模型执行→采样→输出。对象更多，但问题类型相同。你在基础项目里训练的不是某个小应用，而是读系统代码的观察方式。

## D12 十道独立改动题及参考验收

1. 给Budget加入refund，要求不能退超过已用量。验收：失败状态不变；参考先验证0<=amount<=used再更新。
2. 给stream_batches增加drop_last参数。验收：只在显式True时丢尾批；默认行为不变；空输入无空批次。
3. 给CLI增加--format json/text。验收：业务统计不复制，呈现分支在入口层。
4. 给parse_record增加可选metadata。验收：你必须明确schema版本与未知字段策略，不能无意破坏现有精确字段合同。
5. 给TaskStore新增get(id)。验收：未知ID错误与list空列表区分；返回副本或普通dict，不泄露数据库Row。
6. 为原子保存模拟序列化失败。验收：旧文件不变，临时文件清理；不要要求所有文件系统都具备未测试的断电语义。
7. 为异步任务增加故障注入。验收：许可与active计数释放，错误传到调用方。
8. 写一个装饰器记录调用次数。验收：保留返回值与异常，不把所有参数记录到日志导致密钥泄露。
9. 将MemoryTasks替换为TaskStore但保留外部测试。验收：状态合同不变，另加重启持久化测试。
10. 为同一功能写一份“我不理解的三件事”。验收：问题具体到某个数据流或控制流，例如为何捕获异常在with外，不写笼统“Python好复杂”。然后用最小实验逐个回答。

完成标准：给你一个约百行的新模块，你能先运行其最小入口，找出主调用链，写至少一个新的边界测试，并做一个不破坏既有合同的小修改。你不必记住所有Python语法，但需要会用证据把陌生代码逐步变成熟悉代码。
