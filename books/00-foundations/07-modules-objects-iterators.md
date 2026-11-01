# F07 模块、对象、迭代器与接口：从一个文件走到可维护的程序

> 先修 F06；本章不是一天的任务。先完成模块和实例，再学迭代协议，最后写可替换的组件。离线实验：`python -m labs.foundations.objects`、`python -m labs.foundations.iterators`。本章的小文件实验均放进你自己的练习目录，不覆盖课程文件。

## 1. 模块不是文件夹：名字必须有归属

上一章把函数写在同一个文件中，解释器从上往下建立名字。程序变大以后，我们需要回答两个问题：某个名字在哪里定义，修改它会影响哪些调用者。模块首先解决名字归属问题。一个普通的 Python 源文件被加载后对应一个模块对象；模块对象拥有自己的命名空间，其中存放函数、类和变量。函数不是一段被复制到每个调用者文件中的文字，它是对象，可以通过模块属性访问。

假设练习目录有 `prices.py`，其中定义 `unit_price = 2` 与 `cost(count)`。另一个文件执行 `import prices`，当前文件得到名字 `prices`，它指向模块对象。表达式 `prices.cost(3)` 先找当前命名空间中的 prices，再到该对象里找 cost，最后调用函数。点号不是文件路径分隔符，而是属性访问运算符。在这里属性恰好由模块保存；后面同一个点号也能访问实例的方法。

```python
# prices.py
unit_price = 2

def cost(count):
    return count * unit_price
```

```python
# run_prices.py
import prices

print(prices.cost(3))
print(prices.unit_price)
```

运行 `python run_prices.py`，应得到六和二。注意函数体中的 unit_price 在函数定义所属的模块里查找。即使你在调用文件写 `unit_price = 100`，也不会改变 prices 模块的变量。这个边界让两个不同文件都能使用普通名字而不互相污染。理解这个规则之后，“同名变量为什么不生效”就不再需要猜测。

`from prices import cost` 则在当前命名空间直接绑定 cost，因而可以写 `cost(3)`。这不是把整个 prices 模块的所有变量复制过来，也不是让两个文件的全局变量合并。模块仍然需要被找到并加载。两种写法主要改变调用端如何取名字，不改变函数定义时所属的全局命名空间。对于来源不明显的通用名字，如 load、run、parse，保留模块前缀往往更容易阅读。

把 `from prices import unit_price` 得到的名字重新赋值，也不会同步修改 `prices.unit_price`。两个名字一开始引用同一个整数对象，随后重新绑定其中一个名字，另一个引用保持原样。若导入的是列表，调用列表的 append 则会改变那个共享对象。这里仍然适用上一章的“重新绑定与修改对象”区别，导入没有发明第三套赋值规则。

避免 `from module import *`。它使当前文件出现哪些名字依赖另一个模块的导出情况，读者难以定位来源，还可能覆盖已有名字。对初学者而言，显式导入不是繁琐仪式，而是在代码中留下数据来源的路标。阅读大型框架时，先找 import 区域，再找调用点，通常比仅搜索函数名字更快。

## 2. 导入的时间：为什么一次 import 会执行代码

导入不是只读函数签名。第一次正常加载一个模块时，Python 会执行该模块的顶层语句，建立命名空间。函数定义语句创建函数对象，通常不执行函数体；类定义语句要执行类体以构造类。顶层的 print、文件读取、网络请求同样会执行。这意味着把昂贵工作写在模块顶层，会让“只想调用一个小函数”的使用者也付出代价。

```python
# import_probe.py
print("loading import_probe")

def answer():
    print("running answer")
    return 42
```

在同一个解释器进程中先后执行两次 `import import_probe`，正常情况下加载提示只出现一次。Python 用 `sys.modules` 保存已加载模块，之后的导入通常复用这个对象。然后调用 `import_probe.answer()`，才出现 running answer。这三个动作分别是模块加载、复用模块、运行函数，不能合并理解成“import 会运行所有代码”。函数体里的内容要等调用才执行。

缓存以进程为边界。关闭解释器重新启动后，它不是延续上次 Python 内存中的模块缓存。修改文件后，已经运行的进程也不保证自动看到新定义。教学时最容易解释的验证办法是退出旧进程重新启动。`importlib.reload` 可以重执行模块，但旧函数引用、旧实例、其他模块缓存可能仍指向先前对象；先不要把热重载当作所有开发环境问题的万能修复。

循环导入的典型形状是甲模块顶层导入乙，乙又从甲导入一个尚未定义的名字。此时甲只执行了一部分，不能保证所需属性已经存在。解决时先画依赖方向：哪些是基础数据结构，哪些是业务逻辑，哪些是入口。把双方共享的小定义提到第三个低层模块，或将操作推迟到函数调用阶段，通常比交换两行 import 更可靠。后者可能偶然避开当前错误，但没有消除结构问题。

`if __name__ == "__main__":` 把演示与可复用定义隔开。当模块被作为入口运行，特殊名字通常是 `__main__`；作为普通模块导入，它具有导入路径名称。判断条件为真才调用 main。这个写法不是“不让别人用我的代码”，而是允许同一文件同时提供库函数和手动演示。测试导入库函数时不应顺带启动无限运行的 HTTP 服务。

```python
def main():
    print("manual demo")

if __name__ == "__main__":
    main()
```

逐行观察：第一行建立函数名，缩进体先不执行；最后两行在模块加载时判断入口身份；只有条件成立才进入函数。名字 main 没有特殊魔法，你可以叫它其他名字，但社区惯例让入口更容易识别。特殊的是解释器给 `__name__` 的值以及你明确写下的条件判断。

## 3. 包、模块路径与执行入口

包把模块组织进层次。为了先获得可预测的体验，你的练习包可以显式建立空 `__init__.py`。现代 Python 也有命名空间包等机制，因此“没有这个文件就永远不可能是包”是不准确的；当前学习阶段采用普通包即可。目录层次不是任意装饰，它参与导入路径，例如 `labs.foundations.objects` 表示沿包层次定位 objects 模块。

从仓库根目录运行 `python -m labs.foundations.objects` 时，`-m` 告诉解释器按模块名称定位目标并作为入口执行。这里没有 `.py` 后缀，点号表达包层次。直接运行深层路径与模块执行会建立不同的入口上下文，包内相对导入可能因此失败。不要把 `sys.path.append` 随手塞到每个文件开头；先确认工作目录、解释器与执行方式。

相对导入中的一个点表示当前包，例如 `from .functions import batches`。两个点表示向上一级包寻找。它不是相对当前终端工作目录的文件路径，也不取决于运行时 `os.chdir` 到哪里。包上下文来自模块加载方式。相反，`open("data.txt")` 通常按进程当前工作目录解释，这两个“相对”属于完全不同的规则。

搜索路径中若出现你自己命名的 `json.py`、`typing.py`、`sqlite3.py`，可能遮蔽原本想用的标准库模块。症状常是“模块没有本应存在的属性”，而不是文件不存在。诊断时打印模块的 `__file__`，确认实际加载位置；某些内置模块没有普通文件路径，要允许这种情况。修改文件名后还应重启进程，避免继续使用已缓存的旧对象。

模块组织可从三个文件开始：纯计算放 `core.py`，文件或网络交互放 `io_layer.py`，参数解析与流程编排放 `__main__.py` 或单独 CLI 文件。纯计算层不应反向导入命令行入口。这样导入核心算法时不会要求终端参数或启动服务器。分层依据是责任和依赖方向，不是强迫每个函数一个文件。

一个可观察的学习实验是：把同一个解析函数在交互终端、测试文件、命令行入口中各调用一次。若三种调用都要复制代码，说明尚未建立清楚的模块边界；若导入一次便打印菜单并等待输入，说明顶层副作用妨碍了复用。改成函数接收参数并返回值后，调用位置就能自行决定输入来源和输出去向。

## 4. 类、实例与 self：把状态的主人找出来

先看一个不使用任何装饰器的类。类定义描述可以创建什么样的对象。实例是调用这个类后得到的具体对象，每个实例可以保存自己的状态。类不是数据库表，也不是必须与现实世界名词一一对应。它只是组织行为和状态的一种机制，是否值得使用，要看是否确实需要共同维护一组规则。

```python
class Counter:
    def __init__(self, start=0):
        self.value = start

    def add(self, amount):
        self.value = self.value + amount
        return self.value

first = Counter(10)
second = Counter(100)
print(first.add(3))
print(second.value)
```

`Counter(10)` 创建实例并完成初始化；`__init__` 接到的新对象通过第一个参数 self 访问。`self.value = start` 把 value 属性写到这个实例。`first.add(3)` 取得绑定方法，调用时自动把 first 作为 self；可以用 `Counter.add(first, 3)` 帮助理解等价调用关系。self 是惯例名称，不是某个隐藏的全局变量，也不是所有实例共享的内存区域。

运行状态必须按对象分别记录。初始 first.value 为十、second.value 为一百。调用 first.add 后只修改 first 对象，返回十三。second 仍是一百。方法体的 amount 是这一次调用的局部参数；退出调用后，这个局部名字结束其本次使命，但实例上的 value 保留下来。方法每次调用都能访问同一个实例状态，因此对象适合计数器、缓存、预算与连接管理等任务。

`return self.value` 与 `self.value = ...` 是两件事。前者把结果交回调用者，后者改变对象。一个方法可以只返回计算结果，也可以只修改状态，或同时做到这两点。阅读接口时务必确认副作用：仅凭 `result = obj.step()` 的外形，无法判断 obj 是否被修改。良好的命名和文档应说明这个合同。

```python
class WrongBucket:
    items = []

class Bucket:
    def __init__(self):
        self.items = []
```

WrongBucket 的 items 是类属性，实例通过属性查找可能访问同一列表，因此一个实例 append 后另一个也能看到。Bucket 每次初始化都创建新列表，实例状态相互独立。不要只背“类属性共享、实例属性独立”口诀；需要看到分配列表语句执行了几次。类体通常在类定义执行时运行一次，而初始化函数随每个实例创建被调用。

属性查找包含类与继承等规则，初学时先遵守简单纪律：每个实例拥有的可变字段在初始化里创建；类属性用于明确共享的常量或策略；不要随意从外部加上拼写相近的新字段。`budget.usde = 3` 可能不会立即报错，却没有修改真正的 used，这类错误比显式失败更难定位。

## 5. 用 Budget 理解不变量和合法状态转换

运行仓库中的 objects 实验，然后打开源文件。Budget 的核心不是 dataclass 这个名字，而是不变量 `0 <= used <= limit`。不变量指对象在对外可观察的正常状态下始终应该满足的条件。每个公开操作都要从合法状态走到另一个合法状态；若输入不合法，操作应拒绝，而且本例要求拒绝后状态不变。

```python
from labs.foundations.objects import Budget

budget = Budget(10)
budget.consume(3)
budget.consume(4)
print(budget.used, budget.remaining)
try:
    budget.consume(4)
except ValueError as error:
    print(type(error).__name__, str(error))
print(budget.used, budget.remaining)
```

第一次消费前 used 是零，remaining 是十；消费三后 used 为三；消费四后 used 为七；第三次消费四超过剩余三，抛 ValueError，used 仍为七。这里不应该把“发生异常”自动理解为“所有修改都被撤销”。Python 普通对象没有自动事务回滚，本例之所以保持状态，是因为源代码把验证放在修改之前。

| 操作 | 操作前 used | amount | 检查 | 操作后 used |
|---|---:|---:|---|---:|
| 创建 | 无 | 无 | limit 是非负整数 | 0 |
| consume | 0 | 3 | 3 不超过 10 | 3 |
| consume | 3 | 4 | 4 不超过 7 | 7 |
| consume | 7 | 4 | 4 超过 3，拒绝 | 7 |

`type(amount) is int` 是有意采用的严格合同。Python 中 bool 是 int 的子类，`isinstance(True, int)` 会成立，但预算扣费通常不希望布尔值被当成一个 token。严格类型检查在这里表达领域规则，不代表所有 Python 函数都应该拒绝整数子类。判断依据是接口接受什么，不是机械套用某一种检查习惯。

`remaining` 用 property 根据 limit 和 used 即时计算，所以不能在构造后独立缓存一个容易失去同步的剩余值。若同时存 used、remaining、limit，消费时就要维护三者一致；从两个字段推导第三个可以减少状态。property 外形像普通属性，实际访问会调用函数，因此不应在毫无提示的属性读取里执行很慢的网络请求。

这个类没有强制禁止所有外部修改。调用者若直接写 `budget.used = 999`，仍可能破坏不变量。Python 的下划线命名更多是约定，不等于操作系统级访问控制。教学版本通过清楚接口和测试维护规则；更严格的接口可隐藏字段、提供只读属性，但不能把普通对象封装当成不可信代码沙箱。

并发下还会多一个问题：两个调用都先读取剩余值，再同时修改 used，可能破坏原本的逻辑。顺序程序里“先检查后修改”足够解释本例，跨线程或跨进程共享状态则需要锁、事务或其他协调。不要从一个单线程演示直接推断它已经能承担分布式计费。F11 会把操作拆开演示竞争条件。

## 6. dataclass、装饰器与对象表示

`@dataclass` 是执行类转换的装饰器语法。你可以把它直观理解为：先按类定义得到类，再把这个类交给 dataclass 处理，最后用处理后的对象绑定原来的类名。装饰器不是注释，它可能增加方法、注册函数、改变调用方式，甚至返回另一种对象。读陌生框架的装饰器时，要追踪其真实定义。

Budget 的字段 limit 没有默认值，因此构造时必须提供。`used: int = field(default=0, init=False)` 表示默认零且不作为生成初始化方法的调用参数。于是 `Budget(10)` 合法，而试图 `Budget(10, 7)` 不符合生成的构造签名。类型标注 `: int` 本身不会替你执行运行时验证；真正的检查在 `__post_init__` 里。

`__post_init__` 在 dataclass 生成的初始化完成字段赋值后执行，适合验证字段之间关系。它不是每次修改字段时自动触发的钩子。构造通过之后，直接修改 limit 不会自动重新运行它。这个时间点必须清楚，否则你可能以为一个带类型标注的对象永远不可能处于无效状态。

```python
from dataclasses import dataclass, field

@dataclass
class Conversation:
    name: str
    messages: list[str] = field(default_factory=list)

left = Conversation("left")
right = Conversation("right")
left.messages.append("hello")
assert right.messages == []
```

`default_factory=list` 传入的是函数对象 list，不是已经调用得到的某个共享列表。每次构造实例，工厂被调用，产生独立列表。若你写 `list()`，那是在定义字段时立即求值，时间点变了。这里和函数默认可变参数的问题相似：关键始终是对象何时创建、谁拥有它、是否被多次复用。

dataclass 通常生成便于观察的 repr 和基于字段的相等比较。打印对象有助于查看状态，但不要因此把含有 API key 的对象完整写进日志。对象表示是调试接口，不应未经筛选成为外部序列化格式。以后字段增减可能改变 repr；跨进程协议应显式选择字段并用 JSON 等格式编码。

`frozen=True` 可以阻止普通字段重新赋值，但不保证深层不可变。一个冻结对象若包含列表，这个列表仍可能被修改。可哈希性、相等性和可变性彼此相关：若对象作为字典键后参与比较的值改变，查找可能失去预期。初学时用字符串、整数或合适的不可变元组当键，避免拿复杂可变对象直接做缓存键。

## 7. 组合、继承与可替换接口

假设你要生成一份日志报告。统计逻辑负责计算，writer 负责输出。最直接的接口约定是 writer 对象拥有 `write(text)` 方法。调用者只关心这个方法能接收什么、返回什么、可能失败什么，不需要知道它内部是写文件、打印终端还是把文本保存到测试列表里。

```python
class ListWriter:
    def __init__(self):
        self.items = []

    def write(self, text):
        self.items.append(text)

class Reporter:
    def __init__(self, writer):
        self.writer = writer

    def report_count(self, count):
        text = f"count={count}"
        self.writer.write(text)
        return text

writer = ListWriter()
reporter = Reporter(writer)
assert reporter.report_count(3) == "count=3"
assert writer.items == ["count=3"]
```

Reporter 有一个 writer，这叫组合。它不需要继承 ListWriter，因为报告生成器不是一种列表写入器。组合把不同职责放到不同对象，并通过构造参数连接起来。所谓依赖注入，在这个例子中就是把需要的对象传进来，而不是在方法内部固定新建某个具体实现；没有要求额外安装容器框架。

测试时注入 ListWriter，能检查收到的文本而不用捕获真实终端。生产入口可以注入 FileWriter。计算逻辑不变，外部行为可替换。这不是为了抽象而抽象：替换需求已经非常具体，一种运行方式要写文件，另一种测试方式要保留观察记录。反过来，只有一个五行函数且不存在替换需求时，强行建立多层类体系会增加阅读负担。

继承表达一种对象应能在父类型被需要的地方使用。子类不能随意缩窄输入合同或改变结果含义。例如父类允许 write 任意字符串，子类却只接受长度十以内而不提前说明，调用者就无法可靠替换。阅读框架时看到多层继承，先定位实际实例类，再追踪被调用方法在类及其父类中最先找到的定义。

`super()` 并不只是“硬编码调用父类名字”的缩写，它沿方法解析顺序协作。多继承还会引入更复杂的顺序规则。第一版工程实践应优先读懂单继承与组合，并能够通过 `type(obj).__mro__` 观察查找顺序；除非有清楚理由，不要在自己的初学项目引入交叉多继承。能读懂复杂机制不意味着每个小项目都要使用它。

类型提示中的 Protocol 可以描述结构化接口：符合所需方法形状的对象即可被静态分析器视为候选实现，不必继承一个共同业务父类。它帮助编辑器和检查工具发现错别字、参数数量不符等问题，但不能代替运行时输入验证，也不能自动证明一个外部服务真的返回正确结果。合同仍需要文档、测试与边界检查共同维护。

## 8. 可迭代对象、迭代器与一次性状态

列表保存若干对象引用，可以多次遍历。迭代器则保存当前遍历进度，每次取下一项都会推进状态。可迭代对象能通过 `iter(x)` 提供迭代器；迭代器通过 `next(it)` 给出下一项，结束时抛出 StopIteration。for 循环帮你执行这个协议，并把正常结束转换成退出循环。

```python
values = [10, 20, 30]
iterator = iter(values)
assert next(iterator) == 10
assert next(iterator) == 20
assert list(iterator) == [30]
assert list(iterator) == []
assert list(values) == [10, 20, 30]
```

逐步看，values 自身没有因为遍历被删掉元素。变化的是 iterator 记录的位置。第三行消费第一项，第四行消费第二项；`list(iterator)` 接着消费余下内容，不会自动从开头重新开始。最后重新遍历 values 时，它提供了新的迭代器，因此仍能看到全部元素。混淆“数据容器仍存在”和“某个遍历状态耗尽”是流式程序常见错误。

for 语句在概念上先调用 iter，再重复 next，直到 StopIteration。不要在普通业务函数中随意用 StopIteration 表示失败；它在迭代协议里承担正常结束的特殊含义。迭代期间出现 ValueError、文件读取错误等其他异常，for 不会把它们当作正常结束吞掉。

自定义迭代器需要 `__iter__` 与 `__next__`。通常迭代器自己的 `__iter__` 返回 self，表示继续消费当前进度，而可反复遍历的容器每次可以返回新的迭代器。下面的倒计数对象是一次性的：它把剩余计数放在实例中，下一次遍历不会自动恢复初始值。

```python
class Countdown:
    def __init__(self, start):
        if type(start) is not int or start < 0:
            raise ValueError("non-negative integer required")
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value

counter = Countdown(3)
assert list(counter) == [3, 2, 1]
assert list(counter) == []
```

第一次 next 时 current 为三，先保存三，再把状态减为二，返回三；结束判断发生在返回之前。若先减再返回，输出会变成二、一、零，合同改变了。用这类三步状态表训练自己读循环，后续看 token 生成器、分页游标与数据加载器时会直接受益。

## 9. 生成器：函数暂停在哪里

生成器函数的函数体含有 yield。调用它通常只创建生成器对象，不立即执行整个函数体。下一次 next 才开始或继续执行，直到遇到 yield 交出一个值，并保存本次调用的局部状态。之后再次 next，从暂停点之后继续。它不是每次都从第一行重新调用普通函数。

```python
def observed_squares(limit, seen):
    for number in range(limit):
        seen.append(number)
        yield number * number

seen = []
source = observed_squares(4, seen)
assert seen == []
assert next(source) == 0
assert seen == [0]
assert next(source) == 1
assert seen == [0, 1]
assert list(source) == [4, 9]
assert seen == [0, 1, 2, 3]
```

seen 是观察工具，记录真正执行过的输入。创建 source 后没有任何记录，说明不是预先算好所有平方。每取一次值，局部 number 和循环进度一起推进。这里惰性节省的是“暂时不需要的计算与存储”，但需要最终消费全部结果时，总计算量并没有凭空消失。

仓库 `stream_batches` 的输入是 Iterable，输出是 Iterator，每次产出一个列表。类型变量 T 表示输入元素类型与输出列表元素类型保持关联，并不是运行时把元素转换成某种类型。函数既能接列表，也能接生成器，因为它依赖的是迭代协议，而不是 `len(items)` 或按下标取元素。

```python
from labs.foundations.iterators import stream_batches

source = (i * i for i in range(7))
chunks = stream_batches(source, 3)
first = next(chunks)
assert first == [0, 1, 4]
assert list(chunks) == [[9, 16, 25], [36]]
assert list(source) == []
```

每个满批次被 yield 后，代码使用 `batch = []` 重新绑定一个新列表。若改成 `batch.clear()`，先前已经交给下游的列表与内部 batch 仍是同一个对象，清空会让消费者保存的结果也变化。重新绑定让每个批次拥有独立容器，这是上一章别名规则在流式代码中的真实应用。

最后的 `if batch:` 处理不足一个完整批次的尾部。没有它，七个元素按三分批会丢掉最后一个。若无条件 yield batch，输入刚好整除时又会多出空批次。写边界测试时应选零、一、等于批大小、比批大小多一这几组长度，因为它们分别覆盖空输入、尾批、满批和再次进入循环。

生成器中的参数验证也惰性执行。因此 `stream_batches([], 0)` 这一行本身通常只创建对象，错误要到 next 或 list 消费时才抛出。若希望构造时立即验证，可以写一个普通外层函数先检查参数，再返回内部生成器。两个接口都可以合理，关键是明确错误出现的时间，使调用者知道异常边界放在哪里。

## 10. 流式链路的内存、关闭与算法代价

一个每次产生一行的生成器，并不保证整条程序使用常数内存。若下游调用 list 收集所有行，内存仍随输入增长。若统计函数维护所有不同用户的字典，内存随不同用户数增长。需要同时描述输入长度、唯一键数量、批大小等变量，不能只看见 yield 就宣称“无限数据也不会占内存”。

对于 stream_batches，若下游消费后不保存旧批次，函数内部只保留当前批次以及迭代状态，额外引用数量与批大小相关。输入如果本来就是完整列表，其原有内存仍存在；函数节省的是新增的中间结果，不会释放调用者仍然引用的原始列表。讨论复杂度时必须说清楚计入的是总内存还是额外工作内存。

惰性链路可以像流水线：读一行，解析一行，验证一行，更新统计。各阶段之间通过迭代器传递，不必建立全部解析结果的中间列表。但如果排序全部记录，就必须考虑全局顺序需要保存多少信息，或者使用外部排序等办法。并非所有操作都能用同样的低内存模式完成。

生成器若在内部打开文件，还涉及资源生命周期。最简单的教学接口是让调用者用 with 打开文件，再把文件对象传给解析生成器。这样文件何时关闭一眼可见。把 open 隐藏在生成器内部并依赖耗尽后关闭，会在调用者提前 break 时产生理解难度；显式 close 或 contextlib 工具可以管理，但不应指望垃圾回收总在恰好的时间发生。

算法的时间复杂度描述输入变大时操作数量如何增长，不是给某次运行贴一个秒数。对每条输入做常量次解析和计数，大体是线性扫描；若每次为了去重都扫描已积累列表，最坏比较次数会接近一加二一直到输入长度，增长为平方。用集合记录已见键可让通常的成员查询更快，但还要考虑哈希质量、内存以及元素是否可哈希。

只有在读懂数据结构后才谈优化。将列表推导式换成生成器表达式，有时节省中间列表，但若接收者马上要求 list，节省可能消失；将清楚的循环压成一行，并不自动改变复杂度。先写可测的输入输出合同，再用计数或测量确认瓶颈，最后选择结构。后续框架和算子课程同样遵循这一顺序。

## 11. 三个故障现场与逐层诊断

**现场一：同一个程序直接运行成功，作为模块运行却导入失败。** 先记录完整命令、当前目录和解释器路径，再查看导入语句是绝对导入还是相对导入。若项目要求模块入口，回到仓库根目录执行 `python -m ...`。不要一边改工作目录一边修改导入语句，否则你不能知道哪个变化真正起作用。最后从一个新终端重试，以排除临时环境变量掩盖问题。

**现场二：第二个会话出现第一个会话的消息。** 先检查 messages 是类属性、可变默认参数，还是构造时共享了同一个外部列表。用 `first.messages is second.messages` 验证对象身份，比打印两个看起来一样的列表更有信息。修复为每个实例创建独立列表后，再检查复制是否需要深拷贝；若列表内部仍装着共享可变字典，浅层分离不一定满足业务隔离要求。

**现场三：第一次统计正确，第二次变成零。** 记录输入是列表还是生成器，再检查是否对同一迭代器重复消费。修复可以是每次调用一个工厂重新打开数据源，也可以明确缓存一份可重放数据。不要无条件把全部数据转列表，因为它改变内存合同。对大文件，重新打开文件通常比长期保存所有解析对象更符合预期，但也会付出再次读取的成本。

**现场四：所有批次最终都变成空列表。** 在 yield 后发现 clear，结合对象身份即可定位。消费者保存的并不是当时列表内容的自动快照，而是同一个对象的引用。用新建列表解除别名，或在明确需要时复制后再产出。验证不仅比较最终值，还要断言不同批次不是同一个列表；否则某些输入下的巧合可能掩盖问题。

## 12. 十二道分级练习与完整参考答案

每题先写输入、期望结果与状态变化，再运行答案。答案展示一个清楚的正确实现，不要求你的变量名完全相同。涉及异常的题目不能只确认“报错了”，还要检查错误种类和失败之后保留的状态。

### 练习一：方法调用的显式形式

创建两个 Counter，一个从零开始，一个从十开始。只给第一个加二，证明第二个不变，并写出不使用绑定方法外形的调用。

```python
class Counter:
    def __init__(self, value):
        self.value = value

    def add(self, amount):
        self.value += amount
        return self.value

first = Counter(0)
second = Counter(10)
assert Counter.add(first, 2) == 2
assert first.value == 2
assert second.value == 10
```

答案的关键是 self 对应具体实例，而不是类本身。两个构造调用各自建立字段；显式调用把实例放在第一个位置，其他参数依次后移。若传入 second，修改的对象也随之改变。

### 练习二：预算失败不能扣费

预算五，先消费四，再消费二。捕获预期异常后检查 used 与 remaining。

```python
from labs.foundations.objects import Budget

budget = Budget(5)
budget.consume(4)
try:
    budget.consume(2)
except ValueError:
    pass
else:
    raise AssertionError("over-budget call must fail")
assert budget.used == 4
assert budget.remaining == 1
```

else 属于 try 结构，只有没有异常时才执行，因此防止测试在错误实现未抛异常时蒙混通过。最后两条断言验证失败的状态合同，而不只是异常合同。

### 练习三：独立消息容器

用 dataclass 创建会话，证明添加消息不会串到另一会话。不要用 deepcopy 掩盖错误的默认字段定义。

```python
from dataclasses import dataclass, field

@dataclass
class Session:
    name: str
    messages: list[str] = field(default_factory=list)

one = Session("one")
two = Session("two")
one.messages.append("question")
assert one.messages == ["question"]
assert two.messages == []
assert one.messages is not two.messages
```

最后的身份断言指出为什么独立：不是两个容器目前碰巧内容不同，而是它们确实为不同列表。工厂在每次构造时调用，这才从源头防止共享。

### 练习四：有限重试编号

实现 attempts，输入必须为正整数，依次产出从零到次数减一的编号。明确错误在消费时出现。

```python
def attempts(count):
    if type(count) is not int or count <= 0:
        raise ValueError("positive integer count required")
    for index in range(count):
        yield index

assert list(attempts(3)) == [0, 1, 2]
for invalid in [0, -1, True, 1.5]:
    try:
        list(attempts(invalid))
    except ValueError:
        pass
    else:
        raise AssertionError(invalid)
```

这只解决有限次数，不代表已经构建可靠重试系统。真实请求还必须判断哪些错误可重试、何时等待、预算是否允许以及写操作是否幂等。把范围明确限制在编号生成，才能知道这个练习究竟证明了什么。

### 练习五：证明分批惰性

取得第一批后，输入只应访问前三项；剩余输入尚未被计算。

```python
from labs.foundations.iterators import stream_batches

seen = []
def source():
    for number in range(7):
        seen.append(number)
        yield number

chunks = stream_batches(source(), 3)
assert seen == []
assert next(chunks) == [0, 1, 2]
assert seen == [0, 1, 2]
assert list(chunks) == [[3, 4, 5], [6]]
```

记录副作用只是为了观察求值时刻。真实业务中尽量让数据变换少依赖隐藏副作用，否则消费者多取一个元素就可能意外发出一次外部写入。

### 练习六：构造时验证的生成器接口

要求非法 size 在调用函数时立即失败，而不是消费时失败；合法输入仍然惰性。

```python
def checked_batches(items, size):
    if type(size) is not int or size <= 0:
        raise ValueError("positive integer size required")

    def generate():
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    return generate()

try:
    checked_batches([], 0)
except ValueError:
    pass
else:
    raise AssertionError("must validate immediately")
assert list(checked_batches(range(5), 2)) == [[0, 1], [2, 3], [4]]
```

外层没有 yield，因此是普通函数，调用时执行校验；内部函数才是生成器。闭包保存 items 和 size 的引用，返回后仍可使用。这里没有复制输入，也没有提前遍历。

### 练习七：保留顺序的去重流

输入元素是可哈希值，第一次出现时产出，之后跳过。说明为什么输出惰性但内存不一定恒定。

```python
def unique(items):
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item

assert list(unique(["a", "b", "a", "c", "b"])) == ["a", "b", "c"]
```

每个不同值都需要留在 seen 中，额外内存随不同元素数增长。若不保存足够历史，就无法对任意长输入精确判断某个值以前是否出现。这是需求本身的代价，不是换成生成器语法便能消除。

### 练习八：可重放的数据工厂

同一份数字需要统计两遍，输入本身使用生成器。通过工厂每次建立新的遍历状态。

```python
def source_factory():
    return (number * number for number in range(4))

first_total = sum(source_factory())
second_total = sum(source_factory())
assert first_total == second_total == 14
```

工厂返回一个新生成器，而不是把先前创建的同一生成器反复返回。若数据源来自变化中的文件或网络，两次重放未必内容一致，还需要快照或版本标识；本题 range 的输入固定，因此可以要求相同结果。

### 练习九：用假 writer 测试输出

编写 render_and_write 接受数量和 writer，返回生成文本，并验证写入恰好一次。

```python
class RecordingWriter:
    def __init__(self):
        self.calls = []

    def write(self, text):
        self.calls.append(text)

def render_and_write(count, writer):
    if type(count) is not int or count < 0:
        raise ValueError("non-negative integer required")
    text = f"records={count}"
    writer.write(text)
    return text

writer = RecordingWriter()
assert render_and_write(7, writer) == "records=7"
assert writer.calls == ["records=7"]
```

记录调用列表比仅存最后一个值更有用：若错误实现写了两次，列表长度直接暴露问题。测试替身应该保存你真正需要观察的信息，不能让它默默修正被测代码的错误。

### 练习十：自定义可重复遍历容器

实现 Squares，每次 for 都从零开始，避免把容器本身做成一次性迭代器。

```python
class Squares:
    def __init__(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("non-negative integer required")
        self.count = count

    def __iter__(self):
        for number in range(self.count):
            yield number * number

values = Squares(4)
assert list(values) == [0, 1, 4, 9]
assert list(values) == [0, 1, 4, 9]
```

每次调用 `__iter__` 都产生新的生成器和独立局部进度；count 保存的是容器配置，当前位置并不保存在 Squares 实例上。由此可以同时拥有两个互不干扰的遍历。

### 练习十一：显示装饰器的转换过程

写一个包装函数，在调用前记录一次事件，保留返回值。这里只记录本地列表，不引入日志库。

```python
from functools import wraps

calls = []
def recorded(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        calls.append(function.__name__)
        return function(*args, **kwargs)
    return wrapper

@recorded
def add(left, right):
    return left + right

assert calls == []
assert add(2, right=3) == 5
assert calls == ["add"]
assert add.__name__ == "add"
```

定义时 recorded 收到原函数并返回 wrapper；调用 add 时实际进入 wrapper。星号参数收集位置参数与关键字参数，调用原函数时再展开。wraps 帮助保留名字与文档等元数据，但不会自动使包装函数获得原函数所有语义保证。

### 练习十二：分批列表不能相互污染

保存所有批次，修改第一批，后面的批次应该保持原样。

```python
from labs.foundations.iterators import stream_batches

chunks = list(stream_batches(range(5), 2))
assert chunks == [[0, 1], [2, 3], [4]]
assert chunks[0] is not chunks[1]
chunks[0].append(99)
assert chunks[1] == [2, 3]
assert chunks[2] == [4]
```

这里确认的是外层批次容器独立；若元素本身是共享可变对象，修改元素内部仍可能在其他位置可见。是否深拷贝由业务所有权决定，不能把“容器不同”误解成“内部整棵对象图完全隔离”。

## 13. 分阶段项目：可替换的流式报告器

第一阶段只做纯计算：输入一串非负整数，输出数量与总和。写成函数，测试空输入、单个值、多值以及非法值。不要先设计类，也不要读取文件。这个阶段证明统计规则正确，所有外部行为都暂时不存在，因此失败原因集中在逻辑本身。

第二阶段接收任意可迭代输入，包括列表与生成器。测试一个会记录消费事件的源，确认统计确实需要把全部输入消费完；与只取一批的函数形成对照。让你亲眼看到“函数需要完整结果”与“函数可以提前返回部分结果”对求值行为的不同要求。

第三阶段注入 writer 并输出稳定文本。用 RecordingWriter 检查一次输出，再写真实文件适配器。统计函数不要知道文件路径，文件适配器不要重新实现统计。第四阶段把主入口拆进独立模块，并用入口保护避免导入时执行演示。最后从新进程运行和从测试导入各验证一次。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Summary:
    count: int
    total: int

def summarize(values):
    count = 0
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("non-negative integers required")
        count += 1
        total += value
    return Summary(count, total)

def write_summary(values, writer):
    summary = summarize(values)
    text = f"count={summary.count};total={summary.total}"
    writer.write(text)
    return summary

class MemoryWriter:
    def __init__(self):
        self.items = []

    def write(self, text):
        self.items.append(text)

writer = MemoryWriter()
result = write_summary((x for x in [2, 3, 5]), writer)
assert result == Summary(3, 10)
assert writer.items == ["count=3;total=10"]
```

完整解把状态分成三处：生成器拥有读取进度，summarize 的局部变量拥有一次计算的累计状态，writer 拥有外部输出状态。它们没有共享全局计数器，因此重复调用容易理解。若非法输入在中途出现，summarize 抛错，write_summary 尚未调用 writer，因而不会输出半份报告；这是代码顺序带来的具体保证。

最终验收时画出依赖箭头：入口依赖统计与 writer，统计依赖 Summary，Summary 不依赖入口。再回答一次调用的输入从哪里来、对象在哪里创建、谁负责关闭文件、错误在哪里转成用户提示。你能沿这些问题读完自己的小工程，就已经掌握阅读 Agent 组件接口所需的基础方法。

## 14. 从调用点反向读源码：一次完整的追踪练习

真实项目的入口常常只写两行：构造某个对象，调用 run。初看会觉得所有细节都藏起来了。处理这种情况不需要把仓库每个文件读完，而要建立一条可验证的调用链。先找到变量第一次赋值的位置，确定它指向什么对象；再找该类定义，定位调用的方法；最后查看方法读取哪些字段、调用哪些依赖。每一步都记下文件、名字与输入输出，使推断可以回到源码验证。

以 `reporter.report_count(3)` 为例，第一问是 reporter 从哪里来。上方构造语句告诉你它是 Reporter 实例；第二问是构造时传了什么，答案是 ListWriter 实例；第三问是 report_count 是否重新构造 writer，源码显示它使用 `self.writer`。于是测试中传入的那个对象就是最终收到文本的对象，中间没有隐藏替换。这比仅根据类名猜测“应该是打印报告”更精确。

继续向下追踪 `self.writer.write(text)`，你必须区分静态文本与运行时对象。源码里的 writer 名字不会直接告诉你它是文件还是内存实现；构造入口才决定具体类型。若有多个入口，可能各自注入不同对象。阅读依赖注入风格项目时，配置与装配位置往往比某个抽象接口文件更能解释实际行为。先找到本次运行的装配，再分析调用路径。

随后把一份具体输入写进状态表：数量三进入方法，字符串形成 count=3，writer.calls 从空列表变成一个元素，方法返回同一个字符串值。这里的“同一个”应说明是在比较值还是对象身份；通常文本合同关心值，不应依赖字符串是否被解释器复用。把所有中间变量明确写出来，可以暴露隐藏的类型变化，例如原本整数经过格式化成为字符串。

遇到属性时再问是否为普通字段、property 或描述符提供的访问。Budget.remaining 外形没有括号，但源码 property 会做减法；它不从独立存储读取剩余数字。遇到方法时再问是否有装饰器包装；recorded 例子中实际先进入 wrapper，再调用原函数。这样逐层展开，你就能把简短调用式还原为真实执行事件，不会被语法表面欺骗。

最后读异常路径。若数量非法，是否调用 writer？若 writer 自身失败，report_count 是否仍返回文本？本章实现先写再 return，因此写入抛错时不会到达 return。调用者可能已得到部分外部副作用却没有返回结果，这和网络请求超时后的结果不确定性具有相同结构。源码阅读应把每个可能失败点标在时间线上，而不是只画一条成功箭头。

## 15. 闭包、回调与对象生命周期的补课

函数可以作为值传递给另一个函数。所谓回调，就是把一个可调用对象交给别人，让对方在某个条件下调用它，并不是一种独立于函数的新语法。例如 sorted 的 key 参数接受函数；每个元素需要排序键时，它调用这个函数。回调的合同必须说明接收什么参数、返回什么结果以及是否允许副作用，否则调用方与提供方很容易对不上。

```python
def make_multiplier(factor):
    def multiply(value):
        return value * factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
assert double(5) == 10
assert triple(5) == 15
```

外层调用结束后，内部函数仍需要 factor，因此相关引用被闭包保存。double 和 triple 来自两次外层调用，保存不同的环境。这不是把 factor 的名字变成全局变量，也不要求外层函数一直占用正在执行的调用栈。闭包保存所需引用，函数对象继续存在便可以以后调用。

然而闭包捕获名字的方式容易造成循环陷阱。在同一次外层作用域中建立多个函数，它们可能都引用同一个随后变化的变量。不要把“创建函数”自动理解为“把当时所有变量值都拍成照片”。可以使用一个工厂函数建立独立环境，也可以在适合的情况下用默认参数固定某次定义时求得的值。两种写法都应说明求值时间，而不是背神秘修复口诀。

```python
def make_reader(value):
    def read():
        return value
    return read

readers = [make_reader(number) for number in range(3)]
assert [reader() for reader in readers] == [0, 1, 2]
```

这个列表推导式可展开为：建立空 readers；遍历零一二；每次调用 make_reader，获得新的函数对象；把该函数对象追加。最后的推导式又遍历函数列表，逐个调用并收集返回值。先把推导式展开为普通循环，能帮助你辨认哪些位置在创建函数，哪些位置才在执行函数。

闭包和实例都能保存状态。一个很小的无状态策略可直接用函数，一个需要多个相关操作并维护不变量的组件更适合类。不要以“面向对象一定高级”或“函数式一定简洁”决定结构。检查谁拥有状态、谁可以修改、是否需要多个方法共享规则，再选择最容易说明的表达形式。

对象生命周期由引用关系影响，而不是简单等于变量出现的缩进范围。把一个对象加入全局缓存，它可能在函数返回后继续存活；生成器暂停时也保存局部引用；异常堆栈或调试器可能延长某些对象的保留。遇到内存增长，先寻找长期容器是否持续追加，再考虑运行时更复杂的回收细节。仅仅给变量赋 None 不会清除其他位置仍持有的引用。

## 16. 接口版本变化：把能替换写成可检查合同

一个接口至少应说明参数含义、允许范围、返回形状、错误种类、状态副作用与资源所有权。仅写“生成报告”不足以让另一人实现兼容版本。以 writer.write 为例，它接受字符串还是字节？返回字符数还是 None？失败前是否可能已经写入一部分？谁负责关闭底层文件？这些问题不解决，即使两个类都有同名方法，也未必可以安全互换。

本章教学 writer 只约定接受字符串，并在成功时记录一次完整文本；不模拟真实文件系统的部分写入或网络故障。扩展到文件 writer 时可以保持上层关注点，但要明确错误传播和生命周期。不要让报告器偷偷 close 调用者传进来的共享文件，除非合同明确约定它取得所有权。通常谁创建资源，谁负责在清楚的位置关闭，最容易保持一致。

参数默认值也是合同的一部分。若以前省略 size 表示十条，升级后变成一百条，输出值可能仍正确但内存和延迟改变。返回列表改为生成器则改变重复遍历、异常时间和计算时机，不能只因元素相同就称为完全兼容。面对这类修改，应写迁移说明或提供新接口，让调用者有机会调整使用方式。

错误类别同样重要。调用者可能只捕获 ValueError 处理用户输入；如果新实现把所有异常都包成 RuntimeError，会破坏现有错误分流。反过来，捕获所有 Exception 并返回空列表，会让真正的程序缺陷伪装成“没有数据”。接口设计需要让正常空结果和失败可以区分，否则上层无法选择重试、告警还是展示结果。

可以为多个实现运行同一组合同测试。例如 MemoryWriter 和 FileWriter 都应接收包含中文与换行的文本，不无故追加第二份内容；不同实现的额外行为由各自测试覆盖。合同测试不是要求内部代码一样，而是要求调用者可观察的共同部分一致。后续替换模型提供商、向量数据库或缓存实现时，这种思维比统一类名更有价值。

对初学项目，先写最少的真实接口。只有一个实现时仍可通过简单参数传递隔离副作用，不一定需要抽象基类；出现第二种实现后，再提炼已观察到的共同点。过早抽象常把尚未明确的差异藏起来，导致后续不断增加可选参数。良好的接口来自具体用例对照，而不是预先猜遍所有未来需求。

## 17. 本章终检：一次对象、迭代与模块的联合验收

把项目保存为普通包后，用新进程导入核心模块，确认没有打印、没有读用户输入、没有打开长时间运行的服务。随后从模块入口运行演示，确认输出与断言一致。这个对照证明复用和执行两种职责确实被分离，而不是依赖当前终端碰巧设置了某个全局变量。

用一个只有五个元素的生成器作为输入，逐次调用 next，记录每次读入、累计和产出。再把输入换成空生成器、一次性迭代器和列表，比较哪些行为应该相同，哪些行为因为可重放性不同而变化。最后把 writer 换成总会抛错的对象，观察计算是否已完成、是否有返回值、异常是否保留原始含义。三组实验共同检查调用链、数据链和失败链。

毕业标准不是能背出类、包、生成器的定义，而是能解释一段五十行工程代码的每个重要状态属于谁、何时创建、在哪里改变、何时释放。你还应能指出哪个接口允许替换、哪个生成器只能消费一次、哪个装饰器改变了实际调用过程，并通过一个小实验验证判断。达到这一点后，阅读模型客户端、工具注册器与框架数据加载器时，就有了稳定的分析方法。

还有一项容易被忽略的验收是导出边界。模块内部临时辅助函数可以采用下划线开头，提示调用者不要把它当稳定公共接口；这是一种约定，不会阻止访问。真正发布给其他模块使用的函数应有清楚名称和合同。若测试全部绕过公开入口调用内部函数，即使内部测试通过，也可能漏掉装配错误，因此至少保留一条通过公开接口运行的流程。

观察对象时也不要依赖内存地址的具体数字。不同运行的地址可能变化，测试应比较你关心的身份关系，例如两个容器是否同一对象，而不是断言某个 id 等于固定值。对于垃圾回收之后的新对象，运行时甚至可能复用曾经使用过的标识数字；把 id 写进长期业务记录不能替代稳定的业务标识。会话、请求与任务应使用明确生成并持久化的标识，而不是把当前进程中的对象地址当作跨进程身份。

最后建立一次微型代码审查：请把一个函数的输入验证移到副作用之后，再运行原有测试，观察是否有失败。如果没有任何测试发现状态被提前修改，说明验收还缺失败后状态检查。恢复正确顺序，并增加一个能证明拒绝操作不改变对象的断言。这类主动制造可恢复缺陷的实验，能检验你是否真的理解合同，而不仅是复制一份看起来整齐的代码。
