# F09 测试、调试与工程质量：把判断变成可以反复检查的证据

> 先修 F06–F08。实验：`python -m labs.foundations.debugging`。本卷测试：`python -m unittest discover -s tests -p 'test_foundations*.py' -v`。本章把一次失败从需求、复现、定位、修复一直追到回归验证；不要只阅读测试代码而不亲自让一个错误版本失败。

## 1. 测试首先是一个具体问题

假设函数接收一组数并返回均值。你运行一次，屏幕显示四，能否证明它正确？不能，因为还不知道输入是什么、四是不是正确答案、空输入如何处理、单元素是否失败，以及程序是否偷偷修改输入。测试把这些隐含问题写成可执行检查：给定具体前提，执行一个动作，观察结果及副作用，与明确预期比较。

正确预期来自需求、数学定义、独立参考或已确认的协议，不应该只是把当前程序输出复制成期望值。若错误均值程序输出六，你把测试也写成期待六，屏幕变绿只说明测试和程序一致，并不能说明业务正确。写测试之前先用纸笔手算一个小例子，能有效减少这种共同犯错。

测试不等于证明全部输入正确。有限样例只能提供有限证据，因此要选择有辨识力的样例。三条完全相同的普通输入，通常不如一个普通输入、一个单元素和一个空输入。边界样例之所以有用，是因为代码控制流常在这些位置改变：循环是否进入、尾批是否产生、分母是否为零、分支是否拒绝。

一个完整合同包含输入、输出、失败和状态四部分。对 Budget，输入是非负整数消费量，输出没有必要返回新对象，失败是超额或非法金额被拒绝，状态要求失败后 used 不变。若只测异常类型，先扣款后报错的实现可能仍通过；若只测成功后的余额，超额路径可能完全没有证据。

测试名称应说明行为。例如 `test_rejects_empty_sample` 比 `test_case_3` 更容易理解，失败时你立即知道违反什么合同。测试体可以按准备、行动、断言组织，但无需为了三段形式添加很多模板代码。目标是让读者看到输入和预期之间的关系，而不是把简单检查埋在复杂辅助框架中。

## 2. 均值错误的完整定位过程

仓库故意保留 broken_mean，它计算 `sum(values) / (len(values) - 1)`。对二、四、六，和为十二，长度为三，错误分母是二，因此结果六；正确均值应为十二除三，得到四。样本方差中的某些公式会出现减一，但均值定义没有这个修正。变量名或统计术语相似，不代表公式可以互换。

```python
from labs.foundations.debugging import broken_mean, mean

assert broken_mean([2, 4, 6]) == 6
assert mean([2, 4, 6]) == 4
assert mean([9]) == 9
try:
    mean([])
except ValueError:
    pass
else:
    raise AssertionError("empty input must fail")
```

本段第一条断言是对教学错误版本行为的观察，不是正确性验收。真正修复练习应写“输入二四六期待四”，让 broken_mean 失败，再修改自己的练习副本直到通过。不要为了让课程全部运行成功就悄悄删除故意错误函数；它存在的目的正是让错误可以被稳定复现。

缩小失败样例有助于排除干扰。三个元素说明数值不对，一个元素让分母直接变零，立即暴露边界。空输入则需要先决定业务语义：没有观测值时均值未定义，本课程抛 ValueError，而不是返回可能被误认成真实均值的零。修复函数先把输入转成列表，允许一次性迭代器，再检查空，再按真正长度求均值。

这个实现也有明确范围。把全部输入转成列表意味着额外内存随输入长度增长；它方便教学并支持重复访问，不代表是最省内存的在线均值。若需要流式版本，可维护总和与数量，但浮点累积误差、非法元素处理和异常后的状态仍需要定义。不要在修复一个分母错误时同时引入全新复杂算法，使验证范围失控。

调试过程应留下最小证据：原始输入、期望、实际、错误位置、原因、修复和回归测试。记录原因比记住“把减一删掉”更有价值，因为之后可能遇到不同外形但同样混用公式的问题。一个能解释根因的修复，也更容易被别人审查。

## 3. unittest 从发现到断言的每一层

标准库 unittest 可以组织独立测试、运行断言并汇总结果。测试类继承 TestCase，方法名字以 test_ 开头，测试运行器据此发现并调用。每个测试方法应建立自己需要的前提，不依赖另一个测试先执行。运行顺序不是业务流程，不应成为测试正确性的隐藏条件。

```python
import unittest
from labs.foundations.debugging import mean

class MeanTests(unittest.TestCase):
    def test_regular_values(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_single_value(self):
        self.assertEqual(mean([9]), 9)

    def test_empty_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            mean([])

if __name__ == "__main__":
    unittest.main()
```

`self.assertEqual(actual, expected)` 在不相等时抛出测试失败异常，运行器记录为失败并继续其他测试。`assertRaises` 是上下文管理器：主体应抛指定异常，没抛或抛了不匹配异常都会使测试失败。它不只是“捕获错误让程序继续”，而是把错误类别本身变成被检查的合同。

把文件保存为个人练习目录中的 `test_mean.py`，可以直接运行，也可以用 discover 搜索。命令 `python -m unittest discover -s tests -p 'test_foundations*.py' -v` 中，`-m` 运行 unittest 模块，discover 使用发现模式，`-s` 指定开始目录，`-p` 指定文件匹配模式，`-v` 显示详细测试名称。引号防止 Shell 提前把模式展开成当前目录中的文件。

失败与错误在测试报告里有所区分：断言不满足通常记为失败，测试过程中发生未预期异常通常记为错误。两者都需要处理，但诊断起点不同。失败提示实际结果与预期不符，错误可能说明测试前提尚未建立、依赖缺失或代码在断言之前就崩溃。不要只看最后数字，先读具体测试名称和异常堆栈。

普通 assert 适合教学片段，但 Python 优化模式可能移除断言语句，因此应用输入验证不应依赖 assert。正式库函数应显式 raise 合适异常；unittest 断言方法则用于测试框架检查。把“测试中检查不变量”和“生产入口拒绝非法输入”区分开，能避免运行方式改变后验证消失。

## 4. 夹具与独立性：每次测试从可说明的起点开始

夹具是测试所需的初始数据、对象和资源。可以是三条日志、一个内存数据库、一个空预算对象，也可以是临时目录。好的夹具足够小，读者能手算预期；太大的真实数据文件可能包含很多无关信息，使失败难以定位。真实数据有集成价值，但不应替代清楚的小型反例。

setUp 在每个测试前执行，tearDown 在相应清理阶段执行。对数据库等资源，还可以使用 addCleanup 或上下文管理器确保即使中途断言失败也清理。不要在 setUp 里启动真实外部服务或花费 API 预算，除非测试被明确标记并由使用者选择运行；日常单元测试应快速、离线、可重复。

```python
import unittest
from labs.foundations.objects import Budget

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.budget = Budget(10)

    def test_valid_consumption(self):
        self.budget.consume(3)
        self.assertEqual(self.budget.remaining, 7)

    def test_failure_preserves_state(self):
        self.budget.consume(7)
        with self.assertRaises(ValueError):
            self.budget.consume(4)
        self.assertEqual(self.budget.used, 7)
        self.assertEqual(self.budget.remaining, 3)

if __name__ == "__main__":
    unittest.main()
```

每个方法得到新 Budget，因此消费三不会影响另一个方法的起点。若把预算放在类属性或模块全局，两个测试可能共享状态，单独运行成功而整体运行失败。正确修复是隔离状态，不是把测试按某个顺序排好，或者在失败后再重置一个忘记初始化的变量。

临时资源的名字也应独立。所有测试都写固定的 `/tmp/report.json`，并发运行时可能互相覆盖，失败后的残留还会污染下次执行。TemporaryDirectory 为每次测试提供独立目录，退出时清理。测试验证文件内容时应使用自己的目录，不触碰用户真实配置、工作文件或公司数据库。

## 5. 从合同推导边界，而不是机械凑样例

对 batches，输入长度与批大小共同决定分支。长度零覆盖不进入循环；长度一覆盖只有尾批；长度恰好等于 size 覆盖满批后不应多一个空批；长度比 size 多一覆盖满批与尾批共存。非法 size 还包括零、负数、布尔与浮点，因为实现明确要求正整数。这些边界来自代码合同，而不是任意挑数字。

```python
from labs.foundations.functions import batches

for count in range(20):
    for size in range(1, 7):
        source = list(range(count))
        result = batches(source, size)
        flattened = [item for chunk in result for item in chunk]
        assert flattened == source
        assert all(0 < len(chunk) <= size for chunk in result)
        assert all(len(chunk) == size for chunk in result[:-1])
```

嵌套列表推导式按从左到右的循环顺序展开：先遍历每个 chunk，再遍历该 chunk 的 item，把元素追加进新列表。这个 flatten 检查所有元素与顺序保持，但单独它还不足以保证批大小；所以还要检查每批长度及除末批外都满。多条互补性质共同约束结果，比只写一个模糊“总数相同”更有效。

重复值也是有辨识力的输入。若分批函数错误地用 set 去重，输入 range 没有重复便发现不了。加入 `[1, 1, 2, 1]`，确认重复次数和顺序都保留。Unicode 则对文本解析重要，却不必机械加入纯整数求和的每个测试；样例应针对实际语义与可能缺陷。

反例设计可以从错误实现倒推。故意删掉尾批、把大于改为大于等于、把先验证后修改调换顺序，观察测试是否失败。这叫用可恢复的小改动检查测试辨识力。若错误版本仍全部通过，说明测试集存在盲区，而不是测试数量不够多。修复盲区要增加能区分行为的断言。

## 6. 性质、参考实现与变形关系

固定输入输出测试回答“这个例子是否正确”；性质测试回答“很多输入是否都满足某条规律”。对去重保序，结果中没有重复、每个结果来自原输入、顺序按首次出现排列。对排序，结果有序且元素多重集合不变。性质必须足够具体，因为只有“结果有序”会允许错误实现永远返回空列表。

参考实现可以更慢但容易读，例如用朴素双重循环实现小规模运算，再与优化版本比较。它的价值在于独立推导，而不是把优化代码复制一遍换变量名。数值算法还可以加入手算小例子和数学性质，使两个实现共同犯错的机会降低。测试的可信度来自多种独立证据相互支持。

变形测试在难以直接知道完整答案时很有用。均值的每个输入都加常数，结果也应加同一常数；把所有输入重复一遍，均值应不变；对合法 softmax 输入统一加常数，概率理论上不变。实际浮点计算需要合理容差。先说明关系的数学前提，不能把不适用的关系强套到任意函数。

```python
from labs.foundations.debugging import mean

values = [2.0, 4.0, 6.0]
shifted = [value + 10.0 for value in values]
assert abs(mean(shifted) - (mean(values) + 10.0)) < 1e-12
assert mean(values + values) == mean(values)
```

随机输入能扩大覆盖，但必须可复现。使用局部随机生成器并固定种子，保存触发失败的实际输入。只记录种子有时还不够，因为依赖版本或生成流程改变后，输入序列可能不同。发现一个重要随机反例后，最好把它缩小并变成固定回归样例，使未来不依赖重新碰运气。

性质测试也有边界。有限范围并不证明任意长度，随机抽样可能永远错过极稀有形状，参考实现可能也有缺陷。因此报告应写清样例范围与检查性质，而不是把“运行一千次无错”翻译成“数学证明正确”。工程证据可以很强，但必须准确命名。

## 7. 浮点比较：容差不是让失败消失的旋钮

二进制浮点不能精确表示所有十进制小数，因此 `0.1 + 0.2` 与 `0.3` 的内部近似值可能略有差异。对这类数值结果直接要求完全相等，可能拒绝合理计算；但把所有测试都改为近似比较，又会放过整数计数、标识和状态等本应精确的错误。先判断量的语义，再选择比较方式。

绝对容差限制差值大小，相对容差按结果尺度允许误差。接近零时，相对误差可能变得非常大，因此需要绝对项；大数值时只用一个极小绝对阈值可能不合理。容差应来自数值精度、运算长度和应用需求，而不是看失败差多少就随手放宽到刚好通过。

```python
import math
assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-12, abs_tol=1e-12)
assert not math.isclose(1.0, 1.1, rel_tol=1e-12, abs_tol=1e-12)
assert not math.isfinite(float("nan"))
```

还要明确处理 NaN 和 Infinity。NaN 与自己通常也不相等，比较结果可能使普通阈值逻辑失效；若预期结果应为有限值，先检查有限性，再比较误差。不要因为测试工具提供 equal_nan 选项就自动允许 NaN，它可能正是你需要发现的数值爆炸。

数组比较需要同时检查形状和数值。错误的广播可能让某些表达式得到看似合理的结果，却掩盖维度错位。对矩阵乘法，先用不对称尺寸，例如二乘三与三乘四，防止方阵样例把转置错误隐藏起来。再选元素不全相同的矩阵，让顺序与轴的错误有机会表现出来。

## 8. 测试替身：把外部世界变成可控制输入

真实模型 API 有延迟、费用、网络失败和输出随机性。测试 Agent 的解析与状态逻辑时，不需要每次真的访问模型。把客户端作为参数传入，用假客户端返回固定文本，就能精确覆盖成功、非法 JSON、缺字段与超时等分支。这个做法不是假装网络已经验证，而是把当前测试范围明确限定在应用逻辑。

```python
import json

class FixedClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response

def ask_count(client, prompt):
    payload = json.loads(client.generate(prompt))
    if type(payload) is not dict or set(payload) != {"count"}:
        raise ValueError("invalid fields")
    if type(payload["count"]) is not int or payload["count"] < 0:
        raise ValueError("invalid count")
    return payload["count"]

client = FixedClient('{"count":3}')
assert ask_count(client, "count items") == 3
assert client.calls == ["count items"]
```

假对象记录 calls，允许检查调用次数与参数。这能发现无意重复请求；仅返回固定结果而不记录调用，可能无法观察此类副作用。测试替身应模拟接口所需行为，但不应过度复制真实服务内部逻辑，否则你实际上维护了第二套可能同样错误的系统。

替身有多种常见用途：固定返回值、记录调用、故意抛错、提供简化但可工作的内存实现。名字 stub、mock、fake 在不同团队中用法有差异，初学阶段重点理解你需要控制什么、观察什么。不要让术语选择比测试目的更复杂。

如果必须修补某个模块里的函数，patch 位置应是被测代码查找该名字的位置，而不是只看原始定义在哪。模块导入后可能保存自己的引用，这是 F07 名字绑定规则的应用。优先使用显式依赖传参能减少这种技巧性问题，使测试装配更容易阅读。

## 9. 单元、集成、端到端与证据边界

单元测试检查一个相对小的合同，失败时定位快；集成测试检查真实组件如何连接，例如解析结果写入 SQLite；端到端测试从用户入口走完整流程，例如启动本地服务后发 HTTP 请求。三种都重要，不是越接近真实环境越能替代其他类型。一个端到端失败可能有很多原因，小测试帮助迅速缩小范围。

本仓库某些 HTTP 合同通过 dispatch 函数验证：给方法、路径和请求体，检查状态码与响应数据。这证明路由与业务合同，却没有经过真实 socket、HTTP 编码、连接关闭或端口绑定。因此报告应写“dispatch 合同通过”，不能直接声称所有网络传输细节已实测。真正传输需要单独的本地服务演练。

同样，容器模型实验可以解释镜像层与卷，不等于当前环境已经成功构建并运行 Docker。NumPy 参考算子通过，也不等于 CUDA 内核在 H100 上正确或高效。把验证层次写清楚，可以让后续测试补齐真正缺口，而不是因为一个绿色标记误以为所有平台已经覆盖。

外部集成测试应显式选择运行，并有超时、费用和资源边界。默认测试不应该偷偷读取 API key 后发送请求，也不应该启动长期占用 GPU 的任务。用户已有 API 与 H100 资源，意味着可以在适当阶段设计真实验证；它不改变每次测试必须有明确目的和停止条件的要求。

## 10. traceback、断点与最小复现

traceback 展示异常经过的调用链。通常先读最下面的异常类型和消息，再向上找最靠近错误的本项目调用点，检查传入的值。库内部抛错并不意味着库坏了，可能是你的输入违反合同。反过来，也不要在没有证据时假定所有错误都是自己代码问题；最小复现可以帮助区分依赖缺陷与调用错误。

最小复现保留触发错误所需的最少条件。先固定输入和环境，再逐步删除无关部分；每删一部分都重跑，确认错误仍在。最后得到十行脚本，比把整个项目压缩发给别人更便于分析。删除过程中若错误消失，刚删除的部分可能参与触发，也可能改变时序，要重新验证而不是立刻下结论。

打印调试要围绕假设。若怀疑生成器已耗尽，打印类型与消费事件；若怀疑路径错误，打印 cwd 与解析路径；若怀疑字段类型，打印 type 和有限的安全表示。不要把全部用户输入、环境变量和密钥倾倒出来。信息越多不一定越有用，关键是能否区分两个竞争解释。

breakpoint 可以暂停并查看局部变量、调用栈与下一步执行。使用调试器前先有问题，例如“进入这个分支时 size 究竟是什么类型”。否则随机单步可能耗费很多时间却没有结论。断点会改变并发时序，也不适合留在自动化或共享服务中；调试完成后删除或明确禁用。

## 11. 四个故障演练：从现象到可证伪假设

**只在完整测试集失败。** 假设共享全局状态或资源冲突。先单独运行失败测试，再与前一个测试组合运行，观察最小触发集合；检查类属性列表、环境变量修改、当前目录改变和固定临时路径。修复隔离后应验证不同顺序仍通过，不要通过强行排序掩盖依赖。

**本机通过、持续集成失败。** 比较 Python 版本、依赖、工作目录、区域设置、时区、文件大小写和可用权限。一次只验证一个差异。测试不应依赖当前真实日期或某个用户主目录已有文件；把时间与路径作为输入或夹具。若确实需要平台差异，应明确标记并分别验证，而不是用无条件跳过隐藏失败。

**偶尔超时。** 先区分功能死锁、环境慢与阈值过紧。固定 sleep 后断言任务已完成容易受调度影响；更可靠的是用事件或条件同步，并设置宽松的整体上限防止永久挂起。时间性能测试与功能测试分开，避免把毫秒波动当作业务错误。若是真实网络超时，应保存阶段信息，而不是只记录总耗时。

**测试全部通过但用户输入失败。** 将真实失败缩成安全样例，检查它属于哪类未覆盖条件，例如空字段、中文路径、重复请求、生成器输入或错误后状态。先增加能失败的回归测试，再修复。不要只把当前用户数据加进庞大夹具而不解释缺陷类别，否则下一个同类变体仍可能漏掉。

## 12. 十二道练习及完整答案

### 一：把错误均值变成红色测试

```python
import unittest
from labs.foundations.debugging import broken_mean

class BrokenMeanTest(unittest.TestCase):
    def test_definition_of_mean(self):
        self.assertEqual(broken_mean([2, 4, 6]), 4)

suite = unittest.defaultTestLoader.loadTestsFromTestCase(BrokenMeanTest)
result = unittest.TestResult()
suite.run(result)
assert len(result.failures) == 1
assert not result.errors
```

这里在外层断言教学测试确实失败，使整段可以作为可运行演示。内部失败说明错误实现违反均值定义；修复练习应在自己的副本中把被测对象换为正确实现，再确认内部失败数归零。

### 二：检验超额操作后状态

```python
from labs.foundations.objects import Budget
budget = Budget(10)
budget.consume(7)
try:
    budget.consume(4)
except ValueError:
    pass
else:
    raise AssertionError("must reject")
assert (budget.used, budget.remaining) == (7, 3)
```

先消费七让对象处于非初始状态，能发现失败后重置为零等错误修复。只用刚创建的预算测试，可能漏掉“错误时清空历史消费”的缺陷。

### 三：抓住丢尾批错误

```python
def wrong_batches(values, size):
    result = []
    for start in range(0, len(values) - size + 1, size):
        result.append(values[start:start + size])
    return result

source = [0, 1, 2, 3, 4]
actual = wrong_batches(source, 2)
assert actual == [[0, 1], [2, 3]]
assert [item for chunk in actual for item in chunk] != source
```

五个元素按二分批必须留下单元素尾批，错误实现只取完整窗口。这个输入比四个元素更有辨识力，因为四刚好整除时错误版本也会看起来正确。

### 四：重复值必须保留

```python
from labs.foundations.functions import batches
source = [1, 1, 2, 1]
assert batches(source, 3) == [[1, 1, 2], [1]]
assert source == [1, 1, 2, 1]
```

第二条还检查输入未被修改。如果函数合同允许消耗迭代器，输入状态要求就不同；本题针对列表与现有 batches 接口，不能把所有输入类型都机械套同一个断言。

### 五：独立夹具的完整测试

```python
import unittest

class IndependentTests(unittest.TestCase):
    def setUp(self):
        self.values = []

    def test_append_one(self):
        self.values.append(1)
        self.assertEqual(self.values, [1])

    def test_starts_empty(self):
        self.assertEqual(self.values, [])

suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTests)
result = unittest.TestResult()
suite.run(result)
assert result.wasSuccessful()
```

每个测试实例获得自己的初始列表。把列表改为类属性并去掉 setUp 后，再改变执行顺序，观察共享状态如何产生失败；随后恢复正确实现。

### 六：合理浮点比较与有限性

```python
import math

def close_finite(actual, expected):
    return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)

assert close_finite(0.1 + 0.2, 0.3)
assert not close_finite(float("nan"), 0.3)
assert not close_finite(0.4, 0.3)
```

这个容差适合本题小规模双精度演示，不应直接复制到所有低精度矩阵算子。测试文档应说明为什么选这个数量级以及目标计算精度。

### 七：固定随机输入并保留样例

```python
import random
from labs.foundations.debugging import mean

rng = random.Random(2026)
for _ in range(50):
    values = [rng.randint(-10, 10) for _ in range(rng.randint(1, 10))]
    expected = sum(values) / len(values)
    actual = mean(iter(values))
    assert actual == expected, values
```

这里重点检验 mean 接受一次性迭代器，而不是独立证明求和公式。参考值仍使用同一简单数学表达，所以还应保留手算样例；不同测试承担不同证据角色。

### 八：假客户端覆盖坏 JSON

```python
import json

class Client:
    def generate(self):
        return "{"

def parse_response(client):
    return json.loads(client.generate())

try:
    parse_response(Client())
except json.JSONDecodeError:
    pass
else:
    raise AssertionError("invalid JSON must fail")
```

测试完全离线，证明解析失败按预期传播。它没有证明网络连接、认证或模型输出稳定性，验证报告必须保留这条边界。

### 九：临时目录保证资源隔离

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as first, TemporaryDirectory() as second:
    left = Path(first) / "report.txt"
    right = Path(second) / "report.txt"
    left.write_text("left", encoding="utf-8")
    right.write_text("right", encoding="utf-8")
    assert left.read_text(encoding="utf-8") == "left"
    assert right.read_text(encoding="utf-8") == "right"
    assert left != right
```

文件名相同不代表路径相同。独立父目录让两个测试不会因为固定文件名相互覆盖，且清理范围仅限各自资源。

### 十：异常消息只检查稳定信息

```python
from labs.foundations.debugging import mean
try:
    mean([])
except ValueError as error:
    assert "empty" in str(error)
else:
    raise AssertionError("must fail")
```

当消息不是正式外部协议时，逐字匹配完整标点会使无关措辞修改导致测试失败。检查异常类型与必要语义即可；若 API 明确承诺稳定错误代码，应优先检查代码。

### 十一：输入不变性

```python
from labs.foundations.debugging import mean
values = [3, 1, 2]
before = values.copy()
assert mean(values) == 2
assert values == before
```

测试防止某个实现为了计算而原地排序或清空输入。若输入包含嵌套可变结构，浅拷贝不能检测所有深层修改，需要按真实合同保存足够的独立观察值。

### 十二：用一次错误改动检验测试能力

```python
def candidate(values):
    if not values:
        raise ValueError("empty")
    return sum(values) / (len(values) + 1)

cases = [([2, 4, 6], 4), ([9], 9)]
failures = [(values, candidate(values), expected) for values, expected in cases if candidate(values) != expected]
assert len(failures) == 2
```

故意偏移分母让两个样例都失败，说明它们能区分这类缺陷。随后恢复正确实现。不要把错误改动提交到正式参考函数；实验应在个人练习副本中进行。

## 13. 分阶段项目：给日志分析器建立验收证据

第一阶段只测解析：一条合法记录、缺字段、非法数值、中文端点以及空行。每条样例尽可能短，并说明预期是拒绝还是跳过。第二阶段测统计：两三条记录即可手算平均值、错误数与排序，避免直接拿一万行数据当唯一验收。第三阶段测输入输出边界，用临时文件与假 writer。

第四阶段运行真实 CLI 子进程，检查退出码、标准输出与标准错误。不要只看终端文字，因为自动化调用者依赖退出码判断成功；诊断混入标准输出还可能破坏 JSON 管道。第五阶段制造原子保存失败，确认旧报告仍可读取。第六阶段记录未验证的范围，例如真实 Docker、Windows 原生终端和长期大数据性能。

最终交付是一组小测试、一个可复现失败样例和一份简短证据说明。说明包含命令、运行环境、通过的合同以及未运行边界。不需要用“绝对稳定”“全平台兼容”这样的词提高气势；精确说明已知范围，反而让后来的人能继续补充验证而不重复猜测。

## 14. 算法与数据结构补课：结果正确还要在规模内完成

测试首先检查结果，但工程合同可能还包含输入规模与资源上限。如果一份日志从一千行增长到一百万行，程序耗时从一秒变成十几分钟甚至耗尽内存，功能样例通过并不足以满足实际使用。分析复杂度不是为了背公式，而是为了预测输入增长时，哪部分工作会成为限制。

设输入长度为 n。一次遍历，每个元素只做有限次简单操作，工作量通常随 n 线性增长，记为 O(n)。双重循环若对每个元素都扫描全部输入，比较次数接近 n 的平方，记为 O(n²)。这里的大写 O 描述增长的上界级别，忽略常数与较低阶项；它不是某次运行的准确秒数，也不意味着所有线性算法在所有小输入上都比平方算法快。

例如线性查找目标值，最坏情况要检查全部元素；如果目标恰在第一项，很快就能返回。讨论复杂度时要说明最坏、平均还是某种输入分布。二分查找要求数据已有合适排序，每次排除约一半范围，查找步骤随对数增长，但如果每次查找前都重新排序，整个任务还要计入排序成本。不能只引用内部某个步骤的复杂度替代全流程。

列表适合按位置访问和保留顺序，字典适合按键关联值，集合适合成员查询与去重。它们不是谁更高级，而是支持的操作不同。列表中查一个值通常需要逐项比较；字典和集合依赖哈希，通常能较快定位，但消耗额外内存，且键必须满足可哈希条件。最坏情况、实现细节与输入特性仍会影响实际性能。

```python
def contains_linear(values, target):
    checks = 0
    for value in values:
        checks += 1
        if value == target:
            return True, checks
    return False, checks

assert contains_linear([1, 2, 3], 1) == (True, 1)
assert contains_linear([1, 2, 3], 3) == (True, 3)
assert contains_linear([1, 2, 3], 9) == (False, 3)
```

这个计数实验比直接测微秒更容易解释算法增长。输入长度翻倍且目标不存在，比较次数也翻倍。测量时间还受到解释器、缓存、操作系统调度等影响，而操作计数直接暴露结构。当然真实程序不同操作的成本不相同，最终性能仍需要代表性测量；两种方法回答不同问题。

保序去重可以用列表记录已见元素，每次 `item not in seen` 扫描先前列表。若所有元素都不同，第一个检查零项，第二个检查一项，依次增长，总比较数接近 n(n-1)/2。换成集合记录已见键，通常能把成员查询总工作降到接近线性，同时仍用列表保存输出顺序。这是数据结构改变算法成本，不是语法缩写带来的神奇加速。

```python
def unique_ordered(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output

assert unique_ordered([3, 1, 3, 2, 1]) == [3, 1, 2]
```

输出列表与 seen 各有目的：一个保存顺序，一个支持查询。若只返回 set，结果合同改变；若元素是不可哈希列表，就需要定义可哈希键或采用不同策略，不能直接套用。测试应覆盖重复次数、首次出现顺序与不支持类型，明确设计边界。

栈按后进先出组织，适合追踪嵌套或撤销顺序；队列按先进先出组织，适合待处理任务。用 Python 列表尾部 append 和 pop 可以表达栈；反复从列表头部 pop(0) 会移动后续元素，长队列可以使用 collections.deque 的 popleft。选择容器时应看主要操作，不是因为“列表最熟”就让所有任务都用同一种结构。

```python
from collections import deque

pending = deque(["parse", "count", "write"])
order = []
while pending:
    order.append(pending.popleft())
assert order == ["parse", "count", "write"]
```

树和图把关系显式保存。图遍历通常维护待访问队列或栈，以及 visited 集合防止重复访问。没有 visited 时，包含环的图可能永远走不完。后续 Agent 建图不是只会调用图框架就够了，你需要先能手算节点、边、访问集合与待处理容器怎样变化。本章先把这些状态看作普通数据结构，算法卷再逐步增加复杂性。

空间复杂度也要说明计入什么。函数接收已有列表，再创建同样大小的输出，额外内存为 O(n)；如果只维护一个总和与计数，额外工作状态接近常量，但输入列表本身仍占内存。生成器可能减少中间结果，却不能消除为了精确排序或全局去重必须保存的信息。复杂度分析要覆盖整条链路，不能只盯一个 yield。

## 15. 调试时怎样建立可以被推翻的假设

一个好的调试假设包含现象、解释和可区分实验。例如“报告为空，是因为源生成器先被调试打印消费了；如果重新创建生成器，报告应恢复，而对列表则不受影响”。它可以被具体实验推翻。相比之下，“Python 有时候不稳定”既不能定位也不能指导验证，只会促使你随机重启或重装。

把竞争解释并排写出也很有用。文件读不到可能是路径错误、权限不足、编码错误或读到空文件；这些解释有不同证据。cwd 与 resolved path 能检验第一种，异常类型和权限能检验第二种，二进制读取与解码错误能检验第三种，字节大小与内容摘要能检验第四种。一次输出所有环境变量并不能有效区分它们，还可能泄露秘密。

调试日志应记录阶段边界与关键状态，而不是重复打印同一句“到这里了”。例如 parse_started、parsed_count、validation_failed、commit_started、commit_finished 组成可追踪时间线。若只有 commit_started 没有 commit_finished，你知道故障窗口；若记录了 request_id，可以把同一请求的多条事件连起来。时间戳有用，但分布式时钟可能不完全同步，不能只靠时间排序推断因果。

最小复现的缩减过程要保持触发条件。删除并发后竞态消失，不说明原并发一定没问题；把真实文件换成内存字符串后成功，说明差异可能在 IO 或编码边界。每一步缩减都应记录“保留了什么、改变了什么、错误是否仍在”，逐渐缩小原因范围。不要一次删掉半个工程，再因错误消失而失去线索。

修复后需要反向验证假设。若你认为原因是路径相对工作目录，使用明确路径应解决，而改变无关输入不应影响；若你认为原因是列表别名，创建独立对象应解决，而单纯改变量名不应解决。能预测这些对照结果，说明你理解的是机制，不是只找到了一个偶然有效的补丁。

有时错误无法稳定复现。此时先改善可观察性，记录有界的状态和上下文，避免立即大改逻辑。对于共享服务器任务，保留命令、版本、输入摘要和资源限制，有助于之后复现。不要为了诊断而无界保存所有请求体或无限增加日志；观察工具本身也需要数据与存储边界。

## 16. 一个完整的测试小工程：纯核心、文件边界、命令行入口

下面的小工程统计非空文本行数。核心函数接受可迭代文本，文件函数负责打开路径，入口负责参数和退出码。示例刻意保持很小，让每一层的测试范围都能看清。把三层分开不是为了文件数量，而是为了让错误与副作用有明确归属。

```python
from pathlib import Path

def count_nonempty(lines):
    count = 0
    for line in lines:
        if type(line) is not str:
            raise ValueError("text lines required")
        if line.strip():
            count += 1
    return count

def count_file(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return count_nonempty(handle)
```

count_nonempty 对每条字符串去掉首尾空白后判断是否为空。这里空白行不计入，包含中文的行计入。它不负责解释行里的 JSON，不修改原文本，也不要求输入支持 len。count_file 明确拥有自己打开的文件，因此负责关闭。文件错误直接传播给调用者，核心函数不打印任何用户提示。

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

def count_nonempty(lines):
    count = 0
    for line in lines:
        if type(line) is not str:
            raise ValueError("text lines required")
        if line.strip():
            count += 1
    return count

def count_file(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return count_nonempty(handle)

class LineCounterTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(count_nonempty([]), 0)

    def test_whitespace_and_unicode(self):
        self.assertEqual(count_nonempty(["\n", "  \t", "学习\n", "ok"]), 2)

    def test_generator(self):
        self.assertEqual(count_nonempty(line for line in ["a", "", "b"]), 2)

    def test_invalid_line(self):
        with self.assertRaises(ValueError):
            count_nonempty(["a", 3])

    def test_file_boundary(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "中文.txt"
            path.write_text("first\n\n学习\n", encoding="utf-8")
            self.assertEqual(count_file(path), 2)

    def test_missing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.txt"
            with self.assertRaises(FileNotFoundError):
                count_file(path)

suite = unittest.defaultTestLoader.loadTestsFromTestCase(LineCounterTests)
result = unittest.TestResult()
suite.run(result)
assert result.wasSuccessful(), (result.failures, result.errors)
assert result.testsRun == 6
```

六个测试分别回答不同问题。空输入定义边界；空白与中文定义文本规则；生成器验证协议兼容；非法元素验证错误合同；文件边界验证编码与打开流程；缺失文件验证外部失败传播。若六个测试只是换六个不同文件名，却都使用相同内容，就没有这种互补证据。

这份测试还没有验证 CLI 的帮助文本、参数解析、退出码或标准流，因为例子尚未实现 CLI。下一阶段实现后，再通过 subprocess 运行入口，检查真实进程输出。不要让核心函数的成功结果被拿来替代尚未存在的端到端证据。每加一层，就明确新增什么合同、已有测试负责什么。

一个有用的扩展是加入错误行策略：遇到非字符串是否立即失败，还是统计并跳过。两种行为都可能合理，但应分别命名或参数化，并在报告中包含拒绝数量。不能为了测试通过悄悄把 raise 改成 continue，因为这会让调用者误以为所有输入都被处理。测试推动你把业务选择写清楚，而不是替你决定业务选择。

## 17. 质量工具各自能发现什么

格式化工具统一缩进、空格和换行，减少风格争论；静态检查可以发现未使用变量、某些危险写法或明显类型不一致；类型检查器依据标注分析调用关系；单元测试执行具体行为；集成测试验证真实组件连接。它们互补，没有一个工具能独自证明项目正确。格式整齐的代码仍可能算错均值，测试通过的代码也可能泄露密钥或缺少资源上限。

类型标注提供可读合同，但运行时不会自动拒绝所有错误类型。`def add(x: int) -> int` 只是标注，若没有检查，调用者仍可能传入别的对象。静态分析效果依赖标注质量与代码覆盖范围；动态 JSON 数据进入系统时仍需要运行时验证。不要把编辑器没有红线当作外部输入已被安全验证。

代码覆盖率回答哪些行或分支在测试中执行过。执行一行并不意味着检查了它的结果，更不意味着覆盖了所有值与状态组合。一个测试调用函数但没有有意义断言，也可能提高覆盖率。利用覆盖报告寻找未触达的错误路径很有价值，但为了百分比写大量无意义调用，会把工具指标与实际目标混淆。

审查变更时，先看问题与行为变化，再看实现和测试。一个小修复是否引入不相关重构？错误处理是否掩盖了根因？新参数是否改变默认行为？测试是否在错误版本上确实失败？这些问题比只看行数或命名更接近质量。对自己的代码也可以采用同样方法：暂时站在不知道实现细节的调用者角度阅读。

依赖版本影响复现。记录 Python 与关键依赖版本，可以解释不同环境差异；锁定依赖有助于重建，但仍需要更新策略与兼容验证。不要为修复一个测试随手升级所有包，再把成功归因于某个猜测。改变环境属于实验变量，应像修改代码一样可追踪。

持续集成是在干净环境自动运行检查，让每次变更都获得同一套基本证据。它不是云端魔法，只是把你本地应能运行的命令自动化。若本地文档缺少依赖安装或工作目录要求，CI 可能暴露这些隐含前提。修复时把缺失步骤补到工程说明，而不是只在流水线里塞无法解释的命令。

## 18. 测试时钟、随机性与并发：减少不必要的不确定

依赖当前时间的函数容易跨午夜或时区失败。可以把当前时间作为参数传入，或注入一个返回时间的函数。这样测试能精确选择边界时刻，而不必真的等待日期变化。业务入口负责取得真实时间，核心逻辑只处理明确值，和注入模型客户端的思想相同。

```python
def expired(created_at, ttl, now):
    if ttl < 0:
        raise ValueError("ttl must be non-negative")
    return now >= created_at + ttl

assert not expired(100, 10, 109)
assert expired(100, 10, 110)
assert expired(100, 10, 111)
```

等于过期时刻是否算过期，是必须明确的边界选择。本题采用大于等于，因此一百一十已经过期。若测试只用一百和两百，错误地写成大于也能通过；紧贴边界的三个值能区分两种实现。真实时间戳单位也应明确，秒与毫秒混用可能让逻辑完全失真。

随机性可以通过局部生成器注入。不要为了固定一个测试直接重置全局随机状态，影响其他测试；也不要依赖测试运行顺序恰好消费相同数量随机数。保存失败输入比保存“这次跑了很久终于错了”更有价值。对模型输出，固定假响应通常比试图控制远端随机种子更适合核心逻辑测试。

并发测试应通过事件明确协调阶段，而不是猜测 sleep 足够长。你可以让工作线程在一个事件处等待，主测试确认前置状态后再放行；这样竞争窗口可控，超时只作为防止挂死的上限。具体线程与 async 技巧在 F11 展开，本章先建立原则：功能正确性不应依赖测试机器恰好足够快。

取消和失败后清理也需要测试。任务抛错时锁是否释放、文件是否关闭、临时目录是否清理、半成品是否被发布？这些检查往往比成功输出更能反映可靠性。测试时使用短小可控资源，不启动无法停止的后台任务；测试结束后进程不应留下仍在运行的工作线程或服务。

## 19. 写出有用的验证报告与个人调试记录

验证报告先说变更后行为，再说运行了什么以及结果，最后说明重要未验证范围。例如“非法预算在修改状态之前被拒绝；离线单元测试覆盖正常扣减、超额、布尔和负数；未执行多线程共享对象测试”。这比“已全面测试，应该没有问题”更具体，也让下一位读者知道哪里需要补证据。

命令要可以复制，工作目录与环境要明确。若测试失败后修复并重跑，应报告最终结果，并保留必要的根因说明；不必把每次探索过程都写成长日志。若某项测试因为当前环境缺少 Docker 或 GPU 未运行，直接写未运行，不把源码检查冒充运行结果。诚实边界是工程能力的一部分。

个人调试记录可以包含四栏：观察、假设、实验、结论。观察只写实际输出，假设写解释，实验写能区分解释的动作，结论写证据支持程度。避免把猜测混进观察，例如“数据库随机丢数据”其实可能只是你打开了另一个相对路径下的空库；记录真实绝对路径后，解释会完全不同。

每次修复后问三个问题：是什么输入触发，为什么原测试没发现，新增测试如何防止复发。这样你会逐渐积累边界识别能力，而不是只积累一堆特殊补丁。对 Agent、检索和算子同样适用：失败样例、评价标准与可重复证据，比记住某次模型恰好回答正确更有长期价值。

本章最后的通过条件是你能独立制造一个小错误，让测试稳定失败，定位到具体机制，修复后通过，并解释测试仍没有覆盖什么。若只能运行别人提供的命令看到绿色，还没有完成调试训练；若能写清失败前后状态和验证边界，你就已经开始具备修改真实框架代码所需的工作方式。

### 补充审查：五种看起来有断言却没有保护关键行为的测试

第一种是只断言结果为真。若函数应返回包含状态码、数据和错误字段的字典，`assert result` 只证明字典非空，错误字段、错误状态码甚至完全无关内容都可能通过。应选择调用者真正依赖的字段，并检查不该出现的字段是否被排除。断言不必把整个内部对象逐字锁死，但必须覆盖公开合同。

第二种是测试预期直接调用被测函数计算。例如先把 actual 算一遍，再把同一个函数调用一次当 expected，两个结果一致只说明它在当前输入上可重复。对随机函数甚至连这一点都不稳定。预期必须来自独立推导、手算夹具或明确性质；若用参考实现，写明其独立性和适用规模。

第三种是捕获异常后什么也不检查。如果函数根本不抛异常，测试仍然顺利走到末尾，就失去意义。使用 assertRaises 或 try/except/else，让没有抛错本身成为失败。还要防止捕获范围过宽：原本期待 ValueError，实际 NameError 却被 except Exception 吞掉，会让拼写错误伪装成正确拒绝输入。

第四种是只检查最后状态而忽略副作用次数。一个模型请求函数可能重复调用两次，最终答案仍相同，但成本和外部写入次数翻倍。记录调用次数与关键参数，才能发现重复执行。反过来，也不要对所有内部函数调用次数做过细断言，否则合法重构会使测试脆弱；应重点检查外部可见副作用和业务上要求的次数。

第五种是把所有浮动字段直接删除再比较快照。时间戳或随机标识确实需要控制，但如果这些字段本身是合同的一部分，删除它们会制造盲区。更好的方法是注入固定时钟和标识生成器，或者检查格式、唯一性与关系。快照测试适合大段稳定输出，但更新快照前仍要理解差异，不能把接受新快照当作修复。

### 补充手算：一个状态机的测试矩阵如何产生

设任务只有 pending、running、done 三个状态。允许 pending 到 running，running 到 done，其余转换拒绝。测试矩阵有九个起止组合，其中两条成功边、三条原地边、四条其他边。是否允许原地重复，要由幂等合同决定；不能因看起来“没变”就默认成功。先写表，再实现，比在 if 分支里临时决定更清楚。

测试每条拒绝边时，都应先建立对应起点，再调用转换，检查异常和最终状态保持。只从 pending 测所有目标，会遗漏 running 与 done 的行为。若增加 failed 状态，矩阵扩大，新的允许边和恢复路径都应显式审查。测试数量的增长来自状态空间，而不是机械要求每个函数必须十个测试。

进一步加入外部副作用，例如进入 running 时调用工具，完成时保存结果。测试要同时观察状态与调用记录：工具失败时是否仍错误进入 done，重复完成是否再次写入结果，未知状态是否触发任何外部调用。把状态表和副作用表分开记录，能看见“状态没变但重复发出了请求”这类单看最终值发现不了的错误。

本卷的 Budget 是更小的状态机，used 与 limit 定义合法区域；任务服务则把状态保存在数据库里；Agent 以后可能把当前节点、记忆和工具历史一起保存。它们共享同一种测试方法：明确起点、动作、允许结果、失败结果和副作用边界。先在小例子中练熟，你就不会在复杂框架里只依赖一次完整演示来判断正确性。

### 补充实验设计：测得更快之前先证明比较公平

比较两个实现性能时，输入必须相同，输出应先确认一致，计时范围也应相同。一个实现把预处理放在计时外，另一个把预处理算进去，会形成不公平比较。缓存、首次导入、编译和数据加载都可能影响首轮耗时；应按实验目的决定是否包含，并在报告中说明，而不是只挑最快一轮展示。

功能测试通常不应断言某段代码必须在极短毫秒内完成，因为共享机器负载会造成波动。可以检查算法操作计数或设置非常宽松的挂起保护，再用独立基准报告性能分布。未来在 H100 上测试内核时，异步执行与同步边界还会增加新的计时陷阱；当前阶段先建立“计时定义必须明确”的习惯。

最后保留失败输入的最小副本与命令，让未来的自己能在新环境重新验证。若输入含私密数据，构造保持触发结构的脱敏样例，而不是把原始用户记录提交到公开仓库。测试资产应既有诊断价值，也有清楚来源和使用边界；这不是额外装饰，而是使工程能够长期公开维护的基本条件。
