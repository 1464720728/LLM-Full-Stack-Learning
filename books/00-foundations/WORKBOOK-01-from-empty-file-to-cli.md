# 工作簿 A：从空文件开始，亲手长出一个日志分析程序

这不是把F15完整源码再读一遍。你会先写十行程序，亲眼观察变量如何变化；然后把它改成函数，加入类型检查、JSON、文件、错误记录、测试和命令行。每一步只增加一种能力。你应该先在自己的文件中实现，再运行仓库里的阶段参考进行比较。

本工作簿的阶段参考位于`labs/foundations/workshop/step01_loop.py`至`step05_cli.py`。它们是完整可执行的教学阶段，不是未完成占位。最后的工业化程度仍以F15项目合同为准。所有步骤CPU即可，不需要API和GPU。

## A0 建立你的工作台，而不是修改系统目录

从仓库根目录打开终端：

```bash
mkdir -p .practice/workbook-a
python --version
python -c 'from pathlib import Path; print(Path.cwd())'
```

若你的解释器命令是python3，全文对应替换。第一条创建自己的练习目录；第二条证明解释器可执行；第三条证明当前目录。先把输出记录下来。

用编辑器新建`.practice/workbook-a/reporter.py`，只写：

```python
print("hello, course")
```

保存后运行：

```bash
python .practice/workbook-a/reporter.py
```

预期只有一行`hello, course`。如果报找不到文件，先检查保存位置与工作目录；如果输出空白，检查是否保存、是否运行了同名另一个文件；如果看到`>>>`，你可能进入Python交互模式，而不是Shell。提示符不是代码的一部分。

此时你已经建立最小“编辑→保存→运行→观察”闭环。不要跳过这个小步骤直接复制二百行代码：后者失败时，你无法区分环境问题、路径问题和程序问题。

## A1 手工累加三个数，并记录每次状态

把文件改成：

```python
values = [3, 5, 0]
total = 0
for value in values:
    before = total
    total = total + value
    print("before=", before, "value=", value, "after=", total)
print("total=", total)
```

运行前填写表：

| 循环次数 | value | before | after |
|---|---:|---:|---:|
| 第1次 | 3 | 0 | 3 |
| 第2次 | 5 | 3 | 8 |
| 第3次 | 0 | 8 | 8 |

`total = total + value`不是数学等式，它先计算右侧表达式，再把结果绑定到左侧名字。若你把它看成代数等式，容易误以为能两边约掉total推出value=0。程序赋值有时间顺序。

`before=total`让before指向当前整数对象；随后total重新绑定到新整数，before保留旧值。整数不可变，因此这个状态快照可靠。若total换成可变列表，直接赋值不一定形成快照，必须重新分析别名关系。

参考阶段：

```bash
python -m labs.foundations.workshop.step01_loop
```

参考输出：

```text
index value before after
0 3 0 3
1 5 3 8
2 0 8 8
total = 8
```

它还使用enumerate保存索引，并把轨迹作为函数返回值。你暂时不必一次写成同样结构，但必须能把输出每个数字与代码中的一次赋值对应起来。

**停下来做三次改动：** values改为空列表，total应仍为0；改为[7]，输出7；改为[3,-5]，当前版本输出-2。最后一种数值计算没错，但若业务定义tokens不能为负，下一步需要加合同。

## A2 把代码放进函数，弄清参数、局部变量和返回值

改成：

```python
def total_tokens(values):
    total = 0
    for value in values:
        total = total + value
    return total

answer = total_tokens([3, 5, 0])
print(answer)
```

读代码的顺序不等于执行顺序。解释器先执行def语句，创建函数对象，并没有立刻把函数体里的循环跑完。执行`total_tokens(...)`时进入函数，参数values绑定到本次输入列表，total从0开始，最终return把8交给调用者。

局部变量total每次调用重新建立。连续调用`total_tokens([1])`和`total_tokens([2])`分别返回1和2，不是1和3。若结果发生累积，检查是否把状态放在模块全局、对象字段或可变默认参数里。

现在故意把return改成print(total)，再观察answer。函数内部会打印8，外部print(answer)会打印None，因为没有显式return值时函数返回None。这是理解“呈现结果”与“交付数据”的关键实验。

不要只把return改回来就走。写下两个使用场景：别的函数要继续计算时需要return；人要在终端看结果时需要print。可靠工程通常让核心函数return，入口再print。

## A3 增加输入合同，拒绝负数和布尔

```python
def total_tokens(values):
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("token count must be a non-negative integer")
        total += value
    return total
```

这里条件包含两个检查：首先必须是严格的int类型；其次必须不小于0。`or`具有短路行为：前项为真时，后项不再求值。因此value为字符串时，不会继续执行字符串和0比较，而是直接进入raise。

若把两个条件反过来写`value < 0 or type(value) is not int`，字符串会先在比较阶段抛TypeError，用户看到的就不是你设计的ValueError。这不是说所有校验都必须同一种顺序，而是你要理解顺序怎样影响失败类型。

为什么拒绝True？Python中True与1有特殊继承关系，但业务上`tokens=true`很可能是上游格式错误。严格计数合同要把它抓出来，而不是默默当1。

手动执行四组调用：合法[3,5]返回8；[-1]报ValueError；["3"]报ValueError；[True]报ValueError。不要为了“让用户方便”自动把所有输入转int，因为`int(3.7)`会截断、`int(True)`变1、带单位字符串又会失败。转换策略必须来自需求。

## A4 输出多个统计量，不要用位置猜字段

现在不仅要总数，还要记录条数和均值：

```python
def summarize_tokens(values):
    count = 0
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("invalid token count")
        count += 1
        total += value
    mean = total / count if count else None
    return {"count": count, "tokens": total, "mean_tokens": mean}
```

为什么返回dict而不是`[count,total,mean]`？字典把字段意义写进名字，下游不必记“第2项是什么”。返回dataclass也可行；此阶段选dict便于直接转JSON。

`mean = ... if count else None`是条件表达式，不是同时计算两侧。count为0时不会执行除法，因此避免除零。None表示没有可定义均值；0是实际数值，两者不能混用。若数据为空但返回mean=0，后续报表可能误以为该系统平均token消耗为零，掩盖“根本没有样本”。

运行阶段参考：

```bash
python -m labs.foundations.workshop.step02_function
```

输出count=3、tokens=8、mean_tokens约2.6666666666666665。浮点文本长一些并不是失败；若用于用户呈现可以格式化，但内部不要过早round，避免累积误差。

**检查题：** 统计[0,0]和空列表有什么区别？前者count=2、tokens=0、mean=0.0；后者count=0、tokens=0、mean=None。两者都总token为0，但意义不同。

## A5 第一次读取JSON，不要跳过类型观察

在文件里加一个独立小段：

```python
import json
line = '{"request_id":"a","tokens":3}'
row = json.loads(line)
print(type(line).__name__)
print(type(row).__name__)
print(row["request_id"])
print(row["tokens"] + 2)
```

依次输出str、dict、a、5。`json.loads`名字中的s表示从字符串读取，不是从文件路径读取。把路径传给loads只会尝试解析路径文本。若有文件对象，可以逐行读取再loads，或者对整体JSON文件用json.load；JSONL每行一个对象与单个大JSON文件的格式不同。

将tokens改成字符串`"3"`，最后一行会失败，因为字符串加整数没有这里需要的语义。解析器忠实保留输入类型，不会替你理解“看起来像整数”。现在你应该意识到需要一个专门的parse函数来建立内部合同。

参考阶段：

```bash
python -m labs.foundations.workshop.step03_json
```

它输出TokenRecord对象，并打印tokens类型为int。阅读源码，圈出四个边界：顶层结构、字段集合、request_id、tokens。然后不要看文件，自己重写一次。

## A6 用小对象隔开外部数据和内部逻辑

完整解析函数可以写成：

```python
from dataclasses import dataclass
import json

@dataclass(frozen=True)
class TokenRecord:
    request_id: str
    tokens: int

def parse(line):
    row = json.loads(line)
    if not isinstance(row, dict) or set(row) != {"request_id", "tokens"}:
        raise ValueError("exactly request_id and tokens are required")
    if not isinstance(row["request_id"], str) or not row["request_id"].strip():
        raise ValueError("non-empty request_id required")
    if type(row["tokens"]) is not int or row["tokens"] < 0:
        raise ValueError("tokens must be a non-negative integer")
    return TokenRecord(row["request_id"], row["tokens"])
```

`set(row)`对字典迭代键，得到字段名集合。集合相等不要求顺序一致，因此输入JSON字段前后调换不影响合法性。它同时拒绝缺字段和额外字段。

request_id的strip只用于检查是否全空白，当前并没有把strip后的值保存。你应注意“用于校验”和“用于规范化”不同。本阶段保留原ID字符串，因此`"a"`与`" a "`不同；若业务希望统一，需要明确修改并添加测试，不能在别的函数里悄悄strip。

frozen dataclass使普通赋值`record.tokens=5`失败，帮助我们把记录看作经过验证的值。类型注解不是校验，前面的if仍然必要。去掉if再给dataclass传入字符串，它并不会自动转换或拒绝。

## A7 逐行处理坏数据：先画控制流

现在输入是四条文本，最后一条故意坏：

```python
lines = [
    '{"request_id":"a","tokens":3}',
    '{"request_id":"b","tokens":5}',
    '{"request_id":"c","tokens":0}',
    'broken',
]
```

你要实现：合法记录加入统计；坏行记录行号和原因；继续处理后面行。可以把循环画成“取行→空白则跳过→解析→检查重复→成功保存/失败记错→下一行”。先有控制流，再写try。

```python
values = []
errors = []
seen = set()
for line_number, line in enumerate(lines, 1):
    if not line.strip():
        continue
    try:
        record = parse(line)
        if record.request_id in seen:
            raise ValueError("duplicate request_id")
    except ValueError as error:
        errors.append({"line": line_number, "reason": str(error)})
        continue
    seen.add(record.request_id)
    values.append(record.tokens)
```

注意成功状态更新在try/except之后；坏行通过continue跳过它。若把seen.add放到字段校验之前，可能让一条坏记录污染去重状态。控制流的布局表达了“通过验证后才能提交”的顺序。

这个教学阶段捕获ValueError，因为parse的预期错误统一落在这类；完整F15项目定义专门InvalidRecord，避免将别处意外ValueError也当数据错误。你应能解释为什么项目变大后异常分类要更精确，而不是机械照搬小脚本。

运行：

```bash
python -m labs.foundations.workshop.step04_pipeline
```

预期count=3、tokens=8、invalid=1，errors中line=4。具体JSON解析器错误消息可能随Python版本细微变化，测试应检查业务合同和行号，不必把所有底层英文文本锁死。

## A8 把文件看作可迭代的行

先创建输入数据：

```bash
mkdir -p .practice/workbook-a
printf '%s\n' \
  '{"request_id":"a","tokens":3}' \
  '{"request_id":"b","tokens":5}' \
  '{"request_id":"c","tokens":0}' \
  'broken' > .practice/workbook-a/input.jsonl
```

`printf`的一项就是一行，单引号保护JSON双引号不被Shell消耗。用cat确认内容，再运行阶段CLI：

```bash
python -m labs.foundations.workshop.step05_cli --input .practice/workbook-a/input.jsonl
```

你的analyze若只依赖迭代，就可以把文件对象直接传入。无需先`read()`整文件、splitlines，再构造另一个列表。`with open(...) as stream`保证离开作用域后关闭文件，即使解析出现异常也会清理。

同时要准确描述内存：本阶段values列表仍保存所有合法token数，所以不是严格常数内存。如果只算count和sum，可以边读边累计，不保留values；如果要精确分位数，通常需要排序或其他存储策略。决定数据结构要从输出需求反推。

## A9 写第一个测试，不要只比较一张截图

在自己的练习目录中新建`test_reporter.py`。假设你已经把summarize_tokens写在同目录reporter.py：

```python
import unittest
from reporter import summarize_tokens

class ReporterTests(unittest.TestCase):
    def test_regular(self):
        result = summarize_tokens([3, 5, 0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["tokens"], 8)
        self.assertAlmostEqual(result["mean_tokens"], 8 / 3)

    def test_empty(self):
        self.assertEqual(summarize_tokens([]),
                         {"count": 0, "tokens": 0, "mean_tokens": None})

    def test_rejects_boolean(self):
        with self.assertRaises(ValueError):
            summarize_tokens([True])
```

从仓库根运行：

```bash
python -m unittest discover -s .practice/workbook-a -p 'test_*.py' -v
```

如果import reporter失败，检查测试发现的起始目录和文件名；不要把reporter.py随意复制到Python安装目录。模块定位问题应在项目结构层解决。

现在故意把校验改成`isinstance(value,int)`，bool测试应失败。再修复。这一步证明测试真的能抓住你关心的错误，而不是从来只看绿色输出。

## A10 最后才加入命令行适配

argparse把用户输入翻译为Path对象；main接受可选argv，使测试可以直接调用。业务函数不读取sys.argv，也不负责打印。

```python
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path)
args = parser.parse_args(argv)
```

`type=Path`只是把字符串包装成路径，不自动确认存在或可读；打开文件时仍可能失败。入口捕获OSError与UnicodeError，给出input error并退出2。不要在入口捕获所有Exception然后返回成功退出码0。

比较两次运行：默认无参数使用内置数据，方便快速演示；传--input使用明确文件，方便实际工作。默认行为应该在--help中可见，避免用户以为程序分析了某份真实日志。

## A11 从这个小工具走向F15完整版

你现在只有request_id和tokens。F15增加latency_ms、status、分位数、严格模式、atomic_json保存与专门异常。每项都有一个独立需求，不是为了“代码看起来高级”而添加层次。

建议按以下顺序合并：先扩展Record与解析测试；再扩展统计并手算例子；然后加入输出schema_version；最后接CLI和原子保存。每一步跑旧测试，确保新增功能没有破坏已学会的合同。

请实际写一条变更说明：“日志记录新增latency_ms与status，非法延迟拒绝，报告新增nearest-rank分位数；原token总数行为保持，新增空样本与非法浮点测试。”这就是一个可审阅的工程修改，而不是“完善项目”。

## A12 十道诊断题：答案必须含证据

1. **屏幕打印8，但answer是None。** 证据：函数只有print没有return；修复返回值，入口再打印。
2. **同ID重复两次，tokens翻倍。** 证据：缺少seen检查或检查发生在累计之后；修复提交顺序，并测试重复记录不增加总数。
3. **空文件均值除零。** 证据：未定义count=0合同；返回None并测试，不随意返回0。
4. **tokens=true被接受。** 证据：bool是int子类；采用严格整数合同或明确排除bool。
5. **JSON字段换顺序就失败。** 证据：错误地比较键列表顺序；用字段集合比较，除非顺序本来就是协议的一部分。
6. **文件第一行空白，错误行号少1。** 证据：过滤空行后才enumerate；先编号再跳过。
7. **输入文件名带空格失败。** 证据：Shell分成两个参数；给路径加双引号。
8. **日志文字混入JSON。** 证据：正常结果与诊断共用stdout；日志改stderr。
9. **已保存报告忽然只有半份。** 证据：直接覆盖最终文件中断；用同目录临时文件与replace。
10. **函数测试通过但CLI失败。** 证据：边界适配未测试；分别验证参数解析、路径、编码和退出码，不能假定核心函数正确就覆盖全部系统。

完成标准：你能从一个空reporter.py重写这个工具，独立解释每个容器和异常分支，给新需求加测试，再把实现提交到自己的学习分支。请保留你的第一次失败与修复记录，它比一份从未报错的复制品更能证明成长。
