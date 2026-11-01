# F15 项目A：从一条日志到可测试、可复现的分析工具

> 先修F01–F14。入口：`python -m labs.foundations.log_project`。参考实现：`labs/foundations/log_project.py`。本项目只用标准库，默认使用内置合成数据，不读取公司日志、不调用模型、不占GPU。

## 1. 项目需求先写成可验收合同

我们要分析JSONL请求日志，输出有效记录数、无效记录数、token总数、状态计数、平均延迟、p50和p95，以及错误行号。你需要处理坏数据，而不只是对三条干净样例求平均。

输入每条记录恰好四个字段：request_id为非空字符串，latency_ms为有限非负数，tokens为非负整数且不接受布尔，status为ok或error。request_id在文件中唯一。空白行忽略。默认宽容模式记录错误并继续；strict模式遇到第一条错误立即失败。

输出固定schema_version=1。没有有效记录时，均值和分位数为null；状态计数仍包含ok和error两个键，值可以为0。不要因为当前输入没error就省略error键，否则下游每次都要猜输出结构。

这些要求是本项目的设计选择，不是所有日志格式的普遍标准。日志是否允许重复请求ID、错误请求是否计入延迟统计、空白行是否报错，都可能因业务不同而变化。工程能力体现在把选择写清楚并测试，而不是宣称只有一种正确格式。

## 2. 第一阶段：先把数据表示成对象

`Record`使用frozen dataclass：request_id、latency_ms、tokens、status。frozen阻止常规字段重新赋值，帮助我们把解析后的记录看作不可变值；它不等于深度不可变的万能保证，但此处字段都是简单值。

为什么不一直传dict？dict适合接收外部JSON，但内部四处写字符串键容易拼错，字段合同也分散。解析函数把不可信外部表示转换成受控内部对象，之后统计函数可以依赖它满足基本条件。

你先独立写 `parse_record(line)`，只测试一条记录，不急着写CLI。按照顺序做：JSON解析→顶层对象→字段集合→字段类型→字段取值→创建Record。每一步都只做一件事，失败信息描述违反的合同。

```python
from labs.foundations.log_project import InvalidRecord, parse_record

try:
    parse_record("[]")
except InvalidRecord as error:
    assert str(error) == "expected object"
else:
    raise AssertionError("top-level array must fail")
```

一个常见错误是直接`row['tokens']`，当顶层其实为列表时得到不友好的TypeError。另一个错误是把任何异常都当坏JSON，丢失“JSON合法但业务类型错误”的区别。

## 3. 第二阶段：把无效数据与程序bug分开

`InvalidRecord`是本项目预期的数据错误。analyze只捕获它并记录错误；如果代码自身出现IndexError或别的意外bug，不应默默跳过那一行假装数据坏了。过宽捕获会把真实实现缺陷藏在invalid计数里。

JSONDecodeError被parse_record转换为InvalidRecord，并保留异常链。数据类型错误由显式条件转换。strict模式则补充`line N`后重新抛出，使CLI能够把错误定位到原输入行。

错误报告不保存完整原始行，只保存行号和原因。真实日志可能含请求文本、身份信息或密钥；默认最少化错误记录更稳妥。若确实需要隔离坏行，应明确保存位置、访问权限与脱敏策略，而不是自动把所有坏数据复制到公开目录。

## 4. 第三阶段：逐行处理和去重

`for line_number,line in enumerate(lines,1)`使物理行号从1开始。空白行跳过，但后续行号仍对应原文件。若先删除空行再编号，错误位置就与用户打开的文件不一致。

seen集合记录已经接受的request_id。每条记录通过全部字段校验后，再检查重复，然后加入seen和records。重复记录计为invalid，不再次累计tokens。若第一次同ID记录本身非法，未进入seen，后续合法记录仍可接受。这一细节要有意识地定义。

函数接受Iterable[str]，所以可以传打开的文件、列表或生成器。接口不要求先把整个文件读成大字符串。可是精确分位数要保留所有有效延迟，因此整个算法仍是O(n)内存；逐行读取只减少原始文本的额外驻留，不等于全流程常数内存。

## 5. 第四阶段：平均与分位数必须有定义

内置三条有效记录延迟为10、30、20，tokens为3、5、0。均值=(10+30+20)/3=20；token总数8；ok两条，error一条。

分位数有多种定义。本项目采用nearest-rank：排序后取第ceil(p×n)个元素，转换成0起始索引时减1。n=3、p=.5，ceil(1.5)=2，对应20；p=.95，ceil(2.85)=3，对应30。

对四条[10,20,30,40]，nearest-rank p50为20；某些插值定义会得到25。二者不同不说明某个库必然错，而是定义不同。报告字段明确命名为p50_nearest_rank和p95_nearest_rank，防止下游误以为采用了另一种插值。

有限样本中的p95也不是对未来尾延迟的精确保证。三条数据的p95就是最大值，它统计不稳定。后续性能评测卷会讨论样本量与测量策略；基础项目先保证定义和实现一致。

## 6. 第五阶段：把函数接到CLI

运行默认演示：

```bash
python -m labs.foundations.log_project
```

你应看到valid=3、invalid=1、tokens=8，均值20、p50=20、p95=30，错误行号为4。内置第四行是broken，专门训练错误路径。

从文件读取并保存报告：

```bash
mkdir -p .practice/f15
python -m labs.foundations.log_project --output .practice/f15/report.json
python -m json.tool .practice/f15/report.json
```

输入文件使用 `--input 实际路径`；不要照抄中文占位路径。输出通过F08的atomic_json保存，CLI成功后仍把报告打印到stdout。严格模式对默认坏行应返回非零：

```bash
python -m labs.foundations.log_project --strict
```

这里失败是预期行为，不是课程未完成。验收时要区分“测试预计该输入应失败”与“实现意外失败”。

## 7. 第六阶段：系统性测试矩阵

至少覆盖以下组合：空输入、仅空行、单条合法记录、多个状态、坏JSON、顶层数组、缺字段、额外字段、负延迟、NaN、布尔tokens、重复ID、strict遇坏行、分位数边界和输出保存失败。

现有测试验证参考报告、空输入、去重、非法形状和类型、strict行号、分位数定义。核心file_io测试另外验证原子保存失败保留旧值。你应能说明跨模块测试共同覆盖了哪些需求，而不是只看一个测试文件名。

一种强验收方式是故意引入三个bug：让bool作为tokens通过；删除重复ID检查；把ceil改成floor。每次都应有测试失败。若某个bug完全未被发现，说明测试集存在盲区，需要补测试，不是把测试总数堆高。

## 8. 从参考实现到独立实现的任务链

**任务A：最小统计器。** 不看参考，接受三个合法Record，输出count与tokens总数。验收：空列表输出0/0。参考思路sum与len，先不涉及文件。

**任务B：解析边界。** 实现parse_record并补五种非法输入。验收：错误全是明确InvalidRecord；合法记录类型统一。参考：先判断dict和字段集合，再检查值，拒绝bool计数。

**任务C：宽容与严格模式。** 同一输入中夹坏行。验收：宽容模式继续，strict在准确行号失败。参考：只捕获数据错误，strict重抛时加上下文。

**任务D：分位数。** 独立实现nearest-rank，手算n=1、3、4。验收：空输入返回None，p不在(0,1]拒绝。参考：排序、ceil、索引减1。

**任务E：CLI与文件。** 增加--input/--output，错误返回非零。验收：原有函数测试不依赖argparse，stdout始终是报告JSON。参考：业务与适配分开。

**任务F：故障注入。** 模拟输出无法序列化或路径不可写。验收：旧报告保留，错误可见，没有半份最终文件。参考：调用atomic_json并在入口捕获预期异常。

**任务G：扩展为按模型分组。** 新schema加入model字段，并按model分别统计。验收：总体总量等于分组总量之和，未知/缺失模型策略明确。参考：分组后复用统计核心，升级schema，别复制两套算法。

**任务H：大文件优化。** 输入一百万条合成日志，记录峰值内存，解释精确分位数代价。验收：报告必须标注精确或近似算法；若改近似，不能继续沿用原字段假装精确。该任务仅CPU即可完成，先小规模测再放大。

## 9. 你应该交付什么

自己的实现、至少覆盖关键合同的测试、一份运行示例、一份错误示例、一段复杂度说明和一条Git提交。报告中写清数据是合成的、费率或性能数字是否是假设、在哪个Python环境执行。

这个项目已经具备真实工程的骨架：输入边界、领域对象、业务统计、错误合同、持久输出、CLI、测试和版本记录。接下来把其中的核心函数注册为Agent工具，只需增加工具参数适配层；不必重新写统计逻辑。

## 10. 先把一份输入完整走一遍，不跳过任何状态

下面构造七个物理行：空行、合法 a、坏 JSON、重复 a、非法 b、合法 b、合法 c。每行都保留原位置，错误报告应指向原文件行号。非法 b 没有进入 seen，所以后面的合法 b 仍可接受。重复 a 已经通过字段解析，却因唯一性合同被拒绝，不能再次累计 token。

```python
import json
from labs.foundations.log_project import analyze

def line(request_id, latency, tokens, status="ok"):
    return json.dumps({"request_id": request_id, "latency_ms": latency, "tokens": tokens, "status": status})

lines = ["\n", line("a", 10, 3), "broken", line("a", 99, 99), line("b", -1, 2), line("b", 30, 5), line("c", 20, 0, "error")]
report = analyze(lines)
assert report["valid"] == 3
assert report["invalid"] == 3
assert report["tokens"] == 8
assert [error["line"] for error in report["errors"]] == [3, 4, 5]
assert report["latency_ms"]["mean"] == 20
```

| 物理行 | 输入性质 | seen 变化 | 接受记录数 | 错误数 |
|---|---|---|---:|---:|
| 1 | 空白 | 不变 | 0 | 0 |
| 2 | 合法 a | 加入 a | 1 | 0 |
| 3 | 非法 JSON | 不变 | 1 | 1 |
| 4 | 重复 a | 不变 | 1 | 2 |
| 5 | b 延迟负数 | 不变 | 1 | 3 |
| 6 | 合法 b | 加入 b | 2 | 3 |
| 7 | 合法 c，业务状态 error | 加入 c | 3 | 3 |

最后一行的 status=error 是合法业务记录，不等于输入无效。它表示某次被记录请求执行失败，但日志格式正确，因此仍计入有效数、错误状态数和延迟统计。invalid 表示日志合同不成立，两种“错误”必须分开。否则你会把真实服务失败从统计中删掉，得到过于乐观的延迟与成功率。

在 strict 模式中，处理到第三行就抛异常，后面四行不再处理，也不生成最终报告。这不等于前两行的局部内存状态从未存在，而是函数没有返回成功结果。CLI 只有 analyze 成功后才调用 atomic_json，所以严格模式失败不会发布一份看似完整的新报告。这种保证来自具体操作顺序。

## 11. 解析器每一个条件为什么放在这里

json.loads 先负责 JSON 语法，顶层可能是字典、列表、数字、字符串或 null。只有字典符合本项目记录形状，因此必须先检查顶层，再访问字段。字段集合恰好等于四个允许名字，使拼写错误和额外字段都被拒绝。这个严格策略适合教学清楚合同，但未来格式升级需要版本化，不能悄悄添加字段后期待旧解析器接受。

request_id 检查非空字符串并拒绝纯空白，但返回值保留原字符串，不自动 strip。于是 "a" 与 " a " 在当前合同中是不同标识。若你希望标准化，必须明确修改规则并测试重复识别，不能仅在某个分支临时去空白。标识规范化会影响历史数据关联，是语义变化。

latency_ms 允许整数或浮点，但拒绝 bool，因为布尔值在 Python 中具有整数继承关系，却不是合理延迟输入。接着检查非负，并将可表示值转成 float，再确认有限。非常大的 Python 整数转 float 可能抛 OverflowError，参考实现把它翻译成 InvalidRecord，保持“非法延迟属于数据错误”的边界，而不是让一条异常日志终止宽容模式。

tokens 只接受非负整数，因为 token 是计数，不接受三点五。status 只允许 ok 与 error。完成这些步骤后创建 Record，内部统计就不需要每次重新检查同一字段。边界验证的价值是把不确定外部表示转换为明确内部合同，减少后续每层重复猜测。

JSON 解析库、运行时的整数位数限制和输入文件大小仍有独立边界。本项目默认处理正常规模的合成或受控日志，不是对任意不可信超大输入的完整安全网关。若要开放上传，应增加单行字节数、总文件大小、记录数和数值业务范围限制，并分别测试。不要把四个字段检查称作所有资源风险已经解决。

## 12. 去重集合的时机与算法成本

seen 存保存已接受的 request_id。先解析后去重保证非法记录不会占用一个标识；先检查重复再加入保证同一合法记录只统计一次。若把 seen.add 放在字段验证之前，第一条损坏记录就可能阻挡后续修复后的合法记录。若先追加 records 再发现重复，统计结果可能已经污染，需要额外撤销。

成员查询通常利用集合哈希结构，因此整段去重在普通输入下接近线性处理，但 seen 占用与不同合法标识数相关。精确去重本来需要记住历史身份，不能靠生成器语法自动把内存变成常量。若要处理无限流，必须选择有界时间窗口、外部存储或近似去重等不同合同。

records 保存全部合法 Record，latencies 又建立一个延迟列表，nearest_rank 每次还排序一个新列表。当前实现清楚易读，适合基础项目，但会产生多份与有效记录数相关的数据。它调用两次分位数函数，也就重复排序。你可以在理解后优化为排序一次复用，但必须保留相同 nearest-rank 语义和空输入行为。

优化前先写基准输入与正确结果。改变数据结构后，分别验证顺序、重复处理、状态计数、分位数与错误行号。不要因为平均值仍正确就认为优化完全兼容；一个遗漏 duplicate 检查的版本可能在所有唯一 ID 基准中看不出问题。性能样例与功能边界样例应互补。

## 13. 分位数从排序位置手算到测试

对五个延迟一、二、三、四、一百，均值二十二，nearest-rank 中位数三，p95 为一百。平均值被极慢请求拉高，中位数描述中间位置，p95 描述高分位位置。三种统计量回答不同问题，不能只挑一个让服务看起来快的数字。记录样本数量同样重要，小样本尾分位非常不稳定。

nearest-rank 的步骤是复制排序、计算 p 乘样本数、向上取整得到从一起算的排名，再减一转换为 Python 下标。p 必须大于零且不超过一，否则排名可能落在不合理位置。空输入先返回 None，没有可选元素。不要让空列表触发 IndexError 后再捕获成零，那会把“没有样本”伪装成“延迟零”。

```python
from labs.foundations.log_project import nearest_rank
values = [100, 4, 2, 1, 3]
assert nearest_rank(values, 0.5) == 3
assert nearest_rank(values, 0.95) == 100
assert nearest_rank(values, 1.0) == 100
assert values == [100, 4, 2, 1, 3]
assert nearest_rank([], 0.5) is None
```

sorted 返回新列表，所以输入顺序未被修改。这个不变量便于复用调用者数据，但意味着额外分配。若你选择原地 sort，应明确改变函数合同，或先在上层决定数据所有权。不要为了节省一份列表而无声改变其他使用者看到的内容。

分位数不能简单从分组分位数平均得到。两个模型各自 p95 的平均，通常不等于所有请求混合后的 p95，因为分位数依赖整体排序与样本权重。若扩展按模型分组，同时还要总体分位数，需要保留足够原始数据或使用可合并的统计结构。均值可以通过总和与数量合并，分位数没有同样简单的两数公式。

## 14. CLI 的两条路径与原子发布

没有 --input 时，main 调用 sample_lines，使用三条合法记录加一条 broken；有输入路径时，with 打开 UTF-8 文本文件，直接把文件迭代器交给 analyze。这个设计使默认演示不依赖用户数据，真实文件路径又走同一个统计核心。测试可以直接传列表，不必为每个解析样例创建文件。

output 可选。提供时先 atomic_json 保存，再把同一报告打印到 stdout；不提供时只打印。保存失败后不会继续打印成功报告，因为异常进入入口处理。当前原子保存实现保护常见部分写入窗口，但没有承诺所有文件系统断电持久性，F08 的边界仍然适用。

入口捕获 OSError、UnicodeError 与 InvalidRecord，转换成错误提示和非零退出。意外程序缺陷不被全部吞掉，有利于发现真实 bug。参数解析错误由 argparse 处理。不同层的异常有不同来源，调试时应先看错误类别，不要把所有失败都归为日志格式不对。

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from labs.foundations.log_project import main

with TemporaryDirectory() as directory:
    target = Path(directory) / "report.json"
    assert main(["--output", str(target)]) == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    assert (report["valid"], report["invalid"], report["tokens"]) == (3, 1, 8)
```

这段会向标准输出打印报告，是 main 的已定义行为。测试若只关心返回和文件，可以捕获 stdout；若要验证整个进程接口，应使用 subprocess 检查退出码与标准错误。不同层次的测试不能相互冒充，直接调用 main 没有经过操作系统进程启动边界。

## 15. 十二道分级练习与完整答案

### 一：最小统计核

```python
from labs.foundations.log_project import Record

def totals(records):
    records = list(records)
    return {"count": len(records), "tokens": sum(record.tokens for record in records)}
assert totals([]) == {"count": 0, "tokens": 0}
assert totals([Record("a", 10., 3, "ok"), Record("b", 20., 5, "error")]) == {"count": 2, "tokens": 8}
```

这里接收已经验证的 Record，不重复解析 JSON。注意直接手工构造 dataclass 并不会自动执行 parse_record 的验证，调用者必须遵守内部合同。

### 二：合法记录统一延迟类型

```python
from labs.foundations.log_project import parse_record
record = parse_record('{"request_id":"a","latency_ms":10,"tokens":3,"status":"ok"}')
assert type(record.latency_ms) is float
assert record.latency_ms == 10.0
```

统一内部表示减少统计分支，但不能让类型转换悄悄接受非法字符串。字符串 "10" 在当前合同中仍应拒绝。

### 三：布尔计数必须拒绝

```python
from labs.foundations.log_project import InvalidRecord, parse_record
try:
    parse_record('{"request_id":"a","latency_ms":10,"tokens":true,"status":"ok"}')
except InvalidRecord:
    pass
else:
    raise AssertionError("boolean tokens must fail")
```

这题专门防止使用宽松 isinstance(value,int) 导致 True 作为一被接受。JSON 语法合法与领域计数合法是两个检查。

### 四：超大延迟仍属于数据错误

```python
import json
from labs.foundations.log_project import analyze
row = {"request_id": "large", "latency_ms": 10 ** 400, "tokens": 0, "status": "ok"}
report = analyze([json.dumps(row)])
assert report["valid"] == 0
assert report["invalid"] == 1
```

普通有限浮点表示容纳不了该整数，解析器将转换失败归入 InvalidRecord。宽容模式因此可以继续处理后续正常行，测试覆盖了类型转换本身也可能失败的边界。

### 五：重复 ID 不重复累计

```python
import json
from labs.foundations.log_project import analyze
row = {"request_id": "a", "latency_ms": 10, "tokens": 3, "status": "ok"}
report = analyze([json.dumps(row), json.dumps(row)])
assert (report["valid"], report["invalid"], report["tokens"]) == (1, 1, 3)
```

仅检查 invalid 为一还不够，错误实现可能同时把重复 token 加入总量，所以三项一起检查。

### 六：非法首次记录不占用标识

```python
import json
from labs.foundations.log_project import analyze
bad = {"request_id": "a", "latency_ms": -1, "tokens": 3, "status": "ok"}
good = dict(bad, latency_ms=10)
report = analyze([json.dumps(bad), json.dumps(good)])
assert report["valid"] == 1
assert report["invalid"] == 1
assert report["tokens"] == 3
```

good 用新字典，避免修改 bad 造成两行都合法。这个小细节同时检查 F06 的对象共享理解。

### 七：物理行号保留

```python
from labs.foundations.log_project import analyze
report = analyze(["", "\n", "broken"])
assert report["errors"] == [{"line": 3, "reason": "invalid JSON"}]
```

先编号再跳过空行，使错误位置与编辑器一致。若先过滤空行，用户会被错误提示引向第一行。

### 八：严格模式首错定位

```python
from labs.foundations.log_project import InvalidRecord, analyze
try:
    analyze(["\n", "broken", "another broken"], strict=True)
except InvalidRecord as error:
    assert str(error) == "line 2: invalid JSON"
else:
    raise AssertionError("strict mode must fail")
```

首错后不再继续，因此不能期待得到所有错误列表。严格与宽容是两份清楚合同，不是同一个函数偶然不同表现。

### 九：空报告结构稳定

```python
from labs.foundations.log_project import analyze
report = analyze([])
assert report["status_counts"] == {"ok": 0, "error": 0}
assert report["latency_ms"] == {"mean": None, "p50_nearest_rank": None, "p95_nearest_rank": None}
assert report["errors"] == []
```

固定键让下游不必猜测某状态是否出现。None 编码为 JSON null，明确表示没有样本。

### 十：分位数定义不混用

```python
from labs.foundations.log_project import nearest_rank
assert nearest_rank([10, 20, 30, 40], 0.5) == 20
assert nearest_rank([10, 20, 30, 40], 0.95) == 40
```

若另一个库返回中位数二十五，先检查其插值定义，不要立刻判定谁错。报告字段必须说明本项目采用 nearest-rank。

### 十一：生成器输入

```python
from labs.foundations.log_project import analyze, sample_lines
source = (line for line in sample_lines())
report = analyze(source)
assert report["valid"] == 3
assert list(source) == []
```

analyze 消费一次性源后，源耗尽符合迭代器合同。若要再次分析，应重新创建源或重新打开文件，而不是期待自动回到开头。

### 十二：真实 CLI 的失败边界

```python
import subprocess
import sys
result = subprocess.run([sys.executable, "-m", "labs.foundations.log_project", "--strict"], capture_output=True, text=True, timeout=10)
assert result.returncode != 0
assert result.stdout == ""
assert "line 4" in result.stderr
```

默认夹具第四行故意损坏。这个测试证明严格模式通过进程入口失败，并且没有成功 JSON 混入 stdout。

## 16. 三个故障定位与修复验收

**总 token 比预期高。** 先检查重复 ID 是否仍累计，输入状态 error 是否按合同计入，以及你手算是否把同一文件分析了两遍。记录每条接受记录的 ID 与 token 小样例，逐步核对 seen 与 records。不要先修改总和公式，因为加法可能完全正确，错误发生在数据集合选择。

**p95 与另一工具不同。** 对照分位数定义、样本过滤、单位、排序与空值策略。先用四条数据手算，确认差异来自 nearest-rank 与插值，而不是随机更改 ceil。若业务决定采用新定义，要升级报告语义和测试，不能只为了对齐某个数字悄悄改实现。

**报告文件旧内容消失但新内容不完整。** 检查是否绕开 atomic_json 直接用 w 覆盖最终路径，或有另一个写者竞争。用不可序列化对象触发保存失败，确认旧报告保留。程序被强杀与断电属于更强故障模型，需要额外验证，不能由普通异常测试直接推断。

**宽容模式意外中断。** 看异常是否属于 InvalidRecord，还是解析过程中没有翻译的转换错误、输入解码错误或真实程序缺陷。超大延迟转换已在参考实现中明确归类，文件 UTF-8 解码错误则由入口处理，不能简单把任何异常都吞掉计为坏行。先确定边界，再修复对应层。

## 17. 从空文件独立实现：每个阶段只增加一种责任

第一份个人实现只接受已经构造好的 Record 列表，返回 count、tokens 和两个状态计数。不要先写文件读取、命令行和保存，因为它们会让错误来源混在一起。用三条记录手算，确认业务状态 error 仍是一条有效记录。完成后把空列表加入测试，确保两个状态键都存在。

第二阶段增加 parse_record，把外部 JSON 转成 Record。先让一个合法样例通过，再逐个加入顶层数组、缺字段、额外字段、布尔计数和负延迟。每增加一个条件，说明它防止什么错误，而不是把参考实现整段复制过去。特别检查验证顺序：先确认对象形状，再索引字段，避免本应是数据错误的输入触发无关异常。

第三阶段增加逐行循环、物理行号与去重。此时暂时不算分位数，只检查接受集合与错误集合。用前面的七行状态表作为验收，手动列出每轮 seen、records 和 errors。若结果不符，定位到首次状态偏离的那一行，而不是只在最后打印总和。

第四阶段增加均值与 nearest-rank，先独立测试分位数函数，再接入报告。第五阶段增加 strict，确认它在第一条错误处停止并保留异常链。第六阶段才接 CLI 和 atomic_json。这样每层都有独立合同，最后集成失败时可以判断问题在解析、统计还是适配，而不是从几十行入口里猜测。

第七阶段写一份最小回归集，包含正常、空、重复、非法、严格与保存失败。第八阶段在个人分支提交，检查只包含源码、测试和小夹具，不包含真实日志或临时报告。项目不要求一次写出最短代码，要求每一步都能说明为什么正确、如何观察以及换写法会改变什么。

## 18. 聚合为什么有些能合并，有些不能

token 总数可以直接合并各分片总数，状态计数可以逐键相加。均值不能简单平均各分片均值，除非分片样本数相同。一个分片一条延迟一百，另一个分片九条延迟零，各分片均值平均为五十，而整体均值是十。正确合并需要总和与数量，再用总和除总数。

分位数更复杂。一个分片的 p95 只保留一个位置值，丢掉了其他排序信息，无法据此恢复总体排序。两个分片 p95 相同，也可能对应非常不同的总体分布。若以后把日志分到多个进程处理，不能只让每个进程返回 p95 再求平均。精确方案需要保留足够数据或执行外部排序，近似方案则需要可合并摘要及误差说明。

去重也不能随意分片。若相同 request_id 出现在两个分片，各分片内部都认为唯一，直接合并会重复计数。可以按 ID 分区让相同键进入同一处理者，或在合并层再维护全局身份集合；两者都增加设计要求。并行化不是把 for 循环丢进进程池就自动保持业务合同。

错误行号需要全局位置。分片时要记录起始偏移，或让读取器在分片前附带原始行号。若每个分片都从一编号，合并报告无法定位原文件。这里可以看到状态、顺序和身份是数据工程的共同基础，正是 Agent 检索与记忆流水线也要维护的东西。

若扩展按 model 分组，应先定义 model 字段是否必需、空值如何处理、大小写是否标准化，以及 request_id 唯一性是全文件还是每模型。分组总量与总体总量的关系需要测试，报告 schema 应明确新结构。不能在旧四字段解析器里随意忽略新增字段，因为它原本采用严格字段集合合同。

## 19. 精确大文件处理：输入流式与结果流式分开看

本项目逐行读取输入，避免保存整份原始文本，但仍保存 Record、seen、延迟列表与排序结果。有效记录数为 n 时，额外内存随 n 增长；排序时间通常按 n log n 增长。错误列表也可能很大，若全部输入都坏，errors 会随行数增长。资源估算不能只关注成功数据。

一个可行的第一步优化是分离仅需累计的统计与需要全局信息的统计。数量、token 总和、状态数和均值的总和可以在线更新；精确去重仍需身份集合；精确分位数仍需延迟信息。这样减少保存完整 Record 的必要性，但不能把剩余问题说成已解决。每个优化都应列出保留和改变的合同。

错误报告可以设定最多保存多少条详细错误，同时继续累计全部 invalid 数。若采用这个策略，报告必须增加明确的截断标记和省略数量，否则用户可能误以为 errors 数组包含全部错误。这个设计不是简单删掉错误列表，而是改变呈现合同，需要版本和测试。

数值累计也有范围。每条 latency 都有限，许多极大值相加仍可能溢出普通浮点总和。当前项目面向正常请求延迟，未提供任意极端数值下的高精度统计保证；若开放不可信输入，应设定合理业务上限或采用经验证的稳定累计策略，并测试聚合非有限结果。单条验证与整体聚合稳定性是两个层次。

大规模实验先生成小样例验证内容，再按数量逐步放大。记录生成参数、输入摘要、有效比例、运行时间和峰值内存。不要用公司真实日志作为默认训练材料，也不要一次生成巨大文件后才发现字段名错误。CPU 足以完成这个项目，GPU 不会自动加速 JSON 解析与 Python 字典去重。

## 20. 变形测试：让实现重写后仍受合同约束

在没有重复 ID 的前提下，重新排列输入行应保持总量、状态计数与延迟统计，但错误行号会随位置改变。因此不能简单断言整个报告完全相同，而应比较对应的稳定部分。这种区分能训练你知道哪些输出属于内容统计，哪些属于输入位置证据。

把所有合法延迟乘正数 c，均值与 nearest-rank 分位数也应乘 c，token 与状态计数不变。给所有合法 token 加一，总 token 应增加有效记录数，延迟统计不变。加入空白行不改变有效数据统计，却改变后续错误物理行号。每条关系都由合同推导，不需要复制统计实现来生成预期。

```python
import json
from labs.foundations.log_project import analyze
rows = [
    {"request_id": "a", "latency_ms": 10, "tokens": 3, "status": "ok"},
    {"request_id": "b", "latency_ms": 30, "tokens": 5, "status": "error"},
]
base = analyze([json.dumps(row) for row in rows])
scaled = analyze([json.dumps(dict(row, latency_ms=row["latency_ms"] * 2)) for row in rows])
for key in ["mean", "p50_nearest_rank", "p95_nearest_rank"]:
    assert scaled["latency_ms"][key] == 2 * base["latency_ms"][key]
assert scaled["tokens"] == base["tokens"]
assert scaled["status_counts"] == base["status_counts"]
```

这个测试验证缩放关系，但不能替代固定手算样例。例如一个错误实现始终返回零延迟，也可能满足乘二仍零的关系；因此要与前面的具体二十、三十等预期共同使用。多种独立证据相互约束，才比单一性质更有辨识力。

## 21. 最终报告如何让别人复核

交付时先写输入来源与规模，明确合成数据或已获授权的脱敏数据；再写字段合同、重复处理、错误策略与分位数定义；最后给运行命令、代码版本和结果。不要只上传一个漂亮 JSON，却不说明 p95 怎么算、错误请求是否计入或空行如何处理。可复核性来自这些定义。

性能结果要区分文件生成、首次导入、读取解析、排序和写报告。若只报告整个进程耗时，也可以，但要说明包含哪些阶段。重复运行时缓存与系统负载可能影响结果，不能只挑最快一次。当前基础项目的目标是正确工程闭环，性能扩展应在保持合同之后进行。

项目完成门槛是你能不用参考源码，独立解释一条坏行如何从解析失败变成错误报告，解释一条重复行为什么不增加 token，解释严格模式为什么不覆盖旧报告，并手算四条延迟的分位数。再把这套核心作为 Agent 工具时，新增的只是参数与结果适配，不应重新发明统计逻辑。

### 补充验收：报告保存与标准输出的一致性

提供 --output 时，同一份报告既保存到文件又打印到标准输出。测试可以分别解析两者并比较结构，确认没有某个路径遗漏字段或使用不同统计定义。文本缩进不一定需要逐字相同，但解码后的语义应一致。若保存失败，标准输出不应出现一份让自动化调用者误认成功的结果。

```python
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    output = Path(directory) / "report.json"
    result = subprocess.run([sys.executable, "-m", "labs.foundations.log_project", "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout) == json.loads(output.read_text(encoding="utf-8"))
```

再做一次严格模式保护实验：先写入一份已知旧报告，随后用默认坏行与 --strict、--output 运行，确认退出非零且旧文件字节不变。这个测试连接解析失败、入口控制流与持久化边界，比单独测试三个函数更接近用户真实使用路径。

```python
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    output = Path(directory) / "report.json"
    old = b'{"old":true}\n'
    output.write_bytes(old)
    result = subprocess.run([sys.executable, "-m", "labs.foundations.log_project", "--strict", "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert result.stdout == ""
    assert output.read_bytes() == old
```

最后检查错误报告是否包含不必要原始内容。输入中放入一个仅用于测试的标记字符串，触发字段错误，确认 errors 只保存行号与原因，而不是完整行。这个检查不能替代全面隐私审查，但能保护当前最小输出合同，防止未来为了方便调试把整段日志无意加入公开报告。

代码阅读的最终练习是从 main 开始，画出 parser、analyze、parse_record、nearest_rank 与 atomic_json 的调用关系，并在每条边旁标明数据类型。你应能指出哪些函数只计算，哪个函数拥有打开文件的生命周期，哪个函数决定用户错误退出。若某个函数同时解析参数、修改统计、写文件又打印日志，尝试拆分责任并用已有测试确认行为保持。

扩展时保留这张关系图与合同表。新增字段、分组或近似分位数，都应说明改变了哪条边上的数据结构、哪些测试需要更新、旧报告消费者如何处理。工程成长不是不断把更多 if 塞进 main，而是在明确边界内增加能力，使后来继续深入 Agent 工具调用时仍能理解和复用核心逻辑。
