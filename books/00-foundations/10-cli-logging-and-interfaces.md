# F10 命令行、配置、日志与接口：让程序能被人和其他程序可靠使用

> 先修 F06–F09。实验：`python -m labs.foundations.cli --count 7 --batch-size 3`。本章分三轮：先读懂现有入口，再做可测试的小工具，最后增加配置与诊断。所有示例默认离线，写文件只使用自己的练习目录。

## 1. 命令行是一个协议边界

终端命令最终让操作系统启动程序，并把参数交给它。Python 通常通过 sys.argv 看到字符串列表。这里没有自动的整数、布尔或路径类型：你输入三，程序收到的可能是字符串 "3"。入口层需要把外部文字翻译成业务所需类型，验证范围，然后才调用核心函数。命令行因此不是给函数随便套一层 print，而是一份独立接口。

在 `python -m labs.foundations.cli --count 7 --batch-size 3` 中，Shell 先处理引用等语法，Python 解释器识别 `-m` 与模块名，剩余参数再由课程 CLI 解析。不能把 Python 解释器参数与应用参数混为一谈。若把 `--count` 放在模块名之前，可能被解释器当作自己的选项而拒绝，业务代码甚至还没开始执行。

```python
import argparse

parser = argparse.ArgumentParser(description="Show parsed values")
parser.add_argument("--count", type=int, default=7)
args = parser.parse_args(["--count", "3"])
assert args.count == 3
assert type(args.count) is int
```

parse_args 接受一个字符串列表，方便测试不依赖真实终端。返回 Namespace 对象，点号读取解析后的属性。`--count` 是外部选项名，count 是内部属性名；带连字符的选项通常转换为下划线属性，例如 `--batch-size` 对应 batch_size。这个映射是解析器约定，不是 Python 允许变量名包含减号。

argparse 提供帮助、缺失参数提示、类型转换和常见错误退出。它减少重复工作，但不会替你知道业务允许什么范围。type=int 能拒绝 abc，却仍接受负数与零；若批大小必须正整数，还需要额外验证。入口应尽早拒绝不合法输入，避免进入循环后才出现除零或无法终止。

## 2. 逐行读仓库入口

仓库 positive 函数先调用 int，把文字转换为整数。转换失败时捕获 ValueError，再抛 ArgumentTypeError，告诉 argparse 这是某个命令行值不合法。转换成功后检查是否大于零。两类失败消息分别说明“不是整数”和“不是正数”，让用户知道该修改什么。

parser 函数建立 ArgumentParser，添加 count 和 batch_size，默认分别为七与三。main 接收可选 argv，将其交给 parse_args；传 None 时解析真实进程参数，测试时传列表。随后构造 range，调用纯业务函数 batches，把结果编码为 JSON 打印，返回零。模块入口用 `raise SystemExit(main())` 将返回值作为进程退出状态。

```bash
python -m labs.foundations.cli --help
python -m labs.foundations.cli --count 5 --batch-size 2
python -m labs.foundations.cli --count abc
```

第一条展示帮助，第二条应输出五个数字分为二、二、一三批，第三条应在解析阶段失败并返回非零状态。错误示例是故意的，不应把它混入要求全部成功的自动脚本。观察第三条时分别记录标准输出与标准错误，确认没有输出半份成功 JSON。

range 的停止端不包含在内，因此 count 为五得到零到四。batches 不知道 count 参数来自终端，也不负责打印；它只接收数据与批大小。这样核心函数可以被测试、HTTP 服务或 Agent 工具复用，入口只是其中一个适配器。改变入口参数不应要求重新实现分批算法。

## 3. 位置参数、可选参数与布尔开关

位置参数依赖出现位置，例如输入文件路径；可选参数通常以短横线开头，例如 `--output report.json`。是否必填由解析器配置决定，而不是“有两个横线就一定可以省略”。选择时考虑使用频率与歧义：一个必需输入路径适合位置参数，多个容易混淆的数值更适合有名字的选项。

布尔开关不能简单使用 type=bool。因为 Python 中非空字符串 "false" 转成 bool 仍是真，用户会得到与直觉相反的结果。常见做法是 action="store_true"，省略为假，提供开关为真；若需要明确正反选项，要在帮助中写清。外部配置字符串也需要显式解析允许词汇，而不是直接 bool(text)。

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("--limit", type=int, default=10)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args(["data.jsonl", "--limit", "0", "--verbose"])
assert args.input_path == "data.jsonl"
assert args.limit == 0
assert args.verbose is True
```

这里 limit 为零是否允许取决于业务。本例只演示解析，尚未赋予“零表示不处理”还是“零表示不限量”的含义。真正工具应明确语义，最好避免同一个特殊值在不同模块代表相反行为。带单位的名字能降低误用，如 timeout_seconds 与 max_bytes，比含糊的 timeout 和 size 更容易审查。

路径包含空格时，Shell 引用先保证它作为一个参数传入；argparse 不负责恢复已被 Shell 拆散的词。以短横线开头的文件名还可能被误认成选项，可以使用明确路径或解析器支持的选项结束标记。不要把路径拼成 shell 命令字符串再执行，固定 argv 列表更容易保持参数边界。

## 4. 返回值、退出码与标准流不是同一件事

Python 函数返回值可以是列表或字典，进程退出码通常是一个小整数，标准输出则是一串字节或文本。这三种通道用途不同。把函数 return 的字典直接当退出码不会自动成为 JSON；把错误消息打印出来也不会自动让退出码变成失败。入口必须明确协调三者。

约定零表示成功，非零表示失败，自动化系统可以据此决定后续动作。具体非零值的细分要稳定且有文档，不必为每个内部异常发明一个新数字。argparse 常用自己的参数错误状态；业务错误可以在 main 中转成清楚消息和约定状态。不要在核心计算函数中随处 sys.exit，因为调用者可能想捕获错误并继续处理其他任务。

标准输出应保留给主要结果，标准错误用于诊断。一个输出 JSON 的工具若先打印“开始处理”，下游 JSON 解析器会看到无效开头。把日志送到 stderr，让 stdout 保持一份完整机器可读结果，是管道组合的基础。stderr 有输出不一定代表进程失败，警告也可能写那里；退出码才是状态判断的重要信号。

```python
import io
import json

def emit_result(value, stdout, stderr):
    print("processing completed", file=stderr)
    print(json.dumps(value, ensure_ascii=False), file=stdout)
    return 0

stdout = io.StringIO()
stderr = io.StringIO()
assert emit_result({"count": 3}, stdout, stderr) == 0
assert json.loads(stdout.getvalue()) == {"count": 3}
assert stderr.getvalue() == "processing completed\n"
```

StringIO 是内存文本流，具有 write 等接口。测试传入它，就能观察标准流合同而不捕获真实终端。这里再次使用依赖注入：核心输出函数不必知道流最终连接终端、文件还是测试缓冲。外部入口负责选择实际 sys.stdout 与 sys.stderr。

## 5. 错误翻译应该靠近用户边界

低层函数应抛出能解释原因的异常，入口把预期业务错误转成用户能采取行动的提示。例如文件不存在可以显示输入路径，JSON 错误可以显示行号与错误类别，参数范围错误可以指出选项名。程序缺陷不应被伪装成“没有记录”，否则用户可能把错误结果继续交给后续流程。

try 范围要尽量小。若把参数解析、读取、统计、输出全部塞进一个 except ValueError，输出序列化错误也可能被误标为输入非法。可以在各层保留语义，入口只捕获已经定义的业务异常；其他意外错误保留堆栈或由统一边界记录并非零退出。调试模式可以增加细节，但默认提示仍应简明。

错误消息不需要包含完整原始记录。日志行可能有个人数据，配置可能有密钥。用文件名、行号、字段名、错误代码与有限上下文通常足以定位。若确实需要原始样例，应在受控个人环境中处理，不能因为“调试方便”就把所有请求体写到公开仓库或长期共享日志。

失败时是否输出部分结果也必须明确。统计工具可以选择全有或全无，也可以输出包含 errors 的结构化报告，但不能一半输出成功数组后突然追加错误文字。JSONL 流式结果可以逐行提交，但这与单份 JSON 文档是不同接口。调用者必须知道如何判断一个流是否完整结束。

## 6. 配置合并：缺省与显式零的区别

一个工具可能有默认值、配置文件、环境变量和命令行四个来源。真正重要的是来源优先级清楚，并且每个来源经过相同业务验证。例如默认小于文件，文件小于环境，环境小于命令行。若优先级隐藏在不同文件里，用户修改配置后看不到效果，就会不断猜测到底哪份值在生效。

“没有提供”与“提供了零”不能混淆。写 `value or default` 会把零、空字符串和 False 都当成缺省，可能吞掉用户明确选择。可以用 None 或独立 sentinel 表示未提供；若 None 本身也是合法值，就必须使用不会与业务数据冲突的标记。argparse 的默认配置也要配合，否则默认值可能在合并时覆盖文件设置。

```python
MISSING = object()

def merge_config(defaults, file_values, cli_values):
    result = dict(defaults)
    result.update(file_values)
    for name, value in cli_values.items():
        if value is not MISSING:
            result[name] = value
    return result

result = merge_config(
    {"limit": 10, "verbose": False},
    {"limit": 5},
    {"limit": 0, "verbose": MISSING},
)
assert result == {"limit": 0, "verbose": False}
```

合并完成后再进行统一验证，能发现跨字段关系，例如并发数不能超过预算允许的任务数。单个来源合法不代表组合后合法。错误提示最好指出最终值及来源，但敏感字段只显示已设置或未设置。有效配置快照应足以复现实验，又不包含凭证。

文件名不会自动使配置生效。task.toml、config.json 或 .env 只有被程序或框架明确读取才参与运行。后缀决定不了加载时机与优先级。遇到“我改了配置但没变化”，先确认程序打开的绝对路径、加载入口和覆盖来源，而不是同时复制多份同名文件到各个目录。

## 7. 日志事件：让以后的人能还原发生了什么

日志不是把所有变量打印出来，而是记录重要事件及上下文。每条事件应尽量回答哪次运行、哪个任务、处于什么阶段、发生什么、结果怎样。request_id 将同一请求跨多行联系起来，run_id 区分不同进程运行，阶段名帮助定位失败窗口。没有这些信息，多个并发请求的日志交错后很难解释。

Python logging 将日志记录、格式化和输出目标分开。logger 发出事件，handler 决定送到终端还是文件，formatter 决定显示形状，level 控制哪些级别被保留。初学时先建立一个清楚配置，不在每个模块重复 basicConfig，也不随意添加相同 handler 导致每条日志出现多次。

```python
import io
import logging

buffer = io.StringIO()
logger = logging.getLogger("lesson.cli.demo")
logger.handlers.clear()
logger.propagate = False
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(buffer)
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(handler)
try:
    logger.debug("hidden detail")
    logger.info("count=%s", 3)
    assert buffer.getvalue() == "INFO count=3\n"
finally:
    logger.removeHandler(handler)
    handler.close()
```

本例使用独立 logger 并清理自己添加的 handler，避免污染后续练习。DEBUG 事件低于 INFO 门槛，因此不出现。日志模板加参数让 logging 在需要时格式化；不要为了记录一个对象而执行有副作用的昂贵函数。日志调用应帮助观察程序，不应改变业务行为。

级别表达严重程度和处理意图。INFO 记录主要过程，WARNING 表示异常但仍可继续，ERROR 表示某项操作失败，DEBUG 保存开发细节。所有事情都写 ERROR 会让真正失败淹没；只记录成功又会让故障没有线索。级别是应用合同，需要团队一致理解，而不是越高越重要就越值得乱用。

## 8. JSONL、字段单位与时间

JSONL 通常每行保存一个完整 JSON 值，工程日志常选对象。它不是一个跨多行的大数组，因此可以逐行追加和读取。字符串中的换行应由 JSON 编码器转义，使一条事件仍占一个物理行。不要手工拼 JSON 字符串，引号、反斜线和中文处理很容易出错。

```python
import json

event = {"request_id": "r1", "latency_ms": 250.0, "tokens": 3, "status": "ok"}
line = json.dumps(event, ensure_ascii=False, allow_nan=False)
assert json.loads(line) == event
assert "\n" not in line
```

字段名字应带必要单位。latency_ms 与 latency_seconds 清楚区分，tokens 表示数量，bytes 表示字节而不是字符。二百五十毫秒等于零点二五秒，在忽略其他开销且完全串行的理想条件下，对应每秒四次；这不是并发服务吞吐保证。单位与前提不清楚，数字再精确也可能误导。

事件时间戳回答“何时发生”，耗时回答“持续多久”。墙上时钟适合与现实日期联系，却可能被时间同步调整；单调计时器更适合求持续时间。使用 perf_counter 前后差值测一段 CPU 工作，不要用格式化时间字符串相减。GPU 异步任务需要不同同步边界，当前示例不能直接推断 GPU 实际执行时间。

可复现日志应避免不必要随机排序。报告里的字典或端点列表若对外承诺稳定次序，就显式排序。随机种子只能控制相应随机生成器，不能固定文件遍历、网络响应和并发调度。不要把“设置了种子”写成整个系统完全确定的证据。

## 9. 输出版本与兼容性

机器读取的 JSON 报告是一份协议。字段名、类型、含义与缺失规则都是合同。新增可选字段通常比重命名字段容易兼容，但某些严格消费者会拒绝未知字段，因此仍应说明策略。把 tokens 从输出 token 改为输入加输出总量，即使字段名和类型没变，也改变语义，属于重要兼容变化。

schema_version 让消费者知道按哪套规则解释。版本号不是装饰，解析器需要对不支持的版本明确处理。没有数据时平均值用 null 或其他明确约定，不能随意写零，因为零是一个有意义的数值。错误应放在专门结构中，不要让同一个数字字段有时变成错误字符串。

```python
report = {
    "schema_version": 1,
    "count": 0,
    "mean_latency_ms": None,
    "errors": [],
}
assert report["mean_latency_ms"] is None
```

帮助文本也是接口。每个选项要说明默认值、单位、是否联网、是否写文件和资源影响。尤其后续 H100 实验必须明确 device、输入规模、迭代次数与停止条件；有资源可用不代表默认无限运行。CLI 的责任之一就是把重要成本暴露成使用者能理解的选择。

版本管理应区分代码版本与数据格式版本。Git 提交标识帮助复现实现，schema_version 帮助解释输出，两者不能互相替代。一次代码修复可能不改变格式，一次字段语义变化则需要格式策略。报告中同时记录适量的实现版本和非敏感配置，会让长期实验更容易比较。

## 10. 四个诊断案例

**参数明明写了却没生效。** 先打印非敏感有效配置及来源，检查命令是否传到正确入口，选项名是否拼错，是否被后续环境变量或文件覆盖。若 parser 默认值覆盖了文件值，使用明确未提供标记修复。不要通过把同一值写进所有配置源来掩盖优先级错误。

**JSON 管道解析失败。** 分别重定向 stdout 与 stderr，查看成功数据是否夹杂日志或进度条。修复输出通道，而不是让下游跳过所有“看起来不像 JSON”的行，因为那可能顺带丢掉真正损坏数据。一个结果通道应该有明确格式，诊断通道则承担人类可读说明。

**日志重复两次。** 检查 handler 是否重复添加，以及事件是否传播到祖先 logger 后再次输出。每次导入模块都配置日志容易造成这种现象。把配置放在入口，库模块只获取自己的 logger；测试清理自己添加的处理器，避免跨测试污染。

**失败却返回成功状态。** 检查 main 是否捕获错误后仍 return 0，模块入口是否真正使用 main 的返回值。打印错误不等于设置失败。用 subprocess 检查 returncode，再检查标准流，才能验证自动化使用者真正看到的合同。

## 11. 十二道练习与完整答案

### 一：加入起始数字

```python
import argparse
from labs.foundations.functions import batches

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--count", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=2)
args = parser.parse_args(["--start", "10"])
assert batches(range(args.start, args.start + args.count), args.batch_size) == [[10, 11], [12, 13], [14]]
```

起点改变只影响输入构造，batches 无需知道业务含义。正式入口还要复用正数验证；本题集中验证范围停止端必须是 start 加 count，而不是直接把 count 当停止值。

### 二：验证参数错误的进程结果

```python
import json
import subprocess
import sys

ok = subprocess.run([sys.executable, "-m", "labs.foundations.cli", "--count", "5", "--batch-size", "2"], capture_output=True, text=True, timeout=10)
assert ok.returncode == 0
assert json.loads(ok.stdout) == [[0, 1], [2, 3], [4]]
bad = subprocess.run([sys.executable, "-m", "labs.foundations.cli", "--count", "abc"], capture_output=True, text=True, timeout=10)
assert bad.returncode != 0
assert bad.stdout == ""
assert "integer required" in bad.stderr
```

使用当前解释器避免测试安装与运行环境不一致，固定 argv 不经过 Shell。这个测试经过真实进程入口，比只调用 positive 多覆盖了参数解析、退出状态与输出通道。

### 三：布尔开关

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
assert parser.parse_args([]).dry_run is False
assert parser.parse_args(["--dry-run"]).dry_run is True
```

dry-run 的业务含义仍需实现：通常应展示计划而不执行写入。仅解析出 True 不会自动阻止副作用，后续代码必须据此选择行为并有测试。

### 四：保留显式零

```python
def choose(value, default):
    return default if value is None else value
assert choose(None, 10) == 10
assert choose(0, 10) == 0
assert choose(False, True) is False
```

这个接口把 None 专用于未提供，因此可以保留其他假值。若业务允许 None 作为真实配置，就要换独立 sentinel。

### 五：密钥诊断摘要

```python
def safe_config(config):
    return {"model": config.get("model"), "api_key_set": bool(config.get("api_key"))}
secret = "example-private-value"
summary = safe_config({"model": "demo", "api_key": secret})
assert summary == {"model": "demo", "api_key_set": True}
assert secret not in repr(summary)
```

不要显示前后若干字符并声称已经充分脱敏；很多诊断只需要知道是否设置。环境变量名字可以公开，实际值不能随意进入报告。

### 六：结构化日志保持一行

```python
import json
value = {"message": "first\nsecond", "status": "ok"}
line = json.dumps(value, ensure_ascii=False)
assert len(line.splitlines()) == 1
assert json.loads(line)["message"] == "first\nsecond"
```

编码器把内容换行转义，因此物理日志仍是一行。解码后恢复原始文本，说明格式边界与内容语义可以同时保持。

### 七：单位换算

```python
latency_ms = 250
seconds = latency_ms / 1000
ideal_serial_rate = 1 / seconds
assert seconds == 0.25
assert ideal_serial_rate == 4
```

解释必须包含串行、忽略其他开销的前提。真实吞吐还受批处理、并发、排队与资源争用影响，不能只倒数一个平均延迟就宣布容量。

### 八：稳定排序

```python
counts = {"/z": 2, "/a": 2, "/b": 3}
ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
assert ordered == [("/b", 3), ("/a", 2), ("/z", 2)]
```

lambda 接收一对键值，返回排序键：负计数使较大计数先出现，名字决定同分顺序。写成普通命名函数也完全可以，重点是排序合同清楚。

### 九：未知配置键应拒绝

```python
def validate_keys(config):
    unknown = set(config) - {"limit", "verbose"}
    if unknown:
        raise ValueError("unknown keys: " + ",".join(sorted(unknown)))
    return dict(config)
assert validate_keys({"limit": 3}) == {"limit": 3}
try:
    validate_keys({"limti": 3})
except ValueError:
    pass
else:
    raise AssertionError("typo must fail")
```

严格键检查能发现拼写错误。若你决定允许未知字段以支持兼容，必须明确记录这一选择，并保证关键错误不会被静默忽略。

### 十：空统计不伪装成零

```python
def mean_or_none(values):
    values = list(values)
    return None if not values else sum(values) / len(values)
assert mean_or_none([]) is None
assert mean_or_none([0, 0]) == 0
```

没有样本与样本均为零是不同状态。输出 JSON 时前者成为 null，消费者据此决定显示“无数据”而不是“零毫秒”。

### 十一：输出与诊断分离

```python
import io
import json
output = io.StringIO()
diagnostics = io.StringIO()
print("warning: demo", file=diagnostics)
print(json.dumps({"ok": True}), file=output)
assert json.loads(output.getvalue()) == {"ok": True}
assert "warning" in diagnostics.getvalue()
```

在真实入口把两个流换成 sys.stdout 与 sys.stderr。不要在库函数里硬编码终端输出，否则其他入口难以复用和测试。

### 十二：配置快照记录来源

```python
def resolve_limit(default, file_value=None, cli_value=None):
    value, source = default, "default"
    if file_value is not None:
        value, source = file_value, "file"
    if cli_value is not None:
        value, source = cli_value, "cli"
    if type(value) is not int or value < 0:
        raise ValueError("non-negative limit required")
    return {"limit": value, "source": source}
assert resolve_limit(10, 5, 0) == {"limit": 0, "source": "cli"}
```

来源信息能解释为何修改文件未生效，因为命令行具有更高优先级。若字段敏感，只记录来源与设置状态，不记录值。

## 12. 分阶段项目：一个可测试的分批工具入口

第一阶段直接调用现有 batches，保持核心算法不变。第二阶段解析 start、count、batch-size，写清默认值与范围。第三阶段把成功结果作为唯一 stdout JSON，参数错误交给 argparse。第四阶段加入可选诊断，输出到 stderr，并用 StringIO 或 subprocess 检查通道隔离。第五阶段加入非敏感配置快照。

```python
import argparse
import json
import sys
from labs.foundations.functions import batches

def nonnegative(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("integer required") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value

def positive(text):
    value = nonnegative(text)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value

def main(argv=None, stdout=None, stderr=None):
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = argparse.ArgumentParser(description="Create numbered batches without network access")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=nonnegative, default=5)
    parser.add_argument("--batch-size", type=positive, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    result = batches(range(args.start, args.start + args.count), args.batch_size)
    if args.verbose:
        print(f"created {len(result)} batches", file=stderr)
    print(json.dumps({"schema_version": 1, "batches": result}), file=stdout)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

项目允许 count 为零，因此使用 nonnegative，而 batch-size 仍必须正数。不同参数即使都为整数，也可能有不同范围合同。stdout 的默认处理用 is None，而不是 `stdout or sys.stdout`，避免把某个合法但假值的对象误当缺省。最终结果包含 schema_version，为以后增加字段留下明确协议入口。

main 的流参数只影响业务输出，argparse 自己的帮助和错误仍按其默认流行为工作；若要把解析器输出也注入，需要进一步封装并测试，不能声称这段代码已经完全控制所有流。指出这种细节不是削弱实现，而是让接口边界精确。用户帮助、参数错误、成功输出、诊断输出分别都有可观察路径。

最终验收从新终端运行帮助、正常输入、零数量、非法批大小和带诊断五组命令。保存退出码与两个输出通道的结果，确认零数量输出空批次列表，非法批大小不产生成功 JSON，诊断模式仍能让 stdout 被 JSON 解析器读取。完成后，你拥有的不只是能跑的脚本，而是能被自动化流程可靠调用的工具。

## 13. 从调用者角度审查接口：可预测比选项多更重要

一个好工具不需要一开始支持几十个选项。先确定最常见工作流：输入在哪里，输出到哪里，失败怎么知道，如何查看帮助，如何重复同一次运行。每增加一个选项，就增加组合状态和测试责任。若某参数只为一个内部实现细节存在，而用户无法据此做有意义选择，可以先留在实现中，不把复杂度全部转嫁给使用者。

默认值应安全且可解释。对离线文本工具，默认不覆盖已有重要文件可能合理；对性能实验，默认小规模和有限迭代便于先验证环境。用户明确要求 H100 时可以使用 H100，但仍应让运行时长、显存规模和停止方法清楚可见。资源边界是接口的一部分，不是运行出问题以后才加上的补救说明。

如果工具支持覆盖文件，选项名称与帮助应说明覆盖发生在哪里以及是否原子保存。不要让 `--output` 在某些路径追加、某些路径覆盖，却没有文档。输出目录不存在时是自动创建还是失败，也需要一致规则。入口可以提供便利，但不能因为“方便”就递归删除调用者已有目录。

批处理工具还要定义部分失败策略。十个输入中一个损坏，是全部失败，还是输出九个结果并报告一个错误？退出码是否仍为成功？调用者需要通过结构化字段知道处理数、成功数和拒绝数。若你选择容错，必须保留错误证据；若选择严格模式，必须避免在失败之前发布一份被误认为完整的最终报告。

命令行与 HTTP API 可以共享业务函数，却不应完全共享呈现方式。命令行把错误写 stderr 并返回退出码，HTTP 把错误映射为状态码和响应体。核心函数不应该根据调用场景直接打印或发响应头。把适配器分开后，业务测试保持稳定，入口测试分别覆盖各自协议。

## 14. 配置验证的第二层：范围、关联与来源

类型转换只回答“能否表示成这个类型”。范围验证回答“这个值是否允许”，跨字段验证回答“这些值组合是否合理”。例如超时和最大重试都为正数，但超时乘次数可能远超总预算；输入批大小和总数量都合法，却可能导致极大的内存需求。不要把每个参数单独合法误当成整个运行计划合理。

可以先把外部配置转换为一份明确的配置对象，再由一个验证函数集中检查。这样 CLI、文件和环境变量入口都复用相同规则，避免命令行拒绝负数但配置文件却允许。验证完成后尽量避免在深层函数中重新读取环境变量，因为那会让配置在运行中发生隐式变化，破坏可复现性。

环境变量都是字符串或不存在，空字符串与不存在也可能不同。布尔值应允许一组明确词汇，例如 true 与 false，并拒绝拼写错误，而不是任意非空都算真。数值变量要处理空白、整数格式和范围。报错时可以显示变量名，密钥变量则只说明未设置或无效，不打印原值。

配置来源还可以帮助解释默认行为。一个运行摘要可以列出批大小来自命令行，日志级别来自配置文件，模型名来自默认值。对于排查“为什么结果与昨天不同”，来源与版本往往比一张最终数字表更有用。但摘要应只包含必要字段，不把整个环境复制下来，以免混入无关秘密和不可复现噪声。

当配置格式升级时，旧字段不要被静默忽略。可以支持明确别名并警告弃用，也可以拒绝并给迁移提示。若把 timeout 从秒改成毫秒却保留相同字段名，旧文件可能仍解析成功但行为严重变化。字段语义变化应有版本或新名字，不能依赖用户猜到你的新解释。

## 15. 日志的生命周期、体积与上下文

日志会持续增长，长期工具需要考虑保留与轮换。轮换可以按大小或时间分文件，保留一定数量旧文件；具体策略取决于运行环境与运维要求。初学项目先把日志输出到 stderr，让外层工具决定收集位置，通常比每个库模块自行打开固定文件更简单。不要把所有日志写到仓库根目录并忘记排除版本提交。

日志上下文需要跟随任务，而不是使用一个全局可变 request_id。并发请求会交错，若每次调用都改同一个全局变量，日志可能串到别的请求。可以显式把上下文字段传入日志函数，或者在理解生命周期后使用合适的上下文机制。当前阶段显式参数更容易追踪，复杂封装应建立在已经明确的需求之上。

错误日志应包含异常类别与阶段。捕获异常后仅写 `error occurred`，虽然有 ERROR 级别却几乎没有诊断价值；完整堆栈又可能过长且含敏感路径或数据。可以在受控诊断模式记录堆栈，默认输出稳定错误代码和安全上下文。日志策略需要同时考虑可排查性、数据边界和阅读成本。

不要在每个层级重复记录同一个异常。底层记录一次、业务层再记录、入口再记录，会造成三个看似不同的故障。通常在最了解上下文且决定处理策略的边界记录一次，其他层通过异常链传递信息。若确需多层事件，应给出不同阶段含义，而不是重复整段相同堆栈。

结构化日志字段应保持类型稳定。latency_ms 一会儿数字一会儿 "timeout"，统计程序就必须到处特殊处理。超时可以用独立 status 与 error_code 表达，耗时字段保持数字或约定的 null。将机器可分析的数据与人类可读消息分开，有助于以后自动分析课程项目和 Agent 运行轨迹。

## 16. 参数传递与子进程：接口边界不能在下一层丢失

一个 CLI 有时需要调用另一个程序。应优先传固定 argv 列表给 subprocess，而不是把用户参数拼接成 shell 命令。字符串里的空格、引号、分号和通配符在 Shell 中有语法意义；若你本来想把它们作为文件名数据，却启动 Shell 重新解释，参数边界就被改变。F03 已经解释引用规则，这里将其用于程序接口设计。

```python
import subprocess
import sys

text = "a b; c"
result = subprocess.run(
    [sys.executable, "-c", "import sys; print(sys.argv[1])", text],
    capture_output=True,
    text=True,
    timeout=10,
    check=True,
)
assert result.stdout == "a b; c\n"
```

分号在这里作为一个参数内部的普通字符传入，没有启动第二条 Shell 命令。check=True 让非零退出变成异常，timeout 防止无限等待，capture_output 让父进程可以决定如何呈现结果。这些选项各解决一个边界问题，不能用其中一个替代其他几个。

子程序成功退出也不代表输出符合你的协议。若你要求 JSON，还要解析并验证字段；若输出很大，全部 capture 可能占用较多内存，需要流式或文件方案。调用者应限制输入、输出和等待时间，并定义失败时如何保留诊断。一个可组合工具既要做好自己的接口，也要审查它依赖的下游接口。

标准输入同样是一种合同。若支持从 stdin 读取 JSONL，要明确它与文件路径参数是否互斥，读到 EOF 才结束还是按条返回结果。不要让一个工具在没有输入时默默永久等待，却在帮助中没有说明。交互终端与管道的行为可以不同，但必须可预测，否则自动化脚本容易挂住。

## 17. 项目终检：五条可复制的使用路径

第一条路径是新用户查看帮助：不需要 API key，不读取大型文件，不启动 GPU，就能知道如何使用。第二条是最小成功输入：输出可以独立解析，退出码为零。第三条是明显非法输入：尽早失败，消息指出参数或字段，标准输出没有假成功结果。第四条是可恢复外部失败：保留旧文件或已有状态，提示下一步诊断方向。第五条是自动化调用：用固定 argv 和受控流运行，得到可测试结果。

把这五条路径写入自己的验收脚本，比只演示一次漂亮输出更能证明工具可用。每次修改参数、配置优先级或输出结构后，重跑相关路径。若修改仅影响帮助文字，不需要重做全部性能实验；若修改错误映射，则必须检查退出码与通道。验证范围应随真实变化选择，而不是机械地每次运行所有昂贵任务。

最终你应该能解释一份输入从 Shell 字符串进入 argparse，变成类型化参数，再进入纯函数，形成结果，最后被编码到标准输出的全过程。任何一步失败，都有明确传播路径和用户可理解的边界。这个能力会直接迁移到 Agent 工具定义、模型 API 适配器和训练脚本入口，使你不只会调用框架，也能为自己的工程设计可靠接口。

还有一个实际验收是从含空格和中文的目录运行你的个人工具副本。这里要验证的是路径参数被当成一个值、文件编码明确、输出协议不依赖当前工作目录。若工具要求从仓库根目录执行，应在帮助或文档中清楚说明；若工具承诺任意目录可用，就应通过安装包或明确资源定位实现，而不是靠用户猜测。不同承诺对应不同实现与测试，不能只因自己的终端里成功就泛化到全部使用方式。

发布入口前，再读一遍所有默认值和错误消息，确认没有真实密钥、私人路径或公司内部地址被硬编码进示例。示例应使用占位符和可控本地数据，使其他学习者能够独立复现。对于确实需要联网的扩展练习，帮助应说明调用何种服务、凭证从哪里读取以及如何限制次数；这能让使用者在开始前理解成本，而不是运行后才发现脚本做了额外工作。
