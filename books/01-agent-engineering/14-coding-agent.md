# A14 编码 Agent 综合项目：从失败测试到可解释、可审查的修复

编码 Agent 的交付是实际仓库中的可验证修改，不是一段看起来像代码的回答。本章把读取任务、建立环境、定位实现、复现失败、提出补丁、应用修改、运行测试、处理失败和最终报告连成完整闭环。你将先用确定性脚本完成一个临时项目，再构造错误修复、过期编辑和测试篡改反例，最后学习怎样把同样方法用于真实开源代码。

所有主实验只使用自己创建的临时目录和可信测试，CPU 即可运行，不占 H100，不调用真实 API。ScriptedModel 提供预设动作，证明宿主与工具协同正确，不代表模型已经具备代码推理能力。真实模型接入放在闭环稳定之后，并用明确调用预算与独立 verifier 评估。

## 第一课：把用户需求改写成可验证合同

需求是计算税后总价，价格为 price，税率为 rate，公式应为 price 乘一加 rate。错误实现用了减号，相当于折扣。成功标准包括正常税率得到正确总价、零税率保持原价、修改只涉及目标源文件、测试没有被削弱、最终说明与实际验证一致。先写这些条件，才知道 Agent 应完成什么。

不要一开始就让模型“优化整个项目”。范围越模糊，越容易引入无关重构、依赖升级或接口变化。一个好的任务说明给出当前行为、期望行为、已知复现和约束；若信息不足，Agent 先通过只读工具补齐。编码能力包含准确缩小问题，而不是一次生成最多代码。

```python
buggy_source = "def total(price, rate):\n    return price * (1 - rate)\n"
namespace = {}
exec(buggy_source, namespace)
assert namespace["total"](100,0.2) == 80
assert namespace["total"](100,0) == 100
```

这里 exec 只执行本章固定可信字符串，用来直观看公式，不用于任意模型代码沙箱。两个输入已经说明零税率测试单独不足以发现错误，因为加减零都相同。测试应选择能区分候选实现的输入，而不是只覆盖容易通过的例子。后面独立测试还会加入更多价格与税率，避免硬编码一百二十。

需求也可能有未定义范围，例如负价格、税率超过一或货币精度。不要擅自添加业务限制或声称覆盖所有财务场景。本章是普通浮点算术教学，真实金额系统应按具体业务定义数值类型、舍入与范围。当前修复只处理明确的符号错误，范围说明保持准确。

## 第二课：先读仓库规则，再定位相关代码

真实仓库开始工作时，先读项目规则与运行说明，查看当前工作区已有改动，再用文件搜索定位函数、调用者和测试。`rg --files` 了解目录，`rg 'def total|total\('` 查定义与使用，阅读相关上下文。不要只看命中的一行就修改，因为函数可能被多个接口调用，文档也可能规定特殊语义。

搜索结果是线索，不是结论。相同函数名可能出现在示例、旧版本或测试夹具，真正执行路径可能通过包装器。应沿入口到调用链确认：用户请求进入哪个函数，参数怎样传递，结果在哪里使用，测试导入哪个模块。A08 已强调文件路径与实际执行模块可能不同，这里把它应用到代码定位。

已有用户修改应保留。不要为了得到干净基线直接 reset 或删除未跟踪文件；先识别任务相关与无关改动，必要时使用独立工作副本。修改范围应由任务决定，根目录配置、依赖锁文件和共享接口尤其谨慎。可逆编辑通常可以直接完成，但破坏性清理或外部发布不是普通修复步骤。

对初学者，建议每次阅读记录四件事：函数输入、输出、调用者和副作用。输入不仅是参数，还可能有全局配置、环境变量、当前目录和文件状态；输出不仅是 return，还可能写文件或数据库。这个习惯帮助你看懂真实框架核心组件，而不只记函数名字。

## 第三课：复现失败建立证据基线

在修改前运行最小相关测试，确认它确实失败，并记录失败输入、期望与实际。若测试已经通过，说明复现条件不完整或任务描述过期，不能把后续无意义改动称为修复。失败基线还帮助防止改错文件：如果测试始终不受修改影响，需要检查导入路径和环境。

```python
from pathlib import Path
import tempfile
from labs.agents.tools import run_trusted_fixture_tests

TEST_SOURCE = '''import unittest
from price import total
class PriceTests(unittest.TestCase):
    def test_tax(self):
        self.assertAlmostEqual(total(100,0.2),120)
    def test_zero(self):
        self.assertEqual(total(100,0),100)
'''
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    baseline = run_trusted_fixture_tests(root)
    assert baseline["exit_code"] != 0
    assert "FAILED" in baseline["output"]
```

包装器使用当前 Python 解释器、隔离导入选项、明确工作目录与超时，返回退出码、受限日志和截断标志。测试本身由本章创建且可信。真实仓库测试可能写数据库、访问网络或启动服务，不能一概当纯读取；需要隔离与可重置环境。命令名叫 test 不提供安全保证。

初始失败也要分类。断言期望一百二十却得到八十，直接指向行为；ModuleNotFoundError 可能是环境依赖；语法错误可能是代码解析；零测试发现则是测试入口问题。不要把所有非零退出都解释成目标 bug 已复现，必须确认失败原因与任务相关。

## 第四课：读文件得到原文与版本，再提出最小补丁

Workspace.read 返回 path、text 和 sha256。模型或开发者应基于实际原文提出替换，而不是凭记忆生成整个文件。expected_sha256 在提交前再次核对，发现文件已变就拒绝。old 必须恰好出现一次，避免修改错误位置。A08 已完整解释这些机制，本章将它们接入 Agent 循环。

```python
import hashlib
import json
from labs.agents.tools import Workspace
from labs.agents.contracts import ToolCall, Turn, Tool, object_schema
from labs.agents.runtime import Harness, ScriptedModel
from labs.agents.store import Store

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    workspace = Workspace(root)
    observed = workspace.read("price.py")
    assert observed["sha256"] == hashlib.sha256(buggy_source.encode()).hexdigest()
    edit = {"path":"price.py","old":"1 - rate","new":"1 + rate","expected_sha256":observed["sha256"]}
    workspace.replace(**{ "relative":edit["path"], "old":edit["old"], "new":edit["new"], "expected_sha256":edit["expected_sha256"] })
    assert "1 + rate" in workspace.read("price.py")["text"]
```

最小补丁不是永远只改一个字符，而是只改变解决已确认问题所需范围。若接口设计确实有缺陷，可能需要多个文件与测试；但每处改动都应有理由和验证。无关格式化、重命名和依赖升级会增加审查负担，也让失败归因困难。先修问题，再根据明确任务决定是否重构。

补丁还应保留周围语义。税率公式的加号修复不需要改变函数名、参数或返回类型；调用者无需调整。若你修改了接口，就必须查所有调用者并更新契约测试。Agent 不能只看目标文件通过一个测试就宣布整个项目兼容。

## 第五课：完整四轮 Agent 轨迹

课程 coding_project 的脚本模型依次发出 read_file、replace_text、test_fixture，再返回最终说明。每轮工具结果通过 call_id 回到历史，Harness 保存状态与账本。四次模型调用、三次工具调用是这个固定脚本的预期，不是所有编码任务的标准次数。

```python
from labs.agents.capstones import coding_project

with tempfile.TemporaryDirectory() as folder:
    result = coding_project(folder)
    assert result["before_exit"] != 0 and result["after_exit"] == 0
    assert result["status"] == "finished" and result["success"] is True
    assert result["model_calls"] == 4 and result["tool_calls"] == 3
    assert "1 + rate" in result["final_code"]
```

第一轮读文件提供观察，第二轮编辑使用观察摘要，第三轮运行受控测试，第四轮解释已验证结果。最后 coding_project 在 Agent 之外再次运行测试，独立计算 success，模型的“已修复”不参与评分。这是完整编码闭环的核心：自然语言报告必须服从实际环境证据。

源码中的 sha 在脚本构造时根据已知 buggy 字符串计算，这是确定性 fixture 的便利，不是模型真的从工具结果学习了摘要。真实模型应读取第一轮结果并在第二轮参数中使用返回值。若要评估模型能力，必须用真实适配器和任务，而不能把脚本预先知道正确补丁当作智能推理。

测试工具在这个可信临时夹具中注册为无业务副作用，便于重复执行；真实测试可能有外部效果，应重新定义工具合同。Workspace 本身也不是 OS 沙箱，不能把任意陌生代码放进当前进程权限下执行。教学项目的完整性指机制闭环，不代表生产隔离已全部实现。

## 第六课：错误修复方向也要进入测试

模型可能把公式改成 price 加 rate，恰好看起来“用了加号”，却不符合需求。也可能硬编码一百二十，只通过一个样本。独立 verifier 应覆盖多个输入，并检查修改没有破坏零税率。测试的作用是区分候选行为，不是证明模型解释听起来合理。

```python
WRONG_SOURCE = "def total(price, rate):\n    return price + rate\n"
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(WRONG_SOURCE,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    checked = run_trusted_fixture_tests(root)
    assert checked["exit_code"] != 0
```

若测试失败，下一轮应读取具体断言、实际输出和相关源码，形成新的假设。不要机械地把整段日志丢给模型并说“继续修”，更不要无条件增加调用预算。错误可能来自公式、导入缓存、环境或测试入口，必须先定位层次。A13 的第一次偏离分析同样适用于编码轨迹。

修复后仍需理解代码。测试只覆盖有限样本，用户自学目标是能解释每一行，不能因为绿灯就跳过原理。对于这个公式，你应能手算价格二百、税率十分之一得到二百二十，并解释为什么先算一加税率。对于复杂算法，则需要形状、边界、不变量与复杂度分析，其他卷将继续训练。

## 第七课：多轮修正必须重新读取版本

第一次错误补丁改变了文件，第二次修正不能复用最初 sha。正确流程是重新读取当前内容与摘要，再提出新替换。否则版本检查会拒绝，保护你不在过期观察上覆盖新状态。模型若收到 file changed 错误，应更新观察，不应尝试绕过摘要验证。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    workspace = Workspace(root)
    first = workspace.read("price.py")
    workspace.replace("price.py","price * (1 - rate)","price + rate",first["sha256"])
    assert run_trusted_fixture_tests(root)["exit_code"] != 0
    current = workspace.read("price.py")
    assert current["sha256"] != first["sha256"]
    workspace.replace("price.py","price + rate","price * (1 + rate)",current["sha256"])
    assert run_trusted_fixture_tests(root)["exit_code"] == 0
```

这段实验没有把错误补丁藏起来。真实学习记录应保留假设、修改、测试结果和纠正原因，帮助你认识错误机制。最终 PR 描述通常聚焦最终问题与修复，不必复述所有探索，但内部 trace 可以保留诊断证据。对用户交付说明实际改了什么、为什么正确、怎样验证即可。

多轮修正还需要停止条件。相同失败反复出现、补丁来回振荡、预算耗尽或环境无法确认时，应停止并给出已知事实与待解决项。不能为了显示完成而删除测试或改需求。一个诚实的未完成报告比未经验证的“全部修好”更有价值，后续可以根据具体阻塞继续学习。

## 第八课：环境问题与代码问题分开处理

课程开发曾遇到同一秒内等长源码修改后旧字节码被复用，表现为文件已改但测试仍按旧逻辑执行。当前夹具使用全新临时目录，并让测试进程带 -B，避免初次测试写入缓存。-B 不保证忽略所有已经存在的旧缓存，因此“全新目录”也是实验前提，不能删掉前提后套用结论。

诊断导入问题时，检查实际解释器、模块文件路径、源码摘要、工作目录和依赖版本。测试导入的是安装包还是当前源码，可能与直觉不同。修改工作区文件却运行了系统环境旧包，测试当然不反映补丁。不要靠反复修改逻辑碰运气，先证明执行的是哪份代码。

依赖缺失应按项目说明建立环境，不随意删除 import 或跳过失败测试。依赖版本冲突需要理解约束，不能无条件升级所有包。环境变化应记录在报告中，因为它可能影响行为与可复现性。若任务只要求代码修复，额外改变依赖锁文件要有具体必要性。

缓存清理也要限定范围。可以清理本任务生成的受控缓存或建立新隔离环境，不能执行不加范围的删除命令，更不能影响公司共享服务器其他任务。Agent 基础实验使用 CPU 即可，不应为了修一个 Python 函数占用多卡 H100 或改变全局驱动环境。

## 第九课：保护测试与限制编辑范围

工作区根目录限制只说明不能访问根外，不说明根内哪些文件可写。若测试和源码在同一根内，普通 replace_text 可以修改测试。编码任务需要更细政策：源码可编辑，最终 verifier 只读或位于独立受保护环境，配置与依赖修改按任务授权。模型不能通过删除断言获得通过。

```python
class SourceOnlyWorkspace:
    def __init__(self, workspace, writable):
        self.workspace = workspace
        self.writable = frozenset(writable)
    def read(self, path):
        return self.workspace.read(path)
    def replace(self, path, old, new, expected_sha256):
        if path not in self.writable:
            raise PermissionError("path is not writable for this task")
        return self.workspace.replace(path,old,new,expected_sha256)

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    guarded = SourceOnlyWorkspace(Workspace(root),{"price.py"})
    observed = guarded.read("test_price.py")
    try:
        guarded.replace("test_price.py","120","80",observed["sha256"])
    except PermissionError:
        pass
    else:
        raise AssertionError("test modification permitted")
    assert (root / "test_price.py").read_text(encoding="utf-8") == TEST_SOURCE
```

这个包装器使用精确路径白名单，只适合宿主已经规范化的简单课程路径。真实系统要避免同一文件通过不同路径别名绕过规则，仍需 A08 的解析与并发边界。它也不阻止其他任意 shell 工具写测试，所以所有工具入口必须遵守同一政策。权限不是给一个函数加 if 就覆盖整个系统。

保护 verifier 不意味着公开测试不能新增。修复者可以增加合理回归测试，但最终验收仍由独立标准判断。新增测试应表达需求与反例，不只是复制实现公式计算预期，否则实现和测试可能一起错。对复杂项目，审查测试本身与审查补丁同样重要。

## 第十课：四个编码 Agent 失败复盘

故障一，模型声称修好，实际文件没变。原因可能是只输出代码块，没有调用编辑工具，或编辑因摘要冲突失败却被忽略。修复 verifier 检查实际文件内容、diff 与测试，最终文本不作为成功证据。测试构造最终回答“已完成”但不编辑，必须失败。

故障二，测试绿了，因为发现零项。原因是测试路径、文件命名或发现配置被改变。修复检查预期测试身份与数量，在受保护环境运行冻结测试。退出码零只说明命令按约定结束，不证明目标测试执行。A13 已给出这类反例。

故障三，修改正确但测试仍旧行为。原因是导入了另一份包或缓存旧字节码。修复先确认模块路径与源码摘要，再处理受控缓存或环境。不要把环境问题错误归因于模型不会加法，也不要为了通过而硬编码结果。

故障四，局部测试通过，调用者因接口变化失败。原因是只阅读目标函数，未查使用位置与契约。修复沿调用链检查参数与返回值，运行相关集成测试，并保持最小必要接口变化。文本 diff 很小也可能影响很大，风险由语义决定，不只由行数决定。

## 第十一课：十二道完整练习与答案

练习一，验证错误公式的两个边界。答案税率二成返回八十而非一百二十，零税率仍一百。解释为什么只有零税率测试无法区分加减号，是测试设计的基本训练。

```python
ns = {}
exec(buggy_source,ns)
assert ns["total"](100,0.2) == 80
assert ns["total"](100,0) == 100
```

练习二，正确公式不能硬编码。答案用三个不同价格税率检查，使用浮点容差。这里不引入真实金融舍入合同，只验证课程算术关系。

```python
correct = {}
exec("def total(price, rate):\n    return price * (1 + rate)\n",correct)
for price,rate,expected in [(100,0.2,120),(200,0.1,220),(50,0,50)]:
    assert abs(correct["total"](price,rate)-expected) < 1e-9
```

练习三，最终文字不能替代编辑。答案运行只说完成的脚本，独立测试仍失败。状态 finished 表示模型结束，不代表编码任务完成。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    h = Harness(ScriptedModel([Turn("Fixed everything.")]),[])
    s = h.run(h.new("false-coding-claim","fix"))
    assert s.status == "finished" and run_trusted_fixture_tests(root)["exit_code"] != 0
    h.store.close()
```

练习四，旧摘要必须拒绝第二次编辑。答案第一次修复后仍用旧摘要请求另一修改，异常后正确源码保持不变。不能通过刷新摘要而不重新理解内容绕过冲突。

```python
from labs.agents.contracts import ContractError
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    workspace = Workspace(root)
    old = workspace.read("price.py")
    workspace.replace("price.py","1 - rate","1 + rate",old["sha256"])
    try:
        workspace.replace("price.py","1 + rate","1",old["sha256"])
    except ContractError:
        pass
    else:
        raise AssertionError("stale edit accepted")
    assert "1 + rate" in workspace.read("price.py")["text"]
```

练习五，替换片段出现两次必须拒绝。答案构造两个函数都含相同表达式，要求模型扩大上下文而不是默认改第一个。这个测试覆盖目标定位，不涉及真实业务公式。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "x.py").write_text("a = 1 - rate\nb = 1 - rate\n",encoding="utf-8")
    workspace = Workspace(root)
    old = workspace.read("x.py")
    try:
        workspace.replace("x.py","1 - rate","1 + rate",old["sha256"])
    except ContractError:
        pass
    else:
        raise AssertionError("ambiguous edit accepted")
```

练习六，独立 verifier 检查测试文件未改。答案运行课程项目后比较测试源码中的关键断言仍存在。更完整生产验收应比较受保护测试摘要，本题展示最小机制。

```python
with tempfile.TemporaryDirectory() as folder:
    result = coding_project(folder)
    test_text = (Path(folder) / "test_price.py").read_text(encoding="utf-8")
    assert "120" in test_text and "test_zero" in test_text and result["success"]
```

练习七，错误修复 price+rate 必须失败。答案使用同一测试，避免为了新实现改变验收。失败日志应被交给下一轮诊断，而不是忽略。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "price.py").write_text(WRONG_SOURCE,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    assert run_trusted_fixture_tests(root)["exit_code"] != 0
```

练习八，工具结果中测试退出码要严格解释。答案只接受整数零，不把字符串零或布尔 False 当作通过。接口 schema 应在工具层保证类型，verifier 仍可明确检查。

```python
def tests_passed(result):
    return type(result.get("exit_code")) is int and result["exit_code"] == 0
assert tests_passed({"exit_code":0})
assert not tests_passed({"exit_code":"0"})
assert not tests_passed({"exit_code":False})
```

练习九，报告不能声称未执行测试通过。答案比较报告测试集合与宿主记录集合，报告只能引用已运行且通过的项。尚未运行项应单独列出。

```python
def validate_reported_tests(reported, observed_passed):
    if not set(reported) <= set(observed_passed):
        raise ContractError("report invents test evidence")
    return True
assert validate_reported_tests(["tax"],{"tax","zero"})
try:
    validate_reported_tests(["integration"],{"tax","zero"})
except ContractError:
    pass
else:
    raise AssertionError("unrun test reported as passed")
```

练习十，编辑范围必须与任务允许集合一致。答案比较实际改动路径，任何额外文件都触发审查。真实项目可以允许必要新增测试，但应在合同中明确，不用一条全局规则禁止所有测试修改。

```python
def validate_changed_paths(changed, allowed):
    if not set(changed) <= set(allowed):
        raise PermissionError("changes outside task scope")
    return True
assert validate_changed_paths({"price.py"},{"price.py"})
try:
    validate_changed_paths({"price.py","test_price.py"},{"price.py"})
except PermissionError:
    pass
else:
    raise AssertionError("test tampering accepted")
```

练习十一，生成可审查 diff。答案使用标准库 unified_diff，对比旧新文本，检查只改变公式行。diff 是审查材料，不替代实际测试，也不自动授权发布。

```python
import difflib
fixed_source = buggy_source.replace("1 - rate","1 + rate")
diff = "".join(difflib.unified_diff(buggy_source.splitlines(True),fixed_source.splitlines(True),fromfile="price.py",tofile="price.py"))
assert "-    return price * (1 - rate)" in diff
assert "+    return price * (1 + rate)" in diff
```

练习十二，记录完整闭环证据。答案把基线、最终验证、改动范围和模型计数组织成报告，success 由实际条件计算，不复制模型自评。报告不含私有路径或密钥。

```python
with tempfile.TemporaryDirectory() as folder:
    raw = coding_project(folder)
    delivery = {"problem":"tax subtracted instead of added","changed_files":["price.py"],
        "baseline_failed":raw["before_exit"] != 0,"final_tests_passed":raw["after_exit"] == 0,
        "model_calls":raw["model_calls"],"tool_calls":raw["tool_calls"]}
    delivery["success"] = delivery["baseline_failed"] and delivery["final_tests_passed"]
    assert delivery["success"] and delivery["changed_files"] == ["price.py"]
```

## 第十二课：从零组装一个带保护策略的编码闭环

下面独立组装项目，不调用 coding_project。读取工具允许查看源码，编辑工具只允许 price.py，测试工具只运行本题可信测试。脚本模型依次读取、编辑、测试、解释；最后宿主再次独立验证并检查测试文件未被改变。这个版本把编辑权限明确放在工具函数里，而不只是写在提示词中。

```python
from labs.agents.runtime import Budget

def build_guarded_coding_demo(root):
    root = Path(root)
    (root / "price.py").write_text(buggy_source,encoding="utf-8")
    (root / "test_price.py").write_text(TEST_SOURCE,encoding="utf-8")
    workspace = Workspace(root)
    guarded = SourceOnlyWorkspace(workspace,{"price.py"})
    before = run_trusted_fixture_tests(root)
    test_hash = hashlib.sha256((root / "test_price.py").read_bytes()).hexdigest()
    read_tool = Tool("read_source","Read the permitted source file.",
        object_schema({"path":{"type":"string","enum":["price.py"]}}),
        lambda args:guarded.read(args["path"]))
    edit_schema = object_schema({"path":{"type":"string","enum":["price.py"]},
        "old":{"type":"string","minLength":1},"new":{"type":"string"},
        "expected_sha256":{"type":"string","minLength":64,"maxLength":64}})
    edit_tool = Tool("edit_source","Apply one exact source edit.",edit_schema,
        lambda args:guarded.replace(args["path"],args["old"],args["new"],args["expected_sha256"]),True)
    test_tool = Tool("run_tests","Run this trusted fixture only.",object_schema({}),
        lambda args:run_trusted_fixture_tests(root))
    sha = workspace.read("price.py")["sha256"]
    turns = [Turn(calls=[ToolCall("read","read_source",'{"path":"price.py"}')]),
        Turn(calls=[ToolCall("edit","edit_source",json.dumps({"path":"price.py","old":"1 - rate","new":"1 + rate","expected_sha256":sha}))]),
        Turn(calls=[ToolCall("test","run_tests","{}")]),
        Turn("The tax formula now adds the rate; the trusted tests passed.")]
    with Store(root / "run.sqlite") as store:
        harness = Harness(ScriptedModel(turns),[read_tool,edit_tool,test_tool],store=store,
            budget=Budget(model_calls=4,tool_calls=3,tool_output_chars=10000))
        state = harness.run(harness.new("guarded-coding","Fix the tax formula; preserve tests."))
        after = run_trusted_fixture_tests(root)
        tests_unchanged = hashlib.sha256((root / "test_price.py").read_bytes()).hexdigest() == test_hash
        success = state.status == "finished" and before["exit_code"] != 0 and after["exit_code"] == 0 and tests_unchanged
        return {"success":success,"before":before,"after":after,"tests_unchanged":tests_unchanged,
                "model_calls":state.model_calls,"tool_calls":state.tool_calls,"events":state.events}

with tempfile.TemporaryDirectory() as folder:
    guarded_result = build_guarded_coding_demo(folder)
    assert guarded_result["success"] and guarded_result["tests_unchanged"]
    assert guarded_result["model_calls"] == 4 and guarded_result["tool_calls"] == 3
```

逐段解释这份实现。最开始创建源码与测试，保存测试摘要；这份摘要由宿主持有，不交给模型决定。read_source 的 schema 只允许 price.py，编辑 schema 也限制相同路径，因此即使模型请求 test_price.py，在进入函数前就会被合同拒绝。函数内部 SourceOnlyWorkspace 再检查写入集合，形成接口与执行边界一致的政策。

edit_source 标记副作用，若替换后出现异常，Harness 不会把它当纯工具随意重试。run_tests 在本题可信临时环境中可重复，返回退出码与日志；生产测试必须重新分析外部效果。Budget 刚好允许固定脚本四次模型与三次工具，最后一轮完成时不能被转移上限覆盖，这依赖 A06 已验证的终态规则。

Store 使用临时文件，事件与状态可以在函数运行期间检查。最终 success 同时要求基线失败、最终通过、测试未改和运行正常结束。若模型最终文本说通过但测试失败，success 仍 False；若模型偷偷改测试，摘要不匹配也失败。独立验收把自然语言报告与真实事实分开。

这份脚本仍预先知道正确摘要和补丁，不能测真实模型的发现能力。它完整验证的是工具合同、权限、状态和验收闭环。替换为 ChatModel 时应让模型从 read_source 返回值提取摘要，并保留失败样本；不能把脚本结果当作线上成功率。A15 会说明真实适配器能力和协议差异。

## 第十三课：源码阅读要追踪数据，而不只是找函数名

阅读一个陌生编码 Agent 仓库，可以从入口开始画出数据流：任务文本进入配置，配置建立模型与环境，Agent 构造消息，模型返回动作，解析器提取命令，环境执行，观察回到消息，停止条件决定结束。每个箭头都问输入类型、输出结构、异常和状态保存位置。这样你会理解框架为什么这样组织，而不是只记 run、step 的名字。

在本课程 capstones.coding_project 中，root 是明确工作区路径，buggy 是可信初始源码，before 是测试结果字典，Workspace 生成文件工具，turns 是模型响应脚本，test_tool 连接测试执行器，Store 持久化任务状态，Harness 运行循环，verification 独立验收。每个变量都有不同权威：turns 是候选动作，tool 结果是观察，verification 才决定项目成功。

读到 lambda 时不要跳过。Workspace.tools 中 lambda 把模型参数字典字段传给具体 read 或 replace 方法；它是协议参数与业务函数的桥。若字段名拼错，会在调用后异常；若把模型提供 root 直接传入构造器，可能扩大权限。短小适配代码往往是关键边界，值得逐行检查。

读到异常处理时，标出副作用前后。参数解析失败可以证明未编辑，replace 完成后序列化失败则结果未知；测试进程超时不证明代码没有产生效果；数据库保存失败不回滚文件修改。把这些阶段写在注释或阅读笔记中，你就能发现框架宣传与实际实现的差距。

读到缓存时，问缓存键覆盖什么、何时失效、是否包含权限和代码版本。读到重试时，问谁拥有重试、最多真实调用多少、是否保持业务身份。读到 tracing 时，问记录的是请求意图还是已确认结果。源码级掌握来自这些具体问题，而不是把每个类名抄进大纲。

## 第十四课：调试时建立假设树

面对失败日志，可以先列少量互斥或可区分假设：代码逻辑错、修改未应用、运行了旧文件、测试环境错、测试本身不匹配需求。每个假设对应一个最小观察，例如读取当前源码摘要、打印模块路径、直接调用函数、检查依赖版本、查看断言输入。优先选择能排除最多假设且风险低的操作。

不要一次改很多地方再运行。若同时改公式、导入路径、测试和依赖，即使通过也不知道哪项起作用，可能掩盖新问题。先确认一个假设，做最小修改，运行相关验证，再扩展必要回归。这个方法与科学实验相似，适合基础薄弱的学习者逐步建立因果理解。

日志阅读要从最早相关错误开始，而不是只看最后一行。构建失败可能导致后续测试全部找不到模块，真正原因在前面的编译错误；一个测试初始化异常可能让多个用例同时失败，不能逐个改业务函数。找到共同前提，比把每个失败当独立 bug 更有效。

如果错误来自环境，修复环境后重新确认原始 bug 是否仍存在。环境恢复可能改变执行版本，使任务描述不再成立；这时应报告实际观察，而不是强行修改代码制造一个“修复”。编码 Agent 的责任是完成用户目标，不是一定要产生代码 diff。

调试记录应包含假设、证据、决定和结果。长任务中摘要保留已排除假设，避免上下文压缩后重复尝试。失败尝试也有学习价值，但最终交付只保留帮助审查的关键理由与验证范围，不需要把所有探索过程堆给用户。

## 第十五课：从单函数到真实开源任务的四级练习

第一级是单函数、标准库、明确输入输出和独立测试。你应能手算结果，解释每行代码，故意写出错误实现并看测试失败。税率项目属于这一层。通过后再增加多个函数和边界，不必急着接大型框架。

第二级是多文件接口任务。一个模块定义数据结构，另一个调用，测试覆盖兼容行为。练习修改字段、增加可选参数或修复异常传播，要求查所有调用者并写契约测试。此时重点是理解依赖与变更范围，不只是算法本身。可以用本仓库 Agent 的 Tool、Turn 和 Harness 关系做受控阅读。

第三级是小型开源项目的真实问题。选择你已经能手工理解的 bug，固定提交，按项目说明建立环境，先写失败测试再让 Agent 提案。你要能解释 diff 每行，以及为什么测试能捕获原问题。若完全看不懂项目，先做源码阅读任务，不把 Agent 输出当替代学习。

第四级是跨模块任务，涉及配置、持久化、并发或性能。此时建立任务分解与接口合同，必要时用独立审查角色，最终跑集成验收与故障注入。难度一次只增加少数维度，避免代码规模、依赖、硬件和协议同时陌生。你的 Linux 与 Python 基础课程应并行补齐，遇到命令与语法能返回基础卷查证。

开源阅读可以选择 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 作为较紧凑的编码循环研究对象。具体版本以你实际检出的提交为准，记录 commit 和环境，不依赖网上旧教程行号。阅读任务不是只找到某个文件，而是把模型输入、动作解析、环境执行、观察和停止条件映射到本章已经掌握的数据流，再写一个最小错误 fixture 验证理解。

### 深入一：真实仓库任务的环境与权限清单

开始前确认项目使用哪种包管理与测试入口，是否需要网络下载，是否有外部数据库，测试是否会修改共享服务。只安装任务必需依赖，不把公司凭证自动注入未知测试。独立工作目录、受限网络和普通用户权限可以降低风险，但具体沙箱能力仍需按平台验证。

你的个人 PC 适合随时做阅读、标准库实验、小型单元测试和 API Agent 项目。Linux 服务器 CPU 资源可用于容器与服务练习，H100 只在模型推理、训练或算子任务确有需要时使用。编码 Agent 修复基础项目通常不需要 GPU。资源安排应由任务计算需求决定，而不是因为有高端硬件就默认占用。

如果真实仓库需要 GPU 测试，先运行 CPU 或静态相关检查，再按项目规则选择最小设备与有界用例。未运行的 GPU 检查明确标记，不用 CPU 通过冒充完整验证。不要改变共享服务器全局驱动或终止他人任务来让自己的测试通过，环境修复应限于授权范围。

依赖安装与测试执行也可能有网络副作用。项目脚本可以下载模型或数据，甚至运行安装钩子；先阅读命令含义和项目来源，使用隔离环境。Agent 的便利不应让你跳过理解执行内容，这也是从零到能独立维护系统的必要训练。

### 深入二：如何判断补丁足够小又足够完整

最小补丁应覆盖根因及必要回归，不是只追求最少字符。若错误来自输入合同，可能需要入口验证、错误映射和测试三处修改；只在下游加 try/except 隐藏异常虽然行数少，却没有修复根因。相反，为一个符号错误重构整个模块会增加风险，难以审查。

检查补丁时问：是否改变公开接口，是否影响默认行为，是否新增依赖，是否改变持久数据格式，是否影响并发和恢复，是否修改用户未要求区域。每个是都需要具体理由与验证。风险并不只与行数成正比，一个配置默认值变化可能影响所有用户，几十行局部测试可能风险很低。

回归测试应先证明原缺陷，再覆盖相邻边界。税率项目的零税率防止修复破坏原正确行为；不同价格防止硬编码；多轮编辑摘要测试保护执行过程。不要写只检查实现中某个字符串存在的测试来证明业务正确，字符串可以存在但没有被执行。源码检查可以作为范围约束，行为仍需运行验证。

最终查看完整 diff，确认没有临时打印、私有路径、密钥、无关缓存或测试产物混入。生成文件是否需要提交按项目规范决定。不要把“测试通过”当作无需审查 diff 的理由，测试覆盖有限，审查可以发现范围和意图问题。

### 深入三：编码 Agent 与知识助手连接时的边界

真实编码任务常需要检索 API 文档和仓库说明。检索工具返回带来源的材料，模型据此修改代码，但外部文档可能过时或含注入指令。A07 的来源标签与执行政策仍需保留：文档中的命令不自动获得执行授权，版本说明应与实际依赖对应。

可以把检索结果组织为 claim、source、version 和 relevant_code_location。修改前核对所选 API 在当前依赖中确实存在，必要时查看本地源码或运行最小示例。不要因为网页示例调用成功的截图就假设当前版本兼容。A15 的协议与框架集成会继续训练这种版本敏感阅读。

长期记忆可以保存项目约定和已验证排障经验，但要带版本与适用范围。上次某项目通过删除缓存解决问题，不代表所有项目都应自动清理；某框架旧版参数不能永久当当前事实。记忆应帮助提出假设，关键动作仍依据当前代码与环境验证。

跨卷项目可以把有权限的检索工具接到 Harness，给编码节点提供当前文档证据，再用受控编辑与测试完成修改。验收不仅看代码通过，还检查引用来源真实、权限过滤有效、旧资料冲突明确。这样 Agent 执行机制与检索记忆系统真正连接，而不只是增加一个叫 RAG 的工具名。

### 深入四：把四轮修复轨迹拆成消息与效果

第一次模型响应包含 read 调用，宿主把 assistant 工具请求加入 messages，再执行 read_source，返回文本与摘要的 tool 消息。此时文件未改变，ledger 有读取结果。第二次模型响应包含 edit 调用，参数携带旧摘要；编辑成功后文件改变，ledger 保存新摘要，随后 tool 消息告诉模型编辑结果。第三次测试返回退出码，第四次才生成最终说明。

| 阶段 | 模型可见新增观察 | 文件效果 | 宿主验收状态 |
|---|---|---|---|
| 初始复现 | 任务与允许工具 | 错误源码已建立 | 基线失败已确认 |
| 读取完成 | 原文与内容摘要 | 无修改 | 尚未修复 |
| 编辑完成 | 新摘要与字节数 | 公式已改变 | 尚未验证行为 |
| 测试完成 | 退出码与受限日志 | 测试可能产生受控临时效果 | 公开反馈通过或失败 |
| 最终说明 | 模型总结 | 无新增代码效果 | 仍需独立 verifier |
| 独立验收 | 宿主报告 | 检查实际代码与测试完整性 | 决定项目 success |

这张表说明为什么不能在编辑工具返回 ok 后立即宣布修复完成。ok 只证明替换按合同完成，不证明新代码正确；测试通过也只证明已执行用例，不证明所有环境；最终说明更不是新证据。每个阶段的保证应逐步累积，不能提前升级。

如果编辑完成后进程崩溃，恢复可能看到 ledger done 但对话尚未有结果，A10 允许回填缓存观察，不应重复替换；若文件已改但 ledger 未确认，则需要读取当前文件并核对业务意图，不能盲目用旧摘要再写。编码工具有副作用，恢复机制与工单一样需要认真处理，只是文件通常更容易人工检查。

测试结果也要与源码版本绑定。若测试通过后又有一次编辑，旧通过记录不能证明新代码。可以记录测试时源码摘要或提交 id，最终验收确认当前版本与已测版本一致。这个原则在多人协作或多 Agent 并发编辑时尤其重要，避免“测试是绿的”指向另一份代码。

### 深入五：真实模型接入时怎样保持实验可控

真实模型替换 ScriptedModel 后，先保留同一临时夹具、同一工具白名单与同一独立 verifier。模型可能不按预设顺序行动，可能先提出错误补丁、忘记测试或生成无效参数；这些都应成为轨迹，而不是通过放宽验收隐藏。设置小而明确的模型与工具预算，失败后保存原因，不自动无限重试。

实际端点、模型名称和凭证由用户配置，不写进仓库。课程 live adapter 当前只实现非流式 Chat 网络请求，Responses 与 Messages 的离线解析不能直接当作完整在线适配器使用。若选择其他协议，先补齐请求、续接、错误和原始内容保留契约，再测编码能力，避免把协议 bug 当模型不会写代码。

真实评测至少重复多个独立临时项目，记录成功、调用数、用量、延迟和失败类型。一次成功说明该次轨迹完成，不代表稳定能力；一次失败也不代表整个方法无效。按 A13 的任务与配置版本保存结果，比较提示或工具描述时固定其他变量。不要把付费调用放进必须每次通过的离线 CI。

若模型请求越权工具或修改测试，执行器拒绝并记录。最终回答道歉不抵消实际越权效果，实际没有效果也不表示模型策略完美；尝试与效果分开统计。这样你可以同时改进模型提示和宿主边界，而不把安全全部寄托在某个模型版本上。

### 深入六：交付说明与学习验收

完成任务后，说明具体问题、修改后的行为、运行过的验证和剩余限制。对税率项目，可以说减号改为加号，正常税率与零税率测试通过，测试文件未改，未评估真实货币舍入与负输入合同。无需把所有内部事件都贴给用户，也不要声称“全面优化”。清楚范围让下一次扩展有依据。

学习验收比交付验收更严格。你应能不看参考答案重新搭建工具闭环，解释 read hash 为什么必要、为什么编辑是副作用、为什么测试工具不能随意当纯、为什么最终说明不决定 success。然后故意引入错误补丁、过期摘要和测试篡改，预测系统在哪里拒绝，并用实际运行确认。

下一步真实开源练习应要求你先手工写出问题假设，再允许 Agent 提案；审查每一行 diff，追踪调用者，运行最小失败测试和必要回归。若某段代码看不懂，回到基础卷学习语法、数据结构、异常、进程与 Git，不把工具输出当理解的替代。宏大的全栈目标通过这种可验证小闭环逐步建立。

最终你应能在模型建议错误时用代码与证据纠正它，在环境不确定时停止猜测，在权限不足时保持边界，在任务完成后给出可复核结果。编码 Agent 的价值不是让你永远不用理解代码，而是把阅读、试验和验证组织得更高效；你掌握这些机制后，才有能力修改框架核心、调试复杂 Agent 和继续深入算子与模型实现。

### 深入七：把一次修复变成可维护工程资产

修复完成后，保留能代表根因的回归测试，并让测试名称说明行为，例如 test_tax_adds_rate，而不是 test_case_1。测试输入应简单到可手算，失败信息清楚指出期望与实际。若需要多个条件，拆成独立用例，避免第一个断言失败后后面边界从未执行。课程小例子用两个测试建立这一习惯，真实项目可以按接口合同继续扩展。

文档更新应与行为变化匹配。若只修内部符号错误，可能不需要大段文档；若改变参数语义、默认值或错误类型，则需要更新使用说明与迁移提示。不要把实现细节塞进产品用户流程，除非它影响用户选择；工程文档则应保留必要的合同与验证证据。不同读者需要不同层次，清楚表达比重复堆信息更重要。

版本控制提交应只包含任务相关修改。提交前检查未跟踪产物、数据库、日志和缓存，确认没有私人数据。独立临时实验数据库不应误入仓库；若需要 fixture，应使用人工最小数据和明确来源说明。公开仓库中的可复现实验应任何读者都能运行，不依赖你的公司路径或账户。

最后，为后续扩展留下稳定接口和验收，而不是留下一堆未实现核心。这个项目已经完整跑通读取、编辑、测试、恢复边界解释与独立评分；以后增加真实模型、更多文件或检索工具时，可以在现有合同上扩展。每次扩展先说明新增能力与风险，再增加对应测试，保持已掌握基础不回退。

一个有用的最终自测是把所有模型最终回答改成“任务失败”，但保留正确工具轨迹；独立 verifier 仍应依据实际代码判断成功。再反过来保留“任务成功”文本而删除编辑，verifier 应失败。这个对照证明评分没有被叙述操纵，也提醒你用户报告应从宿主验证结果生成，避免模型措辞与真实状态矛盾。

因此最终交付字段应保存验证器版本与被测源码摘要，使后来复核能够确认结论对应哪份实际代码。
