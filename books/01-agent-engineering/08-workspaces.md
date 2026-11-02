# A08 工作区与执行环境：让 Agent 能读、能改，也知道边界在哪里

本章从文件和目录的基础开始，逐步实现一个受限文件工具：只接受工作区内相对路径，只读取小型 UTF-8 文本，修改前核对读到的版本，要求旧文本恰好出现一次，再用临时文件替换。随后学习启动受控测试进程、收集退出码、限制输出和处理超时。所有实验使用临时目录，不访问个人文件，不执行不可信代码。

课程 `Workspace` 是教学路径守卫，不是操作系统沙箱。它能拒绝一些明确越界请求，但无法抵抗恶意并发重命名，也没有限制整个 Python 进程的文件、网络或凭证权限。理解这个边界是本章目标的一部分：你应知道哪些保证来自字符串检查，哪些来自文件系统操作，哪些必须由独立执行环境提供。

## 第一课：文件路径不是文件本身

路径是寻找文件的一串名称，不是内容，也不一定是稳定身份。相对路径从当前基准目录解释，绝对路径从文件系统根或平台规定的根开始。`notes/a.txt` 在两个不同工作区可以指向不同文件；同一路径今天和明天也可能有不同内容。Agent 修改代码时，需要同时知道允许的根目录、目标相对路径和读取时的内容版本。

Python 的 Path 对象帮助组合和检查路径。`root / relative` 表示路径拼接，不是打开文件；`resolve()` 尝试规范化并解析符号链接；`open()` 才实际访问内容。把这三个动作分开看，才能理解为什么“检查过路径安全”与“随后打开的就是同一个文件”之间仍可能存在时间窗口。

```python
# course-example: A08-lf-read
from pathlib import Path
import tempfile
from labs.agents.tools import Workspace

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "notes").mkdir()
    (root / "notes" / "hello.txt").write_text("你好\n", encoding="utf-8", newline="\n")
    workspace = Workspace(root)
    result = workspace.read("notes/hello.txt")
    assert result["text"] == "你好\n"
    assert len(result["sha256"]) == 64
```

TemporaryDirectory 在进入 with 时创建目录，退出时清理本次实验文件。这个清理范围由标准库对象管理，不使用广泛删除命令。`mkdir` 创建子目录，`write_text` 明确编码，`Workspace(root)` 绑定根目录。返回值同时包含文本和 SHA-256 摘要，后者用来检测后续修改前内容是否变化，不是用来证明内容可信。

示例使用 `newline="\n"` 创建 UTF-8/LF 文件，避免 Windows 默认写成 CRLF。Workspace.read 先读原始字节再解码，保留文件实际换行，因此本例读到的文本精确等于 `"你好\n"`。若读取已有 CRLF 文件，结果应保留 `"\r\n"`，摘要也应由那些实际字节计算；不能先统一换行再声称拿到了原文件版本。

文件名与文件内容都可能包含不可信输入。一个看似普通的仓库可以有奇怪名称、符号链接、巨大文件或测试脚本副作用。不能因为来源是 Git 仓库，就认为所有操作都安全。读取文本和执行代码是不同权限级别；本章先练读写受控文件，再明确执行边界。

## 第二课：从词法路径到解析路径，为什么检查两次

Workspace.resolve 首先把输入转换为 Path，拒绝绝对路径和任何 `..` 组件。`..` 表示父目录，允许它会使用户请求容易跳出根目录。这里采用直接拒绝而不是尝试聪明地化简，因此 `sub/../file.txt` 即使最终可能仍在根内，也被拒绝。教学接口优先保持规则简单可解释，调用方可以使用规范的相对路径。

随后逐个检查路径组件是否为符号链接，再调用 resolve，最后确认解析后的路径仍位于 root 内。词法检查处理明显越界表达，符号链接检查处理名称指向别处，最终包含关系检查处理解析后的实际路径。多层检查各有目的，不能只用字符串前缀判断，例如 `/work/project-other` 也以 `/work/project` 开头，却不是它的子目录。

```python
from labs.agents.contracts import ContractError

def expect_error(kind, operation):
    try:
        operation()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    workspace = Workspace(root)
    expect_error(ContractError, lambda: workspace.resolve("../outside.txt"))
    expect_error(ContractError, lambda: workspace.resolve(str(root / "absolute.txt")))
    assert workspace.resolve("sub/file.txt") == root / "sub" / "file.txt"
```

最后一个断言没有要求文件已经存在。resolve 的默认行为可以为尚不存在的尾部生成规范路径，而 read 随后的 open 会决定是否存在。根目录构造时使用 strict=True，保证绑定的根确实存在。区分“路径合法但文件不存在”和“路径越界”有助于给模型准确反馈：前者可能需要列目录确认，后者应该停止越权尝试。

平台差异也要明确。Linux 和 Windows 的路径语法不同，反斜杠、盘符和大小写行为不能靠一个字符串替换解决。课程在当前 Python 所在平台使用 Path 语义，推荐你在 WSL 或 Linux 环境运行这些实验。若产品接受跨平台路径协议，应先定义一种统一表示，再按目标平台严格转换，不能把客户端提供的任意路径直接交给服务器解释。

## 第三课：符号链接与检查后使用的竞争

符号链接像一个名称转向另一个路径。工作区里看见 `docs/link.txt`，它可能实际指向工作区外的文件。课程逐组件拒绝符号链接，因此不仅最终文件是链接会被拒绝，中间目录是链接也会被拒绝。这种保守规则牺牲了部分合法仓库用法，但让初学者能清楚理解根目录边界。

然而检查和打开之间存在时间。程序先检查某目录不是链接，另一个恶意进程随后把它替换成链接，程序再打开时可能访问不同目标。这叫检查后使用竞争。重复检查几次不能根本解决，因为最后一次检查与实际使用之间仍有窗口。需要更强的操作系统级路径解析约束、目录文件描述符策略或独立沙箱，具体实现要按平台能力设计。

课程没有声称抵抗恶意同机并发修改。它的使用前提是受控教学目录、单写入者、没有敌对进程同时改路径。这个前提不是可省略的小字，因为它决定了路径守卫能保证什么。如果你要让 Agent 处理来自未知来源的仓库或执行测试，应该把工作副本放到隔离环境，并限制网络和凭证，而不只是复用这个类。

符号链接之外还有硬链接、挂载点、文件权限和特殊文件等问题。拒绝符号链接不自动处理所有文件系统对象。一个路径可能是设备、管道或目录，read 的行为可能阻塞或失败；生产文件工具应限制为普通文件，并考虑文件数量、总字节和读取时间。教学代码只演示小型文本文件，不应扩展描述成任意文件安全访问器。

安全分析应围绕威胁模型：谁能控制相对路径，谁能改变目录内容，谁能启动并发进程，工具进程拥有哪些系统权限。若只是你自己在临时目录练习，教学守卫足够观察机制；若外部用户能上传任意项目，边界要求就更高。没有威胁模型，争论“这个路径检查安全吗”很难有准确答案。

## 第四课：读取上限按字节计算，解码失败不能悄悄忽略

Workspace.read 以二进制模式打开文件，读取 max_bytes 加一字节。如果得到超过上限的数据，就知道文件太大而拒绝；无需先把整个巨大文件读进内存。多读一个字节是常见边界技巧：恰好等于上限允许，超过哪怕一个字节也能识别。先 stat 再 read 不足以代替这个检查，因为文件可能在两者之间增长。

读取完成后使用 UTF-8 解码，再计算原始字节的 SHA-256。解码严格失败能避免损坏文本被静默改写。若用 errors=ignore，某些字节会消失，随后替换工具可能写回一份不同内容，用户却不知道原文件曾有非 UTF-8 数据。对于二进制文件，应使用专门工具或返回不可文本处理的错误，不要强行当作代码文本。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "small.txt").write_bytes("中".encode("utf-8"))
    assert Workspace(root, max_bytes=3).read("small.txt")["text"] == "中"
    expect_error(ContractError, lambda: Workspace(root, max_bytes=2).read("small.txt"))
    (root / "binary.bin").write_bytes(b"\xff\xfe")
    expect_error(UnicodeDecodeError, lambda: Workspace(root).read("binary.bin"))
```

这个例子说明一个汉字在本例占三个字节，字符上限与字节上限不同。A07 的上下文预算按字符计算，Workspace 的文件容量按字节计算，字段名必须体现单位。把两者混用，会在中文、符号和编码数据上得到意外边界。后续模型输入还会有 token 单位，因此一个系统通常同时维护三种长度。

返回的哈希反映本次实际读取的 bytes。它可以帮助检测后续内容变化，但不是文件锁，也不是访问权限。攻击者若能控制文件内容，当然能生成对应哈希；哈希只是内容标识，不证明来源可信。修改工具使用 expected_sha256 的目的，是避免基于过期观察覆盖用户的新改动。

## 第五课：精确替换为什么需要旧内容与读时摘要

模型常见编辑方式是“把某段旧文本替换成新文本”。若只按文件路径写入新全文，可能覆盖其他人刚加入的修改；若旧文本出现多次，简单 replace 可能改错位置。课程要求调用者提供读取时摘要，并要求 old 非空且恰好出现一次。两条约束分别处理过期版本与目标歧义。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    path = root / "calc.py"
    path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    workspace = Workspace(root)
    before = workspace.read("calc.py")
    result = workspace.replace("calc.py", "return a - b", "return a + b", before["sha256"])
    assert "return a + b" in path.read_text(encoding="utf-8")
    assert result["sha256"] != before["sha256"]
    expect_error(ContractError, lambda: workspace.replace("calc.py", "return a + b", "return 0", before["sha256"]))
```

第一次替换成功后内容摘要变化，第二次仍拿旧摘要请求修改，被明确拒绝。这时正确行为是重新读取、理解新内容、重新提出补丁，而不是自动换成最新摘要继续执行旧替换。摘要冲突意味着你的观察过期，旧计划可能不再适用。盲目刷新摘要只是绕过保护，没有解决语义冲突。

old 必须恰好出现一次。若函数里两个地方都有 `return 0`，应该扩大上下文，例如包含函数名和相邻行，直到唯一定位；若仍无法唯一定位，可以使用更结构化的补丁或语法树工具。不要把 count 不等于一时的行为改成“默认替换第一个”，因为那会把歧义藏起来。可审查的失败比悄悄修改错误位置更好。

new 可以是空字符串，表示删除匹配片段；old 不能空，因为空字符串在文本中处处匹配。工具 schema 对 old 使用非空字符串，对 new 允许空，这体现接口语义区别。删除也是副作用，所以 replace_text 注册为 side_effect=True；如果替换完成后返回路径发生异常，运行时应保留结果不确定，而不是把编辑当成可随意重复的纯操作。

## 第六课：临时文件替换保证原子可见，不保证并发比较交换

直接打开目标文件为写模式会先截断它。如果进程在写入一半时崩溃，读者可能看到半个文件。课程先在目标同目录创建临时文件，写入完整新内容，再用 os.replace 替换目标路径。这样其他普通读取者通常观察到旧文件或新文件，而不是正在写入的一半内容。这里讨论的是单个路径替换的可见性，不是整个项目多文件事务。

源码用 mkstemp 返回文件描述符和临时路径。文件描述符交给 os.fdopen，并在 with 退出时关闭；写入后复制目标文件的权限模式，再执行 os.replace；finally 检查临时路径是否仍存在，若替换前失败则清理。临时文件放在同目录，有助于保持同一文件系统上的替换语义，避免跨文件系统移动导致不同操作路径。

```python
import hashlib
import os

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    target = root / "a.txt"
    target.write_text("old", encoding="utf-8")
    workspace = Workspace(root)
    before = workspace.read("a.txt")
    workspace.replace("a.txt", "old", "new", before["sha256"])
    assert target.read_text(encoding="utf-8") == "new"
    assert not list(root.glob(".course-edit-*"))
    assert workspace.read("a.txt")["sha256"] == hashlib.sha256(b"new").hexdigest()
```

但这不是原子比较交换。函数先读取并核对哈希，再创建临时文件，最后替换；另一写入者可以在核对后、替换前修改目标，新修改仍可能被覆盖。哈希检查保护的是已观察到的过期版本，不提供跨时间窗口的锁。若需要多写入者安全，可以使用同一拥有者串行化、明确文件锁协议、版本化存储或更强的文件系统机制，并要求所有写入者遵守。

原子可见也不等于崩溃后持久。写入和关闭可能仍经过操作系统缓存，断电后是否保留需要更严格的刷新与文件系统保证。课程没有 fsync 文件与目录的完整耐久协议，因此不能声称抵抗断电的事务写入。A10 的数据库事务会进一步区分原子性、一致性与持久化窗口；不要把一个 os.replace 当成全部事务属性。

多文件修改更复杂。先替换文件甲成功，替换文件乙失败，项目处于部分修改状态。可以先在隔离工作副本中完成全部编辑和测试，再以版本控制提交作为审查单元；但提交本身也不是自动部署。对于需要运行中一致更新的配置，应另设计版本目录和指针切换等机制。Agent 编码场景通常优先使用独立工作区，让用户能查看完整 diff 并保留回退路径。

## 第七课：子进程是程序执行，不是字符串输出工具

运行测试意味着执行代码。Python 测试文件可以读取文件、访问网络、启动进程或修改数据库；名字以 test 开头并不降低权限。因此课程 `run_trusted_fixture_tests` 明确只用于仓库受控教学 fixture。`-I` 隔离部分 Python 导入环境，`-B` 不写字节码，但它们不限制操作系统文件和网络权限，不能当作沙箱。

函数使用参数列表调用 subprocess.run，而不是拼接一段 shell 字符串。列表中的每项是一个参数，空格或特殊符号不会自动被 shell 解释。这样可以避免很多命令注入问题，也使实际执行的程序和参数更清楚。但安全并未因此完成：如果被执行的程序本身接受危险参数，或测试代码有副作用，参数列表仍然会忠实执行它们。

退出码零通常表示进程按照约定成功，非零表示失败或其他状态。测试工具应同时返回退出码和受限输出，不能仅在输出里搜索 OK，因为恶意或错误程序可以打印这个词。超时返回课程约定的 124，不代表所有系统或所有程序都使用同样退出码。调用方应知道这是包装器生成的状态，而不是子进程自然返回。

stdout 与 stderr 被合并到临时文件，避免在内存里无限累积日志；完成后只读取前八千一百九十二字节，再多读一个判断截断。这个限制保护返回给模型的输出体积，但临时文件在进程运行期间仍可能增长，因此不是磁盘总量限制。生产执行环境还需要文件系统配额或日志轮转，不能把“只返回前几千字节”误认为“最多产生几千字节”。

超时后主子进程如何结束也取决于执行器。subprocess.run 的 timeout 会处理其直接管理的进程，但测试代码自行创建的后代进程可能需要进程组或容器级回收。课程不提供通用恶意进程树清理。不要为了“彻底清理”执行宽泛 killall，尤其在共享公司服务器上；清理范围必须绑定本任务创建的受控资源。

## 第八课：工作区、进程与容器各自提供什么边界

工作区限制工具愿意访问哪些路径；进程隔离让错误崩溃不直接破坏宿主解释器；容器或其他沙箱可以限制文件系统、网络、用户身份和资源。三者不是同一个安全等级。一个 Python 类即使只暴露 read_file，类所在进程若还拥有任意代码执行入口，模型可能通过另一工具访问整个环境。能力边界需要从所有入口共同分析。

对初学实验，最简单的安排是在临时目录中操作自己写的短文件，用系统 CPU 执行受控测试，结束后清理本任务目录。对未知仓库，应使用独立工作副本与隔离执行环境，默认不放入真实凭证，不挂载敏感主机目录，不开放不必要网络，不使用特权模式。具体隔离配置要按所用平台与任务验证，本章不会用一个通用命令声称覆盖所有情况。

最小权限意味着工具只获得完成任务需要的能力。例如只读审查节点可以读取指定目录但不能写入，编辑节点可以修改允许文件但不能任意执行 shell，测试节点可以运行预定义命令但不能选择任意可执行程序。你可以把这些能力做成不同工具，而不是一个接受任意命令字符串的万能工具。能力越具体，参数验证和审计越容易。

网络也属于工作区之外的重要边界。测试可能尝试下载依赖或访问公司服务；即使文件读写受限，网络外发仍可能泄漏内容。真实执行器应明确允许哪些域名、端口和凭证，并记录外部请求。没有授权的网络失败应解释为环境限制，不应让 Agent 自动寻找绕过路径。修复代码与改变执行环境权限是不同动作。

对于你的 Win11 PC 与 Linux 服务器，本章实验只需要 CPU 和普通临时目录。无需占用 H100，也无需管理员权限。可以在 WSL 或 Linux 上练习路径和进程行为，再在 Windows 上观察差异；差异本身是学习内容，不应通过删掉边界测试掩盖。后续 GPU 算子实验再引入明确设备和时间预算，Agent 文件工具不应顺便获得整个服务器管理权限。

## 第九课：四个真实形状的故障与诊断

故障一，路径前缀检查放过了邻居目录。代码判断字符串以 root 开头，`project-backup` 也通过。修复使用规范路径的祖先关系，而不是文本前缀；测试构造同前缀不同目录。课程 Workspace 最终使用 is_relative_to，但仍受前面讨论的并发路径竞争限制。修复一个词法漏洞，不等于获得完整沙箱。

故障二，模型基于旧文件修改，覆盖用户刚加的一行。修复要求读取摘要与修改前摘要一致，冲突时重新读取并重新推理。测试先 read，再由测试代码写入新内容，最后用旧摘要 replace，目标应保持新内容不变。若实现自动刷新摘要再执行旧补丁，测试也应判失败，因为它绕过了保护目的。

故障三，旧片段出现两次，替换改错函数。修复要求唯一匹配，并让调用方提供更具体上下文。测试两个函数都返回零，old 只写 return 0 时应拒绝；使用包含函数定义的片段时可成功。这里不是让模型学会更长字符串而已，而是给执行器一个可确定验证的目标定位条件。

故障四，测试输出巨大导致内存耗尽或模型上下文溢出。修复将输出导向受控存储，再返回有上限的片段、截断标记和 artifact 引用。课程临时文件方案只限制返回内存，不限制总磁盘，因此生产还要配置磁盘配额。测试应分别验证输出截断与进程资源限制，不能用一个 truncated=true 宣称所有容量问题已解决。

## 第十课：十二道完整练习与答案

练习一，计算文本字节与字符差异。答案使用“中a”，字符数二、UTF-8 字节数四。给工作区三字节上限应失败，四字节应成功。这个练习帮助你在后续上下文和文件限制中始终写清单位。

```python
text = "中a"
assert len(text) == 2 and len(text.encode("utf-8")) == 4
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "text.txt").write_text(text, encoding="utf-8")
    expect_error(ContractError, lambda: Workspace(root, max_bytes=3).read("text.txt"))
    assert Workspace(root, max_bytes=4).read("text.txt")["text"] == text
```

练习二，父目录路径即使最终回到根内也拒绝。答案使用 `sub/../a.txt`，验证课程采用保守词法合同。不要把这个测试改成允许后又忘记更新文档；接口接受范围的改变应是明确设计决定。

```python
with tempfile.TemporaryDirectory() as folder:
    workspace = Workspace(folder)
    expect_error(ContractError, lambda: workspace.resolve("sub/../a.txt"))
    expect_error(ContractError, lambda: workspace.resolve("../a.txt"))
```

练习三，不存在文件与越界路径应属于不同失败。答案合法相对路径触发 FileNotFoundError，越界触发 ContractError。分类有助于模型选择正确下一步：可以核对文件名，但不应重试越权路径。

```python
with tempfile.TemporaryDirectory() as folder:
    workspace = Workspace(folder)
    expect_error(FileNotFoundError, lambda: workspace.read("missing.txt"))
    expect_error(ContractError, lambda: workspace.read("../missing.txt"))
```

练习四，旧摘要拒绝覆盖新内容。答案在 read 后模拟另一个合法写入者，再尝试旧补丁。检查异常后文件保持用户的新内容，证明拒绝没有附带破坏。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    path = root / "a.txt"
    path.write_text("old", encoding="utf-8")
    workspace = Workspace(root)
    observed = workspace.read("a.txt")
    path.write_text("user changed", encoding="utf-8")
    expect_error(ContractError, lambda: workspace.replace("a.txt", "old", "agent changed", observed["sha256"]))
    assert path.read_text(encoding="utf-8") == "user changed"
```

练习五，重复匹配必须拒绝。答案先用两处相同文本触发失败，再用唯一完整行成功。这里的替换是字面字符串，不是正则表达式，点号和括号没有特殊含义，调用方不需要为正则转义。

```python
# course-example: A08-lf-replace
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "a.txt").write_text("first=0\nsecond=0\n", encoding="utf-8", newline="\n")
    workspace = Workspace(root)
    observed = workspace.read("a.txt")
    expect_error(ContractError, lambda: workspace.replace("a.txt", "=0", "=1", observed["sha256"]))
    workspace.replace("a.txt", "first=0", "first=1", observed["sha256"])
    assert workspace.read("a.txt")["text"] == "first=1\nsecond=0\n"
```

这里显式写入 LF，使末尾的逐字符断言在 Windows 和 Linux 上一致。替换操作只改动命中的文本，周围换行继续保留；若输入本来是 CRLF，替换不含换行的片段后仍应是 CRLF。文件版本校验始终使用原始字节摘要。

练习六，删除与空匹配的区别。答案 new 为空可以删除唯一片段，old 为空必须拒绝。删除后的摘要仍应对应实际剩余 bytes，而不是旧文件摘要。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "a.txt").write_text("keep remove", encoding="utf-8")
    workspace = Workspace(root)
    observed = workspace.read("a.txt")
    expect_error(ContractError, lambda: workspace.replace("a.txt", "", "x", observed["sha256"]))
    workspace.replace("a.txt", " remove", "", observed["sha256"])
    assert workspace.read("a.txt")["text"] == "keep"
```

练习七，替换后内容超过上限应保持原文件。答案原始两字节，上限三字节，替换为四字节失败。检查原内容和目录没有残留临时文件；容量检查在创建临时文件前发生。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "a.txt").write_text("ab", encoding="utf-8")
    workspace = Workspace(root, max_bytes=3)
    observed = workspace.read("a.txt")
    expect_error(ContractError, lambda: workspace.replace("a.txt", "ab", "abcd", observed["sha256"]))
    assert (root / "a.txt").read_text() == "ab"
    assert not list(root.glob(".course-edit-*"))
```

练习八，读取二进制文件不能忽略错误。答案写入非法 UTF-8 字节，确认解码失败且文件 bytes 不变。读取函数没有修复或写回权限，因此错误不应改变源文件。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    path = root / "raw.bin"
    path.write_bytes(b"\xff")
    expect_error(UnicodeDecodeError, lambda: Workspace(root).read("raw.bin"))
    assert path.read_bytes() == b"\xff"
```

练习九，工具注册应区分读与写。答案检查 Workspace.tools 中 read_file 为纯工具，replace_text 为副作用工具；这会影响 A06 的异常分类和 A10 的恢复策略。名称看起来只读不够，注册属性必须真实对应函数行为。

```python
with tempfile.TemporaryDirectory() as folder:
    registered = {tool.name:tool for tool in Workspace(folder).tools()}
    assert registered["read_file"].side_effect is False
    assert registered["replace_text"].side_effect is True
    assert registered["replace_text"].schema["additionalProperties"] is False
```

练习十，参数列表不会解释 shell 元字符。答案启动受控 Python 子进程，把含分号与美元符号的普通参数原样打印。测试不调用 shell，不执行参数内容。它证明的是参数传递，不证明任意被执行程序安全。

```python
import subprocess
import sys

argument = "literal; $HOME is text"
completed = subprocess.run([sys.executable, "-I", "-c", "import sys; print(sys.argv[1])", argument],
    capture_output=True, text=True, check=True)
assert completed.stdout.strip() == argument
```

练习十一，运行一个完全由本题创建的可信测试。答案在临时目录写入只做整数加法断言的 unittest，使用课程包装器，检查退出码零。它不下载依赖、不访问网络、不读取用户文件，因此适合离线执行。真实陌生仓库不能据此获得同样信任。

```python
from labs.agents.tools import run_trusted_fixture_tests

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "test_small.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(2+3,5)\n", encoding="utf-8")
    result = run_trusted_fixture_tests(root)
    assert result["exit_code"] == 0 and result["truncated"] is False
```

练习十二，测试失败必须由退出码体现。答案把预期五改为六，进程输出包含失败信息但仍正常退出到包装器，返回非零。调用方不能把 subprocess.run 没抛异常当成测试通过，因为包装器使用 check=False，故意由返回值表达测试结果。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "test_small.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(2+3,6)\n", encoding="utf-8")
    result = run_trusted_fixture_tests(root)
    assert result["exit_code"] != 0
    assert "FAILED" in result["output"]
```

## 第十一课：完整小项目，先生成修改计划，再提交一次受控编辑

下面的小项目把读取、规划、审查与应用分开。计划包含路径、读时摘要、旧片段、新片段和供人查看的统一 diff。应用时仍由 Workspace 再验证摘要与唯一匹配，不能因为计划已经生成就跳过检查。这个结构适合 Agent：模型提出补丁，宿主产生可审查结果，用户或政策批准后才提交；本章离线例子直接应用我们自己明确写出的修复。

```python
from dataclasses import dataclass
import difflib

@dataclass(frozen=True)
class EditPlan:
    path: str
    expected_sha256: str
    old: str
    new: str
    diff: str

def plan_edit(workspace, relative, old, new):
    before = workspace.read(relative)
    if not old or before["text"].count(old) != 1:
        raise ContractError("edit target must occur exactly once")
    after = before["text"].replace(old, new, 1)
    if len(after.encode("utf-8")) > workspace.max_bytes:
        raise ContractError("planned result exceeds file budget")
    diff = "".join(difflib.unified_diff(before["text"].splitlines(keepends=True),
        after.splitlines(keepends=True), fromfile=relative, tofile=relative))
    return EditPlan(relative, before["sha256"], old, new, diff)

def apply_plan(workspace, plan):
    return workspace.replace(plan.path, plan.old, plan.new, plan.expected_sha256)

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    workspace = Workspace(root)
    plan = plan_edit(workspace, "calc.py", "return a - b", "return a + b")
    assert "-    return a - b" in plan.diff and "+    return a + b" in plan.diff
    assert "return a - b" in workspace.read("calc.py")["text"]
    apply_plan(workspace, plan)
    assert "return a + b" in workspace.read("calc.py")["text"]
```

EditPlan 是不可变字段容器，防止普通代码在审批后悄悄修改参数；但 frozen 不会自动验证字段类型，也不是防恶意进程篡改的安全机制。若计划来自外部 JSON，应先按 schema 验证，再构造对象。审批绑定应覆盖路径、摘要、old 与 new，而不是只绑定展示文本 diff，因为展示可能省略内容或被重新格式化。

plan_edit 先读取实际文件，检查唯一匹配，构造完整新文本并检查字节上限。统一 diff 只用于审查，不作为实际执行的唯一真相；apply_plan 使用原始字段，让 Workspace 再次验证当前内容。计划生成之后文件仍保持旧内容，这个断言证明“准备可审查结果”和“产生副作用”确实分离。

splitlines(keepends=True) 保留行结束，使 diff 更忠实。没有末尾换行的文件可能产生不够直观的展示，生产审查器应明确标注这种情况；不能为了美观自动给所有文件补换行，除非任务授权且语义允许。换行风格、编码和权限也可能是项目约束，编辑工具应说明是否保留。当前 Workspace 将文本按 UTF-8 编码写回，不能保证所有原始编码与特殊元数据都原样保留。

这个项目只应用一个文件的一次替换。若要多个计划一起应用，应先全部验证并在隔离工作副本中测试，再决定提交策略。简单循环 apply_plan 不是多文件事务：第二个失败时第一个已经写入。你可以报告部分成功并保留 diff，或在明确单写入者环境中设计备份恢复，但恢复本身也可能失败。A10 的事务概念会帮助你区分哪些对象能由同一数据库原子提交，哪些跨文件系统与外部服务不能。

## 第十二课：版本控制提供审查与回退，不提供运行隔离

Git 能记录文件变化、比较版本、建立分支和提交，但不会限制测试代码的系统权限。一个独立 worktree 可以避免把实验改动混入用户当前工作目录，却仍在同一操作系统用户权限下运行。不要把“开了新分支”当作沙箱。版本隔离解决协作与审查，执行隔离解决权限与资源，两者经常需要一起使用。

Agent 开始修改之前应读取仓库规则、查看工作区已有改动和任务相关文件。已有未提交修改可能属于用户，不能为了让测试干净就重置或覆盖。若要创建独立工作副本，应保留用户原状态，并只在授权范围内处理冲突。对初学者，先养成查看 diff 的习惯：修改前知道基线，修改后知道具体变化，测试后知道哪些变化被验证。

提交信息应描述问题与行为变化，而不是只写“更新文件”。例如“修正加法函数误用减法，新增零值与负数回归”让审查者知道为什么改和怎样验证。不要把运行日志、密钥、公司端点或私人数据放进提交。生成的临时文件和测试输出也应按项目规则处理，避免无关噪声掩盖实际补丁。

回退也要区分目标。撤销某次代码修改、恢复工作区某个文件、回滚已部署系统和补偿外部业务动作，是不同层次。Git 可以帮助恢复代码版本，但不能撤销已经发出的邮件或数据库外部写入。编码 Agent 若执行了迁移或部署，需要另有业务恢复方案，不能只说“有 Git 所以安全”。本章实验只修改临时文本，没有跨系统回滚承诺。

代码审查应看最终 diff，而不只看模型解释。模型可能说只改一个函数，实际误删了测试；也可能做了合理修改但遗漏文件。宿主可以统计修改文件列表、行数、类型和敏感路径，再与任务范围比较。自动规则可以发现明显越界，人类或专门 verifier 再判断语义。A14 将把这个流程扩展为完整编码 Agent。

## 第十三课：测试命令必须有明确的环境合同

同一段代码在不同 Python 版本、依赖和工作目录下可能表现不同。运行测试前应确认解释器、依赖来源、当前目录和环境变量。课程包装器使用 sys.executable，保证启动的是当前解释器，而不是 PATH 中碰巧叫 python 的另一个程序；使用绝对测试目录和明确 cwd，减少相对路径歧义。它不负责自动安装依赖，也不保证陌生项目在隔离导入模式下直接可用。

测试失败可能来自代码缺陷，也可能来自环境缺失。ModuleNotFoundError 可能说明依赖未安装，不能立即通过删除 import 或跳过测试“修复”。应查看项目依赖文件与运行说明，建立最小可复现环境，再判断失败是否属于目标问题。环境修复应记录具体变化，避免为了通过测试任意升级所有依赖，导致行为同时变化难以归因。

环境变量可能包含凭证和配置。给子进程继承整个宿主环境很方便，也可能把不需要的秘密传给测试。生产执行器可以构造最小环境，只传必要 PATH、语言区域和明确配置；但这会改变程序行为，需要记录。课程 -I 忽略部分 Python 环境影响，不代表所有环境变量都不会被测试代码读取，因此仍不能放入真实秘密后执行不可信测试。

工作目录影响相对文件访问。测试在仓库根运行和在 tests 目录运行可能读取不同配置；程序可能错误依赖当前目录。一个好的测试命令应固定 cwd，并让应用通过明确路径定位资源。若为了让测试通过不断切换目录，却没有解释为什么，可能只是掩盖路径设计问题。你应能画出进程启动时的可执行文件、参数、cwd、环境和允许资源五项合同。

输出编码也需要区分用途。课程读取测试日志使用 errors=replace，目的是让诊断文字尽可能可展示；文件编辑读取使用严格 UTF-8，目的是防止损坏内容被写回。两种策略不同且合理，因为日志显示允许替代未知字节，代码修改不能静默改变原文。不要把一个全局“统一解码策略”套在所有数据上。

## 第十四课：资源限制应覆盖 CPU、内存、磁盘与进程数量

一个测试即使没有网络权限，也可能无限循环、分配巨大内存、写满磁盘或创建很多进程。timeout 只覆盖时间的一部分，输出截断只覆盖交付文本的一部分，文件读取上限只覆盖单文件的一部分。完整执行环境应按任务设置 CPU 时间、内存、磁盘、进程数量和网络限制，并记录超限原因。不同平台提供不同机制，需要实际验证，而不是从参数名字推断保证。

对你的共享 Linux 服务器，尤其要区分本任务资源和公司其他任务。实验应该使用独立目录、明确进程身份和受限工作负载；清理只针对自己创建的资源。不要扫描并终止看起来占 GPU 的进程，也不要把系统目录作为临时工作区。Agent 基础练习完全可以在 CPU 上进行，H100 留给确有必要的模型或算子任务。

资源预算可以进入工具合同。read_file 声明最多读取多少字节，run_tests 声明最大时间和返回日志大小，搜索工具声明最多返回多少条，编辑工具声明允许哪些扩展名或路径。模型知道这些限制后可以分批请求，但分批不能绕过任务总预算；执行器仍需累计统计。单次上限与总量上限分别防止大请求和大量小请求，两者都需要。

如果进程超时，保存已经产生的受控日志有助于诊断，但不要把超时当成测试失败的唯一原因。可能是死循环、外部服务等待、锁竞争或环境启动慢。下一步应先构造更小复现和检查最后进展，而不是无条件把 timeout 翻倍。延长时间只有在知道任务确实需要且资源允许时才合理。

测试结束后检查临时文件、后台进程和资源句柄是否清理。课程 with 结构帮助关闭文件和临时目录，但真正的进程树管理需要执行器负责。泄漏在单次实验中可能不明显，多轮 Agent 循环会放大它。可靠性验收应包含重复运行多次后的资源稳定性，但不要用不必要的大负载压测共享服务器。

## 第十五课：完成一个可审查的微型修复任务

综合任务是修复临时项目中的加法函数。先写入错误实现和一个可信测试，运行测试确认失败；再读取文件生成 EditPlan，检查 diff 只涉及目标行；应用计划后运行同一测试，确认通过；最后记录修改前后摘要、退出码和修改范围。这个顺序证明测试能够暴露原缺陷，也证明修复改变了实际行为，而不是只让模型输出一段看似正确解释。

你可以把前面练习十一的测试扩展为调用 calc.py，但要注意隔离导入模式与目录搜索路径。最透明的办法是在测试里用 importlib.util 按明确文件路径加载本题创建的模块，避免依赖当前目录自动进入导入路径。这个加载动作仍然执行模块顶层代码，所以只能用于本题可信文件；路径明确不等于代码可信。

验收时还要做一次过期计划实验：生成计划后修改目标文件，再尝试应用，必须失败且保留新内容。随后重新读取并生成新计划，不能偷偷更新旧计划的 expected_sha256。这样你不仅会修一个函数，还掌握了观察、计划、审批或政策检查、提交、测试和冲突处理的完整流程。

通过本章的门槛不是背出 os.replace 或 subprocess.run 的参数，而是能解释每个动作对应的保证。路径守卫限制哪些输入，摘要检查发现哪类过期，唯一匹配避免哪类歧义，临时替换提供哪种可见性，测试退出码证明什么，超时无法证明什么。你应能在看到一个“安全 shell 工具”或“沙箱工作区”宣传时追问其具体权限、并发、资源和恢复合同，并用最小实验验证。

综合任务的完整实现如下。测试通过文件自身位置找到 calc.py，不依赖用户目录。前后两次运行同一份测试，修复只改变实现文件；若为了通过而修改断言，便不满足这个任务的验收。所有文件都由本段代码创建在临时目录中，运行完成自动清理。

```python
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    test_source = '''import unittest
import importlib.util
from pathlib import Path
path = Path(__file__).with_name("calc.py")
spec = importlib.util.spec_from_file_location("fixture_calc", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class AdditionTests(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(module.add(2, 3), 5)
    def test_zero(self):
        self.assertEqual(module.add(0, 4), 4)
    def test_negative(self):
        self.assertEqual(module.add(-2, -3), -5)
'''
    (root / "test_calc.py").write_text(test_source, encoding="utf-8")
    workspace = Workspace(root)
    before_result = run_trusted_fixture_tests(root)
    assert before_result["exit_code"] != 0
    plan = plan_edit(workspace, "calc.py", "return a - b", "return a + b")
    before_hash = plan.expected_sha256
    applied = apply_plan(workspace, plan)
    after_result = run_trusted_fixture_tests(root)
    assert after_result["exit_code"] == 0
    assert applied["sha256"] != before_hash
    assert (root / "test_calc.py").read_text(encoding="utf-8") == test_source
```

逐行解释测试加载过程：spec_from_file_location 根据明确路径建立模块描述，module_from_spec 创建模块对象，exec_module 执行该文件并把定义放进模块。随后 unittest 调用 module.add。三个用例分别覆盖正数、零和负数，避免只用一个偶然能通过减法的输入，例如第二个参数为零。测试设计应选择能够区分错误实现与正确实现的样本，不能只是镜像你刚写的代码。

这里还检查测试文件保持原样，防止修复通过删除测试获得表面成功。真实仓库当然可能需要新增或修正错误测试，但必须解释测试为何不符合需求，而不能为了绿灯随意降低标准。A14 会进一步讨论环境问题、测试问题和实现问题的区分，本章先用一个完全明确的小任务建立证据链。

最终交付报告应包含问题、修改、验证和边界。问题是加法误用了减法；修改是唯一目标行；验证是原有三个用例修复前失败、修复后通过；边界是仅验证本题输入类别，没有证明任意 Python 对象都支持预期加法，也没有做性能基准。报告无需夸大成“彻底解决全部数学问题”，准确范围更有价值。

如果综合任务中任一测试命令异常退出，先保留受控输出并确认是测试失败还是包装器错误。退出码非零加断言失败通常指向业务行为；模块加载错误可能指向路径或语法；超时可能指向执行挂起。不要把所有情况都映射成“加法仍然错误”，否则下一轮 Agent 会在错误组件上反复修改。诊断应沿着进程启动、模块加载、测试发现、用例执行、结果汇总的顺序逐层核实。

这条证据链最终连接本卷前面各章：A03 定义编辑参数合同，A04 保存计划与状态，A05 区分草稿和提交，A06 处理异常与不确定效果，A07 把必要代码和测试观察送入模型，本章提供受控文件与执行能力。之后的并发、持久化和工作流是在这些明确边界上组合，不是把更多工具堆进提示词就自然获得工程能力。

可选的符号链接实验应在支持该能力且具有权限的平台执行。Windows 的创建权限和配置可能不同，不能把创建链接失败当作 Workspace 防护通过；只有链接实际创建成功，再观察 read 被拒绝，才覆盖了目标路径。为了保持本章主实验跨环境可运行，这里把它作为明确的环境条件测试，不要求读者提升管理员权限。你可以在 Linux 临时目录创建一个指向同目录普通文件的链接，课程即使目标仍在根内也会拒绝，证明策略是“不支持链接”而不是“仅拒绝外部链接”。

另一个容易忽略的范围是 root 本身。构造函数会 resolve 根目录，因此若传入的是一个链接根，它绑定的是解析后的真实目录；后续逐组件检查从这个真实根开始。不能把“子路径禁止符号链接”误读成“输入根路径从未经过链接”。若产品要求根目录必须由宿主预先登记的真实位置提供，应在创建 Workspace 前验证来源与配置权限。模型不应通过参数自行选择任意 root，工具注册时就应由宿主固定它。

同样，max_bytes 应由宿主配置并验证为合理正整数，不应让模型通过工具参数把读取上限改成无限。当前教学类假设构造参数可信，网络服务入口需要补上这项配置验证。
