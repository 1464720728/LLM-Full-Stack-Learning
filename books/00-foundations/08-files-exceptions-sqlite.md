# F08 文件、异常、上下文与 SQLite：失败时也保持数据正确

> 先修 F02、F06、F07。分四次学习：异常路径，资源与文件，关系数据与 SQL，事务与恢复。离线实验：`python -m labs.foundations.file_io`、`python -m labs.foundations.sqlite_lab`。所有破坏性练习只操作自己创建的临时目录或内存数据库。

## 1. 从成功返回走到失败传播

普通函数通过 return 把结果交给调用者，但很多操作并不总能产生合法结果。将文字转换成整数时可能遇到 hello，读取路径时文件可能不存在，保存内容时磁盘可能写满。若把这些情况一律转换为零、空字符串或空列表，调用者就无法区分“合法结果恰好为空”和“操作根本没有完成”。异常为失败提供一条独立控制流。

异常对象保存错误类型及相关信息；raise 发起异常传播。解释器沿当前调用链寻找能匹配的 except。如果当前函数没有处理，当前调用被中断，异常继续交给外层；最终无人处理时，程序入口打印堆栈并失败退出。传播不是把整个进程已经发生的所有修改回滚：在失败之前写入列表、打印终端、发送请求的行为可能已经发生。

```python
def parse_count(text):
    return int(text)

def doubled_count(text):
    count = parse_count(text)
    return count * 2

try:
    result = doubled_count("hello")
except ValueError as error:
    print(type(error).__name__, str(error))
```

逐步看：入口调用 doubled_count，后者调用 parse_count，int 无法解析 hello，抛出 ValueError。parse_count 没有返回整数，doubled_count 的乘法没有执行，赋给 result 的赋值也没有完成。异常越过这两层后被入口捕获，error 在处理块中指向异常对象。不要在后面假定 result 已经获得值；失败路径与成功路径产生的变量状态不同。

错误类型形成继承层次。FileNotFoundError 是 OSError 的一种，JSONDecodeError 是 ValueError 的一种。捕获较宽父类会接住更多情况，捕获具体子类则更容易保留策略差异。例如文件不存在可以提示路径，权限不足应提示权限与目录，JSON 损坏应提示格式位置；三者都叫“读取失败”虽然简单，却削弱诊断能力。

`except Exception` 可以在程序最外层记录意外错误并非零退出，但在核心函数内部随处吞掉所有异常会隐藏代码缺陷。更不能把 KeyboardInterrupt 等控制退出信号随意吞掉。设计捕获范围时，问自己是否知道如何恢复；若只是因为“不想看到红字”，通常应该让错误继续传播，先修复根因。

## 2. try、except、else、finally 的精确顺序

try 块放可能失败且你确实打算处理的操作。except 只处理匹配异常，else 只在 try 正常结束时执行，finally 用于无论哪条路径都应执行的清理。把整个函数都放进一个巨大的 try，容易误把后面逻辑缺陷当作前面文件读取错误处理；缩小 try 范围能让失败来源更明确。

```python
events = []
try:
    events.append("try")
    number = int("12")
except ValueError:
    events.append("except")
else:
    events.append(f"else:{number}")
finally:
    events.append("finally")
assert events == ["try", "else:12", "finally"]
```

把十二改成 x，事件变为 try、except、finally，else 不出现。把 except 换成只捕获 TypeError，ValueError 就会继续传播，但 finally 仍会执行。这个实验要在自己的小文件中一次只改变一个条件，才能将观察到的顺序与语法对应起来。

finally 通常用于释放资源，不应该随意 return。若 finally 返回新值，可能覆盖 try 中原本返回值，甚至压住正在传播的异常，使失败看起来像成功。同样，清理时又抛出新异常可能遮住原先原因。稳妥设计应尽量保持清理简单，必要时保留原始因果链，并让诊断信息说明两个失败分别发生在哪里。

使用 `raise ... from error` 可以在转换异常时保留因果关系。低层转换失败告诉你“不是整数”，上层可以补充“字段 batch_size 非法”。这是增加语义，而不是删除证据。若你只写新的消息却丢掉原始异常，排查时就难以知道是解码、文件读取还是逻辑计算先出错。

```python
def parse_positive_count(text):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError("count must be an integer") from error
    if value <= 0:
        raise ValueError("count must be positive")
    return value
```

这个函数把“无法解析”和“解析成功但范围不合法”分开。第二种情况不需要假装来自 int 的异常，因为 int 正常工作了。错误信息应定位合同违反的位置，并提供调用者能采取行动的信息；无需把完整私密输入无条件放进日志，尤其当文本可能包含密钥或用户数据时。

## 3. 文件对象是一项需要结束的资源

路径描述位置，打开文件得到的对象代表一次打开状态，包括读取模式、编码、当前位置以及底层资源。两个独立打开的文件对象可以各自有位置；同一个对象被两段代码共享时，读写会相互推进位置。关闭文件不是删除文件，它结束这次资源使用，使底层文件描述符等资源可以被回收。

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "notes.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        rest = handle.read()
    assert first == "first\n"
    assert rest == "second\n"
    assert handle.closed
```

外层 with 管理临时目录，内层 with 管理文件。进入内层时打开文件，将返回的对象绑定给 handle；执行缩进体；离开时关闭，即使体内发生异常也会执行退出协议。handle 这个变量在块外仍然可能存在，但它指向的是已关闭对象。with 管理资源生命周期，不是像某些语言那样自动建立全新变量作用域。

文件模式必须按意图选择。r 读取且要求文件存在，w 写入并先截断旧内容，a 追加，x 要求创建新文件且已存在就失败；b 表示字节模式，默认文本模式需要编码转换。看到 w 时应立即想到“旧内容何时失去”，不是只想到“可以写文件”。错误发生在 write 之前，也可能已经因打开方式导致旧内容被清空。

文件对象迭代按行提供文本，通常比 read 全部内容更适合大文件。但一行本身如果非常长，逐行读取也可能占用大量内存；输入大小限制应根据威胁和业务需求设定。教学日志的行都很短，不代表任何来自外部的文本都具备相同边界。不要把样例规模当作真实系统保证。

## 4. 上下文管理器的协议与所有权

with 调用对象的进入方法，随后执行主体，最后调用退出方法。进入方法的返回值绑定到 as 后的名字，不一定就是原对象。退出方法收到异常信息，可以做清理；如果它明确返回真值，还可能抑制异常。文件上下文通常关闭资源并让异常继续传播，所以“用了 with 就不会报错”是错误理解。

```python
class RecordingContext:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter")
        return "resource"

    def __exit__(self, error_type, error, traceback):
        self.events.append("exit:" + (error_type.__name__ if error_type else "ok"))
        return False

events = []
try:
    with RecordingContext(events) as value:
        assert value == "resource"
        events.append("body")
        raise ValueError("demo")
except ValueError:
    events.append("caught")
assert events == ["enter", "body", "exit:ValueError", "caught"]
```

这里 exit 返回 False，告诉解释器不要把 ValueError 当作已处理。若返回 True，外层 except 就不会执行，这可能适合某些明确容错的上下文，但不能随意使用。你应该能从事件列表解释资源退出与异常捕获的先后：先离开正在使用资源的范围，再到外层决定业务处理策略。

资源所有权决定谁关闭。函数如果接收调用者传进来的已打开文件，通常不应擅自关闭，因为调用者可能还要继续使用；若函数自行根据路径打开文件，则应在清楚的边界结束资源。参数命名可以帮助表达这种差异，例如 path 表示位置，stream 表示已打开的输入源，但真正合同仍应写明。

数据库连接的 with 与文件的 with 不一定具有相同语义。标准库 sqlite3 连接上下文用于事务提交或回滚，不等于退出时关闭连接。本仓库演示在 finally 中显式 close。认识协议后就不会机械推断“任何对象放到 with 中都会关闭”，而会查看这个对象的退出行为究竟是什么。

## 5. JSON：语法正确之后还要验证数据合同

JSON 保存对象、数组、字符串、数字、布尔和 null，不保存任意 Python 类、函数或循环引用。loads 将字符串解析成 Python 值，load 从文件对象读取并解析；dumps 返回字符串，dump 写到文件对象。名字末尾的 s 可以帮助记忆字符串版本，但你仍应看函数接收的是文本还是文件对象，避免把路径字符串误当成文件内容。

```python
import json

text = '{"name":"学习","count":3,"done":false}'
value = json.loads(text)
assert value == {"name": "学习", "count": 3, "done": False}
assert json.loads(json.dumps(value, ensure_ascii=False)) == value
```

JSON 的 false 转成 Python 的 False，null 转成 None；对象键通常应按字符串理解。解析成功只说明语法成立，不说明字段齐全、类型正确、范围合法或版本兼容。例如 `{"count": "3"}` 与 `{"count": 3}` 都是合法 JSON，但你的计数合同可能只接受整数。把解析与验证分成两步，错误就更容易定位。

严格检查整数时要考虑 bool。Python 的 True 能参与整数计算，但外部协议中的 true 通常不是“数量一”。可以使用 `type(value) is int` 表达这里的严格要求。字段集合也要明确：允许未知字段便于兼容，但可能隐藏拼写错误；拒绝未知字段更严格，却要求版本迁移有计划。没有一种策略适合所有协议，应明确选择并测试。

浮点特殊值需要注意。Python JSON 默认编码行为可能接受某些非标准特殊数值；仓库 atomic_json 使用 `allow_nan=False`，让 NaN 与 Infinity 失败，避免生成外部严格解析器无法接受的文件。解码外部内容时也应按合同拒绝不允许的数值，而不能只靠编码端设置保护全部入口。

序列化往返不一定保留原始 Python 对象的全部性质。元组可能变成列表，自定义类需要显式转换，字典键类型可能被改变。持久化格式应保存领域数据，而不是直接倾倒整个内存对象。为检查点加入 schema_version，能让程序识别版本，而不是新代码拿旧字段结构继续运行后才在深处出错。

## 6. 原地覆盖的失败窗口与原子替换

假设进度文件当前是合法 JSON。使用 w 打开最终路径后，旧内容被截断；之后即使只写了一半就失败，最终名字仍指向这个不完整文件。此时新旧两份完整内容都不在目标位置。读者可能在写到一半时打开文件，看到缺少右括号的 JSON。我们需要将“准备新内容”和“让新内容成为可见版本”分开。

原子替换模式先在同一目录写临时文件，完整编码、刷新并关闭后，再把临时名字替换到目标路径。读者通过最终名字打开时，应看到旧版本或新版本，而不是我们分多次写入的中间内容。它保护的是发布单个文件版本的观察边界，不会自动把多个文件的替换变成同一个事务。

仓库 atomic_json 首先把路径转换为 Path，并确保父目录存在。temporary 初始为 None，用来区分临时文件是否创建成功。NamedTemporaryFile 负责生成独特名字，`delete=False` 让退出文件上下文后名字仍在，方便关闭后替换。这也意味着失败清理由我们承担，不能假定上下文会替我们删除所有残留。

`json.dump` 写入临时文件，`ensure_ascii=False` 保留可读中文，`allow_nan=False` 拒绝特殊数值。紧接着写一个换行，方便终端查看。flush 将 Python 层缓冲推动到底层，fsync 请求同步文件内容；这两者不是同义词。仅关闭 Python 文件不等于对所有文件系统、所有断电情况作出完整持久性承诺。

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from labs.foundations.file_io import atomic_json

with TemporaryDirectory() as directory:
    path = Path(directory) / "progress.json"
    atomic_json(path, {"step": 1})
    try:
        atomic_json(path, {"step": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("NaN must be rejected")
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 1}
    assert sorted(item.name for item in path.parent.iterdir()) == ["progress.json"]
```

这个验证制造的是序列化失败，确认旧文件不变且临时文件被清理。它没有模拟突然断电、内核崩溃或网络文件系统故障，所以不能把测试通过写成“所有存储失败都不会丢数据”。实现没有对父目录做 fsync，完整断电持久性边界应如实保留。可靠工程包括知道你实际证明了什么。

临时文件放在目标父目录，是为了让替换发生在同一文件系统中。跨文件系统移动可能不能使用相同的原子替换保证。finally 只删除这次明确创建的临时名字，不递归清空整个目录；replace 成功后该临时名字已经消失，missing_ok 允许这种正常情况。清理范围小而可追踪，是避免“修复失败却删除别的数据”的关键。

## 7. 从 JSON 到关系表：一条记录怎样被描述

小配置文件适合 JSON，但当你需要按条件找许多记录、保证请求标识唯一、同时修改多行或防止无效状态，数据库更容易把规则集中执行。SQLite 是嵌入式数据库，应用通过库访问数据库文件，不需要先启动独立数据库服务器。内存数据库适合可重复练习，文件数据库适合观察关闭后重新打开的数据保留。

一张表有列定义和若干行。列描述字段，行表示一条记录。主键用于标识记录，NOT NULL 拒绝缺失值，UNIQUE 限制重复，CHECK 表达合法范围。这些约束不是注释，它们参与写入检查，即使另一个代码入口忘记验证，也不能轻易写入违反约束的值。应用验证负责友好提示，数据库约束负责最后一道一致性检查。

```python
import sqlite3

connection = sqlite3.connect(":memory:")
try:
    connection.execute("""
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'done'))
        )
    """)
    with connection:
        connection.execute(
            "INSERT INTO tasks(title, status) VALUES (?, ?)",
            ("read chapter", "pending"),
        )
    rows = connection.execute("SELECT id, title, status FROM tasks ORDER BY id").fetchall()
    assert rows == [(1, "read chapter", "pending")]
finally:
    connection.close()
```

多行字符串里的 SQL 不是 Python 语法，它由 SQLite 解释。Python 负责建立连接、传入 SQL 与参数、取回结果。fetchall 返回结果行列表，本例每行是元组，按 SELECT 中的列顺序排列。不要把元组下标含义靠记忆散布到很多函数中；项目变大后可集中转换为明确的数据结构。

SQLite 的类型体系与强静态类型数据库并不完全相同。本章用明确约束与应用验证维护合同，不声称写了 INTEGER 就自动排除所有不希望出现的输入。特别是标识字段与空值规则，应按实际 schema 检查。约束设计要靠故意插入坏数据验证，而不是只看建表语句觉得“应该已经限制了”。

## 8. SQL 的四种基本操作逐句拆解

INSERT 添加记录；SELECT 读取；UPDATE 修改；DELETE 删除。WHERE 指定条件，不写 WHERE 的 UPDATE 或 DELETE 会作用于全部匹配范围内的行。在练习中先用 SELECT 查看同一条件会选中什么，再执行修改，可以训练你把条件作为业务合同，而不是在运行后才发现修改范围过大。

```sql
SELECT id, title
FROM tasks
WHERE status = 'pending'
ORDER BY id
LIMIT 10;
```

SELECT 后面列出想得到的列；FROM 指明表；WHERE 保留状态为 pending 的行；ORDER BY 明确排序；LIMIT 最多返回十行。这里的书写顺序不是要求你假想数据库必须逐行按这个顺序执行算法，优化器可以选择执行策略；它描述的是查询语义。没有 ORDER BY 时，不能依赖当前观察到的行顺序。

UPDATE 可以只修改某一列，例如把指定 id 的状态改成 done。执行成功不代表真的找到了记录：条件可能匹配零行，所以业务层要按需要检查影响行数或随后查询。DELETE 同样如此。把“SQL 无语法错误”当作“任务已经存在并成功更新”会让 API 返回虚假的成功。

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE scores(name TEXT PRIMARY KEY, score INTEGER NOT NULL)")
    with db:
        db.executemany("INSERT INTO scores VALUES (?, ?)", [("a", 2), ("b", 5), ("c", 3)])
        cursor = db.execute("UPDATE scores SET score = score + ? WHERE name = ?", (4, "a"))
        assert cursor.rowcount == 1
    assert db.execute("SELECT name, score FROM scores ORDER BY score DESC, name ASC").fetchall() == [("a", 6), ("b", 5), ("c", 3)]
finally:
    db.close()
```

`score = score + ?` 在数据库表达式中使用原列值加增量，适合避免先读到 Python 再写回的多余往返。排序先按分数降序，分数相同再按名字升序，给相同分值定义稳定的展示顺序。稳定排序不是数据库默认赠送的，需要你在语义中明确要求。

## 9. 参数化：把数据与 SQL 结构分离

问号占位符代表一个值，真正数据通过 execute 的第二个参数单独传入。数据库驱动负责把值按数据处理。字符串里面即使包含引号、分号或看起来像 SQL 的文字，也不应该因此变成新的命令。不要用 f-string 拼用户标题、用户名或请求标识进入 SQL 语句。

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE notes(text TEXT NOT NULL)")
    unusual = "Robert'); DROP TABLE notes;--"
    with db:
        db.execute("INSERT INTO notes(text) VALUES (?)", (unusual,))
    assert db.execute("SELECT text FROM notes").fetchone() == (unusual,)
finally:
    db.close()
```

`(unusual,)` 的尾部逗号很重要：它构造一个只有一个元素的元组；没有逗号只是给表达式加括号。驱动需要一组参数，而不是把字符串逐字符误当成参数集合。遇到参数数量不匹配时，先数 SQL 占位符，再检查第二个参数的容器形状，不要通过删除引号随机试错。

值占位符通常不能替代表名或列名。`ORDER BY ?` 绑定字符串不等于动态选择这个名字对应的列，它可能只是对一个常量排序。若确需允许用户选择排序字段，用固定字典把外部选项映射成开发者定义的 SQL 片段，并拒绝未知选项。数据参数化和结构白名单解决不同问题，不能互相替代。

参数化也不负责业务授权。一个合法参数化查询若按用户提供的任意任务 id 返回结果，仍可能泄露不属于该用户的数据。数据与语法分离防止注入，权限条件则决定谁可以操作哪些行。基础项目先把这两层区分清楚，避免用一个安全术语替代全部设计。

## 10. 事务：把多个修改作为一个逻辑整体

事务把若干数据库操作组织为共同提交或共同回滚的单位。仓库 sqlite_lab 先插入 tokens=10 并提交，然后开启一个逻辑事务：先改成二十，再插入负值。第二条违反 CHECK，异常离开连接上下文，未提交修改回滚，所以最终 tokens 仍是十。这里回滚的是这个数据库事务中的修改，不是 Python 列表、终端输出或外部 HTTP 请求。

| 时刻 | 当前动作 | 本事务看到 tokens | 已提交版本 |
|---|---|---:|---:|
| 起点 | 初始插入已提交 | 10 | 10 |
| 修改后 | UPDATE 为 20 | 20 | 10 |
| 失败 | 插入负值违反约束 | 事务失败 | 10 |
| 退出 | 回滚未提交修改 | 10 | 10 |

如果你在 UPDATE 后立即 commit，随后失败就无法通过后一个事务撤销已经提交的二十。事务边界由业务定义：例如从账户甲扣一单位同时给乙加一单位，通常必须一起成功，否则总量被破坏。但真正财务系统还有更多要求；这里用简单库存或计数示例训练原子修改，不把教学代码包装成金融级实现。

连接上下文中的异常若在内部被你捕获并吞掉，退出时看到的就可能是正常路径，从而提交先前成功的语句。想让失败触发回滚，应让异常离开事务边界，或显式执行符合设计的回滚。捕获位置因此会改变数据结果，不是纯粹的代码排版问题。画出异常传播与 commit 的时间线，才能判断实际行为。

并发事务还涉及可见性与锁。SQLite 允许的写入并发、连接线程使用规则与具体事务模式需要按实际版本和配置理解；本课程的最小服务采用串行处理减小范围。不要仅因有数据库事务，就声称任意多个线程共用同一个连接都安全，也不要把 locked 错误一律用无限重试掩盖。

数据库事务不能跨越不参与同一事务的外部系统。先发邮件再插数据库，数据库回滚不会撤回邮件；先写数据库再发请求，请求失败时记录可能已提交。后续 Agent 的工具调用需要状态机、幂等键与恢复流程。现在先学会标出每个动作所属的提交边界，避免把“用了事务”误当成全世界的一键撤销。

## 11. 索引、聚合与关系：从能查到到知道如何组织

主键和索引帮助数据库定位记录。没有合适索引时，按条件查找可能需要扫描很多行；有索引则可以按有序结构更快缩小范围。但索引也占空间，并增加插入、更新、删除的维护成本。不是给每一列都建索引就最好，而应结合实际查询条件、排序与数据规模判断。

可以用 `EXPLAIN QUERY PLAN` 观察 SQLite 选择的计划。计划输出用于理解当前版本和数据条件下的执行方式，不应把某个具体文字顺序写成跨版本永远不变的测试。性能验证应使用代表性数据，确认结果正确后再测耗时。几十行样例中没有明显加速，不代表索引理论错误；它可能小到固定开销占主导。

COUNT、SUM 等聚合把多行归纳为结果。GROUP BY 按键分组，例如按状态统计任务数量。注意 COUNT(*) 统计行数，COUNT(column) 通常只统计该列非空的行；空值处理会改变结果。SQL 的 NULL 不是普通字符串 "null"，与 NULL 比较通常用 IS NULL，而不是期待 `= NULL` 像普通数字相等判断一样工作。

把任务和事件放到两张表时，用任务 id 关联。JOIN 按条件组合行，不是简单把两个列表按相同位置拼起来。若一个任务有三个事件，连接结果就可能出现三行任务信息。理解这种一对多关系非常重要，否则你在连接后直接计数任务，会把同一任务算多次。先手画三条记录，再写查询，比盲调聚合函数更可靠。

规范化的直观目标是减少同一事实在很多地方重复保存。任务标题放任务表，事件只引用任务 id，改标题时不用同步修改每个事件。但有些历史记录需要保存当时快照，那是明确的业务需求，不是错误冗余。数据设计要区分“当前实体状态”和“过去发生事件”，这对 Agent 运行轨迹尤其重要。

## 12. 四个故障现场：证据先于修复

**文件不存在。** 记录 cwd 和解析后的路径，确认你操作的是哪台机器的哪份工程。若路径正确而文件确实缺失，决定它是必需输入还是可以初始化的状态。不要捕获 FileNotFoundError 后一律创建空文件，因为这可能把配置路径拼错伪装成合法空配置。必需配置缺失应清楚失败，可选缓存缺失才适合重新生成。

**JSON 读取失败。** 先区别字节解码错误与 JSON 语法错误，再查看错误位置附近的有限片段。不要把包含密钥的整个配置打印到共享日志。若最终文件尾部截断，回查写入是否使用原地覆盖，是否有并发写者，是否有完整提交标记。简单添加一个右括号通常不能证明业务内容完整，应从可靠旧版本恢复。

**数据库约束失败。** 记录违反的是唯一、非空还是检查条件。重复请求 id 可能表示安全重试，也可能表示同 id 不同参数冲突，应用必须读取原记录并比较合同字段。不要将所有 IntegrityError 都翻译为“任务已经存在”，因为其他约束失败具有不同原因。错误分类应依据实际约束和业务状态。

**数据库被锁。** 检查是否有未结束事务、长时间持有写事务、其他进程仍访问文件。把网络请求或漫长计算放在写事务中，会延长占锁时间。修复应缩短事务、明确连接生命周期并设置有界等待，而不是不停删除数据库文件或无限循环重试。学习项目的数据也应在操作前保留可重建来源。

## 13. 十二道练习及完整解答

### 一：分别处理文件缺失与格式错误

输入路径若不存在返回明确标签，若存在但不是 JSON 返回另一标签；其他错误不吞掉。

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

def inspect_json(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "missing"
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json"
    return "valid"

with TemporaryDirectory() as directory:
    path = Path(directory) / "value.json"
    assert inspect_json(path) == "missing"
    path.write_text("{", encoding="utf-8")
    assert inspect_json(path) == "invalid_json"
    path.write_text("{}", encoding="utf-8")
    assert inspect_json(path) == "valid"
```

分别设置两个 try，让处理策略只针对预期操作。权限错误仍传播，因为函数没有获得“权限错误等于缺失”的授权；解码错误也不被误标为 JSON 语法错误。这是保留错误层次的具体练习。

### 二：证明异常不自动撤销列表修改

```python
values = []
try:
    values.append("already happened")
    raise ValueError("later failure")
except ValueError:
    pass
assert values == ["already happened"]
```

append 已经改变对象，异常只改变后续控制流。若想实现失败不修改，应先计算和验证，再提交修改；或者明确记录补偿步骤。不要把数据库回滚规则套到所有 Python 对象上。

### 三：验证文件在异常后关闭

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "x.txt"
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("partial")
            raise RuntimeError("demo")
    except RuntimeError:
        pass
    assert handle.closed
    assert path.read_text(encoding="utf-8") == "partial"
```

关闭成功不表示内容回滚。这个答案同时证明资源释放与数据原子性是两种合同：with 解决前者，临时文件替换或事务才解决相应的后者。

### 四：严格验证一个检查点

```python
def validate_checkpoint(value):
    if type(value) is not dict or set(value) != {"schema_version", "processed_lines", "input_sha256"}:
        raise ValueError("invalid fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported schema")
    if type(value["processed_lines"]) is not int or value["processed_lines"] < 0:
        raise ValueError("invalid line count")
    digest = value["input_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid digest")
    return dict(value)

assert validate_checkpoint({"schema_version": 1, "processed_lines": 3, "input_sha256": "0" * 64})["processed_lines"] == 3
```

返回浅拷贝避免调用者与输入字典共享外层容器。本结构字段全是不可变标量，因此浅拷贝足够表达这里的隔离需求。真实恢复还必须计算当前输入摘要并比较，仅检查摘要格式不能证明文件相同。

### 五：原子保存失败保留旧文件

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from labs.foundations.file_io import atomic_json

with TemporaryDirectory() as directory:
    target = Path(directory) / "state.json"
    atomic_json(target, {"value": 1})
    try:
        atomic_json(target, {"value": object()})
    except TypeError:
        pass
    else:
        raise AssertionError("unsupported object must fail")
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert len(list(Path(directory).iterdir())) == 1
```

object 实例没有本例支持的 JSON 编码方式，故意触发序列化失败。验收同时检查旧值与临时残留，使测试覆盖发布前的错误清理。

### 六：事务失败恢复旧值

```python
from labs.foundations.sqlite_lab import transaction_demo
assert transaction_demo() == [("tokens", 10)]
```

再打开源码逐句解释为何不是二十：初始十已提交，后续更新与非法插入共享事务，异常离开上下文触发回滚。答案虽然调用简短，解释必须包含提交边界；只背输出不算完成。

### 七：带稳定次序的查询

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE scores(name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    with db:
        db.executemany("INSERT INTO scores VALUES (?, ?)", [("b", 2), ("a", 2), ("c", 1)])
    rows = db.execute("SELECT name, value FROM scores ORDER BY value DESC, name ASC").fetchall()
    assert rows == [("a", 2), ("b", 2), ("c", 1)]
finally:
    db.close()
```

两个分数相同时第二排序键决定顺序。若删去名字排序，不能把偶然结果作为接口保证。稳定顺序对于快照测试、分页和可复现实验都很重要。

### 八：统计分组及空值

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE events(kind TEXT NOT NULL, detail TEXT)")
    with db:
        db.executemany("INSERT INTO events VALUES (?, ?)", [("ok", "a"), ("ok", None), ("error", "x")])
    rows = db.execute("SELECT kind, COUNT(*), COUNT(detail) FROM events GROUP BY kind ORDER BY kind").fetchall()
    assert rows == [("error", 1, 1), ("ok", 2, 1)]
finally:
    db.close()
```

ok 组有两行，却只有一个非空 detail，因此两个计数不同。NULL 表示缺少值，不是长度为零的字符串；把二者混同会使报表结果偏离合同。

### 九：一次转移必须共同成功

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE stock(name TEXT PRIMARY KEY, amount INTEGER NOT NULL CHECK(amount>=0))")
    with db:
        db.executemany("INSERT INTO stock VALUES (?, ?)", [("a", 3), ("b", 0)])
    try:
        with db:
            db.execute("UPDATE stock SET amount=amount-? WHERE name=?", (4, "a"))
            db.execute("UPDATE stock SET amount=amount+? WHERE name=?", (4, "b"))
    except sqlite3.IntegrityError:
        pass
    assert db.execute("SELECT name, amount FROM stock ORDER BY name").fetchall() == [("a", 3), ("b", 0)]
finally:
    db.close()
```

负库存违反约束，整个事务未提交。若你交换两条更新，先增加乙再扣甲，失败后仍应回滚增加；这能证明结果不依赖恰好在第一条就失败。

### 十：安全选择排序字段

```python
def order_clause(option):
    allowed = {"name": "name ASC", "newest": "id DESC"}
    try:
        return allowed[option]
    except KeyError as error:
        raise ValueError("unknown sort option") from error

assert order_clause("name") == "name ASC"
try:
    order_clause("name; DROP TABLE tasks")
except ValueError:
    pass
else:
    raise AssertionError("unknown option must fail")
```

返回值只来自开发者定义的固定片段，外部文字不会直接进入 SQL 结构。业务值仍用参数占位符，两种机制各管一个边界。

### 十一：关闭后重新打开仍可读取

```python
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "state.sqlite3"
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE state(value INTEGER NOT NULL)")
        with db:
            db.execute("INSERT INTO state VALUES (?)", (7,))
    finally:
        db.close()
    db = sqlite3.connect(path)
    try:
        assert db.execute("SELECT value FROM state").fetchall() == [(7,)]
    finally:
        db.close()
```

这验证正常提交和重新连接后的持久化，不等于模拟断电。内存数据库无法用这种重新连接方式保留相同内容，因此要先明确存储介质与生命周期。

### 十二：用连接查询一对多关系

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE tasks(id INTEGER PRIMARY KEY, title TEXT NOT NULL)")
    db.execute("CREATE TABLE events(task_id INTEGER NOT NULL, kind TEXT NOT NULL)")
    with db:
        db.execute("INSERT INTO tasks VALUES (?, ?)", (1, "learn"))
        db.executemany("INSERT INTO events VALUES (?, ?)", [(1, "created"), (1, "done")])
    rows = db.execute("SELECT tasks.title, events.kind FROM tasks JOIN events ON tasks.id=events.task_id ORDER BY events.kind").fetchall()
    assert rows == [("learn", "created"), ("learn", "done")]
finally:
    db.close()
```

同一个任务出现两次，因为它匹配两个事件。这不是重复数据自动被数据库消除的错误，而是连接语义。若目标是统计任务数，应选择符合目标的 distinct 或分组方案，而不是直接计数连接后的全部行。

## 14. 分阶段小项目：可恢复的本地学习进度

第一阶段设计 JSON 合同：版本一、课程 id、完成状态、输入摘要。手写三份有效样例与三份无效样例，先实现纯验证函数，不做文件操作。明确什么叫完成，是看完正文、通过练习还是通过验收；数据字段应表达实际学习证据，不能用一个布尔值掩盖不同阶段。

第二阶段使用 atomic_json 保存，模拟不可编码值失败，证明旧进度保留。第三阶段读取并校验，区别不存在、损坏和版本不支持。不存在可以初始化，损坏不能默默覆盖；先让使用者知道失败并保留原文件，以便诊断与恢复。第四阶段加入输入摘要，防止把一份练习的进度用于另一份内容。

第五阶段改成 SQLite 表，课程 id 为唯一标识，状态受 CHECK 限制，更新时间使用明确约定的文本或数值格式。写入函数只接经过验证的数据，并使用参数化 SQL。第六阶段增加事件表，状态修改和事件插入放进同一事务，制造第二条失败，确认两者都不改变。这里第一次体验“当前状态与历史证据一起提交”的必要性。

最后比较两种存储：JSON 版本容易肉眼审查，适合小型单份状态；数据库版本便于查询多条记录和维护关系。选择依据应是访问模式与一致性需要，而不是认为“数据库一定更专业”。把两者藏在相同上层接口后，可以练习替换，但不要假装它们在并发、性能与事务范围上完全等价。

## 15. 把事务进一步拆开：连接、游标、提交与回滚

连接代表应用与某个数据库之间的会话，不等于数据库文件本身。关闭连接通常释放相关资源，而文件仍存在。游标表示执行语句及获取结果的对象；execute 返回游标，可以 fetchone 取一行，也可以迭代后续行。fetchall 则把剩余结果全部收集到列表。若已经 fetchone 取走一行，再 fetchall 不会重新从第一行开始，这与迭代器进度一致。

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    cursor = db.execute("SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3")
    assert cursor.fetchone() == (1,)
    assert cursor.fetchall() == [(2,), (3,)]
    assert cursor.fetchone() is None
finally:
    db.close()
```

fetchone 没有行时返回 None，不能与“这一行某列为 NULL”混淆。后者通常会得到含 None 的元组，例如 `(None,)`；前者是根本没有下一行。这个区别在判断任务是否存在时尤其重要。把 None 与空元组都笼统当假值可能暂时能运行，但会让接口形状不够明确。

标准库 sqlite3 的事务行为受连接配置与 Python 版本影响，现代接口还有显式事务控制选项。仓库演示采用已测试的默认连接方式，并用 with 管理这组写入；学习时不要在同一练习中随意切换 autocommit、isolation_level 与手工 BEGIN，之后又假定所有上下文行为相同。先记录解释器版本与连接配置，再讨论某次提交发生在哪里。

事务的四个常见性质可以通过具体问题理解。原子性问多个数据库修改能否一起提交；一致性问约束和业务不变量是否被保持；隔离性问并发事务之间能看见什么；持久性问成功提交后数据在相应故障条件下是否保留。数据库机制能提供重要工具，但业务规则若根本没写进约束或操作流程，数据库不能凭空知道什么结果才正确。

例如任务状态从 pending 改成 done，同时插入完成事件，是应用定义的逻辑整体。事务保证两条 SQL 的共同提交，但如果你更新了错误任务 id，数据库可能仍然认为所有约束都满足。正确性还要求 WHERE 条件正确、授权正确、状态转换合法。不要把数据库 ACID 当成对所有应用错误的自动防护。

一个常见竞争是“先查没有，再插入”。两个连接都可能先看见没有，随后同时尝试插入同一标识。仅靠应用层预查不能证明唯一性，必须用数据库 UNIQUE 或主键约束作为最终仲裁，再对重复结果进行业务解释。预查可以改善提示，但不是替代约束的并发保证。

另一种竞争是先读余额到 Python，计算后写回。两个调用都从十读起，各自扣三后写七，最终只体现一次扣减。可以使用受约束的数据库更新表达式，或在恰当事务与隔离策略下处理。理解这个例子时，把两个调用的读取、计算、写入交错排列即可，不必先学习复杂数据库理论。F11 会用更小的共享变量案例重复这种状态分析。

## 16. 表设计补课：空值、外键、约束与迁移

NULL 表示缺少或未知，不等于零、不等于空字符串，也不等于 false。SQL 比较涉及未知结果，因此 WHERE 中的 `column = NULL` 通常不能按初学者期待筛出空值。用 IS NULL 或 IS NOT NULL 明确表达缺失判断。设计时还要问某字段是否真的允许缺失：若每条任务必须有标题，应使用 NOT NULL，而不是让每个调用者后来猜 None 的含义。

主键表达行身份，业务唯一键表达另一种唯一关系。任务 id 可以是数据库生成整数，请求幂等键可以是唯一字符串；它们用途不同。一个任务可能有很多事件，所以事件表自己的事件 id 与关联的 task_id 也不相同。若把这些标识都叫 id 且到处裸传整数，代码很容易把事件编号误用于任务查询；命名应保留领域含义。

外键表示引用关系，例如事件的 task_id 应指向存在的任务。SQLite 外键检查需要按连接正确启用与确认；仅在 schema 写 REFERENCES 而没有验证实际执行状态，可能误以为数据库已经阻止孤儿记录。课程扩展时可在连接初始化执行相关 PRAGMA 并查询确认，随后专门尝试插入不存在的任务 id，观察是否被拒绝。

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("PRAGMA foreign_keys = ON")
    assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
    db.execute("CREATE TABLE tasks(id INTEGER PRIMARY KEY)")
    db.execute("CREATE TABLE events(task_id INTEGER NOT NULL REFERENCES tasks(id))")
    try:
        with db:
            db.execute("INSERT INTO events(task_id) VALUES (?)", (999,))
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("orphan event must fail")
finally:
    db.close()
```

删除被引用任务时要明确策略：拒绝删除、级联删除、保留事件快照或标记软删除，各自代表不同业务语义。不要为了让报错消失就盲目加 CASCADE，因为那可能把学习轨迹一起删除。数据库约束应表达你已经想清楚的规则，而不是用来逃避规则选择。

模式迁移是让已有数据库从旧结构变为新结构。首次建表与升级已有表是两个不同问题：CREATE TABLE IF NOT EXISTS 只会避免表已存在错误，不会自动给旧表补上新增列、索引或约束。需要记录 schema 版本，明确每一步迁移，保留备份，并在旧数据样例上测试升级。不能把删除数据库重建当作有价值用户数据的迁移方案。

初学项目可以先只支持版本一，遇到其他版本明确拒绝并给出说明。这比猜测旧字段含义继续写入更可靠。当你开始支持版本二时，准备一份版本一夹具，运行迁移后检查旧记录保留、新字段合理、重复执行行为明确。迁移的测试对象是已有状态，不是仅在空库上跑建表语句。

## 17. 小项目的完整核心实现与逐段解释

下面实现一个只管理学习状态的内存存储。它把状态修改与事件记录放在同一事务，使用参数化 SQL，并在入口验证课程标识和状态。这里不提供 HTTP，不连接模型，也不声称解决多进程并发；范围明确后，你可以逐条验证每项保证。

```python
import sqlite3

class ProgressStore:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("""
            CREATE TABLE progress(
                lesson TEXT PRIMARY KEY NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('reading', 'practicing', 'passed'))
            )
        """)
        self.db.execute("""
            CREATE TABLE events(
                event_id INTEGER PRIMARY KEY,
                lesson TEXT NOT NULL REFERENCES progress(lesson),
                state TEXT NOT NULL CHECK(state IN ('reading', 'practicing', 'passed'))
            )
        """)

    def set_state(self, lesson, state):
        if type(lesson) is not str or not lesson.strip():
            raise ValueError("non-empty lesson required")
        if state not in {"reading", "practicing", "passed"}:
            raise ValueError("invalid state")
        with self.db:
            self.db.execute(
                "INSERT INTO progress(lesson, state) VALUES (?, ?) "
                "ON CONFLICT(lesson) DO UPDATE SET state=excluded.state",
                (lesson, state),
            )
            self.db.execute(
                "INSERT INTO events(lesson, state) VALUES (?, ?)",
                (lesson, state),
            )

    def get_state(self, lesson):
        row = self.db.execute(
            "SELECT state FROM progress WHERE lesson=?", (lesson,)
        ).fetchone()
        return None if row is None else row[0]

    def history(self, lesson):
        return self.db.execute(
            "SELECT state FROM events WHERE lesson=? ORDER BY event_id", (lesson,)
        ).fetchall()

    def close(self):
        self.db.close()

store = ProgressStore()
try:
    assert store.get_state("F08") is None
    store.set_state("F08", "reading")
    store.set_state("F08", "practicing")
    assert store.get_state("F08") == "practicing"
    assert store.history("F08") == [("reading",), ("practicing",)]
    try:
        store.set_state("F08", "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid state must fail")
    assert store.get_state("F08") == "practicing"
    assert len(store.history("F08")) == 2
finally:
    store.close()
```

构造函数建立连接、启用外键并建表。状态集合同时在应用和数据库中检查，两个位置承担不同责任：应用能给出清楚错误，数据库在其他写入入口也维护约束。这里每个 set_state 都追加一条事件，包括重复设置相同状态；这是本例的明确语义。若你希望相同状态不产生重复事件，应修改合同并添加对应测试，不能仅凭“看起来重复”随手去重。

INSERT 后面的 ON CONFLICT 表示课程键已存在时更新状态；excluded.state 指本次尝试插入的新状态。它不是把整个表替换，也不是删掉旧记录重新插入。这个语法需要目标 SQLite 支持，课程环境应记录 `sqlite3.sqlite_version`；理解版本边界比直接复制任何网上找到的语法更可靠。

get_state 用 fetchone 判断是否存在，空结果返回 None，存在则取元组第一项。这里返回 None 不表示数据库失败：数据库错误会抛异常；None 专门表示合法的未找到状态。把不存在与失败分开后，上层才能决定初始化还是展示错误。history 返回按事件编号排序的记录，保证读取顺序可解释。

本实现仍有应被看见的局限。它允许从 passed 回到 reading，因为我们只限制状态集合，没有限制转移方向；它没有幂等请求标识，因此同一调用重试会再次追加事件；它只使用单个连接，不解决多线程共享。你应把这些作为下一阶段明确的新需求，而不是假设因为代码有类、事务、外键，就自动具备所有工程特性。

分阶段扩展可以先加入转移表，规定 reading 可到 practicing，practicing 可到 passed，并允许明确的重新学习操作。然后给写入加入请求标识，将重复请求与同标识不同内容区分。每加一项规则，先写反例，再改实现，最后验证旧行为未被意外改变。这样工程扩展建立在稳定合同之上，而不是无限增加分支直到样例能通过。

## 18. 文件与数据库恢复：如何证明你读到的是同一份输入

检查点保存已处理行数时，还需要绑定输入身份。假设处理到第一百行后暂停，原文件随后被替换，恢复程序若只跳过一百行，就可能跳过新文件的有效数据。文件路径相同不代表内容相同，修改时间也不能作为绝对可靠的内容身份。可以计算内容摘要，并把摘要与检查点一起保存，恢复前比较。

摘要只能说明按所选算法处理的字节是否匹配，不自动说明业务语义一致。改变换行或编码会改变字节摘要，即使显示文本看起来相同；这是字节级检查点通常需要的保守行为。若你设计规范化后的语义摘要，就必须明确规范化规则与版本，否则不同程序可能对同一内容得到不同结果。

处理过程中输入继续变化又是另一种情况。开始时算一次摘要，之后读取很久，期间文件被追加或替换，检查点可能绑定不上实际处理的版本。可靠策略可以要求输入不可变、先复制到受控快照，或使用明确支持追加的协议。初学项目最容易验证的合同是“输入文件在一次运行与恢复期间保持不变”，并在恢复时检查摘要。

备份数据库也要考虑一致性。对正在写入的数据库文件简单复制一个主文件，未必包含所需的全部当前状态，特别是存在日志或 WAL 文件时。应使用数据库提供的备份机制或在明确停止写入并关闭连接后按正确流程复制。不要因为文件管理器能复制出一个文件，就认为该副本必然是一致数据库快照。

正常恢复演练应包含完整闭环：创建数据，提交，关闭连接，重新打开，核对记录；再制造一个未提交失败，重新打开确认只保留已提交部分；最后用备份创建独立恢复副本并核对。把恢复副本放到新路径，避免测试恢复时覆盖唯一原件。能写入只是持久化的一半，能解释失败后的恢复结果才是另一半。

## 19. 最后一次手算：把四种状态放在不同栏

请把一次保存操作拆成 Python 内存、临时文件、最终路径和数据库四栏。修改内存字典只改变第一栏；json.dump 改变临时文件；os.replace 改变最终路径关联；数据库 commit 改变数据库已提交状态。它们之间没有自动统一事务。若你必须让多个栏表示同一个业务进度，就需要额外协议、恢复标记或把权威状态集中在一个地方。

例如先写报告文件，再更新数据库“报告已完成”，中间崩溃可能留下文件存在而数据库仍显示待处理。反向顺序则可能留下数据库说完成而文件尚不存在。一个简单可恢复方案是把文件路径、内容摘要和状态写入明确状态机，重启后检查实际文件并补齐或重新生成。具体方案取决于文件是否可重建，不能盲目选择一种顺序声称消灭所有失败窗口。

学习时用小数据把每个中断点画出来：临时文件创建前、序列化中、文件同步后、替换后、数据库提交前、提交后。分别写出重启时能观察到什么，以及哪些信息是权威来源。这个练习比背“原子性”“持久性”几个词更接近你以后开发 Agent 检查点、任务队列和模型评测结果存储时真正要做的工作。

### 补充手算：查询结果如何从五行原始记录得到

设事件表依次有任务一的 created、任务一的 retried、任务一的 done、任务二的 created、任务三的 created。若查询全部事件，结果有五行；若按 task_id 分组并计数，得到任务一三次、任务二一次、任务三一次；若筛选 kind 等于 done，则只剩任务一的一条记录。先明确筛选对象，再谈数量，否则很容易把“完成任务数”和“完成事件数”混为一谈。

假设错误重试使任务一出现两条 done 事件，直接 COUNT(*) 会得到二，但完成的不同任务仍只有一个。你可以在语义允许时统计不同 task_id，也可以从任务状态表查询当前 done 状态。两者甚至可能回答不同问题：事件表描述历史曾经完成过，状态表描述目前处于完成状态。若任务允许重新打开，这个差别会真实影响报表，而不是数据库小技巧。

再考虑 LEFT JOIN。任务表有三个任务，事件表只有任务一的事件。普通内连接只返回有匹配的任务一；左连接保留左表的三个任务，没有事件的列以 NULL 表示。此时对事件列计数与对全部连接行计数会不同。要统计零事件任务，必须知道空扩展行不是一条真实事件，不能把它误计为一次活动。

分页也需要稳定合同。使用 LIMIT 和 OFFSET 时，如果排序键不唯一，同分记录在不同查询中的相对位置没有保证；若查询之间数据改变，还可能出现重复或跳过。教学静态数据可先采用唯一 id 排序，明确分页期间输入不变。未来需要在线分页时，再引入基于游标和版本的策略，而不是假定 OFFSET 自动提供一致快照。

索引可以想成按某个键组织的额外查找结构，而不是保存所有查询答案的缓存。例如名字索引帮助按名字定位，却不一定帮助按状态过滤后再按时间排序。复合索引的列顺序也有意义，因为前导列决定结构的主要组织方式。初学时不要背每种查询该建什么索引；先用几百到几千条自己生成的数据比较计划，再解释为何结果行相同而访问方式不同。

观察性能时把连接创建、建表、插入样例和真正查询分开测量。否则你说“查询用了两秒”，实际可能绝大部分时间花在生成测试数据。重复测量应使用同一数据与查询，记录缓存是否已热，并避免把一次偶然较快的运行当作稳定结论。正确性测试需要确定结果；性能实验则需要记录环境与波动，两类证据不能互相代替。

### 补充练习：让失败的第二条语句证明共同回滚

前面的库存例子第一条就可能失败，容易让人误以为根本没有发生任何修改。更有辨识力的实验是先给乙增加，再让甲扣成负数：在事务内部观察到乙已经增加，异常离开后再次查询却恢复为零。这样你能亲眼区分“从未修改过”与“修改发生但尚未提交，随后被回滚”。

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE stock(name TEXT PRIMARY KEY, amount INTEGER CHECK(amount>=0))")
    with db:
        db.executemany("INSERT INTO stock VALUES (?, ?)", [("a", 3), ("b", 0)])
    try:
        with db:
            db.execute("UPDATE stock SET amount=amount+4 WHERE name='b'")
            assert db.execute("SELECT amount FROM stock WHERE name='b'").fetchone() == (4,)
            db.execute("UPDATE stock SET amount=amount-4 WHERE name='a'")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("negative stock must fail")
    assert db.execute("SELECT amount FROM stock WHERE name='b'").fetchone() == (0,)
finally:
    db.close()
```

内层断言检查当前连接在事务中的可见状态，外层断言检查回滚之后的状态。请不要再把内层查询结果当作“已经永久保存”。这个区分是理解数据库日志、恢复和并发可见性的入口。未来阅读框架里的 checkpoint save、commit 或 flush 等名字时，也要逐一确认它承诺的究竟是哪一层完成，而不是因为名字听起来像保存就认为全部落盘。

本章验收还应包含一个文字题：为什么 with 文件能关闭却不能撤销已经写入的文本，而 with 数据库连接在本例中能回滚一组写入却不关闭连接？完整答案必须回到不同对象实现的上下文协议。相同语法只规定调用进入和退出方法，不规定所有对象的退出行为一致；真正保证来自具体实现与合同。这一认识能避免你以后看到框架上下文时凭外形推测语义。

最后整理一份故障证据时，分别记下输入版本、数据库路径、连接配置、事务起止、异常类型和恢复查询结果。不要只保存终端最后一行错误，因为错误之前是否提交往往决定恢复策略。若使用临时内存库，应明确记录关闭后数据消失是设计行为，不是持久化失败；若使用文件库，应记录绝对路径，防止在另一个工作目录中创建了同名空库却误以为数据丢失。

数据保存代码的完成标准包括成功路径与失败路径。成功时能读取相同语义的数据；失败时不会把旧版本误标成新版本，不会把未完成操作伪装成完成，也不会清理不属于自己的文件。把这些要求写成可执行断言后，你就能在以后改变序列化格式、增加字段或替换存储实现时保留可验证的行为基础，而不必每次靠肉眼检查一次输出就宣布可靠。
