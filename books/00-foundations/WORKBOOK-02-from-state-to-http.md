# 工作簿 B：从一个字典到可重启的HTTP任务服务

本工作簿训练你从零设计一个有状态程序。先不用网络，只用字典；再建立对象边界；接入SQLite；设计路由；最后添加HTTP传输与手动请求。你会看见每一层为什么存在，而不是把完整服务当成一团神秘框架代码。

阶段参考为`step06_memory_tasks.py`、`step07_persistence.py`、`step08_routing.py`以及最终`task_service.py`。自动演示都不监听端口。只有明确运行`--serve`才启动回环HTTP服务。

## B0 把需求写成一张可以手算的状态表

最小任务包含id、title、done。创建一个任务时分配新ID，初始done=False；完成已有任务时改成True；列出任务时按ID顺序返回；未知ID必须失败。

先手算下面调用：

```text
add("Learn Python", key="one")
add("Learn Linux", key="two")
complete(1)
list()
```

最终应有两条：id1标题Learn Python且done=True；id2标题Learn Linux且done=False。任何实现，不管字典、SQLite还是远程数据库，都必须满足这个合同。

先不要思考端口、JSON头或Web框架。如果核心状态都没想明白，网络层只会让错误更难看见。Agent的任务规划器、工具执行记录和记忆管理也一样：先列状态与合法转换，再选择技术。

## B1 在空文件里写出第一条任务

在`.practice/workbook-b`新建`tasks.py`：

```python
tasks = {}
next_id = 1

task = {"id": next_id, "title": "Learn Python", "done": False}
tasks[next_id] = task
next_id += 1
print(tasks)
```

运行`python .practice/workbook-b/tasks.py`。你应该得到一个键为1的字典，值又是任务字典。外层键帮助按ID查找；内层字段描述任务。

`next_id += 1`在任务插入之后推进，避免同ID覆盖。若你忘记推进，第二次创建会覆盖第一条，程序可能不报错，因为字典允许给同键重新赋值。写测试时要检查数量和ID唯一性，不仅检查最后一个标题是否正确。

现在把完整脚本再运行一次，任务又从id1开始。原因不是文件没保存，而是内存状态只存在于这次进程；下次启动重新执行`tasks={}`。这是后面需要持久化的具体动机。

## B2 封装成对象，明确谁拥有状态

```python
class MemoryTasks:
    def __init__(self):
        self._tasks = {}
        self._next_id = 1

    def add(self, title):
        task = {"id": self._next_id, "title": title, "done": False}
        self._tasks[task["id"]] = task
        self._next_id += 1
        return dict(task)

    def complete(self, task_id):
        if task_id not in self._tasks:
            raise KeyError(task_id)
        self._tasks[task_id]["done"] = True
        return dict(self._tasks[task_id])

    def list(self):
        return [dict(self._tasks[key]) for key in sorted(self._tasks)]
```

`self._tasks`属于这个实例。创建两个MemoryTasks对象，状态应独立。下划线是“内部实现”的约定，不是不可绕过的安全机制。外部代码不应直接依赖它，以便将来把字典换成数据库。

为什么返回dict(task)副本？否则调用方拿到内部可变对象后，可以直接把done改成任意值，绕过complete合同。当前字段都是简单值，浅复制足够；若任务里有嵌套列表，浅复制仍可能共享内部对象，必须重新分析。

写一个故障实验：把return改成task，外部`created['done']=True`，再list观察内部被改。修复后加入测试，保证返回值修改不影响store。这个测试在阶段参考中已经提供。

## B3 添加输入校验，区分值域和状态错误

创建时title必须是非空字符串，strip后保存；key必须非空。完成时ID必须存在。两种失败性质不同：title非法是输入值错误；ID不存在是资源状态错误。最终HTTP层会把它们分别转成400与404。

不要在MemoryTasks中直接返回HTTP状态码。它是一个可被CLI、测试或Agent工具复用的业务对象，不应该必须知道HTTP。此处抛异常或返回显式结果类型都可，课程采用异常便于保持小接口。

初学者常写`if not title:`，这会拒绝空字符串，却可能接受非空列表或数字。你需要先检查类型，再检查strip后的长度。同样`if task_id:`只检查真值，不保证它是合法整数ID。业务合同不是“看起来不空”这么简单。

## B4 加入幂等键，复现一次“响应丢了”的场景

想象客户端调用add成功，但没收到结果。用户按重试，第二次调用不应再创建一条同意图任务。现在给每个意图一个key，维护`_keys: key -> task_id`。

新增规则：若key没出现，正常创建并保存映射；若key已出现且标题相同，返回原任务；若标题不同，抛冲突。返回值使用`(task, created)`，created区分新建与重放。

运行参考：

```bash
python -m labs.foundations.workshop.step06_memory_tasks
```

预期先返回id1、created=True；同key重试返回id1、created=False；complete后done=True；list只有一条。你应该能画出两张内部表如何变化：_tasks保存业务数据，_keys保存去重映射。

一个常见错误是每次重试都生成新key。这样服务端会把它们视为不同意图，幂等机制失效。key的生命周期应覆盖同一次业务操作的所有重试，而不是只覆盖一个HTTP连接。

另一个错误是只按title去重。两个用户可能真想创建两个同标题任务；同标题不一定同意图。幂等键避免把业务内容本身强行当身份。

## B5 独立测试状态机，不碰网络

你至少写四个测试：新建后数量为1；同key同title重试数量仍1；同key不同title失败且原数据不变；重复complete最终状态相同。

```python
store = MemoryTasks()
first, created = store.add("Learn", "one")
assert created is True
second, created = store.add("Learn", "one")
assert created is False
assert first["id"] == second["id"]
assert len(store.list()) == 1
```

使用assert适合这段学习验证；正式测试放进unittest。输入错误的业务校验仍用显式异常，不能只依赖可能被优化掉的assert。

测试还应检查失败后状态。只确认“抛了异常”不够：一个坏实现可能先创建新任务，再发现冲突抛错，数据库已经变了。操作的原子性与异常类型同样重要。

## B6 为什么换数据库而不改外部接口

现在重启后状态会丢失，需要持久化。理想情况下外部仍调用add、complete、list，内部存储从dict换成SQLite。接口隔离的价值此时出现：调用方不必知道数据位于内存还是文件。

先打开Python交互或写一个新脚本，只执行：

```python
import sqlite3
connection = sqlite3.connect(":memory:")
connection.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT, done INTEGER)")
connection.execute("INSERT INTO tasks VALUES (?, ?, ?)", (1, "Learn", 0))
connection.commit()
print(connection.execute("SELECT id,title,done FROM tasks ORDER BY id").fetchall())
connection.close()
```

预期`[(1,'Learn',0)]`。这里故意使用最小表帮助观察SQL；最终实现增加NOT NULL、CHECK、UNIQUE等约束。`:memory:`表示内存数据库，关闭连接就消失，适合单元测试。换成文件路径才会持久化。

`execute`返回cursor；fetchall读取结果行。SELECT本身没有自动把结果打印到终端。数据库对象与Python列表也不是同一个类型，你需要明确提取和转换。

## B7 事务边界为什么要包住一次业务操作

TaskStore.add先查询已有key，再决定返回原结果或插入新记录；新插入后查询生成的id并转成公开task。with connection把相关修改纳入一个事务上下文，成功提交，异常回滚。

这不意味着查询+插入在任何多进程并发下都天然安全。当前教学服务器串行处理，简化了竞争。若改多worker，要处理两个进程都查询“没有”后同时插入的冲突，不能只依赖先查一次。数据库UNIQUE约束是最终防线，应用应捕获特定冲突并读取已存在结果，而不是把它当随机500。

理解事务时始终问：**哪些动作在同一个数据库里？哪些动作在外部世界？** 写数据库与发邮件不共享一个SQLite事务。若邮件发出后数据库提交失败，回滚不会把邮件收回。这是后续Agent可靠执行必须面对的问题。

## B8 实际做一次关闭→重开→读取

```bash
python -m labs.foundations.workshop.step07_persistence
```

该阶段在临时目录创建数据库，第一连接创建并完成任务，关闭；第二连接打开同一个文件，读出done=True。预期：

```text
[{'id': 1, 'title': 'Persist this task', 'done': True}]
```

这证明跨连接持久化，而不是仅在同一个对象里读回内存。临时目录退出后自动删除，避免实验数据积累。你自己的长期学习记录则用明确持久路径，不要把TemporaryDirectory当长期存储。

现在故意把第二连接路径换成另一个文件，会看到空表；把它换成`:memory:`也会得到独立内存数据库。路径与连接生命周期是数据身份的一部分。

## B9 在不联网的情况下设计HTTP路由

我们定义一个普通函数：`dispatch(store, method, path, body)`。输入method/path是字符串，body是字节，输出(status,payload)。先模拟请求，不启动服务。

```bash
python -m labs.foundations.workshop.step08_routing
```

它打印六次请求：health200、创建201、同键重试200、完成200、列表200、未知路径404。每条输出同时包含方法、路径、状态和JSON结果，便于观察。

路由本质是把外部动作名称映射到业务操作。GET /tasks调用list；POST /tasks解析参数后调用add；POST /tasks/1/complete提取ID后调用complete。请求体的JSON解析属于适配层，数据库逻辑不必知道HTTP字节流。

这个分离对Agent很有价值。工具调用适配器可以把模型输出的name/arguments映射到同一业务操作；HTTP路由和Agent工具都是入口，不必各维护一套相互漂移的核心逻辑。

## B10 逐行追踪POST创建

取正文`b'{"title":"Learn","idempotency_key":"one"}'`，沿代码走：

1. 检查字节数不超过MAX_BODY。这里len是字节长度，符合传输上限定义。
2. URL分解，当前query不支持则明确拒绝；path用于路由。
3. method为POST且route为/tasks，进入创建分支。
4. body.decode('utf-8')把字节转字符串，非法编码抛错。
5. json.loads把字符串转Python对象，坏语法抛错。
6. 检查顶层dict与精确字段集合，防止缺字段或拼错字段。
7. store.add校验业务值、处理幂等、写数据库。
8. created为True选择201，否则选择200。
9. 返回公开task字典，外层HTTP handler再编码并发送。

这条路径有三套合同：传输格式、请求schema、业务状态。它们都可能失败，但错误发生在不同位置。将其全部塞进一个try/except Exception返回“failed”，会让调用者无法判断怎么修复。

## B11 HTTP handler为什么还要处理长度、类型和超时

纯dispatch已经测试正确，真实网络仍有额外问题。客户端声明body长100字节但只发50字节，服务器若无限等待会被挂住；客户端发十兆数据，解析前就可能耗尽资源；客户端把正文标成text/plain，接口可能要明确拒绝。

handler读取Content-Length并验证非负、没有重复；超过上限直接413；有正文时要求application/json；设置读取超时；读到的字节数不足则400。随后才调用dispatch。

返回时JSON先encode，再用字节长度填Content-Length。中文title的Python长度和UTF-8长度不同；如果填错，客户端可能等额外字节或把连接处理搞乱。F02的编码知识在这里真正影响网络正确性。

本服务关闭每次连接，简化持续连接状态。它不实现生产HTTP框架的所有能力，也不处理公网暴露所需的认证、TLS与连接防护。对初学者来说，知道哪些复杂性被明确省略，比把一个简化服务称作“生产级”更重要。

## B12 真实本地请求：把终端、进程、网络、数据库连接起来

终端A从仓库根运行：

```bash
mkdir -p .practice/workbook-b
python -m labs.foundations.task_service --serve \
  --database .practice/workbook-b/tasks.sqlite3 --port 8765
```

终端B运行：

```bash
curl --noproxy '*' --fail-with-body --max-time 10 \
  http://127.0.0.1:8765/health
```

如果连接拒绝，先看终端A是否仍运行和是否成功绑定端口；若端口被占用，选择另一空闲端口并同时改客户端，不要终止未知进程。若返回404，确认路径是/health而不是/；服务能响应404说明网络链路已经部分走通。

创建：

```bash
curl --noproxy '*' --fail-with-body --max-time 10 \
  -H 'Content-Type: application/json' \
  --data '{"title":"learn state machines","idempotency_key":"wb-b-1"}' \
  http://127.0.0.1:8765/tasks
```

重复同一请求，看ID是否保持；改title保留key，看是否409；GET/tasks看数量是否仍一条。最后Ctrl+C停止，重新启动同一数据库，确认状态仍存在。完成这组演练后，你才真正把静态代码、进程与持久数据连起来。

## B13 让失败成为一张可解释的表

| 你故意制造的问题 | 期望层次 | 预期表现 | 修复方向 |
|---|---|---|---|
| 服务没启动 | 连接 | 连接拒绝或无法连接 | 启动并确认监听 |
| 路径/missing | 路由 | 404 | 使用存在的路由 |
| body为`broken` | JSON语法 | 400 | 发送合法JSON |
| title为true | 业务类型 | 400 | 发送非空字符串 |
| 同key不同title | 幂等合同 | 409 | 同意图保持参数，新意图换key |
| 超过正文上限 | 传输资源 | 413 | 缩小请求或重新设计接口 |
| 完成不存在ID | 业务状态 | 404 | 先查询有效ID |
| 数据库目录不可写 | 存储 | 明确启动/写入错误 | 修复自己的路径和权限 |

不要把所有失败都通过重试解决。400重试相同输入仍错；409需要检查意图；连接暂时失败可以有限重试，但写入必须保留幂等键。重试策略是一部分业务逻辑，不是网络库的一个万能参数。

## B14 把项目搬进Docker之前，先问四个问题

程序写到哪里？容器删除后哪些数据需要保留？服务监听容器哪个接口？宿主发布哪个地址？四个问题没答清楚，不要把“docker run成功”当完成。

本卷提供的Docker演练刻意选择日志CLI而非任务HTTP服务，因为后者需要额外设计容器内监听与外部可达性。你的拓展可以为任务服务新增显式host参数，但默认仍回环；在容器用0.0.0.0时，宿主只发布127.0.0.1。要把这个取舍写进帮助和部署文档。

数据库路径应该挂载到专用持久目录或volume，不放在易丢失的可写层里。运行用户必须有写权限。只为解决权限报错就挂载整个宿主根目录、开启privileged或改成root，不符合这个项目的最小权限目标。

## B15 从课程项目到Agent工具：第一条完整衔接

定义一个工具create_task，参数schema要求title和idempotency_key。模型给出参数后，适配器先验证结构，再调用store.add，最后把结果转成工具响应。工具响应应该保留task_id和created，让模型知道是新建还是重放。

假模型测试流程：第一次返回创建调用；第二次由于模拟超时重复同样调用；系统最终数据库只有一条。这个测试不需要真实DeepSeek或GPT API，却能验证你最重要的副作用合同。真实API只用来检验协议适配和模型行为，不应替代基础状态机测试。

接着考虑任务列表是否属于长期记忆：它是结构化事实存储，查询精确且可更新；向量检索适合相似内容召回，但不应取代任务ID和事务状态。你已经从基础数据库学习自然走向Agent记忆设计，而不是先记一堆向量数据库产品名字。

## B16 毕业任务和评分依据

独立完成一个“学习计划服务”，在现有任务上增加课程ID与预计分钟数。要求：课程ID非空，分钟数严格正整数；创建幂等；按课程筛选；完成状态持久化；拒绝非法参数；CLI或HTTP至少一种入口；测试覆盖重试、冲突、重启与错误输入。

建议评分：合同清晰20分，核心状态正确25分，持久化与幂等20分，错误与测试20分，使用文档15分。任何真实密钥进入仓库、无界循环或错误输入悄悄成功，都应先修复再评估功能丰富度。

参考设计不是再引入大型框架，而是扩展TaskStore模式和dispatch校验，给每个新增行为写测试，再更新schema与说明。你要能解释每个新增字段为什么存在、由谁验证、存在哪里、怎样影响重试。

完成这一步，你拥有的能力不是“做过一个TODO App”的名词，而是能从需求推导状态、接口、存储与失败恢复，并把它们写成可以运行和审阅的代码。这会直接支撑后续复杂Agent、工作流、记忆系统和推理服务开发。
