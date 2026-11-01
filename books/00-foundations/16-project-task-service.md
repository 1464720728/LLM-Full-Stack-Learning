# F16 项目B：HTTP＋JSON＋SQLite任务服务，理解一次真实请求的完整路径

> 先修F01–F15。默认离线演示：`python -m labs.foundations.task_service`。参考实现：`labs/foundations/task_service.py`。手动服务仅绑定127.0.0.1，适合本地学习，不是生产部署模板。

## 1. 为什么任务清单能教会Agent工程需要的基本能力

一个任务服务看起来简单，却包含输入校验、状态机、持久化、幂等、错误码、网络消息与资源生命周期。Agent调用外部工具同样需要这些基础。你能把“创建任务超时后再试一次”处理正确，再去做支付、发邮件、更新数据库等有副作用的Agent工具才更稳妥。

本服务提供GET /health、GET /tasks、POST /tasks、POST /tasks/{id}/complete。创建任务需要title和idempotency_key。相同key与相同规范化title重试，返回原任务；相同key但不同title返回409。完成任务重复调用仍保持done=True。

数据库只记录id、title、done、idempotency_key。没有用户登录、权限模型、审计、分页、多进程并发或生产级HTTP防护。这些是明确边界，不是隐藏成“下一版本再修”的bug；本项目负责把单用户本地服务的合同完整实现。后续Agent与系统卷会处理更大部署。

## 2. 先设计状态，而不是先写HTTP代码

每条任务只有todo与done两个状态。创建时todo；complete把todo变done；再次complete保持done。不存在的ID返回404。当前没有重新打开或删除接口，所以客户端不能随意把done变回false。

状态转换可以写成表：

| 当前情况 | 操作 | 返回 | 最终状态 |
|---|---|---|---|
| key不存在 | 创建合法任务 | 201，新任务 | todo |
| 同key同title | 重试创建 | 200，原任务 | 保持原状态 |
| 同key不同title | 创建冲突 | 409 | 不变 |
| 任务todo | complete | 200 | done |
| 任务done | complete | 200 | done |
| ID不存在 | complete | 404 | 不创建任务 |

注意同key重试创建时，如果原任务后来已完成，返回的应是当前done状态，而不是硬编码一个todo副本。数据库查询提供真实当前状态。

## 3. 数据库模式是另一道合同

id是整数主键，done带CHECK限制为0或1，idempotency_key为UNIQUE且NOT NULL。Python层校验提供友好错误，数据库约束防止内部bug绕过基本不变量。两层校验目的不同，不是重复劳动。

`TaskStore`构造时创建或打开数据库，设置Row对象使字段可按名字访问。每个修改在`with self.db`事务里完成。`as_task`只返回公开字段，不把idempotency_key暴露在每次列表里，也不返回数据库内部Row对象。

title先strip再保存。输入`" Learn "`与`"Learn"`视为同一规范化标题；idempotency_key限制字母、数字、下划线和连字符，长度1..80。合同越明确，重试比较越可靠。未来如果规范化规则改变，需要考虑旧数据兼容，而不是随意更改。

## 4. 幂等键究竟解决了什么

假设客户端发送创建任务，服务器插入成功，但响应在网络途中丢失。客户端看到超时，却不能判断任务是否存在。直接发第二次“创建同标题”可能产生重复，因为同标题未必代表同一次意图。

幂等键标识一次业务意图。客户端为这个意图生成稳定key，重试必须复用；新的意图才使用新key。服务端保存key与结果的对应关系。这样无论响应是否丢失，同意图重试都返回同一个ID。

为什么同key不同title要冲突，而不是返回旧任务假装成功？因为客户端可能错误复用了key，如果静默忽略新参数，用户以为创建了B，实际上拿到A。409让调用方发现合同违背。

本服务是串行HTTP服务器；单进程内没有两个handler同时执行add。若改成多线程或多进程，先查询再插入会有竞争，需要让UNIQUE约束成为最终仲裁，并在事务冲突后读取既有结果。不能只把HTTPServer换成ThreadingHTTPServer就宣称具备正确高并发幂等性。

## 5. 把网络层和业务路由分开

`dispatch(store,method,path,body)`接受普通参数，返回(status,payload)。它不打开socket，因此测试可以直接调用，快速覆盖业务合同。HTTP handler负责把真实请求头、字节正文转换成这些参数，再把返回值序列化发出。

POST /tasks先检查body大小，再decode UTF-8，再解析JSON，再检查字段集合，再调用store.add。顺序很重要：大请求应在解析前拒绝；错误编码不应变成服务器崩溃；顶层数组不应被当dict读取。

GET /tasks返回`{"tasks": [...]}`，列表按id排序，保证输出稳定。query参数当前明确不支持，出现时返回400；这比悄悄忽略客户端以为生效的过滤条件更容易排错。未知路径返回404。

## 6. HTTP handler承担哪些额外责任

真实传输层检查重复Content-Length、非法或负长度、超过8192字节的正文、非JSON正文类型、未支持的Transfer-Encoding和不完整正文。读取正文设置超时，避免本地客户端声明长度后一直不发完。

这些检查并不构成全面HTTP服务器安全保证。BaseHTTPRequestHandler及本项目没有生产级连接管理、认证、TLS、慢请求防护和多租户隔离；只绑定回环是教学范围的一部分。不要把它部署到公网，也不要把它作为公司正式任务管理服务。

返回时先把JSON编码为UTF-8字节，再计算Content-Length；发送Content-Type和Connection: close；写入完整body。连接关闭简化了持久连接与请求残留处理，代价是不能复用连接。这是为了可读性做的明确取舍。

## 7. 默认离线演示逐步追踪

运行：

```bash
python -m labs.foundations.task_service
```

demo创建内存数据库，发送同一个创建请求两次，再完成任务，再列出任务。状态序列应为201、200、200、200。第二次没有生成第二个ID；最终列表只有一个done=True的任务。内存数据库在关闭连接后消失，这适合测试，不适合持久学习记录。

逐行阅读demo时，你应能指出每层：json.dumps产生字符串，encode产生字节；dispatch解析请求；store执行SQL；返回普通dict；最外层再json.dumps输出供人检查。数据在不同表示间转换，每一次都有明确责任。

## 8. 手动启动本地服务并真实请求

从仓库根目录：

```bash
mkdir -p .practice/f16
python -m labs.foundations.task_service --serve \
  --database .practice/f16/tasks.sqlite3 --port 8765
```

该终端保持运行。另一个终端执行：

```bash
curl --noproxy '*' --fail-with-body --max-time 10 http://127.0.0.1:8765/health
curl --noproxy '*' --fail-with-body --max-time 10 \
  -H 'Content-Type: application/json' \
  --data '{"title":"finish foundations","idempotency_key":"f16-1"}' \
  http://127.0.0.1:8765/tasks
curl --noproxy '*' --fail-with-body --max-time 10 http://127.0.0.1:8765/tasks
```

先用ASCII标题，减少终端编码问题干扰HTTP理解；之后再测试中文。`--data`让curl发送POST。再次执行同样请求应返回同一任务。修改title但保留同key应得到409，curl返回非零正是预期。

完成任务的ID应使用实际响应里的id，不要假定数据库永远为空：

```bash
curl --noproxy '*' --fail-with-body --max-time 10 -X POST \
  http://127.0.0.1:8765/tasks/1/complete
```

若id不是1，替换对应值。Ctrl+C停止服务，数据库文件仍在；重新启动同一文件应看到之前状态。换一个数据库文件则是另一份数据，服务端口相同不意味着存储相同。

## 9. 测试应该覆盖什么

现有测试直接验证创建/重试/冲突、重复完成、缺失任务、非法JSON/编码/类型、正文过大、SQL元字符按普通数据保存、查询参数合同、重启持久化和demo状态序列。测试不启动socket，不访问外网。

所以验证报告要准确：业务路由与SQLite行为经过CPU自动测试；HTTP真实连接、curl和Docker演练需在自己的环境按本章执行。不能把自动测试“36个通过”改写成“所有操作系统、网络和部署模式都通过”。

你可以加入一个单独手动端到端脚本，启动子进程、等health成功、发送请求、最终清理进程。它要有固定超时、独占端口或动态端口、异常清理，并且只管理自己启动的PID。不要为了测试方便杀掉占端口的未知进程。

## 10. 八个逐级任务与参考方案

**任务A：只写内存状态机。** 先不用SQL，以dict实现创建、完成、列表。验收：完成两次最终相同，未知ID失败。参考：把状态转换表转为方法合同。

**任务B：加入SQLite。** 用参数化查询替换dict。验收：重建TaskStore后数据仍在，数据库约束拒绝重复key。参考：专用临时文件测试持久化，内存数据库测试独立逻辑。

**任务C：实现幂等。** 同key同title返回原ID，同key不同title冲突。验收：数据库记录数不增长。参考：比较规范化title，冲突时不修改既有数据。

**任务D：实现dispatch。** 使用普通参数调用，不启动服务。验收：合法/非法/未知路径有明确状态码，核心方法不依赖HTTP类。

**任务E：真实HTTP。** 写handler桥接dispatch。验收：UTF-8长度正确，Content-Type正确，大body受限，服务只监听回环。

**任务F：分页。** 增加经过校验的limit和after_id，替换当前拒绝query的策略。验收：多页合并无重复无遗漏，排序稳定，limit有上限。参考：WHERE id>? ORDER BY id LIMIT ?；不要用无界返回假装分页。

**任务G：乐观并发。** 加version列，更新要求expected_version。验收：旧版本更新返回冲突，不能覆盖新状态。参考：UPDATE ... WHERE id=? AND version=?，检查rowcount；这是理解Agent并发状态更新的重要练习。

**任务H：Agent工具适配。** 将add和complete注册成两个具有JSON参数schema的工具，先用固定假模型调用。验收：非法模型参数不穿过校验，重复工具调用使用同幂等键。参考：复用TaskStore与dispatch合同，不让模型直接拼SQL。

## 11. 基础卷毕业验收

独立实现日志工具与任务服务，不看参考也能解释入口、数据合同、核心算法、持久化和错误传播。随机给你一个失败输入，你能写复现测试、定位原因、修复并提交Git。给你一个新API或库，你能从官方文档找到最小可运行调用并明确未验证边界。

达到这些能力并不意味着已经成为Linux、Python或后端专家，但已经拥有继续深入Agent、模型与推理系统所需的实操底座。下一阶段应该把这些基础组件放进真正的Agent执行循环与模型运行链路，继续通过代码与测试深化。

## 12. 一次创建请求的完整表示变化

从客户端输入标题开始，数据至少经过五种表示：Python 字典、JSON 字符串、UTF-8 字节、HTTP 正文、数据库行。服务端再反向转换为公开字典、JSON 字节和客户端解析对象。每次转换都有明确合同，不能把字符串长度、字节长度、数据库字段与网络头部混为一谈。

例如标题“学习”只有两个 Unicode 字符，但 UTF-8 编码需要六个字节。完整 JSON 还包含字段名、引号和幂等键，因此 Content-Length 既不是标题字符数，也不是整个 Python 字典的 repr 长度。handler 先将 JSON 编码成字节，再取 len(data)，保证报文长度与实际发送字节一致。

```python
import json
payload = {"title": "学习", "idempotency_key": "trace-1"}
text = json.dumps(payload, ensure_ascii=False)
body = text.encode("utf-8")
assert len(body) > len(text)
assert json.loads(body.decode("utf-8")) == payload
```

网络层读取指定字节后交给 dispatch。dispatch 不需要知道 socket，也不负责读取一半正文；它接到的 body 应是完整字节序列。decode 处理编码，json.loads 处理语法，字段集合检查处理结构，store.add 处理标题与幂等规则。把这条路径分开后，某一步失败就能归到正确层，而不是统一显示服务器出错。

数据库行还包含 idempotency_key，但 as_task 只返回 id、title、done。这个转换是公开接口边界，不应直接把内部 Row 对象序列化出去。未来增加内部版本、审计或重试状态时，公开响应不必自动泄露所有字段。领域存储结构与外部 API 格式可以相关，但不应被迫完全一致。

## 13. 手算四次调用与数据库状态

第一次 POST 创建规范化标题 Learn、键 key-1，数据库没有旧记录，插入新行，返回 created=True，dispatch 映射为 201。第二次相同键与标题，查到旧行并比较一致，返回同一 id、created=False，映射为 200。第三次完成该任务，把 done 从零改成一。第四次再次创建相同意图，返回已有任务当前 done=True。

```python
import json
from labs.foundations.task_service import TaskStore, dispatch
store = TaskStore()
try:
    body = json.dumps({"title": " Learn ", "idempotency_key": "key-1"}).encode()
    first_status, first = dispatch(store, "POST", "/tasks", body)
    repeat_status, repeat = dispatch(store, "POST", "/tasks", body)
    assert (first_status, repeat_status) == (201, 200)
    assert first["id"] == repeat["id"]
    assert first["title"] == "Learn"
    complete_status, completed = dispatch(store, "POST", f'/tasks/{first["id"]}/complete')
    assert complete_status == 200 and completed["done"] is True
    _, retried_after_completion = dispatch(store, "POST", "/tasks", body)
    assert retried_after_completion["done"] is True
    assert len(store.list()) == 1
finally:
    store.close()
```

这里 f-string 使用响应中的实际 id，避免依赖数据库一定从一开始。测试数据库为空时 id 恰好为一，但真实持久文件可能已有数据，删除记录后自增编号也不保证回到一。标识应当作不透明关联值使用，而不是拿它推断当前记录总数。

返回码二零一表示本次新建，二零零表示重试拿到已有结果，两者都可以是成功。客户端不能只接受二零一然后把安全重试的二零零当失败。相反，四零九表示同一键绑定了不同规范化标题，不能把它当作暂时错误盲目重试；必须修正业务意图或键使用方式。

## 14. TaskStore.add 的事务为什么这样包围

标题与键先在数据库写入前验证。标题 strip 后长度一到二百，键限制为明确字符集合和长度。随后进入事务，查询键是否存在。存在时比较规范化标题，一致则返回原行，不一致则抛 Conflict；不存在时插入，再用 lastrowid 读取新行并转换为公开任务。

lastrowid 是当前插入语句产生的标识，不应通过“查询最大 id”替代。并发系统中最大 id 可能属于另一个插入，即使当前串行教学服务暂时看不出差异，也应使用正确接口建立习惯。读取刚插入行还确保返回值与数据库实际默认值一致，例如 done 默认零。

事务内 return 会正常离开连接上下文，按当前连接事务合同提交需要提交的修改；抛 Conflict 则异常离开，执行回滚路径。虽然查询旧行本身没有写入，统一放在事务范围便于解释“查已有或创建”的逻辑整体。若将来增加并发，仍需处理唯一约束竞争，当前先查再插入的串行假设必须保留。

数据库 UNIQUE 是最后约束，但异常语义要由应用解释。同键重复不一定是错误，它可能是网络重试；同键不同内容才是冲突。多进程实现若捕获所有 IntegrityError 一律返回旧任务，也可能掩盖其他约束失败。应区分具体原因并在一致性边界内读取现有记录，而不是把数据库异常全部翻译成成功。

## 15. 完成操作的幂等与状态机限制

complete 先严格检查正整数 id，拒绝 bool、零与负数，再执行 UPDATE done=1 WHERE id=?。如果没有匹配行，rowcount 为零，抛 KeyError；若任务已经 done，再设置一仍保持相同状态，返回成功。因此“重复执行后的状态不再变化”是本操作的幂等性，不要求第一次与第二次内部执行步骤完全相同。

幂等不等于没有副作用，也不等于可以随意重复所有接口。完成操作写同一目标状态，创建操作则通过键保护，删除操作若存在又需要定义重复删除返回什么。每个操作单独设计合同，不能因为某个 HTTP 方法名字通常关联幂等语义，就忽略应用实现是否真的满足。

当前状态机只有未完成和已完成，没有撤回、删除或版本检查。客户端不能在 POST /tasks 中传 done 来跳过创建规则，因为字段集合明确拒绝额外字段。如果要增加重新打开，需要新路由和允许转换表，同时考虑旧客户端、审计与并发更新。扩展应从需求和状态表开始，而不是让模型随意传任意字段写数据库。

## 16. 路由匹配、查询参数与状态码的精确范围

dispatch 首先检查 body 字节数，再用 urlsplit 分离路径与查询。当前出现非空查询参数就返回四零零，不会悄悄忽略过滤条件。GET /tasks 返回固定包装字典，POST /tasks 解析创建请求，完成路径用完整正则匹配正整数 id。完整匹配避免只匹配路径前缀后接受多余尾巴。

`/tasks/0/complete`、负数、非数字或额外路径片段不符合当前完成路由，返回 route not found。一个符合正整数形状但数据库没有的 id，则进入 store 后返回 task not found。二者状态码都为四零四，但原因不同。测试可以检查关键类别，用户错误信息不需要泄露完整内部实现。

HTTP 方法也参与路由。直接调用 dispatch 时，未知组合返回四零四；真实 BaseHTTPRequestHandler 只绑定 do_GET 和 do_POST，其他方法可能由标准库产生自己的未实现响应。因此不能把 dispatch 的所有方法行为直接当成真实传输层行为。这个边界提醒你集成测试为何必要。

完成请求要求空正文。空字节与 JSON 空对象不是同一输入：后者仍有两个字节，会被拒绝。客户端应按接口要求发送，不要因为某个通用 SDK 总给 POST 加 `{}` 就假定服务会接受。严格合同能发现这种适配差异，放宽则需要明确兼容选择与测试。

## 17. HTTP 消息长度、类型与读取超时

handler 先拒绝未支持的 Transfer-Encoding，避免把分块正文误当普通定长字节处理；再检查 Content-Length 是否重复、能否解析为非负整数、是否超过上限。长度超过八千一百九十二字节时，在读取和 JSON 解析之前返回四一三。小限制适合本地任务标题服务，不是所有 HTTP 应用的通用上限。

有正文时要求 Content-Type 为 application/json。没有正文的完成请求不需要凭空附加 JSON 内容。媒体类型描述正文解释方式，字符编码则由本服务明确采用 UTF-8。响应 Content-Type 包含 charset，客户端可以按字节解码再解析 JSON。请求头的大小写由 HTTP 规则和库处理，不能简单用普通字符串字典经验猜测所有行为。

读取正文设置五秒超时，防止客户端声明长度后一直不发完。这个超时在读取正文前设置，不覆盖所有请求行与头部阶段，也不是全面慢连接防护。串行 HTTPServer 一次处理一个请求，慢客户端可能影响其他本地请求。因此服务保持回环教学范围，不应扩张成公网生产入口。

若实际读到字节数少于声明长度，返回不完整正文错误。若客户端声明较小长度却发送更多内容，本服务只处理指定正文并关闭连接，简化残留字节与下一请求的关系。它没有实现完整高性能持久连接管理。理解这个取舍后，你就不会把 Connection: close 当成随意加的头部。

响应先 json.dumps，再 encode UTF-8，计算真实字节长度，发送状态、类型、长度和关闭标记，最后写正文。顺序与 HTTP 报文结构对应：头部结束后才是 body。应用写入 socket 仍可能遇到客户端提前断开，这属于传输失败，不能据此断言数据库动作没有发生。

## 18. 十二道练习与完整解答

### 一：合法创建与公开字段

```python
from labs.foundations.task_service import TaskStore
store = TaskStore()
try:
    task, created = store.add("Learn", "create-1")
    assert created is True
    assert set(task) == {"id", "title", "done"}
    assert task["done"] is False
finally:
    store.close()
```

幂等键属于内部关联，不出现在公开任务字典。字段集合断言防止未来直接泄露整行数据库内容。

### 二：规范化标题重试

```python
from labs.foundations.task_service import TaskStore
store = TaskStore()
try:
    first, created = store.add(" Learn ", "same-1")
    second, repeated = store.add("Learn", "same-1")
    assert created and not repeated
    assert first == second
    assert len(store.list()) == 1
finally:
    store.close()
```

比较发生在 strip 之后，因此两次表示同一标题。当前只去首尾空白，不改变中间空格、大小写或 Unicode 规范化。

### 三：同键不同标题不改状态

```python
from labs.foundations.task_service import Conflict, TaskStore
store = TaskStore()
try:
    original, _ = store.add("A", "key-1")
    try:
        store.add("B", "key-1")
    except Conflict:
        pass
    else:
        raise AssertionError("conflict expected")
    assert store.list() == [original]
finally:
    store.close()
```

测试不只确认异常，还确认原记录未被覆盖。这直接保护幂等键绑定业务意图的合同。

### 四：完成两次仍只有一条任务

```python
from labs.foundations.task_service import TaskStore
store = TaskStore()
try:
    task, _ = store.add("A", "done-1")
    first = store.complete(task["id"])
    second = store.complete(task["id"])
    assert first == second
    assert second["done"] is True
    assert len(store.list()) == 1
finally:
    store.close()
```

完成操作没有创建新行，重复调用也不把状态切回未完成。若实现使用布尔翻转而不是设置 True，这个测试会发现问题。

### 五：不存在任务不被悄悄创建

```python
from labs.foundations.task_service import TaskStore, dispatch
store = TaskStore()
try:
    status, payload = dispatch(store, "POST", "/tasks/999/complete")
    assert status == 404
    assert payload == {"error": "task not found"}
    assert store.list() == []
finally:
    store.close()
```

未知任务完成不是 upsert。本项目明确拒绝，不应为了“请求成功率”自动创建空标题任务。

### 六：JSON 与编码错误

```python
from labs.foundations.task_service import TaskStore, dispatch
store = TaskStore()
try:
    for body in [b"{", b"[]", b"\xff"]:
        status, _ = dispatch(store, "POST", "/tasks", body)
        assert status == 400
    assert store.list() == []
finally:
    store.close()
```

三种失败分别是语法、顶层结构和 UTF-8 编码。它们都不应产生数据库记录，但诊断层次不同。

### 七：正文上限在解析前执行

```python
from labs.foundations.task_service import MAX_BODY, TaskStore, dispatch
store = TaskStore()
try:
    status, payload = dispatch(store, "POST", "/tasks", b"x" * (MAX_BODY + 1))
    assert status == 413
    assert payload == {"error": "request body too large"}
finally:
    store.close()
```

正文甚至不是合法 JSON，仍先因大小被拒绝。验证顺序减少不必要解析工作，并提供一致错误类别。

### 八：查询参数当前明确拒绝

```python
from labs.foundations.task_service import TaskStore, dispatch
store = TaskStore()
try:
    assert dispatch(store, "GET", "/tasks?limit=2")[0] == 400
    assert dispatch(store, "GET", "/tasks")[0] == 200
finally:
    store.close()
```

未来实现分页时，这条合同会改变，因此需要更新版本说明与测试，而不是继续保留旧测试却在代码里静默忽略参数。

### 九：SQL 语法字符只是标题数据

```python
from labs.foundations.task_service import TaskStore
store = TaskStore()
try:
    title = "Robert'); DROP TABLE tasks;--"
    task, _ = store.add(title, "sql-1")
    assert task["title"] == title
    assert len(store.list()) == 1
    assert store.complete(task["id"])["done"] is True
finally:
    store.close()
```

参数化 SQL 使标题作为值保存，表仍存在。这个测试不代表服务已有用户授权控制，只验证数据与 SQL 结构分离。

### 十：文件数据库重开保留状态

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from labs.foundations.task_service import TaskStore
with TemporaryDirectory() as directory:
    database = Path(directory) / "tasks.sqlite3"
    first = TaskStore(database)
    try:
        task, _ = first.add("Persist", "persist-1")
        first.complete(task["id"])
    finally:
        first.close()
    second = TaskStore(database)
    try:
        rows = second.list()
        assert len(rows) == 1 and rows[0]["done"] is True
    finally:
        second.close()
```

这里验证正常提交与连接重建，不模拟断电。使用临时目录避免覆盖个人真实学习进度。

### 十一：完成请求不能携带空对象正文

```python
from labs.foundations.task_service import TaskStore, dispatch
store = TaskStore()
try:
    task, _ = store.add("A", "body-1")
    route = f'/tasks/{task["id"]}/complete'
    assert dispatch(store, "POST", route, b"{}")[0] == 400
    assert store.list()[0]["done"] is False
    assert dispatch(store, "POST", route)[0] == 200
finally:
    store.close()
```

失败后状态仍未完成，证明正文检查发生在状态更新之前。这个顺序保护了拒绝请求不产生业务副作用。

### 十二：默认 demo 状态序列

```python
from labs.foundations.task_service import demo
result = demo()
assert [status for status, _ in result] == [201, 200, 200, 200]
assert len(result[-1][1]["tasks"]) == 1
assert result[-1][1]["tasks"][0]["done"] is True
```

demo 不打开端口，全部通过普通函数调用和内存数据库完成。它适合快速理解状态机，但不能替代真实 HTTP 消息验收。

## 19. 四个故障现场与定位路径

**创建超时后出现两条任务。** 检查客户端重试是否换了幂等键。标题相同不等于同一意图，服务依赖稳定 key 去重。修复客户端保存逻辑操作键，并在未知结果时重用；再检查服务是否在并发模式下正确处理唯一约束竞争。

**重启后任务不见。** 先确认是否使用内存数据库，或相对数据库路径因 cwd 改变指向另一文件。记录绝对路径与启动命令，再查询文件大小和表内容。不要立即断言 SQLite 丢数据，也不要创建更多同名数据库使现场更混乱。

**curl 连接失败。** 先看服务进程是否仍运行，启动端口是否成功，客户端与服务是否在同一网络视角。Windows、WSL、容器和 SSH 远端各自的 localhost 不应混用。确认回环服务后再检查代理变量，使用明确 noproxy 的本地请求减少干扰。

**任务已创建但客户端收到错误。** 响应写入可能在数据库提交后失败，客户端断开或代理超时都可能造成结果不确定。用相同幂等键查询式重试，验证返回同一 id。不要把本地网络异常当成数据库必然回滚的证据，这正是本项目设计幂等机制的原因。

## 20. 分阶段独立实现：先让核心成立，再接真实网络

第一阶段用内存字典实现状态表，明确创建、重复创建、冲突和完成的行为。此时只写普通函数或类，不引入 HTTP。每个方法都通过固定小样例验证，特别检查拒绝操作后的状态。第二阶段换成 SQLite，保持同样公开结果，增加正常关闭后重开的持久化测试。这样能判断问题来自业务规则还是存储适配。

第三阶段实现 dispatch，输入为方法、路径和字节，输出为状态码与字典。把路由判断与数据转换逐项加入，不要让每个路由直接写原始 socket。第四阶段实现 handler，只负责消息读取、长度与类型检查、调用 dispatch 和发送响应。第五阶段才启动本地端口，用 curl 或标准库客户端验证真实传输。

第六阶段进行故障演练：坏编码、重复键不同标题、不存在 id、超大正文、重复完成和重启恢复。每次先预测数据库是否变化，再执行并查询最终状态。第七阶段给客户端增加同键重试流程，模拟“第一次结果已提交但响应未被使用”，确认第二次返回同一任务。这个练习比只写一个 retry 循环更能解释可靠性。

最后将实现、测试、运行说明和边界写入个人分支。说明默认演示不监听端口，--serve 才启动回环服务，数据库路径在哪里，如何停止，哪些能力尚未提供。别人应能从说明完成同样实验，而不需要知道你之前终端里运行过什么。

## 21. 分页扩展的完整数据层参考

当前 GET /tasks 返回全部任务，不适合无限增长的数据。一个清楚的扩展是按 id 游标分页：只取 id 大于 after_id 的记录，按 id 升序，最多 limit 条。limit 必须有上限，after_id 必须非负整数。这个扩展先作为独立函数验证，不悄悄改变现有拒绝 query 的路由合同。

```python
from labs.foundations.task_service import TaskStore

def page(store, after_id=0, limit=2):
    if type(after_id) is not int or after_id < 0:
        raise ValueError("non-negative cursor required")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    rows = store.db.execute(
        "SELECT * FROM tasks WHERE id>? ORDER BY id LIMIT ?",
        (after_id, limit),
    ).fetchall()
    tasks = [store.as_task(row) for row in rows]
    return {"tasks": tasks, "next_after_id": tasks[-1]["id"] if tasks else after_id}

store = TaskStore()
try:
    for index in range(5):
        store.add(f"task-{index}", f"page-{index}")
    cursor, collected = 0, []
    while True:
        result = page(store, cursor, 2)
        if not result["tasks"]:
            break
        collected.extend(result["tasks"])
        cursor = result["next_after_id"]
    assert [task["title"] for task in collected] == [f"task-{i}" for i in range(5)]
    assert len({task["id"] for task in collected}) == 5
finally:
    store.close()
```

这个方案在静态数据下能无重复无遗漏地遍历。数据在分页期间新增时，后续页可能看到新记录；如果需要严格快照，还要定义快照边界。游标分页不是自动冻结整个数据集，删除与过滤条件也会影响语义。接口应说明消费者是在浏览当前数据还是重放某个固定版本。

next_after_id 保留最后返回标识，空页时不凭空加一。消费者通过空 tasks 结束，不能假定每页一定满。若任务 id 中间有缺口，按大于游标查询仍然有效；直接用“第几页乘页大小”猜 id 则会错。这个例子训练把业务标识与位置序号分开。

## 22. 乐观并发扩展：版本是更新前提

假设两个客户端都读取任务版本一。甲先修改标题，版本变二；乙随后根据旧版本提交另一个修改。如果 UPDATE 只按 id，乙可能覆盖甲的新状态。乐观并发让客户端携带 expected_version，数据库只在 id 与版本都匹配时更新，并把版本加一。影响零行表示记录不存在或版本已变化，需要明确区分或返回冲突。

```python
import sqlite3

db = sqlite3.connect(":memory:")
try:
    db.execute("CREATE TABLE tasks(id INTEGER PRIMARY KEY, title TEXT NOT NULL, version INTEGER NOT NULL)")
    with db:
        db.execute("INSERT INTO tasks VALUES (1, 'original', 1)")
    def update_title(title, expected_version):
        with db:
            cursor = db.execute(
                "UPDATE tasks SET title=?,version=version+1 WHERE id=? AND version=?",
                (title, 1, expected_version),
            )
            if cursor.rowcount != 1:
                raise ValueError("stale version")
    update_title("first writer", 1)
    try:
        update_title("stale writer", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("stale update must fail")
    assert db.execute("SELECT title,version FROM tasks WHERE id=1").fetchone() == ("first writer", 2)
finally:
    db.close()
```

这里的原子条件检查由一条 UPDATE 表达，避免先在 Python 检查版本、等待后再无条件写入的竞争窗口。它与创建幂等键解决不同问题：幂等键识别同一次意图，版本检查防止基于过期状态覆盖新状态。真实系统可能同时需要二者，不能二选一后认为全部并发问题已解决。

## 23. Agent 工具适配：模型只提供参数，不直接获得数据库控制

将任务操作暴露给 Agent 时，模型输出应该是有限工具名与 JSON 参数。适配层验证工具名、字段集合、类型和权限，再调用已有 TaskStore 方法。不要让模型生成任意 SQL 或任意 Python 代码直接执行，因为那会绕过本项目已经建立的状态、参数化和错误合同。

先使用固定假模型返回创建与完成参数，验证工具适配，不需要真实 API。给同一逻辑操作分配稳定幂等键，重复模型调用或网络重试都复用该键。工具结果包含明确成功、冲突或输入错误，Agent 再决定是否向用户解释或停止。不能把所有工具错误包装成空字符串，让模型猜发生了什么。

如果将来增加用户账户，必须把任务归属和授权条件加入存储查询与工具上下文。模型知道某个 id，不代表有权操作它；提示词中写“只能操作自己的任务”也不能替代服务端验证。当前本地单用户演示没有这层能力，因此不要用它承载多用户真实数据。

工具输出也应保持最小必要字段。创建结果返回 id、title、done 足以继续流程，不需要返回数据库路径或内部幂等表。日志记录逻辑操作标识、尝试与错误类别，避免记录完整凭证。这里将基础卷的类型、异常、SQL、HTTP 与日志规则合到一起，形成 Agent 开发真正可复用的底层组件。

## 24. 真实传输验收与证据范围

完整端到端演练应启动自己的一份服务，等待 health 可访问，再发送中文创建、同键重试、冲突、完成、列表和非法正文请求，最后停止自己启动的进程。使用独立临时数据库和明确端口，失败时也执行清理。若端口已被占用，选择其他端口或检查自己的实例，不杀掉未知进程。

传输测试特别检查响应 Content-Length 等于 UTF-8 正文字节数、Content-Type 正确、连接按声明关闭。再测试重复 Content-Length、未支持 Transfer-Encoding、错误媒体类型与正文超限，确认这些检查发生在 handler，而不只是 dispatch。业务测试与传输测试分别定位不同缺陷，两者都通过才支持相应范围的结论。

即使本机回环测试通过，也不证明代理、TLS、多进程、容器网络或公网压力全部可用。记录真实执行环境与未验证边界，后续服务卷再扩展。课程的目标是让你能解释一次请求从参数到状态提交再到响应的每一步，并知道结果未知时如何恢复，而不是给一个小服务贴上未经验证的生产级标签。

### 补充：请求结束与业务提交不一定同时发生

把创建请求画成四个时间点：收到完整正文、验证通过、数据库提交、响应被客户端确认。前两个阶段失败通常尚未创建任务；提交后响应丢失则任务已经存在。客户端只能直接观察自己是否收到响应，不能由超时反推服务端停在哪一步。幂等键让它在不确定状态下重新询问同一意图，而不是创造第二个意图。

服务端日志也不应把“开始发送响应”写成“客户端已收到”。普通 HTTP 应用通常无法仅靠一次 write 返回证明用户已经看到结果。可靠协议应定义服务端完成点与客户端确认点，必要时提供查询接口。对于任务清单，重试创建和列表查询足以演示恢复；对于更复杂外部工具，还需要持久运行记录与状态机。

### 补充：完成基础卷后的独立审查题

请不用运行代码，先判断以下变化是否保持合同：把 title.strip 移到重复比较之后；把 UNIQUE 键去掉只保留预查；让完成接口切换 done 布尔值；把 JSON 字符数当作响应字节数；把所有异常都转成二零零。逐项给出一个反例，再运行个人副本验证。它们分别破坏规范化幂等、并发唯一性、重复完成、传输长度和错误语义。

最后把日志分析项目与任务服务都当作可调用工具，比较共同结构：外部输入先验证，核心逻辑返回明确结果，持久修改有提交边界，错误在入口转换，测试隔离外部依赖。你能够独立建立这套结构，才算把基础知识串成工程能力；后续深入 Agent、模型与算子时，继续沿用同样的手算、实现、反例和验证循环。
