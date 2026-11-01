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
