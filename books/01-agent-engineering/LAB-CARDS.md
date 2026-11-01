# 27 张实操卡：每个实验都从运行走向修改与故障分析

这些卡不是新的抽象目录，而是已有可执行实验的逐题使用说明。所有命令在仓库根目录运行，不调用模型 API、不访问互联网、不使用 GPU。每张卡按三次提交完成：第一次记录预测和实际输出；第二次修改一个机制并加测试；第三次构造故障并证明没有误报成功。源码参考在 `labs/agents/experiments.py`，完整组件在同目录其他文件，故障回归在三个 `tests/test_agents_*.py`。

建议把个人作业放自己的学习分支中，命名为 `study/agents-Axx-主题`。实验输出可以保存到自己的临时目录，但不要把任何真实 API key、公司日志或私有文档提交到公开仓库。课程给的 fixture 全是人工构造，适合公开分享。

## L01：画出完整的一轮循环

命令：`python -m labs.agents.experiments 01-loop`。

先预测 model_calls、tool_calls、events 数组，再运行。核心期望为 final=42、model_calls=2、tool_calls=1。你要提交四条消息的具体内容，不只截图最终42。修改任务：把加法变成先做两次独立加法、再求总和。参考解法：第一轮两个不同 call_id，第二轮根据两个结果再调用 add，第三轮 final；应计3次模型、3次工具。故障任务：把第二次调用ID重复使用，验收为明确拒绝，不能在历史中出现同一 ID 对应两个不同含义。

## L02：输入类型不是靠眼睛看

命令：`python -m labs.agents.experiments 02-contract`。

结果包括 valid accepted、boolean rejected、extra rejected、missing rejected。解释为什么 true 在 Python 里容易被误当作整数。修改任务：给 add 增加只允许非负整数的约束，分别测试0、边界、负数。参考解法：schema minimum 改为0，执行函数不做自动类型转换。故障任务：把 `type(x) is int` 换成 isinstance，测试应抓住布尔值漏洞。最后恢复参考实现，保留回归测试作为作业证据。

## L03：保留提供方消息扩展

命令：`python -m labs.agents.experiments 03-chat`。

extension_preserved 应为 true。这里扩展内容是课程占位，不是真实隐藏推理。修改任务：增加一个 provider_specific 测试字段并验证同协议重放保留。参考解法：解析统一 Turn 的同时保存原始 assistant message；不要只重新构造 content 和 tool_calls。故障任务：删去 provider_message 路径，测试应失败。你还需要说明为什么原样保留不代表可以把字段转发给另一家模型。

## L04：Responses 不只是一段文本

命令：`python -m labs.agents.experiments 04-responses`。

continuation 中原 output 的 reasoning 项仍在，后面增加 function_call_output。修改任务：一轮含两个 function_call，要求恰好两个结果。参考解法：对 call_id 集合做完整性检查，然后保留原始 output 的顺序。故障任务：少一个结果或多一个不存在的 ID，必须报错。验收说明要区分“输出项 id”与“函数 call_id”，不能只说都是请求编号。

## L05：Messages 的内容块配对

命令：`python -m labs.agents.experiments 05-messages`。

结果是 assistant content 保留工具请求，随后 user content 放 tool_result。修改任务：增加 text 与两个 tool_use，验证文字和工具块顺序保留。故障任务：把结果改成 role=tool，这不是该协议的正确结构。参考解法：不要用 Chat 的消息结构想当然套用；以 Messages 自己的内容块契约构造。离线验收是结构测试，线上兼容性必须另外实测。

## L06：逐字节传输仍不乱码

命令：`python -m labs.agents.experiments 06-sse`。

输入被切成单字节，输出仍有完整“鲸”字。修改任务：加入 CRLF、多行 data 与注释行，输出应按事件组合。参考解法：增量UTF-8解码，按空行结束事件。故障任务：删除最后一个字节或空行，不能正常完成。请提交一个函数，遍历所有双块切分位置并比较相同输出；不用真实网络就能验证绝大多数分帧边界。

## L07：两个工具的碎片不能混拼

命令：`python -m labs.agents.experiments 07-deltas`。

返回两个工具调用，按 index 排列，参数分别是1+2与2+3。修改任务：随机交错三个调用，但每个调用内部片段顺序保持。参考解法：为每个index维护独立缓冲。故障任务：同一index先给id=a再给id=b，必须报冲突；相同ID出现在两个index也必须拒绝。进一步完成自定义seq重放，完整解法在 SequencedAssembler。

## L08：错误参数成为可理解观察

命令：`python -m labs.agents.experiments 08-errors`。

观察中 ok=false、code=invalid_arguments，模型可以在下一轮解释或纠正。修改任务：脚本模型读到错误后再发正确参数，最后成功。参考解法：不能删除第一轮错误历史，保留修复轨迹；新调用使用新ID。故障任务：未知工具名，执行器应拒绝而非通过 globals 找到同名Python对象。验收同时检查没有业务函数被调用。

## L09：执行前检查预算

命令：`python -m labs.agents.experiments 09-budget`。

模型已提出工具，但 tool_calls=0、status=limit。修改任务：把工具变成会向计数器append的副作用，确认预算为0时计数器仍为空。参考解法：预算检查必须在function之前。故障任务：把检查放到执行之后，测试应指出“最终计数器看似没超，真实效果已发生”。这项题目训练你观察实际效果，而不仅是状态字段。

## L10：可测试的重试策略

命令：`python -m labs.agents.experiments 10-retry`。

前两次瞬时失败，第三次成功，等待记录只有两项。修改任务：让第二次变成PermanentFailure，之后不应再调用。参考解法：只捕获TransientFailure，其他异常向上传播。故障任务：两层嵌套各三次重试，记录真实调用数9，再改成单层负责。等待用注入函数记录，不真的sleep几分钟；固定随机源便于复现退避值。

## L11：裁剪不能破坏工具原子组

命令：`python -m labs.agents.experiments 11-context`。

旧长内容被丢弃，system与最新对话保留。修改任务：插入一轮两个工具请求及结果，确保整组保留或整组丢弃。参考解法：先分组再按预算选，不先对消息做任意切片。故障任务：孤立tool或缺一个结果，发送前拒绝。进一步使用pinned_context，验证原始目标不因历史变长消失，且用户目标仍是user角色。

## L12：摘要必须保留证据身份

命令：`python -m labs.agents.experiments 12-provenance`。

只保留 verified=true 的事实，并保留 source 与 confidence。修改任务：两个来源对同一事实冲突，摘要必须保留双方，不自行投票。参考解法：把冲突作为结构化状态，待核查而非生成一个混合断言。故障任务：一个没有真实来源的claim即使文字很自信也不得标verified。验收通过独立证据表检查，不让模型自填verified直接生效。

## L13：路径访问边界

命令：`python -m labs.agents.experiments 13-paths`。

good=hello，escape=rejected。修改任务：在临时目录建立符号链接指向另一个临时目录，验证拒绝。参考解法：检查各级链接与最终解析位置。故障任务：绝对路径、`../`、不存在文件、二进制文件、超大文件。报告中明确这不是对抗并发恶意进程的OS沙箱，不能把通过路径测试写成“可安全执行任意仓库”。

## L14：修改之前确认你读的版本

命令：`python -m labs.agents.experiments 14-edit`。

返回新文本answer=42和新hash。修改任务：读取后人工改文件，再执行旧hash patch。参考解法：拒绝陈旧编辑，重新读后重新构造patch。故障任务：old片段出现两处，必须拒绝歧义；new超大小也要拒绝。验收不仅看异常，还要确认原文件仍是故障前的完整内容，没有部分写入。

## L15：业务幂等不是调用ID缓存

命令：`python -m labs.agents.experiments 15-idempotency`。

第一次created=true，第二次false，同一request_id只一条工单。修改任务：用不同模型call_id提交同一业务request_id。参考解法：业务表唯一键仍去重，ledger可以有两条调用记录。故障任务：同request_id换title，明确冲突且旧记录不变。请解释为什么每次重试生成新request_id会破坏幂等性。

## L16：在工具执行前重启

命令：`python -m labs.agents.experiments 16-checkpoint`。

saved_version=3，恢复后final=5、tool_calls=1。修改任务：把程序拆成start与resume两个命令，共享自己临时目录的数据库。参考解法：保存模型调用索引、pending和messages，恢复先执行pending。故障任务：只保存messages但丢pending，测试应发现工具未执行。再完成工作簿二中的模型inflight窗口，证明不会跳过未保存的响应。

## L17：拒绝陈旧状态覆盖

命令：`python -m labs.agents.experiments 17-conflict`。

stale_save=rejected。修改任务：同时拿两份State，第一份推进一步，第二份保存。参考解法：UPDATE带预期version条件，rowcount不为1即冲突。故障任务：去掉WHERE version，展示新messages被旧内容覆盖。作业结论必须说明CAS保护状态，不自动保护此前已经执行的外部副作用。

## L18：未知效果不能自动重放

命令：`python -m labs.agents.experiments 18-uncertain`。

status=uncertain，原因说明可能已完成需核查。修改任务：工具写入后抛异常，查目标业务表证明效果存在。参考解法：用request_id查询完成状态，确认后再补录；没有查询能力时保留未知，不猜。故障任务：强制把running改回未执行再重试，观察重复效果。不要把这种故障实验用于任何真实外部发送工具。

## L19：取消后不启动新请求

命令：`python -m labs.agents.experiments 19-cancellation`。

status=cancelled、model_calls=0。修改任务：在模型返回工具后取消，pending保留用于审计但工具不执行。参考解法：终态检查在所有动作之前。故障任务：假定取消等于回滚，构造已经完成一次写入再取消的例子，证明文件仍存在。报告要分别列“未开始”“已完成”“可能仍运行”的动作。

## L20：并发峰值而非耗时猜测

命令：`python -m labs.agents.experiments 20-fanout`。

values是平方列表，peak≤2。修改任务：limit为1和4，验证峰值约束。参考解法：semaphore包工具主体，finally回收资源。故障任务：一个worker失败，其余任务必须取消并await回收；不要只调用cancel后返回。进一步用resource_fanout实现同文件串行、不同文件并发，测试active资源集合不重复。

## L21：批准绑定具体动作

命令：`python -m labs.agents.experiments 21-approval`。

before=waiting、after=finished、authorized=true。修改任务：改为拒绝，后续动作不得发生。参考解法：批准记录包含digest和布尔决定，执行前匹配。故障任务：展示审批后修改动作内容，旧digest应失效。报告解释hash用于绑定，不代替用户阅读动作；测试程序模拟批准不代表生产系统自动同意。

## L22：多个意见不自动变成事实

命令：`python -m labs.agents.experiments 22-multi-agent`。

version的values有v1和v2，conflict=true。修改任务：加入三个重复旧来源与一个更新原始来源，仍保留争议。参考解法：按来源与适用范围核查，不能多数票。故障任务：缺失source的claim，schema或验证器拒绝。扩展到真实worker之前，先用可控数据证明合并逻辑不会制造来源。

## L23：统计分母要明确

命令：`python -m labs.agents.experiments 23-metrics`。

n=3、success_rate=1/3，均值按全部任务计算。修改任务：增加取消或超时记录，预先决定是否计入分母并在报告中写清。参考解法：主任务成功率通常保留所有事先分配任务，环境故障另做分层。故障任务：删除失败行后算成功率，解释为何这是选择偏差。进一步运行evaluation程序，手算pass@k与Wilson区间含义。

## L24：日志最小化与脱敏

命令：`python -m labs.agents.experiments 24-redaction`。

authorization与嵌套api_key被替换。修改任务：新增大小写不同字段和列表嵌套，保持过滤。参考解法：结构化字段名标准化，递归遍历dict/list。故障任务：秘密放在未知字段，说明现有过滤不能覆盖所有形式。最好的设计是原本就不记录授权头；正则是补充，不是让所有原始数据都能放心写日志。

## L25：审批、业务幂等与图的完整工单项目

命令：`python -m labs.agents.experiments 25-ticket-project`。

status=finished、rows=1、created_first=true、created_replay=false。修改任务：拒绝、陈旧审批、相同业务键不同内容、重启后恢复四种场景。参考解法分别是declined且0行、digest冲突、业务键冲突、加载waiting状态后重新review。验收表必须检查数据库真实行数，不能只看最终文字。最终把图保存到GraphSnapshots并验证两次进程对象重建。

## L26：修复、测试、独立验收

命令：`python -m labs.agents.experiments 26-coding-project`。

before_exit非零、after_exit=0、success=true，3次工具和4次模型调用。修改任务：先错误修复再根据失败测试修正；每次编辑使用新读取hash。故障任务：让模型最后宣称通过但实际上没有修复，独立verifier必须失败。进阶任务：将测试放在不可编辑位置，防止删测试伪造成功。真实模型接入前先完成脚本模型全路径测试。

## L27：工具方法层与完整MCP的界线

命令：`python -m labs.agents.experiments 27-tool-rpc`。

先列工具再调用add，结果resultType=complete、text=42。修改任务：增加一个纯查询工具，保持schema映射。故障任务：缺元数据、旧版本、未知方法、错误params类型都拒绝。参考解法见ToolRPC和request_message。最后通过官方SDK完成transport与生命周期验证，记录规范版本；不要把这个离线JSON-RPC函数称作完整MCP server。

## 项目组合挑战与参考路线

挑战一：将L11、L12、L15、L21组合成“带证据的工单助手”。检索得到证据，草拟工单，展示具体动作，批准后以业务request_id创建。必须测试权限变化、无证据、拒绝、重试和恢复。最小参考路线是先用确定性证据fixture，不接真实搜索；所有边界通过后接R卷检索工具，再最后接真实模型。

挑战二：将L06、L07、L09、L20组合成“受预算控制的流式多工具runner”。字节流先完成分帧，工具调用完整结束后验证；只读独立工具可并发，共享资源串行；全局预算先预留；结果按call_id回传。参考实现组件已有，组合时最大的难点是取消与部分结果：失败后要保存已完成观察，未完成项明确状态，不能丢掉它们再宣称全部完成。

挑战三：将L16、L18、L25与Outbox组合成“可恢复任务台”。要求进程在八个窗口任一处退出，恢复时给出正确状态。参考评分方式是故障矩阵，而不是一个happy-path录像。对于无法判断的外部效果，系统正确输出uncertain并提供查询路径，就属于通过；凭空猜成功或无脑重试反而失败。

完成这些组合挑战后，请向自己提出最后一个问题：如果明天换模型、换提供方、换工具后端，这些测试还有哪些能继续保护系统？能够跨模型保留的工程不变量，是你真正学会的能力。
