# 一手资料与版本边界

核查日期：2026-11-01。下面链接均在课程建设时通过一手页面核查。链接可能跟随主分支或当前文档更新；课程不把未实际获取的 commit SHA 或未安装的 SDK 版本伪装成已锁定。阅读开源项目时按正文方法记录自己实际使用的 commit，并保留环境快照。教学 Harness 为本课程原创实现，统计推导和练习亦由课程独立构造。

| 来源 | 本卷用于核查的范围 |
|---|---|
| [OpenAI Function calling](https://developers.openai.com/api/docs/guides/function-calling) | Chat/Responses 函数工具结构与关联 |
| [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/) | 工具调用接口与严格 schema 的提供方边界 |
| [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) | 思考模式字段保留与参数行为不宜跨模型类推 |
| [Claude Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) | tool_use、tool_result 与内容块关联 |
| [LangGraph persistence 原始文档](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/persistence.mdx) | checkpointer 与 store 的职责区别 |
| [LangGraph interrupt API](https://reference.langchain.com/python/langgraph/types/interrupt) | 暂停后节点重跑与副作用幂等要求 |
| [MCP 2026-07-28 发布说明](https://blog.modelcontextprotocol.io/posts/2026-07-28/) | 无状态核心、版本变化、旧握手退出 |
| [MCP 2026-07-28 tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) | 元数据要求、工具方法和完成结果结构 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | 真实 SDK 接入与版本核查入口 |
| [A2A 官方项目](https://github.com/a2aproject/A2A) | Agent 应用间互操作的定位 |
| [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | 编码 Agent 源码阅读入口，核查时标注 v2 |
| [OpenAI Agents Python](https://github.com/openai/openai-agents-python) | runner、tools、handoffs/tracing 的真实框架阅读入口 |
| [Python asyncio](https://docs.python.org/3/library/asyncio-task.html) | 任务、await、gather 与取消语义参考 |
| [Python sqlite3](https://docs.python.org/3/library/sqlite3.html) | 连接、SQL 参数绑定与事务接口参考 |
| [Python subprocess](https://docs.python.org/3/library/subprocess.html) | 参数数组、超时、输出与子进程接口参考 |

## 已验证和未验证

本卷标准库离线实验与测试在课程构建环境运行，结果由仓库统一验证报告记录。在线模型调用未使用真实 key，也未产生 API 费用。Windows、WSL、用户 Linux 服务器及真实模型端点需要学习者按环境说明验收。MCP 方法层演示不是完整 MCP 兼容实现；A15 已在独立环境实际验证 MCP 2.3.0、LangGraph 1.2.19、OpenAI Agents SDK 0.24.0 的离线集成，完整依赖见 sdk-requirements-lock.txt，详细执行证据见 DEPTH_AUDIT.md。A2A 仅有概念核查与课程自定义任务状态练习，不宣称标准兼容服务；跨机传输、认证和真实模型 API 未执行。

本卷不声称所有最新模型都支持相同工具参数、不声称所有协议可无损互转、不声称路径守卫等价安全沙箱，也不把离线脚本模型成绩当真实模型能力。对未来提供方行为，以当前官方文档与契约测试为准。


## 第二轮深写新增核查

| 一手来源或本地发行源码 | 使用范围 |
|---|---|
| [WHATWG Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) | A05 对照标准与教学 SSE 子集，不把子集当完整 EventSource |
| [Python codecs](https://docs.python.org/3/library/codecs.html) | A05 增量 UTF-8 解码器状态与 final 参数 |
| [LangGraph 中断文档](https://docs.langchain.com/oss/python/langgraph/interrupts) | A15 检查点、thread_id、Command 恢复与节点重入 |
| [LangGraph types 源码](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) | 对照已安装 1.2.19 的 interrupt 索引与恢复值路径；未虚构 Git 提交 |
| [MCP SDK 测试文档](https://py.sdk.modelcontextprotocol.io/get-started/testing/) | A15 进程内 Client/Server 与资源退出 |
| [Agents SDK runner 文档](https://openai.github.io/openai-agents-python/running_agents/) | 运行入口、模型轮转、max_turns 与工具结果续接 |
| [Agents SDK 工具 API](https://openai.github.io/openai-agents-python/ref/tool/) | 工具装饰器与调用包装接口，另实际检查安装版本源码 |
| [A2A 规范](https://a2a-protocol.org/latest/specification/) | 任务、状态、消息、产物与终态边界；课程 TaskBook 明确非标准实现 |

核查日期仍为 2026-11-01。真实 SDK 包是本次安装的发行版本，不声称默认分支始终等于安装版本。课程框架例子、状态表与故障程序为原创；第三方源码仅保留帮助定位机制的短摘录，完整实现通过实际安装环境 inspect 阅读。
