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

本卷标准库离线实验与测试在课程构建环境运行，结果由仓库统一验证报告记录。在线模型调用未使用真实 key，也未产生 API 费用。Windows、WSL、用户 Linux 服务器及真实模型端点需要学习者按环境说明验收。MCP 方法层演示不是完整 MCP 兼容实现；完整 SDK、A2A 与 LangGraph 移植任务需要按锁定版本完成，并在自己的实验记录里标注是否实测。

本卷不声称所有最新模型都支持相同工具参数、不声称所有协议可无损互转、不声称路径守卫等价安全沙箱，也不把离线脚本模型成绩当真实模型能力。对未来提供方行为，以当前官方文档与契约测试为准。
