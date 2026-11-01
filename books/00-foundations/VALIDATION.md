# 基础卷验证记录

核查日：2026-11-01。

- Python：3.12.14；NumPy：2.3.5。
- 31 个离线实验入口分别作为独立进程运行，退出码均为 0，每项有 10 秒上限。
- 49 项 unittest 通过，覆盖核心函数、两个项目、工作簿阶段与手写梯度。
- 仅使用 CPU；没有模型 API 调用、GPU 运行、外部数据下载或 Docker 启动。
- 主章 222 个 Python 围栏块通过编译与独立执行/配套夹具；定义型片段的证据范围见深度审计。
- 本地临时 Git 仓库与真实回环 HTTP 另行烟雾验证通过，详细场景见 [DEPTH_AUDIT.md](DEPTH_AUDIT.md)。
- Markdown 围栏配对检查通过。

复验命令：

```bash
python -m unittest discover -s tests -p 'test_foundations*.py' -v
```

HTTP 自动测试直接调用业务路由 dispatch，不创建 socket。SQLite 测试使用内存库和临时目录，包含关闭重开后的持久化检查。并发测试检查上限、顺序与清理，不依赖具体耗时。数值测试用独立有限差分检查解析梯度。

Win11 原生、WSL 配置、SSH/tmux、Docker 构建和 curl 手动演练未执行。另行使用 Python HTTP 客户端与原始 socket 对自建回环子进程验证了消息长度/类型、业务状态、部分坏报文与重启持久化；未验证正文超时、慢请求头、代理/TLS、生产负载和多进程并发。不能把 CPU 自动测试或回环烟雾测试扩张成全部平台与网络环境保证。

逐入口记录见 [validation.json](validation.json)。
