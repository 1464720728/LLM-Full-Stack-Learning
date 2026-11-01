# 基础卷验证记录

核查日：2026-11-01。

- Python：3.12.14；NumPy：2.3.5。
- 31 个离线实验入口分别作为独立进程运行，退出码均为 0，每项有 10 秒上限。
- 49 项 unittest 通过，覆盖核心函数、两个项目、工作簿阶段与手写梯度。
- 仅使用 CPU；没有模型 API 调用、GPU 运行、外部数据下载或 Docker 启动。
- Markdown 围栏配对检查通过。

复验命令：

```bash
python -m unittest discover -s tests -p 'test_foundations*.py' -v
```

HTTP 自动测试直接调用业务路由 dispatch，不创建 socket。SQLite 测试使用内存库和临时目录，包含关闭重开后的持久化检查。并发测试检查上限、顺序与清理，不依赖具体耗时。数值测试用独立有限差分检查解析梯度。

Win11 原生、WSL 配置、SSH/tmux、Docker 构建和实际 HTTP/curl 传输演练未在此环境执行。文档提供手动步骤，不能把 CPU 自动测试扩张成这些环境已经验证的声明。

逐入口记录见 [validation.json](validation.json)。
