# 官方资料与源码阅读任务

核查日期：2026-11-01。以下是本卷使用的原始资料入口，不是复制内容来源。课程解释、合成数据和实现围绕具体学习任务自行组织。文档网站可能展示比本次运行环境更高的小版本；兼容性以本卷实际测试记录和代码使用的接口为准，不以网页标题推断。

| 资料 | 用途 | 阅读任务 |
|---|---|---|
| [Python官方教程](https://docs.python.org/3/tutorial/) | 语法、容器、模块、类的参考 | 先完成F06–F07，再找默认参数与模块入口章节，对照自己的错误案例 |
| [Python 3.12 argparse](https://docs.python.org/3.12/library/argparse.html) | CLI解析器合同 | 找到type、default、store_true与parse_args，解释课程CLI每个选项如何生成 |
| [Python 3.12 unittest](https://docs.python.org/3.12/library/unittest.html) | 自动测试与fixture | 查assertRaises与setUp/tearDown，为任务服务新增一个失败状态测试 |
| [Python asyncio](https://docs.python.org/3/library/asyncio.html) | 异步执行与同步原语 | 找到协程/Task/Semaphore相关入口，说明它们不是同一个对象 |
| [Python sqlite3](https://docs.python.org/3/library/sqlite3.html) | 连接、参数化SQL、事务 | 查连接上下文管理器的事务行为，并验证退出with不等于自动close |
| [NumPy初学者指南](https://numpy.org/doc/stable/user/absolute_beginners.html) | ndarray、shape、轴与基础运算 | 复现F14矩阵例子，再修改shape并解释失败原因 |
| [Microsoft WSL安装](https://learn.microsoft.com/en-us/windows/wsl/install) | Windows/Linux环境搭建 | 检查已有发行版和版本，不盲目重复安装 |
| [Microsoft WSL文件系统](https://learn.microsoft.com/en-us/windows/wsl/filesystems) | 跨文件系统路径与存储位置 | 找出自己代码实际位于Windows盘还是WSL文件系统，记录路径语义 |
| [Pro Git官方在线书](https://git-scm.com/book/en/v2) | Git快照、分支与远端参考 | 对照一次真实暂存区差异，解释commit与push分别做什么 |
| [Docker overview](https://docs.docker.com/get-started/docker-overview/) | Docker对象和生命周期 | 把image/container/volume映射到本课程离线模型，并列出模型省略的机制 |
| [Docker build context](https://docs.docker.com/build/concepts/context/) | 构建上下文与dockerignore | 解释Dockerfile路径与context路径不同，并检查课程专用ignore规则 |

## 怎样从官方文档进入经典源码

Python文档中的“Source code”入口能直接定位CPython标准库实现。先选argparse或unittest中的一个窄功能，不要求通读整个解释器。记录所读分支或commit；若沿文档动态链接进入开发分支，明确它不是本地安装版本的保证。

建议第一个源码任务：从ArgumentParser.parse_args追到参数类型转换，构造非法batch-size，确认错误在哪里转成用户提示。交付一页调用链、一个最小实验和一条边界测试。第二个任务：从unittest测试发现入口理解test_命名与模块导入，解释为何包导入副作用会影响测试。

完成这些任务后再读Agent或推理框架的大仓库，会更容易区分业务组件与通用语言机制。不要用收藏更多仓库代替运行一条路径，也不要把某个项目的star数当实现正确性的证明。

## 资料边界

本卷没有声称某一Python、NumPy或Docker版本是“最新最佳”；示例选择稳定基础接口。WSL安装与Docker构建可能需要网络和系统权限，按官方当前文档与自己的机器条件执行。任何未公开的公司环境配置、账户地址与密钥都不属于课程源码。
