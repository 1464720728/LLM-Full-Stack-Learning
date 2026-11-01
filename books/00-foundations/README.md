# 基础卷：从零建立Linux、Python与工程实践能力

这一卷面向只会少量Linux命令、简单Git和一点Python的学习者。完成标准是能够自己写、测试、运行和调试一个小工程，能解释程序在哪台机器、哪个环境里执行，能把输入经过函数、文件、数据库与HTTP接口的路径追清楚。它为后续Agent、模型、推理框架与Kernel课程提供共同基础。

本卷包含16个主题章、5份长工作簿、31个可直接执行的CPU实验入口、两套完整项目，以及独立自动测试。项目是日志分析CLI和HTTP＋SQLite任务服务。工作簿提供从空文件开始的逐步构建过程，阶段源码允许你在自己的实现卡住时逐步比较。

## 怎样开始

从仓库根目录执行：

```bash
python -m labs.foundations.environment
python -m labs.foundations.workshop.step01_loop
python -m unittest discover -s tests -p 'test_foundations*.py' -v
```

若系统命令为python3，使用python3。本卷大多数实验只依赖标准库；张量、数值和小训练实验需要NumPy。不要看到依赖缺失就使用sudo安装到系统环境，先完成F01的解释器识别与虚拟环境练习。

自动实验默认不联网、不请求API、不打开端口、不占GPU。任务服务需要显式`--serve`才启动回环监听；Docker构建与curl演练是你主动执行的手动项目步骤。没有Docker或GPU也能先完成基础主线。

## 章节导航

| ID | 内容 | 主要交付能力 |
|---|---|---|
| [F01](01-environment-and-learning.md) | 环境与学习闭环 | 识别机器、Shell、目录、解释器与依赖 |
| [F02](02-files-paths-and-encoding.md) | 文件、路径与编码 | 处理相对路径、UTF-8、字节与文件生命周期 |
| [F03](03-shell-pipes-and-text.md) | Shell、管道与文本 | 解释参数、引号、标准流、退出码与文本处理 |
| [F04](04-processes-permissions-environment.md) | 进程、权限与环境 | 读权限位、理解父子进程、正确调用subprocess |
| [F05](05-network-ssh-http-json.md) | 网络、SSH、HTTP与JSON | 分层定位请求失败、校验JSON、使用远端会话 |
| [F06](06-python-values-control-functions.md) | Python数据与函数 | 独立写有边界合同的函数、解释可变对象与别名 |
| [F07](07-modules-objects-iterators.md) | 模块、对象与迭代器 | 追踪self方法、包导入、状态不变量与流式数据 |
| [F08](08-files-exceptions-sqlite.md) | 异常、文件与SQLite | 原子保存、事务、参数化SQL与失败恢复 |
| [F09](09-testing-debugging-and-quality.md) | 测试与调试 | 从失败样例到修复、验证关键性质与边界 |
| [F10](10-cli-logging-and-interfaces.md) | CLI、日志与接口 | 设计可组合命令行、结构化输出与配置优先级 |
| [F11](11-concurrency-async-and-cancellation.md) | 并发与asyncio | 有界并发、取消清理、背压与重试边界 |
| [F12](12-git-from-snapshots-to-contribution.md) | Git与源码修改 | 审阅暂存差异、分支修改、测试后提交 |
| [F13](13-docker-images-containers-volumes.md) | Docker | 解释镜像、容器、volume、构建与端口视角 |
| [F14](14-math-numpy-and-numerics.md) | 数学、张量与数值 | 手算线性层、shape/stride、稳定softmax与导数 |
| [F15](15-project-log-analysis-cli.md) | 项目A：日志分析工具 | 数据合同、错误处理、分位数、CLI与原子输出 |
| [F16](16-project-task-service.md) | 项目B：任务服务 | 状态机、幂等、SQLite、HTTP路由与持久化 |

## 五份工作簿如何穿插

| 工作簿 | 使用时机 | 学习方式 |
|---|---|---|
| [A：空文件到CLI](WORKBOOK-01-from-empty-file-to-cli.md) | F06开始，随F08–F10推进 | 每次只增加一种能力，自己先写，再比较阶段源码 |
| [B：状态到HTTP](WORKBOOK-02-from-state-to-http.md) | F08之后，完成F16时贯通 | 内存状态机→幂等→SQLite→路由→真实请求 |
| [C：Linux诊断训练场](WORKBOOK-03-linux-diagnostic-lab.md) | F01–F05之后，遇故障时复用 | 主动制造小故障，用证据定位执行环境 |
| [D：Python源码逐行阅读](WORKBOOK-04-reading-python-source.md) | F07–F11期间 | 沿入口、数据流、状态所有权、失败路径读源码 |
| [E：手算到训练](WORKBOOK-05-math-by-hand-and-training.md) | F14期间 | 同一小例子用纸笔、循环、NumPy、梯度检查实现 |

工作簿A的前五个阶段都有独立参考入口：`python -m labs.foundations.workshop.step01_loop`，依次到`step05_cli`；B使用`step06_memory_tasks`、`step07_persistence`、`step08_routing`。这些阶段不是最终项目的替代品，完成后还要加入F15/F16的完整合同和验收。

## 建议的学习顺序与关卡

第一关完成F01–F05和工作簿C，交付一份环境报告与三份故障记录。此时不要求写复杂算法，但必须能回答“命令在哪里执行、文件写到哪里、失败发生在哪层”。如果这些问题仍靠猜，先重复小实验，之后部署框架会省很多时间。

第二关完成F06–F10和工作簿A/D，独立重写分批函数与日志解析器，给空输入、非法类型、重复ID写测试。参考答案只在自己的尝试之后阅读；读完关闭文件，再从空白实现一次。能看懂与能写出不是同一验收指标。

第三关完成F11–F13，把程序放进可管理的进程与Git工作流，理解有界并发和Docker生命周期。至少做一次故意引入错误→测试失败→修复→审阅差异→提交。课程要求的是可解释的工程闭环，提交数量本身没有意义。

第四关完成F14和工作簿E，手算矩阵乘法、softmax、线性回归第一步梯度，并用代码验证。能给每个轴命名、写出shape变化、选择合理容差后，再进入模型与Kernel主线。

第五关完成F15/F16及工作簿B，独立实现两个项目。验收包括正确输入、错误输入、重试、重启、输出合同和复杂度说明。你可以在此基础上把任务服务变成后续Agent工具，把日志工具变成评测报告组件，基础练习会继续在大项目中发挥作用。

## 每道题怎样留下可检查的学习证据

建议为自己的练习创建学习分支，答案保存在个人练习目录。每题记录输入与预测，运行结果，错误原因，修复差异，以及能防止回归的测试。不要把真实API密钥、公司日志、个人数据库或模型权重提交到公开仓库。

一份合格证据可以很短：“batches在7项、size=3时应为3/3/1；原实现丢尾批；增加末尾检查；性质测试覆盖0..19项与多种size；测试通过。”它比“我看懂批处理了”更容易评估，也更容易在几周后恢复理解。

如果某项手动实验没有运行，明确写未运行。读完Dockerfile不等于构建成功；dispatch测试通过不等于所有网络传输层已测试；CPU数学正确不等于GPU性能达标。后续各卷也遵守同样证据边界。

## 硬件与资源安排

Windows PC可以用于编辑和标准库Python实验；Linux主线建议在WSL2或已有Linux环境执行。课程假设基础学习可在CPU完成，因此不需要长期占用服务器GPU。GPU卷会在确有学习价值时安排5070 Ti或H100，并明确设备、迭代次数和结束条件。

服务器CPU与Docker练习仍要限定自己的目录、容器和进程。共享机器上不运行全局清理、批量kill或未理解的sudo脚本。资源自由可用不等于可以忽视其他作业；可复现的小实验通常比长期挂服务更适合基础学习。

## 当前验证范围

标准Python环境中已运行本卷全部离线入口与49项unittest，测试包含路径越界、Unicode长度、参数字面性、别名、尾批、预算失败状态、原子保存、SQLite回滚、异步上限、JSON合同、日志错误、幂等、重启持久化和梯度检查。精确结果见[验证记录](VALIDATION.md)。

Win11原生、WSL2、Docker实际构建、SSH/tmux与真实HTTP传输演练未在本次容器环境全部执行；相关步骤提供明确合同与排错方法，学习者需按自己的环境完成。所有参考实现的简化边界已在对应章节说明。

进一步资料见[官方资料与源码阅读任务](SOURCES.md)。完成本卷后，优先沿Agent卷和推理框架/Kernel卷的实践任务继续；不需要把全部语言文档背完才允许前进，但遇到不理解的基础机制应回到对应工作簿补证据。
