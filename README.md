# LLM 全栈学习工程

**从只会 `cd`、`ls` 和一点 Python，走到能够实现 Agent、阅读推理框架、手写模型与 GPU 算子。**

这是六卷中文教材、可执行实验、参考实现、故障练习、综合项目和源码研读任务组成的课程工程。第一版以 Agent 开发和推理框架／算子为两条深入主线，基础知识贯穿全程。学习目标不是记住工具名字，而是能够解释数据如何流动，写出实现，用测试证明自己的判断，并在真实项目中找到对应机制。

**深写版保留 113 个稳定学习单元与 209 个注册实验入口，将 98 个主章和两个跨卷项目按完整小课展开。** 你会在同一章中经历概念建立、逐步手算、代码追踪、边界实验、完整练习答案和项目验收。另有 18 道可独立评分的空白实现题与专题工作簿。逐章变化见 [深写对照](docs/DEPTH_REPORT.md)，实际审读与修正见 [教学审读记录](docs/EDITORIAL_REVIEW.md)；统计不把代码量和命令数量代替讲解深度。

## 从这里开始

1. 阅读 [第一次学习：环境与第一条命令](docs/START_HERE.md)。还没有 Python、Git 或 Linux 环境也可以从这里开始。
2. 阅读 [课程地图与先修关系](docs/LEARNING_PATHS.md)，选择当前阶段，不需要一次装齐全部 GPU 依赖。
3. 按 [每课的学习方法与验收](docs/HOW_TO_STUDY.md) 完成一课。**先预测结果，再运行，再修改，再解释失败。**
4. 查阅 [完整课程索引](docs/CATALOG.md)、[实验索引](docs/LABS.md)、[18 道空白实现与自动评分](practice/README.md) 和 [综合项目](projects/README.md)。

克隆后，在仓库根目录运行：

```bash
python tools/course.py map
python tools/course.py next
python tools/course.py lesson F01
```

长章节内部按小课分次学习，不要求一天读完一章；暂停时记录具体小节和未解决的输入。每次先跟完一个小例子，再进入下一机制。

上述课程导航仅需 Python 标准库。数值实验另外需要 `requirements-cpu.txt` 中的 NumPy。`python` 找不到时，Linux 通常使用 `python3`，Windows 可使用 `py -3`；具体步骤见入门页。

## 六卷教材

| 卷 | 入口 | 从什么开始 | 最终要能做什么 |
|---|---|---|---|
| F 计算机与编程基础 | [基础卷](books/00-foundations/README.md) | 终端、路径、文件、变量 | 独立开发 CLI 与小型服务，使用 Linux、Git、Docker、测试和调试 |
| A Agent 工程 | [Agent 卷](books/01-agent-engineering/README.md) | 一次模型调用和一次工具调用 | 实现可恢复、可观测、可评测的 Agent，理解协议、状态图和多 Agent |
| R 检索、记忆与图 | [知识系统卷](books/02-retrieval-memory-graphs/README.md) | 文档、词项、相似度和证据 | 实现混合检索、RAG、长期记忆、知识图谱与权限隔离 |
| M 模型结构与手写实现 | [模型卷](books/03-model-architectures/README.md) | 向量、概率、注意力 | 实现 Transformer、Dense、MoE、稀疏、线性注意力和状态模型并验证 |
| S 推理框架与系统 | [框架卷](books/04-inference-systems/README.md) | 请求、批次、Token、KV | 实现小型调度系统，沿 vLLM／SGLang 源码追踪并修改核心机制 |
| K 算子、编译器与 GPU | [算子卷](books/05-kernels-hardware/README.md) | 地址、索引、线程和数据块 | 写 CUDA／Triton 算子，验证数值，解释布局、编译和硬件性能 |

## 每个知识点怎样落到实践

课程把实践分成五层：手算一个最小例子；读懂一个短实现；补完或修改一个机制；制造并定位一个错误；把多个机制组装成项目。每卷提供参考实现与答案，但验收要求包括换一组输入、解释不变量和识别失败边界，不能只复制运行输出。

教学模型与生产框架有明确边界：例如 CPU 调度模拟器用来验证状态转换和缓存所有权，不代表实测 vLLM 性能；哈希向量用来理解检索接口，不代表语义 embedding；NumPy 注意力等价性实验不代表已经训练出有用的大模型。进入真实源码后，再逐项补上工程约束。

## 硬件与花费

| 环境 | 主要用途 |
|---|---|
| Win11 普通 Python 或 WSL2，CPU | 基础、Agent、检索、图、数学、小模型与系统模拟的大部分练习 |
| RTX 5070 Ti，推荐通过兼容的 WSL2／Linux 软件栈做 GPU 练习 | 小型模型、Triton／CUDA 入门、数值验证、单卡性能实验 |
| Linux 服务器，CPU／Docker | Linux 操作、容器、服务、网络、分布式逻辑与源码调试 |
| 共享 H100，默认单卡、限定运行规模 | Hopper 特性、真实推理框架、算子 profiling；确认资源空闲后再做多卡实验 |

GPU 和 API 实验显式启动。离线课程导航和 CPU 测试不会自动下载模型、调用付费 API 或占用 GPU。参见 [环境与资源使用](docs/ENVIRONMENTS.md)。

## 工程工具

```bash
python tools/course.py doctor             # 查看解释器、依赖和可选工具
python tools/course.py check              # 校验课程索引、路径、依赖
python tools/course.py verify --track F   # 运行一个分卷的离线测试
python tools/course.py search "KV cache" # 检索教材正文
python tools/course.py record F01 --status practiced --evidence "能解释相对路径并完成练习"
python tools/course.py next               # 根据本地学习记录给出可学章节
python -m practice.grade --module practice.starter --task F-P1 # 做题入口；尚未实现会返回失败反馈
```

本地学习记录保存在被 Git 忽略的 `.course/` 中；课程内容与个人进度分开管理。测试和硬件验证的实际结果见 [验证报告](docs/VALIDATION.md)，内容范围见 [覆盖矩阵](docs/COVERAGE.md)。

## 版本与持续扩展

课程版本见 [VERSION](VERSION)，变更规则见 [贡献与版本规范](CONTRIBUTING.md)。每章有稳定 ID，源码导读尽可能锁定提交 SHA。教材、实验和测试一起更新，不能只改一句概念却保留相反的代码行为。

版本完成范围和维护入口在 [建设状态](docs/BUILD_STATUS.md)，内容维护遵循 [深写标准](docs/DEPTH_STANDARD.md)。如果继续扩展某一方面，先记录新的学习成果和验收，再增加内容；现有章节与练习的稳定 ID 应继续有效。
