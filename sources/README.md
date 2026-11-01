# 原始资料与版本证据

本工程把参考资料放在对应分卷附近，让“这个断言依据什么”可以沿章节追溯。每卷 README 或 sources 文件列出论文、官方文档、代码入口与已核对的快照。引用主分支网页表示阅读入口，不等于冻结了页面内容；源码研读任务优先使用明确提交。

| 分卷 | 资料入口与核查重点 |
|---|---|
| [F 基础](../books/00-foundations/README.md) | Python／Linux／Git／Docker 官方文档，区分操作系统与发行版 |
| [A Agent](../books/01-agent-engineering/README.md) | 模型协议、MCP／A2A、工作流与评测，明确协议日期 |
| [R 知识系统](../books/02-retrieval-memory-graphs/README.md) | 检索论文、向量索引、RAG／GraphRAG、记忆和真实框架 |
| [M 模型](../books/03-model-architectures/README.md) | 原始论文、官方 config／实现，区分结构与权重版本 |
| [S 框架](../books/04-inference-systems/README.md) | 固定提交的 vLLM／SGLang 路径与符号、评测与部署边界 |
| [K 算子](../books/05-kernels-hardware/README.md) | CUDA／Triton／CUTLASS／FlashAttention、硬件特性与算法假设 |

## 根目录环境资料

- [Microsoft WSL 安装](https://learn.microsoft.com/windows/wsl/install)：Win11 安装入口、环境初始化，核查 2026-11-01。
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)：Windows 驱动映射与 Toolkit 安装区别，核查 2026-11-01。
- [Python venv](https://docs.python.org/3/library/venv.html)：独立环境概念与平台路径。
- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)：实际设备安装时重新核对，课程不宣称一组 GPU 依赖在所有设备可用。

## 自动检查使用的固定 Actions 提交

2026-11-01 通过官方仓库 tag ref 核对：

| 项目 | 查询的 tag | 对应提交 |
|---|---|---|
| [actions/checkout](https://github.com/actions/checkout) | v4 | `11d5960a326750d5838078e36cf38b85af677262` |
| [actions/setup-python](https://github.com/actions/setup-python) | v5 | `a26af69be951a213d495a4c3e4e4022e16d87065` |

选择这些已解析的版本是为了固定自动检查依赖，不表示它们是所有用途的最新版本。CI 会联网安装 CPU 依赖；课程实验本身的离线性质与环境安装联网是不同阶段。CI 的实际执行状态以 GitHub Actions 结果为准，本地测试不能替代远端测试结果。

## 阅读一个新项目的证据表

遇到新论文或模型，写下：它试图解决什么瓶颈；数学上改变什么函数或结构；代码里哪个符号实现它；前提是什么；论文测了哪种硬件和负载；框架是否实际支持；你自己的复现能验证哪一个子结论。缺少其中一项时应明确保留未知，不用模型名称和宣传数字填空。
