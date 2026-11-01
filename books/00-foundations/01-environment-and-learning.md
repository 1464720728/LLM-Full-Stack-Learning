# F01 从“复制命令”到能够解释自己的环境

> 起点：只需要能打开一个终端。本章所有自动实验使用 CPU，不请求模型、不调用 GPU。终点：你能准确说出一条命令在哪台机器、哪个操作系统、哪个目录、哪个 Python 环境里运行。

## 1. 一个终端窗口到底连接了什么

终端是输入输出界面，Shell 是解释命令的程序，操作系统负责进程、文件和设备。三者不是同一东西。Windows Terminal 可以开 PowerShell 标签，也可以开 Ubuntu/WSL 标签；它们长得相似，命令语法和路径规则却不同。SSH 连接后，窗口仍在你的电脑上，命令执行位置已经变成远端。若再进入 Docker 容器，命令面对的文件系统和进程视图又变了。

把每次操作前的四个问题写在纸上：**我在哪里运行？我操作哪个目录？我用哪个解释器？我的数据会写到哪里？** 这些问题在部署 vLLM 时比背住一个复杂启动命令更有用。假设你在 Windows 安装了 NumPy，随后在 WSL 运行 Python，出现 `ModuleNotFoundError`，并不矛盾：它们通常是两套独立解释器和包目录。再假设你在服务器宿主机安装了包，容器里仍可能没有。包不是“安装到这台物理机器上所有地方”，而是装进某个环境。

| 当前环境 | 路径例子 | 找解释器 | 常见用途 |
|---|---|---|---|
| Windows PowerShell | `C:\Users\you\course` | `Get-Command python` | 编辑、Windows 程序、启动 WSL |
| WSL Linux Shell | `/home/you/course` | `command -v python` | 本地 Linux 开发与练习 |
| SSH 远端 Linux | `/home/you/course` | `command -v python` | 服务器 CPU、Docker、GPU 作业 |
| Linux 容器 | `/app` | `command -v python` | 版本隔离的应用运行环境 |

`/home/you/course` 在两台机器上拼写相同，也不是同一目录。目录名不是全世界唯一标识，必须和机器、文件系统一起理解。

## 2. 为什么主线采用 Linux，Windows 怎样参与

本课程的 Bash、Docker、CUDA 和服务管理示例以 Linux 为主。你的 Windows PC 可以通过 WSL2 提供接近 Linux 的开发体验；服务器也能作为远端开发环境。基础卷暂时不需要 GPU，所以没有必要为了 `for` 循环、SQL 或 HTTP 开一张 H100。

先在 **PowerShell** 检查：

```powershell
wsl --status
wsl --list --verbose
```

没有安装 WSL 时，微软提供的安装入口是管理员 PowerShell 中的 `wsl --install`，之后按系统提示完成安装和重启。本教材不会让脚本替你改变系统设置。已有 Linux 发行版时先识别它，不必重装；发行版名称以 `wsl --list --verbose` 输出为准。Win11 的 GPU 驱动、WSL 的 Linux 用户空间、Python 中的 CUDA 库是不同组件，GPU 卷会再逐层验证。

把代码主要放在实际运行环境的原生文件系统中。例如 Linux 练习放在 WSL 的 `/home/...`，可减少 Windows 路径、换行与权限语义混杂。Windows 和 WSL 之间可以访问文件，但“能访问”并不等于权限、性能和文件监视行为完全一致。不要一开始同时维护三份名字相同、内容不同的仓库；先选一个主工作副本，明确其他副本如何同步。

## 3. 用一个小程序建立环境身份证

在仓库根目录运行：

```bash
python -m labs.foundations.environment
```

若系统只提供 `python3`，使用 `python3` 代替 `python`。后续统一以你确认的解释器名称为准。输出包含 `python`、`executable`、`platform`、`cwd`、`virtual_env` 和 `numpy_available`。路径随机器不同，不能和书上的例子逐字比较；你应检查它们是否符合你的意图。

打开 `labs/foundations/environment.py`，逐行看：

1. `import sys` 让程序能读取当前解释器的信息，不是启动一个新的 Python。
2. `sys.executable` 是正在运行的解释器路径；它比猜测 `python` 指向哪里可靠。
3. `Path.cwd()` 是进程当前工作目录。它和 `.py` 文件所在目录不是同一个概念。
4. `sys.prefix != sys.base_prefix` 用于观察标准 `venv` 是否生效；这不是对所有环境管理器的万能诊断。
5. `find_spec("numpy")` 检查模块是否可找到，不证明它的所有功能都正常。
6. 程序没有打印整张环境变量表，因为其中可能含 API 密钥；诊断应只读取需要的信息。

你会在后面的课程反复遇到同样思路：先观察，再构建最小复现，最后改变一项变量。不要先把驱动、Python、框架全部重装，再猜是哪一个修好了问题。

## 4. 解释器、包管理器和虚拟环境

`.py` 文件是文本；Python 解释器读取它并执行。`pip` 是安装 Python 包的工具；NumPy 是被安装的包。它们分别解决“怎么运行”“怎么获取依赖”“提供什么功能”。

标准虚拟环境可以这样建立：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -c 'import sys; print(sys.executable)'
python -m pip --version
```

`source` 在当前 Shell 中执行激活脚本，使当前 Shell 的 `PATH` 等变量发生改变。若把激活脚本放进一个新开的子 Shell，子 Shell 结束后，父 Shell 的环境不会被反向修改。因此，“运行一个脚本”和“source 一个脚本”有不同语义。

安装 NumPy 是一次需要联网、可能改变环境的明确操作：

```bash
python -m pip install numpy
```

使用 `python -m pip` 能把安装动作绑定到这个 `python`，避免一个 `pip` 装到环境 A、另一个 `python` 从环境 B 启动。已有 Conda 环境也可以用于课程；不必为了学习而同时叠加多种环境管理器。记录 `python --version` 与 `python -m pip freeze`，并理解 freeze 记录的是当前环境，不是自动保证所有平台都可复现。

激活环境没有给程序额外权限，不会自动加载模型，也不会产生 GPU 使用。退出虚拟环境可以执行 `deactivate`。删掉终端窗口只是关闭一个进程，不等于删除 `.venv` 目录。

## 5. 学习闭环：先预测，再运行，再修改

“我运行成功了”只证明某条路径走通。掌握一个实验至少有四层：能复现、能解释、能改动、能定位故障。每次练习保留一份短记录：输入、预测、实际输出、差异原因、修复后的测试。

例如你准备运行 `print(2 + 3 * 4)`。先预测输出 14，再运行确认。下一步把表达式改成 `(2 + 3) * 4`，解释为何得到 20。最后把一个变量改成字符串，观察报错并查明类型。这比连续复制十段“正确代码”训练得更扎实。

课程里的参考实现应在独立尝试后阅读。读完后先关掉文件，重新实现最小版本；否则容易把“看懂别人写的代码”误认为“自己能设计代码”。读源码也不是从第一行一直看到最后一行：先找入口、输入、输出和测试，再跟一次具体数据流。

## 6. 六个练习与答案

**F01-1：识别执行位置。** 在本地 Windows、WSL 或服务器任选两个环境，记录操作系统、工作目录、解释器路径。验收：解释为什么两个同名目录可能不共享文件。答案要点：机器和挂载命名空间不同；路径只在特定文件系统上下文中有意义。

**F01-2：不靠猜测找包。** 运行 `python -m pip --version` 和环境实验，检查路径是否在预期环境。验收：能指出 `pip` 的安装位置与 Python 解释器的关系。若不一致，先用正确解释器的 `-m pip`，不要默认 `sudo pip install`。

**F01-3：故意改变工作目录。** 从仓库父目录尝试 `python -m labs.foundations.environment`，解释可能出现的模块找不到问题。答案：模块搜索路径通常包含当前工作目录；目录改变会影响能否找到 `labs`。回到仓库根目录再执行，而不是随意复制源码到 Python 安装目录。

**F01-4：增加一个字段。** 让环境报告包含 `sys.version_info.major`。验收：输出仍为合法 JSON，原字段仍存在。参考做法是在字典中添加 `"python_major": sys.version_info.major`，不要用字符串拼接手工制造 JSON。

**F01-5：解释 GPU 不参与的原因。** 本实验为什么不需要 5070 Ti 或 H100？答案：它只是读取 Python、操作系统和目录信息，没有调用任何 GPU 计算库。拥有 GPU 不会自动把普通 Python 代码搬到 GPU。

**F01-6：设计故障记录。** 用不存在的模块名执行 `python -m missing_course_module`，记录准确错误和执行命令。验收：记录中没有密钥、用户名等不必要个人信息，并能恢复正确命令。不要只截“红色文字”；文本形式便于搜索。

## 掌握门槛

给你一个新的终端窗口，你能在两分钟内证明当前机器、Shell、目录与解释器；遇到依赖错误，你能先排查环境而非重新安装所有软件。满足这些条件，再继续文件与 Shell 课程。

原始资料：[Microsoft WSL 安装文档](https://learn.microsoft.com/en-us/windows/wsl/install)、[Python 教程](https://docs.python.org/3/tutorial/)。资料入口核查于 2026-11-01；示例不依赖文档网站显示的最新 Python 小版本。
