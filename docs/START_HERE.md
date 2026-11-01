# 第一次学习：从打开终端到完成一个实验

你现在不必先理解 Transformer，也不必租 GPU。本页的目标很小：知道命令在哪里执行，得到一份课程文件，用一个明确的 Python 解释器运行实验，并能找到错误信息。

## 1. 分清你正在操作哪台机器

Windows 的 PowerShell、Windows 中的 WSL Linux、SSH 登录后的远端 Linux，是三个不同的命令执行环境。它们的当前目录、已安装软件、环境变量和进程不自动相同。你在 PowerShell 安装了 Python，不代表服务器也安装了；你在 WSL 中创建 `.venv`，不能把这个目录直接搬到 Windows 当成同一个虚拟环境使用。

判断位置的方法不是看终端窗口颜色，而是实际运行命令。PowerShell 中输入 `Get-Location` 看目录；Linux 中输入 `pwd`、`hostname`、`whoami` 看当前路径、机器和用户。不要把显示在提示符里的 `$` 或 `PS>` 当成命令的一部分复制进去。教材代码块中通常省略提示符。

优先选择一个环境完成第一阶段。如果暂时只想用 Windows，可以先走第 2 节；想同步练 Linux，就走第 3 节。两条路线都能完成大多数 CPU 课程。

## 2. Windows 原生 CPU 起步

从 Python 官方网站和 Git 官方网站安装适合你的版本：[Python](https://www.python.org/downloads/windows/)、[Git](https://git-scm.com/downloads/win)。课程核心语法目标为 Python 3.10 及以上；初版实际测试解释器版本见验证报告。安装完成后重新打开 PowerShell：

```powershell
py -3 --version
git --version
```

第一行应输出 Python 版本，第二行应输出 Git 版本。若提示“无法识别”，先确认软件确实安装并重新打开终端，不要到处复制未知安装脚本。`py -3` 是 Windows 的 Python 启动器选择 Python 3 的写法。如果系统只有 `python` 而没有 `py`，用 `python --version` 检查并使用该解释器。

选择一个普通用户可写的目录，例如自己的用户目录：

```powershell
cd $env:USERPROFILE
git clone https://github.com/1464720728/LLM-Full-Stack-Learning.git
cd LLM-Full-Stack-Learning
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe tools/course.py doctor
```

`git clone` 把仓库复制到本地。第二个 `cd` 进入复制出来的目录。`-m venv .venv` 让 Python 创建一个属于本工程的解释器环境。随后明确调用 `.venv` 里的 Python，因此不需要修改 PowerShell 执行策略来激活脚本。`pip install` 会联网获取 NumPy，是环境安装步骤，不是实验本身的网络请求。

接下来本页写的 `python ...`，在这条 Windows 路线中都可以替换成 `.\.venv\Scripts\python.exe ...`。例如：

```powershell
.\.venv\Scripts\python.exe -m labs.foundations.paths
.\.venv\Scripts\python.exe tools/course.py lesson F01
```

不要把 Linux 的 `source .venv/bin/activate` 直接粘到 PowerShell。它是 Bash 的命令，环境和脚本路径都不同。

## 3. Win11 中的 Linux 路线

WSL2 让你在 Windows 内运行 Linux 环境。先在 PowerShell 查询当前状态：

```powershell
wsl --status
wsl --list --verbose
```

如果没有安装，按照 [Microsoft 的安装文档](https://learn.microsoft.com/windows/wsl/install) 在管理员 PowerShell 中执行 `wsl --install`，按提示重启并完成 Linux 用户创建。已有 WSL 的机器不需要重复安装。安装会修改系统功能，应在你自己的 PC 上操作；本课程不会自动替你执行。

打开 Ubuntu 等 Linux 发行版后，在 Linux 终端执行：

```bash
pwd
whoami
python3 --version
git --version
```

Ubuntu 缺少课程基础工具时，可以安装基础包：

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```

这组包名和命令针对 Ubuntu/Debian。其他发行版使用自己的包管理器；公司服务器不要盲目复制系统安装命令，优先使用已有环境或请管理员按现有规范配置。

把 Linux 工作副本放在 Linux 自己的用户目录，先避免跨 Windows 挂载目录的权限和性能差异：

```bash
mkdir -p ~/learning
cd ~/learning
git clone https://github.com/1464720728/LLM-Full-Stack-Learning.git
cd LLM-Full-Stack-Learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python tools/course.py doctor
```

`source` 会让当前 Shell 读取激活脚本，把这个虚拟环境的可执行目录放到命令搜索路径前面。它不启动另一台机器，不把所有系统 Python 都替换掉，也不影响别人的 Shell。关闭终端后，下一次进入工程需要再次激活，或者始终使用 `.venv/bin/python` 的完整相对路径。

## 4. 用 CPU 完成第一组实验

先看课程地图和基础卷入口：

```bash
python tools/course.py map
python tools/course.py lesson F01
python -m labs.foundations.paths
python -m labs.foundations.python_values
python tools/course.py verify --track F
```

这里的 `-m labs.foundations.paths` 表示让 Python 把仓库中的模块作为程序运行。请从仓库根目录运行：这个目录里应当有 `README.md`、`books`、`labs` 和 `tools`。如果你进入 `labs/foundations` 再运行同一条 `-m` 命令，Python 可能无法定位顶层 `labs` 包。

测试输出会有每项测试的名字，最后给出运行数量和 `OK` 或失败信息。`OK` 说明这组代码在当前环境通过了这些检查，不说明你已经掌握对应课程。你还应能预测路径如何解析、变量为何得到某个值，并主动改一处输入验证自己的预测。

第一次实验记录可以这样写：

```text
我的预测：相对路径依据进程当前目录解释，而不是依据我脑中认为的项目目录。
执行：python -m labs.foundations.paths
观察：记录实际打印的目录与解析结果。
修改：换一个相对路径，先在纸上推导，再运行。
解释：指出 cwd、__file__ 和 Path.resolve 分别依据什么。
仍不懂：把尚不清楚的问题写在这里，而不是勾选“已完成”。
```

然后记录学习状态：

```bash
python tools/course.py record F01 --status practiced --evidence "完成环境识别与首个实验，能说明当前解释器位置"
python tools/course.py next
```

`practiced` 表示做过；`mastered` 表示你通过了章节末的验收。只有你自己明确记录 `mastered` 才会解锁依赖它的课程。忘记了一点内容不需要删除全部进度，可以把相应章节改回 `reading` 或 `practiced`。

## 5. 遇到错误时先收集什么

最有用的求助材料包括：在哪个终端、当前目录、完整执行命令、Python 版本、完整错误堆栈和期望结果。不要只截最后一行“失败了”；Python 堆栈中最后一行是异常类型，前面的路径与行号说明错误发生在哪里。

```bash
python --version
python -c "import sys; print(sys.executable)"
python tools/course.py doctor
```

`ModuleNotFoundError: numpy` 通常说明当前解释器里没有安装 NumPy，而不是电脑上完全没有 NumPy。用同一个 Python 执行 `python -m pip install -r requirements-cpu.txt`，避免 `pip` 命令来自另一个环境。`No such file` 首先检查当前目录和文件名。`Permission denied` 先区分文件访问权限、目录权限和脚本执行权限，不要立刻给整个目录 `777`。

暂时不解决 GPU 环境不会阻止你学习 Agent、检索、数学或系统模拟。等基础卷、模型卷和算子卷要求使用 GPU 时，再阅读 [环境分层](ENVIRONMENTS.md)。

## 6. 这一页的验收

不看答案回答：你现在运行的 Python 在哪里？当前目录在哪里？`python -m pip` 和单独的 `pip` 有什么差别？Windows 与 WSL 的 `.venv` 能否互换？测试通过为什么不等于会修改代码？

实际做一次：关闭终端，重新进入仓库，选择正确解释器，独立运行实验，故意拼错一个模块名并读出错误原因，然后恢复。能重复完成这个小闭环，就开始基础卷第一章。

本页环境资料核查：2026-11-01。Windows／WSL 安装步骤以链接中的官方当前说明为准；教学核心实验不依赖某个最新 WSL 功能。
