# 工作簿 C：Linux诊断训练场——从“命令报错”追到真正原因

本工作簿把Linux基础变成一组可重复的小故障。目的不是让你背更多命令，而是学会用证据缩小范围。所有写文件操作限制在`.practice/linux-lab`；所有进程只启动短时CPU程序；不需要root、API、Docker daemon或GPU即可完成大部分训练。

每站记录四项：你最初的判断、实际命令、观察结果、最后解释。不要把“换一个命令就好了”当答案。你需要解释原命令为什么不满足目标，以及新命令改变了哪一层。

## C1 目录站：我打开的文件为什么不是刚才那个

从仓库根创建两份同名文件：

```bash
mkdir -p .practice/linux-lab/a .practice/linux-lab/b
printf '%s\n' 'from A' > .practice/linux-lab/a/config.txt
printf '%s\n' 'from B' > .practice/linux-lab/b/config.txt
```

先读绝对明确的相对路径，再在子Shell里改变目录：

```bash
cat .practice/linux-lab/a/config.txt
(
  cd .practice/linux-lab/b
  pwd
  cat config.txt
)
pwd
```

括号创建子Shell，里面cd只影响子Shell，结束后回到原上下文。你会看到from A和from B。`config.txt`这个名字没有变，定位基准变了。

现在用Python观察：

```bash
python -c 'from pathlib import Path; print(Path("config.txt").resolve())'
```

这会打印基于当前目录的绝对路径，即使文件未必存在。resolve不是读取；路径对象不是文件内容。若程序将配置路径写死为`config.json`，从不同目录启动就可能读到不同文件。稳健程序应明确路径相对谁：当前工作目录、配置文件所在目录，还是包资源目录。

**进阶任务：** 设计一个程序接收`--config`路径，并在诊断中打印解析后的路径。验收：从不同目录启动但传同一绝对路径，读到同一内容。答案不是把当前目录强行cd到某固定位置，而是明确输入路径合同。

## C2 参数站：空格把一个路径切成两个

```bash
mkdir -p '.practice/linux-lab/with space'
printf '%s\n' 'one file' > '.practice/linux-lab/with space/note.txt'
cat '.practice/linux-lab/with space/note.txt'
```

再观察参数分组：

```bash
python -c 'import sys; print(sys.argv[1:])' '.practice/linux-lab/with space/note.txt'
```

结果列表只有一项。如果去掉引号，通常变成两项。Shell已经在Python启动前完成拆分，所以Python无法凭空知道你本来想把两个参数拼成一个路径。

单引号适合字面量，双引号适合含变量但要保持单参数的场景：

```bash
course_file='.practice/linux-lab/with space/note.txt'
python -c 'import sys; print(sys.argv[1:])' "$course_file"
```

如果你在Python里用subprocess启动外部程序，参数列表天然保留边界；不要为了“像终端那样写”把全部拼成一个字符串再shell=True。参数列表不是万能安全沙箱，但能避免这一类Shell解释问题。

**答案检查：** 带空格路径失败时，第一步不该重命名所有文件，也不该sudo。先打印argv，证明程序实际收到了什么。

## C3 编码站：Linux里的`^M`究竟从哪来

Windows文本常见CRLF换行，Unix常见LF。一个Shell脚本若带CRLF，解释器路径或参数末尾可能多出回车字符，产生看起来诡异的报错。不要把所有看不见的字符问题归结为“中文编码”。

用Python创建两份可检查的小文件：

```bash
python - <<'PY'
from pathlib import Path
root = Path('.practice/linux-lab')
(root / 'lf.txt').write_bytes(b'one\ntwo\n')
(root / 'crlf.txt').write_bytes(b'one\r\ntwo\r\n')
for name in ['lf.txt', 'crlf.txt']:
    data = (root / name).read_bytes()
    print(name, len(data), repr(data))
PY
```

预期LF为8字节，CRLF为10字节。`repr`把不可见字符显示成转义形式。文本模式读取可能进行换行转换，所以要诊断原始字节时使用read_bytes。

Python heredoc中的`<<'PY'`表示后续内容送给Python标准输入，直到独立一行PY结束；引号使Shell不展开块内的变量和反引号。这个机制不是Python语法，而是Shell在准备输入。

**进阶任务：** 写一个仅检查、不自动修改的脚本，报告课程目录哪些`.sh`含CRLF。验收：不扫描系统目录，不读取大模型二进制，输出文件名和CRLF计数。确认后再针对自己的文件转换，避免全盘批量替换。

## C4 标准流站：为什么管道接不到错误文字

```bash
python -c 'import sys; print("result"); print("warning", file=sys.stderr)' \
  > .practice/linux-lab/out.txt 2> .practice/linux-lab/err.txt
cat .practice/linux-lab/out.txt
cat .practice/linux-lab/err.txt
```

out里是result，err里是warning。再把stdout接给`wc -l`，你会发现warning仍出现在终端，因为普通管道只接stdout。Shell并不知道哪行“看起来像错误”；由程序选择写哪个文件描述符。

把两个流都写进同一文件：

```bash
python -c 'import sys; print("result"); print("warning", file=sys.stderr)' \
  > .practice/linux-lab/combined.txt 2>&1
```

输出顺序可能受缓冲影响，不应据此推断程序调用顺序完全相同。需要精确事件顺序时应记录明确序号、时间和刷新策略，而不是只看混合日志排列。

**定位题：** 一个JSON工具的stdout被INFO日志污染，该修改解析器容忍垃圾文本，还是修复流分离？优先修复工具合同，让结构化结果独占stdout，诊断走stderr。解析器随意截取某段花括号容易在真实内容中误判。

## C5 退出码站：屏幕有输出也可能失败

```bash
python -c 'print("partial output"); raise SystemExit(7)'
course_status=$?
printf 'exit=%s\n' "$course_status"
```

预期exit=7。保存`$?`必须紧跟目标命令；如果中间先运行echo，再读到的就是echo的退出码。正常输出存在并不代表完整成功，程序可能输出一部分后发现错误。

`&&`与`||`围绕退出码工作，不看输出内容。一个工具即使打印ERROR但退出0，Shell仍把它视为成功。因此你写CLI时，错误消息和退出码必须一致。

对于管道，默认常由最后一条命令决定状态，前面失败可能被掩盖。Bash的pipefail改变这种传播规则，但并不替你定义每个程序的语义。例如rg无匹配返回1可能是预期结果，不一定应该终止整项业务。正确做法是先理解各阶段合同，再选择控制流。

**练习：** 写一个程序先打印JSON再退出2，证明下游只看JSON合法无法判断整个操作成功。验收记录需要同时包含退出码和解析结果。

## C6 环境站：同一个python为什么找不到同一个包

先运行：

```bash
command -v python
python -c 'import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)'
python -m pip --version
```

比较输出，不要只看提示符是否有括号。Shell提示符可以被自定义，不能作为唯一证据。某个环境已激活但你显式运行另一个绝对路径解释器，也可能绕过预期环境。

创建标准venv会改变包搜索位置，但不会复制整套操作系统。安装CUDA驱动不是pip负责的事，安装PyTorch包也不自动更新宿主驱动。容器里可见的用户空间与宿主内核又不同，环境问题需要按层拆开。

Conda的activate、venv的activate、uv运行命令属于不同环境管理方式。先选一种适合当前工程的方式并记录，不要因为看到别人命令里有uv就叠加到已有环境。你需要知道“这次使用的解释器是谁”，而不只是工具名字是否热门。

**故障实验：** 在一个新venv里运行NumPy实验，若未安装NumPy应明确报依赖缺失；退出后回到原环境再验证。不要使用sudo pip把包装进系统位置来掩盖隔离。

## C7 模块站：直接运行文件和`-m`为何不同

课程cli模块有相对导入。正确入口：

```bash
python -m labs.foundations.cli
```

若直接运行`python labs/foundations/cli.py`出现相对导入错误，原因是包上下文不同。不要立刻把所有相对导入改成随便能通过的sys.path插入；那可能使测试和部署在另一个目录又失败。

使用Python观察模块位置：

```bash
python -c 'import labs.foundations.cli as m; print(m.__file__); print(m.__package__)'
```

`__file__`显示实际导入文件，`__package__`显示包名。若导入到了意外路径，检查当前目录、环境和同名模块覆盖。例如你把练习文件命名为json.py，可能遮住标准库json，引发难以理解的循环导入错误。

**练习：** 列出三种不宜给练习文件随意使用的标准库名字，如json.py、typing.py、asyncio.py。答案不是禁止这些名字永远使用，而是理解模块查找顺序与命名冲突。

## C8 进程站：终端关闭不等于程序一定停止

只对自己创建的短进程练习：

```bash
python -c 'import time; time.sleep(10)' &
course_pid=$!
printf 'created pid=%s\n' "$course_pid"
ps -p "$course_pid" -o pid,ppid,stat,etime,args
wait "$course_pid"
```

这会等待最多约10秒，CPU几乎不忙。`$!`是最近后台作业的PID，与你当前Shell的`$$`不同。`wait`等待这个子进程并取得它的退出状态。不要把一个随手复制的PID当作永久身份：进程结束后系统可能复用PID。

本实验可以自然结束，不需要kill。如果要练习终止，先检查PID仍属于自己创建的命令，再发送普通终止信号。不要根据“python”名字全局kill，因为公司服务器可能有许多他人Python作业。

tmux保留会话，nohup改变挂断处理，后台`&`改变Shell等待方式，系统服务管理器又负责不同生命周期。它们有交集却不等价。对于学习任务，你首先需要可定位日志、明确PID和明确结束条件，而不是尽可能把它藏到后台。

## C9 权限站：脚本能被Python读，不代表能直接执行

一个`.py`文件可读时，`python script.py`由解释器读取文件；直接`./script.py`还涉及执行权限和shebang。两条命令不是完全相同的启动路径。

不要为了运行普通Python脚本随意给所有文件加x。对Shell脚本也可以`bash script.sh`显式指定解释器，但它仍需要读取权限，且脚本中的命令需要各自权限。sudo放在最外层可能让整个脚本以高权限运行，比只给某个受控docker命令提权范围更大。

读权限实验先使用离线`permissions.py`手算640、750、umask结果。真实chmod练习只操作你自己专用目录内的小文件。观察前后`ls -l`，记录改变了哪一组权限。不要把系统服务目录、SSH目录或公司共享目录当练习对象。

**诊断任务：** 读一个文件失败，列出你会检查的三层。参考：实际用户身份；文件权限；所有父目录可穿过性。更复杂环境再检查ACL、挂载、容器UID映射等，而非第一步777。

## C10 HTTP站：有响应和有正确结果是两件事

启动F16自己的回环服务后，分别请求health、missing和非法POST。把每次结果按连接是否成功、HTTP状态、正文是否JSON、业务是否成功四列记录。

一个HTTP200正文里也可能包含业务错误；一个HTTP400正文可能是合法JSON错误对象。`curl -s`只影响进度显示，不表示请求成功；`-S`让silent模式下仍显示错误；`--fail-with-body`让HTTP错误更容易进入退出码判断。每个开关有具体作用，不要把一长串选项当固定咒语。

代理是另一层。环境中的HTTP_PROXY/HTTPS_PROXY可能让请求经过中间代理；访问本地服务时通常可用`--noproxy '*'`明确绕过。不要在公开日志里打印包含凭据的代理URL。若curl成功而Python失败，要比较两者的代理、证书、超时和目标URL，而不是只看模型名。

**进阶任务：** 写一个只访问127.0.0.1课程服务的健康检查，超时有限，区分连接错误与HTTP错误，并返回明确退出码。先用标准库实现，再比较成熟HTTP客户端接口，理解框架替你处理了哪些细节。

## C11 磁盘站：文件很大、目录很大、传输很大是不同问题

`ls -lh`看逻辑文件大小，`du -h`看目录或文件占用，`df -h`看文件系统容量。你下载模型后空间不足，先确认是哪一个挂载点满了，不要只看家目录总大小。日志、Docker层、缓存、数据集和权重可能分别位于不同位置。

本课程只用小文件做观察，不生成巨大填盘文件。你可以用`stat`观察大小与时间，用Python算输入文件的字节长度与hash。删除自己不再需要的产物之前，先确认它是可再生成的结果还是唯一数据源。

Git忽略的模型权重仍占磁盘，Docker镜像共享层可能让简单求和高估占用，已删除但仍打开的日志可能暂不释放空间。这些现象没有矛盾，因为工具统计对象不同。写报告时准确说“工作目录大小”“镜像占用”或“预计push数据”，不要统称一个大小。

## C12 最后一站：独立处理一个组合故障

情境：你在Windows终端连接Linux服务器，进入一个课程目录，运行任务服务报模块找不到；修复后服务启动但PC浏览器打不开；改绑定后又发现重启任务消失。不要一次修改所有设置，逐层完成以下记录。

第一层确认终端现在是PowerShell、WSL还是SSH远端Shell；第二层确认仓库根目录和解释器；第三层用`-m`启动包入口；第四层确认服务监听在远端回环，PC自己的localhost不是它；第五层选择SSH转发而不是随意公开服务；第六层明确数据库文件路径，避免每次创建内存库或换目录读新文件。

验收答案必须能解释每个症状对应哪层原因。最终产物是一份可复现的故障手册：最小命令、预期观察、实际结果、单步修复、回归检查。这样你未来遇到vLLM、SGLang或Agent框架启动错误时，就有方法把复杂系统拆成可验证的小问题。

## C13 十二条自测，逐条给出可执行证据

1. 当前目录不是源码目录，如何证明？用cwd和模块__file__分别观察。
2. 文件名有空格，如何证明参数被拆开？打印sys.argv。
3. 日志里有乱码，如何区分编码与换行？先读bytes的repr，再选择解码。
4. 管道看不到错误，如何解释？检查stderr是否进入管道。
5. 命令打印正常内容却失败，如何证明？保存并报告退出码。
6. 包已经安装却导入失败，如何解释？比较解释器与pip路径。
7. 相对导入失败，如何修复？确认包入口与工作目录，而非乱改sys.path。
8. 后台进程是否结束，如何确认？记录自己的PID并检查状态，避免按名字批量终止。
9. 文件可读却不能直接运行，为什么？读取权限与执行路径不同。
10. HTTP404是否等于网络断开？不是，它证明某个HTTP服务响应了请求。
11. 重启服务数据消失，先查什么？数据库位置与持久化模式，而不是先查浏览器缓存。
12. GPU为什么不参与这整本工作簿？所有任务都是OS、CPU、文件和网络机制，没有GPU计算调用。

完成这些证据后，你的Linux能力已经从“认识命令名”转为“能解释程序运行环境并定位故障”。后续真正困难的系统问题仍需要更多实践，但你已经有一套不会随框架版本更换而失效的排查方法。
