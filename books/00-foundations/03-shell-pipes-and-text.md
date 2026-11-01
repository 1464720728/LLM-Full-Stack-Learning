# F03 Shell、参数、管道与文本：读懂一整条命令

> 先修 F02。实验：`python -m labs.foundations.text_pipeline`。本章操作仅限 `.practice/f03`。

## 1. Shell 在程序启动之前已经做了很多事

你输入 `python train.py --epochs 3`，Shell 先按自己的语法解析，再寻找 `python` 程序，最后把参数序列交给它。Python 收到的是几个字符串，而不是终端里的所有字符原封不动。引号、变量展开、通配符、重定向往往由 Shell 先处理。

最常见的参数错误来自空格。`cat my notes.txt` 是两个文件参数；`cat "my notes.txt"` 才是一个参数。引号通常不会作为字符串的一部分交给程序，它告诉 Shell 如何分组。单引号常用于保留字面内容，双引号允许变量展开并保留一个参数边界。

```bash
lab_name='two words'
printf '<%s>\n' "$lab_name"
printf '<%s>\n' '$lab_name'
```

第一条输出 `<two words>`，第二条输出 `<$lab_name>`。把引号去掉后，Bash 的分词与路径名展开可能把一个意图中的参数变成多个。写脚本时，路径变量通常应使用双引号：`"$file"`。

Shell 的 `*` 是路径通配符，不是正则表达式。`rg 'error.*timeout' file` 中的 `.*` 才是传给 rg 的正则语法。不要因为它们都出现星号就认为作用相同。

## 2. 三条标准流与退出码

每个常见命令行进程有标准输入 stdin、标准输出 stdout 和标准错误 stderr，常用文件描述符编号分别为 0、1、2。把正常结果和诊断分开，可以让一个工具的结果安全地交给下一个工具。

| 写法 | 作用 |
|---|---|
| `command > out.txt` | stdout 写入文件，通常截断旧文件 |
| `command >> out.txt` | stdout 追加 |
| `command 2> err.txt` | stderr 写入另一个文件 |
| `command > out.txt 2>&1` | stderr 指向此时的 stdout，二者进入 out.txt |
| `left \| right` | left 的 stdout 作为 right 的 stdin |

重定向从左到右处理，因此 `2>&1 > out.txt` 与 `> out.txt 2>&1` 不同。前者先把 stderr 指向原来的 stdout（通常终端），再改变 stdout 的去向。这个细节解释了为什么某些错误“明明重定向了还在屏幕上”。

退出码是另一个通道：通常 0 表示成功，非 0 表示某种失败或特殊结果；具体含义由程序定义。`rg` 没有匹配通常返回 1，不一定表示工具坏了。`$?` 读取上一条命令的退出码，但再运行一条命令就会覆盖它。`a && b` 表示 a 成功才运行 b；`a || b` 表示 a 非零退出才运行 b。

## 3. 构造一条真正有意义的管道

先准备小日志：

```bash
mkdir -p .practice/f03
printf '%s\n' \
  'INFO request=a tokens=4' \
  'ERROR request=b timeout' \
  'INFO request=c tokens=7' \
  'ERROR request=d timeout' \
  'ERROR request=e invalid_json' > .practice/f03/run.log
rg '^ERROR ' .practice/f03/run.log
rg '^ERROR ' .practice/f03/run.log | awk '{print $3}' | sort | uniq -c
```

按阶段手算：第一段筛出 3 行；`awk` 从每行取第三个空白分隔字段，得到 timeout、timeout、invalid_json；`sort` 将相同字符串放在一起；`uniq -c` 统计相邻相同值。缺少排序时，`uniq` 不会帮你做全局分组。输出顺序可能受 locale 影响；严格比较文本时可以为该命令设置 `LC_ALL=C`。

Python 实验使用迭代器做同样的事：`select` 逐行选择，`Counter` 计数，`sorted` 固定输出顺序。预期结果是 `{'invalid_json': 1, 'timeout': 2}`。它不是为了替代 Shell，而是让你发现两者都在组合小型输入输出变换。

## 4. 搜文件和搜内容要分开

`rg --files` 用于列出文件，`rg 'pattern' path` 用于搜索内容。默认是否跳过隐藏或被忽略文件，要看工具选项；“没搜到”不能直接推出“文件不存在”。`find` 适合按类型、名字等文件元数据查询。例如：

```bash
find .practice/f03 -type f -name '*.log'
rg -n 'timeout' .practice/f03
```

`-name '*.log'` 的引号使星号交给 find 处理，而非先被 Shell 展开。若路径中包含换行，按行分隔的文件清单会产生歧义；高级脚本应使用 NUL 分隔，如 `find ... -print0` 配合支持 NUL 的工具。初学时先养成“不要用 `for f in $(ls ...)` 处理任意文件名”的习惯，后面再理解完整原因。

## 5. 为什么 JSON 不宜当普通列文本处理

日志里的第三列恰好是错误类型，适合简单字段拆分。JSON 的字符串可能包含空格、转义或嵌套结构，用 `awk '{print $3}'` 解析 JSON 不可靠。应使用理解格式的解析器，例如 Python 的 `json` 模块。

```bash
printf '%s\n' '{"message":"hello world","tokens":12}' | python -m json.tool
```

格式化只验证 JSON 语法，不保证 `tokens` 一定是非负整数，也不保证某个必需字段存在。F05 和 F15 会把“语法解析”和“业务校验”分开。

读取来自 API 的文本时尤其要区分三种失败：网络请求失败、HTTP 状态失败、JSON/业务结构失败。把 `curl` 的任何输出直接交给 JSON 解析器，可能只得到一个无关的 JSON 错误，遮住前面真正的网络故障。

## 6. 脚本不是命令的随意堆叠

一个可维护脚本应有输入、输出、错误路径和运行目录约定。`set -e` 不是自动正确性开关，它在条件、管道等上下文中有细节；`set -o pipefail` 会让管道更容易暴露前面阶段的失败。初学者应先明确每一步的退出码，再决定是否使用这些选项。

复杂业务逻辑通常放到 Python 更清晰：参数解析、数据校验、异常处理和单元测试更容易组织。Shell 保留给启动程序、连接文件和环境。把五十行 JSON 处理写成高度压缩的一行，不是更专业，往往只是更难定位错误。

## 7. 六个练习与答案

**F03-1 参数边界。** 用 `printf '<%s>\n'` 比较带空格变量有无引号。验收：能解释实参个数变化。答案：未加引号的变量展开可能经历分词；双引号保留单个参数。

**F03-2 计数题。** 把两条 timeout 日志之间插入 invalid_json，先不排序运行 `uniq -c`，再排序。答案：未排序可能得到两组 timeout；排序后总数才合并。

**F03-3 错误流题。** 运行 `python -m labs.foundations.process_runner`，说明返回字典为什么包含两个流。验收：你能写一个 Python 子程序把正常结果给 stdout、错误说明给 stderr。

**F03-4 精确匹配。** 找出以 ERROR 开头而非仅包含 ERROR 的行。答案：用 `^ERROR `；`^` 约束行首。构造 `INFO message=ERROR` 作为反例，检查它不会被匹配。

**F03-5 JSON 陷阱。** 对 `{"message":"a b","tokens":2}` 用空格切分，展示错误，再用 `json.loads` 取得 tokens。答案应说明“能处理这个样例”与“理解格式”是不同能力。

**F03-6 可组合报告器。** 改写 `text_pipeline.error_counts`，接受任意可迭代字符串，而不要求先转成列表。验收：传入生成器也得到相同结果，空输入返回空字典。现有实现就是参考答案：它只迭代一次，不依赖索引或长度。

## 掌握门槛

你能把一条管道拆成输入、变换和输出，知道每个引号由谁处理，能区分搜索文件与搜索内容，并能解释退出码与错误流。此时再读复杂启动脚本，陌生的只应是参数业务含义，而不是整行都像咒语。
