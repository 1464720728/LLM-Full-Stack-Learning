# 从空白实现开始的 18 道可评分练习

各章还有大量手算、阅读、修改、排错与项目任务。本目录额外提供 **空白接口 → 自动反馈 → 完整参考实现** 的闭环，避免只会运行现成代码。所有题目 CPU／离线可做，需要 NumPy。

## 开始方法

先只做基础题，不必一次做完。复制模板到自己的文件：

```bash
cp practice/starter.py practice/my_work.py
python -m practice.grade --module practice.my_work --task F-P1
```

PowerShell 使用 `Copy-Item practice/starter.py practice/my_work.py`。然后用编辑器打开 `practice/my_work.py`，只实现 `parse_jsonl`，保存，再运行评分。未完成题目会提示 `NotImplementedError`，这是明确设计的练习状态，不是课程实验损坏。

可以在同一命令传多个 `--task`。省略时检查全部 18 题。查看标准答案前，先写自己的输入、预期和错误分析。全部参考实现位于 [solutions.py](solutions.py)，评分实现位于 [grade.py](grade.py)。

```bash
python -m practice.grade --module practice.my_work --task M-P1 --task M-P2
python -m practice.grade --module practice.solutions
```

评分不是安全沙箱，会导入你选择的本地 Python 模块。不要用这个入口运行来源不明的代码。它也不是穷尽证明：通过现有测试后，还要自己加边界样例并解释不变量。

## F-P1：带错误报告的 JSON Lines 解析器

输入是字符串迭代序列，每个非空行应是一条 JSON object。返回 `(records, error_lines)`：records 按原顺序保存合法对象；error_lines 保存从 1 开始的错误行号。空白行忽略，合法 JSON 数组／数字也视为错误记录类型。原输入不修改。

例子：`['{"x":1}', 'bad', '  ', '[1]', '{"x":2}']` 返回 `([{'x':1},{'x':2}], [2,4])`。先回答为什么第四行是合法 JSON 却仍要拒绝，再实现。提示是 `enumerate(lines,1)`、`strip()`、`json.loads` 和捕获具体错误。不要用 `eval` 解析数据。

参考思路：逐行处理，把“语法是否合法”和“是否符合业务对象契约”分开；解析成功后检查 `type(value) is dict`，否则也进入错误记录。一个坏行不终止整个文件。进阶修改：增加严格模式，让首个坏行抛异常，并给 CLI 不同退出码。

## F-P2：合并计数但不修改输入

输入两个从字符串到整数的映射，返回逐键相加的新字典。`{'x':2}` 和 `{'x':3,'y':1}` 得到 `{'x':5,'y':1}`，两个原字典保持不变。

这道题练对象别名：`result=left` 不会创建新字典。先画出引用，再比较 `dict(left)`。参考实现从左映射复制，遍历右映射，用 `get(key,0)` 累加。进阶修改：支持负计数并讨论它代表删除、修正还是非法输入；不要未定义业务含义就悄悄删除零值键。

## F-P3：线性插值分位数

将非空 values 排序，q 在 [0,1]，位置为 `(n-1)*q`，取相邻两点线性插值。`[-1,0,4,11]` 的中位数是 2；单元素的所有合法 q 都返回该元素。空数组或非法 q 抛 ValueError。

先在纸上算 q=0、0.25、0.5、1 的位置。不要把某个元素编号四舍五入后直接返回。这个约定与其他分位数定义可能不同，因此报告 p95 时需要说明算法。参考实现用 floor／ceil 后按小数部分加权；进阶任务为多个定义做对照，而不是争论哪一个“永远正确”。

## A-P1：严格工具参数

只接受恰好包含 `a` 和 `b` 的 dict，两个值必须是 Python int，不能是 bool，返回它们的和。缺字段、多字段、类型错误都抛 ValueError。

关键反例是 `{'a':True,'b':1}`。Python 中 bool 是 int 的子类，`isinstance(True,int)` 为真，但 JSON 工具契约通常区分 boolean 和 integer。参考解法检查精确类型和键集合；进阶任务加数值范围并在报错中指出字段，同时不把内部堆栈直接传给模型。

## A-P2：交错工具流式片段

每个片段含非负 index，可含 id、name 和 arguments 字符串片段。不同 index 的片段会交错。同一 index 的身份字段不可冲突；arguments 按接收顺序拼接。结束后按 index 排序，返回 id、name 和解析后的 arguments 对象。身份不完整、JSON 不完整或 arguments 不是对象时抛 ValueError。

先处理 index=1 的 `{"x":`，再处理 index=0 的完整参数，最后处理 index=1 的 `2}`。不能用一个全局字符串接所有工具。也不能在收到第一个 `{` 时就把解析错误当成整个请求坏了；片段本来尚未完成。参考实现用 dict 存每个调用的累积状态，流结束时才解析。本题不等于完整 SSE／UTF-8 parser，字节边界处理在 A／S 卷另练。

## A-P3：显式状态转换

合法转换：ready+tool_call→waiting_tool；waiting_tool+tool_result→ready；ready+answer→finished；ready 或 waiting_tool 遇 cancel→cancelled。其他情况抛 ValueError，尤其 finished 不能重新产生工具调用。

参考实现是一个 `(state,event)` 到新状态的表。进阶题加入 error 与恢复时，先定义哪个错误能恢复、恢复需要什么持久状态，再添加表项。不要为了让测试成功允许所有状态任意跳转。

## R-P1：BM25 的单词项贡献

输入 tf、df、文档数 n、文档长度、平均长度，采用 `log(1+(n-df+0.5)/(df+0.5))` 的 IDF 和标准 k1／b 归一化。tf=0 返回 0。参数要求 tf/df/length 非负、0≤df≤n、n 和平均长度为正、k1>0、0≤b≤1。

手算基准 tf=1、df=1、n=3、length=average=10、k1=1.2、b=.75。长度归一化项为 1.2，词频分子为 2.2，分母也为 2.2，所以分数等于 `log(8/3)`。参考实现先计算 IDF，再计算 norm，不把查询中重复词项的策略偷偷混进这一个单词项函数。进阶题改变文档长度，解释为何相同 tf 的长文档贡献较低。

## R-P2：倒数排名融合 RRF

输入多个按相关性排序的文档 ID 列表。每个榜单第 r 位贡献 `1/(constant+r)`，r 从 1 开始；同一榜单的重复 ID 只计首次出现，后续原始排名位置不重新编号。跨榜单相加，按分数降序、ID 升序打破平局，返回前 k 个 `(id,score)`。

为什么融合排名而非直接相加 BM25 与向量相似度？因为不同检索器的数值尺度不相同。RRF 牺牲一部分绝对分数信息，换取简单可解释的合并。参考实现每个榜单使用 seen 集合，再累加字典。进阶任务构造一个 RRF 不理想的例子，说明它不是万能重排器。

## R-P3：检索前可见性

records 每条含 tenant、allowed、deleted。只返回 tenant 匹配、未删除、allowed 包含 `'*'` 或 `'user:'+actor` 的条目，保留输入顺序。空 allowed 拒绝所有人。

参考解法先取当前主体可匹配的授权集合，再过滤。不要先把所有文档送入检索／模型后才删掉最后答案中的私有文本。本题只教本地可见性谓词，真实系统的 tenant／actor 必须由认证层提供。进阶题加 group 与文档版本，检查失效证据的处理。

## M-P1：带 mask 的稳定 softmax

在最后一维归一化。输入为二维有限浮点数组，mask 可广播；不可见位置输出 0。整行不可见时整行输出 0，不产生 NaN。示例 `[1000,1000,-1000]` 配 `[True,True,False]` 得到 `[.5,.5,0]`。

参考步骤：用 -inf 表示不可见；取行最大值；为整行 -inf 的情况选择安全减数；指数；求和；只在分母非零处相除。这里减最大值不改变结果，因为分子与分母同乘一个常数。进阶题讨论 +inf／NaN 的输入契约，不要无声吞掉不合法输入。

## M-P2：因果 self-attention

输入 Q/K/V 二维数组，序列长度相同，Q/K 最后一维相同，K/V 第一维相同。计算 `softmax(QKᵀ/√d + causal mask)V`，第 i 行只允许 j≤i。返回结果，不返回权重。非法形状抛 ValueError。

参考例子：Q/K 都为 1，所有可见分数相同，V 的三行为 `[1,0]`、`[3,2]`、`[8,4]`。输出依次是第一行本身、前两行平均、前三行平均，即 `[1,0]`、`[2,1]`、`[4,2]`。改变最后一行 V 不应影响前两行输出，这是比一个数值截图更有力的不变量。

## M-P3：线性注意力一步递推

本题 q/k 已经是正值特征向量，不负责特征映射。状态 S 形状为 feature_dim×value_dim，z 为 feature_dim。更新 `S'=S+k⊗v`、`z'=z+k`，输出 `qᵀS'/(qᵀz')`，再返回新状态。不能修改输入状态，分母非正抛 ValueError。

单 token 情况输出应等于 v，因为分子分母中的 qᵀk 抵消。两个 token 时可以直接用两项加权平均对照。参考实现用 outer 和矩阵乘法。进阶题解释为什么这个函数不是 softmax attention 的精确替代；状态固定大小也不意味着保留了任意历史查询需要的全部信息。

## S-P1：分页计数

tokens 为非负整数，block_size 为正整数，返回向上取整页数。0 token 需要 0 页；4 token、block=4 需要 1 页；5 token 需要 2 页。不能用普通除法结果再向下取整。

参考公式 `(tokens+block_size-1)//block_size`。它是整数运算，不会引入大整数浮点舍入。进阶题考虑末页内部碎片，并比较 1 个请求的余量与多个请求总碎片之间的关系。

## S-P2：共享页引用计数

tables 是 request ID 到物理页 ID 列表的映射；统计长度 num_pages 的计数数组。`{'a':[0,1],'b':[0,2]}` 得到 `[2,1,1,0]`。非法物理页 ID 抛 ValueError。

参考实现逐表累加，而不是把所有页放进 set。set 只能判断有没有出现，不能表达两个请求共享页 0。进阶题删除请求 a 后的计数，再判断哪些页可释放；当一个请求要写共享的未满页时，还需 Copy-on-Write，不是只增加引用计数。

## S-P3：有限 token 预算

waiting 是 `(request_id, remaining_tokens)` 顺序列表，budget 是非负整数。从前往后分配，可部分满足一个请求，返回实际分配的非零条目。不能修改原列表，负 remaining／budget 抛 ValueError。

例子 a 剩 3、b 剩 5，预算 5，结果 a 得 3、b 得 2。参考实现每次 take=min(remaining,budget)，然后减预算。进阶题构造 FIFO 饿死短请求的工作负载，再修改策略；公平性与吞吐／首 token 延迟的取舍需要测量，而不是随意打乱列表。

## K-P1：stride 地址

index、shape、strides 维度数相同，index 每一维在 [0,shape)；offset 是 `base+Σ index_i*stride_i`。stride 以元素计，不以字节计。允许负 stride，但必须给正确 base。

二维 shape=(2,3)、strides=(3,1)，位置(1,2)偏移为 5。转置视图 shape=(3,2)、strides=(1,3)，位置(2,1)仍为 5。参考解法先检查坐标再求和。进阶题乘元素字节数得到地址，并解释为何不同 dtype 的相同元素偏移对应不同字节偏移。

## K-P2：分块矩阵乘法

A 为 M×K，B 为 K×N，block 为正整数；按输出行块、列块和归约 K 块累加。必须支持 M/N/K 不能被 block 整除。不能仅在 block=1 时正确。

参考实现初始化输出为零，在每个 K 子块执行子矩阵乘法并累加。若每次赋值而非累加，最后只剩最后一个 K 块。进阶题在 GPU 上实现时讨论尾块 mask、同步与 shared memory 生命周期，NumPy slice 自动截断不代表 CUDA 会自动防越界。

## K-P3：在线 softmax attention

Q/K/V 为二维，Q/K 特征维相同，K/V 长度相同；允许 query 长度与 key 长度不同。本题不含 causal mask。按 key 分块，维护每个 query 的最大值、分母与加权分子，返回 attention 输出。非法形状或非正 block 抛 ValueError。

参考解法每次新最大值为 m'，旧分子／分母乘 exp(m-m')，新权重按 exp(scores-m') 计算。输出只在所有块处理完后相除。评分使用非整块长度和较大 score 尺度，能抓住漏算尾块和没有正确重标定的问题。进阶题增加 causal mask，并单独处理某块全不可见，而不是让 -inf-(-inf) 产生 NaN。

## 做完以后

选一题写一个测试中没有的反例。把输入缩到最小，说明它为什么能抓住某一种错误。然后从空白文件重新实现那一题，并在对应章节记录自己的证据。不要把“我导入了 solutions，18题全过”记成掌握；评分只验证调用到的函数行为。
