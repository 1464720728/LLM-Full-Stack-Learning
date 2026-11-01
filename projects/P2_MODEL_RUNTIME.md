# P2 从一个 token 的计算，走到多请求运行时与真正的算子替换

本项目的目标是让你能够沿着一条请求追踪全部关键数据：输入 token 怎样进入模型，模型怎样产生 logits，采样怎样选出下一 token，调度器怎样交错多个请求，KV 状态怎样归属于不同请求，注意力算法又怎样在保持定义的前提下被替换。每一步都用小规模数值与可运行代码核对。

这里有三份完整参考：[model_runtime.py](model_runtime.py) 把模型接入现有元数据调度器；[incremental_runtime.py](incremental_runtime.py) 实现真正的每请求数值 KV、取消与释放，并将在线注意力替换进完整 decoder；[paged_attention.py](../labs/serving/paged_attention.py) 展示数值分页缓存与共享尾页 COW。三者分别承担不同教学任务，后文会逐一说明状态与接口边界。

先修是模型卷的张量、注意力、Dense decoder 与 KV，小部分框架卷的请求与调度，以及算子卷的在线归一化。你不必先会优化 GPU，也不需要先部署大型权重。所有默认实验用 NumPy CPU 运行。把小系统读透以后，再迁移到实际 GPU 框架，才能知道一个性能优化究竟修改了哪种计算或状态。

## 第一阶段：把一个模型请求写成数据契约

### 1. 文本、token ID、向量与分数不能混成一层

用户看到的是文本，模型常常接收的是 tokenizer 生成的整数序列。整数是词表中的编号，不是这个词的语义大小。编号十不表示比编号五“更重要”，两个编号相加也通常没有语言含义。embedding 表把编号映射到向量，后续模型在这些向量上进行计算。

本项目暂时直接给定 token ID，例如 A 的 prompt 为一、二，B 的 prompt 为二、三、四、五、六。这样可以把 tokenizer 的规则从当前实验中移开，专注于模型和运行时。直接给整数不是说真实服务可以忽略 tokenizer；真实部署还必须核对词表、特殊 token、聊天模板与模型权重是否匹配。

模型返回 logits，一般是对词表中每个候选 token 的实数分数。logits 不是已经归一化的概率，也不是最终字符串。温度为零的本项目采用贪心选择，即选择分数最大的候选编号。之后才把编号加到输出序列，并作为后续模型输入的一部分。若要显示文字，还需要用对应 tokenizer 解码。

用一个三候选例子区分这些对象。假设 logits 是二点零、一点零、负一点零，argmax 返回零，表示第零号候选分数最高，而不是返回数值二点零。softmax 会把三个分数转成概率，仍然保留三个候选；采样或贪心选择才把分布或分数变成一个具体 token。不能把“概率最大值”误当作“token ID”。

### 2. 为什么生成一次后还要再调用模型

自回归生成可以写成重复过程：给定当前上下文，计算下一 token 的分数，选择 token，把它加入上下文，然后重新计算下一位置的分数。后一步条件里包含前一步的实际选择，因此不能在最朴素解码里同时独立决定全部未来 token。模型一次前向能输出多个位置的 logits，但那些位置对应的是已给定输入下的条件，而不是任意未知未来的完整答案。

假设 prompt 为一、二，第一步选出二。下一步输入应是“一、二、二”，不能继续用原来“一、二”并期待模型知道自己刚才选了什么。再选出十二后，输入变成“一、二、二、十二”。如果某个循环只把 token 放进 result，却忘记更新 context，它会反复计算同一输入，产生与真正自回归生成不同的行为。

这也是为什么最终输出列表与内部上下文列表分开保存。输出只包含新生成的 token，方便返回给调用者；上下文包含 prompt 和已生成部分，供模型继续计算。两者有重叠关系但不是同一个对象。若把它们错误地共享成同一个可变列表，输出中可能混入 prompt，或者修改输出时破坏后续输入。

本项目的停止条件是生成指定数量的新 token。真实系统还可能遇到 EOS、停止字符串、用户取消、超时或资源限制。每种条件都需要明确是否把最后一个 token 返回、是否继续计算它的 KV、以及如何释放状态。先固定一个最小条件，是为了让这些后续差异有可比较的基线。

### 3. 三层接口分别负责什么

模型层回答“这个 token 上下文经过当前参数后，得到什么 logits”。调度层回答“这一轮哪些请求允许推进多少个输入位置”。算子层回答“某个明确数学运算怎样实现”。它们最终会连接起来，但先把接口分开，可以避免一个错误同时隐藏在多个层次里。

现有 Engine 只要求 backend 提供 vocab_size 与 logits(token_ids)。这个接口没有请求 ID、缓存句柄或释放方法，因此最自然的实现是无状态地根据完整上下文重新计算。DecoderBackend 正是这样做的。你不能在它内部偷偷放一个全局 cache，然后仍宣称接口支持多个交错请求；它根本不知道当前 token_ids 属于哪个请求。

新增 IncrementalRunner 则显式按 rid 保存状态，提供 start、advance 与 cancel。start 为一个新请求建立初始 KV；advance 推进该请求；cancel 清理该请求的可继续计算状态。它是另一种更明确的接口，目的是先把每请求缓存契约验证清楚，再考虑更复杂的批处理或分页适配。

算子替换通过 OnlineDecoder 完成。它保留 TinyDecoder 的模型参数、投影、位置编码、残差与 MLP，只重写注意力计算路径。这样的替换让你可以在完整模型上检查误差，而不只是在孤立的 Q/K/V 数组上说“两个函数差不多”。替换成立的条件包括 shape、头映射、位置、mask、缩放与 dtype 一致。

## 第二阶段：先手算一个缩小到两维的模型

### 4. 明确简化条件，再写矩阵

为了把每个数算清楚，先使用一个单层、单头、二维向量的模型。它没有归一化、RoPE 和 MLP，只有 embedding、因果注意力、残差与输出投影。这个例子只解释最内层数据流，不能拿它的数值去对照完整 TinyDecoder 的随机参数结果。

词表有三个 token，embedding 依次是 [1,0]、[0,1]、[1,1]。Q、K、V 三个投影矩阵暂时都取单位矩阵，所以投影后向量仍等于输入。输出矩阵的两行分别是 [1,0,-1] 与 [0,1,1]。输入 token 序列为零、一，因此 X 的两行分别为 [1,0] 与 [0,1]。

~~~python
import numpy as np

embedding = np.array([[1., 0.], [0., 1.], [1., 1.]])
output = np.array([[1., 0., -1.], [0., 1., 1.]])
ids = np.array([0, 1])
x = embedding[ids]
q = x.copy()
k = x.copy()
v = x.copy()
scores = q @ k.T / np.sqrt(2)
print(x)
print(scores)
~~~

embedding[ids] 是按行查表，不是把 ID 乘以整个矩阵。k.T 交换两个轴，把原来两行两列的 K 转置，用来形成每个 query 与每个 key 的点积。此处 shape 恰好都是二乘二，容易让错误的转置看起来还能运行，所以后面要用不相等的序列长度与特征维度检测布局错误。

点积结果在对角线上为一，非对角线为零，再除以根号二。令 a=1/√2，分数矩阵就是 [[a,0],[0,a]]。每个分数的行表示哪个 query，列表示哪个 key。请把这两个轴的语义写在纸上；后面因果 mask、softmax 与 V 加权都依赖这个对应关系。

### 5. 把因果 mask 与 softmax 的每一步算完

第零个位置只能读取自己，不能读取位置一。对第零行，第二个分数必须被排除。把无效分数设为负无穷后，指数为零，归一化得到 [1,0]，所以注意力输出就是第零个 V，即 [1,0]。如果你只把未来分数设成零，它的指数仍是一，会错误地参与归一化。

第一个位置允许读取两个 key，分数为 [0,a]。指数为 [1,exp(a)]，分母是 1+exp(a)。定义 p=1/(1+exp(a))，则两项权重为 p 与 1-p。a 约为零点七零七一，因此 p 约为零点三三零二。加权 V 得到 [p,1-p]，因为两个 V 分别只在第一或第二坐标上为一。

再加上当前位置的残差 [0,1]，得到 [p,2-p]。乘输出矩阵后，三个词表分数依次是 p、2-p、2-2p。代入 p 后，第二个分数最大，因此下一 token 是编号一。注意我们比较的是三个分数，输出的却是最大分数所在的编号。

完整代码如下，可以和手算逐项核对：

~~~python
mask = np.arange(2)[None, :] <= np.arange(2)[:, None]
masked = np.where(mask, scores, -np.inf)
shifted = masked - masked.max(axis=1, keepdims=True)
weights = np.exp(shifted)
weights = weights / weights.sum(axis=1, keepdims=True)
attended = weights @ v
hidden = x + attended
logits = hidden @ output
token = int(np.argmax(logits[-1]))
assert token == 1
assert np.allclose(weights[0], [1., 0.])
~~~

axis=1 表示沿 key 轴归一化，每个 query 的权重和为一。keepdims 保留长度为一的轴，让减法和除法沿正确方向广播。如果省略它，在这个恰好方阵的例子里可能得到形状可运行但方向错误的计算，所以应再尝试三行两列的中间量，确认自己理解广播，而不是只观察程序没报错。

### 6. 生成第二个 token 时旧 key/value 怎样复用

第一步已经选出 token 一，新的上下文为零、一、一。新位置的向量仍为 [0,1]，它的 query 与三个 key 的点积分数为 [0,a,a]。两个历史 K/V 没有变化，新增加的是当前位置的 K/V。对这个简化模型而言，旧向量可以直接保留，不需要再次查表和投影。

归一化分母变为 1+2exp(a)。第一个 key 的权重为 1 除以这个分母，后两个各是 exp(a) 除以分母。第一个权重记为 b，则后两个加起来是 1-b，注意力输出为 [b,1-b]，残差后得到 [b,2-b]。虽然输出 token 仍可能是编号一，具体权重与 logits 已经改变，说明模型确实看到了更长上下文。

KV cache 保存的是过去位置的 K 与 V，而不是过去的最终答案字符串。新 query 仍然要与过去全部可见 key 交互，因此使用缓存不会使长历史注意力变成完全不随长度增长的工作。它主要避免重复计算过去位置的投影和中间表示，以及不必要的旧 query 输出。

在完整多层模型里，每一层有自己的 K/V。某个位置在第二层的输入来自第一层处理后的表示，不等于原始 embedding，因此不能让所有层共用一份“token 向量缓存”。你应把缓存理解成“特定模型层、特定头、特定位置下的投影状态”。这个身份层级会直接影响缓存字节、释放和共享规则。

## 第三阶段：读懂完整 TinyDecoder 的实际张量

### 7. 从配置推导参数尺寸，而不是猜变量名字

默认 TinyDecoder 词表大小三十二，隐藏维度十六，query 头数四，KV 头数二，层数二，MLP 中间维度三十二。每个头的维度是十六除以四，得到四。因为采用 GQA，K/V 的总投影通道是两个 KV 头乘每头四维，也就是八，而 Q 的总通道仍为十六。

| 对象 | 默认 shape | 用途 |
|---|---|---|
| embedding | 32×16 | token ID 查成隐藏向量 |
| 每层 Q 投影 | 16×16 | 生成四个 query 头 |
| 每层 K/V 投影 | 各 16×8 | 生成两个 KV 头 |
| 每层输出投影 | 16×16 | 合并多头后变换 |
| gate/up 投影 | 各 16×32 | MLP 两条输入分支 |
| down 投影 | 32×16 | 回到隐藏维度 |
| LM 输出矩阵 | 16×32 | 生成词表 logits |

初始化中的随机种子使这份教学参数可以复现，但这些权重没有经过语言训练。输出序列可用于验证计算和调度一致性，不可用来评估语言知识。一个模型在数学结构上完整，不等于它在任务上已经有能力；结构、参数与训练数据是不同层面。

配置之间还有可整除条件。隐藏维度必须能分成 query 头，query 头数必须能按组对应 KV 头，RoPE 的头内坐标需要两两配对。代码在构造时检查这些条件，因为错误配置不能等到某次 reshape 才以模糊异常出现。以后读取大型模型 config 时，同样应先做这些维度推导。

### 8. 一层 forward 的形状怎样变化

输入两个 token，查表后 X 为两行十六列。RMSNorm 按每一行的隐藏特征归一化，shape 不变。Q 投影得到两行十六列，拆成四个头后是 [4,2,4]；K/V 投影得到两行八列，拆成两个头后是 [2,2,4]。这里的轴顺序是头、位置、头内特征，没有 batch 轴。

对 Q/K 应用 RoPE 后，shape 仍相同，但坐标值依位置旋转。V 不执行这一步。GQA 让两个 query 头共享一个 KV 头：query 头零、一对应 KV 头零，query 头二、三对应 KV 头一。教学 reference 可以显式重复 K/V 以便读懂，优化实现则可以直接按索引读取共享头。

每个 query 头的分数是两行两列，四个头合起来为 [4,2,2]。最后一个轴是 key 位置，softmax 沿这个轴进行。加权 V 后回到 [4,2,4]，merge_heads 合并成 [2,16]，再乘输出投影，最后与输入残差相加。只看最终 shape 相同，无法证明头之间的元素分组正确，必须核对 reshape 与 transpose 的对应关系。

之后进入第二个 RMSNorm，再进入 SwiGLU。gate 与 up 都得到 [2,32]，gate 经过 SiLU，两个分支逐元素相乘，再用 down 投影回 [2,16]，与前一残差结果相加。MLP 前的 norm 输入是注意力残差后的新表示，不能偷用本层最初的归一化结果，否则就改变了模型连接方式。

### 9. logits 接口为什么只取最后一行

TinyDecoder.forward 返回每个输入位置对应的词表分数，shape 是 [T,32]。DecoderBackend 取最后一行，得到长度三十二的向量，再转换成普通 Python 列表。调度器的采样函数只需要这一行来决定下一 token，不需要把所有历史位置的输出再取一遍。

~~~python
class DecoderBackend:
    def __init__(self, model):
        self.model = model
        self.vocab_size = model.config.vocab

    def logits(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        all_positions = self.model.forward(ids)
        last_position = all_positions[-1]
        return last_position.tolist()
~~~

asarray 把输入变成模型需要的数值数组，dtype 显式表达 token 是整数 ID。它不是业务输入验证的全部替代品：一个过于宽松的上游若把浮点值交给转换，可能发生截断。因此真实边界应先验证输入，本项目 Request 与 IncrementalRunner 都对外部 token 参数设定了明确条件。

tolist 把 NumPy 数组转换成普通 Python 值，满足现有 backend 契约。这个小转换在 CPU 教学里方便，但真实 GPU 上频繁把整个词表搬回 CPU 会带来同步与传输开销。读者应学会区分“接口为了清晰做了转换”与“生产系统也应该这样搬数据”，后面的采样后端会说明更贴近设备的实现。

### 10. 先建立不经过调度器的生成基线

直接生成只需要一个模型、一个上下文列表和一个输出列表。每轮计算最后位置 logits，取 argmax，把 token 同时加入上下文与结果。这个版本会重复计算完整历史，速度不是目标；它的作用是作为后续缓存与调度实现的独立参照。

~~~python
def greedy_reference(backend, prompt, count):
    context = list(prompt)
    output_tokens = []
    for _ in range(count):
        logits = backend.logits(context)
        token = int(np.argmax(logits))
        context.append(token)
        output_tokens.append(token)
    return output_tokens
~~~

context=list(prompt) 建立新列表，因此不会修改调用者的原 prompt。output_tokens 每次调用重新创建，避免多个请求共享输出。循环里的 count 是新 token 数，不是上下文总长；长度为五的 prompt 生成四个 token 后，context 长度为九，输出长度为四。

注意最后一个被选出的 token 通常不需要再经过模型，除非还要继续生成。生成四个 token 的四次前向，其输入长度依次为原 prompt 长度、加一、加二、加三。这个差一关系会影响 KV 容量上界与调度器 computed 字段，后面我们会用实际状态表检查它。

## 第四阶段：让两个请求在同一个调度器里交错

### 11. Request 保存的不是一句 prompt，而是一段生命周期

现有 Request 包含请求 ID、prompt、新 token 上限、到达时间、采样参数、输出列表、已计算位置数、状态与结束时间。prompt 是输入材料，output 是生成结果，computed 表示调度层认为已经处理到的输入位置。它们会一起决定下一轮需要做什么，但不能互相替代。

context 属性每次返回 prompt 与 output 的拼接，因此随着输出增长而增长。computed 没有自动跟着增长，因为刚选出的 token 还需要在下一轮作为输入被处理。两者的差 len(context)-computed，就是当前尚未推进的输入位置数。prefill 时这个差可能很大，逐 token decode 时通常为一。

状态从 pending 到 waiting，再到 running，最后进入 finished、cancelled 或 error。pending 表示已提交但尚未到达当前模拟时间，waiting 表示等待运行名额，running 表示已经拥有活跃请求状态。不要把“提交成功”直接理解成“模型已经开始算”，队列层次影响延迟与取消行为。

Request 在提交时还必须是新对象，不能携带上一轮运行留下的 output、computed 或 finish_ms。否则一个表面新请求可能从旧中间状态继续，绕过初始化和资源检查。框架卷的边界测试专门覆盖这个约束；你在自己的 API 中也应决定谁有权创建、恢复或修改运行状态。

### 12. 三个常用预算控制不同东西

token_budget 控制一轮计划中处理的输入 token 总数；chunk_size 控制一个 prefill 请求一次最多推进多少输入位置；max_sequences 控制同时活跃的请求数。把它们都叫“并发”会导致错误推理。例如 max_sequences=2 不表示每轮必然生成两个新 token，因为其中一个请求可能仍在处理长 prompt。

本项目选择 token_budget=3、chunk_size=2、max_sequences=2。请求 A 的 prompt 长度二，请求 B 的长度五，两者各生成四个 token。调度器可以在同一轮给 A 两个位置，再给 B 一个位置；总数三满足预算。A 的完整 prompt 已处理，可以产生首 token；B 只处理一小段，还不能产生本项目定义下的首 token。

预算是严格正整数，布尔值或一点五这样的数不应通过。Python 的 bool 与 int 存在继承关系，因此只写 isinstance(value,int) 可能把 true 当成一。更深的问题是允许浮点预算会令切片、计数和名额比较语义不一致。明确类型条件可以把错误挡在构造阶段，而不是在某个调度分支里才崩溃。

这里的 token 成本并不相等。处理三个 prefill token 与处理三个长上下文 decode 请求的真实代价可能非常不同；预算是调度控制参数，不是准确的浮点计算量或显存字节。后续做性能模型时需要增加上下文长度、模型结构与实际硬件数据，不能把一个总 token 数当成所有成本的共同单位。

### 13. 六轮完整状态轨迹

使用同一份模型参数，默认贪心输出的一个实际记录为：A 生成 [2,12,2,20]，B 生成 [1,17,12,23]。这些编号只服务于本固定教学配置，不代表自然语言能力。修改种子或模型结构后数字可以改变，核心验收仍是直接基线与调度输出一致。

| 轮次 | 本轮计划 | 完成本轮后产生的 token | 模拟累计时间 |
|---|---|---|---:|
| 1 | A prefill 2，B prefill 1 | A→2 | 1.75 |
| 2 | B prefill 2，A decode 1 | A→12 | 3.35 |
| 3 | A decode 1，B prefill 2 | A→2，B→1 | 4.95 |
| 4 | B decode 1，A decode 1 | B→17，A→20 并完成 | 6.15 |
| 5 | B decode 1 | B→12 | 7.25 |
| 6 | B decode 1 | B→23 并完成 | 8.35 |

第一轮结束时，A 的 computed 为二，但 output 已多出一个 token，因此 A 的 context 长度为三。第二轮给 A 一个 decode 输入位置，computed 变三，正好追上旧 context，于是产生下一个输出，新的 context 又长到四。这个“追上输入后产生一个新待处理 token”的节奏贯穿解码过程。

B 前两轮分别处理一和两个 prompt 位置，computed 到三，还差两个。第三轮补完这两个位置，才第一次满足 computed 等于当前 context 长度。因此 B 的第一个输出在第三轮出现，不能把 B 在第一轮已经分到名额解释为“第一轮就完成首 token”。计划进入与输出产生是两个观察点。

第四轮 A 产生第四个新 token 后立即完成并释放页池中的请求引用。最后生成的二十不需要再作为输入计算，因为我们不再要求第五个输出。若你按 prompt 长度加全部输出长度估计“已计算缓存长度”，就会多算一个位置。代码在资源可行性检查里也体现了这个差一关系。

### 14. 自己打印事件，而不是只相信这张表

下面是完整追踪脚本。它使用现有 Engine，打印 batch、token 与 finished 事件。你可以先按表预测，再运行；改变一个预算后重新画表，能检验自己是否真的理解调度规则。

~~~python
from labs.serving.engine import Engine, Request
from projects.model_runtime import DecoderBackend

engine = Engine(
    backend=DecoderBackend(),
    token_budget=3, chunk_size=2, max_sequences=2,
    prefix_entries=0,
)
engine.submit(Request("A", (1, 2), max_new_tokens=4))
engine.submit(Request("B", (2, 3, 4, 5, 6), max_new_tokens=4))
engine.run()
for event in engine.events:
    if event["kind"] in {"batch", "token", "finished"}:
        print(event)
~~~

batch 事件里 start 是本次处理片段在请求上下文中的起始位置，count 是本次推进数量，phase 表示 prefill 或 decode。它们描述调度计划，不是模型内部每个矩阵乘法的时间线。token 事件表示产生了新输出，finished 表示生命周期结束并触发资源回收。

prefix_entries=0 在这个实验中关掉前缀复用，减少需要同时考虑的状态。先把单请求推进和交错规则读清，再开启前缀缓存观察差异。关闭一个机制用于建立对照，不意味着该机制在真实框架中不重要；有清楚基线才能解释新增优化改变了哪些步骤。

### 15. 为什么时间是可手算的模拟值

Engine 用一个明确的成本规则累加时间：每轮固定一，加上每个 prefill token 的零点二五，以及每个 decode token 的零点一。第一轮有三个 prefill token，因此增量为一点七五；第二轮有两个 prefill 加一个 decode，增量为一点六。它没有读取 GPU 时钟，也没有测量 NumPy 的真实耗时。

这套规则让我们研究“调度次序改变后，各请求什么时候获得服务”的逻辑。例如 A 的首 token 时间是一点七五，B 的首 token 时间是四点九五。A 的后续间隔为一点六、一点六、一点二，平均约一点四六七；B 的后续间隔为一点二、一点一、一点一，平均约一点一三三。你可以从事件表直接算出这些指标。

这些数字不能换个标题就变成 H100 延迟。模拟器没有真实 kernel、显存带宽、排队网络、同步、通信或编译开销。即使本地 Python 程序实际运行需要更长或更短时间，report 中的合成时间也不随之改变。它属于模型假设下的推演结果，不是设备观测。

## 第五阶段：识别元数据缓存与数值缓存的区别

### 16. 页池中放 token ID，并不等于已经复用了 K/V

原有 Engine 的 PagePool 存 token ID，用来模拟已处理输入的页表、引用计数与共享关系。它可以验证一个请求释放后其他请求仍持有共享页，也可以验证资源不足时不留下半次分配。但 DecoderBackend 仍然对完整上下文调用 forward，真实数值 K/V 没有从这个页池取出。

因此，这个项目同时包含真实模型计算与缓存管理模拟，两者不能混称为一个已经高效复用数值 KV 的推理引擎。读者应能指出谁管理 token 元数据，谁执行矩阵运算，二者的接口目前传了什么。如果只把 PagePool 的变量名改成 KVCache，并不会改变实际执行路径。

为什么还保留这个阶段？因为状态与所有权本身就值得独立验证。若页表、COW 或释放顺序有错，直接把大张量放进去只会让错误更难看见。用小整数先研究资源管理，再把同一不变量迁移到数值数组，是有明确目标的分层教学过程。

新增 IncrementalRunner 则真正调用 TinyDecoder.step 并保存返回的每层 K/V 数组。它不使用 PagePool，而是为每请求保存连续 NumPy 数组。这样先获得“数值缓存正确”的完整参照。再与 S19 的 TensorKV 分页存储结合时，你有两条独立基线可以检查模型数值与页表管理，而不是一次替换全部内容后只比较最终答案。

### 17. 为每请求建立独立状态

RunningRequest 保存 prompt、新 token 上限、cache、next_logits、output 与 status。这里最容易误解的是 next_logits：它对应已经处理完当前输入后，预测下一个 token 的分数，而不是刚刚选择过的 token 的 one-hot 标签。start 完成 prompt 计算后，就已经拥有第一组 next_logits。

~~~python
from projects.incremental_runtime import IncrementalRunner

runner = IncrementalRunner()
runner.start("A", [1, 2], max_new_tokens=4)
runner.start("B", [2, 3, 4, 5, 6], max_new_tokens=4)
print(runner.snapshot())
~~~

默认模型有两层，每层的 K 与 V shape 分别为 [2,T,4]，dtype 为 float64。A 的 T 为二，每层 K 的元素数是二乘二乘四，共十六；K 与 V 合起来三十二个元素，每元素八字节，一层二百五十六字节，两层五百一十二字节。B 的 T 为五，同理得到一千二百八十字节。

这个数只统计 K/V 数组的元素存储，不包括 Python 对象开销、参数矩阵、临时激活、分配器碎片或预留空间。知道公式的范围比得到一个看似精确的数字更重要。否则你可能拿这个缓存值去预测完整进程 RSS 或 GPU 显存，然后把正常差异误当成内存泄漏。

start 按 prompt 顺序逐 token 调用 step，得到最后位置的 logits 与完整前缀 cache。它在全部 prefill 成功后才把 RunningRequest 放进 requests，避免失败一半就留下看似可运行的请求。这个教学实现的 prefill 是顺序循环，便于逐步观察；它没有声称获得批量矩阵计算的 prefill 性能。

### 18. advance 为什么在最后一次输出时释放缓存

advance 先检查请求仍处于 running，然后从 next_logits 选择 token。如果这是最后一个所需输出，就不再计算该 token 的 K/V，而是清空 cache 与 next_logits，追加输出并标记 finished。否则把选出的 token 送入 step，取得下一组 logits 与更长的 cache，再更新请求状态。

这个顺序让一次推进对应一个明确的输出 token。计算下一状态时，TinyDecoder.step 创建新的缓存数组，不原地修改旧缓存。若计算抛异常，尚未把新输出与新缓存发布到请求对象，较容易保持该步的状态一致。若换成会原地改缓存的后端，就必须重新审查异常中断下的行为，不能直接继承这个保证。

A 开始时缓存长度为二。第一次 advance 产生一个输出并计算其状态，缓存长度变三；第二次变四；第三次变五；第四次产生最终输出后清空缓存。输出长度最终为四，而“完成前最后一次仍持有的缓存长度”为五，正好等于 prompt 二加上前三个生成 token。把这组数字手算一遍，比背容量公式更可靠。

这个 runner 的 finished 不代表模型权重从内存移除。它释放的是该请求的引用，模型对象仍被 runner 保留，可以服务其他请求。实际设备分配器也可能保留可复用内存，因此应用对象释放、张量不可达和操作系统或 GPU 可见占用下降之间，还需要区分具体运行时行为。

### 19. 交错执行为什么不应改变贪心输出

对于参数固定、输入固定且数值路径一致的模型，每个请求的下一 logits 只依赖它自己的上下文与状态。把 A、B 按先后全部做完，或交替推进，应该得到相同的每请求序列。这个性质让我们检查是否误用了共享缓存、共享输出列表或隐藏游标。

~~~bash
python -m projects.incremental_runtime --policy fifo
python -m projects.incremental_runtime --policy round_robin
~~~

两种运行都应满足 matches_full_context_reference=true，且 final 中所有已完成请求 cache_bytes 为零。直接基线每次重算完整上下文，增量路径则使用 cache；这两种不同计算组织方式给出相同结果，比让同一个缓存函数自己对照自己更有说服力。

轮转的顺序由一个双端队列保存。每次从队首取出请求，推进一步；尚未完成就放到队尾，完成则不再入队。若选择 FIFO，尚未完成的请求放回队首，因此它继续运行直到结束。这里的队列只保存请求 ID，真实 K/V 仍按 ID 保存在 requests 中；调度顺序与状态归属是两件事。

不能用不断累加的游标对“每次重新过滤后的活跃列表长度”取模来代替这个队列。设 A 只需要一步，B、C 各需要两步。正确轮转顺序是 A、B、C、B、C。A 完成后列表从 [A,B,C] 缩为 [B,C]，旧游标一若直接作为新下标，会先选 C，把本该轮到的 B 跳过去。这个错误未必改变最终 token，却违反了你声称的调度策略。因此验收既比较数值输出，也检查请求结束时的轮转轨迹。

真实 GPU 上还需要考虑数值执行顺序、kernel 选择与非确定性，因此逐 token 输出一致不能被无限泛化。此处测试小规模 float64 NumPy 数学与请求状态隔离，误差阈值也按这个环境设置。换成低精度 GPU 时，应先比较 logits 误差与采样敏感性，再决定适用验收，而不是简单把所有不一致归咎于调度错误。

### 20. 取消一个请求，另外一个为什么应继续

取消操作只针对某个 rid，把该请求的 cache 与 next_logits 置空，status 改为 cancelled。已经返回的 output 保留，便于解释取消前发生了什么。第二次取消同一请求返回 false，表示没有再次改变活跃状态；尝试 advance 已取消请求则报错，防止无意恢复它。

~~~python
from projects.incremental_runtime import IncrementalRunner

runner = IncrementalRunner()
runner.start("cancel", [1, 2], 4)
runner.start("keep", [2, 3], 2)
runner.advance("cancel")
assert runner.cancel("cancel")
assert runner.snapshot()["cancel"]["cache_bytes"] == 0
runner.run()
assert runner.requests["keep"].status == "finished"
assert runner.requests["cancel"].status == "cancelled"
~~~

这个测试先让被取消请求确实持有并推进过缓存，再取消它。如果一开始就创建一个零输出请求，你可能只测试了“没有分配”，并没有测试“已经分配后回收”。故障与资源测试需要准确选择发生时刻，和 P1 的检索后撤销实验一样。

同步 CPU 上，函数返回后先前 NumPy 操作已结束，所以清理对象引用较直观。GPU 异步执行中，主机发出取消请求时，设备可能仍在读取某些缓冲区。实际框架需要事件、完成通知或延迟回收，不能把本教程的一行置空直接搬成异步设备的安全释放方案。

## 第六阶段：把 K 卷算法真正替换进模型

### 21. 先写清 GQA 头映射与输入尺度

OnlineDecoder 继承 TinyDecoder 的参数初始化、forward 与 step，只重写 _attn。Q/K/V 投影与 RoPE 保留相同计算，旧 K/V 的拼接也保持一致。替换发生在“已得到每头 Q/K/V 后，怎样计算注意力输出”这个位置，之后仍按原方式合并头并执行输出投影。

默认四个 query 头、两个 KV 头，group_size=4//2=2。因此 h 为零或一时，h//group_size 为零；h 为二或三时，为一。每个 query 头把自己的二维 Q 与对应 KV 头的二维 K/V 交给 online_attention。不要用 h % kv_heads，它会得到零、一、零、一，形成另一种分组排列。

online_attention 内部会按头维度做点积分数缩放，因此适配器不能再提前把 Q 除以根号维度，否则缩放发生两次。类似地，它期望的是投影后的 V，不是已经乘了输出投影的结果。算法名称相同不足以保证接口可以直接拼接，必须逐项核对它接受和返回的量。

### 22. 完整 prefill 与单步 decode 使用不同的 mask 条件

完整前向中 query 与 key 都覆盖从零开始的全部输入位置，需要因果 mask，禁止早期 query 读取未来 key。单步 decode 中只有一个 query，它的位置是当前历史最后一位，已有 K/V 都是过去与当前位置，因此全部可以参与，不需要再用局部行号零限制可见 key。

这正是矩形注意力容易出错的地方。如果对形状 [1,D] 的 decode query 使用“key 下标小于等于 query 行号”的规则，query 行号是零，就只剩第一个 key 可见。真正的 query 绝对位置却可能是一千，它应读取整个合法历史。本适配器显式区分全长前向与最后位置单 query 两种情况。

对于其他矩形情况，例如一次解码多个新 token 或带偏移的 chunked prefill，本适配器主动拒绝。它需要一个明确的绝对位置 mask 接口才能正确扩展。拒绝不支持的输入有助于暴露边界；静默把所有矩形都当成无 mask，会让较早的新 query 读到未来新 token。

运行完整替换后的模型：

~~~bash
python -m projects.incremental_runtime --online
~~~

测试不仅比较最终 token，还比较 dense 模型与 online 模型完整 forward 的 logits，并把逐 token step 拼起来与相同 reference 对照。测试覆盖 MQA、GQA 与 MHA 三种 KV 头配置，使用不整除块大小的序列。这样才能同时检查头映射、尾块与缓存路径，而不是只验证一个方便的方形案例。

### 23. 用二维 value 手算在线统计，确认适配器维护了什么

固定一个 query，让三个 key 的分数依次为零、ln2、ln4，对应 value 为 [10,1]、[20,0]、[30,-1]。完整指数权重是 [1,2,4]，所以最终第一坐标为 (10+40+120)/7=170/7，第二坐标为 (1+0-4)/7=-3/7。输出是一个二维向量，不能把分母按 value 坐标各算一遍。

按单 key 分块处理，状态包含最大分数 m、缩放分母 l 与未归一化向量分子 u。第一块后 m=0、l=1、u=[10,1]。第二块最大值升到 ln2，旧统计统一乘二分之一，再加新值，得到 l=1.5、u=[25,0.5]。第三块最大值升到 ln4，再统一乘二分之一并加 [30,-1]，得到 l=1.75、u=[42.5,-0.75]。

最后 u/l 给出 [170/7,-3/7]，与完整归一化一致。这里的 u 从来不是已经除过分母的最终输出，因此更新公式不能拿另一种“保存归一化输出”的实现直接混用。变量同名不足以证明状态含义相同，读源码时要从定义和更新关系确定它维护的量。

~~~python
import numpy as np
from labs.kernels.reference import online_attention

q = np.array([[1., 0.]])
k = np.array([
    [0., 0.],
    [np.sqrt(2) * np.log(2), 0.],
    [np.sqrt(2) * np.log(4), 0.],
])
v = np.array([[10., 1.], [20., 0.], [30., -1.]])
result, trace = online_attention(q, k, v, block=1, return_trace=True)
assert np.allclose(result, [[170/7, -3/7]])
for entry in trace:
    print(entry["start"], entry["m"], entry["l"], entry["numerator"])
~~~

K 的第一坐标乘了根号二，是为了抵消注意力内部除以根号头维度的缩放，让实际分数正好是所需的零、ln2、ln4。这个构造把“我想测试的分数”落实到真实 Q/K 输入，而不是绕过矩阵乘法直接喂一个分数数组。你应能解释这个比例，否则以后很容易在适配器里重复缩放。

### 24. 怎样制造一个能抓住重缩放错误的反例

假设错误实现更新分母时乘 alpha，却忘记给旧分子乘 alpha。第一块看不出错误，因为旧状态全是零；后来最大值升高时，分子与分母采用不同尺度，结果就不再等于完整注意力。如果你只测试所有分数相等的输入，alpha 始终为一，错误可能完全隐藏。

上节分数严格增大，正好能暴露这个问题。另一个反例是先给一个较小分数块，再给一个极大分数块，检查输出是否由新块主导且仍为有限值。还应交换块大小：块一、二、三与不整除序列长度的块，都应在容差内一致。不同块大小改变了状态合并次数，是对不变量的有效检验。

因果性测试则改变未来 K/V，要求早期输出不变。它验证的是可见性，不是一般数值精度。两个实现如果都忘了 mask，随机误差对照可能仍很小；一个独立的未来扰动测试可以揭露这种共同错误。好的验收会同时检查等价性与结构性质，避免两个同样错误的实现互相背书。

## 第七阶段：把数值缓存放入真实分页布局

### 25. 四维数组与页表怎样共同定位一个向量

TensorKV 的 K/V 数组布局是 [physical_pages,block_size,heads,dim]。页表给出某请求的逻辑页对应哪个物理页。访问请求位置 p 时，先用 divmod(p,block_size) 得到逻辑页号与页内槽位，再从该请求的 table 取得物理页号，最后读取数组中的对应 head 与特征坐标。

例如 block_size=4，逻辑位置六对应逻辑页一、槽位二。如果请求页表是 [3,1]，就应读取物理页一的槽位二，而不是物理页六，也不是按物理 ID 大小排序后再读。逻辑顺序来自请求页表；物理页是否连续只影响存储布局与访问性能，不应改变数学上的 token 次序。

预分配四页、每页四槽、每槽一个头、每头二维、float64 时，一份 K 有四乘四乘一乘二个元素，共三十二个，每个八字节，K 占二百五十六字节，K/V 合计五百一十二字节。即使当前只使用两页，数组总分配仍是五百一十二字节；活跃页对应的有效数据量与预分配容量是不同指标。

release 把逻辑页归还空闲池，但不重新申请或缩小那两个大数组。旧数值可能留在无效槽位，正确读取必须遵守每页有效长度。把释放理解为“所有显存立刻清零并还给操作系统”，会误解大多数池化设计。需要清零时可以额外做，但清零不能替代正确的长度与访问范围。

### 26. 共享半页为什么需要 COW

A 已有三个 token，block_size 为四，所以尾页还有一个空位。fork 生成 B 时，两个请求暂时共享同一物理页，引用数为二。随后 A 要追加第四个 token。如果直接写共享页，B 的历史也会被污染；虽然两个请求还共享前三个 token，后续生成已经可能分叉。

正确做法是为 A 分配一个新物理页，复制前三个有效位置的 K/V，再写 A 的新位置，并减少旧页的引用。B 仍指向旧页，之后可以在自己的空槽写不同 token。如果 B 此时已经是旧页唯一拥有者，就不需要再复制一次。COW 的触发条件是共享且将要修改，不是每次 append 都复制整段历史。

下面的完整例子同时检查 token 元数据与实际数值：

~~~python
import numpy as np
from labs.serving.paged_attention import TensorKV

kv = TensorKV(pages=4, block_size=4, heads=1, dim=2)
kv.create("A")
keys = np.array([[[1., 0.]], [[0., 1.]], [[1., 1.]]])
values = np.array([[[10., 0.]], [[0., 20.]], [[30., 30.]]])
kv.append("A", [1, 2, 3], keys, values)
kv.fork("A", "B")
kv.append("A", [4], np.array([[[2., 0.]]]), np.array([[[40., 0.]]]))
kv.append("B", [9], np.array([[[0., 2.]]]), np.array([[[0., 90.]]]))
a_k, a_v = kv.gather("A")
b_k, b_v = kv.gather("B")
assert kv.pool.tokens("A") == [1, 2, 3, 4]
assert kv.pool.tokens("B") == [1, 2, 3, 9]
assert np.array_equal(a_k[:3], b_k[:3])
assert np.array_equal(a_v[:3], b_v[:3])
assert not np.array_equal(a_v[3], b_v[3])
kv.pool.check()
kv.release("A")
kv.release("B")
kv.pool.check()
assert len(kv.pool.free) == 4
~~~

只检查 token 列表是不够的。某个错误实现可能正确复制元数据，却忘记复制 K/V，导致后续注意力读取错误数值。上面两个前缀数组断言明确检查真实数据，最后一项差异则证明分叉后的新值保持独立。最后释放两个请求后，所有页都应回到空闲集合。

### 27. 分配失败为什么应保持旧状态不变

追加操作可能同时需要多个新页。假设原来三个位置、页大小四，一次追加六个位置，总长九，需要三个逻辑页。如果原尾页被共享，还需要为尾页 COW 额外预留一页。所需新页数可以写成新逻辑页数减旧逻辑页数，再加共享尾页复制成本。

在这个例子里，ceil(9/4)-1+1=3，所以必须先确认至少三个空闲页。如果只预留一个、写一部分再发现没有剩余页，旧请求状态就可能被改变一半，恢复和重试都变复杂。PagePool.append 因此先计算 needed，再一次性预留所需页，之后才修改已有页表与内容。

资源不足测试要保存失败前的 table、token 列表、引用数与 free 集合，触发 OutOfPages 后逐项比较。仅仅确认“抛出了异常”不足以证明失败原子性，因为异常前可能已经造成泄漏或污染。这种思路同样适用于张量容量扩展、事务提交和批量工具操作。

### 28. 三个项目阶段如何映射到真实推理框架

到这里，你应该能分别描述三条路径。元数据 Engine 展示调度、请求状态与页池所有权，但数值 backend 重算完整上下文；IncrementalRunner 展示完整多层 decoder 的真实数值 KV 与请求隔离，但缓存是连续数组；TensorKV 展示真实分页数值与 COW，但示范模型比完整 decoder 更小。

实际框架需要把这些责任连接到同一执行路径，还要加上设备生命周期、批量张量、异步完成、分布式 worker、采样元数据与流式输出。现在读 scheduler、model runner、KV manager 和 attention backend 的源码时，你应该能问出具体问题：谁分配页，谁把逻辑页表传给 kernel，谁记录有效长度，何时采样，何时确认设备不再使用缓冲区。

把完整 TinyDecoder 接入 TensorKV 的第一步，是为每层与每个 KV 头建立明确存储维度，并让 append 的输入来自该层投影结果。不能只把所有层返回的 cache 拼成一个不加说明的列表。第二步是让每次查询按该层页表和绝对位置读取；第三步才是与调度计划连接，处理抢占后的重算、前缀复用与取消。

本项目已经提供各责任的完整可执行参照，便于逐项组合。组合时要同时保留完整前向与增量数值对照、页表引用不变量、未来扰动测试与多请求隔离测试。只比较最终一个贪心 token 可能漏掉很大的 logits 差异，因为分数第一名未变时，错误计算仍能产生相同编号。

## 第八阶段：把正确性证据与性能实验接起来

### 29. 先确定你要回答哪一个性能问题

“哪个实现更快”太宽泛。你可能关心单个 softmax 的耗时、完整单请求的首 token 延迟、长上下文逐 token 延迟、多请求总吞吐，或者共享 H100 上一个短实验的资源占用。不同问题需要不同测量边界，不能用一个孤立算子的快慢直接替代服务端到端表现。

若比较 online attention 与 dense attention，先固定数学输入、形状、dtype、mask 和误差容忍度，再固定设备、工具链与编译状态。记录 warmup 是否包含编译，计时是否等待设备完成，测的是单次还是多次平均。CPU wall-clock、GPU event 与服务请求时间回答的是不同范围的问题。

如果你发现优化后一个 kernel 变快，但服务总延迟没有下降，可能因为它原来就只占很小部分，也可能因为新增了布局转换、数据搬运或同步。应把端到端时间分解到实际执行链中，再寻找证据。不要只选择最快的一次样本，也不要在输入规模改变后仍把两个数字直接相除。

### 30. 在你的 PC 与 H100 上怎样分阶段执行

普通 CPU 先完成全部数值与状态实验，RTX 5070 Ti 的兼容软件栈用于单卡正确性与小型 profiling，H100 用于确认相关硬件路径、真实推理框架和需要 Hopper 特性的实验。共享服务器上的多卡任务应在单卡基线清楚后再做，规模、设备和超时都显式指定。

下面是使用课程超时包装器运行一个小型 GPU 正确性实验的形式。设备编号零只是示例，应替换为你实际获准使用的设备。运行前按 K 卷完成依赖与设备检查，本次教材构建没有代替你在这些硬件上执行。

~~~bash
python tools/run_experiment.py --seconds 120 --gpu 0 -- python -m labs.kernels.gpu_check --run
python tools/run_experiment.py --seconds 120 --gpu 0 -- python -m labs.kernels.benchmark --run --op matmul --size 256 --repeats 20
~~~

包装器设置设备可见范围并限制进程运行时间，不是隔离任意不可信代码的沙箱，也不是公司资源调度系统。外部实验若另外启动后台服务或远程作业，还需按对应系统管理其生命周期。对于课程提供的有界本地命令，它帮助你把一次实验的命令、返回码与输出保存在明确记录中。

进入真实 vLLM 或 SGLang 时，先固定框架提交、模型 revision、配置与输入，再跟随 S 卷源码工作坊建立对应调用链。第一次实验应使用能快速加载和验证的小模型，研究一个请求，再增加并发或长上下文。不要把学习“状态如何传递”的目标与“立即装下最大模型”的资源问题绑在一起。

## 第九阶段：十二道实现与诊断练习

### 练习一：证明基线没有修改调用者 prompt

完整解是运行生成前复制 prompt，生成后比较原对象，并分别检查输出长度与总上下文的推导。下面的测试还覆盖生成零个 token；不能因为结果为空就跳过输入是否被修改的检查。

~~~python
from projects.model_runtime import DecoderBackend, direct_generate

prompt = [1, 2]
before = prompt.copy()
result = direct_generate(DecoderBackend(), prompt, 4)
assert prompt == before
assert len(result) == 4
assert direct_generate(DecoderBackend(), prompt, 0) == []
assert prompt == before
~~~

如果把 direct_generate 中的 context=list(prompt) 改成 context=prompt，这项测试应失败。这个故障注入确认测试能够发现目标问题。不要通过把测试的 before 也指向同一列表来“修复”失败，那样两个引用一起被修改，比较会掩盖错误。

### 练习二：从 shape 计算实际 KV 字节

题目是默认两层模型、两个 KV 头、每头四维、五个历史 token、float64。完整计算为二份 K/V乘两层乘两头乘五位置乘四维乘八字节，得到一千二百八十字节。若把 KV 头数改成 query 头数四，会错误地翻倍；GQA 的缓存节省恰好来自这个区别。

用 snapshot 核对这个结果，再把 prompt 长度从五改成七，预测新值一千七百九十二。这个实验不需要训练或 GPU，只检查张量布局与存储单位。若采用另一 dtype，应从真实数组 itemsize 读取每元素字节，不能一直沿用 float64 的八字节。

### 练习三：逐位置比较完整前向与缓存前向

下面是完整参考测试。它比较每个位置的整条词表向量，覆盖的行为比只比较最后生成 token 更细。容差应按当前 dtype 与计算路径选择，不是随意扩大直到测试变绿。

~~~python
import numpy as np
from labs.models.core import TinyDecoder

model = TinyDecoder()
tokens = [1, 3, 2, 7, 4]
expected = model.forward(tokens)
cache = None
actual = []
for token in tokens:
    logits, cache = model.step(token, cache)
    actual.append(logits)
np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
assert [k.shape[1] for k, v in cache] == [len(tokens)] * model.config.layers
~~~

如果结果第一次就不同，先检查参数与输入是否相同，再检查 mask、RoPE 位置与头布局。若前几个位置一致、后来不同，重点看旧缓存拼接、层状态对应和绝对位置增长。按“首次出现差异的位置”缩小范围，比一次打印全部矩阵更有效。

### 练习四：三个请求交错仍应分别正确

给 A、B、C 三个不同 prompt，先各自用直接基线生成，再交给 IncrementalRunner 交错推进。下面的完整实现不硬编码输出编号，因此换一组合法 prompt 后仍可作为验收。

~~~python
from projects.incremental_runtime import IncrementalRunner
from projects.model_runtime import DecoderBackend, direct_generate

runner = IncrementalRunner()
prompts = {"A": [1, 2], "B": [2, 4, 1], "C": [7]}
reference = {}
for rid, prompt in prompts.items():
    reference[rid] = direct_generate(DecoderBackend(runner.model), prompt, 3)
    runner.start(rid, prompt, 3)
runner.run("round_robin")
assert {rid: r.output for rid, r in runner.requests.items()} == reference
~~~

故意使用一份共享 cache 时，应至少选两个不同长度、不同内容的 prompt。若所有请求恰好完全相同，串用状态可能在短样例里不容易显现。测试数据必须打破错误实现依赖的对称性，而不是只选择最方便复制的输入。

### 练习五：取消后新请求不能继承旧状态

完整答案是先推进旧请求一次，取消并确认缓存字节为零，再创建一个新的 rid，使用相同 prompt 生成一个 token，并与无缓存 reference 比较。新 ID 对应新生命周期；现有 runner 故意拒绝复用已经存在的 rid，以免把历史记录与新任务混在一起。

~~~python
import numpy as np
from projects.incremental_runtime import IncrementalRunner

runner = IncrementalRunner()
runner.start("old", [1, 2], 4)
runner.advance("old")
runner.cancel("old")
runner.start("new", [1, 2], 1)
runner.run()
expected = int(np.argmax(runner.model.forward([1, 2])[-1]))
assert runner.requests["new"].output == [expected]
assert runner.snapshot()["old"]["cache_bytes"] == 0
~~~

保留取消前已生成的 output 并不等于保留可继续计算的 cache。前者用于向调用者说明已经发生什么，后者占用资源且可能被误用于继续执行。你应分别检查两个字段，而不是简单要求“取消后对象必须完全空白”。

### 练习六：计算生成上限中的差一

prompt 长度为五，要求生成四个新 token。完整答案是四次预测所依赖的输入长度分别为五、六、七、八，所以最多需要为八个已处理输入位置保留本次继续生成所需状态；最后选出的第九个上下文 token 不必再前向。若要继续第五个输出，就需要处理它。

这不是所有引擎都必须采用的唯一预留方式。某些系统会保守预留更多容量，或者统一为已经输出的 token 建立状态以便续写。关键是把当前实现的实际计算与保留策略说清楚，不能把一个保守上界与一个精确已计算长度当成同一个量。

### 练习七：验证 OnlineDecoder 替换了完整模型路径

完整参考如下。同样参数初始化下比较 full logits，再改变未来输入，检查早期 logits 不变。第一项检查替换等价性，第二项检查因果边界，两者缺一不可。

~~~python
import numpy as np
from labs.models.core import TinyDecoder
from projects.incremental_runtime import OnlineDecoder

dense = TinyDecoder()
online = OnlineDecoder()
tokens = [1, 2, 3, 4, 5]
np.testing.assert_allclose(online.forward(tokens), dense.forward(tokens),
                           rtol=1e-11, atol=1e-11)
changed = [1, 2, 3, 9, 8]
np.testing.assert_allclose(online.forward(tokens)[:3],
                           online.forward(changed)[:3],
                           rtol=1e-11, atol=1e-11)
~~~

如果你只调用一次孤立 online_attention 再打印误差，并没有证明完整 decoder 正在使用它。这个测试直接实例化 OnlineDecoder，并通过正常 forward 进入每层的替换路径。读者还应能沿方法调用解释继承中的 _attn 覆盖如何生效，而不是仅相信类名。

### 练习八：用四个 query 头写出正确映射

两个 KV 头对应四个 query 头时，正确映射为零、零、一、一。完整推导是 group_size=4//2=2，再对 query 头编号整除二。若写成模二，会得到零、一、零、一，这是不同排列，通常无法直接对应同一 checkpoint 的参数布局。

构造两个 KV 头显著不同的数值，再分别按两种映射计算，便能得到反例。若两个 KV 头恰好相同，错误映射会被隐藏。这个练习说明 shape 检查只能证明维度兼容，不能证明元素语义正确；布局测试需要有辨识度的数据。

### 练习九：设计尾块与矩形掩码反例

完整答案包含两组输入。第一组 Q/K 长度七、block=3，最后块只有一个有效 key，检查没有把 padding 当成分数零的有效位置。第二组只有一个 query、三个历史 key，query 处于最后位置，应该读取三个 key；若套用局部 query 行号零的因果规则，只能读取第一个，结果应明显不同。

选择三个不同 value，例如零、十、百，并让分数相同，可以很容易手算：完整可见输出是三者平均，而错误只见第一个输出为零。这个反例把错误放大到肉眼可见，无需依赖极小浮点误差才能发现 mask 语义被改坏。

### 练习十：数值 COW 的验收必须看哪些量

完整答案至少检查共享前后的物理页引用数、两个请求的逻辑 token 序列、旧前缀 K/V 相等、分叉位置 K/V 独立、资源不足时全部旧状态不变，以及释放后页回到空闲池。只验证 table 不同不够，因为新页可能没有复制有效数值；只验证数值相同也不够，因为两个请求可能仍指向同一可写页。

本章第二十六节给出完整可运行 COW 解。进一步扩为跨两页追加时，先手算 needed，再设置刚好够与差一页两个容量条件。两个边界能分别验证分配算术与失败原子性，比随机给一个很大的池只测成功路径更有价值。

### 练习十一：固定随机种子是否足以比较调度策略

在本项目 temperature=0 的贪心基线中，没有抽样随机数分配问题，但真实 logits 仍可能受数值路径影响。若把采样改成非零温度，每个请求应使用独立 RNG 状态，并在恢复时保存它；否则调度顺序改变会改变哪个请求消费哪个随机数，即使 logits 相同也会生成不同结果。

完整答案还应指出：独立 RNG 只解决随机数归属，不解决浮点计算的所有差异。概率接近采样边界时，微小 logits 变化也可能改变选择；argmax 的两个候选几乎同分时，贪心也可能翻转。因此需要分别记录数值误差、随机状态与输出差异，不把所有不一致都归于同一个原因。

### 练习十二：写一份可以复现的性能实验结论

题目给出的八点三五是模拟总时间。完整答案必须注明成本规则与它没有测量硬件这一事实，随后提出真实实验：固定设备与软件版本、模型与参数、输入/输出长度、并发与缓存策略；先正确性验收；区分 warmup、编译与稳定运行；用正确同步计时；报告多次样本及误差范围。

如果只测 kernel，就不要把结果称为端到端服务吞吐；如果测请求端到端，就说明是否包含网络、排队与序列化。一个可信结论应能回答“谁在什么条件下，测量了哪一段过程，观察到了什么”，而不是只给一个最快数字与一个模型名字。

## 最终交付标准

你应提交自己的直接生成基线、逐轮调度表、每请求缓存状态报告、完整模型算子替换对照、一次数值 COW 实验和至少三种故障反例。能关闭参考文档解释 prompt、context、computed、next_logits、output 与 cache 的关系，才算把模型和运行时真正接通。

进一步进入真实框架时，带着这些明确的数据对象去读源码，把每个对象映射到实际版本中的结构、函数和设备张量。先提出一个小而可验证的修改，再用独立 reference 与上游测试约束它。模型变大、设备变多以后，数据与状态关系仍然需要如此清楚；规模扩大不会自动弥补基础理解中的空白。
