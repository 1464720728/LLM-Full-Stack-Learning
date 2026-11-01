# S12 数值、量化与确定性：从一个舍入误差追到一条生成轨迹

模型结构相同、seed 相同、temperature 为零，输出仍可能不同。量化后一个 benchmark 分数下降，也不能立刻说明某个 kernel 错了。本章把浮点表示、归约顺序、量化近似、采样状态、批次变化和任务评测分开，建立从算子到模型再到 Agent 的分层证据。目标是能够定位第一处分歧，而不是反复运行大评测等待平均分“看起来稳定”。

先修 S07 概率与采样、S08 实验设计、矩阵乘法和 attention。所有完整程序使用 NumPy 或课程 CPU 模块。这里的四位量化是透明教学参考，不是某个生产量化格式的替代实现；没有真实 GPU 时仍可完成主要数学与诊断练习。涉及框架确定性开关的事实于 2026-11-01 核查，并明确版本与功能边界。

## 1. 第一课：浮点数保存有限位，实数等式不保证位级相同

计算机不能用有限比特精确表示所有实数。浮点用符号、指数与有效数字表示一定范围内的近似值，数值越大，相邻可表示值的间隔通常也越大。一个很小的增量加到很大的数上，可能在舍入后完全消失。这不是 Python 特有问题，而是有限精度表示的基本性质。

对 float32，先算一亿加负一亿得到零，再加一得到一；先算负一亿加一时，一可能被舍入丢掉，再加一亿得到零。数学实数加法满足结合律，浮点运算一般不严格满足。并行归约改变加法树，就可能改变最后几位，哪怕每个输入元素完全相同。

```python
import numpy as np

a,b,c = np.float32(1e8),np.float32(-1e8),np.float32(1)
left = np.float32(np.float32(a+b)+c)
right = np.float32(a+np.float32(b+c))
assert left == 1 and right == 0
a,b,c = np.float64(1e16),np.float64(-1e16),np.float64(1)
left64 = np.float64(np.float64(a+b)+c)
right64 = np.float64(a+np.float64(b+c))
assert left64 == 1 and right64 == 0
print("float32:",left,right,"float64:",left64,right64)
```

更高精度扩大可准确处理的范围，不会让有限精度重新获得所有实数代数性质。float64 在一亿附近能保留这个一，但到一亿亿附近仍会出现类似现象。选择参考精度时要考虑输入范围，不能把 float64 当作无限精度真值；不过对很多小型教学样本，它足以提供更可靠的比较基线。

常见二进制格式的位数可以帮助建立直觉：float32 通常有二十三个显式小数位与八个指数位，float16 有十个显式小数位与五个指数位，bfloat16 有七个显式小数位与八个指数位。bfloat16 的动态范围与 float32 接近，但精细程度较低；float16 的有效数字更多一些，动态范围却更窄。两者都占两字节，不代表数值行为相同。

实际矩阵乘法还要区分输入存储 dtype、乘法内部精度、累加精度和输出 dtype。输入是 float16，不表示每一步累加都必须 float16；输出是 float32，也不表示乘法完全使用完整 float32 有效位。PyTorch 官方数值精度文档说明不同后端与内部精度路径可能产生差异，不能只看张量 dtype 判断全部计算。[官方数值说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

## 2. 第二课：误差指标先定义，再决定是否通过

绝对误差为候选值与参考值差的绝对值，适合看数值偏离多少。相对误差把绝对误差除以参考值幅度，但参考接近零时会非常大甚至无定义。均方误差概括整体偏差，却可能掩盖少数严重异常；最大绝对误差能发现极端点，但不能独自说明大多数元素的质量。通常需要多项一起看。

allclose 的常见判定形式是 `abs(candidate-reference)<=atol+rtol*abs(reference)`。atol 保护接近零的区域，rtol 随参考尺度放宽。阈值应根据 dtype、运算长度和用途预先设定，不是看见失败后不断放大到通过。某些错误例如跨请求污染，可能在少量样本上恰好小于宽松阈值，因此还需要结构不变量。

shape 必须先相同。NumPy 广播可能让不同 shape 的数组参与比较，甚至在所有值相同时返回通过，但一个 `[2,1]` 输出与期望 `[2]` 并不是同一接口。dtype、有效范围与有限值也应单独检查，避免 NaN/Inf 被后续统计隐藏。比较数值之前先比较“我们是否在比较同一个对象”。

```python
import numpy as np

def error_report(reference,candidate,atol=1e-6,rtol=1e-5):
    ref = np.asarray(reference,dtype=np.float64)
    got = np.asarray(candidate,dtype=np.float64)
    if ref.shape != got.shape or ref.size == 0:
        raise ValueError("same nonempty shape required")
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("nonfinite values require explicit diagnosis")
    difference = np.abs(got-ref)
    threshold = atol+rtol*np.abs(ref)
    return {"max_abs":float(difference.max()),
            "rmse":float(np.sqrt(np.mean(difference**2))),
            "max_scaled":float(np.max(difference/np.maximum(np.abs(ref),1e-12))),
            "allclose":bool(np.all(difference<=threshold)),
            "worst_index":tuple(int(i) for i in np.unravel_index(
                np.argmax(difference),difference.shape))}
r = error_report([0.,1000.],[1e-7,1000.001])
assert r["allclose"] and r["worst_index"] == (1,)
try:
    error_report([[1.],[1.]],[1.,1.])
except ValueError:
    pass
else:
    raise AssertionError("broadcasting hid shape mismatch")
print(r)
```

max_scaled 使用小下限避免除零，应该把这个下限写进定义；它不再是所有位置严格的普通相对误差。worst_index 帮助你回到原张量的层、token、head 或词表位置，而不是只得到一个抽象最大值。对大模型可以保存摘要与最坏点，避免把全部私有激活写到日志。

当误差超过阈值时，不要直接断言实现错误。先检查参考精度、输入是否相同、mask 与布局是否一致；如果数学目标相同，再研究舍入路径。相反，“只是浮点误差”也不能成为解释任何差异的借口。你需要量级、结构与可复现输入来支持判断。


### 2.1 softmax 的稳定写法解决溢出，不解决所有数值问题

对 logits 一千和九百九十九直接求指数，float32 很容易溢出；先减最大值得到零和负一，再求指数就能得到有限权重。两个过程在实数中等价，在有限精度中前者可能得到无穷除无穷，后者正常。因此“数学等价”并不意味着实现对异常范围同样稳健。

但稳定 softmax 不会让极小概率永远非零。一个远低于最大值的分数，指数可能下溢到零；若它本来对总和贡献极小，这可能是可接受近似。若所有候选先被 mask 成负无穷，减最大值仍然没有合法分布，必须明确报错。溢出、下溢与全屏蔽是三种不同问题，不能统一用加一个 epsilon 解决。

```python
import numpy as np

scores = np.array([1000.,999.],dtype=np.float32)
with np.errstate(over="ignore",invalid="ignore"):
    naive = np.exp(scores)
    naive /= naive.sum()
stable = np.exp(scores-scores.max())
stable /= stable.sum()
assert not np.isfinite(naive).all()
assert np.isfinite(stable).all()
np.testing.assert_allclose(stable,[0.7310586,0.2689414],atol=1e-6)
assert np.exp(np.float32(-200)) == 0
print("stable softmax:",stable.tolist(),"naive:",naive.tolist())
```

累加精度也影响分母。很多小权重相加时，低精度累加可能丢失部分贡献；分块 online softmax 则通过每块最大值与归一化统计减少中间存储，但归约顺序与完整矩阵不同。正确目标是满足明确误差与 mask 语义，不是要求所有实现使用相同临时矩阵。

attention 中还有 scale、位置编码与掩码交互。若在错误位置把 scale 应用两次，输出可能仍有限、概率和仍为一，但数学已经改变；若把 mask 后的大负数当成普通可学习分数量化，禁止位置可能重新获得非零权重。数值稳定性与语义正确性需要分别检查，单纯“没有 NaN”不是充分验收。
## 3. 第三课：tiny logit 差异怎样改变贪心输出

设两个最高 logits 为一和零点九九九九，第一项领先万分之一。若第一项减少万分之一、第二项增加万分之一，第二项就成为最大值。temperature=0 没有随机采样，但 argmax 仍会变化。seed 对这个例子没有影响，因为没有消耗随机数。

```python
import numpy as np
baseline = np.array([1.0000,0.9999])
changed = baseline+np.array([-0.0001,0.0001])
assert baseline.argmax() == 0 and changed.argmax() == 1
margin = float(np.sort(baseline)[-1]-np.sort(baseline)[-2])
max_error = float(np.max(np.abs(changed-baseline)))
print({"margin":margin,"max_error":max_error,
       "baseline_token":int(baseline.argmax()),
       "changed_token":int(changed.argmax())})
```

如果每个 logit 的绝对误差最多 ε，而基线第一名比第二名领先超过二倍 ε，那么在这个误差界下第一名不会被第二名翻转。因为最坏情况是第一名下降 ε、第二名上升 ε。若 margin 小于或等于二倍 ε，则可能翻转，但并不保证一定翻转。这个简单界让“误差很小”与“token 是否稳定”建立了具体关系。

一旦某一步 token 不同，自回归后续输入也不同。第十步的巨大 logits 差异可能只是第一步小翻转后的自然结果，不是第十层或第十轮突然数值崩溃。定位应该回到最早的输入仍相同但输出开始不同的位置，再比较 logits 与采样。直接对两段完全分叉文本逐字 diff，无法定位最初原因。

任务答案仍可能相同。例如两条推理措辞不同，最后都输出正确数字；也可能一个微小早期变化导致工具调用参数不同并使任务失败。因此逐位相等、token 相同、文本相似和任务成功是不同层次的标准。评测要同时保留与你的目标相关的层次，不强求一个指标解释全部行为。

## 4. 第四课：手写对称量化，逐步算整数与 scale

量化把连续或高精度数值映射到有限整数集合，再通过 scale 等参数近似恢复。对称四位教学格式使用整数负七到七，scale 等于最大绝对值除七。这里故意不用负八到七，以保持正负对称；真实四位格式可能采用其他编码、分组和偏置，必须查看具体实现。

向量为 `[-2,-0.3,0.1,0.8,1.7]`，最大绝对值二，scale 为二除七。每项除 scale 后约为负七、负一点零五、零点三五、二点八、五点九五，四舍五入到最近整数得到负七、负一、零、三、六。再乘 scale 得到近似数，零点一被量成零，最大绝对误差零点一。

| 原值 | 除 scale | 量化整数 | 反量化 | 绝对误差 |
|---:|---:|---:|---:|---:|
| -2 | -7 | -7 | -2 | 0 |
| -0.3 | -1.05 | -1 | 约-0.285714 | 约0.014286 |
| 0.1 | 0.35 | 0 | 0 | 0.1 |
| 0.8 | 2.8 | 3 | 约0.857143 | 约0.057143 |
| 1.7 | 5.95 | 6 | 约1.714286 | 约0.014286 |

本地 [symmetric_quantize](../../labs/serving/analysis.py) 先检查位宽、非空与有限值，计算 qmax，得到 scale；全零时返回全零与 scale 一，避免除零。之后使用 Python round，再限制到整数范围，最后反量化。Python round 在恰好半整数时采用其明确的舍入规则，跨实现逐位比较时应确认是否一致。

```python
from labs.serving.analysis import symmetric_quantize

values = [-2.,-.3,.1,.8,1.7]
q,scale,reconstructed = symmetric_quantize(values,4)
assert q == [-7,-1,0,3,6]
error = max(abs(a-b) for a,b in zip(values,reconstructed))
assert error <= scale/2+1e-12
assert symmetric_quantize([0.,0.],4) == ([0,0],1.0,[0.,0.])
print({"q":q,"scale":scale,"reconstructed":reconstructed,"max_abs":error})
```

没有裁剪溢出且使用最近整数舍入时，单元素误差理想上不超过 scale 的一半。若 scale 是根据某个校准集确定，而新输入超出范围，饱和裁剪误差可以远大于这个界。教学函数用当前输入最大值确定 scale，因此例子满足范围假设；不能把这个误差界直接套到所有动态激活。

量化存储的整数不是模型的新“真实值”，只有结合 scale 才能解释。忘记 scale、使用错误 group 的 scale 或把符号编码读错，都属于实现错误，而不是正常量化损失。先验证编码/解码契约，再讨论近似是否可接受。

## 5. 第五课：离群值、分组与矩阵误差怎样联系

若一整组数包含一百与零点零一，用最大值决定四位 scale，大约十四点二九，小值很容易量成零。若把小幅值的一组单独量化，它们可以使用更小 scale，保留更多相对细节。这是 per-tensor、per-channel 或 group-wise 量化差异的直觉，但更多 scale 也增加元数据、加载复杂度和 kernel 约束。

分组边界必须与权重布局一致。数学矩阵的一行可能对应一个输出通道，但实际存储可能转置或打包；“每行 scale”必须说明是哪一个语义轴。若 TP 分片后每 rank 重算 scale，与先全局量化再切片可能得到不同结果。比较量化模型时，应记录量化发生在何时、按哪一维分组。

矩阵输出误差可以写成 `ΔY=XΔW`，假设只量化权重且输入相同。某个输出元素的绝对误差不超过各输入绝对值乘相应权重误差的和。即使每个权重误差很小，很多项累加或输入幅度很大时，输出误差仍可能明显。权重最大误差本身不足以证明模型输出无损。

如果激活也量化，则近似输出为 `(X+ΔX)(W+ΔW)`，误差包含 XΔW、ΔXW 与 ΔXΔW。还可能有累加与输出舍入误差。把所有差异归因给“权重四位”会漏掉激活范围、累加 dtype 和后处理。W4A16 等标签只描述部分精度配置，不唯一指定算法。

```python
import numpy as np
from labs.serving.analysis import symmetric_quantize

x = np.array([[2.,-3.]])
w = np.array([[.2,1.7],[-.4,.1]])
q,scale,deq = symmetric_quantize(w.ravel().tolist(),4)
wq = np.array(deq).reshape(w.shape)
difference = np.abs(x@wq-x@w)
bound = np.abs(x)@np.abs(wq-w)
assert np.all(difference <= bound+1e-12)
print({"output_error":difference.tolist(),"elementwise_bound":bound.tolist()})
```

这个界通常偏保守，因为正负误差可能抵消；它的价值是解释误差传播方向，不是预测所有层的最终质量。深层网络还有归一化、非线性和 attention 概率变化，误差不一定简单线性累加。应该用层级测量验证，而不是把单层最坏界乘层数就宣称任务一定失败。


### 5.1 用两行矩阵观察 per-tensor 与 per-row

下面矩阵第一行是百分之一量级，第二行是一百量级。全矩阵共用 scale 时，小行几乎失去信息；每行独立 scale 可以更准确保留小行。示例只比较数值，没有打包四位整数，也没有宣称某个实际 kernel 支持这种分组。真正显存收益还要计算 scale 元数据和存储格式。

```python
import numpy as np
from labs.serving.analysis import symmetric_quantize

w = np.array([[.01,.02],[100.,50.]])
_,global_scale,flat = symmetric_quantize(w.ravel().tolist(),4)
global_reconstruction = np.array(flat).reshape(w.shape)
row_scales = []
rows = []
for row in w:
    _,scale,restored = symmetric_quantize(row.tolist(),4)
    row_scales.append(scale); rows.append(restored)
row_reconstruction = np.array(rows)
global_small_error = np.max(np.abs(global_reconstruction[0]-w[0]))
row_small_error = np.max(np.abs(row_reconstruction[0]-w[0]))
assert row_small_error < global_small_error
print({"global_scale":global_scale,"row_scales":row_scales,
       "global_small_error":float(global_small_error),
       "row_small_error":float(row_small_error)})
```

第一行改进明显，不代表全模型任务一定改善同样比例。第二行仍然可能主导最大误差，某些下游输入又可能对第一行特别敏感。量化评估应同时看局部误差分布与模型输出，而不是只追一个全局 max。分组策略也可能增加读取 scale 的带宽，数值收益与性能成本需要一起测量。

当前 symmetric_quantize 的核心表达式是先算 qmax，再用最大绝对值得到 scale，逐项 round 并 clamp。clamp 用于保证整数落在格式范围内，round 决定最近网格点。若输入都由同一个 maxabs 决定范围，理论上极值落在端点；浮点计算仍可能出现微小舍入，因此限制范围保留了清楚的输出契约。

```python
# analysis.py 中 symmetric_quantize 的核心节选，依赖已校验的 values/bits。
qmax = (1 << (bits-1))-1
scale = max(abs(x) for x in values)/qmax
if scale == 0:
    return [0]*len(values), 1.0, [0.0]*len(values)
quant = [max(-qmax, min(qmax, round(x/scale))) for x in values]
return quant, scale, [q*scale for q in quant]
```

位移表达式一左移 bits减一，得到二的 bits减一次方，再减一形成正端点。四位得到七，八位得到一百二十七。这个实现没有 zero-point，零恰好映射为整数零；非对称量化会增加偏置，可能更适合不对称范围，但也改变编码、解码和矩阵乘法修正项。

程序返回 quant、scale 与 reconstructed 三项，便于你同时检查整数是否合法、元数据是否合理、数值误差多大。只看反量化结果会漏掉编码格式问题，只看整数则不知道它们对应什么尺度。以后读取真实量化 checkpoint，也应分别追这三类信息如何进入 kernel。
## 6. 第六课：至少区分五种复现目标

随机流可复现表示相同 RNG 算法、seed 与消费顺序产生相同随机样本。固定路径可复现表示执行使用相同算子、shape 与归约路径。位级确定表示输出字节完全相同。批次不变性表示同一请求在不同同批请求组合下保持所定义的一致结果。统计可复现表示重复实验的分布或结论稳定。这些目标相关，但强度与用途不同。

同一个请求独立运行两次相同，不证明与其他请求拼 batch 后也相同；同一硬件与版本相同，不证明跨架构或升级后逐位相同；logits 在容差内，不保证贪心 token 不翻转；最终任务正确率相近，不证明每层数值相同。报告应明确具体保证，不使用一个笼统的 deterministic 标签覆盖全部。

批次改变可能让 kernel 选择不同 tile、拆分或归约顺序，即使请求之间在数学上完全隔离，也可能出现小数值差异。这与 S06 的跨请求 mask 错误不同：前者是同一数学问题的数值路径变化，后者直接让请求看到别人的内容。差异很大或随其他请求内容系统性变化时，先检查隔离，不能立即归因于正常舍入。

缓存冷热也可能改变计算路径。完整 prefill 与缓存后 decode 使用不同 query/key shape 或不同 kernel；只要数学目标相同，数值应在合理范围，但逐位相同需要额外机制。分块 prefill、prefix hit 和 online softmax 的分块顺序都应进入确定性实验矩阵。

## 7. 第七课：框架确定性开关有支持边界

vLLM 官方[复现说明](https://docs.vllm.ai/en/latest/usage/reproducibility/)区分 seed、调度确定性与 batch invariance，并将保证限制在相同硬件与版本等条件。在线请求到达会改变调度，仅设置 seed 不能自动冻结执行路径。[batch invariance 文档](https://docs.vllm.ai/en/latest/features/batch_invariance/)介绍相关机制与支持范围；使用时应匹配你安装版本，而不是只复制滚动页面上的变量名。

SGLang 固定快照的[确定性文档](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/docs/docs/advanced_features/deterministic_inference.mdx)提供 enable-deterministic-inference，并列出 FlashInfer、FA3、Triton 后端的功能矩阵。该快照矩阵中 FlashInfer 的 radix cache 组合标为不支持，FA3 与 Triton 的对应项标为支持。这个具体差异说明“有开关”不等于任意后端与缓存组合都获得同一保证。

文档展示非贪心采样与 sampling_seed，但不同 seed 不保证每次输出一定不同。两个随机流可能在某次抽样落到同一 token，分布很尖锐时更常见。正确说法是不同 seed 改变随机序列，可能产生多样结果；相同 seed 的复现还依赖对应模式、数值路径与输入一致。不要把示例性描述扩大成数学保证。

确定性模式可能改变 kernel、归约与调度选择，性能和支持模型范围也可能变化。先用小输入验证功能，再测实际成本，不应默认“确定性免费”或“必然很慢”。本章没有运行真实 GPU 确定性测试，提供的是官方机制核查与实验设计，不能把文档预期写成我们已测得结果。

## 8. 第八课：teacher forcing 让两模型始终回答同一道题

自由生成时，两个模型一旦选出不同 token，后续上下文就不同。teacher forcing 则给两者同一个固定 token 序列，在每个位置比较基于相同历史得到的 logits 或目标 token logprob。它能把数值差异与输入轨迹分叉分开，是定位量化和后端差异的重要工具。

设固定序列为一、二、三、四。比较前缀一时的下一 token 分布，再比较前缀一二，再比较一二三。两个模型始终读取相同 token，而不使用各自采样结果作为后续输入。若第一位置已不同，就从该位置的层级激活开始追踪；若所有 teacher-forced logits 很接近而自由生成分叉，再检查 margin 与采样。

下面构造同 seed 的两个 TinyAttentionBackend，将第二个的参数用本地四位参考量化后反量化，逐前缀比较。它不是真实大模型量化评测，但代码完整展示了实验层级。没有断言“必须生成不同 token”，因为数值改变不保证 argmax 改变。

```python
import numpy as np
from labs.serving.backend import TinyAttentionBackend
from labs.serving.analysis import symmetric_quantize

reference = TinyAttentionBackend(seed=11)
candidate = TinyAttentionBackend(seed=11)
for name in ("embedding","q","k","v","out"):
    weight = getattr(candidate,name)
    _,_,dequantized = symmetric_quantize(weight.ravel().tolist(),4)
    setattr(candidate,name,np.array(dequantized).reshape(weight.shape))
sequence = [1,2,3,4,5]
rows = []
for end in range(1,len(sequence)+1):
    prefix = sequence[:end]
    a = np.array(reference.logits(prefix))
    b = np.array(candidate.logits(prefix))
    assert a.shape == b.shape and np.isfinite(b).all()
    sorted_a = np.sort(a)
    rows.append({"prefix_length":end,
                 "max_abs":float(np.max(np.abs(a-b))),
                 "top1_reference":int(a.argmax()),
                 "top1_candidate":int(b.argmax()),
                 "reference_margin":float(sorted_a[-1]-sorted_a[-2])})
def greedy(model,prompt,steps):
    context = list(prompt)
    result = []
    for _ in range(steps):
        token = int(np.argmax(model.logits(context)))
        result.append(token); context.append(token)
    return result
print("teacher-forced:",rows)
print("free generation:",greedy(reference,[1,2],6),
      greedy(candidate,[1,2],6))
```

两个对象分别初始化，避免修改 candidate 时意外修改 reference 共享数组。若使用浅复制，某些参数可能仍别名共享，导致“量化前后完全一致”的假结果。实验前检查对象与数组所有权，和 S01 请求列表隔离是同一个工程习惯。

逐前缀完整重算很慢，但对小模型和短序列适合作为参考。真实大模型可以一次前向取多个位置 logits，或用完整 KV 路径提高效率，不过比较时仍要确保相同历史与相同位置。性能优化不应改变 teacher-forcing 的实验定义。

## 9. 第九课：逐层定位第一处分歧，而不是只看最后文本

把模型分成 embedding、归一化、QKV 投影、位置编码、attention、输出投影、残差、MLP 和 lm_head 等检查点。先比较输入 token 与 embedding；若相同，再往后。找到第一个超过约定容差的中间量，才有方向判断是权重加载、位置、缓存、mask、归约还是量化造成。

不要在两个模型已经接收不同 token 后继续把每层差异归因给同一个 kernel。先固定 teacher-forcing 输入，或把两者都恢复到分叉前的共同历史。若错误只在第二个 chunk 出现，优先检查位置与缓存；若只在多请求 batch 出现，检查边界与 batch 相关数值路径；若只在 TP>1 出现，检查分片、bias 与 collective。

检查点应保存 shape、dtype、有限值、最大误差和最坏索引，而不一定保存全部张量。对共享环境，尽量使用合成 token 与小模型，避免长期占用 GPU 或导出敏感激活。若必须捕获大张量，先限制层数、位置与样本数，并明确临时文件的生命周期。

数值二分定位可以先比较模型中间层与最终层，缩小到某一半，再逐层检查。但只有当两边输入与模式相同、检查点语义相同，这种方法才有效。融合后端可能没有完全相同中间张量，需要选择数学等价边界，例如整个 attention block 输出，而不是强迫每个内部临时数组逐位对齐。


### 9.1 把“第一处分歧”变成完整的小函数

下面函数比较两个按相同顺序保存的检查点序列，每项由名称与数组组成。先检查名称和 shape，再检查有限值，最后使用预先给定容差；遇到第一处差异就返回证据。它不尝试根据错误大小猜根因，只负责给下一步源码排查一个可靠起点。

```python
import numpy as np

def first_difference(reference,candidate,atol=1e-6,rtol=1e-5):
    if len(reference) != len(candidate):
        raise ValueError("checkpoint counts differ")
    for index,((name,a),(other,b)) in enumerate(zip(reference,candidate)):
        a,b = np.asarray(a),np.asarray(b)
        if name != other or a.shape != b.shape:
            return {"index":index,"name":name,"reason":"identity-or-shape"}
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return {"index":index,"name":name,"reason":"nonfinite"}
        if not np.allclose(a,b,atol=atol,rtol=rtol):
            return {"index":index,"name":name,"reason":"numeric",
                    "max_abs":float(np.max(np.abs(a-b)))}
    return None

base = [("embedding",np.ones(2)),("qkv",np.array([2.,3.])),
        ("attention",np.array([4.,5.]))]
changed = [("embedding",np.ones(2)),("qkv",np.array([2.,3.01])),
           ("attention",np.array([4.1,5.2]))]
result = first_difference(base,changed)
assert result["index"] == 1 and result["name"] == "qkv"
assert first_difference(base,base) is None
print(result)
```

如果 attention 差异更大，但 qkv 已经首先偏离，应先检查投影权重、输入与量化，而不是直接修改 attention kernel。若名称或 shape 不同，先对齐检查点语义，不要继续计算一个没有意义的 max error。函数只在每项对应同一数学量时才提供有效结论。

真实系统可以把这个模式扩展为层级摘要，先每隔几层保存一个点，定位区间后再加密检查。对于异步设备张量，采集摘要本身可能引入同步和性能扰动，因此诊断运行与正式性能运行应分开。不要一边开启大量逐层 CPU 拷贝，一边把测得吞吐当正常服务表现。

容差也应在记录中保存，未来升级后才能复算。若每次随手改 atol，最终“都通过”没有一致含义。对整数索引、页表与请求身份应精确相等；对浮点计算才使用合理容差。不要用数值容差原谅状态机错误，这两类对象的正确性标准完全不同。
## 10. 第十课：从模型到 Agent，环境也会改变输入

Agent 每一步将工具观察加入上下文。文件遍历顺序、时间戳、进程输出、搜索结果、网络状态和工具版本都可能改变下一轮 prompt。即使模型在相同输入下完全确定，环境变化仍会导致轨迹不同。因此 Agent 复现需要记录模型条件与环境条件，不能只设置 sampling seed。

录制工具观察并回放，可以让两个模型接收相同环境消息，帮助隔离模型差异。但回放不代表真实交互成功率，因为模型可能选择不同动作，旧观察未必对应新动作。回放适合固定轨迹上的诊断，真实任务评测仍要在一致规则的环境中执行并由 verifier 判断。

框架/评测 harness 的参数落点也必须核查。一个外层工具的 model 参数不一定传到独立 CLI agent；不同 adapter 可能使用自己的端点、模板与采样配置。S09 的固定 EvalScope 来源与 Agent 卷会继续追调用链。对比量化前后若实际调用了不同服务或默认模型，所有数值分析都失去基础。

任务失败应分类：模型选错、工具调用格式错、权限/环境错误、超时、verifier 差异、服务数值错误等。总成功率只是汇总结果，不能直接定位量化损失。保留每题/每任务的配对结果和失败类型，才能决定下一步该查模型、框架还是环境。

## 11. 第十一课：小分数变化需要配对证据

一百九十八道题中，一题改变约零点五零五个百分点。若基线和量化模型只差一题，单次总分不能告诉你差异是否稳定，也不能说明全部任务无损或明显退化。应对同一题、同一输入与预算记录两者结果，形成“都对、都错、基线对候选错、基线错候选对”四格表。

假设基线对候选错三题，基线错候选对两题，净差只有一题，但总共有五题发生变化。只看总准确率会掩盖这些题的类型与敏感性。应进一步检查它们的 logits margin、输出解析、任务难度与是否有随机采样。配对表帮助找到值得诊断的样本，不自动证明统计显著。

```python
def paired_outcomes(reference,candidate):
    if len(reference) != len(candidate):
        raise ValueError("paired tasks required")
    counts = {"both_correct":0,"both_wrong":0,
              "reference_only":0,"candidate_only":0}
    for a,b in zip(reference,candidate):
        if type(a) is not bool or type(b) is not bool:
            raise ValueError("explicit correctness booleans required")
        key = ("both_correct" if a and b else "reference_only" if a
               else "candidate_only" if b else "both_wrong")
        counts[key] += 1
    return counts
r = paired_outcomes([True,True,False,False],
                    [True,False,True,False])
assert all(value == 1 for value in r.values())
print(r,"one question out of 198 =",100/198,"percentage points")
```

多次运行还要区分平均成功率、pass@k、best-of-k 与多数投票。八次里挑最好结果，与八次分数平均回答不同问题；不能都写成“重复八次验证”。Agent 的工具预算、最大步骤、环境镜像和 verifier 也要一致，否则更多尝试可能只是花了更多资源。

如果目标是发现明显数值回归，先用小而敏感的固定样本和分层日志，不必立刻跑数百个昂贵任务。等算子、模型与协议正确，再扩大任务评测来衡量真实质量。这样既节省共享 GPU/API 成本，也让失败更容易解释。


### 11.1 从一个布尔结果升级为可归因的实验矩阵

设计确定性实验时，先写清楚研究对象。假设研究“目标请求的输出是否受旁边请求影响”，目标请求的 token IDs、采样参数、请求级 seed 和模型权重必须固定。唯一变化是同批邻居。第一次让目标独跑，第二次加入长度很短的邻居，第三次加入较长邻居，第四次交换提交顺序。每次都记录目标请求第一步 logits 的误差和输出 token，而不是把整个批次的字符串直接比较。否则邻居本来就不同，比较失败并没有回答问题。

研究缓存路径时，另建一组实验：关闭缓存的完整 prefill、开启缓存但首次冷访问、已发布相同前缀的热访问。三者必须拥有相同有效 token 和相同位置。记录实际命中的 token 数，防止自以为命中而实际因为 namespace 不同始终冷跑。若首步不同，先固定生成历史检查 logits；若首步相同但后续不同，继续逐步找最早分歧。不要一次同时改变量化位宽、并发、缓存、attention backend 和 TP 数，那样即使发现变化也无法归因。

| 实验轴 | 固定什么 | 改变什么 | 至少保留的证据 |
|---|---|---|---|
| 随机流 | 固定 logits 和采样参数 | 请求交错顺序 | 每请求抽样序列与 seed |
| 批次路径 | 相同目标 token IDs | 邻居及目标所在行 | 目标 logits、margin、批次形状 |
| 缓存路径 | 相同 token 和位置 | 冷、热、完整重算 | 命中边界、页身份、首处分歧 |
| 量化误差 | 相同输入与计算顺序 | 权重表示与分组 | scale、反量化权重、层级误差 |
| 任务质量 | 相同题目和预算 | 已验证的模型候选 | 每题配对结果与失败分类 |

这些轴可以组合，但组合应建立在单轴实验之后。例如先确认缓存正确、量化参考正确，再观察两者一起开启是否暴露格式不匹配。组合失败往往来自接口契约：缓存存的是哪种 dtype，scale 是否随页迁移，backend 是否把反量化后的值当作压缩值再反量化。仅增加随机测试数量不能替代对接口的明确检查。

报告还需要保留一个“负控制”：人为把某个位置偏移一，或者交换两个 KV head，应使诊断器失败。如果刻意破坏后报告仍然全部通过，很可能比较了同一个对象、没有执行候选路径，或者只比较了最终相同的 argmax 而忽略 logits。正控制则是对同一个确定的 CPU 参考重复执行，确认记录与比较工具自身没有制造差异。负控制在独立临时对象上完成，不能把损坏权重写回参考模型。

### 11.2 假量化的数组大小不能充当压缩收益

本章量化函数返回 Python 整数列表、scale 和浮点反量化值。它教的是映射与误差，不是压缩文件格式。Python 的一个小整数对象通常远大于四个 bit；NumPy 的 int8 数组也会为每个元素占一个 byte。把数值限制在负七到七，并不会自动把两个值塞进一个 byte。因此只看数值范围就宣称权重内存减少八倍，是混淆了编码范围与物理存储。

真正的四位打包还要规定高低半字节顺序、符号映射、奇数元素的尾部填充、group 边界、scale 的 dtype、零点、对齐与加载布局。例如一千个四位值理想载荷五百 byte；若每三十二个值一组，共三十二组，每组一个两 byte scale，还要增加六十四 byte，尚未包括对齐与其他元数据。压缩率应以总实际字节计算，不能只算载荷。

反量化到 float32 再调用普通矩阵乘法，适合作为正确性 oracle，却可能在运行时保留压缩权重和浮点副本两份内存。若每个请求都先反量化整个模型，还会增加大量读写工作。这与支持压缩权重直接参与高效 kernel 的生产路径不同。基准中必须说明量化发生在离线转换、加载时还是每次 forward 内；三种计时范围回答不同问题。

KV 量化还多出写入与读取的生命周期问题。新 token 的 K/V 刚产生时要用哪组 scale；一个页逐渐填满时是否允许重新估计 scale；若重新量化旧位置，已经发布的共享页能否被修改；跨机迁移是否同时带上 scale。这些都不是静态权重量化函数可以自动解决的。先保持页内 scale 约定明确，再讨论更细粒度或动态策略，才能让 S03 的共享引用与本章数值误差约束同时成立。

最后，把三种结果分开存放：编码正确性、模型数值误差、实际性能。编码正确性可以用穷举小整数和往返测试证明特定映射；模型误差需要输入分布与任务证据；性能需要真实运行环境、内存统计和计时边界。某个维度通过不会替另两个维度作保证，这正是框架开发者读量化实现时必须保持的分层思维。

## 12. 第十二课：故障、完整练习与数值验收项目

**故障一：只有缓存命中时文本不同。** 固定 token 输入和贪心，分别比较冷 prefill、热前缀与逐 token 路径的 logits。先检查 computed 边界、位置与 KV 身份，再看 backend 是否换了归约路径。若误差巨大或某层突然出现 NaN，不应直接接受为正常确定性差异。

**故障二：量化后全是同一个 token。** 检查 scale 是否为零、是否把所有小权重截成零、是否忘记反量化、是否误用另一个 group 的 scale，以及 logits 是否全相等。全零输入的特殊处理与非零离群值情况不同，不能用 scale=1 对所有问题一刀切。

**故障三：同 seed 在不同并发下变化。** 先用固定 logits 验证请求级 RNG，再用固定输入比较模型 logits。若前者已不同，修随机流归属；若前者一致后者不同，研究 batch、cache 与 kernel 路径。把两层问题分开，避免反复设置同一个 seed 却没有触及根因。

**故障四：任务分数下降但单层误差很小。** 检查敏感 margin、自由生成轨迹、输出解析和环境差异，再做配对样本分析。小局部误差可能在自回归中放大，也可能根本不是模型问题。不能用单层 allclose 通过就保证任务质量不变。

**练习一，结合律。** 用 float32 的一亿、负一亿、一得到两种结果一与零；float64 用一亿亿构造同类例子。答案是有效数字有限与舍入顺序不同，不是某种随机数。提高精度减轻但不消灭该性质。

**练习二，相对误差。** 参考零、候选十的负八次方，绝对误差很小，普通相对误差无定义。需要 atol 或明确分母下限。答案不能把无穷相对误差直接解释成模型完全损坏，应结合尺度与用途判断。

**练习三，margin 稳定界。** 每 logit 误差最多千分之一，第一名领先百分之一，则领先大于二倍误差，argmax 在该界下稳定；领先万分之一则可能翻转。这个界只针对同一输入的该步 logits，不保证未来所有步骤稳定。

**练习四，四位范围。** 本课程使用负七到七，qmax 七；scale 为 maxabs/七。真实负八到七或其他非均匀格式不能直接套同一编码。记录格式、scale、分组和舍入规则，才有可复现实验。

**练习五，全零向量。** 返回全零整数、scale 一与全零反量化，避免除零。这个 scale 是约定，不表示原值范围为一。若随后拼接多组，必须保留每组自己的 scale，不能把零组的约定覆盖其他组。

**练习六，离群值。** 一百与零点零一共用四位 scale，后者很可能量成零。分组可缩小局部范围但增加元数据与 kernel 复杂度。答案应同时说明收益与代价，不能只说“group 越小越好”。

**练习七，矩阵误差。** 只量化 W 时 ΔY=XΔW，元素绝对误差受 abs(X)@abs(ΔW) 上界约束。输入也量化时还有 ΔXW 与 ΔXΔW。只报告权重最大误差不能充分说明输出质量。

**练习八，teacher forcing。** 两模型每步接收相同固定历史，而不是各自生成的历史。它隔离模型数值差异；自由生成评估实际轨迹。两个实验都需要，结论不能互相替代。

**练习九，浅复制陷阱。** candidate 与 reference 若共享权重数组，量化原地修改会同时改变两者，差异可能为零。参考解法独立初始化或显式复制每个数组，检查对象别名。实验对象隔离是数值结论的前提。

**练习十，batch invariance。** 单独重复相同不证明不同 batch 相同。至少测试独跑、与短请求同批、与长请求同批、请求重排、缓存冷热和不同 chunk。记录 logits、token 与任务结果的不同层次，不只记一个 equal 布尔值。

**练习十一，框架开关。** 使用安装版本的支持矩阵，确认模型、attention backend、缓存与并行组合。SGLang 固定文档中 FlashInfer/radix 与其他后端存在差异，不能忽略。未运行 GPU 时只写“文档核查”，不写“保证已验证”。

**练习十二，配对分数。** 基线独对三题、候选独对两题，净差一题，实际五题改变。应检查这五题及失败类别，而不只比较总分。统计结论需要重复与采样假设，不能凭一题差距宣称普遍优劣。

**练习十三，Agent 回放。** 固定工具观察适合隔离模型在同一轨迹上的行为，但新模型动作不同后，旧观察未必有效。回放不是完整真实交互评测。报告要区分诊断轨迹与任务成功率。

**练习十四，NaN。** 首先定位最早非有限中间量，检查输入、mask、scale、溢出与全屏蔽行；不要把 NaN 自动转零再继续生成。那会把明确数值故障变成难以察觉的质量错误，并可能掩盖缓存或布局问题。

项目第一阶段实现误差报告与四位参考，完成手算、全零、离群值、shape 和非有限输入检查。第二阶段对 TinyAttentionBackend 做 teacher forcing 与自由生成，记录 margin 与最早分歧。第三阶段对完整 TinyDecoder 的连续/增量/分页路径做同输入比较，把缓存索引与数值误差分开。

第四阶段在独立真实框架环境中固定小模型 revision，逐项改变 batch、chunk、prefix cache、backend 与并行，记录支持范围与结果。第五阶段才做任务配对评测，明确工具预算与环境。每次实验限制样本、输出长度和总时长，不需要为了确认一个索引错误长时间占用八卡 H100。

掌握门槛：能区分舍入、量化、随机流、批次变化与环境变化；能手算并实现量化参考；能建立误差指标与 margin 解释；能用 teacher forcing 找第一处分歧；能准确说明框架确定性保证的版本边界。下一章进入 [S13 容量估算](13-capacity.md)，把 dtype、KV 与分片落实到每 rank 的资源账本。
