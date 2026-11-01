# M11　状态空间模型：从微分方程到 selective scan 的完整实现

[本卷导航](README.md) · 前置：[线性注意力与 Delta 记忆](10-linear-attention.md) · 下一章：[混合与深度残差](12-hybrid-residuals.md)

这一章从一个具体问题开始：如果不保留全部历史 token，而是维护一组随输入演化的内部量，怎样定义写入、保留与读出？状态空间模型给出一套明确的方程。你会先手算一阶滤波，再扩展到多状态，推导递推与卷积等价，解释连续时间如何离散化，最后实现输入相关的选择性扫描与一个完整教学 block。

SSM 不是 attention 的另一个名字。它从状态动力学出发；attention 从查询对历史记录的加权读取出发。二者可以组合，也能在某些形式下建立联系，但不能因为都有 state 就把公式互换。本章所有核心实验使用 NumPy 和 CPU；涉及 Mamba 时对照本卷已固定版本的官方 reference，不宣称已经编译或运行其 GPU 内核。

## 第一课　状态、输入与输出：先把三个对象分开

设时间步输入为 u，内部状态为 h，输出为 y。最简单离散系统写成 $h_t=ah_{t-1}+bu_t$，$y_t=ch_t+du_t$。a 控制旧状态保留，b 控制当前输入写入，c 控制状态怎样被读取，d 是输入直接通向输出的支路。状态不必与输出相同，也不必与输入维度相同。把它们都叫“隐藏向量”会掩盖不同职责。

取 a 为二分之一，b 与 c 为一，d 为零，初始状态零。输入序列一、零、零、零。第一步旧状态零，写入一，状态和输出为一；第二步没有新写入，旧状态乘二分之一，输出二分之一；第三步四分之一；第四步八分之一。先前输入在后续仍有影响，这就是一个可计算的记忆，而不是神秘的“模型记住了”。

若输入一直为一，状态依次一、一点五、一点七五、一点八七五，逐渐接近二。每步既保留过去的一半，又加入新的一。它不是立刻输出二，因为达到平衡需要时间。这个系统同时体现瞬态与稳态：瞬态描述从初始状态出发怎样变化，稳态描述长期恒定输入下接近哪里。

```python
import numpy as np
from labs.models.core import diagonal_ssm,ssm_convolution,selective_ssm

def scalar_recurrence(inputs,a,b,c,d=0.,initial=0.):
    state=float(initial)
    outputs=[]
    states=[]
    for value in inputs:
        state=a*state+b*value
        outputs.append(c*state+d*value)
        states.append(state)
    return np.asarray(outputs),np.asarray(states)

impulse,impulse_states=scalar_recurrence([1.,0.,0.,0.],.5,1.,1.)
step_response,step_states=scalar_recurrence([1.,1.,1.,1.],.5,1.,1.)
np.testing.assert_allclose(impulse,[1.,.5,.25,.125])
np.testing.assert_allclose(step_response,[1.,1.5,1.75,1.875])
```

输入直接支路 d 不经过状态，所以它只影响当前输出，不持续留在未来状态。若 d 为二，上面的冲激第一输出增加二，后续输出不因这个直接支路继续增加。完整模型常把这条路径与残差或门控组合，读代码时要分清它是在系统内部输出公式里，还是在 block 外部残差里。

## 第二课　多维状态：不同时间尺度怎样组合

把状态扩成 n 维，输入仍为标量。一般线性系统为 $h_t=Ah_{t-1}+Bu_t$，其中 A 是 n 乘 n 矩阵，B 是 n 维向量；输出 $y_t=Ch_t+Du_t$，C 是读取向量。A 可以让不同状态分量相互耦合。课程先用对角 A，此时只保存其对角元素数组，每个状态分量独立衰减，最后由 C 混合。

取两个状态的保留系数零点二和零点九，B 都为一，C 为一和负一。输入冲激一后，第一步两个状态都为一，输出零；第二步状态零点二、零点九，输出负零点七；第三步零点零四、零点八一，输出负零点七七。两个不同时间尺度的差可以形成对变化敏感的滤波，而不只是简单平均。

如果 C 都为正，输出可以混合快速与慢速平滑；如果 C 有正有负，可以形成更复杂响应。状态维度增加提供更多动态基函数，但并不自动保证任务能力。训练需要选择 A、B、C 或它们的参数化，让这些动态与数据中的依赖匹配。

```python
a=np.array([.2,.9])
b=np.array([1.,1.])
c=np.array([1.,-1.])
multi_out,multi_state=diagonal_ssm(np.array([1.,0.,0.]),a,b,c)
np.testing.assert_allclose(multi_out,[0.,-.7,-.77])
np.testing.assert_allclose(multi_state,[.04,.81])
```

状态维不是历史长度。两维状态处理一百万步后仍只有两个数，但无法无损保存一百万个任意独立输入。它保存的是输入经动力学滤波后的统计表示。长时间尺度可以保留某些慢变化或关联，不等于随机访问全部原文。评估时应根据任务需要区分长期趋势、精确复制、顺序与检索。

一般矩阵 A 的稳定性与特征值有关，对角情况则直接看每个系数的绝对值。绝对值小于一时，零输入的该方向衰减；等于一可能保持或振荡；大于一可能增长。存在输入驱动时，即使衰减系数小于一，状态也不必趋于零，因为每步都有新写入。不要把“稳定”误读成“输出永远很小”。

## 第三课　离散递推的完整展开与卷积核

初始状态为零时，第一步 h0 等于 B u0；第二步 h1 等于 A B u0 加 B u1；第三步 h2 等于 A 平方 B u0 加 A B u1 加 B u2。一般地，当前状态是所有过去输入经不同幂次 A 变换后的和。输出读取后得到 $y_t=\sum_{i=0}^tCA^{t-i}Bu_i$，再加直接项 D u_t。

定义因果卷积核第 j 项为 $K_j=CA^jB$，输出就是输入与这个核的因果卷积。若有直接支路 D，可以把它加到核的第零项。这个等价要求 A、B、C 固定，不随时间或输入改变；还要求初始状态项被正确处理。条件缺一不可，不能把所有状态模型都称为固定卷积。

手算标量 a 二分之一、b 二、c 三，核依次六、三、一点五。输入一、二、三，输出第一项六；第二项三乘一加六乘二等于十五；第三项一点五乘一加三乘二加六乘三等于二十五点五。递推状态则依次二、五、八点五，乘 c 三后完全相同。

```python
u_hand=np.array([1.,2.,3.])
rec_hand,_=scalar_recurrence(u_hand,.5,2.,3.)
kernel_hand=np.array([6.,3.,1.5])
conv_hand=np.convolve(u_hand,kernel_hand)[:len(u_hand)]
np.testing.assert_allclose(rec_hand,[6.,15.,25.5])
np.testing.assert_allclose(rec_hand,conv_hand)
rng=np.random.default_rng(11)
u=rng.normal(size=9)
a_diag=np.array([.3,.8])
b_diag=np.array([1.2,-.4])
c_diag=np.array([.7,1.1])
rec,final_state=diagonal_ssm(u,a_diag,b_diag,c_diag)
conv=ssm_convolution(u,a_diag,b_diag,c_diag)
np.testing.assert_allclose(rec,conv,atol=1e-12)
```

卷积形式有利于一次处理整段序列，递推形式适合逐步生成并维护紧凑状态。它们是同一模型的两种执行方式，而不是两种不同训练目标。具体高性能实现可能使用频域卷积、结构化核或扫描算法，但先理解这个展开，才能知道优化到底保持了什么函数。

## 第四课　非零初始状态、分块与直接支路

如果初始状态不是零，输出除了输入卷积，还包含初始状态经过 A 反复传播的贡献。按本章第一输入步先更新再输出的约定，第 t 个输出包含 $CA^{t+1}h_{initial}$。指数是 t 加一，而不是 t，因为初始状态在第一步已经乘过一次 A。这个 off-by-one 很容易让卷积对照失败。

取 a 二分之一、初始状态四、输入全零，第一状态二、第二一、第三二分之一，输出相同。若误写初始项为 a 的 t 次方乘四，就会得到四、二、一，整体错一拍。测试非零初始状态能发现零状态测试看不出的时序问题。

```python
zero_input=np.zeros(3)
nonzero_out,_=scalar_recurrence(zero_input,.5,1.,1.,initial=4.)
initial_term=4.*(.5**np.arange(1,4))
np.testing.assert_allclose(nonzero_out,[2.,1.,.5])
np.testing.assert_allclose(nonzero_out,initial_term)

first,first_state=diagonal_ssm(u[:4],a_diag,b_diag,c_diag)
second,second_state=diagonal_ssm(u[4:],a_diag,b_diag,c_diag,first_state)
np.testing.assert_allclose(np.concatenate([first,second]),rec,atol=1e-12)
np.testing.assert_allclose(second_state,final_state,atol=1e-12)
```

分块延续把第一段最终状态作为第二段初始状态，不需要保留全部过去输入。若你用卷积计算第二段，却只卷积第二段输入而没有初始状态项，就相当于清空历史。混合使用卷积 prefill 与递推 decode 时，必须把 prefill 的最终状态正确传给 decode，不能只返回最后输出。

直接支路 D 与状态输出可以同时存在。对卷积核第零项加 D，表示当前输入的即时贡献；对状态递推本身不作修改。若把 D 加进 B，输入会写入状态并影响未来，函数不同。每个参数的作用位置要从方程确认，不能根据字母相近或 shape 相同随意合并。

## 第五课　连续时间：导数描述的是瞬时变化

连续系统写成 $dh/d\tau=Ah+Bu$，这里 tau 是连续时间，导数表示状态在一个极短时间间隔内的变化率。离散 token 步不是天然的物理秒数，但可以借用连续系统来参数化不同时间尺度。采样间隔 Delta 决定从一次状态更新到下一次走了多长“内部时间”。

先看无输入标量方程 dh/dtau 等于 a h。解为初始状态乘 exp(a tau)。若 a 为负，状态指数衰减；a 为零，状态保持；a 为正，状态增长。经过时间间隔 Delta，旧状态系数是 exp(a Delta)。因此把连续衰减率参数化为负数，再取正 Delta，可以得到零到一之间的离散保留系数。

有输入时，若区间内输入恒定，状态会一边衰减一边接受持续驱动。不能简单把连续 B 原样当作离散写入系数，因为时间间隔越长，累积输入作用也不同。精确离散化需要对这段持续输入的贡献积分。下一课逐步推导，不把公式当作需要背诵的黑箱。

欧拉近似把导数乘 Delta，当作有限步变化，得到离散 a 近似一加 Delta a，离散 b 近似 Delta b。它在足够小步长时合理，但步长大时可能产生连续稳定系统没有的离散不稳定。例如连续 a 为负一、Delta 为三，欧拉旧状态系数负二，绝对值大于一；精确系数 exp(-3) 约零点零四九八，仍然衰减。

```python
continuous_rate=-1.
delta_large=3.
euler_coefficient=1+delta_large*continuous_rate
exact_coefficient=np.exp(delta_large*continuous_rate)
assert euler_coefficient==-2.
assert 0<exact_coefficient<1
```

这个例子不是说欧拉方法总错误，而是说明离散化方法和步长共同决定数值行为。阅读模型代码时，连续系统提供直觉，真正执行的是离散公式；两者需要明确对应。不要看到一个负 A 就直接断言所有离散实现必然稳定。

## 第六课　零阶保持 ZOH：逐项推导输入系数

在一个长度 Delta 的区间内，输入 u 保持常量。标量方程的解为旧状态乘 exp(a Delta)，加上从零到 Delta 的输入贡献积分。某个时刻写入的输入还要经过剩余时间衰减，所以积分是 b u 乘 $\int_0^\Delta \exp(a s)\,ds$。当 a 非零，积分等于 exp(a Delta) 减一再除 a。

于是精确离散系数为 $\bar a=\exp(a\Delta)$，$\bar b=b\,\operatorname{expm1}(a\Delta)/a$。若 a 等于零，积分极限为 Delta，因此 $\bar b=b\Delta$。不能直接在 a 等于零时做零除零，也不能为了避免异常把结果设零，那会删除输入作用。

取 a 负一、b 一、Delta 一、初始零、输入一。保留系数 e 的负一次方，写入系数一减 e 的负一次方，第一状态约零点六三二一二一。第二步同样输入一，状态为 e 的负一次方乘第一状态，再加同样写入，化简为一减 e 的负二次方，约零点八六四六六五。它逐渐接近连续系统稳态一。

```python
def zoh_scalar(rate,input_gain,delta):
    if delta<0:
        raise ValueError("nonnegative step required")
    transition=np.exp(rate*delta)
    input_coefficient=(delta*input_gain if rate==0
                       else np.expm1(rate*delta)/rate*input_gain)
    return float(transition),float(input_coefficient)

transition,input_coefficient=zoh_scalar(-1.,1.,1.)
state1=input_coefficient
state2=transition*state1+input_coefficient
np.testing.assert_allclose([state1,state2],[1-np.exp(-1),1-np.exp(-2)])
np.testing.assert_allclose(zoh_scalar(0.,2.,.3),[1.,.6])
```

使用 expm1 而不是 exp(x) 减一，是为了 x 很小时减少消减误差。比如 x 接近零，exp(x) 接近一，两个相近浮点数相减可能丢失有效位；expm1 专门计算这个差。数学等价不保证数值质量相同，模型代码需要把稳定形式落实到具体函数。

矩阵情形的精确 ZOH 可以写成矩阵指数与积分。若 A 可逆，常见形式为 A 的逆乘 exp(Delta A) 减单位阵再乘 B；但 A 不可逆时积分仍可能存在，不能把某种代数写法的逆不存在误认为系统无法离散化。对角 A 可以逐分量使用标量公式，课程先采用这一清晰路径。

## 第七课　时间尺度：Delta、衰减率与半衰期

离散保留系数 a_bar 决定每步留下多少旧状态。若它在零到一之间，经过 n 步剩下 a_bar 的 n 次方。半衰期满足这个值等于二分之一，所以 n 等于 log 二分之一除 log a_bar。a_bar 越接近一，半衰期越长；接近零则快速遗忘。这个公式让“长记忆”变成可计算的量。

对连续负率 a，经过时间 tau 剩下 exp(a tau)，连续半衰期为 log 二除负 a。离散间隔 Delta 越大，每个 token 跨过的内部时间越长，因此同样 token 数内忘得更快。若 Delta 由输入生成，就可以让不同输入导致不同时间推进，这为选择性状态提供一种机制。

例如连续率负一，Delta 为零点一，离散保留约零点九零四八，半衰期约六点九三 token；Delta 为一，保留约零点三六七九，半衰期约零点六九三 token。后者第一步就已经保留不到一半。不要把“Delta 大”直觉解释成“容量更大”，它在这里控制的是动态时间步。

```python
for delta in (.1,1.):
    coefficient=np.exp(-delta)
    half_life=np.log(.5)/np.log(coefficient)
    np.testing.assert_allclose(half_life,np.log(2)/delta)
```

多状态系统可以让不同状态维拥有不同负率，从而形成多个半衰期。输出读取可以组合快慢响应。训练也可能学习输入相关的 B、C 或 Delta，让写入与读取具有选择性。但时间尺度长并不意味着每个输入都被精确保留；状态仍是有限维的混合表示，多个输入可能相互叠加。

参数化 A 为负的指数，可以保证连续率为负；参数化 Delta 为 softplus，可以保证步长为正。它们是常见的稳定性设计思路，但完整模型还包含输入驱动、读出、门控与有限精度，不能因此宣称所有输出都有统一上界。稳定性分析需要说明研究的是无输入动力学、受限输入还是整个非线性网络。

## 第八课　选择性：参数随输入变化后，固定卷积为何失效

固定 A、B、C 的线性系统对输入满足叠加性与时间平移关系，因此可以用一个固定卷积核描述。若 B、C 或 Delta 由当前输入生成，状态更新系数就随内容变化，通常不再是固定卷积。你仍能把某一次输入展开成一串带不同系数的和，但这些系数依赖该输入本身，不能提前构造一个对所有输入通用的核。

课程标量 selective_ssm 令 Delta 等于 softplus(u)，连续率固定负一，并使用精确 ZOH。正的大输入让 Delta 变大，旧状态衰减更强，同时输入写入系数也按 ZOH 改变。负输入会得到较小正 Delta，但 value 本身仍可为负。这里“选择性”体现为动态系数，不表示模型显式判断某句话重要与否。

用两个不同输入序列检验线性叠加：固定线性系统满足 F(u+v)=F(u)+F(v)，零初始状态下成立；输入相关 Delta 一般不满足。这个反例比只说“有非线性”更具体，因为它直接展示固定卷积推导的前提被破坏。若某个特殊样例碰巧相等，应换用非退化输入，而不是推断系统仍然线性。

```python
u_select=np.array([.2,1.,-.4])
v_select=np.array([.7,-.3,.8])
linear_sum,_=diagonal_ssm(u_select+v_select,a_diag,b_diag,c_diag)
linear_u,_=diagonal_ssm(u_select,a_diag,b_diag,c_diag)
linear_v,_=diagonal_ssm(v_select,a_diag,b_diag,c_diag)
np.testing.assert_allclose(linear_sum,linear_u+linear_v)
select_sum=selective_ssm(u_select+v_select)
select_parts=selective_ssm(u_select)+selective_ssm(v_select)
assert not np.allclose(select_sum,select_parts)
```

输入相关参数不妨碍逐步推理，因为当前输入到来后就能生成本步系数。它会改变并行化路径：不能直接用固定卷积，但如果更新对旧状态仍是仿射形式，可以使用结合性的扫描。选择性与并行性不是非此即彼，关键是找到真正保留的代数结构。

## 第九课　官方 reference 的更新规则：不要混同精确 ZOH

本卷固定的 Mamba reference 入口见 [SOURCE_WALK](SOURCE_WALK.md)。在其中最简单实数路径，旧状态系数使用 exp(delta 乘 A)，写入项使用 delta 乘 B 乘 u。它与前面精确 ZOH 的旧状态衰减相同，但输入系数不同。我们必须按源码实际执行的离散公式建立对照，而不是用连续系统直觉替换它。

取 A 负一、delta 一、B 一、u 一、初始零。reference 简化路径第一状态为一，第二为一加 e 的负一次方；精确 ZOH 第一状态为一减 e 的负一次方，第二为一减 e 的负二次方。两组数都来自各自明确的公式，差异不是 bug，也不能通过放大误差容限把它们称为等价。

```python
reference_states=[]
reference_state=0.
zoh_states=[]
zoh_state=0.
for _ in range(2):
    reference_state=np.exp(-1.)*reference_state+1.
    zoh_state=np.exp(-1.)*zoh_state+(1-np.exp(-1.))
    reference_states.append(reference_state)
    zoh_states.append(zoh_state)
np.testing.assert_allclose(reference_states,[1.,1+np.exp(-1)])
np.testing.assert_allclose(zoh_states,[1-np.exp(-1),1-np.exp(-2)])
assert not np.allclose(reference_states,zoh_states)
```

当 delta 很小时，expm1(delta A) 除 A 的展开为 delta 加上更高阶项，因此两种写入系数可能接近；delta 大时差异更明显。这个观察解释近似关系，但不授权你在模型加载时随意切换离散化。权重训练时适应了具体更新规则，推理必须保持对应定义。

官方 reference 还支持多种 B/C 布局、可选 delta bias 与 softplus、直接支路 D、输出门 z，以及不同数据类型路径。本章先实现一个明确子集：实数 A，输入相关 B/C 共享于通道，正 delta 已经给定，可选 D 与门。我们不声称覆盖复杂数、所有分组布局或融合 CUDA 接口。

## 第十课　带 batch、channel、state 的完整 selective scan

采用官方 reference 常见轴顺序：u 与 delta 为 batch、channel、length；A 为 channel、state；输入相关 B 与 C 为 batch、state、length；内部状态为 batch、channel、state。这里 batch_size 与输入矩阵 B 不要混淆，代码分别使用 batch_size 和 b_param。每个通道有自己的 A 与 delta，但本例在同一 batch/time 下共享 B/C 的状态向量。

第 t 步先计算 transition 为 exp(delta 乘 A)，shape 为 batch、channel、state；写入为 delta 乘 u 再乘 B，同样 shape。状态逐元素更新。输出沿 state 轴与 C 做内积，得到 batch、channel；若 D 存在，加 D 乘当前 u；若门 z 存在，再乘 SiLU(z)。这些广播都应能在纸上写出轴对齐，而不是靠 NumPy 猜测。

```python
def silu_stable(x):
    return x*np.exp(-np.logaddexp(0.,-x))

def selective_scan_numpy(u,delta,a_param,b_param,c_param,d_skip=None,
                         gate=None,initial=None):
    u=np.asarray(u,dtype=float)
    delta=np.asarray(delta,dtype=float)
    batch_size,channels,length=u.shape
    state_dim=a_param.shape[1]
    if delta.shape!=u.shape or a_param.shape!=(channels,state_dim):
        raise ValueError("input or A shape mismatch")
    if b_param.shape!=(batch_size,state_dim,length) or c_param.shape!=b_param.shape:
        raise ValueError("B/C shape mismatch")
    if np.any(delta<=0):
        raise ValueError("positive delta required")
    state=(np.zeros((batch_size,channels,state_dim)) if initial is None
           else np.array(initial,dtype=float,copy=True))
    if state.shape!=(batch_size,channels,state_dim):
        raise ValueError("state shape mismatch")
    outputs=[]
    for t in range(length):
        step=delta[:,:,t]
        transition=np.exp(step[:,:,None]*a_param[None,:,:])
        write=(step*u[:,:,t])[:,:,None]*b_param[:,None,:,t]
        state=transition*state+write
        output=np.sum(state*c_param[:,None,:,t],axis=-1)
        if d_skip is not None:
            output=output+d_skip[None,:]*u[:,:,t]
        if gate is not None:
            output=output*silu_stable(gate[:,:,t])
        outputs.append(output)
    return np.stack(outputs,axis=-1),state

batch_size,channels,length,state_dim=2,3,5,4
scan_u=rng.normal(size=(batch_size,channels,length))
scan_delta=np.logaddexp(0.,rng.normal(size=scan_u.shape))+.01
scan_a=-np.exp(rng.normal(size=(channels,state_dim)))
scan_b=rng.normal(size=(batch_size,state_dim,length))
scan_c=rng.normal(size=scan_b.shape)
scan_d=rng.normal(size=channels)
scan_gate=rng.normal(size=scan_u.shape)
scan_out,scan_final=selective_scan_numpy(scan_u,scan_delta,scan_a,scan_b,scan_c,
                                        scan_d,scan_gate)
assert scan_out.shape==scan_u.shape
assert scan_final.shape==(batch_size,channels,state_dim)
first_scan,first_scan_state=selective_scan_numpy(
    scan_u[:,:,:2],scan_delta[:,:,:2],scan_a,scan_b[:,:,:2],scan_c[:,:,:2],
    scan_d,scan_gate[:,:,:2])
second_scan,second_scan_state=selective_scan_numpy(
    scan_u[:,:,2:],scan_delta[:,:,2:],scan_a,scan_b[:,:,2:],scan_c[:,:,2:],
    scan_d,scan_gate[:,:,2:],first_scan_state)
np.testing.assert_allclose(np.concatenate([first_scan,second_scan],axis=-1),scan_out)
np.testing.assert_allclose(second_scan_state,scan_final)
```

这份函数实际计算完整的所声明子集，包括输入相关 B/C、通道状态、直接支路、门与状态延续。它不是只写一行标量递推的概念演示。但它没有调用官方 CUDA 扩展，没有验证该扩展的安装与性能，也没有实现所有官方可选布局。边界写清楚，才能让学习成果既扎实又不夸大。

代码中 B/C 沿通道共享是一个明确选择，不能把任意 channel-specific B/C 张量直接传入并期待广播正确。扩展到分组或逐通道 B/C 时，应重新写 shape 契约和测试。不同实现的输入轴顺序也可能不同，迁移前应先转置到明确布局，而不是靠 reshape 改变数据含义。

## 第十一课　从 scan 到完整教学 block：投影、因果卷积、门与残差

一个完整状态模型层通常在 scan 前后还有多条路径。下面构造一个可运行教学 block：输入 D 维，先投影成两组 inner 维向量，一组进入局部因果卷积与 SiLU，另一组作为输出门。卷积后的内容生成 B、C 与正 delta；A 由负指数参数化。scan 输出加直接支路、乘门，再经输出投影回 D 维，最后与原输入残差相加。

这个 block 采用前面明确的 reference 风格写入 delta B u，且不包含所有实际模型的归一化、参数共享和初始化细节。它的目的，是让你完整追踪 stateful 层从输入到输出的生命周期。不能把它命名成官方 checkpoint 兼容模型，但也不能把它轻描淡写成只有一个抽象公式：它有独立 full 与 step 路径及缓存测试。

因果深度卷积对每个 inner 通道使用自己的短核，不混合通道。当前位置只读取当前与前若干输入投影。卷积缓存保存的是卷积前的内容投影，而 scan 状态保存的是动态记忆；二者是不同状态，缺少任何一个都会使 step 与 full 不一致。输出门不需要跨时间缓存，因为当前输入已经能生成它。

```python
class TinySelectiveBlock:
    def __init__(self,dim=4,inner=6,state_dim=3,kernel=3,seed=110):
        self.dim,self.inner,self.state_dim,self.kernel=dim,inner,state_dim,kernel
        local_rng=np.random.default_rng(seed)
        self.in_proj=local_rng.normal(0,.3,size=(dim,2*inner))
        self.conv=local_rng.normal(0,.3,size=(kernel,inner))
        self.b_proj=local_rng.normal(0,.3,size=(inner,state_dim))
        self.c_proj=local_rng.normal(0,.3,size=(inner,state_dim))
        self.dt_proj=local_rng.normal(0,.2,size=(inner,inner))
        self.dt_bias=np.full(inner,-1.)
        self.a=-np.exp(local_rng.normal(0,.2,size=(inner,state_dim)))
        self.skip=np.ones(inner)
        self.out_proj=local_rng.normal(0,.3,size=(inner,dim))
    def forward(self,x):
        projected=x@self.in_proj
        raw,gate=np.split(projected,2,axis=-1)
        convolved=np.zeros_like(raw)
        for t in range(len(x)):
            for lag in range(self.kernel):
                if t-lag>=0:
                    convolved[t]+=self.conv[lag]*raw[t-lag]
        content=silu_stable(convolved)
        delta=np.logaddexp(0.,content@self.dt_proj+self.dt_bias)+1e-4
        b=content@self.b_proj
        c=content@self.c_proj
        result,state=selective_scan_numpy(
            content.T[None,:,:],delta.T[None,:,:],self.a,
            b.T[None,:,:],c.T[None,:,:],self.skip,gate.T[None,:,:])
        output=x+result[0].T@self.out_proj
        conv_state=np.zeros((self.kernel-1,self.inner))
        if self.kernel>1:
            recent=raw[-(self.kernel-1):]
            conv_state[-len(recent):]=recent
        return output,{"scan":state[0],"conv":conv_state}
    def step(self,x,cache=None):
        if cache is None:
            scan_state=np.zeros((self.inner,self.state_dim))
            conv_state=np.zeros((self.kernel-1,self.inner))
        else:
            scan_state=np.array(cache["scan"],copy=True)
            conv_state=np.array(cache["conv"],copy=True)
        raw,gate=np.split(x@self.in_proj,2)
        window=np.concatenate([conv_state,raw[None,:]],axis=0)
        convolved=np.zeros(self.inner)
        for lag in range(self.kernel):
            convolved+=self.conv[lag]*window[-1-lag]
        content=silu_stable(convolved)
        delta=np.logaddexp(0.,content@self.dt_proj+self.dt_bias)+1e-4
        b=content@self.b_proj
        c=content@self.c_proj
        scan_state=np.exp(delta[:,None]*self.a)*scan_state
        scan_state+=(delta*content)[:,None]*b[None,:]
        read=np.sum(scan_state*c[None,:],axis=-1)+self.skip*content
        read*=silu_stable(gate)
        output=x+read@self.out_proj
        new_conv=window[1:].copy()
        return output,{"scan":scan_state,"conv":new_conv}

block=TinySelectiveBlock()
block_input=rng.normal(size=(7,4))
block_full,block_full_state=block.forward(block_input)
block_cache=None
block_rows=[]
for row in block_input:
    output,block_cache=block.step(row,block_cache)
    block_rows.append(output)
np.testing.assert_allclose(np.stack(block_rows),block_full,atol=1e-12)
np.testing.assert_allclose(block_cache["scan"],block_full_state["scan"],atol=1e-12)
np.testing.assert_allclose(block_cache["conv"],block_full_state["conv"],atol=1e-12)
```

full 路径先为整段输入计算卷积与参数，再调用 scan；step 路径显式维护卷积窗口和 scan 状态，二者不是简单互相调用，所以比较能覆盖不同数据流。两条路径仍共享权重与核心定义。随机权重只验证结构，不展示语言能力；后续训练需要为全部投影与状态路径建立梯度或使用自动微分框架。

读代码时尤其注意卷积核方向。lag 零乘当前输入，lag 一乘前一个，依此类推。不同卷积库可能在权重排列或相关运算约定上不同，加载时不能只看 kernel shape。用一个只有单个非零核系数的例子，可以明确它到底读取当前还是过去第几步。

## 第十二课　完整反向：状态把未来损失传回过去

对角离散系统的反向可以逐步推导。输出 y_t 为 c 与 h_t 的点积加直接项 d u_t。上游输出梯度 g_t 给当前状态贡献 g_t c，同时未来步骤也传回一份状态梯度。两者相加后，才穿过 h_t 等于 a 逐元素乘旧状态加 b u_t。每个参数被所有时间步共享，所以梯度必须跨时间累加。

对 a 的贡献是当前状态梯度逐元素乘旧状态；对 b 的贡献是当前状态梯度乘当前输入；对 c 的贡献是输出梯度乘当前状态；对 d 的贡献是输出梯度乘当前输入。输入梯度既来自状态写入，也来自直接支路。传给旧状态的梯度是当前状态梯度逐元素乘 a。循环倒序执行，就得到完整的时间反向传播。

这与上一章 Delta 反向的结构相似，但状态更新更简单。相似不等于可以复用同一公式：Delta 的 key 同时参与读取误差与写入，SSM 的固定 a/b 更新没有那两条路径。理解每个中间变量的依赖，才能正确写出反向，而不是记住一个循环模板。

```python
def diagonal_forward_saved(u,a,b,c,d=0.,initial=None):
    state=np.zeros_like(a) if initial is None else np.array(initial,copy=True)
    rows=[]
    saved=[]
    for value in u:
        previous=state.copy()
        state=a*state+b*value
        rows.append(c@state+d*value)
        saved.append((previous,state.copy()))
    return np.asarray(rows),saved

def diagonal_backward(u,a,b,c,d,saved,upstream):
    da=np.zeros_like(a);db=np.zeros_like(b);dc=np.zeros_like(c)
    du=np.zeros_like(u);dd=0.
    future=np.zeros_like(a)
    for t in range(len(u)-1,-1,-1):
        previous,state=saved[t]
        dc+=upstream[t]*state
        dd+=upstream[t]*u[t]
        dh=future+upstream[t]*c
        da+=dh*previous
        db+=dh*u[t]
        du[t]=dh@b+upstream[t]*d
        future=dh*a
    return du,da,db,dc,float(dd),future

grad_u=np.array([.3,-.7,1.2,.4])
grad_a=np.array([.4,.8])
grad_b=np.array([.7,-.2])
grad_c=np.array([1.1,.5])
grad_d=np.array(.3)
grad_initial=np.array([.2,-.1])
grad_upstream=rng.normal(size=4)
grad_output,grad_saved=diagonal_forward_saved(
    grad_u,grad_a,grad_b,grad_c,float(grad_d),grad_initial)
grad_expected=diagonal_backward(grad_u,grad_a,grad_b,grad_c,float(grad_d),
                                grad_saved,grad_upstream)
grad_params=[grad_u,grad_a,grad_b,grad_c,grad_d,grad_initial]
for parameter,expected in zip(grad_params,grad_expected):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=float(diagonal_forward_saved(grad_u,grad_a,grad_b,grad_c,
                    float(grad_d),grad_initial)[0]@grad_upstream)
        parameter[index]=old-1e-6
        minus=float(diagonal_forward_saved(grad_u,grad_a,grad_b,grad_c,
                     float(grad_d),grad_initial)[0]@grad_upstream)
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=2e-7,rtol=2e-7)
```

测试包括非零初始状态和直接支路，所以不会因它们恰好为零而漏掉梯度路径。对零维数组 d 的数值差分也逐元素处理。完整梯度验证不是只检查梯度非零，而是比较每个输入与参数的解析导数和中心差分。这样可以发现漏加共享参数梯度、时间顺序错误或初始状态处理错误。

### 十二点一　选择性 scan 的梯度多了哪些路径

当 delta、B、C 由输入生成，scan 内部首先需要对这些中间参数求导，再通过生成它们的投影回传。单通道实数简化中，transition 为 exp(delta A)，写入为 delta B u。状态梯度对 A 的贡献包含旧状态、transition 与 delta；对 delta 的贡献包含旧状态衰减变化和输入写入变化两部分。漏掉其中一项会得到错误训练。

输出 C 随时间变化时，每步 C 都有自己的直接梯度，但如果 C 来自同一个投影矩阵，该矩阵仍跨时间共享，最终要把所有步骤的贡献相加。u 同时参与写入，还可能生成 delta、B、C 与门，因此完整 block 的输入梯度包含多条支路。只计算 scan 对 u 的直接导数，不等于完整 block 的输入导数。

下面给出无直接支路、无输出门的单通道选择性 scan 反向，保留输入相关 B/C 和 delta。它是第十课多通道函数的一个可手算子集，用来验证指数衰减与写入共同依赖 delta 的链式法则。完整 block 可以由自动微分框架组装这些已验证基本操作，但需要另做整体梯度检查。

```python
def selective_single_forward(u,delta,a,b,c):
    state=np.zeros_like(a)
    output=[]
    saved=[]
    for t in range(len(u)):
        previous=state.copy()
        transition=np.exp(delta[t]*a)
        state=transition*previous+delta[t]*b[t]*u[t]
        output.append(c[t]@state)
        saved.append((previous,transition,state.copy()))
    return np.asarray(output),saved

def selective_single_backward(u,delta,a,b,c,saved,upstream):
    du=np.zeros_like(u);ddelta=np.zeros_like(delta);da=np.zeros_like(a)
    db=np.zeros_like(b);dc=np.zeros_like(c);future=np.zeros_like(a)
    for t in range(len(u)-1,-1,-1):
        previous,transition,state=saved[t]
        dc[t]=upstream[t]*state
        dh=future+upstream[t]*c[t]
        da+=dh*previous*transition*delta[t]
        ddelta[t]=dh@(previous*a*transition+b[t]*u[t])
        db[t]=dh*delta[t]*u[t]
        du[t]=dh@(delta[t]*b[t])
        future=dh*transition
    return du,ddelta,da,db,dc

sel_u=np.array([.2,-.3,.7])
sel_dt=np.array([.4,.8,.2])
sel_a=np.array([-.7,-1.2])
sel_b=rng.normal(size=(3,2))
sel_c=rng.normal(size=(3,2))
sel_upstream=rng.normal(size=3)
sel_out,sel_saved=selective_single_forward(sel_u,sel_dt,sel_a,sel_b,sel_c)
sel_grad=selective_single_backward(sel_u,sel_dt,sel_a,sel_b,sel_c,sel_saved,sel_upstream)
sel_params=[sel_u,sel_dt,sel_a,sel_b,sel_c]
for parameter,expected in zip(sel_params,sel_grad):
    numerical=np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        old=parameter[index]
        parameter[index]=old+1e-6
        plus=selective_single_forward(*sel_params)[0]@sel_upstream
        parameter[index]=old-1e-6
        minus=selective_single_forward(*sel_params)[0]@sel_upstream
        parameter[index]=old
        numerical[index]=(plus-minus)/2e-6
    np.testing.assert_allclose(numerical,expected,atol=2e-7,rtol=2e-7)
```

### 十二点二　梯度消失与爆炸的具体来源

固定对角系统中，跨很多步传回的状态梯度会反复乘 a。绝对值小于一时，远处梯度可能衰减；大于一时可能放大；接近一时保留更久。选择性系统则乘一串输入相关 transition，不同状态方向与不同输入路径可能有不同衰减程度。长记忆与可训练性之间存在数值取舍，不是只把 a 设成一就解决所有问题。

输出读取 C、输入写入 B 和残差也影响梯度路径。某个状态方向衰减慢，但若 B 几乎不写入或 C 几乎不读取，它对任务仍可能不起作用。分析时应同时看动态系数、写入幅度、读出幅度和梯度，而不是只展示 A 的分布。一个完整系统的记忆能力来自这些组件的配合。

## 第十三课　十二组练习与完整参考答案

### 练习一：冲激响应与参数职责

a 为零点八、b 为二、c 为二分之一、d 为零，输入一、零、零。第一状态二，输出一；第二状态一点六，输出零点八；第三状态一点二八，输出零点六四。b 控制写入幅度，c 控制读出幅度，二者乘积决定冲激核首项，但在内部状态尺度上作用不同。

```python
answer_impulse,_=scalar_recurrence([1.,0.,0.],.8,2.,.5)
np.testing.assert_allclose(answer_impulse,[1.,.8,.64])
```

若把 b 乘十、c 除十，零初始线性系统的输入输出可以保持不变，但内部状态放大十倍。这种参数不唯一性说明仅比较隐藏状态坐标不一定反映函数差异，也说明数值范围可能因参数化变化而改变。

### 练习二：阶跃稳态

标量 a 绝对值小于一、恒定输入 u 时，稳态满足 h 等于 a h 加 b u，移项得到 h 为 b u 除一减 a，输出再乘 c 加 d u。a 二分之一、b c u 都为一，稳态二。前四项一、一点五、一点七五、一点八七五，只是逐渐逼近，不应在有限步直接断言精确达到二。

```python
long_step,_=scalar_recurrence(np.ones(20),.5,1.,1.)
assert long_step[-1]<2.
np.testing.assert_allclose(long_step[-1],2*(1-.5**20))
```

### 练习三：非零初始项

a 二分之一、初始四、输入零，输出二、一、二分之一。若同时有输入一、二、三，输出等于零初始卷积结果加这条初始项。完整参考把两部分分别计算后相加，验证时序指数 t 加一。

```python
with_initial,_=scalar_recurrence([1.,2.,3.],.5,1.,1.,initial=4.)
without_initial,_=scalar_recurrence([1.,2.,3.],.5,1.,1.)
np.testing.assert_allclose(with_initial,without_initial+np.array([2.,1.,.5]))
```

### 练习四：直接支路只能改变当前核项

取 d 为三，比较有无直接支路的输出差，应该恰好等于三乘当前输入。不能把 d 加到 b 后再运行，因为那会改变未来状态。完整程序同时构造错误版本，证明二者在非退化序列上不同。

```python
direct,_=scalar_recurrence([1.,0.,0.],.5,1.,1.,d=3.)
plain,_=scalar_recurrence([1.,0.,0.],.5,1.,1.)
wrong_direct,_=scalar_recurrence([1.,0.,0.],.5,4.,1.)
np.testing.assert_allclose(direct-plain,[3.,0.,0.])
assert not np.allclose(direct,wrong_direct)
```

### 练习五：欧拉与精确离散化

连续率负一、步长三，欧拉系数负二，精确系数 e 的负三次方。零输入初始一，欧拉状态一连乘负二会交替增长，精确状态持续衰减。答案应指出这是离散方法与步长造成的差异，不是连续系统本身不稳定。将步长缩小可改善欧拉近似，但不能无条件替代目标模型的离散定义。

### 练习六：零率极限

连续 a 为零，导数仅为 b u，区间内恒定输入使状态增加 Delta b u。ZOH 的输入系数应取极限 Delta b。第六课函数已给出完整代码；测试 delta 零时旧状态系数一、输入系数零，也应成立。这个边界能发现直接使用 expm1 除 a 的零除错误。

```python
np.testing.assert_allclose(zoh_scalar(0.,3.,.2),[1.,.6])
np.testing.assert_allclose(zoh_scalar(-2.,3.,0.),[1.,0.])
```

### 练习七：选择性破坏固定卷积的条件

把 B 改成当前输入的函数，例如 B_t 等于 u_t，则写入成为 u_t 平方。即使 A 和 C 固定，系统对输入不再线性。输入取一与二，分别写入一与四，和为五；合并输入三写入九，不等于五。这个最小反例说明不能再用一个固定线性卷积核描述所有输入。

```python
def input_dependent_write(inputs):
    state=0.;output=[]
    for value in inputs:
        state=.5*state+value*value
        output.append(state)
    return np.asarray(output)
assert input_dependent_write([1.])[0]+input_dependent_write([2.])[0]==5.
assert input_dependent_write([3.])[0]==9.
```

### 练习八：reference 与 ZOH 一步差异

A 负一、delta 一、B u 为一，reference 写入一，ZOH 写入一减 e 的负一次方。答案必须写出两条离散公式，并说明旧状态系数相同。不能把差异称为“实现精度不同”，因为即使无限精度，两条函数也不同。第九课两步代码是完整参考。

### 练习九：选择性 scan 的状态延续

第十课给出两段延续的完整代码。扩展测试把长度五切成一、三、一三段，保持每段 B/C、delta 与 u 对应同一时间范围，最终输出和状态与一次运行一致。错误切片常发生在 B/C 的时间轴位于最后，而某些教学数组习惯时间轴在最前；必须按 shape 契约切片。

```python
pieces=[]
scan_state=None
start=0
for count in (1,3,1):
    stop=start+count
    piece,scan_state=selective_scan_numpy(
        scan_u[:,:,start:stop],scan_delta[:,:,start:stop],scan_a,
        scan_b[:,:,start:stop],scan_c[:,:,start:stop],scan_d,
        scan_gate[:,:,start:stop],scan_state)
    pieces.append(piece)
    start=stop
np.testing.assert_allclose(np.concatenate(pieces,axis=-1),scan_out)
np.testing.assert_allclose(scan_state,scan_final)
```

### 练习十：完整 block 的未来不干扰

改变输入最后一个位置，所有更早输出应保持不变。这个测试覆盖因果卷积、输入相关参数与 scan，而不只是检查某个 mask。状态模型没有显式 attention mask，也仍然必须满足自回归因果性。若卷积误用了对称 padding 或未来窗口，测试会直接暴露。

```python
changed_block_input=block_input.copy()
changed_block_input[-1]+=10.
changed_block_output,_=block.forward(changed_block_input)
np.testing.assert_allclose(changed_block_output[:-1],block_full[:-1],atol=1e-12)
```

### 练习十一：漏掉卷积缓存的故障

用正确 step 执行第一 token 后，下一步把 scan 状态保留，但把卷积缓存清零，再与正确路径比较。结果一般不同，证明完整 block 的状态不只有 scan 矩阵。测试选择 kernel 大于一和非零历史输入，避免退化情况掩盖错误。

```python
_,prefix_cache=block.step(block_input[0])
correct_next,_=block.step(block_input[1],prefix_cache)
broken_cache={"scan":prefix_cache["scan"].copy(),
              "conv":np.zeros_like(prefix_cache["conv"])}
wrong_next,_=block.step(block_input[1],broken_cache)
assert not np.allclose(correct_next,wrong_next)
```

### 练习十二：梯度与实现边界报告

第十二课两套完整反向是参考答案。提交时列出你验证了哪些参数、是否包含初始状态、直接支路、输入相关 delta/B/C，以及哪些外层投影尚未做整体训练验证。不能把 scan 子集的差分通过写成整个官方模型训练已复现。准确界定结果范围，是技术能力的一部分，不是降低要求。

## 第十四课　训练一个时间尺度，再推导并行 scan

先训练一个可解释参数。生成一段固定输入，用真实保留系数零点八五、写入与读取系数一产生目标输出。学习模型只训练自己的保留系数 a，通过 sigmoid 参数化使它位于零到一。初始 a 为零点二，模型忘得太快；训练应把 a 推向目标，使整段输出接近。这个实验实际更新参数并通过时间反向计算梯度，不是只拟合一条已经给出的曲线图。

```python
fit_rng=np.random.default_rng(111)
fit_u=fit_rng.normal(size=20)
fit_target,_=scalar_recurrence(fit_u,.85,1.,1.)
fit_logit=np.log(.2/.8)
fit_losses=[]
for step in range(500):
    coefficient=1/(1+np.exp(-fit_logit))
    fit_a=np.array([coefficient])
    fit_b=np.array([1.])
    fit_c=np.array([1.])
    prediction,saved=diagonal_forward_saved(fit_u,fit_a,fit_b,fit_c)
    difference=prediction-fit_target
    fit_losses.append(float(np.mean(difference**2)))
    gradients=diagonal_backward(fit_u,fit_a,fit_b,fit_c,0.,saved,
                                 2*difference/len(difference))
    logit_gradient=gradients[1][0]*coefficient*(1-coefficient)
    fit_logit-=.1*logit_gradient
learned_coefficient=1/(1+np.exp(-fit_logit))
assert np.isfinite(fit_losses).all()
assert fit_losses[-1]<fit_losses[0]*.001
np.testing.assert_allclose(learned_coefficient,.85,atol=.01)
```

这里 a 在所有时间步共享，所以 diagonal_backward 已经把各步骤的贡献求和。再乘 sigmoid 导数，得到参数 logit 的梯度。若你直接更新 a，可能越过稳定范围；sigmoid 约束避免这个情况，但接近边界时导数也会变小。参数化既影响可表达范围，也影响优化几何，不能只把它当作防报错技巧。

目标由同一类系统生成，所以这是系统辨识的最小实验：从输入输出恢复一个动态参数。它不涉及自然语言，也不证明复杂选择性模型训练成功。你可以增加噪声、多个状态或只观察部分输出，研究参数能否被唯一恢复；这些变化会引出数据可辨识性，而不仅是优化器是否工作。

### 十四点一　扫描组合的完整推导

选择性对角更新每一步都是 h 变成 a_t 逐元素乘 h 加 b_t。把一步表示为一对数组 a、b。先执行第一步再执行第二步，结果为 a2 乘 a1 乘 h，加 a2 乘 b1，再加 b2。因此组合为 a2 乘 a1，以及 a2 乘 b1 加 b2。所有乘法逐元素，状态维彼此独立，组合成本与状态维线性相关。

三步组合改变括号不改变结果，因为普通乘法分配律和结合律成立；改变时间顺序一般会改变加法项。并行 scan 利用结合律，把长序列拆成树状组合或块前缀，减少串行依赖深度。它没有把递推改成无序求和，也没有删除当前输入对系数的影响。

下面用顺序前缀组合建立一个可读参考。虽然代码仍循环，组合对象已经与单步状态分开，因此可以验证后续并行实现。每个前缀变换应用于初始状态，得到对应 h_t。这个参考不是高性能内核，作用是明确结合操作与状态语义。

```python
def compose_diagonal(second,first):
    a2,b2=second
    a1,b1=first
    return a2*a1,a2*b1+b2

def affine_prefix_reference(transitions,writes,initial):
    aggregate=(np.ones_like(initial),np.zeros_like(initial))
    states=[]
    for a,b in zip(transitions,writes):
        aggregate=compose_diagonal((a,b),aggregate)
        states.append(aggregate[0]*initial+aggregate[1])
    return np.asarray(states)

transitions=np.exp(scan_delta[0,:, :].T[:,:,None]*scan_a[None,:,:])
writes=(scan_delta[0]*scan_u[0]).T[:,:,None]*scan_b[0].T[:,None,:]
prefix_states=affine_prefix_reference(transitions,writes,np.zeros((channels,state_dim)))
serial_state=np.zeros((channels,state_dim))
serial_states=[]
for a,b in zip(transitions,writes):
    serial_state=a*serial_state+b
    serial_states.append(serial_state.copy())
np.testing.assert_allclose(prefix_states,np.stack(serial_states),atol=1e-12)
```

并行实现可能采用不同归约顺序，浮点结果未必逐位一致。固定种子只固定输入与参数，不能让所有加乘顺序自动一致。验证应比较误差范围与增长趋势，并单独检查每步公式。若误差在短序列就很大，先怀疑轴、广播或更新顺序；若只在极长序列累积，才进一步分析精度与稳定性。

### 十四点二　为什么不是所有 RNN 都能这样扫描

若更新为 h 等于 tanh(Ah 加 Bu)，非线性直接作用于旧状态，两个步骤复合后通常不能用同样大小的一对 A/B 精确表示。你仍可以抽象地把整个函数作为组合对象，但表示与计算成本可能随长度增长，失去实用的固定规模扫描结构。可扫描性取决于更新族对组合是否封闭，而不是只要写成循环就成立。

选择性 SSM 虽然参数依赖输入，但在计算当前参数后，对旧状态仍是仿射，所以保留这个结构。Delta 更新也能写成矩阵仿射，但矩阵维度与低秩结构带来不同成本。把它们统称“支持 scan”之后，还要继续问组合对象多大、如何利用结构、是否需要分块和近似。数学可能性只是算法设计的第一步。

## 第十五课　深入工作坊：动态系统知识怎样帮助读模型

### 十五点一　可写入与可读出：有状态不等于有用记忆

考虑两个状态，A 对角为零点五和零点九，B 为一和零，C 为一和一。第二个状态虽然保留系数很大，却从零初始开始永远收不到输入，因为 B 的第二分量为零。它有潜在的慢时间尺度，但没有被写入，因而不贡献任务信息。只统计状态维度会高估实际使用的记忆容量。

反过来，B 两项都为一，C 为一和零。第二个状态会持续写入并变化，但当前输出完全不读取它。如果未来层也没有其他路径使用它，它对输入输出函数没有作用。系统理论中会研究可控性与可观测性，初学阶段不必先背完整矩阵判据，但应理解“能否写进去”和“能否读出来”是两个独立问题。

```python
unused_write,_=diagonal_ssm(np.array([1.,2.,3.]),
                            np.array([.5,.9]),np.array([1.,0.]),np.array([1.,1.]))
single_state,_=scalar_recurrence([1.,2.,3.],.5,1.,1.)
np.testing.assert_allclose(unused_write,single_state)
hidden_unread,_=diagonal_ssm(np.array([1.,2.,3.]),
                            np.array([.5,.9]),np.array([1.,1.]),np.array([1.,0.]))
np.testing.assert_allclose(hidden_unread,single_state)
```

在选择性模型中，B/C 随输入变化，某个状态方向可能只对某类内容写入或读取。分析模型时可以记录这些系数与状态贡献，但不能仅凭一个大状态值判断它重要。输出敏感度、后续层使用和任务表现都需要考虑。这与 attention 权重不等于完整解释的道理相似：一个内部统计只揭示局部机制。

### 十五点二　振荡与复数：状态不只有指数衰减

实数标量 a 为负时，状态符号交替；二维旋转衰减矩阵则可以产生更平滑的振荡。设 A 为半径 r 乘二维旋转矩阵，每步状态旋转角度 theta，同时范数乘 r。r 小于一时，轨迹螺旋衰减；r 等于一时理想情况下保持范数。这样的动态可以表达周期性或相位变化，说明状态模型不只是若干低通平均器。

取 r 二分之一、theta 为直角，初始状态一、零，零输入。一步变零、二分之一；两步负四分之一、零；三步零、负八分之一；四步十六分之一、零。每步旋转同时缩小。这个实数二维表示与一对复共轭特征值有关，但你不需要先掌握复杂谱理论也能手算其行为。

```python
rotation_decay=.5*np.array([[0.,-1.],[1.,0.]])
osc_state=np.array([1.,0.])
osc_states=[]
for _ in range(4):
    osc_state=rotation_decay@osc_state
    osc_states.append(osc_state.copy())
np.testing.assert_allclose(osc_states,[[0.,.5],[-.25,0.],[0.,-.125],[.0625,0.]])
```

本章 selective_scan_numpy 只支持实数对角 A，没有实现这种二维耦合或官方复杂数路径。这个例子用于拓展动态系统直觉，不应把它直接塞入当前对角接口。扩展一般矩阵状态需要修改乘法、参数形状、离散化与扫描组合，不能只把 a 数组换成矩阵而保留逐元素运算。

### 十五点三　卷积核的形状告诉你系统偏好什么

冲激响应是系统对单个瞬时输入的输出轨迹。对线性时不变系统，任意输入可以看成不同时间位置的冲激加权和，因此冲激响应决定整个输入输出函数。快速衰减核主要关注近期，慢衰减核保留更久，正负交替核可以对变化和频率产生不同响应。这个视角把抽象 A/B/C 与可视化的时间行为连接起来。

如果输入是缓慢变化的序列，平滑型系统会把高频波动压低；如果输出混合快慢状态的差，可能强调突变。语言 token 并非普通连续信号，不能直接把音频滤波结论当作语言能力证明，但这些动态基元仍提供理解内部时间尺度的工具。训练决定哪些组合对任务有用。

选择性系统没有一个固定冲激核适用于所有内容，因为系数随输入改变。你可以在某个固定背景序列上施加小扰动，观察局部响应，但那是输入相关的分析，不是全局线性卷积核。报告时应说明背景与扰动位置，避免把一次局部实验画成模型的永久属性。

### 十五点四　长期误差怎样累积

设理想状态满足 h_t 等于 a h_{t-1} 加写入，实际计算每步引入误差 e_t。两者差近似满足误差状态等于 a 乘前一步误差加 e_t。若绝对值 a 小于一，过去误差会衰减，但每步新误差仍会累加；若 a 接近一，误差保留更久。误差上界通常与一除一减绝对值 a 有关，因此很慢的动态对有限精度更敏感。

这不意味着慢状态不能使用，而是需要合适的状态精度、参数化和算法。输入与权重可以用较低精度，状态累加可能使用更高精度；具体选择要以实现和测试为准。不能只看权重是 BF16 就断言所有状态也以同样精度保存，更不能在报告中省略状态 dtype。

比较串行与并行 scan 时，应记录每步状态误差、输出误差和最大误差位置。若只看最终输出，C 的抵消可能掩盖大状态差异；若只看状态范数，也可能夸大对任务不敏感方向的误差。两者结合，再看任务指标，才能判断数值差异的实际影响。

### 十五点五　门控与残差不应被合并成一句话

第十一课 block 中，scan 内部的 D 支路把当前内容直接加到 scan 读出；输出门乘在这个合成结果上；输出投影回到隐藏维；最后还有输入残差。四个操作的位置不同。把 D 支路误放到门后，会让它不受门控制；把残差也乘门，会改变恒等路径；把输出投影放到状态更新前，会改变状态维与写入定义。

可以逐项关闭组件定位作用。把输出投影设零，整个 block 应退化为恒等输入；把卷积只保留 lag 零，局部历史缓存不再影响内容，但 scan 仍记忆过去；把 B 投影设零，scan 状态从零开始不再写入，但 D 直接支路仍存在；把 C 投影设零，状态可能继续变化，却不通过状态读出影响输出。每个退化实验都有明确预期。

```python
import copy
zero_out_block=copy.deepcopy(block)
zero_out_block.out_proj.fill(0.)
zero_output,_=zero_out_block.forward(block_input)
np.testing.assert_allclose(zero_output,block_input)
zero_write_block=copy.deepcopy(block)
zero_write_block.b_proj.fill(0.)
_,zero_write_state=zero_write_block.forward(block_input)
np.testing.assert_array_equal(zero_write_state["scan"],
                              np.zeros_like(zero_write_state["scan"]))
```

这些实验不需要真实语言数据，随机输入就能验证结构。它们比只观察生成文本更容易定位错误，因为预期是精确的代数退化。后续把 block 放进模型时，也应保留这种组件级测试，避免大系统掩盖简单接口错误。

### 十五点六　固定状态的服务生命周期

每个请求需要独立 scan 状态与卷积缓存，模型权重可以共享。创建新请求时两者初始化；处理 prompt 后保留最终状态；逐 token 生成继续更新；请求结束后释放。若只重置 scan 而忘记卷积缓存，新请求开头会混入旧请求的局部投影。状态隔离必须覆盖全部动态数组，而不是只覆盖最显眼的 h。

分支搜索可以复制状态快照，但原地内核可能让两个分支意外共享缓冲区。回滚则需要保存早期快照或重算，因为某些衰减、门控和非线性处理不可逆。固定大小使复制成本可控，却不意味着能从最终状态恢复任意历史。历史编辑后仍需从最长不变前缀重新推进。

模型权重更新会使旧状态失效。即使只改变输出 C 或投影，某些中间状态在单层局部可能仍数值可用，完整多层模型后续状态却可能依赖改变后的输出。最安全的服务契约是将状态绑定模型版本，并在版本变化时重建；若要做更细的复用，必须逐层证明依赖不受影响。

### 十五点七　padding 与状态更新的关系

在 batch 中，无效 padding 位置不应默认推进状态。即使输入向量为零，delta、B/C、卷积 bias 或门参数化仍可能产生非零系数，旧状态可能衰减，卷积窗口也可能向前滑动。只在损失上忽略 padding，并不能保证后续有效 token 的状态与独立运行一致。

如果设计要求 padding 不代表真实时间，就应在更新处保持旧状态与旧卷积缓存不变，并跳过输出或标为无效。若 padding 被定义为真实空白时间，则可以推进动态，但这属于另一种语义。接口必须说明，不能由实现细节偶然决定。左 padding、右 padding和中间无效槽位都应分别测试。

对选择性 scan 来说，batch 内每条序列的时间长度不同，最后状态应来自各自最后一个有效 token，而不是公共最大长度。服务框架通常需要有效长度或状态掩码；模型核心只接受整齐张量时，外围适配必须负责正确隔离。这是从数学递推走向可用推理系统的一步。

### 十五点八　双通道 selective scan 的逐项手算

取两个通道、两个状态维，A 的每一行都是负一、负二，两个时间步的 delta 都为 log 二。因此每步 transition 都是二分之一、四分之一。第一步 B 为一、二，C 为一、四分之一，两个通道输入分别一和二，初始状态全零。第一个通道写入为 log 二乘一、二；第二个通道写入为 log 二乘二、四。

第一步状态读出，第一个通道为 log 二加四分之一乘二 log 二，等于一点五 log 二；第二个为二 log 二加四分之一乘四 log 二，等于三 log 二。再令 D 为一、二分之一，直接支路分别加一与一，输出为一加一点五 log 二，以及一加三 log 二。这里没有输出门，避免额外非线性遮住手算主线。

第二步输入改为零和一，B 改为二、一，C 不变。第一通道没有新写入，旧状态乘 transition 后为二分之一 log 二、二分之一 log 二，读取为八分之五 log 二。第二通道旧状态衰减后为 log 二、log 二，新写入为二 log 二、log 二，合成三 log 二、二 log 二，读取三点五 log 二，再加直接支路二分之一。

```python
log_two=np.log(2.)
hand_u=np.array([[[1.,0.],[2.,1.]]])
hand_delta=np.full_like(hand_u,log_two)
hand_a=np.array([[-1.,-2.],[-1.,-2.]])
hand_b=np.array([[[1.,2.],[2.,1.]]])
hand_c=np.array([[[1.,1.],[.25,.25]]])
hand_d=np.array([1.,.5])
hand_scan,hand_state=selective_scan_numpy(hand_u,hand_delta,hand_a,
                                         hand_b,hand_c,hand_d)
np.testing.assert_allclose(hand_scan,[[[1+1.5*log_two,.625*log_two],
                                     [1+3*log_two,.5+3.5*log_two]]])
np.testing.assert_allclose(hand_state,[[[.5*log_two,.5*log_two],
                                      [3*log_two,2*log_two]]])
```

这个例子让每个广播都可见：B/C 在同一时间对两个通道共享，delta 与 u 按通道不同，A 在通道与状态轴上有自己的位置。若把 B 的时间轴与状态轴交换，因为这里尺寸恰好都为二，shape 不会报错，但第二步写入含义会改变。方形小矩阵最容易掩盖轴错误，正式随机测试应同时使用不同的通道数、状态维和长度。

### 十五点九　参数量与状态量：两张独立账本

默认 TinySelectiveBlock 的隐藏维四、inner 六、状态维三、卷积宽三。scan 状态十八个元素，卷积缓存十二个元素，总共三十个持久元素，不随处理过的 token 数增加。若 batch 为八，就需要八份状态；若有多层，每层独立保存。这个公式不包含当前投影、扫描中间量或训练反向保存的激活。

参数账本则不同。输入投影四乘十二，卷积三乘六，B 与 C 投影各六乘三，delta 投影六乘六及六个 bias，A 参数六乘三，直接支路六，输出投影六乘四。总数一百九十二。参数与状态恰好都能用数组 size 统计，但它们的生命周期、共享范围和增长因素不同，不能放在同一概念下。

```python
block_parameters=[block.in_proj,block.conv,block.b_proj,block.c_proj,
                  block.dt_proj,block.dt_bias,block.a,block.skip,block.out_proj]
assert sum(array.size for array in block_parameters)==192
assert block_full_state["scan"].size+block_full_state["conv"].size==30
```

如果把 inner 加倍，很多投影参数按平方或线性增长，状态也随 inner 增长；若只增加状态维，scan 状态与 B/C/A 参数增加，但输入输出投影不一定变化；若只增加卷积宽，局部缓存与卷积参数增加。理解这些方向，才能有目的地设计小模型实验，而不是随意调一组数字后只看总量。

### 十五点十　渐进项目的具体交付

第一阶段交付一个线性系统实验器，支持标量与对角状态、非零初始状态、直接支路、冲激与阶跃输入。它应同时输出递推轨迹与卷积参考，逐项解释差异为零的条件。将 a 设为零、接近一、负数和大于一，观察不同动态。每个图或表都附输入和参数，不让视觉代替可复现数值。

第二阶段交付连续到离散实验器，比较欧拉与精确 ZOH。固定连续负率，改变 delta，记录旧状态系数、输入系数和多步轨迹。特别包含 a 为零与极接近零的边界，以及 delta 为零。写出为什么 expm1 更稳定，为什么矩阵逆公式有适用前提。这个阶段不涉及神经网络，也足以建立扎实的动态系统直觉。

第三阶段交付选择性 scan，严格采用声明的布局和离散公式，支持输入相关 B/C、delta、直接支路、门与状态延续。使用第十五点八的手算作确定测试，再用随机非方形尺寸作回归。记录每个中间张量 shape 与状态轨迹，建立与固定源码 reference 的符号映射。未实现的复杂数或分组布局明确拒绝，不让广播意外接受。

第四阶段交付完整 TinySelectiveBlock 的独立复写。保留 full 与 step 两条路径，测试因果卷积方向、卷积缓存、scan 状态、输出门、直接支路和残差。至少做三种退化实验与未来不干扰测试。若接入自动微分框架，先在微型尺寸上做整体梯度检查，再尝试训练；不要因为 NumPy 前向通过就跳过训练计算图验证。

第五阶段做可解释训练。先完成单参数时间尺度辨识，再增加多个状态与噪声，观察是否仍能恢复输入输出行为。随后选择小型序列任务，例如延迟复制或带干扰的信号保持，明确训练与验证分离。记录任务损失、状态范数、delta 分布和梯度范数。模型能力的结论只针对实际任务，不外推成通用语言能力。

第六阶段才进入性能。固定数学定义与 dtype，分别测整段 prefill、单步 decode 和状态复制。比较串行与 scan 参考时先验证输出，再讨论耗时。服务器 GPU 适合短时受控实验，但安装与编译官方扩展属于单独环境任务；没有执行的路径不得写成已验证。你的报告应让后来者知道哪些结论来自数学、哪些来自 CPU 实验、哪些仍待目标硬件测量。

### 十五点十一　源码阅读的逐行问题表

读输入投影时，问输出如何切成内容与门，切分轴是哪一个；读卷积时，问权重方向、padding 与缓存保存的究竟是卷积前还是卷积后数据；读 delta 时，问是否加 bias、是否 softplus、是否有下限或其他变换；读 A 时，问参数是否经过负指数以及保存 dtype；读 B/C 时，问它们是否随时间、batch、通道或组变化。

读 scan 时，问旧状态系数与输入写入系数分别是什么，不要把二者都概括成“离散化”；读输出时，问 C 沿哪个轴归约，D 支路和门的先后顺序；读返回状态时，问它是最后有效 token 的状态还是公共 padding 后的状态；读自动微分封装时，问哪些中间量被保存、哪些被重算、哪些 dtype 被转换。

把这些答案写成一张变量映射表，每行包含数学符号、代码变量、shape、来源与生命周期。这样即便官方函数把多个操作融合成一个调用，你也能知道需要对照哪些中间语义。若只记函数名，版本更新或内核替换后很容易失去方向；掌握数据流则能迁移到不同实现。

### 十五点十二　模型边界与研究判断

本章实现的线性系统、精确 ZOH toy、reference 风格 selective scan 和完整教学 block 是四个层次。它们共享一些概念，但不是可以互相替换的同一个模型。写实验报告时应准确命名层次，尤其不能把标量精确 ZOH 的结果当作官方 scan 的逐项 oracle，也不能把教学 block 的随机前向当作具名模型的训练复现。

闭源或未公开细节的模型，只能讨论已披露部分。若某报告只说使用混合状态层，却没有给出完整门控或离散化，不应凭另一模型的代码补全。可以提出明确标注的假设性教学版本，并说明它用于理解机制；不能把假设写成事实。这种边界感让你的工程结论更可靠，也便于后续随着新证据更新。

你最终应能从一个状态更新式判断它是否线性时不变、是否有固定卷积形式、是否能用固定规模仿射组合扫描、需要保存哪些推理状态，以及训练梯度怎样跨时间传播。掌握这些问题后，面对新的状态模型，你可以逐项拆解，而不需要依赖“它像 RNN”或“它比 attention 快”这样的粗略标签。

### 十五点十三　误差传播的手算与诊断阈值

设每步舍入或近似产生同方向误差零点零零一，旧误差保留系数零点五，初始误差零。前三步误差为零点零零一、零点零零一五、零点零零一七五，长期上界接近零点零零二。若保留系数零点九九，同样每步误差的稳态尺度接近零点一。两者局部误差相同，长期影响却相差五十倍。

这个计算解释为什么长时间尺度需要更细的数值预算。不能只在一个时间步上比较误差很小，就认定一万步后仍然可忽略。也不能仅凭长期差异较大就断言某个 kernel 公式错误，应该同时检查局部更新误差与系统放大因子。把两者分开，才能判断该修正公式、提高累加精度，还是调整测试阈值。

```python
for coefficient,limit in ((.5,.002),(.99,.1)):
    errors,_=scalar_recurrence(np.full(2000,.001),coefficient,1.,1.)
    np.testing.assert_allclose(errors[-1],limit,rtol=1e-7,atol=1e-9)
```

真实误差不一定同方向，可能部分抵消，所以这个例子是可解释的保守趋势，不是所有实现的精确误差曲线。多维耦合系统还要考虑矩阵方向与条件数，输入相关门又会让放大随时间变化。课程先掌握标量机制，再在实际模型中记录状态轨迹，避免把一个简单上界当成万能预测器。

### 十五点十四　输入验证的范围必须明确

第十课函数聚焦计算路径，实际模块接口还应拒绝空序列、非有限输入、错误 gate shape、错误 D 长度和不匹配初始状态。否则 NumPy 可能通过广播接受本不属于该模型的张量，或者在最后 stack 空列表时抛出与问题无关的异常。给出清楚的前置检查，可以让学习者把错误定位在输入契约，而不是深层算术。

尤其要区分正 delta 的模型要求与 delta 预处理。某些接口接收未变换的原始值，在内部加 bias 和 softplus；本章 selective_scan_numpy 接收已经为正的 delta。把未处理值直接传入会被拒绝，这是正确行为；若重复做 softplus，也会改变系数。每个边界应写明“原始门值”还是“最终步长”，避免同一操作做零次或两次。

同样，A 可以作为最终负率传入，也可以由 log 参数在 block 中转换。scan 接口不应该再次对已经为负的 A 做负指数，除非它的契约明确接收的是 log 参数。很多移植错误来自把参数化阶段和算子阶段混淆，而不是来自递推公式本身。变量命名最好体现这一差别，例如 log_a 与 a_rate。

### 十五点十五　训练复现不能只保存最终状态

恢复训练需要模型参数、优化器状态、随机数状态和数据进度；动态状态只在采用跨片段状态训练时额外保存。若普通训练把每条样本视为独立序列，就应在样本边界重置状态，不能把上一批随机样本的末状态带进下一批。否则模型会学习到数据加载顺序中的伪关联，验证也难以解释。

跨片段训练如果保留状态但截断梯度，应明确片段边界和 detach 规则。前向仍可能利用更早历史，反向却只覆盖当前片段，这是一种有意的训练近似。把它与完整序列反向比较时，输出可以相同而梯度不同。报告应分别说明，而不是只凭 loss 接近就宣称两种训练完全等价。

优化器参数组也要覆盖所有新增组件。若只把 scan 的 A/B/C 放进优化器，却漏掉输入投影、卷积或门，完整 block 仍会运行，但部分组件永远不更新。一个实用检查是在一次训练步骤前后比较每组参数是否有合理变化，并结合梯度范数判断零更新是否符合当前数据与结构。不能把所有零梯度都视为错误，也不能默认它们都正常。

最终的学习目标是把数学定义、执行路径和训练状态三者对应起来。你应知道某个数是固定参数、当前输入生成的系数、跨时间动态状态，还是为了反向保存的中间量。这个分类会直接影响缓存、梯度、checkpoint 与并行执行，也是下一章理解混合架构和深度残差生命周期的基础。

检查实现时还应记录时间轴的方向。把序列倒序送入同一个因果扫描，得到的是反向时间系统，不能与原输出直接比较；若设计双向编码器，可以分别运行正向与反向状态再组合，但它不再满足自回归解码的未来不可见条件。双向状态模型和因果生成模型需要不同的测试，不能共用一个“SSM所以无须mask”的假设。

状态更新本身按时间循环并不自动保证输入预处理因果；任何先看完整序列的归一化、卷积或参数生成步骤，都可能提前引入未来信息。
