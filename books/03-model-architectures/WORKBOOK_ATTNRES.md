# 完整Block AttnRes Decoder工作坊：把深度聚合接进真实子层

[返回M12](12-hybrid-residuals.md) · [固定版本源码核对](SOURCE_WALK.md) · 前置：M05、M06、M12和M17

本工作坊交付完整、可运行的`TinyBlockAttnResDecoder`，不是只接受外部updates的聚合函数。每个attention增量来自实际GQA/RoPE，每个MLP增量来自实际SwiGLU；这些子层产生的输出会回到partial block，参与下一子层的深度聚合。它也能逐token生成logits，并维护每层真实的K/V缓存。

```bash
python -m labs.models.block_attnres_decoder
python -m unittest discover -s tests -p 'test_models_attnres.py' -v
```

运行只需NumPy和CPU，无网络、无权重下载。代码在[labs/models/block_attnres_decoder.py](../../labs/models/block_attnres_decoder.py)，测试在[tests/test_models_attnres.py](../../tests/test_models_attnres.py)。模型使用随机权重，验证的是前向语义与状态正确性；不具备已训练语言能力。

## 1. 先把四类对象放在不同的纸格中

你已有`TinyDecoder`：一个变量x沿着层前进，每个子层输出都加回x。现在把这一个变量拆成“已完成来源列表”和“当前块partial”。再加两种读操作：token attention沿历史token读取K/V，depth attention沿已完成来源与partial读取表示。它们在代码里各有明确位置。

| 对象 | 单次全序列forward的形状 | 何时增长或更新 |
|---|---|---|
| completed中的一个来源 | `[T,D]` | block封存时加入新数组 |
| partial | `[T,D]` | 当前块中的attention/MLP增量累加 |
| depth query | `[D]` | 每个子层各一个参数；本实验初始化为零 |
| 某层K/V cache | `[Hkv,T,d]` | 每处理一个token，序列轴增长 |

请注意最后两种“增长”完全不同。completed沿深度增长，cache沿时间增长。逐token调用时，新token重新走完整个深度，重新构造它自己的completed与partial；不能把上一token的深度列表接到下一token的列表后面。跨时间保存的是每层K/V，由`step`显式传入传出。

## 2. 计数单位先定死：每层有两个子层

实现使用`block_sublayers`，只接受正偶数，最小为2。一层中的attention编号为`2*layer`，MLP编号为`2*layer+1`。设3层、每块4个子层，总计6个子层。前两层的4个增量构成第一个计算块，第三层的2个增量构成尚未满额的最后一块。

为什么限制偶数？我们读取的固定官方伪代码在attention之前检查block边界，一层含attention与MLP两个子层。本教学实现明确保留这种边界位置。若允许奇数大小，却仍只在attention前检查，会让某些边界永远不发生。你当然可以设计允许MLP前封存的新版本，但那需要重新定义并测试生命周期，不能仅去掉一个输入检查。

初始化使用`completed=[]`、`partial=embedding`。第0个attention先读这唯一来源，读完才把embedding封存，并清空partial，让第一次attention的输出成为新计算块的起点。embedding因此只算一次。若初始化completed已经含embedding，partial也等于embedding，第一次聚合会重复计入它；形状仍然合法，权重却已经变了。

## 3. 在纸上走完3层的来源表

用E表示embedding，用`a0,m0,a1,m1,a2,m2`表示三个层依次产生的attention和MLP增量。这里每个符号都是`[T,D]`数组，且增量依赖前面聚合结果，不是预先独立生成的常数。

| 读操作 | 本次读到的来源 | 聚合之后的状态变化 |
|---|---|---|
| attention 0 | E | 封存E；partial开始为a0 |
| MLP 0 | E，a0 | partial变为a0+m0 |
| attention 1 | E，a0+m0 | 不封存，随后再加a1 |
| MLP 1 | E，a0+m0+a1 | 随后再加m1 |
| attention 2 | E，a0+m0+a1+m1 | 先读，再封存旧partial；新partial从a2开始 |
| MLP 2 | E，已封存的前4项，a2 | 新partial加m2 |
| 输出头 | E，已封存的前4项，a2+m2 | 聚合→RMSNorm→词表投影 |

所以CLI显示来源数`[1,2,2,2,2,3,3]`。第5个读操作位于边界，但它读的是封存前的两条来源；第6个读操作才出现三条来源。若你的来源数提前变成3，检查是否在聚合前把旧partial封存，又没有从当前来源中去掉它，从而重复读取同一块。

这张表对应代码`_execute`的顺序：`aggregate`，检查边界并snapshot，计算attention delta，更新partial，第二次`aggregate`，计算MLP delta，再更新partial。逐项把这七个动词写在函数旁边，比记住一整段代码更有用。

## 4. 一个能手算的边界时刻

暂时只看一个token、二维特征。在第三层attention之前，设E=`[2,0]`，旧partial=`[0,4]`，query为零。那么两个score都是零，混合结果为`[1,2]`。接着旧partial封存。为追踪生命周期，先假定attention子层对这个输入产生增量`[3,3]`，它成为新partial；MLP前的三条来源是`[2,0]`、`[0,4]`、`[3,3]`，零query读出`[5/3,7/3]`。

这里假定增量只是纸上定位state的临时办法；真正代码不会从外部接收这个假设值，而是调用`_attn(rms_norm(mixed),...)`计算实际投影、RoPE、带因果mask的attention和输出投影。MLP也调用实际`swiglu`。返回CPU实验后，用`forward(ids, return_trace=True)`取得每次mixed，与纸上来源均值比较；参数维度变大了，生命周期相同。

不要在新块开始时写`partial=old_partial+delta`。这会让旧块既保留在completed，又混进新partial，相当于让同一历史反复经过两个入口。边界后的第一项必须是新delta；非边界时才执行旧partial加delta。

## 5. 深度权重并非注意力模块的token权重

`sources`是`[N,T,D]`，调用`attention_residual`后weights为`[N,T]`。对每个token，N个来源权重之和为1。value保持原幅度，key按最后的D轴RMS归一化。每个子层query可以不同，最终输出头也有独立query。

本实验把所有query初始化为零，保证你有均值这个可手算参照，但代码不把query当常量：`depth_queries`是可修改参数数组，回归测试用非零随机query验证因果性和cache等价。你可以把其中某一行设为一个坐标方向，查看该子层如何根据各来源内容改变权重。请不要只看最终logits差异，要打印该层的weights，确定你修改的是深度选择而不是token匹配。

最终输出使用一次额外的深度聚合，再RMSNorm和词表线性投影。这是本微型语言模型明确给出的输出闭合约定，固定README片段主要定义层内Block逻辑；不能从这个教学头推断任意官方checkpoint的最后一层细节。只使用最后partial作为语言模型头输入会忽略completed；若做这种实验，须把它标成另一个读出变体。

## 6. 快照不是列表里多放一个名字

封存使用`_snapshot(partial)`：复制数组并设为只读。若只执行`completed.append(partial)`，然后下一步对partial原地`+=delta`，列表中的历史也会跟着改变。这个错误不会改变shape，所有代码甚至可能正常跑完，但深度来源已经不再是声明的历史快照。

本实现的partial更新使用新数组，封存仍明确复制，便于你以后修改实现时保持边界。trace也返回独立、只读的快照；它是诊断资料，不是可回写到模型状态的捷径。测试不仅比较数值，还用`np.shares_memory`检查封存来源与模型embedding、后续partial之间不存在共享存储，并验证再运行一次forward不会改变旧trace。

先理解这种保守实现，再在框架卷思考更省内存的版本。优化复制必须有所有权或不可变性依据，不能只因为“复制看起来很慢”就删除。此处没有把NumPy副本设计宣称为高性能实现。

## 7. 从全序列forward到逐tokenstep

`forward`和`step`共用同一个`_execute`，以免两套深度状态机逐渐写出差异。区别在输入token数量、位置和每层过去K/V。全序列forward一次处理T个位置；step只处理新token，位置由各层cache共同长度得到。

新token在第l层的深度聚合只使用这个新位置在较早深度的表示。随后token attention将该位置的q与本层历史K/V及新K/V匹配。因为深度聚合不混合token位置，而token attention有因果mask，改未来token不能影响已生成前缀。当前默认实验full/cache最大绝对误差约`1.5e-15`，属于这个小型float64输入的舍入量级；不同环境不应追求完全相同的最后一位。

step检查每层cache的头数、head_dim、K/V形状、有限数值和共同序列长度。新cache由拼接产生，不能改写调用者传入的旧cache。若你以后实现beam分叉或投机解码，这个所有权契约会比一次简单生成更重要。

## 8. 练习、参考解与完成标准

1. **改block大小。** 将默认4改为2和8，预测3层模型的来源数后运行。答案分别是`[1,2,2,3,3,4,4]`与`[1,2,2,2,2,2,2]`。最后一项是输出头。解释为什么block大于总子层数仍然有embedding和当前partial两条来源。
2. **真实均值验证。** 对trace的每条event独立计算`event['sources'].mean(0)`，比较mixed；同时检查weights按来源轴求和为1。参考：允许约`1e-12`误差，不能把T轴求和为1作为验收。
3. **不能只测零query。** 用固定seed填充非零depth_queries，再比较一次full与逐tokenstep；改末尾token，检查所有更早位置。参考：两项都应保持约`1e-11`内，非零query使内容相关选择真正参与测试。
4. **子层有没有真的接进模型？** 分别临时将每层attention的输出投影o、MLP的down置零，每次从原权重恢复后只改一处。参考：默认随机fixture的每处改动都改变logits；若某层完全不影响输出，应检查partial更新或最终读出是否漏掉路径。
5. **零增量极限。** 把全部o和down置零，默认3层、block4、零query时最终mixed是什么？答案是E/3，因为最后来源为E、一个全零封存块和一个全零partial。最终logits须由`rms_norm(E/3) @ output`计算；epsilon存在时不要声称一定逐位等于`rms_norm(E) @ output`。
6. **故障定位。** 故意在边界把旧partial加回新delta，然后运行来源计数与零增量测试。计数可能仍通过，因为错误是数值路径重复；应进一步对照手算边界fixture或记录delta前后的partial。这个例子说明shape/count测试与数值语义测试必须一起存在。

完成标准：你能独立重写状态机，解释每个封存时刻，指出哪个轴控制深度选择，保持full/cache等价，并证明封存数组不会被后续步骤改写。参考实现与全部验收代码已经提供；扩展练习是在这个完整模型上继续改造，不要求你先填一个隐藏的核心空缺。
