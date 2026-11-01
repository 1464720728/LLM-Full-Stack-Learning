# 固定版本源码工作坊：从教学模型跨到真实仓库

先完成M05、M06、M11、M12和M17。这里不要求安装三个巨大的工程，更不要求下载权重。第一轮只读取固定版本的文件，用本卷CPU数组验证你读到的语义。第二轮才在独立环境装项目，运行该项目自己的小规模测试。这样每一步都有可核对的证据。

本工作坊在2026-11-01实际读取了以下版本。链接中的长串是commit SHA；它标识本次阅读的代码快照，不等于推荐安装的最新发布版本。若你研究更新版本，保留旧报告，再写差异，不要把笔记里的链接直接改成`main`而丢失依据。

| 路线 | 固定来源 | 先回答的问题 |
|---|---|---|
| Llama decoder | [modeling_llama.py](https://github.com/huggingface/transformers/blob/ca8c871ede62b521a75a2ddfc5e1968ac9c8e097/src/transformers/models/llama/modeling_llama.py) | q/k/v投影后是什么维度，在哪一步接cache？ |
| Mamba selective scan | [selective_scan_interface.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py) | 哪个量依赖时间，哪个轴是状态而不是token？ |
| Attention Residuals | [README.md](https://github.com/MoonshotAI/Attention-Residuals/blob/85e22310fe5ee860b4a023de312d791de8a5a5e6/README.md) | softmax在来源深度轴还是token轴，block何时封存？ |

准备一份`source-notes.md`。每个观察写四列：函数名、张量形状、你自己的等价数学表达、它与课程实现的差别。写“用了RoPE”“用了SSM”不能算一行有效笔记；必须进一步回答哪个输入被变换、输出交给谁、何时产生或修改state。

## 第一站：先把Llama的attention骨架读通

打开固定文件，搜索`class LlamaAttention`。不要从文件第一行开始背所有导入。先画出一次调用的输入输出：hidden_states、position_embeddings、attention_mask、past_key_values进入，更新后表示与可选attention权重出来。遇到注解里的`Cache`，先写“外部状态接口”，等读到`.update`才追它的实现；这样不会在几十个辅助类之间迷路。

设batch为2、当前处理token数为3、hidden为8、query头为4、KV头为2、head_dim为2。输入是`[2,3,8]`。q投影输出8通道，k和v各输出4通道。源码的`hidden_shape`最后两个位置是`-1,head_dim`：三个投影共用这段shape代码，但`-1`根据各自元素总数推导，因此q得到4头，k和v得到2头。若把三者都写成4头，会让k/v变成每头1维；若最后一维坚持2，reshape会报错。这是读代码时必须用具体数字核对的第一处。

接着看`transpose(1,2)`。q从`[2,3,4,2]`变成`[2,4,3,2]`，它把head放到矩阵乘法的batch维中，让末两轴成为token与head_dim。k/v相应为`[2,2,3,2]`。课程没有batch轴，`split_heads`得到`[H,T,d]`；去掉真实代码最前面的B后即可对应，不能因为两个实现轴数不同就认为attention定义不同。

请把一次具体元素地址写出来：投影结果`q_raw[b,t,h*d+j]`对应`q[b,h,t,j]`。这比背“reshape后transpose”更可靠。若你交换reshape与transpose，元素分组会不同；张量最终形状即使碰巧相同也未必语义相同。用递增整数填入q_raw，然后对照每个元素，是最廉价的布局测试。

RoPE发生在cache更新之前。原因是缓存的key应当带有它被创建时的绝对位置；后续query到来时，过去key不用重新旋转。该文件把q/k送入`apply_rotary_pos_emb`，v不进入该函数。之后`.update(key_states,value_states,layer_idx)`将本轮K/V交给缓存。单步解码时，本轮query的token维是1，而返回的key/value可能已有前缀长度加1个位置。不要把“所有数组T都必须相同”写成断言：cross attention与带cache的self attention本就允许query和key长度不同。

cache update的层号也不是装饰参数。每一层投影不同，每层缓存必须独立。把两层写到同一个桶，有时不会马上发生shape错误，输出却被其他层的历史污染。本卷`TinyDecoder.step`里的cache列表按层对应，真实接口用layer_idx找到对应层。你的映射表应明确两个接口的同一责任，而不是只抄同名变量。

## 第二站：同名RoPE为何不能直接交换坐标

该固定版本的`rotate_half`把最后一维分为前后两半；课程`core.rope`使用相邻偶奇坐标成对。两者可以表示同一组二维旋转，但坐标布局不同。对于4维向量，课程布局可记为`[a0,b0,a1,b1]`，分半布局可记为`[a0,a1,b0,b1]`。把坐标置换后再做相同频率的旋转，才能比较输出。

手算最简单的90度旋转。相邻布局输入`[1,2,3,4]`，两组都用90度，结果为`[-2,1,-4,3]`。分半布局的同一逻辑向量是`[1,3,2,4]`，旋转结果为`[-2,-4,1,3]`；再置换回来，得到课程结果。若未经置换就把同一个`[1,2,3,4]`分别交给两个布局，差异是预期行为，不能据此宣布一方错误。

这还解释了为什么加载外部checkpoint时，单看参数形状往往不够。q/k投影权重把特征送到哪些输出坐标，与RoPE布局相互配合；你改变坐标约定时，相关权重也需要一致变换。测试一个旋转函数的范数保持，只能证明它是一种旋转，不能证明它与目标checkpoint的布局一致。

练习：独立写两个4维版本，一个用相邻pair，一个用分半pair；加入固定置换，比较50组随机向量与不同位置，最大绝对误差应接近浮点舍入。然后故意移除置换，记录一个不相等的反例。参考解思路是先把偶坐标收在前半、奇坐标收在后半，旋转后按原坐标散回。这个练习不需要真实模型权重，也不需要GPU。

## 第三站：从eager attention追到残差和语言模型头

读`eager_attention_forward`时，先标出四件事：KV如何配合query头数、点积如何缩放、mask在哪里加、softmax沿哪一轴。该固定版本softmax使用float32计算后转回query的dtype；这说明实际推理中的矩阵类型与归一化累积类型可能不同。课程全程float64主要服务数值对照，不是生产精度选择建议。

然后返回`LlamaAttention.forward`，看attention接口的选择。相同模型可以通过不同backend计算attention。模型定义相同，不保证浮点误差、支持的mask形状、是否返回权重以及性能相同。你刚读通的是eager路径；若切换backend，需要重新查该接口的mask语义，不能把eager里的每个中间张量都当作融合kernel必须物化的张量。

接着搜索`LlamaMLP`。先记录gate、up、down三投影的输入输出维度，再读激活函数从config选择这一点。课程`swiglu`明确固定SiLU；真实类从配置取hidden_act。名字叫LlamaMLP本身不等于你已经读取并确认了每一个config字段。对一个具体checkpoint作报告时，代码和配置必须同时有依据。

最后看`LlamaDecoderLayer`。第一段保留residual，对输入做norm，再attention，再相加；第二段再次保留新的residual，做norm、MLP、再相加。注意两个norm的输入不同。若你把同一个初始norm结果同时喂给attention与MLP，就成了另一种并行残差结构；它可能合理，但不是这里读取的串行结构。M17的trace把`after_attention`与`ffn_norm`分开打印，正是为了抓住这类“每个组件都在，连接却接错”的问题。

阅读`LlamaForCausalLM`时，区分语言模型头输出logits与loss。在源码中找到传入标签的位置、哪些位置的logits会被保留，再追loss函数。不能仅凭一个forward返回loss就断定shift发生在当前位置；需要顺调用链找实际shift。课程`tiny_lm`在造样本时明确构造target，而一些框架在loss入口平移标签。把两处都shift会错两位，一处都不shift则可能学成复制输入。

本路线提交物：一张包含embedding、每层attention、每层MLP、final norm、LM head的维度表，以及一张prefill/step状态表。每个表格格子必须有函数名或自己的实验数据支撑。

## 第四站：读Mamba时先读reference，不先读CUDA

打开`selective_scan_ref`。先把字母歧义消除：该源码的batch尺寸和输入矩阵B会使用同一个字母的不同语境；笔记中用`batch_size`和`B_param`区分。输入u的轴序是`[batch,channel,length]`，不同于课程常用的`[length,feature]`。A的形状是`[channel,state]`。state不是过去token的列表；每个时间步把旧状态压缩更新为新状态。

从最简单路径开始：实数A，固定二维B和C，无D，无z，delta已经给定正值，不额外加bias或softplus。源码先构造每一步的乘性系数`exp(delta*A)`，再构造写入项，循环中“旧状态乘系数，加本步写入”，输出通过C聚合状态轴。这与你在M11写的递推共享结构；shared structure并不表示每个系数都一样。

特别核对离散化。课程`selective_ssm`示范连续一阶系统的exact ZOH输入系数；此处reference的写入项采用delta、B、u相乘。取A=-1、delta=1、B=1、u=1、初始状态0：课程exact ZOH一步得到`1-exp(-1)`，约0.6321；该reference在所述简化输入下得到1。两者对旧状态的衰减都是`exp(-1)`，对新输入的离散化不同。这是一个必须能说清的边界，不能拿课程函数直接当官方reference逐项比较后把合理差异当bug。

两步恒定输入下，reference的状态为1和`exp(-1)+1`，exact ZOH状态为`1-exp(-1)`和`1-exp(-2)`。把两条序列都手算，再改delta为0.1，观察差异如何变化。它能训练你区分“来自连续系统的直觉”和“具体实现采用的离散更新”。源码写什么，数值对照就实现什么；理论推导也要标清对应的离散化条件。

第二遍打开variable B/C路径。B、C现在可以随时间变化，使“看到一个输入后如何写入、如何读出”受当前内容影响。请先只增加可变B，保持C固定；验证只有写入不同。再只增加可变C，验证存储状态可以相同而输出不同。最后增加两者。若一次把所有分支都打开，很难判断差异来自布局、广播还是动态参数。

第三遍看D和z。D提供输入直达输出的项，z经过门控作用于结果；这两者不应随意塞进state递推后声称数学等价。输出门控只改变被读出的结果，而状态内门控会改变将来时刻能读到的历史。用一个z在中间时刻接近关闭、后来重新打开的序列，可观察两种设计对未来输出的影响。

最后看`SelectiveScanFn`与`selective_scan_fn`。这一层涉及自动微分接口、连续内存要求、可选编译扩展和保存供反向使用的状态。不要因为reference能在小数组上跑，就推断官方CUDA扩展已经在你的GPU上可用。尤其本卷只执行了独立NumPy实现，未执行该固定版本的GPU代码。进入算子卷时才建立自己的环境矩阵与性能证据。

本路线提交物：四个有手算答案的输入，覆盖固定参数、可变B、可变C、D/z；每例写出首两步state和output。用二维状态让你必须求和，而不是永远停留在一个标量上。

## 第五站：Attention Residuals要顺着深度轴读

打开固定README中的Block伪代码。先数轴：每个来源表示是`[B,T,D]`，多个来源堆起来是`[N,B,T,D]`。N是当前可访问的深度来源数量。norm在特征维上处理这些来源，单个可学习pseudo-query与每个来源key算一个分数，得到`[N,B,T]`。softmax沿N，结果再对来源值加权，回到`[B,T,D]`。

因此同一个位置的不同层历史在竞争权重；这里没有引入一个`[T,T]` token连接矩阵。序列中第7个位置的AttnRes聚合，直接读取的是第7个位置的多个深度来源。那些来源本身可能早已通过token attention带有上下文信息，但那是另一个组件完成的混合。区分两种轴，才能知道缓存规模、mask责任和未来信息从哪里可能泄露。

第二个观察是key和值并不等同：用于评分的表示经过norm，用于加权的值保留其相应来源表示。若把V也替换为归一化后表示，你改变了可传递的幅度信息。课程`attention_residual`保留这个结构，测试全零query产生均匀来源权重，可用来理解这个最基本的操作。

第三个观察是partial block生命周期。README伪代码同时维护已完成的blocks与当前partial_block；在block边界封存当前partial，然后让新block从新的子层输出开始累加。层内attention和MLP各有一次AttnRes聚合；block大小的计数约定涉及这两个子层。因此“每两层一个block”与“每两个子层一个block”不是同一配置。写状态机时必须用自己明确的单位命名，不能只把变量叫size。

课程`advanced.block_depth_aggregate`只展示来源快照和深度聚合，更新由外部输入提供。随后进入[完整Block AttnRes Decoder工作坊](WORKBOOK_ATTNRES.md)：`TinyBlockAttnResDecoder`已经把实际attention、MLP和按子层计数的边界调度完整接起来，附full/step、因果性、来源数量和快照测试。先独立写每步来源清单，再对照完整参考实现；原有TinyDecoder保留为基准。初始化embedding封存与最终深度读出约定都显式说明，不将教学输出头等同任意官方checkpoint。

验收至少包括：所有来源shape一致；零query给出均匀聚合；改变一个来源只影响其能到达的后续深度；来源数量遵守你的block策略；已有历史快照不会被原地修改。最后一个尤其容易被Python列表误导：append一个数组引用后继续原地累加，会把已封存来源一起改掉。课程数值函数生成新数组，你自己写状态管理时仍需检查别名。

不要把AttnRes、普通token attention、MLA和mHC写成同义词。它们分别处理不同对象或连接方式。mHC的流混合矩阵即使也经过某种归一化，其多流状态和约束也不等于此处单一深度来源softmax。M12中的Sinkhorn实验只能证明你理解一种矩阵约束过程，不能作为完整mHC实现的交付物。

## 六个验收题与参考判断

1. **q与k共用`view`模板为什么头数不同？** 输出通道数由不同投影决定，`-1`独立推导。参考数值q是8通道→4×2，k是4通道→2×2。若答案只说“因为GQA”而没有维度推导，不通过。
2. **cache应存旋转前还是旋转后的K？** 本次读取的Llama路径存经过RoPE的K；缓存中的位置固定。其他实现可以采用不同设计，但必须随之实现正确的位置处理，不能混用两套假设。
3. **两种RoPE都保持范数，能否直接加载同一权重？** 不能据此判断；还要核对pair布局、频率、缩放及投影通道对应。给出4维置换例子才算完成。
4. **ZOH与所读scan输入写入是否一致？** 本工作坊的A=-1、delta=1例子分别是0.6321与1。差别来自输入离散化，不能通过扩大误差容忍度把它“测试通过”。
5. **AttnRes的softmax轴为什么不能改成T？** 改成T会让来源间竞争变成时间位置间竞争，改变模型定义；仅有相同输出shape不能证明等价。
6. **源码阅读报告里能否写“支持H100且更快”？** 仅从读取代码不能推出本地可运行与性能结论。必须另有环境、输入、warmup、同步计时和基线；本工作坊的阅读证据不承担这些结论。

当你能不看笔记重画三个模块的state和shape，并独立制造一个布局错、一个离散化错、一个快照别名错，再用最小测试定位它们，你才真正跨过了“看过源代码”的门槛。下一步进入推理卷和算子卷，把正确性reference变成可调度、可测量、可优化的实现。
