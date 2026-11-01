# S19 把真实K/V放进页里：实现可验证的分页注意力

S18最后要求你把元数据页换成真实数值状态。本章给出完整参考解[`paged_attention.py`](../../labs/serving/paged_attention.py)，并逐段推导。你将完成真正的NumPy K/V缓存、页表寻址、数值COW、共享前缀、分块online softmax，以及与完整上下文前向的对照。它在CPU上验证算法与生命周期，不声称拥有GPU PagedAttention Kernel的性能。

## 1. 先划清本实现计算的模型

`PagedTinyDecoder`与S06的TinyAttentionBackend共享同一套随机初始化权重：一个embedding、一个attention head的Q/K/V投影、输出投影与简单残差。没有多层Transformer、位置编码、训练、复杂norm、MLP或真正语言能力。模型小不是为了省略缓存机制，而是为了让每一行数值都能追踪。

完整后端每次根据全部context重算K/V；分页后端只为新token计算K/V，再从已有页读取历史。两者应在浮点容差内给出相同的最后位置logits。我们选择float64作CPU参考，降低验证时的舍入噪声；这不表示真实服务应使用float64。

运行：

```bash
python -m labs.serving.paged_attention
python -m unittest tests.test_serving_tensor_kv -v
```

预期输出各位置误差接近机器精度、A/B分叉后的token历史独立、all_pages_returned=true。具体末位数字受NumPy/BLAS环境影响，不要把`1.11e-16`精确写死为跨平台验收条件；测试使用明确容差。

## 2. 数据布局：四维数组与逻辑页表

K数组shape为`[physical_pages,block_size,kv_heads,head_dim]`，V相同。假设8页、每页4token、2头、每头3维，则一份K有8×4×2×3=192个元素。若float64，每份1536字节，K+V共3072字节，忽略Python对象开销。

页表仍来自PagePool，保存请求逻辑页到物理页的映射。Page.data保留token ID元数据与有效长度，不承载浮点K/V；真实数值住在keys/values数组。这样“这个页有哪些位置有效”和“这些位置的数据是什么”分离，便于检查所有权。

逻辑位置p的物理地址由两次运算得到：`logical,slot=divmod(p,block_size)`，再`bid=tables[rid][logical]`，最后读取`keys[bid,slot]`。你应能在纸上画出p=6、block4、table=[3,1]时读物理页1槽2。这一步不涉及attention数学，却是整个正确性的基础。

数组在创建TensorKV时一次分配。release只归还元数据页，浮点数组仍占着内存，旧数值也可以留在无效槽位中。这模拟了预分配KV池的重要行为：逻辑释放不等于设备/CPU分配器把大数组还给操作系统。

## 3. append的输入契约与验证次序

append接收rid、token列表、对应K和V。若本轮有n个新token，K/V必须都是`[n,heads,dim]`。先做shape与finite检查，再改变页池。这保证明显非法输入不会先占资源再报错。

代码先记录old_size和old_tail，调用PagePool.append完成资源预检与元数据COW，然后检查逻辑尾页是否被换成新物理页。如果发生COW，复制原尾页已经有效的前valid_tail行K/V，再把新增行按逻辑位置写入新页面。

为什么复制发生在元数据append之后仍安全？因为PagePool已完成失败原子资源预检，而COW时旧页原来refs>1，去掉当前请求引用后至少还有另一拥有者，所以旧数值不会被此操作逻辑释放。这里是同步NumPy操作；GPU异步版本还需要额外完成事件，不能直接照搬时序保证。

如果n跨越多个页，循环对每个新位置做divmod寻址；这段Python循环适合正确性参考，不适合高性能GPU。优化时可以把同一页内连续位置合成批量slice copy，但应先保留相同契约与测试。

## 4. gather参考为什么有价值

`blocks(rid)`依次遍历页表，只取每页有效行。`gather`把这些小块拼成连续K/V。随后用普通attention公式计算：Q shape `[h,d]`，K `[t,h,d]`；分数shape `[h,t]`；softmax后与V加权，输出`[h,d]`。

这个版本显式拼接历史，额外分配内存，也会复制数据；它不是性能目标。但它把分页地址翻译和attention数学拆开：如果gather后结果已经错，先检查页表或copy；如果gather正确但online版本错，重点检查统计合并。一个慢而清楚的参考实现非常适合作为优化代码的判据。

本函数计算当前最后query对所有已经存在历史的attention，因此全部历史都可见，不需要额外未来mask。若要一次计算多个query，必须给每query建立因果可见范围，不能把所有query都看完整K/V。这个边界在“把decode函数扩成prefill函数”时最容易漏掉。

## 5. 为什么不能把各页attention结果直接平均

一页算softmax输出后已经按本页指数和归一化；不同页的归一化常数不同。把每页输出取平均，会人为给每页相同总权重，长度不同或logit范围不同就错误。

手算三个score对应权重1、2、3，value为1、4、7。完整输出为`(1×1+2×4+3×7)/(1+2+3)=30/6=5`。前两项作为第一页，页内结果为9/3=3；第三项作为第二页，结果7。简单平均得到5，在这个特例恰好一样，不能因此证明方法正确！把第三个value改为10，完整输出=(1+8+30)/6=6.5，而页均值=(3+10)/2=6.5仍恰好，因为两页总权重都是3。要构造真正反例，改变第三项权重为6：完整=(1+8+60)/9=7.6667，而页均值仍6.5。

这个细节刻意提醒你：测试数据可能因巧合掩盖错误。设计反例时要破坏被错误算法依赖的隐含对称性。随机数据通常有帮助，但一个手算反例更能解释问题。

## 6. Online softmax的状态推导

对已处理所有页维护每头三个量：最大分数m，按m缩放后的指数和l，按m缩放后的加权值和u。它们满足`l=sum(exp(s_i-m))`，`u=sum(exp(s_i-m)*v_i)`，最终输出u/l。

新页最大值为b，新全局最大`m_new=max(m,b)`。旧统计必须乘`alpha=exp(m-m_new)`，因为它们原来是按旧m缩放。新页权重`p=exp(scores-m_new)`。更新：`u_new=alpha*u+sum(p*v)`，`l_new=alpha*l+sum(p)`，`m=m_new`。

初始m=-inf、l=0、u=0。第一个有限页来到时alpha=0，旧项自然消失；后续页都采用相同公式。每个头有自己的m/l/u，不能把不同头共用一个最大值，因为它们是不同softmax分布。

数值例：第一页scores=[0,ln2]、values=[1,4]，m=ln2，l=.5+1=1.5，u=.5+4=4.5。第二页score=ln3、value=7，m_new=ln3，alpha=2/3，新l=1.5×2/3+1=2，新u=4.5×2/3+7=10，输出5。你可以把页顺序交换，数学上应相同；浮点末位可能不同，所以测试用容差而非字符串比较。

## 7. 与Kernel卷的连接

本实现每次在Python中处理一页，使用NumPy einsum；GPU版会把query/key/value tile映射到线程、warp、shared memory和寄存器，并可能融合加载、归约与写回。online统计公式仍然是数学核心，但线程布局、并行归约和边界mask需要另外设计。

PagedAttention还需根据页表从不连续物理块读取，而FlashAttention通常强调分块计算与避免完整attention矩阵落HBM；两者关注点有交集但不是同义词。你现在能明确说出：分页是逻辑历史到物理存储的映射，online softmax是分块归一化合并，两者可以组合。

后续写Triton Kernel时，先用本CPU实现生成小规模参考。覆盖block1/2/4/7、长度1/3/8/13、多个head、非整页尾部、页表乱序。不要只测试物理页刚好连续的情况，否则根本没有验证页表寻址。

## 8. 真实数值COW与前缀复用

元数据COW正确但忘记复制K/V，会让新分支前缀读到零或旧数据。测试`test_cow_copies_tensor_prefix`不仅检查token列表，还比较实际K数组。A的旧前缀必须不变，B的前三行必须与A相同，B新增行必须是自己的值。

前缀缓存也不仅是把computed跳到某个值：它必须同时把对应物理页引用附加到新请求表。`test_prefix_cache_reuses_actual_tensor_pages`先写入四位置K/V、publish、释放旧请求，再让新请求attach并读取相同数值。若只改计数不改页表，测试会直接失败。

模型身份隔离仍然重要。我们的示例只用同一个权重对象；如果你换seed得到另一套Q/K/V投影，不能共享原状态。把新权重命名为另一个namespace，并加入“相同tokens、不同modelseed不得命中”的测试，才算实现模型版本边界。

## 9. 无效尾槽与复用测试

预分配数组可能保留旧数据，所以读取范围必须来自有效长度。测试故意把最后页未使用行写成极大值，如果attention误读这些槽，它会产生明显错误。正确结果完全不受影响。

另一个测试释放A三位置状态，然后同一物理页给B写一个位置。B只允许读取这一行，不能读到A残留的两行。这比每次release把整页清零更能揭露边界错误；清零可能让错误结果“看起来没那么离谱”，却仍然发生了越界语义读取。

生产系统为了隔离也可能清零某些新块，但清零是额外保护，不替代正确mask与长度。你要分别验证数值正确性和数据隔离需求，不能把二者混在一个“清空就行”的补丁里。

## 10. 参考实现的限制与可完成升级

目前只有单层、同步CPU、最后query、固定模型、不含RoPE。扩展多层需要每层独立K/V或cache group布局；加入RoPE要在写K前应用对应位置变换，并保证恢复时位置一致；加入GQA要把query头映射到KV头；批量prefill要加因果mask；GPU异步需要deferred free。

这些限制不是把任务扔到未来：本课程模型卷和算子卷已有相应数学与实现，你可以把它们作为明确接口组合。先完成一个升级，并保留“完整无缓存前向↔增量缓存前向”的数值对照，再加入第二个。多项同时修改后出现差异很难定位。

## 11. 练习与参考验收

1. **页表置换。** 手工安排逻辑页落在非连续物理ID，比较gather与连续参考。正确输出不应依赖ID大小顺序，只依赖table顺序。
2. **构造错误平均。** 用权重[1,2,6]、value[1,4,10]证明页内输出平均错误。再写一个错误函数，确认测试将它判失败；这就是数值算法的负向控制。
3. **多头状态。** 两个头使用不同score范围，一个约1000、一个约-1000。各自减最大后都应有限。若用跨头共同最大，某头可能下溢，测试应能发现。
4. **COW跨两页。** 共享尾页有3行、block4，追加6行，检查旧有效K/V复制、新页写入和原请求不变。资源不足时元数据与数值数组都不变。现有测试覆盖基本失败，你应补多页边界。
5. **增量等价。** 对随机token序列逐位置调用PagedTinyDecoder.append，与完整TinyAttentionBackend.logits比较。记录rtol/atol和dtype。改变seed只改变权重，不应破坏等价性质。
6. **实现GQA。** query头4、KV头2，规定query头0/1读KV头0，2/3读KV头1。用显式repeat的慢参考对照按索引读取版本。验收包括映射错误的负向用例，而不是只测两个KV头恰好相同的数据。
7. **接入调度。** 给Engine增加一个独立的incremental backend协议，由request ID和本轮Work驱动append，而不是继续调用全context logits。参考设计要在prefill chunk完成时暂存最后logits，抢占时清理或恢复数值cache，prefix命中要同步引用。保留现有Backend接口作为另一模式，避免悄悄改变根课程集成项目。

完成后，你已经拥有一条从请求元数据、页池、真实K/V、分块attention到logits的可执行链。接下来把同一组契约映射到GPU Kernel和真实框架，才能清楚知道自己优化了哪一步、还保留了哪些正确性保障。
