# 真实源码研读路线与固定版本

核查日：2026-11-01。以下commit来自本次实际读取的GitHub元数据；文件内容及主要符号也已读取。它们是本课程的**阅读快照**，不是宣称最稳定发布版，也不是本机已构建运行的依赖锁。更新时先建立新快照，再比较接口和测试，勿悄悄把旧报告的链接改为main。

| 项目 | 固定commit | 入口 |
|---|---|---|
| Triton | `0861b90f695e3c866f6f7eadd4be8339378ab017` | [Softmax](https://github.com/triton-lang/triton/blob/0861b90f695e3c866f6f7eadd4be8339378ab017/python/tutorials/02-fused-softmax.py)、[GEMM](https://github.com/triton-lang/triton/blob/0861b90f695e3c866f6f7eadd4be8339378ab017/python/tutorials/03-matrix-multiplication.py)、[Attention](https://github.com/triton-lang/triton/blob/0861b90f695e3c866f6f7eadd4be8339378ab017/python/tutorials/06-fused-attention.py) |
| CUTLASS | `e13f5cd4142c43efc37c2f6339923eb4a57afb99` | [Hopper TMA/WGMMA collective](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) |
| FlashAttention | `4616f5116d33e3ba3cbb71d582caffcfcb49649c` | [Hopper Softmax](https://github.com/Dao-AILab/flash-attention/blob/4616f5116d33e3ba3cbb71d582caffcfcb49649c/hopper/softmax.h) |

## 路线一：先做自己的Softmax，再读成熟实现

完成W1后，定位 `softmax_kernel` 与Python包装。第一遍只比较数学链：最大值、指数、分母、输出。第二遍比较布局：真实代码是否传入行步长、如何安排多行、如何决定Program数。第三遍才阅读资源与性能配置。每次都给出“我自己的版本省略了什么”和“省略会影响什么输入或性能”的对应关系。

任务A：写一张输入/输出契约表，至少包含dtype、设备、shape、stride、空维度、NaN、全无效行。没有看到明确支持的项写“未证实”，不要把自己的期望当源码行为。

任务B：挑一个非连续输入，静态追地址表达式。若官方包装仅处理行stride而默认列stride为1，说明它能处理的布局范围就不是任意二维视图。以实际代码为准，不能用“传了stride”四字概括全部布局。

任务C：把一个优化机制拆成假设。例如重复处理多行可能改变Program数量和资源利用；你的验证应分别看正确性、短行/长行性能和启动开销，而不是只跑官方默认benchmark。

## 路线二：GEMM的地址与Program顺序

先定位 `matmul_kernel`。记录M/N/K、每组stride、输出Tile尺寸和K块尺寸。对自己的3×7乘7×5例子，取一个Program手算它的输入地址。即使生产Tile远大于小矩阵，也可通过mask推理哪些lane有效。

随后阅读Program分组顺序。工作排序可以影响相邻Block对L2的复用机会，但不改变每个C元素的数学定义。任务是证明新顺序仍覆盖每个输出Tile一次，再用局部模型解释缓存收益。若不能证明覆盖，先不要讨论性能。

自动调参不是替代理解。列出候选配置、调参key和选择结果；说明新shape是否复用旧配置、首次运行是否包含搜索成本。不要把一次调参找到的最佳Tile推广到所有GPU。拓展项目：保留三个可解释配置，构建小而清晰的shape矩阵，比较而不是盲目扩大搜索。

## 路线三：Attention先读状态，再读布局

完成W3后，定位 `_attn_fwd_inner`。对应m/l/u与真实代码中的统计量，确认指数底数和缩放。然后看外层forward如何建立Q Tile、如何遍历K/V、如何区分因果阶段。最后看backward如何使用保存的统计量或重计算。

阅读时必须把数学归约轴与分布在线程上的物理数据布局分开。真实累加器可能有一套硬件片段布局，不直接呈现为整齐二维数组。遇到reshape/permutation，不要马上认为数学转置；先判断它是在重解释片段还是改变逻辑矩阵。

验收任务：写一份“同一行注意力从第一个K块到第二个K块”的数据追踪，包含最大值更新、旧状态缩放、分母更新、输出累加、最终归一化。再解释某个全mask或极端值分支为何存在。

## 路线四：CUTLASS的异步协作协议

在固定Hopper文件里，先查 `producer_acquire`、`consumer_wait`、`consumer_release`。它们分别与缓冲区获取、数据就绪等待和消费完成复用相联系；具体完成协议还要结合TMA barrier与矩阵操作提交/等待阅读。不要把方法名的直觉当完整同步证明。

追一轮生产者：获得可写stage→取得对应barrier→发起A/B搬运→推进写状态。追一轮消费者：等待可读stage→发起矩阵工作→等待必要完成→释放可复用stage。对每个阶段记录共享buffer索引与代际。若释放发生在矩阵仍读取旧buffer时，就是数据竞争风险。

注意真实实现可能跨多个MMA批次再释放、流水线预填充与尾部排空单独处理。任务不是把所有代码压成四句伪代码，而是在伪代码旁标出哪些机制维护了异步完成和资源约束。

完成这一路线需要H100验证时，再使用分配的H100。个人5070Ti可完成C++、索引、普通Triton和数值实验，但不能用它的成功推断Hopper专用路径已通过。

## 建议的源码阅读交付格式

每个任务一份短报告：固定commit；文件与符号；输入输出契约；调用关系；手算trace；你的实现与真实实现差异；一个可运行测试；未验证项。引用少量必要符号与片段，保留上游许可和链接，不复制整仓库进课程。若后续提补丁，先写最小失败案例和测试，再描述修复；学习笔记本身不等于生产贡献已经被证明安全。
