# S11 Prefill/Decode 分离：计算分工之后，状态怎样安全交接

Prefill/Decode 分离简称 PD：让一组资源主要计算提示的 prefill，另一组资源主要执行后续 decode。它不是简单启动两个服务再把 HTTP 请求转发一次，因为 decode 需要 prefill 产生的模型状态，而且这些状态可能分散在多层、多头、多 rank 和多个物理页中。本章从工作差异开始，逐步建立传输清单、目标预留、完成协议、取消与重试，再读固定框架的 connector 和接收队列。

先修 S02 调度、S03 页表、S05 在途释放、S08 测量与 S10 并行。所有数值传输示例先在 CPU 数组中完成；分析公式不声称真实网卡或 H100 性能。真实 RDMA、设备间拷贝、跨节点进程和框架部署需要额外环境，本章会明确哪些是已经执行的参考程序，哪些是有资源边界的后续实机练习。

## 1. 第一课：P 与 D 做的是同一个模型的不同工作

prefill 处理已有提示中的多个位置，为每层建立这些位置的 K/V，并产生最后位置的 logits。decode 根据当前历史每次推进新的位置，读取已有 KV，再产生后续 token。两阶段使用同一模型参数与数学结构，但输入行数、历史读取量和调度目标不同。不要把它们理解成两个完全独立、只通过文本交流的模型。

长 prefill 的矩阵通常有较多 token 行，能形成较大计算批次；短步 decode 每请求新增 query 很少，却要读取逐渐增长的历史。具体瓶颈取决于模型、batch、硬件、精度和 kernel，不能绝对说“prefill 只受算力限制、decode 只受带宽限制”。这只是理解工作差异的起点，真正判断要看测量。

共置服务把两类工作放在同一资源上。长 prefill 可能拉长正在流式输出请求的间隔，decode 又可能打断长 prompt 的连续处理。chunked prefill 和调度策略已经能缓解部分干扰；PD 进一步把资源池与调度目标分开，使 P 侧更关注输入处理与首输出，D 侧更关注稳定间隔和活动 KV 容量。

分离的代价也明确：增加至少一次状态交接、多个排队阶段、路由与失败处理，并在交接期间同时占用源和目标资源。低负载下，共置可能更简单且更快；网络慢或状态很大时，搬运成本可能超过减少的干扰。PD 是一种可以测量取舍的架构，不是默认应开启的性能开关。

| 维度 | P 侧常见关注点 | D 侧常见关注点 |
|---|---|---|
| 新 query 数 | 每请求可能很多 | 普通 decode 常为一个 |
| 历史状态 | 构建提示 KV | 持续读取并增长 |
| 延迟目标 | prompt 排队与首输出 | 输出间隔与长请求完成 |
| 容量压力 | 计算中与待发送状态 | 活动会话长期 KV |
| 失败影响 | 已算状态可能需重算 | 可能已有用户可见输出 |

同节点也可以 PD，跨节点也可以只是 TP/PP 而没有 PD。部署位置与执行阶段是两个维度。把“分离”自动理解为跨机器，会漏掉同机不同 GPU 池的实验；把“跨节点”自动理解为 PD，则会混淆层内通信与请求状态搬运。

## 2. 第二课：首 token 到底由谁生成，必须先定义

原始 Transformer 的最后 prompt 位置产生第一输出 token 的 logits。若 P 侧完成全部 prefill 并采样首 token，它需要把这个 token 和相关采样状态交给 D；D 下一步应把首 token 输入模型，计算它的 KV 并产生第二输出。此时传输的 prompt KV 长度与已生成输出数之间有明确差一关系。

另一种设计可以让 P 只传状态，由 D 在某个边界重算最后 prompt 位置或获取 logits 后采样首 token。两者都可能合理，但传输范围、TTFT 计时与随机数消费位置不同。不能让 P 已经采样一次，D 又把同一位置当作未采样重新抽样，否则输出历史与 RNG 会分叉。

设 prompt token 为 `[1,2,3]`，P 已计算三位置并得到首输出四。交给 D 的必要状态可以是 prompt 三位置 KV、output=[4]、computed=3，以及下一步逻辑位置三。D 计算 token 四后 computed 变四，再采样五。若误把 computed 写成四但实际只传三位置 KV，D 会相信一个根本不存在的缓存位置。

| 时刻 | 已知 token | 已有数值 KV | 用户输出 | 下一步 |
|---|---|---|---|---|
| P 开始 | 1、2、3 | 无 | 无 | prefill 三位置 |
| P 完成并采样 | 1、2、3、4 | 1、2、3 的 KV | 4 | 交接 |
| D 接收就绪 | 1、2、3、4 | 同上 | 4 已记录 | 计算位置3的 token4 |
| D 一步完成 | 1、2、3、4、5 | 1、2、3、4 的 KV | 4、5 | 继续或停止 |

如果首 token 已经对用户发送，D 交接失败后的重试不能把首 token 再发一次。输出序号与事件 ID 应成为协议的一部分，或者在重算时明确抑制已提交输出。S05 的“重算缓存不重播输出”和 S07 的“随机流归请求”在这里同时适用。

若 max_new_tokens 为零，不需要生成；若只需要一个 token，P 已生成后是否还值得建立完整 D 状态，要由实现决定。一个优化可以跳过不必要传输，但必须确认输出和清理已经完成。边界请求很小，却经常暴露系统是否把“所有请求都必须走完整流水”当成错误假设。

## 3. 第三课：传输清单需要描述语义，而不只是字节数

一份 KV 字节流只有在接收方知道布局时才有意义。最少要描述请求身份与代次、模型权重版本、adapter 身份、有效 token 范围、层列表、KV head 分片、head dimension、dtype、页大小、页内轴顺序、位置编码语义、源/目标映射以及完成状态。量化 KV 还需要对应 scale 等元数据。

模型名字符串不能唯一标识实际权重。同名模型可能更新 revision、加载不同 adapter 或使用不同配置；旧 KV 接到新权重上一般没有正确含义。tokenizer 版本影响怎样得到 token 序列，但传输协议最终应验证实际 token 身份与模型语义；不能只比较两段可见文本看起来一样就复用状态。

逻辑位置与物理页号必须分开。P 侧页表可能为 `[9,2,7]`，D 侧预留为 `[3,5,1]`，只要对应逻辑范围与数值布局正确，物理页号完全可以不同。把 P 页号直接写入 D 页表通常没有意义，因为两个池的内存地址空间和空闲集合独立。需要的是映射与数据搬运，而不是复制页号。

| 清单字段 | 例子 | 防止的错误 |
|---|---|---|
| request_id + epoch | reqA、代次3 | 旧传输完成污染新请求 |
| weight/adapter identity | 固定 revision、adapter digest | 不同模型状态混用 |
| token range | 半开区间[0,128) | 长度差一、遗漏或重复 |
| layer/head ownership | 层0–15、KV头0–3 | 分片错位 |
| dtype/layout | float16、明确轴顺序 | 同字节按不同格式解释 |
| source/destination map | 逻辑块到两端物理页 | 复制到错误页 |
| transfer_id/status | 唯一提交与完成事件 | 重复通知、提前读取 |
| valid length/checksum | 有效槽与完整性摘要 | 尾部垃圾、传输损坏 |

checksum 可以帮助检测意外损坏或布局不一致，但不替代身份校验和权限。相同字节哈希不说明它属于当前模型，可信的校验也不意味着远端允许你读取这份缓存。教学协议先处理数据完整性，真实多租户服务还要把认证与隔离接到控制面，不能依赖“这是内网”省略请求归属。

清单也要有版本。如果后续加入量化 scale、混合 SSM 状态或新的轴布局，旧接收方应该明确拒绝不支持版本，而不是按旧格式解析一半。可扩展协议的核心不是字段无限多，而是每个字段有定义、兼容条件有规则、未知情况不会被当成成功。


### 3.1 控制面与数据面分别携带什么

控制面交换的是“要做什么、给谁、现在做到哪里”，例如请求身份、目标地址、页映射、传输长度、完成通知和错误。数据面搬运的是 K/V 数值、量化元数据或其他模型状态。两者可以走不同通道，延迟与可靠性也可能不同。控制消息先到，不证明大块数据已经到；数据先到，也不证明接收方已经确认它属于哪个请求。

假设 D 已预留目标页并把地址发给 P，P 开始写入。随后 D 收到 ready 控制消息，但该消息通道并未与设备写入建立完成依赖，D 若立即运行 attention 就可能读到旧值。正确协议需要 ready 的含义包含相应数据可见保证，或在 D 使用前等待具体传输事件。一个字符串叫 ready 并不会自动建立内存顺序。

反过来，目标数据已写完但元数据槽还没有正确发布，D 也不能凭“数组里似乎不是零”猜完成。零本来可能是合法数值，部分写入也可能看起来正常。应有独立版本/状态标志，并且发布顺序与数据完成一致。SGLang 的 metadata gate 与 bootstrap_room 校验就是需要继续追踪的实际机制。

控制面重试还可能重复发送同一目标地址。若源端不知道这是重复通知，可能启动两次写入；若目标在第一次结束后已经复用，第二次写入会污染新状态。因此 request epoch、transfer ID 与目标缓冲区代次应能关联，接收端发布新地址之前必须结束旧写入责任。协议设计不能只考虑“正常网络下按顺序到达”。

教学实现可以先让所有控制与数据都在同一线程同步完成，这样更容易验证语义。随后故意把 ready、copy completion、cancel 分成可重排事件，检查哪些顺序应接受、哪些应等待、哪些应拒绝。先建立可检验的事件契约，再接入真实传输库，能显著缩小异步错误的排查范围。

### 3.2 有效范围、校验摘要与版本兼容

以十六 token 页为例，最后一页只有三个有效位置。清单应说明有效长度三，而不是让接收方把整页十六位置都当成历史。若传输整页，尾部可以作为 padding 字节存在，但 attention 的 seq_len 与逻辑页表仍必须限制可见范围。缓存发布也不能把未计算尾部声明为完整有效前缀。

校验摘要应该针对双方共同约定的有效逻辑范围与布局。如果 P 对物理整页含垃圾尾部算 hash，D 对有效三个位置算 hash，即使真实有效 KV 相同也会不匹配。反过来，只对第一个元素做摘要会漏掉大部分损坏。教学中可以把逻辑连续 gather 后的 bytes 与 shape/dtype 一起摘要，真实系统则需兼顾校验成本与设备同步。

量化格式的摘要还要覆盖 scale 等必要元数据；仅主 int8 字节相同不保证反量化结果相同。若跨不同 dtype 转换，字节相同不再是正确目标，应比较反量化后的数值误差或模型 logits。协议必须明确选择“字节保真”还是“数值等价到容差”，不能在失败后临时换标准。

版本兼容可以先采用严格相等策略：layout_version、模型身份、dtype 与分片参数不一致就拒绝。以后再为支持的转换建立显式转换器及测试。严格拒绝虽然不够灵活，却比静默误读更适合作为第一版工程。扩展能力应建立在可验证映射上，而不是在未知字段出现时全部忽略。
## 4. 第四课：用实际字节公式估算一份状态

统一全注意力模型的完整 KV 字节可估为 `2*层数*token数*KV头数*head_dim*每元素字节`。二表示 K 与 V。设三十二层、八个 KV head、head_dim 一百二十八、每元素二字节，单 token 为十三万一千零七十二字节，也就是一百二十八 KiB；八千一百九十二 token 对应一 GiB。

这是逻辑有效数据量，不必等于实际传输字节。整页传输可能包含尾部未用槽，量化格式有 scale，混合模型有其他状态，协议有元数据，重试会重复传输。若 P 与 D TP 不同，还可能发生重排或复制；若 D 已缓存部分前缀，可只传缺失范围。每项变化都应单独说明，不能把公式结果当真实网络计数。

```python
layers, tokens, kv_heads, head_dim, item_bytes = 32,8192,8,128,2
per_token = 2*layers*kv_heads*head_dim*item_bytes
total = per_token*tokens
assert per_token == 128*1024
assert total == 1024**3
block_size = 16
short_tokens = 8193
allocated_tokens = ((short_tokens+block_size-1)//block_size)*block_size
assert allocated_tokens == 8208
padding_bytes = (allocated_tokens-short_tokens)*per_token
print({"valid_GiB":total/1024**3,"tail_padding_bytes":padding_bytes})
```

一份一 GiB 状态在交接期间可能同时占用 P 源页与 D 目标页，另有 host staging 或转换缓冲区。最终 D 只持有一份，并不意味着交接峰值也只需要一份。高并发时，多份在途状态叠加，是 PD 容量计划中特别容易忽略的部分。

传输时间下界约为字节数除有效带宽，再加建立与同步成本。有效带宽不是网卡铭牌峰值，可能受协议、拓扑、PCIe、HBM 竞争与并发影响。GB/s 与 Gbit/s 差八倍，GiB 与 GB 也不同。报告必须保留单位，不要把一 GiB 在十 GB/s 上的时间简单写成恰好一百毫秒。

## 5. 第五课：完整生命周期需要区分七个事件

一个保守的交接顺序是：D 确认容量并预留目标；P 计算状态；两端确认清单与布局；提交传输；等待目标数据可读；D 接管并开始 decode；P 按协议释放临时源引用。实际系统可能重叠这些步骤，但依赖关系不能省略。尤其是“提交传输”与“目标可读”之间有一段在途期。

P 计算完成只证明源 KV 已经形成；发送 API 返回可能只表示命令进入队列；发送端完成可能说明源缓冲区不再被读取；接收端完成可能说明目标数据已写入；D 进入 attention 前还需要相应 stream/event 依赖保证可见。不同传输库的完成语义不同，不能把任意一个 done 当成全部完成。

| 状态 | 源引用 | 目标引用 | 允许 D 读吗 | 允许复用源吗 |
|---|---:|---:|---|---|
| 已预留、未计算 | 按准备状态 | 有 | 否 | 不能覆盖将使用的源 |
| P 已计算 | 有 | 有 | 否 | 否 |
| 传输已提交 | 有 | 有 | 否 | 否 |
| 目标确认就绪 | 按发送完成协议 | 有 | 是 | 满足源完成条件后 |
| D 正在运行 | 通常已释放临时持有 | 有 | 是 | 是 |
| 请求结束 | 无活动引用 | 待安全清理 | 不再正常读 | 已独立处理 |
| 取消但仍在途 | 保留必要引用 | 保留必要引用 | 不交付新结果 | 等待安全完成 |

表中的“有引用”是逻辑所有权，不必意味着操作系统新分配一次。源页也可能同时被 prefix cache 持有；释放传输引用后，缓存引用仍可保留。D 目标页可能已经预分配在大池中；归还请求所有权不意味着 cudaFree。沿用 S03 的引用账本，才能区分真正泄漏与合法保留。

交接可以看作所有权迁移，但通常存在一段重叠：源与目标都不能提前复用。不要追求“每一时刻只有一份所有者”而破坏数据传输；正确要求是每个在途读写都被某个有效持有覆盖，并且最终所有临时持有都能结束。


### 5.1 把正常交接走成一张逐事件账本

设请求甲需要两页 prompt KV，P 页号为六与九，D 预留页为二与五。P 的 prefix cache 还持有页六，因此开始时页六有缓存引用；请求计算与传输会增加各自责任。为简化表格，下表只列本请求与传输责任，缓存引用另外保留，不能在结束时一起删掉。

| 事件 | P 请求/传输责任 | D 预留/运行责任 | 数据事实 | 下一事件允许条件 |
|---|---|---|---|---|
| D 预留 | 尚未提交发送 | 页2、5被本请求占有 | 目标内容未定义 | 清单匹配 |
| P 计算完 | 页6、9有效且被持有 | 仍预留 | prompt 两页可发送 | 源计算事件完成 |
| 提交传输 | 页6、9不能复用 | 页2、5不能复用 | 可能只写了一部分 | 等待完成 |
| D 确认就绪 | 源释放需满足发送端契约 | 目标成为可读状态 | 逻辑内容一致 | 首输出/位置状态匹配 |
| D 接管 | 临时发送责任结束 | 运行责任继续 | 可执行下一位置 | 正常调度 |
| 最终完成 | 页6缓存引用可仍存在 | 运行引用安全归还 | 不再产生输出 | 终态清理通知 |

这张表能回答一个常见疑问：为什么传输完成后 P 的空闲页数没有完全恢复？因为页六仍有合法 prefix cache 引用。若你只用“空闲页等于开始值”判断泄漏，就会误报。正确检查应按所有权类别计算：请求引用结束、传输引用结束、缓存引用按策略保留。

再把取消插在“提交传输”之后。用户输出应停止，但 P 页六九与 D 页二五仍处于在途保护；完成/取消确认到来后，两端释放本次责任，D 不能将失败目标插入缓存。重复取消只改变一次业务状态，重复完成只结算一次提交。这个场景同时检查状态机、引用计数与数值可见性。

如果 P 在“计算完”之前失败，D 的预留目标没有接收到有效数据，应归还或等待明确恢复；若 D 在接管后失败，P 可能已经释放源，恢复需要重算或从别处缓存获取。失败位置决定可用证据，不同位置不能共用一个“重新发送最后消息”的修复策略。

### 5.2 谁负责最终回收：交接不能留下无人负责的间隙

所有权转移有两个危险方向。若 P 认为 D 已接管而 D 认为尚未接管，可能没有一方保留恢复需要的状态；若双方都认为对方会释放，可能永久泄漏。协议应明确哪一个确认使责任从 P 转到 D，以及确认丢失时谁保留、谁查询、谁重试。

一种保守设计是 P 在收到可验证接管确认前保留源，D 在发送确认前保证目标有效并记录 transfer ID。确认重复发送应安全，P 收到重复确认不会再释放一次。若确认丢失，P 可以查询同一 transfer 的终态，而不是生成一个无法关联的新请求。真实实现可以更高效，但需要提供等价责任保证。

资源上限会迫使系统处理长期未确认状态。可以设置隔离队列、限制新准入、重建连接或终止故障执行组，但不能无条件回收仍可能被远端访问的地址。超时后采取何种动作属于故障模型的一部分，应在实现与文档中明确，不应只写“异常时清理”掩盖风险与代价。
## 6. 第六课：取消、重复通知和重试不能靠一个布尔值解决

请求可能在 P 排队、P 计算、传输中、D 就绪或 D 生成中取消。排队时可以直接撤销未执行工作；计算或传输在途时，先禁止新输出，再等待相关访问达到安全释放条件。取消是业务终态，设备与传输清理可能稍后才完成。把取消处理写成“立即 free 两端所有页”会产生使用已释放内存的问题。

重复完成通知应幂等。同一个 transfer_id 的 ready 到达两次，不能重复增加目标引用；同一个 abort 到达两次，不能重复 free。乱序旧消息还需要请求代次：reqA 代次二的完成不能把代次三的目标标为 ready，即使字符串 request_id 相同。只保存一个 done 布尔值无法区分这些身份。

重试还要定义是重传已有源状态、重新 prefill，还是在 D 本地重算。重传需要源仍有效；重新计算需要恢复已知 token 与 RNG/输出提交状态；本地重算需要 D 有计算能力与预算。选择取决于失败位置，不能把所有错误都无限重试同一个操作。

若 P 已发送首 token，D 失败后重算必须抑制已提交输出，并从正确随机位置继续。若还没有用户可见输出，恢复空间更大，但模型版本与 prompt 仍需一致。用户看到的事件序号与内部 transfer_id 是不同身份：一个输出事件可以跨内部重试保持不重复，而传输每次重试可能拥有新 ID。

超时表示等待超过预算，不自动证明远端写入已经停止。安全协议需要取消确认、完成事件、连接失效后保证、代次隔离或其他明确机制。若实现选择超时后强制回收，它是在特定边界下作出恢复取舍，必须理解其假设，不能把“等够几秒”当成一般内存安全证明。

## 7. 第七课：写一个 CPU 数值交接参考，验证就绪与取消

下面程序真的复制 NumPy 数值数组，但不使用网络、DMA 或 GPU。Manifest 固定模型身份与请求代次，Handoff 持有源和目标，只有完成复制后才允许 decode 读取。传输中取消进入 draining，保留两端引用；完成通知结算后清理。程序刻意没有“超时自动判定安全”的分支。

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Manifest:
    rid: str
    epoch: int
    model: str
    layout: str
    token_count: int

class Handoff:
    def __init__(self, manifest, source, expected_model):
        if manifest.model != expected_model or manifest.layout != "THD-f64":
            raise ValueError("incompatible state identity")
        if source.dtype != np.float64 or source.ndim != 3:
            raise ValueError("expected float64 [token,head,dim]")
        if source.shape[0] != manifest.token_count or not np.isfinite(source).all():
            raise ValueError("invalid state range or values")
        self.manifest = manifest
        self.source = source.copy()
        self.target = np.full_like(source, np.nan)
        self.phase = "reserved"
        self.cancelled = False

    def _identity(self, key):
        if key != (self.manifest.rid, self.manifest.epoch):
            raise ValueError("stale request generation")

    def begin(self):
        if self.phase != "reserved":
            raise ValueError("transfer is not ready to submit")
        self.phase = "in_flight"

    def cancel(self):
        if self.cancelled or self.phase in {"done","cancelled"}:
            return False
        self.cancelled = True
        if self.phase in {"in_flight","draining"}:
            self.phase = "draining"
        else:
            self.source = self.target = None
            self.phase = "cancelled"
        return True

    def complete_copy(self, key):
        self._identity(key)
        if self.phase in {"ready","running","done","cancelled"}:
            return False
        if self.phase not in {"in_flight","draining"}:
            raise ValueError("copy was not submitted")
        if self.cancelled:
            # 此同步模型的完成事件证明没有剩余读写。
            self.source = self.target = None
            self.phase = "cancelled"
        else:
            self.target[...] = self.source
            self.source = None
            self.phase = "ready"
        return True

    def start_decode(self):
        if self.phase != "ready" or self.cancelled:
            raise ValueError("destination is not readable")
        self.phase = "running"
        return self.target.copy()

    def finish(self):
        if self.phase != "running":
            raise ValueError("decode is not running")
        self.target = None
        self.phase = "done"

kv = np.arange(24,dtype=np.float64).reshape(3,2,4)
m = Manifest("A",1,"tiny-revision-1","THD-f64",3)
h = Handoff(m,kv,"tiny-revision-1")
h.begin()
try:
    h.start_decode()
except ValueError:
    pass
else:
    raise AssertionError("read before transfer completion")
assert h.complete_copy(("A",1))
np.testing.assert_array_equal(h.start_decode(),kv)
assert not h.complete_copy(("A",1))
h.finish()
assert h.source is None and h.target is None

cancelled = Handoff(m,kv,"tiny-revision-1")
cancelled.begin(); cancelled.cancel()
assert cancelled.phase == "draining"
assert cancelled.source is not None and cancelled.target is not None
cancelled.complete_copy(("A",1))
assert cancelled.phase == "cancelled"
assert cancelled.source is None and cancelled.target is None
try:
    cancelled.complete_copy(("A",0))
except ValueError:
    pass
else:
    raise AssertionError("stale generation accepted")
print("numeric handoff, readiness, cancellation and stale identity checked")
```

source.copy 使参考拥有独立源数据，避免调用者修改原数组影响演示；真实零拷贝传输通常不会这样复制，而是持有源页直到完成。target 初始填 NaN，让提前读取更容易暴露；正确性仍来自 phase 检查，不是依赖 NaN 恰好出错。start_decode 返回副本，只为了防止外部随意修改内部状态，不代表真实 Runner 每步都复制整份 KV。

complete_copy 在正常路径实际赋值，在取消路径模拟“在途访问已结算”后释放。这个模拟事件由测试显式触发，不是经过若干毫秒自动成立。它验证协议状态与数值保真，不验证网络重排、进程崩溃或设备完成语义。你应把这些未覆盖维度写在报告中，再逐层增加真实传输实现。

这里没有采用多个布尔 source_held/target_held 去独立维护引用计数，而是用 None 与 phase 表达最小所有权。真实共享前缀需要更复杂引用，不能直接删除整个源页。把 Handoff 嵌入 PagePool 时，应只释放本传输持有的引用，不影响 prefix cache 与其他请求。

## 8. 第八课：两个队列与背压，为什么 P 快不等于系统快

PD 至少有 P 等待队列、传输/目标预留等待队列与 D 活动队列。若 P 每轮产生三个可交接请求，而 D 每轮只能接管两个，积压每轮增加一个。十轮后多十份状态，可能同时占用源页、目标预留和元数据。无界队列会把短期吞吐优势变成长期内存问题。

背压可以在 P 开始前预留 D 容量，也可以限制在途传输数与字节数，或者让 P 完成后等待有限队列。早预留降低“算完却无处放”的风险，但目标资源等待 P 时可能空闲；晚预留提高灵活性，却可能积压已算状态。没有一种策略无条件最佳，需要根据服务时间分布与容量测量。

只限制请求数可能不够。一份两百 token 状态与一份八千 token 状态相差很大，传输队列还应考虑字节或页预算。D 的准入也不只是当前 prompt KV，后续生成会继续增长，需要预留或动态管理输出空间。把 P 的 max_num_seqs 当成整个系统可承受并发，是把局部预算误当全局容量。

```python
def bounded_flow(rounds, produce, consume, capacity):
    queue = accepted_total = completed_total = rejected_total = 0
    rows = []
    for step in range(rounds):
        completed = min(queue, consume)
        queue -= completed
        accepted = min(produce, capacity-queue)
        rejected = produce-accepted
        queue += accepted
        accepted_total += accepted
        completed_total += completed
        rejected_total += rejected
        assert accepted_total-completed_total == queue
        assert 0 <= queue <= capacity
        rows.append((step,accepted,completed,rejected,queue))
    return rows,accepted_total,completed_total,rejected_total

rows,accepted,completed,rejected = bounded_flow(10,3,2,4)
assert rows[0] == (0,3,0,0,3)
assert accepted-completed == rows[-1][-1] == 4
assert rejected > 0
print(rows)
```

这个模型先消费旧队列再接受新工作，因此第一轮没有完成项；若换成先生产后消费，具体表格不同，报告应说明事件顺序。它只验证守恒与有界性，不模拟真实 prefill/传输/decode 服务时间。下一阶段可以把每请求字节数纳入准入，观察相同请求数下容量为何不同。

如果前端同时发六十四请求，而 P 活动上限三十二，后面请求可能在前端、P 队列或容量门前等待，取决于实现。D 不必等前三十二全部完成才开始；任何一个请求状态就绪都可能被接管。真实实现还可能分块或分层传输，但不能从“PD”名称自动推断支持粒度。


### 8.1 用字节预算取代“每个请求都一样大”的假设

设传输队列上限四请求，前三个各需要一 GiB，第四个只需要一 MiB。请求数看起来尚未超限，但源与目标重叠占用已接近六 GiB，再加 staging 可能更高。另一个四请求队列若每个只有一 MiB，资源压力完全不同。因此应同时限制条目数与字节/页数，防止大量小元数据或少量大状态分别耗尽不同资源。

字节准入可以维护 reserved_bytes、inflight_bytes、ready_bytes 三类账。预留进入 reserved，提交后移到 inflight，完成后移到 ready，D 接管后从传输队列账移交给活动 KV 账。类别转换不代表物理内存立刻减少，所以总容量计算要避免双计或漏计。每一步写守恒式，比事后看一个总显存数字更容易发现问题。

同一份源前缀被多个传输共享时，逻辑发送字节可以多次计入网络工作，而物理源存储只存在一份。容量账与流量账因此不同。若为了估算网络负载把每份发送都算字节是合理的；若把这些重复发送字节全部当作新增显存，又会高估源容量。必须标明统计对象。

公平性也可以按字节或预计传输时间考虑。纯 FIFO 让一个巨大状态占住传输通道，可能拖慢许多短请求；优先短状态又可能饿死长请求。可以采用分块轮转或年龄提升，但要保持每请求层/页依赖和输出顺序。PD 把 S02 的调度问题扩展到新的资源，不是消除了调度。

### 8.2 池比例是服务能力匹配问题

假设平均每请求 P 花二十毫秒，一台 P 理想每秒处理五十请求；平均每请求 D 持续五百毫秒，若有效并发与批处理使某台 D 每秒完成二十请求，那么一台 P 配一台 D 会积压。这个算例只是用服务率建立直觉，真实 D 吞吐依赖同时活动序列与输出长度，不能直接用单请求时长倒数代替批处理容量。

选择 P/D 比例时，应测相同工作负载下两池的可持续能力、传输能力与 SLO，而不是按 GPU 数平均切半。短输出业务可能 P 更重，长输出业务可能 D 更重，缓存命中变化又会改变 P 工作量。业务分布改变后，原先平衡的比例可能不再合适。

动态调整池比例还涉及权重驻留、模型加载、缓存迁移与连接重建，不是瞬间把一个标签从 P 改成 D。若共享服务器不能长期占卡，先用分析模型和 CPU trace 选择少量候选，再做短时实机验证。这个顺序能把昂贵 GPU 实验集中在真正不确定的参数上。
## 9. 第九课：用排队收益减去搬运成本

本地 [pd_break_even](../../labs/serving/analysis.py) 使用简化模型：共置 TTFT 等于共置排队加 prefill；分离 TTFT 等于 prefill、D 排队与传输之和。它暂时省略 P 排队、重排与重叠，用来让假设清楚可见，不是生产预测器。更完整的模型应把这些阶段逐项加回。

设 prefill 一百毫秒，共置排队三百，D 排队三十，一 GiB KV，建立成本一毫秒。有效带宽一 GB/s 时传输约一千零七十四点七四毫秒，分离明显更慢；十 GB/s 时约一百零八点三七毫秒，分离约二百三十八点三七；五十 GB/s 时约二十二点四七，分离约一百五十二点四七。共置均为四百毫秒。

```python
from labs.serving.analysis import pd_break_even

reports = [pd_break_even(100,300,30,1024**3,bw,setup_ms=1)
           for bw in (1,10,50)]
assert not reports[0]["split_improves_ttft"]
assert reports[1]["split_improves_ttft"]
assert reports[2]["split_improves_ttft"]
for bw,report in zip((1,10,50),reports):
    print(bw,report)
```

在这个模型下，分离改善 TTFT 的条件是减少的排队时间大于传输成本。若加上 P 池排队与布局转换，就要一起放在成本侧。若研究目标是保护正在 decode 请求的 ITL，PD 可能允许新请求 TTFT 稍差但整体交互更稳定；因此不能只用 TTFT 一个数决定是否值得。

传输与计算重叠可以缩短临界路径，但需要可独立推进的工作。例如 D 为甲 decode 时传乙的 KV，或 P 完成某层后开始传已完成部分。但如果 D 的当前 attention 必须等全部所需历史，就不能提前读未完成页。异步接口只是提交方式，真正收益来自依赖图中可并行的部分。


### 9.1 首输出来自 P 时，传输成本可能落在首个间隔里

前面的 pd_break_even 将“完成交接并可输出”近似作为首输出边界，适合用来研究该边界的排队与传输成本。若系统允许 P 一完成 prefill 就直接把首 token 发给用户，TTFT 可以早于 KV 交接完成；此时传输延迟可能主要体现在第一个 token 与第二个 token 之间的长停顿，而不是 TTFT。不能把同一公式无条件套到这两种协议。

设 P 在一百毫秒产生首 token，传输到三百毫秒才完成，D 再用十毫秒产生第二个 token。用户 TTFT 为一百毫秒，但首个 ITL 为二百一十毫秒。若只展示 TTFT，PD 看起来很好，实际用户却在第一个字后等待很久。S08 的原始时间线应保留每个输出时刻，尤其关注交接后的第一个间隔。

另一种协议把首 token 暂存，等 D 就绪后统一开始流式输出，TTFT 变大，但后续间隔可能更平稳。哪一种体验更好取决于产品目标，实验应同时报告首输出、首个间隔和稳态间隔。不要为了让某指标好看而改变流式策略却不告诉读者。

同理，P 端报告的“prefill 完成延迟”不是客户端 TTFT，D 端的“decode 单步时间”不是端到端 ITL。前端、传输与排队会把这些阶段重新组合。对 PD 做归因时，至少记录请求到达 P、P 开始/结束、传输提交/完成、D 准入、D 首步、用户首文本与最终完成。

### 9.2 分层传输怎样形成真正的流水

若模型有多个层，P 完成较早层的 KV 后，可能在继续计算后面层的同时发送已经完成的层状态。D 使用某层 attention 前只需该层对应状态就绪，而不是必然等待所有层一次性到达。接口中的 wait_for_layer_load 正是表达这种细粒度依赖的一种方式。

但是能否实际重叠，取决于 P 写入完成事件、传输读的安全范围、D 的执行顺序和带宽竞争。若发送前必须把所有层先打包到一个大缓冲区，重叠空间会减少；若传输与计算同时争用 HBM，单独测得的峰值带宽也不能相加。需要时间线证明某一段确实与另一段重叠，而不是只看到两个 async 参数。

分块传输还需要区分“页已经完整”与“整请求已经完整”。收到前几页可先登记，但 D 当前 query 若需要全部历史，就仍不能读取缺失尾部。层级就绪与 token 范围就绪是两个维度，清单应表达哪些层的哪些范围有效。一个全局 ready 布尔值无法支持这种细粒度流水而又保持正确。
## 10. 第十课：不同 TP、页大小与缓存命中怎样改变交接

P 的 TP 为二、D 的 TP 为四，KV heads 为八。简单完整 head 分片下，P rank 零持头零到三，rank 一持四到七；D 四 rank 各持两个头。传输需要把 P 零拆成目标零/一两份，P 一拆成目标二/三两份。若直接将 P 每 rank 缓冲区整体复制到同编号 D rank，另外两个 D rank 就缺数据。

若 P 与 D 页大小不同，逻辑 token 范围仍可一致，但物理打包不同。源页边界不能直接作为目标页边界，可能需要 gather/scatter 或中间转换。dtype 不同还涉及量化/反量化和数值误差，不能称为纯字节搬运。协议必须知道哪些组合支持，遇到不兼容应拒绝或走明确转换路径。

前缀缓存又引入局部命中。P 命中一百二十八 token，D 只命中六十四，D 仍需要缺失的六十四前缀与新增部分；P 不必重算已经命中内容，但源缓存引用必须保持到发送完成。若 D 命中更多，则可以减少传输，但双方要验证同一模型、位置与 token 身份，不能只交换一个长度数字就相信内容相同。

共享缓存与传输引用应独立计数。P 释放请求的活动引用后，cache 可能继续保留该页；传输完成释放自己的引用后，也不应把其他请求正在共享的页清掉。D 收到数据后是否插入 prefix cache，要等确认其状态有效；失败或部分传输的目标不能作为正常缓存发布。

混合 attention/SSM、MLA 或其他压缩状态可能需要不同清单与边界。标准 K/V 支持不自动证明所有模型状态都能转移。读取框架 connector 支持列表与实际状态接口，先用一个模型类型完成验证，再扩展。用户希望理解前沿，第一步正是知道哪些不变量可迁移、哪些模型结构改变了状态定义。


### 10.1 用数组实际完成二到四 rank 的头重排

下面把完整逻辑 KV 表示为 `[token,head,dim]`，用每个元素不同的整数检查重排。P 两 rank 各持四个 head，D 四 rank 各持两个。程序先按源 rank 取 head，再根据全局 head 身份送到目标 rank，最后拼回完整逻辑数组。这不是实际网络传输，却能验证一个容易被“直接拷贝”忽略的布局问题。

```python
import numpy as np

full = np.arange(3*8*2).reshape(3,8,2)
p_shards = [full[:,0:4,:].copy(),full[:,4:8,:].copy()]
d_shards = [np.empty((3,2,2),dtype=full.dtype) for _ in range(4)]
for source_rank, source in enumerate(p_shards):
    for local_head in range(4):
        global_head = source_rank*4+local_head
        target_rank, target_head = divmod(global_head,2)
        d_shards[target_rank][:,target_head,:] = source[:,local_head,:]
restored = np.concatenate(d_shards,axis=1)
np.testing.assert_array_equal(restored,full)
assert d_shards[2][0,:,0].tolist() == [8,10]
print("TP2 to TP4 preserves logical head order")
```

source_rank 只决定源分片覆盖哪些全局 head；target_rank 来自目标分片规则，不必等于 source_rank。global_head 是两边共同的逻辑坐标。这个思路也适用于不同 PP 层分配：先以全局层 ID 建立对应，再决定物理目的地，不能用进程编号暗示模型语义。

如果 D 使用不同页大小，除了 head 重排还要对 token 维重新分块。可以先用连续逻辑数组作为 oracle，再实现源页 gather、目标页 scatter；比较恢复后的逻辑值，而不是比较物理字节排列是否相同。最后一页有效长度、非连续物理页和乱序目的映射都应加入验证。

当某种模型使用压缩潜变量缓存，head 维本身可能不再按普通 K/V 形式存在。此时上面的转换器应明确拒绝，另写对应状态映射。可扩展系统不是让一个通用 reshape 处理所有模型，而是让每种支持的状态类型都拥有清楚、可测试的转换契约。
## 11. 第十一课：真实 vLLM connector 的两端契约

固定 [KVConnectorBase_V1](https://github.com/vllm-project/vllm/blob/e0d00e4b8eb0c89ec30921d0695cfdd409e9c211/vllm/distributed/kv_transfer/kv_connector/v1/base.py) 区分 scheduler-side 与 worker-side。调度侧决定外部可命中数量、分配后状态和本轮传输元数据；worker 侧根据元数据实际加载/保存缓存。KVConnectorMetadata 从 Scheduler 传到 Worker，KVConnectorWorkerMetadata 则反向返回并支持聚合。控制信息与数值搬运不是同一对象。

get_num_new_matched_tokens 返回可加载的外部 token 数与是否异步。数量为 None 表示还需要时间确定，应稍后再询问；零表示没有可加载 token，不能同时声称异步加载非零状态。接口说明该查询可能被多次调用，应避免把查询当成已经消费或提交传输的动作。

update_state_after_alloc 在目标块分配后更新 connector。异步加载场景可能对同一请求调用两次，先为外部 token 分配目标，再在传输完成后补其他块。源码特别提醒应根据 num_external_tokens 判断是否加载，不能只看 blocks 是否为空，因为非空块也可能属于没有选择加载的子 connector。

worker 的 start_load_kv 启动加载，wait_for_layer_load 在相应 attention 层使用前等待该层就绪，save_kv_layer 可以安排层级保存，wait_for_save 负责所需同步与提交。这个接口结构允许分层流水，但不保证每个具体 connector 都实现同样重叠效果。读接口之后还要读选中的后端实现与测试。

request_finished 返回的布尔值表示 connector 是否接管异步释放责任。若为真，请求已逻辑结束但块不能立即自由复用，要等相应发送完成通知。get_transfer_results 汇总发送、接收完成与失败信息，失败块也有明确报告路径。它把 S05 的延迟释放从单机执行扩展到了状态传输。

## 12. 第十二课：SGLang 接收队列与延迟释放的真实分支

固定 SGLang [prefill.py](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/disaggregation/prefill.py) 有 bootstrap 队列与 prefill 在途队列；[decode.py](https://github.com/sgl-project/sglang/blob/fa5580f6d0e969a3d44c603d215e2139886e8db2/python/sglang/srt/disaggregation/decode.py) 区分 DecodePreallocQueue 与 DecodeTransferQueue。前者管理接收前资源，后者管理已开始等待传输的请求。不能把“进入 decode 进程”当成“已经可以执行 decode”。

_commit_transfer_to_req 在释放元数据槽前读取首输出、缓存长度、logprobs 等，并校验 bootstrap_room 身份。若元数据尚未就绪或身份不符，不能按正常接管继续。这个路径直接说明传输的不只是 K/V 主数组，还包含让请求从正确位置继续的控制状态。

pop_transferred_with_outcomes 会轮询传输状态，区分成功、失败与尚未完成；PP 场景还使用跨 stage 共识，某个 stage 失败需要让相关 stage 一致处理。否则一部分 rank 把请求加入运行批，另一部分仍等待，后续集合通信参与者与长度可能分叉。

_defer_release 保存请求、开始时间、截止时间、元数据槽与所需确认数。resolve_deferred_releases 对 host 目标要求明确 drain 安全，注释指出超时不能证明远端写停止；device 目标分支在此快照存在 deadline fallback，并记录 timeout outcome。这个差异是实际实现的恢复取舍，不能概括成“所有超时释放都安全”，也不能把本章保守 CPU 参考声称为生产代码的逐字复刻。

释放准备还通过组内 flags 归约，避免仅部分 rank 释放导致准入与后续轮询 collective 大小不一致。代码先提交仍保留的列表，再逐项执行实际释放，防止某个释放异常后重试时再次处理已释放条目。这是异常边界与分布式一致性结合的具体例子，比一句“失败后清理”更值得逐行理解。


### 12.1 逐行读“能不能释放”与“各 rank 是否一致”

下面节选来自固定 decode.py 的 resolve_deferred_releases，省略外层循环与日志。它不是可以单独运行的程序，也不是我们推荐对任意传输采用超时释放；它展示当前快照如何区分 host 与 device 目标，以及为什么必须阅读条件分支才能理解实际保证。

```python
# 固定 SGLang decode.py 的释放就绪判定节选。
if self._is_host_destination(decode_req):
    is_drained = decode_req.kv_receiver.is_abort_release_safe()
    is_ready = is_drained
else:
    is_drained = decode_req.kv_receiver.kv_mgr.is_abort_release_safe(
        decode_req.req.bootstrap_room, required_acks
    )
    is_ready = is_drained or now >= deadline
drained.append(is_drained)
ready.append(is_ready)
ready = all_reduce_flags(
    flags=ready, group=self.gloo_group, op=torch.distributed.ReduceOp.MIN
)
```

第一分支通过 receiver 自己判断 abort 后是否已经安全排空，ready 直接等于 drained。因此仅时间超过 deadline 不会让 host 目标在这里变成可释放。第二分支从 manager 检查对应 bootstrap_room 与 required_acks，同时允许 deadline 到达作为 ready 条件。两个布尔值分别保留，是为了后续区分正常排空和超时回收，并记录不同 outcome。

ready 列表每一项对应同一顺序的 deferred entry。组内使用 MIN 归约布尔标志，意味着只有相关 rank 都认为该项 ready，整体才继续释放。若一个 rank 的列表顺序与另一个不同，即使长度相同也会把不同请求的标志混在一起。因此源码注释强调条目来自一致的 hold 决策，不能在某个 rank 独立插入一个额外条目。

归约后代码先构造 still_held 与 to_release，再把内部 deferred 列表替换为 still_held，最后逐个 _do_release。为什么不一边遍历原列表一边释放？因为某次释放若抛异常，已释放条目可能仍留在原列表，下一轮再次处理就有重复释放或 receiver 已为空的问题。先提交剩余列表，是一种让异常边界更清楚的做法。

_do_release 不只归还 KV，还清理 metadata slot、归还 metadata index，并结束 receiver 状态。只释放大数组却忘了小型元数据槽，也会让长期运行后无法接收新请求。资源清单因此应该同时包含设备页、host staging、请求槽、元数据槽、sender/receiver 对象和队列条目，而不是只盯显存。

阅读到 timeout fallback 时，应继续追 receiver/manager 的失败模型与连接失效机制，理解它为何选择这条恢复路径。仅凭这段代码不能证明所有远端写已经停止，也不能替整个框架下一个无限范围的安全结论。教材准确说明已经看到的分支和仍需验证的假设，才是真正的源码级学习。

## 13. 第十三课：故障工作坊、完整答案与工程里程碑

**故障一：D 输出与共置不同。** 先关闭随机采样，比较传输前后每层有效 KV、逻辑位置与首 token 状态。若数组值已不同，查布局、头分片、dtype 和尾部长度；若 KV 一致但 logits 不同，查模型/adapter 身份与 D 的计算路径；若 logits 一致而 token 不同，查 RNG 与采样状态。按最早差异定位，不要直接说“网络不稳定”。

**故障二：取消后偶发污染新请求。** 记录 transfer_id、请求代次、源/目标页映射、提交与完成事件。若旧远端写入仍可能到达而目标页已复用，新请求会被覆盖。延长 sleep 只能改变概率，修复需要明确释放安全或代次隔离机制。CPU Handoff 可先验证取消进入 draining 而不立即删目标。

**故障三：P 吞吐高，系统却越来越慢。** 观察待发送字节、D 预留页、D 队列与端到端 SLO。若 P 产出超过下游消化能力，积压会持续增长。增加 P 实例可能进一步恶化；应调整准入、池比例、传输能力或 D 容量，并保留拒绝/等待统计。

**故障四：只有某些 TP/PP rank 卡住。** 检查接收就绪与失败共识是否一致，元数据与数值完成是否在同一请求上配对。某个 rank 提前取消或超时释放，可能让其他 rank 的下一次 collective 数量不同。先用小请求和明确日志复现，不要直接扩大超时掩盖状态分叉。

**练习一，首 token 边界。** prompt 三 token，P 已采样首输出四，传三位置 KV。D 的 computed 为三、known 长四，下一步计算 token 四的逻辑位置三。若写 computed 四就跳过未存在 KV；若重新采样 prompt 最后位置就重复消费 RNG。答案必须同时写输出历史与缓存历史。

**练习二，传输字节。** 三十二层、八 KV heads、维度一百二十八、二字节，每 token 一百二十八 KiB，八千一百九十二 token 一 GiB。若按十六 token 整页传八千一百九十三有效 token，容量按八千二百零八计算，额外十五 token 槽。实际协议是否传尾部 padding 需另核查。

**练习三，页号映射。** P 页九不意味着 D 页九；两端页池独立。正确方案按逻辑块建立 source→destination 映射，传对应有效数值并更新 D 页表。只复制页表整数不复制数据，会让 D 读取自己的无关页。

**练习四，什么时候释放源。** 传输提交后仍需源引用，直到所用传输协议证明源不再被读取；目标 ready 也要按具体库语义判断。普通超时不是证明。若 prefix cache 另有引用，释放传输引用后缓存仍可持有，不能清掉整个页。

**练习五，重复完成。** 相同 transfer_id 再次 ready 应无额外 retain/free；旧请求代次应拒绝。Handoff 的 complete_copy 在终态返回 False，在身份不符时报错。二者分别处理重复与陈旧，不应全部默默接受。

**练习六，取消在途。** 先停止正常输出并标记 cancelled，仍保留源/目标直到完成或安全取消确认。参考程序进入 draining，再由显式 complete_copy 结算。若直接把 target=None，真实设备可能仍写原地址，逻辑删除对象无法停止远端 DMA。

**练习七，背压守恒。** 每轮生产三、消费二、容量四，按先消费后生产的参考顺序，队列最终有界且出现拒绝。每轮必须满足累计接受减累计完成等于剩余。请求数限制不足时加字节预算，并分别报告拒绝与等待，不能把未接收请求算成功。

**练习八，PD 获益条件。** 在简化模型中，共置排队减 D 排队大于传输成本才改善 TTFT。完整模型还加 P 排队、布局转换等。若目标是 ITL，可以接受某些 TTFT 取舍，但必须预先定义 SLO，不能看到结果后换指标。

**练习九，不同 TP。** P 二 rank 各四 KV heads，D 四 rank 各二 heads，P零分给D零/一，P一分给D二/三。头数不整除时简单均匀方案不成立，应查复制或其他布局，不能使用整数除法悄悄丢 head。

**练习十，命中长度不等。** P 命中一百二十八，D 命中六十四，仍需传 D 缺少且模型继续需要的范围。命中长度必须附带身份验证；相同长度不证明相同 token。共享源页传输期间应增加必要引用，避免被驱逐。

**练习十一，查询返回 None。** vLLM get_num_new_matched_tokens 的数量为 None 表示尚未确定，应稍后查询；不是零命中，也不是加载失败。若把 None 当零直接本地重算，可能浪费工作；若当成功，可能读取不存在的状态。

**练习十二，接管释放责任。** request_finished 返回 True 表示 connector 需要延迟释放并承担相关责任，不表示块已经空闲。Scheduler 与 connector 必须通过完成结果协调。接口返回值是所有权协议的一部分，不能只当普通成功布尔值。

**练习十三，失败目标能缓存吗。** 不能把部分或失败传输的目标作为有效 prefix 发布。先确认完整范围与身份，再提交缓存可见性；失败应释放或按可验证范围回滚。否则一个请求的传输故障会污染后续大量共享前缀请求。

**练习十四，超时分支怎样读。** 固定 SGLang host 路径要求 drain 安全，device 路径有 deadline fallback 并记录。答案应准确陈述所读分支与假设，不把它推广为所有传输都安全，也不删掉真实代码中的取舍来迎合教学理想。

**练习十五，设计最小实机验证。** 两个受限池、一个短合成 prompt、固定模型 revision、贪心、最多十六输出，先比较共置与分离的 token 和 KV 摘要，再测试取消与故障，最后测阶段时间。未验证协议正确前，不跑长时间大并发，也不把分析带宽公式当设备实测。

工程第一阶段实现 Handoff 与 Manifest 校验，完成正常、提前读、取消、重复、旧代次和错误模型测试。第二阶段把单数组扩展为多层/多页，并用逻辑 token gather 对照完整源。第三阶段加入字节预算与有限队列，记录守恒。第四阶段使用本机 CPU 进程传输小数组，增加真实序列化和进程失败处理。

第五阶段才接入固定真实 connector，先在分配到的少量 GPU 上做短时功能验证，记录源计算结束、传输提交、目标就绪、D 开始与最终释放。最后研究池比例、前缀亲和和 TP 重排，每次只改变一个核心变量。成果应包含状态表、布局清单、数值对照、故障证据和未执行范围，而不是只有启动命令截图。

### 13.1 给故障报告保留足够证据，但不泄露业务内容

一次 PD 故障记录至少包含匿名请求 ID、代次、transfer ID、模型 revision、有效 token 数、布局版本、源/目标 rank、页数、各阶段时间和最终状态。需要定位数值时使用合成 prompt 或摘要，避免把公司用户正文、完整 KV 或内部连接凭据写进公开日志。协议元数据也可能包含地址与令牌，应在分享前做有目的的脱敏。

对重复或陈旧消息，保存“收到的身份”与“当前期望身份”很有用。对提前读，保存提交、完成和 D 执行的事件关系；对泄漏，保存每类引用何时增加与减少；对数值错位，保存第一处分歧的层、逻辑 token、head 与维度。大量无结构日志通常不如这些少量精确信息。

报告中的成功也要分层：数值数组复制一致，说明 CPU 参考保真；进程间传输一致，说明序列化与身份处理在该场景正确；真实设备测试一致，才覆盖所选传输后端与同步路径。不同层的证据可以互相支持，但不能跳级替代。尤其不能把分析模型预测的传输时间写成实测，也不能把没有发生故障的短跑当成所有取消竞态都已证明安全。

最后写出一个最小重放输入与事件序列，例如“预留→提交→取消→旧 ready→当前完成”。其他人应该能在不接触私人数据的情况下重现你的判断。这样的报告既帮助自己掌握系统，也为以后向开源项目提交有价值的 issue 或测试奠定基础。


### 13.2 验收时同时检查正确性与资源终态

正常交接的验收不应停在 D 能生成几个 token。还要确认 P 的临时发送引用结束、D 的活动引用最终归还、元数据槽回收、传输队列没有残留，以及缓存保留符合明确策略。一个短请求数值正确但每次泄漏一个 metadata slot，运行久了仍会完全失去服务能力。

取消验收则同时检查“没有额外可见输出”和“在途访问没有提前失去保护”。这两个目标看起来相反，实际上分别属于业务状态与资源状态。测试应在多个事件边界插入取消，记录哪些阶段立即清理、哪些阶段等待完成，再证明最终不会永久滞留。

保留传输失败后的重放输入，使每次修改都能重新验证同一条取消与释放时间线；不要只保留最终汇总数字。

性能验收最后进行，并且使用与共置基线相同的模型、输入、输出约束和缓存条件。若只测试一个长 prompt，就只能对该条件下的取舍下结论；短输入、长输出、前缀命中和突发流量可能改变池比例与交接成本。把这些边界写进结果，才能为真实部署提供可靠依据。
掌握门槛：能够解释 PD 的计算收益与搬运代价，写出请求跨两端的所有权与完成协议，手算 KV 字节与布局重排，处理取消/重复/旧代次，沿真实源码区分预留、传输与接管。下一章进入 [S12 数值、量化与确定性](12-numerics-determinism.md)，研究正确状态在不同计算路径中为什么仍可能出现数值差异。
