# 完整实操：真实 Embeddings API、向量检索与可选 FAISS

这一实验接上真实语义向量服务，补齐“只会词法基线”的缺口。它已经提供完整适配器、批处理、验证、权限过滤和余弦检索代码；不是让你未来再自行寻找实现。默认模式离线，只有明确加 `--live` 才发送请求。

需要先掌握 R04 与 R08。API 实验在 Win11、WSL 或 Linux CPU 上都可以完成，无需本地 GPU。服务端模型由 API 提供方运行。拥有聊天模型 API 并不自动意味着该服务有 `/embeddings` 接口；必须选择明确支持 embedding 的模型。可以使用 OpenAI 的 embedding 服务或实现相同响应契约的供应商。DeepSeek 聊天模型名称不能直接当作 embedding 模型名填入。

## 1. 先用离线协议夹具验证链路

```bash
python -m labs.knowledge.embedding_api
python -m unittest tests.test_knowledge_embedding_api -v
```

默认向量是手工指定的三维测试数据。程序会明确输出 `mode: injected-offline-transport`，它只能证明分批、响应重排、维数校验与检索流程工作，不能证明语义质量。这个区分很重要：离线 mock 测试通过，不等于账户凭据、供应商兼容性或真实模型质量已经验证。

演示材料只有三句虚构英文，查询是“哪个文档描述小猫或猫”。mock 为猫、狗、KV cache 分别指定不同坐标，查询向量偏向猫坐标。响应故意按反序返回 items，适配器必须使用每条 `index` 恢复输入顺序。如果代码简单把返回数组按当前排列拼起来，就会把猫向量贴到另一条文档上，最终排名看似有数字却完全错误。

`test_response_reordered_by_index` 检查这个问题；重复 index、缺 item、NaN、布尔值、维数不一致、批间 model 身份变化也有独立测试。布尔值特别容易漏：Python 的 bool 是 int 子类，所以代码使用精确 type 检查，而不是简单 `isinstance(x,(int,float))`。

## 2. 显式配置真实请求

适配器读取三个环境变量：`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`。以 OpenAI 官方服务为例，基础地址是 `https://api.openai.com/v1`，模型示例是 `text-embedding-3-small`。这只是接口示例，不保证你的账号有权限、额度或供应商同名实现。

在 Bash 中可以先隐藏输入密钥，再设置非秘密配置：

```bash
read -rs -p "Embedding API key: " EMBEDDING_API_KEY
export EMBEDDING_API_KEY
printf '\n'
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_MODEL="text-embedding-3-small"
python -m labs.knowledge.embedding_api --live --batch-size 2 --max-requests 2
```

密钥不写进脚本或 Git。Windows 用户可以在 WSL 的 Bash 执行同样流程，或在自己的终端环境管理中设置变量。不要把真实密钥粘进教材文件、截图或公开问题单。程序不打印 Authorization，也不会把密钥写入输出 JSON。

该命令只提交三条虚构文档和一条查询，共四个输入；batch-size=2 时最多两次请求。模型的实际 token 计费与价格由服务决定，课程不写死估算成本。客户端有总字符数和请求数上限，但字符预算不是模型 token 上限，也不能替代供应商校验。

若服务支持指定输出维数，可以加 `--dimensions`。此参数并非所有模型或兼容提供方都支持；失败时先查模型文档，不要默默忽略错误后把不同维数的结果混用。真实响应会记录返回 model 名、维数和 usage。

## 3. 一次 HTTP 请求怎样变成向量对象

客户端构造 `POST <base-url>/embeddings`，JSON body 含 model、input 和 `encoding_format: float`；dimensions 只有显式设置时才发送。input 是字符串数组，因此一个请求可以处理多条。返回 data 中每个元素有 index 与 embedding，response model 描述服务实际返回的模型身份。

客户端并不把“HTTP 200”当作全部成功。它检查返回对象、item 数量、index 范围与唯一性、向量非空、元素为有限数值、所有维数一致，再恢复原输入顺序。通过这些检查后才构造 EmbeddingBatch。

批量请求还有跨批一致性。第一批返回768维、第二批返回1024维，必须拒绝；第一批model=A、第二批model=B，也必须拒绝。即使两个模型维数相同，其坐标系也不能默认兼容。课程把文档和查询放在同一次embed调用中，使这一约束可检查；长期缓存还需要记录embedding模型与配置版本。

返回 metadata 包括 adapter_version、requested_model、response_model、model_revision_note、dimension、metric、request_count、reported_total_tokens、创建时间和输入数量。model_revision_note 默认明确写为provider-alias-unpinned：一个动态模型别名不等于不可变权重版本，不能凭返回名字推断模型永远未变。

usage 缺失不等于零费用。任何一批没有报告 total_tokens，汇总的 reported_total_tokens 就是 null、usage_complete 为 false；reported_known_tokens 只累加已报告批次的数值，不能冒充完整账单。离线 fixture 明确报告0，仅表示这个合成响应的协议字段，并不是对真实服务费用的测量。请用两批请求、一批有usage另一批没有的测试检查该区别。

数值稳定性也属于接口正确性。直接计算 sqrt(sum(x*x)) 会把有限的1e308平方成无穷，把1e-308平方成0；前者导致NaN相似度，后者错误当作零向量。`cosine_topk` 先除以向量最大绝对值，再计算归一化范数与点积。余弦对正比例缩放不变，因此相同方向的大数与小数都应接近1；真正全零向量仍不返回任意排名。测试覆盖1e308、1e-308及最小非零浮点数量级，最后把舍入误差夹在[-1,1]中，不用这个夹取掩盖NaN。

## 4. 为什么拒绝重定向，为什么不自动重试

带 bearer key 的请求不应该在不了解目标的情况下转发到另一个地址。适配器拒绝 HTTP 重定向；URL 也不接受内嵌用户名、密码、query 或 fragment，避免秘密混进可记录的地址。远程服务要求 HTTPS，本机 localhost 的 HTTP 只用于明确的本地测试。

网络错误会产生简洁错误，不打印服务端原始响应内容。原始错误文本可能回显输入、内部地址或其他敏感信息，所以日志策略应谨慎。真实调试可在受控环境记录请求ID与错误类别，再到供应商控制台查，不把全部payload贴进公共日志。

课程默认不自动重试。第一批已经成功计费、第二批超时时，整个调用没有形成完整索引，但第一次费用不会自动撤销。如果加入重试，要记录已完成批次、幂等或缓存策略、总尝试预算；不能把数据库事务的原子性想象成跨外部API也自动成立。

## 5. 使用返回的真实向量做精确检索

`cosine_topk` 接收文档向量与查询向量，校验维数和有限数，计算每个余弦，稳定排序返回索引和分数。这里真实模式使用API实际返回的向量，完全不会退回TF-IDF或手工向量。查询为零向量时返回空结果；文档零向量被跳过，避免把无意义向量选为答案。

如果你想保留一次实验，可以显式指定本地输出路径：

```bash
python -m labs.knowledge.embedding_api --live --output /tmp/course-embedding-result.json
```

文件包含虚构输入、向量和版本元数据，不含API key；使用独占创建，路径已存在时不会静默覆盖旧实验。请把结果保存在自己的临时工作目录，之后重新打开并核对dimension与每行长度。真实资料向量也可能泄露内容或支持信息恢复，不应因为“只有数字”就视为非敏感数据。

课程没有保证“kitten查询一定把猫排第一”，因为真实模型与提供方结果必须实测。合理验收是：正确来源排名符合你的标注，失败时能区分模型表征问题与管线错误。离线mock结果好看，不能替代这一实际验证。

## 6. 接回带权限的 CorpusStore

`dense_search(store,context,query,client)` 已实现完整桥接。它先调用store.visible，只有当前授权chunk会被送到embedding服务，再把query一起编码，做余弦排名，返回前再次核对chunk仍然当前可见。它不会把受限材料先编码再在最后过滤。

这个教学实现每次查询都重新编码可见文档，适合小规模可观察实验；生产系统应缓存文档向量。缓存key至少应绑定chunk身份、embedding模型/版本、维数、预处理策略。权限控制仍在候选读取层，缓存存在不等于当前用户有权使用它。对长期索引还需要撤权、删除与版本切换测试。

真实调用前请明确资料能否发送到该供应商。默认示例全为虚构数据；把私人或公司语料换进去是一个新的数据出口决策，不能只因为API可用就自动上传。程序不会扫描你的文件系统或批量读取公司资料。

`test_dense_search_does_not_send_hidden_document` 检查隐藏文档文本没有进入传输payload，而不仅是没有出现在最终答案。这个测试更接近权限隔离的真实边界。

## 7. 可选 FAISS：精确索引与 HNSW 的同场对照

执行：

```bash
python -m labs.knowledge.faiss_lab
```

如果没有安装NumPy或FAISS，结果明确为SKIP，不联网安装。若你要运行，在单独实验环境安装官方支持的CPU包并记录实际版本；不要覆盖正在用于生产的服务器环境。此实验只需CPU，固定一个线程，不使用8×H100。

实验生成2000个32维float32随机向量和40个查询，先L2归一化，再建立IndexFlatIP。归一化后的内积排序等价于余弦排序，所以Flat结果作为精确近邻基准。再建立IndexHNSWFlat，分别使用efSearch=8、32、128，比较ANN找回精确top-10的比例与查询时间。

这是索引算法实验，不是语义检索实验，因为输入是随机向量。它回答“近似索引漏掉多少精确邻居”，不回答“模型找到了多少正确答案”。输出明确记录faiss_version、seed、数据规模、维数、线程数和计时。不要将一次很短的计时直接写成性能结论；做正式报告时还应重复并报告分布。

参数增大通常会增加搜索工作，但单次小规模实验可能因噪声不呈现严格单调延迟。测试不assert某个配置一定更快，也不强制召回某个固定阈值；只检查指标范围和接口行为。追求可解释测量，比写一个容易在别的机器上失败的速度断言更重要。

## 8. 必做练习与参考验收

1. **响应乱序。** 用mock反转items，检索结果应保持不变。修复方案必须按index重建，不能猜供应商总会排序。
2. **跨批维数变化。** 第一批返回3维、第二批4维，整体调用必须失败且不发布混合索引。保留失败原因，别补零伪装兼容。
3. **真正语义测试。** 构造十个词面重叠低但有明确答案的查询，与BM25比较逐题结果。真实embedding未必全胜，记录其独特成功与失败。
4. **版本切换。** 使用两个模型分别编码同样资料，保存两套metadata。禁止A文档向量与B查询向量混用。不能用维数相同作为兼容判断。
5. **授权出口。** 加一份受限文档，检查传输payload里没有它，再测试撤权后新的请求不再传输。仅检查最终top-k不足。
6. **ANN消融。** 保持相同向量、查询与种子，比较不同efSearch的近邻召回和耗时；再用真实标注测任务召回，写出两者为何可能不一致。

完成这份实验后，你已经拥有三种不同层次的实现：可解释词法基线、从零训练的小学习器、真实embedding API与生产索引接口。它们各自回答不同问题，能互相校验，而不应被混成一个“向量检索效果很好”的模糊结论。

接口来源：[OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create)、[Faiss Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started)、[Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。核查日期2026-11-01。适配器离线协议测试已运行；本次未使用真实密钥调用供应商，FAISS缺依赖时明确跳过。
