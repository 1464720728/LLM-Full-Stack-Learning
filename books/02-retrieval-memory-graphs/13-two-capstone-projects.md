# R13 两个完整项目：证据助手与可纠正记忆助手

本章不是功能愿望清单。两个参考实现都已经在仓库中，可以离线运行、关闭后重开、执行失败测试，并在清楚边界的前提下接入模型。先修是 R01–R12，尤其事务、ACL、引用和 CAS。硬件只需普通 CPU。

## 1. 项目 A：带来源和版本的本地知识助手

交付目标不是“聊天界面能回复”，而是能够回答：这条材料从哪里来、当前用户是否能读、这次回答用了哪个版本、更新后旧答案怎样失效、没有证据时是否停止。实现由五个模块组成：text 负责词元和切片；store 负责版本与权限；retrieval 负责候选排序；rag 负责预算与引用；assistant 负责 CLI。

最短运行方式是：

```bash
python -m labs.knowledge.assistant ask "KV cache" --demo
```

它创建临时内存数据库，写入虚构资料并搜索。若想验证真正持久化，在你自己的实验目录创建数据库：

```bash
python -m labs.knowledge.assistant --db /tmp/course-knowledge.db seed
python -m labs.knowledge.assistant --db /tmp/course-knowledge.db ask "KV cache"
```

Windows 原生 Python 可以把路径换成自己建好的目录，例如 `./local-work/course-knowledge.db`，先创建目录；WSL/Linux 可用上述临时路径。不要把实验数据库提交进公共仓库。

第二次命令启动的是新进程。它没有再运行 seed，却仍能回答，说明信息来自磁盘数据库。不要只在同一个交互式 Python 对象上调用两次方法就声称“支持持久化”。

## 2. 项目 A 的逐阶段验收

摄入阶段检查 8 个逻辑文档，其中一份属于另一租户，一份只允许管理员组。普通学习者可见 6 份。每个 chunk 的原文必须满足 `body[start:end]==text`，每条 grants 必须引用已存在 revision。重复 seed 不应让 revision 行数增加。

查询阶段先检查当前 Context 的可见集合；然后保存 lexical terms、BM25 候选、TF-IDF 候选、RRF 合并顺序；最终回答记录 doc_id、revision、chunk_id。一个低层排序 bug 可以通过候选列表定位，不必让大模型解释自己为何错了。

发布阶段检查引用仍可见。将一个已检索文档更新，再校验旧 Claim，必须失败。这样会产生“检索时有，回答前失效”的情况；服务可以重新检索或返回证据已更新，不能继续输出旧内容却标作最新。

拒答阶段使用语料没有的词，结果 abstain=true；权限阶段搜索虚构受限标记，普通用户零结果，管理员一条结果；租户阶段用另一 tenant 的身份查询，不能看到 course 资料，即使某文档 grants='*'。

## 3. 从参考实现到自己的实现

第一次不要整体复制代码再运行。按以下顺序重写：先实现一个只返回列表的内存 store，验证 Chunk 契约；然后写 BM25 并用手算数据测试；接着给 store 增加 SQLite；再添加 revision 和 grants；最后接入 rag 与 CLI。每一步保留测试，观察增加一个能力需要改哪一层。

例如添加标题权重有多种方案：把标题重复拼入正文会改变长度和 df，容易混淆；单独计算标题字段分数更清楚；BM25F 可对多个字段进行加权，但需要明确字段长度归一化。不要为了“效果好一点”偷偷改基础数据含义。

扩展父子检索时可以保留小 chunk 用于搜索，命中后读取父章节作为上下文。必须对父章节重新做权限和预算检查；不能认为子片段可见就意味着整篇文档任何关联内容都可无界加入。当前课程以文档版本为授权单位，若以后引入段落权限，数据模型也必须变化。

## 4. 项目 B：可解释、可纠正、可遗忘的长期记忆

运行：

```bash
python -m labs.knowledge.memory_assistant demo
```

演示先保存虚构学习偏好，再明确纠正它，然后读取 active 状态，最后删除该 key 的所有历史版本。全新数据库的 revision 为 `[1,2]`，最后 `history_after_forget` 为空列表。重复使用同一数据库时版本继续递增；遗忘保留防 ABA 的哈希身份计数器，不会把新记忆伪装成旧 revision=1。

持久化 CLI 允许显式操作：

```bash
python -m labs.knowledge.memory_assistant --db /tmp/course-memory.db remember study.style "numeric examples" --evidence "fictional-message-1" --expected 0
python -m labs.knowledge.memory_assistant --db /tmp/course-memory.db recall
python -m labs.knowledge.memory_assistant --db /tmp/course-memory.db remember study.style "code first" --evidence "fictional-correction-2" --expected 1
python -m labs.knowledge.memory_assistant --db /tmp/course-memory.db forget study.style
```

命令顺序不能随意交换。全新数据库第二次 remember 要基于当前 revision=1；如果重复执行 expected=1，当前已经是 2，应报 ConflictError。这不是程序坏了，而是防止基于过期状态的覆盖。要照抄示例中的数字，使用新的临时数据库；也可以 forget 后以 expected=0 创建，但这时返回版本继续递增，后续更正必须使用实际返回的新版本，不能再硬编码1。

这个 CLI 是本地教学工具，tenant/owner 是调用者输入，因此不提供多用户网络认证。真正服务必须把身份由可信会话注入，再调用相同存储逻辑。

## 5. 项目 B 的质量指标

记忆命中率不是唯一目标。应同时检查：该记忆对本次任务是否必要；是否来自明确证据；是否仍有效；是否属于当前用户与项目；是否与当前指令冲突；删除后是否还通过摘要或缓存出现。过度记忆和错误召回都可以让 Agent 表现更差。

建立 12 条虚构事件：三条明确长期偏好、三条仅本次任务约束、两条更正、两条即将过期状态、一条撤回和一条删除请求。手工标注哪些可保存、哪些应在当前任务有效、哪些旧值应失效。自动系统必须与这套标注逐项比较，不能用“记得很多”当成功。

程序性记忆单独评测。给一条规则两种满足条件和两种不满足条件的环境，观察 Agent 是否误用。对失败经验要测试可恢复性：一次超时被记住后，工具恢复时系统是否仍拒绝尝试。

## 6. 加入真实模型：边界保持不变

模型适配器来自 Agent 工程卷。项目 A 给模型提供已授权证据 JSON，要求返回结构化 claims；项目 B 可让模型提出候选记忆，但写入策略、owner、expected_revision、删除操作必须由运行时校验。模型能力与数据库权限是两条不同边界。

建议先仅开放生成式重述，不开放任意新工具；使用少量虚构资料做 schema、引用、拒答测试。然后开放查询规划，再开放受控记忆写入。每增加一项自由度，都添加对应失败用例和预算限制。

根目录 [综合项目](../../projects/README.md) 将 Agent Harness 与本卷证据工具连接起来。它保留本卷的权限、引用与版本契约，是继续学习异步工具、恢复与最终输出校验的入口。具体脚本和验收以该项目当前 README 为准。

## 7. 练习与参考交付

**练习 1：改为自己的虚构主题。** 建 12 篇与课程不同的文本，必须有两个租户、两个权限组、一组新旧版本与两篇同名文档。完整验收是普通用户只能看到自己的当前授权数据，并且每个回答都有源 ID。不要拿真实公司材料作为首次测试。

**练习 2：加入证据导出。** 完整参考函数：

```python
import json
from pathlib import Path

def export_visible(store, context, destination):
    rows = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "revision": c.revision, "start": c.start,
             "end": c.end, "text": c.text}
            for c in store.visible(context)]
    Path(destination).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
```

它只导出当前可见信息；调用者仍需检查目标文件权限与保存位置。验收中故意指定普通用户，文件里不得出现受限标记。

**练习 3：实现一次纠正演练。** 保存 key=x 旧值，读取 revision，写新值，断言 recall 只含新值、history 有两个版本、旧状态 superseded。完整实现可直接参照 `test_correction`，再换成自己的 key 和值重写一遍。

**练习 4：做故障报告。** 故意关闭外键，再 purge 文档，观察 graph edges 是否残留；恢复外键并补测试。参考结论应区分 SQL 约束与应用过滤：即使过滤暂时隐藏孤儿边，数据完整性仍然坏了，不能因答案暂时没泄露就忽略。

**练习 5：完成个人验收演示。** 录制或写下从全新数据库开始的流程：摄入、查询、越权拒绝、更新、旧引用失效、重启、记忆纠正、遗忘。提交命令、预期与实际输出、测试结果、一个失败及修复。不以界面美观代替这些能力。

**掌握门槛：** 不看参考实现重新完成项目的核心数据流；能在代码审查中指出认证、授权、版本、预算、引用和删除边界；接入新模型时无需重写全部存储和检索层。
