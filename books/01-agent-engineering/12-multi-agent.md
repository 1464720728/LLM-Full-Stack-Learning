# A12 多 Agent：分工有合同，交付有证据，协作有总预算

本章把多 Agent 从角色扮演变成工程系统。一个研究者、一个审查者和一个协调者并不会因为名字不同就自动提高质量；它们需要明确任务、上下文、工具权限、通信结构、共享预算和独立验收。你会先用确定性 worker 构建一个小团队，合并带来源的主张，处理冲突、失败与重派，再设计公平的单 Agent 对照实验。

所有代码离线运行，worker 可以是普通函数，不需要真实模型调用。这样先验证协调机制，再接模型时就能区分是调度与合同出错，还是模型生成质量不足。多 Agent 是多个决策与上下文边界，不一定意味着多个进程、机器或模型实例；一个 Agent 也可以使用多个并发工具。

## 第一课：先问分工解决哪种具体困难

任务适合分工，通常因为存在相对独立子问题、不同专业工具、需要独立复核，或权限应该分离。例如三个互不依赖代码模块可以分别分析；一个角色提出方案，另一个依据测试独立审查；只读研究者与有写权限执行者可以分开。若所有步骤都依赖同一个不断变化文件，让多个 worker 同时修改可能只增加冲突。

分工有成本。每个 worker 需要理解目标、读取材料、生成输出，协调者还要分解、检查和合并。公共上下文可能重复发送，通信可能丢失细节，角色之间可能重复探索。只有当并行性、专门能力或独立验证带来的收益超过这些成本，团队才值得使用。不能把更多模型调用本身当作智能提高的证据。

| 任务形状 | 可能分工 | 主要收益 | 主要风险 |
|---|---|---|---|
| 三份独立资料调查 | 每份一个研究 worker | 并行覆盖 | 来源重复与结论冲突 |
| 一个函数连续修复 | 单作者加独立审查 | 减少写冲突 | 审查者被作者解释带偏 |
| 数据提取后汇总 | 流水线角色 | 合同清楚 | 上游错误传播 |
| 高影响动作 | 提案与批准分离 | 权限边界 | 错把角色身份当真实授权 |
| 开放研究问题 | 有限分层委派 | 扩展探索 | 子任务指数膨胀 |

角色名称不是权限。一个 prompt 写“你是安全审计员”，并不会使它获得真实批准能力；写“你是管理员”也不应扩展工具范围。权限由宿主绑定到任务和身份，worker 只能请求允许动作。团队系统尤其要防止某 worker 的文本被协调者误当高权威指令，跨 Agent 消息仍然需要来源与信任边界。

## 第二课：角色、会话、任务、尝试和业务动作分别编号

角色表示职责，例如 researcher；会话表示一份模型上下文；task_id 表示要完成的子任务；attempt_id 表示这次执行尝试；operation_id 表示可能跨尝试重复的同一业务动作。它们不能共用一个编号。重派任务时 task_id 不变、attempt_id 改变；若仍属于同一业务意图，operation_id 保持稳定。

一个 worker 可以连续执行多个任务，每个任务应有独立状态或明确上下文隔离。否则前一个任务的私有信息和约束可能泄漏到下一个任务。反过来，同一任务可以因失败由另一个 worker 接手，但必须得到已确认事实、未决效果和剩余预算，而不是只收到最初一句目标后从头做。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Assignment:
    task_id: str
    attempt_id: str
    role: str
    goal: str
    allowed_tools: tuple
    input_refs: tuple
    max_calls: int

assignment = Assignment("task-1","attempt-1","researcher","compare documented limits",
    ("read_public","search_public"),("doc-a","doc-b"),3)
assert assignment.task_id != assignment.attempt_id
assert "write_file" not in assignment.allowed_tools
```

这个 dataclass 只组织字段，不自动验证类型或来源。外部派工请求需要 schema 校验，max_calls 必须精确非负整数，工具集合必须是宿主允许集合的子集，input_refs 必须当前身份可访问。模型提出的子任务可以成为候选，但不能直接构造任意权限的 Assignment。

任务编号还帮助合并结果。完成顺序可能变化，worker 可能重发交付，协调者必须按 task_id 与 attempt_id 识别哪份结果当前有效。迟到旧尝试不能覆盖新尝试已确认结果；如果旧尝试产生了副作用，则保留观察用于对账，而不把它当作正常新交付。A09 的拥有者版本与 A10 的账本在团队系统里继续适用。

## 第三课：派工消息应让 worker 不必猜成功标准

一份完整派工至少包含目标、输入材料、允许工具、禁止范围、交付结构、预算、依赖、验收方法和升级条件。目标应具体到可检查，例如“从两份公开文档提取当前接口限制并保留段落来源”，而不是“研究一下”。验收方法告诉 worker 什么证据足够，也让协调者能独立判断完成。

输入材料可以是受控引用，而不是整段复制。worker 按需读取相关片段，减少重复上下文；但引用必须能解析并验证权限。若只给一个文档名字而没有版本或位置，worker 可能读到不同材料。公共背景摘要可以共享，关键原文仍应可追溯，避免摘要误差在多个角色之间放大。

交付结构按任务类型不同。研究任务返回 claims、sources、uncertainties 和 coverage；编码任务返回 changed_files、diff_ref、tests_run 和 remaining_failures；验证任务返回 verdict、evidence 和 scope。不要要求所有角色都交一段自然语言总结，再让协调者用模型猜字段。结构化交付把关键事实变成程序可检查对象。

升级条件说明何时停止并向协调者报告，例如缺少权限、预算不足、发现任务依赖错误、来源冲突或副作用未知。worker 不应为了完成指标擅自扩大范围。一个合理的 blocked 交付可能比无证据的 success 更有价值，协调者应把它作为需要新决策的状态，而不是自动惩罚为无能。

## 第四课：交付 schema 与证据验证是两道门

一个结果包含 source 字符串，只证明字段存在，不证明来源真实支持主张。第一道门检查结构与类型，第二道门检查来源可解析、当前可见、内容与主张一致、适用范围正确。模型自报 verified=true 不应直接进入已验证集合；verified 应由宿主或独立 verifier 根据实际证据写入。

```python
from labs.agents.contracts import validate, object_schema, ContractError

CLAIM_SCHEMA = object_schema({
    "subject":{"type":"string","minLength":1},
    "value":{"type":"string","minLength":1},
    "source":{"type":"string","minLength":1},
})
RESULT_SCHEMA = object_schema({
    "task_id":{"type":"string","minLength":1},
    "attempt_id":{"type":"string","minLength":1},
    "claims":{"type":"array","items":CLAIM_SCHEMA,"maxItems":20},
    "status":{"type":"string","enum":["completed","blocked"]},
})
result = {"task_id":"task-1","attempt_id":"attempt-1","claims":[
    {"subject":"limit","value":"10","source":"doc-a#limit"}],"status":"completed"}
validate(result,RESULT_SCHEMA)
```

schema 的 maxItems 限制单次交付数量，防止 worker 返回无限大列表；字符串仍可另设长度上限。status=completed 是 worker 报告状态，不是协调者验收成功。协调者应增加独立 acceptance 字段，记录 accepted、rejected 或 needs_review。把报告与验收分开，避免 worker 自己决定最终成功指标。

来源验证可以先用离线字典模拟。每个 source_id 对应受控事实，worker 交付的 subject 与 value 必须匹配。真实文本支持关系更复杂，可以用引用片段、结构化数据、程序查询或人工审查，但不能把一个 URL 的存在当作充分证据。课程先用精确映射建立可测试机制。

```python
source_facts = {"doc-a#limit":{"subject":"limit","value":"10"},
                "doc-b#version":{"subject":"version","value":"v2"}}

def verify_claim(claim, sources):
    validate(claim,CLAIM_SCHEMA)
    evidence = sources.get(claim["source"])
    if evidence is None:
        raise ContractError("unknown source")
    if evidence != {"subject":claim["subject"],"value":claim["value"]}:
        raise ContractError("claim not supported by source")
    return dict(claim, verified=True)

assert verify_claim(result["claims"][0],source_facts)["verified"] is True
```

这段验证不使用模型，所以不能处理任意自然语言推理；它展示的是证据与结论必须有外部对应关系。真实系统可以在此接口后接更复杂 verifier，但应保留其不确定性与适用范围，不把“模型审查模型”包装成确定真相。

## 第五课：冲突合并不使用简单多数票

课程 merge_claims 把相同 subject 的不同 value 分组，保留每个值的 worker 和 source，并设置 conflict。三个 worker 说 v1，一个说 v2，不自动让 v1 胜出。前三个可能都复制同一旧文，第四个可能引用最新官方资料。独立性来自证据与方法，而不是会话数量。

```python
from labs.agents.workflows import merge_claims

merged = merge_claims({
    "worker-a":[{"subject":"version","value":"v1","source":"old-doc"}],
    "worker-b":[{"subject":"version","value":"v1","source":"old-doc"}],
    "worker-c":[{"subject":"version","value":"v2","source":"new-doc"}],
})
assert merged["version"]["conflict"] is True
assert len(merged["version"]["values"]) == 2
```

源码以 subject 为键，再把 value 做排序 JSON 编码作为候选值键。它能处理可 JSON 编码的值，不判断不同主张是否其实谈论不同范围。例如一个 version 指软件包版本，另一个指 API 协议版本，表面冲突可能只是 subject 过粗。交付合同应把 subject、predicate、scope 和时间分开，减少语义碰撞。

冲突处理先核对适用范围，再核对来源版本、原始性和验证方法。若一份资料已过期，可以明确说明为何选择新证据；若仍无法判定，保留冲突并标记需要核验。不要把两个版本平均成不存在的中间值，也不要让协调者为了交付完整答案隐藏不确定性。

相同结论也不自动提高可信度。多个 worker 使用相同模型、相同提示、相同检索结果，错误可能高度相关。独立审查可以改变证据路径，例如作者依据文档提出结论，审查者运行最小代码验证；或两个研究者分别读取原始来源，再由协调者比较。方法多样性比换几个 persona 名称更重要。

## 第六课：协调者不是转发器，它承担整合验收

协调者先把用户目标拆成覆盖明确的子任务，再检查 worker 交付是否满足合同，最后验证整体目标。若只把所有结果拼接起来，重复、冲突和遗漏都留给用户处理，团队并没有完成整合。协调者需要维护 coverage 表：每个必要问题由哪个任务负责、交付了哪些证据、是否已验收、还有哪些缺口。

但协调者也不应该重新做每个 worker 的全部工作，否则分工收益消失。可以按风险选择验证强度：结构与身份全量检查，确定性测试自动执行，重要结论追溯原始证据，低风险材料抽样。关键是明确验证范围，不把 worker 的“测试通过”直接当作自己已验证。最终报告应区分 worker 报告与协调者复核。

协调者上下文也有限。若十个 worker 各返回两万字，合并成本可能比原任务更大。交付应包含简洁结构摘要与可按需读取的 artifact 引用，协调者先看覆盖与冲突，再深入相关原文。摘要不能丢失不确定性和失败状态，必要证据仍保留来源。A07 的上下文预算在团队中变成每个角色与协调层共同的问题。

协调者应该是最终交付的单一拥有者，避免多个 worker 各自对用户宣布完成或执行外部通知。内部交付不等于向外发布。若任务授权允许某 worker 直接执行特定动作，也要明确范围、幂等键和审计，不能由角色自行扩大通信对象。工程协作与向他人发送消息是不同权限。

## 第七课：共享文件需要所有权，接口需要版本

多个编码 worker 同时修改同一文件，可能产生文本冲突，也可能产生无文本冲突的语义错误。一个改函数参数，另一个基于旧接口修改调用者，Git 自动合并并不证明程序正确。分工时应给出独占文件区域、共享接口合同和整合测试，必要时使用独立工作副本。

文件所有权是协调规则，不是操作系统权限保证。worker 应知道哪些路径可以改、哪些只能读、哪些接口变化必须先通知依赖方。共享 schema 或类型定义最好由一个拥有者维护，其他 worker 通过明确变更请求协调，避免每个人各自改同一核心文件。任务结束后协调者查看实际 diff，而不只读口头报告。

接口版本可以包含 schema_version、字段名称、类型、单位和错误语义。把 source 改成 sources 不只是重命名，可能从单来源变成多来源，需要更新 consumer 与测试。若旧 worker 仍运行，应拒绝不兼容交付或使用明确迁移，而不是静默把缺字段当空列表。长任务跨上下文和代码版本时，稳定合同尤其重要。

写入冲突也可以用 A08 的读时摘要发现，但它只覆盖课程单写入者工具的已观察过期，不提供完整并发 CAS。多 worker 真正需要隔离、锁或集中提交。最简单可靠的协作往往是各自生成补丁，由协调者统一应用并运行整合测试；这比让所有角色共享一个可随意写的目录更容易理解。

## 第八课：全局预算防止局部合规、整体失控

每个 worker 最多五次模型调用，十个 worker 就可能五十次，再加协调者分解与合并，整体远超用户预期。分层委派还可能指数增长：每个角色再派三个，深度四就有许多任务。需要全局预算账本限制总调用、总费用、总子任务数、委派深度和 deadline，而不是只给每个角色局部限制。

预算应在真实请求前原子预留。worker 先向共享账本申请，成功才发请求，失败则返回 blocked 或 partial。完成后按真实 usage 结算，未知用量保留未决预留。A09 的 AsyncBudget 可以演示同进程共享次数预算，跨进程则需要事务或独立预算服务。仅在所有 worker 结束后汇总，已经来不及阻止超支。

```python
import asyncio
from labs.agents.advanced_solutions import AsyncBudget

async def global_budget_demo():
    budget = AsyncBudget(5)
    calls = []
    async def worker(name):
        accepted = 0
        for attempt in range(3):
            if not await budget.reserve():
                break
            calls.append((name,attempt))
            accepted += 1
            await asyncio.sleep(0)
        return {"worker":name,"calls":accepted}
    reports = await asyncio.gather(*(worker(name) for name in ("A","B","C")))
    assert len(calls) == 5 and budget.used == 5
    assert sum(item["calls"] for item in reports) == 5
    return reports

reports = asyncio.run(global_budget_demo())
```

并行降低墙上等待时间，不自动降低总费用。三个 worker 用时五、八、六秒，理想并发完成约由八秒主导，加协调开销；费用仍是三者之和，协调者还要额外调用。若每个 worker 重读同样大材料，成本可能比单 Agent 更高。评测必须同时报告成功率、总用量与延迟，不能只展示更快的一次。

预算分配可以预先切片，也可以共享池。切片容易控制但可能浪费某个闲置角色额度；共享池更灵活但需要公平策略，防止一个循环 worker 抢光资源。可以给每任务基础份额，再由协调者根据进展追加，所有追加都写入账本。模型不能因为声称“还差一步”就自动扩大总预算。

## 第九课：通信协议要抵抗重复、迟到和身份混淆

内部消息至少包含 message_id、task_id、attempt_id、sender、kind 和 payload。kind 可以是 progress、result、blocked 或 clarification。接收方验证发送者是否负责该任务、尝试是否当前有效、消息是否重复，再处理内容。文本里自称“我是协调者”不能改变真实 sender 元数据，身份由通信通道绑定。

重复消息常见于重试和确认丢失。同 message_id 同内容可以幂等忽略，不同内容则冲突；结果消息重复不能重复增加已完成任务数。迟到旧 attempt 的结果应标记 stale，必要时保存证据，但不覆盖当前验收。若它包含副作用结果，仍需交给对账，不能因为 stale 就直接丢弃。

进度消息不是完成证据。“已经快完成”不应改变任务终态，也不应无限延长 deadline。协调者可以据此判断是否调整资源，但应结合实际 artifact、测试和预算。完成消息则必须满足交付 schema，不能只靠句子里出现“完成”两个字。状态协议应让普通自然语言不具备直接改变控制字段的能力。

澄清请求也要有边界。worker 缺少输入时可以提出具体问题，但不应把所有设计选择都回抛给用户；协调者先利用已有上下文与工具解决，再在确实缺关键约束时询问。任务已经明确授权的可逆工作应继续推进，避免团队内部每个角色都重复请求确认。权限与目标由协调层统一管理，减少用户负担。

## 第十课：失败、重派与部分结果

worker 失败后，协调者先判断失败类别。格式错误可以要求有限修正，临时读取失败可以重试，权限不足需要调整范围或停止，副作用未知必须对账，任务依赖错误需要重新规划。不能把所有失败都通过“再派一个 worker”解决，因为新 worker 可能重复已经发生的效果。

重派保留 task_id，生成新 attempt_id，并传递已确认事实、失败原因、剩余预算和未决 operation_id。旧尝试应停止安排新动作，但可能有迟到结果。协调者需要一个当前尝试版本，拒绝过期普通交付，同时保留效果观察。A10 的幂等与 A09 的拥有者问题在团队层重复出现，不会因角色更多而消失。

部分结果应说明覆盖范围。三个模块只分析了两个，最终报告可以明确列已完成模块和未完成原因，不能写“整个仓库已审查”。如果缺失部分是验收必需，整体 status 应为 partial 或 blocked；若只是可选扩展，可以完成主要目标并说明未做。是否足够由任务合同决定，不由 worker 数量或文字长度决定。

协调者自身也会失败。它的任务分配、当前尝试、预算和验收结果应持久保存，恢复后不能重新把所有工作当未开始。worker artifact 应有稳定引用和内容摘要，协调者重启后可以复核。若只把协作状态放在聊天记忆里，长任务中断后很难可靠接续。

## 第十一课：四个协作失败复盘

故障一，四个研究 worker 一致给出错误版本。它们都读取同一篇过期转载，协调者按多数票接受。修复保留来源链与版本，独立复核原始资料，冲突或一致都不代替证据验证。测试让三个旧来源副本和一个新原始来源并存，协调者不能简单以人数决定真相。

故障二，作者和审查者都说测试通过，实际没有运行。作者输出了一段预期日志，审查者只阅读作者总结。修复把测试执行结果作为宿主产生的结构化 artifact，审查者读取真实退出码与测试范围，协调者复核关键用例。模型生成的日志文本不是测试证据，来源字段必须区分实际工具观察与叙述。

故障三，两个 worker 同时改共享 schema，整合后 consumer 读取旧字段。文本合并没有冲突，但接口语义不一致。修复建立共享接口拥有者、schema_version 与契约测试，接口变更及时通知依赖任务。验收故意把 source 改成 sources，旧 consumer 必须明确失败，不能静默产生空证据。

故障四，子 Agent 各自预算合规，总调用却远超限制。根因是没有共享预算和委派深度上限。修复请求前中心预留，子任务创建也消耗全局配额；预算不足时返回 blocked。测试多个 worker 同时申请，真实请求数不超过总额，不能只看每个局部报告。

## 第十二课：十二道完整练习与答案

练习一，给三种任务选择组织方式。答案：独立查三份文档可并行研究；连续修改同一函数适合单作者并在阶段后审查；独立验证补丁适合审查者使用测试与代码证据。理由分别是输入独立、共享写冲突和验证路径独立。角色数量由依赖决定，不默认越多越好。

练习二，缺来源交付必须在 schema 层失败。答案删除 claim.source，验证器抛 ContractError。这个测试只证明字段必需，后续仍需实际来源核验，不能把结构正确当事实正确。

```python
def expect_error(kind, operation):
    try:
        operation()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")

expect_error(ContractError, lambda: validate({"subject":"limit","value":"10"},CLAIM_SCHEMA))
```

练习三，来源存在但不支持值也必须拒绝。答案引用 doc-a 的 limit=10，却声称二十。verify_claim 比较受控事实，失败发生在证据层，不是格式层。

```python
claim = {"subject":"limit","value":"20","source":"doc-a#limit"}
validate(claim,CLAIM_SCHEMA)
expect_error(ContractError, lambda: verify_claim(claim,source_facts))
```

练习四，未知来源不能进入 verified。答案给出形状合法的伪来源，验证拒绝。真实服务应解析 artifact 或官方文档引用，不能仅检查字符串看起来像 URL。

```python
expect_error(ContractError, lambda: verify_claim(
    {"subject":"limit","value":"10","source":"invented-source"},source_facts))
```

练习五，重复结论仍保留来源，冲突值不被多数覆盖。答案三个 worker 给 v1、一个给 v2，合并后候选值仍为二，conflict 为真。这个结构为后续核验保留信息，不自行裁决。

```python
outputs = {f"old-{i}":[{"subject":"version","value":"v1","source":"same-old-doc"}] for i in range(3)}
outputs["new"] = [{"subject":"version","value":"v2","source":"new-doc"}]
combined = merge_claims(outputs)
assert combined["version"]["conflict"] is True
assert len(combined["version"]["values"]) == 2
```

练习六，预算五次、三个 worker 各想三次，真实总数应为五。答案复用 global_budget_demo，并检查局部报告之和。哪名 worker 获得几次不作为稳定业务保证，公平分配若重要应另有调度政策。

```python
reports = asyncio.run(global_budget_demo())
assert sum(report["calls"] for report in reports) == 5
assert all(0 <= report["calls"] <= 3 for report in reports)
```

练习七，交付身份必须对应当前尝试。完整答案检查 task_id 与 attempt_id，旧 attempt 不能覆盖。这里直接拒绝普通结果，真实系统可另保存迟到副作用观察供对账。

```python
def accept_identity(result, assignment):
    validate(result,RESULT_SCHEMA)
    if result["task_id"] != assignment.task_id or result["attempt_id"] != assignment.attempt_id:
        raise ContractError("stale or unrelated result")
    return True

assert accept_identity(result,assignment)
expect_error(ContractError, lambda: accept_identity(dict(result,attempt_id="old-attempt"),assignment))
```

练习八，子任务工具集合必须是父授权子集。答案使用集合包含判断，工具名称来自宿主注册表。外部文档要求增加 send_secret 不改变允许范围。

```python
def check_delegated_tools(parent_allowed, requested):
    if not set(requested) <= set(parent_allowed):
        raise PermissionError("delegation expands authority")
    return tuple(sorted(set(requested)))

assert check_delegated_tools({"read","search"},{"read"}) == ("read",)
expect_error(PermissionError, lambda: check_delegated_tools({"read"},{"read","send_secret"}))
```

练习九，委派深度必须受限。答案要求精确非负深度与最大值，创建子任务前检查；深度等于上限时不能继续委派。深度限制不替代总子任务数，宽度也需要预算。

```python
def child_depth(current, maximum):
    if type(current) is not int or type(maximum) is not int or current < 0 or maximum < 0:
        raise ValueError("nonnegative integer depths required")
    if current >= maximum:
        raise ContractError("delegation depth exhausted")
    return current + 1

assert child_depth(1,2) == 2
expect_error(ContractError, lambda: child_depth(2,2))
expect_error(ValueError, lambda: child_depth(True,2))
```

练习十，通信消息去重不能只忽略编号。完整答案保存规范化内容，同编号同内容返回 False 表示重复，同编号不同内容冲突。缓存位于内存，仅演示单进程消息身份，持久 inbox 参见 A10。

```python
import json

class MessageInbox:
    def __init__(self):
        self.seen = {}
    def receive(self, message_id, payload):
        encoded = json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)
        if message_id in self.seen:
            if self.seen[message_id] != encoded:
                raise ContractError("message identity conflict")
            return False
        self.seen[message_id] = encoded
        return True

inbox = MessageInbox()
assert inbox.receive("m1",{"task":"t","value":1}) is True
assert inbox.receive("m1",{"value":1,"task":"t"}) is False
expect_error(ContractError, lambda: inbox.receive("m1",{"task":"t","value":2}))
```

练习十一，整体覆盖不能由已完成数量猜测。答案使用必要任务 id 集合，只有全部验收通过才 completed；两个重复交付不等于两个任务完成。可选任务不影响主要完成，但应另列状态。

```python
def coverage_status(required_ids, accepted_ids):
    missing = set(required_ids) - set(accepted_ids)
    return {"status":"completed" if not missing else "partial", "missing":sorted(missing)}

assert coverage_status({"A","B"},["A","A"]) == {"status":"partial","missing":["B"]}
assert coverage_status({"A","B"},["B","A"]) == {"status":"completed","missing":[]}
```

练习十二，公平比较成本必须包含协调者。答案假设两 worker 各用三千 token，协调者分解一千、合并两千，总计九千；不能报告只用了三千或六千。这里是人工单位算例，不代表真实服务账单，实际应记录每次 usage。

```python
usage = {"planner":1000,"worker-a":3000,"worker-b":3000,"merger":2000}
assert sum(usage.values()) == 9000
single_budget = 9000
assert sum(usage.values()) <= single_budget
```

## 第十三课：完整小项目，两个研究 worker 与一个确定性协调者

项目输入两个任务，各自读取一个受控来源。worker 返回结构化 claims，协调者验证身份和 schema，再逐条核对来源，最后合并主张并计算覆盖。一个任务故意提供不被来源支持的值，协调者应拒绝该任务交付，整体为 partial，而不是把漂亮总结当成功。整个过程不调用真实模型，先证明协调逻辑正确。

```python
from labs.agents.workflows import bounded_fanout

async def coordinate(assignments, worker, sources, *, limit=2):
    if len({a.task_id for a in assignments}) != len(assignments):
        raise ContractError("duplicate task assignment")
    async def run_one(assignment):
        try:
            raw = await worker(assignment)
            accept_identity(raw,assignment)
            if raw["status"] != "completed":
                return {"task_id":assignment.task_id,"accepted":False,"reason":"worker_blocked"}
            verified = [verify_claim(claim,sources) for claim in raw["claims"]]
            if not verified:
                raise ContractError("completed research needs evidence")
            return {"task_id":assignment.task_id,"accepted":True,"claims":verified}
        except ContractError as exc:
            return {"task_id":assignment.task_id,"accepted":False,"reason":str(exc)}
    reports = await bounded_fanout(assignments,run_one,limit=limit)
    accepted = {report["task_id"]:report["claims"] for report in reports if report["accepted"]}
    coverage = coverage_status({a.task_id for a in assignments},accepted)
    return {"reports":reports,"coverage":coverage,"claims":merge_claims(accepted)}

async def team_project():
    assignments = [
        Assignment("limit-task","a1","researcher","find limit",("read_public",),("doc-a#limit",),1),
        Assignment("version-task","a1","researcher","find version",("read_public",),("doc-b#version",),1),
    ]
    async def worker(assignment):
        await asyncio.sleep(0)
        if assignment.task_id == "limit-task":
            claim = {"subject":"limit","value":"10","source":"doc-a#limit"}
        else:
            claim = {"subject":"version","value":"invented-v9","source":"doc-b#version"}
        return {"task_id":assignment.task_id,"attempt_id":assignment.attempt_id,
                "status":"completed","claims":[claim]}
    report = await coordinate(assignments,worker,source_facts)
    assert report["coverage"] == {"status":"partial","missing":["version-task"]}
    assert "limit" in report["claims"] and "version" not in report["claims"]
    return report

team_report = asyncio.run(team_project())
```

run_one 把普通合同失败转换为拒收结果，使其他独立任务仍可完成；未知程序异常没有被宽泛吞掉，会触发 bounded_fanout 的失败回收。这个选择体现产品语义：不支持的主张是可报告的交付失败，协调器本身 bug 则不应被伪装成 worker blocked。真实系统可以增加明确错误分类，但不能用 catch all 隐藏实现问题。

verified 列表由宿主验证器生成，worker 原始结果没有写 verified 权限。merge_claims 只接收已验收的交付，因此伪来源或不支持值不会进入最终候选。即使已验证来源之间冲突，合并器仍保留 conflict，交给更高层核验，不自动多数裁决。这样结构验证、证据核验、冲突保留和覆盖判断分别有明确位置。

项目没有检查 worker 只读取了 input_refs 指定来源，因为本例 worker 是受控函数，没有实际工具层。真实模型 worker 应通过受限读取工具实现此约束，不能只在 Assignment 写 allowed_tools 就认为权限已经实施。任务描述是合同，工具执行器才落实能力边界。这个区别在接真实模型时必须补齐。

项目也没有把 max_calls 接入共享预算，因为离线 worker 没有模型请求；前面的 global_budget_demo 已给出预算机制。真实接入应在每次模型或工具请求前预留，而不是在 coordinate 收到结果后才检查 worker 报告。两段教学组件组合时要明确接入点，不能靠名字相似就声称已经集成。

## 第十四课：审查者怎样保持独立，而不是重复作者结论

如果审查者只看到作者的解释，它很容易沿着作者设定的问题框架思考，忽略未提到的失败路径。独立审查应直接获得任务合同、原始输入、实际补丁和测试证据，作者总结只是辅助。对于关键结论，可以先让审查者独立形成判断，再阅读作者解释，比较分歧。这里的独立性是信息路径设计，不是换一个角色名称。

编码任务中，作者可能只测试正常输入，审查者应根据需求寻找边界：空输入、重复调用、错误类型、并发、取消、资源上限和恢复。审查者提出的问题还要变成可复现输入或具体代码位置，不能只写“可能有风险”。协调者根据证据决定修复，修复后重新运行相关测试，避免把审查意见当作事实无需验证。

研究任务中，独立复核可以使用不同原始来源或不同方法。例如文档声称某 API 行为，另一个 worker 用受控 fixture 或实际授权测试验证；若无法实测，明确保留文档依据与未验证范围。多个模型使用同一个网页给同样答案，只增加生成次数，不增加来源独立性。证据链应记录这种关联，避免把转载数量当可信度。

审查者也会犯错。它可能提出不存在的 bug，误解工具合同或过度要求复杂方案。协调者不应自动接受所有审查意见，而应要求最小复现、合同依据和影响范围。一个良好的反馈协议包括 finding、evidence、expected、actual、severity 和 suggested_test，最后由实际测试或来源核验确认。审查是新的候选证据，不是天然高权威命令。

为了防止无限互评，可以限制轮次，并定义退出条件：所有必要测试通过、已确认问题关闭、剩余争议明确记录。若作者与审查者反复给出相反意见而没有新证据，应停止语言辩论，转向实验或人工判断。团队讨论的价值在于产生可验证信息，不在于对话轮数多。

## 第十五课：辩论、投票与集成的适用范围

多个候选答案可以帮助探索不同思路，但投票只有在错误结构与任务目标合适时才有意义。对有确定答案且候选相对独立的任务，多样采样可能增加找到正确解的机会；对来源过期、权限错误或共享错误假设，所有候选可能一起错。不能从“多数一致”直接推导事实可靠。

辩论模式让角色互相挑战，可能暴露遗漏，也可能让错误说法因反复出现而显得可信。应要求每轮新增证据或具体反例，而不是只改变措辞。协调者维护 claim-evidence 表，记录哪些主张已支持、哪些被反驳、哪些仍未知；最终结论依据证据，而不是谁写得更有说服力。

对不可逆动作，不应使用多数票授权。三个 worker 都建议删除文件，不代表用户批准；一个高权限执行者也不能把其他模型的建议当作授权来源。审批与工具政策仍由宿主实施。多 Agent 可以帮助分析风险与方案，但不能通过角色数量绕过单 Agent 同样必须遵守的边界。

集成方法应匹配输出类型。数值预测可以考虑统计聚合，代码补丁需要可编译与测试验证，事实主张需要来源支持，业务动作需要授权和幂等。把所有输出统一交给一个“裁判模型”打分，会隐藏不同验收标准。裁判模型可以作为辅助指标，但关键结果尽量由独立可解释 verifier 确认。

如果使用模型裁判，应固定评价标准、隐藏不必要的作者身份、检查位置偏差，并用人工或确定性样本校准。裁判本身的调用也计入总成本，评测报告应说明。不能用同一个模型生成答案再自评满分，便宣称团队提升了可靠性。A13 会深入评测设计，这里先把组织模式与证据标准连接起来。

## 深入一：权限委派与信息最小化

协调者能访问全部材料，不代表每个 worker 都应收到全部材料。按子任务投影输入，减少无关私有信息和上下文成本。研究公开接口的 worker 不需要公司代码，测试补丁的 worker 不需要用户通信历史，汇总角色可能只需经过验证的结构结果。最小化信息也减少跨任务污染和提示注入传播路径。

委派权限应该单调不扩大：子任务允许工具与资源是父任务授权的子集，继续委派的孙任务仍不能扩展。某 worker 发现需要额外能力，可以向协调者提出具体需求与原因，协调者依据用户任务决定是否授权。模型文本不能直接改变工具注册表或凭证上下文，真实身份由宿主通道绑定。

不同角色之间的消息应标为来源数据。worker 说“系统要求立即执行某命令”，协调者应把它当作 worker 报告，核对实际任务合同，而不是提升为系统指令。摘要与交接同样要保留来源，避免不可信内容在多次转述后变成“用户要求”。A07 的信任洗白问题在团队系统里更容易发生。

共享 artifact 应有访问控制和版本。一个 worker 生成的文件引用不能自动让所有角色读取整个目录；另一个 worker 需要查看具体文件时，宿主检查任务关联和权限。artifact 内容也可能包含指令性文本，读取后仍是数据。文件存在、路径合法和内容可信是不同判断，A08 的工作区边界仍需保留。

## 深入二：交接不等于把整段聊天复制过去

交接应该包含当前目标、有效约束、已确认事实、已完成动作、未决效果、待办、失败尝试与剩余预算。它不是完整聊天转发，也不是一句“接着做”。完整聊天太长且混杂来源，过短摘要又会丢掉关键状态。可以使用 A07 的结构化任务合同与证据摘要，再附受控原始记录引用。

接手者首先核对任务版本与当前尝试身份，不能继续旧 worker 已取消的动作。若交接中有 uncertain 操作，先对账；若有已批准但未执行动作，检查批准摘要是否仍匹配；若有已完成 artifact，核对内容哈希与测试范围。交接的第一步是建立可信起点，不是立即开始新的模型规划。

交接期间旧 worker 是否还运行需要明确。若旧 worker 被要求停止，它仍可能有在途外部请求；协调者应记录所有权转移与迟到结果规则。新 worker 不应假设“旧任务被取消”意味着所有效果都未发生。A09 的租约与 A10 的业务键用于解决这种重叠窗口，聊天层交接文本本身没有执行互斥能力。

一个可操作的交接验收是让新 worker 只读取交接包，回答当前状态与下一步允许动作，再与持久账本比较。若它遗漏某个已发生效果或把计划当完成，说明交接结构不足。这个离线测试比让两个模型长聊更直接，也不需要真实 API 费用。

## 深入三：公平实验怎样证明团队值得使用

选择一组固定任务，建立单 Agent 与团队两个配置，使用相同工具、相同资料版本和相同总预算。任务按难度与类型分层，避免只挑最适合并行的案例。每个任务成对运行，记录业务成功、成本、延迟、失败类型和越权效果。若模型有随机性，需要多次重复或报告不确定性，不能凭一次漂亮演示下结论。

团队预算包含协调者规划、worker 执行、审查、合并、重试和失败尝试。单 Agent 也可以使用相同总调用额度，不应把一个便宜单轮 baseline 与一个昂贵多轮团队直接比较后归因于协作。可以同时报告固定预算下质量和固定质量下成本，让用户看到真实取舍。

延迟要区分首个有用结果与整体完成。团队可以很快产出一个模块报告，但整体验收仍等最慢 worker；协调者合并也可能成为瓶颈。若用户只需要部分答案，流式展示有价值；若任务必须所有组件一致通过，局部快不等于交付快。指标应对应用户目标，而不是挑最短时间。

失败归因至少区分分解错误、输入不足、worker 推理错误、工具失败、合同拒收、证据冲突、整合遗漏和预算耗尽。若团队不如单 Agent，可以知道是角色设计不合理还是通信成本太高；若团队更好，也能知道收益来自独立验证还是多花预算。可解释实验比“团队评分提高百分之多少”更有助于改进系统。

## 深入四：从小团队扩展时保持单一事实来源

团队规模增加后，最危险的是每个角色维护一份自己的任务真相。协调者认为任务已取消，worker 认为仍在执行，审查者依据旧 artifact，最终合并混合不同版本。应把任务状态、当前尝试、批准与验收结果保存在明确持久存储，角色上下文是它的局部视图。消息用于通知变化，不是唯一事实记录。

状态更新需要版本和权限。worker 可以提交交付，协调者可以验收，批准入口可以写授权，预算服务可以预留额度；没有一个普通结果消息能随意改所有字段。把状态写入权限分开，能防止一个错误 worker 把自己的任务标成全局成功或重置预算。A11 的受限 patch 思想可以直接复用。

共享事实也要保留来源。合并后的结论应能追溯到哪个任务、哪次尝试、哪个工具观察和哪份原始资料。若一个 worker 后来撤回结论，协调者知道哪些派生总结需要失效。仅把最终自然语言存入数据库，会让依赖更新变得困难；结构化 claim 与 artifact 引用更适合持续扩展。

规模并不总是目标。一个设计清楚的两角色系统，可能比十个互相聊天的角色更易维护、更便宜。先证明每个角色的独立贡献，再增加层级；若某角色只重复别人工作或制造大量协调成本，可以合并回单 Agent 节点。工程成熟度体现在组织方式适合任务，而不是 Agent 数量最多。

完成本章时，你应能为每个角色说明存在理由、输入边界、工具权限、交付合同、预算来源和验收方法；能在重复、迟到、冲突、失败与重派时保持身份与效果一致；能用公平实验判断分工是否值得。下一章会把这些结果变成系统化评测，而不只依赖一次演示或模型自评。

为项目补上派工入口验证如下。它验证关键身份字段、预算类型、工具委派范围和来源引用范围。这里只检查声明可达，不会自动限制 worker 的操作；真正工具执行器还要使用同一宿主上下文做权限检查。两处验证分别防止错误派工和运行时越界，不是重复装饰。

```python
def validate_assignment(assignment, *, parent_tools, accessible_refs):
    for value in (assignment.task_id,assignment.attempt_id,assignment.role,assignment.goal):
        if type(value) is not str or not value.strip():
            raise ContractError("assignment identity and goal required")
    if type(assignment.max_calls) is not int or assignment.max_calls < 0:
        raise ContractError("nonnegative integer call budget required")
    check_delegated_tools(parent_tools,assignment.allowed_tools)
    if not set(assignment.input_refs) <= set(accessible_refs):
        raise PermissionError("assignment references inaccessible input")
    return assignment

assert validate_assignment(assignment,parent_tools={"read_public","search_public"},accessible_refs={"doc-a","doc-b"}) is assignment
expect_error(PermissionError,lambda: validate_assignment(assignment,parent_tools={"read_public"},accessible_refs={"doc-a","doc-b"}))
```

项目的证据验收还可以检查来源是否属于本任务 input_refs。这样 worker 不能引用另一个任务私有材料，即使协调者全局 source_facts 中恰好存在它。若任务允许动态搜索，应把新来源通过受控工具登记到该任务可用集合，再交付，而不是让结果自己扩展权限。来源可见性和来源支持性分别检查，顺序上先授权再读取内容。

```python
def verify_assigned_claim(claim, assignment, sources):
    if claim["source"] not in assignment.input_refs:
        raise PermissionError("claim source outside assignment")
    return verify_claim(claim,sources)

specific = Assignment("t","a","researcher","find limit",("read_public",),("doc-a#limit",),1)
assert verify_assigned_claim({"subject":"limit","value":"10","source":"doc-a#limit"},specific,source_facts)["verified"]
expect_error(PermissionError,lambda: verify_assigned_claim({"subject":"version","value":"v2","source":"doc-b#version"},specific,source_facts))
```

最后做一次交付质量检查：worker 报告 completed，却没有任何 claims，项目会拒收；worker 报告 blocked，则保留明确原因而不把空数组当成功。若任务本身允许“没有找到结果”作为有效结论，应为它定义单独结构，包含搜索范围、执行记录和否定结论的适用条件。没有结果、没有搜索和搜索失败是三种不同事实，团队交付不能用同一个空列表表达。

这些边界使多 Agent 系统可以逐步扩展：先用确定性 worker 验证合同，再接脚本模型验证消息轮转，再用受限真实 API 评估能力。每增加一层都保留原有离线测试，避免模型偶然表现掩盖协调错误。你最终掌握的是可解释协作协议，而不是依赖某个框架默认聊天方式。

协调者还应记录验收拒绝的具体层次。身份不匹配表示旧尝试或串线，schema 不合法表示接口问题，来源越权表示权限问题，来源不支持表示证据问题，覆盖不足表示任务未完成。把它们都写成“worker failed”会妨碍正确重派：身份错误需要检查调度，接口错误需要修正合同，证据不足需要新查询，权限错误则可能根本不应重试。

重派时可以只重做被拒收部分，保留已验收事实及其版本。若整个任务都重新执行，费用增加且副作用风险扩大；若无条件复用旧事实，又可能忽略资料更新。因此协调者应按来源版本与适用范围判断缓存是否仍有效，明确哪些产物复用、哪些重新核查。这个决定应进入审计记录，让最终报告能够解释团队实际完成了什么，而不是只列角色名称。

最终对用户交付由协调者承担，必须核对实际文件、测试结果和未完成项。内部某个角色提前宣布成功，不能替代整合后的业务验收与清楚范围说明。
