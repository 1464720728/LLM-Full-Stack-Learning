# S15 从 Win11 与 5070 Ti 到共享 H100：每次实验都得到可复核证据

你的个人 PC 可以随时练习，Linux 服务器的 CPU 与 Docker 可以持续使用，H100 则用于真实设备执行、框架和多卡通信问题。本章把这三种条件组织成一条连续路线：先在 CPU 上把输入、状态和验收做正确，再在一张已分配 GPU 上验证真实路径，最后为明确的并行问题扩展到两卡或八卡。无需为每个概念占满服务器，也不把 GPU 实验永远留成一句建议。

先修基础卷环境、Linux、进程、网络与 Docker，S08 的测量和 S13–S14 的容量诊断。本章不会假设你已经熟悉 CUDA 环境。我们逐个解释驱动、运行时、工具链、Python 包、容器、模型身份与设备编号，读完后应能知道每条命令在检查什么，以及检查失败时应该回到哪一层。

**证据边界：本轮编写实际执行了 CPU、脚本 dry-run、并发线程和本地 HTTP 练习；没有在你的 5070 Ti、H100 或真实 vLLM/SGLang GPU 服务上运行。所有 GPU 表格都是实验设计，结果栏必须由真实运行填写。** 官方安装与接口资料于 2026-11-01 核查；会变化的安装矩阵应在你安装时再次核对，不把本书某条命令当成永远兼容的包组合。

## 1. 第一课：三种环境共用学习目标，但不共用一套隐含状态

Windows 原生、WSL Linux 与远端 Linux 是三个不同运行环境。它们有各自的文件路径、Python 解释器、安装包、进程和网络命名空间。你在 Windows 安装了 NumPy，不代表 WSL 的 Python 能导入；在远端激活虚拟环境，也不会改变本机终端。先确认“我现在在哪台机器、哪个 shell、哪个 Python”，比遇错后重复安装更有效。

个人 PC 的日常路线可以把课程仓库与 Linux 练习放在一个固定 WSL 发行版中。这样命令、路径和服务器更接近，但仍要理解 WSL 不是远端 H100。显卡型号、驱动、可用内存和支持的 kernel 路径不同；一个针对 H100 的实现不能仅因名字里有 CUDA 就自动在 5070 Ti 上运行。

当前 [vLLM GPU 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) 以 Linux 为支持入口，并说明 Windows 可通过 WSL 路线使用，原生 Windows 并非同一官方运行路径。这个事实只决定环境选择，不保证某个具体模型、量化格式或 attention backend 已支持你的整套组合。首次成功应从很小、容易核对的计算和模型开始。

服务器 CPU 适合长期进行源码阅读、单元测试、事件回放、HTTP 协议和资源账本练习。公司已有生产 Python、容器与库不应成为练习的修改目标；创建自己可删除的环境和源码副本。隔离的价值是知道改了什么、能否恢复，以及测试运行的到底是不是你编辑的那份代码。

```bash
python -m labs.foundations.environment
python -m labs.kernels.inspect_environment
python -m unittest discover -s tests -p 'test_serving*.py'
```

第一条报告解释器和工作目录，第二条只检查包与命令是否存在，不默认请求 GPU，第三条运行本卷离线回归。缺少依赖时按基础卷安装 CPU 要求，不先安装整个推理框架。先把可观察的地基搭好，后面设备错误才不会与 Python 路径错误混在一起。

## 2. 第二课：驱动、CUDA 运行时、Toolkit 与 wheel 各负责什么

驱动负责操作系统与设备交互；CUDA 运行时和相关库供程序调用；Toolkit 通常包含编译器、头文件与开发工具；Python wheel 可以包含已编译扩展并依赖特定库组合。它们不是同一个版本号。nvidia-smi 显示的 CUDA 兼容信息，不等于当前 Python 包使用的 CUDA 构建版本；nvcc 的版本也不自动决定已安装 PyTorch wheel 的版本。

已有 CUDA wheel 能运行，并不一定需要你先安装完整 Toolkit；但要编译自定义算子或某些框架扩展时，可能需要兼容的编译工具链。反过来，nvcc 能执行也不能证明 Python wheel 的设备代码包含你的 GPU 架构。报 no kernel image、架构不支持或动态库符号错误时，应检查二进制与依赖，而不是只看驱动存在。

WSL 有特别重要的边界。[NVIDIA 的 CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/) 说明 GPU 驱动来自 Windows 宿主并映射到 WSL；不要在 WSL 内安装普通 Linux NVIDIA 显示驱动覆盖这一路径。需要开发工具时使用该指南对应的 WSL Toolkit 安装方式。远端 Linux 服务器的驱动由服务器环境管理，不能照搬 WSL 的宿主关系。

不要把一份网上安装命令同时用于所有机器。先从官方安装页选择操作系统、Python、设备与包版本，再把最终选定组合记入环境记录。若使用容器，镜像提供用户态依赖，但仍通过宿主驱动访问 GPU；容器不是把所有驱动兼容问题全部包起来。出现不匹配时，先保存完整错误和版本指纹，再选择兼容组合。

```bash
python -m labs.kernels.inspect_environment --gpu
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv
nvcc --version
```

第一条显式查询 PyTorch 可见设备；第二条读取设备与驱动信息；第三条仅在安装了 Toolkit 时有意义，找不到 nvcc 不等于所有预编译推理必然不能运行。不要把完整环境变量导出到公开日志，因为其中可能包含 API key。只记录与实验相关的白名单字段。

## 3. 第三课：设备可见性、物理编号与 rank 不能混为一谈

假设你被分配物理 GPU 三，容器只暴露这一张卡。容器内 PyTorch 很可能把它编号为 cuda:0，因为这是可见列表中的第一张。这个零是相对编号，不是说程序跑到了物理 GPU 零。实验记录最好同时保存宿主 index、UUID、容器可见顺序与 local rank，让两侧可以对应。

两卡实验中，visible devices 为三和五，local rank 零和一分别映射到这两个可见设备。TP rank 也是通信组身份，不一定等于宿主卡号。S10 的 group 与 rank 表可以扩展一列 physical device，防止日志里 rank 一被误解为物理卡一。跨节点时每台机器都有 local rank 零，更需要全局 rank 与节点身份。

设备数量也不能仅按字符串长度算。列表零、零有两个元素，但实际上重复同一设备。本卷启动脚本现在要求规范的非负整数编号且互不重复，拒绝这种错误。它仍不替你验证设备是否已分配、是否空闲或是否符合模型拓扑；那些是实际环境检查的工作。

```python
def device_map(physical_ids):
    if not physical_ids or any(type(x) is not int or x < 0 for x in physical_ids):
        raise ValueError("explicit nonnegative physical IDs required")
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("physical devices must be distinct")
    return [{"local_rank":rank,"visible_cuda_index":rank,"physical_index":gpu}
            for rank,gpu in enumerate(physical_ids)]
assert device_map([3,5])[1] == {
    "local_rank":1,"visible_cuda_index":1,"physical_index":5}
try:
    device_map([0,0])
except ValueError:
    pass
else:
    raise AssertionError("duplicate physical device")
print(device_map([3,5]))
```

这段映射是课程采用的可见列表顺序约定，不是对所有集群启动器的自动发现。实际启动器可能重排环境，容器也可能使用 UUID。你要把实际探测结果与计划表对照；发现不一致就停止该实验，先修设备身份，而不是继续跑完再猜数据属于哪张卡。

共享服务器上“当前显存空闲”与“可以长期占用”也不同。你的使用授权允许必要实验，但仍应选择已分配设备和有限窗口，记录开始结束，主动回收自己创建的服务。课程边界是让实验可控、可解释，不要求你每个命令都重新征求许可。

## 4. 第四课：先写实验卡，让输入与停止条件都具体

一张实验卡包含问题、假设、唯一变化轴、模型身份、环境、设备、输入、输出上限、请求数、并发、最长时间、停止条件和产物。它不是行政表格，而是防止“开着服务随便看看”最后没有可靠结论。问题越具体，资源越容易控制，失败也越容易定位。

例子：“一张已分配 H100，五分钟服务窗口，固定小模型与 eager，比较输入二百五十六和一千零二十四 token 的首文本延迟；每档四请求、输出最多十六、并发一；记录实际 token 数、原始响应、启动日志和页池统计。”这个实验不能回答高并发吞吐，也不能回答 graph 的收益，但可以回答长度变化是否与 prefill 成本一致。

停止条件应可观察，例如服务未在规定窗口 ready、首个数值校验失败、连续请求错误、峰值超过任务预算、已达到总输出上限。不要只写“出问题就停止”，因为实验过程中人容易为了得到结果继续加大限制。上限是实验定义的一部分，失败时保存证据再设计下一次，而不是无限续跑。

```python
def check_card(card):
    required = {"question","devices","max_seconds","requests","max_output_tokens","concurrency"}
    if not required <= card.keys():
        raise ValueError("missing experiment fields")
    if not isinstance(card["question"],str) or not card["question"].strip():
        raise ValueError("write a concrete question")
    devices = card["devices"]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("explicit distinct devices required")
    for key,lo,hi in (("max_seconds",1,300),("requests",1,8),
                      ("max_output_tokens",1,64),("concurrency",1,4)):
        if type(card[key]) is not int or not lo <= card[key] <= hi:
            raise ValueError(f"lesson bound: {key}")
    return {"max_generated_tokens":card["requests"]*card["max_output_tokens"],
            "max_seconds":card["max_seconds"],"devices":list(devices)}
card = dict(question="one-card short serving smoke test",devices=[3],
            max_seconds=300,requests=4,max_output_tokens=16,concurrency=1)
assert check_card(card)["max_generated_tokens"] == 64
print(check_card(card))
```

这个校验器只覆盖课程上限，不是集群资源授权器，也没有验证模型、预算和 dtype。它展示把“最多四条、每条十六”转为总输出六十四的明确约束。输入 token 与预热请求也应计入资源计划，不能只限制输出却发送几百万 token 的 prompt。

## 5. 第五课：模型身份比模型名字更精确，下载准备与计时分离

课程的小型真实服务起点是 `Qwen/Qwen2.5-0.5B-Instruct`，固定 revision 为 `7ae557604adf67be50417f59c2c2f167def9a775`。[固定 config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json) 给出二十四层、hidden 八百九十六、十四个 query 头和两个 KV 头。它用于验证真实加载与服务链，不代表所有版本镜像都自动兼容。

准备目录需要权重、config、tokenizer 与生成相关配置，来源 revision 一致。下载时的 main 可能变化，报告只写模型名无法保证下次相同。可以保存文件大小与摘要，尤其是 config、tokenizer 与权重索引；大权重全量哈希需要时间，安排在 GPU 实验前完成，不要把校验开销混入推理延迟。

下面程序只演示对临时文件生成 manifest。实际模型目录使用同样原则，但应限定文件清单，不递归扫描整个用户目录，也不把私有绝对路径写进公开报告。manifest 使用相对文件名，记录内容身份，而不是假定同名文件一定相同。

```python
import hashlib,json,tempfile
from pathlib import Path

def manifest(root,names):
    root = Path(root).resolve()
    rows = []
    for name in names:
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("manifest only reads named files inside the model root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                digest.update(chunk)
        rows.append({"name":name,"bytes":path.stat().st_size,"sha256":digest.hexdigest()})
    return rows

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root/"config.json").write_text('{"layers": 2}\n')
    rows = manifest(root,["config.json"])
    assert len(rows[0]["sha256"]) == 64
    print(json.dumps(rows,indent=2))
```

镜像也应固定 digest，而不是只用 latest 标签。本地镜像检查证明你将运行的具体内容已经存在；它不证明镜像来源可信或模型兼容，仍需核查维护者与安装说明。不要从教材复制一个编造的 sha256；必须使用你实际准备的镜像元数据。

下载、依赖安装和首次 kernel 编译可能消耗显著时间，最好分成准备阶段与计时阶段。若研究冷启动，可以单独测从进程启动到 ready 的时间，但要说明是否包含下载与编译缓存。预热后请求的延迟不能与完整冷启动相混，反之也不能把首次编译开销当每 token 常态。

## 6. 第六课：逐行读有界启动脚本，理解限制保护哪一层

打开 `labs/serving/h100_server.sh`。它默认 dry-run，先解析 run、tp 与 allow-multi-gpu；要求镜像、绝对模型目录和显式 GPU 列表。TP 课程档位为一、二、八；设备数量必须匹配，重复 ID 拒绝，多卡必须显式开启。所有检查发生在真实 Docker run 之前。

脚本使用 Bash 数组保存命令，而不是把用户值拼成一个字符串交给 eval。这能保留路径中的空格与参数边界，减少 shell 解释风险。打印时使用 printf 的 q 格式，让你看到可检查的转义命令。数组并不自动验证参数语义，因此前面的目录、设备和模式检查仍不可省略。

```bash
cmd=(docker run --rm --pull=never --name "$name" --gpus "$gpu_spec"
  --cpus 4 --memory 24g --shm-size 2g
  -p 127.0.0.1:18080:8000
  -v "$COURSE_MODEL_DIR:/model:ro"
  --entrypoint vllm "$COURSE_VLLM_IMAGE"
  serve /model --host 0.0.0.0 --port 8000 --tensor-parallel-size "$tp"
  --gpu-memory-utilization 0.35 --max-model-len 2048 --max-num-seqs 4
  --max-num-batched-tokens 2048 --enforce-eager)
```

这是脚本内部摘录，依赖前面的变量，不应单独执行。docker run 创建容器；rm 表示退出后删除这个容器；pull=never 强制只使用本地镜像，避免检查与启动之间又触发自动拉取；name 使用用户 ID 和当前脚本 PID 生成精确名称。gpus 只暴露指定设备，逗号列表有 Docker 所需的额外 quoting；不要改成 all 来省事，那会改变实验资源范围。

cpus 四限制容器 CPU 配额，并非固定绑定四个核；memory 二十四 g 限制主机内存，不是 GPU 显存；shm-size 二 g 设置共享内存区域大小。Docker 的 [资源约束说明](https://docs.docker.com/engine/containers/resource_constraints/) 区分 CPU 配额、硬内存限制与 swap 规则。当前脚本未显式设置 memory-swap，因此不能宣称“主机内存加 swap 总共绝不超过二十四 g”。

模型目录只读挂载到 /model，防止服务修改准备好的权重；运行时缓存可能仍写在容器可写层，退出后会丢失，这是可复现准备与临时缓存的另一个边界。宿主端口仅在 loopback 发布，容器服务监听所有容器接口，两者并不矛盾：前者限制宿主暴露范围，后者允许端口映射到达容器服务。

gpu memory utilization 零点三五是框架预算参数，不是 GPU 硬隔离，也不是“当前空闲量的百分之三十五”这一普遍保证。max model len 限单请求上下文，max num seqs 限活动序列，max num batched tokens 限每轮预算；S05 与 S13 已解释这些约束不同。enforce eager 让首次实验避开图捕获变量，之后研究图模式时再做受控对照。

## 7. 第七课：dry-run、真实启动、ready 与停止是四个阶段

先设置真实已准备的环境值，再 dry-run。下面明确包含占位符，不可原样运行；填写过程应依据你自己的模型目录、镜像摘要和已分配设备。教材不会替你猜公司路径或镜像版本，也不会自动下载大模型。

```bash
export COURSE_VLLM_IMAGE='vllm/vllm-openai@sha256:填写真实已核验摘要'
export COURSE_MODEL_DIR='/填写真实绝对路径/固定模型'
export COURSE_GPU_IDS='3'
bash labs/serving/h100_server.sh
```

dry-run 验证输入格式并打印命令，不创建容器，也不证明镜像实际存在或能运行。真实执行前，使用同一镜像检查 vllm serve 的 help 与当前参数支持。脚本 run 分支会执行 docker image inspect，要求镜像本地存在；这里不把临时网络拉取隐藏在 GPU 实验中。

```bash
bash labs/serving/h100_server.sh --run
```

服务启动不等于立即 ready。先观察日志，确认模型加载、worker 初始化和接口就绪，再在第二终端请求 health 或模型列表。若某镜像端点不同，按该版本官方服务文档检查，不根据其他服务的路径猜。读取模型列表可以确认客户端应该发送的 model ID，常见本地路径 ID 与仓库名未必一致。

脚本使用 GNU timeout 的三百秒窗口，先发 TERM，再经过额外等待才强制结束。trap 通过精确容器名停止本次容器，不进行全局 docker 清理。窗口包括加载时间，因此慢启动可能还没完成请求就到期；报告应记为“窗口内未 ready”或“实验到期”，而不是自动认定模型崩溃。

退出后检查自己的容器不存在、宿主端口不再监听，以及没有本次遗留 worker。不要用 killall python、docker stop 所有容器或清空共享缓存来恢复环境。资源回收应从创建时就知道对象身份，这与 S03 的页所有权是同一个工程原则，只是对象从内存页变成进程和容器。

## 8. 第八课：第一个真实请求先验证协议与内容，再开始计时

本卷 `http_benchmark.py` 默认只打印计划，必须显式 execute 才发送请求。它接受 completions 文本流式子集，固定短 prompt、temperature 零、有限请求数与输出上限。它不是万能 API 适配器；若服务只提供聊天模板接口，应先写对应适配并说明时间边界，不把 chat 的 delta 格式误当 text。

```bash
python -m labs.serving.http_benchmark --model /model
timeout --signal=TERM --kill-after=5s 90s python -m labs.serving.http_benchmark --model /model --execute --requests 4 --max-tokens 16
```

第一条不联网。第二条只向默认本机 loopback 服务发送四个串行请求，输出最多六十四 token；外层 timeout 提供整个探测进程的边界。模块内部 urlopen 的三十秒 timeout 不应当作严格总 deadline，因为流式读取可能持续产生数据。外层进程停止后，服务端取消是否传播仍要结合框架日志检查。

measure 在每个请求前记录 perf_counter，遇到第一个非空文本片段记首文本时间，记录片段间隔与字符数。SSE 一个片段可能包含多个 token，也可能只含元数据，因此字段叫 ttft_text 和 inter_chunk，不能无条件改名为逐 token ITL。若服务提供 usage，可以读取 completion tokens；没有就保留缺失，不用字符数猜 token 数。

本探针采用明确的 completions 流协议：usage 在最终 [DONE] 标记之前，收到错误事件、缺少最终标记或标记后又有 data 事件都拒绝记成功。它不宣称适配任意提供商的所有流格式；其他协议需要独立适配并保留同样的完成验收。

第一轮内容验收包括返回的模型身份、错误字段、结束标记、输出非空与长度合理性，以及服务是否仍健康。temperature 零有助于对照，但 S12 已说明不同数值路径仍可能影响 argmax；不要要求所有硬件逐字相同而没有误差与确定性前提。若做框架正确性，最好在相同 token 输入下比较 logits 或固定参考输出。

两个探针统一禁止 HTTP 重定向，要求显式配置最终端点。这保证 loopback 初始地址不会通过三零二跳转绕到其他目标，也避免把认证请求悄悄交给新的地址；若服务返回跳转，先核实最终地址，再按相应本地或远端规则配置。

API key 如需使用，应从指定环境变量读取，不写入命令参数、仓库或实验 JSON。默认本机实验不需要付费模型 API；远端接口需要明确端点和协议适配。本章硬件证据以你可控的本地服务为主，避免把第三方服务排队与网络延迟当成自己的 GPU 表现。

## 9. 第九课：并发探测真正同时发请求，不能拿串行脚本猜吞吐

原有 http_benchmark 一次等待一个请求完成，是串行封闭负载。为了研究两个或四个请求共同调度，本卷新增 `labs/serving/concurrent_probe.py`：最多八个请求、最多四个工作线程、每请求最多六十四输出，只允许 loopback，默认 dry-run。它复用同一 SSE 解析与时间定义，使协议差异不会混入比较。

```bash
python -m labs.serving.concurrent_probe --model /model --requests 4 --concurrency 2 --max-tokens 16
timeout --signal=TERM --kill-after=5s 90s python -m labs.serving.concurrent_probe --model /model --requests 4 --concurrency 2 --max-tokens 16 --execute
```

run_bounded 把有限任务提交到线程池，线程数限制同时进行的调用。每个任务记录从整批提交起到线程开始的 client queue 时间、调用墙钟时间、成功或错误类别。结果按原始 index 排序，完成顺序可以不同。一个请求失败不会让其他结果消失，但 CLI 最终返回非零，避免部分失败被自动化误记为全部成功。

client queue 是负载生成器里的等待，不等于服务端排队。一次提交八条、线程数二时，后六条要等待前面调用完成才能真正发出。它是固定最大在途数的封闭负载，不是独立到达的开放负载；S08 的排队结论需要按这个模型解释。报告全批 wall 和每请求时间时，明确是否包含客户端等待。

线程池退出会等待运行中的任务结束，因此它本身不保证硬总时长。外层 timeout 是本章实验边界；若要实现细粒度取消、deadline 与服务端 drain，需要继续完成 S16 的服务项目。不要在文档里因为出现 max_workers 就声称所有资源都已被完整限制。

下面用两线程 barrier 做离线验收，不依赖网络或计时阈值。每两个调用在同一轮 rendezvous，证明确有两个工作同时在途；其中一个故意失败，其他结果仍保留，最终活动计数归零。

```python
import threading
from labs.serving.concurrent_probe import run_bounded
barrier = threading.Barrier(2,timeout=3)
lock = threading.Lock()
state = {"active":0,"peak":0}
def fixture(index):
    with lock:
        state["active"] += 1
        state["peak"] = max(state["peak"],state["active"])
    try:
        barrier.wait()
        if index == 2:
            raise ValueError("intentional local fixture")
        return {"value":index*2}
    finally:
        with lock:
            state["active"] -= 1
report = run_bounded(fixture,count=4,concurrency=2)
assert state == {"active":0,"peak":2}
assert report["successful"] == 3 and report["attempted"] == 4
assert report["rows"][2]["error_type"] == "ValueError"
print({"thread_state":state,"successful":report["successful"]})
```

这证明客户端并发边界与错误聚合，不能证明服务器连续批处理确已发生。要证明后者，需要服务 batch trace 或调度指标显示同一轮包含多个请求。HTTP 同时发出只是必要实验输入，服务内部也可能像 mock_server 一样用锁串行运行。

## 10. 第十课：单卡实验矩阵从长度、并发、缓存和图模式逐项推进

第一组固定并发一与输出上限，改变实际 tokenized 输入长度，例如二百五十六、五百一十二、一千零二十四。不要用重复字符数量代替 token 数；同一 tokenizer、special tokens 与模板都影响长度。记录实际输入 token，并确保输入加输出预算不超过 max model len。

第二组固定输入输出，改变并发一、二、四，记录每请求首文本、总 wall、成功率、实际输出 token 和服务队列。小模型并发提升可能增加吞吐，也可能因客户端、调度或启动开销看不明显。实验允许结论是“该规模下差异不足以解释”，不能为了符合预期删掉慢样本。

第三组研究前缀缓存。先冷请求建立缓存，再提交共享前缀的新后缀，记录实际命中 token。四条请求同时进入时，第一条可能尚未发布可复用状态，不能保证其他三条都命中。热缓存实验还要固定模型、模板、adapter 与 namespace；修改文本但 token 前缀不同，不是同一个命中测试。

第四组比较 eager 与图模式，保持模型、dtype、输入和采样不变。分别记录启动到 ready、第一次请求、有限预热后请求。图模式可能引入捕获成本与持久内存，收益依赖形状和负载；某个 backend 或模型不支持的组合应明确 skip，不把失败强行归为性能差。

| 组别 | 唯一主要变化 | 首要观察 | 不能直接回答 |
|---|---|---|---|
| 长度 | 输入 token 数 | 首文本、prefill 与页数 | 高并发公平性 |
| 并发 | 最大在途请求 | 吞吐、尾延迟、队列 | 开放负载饱和点 |
| 缓存 | 冷与真实命中 | 命中边界、首输出变化 | 所有租户共享收益 |
| 图模式 | eager 与受支持图路径 | 冷启动、稳定态与内存 | 所有形状都加速 |

每组先做极少请求验证内容，再做有限重复。预热请求、失败请求和超时都写入记录，不只保留成功最快样本。服务最长窗口即将结束时不再添加新任务，先等待已有任务结束或按明确取消流程收尾，这比到最后一秒继续发送更容易解释终态。


### 10.1 把长度实验变成可重复的输入文件

默认探针的 prompt 很短，不能用它检验一千 token 的 prefill。现在两个 HTTP 探针都支持 `--prompt-file`，读取一个本地 UTF-8 文件，最多六万五千五百三十六 byte，空文件、非法 UTF-8 和超限文件会明确拒绝，不悄悄截断。dry-run 只显示字节数，不把完整内容打印出来。这个限制控制请求体规模，却不是 tokenizer 长度限制；两项都需要检查。

先用公开、可重复的文本构造候选输入，再用固定模型目录里的 tokenizer 记录实际 token 数。输入文件的摘要也保存到实验档案，使两次运行确实发送同一内容。不能为了凑到目标长度随意删掉后半段，却仍使用原任务答案评测；容量实验可以使用无任务语义的合成文本，质量实验则必须保留题目完整性。

```python
import hashlib,tempfile
from pathlib import Path
from labs.serving.http_benchmark import load_prompt
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/"public-prompt.txt"
    path.write_text("This is a public serving experiment. "*32,encoding="utf-8")
    prompt = load_prompt(path)
    print({"utf8_bytes":len(prompt.encode()),
           "sha256":hashlib.sha256(prompt.encode()).hexdigest(),
           "token_count":"requires the selected model tokenizer"})
    path.write_bytes(b"x"*65537)
    try:
        load_prompt(path)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized fixtures must not be truncated")
```

实际长度文件可以命名为 prompt-short.txt 与 prompt-long.txt，文件名只表达相对分组，实际 token 数写进记录。使用如下命令时，文件必须是你已准备并核对的公开输入；默认仍然 dry-run，确认后再加 execute 并套外层时间限制。

```bash
python -m labs.serving.http_benchmark --model /model --prompt-file prompt-long.txt --requests 4 --max-tokens 16
python -m labs.serving.concurrent_probe --model /model --prompt-file prompt-short.txt --requests 4 --concurrency 2 --max-tokens 16
```

如果使用 Transformers tokenizer，加载应指向已准备的本地目录，设置 local_files_only，避免计时期间自动访问网络；对于本章已核查的标准模型，保持 trust_remote_code 关闭。若某个新模型要求自定义代码，先审查来源与具体代码，再单独建立环境。下载到磁盘的配置文件不天然等于可安全执行的程序，加载策略需要与模型来源相匹配。

特殊 token 与模板要写清楚。纯 completions 输入和聊天 messages 经模板渲染后的输入并非同一 token 序列；本地 encode 是否添加 BOS，也要与服务真实路径核对。可用服务 usage 的 prompt token 数作交叉检查，但字段是否包含缓存命中位置、模板或特殊 token，仍需按接口语义解释。两侧不一致时先修输入身份，再做长度性能结论。


## 11. 第十一课：设备计时与内存峰值需要正确同步边界

PyTorch 的 [CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html) 可以在设备流中记录时间标记，elapsed_time 返回事件间毫秒；读取前要确保结束事件完成。它测的是两个标记之间的设备执行区间，不包含你没有放进区间的 tokenizer、HTTP 或 Python 准备。端到端服务仍需墙钟测量。

下面是可选的一卡矩阵乘法探针，不下载模型，只做固定小张量与十次运算。缺少 torch 或 CUDA 时打印 SKIP；这不是通过 GPU 验证。只在已分配设备上执行，进程启动前设置可见卡。它用于检查设备执行与计时方法，不宣称代表 LLM 性能，也不比较两张不同架构设备的优劣。

```python
import json
try:
    import torch
except ImportError:
    print("SKIP: torch unavailable; no GPU experiment executed")
else:
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable; no GPU experiment executed")
    else:
        device = torch.device("cuda:0")
        with torch.inference_mode():
            torch.manual_seed(7)
            a = torch.randn(256,256,device=device,dtype=torch.float32)
            b = torch.randn(256,256,device=device,dtype=torch.float32)
            reference = a@b
            for _ in range(3):
                result = a@b
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                result = a@b
            end.record()
            end.synchronize()
            torch.testing.assert_close(result,reference)
            print(json.dumps({
                "time_kind":"CUDA_event_ms","iterations":10,
                "total_ms":start.elapsed_time(end),
                "tensor_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                "reserved_bytes":torch.cuda.memory_reserved(device),
                "device":torch.cuda.get_device_name(device)}))
```

reset peak memory stats 重置峰值统计，不释放内存；max memory allocated 描述该进程 PyTorch 张量分配峰值，不等于设备上所有进程或所有库的总占用。相关定义见 [PyTorch 峰值内存接口](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)。本例参考结果和输入仍常驻，所以峰值包含它们，而不是仅算一次 matmul 输出。

预热排除了部分首次初始化，但不能证明所有缓存都已稳定。十次平均也不是尾延迟分布。若研究小 kernel，应进一步考虑测量分辨率、启动开销与重复设计；若研究模型，应测真实形状和数据流。算子卷提供更深入的方法，本卷在这里建立计时边界与诚实报告习惯。

### 11.1 用一张结果表分清启动、预热和稳定态

建议每条记录保留 experiment ID、配置摘要、输入摘要、实际输入 token、输出上限与实际输出、并发档位、请求序号、是否预热、状态、首文本时间、完成时间、缓存命中和资源峰值。不要只保存平均值；四个请求原始结果很小，却足以让后续分析发现首轮异常或一条错误。

启动到 ready 是服务级记录，不能复制成每条请求的 TTFT。预热请求可以单独标记，但不能从资源消耗总量中消失。稳定态也只是这个短窗口内的近似，四条请求不支持对所有生产负载尾延迟作强结论。课程的目标是学会建立证据，不是用极少样本制造漂亮排行榜。

对比 eager 与图模式时，最初可以用 A-B-B-A 的交错顺序减少时间漂移影响：先运行一段 A，再 B，再 B，再 A，并保存每段环境。它仍不能消除所有共享资源变化，但比总在空闲时测 A、繁忙时测 B 更可解释。每次重启是否清除缓存、是否重新捕获，必须一致或明确作为实验轴。

若一张卡同时承担其他合法任务，记录观察到的资源竞争即可，不需要采集其他任务的私有内容。你可以缩小实验或改用已分配窗口；不能为了获得更干净数字干预别人的工作。测量噪声较大时，报告范围和限制比继续加压更诚实，也更符合实际工程中的共享环境。

结果分析先检查有效样本：协议是否完整、输出是否被截断、错误是否计入、输入是否一致。再比较首输出、生成阶段和总体吞吐。最后回到 S13 的容量预测解释峰值。如果预测与观测不符，优先检查页池预分配、加载副本、图持久缓冲和统计口径，而不是直接宣称公式失效。


## 12. 第十二课：两卡与八卡为明确问题服务，不把卡数当能力证明

升级多卡前，先完成单卡正确性、容量账本与停止演练。两卡实验可以验证 TP 矩阵分片、collective 次序、跨 rank 输出映射或单卡放不下的模型；八卡实验可以研究更大并行组与拓扑。它们的价值来自问题需要多个设备，不来自“有八卡所以必须全用”。

小型 Qwen 示例有十四个 query 头和两个 KV 头，不满足常见均匀 TP 八头分片条件。不能因为权重很小就直接 tp 八。两卡也需要核对当前模型类、backend 与 KV 布局；S13 的估算器只覆盖明确整除假设。若选择别的模型，先读取固定 config 与实现，不凭参数规模猜并行合法性。

```bash
export COURSE_GPU_IDS='3,5'
bash labs/serving/h100_server.sh --tp 2 --allow-multi-gpu
```

仍先 dry-run，真实启动再加 run。当前脚本是同节点 TP 教学入口，不是跨节点启动器，也没有自动实现 PP、EP 或 PD。研究这些模式需要单独的固定版本 recipe、通信与状态布局说明；S10–S11 已提供机制和 CPU 参考，GPU 实验应逐个验证对应契约。

多卡正确性先于速度。记录每 rank 设备、组成员、输入 shape、collective 顺序、输出数值和终态；如果一个 rank 没进入相同 collective，其他 rank 可能等待。先在 S10 的 CPU 协调式归约中练习身份与超时，再在真实 NCCL 环境中做小消息、少轮次实验，避免直接从完整大模型的通信挂起开始排查。

速度对照应同时报告单请求延迟与总吞吐。TP 增加通信，小模型可能更慢；DP 可增加副本并发却不缩短单请求；PP 增加阶段传递并有气泡；EP 受路由不均衡影响。把这些不同目标混成“多卡加速比”一个数字，会失去解释力。

### 12.1 从一天的学习任务安排到一次多卡窗口

日常练习可以把长时间任务放在 CPU：阅读固定源码、修改调度策略、运行数值对照、准备公开输入、建立容量表和分析已有 trace。GPU 窗口开始前，代码、模型、镜像和实验卡已经就绪，打开服务后只执行需要硬件回答的部分。这样每次使用 H100 都能集中产出证据，而不是大半时间等待依赖安装。

一次两卡窗口可以只回答三个问题：每 rank 真实拥有哪些张量；小输入的输出是否满足数值参考；通信与计算在这组形状下各占多少。先用有限小输入跑通，再增加一档长度；不用立即扩展到最大上下文和最大并发。若正确性失败，保留日志后结束性能部分，把问题带回 CPU 或源码层分析。

八卡实验应有两卡不能回答的理由，例如验证八个 rank 的通信次序、某个分片布局、跨多个互联域的影响，或单个模型确实需要相应容量。即便如此，也可以限制请求数、生成长度与总窗口。使用八卡并不意味着必须长时间跑满；一个完整的状态与数值验证可能只需很短的执行。

实验结束后立即整理最小结论和下一步。若观察到 TP 二比单卡慢，检查模型大小、消息尺寸与通信固定成本，再设计一个有针对性的形状对照；不要直接得出“TP 没用”。若缓存无收益，先确认实际命中与排队占比；不要靠重复相同命令期待偶然出现想要的数字。每个结果都应该推动一个明确假设，而不是只积累命令历史。

## 13. 第十三课：故障、完整练习与硬件实验档案

**故障一：Windows 能看到显卡，WSL 框架不能运行。** 先确认正在使用 WSL Python，再检查 Windows 驱动映射、WSL 设备可见性、wheel 与 GPU 架构支持。不要在 WSL 里盲装普通 Linux 驱动，也不要把 nvcc 缺失当唯一原因。保存只含必要字段的环境指纹，按层定位。

**故障二：dry-run 成功，真实启动失败。** dry-run 只证明参数构造，不证明镜像存在、Docker GPU runtime 正常、模型完整或框架支持。按 image inspect、容器设备、模型文件、启动日志依次检查。不要把所有检查压成一句“脚本坏了”。

**故障三：两卡配置实际只用了同一张卡。** 检查 GPU 列表去重、可见顺序、local rank 和 UUID。本卷脚本现在拒绝零、零和不规范编号。若启动器内部重排，记录真实映射并修正，不用日志中的 cuda:0 推断宿主物理零。

**故障四：客户端结束后服务仍占 GPU。** 客户端结束不等于服务退出；持久服务本来会保留模型和 KV 池。检查脚本窗口、精确容器名和 cleanup。若是请求取消未排空，则回到 S05/S11 的在途所有权，而不是直接把全部服务进程杀掉。

**练习一，环境边界。** Windows Python、WSL Python、远端 Python 的包与路径独立。参考操作是分别运行 environment 报告并比较 executable、platform、cwd；不能用一个终端 pip install 后假设另一个环境已变化。

**练习二，版本层。** nvidia-smi、nvcc、torch.version.cuda 分别反映不同层。答案需解释驱动兼容、编译工具链与包构建，而不是要求三个字符串完全一样。兼容性由官方矩阵和实际运行共同验证。

**练习三，设备映射。** 物理三、五对应可见零、一是本章约定。记录 UUID 可以在编号变化时识别设备；rank 是通信身份，不等于物理 index。重复物理编号必须在启动前拒绝。

**练习四，实验上限。** 四请求、每条最多十六输出，总上限六十四；预热与失败重试另计。不能只限制请求数而发送无限输入。完整实验卡包括时间、并发、输入和停止条件。

**练习五，模型身份。** 同模型名的 main 会变化；固定 revision、config、tokenizer 和权重清单才能复现。摘要例子验证的是文件内容，不是框架兼容，更不是模型质量。

**练习六，容器限制。** memory 是主机内存，gpu memory utilization 是框架预算，两者不能互换。cpus 是配额而非固定核；未配置 swap 不能声称总内存加 swap 已被同一值完全限制。

**练习七，只读挂载。** 模型目录 ro 保护权重，容器其他可写层仍可存临时缓存。退出后缓存是否保留取决于挂载设计。冷启动实验应说明缓存状态，不能把热缓存结果混入冷启动。

**练习八，端口。** 容器 0.0.0.0 与宿主 127.0.0.1 发布分别作用于不同网络边界。先在远端宿主验证，再通过 SSH 转发到 PC；不要为方便默认公开监听。

**练习九，SSE。** 非空片段时间不一定是逐 token 时间；usage 缺失时保留缺失。字符数不等于 token 数，不能据此计算准确 TPOT。报告字段应忠实于实际观测。

**练习十，并发。** 四条串行请求不能证明并发吞吐。concurrent_probe 限在途线程，但 client queue 不是服务队列，也不是开放到达。结合服务 batch 证据确认真正连续批处理。

**练习十一，超时。** read timeout 不等于总 deadline；线程池也会等待活动任务。外层 timeout 限本次探测进程，服务端取消传播仍需观察。退出码与日志应区分正常完成、窗口到期和实际错误。

**练习十二，GPU 计时。** Event 区间测设备工作，HTTP 墙钟测端到端；两者用途不同。reset peak 不释放内存，allocated 不等于全部设备占用。答案应说明常驻输入与参考也在峰值中。

**练习十三，TP 八。** 十四 query 头不能按常见均匀八分方式分配；不能因为模型小就跳过结构条件。换模型也必须重新核对 config、backend、KV 放置与每 rank 容量。

**练习十四，真实与模拟。** CPU 线程 fixture、dry-run 和 NumPy 数值对照不构成 H100 性能实测。实验档案分别写已执行、文档核查、未运行与 skip，未运行的结果表保持空白，不填“预期速度”。

项目按五阶段推进：环境指纹与 CPU 回归；模型和镜像准备；一次有界单卡服务及停止演练；长度、并发、缓存、图模式的单轴对照；满足准入条件后的两卡，再按课题需要使用八卡。每阶段都有原始日志、有效配置、结果与资源终态，失败也成为可复核材料。

掌握门槛是你能独立解释并修改启动命令，知道每个限制保护哪一层，能实际启动、验证、测量、停止一个服务，并能把结果与 CPU 模型联系起来而不混淆证据。下一章 [S16 综合工程](16-capstones.md) 将这些能力汇成可以持续扩展的推理系统项目。
