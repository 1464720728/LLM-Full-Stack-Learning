# 本卷验证记录

核查日期：2026-11-01。环境：Linux、Python 3.12.14、NumPy 2.3.5。记录对应本卷发布前CPU验证；后续改动应重新运行对应检查，并更新本文件。

## 已执行

| 检查 | 结果 | 证据范围 |
|---|---|---|
| `python -m unittest discover -s tests -p 'test_serving*.py' -q` | 51 tests通过 | 元数据分页/调度、采样/SSE、24主题实验、数值分页KV、9项边界回归，以及3项新探针协议/并发回归 |
| manifest全部30条离线命令逐条独立执行 | 全部退出码0 | 24主题实验、源码pin展示、策略回放、3种状态跟踪、真实NumPy分页KV |
| 本地HTTP smoke，临时loopback端口 | 通过 | health、正确输出token、非法输入400、metrics计数与active释放 |
| Python compileall | 通过 | 本卷Python代码语法，包括可选上游pytest文件；不等于执行可选依赖代码 |
| `bash -n labs/serving/h100_server.sh` | 通过 | Shell语法；不等于Docker或GPU执行 |

本地HTTP smoke以`prompt_token_ids=[1,2,3]`、`max_tokens=4`得到`[4,5,6,7]`，同时检查返回时间标签为`synthetic_ms`。验证只使用localhost和临时子进程，没有调用外部API。

manifest的源码任务是`python -m labs.serving.source_walk --project vllm --show-pin`：仅打印本课程固定坐标，不需要上游clone。S09正文的clone、checkout、`--root`源码检查属于手动准备步骤，没有混入自动离线命令。可选上游pytest安装器也没有进入自动离线列表。

边界回归包含：负数/浮点/bool资源数量拒绝、负数前缀发布不改变缓存、report输出列表不暴露内部可变对象、retain/adopt/unpin失败原子性、无父节点fork不遗留子表、重复页adopt拒绝、已有运行历史请求不能重新提交、同一个Request对象不能进入两个Engine。

## 深写版新增验证

S01–S15 正文中 80 个完整 CPU 或本地 loopback Python 程序实际执行通过；源码方法摘录不作为独立程序成功计数。S17/S18/S19 工作坊分别有 20/19/24 个 Python 块按章共享 namespace 执行通过。S16 的 22 个 Python 块也按章顺序实际执行通过，正文完整程序总计165块；包括有界 OwnedWorker 并发/取消/关闭、前端桥接、完整数值模型路径与故障负控制。S15 的 CUDA Event 例仅语法检查，未运行设备路径。

新增 `collective_cpu.py` 实际 spawn 2 与 4 个 CPU 进程完成归约，非法输入前置拒绝；它不是 NCCL 性能实现。新增 `concurrent_probe.py` 以两线程 barrier 验证并发上限、ID 映射、部分失败保留与最终活动计数归零。两 HTTP 探针默认 dry-run；本地 SSE fixture 实际验证公开输入原样发送、中文、usage、缺终止/错误事件拒绝与 HTTP 重定向拒绝。输入文件超过64KiB、非法UTF8或空输入拒绝，不静默截断。

启动脚本完成 shell 语法及1/2/8卡参数 dry-run，拒绝重复设备、非规范编号和缺少多卡显式开关；`--pull=never` 强制只用本地镜像。dry-run 使用临时目录与明确虚构的镜像身份，不把它计为镜像、Docker或GPU验证。所有原有Engine/Backend/PagePool接口与资源边界回归保持通过。

独立审读修正包括：S13中文数字表达；S15默认urllib重定向绕过初始地址检查；S18缓存逐页满页不变量；S19末query全可见时K/V联合置换不变性导致仅看attention输出无法验证gather顺序。详见 [DEPTH_AUDIT.md](DEPTH_AUDIT.md)。

十九章统一深度检查全部通过，正文合计224,463去代码汉字。逐章数值与运行方式见 [DEPTH_AUDIT.md](DEPTH_AUDIT.md)。

## 尚未执行

- 固定vLLM checkout中的课程pytest：已对照源码helper和接口编写，完成语法检查；本环境没有安装并运行上游依赖。
- H100、RTX 5070 Ti、CUDA/Triton、真实模型GPU推理和多GPU通信。
- Docker GPU服务启动、镜像兼容性、真实模型权重下载与加载。
- DeepSeek/GPT等远端API、付费调用、真实服务吞吐压测。
- EvalScope/Harbor/Terminal-Bench端到端评测任务。

这些项目必须保持“未执行”直到有真实环境记录，不能用课程CPU输出替代。`Engine`时间为模拟值，分页池存token元数据；独立`paged_attention.py`确实存数值K/V，但仍是NumPy CPU参考实现，未证明GPU性能。

机器可读明细见[validation.json](validation.json)。源码证据和适用范围见[SOURCES.md](SOURCES.md)。
