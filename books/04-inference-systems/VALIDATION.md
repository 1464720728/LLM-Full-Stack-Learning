# 本卷验证记录

核查日期：2026-11-01。环境：Linux、Python 3.12.14、NumPy 2.3.5。记录对应本卷发布前CPU验证；后续改动应重新运行对应检查，并更新本文件。

## 已执行

| 检查 | 结果 | 证据范围 |
|---|---|---|
| `python -m unittest discover -s tests -p 'test_serving*.py' -q` | 48 tests通过 | 元数据分页/调度、采样/SSE、24主题实验、数值分页KV、9项边界回归 |
| manifest全部30条离线命令逐条独立执行 | 全部退出码0 | 24主题实验、源码pin展示、策略回放、3种状态跟踪、真实NumPy分页KV |
| 本地HTTP smoke，临时loopback端口 | 通过 | health、正确输出token、非法输入400、metrics计数与active释放 |
| Python compileall | 通过 | 本卷Python代码语法，包括可选上游pytest文件；不等于执行可选依赖代码 |
| `bash -n labs/serving/h100_server.sh` | 通过 | Shell语法；不等于Docker或GPU执行 |

本地HTTP smoke以`prompt_token_ids=[1,2,3]`、`max_tokens=4`得到`[4,5,6,7]`，同时检查返回时间标签为`synthetic_ms`。验证只使用localhost和临时子进程，没有调用外部API。

manifest的源码任务是`python -m labs.serving.source_walk --project vllm --show-pin`：仅打印本课程固定坐标，不需要上游clone。S09正文的clone、checkout、`--root`源码检查属于手动准备步骤，没有混入自动离线命令。可选上游pytest安装器也没有进入自动离线列表。

边界回归包含：负数/浮点/bool资源数量拒绝、负数前缀发布不改变缓存、report输出列表不暴露内部可变对象、retain/adopt/unpin失败原子性、无父节点fork不遗留子表、重复页adopt拒绝、已有运行历史请求不能重新提交、同一个Request对象不能进入两个Engine。

## 尚未执行

- 固定vLLM checkout中的课程pytest：已对照源码helper和接口编写，完成语法检查；本环境没有安装并运行上游依赖。
- H100、RTX 5070 Ti、CUDA/Triton、真实模型GPU推理和多GPU通信。
- Docker GPU服务启动、镜像兼容性、真实模型权重下载与加载。
- DeepSeek/GPT等远端API、付费调用、真实服务吞吐压测。
- EvalScope/Harbor/Terminal-Bench端到端评测任务。

这些项目必须保持“未执行”直到有真实环境记录，不能用课程CPU输出替代。`Engine`时间为模拟值，分页池存token元数据；独立`paged_attention.py`确实存数值K/V，但仍是NumPy CPU参考实现，未证明GPU性能。

机器可读明细见[validation.json](validation.json)。源码证据和适用范围见[SOURCES.md](SOURCES.md)。
