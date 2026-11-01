#!/usr/bin/env python3
"""Reproduce CPU release checks and write a factual report. No GPU/API calls.

Reads the repository's own manifests and runs only registered cpu/offline labs.
Run from a trusted checkout. This is not a sandbox for third-party repositories.
"""
from __future__ import annotations
import datetime as dt
import io
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.course import ROOT, catalog, validate, prepare_command
from tools.check_docs import check
from tools.build_catalog import build


def verify():
    # CPU validation must not accidentally expose a GPU to an optional test.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report_path=ROOT/"docs"/"VALIDATION.md"
    if not report_path.exists():
        report_path.write_text("# 验证报告\n\n正在生成。\n",encoding="utf-8")
    stats=build()
    structural=validate()+check()
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    _,chapters=catalog()
    outcomes=[]
    seen=set()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
    for chapter in chapters:
        for lab in chapter["labs"]:
            if lab["command"] in seen:
                continue
            seen.add(lab["command"])
            entry={"id":lab["id"],"command":lab["command"]}
            if lab.get("network",False) or lab.get("hardware","cpu").lower() not in ("cpu","none"):
                entry.update(status="not_executed",reason="requires network/API or GPU; no such execution claimed")
            else:
                try:
                    completed=subprocess.run(prepare_command(lab,False,False),cwd=ROOT,env=env,
                                             text=True,capture_output=True,timeout=60)
                    skipped=completed.returncode == 0 and re.search(r"^SKIP:",completed.stdout,flags=re.M)
                    entry.update(status="not_executed" if skipped else ("passed" if completed.returncode==0 else "failed"),exit_code=completed.returncode)
                    if skipped:
                        entry["reason"]="optional dependency unavailable; program reported SKIP (not a completed validation)"
                    if completed.returncode:
                        entry["error"]=(completed.stderr+completed.stdout)[-2000:]
                except (ValueError,OSError,subprocess.TimeoutExpired) as exc:
                    entry.update(status="failed",error=str(exc))
            outcomes.append(entry)
            print(entry["id"],entry["status"],flush=True)
    import numpy as np
    optional={name:importlib.util.find_spec(name) is not None for name in ("torch","triton","faiss")}
    data={"checked_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
          "environment":{"python":platform.python_version(),"numpy":np.__version__,"os":platform.system(),"architecture":platform.machine(),"optional_dependencies":optional,"gpu_visibility":"disabled for this CPU validation"},
          "statistics":stats,"structural_errors":structural,
          "tests":{"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"successful":result.wasSuccessful()},
          "lab_commands":outcomes,
          "not_claimed":["Windows-native execution","WSL2 execution","RTX 5070 Ti execution","H100/CUDA/Triton execution","multi-GPU NCCL execution","real model API calls","remote GitHub Actions execution"]}
    (ROOT/"docs"/"validation.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    local=ROOT/".course";local.mkdir(exist_ok=True)
    (local/"test-output.txt").write_text(log.getvalue(),encoding="utf-8")
    counts={kind:sum(x["status"]==kind for x in outcomes) for kind in ("passed","failed","not_executed")}
    lines=["# 第一版验证报告","",f"验证时间（UTC）：{data['checked_at_utc']}。课程版本见 VERSION。",
           "","## 实际执行环境","",f"Python {platform.python_version()}，NumPy {np.__version__}，{platform.system()} {platform.machine()}。本次验证进程隐藏 CUDA 设备，只执行 CPU 路径。",
           "", "可选依赖安装状态：" + "，".join(f"{name}={'已安装' if installed else '未安装'}" for name,installed in optional.items()) + "。",
           "","## 实际结果","",f"- 自动测试：{result.testsRun} 项，失败 {len(result.failures)}，错误 {len(result.errors)}，跳过 {len(result.skipped)}。",
           f"- 注册实验去重命令：CPU／离线通过 {counts['passed']} 条，失败 {counts['failed']} 条，依赖 GPU／联网或缺少可选依赖而未执行 {counts['not_executed']} 条。",
           f"- 课程路径／先修与本地 Markdown 链接／Python 语法问题：{len(structural)} 项。",
           f"- 教材规模：{stats['tracks']} 卷，{stats['chapters']} 个有稳定 ID 的单元，{stats['registered_labs']} 个注册实验入口。",
           "","测试覆盖机制包括状态机与恢复、流式分片、权限与文档版本、引用失效、模型等价性、梯度与数值稳定、分页与 COW、取消与抢占、布局与在线归一化，以及跨卷集成。测试数量不等于覆盖了全部生产故障。",
           "","## 没有宣称执行的部分","","Win11 原生／WSL2、本地 RTX 5070 Ti、H100、CUDA／Triton、NCCL 多卡以及真实模型 API 均未在本次构建环境执行。对应教学实现与操作步骤已提供；Python 语法检查不替代 GPU 编译或运行。真实硬件的误差与性能需要在指定环境按章节验收。",
           "","GitHub Actions 配置存在不代表其远端运行成功；远端状态以仓库 Actions 页面为准。本报告是本地执行证据。",
           "","## 复现","","```bash","python -m pip install -r requirements-cpu.txt","python tools/verify_release.py","```","", "此命令会更新本报告与 validation.json，并执行仓库内声明为离线 CPU 的实验。不会自动运行 GPU／联网实验。部分 unittest 内部还验证多个输入或多个微实验，数字按 unittest 测试方法计。",
           "","## 注册实验执行明细","","| 实验 | 状态 | 命令 |","|---|---|---|"]
    for entry in outcomes:
        lines.append(f"| {entry['id']} | {entry['status']} | `{entry['command']}` |")
    if structural:
        lines += ["","## 尚未修复的结构问题",""]+[f"- {error}" for error in structural]
    report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return data


if __name__=="__main__":
    result=verify()
    failed=bool(result["structural_errors"]) or not result["tests"]["successful"] or any(x["status"]=="failed" for x in result["lab_commands"])
    print(json.dumps({"tests":result["tests"],"structural_errors":result["structural_errors"]},ensure_ascii=False,indent=2))
    raise SystemExit(failed)
