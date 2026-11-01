"""Inspect local upstream checkouts; never clones, installs or changes them."""
import argparse
import json
from pathlib import Path
import re
import subprocess

SNAPSHOTS={
 'vllm':{'commit':'e0d00e4b8eb0c89ec30921d0695cfdd409e9c211','files':{
  'vllm/v1/engine/core.py':['EngineCore','step'],
  'vllm/v1/core/sched/scheduler.py':['Scheduler','schedule','update_from_output'],
  'vllm/v1/core/kv_cache_manager.py':['KVCacheManager','get_computed_blocks','allocate_slots','free'],
  'vllm/v1/worker/gpu_model_runner.py':['GPUModelRunner','execute_model','sample_tokens']}},
 'sglang':{'commit':'fa5580f6d0e969a3d44c603d215e2139886e8db2','files':{
  'python/sglang/srt/managers/scheduler.py':['Scheduler','event_loop_normal','event_loop_overlap','get_next_batch_to_run','run_batch','process_batch_result'],
  'python/sglang/srt/mem_cache/radix_cache.py':['RadixCache','match_prefix','insert','evict'],
  'python/sglang/srt/model_executor/model_runner.py':['ModelRunner','forward']}}}


def inspect(root,project):
    root=Path(root).resolve();config=SNAPSHOTS[project]
    head=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],check=True,text=True,capture_output=True).stdout.strip()
    result={'project':project,'expected_commit':config['commit'],'actual_commit':head,'exact_snapshot':head==config['commit'],'symbols':[]}
    for rel,names in config['files'].items():
        path=root/rel
        if not path.is_file():result['symbols'].append({'path':rel,'error':'missing'});continue
        for line,text in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            m=re.match(r'\s*(?:async\s+)?(?:class|def)\s+(\w+)',text)
            if m and m.group(1) in names:result['symbols'].append({'path':rel,'line':line,'symbol':m.group(1)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',choices=SNAPSHOTS,required=True)
    p.add_argument('--root',type=Path);p.add_argument('--show-pin',action='store_true');args=p.parse_args()
    if args.show_pin:print(json.dumps(SNAPSHOTS[args.project],indent=2));return
    if not args.root:p.error('--root required unless --show-pin')
    print(json.dumps(inspect(args.root,args.project),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
