"""Read-only local tool availability; optional GPU query is explicit."""
import argparse
import importlib.util
import json
import platform
import shutil

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    out={'python':platform.python_version(),'system':platform.system(),'packages':{x:importlib.util.find_spec(x) is not None for x in ['numpy','torch','triton']},'commands':{x:shutil.which(x) is not None for x in ['g++','nvcc','nvidia-smi']},'gpu_query':'not requested'}
    if a.gpu:
        if not out['packages']['torch']:out['gpu_query']='SKIP: torch unavailable'
        else:
            import torch
            out['torch_version']=torch.__version__;out['torch_cuda_build']=torch.version.cuda
            out['gpu_query']=[{'visible_index':i,'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i),'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
