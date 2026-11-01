"""Bounded opt-in real endpoint probe. SSE chunks are NOT necessarily tokens."""
import argparse
import json
import os
import time
import urllib.request
import urllib.parse
from .streaming import SSEDecoder


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise ValueError('HTTP redirects are disabled; configure the final endpoint explicitly')


def load_prompt(path):
    """Read a bounded local UTF-8 fixture; never truncate a benchmark silently."""
    if path is None:
        return 'Count from one to five.'
    with open(path,'rb') as stream:
        raw=stream.read(65537)
    if not raw or len(raw)>65536:
        raise ValueError('prompt fixture must contain 1..65536 UTF-8 bytes')
    return raw.decode('utf-8')


def measure(base_url,model,key='',requests=4,max_tokens=32,prompt='Count from one to five.'):
    reports=[]
    opener=urllib.request.build_opener(_NoRedirect)
    for i in range(requests):
        payload={'model':model,'prompt':prompt,'max_tokens':max_tokens,'temperature':0,'stream':True}
        headers={'Content-Type':'application/json'}
        if key:headers['Authorization']='Bearer '+key
        req=urllib.request.Request(base_url.rstrip('/')+'/completions',data=json.dumps(payload).encode(),headers=headers)
        t0=time.perf_counter();first=None;times=[];characters=0;usage=None;decoder=SSEDecoder();done=False
        with opener.open(req,timeout=30) as response:
            while True:
                # read1 avoids waiting for a full large read buffer on many HTTP responses.
                chunk=response.read1(1024)
                if not chunk:break
                for event in decoder.feed(chunk):
                    if done:raise ValueError('data event after terminal marker')
                    if event=='[DONE]':done=True;continue
                    data=json.loads(event);now=time.perf_counter()
                    if not isinstance(data,dict) or 'error' in data:raise ValueError('invalid or error completion event')
                    usage=data.get('usage') or usage
                    choices=data.get('choices',[])
                    fragment=choices[0].get('text','') if choices else ''
                    if fragment:
                        first=now if first is None else first;times.append(now);characters+=len(fragment)
            decoder.feed(b'',final=True)
        if not done:raise ValueError('completion stream ended without terminal marker')
        elapsed=time.perf_counter()-t0
        gaps=[b-a for a,b in zip(times,times[1:])]
        reports.append({'request':i,'ttft_text_ms':None if first is None else (first-t0)*1000,
                        'e2e_ms':elapsed*1000,'nonempty_chunks':len(times),'output_characters':characters,
                        'mean_inter_chunk_ms':sum(gaps)/len(gaps)*1000 if gaps else None,
                        'output_tokens_from_usage':usage.get('completion_tokens') if usage else None})
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',default='http://127.0.0.1:18080/v1')
    p.add_argument('--model',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--allow-remote',action='store_true')
    p.add_argument('--requests',type=int,default=4);p.add_argument('--max-tokens',type=int,default=32);p.add_argument('--api-key-env',default='LLM_API_KEY')
    p.add_argument('--prompt-file',help='local UTF-8 public fixture, at most 65536 bytes')
    a=p.parse_args()
    if not 1<=a.requests<=16 or not 1<=a.max_tokens<=128:p.error('bounds: requests 1..16; max_tokens 1..128')
    parsed=urllib.parse.urlparse(a.base_url)
    if parsed.scheme not in {'http','https'} or parsed.username or parsed.password:p.error('plain HTTP(S) URL without embedded credentials required')
    if parsed.hostname not in {'localhost','127.0.0.1','::1'} and not a.allow_remote:p.error('remote endpoint requires --allow-remote')
    try:prompt=load_prompt(a.prompt_file)
    except (OSError,ValueError) as exc:p.error(str(exc))
    if not a.execute:
        print(json.dumps({'dry_run':True,'requests':a.requests,'max_output_tokens_total':a.requests*a.max_tokens,'prompt_utf8_bytes':len(prompt.encode('utf-8')),'instruction':'add --execute to send requests'}));return
    print(json.dumps(measure(a.base_url,a.model,os.environ.get(a.api_key_env,''),a.requests,a.max_tokens,prompt=prompt),indent=2))
if __name__=='__main__':main()
