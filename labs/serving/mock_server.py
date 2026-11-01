"""Capstone B: local JSON front end over the reference engine.

Only /health, /metrics and a tiny /v1/completions subset are implemented.
Requests contain prompt_token_ids; this is NOT an OpenAI-compatible tokenizer
or production HTTP server. A lock deliberately serializes engine ownership.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import threading
import uuid
from .engine import Engine, Request


def validate_payload(body):
    if not isinstance(body,dict):raise ValueError('JSON object required')
    prompt=body.get('prompt_token_ids')
    if not isinstance(prompt,list) or not 1<=len(prompt)<=256:raise ValueError('1..256 prompt token IDs required')
    count=body.get('max_tokens',16)
    if type(count) is not int or not 0<=count<=64:raise ValueError('max_tokens must be 0..64')
    return Request(uuid.uuid4().hex,tuple(prompt),count,seed=body.get('seed',0))


def serve(port=18081):
    e=Engine(pages=256);lock=threading.Lock();counts={'accepted':0,'bad_request':0}
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,obj):
            data=json.dumps(obj,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':return self.send_json(200,{'status':'ok','engine':'course-simulator'})
            if self.path=='/metrics':
                with lock:payload={**counts,'active':len(e.running),'free_pages':len(e.pool.free)}
                return self.send_json(200,payload)
            self.send_json(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/v1/completions':return self.send_json(404,{'error':'not found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('invalid body size')
                body=json.loads(self.rfile.read(length));r=validate_payload(body)
                with lock:
                    r.arrival_ms=e.time_ms;e.submit(r);e.run();counts['accepted']+=1
                    # Trim terminal request history to keep long-lived demo bounded.
                    payload={'id':r.rid,'token_ids':r.output,'metrics':r.metrics(),'time_kind':'synthetic_ms'}
                    e.requests.pop(r.rid);e.events.clear()
                self.send_json(200,payload)
            except (ValueError,TypeError,KeyError) as exc:
                with lock:counts['bad_request']+=1
                self.send_json(400,{'error':str(exc)})
            except (BrokenPipeError,ConnectionResetError):
                pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Course mock server: http://127.0.0.1:{server.server_port}; Ctrl+C to stop',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=18081)
    args=p.parse_args();serve(args.port)
if __name__=='__main__':main()
