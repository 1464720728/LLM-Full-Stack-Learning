"""Step 8: route simulated requests without creating a network listener."""
import json
from ..task_service import TaskStore, dispatch

def requests_demo():
    store = TaskStore()
    payload = json.dumps({'title':'Understand routing','idempotency_key':'one'}).encode()
    requests = [('GET','/health',b''), ('POST','/tasks',payload), ('POST','/tasks',payload),
                ('POST','/tasks/1/complete',b''), ('GET','/tasks',b''), ('GET','/missing',b'')]
    try:
        return [(method,path,*dispatch(store,method,path,body)) for method,path,body in requests]
    finally:
        store.close()

if __name__ == '__main__':
    for method,path,status,payload in requests_demo():
        print(method, path, '->', status, json.dumps(payload,ensure_ascii=False))
