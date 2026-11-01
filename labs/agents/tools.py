"""Small capability-shaped tools. A path guard is not an OS security sandbox."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from .contracts import Tool, object_schema, ContractError

class Workspace:
    def __init__(self,root,max_bytes=32768):
        self.root=Path(root).resolve(strict=True)
        self.max_bytes=max_bytes
    def resolve(self,relative):
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts:raise ContractError('relative paths without .. required')
        lexical=self.root/p
        # Refuse symlink components; no protection against hostile concurrent renaming.
        current=self.root
        for part in p.parts:
            current=current/part
            if current.is_symlink():raise ContractError('symlinks are not supported')
        resolved=lexical.resolve()
        if not resolved.is_relative_to(self.root):raise ContractError('path escapes workspace')
        return resolved
    def read(self,relative):
        path=self.resolve(relative)
        with path.open('rb') as f:data=f.read(self.max_bytes+1)
        if len(data)>self.max_bytes:raise ContractError('file too large')
        return {'path':relative,'text':data.decode('utf-8'),'sha256':hashlib.sha256(data).hexdigest()}
    def replace(self,relative,old,new,expected_sha256):
        path=self.resolve(relative)
        before=self.read(relative)
        if before['sha256']!=expected_sha256:raise ContractError('file changed since read')
        if not old or before['text'].count(old)!=1:raise ContractError('old text must match exactly once')
        after=before['text'].replace(old,new,1).encode()
        if len(after)>self.max_bytes:raise ContractError('replacement too large')
        # Atomic visibility of file replacement, not an atomic compare-and-swap against a concurrent writer.
        fd,name=tempfile.mkstemp(dir=path.parent,prefix='.course-edit-')
        try:
            with os.fdopen(fd,'wb') as f:f.write(after)
            os.chmod(name,path.stat().st_mode)
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)
        return {'sha256':hashlib.sha256(after).hexdigest(),'bytes':len(after)}
    def tools(self):
        s={'type':'string','minLength':1,'maxLength':4096}
        return [Tool('read_file','Read one small UTF-8 file beneath the workspace.',object_schema({'path':s}),lambda a:self.read(a['path'])),
                Tool('replace_text','Replace one exact occurrence after checking the read hash.',object_schema({'path':s,'old':s,'new':{'type':'string','maxLength':32768},'expected_sha256':{'type':'string','minLength':64,'maxLength':64}}),lambda a:self.replace(a['path'],a['old'],a['new'],a['expected_sha256']),side_effect=True)]


def arithmetic_tool():
    return Tool('add','Add two bounded integers.',object_schema({'a':{'type':'integer','minimum':-1000000,'maximum':1000000},'b':{'type':'integer','minimum':-1000000,'maximum':1000000}}),lambda a:a['a']+a['b'])

class TicketBook:
    """An idempotent *business operation*: request_id is unique in the same DB tx."""
    def __init__(self,path=':memory:'):
        self.db=sqlite3.connect(str(path))
        self.db.execute('CREATE TABLE IF NOT EXISTS tickets(request_id TEXT PRIMARY KEY,title TEXT NOT NULL,status TEXT NOT NULL)')
        self.db.commit()
    def create(self,request_id,title):
        with self.db:
            row=self.db.execute('SELECT title,status FROM tickets WHERE request_id=?',(request_id,)).fetchone()
            if row:
                if row[0]!=title:raise ContractError('idempotency key conflicts with earlier title')
                return {'request_id':request_id,'title':row[0],'status':row[1],'created':False}
            self.db.execute('INSERT INTO tickets VALUES(?,?,?)',(request_id,title,'open'))
        return {'request_id':request_id,'title':title,'status':'open','created':True}
    def tool(self):
        return Tool('create_ticket','Create one ticket; request_id is a stable business key.',object_schema({'request_id':{'type':'string','minLength':1,'maxLength':64},'title':{'type':'string','minLength':1,'maxLength':200}}),lambda a:self.create(**a),side_effect=True)
    def close(self):self.db.close()


def run_trusted_fixture_tests(root,timeout=5):
    """Only for the course's trusted fixture. This executes Python, not a sandbox.
    -I removes user site and env-derived imports, but gives code normal OS access.
    """
    root=Path(root).resolve(strict=True)
    with tempfile.TemporaryFile(mode='w+b') as output:
        try:
            completed=subprocess.run([sys.executable,'-I','-B','-m','unittest','discover','-s',str(root),'-p','test_*.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,timeout=timeout,check=False)
            code=completed.returncode
        except subprocess.TimeoutExpired:
            code=124
        output.seek(0);data=output.read(8193)
    return {'exit_code':code,'output':data[:8192].decode('utf-8',errors='replace'),'truncated':len(data)>8192}
