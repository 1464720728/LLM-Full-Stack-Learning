"""Train a fully differentiable character context-MLP language model on CPU.

This is an n-gram-conditioned neural LM, not a Transformer. Its small size makes
all gradients inspectable. torch_transformer.py supplies the Transformer project.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .core import cross_entropy, sample_token

CORPUS = ('the cat sits. the dog runs. the cat runs. the dog sits.\n' * 30)

class TinyLanguageModel:
    def __init__(self,vocab,context=3,embedding=8,hidden=24,seed=11):
        if min(vocab,context,embedding,hidden)<1: raise ValueError('positive dimensions required')
        self.vocab=vocab; self.context=context; self.embedding=embedding; self.hidden=hidden
        rng=np.random.default_rng(seed)
        self.params={
            'E':rng.normal(0,.1,(vocab,embedding)),
            'W1':rng.normal(0,1/math.sqrt(context*embedding),(context*embedding,hidden)),
            'b1':np.zeros(hidden),
            'W2':rng.normal(0,1/math.sqrt(hidden),(hidden,vocab)),
            'b2':np.zeros(vocab),
        }
    def forward(self,x):
        x=np.asarray(x,dtype=int); p=self.params
        if x.ndim!=2 or x.shape[1]!=self.context or np.any(x<0) or np.any(x>=self.vocab):
            raise ValueError('inputs must be (N,context) valid ids')
        emb=p['E'][x].reshape(len(x),-1)
        h=np.tanh(emb@p['W1']+p['b1'])
        return h@p['W2']+p['b2'],(x,emb,h)
    def loss_and_grads(self,x,y):
        logits,(ids,emb,h)=self.forward(x); loss,g=cross_entropy(logits,y); p=self.params
        grads={'W2':h.T@g,'b2':g.sum(0)}
        gh=(g@p['W2'].T)*(1-h*h)
        grads.update(W1=emb.T@gh,b1=gh.sum(0))
        ge=(gh@p['W1'].T).reshape(len(ids),self.context,self.embedding)
        grads['E']=np.zeros_like(p['E'])
        # Repeated ids must accumulate. grads['E'][ids] += ge is not equivalent.
        np.add.at(grads['E'],ids,ge)
        return loss,grads
    def update(self,grads,lr=.3,clip_norm=5.):
        norm=math.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        scale=min(1.,clip_norm/max(norm,1e-12))
        for name in self.params: self.params[name]-=lr*scale*grads[name]
        return norm
    def save(self,path,alphabet):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**self.params,config=np.array([self.vocab,self.context,self.embedding,self.hidden]),alphabet=np.array(alphabet))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=cls(*[int(v) for v in data['config']])
            for name in m.params:
                value=data[name]
                if value.shape!=m.params[name].shape: raise ValueError('checkpoint shape mismatch')
                m.params[name]=value.copy()
            return m,str(data['alphabet'])
    def generate(self,prompt,alphabet,length=80,seed=19):
        stoi={c:i for i,c in enumerate(alphabet)}; ids=[stoi[c] for c in prompt]
        if len(ids)<self.context: raise ValueError('prompt shorter than context')
        rng=np.random.default_rng(seed)
        for _ in range(length):
            logits,_=self.forward(np.array([ids[-self.context:]]))
            ids.append(sample_token(logits[0],rng,temperature=.8,top_p=.95))
        return ''.join(alphabet[i] for i in ids)


def dataset(text,alphabet,context):
    stoi={c:i for i,c in enumerate(alphabet)}; ids=np.array([stoi[c] for c in text])
    if len(ids)<=context: raise ValueError('text too short')
    return np.array([ids[i:i+context] for i in range(len(ids)-context)]),ids[context:]


def train(steps=160,seed=11):
    alphabet=''.join(sorted(set(CORPUS))); m=TinyLanguageModel(len(alphabet),seed=seed)
    # Split the continuous stream BEFORE windows, avoiding overlapping windows
    # across the split boundary. Repeated artificial grammar is still shared.
    boundary=int(len(CORPUS)*.8)
    x,y=dataset(CORPUS[:boundary],alphabet,m.context)
    vx,vy=dataset(CORPUS[boundary:],alphabet,m.context)
    initial=m.loss_and_grads(x,y)[0]; rng=np.random.default_rng(seed)
    for _ in range(steps):
        idx=rng.integers(0,len(x),size=64)
        _,g=m.loss_and_grads(x[idx],y[idx]); m.update(g)
    final=m.loss_and_grads(x,y)[0]; validation=m.loss_and_grads(vx,vy)[0]
    return m,alphabet,{'initial_train_ce':initial,'final_train_ce':final,'validation_ce':validation,'steps':steps,'units':'nats/token','data':'synthetic repeated grammar; NOT natural-language benchmark'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=160); p.add_argument('--save',type=Path)
    args=p.parse_args()
    if not 1<=args.steps<=10000: p.error('--steps must be 1..10000')
    m,a,metrics=train(args.steps)
    print(json.dumps(metrics,ensure_ascii=False,indent=2)); print(m.generate('the',a))
    if args.save: m.save(args.save,a); print(f'saved {args.save}')

if __name__=='__main__': main()
