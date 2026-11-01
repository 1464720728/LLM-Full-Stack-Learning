"""Hand-written trainable causal Transformer. Optional PyTorch, CPU by default.

No pretrained weights or datasets are downloaded. CUDA is explicit and bounded.
"""
import argparse
import importlib.util
import json
import math
from .tiny_lm import CORPUS


def build_model(torch,vocab,dim=32,heads=4,layers=2,context=24):
    nn=torch.nn
    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.qkv=nn.Linear(dim,3*dim,bias=False); self.output=nn.Linear(dim,dim,bias=False)
            self.register_buffer('mask',torch.tril(torch.ones(context,context,dtype=torch.bool)))
        def forward(self,x):
            b,t,d=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
            def heads_view(z): return z.view(b,t,heads,d//heads).transpose(1,2)
            q,k,v=map(heads_view,(q,k,v)); scores=(q@k.transpose(-1,-2))/math.sqrt(d//heads)
            scores=scores.masked_fill(~self.mask[:t,:t],float('-inf'))
            out=torch.softmax(scores,dim=-1)@v
            return self.output(out.transpose(1,2).contiguous().view(b,t,d))
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim); self.attn=Attention()
            self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
        def forward(self,x):
            x=x+self.attn(self.n1(x)); return x+self.ff(self.n2(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.token=nn.Embedding(vocab,dim); self.position=nn.Embedding(context,dim)
            self.blocks=nn.ModuleList([Block() for _ in range(layers)]); self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,vocab)
        def forward(self,ids):
            x=self.token(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
            for block in self.blocks: x=block(x)
            return self.head(self.norm(x))
    return Model()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--steps',type=int,default=40)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--require-torch',action='store_true')
    args=p.parse_args()
    if not 1<=args.steps<=1000: p.error('steps must be 1..1000')
    if importlib.util.find_spec('torch') is None:
        if args.require_torch: p.error('PyTorch is not installed; follow environment instructions')
        print('SKIP: optional PyTorch is not installed. NumPy core labs remain available.'); return
    import torch
    torch.manual_seed(13); torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA explicitly requested but unavailable')
    alphabet=''.join(sorted(set(CORPUS))); stoi={s:i for i,s in enumerate(alphabet)}
    data=torch.tensor([stoi[c] for c in CORPUS],dtype=torch.long,device=args.device)
    boundary=int(len(data)*.8); train_data=data[:boundary]; valid_data=data[boundary:]
    context=24; model=build_model(torch,len(alphabet),context=context).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.003); generator=torch.Generator().manual_seed(13)
    def batch(stream,n=8):
        starts=torch.randint(len(stream)-context,(n,),generator=generator)
        x=torch.stack([stream[int(i):int(i)+context] for i in starts]); y=torch.stack([stream[int(i)+1:int(i)+context+1] for i in starts]); return x,y
    def objective(x,y): return torch.nn.functional.cross_entropy(model(x).reshape(-1,len(alphabet)),y.reshape(-1))
    fixed_x,fixed_y=batch(valid_data)
    model.eval()
    with torch.no_grad(): initial=float(objective(fixed_x,fixed_y).cpu())
    model.train()
    for _ in range(args.steps):
        x,y=batch(train_data); loss=objective(x,y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    model.eval()
    with torch.no_grad():
        final=float(objective(fixed_x,fixed_y).cpu())
        ids=torch.tensor([[stoi[c] for c in 'the']],device=args.device)
        for _ in range(64):
            logits=model(ids[:,-context:])[:,-1,:]; next_id=torch.argmax(logits,dim=-1,keepdim=True); ids=torch.cat([ids,next_id],dim=1)
    print(json.dumps({'device':args.device,'steps':args.steps,'initial_validation_ce':initial,'final_validation_ce':final,'parameters':sum(v.numel() for v in model.parameters()),'note':'synthetic repeated grammar, not a capability benchmark'},indent=2))
    print(''.join(alphabet[i] for i in ids[0].tolist()))

if __name__=='__main__': main()
