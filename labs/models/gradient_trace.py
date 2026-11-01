"""A complete one-example MLP LM forward/backward with repeated embedding ids."""
import json
import numpy as np
from .tiny_lm import TinyLanguageModel
from .core import softmax

def trace():
    m=TinyLanguageModel(2,context=2,embedding=1,hidden=1)
    m.params={'E':np.array([[-.5],[.5]]),'W1':np.array([[1.],[2.]]),'b1':np.zeros(1),'W2':np.array([[1.,-1.]]),'b2':np.zeros(2)}
    ids=np.array([[1,1]]); target=np.array([1]); logits,(_,emb,h)=m.forward(ids)
    loss,grads=m.loss_and_grads(ids,target)
    return {'input_ids':ids.tolist(),'embedding_flat':emb.tolist(),'hidden_pre_activation':(emb@m.params['W1']+m.params['b1']).tolist(),'hidden':h.tolist(),'logits':logits.tolist(),'probability':softmax(logits).tolist(),'loss':loss,'grads':{name:value.tolist() for name,value in grads.items()}}

if __name__=='__main__': print(json.dumps(trace(),indent=2))
