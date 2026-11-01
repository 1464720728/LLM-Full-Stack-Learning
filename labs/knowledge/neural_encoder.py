"""A tiny learned dual encoder, trained from scratch on fictional pairs.

This is a mechanism demo, not a pretrained semantic model or a quality benchmark.
No downloads, API calls, torch, or GPU. NumPy only.
"""
import json
import numpy as np
from .text import tokenize

PAIRS=[
 ('feline pet','cat animal'),
 ('canine pet','dog animal'),
 ('key value reuse','cache token'),
 ('queued jobs','scheduler request'),
]


class TinyDualEncoder:
    def __init__(self,texts,dim=12,seed=7):
        if dim<=0:raise ValueError('positive dimension required')
        self.vocab={word:i for i,word in enumerate(sorted({t for s in texts for t in tokenize(s)}))}
        if not self.vocab:raise ValueError('nonempty vocabulary required')
        self.weights=np.random.default_rng(seed).normal(0,0.1,(len(self.vocab),dim))

    def features(self,texts):
        features=np.zeros((len(texts),len(self.vocab)),dtype=np.float64)
        for row,text in enumerate(texts):
            ids=[self.vocab[w] for w in tokenize(text) if w in self.vocab]
            for index in ids:features[row,index]+=1
            if ids:features[row]/=len(ids)
        return features

    def encode(self,texts):
        return self.features(texts)@self.weights

    def loss_and_gradient(self,queries,documents,temperature=0.2):
        if len(queries)!=len(documents) or not queries or temperature<=0:
            raise ValueError('aligned nonempty batches and positive temperature required')
        xq,xd=self.features(queries),self.features(documents)
        q,d=xq@self.weights,xd@self.weights
        scores=q@d.T/temperature
        shifted=scores-scores.max(axis=1,keepdims=True)
        exp=np.exp(shifted)
        probabilities=exp/exp.sum(axis=1,keepdims=True)
        # Stable log-sum-exp, avoiding log(0) after softmax underflow.
        loss=float(np.mean(np.log(exp.sum(axis=1))-np.diag(shifted)))
        grad_scores=(probabilities-np.eye(len(queries)))/len(queries)
        grad_q=grad_scores@d/temperature
        grad_d=grad_scores.T@q/temperature
        gradient=xq.T@grad_q+xd.T@grad_d
        return loss,gradient

    def train(self,queries,documents,steps=250,learning_rate=0.15):
        if steps<0 or learning_rate<=0:raise ValueError('invalid training parameters')
        losses=[]
        for _ in range(steps):
            loss,gradient=self.loss_and_gradient(queries,documents)
            losses.append(loss)
            norm=np.linalg.norm(gradient)
            if norm>5:gradient*=5/norm
            self.weights-=learning_rate*gradient
        losses.append(self.loss_and_gradient(queries,documents)[0])
        return losses

    def search(self,query,documents,k=3):
        if k<0:raise ValueError('nonnegative k required')
        if not any(t in self.vocab for t in tokenize(query)):
            return []
        # Same raw dot-product metric used by the training objective.
        scores=(self.encode([query])@self.encode(documents).T)[0]
        order=sorted(range(len(documents)),key=lambda i:(-float(scores[i]),i))[:k]
        return [(i,float(scores[i])) for i in order]


def demo():
    queries=[q for q,d in PAIRS];documents=[d for q,d in PAIRS]
    model=TinyDualEncoder(queries+documents)
    before=model.search('feline pet',documents,4)
    losses=model.train(queries,documents)
    after=model.search('feline pet',documents,4)
    return {'training_pairs':PAIRS,'initial_loss':losses[0],'final_loss':losses[-1],
            'before':before,'after':after,'unknown_query':model.search('spaceship',documents),
            'limitation':'Tiny fitted vocabulary and training-set demonstration; no general semantic capability claim.'}


if __name__=='__main__':print(json.dumps(demo(),ensure_ascii=False,indent=2))
