"""Executable abstractions for studying recent structural ideas.

These intentionally small functions are NOT checkpoint-compatible replicas of
CSA2, Engram, KDA, CED, mHC, GRPO, or multimodal flagship systems.
"""
import numpy as np
from .core import attention,causal_mask,attention_residual,softmax


def block_depth_aggregate(embedding,updates,query,block_size=2):
    """Block depth-attention illustration with snapshots of additive block output.

    At each sublayer, completed source blocks and current partial block are
    candidates. updates are externally supplied vectors for inspectable tracing.
    """
    if block_size<1: raise ValueError('positive block size required')
    completed=[np.asarray(embedding)]; running=np.zeros_like(embedding,dtype=float); traces=[]
    for i,update in enumerate(updates):
        candidates=completed+([running.copy()] if i%block_size else [])
        mixed,weights=attention_residual(np.stack(candidates),query)
        running=running+update
        traces.append({'input':mixed,'weights':weights,'output':mixed+update})
        if (i+1)%block_size==0:
            completed.append(running.copy()); running=np.zeros_like(embedding,dtype=float)
    return traces


def compressed_sparse_attention(q,k,v,window=2,compression=2,top_k=2):
    """Causal nonoverlapping mean compression + top-k compressed-block retrieval.

    A block is eligible only if its LAST position lies before the local window.
    This prevents both future leakage and overlapping local/compressed records.
    Means, dot-product indexer, and value definitions differ from real CSA.
    """
    if min(window,compression,top_k)<1: raise ValueError('positive sizes required')
    out=[]; chosen=[]
    for i in range(len(q)):
        local_start=max(0,i-window+1)
        eligible=[]
        for start in range(0,local_start,compression):
            end=start+compression
            if end<=local_start:
                eligible.append((start,k[start:end].mean(0),v[start:end].mean(0)))
        if eligible:
            scores=np.array([q[i]@record[1] for record in eligible])
            indices=np.argsort(-scores,kind='stable')[:top_k]
            selected=[eligible[j] for j in indices]
        else: selected=[]
        kk=np.concatenate([np.array([r[1] for r in selected]).reshape(-1,k.shape[1]),k[local_start:i+1]])
        vv=np.concatenate([np.array([r[2] for r in selected]).reshape(-1,v.shape[1]),v[local_start:i+1]])
        out.append(attention(q[i:i+1],kk,vv)[0][0]); chosen.append([r[0] for r in selected])
    return np.asarray(out),chosen


def causal_encoder_decoder(x,wq,wk,wv):
    """Minimal causal encoder states become decoder cross-attention K/V.

    Both stages are causal; no normalizations/FFNs/production cache sharing.
    """
    mask=causal_mask(len(x))
    encoded=attention(x@wq,x@wk,x@wv,mask)[0]
    decoded=attention(x@wq,encoded@wk,encoded@wv,mask)[0]
    return decoded,encoded


def conditional_ngram_memory(ids,table,n=2):
    """Fixed integer n-gram hash lookup; no training, collision resolution or gate.

    This only teaches token-conditioned lookup vs attention, not Engram itself.
    """
    if n<1 or len(table)==0: raise ValueError('invalid memory shape')
    out=[]; slots=[]
    for i in range(len(ids)):
        key=0
        for token in ids[max(0,i-n+1):i+1]: key=(key*257+int(token)+1)%len(table)
        slots.append(key); out.append(table[key])
    return np.asarray(out),slots


def group_advantages(rewards,eps=1e-8):
    """Population-standardized scalar group rewards, one common toy convention."""
    rewards=np.asarray(rewards,dtype=float)
    if rewards.ndim!=1 or len(rewards)<2 or not np.isfinite(rewards).all():
        raise ValueError('finite group with at least two outcomes required')
    return (rewards-rewards.mean())/(rewards.std()+eps)


def clipped_policy_objective(logprob,old_logprob,advantages,clip=.2):
    """Mean clipped scalar importance surrogate; NOT complete GRPO training.

    Returns minimized negative objective and derivative w.r.t. logprob. Omits
    token masking, KL, reward model, rollouts, distributed group accounting.
    """
    lp,old,a=map(lambda x:np.asarray(x,dtype=float),(logprob,old_logprob,advantages))
    if lp.shape!=old.shape or lp.shape!=a.shape or lp.size==0 or not 0<clip<1:
        raise ValueError('matching nonempty arrays and clip in (0,1) required')
    ratio=np.exp(lp-old); clipped=np.clip(ratio,1-clip,1+clip)
    raw=ratio*a; bounded=clipped*a
    use_raw=raw<=bounded
    loss=-float(np.mean(np.minimum(raw,bounded)))
    gradient=np.where(use_raw,-ratio*a/lp.size,0.)
    return loss,gradient


def cross_modal_read(text_queries,image_patches,projection):
    """Image patches -> visual token projection -> text-to-vision cross attention."""
    visual=image_patches@projection
    return attention(text_queries,visual,visual)[0]
