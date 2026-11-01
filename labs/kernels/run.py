"""28 bounded CPU experiments: python -m labs.kernels.run --lab 01 or --all."""
import argparse
import json
import numpy as np
from . import reference as r


def experiment(number):
    rng=np.random.default_rng(20261101)
    if number==1:
        return {'strides':r.c_strides((2,3,4)), 'offset_1_2_3':r.offset((1,2,3),(2,3,4),(12,4,1))}
    if number==2:
        a=np.arange(12).reshape(3,4); t=a.T
        return {'shape':t.shape,'strides_bytes':t.strides,'view_value':int(t[2,1]),'shares_memory':bool(np.shares_memory(a,t))}
    if number==3:
        ids=r.launch_indices(19,4,2);flat=sum(ids,[])
        assert sorted(flat)==list(range(19))
        return {'thread_work':ids,'exactly_once':len(set(flat))==19}
    if number==4:
        return {'contiguous_sectors':r.sectors(np.arange(32)*4),'stride8_sectors':r.sectors(np.arange(32)*32),'misaligned_sectors':r.sectors(4+np.arange(32)*4)}
    if number==5:
        return {'column_stride32':r.bank_conflicts(np.arange(32)*32),'padded_stride33':r.bank_conflicts(np.arange(32)*33),'broadcast':r.bank_conflicts([0]*32)}
    if number==6:
        x=np.array([1e8,1,-1e8,1],np.float32)
        sequential=np.float32(0)
        for v in x:sequential=np.float32(sequential+v)
        return {'sequential':float(sequential),'tree':float(r.tree_sum(x)),'float64':float(x.astype(np.float64).sum())}
    if number==7:
        return {'softmax':r.stable_softmax([1000,1001,1002]).tolist(),'shift_invariance':bool(np.allclose(r.stable_softmax([0,1,2]),r.stable_softmax([1000,1001,1002])))}
    if number==8:
        return {'masked':r.stable_softmax([[2,3,4],[2,3,4]],[[True,False,True],[False,False,False]]).tolist()}
    if number==9:
        x=np.array([[3.,4.]])
        return {'rmsnorm':r.rmsnorm(x,np.ones(2)).tolist(),'layernorm':r.layernorm(x).tolist()}
    if number==10:
        x=rng.normal(size=(2,3));w=rng.normal(size=3);g=rng.normal(size=x.shape)
        dx,dw=r.rmsnorm_backward(x,w,g)
        err=np.max(np.abs(dx-r.finite_difference(lambda z:float((r.rmsnorm(z,w)*g).sum()),x)))
        return {'dx_max_error':float(err),'dw':dw.tolist()}
    if number==11:
        a=np.arange(15).reshape(3,5);b=np.arange(10).reshape(5,2)
        return {'tiled':r.tiled_matmul(a,b,2).tolist(),'equal':bool(np.array_equal(r.tiled_matmul(a,b,2),a@b))}
    if number==12:
        return {'tile_8_flop_per_byte_fp32':2*8**3/(2*8**2*4),'tile_32_flop_per_byte_fp32':2*32**3/(2*32**2*4),'scope':'mainloop model only, excludes output and cache reuse'}
    if number==13:
        q=np.array([[1.]]);k=np.array([[0.],[1.],[2.]]);v=np.array([[10.],[20.],[30.]])
        y,t=r.online_attention(q,k,v,1,return_trace=True)
        return {'output':y.tolist(),'trace':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in row.items()} for row in t]}
    if number==14:
        q=rng.normal(size=(7,3));k=rng.normal(size=(7,3));v=rng.normal(size=(7,4))
        return {'causal_error':float(np.max(np.abs(r.online_attention(q,k,v,3,True)-r.dense_attention(q,k,v,True))))}
    if number==15:
        n,block=10,8
        return {'program_offsets':[[p*block+i for i in range(block)] for p in range(2)],'valid_counts':[min(block,n-p*block) for p in range(2)]}
    if number==16:
        x=rng.normal(size=(2,5));g=rng.normal(size=x.shape);p=r.stable_softmax(x)
        analytic=r.softmax_backward(p,g)
        numeric=r.finite_difference(lambda z:float((r.stable_softmax(z)*g).sum()),x)
        return {'softmax_gradient_error':float(np.max(np.abs(analytic-numeric)))}
    if number==17:
        x=np.array([[0.,0.,0.],[-1.,.2,1.]])
        q,s=r.quantize_int8_rows(x)
        return {'quantized':q.tolist(),'scales':s.tolist(),'max_error':float(np.max(np.abs(x-q*s)))}
    if number==18:
        x=np.arange(-8,8,dtype=np.int8);p,n=r.pack_int4(x)
        return {'bytes':p.tolist(),'roundtrip':r.unpack_int4(p,n).tolist(),'saved_payload_ratio':x.nbytes/p.nbytes}
    if number==19:
        return {'e2m1_values':r.decode_e2m1(np.arange(16)).tolist(),'note':'E2M1 codes are not signed INT4 numbers; block/global scales are additional metadata'}
    if number==20:
        a=rng.normal(size=(4,7));b=rng.normal(size=(7,3))
        return {'int8_gemm_max_error':float(np.max(np.abs(r.int8_matmul(a,b)-a@b))),'accumulation':'int32 before dequantization'}
    if number==21:
        x=np.arange(8,dtype=float).reshape(4,2);w=np.array([np.eye(2),2*np.eye(2),3*np.eye(2)])
        y,c=r.grouped_experts(x,np.array([[0,1],[1,1],[2,0],[0,2]]),w,np.full((4,2),.5))
        return {'counts':c.tolist(),'output':y.tolist(),'duplicate_assignments':'scatter-add, not assignment'}
    if number==22:
        return {'serial':3*(4+8),'pipeline':r.pipeline_time(3,4,8),'assumptions':'independent load/compute engines, two buffers, zero synchronization overhead'}
    if number==23:
        x=r.Roofline(2e9,1e9,1e12,1e11)
        return {'intensity':x.arithmetic_intensity,'lower_bound_ms':x.lower_bound_s*1000,'bottleneck':x.bottleneck,'hardware':'synthetic, not H100 measurement'}
    if number==24:
        return {'two_ranks_ms':1e3*r.ring_allreduce_seconds(2,64*2**20,5e-6,25e9),'eight_ranks_ms':1e3*r.ring_allreduce_seconds(8,64*2**20,5e-6,25e9),'model':'ideal ring, no topology congestion'}
    if number==25:
        x=np.stack([np.arange(8)+rank for rank in range(4)])
        reduced=x.sum(axis=0); shards=np.split(reduced,4)
        assert np.array_equal(np.concatenate(shards),reduced)
        return {'rank_inputs':x.tolist(),'allreduce_on_each_rank':reduced.tolist(),'reduce_scatter_shards':[a.tolist() for a in shards],'allgather_reconstructs':True,'backend':'CPU semantics, no communication'}
    if number==26:
        q=rng.normal(size=(9,5));k=rng.normal(size=(9,5));v=rng.normal(size=(9,3))
        baseline=r.dense_attention(q,k,v)
        return {'block_errors':{str(b):float(np.max(np.abs(r.online_attention(q,k,v,b)-baseline))) for b in [1,2,4,16]}}
    if number==27:
        q=rng.normal(size=(6,4));v=rng.normal(size=(6,3));k=rng.normal(size=(6,4))
        a=r.online_attention(q,k,v,2,True);v[5]+=1e6;b=r.online_attention(q,k,v,2,True)
        return {'earlier_rows_unchanged':bool(np.allclose(a[:5],b[:5])),'last_row_changes':bool(not np.allclose(a[5],b[5]))}
    if number==28:
        from .capstone import correctness_report
        return correctness_report()
    raise ValueError('lab number must be 1..28')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lab',type=int);group.add_argument('--all',action='store_true')
    args=parser.parse_args()
    numbers=range(1,29) if args.all else [args.lab]
    for number in numbers:
        print(json.dumps({'lab':f'K{number:02d}','result':experiment(number)},ensure_ascii=False))
if __name__=='__main__':main()
