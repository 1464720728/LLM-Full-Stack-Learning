"""Optional PyTorch CPU RMSNorm backward + custom_op contract lab."""
import importlib.util

def main():
    if importlib.util.find_spec('torch') is None:
        print('SKIP: PyTorch unavailable. NumPy gradient checks remain in the CPU test suite.');return
    import torch
    class RMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,w,eps):
            inv=torch.rsqrt((x*x).mean(-1,keepdim=True)+eps)
            ctx.save_for_backward(x,w,inv)
            return x*inv*w
        @staticmethod
        def backward(ctx,g):
            x,w,inv=ctx.saved_tensors;u=g*w
            dx=inv*u-x*inv**3*(u*x).mean(-1,keepdim=True)
            dims=tuple(range(x.ndim-1))
            dw=(g*x*inv).sum(dims) if dims else g*x*inv
            return dx,dw,None
    torch.manual_seed(3)
    x=torch.randn(2,5,dtype=torch.double,requires_grad=True)
    w=torch.randn(5,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:RMSNorm.apply(a,b,1e-5),(x,w))
    print('PASS: manual RMSNorm backward gradcheck (CPU double)')
    if not hasattr(torch.library,'custom_op'):
        print('SKIP: torch.library.custom_op unavailable in this PyTorch version');return
    @torch.library.custom_op('llm_course::square',mutates_args=())
    def square(x:torch.Tensor)->torch.Tensor:
        return x*x
    @square.register_fake
    def fake(x):
        # This lab validates contiguous inputs; extend metadata tests before broadening.
        return torch.empty_like(x)
    def setup(ctx,inputs,output):ctx.save_for_backward(inputs[0])
    def backward(ctx,g):return 2*ctx.saved_tensors[0]*g
    square.register_autograd(backward,setup_context=setup)
    result=torch.library.opcheck(square,(x,))
    assert torch.autograd.gradcheck(square,(x,))
    print('PASS: square custom-op registration and gradients',result)
if __name__=='__main__':main()
