// Correctness-first softmax and one-thread-per-query online attention.
// Online attention is deliberately not optimized; fixed local capacity DV<=16.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#define CUDA_OK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)
__global__ void row_softmax(const float* x,float* y,int n){
    __shared__ float scratch[128];
    int row=blockIdx.x,t=threadIdx.x;
    float local=-INFINITY;
    for(int c=t;c<n;c+=128)local=fmaxf(local,x[row*n+c]);
    scratch[t]=local;__syncthreads();
    for(int s=64;s>0;s/=2){if(t<s)scratch[t]=fmaxf(scratch[t],scratch[t+s]);__syncthreads();}
    float maximum=scratch[0];
    // All threads must capture maximum before scratch is reused.
    __syncthreads();
    local=0;
    for(int c=t;c<n;c+=128){float e=expf(x[row*n+c]-maximum);y[row*n+c]=e;local+=e;}
    scratch[t]=local;__syncthreads();
    for(int s=64;s>0;s/=2){if(t<s)scratch[t]+=scratch[t+s];__syncthreads();}
    float denominator=scratch[0];
    for(int c=t;c<n;c+=128)y[row*n+c]/=denominator;
}
__global__ void online_attention(const float* q,const float* k,const float* v,float* out,int n,int d,int dv,bool causal){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    float maximum=-INFINITY,denominator=0,numerator[16];
    for(int z=0;z<dv;++z)numerator[z]=0;
    for(int token=0;token<n;++token){
        if(causal&&token>row)continue;
        float score=0;for(int z=0;z<d;++z)score+=q[row*d+z]*k[token*d+z];score/=sqrtf(float(d));
        float next=fmaxf(maximum,score),alpha=expf(maximum-next),p=expf(score-next);
        for(int z=0;z<dv;++z)numerator[z]=alpha*numerator[z]+p*v[token*dv+z];
        denominator=alpha*denominator+p;maximum=next;
    }
    for(int z=0;z<dv;++z)out[row*dv+z]=numerator[z]/denominator;
}
int main(){
    int count=0;CUDA_OK(cudaGetDeviceCount(&count));if(!count){std::puts("SKIP: no CUDA GPU");return 0;}CUDA_OK(cudaSetDevice(0));
    constexpr int M=3,N=131;std::vector<float>x(M*N),y(M*N);
    for(int i=0;i<M*N;++i)x[i]=1000.f+float(i%17)/5;
    float *dx,*dy;CUDA_OK(cudaMalloc(&dx,x.size()*sizeof(float)));CUDA_OK(cudaMalloc(&dy,y.size()*sizeof(float)));
    CUDA_OK(cudaMemcpy(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    row_softmax<<<M,128>>>(dx,dy,N);CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemcpy(y.data(),dy,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int row=0;row<M;++row){double maximum=*std::max_element(x.begin()+row*N,x.begin()+(row+1)*N),den=0;for(int j=0;j<N;++j)den+=std::exp(double(x[row*N+j])-maximum);for(int j=0;j<N;++j){double ref=std::exp(double(x[row*N+j])-maximum)/den;if(std::abs(y[row*N+j]-ref)>1e-6){std::puts("FAIL softmax");return 2;}}}
    CUDA_OK(cudaFree(dx));CUDA_OK(cudaFree(dy));
    constexpr int S=7,D=5,DV=3;std::vector<float>q(S*D),k(S*D),v(S*DV),out(S*DV);
    for(int i=0;i<S*D;++i){q[i]=float(i%7-3)/4;k[i]=float(i%5-2)/3;}for(int i=0;i<S*DV;++i)v[i]=float(i%11-5)/2;
    float *dq,*dk,*dv,*doo;CUDA_OK(cudaMalloc(&dq,q.size()*sizeof(float)));CUDA_OK(cudaMalloc(&dk,k.size()*sizeof(float)));CUDA_OK(cudaMalloc(&dv,v.size()*sizeof(float)));CUDA_OK(cudaMalloc(&doo,out.size()*sizeof(float)));
    CUDA_OK(cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dk,k.data(),k.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dv,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));
    for(bool causal:{false,true}){
        online_attention<<<1,32>>>(dq,dk,dv,doo,S,D,DV,causal);CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemcpy(out.data(),doo,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int row=0;row<S;++row){std::vector<double>scores(S,-INFINITY);double maximum=-INFINITY;for(int token=0;token<S;++token){if(causal&&token>row)continue;double score=0;for(int z=0;z<D;++z)score+=double(q[row*D+z])*k[token*D+z];scores[token]=score/std::sqrt(double(D));maximum=std::max(maximum,scores[token]);}double den=0;for(double s:scores)den+=std::exp(s-maximum);for(int z=0;z<DV;++z){double ref=0;for(int token=0;token<S;++token)ref+=std::exp(scores[token]-maximum)*v[token*DV+z]/den;if(std::abs(out[row*DV+z]-ref)>1e-5){std::puts("FAIL attention");return 3;}}}
    }
    CUDA_OK(cudaFree(dq));CUDA_OK(cudaFree(dk));CUDA_OK(cudaFree(dv));CUDA_OK(cudaFree(doo));
    std::puts("PASS: stable row softmax and causal/noncausal online attention (correctness only)");
}
