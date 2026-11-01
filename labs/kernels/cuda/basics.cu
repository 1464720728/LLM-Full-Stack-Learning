// Bounded CUDA fundamentals. Build for your actual GPU; see chapter 03.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} } while(0)
__global__ void add(const float* a,const float* b,float* c,int n){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)c[i]=a[i]+b[i];
}
__global__ void reduce_sum(const float* a,float* partial,int n){
    __shared__ float s[128];
    int t=threadIdx.x;int i=blockIdx.x*blockDim.x+t;
    s[t]=(i<n)?a[i]:0.0f;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){if(t<stride)s[t]+=s[t+stride];__syncthreads();}
    if(t==0)partial[blockIdx.x]=s[0];
}
template<int T> __global__ void matmul(const float* a,const float* b,float* c,int m,int n,int k){
    __shared__ float sa[T][T],sb[T][T];
    int row=blockIdx.y*T+threadIdx.y,col=blockIdx.x*T+threadIdx.x;
    float sum=0;
    for(int p=0;p<k;p+=T){
        int ak=p+threadIdx.x,bk=p+threadIdx.y;
        sa[threadIdx.y][threadIdx.x]=(row<m&&ak<k)?a[row*k+ak]:0;
        sb[threadIdx.y][threadIdx.x]=(bk<k&&col<n)?b[bk*n+col]:0;
        __syncthreads();
        for(int j=0;j<T;++j)sum+=sa[threadIdx.y][j]*sb[j][threadIdx.x];
        __syncthreads(); // no thread may overwrite a tile still being consumed
    }
    if(row<m&&col<n)c[row*n+col]=sum;
}
int main(){
    int count=0;CUDA_OK(cudaGetDeviceCount(&count));if(!count){std::puts("SKIP: no CUDA GPU");return 0;}
    CUDA_OK(cudaSetDevice(0));cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,0));
    std::printf("GPU=%s CC=%d.%d\n",prop.name,prop.major,prop.minor);
    constexpr int N=1003;std::vector<float>a(N,1),b(N,2),c(N);
    float *da,*db,*dc,*dp;CUDA_OK(cudaMalloc(&da,N*sizeof(float)));CUDA_OK(cudaMalloc(&db,N*sizeof(float)));CUDA_OK(cudaMalloc(&dc,N*sizeof(float)));CUDA_OK(cudaMalloc(&dp,8*sizeof(float)));
    CUDA_OK(cudaMemcpy(da,a.data(),N*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,b.data(),N*sizeof(float),cudaMemcpyHostToDevice));
    add<<<8,128>>>(da,db,dc,N);CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemcpy(c.data(),dc,N*sizeof(float),cudaMemcpyDeviceToHost));
    for(float x:c)if(x!=3){std::puts("FAIL add");return 2;}
    reduce_sum<<<8,128>>>(da,dp,N);CUDA_OK(cudaGetLastError());std::vector<float>p(8);CUDA_OK(cudaMemcpy(p.data(),dp,8*sizeof(float),cudaMemcpyDeviceToHost));float sum=0;for(float x:p)sum+=x;if(sum!=N)return 3;
    CUDA_OK(cudaFree(da));CUDA_OK(cudaFree(db));CUDA_OK(cudaFree(dc));CUDA_OK(cudaFree(dp));
    constexpr int M=19,K=23,C=17; a.resize(M*K);b.resize(K*C);c.resize(M*C);
    for(int i=0;i<M*K;++i)a[i]=float(i%7-3)/8;for(int i=0;i<K*C;++i)b[i]=float(i%5-2)/4;
    CUDA_OK(cudaMalloc(&da,a.size()*sizeof(float)));CUDA_OK(cudaMalloc(&db,b.size()*sizeof(float)));CUDA_OK(cudaMalloc(&dc,c.size()*sizeof(float)));
    CUDA_OK(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
    matmul<16><<<dim3((C+15)/16,(M+15)/16),dim3(16,16)>>>(da,db,dc,M,C,K);CUDA_OK(cudaGetLastError());CUDA_OK(cudaMemcpy(c.data(),dc,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int i=0;i<M;++i)for(int j=0;j<C;++j){float ref=0;for(int k=0;k<K;++k)ref+=a[i*K+k]*b[k*C+j];if(std::fabs(c[i*C+j]-ref)>1e-4){std::puts("FAIL matmul");return 4;}}
    CUDA_OK(cudaFree(da));CUDA_OK(cudaFree(db));CUDA_OK(cudaFree(dc));std::puts("PASS: add, reduction, tiled matmul (correctness only)");
}
