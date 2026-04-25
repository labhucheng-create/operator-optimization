#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define FLOAT4(a) *(float4*)(&(a))
#define CEIL(a,b) ((a+b-1)/(b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) 
{
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

//目前是使用shared memory的规约版本
__global__ void layernormal(float* a, float* b, int B, int N, 
                            float epsilon, float* gamma, float* beta) 
{
    __shared__ float sdata[1024];  // 在这里声明，共享内存，使用shared就可以做共享内存了
    __shared__ float smean;  // 在这里声明，使用shared就可以做共享变量了
    __shared__ float svar;  // 在这里声明，使用shared就可以做共享变量了
    float mean = 0; //这里是该行元素的平均值
    float sum = 0; //这里是该行元素的总和
    float var = 0; //这里是该行元素的方差
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //表示每一个线程需要处理的元素的下标
    if (idx >= N * B) 
        return;
    
    // 每个线程把自己的值写进去
    sdata[threadIdx.x] = a[idx];
    __syncthreads();

    // 并行规约计算mean
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) 
    {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    //并行规约已经把对应一行的总和值写到0下标了，所以应该对应是下标是0的线程去处理,然后再做一次同步
    if(threadIdx.x == 0)
        smean = sdata[0] / N;
    __syncthreads();

    // 并行规约计算var
    // 每个线程先把自己var值给写进去
    sdata[threadIdx.x] = (a[idx] - smean) * (a[idx] - smean);

    // 并行规约计算var
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) 
    {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    //并行规约已经把对应一行的var值写到0下标了，所以应该对应是下标是0的线程去处理,然后再做一次同步
    if(threadIdx.x == 0)
        svar = sdata[0] / N;
    __syncthreads();
    
#if 0 //把native版本隐藏
    //先把数学公式的表达进行编写
    //先把mean给计算出来
    for(int i = 0; i < N; i++)
        sum += *(a +  blockDim.x * blockIdx.x + i);
    
    mean = sum / N;
    for(int i = 0; i < N; i++)
        var += (*(a +  blockDim.x * blockIdx.x + i) - mean) * (*(a +  blockDim.x * blockIdx.x + i) - mean);
    var = var / N;
#endif

    //更新对应的每个元素里面的归一化的值到b矩阵中
    *(b + idx) = (*(a + idx) - smean) / sqrtf(svar + epsilon) * (*(gamma + threadIdx.x)) + *(beta + threadIdx.x);
}


int main() {
    //这里创建一个4行8列的矩阵
    constexpr int N = 1024;
    constexpr int B = 1024;
    int i,j = 0;
    float* a_h = (float*)malloc(B * N * sizeof(float));
    float* b_h = (float*)malloc(B * N * sizeof(float));

    //这里对应申请模型参数
    float* gamma = (float*)malloc(N * sizeof(float));
    float* beta  = (float*)malloc(N * sizeof(float));
    float epsilon = 1e-5f;

    //这里把对应的矩阵进行初始化
    for (i = 0; i < B * N; i++) {
            a_h[i] = i;
    }

    //把对应的模型参数进行初始化
    memset((void *)beta, 0.0f, N * sizeof(float));

    for(i = 0; i < N; i++)
    {
        gamma[i] = 1.0f;
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* gamma_d = nullptr;
    float* beta_d = nullptr;
    cudaCheck(cudaMalloc((void**)&a_d, B * N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&b_d, B * N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&gamma_d, N * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&beta_d, N * sizeof(float)));
    cudaCheck(cudaMemcpy(a_d, a_h, B * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gamma_d, gamma, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(beta_d, beta, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = N;
    int grid_size = B;

    //算子调用
    layernormal<<<grid_size, block_size>>>(a_d, b_d, B, N, epsilon, gamma_d, beta_d);
   
    //把数据拷贝出来，这里是存放的是归一化之后的数据
    cudaCheck(cudaMemcpy(b_h, b_d, B * N * sizeof(float), cudaMemcpyDeviceToHost));

    //打印查看数据的值
#if 0
    printf("a_h:\n");
    int printf_index = 0;
    for (i = 0; i < B; i++) 
    {
        for(j= 0; j < N; j++)
        {
            if (j == N-1) printf("%f\n", a_h[printf_index]);
            else printf("%f ", a_h[printf_index]);
            printf_index++;
        }
    }

    printf("b_h:\n");
    printf_index = 0;
    for (i = 0; i < B; i++) 
    {
        for(j= 0; j < N; j++)
        {
            if (j == N-1) printf("%f\n", b_h[printf_index]);
            else printf("%f ", b_h[printf_index]);
            printf_index++;
        }
    }
#endif
    return 0;
}