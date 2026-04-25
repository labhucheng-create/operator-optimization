#include <cuda_runtime.h>
#include <cuda_fp16.h>       // __half, __float2half, __half2float
#include <cuda_bf16.h>       // __nv_bfloat16, __float2bfloat16, __bfloat162float
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

#if 0
//目前是使用shared memory的规约版本，并且是用了4个float元素的版本
__global__ void layernormal(float* a, float* b, int B, int N, 
                            float epsilon, float* gamma, float* beta) 
{
    extern __shared__ float sdata[];  // 不指定大小 // 在这里声明，共享内存，使用shared就可以做共享内存了
    __shared__ float smean;  // 在这里声明，使用shared就可以做共享变量了
    __shared__ float svar;  // 在这里声明，使用shared就可以做共享变量了
    float mean = 0; //这里是该行元素的平均值
    float sum = 0; //这里是该行元素的总和
    float var = 0; //这里是该行元素的方差
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //表示每一个线程需要处理的元素的下标
    if (idx >= N/4 * B) 
        return;
    
    // 更改成float版本，目前应该是需要把a矩阵中的元素通过float4的方式进行加载每个线程共享内存中
    float4 tmp_a = FLOAT4(a[idx * 4]);
    sdata[threadIdx.x] = tmp_a.x + tmp_a.y + tmp_a.z + tmp_a.w; //每个线程先把自己对应的元素值给写进去
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
    // 更改成float版本，目前应该是需要把a矩阵中的元素通过float4的方式进行加载每个线程共享内存中
    sdata[threadIdx.x] = (tmp_a.x - smean) * (tmp_a.x - smean);
    sdata[threadIdx.x] += (tmp_a.y - smean) * (tmp_a.y - smean);
    sdata[threadIdx.x] += (tmp_a.z - smean) * (tmp_a.z - smean);
    sdata[threadIdx.x] += (tmp_a.w - smean) * (tmp_a.w - smean);

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
    FLOAT4(b[idx * 4]) = make_float4((tmp_a.x - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4))) + (*(beta + (threadIdx.x * 4))),
                        (tmp_a.y - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4 + 1))) + (*(beta + (threadIdx.x * 4 + 1))),
                        (tmp_a.z - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4 + 2))) + (*(beta + (threadIdx.x * 4 + 2))),
                        (tmp_a.w - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4 + 3))) + (*(beta + (threadIdx.x * 4 + 3))));

    //*(b + idx) = (*(a + idx) - smean) / sqrtf(svar + epsilon) * (*(gamma + threadIdx.x)) + *(beta + threadIdx.x);
}

#else


//目前是使用shared memory的规约版本，并且是用了4个float元素的版本,并且warp内部的线程使用了shuffle指令来进行规约计算
__global__ void layernormal(float* a, float* b, int B, int N, 
                            float epsilon, float* gamma, float* beta) 
{
    extern __shared__ float sdata[];  // 不指定大小 // 在这里声明，共享内存，使用shared就可以做共享内存了
    __shared__ float smean;  // 在这里声明，使用shared就可以做共享变量了
    __shared__ float svar;  // 在这里声明，使用shared就可以做共享变量了
    float mean = 0; //这里是该行元素的平均值
    float sum = 0; //这里是该行元素的总和
    float var = 0; //这里是该行元素的方差

    int idx = blockDim.x * blockIdx.x + threadIdx.x; //表示每一个线程需要处理的元素的下标
    if (idx >= N/4 * B) 
        return;
    
    // 更改成float版本，目前应该是需要把a矩阵中的元素通过float4的方式进行加载每个线程共享内存中
    float4 tmp_a = FLOAT4(a[idx * 4]);
    float tmp_a_sum = tmp_a.x + tmp_a.y + tmp_a.z + tmp_a.w; //每个线程先把自己对应的元素值给写进去

    float4 gamma_tmp = FLOAT4(gamma[threadIdx.x * 4]);
    float4 beta_tmp = FLOAT4(beta[threadIdx.x * 4]);


    // 这里使用shutffle的指令来进行warp的内部的32线程的规约计算
    tmp_a_sum += __shfl_down_sync(0xffffffff, tmp_a_sum, 16);
    tmp_a_sum += __shfl_down_sync(0xffffffff, tmp_a_sum, 8);
    tmp_a_sum += __shfl_down_sync(0xffffffff, tmp_a_sum, 4);
    tmp_a_sum += __shfl_down_sync(0xffffffff, tmp_a_sum, 2);
    tmp_a_sum += __shfl_down_sync(0xffffffff, tmp_a_sum, 1);

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = tmp_a_sum;  // 8 个 warp 各写一个格子
    __syncthreads();                     // 等所有 warp 写完

    if (threadIdx.x < (N/4/32))
    {
        float warp_sum = sdata[threadIdx.x];
        warp_sum += __shfl_down_sync(0x000000ff, warp_sum, 4);
        warp_sum += __shfl_down_sync(0x000000ff, warp_sum, 2);
        warp_sum += __shfl_down_sync(0x000000ff, warp_sum, 1);
        if (threadIdx.x == 0)
            smean = warp_sum / N;
    }
    __syncthreads();                     // 等待剩下的8个线程完成任务

    // 并行规约计算var
    // 每个线程先把自己var值给写进去
    // 更改成float版本，目前应该是需要把a矩阵中的元素通过float4的方式进行加载每个线程共享内存中
    float tmp_a_var_sum = (tmp_a.x - smean) * (tmp_a.x - smean);
    tmp_a_var_sum += (tmp_a.y - smean) * (tmp_a.y - smean);
    tmp_a_var_sum += (tmp_a.z - smean) * (tmp_a.z - smean);
    tmp_a_var_sum += (tmp_a.w - smean) * (tmp_a.w - smean);

    // 这里使用shutffle的指令来进行warp的内部的32线程的规约计算
    tmp_a_var_sum += __shfl_down_sync(0xffffffff, tmp_a_var_sum, 16);
    tmp_a_var_sum += __shfl_down_sync(0xffffffff, tmp_a_var_sum, 8);
    tmp_a_var_sum += __shfl_down_sync(0xffffffff, tmp_a_var_sum, 4);
    tmp_a_var_sum += __shfl_down_sync(0xffffffff, tmp_a_var_sum, 2);
    tmp_a_var_sum += __shfl_down_sync(0xffffffff, tmp_a_var_sum, 1);

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = tmp_a_var_sum;  // (N/4/32) 个 warp 各写一个格子
    __syncthreads();                     // 等所有 warp 写完

    if (threadIdx.x < (N/4/32))
    {
        float warp_var_sum = sdata[threadIdx.x];
        warp_var_sum += __shfl_down_sync(0x000000ff, warp_var_sum, 4);
        warp_var_sum += __shfl_down_sync(0x000000ff, warp_var_sum, 2);
        warp_var_sum += __shfl_down_sync(0x000000ff, warp_var_sum, 1);
        if (threadIdx.x == 0)
            svar = warp_var_sum / N;
    }
    __syncthreads();                     // 等待剩下的8个线程完成任务
    
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
#if 0
    FLOAT4(b[idx * 4]) = make_float4((tmp_a.x - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4))) + (*(beta + (threadIdx.x * 4))),
                        (tmp_a.y - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4 + 1))) + (*(beta + (threadIdx.x * 4 + 1))),
                        (tmp_a.z - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4 + 2))) + (*(beta + (threadIdx.x * 4 + 2))),
                        (tmp_a.w - smean) / sqrtf(svar + epsilon) * (*(gamma + (threadIdx.x * 4 + 3))) + (*(beta + (threadIdx.x * 4 + 3))));
#endif
    FLOAT4(b[idx * 4]) = make_float4((tmp_a.x - smean) / sqrtf(svar + epsilon) * gamma_tmp.x + beta_tmp.x,
                        (tmp_a.y - smean) / sqrtf(svar + epsilon) * gamma_tmp.y + beta_tmp.y,
                        (tmp_a.z - smean) / sqrtf(svar + epsilon) * gamma_tmp.z + beta_tmp.z,
                        (tmp_a.w - smean) / sqrtf(svar + epsilon) * gamma_tmp.w + beta_tmp.w);

    //*(b + idx) = (*(a + idx) - smean) / sqrtf(svar + epsilon) * (*(gamma + threadIdx.x)) + *(beta + threadIdx.x);
}
#endif


// ============================================================
// 模板 LayerNorm：支持 fp32 / fp16 / bf16
// ============================================================

// 把任意类型 T 转成 float，用于内部计算（避免 fp16/bf16 累加溢出），这里用模板函数的方式实现类型转换
template <typename T> __device__ __forceinline__ float to_f32(T v)                   { return (float)v; }
template <> __device__ __forceinline__ float to_f32<__half>(__half v)                { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

// 把 float 转回目标类型 T，用于写回结果
template <typename T> __device__ __forceinline__ T from_f32(float v)                        { return (T)v; }
template <> __device__ __forceinline__ __half from_f32<__half>(float v)                     { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v)      { return __float2bfloat16(v); }

template <typename T>
__global__ void layernormal_template(T* a, T* b, int B, int N,
                                     float epsilon, T* gamma, T* beta)
{
    // VEC_SIZE：128-bit（16字节）一次加载能装几个 T
    // float(4字节) -> 4,  half/bf16(2字节) -> 8
    constexpr int VEC_SIZE = 16 / (int)sizeof(T);

    extern __shared__ float sdata[];
    __shared__ float smean;
    __shared__ float svar;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N / VEC_SIZE * B) return;

    // ---- Step 1: 128-bit 向量化加载，转成 float 数组 ----
    // int4 只是 16 字节的搬运容器，字节内容不变，换视角重新解释成 T 类型
    int4 raw_a      = *(int4*)(&a[idx * VEC_SIZE]);
    T*   raw_a_ptr  = (T*)(&raw_a);

    float f[VEC_SIZE];
    float tmp_sum = 0.0f;
    for (int i = 0; i < VEC_SIZE; i++) {
        f[i] = to_f32(raw_a_ptr[i]);   // 关键：转 float 再累加，防溢出
        tmp_sum += f[i];
    }

    // ---- Step 2: Warp Shuffle 规约求 mean ----
    // num_warps 随 block_size 自动变化：fp32=32, fp16/bf16=16（N=4096时）
    int num_warps       = blockDim.x / 32;
    unsigned mask_all   = 0xffffffff;
    // mask_warps：第二轮规约只用 num_warps 个线程，mask 只开那几位
    unsigned mask_warps = (num_warps >= 32) ? 0xffffffff : ((1u << num_warps) - 1);

    tmp_sum += __shfl_down_sync(mask_all, tmp_sum, 16);
    tmp_sum += __shfl_down_sync(mask_all, tmp_sum, 8);
    tmp_sum += __shfl_down_sync(mask_all, tmp_sum, 4);
    tmp_sum += __shfl_down_sync(mask_all, tmp_sum, 2);
    tmp_sum += __shfl_down_sync(mask_all, tmp_sum, 1);

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = tmp_sum;
    __syncthreads();

    if (threadIdx.x < num_warps) {
        float warp_sum = sdata[threadIdx.x];
        // 用循环代替写死的 stride，自动适配不同 num_warps
        for (int stride = num_warps / 2; stride > 0; stride >>= 1)
            warp_sum += __shfl_down_sync(mask_warps, warp_sum, stride);
        if (threadIdx.x == 0)
            smean = warp_sum / N;
    }
    __syncthreads();

    // ---- Step 3: Warp Shuffle 规约求 var ----
    float tmp_var = 0.0f;
    for (int i = 0; i < VEC_SIZE; i++)
        tmp_var += (f[i] - smean) * (f[i] - smean);

    tmp_var += __shfl_down_sync(mask_all, tmp_var, 16);
    tmp_var += __shfl_down_sync(mask_all, tmp_var, 8);
    tmp_var += __shfl_down_sync(mask_all, tmp_var, 4);
    tmp_var += __shfl_down_sync(mask_all, tmp_var, 2);
    tmp_var += __shfl_down_sync(mask_all, tmp_var, 1);

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = tmp_var;
    __syncthreads();

    if (threadIdx.x < num_warps) {
        float warp_var = sdata[threadIdx.x];
        for (int stride = num_warps / 2; stride > 0; stride >>= 1)
            warp_var += __shfl_down_sync(mask_warps, warp_var, stride);
        if (threadIdx.x == 0)
            svar = warp_var / N;
    }
    __syncthreads();

    // ---- Step 4: 向量化加载 gamma/beta，计算并写回 ----
    int4 raw_g     = *(int4*)(&gamma[threadIdx.x * VEC_SIZE]);
    T*   raw_g_ptr = (T*)(&raw_g);
    int4 raw_bt     = *(int4*)(&beta[threadIdx.x * VEC_SIZE]);
    T*   raw_bt_ptr = (T*)(&raw_bt);

    // rsqrtf = 1/sqrt，一条指令完成，比 /sqrtf 更快
    // 提前计算一次，避免写回时重复算 4/8 次（之前的 bug）
    float inv_std = rsqrtf(svar + epsilon);

    int4 out;
    T*   out_ptr = (T*)(&out);
    for (int i = 0; i < VEC_SIZE; i++) {
        out_ptr[i] = from_f32<T>((f[i] - smean) * inv_std * to_f32(raw_g_ptr[i]) + to_f32(raw_bt_ptr[i]));
    }
    *(int4*)(&b[idx * VEC_SIZE]) = out;
}


int main() {
    constexpr int N = 4096;
    constexpr int B = 2048;
    float epsilon = 1e-5f;

    // ---- fp32 测试 ----
    {
        printf("Testing fp32...\n");
        float* a_h = (float*)malloc(B * N * sizeof(float));
        float* b_h = (float*)malloc(B * N * sizeof(float));
        float* gamma_h = (float*)malloc(N * sizeof(float));
        float* beta_h  = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < B * N; i++) a_h[i] = (float)i / (N * B);
        for (int i = 0; i < N; i++) { gamma_h[i] = 1.0f; beta_h[i] = 0.0f; }

        float *a_d, *b_d, *gamma_d, *beta_d;
        cudaCheck(cudaMalloc(&a_d, B * N * sizeof(float)));
        cudaCheck(cudaMalloc(&b_d, B * N * sizeof(float)));
        cudaCheck(cudaMalloc(&gamma_d, N * sizeof(float)));
        cudaCheck(cudaMalloc(&beta_d,  N * sizeof(float)));
        cudaCheck(cudaMemcpy(a_d, a_h, B * N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(gamma_d, gamma_h, N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(beta_d,  beta_h,  N * sizeof(float), cudaMemcpyHostToDevice));

        // fp32: VEC_SIZE=4, block_size=N/4
        int block_size  = N / 4;
        int shared_size = (block_size / 32) * sizeof(float);   // 存 warp 规约中间结果
        layernormal_template<float><<<B, block_size, shared_size>>>(a_d, b_d, B, N, epsilon, gamma_d, beta_d);
        cudaCheck(cudaMemcpy(b_h, b_d, B * N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("fp32 done. b_h[0]=%.6f\n", b_h[0]);

        free(a_h); free(b_h); free(gamma_h); free(beta_h);
        cudaFree(a_d); cudaFree(b_d); cudaFree(gamma_d); cudaFree(beta_d);
    }

    // ---- fp16 测试 ----
    {
        printf("Testing fp16...\n");
        __half* a_h = (__half*)malloc(B * N * sizeof(__half));
        __half* b_h = (__half*)malloc(B * N * sizeof(__half));
        __half* gamma_h = (__half*)malloc(N * sizeof(__half));
        __half* beta_h  = (__half*)malloc(N * sizeof(__half));
        for (int i = 0; i < B * N; i++) a_h[i] = __float2half((float)i / (N * B));
        for (int i = 0; i < N; i++) { gamma_h[i] = __float2half(1.0f); beta_h[i] = __float2half(0.0f); }

        __half *a_d, *b_d, *gamma_d, *beta_d;
        cudaCheck(cudaMalloc(&a_d, B * N * sizeof(__half)));
        cudaCheck(cudaMalloc(&b_d, B * N * sizeof(__half)));
        cudaCheck(cudaMalloc(&gamma_d, N * sizeof(__half)));
        cudaCheck(cudaMalloc(&beta_d,  N * sizeof(__half)));
        cudaCheck(cudaMemcpy(a_d, a_h, B * N * sizeof(__half), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(gamma_d, gamma_h, N * sizeof(__half), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(beta_d,  beta_h,  N * sizeof(__half), cudaMemcpyHostToDevice));

        // fp16: VEC_SIZE=8, block_size=N/8
        int block_size  = N / 8;
        int shared_size = (block_size / 32) * sizeof(float);
        layernormal_template<__half><<<B, block_size, shared_size>>>(a_d, b_d, B, N, epsilon, gamma_d, beta_d);
        cudaCheck(cudaMemcpy(b_h, b_d, B * N * sizeof(__half), cudaMemcpyDeviceToHost));
        printf("fp16 done. b_h[0]=%.6f\n", __half2float(b_h[0]));

        free(a_h); free(b_h); free(gamma_h); free(beta_h);
        cudaFree(a_d); cudaFree(b_d); cudaFree(gamma_d); cudaFree(beta_d);
    }

    // ---- bf16 测试 ----
    {
        printf("Testing bf16...\n");
        __nv_bfloat16* a_h = (__nv_bfloat16*)malloc(B * N * sizeof(__nv_bfloat16));
        __nv_bfloat16* b_h = (__nv_bfloat16*)malloc(B * N * sizeof(__nv_bfloat16));
        __nv_bfloat16* gamma_h = (__nv_bfloat16*)malloc(N * sizeof(__nv_bfloat16));
        __nv_bfloat16* beta_h  = (__nv_bfloat16*)malloc(N * sizeof(__nv_bfloat16));
        for (int i = 0; i < B * N; i++) a_h[i] = __float2bfloat16((float)i / (N * B));
        for (int i = 0; i < N; i++) { gamma_h[i] = __float2bfloat16(1.0f); beta_h[i] = __float2bfloat16(0.0f); }

        __nv_bfloat16 *a_d, *b_d, *gamma_d, *beta_d;
        cudaCheck(cudaMalloc(&a_d, B * N * sizeof(__nv_bfloat16)));
        cudaCheck(cudaMalloc(&b_d, B * N * sizeof(__nv_bfloat16)));
        cudaCheck(cudaMalloc(&gamma_d, N * sizeof(__nv_bfloat16)));
        cudaCheck(cudaMalloc(&beta_d,  N * sizeof(__nv_bfloat16)));
        cudaCheck(cudaMemcpy(a_d, a_h, B * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(gamma_d, gamma_h, N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(beta_d,  beta_h,  N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        // bf16: VEC_SIZE=8, block_size=N/8（和 fp16 一样，都是 2 字节）
        int block_size  = N / 8;
        int shared_size = (block_size / 32) * sizeof(float);
        layernormal_template<__nv_bfloat16><<<B, block_size, shared_size>>>(a_d, b_d, B, N, epsilon, gamma_d, beta_d);
        cudaCheck(cudaMemcpy(b_h, b_d, B * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        printf("bf16 done. b_h[0]=%.6f\n", __bfloat162float(b_h[0]));

        free(a_h); free(b_h); free(gamma_h); free(beta_h);
        cudaFree(a_d); cudaFree(b_d); cudaFree(gamma_d); cudaFree(beta_d);
    }

    return 0;
}