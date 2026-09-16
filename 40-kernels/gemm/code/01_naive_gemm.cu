// 01_naive_gemm.cu —— 朴素 GEMM：一个线程算一个 C 元素
//
// 配套文档：40-kernels/gemm/01-naive-gemm-and-roofline.md
//
// 本文件演示的是**起点**：一个线程负责 C 的一个元素，K 循环里每次都去
// 全局内存取 A、B。它把文档 §1 的两套算术强度口径都摊在明面上：
//
//   口径一（本 kernel 的实际访存）：每个 FMA 要两次全局读
//           I = 2K / (2K·esize) = 1/esize          → FP32 是 0.25
//   口径三（roofline 原文，理想缓存）：每个元素只从 HBM 读一次
//           I = 2MNK / ((MK+KN+MN)·esize)          → 方阵 n 时 = n/(3·esize)
//
// 二者相差三个数量级，**这个差就是本主题后面五篇要填的坑**。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 01_naive_gemm && ./01_naive_gemm 4096 4096 4096
//
// 本程序自带的参考实现跑在 CPU 上（float64 三重循环），只在形状较小时启用。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// 朴素内核：一个线程 = 一个 C 元素
//
// 每一轮 k 迭代里，线程 (row, col) 做：
//     acc += A[row*K + k] * B[k*N + col]
// 两次全局读、一次 FMA。A 的那次读在同 warp 内是**广播**（同一地址），
// B 的那次读在 col 连续时是**合并**的——但这两件事都不改变「每 FMA 8 字节」
// 这个总量：合并只影响一个请求覆盖多少字节，不影响每个线程要多少字节。
// ---------------------------------------------------------------------------
__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;                      // 累加器住在**寄存器**里
    for (int k = 0; k < K; ++k) {
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    C[(size_t)row * N + col] = acc;        // C 只写一次
}

// 变体：K 循环在 kernel 之外，C 每一步都要读回来再写回去（beta·C 的形式）。
// 文档 §2 用它说明「每个 C 元素被反复读写」的代价。
__global__ void naive_gemm_roundtrip_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K, int k_chunk)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    for (int k0 = 0; k0 < K; k0 += k_chunk) {
        float acc = C[(size_t)row * N + col];      // 读回
        for (int k = k0; k < k0 + k_chunk && k < K; ++k) {
            acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = acc;            // 写回
    }
}

// ---------------------------------------------------------------------------
// CPU 参考（float64），只在形状小的时候跑
// ---------------------------------------------------------------------------
static void gemm_ref(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = (float)acc;
        }
    }
}

template <typename F>
static double time_kernel(F launch, int iters)
{
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    launch();                                  // 预热
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (double)ms / iters;
}

int main(int argc, char** argv)
{
    int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
    int N = (argc > 2) ? std::atoi(argv[2]) : 2048;
    int K = (argc > 3) ? std::atoi(argv[3]) : 2048;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 10;

    std::printf("=== 01_naive_gemm ===\n");
    std::printf("形状 M=%d N=%d K=%d，FP32\n\n", M, N, K);

    // --- 设备规格（运行时读取，不写死） ---
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sm_count = prop.multiProcessorCount;
    std::printf("设备：%s，%d 个 SM，compute capability %d.%d\n",
                prop.name, sm_count, prop.major, prop.minor);

    // --- roofline 两侧的数字（推导值） ---
    const double flops = 2.0 * M * N * (double)K;
    const double bytes_ideal = (double)M * K + (double)K * N + (double)M * N;  // 元素数
    const double bytes_ideal_f32 = bytes_ideal * 4.0;
    const double bytes_issued_f32 = 2.0 * M * N * (double)K * 4.0;             // 每个 FMA 两次读
    std::printf("\n--- 文档 §1 的三套口径（推导值） ---\n");
    std::printf("FLOPs                  = 2MNK            = %.4e\n", flops);
    std::printf("口径一 读字节（每 FMA 2 次读）= 2MNK·esize = %.4e B = %.3f GB\n",
                bytes_issued_f32, bytes_issued_f32 / 1e9);
    std::printf("口径三 理想字节（每元素一次） = (MK+KN+MN)·esize = %.4e B = %.3f MB\n",
                bytes_ideal_f32, bytes_ideal_f32 / 1e6);
    std::printf("口径一 I = %.4f FLOP/byte（= 1/esize）\n", flops / bytes_issued_f32);
    std::printf("口径三 I = %.4f FLOP/byte\n", flops / bytes_ideal_f32);
    std::printf("两者相差 %.1f 倍 —— tiling 要填的就是这个坑\n",
                bytes_issued_f32 / bytes_ideal_f32);

    // --- 分配与初始化 ---
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 4));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 4));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    // 用确定性伪随机填，避免依赖 curand
    {
        size_t n = (size_t)M * K;
        float* h = (float*)std::malloc(n * 4);
        unsigned s = 12345u;
        for (size_t i = 0; i < n; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dA, h, n * 4, cudaMemcpyHostToDevice));
        n = (size_t)K * N;
        for (size_t i = 0; i < n; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dB, h, n * 4, cudaMemcpyHostToDevice));
        std::free(h);
    }

    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    double ms = time_kernel([&] {
        naive_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }, iters);

    const double tflops = flops / (ms * 1e-3) / 1e12;
    std::printf("\n--- 朴素内核（本程序自己测的墙钟时间，只对本机有意义） ---\n");
    std::printf("耗时        = %.3f ms\n", ms);
    std::printf("可达算力    = %.3f TFLOPS\n", tflops);

    float peak_f32 = 0.0f;
    // FP32 向量峰值：128 lane/SM × 2 FLOP × 4 个调度槽。这里不写死数值，
    // 只用「lane 数 × 2 × 时钟」，时钟由 deviceProp 给出。
    const double fp32_lanes = sm_count * 128.0;   // Hopper 及以后的数据中心 SM
    peak_f32 = (float)(fp32_lanes * 2.0 * prop.clockRate * 1e3 / 1e12);
    std::printf("按 %d SM × 128 lane × 2 FLOP × %.2f GHz 估的 FP32 峰值 = %.2f TFLOPS\n",
                sm_count, prop.clockRate / 1e6, peak_f32);
    std::printf("占峰值 = %.2f%%\n", tflops / peak_f32 * 100.0);
    std::printf("⚠ 这个百分比是**本程序在这台机器上跑出来的**，只对该机器有效；\n");
    std::printf("  文档里引用的是 `make verify` 的推导值（1.25%%），两者不是一回事。\n");

    // --- 正确性 ---
    if ((double)M * N * K <= 3.0e8) {
        float* hC = (float*)std::malloc((size_t)M * N * 4);
        float* hA = (float*)std::malloc((size_t)M * K * 4);
        float* hB = (float*)std::malloc((size_t)K * N * 4);
        float* hR = (float*)std::malloc((size_t)M * N * 4);
        CUDA_CHECK(cudaMemcpy(hA, dA, (size_t)M * K * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hB, dB, (size_t)K * N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        gemm_ref(hA, hB, hR, M, N, K);
        double max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            double d = std::fabs((double)hC[i] - hR[i]);
            max_abs = std::fmax(max_abs, d);
            max_rel = std::fmax(max_rel, d / std::fmax(std::fabs((double)hR[i]), 1e-30));
        }
        std::printf("\n--- 正确性 ---\n");
        std::printf("相对 fp64 参考：max|Δ| = %.3e，max 相对元素误差 = %.3e\n", max_abs, max_rel);
        std::printf("⚠ 相对元素误差被接近 0 的元素支配（回链\n");
        std::printf("  10-foundations/numerical-linear-algebra/02），两个口径要分开看。\n");
        std::free(hC); std::free(hA); std::free(hB); std::free(hR);
    }

    // --- K 循环在外面的变体：C 的往返流量 ---
    const int k_chunk = std::max(1, K / 8);
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * 4));
    const double ms2 = time_kernel([&] {
        naive_gemm_roundtrip_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, k_chunk);
    }, iters);
    std::printf("\n--- 变体：K 循环在 kernel 外（C 每步读改写），k_chunk = %d ---\n", k_chunk);
    std::printf("耗时 = %.3f ms，可达算力 = %.3f TFLOPS（占峰值 %.2f%%）\n",
                ms2, flops / (ms2 * 1e-3) / 1e12, flops / (ms2 * 1e-3) / 1e12 / peak_f32 * 100.0);
    std::printf("C 的往返字节 = 2·MN·esize·(K/k_chunk) = %.3f GB\n",
                2.0 * M * N * 4.0 * (double)(K / k_chunk) / 1e9);
    std::printf("k_chunk=1（即 K 循环完全摊到 host 侧）时是 2MNK·esize = %.3f GB，\n",
                2.0 * M * N * (double)K * 4.0 / 1e9);
    std::printf("此时文档口径二的 I = 1/(2·esize) = 0.125，比口径一（0.25）再低一半。\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
