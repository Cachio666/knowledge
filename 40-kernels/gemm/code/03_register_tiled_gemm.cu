// 03_register_tiled_gemm.cu —— 线程级寄存器分块 + 双缓冲
//
// 配套文档：40-kernels/gemm/03-register-tiling-and-occupancy.md
//
// 主线：SMEM 解决的是「HBM 流量」，寄存器解决的是「SMEM 带宽」。
//   每 SM 的 FP32 lane 需要一个操作数 8 字节/周期的供给，而 SMEM 只能交付
//   128 B/周期——差 8 倍。把每个操作数在**寄存器**里用 TM（或 TN）次，
//   这个缺口才补得上。TM=TN=8 恰好把需求压到 128 B/周期，与交付率相等。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 03_register_tiled_gemm && ./03_register_tiled_gemm 4096 4096 4096
//     make regs T=03_register_tiled_gemm        # 看寄存器用量与 spill
//     make 03_register_tiled_gemm NVCCFLAGS="-O3 -std=c++17 -DTM=16 -DTN=16"
//         ↑ 这会 spill：16×16 + 32 = 288 > 255
//
// 双缓冲（本文件的做法）：SMEM 里放两份 tile，第 s+1 段在算第 s 段的同时
// 被搬进来。__syncthreads() 的语义在双缓冲下要小心：进入下一轮之前必须保证
// (a) 上一轮的消费者已经读完，(b) 本轮的写入者已经写完。

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

// --- 分块参数（可被 -D 覆盖）---
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif
#ifndef STAGES
#define STAGES 2        // 2 = 双缓冲；>2 是多级流水
#endif

#define NTHREADS ((BM / TM) * (BN / TN))

// SMEM 行距：+8 个 float（32 字节）让相邻行错开整整 8 个 bank，
// 这样「按列取用」也不会撞进同一个 bank。
// 机制的正文在 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict，
// 此处只交代 GEMM 里的用法：**行距必须是 32 的「错开量」不能是 0（mod 32）**。
#define A_STRIDE (BK + 8)
#define B_STRIDE (BN + 8)

// ---------------------------------------------------------------------------
// 寄存器分块内核（多级缓冲）
//
// 线程 (ty, tx) 负责 C 的 TM×TN 一小块：
//     rows: ty*TM .. ty*TM+TM-1
//     cols: tx*TN .. tx*TN+TN-1
// 每个 k 步：
//     从 SMEM 读 TM + TN 个元素，做 TM·TN 次 FMA
//     → 每个 FMA 的 SMEM 字节 = (TM+TN)/(TM·TN) · esize
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
register_tiled_gemm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K)
{
    __shared__ float As[STAGES][BM * A_STRIDE];
    __shared__ float Bs[STAGES][BK * B_STRIDE];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    // 每个线程的累加器：TM·TN 个寄存器（这是寄存器预算里最大的一块）
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    // 协作搬运：整个 block 每段要搬 BM·BK + BK·BN 个元素
    const int per_stage_a = BM * BK;
    const int per_stage_b = BK * BN;

    // --- 预填充前 STAGES-1 段 ---
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        const int k0 = s * BK;
        for (int i = tid; i < per_stage_a; i += NTHREADS) {
            const int r = i / BK, c = i % BK;
            const int gr = row0 + r, gc = k0 + c;
            As[s][r * A_STRIDE + c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < per_stage_b; i += NTHREADS) {
            const int r = i / BN, c = i % BN;
            const int gr = k0 + r, gc = col0 + c;
            Bs[s][r * B_STRIDE + c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
    }

    const int n_iters = (K + BK - 1) / BK;
    for (int it = 0; it < n_iters; ++it) {
        const int cur = it % STAGES;
        const int nxt = (it + 1) % STAGES;

        __syncthreads();                 // 保证 cur 段已写满、且上一轮的读者已退出

        // --- 发起下一段的搬运（与下面的计算重叠）---
        const int k_next = (it + 1) * BK;
        if (k_next < K) {
            for (int i = tid; i < per_stage_a; i += NTHREADS) {
                const int r = i / BK, c = i % BK;
                const int gr = row0 + r, gc = k_next + c;
                As[nxt][r * A_STRIDE + c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
            }
            for (int i = tid; i < per_stage_b; i += NTHREADS) {
                const int r = i / BN, c = i % BN;
                const int gr = k_next + r, gc = col0 + c;
                Bs[nxt][r * B_STRIDE + c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
            }
        }

        // --- 从 cur 段计算 ---
        // 注意：这里把 nxt 的写与 cur 的读写放在**同一个 __syncthreads 区间**里，
        // 是安全的，因为 nxt != cur 且 STAGES >= 2。STAGES=1 时这段代码不成立
        // （那时必须再加一次 barrier），本文件默认 STAGES=2。
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][(threadIdx.y * TM + i) * A_STRIDE + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[cur][k * B_STRIDE + threadIdx.x * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
    }

    // --- epilogue ---
    // C = alpha·(A·B) + beta·C。beta != 0 时这里要多一次全局**读**；
    // 对一个方阵题，那一次读正好等于理想总字节的 MN/(MK+KN+MN) = 1/3，
    // 也就是**白多 33% 的 HBM 流量**。alpha/beta 的取舍见文档 §3。
    const float alpha = 1.0f;
    const float beta = 0.0f;      // 改成 1.0f 观察 epilogue 的额外读
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + threadIdx.y * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + threadIdx.x * TN + j;
            if (gc >= N) continue;
            float* p = &C[(size_t)gr * N + gc];
            *p = alpha * acc[i][j] + beta * (*p);
        }
    }
}

static void gemm_ref(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = (float)acc;
        }
}

template <typename F>
static double time_kernel(F launch, int iters)
{
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    launch(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return (double)ms / iters;
}

int main(int argc, char** argv)
{
    const int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 4096;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 10;

    std::printf("=== 03_register_tiled_gemm ===\n");
    std::printf("形状 M=%d N=%d K=%d，FP32；BM=%d BN=%d BK=%d，TM=%d TN=%d，STAGES=%d\n\n",
                M, N, K, BM, BN, BK, TM, TN, STAGES);

    // --- 文档 §1 的寄存器与线程算术（推导值）---
    const int threads = (BM / TM) * (BN / TN);
    const int c_regs = TM * TN;
    const int est = TM * TN + (TM + TN);            // 不含寻址与索引寄存器
    std::printf("--- 线程级分块算术（推导值）---\n");
    std::printf("线程数 = (BM/TM)·(BN/TN) = (%d/%d)·(%d/%d) = %d（%d 个 warp）\n",
                BM, TM, BN, TN, threads, threads / 32);
    std::printf("每线程寄存器估算 = TM·TN + (TM+TN) = %d + %d = %d\n",
                c_regs, TM + TN, est);
    std::printf("  C 的累加器占 %d 个，操作数与寻址占 %d 个\n", c_regs, est - c_regs);
    std::printf("  ⚠ **不含**寻址与索引寄存器；真实编译通常还要高 30~60%%（~%d）\n",
                (int)(est * 1.5));
    if (est > 255) {
        std::printf("  ✗ 估算 %d > 255（每线程上限）→ **编译器必须 spill**\n", est);
        std::printf("    光 C 就 %d 个%s\n", c_regs,
                    c_regs > 255 ? "，已经超过 255——与寻址寄存器无关，必然 spill" : "");
    } else {
        std::printf("  ✓ 估算 %d <= 255 → 不 spill\n", est);
    }

    // SMEM 带宽账
    const double per_fma_smem = (double)(TM + TN) / (double)(TM * TN) * 4.0;
    std::printf("\n--- SMEM 带宽账（推导值）---\n");
    std::printf("每个 FMA 的 SMEM 字节 = (TM+TN)/(TM·TN)·esize = %d/%d·4 = %.4f B\n",
                TM + TN, TM * TN, per_fma_smem);
    std::printf("128 个 FP32 lane 全速时需要的 SMEM 带宽 = 128 × %.4f = %.1f B/周期\n",
                per_fma_smem, 128 * per_fma_smem);
    std::printf("共享内存无冲突时的交付率 = 128 B/周期 → %s\n",
                128 * per_fma_smem <= 128.0 ? "**够用**" : "**不够，L1 带宽成为瓶颈**");
    std::printf("（TM=TN=8 时恰好相等；4×4 要 256 B/周期，8×4 要 192 B/周期）\n");

    // SMEM 占用
    const int smem = STAGES * (BM * A_STRIDE + BK * B_STRIDE) * 4;
    std::printf("\n--- SMEM 占用 ---\n");
    std::printf("%d 级缓冲 × (BM·A_STRIDE + BK·B_STRIDE)·4 = %d B = %.1f KB（上限 228 KB）→ %s\n",
                STAGES, smem, smem / 1024.0, smem <= 228 * 1024 ? "可行" : "**不可行**");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int nb = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, register_tiled_gemm_kernel, threads, smem));
    std::printf("\n--- 占用率（运行时查询）---\n");
    std::printf("%d 线程/block，每 SM 可驻留 %d 个 → %d 线程 = %d warp，占用率 %.1f%%\n",
                threads, nb, nb * threads, nb * threads / 32,
                nb * threads / (double)prop.maxThreadsPerMultiProcessor * 100.0);
    std::printf("（上限 %d 线程/SM，%d warp/SM）\n",
                prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerMultiProcessor / 32);

    if (threads > 1024) {
        std::printf("\n⚠ 线程数 %d > 每 block 上限 1024 —— 这个配置在硬件上根本起不来。\n",
                    threads);
        return 1;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 4));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 4));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    {
        unsigned s = 4242u;
        float* h = (float*)std::malloc(std::max((size_t)M * K, (size_t)K * N) * 4);
        for (size_t i = 0; i < (size_t)M * K; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dA, h, (size_t)M * K * 4, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)K * N; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dB, h, (size_t)K * N * 4, cudaMemcpyHostToDevice));
        std::free(h);
    }

    const dim3 block(BM / TM, BN / TN);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const double ms = time_kernel([&] {
        register_tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }, iters);
    std::printf("\n--- 本机墙钟（只对本机有效）---\n");
    std::printf("耗时 = %.3f ms，可达算力 = %.3f TFLOPS\n",
                ms, 2.0 * M * N * (double)K / (ms * 1e-3) / 1e12);

    if ((double)M * N * K <= 3.0e8) {
        float* hC = (float*)std::malloc((size_t)M * N * 4);
        float* hA = (float*)std::malloc((size_t)M * K * 4);
        float* hB = (float*)std::malloc((size_t)K * N * 4);
        float* hR = (float*)std::malloc((size_t)M * N * 4);
        CUDA_CHECK(cudaMemcpy(hA, dA, (size_t)M * K * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hB, dB, (size_t)K * N * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        gemm_ref(hA, hB, hR, M, N, K);
        double e = 0.0;
        for (size_t i = 0; i < (size_t)M * N; ++i)
            e = std::fmax(e, std::fabs((double)hC[i] - hR[i]));
        std::printf("\n--- 正确性 ---\nmax|Δ| vs fp64 参考 = %.3e\n", e);
        std::free(hC); std::free(hA); std::free(hB); std::free(hR);
    }

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return 0;
}
