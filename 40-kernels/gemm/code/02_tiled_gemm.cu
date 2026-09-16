// 02_tiled_gemm.cu —— 共享内存分块 + padding 消除 bank 冲突
//
// 配套文档：40-kernels/gemm/02-shared-memory-tiling.md
//
// 主线一（§1）：把「每个 block 反复从 HBM 取同样几个元素」改成
//   「取一次、放进 SMEM、在 block 内复用 BM（或 BN）次」。
//   降幅 = 2·BM·BN/(BM+BN) = 块级 AI × esize。
//
// 主线二（§3）：SMEM 里的**行距**决定 bank 冲突。
//   本文件把 B **转置**存进 SMEM（BsT[n][k]），因为这样内层循环读 B
//   时 lane 方向是连续的 n——而这恰好是那种「行距 = 32 就撞车」的访问模式。
//   A 不转置（As[m][k]），它的读在同一 warp 内是**广播**，本来就不会冲突。
//
//   PAD 控制 BsT 的行距（单位为 float）。warp 内有 16 个不同地址（tx=0..15，
//   同一 tx 的两个 ty 读同一地址 → 广播），它们落进哪些 bank 由行距决定：
//     PAD=0 → 行距 32 → 16 个地址全挤在同一个 bank → **16 路冲突**
//     PAD=1 → 行距 33 → 16 个地址落在 16 个不同 bank → **无冲突**
//     PAD=2 → 行距 34 → 2 路；PAD=4 → 行距 36 → 4 路；PAD=8 → 行距 40 → 8 路
//   规律：行距的偶数值直接就是冲突路数，奇数行距完全无冲突。
//   程序会**按 bank 映射规则把这个路数算出来并打印**（纯算术，不需要 GPU）。
//
// bank 冲突的**机制**（32 个 bank、4 字节宽、地址按 4 字节轮流分配、
// 一次访问拆成几遍）正文在 20-gpu-architecture/memory-system/03，
// 此处只讲它在 GEMM 里怎么用。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 02_tiled_gemm && ./02_tiled_gemm 4096 4096 4096
//     加 -DPAD=0 / -DPAD=1 / -DPAD=4 / -DPAD=8 对比（默认 1）

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

// --- 分块参数：与文档 02 §1 的表格同一组（BM=BN=128, BK=32 → smem 32 KB，块级 AI 32）---
#define BM 128
#define BN 128
#define BK 32

#ifndef PAD
#define PAD 1          // BsT 的行距 = BK + PAD，单位 float
#endif

#define NTHREADS 256   // 16×16

// ---------------------------------------------------------------------------
// 分块内核
//
// 线程布局：16×16 = 256 线程，每个线程算 8 行 × 2 列 = 16 个 C 元素。
//   tx（0..15）→ 列方向，每线程 2 列（tcol0 = tx*2）
//   ty（0..15）→ 行方向，每线程 8 行（trow0 = ty*8）
//
// 内层循环里的 SMEM 读：
//   a[i] = As[(trow0+i)*BK + k]        ← 同一 warp 内地址相同 → 广播，不冲突
//   b[j] = BsT[(tcol0+j)*(BK+PAD) + k] ← lane 方向 stride = 2·(BK+PAD)
//                                        PAD 决定它撞几个 bank
// ---------------------------------------------------------------------------
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    __shared__ float As[BM * BK];              // A 不转置，读是广播，不需要 padding
    __shared__ float BsT[BN * (BK + PAD)];     // B 转置存放，行距 = BK + PAD

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    const int TILE_M = 8, TILE_N = 2;
    const int trow0 = ty * TILE_M;
    const int tcol0 = tx * TILE_N;

    float acc[TILE_M][TILE_N];
    for (int i = 0; i < TILE_M; ++i)
        for (int j = 0; j < TILE_N; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // --- 1. 协作搬运：A 的 BM×BK（行主序，直接落位）---
        for (int i = tid; i < BM * BK; i += NTHREADS) {
            const int r = i / BK, c = i % BK;
            const int gr = row0 + r, gc = k0 + c;
            As[i] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        // --- B 的 BK×BN，**转置**存成 [n][k] ---
        // 读是合并的（c 连续），写进 SMEM 时 stride = BK+PAD
        for (int i = tid; i < BK * BN; i += NTHREADS) {
            const int r = i / BN, c = i % BN;          // B[k0+r][col0+c]
            const int gr = k0 + r, gc = col0 + c;
            BsT[c * (BK + PAD) + r] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();

        // --- 3. 从 SMEM 取用 ---
        for (int k = 0; k < BK; ++k) {
            float a[TILE_M], b[TILE_N];
            for (int i = 0; i < TILE_M; ++i) a[i] = As[(trow0 + i) * BK + k];
            for (int j = 0; j < TILE_N; ++j) b[j] = BsT[(tcol0 + j) * (BK + PAD) + k];
            for (int i = 0; i < TILE_M; ++i)
                for (int j = 0; j < TILE_N; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // --- epilogue：C = alpha·(A·B) + beta·C ---
    // beta != 0 时这里要多一次全局**读**；见文档 §3 末的账。
    const float alpha = 1.0f;
    const float beta = 0.0f;      // 改成 1.0f 就能把 epilogue 的额外读算进去
    for (int i = 0; i < TILE_M; ++i) {
        const int gr = row0 + trow0 + i;
        if (gr >= M) continue;
        for (int j = 0; j < TILE_N; ++j) {
            const int gc = col0 + tcol0 + j;
            if (gc < N) {
                float* p = &C[(size_t)gr * N + gc];
                *p = alpha * acc[i][j] + beta * (*p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bank 冲突路数：按「32 个 bank、每个 4 字节、地址按 4 字节轮流分配」直接算。
// 这是**地址映射规则的应用**（规则本身正文在 memory-system/03），不是硬件测量。
//
// 考察 warp 内第 j 次 B 读：lane (ty, tx) 的地址 = (2·tx + j)·(BK+PAD) + k。
// 注意 ty 不出现在地址里 —— 同一 tx 的两个 lane 读同一地址（广播，不额外收费）。
// 所以一个 warp 有 16 个**不同**地址，把它们落在哪个 bank 数一遍即可。
// ---------------------------------------------------------------------------
static int bank_conflict_ways(int pad)
{
    const int stride = BK + pad;
    const int k = 7;                       // 任取一个 k，结果与 k 无关
    int max_ways = 1;
    for (int j = 0; j < 2; ++j) {
        int per_bank[32] = {0};
        for (int tx = 0; tx < 16; ++tx) {
            const int word = (2 * tx + j) * stride + k;
            per_bank[word % 32] += 1;      // 每个 (tx, j) 是一个不同地址
        }
        for (int b = 0; b < 32; ++b) max_ways = std::max(max_ways, per_bank[b]);
    }
    return max_ways;
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

    std::printf("=== 02_tiled_gemm ===\n");
    std::printf("形状 M=%d N=%d K=%d，FP32，BM=%d BN=%d BK=%d，PAD=%d\n\n",
                M, N, K, BM, BN, BK, PAD);

    // --- §1 的分块算术（推导值，与 make verify 的 §B3/§B4 一致）---
    const int smem_bytes = BM * BK * 4 + BN * (BK + PAD) * 4;
    const double ai_block = 2.0 * BM * BN * BK / (double)((BM * BK + BK * BN) * 4);
    const double naive_read = 2.0 * M * N * (double)K * 4;
    const double tiled_read = (double)(M / BM) * (N / BN) *
                              ((double)BM * K + (double)K * BN) * 4;
    std::printf("--- 分块算术（推导值）---\n");
    std::printf("SMEM/block = BM·BK·4 + BN·(BK+%d)·4 = %d B = %.1f KB（上限 228 KB）\n",
                PAD, smem_bytes, smem_bytes / 1024.0);
    std::printf("块级 AI（**L2 零命中口径**） = 2·BM·BN/((BM+BN)·4) = %.2f\n", ai_block);
    std::printf("  ↑ BK 被约掉了：这套口径下 AI 只由 BM、BN、esize 决定（BK 改的是 SMEM 占用）\n");
    std::printf("HBM 读字节：朴素 %.3f GB → 分块 %.3f GB，降低 %.1f 倍（= 块级 AI × esize）\n",
                naive_read / 1e9, tiled_read / 1e9, naive_read / tiled_read);
    std::printf("⚠ 零缓存口径。整题级的读字节是 (MK+KN)·esize = %.1f MB，与 BM/BN 无关——\n",
                ((double)M * K + (double)K * N) * 4 / 1e6);
    std::printf("  tiling 不减少总字节，它减少的是**重复搬运的次数**。\n");

    // --- §3 的 bank 冲突分析（纯算术）---
    std::printf("\n--- SMEM bank 冲突（按 bank 映射规则算出的路数（纯算术，不碰硬件））---\n");
    std::printf("B 的行距（float 数）  冲突路数  一次 B 读被拆成几遍\n");
    for (int p : {0, 1, 2, 4, 8}) {
        const int ways = bank_conflict_ways(p);
        std::printf("      BK+%d = %-3d          %2d          %s\n",
                    p, BK + p, ways, ways == 1 ? "1 遍（无冲突）" : "多遍");
    }
    std::printf("本文件 PAD=%d → 行距 %d → %d 路冲突\n", PAD, BK + PAD,
                bank_conflict_ways(PAD));
    std::printf("A 不转置，读是广播（同一 warp 内地址相同）→ 不需要 padding\n");
    std::printf("⚠ PAD=1 完全消除冲突，但行距 33 个 float 破坏了 16 字节对齐，\n");
    std::printf("  用不了 LDS.128（float4 向量读）。要保持对齐又要打散 bank，\n");
    std::printf("  得用 swizzle（XOR 行号到列地址上）——那是 tensor-core 里 SMEM 的常规做法。\n");

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const dim3 block(16, 16);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    int nb_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb_per_sm, tiled_gemm_kernel, NTHREADS, smem_bytes));
    std::printf("\n--- 占用率（运行时查询）---\n");
    std::printf("%d 线程/block，grid = (%d,%d) = %d 个 block\n",
                NTHREADS, grid.x, grid.y, grid.x * grid.y);
    std::printf("每 SM 可驻留 %d 个 block → %d 线程 = %d warp，占用率 %.1f%%\n",
                nb_per_sm, nb_per_sm * NTHREADS, nb_per_sm * NTHREADS / 32,
                nb_per_sm * NTHREADS / (double)prop.maxThreadsPerMultiProcessor * 100.0);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 4));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 4));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    {
        unsigned s = 999u;
        float* h = (float*)std::malloc(std::max((size_t)M * K, (size_t)K * N) * 4);
        for (size_t i = 0; i < (size_t)M * K; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dA, h, (size_t)M * K * 4, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)K * N; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dB, h, (size_t)K * N * 4, cudaMemcpyHostToDevice));
        std::free(h);
    }

    const double ms = time_kernel([&] {
        tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }, iters);
    std::printf("\n--- 本机墙钟（只对本机有效）---\n");
    std::printf("耗时 = %.3f ms，可达算力 = %.3f TFLOPS\n",
                ms, 2.0 * M * N * (double)K / (ms * 1e-3) / 1e12);
    std::printf("换 PAD 重编一次即可对比：make 02_tiled_gemm NVCCFLAGS=\"-O3 -std=c++17 -DPAD=0\"\n");

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
