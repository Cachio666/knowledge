// 06_split_k_gemm.cu —— split-K / Stream-K 与归约的确定性
//
// 配套文档：40-kernels/gemm/06-split-k-and-stream-k.md
//
// 主线：M/N 小、K 大时，(M/BM)·(N/BN) 只有个位数个 block，SM 大量空闲，
// 而算力需求一点没变。split-K 把 K 切给额外的 block 来填机器；代价是
// **归约顺序不确定 → 结果不可复现**。
//
// 本文件实现两条归约路线并**在同一台机器上对比它们的位模式**：
//   路线 A：atomicAdd 直接累加进 C          → 顺序由硬件决定，逐位不可复现
//   路线 B：写到 workspace[S][M][N]，再由第二个 kernel 按固定顺序归约
//                                          → 给定 S，逐位可复现
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 06_split_k_gemm && ./06_split_k_gemm 256 256 65536 32

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

#define BM 128
#define BN 128
#define BK 32

// ---------------------------------------------------------------------------
// split-K 的分段块：block (bx, by, bz) 负责第 bz 段 K，把部分和累加出去
//
// 每段的 K 长度相等（K/splits），最后一段吃掉余数。**每段读到的是 A、B 的
// 不同 K 区间**，所以总读字节不随 splits 增加——split-K 的代价不在主循环，
// 在**部分和 C 的搬运**：splits × MN × esize 的写 + 同样多的读。
// ---------------------------------------------------------------------------
__global__ void splitk_partial_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ out,
                                      int M, int N, int K, int splits,
                                      int mode /*0 = atomicAdd, 1 = workspace*/)
{
    __shared__ float As[BM * (BK + 8)];
    __shared__ float Bs[BK * (BN + 8)];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    const int k_per = (K + splits - 1) / splits;
    const int k_begin = blockIdx.z * k_per;
    const int k_end = min(k_begin + k_per, K);

    const int TILE_M = 8, TILE_N = 2;
    const int trow0 = ty * TILE_M, tcol0 = tx * TILE_N;
    float acc[TILE_M][TILE_N];
    for (int i = 0; i < TILE_M; ++i)
        for (int j = 0; j < TILE_N; ++j) acc[i][j] = 0.0f;

    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            const int r = i / BK, c = i % BK;
            const int gr = row0 + r, gc = k0 + c;
            As[r * (BK + 8) + c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            const int r = i / BN, c = i % BN;
            const int gr = k0 + r, gc = col0 + c;
            Bs[r * (BN + 8) + c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BK && k0 + k < k_end; ++k) {
            float a[TILE_M], b[TILE_N];
            for (int i = 0; i < TILE_M; ++i) a[i] = As[(trow0 + i) * (BK + 8) + k];
            for (int j = 0; j < TILE_N; ++j) b[j] = Bs[k * (BN + 8) + tcol0 + j];
            for (int i = 0; i < TILE_M; ++i)
                for (int j = 0; j < TILE_N; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TILE_M; ++i) {
        const int gr = row0 + trow0 + i;
        if (gr >= M) continue;
        for (int j = 0; j < TILE_N; ++j) {
            const int gc = col0 + tcol0 + j;
            if (gc >= N) continue;
            if (mode == 0) {
                // 路线 A：直接累加。**32 个段谁先谁后由硬件决定。**
                atomicAdd(&out[(size_t)gr * N + gc], acc[i][j]);
            } else {
                // 路线 B：写进自己的那一片 workspace，顺序由后面的 kernel 定。
                out[((size_t)blockIdx.z * M + gr) * N + gc] = acc[i][j];
            }
        }
    }
}

// 路线 B 的归约：按 s = 0,1,...,S-1 的**固定顺序**累加。这个顺序是代码写的，
// 不是硬件定的，所以逐位可复现。
__global__ void splitk_reduce_kernel(const float* __restrict__ partial,
                                     float* __restrict__ C,
                                     int M, int N, int splits)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * N) return;
    float acc = 0.0f;
    for (int s = 0; s < splits; ++s) acc += partial[(size_t)s * M * N + idx];
    C[idx] = acc;
}

// ---------------------------------------------------------------------------
// Stream-K 的工作对划分（host 侧，与 make verify 的 A3 断言同一套口径）
//
// 工作对 = (tile, k-chunk)，共 U = T · (K/BK_chunk) 个。分给固定的 P 个 block，
// 第 b 个 block 拿全局序号 [b·U/P, (b+1)·U/P) 这一段连续的号。
// 与 split-K 的区别：split-K 的 block 数是 T·S（**量化**的），
// Stream-K 的 block 数是你指定的 P（**正好**是机器的槽位数）。
// ---------------------------------------------------------------------------
static void streamk_partition_report(int T, int U_per_tile, int P)
{
    const long long U = (long long)T * U_per_tile;
    std::printf("    Stream-K: P=%d，每个 block 领到的连续工作对数 = ", P);
    long long mn = U, mx = 0, sum = 0;
    for (int b = 0; b < P; ++b) {
        const long long lo = (long long)b * U / P, hi = (long long)(b + 1) * U / P;
        mn = std::min(mn, hi - lo);
        mx = std::max(mx, hi - lo);
        sum += hi - lo;
    }
    std::printf("%lld~%lld（共 %lld/%lld，%s）\n", mn, mx, sum, U,
                sum == U ? "完备划分" : "**有遗漏**");
    std::printf("      负载不均 = %.2f%%（1 个工作对约占总时间 %.3f%%）\n",
                (double)(mx - mn) / (double)mx * 100.0, 100.0 / (double)U);
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
    const int M = (argc > 1) ? std::atoi(argv[1]) : 256;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 65536;
    const int S = (argc > 4) ? std::atoi(argv[4]) : 32;
    const int iters = 5;

    std::printf("=== 06_split_k_gemm ===\n");
    std::printf("形状 M=%d N=%d K=%d（瘦高），BM=%d BN=%d BK=%d，splits S=%d\n\n",
                M, N, K, BM, BN, BK, S);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sm = prop.multiProcessorCount;

    // --- §0/§1 的算术（推导值，与 make verify 的输出一致）---
    const int tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    const double flops = 2.0 * M * N * (double)K;
    const double bw = 3350e9;               // H100 SXM 的规格值，写在这里只作量级对照
    const double red_bytes = 2.0 * S * (double)M * N * 4;
    std::printf("--- 并行度与代价（推导值）---\n");
    std::printf("未切分网格 = (M/BM)·(N/BN) = %d 个 block，机器有 %d 个 SM\n", tiles, sm);
    std::printf("  ⟹ 只有 %.2f%% 的 SM 有活干，空闲 %d 个\n",
                tiles / (double)sm * 100.0, sm - tiles);
    std::printf("切分后网格 = %d × S = %d 个 block，占用 SM = %d\n",
                tiles, tiles * S, std::min(tiles * S, sm));
    std::printf("归约流量（atomicAdd 口径：读改写各一次）= 2·S·MN·esize = %.1f MB\n",
                red_bytes / 1e6);
    std::printf("  按 H100 的 3.35 TB/s 折算 ≈ %.2f us\n", red_bytes / bw * 1e6);
    std::printf("⚠ 3.35 TB/s 是规格值，这里只做量级对照；本仓库没有跑过任何内核。\n");

    std::printf("\n--- 工作对划分（host 侧计算，与 verify.py 的 A3 同一口径）---\n");
    const int U_per_tile = std::max(1, K / 1024);
    std::printf("    split-K: 网格 = T·S = %d（S 只能取整数段 → **量化**）\n", tiles * S);
    for (int P : {sm, sm * 2, 256}) streamk_partition_report(tiles, U_per_tile, P);

    // --- 分配与初始化 ---
    float *dA, *dB, *dC, *dW;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 4));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 4));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    CUDA_CHECK(cudaMalloc(&dW, (size_t)S * M * N * 4));
    {
        unsigned s = 777u;
        float* h = (float*)std::malloc(std::max((size_t)M * K, (size_t)K * N) * 4);
        // 用**量级悬殊**的数据：一半元素很小，让归约顺序的差异显出来
        for (size_t i = 0; i < (size_t)M * K; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dA, h, (size_t)M * K * 4, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)K * N; ++i) { s = s * 1103515245u + 12345u; h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        CUDA_CHECK(cudaMemcpy(dB, h, (size_t)K * N * 4, cudaMemcpyHostToDevice));
        std::free(h);
    }

    const dim3 block(16, 16);
    const int nthreads = block.x * block.y;

    // --- 路线 A：atomicAdd ---
    const dim3 gridA((N + BN - 1) / BN, (M + BM - 1) / BM, S);
    const double msA = time_kernel([&] {
        CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * 4));
        splitk_partial_kernel<<<gridA, block>>>(dA, dB, dC, M, N, K, S, 0);
    }, iters);

    // 跑第二遍，看位模式是否一致
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * 4));
    splitk_partial_kernel<<<gridA, block>>>(dA, dB, dC, M, N, K, S, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* hA1 = (float*)std::malloc((size_t)M * N * 4);
    CUDA_CHECK(cudaMemcpy(hA1, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * 4));
    splitk_partial_kernel<<<gridA, block>>>(dA, dB, dC, M, N, K, S, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* hA2 = (float*)std::malloc((size_t)M * N * 4);
    CUDA_CHECK(cudaMemcpy(hA2, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));

    // --- 路线 B：workspace + 第二个 kernel ---
    const double msB = time_kernel([&] {
        splitk_partial_kernel<<<gridA, block>>>(dA, dB, dW, M, N, K, S, 1);
        splitk_reduce_kernel<<<(M * N + 255) / 256, 256>>>(dW, dC, M, N, S);
    }, iters);
    float* hB1 = (float*)std::malloc((size_t)M * N * 4);
    CUDA_CHECK(cudaMemcpy(hB1, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));

    std::printf("\n--- 两次运行 atomicAdd 路线的位模式 ---\n");
    size_t diff = 0;
    double maxd = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        if (hA1[i] != hA2[i]) ++diff;
        maxd = std::max(maxd, std::fabs((double)hA1[i] - (double)hA2[i]));
    }
    std::printf("不同的元素 = %zu/%zu（%.2f%%），最大差 = %.6e\n",
                diff, (size_t)M * N, diff / (double)(M * N) * 100.0, maxd);
    std::printf("判定：%s\n", diff ? "**两次结果逐位不同 → 不可复现**"
                                  : "两次相同（该形状下未观察到差异）");

    std::printf("\n--- 路线 A（atomicAdd）vs 路线 B（固定顺序归约）---\n");
    size_t diffAB = 0; double maxdAB = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        if (hA1[i] != hB1[i]) ++diffAB;
        maxdAB = std::max(maxdAB, std::fabs((double)hA1[i] - (double)hB1[i]));
    }
    std::printf("不同的元素 = %zu/%zu，最大差 = %.6e\n", diffAB, (size_t)M * N, maxdAB);
    std::printf("路线 B 的可复现性来自「固定顺序」，代价是 workspace 的 %.1f MB 显存\n",
                (double)S * M * N * 4 / 1e6);

    // --- 与 fp64 参考比（口径要分开：相对量级 vs 相对元素）---
    if ((double)M * N * K <= 2.0e9) {
        float* hAf = (float*)std::malloc((size_t)M * K * 4);
        float* hBf = (float*)std::malloc((size_t)K * N * 4);
        float* hR = (float*)std::malloc((size_t)M * N * 4);
        CUDA_CHECK(cudaMemcpy(hAf, dA, (size_t)M * K * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hBf, dB, (size_t)K * N * 4, cudaMemcpyDeviceToHost));
        gemm_ref(hAf, hBf, hR, M, N, K);
        double scale = 0.0, ea = 0.0, eb = 0.0;
        for (size_t i = 0; i < (size_t)M * N; ++i) scale = std::max(scale, std::fabs((double)hR[i]));
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            ea = std::max(ea, std::fabs((double)hA1[i] - (double)hR[i]));
            eb = std::max(eb, std::fabs((double)hB1[i] - (double)hR[i]));
        }
        std::printf("\n--- 正确性（对 fp64 参考）---\n参考量级 max|C| = %.2f\n", scale);
        std::printf("路线 A 相对量级误差 = %.3e；路线 B = %.3e\n", ea / scale, eb / scale);
        std::free(hAf); std::free(hBf); std::free(hR);
    }

    std::printf("\n--- 本机墙钟（只对本机有效）---\n");
    std::printf("路线 A（atomicAdd，含 memset）= %.3f ms，%.3f TFLOPS\n",
                msA, flops / (msA * 1e-3) / 1e12);
    std::printf("路线 B（部分和写回 + 第二个 kernel）= %.3f ms，%.3f TFLOPS\n",
                msB, flops / (msB * 1e-3) / 1e12);
    std::printf("⚠ 本仓库没有跑过这些内核；上面两行只有在真机上跑过才有意义。\n");

    std::free(hA1); std::free(hA2); std::free(hB1);
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dW));
    return 0;
}
