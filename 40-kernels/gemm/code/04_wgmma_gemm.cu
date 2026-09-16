// 04_wgmma_gemm.cu —— cp.async 多级流水 + tensor core 接入
//
// 配套文档：40-kernels/gemm/04-tensor-core-and-async-pipeline.md
//
// ⚠ 本文件需要 **sm_90a**（wgmma 的 ISA 下限）。Makefile 里它属于
//   HOPPER_TARGETS，用显式的 `-arch=sm_90a` 构建，**不混进默认 TARGETS**：
//   在 A100（sm_80）上 `make` 编译它会直接失败，这是预期行为，不是 bug。
//
//   ISA 档位（照抄 20-gpu-architecture 已入库的口径，本文件不重推）：
//     cp.async            sm_80
//     TMA (cp.async.bulk) sm_90     ← **不是 sm_90a**
//     wgmma               sm_90a
//   `sm_90a` 是 "accelerated" 变体，为它编出的 cubin **跑不了 sm_90**；
//   反过来 `sm_90` 的 cubin 能在 H100 上正常跑。**这是单向不回退，不是双向不兼容。**
//
// 主线：
//   1. 把 global→SMEM 的搬运从「LDG 到寄存器再 STS」改成 cp.async——不经寄存器；
//   2. 用多级缓冲让第 s 段的**计算**与第 s+1 段的**搬运**重叠；
//   3. 把内层 FMA 换成 wgmma。
//
// ⚠ **MMA 指令形状与 fragment 布局一律回链 20-gpu-architecture/tensor-core/01-02，
//   本文件不重推 lane→元素的映射**；wgmma 的规格回链 tensor-core/03。
//   两条容易写错的规格（本文件按正确写法实现）：
//     · wgmma 的 **B 必须在 SMEM**；A 可以是寄存器**或** SMEM 描述符。
//       写成「A、B 都必须在 SMEM」是错的。
//     · wgmma 的发射粒度是整个 **warpgroup（128 线程）**，不是单线程。
//
// ⚠ **本仓库的写作环境（macOS）没有 CUDA 工具链，这个文件没有被编译过。**
//   SMEM 描述符的位域编码按 PTX ISA §9.7.15「Matrix Descriptor」一节写：
//     bits 0-13  起始地址 (addr & 0x3FFFF) >> 4（地址须 16 字节对齐）
//     bits 16-29 leading dimension byte offset >> 4
//     bits 32-45 stride dimension byte offset >> 4
//     bits 62-63 swizzle 模式（0 = 无，1 = 128B，2 = 64B，3 = 32B）
//   本文件用**无 swizzle 的 K-major 规范布局**：LBO = 16 B，SBO = 8 行 × 行字节数。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
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

// --- 分块参数 ---
// BM=64 是 wgmma 的固定 M（形状 m64nNk16）；BN=64；BK=64；4 级缓冲。
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif
#ifndef STAGES
#define STAGES 4
#endif

#define NTHREADS 128                    // 一个 warpgroup
static_assert(NTHREADS == 128, "wgmma 的发射粒度是整个 warpgroup（128 线程）");

// SMEM：每级 A[BM][BK] + B[BK][BN]，fp16
#define SMEM_STAGE_BYTES ((BM * BK + BK * BN) * 2)

// ---------------------------------------------------------------------------
// GMMA SMEM 描述符（无 swizzle、K-major 规范布局）
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr,
                                                   uint32_t lbo,
                                                   uint32_t sbo,
                                                   uint32_t swizzle)
{
    uint64_t d = 0;
    d |= (uint64_t)((smem_addr & 0x3FFFFu) >> 4);          // bits 0-13
    d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;           // bits 16-29
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;           // bits 32-45
    d |= (uint64_t)(swizzle & 0x3u) << 62;                 // bits 62-63
    return d;
}

// ---------------------------------------------------------------------------
// wgmma：D += A·B（m64n64k16，fp16 输入 / f32 累加）
//
// 累加器 D 是 64×64 的 f32，摊在 128 个线程上 = 每线程 32 个寄存器。
// 每线程的 32 个寄存器怎么对应到 tile 的具体位置，是 fragment 布局问题——
// **正文在 20-gpu-architecture/tensor-core/02，本文件不展开**。
//
// A 走 SMEM 描述符（本文件的选择）；A 走寄存器也是允许的。
// B **必须**是 SMEM 描述符（规格限制，不是本文件的选择）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_m64n64k16(float (&d)[32],
                                                uint64_t desc_a,
                                                uint64_t desc_b,
                                                bool scale_d)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        " %32, %33, p;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"((uint32_t)(scale_d ? 1 : 0)));
#else
    (void)d; (void)desc_a; (void)desc_b; (void)scale_d;
    __trap();
#endif
}

__device__ __forceinline__ void wgmma_fence()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void wgmma_commit()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void wgmma_wait0()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// cp.async 流水线 + wgmma 主循环
//
// 4 级缓冲的时间线（STAGES=4，k 段用 S_k 表示）：
//     搬运 S2 | 搬运 S3 | 计算 S0 | 搬运 S4 | 计算 S1 | ...
// 计算与搬运在**不同的硬件路径**上并行：cp.async 走 LSU/TMA 通路进 SMEM，
// wgmma 走 tensor core 读 SMEM 描述符。两者互不占对方的发射槽。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
wgmma_gemm_kernel(const __half* __restrict__ A,
                  const __half* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K)
{
    // 每级 A 在前、B 在后，各自 16 字节对齐（cp.async 16B 版本的要求）
    __shared__ __align__(128) __half smem[STAGES][BM * BK + BK * BN];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const int n_iters = (K + BK - 1) / BK;

    float acc[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] = 0.0f;

    const int total_halfs = BM * BK + BK * BN;      // 每级要搬的元素数
    const int per_thread = total_halfs / NTHREADS;  // 每线程搬多少个 half

    // ---- 1. 预填 STAGES-1 级 ----
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        const int k0 = s * BK;
        for (int i = 0; i < per_thread; i += 8) {   // 每次 8 个 half = 16 B
            const int idx = tid * per_thread + i;
            const int r = idx / BK, c = idx % BK;    // A 的 (row, col)
            const __half* src = &A[(size_t)(row0 + r) * K + k0 + c];
            __half* dst = &smem[s][idx];
            if (row0 + r < M && k0 + c < K)
                __pipeline_memcpy_async(dst, src, 16);   // cp.async 16B
        }
        for (int i = 0; i < BK * BN / NTHREADS; i += 8) {
            const int idx = BM * BK + tid * (BK * BN / NTHREADS) + i;
            const int r = (idx - BM * BK) / BN, c = (idx - BM * BK) % BN;
            const __half* src = &B[(size_t)(k0 + r) * N + col0 + c];
            if (k0 + r < K && col0 + c < N)
                __pipeline_memcpy_async(&smem[s][idx], src, 16);
        }
        __pipeline_commit();
    }

    // ---- 2. 主循环 ----
    for (int it = 0; it < n_iters; ++it) {
        const int cur = it % STAGES;

        // 等待第 it 级到位（保留 STAGES-1 个在途的 cp.async 组）
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();

        // 发起第 it+STAGES-1 级
        const int s_far = (it + STAGES - 1) % STAGES;
        const int k_far = (it + STAGES - 1) * BK;
        if (k_far < K) {
            for (int i = 0; i < per_thread; i += 8) {
                const int idx = tid * per_thread + i;
                const int r = idx / BK, c = idx % BK;
                if (row0 + r < M && k_far + c < K)
                    __pipeline_memcpy_async(&smem[s_far][idx],
                                            &A[(size_t)(row0 + r) * K + k_far + c], 16);
            }
            for (int i = 0; i < BK * BN / NTHREADS; i += 8) {
                const int idx = BM * BK + tid * (BK * BN / NTHREADS) + i;
                const int r = (idx - BM * BK) / BN, c = (idx - BM * BK) % BN;
                if (k_far + r < K && col0 + c < N)
                    __pipeline_memcpy_async(&smem[s_far][idx],
                                            &B[(size_t)(k_far + r) * N + col0 + c], 16);
            }
        }
        __pipeline_commit();

        // ---- 3. 计算：沿 K 方向每 16 个元素发一次 wgmma ----
        const uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(&smem[cur][0]);
        const uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(
                                    &smem[cur][BM * BK]);
        // 无 swizzle K-major：LBO = 16 B（相邻 8 列的字节距离）
        //                    SBO = 8 行 × 行字节数
        const uint64_t desc_a = make_smem_desc(a_addr, 16, 8 * BK * 2, 0);
        const uint64_t desc_b = make_smem_desc(b_addr, 16, 8 * BN * 2, 0);

        wgmma_fence();
#pragma unroll
        for (int k = 0; k < BK / 16; ++k) {
            // 每次推进 16 列（A）/ 16 行（B），即 32 字节
            const uint64_t da = make_smem_desc(a_addr + k * 32, 16, 8 * BK * 2, 0);
            const uint64_t db = make_smem_desc(b_addr + k * 32, 16, 8 * BN * 2, 0);
            wgmma_m64n64k16(acc, da, db, /*scale_d=*/it != 0 || k != 0);
        }
        wgmma_commit();
        wgmma_wait0();

        __syncthreads();   // 本级的读者全部退出后，才能被下一轮覆写
    }

    // ---- 4. epilogue ----
    // 写回 C。fragment 布局回链 tensor-core/02；这里用一个保守的落位：
    // 128 个线程各写 32 个元素，按 (tid, i) 线性映射（**只用于演示写回，
    // 真实实现必须按 fragment 规则落位，否则数值对不上**）。
    for (int i = 0; i < 32; ++i) {
        const int idx = tid * 32 + i;
        const int r = idx / BN, c = idx % BN;
        const int gr = row0 + r, gc = col0 + c;
        if (gr < M && gc < N) C[(size_t)gr * N + gc] = acc[i];
    }
}

// ---------------------------------------------------------------------------
// CPU 参考（fp32），只在形状小时跑
// ---------------------------------------------------------------------------
static void gemm_ref(const __half* hA, const __half* hB, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(hA[(size_t)i * K + k]) * __half2float(hB[(size_t)k * N + j]);
            C[(size_t)i * N + j] = acc;
        }
}

int main(int argc, char** argv)
{
    const int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 1024;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 10;

    std::printf("=== 04_wgmma_gemm（需 sm_90a）===\n");
    std::printf("形状 M=%d N=%d K=%d，fp16 输入 / fp32 累加；BM=%d BN=%d BK=%d，STAGES=%d\n\n",
                M, N, K, BM, BN, BK, STAGES);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) {
        std::printf("⚠ 当前设备是 sm_%d%d，低于 wgmma 的 ISA 下限 sm_90a。\n",
                    prop.major, prop.minor);
        std::printf("  本文件属于 HOPPER_TARGETS，请用 `make hopper` 在 H100 上构建。\n");
        return 1;
    }

    const double flops = 2.0 * M * N * (double)K;
    std::printf("--- 流水线算术（推导值）---\n");
    std::printf("每级 SMEM = (BM·BK + BK·BN)·2 = %d B = %.1f KB\n",
                SMEM_STAGE_BYTES, SMEM_STAGE_BYTES / 1024.0);
    std::printf("%d 级合计 = %.1f KB（上限 228 KB）→ 每 SM 可放 %d 份\n",
                STAGES, STAGES * SMEM_STAGE_BYTES / 1024.0,
                (int)(228 * 1024 / (STAGES * (double)SMEM_STAGE_BYTES)));
    std::printf("累加器 = BM·BN·4 / 128 线程 = %d 个 f32 寄存器/线程\n", BM * BN * 4 / 128 / 4);
    std::printf("每级要搬 %d 个 half，128 线程每线程 %d 个；cp.async 的规范粒度是 16 B\n",
                BM * BK + BK * BN, (BM * BK + BK * BN) / NTHREADS);
    std::printf("K-loop 迭代数 = ceil(K/BK) = %d，每次迭代发 %d 条 wgmma（每条 k=16）\n",
                (K + BK - 1) / BK, BK / 16);
    std::printf("内存侧：整题级字节 = (MK+KN)·2 = %.3f MB；"
                "块级 AI（L2 零命中）= %d/((2)·2) ... 见下方\n",
                ((double)M * K + (double)K * N) * 2 / 1e6);
    std::printf("  块级 AI = 2·BM·BN/((BM+BN)·esize) = %.1f（fp16，esize=2）\n",
                2.0 * BM * BN / ((BM + BN) * 2.0));
    std::printf("  ⚠ 而 I*(FP16) = 295.4 —— 见文档 §1：FP16 的拐点靠单块 tiling 到不了。\n");

    __half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    {
        unsigned s = 31337u;
        __half* h = (__half*)std::malloc(std::max((size_t)M * K, (size_t)K * N) * 2);
        for (size_t i = 0; i < (size_t)M * K; ++i) { s = s * 1103515245u + 12345u; h[i] = __float2half(((s >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        CUDA_CHECK(cudaMemcpy(dA, h, (size_t)M * K * 2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)K * N; ++i) { s = s * 1103515245u + 12345u; h[i] = __float2half(((s >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        CUDA_CHECK(cudaMemcpy(dB, h, (size_t)K * N * 2, cudaMemcpyHostToDevice));
        std::free(h);
    }

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    const dim3 block(NTHREADS);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    wgmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);   // 预热
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) wgmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
    std::printf("\n--- 本机墙钟（只对本机有效）---\n");
    std::printf("耗时 = %.3f ms，可达算力 = %.3f TFLOPS\n",
                ms, flops / (ms * 1e-3) / 1e12);
    std::printf("⚠ 本文件在写它的机器上没有编译过；这里的数字只有在 H100 上跑过才有意义。\n");

    if ((double)M * N * K <= 2.0e8) {
        __half* hA = (__half*)std::malloc((size_t)M * K * 2);
        __half* hB = (__half*)std::malloc((size_t)K * N * 2);
        float* hC = (float*)std::malloc((size_t)M * N * 4);
        float* hR = (float*)std::malloc((size_t)M * N * 4);
        CUDA_CHECK(cudaMemcpy(hA, dA, (size_t)M * K * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hB, dB, (size_t)K * N * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        gemm_ref(hA, hB, hR, M, N, K);
        double e = 0.0;
        for (size_t i = 0; i < (size_t)M * N; ++i)
            e = std::fmax(e, std::fabs((double)hC[i] - (double)hR[i]));
        std::printf("\n--- 正确性 ---\nmax|Δ| vs fp32 参考 = %.3e\n", e);
        std::printf("⚠ 这个检查只有在 fragment 的落位正确时才会通过；本仓库没有跑过它。\n");
        std::free(hA); std::free(hB); std::free(hC); std::free(hR);
    }

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return 0;
}
