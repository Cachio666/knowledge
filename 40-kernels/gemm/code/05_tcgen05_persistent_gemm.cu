// 05_tcgen05_persistent_gemm.cu —— persistent kernel + tcgen05 / TMEM
//
// 配套文档：40-kernels/gemm/05-persistent-kernel-and-tcgen05.md
//
// ⚠ 本文件需要 **sm_100a**（tcgen05 与 TMEM 的 ISA 下限）。Makefile 里它属于
//   BLACKWELL_TARGETS，用显式的 `-arch=sm_100a` 构建，**不混进默认 TARGETS**。
//   在 H100（sm_90a）上编它是编不过的——`sm_100a` 与 `sm_90a` 是两个互不覆盖的
//   "accelerated" 变体。
//
// 本文件两条主线：
//
// A. **persistent kernel 的调度结构**（这一部分是可移植的 C++，与架构无关）
//    网格大小固定为「能常驻的 block 数」，每个 block 用一个全局原子计数器
//    领下一个 tile，直到没有 tile 为止。它买到的是三件事：
//      · 每个 block 只付一次启动开销（而不是每个 tile 一次）；
//      · 调度器可以按**实际**速度发牌，而不是按块编号静态分牌；
//      · 尾部的空闲被压到「一个 tile」的量级，而不是「大半个波」。
//
// B. **tcgen05 的接法**（这一部分是 sm_100a 专用）
//    与 wgmma 的两处根本差异（回链 20-gpu-architecture/tensor-core/03-04）：
//      · **发射粒度是 1 个线程**，不是整个 warpgroup；
//      · **累加器 D 住在 TMEM**，不在寄存器文件里。
//    TMEM 的口径是 **per CTA 的 128 lane × 512 列 × 32 bit = 256 KB**——
//    **不是 per SM**（这条最易错；容量数值恰好与每 SM 的寄存器文件相同）。
//
// ⚠ 本仓库的写作环境（macOS）没有 CUDA 工具链，这个文件没有被编译过。
//   tcgen05 的指令描述符（instruction descriptor）位域按 PTX ISA §9.7.18
//   的表格写；TMEM 的分配/读取序列按同节给出的 `tcgen05.alloc` /
//   `tcgen05.mma` / `tcgen05.commit` / `tcgen05.ld` 顺序组织。
//   真机使用前请与 CUTLASS 的 SM100 实现交叉核对。

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

// --- tile 形状 ---
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 64
#endif

// ---------------------------------------------------------------------------
// A. persistent 调度器：用全局计数器把 tile 动态发牌
//
// 网格 = SMs × blocks_per_sm 个常驻 block（由 host 侧按占用率算出来）。
// 每个 block 循环：领号 → 算 → 再领号，直到号发完。
//
// 与「网格 = tile 总数」的静态分牌相比，差别在尾部：
//   静态：tile 数是机器的 7.76 倍时，最后一波只有 76% 的 SM 有活，
//         这 1/8 的时间里 24% 的算力空转。
//   动态：每个 SM 只在「手上这最后一个 tile 还没算完」时才有空转，
//         量级是 1 个 tile，而不是 1 个波。
// ⚠ 静态分牌的 24% 是**波内**的空转比例；动态分牌并没有让尾效应消失，
//   它只是把颗粒度从「波」降到「tile」。
// ---------------------------------------------------------------------------
__device__ unsigned int g_tile_counter = 0u;

struct TileCoord { int m; int n; };

__device__ __forceinline__ TileCoord claim_tile(int n_tiles_m, int n_tiles_n)
{
    const unsigned int id = atomicAdd(&g_tile_counter, 1u);
    // 每个 tile 的**代价相同**时，按行优先发牌即可；代价不均时（split-K、
    // ragged 的 M/N）要在这里改成按代价排序后再发。
    return TileCoord{(int)(id / (unsigned)n_tiles_n), (int)(id % (unsigned)n_tiles_n)};
}

// ---------------------------------------------------------------------------
// B. tcgen05 路径（sm_100a 专用）
// ---------------------------------------------------------------------------

// TMEM 分配：列数是动态分配的，粒度 32 列，合法的 n_cols 是 32/64/128/256/512。
// 一个 M=128、N 列的 f32 累加器占 N 列（128 个 lane 各对应一行）。
__device__ __forceinline__ uint32_t tmem_alloc_columns(int n_cols)
{
    uint32_t tmem_addr = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(&tmem_addr);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        : : "r"(smem_ptr), "r"(n_cols));
#endif
    return tmem_addr;
}

__device__ __forceinline__ void tmem_dealloc(int n_cols)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
        : : "r"(0u), "r"(n_cols));
#endif
}

// 指令描述符（PTX ISA §9.7.18 的表格）：把「精度组合 + 形状 + 转置」编进 32 位。
// 本文件编的是：a/b 为 f16、d 为 f32、无转置、M=BM、N=BN。
__device__ __forceinline__ uint32_t make_idesc_f16_f32(int M, int N)
{
    uint32_t d = 0;
    d |= (1u << 4);              // d format: 1 = f32（0 = f16）
    d |= (0u << 7);              // a format: 0 = f16
    d |= (0u << 10);             // b format: 0 = f16
    d |= (0u << 15);             // transpose a: 0
    d |= (0u << 16);             // transpose b: 0
    d |= ((uint32_t)(N >> 3) & 0x3Fu) << 17;
    d |= ((uint32_t)(M >> 4) & 0x1Fu) << 24;
    return d;
}

// SMEM 描述符的编码与 04 号文件同源（bits 0-13 地址 >> 4，16-29 LBO，32-45 SBO）。
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t addr, uint32_t lbo,
                                                   uint32_t sbo, uint32_t swz)
{
    uint64_t x = 0;
    x |= (uint64_t)((addr & 0x3FFFFu) >> 4);
    x |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    x |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
    x |= (uint64_t)(swz & 0x3u) << 62;
    return x;
}

// tcgen05.mma：**由单个线程发射**（PTX 原文："An issue from a single thread in
// the current CTA would initiate the base operation."）。
// D 必须在 TMEM；A 可在 TMEM 或 SMEM；B **必须**在 SMEM。
__device__ __forceinline__ void tcgen05_mma_f16(uint32_t d_tmem, uint64_t a_desc,
                                                uint64_t b_desc, uint32_t idesc,
                                                bool enable_input_d)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        : : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
            "r"((uint32_t)(enable_input_d ? 1 : 0)));
#endif
}

__device__ __forceinline__ void tcgen05_commit()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
                 : : "r"(0u));
#endif
}

// 从 TMEM 读回寄存器：tcgen05.ld 一次读 32 个 lane × N 列，**必须由整个
// warpgroup 协作**（TMEM 的 128 个 lane 与 warpgroup 的 4 个 warp 一一对应；
// 单个 warp 只能访问自己那 32 个 lane——这条口径回链 tensor-core/04）。
__device__ __forceinline__ void tcgen05_ld_f32(float (&dst)[32], uint32_t tmem_addr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7]),
          "=f"(dst[8]), "=f"(dst[9]), "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
#else
    (void)dst; (void)tmem_addr;
#endif
}

__device__ __forceinline__ void tcgen05_wait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// persistent + tcgen05 内核
//
// 每个 block：
//   1. 一次性分配 TMEM（BN 列向上取整到 32 的倍数）
//   2. 领 tile → 沿 K 推进（SMEM 多级缓冲省略，见 04 号）→ 把 TMEM 读回寄存器
//      → 写回 C
//   3. 直到领不到 tile，再释放 TMEM
//
// ⚠ 第 2 步里的 SMEM 装载被省略成一行注释：它和 04 号文件是同一件事，
//   本文件聚焦的是**调度**与**TMEM**，不重复流水线的实现。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
persistent_tcgen05_gemm_kernel(const __half* __restrict__ A,
                               const __half* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K)
{
    const int n_tiles_m = (M + BM - 1) / BM;
    const int n_tiles_n = (N + BN - 1) / BN;

    // --- TMEM：按列分配，粒度 32 列 ---
    const int tmem_cols = ((BN + 31) / 32) * 32;
    const uint32_t tmem_d = tmem_alloc_columns(tmem_cols);

    const uint32_t idesc = make_idesc_f16_f32(BM, BN);

    for (;;) {
        const TileCoord t = claim_tile(n_tiles_m, n_tiles_n);
        if (t.m >= n_tiles_m) break;

        const int row0 = t.m * BM;
        const int col0 = t.n * BN;

        // 沿 K 推进。每段：
        //   （装载 SMEM：与 04 号相同，省略）
        //   由 **一个线程** 发射 tcgen05.mma，D 指向 TMEM
        const uint32_t a_addr = 0, b_addr = 0;   // ← 占位：真实实现里来自 SMEM
        bool first = true;
        for (int k0 = 0; k0 < K; k0 += BK) {
            const uint64_t da = make_smem_desc(a_addr, 16, 8 * BK * 2, 0);
            const uint64_t db = make_smem_desc(b_addr, 16, 8 * BN * 2, 0);
            if (threadIdx.x == 0) {                    // ← **单线程发射**
                tcgen05_mma_f16(tmem_d, da, db, idesc, /*enable_input_d=*/!first);
            }
            first = false;
        }
        if (threadIdx.x == 0) tcgen05_commit();

        // 读回并写回 C（整个 warpgroup 协作）
        float frag[32];
        tcgen05_ld_f32(frag, tmem_d);
        tcgen05_wait();
        // fragment 的落位规则回链 tensor-core/02；此处按线性占位写回。
        for (int i = 0; i < 32; ++i) {
            const int idx = threadIdx.x * 32 + i;
            const int r = idx / BN, c = idx % BN;
            if (row0 + r < M && col0 + c < N)
                C[(size_t)(row0 + r) * N + col0 + c] = frag[i];
        }
        (void)A; (void)B;
    }

    tmem_dealloc(tmem_cols);
}

int main(int argc, char** argv)
{
    const int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const int K = (argc > 3) ? std::atoi(argv[3]) : 4096;

    std::printf("=== 05_tcgen05_persistent_gemm（需 sm_100a）===\n");
    std::printf("形状 M=%d N=%d K=%d；BM=%d BN=%d BK=%d\n\n", M, N, K, BM, BN, BK);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 10) {
        std::printf("⚠ 当前设备是 sm_%d%d，低于 tcgen05 的 ISA 下限 sm_100a。\n",
                    prop.major, prop.minor);
        std::printf("  本文件属于 BLACKWELL_TARGETS，请用 `make blackwell` 在 B200 上构建。\n");
        return 1;
    }

    // --- 静态分牌 vs 动态发牌的波次算术（推导值）---
    const int tiles = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
    const int sm = prop.multiProcessorCount;
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, persistent_tcgen05_gemm_kernel, 128, 0));
    const int slots = sm * std::max(1, blocks_per_sm);
    std::printf("--- 调度算术（推导值）---\n");
    std::printf("tile 总数 = %d，机器槽位 = %d SM × %d block/SM = %d\n",
                tiles, sm, blocks_per_sm, slots);
    std::printf("静态分牌的波数 = ceil(%d/%d) = %d，最后一波的占用 = %.1f%%\n",
                tiles, slots, (tiles + slots - 1) / slots,
                (tiles % slots ? (tiles % slots) : slots) / (double)slots * 100.0);
    std::printf("  ⟹ 最后一波里 %.1f%% 的槽位在空转，占总时间的 %.2f%%\n",
                (1.0 - (tiles % slots ? (tiles % slots) : slots) / (double)slots) * 100.0,
                (1.0 - (tiles % slots ? (tiles % slots) : slots) / (double)slots) /
                ((tiles + slots - 1) / slots) * 100.0);
    std::printf("动态发牌把尾部压到「1 个 tile」的量级 = 总时间的 %.2f%%\n",
                1.0 / tiles * 100.0);

    // --- TMEM 容量对 CTA 并发的约束（推导值）---
    const int tmem_cols = ((BN + 31) / 32) * 32;
    std::printf("\n--- TMEM 容量约束（推导值）---\n");
    std::printf("TMEM = %d lane × %d 列 × 32 bit = %d KB（**per CTA 口径**，不是 per SM）\n",
                128, 512, 128 * 512 * 4 / 1024);
    std::printf("一个 %d×%d 的 f32 累加器占 %d 列（向上取整到 32 的倍数）\n",
                BM, BN, tmem_cols);
    std::printf("→ 每 CTA 的 512 列最多容纳 %d 个这样的累加器\n", 512 / tmem_cols);
    std::printf("→ 若每个 CTA 只用一个累加器，TMEM 允许每 SM 最多 %d 个 CTA 在跑\n",
                512 / tmem_cols);
    std::printf("⚠ 但每条 tcgen05.alloc 都是 per CTA 的，一个 SM 上多个 CTA 如何\n");
    std::printf("  仲裁 TMEM，本次采用的来源没有说明——这里只做 per CTA 的算术，\n");
    std::printf("  不把它当成 per SM 的硬上限。\n");

    // --- 启动：网格 = 常驻槽位数（persistent）---
    __half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    {
        unsigned s = 2718u;
        __half* h = (__half*)std::malloc(std::max((size_t)M * K, (size_t)K * N) * 2);
        for (size_t i = 0; i < (size_t)M * K; ++i) { s = s * 1103515245u + 12345u; h[i] = __float2half(((s >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        CUDA_CHECK(cudaMemcpy(dA, h, (size_t)M * K * 2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)K * N; ++i) { s = s * 1103515245u + 12345u; h[i] = __float2half(((s >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
        CUDA_CHECK(cudaMemcpy(dB, h, (size_t)K * N * 2, cudaMemcpyHostToDevice));
        std::free(h);
    }

    unsigned int zero = 0u;
    CUDA_CHECK(cudaMemcpyToSymbol(g_tile_counter, &zero, sizeof(zero)));
    persistent_tcgen05_gemm_kernel<<<slots, 128>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("\n--- 启动配置 ---\n网格 = %d 个 block（= 常驻槽位数），每 block 128 线程\n", slots);
    std::printf("⚠ 本文件在写它的机器上没有编译过，也没有跑过。\n");

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return 0;
}
