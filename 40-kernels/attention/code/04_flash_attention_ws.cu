// 04_flash_attention_ws.cu —— warp specialization / pingpong 与变体（**sm_90a**）
//
// 配套文档：40-kernels/attention/04-warp-specialization-and-variants.md
//
// ⚠ **本文件需要 `-arch=sm_90a`**（wgmma 是 "accelerated" 变体，为它编出的 cubin
//   跑不了 sm_90，反之可以）。它进 Makefile 的 `HOPPER_TARGETS`，
//   **不混进默认 TARGETS**，否则在 A100 上 `make` 直接失败。
//   wgmma 的规格（M 恒为 64、N ∈ [8,256] 步长 8、B 必须来自 SMEM、
//   `.aligned` 要求整个 warpgroup 一起发射）**正文在
//   20-gpu-architecture/tensor-core/03-wgmma-and-pipeline，本文只回链、不重推**。
//
// 本文件回答的问题：**为什么要把 warp 分成生产者与消费者**，以及变体怎么改结构。
//
//   ┌── producer warpgroup（128 线程）──┐   搬运 K/V tile → SMEM（TMA / cp.async）
//   │  只做 TMA + mbarrier 的 arrive    │   不参与任何计算
//   └───────────────────────────────────┘
//   ┌── consumer warpgroup（128 线程）──┐   QKᵀ（wgmma）→ 在线 softmax（FP32 向量）
//   │  wgmma + softmax + PV（wgmma）   │   → PV（wgmma）→ 重标定 O
//   └───────────────────────────────────┘
//
// **硬边界**：wgmma 的发射粒度是整个 warpgroup（规格值
// WGMMA_ALIGNED_QUALIFIER_MANDATORY）。所以分工出去做访存的 warp **不能**与
// 发 wgmma 的 warp 混在同一个 warpgroup 里——这不是风格问题，是 ISA 要求。
//
// 变体只在**结构**上动刀子，本文不重推算法定义：
//   - sliding window：KV 的**分块范围**变窄（ceil(w/Bc) 个块而不是 S/Bc 个，
//     verify.py §B11：S=65536、w=1024、Bc=128 时 512 → 8，64×）；
//   - ALiBi / softcap：在**算完 S 之后**给 logits 加一项 / 过一次 tanh，
//     分块的形状完全不变，只多一条逐元素指令。
//
// 编译与运行（需 nvcc + H100；本仓库的写作环境没有工具链，未编译过）：
//     make hopper                        # = nvcc -arch=sm_90a 04_flash_attention_ws.cu
//     ./04_flash_attention_ws 4096 128

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#error "本文件需要 sm_90a：wgmma 的 m64nNk16 形状不在更低架构上"
#endif

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

constexpr int WARP = 32;
constexpr int WG_WARPS = 4;                  // 一个 warpgroup = 4 个连续 warp
constexpr int WG_THREADS = WG_WARPS * WARP;  // 128

// ---------------------------------------------------------------------------
// 生产者：只负责把 K/V tile 搬进 SMEM
//
// 真实实现用 TMA（`cp.async.bulk.tensor`，**规格下限是 sm_90 而不是 sm_90a**）
// 与 mbarrier；这里用 cp.async 的等价写法示意，并把「谁来发指令」标清楚。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void producer_stage(const half* __restrict__ src,
                                               half* __restrict__ dst,
                                               int n_elems, int tid)
{
    // 一个线程就能发起整块搬运（规格值：TMA_SINGLE_THREAD_ISSUE），
    // 这里保留每线程分片的写法只是为了在 sm_80 上也能看懂对照部分。
    for (int i = tid; i < n_elems; i += WG_THREADS) dst[i] = src[i];
    // 真实实现：
    //   if (tid == 0) cp_async_bulk_tensor_2d_global_to_shared(dst, desc, x, y, bar);
    //   mbarrier_arrive_expect_tx(bar, n_bytes);   // 消费者 wait 这个 barrier
}

// ---------------------------------------------------------------------------
// 消费者：wgmma 发 QKᵀ → 在线 softmax → wgmma 发 PV → 重标定 O
//
// **O 的状态重标定（本文的主体）在这里**：α = exp(m_old − m_new) 同时乘到 l 与 O。
// 与 02 篇的区别只是：这里的 QKᵀ 与 PV 走 wgmma（一次发射覆盖 m64 的整块），
// 而那个「一行一个线程」的标量循环换成了一个 warpgroup 协作的 fragment 布局。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void consumer_softmax_update(float* m, float* l, float* acc,
                                                        const float* s_tile, int nq, int nc,
                                                        int d, int lanes_per_row)
{
    // s_tile 是 QKᵀ 的结果 fragment，布局见 tensor-core/02-fragment-layout。
    // 这里只写出重标定那一步——它是本文的正文，也不依赖具体布局。
    for (int r = 0; r < nq; ++r) {
        float m_new = m[r];
        for (int j = 0; j < nc; ++j) m_new = fmaxf(m_new, s_tile[r * nc + j]);
        const float alpha = (m[r] == -INFINITY) ? 0.0f : expf(m[r] - m_new);
        l[r] = l[r] * alpha;
        for (int c = 0; c < d; ++c) acc[r * d + c] *= alpha;      // ← O 也乘同一个 α
        for (int j = 0; j < nc; ++j)
            l[r] += expf(s_tile[r * nc + j] - m_new);             // l 的累加
        // P·V 那一步留在调用者里（它要发 wgmma），这里只管状态更新
        m[r] = m_new;
    }
    (void)lanes_per_row;
}

// ---------------------------------------------------------------------------
// 主内核（骨架）：双 warpgroup，流水 stage 级
//
// 与实际性能强相关的是两个数（verify.py §B7，推导值）：
//   SMEM = stages·(K tile + V tile)·esize = stages·2·Bc·d·esize
//        Bc=128、d=128、fp16：2 级 = 128 KB，3 级 = 192 KB，**4 级 = 256 KB > 228 KB**；
//   O 累加器 = Br·d 个 fp32 ÷ 128 线程 = Br=128、d=128 时 128 个寄存器/线程。
// ---------------------------------------------------------------------------
__global__ void flash_ws_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                                const half* __restrict__ V, half* __restrict__ O,
                                int seq, int d, float scale, int Br, int Bc, int stages,
                                int window, int use_alibi)
{
    extern __shared__ __align__(128) half smem[];
    // smem 按 stage 切成 [stages][2][Bc*d]：偶数片放 K，奇数片放 V
    const int tid = threadIdx.x;
    const int wg = tid / WG_THREADS;             // 0 = producer，1 = consumer
    const int lane = tid % WG_THREADS;

    const int q0 = blockIdx.x * Br;
    float m[Br], l[Br], acc[Br * 128];           // O 累加器：Br×d 个 fp32
    for (int r = 0; r < Br; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }
    for (int i = 0; i < Br * d; ++i) acc[i] = 0.0f;

    // ---- 变体：sliding window 只把 KV 的**范围**截短 ----
    int j_start = 0, j_end = seq;
    if (window > 0) {
        // 因果掩码下第 q0..q0+Br 行只需要 [q0+Br−1−window, q0+Br−1] 这一段
        j_start = (q0 + Br - 1 - window) / Bc * Bc;
        j_start = max(0, j_start);
        j_end   = min(seq, q0 + Br);
    }

    for (int j0 = j_start; j0 < j_end; j0 += Bc) {
        const int nc = min(Bc, j_end - j0);
        const int stage = ((j0 - j_start) / Bc) % stages;      // 环形缓冲

        if (wg == 0) {                                          // ---- 生产者 ----
            half* sK = smem + (size_t)stage * 2 * Bc * d;
            half* sV = sK + (size_t)Bc * d;
            producer_stage(K + (size_t)j0 * d, sK, nc * d, lane);
            producer_stage(V + (size_t)j0 * d, sV, nc * d, lane);
            // 真实实现：mbarrier_arrive + 消费者侧 mbarrier_wait
        } else {                                                // ---- 消费者 ----
            half* sK = smem + (size_t)stage * 2 * Bc * d;
            half* sV = sK + (size_t)Bc * d;
            float s_tile[Br * 128];
            // ① wgmma.m64nNk16：A = Q（寄存器或 SMEM），B = K（**必须 SMEM**）
            //    一次发射覆盖 M=64 行；Br=128、Bc=128、d=128 时需要 16 条
            for (int r = 0; r < Br; ++r)
                for (int j = 0; j < nc; ++j) {
                    float t = 0.0f;
                    for (int c = 0; c < d; ++c)
                        t += __half2float(Q[(size_t)(q0 + r) * d + c]) *
                             __half2float(sK[(size_t)j * d + c]);
                    s_tile[r * nc + j] = t * scale;
                    if (use_alibi) {
                        // ALiBi：**算完 S 之后**加一项，不改分块形状
                        s_tile[r * nc + j] -= 0.1f * (float)(q0 + r - (j0 + j));
                    }
                }
            // ② 在线 softmax + **O 的重标定**（本文的正文）
            consumer_softmax_update(m, l, acc, s_tile, Br, nc, d, 1);
            // ③ wgmma.m64n128k16：A = P（寄存器 fragment），B = V（必须 SMEM）
            for (int r = 0; r < Br; ++r)
                for (int c = 0; c < d; ++c) {
                    float t = 0.0f;
                    for (int j = 0; j < nc; ++j)
                        t += expf(s_tile[r * nc + j] - m[r]) * __half2float(sV[(size_t)j * d + c]);
                    acc[r * d + c] += t;             // ← 上面 consumer_softmax_update 里已乘 α
                }
        }
        __syncthreads();       // 真实实现用 mbarrier，不整块同步
    }

    if (wg == 1)
        for (int r = lane; r < Br; r += WG_THREADS) {
            const float inv = 1.0f / l[r];
            for (int c = 0; c < d; ++c)
                O[(size_t)(q0 + r) * d + c] = __float2half(acc[r * d + c] * inv);
        }
}

int main(int argc, char** argv)
{
    const int seq = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int d   = (argc > 2) ? std::atoi(argv[2]) : 128;
    const float scale = 1.0f / std::sqrt((float)d);

    std::printf("=== 04_flash_attention_ws（**sm_90a**，wgmma + warp specialization）===\n");
    std::printf("    S=%d d=%d\n\n", seq, d);

    std::printf("[1] 规格（**回链，不重推**）：\n");
    std::printf("    wgmma 的 M 恒为 64；N ∈ [8, 256] 步长 8；B 必须来自 SMEM，\n");
    std::printf("    A 可以在寄存器里；`.aligned` 要求整个 warpgroup 一起发射。\n");
    std::printf("    正文：20-gpu-architecture/tensor-core/03-wgmma-and-pipeline。\n\n");

    std::printf("[2] 分工与代价（**推导值**，verify.py §B7）：\n");
    std::printf("    producer 128 线程 + consumer 128 线程 = 256 线程/CTA\n");
    std::printf("    SMEM = stages·2·Bc·d·esize：\n");
    for (int Bc : {64, 128}) {
        for (int st : {2, 3, 4}) {
            const double kb = (double)st * 2 * Bc * d * 2 / 1024.0;
            std::printf("      Bc=%3d stages=%d → %6.0f KB %s\n", Bc, st, kb,
                        kb <= 228.0 ? "（可行）" : "（**超 228 KB 上限**）");
        }
    }
    std::printf("    wgmma 指令数（Br=128, Bc=128, d=128）：QKᵀ 16 条、PV 16 条（每 tile）\n");
    std::printf("    O 累加器：Br=128、d=128 → 16384 个 fp32 ÷ 128 线程 = 128 个寄存器/线程\n");
    std::printf("    ⟹ Br 一动，寄存器就跟着动；Bc 只在 SMEM 那一侧起作用。\n\n");

    std::printf("[3] 变体只改结构（**推导值**，verify.py §B11）：\n");
    const int Bc = 128;
    for (int w : {4096, 1024}) {
        std::printf("    sliding window w=%d：每行只看 ceil(w/Bc)=%d 个 KV 块，"
                    "全量是 ceil(S/Bc)=%d 个（%.0f× 少）\n",
                    w, (w + Bc - 1) / Bc, (seq + Bc - 1) / Bc,
                    (double)((seq + Bc - 1) / Bc) / ((w + Bc - 1) / Bc));
    }
    std::printf("    ALiBi / softcap：分块形状完全不变，只在 S 算完之后多一条逐元素指令\n");
    std::printf("      （ALiBi 加一个与 (i−j) 成正比的项；softcap 过一次 tanh）。\n\n");

    std::printf("[4] 为什么必须把 warp 分开（本文件的核心）：\n");
    std::printf("    wgmma 的发射粒度是**整个 warpgroup**（4 个连续 warp 必须执行同一条）。\n");
    std::printf("    若让同一个 warpgroup 一半去搬数据、一半发 wgmma，就是未定义行为——\n");
    std::printf("    这不是「慢一点」，是**写错了**。producer/consumer 的分工是 ISA 逼出来的。\n");
    std::printf("    TMA 那一侧正好相反：**一个线程就能发起整块搬运**，所以搬运可以只占\n");
    std::printf("    很少的发射槽，把这个 warpgroup 的寄存器让给 O 累加器。\n\n");

    std::printf("⚠ 本仓库的写作环境没有 CUDA 工具链也没有 GPU：上面没有任何**运行结果**，\n");
    std::printf("   全部数字都是推导值或规格值。`make verify` 的 §B7 可复核这些算式。\n");
    return 0;
}
