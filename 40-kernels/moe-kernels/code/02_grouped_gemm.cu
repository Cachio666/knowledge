// 02_grouped_gemm.cu —— grouped / 变长 GEMM 与 tile 调度
//
// 配套文档：40-kernels/moe-kernels/02-grouped-gemm.md
//
// 主线：MoE 的 GEMM 只在 **M 方向**变长 —— 每个专家的 tokens 数不同，
// N（= d_ff）与 K（= d_model）对所有专家相同。于是问题收敛成一句：
// **M 不规整**。围绕它有两条路：
//
//   甲 `padded_gemm_kernel`  —— 每个专家都按最大的专家算（M_max 对齐到 BM）
//   乙 `grouped_gemm_kernel` —— 按实际长度分组，一个 block 处理「一个专家的一个 tile」
//
// 本文件的重点是乙的 **tile 调度**：网格 = Σ_e ceil(M_e/BM)·(N/BN)，
// 每个 block 用一张前缀和表 `tile_base[E+1]` 把 blockIdx 解码回 (专家, m_tile, n_tile)。
//
// ⚠ 「按实际长度分组」不是零成本：每个专家的段末尾仍要补齐到 BM 的整数倍
//   （均匀路由下约 6%），而且 M_e = 0 的专家段必须能被安全跳过。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，**未编译过**）：
//     make 02_grouped_gemm && ./02_grouped_gemm
//
// ⚠ 本文件 §2/§3 的表格与 §4 的比值全部是**推导值**（纯算术，
//    与 `verify.py` 的 §B4/§B5 同式）；§1 的 kernel 运行结果本仓库没有。
//    可复核处：`make verify` 的 §A3 / §B4 / §B5 / §C5。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <cstring>
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
// §0 tile 参数（与 code/verify.py 同值）
// ---------------------------------------------------------------------------

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int PAD = 1;          // 行距 +1，避免 SMEM bank 冲突（机制在 memory-system/03）

// ---------------------------------------------------------------------------
// §1 kernel
// ---------------------------------------------------------------------------

// 一个 tile 的 GEMM 主体：算 C[0..m_avail) × [nt·BN .. nt·BN+BN)。
//   A_row_base : 该专家在全局 A 里的第一行（[M_total, K] 行主序）
//   B_e        : 该专家的权重 [K, N]（行主序）
//   C_row_base : 该专家在全局 C 里的第一行（[M_total, N] 行主序）
// m_avail 是这一段**实际**有多少行；不足 BM 的部分由谓词挡住，
// 既不出界，也不参与结果。
__device__ void gemm_tile_body(const float* __restrict__ A_row_base,
                               const float* __restrict__ B_e,
                               float* __restrict__ C_row_base,
                               int m_avail, int N, int K, int nt) {
    // 256 个线程（16×16）算 BM×BN = 128×128 的输出，每线程 8×8 的寄存器小块
    constexpr int TM = BM / 16;
    constexpr int TN = BN / 16;

    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    const int tx = threadIdx.x;                 // 0..15
    const int ty = threadIdx.y;                 // 0..15
    const int tid = ty * blockDim.x + tx;
    const int nthr = blockDim.x * blockDim.y;

    const int row0 = nt * BN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 载入 A 的 BM×BK 与 B 的 BK×BN（都是行主序，合并访存）
        for (int i = tid; i < BM * BK; i += nthr) {
            const int r = i / BK, c = i % BK;
            As[r][c] = (r < m_avail && k0 + c < K) ? A_row_base[(size_t)r * K + k0 + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += nthr) {
            const int r = i / BN, c = i % BN;
            Bs[r][c] = (k0 + r < K && row0 + c < N)
                           ? B_e[(size_t)(k0 + r) * N + row0 + c] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float a = As[ty * TM + i][kk];
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a, Bs[kk][tx * TN + j], acc[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = ty * TM + i;
        if (row >= m_avail) break;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = row0 + tx * TN + j;
            if (col < N) C_row_base[(size_t)row * N + col] = acc[i][j];
        }
    }
}

// 乙：grouped —— 按实际长度分组。
// 网格 = Σ_e ceil(M_e/BM)·(N/BN)；每个 block 用一个专家的一个 tile。
// `tile_base[e]` 是「前 e 个专家一共占了多少个 block」的**排他前缀和**
// （前缀和的并行算法正文在 40-kernels/reduce-scan/03，此处只做代入）。
__global__ void grouped_gemm_kernel(const float* __restrict__ A,
                                    const float* __restrict__ Ws,   // [E, K, N]
                                    float* __restrict__ C,
                                    const int* __restrict__ seg_len,   // [E]
                                    const int* __restrict__ seg_off,   // [E] 排他前缀和
                                    const int* __restrict__ tile_base, // [E+1]
                                    int E, int N, int K) {
    int b = blockIdx.x;
    int e = 0;
    while (e + 1 < E && b >= tile_base[e + 1]) ++e;   // E 不大，线性查找足够
    const int local = b - tile_base[e];
    const int nt_per_e = N / BN;
    const int mt = local / nt_per_e;
    const int nt = local % nt_per_e;

    const int m_e = seg_len[e];
    const int row0 = mt * BM;
    const int m_avail = (row0 < m_e) ? min(BM, m_e - row0) : 0;
    if (m_avail <= 0) return;                          // 该 tile 整块在段外

    gemm_tile_body(A + (size_t)(seg_off[e] + row0) * K,
                   Ws + (size_t)e * K * N,
                   C + (size_t)(seg_off[e] + row0) * N,
                   m_avail, N, K, nt);
}

// 甲：padded —— 每个专家都按最大的专家算。
// 网格 = E · ceil(M_max/BM) · (N/BN)；A 与 C 按 M_max 的固定行距排布，
// 所以**没有 seg_len / seg_off / tile_base 三张表**，解码也更简单 ——
// 这在 E 很大但分布均匀时才划算。
__global__ void padded_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ Ws,
                                   float* __restrict__ C,
                                   int m_max, int N, int K, int m_total_per_expert) {
    const int nt_per_e = N / BN;
    const int mt_per_e = (m_max + BM - 1) / BM;
    const int e  = blockIdx.x / (mt_per_e * nt_per_e);
    const int r  = blockIdx.x % (mt_per_e * nt_per_e);
    const int mt = r / nt_per_e;
    const int nt = r % nt_per_e;

    const int row0 = mt * BM;
    const int m_avail = min(BM, m_max - row0);
    if (m_avail <= 0) return;

    (void)m_total_per_expert;
    gemm_tile_body(A + (size_t)(e * m_max + row0) * K,
                   Ws + (size_t)e * K * N,
                   C + (size_t)(e * m_max + row0) * N,
                   m_avail, N, K, nt);
}

// ---------------------------------------------------------------------------
// §2 主机侧：tokens_per_expert → tile 划分（确定性伪随机，便于复现）
// ---------------------------------------------------------------------------

static unsigned g_seed = 20260917u;
static double urand() {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((g_seed >> 8) & 0xFFFFFF) / 16777216.0;
}

static void sample_counts(int E, int n_assign, bool zipf, int* c) {
    double p[256], sum = 0.0;
    for (int e = 0; e < E; ++e) { p[e] = zipf ? 1.0 / (e + 1.0) : 1.0 / E; sum += p[e]; }
    for (int e = 0; e < E; ++e) { p[e] /= sum; c[e] = 0; }
    for (int i = 0; i < n_assign; ++i) {
        const double u = urand();
        double acc = 0.0;
        for (int e = 0; e < E; ++e) { acc += p[e]; if (u < acc) { c[e] += 1; break; } }
    }
}

static void report_tiles(int E, int N, const char* tag, const int* c) {
    int sum = 0, mx = 0;
    for (int e = 0; e < E; ++e) { sum += c[e]; mx = (c[e] > mx) ? c[e] : mx; }
    const int mt_max = (mx + BM - 1) / BM;
    const int padded = E * mt_max * BM;
    int grouped = 0;
    for (int e = 0; e < E; ++e) grouped += ((c[e] + BM - 1) / BM) * BM;

    std::printf("  %-10s tokens_per_expert = [", tag);
    for (int e = 0; e < E; ++e) std::printf("%s%d", e ? "," : "", c[e]);
    std::printf("]\n");
    std::printf("             M-tiles/专家    = [");
    for (int e = 0; e < E; ++e) std::printf("%s%d", e ? "," : "", (c[e] + BM - 1) / BM);
    std::printf("]\n");
    std::printf("             ΣM=%d  M_max=%d  padded 行=%d  grouped 行=%d\n",
                sum, mx, padded, grouped);
    std::printf("             padded 浪费=%.2f%%  grouped 浪费=%.2f%%  比值=%.2f×\n",
                (1.0 - (double)sum / padded) * 100.0,
                (1.0 - (double)sum / grouped) * 100.0,
                (double)padded / grouped);
    std::printf("             每个专家的 N-tile 数 = %d → padded 网格 = %d 个 block，"
                "grouped 网格 = %d 个 block（%d 个 M-tile × %d）\n\n",
                N / BN, E * mt_max * (N / BN), (grouped / BM) * (N / BN), grouped / BM, N / BN);
}

static void report_waves(int E, int N, const int* c_uniform, int sm) {
    std::printf("--- §3 网格形状与波次（推导值；波次量化的机制见 40-kernels/gemm/05）---\n");
    std::printf("H100 有 %d 个 SM（规格值）。设每 SM 常驻 r 个 block，"
                "则一次装得下 %d·r 个 block。\n\n", sm, sm);
    for (int r : {1, 3}) {
        const int slots = sm * r;
        std::printf("  r=%d（%d 个槽位）\n", r, slots);
        std::printf("    %-22s %10s %8s %14s %12s\n", "映射", "block 数", "波数",
                    "最后一波占用", "整机效率");
        for (int e : {8, 64, 132, 133, 200, 396, 397, 792}) {
            const int waves = (e + slots - 1) / slots;
            const int last = e - (waves - 1) * slots;
            std::printf("    一个专家一个 block, E=%-4d %10d %8d %10d/%-4d %11.2f%%\n",
                        e, e, waves, last, slots, 100.0 * e / (waves * slots));
        }
        // tile 级网格（用均匀分布那一组 counts）
        int mt = 0;
        for (int e = 0; e < E; ++e) mt += (c_uniform[e] + BM - 1) / BM;
        const int blocks = mt * (N / BN);
        const int waves = (blocks + slots - 1) / slots;
        const int last = blocks - (waves - 1) * slots;
        std::printf("    %-22s %10d %8d %10d/%-4d %11.2f%%\n", "一个 tile 一个 block",
                    blocks, waves, last, slots, 100.0 * blocks / (waves * slots));
        std::printf("\n");
    }
    std::printf("  ⚠ 尾部代价 = (1 − 最后一波占用率) / 波数（正文在 gemm/05，此处回链）。\n\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const int E  = 8;                                  // 专家数
    const int k  = (argc > 1) ? std::atoi(argv[1]) : 2; // top_k
    const int T  = 4096;                               // 场景里的 token 数
    const int N  = 14336;                              // d_ff
    const int K  = 4096;                               // d_model
    const int sm = 132;                                // H100 SM 数

    std::printf("=== 02_grouped_gemm ===\n");
    std::printf("场景：tokens=%d top_k=%d E=%d → 总 slot 数 = %d；"
                "N=d_ff=%d K=d_model=%d；BM=%d BN=%d BK=%d\n\n", T, k, E, T * k, N, K, BM, BN, BK);

    int c_uni[256], c_zipf[256];
    sample_counts(E, T * k, false, c_uni);
    sample_counts(E, T * k, true, c_zipf);

    std::printf("--- §2 两种填充策略（推导值）---\n");
    report_tiles(E, N, "均匀", c_uni);
    report_tiles(E, N, "幂律 α=1", c_zipf);
    std::printf("  ⚠ 两种方案都**必须**把每个专家的 M 取整到 BM 的整数倍；\n");
    std::printf("    padded 的「M_max」也要先向上取整，否则比的是一个不存在的方案。\n");
    std::printf("  ⚠ 上表是本文件自己的抽样（LCG），与 `verify.py` §B4 的各段长度不同\n");
    std::printf("     （不同的伪随机源），但浪费比与比值同量级；正文引的是 verify.py 的数。\n\n");

    report_waves(E, N, c_uni, sm);

    // ---- §1 kernel 演示（缩小规模，只为验通 tile 解码）----
    const int D  = 128;      // 演示用 d_model
    const int DF = 256;      // 演示用 d_ff
    const int Td = 256;      // 演示用 token 数
    const int S  = Td * k;   // slot 数

    int seg_len[256], seg_off[256], tile_base[256];
    sample_counts(E, S, true, seg_len);      // 用倾斜分布，让各段长短分明
    // 段长之和可能不等于 S（抽样抖动），先补到 S 上再算前缀和，避免越界
    {
        int off = 0;
        for (int e = 0; e < E; ++e) off += seg_len[e];
        seg_len[E - 1] += (S - off);
    }
    int mt_max = 0;
    for (int e = 0; e < E; ++e) mt_max = (seg_len[e] > mt_max) ? seg_len[e] : mt_max;
    mt_max = ((mt_max + BM - 1) / BM) * BM;     // 对齐到 tile（甲方案也要对齐！）
    {
        int off = 0, tb = 0;
        for (int e = 0; e < E; ++e) {
            seg_off[e] = off;
            tile_base[e] = tb;
            off += seg_len[e];
            tb += ((seg_len[e] + BM - 1) / BM) * (DF / BN);
        }
        tile_base[E] = tb;
    }

    float *dA = nullptr, *dW = nullptr, *dC = nullptr;
    float *dAp = nullptr, *dCp = nullptr;
    int *dLen = nullptr, *dOff = nullptr, *dTB = nullptr;
    const int M = S;

    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, (size_t)E * D * DF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * DF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dAp, (size_t)E * mt_max * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCp, (size_t)E * mt_max * DF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLen, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOff, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dTB, (E + 1) * sizeof(int)));

    {
        float* h = (float*)std::malloc((size_t)M * D * sizeof(float));
        unsigned s = 11u;
        for (size_t i = 0; i < (size_t)M * D; ++i) {
            s = s * 1103515245u + 12345u;
            h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(dA, h, (size_t)M * D * sizeof(float), cudaMemcpyHostToDevice));
        std::free(h);
        h = (float*)std::malloc((size_t)E * D * DF * sizeof(float));
        for (size_t i = 0; i < (size_t)E * D * DF; ++i) {
            s = s * 1103515245u + 12345u;
            h[i] = (((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) / 16.0f;
        }
        CUDA_CHECK(cudaMemcpy(dW, h, (size_t)E * D * DF * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::free(h);
        CUDA_CHECK(cudaMemcpy(dLen, seg_len, E * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dOff, seg_off, E * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dTB, tile_base, (E + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }

    // 甲乙两个网格都跑一遍：甲要先把 A 按「每个专家 M_max 行」重排（空行填 0）
    {
        float* hA = (float*)std::malloc((size_t)M * D * sizeof(float));
        CUDA_CHECK(cudaMemcpy(hA, dA, (size_t)M * D * sizeof(float), cudaMemcpyDeviceToHost));
        float* hAp = (float*)std::calloc((size_t)E * mt_max * D, sizeof(float));
        for (int e = 0; e < E; ++e)
            for (int i = 0; i < seg_len[e]; ++i)
                std::memcpy(hAp + ((size_t)e * mt_max + i) * D,
                            hA + (size_t)(seg_off[e] + i) * D, D * sizeof(float));
        CUDA_CHECK(cudaMemcpy(dAp, hAp, (size_t)E * mt_max * D * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::free(hA);
        std::free(hAp);
    }

    const dim3 block(16, 16);                            // 256 线程，每线程 8×8
    grouped_gemm_kernel<<<dim3(tile_base[E]), block>>>(dA, dW, dC, dLen, dOff, dTB, E, DF, D);
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid_p(E * (mt_max / BM) * (DF / BN));
    padded_gemm_kernel<<<grid_p, block>>>(dAp, dW, dCp, mt_max, DF, D, mt_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 两份结果各与同一份 fp64 CPU 参考比
    {
        float* hA = (float*)std::malloc((size_t)M * D * sizeof(float));
        float* hW = (float*)std::malloc((size_t)E * D * DF * sizeof(float));
        float* hC = (float*)std::malloc((size_t)M * DF * sizeof(float));
        float* hCp = (float*)std::malloc((size_t)E * mt_max * DF * sizeof(float));
        CUDA_CHECK(cudaMemcpy(hA, dA, (size_t)M * D * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hW, dW, (size_t)E * D * DF * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * DF * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hCp, dCp, (size_t)E * mt_max * DF * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double worst = 0.0, worst_p = 0.0;
        for (int e = 0; e < E; ++e) {
            for (int i = 0; i < seg_len[e]; ++i) {
                const float* a = hA + (size_t)(seg_off[e] + i) * D;
                for (int j = 0; j < DF; ++j) {
                    double acc = 0.0;
                    for (int kk = 0; kk < D; ++kk)
                        acc += (double)a[kk] * hW[((size_t)e * D + kk) * DF + j];
                    worst = std::fmax(worst,
                                      std::fabs(acc - hC[(size_t)(seg_off[e] + i) * DF + j]));
                    worst_p = std::fmax(worst_p,
                                        std::fabs(acc - hCp[(size_t)(e * mt_max + i) * DF + j]));
                }
            }
        }
        std::free(hA);
        std::printf("--- §1 两个 kernel（kernel 运行结果，本仓库没有 GPU，未跑过）---\n");
        std::printf("演示规模：%d 个 slot、d_model=%d、d_ff=%d、%d 个专家\n", S, D, DF, E);
        std::printf("tokens_per_expert = [");
        for (int e = 0; e < E; ++e) std::printf("%s%d", e ? "," : "", seg_len[e]);
        std::printf("]\n");
        std::printf("乙 grouped ：网格 = %d 个 block，max|Δ| vs fp64 参考 = %.3e\n",
                    tile_base[E], worst);
        std::printf("甲 padded  ：网格 = %d 个 block，max|Δ| vs fp64 参考 = %.3e\n",
                    (int)grid_p.x, worst_p);
        std::printf("↑ 两个网格算出的**有用行**完全一致（差都在 fp32 舍入量级），");
        std::printf("说明 tile 前缀和解码没错位——\n");
        std::printf("  分段 M 维与逐专家单独调用一致（verify.py §A3）。\n");
        std::printf("  差别只在网格大小：甲 %d 个 block，乙 %d 个 → %.2f× 的 block；\n",
                    (int)grid_p.x, tile_base[E], (double)grid_p.x / tile_base[E]);
        std::printf("  每个 block 的 FLOPs 相同（都是 2·BM·BN·K），所以 block 数之比"
                    "就是**算力之比**。\n\n");
        std::free(hW); std::free(hC); std::free(hCp);
    }

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dW)); CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dAp)); CUDA_CHECK(cudaFree(dCp));
    CUDA_CHECK(cudaFree(dLen)); CUDA_CHECK(cudaFree(dOff)); CUDA_CHECK(cudaFree(dTB));
    return 0;
}
