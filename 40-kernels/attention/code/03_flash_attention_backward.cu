// 03_flash_attention_backward.cu —— flash attention 反向：重算 S、dQ 的两种累加
//
// 配套文档：40-kernels/attention/03-flash-attention-backward.md
//
// 前向把 S 丢掉了，反向就得**重算**它。本文讲清三件事：
//
// ① **重算是划算的**：重算一次 S 多花 2·S²·d 的 FLOPs（前向的 0.5 倍），
//    省下的是 P 的写 + 读 = 2·S²·esize。S=16384、d=128、单 head 时前者 69.4 µs、
//    后者 320.5 µs —— 重算便宜 4.62 倍（verify.py §B5，**推导值**）。
//
// ② **D = rowsum(dO ∘ O) 必须整行一次算出来**。这是反向能扔掉 P 的前提：
//    dS = P∘(dP − D) 里只有 D 这一项跨块求和，而 D 不需要 P 就能从 O 算出。
//    按块各算各的 D 会**安静地**毁掉 dQ 与 dK，**dV 则纹丝不动**——
//    verify.py §C6：dQ 2.75e-01 / dK 2.92e-01 / dV 2.45e-16。
//
// ③ **dQ 跨 KV 块累加的两种方式**（本文件给了三条路径）：
//      backward_seq_dq_kernel       dQ 在寄存器里顺序累加 → 确定、可复现；
//      backward_atomic_dq_kernel    dQ 用 atomicAdd       → 顺序由硬件决定，位模式会变；
//      backward_workspace_kernel    dQ 写 workspace 再固定顺序归约 → 确定，多一次 launch。
//    verify.py §B6：两种方式的**访存字节数几乎一样**（n_split=8 时差 1.05 MB ≈ 0.3 µs），
//    真正的差别是**确定性**；§C3：换一种顺序，265/384 个元素的位模式就变了。
//
// **三个张量的切分维度不同**（§2 的核心，也最容易绕晕）：
//      dQ —— 按 **Q 的行** 切；dK / dV —— 按 **K 的行** 切。
//    于是一个 block「循环另一维」时，原子累加落在谁头上是不一样的：
//    按 Q 行切 ⟹ dQ 无原子、dK/dV 有原子；按 KV 段切 ⟹ dQ 有原子。
//    **两个方向的原子不可能同时省掉**，只能选一个方向做成「循环 + 局部队列」。
//
// ⚠ 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 03_flash_attention_backward
//     ./03_flash_attention_backward             # S=512，d=64，Bc=64
//     ./03_flash_attention_backward 1024 64 64

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

constexpr int MAX_D  = 128;     // 本文件的简化上限：d ≤ 128
constexpr int MAX_BC = 128;     // Bc ≤ 128

// ---------------------------------------------------------------------------
// 逐块重算 S（**不物化 P**）：一个 Q 行 × 一个 KV 块 → dQ/dK/dV 的贡献
//
//   lse = m + ln l 是前向存下来的那一对数（前向最终值，**不是**该块当时的运行值）
//   D   = Σ_c dO[c]·O[c] 是**整行**的量，从最终的 O 一次算出
// ---------------------------------------------------------------------------
__device__ __forceinline__ void recompute_block(const half* q_row, const half* dO_row,
                                                const half* O_row,
                                                const half* sK, const half* sV,
                                                int nc, int d, float scale, float lse,
                                                float* dq,          // [d] 累加
                                                float* dk,          // [nc*d] 局部
                                                float* dv)          // [nc*d] 局部
{
    float p[MAX_BC], dP[MAX_BC], ds[MAX_BC];

    float D = 0.0f;                                   // ← 整行算，见 §C6
    for (int c = 0; c < d; ++c)
        D += __half2float(dO_row[c]) * __half2float(O_row[c]);

    for (int j = 0; j < nc; ++j) {
        float s = 0.0f;
        for (int c = 0; c < d; ++c)
            s += __half2float(q_row[c]) * __half2float(sK[(size_t)j * d + c]);
        p[j] = expf(s * scale - lse);                 // lse 里已经含了 scale 之前的 m
    }
    for (int j = 0; j < nc; ++j) {
        float t = 0.0f;                               // dP_j = Σ_c dO[c]·V[j][c]
        for (int c = 0; c < d; ++c)
            t += __half2float(dO_row[c]) * __half2float(sV[(size_t)j * d + c]);
        dP[j] = t;
        ds[j] = p[j] * (t - D) * scale;               // dS = P∘(dP − D)·scale
    }
    for (int c = 0; c < d; ++c) {                     // dQ += dS·K_block
        float t = 0.0f;
        for (int j = 0; j < nc; ++j) t += ds[j] * __half2float(sK[(size_t)j * d + c]);
        dq[c] += t;
    }
    for (int j = 0; j < nc; ++j)
        for (int c = 0; c < d; ++c) {
            dk[(size_t)j * d + c] += ds[j] * __half2float(q_row[c]);   // dK = dSᵀ·Q
            dv[(size_t)j * d + c] += p[j] * __half2float(dO_row[c]);   // dV = Pᵀ·dO
        }
}

// ---------------------------------------------------------------------------
// 路径一：按 **Q 的行** 切 —— 一个 block 一行 query，循环全部 KV 块
//   dQ 在寄存器里顺序累加（**无原子、确定**）
//   dK/dV 跨所有 Q 块汇总 ⟹ 只能 atomicAdd
// ---------------------------------------------------------------------------
__global__ void backward_seq_dq_kernel(const half* __restrict__ Q,
                                       const half* __restrict__ K,
                                       const half* __restrict__ V,
                                       const half* __restrict__ O,
                                       const half* __restrict__ dO,
                                       const float* __restrict__ Lse,
                                       float* __restrict__ dQ,
                                       float* __restrict__ dK,
                                       float* __restrict__ dV,
                                       int seq, int d, float scale, int Bc)
{
    extern __shared__ half smem[];
    half* sK = smem;
    half* sV = smem + (size_t)Bc * d;

    const int qi = blockIdx.x;
    if (qi >= seq) return;
    float dq[MAX_D];
    for (int c = 0; c < d; ++c) dq[c] = 0.0f;

    for (int j0 = 0; j0 < seq; j0 += Bc) {
        const int nc = min(Bc, seq - j0);
        for (int i = threadIdx.x; i < nc * d; i += blockDim.x) {
            sK[i] = K[(size_t)(j0 + i / d) * d + i % d];
            sV[i] = V[(size_t)(j0 + i / d) * d + i % d];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float dk[MAX_BC * MAX_D], dv[MAX_BC * MAX_D];
            for (int i = 0; i < nc * d; ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }
            recompute_block(Q + (size_t)qi * d, dO + (size_t)qi * d, O + (size_t)qi * d,
                            sK, sV, nc, d, scale, Lse[qi], dq, dk, dv);
            for (int i = 0; i < nc * d; ++i) {          // ← 按 K 的行汇总，必须原子
                atomicAdd(&dK[(size_t)(j0 + i / d) * d + i % d], dk[i]);
                atomicAdd(&dV[(size_t)(j0 + i / d) * d + i % d], dv[i]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)                              // ← 自己的行，写一次就够
        for (int c = 0; c < d; ++c) dQ[(size_t)qi * d + c] = dq[c];
}

// ---------------------------------------------------------------------------
// 路径二：按 **KV 段** 切 —— 同一个 Q 行块被 n_split 个 block 同时算，dQ 用 atomicAdd
//   顺序由硬件决定 ⟹ **不可复现**
// ---------------------------------------------------------------------------
__global__ void backward_atomic_dq_kernel(const half* __restrict__ Q,
                                          const half* __restrict__ K,
                                          const half* __restrict__ V,
                                          const half* __restrict__ O,
                                          const half* __restrict__ dO,
                                          const float* __restrict__ Lse,
                                          float* __restrict__ dQ,
                                          float* __restrict__ dK,
                                          float* __restrict__ dV,
                                          int seq, int d, float scale, int Bc,
                                          int n_split)
{
    extern __shared__ half smem[];
    half* sK = smem;
    half* sV = smem + (size_t)Bc * d;

    const int qi = blockIdx.x / n_split;
    const int sp = blockIdx.x % n_split;
    if (qi >= seq) return;
    const int per = (seq + n_split - 1) / n_split;
    const int j_lo = sp * per, j_hi = min(seq, j_lo + per);
    if (j_lo >= j_hi) return;

    float dq[MAX_D];
    for (int c = 0; c < d; ++c) dq[c] = 0.0f;
    for (int j0 = j_lo; j0 < j_hi; j0 += Bc) {
        const int nc = min(Bc, j_hi - j0);
        for (int i = threadIdx.x; i < nc * d; i += blockDim.x) {
            sK[i] = K[(size_t)(j0 + i / d) * d + i % d];
            sV[i] = V[(size_t)(j0 + i / d) * d + i % d];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float dk[MAX_BC * MAX_D], dv[MAX_BC * MAX_D];
            for (int i = 0; i < nc * d; ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }
            recompute_block(Q + (size_t)qi * d, dO + (size_t)qi * d, O + (size_t)qi * d,
                            sK, sV, nc, d, scale, Lse[qi], dq, dk, dv);
            for (int i = 0; i < nc * d; ++i) {
                atomicAdd(&dK[(size_t)(j0 + i / d) * d + i % d], dk[i]);
                atomicAdd(&dV[(size_t)(j0 + i / d) * d + i % d], dv[i]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)                              // ← 多个段同时写同一行
        for (int c = 0; c < d; ++c) atomicAdd(&dQ[(size_t)qi * d + c], dq[c]);
}

// ---------------------------------------------------------------------------
// 路径三：按 KV 段切，但 dQ 写进 workspace，再由 dq_reduce_kernel 固定顺序归约
//   —— 结果与段数、与执行顺序都无关
// ---------------------------------------------------------------------------
__global__ void backward_workspace_kernel(const half* __restrict__ Q,
                                          const half* __restrict__ K,
                                          const half* __restrict__ V,
                                          const half* __restrict__ O,
                                          const half* __restrict__ dO,
                                          const float* __restrict__ Lse,
                                          float* __restrict__ dQpart,  // [n_split][seq][d]
                                          float* __restrict__ dK,
                                          float* __restrict__ dV,
                                          int seq, int d, float scale, int Bc,
                                          int n_split, int split_id)
{
    extern __shared__ half smem[];
    half* sK = smem;
    half* sV = smem + (size_t)Bc * d;

    const int qi = blockIdx.x;
    if (qi >= seq) return;
    const int per = (seq + n_split - 1) / n_split;
    const int j_lo = split_id * per, j_hi = min(seq, j_lo + per);
    if (j_lo >= j_hi) return;

    float dq[MAX_D];
    for (int c = 0; c < d; ++c) dq[c] = 0.0f;
    for (int j0 = j_lo; j0 < j_hi; j0 += Bc) {
        const int nc = min(Bc, j_hi - j0);
        for (int i = threadIdx.x; i < nc * d; i += blockDim.x) {
            sK[i] = K[(size_t)(j0 + i / d) * d + i % d];
            sV[i] = V[(size_t)(j0 + i / d) * d + i % d];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float dk[MAX_BC * MAX_D], dv[MAX_BC * MAX_D];
            for (int i = 0; i < nc * d; ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }
            recompute_block(Q + (size_t)qi * d, dO + (size_t)qi * d, O + (size_t)qi * d,
                            sK, sV, nc, d, scale, Lse[qi], dq, dk, dv);
            for (int i = 0; i < nc * d; ++i) {
                atomicAdd(&dK[(size_t)(j0 + i / d) * d + i % d], dk[i]);
                atomicAdd(&dV[(size_t)(j0 + i / d) * d + i % d], dv[i]);
            }
        }
        __syncthreads();
    }
    float* out = dQpart + (size_t)split_id * seq * d + (size_t)qi * d;
    for (int c = 0; c < d; ++c) out[c] = dq[c];
}

__global__ void dq_reduce_kernel(const float* __restrict__ dQpart,
                                 float* __restrict__ dQ, int seq, int d, int n_split)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq * d) return;
    float t = 0.0f;
    for (int s = 0; s < n_split; ++s)                 // **固定顺序**，与 grid 无关
        t += dQpart[(size_t)s * seq * d + i];
    dQ[i] = t;
}

// ---------------------------------------------------------------------------
// CPU double 参考：物化 P 的朴素反向（verify.py 的 attn_backward_ref 同式）
// ---------------------------------------------------------------------------
static void backward_ref(const std::vector<float>& Q, const std::vector<float>& K,
                         const std::vector<float>& V, const std::vector<float>& dO,
                         int seq, int d, float scale,
                         std::vector<double>& dQ, std::vector<double>& dK,
                         std::vector<double>& dV, std::vector<double>& Lse,
                         std::vector<double>& O)
{
    dQ.assign((size_t)seq * d, 0.0);
    dK.assign((size_t)seq * d, 0.0);
    dV.assign((size_t)seq * d, 0.0);
    O.assign((size_t)seq * d, 0.0);
    Lse.assign(seq, 0.0);

    std::vector<double> S((size_t)seq * seq), P((size_t)seq * seq);
    for (int i = 0; i < seq; ++i) {
        double m = -1e300;
        for (int j = 0; j < seq; ++j) {
            double t = 0.0;
            for (int c = 0; c < d; ++c)
                t += (double)Q[(size_t)i * d + c] * (double)K[(size_t)j * d + c];
            S[(size_t)i * seq + j] = t * scale;
            m = std::max(m, S[(size_t)i * seq + j]);
        }
        double l = 0.0;
        for (int j = 0; j < seq; ++j) {
            P[(size_t)i * seq + j] = std::exp(S[(size_t)i * seq + j] - m);
            l += P[(size_t)i * seq + j];
        }
        Lse[i] = m + std::log(l);
        for (int j = 0; j < seq; ++j) P[(size_t)i * seq + j] /= l;   // 归一化的 P
        for (int c = 0; c < d; ++c) {
            double o = 0.0;
            for (int j = 0; j < seq; ++j)
                o += P[(size_t)i * seq + j] * (double)V[(size_t)j * d + c];
            O[(size_t)i * d + c] = o;
        }
    }
    for (int i = 0; i < seq; ++i) {
        std::vector<double> dP(seq, 0.0), D(1, 0.0);
        for (int j = 0; j < seq; ++j) {
            double t = 0.0;
            for (int c = 0; c < d; ++c) t += (double)dO[(size_t)i * d + c] * (double)V[(size_t)j * d + c];
            dP[j] = t;
        }
        for (int c = 0; c < d; ++c)
            D[0] += (double)dO[(size_t)i * d + c] * O[(size_t)i * d + c];   // 整行
        for (int j = 0; j < seq; ++j) {
            const double ds = P[(size_t)i * seq + j] * (dP[j] - D[0]) * scale;
            for (int c = 0; c < d; ++c) {
                dQ[(size_t)i * d + c] += ds * (double)K[(size_t)j * d + c];
                dK[(size_t)j * d + c] += ds * (double)Q[(size_t)i * d + c];
                dV[(size_t)j * d + c] += P[(size_t)i * seq + j] * (double)dO[(size_t)i * d + c];
            }
        }
    }
}

static double maxdiff(const std::vector<float>& a, const std::vector<double>& b,
                      double& norm)
{
    double e = 0.0;
    norm = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::fmax(e, std::fabs((double)a[i] - b[i]));
        norm = std::fmax(norm, std::fabs(b[i]));
    }
    return e;
}

int main(int argc, char** argv)
{
    const int seq = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int d   = (argc > 2) ? std::atoi(argv[2]) : 64;
    const int Bc  = (argc > 3) ? std::atoi(argv[3]) : 64;
    const float scale = 1.0f / std::sqrt((float)d);
    const double S = seq;

    std::printf("=== 03_flash_attention_backward：S=%d d=%d Bc=%d ===\n\n", seq, d, Bc);

    std::printf("[1] 重算 vs 存储（**推导值**，verify.py §B5，单 head 单 batch，d=128）：\n");
    std::printf("    重算一次 S 的额外 FLOPs = 2·S²·d，S=16384 时 = %.3e → %.1f µs\n",
                2.0 * 16384 * 16384 * 128.0, 2.0 * 16384 * 16384 * 128.0 / 989.5e12 * 1e6);
    std::printf("    省下的 P 写 + 读 = 2·S²·esize = %.3f GB → %.1f µs\n",
                2.0 * 16384 * 16384 * 2 / 1e9, 2.0 * 16384 * 16384 * 2 / 3350e9 * 1e6);
    std::printf("    ⟹ 重算便宜 %.2f 倍；反向总 FLOPs ≈ 10·S²·d，是前向的 2.5 倍。\n\n",
                (2.0 * 16384 * 16384 * 2 / 3350e9) / (2.0 * 16384 * 16384 * 128.0 / 989.5e12));

    std::printf("[2] dQ 的两种累加（**推导值**，verify.py §B6，Sq=%d d=%d fp32）：\n", seq, d);
    for (int ns = 1; ns <= 8; ns *= 2) {
        const double a = 2.0 * ns * S * d * 4;                 // 读-改-写
        const double w = (2.0 * ns + 1.0) * S * d * 4;         // 多写一份 + 归约
        std::printf("    n_split=%d：atomic %.2f MB / workspace %.2f MB（差 %.2f MB），"
                    "atomic 侧 %.1f µs @3350 GB/s\n",
                    ns, a / 1e6, w / 1e6, (w - a) / 1e6, a / 3350e9 * 1e6);
    }
    std::printf("    ⚠ 流量几乎一样，差别在**确定性**；workspace 一侧多一次 launch\n");
    std::printf("      （3~5 µs，**引用值，本仓库不复核其测量口径**）。\n\n");

    // ---- 数值校验 ----
    std::mt19937 gen(20260916);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t n = (size_t)seq * d;
    std::vector<float> fQ(n), fK(n), fV(n), fdO(n);
    std::vector<half> hQ(n), hK(n), hV(n), hO(n), hdO(n);
    for (size_t i = 0; i < n; ++i) {
        fQ[i] = nd(gen); fK[i] = nd(gen); fV[i] = nd(gen); fdO[i] = nd(gen);
        hQ[i] = __float2half(fQ[i]); hK[i] = __float2half(fK[i]);
        hV[i] = __float2half(fV[i]); hdO[i] = __float2half(fdO[i]);
    }
    std::vector<double> rdQ, rdK, rdV, rLse, rO;
    backward_ref(fQ, fK, fV, fdO, seq, d, scale, rdQ, rdK, rdV, rLse, rO);
    for (size_t i = 0; i < n; ++i) hO[i] = __float2half((float)rO[i]);
    std::vector<float> hLse(seq);
    for (int i = 0; i < seq; ++i) hLse[i] = (float)rLse[i];

    half *dQ_h, *dK_h, *dV_h, *dO_h, *dOh;
    float *dQf, *dKf, *dVf, *dLse, *dQpart;
    CUDA_CHECK(cudaMalloc(&dQ_h, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK_h, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV_h, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dO_h, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOh, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dQf, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dKf, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVf, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLse, seq * sizeof(float)));
    const int n_split = 4;
    CUDA_CHECK(cudaMalloc(&dQpart, (size_t)n_split * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ_h, hQ.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK_h, hK.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV_h, hV.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dO_h, hdO.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOh, hO.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLse, hLse.data(), seq * sizeof(float), cudaMemcpyHostToDevice));

    const size_t shmem = (size_t)Bc * d * 2 * sizeof(half);
    std::vector<float> gQ1(n), gQ2(n), gQ3(n), gK(n), gV(n);

    auto reset = [&] {
        CUDA_CHECK(cudaMemset(dQf, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(dKf, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(dVf, 0, n * sizeof(float)));
    };

    // 路径一：顺序 dQ
    reset();
    backward_seq_dq_kernel<<<seq, 128, shmem>>>(dQ_h, dK_h, dV_h, dOh, dO_h, dLse,
                                                dQf, dKf, dVf, seq, d, scale, Bc);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gQ1.data(), dQf, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gK.data(), dKf, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gV.data(), dVf, n * sizeof(float), cudaMemcpyDeviceToHost));

    // 路径二：atomic dQ
    reset();
    backward_atomic_dq_kernel<<<seq * n_split, 128, shmem>>>(dQ_h, dK_h, dV_h, dOh, dO_h,
                                                             dLse, dQf, dKf, dVf, seq, d,
                                                             scale, Bc, n_split);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gQ2.data(), dQf, n * sizeof(float), cudaMemcpyDeviceToHost));

    // 路径三：workspace + 固定顺序归约
    reset();
    for (int s = 0; s < n_split; ++s)
        backward_workspace_kernel<<<seq, 128, shmem>>>(dQ_h, dK_h, dV_h, dOh, dO_h, dLse,
                                                       dQpart, dKf, dVf, seq, d, scale,
                                                       Bc, n_split, s);
    dq_reduce_kernel<<<(int)((n + 255) / 256), 256>>>(dQpart, dQf, seq, d, n_split);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gQ3.data(), dQf, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 打印（**本机运行结果**，本仓库没有跑过）----
    double norm;
    const double e1 = maxdiff(gQ1, rdQ, norm), en1 = e1 / norm;
    const double e2 = maxdiff(gQ2, rdQ, norm), en2 = e2 / norm;
    const double e3 = maxdiff(gQ3, rdQ, norm), en3 = e3 / norm;
    const double eK = maxdiff(gK, rdK, norm), enK = eK / norm;
    const double eV = maxdiff(gV, rdV, norm), enV = eV / norm;
    std::printf("[3] 与 double 参考的差（**本机运行结果**）：\n");
    std::printf("    路径一 顺序 dQ   ：dQ %.3e（按范数 %.3e）\n", e1, en1);
    std::printf("    路径二 atomic dQ ：dQ %.3e（按范数 %.3e）\n", e2, en2);
    std::printf("    路径三 workspace ：dQ %.3e（按范数 %.3e）\n", e3, en3);
    std::printf("    dK %.3e（%.3e）  dV %.3e（%.3e）\n\n", eK, enK, eV, enV);

    int nb12 = 0, nb13 = 0;
    for (size_t i = 0; i < n; ++i) {
        const unsigned b1 = *reinterpret_cast<unsigned*>(&gQ1[i]);
        const unsigned b2 = *reinterpret_cast<unsigned*>(&gQ2[i]);
        const unsigned b3 = *reinterpret_cast<unsigned*>(&gQ3[i]);
        if (b1 != b2) ++nb12;
        if (b1 != b3) ++nb13;
    }
    std::printf("    位模式不同的元素个数：顺序 vs atomic %d/%zu，顺序 vs workspace %d/%zu\n",
                nb12, n, nb13, n);
    std::printf("    ⟹ **数值一致、位模式不同**：这就是「dQ 用 atomicAdd 导致不可复现」\n");
    std::printf("       在真机上的形态（verify.py §C3 用换累加顺序模拟了同一件事）。\n");
    std::printf("       要可复现，就必须走 workspace + 固定顺序归约那条路。\n\n");

    std::printf("[4] 三条必须在代码里落到实处的纪律：\n");
    std::printf("    ① D = rowsum(dO ∘ O) **整行**算（recompute_block 的头两行）；\n");
    std::printf("       按块算会毁掉 dQ 与 dK，而 **dV 纹丝不动**（§C6）。\n");
    std::printf("    ② 重算 P 用**前向最终**的 lse = m + ln l，不要用该块当时的运行值\n");
    std::printf("       （§C5：三项全错，最好的一项也错到 9.15e-01）。\n");
    std::printf("    ③ dQ 按 Q 的行、dK/dV 按 K 的行 —— 两个方向的原子不可能同时省掉。\n");

    CUDA_CHECK(cudaFree(dQ_h)); CUDA_CHECK(cudaFree(dK_h)); CUDA_CHECK(cudaFree(dV_h));
    CUDA_CHECK(cudaFree(dO_h)); CUDA_CHECK(cudaFree(dOh));
    CUDA_CHECK(cudaFree(dQf)); CUDA_CHECK(cudaFree(dKf)); CUDA_CHECK(cudaFree(dVf));
    CUDA_CHECK(cudaFree(dLse)); CUDA_CHECK(cudaFree(dQpart));
    return 0;
}
