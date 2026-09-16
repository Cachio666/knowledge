// 02_flash_attention.cu —— flash attention 前向：在线 softmax + 不物化 S
//
// 配套文档：40-kernels/attention/02-flash-attention.md
//
// **(m, l) 的递推不是本文的正文**，它住在
// 40-kernels/softmax-normalization/02-online-softmax（那边逐项推导）。
// 本文只做一件事：**把输出累加器 O 也纳入重标定**，并说清为什么必须这样。
//
// 三条递推（前两条回链，第三条是 attention 特有的）：
//     m_new = max(m_old, rowmax(S_block))                          ← 回链 softmax/02 (1)
//     l_new = l_old·exp(m_old − m_new) + Σ exp(S_block − m_new)    ← 回链 softmax/02 (2)
//     O_new = O_old·exp(m_old − m_new) + exp(S_block − m_new)·V    ← ★ 本文的正文
//
// 第三条为什么成立（文档 §1 的推导骨架）：O 是「按**当前**基准归一化的加权和」，
// 基准从 m_old 抬到 m_new 时，**所有**按旧基准算出来的量都要整体乘 α；l 是这样，
// O 也是这样——两者用的是**同一个 α**，因为它们共享同一个 m。
// 漏掉 O 那一份不会触发任何数值异常，只会安静地错（verify.py §C4：相对误差 33）。
//
// 访存：读 Q/K/V + 写 O = 4·S·d·esize。**S 从未落 HBM**。
//
// ⚠ 状态 (m, l, O) 必须住 float：
//   · α = exp(m_old − m_new) 在 fp16 状态下于 gap > 17.3287（精确口径）时归零，
//     整段已累积的 O 被静默丢弃（verify.py §C2）；
//   · O 的丢弃比 l 的丢弃更值钱——l 丢的是一个计数，O 丢的是算好的输出向量。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 02_flash_attention
//     ./02_flash_attention                # S=1024, d=128, Br=64, Bc=64
//     ./02_flash_attention 4096 128 128 64
//
//   数字分两类：**推导值**（SMEM/寄存器/字节，可用 `make verify` §B4 复核）与
//   **本机运行结果**（逐元素差、耗时；本仓库没有跑过）。

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

constexpr int WARP = 32;
constexpr int MAX_D = 128;      // 本文件的简化上限：d ≤ 128（acc 直接开成定长数组）
constexpr int MAX_BC = 128;     // Bc ≤ 128（s_block / p 的定长上限）

// ---------------------------------------------------------------------------
// flash 前向：一个 block 处理 Br 行 query，循环全部 KV 块
//
// 状态：m（Br 个 float）、l（Br 个 float）、O（Br×d 个 float）。
// **O 的大小与 Br·d 成正比，与分块无关**——这是本内核最硬的资源约束
// （verify.py §B4：Br=128、d=128、4 个 warp 时光 O 就要 128 个寄存器/线程）。
//
// rescale_O = 0 时**只重标定 l，不重标定 O** —— 故意的错误版本，
// 用来在真机上复现 §4.2 那条「安静地错」。
// ---------------------------------------------------------------------------
__global__ void flash_attention_kernel(const half* __restrict__ Q,
                                       const half* __restrict__ K,
                                       const half* __restrict__ V,
                                       half* __restrict__ O,
                                       int seq, int d, float scale,
                                       int Br, int Bc, int rescale_O)
{
    extern __shared__ half smem[];                       // K tile 与 V tile
    half* sK = smem;                                     // Bc × d
    half* sV = smem + (size_t)Bc * d;                    // Bc × d

    const int q0 = blockIdx.x * Br;
    const int tid = threadIdx.x;
    const int qi = q0 + tid;                             // 一线程一行（简化）
    const bool active = (tid < Br) && (qi < seq);

    // ---- O 累加器直接住寄存器：d 个 fp32 ----
    // **这就是 §B4 说的那件事**：Br 越大，每线程的 acc 越长；d=128 时
    // 一个 warp 里的每个 lane 都要扛 128 个 fp32 寄存器，只剩很少的余量给
    // S 的 fragment、m/l 与地址。Bc 完全不改变这项开销。
    float acc[MAX_D];
    float m = -INFINITY, l = 0.0f;
    for (int c = 0; c < d; ++c) acc[c] = 0.0f;

    // ---- 主循环：逐个 KV 块流式推进 ----
    for (int j0 = 0; j0 < seq; j0 += Bc) {
        const int nc = min(Bc, seq - j0);

        // 搬 K/V 到 shared memory（真实实现用 cp.async / TMA + 多级流水）
        for (int i = tid; i < nc * d; i += blockDim.x) {
            sK[i] = K[(size_t)(j0 + i / d) * d + i % d];
            sV[i] = V[(size_t)(j0 + i / d) * d + i % d];
        }
        __syncthreads();

        if (active) {
            const half* q = Q + (size_t)qi * d;
            float s_block[MAX_BC], p[MAX_BC];

            // ① S_block = scale·q·K_blockᵀ，并取行内 max
            float m_new = m;
            for (int j = 0; j < nc; ++j) {
                float s = 0.0f;
                for (int t = 0; t < d; ++t)
                    s += __half2float(q[t]) * __half2float(sK[(size_t)j * d + t]);
                const float sb = s * scale;
                s_block[j] = sb;
                m_new = fmaxf(m_new, sb);
            }

            // ② 重标定因子 α —— (m, l) 递推的正文在 softmax-normalization/02，
            //    这里只用它的结论：α = exp(m_old − m_new) ∈ (0, 1]
            const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);

            // ③ l 与 **O** 用同一个 α 重标定 —— 下面这几行是本文件的核心
            float l_new = l * alpha;
            for (int j = 0; j < nc; ++j) {
                p[j] = expf(s_block[j] - m_new);
                l_new += p[j];
            }
            if (rescale_O) {
                for (int c = 0; c < d; ++c) acc[c] *= alpha;    // ← 漏掉这行 = §4.2
            }
            for (int j = 0; j < nc; ++j)
                for (int c = 0; c < d; ++c)
                    acc[c] += p[j] * __half2float(sV[(size_t)j * d + c]);

            m = m_new;
            l = l_new;
        }
        __syncthreads();          // 下一轮要覆盖 sK / sV，先等所有人读完
    }

    // ---- 写出 O/l：**归一化只做一次**，在循环外 ----
    if (active) {
        const float inv = 1.0f / l;
        for (int c = 0; c < d; ++c)
            O[(size_t)qi * d + c] = __float2half(acc[c] * inv);
    }
}

// ---------------------------------------------------------------------------
static void flash_ref(const std::vector<float>& Q, const std::vector<float>& K,
                      const std::vector<float>& V, int seq, int d, float scale,
                      int Bc, std::vector<double>& O)
{
    // CPU 上的在线 softmax 分块版：与内核逐条对应，用来查内核写对没有
    O.assign((size_t)seq * d, 0.0);
    for (int i = 0; i < seq; ++i) {
        double m = -1e300, l = 0.0;
        std::vector<double> acc(d, 0.0);
        for (int j0 = 0; j0 < seq; j0 += Bc) {
            const int nc = std::min(Bc, seq - j0);
            std::vector<double> s(nc);
            double m_new = m;
            for (int j = 0; j < nc; ++j) {
                double t = 0.0;
                for (int c = 0; c < d; ++c)
                    t += (double)Q[(size_t)i * d + c] * (double)K[(size_t)(j0 + j) * d + c];
                s[j] = t * scale;
                m_new = std::max(m_new, s[j]);
            }
            const double alpha = (m == -1e300) ? 0.0 : std::exp(m - m_new);
            double l_new = l * alpha;
            for (int c = 0; c < d; ++c) acc[c] *= alpha;      // ← O 跟着重标定
            for (int j = 0; j < nc; ++j) {
                const double p = std::exp(s[j] - m_new);
                l_new += p;
                for (int c = 0; c < d; ++c) acc[c] += p * (double)V[(size_t)(j0 + j) * d + c];
            }
            m = m_new;
            l = l_new;
        }
        for (int c = 0; c < d; ++c) O[(size_t)i * d + c] = acc[c] / l;
    }
}

int main(int argc, char** argv)
{
    const int seq = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int d   = (argc > 2) ? std::atoi(argv[2]) : 128;
    const int Br  = (argc > 3) ? std::atoi(argv[3]) : 64;
    const int Bc  = (argc > 4) ? std::atoi(argv[4]) : 64;
    const float scale = 1.0f / std::sqrt((float)d);
    const float esize = 2.0f;

    std::printf("=== 02_flash_attention：S=%d d=%d Br=%d Bc=%d ===\n\n", seq, d, Br, Bc);

    std::printf("[1] 资源（**推导值**，与 verify.py §B4 同源）：\n");
    const double smem = ((double)Br * d + 2.0 * Bc * d) * esize;
    std::printf("    SMEM = (Br·d + 2·Bc·d)·esize = %.0f KB（上限 228 KB）\n", smem / 1024);
    std::printf("    O 累加器 = Br·d 个 fp32 ÷ 线程数：\n");
    std::printf("      128 线程 → %.0f 个寄存器/线程；256 线程 → %.0f 个\n",
                (double)Br * d / 128, (double)Br * d / 256);
    std::printf("    ⚠ O 的寄存器数与 Br·d 成正比、与 Bc **无关**——Bc 可以随便调，Br 不行。\n\n");

    std::printf("[2] 访存（推导值）：flash = 4·S·d·esize = %.2f MB；\n",
                4.0 * seq * d * esize / 1e6);
    std::printf("    朴素（01 篇）= %.2f MB，降幅 %.0f×。\n\n",
                (4.0 * seq * d + 2.0 * seq * seq) * esize / 1e6,
                (4.0 * seq * d + 2.0 * seq * seq) / (4.0 * seq * d));

    std::mt19937 gen(20260916);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<half> hQ((size_t)seq * d), hK((size_t)seq * d), hV((size_t)seq * d);
    std::vector<float> fQ((size_t)seq * d), fK((size_t)seq * d), fV((size_t)seq * d);
    for (size_t i = 0; i < fQ.size(); ++i) {
        fQ[i] = nd(gen); fK[i] = nd(gen); fV[i] = nd(gen);
        hQ[i] = __float2half(fQ[i]); hK[i] = __float2half(fK[i]); hV[i] = __float2half(fV[i]);
    }

    half *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dO, (size_t)seq * d * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<double> ref;
    flash_ref(fQ, fK, fV, seq, d, scale, Bc, ref);

    const size_t shmem = ((size_t)Bc * d * 2) * sizeof(half);
    // 一线程一行 ⟹ 线程数不能少于 Br；再多给一些线程只是为了搬 tile 更快
    const int nthreads = std::max(Br, 128);
    std::vector<half> hO((size_t)seq * d);
    for (int rescale = 1; rescale >= 0; --rescale) {
        flash_attention_kernel<<<(seq + Br - 1) / Br, nthreads, shmem>>>(
            dQ, dK, dV, dO, seq, d, scale, Br, Bc, rescale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(half),
                              cudaMemcpyDeviceToHost));
        double worst = 0.0, norm = 0.0;
        for (size_t i = 0; i < hO.size(); ++i) {
            worst = std::fmax(worst, std::fabs((double)__half2float(hO[i]) - ref[i]));
            norm  = std::fmax(norm, std::fabs(ref[i]));
        }
        std::printf("[3] %s：与 CPU 在线参考的最大绝对差 = %.4e（按范数 %.3e）\n",
                    rescale ? "重标定 O（正确）" : "**忘标定 O**（只重标定 l）",
                    worst, worst / norm);
        if (!rescale)
            std::printf("    ⟹ 这一版**既不报错也不溢出**，只是安静地错\n"
                        "       （verify.py §C4：相对误差 3.3e+01）。成因是 O_old 按旧基准\n"
                        "       累积、而新权重按新基准算，两者直接相加。\n");
    }

    std::printf("\n[4] 本文的正文与回链：\n");
    std::printf("    · (m, l) 两条递推的推到与证明 —— 正文在 40-kernels/softmax-normalization/\n");
    std::printf("      02-online-softmax（本文只回链，不重推）；\n");
    std::printf("    · **O 跟着同一个 α 重标定** —— 正文在本文 §1，上面那段主循环。\n");
    std::printf("    · exp 的阈值（11.0901 / −17.3287 / 88.7228 / −103.9721）——\n");
    std::printf("      正文在 softmax-normalization/01，本文只用结论（verify.py §A0 做一致性核对）。\n");

    CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    return 0;
}
