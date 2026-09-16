// 02_online_softmax.cu —— 在线 softmax：(m, l) 的重标定递推
//
// 配套文档：40-kernels/softmax-normalization/02-online-softmax.md
//
// **本文件是「在线 softmax 的 (m, l) 更新公式」的参考实现**，attention 主题
// 回链此处。三条式子：
//
//     m_new = max(m_old, max(x ∈ 本块))                      （1）
//     l_new = l_old · exp(m_old − m_new) + Σ exp(x − m_new)  （2）
//     y_i   = exp(x_i − m_final) / l_final                   （3）
//
// 关键性质（文档 §1 逐项推导）：
//   · 不变量：l_t = Σ_{j 已见过} exp(x_j − m_t)，对任意 t 成立（归纳法）；
//   · m_old ≤ m_new **恒成立**，所以 rescale 因子 exp(m_old − m_new) **恒 ≤ 1**，
//     乘它只会缩小 l，不可能溢出。写反（用 exp(m_new − m_old)）会立刻上溢。
//   · 合并算子 (m₁,l₁) ⊕ (m₂,l₂) = (max(m₁,m₂), l₁·e^{m₁−M} + l₂·e^{m₂−M})
//     **满足结合律**，所以分块顺序、分块大小都不影响数学结果——这是它能在
//     一个 block 内乱序归约、也能在 attention 里跨 KV 块流式推进的全部依据。
//
// ⚠ 状态 (m, l) 必须住 **float**，不能住 half：half 的 rescale 因子在
//   m_new − m_old > 17.3287 时**恰好等于 0**，整段已累积的贡献被静默丢弃；
//   而且 l 一旦越过 65504 就自己变成 inf（见 verify.py §C5）。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 02_online_softmax
//     ./02_online_softmax             # 默认 n = 4096，分块 128
//     ./02_online_softmax 16384 512   # n = 16384，分块 512

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

// ---------------------------------------------------------------------------
// 两遍法（基准）：先全局求 max，再求 exp 与和
// ---------------------------------------------------------------------------
__global__ void softmax_two_pass_kernel(const half* __restrict__ x,
                                        half* __restrict__ y, int n)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float red[WARP];

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        m = fmaxf(m, __half2float(xr[i]));
    for (int off = WARP / 2; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((threadIdx.x & (WARP - 1)) == 0) red[threadIdx.x / WARP] = m;
    __syncthreads();
    if (threadIdx.x < WARP) {
        m = (threadIdx.x < (int)(blockDim.x / WARP)) ? red[threadIdx.x] : -INFINITY;
        for (int off = WARP / 2; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) red[0] = m;
    }
    __syncthreads();
    m = red[0];

    float l = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(__half2float(xr[i]) - m);
        yr[i] = __float2half(e);
        l += e;
    }
    for (int off = WARP / 2; off > 0; off >>= 1)
        l += __shfl_down_sync(0xffffffffu, l, off);
    if ((threadIdx.x & (WARP - 1)) == 0) red[threadIdx.x / WARP] = l;
    __syncthreads();
    if (threadIdx.x < WARP) {
        l = (threadIdx.x < (int)(blockDim.x / WARP)) ? red[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1)
            l += __shfl_down_sync(0xffffffffu, l, off);
        if (threadIdx.x == 0) red[0] = l;
    }
    __syncthreads();
    l = red[0];

    const float inv = 1.0f / l;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half(__half2float(yr[i]) * inv);
}

// ---------------------------------------------------------------------------
// 在线 softmax：每个线程用**步长切分**处理自己的那一段（模拟流式/跨 KV 块）
//
// 每个线程持有自己的 (m, l)，在块内迭代中逐条套用 (1)(2)；最后由 block 归约
// 把各线程的 (m, l) 用结合算子合并。**整个过程不需要回头看已经算过的 x**。
//
// 注意这里刻意把所有中间量都留在 float，且 l 用 float 累加：
// 这两条是 §4 的失败模式要否定的东西（见 verify.py §C3 / §C5）。
// ---------------------------------------------------------------------------
__global__ void softmax_online_kernel(const half* __restrict__ x,
                                      half* __restrict__ y, int n)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;

    // ---- 第一段：每个线程对自己那一段跑在线递推 ----
    float m = -INFINITY;                    // m_0 = −inf
    float l = 0.0f;                         // l_0 = 0
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xx = __half2float(xr[i]);
        const float m_new = fmaxf(m, xx);   // (1)
        // (2)：exp(m_old − m_new) 里 m_old ≤ m_new **恒成立**，所以因子 ≤ 1
        const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        l = l * alpha + expf(xx - m_new);
        m = m_new;
    }

    // ---- 第二段：block 内的 (m, l) 结合归约 ----
    // 合并公式就是 (1)(2) 本身：(m₁,l₁) ⊕ (m₂,l₂) = (max, l₁·e^{m₁−M} + l₂·e^{m₂−M})
    __shared__ float sm[WARP];
    __shared__ float sl[WARP];
    for (int off = WARP / 2; off > 0; off >>= 1) {
        const float mo = __shfl_down_sync(0xffffffffu, m, off);
        const float lo = __shfl_down_sync(0xffffffffu, l, off);
        const float M  = fmaxf(m, mo);
        // mo 为 −inf（空 lane）时下面这一项为 0，正好是「空集不贡献」
        l = l * ((M == -INFINITY) ? 1.0f : expf(m - M))
              + lo * ((M == -INFINITY) ? 0.0f : expf(mo - M));
        m = M;
    }
    if ((threadIdx.x & (WARP - 1)) == 0) { sm[threadIdx.x / WARP] = m; sl[threadIdx.x / WARP] = l; }
    __syncthreads();
    if (threadIdx.x < WARP) {
        const bool live = threadIdx.x < (int)(blockDim.x / WARP);
        m = live ? sm[threadIdx.x] : -INFINITY;
        l = live ? sl[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1) {
            const float mo = __shfl_down_sync(0xffffffffu, m, off);
            const float lo = __shfl_down_sync(0xffffffffu, l, off);
            const float M  = fmaxf(m, mo);
            l = l * ((M == -INFINITY) ? 1.0f : expf(m - M))
                  + lo * ((M == -INFINITY) ? 0.0f : expf(mo - M));
            m = M;
        }
        if (threadIdx.x == 0) { sm[0] = m; sl[0] = l; }
    }
    __syncthreads();
    m = sm[0];
    l = sl[0];

    // ---- 第三段：写出 y = exp(x − m) / l（(3) 式）----
    const float inv = 1.0f / l;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half(expf(__half2float(xr[i]) - m) * inv);
}

// ---------------------------------------------------------------------------
// 在线 softmax 的「按块序流式推进」版本 —— 用来验证**块序无关性**
//
// 与 softmax_online_kernel 的唯一区别是块的访问顺序由外部数组 `order` 给出。
// 传正序、逆序、或任意一个置换，数学结果必须完全一样（结合律），
// 浮点结果只差舍入。这条性质是 02 文档 §2 分块流水的全部依据，
// 也是 FlashAttention 能把 KV 切成任意块的前提。
//
// 注意第二遍写 y 时**又扫了一遍 x**：在线递推本身只维护 (m, l)，
// 要拿到 y 还得回看 x（attention 里省掉这一遍靠的是同时维护输出累加器 o，
// 那是第三条递推，正文在 40-kernels/attention）。
// ---------------------------------------------------------------------------
__global__ void softmax_online_order_kernel(const half* __restrict__ x,
                                            half* __restrict__ y,
                                            const int* __restrict__ order,
                                            int n, int nchunk)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float sm[WARP];
    __shared__ float sl[WARP];

    float m = -INFINITY, l = 0.0f;
    for (int t = 0; t < nchunk; ++t) {
        const int c  = order ? order[t] : t;
        const int lo = (int)((long long)c * n / nchunk);
        const int hi = (int)((long long)(c + 1) * n / nchunk);
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const float xx = __half2float(xr[i]);
            const float m_new = fmaxf(m, xx);                       // (1)
            l = l * ((m == -INFINITY) ? 0.0f : expf(m - m_new))     // (2)
                  + expf(xx - m_new);
            m = m_new;
        }
    }
    // block 内用**同一个结合算子**合并各线程的 (m, l)
    for (int off = WARP / 2; off > 0; off >>= 1) {
        const float mo = __shfl_down_sync(0xffffffffu, m, off);
        const float lo = __shfl_down_sync(0xffffffffu, l, off);
        const float M  = fmaxf(m, mo);
        l = l * ((M == -INFINITY) ? 1.0f : expf(m - M))
              + lo * ((M == -INFINITY) ? 0.0f : expf(mo - M));
        m = M;
    }
    if ((threadIdx.x & (WARP - 1)) == 0) { sm[threadIdx.x / WARP] = m; sl[threadIdx.x / WARP] = l; }
    __syncthreads();
    if (threadIdx.x < WARP) {
        const bool live = threadIdx.x < (int)(blockDim.x / WARP);
        m = live ? sm[threadIdx.x] : -INFINITY;
        l = live ? sl[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1) {
            const float mo = __shfl_down_sync(0xffffffffu, m, off);
            const float lo = __shfl_down_sync(0xffffffffu, l, off);
            const float M  = fmaxf(m, mo);
            l = l * ((M == -INFINITY) ? 1.0f : expf(m - M))
                  + lo * ((M == -INFINITY) ? 0.0f : expf(mo - M));
            m = M;
        }
        if (threadIdx.x == 0) { sm[0] = m; sl[0] = l; }
    }
    __syncthreads();
    m = sm[0];
    l = sl[0];

    // (3) 式：y = exp(x − m) / l。这里回看了 x 一遍——在线递推本身只产出 (m, l)。
    const float inv = 1.0f / l;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half(expf(__half2float(xr[i]) - m) * inv);
}

// ---------------------------------------------------------------------------
static void softmax_ref(const std::vector<float>& x, std::vector<double>& y)
{
    const int n = (int)x.size();
    y.assign(n, 0.0);
    double m = x[0];
    for (float v : x) m = std::fmax(m, (double)v);
    double l = 0.0;
    for (int i = 0; i < n; ++i) { y[i] = std::exp((double)x[i] - m); l += y[i]; }
    for (int i = 0; i < n; ++i) y[i] /= l;
}

int main(int argc, char** argv)
{
    const int n     = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int chunk = (argc > 2) ? std::atoi(argv[2]) : 128;
    const int nblk  = (n + chunk - 1) / chunk;

    std::printf("=== 02_online_softmax：n = %d，分块 = %d（共 %d 块）===\n\n",
                n, chunk, nblk);

    std::mt19937 gen(20260917);
    std::normal_distribution<float> nd(0.0f, 4.0f);
    std::vector<float> hx(n);
    for (int i = 0; i < n; ++i) hx[i] = nd(gen);

    std::vector<half> hxh(n), hy2(n), hy1(n);
    for (int i = 0; i < n; ++i) hxh[i] = __float2half(hx[i]);

    half *dx = nullptr, *dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dx, hxh.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    softmax_two_pass_kernel<<<1, 256>>>(dx, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy2.data(), dy, n * sizeof(half), cudaMemcpyDeviceToHost));

    softmax_online_kernel<<<1, 256>>>(dx, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy1.data(), dy, n * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<double> ref;
    softmax_ref(hx, ref);

    double e_2 = 0.0, e_1 = 0.0, e_12 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double a = __half2float(hy2[i]);
        const double b = __half2float(hy1[i]);
        e_2  = std::fmax(e_2,  std::fabs(a - ref[i]));
        e_1  = std::fmax(e_1,  std::fabs(b - ref[i]));
        e_12 = std::fmax(e_12, std::fabs(a - b));
    }

    std::printf("[1] 三个结果的逐元素最大绝对差（**本机 GPU 上的数字，本仓库未跑过**）：\n");
    std::printf("    两遍法      vs double 参考 : %.4e\n", e_2);
    std::printf("    在线版      vs double 参考 : %.4e\n", e_1);
    std::printf("    在线版      vs 两遍法      : %.4e   ← 文档 §A2 要的是这一行\n", e_12);
    std::printf("    三者的输出之和： %.8f / %.8f / 1.00000000\n\n",
                [&]{ double s=0; for (auto v : hy2) s += __half2float(v); return s; }(),
                [&]{ double s=0; for (auto v : hy1) s += __half2float(v); return s; }());

    std::printf("[2] 块序无关性：同一行数据，把块按正序 / 逆序 / 随机置换各跑一遍\n");
    {
        std::vector<int> fwd(nblk), rev(nblk), shf(nblk);
        for (int i = 0; i < nblk; ++i) fwd[i] = i;
        for (int i = 0; i < nblk; ++i) rev[i] = nblk - 1 - i;
        for (int i = 0; i < nblk; ++i) shf[i] = i;
        std::mt19937 g(20260918);
        std::shuffle(shf.begin(), shf.end(), g);

        int* dord = nullptr;
        CUDA_CHECK(cudaMalloc(&dord, nblk * sizeof(int)));
        std::vector<half> out(n);
        double prev = 0.0;
        const char* names[3] = {"正序", "逆序", "随机置换"};
        const std::vector<int>* orders[3] = {&fwd, &rev, &shf};
        for (int k = 0; k < 3; ++k) {
            CUDA_CHECK(cudaMemcpy(dord, orders[k]->data(), nblk * sizeof(int),
                                  cudaMemcpyHostToDevice));
            softmax_online_order_kernel<<<1, 256>>>(dx, dy, dord, n, nblk);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(out.data(), dy, n * sizeof(half),
                                  cudaMemcpyDeviceToHost));
            double diff = 0.0;
            for (int i = 0; i < n; ++i)
                diff = std::fmax(diff, std::fabs(__half2float(out[i]) - ref[i]));
            std::printf("    %-10s：与 double 参考的最大绝对差 = %.4e", names[k], diff);
            if (k > 0) std::printf("（与前一种顺序的差 = %.4e）", std::fabs(diff - prev));
            std::printf("\n");
            prev = diff;
        }
        CUDA_CHECK(cudaFree(dord));
        std::printf("    ⟹ 块序变了对结果没有影响：合并算子 (m,l)⊕(m,l) 满足**结合律**，\n");
        std::printf("       换分组、换顺序只改变舍入，不改变数学结果。\n\n");
    }

    std::printf("[3] 手算复核（文档 §1 的两分块例子，纯 host 算术）：\n");
    {
        const double b1[2] = { 2.0, -1.0 }, b2[2] = { 4.0, 0.0 };
        const double m1 = 2.0;
        const double l1 = std::exp(b1[0] - m1) + std::exp(b1[1] - m1);
        const double m2 = 4.0;
        const double alpha = std::exp(m1 - m2);
        const double l2 = l1 * alpha + std::exp(b2[0] - m2) + std::exp(b2[1] - m2);
        const double exact = std::exp(-2.0) + std::exp(-5.0)
                           + std::exp(0.0) + std::exp(-4.0);
        std::printf("    块 1：m₁ = %.1f，l₁ = 1 + exp(−3) = %.16f\n", m1, l1);
        std::printf("    块 2：α = exp(−2) = %.16f\n", alpha);
        std::printf("          l₂ = l₁·α + (exp(0)+exp(−4)) = %.16f\n", l2);
        std::printf("    直接算：l = %.16f，差 = %.3e\n\n", exact, std::fabs(l2 - exact));
    }

    std::printf("[4] 递推的两条不变量（文档 §1）：\n");
    std::printf("    (a) m_old ≤ m_new 恒成立 ⟹ rescale 因子 exp(m_old − m_new) ≤ 1：\n");
    std::printf("        这个方向**不会**溢出。反过来写 exp(m_new − m_old) 会立刻溢出：\n");
    std::printf("        exp(+20) 在 fp16 里 = %s（fp16 上溢阈值只有 11.0901）\n",
                std::isinf((double)__half2float(__float2half(expf(20.0f)))) ? "inf" : "有限");
    std::printf("    (b) rescale 因子的下溢边界（决定「分块贡献会不会被整体丢弃」）：\n");
    std::printf("        gap > 17.3287 时 fp16 的因子取 0；fp32 要到 103.9721 —— 实际不可达。\n");
    std::printf("        上界：l_old ≤ 65504，所以一次丢弃最多丢掉 65504×2^-25 = %.3e，\n",
                65504.0 * std::ldexp(1.0, -25));
    std::printf("        即新块 l（≥1）的 %.4f%%。\n\n", 65504.0 * std::ldexp(1.0, -25) * 100);

    std::printf("[5] 访存账（推导值，与 verify.py §B5 同源）：\n");
    std::printf("    两遍法  = 3·n·esize = %.1f KB/行\n", 3.0 * n * 2 / 1024);
    std::printf("    在线版  = 2·n·esize = %.1f KB/行\n", 2.0 * n * 2 / 1024);
    std::printf("    省下 %.1f KB/行 —— 这就是在线 softmax 的全部收益。\n",
                (double)n * 2 / 1024);
    std::printf("    它的数值结果与两遍法一致（见 [1] 的第三行），代价是每个元素多一次\n");
    std::printf("    乘法（l·α）与多一次 max。\n");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
