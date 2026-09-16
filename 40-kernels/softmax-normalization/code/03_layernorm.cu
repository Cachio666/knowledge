// 03_layernorm.cu —— LayerNorm / RMSNorm 与 Welford 单遍
//
// 配套文档：40-kernels/softmax-normalization/03-normalization-fusion.md
//
// 一行的定义（γ、β 是逐列的可学习参数）：
//
//   LayerNorm:  y_j = (x_j − μ) / sqrt(σ² + eps) · γ_j + β_j
//   RMSNorm:    y_j =  x_j      / sqrt(mean(x²) + eps) · γ_j
//               其中 mean(x²) 未去均值，故**不产生 μ**，也不需要 β
//
// 本文件对 LayerNorm 给出三种求 μ 与 σ² 的写法，差异全在数值上：
//
//   (a) 两遍法    先树形归约出 Σx（→ μ），第二遍再归约 Σ(x−μ)²
//   (b) Σx² − nμ² 一遍同时累加 Σx 与 Σx²，再相减   ← **数值上是灾难**
//   (c) Welford   一遍维护 (μ, M2)，分块后用 Chan 合并
//
// Welford 的两条递推（文档 §1 逐项推导）：
//
//   μ_n  = μ_{n−1} + (x_n − μ_{n−1}) / n
//   M2_n = M2_{n−1} + (x_n − μ_{n−1}) · (x_n − μ_n)
//   其中 M2_n = Σ_{i≤n} (x_i − μ_n)²，σ² = M2_n / n
//
// 分块合并（Chan / parallel variance），也就是 block 归约用的算子：
//
//   μ_ab = μ_a + (μ_b − μ_a) · n_b / n
//   M2_ab = M2_a + M2_b + (μ_b − μ_a)² · n_a·n_b / n
//
// ⚠ 一个必须写清的**限定**：单遍并不自动省 HBM 流量。如果一行装得进片上
//   （寄存器 / SMEM），两遍法的第二遍读的是片上副本，不产生额外 HBM 访问，
//   此时单遍省下的只是片上带宽与指令数；只有行长超过每 block 的 SMEM 预算时，
//   单遍才开始省**显存带宽**（账见文档 §B6）。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 03_layernorm
//     ./03_layernorm                # 默认 hidden = 8192，μ = 1e5
//     ./03_layernorm 4096 0         # hidden = 4096，μ = 0（常规激活）

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
// (a) 两遍法 LayerNorm：先归约 Σx，再归约 Σ(x−μ)²
// ---------------------------------------------------------------------------
__global__ void layernorm_two_pass_kernel(const half* __restrict__ x,
                                          const half* __restrict__ gamma,
                                          const half* __restrict__ beta,
                                          half* __restrict__ y,
                                          int n, float eps)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float red[WARP];

    // 第一遍：Σx（用 float 累加；这一层与本文的主题无关但必须交代）
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __half2float(xr[i]);
    for (int off = WARP / 2; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
    if ((threadIdx.x & (WARP - 1)) == 0) red[threadIdx.x / WARP] = s;
    __syncthreads();
    if (threadIdx.x < WARP) {
        s = (threadIdx.x < (int)(blockDim.x / WARP)) ? red[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    const float mean = red[0] / (float)n;

    // 第二遍：Σ(x−μ)²
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = __half2float(xr[i]) - mean;
        v = fmaf(d, d, v);
    }
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & (WARP - 1)) == 0) red[threadIdx.x / WARP] = v;
    __syncthreads();
    if (threadIdx.x < WARP) {
        v = (threadIdx.x < (int)(blockDim.x / WARP)) ? red[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    // ⚠ eps 放在 **sqrt 之内**：分界点在 σ = sqrt(eps)，不是 eps（见文档 §4.3）
    const float rstd = rsqrtf(red[0] / (float)n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half((__half2float(xr[i]) - mean) * rstd
                                 * __half2float(gamma[i]) + __half2float(beta[i]));
}

// ---------------------------------------------------------------------------
// (b) Σx² − n·μ²：**反例**。一遍同时累加 Σx 与 Σx²，再相减得到方差。
//
// 它比两遍法少一遍访存，看上去更划算——但 μ/σ 一大就崩：
// Σx² ≈ n·μ²，两个大数相减把有效位吃光，结果可以**变成负数**。
// 见 verify.py §C7：μ/σ ≥ 1e4 时误差 > 1（方差的量级都不对）。
// ---------------------------------------------------------------------------
__global__ void layernorm_sumsq_kernel(const half* __restrict__ x,
                                       const half* __restrict__ gamma,
                                       const half* __restrict__ beta,
                                       half* __restrict__ y,
                                       int n, float eps)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float red1[WARP];
    __shared__ float red2[WARP];

    float s = 0.0f, q = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float t = __half2float(xr[i]);
        s += t;
        q = fmaf(t, t, q);
    }
    for (int off = WARP / 2; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
        q += __shfl_down_sync(0xffffffffu, q, off);
    }
    if ((threadIdx.x & (WARP - 1)) == 0) { red1[threadIdx.x / WARP] = s; red2[threadIdx.x / WARP] = q; }
    __syncthreads();
    if (threadIdx.x < WARP) {
        const bool live = threadIdx.x < (int)(blockDim.x / WARP);
        s = live ? red1[threadIdx.x] : 0.0f;
        q = live ? red2[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
            q += __shfl_down_sync(0xffffffffu, q, off);
        }
        if (threadIdx.x == 0) { red1[0] = s; red2[0] = q; }
    }
    __syncthreads();
    const float mean = red1[0] / (float)n;
    // ← 灾难性抵消就发生在这一行
    const float var = red2[0] / (float)n - mean * mean;
    const float rstd = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half((__half2float(xr[i]) - mean) * rstd
                                 * __half2float(gamma[i]) + __half2float(beta[i]));
}

// ---------------------------------------------------------------------------
// (c) Welford 单遍 LayerNorm
//
// 每个线程先对自己那一段跑顺序 Welford，拿到 (μ_t, M2_t, n_t)；
// 再用 Chan 合并式在 block 内做**树形**归约。全程不构造 Σx² 这个中间量，
// 也就是没有相减——那个导致抵消的大数从来不存在。
// ---------------------------------------------------------------------------
struct Moment { float mean, M2, cnt; };

__device__ __forceinline__ Moment chan_combine(Moment a, Moment b)
{
    if (a.cnt == 0.0f) return b;
    if (b.cnt == 0.0f) return a;
    const float n = a.cnt + b.cnt;
    const float d = b.mean - a.mean;
    Moment r;
    r.cnt  = n;
    r.mean = a.mean + d * (b.cnt / n);
    r.M2   = a.M2 + b.M2 + d * d * (a.cnt * b.cnt / n);
    return r;
}

__global__ void layernorm_welford_kernel(const half* __restrict__ x,
                                         const half* __restrict__ gamma,
                                         const half* __restrict__ beta,
                                         half* __restrict__ y,
                                         int n, float eps)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float sm[WARP];
    __shared__ float sM[WARP];
    __shared__ float sc[WARP];

    Moment p{0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        p.cnt += 1.0f;
        const float d = v - p.mean;
        p.mean += d / p.cnt;               // μ_n  = μ_{n−1} + (x_n − μ_{n−1})/n
        p.M2   += d * (v - p.mean);        // M2_n = M2_{n−1} + (x_n − μ_{n−1})(x_n − μ_n)
    }

    // 树形合并：把 (μ, M2, n) 用 Chan 式两两并起来
    for (int off = WARP / 2; off > 0; off >>= 1) {
        Moment o;
        o.mean = __shfl_down_sync(0xffffffffu, p.mean, off);
        o.M2   = __shfl_down_sync(0xffffffffu, p.M2, off);
        o.cnt  = __shfl_down_sync(0xffffffffu, p.cnt, off);
        p = chan_combine(p, o);
    }
    if ((threadIdx.x & (WARP - 1)) == 0) {
        sm[threadIdx.x / WARP] = p.mean;
        sM[threadIdx.x / WARP] = p.M2;
        sc[threadIdx.x / WARP] = p.cnt;
    }
    __syncthreads();
    if (threadIdx.x < WARP) {
        const bool live = threadIdx.x < (int)(blockDim.x / WARP);
        p.mean = live ? sm[threadIdx.x] : 0.0f;
        p.M2   = live ? sM[threadIdx.x] : 0.0f;
        p.cnt  = live ? sc[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1) {
            Moment o;
            o.mean = __shfl_down_sync(0xffffffffu, p.mean, off);
            o.M2   = __shfl_down_sync(0xffffffffu, p.M2, off);
            o.cnt  = __shfl_down_sync(0xffffffffu, p.cnt, off);
            p = chan_combine(p, o);
        }
        if (threadIdx.x == 0) { sm[0] = p.mean; sM[0] = p.M2; sc[0] = p.cnt; }
    }
    __syncthreads();
    const float mean = sm[0];
    const float var  = sM[0] / sc[0];
    const float rstd = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half((__half2float(xr[i]) - mean) * rstd
                                 * __half2float(gamma[i]) + __half2float(beta[i]));
}

// ---------------------------------------------------------------------------
// RMSNorm：不减均值、不用 β，只归约 Σx²
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const half* __restrict__ x,
                               const half* __restrict__ gamma,
                               half* __restrict__ y, int n, float eps)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float red[WARP];

    float q = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float t = __half2float(xr[i]);
        q = fmaf(t, t, q);
    }
    for (int off = WARP / 2; off > 0; off >>= 1)
        q += __shfl_down_sync(0xffffffffu, q, off);
    if ((threadIdx.x & (WARP - 1)) == 0) red[threadIdx.x / WARP] = q;
    __syncthreads();
    if (threadIdx.x < WARP) {
        q = (threadIdx.x < (int)(blockDim.x / WARP)) ? red[threadIdx.x] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1)
            q += __shfl_down_sync(0xffffffffu, q, off);
        if (threadIdx.x == 0) red[0] = q;
    }
    __syncthreads();
    const float rstd = rsqrtf(red[0] / (float)n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half(__half2float(xr[i]) * rstd * __half2float(gamma[i]));
}

// ---------------------------------------------------------------------------
static void moments_ref(const std::vector<float>& x, double& mean, double& var)
{
    const size_t n = x.size();
    mean = 0.0;
    for (float v : x) mean += v;
    mean /= (double)n;
    var = 0.0;
    for (float v : x) var += ((double)v - mean) * ((double)v - mean);
    var /= (double)n;
}

int main(int argc, char** argv)
{
    const int   n   = (argc > 1) ? std::atoi(argv[1]) : 8192;
    const float mu  = (argc > 2) ? (float)std::atof(argv[2]) : 1.0e5f;
    const float eps = 1e-5f;
    const float sigma = 1.0f;

    std::printf("=== 03_layernorm：hidden = %d，μ = %.0f，σ = %.1f ===\n\n", n, mu, sigma);

    std::mt19937 gen(20260919);
    std::normal_distribution<float> nd(mu, sigma);
    std::vector<float> hx(n);
    std::vector<half> hxh(n), g(n), b(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = nd(gen);
        hxh[i] = __float2half(hx[i]);
        g[i] = __float2half(1.0f);
        b[i] = __float2half(0.0f);
    }

    double ref_mean, ref_var;
    moments_ref(hx, ref_mean, ref_var);
    std::printf("[1] 一行统计量的 fp64 参考值（推导值，纯 host 算术）：\n");
    std::printf("    mean = %.8f   var = %.8f   sigma = %.6f\n\n",
                ref_mean, ref_var, std::sqrt(ref_var));

    half *dx, *dg, *db, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dg, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dx, hxh.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, g.data(),   n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(),   n * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<half> out(n);
    const char* names[3] = {"两遍法", "Σx²−nμ²", "Welford 单遍"};
    for (int k = 0; k < 3; ++k) {
        if (k == 0)      layernorm_two_pass_kernel<<<1, 256>>>(dx, dg, db, dy, n, eps);
        else if (k == 1) layernorm_sumsq_kernel   <<<1, 256>>>(dx, dg, db, dy, n, eps);
        else             layernorm_welford_kernel <<<1, 256>>>(dx, dg, db, dy, n, eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dy, n * sizeof(half), cudaMemcpyDeviceToHost));

        // 用输出反推该内核「用到的」方差：y = (x−μ)·rstd ⟹ 对 y 求方差 = σ²·rstd²
        double omean, ovar;
        std::vector<float> of(n);
        for (int i = 0; i < n; ++i) of[i] = __half2float(out[i]);
        moments_ref(of, omean, ovar);
        std::printf("    %-12s：输出均值 %.6e，输出方差 %.6f（应 ≈ %.6f）\n",
                    names[k], omean, ovar, ref_var / (ref_var + eps));
    }

    std::printf("\n[2] 同一份数据的三个统计量（**本机 GPU 上的数字，本仓库未跑过**）：\n");
    std::printf("    两遍法与 Welford 应该一致；Σx²−nμ² 在 μ/σ = %.0f 时已经崩坏。\n",
                (double)mu / sigma);
    std::printf("    verify.py §A5 在 CPU 上把这条复现成了：μ/σ ≥ 1e4 时 Σx²−nμ² 的\n");
    std::printf("    相对方差误差 > 1（方差的量级都不对），且可以给出**负方差**。\n\n");

    for (int k = 0; k < 2; ++k) {
        if (k == 0) layernorm_welford_kernel <<<1, 256>>>(dx, dg, db, dy, n, eps);
        else        rmsnorm_kernel           <<<1, 256>>>(dx, dg, dy, n, eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), dy, n * sizeof(half), cudaMemcpyDeviceToHost));
        std::printf("    %-12s：输出之和 = %.6f，最小 = %.6f，最大 = %.6f\n",
                    k == 0 ? "LayerNorm" : "RMSNorm",
                    [&]{ double s=0; for (auto v : out) s += __half2float(v); return s; }(),
                    [&]{ double s=1e30; for (auto v : out) s = std::fmin(s, __half2float(v)); return s; }(),
                    [&]{ double s=-1e30; for (auto v : out) s = std::fmax(s, __half2float(v)); return s; }());
    }

    std::printf("\n[3] 访存账（推导值，与 verify.py §B6 同源；esize = 2 B）：\n");
    const double ne  = (double)n * 2;
    const double gb  = (double)n * 2;                 // γ 一份；β 同量级
    std::printf("    两遍法（x 不留在片上）: 读 x 两遍 + 写 y 一遍 + 读 γ/β = %.2f KB/行\n",
                (3 * ne + 2 * gb) / 1024);
    std::printf("    单遍  （x 留在片上）  : 读 x 一遍 + 写 y 一遍 + 读 γ/β = %.2f KB/行\n",
                (2 * ne + 2 * gb) / 1024);
    std::printf("    比值 = %.4f（忽略 γ/β 时恰为 3/2）\n", (3*ne+2*gb)/(2*ne+2*gb));
    std::printf("    ⚠ 这只在**一行放不进片上**时才是真的显存流量差；\n");
    std::printf("      一行 %d 个 fp16 = %.1f KB，每 SM 的 SMEM 上限 228 KB —— 放得下。\n",
                n, ne / 1024);
    std::printf("      所以对本程序的形状，单遍省的是片上带宽与指令数，不是 HBM 流量。\n\n");

    std::printf("[4] eps 的位置（文档 §4.3，推导值）：\n");
    std::printf("    内 sqrt(σ²+eps)：eps 从 σ = sqrt(eps) = %.6f 起才起作用\n", std::sqrt(eps));
    std::printf("    外 sqrt(σ²)+eps：eps 从 σ = eps = %.6e 起才起作用\n", eps);
    std::printf("    两个分界点相差 1/sqrt(eps) = %.4f 倍 —— σ 在这中间时两种实现\n",
                1.0 / std::sqrt(eps));
    std::printf("    给出的结果可以差 %.2f 倍（σ → 0 时最大）。\n",
                std::sqrt(eps) / eps);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dg));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
