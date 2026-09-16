// 01_softmax_stable.cu —— 减 max 的必要性：不改变数学结果，只改变可表示范围
//
// 配套文档：40-kernels/softmax-normalization/01-softmax-stability.md
//
// 本文件把文档 §1 的两套口径摊在明面上：
//
//   朴素做法      y_i = exp(x_i) / Σ exp(x_j)
//   稳定做法      y_i = exp(x_i − m) / Σ exp(x_j − m)，  m = max_j x_j
//
// 两者在实数域**恒等**（分子分母同乘 exp(−m)）。差别只在中间量 exp(x−m)
// 落在 [0, 1] 而 exp(x) 落在 [0, ∞)——前者永不溢出，后者会。
//
// 溢出/下溢阈值在**运行时**由 <cuda_fp16.h> 的常量现算，保证与文档 §B1
// 打印的是同一批数：
//
//   fp16 上溢  x > ln 65520 = 11.0901   （65520 = 65504 与 65536 的中点）
//   fp16 下溢  x < −25·ln2  = −17.3287
//   fp32 上溢  x > ln FLT_MAX = 88.7228
//   fp32 下溢  x < −150·ln2 = −103.9721
//
// ⚠ 注意 fp16 的**表观**上界 ln 65504 = 11.089866 不是阈值——阈值是它上方
//   半个 ulp 处的 ln 65520。两者差 2.44e-4，足以让一整段 logits 的判定反过来。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 01_softmax_stable
//     ./01_softmax_stable            # 默认 n = 4096，logit 幅度 8
//     ./01_softmax_stable 8192 12    # n = 8192，幅度 12
//
// 本程序输出里有两类数字，**必须分开看**：
//   · 阈值、字节数、算术强度 —— 推导值，与 `make verify` 的 §B1/§B2/§B5 一致；
//   · 墙钟耗时与逐元素误差 —— 是**那台机器上的运行结果**，本仓库没有跑过。

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
// 朴素内核：不算 max，直接 exp 再除
//
// 对应文档 §0/§4.1 的「起点」。它在 fp64 下是**对的**（见 doc §A1），
// 在 fp16 下只要有一个 logit 越过 11.0901 就整行失效：
// exp 得到 inf，sum 得到 inf，inf/inf = nan。
// ---------------------------------------------------------------------------
__global__ void softmax_naive_kernel(const half* __restrict__ x,
                                     half* __restrict__ y, int n)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;

    // 累加器用 float：这一步不是本文的重点，但它是文档 §4.3 的反面教材
    // （若把 l 用 half 累加，误差会到 1e-2 量级，见 verify.py §C3）。
    float l = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(__half2float(xr[i]));
        yr[i] = __float2half(e);            // ← 这里就可能溢出成 inf
        l += e;
    }
    // block 内树形归约（warp shuffle + shared memory）
    __shared__ float red[WARP];
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

    const float inv = 1.0f / l;             // l = inf 时这里得到 0，inf*0 = nan
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half(__half2float(yr[i]) * inv);
}

// ---------------------------------------------------------------------------
// 稳定内核：先归约出 max，再算 exp 与和
//
// 两遍结构。max 的归约和 l 的归约用同一套 warp shuffle + shared memory 代码，
// 所以它比朴素版**多一遍读写**——这是文档 §2 那笔账的另一半（见 §B5）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_max(float v, float* red)
{
    for (int off = WARP / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    if ((threadIdx.x & (WARP - 1)) == 0) red[threadIdx.x / WARP] = v;
    __syncthreads();
    if (threadIdx.x < WARP) {
        v = (threadIdx.x < (int)(blockDim.x / WARP)) ? red[threadIdx.x] : -INFINITY;
        for (int off = WARP / 2; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red)
{
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
    return red[0];
}

__global__ void softmax_stable_kernel(const half* __restrict__ x,
                                      half* __restrict__ y, int n)
{
    const int row = blockIdx.x;
    const half* xr = x + (size_t)row * n;
    half* yr = y + (size_t)row * n;
    __shared__ float red[WARP];

    // 第一遍：max
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        m = fmaxf(m, __half2float(xr[i]));
    m = block_reduce_max(m, red);

    // 第二遍：exp(x − m) 与它的和
    // x − m ≤ 0 ⟹ exp ∈ (0, 1]，**永不溢出**，与 logit 的绝对大小无关。
    float l = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(__half2float(xr[i]) - m);
        yr[i] = __float2half(e);
        l += e;
    }
    l = block_reduce_sum(l, red);

    const float inv = 1.0f / l;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2half(__half2float(yr[i]) * inv);
}

// ---------------------------------------------------------------------------
// CPU 参考（double，不减 max 也能算对——用来证明「减 max 不改变数学结果」）
// ---------------------------------------------------------------------------
static void softmax_ref(const std::vector<float>& x, std::vector<double>& y)
{
    const int n = (int)x.size();
    y.assign(n, 0.0);
    double l = 0.0;
    for (int i = 0; i < n; ++i) { y[i] = std::exp((double)x[i]); l += y[i]; }
    for (int i = 0; i < n; ++i) y[i] /= l;
}

// ---------------------------------------------------------------------------
// 主机侧：把文档 §1 的阈值在运行时重算一遍
//
// 这些数**不依赖 GPU**，纯 C++ 算术；与 verify.py §B1 的二分解一致。
// ---------------------------------------------------------------------------
static void print_thresholds()
{
    const double ln2 = std::log(2.0);
    // fp16：最大有限数 65504，它上方「本该是」65536，但 65536 在 fp16 里就是 inf
    const double f16_max = 65504.0;
    const double f16_mid = (f16_max + 65536.0) / 2.0;   // 65520
    const double f16_min_sub = std::ldexp(1.0, -24);    // 2^-24
    const double f32_max = (double)FLT_MAX;
    const double f32_min_sub = std::ldexp(1.0, -149);   // 2^-149

    std::printf("  %-28s %14.4f\n", "fp16 exp 上溢 (ln 65520)", std::log(f16_mid));
    std::printf("  %-28s %14.4f   ← **不是阈值**，只是表观值\n",
                "  对照 ln 65504", std::log(f16_max));
    std::printf("  %-28s %14.4f\n", "fp16 exp 下溢 (−25·ln2)",
                std::log(f16_min_sub / 2.0));
    std::printf("  %-28s %14.4f\n", "fp32 exp 上溢 (ln FLT_MAX)", std::log(f32_max));
    std::printf("  %-28s %14.4f\n", "fp32 exp 下溢 (−150·ln2)",
                std::log(f32_min_sub / 2.0));
    std::printf("\n");

    // fp16 域里不存在 11.0901 —— 它落在两个可表示点的格缝里
    const double step = std::ldexp(1.0, -7);            // [8,16) 档的 ulp
    const double below = 11.0859375, above = below + step;
    std::printf("  fp16 在 [8,16) 档的格步长 = 2^-7 = %.7f\n", step);
    std::printf("  这一段里可表示的点（取 4 个）：%.7f, %.7f, %.7f, %.7f\n",
                below, above, above + step, above + 2 * step);
    std::printf("  ln 65504 = %.9f 落在 (%.7f, %.7f) 的**格缝**里\n",
                std::log(f16_max), below, above);
    std::printf("  这个格缝的两个端点一个给 65248.0、一个给 inf —— 所以「fp16 的\n");
    std::printf("  上溢阈值到底是多少」必须先说清口径（见文档 §1 的两套口径）。\n");
}

int main(int argc, char** argv)
{
    const int n       = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const float scale = (argc > 2) ? (float)std::atof(argv[2]) : 8.0f;

    std::printf("=== 01_softmax_stable：n = %d，logit 幅度 = %.1f ===\n\n", n, scale);

    std::printf("[1] 本程序在运行时重算的 exp 阈值（推导值，与 make verify 的 §B1 同源）\n");
    print_thresholds();

    // ---- 造数据 ----
    std::mt19937 gen(20260916);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> hx(n);
    for (int i = 0; i < n; ++i) hx[i] = nd(gen);

    float hmax = hx[0];
    for (float v : hx) hmax = std::fmax(hmax, v);
    std::printf("[2] 输入：n = %d，logit ∈ [%.4f, %.4f]\n", n, *std::min_element(hx.begin(), hx.end()), hmax);

    const double th16 = std::log((65504.0 + 65536.0) / 2.0);
    int n_over = 0;
    for (float v : hx) if (v > th16) ++n_over;
    std::printf("    超过 fp16 上溢阈值 %.4f 的 logit 个数 = %d / %d（%.1f%%）\n",
                th16, n_over, n, 100.0 * n_over / n);
    std::printf("    ⟹ 朴素做法下，这 %d 个位置的 exp 是 inf，整行的分母也是 inf，\n", n_over);
    std::printf("       输出里会出现 %d 个 nan（inf/inf）。\n\n", n_over);

    std::printf("[3] 访存账（推导值，FP16 输入输出）：\n");
    const double bytes_2pass = 3.0 * n * 2;   // 读 2 遍 + 写 1 遍
    const double bytes_1pass = 2.0 * n * 2;   // 融合成一遍时的下界
    std::printf("    本程序的两遍结构 = 3·n·esize = %.1f KB / 行\n", bytes_2pass / 1024);
    std::printf("    融合成一遍的下界 = 2·n·esize = %.1f KB / 行（在线 softmax，见 02）\n",
                bytes_1pass / 1024);
    std::printf("    比值 = %.4f\n\n", bytes_1pass / bytes_2pass);

    // ---- 上机 ----
    std::vector<half> hxh(n), hy_naive(n), hy_stable(n);
    for (int i = 0; i < n; ++i) hxh[i] = __float2half(hx[i]);

    half *dx = nullptr, *dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dx, hxh.data(), n * sizeof(half), cudaMemcpyHostToDevice));

    softmax_naive_kernel<<<1, 256>>>(dx, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy_naive.data(), dy, n * sizeof(half), cudaMemcpyDeviceToHost));

    softmax_stable_kernel<<<1, 256>>>(dx, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy_stable.data(), dy, n * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<double> ref;
    softmax_ref(hx, ref);

    int naive_nan = 0, stable_nan = 0;
    double e_naive = 0.0, e_stable = 0.0;
    for (int i = 0; i < n; ++i) {
        const double a = __half2float(hy_naive[i]);
        const double b = __half2float(hy_stable[i]);
        if (std::isnan(a)) ++naive_nan;
        if (std::isnan(b)) ++stable_nan;
        e_naive  = std::fmax(e_naive,  std::fabs(a - ref[i]));
        e_stable = std::fmax(e_stable, std::fabs(b - ref[i]));
    }

    std::printf("[4] 运行结果（**本机 GPU 上的数字，本仓库未跑过**）：\n");
    std::printf("    朴素内核   : 输出 nan 个数 = %d，最大绝对误差 = %.4e\n",
                naive_nan, e_naive);
    std::printf("    稳定内核   : 输出 nan 个数 = %d，最大绝对误差 = %.4e\n",
                stable_nan, e_stable);
    std::printf("    输出之和   : 朴素 %.6f，稳定 %.6f（应为 1.0）\n\n",
                [&]{ double s = 0; for (auto v : hy_naive) s += __half2float(v); return s; }(),
                [&]{ double s = 0; for (auto v : hy_stable) s += __half2float(v); return s; }());

    std::printf("    稳定内核的 exp 最大值恒为 1.0（max 那一项 exp(0)），因此**永不溢出**；\n");
    std::printf("    它的误差只来自 fp16 存储：u = 2^-11 = %.3e。\n\n", std::ldexp(1.0, -11));

    std::printf("[5] 「减 max 不改变数学结果」的直接证据：\n");
    std::printf("    CPU double 参考实现**没有减 max**，它的输出与稳定内核的最大差\n");
    std::printf("    是 %.4e —— 两者在实数域恒等，差只来自格式。\n", e_stable);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
