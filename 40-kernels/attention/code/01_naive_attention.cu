// 01_naive_attention.cu —— attention 的计算形态，与「把 S 物化到 HBM」的代价
//
// 配套文档：40-kernels/attention/01-attention-as-a-kernel.md
//
// 这个内核**故意写成朴素形态**：把 S = scale·QKᵀ 完整物化到 global memory，
// 再对每一行做两遍 softmax，最后乘 V。它的作用是给出 01 篇 §1 那张算术强度表
// 的「朴素」那一列——**S 的写与读就是这一列和第二列（flash）的全部差额**。
//
// 三种写法（同一份数据，同一个内核模板）：
//   mode 0  两遍 + 物化 S（本文的朴素基准）
//   mode 1  同上，但**不减 max** —— 用来复现 §4.1 的溢出（fp16 上溢阈值 11.0901）
//   mode 2  只写回 S、不读回（量化「S 只写不算读」这条口径会让渐近 AI 从 d 变 2d）
//
// 访存账（verify.py §B1，单 head 单 batch，d=128，fp16，esize=2）：
//   朴素 = (3·S·d + S·d + 2·S²)·esize，AI = 4S²d / 字节 → 渐近 2d/esize = d
//   flash = 4·S·d·esize，               AI = S/esize = S/2       （无上界）
//
// ⚠ 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 01_naive_attention
//     ./01_naive_attention            # 默认 S=1024，d=128
//     ./01_naive_attention 4096 128   # S=4096，d=128
//
//   本文件里的数字分两类：**推导值**（字节、AI、与表对照的部分，可用
//   `make verify` 的 §B1/§B2 复核）与**本机运行结果**（逐元素差与耗时，
//   本仓库没有跑过，不对其数值负责）。

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
constexpr float FP16_OVERFLOW_LN = 11.0901f;   // 推导值，见 softmax-normalization/01 §1

// ---------------------------------------------------------------------------
// 朴素 attention：一个 block 处理一行 query（d 个元素），S 物化到 global
//
// 三遍结构，正是「物化 S」的代价来源：
//   ① 算 S 并写回 HBM（同时归约出本行 max）
//   ② 从 HBM 读回 S，算 exp(x − max) 与它们的和 l，把 exp 写回 HBM
//   ③ 从 HBM 读回 exp，乘 V 得到输出
//
// 不减 max 时（use_max=0）第 ② 步直接 exp(x)：fp16 下 x > 11.0901 就是 inf。
// ---------------------------------------------------------------------------
__global__ void attention_naive_kernel(const half* __restrict__ Q,
                                       const half* __restrict__ K,
                                       const half* __restrict__ V,
                                       half* __restrict__ S,      // S×S，物化的中间矩阵
                                       half* __restrict__ O,
                                       int seq, int d, float scale,
                                       int use_max)
{
    const int qi = blockIdx.x;
    const half* q = Q + (size_t)qi * d;
    half* srow = S + (size_t)qi * seq;
    const int tid = threadIdx.x;

    __shared__ float red[WARP];

    // ---- ① S = scale·q·Kᵀ，写回 HBM ----
    for (int j = tid; j < seq; j += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < d; ++t)                      // 向量化版本用 half2/fp16x8
            acc += __half2float(q[t]) * __half2float(K[(size_t)j * d + t]);
        srow[j] = __float2half(acc * scale);
    }

    // ---- 归约出本行 max（use_max=0 时取 0，等价于不减） ----
    float m = use_max ? -INFINITY : 0.0f;
    if (use_max) {
        for (int j = tid; j < seq; j += blockDim.x)
            m = fmaxf(m, __half2float(srow[j]));
        for (int off = WARP / 2; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if ((tid & (WARP - 1)) == 0) red[tid / WARP] = m;
        __syncthreads();
        if (tid < WARP) {
            m = (tid < (int)(blockDim.x / WARP)) ? red[tid] : -INFINITY;
            for (int off = WARP / 2; off > 0; off >>= 1)
                m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
            if (tid == 0) red[0] = m;
        }
        __syncthreads();
        m = red[0];
    }

    // ---- ② 从 HBM 读回 S，算 exp 与和 l，写回 ----
    float l = 0.0f;
    for (int j = tid; j < seq; j += blockDim.x) {
        const float e = expf(__half2float(srow[j]) - m);   // ← 溢出就发生在这一行
        srow[j] = __float2half(e);
        l += e;
    }
    for (int off = WARP / 2; off > 0; off >>= 1)
        l += __shfl_down_sync(0xffffffffu, l, off);
    if ((tid & (WARP - 1)) == 0) red[tid / WARP] = l;
    __syncthreads();
    if (tid < WARP) {
        l = (tid < (int)(blockDim.x / WARP)) ? red[tid] : 0.0f;
        for (int off = WARP / 2; off > 0; off >>= 1)
            l += __shfl_down_sync(0xffffffffu, l, off);
        if (tid == 0) red[0] = l;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];

    // ---- ③ 从 HBM 读回 exp，乘 V ----
    for (int t = tid; t < d; t += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq; ++j)
            acc += __half2float(srow[j]) * __half2float(V[(size_t)j * d + t]);
        O[(size_t)qi * d + t] = __float2half(acc * inv);
    }
}

// ---------------------------------------------------------------------------
// CPU double 参考：同一份数据、同一个算式，不受格式限制
// ---------------------------------------------------------------------------
static void attention_ref(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, int seq, int d, float scale,
                          std::vector<double>& O)
{
    O.assign((size_t)seq * d, 0.0);
    for (int i = 0; i < seq; ++i) {
        std::vector<double> s(seq), e(seq);
        double m = -1e300;
        for (int j = 0; j < seq; ++j) {
            double acc = 0.0;
            for (int t = 0; t < d; ++t)
                acc += (double)Q[(size_t)i * d + t] * (double)K[(size_t)j * d + t];
            s[j] = acc * scale;
            m = std::max(m, s[j]);
        }
        double l = 0.0;
        for (int j = 0; j < seq; ++j) { e[j] = std::exp(s[j] - m); l += e[j]; }
        for (int t = 0; t < d; ++t) {
            double acc = 0.0;
            for (int j = 0; j < seq; ++j) acc += e[j] * (double)V[(size_t)j * d + t];
            O[(size_t)i * d + t] = acc / l;             // 注意：参考里是归一化后的
        }
    }
}

int main(int argc, char** argv)
{
    const int seq = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int d   = (argc > 2) ? std::atoi(argv[2]) : 128;
    const float scale = 1.0f / std::sqrt((float)d);
    const float esize = 2.0f;                            // fp16

    std::printf("=== 01_naive_attention：S = %d，d = %d，fp16 ===\n\n", seq, d);

    std::printf("[1] 访存账（**推导值**，与 verify.py §B1 同源，单 head 单 batch）：\n");
    const double bytes_naive = (4.0 * seq * d + 2.0 * (double)seq * seq) * esize;
    const double bytes_writeonly = (4.0 * seq * d + 1.0 * (double)seq * seq) * esize;
    const double bytes_flash = 4.0 * seq * d * esize;
    const double flops = 4.0 * (double)seq * seq * d;
    std::printf("    朴素（写 S + 读 S）= (4·S·d + 2·S²)·esize = %.1f MB，"
                "AI = %.1f\n", bytes_naive / 1e6, flops / bytes_naive);
    std::printf("    朴素（只算 S 写回）= %.1f MB，AI = %.1f   ← 换个口径渐近值就翻倍\n",
                bytes_writeonly / 1e6, flops / bytes_writeonly);
    std::printf("    flash（S 不落 HBM）= 4·S·d·esize = %.2f MB，AI = %.1f\n",
                bytes_flash / 1e6, flops / bytes_flash);
    std::printf("    I*(FP16, H100) = 295.4 ⟹ 朴素 AI 是它的 %.1f%%（memory-bound）\n",
                flops / bytes_naive / 295.4 * 100);
    std::printf("    S 矩阵本身：%d² × %.0f B = %.1f MB（**单 head 单 batch**；\n",
                seq, esize, (double)seq * seq * esize / 1e6);
    std::printf("      × 32 head × 8 batch = %.1f GB）\n\n",
                (double)seq * seq * esize * 32 * 8 / 1e9);

    std::mt19937 gen(20260916);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<half> hQ((size_t)seq * d), hK((size_t)seq * d), hV((size_t)seq * d);
    std::vector<float> fQ((size_t)seq * d), fK((size_t)seq * d), fV((size_t)seq * d);
    for (size_t i = 0; i < fQ.size(); ++i) {
        fQ[i] = nd(gen); fK[i] = nd(gen); fV[i] = nd(gen);
        hQ[i] = __float2half(fQ[i]); hK[i] = __float2half(fK[i]); hV[i] = __float2half(fV[i]);
    }

    half *dQ, *dK, *dV, *dS, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dS, (size_t)seq * seq * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dO, (size_t)seq * d * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<double> ref;
    attention_ref(fQ, fK, fV, seq, d, scale, ref);

    std::vector<half> hO((size_t)seq * d);
    for (int use_max = 1; use_max >= 0; --use_max) {
        attention_naive_kernel<<<seq, 256>>>(dQ, dK, dV, dS, dO, seq, d, scale, use_max);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(half),
                              cudaMemcpyDeviceToHost));
        double worst = 0.0, norm = 0.0;
        int nonfinite = 0;
        for (size_t i = 0; i < hO.size(); ++i) {
            const float got = __half2float(hO[i]);
            if (!std::isfinite(got)) { ++nonfinite; continue; }
            worst = std::fmax(worst, std::fabs(got - ref[i]));
            norm  = std::fmax(norm, std::fabs(ref[i]));
        }
        std::printf("[2] %s：与 double 参考的最大绝对差 = %.4e（按范数 %.3e）",
                    use_max ? "减 max（正确）" : "**不减 max**", worst, worst / norm);
        if (nonfinite)
            std::printf("，其中 %d 个元素不是有限值（inf/nan）", nonfinite);
        std::printf("\n");
        if (!use_max)
            std::printf("    ⟹ 与 verify.py §C1 同一条：fp16 里 exp(x) 在 x > %.4f 时上溢，\n"
                        "       上溢之后 inf/inf = nan。**溢出是格式问题，不是算法问题**——\n"
                        "       同样一批 logits 在 fp32 里不会溢出。\n", FP16_OVERFLOW_LN);
    }

    std::printf("\n[3] 结论（文档 §1/§4）：\n");
    std::printf("    ① 朴素 AI 渐近于 d（= %.0f，与 S 无关），只到 I*(FP16) 的 %.1f%%；\n",
                esize == 2.0f ? (double)d : 0.0, 2.0 * d / esize / 295.4 * 100);
    std::printf("    ② 真正致命的是**绝对字节数**：S 矩阵是 O(S²)，S=%d 时单 head\n", seq);
    std::printf("       单 batch 就要 %.1f MB，× heads × batch 之后放不下。\n",
                (double)seq * seq * esize / 1e6);
    std::printf("    ③ 这两条必须分开讲：AI 的比值最多给 2.3× 的性能上界，\n");
    std::printf("       而字节数的降幅随 S 增长（flash 是 O(S·d)）。\n");

    CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dS)); CUDA_CHECK(cudaFree(dO));
    return 0;
}
