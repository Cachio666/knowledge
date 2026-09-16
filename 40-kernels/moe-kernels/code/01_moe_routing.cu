// 01_moe_routing.cu —— MoE 一层要算多少：**两个 GEMM**，以及和一个等参数稠密模型的对照
//
// 配套文档：40-kernels/moe-kernels/01-moe-compute-pattern.md
//
// 主线：单 token 单专家（标准两层 FFN）= up + down = 4·d_model·d_ff；
// MoE 层合计 = 4·tokens·top_k·d_model·d_ff —— **与专家总数 E 无关**。
//
// 但「FLOPs 与 E 无关」不等于「代价与 E 无关」：权重流量 = E·2·d_model·d_ff·esize
// 随 E 线性上涨，于是权重流量的算术强度 2·T·top_k/(E·esize) 随 E **反比下降**，
// 拐点批大小 T* = I*·E·esize/(2·top_k) 随 E 线性上涨。
//
// 本文件三部分：
//   §1 `routing_kernel` —— 一个 warp 一个 token：算 gate logits → 取 top-k → 在 k 个上 softmax
//   §2 主机侧：FLOPs（系数 4 / 2 / 6）与等参数稠密模型的对照
//   §3 主机侧：权重流量、算术强度、拐点批大小随 E 的变化
//   §4 主机侧：两种路由分布的 tokens_per_expert 统计与「按最大填充」的浪费比
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，**未编译过**）：
//     make 01_moe_routing && ./01_moe_routing
//
// ⚠ 本文件 §2~§4 打印的全部是**推导值**（纯算术，与 `verify.py` 的 §B1/§B2/§B3 同式）；
//    §1 的 kernel 运行结果本仓库没有（没有 GPU）。
//    可复核处：`make verify` 的 §A1 / §B1 / §B2 / §B3 / §C4。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define WARP 32

// ---------------------------------------------------------------------------
// §0 场景参数（**本文选定的对照场景，不是模型规格**）
//    与 code/verify.py 的同名常量同值；见 specs.py 的 UNSOURCED。
// ---------------------------------------------------------------------------

namespace scen {
constexpr int    kDModel   = 4096;
constexpr int    kDFF      = 14336;
constexpr int    kExperts  = 8;
constexpr int    kTopK     = 2;
constexpr int    kTokens   = 4096;
constexpr double kEsize    = 2.0;      // FP16
}  // namespace scen

// 规格常量（与 code/specs.py 同值同来源）
namespace spec {
constexpr double kHbmBwGBpsH100   = 3350.0;   // Hopper Architecture In-Depth 表3（HBM3）
constexpr double kFp16DenseTFLOPS = 989.5;    // 同上（dense，不含 2:4 稀疏）
constexpr int    kSmCountH100     = 132;      // 同上
}  // namespace spec

// ---------------------------------------------------------------------------
// §1 routing kernel
//
// 形态：**一个 warp 一个 token**，lane e 负责专家 e（要求 E ≤ 32）。
//   a) 每个 lane 算自己那个专家的 logit = <x, Wg[e]>（d_model 次 FMA）
//   b) k 轮 warp 级 argmax：取到最大值后把它屏蔽掉，再取下一个
//      —— 并列时取**下标小的**（与 verify.py §A1 的口径一致）
//   c) 在选中的 k 个 logit 上做 softmax（**不是**在全 E 上做完再截断）
//
// 为什么是「先取 top-k 再 softmax」而不是「先 softmax 再取 top-k 不重归一」：
// 后者的权重之和 < 1，缺的量正是被丢掉的 E−k 个专家的 softmax 质量
// （verify.py §A1 会把这个缺口打出来）。两者**只有重归一那一步不同**。
//
// 访存：每个 lane 读 d_model 个权重（Wg 的那一行）；一个 warp 读 E·d_model 个。
// 这是本层里唯一「N 只有 E=8」的 GEMM —— 算术强度极低，**是纯访存受限**。
// ---------------------------------------------------------------------------

template <int E, int K>
__global__ void routing_kernel(const float* __restrict__ X,     // [T, D]
                               const float* __restrict__ Wg,    // [E, D]
                               int T, int D,
                               int* __restrict__ idx,           // [T, K]
                               float* __restrict__ w)           // [T, K]
{
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    if (warp >= T) return;

    // a) 本 lane 的专家 logit
    float logit = -CUDART_INF_F;
    if (lane < E) {
        const float* x  = X  + (size_t)warp * D;
        const float* we = Wg + (size_t)lane * D;
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) acc = fmaf(x[k], we[k], acc);
        logit = acc;
    }

    // b) k 轮 warp argmax
    float sel[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        float best_v = logit;
        int   best_e = lane;
#pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
            const int   oe = __shfl_xor_sync(0xffffffffu, best_e, off);
            // 值大的赢；值相等时下标小的赢 —— 这条使归约可交换、可结合
            if (ov > best_v || (ov == best_v && oe < best_e)) { best_v = ov; best_e = oe; }
        }
        sel[j] = best_v;
        if (lane == 0) idx[(size_t)warp * K + j] = best_e;
        if (lane == best_e) logit = -CUDART_INF_F;   // 屏蔽，下一轮取次大
    }

    // c) 只在选中的 k 个 logit 上 softmax（蝶形归约后每个 lane 都持有同一份 sel）
    float m = sel[0];
#pragma unroll
    for (int j = 1; j < K; ++j) m = fmaxf(m, sel[j]);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < K; ++j) { sel[j] = __expf(sel[j] - m); sum += sel[j]; }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < K; ++j) w[(size_t)warp * K + j] = sel[j] / sum;
    }
}

// ---------------------------------------------------------------------------
// §2 MoE 一层的计算量
// ---------------------------------------------------------------------------

static void report_flops() {
    const double d = scen::kDModel, f = scen::kDFF, T = scen::kTokens, k = scen::kTopK;
    const double E = scen::kExperts;

    const double per_pair = 4.0 * d * f;                    // up + down
    const double moe      = 4.0 * T * k * d * f;
    const double up_only  = 2.0 * T * k * d * f;            // 只算 up 投影（系数 2）
    const double swiglu   = 6.0 * T * k * d * f;            // gate + up + down（系数 6）
    const double dff_dense = E * f;                         // 等参数稠密模型的隐层宽度
    const double dense    = 4.0 * T * d * dff_dense;

    std::printf("--- §2 MoE 一层的 FLOPs（推导值）---\n");
    std::printf("d_model=%d  d_ff=%d（每个专家）  E=%d  top_k=%d  tokens=%d\n",
                scen::kDModel, scen::kDFF, scen::kExperts, scen::kTopK, scen::kTokens);
    std::printf("单 token 单专家 = up + down = 2·d·d_ff + 2·d_ff·d = 4·d·d_ff\n");
    std::printf("             = %.0f FLOPs = %.3f MFLOPs\n", per_pair, per_pair / 1e6);
    std::printf("MoE 层合计   = 4·tokens·top_k·d·d_ff = %.6e FLOPs = %.4f TFLOPs\n",
                moe, moe / 1e12);
    std::printf("  —— 上面这个式子里的 E 被约掉了：**与专家总数无关**\n");
    std::printf("对照① 只算 up（系数 2，初版漏掉 down 的那个数）= %.4f TFLOPs（%.0f×）\n",
                up_only / 1e12, moe / up_only);
    std::printf("对照② SwiGLU（三个投影，系数 6）              = %.4f TFLOPs（%.2f×）\n",
                swiglu / 1e12, swiglu / moe);
    std::printf("对照③ 等参数稠密模型：d_ff_dense = E·d_ff = %.0f\n", dff_dense);
    std::printf("      FFN 参数 = %.0f（MoE 也是 %.0f —— **相等**）\n",
                (double)(2.0 * d * dff_dense), (double)(E * 2.0 * d * f));
    std::printf("      一层 FLOPs = %.4f TFLOPs，比值 = %.4f = E/top_k\n",
                dense / 1e12, dense / moe);
    std::printf("每 token 口径：MoE %.3f MFLOPs vs 稠密 %.3f MFLOPs（比值 %.4f）\n\n",
                (4.0 * k * d * f) / 1e6, (4.0 * d * dff_dense) / 1e6,
                (4.0 * d * dff_dense) / (4.0 * k * d * f));
}

// ---------------------------------------------------------------------------
// §3 权重流量、算术强度、拐点批大小
// ---------------------------------------------------------------------------

static void report_roofline() {
    const double d = scen::kDModel, f = scen::kDFF, k = scen::kTopK, es = scen::kEsize;
    const double peak = spec::kFp16DenseTFLOPS * 1e12;
    const double bw   = spec::kHbmBwGBpsH100 * 1e9;
    const double ridge = peak / bw;                     // I*，规格值算出的推导值
    const double moe_flops = 4.0 * scen::kTokens * k * d * f;

    std::printf("--- §3 权重流量与算术强度（推导值）---\n");
    std::printf("H100 的 FP16 拐点 I* = %.1f TFLOPS ÷ %.0f GB/s = %.2f FLOP/B\n",
                spec::kFp16DenseTFLOPS, spec::kHbmBwGBpsH100, ridge);
    std::printf("权重流量的算术强度 = 4·T·k·d·d_ff ÷ (E·2·d·d_ff·esize) = 2·T·k/(E·esize)\n\n");
    std::printf("     %6s %14s %14s %10s %10s\n",
                "E", "每层权重(GB)", "FLOPs(T,定T)", "AI(FLOP/B)", "T* (token)");
    for (int E : {2, 8, 64, 256}) {
        const double wb = (double)E * 2.0 * d * f * es;
        const double ai = 2.0 * scen::kTokens * k / ((double)E * es);
        const double ts = ridge * (double)E * es / (2.0 * k);
        std::printf("     %6d %14.4f %14.4f %10.1f %10.0f\n",
                    E, wb / 1e9, moe_flops / 1e12, ai, ts);
    }
    std::printf("\n     ⚠ T* 与 E 成正比：**E 越大越难做到算力受限**。\n");
    std::printf("     E=8 时每层要 %.0f 个 token 才越过拐点；E=256 时要 %.0f 个。\n\n",
                ridge * 8.0 * es / (2.0 * k), ridge * 256.0 * es / (2.0 * k));

    std::printf("本场景（T=%d）的批大小扫描：\n", scen::kTokens);
    for (int T : {128, 512, 1024, 1181, 1182, 4096, 16384}) {
        const double ai = 2.0 * T * k / (8.0 * es);
        std::printf("      T=%6d → AI=%8.1f FLOP/B = %7.4f×I* → %s\n",
                    T, ai, ai / ridge, ai > ridge ? "算力受限" : "访存受限");
    }
    const double wb8 = 8.0 * 2.0 * d * f * es;
    std::printf("\n本场景的时间下界（只算一层 FFN，不含 attention 与通信）:\n");
    std::printf("  算力侧 = %.4f TFLOPs ÷ %.1f TFLOPS = %.4f ms\n",
                moe_flops / 1e12, spec::kFp16DenseTFLOPS, moe_flops / peak * 1e3);
    std::printf("  带宽侧 = %.4f GB ÷ %.0f GB/s      = %.4f ms\n\n",
                wb8 / 1e9, spec::kHbmBwGBpsH100, wb8 / bw * 1e3);
}

// ---------------------------------------------------------------------------
// §4 tokens_per_expert 的分布与「按最大填充」的浪费比
//
// 用与 verify.py 同式的确定性伪随机（这里用 LCG，便于在 C++ 里复现）
// ---------------------------------------------------------------------------

static unsigned g_seed = 20260917u;
static double urand() {
    g_seed = g_seed * 1103515245u + 12345u;
    return ((g_seed >> 8) & 0xFFFFFF) / 16777216.0;
}

static void report_distribution() {
    const int E = scen::kExperts, T = scen::kTokens, k = scen::kTopK;
    const int N = T * k;
    double p[256], c[256];

    std::printf("--- §4 tokens_per_expert：均匀 vs 幂律（推导值）---\n");
    std::printf("总 slot 数 = tokens × top_k = %d，8 个专家 → 均值 = %.1f\n", N, (double)N / E);
    std::printf("**两种分布的均值相同，差的全在形状上**\n\n");

    for (int mode = 0; mode < 2; ++mode) {
        double sum_p = 0.0;
        for (int e = 0; e < E; ++e) {
            p[e] = (mode == 0) ? 1.0 / E : 1.0 / (e + 1.0);   // 均匀 / Zipf α=1
            sum_p += p[e];
        }
        for (int e = 0; e < E; ++e) p[e] /= sum_p;
        for (int e = 0; e < E; ++e) c[e] = 0.0;
        for (int i = 0; i < N; ++i) {
            const double u = urand();
            double acc = 0.0;
            int pick = E - 1;
            for (int e = 0; e < E; ++e) { acc += p[e]; if (u < acc) { pick = e; break; } }
            c[pick] += 1.0;
        }
        double mn = c[0], mx = c[0], sum = 0.0;
        for (int e = 0; e < E; ++e) { mn = std::fmin(mn, c[e]); mx = std::fmax(mx, c[e]); sum += c[e]; }
        const double mean = sum / E;
        double var = 0.0;
        for (int e = 0; e < E; ++e) var += (c[e] - mean) * (c[e] - mean);
        var /= E;
        std::printf("  %-10s min=%6.0f max=%6.0f mean=%7.1f std=%8.2f max/mean=%5.2f"
                    "  按最大填充的浪费=%.2f%%\n",
                    mode == 0 ? "均匀" : "幂律 α=1", mn, mx, mean, std::sqrt(var), mx / mean,
                    (1.0 - mean / mx) * 100.0);
    }
    std::printf("\n  二项分布的理论 std = sqrt(N·p·(1−p)) = sqrt(%d × %.4f × %.4f) = %.4f\n",
                N, 1.0 / E, 1.0 - 1.0 / E,
                std::sqrt((double)N * (1.0 / E) * (1.0 - 1.0 / E)));
    std::printf("\n  ⚠ 「按最大填充」= 每个专家都按最大的那个专家的长度算一遍：\n");
    std::printf("     浪费比 = 1 − 均值/最大值。均匀下个位数百分比，幂律下过半。\n");
    std::printf("  ⚠ 上表是本文件自己的抽样（LCG），与 `verify.py` §B3 的具体数值不同\n");
    std::printf("     （不同的伪随机源），**结论一致**。正文引的是 verify.py 的数。\n\n");
}

// ---------------------------------------------------------------------------
// main：先跑 §1 的 kernel（需 GPU），再打 §2~§4 的推导表
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const int T = (argc > 1) ? std::atoi(argv[1]) : 64;   // kernel 演示用的 token 数
    const int D = 256;                                     // 演示用的 d_model（4 的倍数）

    std::printf("=== 01_moe_routing ===\n");
    std::printf("场景：d_model=%d d_ff=%d E=%d top_k=%d tokens=%d esize=%.0fB\n\n",
                scen::kDModel, scen::kDFF, scen::kExperts, scen::kTopK, scen::kTokens,
                scen::kEsize);

    // ---- §1 kernel 演示 ----
    float* dX = nullptr;
    float* dWg = nullptr;
    int*   dIdx = nullptr;
    float* dW = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, (size_t)T * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWg, (size_t)scen::kExperts * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dIdx, (size_t)T * scen::kTopK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dW, (size_t)T * scen::kTopK * sizeof(float)));

    {
        float* h = (float*)std::malloc((size_t)T * D * sizeof(float));
        unsigned s = 7u;
        for (size_t i = 0; i < (size_t)T * D; ++i) {
            s = s * 1103515245u + 12345u;
            h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(dX, h, (size_t)T * D * sizeof(float), cudaMemcpyHostToDevice));
        for (size_t i = 0; i < (size_t)scen::kExperts * D; ++i) {
            s = s * 1103515245u + 12345u;
            h[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(dWg, h, (size_t)scen::kExperts * D * sizeof(float),
                              cudaMemcpyHostToDevice));
        std::free(h);
    }

    const int threads = 128;                       // 4 个 warp / block
    const int blocks = (T * WARP + threads - 1) / threads;
    routing_kernel<scen::kExperts, scen::kTopK><<<blocks, threads>>>(dX, dWg, T, D, dIdx, dW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        int*   hIdx = (int*)std::malloc((size_t)T * scen::kTopK * sizeof(int));
        float* hW   = (float*)std::malloc((size_t)T * scen::kTopK * sizeof(float));
        CUDA_CHECK(cudaMemcpy(hIdx, dIdx, (size_t)T * scen::kTopK * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hW, dW, (size_t)T * scen::kTopK * sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("--- §1 routing kernel（kernel 运行结果，本仓库没有 GPU，未跑过）---\n");
        std::printf("前 8 个 token 的 top-%d 选择与权重：\n", scen::kTopK);
        for (int t = 0; t < 8 && t < T; ++t) {
            std::printf("  token %3d :", t);
            for (int j = 0; j < scen::kTopK; ++j)
                std::printf("  e=%d w=%.4f", hIdx[t * scen::kTopK + j], hW[t * scen::kTopK + j]);
            float sum = 0.0f;
            for (int j = 0; j < scen::kTopK; ++j) sum += hW[t * scen::kTopK + j];
            std::printf("   (权重和 = %.6f)\n", sum);
        }
        std::printf("\n  ↑ 权重和恒为 1 —— 这就是「先取 top-k 再 softmax」的口径。\n");
        std::printf("    若改成「全 E 上 softmax 后取 top-k 不重归一」，权重和会明显小于 1。\n\n");
        std::free(hIdx);
        std::free(hW);
    }
    CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dWg));
    CUDA_CHECK(cudaFree(dIdx)); CUDA_CHECK(cudaFree(dW));

    report_flops();
    report_roofline();
    report_distribution();
    return 0;
}
