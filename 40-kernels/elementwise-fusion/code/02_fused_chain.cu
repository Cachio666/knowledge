// 02_fused_chain.cu —— 融合 k 个算子：字节数、上界 k、与它的两个反例
//
// 配套文档：40-kernels/elementwise-fusion/02-fusion-and-traffic.md
//
// 主线：一条 k 个算子的单目链，**不融合**时每个算子各读一遍写一遍 → `2k·N·esize`；
// **融合**成一个 kernel 后只读输入、只写输出 → `2·N·esize`。比值是 `k`。
//
// 但这个上界 `k` 挂在一个前提上：**每个中间结果都往返 HBM**。
// 中间数组装得进 L2 时，不融合的流量本来就≈ 2N，收益趋于 1×。
// 于是真实比值落在 (1, k) 之间——这正是公开报道的端到端收益远低于 k 的原因。
//
// 本文件给出一条**反方向**的失败模式（文档 §4.1）：融合把中间值放进寄存器，
// 寄存器用量随链长线性上涨 → occupancy 下降 → 在途请求不足 → **反而更慢**。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，**未编译过**）：
//     make 02_fused_chain && ./02_fused_chain
//
// ⚠ 本文件打印的全部是**推导值**。本仓库没有运行过本文件。
//    可复核处：`make verify` 的 §A1 / §B3 / §C3。

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

#define WARP 32

// ---------------------------------------------------------------------------
// §0 规格常量（与 code/specs.py 同值同来源）
// ---------------------------------------------------------------------------

namespace spec {
constexpr int    kSmCountH100        = 132;
constexpr double kHbmBwGBpsH100      = 3350.0;
constexpr int    kRegsPerSm          = 65536;   // CUDA C++ Programming Guide §5.1
constexpr int    kRegGranularity     = 8;       // 寄存器按 8 个一档向上取整
constexpr int    kMaxRegsPerThread   = 255;     // 同上（CC 3.0 起）
constexpr int    kMaxWarpsPerSm      = 64;      // = 2048 线程
constexpr int    kMaxThreadsPerSm    = 2048;
constexpr int    kMaxBlocksPerSm     = 32;
constexpr int    kHbmLatencyCycles   = 600;     // 400~800 的中值（memory-system/01）
constexpr double kClockHzDerived     = 67.0e12 / (132.0 * 128.0 * 2.0);  // 推导值
}  // namespace spec

// ---------------------------------------------------------------------------
// §1 一条 k = 4 的链，两个版本
//
// 链：t = exp(x·1.5 + 0.25) → t = t·0.5 − 1.0 → y = tanh(t)
//     （多跑一层来凑够 k = 4 的形态；本文件的重点是**数据流**，不是算式）
// ---------------------------------------------------------------------------

__device__ __forceinline__ float op1(float x) { return expf(x * 1.5f + 0.25f); }
__device__ __forceinline__ float op2(float x) { return x * 0.5f - 1.0f; }
__device__ __forceinline__ float op3(float x) { return tanhf(x); }

// 非融合：三个算子各一个 kernel，中间结果**各写回一块完整的数组**。
// 流量 = 3 × (读 N + 写 N) = 6N·esize（对应 k = 3 的那一行；见 §4 表）。
__global__ void unfused_op1(const float* __restrict__ in, float* __restrict__ t, int n)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) t[i] = op1(in[i]); }

__global__ void unfused_op2(const float* __restrict__ t, float* __restrict__ u, int n)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) u[i] = op2(t[i]); }

__global__ void unfused_op3(const float* __restrict__ u, float* __restrict__ out, int n)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = op3(u[i]); }

// 融合：一个 kernel 把三个算子算完，中间值全程住在寄存器里。
// 流量 = 读 N + 写 N = 2N·esize。
__global__ void fused_chain(const float* __restrict__ in, float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = in[i];          // ← 唯一的一次读
    const float a = op1(x);         // 中间值 a、b 在**寄存器**里
    const float b = op2(a);
    out[i] = op3(b);                // ← 唯一的一次写
}

// ---------------------------------------------------------------------------
// §2 代价模型的纯 CPU 部分
// ---------------------------------------------------------------------------

// 模型甲：中间结果每次往返 HBM
static double unfused_bytes_modelA(long long n, int esize, int k) {
    return 2.0 * k * n * esize;
}
// 融合：读一次 + 写一次
static double fused_bytes(long long n, int esize) {
    return 2.0 * n * esize;
}
// 模型丙（另一种拓扑，k 输入 1 输出）：不融合 3(k−1)N·esize / 融合 (k+1)N·esize
static double unfused_bytes_modelC(long long n, int esize, int k) {
    return 3.0 * (k - 1) * n * esize;
}
static double fused_bytes_modelC(long long n, int esize, int k) {
    return (k + 1.0) * n * esize;
}

static int ceil_to(int x, int g) { return (x + g - 1) / g * g; }

// 链长 k 时的每线程寄存器估算与由此得到的 occupancy。
// ⚠ 这两个系数是**假设**（16 = 索引/地址/循环控制，12 = 每层算子同时活着的
//   中间值与常量）。实际值由编译器决定；本表用的是这条曲线的**形状**。
static int regs_for_chain(int k) { return 16 + 12 * k; }

static void occupancy_for_chain(int k, int block, int* regs_out, int* blocks_out,
                                int* warps_out)
{
    const int r = regs_for_chain(k);
    if (r > spec::kMaxRegsPerThread) {          // 超过硬上限 → 只能 spill
        *regs_out = r; *blocks_out = -1; *warps_out = -1;
        return;
    }
    const int r8 = ceil_to(r, spec::kRegGranularity);
    const int by_regs = spec::kRegsPerSm / (r8 * block);
    int blocks = by_regs;
    if (blocks > spec::kMaxBlocksPerSm) blocks = spec::kMaxBlocksPerSm;
    if (blocks > spec::kMaxThreadsPerSm / block) blocks = spec::kMaxThreadsPerSm / block;
    *regs_out = r8; *blocks_out = blocks; *warps_out = blocks * (block / WARP);
}

// Little 定律：填满带宽所需的在途请求数（每 SM）
static double needed_inflight_requests(int bytes_per_warp_request)
{
    const double ns_per_cycle = 1e9 / spec::kClockHzDerived;
    const double infl_bytes = spec::kHbmBwGBpsH100 * 1e9 *
                              (spec::kHbmLatencyCycles * ns_per_cycle * 1e-9);
    return infl_bytes / spec::kSmCountH100 / bytes_per_warp_request;
}

// ---------------------------------------------------------------------------
// §3 主机侧
// ---------------------------------------------------------------------------

static void correctness_check(int n)
{
    std::printf("§3 两个版本的正确性（与本文件自己的 CPU 参考实现对照）\n");

    float* h_in  = (float*)std::malloc(sizeof(float) * n);
    float* h_out = (float*)std::malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i)
        h_in[i] = -3.0f + 6.0f * ((float)i / (float)n);

    float *d_in = nullptr, *d_out = nullptr, *d_t = nullptr, *d_u = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_t,   sizeof(float) * n));   // 中间张量 1
    CUDA_CHECK(cudaMalloc(&d_u,   sizeof(float) * n));   // 中间张量 2
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (n + block - 1) / block;

    // 不融合：三趟 kernel，两个中间张量各落一份完整的显存
    unfused_op1<<<grid, block>>>(d_in, d_t, n);
    unfused_op2<<<grid, block>>>(d_t, d_u, n);
    unfused_op3<<<grid, block>>>(d_u, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));

    // 融合：一趟 kernel，中间值不落显存
    float* h_fused = (float*)std::malloc(sizeof(float) * n);
    fused_chain<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_fused, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));

    int diff = 0;
    double max_rel = 0.0;
    for (int i = 0; i < n; ++i) {
        if (std::memcmp(&h_fused[i], &h_out[i], sizeof(float)) != 0) ++diff;
        const double d = std::fabs((double)h_fused[i] - (double)h_out[i]);
        const double r = d / std::fmax(1e-30, (double)std::fabs(h_out[i]));
        if (r > max_rel) max_rel = r;
    }
    std::printf("  融合版 vs 非融合版：逐位不同的元素数 = %d，最大相对误差 = %.3e → %s\n",
                diff, max_rel, (diff == 0) ? "**逐位相同**" : "有差异");
    std::printf("  ⚠ 逐位相同是**必然的**：逐元素算子没有跨元素依赖，融合只改数据流、\n");
    std::printf("    不改每个元素上的运算序列。这与归约类算子相反——后者的融合会\n");
    std::printf("    改变加法树形，结果不再逐位可复现（正文在 40-kernels/reduce-scan）。\n");
    std::printf("  ⚠ 但**本仓库没有跑过这两行**：上面是「应当如此」的推导，不是观测。\n\n");

    std::free(h_in); std::free(h_out); std::free(h_fused);
    CUDA_CHECK(cudaFree(d_in));  CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_t));   CUDA_CHECK(cudaFree(d_u));
}

int main()
{
    std::printf("============================================================\n");
    std::printf("02_fused_chain —— 融合 k 个算子的字节数与上界 k\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链、无 GPU）。\n");
    std::printf("  下面打印的是推导值，可用 `make verify` 的 §A1 / §B3 / §C3 复核。\n\n");

    correctness_check(1 << 20);

    // ---- §4 字节数与加速比 ----
    const long long N = 1LL << 24;      // 16777216 个 fp32 = 64 MiB
    const int esize = 4;

    std::printf("§4 融合 k 个算子的字节数与加速比（推导值，N = 2^24 = %lld）\n", N);
    std::printf("  模型甲（中间结果**每次往返 HBM**）\n");
    std::printf("    不融合 = 2k·N·esize；融合 = 2·N·esize；比值 = **k**\n");
    std::printf("  模型乙（中间结果**命中 L2**）\n");
    std::printf("    不融合 ≈ 2·N·esize（中间张量在 L2 里被写被读，不回 HBM）；比值 = **1**\n");
    std::printf("  模型丙（**另一种拓扑**：k 个输入 1 个输出，二叉链）\n");
    std::printf("    不融合 = 3(k−1)·N·esize；融合 = (k+1)·N·esize；比值 = 3(k−1)/(k+1)\n\n");
    std::printf("    %-3s %12s %10s %8s %8s %9s %11s\n",
                "k", "甲:不融合", "甲:融合", "甲比值", "乙比值", "丙比值", "甲时间 µs");
    for (int k = 1; k <= 6; ++k) {
        const double a_unf = unfused_bytes_modelA(N, esize, k);
        const double a_fus = fused_bytes(N, esize);
        const double t = a_unf / (spec::kHbmBwGBpsH100 * 1e9) * 1e6;
        const double rc = 3.0 * (k - 1) / (k + 1);
        std::printf("    %-3d %9.0f MiB %7.0f MiB %8.2f %8.2f %9.4f %11.2f\n",
                    k, a_unf / 1048576.0, a_fus / 1048576.0,
                    a_unf / a_fus, 1.0, rc, t);
    }
    std::printf("    （k = 1 时模型丙没有算子，比值为 0 属退化情形）\n");
    std::printf("    （模型丙的融合字节数**不同于甲**：(k+1)N·esize——它要把 k 个输入\n");
    std::printf("      都读进来。k=2：不融合 %.0f MiB、融合 %.0f MiB → 比值 %.2f）\n\n",
                unfused_bytes_modelC(N, esize, 2) / 1048576.0,
                fused_bytes_modelC(N, esize, 2) / 1048576.0,
                unfused_bytes_modelC(N, esize, 2) / fused_bytes_modelC(N, esize, 2));

    std::printf("  ⚠ 上界 **k** 挂在一个前提上：**每个中间结果都往返 HBM**。\n");
    std::printf("    中间张量装得进 L2 时，不融合的流量本来就≈ 2N·esize，收益趋于 1×。\n");
    std::printf("    真实比值落在 (1, k) 之间——所以**不能**写成「上界趋于 2」，\n");
    std::printf("    更不能写成「融合越多越快」。\n\n");

    // ---- §5 反方向：寄存器压力 ----
    std::printf("§5 融合链的寄存器压力与 occupancy（公式判定，不是硬件测量）\n");
    const int BLOCK = 256;
    const double need_scalar = needed_inflight_requests(128);   // 标量：每 warp 128 B
    const double need_vec    = needed_inflight_requests(512);   // float4：每 warp 512 B
    std::printf("  填满带宽所需的在途请求数（Little 定律，延迟 %d 周期）：\n",
                spec::kHbmLatencyCycles);
    std::printf("    标量（每 warp 请求 128 字节）：%.1f 条；float4（512 字节）：%.1f 条\n",
                need_scalar, need_vec);
    std::printf("  ⚠ 「每 warp 只挂一条在途请求」是**保守假设**；真实硬件允许若干条在飞。\n\n");
    std::printf("  模型：r(k) = 16 + 12·k（**含假设成分**，见文件头注释）；block = %d\n",
                BLOCK);
    std::printf("    %-3s %6s %7s %8s %9s %10s %12s %13s\n",
                "k", "r(k)", "取整8", "块/SM", "warp/SM", "occupancy",
                "≥标量需求?", "≥float4需求?");
    int k_s = 0, k_v = 0, k_spill = -1;
    for (int k = 1; k <= 20; ++k) {
        int r8, blocks, warps;
        occupancy_for_chain(k, BLOCK, &r8, &blocks, &warps);
        if (blocks < 0) {
            std::printf("    %-3d %6d %7s %8s %9s %10s %12s %13s\n",
                        k, regs_for_chain(k), ">255", "—", "—", "—", "spill", "spill");
            if (k_spill < 0) k_spill = k;
            continue;
        }
        const bool ok_s = warps >= need_scalar, ok_v = warps >= need_vec;
        if (ok_s) k_s = k;
        if (ok_v) k_v = k;
        std::printf("    %-3d %6d %7d %8d %9d %9.1f%% %12s %13s\n",
                    k, regs_for_chain(k), r8, blocks, warps,
                    warps * 100.0 / spec::kMaxWarpsPerSm,
                    ok_s ? "是" : "**否**", ok_v ? "是" : "**否**");
    }
    std::printf("\n  ★ 标量访存下，k = %d 起常驻 warp 数就低于填满带宽所需了；\n", k_s + 1);
    std::printf("  ★ 换 float4 后最后一条能过的链长是 k = %d；\n", k_v);
    std::printf("  ★ 而寄存器的**硬上限**要到 k = %d 才撞到（r = %d > %d，只能 spill）。\n",
                k_spill, regs_for_chain(k_spill), spec::kMaxRegsPerThread);
    std::printf("  → 限制融合链长的**不是**寄存器上限，是 occupancy。\n");
    std::printf("    寄存器既是中间值的家，也是买「在途深度」的货币——融合省下的 HBM\n");
    std::printf("    流量，和融合吃掉的寄存器，来自**同一个资源池**。这正是 §6.1 那条\n");
    std::printf("    「融合反而更慢」的机制。\n\n");

    // ---- §6 失败模式清单 ----
    std::printf("§6 失败模式清单（详见文档 §4）\n");
    std::printf("  4.1 融合导致寄存器压力上升 → occupancy 下降 → **反而更慢**（§5 已量化）。\n");
    std::printf("  4.2 把上界 k 当成一定会拿到的收益 → 忽略「中间结果要往返 HBM」这个前提。\n");
    std::printf("  4.3 手工融合时把语义耦合进 kernel → 链改一处要改整个 kernel（回链 60-compilers）。\n");
    std::printf("  4.4 融合后中间张量不再落盘 → 调试时拿不到中间值，排错成本上升。\n");
    std::printf("  ⚠ 本节的全部数字都是**推导值**，本仓库没有在 GPU 上核对过。\n");

    return 0;
}
