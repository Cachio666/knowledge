// 01_elementwise_bound.cu —— 逐元素算子的访存上界与有效带宽
//
// 配套文档：40-kernels/elementwise-fusion/01-memory-bound-limit.md
//
// 主线：一个单目逐元素算子，处理 N 个元素**最少**要往返 HBM `2N·esize` 字节
// ——读一次、写一次，两项都省不掉（就地写法也一样）。把这个数代进 roofline
// 就得到可达时间的下界；而**分母**是「标称带宽」还是「可达带宽」，决定了
// 这个下界是乐观的还是有意义的。
//
// 本文件给出两个 kernel（就地 / 非就地）与一条**定量**的对照表：
//   · 用 `N·esize` 而不是 `2N·esize` 估算，会乐观整整一倍；
//   · 小规模下时间下界根本不是访存，而是 kernel 启动（固定成本）。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，**未编译过**）：
//     make 01_elementwise_bound && ./01_elementwise_bound
//
// ⚠ 本文件打印的全部是**推导值**（由下面的公式与规格常量算出），
// 不是硬件测量。本仓库没有运行过本文件。可复核处：`make verify` 的 §B1 / §B2 / §C1。

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

// ---------------------------------------------------------------------------
// §0 规格常量（与 code/specs.py 同值同来源；本文件不重新溯源）
// ---------------------------------------------------------------------------

namespace spec {
constexpr int   kSmCountH100      = 132;        // Hopper Architecture In-Depth 表3
constexpr double kHbmBwGBpsH100   = 3350.0;     // HBM3，同上（**标称**带宽）
constexpr double kFp32TflopsH100  = 67.0;       // 非 tensor core 的 FP32 向量峰值
constexpr double kLaunchOverheadUsLo = 3.0;     // **引用值**：外部对空 kernel 启动
constexpr double kLaunchOverheadUsHi = 5.0;     // 开销的对照测量，本仓库不复核口径
constexpr double kAchievableGapLoPct = 5.0;     // 可达带宽相对标称的缺口下界
constexpr double kAchievableGapHiPct = 15.0;    // 上界（见文档 §1 的口径说明）
}  // namespace spec

// ---------------------------------------------------------------------------
// §1 两个逐元素 kernel：非就地与就地
//
// 两者的 HBM 流量**完全相同**（都是读一次 + 写一次 = 2N·esize）——
// 就地省下来的是**容量**（输出缓冲不必另占一份）与**地址足迹**
// （触及的 line 数减半），不是流量。这一点是文档 §4.4 那条失败模式的核心。
// ---------------------------------------------------------------------------

__global__ void map_out_of_place(const float* __restrict__ in,
                                 float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);            // 读 in[i] 一次、写 out[i] 一次
}

__global__ void map_in_place(float* __restrict__ buf, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = expf(buf[i]);           // 同样是**读一次 + 写一次**
}

// ---------------------------------------------------------------------------
// §2 两个纯 CPU 的参考实现，只用来给下面的表算数
// ---------------------------------------------------------------------------

// 最少往返 HBM 的字节数：读 N·esize + 写 N·esize
static double min_hbm_bytes(long long n, int esize) { return 2.0 * n * esize; }

// 代入 roofline：t = 字节 ÷ (η · 标称带宽)
static double bound_us(long long n, int esize, double eta)
{
    return min_hbm_bytes(n, esize) / (eta * spec::kHbmBwGBpsH100 * 1e9) * 1e6;
}

// 算术强度：I = FLOPs / Bytes。Bytes 只统计必须往返 HBM 的那一份。
static double intensity(double flops_per_element, int esize)
{
    return flops_per_element / (2.0 * esize);
}

// ---------------------------------------------------------------------------
// §3 主机侧：跑一遍 kernel 做正确性对照，再打印全部推导表
// ---------------------------------------------------------------------------

static void correctness_check(int n)
{
    std::printf("§3 kernel 正确性（与本文件自己的 CPU 参考实现对照）\n");

    float* h_in  = (float*)std::malloc(sizeof(float) * n);
    float* h_out = (float*)std::malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i)
        h_in[i] = -20.0f + 40.0f * ((float)i / (float)n);   // tanh 型取值范围

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (n + block - 1) / block;
    map_out_of_place<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));

    // CPU 参考：同样的 expf，逐个元素
    double max_rel = 0.0;
    for (int i = 0; i < n; ++i) {
        const float ref = expf(h_in[i]);
        const double d = std::fabs((double)h_out[i] - (double)ref);
        const double r = d / std::fmax(1e-30, (double)std::fabs(ref));
        if (r > max_rel) max_rel = r;
    }
    // ⚠ 判据是相对误差，不是逐位相同：CPU 与 GPU 的 `expf` 是两套实现，
    //   最后一位不同属正常。这里要证的是「算法对」，不是「位模式一致」。
    std::printf("  非就地 y = exp(x)：N=%d，与 CPU 参考的最大相对误差 = %.3e → %s\n",
                n, max_rel, (max_rel < 1e-5) ? "通过" : "**不通过**");

    // 就地版本：结果应与非就地完全一致（同样的运算，只是地址相同）
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));
    map_in_place<<<grid, block>>>(d_in, n);
    CUDA_CHECK(cudaGetLastError());
    float* h_inplace = (float*)std::malloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemcpy(h_inplace, d_in, sizeof(float) * n, cudaMemcpyDeviceToHost));
    int diff = 0;
    for (int i = 0; i < n; ++i)
        if (std::memcmp(&h_inplace[i], &h_out[i], sizeof(float)) != 0) ++diff;
    std::printf("  就地 y = exp(y)：与非就地版本逐位不同的元素数 = %d → %s\n",
                diff, (diff == 0) ? "逐位相同" : "**不同**");
    std::printf("  → 两者算的东西完全一样；差别只在**内存占用**与**地址足迹**，\n");
    std::printf("    不在 HBM 流量（都是 2N·esize）。见下面的 §5 表。\n\n");

    std::free(h_in); std::free(h_out); std::free(h_inplace);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main()
{
    std::printf("============================================================\n");
    std::printf("01_elementwise_bound —— 逐元素算子的访存上界与有效带宽\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链、无 GPU）。\n");
    std::printf("  下面打印的是推导值，可用 `make verify` 的 §B1 / §B2 / §C1 复核。\n\n");

    correctness_check(1 << 20);

    // ---- §4 上界：2N·esize 与 roofline ----
    std::printf("§4 最少访存字节与 roofline 时间下界（推导值）\n");

    const long long N    = 1LL << 26;          // 67108864 个 fp32 = 256 MiB
    const int       esize = 4;                 // fp32
    const double    read_only  = (double)N * esize;
    const double    min_bytes  = min_hbm_bytes(N, esize);

    std::printf("  N = 2^26 = %lld 个 fp32（数据本身 %.1f MiB）\n",
                N, read_only / 1048576.0);
    std::printf("    只数「读」        N·esize   = %.0f B = %.1f MiB\n",
                read_only, read_only / 1048576.0);
    std::printf("    最少往返 HBM 字节 2N·esize  = %.0f B = %.1f MiB\n",
                min_bytes, min_bytes / 1048576.0);
    std::printf("  ⚠ 用 N·esize 估算会**乐观整整一倍**（比值 = %.2f）。\n",
                min_bytes / read_only);
    std::printf("  ⚠ 就地写法**不省这一半**：它同样是读一次 + 写一次。\n\n");

    std::printf("  代入 roofline（标称 B = %.0f GB/s）：\n", spec::kHbmBwGBpsH100);
    std::printf("    %-22s %14s %16s\n", "口径", "可达带宽 GB/s", "下界时间 µs");
    const double etas[3] = {
        1.0,
        1.0 - spec::kAchievableGapLoPct / 100.0,
        1.0 - spec::kAchievableGapHiPct / 100.0,
    };
    const char* names[3] = {
        "标称（乐观下界）",
        "可达（缺口 5%）",
        "可达（缺口 15%）",
    };
    for (int i = 0; i < 3; ++i)
        std::printf("    %-22s %14.1f %16.2f\n", names[i],
                    etas[i] * spec::kHbmBwGBpsH100, bound_us(N, esize, etas[i]));
    std::printf("  → 缺口的分解在 20-gpu-architecture/interconnect/04-measuring-bandwidth\n");
    std::printf("    与 memory-system/01-hierarchy-overview §2，**本主题不重推**。\n\n");

    // ---- §5 算术强度 ----
    std::printf("§5 算术强度：为什么逐元素算子永远在斜线段上（推导值）\n");
    const double istar = spec::kFp32TflopsH100 * 1e12 / (spec::kHbmBwGBpsH100 * 1e9);
    std::printf("  I*(FP32 向量) = %.1f TFLOPS ÷ %.0f GB/s = %.1f FLOP/byte\n",
                spec::kFp32TflopsH100, spec::kHbmBwGBpsH100, istar);
    std::printf("    %-26s %10s %8s %9s\n", "每元素运算", "FLOPs/元素", "I", "I/I*");
    struct { const char* name; double fpe; } ops[] = {
        {"y = a·x + b（1 次 FMA）", 2.0},
        {"y = exp(x)（计 1 次）",   1.0},
        {"y = tanh(x)（计 1 次）",  1.0},
    };
    for (auto& o : ops) {
        const double I = intensity(o.fpe, esize);
        std::printf("    %-26s %10.0f %8.3f %8.3f%%\n",
                    o.name, o.fpe, I, I / istar * 100.0);
    }
    std::printf("  I 与 N 无关——这是逐元素算子的定义性质。要翻到算力受限那一侧，\n");
    std::printf("  每元素运算次数必须超过 2·esize·I* = %.0f 次，差 2~3 个数量级。\n\n",
                2.0 * esize * istar);

    // ---- §6 launch 开销 vs 规模 ----
    std::printf("§6 小规模下的固定开销（推导值；启动开销是**引用值** %.1f~%.1f µs）\n",
                spec::kLaunchOverheadUsLo, spec::kLaunchOverheadUsHi);
    std::printf("    %-12s %10s %14s %14s %18s\n",
                "N", "数据", "2N·esize", "访存时间 µs", "启动/访存");
    const int ks[5] = {16, 18, 20, 24, 26};
    for (int k : ks) {
        const long long n = 1LL << k;
        const double by = min_hbm_bytes(n, esize);
        const double t  = bound_us(n, esize, 1.0);
        std::printf("    2^%-9d %7.1f MiB %14.0f %14.2f %8.0f%%~%-8.0f%%\n",
                    k, (double)n * esize / 1048576.0, by, t,
                    spec::kLaunchOverheadUsLo / t * 100.0,
                    spec::kLaunchOverheadUsHi / t * 100.0);
    }
    std::printf("  → N = 2^16 时一次启动的开销是访存时间的 19~32 倍；\n");
    std::printf("    N = 2^26 时同一笔开销只占 1.9%%~3.1%%。**小规模下下界不是访存。**\n\n");

    // ---- §7 失败模式清单 ----
    std::printf("§7 失败模式清单（详见文档 §4）\n");
    std::printf("  4.1 用 N·esize 估算 → 乐观一倍（本文件 §4 已量化）。\n");
    std::printf("  4.2 小规模下忽略 launch 开销 → N ≤ 2^18 时固定成本占主导（§6）。\n");
    std::printf("  4.3 把标称带宽当可达带宽 → 余量算错，凭空多出一块不存在的优化空间。\n");
    std::printf("  4.4 以为就地写法省一半流量 → 它省的是容量与地址足迹，不是流量（§3）。\n");
    std::printf("  ⚠ 本节的全部数字都是**推导值**，本仓库没有在 GPU 上核对过。\n");

    return 0;
}
