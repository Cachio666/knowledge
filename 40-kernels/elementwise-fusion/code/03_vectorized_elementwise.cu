// 03_vectorized_elementwise.cu —— 向量化访存、grid-stride loop、发射开销
//
// 配套文档：40-kernels/elementwise-fusion/03-vectorized-and-grid-stride.md
//
// 主线：向量化把**访存指令数**降到 1/4，但**事务数、sector 数、传输字节一个没动**
// ——它买到的东西不在发射侧，而在**并发需求**：一次 warp 请求从 128 字节变成
// 512 字节，填满同一份带宽所需的在途请求数就掉到 1/4。
//
// 三个 kernel 覆盖三种映射：
//   · 标量直接映射（每线程 1 个元素）
//   · float4 直接映射（每线程 4 个元素，要求 16 字节对齐）
//   · grid-stride loop（固定 grid，循环覆盖整个数组）
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，**未编译过**）：
//     make 03_vectorized_elementwise && ./03_vectorized_elementwise
//
// ⚠ 本文件打印的全部是**推导值**。本仓库没有运行过本文件。
//    可复核处：`make verify` 的 §A2 / §B4 / §B5 / §C2 / §C4。

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

// 本文件**不**包含任何未对齐的向量化访存。下面那个开关打开后才会去试试看，
// 它演示的是 §7 的 4.1 那条失败模式——**在真机上很可能直接以
// "misaligned address" 报错退出**，这不是 bug，是预期行为。
#define DEMO_MISALIGNED_ACCESS 0

// ---------------------------------------------------------------------------
// §0 规格常量（与 code/specs.py 同值同来源）
// ---------------------------------------------------------------------------

namespace spec {
constexpr int    kSmCountH100      = 132;
constexpr double kHbmBwGBpsH100    = 3350.0;
constexpr int    kSectorBytes      = 32;     // 搬运单位
constexpr int    kCacheLineBytes   = 128;    // 记账单位
constexpr int    kMaxVectorBytes   = 16;     // CUDA C++ 向量类型上限（float4）
constexpr int    kHalf2Bytes       = 4;      // __half2 = 两个 half 装进 32-bit 寄存器
constexpr int    kIssueSlotsPerSm  = 4;
constexpr int    kHbmLatencyCycles = 600;    // 400~800 的中值
constexpr double kClockHzDerived   = 67.0e12 / (132.0 * 128.0 * 2.0);  // 推导值
}  // namespace spec

// ---------------------------------------------------------------------------
// §1 三种映射
// ---------------------------------------------------------------------------

// (a) 标量直接映射：每线程 1 个元素，1 条 4 字节读 + 1 条 4 字节写
__global__ void map_scalar(const float* __restrict__ in, float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// (b) float4 直接映射：每线程 4 个元素，1 条 16 字节读 + 1 条 16 字节写。
//     ⚠ 要求 `in` 与 `out` 都按 16 字节对齐，且 n 已补齐到 4 的倍数。
__global__ void map_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n4)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float4 v = in[i];
        out[i] = make_float4(v.x * 2.0f + 1.0f,
                             v.y * 2.0f + 1.0f,
                             v.z * 2.0f + 1.0f,
                             v.w * 2.0f + 1.0f);
    }
}

// (c) grid-stride loop：固定 grid，每线程按 stride 反复取，直到越过 n。
//     适用场景是 grid 与问题规模脱钩（N 巨大、或想让 kernel 常驻）；
//     代价是**每次迭代**都要重算索引与边界。
__global__ void map_grid_stride(const float* __restrict__ in,
                                float* __restrict__ out, int n)
{
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = in[i] * 2.0f + 1.0f;
}

// ---------------------------------------------------------------------------
// §2 纯 CPU 的代价模型
// ---------------------------------------------------------------------------

// 访存指令数（lane 级）：每元素 1 读 + 1 写；向量化后每 4 个元素才 1 读 1 写
static double mem_instr_lane(long long n, int per_thread) {
    return 2.0 * (double)n / per_thread;
}
// 触及的 cache line 数 = L1→L2 的请求条数（记账单位，128 字节）
static double line_requests(long long n, int esize) {
    return (double)n * esize / spec::kCacheLineBytes;
}
// 搬运的 sector 数（搬运单位，32 字节）
static double sectors(long long n, int esize) {
    return (double)n * esize / spec::kSectorBytes;
}
// 填满带宽所需的在途请求数（每 SM）——Little 定律
static double needed_inflight_requests(int bytes_per_warp_request) {
    const double ns_per_cycle = 1e9 / spec::kClockHzDerived;
    const double infl = spec::kHbmBwGBpsH100 * 1e9 *
                        (spec::kHbmLatencyCycles * ns_per_cycle * 1e-9);
    return infl / spec::kSmCountH100 / bytes_per_warp_request;
}

// ---------------------------------------------------------------------------
// §3 主机侧
// ---------------------------------------------------------------------------

// ⚠ 判据用**相对误差**而不是逐位相同：nvcc 默认把 `a*b+c` 收缩成一条 FMA
//   （`-fmad=true`），CPU 上则是先乘后加——两者最后一位可能不同。
//   要逐位可比得两边都显式用 `fmaf` / `__fmaf_rn`，那是另一件事。
#if DEMO_MISALIGNED_ACCESS
// §7 的 4.1 那条失败模式的**可执行**版本：故意把基址偏移 1 个元素后再当成 float4。
// 打开 DEMO_MISALIGNED_ACCESS 后在真机上跑，预期是两种结果之一：
//   · 访问触发 "misaligned address" 运行时报错（多数情况）；
//   · 或者硬件容忍了这次访问，但**结果不对**（首元素漏掉、末尾越界）。
// 两条都属于「要么慢要么错」。注意 `reinterpret_cast<float4*>(d_buf + 1)` 的
// 地址 ≡ 4 (mod 16)，不满足 float4 的 16 字节对齐要求，是**未定义行为**。
static void demo_misaligned(float* d_buf, int n)
{
    const int n_vec = (n - 1 + 3) / 4;      // 从下标 1 开始分组 → 需要的组数
    map_vec4<<<(n_vec + 255) / 256, 256>>>(
        reinterpret_cast<const float4*>(d_buf + 1),
        reinterpret_cast<float4*>(d_buf + 1), n_vec);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());    // ← 未对齐访问的报错通常在这里才浮出来
}
#endif

static double max_rel_err(const float* got, const float* in, int n)
{
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ref = (double)in[i] * 2.0 + 1.0;
        const double d = std::fabs((double)got[i] - ref);
        const double r = d / std::fmax(1e-30, std::fabs(ref));
        if (r > worst) worst = r;
    }
    return worst;
}

static void correctness_check(int n)
{
    std::printf("§2 三种映射的正确性（与本文件自己的 CPU 参考实现的逐元素比对）\n");

    const int n4 = n / 4;                    // n 取 4 的倍数，先不引入尾部问题
    float* h_in  = (float*)std::malloc(sizeof(float) * n);
    float* h_out = (float*)std::malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) h_in[i] = (float)i * 1e-3f - 5.0f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    const int block = 256;

    map_scalar<<<(n + block - 1) / block, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));
    const double e_a = max_rel_err(h_out, h_in, n);
    std::printf("  (a) 标量直接映射：最大相对误差 = %.3e → %s\n",
                e_a, (e_a < 1e-6) ? "通过" : "**不通过**");

    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * n));
    map_vec4<<<(n4 + block - 1) / block, block>>>(
        reinterpret_cast<const float4*>(d_in),
        reinterpret_cast<float4*>(d_out), n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));
    const double e_b = max_rel_err(h_out, h_in, n);
    std::printf("  (b) float4 直接映射（cudaMalloc 的地址天然 256 字节对齐）："
                "最大相对误差 = %.3e → %s\n", e_b, (e_b < 1e-6) ? "通过" : "**不通过**");

    // (c) grid-stride：故意只开 32 个 block（远小于 n/block），逼出多轮迭代
    const int gs_blocks = 32;
    const int rounds = (n + gs_blocks * block - 1) / (gs_blocks * block);
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float) * n));
    map_grid_stride<<<gs_blocks, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost));
    const double e_c = max_rel_err(h_out, h_in, n);
    std::printf("  (c) grid-stride：最大相对误差 = %.3e → %s\n",
                e_c, (e_c < 1e-6) ? "通过" : "**不通过**");
    std::printf("      本次只用 %d 个 block（%d 个线程），每线程 %d 轮——"
                "用途是逼出多轮迭代，\n      不是为了快。\n",
                gs_blocks, gs_blocks * block, rounds);
    std::printf("  ⚠ 本仓库**没有运行过**上面三行；判据是「与 CPU 参考在容差内相等」，\n");
    std::printf("    不是「跑得快」。耗时不在本仓库的可交付范围内。\n\n");

#if DEMO_MISALIGNED_ACCESS
    std::printf("  [DEMO] 未对齐的 float4 访问（预期以 misaligned address 报错，\n");
    std::printf("         或静默给出错位的输出——两者都是 §4.1 说的失败行为）...\n");
    demo_misaligned(d_in, n);
    std::printf("  [DEMO] 上一行没有中止 → 这台机器容忍了它；请检查输出是否错位。\n\n");
#endif

    std::free(h_in); std::free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main()
{
    std::printf("============================================================\n");
    std::printf("03_vectorized_elementwise —— 向量化、grid-stride 与发射开销\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链、无 GPU）。\n");
    std::printf("  下面打印的是推导值，可用 `make verify` 的 §A2 / §B4 / §B5 / §C2 / §C4 复核。\n\n");

    correctness_check(1 << 20);

    // ---- §3 指令数与事务数 ----
    const long long N = 1LL << 26;
    const int esize = 4;

    std::printf("§3 向量化前后的指令条数与访存事务数（推导值，N = 2^26）\n");
    std::printf("    %-24s %16s %18s %8s\n", "", "标量(1 元素/线程)",
                "float4(4 元素/线程)", "比值");
    std::printf("    %-24s %16.0f %18.0f %8.2f\n", "访存指令（lane 级）",
                mem_instr_lane(N, 1), mem_instr_lane(N, 4),
                mem_instr_lane(N, 1) / mem_instr_lane(N, 4));
    std::printf("    %-24s %16.0f %18.0f %8.2f\n", "访存指令（warp 级）",
                mem_instr_lane(N, 1) / WARP, mem_instr_lane(N, 4) / WARP,
                mem_instr_lane(N, 1) / mem_instr_lane(N, 4));
    std::printf("    %-24s %16.0f %18.0f %8.2f\n", "事务（line/请求）数",
                line_requests(N, esize), line_requests(N, esize), 1.0);
    std::printf("    %-24s %16.0f %18.0f %8.2f\n", "搬运 sector 数",
                sectors(N, esize), sectors(N, esize), 1.0);
    std::printf("    %-24s %16.0f %18.0f %8.2f\n", "传输字节",
                (double)N * esize, (double)N * esize, 1.0);
    std::printf("  → 向量化把**指令数**降到 1/4；**事务数、sector 数、字节数一个没动**。\n");
    std::printf("    合并的判定准则（一次 warp 访存覆盖几个 sector）在\n");
    std::printf("    20-gpu-architecture/memory-system/02-coalescing-and-cache-line §1，\n");
    std::printf("    **本主题只代入、不重推**。\n\n");

    // ---- §4 发射侧 ----
    const double iss = spec::kIssueSlotsPerSm * spec::kSmCountH100 * spec::kClockHzDerived;
    const double t_mem = (double)N * 2 * esize / (spec::kHbmBwGBpsH100 * 1e9) * 1e6;
    std::printf("§4 发射侧的时间账（用**推导时钟** %.4f GHz，非规格值）\n",
                spec::kClockHzDerived / 1e9);
    std::printf("  全卡发射能力 = %d 槽/SM/周期 × %d SM × %.4f GHz = %.4e 条 warp 指令/s\n",
                spec::kIssueSlotsPerSm, spec::kSmCountH100,
                spec::kClockHzDerived / 1e9, iss);
    std::printf("  内存时间（N = 2^26，η = 1）= %.2f µs\n", t_mem);
    std::printf("    %-28s %14s %14s %10s\n", "口径", "warp 指令数", "发射时间 µs", "占内存");
    const char* names[4] = {"只数访存指令 标量", "只数访存指令 float4",
                            "每元素共 8 条指令 标量", "每元素共 8 条指令 float4"};
    for (int variant = 0; variant < 4; ++variant) {
        const int per_thread = (variant % 2 == 0) ? 1 : 4;
        const double instr_per_elem = (variant < 2) ? 2.0 : 8.0;   // 只数访存 / 含地址与循环
        const double wi = (double)N * instr_per_elem / per_thread / WARP;
        const double ti = wi / iss * 1e6;
        std::printf("    %-28s %14.0f %14.2f %9.2f%%\n",
                    names[variant], wi, ti, ti / t_mem * 100.0);
    }
    std::printf("  ⚠ 结论与直觉相反：**在大 N 的纯逐元素算子上，发射槽从来不是瓶颈**。\n");
    std::printf("    向量化真正买到的东西在 §5。\n\n");

    // ---- §5 并发需求 ----
    std::printf("§5 填满带宽需要多少在途请求——向量化的真正收益（推导值）\n");
    std::printf("  Little 定律：在途字节 = 带宽 × 延迟。延迟取 %d 周期。\n",
                spec::kHbmLatencyCycles);
    const double need_s = needed_inflight_requests(128);
    const double need_v = needed_inflight_requests(512);
    std::printf("    标量（每 warp 请求 128 字节）：%.1f 条\n", need_s);
    std::printf("    float4（每 warp 请求 512 字节）：%.1f 条\n", need_v);
    std::printf("  → 降到 1/4。上游 memory-system/02 §3 说的「约 60 个降到约 15 个」\n");
    std::printf("    就是这条账（同一条 Little 定律、同一个延迟量级）。\n");
    std::printf("  ⚠ 「每 warp 只挂一条在途请求」是**保守假设**；真实硬件允许若干条在飞。\n\n");

    // ---- §6 grid-stride ----
    std::printf("§6 grid-stride loop：适用场景与代价（推导值）\n");
    const int block = 256, per_thread = 4;
    const long long gs_blocks = (long long)spec::kSmCountH100 * 4;   // 每 SM 4 个 block
    const long long gs_threads = gs_blocks * block;
    std::printf("    直接映射：grid = ceil(N / (block × 每线程元素数))，随 N 一起涨\n");
    std::printf("    grid-stride：grid 固定为 %lld 个 block = %lld 个线程，每线程按 stride 循环\n",
                gs_blocks, gs_threads);
    std::printf("    %-10s %14s %14s %14s %16s\n",
                "N", "需要线程数", "直接映射启动", "gs 每线程轮数", "gs 空转线程占比");
    for (int k = 10; k <= 26; k += 4) {
        const long long n = 1LL << k;
        const long long needed = (n + per_thread - 1) / per_thread;
        const long long dir_threads = ((n + (long long)block * per_thread - 1) /
                                       ((long long)block * per_thread)) * block;
        const long long rounds = (needed + gs_threads - 1) / gs_threads;
        const long long idle = (gs_threads > needed) ? (gs_threads - needed) : 0;
        std::printf("    2^%-7d %14lld %14lld %14lld %15.2f%%\n",
                    k, needed, dir_threads, rounds,
                    (double)idle / (double)gs_threads * 100.0);
    }
    std::printf("  → 小 N（2^10、2^14）时 grid-stride **启动的线程远多于需要**，\n");
    std::printf("    多出来的线程算一次索引就退出；而直接映射只启动刚好够的线程。\n");
    std::printf("    这是「grid-stride 在小问题上更慢」的第一层原因；第二层是\n");
    std::printf("    每次迭代都要重算索引与边界（每轮多约 2 条整数指令 + 1 条比较）。\n");
    std::printf("    反过来，N 大到直接把映射的 grid 撑爆、或要让 kernel 常驻时，\n");
    std::printf("    grid-stride 才是唯一可行的形态。\n\n");

    // ---- §7 失败模式清单 ----
    std::printf("§7 失败模式清单（详见文档 §4）\n");
    std::printf("  4.1 向量化要求 %d 字节对齐：未对齐时要么错（misaligned address），\n",
                spec::kMaxVectorBytes);
    std::printf("      要么回退成逐元素而丢掉向量化的收益。\n");
    std::printf("  4.2 grid-stride 在小问题上比直接映射慢（§6 的「空转线程占比」）。\n");
    std::printf("  4.3 以为 `__half2` 就是向量化的终点——它只有 %d 字节，离 %d 字节的\n",
                spec::kHalf2Bytes, spec::kMaxVectorBytes);
    std::printf("      向量上限还差 %d 倍（半精度要一次搬 8 个 half 才等价于 float4）。\n",
                spec::kMaxVectorBytes / spec::kHalf2Bytes);
    std::printf("  ⚠ 本节的全部数字都是**推导值**，本仓库没有在 GPU 上核对过。\n");

#if DEMO_MISALIGNED_ACCESS
    std::printf("\n[注] DEMO_MISALIGNED_ACCESS 已打开——上面的 §2 里多做了一次\n");
    std::printf("     未对齐的 float4 访问。默认关闭是因为它是未定义行为。\n");
#endif

    return 0;
}
