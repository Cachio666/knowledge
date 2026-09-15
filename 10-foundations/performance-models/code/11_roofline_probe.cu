// roofline 探针：在本机 GPU 上实测峰值算力与峰值带宽。
//
// 构建：make 11_roofline_probe        （见同目录 Makefile）
// 运行：./11_roofline_probe
// 依赖：CUDA Toolkit >= 12.0，GPU 算力 >= sm_80
//
// 本程序做四件事：
//   1. 从 CUDA device 属性读出本机的 SM 数、时钟、显存总线宽度与频率
//   2. 实测 FP32 FMA 峰值算力（实测值，不是规格表抄来的）
//   3. 实测显存带宽（copy 与 read-only 两种口径）
//   4. 打印本机的 roofline 拐点，并把实测值与属性推出的理论值对比
//
// ---------------------------------------------------------------------------
// 口径总声明（每个测量点还会单独再声明一次）
//
// 本程序测的是**本机这一块 GPU 在这一次运行中的表现**，不是任何规格表上的
// 标称值。凡是标称值都会明确写「理论」或「规格」，并注明来源。
//
// 特别强调一条：**本程序测不到 Tensor Core 峰值。**
// 下面用的 FMA 密集 kernel 打的是标量 FP32 流水线，而 LLM 训练推理吃的
// Tensor Core 峰值（H100 SXM BF16 dense 989.5 TFLOPS）比 FP32 高一个数量级。
// 用本程序的 FP32 实测值去和 989.5 比是**错误的**。要测 Tensor Core 必须用
// mma / wgmma 指令或 cuBLAS，那超出本程序的范围。
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(expr)                                                            \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err_));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// 一次乘加 = 2 FLOPs（1 次乘 + 1 次加），全篇统一按此口径计数。
static const double FLOPS_PER_FMA = 2.0;

// cudaMalloc 的原型是 cudaMalloc(void**, size_t)。
// 直接写 cudaMalloc(&d_float_ptr, n) 需要把 float** 隐式转成 void**，
// 而 C++ **不允许**这种转换（void* 的通用性不递归到 void**）。
// 某些 nvcc 版本会放过，但宿主编译器一旦严格就会报
// "cannot convert 'float**' to 'void**'"。这里用临时 void* 中转到类型化指针，
// 比写 (void**)&p 更规范——后者在严格意义上是未定义行为。
template <typename T>
static cudaError_t device_malloc(T** ptr, std::size_t bytes) {
    void* tmp = nullptr;
    cudaError_t err = cudaMalloc(&tmp, bytes);
    *ptr = static_cast<T*>(tmp);
    return err;
}

// ---------------------------------------------------------------------------
// 计时：用 cudaEvent，不用 CPU 时钟——kernel 是异步的，CPU 侧计时会把
// 启动开销算进去，而且测不到真正的 kernel 时长。
// ---------------------------------------------------------------------------
struct Timer {
    cudaEvent_t beg{}, fin{};
    Timer() {
        CHECK(cudaEventCreate(&beg));
        CHECK(cudaEventCreate(&fin));
    }
    ~Timer() {
        cudaEventDestroy(beg);
        cudaEventDestroy(fin);
    }
    void start() { CHECK(cudaEventRecord(beg)); }
    // 返回毫秒
    double stop() {
        CHECK(cudaEventRecord(fin));
        CHECK(cudaEventSynchronize(fin));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, beg, fin));
        return static_cast<double>(ms);
    }
};

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------------------------------------------------------------------
// 峰值算力 kernel。
//
// 为什么要用**多个独立累加器**：
//   FMA 有大约 4 个周期的延迟。如果写成 acc = fma(acc, a, b) 一条链，
//   那么每 4 个周期才能发出下一条 FMA，流水线大部分时间是空的——
//   这时测到的是**延迟**（latency-bound），不是**吞吐**（throughput-bound）。
//   用 UNROLL 条互不依赖的链，就能让 4 个周期里塞进 4 条以上的 FMA，
//   把发射端口占满，这才是峰值。
//   UNROLL 取 16 远大于延迟周期数，留足余量。
//
// 为什么要写 if (s == magic) 这个几乎不可能成立的分支：
//   不加的话，编译器发现 out 从未被读取，会把整个循环当死代码删掉，
//   测出来的「峰值」是 0 耗时。这个分支让编译器必须把计算做完，
//   但实际几乎从不写内存，不会污染带宽测量。
// ---------------------------------------------------------------------------
template <int UNROLL>
__global__ void fma_peak_kernel(float* __restrict__ out, int iters, float a, float b) {
    float acc[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        acc[i] = static_cast<float>(threadIdx.x) * 1e-8f + static_cast<float>(i);
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            // __fmaf_rn 是显式的 fused multiply-add，保证不被编译器拆开
            // 或重新结合，FLOPs 计数才守得住。
            acc[i] = __fmaf_rn(acc[i], a, b);
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += acc[i];
    if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int UNROLL>
__global__ void fma_peak_kernel_f64(double* __restrict__ out, int iters, double a, double b) {
    double acc[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        acc[i] = static_cast<double>(threadIdx.x) * 1e-8 + static_cast<double>(i);
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) acc[i] = __fma_rn(acc[i], a, b);
    }
    double s = 0.0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) s += acc[i];
    if (s == 12345.678) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// 带宽 kernel。用 float4（128 位）访问，保证每个线程每次访存是一条完整的
// cache line 请求；用 grid-stride 循环，让访存地址在整个 buffer 上均匀铺开。
// ---------------------------------------------------------------------------
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        out[i] = in[i];
    }
}

__global__ void read_kernel(const float4* __restrict__ in, float* __restrict__ out, size_t n4) {
    float acc = 0.0f;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}

static void banner(const char* title) {
    std::printf("\n");
    for (int i = 0; i < 78; ++i) std::putchar('=');
    std::printf("\n%s\n", title);
    for (int i = 0; i < 78; ++i) std::putchar('=');
    std::printf("\n");
}

// ===========================================================================
int main() {
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, dev));

    banner("1. 本机设备：直接读 CUDA device 属性");

    // 注意属性的**正确名称**：
    //   显存总线宽度是 cudaDevAttrGlobalMemoryBusWidth，
    //   没有 cudaDevAttrMemoryBusWidth 这个属性（写错会编译不过）。
    //   cudaDevAttrMemoryClockRate 的单位是 **kHz**，不是 Hz。
    int sm_count = 0, sm_clock_khz = 0, mem_clock_khz = 0, bus_width_bits = 0, l2_bytes = 0;
    int cc_major = 0, cc_minor = 0;
    size_t mem_total = 0, mem_free = 0;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev));
    CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev));
    CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev));
    CHECK(cudaMemGetInfo(&mem_free, &mem_total));

    std::printf("\n  %-28s %s\n", "设备名", prop.name);
    std::printf("  %-28s sm_%d%d\n", "计算能力", cc_major, cc_minor);
    std::printf("  %-28s %d\n", "SM 数 (cudaDevAttrMultiProcessorCount)", sm_count);
    std::printf("  %-28s %d kHz = %.3f GHz", "SM 时钟 (cudaDevAttrClockRate)",
                sm_clock_khz, sm_clock_khz / 1e6);
    std::printf("   ← 属性单位是 kHz\n");
    std::printf("  %-28s %d kHz = %.3f GHz", "显存时钟 (cudaDevAttrMemoryClockRate)",
                mem_clock_khz, mem_clock_khz / 1e6);
    std::printf("   ← 属性单位是 kHz\n");
    std::printf("  %-28s %d bit\n",
                "显存总线宽度 (cudaDevAttrGlobalMemoryBusWidth)", bus_width_bits);
    std::printf("  %-28s %.1f MiB\n", "L2 缓存 (cudaDevAttrL2CacheSize)", l2_bytes / 1048576.0);
    std::printf("  %-28s %.2f GiB\n", "显存总量", mem_total / 1073741824.0);
    std::printf("  %-28s %.2f GiB\n", "当前可用显存", mem_free / 1073741824.0);

    banner("2. 理论值：由属性推算的峰值算力与峰值带宽");

    // 每 SM 每周期能发多少条 FP32 FMA。
    // 这是个**硬编码假设**，不是从属性读来的，所以单独声明：
    //   sm_80（A100，GA100） 每 SM 有 64 个 FP32 单元 → 64 FMA/周期
    //   sm_86 / sm_89 / sm_90 每 SM 有 128 个 FP32 单元 → 128 FMA/周期
    int fp32_lanes_per_sm = (cc_major == 8 && cc_minor == 0) ? 64 : 128;
    if (cc_major < 8) fp32_lanes_per_sm = 64;  // 更老的架构一律按 64 估

    double theo_fp32_flops = static_cast<double>(sm_count) * fp32_lanes_per_sm *
                             FLOPS_PER_FMA * (sm_clock_khz * 1e3);
    double theo_fp64_flops = theo_fp32_flops / 2.0;  // sm_80/sm_90：FP64 是 FP32 的 1/2

    // 理论带宽 = 显存时钟 × 总线宽度(字节) × 2（DDR，上升沿+下降沿各传一次）
    double theo_bw = (mem_clock_khz * 1e3) * (bus_width_bits / 8.0) * 2.0;

    // 属性为 0 时理论值会变成 0，后面的「达成率」会除零变成 inf。
    // 这两条不是防御性编程洁癖：cudaDevAttrMemoryClockRate 在部分平台确实报 0。
    const bool have_theo_flops = theo_fp32_flops > 0.0;
    const bool have_theo_bw = theo_bw > 0.0;
    if (!have_theo_flops) {
        std::printf("\n  警告：SM 时钟属性为 0，无法推算理论算力，下面达成率一列显示 n/a。\n");
    }
    if (!have_theo_bw) {
        std::printf("\n  警告：显存时钟或总线宽度属性为 0，无法推算理论带宽，"
                    "达成率一列显示 n/a。\n");
    }

    std::printf("\n  %-34s %.2f TFLOPS\n", "FP32 FMA 理论峰值", theo_fp32_flops / 1e12);
    std::printf("  %-34s %.2f TB/s\n", "显存理论带宽", theo_bw / 1e12);
    std::printf("\n  算式：\n");
    std::printf("    FP32 = SM 数 × 每 SM 的 FP32 单元数 × 2 FLOP/FMA × SM 时钟\n");
    std::printf("         = %d × %d × 2 × %.3e Hz = %.3e FLOP/s\n",
                sm_count, fp32_lanes_per_sm, sm_clock_khz * 1e3, theo_fp32_flops);
    std::printf("    带宽 = 显存时钟 × 总线宽度/8 × 2 (DDR)\n");
    std::printf("         = %.3e Hz × %d/8 B × 2 = %.3e B/s\n",
                mem_clock_khz * 1e3, bus_width_bits, theo_bw);

    std::printf("\n  【口径声明 · 理论值】\n");
    std::printf("  · 「每 SM 的 FP32 单元数」是硬编码假设（本机按 %d 算），\n",
                fp32_lanes_per_sm);
    std::printf("    CUDA 没有对应的 device 属性可读，架构换代时要人工核对。\n");
    std::printf("  · cudaDevAttrClockRate 是**当前** SM 时钟，可能因为功耗/温度\n");
    std::printf("    处于 boost 或降频状态，所以这个「理论值」本身就会漂移。\n");
    std::printf("  · cudaDevAttrMemoryClockRate 在部分平台上报的是数据率的一半或\n");
    std::printf("    直接报 0。若下面的实测带宽与该理论值差得离谱，先怀疑这个属性，\n");
    std::printf("    不要先怀疑测量代码。H100 SXM 的规格带宽是 3.35 TB/s。\n");

    // ---------------------------------------------------------------- 算力实测
    banner("3. 实测峰值算力（FP32 / FP64 FMA 密集 kernel）");

    constexpr int UNROLL = 16;
    constexpr int THREADS = 256;
    int blocks = sm_count * 32;  // 每 SM 32 个 block，确保有足够的 warp 填满发射端口
    int iters = 1 << 16;         // 65536 次外层迭代，让 kernel 跑到几十毫秒量级

    // FP32 与 FP64 各自一个输出指针。**类型必须各自对应**，
    // 复用同一个 float* 传给 double* 形参会编译不过。
    float* d_out32 = nullptr;
    double* d_out64 = nullptr;
    size_t out_elems = static_cast<size_t>(blocks) * THREADS;
    CHECK(device_malloc(&d_out32, out_elems * sizeof(float)));
    CHECK(device_malloc(&d_out64, out_elems * sizeof(double)));

    auto bench_fma_f32 = [&]() {
        Timer t;
        t.start();
        fma_peak_kernel<UNROLL><<<blocks, THREADS>>>(d_out32, iters, 1.000001f, 1e-7f);
        return t.stop();
    };
    auto bench_fma_f64 = [&]() {
        Timer t;
        t.start();
        fma_peak_kernel_f64<UNROLL><<<blocks, THREADS>>>(d_out64, iters, 1.000001, 1e-15);
        return t.stop();
    };

    std::vector<double> ms32, ms64;
    for (int rep = 0; rep < 5; ++rep) {
        bench_fma_f32();  // 预热：第一次包含上下文初始化与频率爬升，丢掉
        ms32.push_back(bench_fma_f32());
        ms64.push_back(bench_fma_f64());
    }
    CHECK(cudaGetLastError());

    double fma_count = static_cast<double>(blocks) * THREADS * iters * UNROLL;
    double flops = fma_count * FLOPS_PER_FMA;
    double t32 = median(ms32), t64 = median(ms64);
    double meas_fp32 = flops / (t32 * 1e-3);
    double meas_fp64 = flops / (t64 * 1e-3);

    std::printf("\n  规模：%d 个 block × %d 线程，每线程 %d 次外层迭代 × %d 路独立 FMA\n",
                blocks, THREADS, iters, UNROLL);
    std::printf("  总 FMA 次数 = %d × %d × %d × %d = %.3e\n", blocks, THREADS, iters, UNROLL,
                fma_count);
    std::printf("  总 FLOPs    = %.3e × %.0f = %.3e\n", fma_count, FLOPS_PER_FMA, flops);
    char r32[32], r64[32];
    if (have_theo_flops) {
        std::snprintf(r32, sizeof r32, "%.1f%%", meas_fp32 / theo_fp32_flops * 100);
        std::snprintf(r64, sizeof r64, "%.1f%%", meas_fp64 / theo_fp64_flops * 100);
    } else {
        std::snprintf(r32, sizeof r32, "n/a");
        std::snprintf(r64, sizeof r64, "n/a");
    }
    std::printf("\n  %-10s %12s %14s %14s %10s\n", "精度", "中位耗时(ms)", "实测 TFLOPS",
                "理论 TFLOPS", "达成率");
    std::printf("  %-10s %12.3f %14.2f %14.2f %10s\n", "FP32", t32, meas_fp32 / 1e12,
                theo_fp32_flops / 1e12, r32);
    std::printf("  %-10s %12.3f %14.2f %14.2f %10s\n", "FP64", t64, meas_fp64 / 1e12,
                theo_fp64_flops / 1e12, r64);
    std::printf("\n  5 次重复的 FP32 耗时：");
    for (double v : ms32) std::printf("%.3f ", v);
    std::printf("(取中位)\n");

    std::printf("\n  【口径声明 · 峰值算力】\n");
    std::printf("  · **实测的是标量 FP32/FP64 FMA 流水线的吞吐，不是 Tensor Core。**\n");
    std::printf("    本程序没有用任何 mma / wgmma 指令，所以这个数字与\n");
    std::printf("    「H100 SXM BF16 dense 989.5 TFLOPS」**不可比**，不要拿去对比。\n");
    std::printf("  · 可能偏低的原因：占用率不足（%d 个 block 未必填满每个 SM 的\n", blocks);
    std::printf("    所有 warp slot）；寄存器压力导致 ptxas 降低并行度；\n");
    std::printf("    boost 时钟在 kernel 执行期间因功耗墙回落。\n");
    std::printf("  · 可能偏高的原因：几乎没有——但若 ptxas 因为 UNROLL 太大而\n");
    std::printf("    把累加器 spill 到本地内存，FMA 会退化成访存，数字会**大幅偏低**。\n");
    std::printf("    用 `make regs` 看寄存器用量与 spill 可以排除这一项。\n");
    std::printf("  · __fmaf_rn / __fma_rn 保证不被重新结合，FLOPs 计数才是准的。\n");

    CHECK(cudaFree(d_out32));
    CHECK(cudaFree(d_out64));

    // ---------------------------------------------------------------- 带宽实测
    banner("4. 实测显存带宽（copy / read-only streaming）");

    // 数据规模必须**远大于 L2**，否则读命中缓存，测到的是 L2 带宽而不是显存带宽。
    // 这里按 device 属性里的 L2 大小动态选：至少 8 倍 L2，且至少 512 MiB。
    // L2 属性在无 L2 的设备上返回 0，此时 8×L2 = 0，靠 512 MiB 那个下界兜住。
    size_t l2_floor = static_cast<size_t>(l2_bytes) * 8;
    size_t want_bytes = std::max<size_t>(l2_floor, 512ull << 20);
    // 再按可用显存收一次：copy 要两条 buffer，合计只占可用显存的 80%，
    // 留 20% 给上下文、驱动开销和碎片，避免分配失败。
    size_t per_buf = std::min(want_bytes, static_cast<size_t>(mem_free / 5 * 2));
    per_buf &= ~static_cast<size_t>(4095);  // 4 KiB 对齐
    size_t n4 = per_buf / sizeof(float4);
    if (n4 == 0) {
        std::fprintf(stderr, "显存不足，无法分配带宽测试 buffer\n");
        return 1;
    }
    per_buf = n4 * sizeof(float4);

    std::printf("\n  L2 大小                     %.1f MiB\n", l2_bytes / 1048576.0);
    if (l2_bytes > 0) {
        std::printf("  单条 buffer                %.1f MiB  (= %.1f 倍 L2)\n",
                    per_buf / 1048576.0, static_cast<double>(per_buf) / l2_bytes);
    } else {
        std::printf("  单条 buffer                %.1f MiB  (L2 属性为 0，无法给出倍数)\n",
                    per_buf / 1048576.0);
    }
    std::printf("  copy 的显存占用            %.1f MiB  (两条 buffer)\n",
                2.0 * per_buf / 1048576.0);

    float4 *d_a = nullptr, *d_b = nullptr;
    float* d_sink = nullptr;
    CHECK(device_malloc(&d_a, per_buf));
    CHECK(device_malloc(&d_b, per_buf));
    CHECK(device_malloc(&d_sink, sizeof(float)));
    CHECK(cudaMemset(d_a, 0x3f, per_buf));  // 保证物理页真的被提交

    int bw_blocks = sm_count * 16;
    int bw_threads = 512;

    auto bench_copy = [&]() {
        Timer t;
        t.start();
        copy_kernel<<<bw_blocks, bw_threads>>>(d_a, d_b, n4);
        return t.stop();
    };
    auto bench_read = [&]() {
        Timer t;
        t.start();
        read_kernel<<<bw_blocks, bw_threads>>>(d_a, d_sink, n4);
        return t.stop();
    };

    std::vector<double> ms_copy, ms_read;
    for (int rep = 0; rep < 5; ++rep) {
        bench_copy();
        ms_copy.push_back(bench_copy());
        ms_read.push_back(bench_read());
    }
    CHECK(cudaGetLastError());

    double tc = median(ms_copy), tr = median(ms_read);
    // copy 一次要搬 2 倍字节：读一遍 A，写一遍 B
    double bw_copy = 2.0 * static_cast<double>(per_buf) / (tc * 1e-3);
    double bw_read = static_cast<double>(per_buf) / (tr * 1e-3);

    std::printf("\n  %-34s %12.3f\n", "copy 中位耗时 (ms)", tc);
    std::printf("  %-34s %12.3f\n", "read-only 中位耗时 (ms)", tr);
    char rc[32], rr[32];
    if (have_theo_bw) {
        std::snprintf(rc, sizeof rc, "%.1f%%", bw_copy / theo_bw * 100);
        std::snprintf(rr, sizeof rr, "%.1f%%", bw_read / theo_bw * 100);
    } else {
        std::snprintf(rc, sizeof rc, "n/a");
        std::snprintf(rr, sizeof rr, "n/a");
    }
    std::printf("\n  %-30s %14s %14s %10s\n", "口径", "实测 TB/s", "理论 TB/s", "达成率");
    std::printf("  %-30s %14.3f %14.3f %10s\n", "copy (读+写，2×字节)", bw_copy / 1e12,
                theo_bw / 1e12, rc);
    std::printf("  %-30s %14.3f %14.3f %10s\n", "read-only (只读，1×字节)", bw_read / 1e12,
                theo_bw / 1e12, rr);

    std::printf("\n  5 次重复的 copy 耗时：");
    for (double v : ms_copy) std::printf("%.3f ", v);
    std::printf("(取中位)\n");

    std::printf("\n  【口径声明 · 显存带宽】\n");
    std::printf("  · copy 口径按「读 %0.1f MiB + 写 %0.1f MiB = 2×字节」计；\n",
                per_buf / 1048576.0, per_buf / 1048576.0);
    std::printf("    read-only 口径只计读一遍。**两者不是同一个指标，不能混用。**\n");
    if (l2_bytes > 0) {
        std::printf("  · buffer 是 L2 的 %.0f 倍，基本排除了缓存命中；\n",
                    static_cast<double>(per_buf) / l2_bytes);
    } else {
        std::printf("  · L2 属性为 0，无法核对 buffer 与 L2 的比值，请人工确认规模够大；\n");
    }
    std::printf("    另：grid-stride 循环会让相邻 block 的访问在时间上有局部性，\n");
    std::printf("    实测值可能因此**偏高**。\n");
    std::printf("  · 写分配（write-allocate）会让 copy 实际写出的字节多于理论值，\n");
    std::printf("    使 copy 口径**偏高**；用 __stcs 之类的流式存储可以规避，本程序没用。\n");
    std::printf("  · 计入了 kernel 启动开销，buffer 越小影响越大；本程序规模够大，\n");
    std::printf("    这一项可以忽略。\n");
    std::printf("  · ECC 开启时 HBM 的有效带宽会低于理论值几个百分点。\n");

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_sink));

    // ---------------------------------------------------------------- 拐点
    banner("5. 本机的 roofline 拐点");

    double ridge_meas = (bw_read > 0.0) ? meas_fp32 / bw_read : 0.0;
    double ridge_theo = (have_theo_bw && have_theo_flops) ? theo_fp32_flops / theo_bw : 0.0;

    std::printf("\n  roofline 拐点 I* = 峰值算力 / 峰值带宽");
    std::printf("（I < I* 带宽受限，I > I* 算力受限）\n\n");
    std::printf("  %-34s %16s\n", "口径", "I* (FLOP/byte)");
    std::printf("  %s\n", "-------------------------------------------------------");
    std::printf("  %-34s %16.1f\n", "实测算力 / 实测带宽 (read-only)", ridge_meas);
    std::printf("  %-34s %16.1f\n", "理论算力 / 理论带宽", ridge_theo);
    std::printf("  %-34s %16s\n", "H100 SXM BF16 dense（规格，本机未必是）", "295");

    std::printf("\n  上面这三行**不是同一个口径**，放在一起只是为了说明差多少：\n");
    std::printf("    · 前两行是**标量 FP32** 的拐点；\n");
    std::printf("    · 第三行是 **Tensor Core BF16 dense** 的拐点。\n");
    std::printf("    · 前者分子是 FP32 峰值（约 %.0f TFLOPS 量级），后者是 989.5 TFLOPS。\n",
                theo_fp32_flops / 1e12);
    std::printf("      **两者的分子差了一个数量级，拐点自然差一个数量级。**\n");
    std::printf("      拿本程序测出的拐点去和 295 比大小，是口径错误。\n");

    std::printf("\n  同一台机器上不同精度的拐点：\n\n");
    std::printf("  %-22s %16s %14s %16s  %s\n", "精度", "峰值 (TFLOPS)", "相对 FP32",
                "I* (FLOP/B)", "来源");
    std::printf("  %s\n", "-------------------------------------------------------------------------------");
    struct Row {
        const char* name;
        double peak;        // 绝对峰值，FLOP/s
        const char* note;
    };
    // FP64 / FP32 用第 3 节的**实测**峰值；后三行是拿 H100 规格表上各精度
    // 峰值的比值乘本机实测 FP32 峰值外推出来的，**不是实测**，单独标注。
    const Row rows[] = {
        {"FP64", meas_fp64, "实测（第 3 节）"},
        {"FP32", meas_fp32, "实测（第 3 节）"},
        {"TF32 TC", meas_fp32 * (494.5 / 67.0), "外推（H100 规格比值）"},
        {"BF16 TC", meas_fp32 * (989.5 / 67.0), "外推（H100 规格比值）"},
        {"FP8 TC", meas_fp32 * (1979.0 / 67.0), "外推（H100 规格比值）"},
    };
    for (const Row& r : rows) {
        std::printf("  %-22s %16.1f %14.2f %16.1f  %s\n", r.name, r.peak / 1e12,
                    r.peak / meas_fp32, (bw_read > 0.0) ? r.peak / bw_read : 0.0, r.note);
    }
    std::printf("\n  【口径声明 · 这个表】\n");
    std::printf("  · 前两行（FP64 / FP32）是第 3 节的**实测**值，分子分母都是本机测的。\n");
    std::printf("  · 后三行（Tensor Core）**不是实测**：本程序根本没有跑 mma/wgmma，\n");
    std::printf("    只是拿「H100 规格表上该精度峰值 ÷ FP32 峰值」这个比值乘上本机实测\n");
    std::printf("    FP32 峰值。它成立的假设是**本机各精度的峰值比例与 H100 相同**，\n");
    std::printf("    对非 Hopper 的卡这个假设不成立。这一列只能当量级参考。\n");
    std::printf("  · 分母统一用实测 read-only 带宽（%.3f TB/s），未按精度区分——\n",
                bw_read / 1e12);
    std::printf("    带宽与精度无关，这样处理是对的。\n");

    banner("6. 本程序的边界");

    std::printf("\n  · 测不到 Tensor Core。要测它必须用 mma/wgmma 或 cuBLAS，\n");
    std::printf("    这是本程序最大的限制，第 3、5 节都因此只能给标量 FP32 的数。\n");
    std::printf("  · 没有测 L2 带宽、共享内存带宽、NVLink 带宽。\n");
    std::printf("  · 没有分离 kernel 启动开销与真实执行时间，只是把规模取大让它可忽略。\n");
    std::printf("  · 所有数字来自**一次进程运行**。GPU 的时钟受功耗、温度、\n");
    std::printf("    同机其他进程影响，换一次运行可能差几个百分点。\n");
    std::printf("    本程序对每个测量跑 5 次取中位，但没有做长期稳定性测试。\n");
    std::printf("  · 没有验证过本程序在 sm_80 之外的架构上的正确性；\n");
    std::printf("    每 SM 的 FP32 单元数是硬编码的，换架构要人工核对第 2 节。\n");

    CHECK(cudaDeviceReset());
    return 0;
}
