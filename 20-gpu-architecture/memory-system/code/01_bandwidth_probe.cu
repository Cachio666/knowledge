// 01_bandwidth_probe.cu —— 各级存储带宽实测
// 对应文档：01-hierarchy-overview.md
//
// 本程序做两件事（与文档「编译与运行」一致）：
//   1. 打印本设备的存储规格：SM 数、每 SM 的统一 L1/共享内存与共享内存
//      上限、L2 容量、显存带宽（规格峰值，由显存时钟与总线宽度折算）
//   2. 按工作集大小逐档扫描一次顺序访存，打印每档的吞吐 ——
//      把打印出的容量代进文档 §1 第五、六步的式子，再看扫描曲线的
//      「容量拐点」落在哪一档，那一档就是当前访问模式下接管的层级
//
// 程序末尾另给一张参考基线表，对比 HBM 拷贝 / HBM 只读 / 共享内存读取
// 三种访问模式的带宽，用于观察「共享内存带宽远高于 HBM 带宽」。
//
// 观测要点：
//   1. 共享内存带宽远高于 HBM 带宽（典型差一个数量级）
//   2. 实测带宽通常只有规格峰值的 70%~90%
//   3. 带宽与延迟是两件事：带宽高不代表单次访问快
//      —— 本程序给的是吞吐，验不出延迟
//
// 构建：make 01_bandwidth_probe
// 运行：./01_bandwidth_probe
//
// 说明：本程序测得的绝对值依 GPU 型号而异，请关注数量级与相对关系。

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err_));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads  = 256;
constexpr int kBlocks   = 1024;
constexpr int kReps     = 20;
constexpr int kSmemIter = 4096;  // 共享内存读取轮数，用来放大片内访问

// 工作集扫描的档位（字节）：8 KB 起、每档 ×4，一路越过 L2。
// 拐点出现在哪一档，那一档就是当前访问模式下真正接管的层级。
const size_t kSweepBytes[] = {8u << 10,  32u << 10, 128u << 10, 512u << 10,
                              2u << 20,  8u << 20,  32u << 20,  128u << 20};

// 每档的目标读取量：把同一块工作集反复读，直到搬动约 256 MB。
// 小工作集单趟只有几纳秒，不放大就量到的是启动开销而不是带宽。
const double kSweepTargetBytes = 256.0 * (1 << 20);

// 全局内存拷贝：读一份写一份。带宽按「读 + 写」的总字节数计。
__global__ void copy_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = in[i];
  }
}

// 只读扫描：只读不写，测纯读带宽。
__global__ void read_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  const int stride = gridDim.x * blockDim.x;
  float acc        = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    acc += in[i];
  }
  if (acc == -1234567.0f) out[0] = acc;
}

// 工作集扫描用的只读 kernel：在 read_kernel 外面再套 inner 轮整数组遍历。
// 内层轮数把测量时间拉长，使小工作集的档位也能被事件量到。
__global__ void sweep_read_kernel(const float* __restrict__ in,
                                  float* __restrict__ out, int n, int inner) {
  const int stride = gridDim.x * blockDim.x;
  float acc        = 0.0f;
  for (int rep = 0; rep < inner; ++rep) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
      acc += in[i];
    }
  }
  if (acc == -1234567.0f) out[0] = acc;
}

// 共享内存读取：先把一块数据搬进共享内存，再反复读取。
// 读取模式 tile[(tid + k) & (bdim - 1)] 保证 warp 内相邻 lane 落在
// 相邻的 4 字节单元上，即映射到 32 个不同的 bank，无冲突。
__global__ void smem_read_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
  extern __shared__ float tile[];
  const int tid  = threadIdx.x;
  const int bdim = blockDim.x;

  // 装载：全局读 + 共享写，各一次
  for (int i = tid; i < bdim; i += bdim) {
    const int g = blockIdx.x * bdim + i;
    tile[i]     = (g < n) ? in[g] : 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll 8
  for (int k = 0; k < kSmemIter; ++k) {
    acc += tile[(tid + k) & (bdim - 1)];
  }
  __syncthreads();

  if (acc == -1234567.0f) out[0] = acc;
}

template <typename Kernel>
float time_kernel(Kernel k, int blocks, int threads, size_t smem, int n,
                  const float* d_in, float* d_out) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  k<<<blocks, threads, smem>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    k<<<blocks, threads, smem>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / kReps;
}

// 与 time_kernel 同构，多一个内层轮数参数。返回单次启动的平均耗时。
float time_sweep(const float* d_in, float* d_out, int n, int inner,
                 int blocks) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  sweep_read_kernel<<<blocks, kThreads>>>(d_in, d_out, n, inner);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    sweep_read_kernel<<<blocks, kThreads>>>(d_in, d_out, n, inner);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / kReps;
}

}  // namespace

int main() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

  // 显存时钟、显存总线宽度与 SM 时钟在 CUDA 13 里已从 cudaDeviceProp 移除
  // （clockRate / memoryClockRate / memoryBusWidth 三者一并移除），
  // 这里统一走 cudaDeviceGetAttribute —— 12.x 与 13.x 都可用。
  int mem_clock_khz = 0;  // 显存时钟，只用来折算带宽
  int mem_bus_bits  = 0;  // 显存总线宽度
  int sm_clock_khz  = 0;  // SM 峰值时钟，用来折算「每周期可交付字节数」
  CUDA_CHECK(
      cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_bus_bits,
                                    cudaDevAttrGlobalMemoryBusWidth, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0));

  // 规格峰值 = 显存时钟 × 总线宽度 ÷ 8 × 2（DDR 每周期传输两次）
  const double peak_gbps = static_cast<double>(mem_clock_khz) * 1e3 *
                           (mem_bus_bits / 8.0) * 2.0 / 1e9;

  std::printf("设备: %s (compute capability %d.%d)\n", p.name, p.major,
              p.minor);
  std::printf("\n--- 存储规格 ---\n");
  std::printf("SM 数                      : %d\n", p.multiProcessorCount);
  std::printf("每 SM 统一 L1/共享内存     : %zu KB\n",
              static_cast<size_t>(p.sharedMemPerMultiprocessor) / 1024);
  std::printf("每 block 共享内存上限      : %zu KB（可选择加入的上限 %zu KB）\n",
              static_cast<size_t>(p.sharedMemPerBlock) / 1024,
              static_cast<size_t>(p.sharedMemPerBlockOptin) / 1024);
  // L2 一行沿用文档 §3 规格表的口径（1 MiB 记作 1 MB），与上面的数组
  // 容量（十进制 MB）不同源：前者对齐文档数字，后者对齐 %f 的打印值。
  std::printf("L2 容量                    : %zu KB (%.1f MB)\n",
              static_cast<size_t>(p.l2CacheSize) / 1024,
              static_cast<double>(p.l2CacheSize) / (1 << 20));
  std::printf("显存总线                   : %d bit @ %d kHz\n", mem_bus_bits,
              mem_clock_khz);
  std::printf("显存带宽（规格峰值）       : %.1f GB/s\n", peak_gbps);
  std::printf("SM 峰值时钟                : %d kHz\n", sm_clock_khz);
  std::printf("  —— 折算口径（文档 §1 第五步）：显存带宽 ÷ **SM 时钟** =\n");
  std::printf("     每周期可交付字节数。上面的显存时钟只用来折算带宽，\n");
  std::printf("     拿它去除带宽会得到偏小的结果。\n");

  const size_t n_sizes = sizeof(kSweepBytes) / sizeof(kSweepBytes[0]);
  const size_t max_bytes = kSweepBytes[n_sizes - 1];

  float* d_scan = nullptr;
  float* d_flag = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scan, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_flag, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_scan, 0, max_bytes));

  std::printf("\n--- 工作集扫描：顺序只读 ---\n");
  std::printf("对照文档 §1 第六步的容量：每 SM 统一池 %zu KB、L2 %.1f MB。\n",
              static_cast<size_t>(p.sharedMemPerMultiprocessor) / 1024,
              static_cast<double>(p.l2CacheSize) / (1 << 20));
  std::printf("%-14s %-14s %-12s %-14s %s\n", "工作集(KB)", "元素数", "内层轮数",
              "耗时(ms)", "吞吐(GB/s)");

  for (size_t i = 0; i < n_sizes; ++i) {
    const size_t ws_bytes = kSweepBytes[i];
    const int    n        = static_cast<int>(ws_bytes / sizeof(float));

    int inner =
        static_cast<int>(kSweepTargetBytes / static_cast<double>(ws_bytes));
    if (inner < 1) inner = 1;
    if (inner > 65536) inner = 65536;

    int blocks = (n + kThreads - 1) / kThreads;
    if (blocks > kBlocks) blocks = kBlocks;

    const float  ms    = time_sweep(d_scan, d_flag, n, inner, blocks);
    const double moved = static_cast<double>(n) * sizeof(float) * inner;
    std::printf("%-14zu %-14d %-12d %-14.4f %.1f\n", ws_bytes / 1024, n, inner,
                ms, moved / (ms * 1e-3) / 1e9);
  }

  std::printf("\n解读：吞吐在某一档之后明显下台阶，那一档就是工作集越过了\n");
  std::printf("该级的容量线、访问开始落到下一级的信号（容量数字见上表规格）。\n");

  // ---- 参考基线：三种访问模式的带宽 ----
  // 16M 元素 = 67 MB（十进制，与下面 %.0f 那一行的口径一致；
  // 换算成 MiB 才是 64）
  const int n_base = 1 << 24;

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n_base * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n_base * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n_base * sizeof(float)));

  const double bytes = static_cast<double>(n_base) * sizeof(float);

  std::printf("\n--- 参考基线：三种访问模式 ---\n");
  std::printf("数组 %.0f MB，block 数 %d，每 block %d 线程\n", bytes / 1e6,
              kBlocks, kThreads);

  const float ms_copy =
      time_kernel(copy_kernel, kBlocks, kThreads, 0, n_base, d_in, d_out);
  const float ms_read =
      time_kernel(read_kernel, kBlocks, kThreads, 0, n_base, d_in, d_out);

  const size_t smem_bytes = kThreads * sizeof(float);
  const float  ms_smem    = time_kernel(smem_read_kernel, kBlocks, kThreads,
                                        smem_bytes, n_base, d_in, d_out);

  // 共享内存读取次数 = block 数 × 每 block 线程数 × 轮数
  const double smem_bytes_moved = static_cast<double>(kBlocks) * kThreads *
                                  kSmemIter * sizeof(float);

  std::printf("%-22s %-14s %s\n", "访问类型", "耗时(ms)", "带宽(GB/s)");
  std::printf("%-22s %-14.4f %.1f\n", "HBM 拷贝（读+写）", ms_copy,
              bytes * 2.0 / (ms_copy * 1e-3) / 1e9);
  std::printf("%-22s %-14.4f %.1f\n", "HBM 只读", ms_read,
              bytes / (ms_read * 1e-3) / 1e9);
  std::printf("%-22s %-14.4f %.1f\n", "共享内存读取", ms_smem,
              smem_bytes_moved / (ms_smem * 1e-3) / 1e9);

  std::printf("\n--- 解读 ---\n");
  std::printf("共享内存带宽应显著高于 HBM —— 这是各级存储分级的直接体现。\n");
  std::printf("注意本程序测的是带宽，不是延迟：HBM 靠大量并行通道堆带宽，\n");
  std::printf("单次访问的延迟仍是数百 cycles，两者并不矛盾。\n");
  std::printf("实测值通常只有规格峰值的 70%%~90%%，属正常范围。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_scan));
  CUDA_CHECK(cudaFree(d_flag));
  return 0;
}
