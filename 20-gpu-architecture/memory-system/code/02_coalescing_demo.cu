// 02_coalescing_demo.cu —— 访存合并对有效带宽的影响
// 对应文档：02-coalescing-and-cache-line.md
//
// 观测要点：
//   1. kStride = 1 时，一个 warp 的 32 次访问落在同一条 128B cache line，1 个事务
//   2. kStride = S 时，一个 warp 的 32 次访问跨越 S 条 cache line，S 个事务
//   3. 四种 stride 下「有用读取次数」恒定，传输字节按 stride 增长，
//      故有效利用率 = 1 / kStride
//
// 能力边界：程序给的是**有效带宽**，验不出**延迟** —— 它能核对的只有「传输
// 字节数变了多少」这一本书，无法核对单次事务要多少个周期（那是文档
// 01-hierarchy-overview 与 03-occupancy-and-latency-hiding 的题目）。绝对
// 带宽数取决于具体机器，没有可比的基准值；能对照的只有四档之间
// 的比值。
//
// 构建：make 02_coalescing_demo
// 运行：./02_coalescing_demo
//
// 口径说明：本文件里「数组多少 MB」按十进制（1e6 字节）计，与程序输出
// 一致（kStride = 8 的数组是 537 MB，不是 512）；文中引用 L2 的
// 40 MB / 50 MB / 126 MB 则沿用文档与厂商规格的口径，两者不同源。
//
// 显存占用：kStride = 8 时数组为 537 MB，是本程序的最大分配。
//
// 工作集提示：stride = 1 时数组为 67 MB。目标环境（sm_80 的 L2 为 40 MB、
// sm_90 为 50 MB）装不下，会真正落到 HBM。若换到 L2 达 126 MB 的新架构
// （如 B100/B200），stride = 1 那一档可能整块驻留 L2，该行数值会偏高、
// 与 1/stride 的拟合变差 —— 这是预期的，不是 bug。

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

constexpr int kThreads = 256;
constexpr int kBlocks  = 2048;
constexpr int kIters   = 32;   // 每个线程固定读取次数
constexpr int kReps    = 20;

constexpr long long kTotalThreads = static_cast<long long>(kBlocks) * kThreads;
constexpr long long kTotalWarps   = kTotalThreads / 32;

// 全部线程共读 kTotalThreads × kIters 次。数组长度随 kStride 增长，
// 使**每个线程的读取次数恒为 kIters**，于是「有用字节」与 kStride 无关，
// 可以直接比较有效带宽。
constexpr long long kReads = kTotalThreads * kIters;

// 数组长度：total_threads × kStride × kIters。
// kStride = 8 时最大，为 1.34 亿个 float = 537 MB（十进制）。
template <int kStride>
constexpr long long array_len() {
  return kTotalThreads * kStride * kIters;
}

// kStride 决定 warp 内相邻 lane 的地址间隔。
//
// 每个 warp 负责一段连续的 32 × kStride 个元素，其中第 lane 个 lane
// 读取本段内的第 lane × kStride 个元素：
//   kStride = 1 → warp 读 32 个连续元素，落在同一条 128B cache line，1 个事务
//   kStride = S → warp 的 32 次读跨越 S 条 cache line，S 个事务
//
// 全部 warp 的段拼起来恰好覆盖整个数组，且每条 cache line 只被一个 warp
// 触及一次。于是「传输字节」为 数组大小，而「有用字节」恒为 kReads × 4。
//
// 有效利用率 = 有用字节 / 传输字节 = 1 / kStride。
//
// 只取 1、2、4、8 四个取值：它们都能整除一条 cache line 含的 32 个 float，
// 因此每条被触及的 cache line 的 4 个 sector 都会被用到 —— sector 粒度
// 与整行粒度算出的利用率一致，不会出现「按 sector 算是 1/8、按整行算是
// 1/16」这种含糊结果。取 16 或 32 就会破坏这个性质。
template <int kStride>
__global__ void strided_sum_kernel(const float* __restrict__ in,
                                   float* __restrict__ out, long long n) {
  const long long tid =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long warp_id = tid >> 5;
  const int       lane    = threadIdx.x & 31;

  const long long warp_span = 32LL * kStride;

  float acc = 0.0f;
  for (int j = 0; j < kIters; ++j) {
    const long long idx =
        (static_cast<long long>(j) * kTotalWarps + warp_id) * warp_span +
        static_cast<long long>(lane) * kStride;
    if (idx < n) acc += in[idx];
  }
  if (acc == -1234567.0f) out[0] = acc;
}

template <int kStride>
float time_case(const float* d_in, float* d_out, long long n) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  strided_sum_kernel<kStride><<<kBlocks, kThreads>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    strided_sum_kernel<kStride><<<kBlocks, kThreads>>>(d_in, d_out, n);
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

struct Case {
  int       stride;
  long long n;
  float     ms;
};

}  // namespace

int main() {
  const long long n_max = array_len<8>();

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n_max * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n_max * sizeof(float)));

  const double useful_bytes = static_cast<double>(kReads) * sizeof(float);

  std::printf("每线程读取 %d 次，全部线程共读 %lld 次 = %.1f MB 有用数据\n",
              kIters, kReads, useful_bytes / 1e6);
  std::printf("数组长度随 stride 增长，最大 %.0f MB\n\n",
              static_cast<double>(n_max) * sizeof(float) / 1e6);

  const Case cases[] = {
      {1, array_len<1>(), time_case<1>(d_in, d_out, array_len<1>())},
      {2, array_len<2>(), time_case<2>(d_in, d_out, array_len<2>())},
      {4, array_len<4>(), time_case<4>(d_in, d_out, array_len<4>())},
      {8, array_len<8>(), time_case<8>(d_in, d_out, array_len<8>())},
  };

  // 以 stride = 1 为基准，其有效利用率定义为 1
  const double base_gbps = useful_bytes / (cases[0].ms * 1e-3) / 1e9;

  // 表头按「显示宽度」手工对齐：printf 的 %-Ns 按字节补空格，而中文一个字
  // 3 字节、只占 2 列，混排时会少补，得自己数空格。
  std::printf(
      "stride   warp 跨线数 数组(MB)    耗时(ms)    有用带宽(GB/s)   "
      "相对利用率     模型值\n");

  for (const Case& c : cases) {
    const double gbps = useful_bytes / (c.ms * 1e-3) / 1e9;
    std::printf("%-9d%-12d%-12.0f%-12.4f%-17.1f%-15.1f%.1f%%\n", c.stride,
                c.stride, static_cast<double>(c.n) * sizeof(float) / 1e6, c.ms,
                gbps, 100.0 * gbps / base_gbps, 100.0 / c.stride);
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("「模型值」一列是 §1 给出的 1/stride 预测（100%% / 50%% / 25%% / 12.5%%），\n");
  std::printf("「相对利用率」一列是实测带宽除以 stride = 1 那一档的实测带宽，\n");
  std::printf("两列同以百分比表示 —— 对照即可看出模型是否成立。\n");
  std::printf("四种 stride 下的「有用读取次数」完全相同，但数组长度按 stride 增长，\n");
  std::printf("即传输字节按 stride 增长，因此「相对利用率」一列应接近「模型值」。\n");
  std::printf("stride = 1 时 warp 的 32 次访问落在同一条 128B cache line 上，\n");
  std::printf("取回的每个字节都被用到；stride = 8 时一次访问跨越 8 条 cache line，\n");
  std::printf("每条只用到 1/8 的字节，有效带宽随之降到 1/8。\n");
  std::printf("本程序只取 1/2/4/8 四个 stride：它们都能整除一条 cache line 所含的\n");
  std::printf("32 个 float，因此每条被触及的 cache line 的 4 个 sector 都会被用到，\n");
  std::printf("sector 粒度与整行粒度给出同样的利用率。取 16 或 32 会破坏这个性质，\n");
  std::printf("使「利用率」取决于内存系统按 sector 还是按整行传输，结论会含糊。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
