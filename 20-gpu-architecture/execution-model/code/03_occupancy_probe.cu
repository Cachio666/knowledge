// 03_occupancy_probe.cu —— 占用率与延迟隐藏
// 对应文档：03-occupancy-and-latency-hiding.md
//
// 观测要点：
//   1. 每 block 的共享内存用量是占用率的限制项之一：用量越大，
//      每 SM 能放下的 block 越少，占用率越低
//   2. cudaOccupancyMaxActiveBlocksPerMultiprocessor 可精确算出
//      给定 kernel、block 尺寸与共享内存用量下的常驻 block 数
//   3. 占用率下降不必然带来吞吐下降 —— 若瓶颈是访存带宽而非延迟，
//      减少 warp 数并不会让带宽变小
//
// 为什么用共享内存而不是 __launch_bounds__ 来调节占用率：
// 本 kernel 只需约 10~16 个寄存器/线程，寄存器从来不是限制项，
// __launch_bounds__ 的 minBlocks 参数根本不会生效，四行输出会完全相同。
// 共享内存用量则是确定生效的限制项，且不依赖编译器行为。
// __launch_bounds__ 的用法见文档第 3 节，此处不演示。
//
// 构建：make 03_occupancy_probe
// 运行：./03_occupancy_probe

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
constexpr int kReps    = 20;

// 流式累加的访存受限 kernel。动态共享内存的用量由启动参数给出，
// 它不参与计算，只用来人为压低每 SM 的常驻 block 数。
__global__ void accumulate_kernel(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
  extern __shared__ float tile[];

  const int stride = gridDim.x * blockDim.x;
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  for (int i = tid; i < n; i += stride) {
    acc += in[i];
  }

  // 让共享内存真正参与计算，避免整块共享内存被编译器判定为无用而影响
  // 占用率计算。tile 至少要有 blockDim.x 个 float。
  tile[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = tile[0];
}

struct Result {
  size_t smem_bytes;
  int    blocks_per_sm;
  int    regs_per_thread;
  float  occupancy;  // 0..1
  float  ms;
  float  gbps;
};

Result probe(size_t smem_bytes, const float* d_in, float* d_out, int n,
             int blocks, int threads_per_sm) {
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, accumulate_kernel));

  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, accumulate_kernel, kThreads, smem_bytes));

  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  // 预热
  accumulate_kernel<<<blocks, kThreads, smem_bytes>>>(d_in, d_out, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    accumulate_kernel<<<blocks, kThreads, smem_bytes>>>(d_in, d_out, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  ms /= kReps;
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));

  const float resident = static_cast<float>(blocks_per_sm) * kThreads;
  return Result{smem_bytes, blocks_per_sm, attr.numRegs,
                resident / static_cast<float>(threads_per_sm), ms,
                static_cast<float>(n) * sizeof(float) / (ms * 1e-3f) / 1e9f};
}

}  // namespace

int main() {
  constexpr int n      = 1 << 24;  // 16M 元素 = 64 MB
  constexpr int blocks = 4096;

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("设备: %s\n", p.name);
  std::printf("每 SM 最大常驻线程 %d，每 SM 共享内存 %zu KB，每 block %d 线程\n\n",
              p.maxThreadsPerMultiProcessor, p.sharedMemPerMultiprocessor / 1024,
              kThreads);

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));

  // 逐个取值探测。最小取 1 KB —— 内核要用 tile[0..kThreads-1]，
  // 即 kThreads × 4 = 1024 字节，低于此值会越界。
  const size_t smems[] = {1u << 10, 8u << 10, 16u << 10, 32u << 10, 48u << 10};

  const Result first = probe(smems[0], d_in, d_out, n, blocks,
                             p.maxThreadsPerMultiProcessor);

  std::printf("寄存器用量 %d/线程（不是本 kernel 的限制项）\n", first.regs_per_thread);
  std::printf("常驻 block 数 = min(线程数限制 %d, 共享内存限制, block 数限制 %d)\n\n",
              p.maxThreadsPerMultiProcessor / kThreads, p.maxBlocksPerMultiProcessor);

  std::printf("%-14s %-16s %-12s %-12s %s\n", "共享内存/block",
              "每 SM 常驻 block", "占用率", "耗时(ms)", "带宽(GB/s)");
  std::printf("%-14zu %-16d %-12.1f %-12.4f %.1f\n", first.smem_bytes / 1024,
              first.blocks_per_sm, first.occupancy * 100.0f, first.ms,
              first.gbps);

  for (size_t i = 1; i < sizeof(smems) / sizeof(smems[0]); ++i) {
    const Result r = probe(smems[i], d_in, d_out, n, blocks,
                           p.maxThreadsPerMultiProcessor);
    std::printf("%-14zu %-16d %-12.1f %-12.4f %.1f\n", r.smem_bytes / 1024,
                r.blocks_per_sm, r.occupancy * 100.0f, r.ms, r.gbps);
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("前几档的「每 SM 常驻 block」可能不变 —— 此时限制项是线程数\n");
  std::printf("（每 SM 最大常驻线程 ÷ 每 block 线程数），共享内存还没用满。\n");
  std::printf("用量继续增大后，共享内存取代线程数成为限制项，占用率随之下台阶。\n");
  std::printf("注意「寄存器/线程」自始至终不变 —— 它不是本 kernel 的限制项。\n");
  std::printf("四项约束取最小：谁先卡住，谁就决定占用率。\n\n");
  std::printf("再对比「占用率」与「带宽」两列：\n");
  std::printf("  · 若带宽基本不随占用率下降而变，说明该 kernel 的瓶颈是访存带宽\n");
  std::printf("    而非延迟 —— 硬件带宽已被打满，再多 warp 也只能排队等内存。\n");
  std::printf("  · 若带宽随占用率同步下降，说明此时仍是延迟受限，warp 数不足以\n");
  std::printf("    掩盖访存延迟。\n");
  std::printf("两种情形都指向同一结论：占用率是**约束**而非**优化目标**。\n");
  std::printf("调优时应先用 profiling 判断瓶颈类型，再决定是否值得为占用率\n");
  std::printf("牺牲寄存器或共享内存。具体数值随 GPU 型号与架构而异。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
