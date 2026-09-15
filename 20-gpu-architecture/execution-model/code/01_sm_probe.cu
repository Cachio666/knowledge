// 01_sm_probe.cu —— 查询 SM 规格与寄存器预算
// 对应文档：01-sm-internals.md
//
// 观测要点：
//   1. 每 SM 的寄存器文件大小与最大常驻 warp 数
//   2. 寄存器文件是「预算」而非「缓存」：每线程占用越多，
//      可常驻 warp 数越少，占用率随之下降
//
// 构建：make 01_sm_probe
// 运行：./01_sm_probe

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

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

int main() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  std::printf("设备: %s (compute capability %d.%d)\n", p.name, p.major,
              p.minor);

  const int regs_per_sm    = p.regsPerMultiprocessor;  // 32-bit 寄存器个数
  const int threads_per_sm = p.maxThreadsPerMultiProcessor;
  const int warps_per_sm   = threads_per_sm / 32;

  std::printf("\n--- SM 规格 ---\n");
  std::printf("SM 数量                : %d\n", p.multiProcessorCount);
  std::printf("每 SM 寄存器文件       : %d 个 32-bit 寄存器 (%d KB)\n",
              regs_per_sm, regs_per_sm * 4 / 1024);
  std::printf("每 SM 最大常驻线程     : %d\n", threads_per_sm);
  std::printf("每 SM 最大常驻 warp    : %d\n", warps_per_sm);
  std::printf("每 SM 共享内存         : %zu KB\n",
              static_cast<size_t>(p.sharedMemPerMultiprocessor) / 1024);
  std::printf("每 SM 最大常驻 block   : %d\n", p.maxBlocksPerMultiProcessor);

  // 单个 warp 占用的寄存器数 = 每线程寄存器数 x 32。
  // 寄存器文件能容纳的 warp 数 = regs_per_sm / (每线程寄存器数 x 32)。
  // 该值一旦小于架构上限 warps_per_sm，寄存器就成为占用率的限制项。
  std::printf("\n--- 寄存器预算（warp = 32 线程）---\n");
  std::printf("%-16s %-16s %-18s %s\n", "每线程寄存器", "单 warp 占用",
              "寄存器可容 warp", "是否成为瓶颈");

  for (int regs : {16, 24, 32, 40, 64, 128, 255}) {
    const int per_warp = regs * 32;
    const int capacity = regs_per_sm / per_warp;
    std::printf("%-16d %-16d %-18d %s\n", regs, per_warp, capacity,
                capacity < warps_per_sm ? "是" : "否");
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("寄存器文件是预算而非缓存：它必须为每 SM 上全部常驻线程静态预留，\n");
  std::printf("无法像 cache 那样按需换入换出。每线程多用一档寄存器，\n");
  std::printf("可常驻 warp 数就下降一档，占用率随之下降。\n");
  std::printf("当「寄存器可容 warp」小于 %d 时，降低每线程寄存器数\n",
              warps_per_sm);
  std::printf("（而非增加）才是提高占用率的正确方向。\n");

  return 0;
}
