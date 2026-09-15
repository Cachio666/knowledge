// 02_divergence_demo.cu —— 分支发散对有效并行度的影响
// 对应文档：02-simt-and-divergence.md
//
// 观测要点：
//   1. active = 0 或 32 时全 warp 走同一条路径，最快
//   2. active 在 1..31 之间时 warp 分裂为 2 条路径，硬件把两条路径的指令
//      都发射一遍，耗时落到接近重路径的那一档
//   3. branchless 版本必须把两条路径都算一遍，故耗时恒定且不低于 divergent
//
// 构建：make 02_divergence_demo
// 运行：./02_divergence_demo

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

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads = 256;
constexpr int kIters   = 32;   // 路径 A 的循环次数，用来放大发散代价
constexpr int kReps    = 20;

// 路径 A：较重的计算。刻意做得足够长，使编译器不会把
// if/else if-convert 成谓词执行——否则发散无法被观测到。
__device__ __forceinline__ float path_heavy(float v) {
  float r = v;
#pragma unroll
  for (int k = 0; k < kIters; ++k) {
    r = fmaf(r, 1.000001f, 1e-6f);
  }
  return r;
}

// 路径 B：轻计算
__device__ __forceinline__ float path_light(float v) { return v + 1.0f; }

// kBranchless = false：真实的 if/else 控制流，会产生分支发散
// kBranchless = true ：两条路径都求值，用三元运算符取结果，无控制流分歧
//
// 控制发散的方式：warp 内前 `active` 个 lane 走路径 A，其余走路径 B。
//   active = 0 或 32 → 全 warp 同路径，路径数 1
//   active = 16      → warp 分裂为两条路径
template <bool kBranchless>
__global__ void divergence_kernel(const float* __restrict__ in,
                                  float* __restrict__ out, int n, int active) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float v    = in[i];
  const int   lane = threadIdx.x & 31;  // warp 内 lane 号

  if (kBranchless) {
    const float heavy = path_heavy(v);
    const float light = path_light(v);
    out[i]            = (lane < active) ? heavy : light;
  } else {
    float r;
    if (lane < active) {
      r = path_heavy(v);
    } else {
      r = path_light(v);
    }
    out[i] = r;
  }
}

template <bool kBranchless>
float time_variant(const float* d_in, float* d_out, int n, int active,
                   int blocks) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  // 预热
  divergence_kernel<kBranchless><<<blocks, kThreads>>>(d_in, d_out, n, active);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kReps; ++r) {
    divergence_kernel<kBranchless><<<blocks, kThreads>>>(d_in, d_out, n,
                                                         active);
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
  constexpr int n      = 1 << 22;  // 4M 元素
  constexpr int blocks = (n + kThreads - 1) / kThreads;

  float* h_in = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
  for (int i = 0; i < n; ++i) {
    h_in[i] = static_cast<float>(i % 1000) * 0.001f;
  }

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  std::printf("元素数 %d，block 数 %d，每 block %d 线程\n\n", n, blocks,
              kThreads);
  std::printf("%-8s %-10s %-16s %-16s %s\n", "active", "路径数",
              "divergent(ms)", "branchless(ms)", "div/branchless");

  float t_light   = 0.0f;  // active = 0：全 warp 走轻路径
  float t_heavy   = 0.0f;  // active = 32：全 warp 走重路径

  for (int active : {0, 4, 8, 16, 24, 32}) {
    const float t_div =
        time_variant<false>(d_in, d_out, n, active, blocks);
    const float t_bl =
        time_variant<true>(d_in, d_out, n, active, blocks);

    if (active == 0)  t_light = t_div;
    if (active == 32) t_heavy = t_div;

    const char* paths = (active == 0 || active == 32) ? "1" : "2";
    std::printf("%-8d %-10s %-16.4f %-16.4f %.2fx\n", active, paths, t_div,
                t_bl, t_div / t_bl);
  }

  std::printf("\n--- 解读 ---\n");
  std::printf("1. active = 0 或 32 时全 warp 走同一条路径，路径数为 1，divergent 最快。\n");
  std::printf("   纯轻路径 %.4f ms，纯重路径 %.4f ms（下一条解读会引用它）。\n",
              t_light, t_heavy);
  std::printf("2. active 在 1..31 之间时 warp 分裂为两条路径，硬件把两条路径的指令\n");
  std::printf("   都发射一遍。轻路径只多一两条指令，省下的算量换不回重路径的开销，\n");
  std::printf("   因此 divergent 的耗时会落到接近**重路径**那一档（%.4f ms）。\n",
              t_heavy);
  std::printf("   这正是有效并行度 32/N 的含义：warp 分裂后按最重的那条路径计时。\n");
  std::printf("   注意本 kernel 的耗时下限由那次全局访存决定，所以分裂出的轻路径\n");
  std::printf("   几乎不额外增加时间 —— 若两条路径的算量相当，耗时会接近两者之和。\n");
  std::printf("3. branchless 版本无论 active 取何值，都必须把两条路径都算一遍，\n");
  std::printf("   因此耗时基本恒定，且不低于 divergent —— 即便在 active = 0 时也是如此。\n");
  std::printf("   这说明「用算术选择消除分支」对**长**分支并不免费：它把控制流分歧\n");
  std::printf("   换成了计算冗余。无分支写法的价值在短分支（省去 BSSY/BRA/BSYNC 的开销），\n");
  std::printf("   不在长分支。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  return 0;
}
