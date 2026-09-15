// 02_divergence_demo.cu —— 分支发散对有效并行度的影响
// 对应文档：02-simt-and-divergence.md
//
// 观测要点：
//   1. active = 0 时 divergent 能整个跳过重路径的 32 条 FMA，两版差距最大；
//      active = 32 时重路径本来就非走不可，divergent 能省的只剩轻路径那一条加法
//   2. active 在 1..31 之间时 warp 分裂为 2 条路径，硬件把两条路径的指令
//      都发射一遍，耗时落到接近重路径的那一档
//   3. branchless 版本必须把两条路径都算一遍，故耗时恒定且不低于 divergent
//      （这是按模型预期的形状，不是实测结论）
//
// 构建：make 02_divergence_demo
// 运行：./02_divergence_demo

#include <cmath>
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

// 两个变体必须逐元素给出相同结果（同一段算术、同一个选择），这里拿
// divergent 的输出做基准比对，报出不一致的元素数。
int count_mismatch(const float* d_a, const float* d_b, float* h_a, float* h_b,
                   int n) {
  CUDA_CHECK(cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) ++bad;
  }
  return bad;
}

}  // namespace

int main() {
  // 工作集刻意压到 L2 以内（2 MB 入 + 2 MB 出）。若让它落在 HBM 上，
  // 访存地板会盖过发射侧的差异，两列耗时之差将退化到噪声。
  constexpr int n      = 1 << 19;  // 512K 元素 = 2 MB
  constexpr int blocks = (n + kThreads - 1) / kThreads;

  float* h_in = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, n * sizeof(float)));
  for (int i = 0; i < n; ++i) {
    h_in[i] = static_cast<float>(i % 1000) * 0.001f;
  }

  float* d_in  = nullptr;
  float* d_out = nullptr;
  float* d_ref = nullptr;  // divergent 版本最后一次的输出，做正确性基准
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ref, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  float* h_ref = nullptr;
  float* h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_ref, n * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, n * sizeof(float)));

  // 本程序赖以观测差异的前提是「工作集驻留 L2」。把它做成可验证的事实，
  // 而不是注释里的断言：前提不成立时会显式提示，免得读者以为程序坏了。
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  const size_t working_set = static_cast<size_t>(n) * 2 * sizeof(float);
  std::printf("设备: %s，L2 缓存 %d KB，工作集（入 + 出）%zu KB\n", p.name,
              p.l2CacheSize / 1024, working_set / 1024);
  if (working_set > static_cast<size_t>(p.l2CacheSize)) {
    std::printf("注意：工作集大于 L2，访存会落到 HBM，两版耗时之差将被访存地板\n");
    std::printf("      吸收，下表可能接近平坦 —— 这不是程序出错，而是本程序赖以\n");
    std::printf("      观测差异的前提不成立。\n");
  }
  std::printf("\n");

  std::printf("元素数 %d，block 数 %d，每 block %d 线程\n\n", n, blocks,
              kThreads);
  std::printf("%-8s %-10s %-16s %-16s %-12s %-12s %s\n", "active", "路径数",
              "divergent(ms)", "branchless(ms)", "diff(ms)", "不一致",
              "div/branchless");

  float t_light   = 0.0f;  // active = 0：全 warp 走轻路径
  float t_heavy   = 0.0f;  // active = 32：全 warp 走重路径
  int   bad       = 0;     // 两个变体输出不一致的元素总数

  for (int active : {0, 4, 8, 16, 24, 32}) {
    const float t_div =
        time_variant<false>(d_in, d_out, n, active, blocks);
    CUDA_CHECK(cudaMemcpy(d_ref, d_out, n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    const float t_bl =
        time_variant<true>(d_in, d_out, n, active, blocks);
    const int miss = count_mismatch(d_ref, d_out, h_ref, h_out, n);
    bad += miss;

    if (active == 0)  t_light = t_div;
    if (active == 32) t_heavy = t_div;

    const char* paths = (active == 0 || active == 32) ? "1" : "2";
    const float ratio = (t_bl > 0.0f) ? (t_div / t_bl) : 0.0f;
    std::printf("%-8d %-10s %-16.4f %-16.4f %-12.4f %-12d %.2fx\n", active,
                paths, t_div, t_bl, t_div - t_bl, miss, ratio);
  }

  std::printf("\n两版输出比对：不一致元素共 %d 个（0 表示逐元素一致，"
              "各档明细见上表「不一致」一列）\n", bad);

  std::printf("\n--- 解读 ---\n");
  std::printf("1. active = 0 时全 warp 走轻路径，divergent 能整个跳过重路径的 32 条 FMA，\n");
  std::printf("   两版差距最大；active = 32 时全 warp 走重路径，重路径本来就非走不可，\n");
  std::printf("   divergent 能省的只剩轻路径那一条加法，差距又回到最窄。\n");
  std::printf("   纯轻路径 %.4f ms，纯重路径 %.4f ms（下一条解读会引用它）。\n",
              t_light, t_heavy);
  std::printf("2. active 在 1..31 之间时 warp 分裂为两条路径，硬件把两条路径的指令\n");
  std::printf("   都发射一遍。发射总量恒为 32 + 1 = 33 条、与 active 无关，所以耗时\n");
  std::printf("   近乎平坦，而其中轻路径只占 1 条，故这个平坦值逼近**重路径**那一档\n");
  std::printf("   （%.4f ms）。这正是有效并行度 32/N 的含义：warp 分裂后按最重的\n",
              t_heavy);
  std::printf("   那条路径计时。此时两版只差一条 SEL 的量级，分不出胜负。\n");
  std::printf("3. branchless 版本无论 active 取何值，都必须把两条路径都算一遍。\n");
  std::printf("   按模型的预期形状：它的耗时基本恒定、与 active 无关，且不低于\n");
  std::printf("   divergent。这是推导出的形状而非实测结论 —— 中间取值时 divergent\n");
  std::printf("   还要付 BSSY/BSYNC 与合流等待，ratio 略大于 1.00 属正常。\n");
  std::printf("   判读以 diff(ms) 一列为准：比值列会被每档固定的启动开销稀释。\n");
  std::printf("   这说明「用算术选择消除分支」对**长**分支并不免费：它把控制流分歧\n");
  std::printf("   换成了计算冗余。无分支写法的价值在短分支（省去 BSSY/BRA/BSYNC 的开销），\n");
  std::printf("   不在长分支。\n");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_ref));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_ref));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
