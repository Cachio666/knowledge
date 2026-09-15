// 03_bank_conflict_demo.cu —— 共享内存 bank 冲突与 padding
// 对应文档：03-shared-memory-bank-conflict.md
//
// 本程序做两件事（与文档「编译与运行」一致）：
//   1. 扫描行距：同一份 32 行的 tile，只改列数（行距取 32 / 33 / 34 / 36），
//      跑同一个「lane 按列取用」的循环，打印每档耗时，并把 §1 第三步的
//      预测 gcd(行距, 32) = 32 / 1 / 2 / 4 并列摆出来（它同时就是以行距 33
//      为 1 的预期耗时比，即 32 : 1 : 2 : 4）。另跑一个「行距 32、行数改成
//      33」的配置（[33][32]），它应与行距 32 那一档完全相同 —— 决定冲突的
//      是行距，不是行数（§4.1）。
//   2. 换 64 位元素（double）重跑同一个按列取用，用于观察 §4.3：
//      这一版的耗时不会按 32 倍给出。
//
// 观测要点：
//   1. 行距为 32 时，warp 内 32 个 lane 读取同一列会全部撞到同一个 bank
//      —— bank = (lane × 32 + c) % 32 = c，32 路冲突
//   2. 行距改为 33 后，bank = (lane × 33 + c) % 32 = (lane + c) % 32
//      —— 随 lane 遍历全部 bank，无冲突
//   3. 两种情形的计算完全相同，只有共享内存的行距不同
//
// 本程序只打印耗时与耗时比值：单次访问的绝对周期数（20～30、56 cycles）
// 与交付率（4、12.8 字节/周期）都是文档 §2 由公开规格推出来的量级，
// 程序没有手段直接读出它们。
//
// 构建：make 03_bank_conflict_demo
// 运行：./03_bank_conflict_demo

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

constexpr int kThreads = 256;  // 8 个 warp
constexpr int kWarps   = kThreads / 32;
constexpr int kRows    = 32;   // 参与比较的 tile 行数
constexpr int kReps    = 200;  // 反复执行，让共享内存访问占主导
constexpr int kTimes   = 10;   // 计时循环的启动次数

// 每个 warp 独占一块 rows × cols 的 tile，行距 = cols（列数）。
// 关键访问是 tile[warp][lane][r]：固定列 r，行随 lane 变化。
//   cols = 32: 地址 = warp×1024 + lane×32 + r，bank = r            → 32 路冲突
//   cols = 33: 地址 = warp×1056 + lane×33 + r，bank = (lane + r) % 32 → 无冲突
// （warp 偏移 = warp × rows × cols 个字。cols 是 32 的倍数时它不影响 bank
//   计算；cols 不是时也只是给每个 warp 加同一个常量偏移。）
template <typename T>
__global__ void column_read_kernel(const T* __restrict__ in,
                                   T* __restrict__ out, int rows, int cols,
                                   int n) {
  // 以最宽的元素类型声明，保证起始地址对齐；float / double 都从这里切。
  extern __shared__ double smem[];
  T* tile = reinterpret_cast<T*>(smem);

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  T* my = tile + static_cast<size_t>(warp) * rows * cols;

  T acc = static_cast<T>(0);

  for (int rep = 0; rep < kReps; ++rep) {
    // 装载：读全局内存写入共享内存。按 [行][列] 顺序，warp 内 lane 对应列，
    // 全局侧是合并访问，共享侧无冲突。
#pragma unroll 4
    for (int r = 0; r < rows; ++r) {
      const int g = static_cast<int>(
          ((static_cast<long long>(rep) * rows + r) * 32 + lane) % n);
      my[r * cols + lane] = in[g];
    }
    __syncthreads();

    // 读取：固定列 r，行随 lane 变化 —— 这是产生冲突的访问模式
#pragma unroll 4
    for (int r = 0; r < rows; ++r) {
      acc += my[lane * cols + r];
    }
    __syncthreads();
  }

  if (acc == static_cast<T>(-1234567.0)) out[0] = acc;
}

// 返回单次启动的平均耗时。共享内存读取次数与行距无关：
// 每线程每轮读 rows 次，共 kReps 轮。
template <typename T>
float time_case(const T* d_in, T* d_out, int n, int blocks, int rows, int cols,
                size_t smem_bytes) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  column_read_kernel<T>
      <<<blocks, kThreads, smem_bytes>>>(d_in, d_out, rows, cols, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(begin));
  for (int r = 0; r < kTimes; ++r) {
    column_read_kernel<T>
        <<<blocks, kThreads, smem_bytes>>>(d_in, d_out, rows, cols, n);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUDA_CHECK_LAST();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return ms / kTimes;
}

// §1 第三步的串行遍数预测：gcd(行距, 32)
int serial_passes(int stride) {
  int a = stride;
  int b = 32;
  while (b != 0) {
    const int t = a % b;
    a           = b;
    b           = t;
  }
  return a == 0 ? 32 : a;
}

}  // namespace

int main() {
  constexpr int n      = 1 << 22;  // 4M 元素
  constexpr int blocks = 2048;

  float* d_in  = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0, n * sizeof(float)));

  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("设备: %s\n", p.name);
  std::printf("每 block %d 线程（%d 个 warp），每 warp 独占一块「行 × 列」的 tile；\n",
              kThreads, kWarps);
  std::printf("每 block 动态共享内存 = warp 数 × 行数 × 列数 × 元素宽度。\n\n");

  // ---- 行距扫描：同一份 32 行的 tile，只改列数 ----
  // 最后一档把行数改成 33、行距仍是 32，用来演示「决定冲突的是行距」。
  const int   lanes[]  = {32, 33, 34, 36, 32};
  const int   rows[]   = {kRows, kRows, kRows, kRows, kRows + 1};
  const char* labels[] = {"行距 32", "行距 33", "行距 34", "行距 36",
                          "行距 32、行数 33（[33][32]）"};
  constexpr int kCases = static_cast<int>(sizeof(lanes) / sizeof(lanes[0]));

  float ms_f[kCases];
  for (int i = 0; i < kCases; ++i) {
    const size_t smem = static_cast<size_t>(kWarps) * rows[i] * lanes[i] *
                        sizeof(float);
    ms_f[i] =
        time_case<float>(d_in, d_out, n, blocks, rows[i], lanes[i], smem);
  }

  // 以「行距 33」（无冲突）为基准 —— 它的串行遍数是 1
  const float base_ms = ms_f[1];

  std::printf("--- 32 位（float）元素：只改列数（行距）---\n");
  std::printf("%-28s %-8s %-8s %-16s %-14s %s\n", "配置", "行距", "行数",
              "串行遍数预期", "耗时(ms)", "相对耗时");
  for (int i = 0; i < kCases; ++i) {
    std::printf("%-28s %-8d %-8d %-16d %-14.4f %.3f\n", labels[i], lanes[i],
                rows[i], serial_passes(lanes[i]), ms_f[i], ms_f[i] / base_ms);
  }
  std::printf("「串行遍数预期」= gcd(行距, 32)（§1 第三步），它同时就是以\n");
  std::printf("「行距 33」那一档为 1 的预期耗时比，即 32 : 1 : 2 : 4。\n");

  std::printf("\n--- 解读（32 位）---\n");
  std::printf("五种配置的计算量完全相同，只有行距不同：32 与 33。\n");
  std::printf("行距 32 与 bank 数 32 相等，导致固定列访问时 lane 与 lane\n");
  std::printf("全部映射到同一个 bank，32 次访问被串行化为 32 轮。\n");
  std::printf("行距 33 与 32 互质，bank 随 lane 遍历 32 个位置，无冲突。\n");
  std::printf("行距 34 / 36 的 gcd 分别是 2 / 4，预期耗时比落在 1 与 32 之间 ——\n");
  std::printf("四档一起给出 gcd 判据的完整形状。\n");
  std::printf("最后一行 [33][32] 把行数改成 33、行距仍是 32，实测应与「行距 32」\n");
  std::printf("那一档基本相同 —— 决定冲突的是行距，不是行数（§4.1）。\n\n");
  std::printf("关于「相对耗时」：不必期待它恰好等于「串行遍数预期」。行距为 32 时\n");
  std::printf("一行的 32 个元素在共享内存里连续，编译器可能把 32 次 LDS.32 向量化成\n");
  std::printf("8 次 LDS.128，冲突形态变成「16 路 × 2 相位」，实测比值会小于 32 倍的\n");
  std::printf("直觉值。看量级即可。\n");
  std::printf("还要注意这个比值是**吞吐侧**的预测：该循环反复读同一块 tile，瓶颈落在\n");
  std::printf("共享内存端口上（§2 的两笔账）。若实测比值远小于预期，先怀疑这个前提\n");
  std::printf("没成立；**延迟侧**的预测比值要小得多，约 2.2 : 1。\n");
  std::printf("padding 会让共享内存用量增加 1/32，可能影响占用率 —— 这也是\n");
  std::printf("「加 padding 不总是有益」的原因（§1 第六步、§4.2）。\n");
  std::printf("全局侧每次只读同一段约 800 KB 的数据（恒在 L2 命中），这是刻意的 ——\n");
  std::printf("让共享内存访问而非全局访存主导耗时。\n");

  // ---- 64 位（double）元素：同样的按列取用 ----
  // 8 warp × 32 行 × 33 列 × 8 字节 = 66 KB，越过了动态共享内存 48 KB 的
  // 默认上限，必须先申请提额（上限见 sharedMemPerBlockOptin）。
  const size_t smem64_max =
      static_cast<size_t>(kWarps) * kRows * 33 * sizeof(double);
  CUDA_CHECK(cudaFuncSetAttribute(column_read_kernel<double>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem64_max)));

  double* d_in64  = nullptr;
  double* d_out64 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out64, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_in64, 0, n * sizeof(double)));

  const int   lanes64[]  = {32, 33};
  const char* labels64[] = {"行距 32（double）", "行距 33（double）"};
  constexpr int kCases64 = static_cast<int>(sizeof(lanes64) / sizeof(lanes64[0]));

  float ms_d[kCases64];
  for (int i = 0; i < kCases64; ++i) {
    const size_t smem =
        static_cast<size_t>(kWarps) * kRows * lanes64[i] * sizeof(double);
    ms_d[i] = time_case<double>(d_in64, d_out64, n, blocks, kRows, lanes64[i],
                                smem);
  }

  std::printf("\n--- 64 位（double）元素：同样的按列取用 ---\n");
  std::printf("每 block 动态共享内存最大需 %.1f KB，已用 cudaFuncSetAttribute 提额。\n",
              smem64_max / 1024.0);
  std::printf("%-28s %-8s %-8s %-16s %-14s %s\n", "配置", "行距", "行数",
              "串行遍数预期", "耗时(ms)", "相对耗时");
  for (int i = 0; i < kCases64; ++i) {
    std::printf("%-28s %-8d %-8d %-16d %-14.4f %.3f\n", labels64[i],
                lanes64[i], kRows, serial_passes(lanes64[i]), ms_d[i],
                ms_d[i] / ms_d[1]);
  }

  std::printf("\n--- 解读（64 位）---\n");
  std::printf("「串行遍数预期」一列仍是 §1 的 4 字节模型，它不能直接套到 64 位访问上：\n");
  std::printf("一条 64 位访问被拆成两个 half-warp（各 16 个 lane）独立发出，判定池从\n");
  std::printf("32 个 lane 缩到 16 个；同时每个 lane 的 8 字节横跨两个相邻 bank，一次\n");
  std::printf("64 位访问的地板被抬到 2 遍。两笔账方向相反，所以「行距 32」那一档的\n");
  std::printf("实测比值不会按 32 倍给出（§4.3）。\n");

  CUDA_CHECK(cudaFree(d_in64));
  CUDA_CHECK(cudaFree(d_out64));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
