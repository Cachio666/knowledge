// 03_scan.cu —— Hillis-Steele 与 Blelloch 扫描（前缀和）
//
// 配套文档：40-kernels/reduce-scan/03-scan-algorithms.md
//
// 主线：两种扫描的 **work 与 depth 都不同**——
//   Hillis-Steele（Kogge-Stone）：work = n·log n，depth = **log n**
//   Blelloch（Brent-Kung）      ：work = 2n−2  ，depth = **2·log n**
// n = 2^10 时是 10240 vs 2046（比值 5.005），depth 是 **10 vs 20**。
//
// ⚠ 两者的 depth **不相等**（都在 Θ(log n) 里，但常数差一倍）。
//   用 Brent 模型 T = work/p + depth 看：p 小（work 主导）→ Blelloch 赢；
//   p 大（depth 主导）→ Hillis-Steele 赢，临界 p = n(1 − 2/log2 n) **逼近 n 本身**。
//   换句话说：**Blelloch 省 work，Hillis-Steele 省步数**，两者不是「谁更快」。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 03_scan && ./03_scan
//
// 对应 `verify.py` 的 §A2 / §B1 / §B2 / §B7 / §C8。

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// Hillis-Steele（Kogge-Stone）：inclusive scan
//
// 第 k 步（k = 0,1,…）里 i 读 i−2^k 并相加。log2(n) 步，每步 n 次加法
// → work = n·log n（**发射口径**：每条 lane 每步都执行一次加法，越界的用
//   谓词屏蔽；真正有效的加法次数是 n·log n − (n−1)）。
// depth = log2(n)。
//
// 代价：n = 2^10 时 10 步 × 2n 次共享内存访问 = 20480 次（读+写各一）。
// 优点：**只有 log2(n) 步** → 屏障只有 log2(n) 次。
// ---------------------------------------------------------------------------
__global__ void hillis_steele_scan(const float* __restrict__ in,
                                   float* __restrict__ out, int n)
{
    extern __shared__ float s[];
    const int i = threadIdx.x;
    if (i < n) s[i] = in[i];
    __syncthreads();

    for (int off = 1; off < n; off <<= 1) {
        float prev = 0.0f;
        if (i >= off) prev = s[i - off];   // 越界的 lane 不做加法（谓词屏蔽）
        __syncthreads();                   // ← 先读旧值，再写
        if (i >= off) s[i] += prev;
        __syncthreads();
    }
    if (i < n) out[i] = s[i];
}

// ---------------------------------------------------------------------------
// Blelloch（Brent-Kung）：exclusive scan，两阶段
//
//   up-sweep  ：自底向上两两求和，n/2 + n/4 + … + 1 = n−1 次加法，log2(n) 层
//   归零      ：根置幺元 0
//   down-sweep：自顶向下分发，每层 n/(2d) 个结点、每结点 1 次加法
//               （另有一次**移位**，移位不是加法）→ 合计 n−1 次
// → work = 2n−2，depth = 2·log2(n)
//
// ⚠ 两阶段之间必须有一次 `__syncthreads()`：up-sweep 的最后一个写者与
//   down-sweep 的第一个读者跨 warp。漏掉它不会报错，只会给出另一种答案。
// ⚠ down-sweep 里两条赋值的**右值**不能写反：
//     t = s[i+d-1]; s[i+d-1] = s[i+2d-1]; s[i+2d-1] = t + s[i+2d-1];
//   若第二条想用「新的 s[i+d-1]」，结果不再是前缀和（而首项往往仍然对）。
// ---------------------------------------------------------------------------
__global__ void blelloch_scan(const float* __restrict__ in,
                              float* __restrict__ out, int n)
{
    extern __shared__ float s[];
    const int i = threadIdx.x;
    if (i < n) s[i] = in[i];
    __syncthreads();

    // ---- up-sweep ----
    for (int d = 1; d < n; d <<= 1) {
        const int k = 2 * d * (i + 1) - 1;      // 每层的右端点
        if (k < n) s[k] += s[k - d];
        __syncthreads();
    }

    // ---- 根置幺元 + **两阶段之间的块级屏障** ----
    if (i == 0) s[n - 1] = 0.0f;
    __syncthreads();                             // ← 漏掉这一句是本篇的头号坑

    // ---- down-sweep ----
    for (int d = n >> 1; d >= 1; d >>= 1) {
        const int k = 2 * d * (i + 1) - 1;
        if (k < n) {
            const float t = s[k - d];            // 旧值必须先取出来
            s[k - d] = s[k];
            s[k] = t + s[k];
        }
        __syncthreads();
    }

    if (i < n) out[i] = s[i];                    // exclusive
}

// ---------------------------------------------------------------------------
// §2 warp 内扫描（Hillis-Steele 的 shuffle 形态）
//
// **扫描用 up shuffle、offset 从 1 递增**；归约用 down、offset 从 width/2 递减。
// 两者方向相反，抄错了不会报错，只会得到另一种答案。
//
// 守卫 `if (rel >= off)` 是**正确性**的一部分，不是性能优化：
// `__shfl_up_sync` 在段内越下界时返回**调用者自己的值**，于是 lane 0
// 在每个 step 都会执行 `v += v` —— 5 个 step 下来恰好把 x[0] 计入 32 次。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_inclusive_scan(float v, int lane)
{
    #pragma unroll
    for (int off = 1; off < WARP; off <<= 1) {
        const float t = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += t;                 // ← 守卫：越界时**不加**
    }
    return v;
}

// 块内扫描：warp 内扫描 → 各 warp 的总和 → 扫描 warp 和 → 加回偏移。
// 这是「先 shuffle 内扫描 + 跨 warp 前缀」的混合形态（CUB BlockScan 的做法），
// 而不是直接在块内跑 Blelloch：n ≈ T 时 depth 主导，HS 形态的步数少一半。
//
// ⚠ 屏障数与**归约**不同，值得单独记一笔：
//     归约的第二级只产出一个标量（warp 0 写全局就完了）→ **1 次**块级屏障；
//     扫描的第二级产出的偏移必须**广播回所有 warp** → **2 次**块级屏障。
//   这是「扫描比归约贵」的一个常被忽略的来源（另一处是 §2 的 smem 流量）。
__global__ void block_scan_hybrid(const float* __restrict__ in,
                                  float* __restrict__ out, int n)
{
    __shared__ float warp_total[WARP];           // 每 warp 的总和
    __shared__ float warp_off[WARP];             // 每 warp 的 exclusive 偏移
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid = tid / WARP;
    const int nwarps = blockDim.x / WARP;

    float v = (tid < n) ? in[tid] : 0.0f;
    v = warp_inclusive_scan(v, lane);            // 第 1 级：warp 内扫描

    if (lane == WARP - 1) warp_total[wid] = v;   // lane 31 持有本 warp 的总和
    __syncthreads();                             // 屏障 1：局部和要跨 warp 可见

    if (wid == 0) {                              // 第 2 级：扫描 warp 总和
        const float w = (lane < nwarps) ? warp_total[lane] : 0.0f;
        const float inc = warp_inclusive_scan(w, lane);
        const float prev = __shfl_up_sync(0xffffffffu, inc, 1);
        // exclusive：不含自己。lane 0 的偏移是 0（加法的幺元）
        warp_off[lane] = (lane == 0) ? 0.0f : prev;
    }
    __syncthreads();                             // 屏障 2：偏移要广播回所有 warp

    if (tid < n) out[tid] = v + warp_off[wid];   // 第 3 级：加回块内偏移
}

// ---------------------------------------------------------------------------
// §4 失败模式：「把 inclusive / exclusive 的位移方向搞反」
//
// 两个常见形态：
//   (a) 块偏移该用 **exclusive**（不含本块）却用了 inclusive（含本块）
//       → 结果整体偏一个块的量，**首元素就不再是 0**（自检能抓到）；
//   (b) 扫描用了 **down** shuffle（归约的写法）→ lane 0 恰好等于全和
//       （正好等于 inclusive 扫描的**最后一项**），而 lane 31 = 32·x[31]。
//       单点自检「查首项」抓不到的是 (c)：
//   (c) `width` 不匹配导致整段一致地偏大（见 02 篇 §C2）——连首项都「成比例地错」。
// ---------------------------------------------------------------------------
__global__ void wrong_direction_scan(const float* __restrict__ in,
                                     float* __restrict__ up_out,
                                     float* __restrict__ down_out, int n)
{
    const int i = threadIdx.x;
    if (i >= n) return;
    // 正确形态
    up_out[i] = warp_inclusive_scan(in[i], i);
    // 方向搞反的形态：用 down shuffle 从 1 开始递增
    float v = in[i];
    #pragma unroll
    for (int off = 1; off < WARP; off <<= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    down_out[i] = v;
}

// ---------------------------------------------------------------------------
// host 侧：work / depth 表 与 Brent 模型的临界 p
// ---------------------------------------------------------------------------

static void report_work_depth(void)
{
    std::printf("§1 Hillis-Steele 与 Blelloch 的 work / depth（推导值）\n");
    std::printf("  %-10s %-7s | %-12s %-12s %-9s | %-14s %-15s | %-8s\n",
                "n", "log2 n", "HS work", "HS 有效加法", "HS depth",
                "Blelloch work", "Blelloch depth", "work 比");
    const int ks[] = {10, 16, 20};
    for (int t = 0; t < 3; ++t) {
        const long long k = ks[t];
        const long long n = 1LL << k;
        const long long hs_work = n * k;             // 发射口径
        const long long hs_valid = n * k - (n - 1);  // 有效加法
        const long long bl_work = 2 * n - 2;         // up (n−1) + down (n−1)
        std::printf("  %-10lld %-7lld | %-12lld %-12lld %-9lld | %-14lld %-15lld | %-8.3f\n",
                    n, k, hs_work, hs_valid, k, bl_work, 2 * k,
                    (double)hs_work / bl_work);
    }
    std::printf("  ⚠ depth **不相等**：10 vs 20、16 vs 32、20 vs 40。\n");
    std::printf("    都在 Θ(log n) 里，但常数差一倍——不是「depth 相同、只差 work」。\n");
    std::printf("  ⚠ work 用**发射口径** n·log n（每条 lane 每步都发射一次加法，\n");
    std::printf("    越界用谓词屏蔽）；真正有效的加法次数是 n·log n − (n−1)。\n\n");
}

static void report_brent(void)
{
    std::printf("§1 Brent 模型 T = work/p + depth：临界 p = n(1 − 2/log2 n)\n");
    const int ks[] = {10, 16, 20};
    for (int t = 0; t < 3; ++t) {
        const long long k = ks[t], n = 1LL << k;
        const double crit = (double)n * (1.0 - 2.0 / k);
        std::printf("  n=2^%-3lld 临界 p = %12.1f（= n 的 %.0f%%）\n",
                    k, crit, 100.0 * crit / n);
        const double ps[] = {10.0, crit - 1, crit + 1};
        for (int j = 0; j < 3; ++j) {
            const double p = ps[j];
            if (p < 1) continue;
            const double ths = (double)(n * k) / p + k;
            const double tbl = (double)(2 * n - 2) / p + 2.0 * k;
            std::printf("      p=%-12.0f HS=%-12.4f Blelloch=%-12.4f → %s 快 %.3fx\n",
                        p, ths, tbl, (ths < tbl) ? "Hillis-Steele" : "Blelloch",
                        (ths > tbl ? ths / tbl : tbl / ths));
        }
    }
    std::printf("  ⚠ 方向：work 项是 work/p，**p 越大它越不重要**。所以\n");
    std::printf("      p 小（work 主导）→ **Blelloch 赢**（work 是 2n 而非 n·log n）；\n");
    std::printf("      p 大（depth 主导）→ **Hillis-Steele 赢**（depth 只有一半）。\n");
    std::printf("    临界 p 逼近元素个数 n 本身，且 p=n 处的优势只有 5%~10%。\n\n");
}

int main()
{
    std::printf("============================================================\n");
    std::printf("03_scan —— Hillis-Steele 与 Blelloch\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链）。\n");
    std::printf("  打印的是推导值，可用 `make verify` 的 §A2 / §B1 / §B2 / §B7 / §C8 复核。\n\n");

    report_work_depth();
    report_brent();

    std::printf("§2 共享内存流量（推导值）：每步 n 读 + n 写\n");
    const int ks[] = {5, 8, 10};
    for (int t = 0; t < 3; ++t) {
        const int k = ks[t], n = 1 << k;
        const int hs = 2 * n * k, bl = 2 * (2 * n - 2);
        std::printf("  n=2^%-3d HS %2d 步 / %6d 次 smem 访问 | Blelloch %2d 步 / %6d 次"
                    " | 比值 %.2f\n",
                    k, k, hs, 2 * k, bl, (double)hs / bl);
    }
    std::printf("  ⚠ 步数 = 屏障次数。n=2^10 时 HS 10 次、Blelloch 20 次。\n");
    std::printf("    单块内 n ≈ T 时 depth 主导 → 用「warp 内 shuffle 扫描 +\n");
    std::printf("    跨 warp 前缀」的混合形态（见 block_scan_hybrid），而不是直接 Blelloch。\n\n");

    // ---- 真机部分（本仓库没有跑过） ----
    const int n = 1024;
    float *din = nullptr, *dout = nullptr, *dup = nullptr, *ddown = nullptr;
    CUDA_CHECK(cudaMalloc(&din, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dup, WARP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddown, WARP * sizeof(float)));
    CUDA_CHECK(cudaMemset(din, 1, n * sizeof(float)));

    hillis_steele_scan<<<1, n, n * sizeof(float)>>>(din, dout, n);
    blelloch_scan<<<1, n, n * sizeof(float)>>>(din, dout, n);
    block_scan_hybrid<<<1, n>>>(din, dout, n);
    wrong_direction_scan<<<1, WARP>>>(din, dup, ddown, WARP);
    CUDA_CHECK(cudaDeviceSynchronize());

    float a = 0.0f, b = 0.0f;
    CUDA_CHECK(cudaMemcpy(&a, dup, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&b, ddown, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("真机输出（本仓库没有跑过，此处只作占位说明）：\n");
    std::printf("  输入全 1（n=32）：正确扫描的 lane 0 = 1.0，方向搞反的 down 版 lane 0 = %.1f\n",
                a);
    std::printf("  （down 版 lane 0 恰好等于全和，正好等于 inclusive 扫描的**最后一项**，\n");
    std::printf("    所以「查首项」这种单点自检在别处能骗过；lane 31 = %.1f = 32×x[31]）\n\n", b);

    std::printf("§4 失败模式清单（代码见 wrong_direction_scan / blelloch_scan 的注释）\n");
    std::printf("  · Blelloch 的 up-sweep 与 down-sweep 之间漏掉 __syncthreads()；\n");
    std::printf("  · inclusive / exclusive 的位移方向搞反（块偏移整体偏一块）；\n");
    std::printf("  · 扫描用 down shuffle（方向反了）；\n");
    std::printf("  · warp 内扫描漏掉 `if (lane >= off)` 守卫 → lane 0 被计入 32 次。\n");

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dup));
    CUDA_CHECK(cudaFree(ddown));
    return 0;
}
