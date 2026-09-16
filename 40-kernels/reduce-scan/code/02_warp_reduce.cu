// 02_warp_reduce.cu —— warp shuffle 与 warp→block→grid 多级归约
//
// 配套文档：40-kernels/reduce-scan/02-warp-shuffle-and-hierarchy.md
//
// 主线：把「块内归约」从**共享内存树**换成 **warp shuffle + 每 warp 一个局部和**，
// 换来的是**屏障次数从 log2(T) 降到 1**、共享内存流量从 O(T) 降到 O(warp 数)。
// 全局访存字节数**两者相同**——归约的全局流量由问题规模决定，与块内怎么做无关。
//
// 本文件的三组对照（全部打印推导值）：
//   1. `__shfl_down_sync` / `__shfl_xor_sync` 的语义（含 `mask` 与 `width`）；
//   2. 两种块内形态的**屏障次数**与**共享内存字节数**；
//   3. `width` 用错的三种后果（其中只有一种是「偏大」）。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 02_warp_reduce && ./02_warp_reduce
//
// 对应 `verify.py` 的 §A3 / §B3 / §C2 / §C4。

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
// 形态 A：共享内存树（经典写法）
//
// 代价：log2(T) 轮，每轮一次读改写（8 B）＋一次块级屏障。
//   T=1024 时：10 轮 → 1023 次读改写 = 8184 B，加初值 1024×4 = 4096 B
//              = 12280 B；屏障 10~11 次。
// ---------------------------------------------------------------------------
__global__ void smem_tree_reduce(const float* __restrict__ g, int n,
                                 float* __restrict__ out)
{
    extern __shared__ float s[];
    const int tid = threadIdx.x;
    const int T = blockDim.x;

    float acc = 0.0f;
    for (int i = tid; i < n; i += T) acc += g[i];
    s[tid] = acc;

    for (int off = T / 2; off > 0; off >>= 1) {
        __syncthreads();                 // ← 每轮一次块级屏障
        if (tid < off) s[tid] += s[tid + off];
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ---------------------------------------------------------------------------
// 形态 B：warp shuffle + 每 warp 一个局部和
//
// 代价：warp 内 5 步 shuffle（**不进共享内存、不需要屏障**）＋
//       每 warp 写 1 个局部和（4 B）＋ 1 次块级屏障 ＋ 第二级 5 步 shuffle。
//   T=1024（32 warp）时：共享内存 32 写 + 32 读 = 256 B；屏障 **1** 次。
//
// ⚠ 第二级的写者是 32 个**不同 warp** 的 lane 0，所以那一次必须是**块级**屏障。
//   只有「同一个 warp 内的 lane 之间」才用 `__syncwarp()`。
// ---------------------------------------------------------------------------
__global__ void shuffle_tree_reduce(const float* __restrict__ g, int n,
                                    float* __restrict__ out)
{
    const int lane = threadIdx.x & (WARP - 1);
    const int wid = threadIdx.x / WARP;
    const int T = blockDim.x;
    const int nwarps = T / WARP;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += T) acc += g[i];

    // warp 内折半：offset 从 width/2 递减到 1（归约用 down）
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float warp_sums[WARP];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();                                   // ← 唯一的块级屏障

    if (wid == 0) {
        acc = (lane < nwarps) ? warp_sums[lane] : 0.0f; // 补 0（加法幺元）
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[blockIdx.x] = acc;
    }
}

// ---------------------------------------------------------------------------
// 形态 C：多级（grid）归约的第二级——用原子操作收尾
//
// 与「第二个 kernel」是同一件事的两条路线（详见 04 篇）：
//   · atomicAdd：少一次启动，但**加法顺序由硬件决定 → 不可复现**；
//   · 第二个 kernel：多一次启动（引用值 3~5 µs），但顺序写死 → 确定。
// 两条路线的**部分和字节数完全相同**（都是 2·G·4）。
// ---------------------------------------------------------------------------
__global__ void grid_reduce_atomic(const float* __restrict__ g, int n,
                                   float* __restrict__ total)
{
    extern __shared__ float s[];
    const int lane = threadIdx.x & (WARP - 1);
    const int wid = threadIdx.x / WARP;
    const int T = blockDim.x;
    const int nwarps = T / WARP;

    float acc = 0.0f;
    for (int i = blockIdx.x * T + threadIdx.x; i < n; i += gridDim.x * T)
        acc += g[i];
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < nwarps) ? s[lane] : 0.0f;
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        // ⚠ 这一条 atomicAdd 的**到达顺序由硬件决定** → 逐位不可复现
        if (lane == 0) atomicAdd(total, acc);
    }
}

// ---------------------------------------------------------------------------
// §2 `width` 的语义：它改变的是**段划分**，不是「跨 warp」
//
// `__shfl_down_sync(mask, v, delta, width)`：
//   · 源 lane = 段内相对下标 + delta；段是 warp 内**等长**的 width 条 lane；
//   · 源 lane 越出**本段**时**不回绕**，返回调用者**自己的值**；
//   · `width` 必须是 2 的幂且 ≤ 32，否则**未定义**（不是取模）。
// `__shfl_xor_sync(mask, v, laneMask, width)`：
//   · 源 lane = `laneid ^ laneMask`；`laneMask < width` 时必在同一段内，
//     所以这个变体**没有「越界保留自身值」这一说**——更抗误用。
//
// 三种后果（§C2 逐条断言，这里只打印结论）：
//   (1) offset 全部 < width  → **结果与 width 无关**（依赖锥恰是本段的 width 条 lane）；
//   (2) offset 全部 ≥ width  → 每一步全部 lane 越界 → `v += v` → 结果**偏大 2^k 倍**；
//   (3) width 写小且循环跟着 width 走 → **不偏大**，只是回答了另一个问题（最难发现）。
//
// ⚠ `__shfl_*` **永不跨 warp**，与 width 无关。把现象写成「跨 warp 读」
//   会让排错方向整个跑偏（去查 block 维度/网格配置，而错在 width 取值上）。
// ---------------------------------------------------------------------------

// 故意写错的版本：循环 offset 照 32 lane 抄，width 却写成 8。
// 预期结果 = 每段的真值 × 4（offset=16 与 offset=8 两步各翻一倍）。
__global__ void wrong_width_reduce(const float* __restrict__ g,
                                   float* __restrict__ out)
{
    const int lane = threadIdx.x & (WARP - 1);
    float v = g[threadIdx.x];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)         // ✗ 照 32 抄的循环
        v += __shfl_down_sync(0xffffffffu, v, off, 8);   // ✗ width 应该是 8 的段
    if ((lane & 7) == 0) out[threadIdx.x / 8] = v;  // 每段首 lane
}

// ---------------------------------------------------------------------------
// §4 「volatile + 双缓冲」的旧写法为什么在现代编译器下失效
//
// 现象：一份 Volta 之前的 warp 内归约代码，在 Pascal 上通过，在 Volta 及以后
//       随机出错（同一份二进制有时对有时错）。
//
// 成因（两层，缺一不可）：
//   · **硬件层**：Volta 的 independent thread scheduling（CC 7.0）取消了
//     「warp 内所有 lane 隐含地停在同一处」这条保证。lane A 写共享内存、
//     lane B 读，两者之间不再有任何顺序。`volatile` 只约束**编译器**不缓存，
//     它不是内存屏障、也不产生 lane 间的顺序——它**从来**不是同步手段。
//   · **编译器层**：`volatile` 只保证「每次访问都真的发出去」，不保证
//     「在正确的时刻发出去」。锁步时代的写法依赖硬件替它排序，
//     这个前提消失后，`volatile` 剩下的那点保证不足以补上顺序。
//
// 规避：warp 内交换用 `__shfl_*_sync` + `__syncwarp()`；带 `_sync` 后缀的
//       原语必须传**正确的掩码**——参与交换的 lane 都要在掩码里。
//       ⚠ 不要把 `__activemask()` 的返回值存起来复用：它是**此刻**活跃的 lane，
//         随发散状态变化，不是一个稳定常量。
// ---------------------------------------------------------------------------

__global__ void legacy_volatile_demo(const float* __restrict__ g,
                                     float* __restrict__ out)
{
    extern __shared__ volatile float vs[];
    const int lane = threadIdx.x & (WARP - 1);
    if (threadIdx.x < WARP) {
        vs[lane] = g[lane];
        // ✗ 旧写法：靠 volatile + （在没有发散的代码里）隐含的锁步来保证顺序
        //   现代写法：__syncwarp() 划界，或用 __shfl_*_sync 根本不用共享内存
        out[lane] = vs[(lane + 1) & (WARP - 1)];
    }
}

// ---------------------------------------------------------------------------
// host 侧：打印屏障次数与字节数
// ---------------------------------------------------------------------------

static void report_hierarchy(int N)
{
    std::printf("§2 两种块内形态的代价（N = %d 个 fp32，推导值）\n", N);
    std::printf("  %-8s %-8s | %-24s %-8s %-10s %-12s\n",
                "T(线程)", "warp 数", "形态", "屏障数", "smem 字节", "全局读字节");
    const int Ts[] = {128, 256, 1024};
    for (int k = 0; k < 3; ++k) {
        const int T = Ts[k], warp = T / WARP;
        int rounds = 0, s2 = T;
        while (s2 > 1) { s2 >>= 1; ++rounds; }
        const int tree_bar = 1 + rounds;
        const int tree_b = 4 * T + 8 * (T - 1);
        const int shfl_bar = 1;
        const int shfl_b = 4 * warp * 2;
        std::printf("  %-8d %-8d | %-24s %-8d %-10d %-12d\n",
                    T, warp, "共享内存树", tree_bar, tree_b, N * 4);
        std::printf("  %-8s %-8s | %-24s %-8d %-10d %-12d\n",
                    "", "", "shuffle + 每 warp 一和", shfl_bar, shfl_b, N * 4);
        std::printf("  → 屏障 %d → %d（差 %d 次）；smem %d B → %d B；"
                    "全局读两者相同\n",
                    tree_bar, shfl_bar, tree_bar - shfl_bar, tree_b, shfl_b);
    }
    std::printf("  ⚠ 数法：共享内存树 = log2(T) 轮 × 每轮一次 8 B 读改写"
                "（+ 初值 T 次写）+ 每轮 1 次屏障；\n");
    std::printf("     shuffle 版 = warp 内 5 步（无屏障、不过 smem）"
                "+ 每 warp 写 1 个（4 B）+ 1 次块级屏障。\n\n");
}

int main()
{
    std::printf("============================================================\n");
    std::printf("02_warp_reduce —— warp shuffle 与多级归约\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链）。\n");
    std::printf("  打印的是推导值，可用 `make verify` 的 §A3 / §B3 / §C2 / §C4 复核。\n\n");

    report_hierarchy(1 << 20);

    std::printf("§2 `__shfl_down_sync(mask, v, delta, width)` 的语义（规格事实）\n");
    std::printf("  · 源 lane = 段内相对下标 + delta；warp 按 width 切成等长的段；\n");
    std::printf("  · 源 lane 越出**本段**时**不回绕**，返回调用者**自己的值**；\n");
    std::printf("  · width 必须是 2 的幂且 ≤ 32，否则未定义；\n");
    std::printf("  · `mask` 必须包含所有参与交换的 lane（收敛要求）；\n");
    std::printf("  · **永不跨 warp**：交换范围恒为调用 warp 的 32 条 lane。\n");
    std::printf("  `__shfl_xor_sync` 的源 lane = laneid ^ laneMask，laneMask < width\n");
    std::printf("  时必在同一段内 → 没有「越界保留自身值」这一说。\n\n");

    std::printf("§2 width 用错的三种后果\n");
    std::printf("  (1) offset 全部 < width → 结果**与 width 无关**；\n");
    std::printf("  (2) offset ≥ width → 该步全部 lane 越界 → `v += v` → 结果**偏大**\n");
    std::printf("      （倍数 = 2^(超界的步数)，例：offsets=[16,8,4,2,1] + width=8 → 4 倍）；\n");
    std::printf("  (3) width 写小且循环跟着 width 走 → **不偏大**，只是换了分组（最难发现）。\n\n");

    // ---- 真机部分（本仓库没有跑过） ----
    const int n = 1 << 20, T = 256;
    const int blocks = (n + T - 1) / T;
    float* dg = nullptr; float* dout = nullptr; float* dtotal = nullptr;
    CUDA_CHECK(cudaMalloc(&dg, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(dg, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dtotal, sizeof(float)));

    smem_tree_reduce<<<blocks, T, T * sizeof(float)>>>(dg, n, dout);
    shuffle_tree_reduce<<<blocks, T>>>(dg, n, dout);
    CUDA_CHECK(cudaMemset(dtotal, 0, sizeof(float)));
    grid_reduce_atomic<<<blocks, T, (T / WARP) * sizeof(float)>>>(dg, n, dtotal);
    wrong_width_reduce<<<1, WARP>>>(dg, dout);
    legacy_volatile_demo<<<1, WARP, WARP * sizeof(float)>>>(dg, dout);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h, dtotal, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("真机输出（本仓库没有跑过，此处只作占位说明）：\n");
    std::printf("  atomicAdd 收尾的全局和 = %.6f\n", h);
    std::printf("  把同一个输入跑两遍、逐位比较：`atomicAdd` 路线**可能不同**\n");
    std::printf("  （见 04 篇；注意「跑两遍恰好相同」不能反过来证明确定性）\n");

    CUDA_CHECK(cudaFree(dg));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dtotal));
    return 0;
}
