// 01_reduce_basics.cu —— 树形归约、segmented 归约、argmax 归约
//
// 配套文档：40-kernels/reduce-scan/01-reduction-algorithms.md
//
// 主线：顺序归约与树形归约的 **work 完全相同**（都是 n−1 次加法），
// 改变的只有 **depth**（n−1 → log2 n）。所以「并行归约」能做的唯一一件事
// 是把**顺序依赖**拆短——work 不能靠并行减少。
//
// 本文件同时给出两个下游要用的**变体**（它们不是「同一公式换个参数」）：
//   · segmented（grouped）归约：变长行、段边界与 warp 边界不对齐；
//   · argmax 归约：值必须与索引**配对**归约，且 NaN 与并列值的口径必须显式定义。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 01_reduce_basics && ./01_reduce_basics
//
// 本文件打印的全是**推导值**（由下面的公式与常量算出），可用于核对
// `verify.py` 的 §A1 / §A4 / §A5 / §B4 / §B5 / §C3。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math_constants.h>

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
// §1 两种归约的形态：work 相同、depth 不同
//
//   顺序：acc = x[0]; for i in 1..n-1: acc += x[i];
//         work = n-1，depth = n-1（每个加法依赖上一个）
//   树形：两两配对折半，每层 n_i/2 次加法
//         work = n-1，depth = ceil(log2 n)
//
// 关键：**work 一项完全相同**。所以并行化减少的不是工作量，是依赖链长度。
// 下面两个 kernel 的加法次数一样，但一个必须串行、一个可以并行。
// ---------------------------------------------------------------------------

// 顺序形态：一个线程吃整段。
__global__ void seq_reduce_kernel(const float* __restrict__ x, int n,
                                  float* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float acc = x[0];
    for (int i = 1; i < n; ++i) acc += x[i];      // depth = n-1
    out[0] = acc;
}

// 每线程先串行吃 n/p 个（work = n/p - 1，depth = n/p - 1），
// 再做 log2(p) 步的 warp 内折半（work = p-1，depth = log2 p）。
// 总 work = (n/p-1)*p + (p-1) = n-1 —— **与顺序形态一模一样**。
__global__ void tree_reduce_kernel(const float* __restrict__ x, int n,
                                   float* __restrict__ out)
{
    const int lane = threadIdx.x & (WARP - 1);
    float acc = 0.0f;

    // 第 1 段：每线程吃 n/p 个元素，串行累加（这一段无法并行）
    const int stride = blockDim.x;
    for (int i = threadIdx.x; i < n; i += stride) acc += x[i];

    // 第 2 段：warp 内折半（段内每步少一半活跃 lane）
    //   ⚠ 归约用 **down** shuffle、offset 从 width/2 递减到 1
    //     （扫描用 up、offset 从 1 递增——两者方向相反，见 03 篇）
    for (int offset = WARP / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);

    // 第 3 段：每 warp 一个局部和，落共享内存，再由 warp 0 折半
    //   写者是 32 个**不同 warp** 的 lane 0 → 这里必须有**块级**屏障
    __shared__ float warp_sums[WARP];
    const int wid = threadIdx.x / WARP;
    const int nwarps = blockDim.x / WARP;
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();                       // ← 唯一的块级屏障

    if (wid == 0) {
        acc = (lane < nwarps) ? warp_sums[lane] : 0.0f;   // ← 补 0（加法幺元）
        for (int offset = WARP / 2; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        if (lane == 0) out[blockIdx.x] = acc;
    }
}

// ---------------------------------------------------------------------------
// §1 变体一：segmented / grouped 归约（变长序列的 per-row reduce）
//
// 形态：`row_start[r]` / `row_len[r]`，对每一行独立归约。
// **段边界与 warp 边界不对齐**是常态（行长可以是 5、17、33…），
// 所以「拿 threadIdx/32 当段号」是错的——那样会把相邻两行串起来。
//
// 做法：先定段（每线程先算自己属于哪一行），再在段内归约。
// 本 kernel 用「一个 block 处理一行」的最简形态：段号由 blockIdx 给出，
// 段内先补零到 32 的倍数、每 lane 串行吃 per 个，再 shuffle 折半。
//
// ⚠ 「补零」是**与算子绑定**的：加法补 0、max 补 −inf、min 补 +inf。
//   换算子时忘掉这一步，短行会被补值顶替（见 01 篇 §4）。
// ---------------------------------------------------------------------------

__global__ void segmented_sum_kernel(const float* __restrict__ x,
                                     const int* __restrict__ row_start,
                                     const int* __restrict__ row_len,
                                     float* __restrict__ out_rows)
{
    const int row = blockIdx.x;
    const int beg = row_start[row];
    const int len = row_len[row];
    const int lane = threadIdx.x & (WARP - 1);
    const int wid = threadIdx.x / WARP;
    const int nwarps = blockDim.x / WARP;

    // 段内每 lane 串行吃 per 个（越界补 0 —— 加法的幺元）
    const int per = (len + blockDim.x - 1) / blockDim.x;
    float acc = 0.0f;
    for (int k = 0; k < per; ++k) {
        const int idx = wid * WARP + lane + k * blockDim.x;
        if (idx < len) acc += x[beg + idx];
    }
    for (int offset = WARP / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);

    __shared__ float warp_sums[WARP];
    if (lane == 0) warp_sums[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        for (int offset = WARP / 2; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        if (lane == 0) out_rows[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// §1 变体二：argmax 归约（值必须与索引**配对**）
//
// 单归约 max 再回头找索引是**错的**：并列值时「谁赢」由实现里的 `>` / `>=`
// 决定，而不是由文档声明的 tie 规则决定。必须把 (value, index) 当成一个
// 整体一起 shuffle、一起比较。
//
// 本文件显式声明口径（两个轴都要选，且必须写进文档）：
//   · NaN 口径：**忽略 NaN**（IEEE 754-2008 maxNum / `fmaxf` 语义）
//     ——另一个口径是「传播 NaN」（IEEE 754-2019 maximum / `np.maximum`），
//       二者在同一输入上给出**不同的索引**。
//   · 并列口径：**取最小索引**（first / argmin-of-index）
//     ——另一个口径是取最大索引。
//
// 比较函数是配对的：先比值，值相等时比索引。
// ---------------------------------------------------------------------------

struct ValIdx { float v; int i; };

__device__ __forceinline__ ValIdx pair_max(ValIdx a, ValIdx b)
{
    // NaN 口径：忽略 NaN（fmaxf 语义：一个参数是 NaN 时返回另一个）。
    // ⚠ 注意这与 `a.v > b.v ? a : b` 不同——后者的 NaN 行为由比较结果决定。
    const bool an = isnan(a.v), bn = isnan(b.v);
    if (an && bn) return (a.i <= b.i) ? a : b;      // 都是 NaN → 比索引
    if (an) return b;
    if (bn) return a;
    if (a.v > b.v) return a;
    if (b.v > a.v) return b;
    return (a.i <= b.i) ? a : b;                    // 并列 → 取最小索引
}

__global__ void argmax_kernel(const float* __restrict__ x, int n,
                              float* __restrict__ out_v,
                              int* __restrict__ out_i)
{
    const int lane = threadIdx.x & (WARP - 1);
    ValIdx best = {-CUDART_INF_F, 0x7fffffff};

    // 每线程先串行找局部最优（值与索引**同时**推进）
    //   起始值的 v 用 −inf 而不是 0：输入可能全为负。
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        best = pair_max(best, ValIdx{x[i], i});

    // 配对 shuffle：值和索引走同一条 shuffle，谁也不能落下
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        ValIdx o;
        o.v = __shfl_down_sync(0xffffffffu, best.v, offset);
        o.i = __shfl_down_sync(0xffffffffu, best.i, offset);
        best = pair_max(best, o);
    }

    __shared__ ValIdx warp_best[WARP];
    const int wid = threadIdx.x / WARP;
    const int nwarps = blockDim.x / WARP;
    if (lane == 0) warp_best[wid] = best;
    __syncthreads();
    if (wid == 0) {
        best = (lane < nwarps) ? warp_best[lane]
                               : ValIdx{-CUDART_INF_F, 0x7fffffff};
        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
            ValIdx o;
            o.v = __shfl_down_sync(0xffffffffu, best.v, offset);
            o.i = __shfl_down_sync(0xffffffffu, best.i, offset);
            best = pair_max(best, o);
        }
        if (lane == 0) { out_v[blockIdx.x] = best.v; out_i[blockIdx.x] = best.i; }
    }
}

// ---------------------------------------------------------------------------
// §4 失败模式：「warp 内交换用 __syncthreads()」
//
// 这是一段**故意写错**的代码，用来说明机制（不放进 main 的调用路径）。
//
// 错法：把块级屏障 `__syncthreads()` 当成 warp 内 lane 之间的同步手段。
//
// 机制（两代硬件、两种失败）：
//   · `__syncthreads()` 是**块级**屏障。PTX `bar.sync` 的记账号径**按 warp 计**
//     （"barriers are executed on a per-warp basis as if all the threads in a
//      warp are active"，到达计数按 warp size 递增，thread-count 操作数必须
//      是 warp size 的整数倍）——**一条 lane 到达即整个 warp 记到**。
//   · 所以它**不能强制已经发散的 lane 收敛**，也**不提供 warp 内 lane 之间
//     的顺序保证**。Volta 之前这会让「尚未到达的 31 条 lane」与已过屏障的
//     lane 之间没有顺序；Volta（CC 7.0）起改为**按线程**强制，同一段代码
//     变成**永不释放**（挂死）。
//   · 正确的做法：warp 内交换走 `__syncwarp()`；跨 warp 的共享内存交换
//     才用 `__syncthreads()`。
//
// 规则：`__syncthreads()` 必须被块内**所有**线程在同一点执行。
//       条件分支里的 `__syncthreads()` 是 UB/死锁，不存在「小心使用」的余地。
//       真机自检：`compute-sanitizer --tool synccheck`。
// ---------------------------------------------------------------------------
__global__ void wrong_warp_sync_demo(const float* __restrict__ x,
                                     float* __restrict__ out, int enable)
{
    __shared__ float s[WARP];
    const int lane = threadIdx.x & (WARP - 1);
    if (threadIdx.x < WARP) {
        s[lane] = x[lane];
        // ✗ 这里需要的是 __syncwarp()；__syncthreads() 既不能强制 warp 内
        //   的收敛，在部分 lane 未到达时也不会按预期释放。
        if (enable) __syncthreads();
        out[lane] = s[(lane + 1) & (WARP - 1)];    // 读到谁写的？没有保证
    }
}

// ---------------------------------------------------------------------------
// host 侧：打印推导值（work / depth / segmented 映射 / argmax 口径）
// ---------------------------------------------------------------------------

static void report_work_depth(int n)
{
    int tree_ops = 0, size = n;
    while (size > 1) { tree_ops += (size + 1) / 2; size = (size + 1) / 2; }
    int depth = 0; size = 1;
    while (size < n) { size *= 2; ++depth; }

    std::printf("§1 两种归约的 work / depth（n = %d，推导值）\n", n);
    std::printf("  %-22s work = %d   depth = %d\n", "顺序（1 线程）", n - 1, n - 1);
    std::printf("  %-22s work = %d   depth = %d\n", "树形", tree_ops, depth);
    std::printf("  → work 相同（都是 n−1）；depth 从 %d 降到 %d，差 %.0f 倍\n\n",
                n - 1, depth, (double)(n - 1) / depth);
    std::printf("  Brent 模型 T = work/p + depth：\n");
    const int ps[] = {1, 32, 1024, 1 << 20};
    for (int k = 0; k < 4; ++k) {
        const int p = ps[k];
        const double t_seq = (double)(n - 1);
        const double t_tree = (double)(n - 1) / p + depth;
        std::printf("    p=%-8d 顺序 T=%.1f（不受 p 影响）；树形 T=%.1f → 加速 %.1fx\n",
                    p, t_seq, t_tree, t_seq / t_tree);
    }
    std::printf("  ⚠ p→∞ 时 T→depth，加速比上界 = work/depth = %.0fx，不随 p 无限增长\n\n",
                (double)(n - 1) / depth);
}

static void report_segmented(const int* lens, int nrows)
{
    std::printf("§1 变体一 segmented 归约的映射（推导值）\n");
    std::printf("  %-8s %-8s | %-34s %-34s\n", "行长 L", "行数",
                "每 warp 一行（32 lane）", "每 block 一行（T=128）");
    for (int r = 0; r < nrows; ++r) {
        const int L = lens[r];
        const int per_w = (L + WARP - 1) / WARP;
        const int per_b = (L + 127) / 128;
        const int use_w = (L < WARP) ? L : WARP;
        std::printf("  %-8d %-8d | 每线程 %-4d lane 利用率 %3.0f%% %-8s | 每线程 %-4d\n",
                    L, 0, per_w, 100.0 * use_w / WARP, "", per_b);
    }
    std::printf("  ⚠ L < 32 时按**最长行**决定迭代数：一次循环要覆盖全部行，\n");
    std::printf("    短行的空转与 Lmin/Lmax 成比例。段边界不对齐 warp 时，\n");
    std::printf("    必须先定段（查表/二分），不能拿 threadIdx/32 当段号。\n\n");
}

int main()
{
    std::printf("============================================================\n");
    std::printf("01_reduce_basics —— 树形归约 / segmented / argmax\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链）。\n");
    std::printf("  下面由公式算出的是**推导值**，可用 `make verify` 的\n");
    std::printf("  §A1 / §A4 / §A5 / §B4 / §B5 / §C3 复核。\n\n");

    report_work_depth(1 << 20);

    const int lens[] = {20, 12, 50, 6};
    report_segmented(lens, 4);

    std::printf("§1 变体二 argmax 的口径（本程序显式声明，推导值）\n");
    std::printf("  NaN 口径  ：**忽略 NaN**（= IEEE 754-2008 maxNum / fmaxf）\n");
    std::printf("              另一口径「传播 NaN」（= IEEE 754-2019 maximum /\n");
    std::printf("              np.maximum）在同一输入上给出**不同的索引**。\n");
    std::printf("  并列口径  ：**取最小索引**（另一口径是取最大索引）。\n");
    std::printf("  ⚠ 只归约值、事后用等值搜索找索引，等于**隐式选定最小索引**，\n");
    std::printf("    与「取最大索引」的声明冲突；值与索引必须配对 shuffle。\n\n");

    // ---- 真机部分（本仓库没有跑过） ----
    const int n = 1 << 20;
    const int nbytes = n * (int)sizeof(float);
    float *dx = nullptr, *dout = nullptr, *dout_rows = nullptr;
    float *darg_v = nullptr; int *darg_i = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, nbytes));
    CUDA_CHECK(cudaMalloc(&dout, 4096 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&darg_v, 4096 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&darg_i, 4096 * sizeof(int)));
    CUDA_CHECK(cudaMemset(dx, 0, nbytes));

    const int nrows = 4;
    int h_start[4], h_len[4];
    h_start[0] = 0;
    for (int r = 0; r < nrows; ++r) {
        h_len[r] = lens[r];
        if (r) h_start[r] = h_start[r - 1] + h_len[r - 1];
    }
    int *d_start = nullptr, *d_len = nullptr;
    CUDA_CHECK(cudaMalloc(&d_start, nrows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_len, nrows * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_start, h_start, nrows * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len, h_len, nrows * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dout_rows, nrows * sizeof(float)));

    const int T = 256;
    seq_reduce_kernel<<<1, 1>>>(dx, n, dout);
    tree_reduce_kernel<<<(n + T - 1) / T, T>>>(dx, n, dout);
    segmented_sum_kernel<<<nrows, 128>>>(dx, d_start, d_len, dout_rows);
    argmax_kernel<<<(n + T - 1) / T, T>>>(dx, n, darg_v, darg_i);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h, dout, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("真机输出（本仓库没有跑过，此处只作占位说明）：\n");
    std::printf("  tree_reduce 的第 0 个 block 局部和 = %.6f\n", h);
    std::printf("  两种形态的加法次数都是 n−1 = %d，逐位结果**可能不同**\n", n - 1);
    std::printf("  （结合顺序不同；参见 04 篇的确定性判据）\n\n");

    std::printf("§4 失败模式（代码见 wrong_warp_sync_demo，本程序不调用它）\n");
    std::printf("  现象：warp 内 lane 间传值「有时对有时错」，换 block 大小/换卡就变\n");
    std::printf("  成因：`__syncthreads()` 是块级屏障，PTX 记账号径**按 warp 计**\n");
    std::printf("        （一条 lane 到达即整个 warp 记到），既不能强制已发散的\n");
    std::printf("        lane 收敛，也不提供 warp 内 lane 之间的顺序保证。\n");
    std::printf("        Volta 前：屏障可能提前释放 → 读到未写完的共享内存；\n");
    std::printf("        Volta 起（CC 7.0）：改为按线程强制 → 部分到达时**永不释放**。\n");
    std::printf("  规避：warp 内交换用 `__syncwarp()`；`__syncthreads()` 必须被\n");
    std::printf("        块内所有线程在同一点执行。自检：compute-sanitizer --tool synccheck\n");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dout_rows));
    CUDA_CHECK(cudaFree(darg_v));
    CUDA_CHECK(cudaFree(darg_i));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_len));
    return 0;
}
