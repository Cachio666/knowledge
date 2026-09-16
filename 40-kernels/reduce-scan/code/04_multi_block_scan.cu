// 04_multi_block_scan.cu —— 跨 block 归约/扫描与**确定性**
//
// 配套文档：40-kernels/reduce-scan/04-multi-block-and-determinism.md
//
// 主线：跨 block 归约有两条路——**atomic 收尾** vs **第二个 kernel**。
// 两者的**部分和字节数完全相同**（都是 2·G·4），差别在
//   · 一次启动的固定开销（引用值 3~5 µs；N 小时它比内核本身还大）；
//   · **确定性**：`atomicAdd` 的到达顺序由硬件决定 → 逐位不可复现。
//
// 确定性的判据（本篇的收束）：
//   · **确定** = 加法的**树形固定**：谁和谁先加写死在代码里，与运行时调度无关。
//   · **不确定** = 加法顺序由**运行时调度**决定：`atomicAdd`、动态领号、
//     按到达顺序触发的信号量归约。
//   · ⚠ 判据看的是**代码里的树形是否固定**，不是「跑两遍结果一不一样」——
//     改顺序**不保证**改变位模式（见下面 §4 的说明与 verify.py §C7）。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 04_multi_block_scan && ./04_multi_block_scan
//
// 对应 `verify.py` 的 §A6 / §B6 / §C7。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// 第一级：每个 block 出一个（或一组）局部和
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce(const float* __restrict__ g,
                                              int n, int tid, int T,
                                              float* __restrict__ smem)
{
    const int lane = tid & (WARP - 1);
    const int wid = tid / WARP;
    const int nwarps = T / WARP;

    float acc = 0.0f;
    for (int i = blockIdx.x * T + tid; i < n; i += gridDim.x * T) acc += g[i];
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) smem[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < nwarps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    return acc;                                   // 只有 tid==0 的值有意义
}

// ---------------------------------------------------------------------------
// 路线 A：atomicAdd 收尾 —— **不确定**
//
// 哪个 block 先把自己的局部和加到 `total` 上，由 block 调度、SM 占用、
// 内存系统的时序共同决定。浮点加法不满足结合律 → 顺序一变结果就变。
// 代价：1 次启动，G 次读改写 = 2·G·4 B。
// ---------------------------------------------------------------------------
__global__ void reduce_atomic(const float* __restrict__ g, int n,
                              float* __restrict__ total)
{
    __shared__ float s[WARP];
    const float acc = block_reduce(g, n, threadIdx.x, blockDim.x, s);
    if (threadIdx.x == 0) atomicAdd(total, acc);  // ← 到达顺序由硬件决定
}

// ---------------------------------------------------------------------------
// 路线 B：workspace + 第二个 kernel —— **确定**
//
// 第一级把每个 block 的局部和按 `blockIdx.x` **写进固定下标**；
// 第二级由一个 block 按 `b = 0,1,2,…` 的固定顺序累加。
// 代价：2 次启动；字节数与路线 A **完全相同**（G 次写 + G 次读 = 2·G·4）。
// ---------------------------------------------------------------------------
__global__ void reduce_phase1(const float* __restrict__ g, int n,
                              float* __restrict__ partials)
{
    __shared__ float s[WARP];
    const float acc = block_reduce(g, n, threadIdx.x, blockDim.x, s);
    if (threadIdx.x == 0) partials[blockIdx.x] = acc;   // ← 下标固定
}

__global__ void reduce_phase2(const float* __restrict__ partials, int nblocks,
                              float* __restrict__ total)
{
    // 单 block、单线程，按固定下标顺序累加 —— 树形写死在代码里
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) acc += partials[b];   // ← 顺序固定
    total[0] = acc;
}

// ---------------------------------------------------------------------------
// 三段式跨 block **扫描**（确定，且不需要全局同步原语）
//
//   1. 每块对本块 tile 做块内 inclusive scan，写出结果，并把**块总和**
//      写进 `block_sums[blockIdx.x]`（下标固定）；
//   2. 对 `block_sums` 做 **exclusive** scan，得到每块的偏移 `block_off[b]`；
//   3. 每块把自己的 tile 结果加上 `block_off[b]`。
//
// ⚠ 第 2 步必须是 **exclusive**：用 inclusive 会把本块总和也算进去，
//   结果整体偏一个块——**首元素立刻不等于 x[0]**（这类错自检抓得到）；
//   真正难抓的是第 3 步漏做（首块正确、后面全错得「成比例」）。
// ---------------------------------------------------------------------------
__global__ void scan_phase1(const float* __restrict__ g, float* __restrict__ out,
                            float* __restrict__ block_sums, int n, int tile)
{
    __shared__ float warp_total[WARP];
    __shared__ float warp_off[WARP];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid = tid / WARP;
    const int nwarps = blockDim.x / WARP;
    const int base = blockIdx.x * tile;
    const int idx = base + tid;

    float v = (idx < n) ? g[idx] : 0.0f;   // 越界读 0：加法的幺元

    // 第 1 级：warp 内 inclusive scan（up shuffle + 守卫）
    #pragma unroll
    for (int off = 1; off < WARP; off <<= 1) {
        const float t = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += t;
    }
    if (lane == WARP - 1) warp_total[wid] = v;
    __syncthreads();

    // 第 2 级：warp 0 扫描各 warp 的总和，给出每 warp 的 exclusive 偏移
    if (wid == 0) {
        float w = (lane < nwarps) ? warp_total[lane] : 0.0f;
        #pragma unroll
        for (int off = 1; off < WARP; off <<= 1) {
            const float t = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += t;
        }
        const float prev = __shfl_up_sync(0xffffffffu, w, 1);
        warp_off[lane] = (lane == 0) ? 0.0f : prev;    // exclusive：不含自己
        // 本块总和 = 最后一个有效 warp 的 inclusive 值 → 写进**固定下标**
        if (lane == nwarps - 1) block_sums[blockIdx.x] = w;
    }
    __syncthreads();

    // 第 3 级：本 warp 的 inclusive 结果 + 块内偏移 = 块内 inclusive
    if (idx < n) out[idx] = v + warp_off[wid];
}

__global__ void scan_phase2(const float* __restrict__ block_sums,
                            float* __restrict__ block_off, int nblocks)
{
    // 单 block 单线程对块总和做 exclusive scan。形态最直白但**树形写死**
    // → 确定。块数远小于 n，顺序扫描的开销可忽略；块数很大时改用
    //   单块内的 warp shuffle 扫描（形态同 scan_phase1 的第 1/2 级）。
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        block_off[b] = acc;              // ← 先记偏移（不含自己）→ exclusive
        acc += block_sums[b];
    }
}

// 第 3 阶段：把块偏移加回 phase1 的结果。
// 这里选择「重算一遍块内扫描」而不是「把 phase1 的中间结果写进 workspace」——
// 前者多花一次块内扫描，后者多一次 n 个元素的往返访存，取舍与 02 篇的
// 「warp 局部和」是同一类问题。
__global__ void scan_phase3(float* __restrict__ out, const float* __restrict__ g,
                            const float* __restrict__ block_off, int n, int tile)
{
    __shared__ float warp_total[WARP];
    __shared__ float warp_off[WARP];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid = tid / WARP;
    const int nwarps = blockDim.x / WARP;
    const int idx = blockIdx.x * tile + tid;

    float v = (idx < n) ? g[idx] : 0.0f;
    #pragma unroll
    for (int off = 1; off < WARP; off <<= 1) {
        const float t = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += t;
    }
    if (lane == WARP - 1) warp_total[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float w = (lane < nwarps) ? warp_total[lane] : 0.0f;
        #pragma unroll
        for (int off = 1; off < WARP; off <<= 1) {
            const float t = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += t;
        }
        const float prev = __shfl_up_sync(0xffffffffu, w, 1);
        warp_off[lane] = (lane == 0) ? 0.0f : prev;
    }
    __syncthreads();
    if (idx < n) out[idx] = v + warp_off[wid] + block_off[blockIdx.x];
}

// ---------------------------------------------------------------------------
// host 侧：两条路线的字节/开销对照，以及确定性的判据
// ---------------------------------------------------------------------------

static void report_paths(void)
{
    std::printf("§1 两条路线的部分和字节数（G = block 数，推导值）\n");
    for (int G = 128; G <= 1024; G *= 2) {
        const int atomic_b = 2 * G * 4;      // G 次读改写
        const int two_b = G * 4 + G * 4;     // G 次写 + G 次读
        std::printf("  G=%-6d atomic 路线 %-8d B | 两级路线 %-8d B | %s\n",
                    G, atomic_b, two_b,
                    (atomic_b == two_b) ? "完全相同" : "不同");
    }
    std::printf("  → 差别不在流量，在**确定性**与**一次启动**（与 gemm/06 §2.1 同构）。\n\n");

    std::printf("§1 固定开销 vs 规模（HBM 3350 GB/s；启动开销是**引用值** 3~5 µs）\n");
    const int ks[] = {18, 20, 24};
    for (int t = 0; t < 3; ++t) {
        const long long N = 1LL << ks[t];
        const double by = (double)N * 4.0;
        const double us = by / 3350e9 * 1e6;
        std::printf("  N=2^%-3d 内核 %8.2f µs | 多一次启动 → 相对开销 %.0f%%~%.0f%%\n",
                    ks[t], us, 100.0 * 3.0 / us, 100.0 * 5.0 / us);
    }
    std::printf("  → N 小时**一次启动就比内核本身贵**；N 大时才摊薄（2^24 降到 15%%~25%%）。\n\n");
}

int main()
{
    std::printf("============================================================\n");
    std::printf("04_multi_block_scan —— 跨 block 归约/扫描与确定性\n");
    std::printf("============================================================\n");
    std::printf("⚠ 本程序本仓库**没有运行过**（写作环境无 CUDA 工具链）。\n");
    std::printf("  打印的是推导值，可用 `make verify` 的 §A6 / §B6 / §C7 复核。\n\n");

    report_paths();

    std::printf("§2 确定性的判据（本篇的收束）\n");
    std::printf("  · **确定** = 加法的**树形固定**：谁和谁先加写死在代码里，\n");
    std::printf("    与运行时的调度时序无关 → 同输入必得同 bit。\n");
    std::printf("    例：workspace + 固定下标归约（本文件路线 B）、\n");
    std::printf("        三段式扫描（本文件 scan_phase1/2/3，**完全确定**）。\n");
    std::printf("  · **不确定** = 加法顺序由**运行时调度**决定。\n");
    std::printf("    例：`atomicAdd`（路线 A）、动态领号/工作窃取、\n");
    std::printf("        按到达顺序触发的信号量归约。\n");
    std::printf("  · 与「精度」无关：各顺序的相对误差都在 1e-7 量级，位模式却可能不同。\n");
    std::printf("  · ⚠ **判据看代码，不看观察结果**：改顺序**不保证**改变位模式\n");
    std::printf("    （verify.py §C7：132 段量级相当时正序与逆序恰好相同），\n");
    std::printf("    所以「跑两遍一样」既不能证明确定、也不能证明不确定。\n\n");

    std::printf("§4 失败模式清单\n");
    std::printf("  · `atomicAdd(float)` 做跨 block 归约 → 每次运行结果可能不同；\n");
    std::printf("    测试里写 `allclose` 抓不到这条，必须**逐位**比较两次运行。\n");
    std::printf("  · grid 的遍历顺序（先 x 后 y、还是先 y 后 x）改变累加顺序；\n");
    std::printf("    把第二级塞进 gridDim.y 时尤其容易忘——且 y/z 维上限只有 65535\n");
    std::printf("    （x 维是 2^31−1，差 5 个数量级）。\n");
    std::printf("  · 三段式扫描的第 2 步用 inclusive 而非 exclusive → 整体偏一个块。\n\n");

    // ---- 真机部分（本仓库没有跑过） ----
    const int n = 1 << 20, T = 256;
    const int blocks = (n + T - 1) / T;
    float *dg = nullptr, *dtot = nullptr, *dpart = nullptr, *dout = nullptr;
    float *dbsum = nullptr, *dboff = nullptr;
    CUDA_CHECK(cudaMalloc(&dg, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dtot, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dpart, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbsum, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dboff, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(dg, 1, n * sizeof(float)));

    // 路线 A：跑两遍，逐位比较
    float a1 = 0.0f, a2 = 0.0f;
    CUDA_CHECK(cudaMemset(dtot, 0, sizeof(float)));
    reduce_atomic<<<blocks, T>>>(dg, n, dtot);
    CUDA_CHECK(cudaMemcpy(&a1, dtot, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dtot, 0, sizeof(float)));
    reduce_atomic<<<blocks, T>>>(dg, n, dtot);
    CUDA_CHECK(cudaMemcpy(&a2, dtot, sizeof(float), cudaMemcpyDeviceToHost));

    // 路线 B：固定下标 + 第二个 kernel
    reduce_phase1<<<blocks, T>>>(dg, n, dpart);
    reduce_phase2<<<1, 1>>>(dpart, blocks, dtot);
    float b1 = 0.0f;
    CUDA_CHECK(cudaMemcpy(&b1, dtot, sizeof(float), cudaMemcpyDeviceToHost));

    // 三段式扫描
    const int tile = 1024;
    const int sblocks = (n + tile - 1) / tile;
    scan_phase1<<<sblocks, tile>>>(dg, dout, dbsum, n, tile);
    scan_phase2<<<1, 1>>>(dbsum, dboff, sblocks);
    scan_phase3<<<sblocks, tile>>>(dout, dg, dboff, n, tile);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("真机输出（本仓库没有跑过，此处只作占位说明）：\n");
    std::printf("  atomic 路线两次运行：%.6f / %.6f —— 位模式相同？%s\n", a1, a2,
                (std::memcmp(&a1, &a2, sizeof(float)) == 0) ? "是" : "否");
    std::printf("    ⚠ 「是」**不能**证明它确定；判据是「顺序写死没有」，而不是一次观察。\n");
    std::printf("  两级路线：%.6f（给定同一个 G，逐位可复现）\n", b1);
    std::printf("  三段式扫描：给定 tile 数即逐位可复现（划分固定 → 树形固定）\n");

    CUDA_CHECK(cudaFree(dg));
    CUDA_CHECK(cudaFree(dtot));
    CUDA_CHECK(cudaFree(dpart));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(dbsum));
    CUDA_CHECK(cudaFree(dboff));
    return 0;
}
