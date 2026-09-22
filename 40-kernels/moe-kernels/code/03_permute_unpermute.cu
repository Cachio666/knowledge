// 03_permute_unpermute.cu —— 路由、重排（permute）与回收（unpermute）
//
// 配套文档：40-kernels/moe-kernels/03-permute-and-unpermute.md
//
// 主线：permute 是**行级 scatter**，unpermute 是**行级 gather + 加权求和**。
// 「一行」是 d_model 个**连续**元素，所以散开的是**行号**、不是元素——
// 合并与否只取决于「warp 内的 lane 沿哪个方向走」：
//
//   甲 `permute_rows_kernel`      lane 沿 d_model 走 → 一个 warp 的访存落在
//                                32×esize 个连续字节上（FP16 是 2 个 32B sector）
//   乙 `permute_scatter_kernel`   lane 沿 slot 走   → 32 个 lane 各占一个 sector，
//                                搬走的字节里有用的只有 6.25%
//
// 以及 unpermute 的**两条路**（同一个模板的两个实例）：
//   丙 `unpermute_kernel<K,1>` —— 做 mask：被截断的 slot 不参与累加
//   丁 `unpermute_kernel<K,0>` —— 不做 mask：把被丢 slot 里的**残留数据**也加了进去
// 两条路跑在同一个缓冲区上，差别就是「对」与「悄悄错」。
//
// ⚠ `dst_row` 与 `valid` 是**两个**数组：被丢的 slot 仍然有它的位置
//   （`offsets[e] + rank`），只是标记为无效。这样才能表达「缓冲区里还有东西」。
//
// 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，**未编译过**）：
//     make 03_permute_unpermute && ./03_permute_unpermute
//
// ⚠ §3 的字节数 / sector 数 / 占比全部是**推导值**（纯算术，
//    与 `verify.py` 的 §B6 / §C3 同式）；§4 的误差是 kernel 的运行结果，
//    本仓库没有 GPU，**没有跑过**。
//    可复核处：`make verify` 的 §A2 / §A4 / §B6 / §C2 / §C3 / §C6。

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

// ===========================================================================
// §1 路由侧：直方图 → 前缀和 → slot 映射
// ===========================================================================

// 1) 直方图：统计 tokens_per_expert。
//    **计数结果确定，但「哪个 SM 先到」不确定**——所以 slot 的具体分配顺序
//    在不同运行之间可以不同（正文 §4.4 的失败模式）。
__global__ void histogram_kernel(const int* __restrict__ idx, int n_slot, int E,
                                 int* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_slot) atomicAdd(&counts[idx[i]], 1);
}

// 2) 排他前缀和：offsets[e] = 前 e 个专家一共占了多少 slot。
//    并行扫描的正文在 40-kernels/reduce-scan/03，这里是单 block 的 Hillis-Steele。
__global__ void scan_offsets_kernel(const int* __restrict__ counts, int E,
                                    int* __restrict__ offsets) {
    __shared__ int tmp[256];
    const int t = threadIdx.x;
    if (t < E) tmp[t] = counts[t];
    __syncthreads();
    for (int off = 1; off < E; off <<= 1) {
        int v = 0;
        if (t < E && t >= off) v = tmp[t - off];
        __syncthreads();
        if (t < E) tmp[t] += v;
        __syncthreads();
    }
    if (t < E) offsets[t] = tmp[t] - counts[t];      // 排他 = 包含 − 自己
}

// 3) 建 slot 映射：每个 (token, 专家) 对一个 slot。
//    `cursor` 是 offsets 的副本，会被原地累加（**顺序由 atomicAdd 决定**）。
//    排位 ≥ C 的 slot 记为**无效**——但它仍然有自己的位置，缓冲区里也仍然有数据。
__global__ void build_slot_map_kernel(const int* __restrict__ idx, int T, int K,
                                      const int* __restrict__ offsets, int capacity,
                                      int* __restrict__ cursor,
                                      int* __restrict__ dst_row,          // [T*K]
                                      unsigned char* __restrict__ valid,  // [T*K]
                                      int* __restrict__ src_of,           // [T*K] 按 slot
                                      unsigned char* __restrict__ slot_valid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * K) return;
    const int t = i / K;
    const int e = idx[i];
    const int s = atomicAdd(&cursor[e], 1);
    const bool ok = (s - offsets[e]) < capacity;
    dst_row[i]  = s;                 // **永远记下真实位置**，无效也记
    valid[i]    = ok ? 1 : 0;
    slot_valid[s] = ok ? 1 : 0;
    if (ok) src_of[s] = t;
}

// ===========================================================================
// §2 permute：两条访存路径
// ===========================================================================

// 甲（合并）：一个 block 搬一行，**lane 沿 d_model 走**。
__global__ void permute_rows_kernel(const float* __restrict__ X, int d, int n_slot,
                                    const int* __restrict__ src_of,
                                    const unsigned char* __restrict__ slot_valid,
                                    float* __restrict__ Xp) {
    const int s = blockIdx.x;
    if (s >= n_slot || !slot_valid[s]) return;       // 无效 slot 不写，缓冲区保留旧数据
    const float* src = X + (size_t)src_of[s] * d;
    float* dst = Xp + (size_t)s * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) dst[c] = src[c];
}

// 乙（反例）：一个线程搬一行，**lane 沿 slot 走**。
// 每次循环迭代，同一个 warp 的 32 个 lane 写向 32 个相距 d 个元素的位置。
__global__ void permute_scatter_kernel(const float* __restrict__ X, int d, int T, int K,
                                       const int* __restrict__ dst_row,
                                       const unsigned char* __restrict__ valid,
                                       float* __restrict__ Xp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // i = t*K + j
    if (i >= T * K || !valid[i]) return;
    const int t = i / K;
    const float* src = X + (size_t)t * d;
    float* dst = Xp + (size_t)dst_row[i] * d;
    for (int c = 0; c < d; ++c) dst[c] = src[c];      // 每次迭代：32 个 lane 各写一行
}

// ===========================================================================
// §3 unpermute：k 个贡献在寄存器里累加，每个 token 只写一行
// ===========================================================================

template <int K, int MASK>
__global__ void unpermute_kernel(const float* __restrict__ Yp, int d, int T,
                                 const int* __restrict__ dst_row,
                                 const unsigned char* __restrict__ valid,
                                 const float* __restrict__ w,
                                 float* __restrict__ out) {
    const int t = blockIdx.x;
    if (t >= T) return;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const int i = t * K + j;
            if (MASK && !valid[i]) continue;          // ← 这一行就是「做不做 mask」
            acc += w[i] * Yp[(size_t)dst_row[i] * d + c];
        }
        out[(size_t)t * d + c] = acc;
    }
}

// ===========================================================================
// §4 访存字节与合并度（推导值）
// ===========================================================================

static void report_traffic(int T, int K, int d, double esize, double weight_gb) {
    const double row = d * esize;
    const double rows = (double)T * K;
    std::printf("--- §4 permute / unpermute 的访存字节与合并度（推导值）---\n");
    std::printf("场景：tokens=%d top_k=%d d_model=%d esize=%.0fB → 一行 = %.0f B；"
                "输入 %d 行，permute 后 %.0f 行\n", T, K, d, esize, row, T, rows);
    std::printf("  permute   读一次写 k 次 = T·d·esize + T·k·d·esize = %.3f MB\n",
                (T * row + rows * row) / 1e6);
    std::printf("  permute   读 k 次写 k 次 = 2·T·k·d·esize            = %.3f MB（多 %.0f%%）\n",
                (2 * rows * row) / 1e6, 100.0 * (2 * rows * row) / (T * row + rows * row) - 100.0);
    std::printf("  unpermute 读 k 次写一次 = T·k·d·esize + T·d·esize = %.3f MB\n",
                (rows * row + T * row) / 1e6);
    std::printf("  两者合计 = 2·(1+k)·T·d·esize = %.3f MB"
                "（同层专家权重 %.4f GB 的 %.2f%%）\n\n",
                2 * (1 + K) * T * row / 1e6, weight_gb,
                100.0 * (2.0 * (1 + K) * T * row) / 1e9 / weight_gb);

    const int sect = 32;
    const int good = (int)((WARP * esize + sect - 1) / sect);
    std::printf("  一个 warp（32 lane）的访存触及几个 %d B sector：\n", sect);
    std::printf("    甲 lane 沿 d_model 走：32×%.0f B = %.0f B 连续 → %d 个 sector"
                "（有用字节占比 100%%）\n", esize, WARP * esize, good);
    std::printf("    乙 lane 沿 slot 走  ：32 个地址相距 %.0f B → %d 个 sector"
                "（有用字节占比 %.2f%%）\n", row, WARP, 100.0 * (WARP * esize) / (WARP * sect));
    std::printf("    放大 = %d / %d = %d×\n\n", WARP, good, WARP / good);
    std::printf("  ⚠ 这两条路径的**输出内容完全一样**，正确性测不出来，\n");
    std::printf("    只有访存计数器（sector 数）能区分。\n\n");
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    // 演示规模（小，便于和 fp64 参考逐元素比）
    const int T = 64, d = 128, K = 2, E = 4;
    const int S = T * K;                     // slot 总数
    const int C = 40;                        // 每个专家开出的槽位数

    std::printf("=== 03_permute_unpermute ===\n");
    std::printf("演示规模：tokens=%d d_model=%d top_k=%d 专家=%d，"
                "每个专家 %d 个 slot（开出的槽位 E×C = %d，要放的 slot 共 %d）\n\n",
                T, d, K, E, C, E * C, S);

    double weight_gb = 8.0 * 2.0 * 4096.0 * 14336.0 * 2.0 / 1e9;

    // ---- 主机侧构造：倾斜路由（45/64 的 token 第一选择是专家 0）----
    int* hIdx = (int*)std::malloc(S * sizeof(int));
    float* hW = (float*)std::malloc(S * sizeof(float));
    {
        unsigned s = 12345u;
        auto rnd = [&s]() { s = s * 1103515245u + 12345u; return ((s >> 8) & 0xFFFF) / 65536.0; };
        for (int t = 0; t < T; ++t) {
            const int first = (t < 45) ? 0 : (int)(rnd() * E) % E;
            int second = (int)(rnd() * E) % E;
            if (second == first) second = (second + 1) % E;
            hIdx[t * K + 0] = first;
            hIdx[t * K + 1] = second;
            const float a = (float)rnd() + 0.5f;
            const float b = (float)rnd() + 0.5f;
            const float m = fmaxf(a, b);
            const float ea = expf(a - m), eb = expf(b - m);
            hW[t * K + 0] = ea / (ea + eb);       // 在 k 个 logit 上 softmax → 和为 1
            hW[t * K + 1] = eb / (ea + eb);
        }
    }

    // ---- 设备侧：直方图 → 前缀和 → slot 映射 ----
    int *dIdx, *dCounts, *dOffsets, *dCursor, *dDstRow, *dSrcOf;
    unsigned char *dValid, *dSlotValid;
    CUDA_CHECK(cudaMalloc(&dIdx, S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCounts, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOffsets, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCursor, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dDstRow, S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSrcOf, S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dValid, S * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dSlotValid, S * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(dIdx, hIdx, S * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dCounts, 0, E * sizeof(int)));
    CUDA_CHECK(cudaMemset(dSlotValid, 0, S * sizeof(unsigned char)));

    const int tb = 128, nb = (S + tb - 1) / tb;
    histogram_kernel<<<nb, tb>>>(dIdx, S, E, dCounts);
    scan_offsets_kernel<<<1, 256>>>(dCounts, E, dOffsets);
    CUDA_CHECK(cudaMemcpy(dCursor, dOffsets, E * sizeof(int), cudaMemcpyDeviceToDevice));
    {
        int hCnt[256], hOff[256];
        CUDA_CHECK(cudaMemcpy(hCnt, dCounts, E * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hOff, dOffsets, E * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("tokens_per_expert = [");
        for (int e = 0; e < E; ++e) std::printf("%s%d", e ? "," : "", hCnt[e]);
        std::printf("]   offsets（排他前缀和）= [");
        for (int e = 0; e < E; ++e) std::printf("%s%d", e ? "," : "", hOff[e]);
        std::printf("]\n");
        int drop = 0;
        for (int e = 0; e < E; ++e) drop += (hCnt[e] > C) ? (hCnt[e] - C) : 0;
        std::printf("被截断的 slot = Σ max(0, M_e − C) = %d / %d = %.1f%%"
                    "（只有 %d 个专家溢出）\n\n", drop, S, 100.0 * drop / S,
                    [&] { int n = 0; for (int e = 0; e < E; ++e) n += (hCnt[e] > C); return n; }());
    }
    build_slot_map_kernel<<<nb, tb>>>(dIdx, T, K, dOffsets, C, dCursor,
                                      dDstRow, dValid, dSrcOf, dSlotValid);

    // ---- 缓冲区：预填「上一次迭代的残留」（这里用另一组随机数代替）----
    float *dX, *dXp, *dOutMasked, *dOutNoMask, *dW, *dXprev;
    CUDA_CHECK(cudaMalloc(&dX, (size_t)T * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXprev, (size_t)S * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXp, (size_t)S * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutMasked, (size_t)T * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutNoMask, (size_t)T * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW, S * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW, hW, S * sizeof(float), cudaMemcpyHostToDevice));

    float* hX = (float*)std::malloc((size_t)T * d * sizeof(float));
    float* hXp = (float*)std::malloc((size_t)S * d * sizeof(float));
    {
        unsigned s = 999u;
        for (size_t i = 0; i < (size_t)T * d; ++i) {
            s = s * 1103515245u + 12345u;
            hX[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        for (size_t i = 0; i < (size_t)S * d; ++i) {
            s = s * 1103515245u + 12345u;
            hXp[i] = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;   // 残留数据
        }
    }
    CUDA_CHECK(cudaMemcpy(dX, hX, (size_t)T * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXprev, hXp, (size_t)S * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXp, dXprev, (size_t)S * d * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    permute_rows_kernel<<<S, 128>>>(dX, d, S, dSrcOf, dSlotValid, dXp);
    unpermute_kernel<K, 1><<<T, 128>>>(dXp, d, T, dDstRow, dValid, dW, dOutMasked);
    unpermute_kernel<K, 0><<<T, 128>>>(dXp, d, T, dDstRow, dValid, dW, dOutNoMask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- §5 结果：masked vs unmasked ----
    {
        int* hDst = (int*)std::malloc(S * sizeof(int));
        unsigned char* hValid = (unsigned char*)std::malloc(S * sizeof(unsigned char));
        float* hA = (float*)std::malloc((size_t)T * d * sizeof(float));
        float* hB = (float*)std::malloc((size_t)T * d * sizeof(float));
        CUDA_CHECK(cudaMemcpy(hDst, dDstRow, S * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hValid, dValid, S * sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hA, dOutMasked, (size_t)T * d * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hB, dOutNoMask, (size_t)T * d * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double eA = 0.0, eB = 0.0, scale = 0.0;
        for (int t = 0; t < T; ++t) {
            float wsum = 0.0f;
            for (int j = 0; j < K; ++j)
                if (hValid[t * K + j]) wsum += hW[t * K + j];
            for (int c = 0; c < d; ++c) {
                // 做 mask 时拿回的是 **X[t] × 幸存权重之和**（不是 X[t] 本身）
                const double ref = (double)wsum * hX[(size_t)t * d + c];
                scale = std::fmax(scale, std::fabs(ref));
                eA = std::fmax(eA, std::fabs(ref - hA[(size_t)t * d + c]));
                // 不做 mask 时多出来的一项是「被丢 slot 里的残留」加权
                double refB = ref;
                for (int j = 0; j < K; ++j) {
                    if (!hValid[t * K + j])
                        refB += (double)hW[t * K + j] *
                                hXp[(size_t)hDst[t * K + j] * d + c];
                }
                eB = std::fmax(eB, std::fabs(refB - hB[(size_t)t * d + c]));
            }
        }
        int bad = 0;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < K; ++j) if (!hValid[t * K + j]) { ++bad; break; }

        std::printf("--- §5 unpermute：做 mask vs 不做 mask（kernel 运行结果，本仓库没跑过）---\n");
        std::printf("做 mask  ：max|Δ| vs 参考 = %.3e（参考量级 %.3e）\n", eA, scale);
        std::printf("不做 mask：max|Δ| vs 参考 = %.3e\n", eB);
        std::printf("↑ 前者的参考是「X[t] × 幸存权重之和」——**做 mask 之后拿回的\n");
        std::printf("  并不是 X[t] 本身**，因为被丢的那几个专家的权重没了。\n");
        std::printf("  要不要把它重新归一化是架构层的选择（见 80-model-architectures/moe/02-capacity-factor.md）；\n");
        std::printf("  **kernel 侧的责任是让这个选择成为可能**——先得有 mask。\n");
        std::printf("  受影响的 token：%d / %d 行整行被污染。\n\n", bad, T);
        std::free(hDst); std::free(hValid); std::free(hA); std::free(hB);
    }

    // ---- 反例 kernel（列级 scatter）：只跑一次，证明它能编、能跑 ----
    permute_scatter_kernel<<<nb, tb>>>(dX, d, T, K, dDstRow, dValid, dXp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("--- 反例 kernel（lane 沿 slot 走的 permute）已执行一次 ---\n");
    std::printf("它写出的内容与合并版**逐元素相同**——正确性测不出来，\n");
    std::printf("只有访存计数器能看出差别（下一节的 sector 数）。\n\n");

    report_traffic(4096, 2, 4096, 2.0, weight_gb);

    CUDA_CHECK(cudaFree(dIdx)); CUDA_CHECK(cudaFree(dCounts)); CUDA_CHECK(cudaFree(dOffsets));
    CUDA_CHECK(cudaFree(dCursor)); CUDA_CHECK(cudaFree(dDstRow)); CUDA_CHECK(cudaFree(dSrcOf));
    CUDA_CHECK(cudaFree(dValid)); CUDA_CHECK(cudaFree(dSlotValid));
    CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dXprev)); CUDA_CHECK(cudaFree(dXp));
    CUDA_CHECK(cudaFree(dOutMasked)); CUDA_CHECK(cudaFree(dOutNoMask)); CUDA_CHECK(cudaFree(dW));
    std::free(hIdx); std::free(hW); std::free(hX); std::free(hXp);
    return 0;
}
