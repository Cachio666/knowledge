// 05_decode_attention.cu —— decode 阶段：GQA/MQA、split-KV、paged KV
//
// 配套文档：40-kernels/attention/05-decoding-attention.md
//
// decode 的每 token 只需要算**一行** query，于是形态完全变了：
//
//   ① **memory-bound 到极端。** 每生成一个 token：
//        FLOPs = 4·S·d·heads_q，字节 = 2·S·d·heads_kv·esize
//        ⟹ AI = 2·r/esize = r（fp16），r = heads_q/heads_kv（GQA 的群组大小）
//      **S 与 d 都被约掉了**。MHA（r=1）的 AI = 1.0，只有 I*(FP16)=295.4 的
//      0.34%；访存时间是算力时间的 295 倍（verify.py §B9，推导值）。
//
//   ② **GQA/MQA 直接按比例削流量**：KV 字节 ∝ 1/r，
//      MQA（heads_kv=1）是 1/32（verify.py §B8）。减少的是**读取**字节，
//      不是 FLOPs——分数照样要对每个 KV 位置算一遍。
//
//   ③ **并行度不够**：grid = batch × heads_q。batch=1、heads_q=32 时只有 32 个
//      block，而 H100 有 132 个 SM（verify.py §C7）。于是要 split-KV：
//      把 KV 切成 n_split 段，各自算一个 (m, l, O)，最后用同一个 ⊕ 合并
//      ——**O 也必须跟着重标定**（§A5 的合并算子）。
//
//   ④ **paged KV 的代价不在字节**：页表本身可以忽略（0.0015%）、页内碎片也
//      可以忽略（0.049%）。真正的代价是**间接寻址的延迟**：一个 Bc=64 的 tile
//      跨 4 页 ⟹ 4 次有依赖的页表查询，而一次片外往返是 400~800 周期，
//      与这个 tile 的计算时间（≈504 周期）同量级（verify.py §B10）——
//      所以页表**必须预取**，不能现取现用。
//      sector 口径回链 20-gpu-architecture/memory-system/02：页内连续 ⟹ 合并无碍。
//
// ⚠ 编译与运行（需 nvcc + GPU；本仓库的写作环境没有工具链，未编译过）：
//     make 05_decode_attention
//     ./05_decode_attention                 # S=1024, d=128, hq=32, hkv=8, n_split=8
//
//   数字分两类：**推导值**（AI、字节、并行度，可用 `make verify` §B8/§B9/§B10/§C7
//   复核）与**本机运行结果**（逐元素差，本仓库没有跑过）。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

constexpr int MAX_D = 128;
constexpr int WARP = 32;

// ---------------------------------------------------------------------------
// decode kernel（**无 split-KV 的版本**）：一个 block 处理 (序列, 一个 query 头)
//
//   heads_kv 个 KV 头被 heads_q 个 query 头共用 ⟹ 同一份 K/V 被读 rep 次
//   ——这就是 GQA/MQA 要解决的东西，但它**不改变任何一条算式**。
//
//   paged KV 用页表间接寻址：K/V 的物理位置由 block_table 给出，
//   一页放 page_tokens 行。
// ---------------------------------------------------------------------------
__global__ void decode_attention_kernel(const half* __restrict__ Q,      // [hq][d]
                                        const half* __restrict__ Kcache, // 物理页池
                                        const half* __restrict__ Vcache,
                                        const int* __restrict__ block_table,  // [n_pages]
                                        half* __restrict__ O,            // [hq][d]
                                        int hq, int hkv, int rep, int seq, int d,
                                        float scale, int page_tokens)
{
    const int h = blockIdx.x;                     // query 头
    const int kv_head = h / rep;                  // 它用哪个 KV 头
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const half* q = Q + (size_t)h * d;

    float m = -INFINITY, l = 0.0f;
    float acc[MAX_D];
    for (int c = 0; c < d; ++c) acc[c] = 0.0f;

    __shared__ float red[WARP];
    __shared__ float sm_s[256];               // 本页的 logits（page_tokens ≤ 256）
    __shared__ float sP[256];                 // 本页的 exp(s − m_new)

    // 逐页推进：每页 page_tokens 行，**页内连续**（所以合并访存没问题）
    const int n_pages = (seq + page_tokens - 1) / page_tokens;
    for (int p = 0; p < n_pages; ++p) {
        const int phys = block_table[p];            // ← 间接寻址：一次有依赖的取数
        const int row0 = p * page_tokens;
        const int nc = min(page_tokens, seq - row0);
        const half* Kp = Kcache + ((size_t)phys * hkv + kv_head) * page_tokens * d;
        const half* Vp = Vcache + ((size_t)phys * hkv + kv_head) * page_tokens * d;

        // 每行 query 的 logits 与 max（一个 block 只有一行 query，全 block 协作）
        float lm = -INFINITY;
        for (int j = tid; j < nc; j += nthreads) {
            float s = 0.0f;
            for (int c = 0; c < d; ++c)
                s += __half2float(q[c]) * __half2float(Kp[(size_t)j * d + c]);
            s *= scale;
            sm_s[j] = s;
            lm = fmaxf(lm, s);
        }
        __syncthreads();
        for (int off = WARP / 2; off > 0; off >>= 1)
            lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, off));
        if ((tid & (WARP - 1)) == 0) red[tid / WARP] = lm;
        __syncthreads();
        if (tid < WARP) {
            lm = (tid < (int)(nthreads / WARP)) ? red[tid] : -INFINITY;
            for (int off = WARP / 2; off > 0; off >>= 1)
                lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, off));
            if (tid == 0) red[0] = lm;
        }
        __syncthreads();
        lm = red[0];

        // **同一条重标定式子**：α 同时乘到 l 与 O 上（本文 §1 的递推，与 02 篇同源）
        const float m_new = fmaxf(m, lm);
        const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float lsum = 0.0f;
        for (int j = tid; j < nc; j += nthreads) lsum += expf(sm_s[j] - m_new);
        for (int off = WARP / 2; off > 0; off >>= 1)
            lsum += __shfl_down_sync(0xffffffffu, lsum, off);
        if ((tid & (WARP - 1)) == 0) red[tid / WARP] = lsum;
        __syncthreads();
        if (tid < WARP) {
            lsum = (tid < (int)(nthreads / WARP)) ? red[tid] : 0.0f;
            for (int off = WARP / 2; off > 0; off >>= 1)
                lsum += __shfl_down_sync(0xffffffffu, lsum, off);
            if (tid == 0) red[0] = lsum;
        }
        __syncthreads();
        l = l * alpha + red[0];
        for (int c = 0; c < d; ++c) acc[c] *= alpha;           // ← O 跟着同一个 α
        for (int j = tid; j < nc; j += nthreads) sP[j] = expf(sm_s[j] - m_new);
        __syncthreads();
        for (int c = tid; c < d; c += nthreads) {              // 每线程负责若干列
            float t = 0.0f;
            for (int j = 0; j < nc; ++j) t += sP[j] * __half2float(Vp[(size_t)j * d + c]);
            acc[c] += t;
        }
        m = m_new;
        __syncthreads();
    }
    for (int c = tid; c < d; c += nthreads)
        O[(size_t)h * d + c] = __float2half(acc[c] / l);
}

// ---------------------------------------------------------------------------
// split-KV 的合并：每段独立算出 (m_i, l_i, O_i)，按下式合成
//     M = max_i m_i ;  l = Σ l_i·e^{m_i−M} ;  O = Σ O_i·e^{m_i−M}
//   —— 与 §A5、§1 的 ⊕ 是同一条式子，**O 必须跟着重标定**
// ---------------------------------------------------------------------------
static void merge_splits_cpu(const std::vector<double>& ms, const std::vector<double>& ls,
                             const std::vector<std::vector<double>>& Os, int d,
                             std::vector<double>& O)
{
    const double M = *std::max_element(ms.begin(), ms.end());
    double l = 0.0;
    std::vector<double> acc(d, 0.0);
    for (size_t i = 0; i < ms.size(); ++i) {
        const double a = std::exp(ms[i] - M);
        l += ls[i] * a;
        for (int c = 0; c < d; ++c) acc[c] += Os[i][c] * a;
    }
    O.assign(d, 0.0);
    for (int c = 0; c < d; ++c) O[c] = acc[c] / l;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    const int seq      = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int d        = (argc > 2) ? std::atoi(argv[2]) : 128;
    const int hq       = (argc > 3) ? std::atoi(argv[3]) : 32;
    const int hkv      = (argc > 4) ? std::atoi(argv[4]) : 8;
    const int n_split  = (argc > 5) ? std::atoi(argv[5]) : 8;
    const int page_tok = 16;
    const int rep = hq / hkv;
    const float scale = 1.0f / std::sqrt((float)d);
    const int sm_count = 132;                 // H100 SXM（规格值，specs.py）

    std::printf("=== 05_decode_attention：S=%d d=%d heads_q=%d heads_kv=%d "
                "n_split=%d ===\n\n", seq, d, hq, hkv, n_split);

    std::printf("[1] decode 的算术强度（**推导值**，verify.py §B9）：\n");
    std::printf("    每 token：FLOPs = 4·S·d·heads_q = %.1f MFLOP\n",
                4.0 * seq * d * hq / 1e6);
    std::printf("              字节  = 2·S·d·heads_kv·esize = %.1f MB\n",
                2.0 * (double)seq * d * hkv * 2 / 1e6);
    const double r = (double)hkv / hq;
    const double ai = 2.0 / (2.0 * r);
    std::printf("    ⟹ AI = 2·r/esize = r = %.2f FLOP/byte（**S 与 d 都约掉了**）\n", ai);
    std::printf("       I*(FP16, H100) = 295.4 ⟹ decode 只到它的 %.2f%%\n", ai / 295.4 * 100);
    std::printf("       访存时间 ÷ 算力时间 = I*/AI = %.1f 倍\n", 295.4 / ai);
    std::printf("    每 token 时间（推导值）= %.1f MB ÷ 3350 GB/s = %.1f µs\n\n",
                2.0 * seq * d * hkv * 2 / 1e6, 2.0 * seq * d * hkv * 2 / 3350e9 * 1e6);

    std::printf("[2] GQA/MQA 按比例削流量（**推导值**，verify.py §B8）：\n");
    std::printf("    heads_kv  相对 MHA   KV 字节\n");
    for (int h : {32, 16, 8, 4, 2, 1}) {
        std::printf("    %8d  %8.3f   %.1f MB\n", h, (double)h / 32,
                    2.0 * seq * d * h * 2 / 1e6);
    }
    std::printf("    ⟹ 减少的是**读取**字节，不是 FLOPs。\n\n");

    std::printf("[3] 并行度（**推导值**，verify.py §C7）：\n");
    const int grid = 1 * hq;                  // batch = 1
    std::printf("    不切 KV：grid = batch × heads_q = %d，占 %d 个 SM 的 %.1f%%\n",
                grid, sm_count, (double)grid / sm_count * 100);
    std::printf("    切 KV：grid = %d × n_split(%d) = %d，每 SM 分到 %.2f 个 block\n\n",
                grid, n_split, grid * n_split, (double)grid * n_split / sm_count);

    std::printf("[4] paged KV（**推导值**，verify.py §B10）：\n");
    const double page_bytes = (double)page_tok * d * 2;
    std::printf("    一页 = %d 行 × %d × 2 B = %.1f KB；页表 %d 项 × 4 B = %.1f KB\n",
                page_tok, d, page_bytes / 1024, (seq + page_tok - 1) / page_tok,
                (seq + page_tok - 1) / page_tok * 4 / 1024.0);
    std::printf("    KV 本体 = %.1f MB ⟹ 页表占 %.4f%%，碎片（半页）占 %.3f%%\n",
                2.0 * seq * d * hkv * 2 / 1e6,
                (seq + page_tok - 1) / page_tok * 4.0 / (2.0 * seq * d * hkv * 2) * 100,
                (page_tok / 2.0 * 2 * d * hkv * 2) / (2.0 * seq * d * hkv * 2) * 100);
    const int Bc = 64;
    const double cycles = 2.0 * 128 * Bc * d / (989.5e12 / sm_count) * 1.8e9;
    std::printf("    一个 Bc=%d 的 KV tile 跨 %d 页 ⟹ %d 次页表查询；"
                "该 tile 计算 ≈ %.0f 个周期\n",
                Bc, (Bc + page_tok - 1) / page_tok, (Bc + page_tok - 1) / page_tok, cycles);
    std::printf("    片外往返 400~800 周期（规格值）⟹ **同量级**：页表必须预取。\n\n");

    // ---- 数值校验：GQA 与「把 KV 头复制 rep 份」等价；split-KV 合并与单次一致 ----
    std::mt19937 gen(20260916);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> fQ((size_t)d), fK((size_t)seq * d), fV((size_t)seq * d);
    for (auto& x : fQ) x = nd(gen);
    for (auto& x : fK) x = nd(gen);
    for (auto& x : fV) x = nd(gen);

    auto attn1 = [&](int lo, int hi, double& mm, double& ll, std::vector<double>& oo) {
        std::vector<double> s(hi - lo);
        mm = -1e300;
        for (int j = 0; j < hi - lo; ++j) {
            double t = 0.0;
            for (int c = 0; c < d; ++c) t += (double)fQ[c] * (double)fK[(size_t)(lo + j) * d + c];
            s[j] = t * scale;
            mm = std::max(mm, s[j]);
        }
        ll = 0.0;
        oo.assign(d, 0.0);
        std::vector<double> p(hi - lo);
        for (int j = 0; j < hi - lo; ++j) { p[j] = std::exp(s[j] - mm); ll += p[j]; }
        for (int c = 0; c < d; ++c) {
            double t = 0.0;
            for (int j = 0; j < hi - lo; ++j) t += p[j] * (double)fV[(size_t)(lo + j) * d + c];
            oo[c] = t;                                  // 未归一化的 Σp·v
        }
    };

    double m_full, l_full;
    std::vector<double> o_full;
    attn1(0, seq, m_full, l_full, o_full);
    std::vector<double> full(d);
    for (int c = 0; c < d; ++c) full[c] = o_full[c] / l_full;

    const int per = (seq + n_split - 1) / n_split;
    std::vector<double> ms, ls;
    std::vector<std::vector<double>> Os;
    for (int i = 0; i < n_split; ++i) {
        const int lo = i * per, hi = std::min(seq, lo + per);
        if (lo >= hi) continue;
        double mm, ll;
        std::vector<double> oo;
        attn1(lo, hi, mm, ll, oo);
        ms.push_back(mm); ls.push_back(ll); Os.push_back(oo);
    }
    std::vector<double> merged;
    merge_splits_cpu(ms, ls, Os, d, merged);

    double worst = 0.0, norm = 1e-30;
    for (int c = 0; c < d; ++c) {
        worst = std::fmax(worst, std::fabs(merged[c] - full[c]));
        norm = std::fmax(norm, std::fabs(full[c]));
    }
    std::printf("[5] split-KV 合并（**推导值**，CPU double，verify.py §A5 同一条式子）：\n");
    std::printf("    n_split=%d 合并后与单次计算的最大差 = %.3e（按范数 %.3e）\n",
                n_split, worst, worst / norm);
    std::printf("    ⟹ O 与 l 用的是**同一个 α = e^{m_i−M}**；只合并 l 会得到\n");
    std::printf("       「每段权重都是 1」的错结果（verify.py §C4 是它的前向版本）。\n\n");

    std::printf("⚠ 本仓库的写作环境没有 CUDA 工具链也没有 GPU：上面除 [5] 之外\n");
    std::printf("   没有运行结果，全部是推导值或规格值。`make verify` 的\n");
    std::printf("   §B8/§B9/§B10/§C7 可复核这些算式。\n");
    return 0;
}
