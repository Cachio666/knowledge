"""roofline 模型：算术强度如何决定一个算子受限于算力还是带宽。

运行：python3 09_roofline_model.py
依赖：numpy（仅用于格式化与对数轴；全部计算在本机 CPU 上完成，不需要 GPU）

硬件规格来自公开规格表，不凭记忆。口径声明如下：

  · NVIDIA H100 SXM（Hopper / sm_90 / 80 GB HBM3）。
    NVIDIA 规格表上的 Tensor Core 数字带 "*"，脚注写明 **with sparsity**
    （2:4 结构化稀疏）。本程序一律使用**不含稀疏的 dense 值**，
    它是 sparse 值的一半。两者相差正好 2 倍，混用会得出错误的 I*。
  · 交叉核对：989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/byte，
    与 MLSysBook 附录（mlsysbook.ai/vol1/backmatter/appendix_assumptions.html）
    和 Modal GPU Glossary 给出的 ridge point 295 一致。
"""

import math

import numpy as np


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 硬件常量
# NVIDIA H100 SXM。dense = 不含 2:4 结构化稀疏；sparse 恒为 dense 的 2 倍。
H100_SXM = {
    "FP64":        (34e12,     34e12),        # 非 Tensor Core
    "FP64 TC":     (67e12,     67e12),
    "FP32":        (67e12,     67e12),        # 非 Tensor Core
    "TF32 TC":     (494.5e12,  989e12),
    "BF16 TC":     (989.5e12,  1979e12),
    "FP16 TC":     (989.5e12,  1979e12),
    "FP8 TC":      (1979e12,   3958e12),
}
H100_SXM_BW = 3.35e12          # B/s，80 GB HBM3
H100_SXM_MEM = 80 * 2 ** 30    # 80 GB

# 对照：H100 PCIe（HBM2e，80 GB）
H100_PCIE = {"BF16 TC": (756.5e12, 1513e12), "TF32 TC": (378e12, 756e12)}
H100_PCIE_BW = 2.0e12

# 本程序通篇使用的平衡点：BF16 dense
I_STAR = H100_SXM["BF16 TC"][0] / H100_SXM_BW
P_PEAK = H100_SXM["BF16 TC"][0]


def attained_tflops(I):
    """roofline 屋顶的高度，单位 TFLOPS。"""
    return min(P_PEAK, H100_SXM_BW * I) / 1e12

rule("1. 算术强度：判断瓶颈类型的唯一自变量")

print("一个算子要执行两类动作：做浮点运算，和搬字节。二者的**比值**决定了")
print("它会被哪一边卡住：")
print()
print("    I = FLOPs / Bytes        （单位：FLOP/byte，算术强度 arithmetic intensity）")
print()
print("为什么它是「唯一」的自变量：")
print()
print("  设一个算子总共做 W 次浮点运算、搬 Q 个字节。")
print("  机器的峰值算力是 P（FLOP/s），峰值带宽是 B（byte/s）。")
print("  那么：")
print()
print("      计算耗时  t_cmp = W / P")
print("      访存耗时  t_mem = Q / B")
print()
print("  两者是**串行依赖**的（数据不到就算不了），所以总时间不是相加，是取大：")
print()
print("      t = max(t_cmp, t_mem) = max(W/P, Q/B) = (Q/B) · max(W·B/(Q·P), 1)")
print()
print("  括号里那个 W·B/(Q·P) 可以改写成 I / I*（见第 3 节）。")
print("  于是「谁大」这个问题**只由 I 与 I* 的比决定**——")
print("  与算子的规模 M、N、K 的绝对值无关，与 FLOPs 总量无关。")
print("  这就是为什么可以把所有算子都画在同一张 roofline 图上。")

print()
print("口径声明：本节及全篇的 Bytes 只统计**必须往返 HBM 的字节**——")
print("读输入、写输出。寄存器与共享内存里的中间累加、L2 命中的复用一律不计，")
print("因为 roofline 描述的是「算力与 HBM 带宽」两个上界，不是某个具体 kernel 的实现。")
print("一次乘加（FMA）计 2 FLOPs：a·b+c 里有 1 次乘、1 次加。")

rule("2. 机器平衡点 I*：把 H100 SXM 的真实规格代进去")

print("令 t_cmp = t_mem，解出那个分界点：")
print()
print("    W/P = Q/B   ⟹   W/Q = P/B   ⟹   I* = P_peak / B_peak")
print()
print("I* 完全由机器决定，与算子无关。规格来源：NVIDIA H100 datasheet 规格表。")
print()
print("  %-10s %16s %16s %14s  %s"
      % ("精度", "dense TFLOPS", "含稀疏 TFLOPS", "I* (FLOP/B)", "说明"))
print("  " + "-" * 76)
for name, (dense, sparse) in H100_SXM.items():
    star = dense / H100_SXM_BW
    note = {"FP64": "非 Tensor Core，标量 FP64 单元",
            "FP64 TC": "Tensor Core",
            "FP32": "非 Tensor Core",
            "TF32 TC": "Tensor Core，输入截断到 10 位尾数",
            "BF16 TC": "Tensor Core，LLM 训练主力",
            "FP16 TC": "Tensor Core，与 BF16 同吞吐",
            "FP8 TC": "Tensor Core，LLM 推理主力"}[name]
    print("  %-10s %16.1f %16.1f %14.1f  %s" % (name, dense / 1e12, sparse / 1e12, star, note))

print()
print("显存：%.0f GB HBM3，带宽 %.2f TB/s（%s 规格表的 3.35 TB/s）"
      % (H100_SXM_MEM / 2 ** 30, H100_SXM_BW / 1e12, "NVIDIA H100 datasheet"))
print("互联：NVLink 4，900 GB/s **双向**（每方向 450 GB/s），共 18 条 link × 50 GB/s")
print("功耗：TDP 最高 700 W（可配置）")

print()
print("**I* 随精度变化极大，这就是本节存在的理由。**")
print("同一台机器，换个精度，平衡点差 %.0f 倍："
      % (H100_SXM["FP8 TC"][0] / H100_SXM["FP64"][0]))
print()
print("  %-10s %12s   %s" % ("精度", "I* (FLOP/B)", "相对 BF16"))
print("  " + "-" * 52)
base = H100_SXM["BF16 TC"][0] / H100_SXM_BW
for name in ("FP64", "FP32", "TF32 TC", "BF16 TC", "FP8 TC"):
    star = H100_SXM[name][0] / H100_SXM_BW
    print("  %-10s %12.1f   %6.2f 倍" % (name, star, star / base))
print()
print("说「H100 的 roofline 拐点是 295 FLOP/byte」时，**必须同时说清是 BF16 dense**。")
print("换成 FP32 就变成 20.0，结论会反转：一个 I = 100 的算子，")
print("在 BF16 下是带宽受限（100 < 295），在 FP32 下却是算力受限（100 > 20）。")
print("不写精度的 roofline 论断是没有意义的。")

print()
print("把同代不同封装、同封装不同代都摆在一起看 I* 往哪边走：")
print()
print("  %-12s %14s %14s %14s %10s" % ("型号", "BF16 dense", "显存带宽", "I* (BF16)", "相对 H100SXM"))
print("  " + "-" * 70)
for label, bw, peak in (("H100 PCIe", H100_PCIE_BW, H100_PCIE["BF16 TC"][0]),
                        ("H100 SXM", H100_SXM_BW, H100_SXM["BF16 TC"][0]),
                        ("H200 SXM", 4.8e12, 989.5e12)):
    print("  %-12s %11.1f TF %11.2f TB/s %14.1f %13.2f 倍"
          % (label, peak / 1e12, bw / 1e12, peak / bw,
             (peak / bw) / (H100_SXM["BF16 TC"][0] / H100_SXM_BW)))
print()
print("（H200 SXM：与 H100 SXM 同一 GH100 核心，算力完全相同，")
print("  只有显存换成 141 GB HBM3e、带宽 4.8 TB/s。来源：NVIDIA H200 datasheet。）")
print()
print("两个对比给出的是**同一个方向**：")
print()
print("  · 同代换封装（PCIe → SXM）：算力 ×%.2f，带宽 ×%.2f ⇒ I* 从 %.0f 降到 %.0f"
      % (H100_SXM["BF16 TC"][0] / H100_PCIE["BF16 TC"][0], H100_SXM_BW / H100_PCIE_BW,
         H100_PCIE["BF16 TC"][0] / H100_PCIE_BW, I_STAR))
print("  · 同封装换代（H100 → H200）：算力 ×1.00，带宽 ×%.2f ⇒ I* 从 %.0f 降到 %.0f"
      % (4.8e12 / H100_SXM_BW, I_STAR, 989.5e12 / 4.8e12))
print()
print("**在这两组数据里，带宽的涨幅都不低于算力，所以 I* 都是左移的。**")
print("也就是说：这些机器换代的净效果是「更多算子被划到带宽受限那一侧」，")
print("不是反过来。（本程序初稿在这里写的是「算力涨得比带宽快、I* 右移」，")
print("被上表数字直接推翻，已按实测数字改正。）")
print()
print("结论分两层：")
print("  ① 不要凭印象假定 I* 的移动方向。它等于 Δ算力 / Δ带宽，")
print("     哪边快就朝哪边走，**每一代必须用规格表实算**。")
print("  ② 就现有这几款 Hopper 而言，方向是左移。这不仅是个数字游戏：")
print("     H200 相对 H100 的推理提速（NVIDIA 标称 Llama-2-70B 约 1.9×）")
print("     几乎全部来自这条左移的斜线——decode 是带宽受限的，")
print("     而 H200 一 FLOP 的算力都没多给。")

rule("3. roofline 两段线：推导")

print("硬件的能力边界有两条：算力不能超过 P_peak，带宽不能超过 B_peak。")
print("两条约束同时成立，所以可达算力取两者的小者：")
print()
print("    P_attained(I) = min( P_peak ,  B_peak × I )")
print()
print("推导（从第 1 节的时间式出发）：")
print()
print("    t = max(W/P, Q/B)")
print("    可达算力 = W / t = W / max(W/P, Q/B)")
print("             = min( W·P/W , W·B/Q )          ←  1/max(a,b) = min(1/a, 1/b)")
print("             = min( P , B·(W/Q) )")
print("             = min( P_peak , B_peak · I )    ∎")
print()
print("分界点就是两条线的交点 B·I = P，即 I = P/B = I*：")
print()
print("  · I < I*：B·I < P，带宽线更低 ⇒ 可达算力 = B_peak × I")
print("           时间被访存主导，叫 **memory-bound（带宽受限）**。")
print("           此时可达算力与 I 成正比——提高算术强度能线性提升性能。")
print("  · I > I*：P 更低 ⇒ 可达算力 = P_peak，饱和。")
print("           时间被计算主导，叫 **compute-bound（算力受限）**。")
print("           此时再怎么提高算术强度也拿不到更多算力，只能换更快的精度。")
print()
print("把比值关系写清楚（这是两段线为何只差一个 I/I* 的原因）：")
print()
print("    t_cmp / t_mem = (W/P) / (Q/B) = (W/Q) · (B/P) = I / I*")
print()
print("    I / I* < 1  ⟺  t_cmp < t_mem  ⟺  memory-bound")
print()
print("也就是说：**算子落在哪一侧，等价于 I 与 I* 谁大**，没有第三种情形。")

rule("4. 把 LLM 的典型算子放上去")

print("用一组具体的 LLM 形状。取 Llama-3-70B 量级的超参：")
D_MODEL, N_LAYER, N_HEAD, N_KV, D_HEAD, D_FF, VOCAB = 8192, 80, 64, 8, 128, 28672, 128256
print("  d_model = %d, n_layers = %d, n_heads = %d, n_kv_heads = %d, head_dim = %d"
      % (D_MODEL, N_LAYER, N_HEAD, N_KV, D_HEAD))
print("  d_ff = %d, vocab = %d" % (D_FF, VOCAB))
print()
print("权重按 BF16 存储（每元素 2 byte），对照的峰值取 BF16 dense。")
print("本节的 I* = %.1f FLOP/byte（BF16 dense）。" % I_STAR)

ESZ = 2  # BF16

# (标签, 形状说明, FLOPs 推导文字, FLOPs 值, Bytes 推导文字, Bytes 值)
ops = []

# --- prefill 大 GEMM
for M, tag in ((4096, "prefill 大 GEMM（长序列）"), (512, "prefill 小 GEMM（短序列）")):
    K = N = D_MODEL
    f = 2 * M * K * N
    b = ESZ * (M * K + K * N + M * N)
    ops.append((tag,
                "X[%d×%d] @ W[%d×%d]" % (M, K, K, N),
                "2·M·K·N = 2 × %d × %d × %d" % (M, K, N), f,
                "2·(M·K + K·N + M·N) = 2 × (%d + %d + %d)" % (M * K, K * N, M * N), b))

# --- decode GEMV
M, K, N = 1, D_MODEL, D_MODEL
f = 2 * M * K * N
b = ESZ * (M * K + K * N + M * N)
ops.append(("decode GEMV（batch=1）",
            "x[1×%d] @ W[%d×%d]" % (K, K, N),
            "2·1·K·N = 2 × %d × %d" % (K, N), f,
            "2·(K + K·N + N) = 2 × (%d + %d + %d)" % (K, K * N, N), b))

# --- attention QK^T / PV
S, d = 4096, D_HEAD
f = 2 * S * S * d
b_qk = ESZ * (S * d + S * d + S * S)
ops.append(("attention QK^T",
            "Q[%d×%d] @ Kᵀ[%d×%d] → scores[%d×%d]" % (S, d, d, S, S, S),
            "2·S²·d = 2 × %d × %d" % (S * S, d), f,
            "2·(S·d + S·d + S²) = 2 × (%d + %d + %d)" % (S * d, S * d, S * S), b_qk))
b_pv = ESZ * (S * S + S * d + S * d)
ops.append(("attention PV",
            "P[%d×%d] @ V[%d×%d] → O[%d×%d]" % (S, S, S, d, S, d),
            "2·S²·d = 2 × %d × %d" % (S * S, d), f,
            "2·(S² + S·d + S·d) = 2 × (%d + %d + %d)" % (S * S, S * d, S * d), b_pv))

# --- softmax
f = 5 * S * S
b = ESZ * (S * S + S * S)
ops.append(("softmax（%d×%d）" % (S, S),
            "逐元素：max + 减 + exp + 累加 + 除 = 5 FLOP/元素",
            "5·S² = 5 × %d" % (S * S), f,
            "2·(读 S² + 写 S²) = 2 × (%d + %d)" % (S * S, S * S), b))

# --- LayerNorm
M, d = 4096, D_MODEL
f = 6 * M * d
b = ESZ * (M * d + M * d)
ops.append(("LayerNorm",
            "[%d×%d] 沿 d 归一化" % (M, d),
            "6·M·d = 6 × %d（每元素 6 次运算）" % (M * d), f,
            "2·(读 M·d + 写 M·d) = 2 × (%d + %d)" % (M * d, M * d), b))

# --- SiLU
M, d = 4096, D_FF
f = 5 * M * d
b = ESZ * (M * d + M * d)
ops.append(("SiLU（elementwise）",
            "[%d×%d] 逐元素" % (M, d),
            "5·M·d = 5 × %d（每元素 5 次运算）" % (M * d), f,
            "2·(读 M·d + 写 M·d) = 2 × (%d + %d)" % (M * d, M * d), b))

print()
print("逐个算子的算术强度（每个都给出 FLOPs 与 Bytes 的来源）：")
for i, (tag, shape, fexpr, f, bexpr, b) in enumerate(ops, 1):
    I = f / b
    attained = min(P_PEAK, H100_SXM_BW * I)
    side = "memory-bound" if I < I_STAR else "compute-bound"
    print()
    print("  【%d】%s" % (i, tag))
    print("       形状：%s" % shape)
    print("       FLOPs = %-62s = %.3e" % (fexpr, f))
    print("       Bytes = %-62s = %.3e" % (bexpr, b))
    print("       I     = FLOPs / Bytes = %.3e / %.3e = %.3e FLOP/byte"
          % (f, b, I))
    print("       判定  = I / I* = %.3e / %.1f = %.4g  ⇒  %s" % (I, I_STAR, I / I_STAR, side))
    print("       可达算力 = min(P_peak, B_peak·I) = %.3e TFLOPS，为峰值的 %.2f%%"
          % (attained / 1e12, attained / P_PEAK * 100))

print()
print("汇总表（按 I 排序）：")
print()
print("  %-22s %13s %13s %13s %-15s" % ("算子", "FLOPs", "Bytes", "I (FLOP/B)", "落点"))
print("  " + "-" * 82)
for tag, shape, fexpr, f, bexpr, b in sorted(ops, key=lambda o: o[3] / o[5]):
    I = f / b
    print("  %-22s %13.3e %13.3e %13.3e %-15s"
          % (tag, f, b, I, "memory-bound" if I < I_STAR else "compute-bound"))

print()
print("从这张表能读出四件事，每一条都能在上面找到对应数字：")
print()
print("① **同一类运算，batch/序列维一变，落点就翻转。**")
print("   prefill GEMM 在 M=4096 时 I = %.0f，是 I* 的 %.1f 倍；"
      % (ops[0][3] / ops[0][5], (ops[0][3] / ops[0][5]) / I_STAR))
print("   M 缩到 512 后 I 掉到 %.0f，只剩 I* 的 %.1f 倍。"
      % (ops[1][3] / ops[1][5], (ops[1][3] / ops[1][5]) / I_STAR))
print("   M 缩到 1（decode GEMV）后 I = %.3f，是 I* 的 1/%.0f。"
      % (ops[2][3] / ops[2][5], I_STAR / (ops[2][3] / ops[2][5])))
print("   **同一个矩阵乘 kernel，形状不同就是两种瓶颈**——这就是为什么")
print("   推理引擎要为大 batch 和小 batch 准备不同的 kernel。")
print()
print("② **注意力是一个纯带宽受限算子。** QK^T 与 PV 的 I 都是 %.1f。"
      % (ops[3][3] / ops[3][5]))
print("   原因是 S×S 的中间矩阵在分子上只出现一次（乘 d），在分母上却带着")
print("   一个 S² 项。把 I = 2S²d / 2(2Sd + S²) = S·d/(2d + S) 对 S→∞ 取极限")
print("   得到 I → d = %d——**序列越长越趋近 head_dim，永远够不到 I***。" % D_HEAD)
print("   而且收敛很慢：S=4096、d=%d 时 I = %.1f，只有渐近值 %d 的 %.0f%%，"
      % (D_HEAD, 4096 * D_HEAD / (2 * D_HEAD + 4096), D_HEAD,
         (4096 * D_HEAD / (2 * D_HEAD + 4096)) / D_HEAD * 100))
print("   要到 S ≈ 100·d = %d 才走到 98%%。换句话说，在实用序列长度范围内"
      % (100 * D_HEAD))
print("   head_dim 直接就是它的算术强度上限，而 head_dim 通常是 64～128。")
print("   只有把 S×S 矩阵留在片上（flash attention），分母里的 S² 项才消失。")
print("   这时两个矩阵乘都算上：FLOPs = 4S²d，Bytes 只剩读 Q/K/V 写 O = 2·4Sd，")
print("   I = 4S²d / 8Sd = S/2 = %d（S=%d）——从带宽受限翻到算力受限。"
      % (S // 2, S))
print("   这才是 flash attention 真正省下来的东西：不是 FLOPs（一点没少），")
print("   是那 S²×2 = %.1f MB 中间矩阵的往返搬运。" % (S * S * ESZ / 1e6))
print()
print("③ **所有 elementwise / 归约算子的 I 都在 1～2 之间。**")
print("   softmax %.2f（max、减、exp、累加、除）、LayerNorm %.2f（均值、减、平方、"
      % (ops[5][3] / ops[5][5], ops[6][3] / ops[6][5]))
print("   rsqrt、乘 γ、加 β）、SiLU %.2f（取负、exp、加 1、取倒数、乘 x）。"
      % (ops[7][3] / ops[7][5]))
print("   它们对每个元素只做常数次运算，却必须读一次写一次（4 byte/元素）。")
print("   离 I* = %.0f 差 %.1f 个数量级，**可达算力不到峰值的 1%%**"
      % (I_STAR, math.log10(I_STAR / 1.25)))
print("   （表里 softmax 只有 %.2f TFLOPS，即峰值的 %.2f%%）。"
      % (attained_tflops(1.25), attained_tflops(1.25) / (P_PEAK / 1e12) * 100))
print("   所以「把 LayerNorm 算得更快」这件事，从 roofline 看是没有意义的；")
print("   有意义的是「少搬一次」（算子融合）。")
print()
print("④ **只有在 I 附近才谈得上调优。** I ≫ I* 时该做的是换低精度峰值；")
print("   I ≈ I* 时每一点算术强度都是真金白银；I ≪ I* 时只该做融合与量化。")

rule("5. decode 为什么必然是 memory-bound：70B 模型的定量账")

P_PARAMS = 70e9
W_BYTES = P_PARAMS * ESZ
print("设模型有 P = %.0f 个参数，权重以 BF16 存（%d byte/参数）。" % (P_PARAMS, ESZ))
print()
print("decode 阶段每生成一个 token，要做的运算是「权重矩阵 × 向量」：")
print("  · 每个参数恰好参与一次乘加 ⇒ FLOPs = 2·P")
print("  · 而**没有任何复用**：读进一个权重、用一次、丢掉")
print("    ⇒ Bytes ≥ 权重总量 = P × 2")
print()
print("两式相除，向量长度、层数、隐藏维全部约掉：")
print()
print("    I_decode = 2P / (2P) = 1.0 FLOP/byte")
print()
print("**这个 1.0 与模型大小无关，与精度无关，是「权重每被读一次只用一次」的必然结果。**")
print("用第 4 节同一套口径算出的 decode GEMV 行（解析计算，不是 GPU 实测）：")
I_GEMV = ops[2][3] / ops[2][5]
print("    I = %.4f FLOP/byte" % I_GEMV)
print("    比 1.0 略小，因为还要读写输入输出向量共 %.0f 字节，"
      % (2 * (D_MODEL + D_MODEL)))
print("    相对 %.3e 字节的权重矩阵可以忽略。" % (D_MODEL * D_MODEL * ESZ))
print()
print("代入 H100 SXM 的硬件常数：")
t_mem = W_BYTES / H100_SXM_BW
t_cmp = 2 * P_PARAMS / P_PEAK
print()
print("  读一遍权重的字节数  = %.0f × %d = %.3e byte = %.1f GB"
      % (P_PARAMS, ESZ, W_BYTES, W_BYTES / 1e9))
print("  访存时间下限 t_mem  = %.3e / %.2e = %.4f s = %.1f ms"
      % (W_BYTES, H100_SXM_BW, t_mem, t_mem * 1e3))
print("  计算时间        t_cmp = 2 × %.0f / %.3e = %.4f s = %.4f ms"
      % (P_PARAMS, P_PEAK, t_cmp, t_cmp * 1e3))
print("  比值 t_mem / t_cmp  = %.1f 倍" % (t_mem / t_cmp))
print()
print("  单 token 吞吐下限   = 1 / %.4f s = %.1f token/s" % (t_mem, 1 / t_mem))
print("  **算力利用率        = t_cmp / t_mem = %.4f%%**"
      % (t_cmp / t_mem * 100))
print("  （等价地：I / I* = %.3f / %.1f = %.4f%%）"
      % (I_GEMV, I_STAR, I_GEMV / I_STAR * 100))

print()
print("也就是说：H100 有 %.0f TFLOPS 的 BF16 算力，而 decode 一个 70B 模型"
      % (P_PEAK / 1e12))
print("只能用到其中的 %.2f%%——**超过 99.6%% 的 Tensor Core 在空转**。"
      % (t_cmp / t_mem * 100))
print("这不是实现问题。只要权重必须以「读一次用一次」的方式过一遍 HBM，")
print("这个上限就搬不掉。唯一的出路是提高复用：加 batch，或者用更小的权重。")

print()
print("顺带一个常被忽略的事实：")
print("  BF16 权重 %.0f GB **大于** 单卡 H100 SXM 的 %.0f GB HBM。"
      % (W_BYTES / 1e9, H100_SXM_MEM / 2 ** 30))
print("  所以 70B 单卡根本放不下，至少需要 2 卡（张量并行 TP=2）。")
print("  TP=N 时每卡持有 1/N 的权重，但**每张卡仍然要把自己那份完整读一遍**：")
print()
print("  %-8s %16s %16s %18s" % ("TP", "每卡权重 (GB)", "每卡 t_mem (ms)", "全系统 token/s"))
print("  " + "-" * 62)
for tp in (2, 4, 8, 16):
    per = W_BYTES / tp
    tt = per / H100_SXM_BW
    print("  %-8d %16.1f %16.2f %18.1f" % (tp, per / 1e9, tt * 1e3, 1 / tt))
print()
print("每卡吞吐恒为 %.1f token/s——**张量并行不改变每卡的算力利用率**，"
      % (1 / t_mem))
print("它只是把一份装不下的权重切开，让 N 张卡同时为**同一个 token** 出力。")
print("总吞吐线性涨，利用率一点没涨。这正是 decode 优化的核心矛盾。")
print()
print("口径声明：上表的 token/s 是**纯带宽上界**，假设张量并行没有任何通信开销。")
print("真实的 TP 每层要做两次 all-reduce（attention 后、FFN 后），走 NVLink，")
print("所以真实系统的吞吐会明显低于这一列。本表的作用是给出「就算通信全免费，")
print("也就这么多」——它是上界，不是预测值。")

print()
print("再把 KV cache 的读取加上，就能看出 batching 的作用：")
kv_per_tok = 2 * N_LAYER * N_KV * D_HEAD * ESZ
print("  每 token 每层的 KV cache = 2(K 和 V) × n_kv_heads × head_dim × 2 byte")
print("                            = 2 × %d × %d × %d = %d byte"
      % (N_KV, D_HEAD, ESZ, kv_per_tok // N_LAYER))
print("  全模型 K+V 合计 = %.4f MB/token（上下文每长一个 token 就多这么多）"
      % (kv_per_tok / 1e6))
print()
print("同时必须补上前面漏掉的一项：**decoding 时注意力本身也有 FLOPs**。")
print("当前 token 要对 S 个历史 token 做 QK^T 和 PV，每层每 head 是 4·S·d_head，")
print("所以每 token 是 4·S·d_model·n_layers = 4 × S × %d × %d = %.3e × S。"
      % (D_MODEL, N_LAYER, 4 * D_MODEL * N_LAYER))
print("S=8192 时这一项 = %.3e，是线性层 2P = %.3e 的 %.1f%%。"
      % (4 * 8192 * D_MODEL * N_LAYER, 2 * P_PARAMS,
         4 * 8192 * D_MODEL * N_LAYER / (2 * P_PARAMS) * 100))
print("短上下文时可以忽略，**长上下文时不能**——下面把它算进去。")
print()


def decode_intensity(B, S):
    fl = B * (2 * P_PARAMS + 4 * S * D_MODEL * N_LAYER)
    by = W_BYTES + B * S * kv_per_tok
    return fl / by, fl, by


print("  %-6s %12s %12s %12s %12s %12s"
      % ("B", "权重 (GB)", "KV (GB)", "总字节 (GB)", "I (F/B)", "利用率"))
print("  " + "-" * 76)
for B in (1, 8, 32, 64, 128, 256):
    I, fl, by = decode_intensity(B, 8192)
    util = min(P_PEAK, H100_SXM_BW * I) / P_PEAK
    print("  %-6d %12.1f %12.1f %12.1f %12.2f %11.2f%%"
          % (B, W_BYTES / 1e9, (by - W_BYTES) / 1e9, by / 1e9, I, util * 100))
print()
print("两个效应同时发生：")
print("  · 权重 %.0f GB 是**一次性**的，不随 B 增长；B 越大，这部分越被摊薄。"
      % (W_BYTES / 1e9))
print("  · KV cache 是**每序列各一份**，随 B 线性增长。")
print()
print("所以 I 随 B 上升，利用率从 B=1 的 %.2f%% 涨到 B=256 的 %.2f%%，但**始终**"
      % (min(P_PEAK, H100_SXM_BW * decode_intensity(1, 8192)[0]) / P_PEAK * 100,
         min(P_PEAK, H100_SXM_BW * decode_intensity(256, 8192)[0]) / P_PEAK * 100))
print("远低于 100%——**提高 batch 能把 decode 从「惨」救到「不那么惨」，救不回「好」**。")
print()
print("为什么 batching 的收益会**饱和**？令 B→∞：")
print()
print("    I_∞(S) = (2P + 4·S·d_model·n_layers) / (S·kv)")
print("           = 2P/(S·kv) + 4·d_model·n_layers/kv")
print("           = %.1f + %.1f      （前项随 S 缩小，后项与 S 无关）"
      % (2 * P_PARAMS / (8192 * kv_per_tok), 4 * D_MODEL * N_LAYER / kv_per_tok))
print()
print("  %-10s %14s %14s %14s" % ("上下文 S", "2P/(S·kv)", "常数项", "I_∞(S)"))
print("  " + "-" * 56)
for Sx in (1024, 4096, 8192, 32768, 131072):
    a = 2 * P_PARAMS / (Sx * kv_per_tok)
    b = 4 * D_MODEL * N_LAYER / kv_per_tok
    print("  %-10d %14.1f %14.1f %14.1f" % (Sx, a, b, a + b))
print()
print("**长上下文确实更受带宽限制**（I_∞ 随 S 下降），这与「长上下文计算量大」的")
print("直觉相反。但注意它有**下限** %.1f：注意力自身的 FLOPs 和 KV 字节都正比于 S，"
      % (4 * D_MODEL * N_LAYER / kv_per_tok))
print("把 S 约掉了，只剩一个与 S 无关的常数项。")
print()
print("--- 这里必须修正一个过强的说法 ---")
print()
print("上表 S=1024 那一行的 I_∞ = %.1f **大于** I* = %.0f。"
      % (2 * P_PARAMS / (1024 * kv_per_tok) + 4 * D_MODEL * N_LAYER / kv_per_tok, I_STAR))
print("也就是说「decode 必然是带宽受限」这句话**在短上下文 + 大 batch 下不成立**。")
print("解 I(B,S) = I* 得到临界的 batch：")
print()
print("    B*(S) = I*·2P / ( 2P − (I*·kv − 4·d_model·n_layers)·S )")
print()
K_SLOPE = I_STAR * kv_per_tok - 4 * D_MODEL * N_LAYER
print("  其中 I*·kv − 4·d_model·n_layers = %.3e。分母在 S = %.0f 处变号："
      % (K_SLOPE, 2 * P_PARAMS / K_SLOPE))
print("  超过这个上下文长度，B* 无解 ⇒ **decode 在任何 batch 下都是带宽受限**。")
print()
print("  %-10s %16s  %s" % ("上下文 S", "临界 batch B*", "含义"))
print("  " + "-" * 62)
for Sx in (512, 1024, 2048, 2560, 3000, 8192):
    den = 2 * P_PARAMS - K_SLOPE * Sx
    if den <= 0:
        print("  %-10d %16s  %s" % (Sx, "无解", "永远带宽受限，与 batch 无关"))
    else:
        bstar = I_STAR * 2 * P_PARAMS / den
        tag = "可达（大 batch 服务）" if bstar < 1024 else "batch 大到不现实"
        print("  %-10d %16.0f  %s" % (Sx, bstar, tag))
print("  %-10.0f %16s  %s" % (2 * P_PARAMS / K_SLOPE, "∞", "分母变号的临界点"))
print()
print("所以正确的表述是：")
print()
print("  · **S ≳ %.0f 时，decode 在任何 batch 下都是带宽受限**——这是绝大多数"
      % (2 * P_PARAMS / K_SLOPE))
print("    长上下文推理服务的实际情况。")
print("  · **S 较短时存在一个有限的大 batch B*(S)**，越过去就翻转成算力受限。")
print("    B*(S) 随 S 减小而减小（S=512 时只需 %.0f），这个量级在真实服务里"
      % (I_STAR * 2 * P_PARAMS / (2 * P_PARAMS - K_SLOPE * 512)))
print("    是够得到的——**短上下文 + 大 batch 的 decode 会退化成 prefill**，")
print("    这正是「continuous batching」「chunked prefill」在工程上成立的根据。")
print()
print("（初稿这里写的是「decode 在任何 batch 下都是带宽受限」，被上面 S=1024")
print("  那一行的数字直接推翻，已改成带临界条件的版本。）")

rule("6. roofline 曲线（ASCII，双对数）")

XLIM, YLIM = (0.1, 1e4), (0.1, 3000.0)   # 纵轴单位 TFLOPS
W, H = 62, 26


def _col(x, lim, n):
    lo, hi = math.log10(lim[0]), math.log10(lim[1])
    f = (math.log10(x) - lo) / (hi - lo)
    return min(n - 1, max(0, int(round(f * (n - 1)))))


def _row(y, lim, n):
    lo, hi = math.log10(lim[0]), math.log10(lim[1])
    f = (math.log10(y) - lo) / (hi - lo)
    return min(n - 1, max(0, n - 1 - int(round(f * (n - 1)))))


grid = [[' '] * W for _ in range(H)]
for c in range(W):
    fx = c / (W - 1)
    x = 10 ** (math.log10(XLIM[0]) + fx * (math.log10(XLIM[1]) - math.log10(XLIM[0])))
    grid[_row(attained_tflops(x), YLIM, H)][c] = '*'

marklabels = [
    ("P", "prefill GEMM M=4096", ops[0][3] / ops[0][5]),
    ("p", "prefill GEMM M=512", ops[1][3] / ops[1][5]),
    ("D", "decode GEMV", ops[2][3] / ops[2][5]),
    ("Q", "QK^T", ops[3][3] / ops[3][5]),
    ("V", "PV", ops[4][3] / ops[4][5]),
    ("S", "softmax", ops[5][3] / ops[5][5]),
    ("N", "LayerNorm", ops[6][3] / ops[6][5]),
    ("E", "SiLU", ops[7][3] / ops[7][5]),
]
cellmap = {}
for ch, lab, I in marklabels:
    key = (_row(attained_tflops(I), YLIM, H), _col(I, XLIM, W))
    cellmap.setdefault(key, []).append((ch, lab, I))
collisions = []
for (r, c), items in cellmap.items():
    if len(items) == 1:
        grid[r][c] = items[0][0]
    else:
        grid[r][c] = '#'          # 不能也用 '*'，否则和屋顶线分不开
        collisions.append(items)

ylabel = {}
for e in range(-1, 4):
    n = _row(10.0 ** e, YLIM, H)
    ylabel[n] = ylabel.get(n, "") or "1e%d" % e

print()
print("可达算力 (TFLOPS)，BF16 dense。横轴 = 算术强度 I (FLOP/byte)，双对数。")
print("横轴范围 %.1f ～ 1e4，纵轴范围 %.1f ～ 3000，每格 %.2f 个数量级。"
      % (XLIM[0], YLIM[0], (math.log10(YLIM[1]) - math.log10(YLIM[0])) / (H - 1)))
print("'*' 连成的是 roofline 屋顶 min(P_peak, B_peak·I)；字母是算子；")
print("'#' 是同格算子（见下方图例）。横轴最右端的 1e4 刻度标签因宽度放不下而略去。")
print()
for r in range(H):
    print("  %-6s|%s" % (ylabel.get(r, ""), "".join(grid[r])))
print("  %-6s+%s" % ("", "-" * W))
xlab = [" "] * W
for e in range(-1, 5):
    s = "1e%d" % e
    c = _col(10.0 ** e, XLIM, W)
    if c + len(s) >= W:          # 末尾放不下就不画，避免把标签截断成假数字
        continue
    for i, ch in enumerate(s):
        xlab[c + i] = ch
print("  %-6s %s" % ("", "".join(xlab)))
print("  %-6s %s" % ("", "".join(("|" if xlab[c] != " " else " ") for c in range(W))))
print()
print("图例（括号内为算出的 I）：")
for ch, lab, I in marklabels:
    print("  %s = %-24s I = %9.3f" % (ch, lab, I))
if collisions:
    print()
    print("  同一格（'#'）里的算子——在 roofline 视角下它们没有区别：")
    for items in collisions:
        print("    # : %s" % "、".join("%s(%s, I=%.3f)" % (ch, lab, I) for ch, lab, I in items))
    print("  这不是画图偷懒，是模型本身的性质：**纯 elementwise 算子的算术强度")
    print("  与张量形状无关**，只由「每个元素算几次、读几次写几次」决定。")
    print("  注意 QK^T 与 PV 的 '#' 压在斜线上——它们恰好落在屋顶曲线上，")
    print("  说明这两个算子的可达算力就是带宽给的那个值，一点算力余量都没有。")

print()
print("读图：")
print("  · 斜线段（左）：I < %.0f 时屋顶 = B_peak × I，每十倍算术强度换十倍算力。" % I_STAR)
print("  · 水平段（右）：I > %.0f 时屋顶 = P_peak = %.0f TFLOPS，封顶。" % (I_STAR, P_PEAK / 1e12))
n_slope = sum(1 for c in range(W) if attained_tflops(
    10 ** (math.log10(XLIM[0]) + c / (W - 1) * (math.log10(XLIM[1]) - math.log10(XLIM[0]))))
    < P_PEAK / 1e12)
print("  · 拐点在 I* = %.1f 处，对应 %d 个网格列里有 %d 列是斜线。"
      % (I_STAR, W, n_slope))
print("  · 算子的落点分三撮：")
print("      D/S/E/N（I≈1～1.5）在斜线最下端，可达算力只有几个 TFLOPS；")
print("      Q/V（I=%.0f）压在斜线中段，可达算力 %.0f TFLOPS；"
      % (ops[3][3] / ops[3][5], attained_tflops(ops[3][3] / ops[3][5])))
print("      p/P（I=%.0f / %.0f）已经越过拐点，踩在水平段上。"
      % (ops[1][3] / ops[1][5], ops[0][3] / ops[0][5]))
print("    也就是说，**这张图上唯一能吃到峰值算力的只有 prefill 的 GEMM**。")

rule("7. 断言自检")

checks = [
    ("I* 与公开来源一致（989/3.35 ≈ 295）", abs(I_STAR - 295.2) < 0.5),
    ("decode GEMV 的 I 落在 1.0 附近", 0.99 < I_GEMV < 1.0),
    ("70B BF16 权重超过单卡 80 GB 显存", W_BYTES > H100_SXM_MEM),
    ("decode 单 token 算力利用率 < 0.5%", t_cmp / t_mem < 0.005),
    ("QK^T 在 S=4096 时是 memory-bound", ops[3][3] / ops[3][5] < I_STAR),
    ("QK^T 的 I 随 S 单调上升但不越过 head_dim",
     all((s * D_HEAD / (2 * D_HEAD + s)) < D_HEAD for s in (128, 1024, 4096, 65536))
     and 4096 * D_HEAD / (2 * D_HEAD + 4096) < 65536 * D_HEAD / (2 * D_HEAD + 65536)),
    ("S=4096 时 QK^T 的 I 已达到渐近值 head_dim 的 90% 以上",
     (4096 * D_HEAD / (2 * D_HEAD + 4096)) / D_HEAD > 0.9),
    ("prefill GEMM M=4096 是 compute-bound", ops[0][3] / ops[0][5] > I_STAR),
    ("所有 elementwise 算子 I < 2", all(o[3] / o[5] < 2 for o in ops[5:])),
    ("FP32 的 I* 低于 BF16 的 I*", 67e12 / H100_SXM_BW < I_STAR),
    ("张量并行不改变每卡 decode 利用率（时间下限恒定）",
     abs((W_BYTES / 2) / H100_SXM_BW * 2 - W_BYTES / H100_SXM_BW) < 1e-12),
    ("S=8192 时 I_∞ < I*（长上下文必然带宽受限）",
     (2 * P_PARAMS / (8192 * kv_per_tok) + 4 * D_MODEL * N_LAYER / kv_per_tok) < I_STAR),
    ("S=1024 时 I_∞ > I*（短上下文 + 大 batch 可以翻转）",
     (2 * P_PARAMS / (1024 * kv_per_tok) + 4 * D_MODEL * N_LAYER / kv_per_tok) > I_STAR),
    ("临界上下文长度在 1000～2000 之间",
     1000 < 2 * P_PARAMS / (I_STAR * kv_per_tok - 4 * D_MODEL * N_LAYER) < 2000),
    ("B*(S) 随 S 增大（S=512 的临界 batch 小于 S=1024 的）",
     (I_STAR * 2 * P_PARAMS / (2 * P_PARAMS - (I_STAR * kv_per_tok - 4 * D_MODEL * N_LAYER) * 512)
      < I_STAR * 2 * P_PARAMS / (2 * P_PARAMS - (I_STAR * kv_per_tok - 4 * D_MODEL * N_LAYER) * 1024))),
    ("KV cache 字节数正比于 n_layers（80 层 × 4096 B）",
     kv_per_tok == N_LAYER * 4096),
]
for desc, ok in checks:
    print("  [%s] %s" % ("通过" if ok else "**失败**", desc))
print()
print("全部 %d 项%s。" % (len(checks),
                        "通过" if all(o for _, o in checks) else "**未全部通过，见上**"))

rule("8. 本程序的边界")

print("  · 全部是解析计算，没有在 GPU 上跑任何 kernel。所有数字是**模型给出的上界**，")
print("    不是实测性能。本程序**不给**「能到上界的百分之几」的经验系数——")
print("    那个比例完全取决于具体 kernel（占用率、启动开销、尾效应、L2 命中），")
print("    在没跑过之前任何具体数字都是编的。")
print("    本程序只负责给出「不可能超过多少」这条线。")
print("  · 只统计必须往返 HBM 的字节。不考虑 L2 命中（H100 有 50 MB L2，")
print("    对 KV cache 和大 GEMM 的复用有实质帮助），也不考虑 cache line 未对齐的浪费。")
print("  · 峰值算力取 dense（无 2:4 结构化稀疏）。若模型真能用上稀疏，")
print("    I* 会翻倍到 %.0f，判定的两侧会整体移动。" % (I_STAR * 2))
print("  · H100 存在 boost clock 漂移，datasheet 的峰值是理论上限；")
print("    实测峰值算力通常低于标称几个百分点，实测带宽则可能因 L2 而偏高。")
print("  · 第 4 节的 FLOPs 计数口径为「exp 等超越函数计 1 次运算」。")
print("    硬件上 SFU 的 exp 吞吐远低于 FMA，所以 elementwise 算子的真实耗时")
print("    会比按此口径估的更差——这不影响它们在 roofline 上被判为带宽受限，")
print("    只影响可达算力的具体数值。")
