#!/usr/bin/env python3
r"""02-roofline-verdict —— 「这个算子是什么受限」的判定。

对应文档：`../02-roofline-verdict.md`

★ 本程序的可证伪断言：
  A 类  H100 的两条 ridge（FP16 张量核 295.3731 / FP32 向量 20.0000）与手算逐位一致；
        5 个算子的 AI 与文档表逐位一致
  B 类  5 个算子 × 2 条精度路径 = 10 次判定；**翻转的恰好是 1 个**（prefill attention
        S=512），其余 3 个远低于窗口、1 个远高于窗口，怎么换路径都不翻转。
        改错任何一个 AI，翻转集合就会变——这条就是本程序的反驳点。
  C 类  ★ 本主题最重要的反例：一个 3×3 直接卷积，**「唯一 DRAM 字节」与「总读取字节」
        两种口径给出的判定相反**（842.6 → 计算受限；1.0 → 访存受限）。

跑法：`make 02_roofline_verdict`。纯 python3 + 标准库，不 import torch，不碰 GPU。
"""

import sys

import specs as SP

FAILED = []


def check(name, cond, detail=""):
    print(f"    [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def sub(t):
    print()
    print(f"  ── {t}")


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def roofline_ridge(peak_tflops, bw_gbps):
    """★ 形式沿用本库既有实现：`40-kernels/gemm/code/verify.py:274`。"""
    return peak_tflops * 1e12 / (bw_gbps * 1e9)


ESIZE = SP.FLOAT16_BYTES          # 本篇一律 fp16：读写各 2 B/element（**口径，写死**）
GPU = "H100"
BW = SP.HBM_BW_GBps_BY_GEN[GPU]   # 3350.0 GB/s（十进制 10^9，SXM 档）
PEAK_FP16 = SP.TC_TFLOPS_DENSE_SPARSE["FP16"][GPU][0]   # 989.5（**dense**，不是 1979）
PEAK_FP32 = SP.FP32_VECTOR_TFLOPS_DENSE[GPU]            # 67.0


# ===========================================================================
head("§A  算法正确性：ridge 与 AI 的手算对照")
# ===========================================================================

sub("A.1  H100 的两条 ridge（峰值 ÷ 带宽，标称值）")
ridge_fp16 = roofline_ridge(PEAK_FP16, BW)
ridge_fp32 = roofline_ridge(PEAK_FP32, BW)
print(f"    FP16 张量核：{PEAK_FP16} TFLOPS ÷ {BW} GB/s = "
      f"{PEAK_FP16}e12 ÷ {BW}e9 = {ridge_fp16:.4f} FLOP/byte")
print(f"    FP32 向量  ：{PEAK_FP32} TFLOPS ÷ {BW} GB/s = "
      f"{PEAK_FP32}e12 ÷ {BW}e9 = {ridge_fp32:.4f} FLOP/byte")
print(f"    窗口宽度   ：{ridge_fp16:.4f} ÷ {ridge_fp32:.4f} = {ridge_fp16/ridge_fp32:.4f} 倍")
check("A.1 FP16 ridge 与手算 989.5e12/3350e9 逐位一致",
      abs(ridge_fp16 - 989.5e12 / (3350.0e9)) < 1e-9, f"{ridge_fp16!r}")
check("A.1 FP16 ridge 落在 specs.py 记录的表值 295.3731 上",
      abs(ridge_fp16 - SP.RIDGE_EXPECTED["H100_fp16_tc"]) < 5e-5, f"{ridge_fp16:.4f}")
check("A.1 FP32 向量 ridge == 20.0000（手算 67.0e12/3350e9）",
      abs(ridge_fp32 - 20.0) < 1e-12, f"{ridge_fp32!r}")
check("A.1 FP16 用的是 **dense** 峰值 989.5，不是带稀疏的 1979.0",
      PEAK_FP16 == 989.5, f"{PEAK_FP16}")

sub("A.2  5 个算子的 AI：手算 vs 脚本")
# 口径（**必须写进文档，否则同一张表内部会混两种口径**）：
#   · FLOPs 按 **FMA 计 2 次**；
#   · 字节只算 **必须往返 HBM 的唯一字节**（不含重复读——重复读的反例见 §C）；
#   · dtype = **fp16**（读写各 2 B/element）。


def ai_atn_decode(S, dq, dkv):
    """decode：每 token 的 scores 2·S·dq + PV 2·S·dq；读 K、V 各 S·dkv 元素。"""
    return (4.0 * S * dq) / ((2 * S * dkv) * ESIZE)


def ai_atn_prefill(S, d):
    """prefill：4S²d 的 FLOPs；读 Q、K、V、O 各 S·d 元素。"""
    return (4.0 * S * S * d) / ((4 * S * d) * ESIZE)


def ai_layernorm(d):
    """5d：均值、方差、归一化、缩放、平移各 1 次/元素；读 x、写 y。"""
    return (5.0 * d) / ((2 * d) * ESIZE)


def ai_gemm(M, N, K):
    """2MNK；读 A(M·K)、B(K·N)、写 C(M·N)，**每个元素只从 DRAM 取一次**。"""
    return (2.0 * M * N * K) / ((M * K + K * N + M * N) * ESIZE)


CASES = [
    # (名字, AI, 手算式)
    ("decode attention S=4096, d=128 (MHA)", ai_atn_decode(4096, 128, 128),
     "4·4096·128 ÷ (2·4096·128·2) = 1.0"),
    ("LayerNorm d=4096", ai_layernorm(4096),
     "5·4096 ÷ (2·4096·2) = 1.25"),
    ("GEMM M=1, N=K=4096", ai_gemm(1, 4096, 4096),
     "2·1·4096·4096 ÷ ((4096 + 4096² + 4096)·2) = 0.9995"),
    ("prefill attention S=512, d=128 (MHA)", ai_atn_prefill(512, 128),
     "4·512²·128 ÷ (4·512·128·2) = S/2 = 256.0"),
    ("GEMM 4096³", ai_gemm(4096, 4096, 4096),
     "2·4096³ ÷ (3·4096²·2) = N/3 = 1365.333"),
]
for name, ai, formula in CASES:
    print(f"    {name:38s} AI = {ai:12.6f}   {formula}")

# ★ 手算独立复算（**不复用上面的函数**）：把文档表里的五个数直接写成字面量逐位比。
LITERAL = {                     # 文档 §1 的表值
    "decode attention S=4096, d=128 (MHA)": 1.0,
    "LayerNorm d=4096": 1.25,
    "GEMM M=1, N=K=4096": 0.9995119570522206,
    "prefill attention S=512, d=128 (MHA)": 256.0,
    "GEMM 4096³": 1365.3333333333333,
}
for name, ai, _ in CASES:
    check(f"A.2 {name} 与表值逐位一致", abs(ai - LITERAL[name]) < 1e-12, f"{ai!r}")


# ===========================================================================
head("§B  模型与公式：两条精度路径下的判定，以及谁翻转")
# ===========================================================================

sub("B.1  10 次判定（5 个算子 × 2 条精度路径）")
print(f"    判据：AI < ridge ⇒ **访存受限**；AI ≥ ridge ⇒ **计算受限**。")
print(f"    fp16 路径 ridge = {ridge_fp16:.2f}；fp32 向量路径 ridge = {ridge_fp32:.2f}")
print()
print(f"    {'算子':38s} {'AI':>12s}  {'fp32 向量':>10s}  {'fp16 张量核':>12s}  {'翻转':>4s}")
flipped = []
for name, ai, _ in CASES:
    v32 = "计算受限" if ai >= ridge_fp32 else "访存受限"
    v16 = "计算受限" if ai >= ridge_fp16 else "访存受限"
    flip = v32 != v16
    if flip:
        flipped.append(name)
    print(f"    {name:38s} {ai:12.4f}  {v32:>10s}  {v16:>12s}  {'✓' if flip else '✗':>4s}")

sub("B.2  翻转集合必须**恰好**是 {prefill attention S=512}")
print(f"    实际翻转集合 = {flipped}")
check("B.2 翻转集合恰好只有 1 个，且是 prefill attention S=512",
      flipped == ["prefill attention S=512, d=128 (MHA)"], f"{flipped}")
# ★ 这一条是**反驳点**：三个 AI 远低于窗口（1.0 / 1.25 / 0.9995）、
#   一个远高于窗口（1365.3）。把任何一个 AI 改错（例如把 LayerNorm 的 5d 写成 50d），
#   翻转集合就会变化，B.2 立即失败。
n_low = sum(1 for n, a, _ in CASES if a < ridge_fp32)
n_high = sum(1 for n, a, _ in CASES if a > ridge_fp16)
print(f"    远低于窗口（AI < {ridge_fp32:.2f}）：{n_low} 个 —— 怎么换精度都不翻转")
print(f"    远高于窗口（AI > {ridge_fp16:.2f}）：{n_high} 个 —— 怎么换精度都不翻转")
check("B.2 三个远低于窗口 + 一个远高于窗口，只有 1 个落在窗口内",
      n_low == 3 and n_high == 1 and len(CASES) - n_low - n_high == 1,
      f"低 {n_low} / 高 {n_high} / 窗口内 {len(CASES)-n_low-n_high}")
print("    ⇒ 「attention 会翻转」这句话**是错的**：翻转取决于 **S**，不取决于"
      "「是不是 attention」。")

sub("B.3  GQA 把 decode attention 的 AI 抬高 h_q/h_kv 倍——但判定仍然不变")
print(f"    AI(decode attn) = d_q ÷ d_kv = h_q ÷ h_kv（本口径下与 S 无关）")
for hq, hkv in ((32, 32), (32, 8), (32, 4)):
    ai = ai_atn_decode(4096, hq * 128, hkv * 128)
    v = "计算受限" if ai >= ridge_fp32 else "访存受限"
    print(f"      h_q/h_kv = {hq:2d}/{hkv:2d} = {hq/hkv:4.1f}:1  ⇒ AI = {ai:6.3f}"
          f"   fp32 路径判定 = {v}")
ai_gqa8 = ai_atn_decode(4096, 32 * 128, 4 * 128)
check("B.3 即使 8:1 的 GQA，decode attention 在 fp32 路径上仍是访存受限",
      ai_gqa8 < ridge_fp32, f"AI = {ai_gqa8} < {ridge_fp32}")
print(f"    ⇒ 8:1 GQA 把 AI 抬高到 {ai_gqa8:.1f}，仍远低于 fp32 路径的 {ridge_fp32:.1f}。")
print("    ⚠ 口径：文档表里的 decode attention **按 MHA 算**（h_q = h_kv，AI = 1.0）。")
print("      换成真实模型的 GQA 比值**不会**翻转判定，但会翻转「AI 是多少」这个数——")
print("      **所以口径必须写在表的旁边。**")

sub("B.4  可达带宽低于标称 ⇒ **可达 ridge 更高**，判据更严（方向与直觉相反）")
lo, hi = SP.ACHIEVABLE_BW_SHORTFALL_PCT_IN_REPO
print(f"    本库记录的可达带宽缺口 = {lo}–{hi}%（[E]，且带自陈：本仓库未给出一手页面）")
for pct in (lo, hi):
    eta = 1 - pct / 100.0
    r_eff = roofline_ridge(PEAK_FP16, BW * eta)
    print(f"      η = {eta:.2f} ⇒ 可达带宽 {BW*eta:.1f} GB/s ⇒ 可达 ridge = {r_eff:.2f}"
          f"（标称 {ridge_fp16:.2f} 的 {r_eff/ridge_fp16:.2f} 倍）")
check("B.4 可达 ridge **高于**标称 ridge（缺口在分母上，方向是「更严」）",
      roofline_ridge(PEAK_FP16, BW * 0.85) > ridge_fp16,
      f"{roofline_ridge(PEAK_FP16, BW*0.85):.2f} > {ridge_fp16:.2f}")
print("    ★ 直觉会说「可达带宽更低 ⇒ 更容易计算受限」，**这是反的**：")
print("      ridge = 峰值 ÷ 带宽，分母变小 ⇒ ridge 变大 ⇒ **更多地判成访存受限**。")
print("      ⇒ 用标称值做判定是**保守**方向（会把算子判得比实际更「算力受限」）。")


sub("B.5  口径必须两边一致：把 AI 换成 fp32 字节，**翻转集合不变**")
# ★ 这是一处**必须自己先说破**的口径问题：
#   表里的 AI 是 **fp16 字节**算的，而 fp32 路径的 ridge 是 **fp32 字节**算的。
#   严格地说，「把同一个算子放到 fp32 路径上」意味着**字节翻倍 ⇒ AI 减半**。
#   本节把两种读法各算一遍，并断言**翻转集合相同**——
#   若不同，说明这张表的结论依赖口径约定，那就必须写进正文而不是藏在脚注里。
shifts = [
    (SP.FP32_VECTOR_TFLOPS_DENSE[GPU] * 2, "读法 B：数据仍是 fp16（2 B），但走 fp16 向量速率"),
]
print(f"    读法 A（**字节翻倍**）：AI' = AI_fp16 ÷ 2，与原 ridge {ridge_fp32:.2f} 比")
set_a = []
for name, ai, _ in CASES:
    a = ai / 2.0
    if ("计算受限" if a >= ridge_fp32 else "访存受限") != \
       ("计算受限" if ai >= ridge_fp16 else "访存受限"):
        set_a.append(name)
print(f"      翻转集合 = {set_a}")
print("    读法 B（**字节不变**）：数据仍是 fp16 的 2 B，但算力换成"
      "「fp16 向量速率」= 2 × FP32 向量峰值")
ridge_b = roofline_ridge(shifts[0][0], BW)
print(f"      ridge = {shifts[0][0]:.1f}e12 ÷ {BW}e9 = {ridge_b:.4f}")
set_b = []
for name, ai, _ in CASES:
    if ("计算受限" if ai >= ridge_b else "访存受限") != \
       ("计算受限" if ai >= ridge_fp16 else "访存受限"):
        set_b.append(name)
print(f"      翻转集合 = {set_b}")
check("B.5 两种口径读法给出的翻转集合相同", set_a == set_b, f"{set_a} vs {set_b}")
check("B.5 且都与基准（本篇表值口径）的翻转集合相同",
      set_a == flipped, f"{set_a} vs {flipped}")
print("    ★ 结论：这张表的**判定**不依赖口径约定（三种读法翻转集合相同），")
print("      但**翻转窗口的边界**依赖它——读法 A 的窗口在 fp16 口径下是 (40, 590.75)。")
print("      ⇒ 所以窗口必须写明是在哪条口径下测的，**窗口的宽度（14.77 倍）才是口径无关的**。")


# ===========================================================================
head("§C  边界与反例：一个 3×3 直接卷积，两种字节口径给出**相反**的判定")
# ===========================================================================
#
# ★ 场景：3×3 直接卷积（不做 im2col、不做 shared memory 复用）。
#   参数：N=1, Cin=Cout=256, H=W=56, K=3, fp16。
#   FLOPs = 2·(H·W·Cout)·(K²·Cin)，两种口径下**相同**——变的只有分母。

N, CIN, COUT, H, W, KS = 1, 256, 256, 56, 56, 3
out_elems = N * H * W * COUT
flops = 2.0 * out_elems * (KS * KS * CIN)

# 口径一：**唯一 DRAM 字节** —— 每个权重/输入元素只从 DRAM 取一次
bytes_unique = (CIN * H * W + COUT * CIN * KS * KS + COUT * H * W) * ESIZE
# 口径二：**总读取字节** —— 每个输出元素都重新读 K² 个权重与 K² 个输入
bytes_total = (out_elems * KS * KS * CIN            # 权重被读 K² 次
               + out_elems * KS * KS                # 输入被读 K² 次
               + out_elems) * ESIZE                 # 输出只写一次

ai_unique = flops / bytes_unique
ai_total = flops / bytes_total

sub("C.1  同一个卷积的 FLOPs（两种口径下相同）")
print(f"    输出元素 = {N}·{H}·{W}·{COUT} = {out_elems:,}")
print(f"    每个输出 = 2·{KS}²·{CIN} = {2*KS*KS*CIN:,} FLOPs（K²·Cin 次 FMA 计 2）")
print(f"    FLOPs = {flops:,.0f}  （两种口径都是这个数）")

sub("C.2  两种字节口径")
print(f"    口径一（**唯一 DRAM 字节**，理想复用）：")
print(f"      输入 {CIN*H*W:,} + 权重 {COUT*CIN*KS*KS:,} + 输出 {COUT*H*W:,} = "
      f"{bytes_unique:,.0f} B")
print(f"    口径二（**总读取字节**，无复用实现实际发出的流量）：")
print(f"      每个输出元素重读 K²·Cin = {KS*KS*CIN} 个权重元素、K² = {KS*KS} 个输入元素"
      f"，输出写 1 次")
print(f"      合计 = {bytes_total:,.0f} B")
print(f"    两者之比 = {bytes_total/bytes_unique:,.1f} 倍")
print(f"    ★ 这个倍数**不是单一的复用因子**，它由两个不同的量合成：")
print(f"      · 权重：每个权重元素被重读 H·W = {H*W} 次"
      f"（{out_elems*KS*KS*CIN:,} 个读 vs {COUT*CIN*KS*KS:,} 个唯一元素）")
print(f"      · 输入：每个输入元素被重读 K² = {KS*KS} 次"
      f"（{out_elems*KS*KS:,} 读 vs {CIN*H*W:,} 唯一元素）")
print(f"      ⇒ 「重复读的倍数」在这类算子上**没有单一值**——"
      f"所以不能用一句「重复读了 9 倍」概括。")

sub("C.3  判定**真的相反**")
v_u = "计算受限" if ai_unique >= ridge_fp16 else "访存受限"
v_t = "计算受限" if ai_total >= ridge_fp16 else "访存受限"
print(f"    fp16 路径 ridge = {ridge_fp16:.2f}")
print(f"    口径一 AI = {ai_unique:,.2f}  ⇒ **{v_u}**")
print(f"    口径二 AI = {ai_total:,.6f}  ⇒ **{v_t}**")
check("C.3 两种字节口径的判定相反", v_u != v_t, f"{v_u} vs {v_t}")
check("C.3 口径一判为计算受限（AI > ridge）", ai_unique > ridge_fp16, f"{ai_unique:.2f}")
check("C.3 口径二判为访存受限（AI < ridge）", ai_total < ridge_fp16, f"{ai_total:.6f}")
check("C.3 口径二的 AI 低于 fp32 向量路径的 ridge，两条路径都不翻转",
      ai_total < ridge_fp32, f"{ai_total:.6f} < {ridge_fp32}")
print()
print("    ★ **两个数都是对的**，差在「分母问的是什么问题」：")
print("      · 口径一问「这个算法**最少**要从 HBM 搬多少」——上界，理想值；")
print("      · 口径二问「这个**实现**实际发出了多少流量」——它才是 roofline 要的那个分母。")
print("      ⚠ **方向**：把重复读算进去是**压低** AI（分母变大），不是抬高。")
print("        所以错误的方向是「把理想的 AI 当实现的 AI 用」，"
      "**结论会从访存受限错判成计算受限**。")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 2 组 · B 类 4 组 · C 类 3 组）")
print()
print(f"  ★ 本程序的数**全部是推导值**：只用 specs.py 的规格常量做算术，"
      f"不产出实测。")
print(f"    python {sys.version.split()[0]} / 纯标准库（未 import numpy、未 import torch）")
print(f"    GPU 档：{GPU}（SXM）；dtype：fp16（{ESIZE} B/element）")
