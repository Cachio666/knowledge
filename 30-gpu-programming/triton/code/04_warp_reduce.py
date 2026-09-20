# -*- coding: utf-8 -*-
"""04 归约的跨 warp 代价：块超过一个 warp 时多出来一个**真实的**相位。

★ 三相分解**不是本模型的发明**，它的官方出处是本文件负责的那条锚点 `specs.ANCHORS["G2"]`
  （A 级，官方文档逐字）：

    「The compiler will generate **butterfly shuffles** to reduce within each warp, then pick a
      **leader warp** to reduce the remaining 4 values per row through **shared memory**.」

  ★ 源码侧的对应实现（编译器里那几个 reduce 相关的 pass / 函数名）**本仓库没有一手核对**，
    按 `specs.NOT_FIRST_HAND` 的纪律**不引用**（`01_block_model.py` 的 `C.6` 扫源码确认它们
    一个都没有出现在本目录的代码里）。

★ 本文件做的三件事：
  (a) **真的按三相跑一遍**（蝶形用 xor 伙伴合并、跨 warp 用一块 numpy 数组当共享内存 + 计数 barrier），
      与 `refs.py`/`naive_reduce` 的独立参照比对 —— 证明这套分解是一个**算法**，不是记账；
  (b) 把三个相位各自的代价算出来，证明它**不是常数**、也**不是块大小的单一函数**；
  (c) 把「跨 warp 相位什么时候出现」钉成一条**可机械判定**的判据：`wpc[axis] > 1`。

★ 四处限定语（计划 §0.7 F10）：迷你实现不是 Triton；5 条规则自造（含三相 ops 计数公式）；
  B 类全是模型值、本主题无实测；不是硬件也不是 Triton 编译器。
"""

import sys

import numpy as np

import checks as C
import minitriton as mt
import refs
import specs

COVERED_ANCHORS = ("G2",)

chk = C.Checks("04_warp_reduce")

print("  解释器 = %s" % sys.executable)
print("  python %s / numpy %s（**无 GPU**；`triton` 包全机器不存在）"
      % (sys.version.split()[0], np.__version__))
print("  ★ 本文件负责的锚点 `G2` 逐字：")
print("      「%s」" % specs.ANCHORS["G2"]["quote"])


def layout_1d(B, W):
    """一维块 [B] 在 W 个 warp 上的**精确切分**布局（lane 全摊在那唯一一维上）。"""
    if B % (32 * W) != 0:
        raise mt.PlanInfeasible("B=%d 不能被 32*W=%d 整除" % (B, 32 * W))
    return mt.Layout([B], [B // (32 * W)], [32], [W], order=[0], dtype="f32")


# ---------------------------------------------------------------------------
# 1. 三相分解是真的算法：真的跑一遍，与两条独立参照逐元素比对
# ---------------------------------------------------------------------------

C.rule("1 三相分解是真的算法：跑一遍，与两条独立参照逐元素比对")

B, W = 256, 4
Lay = layout_1d(B, W)
plan = mt.ReducePlan(Lay, 0)
print("块 [%d]、num_warps=%d → %r" % (B, W, Lay))
print("  每线程持有 %d 个元素；沿归约轴摊开 %d 个线程 = %d lane × %d warp"
      % (Lay.spt[0], Lay.axis_threads(0), Lay.tpw[0], Lay.wpc[0]))
print("  分解：%r" % (plan,))

vals = np.arange(1, B + 1, dtype=np.int64)          # 整数，重结合下也应逐位相等
bc = mt.BarrierCounter()
sc = mt.ShuffleCounter()
got, nbar = mt.simulate_three_phase_reduce(vals, Lay, 0, barrier=bc, shuffle=sc)
ref_sum = int(vals.sum())
ref_naive = int(mt.naive_reduce(vals))
print("三相结果 = %d ｜ numpy.sum = %d ｜ 纯 Python 循环 = %d" % (got, ref_sum, ref_naive))
print("模拟里真的发生的：barrier %d 次，warp 内 xor shuffle %d 次" % (nbar, sc.n))
chk.A(got == ref_sum and got == ref_naive,
      "A.1 三相分解跑出的结果与 numpy.sum、与纯 Python 循环参照**逐位相等**"
      "（整数，无重结合余量）",
      "三相 %d == numpy %d == 标量循环 %d" % (got, ref_sum, ref_naive))
chk.A(nbar == 2 and sc.n == plan.warp_steps * Lay.n_threads * plan.positions_per_thread,
      "A.1b 模拟里 barrier **真的**发生了 %d 次、xor shuffle **真的**发生了 %d 次，"
      "与模型算出来的数逐个相等（`barrier %d`、`shuffle %d`）"
      "—— 三相的 ops 计数公式是**本模型自造的**（`specs.MODEL_OWN_RULES['R4']`），"
      "但相位本身是锚点 `G2` 给的" % (nbar, sc.n, plan.barriers,
                                  plan.warp_steps * Lay.n_threads * plan.positions_per_thread),
      "barrier %d（模型 %d）｜ shuffle %d（模型 %d）"
      % (nbar, plan.barriers, sc.n,
         plan.warp_steps * Lay.n_threads * plan.positions_per_thread))

fv = (np.arange(1, B + 1, dtype=np.float32) * 0.1)      # ★ 不是 0.5：0.5 乘整数在 f32 上**精确**，
f_got, _ = mt.simulate_three_phase_reduce(fv, Lay, 0)   #   重结合误差恒为 0，那样的容差判据是死的
f_ref = float(np.float64(fv.astype(np.float64).sum()))
rel = abs(float(f_got) - f_ref) / abs(f_ref)
tol = refs.rel_err_bound(B)
print("浮点版本（步长 0.1，**不可精确表示**）：三相 = %.6f ｜ float64 参照 = %.6f ｜ "
      "相对误差 %.3e（上界 %.3e）" % (float(f_got), f_ref, rel, tol))
chk.A(rel <= tol and rel > 0.0,
      "A.1c 浮点上三相结果落在**重结合推导出的**误差上界内，且误差**严格大于 0**"
      "（容差 = n×2⁻²³×64，不是拍脑袋）", "%.3e <= %.3e" % (rel, tol))

# A.4：归约后的形状 —— 被归约的那条轴必须消失
out_shape = tuple(np.asarray(got).shape)
chk.A(out_shape == () and plan.positions_per_thread == 1,
      "A.4 一维块归约掉唯一那条轴之后，结果是一个**标量**（形状 ()）；"
      "每个线程恰好持有 %d 个「除 axis 外的位置」" % plan.positions_per_thread,
      "形状 = %r ｜ positions_per_thread = %d" % (out_shape, plan.positions_per_thread))


# ---------------------------------------------------------------------------
# 2. 三个相位各自都是**承重的**：关掉任何一个，结果就错
# ---------------------------------------------------------------------------

C.rule("2 缺陷注入：关掉任意一个相位，结果必须变错")

g_full, _ = mt.simulate_three_phase_reduce(vals, Lay, 0)
g_no3, _ = mt.simulate_three_phase_reduce(vals, Lay, 0, skip=(3,))
g_no23, _ = mt.simulate_three_phase_reduce(vals, Lay, 0, skip=(2, 3))
g_no123, _ = mt.simulate_three_phase_reduce(vals, Lay, 0, skip=(1, 2, 3))
print("完整三相        = %d（对的）" % g_full)
print("关掉跨 warp     = %d（差 %d）" % (g_no3, ref_sum - g_no3))
print("关掉蝶形+跨 warp = %d（差 %d）" % (g_no23, ref_sum - g_no23))
print("三个相位全关     = %d（差 %d）" % (g_no123, ref_sum - g_no123))
chk.C(g_no3 != ref_sum and g_no23 != ref_sum and g_no123 != ref_sum,
      "C.1 三个相位逐一关掉，输出**都**变错 —— 每一个相位都是**承重的**，不是记账",
      "误值 %d / %d / %d，真值 %d" % (g_no3, g_no23, g_no123, ref_sum))

own = Lay.owners()
pred_no3 = int(vals[own < 32].sum())          # 只剩 warp 0 覆盖的那些元素
pred_no23 = int(vals[own == 0].sum())         # 只剩 0 号线程持有的那些元素
print("关掉跨 warp 后的误值 = %d；从布局推出的「warp 0 覆盖的元素之和」= %d"
      % (g_no3, pred_no3))
print("关掉蝶形+跨 warp 后的误值 = %d；从布局推出的「0 号线程持有的元素之和」= %d"
      % (g_no23, pred_no23))
print("  （0 号线程持有全局下标 %r，其值 %r）"
      % (np.flatnonzero(own == 0).tolist(), vals[own == 0].tolist()))
print("  （warp 0 持有全局下标 %d..%d）"
      % (np.flatnonzero(own < 32).min(), np.flatnonzero(own < 32).max()))
chk.C(g_no3 == pred_no3 and g_no23 == pred_no23,
      "C.1b 两个误值**能逐位从布局独立算出来**（warp 0 覆盖的 64 个元素之和 = %d；"
      "0 号线程持有的 2 个元素之和 = %d）——说明关掉的相位负责的**正是这些元素**"
      % (pred_no3, pred_no23),
      "实测 %d == 推得 %d ｜ 实测 %d == 推得 %d" % (g_no3, pred_no3, g_no23, pred_no23))
chk.C(pred_no3 / ref_sum < 0.1,
      "C.1c warp 0 只覆盖全部元素的 %.1f%%，所以关掉跨 warp 相位的误值是真值的 %.1f 分之一"
      "——这个相位的缺失**不是舍入误差级别**的问题" % (100.0 * 64 / B, ref_sum / pred_no3),
      "pred_no3 = %d，真值 = %d" % (pred_no3, ref_sum))

# C.2：★ 把 shuffle 计数器改成「什么都不记」，A.1b 必须失配
_save_xor = mt.ShuffleCounter.xor_round


def no_count(self, n_threads, positions):
    return None


mt.ShuffleCounter.xor_round = no_count
try:
    sc2 = mt.ShuffleCounter()
    mt.simulate_three_phase_reduce(vals, Lay, 0, shuffle=sc2)
    cnt_bad = sc2.n
finally:
    mt.ShuffleCounter.xor_round = _save_xor
chk.C(cnt_bad != sc.n,
      "C.2 把 shuffle 计数器改成「什么都不记」，A.1b 的观测量从 %d 掉到 %d —— "
      "**「蝶形真的发生了」是可证伪的**，不是一句声明" % (sc.n, cnt_bad),
      "改错后 shuffle 计数 = %d（正确 %d）" % (cnt_bad, sc.n))

# C.3：★ 把被测模拟器的**跨 warp 相位**改成「只取 warp 0 的局部和」（= `make falsify` 的 D4）
_save_sim = mt.simulate_three_phase_reduce


def sim_defect(values, layout, axis, barrier=None, shuffle=None, skip=()):
    """与真实现同签名，但**跨 warp 相位**用「只留 warp 0」代替「各 warp 求和」。"""
    a = np.asarray(values if not isinstance(values, mt.Tile) else values.data)
    spt, tpw, wpc = layout.spt[axis], layout.tpw[axis], layout.wpc[axis]
    new_shape = a.shape[:axis] + (wpc, tpw, spt) + a.shape[axis + 1:]
    t = a.reshape(new_shape)
    t = t.sum(axis=axis + 2, keepdims=True)
    t = np.broadcast_to(t, new_shape[:axis] + (wpc, tpw, 1) + a.shape[axis + 1:]).copy()
    for k in range(int(np.log2(tpw)) if tpw > 1 else 0):
        partner = np.arange(tpw) ^ (1 << k)
        t = t + np.take(t, partner, axis=axis + 1)
    t = t.take(0, axis=axis + 1)
    t = t.take(0, axis=axis)                       # ← DEFECT：丢掉其余 warp 的部分和
    return t.reshape(a.shape[:axis] + a.shape[axis + 1:]), 2


mt.simulate_three_phase_reduce = sim_defect
try:
    g_defect, _ = mt.simulate_three_phase_reduce(vals, Lay, 0)
finally:
    mt.simulate_three_phase_reduce = _save_sim
chk.C(g_defect != ref_sum and g_defect == pred_no3,
      "C.3 给被测模拟器注入「跨 warp 只取 warp 0」的缺陷 → A.1 的判据**必然失败**"
      "（得到 %d ≠ 真值 %d，且等于从布局独立算出的 %d）"
      "—— 这与 `make falsify` 的 **D4** 是同一个缺陷" % (g_defect, ref_sum, pred_no3),
      "注入后 %d；A.1 判据是「逐位相等」⇒ 立刻失败")


# ---------------------------------------------------------------------------
# 3. 跨 warp 相位什么时候出现：由 wpc[axis] 决定，不由块大小单独决定
# ---------------------------------------------------------------------------

C.rule("3 跨 warp 相位出现在 wpc[axis] > 1 时 —— 块大小本身不决定它")

print("  B     W    spt  tpw  wpc   threads_along  相位1  相位2(蝶形)  相位3(跨warp)  smem   barrier")
table = []
for Bx in [32, 64, 128, 256, 512, 1024]:
    for Wx in [1, 2, 4, 8]:
        if Bx % (32 * Wx):
            continue
        Lx = layout_1d(Bx, Wx)
        p = mt.ReducePlan(Lx, 0)
        table.append((Bx, Wx, Lx, p))
        print("  %-5d %-4d %-4d %-4d %-4d %-15d %-6d %-12d %-13d %-5d %d"
              % (Bx, Wx, Lx.spt[0], Lx.tpw[0], Lx.wpc[0], Lx.axis_threads(0),
                 p.phase1_ops, p.phase2_ops, p.phase3_ops, p.smem_bytes, p.barriers))

no_phase3 = [t for t in table if t[3].barriers == 0]
with_phase3 = [t for t in table if t[3].barriers > 0]
# ★★ 这里必须小心（第一版是**恒真**的）：如果按 `cross_warp` 分组，再断言
#    「有相位的那些 wpc > 1」，那只是在重复「分组依据本身」——变异测试当场证明它抓不住
#    把 `wpc > 1` 改成 `wpc > 0` 这种改错（`wpc` 最小就是 1，判据恒真）。
#    ⇒ 改法：**按布局参数 `wpc[0]` 分组**，然后断言两件事：
#      ① 「按 wpc 分组」与「按 barrier 分组」给出**同一个划分**（两条独立判据必须一致）；
#      ② 两组各自的 barrier 值就是 0 与 2。
#    这样 `wpc` 判据、`cross_warp` 判据、`barriers` 判据**任意一侧**被改错都会失配。
wpc_gt1 = [t for t in table if t[2].wpc[0] > 1]
wpc_eq1 = [t for t in table if t[2].wpc[0] == 1]
print("按 wpc[0] 分组：wpc>1 的 %d 个、wpc=1 的 %d 个" % (len(wpc_gt1), len(wpc_eq1)))
print("按 barrier 分组：barrier>0 的 %d 个、barrier=0 的 %d 个"
      % (len(with_phase3), len(no_phase3)))
chk.B([t[2].wpc[0] > 1 for t in table] == [t[3].cross_warp for t in table]
      and [t[3].barriers > 0 for t in table] == [t[3].cross_warp for t in table],
      "B.1 三条判据给出**同一个划分**：`wpc[axis] > 1`、`cross_warp`、`barriers > 0` "
      "在全部 %d 个组合上逐项一致 —— ⇒ 跨 warp 相位**恰好**出现在 `wpc[axis] > 1` 的组合上，"
      "**一个不多一个不少**（这是锚点 `G2` 给的机制在模型里的口径）" % len(table),
      "wpc>1 组 %d 个、wpc=1 组 %d 个；三列逐项相等的组合数 = %d"
      % (len(wpc_gt1), len(wpc_eq1),
         sum(1 for t in table if (t[2].wpc[0] > 1) == t[3].cross_warp
             == (t[3].barriers > 0))))
chk.B(all(t[3].barriers == 0 for t in wpc_eq1) and all(t[3].barriers == 2 for t in wpc_gt1),
      "B.1b `wpc = 1` 的 %d 个组合 barrier 一律为 **0**；`wpc > 1` 的 %d 个一律为 **2**"
      "（没有第三个取值）" % (len(wpc_eq1), len(wpc_gt1)),
      "barrier 取值集合 = %r" % (sorted({t[3].barriers for t in table}),))

B32 = mt.ReducePlan(layout_1d(32, 1), 0)
B256_4 = mt.ReducePlan(layout_1d(256, 4), 0)
chk.B(B32.barriers == 0 and B256_4.barriers == 2,
      "B.2 [B=32, W=1] 跨 warp 相位**根本不存在**；[B=256, W=4] 有 2 个 barrier —— "
      "「块内归约是一个常数代价」这个想当然的假设，**在这两个配置之间就已经不成立**",
      "B=32: barrier 0 / smem 0 B ｜ B=256,W=4: barrier 2 / smem %d B" % B256_4.smem_bytes)

r4 = mt.ReducePlan(layout_1d(1024, 4), 0)
r8 = mt.ReducePlan(layout_1d(1024, 8), 0)
print("同样的 B=1024：W=4 → smem %d B / barrier %d ｜ W=8 → smem %d B / barrier %d"
      % (r4.smem_bytes, r4.barriers, r8.smem_bytes, r8.barriers))
chk.B(r8.smem_bytes > r4.smem_bytes and r8.phase2_ops != r4.phase2_ops,
      "B.3 块大小一样、只改 warp 划法，跨 warp 的共享内存字节数就变了（%d B → %d B）——"
      "代价是 **(B, W) 的函数**，不是 B 的函数" % (r4.smem_bytes, r8.smem_bytes),
      "W=4: ops=%d smem=%d B ｜ W=8: ops=%d smem=%d B"
      % (r4.total_ops, r4.smem_bytes, r8.total_ops, r8.smem_bytes))
b_has64 = [t for t in table if t[0] == 64 and t[1] == 2][0]
b_has1024 = [t for t in table if t[0] == 1024 and t[1] == 1][0]
chk.B(b_has64[3].cross_warp and not b_has1024[3].cross_warp,
      "B.3b ★ **[B=64, W=2] 有跨 warp 相位，而 [B=1024, W=1] 没有** —— "
      "块大小差了 16 倍，有相位的反而是**小的那个**。⇒ 「块大就要跨 warp」是错的",
      "B=64,W=2: barrier %d ｜ B=1024,W=1: barrier %d"
      % (b_has64[3].barriers, b_has1024[3].barriers))


# ---------------------------------------------------------------------------
# 4. 代价随块大小增长：不是常数
# ---------------------------------------------------------------------------

C.rule("4 代价随块大小**增长**：固定 W=4，B 从 128 涨到 4096")

seq = [(Bx, mt.ReducePlan(layout_1d(Bx, 4), 0)) for Bx in [128, 256, 512, 1024, 2048, 4096]]
print("  B     total_ops   相位1   相位2(蝶形, 常数)   相位3   smem   barrier   ops/B")
prev = None
mono = True
for Bx, p in seq:
    if prev is not None and p.total_ops <= prev:
        mono = False
    prev = p.total_ops
    print("  %-5d %-11d %-7d %-18d %-6d %-6d %-8d %.2f"
          % (Bx, p.total_ops, p.phase1_ops, p.phase2_ops, p.phase3_ops,
             p.smem_bytes, p.barriers, p.total_ops / Bx))
ratio = seq[-1][1].total_ops / seq[0][1].total_ops
print("B 从 %d 涨到 %d（%d 倍），total_ops 从 %d 涨到 %d（%.2f 倍）"
      % (seq[0][0], seq[-1][0], seq[-1][0] // seq[0][0],
         seq[0][1].total_ops, seq[-1][1].total_ops, ratio))
chk.B(mono and seq[-1][1].total_ops > seq[0][1].total_ops * 5,
      "B.4 total_ops 随块大小**严格单调增长**，%d 倍块大小带来 %.2f 倍代价 —— "
      "「归约的代价与块大小无关」是错的（ops: %d → %d）"
      % (seq[-1][0] // seq[0][0], ratio, seq[0][1].total_ops, seq[-1][1].total_ops),
      "增长全部来自**线程内相位**（%d → %d），不是来自蝶形"
      % (seq[0][1].phase1_ops, seq[-1][1].phase1_ops))
chk.B(all(p.phase2_ops == seq[0][1].phase2_ops for _, p in seq),
      "B.5 但**蝶形相位本身**在 W 固定时是常数（每线程只持 1 个部分和）——"
      "增长来自线程内相位，**不是**来自 shuffle。把「涨了」笼统归给 shuffle 是错的",
      "相位2 恒为 %d ops" % seq[0][1].phase2_ops)
chk.B(seq[-1][1].phase3_smem_bytes == seq[0][1].phase3_smem_bytes,
      "B.6 跨 warp 的**字节数**也与块大小无关（只与 warp 数有关）——"
      "跨 warp 相位「存在与否」是**结构差异**，「字节多少」是 **warp 数**的函数",
      "smem 恒为 %d B" % seq[0][1].phase3_smem_bytes)


# ---------------------------------------------------------------------------
# 5. 落到 softmax 上：行统计量的 barrier 账
# ---------------------------------------------------------------------------

C.rule("5 落到 softmax 上：一行 [BLOCK_N] 的 (max, sum) 要付几个 barrier")

print("  BLOCK_N   W   归约 barrier  广播 barrier  每块合计   两趟合计")
for BN, Wn in [(32, 1), (64, 2), (128, 4), (256, 4), (1024, 8)]:
    Ln = layout_1d(BN, Wn)
    pn = mt.ReducePlan(Ln, 0)
    bcn = pn.broadcast_back()
    per = pn.barriers + bcn.barriers
    print("  %-9d %-4d %-14d %-13d %-11d %d"
          % (BN, Wn, pn.barriers, bcn.barriers, per, per * 2))
row = [(BN, mt.ReducePlan(layout_1d(BN, Wn), 0)) for BN, Wn in [(32, 1), (256, 4)]]
tot32 = row[0][1].barriers + row[0][1].broadcast_back().barriers
tot256 = row[1][1].barriers + row[1][1].broadcast_back().barriers
print("BLOCK_N=32（一个 warp）每个列块 %d 个 barrier；BLOCK_N=256（4 个 warp）%d 个"
      % (tot32, tot256))
chk.B(tot32 == 0 and tot256 == 4,
      "B.7 softmax 的行统计量：BLOCK_N=32 一行 0 个 barrier，BLOCK_N=256 一行 4 个"
      "（归约 2 + 广播回 2）——**同一个 `tl.max` / `tl.sum`，跨过一个 warp 就多出 4 个 barrier**",
      "32 → %d，256 → %d" % (tot32, tot256))


# ---------------------------------------------------------------------------
# 6. 二维情形：同样的 reduce(axis=1)，布局决定有没有跨 warp 相位
# ---------------------------------------------------------------------------

C.rule("6 二维：同一个 reduce(axis=1)，两种布局，barrier 2 vs 0")

SH = (128, 128)
LL = mt.coalesced_layout(SH, 4, order=[1, 0])
LR = mt.coalesced_layout(SH, 4, order=[0, 1])
pL = mt.ReducePlan(LL, 1)
pR = mt.ReducePlan(LR, 1)
print("布局 L   （内维合并）: threads_along=%d → %r" % (LL.axis_threads(1), pL))
print("布局 Lrow（第 0 维合并）: threads_along=%d → %r" % (LR.axis_threads(1), pR))

X = ((np.arange(SH[0] * SH[1]).reshape(SH) % 89) * 0.5).astype(np.float32)
sL, nbL = mt.simulate_three_phase_reduce(X, LL, 1)
sR, nbR = mt.simulate_three_phase_reduce(X, LR, 1)
ref_rows = X.astype(np.float64).sum(axis=1)
eL = float(np.max(np.abs(sL - ref_rows)))
eR = float(np.max(np.abs(sR - ref_rows)))
print("三相模拟在 L 上的逐行和误差 = %.3e（barrier %d）" % (eL, nbL))
print("三相模拟在 Lrow 上的逐行和误差 = %.3e（barrier %d）" % (eR, nbR))
chk.A(eL <= refs.rel_err_bound(SH[1]) and eR <= refs.rel_err_bound(SH[1]),
      "A.2 两种布局下三相模拟都算出了正确的逐行和（%d 行全比）" % SH[0],
      "max_abs_err: L %.3e ｜ Lrow %.3e（上界 %.3e）"
      % (eL, eR, refs.rel_err_bound(SH[1])))
chk.A(nbL == 2 and nbR == 0,
      "A.2b **同一个函数的同一次调用**，在 L 上真的发生了 2 次 barrier，在 Lrow 上 0 次 ——"
      "这不是估计，是模拟里数出来的",
      "barrier: L %d ｜ Lrow %d" % (nbL, nbR))

# A.6：★ 锚点 G2 里的那个 **4** 是**那个例子**的位数，不是常数
wcps = sorted({t[2].wpc[0] for t in table})
print("本模型 18 组配置里出现过的 wpc 值：%r" % (wcps,))
chk.A("remaining 4 values" in specs.ANCHORS["G2"]["quote"] and len(wcps) >= 3,
      "A.6 锚点 `G2` 原句里的 **4** 是**那个例子**（num_warps=4）的位数，不是常数 —— "
      "本模型的 18 组配置里 wpc 取过 %r 三个值。把「4」读成「跨 warp 相位总是 4 个值」"
      "是**作用域误读**（P7）" % (wcps,),
      "wpc 取值 = %r；锚点原文含 `remaining 4 values`" % (wcps,))


# ---------------------------------------------------------------------------
# 7. C 类：把「常数代价」这个假设钉死为假
# ---------------------------------------------------------------------------

C.rule("7 C 类：「块内归约是常数代价」在两个配置之间就已经不成立")

cfgs = [(32, 1), (64, 2), (128, 4), (256, 4), (1024, 8)]
bar_set = sorted({mt.ReducePlan(layout_1d(b, w), 0).barriers for b, w in cfgs})
smem_set = sorted({mt.ReducePlan(layout_1d(b, w), 0).smem_bytes for b, w in cfgs})
print("五个配置的 (barrier, smem) 取值集合：barrier ∈ %r ｜ smem ∈ %r" % (bar_set, smem_set))
chk.C(len(bar_set) > 1 and len(smem_set) > 1,
      "C.4 五个配置给出**不止一个** (barrier, smem) 取值 —— "
      "「块内归约是一个常数代价」这条假设，**不需要跑到真实硬件上就能证伪**"
      "（它在模型里就已经是假的）",
      "barrier 取值 %r ｜ smem 取值 %r" % (bar_set, smem_set))

# C.5：★ 容差判据是**活的**（不是装饰）—— 实测误差严格大于 0，两侧都验
tight = rel / 2.0
chk.C(rel > 0.0 and not (rel <= tight) and (rel <= tol),
      "C.5 实测相对误差 %.6e **严格大于 0**，所以 A.1c 的判据是**活的**："
      "把上界收紧到它的一半（%.3e）立刻失败，用正确上界（%.3e = n×eps×64）通过"
      "—— 容差不是装饰" % (rel, tight, tol),
      "rel=%.6e ｜ 收紧后上界=%.3e ⇒ 失败 ｜ 正确上界=%.3e ⇒ 通过" % (rel, tight, tol))

# C.6：关掉相位 1 也必须错（三个相位里最容易被当成「无代价」的那个）
g_no1, _ = mt.simulate_three_phase_reduce(vals, Lay, 0, skip=(1,))
chk.C(g_no1 != ref_sum,
      "C.6 连**线程内相位**（最容易被当成「反正本来就要算」的那一步）关掉也会错："
      "得到 %d ≠ 真值 %d" % (g_no1, ref_sum),
      "线程内相位负责的是每个线程寄存器里那 %d 个元素的合并" % Lay.spt[0])

# A.7/A.8：换一组 (B, W) 再验一遍 —— 三相结果仍然精确、shuffle 计数仍然等于模型
B2, W2 = 512, 8
Lay2 = layout_1d(B2, W2)
plan2 = mt.ReducePlan(Lay2, 0)
vals2 = np.arange(1, B2 + 1, dtype=np.int64)
sc2b = mt.ShuffleCounter()
got2, nbar2 = mt.simulate_three_phase_reduce(vals2, Lay2, 0, shuffle=sc2b)
chk.A(int(got2) == int(vals2.sum()) == int(mt.naive_reduce(vals2)),
      "A.7 换一组配置（B=%d, W=%d）三相结果仍与 numpy 和纯 Python 循环**逐位相等**"
      % (B2, W2), "三相 %d == numpy %d == 标量 %d"
      % (int(got2), int(vals2.sum()), int(mt.naive_reduce(vals2))))
chk.A(sc2b.n == plan2.warp_steps * Lay2.n_threads * plan2.positions_per_thread,
      "A.8 shuffle 计数公式（本模型自造）在第二组配置上也成立："
      "实测 %d == 模型 log2(tpw)×线程数×每线程位置数 = %d"
      % (sc2b.n, plan2.warp_steps * Lay2.n_threads * plan2.positions_per_thread),
      "warp_steps=%d ｜ n_threads=%d ｜ positions=%d"
      % (plan2.warp_steps, Lay2.n_threads, plan2.positions_per_thread))

# C.7：**只**关掉蝶形相位（保留线程内与跨 warp）也必须错
g_no2, _ = mt.simulate_three_phase_reduce(vals, Lay, 0, skip=(2,))
chk.C(g_no2 != ref_sum,
      "C.7 只关掉**蝶形**相位（保留线程内与跨 warp）也会错：得到 %d ≠ 真值 %d —— "
      "蝶形不是「为了快才做的装饰」，它是**唯一**能把同一 warp 内不同 lane 的部分和合并起来的一步"
      % (g_no2, ref_sum),
      "误值 = %d（真值 %d，差 %d）" % (g_no2, ref_sum, ref_sum - g_no2))

# A.9：`positions_per_thread` 与布局参数的关系（本模型的自定语义，可机械复核）
chk.A(plan.positions_per_thread == Lay.numel // (Lay.n_threads * Lay.spt[0]) == 1
      and plan2.positions_per_thread == Lay2.numel // (Lay2.n_threads * Lay2.spt[0]) == 1,
      "A.9 「每线程的位置数」= numel ÷ (线程数 × spt) 在两个配置上都等于 1 —— "
      "被归约的那条轴**全部摊给线程**时，每个线程只留一个部分和（这是三相计数公式的前提，"
      "公式本身是自造的：`specs.MODEL_OWN_RULES['R4']`）",
      "B=%d: %d ｜ B=%d: %d" % (B, plan.positions_per_thread, B2, plan2.positions_per_thread))

sys.exit(chk.summary())
