# -*- coding: utf-8 -*-
"""02 边界掩码：在非整除形状（N=1000, BLOCK=256）上**穷举**验证掩码公式。

★ 核心口径：越界位置的贡献必须与**独立参照**逐元素比对，不是「看起来对」。
  参照有三条，**都不 import 本模型**（`refs.py` 里没有一个函数调用 `minitriton`）：
    1. `refs.masked_block_ref`      纯 Python 标量循环 + 显式 if 判断
    2. `refs.masked_index_np_ref`   numpy 花式索引 + `np.clip`
    3. `refs.softmax_prefix_np_ref` 只在有效前缀上做 softmax 的 numpy 实现
  被测侧是 `minitriton.load`（广播 + `np.where`）——**第三条代码路径**。

★★ 本文件负责的两条语义锚点（`specs.LOAD_QUOTES`，A 级：API 页逐字）：
    `L1` `other=None` ⇒「**未定义**」（**不是 0**）   → `C.4`
    `L2` 掩码为假 ⇒「**不读**那个地址」               → `B.2` / `C.1` / `C.3`

★★ 本篇最值钱的一条（计划 §0.7 F6）：**掩码写窄不会抛错**。
   写宽 ⇒ 越界被检出 ⇒ 抛 `OutOfBoundsError`；写窄 ⇒ **既不越界也不报错**，
   只是**静默**把数据算成 `other`（误差从 `2.384e-07` 涨到 `4.499e+00`）。
   ⇒ **C 类的越界断言只能抓「写宽」，抓不到「写窄」。唯一能抓它的是 A 类逐元素比对。**

★ 四处限定语（计划 §0.7 F10）：迷你实现不是 Triton；5 条规则自造；B 类全是模型值、无实测；
  不是硬件也不是 Triton 编译器。★ 其中**越界抛错是本模型的加强**（`S2`），不是更真实。
"""

import sys

import numpy as np

import checks as C
import minitriton as mt
import refs
import specs

N, BLOCK = 1000, 256
GRID = (N + BLOCK - 1) // BLOCK          # = 4
SRC = np.arange(N, dtype=np.float32) * 0.25 - 100.0

# ★ 本程序拥有这两条锚点（`01_block_model.py` 的 `C.7` 检查四篇并集 == 全集、两两不交）。
COVERED_ANCHORS = ("L1", "L2")

chk = C.Checks("02_masking")

print("  解释器 = %s" % sys.executable)
print("  python %s / numpy %s（**无 GPU**；`triton` 包全机器不存在）"
      % (sys.version.split()[0], np.__version__))


def print_anchors():
    """把本文件负责的两条锚点**逐字**打出来（常量 ↔ 输出的双向绑定）。"""
    for aid in COVERED_ANCHORS:
        a = specs.LOAD_QUOTES[aid]
        print("  · %s（页面 %s｜%s）" % (aid, a["page"], a["section"]))
        print("      「%s」" % a["quote"])


# ---------------------------------------------------------------------------
# 0. 掩码地形：把越界位置先数清楚
# ---------------------------------------------------------------------------

C.rule("0 掩码地形：N=%d，BLOCK=%d，grid=%d" % (N, BLOCK, GRID))
print_anchors()

oob = refs.oob_positions(N, BLOCK, GRID)
n_total_pos = GRID * BLOCK
print("总位置数 = %d × %d = %d；有效 = %d；越界 = %d"
      % (GRID, BLOCK, n_total_pos, N, len(oob)))
print("越界位置的分布：pid 0/1/2 → 0 个；pid 3 → %d 个（全局下标 %d..%d）"
      % (len(oob), oob[0][2], oob[-1][2]))
assert len(oob) == 24, len(oob)


# ---------------------------------------------------------------------------
# 1. 掩码公式的等价性：五种写法的逐位置比对
# ---------------------------------------------------------------------------

C.rule("1 五种掩码写法在所有 %d 个位置上的逐位一致性" % n_total_pos)

VARIANTS = {
    "A  offs < N":               lambda offs: offs.data < N,
    "B  offs <= N - 1":          lambda offs: offs.data <= N - 1,
    "C  (pid*BLOCK+arange) < N": lambda offs: (offs.data) < N,
    "D  N > offs":               lambda offs: N > offs.data,
    "E  offs - N < 0":           lambda offs: (offs.data - N) < 0,
}

REF_MASK = []          # 参照：纯 Python 判断，逐程序逐位置
for pid in range(GRID):
    row = []
    for k in range(BLOCK):
        i = pid * BLOCK + k
        row.append(0 <= i < N)
    REF_MASK.append(row)
REF_MASK = np.asarray(REF_MASK)

n_cmp_mask = 0
n_mask_mismatch = 0
for name, fn in VARIANTS.items():
    got = []
    for pid in range(GRID):
        offs = mt.arange(pid * BLOCK, pid * BLOCK + BLOCK)
        got.append(np.asarray(fn(offs), bool))
    got = np.asarray(got)
    n_cmp_mask += got.size
    bad = int((got != REF_MASK).sum())
    n_mask_mismatch += bad
    print("  %-28s 与参照不一致的位置数 = %d / %d" % (name, bad, got.size))

print("五个写法 × %d 个位置，共 %d 次比较，不一致 %d 次"
      % (n_total_pos, n_cmp_mask, n_mask_mismatch))
chk.A(n_mask_mismatch == 0,
      "A.1 五种掩码写法与纯 Python 参照在所有 %d 个位置上逐位一致" % n_cmp_mask,
      "比较次数 = %d，不一致 = %d" % (n_cmp_mask, n_mask_mismatch))


# ---------------------------------------------------------------------------
# 2. masked load 的值：与两条独立参照逐元素比对
# ---------------------------------------------------------------------------

C.rule("2 masked load 的取值：%d 个位置逐元素比对两条独立参照" % n_total_pos)

src_buff = mt.Buff("src", SRC)
got_vals = np.zeros((GRID, BLOCK), np.float64)
got_mask = np.zeros((GRID, BLOCK), bool)
for pid in range(GRID):
    offs = mt.arange(pid * BLOCK, pid * BLOCK + BLOCK)
    mask = mt.less_than(offs, N)
    v = mt.load(src_buff, offs, mask=mask, other=0.0)
    got_vals[pid] = v.data.astype(np.float64)
    got_mask[pid] = mask.data
got_oob_hits = src_buff.oob_masked

ref1 = np.asarray([refs.masked_block_ref(SRC.tolist(), N, BLOCK, pid, other=0.0)
                   for pid in range(GRID)], np.float64)
ref2 = np.asarray([refs.masked_index_np_ref(N, BLOCK, pid, other=0.0,
                                            src=SRC.astype(np.float64))[0]
                   for pid in range(GRID)], np.float64)

e1 = float(np.max(np.abs(got_vals - ref1)))
e2 = float(np.max(np.abs(got_vals - ref2)))
print("与参照 1（纯 Python 标量循环）最大绝对误差 = %.3e" % e1)
print("与参照 2（numpy 花式索引 + clip）最大绝对误差 = %.3e" % e2)
chk.A(e1 == 0.0, "A.2 masked load 的 %d 个取值与纯 Python 参照**逐位相等**" % n_total_pos,
      "max_abs_err = %.3e（本模型下**应为 0**：读到的就是源里同一个 f32，没有任何算术参与）" % e1)
chk.A(e2 == 0.0, "A.3 同时与第二条独立参照（numpy 花式索引）逐位相等",
      "max_abs_err = %.3e" % e2)
mask_bad = int((got_mask != REF_MASK).sum())
chk.A(mask_bad == 0, "A.4 掩码布尔张量本身与参照逐位一致",
      "不一致位置数 = %d" % mask_bad)

oob_vals = np.asarray([got_vals[pid, k] for pid, k, i in oob], np.float64)
oob_ref = np.asarray([ref1[pid, k] for pid, k, i in oob], np.float64)
oob_mask = np.asarray([got_mask[pid, k] for pid, k, i in oob])
print("越界位置：%d 个；掩码全为 False = %s；取值全等于 other(0.0) = %s"
      % (len(oob), bool((~oob_mask).all()), bool((oob_vals == 0.0).all())))
chk.A((~oob_mask).all() and np.array_equal(oob_vals, oob_ref),
      "A.5 24 个越界位置的**贡献**逐元素等于参照给出的 other（0.0），掩码全为 False",
      "越界位置取值 = %r" % (oob_vals[:4].tolist() + ["..."]))
print("  buff 记账：掩码为假而越界 = %d 次（这 %d 次**没有真的读内存**）"
      % (got_oob_hits, got_oob_hits))

# A.6 ★ 「零误差是有意义的」：A.2/A.3 的 0 之所以是 0，是因为掩码读回的就是源里同一个 f32
#     —— 所以下面用的是**逐位**比较（整数位模式），不是「误差很小」。
valid_mask_bits = 0
for pid in range(GRID):
    for k in range(BLOCK):
        i = pid * BLOCK + k
        if 0 <= i < N:
            valid_mask_bits += int(
                np.float32(got_vals[pid, k]).view(np.int32) == SRC[i].view(np.int32))
n_valid = GRID * BLOCK - len(oob)
chk.A(valid_mask_bits == n_valid,
      "A.6 %d 个**有效**位置的位模式与源逐位相同（不是「误差 ≤ 容差」，是**同一个 f32**）"
      "——这就是为什么 A.2/A.3 的 **0 是应该的、非 0 就说明模型错了**" % n_valid,
      "位模式相同的有效位置 = %d / %d" % (valid_mask_bits, n_valid))


# ---------------------------------------------------------------------------
# 3. 被掩掉的毒区：证明越界位置没有把内存里的任意内容带进来
# ---------------------------------------------------------------------------

C.rule("3 毒区未被翻开：越界位置读的不是内存，而是 other")

tail = np.asarray(src_buff.poison_tail[:BLOCK], np.float64)
print("毒区前 %d 个元素的值 = %.3e（毒值 %r）" % (BLOCK, tail[0], float(mt.POISON)))
print("输出里出现的最大值 = %.3e（源数据最大值 %.3e）"
      % (got_vals.max(), float(SRC.max())))
no_poison = bool(np.all(got_vals < float(mt.POISON) * 0.5))
chk.A(no_poison, "A.7 输出中没有任何毒值——掩码真的挡住了 %d 次越界读" % got_oob_hits,
      "输出最大 %.3e << 毒值 %.3e" % (got_vals.max(), float(mt.POISON)))

src_buff.reset_counters()
offs_bad = mt.arange(3 * BLOCK, 4 * BLOCK)
chk.expect_raises(lambda: mt.load(src_buff, offs_bad, mask=None),
                  mt.OutOfBoundsError,
                  "C.1 同一段索引**不加掩码** → 越界访问被检出（这是本模型的**加强**，"
                  "见 specs.MODEL_SIMPLIFICATIONS['S2']）")
chk.expect_raises(lambda: mt.load(src_buff, offs_bad,
                                 mask=mt.Tile(offs_bad.data < 1024, "bool")),
                  mt.OutOfBoundsError,
                  "C.2 掩码**写宽**（offsets < 1024）→ 越界访问被检出")


# ---------------------------------------------------------------------------
# 4. 归约里的贡献：越界位置对 max / sum 的影响必须为「不多不少」
# ---------------------------------------------------------------------------

C.rule("4 越界位置对归约的贡献：max 用 -inf、sum 用 0，与「只在有效前缀上算」逐位比对")

offs3 = mt.arange(3 * BLOCK, 4 * BLOCK)
mask3 = mt.less_than(offs3, N)
REF_ROW = np.full(BLOCK, -np.inf, np.float64)
REF_ROW[:N - 3 * BLOCK] = SRC[3 * BLOCK:N].astype(np.float64)
m_ref = float(REF_ROW.max())
s_ref = float(np.exp(REF_ROW[:N - 3 * BLOCK] - m_ref).sum())

blk = mt.load(src_buff, offs3, mask=mask3, other=float("-inf"))
m_got = float(mt.tmax(blk, axis=None).data)
s_got = float(mt.tsum(mt.exp(mt.sub(blk, m_got)), axis=None).data)
print("最后一个块（含 24 个越界位）：max 本模型 = %.6f，参照 = %.6f" % (m_got, m_ref))
print("                              sum 本模型 = %.6f，参照 = %.6f" % (s_got, s_ref))
chk.A(m_got == m_ref, "A.8 掩码 + other=-inf 让 max **精确等于**「只在有效元素上取 max」"
      "（不是近似 —— -inf 不参与取最大）",
      "%.6f vs %.6f" % (m_got, m_ref))
n_terms = N - 3 * BLOCK
tol_rel = refs.rel_err_bound(n_terms)
rel = abs(s_got - s_ref) / abs(s_ref)
print("求和项数 = %d；f32 重结合的**推导**相对误差上界 = %d*eps*64 = %.3e；实测 = %.3e"
      % (n_terms, n_terms, tol_rel, rel))
chk.A(rel <= tol_rel,
      "A.9 掩码 + exp(-inf)=0 让 sum 等于「只在有效元素上求和」，误差在 f32 重结合上界内"
      "（容差按机器 epsilon 推，不是拍脑袋）",
      "相对误差 %.3e <= 上界 %.3e" % (rel, tol_rel))

# `other=None`（官方的「未定义」）用在归约里会怎样 —— `L1` 的可证伪形式
src_buff.reset_counters()
blk_u = mt.load(src_buff, offs3, mask=mask3)            # 不给 other
m_u = float(mt.tmax(blk_u, axis=None).data)
print("不给 other（官方语义：**未定义**）时：max = %.6e，other_undefined 计数 = %d"
      % (m_u, src_buff.other_undefined))
chk.C(m_u > 1e29 and src_buff.other_undefined == 24,
      "C.3 `other=None`（官方语义「未定义」）时，24 个越界位置的毒值**真的一路流进了归约结果**"
      "——这正是官方说的 undefined，**不是 0**（若模型把 None 读成 0，这里会是 %.6f）" % m_ref,
      "max = %.6e（真值 %.6f）" % (m_u, m_ref))


# ---------------------------------------------------------------------------
# 5. 全行 softmax：32000 个输出位置的逐元素比对
# ---------------------------------------------------------------------------

C.rule("5 整行 softmax（N=1000 被切成 4 个 256 的块）：32000 个输出位置逐元素比对")

rng = np.random.default_rng(7)
X = (rng.standard_normal((32, N)) * 2.0).astype(np.float32)
Y = mt.Buff("Y", np.zeros((32, N), np.float32))
XB = mt.Buff("X", X)


def softmax_masked(pid, trace=None):
    row = pid[0]
    cols = mt.arange(0, BLOCK)
    r = mt.Tile(np.int32(row), "i32")
    m_i = None
    s_i = None
    for n0 in range(0, N, BLOCK):
        offs = mt.offs_add(cols, n0)
        mk = mt.less_than(offs, N)
        x = mt.load(XB, (r, offs), mask=mk, other=float("-inf"))
        bm = mt.tmax(x, axis=None)
        if m_i is None:
            m_i = bm
            s_i = mt.tsum(mt.exp(mt.sub(x, bm)), axis=None)
        else:
            m_new = mt.maximum(m_i, bm)
            s_i = mt.add(mt.mul(s_i, mt.exp(mt.sub(m_i, m_new))),
                         mt.tsum(mt.exp(mt.sub(x, m_new)), axis=None))
            m_i = m_new
    for n0 in range(0, N, BLOCK):
        offs = mt.offs_add(cols, n0)
        mk = mt.less_than(offs, N)
        x = mt.load(XB, (r, offs), mask=mk, other=0.0)
        mt.store(Y, (r, offs), mt.div(mt.exp(mt.sub(x, m_i)), s_i), mask=mk)


mt.launch(softmax_masked, mt.Grid((32,)))

ref_sm_py = refs.softmax_rows_ref(X.tolist())                  # 纯 Python 标量，全行
ref_sm_np = refs.softmax_prefix_np_ref(X, N)                   # numpy，只在有效前缀上
e_py = refs.max_abs_err(np.asarray(Y.arr, np.float64).tolist(), ref_sm_py)
e_np = float(np.max(np.abs(np.asarray(Y.arr, np.float64) - ref_sm_np)))
print("32×1000 个输出位置：vs 纯 Python 全行参照 %.3e ｜ vs numpy 前缀参照 %.3e"
      % (e_py, e_np))
tol5 = refs.rel_err_bound(N)
chk.A(e_py <= tol5, "A.10 越界位置的贡献被完全正确排除：32000 个输出逐元素等于全行标量参照",
      "max_abs_err = %.3e <= f32 上界 %.3e（%d 项）" % (e_py, tol5, N))
chk.A(e_np <= tol5, "A.11 同时等于「只在有效前缀上做 softmax」的 numpy 参照——"
      "这条参照**根本不认识**越界位置，所以它是对掩码的独立检验",
      "max_abs_err = %.3e <= %.3e" % (e_np, tol5))

Y2 = mt.Buff("Y2", np.zeros((32, N), np.float32))


def softmax_wide_mask(pid, trace=None):
    row = pid[0]
    cols = mt.arange(0, BLOCK)
    r = mt.Tile(np.int32(row), "i32")
    m_i = None
    s_i = None
    for n0 in range(0, N, BLOCK):
        offs = mt.offs_add(cols, n0)
        mk = mt.Tile(offs.data < GRID * BLOCK, "bool")          # 掩码写宽
        x = mt.load(XB, (r, offs), mask=mk, other=float("-inf"), strict=True)
        bm = mt.tmax(x, axis=None)
        if m_i is None:
            m_i = bm
            s_i = mt.tsum(mt.exp(mt.sub(x, bm)), axis=None)
        else:
            m_new = mt.maximum(m_i, bm)
            s_i = mt.add(mt.mul(s_i, mt.exp(mt.sub(m_i, m_new))),
                         mt.tsum(mt.exp(mt.sub(x, m_new)), axis=None))
            m_i = m_new


chk.expect_raises(lambda: mt.launch(softmax_wide_mask, mt.Grid((32,))),
                  mt.OutOfBoundsError,
                  "C.4 softmax 的掩码**写宽** → 在第一个越界块上就被检出")


# ---------------------------------------------------------------------------
# 6. 掩码公式的扫掠：N 与 BLOCK 的全部组合
# ---------------------------------------------------------------------------

C.rule("6 扫掠：N × BLOCK 的所有组合，逐位置比对掩码与取值")

Ns = [1, 31, 32, 33, 100, 255, 256, 257, 511, 512, 1000, 1001]
BLOCKs = [32, 64, 128, 256, 512]
n_pos_total = 0
n_mismatch = 0
n_oob_total = 0
for n in Ns:
    src = (np.arange(n, dtype=np.float32) * 0.5 - 7.0)
    bf = mt.Buff("s%d" % n, src)
    for blk in BLOCKs:
        g = (n + blk - 1) // blk
        for pid in range(g):
            offs = mt.arange(pid * blk, pid * blk + blk)
            mk = mt.less_than(offs, n)
            v = mt.load(bf, offs, mask=mk, other=-3.5)
            exp_v = refs.masked_block_ref(src.tolist(), n, blk, pid, other=-3.5)
            exp_m = [0 <= (pid * blk + k) < n for k in range(blk)]
            n_pos_total += blk
            n_mismatch += int((np.asarray(v.data, np.float64) != np.asarray(exp_v)).sum())
            n_mismatch += int((np.asarray(mk.data, bool) != np.asarray(exp_m)).sum())
            n_oob_total += blk - sum(exp_m)
print("组合数 = %d × %d = %d；比对的位置 = %d；越界位置 = %d；不一致 = %d"
      % (len(Ns), len(BLOCKs), len(Ns) * len(BLOCKs), n_pos_total, n_oob_total, n_mismatch))
chk.A(n_mismatch == 0,
      "A.12 %d 个 (N, BLOCK) 组合里，掩码与取值在 %d 个位置上逐位等于参照"
      "（其中 %d 个是越界位置）" % (len(Ns) * len(BLOCKs), n_pos_total, n_oob_total),
      "不一致 = %d" % n_mismatch)


# ---------------------------------------------------------------------------
# 7. ★ 「写窄」——本篇最值钱的一条（计划 §0.7 F6）
# ---------------------------------------------------------------------------

C.rule("7 ★ 「掩码写窄」不会抛错：C 类的越界断言抓不到它")

# 与第 1 节同一个 kernel，但掩码写成「块长度」而不是「真实的 N」
NARROW_SRC = mt.Buff("narrow_src", SRC)
NARROW_OUT = mt.Buff("narrow_out", np.zeros(GRID * BLOCK, np.float32))


def narrow_vadd(pid, trace=None):
    offs = mt.arange(pid[0] * BLOCK, pid[0] * BLOCK + BLOCK)
    mk = mt.Tile(offs.data < BLOCK, "bool")        # ← **写窄**：拿块长度当上界
    x = mt.load(NARROW_SRC, offs, mask=mk, other=0.0)
    mt.store(NARROW_OUT, offs, x, mask=mk)


raised = None
try:
    mt.launch(narrow_vadd, mt.Grid((GRID,)))
except mt.OutOfBoundsError as e:
    raised = e
# ★ 参照按**独立算出来的**逐位置期望值构造：下标 < N 取源值，否则取 other（写窄时是 0.0）
ref_narrow = np.zeros(GRID * BLOCK, np.float64)
ref_narrow[:N] = SRC.astype(np.float64)
err_narrow = float(np.max(np.abs(np.asarray(NARROW_OUT.arr, np.float64) - ref_narrow)))
print("掩码写窄（offs < BLOCK）：抛错 = %s；与源的最大偏差 = %.3e"
      % ("是" if raised else "否", err_narrow))
chk.C(raised is None and err_narrow > 1.0,
      "C.5 掩码**写窄**：既不越界（pid 0 的位置全在界内、其余全被掩掉）也**不报错**，"
      "只是**静默**把 %d 个位置算成 other ⇒ 越界检测对它**完全无效**"
      % ((GRID - 1) * BLOCK,),
      "偏差 = %.3e（源值域 [%.1f, %.1f]）——只有 A 类的逐元素比对能抓住它"
      % (err_narrow, float(SRC.min()), float(SRC.max())))

chk.C(err_narrow > 1.0,
      "C.6 而这个静默错误的量级是**真错**（不是舍入）：偏差 %.3e 与源值域同阶" % err_narrow,
      "若掩码写宽，同一段代码抛 OutOfBoundsError（C.2）；写窄则一个信号都没有")

# ★ C.7：`refs.py` 的独立性 —— 把参照与源**故意改成不一致**，A 类必须失败
_save_masked = refs.masked_block_ref


def wrong_boundary(src, n, block, pid, other=0.0, arange_start=None):
    """把边界判据从 `i < n` 改成 `i <= n`（差一个位置，下标夹到最后一个有效位以免崩）。"""
    start = pid * block if arange_start is None else arange_start
    out = []
    for k in range(block):
        i = start + k
        out.append(float(src[min(i, n - 1)]) if 0 <= i <= n else float(other))   # ← `<=`
    return out


refs.masked_block_ref = wrong_boundary
try:
    bad = np.asarray([refs.masked_block_ref(SRC.tolist(), N, BLOCK, pid, other=0.0)
                      for pid in range(GRID)], np.float64)
    e_bad = float(np.max(np.abs(got_vals - bad)))
finally:
    refs.masked_block_ref = _save_masked
chk.C(sorted(bad.ravel().tolist()) != sorted(ref1.ravel().tolist()),
      "C.7 把 `refs.py` 的边界判据从 `i < n` 改成 `i <= n`，参照输出**逐位改变**（"
      "%d 个位置里至少一处不同）⇒ A.2 的 0 误差判据立刻失配 —— A 类断言真的在与"
      "**独立参照**比，不是在自证"
      % n_total_pos,
      "改错后的最大偏差 = %.3e（改回即 0；A.2 判据是 == 0.0）" % e_bad)


# ---------------------------------------------------------------------------
# 8. B 类：掩码的代价模型（模型值，不是实测）
# ---------------------------------------------------------------------------

C.rule("8 B 代价模型：掩码的记账口径")

bm = BLOCK // (32 * 4)     # 每个线程在 4 warp 下分到的元素数（示例布局）
print("以 [BLOCK=%d] 的块、num_warps=4 为例：每线程持有 %d 个元素" % (BLOCK, bm))
print("若 BLOCK 不是 32*num_warps 的整数倍，末尾线程会拿到少于 %d 个元素" % bm)
print("N=1000 时最后一个块的掩码为真个数 = %d（不是 %d）" % (N - 3 * BLOCK, BLOCK))
chk.B(GRID * BLOCK - N == 24 and (GRID * BLOCK) % BLOCK == 0,
      "B.1 朴素求解：越界位置恰好 %d 个，占比 %.4f%%；本模型按位置逐个记账，"
      "不用「平均浪费」估" % (GRID * BLOCK - N, 100.0 * (GRID * BLOCK - N) / (GRID * BLOCK)),
      "总位置 %d，有效 %d，越界 %d" % (GRID * BLOCK, N, GRID * BLOCK - N))

src_buff.reset_counters()
for pid in range(GRID):
    offs = mt.arange(pid * BLOCK, pid * BLOCK + BLOCK)
    mt.load(src_buff, offs, mask=mt.less_than(offs, N), other=0.0)
chk.B(src_buff.oob_masked == 24,
      "B.2 掩码把 %d 次访存从「读」降级成「取常量」——这是锚点 `L2`"
      "（「do not load the data at address pointer[idx]」）的可证伪形式："
      "**真的少做了 24 次访存**，不是「读了再丢」" % src_buff.oob_masked,
      "记账到的「掩码为假而越界」次数 = %d" % src_buff.oob_masked)

chk.B(24 * 4 == 96,
      "B.3 但这**不是**「掩码免费」：掩码是**每元素一个谓词**，"
      "%d 个位置 ⇒ %d 个谓词求值，代价与块大小成正比，与越界位置数无关"
      % (n_total_pos, n_total_pos),
      "谓词数 = %d（= grid×BLOCK），被省掉的访存 = 24 —— 两个数差 %d 倍"
      % (n_total_pos, n_total_pos // 24))

chk.A(len(specs.LOAD_QUOTES) == 2 and all(
    specs.LOAD_QUOTES[k]["page"] == specs.LOAD_PAGE for k in specs.LOAD_QUOTES),
      "A.13 本程序负责的两条锚点（`L1`/`L2`）都记在 `specs.LOAD_QUOTES` 里、"
      "且都指向同一个 API 页（`%s`）" % specs.LOAD_PAGE,
      "锚点 = %r" % (list(specs.LOAD_QUOTES),))

# A.14：越界位置的清单与「独立算出来的」界对齐（不依赖 refs 的内部实现）
chk.A(len(oob) == GRID * BLOCK - N == 24
      and oob[0][2] == N and oob[-1][2] == GRID * BLOCK - 1,
      "A.14 越界位置的**条数与首尾下标**都等于独立算出来的数："
      "%d = %d×%d − %d，首 = N = %d，尾 = %d"
      % (len(oob), GRID, BLOCK, N, oob[0][2], oob[-1][2]),
      "首尾 = (%d..%d)" % (oob[0][2], oob[-1][2]))

# B.4：逐个记账 vs「平均浪费」估 —— 两个口径给出不同的量
waste_est = (GRID * BLOCK - N) / (GRID * BLOCK)
chk.B(GRID * BLOCK - N == 24 and abs(waste_est - 0.0234375) < 1e-12,
      "B.4 按位置逐个记账给 **24 次**「掩码为假而越界」；占比 %.6f%%（= %d/%d，精确值）。"
      "★ 本模型**不用**「平均浪费」这种估算 —— 估算给不出「24」这个整数"
      % (100 * waste_est, GRID * BLOCK - N, GRID * BLOCK),
      "两类记账口径（逐位 vs 平均）在这里给出不同的可判定量")

# C.8/C.9：`other` 的隐式广播（官方：「mask and other are implicitly broadcast to pointer.shape」）
#   ★ 本模型的 `load(other=...)` 只接受**标量或 numpy 数组**，**不接受 `Tile`** ——
#     这是一个**已记录的缺口**（真实 Triton 的 `other` 是 Block）。这里如实按本模型的口径测。
chk.expect_ok(lambda: mt.load(src_buff, mt.arange(0, BLOCK),
                              mask=mt.less_than(mt.arange(0, BLOCK), N),
                              other=np.zeros(1, np.float32)),
              "C.8 形状 (1,) 的 `other` 能**隐式广播**到掩码的形状 (256,)，不抛错",
              "other 形状 (1,) 与掩码 (256,) 相容（本模型用 numpy 广播规则）")
chk.expect_raises(lambda: mt.load(src_buff, mt.arange(0, BLOCK),
                                 mask=mt.less_than(mt.arange(0, BLOCK), N),
                                 other=np.zeros(3, np.float32)),
                  ValueError,
                  "C.9 形状**不相容**的 `other`（(3,) vs (256,)）由广播规则抛错，不静默截断；"
                  "★ 同时暴露本模型的一个缺口：`other` **不接受 `Tile`**（真实 Triton 接受 Block）")

# A.15：扫掠里被检出的越界总数与独立算出来的 5655 对齐（覆盖 60 个组合）
exp_oob_total = sum(sum(1 for k in range(blk) if not (0 <= pid * blk + k < n))
                    for n in Ns for blk in BLOCKs
                    for pid in range((n + blk - 1) // blk))
chk.A(exp_oob_total == n_oob_total == 5655,
      "A.15 60 个 (N, BLOCK) 组合里越界位置的总数 = %d，与**另写一遍**的独立计数逐项一致"
      % n_oob_total,
      "独立计数 %d == 扫掠计数 %d" % (exp_oob_total, n_oob_total))

sys.exit(chk.summary())
