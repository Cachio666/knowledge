# -*- coding: utf-8 -*-
"""01 块级程序模型：块内张量的元素**归哪个线程管，由编译器决定**。

覆盖：1D / 2D block id 网格；`tl.arange` 的等价物；块内 `sum`/`max` 归约（含 `axis` / `keep_dims`）；
      fused 在线 softmax（两趟，行统计量在寄存器里滚动，不物化 N×N 中间矩阵）。

★ 证据强度（与 `30-gpu-programming/cuda/code/` 同一套三分法）：
  · **A 类（算法正确性）**：本程序的解释器跑 numpy，与 `refs.py` 的**独立参照**逐元素比对。
    这是最强的一档，但参照与被测跑在**同一台机器、同一个 numpy** 上。
  · **B 类（代价模型）**：**全部是模型值，本机无 GPU、`triton` 包全机器不存在**，
    没有一条是实测。单位是模型自定的「标量操作数 / 共享内存字节 / barrier 次数」，
    不是指令周期、不是带宽。它们只能支撑**关于机制与相对量的结论**。
  · **C 类（边界与反例）**：故意注入缺陷后断言必须失败。

★ 四处限定语（计划 §0.7 F10，逐处必写）：
  1. **迷你实现，不是 Triton**——本模型只实现了一个**精确切分**的 layout 子集（`specs.MODEL_SIMPLIFICATIONS["S1"]`）。
  2. **摊维规则 / 属主算法 / 转换定价公式 / 三相 ops 计数 / 交叉点，这 5 条是本模型自造的**，
     不声称是 Triton 的规定（`specs.MODEL_OWN_RULES`）。
  3. **B 类全部是模型值，本主题无实测**。
  4. **不是硬件，也不是 Triton 编译器**。

★ 本程序还担任**跨程序机械检查**的落点（沿用 `cuda/code/01_sync_contract.py` 的做法）：
  它是四篇里的第一篇，`make run` 第一个跑，所以把「全集覆盖」「禁用清单」「import 扫描」
  「元检验锚点完整性」四条**跨文件**的检查放在这里。
"""

import ast
import math
import os
import re
import sys

import numpy as np

import checks as C
import minitriton as mt
import refs
import specs

M, N = 32, 1000
BLOCK_N = 256
GRID = (M,)

# ★ `COVERED_ANCHORS`：本程序**拥有**哪些语义锚点。四篇的**并集必须等于 `specs` 的全部锚点**、
#   **两两不交**（`C.7` 检查）。本程序owning 0 条——它是检查器，不是锚点的消费者。
COVERED_ANCHORS = ()

chk = C.Checks("01_block_model")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ("01_block_model.py", "02_masking.py", "03_layout.py", "04_warp_reduce.py")
AUX = ("minitriton.py", "refs.py", "checks.py", "meta_falsify.py")
LABEL_RE = re.compile(r"^([ABC])\.(\d+[a-z]*)\b")
CB_RE = re.compile(r"^(\d\d):([ABC]\.\d+[a-z]*)$")


def print_env():
    print("  解释器 = %s" % sys.executable)
    print("  python %s / numpy %s（**无 GPU**；`triton` 包全机器不存在）"
          % (sys.version.split()[0], np.__version__))


def labels_of(path):
    """用 `ast` 从**源码字符串字面量**里提取断言标签（不执行那个程序）。

    ★ 为什么用 `ast` 而不是正则扫全文：注释与 docstring 里出现的 `A.1` 不算断言。
      判据来自 `cuda/code/Makefile` 第 110 行的同一条经验（`ast` 而非 grep）。
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            m = LABEL_RE.match(node.value)
            if m:
                out.add("%s.%s" % (m.group(1), m.group(2)))
    return out


def covered_anchors_of(path):
    """从源码里读出 `COVERED_ANCHORS`（不 import —— 那会执行整个程序）。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "COVERED_ANCHORS":
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("%s 没有 COVERED_ANCHORS" % path)


BANNED = ("torch", "triton", "numba", "cuda")


def lint_imports(files):
    """`ast` 扫描：静态 import 与**动态导入**（`import_module` / `__import__`）都算。

    ★ 为什么用 `ast` 而不是 grep：注释与 docstring 里出现 `triton` 这个词**不算违规**
      （本主题的文档串里到处都是），真 import 才算。判据沿用 `cuda/code/Makefile` 第 110 行。
    ★ **必须含 `specs.py`**（计划 §2 P6：声称「都不 import torch」却只查了编号程序）。
    """
    imp_hits, dyn_hits = [], []
    for fn in files:
        tree = ast.parse(open(os.path.join(CODE_DIR, fn), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    if al.name.split(".")[0] in BANNED:
                        imp_hits.append((fn, al.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BANNED:
                    imp_hits.append((fn, node.module))
            elif isinstance(node, ast.Call):
                f = node.func
                nm = getattr(f, "attr", None) or getattr(f, "id", None)
                if nm in ("import_module", "__import__") and node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                            and a0.value.split(".")[0] in BANNED:
                        dyn_hits.append((fn, a0.value))
    return imp_hits, dyn_hits


if "--lint" in sys.argv:
    # `make lint`：不跑代价模型、不读外部文档的快速入口（同一份扫描代码，两处复用）
    _imp, _dyn = lint_imports(PROGRAMS + AUX + ("specs.py",))
    print("AST import 扫描：%d 个 .py（四个编号程序 + 四个支撑模块 + specs.py）"
          % (len(PROGRAMS) + len(AUX) + 1,))
    print("  静态 import 命中 = %r" % (_imp,))
    print("  动态导入命中   = %r" % (_dyn,))
    sys.exit(1 if (_imp or _dyn) else 0)


# ---------------------------------------------------------------------------
# 1. 1D 网格：向量加（带边界掩码）
# ---------------------------------------------------------------------------

C.rule("1.1 1D block id 网格 + tl.arange 等价物 + 掩码读写")
print_env()
print()

a = mt.Buff("a", np.linspace(-3.0, 3.0, N, dtype=np.float32))
b = mt.Buff("b", np.arange(N, dtype=np.float32) * 0.001 + 0.5)
c = mt.Buff("c", np.zeros(N, dtype=np.float32))

MASTER_ORDER = []
MASK_TRUE = []


def vadd_kernel(pid, trace=None):
    offs = mt.arange(pid[0] * 256, pid[0] * 256 + 256)      # tl.arange 的等价物
    mask = mt.less_than(offs, N)                            # 越界位置
    x = mt.load(a, offs, mask=mask, other=0.0)
    y = mt.load(b, offs, mask=mask, other=0.0)
    mt.store(c, offs, mt.add(x, y), mask=mask)
    trace.append(("vadd", pid[0], int(mask.data.sum())))
    MASTER_ORDER.append(pid[0])
    MASK_TRUE.append(int(mask.data.sum()))


g1 = mt.Grid((4,))
r1 = mt.launch(vadd_kernel, g1)
print("grid %r，程序数 %d，pid 顺序（axis 0 最快变化）：%r"
      % (g1.dims, g1.size, [p for p in g1.ids()][:4]))
print("每个 program 的掩码为真个数：%r" % MASK_TRUE)

# A.1 pid 顺序：与「按 axis 0 最快变化手工枚举」独立算出来的顺序比
ref_order = []
for flat in range(4):
    r_ = flat
    for d in g1.dims:
        ref_order.append(r_ % d)
        r_ //= d
chk.A(MASTER_ORDER == ref_order,
      "A.1 启动顺序逐项等于「按 axis 0 最快变化手工枚举」——grid 的 pid 顺序是**算出来的**，"
      "不是调用方传进来的顺序",
      "实际 %r == 独立枚举 %r" % (MASTER_ORDER, ref_order))

# A.2 / A.3 逐元素比对（参照走 refs.py 的**纯 Python 标量循环**，不碰 minitriton）
A_LIST, B_LIST = a.arr.tolist(), b.arr.tolist()
ref_c = [0.0] * N
for pid in range(4):
    ra = refs.masked_block_ref(A_LIST, N, 256, pid, other=0.0)
    rb = refs.masked_block_ref(B_LIST, N, 256, pid, other=0.0)
    for k in range(256):
        i = pid * 256 + k
        if i < N:
            ref_c[i] = ra[k] + rb[k]
err_vadd = max(abs(float(c.arr[i]) - ref_c[i]) for i in range(N))
print("向量加：与纯 Python 标量参照的最大绝对误差 = %.3e" % err_vadd)
chk.A(err_vadd <= 1e-6, "A.2 向量加 1000 个位置逐元素等于独立参照（f32 容差 1e-6）",
      "max_abs_err = %.3e" % err_vadd)
n_tail_ok = int(np.sum(np.asarray(c.poison_tail[:24]) == mt.POISON))
chk.A(n_tail_ok == 24,
      "A.3 紧邻有效区的 24 个毒值全部完好——越界写没有发生（第 4 个 program 有 24 个越界位置）",
      "毒区前 24 个仍是毒值的个数 = %d/24" % n_tail_ok)

# A.4 掩码为真的个数：与独立算出来的 (min(BLOCK, N - pid*BLOCK)) 比
ref_true = [min(256, max(0, N - pid * 256)) for pid in range(4)]
chk.A(MASK_TRUE == ref_true,
      "A.4 四个 program 的掩码为真个数逐项等于独立算出的 min(BLOCK, N − pid×BLOCK)",
      "%r == %r（最后一个块只有 %d 个真实元素）" % (MASK_TRUE, ref_true, ref_true[-1]))


# ---------------------------------------------------------------------------
# 2. 2D 网格
# ---------------------------------------------------------------------------

C.rule("2 2D block id 网格：grid=(M//BM, N//BN)，axis 0 最快变化")

BM, BN = 16, 256
x2 = mt.Buff("x2", np.arange(M * 1024, dtype=np.float32).reshape(M, 1024) % 97.0)
y2 = mt.Buff("y2", np.zeros((M, 1024), dtype=np.float32))
PID2 = []


def scale2d_kernel(pid, trace=None):
    rows = mt.Tile(mt.offs_add(mt.arange(0, BM), pid[0] * BM).data[:, None], "i32")
    cols = mt.offs_add(mt.arange(0, BN), pid[1] * BN)
    mask = mt.less_than(cols, 1024)
    v = mt.load(x2, (rows, cols), mask=mask, other=0.0)
    mt.store(y2, (rows, cols), mt.add(mt.mul(v, 2.0), 1.0), mask=mask)
    trace.append(("scale2d", pid[0], pid[1]))
    PID2.append(pid)


g2 = mt.Grid((M // BM, 1024 // BN))
r2 = mt.launch(scale2d_kernel, g2)
print("grid = %r（%d 个 program），前 4 个 pid = %r"
      % (g2.dims, g2.size, list(g2.ids())[:4]))
ref2 = np.asarray(x2.arr, np.float64) * 2.0 + 1.0
err2 = float(np.max(np.abs(np.asarray(y2.arr, np.float64) - ref2)))
print("2D 逐元素：最大绝对误差 = %.3e" % err2)
chk.A(err2 <= 1e-3, "A.5 2D 网格逐元素结果与 numpy 参照一致",
      "max_abs_err = %.3e（值域约 195，1e-3 相对约 5e-6）" % err2)

# A.6 pid 的 2D 顺序与尺寸：与独立枚举比（**这是 A.1 的 2D 版**）
ref2_pids = [(flat % g2.dims[0], flat // g2.dims[0]) for flat in range(g2.size)]
chk.A(PID2 == ref2_pids and len(PID2) == g2.size,
      "A.6 2D 网格发射了 %d 个 program，pid 顺序逐项等于独立枚举（axis 0 最快变化）" % g2.size,
      "前 3 个：%r == %r" % (PID2[:3], ref2_pids[:3]))


# ---------------------------------------------------------------------------
# 3. 块内归约 sum / max
# ---------------------------------------------------------------------------

C.rule("3 块内 sum / max 归约（含 axis / keep_dims 语义）")

t = mt.Tile(np.arange(1, 257, dtype=np.float32).reshape(16, 16), "f32")
alls = mt.tsum(t)
rowsum = mt.tsum(t, axis=1)
colmax = mt.tmax(t, axis=0)
rowsum_kd = mt.tsum(t, axis=1, keep_dims=True)

print("sum(axis=None) = %.1f   （真值 %d）" % (float(alls.data), 16 * 16 * 257 // 2))
print("sum(axis=1) 形状 %r，前 3 个 = %r" % (rowsum.shape, rowsum.data[:3].tolist()))
print("max(axis=0) 形状 %r，前 3 个 = %r" % (colmax.shape, colmax.data[:3].tolist()))
print("sum(axis=1, keep_dims=True) 形状 %r" % (rowsum_kd.shape,))

naive_all = mt.naive_reduce(t)
chk.A(abs(naive_all - float(alls.data)) <= 1e-3,
      "A.7 axis=None 的归约等于纯 Python 循环参照（naive_reduce 不调 numpy 的 sum）",
      "naive=%.4f  本解释器=%.4f" % (naive_all, float(alls.data)))

ref_rowsum = refs.row_sum_ref(t.data.tolist())
err_rs = max(abs(float(rowsum.data[i]) - ref_rowsum[i]) for i in range(16))
chk.A(err_rs <= 1e-3, "A.8 axis=1 的逐行和等于 refs.row_sum_ref 的标量参照",
      "max_abs_err = %.3e（每行 16 项，和约 2056）" % err_rs)
chk.A(rowsum_kd.shape == (16, 1), "A.9 keep_dims=True 保留被归约的维（形状 (16,) → (16,1)）")

ref_colmax = [max(float(t.data[i, j]) for i in range(16)) for j in range(16)]
colmax_bad = sum(1 for j in range(16) if float(colmax.data[j]) != ref_colmax[j])
chk.A(colmax_bad == 0,
      "A.10 axis=0 的逐列最大逐列等于 refs 风格的独立标量扫描（16 列全比）",
      "不一致的列数 = %d；max(axis=0) 前 3 个 = %r" % (colmax_bad, colmax.data[:3].tolist()))

# A.11 归约只在一条轴上发生：axis=1 之后形状少一维，且每行和 == 全归约的对照量
rowsum_from_all = float(alls.data) / 16.0
chk.A(abs(sum(ref_rowsum) / 16.0 - rowsum_from_all) <= 1e-3,
      "A.11 逐行和的平均 == 全归约/16（两条路径落到同一个数上）",
      "%.6f vs %.6f" % (sum(ref_rowsum) / 16.0, rowsum_from_all))


# ---------------------------------------------------------------------------
# 4. fused 在线 softmax
# ---------------------------------------------------------------------------

C.rule("4 fused 在线 softmax（两趟，行统计量在寄存器里滚动）")

rng = np.random.default_rng(20260920)
X = (rng.standard_normal((M, N)) * 1.5).astype(np.float32)
xb = mt.Buff("X", X)
yb = mt.Buff("Y", np.zeros((M, N), dtype=np.float32))
STATS = []


def softmax_kernel(pid, trace=None):
    row = pid[0]
    cols = mt.arange(0, BLOCK_N)
    r = mt.Tile(np.int32(row), "i32")

    # ---- 第 1 趟：滚动 (m_i, s_i)，不物化 N×N ----
    m_i = None
    s_i = None
    n_reduce = 0
    for n0 in range(0, N, BLOCK_N):
        offs = mt.offs_add(cols, n0)
        mask = mt.less_than(offs, N)
        x = mt.load(xb, (r, offs), mask=mask, other=float("-inf"))
        blk_max = mt.tmax(x, axis=None)
        if m_i is None:
            m_new = blk_max
            s_i = mt.tsum(mt.exp(mt.sub(x, m_new)), axis=None)
        else:
            m_new = mt.maximum(m_i, blk_max)
            rescale = mt.mul(s_i, mt.exp(mt.sub(m_i, m_new)))
            s_i = mt.add(rescale, mt.tsum(mt.exp(mt.sub(x, m_new)), axis=None))
        m_i = m_new
        n_reduce += 1

    # ---- 第 2 趟：重算并写回 ----
    for n0 in range(0, N, BLOCK_N):
        offs = mt.offs_add(cols, n0)
        mask = mt.less_than(offs, N)
        x = mt.load(xb, (r, offs), mask=mask, other=0.0)
        y = mt.div(mt.exp(mt.sub(x, m_i)), s_i)
        mt.store(yb, (r, offs), y, mask=mask)
    trace.append(("softmax_row", row, n_reduce,
                  float(np.asarray(m_i.data).reshape(-1)[0]),
                  float(np.asarray(s_i.data).reshape(-1)[0])))
    STATS.append((row, n_reduce,
                  float(np.asarray(m_i.data).reshape(-1)[0]),
                  float(np.asarray(s_i.data).reshape(-1)[0])))


g4 = mt.Grid((M,))
r4 = mt.launch(softmax_kernel, g4)

print("grid = %r，%d 个 program，每个 program 归约 %d 次（每 %d 列一次）"
      % (g4.dims, g4.size, STATS[0][1], BLOCK_N))
print("program 0 的 (m_i, s_i) = (%r, %r)" % (STATS[0][2], STATS[0][3]))

got = np.asarray(yb.arr, np.float64)
ref_sm = refs.softmax_rows_ref(X.tolist())                        # 参照式 1：减行最大
ref_sm2 = refs.softmax_rows_ref_shifted(X.tolist(), c_index=0)    # 参照式 2：减首元素
ref_sm3 = refs.softmax_rows_stream_ref(X.tolist(), BLOCK_N)       # 参照式 3：按块递推（标量）

e1 = refs.max_abs_err(got.tolist(), ref_sm)
e2 = refs.max_abs_err(got.tolist(), ref_sm2)
e3 = refs.max_abs_err(got.tolist(), ref_sm3)
tol = 1e-5
print("逐元素最大绝对误差： vs 减行最大参照 %.3e ｜ vs 减首元素参照 %.3e ｜ vs 按块递推参照 %.3e"
      % (e1, e2, e3))
chk.A(e1 <= tol, "A.12 fused 在线 softmax 逐元素等于「减行最大」标量参照（32000 个位置全比）",
      "max_abs_err = %.3e，容差 %.0e" % (e1, tol))
chk.A(e2 <= tol, "A.13 同时等于「减行内首元素」参照——**平移量的选择不改变结果**"
      "（这条参照根本不求最大值，所以它抓的是「减错了 / 忘减了」）",
      "max_abs_err = %.3e" % e2)
chk.A(e3 <= tol, "A.14 同时等于「按块递推」标量参照——在线递推式本身被独立复核",
      "max_abs_err = %.3e" % e3)

rowsum_out = got.sum(axis=1)
print("输出逐行和：min=%.8f max=%.8f（应为 1）" % (rowsum_out.min(), rowsum_out.max()))
chk.A(abs(rowsum_out - 1.0).max() <= 1e-5, "A.15 输出每行和 == 1（softmax 的归一性）",
      "max|rowsum-1| = %.3e" % float(abs(rowsum_out - 1.0).max()))

# A.16 滚动的行统计量本身对不对：m_i 必须等于 refs.row_max_ref 的行最大
ref_m = refs.row_max_ref(X.tolist())
m_bad = [row for row, _, mi, _ in STATS if mi != ref_m[row]]
chk.A(not m_bad,
      "A.16 每个 program 滚动出来的 m_i 逐行**精确等于** refs.row_max_ref 的行最大"
      "（%d 行全比）" % M,
      "不一致的行 = %r" % (m_bad[:5],))

# A.17 s_i 与「以 m_i 为平移量的指数和」一致（容差按 f32 求和阶数推）
s_bad = []
for row, _, mi, si in STATS:
    ref_s = math.fsum(math.exp(float(v) - mi) for v in X[row])
    if abs(si - ref_s) / abs(ref_s) > refs.rel_err_bound(N):
        s_bad.append((row, si, ref_s))
chk.A(not s_bad,
      "A.17 滚动出来的 s_i 落在「以 m_i 为平移量的指数和」的 f32 重结合上界内"
      "（容差 = n×2⁻²³×64，按机器 epsilon 推，不是拍脑袋）",
      "越界的行 = %r；容差 %.3e" % (s_bad[:3], refs.rel_err_bound(N)))


# ---------------------------------------------------------------------------
# 5. C 类：边界与反例
# ---------------------------------------------------------------------------

C.rule("5 C 边界与反例")

# C.1：不做最大值平移 → f32 上溢成 inf/inf = NaN
BIG = (X.astype(np.float64) + 200.0).astype(np.float32)     # exp(200) 在 f32 上上溢
xb2 = mt.Buff("Xbig", BIG)
yb2 = mt.Buff("Ybig", np.zeros((M, N), dtype=np.float32))


def naive_no_max_kernel(pid, trace=None):
    """故意**不**减最大值的版本——用来证明「减最大」这一步不是装饰。"""
    row = pid[0]
    cols = mt.arange(0, BLOCK_N)
    r = mt.Tile(np.int32(row), "i32")
    s_i = None
    for n0 in range(0, N, BLOCK_N):
        offs = mt.offs_add(cols, n0)
        mask = mt.less_than(offs, N)
        x = mt.load(xb2, (r, offs), mask=mask, other=float("-inf"))
        acc = mt.tsum(mt.exp(x), axis=None)
        s_i = acc if s_i is None else mt.add(s_i, acc)
    for n0 in range(0, N, BLOCK_N):
        offs = mt.offs_add(cols, n0)
        mask = mt.less_than(offs, N)
        x = mt.load(xb2, (r, offs), mask=mask, other=0.0)
        mt.store(yb2, (r, offs), mt.div(mt.exp(x), s_i), mask=mask)


mt.launch(naive_no_max_kernel, mt.Grid((M,)))
n_nan = int(np.isnan(yb2.arr[:M, :N]).sum())
ref_nan_all = yb2.arr[:M, :N].size          # 32000 = 32 × 1000，独立算出来
print("无最大值平移版本：NaN 个数 = %d / %d" % (n_nan, M * N))
chk.C(n_nan == ref_nan_all == 32000,
      "C.1 「忘掉最大值平移」在 x≈200 的输入上**真的**产出 NaN：32000/32000 个输出位置**全是** NaN"
      "（不是「有一些」——归一化常数本身变成了 inf）",
      "NaN 占比 %d/%d；正确版本在此输入上应全为有限值" % (n_nan, M * N))

# 正确版本在同一个输入上必须全是有限值
xb3 = mt.Buff("Xbig3", BIG)
yb3 = mt.Buff("Ybig3", np.zeros((M, N), dtype=np.float32))


def softmax_kernel_on(xsrc, ydst, pid, trace=None):
    row = pid[0]
    cols = mt.arange(0, BLOCK_N)
    r = mt.Tile(np.int32(row), "i32")
    m_i = None
    s_i = None
    for n0 in range(0, N, BLOCK_N):
        offs = mt.offs_add(cols, n0)
        mask = mt.less_than(offs, N)
        x = mt.load(xsrc, (r, offs), mask=mask, other=float("-inf"))
        bm = mt.tmax(x, axis=None)
        if m_i is None:
            m_i = bm
            s_i = mt.tsum(mt.exp(mt.sub(x, bm)), axis=None)
        else:
            m_new = mt.maximum(m_i, bm)
            s_i = mt.add(mt.mul(s_i, mt.exp(mt.sub(m_i, m_new))),
                         mt.tsum(mt.exp(mt.sub(x, m_new)), axis=None))
            m_i = m_new
    for n0 in range(0, N, BLOCK_N):
        offs = mt.offs_add(cols, n0)
        mask = mt.less_than(offs, N)
        x = mt.load(xsrc, (r, offs), mask=mask, other=0.0)
        mt.store(ydst, (r, offs), mt.div(mt.exp(mt.sub(x, m_i)), s_i), mask=mask)


mt.launch(lambda pid, trace=None: softmax_kernel_on(xb3, yb3, pid, trace), mt.Grid((M,)))
finite = bool(np.isfinite(yb3.arr[:M, :N]).all())
print("有最大值平移版本：全部有限 = %s" % finite)
chk.A(finite, "A.18 对照：同一个输入上，减最大值的版本**全为有限值**"
      "（所以 C.1 的 NaN 不是「这个输入太毒」，是**缺了那一步**）",
      "finite=%s；该输入的元素级最大值 %.1f ⇒ exp(最大) 在 f32 上必然上溢"
      % (finite, float(BIG.max())))

# C.2/C.3：缺陷注入——掩码写窄 / 写宽
C.rule("5b 缺陷注入：断言必须真的会失败")
save_lt = mt.less_than


def narrow_mask(offs, n):
    """掩码**写窄**了：拿块长度 256 去比，而不是真实的 N=1000。"""
    return mt.Tile(offs.data < 256, dtype="bool")


def wide_mask(offs, n):
    """掩码**写宽**了：把整个 1024 都当有效。"""
    return mt.Tile(offs.data < 1024, dtype="bool")


mt.less_than = narrow_mask
c.arr[...] = 0
raised = None
try:
    mt.launch(vadd_kernel, mt.Grid((4,)))
except mt.OutOfBoundsError as e:
    raised = e
mt.less_than = save_lt
err_broken = max(abs(float(c.arr[i]) - ref_c[i]) for i in range(N))
print("掩码写窄（offs < 256）后：抛错 = %s；与参照的最大误差 = %.3e"
      % ("是" if raised else "否", err_broken))
chk.C(raised is None and err_broken > 1.0,
      "C.2 掩码**写窄**不会触发越界检测（**根本没有越界访问**），只会静默算错——"
      "唯一能抓住它的是 A.2 的逐元素比对",
      "误差从 %.3e 涨到 %.3e，A.2 判据（<=1e-6）必然失败" % (err_vadd, err_broken))

mt.less_than = wide_mask
raised = None
try:
    mt.launch(vadd_kernel, mt.Grid((4,)))
except mt.OutOfBoundsError as e:
    raised = e
mt.less_than = save_lt
chk.C(raised is not None,
      "C.3 掩码**写宽**触发越界访问 → 抛 OutOfBoundsError（写宽**能**被抓，写窄**不能**）",
      "错误信息：%s" % (str(raised)[:140] if raised else "（没有抛错）"))

c.arr[...] = 0
mt.launch(vadd_kernel, mt.Grid((4,)))
err_restored = max(abs(float(c.arr[i]) - ref_c[i]) for i in range(N))
chk.A(err_restored <= 1e-6, "A.19 缺陷注入测试后复原再跑，A.2 重新成立",
      "max_abs_err = %.3e" % err_restored)

# C.4：反证——把 store 的掩码去掉
save_store = mt.store


def broken_store(buff, index, value, mask=None, strict=True):
    return save_store(buff, index, value, mask=None, strict=strict)


mt.store = broken_store
try:
    mt.launch(vadd_kernel, mt.Grid((4,)))
    chk.C(False, "（反例分支）去掉 store 掩码后仍安然通过")
except mt.OutOfBoundsError as e:
    chk.C(True, "C.4 去掉**写**掩码后越界报错——「掩码真的挡住了写」有判别力",
          "错误信息：" + str(e)[:140])
finally:
    mt.store = save_store

# C.5：★ 「自造真值」的正面检验之一 —— 把**被测的解释器**改错，A 类必须失败
#   （注意：**不能**用「换一个平移量」来构造反例 —— softmax 对任意常数平移不变，
#     换平移量在数学上仍然是**对的**，那样构造出来的断言是**假的可证伪性**。
#     这里改的是 exp 本身，那才是真的错。）
_save_exp = mt.exp
yb.arr[...] = 0


def bad_exp(x):
    """`exp(x+1)` —— 真的错（比正确值大 e 倍）。"""
    return mt.Tile(np.exp(x.data.astype(np.float32) + np.float32(1.0)), dtype="f32")


mt.exp = bad_exp
try:
    mt.launch(softmax_kernel, g4)
    e_bad = refs.max_abs_err(np.asarray(yb.arr, np.float64).tolist(), ref_sm)
finally:
    mt.exp = _save_exp
chk.C(e_bad > tol,
      "C.5 把**被测解释器**的 `exp` 改成 `exp(x+1)`，A.12 的观测量立刻**超出容差**——"
      "A 类断言真的在检验解释器，不是在自证",
      "改错后 max_abs_err = %.3e > 容差 %.0e" % (e_bad, tol))

yb.arr[...] = 0
mt.launch(softmax_kernel, g4)
e_restore = refs.max_abs_err(np.asarray(yb.arr, np.float64).tolist(), ref_sm)
chk.A(e_restore <= tol, "A.20 恢复 `exp` 后 A.12 重新成立",
      "max_abs_err = %.3e" % e_restore)

# C.6：★ `refs.py` 的独立性是 A 类的**全部意义** —— 把参照改成不一致，A 类也必须失败
_save_shifted = refs.softmax_rows_ref_shifted


def wrong_shifted(x, c_index=0):
    """漏掉**最后一个**元素的指数项（分母少一项）——参照与源不一致。"""
    out = []
    for row in x:
        c = float(row[c_index])
        es = [math.exp(float(v) - c) for v in row]
        s = 0.0
        for e in es[:-1]:                       # ← 少累加一项
            s += e
        out.append([e / s for e in es])
    return out


refs.softmax_rows_ref_shifted = wrong_shifted
try:
    e_ref_bad = refs.max_abs_err(got.tolist(),
                                 refs.softmax_rows_ref_shifted(X.tolist(), c_index=0))
finally:
    refs.softmax_rows_ref_shifted = _save_shifted
chk.C(e_ref_bad > tol,
      "C.6 把 `refs.py` 的参照改成「分母少累加一项」，A.13 的观测量立刻**超出容差**——"
      "判据是**故意把两者改成不一致，断言必须失败**（`CONTRIBUTING` 的独立性判据）",
      "改错后 max_abs_err = %.3e > 容差 %.0e" % (e_ref_bad, tol))


# ---------------------------------------------------------------------------
# 6. B 类：代价模型（模型值，不是实测）
# ---------------------------------------------------------------------------

C.rule("6 B 代价模型：softmax 的行统计量归约要付几个相位")

LAY = mt.Layout([BLOCK_N], [2], [32], [4], order=[0], dtype="f32")
plan = mt.ReducePlan(LAY, 0)
bc = plan.broadcast_back()
print("行统计量的 tile 是 [%d]，布局 %r" % (BLOCK_N, LAY))
print("  归约：线程内 %d ops ｜ warp 内蝶形 %d 步 / %d ops ｜ 跨 warp %d ops / %dB / %d barrier"
      % (plan.phase1_ops, plan.warp_steps, plan.phase2_ops,
         plan.phase3_ops, plan.phase3_smem_bytes, plan.phase3_barriers))
print("  广播回：%s（%d ops / %dB / %d barrier）"
      % (bc.kind, bc.ops, bc.smem_bytes, bc.barriers))
per_step = plan.barriers + bc.barriers
n_blocks = N // BLOCK_N + (1 if N % BLOCK_N else 0)
total_barriers = per_step * 2 * n_blocks
print("  每个列块：%d 次 barrier ｜ 全行两趟共 %d 次 barrier（%d 个列块 × 2 趟）"
      % (per_step, total_barriers, n_blocks))

chk.B(plan.phase3_barriers == 2 and bc.barriers == 2,
      "B.1 [BLOCK_N=256] 的行归约**不是**一个常数代价：它有跨 warp 相位（2 barrier）"
      "且广播回还要再来 2 次",
      "warp 内摊开 %d 个线程 / 跨 %d 个 warp；跨 warp 相位 smem %d B"
      % (LAY.tpw[0], LAY.wpc[0], plan.phase3_smem_bytes))

L32 = mt.Layout([32], [1], [32], [1], order=[0], dtype="f32")
p32 = mt.ReducePlan(L32, 0)
chk.B(p32.barriers == 0 and p32.broadcast_back().barriers == 0,
      "B.2 对照：同一个归约放在**一个 warp** 的 32 元素块上 → 跨 warp 相位消失，"
      "广播回也是 0（值本来就在本线程里）",
      "B=32 布局 barrier = %d ｜ 广播 barrier = %d"
      % (p32.barriers, p32.broadcast_back().barriers))

chk.B(n_blocks == 4 and total_barriers == 32,
      "B.3 一行 1000 个元素被切成 %d 个列块，两趟合计 %d 次 barrier"
      "（本模型的口径：**每趟**的每次归约都单独付账，不假设编译器能合并）" % (n_blocks, total_barriers),
      "per_step=%d，n_blocks=%d，×2 趟 ⇒ %d" % (per_step, n_blocks, total_barriers))

slots = g2.size * BM * BN
useful = M * 1024
chk.B(slots == g2.size * BM * BN,
      "B.4 2D 网格发射的线程槽 = %d（%d 个 program × %d×%d），实际元素 = %d ⇒ 每槽 %.2f 个元素"
      % (slots, g2.size, BM, BN, useful, useful / slots),
      "本模型只算槽位，不主张任何关于硬件占用率的结论（本机无 GPU）")


# ---------------------------------------------------------------------------
# 7. 跨程序机械检查（P6 的机械载体）
# ---------------------------------------------------------------------------

C.rule("7 跨程序机械检查：锚点全集 / 禁用清单 / import 扫描 / 元检验锚点")

# C.6：★ 禁用清单（计划 §0.7 F11）——探针自报「未一手复核」的东西**不得被引用**
#   扫**除 specs.py 以外**的全部 .py（specs.py 就是那张清单本身）。
forbidden = list(specs.NOT_FIRST_HAND.values())
# ★ 待查的字面量**从 `specs.py` 读**，不写在检查器里：检查器自己含这些字面量的话，
#   它扫自己就会命中（**自指陷阱**，第一版就是这么失败的）。
FORBID_TOKENS = specs.FORBIDDEN_TOKENS
hits = []
for fn in PROGRAMS + AUX:
    txt = open(os.path.join(CODE_DIR, fn), encoding="utf-8").read()
    for tok in FORBID_TOKENS:
        if tok in txt:
            hits.append((fn, tok))
print("禁用清单：%d 条（%s）" % (len(forbidden), "、".join("N%d" % i for i in range(1, 8))))
print("待查字面量 %d 个（从 specs.FORBIDDEN_TOKENS 读）；扫描的文件：%s"
      % (len(FORBID_TOKENS), PROGRAMS + AUX))
print("  **不含 specs.py**——它自己就是那张清单（自指陷阱）")
chk.C(not hits,
      "C.11 禁用清单里的字面量在**四个编号程序与四个支撑模块**里出现 0 次——"
      "「未一手复核」的东西没有被引用",
      "命中 = %r" % (hits[:5],))

# C.7：★ 锚点全集覆盖 + 两两不交 + `covered_by` 的标签真实存在
own = {p: covered_anchors_of(os.path.join(CODE_DIR, p)) for p in PROGRAMS}
all_anchors = tuple(specs.ANCHORS) + tuple(specs.LOAD_QUOTES)
union = [x for v in own.values() for x in v]
print("四篇各自拥有的锚点：%s" % {k: list(v) for k, v in own.items()})
print("锚点全集 = %r（%d 条）" % (all_anchors, len(all_anchors)))
chk.C(sorted(union) == sorted(all_anchors) and len(union) == len(set(union)),
      "C.7 四篇 `COVERED_ANCHORS` 的**并集 == specs 的全部 %d 条锚点**，且**两两不交**"
      "——在 specs.py 里加一条没人认领的锚点，这条断言立刻失败" % len(all_anchors),
      "并集 %r == 全集 %r，重复项 = %r"
      % (sorted(union), sorted(all_anchors), [x for x in union if union.count(x) > 1]))

labels = {p: labels_of(os.path.join(CODE_DIR, p)) for p in PROGRAMS}
# ★★ `covered_by` 的校验必须覆盖**全部四个注册表**，不只是两条外部事实表：
#    第一版只查了 `ANCHORS` / `LOAD_QUOTES`，于是 `MODEL_OWN_RULES` 与
#    `MODEL_SIMPLIFICATIONS` 里的 24 个标签**一个都没被校验** ——
#    变异测试把它们全部报成「漏网」（改一个标签，没有任何程序失败）。
#    ⇒ 「每条断言/常量带 covered_by」这句话，只有在**四处都查**时才成立（P6 的机械载体）。
REGISTRIES = (("ANCHORS", specs.ANCHORS), ("LOAD_QUOTES", specs.LOAD_QUOTES),
              ("MODEL_OWN_RULES", specs.MODEL_OWN_RULES),
              ("MODEL_SIMPLIFICATIONS", specs.MODEL_SIMPLIFICATIONS))
bad_cb = []
for group, reg in REGISTRIES:
    for aid, a in reg.items():
        if not a.get("covered_by"):
            bad_cb.append((group, aid, "covered_by 为空"))
        for cb in a["covered_by"]:
            m = CB_RE.match(cb)
            if not m:
                bad_cb.append((group, aid, cb + " 形状不合 NN:类.号"))
                continue
            prog = PROGRAMS[int(m.group(1)) - 1]
            if m.group(2) not in labels[prog]:
                bad_cb.append((group, aid, cb + " 在 %s 里不存在" % prog))
n_cb = sum(len(a.get("covered_by", ())) for _, reg in REGISTRIES for a in reg.values())
chk.C(not bad_cb,
      "C.8 **四个注册表**（`ANCHORS` / `LOAD_QUOTES` / `MODEL_OWN_RULES` / "
      "`MODEL_SIMPLIFICATIONS`）里 %d 个 `covered_by` 标签，**每一个都真实存在于"
      "对应程序的断言清单里**（用 `ast` 从源码字面量提取，不是手抄的清单）" % n_cb,
      "不合规 = %r" % (bad_cb[:6],))

# C.9b：`specs.META_DEFECTS` 与 `meta_falsify.DEFECTS` 的**双向绑定**
#   ★ 两处各记一份「注入了哪些缺陷」的账 —— 它们一旦漂开，元检验的**结论**就不可信
#     （元检验说抓住了 D3，而 specs 说没有 D3）。这条断言把两份账钉在一起：
#     条数相同、**每一条缺陷名逐字相同**、**每一条点名的程序都存在于磁盘上**。
mi = ast.parse(open(os.path.join(CODE_DIR, "meta_falsify.py"), encoding="utf-8").read())
meta_names, meta_progs = [], []
for node in mi.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "DEFECTS" for t in node.targets):
        for elt in node.value.elts:
            meta_names.append(ast.literal_eval(elt.elts[0]))
            meta_progs.append(tuple(ast.literal_eval(elt.elts[3])))
spec_names = [specs.META_DEFECTS[k][0] for k in sorted(specs.META_DEFECTS)]
# `meta_falsify.DEFECTS` 的缺陷名带 `D1 ` 这样的前缀，`specs` 那份不带 ——
# 所以判据是「specs 的每一条**被恰好一条** meta 记录包含」，不是「逐字相等」。
name_ok = [n for n in spec_names if sum(1 for m in meta_names if n in m) != 1]
file_ok = [p for ps in meta_progs for p in ps
           if not os.path.exists(os.path.join(CODE_DIR, p))]
chk.C(len(meta_names) == len(spec_names) == 5 and not name_ok and not file_ok,
      "C.9 `specs.META_DEFECTS`（5 条）与 `meta_falsify.DEFECTS`（5 条）**逐条**对得上："
      "缺陷名逐字相同，且每一条点名的「应该被哪个程序抓住」都**真的存在于磁盘上**"
      "——两份账漂开时，元检验的结论就不再可信",
      "specs 有而 meta 没有的 = %r ｜ 点名但不存在的文件 = %r" % (name_ok, file_ok))

# C.10b：`FORBIDDEN_TOKENS` 必须是 `NOT_FIRST_HAND` 那张清单的**短字面量投影**
#   ★ 这一条把「人工写的禁用清单」与「机器扫的字面量集」钉在一起，**两个方向都可证伪**：
#     改一个 token（把它的一个字符换掉），或改清单原文里对应的那个词，两边都会失配。
blob = " ".join(specs.NOT_FIRST_HAND.values())
miss = [t for t in specs.FORBIDDEN_TOKENS if t not in blob]
chk.C(bool(len(specs.FORBIDDEN_TOKENS)) and not miss,
      "C.10 `FORBIDDEN_TOKENS` 的 %d 个字面量**每一个都出现在** `NOT_FIRST_HAND` 的"
      "某一条原文里 —— 禁用清单与它的机器可核形式是**同一份账**"
      % len(specs.FORBIDDEN_TOKENS),
      "没出现在清单原文里的 token = %r" % (miss,))

# C.9：`must_contain` 必须是 `quote` 的逐字子串（防止「引文与它支持的话不匹配」的另一半）
bad_mc = []
for group, reg in (("ANCHORS", specs.ANCHORS), ("LOAD_QUOTES", specs.LOAD_QUOTES)):
    for aid, a in reg.items():
        for s in a["must_contain"]:
            if s not in a["quote"]:
                bad_mc.append((aid, s[:40]))
chk.A(not bad_mc,
      "A.20 七条锚点的每一条 `must_contain` 都是它自己 `quote` 的逐字子串"
      "（引文与断言作用域的第一道自洽检查）",
      "不合规 = %r" % (bad_mc[:6],))

# C.10：import 扫描（**含 specs.py**，含 `import_module` 动态导入）—— P6 的机械载体
imp_hits, dyn_hits = lint_imports(PROGRAMS + AUX + ("specs.py",))
chk.C(not imp_hits and not dyn_hits,
      "C.12 九个 .py（四个编号程序 + 四个支撑模块 + `specs.py`）**都不 import** "
      "torch / triton / numba / cuda，也没有动态导入它们",
      "静态 import 命中 = %r ｜ 动态 import 命中 = %r" % (imp_hits[:4], dyn_hits[:4]))

# C.11：★ 元检验的锚点必须在 `minitriton.py` 里**恰好出现 1 次**
#   否则 `make falsify` 会**静默变成空操作**（补丁打不上、程序当然全过）。
mini = open(os.path.join(CODE_DIR, "minitriton.py"), encoding="utf-8").read()
meta_tree = ast.parse(open(os.path.join(CODE_DIR, "meta_falsify.py"),
                           encoding="utf-8").read())
anchors_in_meta = []
for node in meta_tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "DEFECTS" for t in node.targets):
        for elt in node.value.elts:               # 每个元素是 (缺陷名, 锚点, 替换, 程序表)
            anchors_in_meta.append(ast.literal_eval(elt.elts[1]))
uniq_bad = [(a, mini.count(a)) for a in anchors_in_meta if mini.count(a) != 1]
print("meta_falsify.py 里的缺陷锚点：%d 条；`specs.META_DEFECTS` 记了 %d 条"
      % (len(anchors_in_meta), len(specs.META_DEFECTS)))
chk.C(len(anchors_in_meta) == 5 and not uniq_bad,
      "C.13 元检验注入的 5 种缺陷，每条的**锚点在 minitriton.py 里恰好出现 1 次**——"
      "锚点漂了 `make falsify` 会**静默变成空操作**（补丁打不上、程序当然全过），"
      "这条断言把它钉住",
      "锚点数 = %d（应为 5）；出现次数不为 1 的 = %r" % (len(anchors_in_meta), uniq_bad))

# C.12：`specs.META_DEFECTS` 的条数与「应该被抓住的程序」也对齐（双向绑定）
chk.A(len(specs.META_DEFECTS) == 5 and all(
    specs.META_DEFECTS[k][1].endswith(".py") or " / " in specs.META_DEFECTS[k][1]
    for k in specs.META_DEFECTS),
      "A.21 `specs.META_DEFECTS` 记的 5 条缺陷，每条都点名了「应该被哪个程序抓住」",
      "键 = %r" % (list(specs.META_DEFECTS),))

# C.13：模型的简化与自造规则都必须**非空且被 covered_by 认领**
chk.A(bool(specs.MODEL_OWN_RULES) and bool(specs.MODEL_SIMPLIFICATIONS)
      and len(specs.MODEL_OWN_RULES) == 5 and len(specs.MODEL_SIMPLIFICATIONS) == 3,
      "A.22 本模型的 5 条自造规则与 3 条已记录简化都在 `specs.py` 里（各自带 `basis`/`consequence`）",
      "自造规则 %d 条、简化 %d 条" % (len(specs.MODEL_OWN_RULES),
                                  len(specs.MODEL_SIMPLIFICATIONS)))

# A.23：numpy 的归约与 naive 参照在整数上没有重结合余量（A.7 的加强版）
ivals = np.arange(1, 257, dtype=np.int64)
chk.A(int(mt.naive_reduce(ivals)) == int(ivals.sum()) == 32896,
      "A.23 整数输入上「纯 Python 循环参照」与 numpy 的 sum **逐位相等**（无重结合余量）",
      "naive=%d numpy=%d 闭式=%d" % (int(mt.naive_reduce(ivals)), int(ivals.sum()),
                                   256 * 257 // 2))

# C.11/C.12：形状与网格的边界（**契约的两个方向**）
chk.expect_raises(lambda: mt.arange(5, 5), mt.ShapeError,
                  "C.14 `tl.arange` 的等价物是**半开区间** [lo, hi)：空区间直接抛 ShapeError")
chk.A(mt.arange(3, 7).data.tolist() == [3, 4, 5, 6],
      "A.24 `arange(3, 7)` 给出 4 个元素且**不含 7**（半开区间 —— 这是掩码算术的前提）",
      "实得 %r" % (mt.arange(3, 7).data.tolist(),))
chk.expect_raises(lambda: mt.Grid([2, 0]), mt.ShapeError,
                  "C.15 grid 维度必须为正（0 或负数直接抛 ShapeError，不静默产出空 grid）")

# B.5：广播回代价的模型值（**模型值，不是实测**）
chk.B(bc.kind == "smem-broadcast" and bc.positions_per_thread == 1 and bc.barriers == 2,
      "B.5 广播回的模型口径：`%s`，每线程 %d 个位置，%d 个 barrier、%d B —— "
      "这四个数都是**本模型算出来的**，不是实测" % (bc.kind, bc.positions_per_thread,
                                              bc.barriers, bc.smem_bytes),
      "★ 口径自定（`specs.MODEL_OWN_RULES['R3']` 那一族）")

# C.13：又一处 refs 独立性 —— 换一条参照（行和）再验一次
_save_rowsum = refs.row_sum_ref
refs.row_sum_ref = lambda x: [sum(float(v) for v in row[:-1]) for row in x]   # 少一项
try:
    bad_rs = refs.row_sum_ref(t.data.tolist())
finally:
    refs.row_sum_ref = _save_rowsum
bad_cnt = sum(1 for i in range(16) if abs(float(rowsum.data[i]) - bad_rs[i]) > 1e-3)
chk.C(bad_cnt == 16,
      "C.16 把 `refs.row_sum_ref` 改成「少累加一项」，16 行**全部**超出容差 ⇒ "
      "A.8 的 `refs.py` 参照是**真的在参与判断**（这是第二处 refs 独立性检验）",
      "超差的行数 = %d / 16" % bad_cnt)

sys.exit(chk.summary())
