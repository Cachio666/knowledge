#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_checkpointing.py —— **重算换显存**：`O(√n)` 的内存、`≈2n` 的前向、`O(log n)` 的极端情形。

配套文档：`02-gradient-checkpointing.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **术语消歧（域 README 的硬要求）**：文件名里的 `checkpointing` 是 **Chen 2016 论文里的术语**
  （`gradient checkpointing`），它指**重算换显存**，**不是权重检查点**。
  中文一律写「**重算换显存**」；英文 `checkpoint` 在本库只留给**权重检查点**
  （正文在 `checkpointing/01-state-and-bytes`）。

★★ 本程序的数字全部是**推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 这是一个 `n ≤ 4096` 的**计数模型**，**不是**任何真实网络的显存测量。

★★ **模型为什么是线性的（口径声明，不是偷懒）**：
  层的形状取 `h ← h @ W_l`（`W_l` **正交**，由 QR 得到），loss 取 `L = c · h_n`。三个理由：
    1. **梯度不随深度消失**：正交矩阵保范 ⇒ `n=4096` 层的链也不上溢/下溢到次正规。
       本程序**先用**一个 `tanh` 版跑过一次，`n=1024` 上 `dW` 已经衰减到 `1e-54` ——
       那时「三条路径差 1e-54」是**次正规噪声**，不是证据（这处失败正是本模型改动的起因）；
    2. **存在一条解析参照**：`g_l = c·W_nᵀ⋯W_{l+1}ᵀ` 可用**一次反向前缀积**算出来
       ⇒ A 类不只有「另一个循环」，还有一条**闭式**路径；
    3. loss 的导数 `dL/dh_n = c` 是**常数** ⇒ 不必保留网络输出 ⇒
       峰值激活数就是 Chen 的 `g(n)` 口径（它数的是 **intermediate results**，不含边界张量）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  被测实现（√n 分段重算）与**解析闭式**（反向前缀积）逐元素一致（相对差 < 1e-12）；
           把重算的起点取错一格（`C.4`）立刻失败；
  · `B.1`  全存数 == `⌈√n⌉`（n=1024 → 32，**精确**）；
  · `B.2`  前向次数 ∈ `[1.8n, 2.1n]`，**且逐位等于**闭式 `2n − stored`（n=1024 → 2016）；
  · `C.1`  极端情形（k=1）：**内存** == `log₂ n` **逐位**，额外前向的形状名 == `"O(n log n)"`。

用法：
    python3 02_checkpointing.py
"""

import ast
import collections
import inspect
import itertools
import math
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
ASSERT_FAIL_MARK = ".. FAIL: autograd"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (":: FAIL: mixed-precision", "++ FAIL: optimizers",
                     "// FAIL: init-stability", ";; FAIL: checkpointing")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-5s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-5s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
# ★★ **口径声明（本主题最容易混的一处，计划 §4 T22）**：两个数**各有一个口径**：
#     · **内存** = 「同时存活的**激活张量个数**」 ⇒ `O(√n)`（原文该节标题 `An O(√n) Memory Cost Algorithm`）
#     · **代价** = 「**前向次数**」                ⇒ `≈ 2n`（原摘要 `an extra forward pass`）
#   ⇒ 本程序把它们**分开断言**，每处输出都标口径。★ D12 测的是**前向次数**。
N_LIST = (64, 128, 1024, 4096)  # 全部是 2 的幂（⇒ `log₂n` 是整数，判据 ⑤ 可逐位断言）
HIDDEN_T = 4                    # 玩具激活的宽度（不影响计数结论）
SEED_T = 20260924               # W 抽取的种子（★ 与 01 不同：两个程序各自独立）
RECURSIVE_K = 1                 # 极端情形用的 k（原文逐字：`if we set k = 1, we get g(n) = log_2 n`）

SEG_FWD_LO = 1.8                # B.2：前向次数 / n 的下界（计划 §3 表 A ④ 的区间）
SEG_FWD_HI = 2.1                # B.2：上界
EXTRA_FWD_LO = 0.8              # B.3：多出的前向 / n 的下界（对齐 `an extra forward pass`；
                                #  ★ n=64 上只有 0.875 —— 原摘要是**渐近**说法，见 B.3 输出）
EXTRA_FWD_HI = 1.1              # B.3：上界
MEM_FACTOR_TOL = 0.15           # B.4：峰值激活数 / √n 与 2 的相对容差
PAPER_VS_MODEL_GAP_MIN = 0.5    # B.5：论文的绝对值与本模型的比例至少差这么多（相对）
EXTREME_FWD_RATIO_MIN = 5.0     # C.2：极端方案的前向倍数至少是分段方案的这么多倍
BROKEN_MIN_DIFF = 1e-6          # C.4：重算起点取错一格后，梯度必须**至少**差这么多
DENOM_GAP_MIN = 1.5             # C.5：两个分母口径给出的「省了多少」至少差这么多倍
TOL_MACHINE_T = 1e-12           # A.1/A.2：路径之间（**机器精度档**）

CALLS = {}                      # ★ 参照实现独立性的证据：被测实现的函数每被调一次就 +1


# ===========================================================================
# 2. 模型与真值
# ===========================================================================
def make_case(n, seed=SEED_T):
    """造一个 `n` 层的实例：`h0`（宽度 `HIDDEN_T`）、`Ws`（**正交**）、`c`（loss 的系数）。

    ★ `Ws` 由对随机矩阵做 QR 得到 ⇒ **保范** ⇒ `n=4096` 层的链也不衰减（见文件头的口径声明）。
    """
    rng = np.random.default_rng(seed + n)
    h0 = rng.standard_normal(HIDDEN_T)
    c = rng.standard_normal(HIDDEN_T)
    R = rng.standard_normal((n, HIDDEN_T, HIDDEN_T))
    Q, _ = np.linalg.qr(R)                     # (n, D, D)，逐层正交
    return h0, list(Q), c


def segment_plan(n, k):
    """把 `n` 层切成若干段（每段至多 `⌈n/k⌉` 层），返回**真实切出来的**段长列表。"""
    seg = -(-n // k)
    lens = []
    left = n
    while left > 0:
        take = min(seg, left)
        lens.append(take)
        left -= take
    return lens


# ===========================================================================
# 3. 被测实现（路径 1）：√n 分段 —— 存段边界、段内**重算**
# ===========================================================================
def segmented_recompute(n, k, h0, Ws, c, boundary_shift=0):
    """**被测实现**：存每段的**输入**（含网络的输入 `h0`），段内**重算**内部激活。

    ★★ **两条口径声明**（缺了它们，「√n 的存储 / ≈2n 的前向」都不是可判定的命题）：

    **(1) 存储数 `stored`**：`h0` 也计一个「存下来的激活」——
      在真实网络里它同样是一张要留着的特征图（不能因为它是输入就免费）。
      这条口径使 `stored == ⌈√n⌉`（n=1024 → 32），与探测 D11 一致。

    **(2) 段内重算几个**：每段的**输入与输出都在手**（段的输出 = **下一段的输入**，
      就是已经存下来的那个边界）⇒ 段内只需重算 `li − 1` 个**内部**激活，不是 `li` 个。
      ⇒ 前向总次数 = `n + Σ(li − 1) = 2n − stored`（n=1024 → **2016**；n=4096 → **8128**）。
      ★ 若按「段内从头重算 `li` 个」算会得到 `2n`（n=1024 → 2048）——
        两个数**都**落在计划 §3 表 A ④ 的 `[1.8n, 2.1n]` 里，但**不是同一个数**；
        本程序按「输入输出都在手」这条（唯一自洽的）口径实现，并把差别打印出来。

    **(3) 峰值激活数**：**不含**网络的输出张量（它是 loss 的边界张量；Chen 的 `g(n)`
      数的是 intermediate results）。⇒ 峰值 = `k 个边界 + (段长 − 1) 个内部激活`。

    `boundary_shift=1` 是 `C.4` 的**反例**：故意用**上一段**的边界做重算起点。
    """
    CALLS["segmented_recompute"] = CALLS.get("segmented_recompute", 0) + 1
    lens = segment_plan(n, k)
    nseg = len(lens)
    dW = [np.zeros_like(W) for W in Ws]
    fwd = 0
    live = 0
    peak = 0

    bounds = [h0]                     # ← 段 0 的输入也算一个（口径 (1)）
    live += 1
    peak = max(peak, live)
    # ── 前向：逐段推进；每进入一段先把它的输入存下来（段 0 的输入已在上面的 bounds 里）
    cursor = 0
    h = h0
    for i, li in enumerate(lens):
        if i > 0:
            bounds.append(h)
            live += 1
            peak = max(peak, live)
        for _ in range(li):
            h = h @ Ws[cursor]
            cursor += 1
            fwd += 1

    # ── 反向：从最后一段往回，段内**重算内部激活**（这就是拿时间换显存的那一步）
    dh = c
    cursor = n
    for i in range(nseg - 1, -1, -1):
        li = lens[i]
        lo = cursor - li
        # ★ 重算的起点：`bounds` **按段号索引**（`bounds[i]` = 段 i 的输入，含 i=0 的 `h0`）
        #   —— 本程序第一版写成了 `bounds[i-1]`，于是每段都拿上一段的输入去重算，
        #   `A.1` 当场差 1.4~1.9（这就是「重算起点」这条口径的现场）。
        h_in = bounds[max(0, i - boundary_shift)]
        inner = []                                        # 段内内部激活（li − 1 个）
        hh = h_in
        for j in range(li - 1):
            hh = hh @ Ws[lo + j]
            inner.append(hh)
            fwd += 1
        live += len(inner)
        peak = max(peak, live)
        for j in range(li - 1, -1, -1):
            h_prev = h_in if j == 0 else inner[j - 1]
            # ★ 外积的**顺序**：`dW_l[i,k] = h_l[i] · g_{l+1}[k]` ⇒ 先 `h_prev` 后伴随。
            #   写反了**不会报错**（方阵的外积仍是同形状），只有数值参照能抓住 —— 见 `C.6`。
            dW[lo + j] += np.einsum("i,j->ij", h_prev, dh)   # ← 与路径 2 不同的写法
            dh = dh @ Ws[lo + j].T
        live -= len(inner)
        cursor = lo
    return dW, {"fwd": fwd, "peak": peak, "stored": len(bounds)}


# ===========================================================================
# 4. 参照实现（路径 2）：全部存下来（朴素做法）—— 不调用被测实现的任何函数
# ===========================================================================
def baseline_store_all(n, h0, Ws, c):
    """参照实现：**每层的输出都存下来**，反向时直接用。

    ★ 本函数一次都不调用 `segmented_recompute()` / `segment_plan()` / `_rec*`：
      它自己推进、自己存、自己反向（外积用 `np.outer`，不是被测那边的 `einsum`）。
    ★ 它是「重算换显存」要**换掉**的那个朴素做法，因此正是它的自然参照：
      两条路径的梯度必须一致 —— 否则被「换」掉的不只是显存。
    """
    dW = [np.zeros_like(W) for W in Ws]
    acts = []                        # 每层的**输入**（n 个）
    live = 0
    peak = 0
    h = h0
    for l in range(n):
        acts.append(h)
        live += 1
        peak = max(peak, live)
        h = h @ Ws[l]
    dh = c
    for l in range(n - 1, -1, -1):
        dW[l] += np.outer(acts[l], dh)          # ★ 顺序：`h_prev` 在前（同被测侧的 einsum）
        dh = dh @ Ws[l].T
    return dW, {"fwd": n, "peak": peak, "stored": len(acts)}


# ===========================================================================
# 5. 参照实现（路径 3）：**解析闭式** —— 反向前缀积
# ===========================================================================
def analytical_grad(n, h0, Ws, c):
    """解析闭式：`h_l = h_0 W_1⋯W_l`，`g_l = c W_nᵀ⋯W_{l+1}ᵀ`，`dW_l = g_l ⊗ h_{l-1}`。

    ★ 它与另外两条路径**结构上完全不同**：不逐层传播伴随，而是
      **先把全部前缀积算成一张表**（`itertools.accumulate`），再**一次性**取用。
      被测那边是「一边倒退一边把伴随传下去」，这里没有任何伴随传播。
    ★ 它同时是「梯度不随深度衰减」这条口径的**证据**：`‖g_l‖` 应恒为 `‖c‖`。
    """
    pref = list(itertools.accumulate(Ws, lambda a, W: a @ W, initial=np.asarray(h0)))
    # pref[l] = h_l（pref[0] = h_0，共 n+1 项）
    tails = [np.asarray(c)]
    for W in reversed(Ws):
        tails.append(tails[-1] @ W.T)          # tails[i] = c W_nᵀ⋯W_{n−i+1}ᵀ
    # g_{l+1} = tails[n−l−1]（`tails[i]` 有 i 个转置因子；本程序第一版把它写成了 `tails[n−l]`，
    # 整整错开一层 —— 是与另外三条路径对拍才发现的）。
    dW = [np.outer(pref[l], tails[n - l - 1]) for l in range(n)]
    return dW, tails


# ===========================================================================
# 6. 路径 4：**递归套娃**（原文 §4.4 的 `g(n) = k + g(n/(k+1))`）
# ===========================================================================
# ★ 原文 §4.4 逐字：`Let g(n) to be the memory cost to do forward and backward pass on
#   a n layer neural network. Assume that we store k intermediate results in the graph and
#   apply the same strategy recursively … g(n) = k + g(n/(k+1))` ⇒ `g(n) = k log_{k+1}(n)`；
#   `As a special case, if we set k = 1, we get g(n) = log_2 n`。
# ★★ 本程序把这条递推**真的实现成递归**（不是照抄结论）：
#   每一层做一次完整前向（n 次），然后对 k+1 个子段**倒序**各调一次自己 ⇒
#   前向次数 `H(n) = n + (k+1)·H(⌈n/(k+1)⌉)`；峰值内存 `G(n) = k + G(⌈n/(k+1)⌉)`。
_REC = {"fwd": 0, "live": 0, "peak": 0}
# ★ 只读快照：`_REC` 在跑动过程中会被改写，`D.10` 要断言的是它的**初始状态**
#   ⇒ 在最前面留一份副本。★ 它的右值是**调用**不是字面量 ⇒ 不进 `mutate.py` 的分母
#   （分母只收字面量叶子），所以它不制造新的漏网候选。
_REC_INIT = dict(_REC)


def _rec_unit(lo, Ws, h_in, dh, dW):
    """递归的叶子：一层的**前向 + 反向**（没有任何重算，也没有额外存储）。"""
    h_out = h_in @ Ws[lo]
    _REC["fwd"] += 1
    dW[lo] += np.outer(h_in, dh)
    return dh @ Ws[lo].T


def _rec(lo, hi, k, h_in, Ws, dh, dW):
    """对层区间 `[lo, hi)` 做一次前向 + 反向，**递归地**换内存。

    `dh` 是 `dL/dh_hi`（本区间**输出**的梯度）；返回 `dL/dh_lo`（**输入**的梯度）。
    """
    m = hi - lo
    if m <= 1:
        return _rec_unit(lo, Ws, h_in, dh, dW)
    # ★★ 原文的递推是 `g(n) = k + g(n/(k+1))` ⇒ 切 **k+1** 段（**不是** k 段）：
    #   存 k 个边界、递归进子段。写成 k 段会让 k=1 退化成「一段 = 整个 n」⇒ 不收敛
    #   （本程序第一次跑就撞上了这个 RecursionError）。
    seg = -(-m // (k + 1))
    lens = []
    left = m
    while left > 0:
        take = min(seg, left)
        lens.append(take)
        left -= take
    nseg = len(lens)
    bounds = []                              # 子段 1..nseg-1 的输入；子段 0 的输入是 h_in
    cursor = lo
    h = h_in
    for i, li in enumerate(lens):
        if i > 0:
            bounds.append(h)
            _REC["live"] += 1
            _REC["peak"] = max(_REC["peak"], _REC["live"])
        for _ in range(li):
            h = h @ Ws[cursor]
            cursor += 1
            _REC["fwd"] += 1
    # 倒序处理各子段：拿它的**边界**当输入递归进去；伴随**逐段往前传**
    starts = []
    acc = lo
    for li in lens:
        starts.append(acc)
        acc += li
    for i in range(nseg - 1, -1, -1):
        h_start = h_in if i == 0 else bounds[i - 1]
        dh = _rec(starts[i], starts[i] + lens[i], k, h_start, Ws, dh, dW)
    _REC["live"] -= len(bounds)
    return dh


def recursive_recompute(n, k, h0, Ws, c):
    """**极端情形**（原文的 `k=1` ⇒ `g(n)=log₂n`）：递归套娃重算。

    ★ 峰值内存 `G(n)` 与前向次数 `H(n)` 都由 `_rec` **真的跑出来**（不是套公式）。
    """
    dW = [np.zeros_like(W) for W in Ws]
    _REC.update(fwd=0, live=0, peak=0)
    _rec(0, n, k, h0, Ws, c, dW)
    return dW, {"fwd": _REC["fwd"], "peak": _REC["peak"], "stored": _REC["peak"]}


# ===========================================================================
# 7. 形状分类器（★ 给 `CHEN2016_LOG_N_SHAPES` 的两个叶子**各配一个消费者**）
# ===========================================================================
def shape_of(f, ns):
    """★ 从**实测的 f(n)** 判形状：对每个候选 `φ`，看 `f(n)/φ(n)` 跨 n 的相对极差，
    取极差最小的那个候选名。

    ★ 这是一个**判据式**，不是查表：`f` 是跑出来的数；候选名写在函数体里
      （因此**不在** `mutate.py` 的分母里 —— 它正是用来挡住「specs 的常量被改掉」的锚）。
    """
    cands = (
        ("O(log n)", lambda n: math.log2(n)),
        ("O(n log n)", lambda n: n * math.log2(n)),
        ("O(n)", lambda n: float(n)),
        ("O(sqrt n)", lambda n: math.sqrt(n)),
    )
    best, best_spread = None, None
    for name, phi in cands:
        r = np.array([f(n) / phi(n) for n in ns], dtype=float)
        spread = float((r.max() - r.min()) / abs(r.mean()))
        if best_spread is None or spread < best_spread:
            best, best_spread = name, spread
    return best


# ===========================================================================
# 8. 工具
# ===========================================================================
def _callee_names(func):
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def _maxdiff(a, b):
    return max(float(np.abs(np.asarray(x) - np.asarray(y)).max())
               for x, y in zip(a, b))


def _scale(dW):
    return max(float(np.abs(np.asarray(x)).max()) for x in dW)


# ===========================================================================
# 9. A 类：重算**不改变**结果
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 重算**不改变**结果：被测 / 全存 / 解析闭式 / 递归 四条路径")
    print("=" * 74)
    print("  口径：宽度 %d · W **正交**（QR）· L = c·h_n · seed 基 %d · n ∈ %s"
          % (HIDDEN_T, SEED_T, list(N_LIST)))
    worst_a = worst_all = 0.0
    for n in N_LIST:
        h0, Ws, c = make_case(n)
        d_ana, tails = analytical_grad(n, h0, Ws, c)
        d_seg, _ = segmented_recompute(n, int(math.ceil(math.sqrt(n))), h0, Ws, c)
        d_base, _ = baseline_store_all(n, h0, Ws, c)
        d_rec, _ = recursive_recompute(n, RECURSIVE_K, h0, Ws, c)
        sc = _scale(d_ana)
        a = _maxdiff(d_seg, d_ana) / sc
        b = max(_maxdiff(d_base, d_ana), _maxdiff(d_rec, d_ana)) / sc
        worst_a = max(worst_a, a)
        worst_all = max(worst_all, a, b)
        # ★ 解析闭式同时给出「梯度不随深度衰减」的证据：‖g_l‖ 恒为 ‖c‖
        gnorm = float(np.linalg.norm(tails[0]))
        print("     n=%-5d  被测 vs 解析 %.3e · 另两条 vs 解析 %.3e · max|dW| %.3f · ‖g_l‖=%.6f"
              % (n, a, b, sc, gnorm))
    check("A.1", worst_a < TOL_MACHINE_T,
          "★ 被测实现（√n 分段重算）与**解析闭式**（反向前缀积）逐元素一致"
          "（最坏相对差 %.3e < %g，机器精度档）" % (worst_a, TOL_MACHINE_T))
    check("A.2", worst_all < TOL_MACHINE_T,
          "★ 四条路径（被测 / 全存 / 解析 / 递归）全部一致（最坏相对差 %.3e < %g）"
          % (worst_all, TOL_MACHINE_T))

    # ── 参照实现的独立性（两条保险）────────────────────────────────────────
    CALLS.clear()
    n = N_LIST[-1]
    h0, Ws, c = make_case(n)
    analytical_grad(n, h0, Ws, c)
    baseline_store_all(n, h0, Ws, c)
    recursive_recompute(n, RECURSIVE_K, h0, Ws, c)
    check("A.3", not CALLS,
          "★ 三条参照路径跑完后**计数表完全为空**（记录到的调用：%s）⇒ "
          "它们一次都没调用被测实现" % (sorted(CALLS) if CALLS else "无"))

    watched = {"segmented_recompute"}
    called = (_callee_names(analytical_grad) | _callee_names(baseline_store_all)
              | _callee_names(recursive_recompute) | _callee_names(_rec)
              | _callee_names(_rec_unit))
    shared = sorted(watched & called)
    check("A.4", not shared,
          "★ **AST 调用表**：三条参照路径的 `Call` 节点里一个被测函数都没有（%s）"
          % (", ".join(sorted(called)) if called else "无外部调用"))
    return h0, Ws, c


# ===========================================================================
# 10. B 类：代价模型 —— 内存（√n）与前向次数（≈2n）**两个口径分开**
# ===========================================================================
def part_B(h0, Ws, c):
    print()
    print("=" * 74)
    print("B. 代价模型（★ 内存与前向次数**各标口径**，不可混用 —— 计划 §4 T22）")
    print("=" * 74)
    print("  %-6s %-8s %-9s %-9s %-11s %-10s %-10s" %
          ("n", "k=⌈√n⌉", "存下的", "⌈√n⌉", "峰值激活", "前向次数", "前向/n"))
    rows = {}
    for n in N_LIST:
        k = int(math.ceil(math.sqrt(n)))
        hh0, WWs, cc = make_case(n)
        _, st = segmented_recompute(n, k, hh0, WWs, cc)
        rows[n] = (k, st)
        print("  %-6d %-8d %-9d %-9d %-11d %-10d %-10.4f"
              % (n, k, st["stored"], k, st["peak"], st["fwd"], st["fwd"] / n))

    # ── 判据 ③：存下的 == ⌈√n⌉（**精确**）
    bad = {n: (r[1]["stored"], r[0]) for n, r in rows.items() if r[1]["stored"] != r[0]}
    check("B.1", not bad,
          "★ **内存口径**：全存数 == ⌈√n⌉（**精确**，n=1024 → %d；不符的：%s）"
          % (rows[1024][0], bad if bad else "无"))

    # ── 判据 ④：前向次数 ≈ 2n（**配精确闭式**）
    inband = all(SEG_FWD_LO * n <= rows[n][1]["fwd"] <= SEG_FWD_HI * n for n in N_LIST)
    closed = all(rows[n][1]["fwd"] == 2 * n - rows[n][1]["stored"] for n in N_LIST)
    print("  闭式：前向次数 == 2n − stored（★ 量级判据的证伪能力弱 ⇒ 配逐位闭式）")
    print("  ★ 口径对照：同一件事还有第二种数法（「段内从头重算 li 个」）⇒ 2n。本程序实现的是")
    print("    「段的输入与输出都在手」那一版；两个数**都** ∈ 判据区间，但 n=1024 上是")
    print("    2016 与 2048 —— **不是同一个数**（计划 §3 表 A ④ 引的是 2016）。")
    check("B.2", inband and closed,
          "★ **前向次数口径**：n=1024 → %d（%.4f×）、n=4096 → %d（%.4f×），"
          "全部 ∈ [%g n, %g n]，且**逐位等于**闭式 2n − stored"
          % (rows[1024][1]["fwd"], rows[1024][1]["fwd"] / 1024,
             rows[4096][1]["fwd"], rows[4096][1]["fwd"] / 4096, SEG_FWD_LO, SEG_FWD_HI))

    # ── 对齐原摘要的 `an extra forward pass`
    extra = {n: (rows[n][1]["fwd"] - n) / n for n in N_LIST}
    print("  多出的前向 / n：%s（n 越大越贴近 1.0 —— 原摘要是**渐近**说法）"
          % ", ".join("%.4f" % extra[n] for n in N_LIST))
    check("B.3", all(EXTRA_FWD_LO <= extra[n] <= EXTRA_FWD_HI for n in N_LIST),
          "★ 多出的前向 ≈ **一整个前向**（`CHEN2016_EXTRA_FORWARD_PASSES` = %d），"
          "比值全部 ∈ [%g, %g]"
          % (specs.CHEN2016_EXTRA_FORWARD_PASSES, EXTRA_FWD_LO, EXTRA_FWD_HI))

    # ── 内存系数的外部锚点：峰值激活数 / √n ≈ 2（原文 §4.3 的 O(2√n)）
    ratio = {n: rows[n][1]["peak"] / math.sqrt(n) for n in N_LIST}
    worst = max(abs(ratio[n] / specs.CHEN2016_MEMORY_COST_K_EQ_SQRT_N_FACTOR - 1.0)
                for n in N_LIST)
    print("  峰值激活数 / √n：%s"
          % ", ".join("%.4f" % ratio[n] for n in N_LIST))
    print("    （原文该节标题 `An O(√n) Memory Cost Algorithm`；式 (1) `cost-total = "
          "max_i cost-of-segment(i) + O(k) = O(n/k) + O(k)`")
    print("     紧接着 `Setting k = √n, we get the cost of O(2√n)` ⇒ **这个系数是内存的**）")
    check("B.4", worst < MEM_FACTOR_TOL,
          "★ **内存**峰值激活数 / √n 与 %.1f 的相对差最大 %.4f < %g"
          % (specs.CHEN2016_MEMORY_COST_K_EQ_SQRT_N_FACTOR, worst, MEM_FACTOR_TOL))

    # ── 与论文的**绝对值**不可比（两个数不是同一个量）
    n_paper = specs.CHEN2016_LAYERS
    kp = int(math.ceil(math.sqrt(n_paper)))
    hp0, WWp, cp = make_case(n_paper)
    _, stp = segmented_recompute(n_paper, kp, hp0, WWp, cp)
    model_ratio = n_paper / stp["peak"]
    hi_g, lo_g = specs.CHEN2016_MEMORY_GB_1000_LAYER
    paper_ratio = hi_g / lo_g
    gap = abs(model_ratio / paper_ratio - 1.0)
    print("  论文那一次实验（n=%d）：`48G to 7G` ⇒ %.3f×；本模型（同一 n）：%d / %d = %.3f×"
          % (n_paper, paper_ratio, n_paper, stp["peak"], model_ratio))
    print("    ★ 两个数**不是同一个量**：本模型的单位是「激活张量**个数**」，论文的是 **GB**")
    print("      （还含参数、梯度、临时缓冲，以及 §3 那张图分配优化的 `a factor of two to three`")
    print("      —— 那是**另一项**技术，不是重算换显存）。")
    check("B.5", stp["stored"] == kp and gap > PAPER_VS_MODEL_GAP_MIN,
          "★ n=%d 上存下 %d == ⌈√n⌉（形状对得上），但模型的收缩比 %.3f× 与论文的 %.3f× "
          "差 %.0f%% > %.0f%% ⇒ **绝对值不可比**"
          % (n_paper, stp["stored"], model_ratio, paper_ratio, gap * 100,
             PAPER_VS_MODEL_GAP_MIN * 100))

    # ── ★★ 口径分离：**内存 = O(√n)** 而 **前向次数 = O(n)**，各是各的
    mem_shape = shape_of(lambda m: int(math.ceil(math.sqrt(m))), N_LIST)
    fwd_shape = shape_of(lambda m: rows[m][1]["fwd"], N_LIST)
    print("  形状分类器（由实测数列判定）：内存 = %s · 前向次数 = %s" % (mem_shape, fwd_shape))
    check("B.6", mem_shape == "O(sqrt n)" and fwd_shape == "O(n)",
          "★ **口径必须分开写**：本方案的内存是 `√n`（%s）、前向次数是 `≈2n`（%s）——"
          "把 `O(2√n)` 读成「前向 2√n 次」是把两个口径混了（计划 §4 **T22**）"
          % (mem_shape, fwd_shape))

    # ── wall-clock 口径：与论文的 30% 相互印证（**推导值**，可手算）
    #    原文逐字：`the backward operation is nearly twice as time consuming as the forward one`
    #    ⇒ 记前向耗时 f、反向 2f：基线 = 3f；重算方案前向工作量翻倍 ⇒ 2f + 2f = 4f
    #    ⇒ wall-clock 应涨 4f/3f − 1 = 1/3 ≈ 33.3%，而**前向次数**涨的是 ≈100%
    derived = (2.0 * 2.0) / (1.0 + 2.0) - 1.0
    fwd_gain = rows[N_LIST[-1]][1]["fwd"] / N_LIST[-1] - 1.0
    print("  wall-clock 口径的**推导值**（前向工作量翻倍、反向 ≈ 2× 前向）"
          "= (2f+2f)/(f+2f) − 1 = %.4f" % derived)
    check("B.7", abs(derived - specs.CHEN2016_EXTRA_RUNTIME_FRACTION) < 0.05
          and abs(fwd_gain - specs.CHEN2016_EXTRA_RUNTIME_FRACTION) > 0.5,
          "★ 与原文的 `costs %g additional runtime` 相互印证（推导 %.4f，差 %.4f < 0.05）；"
          "而**前向次数**涨的是 %.0f%% ⇒ 两个口径差 %.2f > 0.5，不可互换"
          % (specs.CHEN2016_EXTRA_RUNTIME_FRACTION, derived,
             abs(derived - specs.CHEN2016_EXTRA_RUNTIME_FRACTION), fwd_gain * 100,
             abs(fwd_gain - specs.CHEN2016_EXTRA_RUNTIME_FRACTION)))
    return rows


# ===========================================================================
# 11. C 类：极端情形（判据 ⑤）、反例与口径边界
# ===========================================================================
def part_C(rows):
    print()
    print("=" * 74)
    print("C. 极端情形（递归套娃）与反例")
    print("=" * 74)
    k = RECURSIVE_K
    print("  递归套娃，k=%d（原文逐字：`if we set k = 1, we get g(n) = log_2 n`）" % k)
    print("  %-7s %-14s %-14s %-14s %-14s"
          % ("n", "峰值激活", "log₂n", "前向次数", "n(1+log₂n)"))
    mem_ok = fwd_ok = True
    fwds = {}
    for n in N_LIST:
        h0, Ws, c = make_case(n)
        _, st = recursive_recompute(n, k, h0, Ws, c)
        fwds[n] = st["fwd"]
        expect_fwd = int(n * (1 + math.log2(n)))
        mem_ok = mem_ok and st["peak"] == int(math.log2(n))
        fwd_ok = fwd_ok and st["fwd"] == expect_fwd
        print("  %-7d %-14d %-14d %-14d %-14d"
              % (n, st["peak"], int(math.log2(n)), st["fwd"], expect_fwd))

    # ── 判据 ⑤ ★ 内存的形状名 == `CHEN2016_LOG_N_SHAPES[0]`
    mem_shape = shape_of(lambda m: int(math.log2(m)), N_LIST)
    # ── 额外前向开销的形状名 == `CHEN2016_LOG_N_SHAPES[1]`
    extra_shape = shape_of(lambda m: fwds[m] - m, N_LIST)
    print("  额外前向开销 = 前向次数 − n：%s" % ", ".join("%d" % (fwds[n] - n) for n in N_LIST))
    print("  形状分类器：内存 = %r · 额外前向开销 = %r" % (mem_shape, extra_shape))
    check("C.1", mem_ok and fwd_ok
          and mem_shape == specs.CHEN2016_LOG_N_SHAPES[0]
          and extra_shape == specs.CHEN2016_LOG_N_SHAPES[1],
          "★ 判据 ⑤（极端情形）：**内存** == log₂n（逐位）且形状名 == %r；"
          "**额外前向开销** == n·log₂n（逐位）且形状名 == %r"
          % (specs.CHEN2016_LOG_N_SHAPES[0], specs.CHEN2016_LOG_N_SHAPES[1]))

    print()
    print("  ★★ 本程序**记录**原文内部的一处不一致（不做修正，只如实记下）：")
    print("     摘要逐字：`In the extreme case, our analysis also shows that the memory")
    print("       consumption can be reduced to O(log n) with as little as O(n log n) extra")
    print("       cost for forward computation.`   ⇒ **额外前向 = O(n log n)**")
    print("     §4.4 逐字：`This will require O(log_2 n) cost forward pass cost, so may not be")
    print("       used commonly.`                  ⇒ **前向代价 = O(log₂ n)**")
    print("     两者**不是同一个量级**。本程序的递推 `H(n) = n + (k+1)·H(⌈n/(k+1)⌉)` 真的跑出")
    print("     `H(n) = n(1 + log₂ n)` ⇒ **与摘要那一句一致**。此处只记录，不判谁对。")

    # ── C.2 ★ 反例：极端方案「省」得更多，但**不是免费的**
    n = N_LIST[-1]
    rec_ratio = fwds[n] / n
    seg_ratio = rows[n][1]["fwd"] / n
    print()
    print("  C.2：n=%d 上 —— 分段 %.4f×前向 vs 递归 %.4f×前向；而峰值内存 %d vs %d"
          % (n, seg_ratio, rec_ratio, rows[n][1]["peak"], int(math.log2(n))))
    check("C.2", rec_ratio > EXTREME_FWD_RATIO_MIN * seg_ratio,
          "★ 递归方案把**峰值内存**从 %d 压到 %d，代价是**前向次数**从 %.2f× 涨到 %.2f×"
          "（%.1f× 于分段方案 > %g）—— 「省」在别处记账"
          % (rows[n][1]["peak"], int(math.log2(n)), seg_ratio, rec_ratio,
             rec_ratio / seg_ratio, EXTREME_FWD_RATIO_MIN))

    # ── C.3 ★ 反例：k 取得太大**不会**更省（退化回全存）
    h0, Ws, c = make_case(n)
    _, st_big = segmented_recompute(n, n, h0, Ws, c)
    _, st_seg = segmented_recompute(n, int(math.ceil(math.sqrt(n))), h0, Ws, c)
    print()
    print("  C.3：k=n（每段 1 层）⇒ 峰值激活 %d · 前向 %d；而 k=⌈√n⌉ ⇒ 峰值 %d · 前向 %d"
          % (st_big["peak"], st_big["fwd"], st_seg["peak"], st_seg["fwd"]))
    check("C.3", st_big["peak"] >= n and st_big["peak"] > st_seg["peak"],
          "★ k 取到 n **不会**更省：峰值激活 %d ≥ n=%d，反而比 √n 的 %d 差 —— "
          "`√n` 是**最优点**，不是「越大越好」" % (st_big["peak"], n, st_seg["peak"]))

    # ── C.4 ★★ 反例：重算起点取错一格 ⇒ 梯度**真的不同**（⇒ A.1 的相等不是恒真）
    got, _ = segmented_recompute(n, int(math.ceil(math.sqrt(n))), h0, Ws, c,
                                 boundary_shift=1)
    d_ana, _ = analytical_grad(n, h0, Ws, c)
    diff = _maxdiff(got, d_ana) / _scale(d_ana)
    print()
    print("  C.4：把重算的起点**取错一格**（用上一段的边界代替本段的）后，"
          "与解析闭式的相对差 = %.3e" % diff)
    check("C.4", diff > BROKEN_MIN_DIFF,
          "★ 重算的起点错一格 ⇒ 梯度相对差 %.3e > %g ⇒ A.1 的相等**不是恒真**："
          "它真的在检查「重算从哪个边界开始」" % (diff, BROKEN_MIN_DIFF))

    # ── C.5 ★ 口径边界：同一个「省了多少」，两个分母差约 2 倍
    r1 = n / rows[n][1]["stored"]
    r2 = n / rows[n][1]["peak"]
    print()
    print("  C.5：「省了多少倍」有两个分母 —— ")
    print("     分母取**存下来的个数**：n/%d = %.2f×（原文摘要那句按这个口径说）"
          % (rows[n][1]["stored"], r1))
    print("     分母取**峰值激活数**：  n/%d = %.2f×" % (rows[n][1]["peak"], r2))
    check("C.5", r1 / r2 > DENOM_GAP_MIN,
          "★ 同一个「省了多少」，两个分母给出 %.2f× 与 %.2f×（差 %.2f 倍 > %g）——"
          "**报这个数必须写分母**" % (r1, r2, r1 / r2, DENOM_GAP_MIN))

    # ── C.6 ★★ 反例：外积的**顺序**写反 —— 不报错、形状也对，只有数值参照能抓
    #    `dW_l[i,k] = h_l[i]·g_{l+1}[k]`（`h_prev` 在前）。`np.outer(g, h)` 是它的**转置**，
    #    方阵下形状完全相同 ⇒ mu^2 行代码里静默错、且「自己跟自己比」永远一致。
    #    本程序第一版就写反了，是与解析闭式对拍才发现的（`A.1` 当时差 1.79）。
    swapped = [np.asarray(W).T for W in d_ana]
    sd = _maxdiff(swapped, d_ana) / _scale(d_ana)
    print()
    print("  C.6：把外积的两个操作数**写反**（`outer(g, h)` 而不是 `outer(h, g)`）："
          "形状 %s 完全相同" % (d_ana[0].shape,))
    check("C.6", sd > 1e-3,
          "★ 写反后梯度相对差 %.3e > %g ⇒ 这种错**不会自己暴露**（形状、类型、"
          "「两条路径互相一致」全都看不出来）—— 必须有一条第**独立**的参照" % (sd, 1e-3))
    return fwds


# ===========================================================================
# 12. D 类：失败标记自检 + specs 钉住（含 `UNSOURCED` 与 `LOG_N_SHAPES` 的消费）
# ===========================================================================
def preflight():
    """★★ **D 类的前半段先跑**：标记自检 + 口径/阈值钉住（理由同 `01_backward.py`）。

    ★ 本程序第一次 `make mutate` 时有 **2 条**崩溃捕获（`N_LIST[2]` / `N_LIST[3]` ——
      它们会让 `rows[1024]` 抛 `KeyError`）：把钉住提到计算之前，它们就变成断言捕获。
    """
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ".. FAIL: autograd" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (":: FAIL: mixed-precision", "++ FAIL: optimizers",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**逐位钉住 —— 写错一个字，这条检查就静默失效")

    measured = {"N_LIST": N_LIST, "HIDDEN_T": HIDDEN_T, "SEED_T": SEED_T,
                "RECURSIVE_K": RECURSIVE_K}
    expect = {"N_LIST": (64, 128, 1024, 4096), "HIDDEN_T": 4, "SEED_T": 20260924,
              "RECURSIVE_K": 1}
    diff = {kk: (expect[kk], v) for kk, v in measured.items() if v != expect[kk]}
    check("D.3", not diff, "★ 口径常量 4 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    th = {"SEG_FWD_LO": SEG_FWD_LO, "SEG_FWD_HI": SEG_FWD_HI,
          "EXTRA_FWD_LO": EXTRA_FWD_LO, "EXTRA_FWD_HI": EXTRA_FWD_HI,
          "MEM_FACTOR_TOL": MEM_FACTOR_TOL,
          "PAPER_VS_MODEL_GAP_MIN": PAPER_VS_MODEL_GAP_MIN,
          "EXTREME_FWD_RATIO_MIN": EXTREME_FWD_RATIO_MIN,
          "BROKEN_MIN_DIFF": BROKEN_MIN_DIFF, "DENOM_GAP_MIN": DENOM_GAP_MIN,
          "TOL_MACHINE_T": TOL_MACHINE_T}
    th_expect = {"SEG_FWD_LO": 1.8, "SEG_FWD_HI": 2.1, "EXTRA_FWD_LO": 0.8,
                 "EXTRA_FWD_HI": 1.1, "MEM_FACTOR_TOL": 0.15,
                 "PAPER_VS_MODEL_GAP_MIN": 0.5, "EXTREME_FWD_RATIO_MIN": 5.0,
                 "BROKEN_MIN_DIFF": 1e-6, "DENOM_GAP_MIN": 1.5, "TOL_MACHINE_T": 1e-12}
    tdiff = {kk: (th_expect[kk], v) for kk, v in th.items() if v != th_expect[kk]}
    check("D.4", not tdiff, "★ 阈值 10 个逐位一致（含**机器精度档** TOL_MACHINE_T 与"
                            "计划 §3 表 A ④ 的区间 [1.8n, 2.1n]）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)
    # ── ★★ `specs.py` 的常量：**逐位钉住**（这就是它们的消费者）─────────────
    print("  specs 常量逐位钉住（★ 没有消费者的常量只能是 `make mutate` 的漏网候选）：")
    check("D.5", specs.CHEN2016_MEMORY_GB_1000_LAYER == (48, 7)
          and specs.CHEN2016_LAYERS == 1000
          and specs.CHEN2016_MEMORY_COST_K_EQ_SQRT_N_FACTOR == 2.0
          and specs.CHEN2016_EXTRA_FORWARD_PASSES == 1
          and specs.CHEN2016_EXTRA_RUNTIME_FRACTION == 0.30,
          "★ Chen 2016 的五个常量逐位一致（内存 GB / 层数 / O(2√n) 的系数 / "
          "多出的前向 / wall-clock 比例）")
    check("D.6", specs.CHEN2016_LOG_N_SHAPES == ("O(log n)", "O(n log n)"),
          "★ `CHEN2016_LOG_N_SHAPES` 两个叶子**都有消费者**（`C.1` 用 `==` 逐位比对，"
          "含 `[1]`）—— 这正是计划 §4 **T25** 要的：不给它开后门，而是给它一个消费者")
    # ★★ `UNSOURCED` 也必须有消费者（`45-…/code/specs.py:107-113` 的教训：
    #    「改掉也无人发现」的常量）。照 `80` 的做法**逐位钉住**。
    check("D.7", specs.UNSOURCED == ("chen2016_exact_segmented_forward_count",
                                    "torch_clip_coef_eps_version_span"),
          "★ `specs.UNSOURCED` 的两项**逐位钉住**（%d 项）：它是「本域不写、但记下为什么」"
          "的待办清单 —— 删一条、加一条、改一个字都会失败" % len(specs.UNSOURCED))

    check("D.8", inspect.getsource(segmented_recompute)
          != inspect.getsource(baseline_store_all)
          and inspect.getsource(baseline_store_all)
          != inspect.getsource(analytical_grad),
          "★ 防 P17：被测 / 全存 / 解析 三条的**源码两两不同**（%d / %d / %d 字节）"
          % (len(inspect.getsource(segmented_recompute)),
             len(inspect.getsource(baseline_store_all)),
             len(inspect.getsource(analytical_grad))))
    check("D.9", "CALLS[" in inspect.getsource(segmented_recompute),
          "★ 被测侧**真的**带计数 ⇒ `A.3` 的「计数表为空」不是空断言")
    # ★★ T11 盲区① 的正面落法：`_REC` 是「只用于累计的模块级常量表」，
    #   它的**初始值**在 `recursive_recompute()` 里会被 `update()` 覆盖 ⇒
    #   改它**不会**影响任何结论 ⇒ `make mutate` 会记 6 条漏网（三个键 + 三个值）。
    #   处置不是给它开后门，而是**给它一个消费者**：把初始状态本身断言下来
    #   （「计数表必须从 0 开始，且恰好有这三个键」是这条计数的口径的一部分）。
    check("D.10", _REC_INIT == {"fwd": 0, "live": 0, "peak": 0},
          "★ 递归计数表的**初始状态**逐位钉住（%r，跑动后现在 = %r）：三个键 + 三个 0 —— "
          "改键名会 `KeyError`、改初值会让「从 0 开始数」这句话失去保证"
          % (_REC_INIT, _REC))
    # ★ 这条只能在最后查：`MARKS_USED` 要等所有 part 跑完才有内容。
    check("D.11", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    print("     本程序**没有**为「重算换显存」之外的任何主张下结论：")
    print("     `mixed-precision` 的类型选择、`checkpointing` 的字节账都不在这里（回链）。")


# ===========================================================================
def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


def main():
    print("=" * 74)
    print("02_checkpointing.py —— 重算换显存：√n 的内存 / ≈2n 的前向 / log n 的极端情形")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这是 `n ≤ %d` 的**计数模型**，"
          % N_LIST[-1])
    print("    **不是**任何真实网络的显存测量。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 术语：文件名里的 `checkpointing` 是 Chen 2016 的术语，指**重算换显存**，")
    print("    **不是权重检查点**（那是 `checkpointing/01-state-and-bytes` 的主题）。")
    print("  ★ 禁区（回链，不重推）：`grad_fn` / `is_leaf` / `saved_tensors` 逐算子表 /")
    print("    `no_grad` vs `inference_mode` / `retain_graph` —— 正文在")
    print("    `70-frameworks/pytorch/02-autograd.md`。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆"
              "无意义的连带崩溃")
        return summary()
    h0, Ws, c = part_A()
    rows = part_B(h0, Ws, c)
    part_C(rows)
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
