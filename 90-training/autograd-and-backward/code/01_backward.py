#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_backward.py —— 反向传播本身（链式法则 / vjp / 伴随），以及**数值梯度对拍的方法学**。

配套文档：`01-backward-from-scratch.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ 本程序的数字**全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ 本主题的**禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）：
  `grad_fn` 的结构、`is_leaf`、`saved_tensors` 逐算子表、`no_grad` vs `inference_mode`、
  `retain_graph` 的默认值、`backward()` vs `autograd.grad()` 的差别 ——
  **一律回链** `70-frameworks/pytorch/02-autograd.md`（**全篇 256 行**）
  与 `00-meta/glossary.md:203-211`，本程序**一行都不写**。
  自定义反向的**实现**回链 `70-frameworks/tensorflow/03-custom-ops-and-export.md:106-150,217`；
  控制流与反向模式的可微性回链 `70-frameworks/jax/03-sharding-and-differences.md:110-112,379-381`；
  不可微算子的反向回链 `130-quantization/qat/01-ste.md`。
  ⇒ 本程序只做两件事：**从零写一遍反向传播**，以及**给它配一个能证伪的数值参照**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  两条**结构上完全不同**的解析路径（矩阵公式 vs 显式 tape 的伴随扫描）逐元素一致
           （相对差 < 1e-14，机器精度档）—— 把 tape 里任一 vjp 改成错的立刻失败；
  · `A.2`  手写反向 vs **中心差分**（数值参照）：**两个口径**都 < 1e-5
           （口径 a：`max|Δ| / max|g|`；口径 b：逐元素相对，= 探测脚本 D1 的口径）——
           把 `tanh'` 写成 `1 - t`（少一个平方）立刻失败；
  · `B.1`  中心差分的前向**次数**恰好 `2P`（P=54 ⇒ 108 次），反向模式 1 次 ⇒ 比值 108 ≥ 100 ——
           少算一次前向立刻失败；
  · `C.1`  `h` 太大（1e-1）时中心差分的误差**真的**超阈值；`C.2` `h` 太小（1e-12）时**也**超
           —— 即「对拍必须扫 h，不能拍一个值」，两个方向都真的会犯。

用法：
    python3 01_backward.py
"""

import ast
import collections
import inspect
import math
import os
import sys

import numpy as np

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（autograd-and-backward）取 `.. FAIL: autograd`。
#   ★ 这个字面量**不是**照抄任何别的主题的：域 README 把五个主题的标记写死为
#     五个不同形态（本主题用 `..`），**不跨主题照抄**是本库纪律。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；
#     `D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ".. FAIL: autograd"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (":: FAIL: mixed-precision", "++ FAIL: optimizers",
                     "// FAIL: init-stability", ";; FAIL: checkpointing")

FAILS = []
MARKS_USED = set()
# ★ 计数器用 `Counter()` 而**不是**字面量字典：`mutate.py` 的分母是「模块级常量的
#   **字面量叶子**」，写成 `{"A": 0, ...}` 会把不承载判据的初值塞进分母（灌水）。
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。"""
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
# ⚠ 本程序**不 import specs**（`specs.py`）：本主题的三个 `specs.py` 常量
#   （Chen 2016 的内存/前向账、PyTorch 的裁剪 `1e-6`）都属 `02` 与 `03` 的判据，
#   本程序的真值是**数值参照**（中心差分 / tape），**不是外部规格** ⇒
#   强行搬一个常量进来只会制造「照抄但不使用」的漏网候选（计划 §4 T11 盲区①）。
#   ★ 这条是**刻意**的，不是遗漏；见 §5 判据表与文档的「参考」节。

M_ROWS = 6              # 隐层宽度 m
D_COLS = 8              # 输入宽度 d
N_W1 = 48               # == M_ROWS * D_COLS（`D.3` 里逐位钉住）
N_W2 = 6                # == M_ROWS
N_PARAMS = 54           # == N_W1 + N_W2（= 中心差分要探的自由度，也是元素个数）
SEED = 20260923         # 参数抽取的种子（★ 与探测脚本 probe_90.py 的 rng(90) **不同**）
W_SCALE = 0.5           # W1/W2 ~ N(0,1) * W_SCALE
# ★ 输入向量**全正数**存放：负字面量是 `UnaryOp(USub, Constant(…))`，
#   `mutate.py` 的 `_walk` 对它返回空 ⇒ **不进分母**（计划 §3 开头的盲区②）。
#   这里刻意避开，使这 5 个叶子真的被变异到。
X_VEC = (0.3, 0.7, 1.1, 0.2, 0.4, 0.9, 0.5, 0.8)

H_STEP = 1e-6           # 中心差分的默认步长（判据 ①② 用的就是它）
# 步长扫描表（C.1/C.2 用）：两端各有一个**失效区**，中间才是可用的窗口。
H_SWEEP = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12)

TOL_CENTRAL_REL = 1e-5      # 判据 ①：手写反向 vs 中心差分（**计划 §3 表 A 的阈值**）
TOL_MACHINE_REL = 1e-14     # 判据 ②：两条解析路径之间（**机器精度档**）
TOL_ONE_SIDED_MIN = 10.0    # C.4：单侧差分的误差至少比中心差分大这么多倍
TAIL_BAD_FACTOR = 1000.0    # C.1/C.2：两端失效区的误差至少是最优 h 处的这么多倍
FWD_RATIO_MIN = 100.0       # B.1：中心差分的前向次数 / 反向模式的前向次数
CALLS = {}                  # ★ 参照实现独立性的**证据**：被测实现的函数每被调一次就 +1


# ===========================================================================
# 2. 被测实现（路径 1）：矩阵形式的链式法则
# ===========================================================================
def forward_formula(w):
    """路径 1 的前向。`w` 是**扁平**的参数向量（先 `W1` 后 `W2`）。

    模型：`L(w) = W2 · tanh(W1 x)`，`W1 ∈ R^{m×d}`、`W2 ∈ R^m`、`x` 固定。
    `L` 对 `W2` 的梯度 = `tanh(W1 x)`，对 `z = W1 x` 的梯度 = `W2 ⊙ (1 - tanh²z)`。
    """
    CALLS["forward_formula"] = CALLS.get("forward_formula", 0) + 1
    W1 = w[:N_W1].reshape(M_ROWS, D_COLS)
    W2 = w[N_W1:]
    x = np.array(X_VEC)
    z = W1 @ x
    t = np.tanh(z)
    return float(W2 @ t), W1, W2, x, z, t


def backward_formula(w):
    """路径 1 的**反向**（vjp 的伴随形式）：

    反向模式的核心是**一次**从标量出发的伴随扫描 —— 每一步只做一次「转置乘」：
        L → (dL/dy = 1) → dL/dW2 = t · 1, dL/dt = W2 · 1
        → dL/dz = dL/dt ⊙ tanh'(z) → dL/dW1 = dL/dz ⊗ x
    ★ 注意 `dL/dW1` 是一个**外积**（`np.outer`）：这正是「伴随 × 输入」的形状，
      不是「梯度 × 梯度」—— 写成内积或逐元素乘都是错的（`A.2` 会抓住）。
    """
    CALLS["backward_formula"] = CALLS.get("backward_formula", 0) + 1
    _, W1, W2, x, z, t = forward_formula(w)
    dW2 = t                                   # dL/dW2 = t（因为 dL/dy = 1）
    dz = W2 * (1.0 - t * t)                   # tanh'(z) = 1 - tanh²(z)
    dW1 = np.outer(dz, x)                     # dL/dW1 = (dL/dz) ⊗ x
    return np.concatenate([dW1.ravel(), dW2])


# ===========================================================================
# 3. 参照实现**路径 2**：显式 tape 的伴随扫描（不调用路径 1 的任何函数）
# ===========================================================================
# ★ `CONTRIBUTING:232-235`：A 类必须有一条与被检验代码路径**不同**的参照；
#   `00-meta/glossary.md:259` 给出机械判据：**参照实现不得调用被测实现的任何函数**。
# ★★ 两条路径的差别是**真的**，不是换了个名字：
#     · 路径 1：闭式矩阵表达式（`@`、`np.outer`），一步到位；
#     · 路径 2：**先建一张 tape**（每一步登记自己的局部 vjp），再**从标量反向扫一遍**
#       累加伴随 —— 它不知道整体公式长什么样，只知道每一步的局部规则。
#       把 `M_ROWS` 或模型结构换掉，路径 1 要重推，路径 2 只要加一条 op。
def grad_by_tape(w):
    """独立参照：把 `L = W2 · tanh(W1 x)` 建成一条 tape，反向扫出全部伴随。

    ★ 本函数**一次都不调用** `forward_formula()` / `backward_formula()`：
      它自己做前向（每个 op 登记一条局部 vjp），反向时按登记顺序倒着走。
    """
    W1 = w[:N_W1].reshape(M_ROWS, D_COLS)
    W2 = w[N_W1:]
    x = np.array(X_VEC)

    vals = {}          # id -> 这个节点的值
    rules = []         # [(输出的 id, [(父亲的 id, 局部 vjp)])]，按创建顺序
    counter = [0]

    def node(v):
        counter[0] += 1
        vals[counter[0]] = np.asarray(v, dtype=float)
        return counter[0]

    kx = node(x)
    kw1 = node(W1)
    kw2 = node(W2)
    kz = node(W1 @ x)
    rules.append((kz, [(kw1, lambda g: np.outer(g, x)),
                       (kx, lambda g: W1.T @ g)]))
    kt = node(np.tanh(vals[kz]))
    rules.append((kt, [(kz, lambda g: g * (1.0 - vals[kt] ** 2))]))
    ky = node(W2 @ vals[kt])                    # 标量 L
    rules.append((ky, [(kw2, lambda g: g * vals[kt]),
                       (kt, lambda g: g * W2)]))

    adj = {ky: 1.0}                             # 从标量出发，伴随初值 = 1
    for out, parents in reversed(rules):        # ← 反向扫
        gout = adj.get(out)
        if gout is None:
            continue
        for pid, local in parents:
            adj[pid] = adj.get(pid, 0.0) + local(gout)
    return np.concatenate([np.asarray(adj[kw1]).ravel(), np.asarray(adj[kw2]).ravel()])


# ===========================================================================
# 4. 参照实现**路径 3**：中心差分（数值参照，自带一条独立的前向）
# ===========================================================================
def loss_of_w(w):
    """路径 3 的**独立**前向：逐行点积 + 标量 `math.tanh`。

    ★ 它与路径 1 的 **数学定义相同**（对拍的前提），但**一行代码都不共享**：
      路径 1 用 `np.tanh(W1 @ x)`（矩阵形式、向量化），这里用 `for` 循环 + `np.dot`
      + 标准库的 `math.tanh`（标量）。`D.5`/`D.6` 两条断言守着这条独立性。
    """
    W1 = w[:N_W1].reshape(M_ROWS, D_COLS)
    W2 = w[N_W1:]
    acc = 0.0
    for i in range(M_ROWS):
        acc += float(W2[i]) * math.tanh(float(np.dot(W1[i], X_VEC)))
    return acc


def central_difference_grad(w, h):
    """路径 3：`(g, 前向次数)`。中心差分每个自由度要 **2** 次前向（+h 与 -h）。"""
    g = np.zeros_like(w)
    evals = 0
    for i in range(w.size):
        wp = w.copy()
        wm = w.copy()
        wp[i] += h
        wm[i] -= h
        g[i] = (loss_of_w(wp) - loss_of_w(wm)) / (2.0 * h)
        evals += 2
    return g, evals


def one_sided_grad(w, h):
    """C.3 的对照物：**单侧**前向差分（只有截断误差的**一阶**项被消掉）。"""
    g = np.zeros_like(w)
    for i in range(w.size):
        wp = w.copy()
        wp[i] += h
        g[i] = (loss_of_w(wp) - loss_of_w(w)) / h
    return g


# ===========================================================================
# 5. 工具
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    先例：`make lint` 与 `mutate.py` 都走 AST 而不是 grep —— 注释里出现一个名字
    和真的调用它是两件事。
    """
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    """用 **AST** 从本文件里数出模块级的 `ASSERT_FAIL_MARK = <字符串字面量>`。"""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def _relerr(a, b):
    """两个口径**都**给：`max|Δ| / max|参照|`，以及逐元素相对（探测脚本 D1 的口径）。

    ★★ 口径是独立的一环（本库 m03/m10 那一类错误）：同一个「相对差」有两种写法，
      它们在**近零元素**上差好几个数量级 —— 报数必须写明用的是哪一个。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (float(np.abs(a - b).max()) / float(np.abs(b).max()),
            float(np.max(np.abs(a - b) / (np.abs(b) + 1e-30))))


def make_params():
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(N_PARAMS) * W_SCALE


# ===========================================================================
# 6. A 类：反向传播本身
# ===========================================================================
def part_A(w):
    print("=" * 74)
    print("A. 反向传播本身：两条解析路径 + 数值参照")
    print("=" * 74)
    print("  口径：m=%d · d=%d · P=%d（= %d + %d）· seed=%d · W~N(0,1)·%g · x=%s"
          % (M_ROWS, D_COLS, N_PARAMS, N_W1, N_W2, SEED, W_SCALE, list(X_VEC)))

    g_formula = backward_formula(w)
    g_tape = grad_by_tape(w)
    rel_a, rel_e = _relerr(g_formula, g_tape)
    print("  路径 1（矩阵公式）与路径 2（tape 伴随扫描）：")
    print("     口径 a  max|Δ|/max|·| = %.3e" % rel_a)
    print("     口径 b  逐元素相对      = %.3e" % rel_e)
    check("A.1", rel_a < TOL_MACHINE_REL,
          "★ 两条**结构上完全不同**的解析路径逐元素一致（口径 a %.3e < %g，"
          "机器精度档）" % (rel_a, TOL_MACHINE_REL))

    g_num, evals = central_difference_grad(w, H_STEP)
    rel_a_n, rel_e_n = _relerr(g_num, g_formula)
    print("  路径 1 与路径 3（中心差分，h=%g）：" % H_STEP)
    print("     口径 a  max|Δ|/max|·| = %.3e" % rel_a_n)
    print("     口径 b  逐元素相对      = %.3e   ← ★ 探测脚本 probe_90.py 的 D1 用的是**这一个**"
          % rel_e_n)
    print("     （D1 在它的口径/种子上给 1.061e-07；本程序换了 seed 与 x，"
          "两个数**不是同一个测量**）")
    check("A.2", rel_a_n < TOL_CENTRAL_REL and rel_e_n < TOL_CENTRAL_REL,
          "★ 手写反向 vs 中心差分：两个口径都 < %g（口径 a %.3e · 口径 b %.3e）"
          % (TOL_CENTRAL_REL, rel_a_n, rel_e_n))

    # ── ★★ 参照实现的独立性（两条保险）────────────────────────────────────
    CALLS.clear()
    grad_by_tape(w)
    central_difference_grad(w, H_STEP)
    one_sided_grad(w, H_STEP)
    check("A.3", not CALLS,
          "★ 三条参照路径跑完后**计数表完全为空**（记录到的调用：%s）⇒ "
          "它们一次都没调用被测实现的函数" % (sorted(CALLS) if CALLS else "无"))

    watched = {"forward_formula", "backward_formula"}
    called = (_callee_names(grad_by_tape) | _callee_names(central_difference_grad)
              | _callee_names(loss_of_w) | _callee_names(one_sided_grad))
    shared = sorted(watched & called)
    check("A.4", not shared,
          "★ **AST 调用表**：三条参照路径的 `Call` 节点里一个被测函数都没有（%s）"
          % (", ".join(sorted(called)) if called else "无外部调用"))
    return g_formula, g_num, evals


# ===========================================================================
# 7. B 类：代价模型 —— 数值梯度贵在哪
# ===========================================================================
def part_B(evals):
    print()
    print("=" * 74)
    print("B. 代价模型：中心差分的前向次数 vs 反向模式的前向次数")
    print("=" * 74)
    print("  被测的自由度 P = %d（%d 个 W1 元素 + %d 个 W2 元素）" % (N_PARAMS, N_W1, N_W2))
    print("  中心差分的实测前向次数 = %d（= 2P）" % evals)
    print("  反向模式：1 次前向 + 1 次反向扫描 ⇒ 记 **1** 次前向（扫描不是前向）")
    print("  ⇒ 比值 = %.1f×" % (evals / 1.0))
    check("B.1", evals == 2 * N_PARAMS and evals / 1.0 >= FWD_RATIO_MIN,
          "★ 数值梯度要 %d 次前向（== 2P）、反向模式要 1 次 ⇒ %.0f× ≥ %g"
          % (evals, evals / 1.0, FWD_RATIO_MIN))

    print()
    print("  ★ 口径提示：这里数的是**前向次数**（本域 B 类的两个口径之一）。")
    print("    它与 wall-clock 不是一回事 —— Chen 2016 逐字写着")
    print("    `the backward operation is nearly twice as time consuming as the forward one`，")
    print("    所以「前向翻倍」在墙上时钟里并不是翻倍。那个换算的正文在")
    print("    `02-gradient-checkpointing.md`（本程序不重推）。")


# ===========================================================================
# 8. C 类：边界与反例 —— 数值梯度的步长窗口
# ===========================================================================
def part_C(w, g_formula):
    print()
    print("=" * 74)
    print("C. 边界与反例：中心差分有一个**双向**失效区")
    print("=" * 74)
    print("  h 扫描（口径 a，基准 = 手写反向）：")
    errs = {}
    for h in H_SWEEP:
        g, _ = central_difference_grad(w, h)
        errs[h] = _relerr(g, g_formula)[0]
        print("     h=%-8.0e  误差 %.3e%s" % (h, errs[h],
                                            "   ← ★ 失效区（太大）" if h >= 1e-1 else
                                            "   ← ★ 失效区（太小）" if h <= 1e-12 else ""))
    best = min(errs, key=lambda k: errs[k])

    check("C.1", errs[1e-1] / errs[best] > TAIL_BAD_FACTOR
          and errs[1e-1] > TOL_CENTRAL_REL,
          "★ h=1e-1 太大：截断误差 %.3e，是最优 h 处的 %.0f×（> %g），"
          "且**超过判据 ① 自己的阈值 %g** —— 随手取大 h 会让对拍假失败"
          % (errs[1e-1], errs[1e-1] / errs[best], TAIL_BAD_FACTOR, TOL_CENTRAL_REL))
    check("C.2", errs[1e-12] / errs[best] > TAIL_BAD_FACTOR
          and errs[1e-12] > TOL_CENTRAL_REL,
          "★ h=1e-12 太小：灾难性抵消把差值吞掉，误差 %.3e，是最优 h 处的 %.0f×（> %g），"
          "且**超过判据 ① 自己的阈值 %g** —— **小 h 这一侧也真的会错**"
          % (errs[1e-12], errs[1e-12] / errs[best], TAIL_BAD_FACTOR, TOL_CENTRAL_REL))
    # ★ 判据写成「最优 h 落在扫描表的**内部**」而不是「等于 1e-5」——
    #   换种子/换尺度会让极值点左右移动，钉死一个值就是本库的「判据的小 n 不稳定」。
    check("C.3", 0 < H_SWEEP.index(best) < len(H_SWEEP) - 1
          and errs[best] < TOL_CENTRAL_REL,
          "★ 可用窗口在**中间**：本扫描的最小误差在 h=%g 处（%.3e），两端都更差 —— "
          "⇒ **对拍必须扫 h，不能拍一个值**" % (best, errs[best]))

    # ── C.4 单侧差分 vs 中心差分：阶数差一次 ──────────────────────────────
    h = 1e-3
    g1 = one_sided_grad(w, h)
    g2, _ = central_difference_grad(w, h)
    e1 = _relerr(g1, g_formula)[0]
    e2 = _relerr(g2, g_formula)[0]
    print()
    print("  同一个 h=%g 上：单侧差分 %.3e · 中心差分 %.3e · 比值 %.1f×"
          % (h, e1, e2, e1 / e2))
    check("C.4", e1 / e2 > TOL_ONE_SIDED_MIN,
          "★ 单侧差分的误差比中心差分大 %.1f× > %g —— 这不是品味问题："
          "单侧只消掉一阶项、中心消掉二阶项" % (e1 / e2, TOL_ONE_SIDED_MIN))

    # ── C.5 ★ 反例：把「数值梯度更可信」当默认 ──────────────────────────────
    print()
    print("  C.5：反例 —— 「数值梯度比解析梯度可信」在本设定下**不成立**")
    print("     解析（两条独立路径）之间 %.3e" % _relerr(backward_formula(w), g_formula)[0])
    print("     数值在 h=1e-6 上偏离解析 %.3e" % errs[1e-6])
    print("     数值在 h=1e-12 上偏离解析 %.3e" % errs[1e-12])
    check("C.5", errs[1e-12] > errs[1e-6] * 1e3,
          "★ 同一个数值方法，只因为 h 取小了，误差从 %.3e 涨到 %.3e（%.0f×）——"
          "「数值的一定对」是错的；对拍的结论**依赖 h**"
          % (errs[1e-6], errs[1e-12], errs[1e-12] / errs[1e-6]))
    return errs


# ===========================================================================
# 9. D 类：失败标记自检 + 口径/阈值钉住 + 独立性
# ===========================================================================
def preflight():
    """★★ **D 类的前半段先跑**：标记自检 + 口径/阈值钉住。

    ★ 为什么必须**放在最前面**：口径常量（`M_ROWS` / `N_W1` / `N_PARAMS` / `H_SWEEP` …）
      一旦被改，后面的**形状检查**会先崩（`ValueError: cannot reshape array of size 48…`）——
      那时 `mutate.py` 记的是**崩溃捕获**，而「崩溃捕获不算真捕获」
      （它证明代码炸了，不是判据抓住了错）。**本程序第一次跑 `make mutate` 时就出现了 6 条**
      （`M_ROWS` / `D_COLS` / `N_W1` / `N_PARAMS` / `H_SWEEP[0]` / `H_SWEEP[5]` / `H_SWEEP[8]`）。
      ⇒ 把钉住**提到任何计算之前**，这些变异就变成**断言捕获**。
      ★ 失败时**立刻返回**：此时再去跑 A/B/C 只会产生一堆无意义的连带崩溃。

    返回 `True` 表示可以继续。
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
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    # ★ 「等于该值」：把**本主题被分配的那个字面量**写死在这里。
    #   ★ 这不是同义反复：`mutate.py` 只变异**模块级**常量，函数体里的这个字面量
    #     不在它的分母里 ⇒ 它正是用来挡住「标记被改掉 / 被跨主题照抄」的那把锁。
    check("D.2", ASSERT_FAIL_MARK == ".. FAIL: autograd"
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (":: FAIL: mixed-precision", "++ FAIL: optimizers",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    measured = {"M_ROWS": M_ROWS, "D_COLS": D_COLS, "N_W1": N_W1, "N_W2": N_W2,
                "N_PARAMS": N_PARAMS, "SEED": SEED, "W_SCALE": W_SCALE,
                "H_STEP": H_STEP, "X_VEC": X_VEC, "H_SWEEP": H_SWEEP}
    expect = {"M_ROWS": 6, "D_COLS": 8, "N_W1": 48, "N_W2": 6, "N_PARAMS": 54,
              "SEED": 20260923, "W_SCALE": 0.5, "H_STEP": 1e-6,
              "X_VEC": (0.3, 0.7, 1.1, 0.2, 0.4, 0.9, 0.5, 0.8),
              "H_SWEEP": (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12)}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 10 个逐位一致（这些是**结论的测量条件**；改任一即失败）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", N_W1 == M_ROWS * D_COLS and N_W2 == M_ROWS and N_PARAMS == N_W1 + N_W2,
          "★ 三个长度的**内部一致性**也钉住（%d×%d=%d · %d · %d+%d=%d）—— "
          "否则「54 个元素」这句话会与形状脱钩"
          % (M_ROWS, D_COLS, N_W1, N_W2, N_W1, N_W2, N_PARAMS))

    th = {"TOL_CENTRAL_REL": TOL_CENTRAL_REL, "TOL_MACHINE_REL": TOL_MACHINE_REL,
          "TOL_ONE_SIDED_MIN": TOL_ONE_SIDED_MIN, "TAIL_BAD_FACTOR": TAIL_BAD_FACTOR,
          "FWD_RATIO_MIN": FWD_RATIO_MIN}
    th_expect = {"TOL_CENTRAL_REL": 1e-5, "TOL_MACHINE_REL": 1e-14,
                 "TOL_ONE_SIDED_MIN": 10.0, "TAIL_BAD_FACTOR": 1000.0,
                 "FWD_RATIO_MIN": 100.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 5 个逐位一致（含**机器精度档** TOL_MACHINE_REL —— "
                            "放宽它，A.1 就退化成恒真）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


def part_D():
    print()
    print("=" * 74)
    print("D. 独立性证据 与「标记只用了本主题那一个」")
    print("=" * 74)
    check("D.5", inspect.getsource(backward_formula) != inspect.getsource(grad_by_tape),
          "★ 防 P17：路径 1 与路径 2 的**源码不同**（`inspect.getsource` 比较，%d vs %d 字节）"
          % (len(inspect.getsource(backward_formula)), len(inspect.getsource(grad_by_tape))))

    counted_src = inspect.getsource(forward_formula) + inspect.getsource(backward_formula)
    print("     （路径 1 的两个函数带 `CALLS` 计数：%d 字节的源码里 %d 处）"
          % (len(counted_src), counted_src.count("CALLS[")))
    check("D.6", "CALLS[" in counted_src,
          "★ 被测侧**真的**带计数 ⇒ `A.3` 的「计数表为空」不是空断言")

    # ★ 这条**只能在最后**查：它断言的是「本程序**实际打印过**的失败标记 ⊆ 本主题那一个」，
    #   在所有 part 都跑完之后才有内容（`preflight()` 里查不到任何东西）。
    check("D.7", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）—— 跨主题照抄标记"
          "由这一条 + `mutate.py` 的 AST 读取双重挡住"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))

    print("     本程序**没有** import `specs`：本主题的三个 specs 常量都属 `02`/`03` 的判据，"
          "本程序的真值是数值参照 —— 这条是刻意的（见文件头 §1）。")


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
    print("01_backward.py —— 反向传播本身，与数值梯度对拍的方法学")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区（一律回链，本程序不写）：`grad_fn` / `is_leaf` / `saved_tensors` 逐算子表 /")
    print("    `no_grad` vs `inference_mode` / `retain_graph` / `backward()` vs `autograd.grad()`")
    print("    —— 正文在 `70-frameworks/pytorch/02-autograd.md`（全篇 256 行）"
          "与 `00-meta/glossary.md:203-211`。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆"
              "无意义的连带崩溃")
        return summary()
    w = make_params()
    g_formula, g_num, evals = part_A(w)
    part_B(evals)
    part_C(w, g_formula)
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
