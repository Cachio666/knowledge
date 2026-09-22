#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_gating_routing.py —— gating 做什么、top-k 怎么选、两条「随机」机制、token choice vs expert choice。

配套文档：`01-gating-and-routing.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形：编号程序本机可跑 ⇒ 断言在程序内），
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本程序的数字**全部是推导值或本机实测值**，测量条件是：

> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 它们支撑的是「**路由这一步在结构上是什么**」这类结论，
> 不支撑任何关于真实模型训练或 GPU 的结论。

★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）：

  · `A.1`  甲口径的 gate 权重**每行之和 == 1**（`|Σ−1| ≤ 1e-12`）—— 去掉「在 k 个上重归一」
           这一步，行和就会 < 1（那正是 `40-kernels` 的 A1 用「丙」打出来的缺口）；
  · `A.2`  **两条独立代码路径**（`take_along_axis` 向量化 vs 逐行 Python 循环）给出同一个权重矩阵
           （最大逐元素差 `< 1e-14`）；
  · `A.3`  参照实现跑完后**调用计数表完全为空**；
  · `A.4`  并列 logit 的 tie-break 是「取下标小的」，且 `k` 取自规格而不是写死的数；
  · `B.2`  每个专家的平均槽位数是**精确有理数** `T·k/E`（`Fraction` 比较）；
  · `C.1`  GShard 的 random routing **两个端点精确**：`g₂ ≡ 1` ⇒ 派出 `T` 个、`g₂ ≡ 0` ⇒ 派出 `0` 个；
  · `C.2`  它真的**省容量**：真实分布下派出的第 2 专家数**严格小于** `T`；
  · `C.3`  Switch 的探索噪声**两个端点精确**：`eps = 0` ⇒ top-k **逐位不变**；`eps > 0` ⇒
           有**非零**比例的选择发生变化；
  · `C.4`  Table 11 的四个数**极差 ≤ 0.2**（原文说 `no material speed performance difference` / 只看质量
           `lower is better`）—— 把「jitter 最好」这个转述钉在它真实的分寸上。

★★ **本程序刻意不做的事**（计划 §2 的边界表，越界即作废）：

  * **不重讲** MoE 的 FLOPs / 权重流量 / 算术强度 / 拐点批大小（正文在
    `40-kernels/moe-kernels/01-moe-compute-pattern.md`，此处回链）。本程序**一个 FLOPs 都不算**；
  * ★★ **不做三种 gating 归一化口径（甲/乙/丙）的「实现与比较」**——那是**禁区**：
    正文与实现都在 `40-kernels/moe-kernels/code/verify.py` 的 **A1（`:273-300`）**。
    本程序只实现**甲**这一种口径（用来撑判据 ① 与真实规格），**不实现、不比较乙与丙**；
  * **不讲** permute / unpermute / grouped GEMM / slot mask（正文在 `40-kernels/moe-kernels/02,03`）。

用法：
    python3 01_gating_routing.py
"""

import ast
import collections
import inspect
import os
import sys
from fractions import Fraction

import numpy as np

# ★ 同目录的共享事实源（`code/` 自包含，`CONTRIBUTING:36`）。★ 本主题**不 import 任何框架**。
import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（`moe`）取 `## FAIL: moe`。
#   ★ 这个字面量**不是**照抄任何别的主题的：域 README 把五个主题的标记写死为
#     五个不同形态（`moe` 用 `##`），**不跨主题照抄**是本库纪律。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；
#     `D.5`/`D.6` 两条断言守着它：文件里恰好一次、等于本主题被分配的字面量、且含 `FAIL`（R18）。
ASSERT_FAIL_MARK = "## FAIL: moe"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

FAILS = []
MARKS_USED = set()
# ★ 计数器用 `Counter()` 而**不是**字面量字典：`mutate.py` 的分母是「模块级常量的
#   **字面量叶子**」，写成 `{"A": 0, ...}` 会把初值塞进分母（改它们只动汇总行）。
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
# 1. 口径常量（★ 全部在 `D.1`–`D.4` 被逐位钉住：改一个字节，本程序就报错）
# ===========================================================================
# ★★ 为什么「口径常量」也要钉住：它们不是「可调的参数」，而是**结论所引用的测量条件**。
#    文档 §5 与本程序的输出都写着「T=512 / E=8 / k=2 / logits ~ N(0,1)」——
#    静默改掉其中任何一个，**大部分断言的输出会变而文档不会**（两条路径依然互相一致），
#    这正是本域最想防的那类漂移。
#    （先例：`45-…/code/01_operator_inventory.py:526-541` 的「C6：逐个字段钉住」。）
T_N = 512                # 机制段的 token 数
E_N = 8                  # 机制段的专家数（★ 与真实模型的 128 分开：机制段要小到能逐行打印）
K_N = 2                  # 机制段的 top_k
D_N = 32                 # logits 的「上下文宽度」——**只用来生成 logits**（gate GEMM 的 K 维）
SEED = 20260922          # 数据种子（= 探测脚本用的那一个）
LOGIT_STD = 1.0          # logits 的抽样标准差（★ 它是口径的一部分：换掉它，C 段的数全变）
TRIALS = 20              # C.1 的重复次数（用来看「实测均值」稳不稳）
EPS_ZERO = 0.0           # C.3 的噪声端点一：**不加噪声**
EPS_JITTER = 1.0         # C.3 的噪声端点二：伪码里 `minval=1-eps, maxval=1+eps` 的 `eps`

# ── 阈值（★ 逐个在 `D.2` 钉住，并说明它属于哪一档）────────────────────────────
ROWSUM_TOL = 1e-12       # A.1 权重行和与 1 的偏差（**机器精度档**）
REF_TOL = 1e-14          # A.2 两条独立路径的最大逐元素差（**机器精度档**）
HOEFFDING_DELTA = 1e-6   # C.1 的 Hoeffding 界用的 δ（**推导用**，不是测出来的）
SPREAD_MAX = 0.2         # C.4 四个探索策略的极差上限（**量级档**，见 §C.4 的出处）


# ★ 被测实现的这个函数每被调一次就 +1；参照实现跑完后，计数**必须不动**。
CALLS = {}


# ===========================================================================
# 2. 被测实现（路径 1）—— 「先取 top-k，再在这 k 个 logit 上做 softmax」（甲口径）
# ===========================================================================
def route_topk(logits, k):
    """第 `j` 个选择：按 `stable` argsort 取前 `k` 个下标（并列时**取下标小的**）。

    ★ `kind="stable"` 是口径的一部分：它把「并列取小下标」这件事实现在排序里。
      `A.4` 用一个人造的并列例子把这条口径钉住。
    """
    CALLS["route_topk"] = CALLS.get("route_topk", 0) + 1
    return np.argsort(-logits, axis=1, kind="stable")[:, :k]


def gate_topk_softmax(logits, k):
    """被测实现（路径 1）：**向量化**地在每行的 top-k logit 上做 softmax。

    ★ 逐字出处（Switch Transformer §2.2，[C]）：「the router … `p = softmax(x·W_r)`…
      for each token we select the top-k … 」——**本程序的口径**是
      **只在被选中的 k 个 logit 上归一化**（行和恒为 1）。
    ★★ **三种口径（甲/乙/丙）的实现与比较是禁区**：正文在
      `40-kernels/moe-kernels/code/verify.py` 的 A1（`:273-300`）（正文在彼，此处回链）。
      本函数**只实现甲**，且**不提供任何切换口径的开关** —— 那是故意的：
      一旦本程序能算出「丙」，它就变成了那篇正文的一份**会漂移的副本**。
    """
    CALLS["gate_topk_softmax"] = CALLS.get("gate_topk_softmax", 0) + 1
    idx = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    sel = np.take_along_axis(logits, idx, axis=1)     # [T, k]
    sel = sel - sel.max(axis=1, keepdims=True)        # ← 稳定化（减最大值）
    e = np.exp(sel)
    return idx, e / e.sum(axis=1, keepdims=True)


# ===========================================================================
# 3. ★★ 参照实现（路径 2）—— **不调用被测实现的任何函数**
# ===========================================================================
# ★ `CONTRIBUTING:232-235`：A 类必须有一条与被检验代码路径**不同**的参照；
#   `00-meta/glossary.md:240` 给出机械判据：**参考实现不得调用被测实现的任何函数**
#   （调用计数器必须恰为 0），**且不得比被测实现更复杂**。
# ★★ 两条路径的差别是**真的**，不是换了个名字：
#     · 被测：向量化 `take_along_axis` + **减最大值** + numpy 的规约；
#     · 参照：**逐行** Python 循环 + **不减最大值** + 标准库 `math.exp` 逐个求和。
#   ⇒ 求和顺序与稳定化方式都不同，所以结果**只到浮点舍入**（`A.2` 实测非零）。
#   `A.3`/`D.8` 用 **AST 调用表为空**挡住「两边其实是同一行代码」
#   （计划 §0.1 的 **P17**：探针初版就踩过这个坑，报出的 `0.000e+00` 是构造出来的）。
def gate_ref_rows(logits, k):
    """独立参照（路径 2）：**逐行**算 top-k 与权重，只用标准库的 `math`。

    ★ 与路径 1 不共享任何辅助函数；**不减最大值**（所以求和顺序与指数都不同）。
    ★ 它**不需要** `route_topk`：下标由 `sorted` 现算。
    """
    import math
    rows = []
    for row in logits:
        order = sorted(range(len(row)), key=lambda i: (-float(row[i]), i))[:k]
        vals = [math.exp(float(row[i])) for i in order]
        tot = math.fsum(vals)
        rows.append((order, [v / tot for v in vals]))
    return rows


def draw_logits(seed, t, e, d, std):
    """★ **抽取顺序是口径的一部分**：先抽 gate 权重 `Wg[d, E]`，再抽 `X[t, d]`，
    最后 `logits = X @ Wg`。★ 形状写成 `(d, e)` 而不是 `(e, d)`，
    这样**转置写错**不会被形状巧合掩盖（转置后会直接抛形状错误）。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, d)) * std
    wg = rng.standard_normal((d, e)) * std
    return x @ wg


# ===========================================================================
# 4. A 类：权重口径与两条独立路径
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. gating 的权重口径（甲：**先取 top-k，再在这 k 个 logit 上 softmax**）")
    print("=" * 74)
    print("  口径：T=%d · E=%d · k=%d · d=%d · 种子 %d · X、Wg ~ N(0, %g²)"
          % (T_N, E_N, K_N, D_N, SEED, LOGIT_STD))
    print("  ★ 禁区（计划 §2）：甲/**乙**/**丙** 三种口径的**实现与比较**在")
    print("    40-kernels/moe-kernels/code/verify.py 的 A1（:273-300）（正文在彼，此处回链）。")
    print("    本程序**只实现甲**，且**没有**切换口径的开关。")

    logits = draw_logits(SEED, T_N, E_N, D_N, LOGIT_STD)
    idx, w = gate_topk_softmax(logits, K_N)

    # ── A.0 ★ 闭式参照：logits 的标准差 ────────────────────────────────────
    # ★ 为什么配这一条：`LOGIT_STD` 这个口径常量在**别的断言里只被钉住**（钉住是防漂移，
    #   不是证据）。它的**推导值**是 `Var(logits) = d · LOGIT_STD⁴`
    #   （`X` 与 `Wg` 各贡献一个 `LOGIT_STD²`，再乘 `d` 项）⇒ 标准差 `= LOGIT_STD²·√d`。
    #   **可手算**（P15）：`1²·√32 = 5.6569`。实测值必须落在它的 ±5% 内。
    sd = float(logits.std())
    sd_closed = LOGIT_STD ** 2 * np.sqrt(D_N)
    check("A.0", abs(sd / sd_closed - 1.0) < 0.05,
          "★ 闭式参照（**推导值**）：logits 的标准差 = `LOGIT_STD²·√d` = %.4f，"
          "实测 %.4f（相对差 %.3f%%，±5%% 内）⇒ `LOGIT_STD` 这个口径常量**真的在算式里**"
          % (sd_closed, sd, abs(sd / sd_closed - 1.0) * 100))

    # ── A.1 判据 ①：权重行和 == 1 ─────────────────────────────────────────
    dev = float(np.abs(w.sum(axis=1) - 1.0).max())
    check("A.1", dev <= ROWSUM_TOL,
          "★ 判据 ①：甲口径下每行权重之和与 1 的最大偏差 %.3e ≤ %g ⇒ **行和恒为 1**"
          "（把「在 k 个上重归一」这一步去掉，行和就 < 1 —— 那个缺口是乙/丙的正文，"
          "在 40-kernels 的 A1 里，此处**只回链**）" % (dev, ROWSUM_TOL))

    # ── A.2/A.3 ★★ 两条独立路径 ───────────────────────────────────────────
    print()
    print("  两条**独立**代码路径：① 向量化 take_along_axis + 减最大值（被测）")
    print("                          ② 逐行 Python 循环 + **不减最大值** + math.exp（参照）")
    ref = gate_ref_rows(logits, K_N)
    idx_ref = np.array([r[0] for r in ref], dtype=np.int64)
    w_ref = np.array([r[1] for r in ref], dtype=np.float64)
    d_idx = int(np.abs(idx - idx_ref).max()) if idx.shape == idx_ref.shape else -1
    d_w = float(np.abs(w - w_ref).max())
    check("A.2", d_idx == 0 and d_w < REF_TOL and d_w != 0.0,
          "★ 两条独立路径：下标最大差 %d、权重最大逐元素差 %.3e < %g（**机器精度档**，"
          "不是 `== 0.0`：真独立的两条路径不可能逐位相同）" % (d_idx, d_w, REF_TOL))

    # ★ 参照跑完后，计数表必须**整个是空的** —— 比「那个键是 0」更强：
    #   它等于「参照一次都没调用被测侧的任何函数」（含将来新增的函数）。
    CALLS.clear()
    gate_ref_rows(logits, K_N)
    check("A.3", not CALLS,
          "★ 参照实现跑完后**计数表完全为空**（记录到的调用：%s）⇒ "
          "它一次都没调用被测实现的函数" % (sorted(CALLS) if CALLS else "无"))

    # ── A.4 tie-break 与「真的是 top-k」 ──────────────────────────────────
    ties = np.array([[1.0, 1.0, 0.5, 0.25], [0.5, 1.0, 1.0, 0.25]])
    idx_tie = route_topk(ties, 2)
    ok_tie = idx_tie.tolist() == [[0, 1], [1, 2]]
    # ★★ 真正的性质：**选中的最小的那个 logit ≥ 被排除的最大的那个**（逐行）。
    #   这是「top-k」的定义本身；把 `argsort(-x)` 写成 `argsort(x)`、
    #   或把 `[:, :k]` 写成 `[:, -k:]`，这一条立刻失败。
    sel_min = np.take_along_axis(logits, idx, axis=1).min(axis=1)
    mask = np.ones_like(logits, dtype=bool)
    np.put_along_axis(mask, idx, False, axis=1)
    exc_max = np.where(mask, logits, -np.inf).max(axis=1)
    topk_ok = bool((sel_min >= exc_max).all())
    check("A.4", ok_tie and topk_ok and K_N == 2,
          "★ 并列时**取下标小的**（%s；`argsort(-x, kind=\"stable\")` 的语义）；"
          "且「top-k」的定义成立：**被选中的最小 logit ≥ 被排除的最大 logit**（逐 %d 行全真）"
          % (idx_tie.tolist(), T_N))
    return logits, idx, w


# ===========================================================================
# 5. B 类：真实模型的锚点（**只记规格关系**，FLOPs 回链）
# ===========================================================================
def part_B(logits, idx, w):
    print()
    print("=" * 74)
    print("B. 真实模型的规格锚点（Qwen3-30B-A3B；★ FLOPs / 权重流量**不在本域**）")
    print("=" * 74)
    print("  ★ 本段**不算任何 FLOPs**。MoE 一层的 FLOPs / 权重流量 / 算术强度 / 拐点批大小")
    print("    的正文在 40-kernels/moe-kernels/01-moe-compute-pattern.md（正文在彼，此处回链）。")

    moe = specs.QWEN3_30B_A3B_MOE
    e_real, k_real = moe["num_experts"], moe["num_experts_per_tok"]

    # ── B.1 ★ 用**真实规格的维数**跑一遍被测实现 ──────────────────────────
    # ★ 为什么要在真实维数上跑：机制段是 `E=8, k=2`，而真实模型是 `E=128, k=8`
    #   —— 只在玩具维数上验过就宣布「规格成立」是把两件事混为一谈（本库的经典错误模式）。
    logits_real = draw_logits(SEED + 7, 64, e_real, D_N, LOGIT_STD)
    _, w_real = gate_topk_softmax(logits_real, k_real)
    nz = int((w_real > 0).sum(axis=1).max())
    ratio = Fraction(k_real, e_real)
    check("B.1", nz == k_real and ratio == Fraction(1, 16),
          "★ 在**真实维数**上跑（T=64, E=%d, k=%d）：每个 token 的 gate 权重里"
          "**非零项恰为 %d 个**；活跃专家占比 = top_k/E = %s = %.2f%%（**精确有理数**）"
          % (e_real, k_real, nz, ratio, float(ratio) * 100))

    # ── B.2 平均槽位数是精确有理数 ───────────────────────────────────────
    slots = Fraction(T_N * K_N, E_N)
    print()
    print("  平均每个专家的槽位数（机制段口径 T=%d, k=%d, E=%d）：" % (T_N, K_N, E_N))
    print("    T·k/E = %d×%d/%d = %s" % (T_N, K_N, E_N, slots))
    exact = (slots == Fraction(T_N * K_N, E_N)) and slots == Fraction(128)
    check("B.2", exact,
          "★ 它是**精确有理数** `T·k/E` = %s（`Fraction` 比较，不是浮点）；"
          "代入真实规格 T=4096 / k=8 / E=128 得 **%s**（也是精确整数）"
          % (slots, Fraction(4096 * k_real, e_real)))

    # ── B.3 gate GEMM 的 N 维就是 E；且两个模型是**两种不同的路由** ─────────
    print()
    print("    gate GEMM 的 N 维 = E = %d（这是本层唯一『N 只有 %d』的 GEMM）；" % (e_real, e_real))
    print("    ★ 三个模型各有各的 gating：Switch = top-1、GShard = top-2、Qwen3-30B-A3B = top-%d。"
          % k_real)
    check("B.3", specs.GSHARD_NUM_SELECTED_EXPERTS == 2 and k_real == 8 and k_real != 1,
          "★ **top_k 是模型规格，不是常数**：GShard 逐字「each token is dispatched to **at most"
          " two experts**」（= 2）、Switch 是 top-1（aux loss 的 `f_i` 逐字用 `argmax`）、"
          "本模型是 top-%d ⇒ 引用「MoE 的 top-k」时必须写出是哪一个" % k_real)


# ===========================================================================
# 6. C 类：两条「随机」机制（**同名不同指**）+ 边界与反例
# ===========================================================================
def _hoeffding_bound(t, delta):
    """★ **推导值**：`T` 个独立 Bernoulli 的均值偏离期望的 Hoeffding 界 `sqrt(ln(2/δ) / (2T))`。

    ★ 为什么写出来：C.1 要断言「实测派出的个数 ≈ Σg₂」，而**没有这条界的话，那个容差
      就是拍脑袋定的**。写成闭式之后，容差**可手算复核**（P15）。
    """
    import math
    return math.sqrt(math.log(2.0 / delta) / (2.0 * t))


def part_C(logits, idx, w):
    print()
    print("=" * 74)
    print("C. 两条「随机」机制 —— ★ 它们**同名不同指**，而且**都不是**对方")
    print("=" * 74)

    # ── C.1/C.2 GShard 的 `Random routing`（为**省容量**）─────────────────
    # ★ 逐字（GShard §2.2，[C]）：「because `y_s` is a weighted average of what selected
    #   experts return, if the weight for the 2nd expert is very small, we can simply ignore
    #   the 2nd expert to conserve the overall expert capacity. Hence, in addition to
    #   respecting the expert capacity constraint, `GATE(·)` dispatches to the 2nd-best expert
    #   with the probability proportional to its weight `g₂`.」
    # ★★ 口径：本程序把 `g₂` 定义为**甲口径下第 2 个选择的权重**（`w[:, 1]`）。
    #   论文只写了「proportional to its weight g₂」，**没有**规定 `g₂` 用哪种归一化 ⇒
    #   本程序把这一点写成显式口径，而不是假装它没有歧义。
    g2 = w[:, 1].astype(np.float64)
    analytic = float(g2.sum())
    rng = np.random.default_rng(SEED + 1)
    counts = []
    for _ in range(TRIALS):
        u = rng.random(T_N)
        counts.append(int((u < g2).sum()))
    emp = float(np.mean(counts))
    bound = analytic + T_N * _hoeffding_bound(T_N, HOEFFDING_DELTA)

    print("  C.1 — GShard 的 `Random routing`：**按 `g₂` 的概率决定要不要派第 2 个专家**")
    print("       口径：g₂ = 甲口径下第 2 个选择的权重；期望派出的条数 = Σ_t g₂ = **%.3f**"
          % analytic)
    print("       实测（%d 次重复）：均值 %.3f、最小 %d、最大 %d"
          % (TRIALS, emp, min(counts), max(counts)))
    print("       Hoeffding 界（δ=%g，T=%d，**推导值**）：|实测−期望| ≤ %.1f"
          % (HOEFFDING_DELTA, T_N, T_N * _hoeffding_bound(T_N, HOEFFDING_DELTA)))

    # ★★ 两个端点：**精确**，不是统计 —— 它们才是这条断言的证伪力所在
    n_one = int((np.random.default_rng(SEED + 2).random(T_N) < 1.0).sum())
    n_zero = int((np.random.default_rng(SEED + 2).random(T_N) < 0.0).sum())
    check("C.1", n_one == T_N and n_zero == 0 and 0 < emp < T_N
          and abs(emp - analytic) <= T_N * _hoeffding_bound(T_N, HOEFFDING_DELTA),
          "★ **两个端点精确**：`g₂ ≡ 1` ⇒ 派出 **%d = T** 个；`g₂ ≡ 0` ⇒ 派出 **%d** 个；"
          "真实分布 ⇒ **0 < %.3f < T** 且在 Hoeffding 界内 ⇒ 这条机制真的被实现了"
          "(不是空断言)" % (n_one, n_zero, emp))

    saved = 1.0 - emp / T_N
    check("C.2", emp < T_N and saved > 0.0,
          "★ 它**真的省容量**（逐字：`to conserve the overall expert capacity`）："
          "被忽略掉的第 2 选择占 **%.1f%%**（%.1f/%d）—— 端点 `g₂≡1` 时这个比例恰为 **0%%**"
          % (saved * 100, T_N - emp, T_N))

    # ── C.3 Switch 的探索噪声（为**探索**，不是为省容量）──────────────────
    # ★★ 原文两处**互相不一致**，本程序把两处都写出来：
    #   · Appendix C 正文逐字：「4) **multiplicative jitter noise on the incoming
    #     representation**」——作用对象是**输入表示**、方式是**乘性**；
    #   · Figure 14 的伪码逐字：`if is_training : # Add noise for exploration across experts.`
    #     下一行 `router_logits += mtf.random_uniform(shape=router_logits.shape,
    #     minval=1-eps, maxval=1+eps)` ——作用对象是 **router_logits**、方式是**加性**。
    #   ⇒ 本程序实现**伪码**那一版（它是可执行的那一份），并在输出里标明这一点。
    print()
    print("  C.3 — Switch 的**探索噪声**：与 C.1 **同名不同指**（那条为省容量，这条为探索）")
    print("       ⚠ 论文两处不一致：正文说 `multiplicative jitter noise on the **incoming")
    print("         representation**`；伪码做的是 `router_logits += uniform(1-eps, 1+eps)`。")
    print("         本程序实现**伪码**那一版（可执行的那一份）。")

    def jitter(logits_, eps, rng_):
        """伪码那一版的探索噪声：`router_logits += uniform(1-eps, 1+eps)`。"""
        if eps == 0.0:
            return logits_
        return logits_ + rng_.uniform(1.0 - eps, 1.0 + eps, size=logits_.shape)

    base = route_topk(logits, K_N)
    for eps in (EPS_ZERO, EPS_JITTER):
        j = route_topk(jitter(logits, eps, np.random.default_rng(SEED + 3)), K_N)
        changed = float((j != base).any(axis=1).mean())
        print("       eps=%.1f ⇒ 选择发生变化的 token 占 **%6.2f%%**" % (eps, changed * 100))
        if eps == EPS_ZERO:
            changed_zero = changed
        else:
            changed_one = changed
    check("C.3", changed_zero == 0.0 and 0.0 < changed_one < 1.0,
          "★ **两个端点精确**：`eps=0` ⇒ 选择**逐位不变**（%.1f%%）；`eps=%.1f` ⇒ "
          "**非零**比例（%.1f%%）变化 ⇒ 噪声是真的（不是装饰）"
          % (changed_zero * 100, EPS_JITTER, changed_one * 100))

    # ── C.4 Table 11 的分寸：★ **表题与正文的方向互相冲突** ────────────────
    # ★★ 这是本程序发现的**第二处论文内部不一致**（第一处见 C.3）：
    #   · 表题（caption）逐字：「…under different randomness-strategies for selecting the
    #     expert (**lower is better**). There is no material speed performance difference…」
    #   · 同一张表的列头逐字是 `Model Quality (Neg. Log Perp.) ( ↑ )` —— **带向上的箭头**；
    #   · 而正文逐字：「Throughout this work, we use **input jitter** to inject noise as we have
    #     found it to **empirically perform the best**.」
    #   ⇒ 表头那个值 `input jitter` 是四者里**最大**的（−1.468）：按「越大越好」读，正文
    #     与表格自洽；按表题的「越小越好」读，**最好的是 argmax**，正文那句就反了。
    #   ⇒ **本域不替论文选方向**：只做两件事——(a) 报出极差（**两个方向下都成立**的结论）、
    #     (b) 把「两种读法给出不同的第一名」这件**冲突事实**本身钉住。
    nll = specs.SWITCH_EXPLORATION_NLL
    vals = sorted(nll.values())
    spread = vals[-1] - vals[0]
    best_hi = max(nll, key=nll.get)     # 按表头 ↑（或正文那句）的读法
    best_lo = min(nll, key=nll.get)     # 按表题 `lower is better` 的读法
    print()
    print("  C.4 — Switch Appendix C 的 Table 11（负对数困惑度）：%s" % nll)
    print("       极差 = %.3f" % spread)
    print("       按列头 `↑` / 正文「jitter perform the best」读 ⇒ 第一名 = %s（%.3f）"
          % (best_hi, nll[best_hi]))
    print("       按表题 `(lower is better)` 读        ⇒ 第一名 = %s（%.3f）"
          % (best_lo, nll[best_lo]))
    print("       ★ 两种读法**给出不同的第一名** ⇒ 论文的表题与正文冲突，本域**不选方向**。")
    check("C.4", 0.0 < spread <= SPREAD_MAX and best_hi != best_lo
          and specs.SWITCH_EXPLORATION_DIRECTION_FROM_CAPTION == "lower_is_better"
          and best_hi == "input_jitter",
          "★ **两条**：(a) 四个策略的极差 %.3f ≤ %g ⇒「**没有实质差别**」"
          "（这一条**两个方向下都成立**，是唯一能下的结论）；"
          "(b) 两种读法的第一名**不同**（%s vs %s）⇒ 冲突是真的，不是本域读错"
          % (spread, SPREAD_MAX, best_hi, best_lo))

    # ── C.5 反例：token choice 的负载**不可能**由单侧决定 ──────────────────
    # ★ 逐字（Expert Choice §3.1，[C]）：「Conventional MoE models employ **token-choice
    #   routing** which independently selects the top-k experts for each token」；
    #   而「**an expert choice method independently selects top-k tokens for each expert**,
    #   where k is a fixed expert capacity」「Despite its simplicity, expert choice achieves
    #   **perfect load balancing by design**」。
    # ★★ 可机械判定的差别：**expert choice 的每个专家拿到的条数可以被凑成完全相等**，
    #   而 token choice 的条数由**各 token 独立**的决定累加而来，**不能**被凑成相等。
    print()
    print("  C.5 — token choice vs expert choice：**谁选谁**决定负载能不能精确均衡")
    tc = np.bincount(idx.ravel(), minlength=E_N).astype(float)
    # expert choice：专家各取同样多的 token（这正是 EC 的 k = n·c/e 的定义，见 02 的 C 段）
    ec = np.full(E_N, T_N * K_N / E_N, dtype=float)
    print("       token choice 的每专家条数：%s" % tc.astype(int).tolist())
    print("       expert choice 的每专家条数：%s（**构造上恒等**）" % ec.astype(int).tolist())
    check("C.5", tc.max() > tc.min() and ec.max() == ec.min(),
          "★ token choice（`each token selects the top-k experts`）之下，条数由**独立决定**累加 ⇒"
          " max/mean = %.2f ≠ 1（不均）；expert choice（`each expert selects top-k tokens`）"
          "之下每个专家的条数**构造上相等** ⇒ max/mean = **%.2f**"
          % (tc.max() / tc.mean(), ec.max() / ec.mean()))


# ===========================================================================
# 7. D 类：失败标记自检 + 口径钉住
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    先例：`make lint` 与 `mutate.py` 都走 AST 而不是 grep —— 注释里出现一个名字
    和真的调用它是两件事。
    ★ 它同时是**参照实现独立性**的探测器（`D.8`）。
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


def pin_part():
    """★★ **口径 / 阈值 / specs 的钉住 —— 刻意放在 A/B/C 之前**。

    ★ 为什么放最前：这几条是**所有判据的前提**，而且它们必须**先于**任何
      「键名找不到」「形状对不上」这类**结构性崩溃**报出来。
      否则那些变异会被 `make mutate` 记成**崩溃捕获**，而崩溃捕获**不算真捕获**
      （它证明「代码炸了」，不是「判据抓住了错」）。Chunk 2 已确立此结构（本域硬要求）。
    """
    print("=" * 74)
    print("D. 口径/阈值/specs 的钉住（★ 放在 A/B/C **之前**：先钉住前提，再看数）")
    print("=" * 74)

    # ── 口径常量 ─────────────────────────────────────────────────────────
    measured = {"T_N": T_N, "E_N": E_N, "K_N": K_N, "D_N": D_N, "SEED": SEED,
                "LOGIT_STD": LOGIT_STD, "TRIALS": TRIALS,
                "EPS_ZERO": EPS_ZERO, "EPS_JITTER": EPS_JITTER}
    expect = {"T_N": 512, "E_N": 8, "K_N": 2, "D_N": 32, "SEED": 20260922,
              "LOGIT_STD": 1.0, "TRIALS": 20, "EPS_ZERO": 0.0, "EPS_JITTER": 1.0}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 9 个逐位一致（含 `LOGIT_STD` 与 `TRIALS` —— 它们不是可调参数，"
          "而是**结论所引用的测量条件**）%s" % ("" if not diff else " ← 差异 %s" % diff))

    # ── 阈值 ─────────────────────────────────────────────────────────────
    th = {"ROWSUM_TOL": ROWSUM_TOL, "REF_TOL": REF_TOL,
          "HOEFFDING_DELTA": HOEFFDING_DELTA, "SPREAD_MAX": SPREAD_MAX}
    th_expect = {"ROWSUM_TOL": 1e-12, "REF_TOL": 1e-14,
                 "HOEFFDING_DELTA": 1e-6, "SPREAD_MAX": 0.2}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 4 个逐位一致（含 2 个**机器精度档**：ROWSUM_TOL / REF_TOL —— 放宽它们，"
          "A.1/A.2 就退化成恒真）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── specs 的常量：**键名先钉、取值后钉** ─────────────────────────────
    # ★ **键名先钉** —— 顺序是刻意的：`mutate.py` 的分母含**字典的键**，
    #   键名被改后按字面量取值会抛 `KeyError`（那是**崩溃**，不算断言捕获）。
    #   所以键名检查必须**排在**任何按名取值的语句之前。
    keys_ok = (set(specs.QWEN3_30B_A3B_MOE) == {"num_experts", "num_experts_per_tok",
                                                "moe_intermediate_size"}
               and set(specs.QWEN3_30B_A3B_ROUTER_FROM_MIRROR)
               == {"router_aux_loss_coef", "norm_topk_prob"}
               and set(specs.SWITCH_EXPLORATION_NLL)
               == {"argmax", "sample_softmax", "input_dropout", "input_jitter"}
               and set(specs.SWITCH_MOE_SPEED_BY_CF) == {2.0, 1.25})
    check("D.3", keys_ok,
          "★ `specs.py` 四组字典的**键名**逐位钉住（MoE 三个、router 两个、探索四个、"
          "速度两个；先钉键名，否则键名一变、下一段就 `KeyError` ⇒ 那是崩溃不是断言捕获）")
    if not keys_ok:                      # ★ 键名不对就不再按名取值（避免崩溃吞掉结论）
        return

    # ★★ 为什么这里要**重钉一次上游常量**：`mutate.py` 会改 `specs.py` 的每一个叶子，
    #   而有些改动**保住了比值**（例如 `num_experts` 与 `num_experts_per_tok` 同时改，
    #   `k/E` 仍是 1/16）⇒ 只有**逐位钉住**才挡得住。
    #   这正是 R17 后半段「给它一个消费者」的落法。
    moe_pin = (specs.QWEN3_30B_A3B_MOE["num_experts"],
               specs.QWEN3_30B_A3B_MOE["num_experts_per_tok"],
               specs.QWEN3_30B_A3B_MOE["moe_intermediate_size"])
    router_pin = (specs.QWEN3_30B_A3B_ROUTER_FROM_MIRROR["router_aux_loss_coef"],
                  specs.QWEN3_30B_A3B_ROUTER_FROM_MIRROR["norm_topk_prob"])
    nll_pin = (specs.SWITCH_EXPLORATION_NLL["argmax"],
               specs.SWITCH_EXPLORATION_NLL["sample_softmax"],
               specs.SWITCH_EXPLORATION_NLL["input_dropout"],
               specs.SWITCH_EXPLORATION_NLL["input_jitter"])
    speed_pin = (specs.SWITCH_MOE_SPEED_BY_CF[2.0], specs.SWITCH_MOE_SPEED_BY_CF[1.25])
    check("D.4", moe_pin == (128, 8, 768) and router_pin == (0.001, True)
          and nll_pin == (-1.471, -1.570, -1.480, -1.468) and speed_pin == (840.0, 790.0)
          and specs.GSHARD_NUM_SELECTED_EXPERTS == 2
          and specs.SWITCH_CAPACITY_FACTOR_FINDING == (1.0, 1.25)
          and specs.SWITCH_AUX_ALPHA == 1e-2
          and specs.SWITCH_EXPLORATION_DIRECTION_FROM_CAPTION == "lower_is_better",
          "★ `specs.py` 的常量**逐位钉住**：MoE=%s（[B] 转记）、router=%s（**镜像取得**，"
          "未与官方 repo 比对 sha256）、Table 11=%s、X6 的速度=%s（[C]）；"
          "表题声称的方向 %r（★ 它与同表列头 `↑` 及正文冲突，见 C.4）"
          % (moe_pin, router_pin, nll_pin, speed_pin,
             specs.SWITCH_EXPLORATION_DIRECTION_FROM_CAPTION))


def mark_part():
    """失败标记自检 + 参照实现的独立性（AST 级）—— 放在最后跑。"""
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 参照实现的独立性")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.5", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)

    # ★ 「等于该值」：把**本主题被分配的那个字面量**写死在这里（R18）。
    #   ★ 这不是同义反复：`mutate.py` 只变异**模块级**常量，函数体里的这个字面量
    #     不在它的分母里 ⇒ 它正是用来挡住「标记被改掉 / 被跨主题照抄」的那把锁。
    check("D.6", ASSERT_FAIL_MARK == "## FAIL: moe"
          and "FAIL" in ASSERT_FAIL_MARK
          and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.7", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                       "~~ FAIL: normalization",
                                       "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    # ── 参照实现的独立性（AST 级）─────────────────────────────────────────
    # ★ 被监视的名字 = 被测侧与数据生成侧的函数（参照**一个都不该调用**）。
    wnames = {route_topk.__name__, gate_topk_softmax.__name__, draw_logits.__name__}
    called = _callee_names(gate_ref_rows)
    shared = sorted(wnames & called)
    counted_src = inspect.getsource(gate_topk_softmax)
    check("D.8", "CALLS[" in counted_src and not shared,
          "被测的 `gate_topk_softmax` 带 `CALLS` 计数；参照实现的 **AST 调用表里一个都没有它**"
          "（也没共用 `route_topk` / `draw_logits`）⇒ A.3 的「计数为 0」不是空断言%s"
          % ("" if not shared else " ← 参照调用了 %s" % shared))


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
    print("01_gating_routing.py —— gating / top-k / 两条「随机」机制 / token choice vs expert choice")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区（计划 §2 的边界表）：本程序**不做**三种 gating 归一化口径（甲/乙/丙）的")
    print("    实现与比较（正文在 40-kernels/moe-kernels/code/verify.py 的 A1（:273-300））；")
    print("    也**不算**任何 FLOPs / 权重流量 / 算术强度（正文在 40-kernels/moe-kernels/01）。")
    pin_part()      # ★ D.1–D.4：先钉住前提（见 pin_part 的说明）
    logits, idx, w = part_A()
    part_B(logits, idx, w)
    part_C(logits, idx, w)
    mark_part()     # ★ D.5–D.8：标记自检与独立性
    return summary()


if __name__ == "__main__":
    sys.exit(main())
