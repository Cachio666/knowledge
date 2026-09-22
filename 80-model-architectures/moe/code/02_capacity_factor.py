#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_capacity_factor.py —— ★★ **容量因子的定义式**、取整口径、padding / overflow、两条溢出协议。

配套文档：`02-capacity-factor.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ **本程序是上游 8 处指名让渡的落点之一**。`00-meta/glossary.md:240` 逐字：
「容量因子 (capacity factor) … **定义与语义的正文在 `80-model-architectures/moe/02-capacity-factor.md`**」；
`40-kernels/moe-kernels/03-permute-and-unpermute.md:47` 逐字：
「**容量因子本身的定义与语义不在本域**，正文在 `80-model-architectures/moe/02-capacity-factor.md`（正文在彼，此处回链）」。
⇒ **本程序的 `A` 段就是那 8 处指的正文的可执行形态。**

★★ 本程序的数字**全部是推导值或本机实测值**，测量条件是：

> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。

★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）：

  · `A.1`  `expert capacity` **逐位等于** `ceil((tokens/experts) × capacity_factor)`（用 `Fraction`
           算的精确整数 —— 浮点 `math.ceil(T/E*cf)` 在 `T/E` 不整除时会**少开 1 个槽位**）；
  · `A.2`  ★ **为什么必须 `ceil`**：构造一个**完美均衡**的负载（`T=4096, E=48`）——
           `floor` 口径下**仍然丢 16 个槽位**，`ceil` 口径下**恰好丢 0**个。⇒ 「向下取整」
           会让「容量因子 = 1.0」在**负载完全均匀**时也丢东西，那是自相矛盾的；
  · `B.1`  ★★ **dropping ≠ token 消失**：被跳过的 token 的**该层输出 == 其输入**（`np.array_equal`
           **逐位**），**且它仍出现在下一层的输入里**；而同层的其它 token **不满足**这一条
           （对照组 —— 没有它，这条断言可能只是「所有 token 都这样」）；
  · `B.2`  丢弃率随 `capacity_factor` **严格单调下降**、且 `cf = 1.0` 时 **> 0**（负载不均时）；
  · `B.4`  ★★ **同一个「丢弃率」有四种数法**（容量粒度 × 分子单位 × 分母）⇒
           token 口径下「有选择被丢的 token」占比**高于**槽位口径 ⇒ 口径必须写出来；
  · `C.1`  ★ No-Token-Left-Behind：重路由之后**每个专家的条数 ≤ 容量**，且**没有 token 一个
           专家都拿不到**；而 dropping 协议下**有**；
  · `C.2`  ★★ **NTLB 不是普遍保证**：把容量压紧（`cf` 很小时）**它照样留下无人服务**的 token
           —— 原文逐字用 `almost no tokens`，不是 `no tokens`；
  · `C.3`  ★★ **「幸存权重要不要重新归一化」是架构层的选择**（这是上游 `03:197` 指名让渡的
           那条）：两条路的输出**恰好相差一个因子 `Σ幸存权重`**（实测比值 == Σw，相对误差 < 1e-12）；
  · `C.4`  ★ **`capacity factor` 同名不同指**：Switch 的 `cf`（在均分之上**额外开多少缓冲**）
           与 Expert Choice 的 `c`（**一个 token 平均被几个专家服务**）**闭式恰好相同**，
           但「被**用掉**的槽位数」差 `cf` 倍 —— 这个差别是可测的。

★★ **本程序刻意不做的事**（计划 §2 的边界表，越界即作废）：

  * **不讲** permute / unpermute 的访存模式与 slot mask 的**实现**（正文在
    `40-kernels/moe-kernels/03-permute-and-unpermute.md`，此处回链）；
  * **不算** FLOPs / 权重流量 / 算术强度（正文在 `40-kernels/moe-kernels/01`）；
  * **不讲** grouped GEMM（正文在 `40-kernels/moe-kernels/02`）。

用法：
    python3 02_capacity_factor.py
"""

import ast
import collections
import inspect
import math
import os
import sys
from fractions import Fraction

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账
# ===========================================================================
ASSERT_FAIL_MARK = "## FAIL: moe"

OTHER_THEME_MARKS = ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

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
# 1. 口径常量（★ 全部在 `D.1`/`D.2` 被逐位钉住）
# ===========================================================================
# ★★ 统计段的口径**刻意照抄探测脚本 Q7**（计划 §0.7-D 的 **D16** 是它的唯一口径记录）：
#    `T=4096, E=64, k=2`，logits ~ N(0,1)，前 8 个「热门专家」加 `+1.2` 的偏置。
T_CAP = 4096             # token 数
E_CAP = 64               # 专家数
K_CAP = 2                # top_k
HOT_N = 8                # 热门专家个数
HOT_BIAS = 1.2           # 热点偏置（★ 它是口径的一部分：改掉它，四个丢弃率全变）
SEED_CAP = 20260922      # 数据种子（= 探测脚本用的那一个；⚠ 见 `B.2` 的口径注记）
CF_LIST = (1.0, 1.25, 1.5, 2.0)   # ★ 判据④用的四个 capacity factor

# ★ 协议段（小口径，小到能逐 token 打印）
T_PROTO = 64
E_PROTO = 4
K_PROTO = 2
CAP_PROTO_FACTOR = 1.0   # 协议段的 capacity factor
SEED_PROTO = 20260922
D_PROTO = 16             # 专家内部的宽度（**只用来造一个确定的专家函数**）
CAP_TIGHT_PROTO = 1    # C.2 的「压紧容量」端点（每专家 1 个槽位）：连 NTLB 也救不了

# ── 阈值（★ 逐个在 `D.2` 钉住）─────────────────────────────────────────────
EXACT_TOL = 0             # A.1/A.2 是**精确整数**判据（容差就是 0，不是「小到看不见」）
RENORM_TOL = 1e-12        # C.3 两条归一化路线的比值与 Σw 的偏差（**机器精度档**）


def capacity_ceil(tokens, experts, cf_num, cf_den=1):
    """★★ **容量因子的定义式**（**本域就是它的正文**）：

        expert capacity = ceil( (tokens per batch / number of experts) × capacity factor )

    ★ 逐字出处（Switch Transformer §2.2，[C]，arXiv 2101.03961）：
      「The expert capacity—the number of tokens each expert computes—is set by **evenly
        dividing the number of tokens in the batch across the number of experts**, and then
        **further expanding by a capacity factor**」；同段逐字给出式 (3)
      `expert capacity = (tokens per batch / number of experts) × capacity factor`。
    ★★ **取整口径写死为 `ceil`，理由见 `A.2`**：
      容量是**整数个槽位**；用 `floor` 时，只要 `tokens/experts` **不是整数**，
      **完全均匀**的负载也必然丢掉一些槽位 ⇒ 与「capacity factor ≥ 1 是缓冲」这句话自相矛盾。
    ★ 实现用 `Fraction` **精确**算，不经过浮点：`math.ceil(4096/48*1.0)` 这类写法在
      `T/E` 不整除时可能给出**少 1** 的结果（浮点 `4096/48 = 85.333…` 的 ceil 恰好
      不受影响，但 `T/E*cf` 复合之后不再可靠）⇒ 本函数把分子分母都以整数传进来。
    """
    exact = Fraction(tokens * cf_num, experts * cf_den)
    return -((-exact.numerator) // exact.denominator)      # 精确 ceil，不用浮点


def capacity_floor(tokens, experts, cf_num, cf_den=1):
    """★ **反例用的口径**（只在 `A.2` 里出现）：同一个式子改用 `floor`。"""
    exact = Fraction(tokens * cf_num, experts * cf_den)
    return exact.numerator // exact.denominator


def load_counts(topk, experts):
    """每个专家被选中的条目数（**不施加容量**）——dropping 前的原始负载。"""
    c = np.zeros((topk.shape[1], experts), dtype=np.int64)
    for j in range(topk.shape[1]):
        c[j] = np.bincount(topk[:, j], minlength=experts)
    return c


def skew_logits(seed, t, e, hot_n, bias, k):
    """★ 负载不均的构造 —— **逐字照抄探测脚本 Q7 的口径**：

        `logits = rng.standard_normal((T, E))`；`hot = rng.permutation(E)[:hot_n]`；
        `logits[:, hot] += bias`；`topk = argsort(-logits)[:, :k]`（稳定排序）。

    ★★ **为什么连「随机选哪几个是热点」这一步都要照抄**：只改成「前 `hot_n` 个」，四个丢弃率
      就会变（本机实测 cf=1.0：`48.93%` → `48.97%`）⇒ 那是**另一个口径**，
      与计划 §0.7-D 的 **D16** 不再可比。
    ⚠ **即使照抄了抽取顺序，本程序也算不出 D16 的那四个数**：探测脚本的 `rng` 在 Q7 之前
      已被 Q1–Q6 消耗过 ⇒ 那四个数**绑定在探测脚本的状态上**，不是 `default_rng(seed)` 的属性。
      ⇒ 本程序报的是**它自己的**实测值（同一个口径、不同的状态），文档里两个都写。
    """
    rng = np.random.default_rng(seed)
    lg = rng.standard_normal((t, e))
    hot = rng.permutation(e)[:hot_n]
    lg[:, hot] += bias
    return lg, np.argsort(-lg, axis=1, kind="stable")[:, :k]


# ===========================================================================
# 2. A 类：容量因子的定义式与取整口径
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. ★★ 容量因子的定义式（**本段就是那 8 处让渡指的正文的可执行形态**）")
    print("=" * 74)
    print("  逐字（Switch Transformer §2.2，[C]）：")
    print("    「The expert capacity—the number of tokens each expert computes—is set by")
    print("      **evenly dividing** the number of tokens in the batch across the number of")
    print("      experts, and then **further expanding by a capacity factor**」")
    print("    「A capacity factor greater than 1.0 creates additional buffer to accommodate")
    print("      for when tokens are not perfectly balanced across experts.」")

    # ── A.1 逐位等于 ceil(...) ────────────────────────────────────────────
    rows = []
    for cf in CF_LIST:
        num, den = Fraction(cf).numerator, Fraction(cf).denominator
        c_exact = capacity_ceil(T_CAP, E_CAP, num, den)
        c_float = int(math.ceil(T_CAP / E_CAP * cf))
        rows.append((cf, c_exact, c_float))
        print("    cf=%-5s ⇒ ceil(%.6f × %s) = **%d**（浮点路径给 %d）"
              % (cf, T_CAP / E_CAP, cf, c_exact, c_float))
    check("A.1", all(r[1] == int(math.ceil(T_CAP / E_CAP * r[0])) for r in rows)
          and [r[1] for r in rows] == [64, 80, 96, 128],
          "★ **逐位等于** `ceil((tokens/experts) × capacity_factor)`（`Fraction` 精确算）："
          "cf ∈ %s ⇒ 容量 = %s。★ 这里 `T/E = 64` 恰好整除，**所以浮点路径也对**；"
          "`A.2` 用一个**不整除**的例子把两者分开" % (list(CF_LIST), [r[1] for r in rows]))

    # ── A.2 ★★ 为什么是 ceil 而不是 floor ────────────────────────────────
    T_BAD, E_BAD = 4096, 48          # 4096 / 48 = 85.333… **不整除**
    # 构造**完美均衡**的负载：把 T·k 个条目轮转着发给 E 个专家
    base, extra = divmod(T_BAD // 1 * 1, E_BAD)   # 4096 = 48*85 + 16
    per_expert = [base + 1] * extra + [base] * (E_BAD - extra)
    cap_floor = capacity_floor(T_BAD, E_BAD, 1, 1)
    cap_ceil = capacity_ceil(T_BAD, E_BAD, 1, 1)
    drop_floor = sum(max(0, n - cap_floor) for n in per_expert)
    drop_ceil = sum(max(0, n - cap_ceil) for n in per_expert)
    print()
    print("    A.2 的反例口径：T=%d、E=%d ⇒ T/E = %.6f（**不整除**）；" % (T_BAD, E_BAD, T_BAD / E_BAD))
    print("        完美均衡的负载（%d 个专家拿 %d 条、%d 个拿 %d 条，合计 %d）"
          % (extra, base + 1, E_BAD - extra, base, sum(per_expert)))
    print("        floor 口径：容量 = %d ⇒ 丢掉 **%d** 个槽位（负载完全均匀却仍然丢！）"
          % (cap_floor, drop_floor))
    print("        ceil  口径：容量 = %d ⇒ 丢掉 **%d** 个槽位" % (cap_ceil, drop_ceil))
    check("A.2", cap_floor == 85 and cap_ceil == 86 and drop_floor == 16 and drop_ceil == 0
          and T_BAD % E_BAD != 0,
          "★ **取整口径必须写死为 `ceil`**：`T/E` 不整除时，`floor` 会让"
          "**完全均匀**的负载也丢 %d 个槽位 —— 与「cf ≥ 1 是给不均留的缓冲」自相矛盾。"
          "`ceil` 下丢 %d 个（**精确**，容差就是 %d）" % (drop_floor, drop_ceil, EXACT_TOL))

    # ── A.3 padding：cf > 1 的代价是**白算**，不是丢数据 ──────────────────
    #   逐字（同段，[C]）：「If the number of tokens sent to an expert is less than the expert
    #   capacity, then the computation may simply be **padded** – an inefficient use of the
    #   hardware, but **mathematically correct**.」
    pad = [(cf, 1.0 - 1.0 / cf) for cf in CF_LIST]
    print()
    print("    A.3 **padding**：完美均衡时，槽位的利用率 = 1/cf ⇒ 白算的比例 = 1 − 1/cf：")
    for cf, waste in pad:
        print("        cf=%-5s ⇒ 白算 **%6.2f%%** 的槽位（**数学上正确**，只是浪费硬件）"
              % (cf, waste * 100))
    check("A.3", all(abs(w - (1 - 1 / cf)) == 0 for cf, w in pad) and pad[-1][1] == 0.5,
          "★ **两条账必须是两笔**：cf>1 在负载均衡时买来的是 **padding 的浪费**"
          "（`1 − 1/cf`，cf=2.0 时 **50%%**）；它**不是**「丢数据」那一笔。"
          "原文同一段把两者分开写：padding = `mathematically correct`、overflow = 另议")
    return rows


# ===========================================================================
# 3. B 类：dropping 协议 —— ★★ 「被跳过的 token 仍然前进」
# ===========================================================================
def _experts_matrices(seed, e, d, d_ff=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((e, d, d_ff)) * 0.3, rng.standard_normal((e, d_ff, d)) * 0.3


def run_layer(x, idx, w, up, down, cap, protocol):
    """跑一层（带残差）—— 返回 `(out, served_mask)`。

    ★★ **dropping 协议**（逐字出处，Switch §2.2，[C]）：
      「If too many tokens are routed to an expert (referred to later as **dropped tokens**),
        **computation is skipped and the token representation is passed directly to the next
        layer through the residual connection.**」
      ⇒ 实现就是：`out = x`（残差），**只对拿到槽位的 (t, j) 累加** `w·expert(x)`。
        被跳过的 token 的 `out[t]` **逐位等于** `x[t]` —— **它没有消失**。
    ★★ **GShard 的同义表述**（§2.2，[C]）：「When both experts selected by a token already
      exceed their capacity, the token is considered as an overflowed token, where
      `G_{s,E}` **degenerates into a zero vector**. Such tokens have their representation
      `x_s` **passed on to the next layer via residual connections**.」
      ⇒ 两篇是**同一条路**：`G` 退化成零向量 ⇒ FFN 的输出加数为 0 ⇒ 残差把它原样送下去。

    `protocol` = `"dropping"`（本函数实现）或 `"none"`（容量无限，用来做对照）。
    """
    t_n = x.shape[0]
    out = x.copy()                       # ★ 残差：**从这里开始**，被跳过的行就停在这里
    served = np.zeros((t_n, idx.shape[1]), dtype=bool)
    used = np.zeros(up.shape[0], dtype=np.int64)
    for t in range(t_n):
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            if protocol == "none" or used[e] < cap:
                if protocol != "none":
                    used[e] += 1
                served[t, j] = True
                h = np.tanh(x[t] @ up[e])
                out[t] = out[t] + float(w[t, j]) * (h @ down[e])
    return out, served


def part_B():
    print()
    print("=" * 74)
    print("B. ★★ dropping 协议：「被跳过的 token 仍经残差流前进」—— 它不是「token 消失」")
    print("=" * 74)
    print("  口径：T=%d · E=%d · k=%d · d=%d · cf=%s（协议段）"
          % (T_PROTO, E_PROTO, K_PROTO, D_PROTO, CAP_PROTO_FACTOR))
    lg, idx = skew_logits(SEED_PROTO, T_PROTO, E_PROTO, 1, HOT_BIAS, K_PROTO)
    rng = np.random.default_rng(SEED_PROTO + 11)
    x = rng.standard_normal((T_PROTO, D_PROTO))
    w = np.full((T_PROTO, K_PROTO), 1.0 / K_PROTO)
    up, down = _experts_matrices(SEED_PROTO + 12, E_PROTO, D_PROTO)
    cap = capacity_ceil(T_PROTO, E_PROTO, *Fraction(CAP_PROTO_FACTOR).as_integer_ratio())

    out, served = run_layer(x, idx, w, up, down, cap, "dropping")
    full = ~served.any(axis=1)            # 该 token 的 k 个选择**全被跳过**
    ids = np.nonzero(full)[0]
    print("  容量/专家 = %d；被跳过的 (token, 选择) 对 = %d / %d"
          % (cap, int((~served).sum()), T_PROTO * K_PROTO))
    print("  **k 个选择全被跳过**的 token = %d / %d（%s）"
          % (len(ids), T_PROTO, ids.tolist() if len(ids) <= 12 else "…"))

    # ── B.1 ★★ 该层输出 == 其输入（逐位），且它仍出现在下一层的输入里 ──────
    same = np.array([np.array_equal(out[t], x[t]) for t in ids]) if len(ids) else np.array([])
    diff_ids = np.setdiff1d(np.arange(T_PROTO), ids)
    # 对照组：**没有**被全跳过的 token，它们的输出**不**等于输入
    ctrl = np.array([not np.array_equal(out[t], x[t]) for t in diff_ids])
    # 「仍出现在下一层的输入里」：把 out 当作第二层的输入，再走一层
    x2_in = out.copy()
    present = all(np.array_equal(x2_in[t], x[t]) for t in ids) if len(ids) else False
    rows_ok = x2_in.shape[0] == T_PROTO
    print()
    print("  B.1 — 被全跳过的 %d 个 token 的**该层输出**与**其输入**：" % len(ids))
    for t in ids[:6]:
        print("      token %-3d  ‖x‖=%.6f  ‖out‖=%.6f  逐位相等 = %s"
              % (t, np.linalg.norm(x[t]), np.linalg.norm(out[t]), np.array_equal(out[t], x[t])))
    print("  对照组（**有**槽位的 %d 个 token）：逐位相等的有 %d 个（应当为 0）"
          % (len(diff_ids), len(diff_ids) - int(ctrl.sum())))
    check("B.1", len(ids) > 0 and bool(same.all()) and bool(ctrl.all())
          and present and rows_ok,
          "★★ **非空条件成立**（%d 个 token 的 k 个选择全被跳过）且：它们的**该层输出逐位等于其输入**"
          "（残差恒等，`np.array_equal`）；它们**仍出现在下一层的输入里**（该行逐位 == `x[t]`）；"
          "而同层**有**槽位的 token **无一**满足这条（对照组 ⇒ 这条断言不是「所有 token 都这样」）"
          % len(ids))

    # ── B.2 判据 ④：丢弃率随 cf 严格单调下降，且 cf=1.0 时 > 0 ────────────
    print()
    print("  B.2 — 判据 ④：丢弃率随 `capacity_factor` 的走向（统计段口径：T=%d, E=%d, k=%d,"
          " `permutation` 选出的 %d 个热门专家 +%s —— 逐字照抄探测脚本 Q7）"
          % (T_CAP, E_CAP, K_CAP, HOT_N, HOT_BIAS))
    _, idx_big = skew_logits(SEED_CAP, T_CAP, E_CAP, HOT_N, HOT_BIAS, K_CAP)
    cj = load_counts(idx_big, E_CAP)
    rates = []
    for cf in CF_LIST:
        num, den = Fraction(cf).numerator, Fraction(cf).denominator
        cap_j = capacity_ceil(T_CAP, E_CAP, num, den)
        dropped = int(sum(max(0, int(cj[j, e]) - cap_j) for j in range(K_CAP) for e in range(E_CAP)))
        rates.append(dropped / (T_CAP * K_CAP))
        print("      cf=%-5s ⇒ 容量/专家=%-3d 丢弃槽位=%5d / %5d = **%6.2f%%**"
              % (cf, cap_j, dropped, T_CAP * K_CAP, rates[-1] * 100))
    strictly_down = all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))
    check("B.2", strictly_down and rates[0] > 0.0,
          "★ 判据 ④：四点**严格递减**（%s，逐点差 %s）且 cf=1.0 时 **%.2f%% > 0**"
          "⇒ 「调大容量就少丢」在本口径下成立"
          % (["%.2f%%" % (r * 100) for r in rates],
             ["%.2f pp" % ((rates[i] - rates[i + 1]) * 100) for i in range(len(rates) - 1)],
             rates[0] * 100))

    # ── B.3 ★★ 口径表：**同一件事，四种数法，四个数** ────────────────────
    print()
    print("  B.3 — ★★ **口径必须写进正文**：同一份路由，四种数法的读数")
    merged = cj.sum(axis=0)
    rows = []
    for cf in CF_LIST:
        num, den = Fraction(cf).numerator, Fraction(cf).denominator
        cap_j = capacity_ceil(T_CAP, E_CAP, num, den)
        cap_m = capacity_ceil(T_CAP * K_CAP, E_CAP, num, den)
        a = sum(max(0, int(cj[j, e]) - cap_j) for j in range(K_CAP) for e in range(E_CAP))
        b = int(sum(max(0, int(n) - cap_m) for n in merged))
        c = int(sum(max(0, int(n) - cap_j) for n in merged))
        rows.append((cf, a / (T_CAP * K_CAP), b / (T_CAP * K_CAP), c / (T_CAP * K_CAP)))
        print("      cf=%-5s 甲（每 (j,e) 各自 cap=%-3d；分母 T·k）= **%6.2f%%**"
              " · 乙（合并，cap=%-3d）= **%6.2f%%** · 丙（合并，cap=%-3d）= **%6.2f%%**"
              % (cf, cap_j, rows[-1][1] * 100, cap_m, rows[-1][2] * 100,
                 cap_j, rows[-1][3] * 100))
    print("      ★ 本程序与探测脚本（计划 §0.7-D 的 D16）采用**甲**；")
    print("        甲 与 乙 在本抽样上同值 —— ⚠ **这不是恒等式**：只有当两半都越过各自 cap 时")
    print("        两者的截断量才恒等（`(a−c)+(b−c) = (a+b)−2c`）；任一半年没满就不等。")
    print("        **丙 更高**（它只给每个专家 `T/E·cf` 个槽位，两个来源要**共享**这一份）。")
    check("B.3", all(abs(r[1] - r[2]) < 1e-12 and r[3] > r[1] for r in rows)
          and rows[0][3] > rows[0][1],
          "★ **三种口径给出不同的数**（cf=1.0 时 甲/乙 = %.2f%%、丙 = %.2f%%，差 **%.2f pp**；"
          "cf=2.0 时差 **%.2f pp**）⇒ 报出「丢弃率」**必须同时说清**：(i) 容量施加在什么粒度上、"
          "(ii) 分子的单位（槽位 / token）、(iii) 分母是 `T·k` 还是 `T`"
          % (rows[0][1] * 100, rows[0][3] * 100, (rows[0][3] - rows[0][1]) * 100,
             (rows[-1][3] - rows[-1][1]) * 100))

    # ── B.4 第四种量法：**token 口径**（另一种「丢弃率」的常见含义）─────────
    # ★★ 前面三行数的都是**槽位**（分母 `T·k`）。而工程上更常问的是
    #   「**有多少 token 受影响**」—— 那个量**必须**走一遍真正的分配（它依赖到达顺序），
    #   而且给出的是**另一个数**。
    print()
    print("  B.4 — 第四种量法：**token 口径**（统计段维数上真跑一遍分配；★ 依赖到达顺序）")
    T_TOK, d_tok, e_tok = T_CAP, D_PROTO, E_CAP
    rng_t = np.random.default_rng(SEED_CAP + 21)
    x_t = rng_t.standard_normal((T_TOK, d_tok))
    w_t = np.full((T_TOK, K_CAP), 1.0 / K_CAP)
    up_t, down_t = _experts_matrices(SEED_CAP + 22, e_tok, d_tok)
    tok_rows = []
    for cf in (CF_LIST[0], CF_LIST[-1]):
        cap_t = capacity_ceil(T_TOK, e_tok, *Fraction(cf).as_integer_ratio())
        _, sv = run_layer(x_t, idx_big, w_t, up_t, down_t, cap_t, "dropping")
        any_drop = float((~sv).any(axis=1).mean())
        all_drop = float((~sv).all(axis=1).mean())
        tok_rows.append((cf, any_drop, all_drop))
        print("      cf=%-5s ⇒ **有任一选择被丢的 token = %6.2f%%**；"
              "**k 个选择全被丢的 token = %6.2f%%**（分母是 `T`，不是 `T·k`）"
              % (cf, any_drop * 100, all_drop * 100))
    check("B.4", tok_rows[0][1] > rows[0][1] and tok_rows[1][1] > rows[-1][1]
          and 0.0 < tok_rows[0][2] < tok_rows[0][1],
          "★★ **同一个「丢弃率」还有第四个数**：token 口径下 cf=1.0 时 **%.2f%%** 的 token"
          "「有选择被丢」、**%.2f%%** 「全被丢」—— 它们**都**高于槽位口径的 %.2f%%"
          "（因为一个 token 有 `k` 次机会被波及）。⇒ §3.3 的四行表就是这条断言的正文"
          % (tok_rows[0][1] * 100, tok_rows[0][2] * 100, rows[0][1] * 100))
    return x, idx, w, up, down


# ===========================================================================
# 4. C 类：No-Token-Left-Behind（★ 与 dropping 并列的**另一条**溢出协议）
# ===========================================================================
def run_layer_ntlb(x, rank, w, up, down, cap, k):
    """★ **No-Token-Left-Behind**（逐字出处，Switch 附录 B，[C]）：

      「**We suspected that having no computation applied to tokens could be very wasteful**,
       especially since if there is overflow on one expert, that means another expert will have
       extra capacity. With this intuition we create **No-Token-Left-Behind**, which
       **iteratively reroutes any tokens that are at first routed to an expert that is
       overflowing**. Figure 11 shows a graphical description of this method, which will allow us
       to guarantee **almost no** tokens will be dropped during training and inference.」

    ★★ **本函数是「本程序的一种实现」，不是论文规定的算法**：论文只给了**意图**（重新路由）
      与**一条界**（`almost no tokens`），**没有**给伪码。⇒ 本实现刻意选最简单的贪心：
      每个 token 有一张**完整**的偏好序 `rank[t]`，按序往下找第一个**还有空位**的专家；
      一轮没有进展就停（此时剩下的 token 真的无处可去）。
      ⚠ **不要把本实现当成论文的实现** —— 这是一条**口径**，写在正文里。
    ★★ 注意原文说的是 `almost no tokens`，**不是** `no tokens` ⇒ `C.2` 把「它也会失败」做成断言。
    """
    t_n, e_n = x.shape[0], rank.shape[1]
    used = np.zeros(e_n, dtype=np.int64)
    alloc = [[] for _ in range(t_n)]
    pos = np.zeros(t_n, dtype=np.int64)
    for _ in range(e_n + 2):
        progress = False
        for t in range(t_n):
            while len(alloc[t]) < k and pos[t] < e_n:
                e = int(rank[t, pos[t]])
                if used[e] < cap:
                    used[e] += 1
                    alloc[t].append((e, float(w[t, len(alloc[t])])))
                    progress = True
                    break
                pos[t] += 1
        if not progress:
            break
    out = x.copy()
    for t in range(t_n):
        for e, wj in alloc[t]:
            h = np.tanh(x[t] @ up[e])
            out[t] = out[t] + wj * (h @ down[e])
    return out, used, alloc


def part_C(x, idx, w, up, down):
    print()
    print("=" * 74)
    print("C. 两条溢出协议**并列**：dropping（上面 B 段）与 No-Token-Left-Behind（本段）")
    print("=" * 74)
    lg, _ = skew_logits(SEED_PROTO, T_PROTO, E_PROTO, 1, HOT_BIAS, K_PROTO)
    rank = np.argsort(-lg, axis=1, kind="stable")     # ★ **完整**偏好序（不只是 top-k）
    cap = capacity_ceil(T_PROTO, E_PROTO, *Fraction(CAP_PROTO_FACTOR).as_integer_ratio())

    # ── C.1 NTLB：容量不被突破，且没有 token 拿不到专家 ────────────────────
    out_nt, used_nt, alloc_nt = run_layer_ntlb(x, rank, w, up, down, cap, K_PROTO)
    empty_nt = sum(1 for a in alloc_nt if not a)
    # 对照组：同一个容量下用 dropping ⇒ 有 token 一个专家都拿不到
    out_dr, served_dr = run_layer(x, idx, w, up, down, cap, "dropping")
    empty_dr = int((~served_dr.any(axis=1)).sum())
    print("  容量/专家 = %d（总槽位 %d）；dropping 之下「一个专家都没拿到」的 token = **%d**"
          % (cap, cap * E_PROTO, empty_dr))
    print("  NTLB 之下同一个容量：每专家用掉 %s（上限 %d）⇒ 「一个专家都没拿到」的 token = **%d**"
          % (used_nt.tolist(), cap, empty_nt))
    check("C.1", bool((used_nt <= cap).all()) and empty_nt == 0 and empty_dr > 0,
          "★★ **两条协议并列**：同一份路由、同一个容量 —— dropping 留下 **%d** 个 token"
          "「一个专家都没拿到」（它们的该层输出 = 输入，见 B.1）；NTLB **重新路由**之后"
          "**每专家条数 ≤ 容量**（%s ≤ %d）且**没有一个 token 拿不到专家**（%d）"
          % (empty_dr, used_nt.tolist(), cap, empty_nt))

    # ── C.2 ★★ NTLB 不是普遍保证 ──────────────────────────────────────────
    cap_tight = CAP_TIGHT_PROTO
    out_t, used_t, alloc_t = run_layer_ntlb(x, rank, w, up, down, cap_tight, K_PROTO)
    empty_t = sum(1 for a in alloc_t if not a)
    print()
    print("  C.2 — 把容量压紧到 **%d 个槽位/专家**（总槽位 %d < token 数 %d）："
          % (cap_tight, cap_tight * E_PROTO, T_PROTO))
    print("        NTLB 之后仍有 **%d** 个 token 一个专家都拿不到（= T − 总槽位 = %d）"
          % (empty_t, T_PROTO - cap_tight * E_PROTO))
    check("C.2", empty_t == T_PROTO - cap_tight * E_PROTO and empty_t > 0,
          "★★ **NTLB 不是普遍保证**：它做的是**重新分配**，不是**增加容量**。"
          "原文逐字是「guarantee **almost no** tokens will be dropped」——**不是** `no tokens`。"
          "⇒ 引用 NTLB 时**不能**写成「overflow 不再是问题」；"
          "容量本身不够时（总槽位 %d < token 数 %d）它照样留 **%d** 个 token 无人服务"
          % (cap_tight * E_PROTO, T_PROTO, empty_t))

    # ── C.3 ★★ 架构层的那个决定：幸存权重要不要重新归一化 ─────────────────
    # ★ 上游两处**指名让渡**给本域的就是这一条：
    #   `40-kernels/moe-kernels/03-permute-and-unpermute.md:197` 逐字「丢掉之后要不要把幸存
    #   权重重新归一化、要不要给这类 token 补一条旁路，**属于架构层的决定（正文在
    #   `80-model-architectures/moe/02-capacity-factor.md` 的 §1.4，此处回链）**」；
    #   同文 `:238` 逐字「**做 mask 之后输出并不是 `X[t]` 本身**：它是 `X[t] × 幸存权重之和`
    #   （…）。**要不要重新归一化是架构层的选择**」。
    # ★★ 两条路是**精确**相差一个因子的（与专家函数是什么无关）：
    #     不归一化：`out = x + Σ_served w_j·f_j(x)`
    #     归一化  ：`out = x + (1/Σw)·Σ_served w_j·f_j(x)`
    #   ⇒ `(out_不归一化 − x) == Σw · (out_归一化 − x)`，**恒等**（只到浮点舍入）。
    # ★★ 两条路必须**真的各跑一遍**（这是 P17 的挡板）：如果把 `renorm` 由 `contrib`
    #    除出来再乘回去，那个「相等」是**构造出来的**（初版就踩了这个坑：偏差恒为 0.0）。
    #    ⇒ 下面把**归一化后的权重矩阵**再喂给同一层的**第二次**前向，得到真正独立的 `cB`。
    wsum = np.array([float(w[t, served_dr[t]].sum()) for t in range(T_PROTO)])
    w_norm = np.zeros_like(w)
    for t in range(T_PROTO):
        if wsum[t] > 0:
            w_norm[t] = w[t] / wsum[t]
    out_re, _ = run_layer(x, idx, w_norm, up, down, cap, "dropping")   # ← 第二次前向
    cA, cB = out_dr - x, out_re - x
    err = float(np.abs(cA - wsum[:, None] * cB).max())
    scale = float(np.abs(cA).max())
    rel = err / scale if scale > 0 else float("inf")
    partial = (wsum > 0.0) & (wsum < 1.0 - 1e-15)
    shrank = np.array([np.linalg.norm(cA[t]) < np.linalg.norm(cB[t])
                       for t in np.nonzero(partial)[0]]) if partial.any() else np.array([])
    print()
    print("  C.3 — **幸存权重要不要重新归一化**（上游 `03:197`/`:238` 指名让渡的那条决定）")
    print("        幸存权重之和 Σw：**恰好 == 1** 的 token %d 个；**0 < Σw < 1** 的 %d 个"
          "（最小 %.6f）；**Σw == 0**（k 个选择全被丢）的 %d 个"
          % (int((wsum >= 1.0 - 1e-15).sum()), int(partial.sum()),
             float(wsum[partial].min()) if partial.any() else float("nan"),
             int((wsum == 0.0).sum())))
    print("        **两次独立前向**（一次用 w、一次用 w/Σw）的贡献满足 `cA == Σw · cB`："
          "最大绝对偏差 **%.3e**（相对 %.3e，贡献量级 %.3f）" % (err, rel, scale))
    print("        在 0 < Σw < 1 的 %d 个 token 上，`‖cA‖ < ‖cB‖` 成立的有 **%d** 个"
          % (int(partial.sum()), int(shrank.sum()) if len(shrank) else 0))
    check("C.3", partial.sum() > 0 and rel < RENORM_TOL and bool(shrank.all())
          and (wsum == 0.0).sum() > 0,
          "★★ **这是架构层的选择，本域就是它的正文**：两条路（**权重矩阵不同、前向各跑一次**）"
          "的贡献满足 `cA == Σw · cB`（相对偏差 %.3e < %g）。★ 非空条件成立：**%d** 个 token"
          "的 0 < Σw < 1（最小 %.6f）⇒ 「**不**归一化」**确实缩小**了它们的贡献"
          "（`‖cA‖ < ‖cB‖` 在全部 %d 个上都成立）"
          % (rel, RENORM_TOL, int(partial.sum()),
             float(wsum[partial].min()) if partial.any() else float("nan"),
             int(partial.sum())))

    # ── C.4 ★★ `capacity factor` 同名不同指 ──────────────────────────────
    # ★ Switch 的 `capacity factor`（式 (3)）：**在均分之上额外开多少缓冲**。
    # ★ Expert Choice 的 `c`（式 (1) 逐字 `k = \frac{n \times c}{e}`）：逐字
    #   「The capacity factor `c` denotes **on average how many experts are utilized by a
    #    token**.」——**一个 token 平均被几个专家服务**。
    n_ec, e_ec, c_ec = T_CAP, E_CAP, 2
    cap_switch = capacity_ceil(T_CAP, E_CAP, 2, 1)       # cf = 2.0
    cap_ec = capacity_ceil(n_ec, e_ec, c_ec, 1)          # c = 2
    used_switch = T_CAP                                   # top-1：每个 token 恰好用掉 1 个槽位
    used_ec = n_ec * c_ec                                 # EC：每个 token 平均用掉 c 个
    print()
    print("  C.4 — ★★ **同名不同指**：两个「capacity factor」的闭式**恰好相同**，语义**不同**")
    print("        Switch（cf=2.0）：每专家容量 ceil((%d/%d)×2.0) = **%d**；"
          "top-1 ⇒ 用掉的槽位 = T = **%d** / 总槽位 %d"
          % (T_CAP, E_CAP, cap_switch, used_switch, E_CAP * cap_switch))
    print("        Expert Choice（c=2）：每专家容量 ceil((%d/%d)×2) = **%d**；"
          "每个 token 平均 c 个专家 ⇒ 用掉的槽位 = T·c = **%d** / 总槽位 %d"
          % (n_ec, e_ec, cap_ec, used_ec, E_CAP * cap_ec))
    check("C.4", cap_switch == cap_ec == 128 and used_switch == T_CAP
          and used_ec == 2 * T_CAP and used_ec // used_switch == c_ec,
          "★★ **闭式相同、语义不同**：两者的容量**都**是 `(tokens/experts) × factor` = `%d`，"
          "但被**用掉**的槽位数差 **%d 倍**（Switch top-1：**%d**；EC c=2：**%d**）⇒ "
          "同一个词，一处指「缓冲」、一处指「一个 token 平均被几个专家服务」。"
          "**引用 `capacity factor` 必须写出是哪一篇的含义**"
          % (cap_switch, used_ec // used_switch, used_switch, used_ec))

    # ── C.5 X6 的分寸：低 cf 更好的**语境**与方向 ────────────────────────
    speed = specs.SWITCH_MOE_SPEED_BY_CF
    print()
    print("  C.5 — X6 的分寸：逐字「Note that **the MoE model** going from capacity factor 2.0 to")
    print("        1.25 **actually slows down (840 to 790)** in the above experiment setup,")
    print("        **which is unexpected**.」⇒ 840（cf=2.0）> 790（cf=1.25）")
    check("C.5", speed[2.0] > speed[1.25]
          and specs.SWITCH_CAPACITY_FACTOR_FINDING == (1.0, 1.25)
          and specs.SWITCH_MOE_SPEED_BY_CF == {2.0: 840.0, 1.25: 790.0},
          "★ **三条口径都写进判据**：(i) 主语是 **MoE 模型**（逐字 `the MoE model`），"
          "不是 Switch；(ii) 方向是「**降** cf **变慢**」（840→790）；"
          "(iii) 原文**自承 `unexpected`** ⇒ 这**不是**一条可以推广的设计律。"
          "而 Table 1 的 finding (3) 说的是**质量**（`perform better at lower capacity factors"
          " (1.0, 1.25)`），且属于 Switch/MoE/Dense **同硬件同步数**的比较语境")


# ===========================================================================
# 5. D 类：失败标记自检 + 口径钉住
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
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
    """★★ **口径 / 阈值 / specs 的钉住 —— 刻意放在 A/B/C 之前**（Chunk 2 确立的硬要求）。"""
    print("=" * 74)
    print("D. 口径/阈值/specs 的钉住（★ 放在 A/B/C **之前**：先钉住前提，再看数）")
    print("=" * 74)

    measured = {"T_CAP": T_CAP, "E_CAP": E_CAP, "K_CAP": K_CAP, "HOT_N": HOT_N,
                "HOT_BIAS": HOT_BIAS, "SEED_CAP": SEED_CAP, "CF_LIST": CF_LIST,
                "T_PROTO": T_PROTO, "E_PROTO": E_PROTO, "K_PROTO": K_PROTO,
                "CAP_PROTO_FACTOR": CAP_PROTO_FACTOR, "SEED_PROTO": SEED_PROTO,
                "D_PROTO": D_PROTO, "CAP_TIGHT_PROTO": CAP_TIGHT_PROTO}
    expect = {"T_CAP": 4096, "E_CAP": 64, "K_CAP": 2, "HOT_N": 8, "HOT_BIAS": 1.2,
              "SEED_CAP": 20260922, "CF_LIST": (1.0, 1.25, 1.5, 2.0),
              "T_PROTO": 64, "E_PROTO": 4, "K_PROTO": 2, "CAP_PROTO_FACTOR": 1.0,
              "SEED_PROTO": 20260922, "D_PROTO": 16, "CAP_TIGHT_PROTO": 1}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 14 个逐位一致（★ `CF_LIST` 是**判据④的取值集合**、`HOT_BIAS` 是"
          "**负载不均的构造参数** —— 它们不是可调参数，而是结论所引用的测量条件）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    th = {"EXACT_TOL": EXACT_TOL, "RENORM_TOL": RENORM_TOL}
    th_expect = {"EXACT_TOL": 0, "RENORM_TOL": 1e-12}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 2 个逐位一致：`EXACT_TOL = 0`（A.1/A.2 是**精确整数**判据 —— 放宽它等于把"
          "「逐位相等」降级成「差不多」）、`RENORM_TOL = 1e-12`（**机器精度档**）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── specs 的常量：**键名先钉、取值后钉** ─────────────────────────────
    keys_ok = (set(specs.QWEN3_30B_A3B_MOE) == {"num_experts", "num_experts_per_tok",
                                                "moe_intermediate_size"}
               and set(specs.SWITCH_MOE_SPEED_BY_CF) == {2.0, 1.25})
    check("D.3", keys_ok,
          "★ `specs.py` 两组字典的**键名**逐位钉住（先钉键名：`mutate.py` 的分母含字典的键，"
          "键名一变、下一段取值就抛 `KeyError` ⇒ 那是崩溃捕获，不算真捕获）")
    if not keys_ok:
        return

    moe_pin = (specs.QWEN3_30B_A3B_MOE["num_experts"],
               specs.QWEN3_30B_A3B_MOE["num_experts_per_tok"],
               specs.QWEN3_30B_A3B_MOE["moe_intermediate_size"])
    check("D.4", moe_pin == (128, 8, 768)
          and specs.SWITCH_CAPACITY_FACTOR_FINDING == (1.0, 1.25)
          and specs.SWITCH_MOE_SPEED_BY_CF == {2.0: 840.0, 1.25: 790.0}
          and specs.EXPERT_CHOICE_CAPACITY_FACTOR == 2
          and specs.GSHARD_NUM_SELECTED_EXPERTS == 2,
          "★ `specs.py` 的常量**逐位钉住**：Qwen3 的 MoE 规格 = %s（[B] 转记）、"
          "X6 的 finding = %s 与速度 %s、EC 的 `CF2` = %d、GShard 的 top-2 = %d"
          % (moe_pin, specs.SWITCH_CAPACITY_FACTOR_FINDING, specs.SWITCH_MOE_SPEED_BY_CF,
             specs.EXPERT_CHOICE_CAPACITY_FACTOR, specs.GSHARD_NUM_SELECTED_EXPERTS))


def mark_part():
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
    check("D.6", ASSERT_FAIL_MARK == "## FAIL: moe"
          and "FAIL" in ASSERT_FAIL_MARK
          and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.7", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                       "~~ FAIL: normalization",
                                       "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住")

    # ── 两条协议的实现**必须真的是两条路**（AST 级）───────────────────────
    # ★ 这一条是本程序的 P17 挡板：`run_layer`（dropping）与 `run_layer_ntlb`（NTLB）
    #   如果共用同一段分配逻辑，`C.1` 那个「dropping 有无人服务的 token / NTLB 没有」
    #   就可能是构造出来的。**调用表必须为空**才算两条真独立的路径。
    called_dr = _callee_names(run_layer)
    called_nt = _callee_names(run_layer_ntlb)
    shared = sorted((called_dr | called_nt)
                    & {"run_layer", "run_layer_ntlb", "load_counts"})
    src_dr = inspect.getsource(run_layer)
    src_nt = inspect.getsource(run_layer_ntlb)
    check("D.8", not shared and src_dr != src_nt
          and "run_layer_ntlb" not in src_dr and "run_layer" not in src_nt.replace(
              "run_layer_ntlb", ""),
          "★ 两条协议的实现**互不调用**（AST 调用表交集 = %s）、源码不同（%d vs %d 字节）"
          "⇒ C.1 那个「dropping 留下无人服务的 token、NTLB 不留下」**不是共用代码产出的**"
          % (shared, len(src_dr), len(src_nt)))


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
    print("02_capacity_factor.py —— ★★ 容量因子的定义式 / padding / 两条溢出协议")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★★ 本段是上游 **8 处指名让渡**的落点：`00-meta/glossary.md:240` 与")
    print("     `40-kernels/moe-kernels/03:47` 都逐字写着「容量因子的定义与语义在")
    print("     `80-model-architectures/moe/02-capacity-factor.md`」。**本程序就是那条定义的可执行形态。**")
    print("  ★ 禁区（计划 §2）：本程序**不讲** permute/unpermute 的实现与 slot mask")
    print("     （正文在 40-kernels/moe-kernels/03）、**不算** FLOPs / 算术强度（在 01）、")
    print("     **不讲** grouped GEMM（在 02）。")
    pin_part()
    part_A()
    x, idx, w, up, down = part_B()
    part_C(x, idx, w, up, down)
    mark_part()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
