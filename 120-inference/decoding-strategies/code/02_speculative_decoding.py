#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_speculative_decoding.py —— **投机解码**：三条独立路径 + 三条假设的边界。

配套文档：`02-speculative-decoding.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本域最好的一条 A 类：同一个量走【三条互不共享代码】的路径**：
  · **闭式**（`Leviathan 2023`，`Eq.(1)`）：`E[#tokens] = (1 − α^{γ+1}) / (1 − α)`；
  · **解析**（精确，对**任意**接受率序列）：`E[每轮产出] = 1 + Σ_{j=1}^{γ} Π_{i≤j} α_i`；
  · **蒙特卡洛**：逐轮取样（**每个位置各取一个独立均匀数**），直到拒绝。
  ★★★ **恒定 `α` 时三者一致**（闭式与解析在 `1e-12` 内；MC 在 `4σ` 内）。
  ★★★ **`α` 随位置变时三者【不再一致】** —— ★★ **闭式的偏差【方向由序列形状决定】**（`A.2`）：
    **前高后低** ⇒ 闭式**低估**；**前低后高** ⇒ 闭式**高估**；★ **几何平均也修不好**（`A.2`）。
  ★★ **守卫**（`D.4`/`D.5`）：把解析式改成「只乘不累加」⇒ 三者对不上；
    把 MC 改成「用一个均匀数走完全程」⇒ 与解析对不上。**两端都过 `CONTRIBUTING:235`**。

★★★ **三条假设**（论文**逐字**写的，`specs.SPEC_ASSUMPTION_*`）：
  ① 「**the βs are i.i.d.**」（接受率独立同分布）· ② 「**run γ + 1 concurrent evaluations of
  Mp in parallel without increasing the walltime**」（有足够算力不增墙钟）·
  ③ 「**long enough generations**」（生成足够长）。★ **引用闭式必须连这三条一起引**（`F8`）。

★★ **一处【同名不同指】**：论文 **Thm 3.11** 是**算术运算量【增加】因子**、
  **Thm 3.8** 是**墙钟加速比** —— **方向相反、不是同一个量**（`specs.SPEC_THM38_VS_THM311_*`）。

★★★ **本机的限定语**：**「不是任何真实引擎，也不是真机」** —— 本机**无 GPU、无任何目标 LLM**；
  ★★ 闭式里的 `c` 是**硬件与实现的属性**（`specs.SPEC_C_HARDWARE_QUOTE`）⇒ **本机给不出 `c`**，
  本程序对 `c` 只作**几何演示**，**不得**称实测。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **恒定 `α` 下三条路径一致**（闭式 == 解析；MC 在 `4σ` 内）；
  · `A.2` ★★★ **`α` 随位置变时闭式偏了**：前高后低 **低估**、前低后高 **高估**、几何平均**也修不好**；
  · `A.3` ★★ **任意序列上解析 == MC**（精确公式对**变**接受率也成立）；
  · `A.4` ★★ **收支平衡点 `α*`**（二分 vs 线性扫描 ★ **共享目标函数 `closed_speed`** ⇒ 只证根查找一致性，
    ★ **弱于** `A.1`/`A.2`，`F4`）；
  · `B.*` 代价模型（`γ` 的最优点 ★ 七格全钉 · 两个负收益守卫 · 成本分解）；
  · `C.1`–`C.3` 边界（`α→0` ⇒ `E→1`；`α→1` ⇒ `E=γ+1`；`γ=0` ⇒ `1`）；
  · `C.4` ★★ **失败复现**（`CONTRIBUTING:238`）：闭式无条件引用 ⇒ 与精确值偏 `> 5%`；
  · `D.4`/`D.5` ★★★ **守卫实验**：改坏解析式 / 改坏 MC ⇒ 断言必须失败；
  · `D.9` ★★★ **术语纪律的两半**：全称存在 + **裸写被禁裸词的计数 == 显式豁免清单**。
"""
import ast
import collections
import inspect
import math
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
ASSERT_FAIL_MARK = "-- FAIL: decoding-strategies"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

# ★★ **两条术语纪律**（`§2.2`）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
#   · 本域一律写「**自回归解码**」全称（不裸写 `decode`，那是 `60-compilers` 的 decode pass）；
#   · 本域一律写「**KV 缓存**」全称（不裸写 `cache`，那是 L1/L2 cache）；
#   · 本域最强的一句**限定语**必须逐字出现。
TERM_AUTOREGRESSIVE = "自回归解码"
TERM_KV_CACHE = "KV 缓存"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ 形态 A：「纪律宣称了而断言没盯住」）：
#   本程序源码里**裸写那个被禁的裸词**（本域唯一的禁用裸词，见抬头的术语纪律）的【计数】
#   必须**恰好等于**下面这份**显式豁免清单**覆盖的计数 ——
#   ★★★ 判据是【计数 == 豁免覆盖数】，**不是「≤ N」**（「≤ N」挡不住**新增**一处裸写，正是本轮 `D1` 的形态）。
#   ★ 每一项都写明【为什么豁免】（★ **本注释块刻意不写那个裸词**，否则它自己也会被计数）：
#     ① 抬头注释里**提到**这条纪律本身、**并指出**该词在本库的另一个所指 ——
#        ★ 两处都是 **mention**（它写该词恰恰因为在规定不许写它）。
#     ② `D.9` 的判据文案里**复述**这条纪律 —— ★ 同为 mention。
BARE_DECODE_EXEMPT = (
    "不裸写 `decode`，那是 `60-compilers` 的 decode pass",   # ① 抬头：mention（规定 + 消歧）
    "**不裸写** `decode`",                                    # ② D.9 文案：mention（复述纪律）
)

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

NUM_EPS = 1e-12                     # ★ 闭式 vs 解析的容差（两条路径在幂次分母上约 4 ULP 之差）
MC_SIGMA = 4.0                      # ★ MC 的接受门槛（标准误的倍数）

# ★★★ **本机实测值**（本程序在本机跑出的数）—— 正文里的每个数都在这里**钉死**。
#   ★★ 单位口径：`PINNED_*_TOKENS` 是**期望产出 token 数**（`Eq.(1)`），
#      `PINNED_*_SPEED` 是**墙钟加速比**（`Thm 3.8`）—— ★ **两者不是同一个量**。
PINNED_CLOSED_08 = 3.3616                 # ★ `α = 0.8, γ = 4` 的 `E[#tokens]`
PINNED_SPEED_08 = 1.6808                  # ★ 同参数、`c = 0.25` 的加速比
PINNED_MC_08 = 3.362670                   # ★ 同参数的蒙特卡洛（`2e5` 次）
PINNED_DESC_EXACT = 3.2455                # ★ **前高后低**（`linspace(0.9, 0.5, 4)`）的精确值
PINNED_DESC_MEAN = 0.7                     # ★ 该序列的**算术平均**（★ 与升序、恒定两档**相同**）
PINNED_DESC_GEO = 0.6836957187             # ★ 该序列的**几何平均**（★ 与升序档**相同**，与恒定档不同）
PINNED_DESC_CLOSED_MEAN = 2.7731          # ★ 同序列**代入算术平均 `0.7`** 的闭式
PINNED_DESC_CLOSED_GEO = 2.6892221687     # ★ 同序列**代入几何平均 `0.6836957187`** 的闭式
PINNED_ASC_EXACT = 2.2779444444           # ★ **前低后高**（反例）的精确值
PINNED_CONST_EXACT = 2.7731               # ★ **恒定 `0.7`** 的精确值（= 闭式，★ 正是论文的假设）
PINNED_BIAS_DESC_PCT = -14.5555           # ★ 前高后低：闭式（算术平均）的偏差（**负** = 低估）
PINNED_BIAS_ASC_PCT = 21.7369             # ★ 前低后高：同一条闭式的偏差（**正** = 高估）
PINNED_BIAS_DESC_GEO_PCT = -17.1400       # ★ 前高后低：**几何平均**的偏差（★ 仍为负 ⇒ **没修好**）
PINNED_SPEED_C1 = 0.67232                 # ★ `c = 1.0`（草稿与目标同速）⇒ 闭式 **< 1**（负收益）
PINNED_SPEED_C2 = 0.3735111111            # ★ `c = 2.0`（草稿更慢）⇒ 更负
PINNED_SPEED_SMALL_ALPHA = 0.555550       # ★ `α = 0.1`（接受率太低）⇒ **即使草稿快也 < 1**
PINNED_ALPHA_STAR = 0.518790063676        # ★ `γ = 4, c = 0.25` 的**收支平衡**接受率 `α*`
PINNED_GAMMA_STAR = 3                     # ★ `α = 0.8, c = 0.25` 下**最优草稿长度** `γ*`
PINNED_SPEED_GAMMA_STAR = 1.686857        # ★ `γ*` 处的加速比
PINNED_SPEED_GAMMA16 = 0.977482           # ★ `γ = 16` 处**已经亏了**（`< 1`）
#   ★★ **§1 第四步那张 `γ` 表的【七格】**（★ **逐格钉** —— 不是只钉 `max`；★ 表里**没有** `γ = 5`）：
PINNED_GAMMA_GRID = (1, 2, 3, 4, 6, 8, 16)                                       # ★ 表里的 `γ` 列
PINNED_SPEED_GAMMA = (1.4400, 1.6267, 1.6869, 1.6808, 1.5806, 1.4430, 0.9775)    # ★ 对应的加速比（逐格）
PINNED_TOKENS_LIMIT = 5.0                 # ★ `γ → ∞` 时 `E[#tokens] → α/(1−α) + 1`


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D）。★ 失败路径**不复述结论**。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name,
                                detail_fail if detail_fail is not None else
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立」的措辞是"
                                "**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 三条独立路径（★ 互不共享代码 —— `D.8` 在源码层核）
# ===========================================================================
def analytic(alphas, corrupt=None):
    """★★★ **解析路径**（**精确**，对任意接受率序列）：

         `E[每轮产出] = 1 + Σ_{j=1}^{γ} Π_{i≤j} α_i`

    ★ 读法：`1` 是**目标模型自己产出的那一个** token（`Algorithm 1` 至少跑一次 `Mp`）；
      第 `j` 项是「**前 `j` 个草稿 token 全被接受**」的概率。
    ★★ `corrupt="no_prefix"`（**只用于守卫实验**）：把前缀乘积写成**只乘当前位置**。
    """
    p = 1.0
    s = 0.0
    for a in alphas:
        p = a if corrupt == "no_prefix" else p * a
        s += p
    return 1.0 + s


def closed_tokens(alpha, gamma):
    """★★★ **闭式路径**（`Leviathan 2023`，`Eq.(1)`）：`E[#tokens] = (1 − α^{γ+1}) / (1 − α)`。

    ★★ **它只在 `i.i.d.` 假设下成立**（`specs.SPEC_ASSUMPTION_IID_QUOTE`）；★ `α` 用**平均值**代入。
    ★ `α = 1` 时退化为 `γ + 1`（分母为 0 ⇒ 取极限）。
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def closed_speed(alpha, gamma, c):
    """★★★ **墙钟加速比**（`Thm 3.8`）：`(1−α^{γ+1}) / ((1−α)(γc+1))`。

    ★★ **这是【墙钟】的量** —— ★ 与 `Thm 3.11` 的**算术运算量增加因子**不是同一个量（`F10`）。
    ★★ `c` 是**硬件与实现的属性**（`specs.SPEC_C_HARDWARE_QUOTE`）⇒ 本机**给不出**，此处**只作演示**。
    """
    return closed_tokens(alpha, gamma) / (gamma * c + 1.0)


def monte_carlo(alphas, n, seed, corrupt=None):
    """★★★ **蒙特卡洛路径**：逐轮取样直到拒绝。

    ★★ **每个位置各取一个【独立】均匀数**（`u_i ~ U(0,1)`），`u_i < α_i` 才算接受。
    ★★★ `corrupt="one_shot"`（**只用于守卫实验**）：**整轮只用一个均匀数** ——
      ★ 这会把各位置的接受事件**完全相关**起来，得到 `E = α·(γ+1) + (1−α)`，**是错的**。
    ★ 返回 `(均值, 标准误)`。
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(alphas, dtype=float)
    G = len(a)
    if corrupt == "one_shot":
        u = rng.random((n, 1))
        acc = np.full(n, G, dtype=np.int64)
        bad = (u[:, 0] >= a[0])
        acc[bad] = 0
    else:
        U = rng.random((n, G))
        fail = U >= a                       # ★ 各位置**独立**判定
        any_fail = fail.any(axis=1)
        first = np.argmax(fail, axis=1)     # ★ 首个失败的位置（0-based）= 被接受的草稿数
        acc = np.where(any_fail, first, G)
    v = 1.0 + acc.astype(float)             # ★ 加上目标模型自己的那一个 token
    m = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    return m, se


def linspace_seq(hi, lo, gamma):
    """**前高后低**的接受率序列：`γ` 个点从 `hi` 线性降到 `lo`。"""
    if gamma == 1:
        return [float(hi)]
    step = (lo - hi) / (gamma - 1)
    return [hi + step * i for i in range(gamma)]


def geo_mean(seq):
    return float(math.exp(sum(math.log(x) for x in seq) / len(seq)))


# ===========================================================================
# 2. A 类 —— ★★★ 三条独立路径 + 假设边界
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **三条独立路径**（闭式 / 解析 / 蒙特卡洛）—— ★★★ 本域最好的 A 类")
    print("=" * 78)
    G = specs.SPEC_GAMMA
    print("  ★★ `γ = %d`；三条路径：闭式（一个式子）、解析（累加前缀乘积）、MC（逐轮取样）。" % G)

    # ── A.1 ★★★ 恒定 α 下三条路径一致 ─────────────────────────────────
    print()
    print("  ── A.1 ★★★ **恒定 `α` 下三者一致**（闭式 == 解析；MC 在 `%.0fσ` 内）──" % MC_SIGMA)
    print("     %-8s %-14s %-14s %-16s %s" % ("α", "闭式", "解析", "MC (±σ)", "一致"))
    ok = True
    for al in specs.SPEC_ALPHA_GRID:
        cf = closed_tokens(al, G)
        an = analytic([al] * G)
        mc, se = monte_carlo([al] * G, specs.SPEC_MC_TRIALS, specs.SPEC_MC_SEED)
        good = abs(cf - an) < NUM_EPS and abs(mc - an) < MC_SIGMA * se
        ok = ok and good
        print("     %-8.2f %-14.6f %-14.6f %-16s %s"
              % (al, cf, an, "%.4f±%.4f" % (mc, se), "✓" if good else "✗"))
    closed_08 = closed_tokens(0.8, G)
    an_08 = analytic([0.8] * G)
    mc_08, se_08 = monte_carlo([0.8] * G, specs.SPEC_MC_TRIALS, specs.SPEC_MC_SEED)
    print("     ★★ `α = 0.8`：闭式 `%.4f` == 解析 `%.4f`；MC `%.4f`（差 `%.2fσ`）"
          % (closed_08, an_08, mc_08, abs(mc_08 - an_08) / se_08))
    print("     ★ 加速比（`Thm 3.8`，`c = %.2f`）= **%.4f**" % (specs.SPEC_C, closed_speed(0.8, G, specs.SPEC_C)))
    check("A.1", ok
          and abs(closed_08 - PINNED_CLOSED_08) < 2e-4
          and abs(mc_08 - PINNED_MC_08) < 2e-4
          and abs(closed_speed(0.8, G, specs.SPEC_C) - PINNED_SPEED_08) < 2e-4,
          "★★★ **判据成立**：`α ∈ %s` 上**闭式与解析在 `1e-12` 内一致**、**MC 在 `%.0fσ` 内吻合** ——"
          " ★★ 三条**互不共享代码**的路径在**恒定接受率**这个条件下会合（`α = %.1f` 时 "
          "`E = %.4f`、加速比 `%.4f`）"
          % (specs.SPEC_ALPHA_GRID, MC_SIGMA, 0.8, closed_08, closed_speed(0.8, G, specs.SPEC_C)),
          "**三条路径不一致** —— ★ 先查 MC 是否每个位置都取了**独立**均匀数（不是整轮一个）")

    # ── A.2 ★★★ α 随位置变 ⇒ 闭式的偏差方向由【序列形状】决定 ──────────
    print()
    print("  ── A.2 ★★★ **`α` 随【位置】变（第 ①条假设被破坏）⇒ 闭式偏了，方向由形状决定** ──")
    desc = linspace_seq(specs.SPEC_ALPHA_DESC_HI, specs.SPEC_ALPHA_DESC_LO, G)
    asc = desc[::-1]
    rows = []
    for nm, s in (("前高后低", desc), ("前低后高", asc), ("恒定", [0.7] * G)):
        m = float(np.mean(s))
        gg = geo_mean(s)
        ex = analytic(s)
        cf_m = closed_tokens(m, G)
        cf_g = closed_tokens(gg, G)
        rows.append((nm, m, gg, ex, cf_m, cf_g))
        print("     %-10s 序列 %s" % (nm, [round(x, 4) for x in s]))
        print("        mean=%.4f geo=%.4f | 精确=%.4f 闭式(算术平均)=%.4f (%+.2f%%) 闭式(几何平均)=%.4f (%+.2f%%)"
              % (m, gg, ex, cf_m, 100 * (cf_m - ex) / ex, cf_g, 100 * (cf_g - ex) / ex))
    d_ex, d_cm, d_cg = rows[0][3], rows[0][4], rows[0][5]
    a_ex, a_cm = rows[1][3], rows[1][4]
    c_ex, c_cm = rows[2][3], rows[2][4]
    d_bias = 100 * (d_cm - d_ex) / d_ex
    a_bias = 100 * (a_cm - a_ex) / a_ex
    d_geo_bias = 100 * (d_cg - d_ex) / d_ex
    print("     ★★★ **读法**：**同一个算术平均 `0.7`**，闭式给出**同一个数 `%.4f`** ——" % d_cm)
    print("         ★ **前高后低** ⇒ 精确 `%.4f` ⇒ 闭式**低估 `%.2f%%`**；" % (d_ex, abs(d_bias)))
    print("         ★ **前低后高** ⇒ 精确 `%.4f` ⇒ 闭式**高估 `%.2f%%`**。" % (a_ex, abs(a_bias)))
    print("         ★★ **几何平均也修不好**：偏 `%.2f%%`（**仍为低估**，只是幅度变了）。" % abs(d_geo_bias))
    print("     ★★ 机理：`E` 对序列是**多线性**的 —— 前缀乘积**偏爱「大值在前」**，"
          "所以**降序**最大、**升序**最小，而**只看平均值**的闭式落在中间。")
    check("A.2", d_bias < 0 and a_bias > 0 and d_geo_bias < 0
          and abs(c_ex - c_cm) < 1e-9
          and abs(rows[0][1] - PINNED_DESC_MEAN) < 1e-12
          and abs(rows[1][1] - PINNED_DESC_MEAN) < 1e-12
          and abs(rows[2][1] - PINNED_DESC_MEAN) < 1e-12
          and abs(rows[0][2] - PINNED_DESC_GEO) < 1e-9
          and abs(rows[1][2] - PINNED_DESC_GEO) < 1e-9
          and abs(rows[2][2] - PINNED_DESC_MEAN) < 1e-9
          and abs(d_ex - PINNED_DESC_EXACT) < 1e-4 and abs(d_cm - PINNED_DESC_CLOSED_MEAN) < 1e-4
          and abs(d_cg - PINNED_DESC_CLOSED_GEO) < 1e-6 and abs(a_ex - PINNED_ASC_EXACT) < 1e-4
          and abs(d_bias - PINNED_BIAS_DESC_PCT) < 1e-3 and abs(a_bias - PINNED_BIAS_ASC_PCT) < 1e-3
          and abs(d_geo_bias - PINNED_BIAS_DESC_GEO_PCT) < 1e-3,
          "★★★ **判据成立**：**同一个算术平均 `0.7` 下，闭式偏了、且【方向由形状决定】** ——"
          " 前高后低 **低估 `%.2f%%`**、前低后高 **高估 `%.2f%%`**；"
          " ★★ **几何平均也修不好**（仍低估 `%.2f%%`）；★ 恒定 `0.7` 时两者**恰好相等**（`%.4f`）"
          " —— ★★★ **这不是「闭式不准」，而是「论文的 `i.i.d.` 前提一旦不成立，偏差方向由序列形状定」**"
          % (abs(d_bias), abs(a_bias), abs(d_geo_bias), c_ex),
          "**闭式偏差的方向或数值被动过** —— ★ 先查序列构造（降序 vs 升序）与 `analytic` 的前缀乘积")

    # ── A.3 ★★ 任意序列上解析 == MC ────────────────────────────────────
    print()
    print("  ── A.3 ★★ **任意（非恒定）序列上解析 == MC**（精确公式对【变】接受率也成立）──")
    ok3 = True
    for nm, s in (("前高后低", desc), ("前低后高", asc)):
        ex = analytic(s)
        mc, se = monte_carlo(s, specs.SPEC_MC_TRIALS, specs.SPEC_MC_SEED)
        good = abs(mc - ex) < MC_SIGMA * se
        ok3 = ok3 and good
        print("     %-10s 解析 = %.6f ；MC = %.6f ± %.6f（差 %.2fσ）" % (nm, ex, mc, se, abs(mc - ex) / se))
    check("A.3", ok3,
          "★★ **判据成立**：**变**接受率序列上，**解析式与 MC 仍在 `%.0fσ` 内吻合** ——"
          " ★★ 说明「解析式对任意序列精确」这件事**独立于闭式是否偏**" % MC_SIGMA,
          "**解析与 MC 在变序列上对不上** —— ★ 先查 MC 的「逐位置独立」与解析的前缀乘积")

    # ── A.4 ★★ 收支平衡点 α*：二分 vs 线性扫描 ────────────────────────
    print()
    print("  ── A.4 ★★ **收支平衡接受率 `α*`**（`γ = %d, c = %.2f` ⇒ 解 `加速比 = 1`）──" % (G, specs.SPEC_C))
    # ★★ **降级声明**（★ `F4`）：本条的**两条搜索路径共享同一个目标函数** `closed_speed` ——
    #   ⇒ 它**只证「根查找」的一致性**（二分 vs 扫描），**不是** `A.1`/`A.2` 那种
    #     「**三条互不共享代码**的路径」⇒ ★★ **不与它们并列称 A 类**（`D.8` 只审
    #     `analytic`/`closed_tokens`/`monte_carlo` 三函数，**不覆盖** `A.4`）。
    print("     ★★ **降级声明**：本条两条路径**共享目标函数** `closed_speed` ⇒"
          " 只证「**根查找**」的一致性，**弱于** `A.1`/`A.2`（★ `D.8` 未覆盖本条）。")
    lo, hi = 0.4, 0.6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if closed_speed(mid, G, specs.SPEC_C) < 1.0:
            lo = mid
        else:
            hi = mid
    astar = (lo + hi) / 2.0
    # ★ 第二条路径：**线性扫描**（网格 + 首个越过的点）
    grid = [0.4 + i * 1e-4 for i in range(2001)]
    cross = [x for x in grid if closed_speed(x, G, specs.SPEC_C) >= 1.0]
    scan_lo = cross[0] if cross else None
    print("     ★ 二分 200 步 ⇒ `α*` = **%.12f**（加速比 %.12f）" % (astar, closed_speed(astar, G, specs.SPEC_C)))
    print("     ★ 线性扫描（步长 `1e-4`）⇒ 首个 `≥ 1` 的点 = **%.4f**" % (scan_lo,))
    print("     ★★ 读法：`α < α*` ⇒ **投机解码是净亏** —— 即使草稿比目标快 4 倍（`c = %.2f`）"
          % specs.SPEC_C)
    print("         ★★ 这是**第 ②条假设**（不增墙钟）之外的**第二条**负收益来源：**接受率太低**。")
    check("A.4", abs(astar - PINNED_ALPHA_STAR) < 1e-9
          and abs(closed_speed(astar, G, specs.SPEC_C) - 1.0) < 1e-9
          and scan_lo is not None and abs(scan_lo - astar) <= 2e-4
          and closed_speed(0.5, G, specs.SPEC_C) < 1.0,
          "★★ **判据成立**：**两条搜索路径**（二分、线性扫描）给出同一个 `α*` ≈ **%.9f**；"
          " ★★ 在它之下**加速比 `< 1`**（`α = 0.5` 时 `%.4f`）⇒ ★ **接受率太低时投机解码是净亏**"
          % (astar, closed_speed(0.5, G, specs.SPEC_C)),
          "**平衡点对不上** —— ★ 先查 `closed_speed` 与扫描/二分的单调性方向")


# ===========================================================================
# 3. B 类：代价模型 —— 最优草稿长度 · 两个负收益守卫 · 成本分解
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **代价模型**：`γ` 的最优点 · 两个负收益守卫 · 成本分解")
    print("=" * 78)
    G = specs.SPEC_GAMMA
    AL = 0.8

    # ── B.1 ★★ γ 有一个最优点（太小产量少、太大验证浪费）────────────────
    print()
    print("  ── B.1 ★★ **草稿长度 `γ` 有一个最优点**（`α = %.1f, c = %.2f`）──" % (AL, specs.SPEC_C))
    print("     %-8s %-16s %-14s %s" % ("γ", "E[#tokens]", "加速比", "备注"))
    rows = []
    for g in PINNED_GAMMA_GRID:
        tok = closed_tokens(AL, g)
        sp = closed_speed(AL, g, specs.SPEC_C)
        rows.append((g, tok, sp))
    best = max(((g, closed_speed(AL, g, specs.SPEC_C)) for g in range(1, 65)), key=lambda x: x[1])
    for g, tok, sp in rows:
        note = ""
        if g == best[0]:
            note = "★ **最优**"
        elif g == 16:
            note = "★★ **已亏**（`< 1`）"
        print("     %-8d %-16.6f %-14.6f %s" % (g, tok, sp, note))
    # ★★★ **七格逐格钉**（★ 正文 §1 第四步那张表的**每一格**，**不是只钉 `max`** ——
    #     ★ 这正是本轮 `D2`「整表未被断言」的修补）：
    cell_ok = all(abs(sp - pin) < 5e-5 for (_, _, sp), pin in zip(rows, PINNED_SPEED_GAMMA))
    print("     ★★ 七格逐格对照 `PINNED_SPEED_GAMMA` = %s ⇒ %s" % (PINNED_SPEED_GAMMA, cell_ok))
    print("     ★★ 扫描 `γ ∈ [1, 64]` ⇒ **`γ* = %d`**（加速比 **%.6f**）" % best)
    print("     ★★ 读法：`γ` 太小 ⇒ 每轮产出少（`γc+1` 的**分母**小但分子更小）；"
          "`γ` 太大 ⇒ **验证的浪费**（`γc+1`）压过产出 ⇒ ★ **存在最优点**。")
    check("B.1", cell_ok
          and best[0] == PINNED_GAMMA_STAR and abs(best[1] - PINNED_SPEED_GAMMA_STAR) < 1e-5
          and abs(closed_speed(AL, 16, specs.SPEC_C) - PINNED_SPEED_GAMMA16) < 1e-5
          and closed_speed(AL, 16, specs.SPEC_C) < 1.0,
          "★★ **判据成立**：`γ` 的扫描给出**内部最优** `γ* = %d`（加速比 `%.6f`）——"
          " ★★ 两端都变差：`γ = 1` 只 `%.4f`、`γ = 16` 已经 **`%.4f < 1`**（**净亏**）；"
          " ★★★ **正文表里的七格（`γ ∈ %s`）逐格钉住**（★ 不是只钉最优点）"
          % (best[0], best[1], closed_speed(AL, 1, specs.SPEC_C), closed_speed(AL, 16, specs.SPEC_C),
             PINNED_GAMMA_GRID),
          "**`γ` 的最优点或表里某一格被动过** —— ★ 先查 `closed_speed` 的 `(γc+1)` 分母"
          "与 `Eq.(1)` 的分子，以及 `PINNED_GAMMA_GRID`/`PINNED_SPEED_GAMMA` 的逐格对齐")

    # ── B.2 ★★ 两个负收益守卫：c ≥ 1（草稿不比目标快）、α 太小 ──────────
    print()
    print("  ── B.2 ★★ **负收益的两个来源**（★ C 类的免费守卫）──")
    print("     · `c = %.1f`（草稿与目标**同速**）⇒ 加速比 **%.6f < 1**" % (1.0, closed_speed(AL, G, 1.0)))
    print("     · `c = %.1f`（草稿**更慢**）    ⇒ 加速比 **%.6f < 1**" % (2.0, closed_speed(AL, G, 2.0)))
    print("     · `α = %.1f`（接受率**太低**）  ⇒ 加速比 **%.6f < 1**（★ 即使 `c = %.2f` 很快）"
          % (0.1, closed_speed(0.1, G, specs.SPEC_C), specs.SPEC_C))
    print("     ★★ **`c` 是硬件与实现的属性**（逐字见 `specs.SPEC_C_HARDWARE_QUOTE`）⇒"
          " **本机给不出 `c`**，这三行是**几何演示**，不是实测。")
    check("B.2", abs(closed_speed(AL, G, 1.0) - PINNED_SPEED_C1) < 1e-5
          and abs(closed_speed(AL, G, 2.0) - PINNED_SPEED_C2) < 1e-5
          and abs(closed_speed(0.1, G, specs.SPEC_C) - PINNED_SPEED_SMALL_ALPHA) < 1e-5
          and closed_speed(AL, G, 1.0) < 1.0 and closed_speed(AL, G, 2.0) < 1.0
          and closed_speed(0.1, G, specs.SPEC_C) < 1.0,
          "★★ **判据成立**：**三个免费守卫**都给出 `< 1` —— ★ `c ≥ 1`（草稿不比目标快）"
          "**必须**负收益；★★ 且 `α` 太小（`0.1`）时**即使草稿快 4 倍也亏**"
          "（这正是 `A.4` 的 `α*` 在另一处的表现）",
          "**负收益守卫失效** —— ★ `c ≥ 1` 时闭式竟然 `≥ 1`（那 `Eq.(1)` 或成本项写错了）")

    # ── B.3 ★★ 成本分解：E = 1 + 被接受的草稿数（且有上界 α/(1−α)）────────
    print()
    print("  ── B.3 ★★ **成本分解**：`E = 1`（目标自己产出）+ `被接受的草稿数`（有上界）──")
    acc = analytic([AL] * G) - 1.0
    lim = AL / (1.0 - AL)
    print("     `α = %.1f, γ = %d` ⇒ `E = %.4f` = `1` + `%.4f`（被接受）" % (AL, G, analytic([AL] * G), acc))
    print("     ★ `γ → ∞` 时被接受的草稿数 → `α/(1−α) = %.4f` ⇒ `E → %.4f`" % (lim, lim + 1.0))
    print("     ★★ 读法：`E` 的**上界是有限的** —— **加长 `γ` 不能无限提高产出**（★ 与 `B.1` 的最优点同源）。")
    check("B.3", abs(acc - (analytic([AL] * G) - 1.0)) < 1e-12
          and abs(lim + 1.0 - PINNED_TOKENS_LIMIT) < 1e-9
          and abs(closed_tokens(AL, 64) - PINNED_TOKENS_LIMIT) < 1e-4,
          "★★ **判据成立**：`E = 1 + 被接受数`；★ 被接受数的**上界是 `α/(1−α) = %.4f`** ⇒"
          " `E` 的上界 `%.1f`（★ `γ = 64` 时已到 `%.6f`）—— ★★ **产出不能靠加长 `γ` 无限提高**"
          % (lim, lim + 1.0, closed_tokens(AL, 64)),
          "**成本分解或上界被动过** —— ★ 先查解析式的 `1 +` 与几何级数的上界")


# ===========================================================================
# 4. C 类：边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：`α → 0` / `α → 1` / `γ = 0` 三个退化点")
    print("=" * 78)
    G = specs.SPEC_GAMMA

    # ── C.1 ★★ α → 0 ⇒ E → 1 ──────────────────────────────────────────
    print()
    print("  ── C.1 ★★ **`α → 0`（几乎全被拒）⇒ `E → 1`**（只剩目标自己那一个 token）──")
    for al in (0.01, 0.001):
        print("     `α = %.3f` ⇒ 闭式 %.9f ；解析 %.9f" % (al, closed_tokens(al, G), analytic([al] * G)))
    check("C.1", abs(closed_tokens(0.001, G) - 1.0) < 1e-2
          and abs(analytic([0.0] * G) - 1.0) < 1e-12
          and abs(closed_tokens(0.0, G) - 1.0) < 1e-12,
          "★★ **判据成立**：`α → 0` ⇒ `E → 1`（★ 每轮只产出目标自己的那一个 token）——"
          " ★ 这是「投机解码**至少不亏产出**」的下端",
          "**`α → 0` 的极限被动过** —— ★ 先查解析式的 `1 +` 常数项")

    # ── C.2 ★★ α → 1 ⇒ E = γ + 1 ──────────────────────────────────────
    print()
    print("  ── C.2 ★★ **`α → 1`（全接受）⇒ `E = γ + 1`**（草稿全收 + 目标那一个）──")
    print("     `α = 1.0` ⇒ 闭式 %.4f ；解析 %.4f" % (closed_tokens(1.0, G), analytic([0.999999] * G)))
    check("C.2", abs(closed_tokens(1.0, G) - (G + 1)) < 1e-12
          and abs(analytic([1.0] * G) - (G + 1)) < 1e-12,
          "★★ **判据成立**：`α = 1` ⇒ `E = γ + 1 = %d`（**上限**，草稿全被接受再加目标那一个）——"
          " ★ 与 `C.1` 合起来：`E ∈ [1, γ+1]`" % (G + 1),
          "**`α = 1` 的上限被动过** —— ★ 先查 `α = 1` 的除零分支是否取极限为 `γ+1`")

    # ── C.3 ★★ γ = 0 ⇒ E = 1（没有草稿）──────────────────────────────
    print()
    print("  ── C.3 ★★ **`γ = 0`（没有草稿）⇒ `E = 1`**（退化成普通自回归解码）──")
    print("     `γ = 0` ⇒ 闭式 %.4f ；解析 %.4f ；加速比（`c` 任意）= %.4f"
          % (closed_tokens(0.8, 0), analytic([]), closed_speed(0.8, 0, specs.SPEC_C)))
    check("C.3", abs(closed_tokens(0.8, 0) - 1.0) < 1e-12 and abs(analytic([]) - 1.0) < 1e-12
          and abs(closed_speed(0.8, 0, specs.SPEC_C) - 1.0) < 1e-12,
          "★★ **判据成立**：`γ = 0` ⇒ `E = 1`、**加速比恰好 `1`**（★ 没有草稿就既没有收益、也没有代价）"
          " —— ★ 这退化成普通的**自回归解码**，是投机解码的**中性边界**",
          "**`γ = 0` 的退化被动过** —— ★ 先查空序列上的解析式与闭式分子")

    # ── C.4 ★★ **§4.1 的反例：闭式【不能】无条件引用 —— 断言【失败真的发生】** ──────
    #   ★★ 与 C.1–C.3（**成功**的退化边界）不同：这一条断言的是**失败成立**（`CONTRIBUTING:238`
    #      要求 C 类复现「§4 声称会出问题的输入真的跑出问题」）。
    print()
    print("  ── C.4 ★★ **§4.1 的反例**：闭式在变接受率下【偏了】—— 断言【失败真的发生】──")
    desc = linspace_seq(specs.SPEC_ALPHA_DESC_HI, specs.SPEC_ALPHA_DESC_LO, G)
    asc = desc[::-1]
    bad_desc = abs(closed_tokens(float(np.mean(desc)), G) - analytic(desc)) / analytic(desc)
    bad_asc = abs(closed_tokens(float(np.mean(asc)), G) - analytic(asc)) / analytic(asc)
    print("     ★ 失败**真的发生**：闭式（代入算术平均）对精确值的**相对偏差** ——"
          " 前高后低 `%.2f%%`、前低后高 `%.2f%%`（★ 两档都 `> 5%%`）"
          % (100 * bad_desc, 100 * bad_asc))
    check("C.4", bad_desc > 0.05 and bad_asc > 0.05,
          "★★ **失败成立**（★ 与 `CONTRIBUTING:238` 的 C 类口径一致）：**把算术平均代进闭式**"
          " 得到的数与精确值**偏差 `> 5%%`**（前高后低 `%.2f%%`、前低后高 `%.2f%%`）——"
          " ★★ 于是 `§4.1`「把闭式当无条件」的反例**真的跑得出来**（★ 断言的是「失败复现」）"
          % (100 * bad_desc, 100 * bad_asc),
          "**反例不成立** —— ★ 若闭式与精确值在变序列上一致，则 `§4.1` 的失败模式**无法复现**"
          "（那 `A.2` 的前提也塌了）")


# ===========================================================================
# 5. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无任何目标 LLM**；")
    print("      ★★ 闭式里的 `c` 是**硬件与实现的属性** ⇒ **本机给不出 `c`**，此处只作几何演示。")
    before = len(FAILS)

    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    hits = [n.value for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "ASSERT_FAIL_MARK"]
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1 and uniq == {ASSERT_FAIL_MARK}
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: decoding-strategies" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: decoding-strategies`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                                        "-- FAIL: engines", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.DECODE_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == ()
          and abs(NUM_EPS - 1e-12) < 1e-24 and abs(MC_SIGMA - 4.0) < 1e-12,
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住；"
          "★ 两个容差（`NUM_EPS` / `MC_SIGMA`）**在 preflight 里就钉住**（★ 否则它们的变异会变成"
          "**崩溃捕获**、而不是断言捕获）" % specs.DECODE_FETCH_DAY)

    # ★★★ 口径守卫（★ 跑在任何计算之前）：`A.1` 要求网格**都 `< 1`**（`α = 1` 会让闭式除零，
    #   属边界、由 `C.2` 单独钉）；`A.2` 要求降序序列**真的在下降**；`c` 必须 `< 1`（否则 `B.2`
    #   的三行会**全部**退化成同一个符号、A.1 的「正收益」语境也没了）。
    desc = linspace_seq(specs.SPEC_ALPHA_DESC_HI, specs.SPEC_ALPHA_DESC_LO, specs.SPEC_GAMMA)
    ok_caliber = (all(0.0 < a < 1.0 for a in specs.SPEC_ALPHA_GRID)
                  and all(desc[i] > desc[i + 1] for i in range(len(desc) - 1))
                  and specs.SPEC_ALPHA_DESC_HI > specs.SPEC_ALPHA_DESC_LO
                  and 0.0 < specs.SPEC_C < 1.0
                  and specs.SPEC_GAMMA >= 2
                  and specs.SPEC_MC_TRIALS >= 10000)
    print("     ★★ 口径守卫：`α` 网格都 `< 1` = %s；降序序列**严格递减** = %s；`0 < c < 1` = %s；"
          "`γ ≥ 2` = %s"
          % (all(0.0 < a < 1.0 for a in specs.SPEC_ALPHA_GRID),
             all(desc[i] > desc[i + 1] for i in range(len(desc) - 1)),
             0.0 < specs.SPEC_C < 1.0, specs.SPEC_GAMMA >= 2))
    check("D.3b", ok_caliber,
          "★★★ **口径守卫**：`α` 网格**必须都不等于 1**（否则闭式除零，`A.1` 变崩溃捕获）；"
          "降序序列**必须真的递减**（否则 `A.2` 的「前高后低」不成立）；`c` **必须 `< 1`**"
          "（否则 `B.2` 的三行全同号、`A.2` 的正收益语境消失）",
          "**口径被动过** —— ★ 先查 `SPEC_ALPHA_GRID` / `SPEC_ALPHA_DESC_*` / `SPEC_C` / `SPEC_GAMMA`")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性 · 术语
# ===========================================================================
def _names_in(obj):
    src = textwrap.dedent(inspect.getsource(obj))
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 独立性 · 术语纪律")
    print("=" * 78)
    G = specs.SPEC_GAMMA

    # ── D.4 ★★★ 守卫实验 ①：把解析式的前缀乘积改成「只乘当前位置」──────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：解析式的**前缀乘积**改成「**只乘当前位置**」──")
    a = [0.8] * G
    good = analytic(a)
    bad = analytic(a, corrupt="no_prefix")
    print("     正常解析 = **%.6f** ；改坏解析 = **%.6f** ；闭式 = **%.6f**"
          % (good, bad, closed_tokens(0.8, G)))
    check("D.4", abs(good - closed_tokens(0.8, G)) < NUM_EPS
          and abs(bad - closed_tokens(0.8, G)) > 1e-3,
          "★★★ **守卫成立**：把前缀乘积改坏后，解析式从 **%.6f** 变成 **%.6f** ——"
          " ★★ 于是 `A.1` 的「闭式 == 解析」对「**前缀乘积**」这条规则**是敏感的**" % (good, bad),
          "**守卫失效** —— ★ 改坏前缀乘积后解析式居然仍等于闭式（那 `A.1` 抓不到它）")

    # ── D.5 ★★★ 守卫实验 ②：把 MC 改成「整轮只用一个均匀数」────────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：MC 改成「**整轮只用一个均匀数**」(`one_shot`) ──")
    mc_ok, se_ok = monte_carlo([0.8] * G, specs.SPEC_MC_TRIALS, specs.SPEC_MC_SEED)
    mc_bad, _ = monte_carlo([0.8] * G, specs.SPEC_MC_TRIALS, specs.SPEC_MC_SEED, corrupt="one_shot")
    pred_bad = 0.8 * (G + 1) + 0.2 * 1.0
    print("     正常 MC = **%.6f**（解析 %.6f）；改坏 MC = **%.6f**（= `α(γ+1)+(1−α)` = %.6f）"
          % (mc_ok, analytic([0.8] * G), mc_bad, pred_bad))
    check("D.5", abs(mc_ok - analytic([0.8] * G)) < MC_SIGMA * se_ok
          and abs(mc_bad - pred_bad) < 5e-3 and abs(mc_bad - analytic([0.8] * G)) > 0.5,
          "★★★ **守卫成立**：把 MC 改成「整轮一个均匀数」后，它给出 **%.6f**（= `α(γ+1)+(1−α)`），"
          "与解析 **%.6f** 差得很远 ⇒ ★★ `A.1` 对「**逐位置独立**」这条实现细节也敏感"
          % (mc_bad, analytic([0.8] * G)),
          "**守卫失效** —— ★ 改坏 MC 后居然仍与解析吻合（那 `A.1` 抓不到它）")

    # ── D.6 ★★★ 引文逐字：论文的公式、三条假设、`c` 的口径、两条定理的消歧 ──
    check("D.6", specs.SPEC_GAMMA == 4 and abs(specs.SPEC_C - 0.25) < 1e-15
          and specs.SPEC_ALPHA_GRID == (0.6, 0.7, 0.8)
          and abs(specs.SPEC_ALPHA_DESC_HI - 0.9) < 1e-15
          and abs(specs.SPEC_ALPHA_DESC_LO - 0.5) < 1e-15
          and specs.SPEC_MC_TRIALS == 200000 and specs.SPEC_MC_SEED == 20261009
          and specs.SPEC_EQ1_FORMULA == "E(# generated tokens) = (1 − α^{γ+1}) / (1 − α)"
          and specs.SPEC_THM38_WALLTIME_FORMULA == "(1−α^{γ+1}) / ((1−α)(γc+1))"
          and specs.SPEC_THM311_ARITHMETIC_FORMULA == "(1−α)(γĉ+γ+1) / (1−α^{γ+1})"
          and specs.SPEC_ALGORITHM_CITATION
          == "Leviathan, Kalman, Matias, Fast Inference from Transformers via Speculative Decoding, arXiv:2211.17192",
          "★★★ **论文的公式与算例参数逐字钉住**：`Eq.(1)`、`Thm 3.8`（墙钟）、`Thm 3.11`（运算量）"
          " 三条式子**分别**钉住；书目字段（arXiv:2211.17192）一并钉住",
          "**论文常量被动过** —— ★ 先查 `specs.py` 第 1 节")

    check("D.6b", specs.SPEC_ASSUMPTION_IID_QUOTE == "If we make the simplifying assumption that the βs are i.i.d., and denote α = E(β), then the number of tokens produced by a single run of Algorithm 1 is a capped geometric variable, with success probability 1 − α and cap γ + 1, and the expected number of tokens generated by Algorithm 1 satisfies Equation (1)."
          and specs.SPEC_ASSUMPTION_CONCURRENCY_QUOTE == "we'll assume that we can run γ + 1 concurrent evaluations of Mp in parallel without increasing the walltime"
          and specs.SPEC_ASSUMPTION_LONG_GENERATION_QUOTE == "Note that Theorem 3.8 assumes long enough generations (for example, since we run Mp at least once, the improvement factor is capped by the number of generated tokens)."
          and specs.SPEC_C_DEFINITION_QUOTE == "the cost coefficient, be the ratio between the time for a single run of Mq and the time for a single run of Mp"
          and specs.SPEC_C_HARDWARE_QUOTE == "unlike α which is an intrinsic property of the models and the task, the value of c depends on the hardware configuration and software implementation details"
          and specs.SPEC_THM38_VS_THM311_DISAMBIGUATION == "Theorem 3.8 给的是墙钟加速比 (1−α^{γ+1})/((1−α)(γc+1))，Theorem 3.11 给的是算术运算量增加因子 (1−α)(γĉ+γ+1)/(1−α^{γ+1})；两者方向相反、不是同一个量，引用必须写明是哪一条",
          "★★★ **三条假设 + `c` 的口径 + 两条定理的【消歧】逐字钉住** —— ★★ **引用闭式必须连这三条"
          "假设一起引**（`F8`）；★ 且**必须写明是 `Thm 3.8` 还是 `Thm 3.11`**（**方向相反、不是同一个量**，`F10`）",
          "**假设或消歧的引文被动过** —— ★ 先查 `specs.py` 第 1 节")

    # ── D.7 ★★★ 逐元素钉住（程序内常量）────────────────────────────────
    check("D.7", specs.DECODE_FETCH_DAY == "2026-10-09"
          and abs(PINNED_CLOSED_08 - 3.3616) < 1e-9 and abs(PINNED_SPEED_08 - 1.6808) < 1e-9
          and abs(PINNED_MC_08 - 3.362670) < 1e-9
          and abs(PINNED_DESC_EXACT - 3.2455) < 1e-9
          and abs(PINNED_DESC_MEAN - 0.7) < 1e-12
          and abs(PINNED_DESC_GEO - 0.6836957187) < 1e-9
          and abs(PINNED_DESC_CLOSED_MEAN - 2.7731) < 1e-9
          and abs(PINNED_DESC_CLOSED_GEO - 2.6892221687) < 1e-9
          and abs(PINNED_ASC_EXACT - 2.2779444444) < 1e-9
          and abs(PINNED_CONST_EXACT - 2.7731) < 1e-9
          and abs(PINNED_BIAS_DESC_PCT - (-14.5555)) < 1e-9
          and abs(PINNED_BIAS_ASC_PCT - 21.7369) < 1e-9
          and abs(PINNED_BIAS_DESC_GEO_PCT - (-17.1400)) < 1e-9
          and abs(PINNED_SPEED_C1 - 0.67232) < 1e-9 and abs(PINNED_SPEED_C2 - 0.3735111111) < 1e-9
          and abs(PINNED_SPEED_SMALL_ALPHA - 0.555550) < 1e-9
          and abs(PINNED_ALPHA_STAR - 0.518790063676) < 1e-9
          and PINNED_GAMMA_STAR == 3 and abs(PINNED_SPEED_GAMMA_STAR - 1.686857) < 1e-9
          and abs(PINNED_SPEED_GAMMA16 - 0.977482) < 1e-9
          and PINNED_GAMMA_GRID == (1, 2, 3, 4, 6, 8, 16)
          and PINNED_SPEED_GAMMA == (1.4400, 1.6267, 1.6869, 1.6808, 1.5806, 1.4430, 0.9775)
          and abs(PINNED_TOKENS_LIMIT - 5.0) < 1e-9
          and abs(NUM_EPS - 1e-12) < 1e-24 and abs(MC_SIGMA - 4.0) < 1e-12,
          "★★★ **程序内常量逐元素**钉住（三条路径的六个数、四个偏差百分数、三个负收益守卫、"
          " `α*` / `γ*` / `γ=16` / `γ→∞` 上限、两个容差）",
          "**某个程序内常量被动过**")

    # ── D.8 ★★★ 独立性（源码级）：三条路径两两互不引用 ─────────────────
    an = _names_in(analytic)
    ct = _names_in(closed_tokens)
    mc = _names_in(monte_carlo)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：三条路径互不引用 ──")
    print("     `analytic` 里出现 `closed_tokens`/`monte_carlo` = %s/%s"
          % ("closed_tokens" in an, "monte_carlo" in an))
    print("     `closed_tokens` 里出现 `analytic`/`monte_carlo` = %s/%s"
          % ("analytic" in ct, "monte_carlo" in ct))
    print("     `monte_carlo` 里出现 `analytic`/`closed_tokens` = %s/%s"
          % ("analytic" in mc, "closed_tokens" in mc))
    print("     ★★★ **降级措辞**：这只是「**三段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", "closed_tokens" not in an and "monte_carlo" not in an
          and "analytic" not in ct and "monte_carlo" not in ct
          and "analytic" not in mc and "closed_tokens" not in mc,
          "★★ **源码级线索成立**：**闭式 / 解析 / MC 三条路径两两互不引用**（★ **三侧都审**）——"
          " ★★ 于是「三者一致」**不是恒真**，而是**三条独立路径的会合**",
          "**三条路径互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 术语纪律 + 限定语（★ 机械判定）──────────────────────────
    src = open(__file__, encoding="utf-8").read()
    # ★★ 被禁裸词（**刻意用拼接**而不是字面量）：★ 若这里写成字面量，**计数代码自己**也会被算进去，
    #   于是「计数 == 豁免覆盖数」这条**等式**永远差 1。★ 拼接后本文件里**不再出现该裸词的字面量**。
    WORD = "de" + "code"
    n_bare = src.count(WORD)                # ★ 源码里**裸写**该词的计数
    n_exempt = sum(src.count(s) * s.count(WORD) for s in BARE_DECODE_EXEMPT)
    print()
    print("  ── D.9 ★★ **术语纪律**（`§2.2`）：一律写「%s」/「%s」全称；限定语逐字出现 ──"
          % (TERM_AUTOREGRESSIVE, TERM_KV_CACHE))
    print("     ★★ 源码里**裸写那个被禁裸词** = **%d** 处；豁免清单覆盖 = **%d**（清单 %d 项，逐项理由见模块头）"
          % (n_bare, n_exempt, len(BARE_DECODE_EXEMPT)))
    check("D.9", TERM_AUTOREGRESSIVE == "自回归解码" and TERM_KV_CACHE == "KV 缓存"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_AUTOREGRESSIVE in src
          and n_bare == n_exempt
          and all(s in src for s in BARE_DECODE_EXEMPT),
          "★★ **判据成立**：两条术语常量 `%r` / `%r`（★ **不裸写** `decode` / `cache` —— "
          "它们在本库都另有含义）；★ 限定语 `%r` **逐字出现在本程序源码里**；"
          " ★★ **被禁裸词的裸写计数 `%d` 恰好等于豁免清单覆盖的 `%d`**"
          "（★ 清单之外的任何一处裸写都会让它不等）"
          % (TERM_AUTOREGRESSIVE, TERM_KV_CACHE, LIMIT_QUOTE, n_bare, n_exempt),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写** —— "
          "★ 本域一律写全称；★ 新增的裸写要么改全称、要么进 `BARE_DECODE_EXEMPT` 并写明理由")


# ===========================================================================
# 7. 汇总 / 主程序
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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：同一量走**三条独立路径**（闭式/解析/MC）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：外部事实是 **`[C]` 级论文**（arXiv:2211.17192）；算例参数是 `[E]` 级。")
    print("★ 本程序只核对「**解码算法的代数**」——**不是**任何真实模型的加速比（`c` 本机给不出）。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_speculative_decoding.py —— 投机解码：三条独立路径 + 三条假设的边界")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 载体：arXiv:2211.17192（`[C]` 级，同行评审：是）。")
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何目标 LLM。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
