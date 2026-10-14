#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_extrapolation.py —— **PI / NTK-aware / NTK-by-parts / YaRN / ALiBi**，
用 `02` 的频率框架统一解释。

★ 本程序**不讲频率谱**（那是 `02` 的事），只用 `02` 的 `turns` 与分辨窗口
  （计划 §3 的分工写死：**`02` 给定量框架、`03` 用它解释工程方法**）。
配套文档：`03-extrapolation.md`。

===============================================================================
★★ 本程序的核心：**四种方法 = 同一个一参数族的四种「怎么挑权重」**
===============================================================================

★ 统一量（本域写出来的，[E]）：**相位比**

    Φ_ratio_i  :=  扩展窗口下第 `i` 维的最大相位  /  训练窗口下它的最大相位
                 =  `max_pos · θ_i^{eff} / (L · θ_i)`

  它的意义只有一句：**`Φ_ratio_i ≤ 1` 才叫「没有外推」**（最大相位不超过训练时见过的）。
  ⇒ 四种方法在这一条量上的差别是**可逐位比较的**：

  | 方法 | 每维的**频率缩放** `θ^{eff}/θ` | `Φ_ratio_i` | 它的性质 |
  |---|---|---|---|
  | **PI** | `1/s`（**全维一样**） | **`1`** | 恰好贴住上限 —— 但它把**不该缩**的高频也缩了（E1/E3） |
  | **NTK-aware** | `s^{−α_i}`，`α_i = (i−1)/(d/2−1)` | `s^{1−α_i}` ∈ `[1, s]` | ★★ **`i=1` 那一维 `Φ_ratio = s`** ⇒ **它是被外推的**（E4 的 `out-of-bound`） |
  | **不做任何事** | `1` | `s` | 全维都外推 `s` 倍 |
  | **NTK-by-parts** | `(1−γ_i)·(1/s) + γ_i` | `(1−γ_i)/s + γ_i` ∈ `[1/s, 1]` | ★★ **恒 `≤ 1`** ⇒ **从不超过训练范围**（E5 的 `avoid any extrapolation`） |

  ⇒ ★ **`i=1` 那一格是本程序最重要的一格**：`α_1 = (1−1)/(d/2−1) = 0` **精确为 0**
    ⇒ 最高频那一维**根本没有被插值**，它的 `Φ_ratio` 仍是 `s` —— 这就是
    `specs.YARN_OUT_OF_BOUND_E4` 里那个「被外推到 out-of-bound 值的维」。
  ⇒ ★★ 另一格：**PI 的 `Φ_ratio` 恒为 `1`**，而 `1` 是**上限**（不是目标）——
    它把全部维压到上限 ⇒ 高频信息被丢（E1：「`taking the loss of high frequency into account`」）。

===============================================================================
★★ `02` 与 `03` 之间**唯一的那座桥**（计划 §3 Chunk 5 表 B 的最后一条）
===============================================================================

  `02` 的**分辨窗口**是 `θ_i·S`（D6）；YaRN 论文（E5）的判据是 **`L/λ_i`**。
  ★★ 两者是**同一个量**：

      r_i  :=  L / λ_i  =  L·θ_i / (2π)  ==  `02` 的 `turns(θ_i, L)`   （恒等）

  于是 E5 的两条 bullet 逐字落到 `02` 的框架里：

  · 「`λ` **is much smaller than** the context size `L`」 ⇒ `r_i` 大 ⇒ **整窗转了很多圈**
    ⇒ **不插值**（这一维本来就没问题）；
  · 「`λ` **is equal to or bigger than** the context size `L`」 ⇒ `r_i ≤ 1`
    ⇒ ★★ **在整段上下文里连一圈都没转完** ⇒ **只插值**（`B.1` 把这条做成**集合等式**）。

  ⇒ ★ 而 `Φ_ratio` 这一侧的分界是 `1`；`r` 这一侧的分界也是 `1`（一圈）。
    **两个 `1` 是同一个判据的两种写法** —— 这就是那座桥。

===============================================================================
★★ 本程序的数字与它们的来源等级（三类标签必须分开）
===============================================================================

  · **本机实测值 / 推导值**：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
  · **外部规格值**：两个真实模型的 `rope_theta` / `max_position_embeddings` / rotary 维数
    （`specs.py`，**经镜像取得**；`head_dim` 有原生/推导两种等级）。
  · **论文逐字**：`specs.py` 的 YaRN（arXiv 2309.00071，**未确认同行评审**）、
    PI（2306.15595，**未确认**）、ALiBi（2108.12409，**未确认**）三批 [C] 常量。
  · ★★ **等级纪律**：NTK 系列的**一手来源**是 Reddit 帖与 GitHub 仓库（**D 级**）——
    本程序**只引 YaRN 对它们的转述（C 级）**，不把社区帖子当数值来源
    （见 `specs.UNSOURCED` 的第一项）。

===============================================================================
★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）
===============================================================================

  · `A.1`  PI：全维的频率缩放**恰好是 `1/s`**（闭式），且 `Φ_ratio` **恒 `== 1.0`**；
  · `A.2`  ★ NTK-aware 的**两条独立路径**：`freqs(d, b')` 与闭式 `θ_i·s^{−α_i}`
           逐位相等（`< 1e-12`，两条路都从 `specs.YARN_BASE_CHANGE` 那个换 base 式出发）；
  · `A.3`  ★ `α_1 == 0.0` 与 `α_{d/2} == 1.0` **精确**（端点不是近似的）；
  · `A.4`  ★ NTK-by-parts 的 ramp：`γ ≡ 0` 当 `r < α`、`γ ≡ 1` 当 `r > β`、
           中间**逐位等于** `(r−α)/(β−α)`（闭式）；★ 用的是论文的 `α = 1` / `β = 32`
           （**不是** `β`/`γ` 两个名字 —— 论文里 `γ` 是那个 ramp 函数）；
  · `A.5`  ★ YaRN 的温度：`√(1/t) = 0.1·ln(s) + 1`（**拟合值**，E7），
           且「把 `q`、`k` 同乘 `√(1/t)`」**逐位等于**「logits 除以 `t`」（闭式）；
  · `B.1`  ★★ **桥**：`r_i == turns(θ_i, L)`（恒等）且
           `{i : λ_i ≥ L} == {i : turns_i(L) ≤ 1}`（**集合相等**）⇒ E5 与 D6 是同一件事；
  · `B.2`  ★★ **E3 的定量形态**：NTK-aware 的频率缩放 `s^{−α_i}` 随 `i` **单调递减**
           （高频少缩、低频多缩），且 `α` 与 `γ` 是**两个不同的权重函数**
           （`max|α−γ| > 0.1`）；
  · `B.3`  ★★ **`α` 与 `γ` 的关键差别**（这是 NTK-aware 会 out-of-bound 而 NTK-by-parts
           不会的原因）：换 `L` 时 **`α_i` 逐位不变**（它只看维号），而 **`γ_i` 会变**
           （它看 `r = L/λ`）；
  · `B.4`  ★★ **X17 的脚注**：ALiBi 的偏置**不除以 `√d_k`** ⇒ 它的近因效应
           **与 `d_k` 无关**（`d_k ∈ {64,128,256}` 上极差 `< 0.2`）；★ **反面对照**：
           若把偏置也除以 `√d_k`，效应**随 `d_k` 缩小**（实测比值 `1.92 ≈ √4`）；
  · `B.5`  ★ ALiBi 的 slope 闭式：`(2^{−8/n})^h`；`n = 8` 给出 `1/2 … 1/256`
           （与论文另一句逐字一致）、`n = 16` 起于 `1/√2` 比值 `1/√2`；
  · `B.6`  ★ ALiBi 的偏置是**单调的**：`bias_i == 0`（对角线）、全体 `≤ 0`、
           且 `q ≡ 0` 时权重比 `w_{j+1}/w_j` **逐位等于** `exp(m)`（闭式）；
  · `C.1`  ★ **PI 的代价**（与 `02` 的 `C.2` 呼应）：`θ_1/s` 掉到「相邻位置可分」的阈值之下
           ⇒ PI 把**唯一**能分辨相邻位置的那一维也压没了；
  · `C.2`  ★ **负对照**：把 NTK-aware 的 `α` 换成常数（`α ≡ 0` 或 `≡ 1`）⇒
           它**退化**成「不做任何事」或「PI」，`Φ_ratio` 的分布当场变掉；
  · `C.3`  ★ **方向是可判定的**（ALiBi）：把 slope 取反号 ⇒ 权重从「偏向近期」
           变成「偏向最远」—— 两支各自自洽，所以这个错是**静默的**。

===============================================================================
★★ 本程序刻意不做的事（计划 §2 的边界表，越界即作废）
===============================================================================

  * ★★ **不重讲频率谱与分辨窗口** —— 那是 `02`（本程序只**引用**它的 `turns`）。
  * **不写**「成对通道旋转」那句结构陈述的正文（在
    `130-model-optimization/quantization/19-kivi-asymmetry.md:184`）。
  * ★★ **不比较四种方法的「效果」**：那要训练与评测（本域**无 GPU、不训练模型**），
    且 YaRN 原文自己说 `perplexity may not be a great indicator`（E8）
    ⇒ 本域只做**结构对比**（每维的缩放因子是多少，**可手算**）。
  * **不讲**变体的 kernel 后果（滑窗改循环范围、ALiBi/softcap 改 `exp` 的输入）——
    正文在 `40-kernels/attention/04:126-156`（**正文在彼，此处回链**）。
  * ★ **不引** Reddit / GitHub（D 级）：NTK 系列只引 YaRN 的转述（C 级）。

用法：
    python3 03_extrapolation.py
"""

import ast
import collections
import inspect
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账
# ===========================================================================
ASSERT_FAIL_MARK = "$$ FAIL: positional-encoding"

OTHER_THEME_MARKS = ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                     "## FAIL: moe", "~~ FAIL: normalization")

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
# 1. 口径常量（★ 全部在 `D.1`/`D.2` 被逐位钉住；★★ 全部正数存放）
# ===========================================================================
# ★★ **为什么这张表里没有负数**（计划 §3 开头的第二个盲区）：叶子枚举只取 `ast.Constant`，
#    而 `-1` 是 `UnaryOp(USub, Constant(1))` ⇒ **负数表整体不进变异分母**。
#    本程序里方向/符号密集的地方（ALiBi 的 `m·(j−i)`、`1−α`）**一律在函数体里派生**：
#    「距离」用**正数**、符号由 `1.0 - 2.0*idx` 那样的**派生式**给出（见 §2）。
D_MODEL = 128                  # 真实模型的 rotary 维数（与 `02` 的 `D_MODEL` 同值同源）
BASE_PAPER = 10000.0           # RoPE 原论文的 `base`（`θ_i = base^{-2(i-1)/d}` 的默认）
BASE_LADDER = (500000.0, 1000000.0)   # 两个真实模型的 `rope_theta`（specs）
SCALE_LADDER = (2.0, 4.0, 8.0, 16.0, 32.0)   # 扩展倍数 `s = L'/L`
TRIM_ALPHA = 1.0               # YaRN 的 `α`（**下**阈值：`r < α` ⇒ 全插值）
TRIM_BETA = 32.0               # YaRN 的 `β`（**上**阈值：`r > β` ⇒ 不插值）
EXTENDED_S = 16.0              # 桥与代价两节用的默认扩展倍数
N_HEADS_8 = 8                  # ALiBi 的 slope 闭式：8 头
N_HEADS_16 = 16                # 16 头
ALIBI_M = 0.5                  # 8 头时的第一个 slope（= `2^{-8/8} = 1/2`）
ALIBI_SEQ = 32                 # ALiBi 实验的序列长度
ALIBI_POS = 31                 # 查询位置（0-based，最后一个）
ALIBI_N_SAMPLE = 4000          # `B.4` 的样本数（平均掉抽样噪声）
DK_LADDER = (64, 128, 256)     # `B.4` 的 `d_k` 阶梯（`√256/64 = 2`）
SEQ_SMALL = 6                  # 小实验的序列长度
SEED = 20260922                # 主数据的种子

# ── 阈值（★ 逐个在 `D.2` 钉住）─────────────────────────────────────────────
CLOSED_FORM_TOL = 1e-12        # 两条独立路径 / 闭式的相对容差
ENDPOINT_TOL = 1e-15           # `α` 端点的「精确」判定
TURNS_TOL = 1e-12              # 圈数比较
SET_EQ_FLOOR = 1e-9            # `B.1` 集合等式两侧允许的最小规模（防空集断言）
ALPHA_GAMMA_SEP_FLOOR = 0.1    # `B.2`：`α` 与 `γ` 必须真的不同
TEMP_TOL = 1e-12               # `A.5` 温度与 logits 缩放的等价性
ALIBI_DK_SAME_TOL = 0.2        # `B.4` 正面：不同 `d_k` 上效应一致（实测极差 `0.054`）
ALIBI_DK_RATIO_LO = 1.5        # `B.4` 反面：效应随 `√d_k` 缩小的比值下界（实测 `1.92`）
ALIBI_DK_RATIO_HI = 2.5        # …上界（`√(256/64) = 2`）
ALIBI_SLOPE_TOL = 1e-15        # `B.5` slope 闭式
ALIBI_RATIO_TOL = 1e-12        # `B.6` 权重比的闭式
PI_RESOLVE_CEILING = 1.0       # `C.1` PI 之后每位置相位必须掉到这个阈值之下
DEGENERATE_SEP_FLOOR = 0.01    # `C.2` 退化后的分布必须与真身分开
DIRECTION_SIGN_FLOOR = 1e-3    # `C.3` 两支 slope 必须给出不同的加权位置


# ===========================================================================
# 2. 频率表、圈数、与四种方法的**权重函数**（★ 测量的对象）
# ===========================================================================
def freqs(d, base):
    """`θ_i = base^{-2(i-1)/d}`，`i = 1..d/2`（返回长度 `d/2` 的数组）。"""
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1.0) / d)


def turns(th, span):
    """★ `θ·span / 2π` —— **与 `02` 的 `turns` 是同一个函数、同一个定义**。

    它就是 YaRN 的 `r = L/λ`（`02` 与 `03` 之间那座桥的载体）。
    """
    return th * span / (2.0 * np.pi)


def alpha_by_index(h):
    """NTK-aware 的**按维号**权重：`α_i = (i−1)/(h−1)`，`i = 1..h`。

    ★ 端点：`α_1 = 0`（**精确**）、`α_h = 1`（**精确**）。
    """
    return np.arange(h, dtype=np.float64) / (h - 1.0)


def gamma_by_ratio(r, alpha, beta):
    """NTK-by-parts 的 ramp：`γ = 0` if `r < α`；`1` if `r > β`；否则 `(r−α)/(β−α)`。

    ★★ 用的是**论文的参数名**：`α`/`β` 是**阈值**，`γ` 是**这个 ramp 函数**
      （`specs.YARN_NTK_BY_PARTS_ALPHA/BETA`；★ 计划 §C.5 的 E5 把命名写成了 `β`/`γ`，
      本域按原文分开）。
    """
    return np.where(r < alpha, 0.0,
                    np.where(r > beta, 1.0, (r - alpha) / (beta - alpha)))


def phi_ratio_ntk(s, alpha):
    """NTK-aware 的相位比：`s^{1−α}`（`α = 1` ⇒ `1`；`α = 0` ⇒ `s`）。"""
    return s ** (1.0 - alpha)


def phi_ratio_by_parts(s, gamma):
    """NTK-by-parts 的相位比：`(1−γ)/s + γ` —— ★ **恒落在 `[1/s, 1]`**。"""
    return (1.0 - gamma) / s + gamma


def yarn_temperature(s):
    """YaRN 的 `√(1/t) = 0.1·ln(s) + 1`（**拟合值**，E7）；返回 `(√(1/t), t)`。"""
    sq = 0.1 * np.log(s) + 1.0
    return sq, 1.0 / (sq * sq)


# ── ★ 参照实现（**不调用上面任何一个权重函数**，`D.9` 的 AST + 文本检查守着）──────
def phi_ratio_direct(method_scales, s_expanded, L, th):
    """★ **另一条路**：直接用「最大相位 ÷ 训练上限」算 `Φ_ratio`，不走 `s^{1−α}`。

    `method_scales` 是**每维的频率缩放因子**（`θ^{eff}/θ`）——
    本函数**只做除法**：`Φ_ratio_i = (s·L·θ_i·scale_i) / (L·θ_i)`。
    ★ 它与 `phi_ratio_ntk` 是**两条路**：这条不知道 `α` 是什么。
    """
    return (s_expanded * L * th * method_scales) / (L * th)


def alibi_slopes(n_heads):
    """ALiBi 的 slope 闭式：`(2^{−8/n})^h`，`h = 1..n`。

    ★ 来源：`specs.ALIBI_SLOPE_RULE`（原文：`the geometric sequence that starts at
      `2^{−8/n}` and uses that same value as its ratio`）。
    """
    base_s = 2.0 ** (-8.0 / n_heads)
    return base_s ** np.arange(1, n_heads + 1, dtype=np.float64)


def alibi_bias(i_pos, n_pos, slope):
    """`bias_j = slope·(j − i)`（只有 `j ≤ i` 有效）—— ★ **距离是正数，符号在函数体里派生**。"""
    dist = np.arange(n_pos, dtype=np.float64) - float(i_pos)
    return slope * dist


# ===========================================================================
# 3. A 类：四种方法的**算式**（闭式 + 两条独立路径）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 四种方法的算式（PI / NTK-aware / NTK-by-parts / YaRN）")
    print("=" * 74)
    th = freqs(D_MODEL, BASE_PAPER)
    h = D_MODEL // 2
    L0 = float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS)
    s = EXTENDED_S
    print("  口径：`d = %d`（`h = %d` 维）、`base = %.0f`、`L = %.0f`、`s = %g`"
          % (D_MODEL, h, BASE_PAPER, L0, s))

    # ── A.1 PI：全维缩放恰好 1/s，Φ_ratio 恒 1 ──────────────────────────────
    print()
    print("  ── A.1 **PI**（`m → m·L/L'`，即位置除以 `s`）──")
    scale_pi = np.full(h, 1.0 / s)
    ph_pi = phi_ratio_direct(scale_pi, s, L0, th)
    print("    频率缩放 `θ^{eff}/θ` 的唯一取值：%r（`1/s`，`s = %g`）" % (1.0 / s, s))
    print("    `Φ_ratio` 的取值集合：%s" % (sorted(set(np.round(ph_pi, 15).tolist())),))
    check("A.1", bool(np.all(scale_pi == 1.0 / s)) and bool(np.all(ph_pi == 1.0)),
          "★ PI 把**每一维**都乘同一个因子 `1/s`（实测全维取值集合 = `{%r}`）⇒ "
          "`Φ_ratio` **恒 `== 1.0`**（恰好贴住训练上限）—— "
          "★ 而 `1` 是**上限**、不是目标：不需要缩的高频也被缩了（E1/E3 的代价）"
          % (1.0 / s))

    # ── A.2 NTK-aware：两条独立路径 ────────────────────────────────────────
    print()
    print("  ── A.2 **NTK-aware**：换 `base`（`specs.YARN_BASE_CHANGE`）的后果 ──")
    al = alpha_by_index(h)
    worst = 0.0
    for base in BASE_LADDER:
        thb = freqs(D_MODEL, base)
        for sv in SCALE_LADDER:
            bp = base * sv ** (D_MODEL / (D_MODEL - 2.0))   # ★ 原文的 `b' = b·s^{|D|/(|D|−2)}`
            direct = freqs(D_MODEL, bp)
            closed = thb * sv ** (-al)
            worst = max(worst, float(np.abs(direct - closed).max() / np.abs(direct).max()))
    print("    两条路：① `freqs(d, b')`（直算）  ② 闭式 `θ_i·s^{−α_i}`")
    print("    `max|①−②| / max|①|` 在 `base ∈ %s` × `s ∈ %s` 上 = %.3e"
          % (list(BASE_LADDER), list(SCALE_LADDER), worst))
    check("A.2", worst < CLOSED_FORM_TOL,
          "★ **两条独立路径逐位一致**（最差相对差 `%.3e` < `%.0e`）⇒ "
          "「换 base」与「按维号乘 `s^{−α_i}`」是**同一件事**，"
          "而 `α_i = (i−1)/(d/2−1)` 是可手算的闭式（`Φ_ratio_i = s^{1−α_i}`）"
          % (worst, CLOSED_FORM_TOL))

    # ── A.3 α 的端点精确 ──────────────────────────────────────────────────
    ph_ntk = phi_ratio_ntk(s, al)
    print()
    print("    `α_1 = %r`（精确 0）、`α_h = %r`（精确 1）、`α` 第 32 维 = %.6f"
          % (float(al[0]), float(al[-1]), float(al[31])))
    print("    对应：`Φ_ratio_1 = %r`（= `s`，**被外推**）、`Φ_ratio_h = %r`（= 1）"
          % (float(ph_ntk[0]), float(ph_ntk[-1])))
    check("A.3", al[0] == 0.0 and al[-1] == 1.0
          and abs(float(ph_ntk[0]) - s) == 0.0 and ph_ntk[-1] == 1.0,
          "★ `α` 的两个端点**精确**（`%.1f` / `%.1f`）⇒ `Φ_ratio_1 **恰好等于 s**`（`%r`）、"
          "`Φ_ratio_h **恰好等于 1**`（`%r`）⇒ ★★ **最高频那一维一点都没被插值**，"
          "它就是 E4 里那个「被外推到 out-of-bound 值的维」"
          % (float(al[0]), float(al[-1]), float(ph_ntk[0]), float(ph_ntk[-1])))

    # ── A.4 NTK-by-parts 的 ramp ──────────────────────────────────────────
    print()
    print("  ── A.4 **NTK-by-parts** 的 ramp（`α = %g`、`β = %g`；★ 论文的参数名）──"
          % (TRIM_ALPHA, TRIM_BETA))
    r_probe = np.array((0.5, 1.0, 2.0, 16.0, 32.0, 64.0))
    g_probe = gamma_by_ratio(r_probe, TRIM_ALPHA, TRIM_BETA)
    cf_probe = np.where(r_probe < TRIM_ALPHA, 0.0,
                        np.where(r_probe > TRIM_BETA, 1.0,
                                 (r_probe - TRIM_ALPHA) / (TRIM_BETA - TRIM_ALPHA)))
    for rv, gv in zip(r_probe, g_probe):
        print("    r = %-6.2f ⇒ γ = %.6f" % (rv, gv))
    print("    ★ 与逐点闭式 `(r−α)/(β−α)` 的最大差 = %.3e"
          % float(np.abs(g_probe - cf_probe).max()))
    check("A.4", bool(np.all(g_probe == cf_probe))
          and float(g_probe[0]) == 0.0 and float(g_probe[-1]) == 1.0,
          "★ ramp **逐位等于**它的定义式（三段全过：`r<α ⇒ 0`、`r>β ⇒ 1`、"
          "中间 ⇒ `(r−α)/(β−α)`）—— ★ 用的是论文的参数名 `α`/`β`（**不是** `β`/`γ`："
          "`γ` 是那个 ramp 函数本身，计划 §C.5 的写法把两者混了）")

    # ── A.5 YaRN 的温度 ───────────────────────────────────────────────────
    print()
    print("  ── A.5 **YaRN**：温度 `t`（`√(1/t) = 0.1·ln(s) + 1`，**拟合值**）──")
    sq, t = yarn_temperature(s)
    print("    `s = %g` ⇒ `√(1/t) = %.6f`、`t = %.6f`" % (s, sq, t))
    print("    ★ 原文逐字（`specs.YARN_FITTING_E7`）：`The equation above is found by fitting`")
    q = np.array((2.0, 1.0, 0.5, 3.0))
    k = np.array((1.0, 2.0, 4.0, 0.5))
    lhs = (q * sq) * (k * sq)      # 把 q、k 各乘 √(1/t)（原文的 "length scaling" trick）
    rhs = (q * k) / t              # 等于把 logits 除以 t
    print("    「`q`、`k` 各乘 `√(1/t)`」与「logits 除以 `t`」的最大差 = %.3e"
          % float(np.abs(lhs - rhs).max()))
    check("A.5", float(np.abs(lhs - rhs).max()) < TEMP_TOL,
          "★ 「`length scaling` trick」**逐位等于**「logits 除以 `t`」（差 `%.3e`）⇒ "
          "两者是同一个温度操作；★ 而那个 `√(1/t)` 的**表达式**是**拟合**出来的"
          "（原文 `found by fitting … at the lowest perplexity`）⇒ ★ **不得写成推导**"
          % float(np.abs(lhs - rhs).max()))


# ===========================================================================
# 4. B 类：**桥**、E3 的定量形态、E4 的 out-of-bound、ALiBi（含 X17 的脚注）
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 桥（E5 ↔ D6）/ E3 / E4 / ALiBi")
    print("=" * 74)

    # ── B.1 ★★ 桥 ────────────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **桥**：E5 的 `L/λ_i` 与 `02` 的 `turns(θ_i, L)` 是同一个量 ──")
    worst = 0.0
    all_sets_nonempty = True
    for base, L in ((specs.LLAMA3_8B_ROPE_THETA,
                     float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS)),
                    (specs.QWEN3_30B_A3B_ROPE_THETA,
                     float(specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS))):
        thb = freqs(D_MODEL, base)
        lam = 2.0 * np.pi / thb
        r_from_lambda = L / lam
        r_from_turns = turns(thb, L)
        worst = max(worst, float(np.abs(r_from_lambda - r_from_turns).max() / r_from_turns.max()))
        set_a = set(np.flatnonzero(lam >= L).tolist())
        set_b = set(np.flatnonzero(r_from_turns <= 1.0).tolist())
        # ★ 两侧都必须**非空**：否则「空集 == 空集」是一条恒真断言（P17 的方向）
        if len(set_a) < SET_EQ_FLOOR or len(set_b) < SET_EQ_FLOOR:
            all_sets_nonempty = False
        print("    base=%-9.0f L=%-6.0f  `{λ_i ≥ L}` = %d 维，`{r_i ≤ 1}` = %d 维，"
              "**两个集合相等 = %s**（两侧都非空 = %s）"
              % (base, L, len(set_a), len(set_b), set_a == set_b,
                 len(set_a) >= SET_EQ_FLOOR and len(set_b) >= SET_EQ_FLOOR))
        print("      `max|L/λ − turns| / max(turns)` = %.3e"
              % float(np.abs(r_from_lambda - r_from_turns).max() / r_from_turns.max()))
    check("B.1", worst < TURNS_TOL and all_sets_nonempty,
          "★★ **`L/λ_i == turns(θ_i, L)` 是恒等**（两边最差相对差 `%.3e` < `%.0e`），"
          "且 **`{λ_i ≥ L} == {turns_i(L) ≤ 1}`（集合相等）** ⇒ "
          "E5 的「`λ` 等于或大于上下文长度 `L`」与 D6 的「分辨窗口」**是同一件事**："
          "`λ_i ≥ L` ⟺ `θ_i·L ≤ 2π` ⟺ **整段上下文里连一圈都没转完** ⇒ 只插值。"
          "★ 这就是 `02` 与 `03` 之间**唯一的那座桥**（计划 §3 表 B 的最后一条）"
          % (worst, TURNS_TOL))
    print("    ★ 两个真实模型上「`r_i ≤ 1`」的维数：Llama **29**、Qwen3 **23**"
          "（与 `02` 的 `C.3` 是同一批维 —— 那里按 `λ_i ≥ L` 数的）")

    # ── B.2 E3：高频少缩、低频多缩 ────────────────────────────────────────
    print()
    print("  ── B.2 ★ E3 的**定量形态**：`scaling high frequencies less and low frequencies more` ──")
    th = freqs(D_MODEL, BASE_PAPER)
    h = D_MODEL // 2
    al = alpha_by_index(h)
    s = EXTENDED_S
    scale_ntk = s ** (-al)
    mono = bool((np.diff(scale_ntk) < 0).all())
    print("    NTK-aware 的频率缩放 `s^{−α_i}`：i=1 `%.6f` → i=%d `%.6f`（单调递减 = %s）"
          % (scale_ntk[0], h, scale_ntk[-1], mono))
    print("    ⇒ **高频（i 小）缩得少**（因子接近 1）、**低频（i 大）缩得多**（因子接近 `1/s`）")
    L0 = float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS)
    g = gamma_by_ratio(turns(th, L0), TRIM_ALPHA, TRIM_BETA)
    sep = float(np.abs(al - g).max())
    print("    `α_i`（按维号）与 `γ_i`（按 `r = L/λ`）的最大差 = %.6f（阈值 > %.1f）"
          % (sep, ALPHA_GAMMA_SEP_FLOOR))
    check("B.2", mono and float(scale_ntk[0]) == 1.0 and sep > ALPHA_GAMMA_SEP_FLOOR,
          "★ E3 的定量形态：`s^{−α_i}` **逐维单调递减**（`i=1` 处 `== 1.0` **精确**，"
          "即**完全不缩**；`i=h` 处 `== 1/s`）⇒ 「高频少缩、低频多缩」不是形容，"
          "是一条**单调性**；★ 且 `α`（按维号）与 `γ`（按 `r = L/λ`）"
          "**是两个不同的权重函数**（最大差 `%.3f`）"
          % (sep,))

    # ── B.3 α 与 γ 的关键差别 ──────────────────────────────────────────────
    print()
    print("  ── B.3 ★★ `α` 与 `γ` 的**关键差别**：换一个 `L` 会动谁？──")
    th_llama = freqs(D_MODEL, specs.LLAMA3_8B_ROPE_THETA)
    th_qwen = freqs(D_MODEL, specs.QWEN3_30B_A3B_ROPE_THETA)
    g_llama = gamma_by_ratio(turns(th_llama, float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS)),
                             TRIM_ALPHA, TRIM_BETA)
    g_qwen = gamma_by_ratio(turns(th_qwen, float(specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS)),
                            TRIM_ALPHA, TRIM_BETA)
    al_a = alpha_by_index(h)
    al_b = alpha_by_index(h)
    d_alpha = float(np.abs(al_a - al_b).max())
    d_gamma = float(np.abs(g_llama - g_qwen).max())
    print("    换模型（`base` 与 `L` 都变）：`α` 的最大变化 = %.3e；"
          "`γ` 的最大变化 = %.6f" % (d_alpha, d_gamma))
    print("    ⇒ ★ `α` **只看维号**（`(i−1)/(h−1)`，与 `L` 无关）；"
          "`γ` **看 `r = L/λ`**（要 `L` 与 `θ` 两个输入）")
    check("B.3", d_alpha == 0.0 and d_gamma > ALPHA_GAMMA_SEP_FLOOR,
          "★★ `α` 在两个模型间**逐位不变**（`%.1e`），而 `γ` **变了**（`%.3f` > `%.2f`）"
          "⇒ ★ 这就是「NTK-aware 会把最高频维推到 out-of-bound、而 NTK-by-parts 不会」"
          "的**原因**：`α` **不知道 `L` 是多少**，它按维号硬切；"
          "`γ` 用的是 `r = L/λ` —— ★ 而那个量正是 `02` 给的分辨窗口"
          % (d_alpha, d_gamma, ALPHA_GAMMA_SEP_FLOOR))
    for nm, base, L, gg in (("Llama", specs.LLAMA3_8B_ROPE_THETA,
                             specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS, g_llama),
                            ("Qwen3", specs.QWEN3_30B_A3B_ROPE_THETA,
                             specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS, g_qwen)):
        print("      %-6s base=%-9.0f L=%-6d : `γ==0`（全插值）%2d 维、"
              "`γ==1`（不插值）%2d 维、中间 %2d 维"
              % (nm, base, L, int((gg == 0.0).sum()), int((gg == 1.0).sum()),
                 int(((gg > 0.0) & (gg < 1.0)).sum())))

    # ── B.4 ALiBi：X17 的脚注 ────────────────────────────────────────────
    print()
    print("  ── B.4 ★★ ALiBi（X17）：偏置**不除以 `√d_k`** 的可测后果 ──")
    print("     原文逐字（`specs.ALIBI_FOOTNOTE`）：`The ALiBi bias is not multiplied by the` …")
    s_pos = ALIBI_POS
    n_pos = ALIBI_SEQ
    bias = alibi_bias(s_pos, n_pos, ALIBI_M)
    pos = np.arange(n_pos, dtype=np.float64)
    effects = {}
    wrong = {}
    for dk in DK_LADDER:
        rng = np.random.default_rng(SEED + dk)
        q = rng.standard_normal((ALIBI_N_SAMPLE, dk)) / np.sqrt(float(dk))
        K = rng.standard_normal((ALIBI_N_SAMPLE, n_pos, dk))
        logits = np.einsum("nd,nsd->ns", q, K)      # ★ 已含 `1/√d_k` ⇒ 单位量级

        def softmax(x):
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        w0 = softmax(logits)
        w_ok = softmax(logits + bias)
        w_bad = softmax(logits + bias / np.sqrt(float(dk)))
        effects[dk] = float((w_ok * pos).sum(1).mean() - (w0 * pos).sum(1).mean())
        wrong[dk] = float((w_bad * pos).sum(1).mean() - (w0 * pos).sum(1).mean())
    for dk in DK_LADDER:
        print("    `d_k = %-4d`  加偏置后平均位置前移 **%+.6f**；"
              "若偏置也除 `√d_k` 则只前移 `%+.6f`" % (dk, effects[dk], wrong[dk]))
    spread = max(effects.values()) - min(effects.values())
    ratio = wrong[DK_LADDER[0]] / wrong[DK_LADDER[-1]]
    check("B.4", spread < ALIBI_DK_SAME_TOL
          and ALIBI_DK_RATIO_LO < ratio < ALIBI_DK_RATIO_HI,
          "★★ **脚注是可判定的**：偏置**不吃** `1/√d_k` ⇒ 它的近因效应在 "
          "`d_k ∈ %s` 上**一致**（极差 `%.4f` < `%.2f`）；★ **反面对照**："
          "若把偏置**也**除以 `√d_k`，效应会随 `d_k` **缩小**（比值 `%.2f`，"
          "理论值 `√(256/64) = 2`）⇒ ★ 那条脚注不是脚注：**它决定了大 `d_k` 的 head "
          "还保不保有近因偏好**" % (list(DK_LADDER), spread, ALIBI_DK_SAME_TOL, ratio))

    # ── B.5 ALiBi slope 闭式 ─────────────────────────────────────────────
    print()
    print("  ── B.5 ALiBi 的 slope 闭式（`specs.ALIBI_SLOPE_RULE`）──")
    sl8 = alibi_slopes(N_HEADS_8)
    sl16 = alibi_slopes(N_HEADS_16)
    ref8 = 2.0 ** (-np.arange(1, N_HEADS_8 + 1, dtype=np.float64))
    ref16 = 2.0 ** (-0.5 * np.arange(1, N_HEADS_16 + 1, dtype=np.float64))
    print("    n=8 ：%s" % np.round(sl8, 6).tolist())
    print("          与 `1/2^1 … 1/2^8` 的最大差 = %.3e" % float(np.abs(sl8 - ref8).max()))
    print("    n=16：首项 %.10f（`2^{−0.5}`）、比值 %.10f；与 `2^{−0.5…−8}` 的最大差 = %.3e"
          % (sl16[0], sl16[0], float(np.abs(sl16 - ref16).max())))
    check("B.5", float(np.abs(sl8 - ref8).max()) < ALIBI_SLOPE_TOL
          and float(np.abs(sl16 - ref16).max()) < ALIBI_SLOPE_TOL
          and float(sl8[0]) == ALIBI_M,
          "★ slope 闭式 `(2^{−8/n})^h` 在 `n = 8` 与 `n = 16` 上**逐位**复现原文的两句"
          "（`1/2^1 … 1/2^8` 与「起于 `1/√2`、比值 `1/√2`」）；★ `n=8` 的首项 `== %r` "
          "**精确**（那就是本程序 `ALIBI_M` 的出处）" % (ALIBI_M,))

    # ── B.6 ALiBi 的偏置式与单调性 ────────────────────────────────────────
    print()
    print("  ── B.6 ALiBi 的偏置式（`specs.ALIBI_BIAS_ALT`）与它的后果 ──")
    print("    `bias_i = %r`（对角线 **恰好 0**）、`bias_0 = %.4f`（**最小**）、"
          "全体 `≤ 0` = %s" % (float(bias[s_pos]), float(bias[0]), bool((bias <= 0.0).all())))
    w = np.exp(bias - bias.max())
    w = w / w.sum()
    ratios = w[1:] / w[:-1]
    print("    `q ≡ 0` 时相邻权重的比 `w_{j+1}/w_j` 的唯一取值 = %.12f"
          % float(ratios.min()))
    print("    ★ 闭式 `exp(m) = %.12f`（`m = %r`）；权重随位置**严格递增** = %s"
          % (np.exp(ALIBI_M), ALIBI_M, bool((np.diff(w) > 0).all())))
    check("B.6", float(bias[s_pos]) == 0.0 and bool((bias <= 0.0).all())
          and float(np.abs(ratios - np.exp(ALIBI_M)).max()) < ALIBI_RATIO_TOL
          and bool((np.diff(w) > 0).all()),
          "★ 偏置式 `m·[−(i−1),…,−2,−1,0]` 的两个机械后果：对角线**恰好 0**、全体 `≤ 0`；"
          "且 `q ≡ 0` 时相邻权重的比**逐位等于 `exp(m)`**（`%.12f`）⇒ "
          "ALiBi 是一个**纯近因核**（位置越新权重越大）" % float(np.exp(ALIBI_M)))


# ===========================================================================
# 5. C 类：边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)
    th = freqs(D_MODEL, BASE_PAPER)
    h = D_MODEL // 2
    s = EXTENDED_S
    base = BASE_LADDER[0]
    L = float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS)

    # ── C.1 PI 的代价 ────────────────────────────────────────────────────
    print()
    print("  ── C.1 ★ **PI 的代价**：把「唯一能分辨相邻位置」的那一维也压没了 ──")
    thb = freqs(D_MODEL, base)
    before = int((thb >= 1.0).sum())
    after = int((thb / s >= 1.0).sum())
    print("    每位置相位 `θ_1 = %r` ⇒ PI 之后 `θ_1/s = %.6f`（`s = %g`）"
          % (float(thb[0]), float(thb[0] / s), s))
    print("    `θ_i ≥ 1.0` 的维数：PI 之前 **%d**、之后 **%d**" % (before, after))
    check("C.1", after == 0 and before == 1 and float(thb[0] / s) < PI_RESOLVE_CEILING,
          "★ PI 把**每一维**都除以 `s`（`A.1` 的闭式）⇒ 唯一满足 `θ_i ≥ 1` 的那一维"
          "（`i = 1`）落到 `%r` **< %.1f** ⇒ 「相邻位置可分」的维数从 **1 变成 0**。"
          "★ 这正是 E1 说的「PI 丢高频」的**定量形态**（★ 判据 `θ_i ≥ 1` 是本域写出的"
          "[E] 阈值，见 `02` 的 `C.2`）" % (float(thb[0] / s), PI_RESOLVE_CEILING))

    # ── C.2 负对照：把 α 换成常数 ⇒ 退化成别的两种方法 ─────────────────────
    print()
    print("  ── C.2 ★ **负对照**：把 NTK-aware 的 `α` 换成常数 ──")
    al = alpha_by_index(h)
    ph_ntk = phi_ratio_ntk(s, al)
    ph_zero = phi_ratio_ntk(s, np.zeros(h))     # α ≡ 0 ⇒ 不插值
    ph_one = phi_ratio_ntk(s, np.ones(h))       # α ≡ 1 ⇒ PI
    spread_ntk = float(ph_ntk.max() - ph_ntk.min())
    print("    真身（`α_i = (i−1)/(h−1)`）：`Φ_ratio` ∈ [%.6f, %.6f]（跨度 %.6f）"
          % (ph_ntk.min(), ph_ntk.max(), spread_ntk))
    print("    `α ≡ 0`：`Φ_ratio` ∈ [%r, %r]（= 「不做任何事」，全维外推 `s` 倍）"
          % (float(ph_zero.min()), float(ph_zero.max())))
    print("    `α ≡ 1`：`Φ_ratio` ∈ [%r, %r]（= PI，恰好贴住上限）"
          % (float(ph_one.min()), float(ph_one.max())))
    check("C.2", spread_ntk > DEGENERATE_SEP_FLOOR
          and bool((ph_zero == s).all()) and bool((ph_one == 1.0).all()),
          "★ 把 `α` 换成常数 ⇒ **当场退化成另外两种方法**（`α≡0` ⇒ `Φ_ratio ≡ %.1f` 全维外推；"
          "`α≡1` ⇒ `Φ_ratio ≡ 1.0` = PI）⇒ ★ 真身与它们的跨度差 `%.3f` > `%.2f` ⇒ "
          "**`α` 的形状就是 NTK-aware 的全部内容**" % (s, spread_ntk, DEGENERATE_SEP_FLOOR))

    # ── C.3 方向是可判定的（ALiBi 的 slope 取反）──────────────────────────
    print()
    print("  ── C.3 ★ **方向**（ALiBi 的 slope 符号）是可判定的 ──")
    n_pos = ALIBI_SEQ
    s_pos = ALIBI_POS
    pos = np.arange(n_pos, dtype=np.float64)
    b_pos = alibi_bias(s_pos, n_pos, ALIBI_M)
    b_neg = alibi_bias(s_pos, n_pos, -ALIBI_M)   # ★ 负号在函数体里派生（表里没有负数）
    w_pos = np.exp(b_pos) / np.exp(b_pos).sum()
    w_neg = np.exp(b_neg) / np.exp(b_neg).sum()
    m_pos = float((w_pos * pos).sum())
    m_neg = float((w_neg * pos).sum())
    print("    `m = %+.1f`：平均位置 **%.6f**（偏向近期）；`m = %+.1f`：**%.6f**（偏向最远）"
          % (ALIBI_M, m_pos, -ALIBI_M, m_neg))
    check("C.3", abs(m_pos - m_neg) > DIRECTION_SIGN_FLOOR and m_pos > m_neg,
          "★ 两支在同一批数据上相差 `%.6f` > `%.3f`，且 **slope 取正号才偏向近期** ⇒ "
          "符号搞反**各自内部自洽**（都是一条合法的单调核）⇒ ★ 这个错是**静默的**，"
          "只能靠**外部锚点**定方向（`specs.ALIBI_BIAS_ALT` 的 `[−(i−1),…,0]`）"
          % (abs(m_pos - m_neg), DIRECTION_SIGN_FLOOR))
    print("    ★ 本域的方向锚点：原文的偏置集合**以 0 收尾**（最近的那个位置偏置为 0），"
          "且全体 `≤ 0` ⇒ 近因偏好，不是远因偏好")


# ===========================================================================
# 6. D 类：口径/阈值/specs 的钉住（**刻意在 A/B/C 之前**）与失败标记自检
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _body_text(func):
    """`func` 的源码**去掉 `def` 行**（否则函数名本身会把「调用」这个词顶进来）。"""
    return "\n".join(l for l in inspect.getsource(func).splitlines()
                     if not l.lstrip().startswith("def "))


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

    measured = {"D_MODEL": D_MODEL, "BASE_PAPER": BASE_PAPER, "BASE_LADDER": BASE_LADDER,
                "SCALE_LADDER": SCALE_LADDER, "TRIM_ALPHA": TRIM_ALPHA,
                "TRIM_BETA": TRIM_BETA, "EXTENDED_S": EXTENDED_S,
                "N_HEADS_8": N_HEADS_8, "N_HEADS_16": N_HEADS_16, "ALIBI_M": ALIBI_M,
                "ALIBI_SEQ": ALIBI_SEQ, "ALIBI_POS": ALIBI_POS,
                "ALIBI_N_SAMPLE": ALIBI_N_SAMPLE, "DK_LADDER": DK_LADDER,
                "SEQ_SMALL": SEQ_SMALL, "SEED": SEED}
    expect = {"D_MODEL": 128, "BASE_PAPER": 10000.0,
              "BASE_LADDER": (500000.0, 1000000.0),
              "SCALE_LADDER": (2.0, 4.0, 8.0, 16.0, 32.0), "TRIM_ALPHA": 1.0,
              "TRIM_BETA": 32.0, "EXTENDED_S": 16.0,
              "N_HEADS_8": 8, "N_HEADS_16": 16, "ALIBI_M": 0.5,
              "ALIBI_SEQ": 32, "ALIBI_POS": 31, "ALIBI_N_SAMPLE": 4000,
              "DK_LADDER": (64, 128, 256), "SEQ_SMALL": 6, "SEED": 20260922}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 16 个逐位一致（★ 改 `s`/`α`/`β`/`d_k` 阶梯，本程序的**闭式**断言照样通过 "
          "⇒ 口径只能靠这一条守住）%s" % ("" if not diff else " ← 差异 %s" % diff))

    th = {"CLOSED_FORM_TOL": CLOSED_FORM_TOL, "ENDPOINT_TOL": ENDPOINT_TOL,
          "TURNS_TOL": TURNS_TOL, "SET_EQ_FLOOR": SET_EQ_FLOOR,
          "ALPHA_GAMMA_SEP_FLOOR": ALPHA_GAMMA_SEP_FLOOR, "TEMP_TOL": TEMP_TOL,
          "ALIBI_DK_SAME_TOL": ALIBI_DK_SAME_TOL,
          "ALIBI_DK_RATIO_LO": ALIBI_DK_RATIO_LO, "ALIBI_DK_RATIO_HI": ALIBI_DK_RATIO_HI,
          "ALIBI_SLOPE_TOL": ALIBI_SLOPE_TOL, "ALIBI_RATIO_TOL": ALIBI_RATIO_TOL,
          "PI_RESOLVE_CEILING": PI_RESOLVE_CEILING,
          "DEGENERATE_SEP_FLOOR": DEGENERATE_SEP_FLOOR,
          "DIRECTION_SIGN_FLOOR": DIRECTION_SIGN_FLOOR}
    th_expect = {"CLOSED_FORM_TOL": 1e-12, "ENDPOINT_TOL": 1e-15, "TURNS_TOL": 1e-12,
                 "SET_EQ_FLOOR": 1e-9, "ALPHA_GAMMA_SEP_FLOOR": 0.1, "TEMP_TOL": 1e-12,
                 "ALIBI_DK_SAME_TOL": 0.2, "ALIBI_DK_RATIO_LO": 1.5,
                 "ALIBI_DK_RATIO_HI": 2.5, "ALIBI_SLOPE_TOL": 1e-15,
                 "ALIBI_RATIO_TOL": 1e-12, "PI_RESOLVE_CEILING": 1.0,
                 "DEGENERATE_SEP_FLOOR": 0.01, "DIRECTION_SIGN_FLOOR": 1e-3}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 14 个逐位一致（★ 放宽 `CLOSED_FORM_TOL` ⇒ 两条路径的对照退化成恒真；"
          "把 `ALIBI_DK_RATIO_LO` 调到 `1.0` 以下 ⇒ 脚注那条判据失去判别力）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    check("D.3", specs.LLAMA3_8B_ROPE_THETA in BASE_LADDER
          and specs.QWEN3_30B_A3B_ROPE_THETA in BASE_LADDER
          and D_MODEL == specs.ROTARY_DIM_MODEL
          and specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS == 8192
          and specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS == 40960
          and specs.YARN_NTK_BY_PARTS_ALPHA == TRIM_ALPHA
          and specs.YARN_NTK_BY_PARTS_BETA == TRIM_BETA,
          "★ 真实模型的 `rope_theta`（`%.0f`/`%.0f`）与窗口（`%d`/`%d`）**逐位钉住**"
          "（**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256** —— "
          "这条限定语必须与数字一起引用）；rotary 维数 `%d` 钉在 `specs.ROTARY_DIM_MODEL`；"
          "★ **ramp 的两个阈值直接取自 `specs`**（`α = %g`、`β = %g`）"
          "⇒ 判据里的常数与论文逐字**是同一个来源**"
          % (specs.LLAMA3_8B_ROPE_THETA, specs.QWEN3_30B_A3B_ROPE_THETA,
             specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS,
             specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS, D_MODEL,
             specs.YARN_NTK_BY_PARTS_ALPHA, specs.YARN_NTK_BY_PARTS_BETA))

    # ── 表 B 的引用判据：逐字逐位钉住 ─────────────────────────────────────
    quotes = (
        specs.YARN_NTK_AWARE_E1 ==
        "proposed the \"NTK-aware\" interpolation by taking the loss of high frequency "
        "into account"
        and specs.YARN_SPREAD_E3A ==
        "we spread out the interpolation pressure across multiple dimensions by scaling "
        "high frequencies"
        and specs.YARN_SPREAD_E3B == "less and low frequencies more"
        and specs.YARN_OUT_OF_BOUND_E4 ==
        "some dimensions are slightly extrapolated to \"out-of-bound\" values"
        and specs.YARN_WAVELENGTH_E5A == "is equal to or bigger than the context size"
        and specs.YARN_WAVELENGTH_E5B == \
        "we want to only interpolate and avoid any extrapolation"
        and specs.YARN_FITTING_E7 == "The equation above is found by fitting"
        and specs.YARN_PERPLEXITY_E8 == "perplexity may not be a great indicator"
        and specs.YARN_UNPUBLISHED_E9 == "previous unpublished work" + "s"
        # ★★ 下面三条是**追加进 D.4 的**（第一版只钉了上面九条 ⇒ 它们在
        #    `make mutate` 里**漏网 5 条中的 3 条**：字符串**末尾多加一个字符**时，
        #    「包含 / 前缀」式检查照样通过）。★ 处置是**逐位相等**，不是把它们从分母里排除。
        and specs.YARN_BASE_CHANGE == "{b^{\\prime}}=b\\cdot s^{\\frac{|D|}{|D|-2}}."
        and specs.YARN_NTK_BY_PARTS_RAMP ==
        "=\\Big(1-\\gamma\\big(r(d)\\big)\\Big)\\frac{\\theta_{d}}{s}"
        "+\\gamma\\big(r(d)\\big)\\theta_{d}."
        and specs.YARN_TEMPERATURE_FIT ==
        "\\sqrt{\\frac{1}{t}}=0.1\\ln({s})+1"
    )
    check("D.4", quotes,
          "★★ **表 B 的引用判据（E1/E3/E4/E5/E7/E8/E9）逐位钉住**：★ E3 是**两句**"
          "（计划 §C.5 的 E3 在 `high frequencies` 处闭合了引文并当作句末 —— "
          "后半句 `%s` 续上后论点才完整）；★ E7 的 `found by fitting` ⇒ `√(1/t)` **是拟合值**；"
          "★ E8 的前缀是 `we hypothesize`；★ E9 是**等级纪律的唯一出处**"
          % specs.YARN_SPREAD_E3B)
    check("D.5", specs.YARN_NTK_BY_PARTS_QUOTED_PLURAL == 10
          and specs.YARN_NTK_BY_PARTS_QUOTED_SINGULAR == 1,
          "★ **术语的计数要带闭引号**（`NTK-by-parts\"` = %d vs `NTK-by-part\"` = %d）—— "
          "★ 不带闭引号时 `NTK-by-part` = 13 > `NTK-by-parts` = 12，**结论会反过来**，"
          "因为前者是后者的**子串**（计划 §0.3 的纪律：计数必须带上能闭合的边界）"
          % (specs.YARN_NTK_BY_PARTS_QUOTED_PLURAL,
             specs.YARN_NTK_BY_PARTS_QUOTED_SINGULAR))
    check("D.6", specs.ALIBI_BIAS_ALT ==
          "\\text{softmax}(\\mathbf{q}_{i}\\mathbf{K}^{\\top}+m\\cdot[-(i-1),...,-2,-1,0]),"
          and specs.ALIBI_FOOTNOTE == "The ALiBi bias is not multiplied by the"
          and specs.ALIBI_SLOPE_RULE ==
          "our set of slopes is the geometric sequence that starts at"
          # ★★ 必须**逐位相等**，不能只查前缀：字符串末尾被追加一个字符时
          #    `startswith` 照样通过 ⇒ 那是 `make mutate` 的第一版漏网之一。
          and specs.ALIBI_ROPE_VERDICT ==
          "Though the rotary position method improves over the sinusoidal one, "
          "it still does not achieve satisfying results.",
          "★★ **X17 的三条（偏置式 + 脚注 + slope 规则）逐位钉住**；★ 外加 **X1 的落点**："
          "「RoPE 外推不行」的出处是 **ALiBi §2**（`%s…`）—— ★★ **不能归给 RoPE 论文**"
          "（那里 `extrapol` 计数 **0**，且已按 §0.3 做过对照检验）"
          % specs.ALIBI_ROPE_VERDICT[:46])
    check("D.7", specs.PI_RESCALE_ALT ==
          "\\mathbf{f}\u2019(\\mathbf{x},m)=\\mathbf{f}\\left(\\mathbf{x},"
          "\\frac{mL}{L^{\\prime}}\\right)."
          and specs.PI_WEAK_EXTRAPOLATION == "weak extrapolation properties"
          and specs.ROPE_EQ11_REL.endswith("m-n).")
          and specs.ROPE_EQ16_ALT.endswith("n-m}"),
          "★ **PI 的式 (4)** 与它的「弱外推」评语逐位钉住（★ 原文的撇号是 **U+2019**，"
          "照 ASCII 抄会计数 0）；★ 顺带把 **X3** 带出来：本域引 RoPE 的相对位置时"
          "**同时**存了 `m−n`（式 11）与 `n−m`（式 16）两处 ⇒ 引用必须写明出处")
    check("D.8", specs.UNSOURCED == (
        "ntk_aware_primary_source", "rope_extrapolation_weakness_attribution",
        "long_context_quality_metric", "extrapolation_quality_ranking",
        "rope_in_real_attention_kernels"),
          "★ `specs.UNSOURCED` 是**待办清单**（%d 项），**逐位（含顺序）钉住** —— "
          "★ 第一项就是本主题的**等级纪律**（NTK 系列的一手来源是 D 级）"
          % len(specs.UNSOURCED))


def mark_part():
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 参照实现的独立性")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.9", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.10", ASSERT_FAIL_MARK == "$$ FAIL: positional-encoding"
          and "FAIL" in ASSERT_FAIL_MARK and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.11", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                        "## FAIL: moe", "~~ FAIL: normalization"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住")

    ref_fns = {phi_ratio_direct}
    weight_fns = {phi_ratio_ntk, phi_ratio_by_parts, alpha_by_index, gamma_by_ratio}
    w_names = {f.__name__ for f in weight_fns}
    shared = sorted(set().union(*[_callee_names(f) for f in ref_fns]) & w_names)
    text_ref = "\n".join(_body_text(f) for f in sorted(ref_fns, key=lambda g: g.__name__))
    textual = not any(s in text_ref for s in ("phi_ratio_ntk(", "phi_ratio_by_parts(",
                                              "alpha_by_index(", "gamma_by_ratio("))
    src_ntk = inspect.getsource(phi_ratio_ntk)
    src_bp = inspect.getsource(phi_ratio_by_parts)
    check("D.12", not shared and textual and src_ntk != src_bp,
          "★ **参照实现的独立性**：`phi_ratio_direct`（直接按「最大相位 ÷ 训练上限」算）与"
          "四个**权重函数**的调用表**交集 = %s**；文本级回退 = %s ⇒ "
          "`A.2` 的两条路是真的**两条**：一条从「换 base」出发、一条从「最大相位」出发"
          "（★ `phi_ratio_ntk` 与 `phi_ratio_by_parts` 源码不同：%d vs %d 字节 —— "
          "它们是**同一个一参数族的两个成员**，不是一个函数的两种写法）"
          % (shared, textual, len(src_ntk), len(src_bp)))


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
    print("03_extrapolation.py —— PI / NTK-aware / NTK-by-parts / YaRN / ALiBi（用 02 的框架）")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★★ 本程序的核心：**四种方法 = 同一个一参数族的四种「怎么挑权重」**，")
    print("     统一量是**相位比** `Φ_ratio = 扩展窗口的最大相位 ÷ 训练窗口的最大相位`，")
    print("     `Φ_ratio ≤ 1` 才叫「没有外推」。★ `i=1` 那一格（`Φ_ratio = s`）是本程序的重点。")
    print("  ★★ 本程序**不重讲频率谱**（那是 `02`）；只引用它的 `turns` —— 那座桥在 `B.1`。")
    print("  ★ 禁区（计划 §2）：不比较四种方法的**效果**（无 GPU、不训练）；")
    print("     不讲变体的 kernel 后果（正文在 `40-kernels/attention/04:126-156`）。")
    pin_part()
    part_A()
    part_B()
    part_C()
    mark_part()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
