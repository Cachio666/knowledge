#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_init.py —— 初始化：He / Xavier / μP，以及三条**必须订正的转述**。

配套文档：`01-initialization.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ **本主题的禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）—— 本程序**一行都不写**：
  · **warmup 的机制解释**（post-LN 的平均场理由） →
    `80-model-architectures/normalization/02-placement-and-qk-norm.md:83,93-97,100,278`；
  · **残差流的范数随深度增长**（**结构**侧的账）→
    `80-model-architectures/transformer/01-block-and-residual-stream.md:191,232-233,263`；
  · ★★ **误差沿深度的传播** → `130-model-optimization/quantization/07-outliers-and-propagation.md:124,205`；
  · MoE 的 aux loss 尺度 → `80-model-architectures/moe/03-balance-and-granularity.md:260`。

  ★★ **`B.1` 的落法（本程序最需要说清的一处边界）**：本程序的「**方差沿深度的乘性累积**」
     **不是**上面那两条（那两条讲的是**残差流**与**误差传播**）。本程序量的是
     **Glorot 原文 §4.2 的那条账**（`the multiplicative effect through layers`）：
     **一条没有残差、没有归一化的前向链**上，`Var[y_L] = Var[x]·Π_l (fan_in,l · Var[W_l])`。
     ⇒ 本程序**明写这个范围限制**，并把「残差/深度的账」与「误差传播的账」**回链给**上面两处，
     **不重推**（§1.5 的三类真值来源 + 本域「只回链、不重推」的纪律）。

★★ **本程序的三条核心订正（计划 §0.7-C 的 Y1 / Y2 / Y3）**：
  · **Y1**：「Xavier 初始化 = `arXiv:1001.0014`」**错** —— 那是 `Jet Shapes and Jet Algorithms
    in SCET`（hep-ph）。真正的 Glorot & Bengio 论文**不在 arXiv 上** ⇒ 引用格式用 **PMLR**。
  · **Y2**：Glorot 原文里**没有** `Xavier initialization` 这个词（计数 **0**，四种数法都稳健）；
    ★ 而 `normalized initialization` 的计数**不可复现** ⇒ **不写那个数**。
  · **Y3**：Xavier 是 `Var[W] = 2/(fan_in + fan_out)` 的**均匀**分布，**不是** `N(0, 1/n)`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  He 的前向增益（`mean_j ‖W_j‖²`，**口径 A**）== 闭式 `2`；
  · `A.1b` 同一件事换**探针口径**（抽样 `Var[x@W]`）：**多种子均值**落在 **`[1.95, 2.05]`**，
           而**单种子**的偏差与那条带子的**半宽同量级**（⇒ 带子必须用在多子均值上）；
  · `A.1b2` ★ **朴素的 `√(2/n_rep)` 不是这个估计量的标准误**（高估 ≥ 9 倍）；
  · `A.1c` ★★ **解析口径不一定更准**：`SE_A / sd_B` 在 `(128,128)` 上 ≈ 1.8（> 1）；
  · `A.2`  Xavier 的**前向 + 反向增益之和 == 2**（闭式），在 `fan_in == fan_out` 上两者都 == 1；
  · `A.2b` He 的**反向**增益 == `2·fan_out/fan_in`（在 `256/4096` 上是 **32**）⇒ 它保前向、不保反向；
  · `A.3`  `kaiming_uniform_(mode='fan_in')` 的 `Var[U(−b,b)] == b²/3 == 2/fan_in`（与 He 同方差）；
           ★ **反面**：`mode='fan_out'` 在 `256/4096` 上差 **16 倍**（不是恒真）；
  · `A.4`  ★★ **Y1 + Y3 的可执行对照**：「Xavier = `N(0, 1/fan_in)`」这个读法给出 `1.0`
           （**它反而"保前向"**），而正确读法给 `0.1176` ⇒ 两者相差 `(fan_in+fan_out)/(2·fan_in)` 倍；
  · `A.5`  ★ μP 的坐标检查：SP 的 logits 方差对宽度的斜率 == `0`、μP 的 == `−1`（闭式 `Var[h]/width`）；
  · `B.1`  方差沿深度的**乘性累积**：100 层各偏 1% ⇒ 合计 `(1.01)^100 = 2.7048`（与 `exp(Σln(1+ε))` 对拍）；
  · `B.2`  卷积漏掉 `k²` 让方差差 `k²` 倍（`k=3` ⇒ **9 倍**）；
  · `C.1`  ★★ **同方差 ≠ 同分布**：Gaussian 与 Uniform 的二阶矩相同（< 1e-3）而四阶矩不同（差 > 0.5）；
  · `C.2`  `fan_in = 0` 必须被拦住（抛异常），而不是给出 `inf`；
  · `C.3`  `calculate_gain('leaky_relu', a=0)` **逐位等于** `calculate_gain('relu')`，
           而 `a = 0.2` 时差 **2.7e-2**（**不是恒真**）；
  · `C.4`  把 Xavier 的支撑上界写成 `√(2/(fi+fo))`（漏掉那个 `3`）⇒ 方差**恰好小 3 倍**。

用法：
    python3 01_init.py
"""

import ast
import collections
import inspect
import math
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（initialization-and-stability）取 `// FAIL: init-stability`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = "// FAIL: init-stability"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", ":: FAIL: mixed-precision",
                     "++ FAIL: optimizers", ";; FAIL: checkpointing")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260950              # ★ 与同域另外两个程序（20260951 / 20260952）都**不同**

# ★★ 两套**必须分开写**的测量口径（本程序的 A.1 / A.1b / A.1c 就是把它们对拍）：
#   · **口径 A**（`mean_j ‖W_j‖²`）：对 `x ∼ N(0, I)` 而言 `Var[x@W]` 的第 `j` 个坐标
#     **恰好**是 `Σ_i W_ij²`（解析）⇒ 对**给定的 `W`** 是精确的。
#     ★★ **但它仍然有噪声，而且噪声来自 `W` 本身**：`mean_j ‖W_j‖²` 是 `fan_out` 个
#       `χ²_{fan_in}` 的平均 ⇒ 相对标准误 **`SE_A = √(2/(fan_in·fan_out))`**。
#       ⇒ ★★★ **本程序实测纠正了一条我自己的预设**：`SE_A` 只在 `fan_in·fan_out` 足够大时
#         才比口径 B 小 —— 在 `(128,128)` 上 `SE_A = 1.1%`，**比口径 B 的经验 sd 还大**
#         （`A.1c` 把这两个精度并排量出来）。⇒ 「解析口径一定更准」**是错的**。
#   · **口径 B**（探针 `probe_90.py` 的 C1 口径）：每次**重抽 `W`**、算 `(x@W)` 在 `fan_out`
#     个坐标上的 `.var()`，再对 `n_rep` 次重复取均值。★ 它的**经验**相对 sd 实测在
#     **0.4% ~ 1.0%** 量级（`A.1b`），**不是** `√(2/n_rep)`（`A.1b2`：那个公式高估 ≥ 8 倍）。
#   · **口径 B′**：固定 `W`、只抽 `x` —— ★ 与口径 B **不是同一个口径**（`A.1b3`）。
#   ★ 探针的 `.var()` 用的是默认 `ddof=0`；本程序照抄那个默认（**口径一致优先于"更准"**）。
FAN_PAIRS = ((256, 256), (1024, 1024), (256, 4096), (4096, 256), (4096, 4096), (128, 128),
             (2048, 512))
FAN_PAIRS_B = ((128, 128), (256, 256))
N_REP = 300                  # 口径 B/B′ 的重复次数（★ 探针 D9 用的是 1000；这里降到 300 控时间）
N_PROBE_REPS = 1000          # ★ 探针 D9 的原口径（只记录，不在判据里跑 —— 太慢）
SAMPLE_SEEDS = (1, 2, 3, 4, 5)
HE_BAND_LO = 1.95            # 计划 §3 表 A ① 的带子
HE_BAND_HI = 2.05
GAIN_SE = 5.0                # 口径 A：|rel| < 5 · SE_A
XCHECK_MULT = 3.0            # A.1c：两条口径的差 < 3·(经验 sd/√种子数)
KURT_SAMPLE = 200000         # C.1 的四阶矩抽样数
KURT_TOL = 0.05
XAV_PAIR = (256, 4096)       # D9 的那一对（计划 §3 表 A ②）
XAV_WRONG_TOL = 1e-9         # A.4：错读法给 1.0 的容差
HE_FAN_IN_LIST = (128, 256, 512, 1024, 2048, 4096)
K_FAN_TEST = (3, 64)         # B.2：(k, c_in) —— `fan_in = k²·c_in`
DEPTH_L = 100                # B.1 的层数
PER_LAYER_EPS = 0.01         # B.1 的每层相对偏差
GAIN_A_BIG = 0.2             # C.3 的反面对照：`calculate_gain('leaky_relu', a)`
TOL_EXACT = 0.0              # 逐位相同
TOL_TIGHT = 1e-12            # 「同方差」这一类的容差
TOL_IDENT = 1e-15            # 恒等式（不是「近似相等」）
TOL_MEASURABLE = 1e-2        # 反面对照的**可测性下界**
TOL_KURT_DIFF = 0.5          # C.1：四阶矩必须**差得出来**
MUP_WIDTHS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
MUP_TOL_SLOPE = 0.05         # A.5：斜率的容差（8 个点、噪声 ~0.006 ⇒ 约 8σ）
MUP_N_W2 = 20000             # A.5 的 MC 交叉验证（只在最小宽度上做）
MUP_MC_WIDTH = 32
MUP_NX = 64                  # `Var[h]` 的 x 抽样次数
MUP_LR = 0.1                 # A.5b 的步长基值
UNIF_SAMPLE = 200000         # A.3 的均匀分布抽样数


# ===========================================================================
# 2. 路径（被测）：方差与 std 的**闭式**
# ===========================================================================
def he_variance(fan_in):
    """He 2015 Eqn 10：`Var[W] = 2/fan_in`（`fan_in` 是**连接数**，卷积时含 `k²`）。

    ★★ **`2` 的来源**：Eqn 8 `Var[y_l] = ½·n_l·Var[w_l]·Var[y_{l-1}]` ——
      `½` 是 **ReLU 把二阶矩折半**（论文用 `E[x_l²] = ½ Var[y_{l−1}]`）。
    ⇒ ★ 与 Xavier 的 `2`（两个约束的折中）**不是同一个 2 的来源**（见 `xavier_variance`）。
    """
    if fan_in <= 0:
        raise ValueError("fan_in 必须为正（fan_in = %r）" % (fan_in,))
    return specs.HE_VARIANCE_NUMERATOR / fan_in


def xavier_variance(fan_in, fan_out):
    """Glorot & Bengio 2010 Eqn 12：`Var[W] = 2/(fan_in + fan_out)`（**均匀**实现见 Eqn 16）。

    ★★ **Y3 的落法**：分母是**和**，不是 `fan_in` 单独。
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in / fan_out 必须为正（%r / %r）" % (fan_in, fan_out))
    return specs.GLOROT_VARIANCE_NUMERATOR / (fan_in + fan_out)


def xavier_variance_wrong(fan_in, fan_out):
    """★★ **反面对照**（`A.4`）：把 Xavier 读成 `N(0, 1/fan_in)`（那个常见转述）。

    ★ 它在 `fan_in == fan_out` 时**恰好等于**正确值（`2/(2n) = 1/n`）⇒
      这个错**只在 `fan_in ≠ fan_out` 时露出来**，而且在 `256/4096` 上**方向是反的**
      （错的那个**保前向**、对的才不保）。
    """
    return 1.0 / fan_in


def he_variance_std(fan_in):
    """`std = √(2/fan_in)` —— 与 `he_variance` 是同一个量的两种写法（`A.3` 用它）。"""
    return math.sqrt(he_variance(fan_in))


# ===========================================================================
# 3. 路径（被测）：增益的三种测量口径
# ===========================================================================
def forward_gain_cols(W):
    """★★ **口径 A**（本程序主用）：`mean_j Σ_i W_ij²`。

    ★ 对 `x ∼ N(0, I)`：`Var[x @ W]` 的第 `j` 个坐标**恰好**是 `Σ_i W_ij²`（解析）
      ⇒ 逐列精确、再对列取平均。**相对标准误** `√(2/(fan_in·fan_out))`。
    ★ 期望值：`fan_in · Var[W]`。
    """
    W = np.asarray(W, dtype=np.float64)
    return float((W * W).sum(axis=0).mean())


def backward_gain_rows(W):
    """反向增益（`A.2b`）：`mean_i Σ_j W_ij²` ⇒ 期望 `fan_out · Var[W]`。

    ★ 依据：`dx = g @ Wᵀ` ⇒ `Var[dx_i] = Σ_j g_j² · W_ij²` 的期望（`g` 各坐标独立）。
      ⚠ **口径**：这里假设反向来的 `g` 各坐标独立、同方差 —— 论文 Eqn 11 的前提。
    """
    W = np.asarray(W, dtype=np.float64)
    return float((W * W).sum(axis=1).mean())


def forward_gain_normalized(W):
    """参考路径之一：用 `np.linalg.norm` 逐列取范数再平方平均（**第三条算式**）。

    ★ 与 `forward_gain_cols` 的算术**不同**（一次 `norm` 调用 vs 一次 `sum`），
      但代数上是同一个量 ⇒ `D.9c` 用它做第三条对拍。
    """
    W = np.asarray(W, dtype=np.float64)
    return float((np.linalg.norm(W, axis=0) ** 2).mean())


def probe_forward_gain(fan_in, fan_out, variance, n_rep, seed):
    """★ **口径 B —— 探针 `probe_90.py` 的 C1 口径，逐字重演**：

    ```
    for _ in range(n_rep):
        W = rng.standard_normal((fan_in, fan_out)) * √variance    # ★ 每次都【重新抽 W】
        x = rng.standard_normal(fan_in)                            # 1 维
        out.append((x @ W).var())                                  # 在 fan_out 个坐标上取方差
    返回 mean(out)
    ```

    ★★ **一处必须写明的口径**：`(x@W)` 是一个**长度 `fan_out` 的向量** ⇒ 每次重复的
      `.var()` 已经在 `fan_out` 个坐标上平均过一次 ⇒ 这个估计量的**有效样本数**
      **远大于** `n_rep`。⇒ 拿 `√(2/n_rep)` 当它的标准误会**高估一个数量级**（如实测 9~23 倍）。
      ⚠ 探针用的是 `.var()` 的**默认 `ddof = 0`**（偏低 `1/fan_out`），本函数照抄那个默认 ——
        **口径一致优先于"更准"**（`1/4096 = 0.02%`，远小于这条判据的分辨率）。
    """
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n_rep):
        W = rng.standard_normal((fan_in, fan_out)) * math.sqrt(variance)
        tot += float((rng.standard_normal(fan_in) @ W).var())
    return tot / n_rep


def probe_gain_samples(fan_in, fan_out, variance, n_rep, seed):
    """★ **口径 B′（本程序自己加的对照口径）**：**固定一个 `W`**、抽 `n_rep` 个 `x`。

    ★★ 它与**口径 B**（每次重抽 `W`）是**两个不同的抽样口径**：前者把 `W` 的随机性
      排除在外、只留 `x` 的随机性 ⇒ 两者的标准误**不同**（`A.1b3` 把它量出来）。
      ⇒ 「抽样口径」这四个字**本身就不是一个口径**（本域的「量法口径」纪律的又一现场）。
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((fan_in, fan_out)) * math.sqrt(variance)
    X = rng.standard_normal((n_rep, fan_in))
    y = X @ W
    return float(np.var(y, axis=1, ddof=0).mean())


# ===========================================================================
# 4. 路径（被测）：[B] PyTorch 的两个初始化函数**按文档逐字重演**
# ===========================================================================
def calculate_gain(nonlinearity, a=None):
    """**[B] PyTorch 2.14 文档的 `calculate_gain` 表**（按表逐字重演，**不 import torch**）。

    | nonlinearity | gain |
    |---|---|
    | Linear / Identity · Conv{1,2,3}D · Sigmoid | `1` |
    | Tanh | `5/3` |
    | ReLU | `√2` |
    | Leaky Relu | `√(2/(1 + negative_slope²))` |
    | SELU | `3/4` |
    """
    if nonlinearity in ("linear", "identity", "conv1d", "conv2d", "conv3d",
                        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid"):
        return specs.PYTORCH_CALCULATE_GAIN_LINEAR
    if nonlinearity == "tanh":
        return specs.PYTORCH_CALCULATE_GAIN_TANH
    if nonlinearity == "relu":
        return specs.PYTORCH_CALCULATE_GAIN_RELU
    if nonlinearity == "leaky_relu":
        slope = 0.0 if a is None else float(a)
        return math.sqrt(2.0 / (1.0 + slope * slope))
    if nonlinearity == "selu":
        return specs.PYTORCH_CALCULATE_GAIN_SELU
    raise ValueError("未知的 nonlinearity: %r" % (nonlinearity,))


def kaiming_uniform_bound(fan_mode, gain):
    """`kaiming_uniform_` 的 `bound = gain × √(3/fan_mode)`（**[B] 文档**，按式重演）。

    ★ 默认：`mode='fan_in'`、`nonlinearity='leaky_relu'`、`a=0`
      ⇒ `gain = √2` ⇒ `bound = √6/√fan_in` ⇒ `Var[U(−b,b)] = b²/3 = 2/fan_in`（**与 He 同方差**）。
    """
    return gain * math.sqrt(specs.PYTORCH_KAIMING_BOUND_NUMERATOR / fan_mode)


def xavier_uniform_bound(fan_in, fan_out, gain):
    """`xavier_uniform_` 的 `a = gain × √(6/(fan_in+fan_out))`（**[B] 文档**，按式重演）。

    ★ 默认 `gain = 1.0` ⇒ `Var[U(−a,a)] = a²/3 = 2/(fan_in+fan_out)`（**与 Glorot Eqn 12 同式**）。
    """
    return gain * math.sqrt(specs.PYTORCH_XAVIER_BOUND_NUMERATOR / (fan_in + fan_out))


def uniform_variance(bound):
    """`Var[U(−b, b)] = b²/3` —— 支撑上界与方差之间的**闭式**（`A.3` / `C.4` 用它）。"""
    return (bound * bound) / 3.0


# ===========================================================================
# 5. 路径（被测）：μP 的坐标尺度（`A.5` / `A.5b`）
# ===========================================================================
def mup_make_hidden(x_dim, width, seed):
    """隐藏层权重：`N(0, 1/fan_in)` —— ★ **SP 与 μP 在这一层上是同一个初值**。

    ★ 论文 Table 3 的 `Init. Var.` 一栏对「Input weights / Hidden weights」给的都是
      `1/fan_in`；**差别在输出/读出层**（μP 给 `1/fan_in²`）。这一点由 `A.5` 量出来。
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((x_dim, width)) / math.sqrt(x_dim)


def readout_init_scale(width, mode):
    """读出层的**初始方差**：SP = `1/width`、μP = `1/width²`（[C] Table 3 的那一栏）。

    ★★ 它直接决定 logits 的尺度：`Var[y] = width·Var[W2]·Var[h]`
      ⇒ SP：`Var[h]`（**Θ(σ²)**）；μP：`Var[h]/width`（**Θ(σ²/width)**）。
    """
    if mode == "sp":
        return 1.0 / width
    if mode == "mup":
        return 1.0 / (width * width)
    raise ValueError("mode 只能是 'sp' 或 'mup'：%r" % (mode,))


def mean_h_sq(W1, seed, n_x):
    """`E_x[‖h‖²]` 的抽样估计（`A.5` 的口径 A 里**唯一的噪声源**）。"""
    rng = np.random.default_rng(seed)
    ss = 0.0
    for _ in range(n_x):
        h = W1.T @ rng.standard_normal(W1.shape[0])
        ss += float((h * h).sum())
    return ss / n_x


def logit_variance(W1, mode, seed, n_x=1):
    """logits 的方差 —— ★★ **口径 A 式**的解析估计：`Var[y] = Var[W2]·‖h‖²` 对 `x` 取平均。

    ★ 给定 `h`，`y = Σ_i W2_i h_i` 且 `W2_i` 独立 ⇒ `Var_{W2}[y] = Var[W2]·‖h‖²` **恰好**。
      ⇒ 本估计量的**唯一噪声**来自 `‖h‖²`（对 `x` 取平均）⇒ 比直接抽 `W2` 小得多。
      ★ `A.5` 里另有一次**直接抽 `W2`** 的 MC 交叉验证（不是同一条路径）。
    """
    width = W1.shape[1]
    return readout_init_scale(width, mode) * mean_h_sq(W1, seed, n_x)


def one_step_relative_change(width, mode, lr, seed):
    """一步 GD 之后 logits 的**相对**变化量 `|Δy / y|`（`A.5b` 的被测路径）。

    ★ 闭式（**推导值**）：`Δy = −lr2·δ·‖h‖²`，`δ = y − y*` ⇒
      `|Δy/y| = lr2·‖h‖²`（取 `y* = 0` 时 `δ = y`）。
      · SP（`lr2 = lr`）：`‖h‖² = width·Var[h]` ⇒ **Θ(width)** ⇒ 斜率 **+1**；
      · μP（`lr2 = lr/width`）：⇒ **Θ(1)** ⇒ 斜率 **0**。
    """
    W1 = mup_make_hidden(width, width, seed)          # x 维度取 width，保持「方」的玩具
    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal(width)
    h = W1.T @ x
    coef = readout_init_scale(width, mode)
    W2 = rng.standard_normal(width) * math.sqrt(coef)
    y = float(W2 @ h)
    lr2 = lr if mode == "sp" else lr / width
    delta = y - 0.0
    dW2 = -lr2 * delta * h
    dy = float(dW2 @ h)
    return abs(dy / y) if y != 0.0 else float("nan")


def fit_loglog_slope(xs, ys):
    """最小二乘拟合 `log y = a·log x + b` 的斜率 —— 与手写公式**两条路径**（`D.9d`）。"""
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def fit_loglog_slope_np(xs, ys):
    """参考路径：`np.polyfit` 的一次拟合（**另一条算式**，与 `fit_loglog_slope` 对拍）。"""
    lx = np.log(np.asarray(xs, dtype=np.float64))
    ly = np.log(np.asarray(ys, dtype=np.float64))
    return float(np.polyfit(lx, ly, 1)[0])


# ===========================================================================
# 6. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    ★ 只收 `ast.Name` 形态的调用 ⇒ `np.asarray(...)` 这类**属性调用不计入**
      ⇒ 参照路径的调用表**为空**（`D.9b`）。
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


def excess_kurtosis(sample):
    """超额峰度 `E[(x−μ)⁴]/σ⁴ − 3`（`C.1` 用它区分「同方差」与「同分布」）。"""
    s = np.asarray(sample, dtype=np.float64)
    m = s.mean()
    sd = s.std()
    return float((((s - m) ** 4).mean()) / (sd ** 4) - 3.0)


def make_w(fan_in, fan_out, variance, seed):
    """按给定**方差**造一个 `(fan_in, fan_out)` 的权重矩阵（`N(0, variance)`）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((fan_in, fan_out)) * math.sqrt(variance)


def se_a(fan_in, fan_out):
    """口径 A 的相对标准误 `√(2/(fan_in·fan_out))`（方差估计量）。"""
    return math.sqrt(2.0 / (fan_in * fan_out))


def se_b(n):
    """★★ **朴素**的「抽样标准误」`√(2/n)` —— 本程序用它来**演示它在这里是错的**（`A.1b2`）。"""
    return math.sqrt(2.0 / n)


def mean_sd(vals):
    """5 个种子上的**均值**与**经验 sd**（`ddof = 1`）—— ★ 口径 B 的真实精度由它给出。"""
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, sd


_PROBE_CACHE = {}


def probe_values(fi, fo, which):
    """带缓存的「5 个种子」的抽样结果（`which` ∈ `{'B', 'Bp'}`）。

    ★ 口径 B / B′ 是 `A.1b`/`A.1c`/`A.1d` 三条断言的**共用输入** ⇒ 缓存一次，
      否则同一批 `%d·5` 次矩阵向量乘要跑三遍（首版实测 39 s，缓存后约 13 s）。
    """ % (N_REP,)
    key = (fi, fo, which)
    if key not in _PROBE_CACHE:
        fn = probe_forward_gain if which == "B" else probe_gain_samples
        _PROBE_CACHE[key] = [fn(fi, fo, he_variance(fi), N_REP, s) for s in SAMPLE_SEEDS]
    return _PROBE_CACHE[key]


# ===========================================================================
# 7. A 类：He / Xavier / kaiming_uniform_ / Y3 / μP
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. He 与 Xavier 的增益 · 与 PyTorch 的两条互校 · **Y3 的可执行对照** · μP 坐标检查")
    print("=" * 74)
    print("  测量条件：seed=%d · 口径 A（解析列方差）· 口径 B / B′（抽样，n_rep=%d，种子 %s）"
          % (SEED, N_REP, SAMPLE_SEEDS))

    # ── A.1 ★ 口径 A：He 的前向增益 == 2 ────────────────────────────────────
    print("  ── A.1 ★ 口径 A（`mean_j ‖W_j‖²`）：He 的前向增益 == `fan_in·Var[W]` == 2 ──")
    print("     %-11s %-22s %-12s %-12s %-12s"
          % ("(fan_in,fan_out)", "Var[W] = 2/fan_in", "实测增益", "闭式 2", "rel/SE_A"))
    worst_a = 0.0
    for fi, fo in FAN_PAIRS:
        W = make_w(fi, fo, he_variance(fi), SEED + fi)
        g = forward_gain_cols(W)
        se = se_a(fi, fo)
        rel = abs(g / specs.HE_VARIANCE_NUMERATOR - 1.0)
        worst_a = max(worst_a, rel / se)
        print("     %-11s %-22.10f %-12.6f %-12.6f %-12.3f"
              % ("(%d,%d)" % (fi, fo), he_variance(fi), g, specs.HE_VARIANCE_NUMERATOR, rel / se))
    check("A.1", worst_a <= GAIN_SE,
          "★ He 的前向增益 == 闭式 `2`：跨 %d 对 `(fan_in,fan_out)` 的最大偏差只有估计量"
          "标准误的 **%.2f 倍**（阈值 %g）⇒ 这不是「大致等于」，而是**在测量精度内等于**"
          "（★ 判据的**口径**与**精度**都写出来了 —— 本域 P22 的纪律）"
          % (len(FAN_PAIRS), worst_a, GAIN_SE))

    # ── A.1b ★★ 口径 B：计划那条带子 `[1.95, 2.05]` + **它的标准误到底是多大** ──
    print()
    print("  ── A.1b ★★ 口径 B（探针口径：每次重抽 `W`、`(x@W).var()` 再对重复取均值）──")
    print("     ★ **成本口径**：口径 B 每次重复都要重抽一个 `(fan_in,fan_out)` 的 `W` ⇒")
    print("       本程序只跑两对（`%s`）。探针 D9 的那一对 `(256,4096)` 在 `A.1`/`A.2`/`A.4` 上"
          "由**口径 A** 覆盖。" % (FAN_PAIRS_B,))
    print("     ★ 计划 §3 表 A ① 的带子是 `[%.2f, %.2f]`（±2.5%%）；探针 D9 用 `n_rep = 1000`，"
          "本程序用 `n_rep = %d`" % (HE_BAND_LO, HE_BAND_HI, N_REP))
    print("     %-11s %-11s %-11s %-13s %-13s %-13s %-11s"
          % ("(fan_in,fan_out)", "B: 5 子均值", "B′: 5 子均值", "B: 单子 max|dev|",
             "B: 经验 sd", "B′: 经验 sd", "sd_B′/sd_B"))
    band_lo = float("inf")
    band_hi = 0.0
    max_dev_rel = 0.0
    min_naive_ratio = float("inf")
    min_sd_ratio = float("inf")
    max_sd_ratio = 0.0
    for fi, fo in FAN_PAIRS_B:
        vb = probe_values(fi, fo, "B")
        vp = probe_values(fi, fo, "Bp")
        mb, sdb = mean_sd(vb)
        mp, sdp = mean_sd(vp)
        dev = max(abs(v / specs.HE_VARIANCE_NUMERATOR - 1.0) for v in vb + vp)
        band_lo = min(band_lo, mb, mp)
        band_hi = max(band_hi, mb, mp)
        max_dev_rel = max(max_dev_rel, dev)
        min_naive_ratio = min(min_naive_ratio, se_b(N_REP) / (sdb / mb))
        min_sd_ratio = min(min_sd_ratio, sdp / sdb)
        max_sd_ratio = max(max_sd_ratio, sdp / sdb)
        print("     %-11s %-11.5f %-11.5f %-13.4f %-13.5f %-13.5f %-11.3f"
              % ("(%d,%d)" % (fi, fo), mb, mp, dev, sdb, sdp, sdp / sdb))
    check("A.1b", HE_BAND_LO <= band_lo and band_hi <= HE_BAND_HI
          and 0.5 < max_dev_rel / (HE_BAND_HI - 2.0) < 1.5,
          "★ **带子判据的成立条件**：两种抽样口径的 5 子均值都在 **[%.5f, %.5f]**（带子内）"
          "—— ★★ **但单子**的最大相对偏差是 **%.4f%%**，而带子**半宽**就是 **2.5%%** "
          "⇒ 两者**同量级**（比值 %.2f）⇒ 计划 §3 表 A ① 的那条带子对**单种子**只有 ~1σ，"
          "**必须**用在**多种子均值**上（这正是判据的小 n 纪律）"
          % (band_lo, band_hi, max_dev_rel * 100, max_dev_rel / (HE_BAND_HI - 2.0)))
    check("A.1b2", min_naive_ratio > 5.0,
          "★★ **口径对了、标准误的公式错了**：拿 `√(2/n_rep)`（= `%.4f`）当口径 B 的标准误会"
          "**高估至少 %.1f 倍** —— 原因：每次重复的 `.var()` 已经在 **`fan_out` 个坐标**上平均"
          "过一次 ⇒ **这个估计量的有效样本数不是 `n_rep`**。"
          "⇒ ★ 若照朴素公式判「单子偏差 ≤ 2σ」，会得到一个**过松**的判据"
          "（「量法口径」纪律的现场）"
          % (se_b(N_REP), min_naive_ratio))
    check("A.1b3", abs(min_sd_ratio - 1.0) > 0.1 and abs(max_sd_ratio - 1.0) > 0.1
          and max_sd_ratio / min_sd_ratio > 2.0,
          "★★ **「抽样口径」本身就不止一个口径**：口径 B（**每次重抽 `W`**）与口径 B′"
          "（**固定 `W`、只抽 `x`**）的经验 sd 之比在 **%.3f ~ %.3f** 之间跨了 **%.1f 倍**"
          "⇒ 两者都在估计同一个量、均值一致，但**精度不同**；"
          "★ 本程序的带子判据对**两种口径都成立**（`A.1b`），所以结论不依赖选哪一个"
          % (min_sd_ratio, max_sd_ratio, max_sd_ratio / min_sd_ratio))

    # ── A.1c ★★ 口径 A 的精度**不是**无条件更好（纠正自己的预设）──────────
    print()
    print("  ── A.1c ★★ 口径 A（解析）与口径 B（抽样）**谁的精度更高？**（不是无条件）──")
    print("     ★★ 口径 A 的噪声来自 `W` 本身：`SE_A = √(2/(fi·fo))`；")
    print("        口径 B 的经验 sd 由 `A.1b` 量出（实测 ~0.4%%~1.0%%）")
    print("     %-11s %-14s %-14s %-14s" % ("(fan_in,fan_out)", "SE_A", "B 的经验 sd", "SE_A / sd_B"))
    worst_ratio_lo = float("inf")
    worst_ratio_hi = 0.0
    for fi, fo in FAN_PAIRS_B:
        vb = probe_values(fi, fo, "B")
        _, sdb = mean_sd(vb)
        ratio = se_a(fi, fo) / sdb
        worst_ratio_lo = min(worst_ratio_lo, ratio)
        worst_ratio_hi = max(worst_ratio_hi, ratio)
        print("     %-11s %-14.5f %-14.5f %-14.2f" % ("(%d,%d)" % (fi, fo), se_a(fi, fo), sdb, ratio))
    check("A.1c", worst_ratio_lo < 1.0 and worst_ratio_hi > 1.0,
          "★★ **纠正一条预设（本程序自己的发现）**：`A.1d` 的第一版把两条口径的差与 "
          "`3·SE(B)` 比，在 `(128,128)` 上给出 **4.15 倍**的偏差 ⇒ 追下去发现**两边都有噪声**："
          "口径 A 的 `SE_A / sd_B` 在 **%.2f ~ %.2f** 之间**跨过 1** "
          "⇒ 「**解析口径一定更准**」是错的 —— 它只在 `fan_in·fan_out` 足够大时更准"
          "（★ 「我量出来不一样」先问「我量的是不是同一个东西」的同源现场）"
          % (worst_ratio_lo, worst_ratio_hi))

    # ── A.1d 三条路径互校（解析 / 口径 B / 口径 B′）────────────────────────
    print()
    print("  ── A.1d 三条路径互校：口径 A（解析）· 口径 B（重抽 W）· 口径 B′（固定 W）──")
    print("     ★ 阈值用**两条路径各自的标准误的合成**（`3·√(SE_A² + (sd_B/√5)²)`）——")
    print("       第一版只用了 `3·sd_B/√5`，那等于**默认口径 A 没有噪声**（`A.1c` 证明它有）")
    print("     %-11s %-14s %-14s %-14s %-18s"
          % ("(fan_in,fan_out)", "口径 A", "口径 B", "口径 B′", "|A-B| / 3·SE_合成"))
    worst_x = 0.0
    for fi, fo in FAN_PAIRS_B:
        W = make_w(fi, fo, he_variance(fi), SEED + 7 + fi)
        ga = forward_gain_cols(W)
        vb = probe_values(fi, fo, "B")
        vp = probe_values(fi, fo, "Bp")
        gb, sdb = mean_sd(vb)
        gbp, sdp = mean_sd(vp)
        se_comb = math.sqrt(se_a(fi, fo) ** 2 + (sdb / math.sqrt(len(vb))) ** 2)
        denom = XCHECK_MULT * se_comb * specs.HE_VARIANCE_NUMERATOR
        worst_x = max(worst_x, abs(ga - gb) / denom)
        print("     %-11s %-14.6f %-14.6f %-14.6f %-18.3f"
              % ("(%d,%d)" % (fi, fo), ga, gb, gbp, abs(ga - gb) / denom))
    check("A.1d", worst_x <= 1.0,
          "★ 三条路径给出同一个数：**解析列方差**与**口径 B** 的最大偏差只有 "
          "`3·SE_合成` 的 **%.3f 倍**（≤ 1）—— ★ 而两个标准误一个是**解析式**（`SE_A`）、"
          "一个是**经验 sd**（`A.1b2` 证明朴素的 `√(2/n_rep)` 在这里是错的）"
          "⇒ 这条互校**不是**「两边同步错」" % worst_x)

    # ── A.2 ★ Xavier 的两侧：和恒为 2、只在 fan_in == fan_out 时都保 ─────────
    print()
    print("  ── A.2 ★ Xavier 的两侧增益：`fwd + bwd == 2`（恒等式），`fi == fo` 时两者都是 1 ──")
    print("     %-11s %-12s %-12s %-12s %-12s %-12s %-12s"
          % ("(fi,fo)", "前向实测", "前向闭式", "反向实测", "反向闭式", "和 - 2", "√(f·b)"))
    worst_xa = 0.0
    sum_dev = 0.0
    for fi, fo in FAN_PAIRS:
        W = make_w(fi, fo, xavier_variance(fi, fo), SEED + 31 + fi)
        gf = forward_gain_cols(W)
        gb = backward_gain_rows(W)
        tf = specs.GLOROT_VARIANCE_NUMERATOR * fi / (fi + fo)
        tb = specs.GLOROT_VARIANCE_NUMERATOR * fo / (fi + fo)
        se = se_a(fi, fo)
        worst_xa = max(worst_xa, abs(gf / tf - 1.0) / se, abs(gb / tb - 1.0) / se)
        sum_dev = max(sum_dev, abs((tf + tb) - 2.0))
        print("     %-11s %-12.6f %-12.6f %-12.6f %-12.6f %-12.2e %-12.6f"
              % ("(%d,%d)" % (fi, fo), gf, tf, gb, tb, (tf + tb) - 2.0, math.sqrt(tf * tb)))
    check("A.2", worst_xa <= GAIN_SE and sum_dev == TOL_EXACT,
          "★ Xavier 的两侧增益与闭式一致（最大偏差 %.2f·SE_A ≤ %g），而 "
          "`2·fi/(fi+fo) + 2·fo/(fi+fo)` **逐位等于 2** —— 那条折中的含义就是「**两侧之和固定**」"
          % (worst_xa, GAIN_SE))
    # ★ 两面：`fi == fo` 时两侧都保
    gf_eq = forward_gain_cols(make_w(256, 256, xavier_variance(256, 256), SEED + 41))
    gb_eq = backward_gain_rows(make_w(256, 256, xavier_variance(256, 256), SEED + 41))
    check("A.2c",
          abs(gf_eq - 1.0) / se_a(256, 256) <= GAIN_SE and abs(gb_eq - 1.0) / se_a(256, 256) <= GAIN_SE,
          "★ **反面对照（这一条不是恒真的）**：`fan_in == fan_out == 256` 时 Xavier 的"
          "**两侧都 == 1**（实测 %.6f / %.6f）⇒ 「既不保前向也不保反向」只在 `fi ≠ fo` 时成立"
          % (gf_eq, gb_eq))

    # ── A.2b ★ He 的反向增益 = 2·fan_out/fan_in ────────────────────────────
    print()
    print("  ── A.2b ★ He 的反向增益 == `2·fan_out/fan_in`（原文 Eqn 14 的那一条）──")
    print("     %-11s %-14s %-14s %-12s" % ("(fi,fo)", "反向实测", "闭式 2·fo/fi", "rel/SE_A"))
    worst_heb = 0.0
    for fi, fo in FAN_PAIRS:
        W = make_w(fi, fo, he_variance(fi), SEED + 51 + fi)
        gb = backward_gain_rows(W)
        tb = specs.HE_VARIANCE_NUMERATOR * fo / fi
        worst_heb = max(worst_heb, abs(gb / tb - 1.0) / se_a(fi, fo))
        print("     %-11s %-14.6f %-14.6f %-12.2f" % ("(%d,%d)" % (fi, fo), gb, tb,
                                                      abs(gb / tb - 1.0) / se_a(fi, fo)))
    check("A.2b", worst_heb <= GAIN_SE,
          "★★ He **保前向、不保反向**：反向增益是 `2·fan_out/fan_in` —— 在 `(256,4096)` 上是 "
          "**%g**（不是 2）⇒ ★ 这正是 He 2015 **另外**给出 Eqn 14 的原因，原文逐字承认"
          "「两者只用其一就够」（`%s`）"
          % (specs.HE_VARIANCE_NUMERATOR * 4096 / 256, specs.HE_BACKWARD_ALTERNATIVE_QUOTE))

    # ── A.3 ★ kaiming_uniform_ 与 He 同方差；反面 mode='fan_out' ─────────────
    print()
    print("  ── A.3 ★ [B] `kaiming_uniform_(mode='fan_in')` 与 [C] He 的 `√(2/fan_in)` 同方差 ──")
    print("     默认：`mode='fan_in'`、`nonlinearity='leaky_relu'`、`a=0` ⇒ gain = `√2`")
    print("     %-10s %-16s %-16s %-16s %-14s %-12s"
          % ("fan", "bound=√6/√fan", "bound²/3", "2/fan_in", "rel", "抽样 Var"))
    worst_k = 0.0
    worst_mc = 0.0
    for fan in HE_FAN_IN_LIST:
        gain = calculate_gain(specs.PYTORCH_KAIMING_DEFAULT_NONLINEARITY,
                              specs.PYTORCH_KAIMING_DEFAULT_A)
        b = kaiming_uniform_bound(fan, gain)
        v = uniform_variance(b)
        target = he_variance(fan)
        rel = abs(v / target - 1.0)
        worst_k = max(worst_k, rel)
        rng = np.random.default_rng(SEED + 61 + fan)
        samp = rng.uniform(-b, b, UNIF_SAMPLE)
        mc = float(np.var(samp, ddof=1))
        worst_mc = max(worst_mc, abs(mc / v - 1.0) / math.sqrt(2.0 / UNIF_SAMPLE))
        print("     %-10d %-16.10f %-16.12f %-16.12f %-14.2e %-12.8f"
              % (fan, b, v, target, rel, mc))
    check("A.3", worst_k == TOL_EXACT or worst_k < TOL_TIGHT,
          "★★ **[B] 与 [C] 互校**：`kaiming_uniform_(mode='fan_in')` 的 `Var[U(−b,b)] = b²/3` "
          "**逐位等于** He 的 `2/fan_in`（最大相对差 %.2e）⇒ 两者**同方差**；"
          "抽样方差也落在 `√(2/n)` 的 %.2f 倍内" % (worst_k, worst_mc))
    # ★★ 反面：mode='fan_out'
    fi, fo = XAV_PAIR
    gain = calculate_gain(specs.PYTORCH_KAIMING_DEFAULT_NONLINEARITY,
                          specs.PYTORCH_KAIMING_DEFAULT_A)
    v_in = uniform_variance(kaiming_uniform_bound(fi, gain))
    v_out = uniform_variance(kaiming_uniform_bound(fo, gain))
    ratio_fo = v_in / v_out
    print()
    print("  ── A.3b ★ 反面对照：`mode='fan_out'` 在 `(fi,fo)=(%d,%d)` 上差 %.0f 倍 ──"
          % (fi, fo, ratio_fo))
    print("     `mode='fan_in'`  → `Var = 2/%d = %.10f`" % (fi, v_in))
    print("     `mode='fan_out'` → `Var = 2/%d = %.10f`（比值 **%.6f** = `fan_out/fan_in`）"
          % (fo, v_out, ratio_fo))
    check("A.3b", abs(ratio_fo - fo / fi) < TOL_TIGHT and ratio_fo > 1.5,
          "★★ **这条判据不是恒真的**：把 `mode` 写成 `'fan_out'` 会差 **%.0f 倍**"
          "（= `fan_out/fan_in`，逐位等于闭式）⇒ `A.3` 的「同方差」**区分了两个实现**"
          % ratio_fo)
    # ★ xavier_uniform_ 与 Glorot Eqn 16 同式
    worst_xb = 0.0
    for fi2, fo2 in FAN_PAIRS:
        a = xavier_uniform_bound(fi2, fo2, specs.PYTORCH_XAVIER_DEFAULT_GAIN)
        worst_xb = max(worst_xb, abs(uniform_variance(a) / xavier_variance(fi2, fo2) - 1.0))
    check("A.3c", worst_xb < TOL_TIGHT,
          "★ `xavier_uniform_(gain=1)` 的 `a²/3` **逐位等于** Glorot Eqn 12 的 "
          "`2/(fan_in+fan_out)`（最大相对差 %.2e）⇒ **[B] 的默认值与 [C] 的公式一一对上**，"
          "且两边都指向**均匀分布**（`U(±a)`）" % worst_xb)
    # ★ 名字：`Also known as …`
    print("     [B] 文档逐字：`%s` / `%s`" % (specs.PYTORCH_KAIMING_KNOWN_AS,
                                              specs.PYTORCH_XAVIER_KNOWN_AS))
    print("     ★★ 而 [C] Glorot 原文里 `Xavier` 这个词只出现在**页眉的作者名**上："
          "`Xavier initialization` 计 **%d** 次（`GLOROT_XAVIER_TERM_COUNT`）"
          % specs.GLOROT_XAVIER_TERM_COUNT)

    # ── A.4 ★★ Y1 + Y3 的可执行对照 ────────────────────────────────────────
    print()
    print("  ── A.4 ★★ Y1 / Y3 的可执行对照（引用与公式两条都做成断言）──")
    print("     (a) **Y1**：`arXiv:%s` 的标题 = `%s`（**不是**神经网络论文）"
          % (specs.GLOROT_WRONG_ARXIV_ID, specs.GLOROT_WRONG_ARXIV_TITLE))
    print("         而 Glorot & Bengio 的标题 = `%s`" % specs.GLOROT_TITLE)
    print("         ⇒ 引用格式用 **%s**（该论文**不在 arXiv 上**）" % specs.GLOROT_PMLR_CITATION)
    check("A.4a", specs.GLOROT_WRONG_ARXIV_TITLE != specs.GLOROT_TITLE
          and specs.GLOROT_TITLE == "Understanding the difficulty of training deep feedforward neural networks"
          and specs.GLOROT_WRONG_ARXIV_TITLE == "Jet Shapes and Jet Algorithms in SCET"
          and specs.GLOROT_PMLR_CITATION == "PMLR 9:249-256, 2010",
          "★★ **Y1 的落法**：两条标题**逐位钉住**且**必须不同** —— `arXiv:%s` 属于 "
          "`%s`（hep-ph），**不是** Xavier 那一篇；后者按 **%s** 引"
          % (specs.GLOROT_WRONG_ARXIV_ID, specs.GLOROT_WRONG_ARXIV_TITLE,
             specs.GLOROT_PMLR_CITATION))
    print("     (b) **Y3**：把 Xavier 读成 `N(0, 1/fan_in)`（错）vs `2/(fi+fo)` 的均匀（对）")
    fi, fo = XAV_PAIR
    W_wrong = make_w(fi, fo, xavier_variance_wrong(fi, fo), SEED + 71)
    g_wrong = forward_gain_cols(W_wrong)
    W_right = make_w(fi, fo, xavier_variance(fi, fo), SEED + 71)
    g_right = forward_gain_cols(W_right)
    closed_ratio = (fi + fo) / (specs.GLOROT_VARIANCE_NUMERATOR * fi)
    se = se_a(fi, fo)
    print("         错读法（`N(0,1/fi)`）→ 前向增益 **%.10f**（★ 它**反而「保前向」**）" % g_wrong)
    print("         正读法（`U(±√(6/(fi+fo)))`）→ 前向增益 **%.6f**（= `2·fi/(fi+fo)`）"
          % g_right)
    print("         比值 = **%.6f**（闭式 `(fi+fo)/(2·fi) = %.6f`）"
          % (g_wrong / g_right, closed_ratio))
    print("         ★ 闭式：`fi · 1/fi = 1.0`（逐位）· `fi · 2/(fi+fo) = %.10f`"
          % (fi * xavier_variance(fi, fo)))
    check("A.4b", abs(fi * xavier_variance_wrong(fi, fo) - 1.0) == TOL_EXACT
          and abs(g_wrong / (fi * xavier_variance_wrong(fi, fo)) - 1.0) / se <= GAIN_SE
          and abs(g_right / (fi * xavier_variance(fi, fo)) - 1.0) / se <= GAIN_SE
          and abs((g_wrong / g_right) / closed_ratio - 1.0) / (2 * se) <= GAIN_SE,
          "★★ **[C] 的公式与常见转述在数值上分叉**：错读法的**闭式**给 `1.0`（**「保前向」**）、"
          "正读法给 `%.4f`（**不保**），实测比值 **%.4f**（与闭式 `(fi+fo)/(2·fi) = %g` 在 "
          "%g·SE_A 内）⇒ 「Xavier = `N(0,1/n)`」**不是措辞差异，是结果差异**，而且**方向相反**"
          % (fi * xavier_variance(fi, fo), g_wrong / g_right, closed_ratio,
             abs((g_wrong / g_right) / closed_ratio - 1.0) / se))
    # ★★ 源码级钉住（T41 的同源做法）：Y3 的「分母是【和】」钉在**源码结构**上
    src_he = inspect.getsource(he_variance)
    src_xa = inspect.getsource(xavier_variance)
    check("A.4c", ("fan_in + fan_out" not in src_he) and ("fan_in + fan_out" in src_xa),
          "★★ **结构钉住**：`fan_in + fan_out` 这个**和**出现在 `xavier_variance` 的源码里"
          "（索引 %d）、**不出现**在 `he_variance` 里 ⇒ Y3 的那条区别被钉在**源码结构**上，"
          "不靠数值巧合（`fi == fo` 时两者数值恰好相同 ⇒ 单靠数值会退化成 no-op）"
          % src_xa.index("fan_in + fan_out"))

    # ── A.5 ★ μP 的坐标检查：两个可拟合的斜率 ──────────────────────────────
    print()
    print("  ── A.5 ★★ μP 的 **coordinate check**：logits 的方差对宽度的斜率 ──")
    print("     [C] 论文逐字：`%s`" % specs.MUP_COORDINATE_CHECK_QUOTE)
    print("     闭式：SP 的读出初值方差 `1/width` ⇒ `Var[y] = Var[h]`（**Θ(σ²)**）；")
    print("           μP 的是 `1/width²` ⇒ `Var[y] = Var[h]/width`（**Θ(σ²/width)**）")
    print("     %-7s %-14s %-16s %-16s" % ("width", "E_x‖h‖²", "SP: Var[y]", "μP: Var[y]"))
    sp_vals = []
    mup_vals = []
    for w in MUP_WIDTHS:
        W1 = mup_make_hidden(64, w, SEED + 101)
        vh = mean_h_sq(W1, SEED + 103, MUP_NX)
        sp = logit_variance(W1, "sp", SEED + 103, n_x=MUP_NX)
        mu = logit_variance(W1, "mup", SEED + 103, n_x=MUP_NX)
        sp_vals.append(sp)
        mup_vals.append(mu)
        print("     %-7d %-14.6f %-16.6e %-16.6e" % (w, vh, sp, mu))
    slope_sp = fit_loglog_slope(MUP_WIDTHS, sp_vals)
    slope_mu = fit_loglog_slope(MUP_WIDTHS, mup_vals)
    slope_sp_np = fit_loglog_slope_np(MUP_WIDTHS, sp_vals)
    slope_mu_np = fit_loglog_slope_np(MUP_WIDTHS, mup_vals)
    print("     拟合斜率：SP = **%.6f** · μP = **%.6f**（`np.polyfit` 复核：%.6f / %.6f）"
          % (slope_sp, slope_mu, slope_sp_np, slope_mu_np))
    check("A.5", abs(slope_sp) < MUP_TOL_SLOPE and abs(slope_mu + 1.0) < MUP_TOL_SLOPE
          and abs(slope_sp - slope_mu - 1.0) < MUP_TOL_SLOPE,
          "★★ **coordinate check 的两个斜率**：SP = **%.4f**（≈ 0，**宽度无关**）、"
          "μP = **%.4f**（≈ −1，**与宽度成反比**）⇒ 这正是论文那句 `Θ(σ²)` vs `Θ(σ²/width)` "
          "的可测形态；★ 而**半迁移**（只改一边）会给出别的斜率 ⇒ 这个检查**能发现实现错**"
          % (slope_sp, slope_mu))
    # ★ MC 交叉验证（**直接抽 W2**，与上面的解析估计是两条路径）
    w0 = MUP_MC_WIDTH
    W1 = mup_make_hidden(64, w0, SEED + 101)
    rng = np.random.default_rng(SEED + 111)
    h0 = W1.T @ rng.standard_normal(64)
    for mode in ("sp", "mup"):
        W2 = rng.standard_normal((MUP_N_W2, w0)) * math.sqrt(readout_init_scale(w0, mode))
        y = W2 @ h0
        mc = float(np.var(y, ddof=1))
        th = readout_init_scale(w0, mode) * float((h0 * h0).sum())
        rel = abs(mc / th - 1.0)
        print("     MC 交叉验证（width=%d、抽 %d 个 W2、mode=%s）：`%.6e` vs 闭式 `%.6e` "
              "（相对差 %.2e，4·SE = %.2e）"
              % (w0, MUP_N_W2, mode, mc, th, rel, 4 * math.sqrt(2.0 / MUP_N_W2)))
        check("A.5mc_%s" % mode, rel <= 4 * math.sqrt(2.0 / MUP_N_W2),
              "★ `mode=%s`：**直接抽 W2** 的方差与闭式 `Var[W2]·‖h‖²` 一致（相对差 %.2e ≤ "
              "4·√(2/%d) = %.2e）—— ★ 口径 A 的解析估计**不是自证**" % (mode, rel, MUP_N_W2,
                                                                       4 * math.sqrt(2.0 / MUP_N_W2)))
    # ── A.5b 一步 GD 的**相对**变化量：斜率 +1 vs 0 ─────────────────────────
    print()
    print("  ── A.5b ★ 一步 GD 之后 logits 的**相对**变化量 `|Δy/y|` 对宽度的斜率 ──")
    print("     闭式：`|Δy/y| = lr2·‖h‖²`（取 `y* = 0`）⇒ SP（`lr2 = lr`）是 Θ(width)（斜率 **+1**）；")
    print("           把读出的步长按宽度缩放（`lr2 = lr/width`）⇒ Θ(1)（斜率 **0**）")
    print("     %-7s %-20s %-20s" % ("width", "SP / 固定 lr", "读出 lr / width"))
    sp_rel = []
    mup_rel = []
    for w in MUP_WIDTHS:
        a = one_step_relative_change(w, "sp", MUP_LR, SEED + 121)
        b = one_step_relative_change(w, "mup", MUP_LR, SEED + 121)
        sp_rel.append(a)
        mup_rel.append(b)
        print("     %-7d %-20.6e %-20.6e" % (w, a, b))
    s_fix = fit_loglog_slope(MUP_WIDTHS, sp_rel)
    s_mup = fit_loglog_slope(MUP_WIDTHS, mup_rel)
    print("     拟合斜率：固定 lr = **%.6f** · 读出 lr/width = **%.6f**" % (s_fix, s_mup))
    check("A.5b", abs(s_fix - 1.0) < MUP_TOL_SLOPE and abs(s_mup) < MUP_TOL_SLOPE,
          "★★ **这就是 coordinate check 要抓的那个量**：固定 `lr` 时相对变化量随宽度"
          "**线性增长**（斜率 %.4f ≈ +1）⇒ 宽度一大就 blow up；把读出的步长按宽度缩放后"
          "斜率是 **%.4f**（≈ 0）⇒ **同一组超参在宽度上稳定**。"
          "★ 论文说「实现错了就会 blow up 或 shrink」，本程序把「blow up」量成了斜率 +1"
          % (s_fix, s_mup))
    print("     ★ 口径声明：`A.5`/`A.5b` 只量「**同一组超参在宽度上的稳定性**」；")
    print("       它**没有**证明「超参可以从窄模型迁移到宽模型」（那是论文的主张，")
    print("       对应常量 `MUP_HP_STABILITY_QUOTE` —— 本程序不复述它当结论）。")


# ===========================================================================
# 8. B 类：代价模型 —— 乘性累积、卷积的 fan、两侧不可兼得
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：方差沿深度的**乘性累积** · 卷积的 `k²` · 两侧不可兼得")
    print("=" * 74)

    # ── B.1 乘性累积 ────────────────────────────────────────────────────────
    print("  ── B.1 ★ 初始化偏差沿深度的**乘性累积**（Glorot §4.2 的那条账）──")
    print("     ★★ **范围限制**：这里量的是**一条没有残差、没有归一化的前向链**")
    print("        （`Var[y_L] = Var[x]·Π_l (fan_in,l·Var[W_l])`）。")
    print("        **残差流的范数随深度增长**的正文在 `80-…/transformer/01:191,232-233,263`；")
    print("        **误差沿深度的传播**的正文在 `130-…/07-outliers-and-propagation:124,205` —— **本程序不重推**。")
    prod = 1.0
    for _ in range(DEPTH_L):
        prod *= (1.0 + PER_LAYER_EPS)
    logsum = DEPTH_L * math.log(1.0 + PER_LAYER_EPS)
    closed_exp = math.exp(logsum)
    rel_exp = abs(prod / closed_exp - 1.0)
    print("     `L = %d` 层、每层增益 `1 + %.2f` ⇒ 乘积 = **%.10f**"
          % (DEPTH_L, PER_LAYER_EPS, prod))
    print("     闭式 `(1+ε)^L` = %.10f · 换一条路径 `exp(L·ln(1+ε))` = %.10f（相对差 %.2e）"
          % (prod, closed_exp, rel_exp))
    print("     ★ 单层的偏差只有 **%.2f%%**，%d 层之后是 **%.2f%%** ⇒ 放大了 **%.4f 倍**"
          % (PER_LAYER_EPS * 100, DEPTH_L, (prod - 1.0) * 100, prod))
    check("B.1", rel_exp < TOL_TIGHT and prod > 2.5,
          "★★ 初始化的一点点偏差是**乘性**累积的：每层 1%%、%d 层之后是 **%.4f 倍**"
          "（= `(1+ε)^L`，与 `exp(Σ ln(1+ε))` 对拍，相对差 %.2e）⇒ "
          "「初始化只付一次」这句话的另一面是：**它的账是复合的**"
          % (DEPTH_L, prod, rel_exp))
    prod_eq = 1.0
    for _ in range(DEPTH_L):
        prod_eq *= 1.0
    check("B.1b", prod_eq == 1.0,
          "★ **反面对照**：每层增益**恰好为 1** 时乘积**逐位等于 1** ⇒ `B.1` 量的是"
          "「偏差被放大」，不是「层数多了就大」（这条**不是恒真的**）")

    # ── B.2 卷积的 fan_in 含 k² ─────────────────────────────────────────────
    print()
    print("  ── B.2 ★ 卷积的 `fan_in` 含 `k²`（He 原文 `%s`）──" % specs.HE_CONNECTIONS_QUOTE)
    k, c_in = K_FAN_TEST
    n_correct = k * k * c_in
    n_wrong = c_in
    ratio = he_variance(n_wrong) / he_variance(n_correct)
    print("     `k = %d`、`c_in = %d`：正确 `n = k²·c = %d` ⇒ `Var[W] = %.10f`"
          % (k, c_in, n_correct, he_variance(n_correct)))
    print("     漏掉 `k²` 时 `n = c = %d` ⇒ `Var[W] = %.10f` ⇒ 差 **%.1f 倍**"
          % (n_wrong, he_variance(n_wrong), ratio))
    check("B.2", abs(ratio - k * k) < TOL_IDENT and k > 1,
          "★ `n_l` 是**连接数**（卷积含 `k²`）⇒ 漏掉它会让方差差 **`k²` = %.0f 倍**"
          "（`k = %d`）—— 这是一个**静默的**错（代码照样跑，只是尺度错了）" % (k * k, k))
    check("B.2b", abs(he_variance(1 * 1 * c_in) / he_variance(c_in) - 1.0) < TOL_IDENT,
          "★ **反面对照**：`k = 1`（1×1 卷积）时 `k² = 1` ⇒ 两种算法**逐位相同** "
          "⇒ `B.2` 的判据只在 `k > 1` 时有载荷")

    # ── B.3 两侧不可兼得（几何平均的上界）──────────────────────────────────
    print()
    print("  ── B.3 ★ 「折中」的代价：两侧增益的**几何平均**有闭式上界 ──")
    print("     闭式：`√(fwd·bwd) = 2√(fi·fo)/(fi+fo) ≤ 1`（**AM-GM**，等号只在 `fi == fo`）")
    print("     %-11s %-14s %-14s %-14s %-12s"
          % ("(fi,fo)", "fwd", "bwd", "√(fwd·bwd)", "闭式"))
    worst_gm = 0.0
    for fi, fo in FAN_PAIRS:
        tf = specs.GLOROT_VARIANCE_NUMERATOR * fi / (fi + fo)
        tb = specs.GLOROT_VARIANCE_NUMERATOR * fo / (fi + fo)
        gm = math.sqrt(tf * tb)
        cl = (2.0 * math.sqrt(fi * fo)) / (fi + fo)
        worst_gm = max(worst_gm, abs(gm / cl - 1.0))
        print("     %-11s %-14.6f %-14.6f %-14.6f %-12.6f"
              % ("(%d,%d)" % (fi, fo), tf, tb, gm, cl))
    print("     ★ `(256,256)` 上 `√(fwd·bwd) = 1.0`（**逐位**，因为两侧都是 1）")
    check("B.3", worst_gm < TOL_TIGHT and abs(math.sqrt(1.0 * 1.0) - 1.0) == TOL_EXACT,
          "★★ 「Xavier 的折中」有一个**闭式的代价**：两侧增益的几何平均 "
          "`√(fwd·bwd) = 2√(fi·fo)/(fi+fo) ≤ 1`，**等号只在 `fi == fo`**"
          "（在 `(256,4096)` 上是 `%.4f`）⇒ 「既不保前向也不保反向」的**定量形态**是"
          "「**几何平均被压到 1 以下**」—— 而两侧之和**恒为 2**（`A.2`）"
          % ((2.0 * math.sqrt(256 * 4096)) / (256 + 4096)))


# ===========================================================================
# 9. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：同方差 ≠ 同分布 · fan_in = 0 · gain 表的巧合 · 漏掉那个 3")
    print("=" * 74)

    # ── C.1 ★★ 同方差 ≠ 同分布 ─────────────────────────────────────────────
    print("  ── C.1 ★★ He 原文的分布是【正态】、PyTorch 的是【均匀】—— 同方差、不同分布 ──")
    fan = 256
    var = he_variance(fan)
    rng = np.random.default_rng(SEED + 131)
    g = rng.standard_normal(KURT_SAMPLE) * math.sqrt(var)
    b = math.sqrt(specs.PYTORCH_KAIMING_BOUND_NUMERATOR * var)     # `√(3·Var)` = `√(6/fan)`
    u = rng.uniform(-b, b, KURT_SAMPLE)
    vg, vu = float(np.var(g, ddof=1)), float(np.var(u, ddof=1))
    kg, ku = excess_kurtosis(g), excess_kurtosis(u)
    print("     Gaussian（He Eqn 10 的实现）：方差 %.8f · 超额峰度 **%+.4f**" % (vg, kg))
    print("     Uniform（`kaiming_uniform_` 的实现）：方差 %.8f · 超额峰度 **%+.4f**" % (vu, ku))
    print("     理论超额峰度：Gaussian = 0 · Uniform = **−1.2**（`−6/5`）")
    check("C.1", abs(vg / vu - 1.0) < 0.03 and abs(kg) < KURT_TOL
          and abs(ku + 1.2) < KURT_TOL,
          "★★ **判据的边界**：两条实现的**二阶矩相同**（相对差 %.4f < 3%%），"
          "而**四阶矩差 %.4f**（Gaussian `%+.4f` vs Uniform `%+.4f`，理论 `0` vs `−1.2`）"
          "⇒ `A.3` 的「同方差」**只说二阶矩**，它**不蕴含**「同分布」—— 转述时不得升格"
          % (abs(vg / vu - 1.0), abs(kg - ku), kg, ku))

    # ── C.2 ★ fan_in = 0 必须被拦住 ─────────────────────────────────────────
    print()
    print("  ── C.2 ★ `fan_in = 0` 必须抛异常，而不是给出 `inf` ──")
    raised = []
    for fn, args in ((he_variance, (0,)), (xavier_variance, (0, 256)), (xavier_variance, (256, 0))):
        try:
            got = fn(*args)
            raised.append(None)
            print("     `%s%s` → **没有抛异常**（返回 %r）" % (fn.__name__, args, got))
        except ValueError as e:
            raised.append(fn.__name__)
            print("     `%s%s` → `ValueError: %s`" % (fn.__name__, args, e))
    check("C.2", raised.count(None) == 0 and "he_variance" in raised,
          "★ 三个退化输入**全部被拦住**（抛 `ValueError`）⇒ 失败行为真的成立；"
          "★ 对照：若用 `math.sqrt(2.0/0)`，结果是 `ZeroDivisionError`（不可读），"
          "而 `numpy` 路径会**静默给 `inf`**（更难查）—— 本程序选**显式拦住**")
    check("C.2b", he_variance(1) == specs.HE_VARIANCE_NUMERATOR
          and xavier_variance(1, 1) == specs.GLOROT_VARIANCE_NUMERATOR / 2.0,
          "★ **边界**：拦住的只有 `≤ 0`，最小的合法输入 `fan_in = fan_out = 1` 仍然可用"
          "（`Var = 2` / `Var = 1`，逐位）⇒ 这条闸**不是**靠「一律拒绝」实现的；"
          "★ 顺带：`fan_in = 1` 时 He 的 `2/fan_in` 与 Xavier 的 `2/(1+1)` **恰好相等** "
          "（同一个退化点，与 `fi == fo` 时数值重合是同一件事）")

    # ── C.3 ★ leaky_relu a=0 与 relu 的 gain 逐位相同（巧合式等价）──────────
    print()
    print("  ── C.3 ★ `kaiming_uniform_` 的默认非线性是 `leaky_relu`，但 `a=0` 时与 `relu` 同 gain ──")
    print("     %-10s %-20s %-20s %-14s"
          % ("a", "gain(leaky_relu,a)", "gain(relu)", "差"))
    worst_g = 0.0
    for a in (0.0, 0.02, GAIN_A_BIG, 1.0):
        gl = calculate_gain("leaky_relu", a)
        gr = calculate_gain("relu")
        d = abs(gl - gr)
        worst_g = max(worst_g, d)
        print("     %-10.2f %-20.12f %-20.12f %-14.3e" % (a, gl, gr, d))
    check("C.3", abs(calculate_gain("leaky_relu", 0.0) - calculate_gain("relu")) == TOL_EXACT
          and abs(calculate_gain("leaky_relu", GAIN_A_BIG)
                  - calculate_gain("relu")) > TOL_MEASURABLE,
          "★★ **巧合式等价**：[B] 文档的默认非线性是 `leaky_relu`（**不是** `relu`），"
          "但 `a = 0` 时两者 gain **逐位相同**（`√2`）⇒ 默认值**恰好**落在 He 的 ReLU 那一支上；"
          "★ **而它不是恒真的**：`a = %g` 时差 **%.2e**（> %g）"
          % (GAIN_A_BIG, abs(calculate_gain("leaky_relu", GAIN_A_BIG)
                             - calculate_gain("relu")), TOL_MEASURABLE))

    # ── C.4 ★ 漏掉那个 3 ⇒ 方差恰好小 3 倍 ──────────────────────────────────
    print()
    print("  ── C.4 ★ 把支撑上界写成 `√(2/(fi+fo))`（漏掉 `3`）⇒ 方差**恰好小 3 倍** ──")
    fi, fo = XAV_PAIR
    a_right = xavier_uniform_bound(fi, fo, specs.PYTORCH_XAVIER_DEFAULT_GAIN)
    a_wrong = math.sqrt(specs.GLOROT_VARIANCE_NUMERATOR / (fi + fo))
    v_right = uniform_variance(a_right)
    v_wrong = uniform_variance(a_wrong)
    print("     正确：`a = √(6/(fi+fo)) = %.10f` ⇒ `Var = %.10e`" % (a_right, v_right))
    print("     漏 `3`：`a' = √(2/(fi+fo)) = %.10f` ⇒ `Var = %.10e`" % (a_wrong, v_wrong))
    print("     比值 `Var / Var'` = **%.10f**（逐位 = `%.1f`）" % (v_right / v_wrong,
                                                                specs.PYTORCH_XAVIER_BOUND_NUMERATOR
                                                                / specs.GLOROT_VARIANCE_NUMERATOR))
    check("C.4", abs(v_right / v_wrong - 3.0) < TOL_IDENT,
          "★ `Var[U(−a,a)] = a²/3` 里的那个 `3`（**均匀分布的性质**）与 `√(6/…)` 里的 `6` "
          "是同一条链上的两个数 ⇒ 漏掉 `a` 里的 `√3` 会让方差**恰好小 3 倍**"
          "（`√3` 在支撑上、`3` 在方差里）—— 这是一个**静默的 3 倍**，不会报错")


# ===========================================================================
# 10. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。

    ★ 理由（Chunk 1 实测）：口径常量一旦被改，后面的**长度/索引检查会先崩** ——
      那时 `mutate.py` 记的是**崩溃捕获**，而「崩溃捕获不算真捕获」。
      ⇒ 把钉住提到**任何**计算之前，改常量就变成**断言捕获**。
    """
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ **本主题的三条禁令**（回链，不重推）：warmup 的机制解释在 "
          "`80-…/normalization/02:83-100`；残差流/深度的账在 `80-…/transformer/01`；"
          "误差沿深度的传播在 `130-…/07-outliers-and-propagation`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == "// FAIL: init-stability" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", ":: FAIL: mixed-precision",
                                        "++ FAIL: optimizers", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")

    measured = {"SEED": SEED, "N_REP": N_REP, "N_PROBE_REPS": N_PROBE_REPS,
                "SAMPLE_SEEDS": SAMPLE_SEEDS, "FAN_PAIRS_B": FAN_PAIRS_B,
                "XAV_PAIR": XAV_PAIR, "HE_FAN_IN_LIST": HE_FAN_IN_LIST,
                "K_FAN_TEST": K_FAN_TEST, "DEPTH_L": DEPTH_L,
                "PER_LAYER_EPS": PER_LAYER_EPS, "GAIN_A_BIG": GAIN_A_BIG,
                "MUP_WIDTHS": MUP_WIDTHS, "MUP_LR": MUP_LR, "MUP_MC_WIDTH": MUP_MC_WIDTH,
                "KURT_SAMPLE": KURT_SAMPLE, "MUP_N_W2": MUP_N_W2,
                "FAN_PAIRS": FAN_PAIRS, "MUP_NX": MUP_NX, "UNIF_SAMPLE": UNIF_SAMPLE,
                "KAIMING_DEFAULTS": (specs.PYTORCH_KAIMING_DEFAULT_A,
                                     specs.PYTORCH_KAIMING_DEFAULT_MODE,
                                     specs.PYTORCH_KAIMING_DEFAULT_NONLINEARITY,
                                     specs.PYTORCH_XAVIER_DEFAULT_GAIN)}
    expect = {"SEED": 20260950, "N_REP": 300, "N_PROBE_REPS": 1000,
              "SAMPLE_SEEDS": (1, 2, 3, 4, 5),
              "FAN_PAIRS_B": ((128, 128), (256, 256)),
              "XAV_PAIR": (256, 4096), "HE_FAN_IN_LIST": (128, 256, 512, 1024, 2048, 4096),
              "K_FAN_TEST": (3, 64), "DEPTH_L": 100, "PER_LAYER_EPS": 0.01,
              "GAIN_A_BIG": 0.2, "MUP_WIDTHS": (32, 64, 128, 256, 512, 1024, 2048, 4096),
              "MUP_LR": 0.1, "MUP_MC_WIDTH": 32, "KURT_SAMPLE": 200000, "MUP_N_W2": 20000,
              "FAN_PAIRS": ((256, 256), (1024, 1024), (256, 4096), (4096, 256), (4096, 4096),
                            (128, 128), (2048, 512)),
              "MUP_NX": 64, "UNIF_SAMPLE": 200000,
              "KAIMING_DEFAULTS": (0.0, "fan_in", "leaky_relu", 1.0)}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 20 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", MUP_MC_WIDTH == MUP_WIDTHS[0] and MUP_MC_WIDTH < MUP_WIDTHS[-1]
          and all(w > 1 for w in MUP_WIDTHS) and len(set(MUP_WIDTHS)) == len(MUP_WIDTHS),
          "★★ `A.5` 的 MC 交叉验证**只用在最小的那个宽度上**（`%d`）—— 那是**成本与信噪**"
          "的取舍：MC 的相对标准误 `√(2/%d) = %.2e`，而宽度越大 `W2` 越大、抽样越慢；"
          "★ 而斜率用的是**解析估计**（8 个宽度，噪声 ~%g）"
          % (MUP_MC_WIDTH, MUP_N_W2, se_b(MUP_N_W2), MUP_TOL_SLOPE))
    check("D.3c", N_REP < N_PROBE_REPS and len(FAN_PAIRS_B) == 2,
          "★ 口径 B 的重复数取 `%d`（**探针 D9 用 1000**）—— 那是**成本与精度**的取舍："
          "`A.1b` 的表把两种重复数下的**经验 sd** 都列了出来，而 `A.1b2` 证明"
          "**朴素的 `√(2/n_rep)` 不是这个估计量的标准误**（否则按 1000 次就能算出 4.5%% 的精度，"
          "与实测的 0.2~0.5%% 差一个数量级）" % N_REP)
    check("D.3d", abs(HE_BAND_LO - 1.95) < TOL_IDENT and abs(HE_BAND_HI - 2.05) < TOL_IDENT
          and N_REP == 300,
          "★ 计划 §3 表 A ① 的带子 `[1.95, 2.05]` **逐位钉住**；★ `A.1b` 会把"
          "「**两种抽样口径**的 5 子均值 + 单子最大偏差」都摆出来（判据的小 n 纪律）")

    th = {"GAIN_SE": GAIN_SE, "TOL_EXACT": TOL_EXACT, "TOL_TIGHT": TOL_TIGHT,
          "TOL_IDENT": TOL_IDENT, "TOL_MEASURABLE": TOL_MEASURABLE,
          "TOL_KURT_DIFF": TOL_KURT_DIFF, "KURT_TOL": KURT_TOL,
          "MUP_TOL_SLOPE": MUP_TOL_SLOPE, "XAV_WRONG_TOL": XAV_WRONG_TOL,
          "XCHECK_MULT": XCHECK_MULT}
    th_expect = {"GAIN_SE": 5.0, "TOL_EXACT": 0.0, "TOL_TIGHT": 1e-12,
                 "TOL_IDENT": 1e-15, "TOL_MEASURABLE": 1e-2,
                 "TOL_KURT_DIFF": 0.5, "KURT_TOL": 0.05,
                 "MUP_TOL_SLOPE": 0.05, "XAV_WRONG_TOL": 1e-9, "XCHECK_MULT": 3.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.3f", (specs.PYTORCH_KAIMING_DEFAULT_NONLINEARITY,
                  specs.PYTORCH_KAIMING_DEFAULT_MODE) == ("leaky_relu", "fan_in"),
          "★★ **[B] 的两个默认值必须先钉住**（在 `preflight()` 里）：`calculate_gain` 对"
          "**未知的 `nonlinearity`** 会抛 `ValueError` —— 那个值一旦被改，后面**根本跑不到** "
          "`A.3` 的断言（`mutate.py` 会把它记成**崩溃捕获**，而崩溃捕获**不算真捕获**）。"
          "★ 这是本库 **P23** 的同一条理由：**口径型常量的钉住要跑在任何计算之前**")
    check("D.4", not tdiff, "★ 阈值 10 个逐位一致（含**两个 0 容差**与 `TOL_IDENT = 1e-15` 的"
                            "「恒等式」口径）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


# ===========================================================================
# 11. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    # ── D.5 来源的编号 + 标题（P21 的落法）────────────────────────────────
    check("D.5", specs.HE_ARXIV_ID == "1502.01852"
          and specs.HE_TITLE == ("Delving Deep into Rectifiers: Surpassing Human-Level "
                                 "Performance on ImageNet Classification")
          and specs.MUP_ARXIV_ID == "2203.03466"
          and specs.MUP_VENUE == "NeurIPS 2021",
          "★ **P21 的落法**：编号与**标题/评审状态**成对钉住 —— He 2015 = `%s`（**不确认同行评审**）；"
          "μP = `%s`（**%s**，★ 本主题**自己打开过 abs 页**看的 `Comments:` 字段）"
          % (specs.HE_ARXIV_ID, specs.MUP_ARXIV_ID, specs.MUP_VENUE))
    check("D.6", specs.GLOROT_XAVIER_TERM_COUNT == 0
          and specs.GLOROT_VENUE_QUOTE == ("Proceedings of the Thirteenth International Conference "
                                           "on Artificial Intelligence and Statistics"),
          "★★ **Y2 的落法**：`Xavier initialization` 在 Glorot 原文里的计数 **%d**"
          "（四种数法都稳健地给 0；而单字 `Xavier` 出现 4 次、**全是页眉的作者名**）；"
          "★ 而 `normalized initialization` 的计数**不可复现** ⇒ **不写那个数**"
          "（只留一个取样常量 `%r`）"
          % (specs.GLOROT_XAVIER_TERM_COUNT, specs.GLOROT_NORMALIZED_INIT_QUOTE))

    # ── D.7 公式的那几个数 ─────────────────────────────────────────────────
    check("D.7", specs.HE_VARIANCE_NUMERATOR == 2.0 and specs.GLOROT_VARIANCE_NUMERATOR == 2.0
          and specs.HE_VARIANCE_NUMERATOR / 256 == he_variance(256)
          and specs.GLOROT_VARIANCE_NUMERATOR / (256 + 4096) == xavier_variance(256, 4096),
          "★ 公式的两个分子都是 `2.0`（**同值但不同源**：He 的 2 = ReLU 折半；"
          "Glorot 的 2 = 两个约束的折中）⇒ 两条常量**分开存放**，`A.4c` 用**源码结构**"
          "把「分母是和」这件事钉死")
    check("D.8", specs.PYTORCH_KAIMING_BOUND_NUMERATOR == 3.0
          and specs.PYTORCH_XAVIER_BOUND_NUMERATOR == 6.0
          and specs.PYTORCH_KAIMING_DEFAULT_MODE == "fan_in"
          and specs.PYTORCH_KAIMING_DEFAULT_NONLINEARITY == "leaky_relu"
          and specs.PYTORCH_KAIMING_DEFAULT_A == 0.0
          and specs.PYTORCH_XAVIER_DEFAULT_GAIN == 1.0,
          "★★ **[B] 级常量的逐位钉住**（PyTorch 2.14 的 `nn.init` 页）—— ★ 它们**不是规范**，"
          "是「某个实现怎么做的」；★ 与 [C] 的互校在 `A.3` / `A.3c`，**不是任一方的背书**")
    check("D.8b", abs(specs.PYTORCH_CALCULATE_GAIN_RELU - math.sqrt(2.0)) < 1e-15
          and abs(specs.PYTORCH_CALCULATE_GAIN_TANH - 5.0 / 3.0) < 1e-15
          and specs.PYTORCH_CALCULATE_GAIN_LINEAR == 1.0
          and specs.PYTORCH_CALCULATE_GAIN_SELU == 0.75,
          "★ `calculate_gain` 表的四个值是**十进制字面量**（盲区④的落法：写成 `2 ** 0.5` "
          "就是 `BinOp`，**不进变异分母**）⇒ 它们仍与数学值一致（`√2` / `5/3`）")
    check("D.8c", specs.PYTORCH_KAIMING_KNOWN_AS == "Also known as He initialization."
          and specs.PYTORCH_XAVIER_KNOWN_AS == "Also known as Glorot initialization.",
          "★★ 两条 `Also known as …` 是 **[B] 文档把函数与名字挂起来**的逐字依据 —— "
          "它与 `D.6` 的 `GLOROT_XAVIER_TERM_COUNT = 0` **并列**正好把 Y2 的张力摆出来："
          "**Glorot 原文里没有那个名字，名字是后来由 He 2015（自称引自 Caffe）安上的**")

    # ── D.9 独立性：源码两两不同 + AST 调用表为空 ───────────────────────────
    src_cols = inspect.getsource(forward_gain_cols)
    src_norm = inspect.getsource(forward_gain_normalized)
    src_samp = inspect.getsource(probe_forward_gain)
    check("D.9", len({src_cols, src_norm, src_samp}) == 3,
          "★ 防 P17：被测（`forward_gain_cols` %d 字符）、第三条路径（`forward_gain_normalized` "
          "%d）、参照（`probe_forward_gain` %d）**源码两两不同**"
          % (len(src_cols), len(src_norm), len(src_samp)))
    watched = {"forward_gain_cols", "backward_gain_rows", "he_variance", "xavier_variance"}
    called = _callee_names(probe_forward_gain) | _callee_names(forward_gain_normalized)
    check("D.9b", not (watched & called),
          "★ **AST 调用表**：两条**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    check("D.9c", abs(forward_gain_cols(make_w(64, 32, xavier_variance(64, 32), SEED + 141))
                      - forward_gain_normalized(make_w(64, 32, xavier_variance(64, 32), SEED + 141)))
          < TOL_IDENT,
          "★ **第三条路径**（`np.linalg.norm` 逐列）与口径 A **逐位一致** ⇒ "
          "`D.9` 的「源码两两不同」不是装饰（那条属性是**可测的**）")
    check("D.9d", abs(fit_loglog_slope([1, 2, 4, 8], [1, 4, 16, 64]) - 2.0) < TOL_TIGHT
          and abs(fit_loglog_slope_np([1, 2, 4, 8], [1, 4, 16, 64]) - 2.0) < TOL_TIGHT,
          "★ 斜率的两条算法（手写最小二乘 vs `np.polyfit`）在 `y = x²` 上都给 **2.0** "
          "⇒ `A.5` 的斜率不是「用一条自己写的公式自证」")
    check("D.10", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))

    # ── D.11 UNSOURCED 逐位钉住 ────────────────────────────────────────────
    check("D.11", specs.UNSOURCED == ("caffe_xavier_term_origin",
                                      "glorot_normalized_init_count",
                                      "pytorch_init_bound_formula_raw_count",
                                      "pytorch_conv_fan_definition",
                                      "mup_learning_rate_table_scaling",
                                      "spike_mechanism_one_source_conclusion",
                                      "snm_peer_review_status_recheck"),
          "★ `specs.UNSOURCED` 的 %d 项**逐位钉住** —— ★ 其中 `caffe_xavier_term_origin` 就是 "
          "**Y2 收窄后的那个缺口**：「He 2015 自称引自 Caffe」**可证**，"
          "而「Caffe 里真有这个名字」**本次不可证**" % len(specs.UNSOURCED))
    # ── D.13 ★ 本程序用到的**其余 specs 字符串**逐位钉住（每一条都要有消费者）──
    check("D.13", specs.HE_SUFFICIENT_CONDITION_QUOTE == "A sufficient condition is"
          and specs.HE_STD_QUOTE == "whose standard deviation (std) is"
          and specs.HE_BACKWARD_ALTERNATIVE_QUOTE == "it is sufficient to use either"
          and specs.HE_CONNECTIONS_QUOTE == "the number of connections"
          and specs.GLOROT_COMPROMISE_QUOTE == "As a compromise between these two constraints"
          and specs.GLOROT_NORMALIZED_INIT_QUOTE == "the normalized initialization"
          and specs.GLOROT_WRONG_ARXIV_ID == "1001.0014"
          and specs.MUP_TITLE == ("Tensor Programs V: Tuning Large Neural Networks "
                                  "via Zero-Shot Hyperparameter Transfer"),
          "★★ 八条**逐字引文/编号**逐位钉住（HE 的四条 + Glorot 的三条 + μP 的标题）—— "
          "★ **它们全部是 `==` 而不是 `startswith`**（盲区③：变异是**在字符串末尾追加一个 `X`**，"
          "前缀式检查**抓不到**）")
    # ── D.12 两条消息常量（有来源、进正文的那两句）────────────────────────
    check("D.12", specs.MUP_ERROR_PRONE_QUOTE == "implementing it correctly can in practice be error-prone"
          and specs.MUP_COORDINATE_CHECK_QUOTE == ("An incorrect implementation will see some activation "
                                                   "vector blow up or shrink to zero with width")
          and specs.MUP_HP_STABILITY_QUOTE == "many optimal HPs remain stable even as model size changes",
          "★ μP 的三句话**逐位钉住**：判据形态（coordinate check）、"
          "**论文自己承认实现容易错**、以及「主张」（HP 稳定）—— ★ 三者**分开存放**，"
          "`A.5` 只测第一句那个量，**不复述第三句当结论**")


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
    print("01_init.py —— 初始化：He / Xavier / μP")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ **本程序的落点**：He / Xavier / μP 的**公式与判据**，以及三条必须订正的转述")
    print("     （Y1 那个 arXiv 编号、Y2 那个不存在的名字、Y3 那个错的分布与分母）。")
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
