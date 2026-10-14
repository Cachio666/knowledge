#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_fp8.py —— **训练循环里怎么用 fp8**：分工、per-tensor scaling、溢出的处置。

配套文档：`03-fp8-training.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的收窄原则（写死，不许越界）**：
> **只写训练循环里怎么用 fp8。位域与 scale 粒度一律回链。**

  具体地，下面这些**不在本程序、也不在本文**（计划 §2 的边界表，结构层审查 **M1**）：
   · **fp8 的位分配与四个变体**（`e4m3fn` / `e5m2` / `fnuz`、±448 vs ±57344、
     **按 torch 版本的溢出差异**）→ `130-model-optimization/quantization/01-fp8-family.md` **§0 全节**；
   · **scale 的作用域与 bit 开销的另一半**（每通道 / 每 token / 每 32 元素的块）→
     `130-model-optimization/quantization/02-e8m0-and-mx.md:276` + `130-model-optimization/quantization/06-granularity.md`；
   · **累加器宽度的误差界** → `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md`。
  ⇒ 本程序**把格式参数当输入**（`(最大有限数, 最小正规数, eps)` 三个数，取自 `specs.py` 的 [C] 常量），
    **不解释它们是怎么从 8 位里分出来的**。

★★ **库内硬要求**（`130-model-optimization/quantization/02-e8m0-and-mx.md:276` 逐字）：
> 「**凡是写「scale」这个词的地方，同时写出它的作用域（每张量 / 每通道 / 每 token /
>   每 32 个元素的块）与它的 bit 开销**」
  ⇒ 本程序**凡出现 scale 处一律写「**per-tensor（作用域 = 每张量）+ 32 bit/张量**」**，
    并在 `B.1` 里把这个开销**算出来、断言下来**。

★★ **方法学让步（必须写明）**：fp8 论文自己的训练实验是 **`simulated FP8`** —— 逐字
> `Training experiments were carried out with simulated FP8 - tensor values were clipped to
>  only those that could be represented in FP8` 与 `Arithmetic was carried out using the wider
>  representation`。
  本程序是**同一个形态**：**算术全在 fp64/fp32 上做，只有「存储/输入」被量化**。
  ⇒ 因此它**不得**支撑任何关于「真 fp8 算术」（吞吐、累加顺序、Tensor Core）的结论。

★★ 本程序的数字全部是推导值，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  前向张量用 `e4m3fn`、反向张量用 `e5m2` 的**分工表**逐位对上 [C] 原文的措辞；
  · `A.3`  量化器的最大相对误差 **≤ `eps/2`**（RNE）—— 把舍入改成截断立刻失败；
  · `B.2`  delayed scaling 的溢出率与闭式 **`1/(k+1)`** 对上（`k` 是历史长度）；
  · `C.1`  把 scale **限制成只能取 2 的幂**时，表示范围**真的**只用掉一半；
  · `C.2`  一个**突增**的批在 delayed scale 下**真的**饱和；而「跳步」策略**真的**把整步更新丢掉。

用法：
    python3 03_fp8.py
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
# ★★ 失败标记：本主题（mixed-precision）取 `:: FAIL: mixed-precision`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ":: FAIL: mixed-precision"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", "++ FAIL: optimizers",
                     "// FAIL: init-stability", ";; FAIL: checkpointing")

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
        print("  [ok]   %-5s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-5s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260927              # ★ 与 `01`（20260925）、`02`（20260926）都**不同**
D_IN = 32                    # 输入宽度
M_HID = 16                   # 隐层宽度
N_GRAD = 4096                # 量化器扫描用的元素数
# ★★ 分工（**值取自 `specs.py`**，来源是 [C] 2209.05433 的逐字）：
FORMAT_FORWARD = "e4m3fn"    # 前向：权重与激活
FORMAT_BACKWARD = "e5m2"     # 反向：梯度
# ★ scale 的**作用域与 bit 开销**（库内硬要求，见文件头）：
SCALE_SCOPE = "per-tensor"   # 作用域 = **每张量**
SCALE_BITS = 32              # 开销 = 每张量 1 个 fp32
# ★ 只允许 2 的幂的 scale 时用的小数因子（C.1）：
REAL_SCALE_AIM = 31.99       # ⇒ `2^floor(log2(31.99)) = 16`，浪费 `1 − 16/31.99 = 49.98%`
N_SMALL = 4096               # C.1 的张量元素数
N_LARGE = 4096000            # B.1 的「×1000」张量元素数
SPIKE_FACTOR = 25.0          # C.2 的突增倍数
K_DELAYED = 8                # B.2/C.2 的 delayed scaling 历史长度（★ 本程序的选择，不是引文）
K_SWEEP = (1, 2, 4, 8, 16)   # B.2 的历史长度扫描表
N_AMAX = 100000              # B.2 的蒙特卡洛样本数
TRAIN_STEPS = 5              # A.4 的训练步数（只要够验证「哪一步用哪个编码」，不求收敛）

TOL_CITED = 0.0              # A.1/A.2：逐字比对是**逐位**的
TOL_Q_REL = 0.0              # A.3：`max rel err <= eps/2` 是**硬上界**（RNE 的定义）
TOL_AMAX_EXACT = 1e-12       # A.2b：`amax·s` 与 `max_finite` 的相对差（实数 scale 是**任意实数**）
TOL_IDENTITY = 0.0           # A.4：恒等量化下与 fp32 参照**逐位相同**
TOL_OVERFLOW_LAW = 0.02      # B.2：MC 与闭式 `1/(k+1)` 的相对差
MIN_HITS_AMAX = 200          # B.2 的小 n 门槛（★ P22 的落法：先打印参照量级）
TOL_RANGE_LOSS = 0.55        # C.1：2 的幂 scale 下「用到的范围比例」必须低于这个上界
TOL_SAT_MIN = 100            # C.2：突增批在 delayed scale 下的饱和元素数下界


# ===========================================================================
# 2. 格式参数：**当作输入**（不推导它是怎么从 8 位里分出来的）
# ===========================================================================
def fmt_params(fmt):
    """`fmt` -> `(最大有限数, 最小正规数, eps)`。**值来自 `specs.py`（[C] 常量）**。

    ★ 本函数**不**解释位分配 —— 「这 8 位怎么分、四个变体差在哪」的正文在
      `130-model-optimization/quantization/01-fp8-family.md` §0，**此处回链**。
    ★ 本函数也**不**给出尾数位数：`eps` 是那个信息的**唯一**载体（`eps = 2^-M`），
      而 `M` 是位域的事。
    """
    return specs.FP8_FORMATS[fmt]


# ===========================================================================
# 3. 量化器（被测侧）：饱和 + 舍入到最近（RNE），**不做位运算**
# ===========================================================================
def fp8_round(x, fmt):
    """把一个实数张量舍到 `fmt` 的可表示网格上，**溢出饱和到最大有限数**。

    ```
    step(mag) = max( 2^(e − M),  最小正规数 · 2^−M )      e = floor(log2(mag))，M = −log2(eps)
    q         = min( rint(mag/step) · step, 最大有限数 )
    ```

    ★★ **饱和这一步的依据是 [C] 原文逐字**：
      `Values that overflow are then saturated to the maximum representable value.`
      ⇒ 本函数**只做饱和**，**不对任何「格式的溢出语义是什么」下结论** ——
        格式层面的溢出（NaN / `inf` / **按 torch 版本不同**）是位分配那一层的事，
        正文在 `130-model-optimization/quantization/01-fp8-family.md` §0/§3，**此处回链**。
    ★ `min(..., 最大有限数)` **必须放在舍入之后**：不这样的话 `448` 附近的量会先被
      舍到 `480`（那个码字在 `e4m3fn` 里被 NaN 占掉，是位分配的内容）再被夹回来。
    """
    ax = np.abs(np.asarray(x, dtype=np.float64))
    man_bits = -math.log2(fmt_params(fmt)[2])
    expo = np.floor(np.log2(np.where(ax > 0.0, ax, 1.0)))
    step = np.maximum(np.exp2(expo - man_bits),
                      fmt_params(fmt)[1] * fmt_params(fmt)[2])
    r = np.minimum(np.rint(ax / step) * step, fmt_params(fmt)[0])
    return np.sign(np.asarray(x, dtype=np.float64)) * np.where(ax == 0.0, 0.0, r)


def quantize_tensor(x, fmt, scale, mode="fp8"):
    """**per-tensor（作用域 = 每张量）scaling**：编码时乘 `scale`、解码时除回去。

    ★ 调用方要为每一次调用记一笔（`B.1` 的开销账要用它）—— 「每张量一个 scale」这件事
      之所以是一个**可数的**开销，正是因为缩放是**按张量**发生的。
    ★ `mode="identity"` 是 `A.4` 的阴性对照：**既量化也不缩放**（原样返回）。

    ★★ 本函数的**开销**：每张量 **1 个 fp32 = 32 bit**，**与元素数无关** ——
      这就是「per-tensor scaling 是软件层」在字节上的样子（`B.1` 把它算出来）。
    """
    a = np.asarray(x, dtype=np.float64)
    if mode == "identity":
        return a
    return fp8_round(a * scale, fmt) / scale


def amax(x):
    """张量的最大绝对值。★ 空张量与全零张量要能区分开（本程序不产生这两种情形）。"""
    a = np.abs(np.asarray(x, dtype=np.float64))
    return float(a.max()) if a.size else 0.0


def scale_just_in_time(x, fmt):
    """**just-in-time scaling**：由**当前**张量自己的 `amax` 定 scale。

    逐字依据（[C] **FP8-LM** 附录 A.2，见 `specs.FP8_LM_JUST_IN_TIME_QUOTE`）：
      「determining the scaling factor based on the maximum absolute value (amax) of the
        tensor being generated」
    ★ 它的**代价**（同段逐字）：`this approach is often infeasible because it necessitates
      multiple passes through the data` —— 要先算出 `amax` 才能定 scale，等于多读一遍。
    """
    return fmt_params(fmt)[0] / max(amax(x), 1e-30)


def scale_delayed(history, fmt, k):
    """**delayed scaling**：由**前 `k` 次**观测到的 `amax` 定 scale。

    逐字依据（[C] **FP8-LM** 附录 A.2，见 `specs.FP8_LM_DELAYED_QUOTE`）：
      「selecting the scaling factor based on the maximum absolute values observed in a
        certain number of preceding iterations」
    ★ 它的**代价**（同段逐字）：`necessitates the storage of a history of maximum values as
      additional parameters of the FP8 operators`。
    ★★ **这两个术语不在 fp8 论文（2209.05433）里** —— 它们在那篇里各出现 **0** 次，
      论文自己逐字说 `Details of the heuristics to select the scaling factors are
      beyond the scope of this paper`（计划 §4 的 **T17**）。本程序**只引 FP8-LM 的定义**。
    """
    if not history:
        return None
    return fmt_params(fmt)[0] / max(max(history[-k:]), 1e-30)


def n_saturated(x, fmt, scale, rel=1e-9):
    """按 `scale` 编码时会有多少个元素被**饱和**掉（`|x·scale| > 最大有限数`）。

    ★ `rel` 那个相对余量是**必需的**：`scale = 最大有限数 / amax` 是浮点除法，
      最大那个元素乘回去会落在 `最大有限数` 的**两侧**（±1 ulp）⇒ 不留余量会把它
      **误判成饱和**。本程序第一版就误判了 29 个元素（当时 `just-in-time` 的饱和数
      竟然不是 0）—— 这是「量法口径」的一个实例。
    """
    a = np.abs(np.asarray(x, dtype=np.float64)) * scale
    return int(np.count_nonzero(a > fmt_params(fmt)[0] * (1.0 + rel)))


# ===========================================================================
# 4. 参照实现：**fp32 全精度**的一步（不调用被测侧的任何函数）
# ===========================================================================
def plain_forward(W1, W2, x):
    """参照前向：`h = relu(W1 x)`、`out = W2 h`。型号与被测侧**无关**写法。"""
    z = np.asarray(W1, dtype=np.float64) @ np.asarray(x, dtype=np.float64)
    h = np.maximum(z, 0.0)
    return h, np.asarray(W2, dtype=np.float64) @ h


def plain_backward(W1, W2, x, y):
    """参照反向：**手推的闭式**（不是 `inner_step` 的复用）。

    ```
    out − y  →  dW2 = dout·h   →  dh = dout·W2   →  dz = dh·(h>0)   →  dW1 = dz ⊗ x
    ```
    """
    h, out = plain_forward(W1, W2, x)
    dout = out - np.asarray(y, dtype=np.float64)
    dW2 = dout * h
    dh = dout * np.asarray(W2, dtype=np.float64)
    dz = dh * (h > 0.0)
    dW1 = np.outer(dz, np.asarray(x, dtype=np.float64))
    return dW1, dW2


# ===========================================================================
# 5. 被测侧：一次 fp8 训练步（**前向 e4m3fn、反向 e5m2**）
# ===========================================================================
def fp8_step(W1, W2, x, y, lr, mode="fp8", grad_scale=None, encode_log=None):
    """一次训练步。**张量 → 编码**的分配就是 `A.1` 要断言的「分工」：

    | 张量 | 方向 | 编码 |
    |---|---|---|
    | `W1` / `W2` / `x` / `h` / `out` | 前向 | `e4m3fn` |
    | `dout` / `dW2` / `dh` / `dz` / `dW1` | 反向 | `e5m2` |

    ★ `encode_log` 不为 `None` 时，把 `(张量名, 编码, scale)` 逐个记进去 —— `A.1` 与 `B.1` 用它。
    ★ `grad_scale` 不为 `None` 时，**梯度张量**用它（而不是 just-in-time）—— C.2 用它造出饱和。
    ★★ **算术全在 fp64 上做**（`simulated FP8` 的形态，见文件头）。
    """
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def enc(name, t, fmt):
        s = scale_just_in_time(t, fmt)
        if encode_log is not None:
            encode_log.append((name, fmt, s))
        return quantize_tensor(t, fmt, s, mode)

    W1e = enc("W1", W1, FORMAT_FORWARD)
    W2e = enc("W2", W2, FORMAT_FORWARD)
    xe = enc("x", x, FORMAT_FORWARD)
    h = np.maximum(W1e @ xe, 0.0)
    he = enc("h", h, FORMAT_FORWARD)
    out = W2e @ he
    oute = enc("out", out, FORMAT_FORWARD)

    dout = oute - y
    if encode_log is not None:
        encode_log.append(("dout", FORMAT_BACKWARD,
                           scale_just_in_time(dout, FORMAT_BACKWARD) if grad_scale is None
                           else grad_scale))
    if grad_scale is None:
        douts = quantize_tensor(dout, FORMAT_BACKWARD,
                                scale_just_in_time(dout, FORMAT_BACKWARD), mode)
    else:
        douts = quantize_tensor(dout, FORMAT_BACKWARD, grad_scale, mode)
    dW2 = douts * he
    dh = douts * W2e
    dz = dh * (he > 0.0)
    dW1 = np.outer(dz, xe)

    for nm, t in (("dW2", dW2), ("dh", dh), ("dz", dz), ("dW1", dW1)):
        if encode_log is not None:
            encode_log.append((nm, FORMAT_BACKWARD,
                               scale_just_in_time(t, FORMAT_BACKWARD) if grad_scale is None
                               else grad_scale))

    gs = grad_scale
    dW2g = quantize_tensor(dW2, FORMAT_BACKWARD,
                           scale_just_in_time(dW2, FORMAT_BACKWARD) if gs is None else gs, mode)
    dW1g = quantize_tensor(dW1, FORMAT_BACKWARD,
                           scale_just_in_time(dW1, FORMAT_BACKWARD) if gs is None else gs, mode)
    loss = 0.5 * float(((oute - y) ** 2).sum())
    return W1 - lr * dW1g, W2 - lr * dW2g, loss, (dW1, dW2)


def make_case(seed=SEED):
    """一个固定的小模型与一批固定数据（**全部由种子决定**，换种子会换数）。"""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((M_HID, D_IN)) * 0.3
    W2 = rng.standard_normal(M_HID) * 0.3
    x = rng.standard_normal(D_IN)
    y = rng.standard_normal(M_HID) * 0.5
    return W1, W2, x, y


# ===========================================================================
# 6. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
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


def overflow_rate_mc(k, n=N_AMAX, seed=SEED):
    """推导值：`amax` 是**独立同分布**时，「本步的 `amax` 超过前 `k` 步的最大值」的比例。

    这一比例**就是** delayed scaling 的**溢出率** —— 因为 scale 由前 `k` 步的
    `amax` 定，一旦本步的 `amax` 更大，`amax·scale` 就超过最大有限数。

    返回 `(实测比例, 期望命中数)`。★ 后者是**参照量级**（P22）。
    """
    rng = np.random.default_rng(seed)
    a = rng.random(n)                       # iid ⇒ 期望溢出率 = 1/(k+1)（对称性）
    win = np.lib.stride_tricks.sliding_window_view(a, k)
    prev_max = win[:-1].max(axis=1)         # 第 t 步的「前 k 步最大值」（t = k..n-1）
    hits = int(np.count_nonzero(a[k:] > prev_max))
    return hits / float(n - k), float((n - k) / (k + 1))


# ===========================================================================
# 7. A 类：分工、scaling 的层属、量化器本身
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 训练循环里的 fp8：分工（前向 e4m3fn / 反向 e5m2）与 scaling 的层属")
    print("=" * 74)
    print("  口径：seed=%d · d=%d · m=%d · 前向编码=%s · 反向编码=%s"
          % (SEED, D_IN, M_HID, FORMAT_FORWARD, FORMAT_BACKWARD))

    W1, W2, x, y = make_case()

    # ── A.1 ★★ 分工表 ───────────────────────────────────────────────────────
    print("  ── A.1 分工：**哪个张量用哪个编码** ──")
    log = []
    fp8_step(W1, W2, x, y, 0.002, mode="fp8", encode_log=log)
    fwd_t = sorted({nm for nm, fmt, _s in log if fmt == FORMAT_FORWARD})
    bwd_t = sorted({nm for nm, fmt, _s in log if fmt == FORMAT_BACKWARD})
    print("  前向张量（%s）：%s" % (FORMAT_FORWARD, ", ".join(fwd_t)))
    print("  反向张量（%s）：%s" % (FORMAT_BACKWARD, ", ".join(bwd_t)))
    check("A.1", FORMAT_FORWARD == "e4m3fn" and FORMAT_BACKWARD == "e5m2"
          and fwd_t == ["W1", "W2", "h", "out", "x"]
          and bwd_t == ["dW1", "dW2", "dh", "dout", "dz"],
          "★ 分工逐位成立：前向 `{%s}` 用 **%s**、反向 `{%s}` 用 **%s**"
          % (",".join(fwd_t), FORMAT_FORWARD, ",".join(bwd_t), FORMAT_BACKWARD))
    check("A.1b", specs.FP8_DIVISION_QUOTE
          == "E4M3 for weight and activation tensors, and E5M2 for gradient tensors"
          and specs.FP8_DIVISION_IS_CITED_QUOTE == "This is consistent with findings in",
          "★ 而**这条分工的依据是引用来的**（[C] 原文逐字 %r 紧跟着 %r）——"
          "**本文自己只给了 `The recommended use of ...`，没有给证明**"
          "⇒ 正文不得写「论文证明前向用 E4M3、反向用 E5M2」"
          % (specs.FP8_DIVISION_IS_CITED_QUOTE, specs.FP8_DIVISION_QUOTE))

    # ── A.2 ★★ per-tensor scaling 是**软件层**，不是格式的一部分 ─────────────
    print()
    print("  ── A.2 per-tensor scaling 是软件层（[C] 逐字）──")
    print("     %r" % specs.FP8_SOFTWARE_SCALING_QUOTE)
    print("     原文给的理由：`the scaling factor can take on any real value …, while")
    print("     programmable bias is equivalent to allowing only powers of 2 as scaling factors.`")
    rng = np.random.default_rng(SEED)
    t = rng.standard_normal(N_SMALL) * 3.7
    s_real = fmt_params("e4m3fn")[0] / amax(t)
    prod = amax(t) * s_real
    print("  `amax` = %.6f · 实数 scale = %.6f · `amax·scale` = %.12f（目标 %.1f）"
          % (amax(t), s_real, prod, fmt_params("e4m3fn")[0]))
    check("A.2", abs(prod / fmt_params("e4m3fn")[0] - 1.0) < TOL_AMAX_EXACT,
          "★ 实数 scale 能把 `amax` **精确**摆到最大有限数上（相对差 %.3e < %g）—— "
          "这正是「软件层可以取任意实数」在可表示性上的样子"
          % (abs(prod / fmt_params("e4m3fn")[0] - 1.0), TOL_AMAX_EXACT))
    check("A.2b", specs.FP8_BEYOND_SCOPE_QUOTE == "beyond the scope of this paper",
          "★ 而 scale 的**选取启发式不在这篇论文里**（逐字 %r）⇒ 本程序引的是 "
          "**FP8-LM 附录 A.2** 的定义，不是把实现文档当规范（**P19**）"
          % specs.FP8_BEYOND_SCOPE_QUOTE)

    # ── A.3 量化器本身：最大相对误差 ≤ eps/2 ────────────────────────────────
    print()
    print("  ── A.3 量化器：正规区内的最大相对误差 ≤ `eps/2` ──")
    print("  %10s %14s %18s %14s" % ("格式", "eps", "实测最大相对误差", "eps/2"))
    ok = True
    for fmt in sorted(specs.FP8_FORMATS):
        v = rng.standard_normal(N_GRAD) * 10.0
        m, mn, ep = fmt_params(fmt)
        sel = (np.abs(v) >= mn) & (np.abs(v) <= m)
        rel = np.abs(fp8_round(v, fmt) - v) / np.abs(v)
        worst = float(rel[sel].max())
        print("  %10s %14.6f %18.6f %14.6f" % (fmt, ep, worst, ep / 2.0))
        if not (worst <= ep / 2.0 + 1e-12):
            ok = False
    check("A.3", ok,
          "★ 两个格式的实测最大相对误差都 **≤ `eps/2`**（舍入到最近的**硬上界**，"
          "不是「大约」）⇒ 把 `np.rint` 换成截断立刻失败")
    # ★★ **A.3b/A.3c 是结构翻转抽查换来的第二把锁**：上面那条只扫 `|v| ≤ 最大有限数`，
    #   **对溢出区完全盲**。实测：把「钳位」从**舍入之后**挪到**舍入之前**
    #   （`rint(min(ax, MAX)/step)*step`）时，100 万个样本里 **499211 个不同**、
    #   最大差 `448` —— 而 `A.3`/`C.2` **全都过**（rc=0）。
    #   那一版对**很大的输入**会给出 **0**（因为 `448/step` 在小数位不足 1）。
    print()
    print("  ── A.3b 溢出区：`|x| ≥ 最大有限数` 时必须**精确饱和**到 `±最大有限数` ──")
    print("  %10s %14s %22s %22s" % ("格式", "最大有限数", "测试输入（若干档）", "输出"))
    sat_ok = True
    bound_ok = True
    for fmt in sorted(specs.FP8_FORMATS):
        mx = fmt_params(fmt)[0]
        xs = np.array([mx, mx * 1.001, mx * 2.0, mx * 1000.0, 1e30])
        out = np.concatenate([fp8_round(xs, fmt), fp8_round(-xs, fmt)])
        print("  %10s %14.1f %22s %22s" % (fmt, mx, "|x| ∈ {mx, 1.001mx, 2mx, 1e3mx, 1e30}",
                                           "±%.1f" % float(np.abs(out).max())))
        if not np.all(np.abs(out) == mx):
            sat_ok = False
        if float(np.abs(out).max()) > mx:
            bound_ok = False
    check("A.3b", sat_ok,
          "★ 五档溢出输入（含 `1e30`）**全部精确饱和**到 `±最大有限数` —— 方向与量级都查了")
    check("A.3c", bound_ok,
          "★ 输出**永不越过** `最大有限数`（`max|q| ≤ 最大有限数`）—— "
          "与 A.3b 一起把「饱和」这件事两面夹住")

    # ── A.4 阴性对照：**恒等量化**下与 fp32 参照**逐位相同** ─────────────────
    print()
    print("  ── A.4 阴性对照：把量化器与 scaling 都换成**恒等**，与 fp32 参照对拍 ──")
    g = x
    w1f, w2f, loss_fp8, (dw1f, dw2f) = fp8_step(W1, W2, x, y, 0.002, mode="identity")
    dw1p, dw2p = plain_backward(W1, W2, x, y)
    d1 = float(np.abs(dw1f - dw1p).max())
    d2 = float(np.abs(dw2f - dw2p).max())
    h, out = plain_forward(W1, W2, x)
    loss_p = 0.5 * float(((out - y) ** 2).sum())
    print("  `dW1` 最大逐元素差 = %.3e · `dW2` = %.3e · loss 差 = %.3e"
          % (d1, d2, abs(loss_fp8 - loss_p)))
    check("A.4", d1 == 0.0 and d2 == 0.0 and loss_fp8 == loss_p,
          "★ 恒等量化下，被测侧的 `dW1` / `dW2` / loss 与**独立手推的 fp32 参照**"
          "**逐位相同**（0.000e+00）⇒ 本程序的 fp8 通路**没有额外的接线错误**"
          "（量化与 scaling 才是它唯一的额外来源）")
    return log


# ===========================================================================
# 8. B 类：代价模型 —— scale 的 bit 开销、delayed scaling 的溢出率
# ===========================================================================
def part_B(log):
    print()
    print("=" * 74)
    print("B. 代价模型：scale 的开销账，与 delayed scaling 的**溢出率闭式**")
    print("=" * 74)

    # ── B.1 ★★ scale 的**作用域与 bit 开销**（库内硬要求）──────────────────
    print("  ── B.1 scale 的作用域与 bit 开销 ──")
    print("     作用域 = **%s**（每张量）· 开销 = 每张量 **%d bit**（1 个 fp32）"
          % (SCALE_SCOPE, SCALE_BITS))
    n_tensors = len({nm for nm, _f, _s in log})   # ★ 由 `A.1` 那张分工表**数出来**，不写死
    print("     一次训练步里被编码/缩放的**不同张量**个数 = %d（= A.1 那张表的大小）" % n_tensors)
    rows = []
    for n in (N_SMALL, N_LARGE):
        data_bits = 8 * n                # fp8 的数据位（★ 8 是「一个元素的位宽」，不是位分配）
        overhead = n_tensors * SCALE_BITS
        rows.append((n, data_bits, overhead, overhead / float(data_bits)))
        print("     %10d 个元素：数据 %12d bit · scale 开销 %6d bit · 相对开销 %.6f%%"
              % (n, data_bits, overhead, 100.0 * overhead / data_bits))
    check("B.1", rows[0][2] == rows[1][2] and abs(rows[0][3] / rows[1][3] - 1000.0) < 1e-6,
          "★ 元素数 ×1000 时 **scale 的开销一个 bit 都不涨**（都是 %d bit）、相对开销**降 1000 倍** —— "
          "这就是「per-tensor scaling 是**软件层**」在字节上的样子：它**不占元素位宽**"
          % rows[0][2])
    check("B.1b", SCALE_SCOPE == "per-tensor" and SCALE_BITS == 32,
          "★ 作用域（%s）与 bit 开销（%d）**逐位钉住** —— 库内硬要求：凡写 scale 必写这两项"
          % (SCALE_SCOPE, SCALE_BITS))
    print("     ★ 其它粒度（每通道 / 每 token / 每 32 元素的块）的正文在")
    print("       `130-model-optimization/quantization/06-granularity.md` 与 `02-e8m0-and-mx.md:276`，**此处回链**。")

    # ── B.2 delayed scaling 的**溢出率 = 1/(k+1)** ──────────────────────────
    print()
    print("  ── B.2 delayed scaling 的溢出率：实测 vs 闭式 `1/(k+1)` ──")
    print("     闭式的理由：`amax` 独立同分布时，`k+1` 个数里**最大的是最后一个**的概率 = `1/(k+1)`")
    print("     （★ `amax` 用 `U(0,1)` 抽样 —— 闭式只依赖**连续同分布**，不依赖具体分布）")
    print("  %6s %16s %16s %14s %14s" % ("k", "实测溢出率", "闭式 1/(k+1)", "相对差", "期望命中数"))
    worst = 0.0
    for k in K_SWEEP:
        rate, expect = overflow_rate_mc(k)
        rel = abs(rate - 1.0 / (k + 1)) * (k + 1)
        worst = max(worst, rel)
        print("  %6d %16.6f %16.6f %14.6f %14.0f" % (k, rate, 1.0 / (k + 1), rel, expect))
        if expect < MIN_HITS_AMAX:
            worst = 1e9
    # ── B.2c `scale_delayed` 的语义：取**前 k 次的最大值** ──────────────────
    # ★★ 这一条是**结构翻转抽查**换来的：`scale_delayed` 原来只被文档提到、**没有断言消费者**
    #   （= 计划 §4 的 T11 盲区①：没被消费的代码改了也无人发现）。实测把 `max` 改成 `min`
    #   （或用整个历史而不是前 `k` 项）时 **rc=0、断言全过** ⇒ 补一条直接的语义判据。
    hist = (100.0, 1.0, 2.0)
    s_k3 = scale_delayed(hist, "e5m2", 3)
    s_k2 = scale_delayed(hist, "e5m2", 2)
    maxv = fmt_params("e5m2")[0]
    print("  历史 %s ⇒ k=3 的 scale = %.6f（= 最大有限数/%.1f）· k=2 的 scale = %.6f（= 最大有限数/%.1f）"
          % (list(hist), s_k3, maxv / s_k3, s_k2, maxv / s_k2))
    check("B.2c", abs(s_k3 - maxv / 100.0) < 1e-9 and abs(s_k2 - maxv / 2.0) < 1e-9,
          "★ `scale_delayed` 的语义逐位钉住：取**前 `k` 项的 `max`**（k=3 用 100.0、k=2 只用后两项"
          "的最大值 2.0）⇒ 把 `max` 换成 `min`、或把 `history[-k:]` 换成整个 `history`，"
          "两条都会立刻失败")

    check("B.2", worst < TOL_OVERFLOW_LAW,
          "★ 五个 `k` 上实测溢出率与 `1/(k+1)` 的相对差最大 %.3e < %g，且每个 `k` 的"
          "**期望命中数 ≥ %d**（★ P22：先打印参照量级）" % (worst, TOL_OVERFLOW_LAW, MIN_HITS_AMAX))
    print("     ⇒ **`k` 越小、溢出越频繁**：`k = 1` 时**一半**的步会溢出。")
    print("       这正是 [C] 原文说 `Weight update skipping … is not a good choice for FP8` 的")
    print("       **定量背景** —— 在 fp8 上按 fp16 那套「溢出就跳步」办，跳掉的步数太多。")


# ===========================================================================
# 9. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：2 的幂 scale 真的浪费范围；突增批真的饱和、真的丢掉整步")
    print("=" * 74)
    rng = np.random.default_rng(SEED)

    # ── C.1 只允许 2 的幂的 scale ⇒ 范围**真的**只用掉一半 ──────────────────
    print("  ── C.1 把 scale 限制成「只能取 2 的幂」（格式层的 bias 只能这么干）──")
    t = rng.standard_normal(N_SMALL) * 3.0
    aim = REAL_SCALE_AIM
    amax_t = fmt_params("e4m3fn")[0] / aim                # 挑一个 amax 让 s_real 恰好 = aim
    s_real = fmt_params("e4m3fn")[0] / amax_t
    s_pow = 2.0 ** math.floor(math.log2(s_real))          # 只能往下取，否则溢出
    used_real = amax_t * s_real / fmt_params("e4m3fn")[0]
    used_pow = amax_t * s_pow / fmt_params("e4m3fn")[0]
    print("  真实 scale = %.6f（2^%.4f）⇒ 只能取 %.4f（2^%d）"
          % (s_real, math.log2(s_real), s_pow, int(math.log2(s_pow))))
    print("  `amax·scale / 最大有限数`：实数 scale = %.6f · 2 的幂 scale = **%.6f**"
          % (used_real, used_pow))
    print("  ⇒ 2 的幂 scale 下，`amax` 只摆到 %.2f（而不是 %.1f）"
          % (amax_t * s_pow, fmt_params("e4m3fn")[0]))
    check("C.1", used_real > 1.0 - 1e-9 and used_pow < TOL_RANGE_LOSS,
          "★ 实数 scale 用满范围（%.6f），而**只能取 2 的幂**时只用到 **%.4f**（< %g）—— "
          "**表示范围真的浪费掉一半**。这就是 [C] 原文说「软件层更强」的那半个理由"
          % (used_real, used_pow, TOL_RANGE_LOSS))
    check("C.1b", s_pow == 16.0 and abs(s_real - REAL_SCALE_AIM) < 1e-9,
          "★ 两档的取值逐位钉住（实数 %.4f、2 的幂 %g）—— 没有这个对照，C.1 会退化成恒真"
          % (s_real, s_pow))

    # ── C.2 ★ 突增批：delayed scale **真的**饱和；「跳步」**真的**丢掉整步 ───
    print()
    print("  ── C.2 突增批（`x` 乘 %g）：delayed scale 下的饱和与「跳步」的代价 ──" % SPIKE_FACTOR)
    W1, W2, x, y = make_case()
    _h, _o = plain_forward(W1, W2, x)
    dW1_n, dW2_n = plain_backward(W1, W2, x, y)
    x_sp = x * SPIKE_FACTOR
    dW1_s, dW2_s = plain_backward(W1, W2, x_sp, y)
    a_normal = max(amax(dW1_n), amax(dW2_n))
    a_spike = max(amax(dW1_s), amax(dW2_s))
    # ★ 历史 = 前 `K_DELAYED` 个**正常批**的 amax（这里造一个略微波动的历史，
    #   好让 `scale_delayed` 的 `max` / `history[-k:]` 两件事都**真的**参与运算）。
    history = [a_normal * f for f in (0.7, 0.8, 0.9, 1.0, 0.95, 0.85, 0.75, 1.0)]
    s_delayed = scale_delayed(history, "e5m2", K_DELAYED)   # 由历史定（delayed）
    # ★ just-in-time 的尺度取自**本批两个梯度张量的联合 amax** —— 两个张量共用同一个
    #   scale 是本节的简化（真实实现是**每张量一个** scale，作用域仍是 per-tensor）。
    s_jit = fmt_params("e5m2")[0] / a_spike
    print("  正常批 `amax` = %.6f · 突增批 `amax` = %.6f（放大 %.1f×）"
          % (a_normal, a_spike, a_spike / a_normal))
    print("  delayed scale = %.6f（由正常批定）· just-in-time scale = %.6f" % (s_delayed, s_jit))
    n_sat_d = n_saturated(dW1_s, "e5m2", s_delayed) + n_saturated(dW2_s, "e5m2", s_delayed)
    n_sat_j = n_saturated(dW1_s, "e5m2", s_jit) + n_saturated(dW2_s, "e5m2", s_jit)
    print("  突增批的**饱和元素数**：delayed = **%d** · just-in-time = %d" % (n_sat_d, n_sat_j))
    check("C.2", n_sat_d >= TOL_SAT_MIN and n_sat_j == 0,
          "★ 突增批在 delayed scale 下**真的饱和**（%d 个元素 ≥ %d），"
          "而 just-in-time 下**一个都不饱和**（%d）⇒ 「scale 是陈旧的」这件事是可证的"
          % (n_sat_d, TOL_SAT_MIN, n_sat_j))
    # 「跳步」vs「饱和」：突增批那一步的更新量
    dW1_q = quantize_tensor(dW1_s, "e5m2", s_delayed)     # 饱和策略：照旧编码并应用
    dW1_j = quantize_tensor(dW1_s, "e5m2", s_jit)          # just-in-time：不失真
    delta_sat = float(np.abs(dW1_q).max())
    delta_skip = 0.0 if n_sat_d > 0 else delta_sat         # 跳步：饱和 ⇒ 整步丢弃
    distort = float(np.abs(dW1_q - dW1_s).max())
    jit_distort = float(np.abs(dW1_j - dW1_s).max())
    print("  那一步 `dW1` 的偏差：delayed 编码后 = %.4e · just-in-time 编码后 = %.4e"
          % (distort, jit_distort))
    print("  该步的更新量（`max|dW1|`）：**跳步策略 = %.3e**（整步被丢）· 饱和策略 = %.3e"
          % (delta_skip, delta_sat))
    check("C.2b", delta_skip == 0.0 and delta_sat > 0.0,
          "★ 「跳步」把突增批那一步的更新**精确地丢干净**（%.3e == 0），"
          "而「饱和」策略至少还走了一步（%.3e）⇒ 与 B.2 的 `1/(k+1)` 合起来，"
          "「跳步不适用于 fp8」这句话就是**可量化**的" % (delta_skip, delta_sat))
    check("C.2d", distort > jit_distort,
          "★ 而且 delayed scale 下那一步的梯度**本身**也被饱和扭曲了（偏差 %.4e > "
          "just-in-time 的 %.4e）—— 「饱和」不是没代价，只是代价比「整步丢掉」小"
          % (distort, jit_distort))
    check("C.2c", specs.FP8_OVERFLOW_NOT_SKIP_QUOTE == "is not a good choice for FP8",
          "★ 而那句话是 [C] **逐字**的（%r）—— 本程序只负责把它**定量**，不负责改写它"
          % specs.FP8_OVERFLOW_NOT_SKIP_QUOTE)


# ===========================================================================
# 10. D 类：失败标记自检 + 口径/阈值钉住 + specs 钉住 + 独立性
# ===========================================================================
def preflight():
    """★★ **D 类的前半段先跑**：标记自检 + 口径/阈值钉住（理由见 Chunk 1 的 P23）。

    ★★ **必须放在任何计算之前**：口径常量（`D_IN` / `M_HID` / `N_GRAD` / `N_SMALL` /
      `K_SWEEP` …）一旦被改，后面的**形状/索引检查**会先崩 —— 那时 `mutate.py` 记的是
      **崩溃捕获**，而「崩溃捕获不算真捕获」。⇒ 钉住提到**任何**计算之前（本库的 **P23**）。
    """
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 方法学让步（[C] 原文逐字 %r）：算术全在宽类型上做，只有**存储/输入**被量化"
          % specs.FP8_SIMULATED_QUOTE)
    print("     ⇒ 本程序**不得**支撑任何关于「真 fp8 算术」（吞吐 / 累加顺序 / Tensor Core）的结论。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ":: FAIL: mixed-precision" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", "++ FAIL: optimizers",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")

    measured = {"SEED": SEED, "D_IN": D_IN, "M_HID": M_HID, "N_GRAD": N_GRAD,
                "FORMAT_FORWARD": FORMAT_FORWARD, "FORMAT_BACKWARD": FORMAT_BACKWARD,
                "SCALE_SCOPE": SCALE_SCOPE, "SCALE_BITS": SCALE_BITS,
                "REAL_SCALE_AIM": REAL_SCALE_AIM, "N_SMALL": N_SMALL, "N_LARGE": N_LARGE,
                "SPIKE_FACTOR": SPIKE_FACTOR, "K_DELAYED": K_DELAYED, "K_SWEEP": K_SWEEP,
                "N_AMAX": N_AMAX, "TRAIN_STEPS": TRAIN_STEPS}
    expect = {"SEED": 20260927, "D_IN": 32, "M_HID": 16, "N_GRAD": 4096,
              "FORMAT_FORWARD": "e4m3fn", "FORMAT_BACKWARD": "e5m2",
              "SCALE_SCOPE": "per-tensor", "SCALE_BITS": 32,
              "REAL_SCALE_AIM": 31.99, "N_SMALL": 4096, "N_LARGE": 4096000,
              "SPIKE_FACTOR": 25.0, "K_DELAYED": 8, "K_SWEEP": (1, 2, 4, 8, 16),
              "N_AMAX": 100000, "TRAIN_STEPS": 5}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 16 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", N_LARGE == 1000 * N_SMALL,
          "★ 「×1000」那个对照的两端**由同一条关系钉住**（%d = 1000 × %d）—— "
          "B.1 的「相对开销降 1000 倍」全靠它" % (N_LARGE, N_SMALL))
    check("D.3c", math.log2(REAL_SCALE_AIM) != math.floor(math.log2(REAL_SCALE_AIM)),
          "★ C.1 的 `REAL_SCALE_AIM` **不是** 2 的幂（否则「浪费一半」会退化成恒真）")

    th = {"TOL_CITED": TOL_CITED, "TOL_Q_REL": TOL_Q_REL, "TOL_AMAX_EXACT": TOL_AMAX_EXACT,
          "TOL_IDENTITY": TOL_IDENTITY, "TOL_OVERFLOW_LAW": TOL_OVERFLOW_LAW,
          "MIN_HITS_AMAX": MIN_HITS_AMAX, "TOL_RANGE_LOSS": TOL_RANGE_LOSS,
          "TOL_SAT_MIN": TOL_SAT_MIN}
    th_expect = {"TOL_CITED": 0.0, "TOL_Q_REL": 0.0, "TOL_AMAX_EXACT": 1e-12,
                 "TOL_IDENTITY": 0.0, "TOL_OVERFLOW_LAW": 0.02, "MIN_HITS_AMAX": 200,
                 "TOL_RANGE_LOSS": 0.55, "TOL_SAT_MIN": 100}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 8 个逐位一致（含**两个 0 容差**：逐字比对、逐位恒等）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    # ★★ 格式表的**键集**必须在这里钉住（**任何计算之前**）：改一个键名会让 `fmt_params()`
    #   抛 `KeyError` —— 那是**崩溃捕获**，而「崩溃捕获不算真捕获」。本程序第一次
    #   `make mutate` 时有 **2 条**崩溃捕获，正是这两个键；钉住后它们变成断言捕获。
    check("D.4b", tuple(sorted(specs.FP8_FORMATS)) == ("e4m3fn", "e5m2"),
          "★ 格式表的**键集**逐位钉住（%s）—— 提前钉住它，改键名就是**断言捕获**而不是 `KeyError`"
          % (tuple(sorted(specs.FP8_FORMATS)),))
    # ★★ scale 的作用域与 bit 开销：**本程序的局部常量**与 `specs.py` 的常量**必须一致**
    #   （`CONTRIBUTING:36` 的「`code/` 自包含」⇒ 跨文件同值各写一份，靠这一行互钉）。
    check("D.4c", specs.FP8_SCALE_SCOPE == SCALE_SCOPE and specs.FP8_SCALE_BITS == SCALE_BITS
          and specs.FP8_SCALE_SCOPE == "per-tensor" and specs.FP8_SCALE_BITS == 32,
          "★ scale 的**作用域**（%r）与 **bit 开销**（%d）在本程序与 `specs.py` 里**两处一致** "
          "—— 没有这一条，`specs` 的那两个常量**没有任何消费者**（`make mutate` 会记 2 条漏网）"
          % (specs.FP8_SCALE_SCOPE, specs.FP8_SCALE_BITS))
    return len(FAILS) == before


def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住（fp8 的**训练循环侧**）与独立性证据")
    print("=" * 74)
    check("D.5", specs.FP8_FORMATS == {"e4m3fn": (448.0, 0.015625, 0.125),
                                       "e5m2": (57344.0, 6.103515625e-05, 0.25)},
          "★ 两个格式的**三个参数**逐位一致（最大有限数 / 最小正规数 / eps，共 **6** 个值）—— "
          "★ 它们只是**输入**：位分配与四个变体的正文在 `130-…/01-fp8-family.md` §0。"
          "★★ 幂次（`2^-6` / `2^-3` / `2^-14` / `2^-2`）写在 `specs.py` 的**注释**里、"
          "值写成**十进制字面量** —— 写成 `2.0 ** -6` 时 `mutate.py` 的 `_walk` 下钻不到，"
          "那四个值会**整体逃出分母**（本目录实测）")
    check("D.6", specs.FP8_SOFTWARE_SCALING_QUOTE
          == "Leaving the per-tensor scaling to software implementation",
          "★ 「per-tensor scaling 是软件层」的逐字依据**逐位**一致（A.2 判的就是它）")
    check("D.7", specs.FP8_BEYOND_SCOPE_QUOTE == "beyond the scope of this paper"
          and specs.FP8_OVERFLOW_NOT_SKIP_QUOTE == "is not a good choice for FP8"
          and specs.FP8_SIMULATED_QUOTE == "simulated FP8"
          and specs.FP8_DIVISION_IS_CITED_QUOTE == "This is consistent with findings in",
          "★ 另外四条逐字依据逐位一致（超纲声明 / 溢出处置 / simulated FP8 / 分工是引用来的）"
          " —— ★ **一律用 `==`**：它们的变异形态是末尾追加字符（计划 §3 开头盲区③）")
    check("D.8", specs.FP8_LM_JUST_IN_TIME_QUOTE
          == ("determining the scaling factor based on the maximum absolute value (amax) "
              "of the tensor being generated")
          and specs.FP8_LM_DELAYED_QUOTE
          == ("selecting the scaling factor based on the maximum absolute values observed "
              "in a certain number of preceding iterations"),
          "★ `just-in-time` 与 `delayed scaling` 的**逐字定义**逐位一致（来源 = [C] **FP8-LM "
          "附录 A.2** —— ★ 这两个术语在 fp8 论文里各 **0** 次，见 T17）")
    check("D.9", specs.UNSOURCED == ("bf16_dynamic_range_exact_equality_claim",
                                      "torch_e4m3fn_overflow_rule_version_span",
                                      "fp8_scaling_growth_rule_concrete_values"),
          "★ `specs.UNSOURCED` 的三项**逐位钉住**（%d 项）—— 其中一项正是「scale 的具体"
          "增长/回退规则没有可引的一手来源」，所以 `K_DELAYED` 与 `K_SWEEP` 标为**本程序的选择**"
          % len(specs.UNSOURCED))
    check("D.10", K_DELAYED in K_SWEEP and all(k >= 1 for k in K_SWEEP),
          "★ 本程序选的历史长度 %d 落在扫描表 %s 里（闭式 `1/(k+1)` 要求 `k ≥ 1`）"
          % (K_DELAYED, list(K_SWEEP)))

    # ── 独立性：参照路径不调用被测路径 ──────────────────────────────────────
    check("D.11", inspect.getsource(plain_backward) != inspect.getsource(fp8_step),
          "★ 防 P17：参照（`plain_backward`）与被测（`fp8_step`）的**源码不同**"
          "（%d vs %d 字节）"
          % (len(inspect.getsource(plain_backward)), len(inspect.getsource(fp8_step))))
    watched = {"fp8_step", "quantize_tensor", "fp8_round", "scale_just_in_time"}
    called = _callee_names(plain_backward) | _callee_names(plain_forward)
    check("D.12", not (watched & called),
          "★ **AST 调用表**：两条参照路径的 `Call` 节点 = %s —— 一个被测函数都没有"
          % (", ".join(sorted(called)) if called else "无"))
    check("D.13", "np.rint" in inspect.getsource(fp8_round),
          "★ 被测量化器**真的**用了舍入到最近（`np.rint`）⇒ `A.3` 的 `≤ eps/2` 不是空断言"
          "（改成截断就会超界）")
    check("D.14", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))


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
    print("03_fp8.py —— 训练循环里怎么用 fp8：分工、per-tensor scaling、溢出的处置")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 收窄原则（写死）：**只写训练循环里怎么用 fp8，位域与 scale 粒度一律回链** ——")
    print("     fp8 的位分配与四个变体在 `130-model-optimization/quantization/01-fp8-family.md` §0；")
    print("     scale 的其它粒度在 `130-model-optimization/quantization/06-granularity.md`；")
    print("     累加器宽度的误差界在 `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md`。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    log = part_A()
    part_B(log)
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
