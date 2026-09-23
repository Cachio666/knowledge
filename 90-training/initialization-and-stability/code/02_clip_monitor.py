#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_clip_monitor.py —— 梯度裁剪，以及**范数监控的判据**。

配套文档：`02-clipping-and-monitoring.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ **本主题的禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）—— 本程序**一行都不写**：
  · ★★ **warmup 的机制解释**（post-LN 的平均场理由、`hypothesize` 的措辞纪律）→
    `80-model-architectures/normalization/02-placement-and-qk-norm.md:83,93-97,100,278`；
  · **残差流的范数随深度增长** → `80-…/transformer/01:191,232-233,263`；
  · **误差沿深度的传播** → `130-quantization/theory/03-outliers-and-propagation.md:124,205`；
  · **优化器状态的分片与通信量** → `100-parallelism/README.md:14`（`zero-fsdp`）。
  ★ 本程序只写**单卡口径的范数与监控**；`B.1` 里那句「监控只需要每层一个标量」
    **是纯代数的**（`√Σ ‖g_l‖²` 与拼接/顺序无关），**不是**分布式的通信方案（那在 `100-parallelism`）。

★★ **两处回链（阈值与判据都定义在别处，本程序只实现）**：
  ① **裁剪的判据与阈值**在 `90-training/autograd-and-backward/03-gradient-lifecycle.md`
     的「表 A ⑦」——**在那里只写一次**（`T ≥ 1`、上侧 `≤ 1.1e-6/max(T,1)`、
     ★★ **下侧 `== ε/(T+ε)`，即 `A.2b`**）。★ **回链时必须一并带上 `A.2b`**（**T26**：
     只写单侧会放过一个「删掉保护项 `+1e-6`」的实现，而它**结果反而更精确**）。
     ★ 本程序的 `A.1c` 把 **T26 那个见证本身**做成了一个可执行断言。
  ② **`unscale` 的时机与溢出 ⇒ 跳步**在 `90-training/mixed-precision/02-amp-execution.md`
     —— 本程序**不重定义**那条判据，只在 `C.1`/`C.2` 里把它与**监控**的接口钉住。

★★ **本程序的新内容 = 监控的判据**（裁剪本身只实现、不重定义）：
  · 监控量必须是**裁剪前**的范数（[B] 文档逐字：返回值 = `Total norm of the parameter
    gradients (viewed as a single vector)`）⇒ 取裁剪后的量会**永远不触发**；
  · 判**非有限值**必须用 `isfinite`，不能用 `>`（`nan > thr` **恒为假**）；
  · 「裁剪触发率」有一条**闭式**（Gaussian 梯度 ⇒ `P(χ²_n > (max_norm/σ)²)`）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  裁剪的**两侧**判据：上侧 `≤ 1.1e-6/max(T,1)`（`T ≥ 1`）**且**下侧 `== ε/(T+ε)`；
  · `A.1c` ★★ **只写单侧会放过错实现**：把 `+ε` 删掉后**上侧仍然通过**（更精确）而下侧失败；
  · `A.2`  监控量 == **裁剪前**范数（逐位），而裁剪后范数 == `max_norm·T/(T+ε)`；
  · `A.2b` ★★ 用**裁剪后**的范数做监控 ⇒ 阈值**100% 不触发**（假阴性），而 pre 的触发率 > 0；
  · `A.3`  ★★ `nan > thr` 是 `False`、`inf > thr` 是 `True`、`not isfinite(nan)` 是 `True`
           ⇒ 「用 `>` 写监控」对 `nan` **半失效**；
  · `A.4`  ★★ 全局裁剪 vs **逐张量**裁剪：总范数差 **`√L` 倍**（`L = 3` ⇒ `1.732051`，闭式）；
  · `A.5`  ★★ **触发率的闭式**：MC 与 `P(χ²_n > (max_norm/σ)²)` 一致（偶数 `n` 的 Erlang 式）；
  · `B.1`  监控的**字节账**：全局面板可由 `L` 个**标量**重算，且**与顺序无关**；
  · `B.2`  单层范数被**层宽**主导（`‖g‖ ∝ √n`）⇒ 跨层比较必须用每元素 RMS；
  · `C.1`  `max_norm = 0` ⇒ 梯度清零、更新为 `0`，而**监控（看 pre）说"正常"**；
  · `C.2`  `error_if_nonfinite` 的默认值是 `False` ⇒ `inf`/`nan` **默认静默通过**；
  · `C.3`  对**全 0 的层**做逐张量归一化 ⇒ 静默 `nan`（而全局范数不会）；
  · `C.4`  ★★ **源码级次序钉住**：监控在裁剪**之前**（数值判据在「不触发」那一档上退化成 no-op）。

用法：
    python3 02_clip_monitor.py
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
SEED = 20260951              # ★ 与同域另外两个程序（20260950 / 20260952）都**不同**

# ★★ 裁剪的阈值口径 —— **判据与阈值定义在 `autograd-and-backward/03` 的「表 A ⑦」**，
#   本程序**只实现、不重定义**。`ε` 从 `specs.py` 读（[B] PyTorch v2.9.0 的源码口径）。
#   ★ `T ≥ 1` 是判据的一部分（计划 §3 表 A ⑦ 写死）；`T` 是**裁剪前**的全局范数。
CLIP_T_LIST = (2.0, 4.0, 8.0)      # ★ `T ≥ 1`
CLIP_MAX_NORM = 1.0
CLIP_MAX_NORM_LIST = (1.0, 0.5, 0.1)
CLIP_TOL_REL = 1.1e-6              # 上侧：实际阈值是 `CLIP_TOL_REL / max(T, 1)`
CLIP_TOL_CLOSED = 1e-6             # 下侧（`A.2b`）：与闭式 `ε/(T+ε)` 的相对偏差
CLIP_T_SMALL = 0.5                 # `A.1b`：`T < max_norm` ⇒ **不发生裁剪**那一档
TENSOR_SHAPES = ((64, 64), (128, 32), (256, 256))   # `A.4`：三个张量（`L = 3`）
MON_STEPS = 200                    # `A.2` 的监控步数
MON_SIGMA = 0.0037                 # 监控那批梯度的尺度（★ 让触发率**既不是 0 也不是 1**）
N_MON_ELEMS = 8                    # `A.5`：玩具「模型」的梯度元素总数（**偶数** ⇒ Erlang 闭式）
SIGMA_LIST = (0.2, 0.4, 0.8)       # `A.5`：三个尺度（触发率应跨一个数量级）
TRIG_SAMPLES = 200000              # `A.5` 的 MC 抽样数
TRIG_SE_MULT = 4.0                 # MC 容差 = 4 · 相对标准误
NORM_T = 10.0                      # 构造一个范数**恰好**为它的梯度
NORM_T_INF = 100.0                 # `B.3`：让 `‖g‖_inf` 也**超过** `max_norm`（两种口径都触发）
LR = 0.1                           # `C.4` 的步长
LAYER_N_LIST = (64, 256, 1024, 4096)   # `B.2`：层宽扫描
RMS_DRAWS = 400                    # `B.2`：每层的抽样次数（取均值以压低 `1/√(2n)` 的噪声）
MON_THRESHOLD = 3.0                # `A.3`：监控用的阈值
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   （`1e300 + 1.0 == 1e300`、`1e20 + 1.0 == 1e20`）⇒ 那样的叶子**变异后逐位不变**、
#   **任何断言都抓不到**。本常量因此取 **`1.0e15`**（`< 2^53 = 9.007e15` ⇒ `+1.0` 会改变它）
#   并进 `preflight()` 的钉住表 ⇒ 变异变成**断言捕获**。
INF_BIG = 1.0e15                   # `A.3` 表里那个「大而有限」的值（构造 `inf` 另用 `INF_MUL`）
TOL_EXACT = 0.0                    # 逐位相同
TOL_TIGHT = 1e-12
TOL_IDENT = 1e-15
TOL_MEASURABLE = 1e-2              # 反面对照的**可测性下界**
TOL_RMS = 0.02                     # `B.2`：每元素 RMS 跨层的一致性
TOL_CHEB = 1e-9                    # `B.3`：`‖·‖_inf` 与 `‖·‖_2` 的夹逼不等式


# ===========================================================================
# 2. 路径（被测）：范数与裁剪
# ===========================================================================
def global_norm_from_layers(layers):
    """**被测**：全局范数 = `√(Σ_l Σ_i g_li²)`（[B] 文档的「当成一个向量」口径）。

    ★★ **[B] 文档逐字**（`clip_grad_norm_` 的正文）：
      `The norm is computed over the norms of the individual gradients of all parameters,
       as if the norms of the individual gradients were concatenated into a single vector.`
      ⇒ 对 `norm_type = 2` 而言 「把范数拼接」与「把梯度拼接」**等价**
        （`(Σ_l ‖g_l‖^p)^{1/p}` 就是拼接向量的 `p` 范数）—— `B.3` 把 `p = 2` 与 `p = inf`
        两种口径分开量。
    """
    tot = 0.0
    for g in layers:
        a = np.asarray(g, dtype=np.float64)
        tot += float((a * a).sum())
    return math.sqrt(tot)


def global_norm_concat(layers):
    """★ **参照路径**：把全部梯度**真的拼接**起来再取 `np.linalg.norm`（另一条算式）。"""
    flat = np.concatenate([np.asarray(g, dtype=np.float64).ravel() for g in layers]) \
        if len(layers) > 1 else np.asarray(layers[0], dtype=np.float64).ravel()
    return float(np.linalg.norm(flat))


def global_norm_from_scalars(layer_norms):
    """★ **参照路径 2**：只拿**每层一个标量**（`‖g_l‖`）重算全局范数 = `√Σ ‖g_l‖²`。

    ★★ 它是 `B.1` 的「监控的字节账」的被测对象：**全局面板可以从 `L` 个标量重算**。
    """
    return math.sqrt(sum(float(n) * float(n) for n in layer_norms))


def layer_norms(layers):
    """每层的范数（监控时**真正要记的那 `L` 个数**）。"""
    return [math.sqrt(float((np.asarray(g, dtype=np.float64) ** 2).sum())) for g in layers]


def clip_global_norm(layers, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """按全局范数缩放。返回 `(裁剪后的层列表, 裁剪【前】的全局范数)`。

    ★★ **返回值里的那个范数就是监控量**（[B] 文档：`clip_grad_norm_` 的返回值 =
      `Total norm of the parameter gradients (viewed as a single vector)`）——
      **它是裁剪前的**。★ `A.2`/`A.2b` 把这件事的后果量出来。
    ★ 判据与阈值在 `90-training/autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦
      （**在那里只写一次**）—— **本程序只实现、不重定义**。
    """
    total = global_norm_from_layers(layers)
    coef = max_norm / (total + eps)
    if coef < 1.0:
        return [np.asarray(g, dtype=np.float64) * coef for g in layers], total
    return [np.asarray(g, dtype=np.float64) for g in layers], total


def clip_report_post(layers, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """★★ **反面对照**：把**裁剪后**的范数当监控量（`A.2b` 的被测对象）。"""
    clipped, _ = clip_global_norm(layers, max_norm, eps)
    return clipped, global_norm_from_layers(clipped)


def clip_no_eps(layers, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """★★ **错实现（`A.1c` 的被测对象）**：把分母的保护项 `+ε` **删掉**（`coef = max_norm/T`）。

    ★ 它让裁剪**更精确**（相对误差变成 `0`）⇒ **上侧判据抓不到它** —— 只有下侧（`A.2b`）能抓。
    """
    total = global_norm_from_layers(layers)
    coef = max_norm / total
    if coef < 1.0:
        return [np.asarray(g, dtype=np.float64) * coef for g in layers], total
    return [np.asarray(g, dtype=np.float64) for g in layers], total


def clip_per_tensor(layers, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """★★ **反面对照（`A.4`）**：**逐张量**各自裁到 `max_norm`（不是全局口径）。"""
    out = []
    for g in layers:
        a = np.asarray(g, dtype=np.float64)
        n = math.sqrt(float((a * a).sum()))
        coef = max_norm / (n + eps)
        out.append(a * coef if coef < 1.0 else a)
    return out


def clip_by_norm_type(layers, max_norm, norm_type, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """按 `p` 范数裁剪（`norm_type = 2.0` 或 `float('inf')`）。

    ★ [B] 文档逐字：`norm_type (float, optional) – type of the used p-norm. Can be 'inf' for
      infinity norm. **Default: 2.0**` ⇒ 同一个 `max_norm` 在两种口径下**含义不同**（`B.3`）。
    """
    if norm_type == float("inf"):
        total = max(abs(float(x)) for g in layers for x in np.asarray(g).ravel())
    else:
        total = global_norm_from_layers(layers)
    coef = max_norm / (total + eps)
    if coef < 1.0:
        return [np.asarray(g, dtype=np.float64) * coef for g in layers], total
    return [np.asarray(g, dtype=np.float64) for g in layers], total


def train_step_with_monitor(layers, max_norm, lr, scale=1.0):
    """★★ **管线**：`unscale → monitor（读【裁剪前】的范数）→ clip → ×(−lr)`。

    ★★ **次序是判据的一部分**（`C.4` 的源码级钉住）：`global_norm_from_layers` 的调用
      必须**早于** `clip_global_norm`。
    """
    unscaled = [np.asarray(g, dtype=np.float64) / scale for g in layers]
    pre_norm = global_norm_from_layers(unscaled)
    clipped, _ = clip_global_norm(unscaled, max_norm)
    dw = [-lr * g for g in clipped]
    return dw, pre_norm


def train_step_monitor_after_clip(layers, max_norm, lr, scale=1.0):
    """★★ **反面对照**：把**监控**挪到裁剪**之后**（`C.4` 的被测对象）。

    ⚠★ 它在 `T < max_norm`（**不触发裁剪**）那一档上**逐位给出同一个数** ⇒
    **数值断言抓不到它** ⇒ 必须靠源码级次序钉住（`C.4`）。
    """
    unscaled = [np.asarray(g, dtype=np.float64) / scale for g in layers]
    clipped, _ = clip_global_norm(unscaled, max_norm)
    post_norm = global_norm_from_layers(clipped)
    dw = [-lr * g for g in clipped]
    return dw, post_norm


def monitor_ok(value, threshold):
    """★ **错写法的监控**：用 `>` 比 —— `nan` 会被**静默放行**（`A.3`）。"""
    return value > threshold


def monitor_ok_finite(value, threshold):
    """★ **正确写法的监控**：先要求有限，再比阈值。"""
    if not math.isfinite(value):
        return False
    return value > threshold


# ===========================================================================
# 3. 路径（被测）：触发率的闭式
# ===========================================================================
def chi2_sf_even(x, n):
    """`P(χ²_n > x)`，`n` 为**偶数**：`e^{−x/2} Σ_{k=0}^{n/2−1} (x/2)^k / k!`（Erlang）。

    ★★ 依据：Gaussian 梯度 `g ∼ N(0, σ²I_n)` ⇒ `‖g‖²/σ² ∼ χ²_n`
      ⇒ `P(‖g‖ > max_norm) = P(χ²_n > (max_norm/σ)²)`。
      ★ 偶数 `n` 时上尾就是有限和（**没有特殊函数**）⇒ 本机只用 `math.exp` 就能算，
        **不需要 `scipy`**（`requirements.txt` 里只有 numpy）。
    """
    if n % 2 != 0:
        raise ValueError("本闭式只对偶数 n 成立（n = %r）" % (n,))
    h = n // 2
    s = 0.0
    term = 1.0
    for k in range(h):
        if k > 0:
            term *= (x / 2.0) / k
        s += term
    return math.exp(-x / 2.0) * s


def trigger_rate_theory(sigma, max_norm, n):
    """`P(‖g‖ > max_norm)` 的闭式（`g ∼ N(0, σ²I_n)`）。"""
    return chi2_sf_even((max_norm / sigma) ** 2, n)


# ===========================================================================
# 4. 工具
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


def make_layers(seed, shapes=None, sigma=1.0):
    """造一批梯度张量（`N(0, σ²)`），可缩放到**恰好**指定的全局范数。"""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s) * sigma for s in (shapes or TENSOR_SHAPES)]


def scale_to_norm(layers, target):
    g = global_norm_from_layers(layers)
    return [np.asarray(l, dtype=np.float64) * (target / g) for l in layers]


def se_rel(p, n):
    """二项比例估计的相对标准误（`A.5` 的 MC 容差）。"""
    return math.sqrt(max(1.0 - p, 1e-12) / (p * n))


# ===========================================================================
# 5. A 类：裁剪的两侧判据 · 监控量 · 触发率闭式
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 裁剪的两侧判据（回链表 A ⑦ + `A.2b`）· **监控的判据** · 触发率闭式")
    print("=" * 74)
    print("  测量条件：seed=%d · ε=%g（[B] PyTorch v2.9.0）· max_norm=%g · 张量 %s"
          % (SEED, specs.TORCH_CLIP_COEF_DENOM_EPS, CLIP_MAX_NORM, TENSOR_SHAPES))

    # ── A.1 ★ 两侧判据（回链 Chunk 1 的表 A ⑦）────────────────────────────
    print("  ── A.1 ★ 裁剪：上侧 `≤ 1.1e-6/max(T,1)` + **下侧 `== ε/(T+ε)`**（**两侧**）──")
    print("     ★★ 判据与阈值定义在 `autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦")
    print("     （**在那里只写一次**）—— **本程序只实现、不重定义**。")
    print("     %-6s %-10s %-16s %-14s %-14s %-14s"
          % ("T", "max_norm", "裁剪后范数", "相对误差", "阈值 1.1e-6/T", "闭式相对偏差"))
    worst_up = 0.0
    worst_cf = 0.0
    for T in CLIP_T_LIST:
        base = scale_to_norm(make_layers(SEED + int(T) * 7), T)
        for max_norm in CLIP_MAX_NORM_LIST:
            clipped, _ = clip_global_norm(base, max_norm)
            new = global_norm_from_layers(clipped)
            rel = abs(new / max_norm - 1.0)
            thr = CLIP_TOL_REL / max(T, 1.0)
            pred = specs.TORCH_CLIP_COEF_DENOM_EPS / (T + specs.TORCH_CLIP_COEF_DENOM_EPS)
            cf = abs(rel / pred - 1.0)
            worst_up = max(worst_up, rel / thr)
            worst_cf = max(worst_cf, cf)
            print("     %-6g %-10g %-16.12f %-14.3e %-14.3e %-14.3e"
                  % (T, max_norm, new, rel, thr, cf))
    check("A.1", worst_up <= 1.0,
          "★ **上侧**：裁剪后范数与 `max_norm` 的相对误差最大只有阈值的 %.4f 倍（≤ 1）—— "
          "`T ≥ 1` 是判据的一部分（`T ∈ %s`）" % (worst_up, CLIP_T_LIST))
    check("A.1b", worst_cf < CLIP_TOL_CLOSED,
          "★★ **下侧（`A.2b`）**：实测相对误差**等于**闭式 `ε/(T+ε)`（最大相对偏差 %.3e < %g）"
          % (worst_cf, CLIP_TOL_CLOSED))
    # ★★ A.1c：T26 的见证 —— 删掉 `+ε` 会让**上侧通过、下侧失败**
    print("  ── A.1c ★★ **T26 的可执行版**：把 `+ε` 删掉的实现 ⇒ 上侧**通过**、下侧**失败** ──")
    T = CLIP_T_LIST[0]
    base = scale_to_norm(make_layers(SEED + 900), T)
    clipped_ok, _ = clip_global_norm(base, CLIP_MAX_NORM)
    clipped_bad, _ = clip_no_eps(base, CLIP_MAX_NORM)
    rel_ok = abs(global_norm_from_layers(clipped_ok) / CLIP_MAX_NORM - 1.0)
    rel_bad = abs(global_norm_from_layers(clipped_bad) / CLIP_MAX_NORM - 1.0)
    up_bad = rel_bad / (CLIP_TOL_REL / max(T, 1.0))
    pred = specs.TORCH_CLIP_COEF_DENOM_EPS / (T + specs.TORCH_CLIP_COEF_DENOM_EPS)
    cf_bad = abs(rel_bad / pred - 1.0)
    print("     正确实现：相对误差 %.6e · 上侧比值 %.4f · 下侧偏差 %.3e" % (rel_ok, worst_up, worst_cf))
    print("     **删掉 `+ε`**：相对误差 **%.3e**（★ 它更精确 —— 比正确实现小 9 个数量级！）"
          "· 上侧比值 **%.4f**（≤ 1 ⇒ 通过）· 下侧偏差 **%.4f**（远超 %g ⇒ 失败）"
          % (rel_bad, up_bad, cf_bad, CLIP_TOL_CLOSED))
    check("A.1c", up_bad <= 1.0 and cf_bad > 0.5 and rel_bad < rel_ok,
          "★★ **为什么判据必须两侧（T26）**：删掉 `+ε` 的实现给出的相对误差是 **%.3e**"
          "（**比正确实现的 %.3e 小 9 个数量级** ⇒ 它更精确）"
          "⇒ **上侧判据反而更满意**（比值 %.4f ≤ 1）、**只有下侧**（`|rel/闭式 − 1| = %.4f`）"
          "能抓住它 —— ★ 这正是 Chunk 1 实测「删掉后 25 条断言全绿」的那个洞"
          % (rel_bad, rel_ok, up_bad, cf_bad))
    # ★ A.1d：T < max_norm ⇒ 不裁剪
    base = scale_to_norm(make_layers(SEED + 91), CLIP_T_SMALL)
    clipped, _ = clip_global_norm(base, CLIP_MAX_NORM)
    check("A.1d", global_norm_from_layers(clipped) == global_norm_from_layers(base)
          and CLIP_T_SMALL < CLIP_MAX_NORM,
          "★ 阴面对照：`T = %g < max_norm = %g` 时**不发生裁剪**（裁剪后范数 == `T`，逐位）⇒ "
          "「裁剪后 == `max_norm`」只在 `T > max_norm` 时成立" % (CLIP_T_SMALL, CLIP_MAX_NORM))

    # ── A.2 ★★ 监控量 = 裁剪【前】的范数 ────────────────────────────────────
    print()
    print("  ── A.2 ★★ 监控量必须是**裁剪前**的范数（[B] 文档的返回值口径）──")
    print("     [B] 逐字：`%s`" % specs.TORCH_CLIP_RETURN_QUOTE)
    pre_trig = 0
    worst_pre = 0.0
    worst_post = 0.0
    norms_seen = []
    for t in range(MON_STEPS):
        layers = make_layers(SEED + 2000 + t, sigma=MON_SIGMA)
        T = global_norm_from_layers(layers)
        norms_seen.append(T)
        clipped, pre = clip_global_norm(layers, CLIP_MAX_NORM)
        post = global_norm_from_layers(clipped)
        if T > CLIP_MAX_NORM:
            pre_trig += 1
            worst_pre = max(worst_pre, abs(T - pre))
        pred = CLIP_MAX_NORM * T / (T + specs.TORCH_CLIP_COEF_DENOM_EPS)
        if T > CLIP_MAX_NORM:
            worst_post = max(worst_post, abs(post / pred - 1.0))
    print("     %d 步里触发裁剪的有 **%d** 步（%.1f%%）· `pre − T` 的逐位最大差 **%.1e**"
          " · `post` 与闭式 `max_norm·T/(T+ε)` 的最大相对差 **%.3e**"
          % (MON_STEPS, pre_trig, pre_trig / MON_STEPS * 100, worst_pre, worst_post))
    check("A.2", pre_trig > 0 and worst_pre == TOL_EXACT and worst_post < CLIP_TOL_CLOSED,
          "★★ **返回的/记录的那个范数就是裁剪前的 `T`**（逐位相等，最大差 %.1e），"
          "而**裁剪后**的范数 == 闭式 `max_norm·T/(T+ε)`（最大相对差 %.3e）—— "
          "★★ 两者**不是同一个量**，而「用哪个做监控」是有后果的（`A.2b`）" % (worst_pre, worst_post))
    # ★★ A.2b：用 post 做监控 ⇒ 永远不触发
    trig_ok = 0
    trig_bad = 0
    max_post = 0.0
    n_elems = sum(a * b for a, b in TENSOR_SHAPES)
    for t in range(MON_STEPS):
        layers = make_layers(SEED + 2000 + t, sigma=MON_SIGMA)
        _, pre = clip_global_norm(layers, CLIP_MAX_NORM)
        _, post = clip_report_post(layers, CLIP_MAX_NORM)
        if monitor_ok_finite(pre, CLIP_MAX_NORM):
            trig_ok += 1
        if monitor_ok_finite(post, CLIP_MAX_NORM):
            trig_bad += 1
        max_post = max(max_post, post)
    print("  ── A.2b ★★ 用 **post** 做监控：阈值 `max_norm` 的触发率 = **%d / %d** ──"
          % (trig_bad, MON_STEPS))
    print("     `pre` 的触发率 = **%d / %d**（%.1f%%）；`post` 的最大值 = **%.12f**（= `max_norm·T/(T+ε)`）"
          % (trig_ok, MON_STEPS, trig_ok / MON_STEPS * 100, max_post))
    spread = (max(norms_seen) - min(norms_seen)) / (sum(norms_seen) / len(norms_seen))
    print("     ★★ `‖g‖` 在 %d 步上的**相对极差只有 %.4f%%** —— 而它的相对**标准误**是闭式 "
          "`1/√(2n) = %.4f%%`（`n = %d`）⇒ 极差 ≈ %.1f 个标准误（%d 个样本的极差本来就该是这个量级）"
          "⇒ 触发率在这个 `n` 上几乎是**阶跃**的"
          % (MON_STEPS, spread * 100, 100 / math.sqrt(2 * n_elems), n_elems,
             spread / (1 / math.sqrt(2 * n_elems)), MON_STEPS))
    check("A.2b", trig_bad == 0 and 0 < trig_ok < MON_STEPS and max_post <= CLIP_MAX_NORM,
          "★★ **监控量的口径会决定判据有没有用**：`post ≤ max_norm` **恒成立** ⇒ 用裁剪后的"
          "范数监控时阈值 `max_norm` **永远不触发**（**假阴性 100%%**，实测 %d/%d）；"
          "而 `pre` 的触发率是 **%.1f%%**（%d/%d，**既不是 0 也不是 1**）⇒ 这是一条**可证伪**"
          "的口径判据，不是措辞偏好" % (trig_bad, MON_STEPS, trig_ok / MON_STEPS * 100,
                                        trig_ok, MON_STEPS))

    # ── A.3 ★★ 非有限值必须用 `isfinite` 判 ────────────────────────────────
    print()
    print("  ── A.3 ★★ 监控 `nan` / `inf`：`>` 与 `isfinite` 的四种组合 ──")
    nan_v = float("nan")
    inf_v = float("inf")
    rows = [("nan", nan_v), ("inf", inf_v), ("1e300", INF_BIG), ("%g" % MON_THRESHOLD, MON_THRESHOLD)]
    okay_gt = []
    okay_fin = []
    for name, v in rows:
        a = monitor_ok(v, MON_THRESHOLD)
        b = monitor_ok_finite(v, MON_THRESHOLD)
        okay_gt.append(a)
        okay_fin.append(b)
        print("     %-8s `> thr` = **%-5s** · `isfinite 且 > thr` = **%-5s**" % (name, a, b))
    check("A.3", okay_gt == [False, True, True, False] and okay_fin == [False, False, True, False]
          and math.isnan(nan_v) and not math.isfinite(nan_v) and math.isfinite(INF_BIG),
          "★★ **用 `>` 写的监控对 `nan` 半失效**：`nan > thr` **恒为 `False`**（静默放行）、"
          "`inf > thr` 是 `True`（会被抓住）；而 `isfinite` 版本**两种都抓**。"
          "⇒ 监控的判据必须写成「**先 `isfinite` 再比阈值**」，否则最该报警的那一类会被吞掉")
    # ★★ A.3b：把 inf 真的灌进裁剪管线
    layers = make_layers(SEED + 31, shapes=((2, 2),))
    flat = np.asarray(layers[0], dtype=np.float64).ravel()
    flat[0] = inf_v * 1.0
    bad = [flat.reshape(2, 2)]
    total = global_norm_from_layers(bad)
    clipped, _ = clip_global_norm(bad, CLIP_MAX_NORM)
    n_nan = int(np.count_nonzero(np.isnan(np.asarray(clipped[0]))))
    n_zero = int(np.count_nonzero(np.asarray(clipped[0]) == 0.0))
    print("     把 **一个 `inf`** 灌进 4 个元素的梯度：`T = %r` ⇒ 裁剪后 `nan` %d 个 · 精确 0 %d 个"
          % (total, n_nan, n_zero))
    check("A.3b", not math.isfinite(total) and n_nan == 1 and n_zero == 3,
          "★★ **接口（回链 `mixed-precision/02`）**：`T = inf` ⇒ `coef = 0` ⇒ 有限元素归 0、"
          "`inf×0 = nan`（`1` 个 `nan` + `3` 个精确 `0`）⇒ **裁剪不能替代溢出处置**"
          "（跳步在 `mixed-precision/02`）；★ 而**监控**这一侧的要求是：判据必须用 `isfinite`"
          "（`A.3`）—— **两件事，不能互相替代**")
    has_nan, has_inf = False, False
    for name, v in rows:
        if not monitor_ok_finite(v, MON_THRESHOLD) and math.isfinite(v) is False:
            if math.isnan(v):
                has_nan = True
            if math.isinf(v):
                has_inf = True
    check("A.3c", has_nan and has_inf,
          "★ 阴面对照：`isfinite` 版本**分别**拒绝了 `nan` 与 `inf`（两类都覆盖）⇒ "
          "`A.3` 不是「只对 `nan` 特殊照顾」")

    # ── A.4 ★★ 全局裁剪 vs 逐张量裁剪 ──────────────────────────────────────
    print()
    print("  ── A.4 ★★ 范数的口径：**全局**（当成一个向量）vs **逐张量** ──")
    print("     [B] 逐字：`%s`" % specs.TORCH_CLIP_SCOPE_QUOTE)
    # ★ 构造：**每一层**的范数都**大于** `max_norm`（否则「未触发的那一层」会破坏 `√L` 闭式）
    layers = [scale_to_norm([l], 2.0 * CLIP_MAX_NORM)[0] for l in make_layers(SEED + 41)]
    before = global_norm_from_layers(layers)
    gl, _ = clip_global_norm(layers, CLIP_MAX_NORM)
    pt = clip_per_tensor(layers, CLIP_MAX_NORM)
    n_gl = global_norm_from_layers(gl)
    n_pt = global_norm_from_layers(pt)
    L = len(layers)
    print("     三层各自先归一到 `2·max_norm`：裁剪前全局范数 %.6f · 全局裁剪后 **%.9f** · "
          "逐张量裁剪后 **%.9f**" % (before, n_gl, n_pt))
    print("     逐张量各层范数：%s（每一层各自压到 `max_norm`）"
          % ["%.6f" % v for v in layer_norms(pt)])
    print("     比值 = **%.6f**（闭式 `√L = √%d = %.6f`）" % (n_pt / n_gl, L, math.sqrt(L)))
    check("A.4", abs(n_gl / CLIP_MAX_NORM - 1.0) < CLIP_TOL_CLOSED
          and abs((n_pt / n_gl) / math.sqrt(L) - 1.0) < CLIP_TOL_REL / 2.0,
          "★★ **两种口径相差 `√L` 倍**（`L = %d` ⇒ **%.6f**，与闭式 `√L = %.6f` 的相对差 "
          "%.1e **全部来自每层各自的 `+ε`**）⇒ "
          "「逐张量各自裁到 `max_norm`」**不是**同一个 `max_norm`：它的全局范数被放大 `√L` 倍。"
          "★ 这条判据**区分了两个实现**（不是措辞差异）"
          % (L, n_pt / n_gl, math.sqrt(L),
             abs((n_pt / n_gl) / math.sqrt(L) - 1.0)))

    # ── A.5 ★★ 触发率的闭式 ────────────────────────────────────────────────
    print()
    print("  ── A.5 ★★ **裁剪触发率**的闭式：`P(χ²_n > (max_norm/σ)²)`（偶数 `n` 的 Erlang 式）──")
    print("     `g ∼ N(0, σ²I_n)`、`n = %d` ⇒ `‖g‖²/σ² ∼ χ²_n`" % N_MON_ELEMS)
    print("     %-8s %-18s %-18s %-14s %-14s"
          % ("σ", "MC 实测触发率", "闭式", "相对差 / 4·SE", "SE"))
    worst_t = 0.0
    rates = []
    for sigma in SIGMA_LIST:
        rng = np.random.default_rng(SEED + 500 + int(sigma * 100))
        g = rng.standard_normal((TRIG_SAMPLES, N_MON_ELEMS)) * sigma
        norms = np.sqrt((g * g).sum(axis=1))
        obs = float((norms > CLIP_MAX_NORM).mean())
        th = trigger_rate_theory(sigma, CLIP_MAX_NORM, N_MON_ELEMS)
        ser = se_rel(th, TRIG_SAMPLES)
        rel = abs(obs / th - 1.0)
        worst_t = max(worst_t, rel / (TRIG_SE_MULT * ser))
        rates.append(obs)
        print("     %-8.2f %-18.5f %-18.5f %-14.4f %-14.4f"
              % (sigma, obs, th, rel / (TRIG_SE_MULT * ser), ser))
    check("A.5", worst_t <= 1.0 and max(rates) / min(rates) > 10.0,
          "★★ **监控可以有一个闭式的预报**：触发率 == `P(χ²_n > (max_norm/σ)²)`"
          "（最大相对差只有 `4·SE` 的 %.3f 倍）；★ 而三个 `σ` 上触发率跨 **%.0f 倍**"
          "（%.4f → %.4f）⇒ 「看裁剪触发率」**真的能看见梯度尺度的变化**"
          % (worst_t, max(rates) / min(rates), max(rates), min(rates)))
    check("A.5b", abs(trigger_rate_theory(1.0, 1.0, 2) - math.exp(-0.5)) < TOL_IDENT,
          "★ **闭式的自检**：`n = 2` 时 `P(χ²_2 > 1) = e^{−1/2}` **逐位**（手算可复核）⇒ "
          "`chi2_sf_even` 不是「一个碰巧给出近似值的函数」")


# ===========================================================================
# 6. B 类：代价模型 —— 监控的字节账、层宽主导、两种范数类型
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：监控的**字节账** · 层宽主导单层范数 · 两种范数类型")
    print("=" * 74)

    # ── B.1 监控只需要每层一个标量 ──────────────────────────────────────────
    print("  ── B.1 ★★ 全局面板可以由 **每层一个标量** 重算（`√Σ ‖g_l‖²`）──")
    layers = make_layers(SEED + 61)
    norms = layer_norms(layers)
    a = global_norm_from_layers(layers)
    b = global_norm_from_scalars(norms)
    c = global_norm_from_scalars(list(reversed(norms)))
    elems = sum(int(np.asarray(g).size) for g in layers)
    print("     直接算（当成一个向量）= %.15f · 由 %d 个标量重算 = %.15f · 逆序重算 = %.15f"
          % (a, len(norms), b, c))
    print("     ★ **记录量**：`L = %d` 个标量与 `Σ n_l = %d` 个元素 ⇒ 比值 **%.2e**"
          % (len(norms), elems, len(norms) / elems))
    check("B.1", abs(a / b - 1.0) < TOL_IDENT and abs(a / c - 1.0) < TOL_IDENT
          and len(norms) < elems,
          "★★ **监控的账**：全局范数可以**只由 `L` 个标量**重算（相对差 < %g），且**与顺序无关**"
          "（逆序给出同一个数）⇒ 记全局面板的成本是 `L = %d` 个数，而梯度本身是 `%d` 个数"
          "（比值 %.1e）—— ★ 这是**纯代数**，与「怎么把它们收集到一起」是两件事"
          "（后者在 `100-parallelism`）" % (TOL_IDENT, len(norms), elems, len(norms) / elems))

    # ── B.2 层宽主导单层范数 ────────────────────────────────────────────────
    print()
    print("  ── B.2 ★ 单层范数被**层宽**主导：`‖g‖ ≈ σ√n` ⇒ 跨层比较必须用每元素 RMS ──")
    print("     ★ 每层抽 `M = %d` 次取均值（★ 单次抽样的 `‖g‖` 相对 sd 是 `1/√(2n)` —— "
          "`n = 64` 时高达 **%.1f%%**，不平均就看不出 `√n` 规律）" % (RMS_DRAWS, 100 / math.sqrt(2 * 64)))
    print("     %-8s %-16s %-16s %-16s" % ("n", "‖g‖（M 次均值）", "‖g‖/√n（RMS）", "σ=1 的闭式 √n"))
    rms_vals = []
    norm_vals = []
    for n in LAYER_N_LIST:
        rng = np.random.default_rng(SEED + 700 + n)
        nn = float(np.sqrt((rng.standard_normal((RMS_DRAWS, n)) ** 2).sum(axis=1)).mean())
        norm_vals.append(nn)
        rms_vals.append(nn / math.sqrt(n))
        print("     %-8d %-16.6f %-16.6f %-16.6f" % (n, nn, nn / math.sqrt(n), math.sqrt(n)))
    spread_rms = (max(rms_vals) - min(rms_vals)) / (sum(rms_vals) / len(rms_vals))
    spread_norm = max(norm_vals) / min(norm_vals)
    print("     ★ `‖g‖/√n` 的极差 **%.4f%%**（< %g）· 而 `‖g‖` 本身跨 **%.4f 倍**"
          "（闭式 `√(4096/64) = 8`）" % (spread_rms * 100, TOL_RMS, spread_norm))
    check("B.2", spread_rms < TOL_RMS and abs(spread_norm / math.sqrt(4096 / 64) - 1.0) < 0.02,
          "★★ **监控的口径错误**：`‖g_l‖` 主要由 `√n_l` 决定（每元素 RMS 跨层极差只有 "
          "%.2f%%）⇒ 直接比较**不同层的范数**会把「**层更宽**」读成「**梯度更大**」"
          "（`‖g‖` 自己跨 %.2f 倍 ≈ `√(4096/64) = 8`）—— ★ 跨层比较必须用**每元素 RMS**"
          % (spread_rms * 100, spread_norm))

    # ── B.3 两种范数类型 ────────────────────────────────────────────────────
    print()
    print("  ── B.3 ★ `norm_type` 的两种口径（默认 `2.0`，可 `inf`）──")
    print("     [B] 文档逐字：`norm_type` 默认 **%g**，`Can be 'inf' for infinity norm`"
          % specs.PYTORCH_CLIP_DEFAULT_NORM_TYPE)
    # ★★ 构造：**两种口径都必须触发**（否则「不裁剪」那一支会把两条闭式都变成同一个数）
    #    ⇒ 需要 `‖g‖_inf > max_norm` **且** `‖g‖_2 > max_norm`
    layers = scale_to_norm(make_layers(SEED + 81), NORM_T_INF)
    n2 = global_norm_from_layers(layers)
    ninf = max(abs(float(x)) for g in layers for x in np.asarray(g).ravel())
    _, _ = clip_by_norm_type(layers, CLIP_MAX_NORM, specs.PYTORCH_CLIP_DEFAULT_NORM_TYPE)
    c2, t2 = clip_by_norm_type(layers, CLIP_MAX_NORM, specs.PYTORCH_CLIP_DEFAULT_NORM_TYPE)
    cinf, tinf = clip_by_norm_type(layers, CLIP_MAX_NORM, float("inf"))
    max_after_2 = max(abs(float(x)) for g in c2 for x in np.asarray(g).ravel())
    max_after_inf = max(abs(float(x)) for g in cinf for x in np.asarray(g).ravel())
    print("     裁剪前：`‖g‖_2 = %.6f > max_norm` · `‖g‖_inf = %.6f > max_norm` ⇒ **两种口径都触发**"
          % (n2, ninf))
    print("     `‖g‖_2 = %.6f` · `‖g‖_inf = %.6f` · 夹逼 `‖·‖_inf ≤ ‖·‖_2 ≤ √n·‖·‖_inf` :"
          " %.1f ≤ %.1f ≤ %.1f" % (n2, ninf, ninf, n2,
                                   math.sqrt(sum(int(np.asarray(g).size) for g in layers)) * ninf))
    n_elems = sum(int(np.asarray(g).size) for g in layers)
    asym = math.sqrt(2.0 * math.log(n_elems) / n_elems)
    closed_2 = CLIP_MAX_NORM * ninf / (t2 + specs.TORCH_CLIP_COEF_DENOM_EPS)
    closed_inf = CLIP_MAX_NORM * ninf / (tinf + specs.TORCH_CLIP_COEF_DENOM_EPS)
    print("     `norm_type=2`：`coef = %.9f` ⇒ 最大元素 **%.9f**（闭式 `a·‖g‖_inf/(‖g‖_2+ε)`"
          " = %.9f）" % (CLIP_MAX_NORM / (t2 + specs.TORCH_CLIP_COEF_DENOM_EPS), max_after_2, closed_2))
    print("     `norm_type=inf`：`coef = %.9f` ⇒ 最大元素 **%.9f**（**逐元素**上界 = `max_norm`）"
          % (CLIP_MAX_NORM / (tinf + specs.TORCH_CLIP_COEF_DENOM_EPS), max_after_inf))
    print("     ★★ `‖g‖_inf/‖g‖_2` = **%.6f**，而 Gaussian 的渐近 `√(2 ln n / n) = %.6f`"
          "（`n = %d`）⇒ 2 的口径下**单元素离 `max_norm` 还差 %.0f 倍**"
          % (ninf / n2, asym, n_elems, CLIP_MAX_NORM / max_after_2))
    check("B.3", ninf > CLIP_MAX_NORM and n2 > CLIP_MAX_NORM
          and ninf <= n2 + TOL_CHEB and n2 <= math.sqrt(n_elems) * ninf + TOL_CHEB
          and abs(max_after_inf / closed_inf - 1.0) < CLIP_TOL_CLOSED
          and abs(max_after_2 / closed_2 - 1.0) < CLIP_TOL_CLOSED
          and abs((ninf / n2) / asym - 1.0) < 0.3,
          "★★ **同一个 `max_norm` 在两种口径下的含义差在【逐元素】那一侧**："
          "`norm_type=inf` 把**每个元素**都压到 `max_norm`（实测 **%.9f**，逐位等于闭式）；"
          "而 `norm_type=2`（**默认**）只保证**全局**范数 ≤ `max_norm` —— 单个元素只有 "
          "**%.6f**，是 `max_norm` 的 **%.1f%%**（闭式 `a·‖g‖_inf/(‖g‖_2+ε)` 逐位吻合）。"
          "★ 而 `‖g‖_inf/‖g‖_2` 随 `n` 走 `√(2 ln n / n)`（`n = %d` 上 %.6f vs 渐近 %.6f，"
          "差 %.0f%%）⇒ **2 口径的保护强度取决于 `n`**，不是一个固定的上界"
          % (max_after_inf, max_after_2, max_after_2 / CLIP_MAX_NORM * 100, n_elems,
             ninf / n2, asym, abs((ninf / n2) / asym - 1.0) * 100))


# ===========================================================================
# 7. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：静默停训 · 默认的 `error_if_nonfinite` · 全 0 层 · 源码级次序")
    print("=" * 74)

    # ── C.1 ★★ `max_norm = 0` ⇒ 静默停训，而监控看不到 ─────────────────────
    print("  ── C.1 ★★ `max_norm = 0`：更新为 0，而**监控（看 pre）说「正常」** ──")
    layers = make_layers(SEED + 91, sigma=MON_SIGMA)
    dw, pre = train_step_with_monitor(layers, 0.0, LR)
    post = global_norm_from_layers([-d / LR for d in dw]) if dw else 0.0
    dw_norm = math.sqrt(sum(float((np.asarray(d) ** 2).sum()) for d in dw))
    print("     裁剪前范数 `pre = %.6f`（**监控看到的是这个数**）· 梯度全清零（`post = %.1f`）"
          " · `‖Δw‖ = %.1f`" % (pre, post, dw_norm))
    alarm = monitor_ok_finite(pre, MON_THRESHOLD)
    print("     监控（看 `pre`，阈值 %g）会不会报警：**%s**（`pre = %.6f` 是个**平平无奇**的数）"
          % (MON_THRESHOLD, alarm, pre))
    check("C.1", math.isfinite(pre) and pre > 0.0 and post == 0.0 and dw_norm == 0.0
          and not alarm,
          "★★ **「监控量取 pre」是双刃的**：`max_norm = 0` 时训练**完全停住**"
          "（梯度范数精确 `0`、`‖Δw‖` 精确 `0`），而监控看的是**裁剪前**的范数 —— "
          "它看到的是 **%.6f**（一个平平无奇的有限值），**报警不响**（阈值 %g）⇒ "
          "这个失效模式**从「看 pre」的监控上看不见**。"
          "⇒ 范数监控必须**同时**看**裁剪后**的量（或 `‖Δw‖`），而且那里需要的是**下界**"
          % (pre, MON_THRESHOLD))

    # ── C.2 ★★ `error_if_nonfinite` 的默认值 ───────────────────────────────
    print()
    print("  ── C.2 ★★ `error_if_nonfinite` 的默认值是 `False` ⇒ 非有限值**默认静默通过** ──")
    raised_default = None
    raised_checked = None
    try:
        clip_checked = clip_with_flag(make_layers(SEED + 93, shapes=((2, 2),)),
                                      CLIP_MAX_NORM, specs.PYTORCH_CLIP_ERROR_IF_NONFINITE_DEFAULT)
        raised_default = "没有抛异常"
    except ValueError as e:
        raised_default = "ValueError: %s" % e
    try:
        clip_with_flag([np.array([[float("inf"), 1.0], [1.0, 1.0]])], CLIP_MAX_NORM, True)
        raised_checked = "没有抛异常"
    except ValueError as e:
        raised_checked = "ValueError: %s" % e
    print("     默认（`error_if_nonfinite=%r`）在**干净**梯度上：%s"
          % (specs.PYTORCH_CLIP_ERROR_IF_NONFINITE_DEFAULT, raised_default))
    print("     默认值在**含 `inf`** 的梯度上：%s"
          % ("**静默通过**（`total_norm = inf` 不报错）"
             if not specs.PYTORCH_CLIP_ERROR_IF_NONFINITE_DEFAULT else "会报错"))
    print("     显式传 `True` 时：%s" % raised_checked)
    clean_ok = True
    try:
        clip_with_flag([np.array([[float("inf"), 1.0], [1.0, 1.0]])], CLIP_MAX_NORM, False)
    except ValueError:
        clean_ok = False
    check("C.2", specs.PYTORCH_CLIP_ERROR_IF_NONFINITE_DEFAULT is False
          and "没有抛异常" == raised_default and clean_ok
          and raised_checked.startswith("ValueError"),
          "★★ **默认值本身就是一条工程结论**（[B] 逐字：`if True, an error is thrown if the "
          "total norm of the gradients from parameters is nan , inf , or -inf . **Default: "
          "False**`）⇒ 默认管线下 `inf`/`nan` **不报错**、静默地走完裁剪与更新；"
          "★ 本程序按这条规格实现了两支：`False` ⇒ 不报错、`True` ⇒ 抛异常（实测）")

    # ── C.3 ★ 全 0 的层：全局范数不会 nan，逐张量归一化会 ──────────────────
    print()
    print("  ── C.3 ★ 一个**全 0 的层**：全局范数相安无事，而**逐层归一化**给 `nan` ──")
    layers = make_layers(SEED + 95, shapes=((4, 4), (4, 4)), sigma=1.0)
    layers.append(np.zeros((4, 4)))
    gn = global_norm_from_layers(layers)
    clipped, _ = clip_global_norm(layers, CLIP_MAX_NORM)
    n_nan_global = int(sum(np.count_nonzero(np.isnan(np.asarray(g))) for g in clipped))
    per_norm = []
    for g in layers:
        a = np.asarray(g, dtype=np.float64)
        n = math.sqrt(float((a * a).sum()))
        per_norm.append(float(np.asarray(a / n).ravel()[0]) if n > 0 else float("nan"))
    print("     三层（第三层全 0）：全局范数 = **%.6f**（有限、不受影响）"
          " · 全局裁剪后的 `nan` 个数 = **%d**" % (gn, n_nan_global))
    print("     而**逐层归一化** `g/‖g‖` 在第三层上给出 **%r**（`0/0`）" % per_norm[2])
    check("C.3", math.isfinite(gn) and n_nan_global == 0 and math.isnan(per_norm[2])
          and abs(per_norm[0]) <= 1.0 + TOL_TIGHT,
          "★★ **一个静默 `nan` 的入口**：全 0 的层在**全局口径**下无害（范数有限、裁剪后无 "
          "`nan`），但任何**逐层归一化**（`g/‖g‖`）都会在那里 `0/0 = nan` ⇒ "
          "污染随后全部参数。★ 这正是「口径选错」的代价：**同样的数据，全局口径安全、"
          "逐张量口径中毒**")

    # ── C.4 ★★ 源码级次序钉住（T41）─────────────────────────────────────────
    print()
    print("  ── C.4 ★★ **源码级次序**：监控必须在裁剪**之前** ──")
    src_ok = inspect.getsource(train_step_with_monitor)
    i_mon = src_ok.index("global_norm_from_layers")
    i_clip = src_ok.index("clip_global_norm")
    i_lr = src_ok.index("-lr * ")
    print("     `train_step_with_monitor` 源码索引：监控(`global_norm_from_layers`) @%d → "
          "裁剪 @%d → `-lr *` @%d" % (i_mon, i_clip, i_lr))
    # ★ 见证：不触发裁剪那一档上，两种次序**逐位相同**
    layers = scale_to_norm(make_layers(SEED + 97), CLIP_T_SMALL)
    dw_a, n_a = train_step_with_monitor(layers, CLIP_MAX_NORM, LR)
    dw_b, n_b = train_step_monitor_after_clip(layers, CLIP_MAX_NORM, LR)
    same = max(abs(float(x - y)) for d1, d2 in zip(dw_a, dw_b)
               for x, y in zip(np.asarray(d1).ravel(), np.asarray(d2).ravel()))
    layers_big = scale_to_norm(make_layers(SEED + 97), NORM_T)
    dw_c, n_c = train_step_with_monitor(layers_big, CLIP_MAX_NORM, LR)
    dw_d, n_d = train_step_monitor_after_clip(layers_big, CLIP_MAX_NORM, LR)
    print("     `T = %.2f < max_norm`（**不触发**）：两种次序的范数 `%.6f` vs `%.6f`（**逐位相同**）"
          % (CLIP_T_SMALL, n_a, n_b))
    print("     `T = %.1f > max_norm`（**触发**）：`%.6f` vs `%.6f`（相差 %.1f 倍）"
          % (NORM_T, n_c, n_d, n_c / n_d))
    check("C.4", i_mon < i_clip < i_lr and n_a == n_b and n_c != n_d,
          "★★ **为什么必须是结构钉住（T41）**：在 `T < max_norm`（**不触发裁剪**）那一档上，"
          "把监控挪到裁剪**之后**给出**逐位相同**的范数（`%.6f` vs `%.6f`）⇒ "
          "**数值断言抓不到这个错序**；而触发档上它给出 de 数（`%.6f` vs `%.6f`）。"
          "⇒ 次序被**钉在源码**上（`%d < %d < %d`）" % (n_a, n_b, n_c, n_d, i_mon, i_clip, i_lr))


def clip_with_flag(layers, max_norm, error_if_nonfinite,
                   eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """按 [B] 文档的 `error_if_nonfinite` 规格实现的两支（`C.2` 的被测对象）。

    ★ [B] 逐字：`if True, an error is thrown if the total norm of the gradients from
      parameters is nan , inf , or -inf . Default: False`
    """
    total = global_norm_from_layers(layers)
    if error_if_nonfinite and not math.isfinite(total):
        raise ValueError("梯度总范数不是有限值（total_norm = %r）" % (total,))
    coef = max_norm / (total + eps)
    if coef < 1.0:
        return [np.asarray(g, dtype=np.float64) * coef for g in layers], total
    return [np.asarray(g, dtype=np.float64) for g in layers], total


# ===========================================================================
# 8. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ **两处回链**（不重推）：裁剪的判据与阈值在 "
          "`autograd-and-backward/03` 的表 A ⑦（**含 `A.2b`**，T26）；"
          "`unscale` 的时机/溢出跳步在 `mixed-precision/02`。")
    print("  ★★ **禁区**：`LR warmup` 的机制解释在 `80-…/normalization/02:83-100`；"
          "范数的分片与通信量在 `100-parallelism/README.md:14`。")
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

    measured = {"SEED": SEED, "CLIP_T_LIST": CLIP_T_LIST, "CLIP_MAX_NORM": CLIP_MAX_NORM,
                "CLIP_MAX_NORM_LIST": CLIP_MAX_NORM_LIST, "TENSOR_SHAPES": TENSOR_SHAPES,
                "MON_STEPS": MON_STEPS, "MON_SIGMA": MON_SIGMA,
                "N_MON_ELEMS": N_MON_ELEMS, "SIGMA_LIST": SIGMA_LIST,
                "TRIG_SAMPLES": TRIG_SAMPLES, "NORM_T": NORM_T, "NORM_T_INF": NORM_T_INF, "LR": LR,
                "LAYER_N_LIST": LAYER_N_LIST, "MON_THRESHOLD": MON_THRESHOLD,
                "CLIP_T_SMALL": CLIP_T_SMALL, "RMS_DRAWS": RMS_DRAWS, "INF_BIG": INF_BIG}
    expect = {"SEED": 20260951, "CLIP_T_LIST": (2.0, 4.0, 8.0), "CLIP_MAX_NORM": 1.0,
              "CLIP_MAX_NORM_LIST": (1.0, 0.5, 0.1),
              "TENSOR_SHAPES": ((64, 64), (128, 32), (256, 256)),
              "MON_STEPS": 200, "MON_SIGMA": 0.0037, "N_MON_ELEMS": 8,
              "SIGMA_LIST": (0.2, 0.4, 0.8), "TRIG_SAMPLES": 200000, "NORM_T": 10.0,
              "NORM_T_INF": 100.0,
              "LR": 0.1, "LAYER_N_LIST": (64, 256, 1024, 4096), "MON_THRESHOLD": 3.0,
              "CLIP_T_SMALL": 0.5, "RMS_DRAWS": 400, "INF_BIG": 1.0e15}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 18 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", min(CLIP_T_LIST) >= 1.0 and CLIP_T_SMALL < CLIP_MAX_NORM
          and all(T > CLIP_MAX_NORM for T in CLIP_T_LIST)
          and len(CLIP_MAX_NORM_LIST) == 3,
          "★ **`T ≥ 1` 与 `T > max_norm` 这两条取值范围**成立（`T ∈ %s`、`max_norm ∈ %s`）—— "
          "它们是判据的一部分（**回链 Chunk 1 的 C.3**）" % (CLIP_T_LIST, CLIP_MAX_NORM_LIST))
    check("D.3c", N_MON_ELEMS % 2 == 0 and N_MON_ELEMS > 0,
          "★ `A.5` 的触发率闭式**只对偶数 `n` 成立**（`n = %d`，`chi2_sf_even` 会对奇数抛异常）"
          "—— 这个取值范围是判据的一部分" % N_MON_ELEMS)
    n_elems = sum(a * b for a, b in TENSOR_SHAPES)
    check("D.3d", abs(MON_SIGMA * math.sqrt(n_elems) / CLIP_MAX_NORM - 1.0) < 0.5
          and min(SIGMA_LIST) * math.sqrt(N_MON_ELEMS) < CLIP_MAX_NORM
          < max(SIGMA_LIST) * math.sqrt(N_MON_ELEMS),
          "★★ 两处尺度都是**故意卡在阈值附近**的：`A.2`/`A.2b` 的 `σ√n = %.4f` "
          "（`max_norm = %g`）⇒ 触发率**既不是 0 也不是 1**（否则「假阴性 100%%」不可测）；"
          "★ `A.5` 的三个 `σ` **跨过**阈值（`σ√n` 从 %.2f 到 %.2f）⇒ 触发率跨一个数量级"
          % (MON_SIGMA * math.sqrt(n_elems), CLIP_MAX_NORM,
             min(SIGMA_LIST) * math.sqrt(N_MON_ELEMS), max(SIGMA_LIST) * math.sqrt(N_MON_ELEMS)))
    check("D.3e", NORM_T > CLIP_MAX_NORM and CLIP_T_SMALL < CLIP_MAX_NORM,
          "★ `B.3`/`C.4` 的两个档位：`NORM_T = %g > max_norm`（**触发**）与 "
          "`CLIP_T_SMALL = %g < max_norm`（**不触发**）—— 后者正是 `C.4` 里"
          "「数值上测不出错序」的那一档" % (NORM_T, CLIP_T_SMALL))

    th = {"CLIP_TOL_REL": CLIP_TOL_REL, "CLIP_TOL_CLOSED": CLIP_TOL_CLOSED,
          "TOL_EXACT": TOL_EXACT, "TOL_TIGHT": TOL_TIGHT, "TOL_IDENT": TOL_IDENT,
          "TOL_MEASURABLE": TOL_MEASURABLE, "TOL_RMS": TOL_RMS, "TOL_CHEB": TOL_CHEB,
          "TRIG_SE_MULT": TRIG_SE_MULT}
    th_expect = {"CLIP_TOL_REL": 1.1e-6, "CLIP_TOL_CLOSED": 1e-6, "TOL_EXACT": 0.0,
                 "TOL_TIGHT": 1e-12, "TOL_IDENT": 1e-15, "TOL_MEASURABLE": 1e-2,
                 "TOL_RMS": 0.02, "TOL_CHEB": 1e-9, "TRIG_SE_MULT": 4.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 9 个逐位一致（含**两个 0 容差**与 `CLIP_TOL_CLOSED` —— "
                            "它就是 `A.2b` 的下侧，**回链 Chunk 1 的 T26**）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


# ===========================================================================
# 9. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", specs.TORCH_CLIP_COEF_DENOM_EPS == 1e-6,
          "★★ 裁剪分母的保护项 `ε == 1e-6`（[B] **PyTorch v2.9.0 的源码**，不是规范）——"
          "★ 本程序**只消费、不重定义**：判据与阈值（含 `T ≥ 1`、上侧 `1.1e-6/max(T,1)`、"
          "下侧 `== ε/(T+ε)` 的 `A.2b`）在 `autograd-and-backward/03` 的**表 A ⑦** 写一次")
    check("D.6", specs.TORCH_CLIP_RETURN_QUOTE
          == "Total norm of the parameter gradients (viewed as a single vector)"
          and specs.TORCH_CLIP_SCOPE_QUOTE
          == "as if the norms of the individual gradients were concatenated into a single vector.",
          "★★ **监控量的口径依据**：① 返回值 = **裁剪前**的范数（`A.2`）；"
          "② 范围 = **全局**（`A.4`）—— 两条都是 [B] 文档的逐字（**不是规范**）")
    check("D.7", specs.PYTORCH_CLIP_DEFAULT_NORM_TYPE == 2.0
          and specs.PYTORCH_CLIP_ERROR_IF_NONFINITE_DEFAULT is False,
          "★★ 两个**默认值**逐位钉住：`norm_type = 2.0`、`error_if_nonfinite = False` —— "
          "★ 后者是 `C.2` 的核心（`inf`/`nan` **默认静默通过**）；"
          "★ 它是 **bool**（不是 `None`）⇒ `mutate.py` 会把它翻转、断言当场失败（**盲区⑤**）")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(global_norm_from_layers)
    src_b = inspect.getsource(global_norm_concat)
    src_c = inspect.getsource(global_norm_from_scalars)
    check("D.8", len({src_a, src_b, src_c}) == 3,
          "★ 防 P17：被测（`global_norm_from_layers` %d 字符）、参照（`global_norm_concat` %d）、"
          "参照 2（`global_norm_from_scalars` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c)))
    watched = {"global_norm_from_layers", "clip_global_norm", "layer_norms",
               "clip_by_norm_type", "clip_per_tensor"}
    called = _callee_names(global_norm_concat) | _callee_names(global_norm_from_scalars)
    check("D.9", not (watched & called),
          "★ **AST 调用表**：两条**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    layers = make_layers(SEED + 111)
    check("D.9b", abs(global_norm_from_layers(layers) / global_norm_concat(layers) - 1.0) < TOL_IDENT
          and abs(global_norm_from_layers(layers)
                  / global_norm_from_scalars(layer_norms(layers)) - 1.0) < TOL_IDENT,
          "★ 三条路径**逐位一致**（相对差 < %g）⇒ `D.8` 的「源码两两不同」不是装饰"
          % TOL_IDENT)
    check("D.10", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.11", specs.UNSOURCED == ("caffe_xavier_term_origin",
                                      "glorot_normalized_init_count",
                                      "pytorch_init_bound_formula_raw_count",
                                      "pytorch_conv_fan_definition",
                                      "mup_learning_rate_table_scaling",
                                      "spike_mechanism_one_source_conclusion",
                                      "snm_peer_review_status_recheck"),
          "★ `specs.UNSOURCED` 的 %d 项**逐位钉住**" % len(specs.UNSOURCED))
    check("D.12", specs.HE_CONNECTIONS_QUOTE == "the number of connections"
          and specs.GLOROT_XAVIER_TERM_COUNT == 0,
          "★ 本程序**不引** He 的方差公式与 Xavier 的名字（那是 `01` 的正文），"
          "但把两条**同一份 `specs.py`** 的常量钉住 ⇒ 三个程序共享的事实源是**一致**的")


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
    print("02_clip_monitor.py —— 梯度裁剪 与 **范数监控的判据**")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ **本程序的新内容 = 监控的判据**（裁剪本身**回链**、只实现不重定义）。")
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
