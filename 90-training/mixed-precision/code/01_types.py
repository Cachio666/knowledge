#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_types.py —— **主权重 / 梯度 / 优化器状态 / loss 四者的类型选择表**。

配套文档：`01-type-selection.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ 本程序是**上游指名让渡**的那一件事的落点：
> `10-foundations/numerical-linear-algebra/04-iterative-methods.md:386` 逐字写着
> 「**主权重、优化器状态、loss scaling 三者的类型选择**」由 `90-training/mixed-precision` 承担。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ 本主题的**禁区**（`CONTRIBUTING` §7 + 域 README 的边界表；**本主题禁区很大**）：
  · 「**为什么 fp16 要 loss scaling、bf16 为什么不用**」**整块** →
    `10-foundations/floating-point/04-low-precision-formats.md:24,28,41,65,189,195-201,211,219-231,303`；
  · **三个后果的分解**（范围 / 精度 / 累加宽度）→ 同上 `:24,303`；
  · **格式的位分配**（fp16 选精度、bf16 选范围）→ `10-foundations/floating-point/01-ieee754-format.md:189-226`；
  · **累加器宽度的误差界**（fp32 → fp16 放大 8192 倍、split-K 的顺序不确定性）→
    `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md:118-126,304-306,326`；
  · **主权重「为什么必须 fp32」的数值理由** → `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266`；
  · **fp8 的位分配与四个变体**、**scale 的作用域与 bit 开销** → 见 `03_fp8.py` 的抬头。
  ⇒ 本程序**一行都不重写**这些；它只做一件事：把「**哪个张量用什么类型**」落成一张
    **可复算的表**，并给每一行配一条**能证伪**的判据。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  更新被吞掉的临界 `Δw` == `ULP/2` —— 用**两条独立路径**（二分法 / `nextafter`）取到
           同一个数（相对差 < 1e-3）；把二分的方向写反立刻失败；
  · `A.2`  fp16 的数值边界 —— `numpy.finfo` 与**按 IEEE 定义算**两条路径逐位一致；
  · `A.5`  下溢比例的**闭式**与**蒙特卡洛**互验（相对差 < 5%）；把闭式的阈值从 `thr/2`
           改成 `thr` 立刻失败（那正是 `probes/README.md` 记的那处旧错）；
  · `B.1`  临界比值 `|w| / Δw_crit` ∈ `[2048, 4096)`，且在 2 的幂处**精确 2048** ——
           与 Micikevicius 2017 逐字 `at least 2048 times` 对齐；
  · `C.1`  **直接拿 fp16 当权重**时 2000 步更新**全被吞**、权重恒为 `1.000000`（失败行为真的成立）。

用法：
    python3 01_types.py
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
#   ★ 这个字面量**不是**照抄任何别的主题的：域 README 把五个主题的标记写死为
#     五个不同形态（本主题用 `::`），**不跨主题照抄**是本库纪律。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ":: FAIL: mixed-precision"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", "++ FAIL: optimizers",
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
SEED = 20260925              # ★ 与探测脚本的 `default_rng(90)` **不同**（换种子是本库的常规）
BISECT_STEPS = 80            # 「更新被吞」的临界 Δw 用**二分**求（独立于 `nextafter` 的一条路）
# ★ `|w|` 的扫描表：**全正数**存放 —— 负字面量是 `UnaryOp(USub, Constant(…))`，
#   `mutate.py` 的 `_walk` 对它返回空 ⇒ **不进分母**（计划 §3 开头的盲区②）。
# ★★ **判据① 的扫描表必须由「fp16 里恰好可表示」的值组成**：`1 / 10 / 100 / 1000`
#   都是 fp16 的格点。**理由是本程序实测出来的**（`probes/README.md` 也记了同一件事）：
#   `0.01` 与 `0.1` **不在**格点上，它们的临界 Δw **小于** `ULP/2` ——
#   对 `w=0.1` 实测比值是 **0.200000**、对 `w=0.01` 是 **1.560000**。
#   ⇒ 那两行不是「判据失败」，而是**另一个量**（`ULP/2` 减去 `w` 到格点的距离），见 `A.1b`。
W_GRID = (1.0, 10.0, 100.0, 1000.0)
# ★ 对照组：**不在** fp16 格点上的两个值（`A.1b` 用闭式解释它们的比值为什么不是 1）。
W_OFFGRID = (0.01, 0.1)
# ★ 其中**恰为 2 的幂**的那几个：B.1 要断言 `|w|/Δw_crit` **精确等于 2048**（只在它们上成立）。
W_POW2 = (1.0, 2.0, 4.0, 8.0)
W_NONPOW2 = (10.0, 100.0, 1000.0)   # 对照：它们给 2560 / 3200 / 4000（**都 ≥ 2048**）

LR = 1e-3                    # 更新轨迹的学习率
GRAD = 1e-3                  # 每步的（恒定）梯度
W0 = 1.0                     # 初始权重
STEPS = 2000                 # 轨迹长度（判据②）
TRACE_SHOW = (10, 100, 500, 2000)   # 轨迹里打印的四个检查点

SIGMAS = (1e-6, 1e-5, 1e-4, 1e-3)   # 下溢比例的 σ 扫描表（判据⑥）
SIGMA_JUDGED = 3             # ★ **只有前 3 个进判据** —— 见下 `MIN_EXPECTED_HITS` 的理由
MC_N = 2000000               # 蒙特卡洛样本数（每个 σ）

# ── 判据④的三个动态范围（**同口径**，各带 ±0.05 容差）────────────────────────
# ★★ 计划 §0.7-D 的 **D4b**：原稿写「差 ≈ 7.5 个数量级」**是自造真值**
#    （`76.46 − 12.04 = 64.42`，而且那两个数**不是同一个口径**）⇒ 本程序用**同口径的三个数**，
#    并**只对这三个数**下断言（不写「差多少个数量级」这种混口径的话）。
RANGE_FP16_NORMAL = 9.03     # log10(65504 / 2^-14)   —— fp16 **正规区间**
RANGE_FP16_ALL = 12.04       # log10(65504 / 2^-24)   —— fp16 **含次正规**
RANGE_FP32_NORMAL = 76.46    # log10(3.40e38 / 1.18e-38) —— fp32 **正规区间**
TOL_RANGE = 0.05             # 三个数的容差（计划 §3 表 A ④）

# ── 其余阈值 ───────────────────────────────────────────────────────────────
TOL_ULP_RATIO = 1e-3         # A.1：实测临界 / (ULP/2) 与 1 的相对差（计划 §3 表 A ①）
TOL_BOUNDARY_REL = 1e-12     # A.2：两条路径算 fp16 边界（机器精度档）
TOL_RANGE_CROSS = 1e-6       # A.3：两个来源算三个动态范围（同一批推导，两条来源）
TOL_BF16_RANGE_REL = 1e-4    # A.4：bf16/fp32 量程比与 1 的偏差上界（★ 见 A.4 的说明）
TOL_UNDERFLOW_REL = 0.05     # A.5：MC 与闭式的相对差（计划 §0.1 订正后的判据）
MIN_EXPECTED_HITS = 300      # ★★ A.5 的**小 n 门槛**（P22 的落法：打印参照量级）
RATIO_MIN = 2048             # B.1：外部锚点（Micikevicius 逐字 `at least 2048 times`）
RATIO_MAX = 4096             # B.1：上界（比值 = 2·|w|/ULP，落在 [2^11, 2^12)）
TOL_MASTER_VS_CLOSED = 1e-4  # C.1：主权重 fp32 与闭式的差（计划 §3 表 A ②）
SWALLOW_TOLERANCE = 0.95     # C.2 阴性对照：更新量放大到 ULP 之上后，被吞比例必须 < 这个
# ★ C.2 的倍数取 **1024 = 2^10**：更新量 `lr·g = 1e-6` 乘 1024 得 `1.024e-3`，
#   **刚越过** `|w|=1` 处的 `ULP/2 = 4.883e-4`。
#   ★ 取 4 倍是**不够的**（`4e-6` 仍远小于 `ULP/2`，被吞比例还是 1.0）—— 本程序第一次跑就撞上了。
#   ★ 顺带：`1024` 与 `02_amp.py` 的 loss scaling 用的倍率是同一个数，**这不是巧合**
#     （把更新量整体乘大 = loss scaling 干的事），但**两处的常量各自独立**，
#     不是一个跨程序的共享模块（`CONTRIBUTING:36`：`code/` 必须自包含）。
UPDATE_FACTOR = 1024.0

# ── B.2 的字节账（每个参数、每种类型选择）──────────────────────────────────
# ★ 口径：**只数「一个参数」在这四种类型选择下各占几字节**，
#   不数 RNG / dataloader / 分片头（那些的正文在 `90-training/checkpointing/01-state-and-bytes`
#   与 `100-parallelism`）。
ITEMSIZE_FP32 = 4            # = np.dtype(np.float32).itemsize（D.5 里与真 dtype 对拍）
ITEMSIZE_FP16 = 2            # = np.dtype(np.float16).itemsize
MIXED_MASTER_BYTES = 4       # 混合精度：fp32 主权重
MIXED_COPY_BYTES = 2         # 混合精度：**多出来的**那份 fp16 计算副本（全 fp32 口径里没有它）
MIXED_GRAD_BYTES = 2         # 混合精度：fp16 梯度
MIXED_OPT_BYTES = 8          # 优化器状态（一阶矩 m + 二阶矩 v，各 4）—— 两列**相同**
FP32_WEIGHT_BYTES = 4        # 全 fp32：参数本身（对应混合精度那两行）
FP32_GRAD_BYTES = 4          # 全 fp32：梯度
FP32_OPT_BYTES = 8           # 全 fp32：优化器状态
CALLS = {}                   # ★ 参照实现独立性的**证据**：被测实现的函数每被调一次就 +1


# ===========================================================================
# 2. 路径 1：**从 `numpy.finfo` 读**（一个实现的表）
# ===========================================================================
def boundary_from_finfo(dtype):
    """路径 1：把边界值**读**出来。返回 `(max, 最小次正规, 最小正规, eps)`。

    ★ 这是「某实现的表」—— 它**不是**定义。定义在 IEEE 754 里，见路径 2。
    """
    fi = np.finfo(dtype)
    sub = float(np.nextafter(dtype(0), dtype(1)))
    return float(fi.max), sub, float(fi.tiny), float(fi.eps)


# ===========================================================================
# 3. 路径 2：**按 IEEE 754 的位参数算**（不读任何实现的表）
# ===========================================================================
def boundary_from_definition(exp_bits, man_bits, bias):
    """路径 2：只由 `(E, M, bias)` 三个位参数算出四个边界值。

    ```
    field_max       = 2^E − 1             全 1 的指数域
    emax            = field_max − 1 − bias      ← ★ 「−1」是因为全 1 指数域留给 inf/NaN
    max_finite      = (2 − 2^−M) · 2^emax
    最小正规数       = 2^(1 − bias)
    最小次正规数     = 2^(1 − bias − M)
    eps             = 2^−M
    ```

    ★★ **本函数第一版把这个式子写错过**（实在的教训，留在这里）：
      写成 `max_finite = (2 − 2^−M) · 2^(emax − bias)` 时，fp16 给 `1.999…` 而不是 `65504` ——
      因为**偏置只作用一次**：`emax` 已经是**偏置后**的指数，再减 `bias` 就减了第二次。
      fp32 那一列会**看不出错**（`bias = 127`、`emax − bias = 0`，而正确的 `emax = 127`，
      只差在「乘 `2^0` 还是乘 `2^127`」——**两者都算出个像样的数**）⇒
      **这正是「用两条路径对拍」的价值**：单看一条路径，错的那个式子照样「跑得通」。

    ★★ 这条路径**不调用路径 1 的任何东西**（`D.6`/`D.7` 用调用计数与 AST 调用表两重保险查它）。
    ★ 本函数**不处理** fp8，而且**不是**「忘了」：它的关键前提是
      「**全 1 指数域留给 inf/NaN**」，而 `e4m3fn` **没有 inf**、全 1 指数域仍是有效指数
      ⇒ 「按 IEEE 通式算」在 fp8 上**就是错的**。那是位分配，正文在
      `130-quantization/numerical-formats/01-fp8-family.md` §0，**本程序回链、不算**。
    """
    field_max = (1 << exp_bits) - 1
    emax = field_max - 1 - bias
    max_finite = (2.0 - 2.0 ** (-man_bits)) * 2.0 ** emax
    min_normal = 2.0 ** (1 - bias)
    min_subnormal = 2.0 ** (1 - bias - man_bits)
    eps = 2.0 ** (-man_bits)
    return max_finite, min_subnormal, min_normal, eps


# ===========================================================================
# 4. 临界 Δw：**两条独立路径**（判据① 的分子与分母各来自一条）
# ===========================================================================
def ulp_by_nextafter(w, dtype=np.float16):
    """路径 A：`ULP(w)` = 「下一个可表示数」减自己。（一条路：`nextafter`，带调用计数）"""
    CALLS["ulp_by_nextafter"] = CALLS.get("ulp_by_nextafter", 0) + 1
    x = dtype(w)
    nxt = np.nextafter(x, dtype(np.inf))
    return abs(float(nxt) - float(x))


def crit_dw_by_bisection(w, steps=BISECT_STEPS, dtype=np.float16):
    """路径 A：**二分**求最小的 `Δw` 使 `dtype(w + Δw) != dtype(w)`。（被测侧，带调用计数）

    ★★ 两条路径**结构上完全不同**：一条问「相邻两个可表示数差多少」（`nextafter`），
      另一条**从不构造相邻数**，而是反复把区间对半、只看**加法结果是否变了**（二分）。
      ⇒ 它们给出的比值若恒为 1，那才是有内容的（`A.1` + `D.6`/`D.7`）。

    ★★ **独立性是构造出来的**：二分用的初始区间是 `[0, max(1, |w|)]` ——
      一个**由问题本身给定**的宽区间，**`steps` 次对半后精度远超任何 ULP**
      （`|w|=1000` 时 80 次对半的格距 ≈ `1e-21`）。
      ⇒ **路径 A 一次也不调用路径 B**（`D.6` 的计数表 + `D.7` 的 AST 调用表查这件事）。
    """
    CALLS["crit_dw_by_bisection"] = CALLS.get("crit_dw_by_bisection", 0) + 1
    lo, hi = 0.0, max(1.0, abs(float(w)))
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        if dtype(w + mid) != dtype(w):
            hi = mid
        else:
            lo = mid
    return hi


# ===========================================================================
# 5. 训练轨迹：**直接 fp16** vs **主权重 fp32**（判据②）
# ===========================================================================
def train_direct_low_precision(steps, dtype=np.float16):
    """权重本身就用低精度存 —— 每次更新都可能被静默吞掉。

    返回 `(final_w, swallowed_steps, first_swallowed_step)`。
    """
    w = dtype(W0)
    swallowed = 0
    first = None
    for t in range(1, steps + 1):
        new = dtype(w - dtype(LR * GRAD))
        if new == w:
            swallowed += 1
            if first is None:
                first = t
        w = new
    return float(w), swallowed, first


def train_master_fp32(steps, dtype=np.float32):
    """主权重 fp32：更新在 fp32 上做（前向用什么精度是**另一件事**，见 `02`）。"""
    w = dtype(W0)
    for _ in range(steps):
        w = dtype(w - dtype(LR * GRAD))
    return float(w)


def closed_form_weight(steps):
    """推导值：`w0 − n·lr·g`（每步梯度恒为 `GRAD` 时的闭式）。"""
    return W0 - steps * LR * GRAD


# ===========================================================================
# 6. 下溢比例：**闭式** 与 **蒙特卡洛**（判据⑥，两条独立路径）
# ===========================================================================
def underflow_closed_form(sigma, threshold):
    """闭式：`g ~ N(0, σ)` 时 `P(|g| < threshold) = erf(threshold / (σ√2))`。

    ★★ 这里的 `threshold` 必须是**归零阈值** —— 舍入到最近时它是**最小次正规的一半**
      （`2^-25`），**不是**最小次正规本身（`2^-24`）。原稿把它取成 `2^-24`，
      于是 MC 与闭式之比恒在 **0.47 ~ 0.59**（`probes/README.md` 记着这个观察）。
      ⇒ 本程序传进来的是 `thr_over_two()` 的结果，`A.5` 另有一条**反面对照**。
    """
    return math.erf(threshold / (sigma * math.sqrt(2.0)))


def underflow_monte_carlo(sigma, n, dtype=np.float16):
    """蒙特卡洛：抽 `n` 个 `N(0, σ)`，数多少个**恰好变成 0**。

    返回 `(比例, 绝对值严格为 0 的个数, 落入次正规区的个数)`。
    ★ 后两个数是**「参照量级」**：P22 要求凡拿「机器精度档的相等」当判据的地方，
      先把参照量级打出来 —— 样本数太少时，MC 的噪声会与判据阈值同量级。
    """
    g = np.random.default_rng(SEED).standard_normal(n) * sigma
    h = np.asarray(g, dtype=dtype)
    zeroed = int(np.count_nonzero(h == 0))
    subnormal = int(np.count_nonzero((h != 0) & (np.abs(h) < float(np.finfo(dtype).tiny))))
    return zeroed / float(n), zeroed, subnormal


def thr_over_two():
    """归零阈值 = **最小次正规的一半**（舍入到最近，平局取偶）。

    ★ 计划 §0.1 的订正：「**下溢到 0」的阈值不是最小次正规，而是它的一半**。
    """
    return specs.F16_MIN_SUBNORMAL / 2.0


# ===========================================================================
# 7. 工具
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
    """两个口径**都**给：`max|Δ| / max|参照|`，以及逐元素相对。

    ★★ 口径是独立的一环（本库的常规错误模式）：同一个「相对差」有两种写法，
      它们在**近零元素**上差好几个数量级 —— 报数必须写明用的是哪一个。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (float(np.abs(a - b).max()) / float(np.abs(b).max()),
            float(np.max(np.abs(a - b) / (np.abs(b) + 1e-30))))


# ===========================================================================
# 8. A 类：类型选择表的**每一行**的可复算判据
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 四行的类型选择各自的判据（fp16 的边界 / 动态范围 / 下溢）")
    print("=" * 74)
    print("  口径：seed=%d · 二分 %d 次 · |w| 扫描 %s · σ 扫描 %s · MC %d 样本"
          % (SEED, BISECT_STEPS, list(W_GRID), list(SIGMAS), MC_N))

    # ── A.1 ★★ 判据①：更新被吞的临界 Δw == ULP/2（**两条独立路径**）────────────
    print()
    print("  ── A.1 判据①：`w + Δw == w`（更新被吞）的临界 Δw 与 ULP/2 ──")
    print("  ★★ 扫描表必须由 **fp16 里恰好可表示**的 `|w|` 组成（`1 / 10 / 100 / 1000`）——")
    print("     理由见 A.1b：**不在格点上的 `w` 给的是另一个量**。")
    print("  %12s %14s %14s %16s %12s" % ("|w|", "ULP(nextafter)", "ULP/2", "二分实测临界", "实测/(ULP/2)"))
    ratios = []
    for w in W_GRID:
        u = ulp_by_nextafter(w)
        c = crit_dw_by_bisection(w)
        ratios.append((w, c / (u / 2.0)))
        print("  %12.3f %14.6e %14.6e %16.6e %12.6f" % (w, u, u / 2.0, c, c / (u / 2.0)))
    worst = max(abs(r - 1.0) for _, r in ratios)
    check("A.1", worst < TOL_ULP_RATIO,
          "★ 四档 |w| 上「二分实测临界 / (ULP/2)」与 1 的最大相对差 %.3e < %g "
          "（★ 分子来自**二分**、分母来自 **nextafter** —— 两条结构不同的路径）"
          % (worst, TOL_ULP_RATIO))
    # ★★ P22 的落法：这一条判据是「机器精度档的相等」吗？**先确认那个量没退化。**
    #   `ULP(w)` 是**格距**：只要它 > 最小次正规数，`w + Δw` 就落在**正规**格上，
    #   「二分实测」测的就是格距本身、而不是次正规噪声。
    #   （Chunk 1 踩过反向的那个坑：`dW` 衰减到 `1e-54` 时，「三条路径差 `1e-54`」不是证据。）
    ulps = [ulp_by_nextafter(w) for w in W_GRID]
    print("  ★ 参照量级：四档的 ULP 最小 = %.4e，最小次正规数 = %.4e ⇒ 比值 %.1f×"
          % (min(ulps), specs.F16_MIN_SUBNORMAL, min(ulps) / specs.F16_MIN_SUBNORMAL))
    check("A.1c", min(ulps) > specs.F16_MIN_SUBNORMAL,
          "★ **P22 检查**：四档的 `ULP` 全部 **>** 最小次正规数（%.4e > %.4e）⇒ "
          "「二分实测」测的是**格距**，不是**次正规噪声** —— 这条相等才有内容"
          % (min(ulps), specs.F16_MIN_SUBNORMAL))

    # ── A.1b 一般式：临界 Δw = ULP/2 + (grid(w) − w)（**解释那两行「不是 1」**）──
    print("  ── A.1b 一般式：`Δw_crit = ULP/2 + (fp16(w) − w)` ──")
    print("  %10s %16s %16s %16s %16s %12s"
          % ("|w|", "fp16(w)", "fp16(w)−w", "ULP/2", "闭式 Δw_crit", "实测/闭式"))
    off_worst = 0.0
    for w in W_OFFGRID:
        g = float(np.float16(w))
        u = ulp_by_nextafter(w)
        closed = u / 2.0 + (g - w)
        c = crit_dw_by_bisection(w)
        rel = abs(c / closed - 1.0)
        off_worst = max(off_worst, rel)
        print("  %10.3f %16.10f %16.6e %16.6e %16.6e %12.6f" % (w, g, g - w, u / 2.0, closed, c / closed))
    check("A.1b", off_worst < TOL_ULP_RATIO,
          "★ `w` **不在** fp16 格点上时，临界 Δw 是 `ULP/2 − (w − grid(w))` 或 `ULP/2 + (grid(w) − w)` "
          "—— 统一写成 `ULP/2 + (fp16(w) − w)`；两档的实测/闭式最大相对差 %.3e < %g"
          % (off_worst, TOL_ULP_RATIO))
    print("  ★ 这就是为什么判据① 的扫描表必须是**格点值**：同一个「更新被吞」的机制，")
    print("    在非格点上给出的比值是 0.200000（`w=0.1`）与 1.560000（`w=0.01`）——")
    print("    那两个数**不是判据失败**，而是**另一个量**（本行给了它的闭式）。")

    # ── A.2 判据③：fp16 的数值边界（**两条独立来源**）─────────────────────────
    print()
    print("  ── A.2 判据③：fp16 的数值边界：`numpy.finfo` vs 按 IEEE 定义算 ──")
    b_read = boundary_from_finfo(np.float16)
    b_calc = boundary_from_definition(specs.F16_EXP_BITS, specs.F16_MAN_BITS, specs.F16_BIAS)
    names = ("最大有限数", "最小次正规数", "最小正规数", "eps")
    diffs = []
    for nm, vr, vc in zip(names, b_read, b_calc):
        rel = abs(vr - vc) / abs(vc)
        diffs.append(rel)
        print("  %-14s 读得 %-24.17g 算得 %-24.17g 相对差 %.3e" % (nm, vr, vc, rel))
    check("A.2", max(diffs) < TOL_BOUNDARY_REL,
          "★ 四个边界值两条路径逐位一致（最大相对差 %.3e < %g，机器精度档）"
          % (max(diffs), TOL_BOUNDARY_REL))
    check("A.2b", b_read[0] == specs.F16_MAX_FINITE
          and b_read[1] == specs.F16_MIN_SUBNORMAL and b_read[2] == specs.F16_MIN_NORMAL,
          "★ 其中三个与 `specs.py` 的 [A] 常量**逐位**相等（%.17g / %.17g / %.17g）"
          % (b_read[0], b_read[1], b_read[2]))

    # ── A.3 判据④：**三个动态范围必须同口径** ────────────────────────────────
    print()
    print("  ── A.3 判据④：三个动态范围（★ **各自标口径，不可两两相减**）──")
    print("  ★★ 这一行是 **D4b 的订正现场**：原稿写「差 ≈ 7.5 个数量级」**无来源** ——")
    print("     `76.46 − 12.04 = 64.42`，且那两个数**不同口径**（fp16 含次正规 vs fp32 正规）。")
    f16 = boundary_from_finfo(np.float16)
    f32 = boundary_from_finfo(np.float32)
    r_norm = math.log10(f16[0] / f16[2])          # fp16 正规区间
    r_all = math.log10(f16[0] / f16[1])           # fp16 含次正规
    r32 = math.log10(f32[0] / f32[2])             # fp32 正规区间
    # ★ 第二条来源：用 `specs.py` 的 [A] 位参数**重算一遍**（不读 finfo）
    d16 = boundary_from_definition(specs.F16_EXP_BITS, specs.F16_MAN_BITS, specs.F16_BIAS)
    d32 = boundary_from_definition(specs.F32_EXP_BITS, specs.F32_MAN_BITS, specs.F32_BIAS)
    r_norm2 = math.log10(d16[0] / d16[2])
    r_all2 = math.log10(d16[0] / d16[1])
    r32_2 = math.log10(d32[0] / d32[2])
    print("  fp16 正规区间  log10(max/2^-14)      = %.4f  （第二条来源 %.4f）" % (r_norm, r_norm2))
    print("  fp16 含次正规  log10(max/2^-24)      = %.4f  （第二条来源 %.4f）" % (r_all, r_all2))
    print("  fp32 正规区间  log10(max/1.18e-38)   = %.4f  （第二条来源 %.4f）" % (r32, r32_2))
    cross = max(abs(r_norm - r_norm2), abs(r_all - r_all2), abs(r32 - r32_2))
    check("A.3", abs(r_norm - RANGE_FP16_NORMAL) < TOL_RANGE
          and abs(r_all - RANGE_FP16_ALL) < TOL_RANGE
          and abs(r32 - RANGE_FP32_NORMAL) < TOL_RANGE,
          "★ 三个数与计划 §3 表 A ④ 逐项对上（容差 ±%g）：%.4f / %.4f / %.4f"
          % (TOL_RANGE, r_norm, r_all, r32))
    check("A.3b", cross < TOL_RANGE_CROSS,
          "★ 同样这三个数由**两条来源**（`numpy.finfo` 的表 / `specs` 的 [A] 位参数）算出，"
          "最大相对差 %.3e < %g" % (cross, TOL_RANGE_CROSS))
    print("  ★ **本程序不下「两个口径差了 N 个数量级」这种结论** —— 不同口径相减没有意义。")

    # ── A.4 判据⑤：bf16 的动态范围与 fp32 **几乎**相同 ───────────────────────
    print()
    print("  ── A.4 判据⑤：bf16 与 fp32 的量程比（★ 见下面的口径说明）──")
    bf = boundary_from_definition(specs.BF16_EXP_BITS, specs.BF16_MAN_BITS, specs.BF16_BIAS)
    bf_range = math.log10(bf[0] / bf[2])
    ratio = bf_range / r32
    print("  bf16 量程 = %.6f · fp32 量程 = %.6f · 比值 = %.10f" % (bf_range, r32, ratio))
    print("  ★★ **口径说明（本程序实测的一处偏差）**：计划 §0.7-D 的 D13b 写「比值 == 1.0000」——")
    print("     那个 `1.0000` 是 `probe_90b.py` 用 `{:.4f}` **打印**出来的；")
    print("     **精确比值是 %.10f ≠ 1**（差 %.3e）。" % (ratio, 1.0 - ratio))
    print("     它来自**尾数位不同**（bf16 的 `(2−2^-7)` vs fp32 的 `(2−2^-23)`），")
    print("     **不是**指数位不同 —— 两者的 `emax − emin` 都 = 253。")
    check("A.4", round(ratio, 4) == 1.0 and abs(1.0 - ratio) < TOL_BF16_RANGE_REL,
          "★ 判据⑤ **按 D13b 的口径**成立：比值到 4 位小数是 1.0000；"
          "而精确值 %.10f 与 1 的差 %.3e < %g（**偏差非零，且它来自尾数位**）"
          % (ratio, 1.0 - ratio, TOL_BF16_RANGE_REL))
    check("A.4b", (specs.BF16_EXP_BITS - specs.BF16_BIAS) == (specs.F32_EXP_BITS - specs.F32_BIAS),
          "★ 那句话**真正**成立的形式：bf16 与 fp32 的指数位相同 ⇒ `emax − emin` 相同（253）")

    # ── A.5 判据⑥：下溢比例的闭式与 MC 互验 ──────────────────────────────────
    print()
    print("  ── A.5 判据⑥：下溢比例的闭式 vs 蒙特卡洛 ──")
    thr2 = thr_over_two()
    print("  归零阈值 = 最小次正规 / 2 = %.6e（= 2^-25；★ 不是 %.6e 那个 2^-24）"
          % (thr2, specs.F16_MIN_SUBNORMAL))
    print("  %8s %16s %16s %16s %12s %14s" % ("σ", "闭式 erf(thr/2)", "MC 实测", "相对差", "期望命中数", "实测命中数"))
    ok = True
    for i, sigma in enumerate(SIGMAS):
        p_closed = underflow_closed_form(sigma, thr2)
        p_mc, hits, sub = underflow_monte_carlo(sigma, MC_N)
        rel = abs(p_mc - p_closed) / p_closed
        expect = MC_N * p_closed
        judged = i < SIGMA_JUDGED
        mark = "进判据" if judged else "★ 只报不判"
        print("  %8.0e %16.6f %16.6f %16.6f %12.0f %14d   %s"
              % (sigma, p_closed, p_mc, rel, expect, hits, mark))
        # ★★ P22：先确认「参照量级」够大，再拿它当判据
        if judged:
            if expect < MIN_EXPECTED_HITS:
                ok = False
            if rel >= TOL_UNDERFLOW_REL:
                ok = False
    check("A.5", ok,
          "★ 进判据的 %d 个 σ 上：MC 与闭式的相对差都 < %g，**且**每个 σ 的期望命中数 ≥ %d "
          "（★ 这就是 P22 要求的「先把参照量级打出来」—— 命中数太少时 MC 噪声与阈值同量级）"
          % (SIGMA_JUDGED, TOL_UNDERFLOW_REL, MIN_EXPECTED_HITS))
    # ★ 反面对照：把阈值取成**最小次正规本身**（旧错的形态），比值会掉到 ~0.5
    print("  ── A.5b 反面对照：把阈值取成 `2^-24`（最小次正规本身，**不是它的一半**）──")
    p_old = underflow_closed_form(SIGMAS[0], specs.F16_MIN_SUBNORMAL)
    p_mc0, _, _ = underflow_monte_carlo(SIGMAS[0], MC_N)
    print("  闭式(用 2^-24) = %.6f · MC = %.6f · 比值 MC/闭式 = %.4f"
          % (p_old, p_mc0, p_mc0 / p_old))
    check("A.5b", abs(p_mc0 / p_old - 0.5) < 0.15 and p_mc0 / p_old < 1.0,
          "★ 用**错的**阈值（`2^-24`）时 MC 只有闭式的一半（比值 %.4f）—— "
          "这正是 `probes/README.md` 记的「0.47~0.59」那个观察；"
          "⇒ 「阈值 = 最小次正规的一半」这句话**是被这个反面对照证伪出来的**，不是假设的"
          % (p_mc0 / p_old))
    return r_norm, r_all, r32


# ===========================================================================
# 9. B 类：代价模型 —— 精度损失（临界比值）与字节账
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：更新被吞的**临界比值**，以及四行类型选择的**字节账**")
    print("=" * 74)

    # ── B.1 ★★ 精度损失：临界比值与 Micikevicius 的 `at least 2048 times` ──────
    print("  ── B.1 临界比值 |w| / Δw_crit —— 外部锚点：Micikevicius 2017 逐字")
    print("     `This can happen when the magnitude of a normalized weight value is at")
    print("      least 2048 times larger that of the weight update.`  [C] arXiv 1710.03740")
    print("     （评审：**已确认** —— 该文 abs 页 `Comments:` 逐字 `Published as a conference paper")
    print("       at ICLR 2018`；★ 注意那句在 **ar5iv 正文里计 0**，两个载体别混）")
    print("  %12s %16s %14s %12s %10s" % ("|w|", "Δw_crit(二分)", "|w|/Δw_crit", "|w|/(ULP/2)", "是 2 的幂?"))
    exact_ok = True
    inband_ok = True
    approx_ok = True
    for w in W_POW2 + W_NONPOW2:
        c = crit_dw_by_bisection(w)
        r = w / c
        r_an = w / (ulp_by_nextafter(w) / 2.0)
        is_pow2 = (math.log2(w) == math.floor(math.log2(w)))
        print("  %12.3f %16.6e %14.4f %12.4f %10s" % (w, c, r, r_an, "是" if is_pow2 else "否"))
        # ★★ **区间判据挂在解析路上**：二分只能**逼近**临界值（从上方收敛），
        #   于是 `w/crit` 会落在 2048 的**下面**一点点（~1e-13 相对），
        #   `2048 <= r` 会假失败。⇒ 区间用 `r_an`（`nextafter` 给的精确值），
        #   二分那一路另用一条**近似**判据（`approx_ok`）。这个分工是本程序实测出来的。
        if not (RATIO_MIN <= r_an < RATIO_MAX):
            inband_ok = False
        if is_pow2 and r_an != float(RATIO_MIN):
            exact_ok = False
        if abs(r - r_an) > TOL_ULP_RATIO * r_an:
            approx_ok = False
    check("B.1", inband_ok and exact_ok and approx_ok,
          "★（解析路）临界比值落在 [%d, %d) —— 下界 = `2^11` = 论文的 `at least 2048`；"
          "且在 2 的幂处**精确 == %d**（那正是「右移 11 位」）"
          % (RATIO_MIN, RATIO_MAX, RATIO_MIN))
    check("B.1b", approx_ok,
          "★（二分路）**独立**逼近同一个量：二分实测的临界与解析值在 1e-3 相对容差内 "
          "⇒ 上面那条不是「拿解析值当自己的参照」")
    print("  ★ 口径：论文给的是**下界**（`at least`），实测比值 = 2·|w|/ULP ∈ [2^11, 2^12)；")
    print("    ⇒「至少 2048 倍」与「右移 11 位」是同一件事的两种说法。")
    print("  ★ 两个比值的分工：`(ULP/2)` 那一列走 `nextafter`（**解析**，能精确给 2048），")
    print("    前两列走**二分**（从上方收敛，只能逼近）⇒ 「精确」这条断言挂在解析路上。")

    # ── B.2 字节账：四种类型选择各占几字节 ──────────────────────────────────
    print()
    print("  ── B.2 字节账：**一个参数**在两张表上各占几字节 ──")
    rows = [("权重", MIXED_MASTER_BYTES + MIXED_COPY_BYTES, FP32_WEIGHT_BYTES),
            ("梯度", MIXED_GRAD_BYTES, FP32_GRAD_BYTES),
            ("优化器状态(m+v)", MIXED_OPT_BYTES, FP32_OPT_BYTES)]
    mp_tot = sum(r[1] for r in rows)
    f32_tot = sum(r[2] for r in rows)
    print("  %-18s %14s %14s %10s" % ("行", "混合精度(字节)", "全 fp32(字节)", "差"))
    for nm, a, b in rows:
        print("  %-18s %14d %14d %+10d" % (nm, a, b, a - b))
    print("  %-18s %14d %14d %+10d" % ("合计", mp_tot, f32_tot, mp_tot - f32_tot))
    print("  ⇒ 混合精度那一行是 **fp32 主权重 %d + fp16 计算副本 %d**（全 fp32 口径里"
          % (MIXED_MASTER_BYTES, MIXED_COPY_BYTES))
    print("     **没有**那份副本）；省下的是梯度的宽度（%d → %d），多出的正是那份副本。"
          % (FP32_GRAD_BYTES, MIXED_GRAD_BYTES))
    saved = FP32_GRAD_BYTES - MIXED_GRAD_BYTES
    extra = MIXED_COPY_BYTES
    check("B.2", mp_tot == f32_tot and saved == extra and MIXED_OPT_BYTES == FP32_OPT_BYTES,
          "★ 两列合计**相等**（%d == %d 字节/参数）：梯度省 %d、副本多 %d ⇒ **净 0**。"
          "混合精度省的是**计算与带宽**，**不是状态的字节**" % (mp_tot, f32_tot, saved, extra))
    check("B.2b", (MIXED_MASTER_BYTES, MIXED_COPY_BYTES, MIXED_GRAD_BYTES, MIXED_OPT_BYTES)
          == (ITEMSIZE_FP32, ITEMSIZE_FP16, ITEMSIZE_FP16, 2 * ITEMSIZE_FP32),
          "★ 每一行的字节数与**真的 dtype 宽度**对得上（fp32 的 m 与 v 各 %d ⇒ %d）"
          % (ITEMSIZE_FP32, 2 * ITEMSIZE_FP32))
    check("B.2c", ITEMSIZE_FP32 == np.dtype(np.float32).itemsize
          and ITEMSIZE_FP16 == np.dtype(np.float16).itemsize,
          "★ 这两个字节数**由 numpy 真的 dtype 复核**（%d / %d）—— "
          "不是本程序自己写的数" % (np.dtype(np.float32).itemsize, np.dtype(np.float16).itemsize))
    print("  ★ 口径边界：这里只数「一个参数」在**四种类型选择**下的字节；")
    print("    **状态由哪些东西组成**（含 RNG / dataloader）正文在")
    print("    `90-training/checkpointing/01-state-and-bytes`；**分片与通信量**正文在 `100-parallelism`。")
    print("    ★★ 本处的 `%d 字节/参数` 与 ZeRO 的 `16Ψ` **同数不同物**（那个含分片口径）——" % mp_tot)
    print("       本程序**不引用** ZeRO 的那个数，以免把两个口径混起来。")


# ===========================================================================
# 10. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：低精度当权重**真的**会失效；放大更新量后**真的**能救回来")
    print("=" * 74)

    # ── C.1 ★★ 判据②：直接 fp16 从第 1 步起就完全失效 ────────────────────────
    print("  ── C.1 判据②：`w0=%g`、`lr=%g`、每步梯度 `%g`、**%d 步** ──" % (W0, LR, GRAD, STEPS))
    print("  更新量 = lr·g = %.3e；而 `|w|=1` 处的 `ULP/2` = %.3e"
          % (LR * GRAD, ulp_by_nextafter(W0) / 2.0))
    print("  %8s %18s %12s %18s %18s" % ("步数", "直接 fp16", "被吞步数", "主权重 fp32", "闭式 w0−n·lr·g"))
    for n in TRACE_SHOW:
        d, sw, _f = train_direct_low_precision(n)
        m = train_master_fp32(n)
        print("  %8d %18.6f %12d %18.6f %18.6f" % (n, d, sw, m, closed_form_weight(n)))
    d2000, swallowed, first = train_direct_low_precision(STEPS)
    m2000 = train_master_fp32(STEPS)
    cl = closed_form_weight(STEPS)
    print("  ⇒ 第一次被吞发生在**第 %s 步**；此后权重**永不改变**" % first)
    check("C.1", d2000 == W0 and swallowed == STEPS and first == 1,
          "★ 直接 fp16 的权重 2000 步后**精确等于** `%g`（%d 步**全被吞**、第一步就被吞）"
          "—— 失败行为真的成立" % (W0, swallowed))
    check("C.1b", abs(m2000 - cl) < TOL_MASTER_VS_CLOSED,
          "★ 主权重 fp32 与闭式的差 %.3e < %g（%.6f vs %.6f）"
          % (abs(m2000 - cl), TOL_MASTER_VS_CLOSED, m2000, cl))
    check("C.1c", abs(m2000 - cl) > 0.0,
          "★ 但那个差**不是 0**（%.3e）：fp32 也不是精确的（`lr·g` 的累积舍入）—— "
          "本程序不把「fp32 = 精确」写成前提" % abs(m2000 - cl))

    # ── C.2 阴性对照：把更新量放大到 ULP 之上，**被吞的比例必须掉下来** ────────
    print()
    print("  ── C.2 阴性对照：把更新量乘 %g（越过 `ULP/2`）──" % UPDATE_FACTOR)
    lr_big = LR * UPDATE_FACTOR
    w = np.float16(W0)
    swallowed_big = 0
    for _ in range(STEPS):
        new = np.float16(w - np.float16(lr_big * GRAD))
        if new == w:
            swallowed_big += 1
        w = new
    frac = swallowed_big / float(STEPS)
    print("  放大后：更新量 %.3e > ULP/2 %.3e ⇒ 被吞 %d / %d 步（%.4f）"
          % (lr_big * GRAD, ulp_by_nextafter(W0) / 2.0, swallowed_big, STEPS, frac))
    check("C.2", frac < SWALLOW_TOLERANCE,
          "★ 把更新量放大 %g 倍后，被吞比例从 **1.0000** 掉到 %.4f < %g ⇒ "
          "「失效」不是恒真的（阴性对照成立）" % (UPDATE_FACTOR, frac, SWALLOW_TOLERANCE))
    check("C.2b", (W_POW2, W_NONPOW2) == ((1.0, 2.0, 4.0, 8.0), (10.0, 100.0, 1000.0)),
          "★ 扫描表的**两个分组逐位钉住** —— 没有 2 的幂那一组，`B.1` 的「精确 2048」"
          "就退化成「落在区间里」")


# ===========================================================================
# 11. D 类：失败标记自检 + 口径/阈值钉住 + 独立性
# ===========================================================================
def preflight():
    """★★ **D 类的前半段先跑**：标记自检 + 口径/阈值钉住（理由见 Chunk 1 的 P23）。

    ★★ **必须放在任何计算之前**：口径常量（`W_GRID` / `BISECT_STEPS` / `STEPS` / `STEPS` …
      与那几个字节常量）一旦被改，后面的**形状/长度检查**会先崩
      （`IndexError` / `ValueError`）—— 那时 `mutate.py` 记的是**崩溃捕获**，
      而「崩溃捕获不算真捕获」（它证明代码炸了，不是判据抓住了错）。
      ⇒ 把钉住提到**任何**计算之前，这些变异就变成**断言捕获**（它就是本库的 **P23**）。

    返回 `True` 表示可以继续。
    """
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型的结论。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ":: FAIL: mixed-precision"
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", "++ FAIL: optimizers",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住 —— 写错一个字，这条检查就静默失效了")

    measured = {"SEED": SEED, "BISECT_STEPS": BISECT_STEPS, "W_GRID": W_GRID,
                "W_OFFGRID": W_OFFGRID,
                "W_POW2": W_POW2, "W_NONPOW2": W_NONPOW2, "LR": LR, "GRAD": GRAD,
                "W0": W0, "STEPS": STEPS, "SIGMAS": SIGMAS, "MC_N": MC_N,
                "TRACE_SHOW": TRACE_SHOW, "SIGMA_JUDGED": SIGMA_JUDGED,
                "UPDATE_FACTOR": UPDATE_FACTOR}
    expect = {"SEED": 20260925, "BISECT_STEPS": 80,
              "W_GRID": (1.0, 10.0, 100.0, 1000.0), "W_OFFGRID": (0.01, 0.1),
              "W_POW2": (1.0, 2.0, 4.0, 8.0), "W_NONPOW2": (10.0, 100.0, 1000.0),
              "LR": 1e-3, "GRAD": 1e-3, "W0": 1.0, "STEPS": 2000,
              "SIGMAS": (1e-6, 1e-5, 1e-4, 1e-3), "MC_N": 2000000,
              "TRACE_SHOW": (10, 100, 500, 2000), "SIGMA_JUDGED": 3,
              "UPDATE_FACTOR": 1024.0}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 15 个逐位一致（这些是**结论的测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3d", all(float(np.float16(w)) == w for w in W_GRID)
          and all(float(np.float16(w)) != w for w in W_OFFGRID),
          "★ **A.1 的扫描表必须落在 fp16 格点上**这件事单独钉住（%s 全在格点、%s 全不在）"
          % (list(W_GRID), list(W_OFFGRID)))

    bytes_tab = {"MIXED_MASTER_BYTES": MIXED_MASTER_BYTES, "MIXED_COPY_BYTES": MIXED_COPY_BYTES,
                 "MIXED_GRAD_BYTES": MIXED_GRAD_BYTES, "MIXED_OPT_BYTES": MIXED_OPT_BYTES,
                 "FP32_WEIGHT_BYTES": FP32_WEIGHT_BYTES, "FP32_GRAD_BYTES": FP32_GRAD_BYTES,
                 "FP32_OPT_BYTES": FP32_OPT_BYTES,
                 "ITEMSIZE_FP32": ITEMSIZE_FP32, "ITEMSIZE_FP16": ITEMSIZE_FP16}
    bytes_expect = {"MIXED_MASTER_BYTES": 4, "MIXED_COPY_BYTES": 2, "MIXED_GRAD_BYTES": 2,
                    "MIXED_OPT_BYTES": 8, "FP32_WEIGHT_BYTES": 4, "FP32_GRAD_BYTES": 4,
                    "FP32_OPT_BYTES": 8, "ITEMSIZE_FP32": 4, "ITEMSIZE_FP16": 2}
    bdiff = {k: (bytes_expect[k], v) for k, v in bytes_tab.items() if v != bytes_expect[k]}
    check("D.3b", not bdiff, "★ 字节账的 9 个常量逐位一致（B.2 的「合计相等」全靠它们）%s"
          % ("" if not bdiff else " ← 差异 %s" % bdiff))
    check("D.3c", (MIXED_MASTER_BYTES + MIXED_COPY_BYTES + MIXED_GRAD_BYTES + MIXED_OPT_BYTES) == 16
          and (FP32_WEIGHT_BYTES + FP32_GRAD_BYTES + FP32_OPT_BYTES) == 16,
          "★ 两列合计**各 16** 这件事**单独钉住**（B.2 的结论就是这 16 == 16）")

    th = {"TOL_RANGE": TOL_RANGE, "TOL_ULP_RATIO": TOL_ULP_RATIO,
          "TOL_BOUNDARY_REL": TOL_BOUNDARY_REL, "TOL_RANGE_CROSS": TOL_RANGE_CROSS,
          "TOL_BF16_RANGE_REL": TOL_BF16_RANGE_REL, "TOL_UNDERFLOW_REL": TOL_UNDERFLOW_REL,
          "MIN_EXPECTED_HITS": MIN_EXPECTED_HITS, "RATIO_MIN": RATIO_MIN, "RATIO_MAX": RATIO_MAX,
          "TOL_MASTER_VS_CLOSED": TOL_MASTER_VS_CLOSED, "SWALLOW_TOLERANCE": SWALLOW_TOLERANCE,
          "RANGE_FP16_NORMAL": RANGE_FP16_NORMAL, "RANGE_FP16_ALL": RANGE_FP16_ALL,
          "RANGE_FP32_NORMAL": RANGE_FP32_NORMAL}
    th_expect = {"TOL_RANGE": 0.05, "TOL_ULP_RATIO": 1e-3, "TOL_BOUNDARY_REL": 1e-12,
                 "TOL_RANGE_CROSS": 1e-6, "TOL_BF16_RANGE_REL": 1e-4,
                 "TOL_UNDERFLOW_REL": 0.05, "MIN_EXPECTED_HITS": 300, "RATIO_MIN": 2048,
                 "RATIO_MAX": 4096, "TOL_MASTER_VS_CLOSED": 1e-4, "SWALLOW_TOLERANCE": 0.95,
                 "RANGE_FP16_NORMAL": 9.03, "RANGE_FP16_ALL": 12.04, "RANGE_FP32_NORMAL": 76.46}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 14 个逐位一致（含**判据④的三个动态范围**与"
                            "**判据⑥的小 n 门槛** MIN_EXPECTED_HITS）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.5", ITEMSIZE_FP32 == np.dtype(np.float32).itemsize
          and ITEMSIZE_FP16 == np.dtype(np.float16).itemsize,
          "★ 字节常量与**真的 dtype 宽度**对拍（%d / %d）"
          % (np.dtype(np.float32).itemsize, np.dtype(np.float16).itemsize))
    return len(FAILS) == before


def part_D():
    print()
    print("=" * 74)
    print("D. 独立性证据、specs 常量逐位钉住、以及「标记只用了本主题那一个」")
    print("=" * 74)
    # ── ★★ 参照实现的独立性（判据① 的两条路径，**双向**）────────────────────
    CALLS.clear()
    ulp_by_nextafter(1.0)
    check("D.6", list(CALLS) == ["ulp_by_nextafter"],
          "★ **参照侧跑完后计数表里只有它自己**（%s）⇒ 路径 B（`nextafter`）"
          "一次也没调用路径 A（二分）" % sorted(CALLS))
    CALLS.clear()
    crit_dw_by_bisection(1.0)
    check("D.6b", list(CALLS) == ["crit_dw_by_bisection"],
          "★ **反方向也查一遍**：被测侧（二分）跑完后计数表里**也**只有它自己（%s）"
          "⇒ 两条路径**互不调用**（★ 只查单向会漏网：本程序的结构翻转抽查实测到过 "
          "「二分改成调用 nextafter」而 rc=0 的版本）" % sorted(CALLS))
    watched = {"crit_dw_by_bisection", "ulp_by_nextafter"}
    called_ref = _callee_names(ulp_by_nextafter)
    called_meas = _callee_names(crit_dw_by_bisection)
    check("D.7", not (watched & called_ref) and not (watched & called_meas) and called_ref,
          "★ **AST 调用表双向为空**：`ulp_by_nextafter` 的 `Call` 节点 = %s · "
          "`crit_dw_by_bisection` 的 = %s —— 两条路径的调用表里都没有对方"
          % (", ".join(sorted(called_ref)) or "无", ", ".join(sorted(called_meas)) or "无"))
    check("D.7b", "ulp_by_nextafter" not in inspect.getsource(crit_dw_by_bisection)
          and "crit_dw_by_bisection" not in inspect.getsource(ulp_by_nextafter),
          "★ **源码文本级**的第三重保险：两条路径的源码（**含注释**）里都不出现对方的名字 —— "
          "它比 AST 更严、也更脆（改注释会触发它），所以三条一起上")
    check("D.8", inspect.getsource(ulp_by_nextafter) != inspect.getsource(crit_dw_by_bisection)
          and inspect.getsource(boundary_from_finfo) != inspect.getsource(boundary_from_definition)
          and inspect.getsource(underflow_closed_form) != inspect.getsource(underflow_monte_carlo),
          "★ 防 P17：三对「路径 1 / 路径 2」的**源码两两不同**（`nextafter`/二分、表/定义、"
          "`erf`/MC）")

    # ── specs 常量：**逐位钉住**（这就是它们的消费者）───────────────────────
    print("  specs 常量逐位钉住（★ 没有消费者的常量只能是 `make mutate` 的漏网候选）：")
    check("D.9", specs.F16_EXP_BITS == 5 and specs.F16_MAN_BITS == 10 and specs.F16_BIAS == 15
          and specs.F32_EXP_BITS == 8 and specs.F32_MAN_BITS == 23 and specs.F32_BIAS == 127
          and specs.BF16_EXP_BITS == 8 and specs.BF16_MAN_BITS == 7 and specs.BF16_BIAS == 127,
          "★ IEEE 的九个**位参数**逐位一致（fp16 / fp32 / bf16）")
    check("D.10", specs.F16_MAX_FINITE == 65504.0
          and specs.F16_MIN_SUBNORMAL == 5.960464477539063e-08
          and specs.F16_MIN_NORMAL == 6.103515625e-05,
          "★ fp16 的三个边界常量逐位一致（判据③ 的期望值从这里来）")
    check("D.11", specs.MICIKEVICIUS_WEIGHT_TO_UPDATE_RATIO_MIN == 2048,
          "★ 外部锚点 `2048` 逐位一致（B.1）")
    check("D.12", specs.MICIKEVICIUS_ZERO_BELOW_QUOTE == "any value whose magnitude is smaller than",
          "★ 论文逐字**片段**逐位一致（%r）—— ★ **用 `==`，不用 `startswith`/`in`**："
          "这些字符串常量的变异形态是**末尾追加一个 `X`**（计划 §3 开头的盲区③）"
          % specs.MICIKEVICIUS_ZERO_BELOW_QUOTE)
    check("D.13", specs.UNSOURCED == ("bf16_dynamic_range_exact_equality_claim",
                                      "torch_e4m3fn_overflow_rule_version_span",
                                      "fp8_scaling_growth_rule_concrete_values"),
          "★ `specs.UNSOURCED` 的三项**逐位钉住**（%d 项）：它是「本域不写、但记下为什么」"
          "的待办清单 —— 删一条、加一条、改一个字都会失败" % len(specs.UNSOURCED))
    check("D.14", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）—— 跨主题照抄标记"
          "由这一条 + `mutate.py` 的 AST 读取双重挡住"
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
    print("01_types.py —— 主权重 / 梯度 / 优化器状态 / loss 四者的类型选择")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★ 禁区（一律回链，本程序不写）：「为什么 fp16 要 loss scaling、bf16 为什么不用」")
    print("    整块在 `10-foundations/floating-point/04-low-precision-formats.md`；")
    print("    格式位分配在 `10-foundations/floating-point/01-ieee754-format.md:189-226`；")
    print("    累加器宽度的误差界在 `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md`；")
    print("    主权重「为什么必须 fp32」的数值理由在同域 `nla/04-iterative-methods.md:229-266`。")
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
