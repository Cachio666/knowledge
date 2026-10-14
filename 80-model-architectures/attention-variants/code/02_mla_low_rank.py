#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_mla_low_rank.py —— MLA：低秩压缩、矩阵吸收、以及与 RoPE 的结构性冲突。

配套文档：`02-mla-low-rank.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本程序的数字**全部是推导值或本机实测值**，测量条件是：

> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 它们支撑的是「**低秩压缩与位置旋转不能共存**」这类**代数/结构**结论，
> 不支撑任何关于真实模型训练或 GPU 的结论。

★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）：

  · `A.1`  **吸收**与**显式**两条路径给出同一个分数矩阵（最大逐元素差 `< 1e-12`，
           **机器精度档**）；
  · `A.2`  参照（吸收）路径跑完后**调用计数表完全为空**；
  · `A.4`  MLA 的 KV 元素数是 **`(d_c + d_h^R)·l`**，**不是** `d_c·l` —— 把解耦 RoPE 的
           那一半去掉，断言立刻失败；
  · `B.2`  ★★ 判据的本体：**无 RoPE 时**分数矩阵能被 `d_c` 维的「只依赖 t」的因子**精确**复现
           （残差到机器精度），**有 RoPE 时不能**（残差 >> 机器精度）
           —— 这就是「`R` 夹在 q 与 c 之间、不可吸收」的**可测形态**；
  · `B.3`  把 RoPE **吸收进去**（先吸收再旋转）与正确形式的差 `> 1.0`；
  · `C.1`  MLA 的元素数式子里**没有 head 数因子**（压缩是跨 head 联合的、解耦 key 是共享的），
           而 MHA 的式子里有；
  · `C.3`  **只旋转 `k` 不旋转 `q`** 的写法**不满足**相对位置性质
           （把「两边都旋转」改成「只转 k」，这条断言立刻失败）。

★★ **本程序刻意不做的事**（计划 §2 的边界表，越界即作废）：

  * **不重推 RoPE 的定义与两套实现约定** —— 正文在 `positional-encoding/01-rope-definition`
    （正文在彼，此处回链）。本程序只**用它**，不解释它；
  * **不算 KV 的字节**：六项乘积的正文在
    `130-model-optimization/quantization/18-kv-cache-basics.md:48-76`（正文在彼，此处回链）。
    本程序只算**元素数**，且度量口径照抄论文的逐字
    （`measured by the number of elements, regardless of the storage precision`）；
  * **不讲** MLA 对 **kernel 结构**的影响（正文在 `40-kernels/attention/04:126-156`）。

用法：
    python3 02_mla_low_rank.py
"""

import ast
import collections
import inspect
import os
import sys
from fractions import Fraction

import numpy as np

# ★ 同目录的共享事实源（`code/` 自包含，`CONTRIBUTING:36`）。
import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题取 `@@ FAIL: attention-variants`（域 README 写死；**不跨主题照抄**）。
ASSERT_FAIL_MARK = "@@ FAIL: attention-variants"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("!! FAIL: transformer", "## FAIL: moe",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D）。"""
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
# 1. 口径常量（★ 全部在 `D.3`/`D.4` 被逐位钉住）
# ===========================================================================
# ★★ 判据的**非空性**依赖 `D_C < S`：只有压缩维度真的小于序列长度，
#    「用 `d_c` 维因子去复现分数矩阵」才可能失败（`d_c ≥ S` 时它恒可复现 —— 那是空断言）。
#    这一条在 `D.5` 被钉住，并在 `B.2` 的说明里打印出来。
D_MODEL = 32             # 输入的宽度 d（本程序用**单个 head**：d_h = d）
D_C = 4                  # KV 的压缩维度 d_c（★ 必须 < S）
S = 12                   # 位置数（★ 必须 > D_C）
SEED = 20260922          # 数据种子（= 探测脚本 probe_80g.py 用的那一个）
ROPE_BASE = 10000.0      # RoPE 的 base（★ 定义在 positional-encoding/01，此处只当口径）
L_LAYERS = 4             # 层数 l（元素数式子里的那个 l）
N_H_TEST = 32            # C.1 用的 head 数（只用来演示「式子对 head 数敏不敏感」）

# ── 阈值 ────────────────────────────────────────────────────────────────────
ABSORB_TOL = 1e-12       # A.1 吸收 vs 显式（**机器精度档**）
ROPE_DIFF_MIN = 1.0      # B.3 错误吸收的差的下限（**量级档**）
DIFF_RATIO_MIN = 1e15    # B.3 与 A.1 的差之比的下限（★ 计划 §3 表 A ④ 的阈值）
FIT_TOL = 1e-10          # B.2 无 RoPE 时的最小二乘残差（**机器精度档**）
FIT_MIN = 1e-2           # B.2 有 RoPE 时的最小二乘残差下限（**量级档**）
SHIFT_TOL = 1e-12        # C.3 相对位置性质（**机器精度档**）
SHIFT_MIN = 1e-1         # C.3 只转 k 不转 q 时的偏差下限（**量级档**）

# 调用计数器：**参照实现独立性**的证据（`00-meta/glossary.md` 的「reference implementation（参考实现）」条的机械判据）。
CALLS = {}


# ===========================================================================
# 2. 被测实现（路径 1）—— 低秩压缩的**显式**路径
# ===========================================================================
# MLA 的压缩（论文 §2.1.2 式 (9)-(11) 的结构，用一个 head 表示）：
#     c = x·W_DKV          （压到 d_c 维 —— ★ 这**就是** KV cache 里存的东西）
#     k = c·W_UK ,  v = c·W_UV
def mla_c(x, w_dkv):
    """低秩**下投影**：`c = x·W_DKV`（`d → d_c`）。★ 它就是「压缩后的 KV」。"""
    CALLS["mla_c"] = CALLS.get("mla_c", 0) + 1
    return x @ w_dkv


def mla_explicit(x, w_q, w_dkv, w_uk):
    """被测实现（路径 1）：**显式**算出 `c` → `k` → 分数矩阵。

    ★ 它**真的物化** `k`（`d_c → d` 的上投影），所以每条路径都要多一次矩阵乘；
      这正是「吸收」要省掉的那一步。
    ★ 它**不**含 RoPE —— 本程序把 RoPE 单独放在 B 段，正是为了把两件事分开测。
    """
    CALLS["mla_explicit"] = CALLS.get("mla_explicit", 0) + 1
    c = mla_c(x, w_dkv)
    k = c @ w_uk
    q = x @ w_q
    return q @ k.T, c


# ===========================================================================
# 3. ★★ 参照实现（路径 2）—— **不调用被测实现的任何函数**
# ===========================================================================
# ★★ 「矩阵吸收」是什么：把 `W_UK` **吸收进** `W_Q`：
#       scores = (x W_Q)(c W_UK)^T = x (W_Q W_UK^T) c^T = (x W_eff) c^T
#     ⇒ 推论：`W_eff = W_Q W_UK^T` 可以**离线预计算一次**，之后每次只需要
#        `(x W_eff) · c^T` —— **不再需要算出 `k`**（这正是「只需要缓存 `c`」的代数依据）。
# ★ 它与路径 1 的差别是**真的**：少一次矩阵乘 + 收缩顺序不同（先合并两个权重矩阵）。
def mla_absorbed(x, w_q, w_uk, c):
    """独立参照（路径 2）：先吸收 `W_UK` 到 `W_Q`，再作用到 `x` 上。

    ★ 它**不调用** `mla_c` / `mla_explicit`：`c` 由调用方（用被测路径算出的同一份）传入，
      而 `W_eff` 由两个权重直接相乘得到 —— 这是与「显式算出 k」**不同的一条算式**。
    """
    w_eff = w_q @ w_uk.T                  # (d, d_c)  ← 离线一次，与序列长度无关
    return (x @ w_eff) @ c.T


# ===========================================================================
# 4. RoPE（★ 定义与两套实现约定的正文在 `positional-encoding/01`，此处只**用**）
# ===========================================================================
def rope(x, pos, base=ROPE_BASE):
    """RoPE：按**相邻对**旋转（原论文约定）。

    ★★ **本函数是「正文在别处、此处回链」的现场**：RoPE 的旋转定义、两套实现约定
      （相邻对 vs `rotate_half`）、以及「两者是同一操作被坐标置换共轭」的结论，
      正文在 `positional-encoding/01-rope-definition`（正文在彼，此处回链）。
      本程序**只用它**，不解释它 —— 这里唯一的纪律是：**`q` 与 `k` 都要旋转**（见 `C.3`）。
    """
    dd = x.shape[-1]
    i = np.arange(1, dd // 2 + 1, dtype=np.float64)
    th = base ** (-2.0 * (i - 1) / dd)
    xr = x.reshape(*x.shape[:-1], dd // 2, 2)
    c, s = np.cos(pos * th), np.sin(pos * th)
    a, b = xr[..., 0], xr[..., 1]
    return np.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape)


def scores_rope_explicit(x, w_q, w_dkv, w_uk, positions):
    """**正确**形式：`scores[t,s] = (R_t q_t) · (R_s k_s)` —— **q 与 k 都旋转**。

    ★ 论文逐字（arXiv 2405.04434 §2.1.3）：「RoPE is position-sensitive for both keys and
      queries.」⇒ 少旋转任何一边，得到的都**不是**注意力分数。
    """
    c = x @ w_dkv
    k = c @ w_uk
    q = x @ w_q
    k_r = np.stack([rope(k[t], positions[t]) for t in range(len(positions))])
    q_r = np.stack([rope(q[t], positions[t]) for t in range(len(positions))])
    return q_r @ k_r.T


def scores_rope_absorbed_wrong(x, w_q, w_uk, c, positions):
    """**错误**的「吸收」：先把 `W_UK` 吸收掉，再对**吸收后的低维 q** 套旋转。

    ★ 这是「想让 cache 里只存 `c`，所以把旋转也一起搬走」的那一步。
      论文说它不可行，理由逐字：「a RoPE matrix related to the currently generating token
      will lie between `W^Q` and `W^UK` and matrix multiplication does not obey a
      commutative law」。
    """
    w_eff = w_q @ w_uk.T                        # (d, d_c)
    q_eff = x @ w_eff                           # (S, d_c) ← 已经压到 d_c 维
    q_rot = np.stack([rope(q_eff[t], positions[t]) for t in range(len(positions))])
    return q_rot @ c.T


# ===========================================================================
# 5. 元素数（判据⑤）与「等于 GQA 几组」的闭式
# ===========================================================================
def mla_kv_elements(d_c, d_h_r, layers):
    """MLA 每个 token 的 KV cache **元素数**。

    ★★ 论文逐字（arXiv 2405.04434 §2.1.3）：「During inference, the decoupled key should
      also be cached. Therefore, DeepSeek-V2 requires a total KV cache containing
      `(d_c + d_h^R)l` elements.」
    ★★ **两半都要**：`d_c`（低秩压缩的那一半）+ `d_h^R`（**解耦 RoPE** 的那一半）。
      常见错误是写成 `d_c·l` —— 那漏掉了解耦的代价。
    ★ 口径：**元素数**（不是字节）；度量口径照抄论文的逐字
      「measured by the number of elements, regardless of the storage precision」。
    ★ ★ 式子里**没有 head 数因子**：因为压缩是**跨 head 联合**的（逐字：「the low-rank
      **joint** compression for keys and values」），而解耦的 key 是**所有 head 共享的一条**
      （逐字：「a **shared** key `k_t^R`」）—— 见 `C.1`。
    """
    return (d_c + d_h_r) * layers


def mha_kv_elements(n_h, d_h, layers):
    """MHA 每个 token 的 KV cache 元素数 —— 论文 Table 1 逐字：`2 n_h d_h l`。

    ★ 放在这里**只为对照**（`C.1` 要说明 MLA 的式子里没有 head 数因子）。
      本域**不重讲** KV cache 的字节公式（正文在
      `130-model-optimization/quantization/18-kv-cache-basics:48-76`，正文在彼，此处回链）。
    """
    return 2 * n_h * d_h * layers


def gqa_equivalent_groups(d_c, d_h_r, d_h):
    """★ 闭式：「MLA 的元素数等于 **GQA 的几组**」。

    由 `(d_c + d_h^R)·l = 2·n_g·d_h·l`（论文 Table 1 的两行）解出

        `n_g = (d_c + d_h^R) / (2 d_h)`

    ★★ 这条闭式**只需要论文自述的两个比例**（`d_c = 4 d_h`、`d_h^R = d_h/2`），
      **不需要** DeepSeek-V2 的绝对维度 —— 所以它在本环境可核算。
      这就是「配一条闭式参照」的落法（计划 §3 Chunk 1 的裁决：纯量级判据证伪能力弱）。
    """
    return Fraction(d_c + d_h_r, 1) / (2 * Fraction(d_h, 1))


# ===========================================================================
# 6. A 类：低秩压缩与矩阵吸收
# ===========================================================================
def draw_mla(seed, s, d, d_c):
    """★ **抽取顺序是口径的一部分**：`X` → `W_q` → `W_DKV` → `W_UK`。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((s, d))
    w_q = rng.standard_normal((d, d)) / np.sqrt(d)
    w_dkv = rng.standard_normal((d, d_c)) / np.sqrt(d)
    w_uk = rng.standard_normal((d_c, d)) / np.sqrt(d_c)
    return x, w_q, w_dkv, w_uk


def part_A():
    print("=" * 74)
    print("A. 低秩压缩与「矩阵吸收」")
    print("=" * 74)
    print("  口径：d=%d · d_c=%d · S=%d · 种子 %d（单 head：d_h = d）"
          % (D_MODEL, D_C, S, SEED))
    print("  结构：c = x·W_DKV（d→d_c）·  k = c·W_UK ·  scores = (x·W_Q)·kᵀ")
    print("  吸收：scores = ((x·(W_Q·W_UKᵀ))·cᵀ —— `W_Q·W_UKᵀ` 离线算一次即可")

    x, w_q, w_dkv, w_uk = draw_mla(SEED, S, D_MODEL, D_C)
    explicit, c = mla_explicit(x, w_q, w_dkv, w_uk)
    absorbed = mla_absorbed(x, w_q, w_uk, c)
    d = float(np.abs(explicit - absorbed).max())
    mag = float(np.abs(explicit).max())
    print("    吸收 vs 显式：最大逐元素差 %.3e（分数矩阵量级 %.3f ⇒ 相对 %.2e）"
          % (d, mag, d / mag))
    check("A.1", d < ABSORB_TOL,
          "**无 RoPE 时**吸收与显式等价（最大逐元素差 %.3e < %g，机器精度档）"
          % (d, ABSORB_TOL))

    # ★ 参照跑完后，计数表必须**整个是空的**。
    CALLS.clear()
    mla_absorbed(x, w_q, w_uk, c)
    check("A.2", not CALLS,
          "★ 参照（吸收）路径跑完后**计数表完全为空**（记录到的调用：%s）⇒ "
          "它一次都没调用被测实现的函数" % (sorted(CALLS) if CALLS else "无"))

    src_explicit = inspect.getsource(mla_explicit)
    src_absorbed = inspect.getsource(mla_absorbed)
    check("A.3", src_explicit != src_absorbed,
          "★ 防 P17：被测与参照的**源码不同**（`inspect.getsource` 比较，%d vs %d 字节）"
          % (len(src_explicit), len(src_absorbed)))

    # ── A.4 判据⑤：元素数是 (d_c + d_h^R)·l，**不是** d_c·l ────────────────
    # ★★ **口径：以 `d_h` 为符号单位**（`unit`）—— 所有结果都是 `d_h` 的倍数，
    #    **结论与这个单位取多少无关**（论文只给了比例，见 `specs.py`）。
    unit = Fraction(D_MODEL)
    dc_u = Fraction(specs.DEEPSEEK_V2_D_C_OVER_D_H) * unit        # = 4·d_h
    dhr_u = Fraction(specs.DEEPSEEK_V2_D_H_R_OVER_D_H) * unit     # = 0.5·d_h
    both = mla_kv_elements(dc_u, dhr_u, L_LAYERS)
    only_c = mla_kv_elements(dc_u, 0, L_LAYERS)
    print()
    print("  A.4：元素数 = (d_c + d_h^R)·l；以 `d_h` 为单位（d_c = %s·d_h、d_h^R = %s·d_h、l=%d）"
          % (specs.DEEPSEEK_V2_D_C_OVER_D_H, specs.DEEPSEEK_V2_D_H_R_OVER_D_H, L_LAYERS))
    print("       (d_c + d_h^R)·l = **%s·d_h** ；只算压缩那一半 d_c·l = %s·d_h"
          "（少了 %s·d_h = d_h^R·l）"
          % (both / unit, only_c / unit, (both - only_c) / unit))
    check("A.4", both == (dc_u + dhr_u) * L_LAYERS and both != only_c
          and Fraction(both, only_c) == 1 + Fraction(dhr_u, dc_u),
          "★ 判据⑤：MLA 的元素数 = **`(d_c + d_h^R)·l`**（%s·d_h），**不是** `d_c·l`"
          "（%s·d_h）—— 解耦 RoPE 的那一半**也在 cache 里**（论文逐字「the decoupled key "
          "should also be cached」）" % (both / unit, only_c / unit))

    # ── A.5 闭式：「等于 GQA 的几组」─────────────────────────────────────
    # ★ 论文自述的两个**比例**（[C]，见 specs.py）：d_c = 4 d_h、d_h^R = d_h/2。
    #   ⇒ 用**以 d_h 为单位**的量代入，d_h 被消掉，得到一个**精确有理数**。
    n_g = gqa_equivalent_groups(dc_u, dhr_u, unit)
    per_dh = Fraction(dc_u + dhr_u, unit)
    print()
    print("  A.5：闭式 `n_g = (d_c + d_h^R)/(2·d_h)`；代入**论文自述的比例**")
    print("       d_c = %s·d_h、d_h^R = %s·d_h ⇒ (d_c+d_h^R) = %s·d_h"
          % (specs.DEEPSEEK_V2_D_C_OVER_D_H, specs.DEEPSEEK_V2_D_H_R_OVER_D_H, per_dh))
    print("       ⇒ n_g = %s（= %.2f 组）；论文 Table 1 的另一行写作 ≈ %s·d_h·l"
          % (n_g, float(n_g), per_dh))
    check("A.5", n_g == Fraction(9, 4) and per_dh == Fraction(9, 2),
          "★ 纯由论文自己的两个比例复现出 **%s 组**（= 论文逐字 `equal to GQA with only "
          "2.25 groups`），且 Table 1 的 `≈ 9/2·d_h·l` 也精确成立（%s）—— "
          "**不需要 DeepSeek-V2 的绝对维度**" % (n_g, per_dh))


# ===========================================================================
# 7. B 类：RoPE 与低秩压缩的结构性冲突
# ===========================================================================
def _best_dc_factor_fit(a, c):
    """★ 判据的本体：**最好的「可吸收」近似有多好？**

    所谓「可吸收」，就是存在一个**只依赖 t** 的 `d_c` 维向量 `u_t`，使

        `scores[t, s] = u_t · c_s`   （对所有 t, s）

    —— 这正是「把 `W_UK` 吸收进 `W_Q`」这句话的数学内容（`u_t = x_t·W_Q·W_UKᵀ`）。
    检验方法：解最小二乘 `min_U ‖A − U·Cᵀ‖_F`，返回**相对残差** `残差/‖A‖_F`。

    ★★ 为什么它是一条**非空的**判据：只有当 `d_c < S` 时「可吸收」才可能失败
      （`d_c ≥ S` 时 `d_c` 维因子足以张满 `R^S`，残差**恒为 0** —— 那是空断言）。
      本程序的 `D_C < S` 由 `D.5` 钉住。
    """
    u = a @ np.linalg.pinv(c.T)          # (S,S) @ (S,d_c) -> (S,d_c) 的最小范数解
    resid = float(np.linalg.norm(a - u @ c.T)) / float(np.linalg.norm(a))
    return resid


def part_B():
    print()
    print("=" * 74)
    print("B. ★ RoPE 与低秩压缩的结构性冲突")
    print("=" * 74)
    print("  RoPE 的定义与两套实现约定的正文在 positional-encoding/01（正文在彼，此处回链）；")
    print("  本段只**用**它，并把它与「吸收」放在同一把尺子下量。")
    x, w_q, w_dkv, w_uk = draw_mla(SEED, S, D_MODEL, D_C)
    positions = np.arange(S, dtype=np.float64)
    c = x @ w_dkv
    scores_plain, _ = mla_explicit(x, w_q, w_dkv, w_uk)
    scores_rope = scores_rope_explicit(x, w_q, w_dkv, w_uk, positions)

    # ── B.1 正确形式：q 与 k 都旋转 ───────────────────────────────────────
    print()
    print("  B.1：正确形式 `scores[t,s] = (R_t q_t)·(R_s k_s)` —— **q 与 k 都旋转**")
    print("       （论文逐字：`RoPE is position-sensitive for both keys and queries.`）")
    k = c @ w_uk
    q = x @ w_q
    only_k = np.stack([q[t] for t in range(S)]) @ np.stack(
        [rope(k[t], positions[t]) for t in range(S)]).T
    only_q = np.stack([rope(q[t], positions[t]) for t in range(S)]) @ k.T
    check("B.1", float(np.abs(scores_rope - only_k).max()) > ROPE_DIFF_MIN
          and float(np.abs(scores_rope - only_q).max()) > ROPE_DIFF_MIN,
          "少旋转任何一边都得到**另一个对象**：只转 k 差 %.3f、只转 q 差 %.3f（都 > %g）"
          % (float(np.abs(scores_rope - only_k).max()),
             float(np.abs(scores_rope - only_q).max()), ROPE_DIFF_MIN))

    # ── B.2 ★★ 判据的本体：同一把尺子量「可吸收性」─────────────────────
    r_plain = _best_dc_factor_fit(scores_plain, c)
    r_rope = _best_dc_factor_fit(scores_rope, c)
    print()
    print("  B.2：用 `d_c=%d` 维的「只依赖 t」因子去复现分数矩阵（最小二乘，相对残差）"
          % D_C)
    print("       无 RoPE：%.3e     有 RoPE：%.3e      ⇒ 相差 %.3e 倍"
          % (r_plain, r_rope, r_rope / max(r_plain, 1e-300)))
    check("B.2", r_plain < FIT_TOL and r_rope > FIT_MIN,
          "★ **可吸收 ⇔ 存在 `d_c` 维因子精确复现分数**：无 RoPE 时残差到机器精度（%.3e），"
          "有 RoPE 时残差是**一个量级**（%.3e）⇒ 论文那句 `W^UK cannot be absorbed into W^Q "
          "any more` 在**数值上就是这件事**" % (r_plain, r_rope))

    # ── B.3 判据④：错误吸收与正确形式的差 ────────────────────────────────
    wrong = scores_rope_absorbed_wrong(x, w_q, w_uk, c, positions)
    d_rope = float(np.abs(scores_rope - wrong).max())
    d_plain = float(np.abs(scores_plain - mla_absorbed(x, w_q, w_uk, c)).max())
    ratio = d_rope / d_plain
    print()
    print("  B.3：「先吸收再旋转」vs 正确形式：最大差 %.4f（分数矩阵量级 %.3f）"
          % (d_rope, float(np.abs(scores_rope).max())))
    print("       同一个量在**无 RoPE** 时是 %.3e ⇒ 两者相差 %.3e 倍"
          % (d_plain, ratio))
    print("       ⚠ ★ **具体数值不写进正文**：它是**单次抽样**的产物（计划 §0.7-D 的 D15：控制者"
          " 报的 18.90 / 审查者的 12.64 / 实施者修正后的 13.64 都是单次抽样），"
          "只有**量级**承载信息。")
    check("B.3", d_rope > ROPE_DIFF_MIN and ratio > DIFF_RATIO_MIN,
          "★ 判据④：把 RoPE 吸收进去后的差 %.4f **> %g**，且与判据③的机器精度差相差 "
          "**%.3e 倍 > %g**" % (d_rope, ROPE_DIFF_MIN, ratio, DIFF_RATIO_MIN))


# ===========================================================================
# 8. C 类：解耦 RoPE 的代价，与两处反例
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 解耦 RoPE 的代价（边界与反例）")
    print("=" * 74)

    # ── C.1 式子里**没有 head 数因子** ────────────────────────────────────
    # ★ 同样以 `d_h` 为**符号单位**（见 A.4 的口径）。
    unit = Fraction(D_MODEL)
    dc_u = Fraction(specs.DEEPSEEK_V2_D_C_OVER_D_H) * unit
    dhr_u = Fraction(specs.DEEPSEEK_V2_D_H_R_OVER_D_H) * unit
    mla_32 = mla_kv_elements(dc_u, dhr_u, L_LAYERS)      # ★ 式子**没有** head 数入参
    mla_64 = mla_kv_elements(dc_u, dhr_u, L_LAYERS)      # 换一个 head 数也没处可传
    mha_32 = mha_kv_elements(N_H_TEST, unit, L_LAYERS)
    mha_64 = mha_kv_elements(2 * N_H_TEST, unit, L_LAYERS)
    print("  C.1：两张式子的**形状不同**（论文 Table 1）：")
    print("       MHA：`2·n_h·d_h·l` —— n_h=%d → %s·d_h，n_h=%d → %s·d_h（**随 head 数翻倍**）"
          % (N_H_TEST, mha_32 / unit, 2 * N_H_TEST, mha_64 / unit))
    print("       MLA：`(d_c + d_h^R)·l` —— **式子里根本没有 head 数** ⇒ %s·d_h（不变）"
          % (mla_32 / unit))
    print("       ⇒ 逐字理由：压缩是**跨 head 联合**的（`low-rank joint compression`），")
    print("         而解耦的 key 是**所有 head 共享**的一条（`a shared key k_t^R`）")
    check("C.1", mla_32 == mla_64 and mha_64 == 2 * mha_32,
          "★ MLA 的元素数对 head 数**不敏感**（%s·d_h → %s·d_h），MHA 的**敏感**"
          "（%s·d_h → %s·d_h，恰 2 倍）⇒ 把 MLA 的式子按 MHA 的形状乘上 head 数会**算错**"
          % (mla_32 / unit, mla_64 / unit, mha_32 / unit, mha_64 / unit))

    # ── C.2 反例：把 d_h^R 去掉 ⇒ 位置信息全丢 ───────────────────────────
    # 可测形态：**完全没有 RoPE** 的分数矩阵对「把同一批 token 重新排列」是不变的。
    x, w_q, w_dkv, w_uk = draw_mla(SEED, S, D_MODEL, D_C)
    perm = np.array([S - 1 - t for t in range(S)])      # 反序置换
    a_plain, _ = mla_explicit(x, w_q, w_dkv, w_uk)
    a_perm, _ = mla_explicit(x[perm], w_q, w_dkv, w_uk)
    # 反序后再反序 = 还原 ⇒ 无 RoPE 时 `a_plain == a_perm[perm]`（逐位）
    same_plain = float(np.abs(a_plain - a_perm[perm][:, perm]).max())
    pos = np.arange(S, dtype=np.float64)
    a_rope = scores_rope_explicit(x, w_q, w_dkv, w_uk, pos)
    a_rope_perm = scores_rope_explicit(x[perm], w_q, w_dkv, w_uk, pos)
    diff_rope = float(np.abs(a_rope - a_rope_perm[perm][:, perm]).max())
    print()
    print("  C.2：反例 —— 若 `d_h^R = 0`（不保留解耦的那一份），元素数退化成 `d_c·l`，")
    print("       但**位置信息也就没了**。可测形态：把 token 重新排列（反序）后看分数矩阵")
    print("       无 RoPE：还原后最大差 %.3e（**对排列不变** ⇒ 位置确实没进算式）"
          % same_plain)
    print("       有 RoPE：还原后最大差 %.4f（排列改变了分数 ⇒ 位置真的进了算式）" % diff_rope)
    check("C.2", same_plain < SHIFT_TOL and diff_rope > SHIFT_MIN,
          "★ 两半各自都是**必需**的：去掉 `d_h^R` 那一半，元素数变小了、但位置信息也没了"
          "（无 RoPE 对排列不变：%.3e < %g；有 RoPE：%.4f > %g）"
          % (same_plain, SHIFT_TOL, diff_rope, SHIFT_MIN))

    # ── C.3 反例：**只旋转 k 不旋转 q** 不满足相对位置性质 ────────────────
    # ★ 这是控制者探测时真实踩过的坑（计划 §0.7-D 的 D15 注：`probe_80g.py` 初版的
    #   「显式」路径只旋转了 k）。拿它作差得到的是「**两个都不合法的对象之差**」。
    rng = np.random.default_rng(SEED + 1)
    qv = rng.standard_normal(D_MODEL)
    kv = rng.standard_normal(D_MODEL)
    delta = 3
    good, bad = [], []
    for m in range(delta + 1, S):
        good.append(float(rope(qv, float(m)) @ rope(kv, float(m - delta))))
        bad.append(float(qv @ rope(kv, float(m - delta))))     # ← q 没旋转
    spread_good = max(good) - min(good)
    spread_bad = max(bad) - min(bad)
    print()
    print("  C.3：固定 `delta=%d`、扫 `m`（`n = m − delta`），看分数是否只依赖 `delta`：" % delta)
    print("       两边都旋转（正确）：极差 %.3e（**只依赖 delta**）" % spread_good)
    print("       只转 k 不转 q（错误）：极差 %.4f（**随 m 变**）" % spread_bad)
    check("C.3", spread_good < SHIFT_TOL and spread_bad > SHIFT_MIN,
          "★ RoPE 要求 **q 与 k 都被旋转**：只转 k 的写法**不满足**相对位置性质"
          "（极差 %.4f > %g），所以拿它作差是「两个都不合法的对象之差」"
          " —— 这正是计划 §0.7-D 给 `probe_80g.py` 记下的那处瑕疵" % (spread_bad, SHIFT_MIN))

    # ── C.4 ★★ M6 的**对照组**：一个**共享代码的「假参照」** ⇒ 断言「测不出来」 ──
    # ★★ 计划 §3 Chunk 2 的验收要求：「把参考实现与被测实现**共享一段代码**，断言应当
    #    **测不出来**（即这个「假通过」现象**必须能被复现**）」。
    # ★ `A.1` 的判据是容差式（`< 1e-12`）：假参照**直接调用被测实现**时，逐元素差恒为
    #    **精确的 `0.0`**，而它**照样通过** ⇒ 容差对这种情况**没有分辨力**。
    # ★ 揭穿它的是**机械检查**：`inspect.getsource` 不等（`A.3`）与 **AST 调用表**
    #    （下面用的是与 `D.6` 同一个探测器）。
    # ★ `== 0.0` 在这里是**合法用法**：它落在 `00-meta/glossary.md` 的「reference implementation（参考实现）」条说的「**两种相反
    #    含义**」的**第二种**（共享代码 ⇒ 测不出来），不是第一种（正确）。
    def mla_shared_ref(xx, wq, wdkv, wuk):
        """★ 故意的「假参照」：它**直接调用被测实现**（这就是 P17 的现场）。"""
        return mla_explicit(xx, wq, wdkv, wuk)[0]

    c_now = x @ w_dkv
    d_shared = float(np.abs(mla_explicit(x, w_q, w_dkv, w_uk)[0]
                            - mla_shared_ref(x, w_q, w_dkv, w_uk)).max())
    d_indep = float(np.abs(mla_explicit(x, w_q, w_dkv, w_uk)[0]
                           - mla_absorbed(x, w_q, w_uk, c_now)).max())
    caught_shared = "mla_explicit" in _callee_names(mla_shared_ref)   # ← 探测器
    caught_real = "mla_explicit" in _callee_names(mla_absorbed)
    print()
    print("  C.4：★★ 对照组（M6）——「参照」是一个**直接调用被测实现**的假参照：")
    print("       逐元素差 = **%.1f**（精确 0.0！）而 A.1 的判据 `< %g` **照样通过**"
          % (d_shared, ABSORB_TOL))
    print("       真独立时同一个量 = %.3e ≠ 0.0 ⇒ 两种情形**只是量级不同**，"
          "容差**分辨不出来**" % d_indep)
    print("       ★ 但 AST 调用表**分辨得出来**：假参照里含 `mla_explicit` = %s，"
          "真参照里含 = %s" % (caught_shared, caught_real))
    check("C.4", d_shared == 0.0 and d_indep != 0.0
          and caught_shared and not caught_real,
          "★ 假通过**可复现**：共享代码 ⇒ 差是**精确 `0.0`**（%.1f），独立路径 ⇒ %.3e "
          "—— 两者都 `< %g`。★ 揭穿它的是**AST 调用表**（假参照 True / 真参照 False）"
          "与调用计数为 0（`A.2`），**不是**容差" % (d_shared, d_indep, ABSORB_TOL))


# ===========================================================================
# 9. D 类：失败标记自检 + 口径钉住
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    ★ 它同时是**参照实现独立性**的探测器（`D.6`）与**假通过**的探测器（`C.4`）。
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

    ★ 为什么放最前：这几条是**所有判据的前提**，而且必须**先于**任何「形状对不上」
      这类**结构性崩溃**报出来。否则那些变异会被 `make mutate` 记成**崩溃捕获**，
      而崩溃捕获**不算真捕获**（它证明「代码炸了」，不是「判据抓住了错」）。
      本主题实测：`D_MODEL` / `D_C` 被改后，`rope()` 会因维度不是偶数先抛
      `ValueError` —— 先钉住之后，这些变异全部落进**断言捕获**。
    """
    print("=" * 74)
    print("D. 口径/阈值/specs 的钉住（★ 放在 A/B/C **之前**：先钉住前提，再看数）")
    print("=" * 74)

    # ── 口径常量 ─────────────────────────────────────────────────────────
    measured = {"D_MODEL": D_MODEL, "D_C": D_C, "S": S, "SEED": SEED,
                "ROPE_BASE": ROPE_BASE, "L_LAYERS": L_LAYERS, "N_H_TEST": N_H_TEST}
    expect = {"D_MODEL": 32, "D_C": 4, "S": 12, "SEED": 20260922,
              "ROPE_BASE": 10000.0, "L_LAYERS": 4, "N_H_TEST": 32}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 7 个逐位一致（它们是**结论所引用的测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    # ── 阈值 ─────────────────────────────────────────────────────────────
    th = {"ABSORB_TOL": ABSORB_TOL, "ROPE_DIFF_MIN": ROPE_DIFF_MIN,
          "DIFF_RATIO_MIN": DIFF_RATIO_MIN, "FIT_TOL": FIT_TOL,
          "FIT_MIN": FIT_MIN, "SHIFT_TOL": SHIFT_TOL, "SHIFT_MIN": SHIFT_MIN}
    th_expect = {"ABSORB_TOL": 1e-12, "ROPE_DIFF_MIN": 1.0,
                 "DIFF_RATIO_MIN": 1e15, "FIT_TOL": 1e-10,
                 "FIT_MIN": 1e-2, "SHIFT_TOL": 1e-12, "SHIFT_MIN": 1e-1}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 7 个逐位一致（含 3 个**机器精度档**：ABSORB_TOL / FIT_TOL / SHIFT_TOL —— "
          "放宽它们，判据就退化成恒真）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── ★ B.2 那条判据的**非空条件**：`d_c < S` ───────────────────────────
    check("D.3", D_C < S,
          "★ `B.2` 的非空条件：`d_c=%d < S=%d` —— 只有压缩维度真的小于序列长度，"
          "「用 `d_c` 维因子复现分数」才**可能失败**（`d_c ≥ S` 时残差恒为 0，那是空断言）"
          % (D_C, S))

    # ── specs 的常量逐位钉住 ──────────────────────────────────────────────
    check("D.4", (specs.DEEPSEEK_V2_D_C_OVER_D_H, specs.DEEPSEEK_V2_D_H_R_OVER_D_H) == (4, 0.5),
          "★ `specs.py` 的 MLA 两个比例**逐位钉住** = (%s, %s)（[C] 级，论文 §2.1.3 逐字）"
          % (specs.DEEPSEEK_V2_D_C_OVER_D_H, specs.DEEPSEEK_V2_D_H_R_OVER_D_H))


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
    check("D.6", ASSERT_FAIL_MARK == "@@ FAIL: attention-variants"
          and "FAIL" in ASSERT_FAIL_MARK
          and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.7", OTHER_THEME_MARKS == ("!! FAIL: transformer", "## FAIL: moe",
                                       "~~ FAIL: normalization",
                                       "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    # ── 参照实现的独立性（AST 级）─────────────────────────────────────────
    watched = {mla_c, mla_explicit, mla_absorbed, draw_mla,
               scores_rope_explicit, scores_rope_absorbed_wrong}
    wnames = {f.__name__ for f in watched}
    called = _callee_names(mla_absorbed)
    shared = sorted(wnames & called)
    counted_src = inspect.getsource(mla_c) + inspect.getsource(mla_explicit)
    check("D.8", "CALLS[" in counted_src and not shared,
          "被测的 `mla_c` / `mla_explicit` 带 `CALLS` 计数；参照（吸收）实现的 **AST 调用表里"
          "一个都没有它们**（也没共用 `draw_mla`）⇒ A.2 的「计数为 0」不是空断言%s"
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
    print("02_mla_low_rank.py —— MLA：低秩压缩、矩阵吸收、与 RoPE 的结构性冲突")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区（计划 §2 的边界表）：本程序**不**重推 RoPE 的定义")
    print("    （正文在 positional-encoding/01，此处回链）；**不**算 KV 的字节")
    print("    （六项乘积的正文在 130-model-optimization/quantization/18-kv-cache-basics:48-76）。")
    pin_part()      # ★ D.1–D.4：先钉住前提（见 pin_part 的说明）
    part_A()
    part_B()
    part_C()
    mark_part()     # ★ D.5–D.8：标记自检与独立性
    return summary()


if __name__ == "__main__":
    sys.exit(main())
