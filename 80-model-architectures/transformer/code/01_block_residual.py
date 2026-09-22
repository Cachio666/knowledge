#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_block_residual.py —— 一个 block 的构成，与「残差流」这条贯穿通路。

配套文档：`01-block-and-residual-stream.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形：编号程序本机可跑 ⇒ 断言在程序内），
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本程序的数字**全部是推导值或本机实测值**，测量条件是：

> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 它们支撑的是「**残差流的形状决定了它的数值条件**」这类**结构**结论，
> 不支撑任何关于真实模型训练或 GPU 的结论。

★★ 可证伪断言（把这四段改错，本程序必须非零退出）：

  · `A.1`  pre-LN 的逐层增量比值落在 [0.7, 1.3]（把 `h + F(LN(h))` 写成 `LN(h) + F(h)`
           或去掉 LN，比值立刻越界）；
  · `A.2`  **两条独立代码路径**给出同一条残差流（相对差 < 1e-14），
           **且参照实现调用被测实现的函数次数恰为 0**（把参照改成调用 `layernorm()` 立刻失败）；
  · `B.1`  post-LN 的残差流范数 == √D（相对误差 < 1e-6）——把 `LN(h + F(h))` 写成
           `h + F(h)` 或 `LN(h) + F(h)` 立刻失败；
  · `C.1`  `final_norm` 前后接 `lm_head` 的 logits std 之比 > 10×（把 `layernorm()` 那一层
           去掉立刻失败）。

用法：
    python3 01_block_residual.py
"""

import ast
import collections
import inspect
import os
import sys

import numpy as np

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（transformer）取 `!! FAIL: transformer`。
#   ★ 这个字面量**不是**照抄任何别的主题的：域 README 把五个主题的标记写死为
#     五个不同形态（`transformer` 用 `!!`），**不跨主题照抄**是本库纪律。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；
#     `D.1`/`D.2` 两条断言守着它：文件里恰好一次、且与其他四个主题的标记都不同。
ASSERT_FAIL_MARK = "!! FAIL: transformer"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("@@ FAIL: attention-variants", "## FAIL: moe",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

FAILS = []
MARKS_USED = set()
# ★ 计数器用 `Counter()` 而**不是**字面量字典：`mutate.py` 的分母是「模块级常量的
#   **字面量叶子**」，写成 `{"A": 0, ...}` 会把 8 个**不承载任何判据**的初值塞进分母
#   （改它们只动打印出来的汇总行，不动任何结论）⇒ 那是给分母灌水。
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
# 1. 口径常量（★ 全部在 `D.3` 被钉住：改一个字节，本程序就报错）
# ===========================================================================
# ★★ 为什么「口径常量」也要钉住：它们不是「可调的参数」，而是**结论所引用的测量条件**。
#    文档 §5 与本程序的输出都写着「D=256 / L=24 / N=64 / 5 个种子」——
#    静默改掉其中任何一个，**输出会变而文档不会**，这正是本域最想防的那类漂移。
#    （先例：`45-kernel-lifecycle/requirement-discovery/code/01_operator_inventory.py:526-541`
#     的「C6：逐个字段钉住」——那里也是把常量本身当断言对象。）
HIDDEN = 256            # 残差流宽度 D
DEPTH = 24              # 层数 L
N_TOKENS = 64           # 每批的 token 数
SEEDS = 5               # 跨种子重复次数
SEED_BASE = 3000        # 第一个种子（= 探测脚本 probe_80d.py 用的 3000）
INIT_SCALE = 0.5        # x_0 的初始化尺度（probe_80d.py 同）
WEIGHT_B_SCALE = 0.01   # 子层偏置的初始化尺度（probe_80d.py 同）
LN_EPS = 1e-6           # LayerNorm 的 eps（★ normalization 主题的正题，此处只当口径）
VOCAB = 32000           # `lm_head` 的输出宽度（只影响 logits 的抽样噪声，不影响结论）

# ── 阈值 ────────────────────────────────────────────────────────────────────
# ★ 每条阈值都在 `D.4` 被钉住，并**说明它属于哪一档**（机器精度档 / 量级档 / 区间）。
INC_RATIO_LO = 0.7       # A.1 区间下界
INC_RATIO_HI = 1.3       # A.1 区间上界
REF_REL_TOL = 1e-14      # A.2 两条独立路径的相对差（**机器精度档**）
POST_LN_REL_TOL = 1e-6   # B.1 post-LN 的 ‖h‖ 与 √D 的相对误差
POST_LN_EPS0_REL_TOL = 1e-14   # B.2 eps=0 档（**机器精度档**）
FINAL_NORM_RATIO_MIN = 10.0    # C.1 量级档
FINAL_NORM_CLOSED_FORM_TOL = 0.02   # C.2 与闭式 ‖h_L‖/√D 的相对差（有限 VOCAB 的抽样噪声）
DENOM_FACTOR_MIN = 2.0   # C.3 两个分母口径给出的倍数至少差这么多倍

# 调用计数器：**参照实现独立性**的证据（`00-meta/glossary.md:240` 的机械判据）。
# ★ 被测实现的这两个函数每被调一次就 +1；参照实现跑完后，计数**必须不动**。
CALLS = {}


# ===========================================================================
# 2. 被测实现（路径 1）
# ===========================================================================
def layernorm(x, eps):
    """LayerNorm —— ★ **公式的正文在 `40-kernels/softmax-normalization/03:55-71`**
    （正文在彼，此处回链）。本程序只用它，不重推它。"""
    CALLS["layernorm"] = CALLS.get("layernorm", 0) + 1
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def sublayer(h, W, b):
    """一个子层 F。★ 这里刻意用**最简的形态**（`tanh` + 两个方阵），
    因为本程序要测的是**残差流的形状**，不是某个具体子层的能力——
    换一个子层，A.1/B.1 的**结论形态**不变（见 `C.4` 的反例段）。"""
    CALLS["sublayer"] = CALLS.get("sublayer", 0) + 1
    return np.tanh(h @ W) @ W.T + b


def draw_params(seed, depth, hidden, n_tokens, init_scale, b_scale):
    """★ **初始化与抽取顺序是口径的一部分**（probe_80d.py 同）：
    `x_0` → 全部 `W`（`N(0,1)/√D`）→ 全部 `b`（`N(0,1)·0.01`）。
    参照实现必须**逐个一致**地抽，否则两条路径比的不是同一条残差流。"""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n_tokens, hidden)) * init_scale
    W = [rng.standard_normal((hidden, hidden)) / np.sqrt(hidden) for _ in range(depth)]
    b = [rng.standard_normal(hidden) * b_scale for _ in range(depth)]
    return x0, W, b


def stream_pre_ln(seed, depth, hidden, n_tokens, init_scale, eps):
    """被测实现：pre-LN 的残差流 `h ← h + F(LN(h))`。

    返回**每层之后**的 ‖h‖（每 token 平均）—— ★ 含 `norms[0] = ‖x_0‖`，
    所以 `diff(norms)` 有 `depth` 项，与文档 A.1 的口径一致。
    """
    x0, W, b = draw_params(seed, depth, hidden, n_tokens, init_scale, WEIGHT_B_SCALE)
    h = x0
    norms = [float(np.linalg.norm(h, axis=-1).mean())]
    for l in range(depth):
        h = h + sublayer(layernorm(h, eps), W[l], b[l])
        norms.append(float(np.linalg.norm(h, axis=-1).mean()))
    return np.array(norms)


def stream_post_ln(seed, depth, hidden, n_tokens, init_scale, eps):
    """被测实现：post-LN 的残差流 `h ← LN(h + F(h))`。"""
    x0, W, b = draw_params(seed, depth, hidden, n_tokens, init_scale, WEIGHT_B_SCALE)
    h = x0
    norms = [float(np.linalg.norm(h, axis=-1).mean())]
    for l in range(depth):
        h = layernorm(h + sublayer(h, W[l], b[l]), eps)
        norms.append(float(np.linalg.norm(h, axis=-1).mean()))
    return np.array(norms)


# ===========================================================================
# 3. ★★ 参照实现（路径 2）—— **不调用被测实现的任何函数**
# ===========================================================================
# ★ `CONTRIBUTING:232-235`：A 类必须有一条与被检验代码路径**不同**的参照；
#   `00-meta/glossary.md:240` 给出机械判据：**参考实现不得调用被测实现的任何函数**
#   （调用计数器必须恰为 0），**且不得比被测实现更复杂**。
# ★★ 两条路径的差别是**真的**，不是换了个名字：
#     · 被测：按层 `for` 循环 + 两个命名函数 + `@`（`x @ W` 与 `y @ W.T`）；
#     · 参照：`functools.reduce` 的 scan + **内联**的归一化（`np.var`，不是
#       「先减均值再平方求均值」）+ `np.einsum` 的两条收缩路径。
#   `D.5` 用 `inspect.getsource(...)` 不等来挡住「两边其实是同一行代码」（P17）。
def stream_pre_ln_ref(seed, depth, hidden, n_tokens, init_scale, eps):
    """独立参照：重算 pre-LN 的残差流，**一次都不调用** `layernorm()` / `sublayer()`。"""
    import functools

    rng = np.random.default_rng(seed)
    h0 = rng.standard_normal((n_tokens, hidden)) * init_scale
    Ws = np.stack([rng.standard_normal((hidden, hidden)) / np.sqrt(hidden)
                   for _ in range(depth)])
    bs = np.stack([rng.standard_normal(hidden) * WEIGHT_B_SCALE for _ in range(depth)])

    def step(hh, l):
        mu = hh.mean(axis=-1, keepdims=True)
        var = np.var(hh, axis=-1, keepdims=True)             # ← 不同的归一化路径
        z = (hh - mu) / np.sqrt(var + eps)
        t = np.tanh(np.einsum("nd,de->ne", z, Ws[l]))        # ← 不同的收缩写法
        y = np.einsum("ne,de->nd", t, Ws[l])
        return hh + y + bs[l]                                # ← 偏置加法位置不同

    out = functools.reduce(lambda acc, l: acc + [step(acc[-1], l)],
                           range(depth), [h0])
    return np.array([float(np.linalg.norm(v, axis=-1).mean()) for v in out])


def stream_post_ln_ref(seed, depth, hidden, n_tokens, init_scale, eps):
    """独立参照：重算 post-LN 的残差流（同上的独立性）。

    ★ 连**参数抽取**也不调 `draw_params()` —— 它自己重抽一遍（顺序与口径一致）。
      「共享的只有**口径常量**与**随机数抽取顺序**，没有共享任何函数」才叫独立。
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n_tokens, hidden)) * init_scale
    Ws = [rng.standard_normal((hidden, hidden)) / np.sqrt(hidden) for _ in range(depth)]
    bs = [rng.standard_normal(hidden) * WEIGHT_B_SCALE for _ in range(depth)]
    h = x0
    norms = [float(np.linalg.norm(h, axis=-1).mean())]
    for l in range(depth):
        pre = h + np.tanh(np.einsum("nd,de->ne", h, Ws[l])) @ Ws[l].T + bs[l]
        mu = pre.mean(axis=-1, keepdims=True)
        h = (pre - mu) / np.sqrt(np.var(pre, axis=-1, keepdims=True) + eps)
        norms.append(float(np.linalg.norm(h, axis=-1).mean()))
    return np.array(norms)


# ===========================================================================
# 4. A 类：残差流本身的形状
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 残差流：pre-LN 的增量、与两条独立路径的一致性")
    print("=" * 74)
    print("  口径：D=%d · L=%d · N=%d · %d 个种子（%d…）· x_0 尺度 %g · "
          "W~N(0,1)/√D · b~N(0,1)·%g · LN eps=%g"
          % (HIDDEN, DEPTH, N_TOKENS, SEEDS, SEED_BASE, INIT_SCALE, WEIGHT_B_SCALE, LN_EPS))

    pre = np.array([stream_pre_ln(SEED_BASE + s, DEPTH, HIDDEN, N_TOKENS,
                                  INIT_SCALE, LN_EPS) for s in range(SEEDS)])
    mean_norms = pre.mean(axis=0)
    inc = np.diff(mean_norms)
    head, tail = inc[:5].mean(), inc[-5:].mean()
    print("  残差流范数（跨种子均值）：首层之后 %.2f → 末层 %.2f" % (mean_norms[1], mean_norms[-1]))
    print("  逐层增量的首 5 项：%s" % np.round(inc[:5], 2))
    print("  逐层增量的末 5 项：%s" % np.round(inc[-5:], 2))
    print("  ⇒ 增量 首 5 项均值 %.2f → 末 5 项均值 %.2f（比值 %.4f）"
          % (head, tail, tail / head))
    print("  ⇒ 换个口径（首项 → 末项）：%.2f → %.2f（比值 %.4f）"
          % (inc[0], inc[-1], inc[-1] / inc[0]))
    print("    ★ 两个口径**都能落进同一个判定区间**，但它们不是同一个统计量 ——"
          " 文档 §4 记着这条（D10 的 `12.32 → 9.89` 是**首项→末项**，"
          "而判据①写的口径是**首/末 5 项均值**）。")
    per_seed = [(np.diff(pre[s])[-5:].mean() / np.diff(pre[s])[:5].mean()) for s in range(SEEDS)]
    spread = max(per_seed) - min(per_seed)
    print("  逐种子的比值：%s（跨种子极差 %.4f）"
          % (", ".join("%.4f" % v for v in per_seed), spread))

    check("A.1", INC_RATIO_LO <= tail / head <= INC_RATIO_HI,
          "逐层增量比值 %.4f ∈ [%g, %g]（★ 「**近似**恒定」，不是「恒定」："
          "末 5 项比首 5 项低 %.1f%%）" % (tail / head, INC_RATIO_LO, INC_RATIO_HI,
                                        (1 - tail / head) * 100))

    # ── ★★ 两条独立路径 ────────────────────────────────────────────────────
    ref = stream_pre_ln_ref(SEED_BASE, DEPTH, HIDDEN, N_TOKENS, INIT_SCALE, LN_EPS)
    impl = pre[0]
    denom = float(np.abs(impl).max())
    rel = float(np.abs(impl - ref).max()) / denom
    print("  两条路径的残差流范数：被测末层 %.6f · 参照末层 %.6f" % (impl[-1], ref[-1]))
    print("  最大绝对差 %.3e · 相对（÷ max‖·‖ = %.2f）= %.3e" % (np.abs(impl - ref).max(), denom, rel))
    check("A.2", rel < REF_REL_TOL,
          "两条**独立**代码路径给出同一条残差流（相对差 %.3e < %g）" % (rel, REF_REL_TOL))

    # ★ 参照跑完后，计数表必须**整个是空的** —— 比「那两个键是 0」更强：
    #   它等于「参照一次都没调用被测侧的任何函数」（含将来新增的函数）。
    CALLS.clear()
    stream_pre_ln_ref(SEED_BASE, DEPTH, HIDDEN, N_TOKENS, INIT_SCALE, LN_EPS)
    stream_post_ln_ref(SEED_BASE, DEPTH, HIDDEN, N_TOKENS, INIT_SCALE, LN_EPS)
    check("A.3", not CALLS,
          "★ 参照实现跑完后**计数表完全为空**（记录到的调用：%s）⇒ "
          "它一次都没调用被测实现的函数" % (sorted(CALLS) if CALLS else "无"))
    return pre, mean_norms, inc


# ===========================================================================
# 5. B 类：post-LN 的残差流恒为 √D
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 残差流范数在 post-LN 下恒为 √D")
    print("=" * 74)

    post = np.array([stream_post_ln(SEED_BASE + s, DEPTH, HIDDEN, N_TOKENS,
                                    INIT_SCALE, LN_EPS) for s in range(SEEDS)])
    last = post[:, -1]
    rel = float(np.abs(last / np.sqrt(HIDDEN) - 1.0).max())
    print("  √D = %.10f" % np.sqrt(HIDDEN))
    print("  post-LN 末层范数（5 个种子）：%s"
          % ", ".join("%.9f" % v for v in last))
    print("  → 相对误差最大 %.3e（跨种子极差 %.2e，与 D11 的 1.5e-08 同档）"
          % (rel, float(last.max() - last.min())))
    check("B.1", rel < POST_LN_REL_TOL,
          "post-LN 的 ‖h_L‖ == √D（相对误差 %.3e < %g）" % (rel, POST_LN_REL_TOL))

    # ★ eps 才是那个「不等」的来源：eps=0 时它是**恒等式**
    z = np.array([stream_post_ln(SEED_BASE + s, DEPTH, HIDDEN, N_TOKENS, INIT_SCALE, 0.0)
                  for s in range(SEEDS)])
    rel0 = float(np.abs(z[:, -1] / np.sqrt(HIDDEN) - 1.0).max())
    print("  eps=0 时末层范数：%s" % ", ".join("%.12f" % v for v in z[:, -1]))
    check("B.2", rel0 < POST_LN_EPS0_REL_TOL,
          "★ eps=0 时它是**恒等式**：相对误差 %.3e < %g（B.1 的 %.3e 全部来自 eps）"
          % (rel0, POST_LN_EPS0_REL_TOL, rel))
    return post


# ===========================================================================
# 6. C 类：final_norm 的存在理由 / 口径边界 / 反例
# ===========================================================================
def part_C(mean_norms, inc):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)
    print("  C.1/C.2：末层残差流直接接 `lm_head`，还是先过 `final_norm`？")

    seed = SEED_BASE + 4000          # 4242（= probe_80d.py【4】的种子）
    x0, W, b = draw_params(seed, DEPTH, HIDDEN, N_TOKENS, INIT_SCALE, WEIGHT_B_SCALE)
    h = x0
    for l in range(DEPTH):
        h = h + sublayer(layernorm(h, LN_EPS), W[l], b[l])
    rng = np.random.default_rng(seed + 1)
    E = rng.standard_normal((HIDDEN, VOCAB)) / np.sqrt(HIDDEN)   # lm_head 权重
    raw = float((h @ E).std())
    normed = float((layernorm(h, LN_EPS) @ E).std())
    ratio = raw / normed
    h_norm = float(np.linalg.norm(h, axis=-1).mean())
    closed = h_norm / np.sqrt(HIDDEN)            # ← 闭式预测（独立于 logits 的抽样）
    print("     ‖h_L‖（每 token 平均）= %.2f · 未归一化 logits std = %.4f · "
          "过 final_norm 后 = %.4f" % (h_norm, raw, normed))
    print("     ⇒ 比值 = %.3f×" % ratio)
    check("C.1", ratio > FINAL_NORM_RATIO_MIN,
          "final_norm 前后接 lm_head 的 logits std 之比 %.3f× > %.0f×"
          % (ratio, FINAL_NORM_RATIO_MIN))

    # ★ 闭式：logits 的第 j 列是 h·e_j，e_j 的每个分量 ~N(0,1/D)
    #   ⇒ std(h·e_j) = ‖h‖/√D（对整批 token 求 std，样本量 N=64）。
    #   过 final_norm 后 ‖LN(h)‖ = √D ⇒ 归一化后的 std = √D/√D = 1。
    #   ⇒ **比值应当 == ‖h_L‖/√D**。这是一条与被测路径**完全独立**的参照（解析式）。
    relc = abs(ratio / closed - 1.0)
    print("     闭式预测 ‖h_L‖/√D = %.4f ⇒ 与实测比值的相对差 %.4f（有限 VOCAB=%d 的抽样噪声）"
          % (closed, relc, VOCAB))
    check("C.2", relc < FINAL_NORM_CLOSED_FORM_TOL,
          "实测比值与**闭式** ‖h_L‖/√D 的相对差 %.4f < %g（独立参照：解析式）"
          % (relc, FINAL_NORM_CLOSED_FORM_TOL))

    # ── C.3 ★ 口径边界：同一个「增长倍数」，两个分母差 2.5 倍 ──────────────
    print()
    print("  C.3：同一个量、两个分母 —— 「残差流涨了几倍」必须写明分母是什么")
    r_x0 = mean_norms[-1] / mean_norms[0]
    r_x1 = mean_norms[-1] / mean_norms[1]
    print("     分母取**初始 x_0**（|x_0| = %.2f）：%.2f×" % (mean_norms[0], r_x0))
    print("     分母取**第一层之后**（%.2f）：%.2f×" % (mean_norms[1], r_x1))
    print("     ★ 后者才是「逐层的账」的口径：x_0 只是初始化尺度，"
          "第一层的跳跃（%.2f，是 x_0 的 %.1f 倍）是**初始化**效应，不是深度效应。"
          % (mean_norms[1], mean_norms[1] / mean_norms[0]))
    check("C.3", r_x0 / r_x1 > DENOM_FACTOR_MIN,
          "两个分母口径给出的倍数 %.2f× vs %.2f×，差 %.2f 倍 > %g"
          % (r_x0, r_x1, r_x0 / r_x1, DENOM_FACTOR_MIN))

    # ── C.4 ★ 反例：一个被广泛转述、但在本设定下**不成立**的预期 ───────────
    print()
    print("  C.4：反例 —— 「pre-LN 里去掉 LN，残差流会**发散**」在本设定下**不成立**")
    h2 = x0
    no_ln = [float(np.linalg.norm(h2, axis=-1).mean())]
    for l in range(DEPTH):
        h2 = h2 + sublayer(h2, W[l], b[l])
        no_ln.append(float(np.linalg.norm(h2, axis=-1).mean()))
    print("     有 LN：末层 %.2f（首层 %.2f 的 %.2f×）"
          % (mean_norms[-1], mean_norms[1], r_x1))
    print("     无 LN：末层 %.2f（首层 %.2f 的 %.2f×）"
          % (no_ln[-1], no_ln[1], no_ln[-1] / no_ln[1]))
    check("C.4", no_ln[-1] / mean_norms[-1] < 2.0,
          "去掉 LN 只是把末层范数抬高到 %.2f× （%.2f vs %.2f），**不是发散** —— "
          "因为 `tanh` 把子层输出压在 O(√D)。★ 这条反例说明「去掉 LN 就发散」"
          "依赖**子层自身的形态**，不能当通式。"
          % (no_ln[-1] / mean_norms[-1], no_ln[-1], mean_norms[-1]))
    return ratio


# ===========================================================================
# 7. D 类：失败标记自检 + 口径钉住
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


def part_D(pre):
    print()
    print("=" * 74)
    print("D. 失败标记自检 与 口径/阈值钉住")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)

    # ★ 失败行的前缀就是标记本身（`mutate.py` 用**行首锚定**判定），且本程序打印时用的标记
    #   只有一个取值 —— 全部通过时它为空集，出现失败时它只能是本主题的那一个。
    # ★ 「等于该值」：把**本主题被分配的那个字面量**写死在这里。
    #   ★ 这不是同义反复：`mutate.py` 只变异**模块级**常量，函数体里的这个字面量
    #     不在它的分母里 ⇒ 它正是用来挡住「标记被改掉 / 被跨主题照抄」的那把锁。
    check("D.2", ASSERT_FAIL_MARK == "!! FAIL: transformer"
          and "FAIL" in ASSERT_FAIL_MARK
          and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.2b", OTHER_THEME_MARKS == ("@@ FAIL: attention-variants", "## FAIL: moe",
                                        "~~ FAIL: normalization",
                                        "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    # ★ 口径常量：**结论引用的测量条件**，改一个字节就会让输出与文档脱钩。
    measured = {
        "HIDDEN": HIDDEN, "DEPTH": DEPTH, "N_TOKENS": N_TOKENS, "SEEDS": SEEDS,
        "SEED_BASE": SEED_BASE, "INIT_SCALE": INIT_SCALE,
        "WEIGHT_B_SCALE": WEIGHT_B_SCALE, "LN_EPS": LN_EPS, "VOCAB": VOCAB,
    }
    expect = {"HIDDEN": 256, "DEPTH": 24, "N_TOKENS": 64, "SEEDS": 5, "SEED_BASE": 3000,
              "INIT_SCALE": 0.5, "WEIGHT_B_SCALE": 0.01, "LN_EPS": 1e-6, "VOCAB": 32000}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "口径常量 9 个逐位一致（这些是**结论的测量条件**；改任一即失败）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    print("     残差流范数的形状：norms 长度 %d = L+1（含 x_0）⇒ diff 有 %d 项"
          % (pre.shape[1], pre.shape[1] - 1))

    # ★ 阈值：**逐个说明它属于哪一档**——机器精度档的阈值一旦被悄悄放宽，
    #   整条判据就从「测出来」退化成「总会通过」。
    th = {"INC_RATIO_LO": INC_RATIO_LO, "INC_RATIO_HI": INC_RATIO_HI,
          "REF_REL_TOL": REF_REL_TOL, "POST_LN_REL_TOL": POST_LN_REL_TOL,
          "POST_LN_EPS0_REL_TOL": POST_LN_EPS0_REL_TOL,
          "FINAL_NORM_RATIO_MIN": FINAL_NORM_RATIO_MIN,
          "FINAL_NORM_CLOSED_FORM_TOL": FINAL_NORM_CLOSED_FORM_TOL,
          "DENOM_FACTOR_MIN": DENOM_FACTOR_MIN}
    th_expect = {"INC_RATIO_LO": 0.7, "INC_RATIO_HI": 1.3, "REF_REL_TOL": 1e-14,
                 "POST_LN_REL_TOL": 1e-6, "POST_LN_EPS0_REL_TOL": 1e-14,
                 "FINAL_NORM_RATIO_MIN": 10.0, "FINAL_NORM_CLOSED_FORM_TOL": 0.02,
                 "DENOM_FACTOR_MIN": 2.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "阈值 8 个逐位一致（含 2 个**机器精度档**：REF_REL_TOL / "
                            "POST_LN_EPS0_REL_TOL —— 放宽它们，判据就退化成恒真）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    src_impl = inspect.getsource(stream_pre_ln)
    src_ref = inspect.getsource(stream_pre_ln_ref)
    check("D.5", src_impl != src_ref,
          "★ 防 P17：被测与参照的**源码不同**（`inspect.getsource` 比较，%d vs %d 字节）"
          % (len(src_impl), len(src_ref)))

    # ★ 参照实现**不许调用**被测侧任何函数 —— 用 **AST** 看真实的 `Call` 节点
    #   （★ 不能用文本包含：参照函数的**文档串**里就写着这些名字，那是注释不是调用；
    #     本库的纪律是「AST 而非 grep」，`make lint` 同理）。
    watched = {layernorm, sublayer, draw_params, stream_pre_ln, stream_post_ln}
    wnames = {f.__name__ for f in watched}
    called = _callee_names(stream_pre_ln_ref) | _callee_names(stream_post_ln_ref)
    shared = sorted(wnames & called)
    counted_src = inspect.getsource(layernorm) + inspect.getsource(sublayer)
    check("D.6", "CALLS[" in counted_src and not shared,
          "被测的那两个函数带 `CALLS` 计数；参照实现的 **AST 调用表里一个都没有它们**"
          "（也没共用 `draw_params`）⇒ A.3 的「计数为 0」不是空断言%s"
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
    print("01_block_residual.py —— 一个 block 的构成，与「残差流」这条贯穿通路")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区：本程序**不**做算子清单、FLOPs、参数量反算核对"
          "（正文在 45-kernel-lifecycle/requirement-discovery，此处回链）；"
          "也**不**重推 LayerNorm 的公式（正文在 40-kernels/softmax-normalization/03:55-71）。")
    pre, mean_norms, inc = part_A()
    part_B()
    part_C(mean_norms, inc)
    part_D(pre)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
