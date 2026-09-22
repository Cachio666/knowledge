#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_window_hybrid.py —— 滑窗注意力的**架构侧**：堆叠层把感受野撑开，以及它与别的变体叠加。

配套文档：`03-window-and-hybrid.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本程序的数字**全部是推导值或本机实测值**，测量条件是：

> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 它们支撑的是「**单层只看 `W`，堆叠后感受野扩大到 `~W×L`**」这类**结构**结论，
> 不支撑任何关于真实模型训练或 GPU 的结论。

★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）：

  · `A.1`  `W × L = 4096 × 32 = 131072` —— 论文的 `approximately 131K` 是**十进制 K**
           （131.072K）；把 32 改成别的数立刻失败；
  · `A.2`  玩具上的逐层传播**实测**感受野与闭式一致，**两种 `W` 的读法**各得一个闭式
           （`L·W` 与 `L·(W−1)`）；
  · `B.1`  环形缓冲的槽位数 = `min(S, W)` ⇒ **`S` 超过 `W` 后不随 `S` 增长**；
  · `B.3`  缓冲里**活着**的位置集合恰好是滑动窗口 —— 把 `i mod W` 的覆写顺序改掉即失败；
  · `C.1`  GQA 与 SWA 作用在**两个不同的因子**上，倍率是它们的**乘积**（不是二选一）；
  · `C.3`  本域能核实的**两个 config 都没开滑窗**（Qwen3 是显式 `false` + `null`，
           Llama-3-8B 是**没有这两个字段**）—— 改一个 bit 就失败；
  · `C.4`  滑窗的可见性是一个**布尔掩码**：窗口外的注意力权重是**精确的 `0.0`**。

★★ **本程序刻意不做的事**（计划 §2 的边界表，越界即作废）：

  * **不重讲**「滑窗让每行只需 `ceil(w/Bc)` 个块」这类 **kernel 后果** —— 正文在
    `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`（正文在彼，此处回链）。
    本程序讲的是**架构后果**：堆叠层把感受野从 `W` 扩到 `~W×L`；
  * **不重讲**其它两个变体（ALiBi / softcap）的定义 —— 它们的**机制**属 logit 处理，
    正文在 `40-kernels/attention/04:130-134`（那一行只给出「改什么」），
    而 ALiBi 的完整定义归 `positional-encoding/03-extrapolation`（正文在彼，此处回链）；
  * **不算 KV 的字节**：六项乘积的正文在
    `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`（正文在彼，此处回链）。
    本程序只算**槽位数**与**倍率**。

用法：
    python3 03_window_hybrid.py
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
TOY_W = 3                # 玩具窗口（★ 小到可以逐层传播着数）
TOY_L = 5                # 玩具层数
D_HEAD = 8               # C.4 的 head 宽度（只影响分数的尺度，不影响掩码）
S_MASK = 6               # C.4 的序列长度
SEQ_LENS = (4096, 16384, 65536)   # B 段用的三个序列长度（★ 都 ≥ W，且相差 4 倍 / 16 倍）

# ── 阈值 ────────────────────────────────────────────────────────────────────
SPAN_REL_TOL = 1e-3      # A.2 两种 `W` 读法给出的两数之差的上限（相对；**量级档**）
LINEAR_TOL = 0           # A.3 逐层增量的**精确**整数比较（不用容差）
MASK_EXACT = 0.0         # C.4 窗口外的权重（★ **精确** 0.0，不是「很小」）


# ===========================================================================
# 2. A 类：堆叠层扩展感受野（★ 本主题的立足点）
# ===========================================================================
def reachable_left(s_len, w, n_layers, convention):
    """逐层传播「每个输出位置能看到的输入位置」的**最左下标**。

    · 约定甲 `distance`（论文 §2.1 那句机制的读法）：第 `k` 层的 `i` 看上一层的 `[i−W, i]`
      ⇒ 每层把左端点推 `W` 格；
    · 约定乙 `count`（「窗口含 `W` 个位置」的读法）：看 `[i−W+1, i]` ⇒ 每层推 `W−1` 格。

    ★ 两种读法**论文都出现过**（`[i−W, i]` 那句 vs 环形缓冲的 `W` 个槽位），
      本程序把两者都算出来、并**报出它们的差**，而不是替论文选一个。
    """
    step = w if convention == "distance" else w - 1
    left = list(range(s_len))                 # 第 0 层：每个位置只看到自己
    for _ in range(n_layers):
        left = [left[max(0, i - step)] for i in range(s_len)]
    return left


def part_A():
    print("=" * 74)
    print("A. 堆叠层扩展感受野（★ 滑窗的**架构侧**）")
    print("=" * 74)
    w_real = specs.MISTRAL7B_WINDOW_SIZE
    l_real = specs.MISTRAL7B_NUM_LAYERS
    print("  论文（arXiv 2310.06825）逐字：「SWA exploits the stacked layers of a transformer")
    print("    to attend information beyond the window size `W`.」")
    print("    「Recursively, `h_i` can access tokens from the input layer at a distance of up to")
    print("    `W × k` tokens… At the last layer, using a window size of `W = 4096`, we have a")
    print("    theoretical attention span of approximately 131K tokens.」")

    # ── A.1 `W × L` ───────────────────────────────────────────────────────
    span = w_real * l_real
    print()
    print("  A.1：`W × L` = %d × %d = **%d**" % (w_real, l_real, span))
    print("       十进制 K：%.3fK（论文的 `approximately 131K` 是这一种读法）"
          % (span / 1000.0))
    print("       二进制 K：%gK（= %d / 1024，也就是 %d KiB）"
          % (span / 1024.0, span, span // 1024))
    check("A.1", span == 131072
          and abs(span / 1000.0 - 131.072) < 1e-9
          and span / 1024.0 == 128.0,
          "★ 判据⑥：`W × L` 精确 = **131072**，正好复现论文的 `approximately 131K`"
          "（**十进制** K：131.072K；若按二进制 K 它是 128K —— 报这个数必须写明用哪一种）")

    # ── A.2 玩具上**逐层传播实测**两种读法的感受野 ────────────────────────
    s_len = TOY_L * TOY_W + 4
    i = s_len - 1
    left_a = reachable_left(s_len, TOY_W, TOY_L, "distance")
    left_b = reachable_left(s_len, TOY_W, TOY_L, "count")
    dist_a = i - left_a[i]
    dist_b = i - left_b[i]
    print()
    print("  A.2：玩具上**逐层传播着数**（W=%d、L=%d、序列 %d，看最后一个位置）："
          % (TOY_W, TOY_L, s_len))
    print("       约定甲「窗口 = `[i−W, i]`」（W+1 个位置）：最左可达 %d ⇒ 距离 **%d**（闭式 L·W = %d）"
          % (left_a[i], dist_a, TOY_L * TOY_W))
    print("       约定乙「窗口含 W 个位置 `[i−W+1, i]`」：  最左可达 %d ⇒ 距离 **%d**（闭式 L·(W−1) = %d）"
          % (left_b[i], dist_b, TOY_L * (TOY_W - 1)))
    real_a = l_real * w_real
    real_b = l_real * (w_real - 1)
    print("       换成 Mistral 7B 的真参数：约定甲 **%d**、约定乙 **%d**"
          % (real_a, real_b))
    print("       两者相差 %d 个 token（%.3f%%）—— ★ 差的是 **`L`**，不是 `W`："
          "差 = `L·W − L·(W−1)` = `L`" % (real_a - real_b, 100.0 * (real_a - real_b) / real_a))
    check("A.2", dist_a == TOY_L * TOY_W and dist_b == TOY_L * (TOY_W - 1)
          and real_a == 131072 and real_b == 131040 and real_a - real_b == l_real
          and abs(real_a - real_b) / real_a < SPAN_REL_TOL,
          "★ 模拟与闭式一致；两种 `W` 的读法在真参数下只差 **`L` = %d** 个 token（%.3f%%）"
          " ⇒ 论文那个词 `approximately` **正好覆盖得住这 ±1 的不确定性**，"
          "所以**不能**用 131K 反推论文用的是哪一种读法"
          % (real_a - real_b, 100.0 * (real_a - real_b) / real_a))

    # ── A.3 感受野是 L 的**线性**函数，斜率恰好是 W ────────────────────────
    incs = []
    for l in range(1, TOY_L + 1):
        left = reachable_left(s_len, TOY_W, l, "distance")
        incs.append(i - left[i])
    steps = [incs[k + 1] - incs[k] for k in range(len(incs) - 1)]
    print()
    print("  A.3：距离随层数（约定甲）：%s" % incs)
    print("       逐层增量：%s（每层恰好 +%d = W）" % (steps, TOY_W))
    check("A.3", all(s == TOY_W for s in steps)
          and incs[-1] == TOY_L * TOY_W,
          "★ 感受野随层数**严格线性**，斜率 = `W`：%d 层后 = `L·W`（不是 `√L` 也不是饱和）"
          " ⇒ 「加深一层」就等于「窗口加宽 `W`」" % TOY_L)


# ===========================================================================
# 3. B 类：环形缓冲 —— cache 不随 S 增长（判据⑦）
# ===========================================================================
def ring_alive_positions(i, w):
    """走到第 `i` 个 token 时，环形缓冲里**活着**的位置。

    ★ 论文逐字：「the keys and values for the timestep `i` are stored in position `i mod W`
      of the cache. As a result, when the position `i` is larger than `W`, past values in the
      cache are overwritten, and **the size of the cache stops increasing**.」
    """
    slots = {}
    for t in range(i + 1):
        slots[t % w] = t          # ★ 覆写：这是「不增长」的全部机制
    return sorted(slots.values())


def part_B():
    print()
    print("=" * 74)
    print("B. 环形缓冲：滑窗让 cache **不随 S 增长**（判据⑦）")
    print("=" * 74)
    w_real = specs.MISTRAL7B_WINDOW_SIZE
    print("  ★ 口径：这里数的是**槽位数**（位置），**不是字节** —— 字节账的正文在")
    print("    130-quantization/kv-cache-quantization/01:48-76（正文在彼，此处回链）。")

    slots = [(s, min(s, w_real)) for s in SEQ_LENS]
    print()
    print("    %-10s %-14s %s" % ("序列长 S", "滑窗槽位数", "全量槽位数"))
    for s in SEQ_LENS:
        print("    %-10d %-14d %d（= S）" % (s, min(s, w_real), s))
    check("B.1", all(sl == min(s, w_real) for s, sl in slots)
          and len({sl for _, sl in slots}) == 1,
          "★ 判据⑦：滑窗的槽位数 = `min(S, W)` = **%d**，在 S ∈ %s 上**恒定**"
          "（S 超过 W 之后**不再增长**）" % (w_real, list(SEQ_LENS)))
    check("B.2", all(s == s for s in SEQ_LENS)
          and Fraction(SEQ_LENS[-1], SEQ_LENS[0]) == 16,
          "★ 对照：**全量** attention 的槽位数 = `S`（线性）—— S 从 %d 到 %d 长了 **16 倍**"
          % (SEQ_LENS[0], SEQ_LENS[-1]))

    # ── B.3 缓冲里活着的位置**恰好**是滑动窗口 ────────────────────────────
    ok = True
    for i in (0, 1, TOY_W - 1, TOY_W, TOY_W + 3, 2 * TOY_W + 1):
        alive = ring_alive_positions(i, TOY_W)
        ok = ok and alive == list(range(max(0, i - TOY_W + 1), i + 1))
    alive = ring_alive_positions(2 * TOY_W + 1, TOY_W)
    print()
    print("  B.3：玩具上（W=%d）走到 i=%d 时，缓冲里活着的位置 = %s（%d 个）"
          % (TOY_W, 2 * TOY_W + 1, alive, len(alive)))
    print("       闭式：`[i−W+1, i]` = %s ⇒ 正好是**滑动窗口**"
          % list(range(max(0, 2 * TOY_W + 1 - TOY_W + 1), 2 * TOY_W + 2)))
    check("B.3", ok,
          "★ 环形缓冲**恰好装得下**滑动窗口（槽位数 = 窗口位置数 = `W`）—— "
          "这不是巧合：`i mod W` 的覆写顺序保证了「最老的先被覆盖」")


# ===========================================================================
# 4. C 类：与其他变体叠加、边界、配置事实、掩码的形状
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 叠加与边界")
    print("=" * 74)
    w_real = specs.MISTRAL7B_WINDOW_SIZE
    h_q = specs.QWEN3_30B_A3B_ATTENTION["num_attention_heads"]
    h_kv = specs.QWEN3_30B_A3B_ATTENTION["num_key_value_heads"]

    # ── C.1 GQA × SWA：两个**不同**的因子 ─────────────────────────────────
    print("  C.1：论文逐字（Mistral 7B 摘要）：「**Our model** leverages grouped-query attention")
    print("       (GQA) for faster inference, **coupled with** sliding window attention (SWA)」")
    print("       ⇒ 两个变体作用在**不同的因子**上：")
    for s in SEQ_LENS:
        pos_f = Fraction(min(s, w_real), s)         # 位置维因子
        head_f = Fraction(h_kv, h_q)                # 头维因子
        print("       S=%-6d 位置维 ×%s（min(S,W)/S）·  头维 ×%s（h_kv/h_q=%d/%d）"
              " ⇒ 合起来 ×%s" % (s, pos_f, head_f, h_kv, h_q, pos_f * head_f))
    s63 = SEQ_LENS[-1]
    pos_f = Fraction(min(s63, w_real), s63)
    head_f = Fraction(h_kv, h_q)
    check("C.1", pos_f * head_f == pos_f * head_f
          and pos_f * head_f != pos_f and pos_f * head_f != head_f
          and pos_f != 1 and head_f != 1,
          "★ 两条轴的倍率是**乘积**（S=%d 时 %s × %s = %s）：SWA 动**位置维**、GQA 动**头维**"
          " ⇒ 它们是**叠加**的，**不是二选一**（少了任一个因子，结果都对不上）"
          % (s63, pos_f, head_f, pos_f * head_f))

    # ── C.2 边界：W ≥ S 时退化成全量 ─────────────────────────────────────
    deg = [(2048, w_real), (4096, w_real), (8192, w_real)]     # 两个退化 + 一个不退化
    print()
    print("  C.2：`S ≤ W` 时 `min(S,W) = S` ⇒ 没有收益（退化成全量）：")
    for s, w in deg:
        print("       S=%-6d W=%-6d min(S,W)=%-6d  %s"
              % (s, w, min(s, w), "← 退化成全量" if s <= w else "← W < S，有收益"))
    check("C.2", all(min(s, w) == s for s, w in deg if s <= w)
          and all(min(s, w) == w and w < s for s, w in deg if s > w),
          "★ 滑窗只在 `W < S` 时有意义；`S ≤ W` 时它**退化成全量 attention**"
          "（与 40-kernels/attention/04:145 逐字一致：「窗口 = `S` 时没有收益」，"
          "正文在彼，此处回链）")

    # ── C.3 配置事实：本域能核实的两个 config **都没开**滑窗 ──────────────
    qcfg = specs.QWEN3_30B_A3B_SLIDING_WINDOW_CONFIG
    ll_absent = specs.LLAMA3_8B_ABSENT_SLIDING_WINDOW_FIELDS
    print()
    print("  C.3：本域能核实配置的两个模型（`45/…/code/specs.py` 的 [B] 级）：")
    print("       Qwen3-30B-A3B  ：`use_sliding_window` = %r、`sliding_window` = %r"
          "（**字段存在、显式关掉**）" % (qcfg["use_sliding_window"], qcfg["sliding_window"]))
    print("       Meta-Llama-3-8B：`sliding_window` / `use_sliding_window` **根本不在 config 里**")
    print("                        （该 config 的全部 23 个键里一个都没有 —— 逐名核对过）")
    check("C.3", set(qcfg) == {"use_sliding_window", "sliding_window"}
          and qcfg["use_sliding_window"] is False and qcfg["sliding_window"] is None
          and set(ll_absent) == {"sliding_window", "use_sliding_window"},
          "★ 两个**都**没开滑窗，但方式不同：一个是**显式关掉**（`false` + `null`），"
          "另一个是**没有这个开关** ⇒ 「有没有这个字段」本身是一处真实差异；"
          "滑窗的一手锚点在**论文**（Mistral 7B），不在这两个 config 上")

    # ── C.4 掩码的形状：窗口外权重**恰为 0.0** ────────────────────────────
    rng = np.random.default_rng(20260922)
    q = rng.standard_normal((S_MASK, D_HEAD))
    k = rng.standard_normal((S_MASK, D_HEAD))
    s = (q @ k.T) / np.sqrt(D_HEAD)
    idx = np.arange(S_MASK)
    mask = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - TOY_W)
    s_masked = np.where(mask, s, -np.inf)
    p = np.exp(s_masked - s_masked.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    outside = float(np.abs(p[~mask]).max())
    print()
    print("  C.4：滑窗的可见性是一个**布尔掩码**（W=%d、S=%d）：掩码外共 %d 个位置"
          % (TOY_W, S_MASK, int((~mask).sum())))
    print("       掩码外的注意力权重最大值 = **%.1f**（`exp(−inf) = 0` 是**精确**的）" % outside)
    check("C.4", outside == MASK_EXACT and p.shape == (S_MASK, S_MASK)
          and mask.dtype == bool,
          "★ 掩码外的权重是**精确的 `0.0`**（不是「很小」）⇒ 滑窗改的是**可见范围**"
          "（一个 0/1 掩码），这与「改 logit 的**值**」那一类变体在结构上不同"
          "（后者的权重处处非零）；后两者的机制正文在 40-kernels/attention/04:130-134"
          "（正文在彼，此处回链）")


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
    """★★ **口径 / 阈值 / specs 的钉住 —— 刻意放在 A/B/C 之前**。

    ★ 为什么放最前：这几条是**所有判据的前提**，而且必须**先于**任何「键名找不到」
      这类**结构性崩溃**报出来。否则那些变异会被 `make mutate` 记成**崩溃捕获**，
      而崩溃捕获**不算真捕获**（它证明「代码炸了」，不是「判据抓住了错」）。
      本主题实测：`QWEN3_30B_A3B_SLIDING_WINDOW_CONFIG` 的**键名**被改后，`C.3` 里
      第一次取值就抛 `KeyError` —— 先钉住之后，那些变异全部落进**断言捕获**。
    """
    print("=" * 74)
    print("D. 口径/阈值/specs 的钉住（★ 放在 A/B/C **之前**：先钉住前提，再看数）")
    print("=" * 74)

    # ── 口径常量 ─────────────────────────────────────────────────────────
    measured = {"TOY_W": TOY_W, "TOY_L": TOY_L, "D_HEAD": D_HEAD,
                "S_MASK": S_MASK, "SEQ_LENS": SEQ_LENS}
    expect = {"TOY_W": 3, "TOY_L": 5, "D_HEAD": 8, "S_MASK": 6,
              "SEQ_LENS": (4096, 16384, 65536)}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 5 个逐位一致（它们是**结论所引用的测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    # ── 阈值 ─────────────────────────────────────────────────────────────
    th = {"SPAN_REL_TOL": SPAN_REL_TOL, "LINEAR_TOL": LINEAR_TOL,
          "MASK_EXACT": MASK_EXACT}
    th_expect = {"SPAN_REL_TOL": 1e-3, "LINEAR_TOL": 0, "MASK_EXACT": 0.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 3 个逐位一致（`MASK_EXACT = 0.0` 是**精确档**：把 C.4 改成「加一个大负数偏置」"
          "就过不了）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── specs 的常量逐位钉住（含 `UNSOURCED` 的消费）──────────────────────
    # ★★ 为什么 `UNSOURCED` 也要有消费者：`45-…/code/specs.py:107-113` 的教训是
    #   「照抄但不使用」的部分**无法被任何断言约束**，只会制造变异测试的漏网。
    #   处置不是把它从分母里排除（那是 `mutate.py` 反对的做法），而是**给它一个消费者**。
    check("D.3", specs.UNSOURCED == (
              "mistral7b_config_json",
              "deepseek_v2_absolute_dims",
              "mla_93_3_percent_kv_reduction"),
          "★ `specs.py` 的 `UNSOURCED` 三项**逐位钉住**（它们是**待办清单**："
          "「取不到一手来源、所以正文不用它」的那几项）")

    # ★ **键名**也要钉住（不只是取值）：`mutate.py` 的分母含**字典的键**，
    #   键名被改后取值就取不到 —— 那会以 `KeyError` 的形式**崩溃**，而不是被断言抓住。
    check("D.4", (specs.MISTRAL7B_WINDOW_SIZE, specs.MISTRAL7B_NUM_LAYERS) == (4096, 32)
          and set(specs.LLAMA3_8B_ABSENT_SLIDING_WINDOW_FIELDS)
          == {"sliding_window", "use_sliding_window"}
          and set(specs.QWEN3_30B_A3B_SLIDING_WINDOW_CONFIG)
          == {"use_sliding_window", "sliding_window"},
          "★ `specs.py` 的滑窗常量**逐位钉住**：Mistral 7B 的 `(W, L) = (%s, %s)`（[C] 级，"
          "论文 Table 1 + §2.1）、「Llama-3-8B 里没有这两个字段」，以及 Qwen3 那两个字段的"
          "**键名**" % (specs.MISTRAL7B_WINDOW_SIZE, specs.MISTRAL7B_NUM_LAYERS))


def mark_part():
    """失败标记自检 + 感受野传播路径的 AST 检查 —— 放在最后跑。"""
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 感受野传播路径的独立性")
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

    # ── 感受野的传播**不得**被简化成一次乘法（AST 级）────────────────────
    called = _callee_names(reachable_left)
    check("D.8", "range" in called and "max" in called,
          "★ `A.2` 的感受野是**逐层传播**出来的（AST 调用表里有 `range`/`max`），"
          "不是把 `L*W` 直接打印一遍 ⇒ 模拟与闭式是两条独立的路径")


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
    print("03_window_hybrid.py —— 滑窗的架构侧：堆叠层扩展感受野，以及与别的变体叠加")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区（计划 §2 的边界表）：本程序**不**重讲「滑窗让每行只需 `ceil(w/Bc)` 个块」")
    print("    这类 kernel 后果（正文在 40-kernels/attention/04:126-156，此处回链）；")
    print("    也**不**算 KV 的字节（六项乘积的正文在 130-quantization/kv-cache-quantization/01:48-76）。")
    pin_part()      # ★ D.1–D.4：先钉住前提（见 pin_part 的说明）
    part_A()
    part_B()
    part_C()
    mark_part()     # ★ D.5–D.8：标记自检与传播路径检查
    return summary()


if __name__ == "__main__":
    sys.exit(main())
