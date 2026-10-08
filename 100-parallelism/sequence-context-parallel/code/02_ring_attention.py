#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_ring_attention.py —— Ring Attention：**跨设备的【块统计量重缩放】**。

配套文档：`02-ring-attention.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★★ 本程序**是纯 numpy 单进程**：所谓「跨设备」被实现成**显式的张量搬运**
>   （把 KV 块按环序**逐个喂给**每一段 query）—— 本域的知识对象是**切分策略**，
>   不是通信原语的实现（通信算法的正文在 `110-communication`）。
> ★★ **随机数**：数据由 `numpy.random.default_rng(991)` 生成，种子 **`991`（不是 2 的幂）**，
>   写在 `_make_inputs()` 的**函数体里**（★ 不是模块级常量 —— 改它对任何断言都没有可观察后果，
>   放进 `mutate.py` 的分母只会制造一条**不过期的漏网**）。

★★★ **术语纪律（本主题最容易写错的一处）**：
  **本程序【不使用】「online softmax」这个词。** 本域写的是**跨设备**的
  **块统计量重缩放**，措辞一律照 Ring Attention 原文：
  · `specs.RING_PERMUTATION_QUOTE`（**置换不变性**）；
  · `specs.RING_RESCALE_QUOTE`（**只要每块的统计量被正确地结合起来做重缩放**）。
  ★ 「online softmax」这个词与它所指的**块内**算法，正文在
  `40-kernels/softmax-normalization/02-online-softmax.md`（那篇逐字自述「本文是 `(m, l)`
  更新公式的正文」）⇒ **本主题回链、一字不写**。★ 本程序的 `C.1` 讲的**不是**那个算法，
  而是「**跨块的统计量没有结合好**」在**放大尺度**下的后果。

★★★ **一条判据性的自查**（计划 §1.3.3）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 `ring_attention()` ⇒ 无从与全量注意力比（不会通过）；
    `C.1` 若删掉那个**朴素变体** ⇒ 无从断言 `nan`。

★★★ **一条必须写明的反例纪律**（本程序 `C.1` 的全部意义）：
  > **反例必须证明它【真的会错】，而不是「它这次没错」。**
  ★★ 实测：**原尺度下朴素版与参照只差 `1 ulp` 档（看起来完全正确）**
    ⇒ 只在该尺度下断言「有差」是**无效**的（那点差与「重缩放对不对」无关）。
    ⇒ 判据**必须找到能触发它的尺度**（`specs.RING_AMPLIFY`），再在那里断言 `nan`。
  ★★★ **且那条判据直接断言 `nan`，不是断言 `inf`** —— 本机实测那个坏结果里
    `inf` 计数是 **`0`**、`nan` 计数是 **满的** ⇒ ★ **「查 `inf`」的调试方法在这里失效**。

★★ **覆盖边界**：本程序的 A 类守的是「**跨块统计量结合得对不对**」，
  **不守**「真实集群上 ring attention 的性能」。★★ **且 C 类不得装「成功行为」**
  （`CONTRIBUTING:238`）：`C.1` 断言的是「**不做重缩放 ⇒ 放大尺度下给 `nan`**」这个**失败**。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **「在线 softmax」的 `(m, l)` 更新公式本身** →
    `40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`；
  · **`S×S` 矩阵的访存账**（FlashAttention 的字节数收益）→
    `40-kernels/attention/03-flash-attention-backward.md` 与 `00-meta/glossary.md:261`
    （★ 本程序只做**数组规模**的对照，**不重算**那些字节账）；
  · **集合通信原语的算法与复杂度**（ring allreduce、reduce-scatter / all-gather 的阶段划分）→
    `110-communication/collectives`（**未建成**）与 `10-foundations/performance-models/02:209`
    （★ 本程序的「环」是**数据搬运的次序**，不是原语实现）；
  · **通信-计算重叠的机制**（双缓冲 / 分块流水）→ `110-communication` 的 `overlap`（**未建成**）
    （★ 本程序只引 `RING_OVERLAP_QUOTE` 的**条件**，不写重叠的机制）；
  · **位置编码的外推** → `80-model-architectures/positional-encoding/02,03`；
  · **注意力变体的结构**（MHA / MQA / GQA 的定义与配比）→
    `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`
    （★ CP 文档里那句「用 MQA/GQA 降 KV 通信量」本程序**只引不展开**）；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02:417`（★ **不是 `:223`** —— 那一行是纯 TP 段）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **带重缩放的跨块注意力 == 全量注意力**（`RING_CP_GRID` 五档，机器精度档）；
  · `A.2`  ★★ **重缩放那一条路径【真的被走到过】**（缓存的最大值在各 KV 块之间**变过**）
             + `cp = 1` 时差**恰为 `0.0`**（退化对照）；
  · `A.1g` ★★★ **守卫**：把「重缩放累积统计量」那一步去掉（`alpha ≡ 1`）⇒ `A.1` **必须失败**；
  · `B.1`  ★★ 通信量 `∝ S`（每台设备只发自己那一块 KV 一次）⇒ 随 `S` **线性**、且**有上界** `2·S·d`；
  · `B.2`  ★ 显存：全量 `S×S` vs 分块 `block×block`（★ **引用，不重推**）—— 做规模对照；
  · `C.1`  ★★★ **朴素版（不做重缩放）在【放大尺度】下给 `nan`**：`nan` 计数满、**`inf` 计数为 `0`**；
              ★★ 且它在**原尺度下只差 `1 ulp` 档**（**看起来完全正确** ⇒ 所以必须找尺度）；
  · `C.1g` ★★★ **守卫**：**同一个放大尺度**下，带重缩放的版本仍然**有限**、且与参照相符
              ⇒ 证明坏掉的是**重缩放**、不是「尺度本身」；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **失败标记自检**。
"""
import ast
import collections
import inspect
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★ 失败标记：本主题取 `{{ FAIL: seq-context-parallel`（★ 只允许出现在 `print()` 的字符串里）。
ASSERT_FAIL_MARK = "{{ FAIL: seq-context-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                     "== FAIL: pipeline-parallel", "|| FAIL: expert-parallel",
                     "?? FAIL: zero-fsdp")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★ 口径常量**从 `specs.py` 读**（这些行不产生可变异叶子：`_walk()` 不下钻 `Attribute`）
S_LEN = specs.ATTN_S
D_HEAD = specs.ATTN_D
CP_GRID = specs.RING_CP_GRID
AMPLIFY = specs.RING_AMPLIFY
TOL_MACHINE = specs.TOL_MACHINE


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
    """
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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立 / 守卫成立」的"
                                "措辞是**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 数据与两个实现
#     ★★★ **参照（全量注意力）不得调用被测（分块注意力）的任何函数**：
#        两者**各自独立**地算 softmax —— 参照在**整行**上取最大值，
#        分块版**永远看不到整行**（它一块一块地更新缓存）。见 `D.8` 的源码级线索。
# ===========================================================================
def _make_inputs(scale=1.0):
    """生成 `(Q, K, V)`；★ 种子 `991`（**不是 2 的幂**）写在函数体里（见模块头）。"""
    rng = np.random.default_rng(991)
    q = rng.standard_normal((S_LEN, D_HEAD)) * scale
    k = rng.standard_normal((S_LEN, D_HEAD)) * scale
    v = rng.standard_normal((S_LEN, D_HEAD))            # ★ `V` 不参与放大（只放大 Q/K 的分数）
    return q, k, v


def full_attention(q, k, v):
    """**参照**：一次算完的注意力（在整行上取最大值后做 softmax）。

    ★★ 它**只看整行**（`axis=1` 的全局最大），**没有**任何「块」的概念
      ⇒ 与分块版是**两条独立的算法路径**。
    """
    scores = q @ k.T / np.sqrt(D_HEAD)
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    return (e @ v) / e.sum(axis=1, keepdims=True)


def _accumulate(scores, vb, acc, l, m, rescale):
    """处理**一个** KV 块，返回 `(acc, l, m, moved)`。

    ★★★ 这就是**跨设备的块统计量重缩放**的落点（`specs.RING_RESCALE_QUOTE`）：
      · `rescale=True`：把已累积的 `(acc, l)` **按新旧最大值之比**缩回来，再并入本块；
      · `rescale=False`（**朴素版**）：直接把 `exp(分数)` 累加 —— 本块**不参与**任何重缩放。

    ★★ `moved` = 这一块是否把缓存的最大值**推高了**（`A.2` 用它证明重缩放**真的被走到过**）。
    ★★ `-inf` 的兜底：整块被 mask 掉时 `scores` 全是 `-inf` ⇒ `mn` 与 `m` 都可能还是 `-inf`
      ⇒ `exp(-inf − (−inf)) = nan` ⇒ 这里用 `np.isfinite` 把两处都兜住
      （★ `03` 的 `C.3` 专门量这个兜底）。
    """
    if not rescale:
        e = np.exp(scores)
        return acc + e @ vb, l + e.sum(axis=1, keepdims=True), m, False
    bmax = scores.max(axis=1, keepdims=True)
    mn = np.maximum(m, bmax)
    with np.errstate(invalid="ignore"):
        alpha = np.exp(np.where(np.isfinite(m) & np.isfinite(mn), m - mn, 0.0))
    safe = np.where(np.isfinite(mn), mn, 0.0)
    e = np.exp(scores - safe)
    return (acc * alpha + e @ vb, l * alpha + e.sum(axis=1, keepdims=True), mn,
            bool(np.any(mn > m)))


def ring_attention(q, k, v, cp, rescale=True, break_rescale=False):
    """分块注意力，KV 块**按环序**逐个流过每一段 query。

    ★★ **返回** `(输出, 缓存最大值被推高的次数)` —— 后者是 `A.2` 的证据。
    ★★ 环序：第 `r` 段 query 与**它自己那块**先配对，再依次收下游/上游的块
      （`(r, r+1, …)` 再绕回 `(…, r−1)`）⇒ 这正是「所有设备围成一个环」的次序。
    ★★★ `break_rescale=True` 是**守卫**用的：它把 `alpha` 强制成 `1`，
      即**不把已累积的统计量缩回来** —— 一个**真的会改变被判量**的错法（`A.1g`）。
    """
    blk = S_LEN // cp
    out = np.zeros_like(q)
    moved_total = 0
    for r in range(cp):
        rows = slice(r * blk, (r + 1) * blk)
        qr = q[rows]
        acc = np.zeros((blk, D_HEAD))
        l = np.zeros((blk, 1))
        m = np.full((blk, 1), -np.inf)
        for c in list(range(r, cp)) + list(range(0, r)):
            cols = slice(c * blk, (c + 1) * blk)
            scores = qr @ k[cols].T / np.sqrt(D_HEAD)
            if break_rescale:
                bmax = scores.max(axis=1, keepdims=True)
                mn = np.maximum(m, bmax)
                with np.errstate(invalid="ignore"):
                    alpha = np.ones_like(mn)          # ★ 守卫：**不缩回**累积量
                safe = np.where(np.isfinite(mn), mn, 0.0)
                e = np.exp(scores - safe)
                acc, l, m = acc * alpha + e @ v[cols], l * alpha + e.sum(axis=1, keepdims=True), mn
                moved_total += int(np.any(mn > m) or False)
            else:
                acc, l, m, moved = _accumulate(scores, v[cols], acc, l, m, rescale)
                moved_total += int(moved)
        out[rows] = acc / l
    return out, moved_total


def _diff(a, b):
    """`max|a − b|`（★ 先把 `nan` 暴露出来：`nan` 存在时 `np.max` 会给 `nan`）。"""
    return float(np.max(np.abs(a - b)))


# ===========================================================================
# 2. A 类：跨块统计量重缩放 == 全量注意力（判据 A.1 / A.1g / A.2）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 算法正确性：**置换不变性 + 块统计量重缩放** == 一次算完")
    print("=" * 78)
    print("  ★★ 逐字依据（`specs.RING_PERMUTATION_QUOTE` / `specs.RING_RESCALE_QUOTE`）：")
    print("     `%s`" % specs.RING_PERMUTATION_QUOTE)
    print("     `%s`" % specs.RING_RESCALE_QUOTE)
    q, k, v = _make_inputs()
    ref = full_attention(q, k, v)

    # ── A.1 ★★★ 五档 CP 度都与全量一致 ─────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **五档 CP 度**下的 `max|分块 − 全量|`（原尺度）──")
    worst = 0.0
    a1_rows = []
    for cp in CP_GRID:
        got, moved = ring_attention(q, k, v, cp)
        d = _diff(got, ref)
        worst = max(worst, d)
        a1_rows.append((cp, d, moved))
        print("     cp=%2d ⇒ `max|diff|` = **%.6e**（缓存最大值被推高 %d 次）" % (cp, d, moved))
    print("     ★★ `cp = 1` 那一档**恰为 `0.0`**：环上只有一台设备 ⇒ 只有一个块 ⇒ 没有可重缩放的东西。")
    check("A.1", worst <= TOL_MACHINE and all(np.isfinite(x) for _, x, _ in a1_rows),
          "★★★ **判据成立**：五档 CP 度的分块结果与全量注意力**逐位相符**"
          "（最大差 **%.6e** ≤ 机器精度档）⇒ ★★ **块统计量结合对了**" % worst,
          "**分块与全量不符** —— ★ 先查 `alpha` 的**方向**（是 `m − mn` 还是 `mn − m`）")

    # ── A.2 ★★ 重缩放那条路径真的被走到过 ────────────────────────────────────
    print()
    print("  ── A.2 ★★ **重缩放【真的被走到过】**（否则 `A.1` 是空的）──")
    n_moved = {cp: m for cp, _, m in a1_rows}
    cp1_exact = dict((cp, d) for cp, d, _ in a1_rows)[1] == 0.0
    print("     缓存最大值被推高的次数：%s"
          % " · ".join("cp=%d: %d" % (cp, n_moved[cp]) for cp in CP_GRID))
    print("     ⇒ ★★ 若某个 `cp > 1` 的档**一次都没推高**，那一档的重缩放代码就没被执行过，")
    print("        `A.1` 在那一档上是**空的** ⇒ 本判据把这件事挡住。")
    check("A.2", all(n_moved[cp] > 0 for cp in CP_GRID if cp > 1) and cp1_exact,
          "★★ **判据成立**：所有 `cp > 1` 的档里缓存最大值都**被推高过**（重缩放执行过），"
          "且 `cp = 1` 那一档的差**恰为 `0.0`**（退化对照）",
          "**某一档一次都没推高缓存最大值** —— `A.1` 在那一档上退化成空断言")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    bad, _ = ring_attention(q, k, v, 4, break_rescale=True)
    d_bad = _diff(bad, ref)
    print()
    print("  ── A.1g ★★★ **守卫**：把「缩回累积统计量」那一步去掉（`alpha ≡ 1`）──")
    print("     改错之后 `max|diff|` = **%.6e**" % d_bad)
    check("A.1g", d_bad > TOL_MACHINE,
          "★★ **守卫成立**：不缩回累积量 ⇒ 与全量的差变成 **%.6e** > 阈 ⇒ `A.1` **能失败** ✓"
          "（★ 这个错法**真的会改变被判量** —— 不是「换个等价写法」）" % d_bad,
          "**守卫失效** —— 不缩回累积量之后 `A.1` 仍然通过 ⇒ `A.1` 是恒真的")
    print("     ★★ **为什么守卫选它、不选「换个块大小」**：块大小只改变**环上有几段**，")
    print("        不改变**每一步要不要重缩放** ⇒ 那样的「守卫」不会让 `A.1` 失败（恒真）。")


# ===========================================================================
# 3. B 类：通信量 ∝ S，以及显存的规模对照（判据 B.1 / B.2）
# ===========================================================================
def _comm_bytes(seq, d_head, cp, elem_bytes):
    """每台设备在**一次**前向里发出的 KV **字节数**：`(cp−1) × 2 × (seq/cp) × d × 每元素字节`。

    ★ `2` 是 `K` 与 `V` 两份；★ `cp−1` 是「除了自己那块之外，环上其余的块各过一次」。
    ★★ 这是**本程序自己模型里的算术**（不是任何来源的实测）⇒ 标为**推导值**。
    """
    return (cp - 1) * 2 * (seq // cp) * d_head * elem_bytes


def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：**通信量 ∝ S**（每台设备只发自己那一块 KV 一次）")
    print("=" * 78)
    print("  ★★ 逐字依据（`specs.RING_SEND_KV_QUOTE`）：")
    print("     `%s`" % specs.RING_SEND_KV_QUOTE)
    print("     ⇒ ★★ **每台设备发一次、收一次** ⇒ 在环上转一圈，**每台设备的通信量 ∝ S**。")
    print("     ★★ **本模块不复核原文的测量口径**（那是原文在真集群上的事）；")
    print("        下面是**本程序自己模型里的算术**，标为**推导值**。")

    # ── B.1 ★★ 通信量 ∝ S，且有上界 2Sd ─────────────────────────────────────
    print()
    print("  ── B.1 ★★ 通信量随 `S` 的**线性**、以及它的**上界** `2·S·d` ──")
    cp = 8
    elem_bytes = 4                       # ★ 单精度：本程序自己模型的取值（推导值，不是规格）
    seqs = (256, 512, 1024, 2048)
    rows = []
    for sq in seqs:
        c = _comm_bytes(sq, D_HEAD, cp, elem_bytes)
        cap = 2 * sq * D_HEAD * elem_bytes
        rows.append((sq, c, cap))
        print("     S=%5d ⇒ 每台设备发出 **%10d** 字节 · 上界 `2·S·d·每元素字节` = %10d · 占比 %.4f"
              % (sq, c, cap, c / cap))
    doubles = [rows[i + 1][1] / rows[i][1] for i in range(len(rows) - 1)]
    cap_ok = all(c <= cap for _, c, cap in rows)
    print("     ★★ `S` 翻倍 ⇒ 通信量变成 %s（**全为 `2.0000`**）"
          % " · ".join("%.4f" % x for x in doubles))
    print("     ★★ 每台设备的占比 `(cp−1)/cp = %.4f`（**与 `S` 无关**）⇒ 通信量是 `O(S)`、"
          "**不是 `O(S²)`** —— 因为传的是 KV（长度 `S`），**不是注意力矩阵**。" % ((cp - 1) / cp))
    check("B.1", all(abs(x - 2.0) < 1e-12 for x in doubles) and cap_ok,
          "★★ **判据成立**：通信量随 `S` **严格线性**（`S` 翻倍 ⇒ `2.0000` 倍），"
          "且**有上界** `2·S·d`（占比恰为 `(cp−1)/cp = %.4f`）⇒ ★ 传的是 KV 不是 `S×S` 矩阵"
          % ((cp - 1) / cp),
          "**通信量不是线性的** —— ★ 检查那两份 `K`/`V`（`2`）与 `cp−1` 次是否漏了")

    # ── B.2 ★ 显存的规模对照（★ 引用，不重推）──────────────────────────────
    print()
    print("  ── B.2 ★ 显存：全量 `S×S` vs 分块 `block×block`（★ **引用，不重推**）──")
    print("     ★★ 逐字依据（`specs.RING_QUADRATIC_QUOTE`）：`%s`" % specs.RING_QUADRATIC_QUOTE)
    print("     ★★ 另一句（`specs.RING_ACCUMULATION_QUOTE`）：`%s`" % specs.RING_ACCUMULATION_QUOTE)
    blk_fixed = 64
    print("     ★ 固定**每块** `block = %d`（即 `cp = S/block` 随 `S` 一起长）⇒" % blk_fixed)
    rows2 = []
    for sq in (256, 512, 1024, 2048, 4096):
        full = sq * sq
        blk = blk_fixed * blk_fixed
        rows2.append((sq, full, blk))
        print("        S=%5d ⇒ 全量分数矩阵 **%9d** 个元素 · 单块的分数矩阵 **%6d** 个元素"
              % (sq, full, blk))
    quad = [rows2[i + 1][1] / rows2[i][1] for i in range(len(rows2) - 1)]
    const = {blk for _, _, blk in rows2}
    print("     ★★ `S` 翻倍 ⇒ 全量变成 %s（**全为 `4.0000`**，`O(S²)`）；"
          % " · ".join("%.4f" % x for x in quad))
    print("        单块的分数矩阵**恒为 %d** 个元素（**与 `S` 无关**）—— "
          "这就是「显存 ∝ 块大小」的意思。" % (blk_fixed * blk_fixed))
    print("     ★★ **本程序不重推那句结论**：它是原文（**同行评审未确认**）的**引用**；")
    print("        这里做的只是**数组规模**的对照（`S²` vs `block²`），不是它的测量复核。")
    check("B.2", all(abs(x - 4.0) < 1e-12 for x in quad) and len(const) == 1,
          "★ **判据成立**：全量分数矩阵随 `S` **二次增长**（`S` 翻倍 ⇒ `4.0000` 倍）、"
          "单块分数矩阵**与 `S` 无关** ⇒ ★ 这是**规模对照**；**结论本身是引用**（原文同行评审未确认）",
          "**规模对照不成立** —— ★ 检查全量的那一项是不是写成了 `S`（而不是 `S²`）")


# ===========================================================================
# 4. C 类：朴素版在**放大尺度**下给 `nan`（判据 C.1 / C.1g）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**不做重缩放的朴素版在放大尺度下给 `nan`**")
    print("=" * 78)
    cp = 4
    q, k, v = _make_inputs()
    ref = full_attention(q, k, v)
    naive, _ = ring_attention(q, k, v, cp, rescale=False)
    d_orig = _diff(naive, ref)
    print("  ★★★ 第一步：**原尺度下的对照** —— 朴素版与参照只差 **%.6e**" % d_orig)
    print("     ⇒ ★★★ **它【看起来完全正确】。** 这正是本判据存在的理由：")
    print("        「原尺度下没差」**不能**用来支持「朴素版对」——")
    print("        ★★ **反例必须证明它【真的会错】，而不是「它这次没错」**。")

    q2, k2, v2 = _make_inputs(AMPLIFY)
    ref2 = full_attention(q2, k2, v2)
    naive2, _ = ring_attention(q2, k2, v2, cp, rescale=False)
    guarded2, _ = ring_attention(q2, k2, v2, cp, rescale=True)
    n_nan = int(np.isnan(naive2).sum())
    n_inf = int(np.isinf(naive2).sum())
    total = int(naive2.size)
    d_guard = _diff(guarded2, ref2)
    print()
    print("  ★★★ 第二步：**把 `Q`/`K` 放大 %g 倍**（`specs.RING_AMPLIFY`）⇒ 分数进入溢出区" % AMPLIFY)
    print("     朴素版：`nan` 计数 = **%d / %d**（满）· **`inf` 计数 = %d**" % (n_nan, total, n_inf))
    print("     ⇒ ★★★ **判据直接断言 `nan`，不是断言 `inf`** ——")
    print("        实测 `inf` 计数是 **`0`** ⇒ **「查 `inf`」的调试方法在这里失效**（`inf/inf` 之后只剩 `nan`）。")
    print("     带重缩放的版本（**同一个放大尺度**）：与参照差 **%.6e**、有限 = %s"
          % (d_guard, bool(np.isfinite(guarded2).all())))
    check("C.1", d_orig <= TOL_MACHINE and n_nan == total and n_inf == 0,
          "★★★ **判据成立**（三个条件）：① **原尺度下朴素版只差 %.3e**（看起来完全正确）；"
          "② **放大 %g 倍后 `nan` 计数 = %d/%d（满）**；③ **`inf` 计数 = 0**"
          " ⇒ ★★ 「它这次没错」被换成了「**在能触发它的尺度下它真的错**」"
          % (d_orig, AMPLIFY, n_nan, total),
          "**朴素版在放大尺度下没有给出满屏 `nan`** —— ★ 那个「反例」于是不构成反例")
    check("C.1g", np.isfinite(guarded2).all() and d_guard <= TOL_MACHINE,
          "★★★ **守卫成立**：**同一个放大尺度**下，带重缩放的版本**有限**且与参照差 **%.3e**"
          " ⇒ ★★ 坏掉的是**重缩放**（跨块统计量没结合好），**不是「尺度本身」**" % d_guard,
          "**带重缩放的版本在放大尺度下也坏了** —— 那说明问题出在尺度，不是重缩放")
    print()
    print("     ★★ **本节的 `C.1` 与 `40-kernels/…/02-online-softmax.md` 的关系**：")
    print("        那是**块内**的 `(m, l)` 更新公式的正文；本节讲的是**跨设备**的块统计量结合。")
    print("        ★★ 本程序**不使用「online softmax」这个词** —— 措辞照原文（置换不变性 + 统计量重缩放）。")


# ===========================================================================
# 5. 前哨：口径 / 阈值 / 标记自检
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】**（否则会污染 mutate 的分类）。
    """
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` + `numpy %s`，macOS arm64，"
          "**无 GPU / 无多卡 / 无 NCCL / 无真实网络**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 验证的是**切分的代数**。")
    print("  ★★ **判据依赖 BLAS**（`q @ k.T` 走 `numpy` 的矩阵乘）⇒ ★ **换机器可能需要重新标定**：")
    print("     `A.1` 的差落在 **1e-16 档**、`A.1g` 的差是 **`O(1)` 档** —— 两者相隔约 15 个数量级，")
    print("     所以「阈取 `1e-12`」在换 BLAS 后仍然安全；★ 但**具体数值**会变。")
    print("  ★★ **本程序不使用「online softmax」这个词**（见模块头与 `D.2`）。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · 「在线 softmax」的 `(m, l)` 更新公式本身 → "
          "`40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`；")
    print("      · `S×S` 矩阵的访存账（FlashAttention 的字节数收益）→ "
          "`40-kernels/attention/03-flash-attention-backward.md`、`00-meta/glossary.md:261`；")
    print("      · 集合通信原语的算法与复杂度 → `110-communication/collectives`（未建成）"
          "与 `10-foundations/performance-models/02:209`；")
    print("      · **通信-计算重叠的机制** → `110-communication` 的 `overlap`（未建成）；")
    print("      · 位置编码的外推 → `80-model-architectures/positional-encoding/02,03`；")
    print("      · 注意力变体的结构（MHA / MQA / GQA 的定义与配比）→ "
          "`80-model-architectures/attention-variants/01-mha-mqa-gqa.md`；")
    print("      · TP 每层 `2+2` 次 all-reduce → `tensor-parallel/03` 与 "
          "`10-foundations/performance-models/02:417`（★ **不是 `:223`**）。")
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
    check("D.2", ASSERT_FAIL_MARK == "{{ FAIL: seq-context-parallel" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `{{ FAIL: seq-context-parallel`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "|| FAIL: expert-parallel",
                                        "?? FAIL: zero-fsdp"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住")
    check("D.3", specs.TOL_MACHINE == 1.0e-12 and specs.ATTN_S == 512
          and specs.ATTN_D == 64 and specs.RING_CP_GRID == (1, 2, 4, 8, 16)
          and specs.RING_AMPLIFY == 100.0,
          "★★ 口径常量逐位钉住：`S = %d`、`d = %d`、CP 网格 = %s、放大倍数 = %g"
          % (specs.ATTN_S, specs.ATTN_D, specs.RING_CP_GRID, specs.RING_AMPLIFY))
    check("D.3b", all(isinstance(x, int) for x in specs.RING_CP_GRID)
          and all(specs.ATTN_S % x == 0 for x in specs.RING_CP_GRID)
          and specs.RING_CP_GRID[0] == 1,
          "★★ CP 网格全为整数、**每一档都整除 `S`**（`A.1` 因此**不需要补齐**）、"
          "且**首档是 `1`**（退化对照）⇒ ★★ **「不整除」的代价单独在 `03` 的 `B.1` 里量**")
    check("D.4", 0 < specs.RING_AMPLIFY < 2 ** 53 and 0 < specs.TOL_MACHINE < 1e-6,
          "★★ **盲区⑥ 的落法**：放大倍数与阈都 `< 2^53` ⇒ `mutate.py` 的 `v + 1.0` "
          "**能**改变它们（否则那类叶子**免检**）")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：specs 常量逐位钉住 · 逐元素钉住 · 引文纪律 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · specs 常量逐位钉住 · **逐元素**钉住 · 引文纪律 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★ Ring Attention 的引文（**载体 + 计数**）──────────────────────
    check("D.5", specs.RING_PERMUTATION_QUOTE
          == "we leverage the permutation invariance property of the inner loop’s key-value block operations."
          and specs.RING_RESCALE_QUOTE
          == ("the self-attention between a query block and a group of key-value blocks can be "
              "computed in any order, as long as the statistics of each block are combined "
              "correctly for rescaling.")
          and specs.RING_ORDER_INVARIANT_QUOTE
          == "the results are invariant to the ordering of these blockwise computations.",
          "★★★ Ring Attention 的**三条**引文逐字钉住 —— ★★ 第 1 条里的撇号是 **U+2019（`’`）**、"
          "**不是 ASCII `'`**；\n"
          "        ★★ 载体 = `https://ar5iv.labs.arxiv.org/html/2310.01889` 正文，"
          "抓取日 **2026-10-08**，逐条在**原始字节**上 `count()` = **1**；\n"
          "        ★★★ **同行评审：本模块不声称已确认**（abs 页上 `ICLR` / `MLSys` / `Accepted` / "
          "`Proceedings` / `Conference` / `NeurIPS` 各 **0** 次）",
          "**Ring Attention 的引文常量被动过**")

    # ── D.6 ★★★ 术语纪律（不用 online softmax）─────────────────────────────
    check("D.6", "online softmax" not in "".join(
        [specs.RING_PERMUTATION_QUOTE, specs.RING_RESCALE_QUOTE, specs.RING_ORDER_INVARIANT_QUOTE])
          and "permutation invariance" in specs.RING_PERMUTATION_QUOTE
          and "rescaling" in specs.RING_RESCALE_QUOTE
          and "online softmax" not in inspect.getsource(ring_attention),
          "★★★ **术语纪律的可执行版**：本主题引的措辞是 **`permutation invariance`** 与 "
          "**`rescaling`**；\n"
          "        ★★ 被测函数 `ring_attention()` 的源码里**不出现 `online softmax`** —— "
          "本域写的是**跨设备**的块统计量重缩放，不是块内的 `(m, l)` 更新\n"
          "        （★ 那一套的正文在 `40-kernels/softmax-normalization/02-online-softmax.md`，此处回链）",
          "**术语被动过** —— 或者 `ring_attention()` 里出现了那个术语")

    # ── D.7 ★★ 通信/显存/零开销三句引文 + 逐元素钉住 ───────────────────────
    check("D.7", specs.RING_SEND_KV_QUOTE
          == ("each device sends a copy of its key-value blocks being used for blockwise "
              "computation to the next device in the ring, while simultaneously receiving "
              "key-value blocks from the previous one.")
          and specs.RING_OVERLAP_QUOTE
          == ("As long as block computations take longer than block transfers, overlapping "
              "these processes results in no added overhead compared to standard transformers.")
          and specs.RING_QUADRATIC_QUOTE
          == "the self-attention has memory cost quadratic in the input sequence length"
          and specs.RING_ACCUMULATION_QUOTE
          == ("the accumulation of key-value blocks leads to increased memory usage, "
              "which defeats the purpose of reducing memory cost."),
          "★★ 通信 / 零开销 / 显存三条引文逐字钉住 —— ★★ 注意 `RING_OVERLAP_QUOTE` 说的是"
          "**有条件的**零开销（`As long as block computations take longer than block transfers`）"
          "⇒ ★ **不得转述成「一定零开销」**",
          "**这三条引文被动过**")
    check("D.7b", specs.RING_CP_GRID == (1, 2, 4, 8, 16)
          and specs.ATTN_S == 512 and specs.ATTN_D == 64 and specs.RING_AMPLIFY == 100.0,
          "★★★ **四个常量【逐元素】钉住**（五档 CP 度 · `S` · `d` · 放大倍数）—— "
          "★ 只钉「整体」会让**元素级**变异漏网",
          "**网格或口径常量的元素被动过**")

    # ── D.8 ★★ 独立性线索（源码级）──────────────────────────────────────────
    src_full = inspect.getsource(full_attention)
    src_ring = inspect.getsource(ring_attention)
    calls_full = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(src_full)))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_ring = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(src_ring)))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    overlap = calls_full & calls_ring & {"full_attention", "ring_attention", "_accumulate"}
    check("D.8", src_full != src_ring and not overlap
          and "ring_attention" not in calls_full and "full_attention" not in calls_ring,
          "★★ **源码级线索**（★ **不是证明**）：参照 `full_attention()` **不调用**被测的 "
          "`ring_attention()`；\n"
          "        ★★ 两条路径的 AST 调用表：参照 = %s · 被测 = %s。\n"
          "        ★★★ **如实标注**：参照那一侧**本身就是【空集】** —— 它只做 `np.*` 的"
          "**属性调用**，`ast.Name` 一处都没有 ⇒ **「调用表交集为空」这一路没有信息量**；\n"
          "        ★★ 独立性**【只】**由 `A.1g`（抽掉重缩放 ⇒ 判据失败）**行为级**兜底 —— "
          "本判据**不**证明「参照是独立的**实现**」"
          % (", ".join(sorted(calls_full)) or "**空集**",
             ", ".join(sorted(calls_ring)) or "**空集**"),
          "**参照调用了被测** —— 那样 `A.1` 就退化成自证")

    # ── D.9 ★★ 随机数的种子与可复现性 ───────────────────────────────────────
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    consts = {t.id for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name)}
    seeds_at_module = [t for t in consts if "SEED" in t or "seed" in t]
    q_a, _, _ = _make_inputs()
    q_b, _, _ = _make_inputs()
    check("D.9", not seeds_at_module and np.array_equal(q_a, q_b)
          and "default_rng(991)" in inspect.getsource(_make_inputs),
          "★★ **种子处置**：种子 `991`（**不是 2 的幂**）写在 `_make_inputs()` 的**函数体里**、"
          "**不是模块级常量**（模块级里 `SEED` 这个名字计 **0** 处）；\n"
          "        ★ 两次调用逐位相同 ⇒ 可复现。★★ 理由：改种子对任何断言都**没有可观察后果**"
          " ⇒ 放进 `mutate.py` 的分母只会制造一条**不过期的漏网**",
          "**种子被提到了模块级** —— 那会让它进变异分母并成为漏网")


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
    print("★ 本程序验证的是「跨块统计量结合得对不对」；**不是 NCCL，也不是真机**。")
    print("★★ 判据依赖 BLAS ⇒ **换机器可能需要重新标定**（阈 `1e-12` 与两个量级都相隔很远，安全）。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_ring_attention.py —— Ring Attention：跨设备的**块统计量重缩放**")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 验证的是**切分的代数**。")
    print("  ★★★ **反例纪律**：原尺度下朴素版**看起来完全正确**（差 `1 ulp` 档）⇒")
    print("      **必须找到能触发它的尺度**（`C.1` 把 `Q`/`K` 放大 %g 倍）再断言 `nan`，" % AMPLIFY)
    print("      且**直接断言 `nan`**（实测 `inf` 计数是 `0` ⇒「查 inf」的方法失效）。")
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
