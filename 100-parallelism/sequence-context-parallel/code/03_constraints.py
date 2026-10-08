#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_constraints.py —— **SP 与 CP 的分工**、`cp` 的约束、以及 **causal mask 跨块**。

配套文档：`03-constraints.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」**。
★★ **随机数**：种子 **`991`（不是 2 的幂）**，写在 `_make_inputs()` 的**函数体里**
  （★ 不是模块级常量 —— 见 `02` 的 `D.9` 同款理由）。

★★★ **本程序的核心：`causal mask` 是**唯一**需要「绝对位置」的地方。**
  第 `r` 段 query 的**绝对**位置 = `r·(S/cp) + arange(S/cp)`；
  第 `c` 块 key 的**绝对**位置 = `c·(S/cp) + arange(S/cp)`；掩码条件是 `kpos <= qpos`。
  ⇒ ★★★ **用【局部位置】**（`arange`）代替绝对位置**不会报错、只会算错** ——
    这就是本主题最强的一条 C 类（`C.1`）。

★★ **`S % cp != 0`（P25）**：序列长度不能被块数整除时，每台设备分到 `ceil(S/cp)` 个 token，
  总计算量从 `S` 涨到 `cp·ceil(S/cp)` ⇒ **补齐代价**（`B.1` 用真的数钉住它）。
  ★★ 补齐之后，**补齐位置的 key 对真实 query 必须全部不可见**（`B.2` 钉住这件事）。

★★★ **`-inf` 的兜底**：整块被 mask 掉时那一块的 `max` 是 `-inf` ⇒
  若缓存的运行最大值**也还是** `-inf`，`exp(-inf − (−inf)) = nan`。
  ⇒ **运行最大值必须兜住**（`C.3` 定量地钉住这件事，并证明**不兜底真的会给 `nan`**）。

★★★ **一条判据性的自查**（计划 §1.3.3）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 `causal_blockwise()` ⇒ 无从与单卡 causal 注意力比（不会通过）；
    `C.1`/`C.3` 若删掉那两个**错法变体** ⇒ 无从断言「真的会错」。

★★ **覆盖边界**：本程序的 A 类守的是「**跨块的 causal 掩码与统计量结合对不对**」，
  **不守**「真实集群上 CP 的性能」。★★ **且 C 类不得装「成功行为」**（`CONTRIBUTING:238`）：
  `C.1`/`C.2`/`C.3` 断言的都是**失败**（不报错而算错 / 给 `nan`），不是「一致」这个成功。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **「在线 softmax」的 `(m, l)` 更新公式本身** →
    `40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`；
    ★★★ **本程序不使用「online softmax」这个词** —— 措辞照 Ring Attention 原文
    （**置换不变性** + **统计量重缩放**）；本程序写的是**跨设备**的块统计量结合。
  · **`S×S` 矩阵的访存账**（FlashAttention 的字节数收益）→
    `40-kernels/attention/03-flash-attention-backward.md` 与 `00-meta/glossary.md:261`；
  · **位置编码的外推**（PI / NTK / ALiBi、混叠周期）→
    `80-model-architectures/positional-encoding/02,03`
    （★ 本程序只谈「**用哪一段的绝对位置**」，**不谈位置编码本身怎么造**）；
  · **集合通信原语的算法与复杂度** → `110-communication/collectives`（**未建成**）与
    `10-foundations/performance-models/02:209`（★ 本程序的「环」只是**次序**）；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02:417`（★ **不是 `:223`** —— 那一行是纯 TP 段）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **带 causal mask 的跨块注意力 == 单卡 causal 注意力**（`CP_GRID` 四档）；
  · `A.1g` ★★★ **守卫**：不做 causal mask ⇒ `A.1` **必须失败**；
  · `A.2`  ★★ 掩码**真的在起作用**（同一份数据，有/无掩码的结果**不同**）；
  · `B.1`  ★★ **`S % cp != 0` 的补齐代价**（`S = 100`、`cp = 8` ⇒ 补齐到 `104`，代价 `4` 个 token 的当量）；
  · `B.2`  ★★ 补齐位置的 key 对真实 query **全部不可见**（`-inf`），且只读前 `S` 行；
  · `C.1`  ★★★ **本主题最强的一条 C 类**：块内 token 用**局部位置** ⇒ **不报错、结果错**；
  · `C.2`  ★★ 阴性对照：**不做 causal mask** ⇒ 也真的会错；
  · `C.3`  ★★★ **整块被 mask 掉 ⇒ 运行最大值是 `-inf`** ⇒ 不兜底给 `nan`，兜住之后正确；
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
CP_GRID = specs.CP_GRID
TOL_MACHINE = specs.TOL_MACHINE


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）。
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
#     ★★★ **参照（单卡 causal 注意力）不得调用被测（跨块）的任何函数** —— 见 `D.8`。
# ===========================================================================
def _make_inputs(seq):
    """生成 `(Q, K, V)`；★ 种子 `991`（**不是 2 的幂**）写在函数体里。"""
    rng = np.random.default_rng(991)
    return (rng.standard_normal((seq, D_HEAD)),
            rng.standard_normal((seq, D_HEAD)),
            rng.standard_normal((seq, D_HEAD)))


def causal_full(q, k, v):
    """**参照**：**单卡**的 causal 注意力（一次算完）。

    ★★ 它用 `arange(seq)` 的**绝对位置**建掩码，**只在整行上**取最大值
      ⇒ 与跨块版是**两条独立的算法路径**。
    """
    seq = q.shape[0]
    pos = np.arange(seq)
    scores = q @ k.T / np.sqrt(D_HEAD)
    scores = np.where(pos[None, :] <= pos[:, None], scores, -np.inf)
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    return (e @ v) / e.sum(axis=1, keepdims=True)


def _step(scores, vb, acc, l, m, guard=True):
    """并入**一个** KV 块的统计量。`guard=False` 是**不兜底 `-inf`** 的错法（`C.3` 用它）。

    ★★★ `guard=True` 的两处兜底（正是 `C.3` 要证明「必须有」的东西）：
      · `alpha`：`m` 与 `mn` 都可能还是 `-inf` ⇒ `exp(-inf − (−inf)) = nan`；
      · `e`：`mn` 是 `-inf` 时 `exp(scores − mn)` 同样是 `nan`
        ⇒ 用 `safe = mn if finite else 0`，于是 `exp(-inf − 0) = 0`（**正确**：那一块全不可见）。
    """
    bmax = scores.max(axis=1, keepdims=True)
    mn = np.maximum(m, bmax)
    if guard:
        with np.errstate(invalid="ignore"):
            alpha = np.exp(np.where(np.isfinite(m) & np.isfinite(mn), m - mn, 0.0))
        safe = np.where(np.isfinite(mn), mn, 0.0)
        e = np.exp(scores - safe)
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            alpha = np.exp(m - mn)
            e = np.exp(scores - mn)
    return acc * alpha + e @ vb, l * alpha + e.sum(axis=1, keepdims=True), mn


def causal_blockwise(q, k, v, cp, order="ring", local_pos=False, use_mask=True, guard=True):
    """跨块的 causal 注意力；`kv` 块按 `order` 流过每一段 query。

    ★★★ **`local_pos=True` 是本主题最强的那条错法**：块内的位置用 `arange(block)`
      （**局部**），而不是 `r·block + arange(block)`（**绝对**）⇒ **不报错、结果错**。
    ★★ `order='reverse'`（`cp−1, …, 0`）是**同样合法**的环向 ⇒ 对第 `0` 段 query 来说，
      前几个块**整块**都被 mask 掉 ⇒ 运行最大值会长时间停在 `-inf`（`C.3` 用它）。
    """
    seq = q.shape[0]
    blk = seq // cp
    out = np.zeros((seq, D_HEAD))
    for r in range(cp):
        rows = slice(r * blk, (r + 1) * blk)
        qr = q[rows]
        qpos = np.arange(r * blk, (r + 1) * blk) if not local_pos else np.arange(blk)
        acc = np.zeros((blk, D_HEAD))
        l = np.zeros((blk, 1))
        m = np.full((blk, 1), -np.inf)
        cols = list(range(cp - 1, -1, -1)) if order == "reverse" else (
            list(range(r, cp)) + list(range(0, r)))
        for c in cols:
            cs = slice(c * blk, (c + 1) * blk)
            kpos = np.arange(c * blk, (c + 1) * blk) if not local_pos else np.arange(blk)
            scores = qr @ k[cs].T / np.sqrt(D_HEAD)
            if use_mask:
                scores = np.where(kpos[None, :] <= qpos[:, None], scores, -np.inf)
            acc, l, m = _step(scores, v[cs], acc, l, m, guard)
        out[rows] = acc / l
    return out


def _diff(a, b):
    return float(np.max(np.abs(a - b)))


# ===========================================================================
# 2. A 类：带 causal mask 的跨块 == 单卡（判据 A.1 / A.1g / A.2）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 算法正确性：**causal mask 跨块** == 单卡 causal 注意力")
    print("=" * 78)
    print("  ★★★ **绝对位置**：第 `r` 段 query 的绝对位置 = `r·(S/cp) + arange(S/cp)`；")
    print("      掩码条件是 `kpos <= qpos` —— ★★ **这是本程序里【唯一】需要绝对位置的地方**。")
    q, k, v = _make_inputs(S_LEN)
    ref = causal_full(q, k, v)

    # ── A.1 ★★★ 四档 CP 度都与单卡 causal 一致 ──────────────────────────────
    print()
    print("  ── A.1 ★★★ **四档 CP 度**下的 `max|跨块 − 单卡|`（带 causal mask）──")
    worst = 0.0
    rows = []
    for cp in CP_GRID:
        got = causal_blockwise(q, k, v, cp)
        d = _diff(got, ref)
        worst = max(worst, d)
        rows.append((cp, d))
        print("     cp=%d ⇒ `max|diff|` = **%.6e**%s"
              % (cp, d, "" if np.isfinite(d) else "   ← ★★★ **不是有限值（`nan` / `inf`）**"))
    print("     ★★ `cp = 1` 那一档**恰为 `0.0`**（只有一块 ⇒ 与单卡是同一条路径）。")
    # ★★★ **护栏**（照 `02` 的 `A.1` 同款）：`max(0.0, nan) == 0.0` ⇒ 见下方 `check` 的说明。
    check("A.1", worst <= TOL_MACHINE and all(np.isfinite(x) for _, x in rows),
          "★★★ **判据成立**：四档 CP 度下带 causal mask 的跨块结果与单卡 causal 注意力"
          "**逐位相符**（最大差 **%.6e** ≤ 机器精度档），★ 且**四档的差【全部有限】**" % worst,
          "**跨块 causal 与单卡不符** —— 含「差里出现 `nan` / `inf`」这一支："
          "`max(0.0, nan) == 0.0` 会把 `nan` **吞掉**（累积量会停在上一次的有限值上）"
          "⇒ 本判据**必须逐档查有限性**。★ 先查**块内的位置**用的是绝对还是局部（`C.1` 复现它）")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    bad = causal_blockwise(q, k, v, 4, use_mask=False)
    d_bad = _diff(bad, ref)
    print()
    print("  ── A.1g ★★★ **守卫**：**不做 causal mask**（把掩码那一步去掉）──")
    print("     改错之后 `max|diff|` = **%.6e**" % d_bad)
    check("A.1g", d_bad > TOL_MACHINE,
          "★★ **守卫成立**：不做掩码 ⇒ 与单卡的差变成 **%.6e** > 阈 ⇒ `A.1` **能失败** ✓"
          "（★ 这个错法**真的会改变被判量**：它让每一行看到未来的 key）" % d_bad,
          "**守卫失效** —— 去掉掩码之后 `A.1` 仍然通过 ⇒ `A.1` 是恒真的")

    # ── A.2 ★★ 掩码真的在起作用（有/无掩码的结果不同）──────────────────────
    print()
    print("  ── A.2 ★★ **掩码真的在起作用**：同一份数据，有掩码 / 无掩码的结果**不同** ──")
    with_mask = causal_blockwise(q, k, v, 4)
    no_mask = causal_blockwise(q, k, v, 4, use_mask=False)
    gap = _diff(with_mask, no_mask)
    print("     有掩码 vs 无掩码 的 `max|diff|` = **%.6e**" % gap)
    print("     ⇒ ★★ 若两者**相同**，说明掩码那一条路径**根本没被执行**（`A.1` 会退化成对空操作判等）。")
    check("A.2", gap > TOL_MACHINE,
          "★★ **判据成立**：掩码真的改变了结果（`max|diff| = %.6e`）⇒ `A.1` 的掩码路径**非空**" % gap,
          "**有掩码与无掩码的结果相同** —— 掩码没有生效")


# ===========================================================================
# 3. B 类：`S % cp != 0` 的补齐代价（判据 B.1 / B.2）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：**`S %% cp != 0` 的补齐代价**（P25）")
    print("=" * 78)
    print("  ★★ 逐字依据（`specs.CP_CHUNK_KV_QUOTE`）：")
    print("     `%s`" % specs.CP_CHUNK_KV_QUOTE)

    # ── B.1 ★★ 补齐代价 ─────────────────────────────────────────────────────
    seq, cp = specs.CP_PAD_S, specs.CP_PAD_DEGREE
    blk = -(-seq // cp)                       # ceil
    padded = blk * cp
    cost = padded - seq
    ratio = cost / seq
    print()
    print("  ── B.1 ★★ `S = %d`、`cp = %d` ⇒ 每台设备分到 `ceil(S/cp) = %d` 个 token ──" % (seq, cp, blk))
    print("     补齐后的总长度 = `cp · ceil(S/cp)` = **%d**（多出 **%d** 个 token 的当量）" % (padded, cost))
    print("     相对代价 = `%d / %d` = **%.4f**（%.2f%%）" % (cost, seq, ratio, 100 * ratio))
    print("     ⇒ ★★ **它是「向上取整」的代数后果**，与实现质量无关；")
    print("        ★ `cp` 越大、`S` 越不被整除，代价越大 —— 但它**恒 `<= (cp−1)/cp` 的一小截**。")
    check("B.1", blk == 13 and padded == 104 and cost == 4 and abs(ratio - 0.04) < 1e-12,
          "★★ **判据成立**：`S = %d`、`cp = %d` ⇒ 每块 **%d** 个 token、补齐后 **%d**、"
          "代价 **%d** 个（**%.2f%%**）" % (seq, cp, blk, padded, cost, 100 * ratio),
          "**补齐代价算错了** —— ★ 检查是不是用了 `S // cp`（下取整）而不是 `ceil`")

    # ── B.2 ★★ 补齐位置的 key 对真实 query 全部不可见 ──────────────────────
    print()
    print("  ── B.2 ★★ **补齐位置的 key 必须对真实 query 全部不可见** ──")
    # ★ 造一个 `padded` 长度的序列：真实数据在前 `seq` 行，其余填零；位置用绝对下标。
    qq, kk, vv = _make_inputs(padded)
    pos = np.arange(padded)
    sc = qq @ kk.T / np.sqrt(D_HEAD)
    sc = np.where(pos[None, :] <= pos[:, None], sc, -np.inf)
    # ★★ 只看**前 `seq` 行 query**（真实 token）对**补齐位置**的 key ——
    #    补齐行自己的 query 当然能看见彼此（它们的绝对位置也 `>= seq`），那不是本题要问的。
    vis = np.isfinite(sc[:seq, seq:])
    ref_real = causal_full(qq[:seq], kk[:seq], vv[:seq])
    # ★ 用同一套掩码、但**只看前 `seq` 行**：跨越补齐边界的注意力
    m = sc.max(axis=1, keepdims=True)
    e = np.exp(np.where(np.isfinite(sc), sc - m, -np.inf))
    got = (e @ vv) / e.sum(axis=1, keepdims=True)
    d_real = _diff(got[:seq], ref_real)
    print("     前 `%d` 行 query 对补齐位置（第 `%d` 行起）的**可见格子数** = **%d**" % (seq, seq, int(vis.sum())))
    print("     前 `%d` 行 query 的注意力输出 与 只在前 `%d` 行上算的结果 的差 = **%.6e**"
          % (seq, seq, d_real))
    print("     ⇒ ★★ **补齐的 key 只是占位**：它们的绝对位置 `>= S` 恒大于任何真实 query 的位置")
    print("        ⇒ 掩码把它们全部挡住 ⇒ 真实行的结果**与不补齐时一致**。")
    check("B.2", int(vis.sum()) == 0 and d_real <= TOL_MACHINE,
          "★★ **判据成立**：① 前 `%d` 行 query 对补齐位置**可见格子数恰为 `0`**"
          "（掩码把它们全挡住了）；② 真实行的结果与「不补齐时」一致（差 **%.3e**）"
          "⇒ ★ 补齐**只花算力、不污染结果**" % (seq, d_real),
          "**补齐位置的 key 被真实 query 看到了** —— 那会污染结果（不是「只花算力」）")
    print("     ★★ **一处必须写明的边界**：补齐行的 query（第 `%d` 行起）**结果是垃圾** ——" % seq)
    print("        本程序只读前 `%d` 行；★ 端到端实现里要靠「每台设备处理 `ceil(S/cp)` 个 token」"
          % seq)
    print("        这一条**按长度剪掉**，否则会把垃圾写回。")


# ===========================================================================
# 4. C 类：三条真的会失败的反例（判据 C.1 / C.2 / C.3）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**局部位置** · **无掩码** · **`-inf` 的运行最大值**")
    print("=" * 78)
    q, k, v = _make_inputs(S_LEN)
    ref = causal_full(q, k, v)

    # ── C.1 ★★★ 局部位置（本主题最强的一条）────────────────────────────────
    print()
    print("  ── C.1 ★★★ **本主题最强的一条 C 类**：块内的 token 用【局部位置】──")
    print("     即 `qpos = arange(block)`（而不是 `r·block + arange(block)`）")
    print("     ⇒ ★★★ **不报错（形状、类型、算术全对），只是掩码跨块时错位。**")
    for cp in (2, 4):
        got = causal_blockwise(q, k, v, cp, local_pos=True)
        d = _diff(got, ref)
        print("     cp=%d ⇒ `max|diff|` = **%.6e**（★ 没有任何异常抛出）" % (cp, d))
    d2 = _diff(causal_blockwise(q, k, v, 2, local_pos=True), ref)
    d4 = _diff(causal_blockwise(q, k, v, 4, local_pos=True), ref)
    check("C.1", d2 > 1e-3 and d4 > 1e-3 and np.isfinite(d2) and np.isfinite(d4),
          "★★★ **判据成立**：用局部位置 ⇒ `cp=2` 差 **%.6e**、`cp=4` 差 **%.6e**，"
          "★ 且**没有异常**（`C.2` 是另一条独立的阴性对照）" % (d2, d4),
          "**用局部位置没有让结果变错** —— 那条掩码路径可能没被执行")

    # ── C.2 ★★ 阴性对照：不做 causal mask ───────────────────────────────────
    print()
    print("  ── C.2 ★★ 阴性对照：**不做 causal mask** ──")
    d_nomask = _diff(causal_blockwise(q, k, v, 4, use_mask=False), ref)
    print("     `max|diff|` = **%.6e** ⇒ ★ 它也**真的会错**（与 `C.1` 是两条不同的错法）" % d_nomask)
    print("     ⇒ ★★ 逐字依据（`specs.CP_CAUSAL_QUOTE`）：`%s`" % specs.CP_CAUSAL_QUOTE)
    check("C.2", d_nomask > 1e-3,
          "★★ **判据成立**：不做 causal mask ⇒ 差 **%.6e**（真的会错）"
          "⇒ ★ 与 `C.1` 一起，**两条不同的错法各自被单独钉住**" % d_nomask,
          "**不做掩码也没有变错** —— 那 `A.1` 的掩码分支是空的")

    # ── C.3 ★★★ 整块被 mask 掉 ⇒ 运行最大值是 -inf ──────────────────────────
    print()
    print("  ── C.3 ★★★ **整块被 mask 掉** ⇒ 运行最大值是 `-inf` ⇒ **必须兜住** ──")
    print("     ★★ 场景：`order='reverse'`（`cp−1, …, 0`，**同样合法的环向**）下，")
    print("        第 `0` 段 query 最先收到的几个 KV 块**整块都在它右边** ⇒ 全被 mask 掉。")
    cp = 4
    blk = S_LEN // cp
    good = causal_blockwise(q, k, v, cp, order="reverse", guard=True)
    bad3 = causal_blockwise(q, k, v, cp, order="reverse", guard=False)
    d_good = _diff(good, ref)
    n_nan = int(np.isnan(bad3).sum())
    n_inf = int(np.isinf(bad3).sum())
    print("     **兜住**（`safe = mn if finite else 0`）：与单卡的差 **%.6e**、有限 = %s"
          % (d_good, bool(np.isfinite(good).all())))
    print("     **不兜底**：`nan` 计数 = **%d / %d** · `inf` 计数 = **%d**"
          % (n_nan, bad3.size, n_inf))
    print("     ⇒ ★★★ **`exp(-inf − (−inf)) = nan`** —— 这就是为什么运行最大值**必须**兜底；")
    print("        ★★ 而且（与 `02` 的 `C.1` 同款）**判据要断言 `nan` 而不是 `inf`**。")
    print("     ⇒ ★★ **本机实测**：第 `%d` 段 query（位置 `0..%d`）在 reverse 环向下，"
          "前 `%d` 个 KV 块整块不可见。" % (0, blk - 1, cp - 1))
    check("C.3", d_good <= TOL_MACHINE and n_nan > 0 and n_inf == 0
          and bool(np.isfinite(good).all()),
          "★★★ **判据成立**（四个条件）：① 兜住之后与单卡一致（差 **%.3e**）；② **不兜底给 `nan`**"
          "（计数 **%d**）；③ `inf` 计数为 **%d**；④ 兜住的版本**全部有限**" % (d_good, n_nan, n_inf),
          "**不兜底也没有给 `nan`** —— 那个 `-inf` 场景没有被触发，这条判据是空的")


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
    print("  ★★ **判据依赖 BLAS**（`q @ k.T` 走 `numpy` 的矩阵乘）⇒ ★ **换机器可能需要重新标定**")
    print("     （`A.1` 的差在 1e-16 档、三条 C 类的差在 1e-1~1e0 档，相隔十几个数量级 ⇒ 阈安全）。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · 「在线 softmax」的 `(m, l)` 更新公式本身 → "
          "`40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`；")
    print("        ★★★ **本程序不使用「online softmax」这个词** —— 措辞照 Ring Attention 原文"
          "（置换不变性 + 统计量重缩放）。")
    print("      · `S×S` 矩阵的访存账（FlashAttention 的字节数收益）→ "
          "`40-kernels/attention/03-flash-attention-backward.md`、`00-meta/glossary.md:261`；")
    print("      · 位置编码的外推（PI / NTK / ALiBi、混叠周期）→ "
          "`80-model-architectures/positional-encoding/02,03`"
          "（★ 本程序只谈「**用哪一段的绝对位置**」，不谈编码本身怎么造）；")
    print("      · 集合通信原语的算法与复杂度 → `110-communication/collectives`（未建成）"
          "与 `10-foundations/performance-models/02:209`；")
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
    check("D.3", specs.CP_GRID == (1, 2, 4, 8) and specs.ATTN_S == 512 and specs.ATTN_D == 64
          and specs.CP_PAD_S == 100 and specs.CP_PAD_DEGREE == 8 and specs.TOL_MACHINE == 1.0e-12,
          "★★ 口径常量逐位钉住：CP 网格 = %s、`S = %d`、`d = %d`、补齐组 = `(%d, %d)`"
          % (specs.CP_GRID, specs.ATTN_S, specs.ATTN_D, specs.CP_PAD_S, specs.CP_PAD_DEGREE))
    check("D.3b", all(isinstance(x, int) for x in specs.CP_GRID)
          and all(specs.ATTN_S % x == 0 for x in specs.CP_GRID)
          and specs.CP_GRID[0] == 1
          and specs.CP_PAD_S % specs.CP_PAD_DEGREE != 0,
          "★★ CP 网格全为整数、**每一档都整除 `S`**（`A.1` 不需要补齐）、**首档是 `1`**；"
          "★ 且**补齐那一组【故意不整除】**（`%d %% %d = %d != 0`）⇒ `B.1` 才有东西可量"
          % (specs.CP_PAD_S, specs.CP_PAD_DEGREE, specs.CP_PAD_S % specs.CP_PAD_DEGREE))
    check("D.4", 0 < specs.CP_PAD_S < 2 ** 53 and specs.CP_PAD_DEGREE > 0,
          "★★ **盲区⑥ 的落法**：补齐组两个常量都 `< 2^53` ⇒ `mutate.py` 的 `v + 1.0` **能**改变它们")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：specs 常量逐位钉住 · 逐元素钉住 · 引文纪律 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · specs 常量逐位钉住 · **逐元素**钉住 · 引文纪律 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★★ SP 与 CP 的分工（本主题的题眼）─────────────────────────────
    check("D.5", specs.CP_VS_SP_QUOTE
          == ("Unlike prior SP (sequence parallelism), which only splits the sequence of Dropout "
              "and LayerNorm activations, CP partitions the network inputs and all activations "
              "along the sequence dimension.")
          and specs.CP_DEFINITION_QUOTE
          == "Context Parallelism (CP) is a parallelization scheme on the sequence-length dimension.",
          "★★★ **本主题的题眼**（SP 与 CP 的分工）逐字钉住：**SP 只切 Dropout 与 LayerNorm 的激活**"
          "（= `01` 里那 `10sbh` 那一块），**CP 连网络输入与【全部】激活一起按序列维切**。\n"
          "        ★★ 载体 = GitHub `raw.githubusercontent.com/NVIDIA/Megatron-LM/main/"
          "docs/user-guide/features/context_parallel.md`，**抓取日 2026-10-08**，"
          "原始字节各 **1** 次；\n"
          "        ★★★ 来源级别 = **[B]（某实现的官方文档）**，**不是 [A]（规范）** —— 它是**活文档**"
          "（无版本号）⇒ 引用写明抓取日、**不做行号锚点**",
          "**SP/CP 分工那两句被动过** —— 或者载体/级别写错了")
    print("     ⇒ ★★ **它把 `01` 的结论与 `02` 的机制接起来**：")
    print("        `01` 说 SP 的收益是 `10sbh`（`O(s)`）；`02` 说 CP 的机制是「块统计量重缩放」；")
    print("        ⇒ ★★★ **序列越长，SP 越不够用、CP 越必需**（`01` 的 `B.2` 给出引子，本节收口）。")

    # ── D.6 ★★ `cp` 的约束（四条）─────────────────────────────────────────
    check("D.6", specs.CP_QKV_QUOTE
          == ("For attention, the Q (query) of each token must combine with the KV (key and value) "
              "of all tokens in the same sequence.")
          and specs.CP_ALLGATHER_QUOTE
          == "CP therefore requires an additional all-gather across GPUs to collect the full sequence of KV."
          and specs.CP_CHUNK_KV_QUOTE
          == ("To reduce activation memory footprint, each GPU stores only the KV of a sequence "
              "chunk in forward and gathers KV again in backward.")
          and specs.CP_NO_CHANGE_QUOTE
          == ("With CP, all modules except attention (for example, Linear and LayerNorm) can "
              "work as usual without any changes, because they do not have inter-token "
              "operations."),
          "★★ **`cp` 的四条约束**逐字钉住：\n"
          "        ① 注意力要看到**全序列的 KV**（`CP_QKV_QUOTE`）⇒ ② 必须有一次额外的 all-gather"
          "（`CP_ALLGATHER_QUOTE`）；\n"
          "        ③ 前向**只存自己那一段的 KV**、反向**再 gather 一次**（`CP_CHUNK_KV_QUOTE`）"
          "⇒ ★ 用**通信**换**显存**；\n"
          "        ④ **只有 attention 要改**，其余模块照常（`CP_NO_CHANGE_QUOTE`）\n"
          "        ★★ 载体同 `D.5`，抓取日 2026-10-08，原始字节各 **1** 次",
          "**`cp` 的约束引文被动过**")

    # ── D.7 ★★★ 环 + causal 张力 + 每卡显存 ───────────────────────────────
    check("D.7", specs.CP_RING_P2P_QUOTE
          == ("The all-gather and reduce-scatter are implemented as point-to-point communications "
              "in a ring topology.")
          and specs.CP_RING_HIGH_PERF_QUOTE == "but targets higher performance by"
          and specs.CP_CAUSAL_QUOTE
          == ("avoiding extra work from lower-triangle causal masking while keeping load "
              "balanced across GPUs")
          and specs.CP_CP_DEGREE_QUOTE
          == ("With CP, each GPU computes on part of the sequence, which scales down both "
              "compute and communication by the CP degree.")
          and specs.CP_MEM_PER_GPU_QUOTE
          == ("The activation memory footprint per GPU is also smaller by the CP degree, "
              "which reduces OOM risk."),
          "★★ 五条引文逐字钉住 —— 其中 **`CP_RING_HIGH_PERF_QUOTE` 是实现文档【自己承认】**"
          "「CP 与 Ring Attention 是同一类机制」（原句那里有一个 markdown 链接 ⇒ **只取链接之后**）；\n"
          "        ★★★ **`CP_CAUSAL_QUOTE` 是 `C.1` 的锚**：**causal mask 与负载均衡是有张力的**"
          "（下三角之外的块算白算），实现**特意避开**这份额外工作\n"
          "        ⇒ ★★ 本程序 `C.1` 量的是这份张力**写错时的样子**（局部位置）。\n"
          "        ★★ 载体同 `D.5`（活文档，抓取日 2026-10-08，原始字节各 **1** 次）",
          "**这五条引文被动过**")
    check("D.7b", specs.CP_GRID == (1, 2, 4, 8) and specs.CP_PAD_S == 100
          and specs.CP_PAD_DEGREE == 8 and specs.ATTN_S == 512 and specs.ATTN_D == 64,
          "★★★ **五个常量【逐元素】钉住**（四档 CP 度 · 补齐组两个 · `S` · `d`）—— "
          "★ 只钉「整体」会让**元素级**变异漏网",
          "**网格或口径常量的元素被动过**")

    # ── D.8 ★★ 独立性线索（源码级）──────────────────────────────────────────
    src_full = inspect.getsource(causal_full)
    src_blk = inspect.getsource(causal_blockwise)
    calls_full = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(src_full)))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_blk = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(src_blk)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("D.8", src_full != src_blk
          and "causal_blockwise" not in calls_full and "causal_full" not in calls_blk
          and not ({"causal_blockwise", "causal_full", "_step"} & calls_full),
          "★★ **源码级线索**（★ **不是证明**）：参照 `causal_full()` **不调用**被测的 "
          "`causal_blockwise()`；\n"
          "        ★★ 两条路径的 AST 调用表：参照 = %s · 被测 = %s。\n"
          "        ★★★ **如实标注**：参照那一侧**本身就是【空集】** —— `causal_full()` 只做 "
          "`np.*` 的**属性调用**，`ast.Name` 一处都没有 ⇒ **「调用表交集为空」这一路没有信息量**；\n"
          "        ★★ 独立性**【只】**由 `A.1g`（去掉掩码 ⇒ 判据失败）**行为级**兜底 —— "
          "本判据**不**证明「参照是独立的**实现**」"
          % (", ".join(sorted(calls_full)) or "**空集**",
             ", ".join(sorted(calls_blk)) or "**空集**"),
          "**参照调用了被测** —— 那样 `A.1` 就退化成自证")

    # ── D.9 ★★ 术语纪律 + 其它主题（本域）的边界对齐 ────────────────────────
    check("D.9", "online softmax" not in inspect.getsource(causal_blockwise)
          and "online softmax" not in "".join(
              [specs.CP_CAUSAL_QUOTE, specs.CP_VS_SP_QUOTE, specs.CP_MEM_PER_GPU_QUOTE])
          and "permutation invariance" in specs.RING_PERMUTATION_QUOTE,
          "★★★ **术语纪律的可执行版**：被测函数与被引的 CP 引文里**都不出现 `online softmax`**；"
          "本主题的措辞照 Ring Attention 原文（**`permutation invariance`** + **统计量重缩放**）。\n"
          "        ★ 块内那套 `(m, l)` 更新的正文在 "
          "`40-kernels/softmax-normalization/02-online-softmax.md`，此处回链",
          "**术语被动过** —— 或者引文里混进了不该有的词")

    # ── D.10 ★★ 种子处置 ───────────────────────────────────────────────────
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    consts = {t.id for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name)}
    seeds_at_module = [t for t in consts if "SEED" in t.upper()]
    a1, _, _ = _make_inputs(S_LEN)
    a2, _, _ = _make_inputs(S_LEN)
    check("D.10", not seeds_at_module and np.array_equal(a1, a2)
          and "default_rng(991)" in inspect.getsource(_make_inputs),
          "★★ **种子处置**：种子 `991`（**不是 2 的幂**）写在 `_make_inputs()` 的**函数体里**"
          "（模块级里 `SEED` 这个名字计 **0** 处）；★ 两次调用逐位相同 ⇒ 可复现",
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
    print("★ 本程序验证的是「跨块的 causal 掩码与统计量结合对不对」；**不是 NCCL，也不是真机**。")
    print("★★ 判据依赖 BLAS ⇒ **换机器可能需要重新标定**。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_constraints.py —— SP 与 CP 的分工 · `cp` 的约束 · **causal mask 跨块**")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 验证的是**切分的代数**。")
    print("  ★★★ 本程序最强的一条 C 类是 `C.1`：**块内 token 用局部位置** ⇒ "
          "**不报错、只是错**。")
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
