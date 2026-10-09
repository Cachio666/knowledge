#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_sampling_and_search.py —— **采样与搜索**：greedy / beam / temperature / top-k / **top-p**。

配套文档：`01-sampling-and-search.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本程序最强的一条 A 类：搜索空间【可数】⇒ beam 能与【穷举】对拍**。
  · **被测** = `beam`：**增量式**束搜索（每步展开 `B` 条束、按累计对数概率剪枝回 `B`）；
  · **参照** = `exhaustive_best`：**把 `V^L` 条序列逐条枚举**（`itertools.product`），
    不剪枝、不增量、无共享代码（`D.8` 在源码层核）；
  · **判据**：★ `B = V^L`（**不剪枝**）时，beam 的最优序列与穷举**逐位相同**；
    ★★ 而 `B = 1`（= 贪心）时**未必** —— 这正是「beam 是近似搜索」的落点（`A.3`）。
  ★★ **两条守卫**（`D.4`/`D.5`）：把 beam 改成「只展开最优子节点」⇒ 对不上；
    把穷举改成「返回最差序列」⇒ 对不上。**两端都过 `CONTRIBUTING:235`**。

★★ **采样三兄弟的语义差**（`B.*` / `C.*`）：
  · **greedy** = 每步取 argmax（**确定性**，候选集 = `1`）；
  · **beam(B)** = 维护 `B` 条候选序列（**搜索**，代价 = `B`）；
  · **temperature** = 对 logits 除以 `T` 再 softmax ⇒ ★★ **只改形状，不改候选集大小**（恒为 `V`）；
  · **top-k** = 保留最高的 `k` 个 ⇒ 候选集**定值** `k`；
  · **top-p**（nucleus）= 保留累积概率刚过 `p` 的**最小**集合 ⇒ 候选集**动态**（随分布变）。

★★ **覆盖边界**：本程序守的是「**解码算法的代数与不变量**」（搜索空间 / 候选集 / 归一化）。
  **不守**任何**真实模型**的采样行为（本机**无 GPU、无任何目标 LLM**）。
  ★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **`B = V^L` 时 beam 与穷举逐位一致**（搜索空间可数）；
  · `A.2` ★★ **贪心 == `beam(1)`**（两条独立实现会合）；
  · `A.3` ★★★ **`beam` 不精确**：贪心在 **8/12** 个模型上错过穷举最优、`beam(2)` 在 **2/12** 上仍错过；
  · `B.1` ★★ **beam 的代价 = 活束数 `B`**（每层恰好保留 `B` 条）；
  · `B.2` ★★ **top-k 候选集【定值】vs top-p 【动态】**（同一阈值下大小随分布变）；
  · `B.3` ★★ **temperature 不改候选集大小**（恒为 `V`），只改熵（单调）；
  · `C.1`/`C.2` ★★ **两端退化守卫**：`p → 0`、`k = 1` ⇒ 退化成贪心；`p = 1`、`k ≥ V` ⇒ 全集；
  · `D.4`/`D.5` ★★★ **守卫实验**：改坏 beam / 改坏穷举 ⇒ 断言必须失败。
"""
import ast
import collections
import hashlib
import inspect
import itertools
import math
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（decoding-strategies）取 `-- FAIL: decoding-strategies`；
#   `mutate.py` 用 AST 读出（不写死）。
ASSERT_FAIL_MARK = "-- FAIL: decoding-strategies"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

# ★★ **三条术语纪律**（`§2.2`）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
#   · 本域一律写「**自回归解码**」全称（不裸写 `decode`，那是 `60-compilers` 的 decode pass）；
#   · 本域一律写「**推理批 / 在飞请求数**」全称（不裸写 `batch`，那是训练的 mini-batch）；
#   · 本域一律写「**请求调度器**」全称（不裸写 `scheduler`，那是 GPU 的 warp scheduler）；
#   · 本域最强的一句**限定语**必须逐字出现。
TERM_AUTOREGRESSIVE = "自回归解码"
TERM_INFLIGHT = "推理批 / 在飞请求数"
TERM_SCHEDULER = "请求调度器"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

SCORE_EPS = 1e-9                    # ★ 两条路径都在「累计对数概率」上，容差取小值

# ★★★ **本机实测值**（本程序在本机跑出的数）—— 正文里的每个数都在这里**钉死**。
#   搜索侧（12 个哈希模型，`V = 4`、`L = 4`、穷举空间 `4^4 = 256`）：
PINNED_N_MODELS = 12                 # ★ 模型个数
PINNED_GREEDY_MISS = 8               # ★ 贪心错过穷举最优的模型个数
PINNED_BEAM2_MISS = 2                # ★ `beam(2)` 仍错过穷举最优的模型个数
PINNED_M1_GREEDY = (2, 2, 0, 0)      # ★ 模型 `m1` 上贪心解出的序列
PINNED_M1_EXHAUSTIVE = (2, 1, 3, 3)  # ★ 模型 `m1` 上穷举的最优序列（★ 与上一行**不同**）
PINNED_M1_BEAM2 = (2, 1, 3, 3)       # ★ 模型 `m1` 上 `beam(2)` 的解（★ **找回了最优**）
#   采样侧（两个**钉死的**分布 + 一组固定 logits）：
SAMPLE_PEAKED = (0.90, 0.05, 0.03, 0.015, 0.005)   # ★ 尖峰（★ 逐元素写，不用 `*` —— `_walk` 不下钻 BinOp）
SAMPLE_FLAT = (0.2, 0.2, 0.2, 0.2, 0.2)            # ★ 均匀
SAMPLE_LOGITS = (0.3, 0.9, 0.1, 0.5)               # ★ 温度-熵演示用的固定 logits
PINNED_TOPP_PEAKED_09 = 1            # ★ 尖峰上 `top-p = 0.9` 的候选集大小
PINNED_TOPP_PEAKED_095 = 2           # ★ 尖峰上 `top-p = 0.95` 的候选集大小（★ **同一个分布上**更大）
PINNED_TOPP_FLAT_09 = 5              # ★ 均匀上 `top-p = 0.9` 的候选集大小（★ 全集）
PINNED_TOPP_FLAT_05 = 3              # ★ 均匀上 `top-p = 0.5` 的候选集大小
PINNED_ENTROPY = (1.203755, 1.340090, 1.374964)   # ★ `SAMPLE_LOGITS` 在 `T ∈ (0.5,1.0,2.0)` 上的熵


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D）。★ 失败路径**不复述结论**。"""
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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立」的措辞是"
                                "**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 玩具模型：**确定性哈希 logits**（★ 无 RNG 状态 ⇒ 结论可复现）
# ===========================================================================
def model_logits(salt, prefix):
    """★★ 玩具「模型」：对 `(salt, prefix)` 取 `md5`，把前 `V` 个字节映到 logits。

    ★★ **它不是一个真实的语言模型** —— 它只是一台**确定性的、可复现的**打分器，
      用来让「贪心会不会掉进陷阱」这件事可以被**逐位复核**。
    """
    h = hashlib.md5(("%s,%s" % (salt, ",".join(map(str, prefix)))).encode()).digest()
    return np.array([h[i] / 255.0 for i in range(specs.SEARCH_VOCAB)])


def probs(logits):
    """数值稳定的 softmax。"""
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_temp(logits, T):
    """**temperature**：`softmax(logits / T)`。★ `T` **不改候选集大小**（恒为整个词表）。"""
    return probs(logits / T)


def entropy(p):
    """香农熵（nats）。★ 用来量化「temperature 只改形状」。"""
    q = np.clip(p, 1e-300, None)
    return float(-(q * np.log(q)).sum())


# ===========================================================================
# 2. 搜索：被测 = **增量式 beam**；参照 = **穷举**
# ===========================================================================
def beam(salt, L, B, corrupt=None, trace=False):
    """★★★ **被测实现**：**增量式**束搜索。

    ★ 每步把当前 `B` 条束各展开 `V` 个子节点，按**累计对数概率**排序，剪枝回 `B` 条。
    ★ 与 `exhaustive_best` **无共享代码**（一个增量剪枝、一个全枚举；`D.8` 在源码层核）。
    ★★ `corrupt`（**只用于守卫实验**）：
      · `"greedy_child"` —— 每步只展开**当前最优那一条**束的**最优子节点**（退化成贪心）；
    ★★ `trace=True` 时额外返回**每层保留的束数**（`B.1` 直接断言它）。
    """
    beams = [((), 0.0)]
    widths = []
    for _ in range(L):
        if corrupt == "greedy_child":
            seq, s = beams[0]
            p = probs(model_logits(salt, seq))
            t = int(np.argmax(p))
            beams = [(seq + (t,), s + math.log(p[t]))]
        else:
            cand = []
            for seq, s in beams:
                p = probs(model_logits(salt, seq))
                for t in range(specs.SEARCH_VOCAB):
                    cand.append((seq + (t,), s + math.log(p[t])))
            cand.sort(key=lambda x: -x[1])
            beams = cand[:B]
        widths.append(len(beams))
    if trace:
        return beams, widths
    return beams


def beam_best(salt, L, B, corrupt=None):
    """取 beam 的最优序列（`beam` 返回 `[(序列, 分数), ...]`，故取首项的序列段）。"""
    return beam(salt, L, B, corrupt=corrupt)[0][0]


def greedy(salt, L):
    """**贪心**：逐步取 argmax。★ 与 `beam` 是**两条独立实现**（这里没有「束」的概念）。"""
    seq = ()
    for _ in range(L):
        p = probs(model_logits(salt, seq))
        t = int(np.argmax(p))
        seq = seq + (t,)
    return seq


def exhaustive_best(salt, L, corrupt=None):
    """★★★ **参照实现**：**把 `V^L` 条序列逐条枚举**，比较累计对数概率，取最优。

    ★★ 它**不含剪枝、不含增量状态** —— 这正是它与 `beam` 的**路径独立**之处。
    ★★ `corrupt="worst"`（**只用于守卫实验**）：返回**最差**的序列。
    """
    V = specs.SEARCH_VOCAB
    best = None
    best_s = -1e18
    worst = None
    worst_s = 1e18
    for seq in itertools.product(range(V), repeat=L):
        s = 0.0
        ok = True
        for d, t in enumerate(seq):
            p = probs(model_logits(salt, seq[:d]))
            if p[t] <= 0.0:
                ok = False
                break
            s += math.log(p[t])
        if not ok:
            continue
        if s > best_s:
            best_s, best = s, seq
        if s < worst_s:
            worst_s, worst = s, seq
    return worst if corrupt == "worst" else best


# ===========================================================================
# 3. 采样变换：top-k（**定值**）/ top-p（**动态**）/ temperature
# ===========================================================================
def top_k_set(p, k):
    """**top-k**：保留概率最高的 `k` 个（★ 大小 = `min(k, V)`，**与分布无关**）。"""
    order = np.argsort(-np.asarray(p, dtype=float))
    return [int(i) for i in order[:min(k, len(p))]]


def top_p_set(p, thr):
    """**top-p（nucleus）**：保留**累积概率刚过 `thr` 的最小集合**（★ 大小**随分布变**）。

    ★ 机制：按概率降序累加，遇到第一个使累计 `≥ thr` 的位置就截止（**包含它**）。
    ★★ 不变式：被保留集合的**累计概率 ∈ `[thr, thr + p_max)`**（`B.2` 断言它）。
    """
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, thr) + 1)
    return [int(i) for i in order[:k]]


# ===========================================================================
# 4. A 类 —— ★★★ **搜索空间可数：beam 与穷举对拍**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **搜索空间可数**：被测 = 增量式 beam；参照 = 全枚举穷举（两条独立路径）")
    print("=" * 78)
    V, L = specs.SEARCH_VOCAB, specs.SEARCH_LEN
    seeds = specs.SEARCH_MODEL_SEEDS
    full = V ** L
    print("  ★★ 玩具词表 `V = %d`、长度 `L = %d` ⇒ **穷举空间** `V^L = %d` 条序列。" % (V, L, full))
    print("  ★★ 两条路径：`beam` 是**增量 + 剪枝**，`exhaustive_best` 是**全枚举**。")

    # ── A.1 ★★★ B = V^L 时不剪枝 ⇒ 与穷举逐位一致 ────────────────────────
    print()
    print("  ── A.1 ★★★ **`B = V^L = %d`（不剪枝）时，beam 与穷举逐位一致** ──" % full)
    print("     %-6s %-18s %-18s %s" % ("模型", "beam(V^L)", "穷举最优", "一致"))
    ok_all = True
    for s in seeds:
        b = beam_best(s, L, full)
        e = exhaustive_best(s, L)
        ok = (b == e)
        ok_all = ok_all and ok
        print("     %-6s %-18s %-18s %s" % (s, b, e, "✓" if ok else "✗"))
    check("A.1", ok_all,
          "★★★ **判据成立**：**%d 个模型**上，`beam(V^L)` 与**全枚举穷举**给出的最优序列"
          "**逐位相同** —— ★ 一个是增量剪枝、一个是全枚举，**两条独立路径会合**"
          % len(seeds),
          "**beam 与穷举对不上** —— ★ 先查 `B ≥ V^L` 时是否真的没有剪枝、以及打分的归一化")

    # ── A.2 ★★ 贪心 == beam(1)（两条独立实现会合）────────────────────────
    print()
    print("  ── A.2 ★★ **贪心 == `beam(1)`**：两条**独立实现**（直取 argmax vs 束机制）──")
    mism = [s for s in seeds if greedy(s, L) != beam_best(s, L, 1)]
    print("     `%-6s %-18s %-18s %s" % ("模型", "greedy", "beam(1)", "一致"))
    for s in seeds[:4]:
        print("     %-6s %-18s %-18s %s"
              % (s, greedy(s, L), beam_best(s, L, 1), "✓" if s not in mism else "✗"))
    print("     ★★ 全部 **%d 个**模型上一致 = %s" % (len(seeds), not mism))
    check("A.2", not mism and 1 in specs.SEARCH_BEAM_GRID,
          "★★ **判据成立**：**%d 个模型**上 `贪心 == beam(1)` —— ★★ **束宽为 1 的束搜索就是贪心**"
          "（两条独立实现会合）；★ 故「贪心」是**搜索的一个退化点**，不是另一件事"
          % len(seeds),
          "**贪心与 beam(1) 不一致** —— ★ 先查 beam 的排序键（累计对数概率）与剪枝宽度")

    # ── A.3 ★★★ beam 是【近似搜索】：贪心 / beam(2) 都会错过最优 ────────
    print()
    print("  ── A.3 ★★★ **beam 不精确**：贪心错过最优的模型数、`beam(2)` 仍错过的模型数 ──")
    g_miss = [s for s in seeds if greedy(s, L) != exhaustive_best(s, L)]
    b2_miss = [s for s in seeds if beam_best(s, L, 2) != exhaustive_best(s, L)]
    print("     ★ **贪心 ≠ 穷举最优** 的模型（%d 个）：%s" % (len(g_miss), g_miss))
    print("     ★ **`beam(2)` ≠ 穷举最优** 的模型（%d 个）：%s" % (len(b2_miss), b2_miss))
    print("     ★ 模型 `m1`：贪心 = %s，`beam(2)` = %s，穷举最优 = %s"
          % (greedy("m1", L), beam_best("m1", L, 2), exhaustive_best("m1", L)))
    check("A.3", len(g_miss) == PINNED_GREEDY_MISS and len(b2_miss) == PINNED_BEAM2_MISS
          and greedy("m1", L) == PINNED_M1_GREEDY
          and exhaustive_best("m1", L) == PINNED_M1_EXHAUSTIVE
          and beam_best("m1", L, 2) == PINNED_M1_BEAM2
          and beam_best("m1", L, 2) == exhaustive_best("m1", L),
          "★★★ **判据成立**：**贪心在 %d/%d 个模型上错过穷举最优**（★ 局部最优陷阱），"
          "**`beam(2)` 在 %d/%d 上仍错过**（★ ★★ **beam 是近似搜索，不是精确搜索**）；"
          "★ 模型 `m1`：贪心 `%s`、`beam(2)` `%s`、穷举 `%s` ⇒ **加宽束能找回最优**"
          % (len(g_miss), len(seeds), len(b2_miss), len(seeds),
             PINNED_M1_GREEDY, PINNED_M1_BEAM2, PINNED_M1_EXHAUSTIVE),
          "**错过最优的计数或某个具体序列被动过** —— ★ 先查玩具模型（哈希 logits）与剪枝键")


# ===========================================================================
# 5. B 类：代价模型 —— 候选集大小（beam 的束宽 / top-k 的 k / top-p 的动态）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **代价模型**：候选集大小 —— beam 的束宽 · top-k 的定值 · top-p 的动态")
    print("=" * 78)
    V, L = specs.SEARCH_VOCAB, specs.SEARCH_LEN

    # ── B.1 ★★ beam 的代价 = 活束数 B（每层恰好保留 B 条）─────────────────
    print()
    print("  ── B.1 ★★ **beam 的代价 = 活束数 `B`**：每层剪枝后恰好剩 `B` 条 ──")
    print("     %-10s %s" % ("B", "每层的活束数"))
    w_ok = True
    for B in specs.SEARCH_BEAM_GRID:
        _, widths = beam(specs.SEARCH_MODEL_SEEDS[0], L, B, trace=True)
        w_ok = w_ok and all(w == B for w in widths)
        print("     %-10d %s" % (B, widths))
    _, wfull = beam(specs.SEARCH_MODEL_SEEDS[0], L, V ** L, trace=True)
    print("     ★ `B = V^L = %d` 的活束数：%s（★ 末层 `%d`）" % (V ** L, wfull, wfull[-1]))
    check("B.1", w_ok and max(specs.SEARCH_BEAM_GRID) < V ** L,
          "★★ **判据成立**：束宽 `B ∈ %s` 时，**每一层剪枝后恰好保留 `B` 条**活序列 ——"
          " ★★ 这就是 beam 的**代价**：内存/算力 `∝ B`（与贪心的 `1` 拉开 `B` 倍）"
          % (specs.SEARCH_BEAM_GRID,),
          "**活束数不等于 B** —— ★ 先查剪枝回 `B` 的那一步与候选总数（`B ≤ V^depth` 才成立）")

    # ── B.2 ★★ top-k 定值 vs top-p 动态 ────────────────────────────────
    print()
    print("  ── B.2 ★★ **top-k 候选集【定值】 vs top-p 【动态】**（★ 两者的语义差）──")
    peaked = np.array(SAMPLE_PEAKED)
    flat = np.array(SAMPLE_FLAT)
    print("     %-10s %-14s %-14s %s" % ("阈值", "尖峰 |top-p|", "均匀 |top-p|", "两者相等?"))
    tp_pairs = []
    for thr in specs.SAMPLE_TOPP_GRID:
        a, b = len(top_p_set(peaked, thr)), len(top_p_set(flat, thr))
        tp_pairs.append((thr, a, b))
        print("     p=%-8.2f %-14d %-14d %s" % (thr, a, b, a == b))
    print("     %-10s %-14s %-14s %s" % ("k", "尖峰 |top-k|", "均匀 |top-k|", "两者相等?"))
    tk_pairs = []
    for k in specs.SAMPLE_TOPK_GRID:
        a, b = len(top_k_set(peaked, k)), len(top_k_set(flat, k))
        tk_pairs.append((k, a, b))
        print("     k=%-8d %-14d %-14d %s" % (k, a, b, a == b))
    # ★ top-p 的不变式：保留集合的累计概率 ∈ [thr, thr + p_max)
    inv_ok = True
    for p in (peaked, flat):
        for thr in specs.SAMPLE_TOPP_GRID:
            idx = top_p_set(p, thr)
            cum = float(np.sum(p[idx]))
            inv_ok = inv_ok and (thr <= cum < thr + float(p.max()) + 1e-12)
    check("B.2", all(a != b for (_, a, b) in tp_pairs)
          and all(a == b for (_, a, b) in tk_pairs)
          and inv_ok
          and len(top_p_set(peaked, 0.9)) == PINNED_TOPP_PEAKED_09
          and len(top_p_set(peaked, 0.95)) == PINNED_TOPP_PEAKED_095
          and len(top_p_set(flat, 0.9)) == PINNED_TOPP_FLAT_09
          and len(top_p_set(flat, 0.5)) == PINNED_TOPP_FLAT_05,
          "★★ **判据成立**：★ **top-k 的候选集大小恒为 `k`**（尖峰与均匀相同）；"
          "★★ **top-p 的大小【随分布变】**（尖峰 `p=0.9` 给 `%d`、均匀给 `%d`）——"
          " 这正是「**动态候选集**」的含义；★ 且保留集合的累计概率恒在 `[p, p+p_max)` 内"
          % (len(top_p_set(peaked, 0.9)), len(top_p_set(flat, 0.9))),
          "**top-k / top-p 的对照被动过** —— ★ 先查 `top_p_set` 的累积截止与 `top_k_set` 的截断")

    # ── B.3 ★★ temperature 不改候选集大小 ──────────────────────────────
    print()
    print("  ── B.3 ★★ **temperature 只改形状，不改候选集大小**（恒为整个词表 `V`）──")
    lg = np.array(SAMPLE_LOGITS)
    print("     %-8s %-14s %-10s %s" % ("T", "熵 (nats)", "|候选集|", "argmax"))
    ents = []
    size_ok = True
    for T in specs.SAMPLE_TEMP_GRID:
        p = softmax_temp(lg, T)
        cols = int((p > 0).sum())
        size_ok = size_ok and cols == V
        ents.append(round(entropy(p), 6))
        print("     %-8.2f %-14.6f %-10d %d" % (T, entropy(p), cols, int(np.argmax(p))))
    mono = ents[0] < ents[1] < ents[2]
    print("     ★★ 读法：**熵随 `T` 单调上升**（`%s`），而 **`|候选集|` 恒为 `%d`** ——"
          % (ents, V))
    print("         ★★ 所以 **temperature 单独用【不降低代价】**；降代价要靠 **top-k / top-p 的截断**。")
    check("B.3", size_ok and mono
          and all(abs(e - pin) < 2e-6 for e, pin in zip(ents, PINNED_ENTROPY))
          and 1.0 in specs.SAMPLE_TEMP_GRID,
          "★★ **判据成立**：`T ∈ %s` 上熵 `%s`（**单调升**），而候选集大小**恒为 `%d`**"
          " —— ★★ **temperature 与 top-k / top-p 是两件不同的事**：前者**重加权**、后者**截断**"
          % (specs.SAMPLE_TEMP_GRID, ents, V),
          "**温度-熵的单调性或候选集大小被动过** —— ★ `T` 是否真的只除在 logits 上")


# ===========================================================================
# 6. C 类：边界与反例 —— 两端退化守卫
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**两端退化** —— `p→0`/`k=1` 退化成贪心；`p=1`/`k≥V` 退化成全集")
    print("=" * 78)
    V, L = specs.SEARCH_VOCAB, specs.SEARCH_LEN
    peaked = np.array(SAMPLE_PEAKED)
    flat = np.array(SAMPLE_FLAT)

    # ── C.1 ★★ top-p 的两端 ──────────────────────────────────────────
    print()
    print("  ── C.1 ★★ **top-p 的两端**：`p` 极小 ⇒ 单元素（= 贪心）；`p = 1` ⇒ 全集 ──")
    nvoc = len(peaked)                     # ★ 采样侧的词表大小 = 分布的支集大小（★ 不是 SEARCH_VOCAB）
    tiny_peak = top_p_set(peaked, 1e-6)
    tiny_flat = top_p_set(flat, 1e-6)
    one_peak = top_p_set(peaked, 1.0)
    print("     `p = 1e-6` 于尖峰 ⇒ 大小 %d（元素 %s）；于均匀 ⇒ 大小 %d" % (len(tiny_peak), tiny_peak, len(tiny_flat)))
    print("     `p = 1.0`  于尖峰 ⇒ 大小 %d；于均匀 ⇒ 大小 %d" % (len(one_peak), len(top_p_set(flat, 1.0))))
    check("C.1", len(tiny_peak) == 1 and len(tiny_flat) == 1
          and tiny_peak[0] == int(np.argmax(peaked)) and len(one_peak) == nvoc,
          "★★ **判据成立**：`p → 0` ⇒ 候选集缩到**单个** token（★ **正是贪心取的那个 argmax**）；"
          " `p = 1` ⇒ 候选集涨到**整个支集 `%d`** —— ★ 两端都退化成已知的极端" % nvoc,
          "**top-p 的端点被动过** —— ★ 先查累积截止（`≥ p`）与是否包含截止那一项")

    # ── C.2 ★★ top-k 的两端 ──────────────────────────────────────────
    print()
    print("  ── C.2 ★★ **top-k 的两端**：`k = 1` ⇒ 贪心；`k ≥ 词表` ⇒ 全集 ──")
    k1 = top_k_set(peaked, 1)
    kV = top_k_set(peaked, nvoc)
    kbig = top_k_set(flat, nvoc + 3)
    print("     `k = 1` ⇒ %s（= argmax %d）；`k = %d` ⇒ 大小 %d；`k = %d` ⇒ 大小 %d"
          % (k1, int(np.argmax(peaked)), nvoc, len(kV), nvoc + 3, len(kbig)))
    check("C.2", len(k1) == 1 and k1[0] == int(np.argmax(peaked))
          and len(kV) == nvoc and len(kbig) == nvoc,
          "★★ **判据成立**：`k = 1` ⇒ **单元素**（= 贪心）；`k ≥ 支集大小` ⇒ **被封顶到 `%d`**"
          " —— ★★ 所以 **top-k 的候选集大小是 `min(k, V)`，与分布无关**" % nvoc,
          "**top-k 的端点被动过** —— ★ 先查 `min(k, V)` 的封顶")

    # ── C.3 ★★ L = 1 时穷举 == 贪心（免费的独立守卫）────────────────────
    print()
    print("  ── C.3 ★★ **`L = 1` 时穷举 == 贪心**（长度 1 的搜索没有「陷阱」可言）──")
    ok1 = all(exhaustive_best(s, 1) == greedy(s, 1) for s in specs.SEARCH_MODEL_SEEDS)
    print("     `L = 1`：全部 **%d 个**模型上 `穷举 == 贪心` = %s" % (len(specs.SEARCH_MODEL_SEEDS), ok1))
    print("     ★★ 这是「`A.3` 的错过最优**必须靠长度 ≥ 2`**」的落点 ——")
    print("        长度 1 时每一步都是最后一步，**没有未来可牺牲** ⇒ 贪心必然最优。")
    check("C.3", ok1,
          "★★ **判据成立**：`L = 1` 时**穷举与贪心恒一致**（★ 免费守卫）——"
          " ★★ 反过来，`A.3` 的「贪心错过最优」**必须靠 `L ≥ 2` 的搜索空间**",
          "**长度 1 时贪心居然不是最优** —— ★ 那说明打分函数或枚举被动过")


# ===========================================================================
# 7. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无任何目标 LLM**；")
    print("      ★★ 但**本程序的 A 类是真的**：**beam 与穷举是两条独立路径**（搜索空间可数）。")
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

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: decoding-strategies" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: decoding-strategies`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                                        "-- FAIL: engines", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.DECODE_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == ()
          and abs(SCORE_EPS - 1e-9) < 1e-18,
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住；"
          "★ 容差 `SCORE_EPS` **在 preflight 里就钉住**（★ 否则它的变异会变成**崩溃捕获**、"
          "而不是断言捕获）" % specs.DECODE_FETCH_DAY)

    # ★★★ 口径守卫（★ 跑在任何计算之前）：`A.1` 的「不剪枝」要成立，**束宽网格的上端必须真能封住**；
    #   `A.2` 要成立，网格**必须含 `1`**；若网格里出现了 `≥ V^L`，`A.1`/`A.2` 的分工就塌了。
    V, L = specs.SEARCH_VOCAB, specs.SEARCH_LEN
    full = V ** L
    ok_grid = (1 in specs.SEARCH_BEAM_GRID
               and all(1 <= b < full for b in specs.SEARCH_BEAM_GRID)
               and len(specs.SEARCH_MODEL_SEEDS) >= 2
               and V >= 2 and L >= 2
               and max(specs.SEARCH_BEAM_GRID) < full)
    print("     ★★ 口径守卫：网格含 `1` = %s；`max(B) < V^L = %d` = %s；`V,L ≥ 2` = %s"
          % (1 in specs.SEARCH_BEAM_GRID, full, max(specs.SEARCH_BEAM_GRID) < full, V >= 2 and L >= 2))
    check("D.3b", ok_grid,
          "★★★ **口径守卫**：束宽网格**必须含 `1`**（`A.2` 的落点）、且**必须都 `< V^L`**"
          "（否则 `A.1` 的「`B = V^L` 时不剪枝」与网格里的档会重复）——"
          " ★ 缺了任一条，`A.1`/`A.2` 的分工就**退化成恒真**（Chunk 1 的 `H1` 形态）",
          "**网格的覆盖被动过** —— ★ 先查 `SEARCH_BEAM_GRID` / `SEARCH_VOCAB` / `SEARCH_LEN`")
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性 · 术语
# ===========================================================================
def _names_in(obj):
    src = textwrap.dedent(inspect.getsource(obj))
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 独立性 · 术语纪律")
    print("=" * 78)
    V, L = specs.SEARCH_VOCAB, specs.SEARCH_LEN
    full = V ** L

    # ── D.4 ★★★ 守卫实验 ①：把 beam 改成「只展开最优子节点」────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：beam 每步只展开**最优那一条**束的**最优子节点** ──")
    g_bad_miss = [s for s in specs.SEARCH_MODEL_SEEDS
                  if beam_best(s, L, full, corrupt="greedy_child") != exhaustive_best(s, L)]
    print("     `B = V^L` 的**正常** beam ⇒ 错过最优 **0** 个；**改坏**的 beam ⇒ 错过 **%d** 个"
          % len(g_bad_miss))
    check("D.4", len(g_bad_miss) >= 1
          and all(beam_best(s, L, full) == exhaustive_best(s, L) for s in specs.SEARCH_MODEL_SEEDS),
          "★★★ **守卫成立**：把 beam 的**展开**改坏（只跟最优子节点）后，它在 **%d 个**模型上"
          "**不再等于穷举** —— ★★ 于是 `A.1` 的「逐位一致」对「**每步展开全部 `V` 个子节点**」"
          "这条规则**是敏感的**" % len(g_bad_miss),
          "**守卫失效** —— ★ 改坏展开后 beam 居然仍等于穷举（那 `A.1` 抓不到它 ⇒ 需要加判据）")

    # ── D.5 ★★★ 守卫实验 ②：把穷举改成「返回最差序列」────────────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：把**穷举**改成「返回**最差**序列」(`worst`) ──")
    bad_ref = [s for s in specs.SEARCH_MODEL_SEEDS
               if beam_best(s, L, full) != exhaustive_best(s, L, corrupt="worst")]
    print("     `N = %d` 个模型上，正常穷举 == beam **全部成立**；坏穷举 == beam 的有 **%d** 个"
          % (len(specs.SEARCH_MODEL_SEEDS), len(bad_ref)))
    check("D.5", len(bad_ref) == len(specs.SEARCH_MODEL_SEEDS),
          "★★★ **守卫成立**：把参照（穷举）改成「返回最差序列」后，它与 beam 在 **全部 %d 个**"
          "模型上**都对不上** ⇒ `A.1` 对**参照的表达式**也敏感" % len(bad_ref),
          "**守卫失效** —— ★ 穷举返回最差序列后仍与 beam 一致（那 `A.1` 抓不到它）")

    # ── D.6 ★★★ 引文/取值逐字：`specs` 的搜索与采样常量 ────────────────────
    check("D.6", specs.SEARCH_VOCAB == 4 and specs.SEARCH_LEN == 4
          and specs.SEARCH_BEAM_GRID == (1, 2, 3)
          and specs.SEARCH_MODEL_SEEDS == ("m0", "m1", "m2", "m3", "m4", "m5",
                                           "m6", "m7", "m8", "m9", "m10", "m11")
          and specs.SAMPLE_TOPP_GRID == (0.5, 0.9, 0.95)
          and specs.SAMPLE_TOPK_GRID == (1, 2, 3)
          and specs.SAMPLE_TEMP_GRID == (0.5, 1.0, 2.0),
          "★★★ **取值逐元素钉住**：玩具词表/长度、束宽网格、**12 个模型盐值**、"
          " top-p / top-k / temperature 三个网格 —— ★★ 每个元组都**逐元素** `==`（★ 不靠「整体」）",
          "**某个 `specs` 常量被动过** —— ★ 先查 `specs.py` 第 2 节")

    # ── D.7 ★★★ 逐元素钉住（程序内常量）────────────────────────────────
    check("D.7", specs.DECODE_FETCH_DAY == "2026-10-09"
          and SAMPLE_PEAKED == (0.90, 0.05, 0.03, 0.015, 0.005)
          and SAMPLE_FLAT == (0.2, 0.2, 0.2, 0.2, 0.2)
          and SAMPLE_LOGITS == (0.3, 0.9, 0.1, 0.5)
          and abs(sum(SAMPLE_PEAKED) - 1.0) < 1e-12 and abs(sum(SAMPLE_FLAT) - 1.0) < 1e-12
          and PINNED_N_MODELS == len(specs.SEARCH_MODEL_SEEDS)
          and PINNED_GREEDY_MISS == 8 and PINNED_BEAM2_MISS == 2
          and PINNED_M1_GREEDY == (2, 2, 0, 0) and PINNED_M1_EXHAUSTIVE == (2, 1, 3, 3)
          and PINNED_M1_BEAM2 == (2, 1, 3, 3)
          and PINNED_TOPP_PEAKED_09 == 1 and PINNED_TOPP_PEAKED_095 == 2
          and PINNED_TOPP_FLAT_09 == 5 and PINNED_TOPP_FLAT_05 == 3
          and PINNED_ENTROPY == (1.203755, 1.340090, 1.374964)
          and abs(SCORE_EPS - 1e-9) < 1e-18,
          "★★★ **程序内常量逐元素 + 分布归一化**钉住（两个分布各 5 个元素、固定 logits 4 个、"
          " 「错过最优」的两个计数、`m1` 的三个序列、四个 top-p 大小、三个熵值）",
          "**某个程序内常量被动过**")

    # ── D.8 ★★★ 独立性（源码级）：两实现互不引用 ────────────────────────
    bn = _names_in(beam)
    en = _names_in(exhaustive_best)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     `beam` 里出现 `exhaustive_best` = %s ；`exhaustive_best` 里出现 `beam` = %s"
          % ("exhaustive_best" in bn, "beam" in en))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", "exhaustive_best" not in bn and "beam" not in en
          and "beam_best" not in bn and "greedy" not in en,
          "★★ **源码级线索成立**：**被测（增量剪枝）与参照（全枚举）互不引用**（★ **两侧都审**）——"
          " ★★ 于是「两条路径一致」**不是恒真**，而是**两条独立路径的会合**",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 术语纪律 + 限定语（★ 机械判定）──────────────────────────
    src = open(__file__, encoding="utf-8").read()
    print()
    print("  ── D.9 ★★ **术语纪律**（`§2.2`）：一律写「%s」全称；限定语逐字出现 ──"
          % TERM_AUTOREGRESSIVE)
    check("D.9", TERM_AUTOREGRESSIVE == "自回归解码" and TERM_INFLIGHT == "推理批 / 在飞请求数"
          and TERM_SCHEDULER == "请求调度器"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_AUTOREGRESSIVE in src,
          "★★ **判据成立**：三条术语常量 `%r` / `%r` / `%r`（★ **不裸写** `decode` / `batch` / "
          "`scheduler` —— 它们在本库都另有含义）；★ 限定语 `%r` **逐字出现在本程序源码里**"
          % (TERM_AUTOREGRESSIVE, TERM_INFLIGHT, TERM_SCHEDULER, LIMIT_QUOTE),
          "**术语或限定语被动过** —— ★ 本域一律写全称、并带最强限定语")


# ===========================================================================
# 9. 汇总 / 主程序
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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：搜索空间可数（beam vs 穷举）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：玩具模型是 **`[E]` 级（本库实现 / 假设）**；"
          "★ **本程序不引用任何外部来源的数字**（采样参数在框架里的名字回链 `70-frameworks`）。")
    print("★ 本程序只核对「**解码算法的代数与不变量**」——**不是**任何真实模型的采样行为。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_sampling_and_search.py —— 采样与搜索：greedy / beam / temperature / top-k / top-p")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何目标 LLM。")
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
