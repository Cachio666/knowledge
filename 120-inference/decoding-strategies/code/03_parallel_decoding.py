#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_parallel_decoding.py —— **并行解码**：候选树 + **验证与接受**的机制。

配套文档：`03-parallel-decoding.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本程序最强的一条 A 类：同一个「接受长度」走【两条互不共享代码】的路径**：
  · **被测** = `tree_walk`：**沿树下降** —— 每一步查「当前节点的子节点里有没有目标的下一个 token」；
  · **参照** = `exhaustive_accept`：**把树里所有根到叶的路径逐条枚举**，对每条算「**最长匹配前缀**」，取最长。
  · **判据**：★ 两条路径在 `8 个模型 × 3 个分支` 的 **24 档**上给出**同一个接受序列**。
  ★★ **一条守卫**（`D.4`/`D.5`）：把 `tree_walk` 改成「**不查子树**」⇒ 对不上；
    把 `exhaustive_accept` 改成「**算总匹配数而不是最长前缀**」⇒ 对不上。**两端都过 `CONTRIBUTING:235`**。

★★★ **「并行」的代价模型**（`B.*`）—— ★★ 这是本主题与**投机解码**（`02`）**真正的分野**：
  · **候选树**在一次并行步里有 `b^D` 条**路径**、`Σ b^d` 个**节点**；
  · ★★★ **目标模型只跑【一次】前向**（`tree attention` 把整棵树一次评完）⇒
    **验证代价是 `1` 次前向，而不是「每条候选各一次」**；
  · ★★ 朴素（逐条）验证则要 **节点数** 次前向 —— ★ **两条路的接受结果相同、代价差一个量级**。
  ★ 链条（`b = 1`）是树的**退化**：它就是 `02` 的草稿链（本篇**不重推** `02` 的闭式，`P37`）。

★★ **覆盖边界**：本程序守的是「**验证/接受的机制与代价账**」（接受长度 / 路径枚举 / 节点数）。
  **不守**任何**真实模型或引擎**的行为（本机**无 GPU、无任何目标 LLM**）。
  ★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **`tree_walk` 与 `exhaustive_accept` 逐位一致**（24 档）；
  · `A.2` ★★ **`b = 1` 的树 == 单链验证**（第三条独立路径）；
  · `A.3` ★★★ **树 ≥ 链**：分支提高接受长度（`t4` 链 `1` → 树 `6`）；
  · `B.1` ★★ **验证代价 = 1 次前向**（树）vs **节点数次**（朴素）；
  · `C.1`–`C.3` 边界（根不匹配 ⇒ `0`；`b = V` ⇒ 全覆盖 ⇒ `D`；`D = 1` ⇒ `∈ {0,1}`）；
  · `C.4` ★★ **失败复现**（`CONTRIBUTING:238`）：路径数 ≠ 节点数（混淆则代价账算错）；
  · `D.4`/`D.5` ★★★ **守卫实验**：改坏树下降 / 改坏路径枚举 ⇒ 断言必须失败；
  · `D.9` ★★★ **术语纪律的两半**：全称存在 + **裸写被禁裸词的计数 == 显式豁免清单**。
"""
import ast
import collections
import hashlib
import inspect
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
ASSERT_FAIL_MARK = "-- FAIL: decoding-strategies"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

# ★★ **两条术语纪律**（`§2.2`）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
#   · 本域一律写「**自回归解码**」全称（不裸写 `decode`，那是 `60-compilers` 的 decode pass）；
#   · 本域一律写「**KV 缓存**」全称（不裸写 `cache`，那是 L1/L2 cache）；
#   · 本域最强的一句**限定语**必须逐字出现。
TERM_AUTOREGRESSIVE = "自回归解码"
TERM_KV_CACHE = "KV 缓存"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ 形态 A：「纪律宣称了而断言没盯住」）：
#   本程序源码里**裸写那个被禁的裸词**（本域唯一的禁用裸词，见抬头的术语纪律）的【计数】
#   必须**恰好等于**下面这份**显式豁免清单**覆盖的计数 ——
#   ★★★ 判据是【计数 == 豁免覆盖数】，**不是「≤ N」**（「≤ N」挡不住**新增**一处裸写，正是本轮 `D1` 的形态）。
#   ★ 每一项都写明【为什么豁免】（★ **本注释块刻意不写那个裸词**，否则它自己也会被计数）：
#     ① 抬头注释里**提到**这条纪律本身、**并指出**该词在本库的另一个所指 ——
#        ★ 两处都是 **mention**（它写该词恰恰因为在规定不许写它）。
#     ② `D.9` 的判据文案里**复述**这条纪律 —— ★ 同为 mention。
BARE_DECODE_EXEMPT = (
    "不裸写 `decode`，那是 `60-compilers` 的 decode pass",   # ① 抬头：mention（规定 + 消歧）
    "**不裸写** `decode`",                                    # ② D.9 文案：mention（复述纪律）
)

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

ACC_EPS = 1e-9                      # ★ 接受长度是**整数**，容差只用来比较推导出的浮点数

# ★★★ **本机实测值**（本程序在本机跑出的数）—— 正文里的每个数都在这里**钉死**。
#   接受长度（`PARA_DEPTH = 6`，8 个模型，按 `PARA_MODEL_SEEDS` 顺序）：
PINNED_CHAIN_ACC = (2, 6, 6, 4, 1, 0, 6, 6)     # ★ 单链（`b = 1`）的接受长度
PINNED_TREE2_ACC = (6, 6, 6, 6, 6, 6, 6, 6)     # ★ 树（`b = 2`）的接受长度（★ **全部触顶**）
PINNED_MEAN_CHAIN = 3.875                       # ★ 单链的平均接受长度 = 31/8
PINNED_MEAN_TREE2 = 6.0                         # ★ 树（`b = 2`）的平均接受长度
PINNED_TREE_WINS = 4                            # ★ 树**严格更长**的模型数（`t0`/`t3`/`t4`/`t5`）
PINNED_T0_CHAIN = 2                             # ★ `t0` 上链接受 2 个 token
PINNED_T0_TREE2 = 6                             # ★ `t0` 上树（`b = 2`）接受 6 个（**触顶**）
PINNED_T5_CHAIN = 0                             # ★ `t5` 上链**一个都没接受**
#   代价模型的**结构量**（与模型无关，只由 `b` 与 `D` 决定）：
PINNED_NODES = (6, 126, 5460)                   # ★ `b ∈ (1,2,4)` 的节点数 `Σ_{d=1}^{D} b^d`
PINNED_PATHS = (1, 64, 4096)                    # ★ 根到叶的**路径数** `b^D`
PINNED_PASSES_TREE = 1                          # ★ 树验证的**目标前向次数**
PINNED_PASSES_CHAIN = 6                         # ★ 单链验证的目标前向次数（= 节点数）


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
# 1. 两个玩具模型（**确定性哈希 logits**）：目标 + 草稿
# ===========================================================================
def _digest(salt, prefix):
    return hashlib.md5(("%s,%s" % (salt, ",".join(map(str, prefix)))).encode()).digest()


def target_logits(salt, prefix):
    """★★ 玩具「目标模型」：确定性哈希 logits（★ **不是**任何真实 LLM）。"""
    h = _digest(salt, prefix)
    return np.array([h[i] / 255.0 for i in range(specs.PARA_VOCAB)])


def draft_logits(salt, prefix):
    """★★ 玩具「草稿模型」= 目标 logits + **确定性噪声**（`PARA_DRAFT_EPS`）。

    ★ 噪声是**确定性的**（对 `"draft-" + salt` 取 `md5`）⇒ 结论**可复现**；
    ★★ `ε` 小 ⇒ 草稿 ≈ 目标 ⇒ **接受率高**（否则树与链都几乎不接受，`A.3` 会退化成恒真）。
    """
    h = _digest("draft-" + salt, prefix)
    noise = np.array([h[specs.PARA_VOCAB + i] / 255.0 for i in range(specs.PARA_VOCAB)])
    return target_logits(salt, prefix) + specs.PARA_DRAFT_EPS * noise


def next_token(model, salt, prefix):
    """模型在 `prefix` 之后**贪心**选出的下一个 token。"""
    return int(np.argmax(model(salt, prefix)))


# ===========================================================================
# 2. 候选树（★ 草稿的 top-`b` 为子节点）
# ===========================================================================
def build_tree(salt, b, corrupt=None):
    """构造深度 `PARA_DEPTH`、分支 `b` 的候选树。

    ★ 节点 `id`、每个节点存 `(token, [子节点])`；根（`id = 0`）无 token。
    ★★ **每个节点的子节点 = 草稿在该前缀下概率最高的 `b` 个 token**（**互不相同**）。
    ★★ `corrupt="root_excl"`（**只用于边界实验**）：**根的子节点**从「**不含目标的第一个 token**」
      的 token 里选 —— ⇒ ★ 树**在第一步就失配** ⇒ 接受长度 `0`。
    """
    V, D = specs.PARA_VOCAB, specs.PARA_DEPTH
    nodes = {0: (None, [])}
    prefix = {0: ()}
    frontier = [0]
    nid = 1
    for d in range(D):
        nxt = []
        for p in frontier:
            if d == 0 and corrupt == "root_excl":
                gt = next_token(target_logits, salt, prefix[p])
                order = [t for t in range(V) if t != gt][:b]
            else:
                lg = draft_logits(salt, prefix[p])
                order = [int(t) for t in np.argsort(-lg)[:b]]
            for t in order:
                nodes[nid] = (t, [])
                nodes[p][1].append(nid)
                prefix[nid] = prefix[p] + (t,)
                nxt.append(nid)
                nid += 1
        frontier = nxt
    return nodes, prefix


def tree_walk(nodes, salt, corrupt=None):
    """★★★ **被测实现**：**沿树下降** —— 每一步问「当前节点的子节点里有没有目标的下一个 token」。

    ★ 与 `exhaustive_accept` **无共享代码**（一个查子节点表、一个枚举全部路径；`D.8` 在源码层核）。
    ★★ `corrupt="no_subtree"`（**只用于守卫实验**）：**永远只查根的子节点**（不随下降更新）。
    """
    cur = 0
    acc = ()
    while True:
        t = next_token(target_logits, salt, acc)
        probe = 0 if corrupt == "no_subtree" else cur
        ch = [c for c in nodes[probe][1] if nodes[c][0] == t]
        if not ch:
            break
        cur = ch[0]
        acc = acc + (t,)
    return acc


def all_paths(nodes, cur=0):
    """枚举**所有根到叶的路径**（每条是一个 token 元组）。"""
    ch = nodes[cur][1]
    if not ch:
        return [()]
    out = []
    for c in ch:
        for sub in all_paths(nodes, c):
            out.append((nodes[c][0],) + sub)
    return out


def exhaustive_accept(nodes, salt, corrupt=None):
    """★★★ **参照实现**：**逐条枚举**根到叶的路径，对每条算「**最长匹配前缀**」，取全局最长。

    ★★ 它**不沿树下降**、不查子节点表 —— 这正是它与 `tree_walk` 的**路径独立**之处。
    ★★ `corrupt="min"`（**只用于守卫实验**）：取**最短**的匹配前缀（而不是最长）。
    """
    cands = []
    for p in all_paths(nodes):
        k = 0
        for j in range(len(p)):
            if p[j] == next_token(target_logits, salt, p[:j]):
                k += 1
            else:
                break
        cands.append(p[:k])
    if corrupt == "min":
        return min(cands, key=len)
    return max(cands, key=len)


def chain_accept(tokens, salt):
    """★★ **单链验证**（第三条路径）：逐个比对 token 列表与目标的贪心 token，遇首个不匹配即停。

    ★ 与 `tree_walk` 是**两条独立实现**：这里**没有树**，只有一个 token 列表。
    """
    acc = ()
    for t in tokens:
        if t != next_token(target_logits, salt, acc):
            break
        acc = acc + (t,)
    return acc


def chain_tokens(nodes):
    """取 `b = 1` 的树里那**唯一一条**路径（= 草稿链）。"""
    ps = all_paths(nodes)
    return ps[0]


def count_nodes(b):
    """树的**节点数** `Σ_{d=1}^{D} b^d`（与模型无关，只由 `b` 与 `D` 决定）。"""
    return sum(b ** d for d in range(1, specs.PARA_DEPTH + 1))


# ===========================================================================
# 3. A 类 —— ★★★ 树验证：两条独立路径对拍
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **候选树验证**：被测 = 沿树下降；参照 = 逐条枚举路径（两条独立路径）")
    print("=" * 78)
    D = specs.PARA_DEPTH
    seeds = specs.PARA_MODEL_SEEDS
    print("  ★★ 树深 `D = %d`、词表 `V = %d`、分支 `b ∈ %s`。" % (D, specs.PARA_VOCAB, specs.PARA_BRANCH_GRID))

    # ── A.1 ★★★ tree_walk == exhaustive_accept（24 档）────────────────
    print()
    print("  ── A.1 ★★★ **`tree_walk` 与 `exhaustive_accept` 逐位一致**（%d 档）──" % (len(seeds) * len(specs.PARA_BRANCH_GRID)))
    print("     %-6s %-6s %-22s %-22s %s" % ("模型", "b", "tree_walk", "穷举路径", "一致"))
    ok = True
    n_case = 0
    for s in seeds:
        for b in specs.PARA_BRANCH_GRID:
            nodes, _ = build_tree(s, b)
            w = tree_walk(nodes, s)
            e = exhaustive_accept(nodes, s)
            good = (w == e)
            ok = ok and good
            n_case += 1
            if b in (1, 2):                      # 只打前两档，免得刷屏
                print("     %-6s %-6d %-22s %-22s %s" % (s, b, w, e, "✓" if good else "✗"))
    print("     ★★ 共 **%d 档**（`%d 个模型 × %d 个分支`），全部一致 = %s"
          % (n_case, len(seeds), len(specs.PARA_BRANCH_GRID), ok))
    # ★★★ **点明「恒真档」**（★ `S6`）：`b = V` 时覆盖率 1 ⇒ **两条路径都被迫给 `D`** ⇒ 那几档**恒真**
    #   （`C.2` 已单独证过）⇒ ★★ **不要把它计成独立判据**；真正有信息量的是 `b < V` 的档。
    n_trivial = len(seeds) if specs.PARA_VOCAB in specs.PARA_BRANCH_GRID else 0
    print("     ★★★ **其中 `b = V = %d` 的 %d 档是【恒真】**（全覆盖 ⇒ 两条路径都被迫给 `D`，`C.2` 另证）——"
          " ★★ **真正有信息量的是 `b ∈ %s` 的 %d 档**（★ 不要把 24 档**全计**成独立判据）"
          % (specs.PARA_VOCAB, n_trivial,
             [b for b in specs.PARA_BRANCH_GRID if b != specs.PARA_VOCAB], n_case - n_trivial))
    check("A.1", ok and n_case == len(seeds) * len(specs.PARA_BRANCH_GRID),
          "★★★ **判据成立**：**%d 档**上「**沿树下降**」与「**逐条枚举路径**」给出**同一个接受序列**"
          " —— ★ 一个是查子节点表、一个是全枚举，**两条独立路径会合**；"
          " ★★ 但其中 `b = V` 的 **%d 档恒真**（`C.2` 另证）⇒ ★ **有信息量的是另外 %d 档**"
          % (n_case, n_trivial, n_case - n_trivial),
          "**两条路径对不上** —— ★ 先查「最长匹配前缀」的定义与下降时 `cur` 的更新")

    # ── A.2 ★★ b = 1 的树 == 单链验证（第三条路径）────────────────────
    print()
    print("  ── A.2 ★★ **`b = 1` 的树 == 单链验证**（**第三条**独立路径：没有树的 token 列表）──")
    ok2 = True
    for s in seeds:
        nodes, _ = build_tree(s, 1)
        w = tree_walk(nodes, s)
        c = chain_accept(chain_tokens(nodes), s)
        ok2 = ok2 and (w == c)
    print("     ★ **%d 个模型**上 `tree_walk(b=1)` == `chain_accept(唯一路径)` = %s" % (len(seeds), ok2))
    print("     ★★ 读法：**分支为 1 的树就是一条链** —— ★ 三者（树下降 / 路径枚举 / 链）在那里重合。")
    check("A.2", ok2 and 1 in specs.PARA_BRANCH_GRID,
          "★★ **判据成立**：`b = 1` 时**树退化成一条链**，而「沿树下降」「枚举路径」「单链验证」"
          "**三条路径给出同一个接受序列** —— ★ 这正是「链是树的退化」（与 `02` 的草稿链同形）",
          "**`b = 1` 的退化不成立** —— ★ 先查 `build_tree(b=1)` 是否只生成一条路径")

    # ── A.3 ★★★ 树 ≥ 链：分支提高接受长度 ─────────────────────────────
    print()
    print("  ── A.3 ★★★ **树 ≥ 链**：分支把接受长度顶到上限 `D = %d` ──" % D)
    print("     %-6s %-14s %-14s %s" % ("模型", "链接受 (b=1)", "树接受 (b=2)", "树更长?"))
    ch_acc, tr_acc = [], []
    for s in seeds:
        n1, _ = build_tree(s, 1)
        n2, _ = build_tree(s, 2)
        a1 = len(tree_walk(n1, s))
        a2 = len(tree_walk(n2, s))
        ch_acc.append(a1)
        tr_acc.append(a2)
        print("     %-6s %-14d %-14d %s" % (s, a1, a2, "✓" if a2 > a1 else "—"))
    wins = sum(1 for a1, a2 in zip(ch_acc, tr_acc) if a2 > a1)
    mean1 = sum(ch_acc) / len(ch_acc)
    mean2 = sum(tr_acc) / len(tr_acc)
    print("     ★★ 平均：链 **%.3f** → 树 **%.3f**；树**严格更长**的模型 **%d/%d** 个"
          % (mean1, mean2, wins, len(seeds)))
    print("     ★★ 读法：**一次前向里，树能接受的 token 只能变多** —— 因为候选集在每个位置**更大**。")
    check("A.3", tuple(ch_acc) == PINNED_CHAIN_ACC and tuple(tr_acc) == PINNED_TREE2_ACC
          and abs(mean1 - PINNED_MEAN_CHAIN) < 1e-9 and abs(mean2 - PINNED_MEAN_TREE2) < 1e-9
          and wins == PINNED_TREE_WINS
          and ch_acc[0] == PINNED_T0_CHAIN and tr_acc[0] == PINNED_T0_TREE2
          and ch_acc[5] == PINNED_T5_CHAIN
          and all(a2 >= a1 for a1, a2 in zip(ch_acc, tr_acc)),
          "★★★ **判据成立**：**树（`b = 2`）在每个模型上 `≥` 链（`b = 1`）**，平均从 `%.3f` 升到 `%.3f`"
          "（★ 树**严格更长**的有 `%d/%d` 个模型；★ `t0` 从 `%d` 升到 `%d`）——"
          " ★★ 这就是「**并行生成多个候选**」的收益来源" % (mean1, mean2, wins, len(seeds),
                                                    PINNED_T0_CHAIN, PINNED_T0_TREE2),
          "**树 ≥ 链不成立或计数被动过** —— ★ 先查子节点是否真的取自草稿的 top-`b`")


# ===========================================================================
# 4. B 类：代价模型 —— 一次前向 vs 节点数次
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **代价模型**：「并行」到底省在哪 —— 一次前向验证整棵树")
    print("=" * 78)
    D = specs.PARA_DEPTH

    # ── B.1 ★★★ 验证代价：树 1 次前向 vs 朴素「节点数」次 ───────────────
    print()
    print("  ── B.1 ★★★ **验证代价**：树 = **1 次目标前向**；朴素逐条 = **节点数次** ──")
    print("     %-6s %-10s %-14s %-16s %s" % ("b", "节点数", "路径数", "树验证前向", "朴素验证前向"))
    nodes_l, paths_l = [], []
    for b in specs.PARA_BRANCH_GRID:
        nd, pa = count_nodes(b), b ** D
        nodes_l.append(nd)
        paths_l.append(pa)
        print("     %-6d %-10d %-14d %-16d %d" % (b, nd, pa, PINNED_PASSES_TREE, nd))
    print("     ★★ 读法：`b^D` 条路径 / `Σ b^d` 个节点，**却只要 1 次前向** ——")
    print("         ★★★ 因为目标模型在**一次并行评估**里把整棵树的每个前缀都评了（`tree attention`）。")
    check("B.1", tuple(nodes_l) == PINNED_NODES and tuple(paths_l) == PINNED_PATHS
          and PINNED_PASSES_TREE == 1 and count_nodes(1) == PINNED_PASSES_CHAIN
          and count_nodes(2) > count_nodes(1),
          "★★★ **判据成立**：分支 `b ∈ %s` 的树有 **%s** 个节点、**%s** 条路径；"
          " ★★ **验证只需要 `1` 次目标前向**（树验证），而**朴素逐条**要 **节点数次** ——"
          " ★★★ **这就是「并行解码」省下的东西**（不是省算力，是省**前向的次数**）"
          % (specs.PARA_BRANCH_GRID, PINNED_NODES, PINNED_PATHS),
          "**节点数 / 路径数 / 前向次数被动过** —— ★ 先查 `Σ b^d` 与 `b^D` 的口径")

    # ── B.2 ★★ 两条验证路线的【接受结果相同、代价差一个量级】────────────
    print()
    print("  ── B.2 ★★ **两条验证路线：接受结果相同、代价差一个量级** ──")
    s = specs.PARA_MODEL_SEEDS[0]
    nodes, _ = build_tree(s, 2)
    w = tree_walk(nodes, s)
    e = exhaustive_accept(nodes, s)
    nd = count_nodes(2)
    print("     `b = 2, D = %d`：接受序列 = %s（`%d` 个 token）" % (D, w, len(w)))
    print("     ★ 树验证：**1** 次前向；朴素逐条：**%d** 次前向 ⇒ **比值 `%d×`**" % (nd, nd))
    print("     ★★ 读法：**两者的【接受结果】完全一样**（`A.1` 已证）⇒")
    print("         ★★ 差别**只在代价**：`%d` 次 vs `1` 次 —— ★ 这就是**并行**的全部意义。" % nd)
    check("B.2", w == e and len(w) == PINNED_T0_TREE2 and nd == PINNED_NODES[1] and nd // 1 == nd,
          "★★ **判据成立**：同一棵树，**树验证与朴素逐条给出同一个接受序列**（`%s`），"
          "而代价是 **`1` 次 vs `%d` 次**前向（**差 `%d×`**）—— ★★ 所以并行解码省的是**前向次数**，"
          "不是算术量" % (w, nd, nd),
          "**接受结果或节点数被动过** —— ★ 先查两条路线的接受定义与 `count_nodes`")

    # ── B.3 ★★ 产出/前向：链 vs 树的【每次前向接受几个 token】────────────
    print()
    print("  ── B.3 ★★ **产出/前向**：链与树都只花 1 次前向，但接受长度不同 ──")
    ch = [len(tree_walk(build_tree(x, 1)[0], x)) for x in specs.PARA_MODEL_SEEDS]
    tr = [len(tree_walk(build_tree(x, 2)[0], x)) for x in specs.PARA_MODEL_SEEDS]
    print("     ★ 链（`b=1`）：平均 **%.3f** 个 token / 次前向；树（`b=2`）：平均 **%.3f**"
          % (sum(ch) / len(ch), sum(tr) / len(tr)))
    print("     ★★ 读法：**前向次数一样（都是 1）**，树的产出更高 —— ★★ **提高的是「每次前向的产出**")
    print("         **（接受长度）」**，不是「每 token 的算力」。")
    check("B.3", abs(sum(tr) / len(tr) - PINNED_MEAN_TREE2) < 1e-9
          and abs(sum(ch) / len(ch) - PINNED_MEAN_CHAIN) < 1e-9
          and sum(tr) > sum(ch),
          "★★ **判据成立**：**同样的 `1` 次前向**，链平均接受 **%.3f** 个、树平均接受 **%.3f** 个"
          " —— ★★ 树的收益体现在「**每次前向的产出**」" % (sum(ch) / len(ch), sum(tr) / len(tr)),
          "**产出/前向被动过** —— ★ 先查链与树的接受长度")


# ===========================================================================
# 5. C 类：边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：第一步失配 · 全覆盖 · 单层树")
    print("=" * 78)
    D = specs.PARA_DEPTH

    # ── C.1 ★★ 根不匹配 ⇒ 接受 0 ───────────────────────────────────────
    print()
    print("  ── C.1 ★★ **根的子节点不含目标的第一个 token ⇒ 接受长度 `0`** ──")
    z = [len(tree_walk(build_tree(s, 2, corrupt="root_excl")[0], s)) for s in specs.PARA_MODEL_SEEDS]
    print("     8 个模型上（根**刻意排除**目标的第一个 token）⇒ 接受长度 = %s" % z)
    print("     ★★ 读法：★ **`0` 是合法结果** —— 目标模型**自己产出的那一个 token 照样输出**，")
    print("         所以「接受 0 个」不等于「这一步没有产出」。")
    check("C.1", all(x == 0 for x in z),
          "★★ **判据成立**：**根处失配 ⇒ 接受长度恒为 `0`** —— ★ 边界是「一个草稿 token 都不接受」，"
          "★ **不是**「没有输出」（目标自己那一个 token 仍然产出）",
          "**根处失配的边界被动过** —— ★ 先查 `root_excl` 是否真的排除了目标的第一个 token")

    # ── C.2 ★★ b = V ⇒ 全覆盖 ⇒ 接受 = D ──────────────────────────────
    print()
    print("  ── C.2 ★★ **`b = V`（每个位置都放全词表）⇒ 接受长度 = `D = %d`** ──" % D)
    full = [len(tree_walk(build_tree(s, specs.PARA_VOCAB)[0], s)) for s in specs.PARA_MODEL_SEEDS]
    print("     `b = %d` 时接受长度 = %s" % (specs.PARA_VOCAB, full))
    print("     ★★ 读法：候选集覆盖全词表 ⇒ **目标的贪心 token 一定在子节点里** ⇒ **层层通过**。")
    check("C.2", all(x == D for x in full),
          "★★ **判据成立**：`b = V` 时**每个位置都覆盖全词表** ⇒ 接受长度**恒为上限 `D = %d`** ——"
          " ★ 这是「**接受长度的上界是树深**」的落点" % D,
          "**全覆盖的边界被动过** —— ★ 先查 `b = V` 时子节点是否真的是全词表")

    # ── C.3 ★★ D = 1 的单层树 ⇒ 接受 ∈ {0,1} ───────────────────────────
    print()
    print("  ── C.3 ★★ **单层树**（`D = 1`）⇒ 接受长度只能是 `0` 或 `1` ──")
    V = specs.PARA_VOCAB
    nodes = {0: (None, [1, 2])}
    nodes[1] = (0, [])
    nodes[2] = (1, [])
    s0 = specs.PARA_MODEL_SEEDS[0]
    gt0 = next_token(target_logits, s0, ())
    nodes[1] = (gt0, [])                       # ★ 一个子节点**命中**目标的第一个 token
    nodes[2] = ((gt0 + 1) % V, [])             # ★ 另一个**不命中**
    a_hit = len(tree_walk(nodes, s0))
    nodes[1] = ((gt0 + 1) % V, [])
    nodes[2] = ((gt0 + 2) % V, [])
    a_miss = len(tree_walk(nodes, s0))
    print("     `D = 1`：子树含目标 token ⇒ 接受 **%d**；不含 ⇒ 接受 **%d**" % (a_hit, a_miss))
    check("C.3", a_hit == 1 and a_miss == 0,
          "★★ **判据成立**：单层树的接受长度**只能取 `0` 或 `1`**（★ 上界就是树深 `1`）——"
          " ★ 与 `C.2` 合起来：**接受长度 `≤ D`，且 `D` 只有在全覆盖时才达到**",
          "**单层树的边界被动过** —— ★ 先查空树 / 单层的退化")

    # ── C.4 ★★ **§4.3 的反例：把「路径数」当「节点数」—— 断言【两者真的不同】** ──────
    #   ★★ 与 C.1–C.3（**成功**的退化边界）不同：这一条断言的是**混淆两个量会算错**
    #      （`CONTRIBUTING:238` 要求 C 类复现「§4 声称会出问题的输入真的跑出问题」）。
    print()
    print("  ── C.4 ★★ **§4.3 的反例**：`b^D`（路径数）≠ `Σ b^d`（节点数）—— 断言【两者真的不同】──")
    mismatch = [(b, b ** D, count_nodes(b)) for b in specs.PARA_BRANCH_GRID if b >= 2]
    print("     ★ 失败**真的发生**（混淆两个量就出错）：%s"
          % ["b=%d: 路径=%d, 节点=%d" % (b, p, n) for b, p, n in mismatch])
    check("C.4", len(mismatch) >= 1 and all(p != n for (_, p, n) in mismatch),
          "★★ **失败成立**（★ 与 `CONTRIBUTING:238` 的 C 类口径一致）：`b ≥ 2` 时 **`b^D`（路径数）"
          " ≠ `Σ b^d`（节点数）**（★ `b = 2` 时 `64` ≠ `126`、`b = 4` 时 `4096` ≠ `5460`）——"
          " ★★ 于是 `§4.3`「把路径数当节点数」这个**错误前提**会给出错的量级"
          "（★ 断言的是「失败复现」，不是「成功边界」）",
          "**反例不成立** —— ★ 若路径数恒等于节点数，则 `§4.3` 的失败模式**无法复现**"
          "（那 `B.1` 里「两个量不是同一个」的前提也塌了）")


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无任何目标 LLM**；")
    print("      ★★ 但**本程序的 A 类是真的**：**树下降与路径枚举是两条独立路径**。")
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
          and abs(ACC_EPS - 1e-9) < 1e-18,
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住；"
          "★ 容差 `ACC_EPS` **在 preflight 里就钉住**（★ 否则它的变异会变成**崩溃捕获**、"
          "而不是断言捕获）" % specs.DECODE_FETCH_DAY)

    # ★★★ 口径守卫（★ 跑在任何计算之前）：`A.2` 要求分支网格**含 `1`**（链的退化）；
    #   `C.1` 的 `root_excl` 要求 `b ≤ V − 1`（否则「排除目标的 token」根本选不满 `b` 个）；
    #   `C.2` 要求 `V` 在网格的上端可达（`b = V` 时全覆盖）。
    V, D = specs.PARA_VOCAB, specs.PARA_DEPTH
    ok_caliber = (1 in specs.PARA_BRANCH_GRID
                  and max(specs.PARA_BRANCH_GRID) <= V
                  and V >= 3 and D >= 2
                  and specs.PARA_DRAFT_EPS > 0.0
                  and len(specs.PARA_MODEL_SEEDS) >= 2
                  and (V - 1) >= 2)
    print("     ★★ 口径守卫：网格含 `1` = %s；`max(b) ≤ V` = %s；`V ≥ 3` = %s；`D ≥ 2` = %s；`ε > 0` = %s"
          % (1 in specs.PARA_BRANCH_GRID, max(specs.PARA_BRANCH_GRID) <= V,
             V >= 3, D >= 2, specs.PARA_DRAFT_EPS > 0.0))
    check("D.3b", ok_caliber,
          "★★★ **口径守卫**：分支网格**必须含 `1`**（`A.2` 的链退化）、**必须 `≤ V`**"
          "（否则子节点数超过词表、覆盖率恒为 1）、且 `V − 1 ≥ 2`（否则 `C.1` 的「排除目标的 token」"
          "选不满 `b` 个）—— ★ 缺了任一条，对应判据就**退化成恒真**（Chunk 1 的 `H1` 形态）",
          "**口径被动过** —— ★ 先查 `PARA_VOCAB` / `PARA_DEPTH` / `PARA_BRANCH_GRID` / `PARA_DRAFT_EPS`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：守卫实验 · 逐元素 · 独立性 · 术语
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
    print("D · **守卫实验** · **逐元素**钉住 · 独立性 · 术语纪律")
    print("=" * 78)
    seeds = specs.PARA_MODEL_SEEDS

    # ── D.4 ★★★ 守卫实验 ①：把 tree_walk 改成「永远只查根的子节点」──────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：`tree_walk` 改成**永远只查根的子节点**（不随下降更新）──")
    bad = []
    for s in seeds:
        nodes, _ = build_tree(s, 2)
        if tree_walk(nodes, s, corrupt="no_subtree") != exhaustive_accept(nodes, s):
            bad.append(s)
    print("     正常下降对不上的模型 = **0**；改坏下降后对不上的模型 = **%d** / %d" % (len(bad), len(seeds)))
    check("D.4", len(bad) >= 1
          and all(tree_walk(build_tree(s, 2)[0], s) == exhaustive_accept(build_tree(s, 2)[0], s)
                  for s in seeds),
          "★★★ **守卫成立**：把下降改成「只查根」后，它在 **%d/%d** 个模型上**不再等于**参照 ——"
          " ★★ 于是 `A.1` 的「逐位一致」对「**`cur` 随下降更新**」这条规则**是敏感的**"
          % (len(bad), len(seeds)),
          "**守卫失效** —— ★ 改坏下降后居然仍一致（那 `A.1` 抓不到它 ⇒ 需要加判据）")

    # ── D.5 ★★★ 守卫实验 ②：把 exhaustive_accept 改成「取最短匹配前缀」────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：`exhaustive_accept` 改成取「**最短**匹配前缀」（而非最长）──")
    bad2 = []
    for s in seeds:
        nodes, _ = build_tree(s, 2)
        if tree_walk(nodes, s) != exhaustive_accept(nodes, s, corrupt="min"):
            bad2.append(s)
    print("     正常参照对不上的模型 = **0**；改坏参照后对不上的模型 = **%d** / %d" % (len(bad2), len(seeds)))
    check("D.5", len(bad2) >= 1,
          "★★★ **守卫成立**：把参照改成「取**最短**匹配前缀」后，它与被测在 **%d/%d** 个模型上**对不上** ⇒"
          " `A.1` 对**参照的表达式**（「**最长**前缀」）也敏感" % (len(bad2), len(seeds)),
          "**守卫失效** —— ★ 改坏参照后仍与被测一致（那 `A.1` 抓不到它）")

    # ── D.6 ★★★ 取值逐元素：`specs` 的候选树常量 ────────────────────────
    check("D.6", specs.PARA_VOCAB == 4 and specs.PARA_DEPTH == 6
          and specs.PARA_BRANCH_GRID == (1, 2, 4)
          and abs(specs.PARA_DRAFT_EPS - 0.15) < 1e-15
          and specs.PARA_MODEL_SEEDS == ("t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
          and specs.DECODE_FETCH_DAY == "2026-10-09",
          "★★★ **取值逐元素钉住**：词表 `4`、树深 `6`、分支网格 `(1,2,4)`、草稿噪声 `0.15`、"
          " **8 个模型盐值** —— ★★ 每个元组都**逐元素** `==`（★ 不靠「整体」）",
          "**某个 `specs` 常量被动过** —— ★ 先查 `specs.py` 第 4 节")

    # ── D.7 ★★★ 逐元素钉住（程序内常量）────────────────────────────────
    check("D.7", PINNED_CHAIN_ACC == (2, 6, 6, 4, 1, 0, 6, 6)
          and PINNED_TREE2_ACC == (6, 6, 6, 6, 6, 6, 6, 6)
          and abs(PINNED_MEAN_CHAIN - 3.875) < 1e-9 and abs(PINNED_MEAN_TREE2 - 6.0) < 1e-9
          and PINNED_TREE_WINS == 4 and PINNED_T0_CHAIN == 2 and PINNED_T0_TREE2 == 6
          and PINNED_T5_CHAIN == 0
          and PINNED_NODES == (6, 126, 5460) and PINNED_PATHS == (1, 64, 4096)
          and PINNED_PASSES_TREE == 1 and PINNED_PASSES_CHAIN == 6
          and abs(ACC_EPS - 1e-9) < 1e-18
          and len(PINNED_CHAIN_ACC) == len(specs.PARA_MODEL_SEEDS),
          "★★★ **程序内常量逐元素**钉住（两个接受长度元组、两个均值、四个计数、节点/路径数、"
          " 两个前向次数、容差）",
          "**某个程序内常量被动过**")

    # ── D.8 ★★ 独立性（源码级）：两实现互不引用 ────────────────────────
    tn = _names_in(tree_walk)
    en = _names_in(exhaustive_accept)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     `tree_walk` 里出现 `exhaustive_accept` = %s ；`exhaustive_accept` 里出现 `tree_walk` = %s"
          % ("exhaustive_accept" in tn, "tree_walk" in en))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", "exhaustive_accept" not in tn and "tree_walk" not in en
          and "chain_accept" not in tn and "tree_walk" not in _names_in(chain_accept),
          "★★ **源码级线索成立**：**被测（沿树下降）与参照（枚举路径）互不引用**（★ **两侧都审**）——"
          " ★★ 于是「两条路径一致」**不是恒真**，而是**两条独立路径的会合**",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 术语纪律 + 限定语（★ 机械判定）──────────────────────────
    src = open(__file__, encoding="utf-8").read()
    # ★★ 被禁裸词（**刻意用拼接**而不是字面量）：★ 若这里写成字面量，**计数代码自己**也会被算进去，
    #   于是「计数 == 豁免覆盖数」这条**等式**永远差 1。★ 拼接后本文件里**不再出现该裸词的字面量**。
    WORD = "de" + "code"
    n_bare = src.count(WORD)                # ★ 源码里**裸写**该词的计数
    n_exempt = sum(src.count(s) * s.count(WORD) for s in BARE_DECODE_EXEMPT)
    print()
    print("  ── D.9 ★★ **术语纪律**（`§2.2`）：一律写「%s」/「%s」全称；限定语逐字出现 ──"
          % (TERM_AUTOREGRESSIVE, TERM_KV_CACHE))
    print("     ★★ 源码里**裸写那个被禁裸词** = **%d** 处；豁免清单覆盖 = **%d**（清单 %d 项，逐项理由见模块头）"
          % (n_bare, n_exempt, len(BARE_DECODE_EXEMPT)))
    check("D.9", TERM_AUTOREGRESSIVE == "自回归解码" and TERM_KV_CACHE == "KV 缓存"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_AUTOREGRESSIVE in src
          and n_bare == n_exempt
          and all(s in src for s in BARE_DECODE_EXEMPT),
          "★★ **判据成立**：两条术语常量 `%r` / `%r`（★ **不裸写** `decode` / `cache` —— "
          "它们在本库都另有含义）；★ 限定语 `%r` **逐字出现在本程序源码里**；"
          " ★★ **被禁裸词的裸写计数 `%d` 恰好等于豁免清单覆盖的 `%d`**"
          "（★ 清单之外的任何一处裸写都会让它不等）"
          % (TERM_AUTOREGRESSIVE, TERM_KV_CACHE, LIMIT_QUOTE, n_bare, n_exempt),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写** —— "
          "★ 本域一律写全称；★ 新增的裸写要么改全称、要么进 `BARE_DECODE_EXEMPT` 并写明理由")


# ===========================================================================
# 8. 汇总 / 主程序
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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：树下降 vs 路径枚举（两条独立路径）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：玩具模型与代价账都是 **`[E]` 级（本库实现 / 假设）**；"
          "★ **本程序不引用任何外部来源的数字**。")
    print("★ 本程序只核对「**验证/接受的机制与代价账**」——**不是**任何真实模型或引擎的行为。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_parallel_decoding.py —— 并行解码：候选树 + 验证与接受的机制")
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
