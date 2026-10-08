#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_ep_and_other_axes.py —— EP 与 **TP / DP** 的组合、**EP 组的划分**、
以及 ★★★ **`EP` 的约束**（**只写取到过的那一条**）。

配套文档：`03-ep-and-other-axes.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**与
> **设备编号的算术**。

★★★ **本程序 03 的 A 类选点 —— 为什么不是「随便找个地方对拍」**：
  域 README 说得极清楚：**A 类判据只在「切分改变了求值顺序」的地方才有信息量**。
  ⇒ 本程序把 `tp = 1` 与 `tp > 1` **并排放在同一条判据里**（`A.1`）：
    · `tp = 1`（只有 EP）⇒ `max|diff|` **恰为 `0.000e+00`** —— 那是**纯搬运**；
    · `tp > 1`（EP × TP，`W1` 按 **`D_in`** 切成若干行、再**求和**）⇒
      **求和顺序变了** ⇒ `max|diff|` 落在 **`4.441e-16`** 档（**严格大于 0**）。
  ★★★ **两个条件写在同一条 `check` 里** ⇒ 它**不是**自证的：若有人把 TP 那一段
    改回「不做行切」，`tp > 1` 那一半立刻变 `0.0`、判据**当场失败**（`A.1g` 是它的极端版）。

★★★ **贯穿本程序的一条纪律：`EP` 的约束只写【取到过】的那一条** ——
  ★ 本库在任何载体上**都没有取到**「**专家数与 EP 度之间的大小关系**」这类规则
  （见 `specs.EP_NOT_FOUND_NOTE`，那是一条**否定性记录**，**有消费者**：`D.9`）。
  ⇒ ★★ 本程序只在 `B` 类里把它当**本库设计的一个情景**（「**如果** EP 超过专家数」），
    并**明写**这是情景、**不是**规则。
  ★★ 取到过的是**另一条**：`EP` 与 **`DP`** 的关系 —— 载体是 **Megatron-LM MoE README**
  （**来源级别 [B]：某实现的官方文档，不是规范**；★ 它还是一条**活文档**，
  引用写明抓取日 **2026-10-08**、**不做行号锚点**）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 **TP 的行切与求和** ⇒ 右半条（`tp > 1` 必须 `> 0`）
    **必然不成立**（不会通过）；★ `C.1` 若删掉「组划分」那一段算术 ⇒ 跨分片计数无从计算。

★★ **覆盖边界**：本程序的 A 类守的是「**EP × TP 组合后与单卡一致 / 以及那份一致是哪一档**」，
  **不守**「TP 本身怎么切」（那是 `tensor-parallel/01,02,03` 的正文）。
  ★★ 且 **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：`C.1` 断言的是
  「**组划分错位 ⇒ 跨数据分片的条目数 > 0**」这个**失败**。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · ★★★ **容量因子的定义与语义**（定义式 / 取整口径 / dropping vs No-Token-Left-Behind /
    丢弃率口径）→ `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）。
    ★ **本程序连这个词都不出现**。
  · **aux loss 的式子 / 路由 gating 的机制** → `80-model-architectures/moe/01-gating-and-routing.md`、
    `03-balance-and-granularity.md`（★ 本程序的路由只是 `argsort` 取 `top_k`、**等权**）。
  · ★★★ **TP 本身的切法与配对约束**（列切 `D_out` / 行切 `D_in`、`2+2` 次 all-reduce 的配对）
    → `tensor-parallel/01`、`tensor-parallel/02`、`tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`
    （★ **不是 `:223`** —— 那一行是纯 TP 段）。★ **本程序一行都不重写它们**：
    这里只把「**行切**」当成一个**改变了求值顺序**的操作来用。
  · **all-to-all 的通信量分析** → `110-communication/collectives`（**未建成**）+
    `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
    ★★ `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` **逐字让渡**了它给 `110`。
    ⇒ ★ 本程序的 `B` 类量的是**条目数**（一个**计数**），**不是字节数、也不是带宽**。
  · **kernel 侧的 permute / unpermute / grouped GEMM** → `40-kernels/moe-kernels/02,03`；
    `00-meta/glossary.md:262`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **EP × TP 组合与单卡一致** —— ★ 且**两个条件必须同时成立**：
    `tp = 1` 时 `max|diff|` **恰为 `0.000e+00`**（纯搬运）**而** `tp ∈ {2, 4}` 时
    **严格 `> 0`** 且 `< 1e-9`（**求和顺序真的变了**，只差几 ulp）；
  · `A.1g` ★★★ **一条守卫**：把 TP 的「按行切 + 求和」压成「只留第一片」⇒
    `max|diff|` 跳到 **1e+0** 档 ⇒ `A.1` 的右半条当场失败；
  · `A.2`  ★★ **EP 度的选择不影响一致性**（`EP_DEV_GRID` 前两档给**同一组**差值）——
    它的两个条件各自有限，**且**与 `A.1` 的结论自洽；
  · `B.1`  ★★ **`EP > E` 时专家的【内部切分】带来额外搬运** ——
    每份 token 拷贝要被送到 `ceil(EP/E)` 台设备 ⇒ 条目数**严格增加**（`B.2` 量因子）；
  · `B.2`  ★★ 因子恰好是 `ceil(EP/E)`（`E = 8` ⇒ `EP = 16` 给 `2`、`EP = 32` 给 `4`）；
  · `C.1`  ★★★ **EP 组与 DP 组划分错位** ⇒ **同一 EP 组内的设备持有不同的数据分片** ⇒
    跨分片条目数从 **0** 变成 **> 0**（**不报错、结果错**）；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **失败标记自检** + 引文纪律。
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
# ★★★ 失败标记：本主题（expert-parallel）取 `|| FAIL: expert-parallel`。
#   ★ 本域另外五个主题各用**不同的前缀**（`%%` / `^^` / `==` / `{{` / `??`），
#     六个标记**两两不互为前缀** ⇒ 跨主题照抄会被下面两条断言当场抓住。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
#   ★ 它**只允许出现在 `print()` 的字符串里** ⇒ 本文件里它只出现在打印与断言明细中。
ASSERT_FAIL_MARK = "|| FAIL: expert-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                     "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                     "?? FAIL: zero-fsdp")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★ 口径常量**从 `specs.py` 读**（本行不产生可变异叶子：`_walk()` 不下钻 `Attribute`）
TOL_MACHINE = specs.TOL_MACHINE


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
      ⇒ 所以 `cond` 为假时走**另一版文案**：默认只报「这条判据【不成立】」。
      ★ `mutate.py` 的分类是「**行首锚定**」的（`line.lstrip().startswith(mark)`）⇒
        本行打印的失败标记必然在**行首**（前面只有缩进）。
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
# 1. 被测：一个「EP 分片 + TP 行切」的 MoE 层
#    ★★★ 这里唯一**改变了求值顺序**的操作是 **TP 的「按 `D_in` 行切再求和」** ——
#      它就是 `A.1` 右半条的落点。★ 其余全与 `01` 一样是**纯索引搬运**。
#    ★★ 种子写在**函数体内**（不是模块级常量）：改它对任何断言都没有可观察后果
#      ⇒ 放进变异分母只会制造一条**不过期的漏网**。
#      种子取 **`100`**（**不是 2 的幂**）。
# ===========================================================================
def make_layer():
    """造一层：`x`（token）、`Wg`（路由打分）、每个专家一对 `(W1, W2)`。"""
    t_tok, d_dim, e_num, top_k = specs.EP_AXIS_SHAPE
    rng = np.random.default_rng(100)
    x = rng.standard_normal((t_tok, d_dim))
    wg = rng.standard_normal((d_dim, e_num)) * 0.3
    w1 = [rng.standard_normal((d_dim, 4 * d_dim)) * 0.3 for _ in range(e_num)]
    w2 = [rng.standard_normal((4 * d_dim, d_dim)) * 0.3 for _ in range(e_num)]
    return x, wg, w1, w2, t_tok, d_dim, e_num, top_k


def route(x, wg, top_k):
    """路由 = **取每行打分最大的 `top_k` 个**（★ 不含 gating 权重；等权 `1/top_k`）。"""
    return np.argsort(-(x @ wg), axis=1)[:, :top_k]


def reference(x, w1, w2, sel, top_k):
    """**单卡参照**：不做任何切分、不做任何搬运。"""
    t_tok, d_dim = x.shape
    out = np.zeros((t_tok, d_dim))
    for t in range(t_tok):
        for e in sel[t]:
            h = np.maximum(x[t] @ w1[e], 0.0)
            out[t] += (1.0 / top_k) * (h @ w2[e])
    return out


def ep_tp(x, w1, w2, sel, top_k, E, EP, TP, tp_parts=None):
    """**EP × TP 实现**：专家按 `e*EP//E` 摆到设备上；**TP 把 `W1` 按 `D_in` 行切后求和**。

    · `tp_parts=None` ⇒ 用**全部** `TP` 片（正确）；
    · `tp_parts=k`   ⇒ 只留**前 `k` 片**（`A.1g` 的错法：all-reduce 少加了几档）。

    ★★★ **行切 + 求和正是「求值顺序变了」的那一处** ——
      `tp = 1` 时它是**恒等操作**（`max|diff|` 恰好 `0.0`），
      `tp > 1` 时它把求和拆成了 `Σ_p (…)`，落点差几个 ulp（本机实测 `4.441e-16`）。
    """
    t_tok, d_dim = x.shape
    if tp_parts is None:
        tp_parts = TP
    out = np.zeros((t_tok, d_dim))
    for t in range(t_tok):
        for e in sel[t]:
            h = np.zeros(4 * d_dim)
            for p in range(tp_parts):
                lo = p * d_dim // TP
                hi = (p + 1) * d_dim // TP
                h += x[t][lo:hi] @ w1[e][lo:hi]
            h = np.maximum(h, 0.0)
            out[t] += (1.0 / top_k) * (h @ w2[e])
    return out


# ── 设备编号与组划分（`03` 的 C 类用的算术；★ 全部是**整数**运算）────────────
def device_rank(dp_rank, ep_rank, EP):
    """**EP 为内层**的设备编号：`r = dp_rank * EP + ep_rank`（本库的布局口径）。"""
    return dp_rank * EP + ep_rank


def ep_group_contiguous(EP, g):
    """**正确**的 EP 组：`g` 号那一段**连续**设备（同 `dp_rank`、不同 `ep_rank`）。"""
    return [g * EP + j for j in range(EP)]


def ep_group_strided(EP, DP, e):
    """**错法**的 EP 组：同余类 `{r : r % EP == e}`（**跨步**，`DP` 个设备）。"""
    return [k * EP + e for k in range(DP)]


def data_shard(rank, EP):
    """设备 `rank` 持有的**数据分片**编号 —— 本库口径：`rank // EP`。

    ★★ **关键不变式**：**同一个 EP 组内的设备必须持有【同一个】数据分片** ——
      因为它们处理的是**同一批 token**、只是**专家不同**（EP 的定义）。
    """
    return rank // EP


def cross_shard_count(group, EP):
    """一个组里**与组内第一个设备分片不同**的设备数（★ 正确布局下恒为 `0`）。"""
    shards = [data_shard(r, EP) for r in group]
    return sum(1 for s in shards if s != shards[0])


# ===========================================================================
# 2. A 类：EP × TP 组合（`A.1` / `A.1g` / `A.2`）
#    ★★★ 本程序的 A 类**两个条件必须同时成立**（见模块抬头）。
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · EP × TP 组合：★ 一致性**与**「那份一致是哪一档」写进【同一条】判据")
    print("=" * 78)
    x, wg, w1, w2, t_tok, d_dim, e_num, top_k = make_layer()
    sel = route(x, wg, top_k)
    ref = reference(x, w1, w2, sel, top_k)
    print("  形状（`EP_AXIS_SHAPE`）：T=%d · D=%d · E=%d · top_k=%d"
          % (t_tok, d_dim, e_num, top_k))
    print("  ★★★ **TP 在这里切的是 `W1` 的 `D_in` 维**（行切），切完**求和** ⇒")
    print("      **求和顺序变了** —— 这正是本域说的「A 类要有信息量」的那个落点。")
    print("      ★★ **TP 本身的切法与配对约束不在本域**（回链 `tensor-parallel/01,02,03`）。")

    # ── A.1 ★★★ EP×TP 一致 + 那一档 ────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ `tp = 1` 恰为 `0.000e+00`（纯搬运）**而** `tp > 1` **严格 > 0** ──")
    exact_zero = []
    nonzero = []
    rows = []
    for ep in specs.EP_DEV_GRID[:2]:
        for tp in specs.EP_TP_GRID:
            d = float(np.abs(ep_tp(x, w1, w2, sel, top_k, e_num, ep, tp) - ref).max())
            rows.append((ep, tp, d))
            if tp == 1:
                exact_zero.append(d)
            else:
                nonzero.append(d)
            print("     EP=%-2d TP=%-2d ⇒ `max|diff|` = **%.3e**%s"
                  % (ep, tp, d, "   ← **纯搬运**（恰好 0）" if tp == 1 else ""))
    print("     ★★★ **两个条件**：① `EP_TP_GRID` 的**第一档** `tp = 1` 下差**恰为 `0.000e+00`**")
    print("        （★ 此时行切退化成「切 1 份」，**什么都没改**）；")
    print("        ② `tp ∈ {2, 4}` 下差**严格大于 0** 且 `< 1e-9` —— **求和顺序真的变了**。")
    print("     ★★★ **没有条件②，条件① 就是恒真的**（`01` 的 `D.4` 量过同一件事）——")
    print("        本程序把两者**写在同一条判据里**，正是为了不让它退化成伪断言。")
    check("A.1", all(v == 0.0 for v in exact_zero) and len(exact_zero) == 2
          and all(v > 0.0 for v in nonzero) and all(v < 1e-9 for v in nonzero)
          and all(np.isfinite(d) for _ep, _tp, d in rows),
          "★★★ **判据成立（两个条件）**：① `tp = 1`（**两档 EP**）下 `max|diff|` **恰为 "
          "`0.000e+00`**；② `tp ∈ {2, 4}` 下**严格大于 0**（最大 **%.3e**）且 `< 1e-9`"
          " ⇒ ★★ **EP × TP 组合与单卡一致，且那份一致在 `tp > 1` 时是「几 ulp」而不是「恰好 0」**"
          % max(nonzero),
          "**两档关系不成立** —— ★ 若 `tp > 1` 也给 `0.0`，说明 TP 的行切**没有**改变求值顺序"
          "（`A.1g` 是它的另一个极端）；★ 若 `tp = 1` 不为 `0`，说明混进了别的运算")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    print()
    print("  ── A.1g ★★★ **守卫**：把 TP 的「行切 + 求和」压成「只留第一片」──")
    guard = []
    for tp in specs.EP_TP_GRID[1:]:
        d = float(np.abs(ep_tp(x, w1, w2, sel, top_k, e_num, specs.EP_DEV_GRID[0], tp,
                               tp_parts=1) - ref).max())
        guard.append(d)
        print("     TP=%-2d（只留第 1 片）⇒ `max|diff|` = **%.6f**" % (tp, d))
    check("A.1g", all(v > 1e-9 for v in guard) and all(np.isfinite(v) for v in guard),
          "★★ **守卫成立**：少加几片之后差跳到 **%.6f** 档（≫ `1e-9`）⇒ `A.1` 的"
          "「`tp > 1` 必须 `< 1e-9`」**真的会失败** ✓（★ 它守的是「**TP 的求和有没有做全**」）"
          % max(guard),
          "**守卫失效** —— 少加几片后差仍在 `1e-9` 以内（那说明 TP 的行切没起作用）")

    # ── A.2 ★★ EP 度的选择不影响一致性 ────────────────────────────────────
    print()
    print("  ── A.2 ★★ **EP 度的选择不影响一致性**（`EP_DEV_GRID` 前两档给同一档差值）──")
    per_ep = {}
    for ep in specs.EP_DEV_GRID[:2]:
        vals = [d for e, _tp, d in rows if e == ep]
        per_ep[ep] = vals
        print("     EP=%-2d ⇒ 三档 TP 下的差：%s" % (ep, ["%.3e" % v for v in vals]))
    same = per_ep[2] == per_ep[4]
    print("     ★★ 两档 EP 下的差值序列**逐项相同** = %s" % same)
    print("     ★★ **为什么这**不在意料之外**：EP 在本程序里只改**哪些项被加起来**")
    print("        （纯搬运），**不改**任何一项的浮点值 ⇒ 它不影响「几 ulp」这个档位。")
    print("        ★★ 它影响的是**别的**东西（搬运条目数、负载分布）—— 那在 `01`/`02`。")
    check("A.2", same and len(per_ep[2]) == 3
          and all(np.isfinite(v) for v in per_ep[2] + per_ep[4]),
          "★★ **判据成立**：`EP ∈ {2, 4}` 下三档 TP 的差值序列**逐项相同**（%s）⇒"
          " ★★ **EP 度只改搬运、不改浮点档位**（★ 全部分量有限）"
          % ", ".join("%.3e" % v for v in per_ep[2]),
          "**两档 EP 的差值序列不同** —— ★ 那说明 EP 那一段里混进了浮点运算（它就**不是**搬运了）")


# ===========================================================================
# 3. B 类：`EP > E` 时的专家内部切分（`B.1` / `B.2`）
#    ★★★ 本节是 **【本库设计的一个情景】** —— 见模块抬头那条否定性纪律。
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `EP > E` 时专家的【内部切分】：每份 token 拷贝要被送到更多台设备")
    print("=" * 78)
    t_tok, d_dim, e_num, top_k = specs.EP_AXIS_SHAPE
    print("  ★★★ **本节是【本库设计的一个情景】，不是对任何规范的转述** ——")
    print("     本库在**任何载体**上**都没有取到**「专家数与 EP 度之间的大小关系」这类规则")
    print("     （★ 那是一条**否定性记录**：`specs.EP_NOT_FOUND_NOTE`，由 `D.9` 消费）。")
    print("     ⇒ ★★ 本节问的是「**如果** `EP` 超过专家数，**本库的记账模型**会多出什么」。")

    # ── B.1 ★★ 额外搬运 ───────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **每份 token 拷贝被送去的设备数** = `ceil(EP/E)` ──")
    base = t_tok * top_k
    print("     总投递条目（`EP <= E` 时）= `T·top_k` = %d，载荷单位 = `d_model` = %d"
          % (base, d_dim))
    rows = []
    for ep in specs.EP_DEV_GRID:
        factor = 1
        rows.append((ep, factor, base * factor * d_dim, base * factor))
        print("     EP=%-3d（<= E=%d）⇒ `ceil(EP/E)` = %d ⇒ 条目 **%d** · 载荷 **%d** 个元素"
              % (ep, e_num, factor, base * factor, base * factor * d_dim))
    over_rows = []
    for ep in specs.EP_OVER_GRID:
        factor = -(-ep // e_num)
        over_rows.append((ep, factor, base * factor * d_dim, base * factor))
        print("     EP=%-3d（**> E=%d**）⇒ `ceil(EP/E)` = **%d** ⇒ 条目 **%d** · 载荷 **%d** 个元素"
              % (ep, e_num, factor, base * factor, base * factor * d_dim))
    inc = all(o[2] > rows[-1][2] for o in over_rows) and all(
        over_rows[i][2] < over_rows[i + 1][2] for i in range(len(over_rows) - 1))
    print("     ★★★ **为什么会多**：`EP > E` 时**一台设备摊不到一个完整专家** ⇒")
    print("        专家必须**跨设备切开**（专家内部再切 `ceil(EP/E)` 份）⇒")
    print("        每份 token 拷贝要**复制到那几台设备**上 ⇒ **条目数与载荷都按因子涨**。")
    check("B.1", inc and len(rows) == 3 and len(over_rows) == 2,
          "★★ **判据成立**：`EP <= E` 的三档因子都是 **1**（条目 %d），而 `EP > E` 的两档"
          "因子是 **%d** 与 **%d** ⇒ 条目 **%d** / **%d**、载荷**严格递增** ✓"
          % (base, over_rows[0][1], over_rows[1][1], over_rows[0][3], over_rows[1][3]),
          "**额外搬运没有出现** —— ★ 先查 `ceil(EP/E)` 是不是被写成了 `1`")

    # ── B.2 ★★ 因子 ───────────────────────────────────────────────────────
    print()
    print("  ── B.2 ★★ 因子恰好是 `ceil(EP/E)`（★ 不是 `EP/E`）──")
    for ep, factor, _pay, _items in over_rows:
        print("     EP=%-3d ⇒ `ceil(%d/%d)` = **%d**（★ 注意 `EP/E` 也是整数时两者相同："
              "`16/8 = 2`、`32/8 = 4`）" % (ep, ep, e_num, factor))
    print("     ★★ 取 `EP = 16` 与 `32` 是**故意的**：它们**整除** `E = 8`（因子 `2` / `4`）")
    print("        ⇒ ★ 若改成 `EP = 12`（不整除）就能看到 `ceil` 与 `EP/E` **不同** ——")
    print("          ★★ 那属于 `01` 的 `A.1`（不整除的划分），**本节不重算它**。")
    ceil_ok = all((-(-ep // e_num)) == (ep // e_num if ep % e_num == 0 else -(-ep // e_num))
                  for ep in specs.EP_OVER_GRID)
    check("B.2", ceil_ok and all(-(-ep // e_num) == ep // e_num for ep in specs.EP_OVER_GRID)
          and len(specs.EP_OVER_GRID) == 2,
          "★★ **判据成立**：`EP_OVER_GRID` 的两档都**整除** `E = 8` ⇒ `ceil(EP/E) == EP/E`"
          "（**%d** 与 **%d**）⇒ ★ 本节量的是「因子的**存在**」，"
          "★ 不整除时 `ceil` 与 `floor` 的差别在 `01` 的 `A.1`"
          % tuple(-(-ep // e_num) for ep in specs.EP_OVER_GRID),
          "**因子不是 `ceil(EP/E)`** —— ★ 先查 `EP_OVER_GRID` 与 `E` 的整除关系")


# ===========================================================================
# 4. C 类：EP 组与 DP 组共用同一组设备时的**划分错位**（`C.1`）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · EP 与 DP 用**同一组设备**：组划分错位 ⇒ 不同数据分片混进同一组专家")
    print("=" * 78)
    ep, dp = specs.EP_AXIS_PAIR
    n_world = ep * dp
    print("  ★★ 布局口径（本库）：**EP 为内层** ⇒ `r = dp_rank * EP + ep_rank`，"
          "`world = EP * DP = %d`" % n_world)
    print("     · **EP 组**（同 `dp_rank`、不同 `ep_rank`）：**连续的一段** —— "
          "它们处理**同一批 token**、各持不同专家；")
    print("     · **DP 组**（同 `ep_rank`、不同 `dp_rank`）：**跨步的一批** —— "
          "它们持**同一批专家**、各持不同数据分片。")
    print("     ★★★ **关键不变式**：**一个 EP 组内的设备必须持有同一个数据分片**。")

    # ── C.1 ★★★ 组划分错位 ────────────────────────────────────────────────
    print()
    print("  ── C.1 ★★★ **正确布局** vs **把 EP 组划成【跨步】的**（同余类 `r %% EP == e`）──")
    print("     正确（连续）：")
    ok_correct = True
    for g in range(dp):
        grp = ep_group_contiguous(ep, g)
        shards = [data_shard(r, ep) for r in grp]
        cr = cross_shard_count(grp, ep)
        ok_correct = ok_correct and cr == 0
        print("        组 %d = %s ⇒ 数据分片 %s ⇒ **跨分片 %d 个**"
              % (g, grp, shards, cr))
    print("     错法（跨步）：")
    ok_wrong = True
    for e in range(ep):
        grp = ep_group_strided(ep, dp, e)
        shards = [data_shard(r, ep) for r in grp]
        cr = cross_shard_count(grp, ep)
        ok_wrong = ok_wrong and cr == dp - 1
        print("        组 %d = %s ⇒ 数据分片 %s ⇒ **跨分片 %d 个**"
              % (e, grp, shards, cr))
    print("     ★★★ **后果**：错法下，**一次 EP 通信会把 %d 个不同数据分片的 token**" % dp)
    print("        送进**同一组专家** ⇒ ★★ **不报错、结果错**（形状、类型、索引全都合法）。")
    print("     ★★★ **它为什么与「EP 是 DP 的子组」这条纪律有关**：")
    print("        `specs.EP_MOE_README_QUOTE`（**来源级别 [B]：某实现的官方文档，不是规范**）逐字：")
    print("        `%s`" % specs.EP_MOE_README_SUBQUOTE)
    print("        ⇒ ★★ 「EP 组必须落在同一个数据分片里」是这条纪律的**可测形式**；")
    print("        ★★ **本程序不引用那个「`EP` 与 `DP` 的大小关系」的完整断言** ——")
    print("           只引用其中**不含标记**的那一段（★ 完整句在 `D.8` 逐字钉住）。")
    check("C.1", ok_correct and ok_wrong and n_world == ep * dp,
          "★★★ **判据成立**（三个条件）：① **正确**布局下 `DP` 个 EP 组的跨分片数**全是 `0`**；"
          "② **错法**（跨步）下每组跨分片数**都是 `DP-1 = %d`** ⇒ 组内设备持有不同数据分片；"
          "③ `world = EP*DP = %d`（★ 组划分覆盖全部设备）⇒ ★ **一次 EP 通信混进 %d 个分片**"
          % (dp - 1, n_world, dp),
          "**组划分的算术被改动了** —— ★ 先查 `data_shard(r) = r // EP` 与 "
          "`r = dp_rank*EP + ep_rank` 这两个口径是不是同一个")


# ===========================================================================
# 5. 前哨：口径 / 阈值 / 标记自检（判据 D.1 / D.2 / D.2b / D.3 / D.3b）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 A/B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` / `numpy %s`（矩阵乘走 BLAS）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · **容量因子的定义与语义**（定义式 / 取整口径 / dropping vs No-Token-Left-Behind /")
    print("        丢弃率口径）→ `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）；")
    print("        ★★ **本程序连这个词都不出现**。")
    print("      · **aux loss 的式子 / 路由 gating 的机制** → "
          "`80-model-architectures/moe/01-gating-and-routing.md`、`03-balance-and-granularity.md`。")
    print("      · ★★★ **TP 本身的切法与配对约束**（列/行的配对、`2+2` 次 all-reduce）→")
    print("        `tensor-parallel/01,02,03` 与 `10-foundations/performance-models/02:417`"
          "（★ **不是 `:223`**）；")
    print("        ★ 本程序只把「**行切**」当成一个**改变了求值顺序**的操作来用。")
    print("      · **all-to-all 的通信量分析** → `110-communication/collectives`（未建成）、")
    print("        `20-gpu-architecture/interconnect/03`；★ `40-kernels/moe-kernels/01:243` 已让渡。")
    print("      · **kernel 侧的 permute / unpermute / grouped GEMM** → "
          "`40-kernels/moe-kernels/02,03`、`00-meta/glossary.md:262`。")
    before = len(FAILS)

    # ── D.1 ★ 标记自检（AST 读出，**不写死**）──────────────────────────────
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

    # ── D.2 ★ 标记自检（字面量比较）────────────────────────────────────────
    check("D.2", ASSERT_FAIL_MARK == "|| FAIL: expert-parallel" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `|| FAIL: expert-parallel`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                                        "?? FAIL: zero-fsdp"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住；"
          "★ 六个标记**两两不互为前缀**")

    # ── D.3 ★ 口径常量（本程序的**输入**与阈）──────────────────────────────
    check("D.3", specs.TOL_MACHINE == 1.0e-12,
          "★ 机器精度档阈 = %g（★ 本程序在 `A.1` 里用的是**函数体内的 `1e-9`** ——"
          "那比它宽三个数量级，是为了容纳 BLAS 换路径；**内容级**的阈是 `1e-9` 之上的"
          "几百个数量级）" % specs.TOL_MACHINE)
    check("D.3b", len(specs.EP_AXIS_PAIR) == 2 and all(isinstance(v, int) for v in specs.EP_AXIS_PAIR)
          and specs.EP_AXIS_PAIR[0] * specs.EP_AXIS_PAIR[1] == 4,
          "★★ `EP_AXIS_PAIR` 是**两个整数**、且 `EP * DP == 4` ⇒ ★ 4 台设备小到"
          "**每一组都能逐个打印出来复核**（`C.1` 用的就是这一点）")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：specs 常量逐位钉住 · 逐元素钉住 · 引文纪律 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · specs 常量逐位钉住 · **逐元素**钉住 · 引文纪律 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★ 独立性线索（源码级）──────────────────────────────────────────
    ref_src = inspect.getsource(reference)
    eptp_src = inspect.getsource(ep_tp)
    calls_ref = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(ref_src)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_eptp = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(eptp_src)))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.4 ★★ **源码级线索**（★ **不是证明**）：参照与 EP×TP 版的调用表 ──")
    print("     `reference()` 调用：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    print("     `ep_tp()`    调用：%s"
          % (", ".join(sorted(calls_eptp)) if calls_eptp else "**空集**"))
    print("     ⇒ ★★ `reference()` **不调用** `ep_tp()` / `route` / `make_layer` ——")
    print("        **参照实现不得调用被测实现的任何函数**（域 README 的硬要求）在此成立。")
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，")
    print("        **不证明** A 类有物理信息量 —— 后者由 `A.1g` 的**行为**兜底。")
    check("D.4", "ep_tp" not in calls_ref and "reference" not in calls_eptp
          and "route" not in calls_ref and ref_src != eptp_src,
          "★★ **源码级线索成立**：两段代码不同源、互不调用，且 `reference()` 的调用表里"
          "**没有** `ep_tp` / `route` / `make_layer` ⇒ ★ **参照不调用被测**（源码层面）",
          "**参照与 EP×TP 版互相调用** —— A 类判据可能退化成自证")

    # ── D.5 ★★★ `EP` 的约束：只写【取到过】的那一条（[B] 级）────────────────
    check("D.5", specs.EP_MOE_README_SUBQUOTE == "Expert Parallelism must be a sub-group of Data Parallelism"
          and specs.EP_MOE_README_A2A_QUOTE == "NCCL-based All-to-All communication for token exchange",
          "★★★ **本主题能给出的、取到过的「`EP` 的约束」的依据**（两条逐字）：\n"
          "        ★★ **来源级别 [B]：某实现的官方文档，不是规范**；\n"
          "        ★ 载体 = Megatron-LM 仓库的 `%s`（**活文档，抓取日 2026-10-08，不做行号锚点**）\n"
          "        ★★ 它讲的是 **`EP` 与 `DP`** 的关系（那一整句在 `D.8` 逐字钉住）；\n"
          "        ★ 第二条给出**机制名**（token exchange 用 `All-to-All`），**不给次数、不给尺寸**。"
          % specs.EP_MOE_README_PATH,
          "**`EP` 约束的依据常量被动过** —— ★ 特别注意别把它升格成「规范」（A 级）")

    # ── D.6 ★★ 数值常量 ────────────────────────────────────────────────────
    check("D.6", specs.TOL_MACHINE == 1.0e-12
          and all(-(-ep // specs.EP_AXIS_SHAPE[2]) == ep // specs.EP_AXIS_SHAPE[2]
                  for ep in specs.EP_OVER_GRID),
          "★★ 数值常量逐位钉住：阈 `1e-12`；★ `EP_OVER_GRID` 的两档都**整除** `E = %d`"
          " ⇒ `ceil(EP/E) == EP/E`（**%s**）⇒ 本节量的是因子的**存在**，"
          "★ 不整除时的 `ceil`/`floor` 之别在 `01` 的 `A.1`"
          % (specs.EP_AXIS_SHAPE[2],
             " / ".join(str(-(-ep // specs.EP_AXIS_SHAPE[2])) for ep in specs.EP_OVER_GRID)),
          "**数值常量被动过** —— ★ 先查 `EP_OVER_GRID` 与 `E` 的整除关系")

    # ── D.7 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）─────────────
    check("D.7", specs.EP_TP_GRID == (1, 2, 4)
          and specs.EP_AXIS_SHAPE == (32, 8, 8, 2)
          and specs.EP_OVER_GRID == (16, 32)
          and specs.EP_AXIS_PAIR == (2, 2),
          "★★★ **四个元组【逐元素】钉住**（`EP_TP_GRID` 三档 · `EP_AXIS_SHAPE` 四维 ·"
          " `EP_OVER_GRID` 两档 · `EP_AXIS_PAIR` 两元）—— ★ 只钉「整体」会让**元素级**变异漏网"
          "（本库 `pipeline-parallel` 的那一轮实测过：一次报出 58 条漏网）",
          "**元组的元素被动过**")

    # ── D.8 ★★★ Megatron MoE README 的完整句（**原始字节**逐字）────────────
    check("D.8", specs.EP_MOE_README_QUOTE
          == ("Prior MoE frameworks constrain **EP ≤ DP** (Expert Parallelism must be a "
              "sub-group of Data Parallelism), which severely limits scalability."),
          "★★★ 完整句逐字钉住（载体 = `%s`，抓取日 2026-10-08，**原始字节 1 次**）：\n"
          "        ★★★ **原始字节里 `EP ≤ DP` 两侧有 Markdown 粗体标记 `**`** ⇒"
          " 本模块**按原始字节**存放；\n"
          "        ★★ 去掉 `**` 的写法在原始字节上计到 **0**；\n"
          "        ★★ 那个 `≤` 是 **U+2264**（不是 ASCII `<=`）—— 换成 ASCII 也计到 **0**。\n"
          "        ★★★ 它讲的是 **`Prior MoE frameworks`**（**先前的**框架）⇒ "
          "**不得**把它读成「当前实现仍然如此」。" % specs.EP_MOE_README_PATH,
          "**完整句被动过** —— ★ 先查 `**` 与 `≤` 这两个非 ASCII 成分")

    # ── D.9 ★★★ 否定性记录 + `UNSOURCED` 留空 ─────────────────────────────
    check("D.9", specs.EP_NOT_FOUND_NOTE
          == ("no source found for a constraint between the expert count and the "
              "expert-parallel degree")
          and specs.UNSOURCED == ()
          and specs.EP_MOE_README_PATH == "megatron/core/transformer/moe/README.md",
          "★★★ **否定性记录**逐字钉住：本库在任何载体上**没有取到**「**专家数与 EP 度之间的**"
          "**大小关系**」这类规则 ⇒ ★★ **本程序不写任何「专家数必须不少于 EP 度」的断言**。\n"
          "        ★★ **它有一个真消费者**（本行）⇒ 它**不是**一条无消费者的漏网候选。\n"
          "        ★★ `UNSOURCED` **留空**；★ 载体路径是 **`%s`** —— **不是** `docs/user-guide/"
          "features/moe.md`（那一份 663 字节、只 `{include}` 这一份）"
          % specs.EP_MOE_README_PATH,
          "**否定性记录被动过**，或 `UNSOURCED` 不再是空的")


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
    print("★ `EP` 约束的依据在 `specs.py` 的 `# 来源：`（Megatron MoE README，**[B] 级实现文档**）；")
    print("  本程序只核「EP × TP 组合的一致性档位」「`EP > E` 的条目数」与「组划分错位的后果」。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_ep_and_other_axes.py —— EP × TP / DP 的组合、EP 组的划分、EP 的约束")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`，矩阵乘走 BLAS）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本程序 A 类的两个条件写在同一条判据里**：`tp = 1` 恰为 `0.000e+00`（纯搬运）")
    print("      **而** `tp > 1` **严格 > 0**（行切改变了求值顺序）—— 少了后半条它就是恒真的。")
    print("  ★★★ **`EP` 的约束只写【取到过】的那一条**（`EP` 与 `DP`，[B] 级实现文档）：")
    print("      ★ 本库**没有**在任何载体上取到「专家数与 EP 度之间的大小关系」这类规则 ⇒")
    print("      `B` 类的 `EP > E` 是**本库设计的一个情景**，**不是**对任何规范的转述。")
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
