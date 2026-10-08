#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_load_imbalance.py —— ★★★ **EP 的负载账**：**发出量均衡、接收量不均**，
以及 ★★ **EP 度越大越不均**。

配套文档：`02-load-imbalance.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序量的是**索引搬运的计数**，
> **没有一次真实通信**。

★★★ **本程序的判据分两类，写法【必须不同】**：
  · `A` 类是**恒等式**（每源设备的**发出量**恒等于 `T·top_k/EP`）——
    它对**每一个**种子、**每一档** EP 都成立 ⇒ ★★ **不需要多样本**，报一个数即可；
  · `B` 类是**统计量**（**接收量**的 `max/min`）—— 它**逐种子不同** ⇒
    ★★★ **判据必须写明「用多个种子」，并给出【范围】**（`50-performance` 的
    「判据的小 n 稳定性」纪律）。★★ **只报一个均值而不报范围 = 不可复核**。
  ⇒ ★★★ 本程序因此**严禁**把 B 类的数写成「EP=8 时是 `1.3963`」这样的**单词**；
    一律写成「**八个种子**（`specs.EP_IMB_SEEDS`）下的**均值与范围**」。

★★★ **【初始分布】必须与数一起报**（本条是本主题最容易被误读的地方）：
  本程序用的是 **`scores ~ N(0,1)` 的 `(T, E)` 矩阵、取每行最大的 `top_k` 个**
  （`EP_IMB_MODEL` = `(256, 8, 2)`）。
  ★★★ **换一种初始分布（例如「均匀随机抽 `top_k` 个专家」）会给出不同的比值** ——
  所以两个「接收不均」的数**只有在初始分布相同时才可比**。
  ★★ 这一点是**实测教训**：同一段代码、同一个 `T/E/top_k`，**只换初始分布**，
  报出的比值就会差出可观的一截 ⇒ ★ 任何「某某处测得 `1.84`」的说法，
  **没有初始分布就不能复核**。

★★★ **贯穿本程序的一条纪律（关于一切「丢弃」的数）—— ★ Chunk 5 已【修正方向】**：
  ★★★ **本程序【自己】不给任何丢弃率数字**：本程序的数是**某一个种子下的【计数】**，
    不是丢弃率 ⇒ ★★ **不得**把它们与任何来源的丢弃率比较
    （口径先在 `80-model-architectures/moe/02-capacity-factor.md` §3.3 对齐）。
  ★★ **但「某来源给过丢弃率经验值」是一条【取到过】的外部事实**（★ 这**不再**是
    「未取到」的否定性断言 —— 原说法是一条**假否定**，见 `specs.py` 抬头）：
    · ★★★ **Switch Transformer §2.2 逐字**（`specs.SWITCH_DROPPED_RATE_QUOTE`，
      散文，**原始字节 1 次**）：`… the number of tokens dropped`；紧接着的 `typically <1%`
      **在 `<math>` 里** ⇒ 单独登记为 `specs.SWITCH_DROPPED_RATE_ALT`
      （**原始字节 2 次**；★ 那 2 次来自**同一个** `<math>` 元素的 `alttext` 与 `x-tex` 注解，
      **不是两处出处**）。
    · ★★ **GShard 侧【确实没有】**：复核（2026-10-08，ar5iv 正文）—— 剥离标签后 `%` **12 处**
      **全在** FLOPS / roofline / 重算占比 / 执行时间占比，`drop` **5 处**全在 `dropout`
      或「**收益下降**」（`this drop in gains`）⇒ **GShard 不给丢弃率**。
  ★★★ **一条必须先说的教训**（与 `specs.py` 抬头同）：**「数不到」与「不存在」是两件事** ——
    本程序曾用 **`capacity`** 当锚点在两篇正文里数 `%`、各得 **0** 次，据此写下
    「本库在任何载体上都没有取到丢弃率数字」—— ★★★ **那句话是假的**：
    Switch 那一句的锚点是 **`dropped tokens`**，而 `capacity` 距它最近的 `%` **694 字节**。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉**路由**（`top_k` 的选取）⇒ 发出量根本无从计算
    （不会通过）；★ `B.1` 若删掉「按设备归并专家」那一步 ⇒ 不均无从谈起；
    ★★ `C.1` 是**内容**判据（删掉上限就无从比）。

★★ **覆盖边界**：本程序的 A 类守的是「**路由的计数守恒**」，
  **不守**「路由本身好不好」（那是 `80-model-architectures/moe/01,03` 的正文）。
  ★★ 且 **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：`C.1` 断言的是
  「**施加一个上限 ⇒ 不均转移成丢弃**」这个**可观测的后果**，不是「实现正确」。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · ★★★ **容量因子的定义与语义**（**定义式** / **取整口径** /
    **dropping vs No-Token-Left-Behind** / **丢弃率口径**）→
    `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）
    —— ★★ **方向是【本域让渡给该篇】**（★ 该篇自己接住的是 `glossary` 与 `moe-kernels`
    的让渡；实测它全篇里 `EP` / `all-to-all` / `并行` / `设备` / `通信` **各 0 次**）。
    ⇒ ★★ **本程序一个字都不写那四样**：这里的「上限」是 `specs.EP_CAP_LIMIT`
    **直接给定的一个整数**（本库设计），**不是**从任何公式算出来的；
    超了**怎么办**（丢 / 重路由）本程序**不表态**，只数「**超过上限的条目数**」。
  · **aux loss 的式子 / 路由 gating 的机制** → `80-model-architectures/moe/01-gating-and-routing.md`、
    `03-balance-and-granularity.md`。★ 本程序的路由只是 `argsort` 取 `top_k`，
    **不含 gating 权重**（★ 与第一代探测 `probe_100f.py` 的口径一致）。
  · **all-to-all 的通信量分析** → `110-communication/collectives`（**未建成**）+
    `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
    ★★ `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` **逐字让渡**了它给 `110`。
    ⇒ ★ 本程序量的是**搬运的条数**（一个**计数**），**不是字节数、也不是带宽**。
  · **kernel 侧的 permute / unpermute / grouped GEMM** → `40-kernels/moe-kernels/02,03`；
    `00-meta/glossary.md:262`。
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（★ **不是 `:223`**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **发出量是【恒等式】**：每个种子 × 每档 EP 下**每源设备的发出量
    恒等于 `T·top_k/EP`** ⇒ `max/min` **恰为 `1.000000`**（★ 不是「大约均衡」）；
  · `A.1g` ★★★ **一条守卫**：把「发出量」换成「**接收量**」⇒ 恒等性**立刻不成立**；
  · `A.2`  ★★★ **「专家→设备」的映射是【连续的一段】**（单调不减 + 与等价的块写法逐元素相等）
    —— ★★ 它是 `B`/`C` 两节**所有数**的基础，而 `A.1` **钉不住它**
    （发出量与「走到哪台设备」无关）⇒ ★ **换成跨步映射 `e % EP` 时本判据当场失败**；
  · `B.1`  ★★★ **接收量的不均随 EP 单调恶化**：`EP_IMB_SEEDS` 的**八个种子**下
    **均值严格递增**，且 **8/8 个种子逐个**都严格递增（★ 报**范围**，不只报均值）；
  · `B.2`  ★★ **成因**：每设备的专家数 `E/EP` 随 EP 严格减少（`4 → 2 → 1`）—— 这就是
    「专家级波动不再被平均掉」的可复核形式；
  · `C.1`  ★★★ **施加一个专家上限 ⇒ 不均【转移】成丢弃**：每设备**收下的**条目数被削平，
    但「**超过上限的条目数**」在设备之间**不均**（至少一台丢、至少一台不丢）；
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
# 1. 被测：一次「按设备归并专家」的负载账（★ 纯计数，无浮点）
#    ★★★ 种子写在**函数体内**（不是模块级常量）—— 见 `specs.EP_IMB_SEEDS` 的说明：
#      本主题的**种子集**本身是判据口径的一部分，所以它**在 `specs.py` 里**；
#      而「怎么用这些种子」在函数体里。
# ===========================================================================
def route_topk(seed, t_tok, e_num, top_k):
    """**路由**：`scores ~ N(0,1)` 的 `(T, E)` 矩阵，取每行最大的 `top_k` 个。

    ★★★ **这就是本程序的【初始分布】** —— 报任何不均的数**必须**带上它。
    ★ 不含 gating 权重（那是 `80-model-architectures/moe/01,03` 的正文）。
    """
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((t_tok, e_num))
    return np.argsort(-scores, axis=1)[:, :top_k]


def device_of_expert(e, E, EP):
    """**目的设备的编号口径**：专家 `e` 落在哪台设备上 = `e * EP // E`（**均摊口径**）。

    ★★ 它与 `01` 的 `device_of_expert` 是**同一个映射**（域级探测脚本口径）。
    ★★★ **本程序 `B`/`C` 两节报出的每一个数都建在它上面** ⇒ `A.2` 专门钉它。
    ★★ 它**总是**给出 `0 <= dev < EP`，且**每设备拿【连续的一段】专家**
      （设备 `d` 拿 `[d*E//EP, (d+1)*E//EP)`）—— ★ 这正是 `A.2` 钉的两个不变量。
    """
    return e * EP // E


def device_of_token(t, T, EP):
    """**源设备的编号口径**：token `t` 由哪台设备发 = `t * EP // T`（**均摊口径**）。

    ★★ 与目的设备用**同一个**口径（同一段算术、同一台设备的「连续块」约定）。
    """
    return t * EP // T


def move_matrix(sel, t_tok, e_num, top_k, EP):
    """**搬运量矩阵** `M[s][d]` = 「源设备 `s` 发往目的设备 `d`」的条目数。

    · **源设备**：`device_of_token(t, T, EP)`；
    · **目的设备**：`device_of_expert(e, E, EP)`。
    ★★★ **两处都调【具名函数】**（而不是就地写算式）——
      ★ 就地写算式会让 `A.2` 变成「对自己重写一遍公式」的**自证**
        （本库 Chunk 5 实测：那样把映射改成 `e % EP` 时 `A.2` 照样通过）。
    """
    t_tok_local = sel.shape[0]
    m = np.zeros((EP, EP), dtype=int)
    for t in range(t_tok_local):
        s = device_of_token(t, t_tok, EP)
        for e in sel[t]:
            m[s, device_of_expert(e, e_num, EP)] += 1
    return m


def rate(m):
    """**不均比** = 该维度上的 `max/min`（★ 只在**计数**上取比值，不涉及浮点求和）。"""
    return float(m.max()) / float(m.min())


# ===========================================================================
# 2. A 类：发出量是【恒等式】（`A.1` / `A.1g`）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ **发出量是恒等式**：每个种子 × 每档 EP，每源设备恒发 `T·top_k/EP`")
    print("=" * 78)
    t_tok, e_num, top_k = specs.EP_IMB_MODEL
    expect = t_tok * top_k
    print("  ★★ 依据（**纯计数**）：每个 token **恰**被路由到 `top_k = %d` 个专家，" % top_k)
    print("     而每台源设备**恰**持有 `T/EP = %d/EP` 个 token" % t_tok)
    print("     ⇒ 每源设备**恒**发 `T·top_k/EP` 条 —— ★★★ 这条**与初始分布无关**（恒等式）。")
    print("     ★★ 总投递条数 = `T·top_k` = %d × %d = **%d**（★ 与 EP 无关）"
          % (t_tok, top_k, expect))

    # ── A.1 ★★★ 发出量的恒等性 ─────────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ 八个种子（`EP_IMB_SEEDS`）× 三档 EP：**发出量** `max/min` ──")
    ok_a1 = True
    rows = []
    for ep in specs.EP_DEV_GRID:
        per_send = expect // ep
        vals = []
        for sd in specs.EP_IMB_SEEDS:
            sel = route_topk(sd, t_tok, e_num, top_k)
            m = move_matrix(sel, t_tok, e_num, top_k, ep)
            sends = m.sum(axis=1)
            vals.append((int(sends.min()), int(sends.max()), rate(sends)))
        mn = min(v[0] for v in vals)
        mx = max(v[1] for v in vals)
        rt = max(v[2] for v in vals)
        ok_row = (mn == per_send) and (mx == per_send) and (rt == 1.0)
        ok_a1 = ok_a1 and ok_row
        rows.append((ep, per_send, mn, mx, rt))
        print("     EP=%-2d ⇒ 每源设备应发 `T·top_k/EP` = %d/%-2d = **%d** ·"
              " 实测 min=%d · max=%d · **max/min = %.6f**"
              % (ep, expect, ep, per_send, mn, mx, rt))
    print("     ★★★ **`max/min` 恒为 `1.000000`（不是「大约均衡」）** ⇒ 这是**恒等式**、")
    print("        **不是统计量**：它对**每一个**种子、**每一档** EP 都成立 ⇒ ★ **不需要多样本**。")
    print("     ★★ 物理含义：EP 的**发送侧**是**结构性均衡**的 —— 每个 token 都要走 `top_k` 条路，")
    print("        设备数怎么变都不会多一条或少一条。★★ **不均只发生在【接收侧】**（见 `B`）。")
    check("A.1", ok_a1,
          "★★★ **判据成立**：`EP_DEV_GRID` **三档** × `EP_IMB_SEEDS` **八个种子**下，"
          "每源设备的发出量**恒等于** `T·top_k/EP`（%s），`max/min` **恰为 `1.000000`**"
          " ⇒ ★ 发出量是**恒等式**，不是统计量"
          % " / ".join("%d→%d" % (ep, per) for ep, per, _mn, _mx, _rt in rows),
          "**发出量不再恒等** —— ★ 先查路由是不是真的每个 token 选了 `top_k` 个专家"
          "（`A.1g` 复现「选少了」与「把接收量当发出量」这两种写法）")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    print()
    print("  ── A.1g ★★★ **守卫**：把「发出量」换成「**接收量**」⇒ 恒等性立刻不成立 ──")
    gaps = []
    for ep in specs.EP_DEV_GRID:
        worst = 0.0
        for sd in specs.EP_IMB_SEEDS:
            sel = route_topk(sd, t_tok, e_num, top_k)
            m = move_matrix(sel, t_tok, e_num, top_k, ep)
            worst = max(worst, rate(m.sum(axis=0)))      # ★ 接收侧
        gaps.append((ep, worst))
    for ep, worst in gaps:
        print("     EP=%-2d ⇒ **接收量** `max/min` 的最坏值 = **%.6f**（★ 恒等性的反例）"
              % (ep, worst))
    check("A.1g", all(w > 1.0 for _ep, w in gaps),
          "★★ **守卫成立**：同一个矩阵换一个轴（**接收量**）⇒ `EP_DEV_GRID` **三档**下"
          "`max/min` 都 **> 1**（最坏 **%.6f**）⇒ ★★ 「发出量恒等」这条判据**真的会失败**，"
          "**不是**无条件成立的伪断言 ✓" % max(w for _ep, w in gaps),
          "**守卫失效** —— 接收量也恒等（那说明这台机器上路由退化了）")


    # ── A.2 ★★★ 两张【设备编号表】（★ `B`/`C` 两节报的**每一个数**都建在它们上面）──
    print()
    print("  ── A.2 ★★★ **设备编号口径**：`device_of_expert(e) = e*EP//E`（**专家→设备**）")
    print("     与 `device_of_token(t) = t*EP//T`（**token→设备**）⇒ 每台拿**连续的一段** ──")
    map_rows = []
    ok_map = True
    for ep in specs.EP_DEV_GRID:
        dest = [device_of_expert(e, e_num, ep) for e in range(e_num)]
        srcs = [device_of_token(t, t_tok, ep) for t in range(t_tok)]
        sizes_e = [dest.count(d) for d in range(ep)]
        sizes_t = [srcs.count(d) for d in range(ep)]
        mono_e = all(dest[i] <= dest[i + 1] for i in range(e_num - 1))
        mono_t = all(srcs[i] <= srcs[i + 1] for i in range(t_tok - 1))
        block_e = (dest == [e // (e_num // ep) for e in range(e_num)])
        block_t = (srcs == [t // (t_tok // ep) for t in range(t_tok)])
        ok_row = mono_e and mono_t and block_e and block_t \
            and sizes_e == [e_num // ep] * ep and sizes_t == [t_tok // ep] * ep \
            and dest[0] == 0 and dest[-1] == ep - 1
        ok_map = ok_map and ok_row
        map_rows.append((ep, sizes_e, sizes_t, ok_row))
        print("     EP=%-2d ⇒ 专家→设备 %s · token→设备（前 %d 个）%s ⇒ **两张表都是连续块 = %s**"
              % (ep, sizes_e, t_tok // ep, sizes_t[:t_tok // ep],
                 "是" if ok_row else "否 ✗"))
    print("     ★★★ **为什么这一条必须单独立一条**：本程序在 `B`/`C` 两节报出的**全部**数")
    print("        （接收量 `max/min`、被削平的比值、丢弃条数）**都建在「专家→设备」这张表上** ——")
    print("        ★★ 而 `A.1`（发出量）**钉不住它**：发出量只数「每个 token 走了几条路」，")
    print("           与「它走到**哪台**设备」**无关**。")
    print("     ★★★ **一条具体的错法**：把 `e*EP//E` 换成 `e % EP`（**跨步**）——")
    print("        ★ 每设备**专家数仍然相等** ⇒ `A.1`/`A.1g`/`B.1`/`B.2` **全部照旧通过**，")
    print("          但每台设备拿到的是**一组同余类**而不是连续的一段。")
    print("        ★★ 它**会改掉**接收量的那三个头条数（均值 `1.0593 → 1.0851` 等）")
    print("           ⇒ ★★★ **没有本判据，头条的三个数就没有任何断言保护**。")
    print("     ★★ 本判据读到的是**具名函数**的返回值（不是就地重写一遍公式）⇒")
    print("        ★ 改函数体，本判据**真的会失败**（★ 这是它与「对自己重写公式」的分界）。")
    print("     ★★ 用的是**两个不变量**：**① 单调不减**（= 连续的一段）**+**")
    print("        ② 与 `x // (N//EP)` 这个**等价的块写法**逐元素相等。")
    check("A.2", ok_map and len(map_rows) == 3,
          "★★★ **判据成立**：`EP_DEV_GRID` **三档**下两张设备编号表**都是连续块**"
          "（**单调不减**、且与**等价的块写法** `x // (N//EP)` 逐元素相等）、"
          "且**每台设备拿到的个数相等**（专家 `E/EP` 个、token `T/EP` 个）"
          " ⇒ ★★ **`B`/`C` 两节所有数所依赖的那个映射被钉住了**"
          "（★ 换成 `e % EP` 这种跨步映射 ⇒ 本判据**当场失败**）",
          "**设备编号口径被动过** —— ★ 它是 `B`/`C` 两节**所有数**的基础；"
          "★ 先查 `device_of_expert()` 是不是还给出**连续的一段**专家（`e*EP//E`），"
          "★ 而不是一组同余类（`e % EP`）")


# ===========================================================================
# 3. B 类：接收量的不均（`B.1` / `B.2`）—— ★★★ 统计量，必须报范围
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · ★★★ 接收量的不均：**随 EP 单调恶化**（★ 统计量 ⇒ 报均值【与范围】）")
    print("=" * 78)
    t_tok, e_num, top_k = specs.EP_IMB_MODEL
    print("  ★★★ **初始分布（报数必须带上它）**：`scores ~ N(0,1)` 的 `(%d, %d)` 矩阵、"
          % (t_tok, e_num))
    print("     取每行最大的 `top_k = %d` 个 —— ★★ 换成别的初始分布会给出**不同的比值**。"
          % top_k)

    # ── B.1 ★★★ 均值与范围 + 8/8 逐种子单调 ───────────────────────────────
    print()
    print("  ── B.1 ★★★ **%d 个种子**（`EP_IMB_SEEDS`）下的接收量 `max/min` ──"
          % len(specs.EP_IMB_SEEDS))
    stats = {}
    per_seed = {}
    for ep in specs.EP_DEV_GRID:
        vals = []
        for sd in specs.EP_IMB_SEEDS:
            sel = route_topk(sd, t_tok, e_num, top_k)
            m = move_matrix(sel, t_tok, e_num, top_k, ep)
            vals.append(rate(m.sum(axis=0)))
        stats[ep] = (float(np.mean(vals)), min(vals), max(vals))
        per_seed[ep] = vals
    for ep in specs.EP_DEV_GRID:
        mean, lo, hi = stats[ep]
        print("     EP=%-2d（每设备 %d 个专家）⇒ **均值 %.4f** · 范围 **[%.4f, %.4f]**"
              % (ep, e_num // ep, mean, lo, hi))
    n = len(specs.EP_IMB_SEEDS)
    mono_seed = sum(1 for i in range(n)
                    if per_seed[2][i] < per_seed[4][i] < per_seed[8][i])
    mono_mean = stats[2][0] < stats[4][0] < stats[8][0]
    print("     ★★ **逐种子单调**：%d/%d 个种子上 `EP=2 < EP=4 < EP=8` 都成立" % (mono_seed, n))
    print("     ★★★ **均值单调**：%.4f → %.4f → %.4f"
          % (stats[2][0], stats[4][0], stats[8][0]))
    print("     ★★★ **为什么必须报【范围】而不只报均值**：这是**统计量**（逐种子不同）——")
    print("        只给一个数会让人以为它是**恒等式**（`A.1` 那种），从而**放松警惕**。")
    print("     ★★★ **必须写明【初始分布】**：本程序用的是 `scores ~ N(0,1)` 取 `top_k`；")
    print("        换一种初始分布（例如均匀随机抽 `top_k` 个）**会给出不同的比值**。")
    check("B.1", mono_mean and mono_seed == n,
          "★★★ **判据成立**：`EP_IMB_SEEDS` 的 **%d 个种子**下接收量 `max/min` 的"
          "**均值严格递增**（%.4f → %.4f → %.4f），且 **%d/%d 个种子逐个**都递增"
          " ⇒ ★★ **EP 度越大，接收越不均**（★ 范围一并给出，口径 = 初始分布 + 种子集）"
          % (n, stats[2][0], stats[4][0], stats[8][0], mono_seed, n),
          "**单调性不成立** —— ★ 检查「按设备归并专家」那一步是不是把 `E/EP` 写成了 `E/1`；"
          "★★ 也可能是**初始分布被换掉了**（换分布会改比值，但**不改单调性** ⇒ 若连单调性都没了，"
          "问题在归并而不是分布）")

    # ── B.2 ★★ 成因 ───────────────────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **成因**：每设备摊到的专家数 `E/EP` 随 EP **严格减少** ──")
    counts = [e_num // ep for ep in specs.EP_DEV_GRID]
    for ep, c in zip(specs.EP_DEV_GRID, counts):
        print("     EP=%-2d ⇒ 每设备 **%d** 个专家" % (ep, c))
    print("     ★★★ 专家级的负载**波动幅度**与 EP 无关（那是**路由**的性质），")
    print("        但**每台设备把几个专家的负载加在一起** —— 加得越多，相对波动被**平均掉**得越多。")
    print("        ⇒ `EP=8`（每设备 **1** 个专家）时**一点都不被平均** ⇒ 最不均。")
    print("     ★★★ **本域【不】由此推出任何关于真实集群的结论** —— 这是**计数**的性质，")
    print("        不是任何实现的实测（★ 强制限定语：**不是 NCCL，也不是真机**）。")
    check("B.2", counts == sorted(counts, reverse=True) and len(set(counts)) == len(counts)
          and counts[-1] == 1,
          "★★ **判据成立**：`E/EP` 在 `EP_DEV_GRID` 上严格递减（%s），且最小档 `EP=%d` 时"
          "每设备恰好 **1** 个专家 ⇒ ★★ 「**专家级波动不再被平均掉**」这件事**可复核**"
          % (" → ".join(str(c) for c in counts), specs.EP_DEV_GRID[-1]),
          "**每设备专家数没有递减** —— ★ 那 `B.1` 的单调性就失去了机制上的解释")


# ===========================================================================
# 4. C 类：一个上限 ⇒ 不均【转移】成丢弃（`C.1`）
#    ★★★ 本节的「上限」是 `specs.EP_CAP_LIMIT` **直接给定的整数** ——
#      ★★ **不是**从任何公式算出来的；★ 超了**怎么办**本程序**不表态**
#      （丢 / 重路由是 `80-model-architectures/moe/02-capacity-factor.md` 的正文）。
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**一个专家上限 ⇒ 不均从「收得多」转移成「丢得多」**")
    print("=" * 78)
    t_tok, e_num, top_k = specs.EP_IMB_MODEL
    cap = specs.EP_CAP_LIMIT
    ep8 = specs.EP_DEV_GRID[-1]
    print("  ★★★ **本节的禁区（回链，一行都不重写）**：**容量因子的定义式 / 取整口径 /")
    print("     dropping vs No-Token-Left-Behind / 丢弃率口径** —— 四样的正文都在")
    print("     `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）。")
    print("     ⇒ ★★ 本节的「上限」是一个**直接给定的整数** `cap = %d`（`EP_CAP_LIMIT`），" % cap)
    print("        **不是任何公式的输出**；★ 超了怎么办本程序**不表态**，只数**条数**。")

    # ── C.1 ★★★ 不均转移 ──────────────────────────────────────────────────
    print()
    print("  ── C.1 ★★★ 上限压上之后：**收下的**被削平，**超过上限的**在设备之间不均 ──")
    sd0 = specs.EP_IMB_SEEDS[-1]
    sel = route_topk(sd0, t_tok, e_num, top_k)
    load = np.bincount(sel.ravel(), minlength=e_num)
    # ★★ 每台设备上的**是哪些专家**由 `device_of_expert()` 决定（★ 不写死「连续块」的切片）
    #    ⇒ ★ 本节报出的数也**跟着那个映射走**（`A.2` 钉的就是它）。
    members = [[e for e in range(e_num) if device_of_expert(e, e_num, ep8) == d] for d in range(ep8)]
    recv_raw = np.array([int(load[members[d]].sum()) for d in range(ep8)])
    kept = np.array([int(np.minimum(load[members[d]], cap).sum()) for d in range(ep8)])
    over = np.array([int(np.maximum(load[members[d]] - cap, 0).sum()) for d in range(ep8)])
    # ★★★ **比值前先钉住【分母】**（本库 Chunk 4 的纪律：不给「不可能出 `nan`」的地方加护栏）：
    #   这里**确实**有一个可失败的**前置条件** —— `rate()` 是 `max/min`，`min == 0` 时
    #   Python **抛 `ZeroDivisionError`**（**不是**返回 `inf`/`nan`）⇒ 要守的是**分母非零**，
    #   **不是**「结果有限」。★ 原写法是 `np.isfinite(rate(...))`，理由写「`min` 可能为 `0`」
    #   —— ★★★ 那是一条**编机制从句**（结论对得上、机制是编的）⇒ Chunk 5 改成守分母。
    den_ok = int(recv_raw.min()) > 0 and int(kept.min()) > 0
    r_raw = rate(recv_raw) if den_ok else float("inf")
    r_kept = rate(kept) if den_ok else float("inf")
    print("     一个种子（`EP_IMB_SEEDS` 的最后一个 = **%d**）、`EP=%d`（每设备 1 个专家）："
          % (sd0, ep8))
    print("        每设备的专家负载      = %s" % list(map(int, load)))
    print("        **不加上限时收下的**  = %s ⇒ `max/min` = **%.4f**"
          % (list(map(int, recv_raw)), r_raw))
    print("        **加上限 `cap=%d` 后收下的** = %s ⇒ `max/min` = **%.4f**"
          % (cap, list(map(int, kept)), r_kept))
    print("        **超过上限的条数**    = %s" % list(map(int, over)))
    print("     ★★★ **两件事同时发生**：①「收下的」被**削平**（`max/min` 由 **%.4f** 降到 **%.4f**）；"
          % (r_raw, r_kept))
    print("        ② 但**不均没有消失** —— 它**转移**到「超过上限的条数」上：")
    print("        **%d 台设备丢了、%d 台一台没丢**（丢的条数 %d..%d）"
          % (int((over > 0).sum()), int((over == 0).sum()), int(over[over > 0].min()),
             int(over.max())))
    print("     ★★ **这就是「EP 下负载不均的可观测形态」**：它不会因为加了一个上限就消失，")
    print("        只会从「某台设备忙」变成「**某台设备上的 token 被算了一半就没了**」。")
    print("     ★★★ **一个种子的值**：上面的每一个数都只对种子 **%d** 成立 ——" % sd0)
    print("        **不得**把它当成「典型值」；★★★ **更不得**把它与任何来源的丢弃率比较")
    print("        （★ 口径先在 `80-model-architectures/moe/02-capacity-factor.md` §3.3 对齐）——")
    print("        ★★ 本库**取到过** Switch §2.2 的 `typically <1%`（`specs.SWITCH_DROPPED_RATE_QUOTE`")
    print("           与 `_ALT`；详见本文 §4.4）—— ★ 但本节这些数**只对一个种子成立**，两者不可比。")
    check("C.1", den_ok and r_kept < r_raw and int((over > 0).sum()) > 0
          and int((over == 0).sum()) > 0,
          "★★★ **判据成立**（四个条件）：① **两个比值的分母都非零**（`den_ok`，见上）；"
          "② 加上限后「收下的」`max/min` 由 **%.4f** 降到 **%.4f**；"
          "③ **有**设备超过上限（%d 台）；④ **也有**设备没超（%d 台）⇒ ★★ "
          "**不均转移了、没有消失**。★★ 条件① 守的是**分母** —— `rate()` 是 `max/min`，"
          "`min == 0` 时 Python **抛 `ZeroDivisionError`**（**不是**返回 `inf`/`nan`）"
          % (r_raw, r_kept, int((over > 0).sum()), int((over == 0).sum())),
          "**不均没有转移** —— ★ 检查上限是不是压得太松（一台都没超）或太紧（全都超了）；"
          "★ 或某一台设备**一条都没收到**（`min == 0`）")


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
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` / `numpy %s`（**纯计数**，"
          "无浮点求和）。" % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序量的是**索引搬运的计数**。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · **容量因子的定义与语义**（定义式 / 取整口径 / dropping vs "
          "No-Token-Left-Behind / 丢弃率口径）→")
    print("        `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）；")
    print("        ★★ 本程序的「上限」是 `EP_CAP_LIMIT` **直接给定的整数**，不是公式的输出。")
    print("      · **aux loss 的式子 / 路由 gating 的机制** → "
          "`80-model-architectures/moe/01-gating-and-routing.md`、`03-balance-and-granularity.md`。")
    print("      · **all-to-all 的通信量分析** → `110-communication/collectives`（未建成）、")
    print("        `20-gpu-architecture/interconnect/03`；★ `40-kernels/moe-kernels/01:243` 已让渡。")
    print("      · **kernel 侧的 permute / unpermute / grouped GEMM** → "
          "`40-kernels/moe-kernels/02,03`、`00-meta/glossary.md:262`。")
    print("      · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与 "
          "`10-foundations/performance-models/02:417`（★ **不是 `:223`**）。")
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
          "★ 机器精度档阈 = %g（★ 本程序**只**在 `D.6` 里用它：本程序全是**计数**，"
          "A 类是**精确等式**、B/C 类是**比值** ⇒ 内容级的口径不用机器精度阈）"
          % specs.TOL_MACHINE)
    check("D.3b", len(specs.EP_IMB_SEEDS) == 8
          and isinstance(specs.EP_CAP_LIMIT, int)
          and all(isinstance(s, int) for s in specs.EP_IMB_SEEDS),
          "★★ `EP_IMB_SEEDS` 恰好 **%d** 个、全是整数（★ B 类判据的**小 n** 就写在判据里）；"
          "`EP_CAP_LIMIT` 是整数（**直接给定的上限**，不是公式的输出）"
          % len(specs.EP_IMB_SEEDS))
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
    route_src = inspect.getsource(route_topk)
    move_src = inspect.getsource(move_matrix)
    route_calls = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(route_src)))
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    move_calls = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(move_src)))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.4 ★★ **源码级线索**（★ **不是证明**）──")
    print("     `route_topk()` 调用：%s" % (", ".join(sorted(route_calls)) if route_calls else "**空集**"))
    print("     `move_matrix()` 调用：%s" % (", ".join(sorted(move_calls)) if move_calls else "**空集**"))
    print("     ⇒ ★★ 「**路由**」与「**按设备归并**」是**两段独立代码** ——")
    print("        所以 `A.1`（路由的计数守恒）**不是** `move_matrix()` 自己的重述。")
    print("     ★★★ **降级措辞**：这只是**源码级线索**，**不证明** A/B 类判据有物理信息量 ——")
    print("        后者由 `A.1g`（换成接收量 ⇒ 恒等性立刻失败，这是**行为级**的）兜底。")
    check("D.4", "route_topk" not in move_calls and "move_matrix" not in route_calls
          and route_src != move_src,
          "★★ **源码级线索成立**：两段代码不同源、且**互不调用**"
          "（`route_topk` 的调用表 = %s）⇒ ★ `A.1` 与 `B.1` 量的是**两件不同的事**"
          % (", ".join(sorted(route_calls)) if route_calls else "**空集**"),
          "**路由与归并互相调用** —— A 类可能退化成分派的重述")

    # ── D.5 ★★★ 引文纪律（★ 本程序的**外部事实**依据）───────────────────────
    check("D.5", specs.SWITCH_DROPPED_RATE_QUOTE
          == ("Throughout our experiments we didn’t notice any dependency on the number of "
              "experts for the number of tokens dropped")
          and specs.SWITCH_DROPPED_RATE_ALT == "&lt;1\\%"
          and specs.SWITCH_ALLTOALL_COUNT == 2 and specs.SWITCH_ALLTOALL_TERM == "All-to-All"
          and specs.GSHARD_ALLTOALL_TERM == "AllToAll"
          and specs.GSHARD_COLLECTIVE_QUOTE == "communication operators in the MoE model are",
          "★★★ **「数不到 ≠ 不存在」的实物 + 两条取到过的依据**逐字钉住：\n"
          "        ★★★ **恢复的那条引文**（Switch §2.2，散文，原始字节 **1** 次）：\n"
          "        `Throughout our experiments we didn’t notice any dependency on the\n"
          "         number of experts for the number of tokens dropped`\n"
          "        ★★ 紧随其后的 `typically <1%` 在 `<math>` 里 ⇒ 取 **alttext 形态**\n"
          "        `&lt;1\\%`（原始字节 **2** 次；★ 两次出自**同一个** `<math>` 元素的\n"
          "        `alttext` 与 `x-tex` 注解 —— **不是两处出处**）。\n"
          "        ★ 本程序**原来**在这里写的是「两篇正文里没有任何丢弃率数字」—— ★★★ **那条是假否定**。\n"
          "        ★★ 保留的另一半（**GShard 侧确实没有**）：`AllToAll` **15** 次是**字符串**次数\n"
          "        （**不是**通信次数），且 GShard 正文的 `%` 12 处 / `drop` 5 处**都不给丢弃率**。\n"
          "        ★★ 载体 = ar5iv `2101.03961` / `2006.16668` 正文，抓取日 2026-10-08",
          "**引文常量被动过** —— ★ 先查它是不是跨了 `<math>`（那样会计到 `0`），"
          "★ 或 `didn’t` 里的 `’` 被写成了 ASCII `'`")

    # ── D.6 ★★ 数值常量 ────────────────────────────────────────────────────
    check("D.6", specs.TOL_MACHINE == 1.0e-12 and specs.EP_CAP_LIMIT == 64
          and specs.EP_IMB_MODEL[0] * specs.EP_IMB_MODEL[2] // specs.EP_IMB_MODEL[1]
          == specs.EP_CAP_LIMIT,
          "★★ 数值常量逐位钉住：阈 `1e-12`；上限 `%d` —— ★★ 本输入下一共 `%d` 条投递、"
          "分给 `%d` 个专家 ⇒ **平均每个专家 `%d` 条**，而上限**恰好压在这个平均上**"
          " ⇒ 「有的超、有的不超」**两侧都量得到**。\n"
          "        ★★ **本库只声明这是一条【让两侧都可观测】的取值** —— ★★「上限该由 token 数与"
          "专家数怎么算出」**不是本库的话**（那是回链那一篇的正文）"
          % (specs.EP_CAP_LIMIT,
             specs.EP_IMB_MODEL[0] * specs.EP_IMB_MODEL[2], specs.EP_IMB_MODEL[1],
             specs.EP_CAP_LIMIT),
          "**数值常量被动过** —— ★ 上限与平均负载不再相等时，`C.1` 的两侧条件会失衡")

    # ── D.7 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）─────────────
    check("D.7", specs.EP_DEV_GRID == (2, 4, 8)
          and specs.EP_IMB_SEEDS == (3, 7, 11, 13, 42, 100, 991, 2026)
          and specs.EP_IMB_MODEL == (256, 8, 2),
          "★★★ **三个元组【逐元素】钉住**（`EP_DEV_GRID` 三档 · `EP_IMB_SEEDS` **八个种子** ·"
          " `EP_IMB_MODEL` 三维）—— ★ 只钉「整体」会让**元素级**变异漏网"
          "（本库 `pipeline-parallel` 的那一轮实测过：一次报出 58 条漏网）。\n"
          "        ★★★ **种子集必须逐元素钉住**：`B.1` 报的是**这八个种子**的均值与范围 ——"
          "换掉任何一个种子都会改变报出的数（★ 与同域 `sequence-context-parallel`"
          "【不把种子放进 `specs.py`】的处置**不同**，理由见 `specs.EP_IMB_SEEDS`）",
          "**元组的元素被动过** —— ★ 种子集被动过时，`B.1` 报出的数**不可复核**")

    # ── D.8 ★★ 元组元素与 `EP_DEV_GRID` 的关系（★ 可手算复核）─────────────
    check("D.8", all(specs.EP_IMB_MODEL[1] % ep == 0 for ep in specs.EP_DEV_GRID)
          and all(ep & (ep - 1) == 0 for ep in specs.EP_DEV_GRID)
          and all(s % 2 != 0 or s not in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
                  for s in specs.EP_IMB_SEEDS),
          "★★ `EP_DEV_GRID` **全整除** `E = %d`（⇒ 每设备恰好 `E/EP` 个专家，没有余数干扰）；"
          "三档**全为 2 的幂**（★ 但**种子不是**）；★★ **八个种子都不是 2 的幂**"
          % specs.EP_IMB_MODEL[1],
          "**整除性或种子纪律被破坏** —— ★ `E/EP` 除不尽时 `B.2` 的「每设备专家数」就不再是整数")


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
    print("★ 本程序的 A 类是**恒等式**（发出量），B 类是**统计量**（接收量，报均值与范围）：")
    print("  ★ 两者的写法**必须不同** —— 把 B 类的数当恒等式用，是本主题最容易犯的错。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_load_imbalance.py —— EP 的负载账：发出量均衡、接收量不均（且 EP 越大越不均）")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`，**纯计数**）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序量的是**索引搬运的计数**。")
    print("  ★★★ **本程序两类的写法不同**：`A` 是**恒等式**（一个数就够）、")
    print("      `B` 是**统计量**（必须报 **%d 个种子**的**均值与范围**，且写明**初始分布**）。"
          % len(specs.EP_IMB_SEEDS))
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
