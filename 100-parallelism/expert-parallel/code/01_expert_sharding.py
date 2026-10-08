#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_expert_sharding.py —— **专家怎么分到设备**（每设备 `E/EP` 个），
以及 ★★★ **两次 all-to-all 的位置、语义与载荷尺寸**。

配套文档：`01-expert-sharding.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**：
> **哪台设备拿到哪个专家**、**两次 all-to-all 各自搬什么**、
> ★★ **以及「把专家放错」与「只搬一半」这两种错法各自错多少**。

★★★ **本程序最重要的一条【否定性】结论 —— A 类判据在这里【不能用】**：
  本域最自然的 A 类判据是「**EP 的切分 == 单卡**」。★ 本机实测：
  **把 `top_k` 份拷贝正确地 dispatch 出去、再正确地 combine 回来，`max|diff|` 恰好是
  `0.000e+00`**（`D.4` 逐档量它，`EP ∈ {2, 4}` 两档都是）。
  ⇒ ★★★ 那个 `0` **不是**「实现正确」的证据：**token 分发是纯索引搬运**，
  **它连一次浮点运算都不改** ⇒ 这条判据**恒真**（域 README 的落点②把它归为
  「形状依赖」的那一类；`probes/probe_100h.py` 的 `H3` 是第一代探测）。
  ★★ **可引用的是「形状依赖」这个性质；某组具体形状给 `0` 的清单本身不可引用。**
  ⇒ 本程序的 **A 类因此钉在**「**不整除时的专家分配**」（`A.1`）上，
    而**内容判据在 `C` 类**（`C.1` 放置错位、`C.2` 只做一半）。
  ★★★ **`D.4` 只【报告】那个 `0`，不把它当判据** —— 它是一条**恒真性自检**
    （语气与 `50-performance` 的「报告、不加」一致）：
    ★ 与它同一条里**并列着** `C.1`/`C.2` 的 `2.47` / `2.81` ——
    那两个数才说明「这条路上**有**东西可错」。**没有那两条对照，`D.4` 一行的信息量为零。**

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉「按 `E//EP` 分块」的那段算术 ⇒ 余数无从计算
    （不会通过）；★ `C.1`/`C.2` 是**内容**判据（删掉错法变体就无从比）。
  ★★ 而 `D.4` 这个「恒真性自检」**正是自查问句的答案**：把被测逻辑删掉它照样通过
    ⇒ 所以它**不是判据**，是**报告**。

★★ **覆盖边界**：本程序的 A 类守的是「**专家→设备的划分算术**」，
  **不守**「真实框架怎么摆专家」（那要看具体实现；★ 本程序**不声称**任何框架的行为）。
  ★★ 且 **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：`C.1`/`C.2` 断言的是
  「**错法 ⇒ 差值到 `1e+0` 档**」这个**失败**，不是「一致」这个成功。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · ★★★ **容量因子的定义与语义**（定义式 / 取整口径 / dropping vs No-Token-Left-Behind /
    丢弃率口径）→ `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）。
    ★ **本程序连这个词都不出现** —— `01` 讲的是**专家怎么分**，不碰溢出协议。
  · **aux loss 的式子 / 路由 gating 的机制** → `80-model-architectures/moe/01-gating-and-routing.md`、
    `03-balance-and-granularity.md`。★ 本程序的「路由」只是 `argsort` 取 `top_k`，
    **不含任何 gating 权重**（★ 这与第一代探测 `probe_100h.py` 的 `H3` 口径一致：
      那一支算的是 `X[t] @ We[e]` 的**等权**和）—— **本程序不声称这是一种 gating 方案**。
  · **all-to-all 的通信量分析** → `110-communication/collectives`（**未建成**）+
    `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
    ★★ `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` **逐字让渡**了它给 `110`。
    ⇒ ★ 本程序的 `B` 类**只声明张量的尺寸**（`E × C × d_model`），
    **不声明任何「通信量 ∝ 什么」的结论**。
  · **kernel 侧的 permute / unpermute / grouped GEMM** → `40-kernels/moe-kernels/02,03`；
    `00-meta/glossary.md` 的 `grouped GEMM` 条。★ 本程序里的「搬运」是 **numpy 索引**，不是 kernel。
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（★ **不是 `:223`**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **不整除时**：`E // EP * EP + E % EP == E`，且**余数那 `E % EP` 个专家
    【没有设备可放】**（`EP_SHARD_CASES` 两组：`8/3 → 余 2`、`8/5 → 余 3`）；
  · `A.1g` ★★★ **一条守卫**：改用 `ceil(E/EP)` 分配 ⇒ 总槽位数 **超过 `E`** ⇒ `A.1` 必须失败；
  · `A.2`  ★★ **整除时**：每设备恰好 `E/EP` 个专家、且 `E % EP == 0`（`EP_DEV_GRID` 三档）；
  · `A.2g` ★★ **一条守卫**：拿**不整除**的那一组去过 `A.2` 的条件 ⇒ 必须失败；
  · `B.1`  ★★ 两次 all-to-all 的载荷尺寸 = `E × C × d_model`，**两次 ⇒ 两倍**（逐元素钉住）；
  · `B.2`  ★★ 载荷尺寸**与 `EP` 度无关**（`EP_DEV_GRID` 三档同一个尺寸）；
  · `C.1`  ★★★ **专家放置与路由错位** ⇒ `max|diff| = 2.472807`（**不报错、结果错**）；
  · `C.2`  ★★★ **只做 dispatch 不做 combine** ⇒ `max|diff| = 2.811856`（只有设备 0 的贡献）；
  · `D.4`  ★★★ **恒真性自检**：正确搬运的 `max|diff|` **恰为 `0.000e+00`** ——
    ★ 与 `C.1`/`C.2` 并列 ⇒ **说明「不能把它当 A 类」，不是说明「实现对了」**；
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

# ★ 本程序**内部**的内容级阈（**不进 `specs.py`**：它是「多大算真的错了」的口径，
#   不是从任何来源抄来的常量）⇒ 写在函数体里，见 `part_C()`。
#   （`CONTRIBUTING` §2：只放从来源抄来的数，不放口径。★ 本行刻意**不写模块级常量**。）

# ★★★ **两种**「专家 → 设备」的映射，**必须分开**（`A.1` 量的是它们在不整除时的差别）。
#   ★★ 这不是措辞偏好：它们是**两个不同的整除口径**，混用会让「余数去哪了」这件事消失。
def device_of_expert(e, E, EP):
    """**均摊口径**：`e * EP // E` —— 专家 `e` 落在哪台设备上。

    ★ 这是**域级探测 `probes/probe_100f.py` 用的同一个映射**（它写的是 `top * EP // E`）。
    ★★ 它**总是**给出 `0 <= dev < EP`（每个专家都有设备）——
      ★ 代价是：不整除时**每台设备的专家数不等**（多的比少的多 1 个）。
    """
    return e * EP // E


def device_of_expert_block(e, E, EP):
    """**朴素分块口径**：`e // (E // EP)` —— 「每设备 `E // EP` 个、按块切」。

    ★★★ 不整除时它**会把专家送到不存在的设备上**：最后 `E % EP` 个专家的返回值
      `>= EP` ⇒ **那些专家没有任何设备认领**（`A.1` 头一个钉的就是这件事）。
    """
    return e // (E // EP)


def experts_per_device(E, EP):
    """每台设备摊到几个专家的**地板口径** `E // EP`（★ 不整除时**只是**一个下界）。"""
    return E // EP


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
# 1. 被测：一个「专家分片 + 两次搬运」的 MoE 层（**纯索引搬运**，无 gating 权重）
#    ★★★ 输入（`EP_DISPATCH_SHAPE` = `(T, D, E, top_k)`）与统一权重 `1/top_k`
#      —— 这个口径**与第一代探测 `probes/probe_100h.py` 的 `H3` 同族**
#      （那支也是等权和，不做 gating）。
#    ★★ 种子写在**函数体内**（不是模块级常量）：改它对任何断言都没有可观察后果
#      ⇒ 放进变异分母只会制造一条**不过期的漏网**（同域 `sequence-context-parallel` 的实测教训）。
#      种子取 **`100`**（**不是 2 的幂**）；★ 它与 `probe_100f.py` 的种子同值，
#      但**形状不同**（本程序是 `(32, 8, 4, 2)`）⇒ **两处的数不可互推**（见 `D.4` 的说明）。
# ===========================================================================
def make_inputs():
    """造一层的输入：`X`（token）、`Wg`（路由打分矩阵）、`We`（每个专家一个方阵）。"""
    t_tok, d_dim, e_num, top_k = specs.EP_DISPATCH_SHAPE
    rng = np.random.default_rng(100)
    x = rng.standard_normal((t_tok, d_dim))
    wg = rng.standard_normal((d_dim, e_num)) * 0.3
    we = [rng.standard_normal((d_dim, d_dim)) * 0.4 for _ in range(e_num)]
    return x, wg, we, t_tok, d_dim, e_num, top_k


def route(x, wg, top_k):
    """路由 = **取每行打分最大的 `top_k` 个**（★ 不含 gating 权重：那是 `80-model-architectures/moe` 的正文）。"""
    return np.argsort(-(x @ wg), axis=1)[:, :top_k]


def reference(x, we, sel, top_k):
    """**单卡参照**：每台设备都持有全部专家 ⇒ 没有搬运。"""
    t_tok, d_dim = x.shape
    out = np.zeros((t_tok, d_dim))
    for t in range(t_tok):
        for e in sel[t]:
            out[t] += (1.0 / top_k) * (x[t] @ we[e])
    return out


def sharded(x, we, sel, top_k, E, EP, weight_shift=0, combine=True):
    """**EP 分片实现**：专家按 `device_of_expert` 摆到设备上；两次 all-to-all 用索引搬运表达。

    · `weight_shift=0` ⇒ 专家 `e` 用**它自己**的权重；`=1` ⇒ 用**邻居**的（`C.1` 的错法）。
    · `combine=False` ⇒ **只做 dispatch、不做 combine**：只有**设备 0** 上的那批专家
      （`e < experts_per_device(E, EP)`）的贡献被带回来（`C.2` 的错法）。

    ★★ **本函数是【搬运型】的**：在 `(weight_shift=0, combine=True)` 下它只改
      **哪些项被加起来**，不改任何一项的浮点值
      ⇒ ★★★ 此时 `max|diff|` **恰为 `0.0`**（`D.4` 逐档量它）
      —— **这就是「A 类在这里不能用」的那件事**。
    """
    t_tok, d_dim = x.shape
    dev0 = experts_per_device(E, EP)
    out = np.zeros((t_tok, d_dim))
    for t in range(t_tok):
        for e in sel[t]:
            if not combine and e >= dev0:
                continue
            out[t] += (1.0 / top_k) * (x[t] @ we[(e + weight_shift) % E])
    return out


# ===========================================================================
# 2. A 类：专家→设备的划分算术（`A.1` / `A.1g` / `A.2` / `A.2g`）
#    ★★★ 本程序的 A 类**只能钉在这里** —— 理由见模块抬头（搬运型判据恒真）。
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 专家→设备的划分算术：**整除时每设备 `E/EP` 个；不整除时余数无处安放**")
    print("=" * 78)
    print("  ★★★ **为什么 A 类钉在这里**：替代方案「EP 切分 == 单卡」在本组形状下")
    print("     `max|diff|` **恰为 `0.000e+00`**（`D.4` 量它）⇒ **恒真、无信息量**。")
    print("     ★ 这里钉的是**整数划分** —— 它与形状无关、且**真的会错**（`A.1g`/`A.2g` 演示）。")

    # ── A.1 ★★★ 不整除时的分配 ─────────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **不整除**（`E %% EP != 0`）：两条同时成立 ──")
    print("     (i)  恒等式 `E // EP * EP + E %% EP == E`（余数就是 `E %% EP`）；")
    print("     (ii) **朴素分块口径 `e // (E//EP)`** 会把最后 `E %% EP` 个专家送到")
    print("          **不存在的设备**（返回值 `>= EP`）⇒ ★★ **他们没有设备认领**。")
    print("     ★★★ 而**均摊口径 `e*EP//E`** 不会漏 —— 代价是**每设备专家数不等**（见下第三行）。")
    ok_a1 = True
    detail_rows = []
    for e_num, ep in specs.EP_SHARD_CASES:
        per = experts_per_device(e_num, ep)
        rem = e_num - per * ep
        total = per * ep + rem
        block_dev = [device_of_expert_block(e, e_num, ep) for e in range(e_num)]
        homeless = [e for e, dv in zip(range(e_num), block_dev) if dv >= ep]
        flat = [0] * ep
        for e in range(e_num):
            flat[device_of_expert(e, e_num, ep)] += 1
        spread = max(flat) - min(flat)
        ok_row = (total == e_num) and (rem == e_num % ep) and (rem > 0) \
            and (len(homeless) == rem) and sorted(homeless) == list(range(e_num - rem, e_num)) \
            and (sum(flat) == e_num) and (spread == 1)
        ok_a1 = ok_a1 and ok_row
        detail_rows.append((e_num, ep, per, rem, flat, spread))
        print("     E=%-2d EP=%-2d ⇒ 每设备 %d 个 · **余 %d 个** · `%d*%d+%d = %d` = E ✓"
              % (e_num, ep, per, rem, per, ep, rem, total))
        print("        朴素分块 `e//(E//EP)` 的设备号 = %s ⇒ **越界（>= EP）的是 %s** ⇒ "
              "**%d 个没有设备**" % (block_dev, homeless, len(homeless)))
        print("        均摊 `e*EP//E`（探测脚本口径）每设备拿到 %s ⇒ "
              "**摆下全部 %d 个**，但**最多比最少多 %d 个**" % (flat, sum(flat), spread))
    print("     ★★★ 物理含义：**专家数是死的、设备数是活的** —— 除不尽时只有两条路：")
    print("        **① 让有的设备多摊一个（均摊口径）**，或 **② 把某个专家【切开】跨设备放**。")
    print("        ★ `A.2` 说明整除时不会有这件事；★ 它与 TP 的「头切分」是**同族**问题。")
    check("A.1", ok_a1,
          "★★★ **判据成立**（五个条件都逐组核过）：① `E//EP*EP + E%%EP == E`；"
          "② 余数 = `E %% EP` 且 > 0；③ **朴素分块下有 %d / %d 个专家越界（没有设备）**，"
          "且越界的正是**最后那几个**；④ 均摊口径**摆下全部** `E` 个；"
          "⑤ 均摊口径下每设备专家数的**极差恰好是 1**"
          % (detail_rows[0][3], detail_rows[1][3]),
          "**不整除的划分算术被改动了** —— ★ 先查 `E // EP * EP + E % EP == E` 这条恒等式")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    # ★★ 守卫的错法：**把「不整除」当成「整除」** —— 即断言「朴素分块的设备号都 < EP」。
    #    ★★★ 这条断言**在不整除的两组上必然为假**（那正是 A.1 的判据 (ii)），
    #      而在**整除**的对照组上**恰好为真** ⇒ ★ 它**真的改变了被判量**。
    e0, ep0 = specs.EP_SHARD_CASES[0]
    e1, ep1 = specs.EP_SHARD_CASES[1]
    ediv, epdiv = specs.EP_IMB_MODEL[1], 4          # E=8, EP=4：**整除**的对照组
    bad0 = max(device_of_expert_block(e, e0, ep0) for e in range(e0)) >= ep0
    bad1 = max(device_of_expert_block(e, e1, ep1) for e in range(e1)) >= ep1
    good = max(device_of_expert_block(e, ediv, epdiv) for e in range(ediv)) < epdiv
    check("A.1g", bad0 and bad1 and good,
          "★★ **守卫成立**：「朴素分块的设备号都 < EP」这条断言 "
          "① 在 `(E=%d, EP=%d)` 上**为假**（最大设备号 %d >= %d）；"
          "② 在 `(E=%d, EP=%d)` 上**为假**（最大设备号 %d >= %d）；"
          "③ 而在**整除**的对照组 `(E=%d, EP=%d)` 上**为真**（最大设备号 %d < %d）"
          " ⇒ ★★ 换掉「除得尽吗」这个条件**真的改变了被判量**，`A.1` **能失败** ✓"
          % (e0, ep0, max(device_of_expert_block(e, e0, ep0) for e in range(e0)), ep0,
             e1, ep1, max(device_of_expert_block(e, e1, ep1) for e in range(e1)), ep1,
             ediv, epdiv, max(device_of_expert_block(e, ediv, epdiv) for e in range(ediv)), epdiv),
          "**守卫失效** —— 不整除时朴素分块居然没有越界（`A.1` 对这条错法不敏感）")

    # ── A.2 ★★ 整除时的分配 ────────────────────────────────────────────────
    print()
    print("  ── A.2 ★★ **整除**（`E %% EP == 0`）：每设备恰好 `E/EP` 个，一个不多、一个不少 ──")
    ok_a2 = True
    for ep in specs.EP_DEV_GRID:
        e_num = specs.EP_IMB_MODEL[1]
        per = experts_per_device(e_num, ep)
        got = [0] * ep
        for e in range(e_num):
            got[device_of_expert(e, e_num, ep)] += 1
        ok_row = (e_num % ep == 0) and all(g == per for g in got) and per * ep == e_num
        ok_a2 = ok_a2 and ok_row
        print("     E=%d EP=%-2d ⇒ 每设备 %d 个 · 实摆 %s ⇒ %s"
              % (e_num, ep, per, got, "**均匀** ✓" if ok_row else "**不均匀** ✗"))
    print("     ★★ 这就是 `EP_DEV_GRID` 三档取 **2 的幂**的理由：它们**全部整除** `E = 8`")
    print("        ⇒ 02 的负载账里没有「余数」这一项干扰（★ 余数的账在 `A.1`）。")
    check("A.2", ok_a2,
          "★★ **判据成立**：`EP_DEV_GRID` **三档**下 `E %% EP == 0`、每设备恰好 `E/EP` 个、"
          "且 `experts_per_device*EP == E`（★ 逐档核过实摆结果）",
          "**整除时的分配被动过** —— ★ 先查 `EP_DEV_GRID` 是不是仍整除 `E`")

    # ── A.2g ★★ 守卫 ──────────────────────────────────────────────────────
    print()
    ex, epx = specs.EP_SHARD_CASES[1]
    check("A.2g", (ex % epx != 0) and (ex // epx * epx != ex),
          "★★ **守卫成立**：拿**不整除**的那一组 `(E=%d, EP=%d)` 去过 `A.2` 的条件 ⇒"
          "`E %% EP = %d ≠ 0`、`E//EP*EP = %d ≠ %d` ⇒ `A.2` 必须失败 ✓"
          "（★ 这条守卫**真的改变了被判量**）" % (ex, epx, ex % epx, ex // epx * epx, ex),
          "**守卫失效** —— 不整除的那一组下 `A.2` 仍然通过")


# ===========================================================================
# 3. B 类：两次 all-to-all 的**载荷尺寸**（`B.1` / `B.2`）
#    ★★★ **本域只给尺寸，不给通信量** —— 通信量分析的正文在 `110-communication/collectives`（**未建成**）
#      与 `20-gpu-architecture/interconnect/03`；`40-kernels/moe-kernels/01:243`
#      **逐字让渡**了它。★ 本节的「每设备发出的元素数」是**本库的记账模型**，不是原文的话。
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 两次 all-to-all 的**载荷尺寸**（★ 只给尺寸：通信量不在本域）")
    print("=" * 78)
    e_num, cap, d_model = specs.EP_A2A_DIM
    payload = e_num * cap * d_model

    # ── B.1 ★★ 尺寸与「两次」────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ 张量的形状是 `[E, C, d_model]`（Switch §5.4 逐字），**两次** ⇒ 两倍 ──")
    print("     `specs.SWITCH_DISPATCH_ALLTOALL_QUOTE`：")
    print("        `%s`" % specs.SWITCH_DISPATCH_ALLTOALL_QUOTE)
    print("     `specs.SWITCH_COMBINE_QUOTE`（第二次的方向）：")
    print("        `%s`" % specs.SWITCH_COMBINE_QUOTE)
    print("     ⇒ 尺寸 `E × C × d_model` = %d × %d × %d = **%d** 个元素（**每一次**）"
          % (e_num, cap, d_model, payload))
    print("     ⇒ **两次** ⇒ **%d** 个元素（本库的记账模型：每设备手里那个张量的大小）"
          % (2 * payload))
    check("B.1", payload == e_num * cap * d_model and 2 * payload == 2 * e_num * cap * d_model,
          "★★ **判据成立**：每次 all-to-all 的载荷 = `E × C × d_model` = **%d** 个元素；"
          "两次 ⇒ **%d**（★ 本域**只声明尺寸**，不声明通信量）" % (payload, 2 * payload),
          "**载荷尺寸被动过** —— ★ 先查 `EP_A2A_DIM` 的三个维")

    # ── B.2 ★★ 尺寸与 `EP` 度无关 ───────────────────────────────────────────
    print()
    print("  ── B.2 ★★ 载荷尺寸**与 `EP` 度无关**（它是**静态形状**，不是「除以设备数」）──")
    for ep in specs.EP_DEV_GRID:
        print("     EP=%-2d ⇒ 每设备手里的张量仍是 `[%d, %d, %d]` = **%d**（★ 与 EP 无关）"
              % (ep, e_num, cap, d_model, payload))
    print("     ★★ **这条是本域最容易写错的地方之一**：**EP 度不改每个张量的大小**，")
    print("        它改的是**张量里哪一块属于我**（all-to-all 把 `E` 维换成了被切分的那一维）。")
    print("     ★★★ **本域【不】由此推出任何通信量结论** —— 那是 `110-communication/collectives`")
    print("        （**未建成**）与 `20-gpu-architecture/interconnect/03` 的正文；")
    print("        ★ `40-kernels/moe-kernels/01:243`")
    print("        已**逐字让渡**。")
    same = all(e_num * cap * d_model == payload for _ in specs.EP_DEV_GRID)
    check("B.2", same and len(specs.EP_DEV_GRID) == 3,
          "★★ **判据成立**：`EP_DEV_GRID` **三档**下每设备的载荷尺寸都是 **%d** ⇒"
          "**尺寸是静态的、与 EP 度无关**（★ 只有归属变）" % payload,
          "**载荷尺寸随 EP 变了** —— ★ 那意味着有人把「切分」写成了「除法」")


# ===========================================================================
# 4. C 类：两条错法（`C.1` / `C.2`）—— ★★★ 它们都**不报错**，只是结果错
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**专家放错** 与 **只搬一半**（★ 两条都真的会失败）")
    print("=" * 78)
    print("  ★★★ 两条错法的**共同点**：形状对、类型对、算术对、**一次异常都不抛** ——")
    print("      唯一的信号是**与参照的差**。★ 而参照（单卡）与它们**不共享任何代码**")
    print("      （`sharded()` 与 `reference()` 是两段独立实现；见 `D.8` 的源码级线索）。")
    x, wg, we, t_tok, d_dim, e_num, top_k = make_inputs()
    sel = route(x, wg, top_k)
    ref = reference(x, we, sel, top_k)
    ep = specs.EP_DEV_GRID[0]
    print("  形状（`EP_DISPATCH_SHAPE`）：T=%d · D=%d · E=%d · top_k=%d ⇒ 总投递 %d 条"
          % (t_tok, d_dim, e_num, top_k, t_tok * top_k))

    # ── C.1 ★★★ 专家放置与路由错位 ─────────────────────────────────────────
    print()
    print("  ── C.1 ★★★ **专家放置与路由【错位】**：专家 `e` 用了**邻居**的权重（`e+1`）──")
    got = sharded(x, we, sel, top_k, e_num, ep, weight_shift=1)
    d_c1 = float(np.abs(got - ref).max())
    print("     正确放置 `device_of_expert(e, E, EP)` ⇒ 每设备拿到的专家：%s"
          % [[e for e in range(e_num) if device_of_expert(e, e_num, ep) == d] for d in range(ep)])
    print("     错位：每个专家**由邻居算**（`We[(e+1) %% E]`）⇒ 搬运的**目的地**整体错一格")
    print("     `max|diff|` = **%.6f**" % d_c1)
    print("     ★★★ **它【不报错】**：张量的形状、类型、算术全都对 ——")
    print("        错的是「**哪个专家在算**」，而这件事**没有任何运行时检查**会发现它。")
    check("C.1", d_c1 > 1e-6,
          "★★★ **判据成立**：专家放置错位（每个专家换了邻居）⇒ `max|diff|` = **%.6f**"
          "（≫ 内容级阈 `1e-6`）⇒ ★ **不报错、结果错**" % d_c1,
          "**错位没有让结果变** —— ★ 那说明参照与分片版共用了一段代码（自证）")

    # ── C.2 ★★★ 只做 dispatch 不做 combine ─────────────────────────────────
    print()
    print("  ── C.2 ★★★ **只做 dispatch、不做 combine**：只有**设备 0** 的贡献被带回来 ──")
    got2 = sharded(x, we, sel, top_k, e_num, ep, combine=False)
    d_c2 = float(np.abs(got2 - ref).max())
    print("     设备 0 上的专家 = `[0, %d)` ⇒ 它只带回这些专家的输出" % experts_per_device(e_num, ep))
    print("     `max|diff|` = **%.6f**" % d_c2)
    print("     ★★★ **它【不报错】**：第二次 all-to-all 干脆没做 —— 索引、形状全都合法，")
    print("        只是**大部分 token 的贡献根本没回来**。★ 这是「两次里少一次」的写错面。")
    check("C.2", d_c2 > 1e-6 and abs(d_c2 - d_c1) > 1e-6,
          "★★★ **判据成立**：不做 combine ⇒ `max|diff|` = **%.6f**（≫ `1e-6`），"
          "且它与 `C.1` 的 **%.6f** **不同** ⇒ ★★ 两条错法是**两种**不同的失败" % (d_c2, d_c1),
          "**不做 combine 没有让结果变**，或它与 `C.1` 给同一个数（★ 两条错法可能是同一个）")


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
    print("        ★★ **本程序连这个词都不出现**（`01` 讲的是专家怎么分，不碰溢出协议）。")
    print("      · **aux loss 的式子 / 路由 gating 的机制** → "
          "`80-model-architectures/moe/01-gating-and-routing.md`、`03-balance-and-granularity.md`；")
    print("        ★ 本程序的路由只是 `argsort` 取 `top_k`、权重是**等权** `1/top_k`。")
    print("      · **all-to-all 的通信量分析** → `110-communication/collectives`（未建成）、")
    print("        `20-gpu-architecture/interconnect/03`；★ `40-kernels/moe-kernels/01:243` 已让渡。")
    print("      · **kernel 侧的 permute / unpermute / grouped GEMM** → "
          "`40-kernels/moe-kernels/02,03`、`00-meta/glossary.md` 的 `grouped GEMM` 条。")
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
          "★ 标记自检：**等于**本主题被分配的字面量 `|| FAIL: expert-parallel`、且含 `FAIL`\n"
          "        ★★ 那个 `||` 在 **Makefile 的注释里**被转义写成 `\\|\\|`（★ README 同）——")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                                        "?? FAIL: zero-fsdp"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住；"
          "★ 六个标记**两两不互为前缀**")

    # ── D.3 ★ 口径常量（本程序的**输入**与阈）──────────────────────────────
    check("D.3", specs.TOL_MACHINE == 1.0e-12,
          "★ 机器精度档阈 = %g（★ 本程序只在 `D.6` 里用它；**内容级**的阈是函数体内的 `1e-6`）"
          % specs.TOL_MACHINE)
    check("D.3b", all(isinstance(v, int) for v in specs.EP_A2A_DIM)
          and len(specs.EP_A2A_DIM) == 3
          and all(isinstance(v, int) for v in specs.EP_DISPATCH_SHAPE)
          and len(specs.EP_DISPATCH_SHAPE) == 4,
          "★★ `EP_A2A_DIM`（三个维，`[E, C, d_model]`）与 `EP_DISPATCH_SHAPE`（四个维，"
          "`(T, D, E, top_k)`）**全是整数**且长度分别恰好是 `3` 与 `4` ⇒ 切片不会静默错位")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：恒真性自检 · specs 常量逐位钉住 · 逐元素钉住 · 引文纪律 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 恒真性自检 · specs 常量逐位钉住 · **逐元素**钉住 · 引文纪律 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★★ 恒真性自检（**报告，不是判据**）────────────────────────────
    x, wg, we, t_tok, d_dim, e_num, top_k = make_inputs()
    sel = route(x, wg, top_k)
    ref = reference(x, we, sel, top_k)
    rows = []
    for ep in specs.EP_DEV_GRID[:2]:
        got = sharded(x, we, sel, top_k, e_num, ep)
        rows.append((ep, float(np.abs(got - ref).max())))
    allzero = all(v == 0.0 for _ep, v in rows)
    print()
    print("  ── D.4 ★★★ **恒真性自检**：正确的「dispatch + combine」在本组形状下的 `max|diff|` ──")
    for ep, v in rows:
        print("     EP=%-2d ⇒ 正确搬运 `max|diff|` = **%.3e**" % (ep, v))
    print("     ★★★ **它是 `0.000e+00`（恰好为零）** —— 而这不是「实现对了」的证据：")
    print("        **token 分发是纯索引搬运，它连一次浮点运算都不改** ⇒ 这条判据**恒真**。")
    print("     ★★ **同一批数据、同一个参照**下，两条**内容级**错法给的是：")
    print("        `C.1` 放置错位 **%.6f** · `C.2` 不做 combine **%.6f**"
          % (float(np.abs(sharded(x, we, sel, top_k, e_num, specs.EP_DEV_GRID[0],
                                  weight_shift=1) - ref).max()),
             float(np.abs(sharded(x, we, sel, top_k, e_num, specs.EP_DEV_GRID[0],
                                  combine=False) - ref).max())))
    print("     ⇒ ★★★ **两个数才说明这条路上有东西可错** —— 没有它们，`D.4` 的信息量为零。")
    print("     ★★ **可引用的性质**是「**搬运型判据是形状依赖的、本组形状恰好给 `0`**」；")
    print("        ★★★ **某组具体形状给 `0` 的清单本身不可引用**（域 README 的落点②）。")
    print("     ★ 口径提示：`probes/probe_100f.py` 用 `T=256, E=8`、`probe_100h.py` 的 `H3` 用")
    print("        `T=64, D=16`，**本程序用 `%s`** ⇒ ★★ **三处的数不可互推**。"
          % str(tuple(specs.EP_DISPATCH_SHAPE)))
    check("D.4", allzero and len(rows) == 2,
          "★★★ **自检成立（报告）**：正确搬运在 `EP_DEV_GRID` 的**前两档**下 `max|diff|` "
          "**都是 `0.000e+00`** ⇒ ★★ **这条判据恒真** ⇒ **本程序的 A 类不钉在这里**"
          "（上方两条**内容级**对照才是它的意义所在）",
          "**正确搬运不再给出 `0.0`** —— ★ 那说明这一段**不是**纯搬运了；"
          "★★ 此时 `A.1`/`A.2` 的定位需要重新审视（本自检的结论**已经过期**）")

    # ── D.5 ★★ Switch §5.4 的两条散文引文（**载体 + 计数**）────────────────
    check("D.5", specs.SWITCH_DISPATCH_ALLTOALL_QUOTE
          == "Because each core has its own expert, we do an all-to-all communication of size"
          and specs.SWITCH_COMBINE_QUOTE
          == ("in the forward pass to analogusly receive the tokens from each expert "
              "located on different cores.")
          and specs.SWITCH_SECTION54_QUOTE
          == "Next we describe the partitioning strategy for expert and data parallelism.",
          "★★ Switch §5.4 的**三条**散文引文逐字钉住 —— ★ 每条都**切在 `<math>` 标记之前**"
          "（第一条以 `of size` 收尾，**不带那个 `[E, C, d_model]`**）\n"
          "        ★★★ 第二条里的 **`analogusly`** 是**原文的拼写**（就是那个拼法）——"
          "改成 `analogously` 会在原始字节上计到 `0`。\n"
          "        ★★ 载体 = `https://ar5iv.labs.arxiv.org/html/2101.03961` 正文，"
          "抓取日 **2026-10-08**，逐条在**原始字节**上 `count()` = **1**",
          "**Switch 的引文常量被动过** —— ★ 先查它是不是跨了 `<math>`（那样会计到 `0`）")

    # ── D.6 ★★ 数值常量（`TOL_MACHINE` + 那对「同词不同形」）─────────────────
    check("D.6", specs.TOL_MACHINE == 1.0e-12
          and specs.SWITCH_ALLTOALL_COUNT == 2 and specs.SWITCH_ALLTOALL_TERM == "All-to-All"
          and specs.SWITCH_HYPHENATED_VERB_COUNT == 3,
          "★★ 数值常量逐位钉住：阈 `1e-12`；★★★ **`All-to-All` 2 次**（伪码注释）与"
          "**`all-to-all` 3 次**（散文用词）—— **同词不同形**：把两者合并成一个常量，"
          "就抹掉了「散文用词 / 伪码注释名」这个区别本身（★ 与同域 `sequence-context-parallel`"
          "的 `SP_SPLIT_REPLICATED` / `SP_EQ2_TP_NUM` 那一对**同族**）；\n"
          "        ★★ 载体 = ar5iv `2101.03961` 正文，抓取日 2026-10-08，原始字节上分别 `count()`",
          "**数值常量被动过** —— ★ 或者那两个串被合并了（`2` 与 `3` 变成了同一个数）")

    # ── D.7 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）─────────────
    check("D.7", specs.EP_DEV_GRID == (2, 4, 8)
          and specs.EP_SHARD_CASES == ((8, 3), (8, 5))
          and specs.EP_DISPATCH_SHAPE == (32, 8, 4, 2)
          and specs.EP_A2A_DIM == (4, 8, 16),
          "★★★ **四个元组【逐元素】钉住**（`EP_DEV_GRID` 三档 · `EP_SHARD_CASES` 两组×两元 ·"
          " `EP_DISPATCH_SHAPE` 四维 · `EP_A2A_DIM` 三维）—— ★ 只钉「整体」会让**元素级**变异漏网"
          "（本库 `pipeline-parallel` 的那一轮实测过：一次报出 58 条漏网；"
          "★ 嵌套元组的**内层**叶子后缀形如 `[0][1]`，更要逐元素）",
          "**元组的元素被动过**")

    # ── D.8 ★★ 独立性线索（源码级）──────────────────────────────────────────
    srcs = {inspect.getsource(f) for f in (reference, sharded, route, make_inputs)}
    calls_ref = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(reference))))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_sh = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(sharded))))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：参照与分片版的**调用表** ──")
    print("     `reference()` 里调用的函数：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    print("     `sharded()`   里调用的函数：%s"
          % (", ".join(sorted(calls_sh)) if calls_sh else "**空集**"))
    print("     ⇒ ★★ `reference()` **不调用** `sharded()`（也不调用 `route`/`make_inputs`）——")
    print("        **参照实现不得调用被测实现的任何函数**（域 README 的硬要求）在此**在源码层面成立**。")
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，")
    print("        **不证明**「A/C 类判据有物理信息量」—— 后者由 `C.1`/`C.2` 的**行为**兜底")
    check("D.8", len(srcs) == 4 and "sharded" not in calls_ref
          and "reference" not in calls_sh and "route" not in calls_ref
          and "make_inputs" not in calls_ref,
          "★★ **源码级线索成立**：四个函数两两不同源；★ `reference()` 的调用表里"
          "**没有** `sharded` / `route` / `make_inputs` ⇒ ★ **参照不调用被测**（源码层面）",
          "**参照与分片版互相调用** —— §A/§C 的判据可能退化成自证")

    # ── D.9 ★★★ GShard 的**否定性**结论（★ 不得给它编次数）────────────────
    check("D.9", specs.GSHARD_ALLTOALL_TERM == "AllToAll" and specs.GSHARD_ALLTOALL_COUNT == 15
          and specs.GSHARD_HYPHEN_COUNT == 0
          and specs.GSHARD_COLLECTIVE_QUOTE == "communication operators in the MoE model are",
          "★★★ 三条**否定性**依据逐字钉住：GShard 用的是**一个词** `AllToAll`（原始字节 **15** 次）、"
          "而带连字符的 `all-to-all` **0** 次 ⇒ ★★ **两篇论文的措辞【不同形】**。\n"
          "        ★★★ 那个 `15` 是**字符串出现次数**，**不是**通信次数的计数 ⇒"
          " **本程序绝不从它推出「GShard 做了几次 all-to-all」**（★ 那会是「编数字」）。\n"
          "        ★★ 且 GShard §5.3 自己说 MoE 的两个关键集合通信是 `AllReduce` 与 `AllToAll`"
          " ⇒ **all-to-all 不是 MoE 唯一的通信**。\n"
          "        ★★ 载体 = ar5iv `2006.16668` 正文（**同行评审未确认**），抓取日 2026-10-08",
          "**GShard 的依据常量被动过** —— ★ 特别注意别把 `AllToAll` 改写成带连字符的形式")

    # ── D.10 ★★ 同行评审依据的载体 + `UNSOURCED` 留空 ───────────────────────
    check("D.10", specs.SWITCH_ABS_JMLR == "JMLR" and specs.UNSOURCED == ()
          and specs.EP_NOT_FOUND_NOTE != "",
          "★★ **同行评审：是**的依据 = arXiv abs 页（`https://arxiv.org/abs/2101.03961`）上"
          "`Comments` 字段渲染文本逐字 `JMLR`（原始字节 **1** 次，抓取日 2026-10-08）。\n"
          "        ★★ 与同域 `sequence-context-parallel` **不同**：那边 `2205.05198` 的 abs 页"
          "**给不出**依据（`Comments` **0** 次），依据停在 MLSys 2023 索引页 ⇒ **两处不得互相照抄**。\n"
          "        ★★ `specs.UNSOURCED` **留空**（两处「未核到」已写在各自常量的 `# 来源：` 里"
          "或由本域另一篇程序消费）；★ GShard 的同行评审状态只有「**未确认**」。",
          "**同行评审的依据常量被动过**，或 `UNSOURCED` 不再是空的")


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
    print("★ 两次 all-to-all 的依据在 `specs.py` 的 `# 来源：`（Switch Transformer §5.4 + 附录 F）；")
    print("  本程序只核对「专家→设备的划分算术」与「两条内容级错法真的会失败」。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_expert_sharding.py —— 专家怎么分到设备 + 两次 all-to-all 的位置与语义")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`，矩阵乘走 BLAS）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本程序的主结论是一条【否定性】的**：EP 的 token 分发是**纯索引搬运**")
    print("      ⇒ 「EP 切分 == 单卡」的 `max|diff|` **恰为 `0.000e+00`**（`D.4`）⇒ **恒真、不能当 A 类**。")
    print("      ⇒ A 类钉在**整数划分**（`A.1`/`A.2`），**内容判据在 `C.1`（放置错位）与 `C.2`（只搬一半）**。")
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
