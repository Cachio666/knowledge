#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_communication.py —— **ZeRO 三个阶段的通信量**。

配套文档：`02-communication.md`。

★★★ **本程序的一句话**：`P_os` 与 `P_g` **不增加任何通信**；`P_os+g` 是 **`2Ψ`**
（**与 baseline DP 完全相同**）；**只有** `P_os+g+p` 是 **`3Ψ`（`1.5x`）**。

★★★ **出处纪律（三条，逐条落实在断言与打印里）**：
  ① **`P_os` 的「不增通信」以 §7 开头的总结句为准**（`specs.ZERO_NO_EXTRA_COMM_QUOTE`，
     逐字 `incurs no additional communication using`）；★ **§1 列表的 `same communication
     volume as DP` 可作同向互证**（`specs.ZERO_S1_SAME_COMM_QUOTE`，`P_os` 与 `P_os+g` 各一次）
     —— ★★★ **原文【没有】 `P_os` 单独的通信量等式** ⇒ **不得自己造一个 `P_os = 2Ψ`
     并标成「论文式(N)」**（那是**编来源 / 编机制从句**）。
     ★ 本程序的 `comm_units()` **不给 `P_os` 编等式**：它走的是「**reduce-scatter(1) +
     参数 all-gather(1) = 2**」这条 **baseline 就有的**结构（`ZERO_RS_AG_QUOTE` 是它的依据）。
  ② **baseline 逐字是** `the standard DP incurs 2Ψ data movement during each training step.`
     （§7.1，`specs.ZERO_BASELINE_QUOTE`）⇒ 本程序把它拆成**两条不跨标记的载体片段**
     （`ZERO_BASELINE_HEAD` / `ZERO_BASELINE_TAIL`）存证。
  ③ **`K` 不是通信量** —— 它是**优化器状态的内存乘子**（`= 12`，`specs.ZERO_K_MEMORY_MULTIPLE`）。
     ★ 本程序的通信模型里**一个 `K` 都不出现**（`D.8` 用**源码级线索**钉住这件事）。

★★★ **一条必须写明的【记号桥】（不是订正）**：
  `10-foundations/performance-models/02-amdahl-and-scaling.md` 里的 `S` 是**梯度字节**，
  而 ZeRO 的 `Ψ` 是**参数量** —— 梯度 fp16、每参数 `2` 字节 ⇒ **`S = 2Ψ`（字节）**
  ⇒ ★★★ **该文的 `2S → 3S` 就是论文的 `2Ψ → 3Ψ`**（`D.7` 把这条桥的算术钉住）。
  ★★★ **那句话是【对的】，一个字都不许改** —— 计划 §5 v3 已**复核撤回**对它「订正」的判定
  （成因是 v1/v2 **没引 baseline 的定义**，见 `ZERO_RS_AG_QUOTE` 与计划新立的 **`P30`**）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：把 `param_all_gathers()` 里的「`P_os+g+p` 要 gather **两次**」删掉
    （即所有 ZeRO 阶段都 gather 一次）⇒ `A.1`（`P_os+g+p` 那档给出 `2Ψ ≠ 3Ψ`）与 `A.2` 的 `3Ψ`
    **必然失败** ⇒ ★ **不是自证**（`A.1g`/`C.1g` 把「记错 ZeRO 阶段」的反例真的跑了一遍）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **集合通信原语的算法与复杂度**（ring all-reduce 的 `V(N) = 2(N−1)/N × S`、阶段划分）→
    `10-foundations/performance-models/02-amdahl-and-scaling.md:209,428` 与
    `110-communication/collectives`）；
  · **拓扑受限的集合通信**（bisection 下界 / 分层）→
    `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **链路带宽的三种口径**与 NVLink / IB 的规格值 →
    `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`；
  · **通信-计算重叠的机制** → `110-communication/overlap/`（`overlap`）；
  · ★★★ **`16Ψ` 的单卡口径与它的字节分解** → `90-training/checkpointing/01-state-and-bytes.md`
    （**本域只写「分片后」的账**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **三个 ZeRO 阶段逐档**：`P_os`/`P_os+g` = `1 + 1 = 2Ψ`、`P_os+g+p` = `1 + 2 = 3Ψ`
         —— 分别对上 §7 开头 / §1 列表 / §7.2 / §7.3 的逐字（★ **覆盖三档**，不只是 `P_os+g`）；
  · `A.1g` ★★★ **一条守卫**：把 `P_os+g` 记成「参数 gather 两次」（= 记错 ZeRO 阶段）⇒ `3Ψ ≠ 2Ψ`
         ⇒ `A.1` 的判据**必须失败**；
  · `A.2`  ★★★ `P_os+g+p` 的通信量 = `1 + 2 = 3Ψ`，且 **== `2 × 1.5`（`2` 来自 §7.1、
         `1.5` 来自 §7.3，两处**各自独立**的逐字出处）**；
  · `B.1`  ★★ 五档（含 `P_g`）的通信量对照表：**`P_os`/`P_g`/`P_os+g` 都不增**（`= 2Ψ`），
         只有 `P_os+g+p` 是 `3Ψ`；
  · `B.2`  ★★ `3Ψ ÷ 2Ψ = 1.5` == `ZERO_1_5X_VERBATIM`（§7.3 逐字）；
  · `C.1`  ★★★ **反例「把 `3Ψ` 套到 `P_os` 上」**（`reschedule_param=True`）⇒ 与 baseline
         `2Ψ` **直接矛盾**，矛盾量 `|3Ψ − 2Ψ| = 1 > 0` ⇒ ★★ **「记错 ZeRO 阶段」必然失败**；
  · `C.1g` ★★★ **一条守卫**：把 `C.1` 里的**错法**换回**对法**（`P_os` 正确记成 `2Ψ`）⇒
         矛盾量归零 ⇒ `C.1` 的不等式**必然失败**；
  · `D.*`  引文逐条钉住 + `K` 不进通信 + 记号桥 + **失败标记自检**。
"""
import ast
import collections
import inspect
import sys
import textwrap

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（zero-fsdp）取 `?? FAIL: zero-fsdp`。
#   ★ 本域另外五个主题各用**不同的前缀**（`%%` / `^^` / `==` / `{{` / `||`），
#     六个标记**两两不互为前缀** ⇒ 跨主题照抄会被下面两条断言当场抓住。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
#   ★ 它**只允许出现在 `print()` 的字符串里** ⇒ 本文件里它只出现在打印与断言明细中。
ASSERT_FAIL_MARK = "?? FAIL: zero-fsdp"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                     "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                     "|| FAIL: expert-parallel")

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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立」的措辞是"
                                "**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 被测：一个「每步通信量（单位 Ψ）」的模型
#    ★★★ 它**没有**给 `P_os` 编一个等式（出处纪律①）—— 它走的是 **baseline 就有的结构**：
#      每步一次 **reduce-scatter**（归约梯度）+ 若干次 **参数 all-gather**。
#      ★ `ZERO_RS_AG_QUOTE`（§7.1 逐字）就是这个结构的依据：
#        「reduce-scatter(`Ψ`) + all-gather(`Ψ`) = `2Ψ`」—— **基线的 `2Ψ` 里【已含一次 all-gather】**。
#    ★★ `reschedule_param=True` 是**错法开关**（`A.1g`/`C.1g` 用它）：把「参数 all-gather」
#      从 1 次改成 2 次 —— 即「把 `P_os`/`P_os+g` 错记成 `P_os+g+p`」。
# ===========================================================================
def param_all_gathers(stage, reschedule_param=False):
    """该 ZeRO 阶段每步【参数 all-gather】的次数。

    · **baseline**（以及 `P_os`/`P_g`/`P_os+g`）：**1 次** —— 更新完各自的参数分片后，
      把参数**收回来**（§7.2 逐字：`an all-gather is performed to collect all the updated parameters`）。
    · **`P_os+g+p`**：**2 次** —— 把参数 all-gather **摊到整个前向**，用完即丢；
      反向还要**再 gather 一次**（§7.3 逐字：`needs to happen once again for the backward propagation`）。
    """
    if stage == "P_os+g+p" or reschedule_param:
        return 2
    return 1


def comm_units(stage, reschedule_param=False):
    """每步通信量，单位 **Ψ**（元素）：`reduce-scatter(1) + 参数 all-gather 次数`。

    ★★★ **本函数里【一个 `K` 都没有】** —— `K`（`= 12`）是**优化器状态的内存乘子**，
    属于**显存**那一侧的账（本主题 `01` 用），**不参与**通信量（出处纪律③，`D.8` 钉住）。
    """
    return 1 + param_all_gathers(stage, reschedule_param)


# ===========================================================================
# 2. A 类：算术复现（`Ψ+Ψ=2Ψ` 与 `3Ψ`）—— 两条来源不同的路径对拍
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 算术复现：`Ψ+Ψ=2Ψ`（基线）与 `3Ψ`（`P_os+g+p`）")
    print("=" * 78)
    print("  ★★★ **被测走的是结构、参照走的是论文逐字** —— 两条来源不同的路径：")
    print("      · 被测：`reduce-scatter(1) + 参数 all-gather(g)`（`g` 由 ZeRO 阶段定）；")
    print("      · 参照：§7.1 的 `2`（`ZERO_BASELINE_MULTIPLE`）与 §7.3 的 `1.5`（`ZERO_1_5X_VERBATIM`）。")

    # ── A.1 ★★★ 三个 ZeRO 阶段各自的通信量（**覆盖三档**，不只是 `P_os+g`）──────
    #   ★★★ **为什么必须覆盖全三档**：若 `A.1` 只断言 `P_os+g`，那么「把
    #     `param_all_gathers()` 里的 ZeRO 阶段判别**整体删掉**（永远返回 `1`）」**仍然通过** ——
    #     因为 `P_os+g` 那档 `1 + 1 = 2` **恰好等于**论文 §7.1 的常量 `2`。
    #     ⇒ ★ 加进 `P_os+g+p` 那档（`3Ψ ≠ 2Ψ`）后，删掉判别**必然在某档失败**
    #       （结构层审查者实测过这个漏洞）。
    print()
    print("  ── A.1 ★★★ 三个 ZeRO 阶段：`P_os`/`P_os+g` = `2Ψ`、`P_os+g+p` = `3Ψ` ──")
    print("     §7 开头（`specs.ZERO_NO_EXTRA_COMM_QUOTE`）：`%s`" % specs.ZERO_NO_EXTRA_COMM_QUOTE)
    print("     §1 列表（`specs.ZERO_S1_SAME_COMM_QUOTE`）：`%s`" % specs.ZERO_S1_SAME_COMM_QUOTE)
    print("     §7.2（`specs.ZERO_POG_SAME_QUOTE`）：`%s`" % specs.ZERO_POG_SAME_QUOTE)
    print("     §7.3（`specs.ZERO_POGP_1_5X_QUOTE`）：`%s`" % specs.ZERO_POGP_1_5X_QUOTE)
    a1_pos = comm_units("P_os")
    a1_posg = comm_units("P_os+g")
    a1_posgp = comm_units("P_os+g+p")
    print("     被测：`P_os`     = 1 + %d = **%dΨ**（§7 开头「不增通信」/ §1 列表同值）"
          % (param_all_gathers("P_os"), a1_pos))
    print("     被测：`P_os+g`   = 1 + %d = **%dΨ**（§7.2 逐字 `%s`）"
          % (param_all_gathers("P_os+g"), a1_posg, specs.ZERO_POG_SAME_QUOTE))
    print("     被测：`P_os+g+p` = 1 + %d = **%dΨ**（§7.3 逐字 `%s`）"
          % (param_all_gathers("P_os+g+p"), a1_posgp, specs.ZERO_POGP_1_5X_QUOTE))
    print("     参照：§7.1 逐字 `%s` … ⇒ 基线 `ZERO_BASELINE_MULTIPLE = %d`"
          % (specs.ZERO_BASELINE_HEAD, specs.ZERO_BASELINE_MULTIPLE))
    check("A.1", a1_pos == specs.ZERO_BASELINE_MULTIPLE == 2
          and a1_posg == specs.ZERO_BASELINE_MULTIPLE == 2 and a1_posgp == 3,
          "★★★ **判据成立**：`P_os` = **%dΨ**（§7 开头「不增通信」）、`P_os+g` = **%dΨ**"
          "（§7.2 逐字 `%s`）、`P_os+g+p` = **%dΨ**（§7.3 逐字 `%s`）"
          " ⇒ ★★ **三个 ZeRO 阶段逐档与论文逐字对上**（不只是 `P_os+g` 那一格）"
          % (a1_pos, a1_posg, specs.ZERO_POG_SAME_QUOTE, a1_posgp, specs.ZERO_POGP_1_5X_QUOTE),
          "**三个 ZeRO 阶段的通信量被动过** —— ★ 先查每档的参数 all-gather 次数"
          "（`P_os`/`P_os+g` = 1 次、`P_os+g+p` = 2 次）")

    # ── A.1g ★★★ 守卫（记错 ZeRO 阶段）───────────────────────────────────────────
    a1_bad = comm_units("P_os+g", reschedule_param=True)
    check("A.1g", a1_bad != specs.ZERO_BASELINE_MULTIPLE,
          "★★ **守卫成立**：把 `P_os+g` **错记成** `P_os+g+p`（参数 gather 两次）⇒ 算得 **%dΨ ≠ %d**"
          " ⇒ ★★ 换掉「记哪个 ZeRO 阶段」**真的改变了被判量**，`A.1` **能失败** ✓"
          % (a1_bad, specs.ZERO_BASELINE_MULTIPLE),
          "**守卫失效** —— 记错 ZeRO 阶段后通信量居然没变（`A.1` 对 ZeRO 阶段不敏感）")

    # ── A.2 ★★★ `P_os+g+p` 的 `3Ψ` ──────────────────────────────────────────
    print()
    print("  ── A.2 ★★★ `P_os+g+p`：`1 + 2 = 3Ψ`（前向一次 + 反向一次 all-gather）──")
    print("     `specs.ZERO_POGP_HEAD`：`%s`" % specs.ZERO_POGP_HEAD)
    print("     `specs.ZERO_POGP_1_5X_QUOTE`：`%s`" % specs.ZERO_POGP_1_5X_QUOTE)
    a2 = comm_units("P_os+g+p")
    ref = specs.ZERO_BASELINE_MULTIPLE * specs.ZERO_1_5X_VERBATIM
    print("     被测：1 + %d = **%dΨ**" % (param_all_gathers("P_os+g+p"), a2))
    print("     参照：`%d × %s = %s`（`2` 来自 §7.1、`1.5` 来自 §7.3 —— **两处各自独立**）"
          % (specs.ZERO_BASELINE_MULTIPLE, specs.ZERO_1_5X_VERBATIM, ref))
    check("A.2", a2 == 3 and abs(a2 - ref) < TOL_MACHINE,
          "★★★ **判据成立**：`P_os+g+p` 通信量 = **%dΨ**（结构：1 + 2）== **%s**（`2 × 1.5`，两条逐字出处）"
          % (a2, ref),
          "**`P_os+g+p` 的账被动过** —— ★ 先查它的参数 all-gather 是不是仍为 2 次")


# ===========================================================================
# 3. B 类：五档 ZeRO 阶段的通信量对照表（**不增 / 2Ψ / 3Ψ**）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 五档 ZeRO 阶段的通信量对照表（**不增 / 2Ψ / 3Ψ**）")
    print("=" * 78)
    baseline = comm_units("DP-baseline")
    rows = []
    for stage in ("P_os", "P_g", "P_os+g", "P_os+g+p"):
        rows.append((stage, comm_units(stage)))
    print("  ★ baseline（`DP-baseline`）= **%dΨ**" % baseline)
    print("  ★★ **`P_g` 是「只切梯度」那一档**（论文把它与 `P_os` 合称 `P_os+g`）—— "
          "它与 `P_os` **一样不增通信**。")
    for stage, u in rows:
        mark = "**不增**（= baseline）" if u == baseline else "**增到 %dΨ**" % u
        print("    %-10s ⇒ %s" % (stage, mark))

    no_increase = [u for s, u in rows if s in ("P_os", "P_g", "P_os+g")]
    check("B.1", no_increase == [baseline] * 3 and rows[3][1] == 3
          and comm_units("P_os") == baseline and comm_units("P_g") == baseline,
          "★★ **判据成立**：`P_os`/`P_g`/`P_os+g` 三档**都不增通信**（都 `= %dΨ`）；"
          "只有 `P_os+g+p` 是 `3Ψ`" % baseline,
          "**通信量对照表被动过** —— ★ 先查 `P_os`/`P_g`/`P_os+g` 是否仍等于 baseline")

    # ── B.2 ★★ `3Ψ ÷ 2Ψ = 1.5` ─────────────────────────────────────────────
    ratio = comm_units("P_os+g+p") / comm_units("P_os+g")
    print("  ★★ `3Ψ ÷ 2Ψ` = **%.4f**（§7.3 逐字 `%s`）—— ★ **注意它不是「多了 1.5 倍」**："
          "论文的措辞是「**是原本的 1.5 倍**」，而 §1 列表把它写成「**多了 50%%**」（`%s`）"
          % (ratio, specs.ZERO_POGP_1_5X_QUOTE, specs.ZERO_S1_50PCT_QUOTE))
    check("B.2", abs(ratio - specs.ZERO_1_5X_VERBATIM) < TOL_MACHINE
          and specs.ZERO_1_5X_VERBATIM == 1.5,
          "★★ **判据成立**：`3Ψ ÷ 2Ψ` = **%.4f** == `ZERO_1_5X_VERBATIM = %s`（§7.3 逐字）"
          % (ratio, specs.ZERO_1_5X_VERBATIM),
          "**倍数被动过** —— ★ 先查 `ZERO_1_5X_VERBATIM` 与那两句逐字引文")


# ===========================================================================
# 4. C 类：**反例「把 `3Ψ` 套到 `P_os` 上」** —— 与 baseline `2Ψ` 直接矛盾
#    ★★★ `CONTRIBUTING:238`：**C 类断言成立时程序应当失败** ⇒ 本节**只装「错法」**，
#      断言「**按错法算会得到一个与真值显著不同的数**」（这里是「矛盾量 > 0」）。
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**记错 ZeRO 阶段** ⇒ 与 baseline 直接矛盾")
    print("=" * 78)
    print("  ★★★ 这是 `performance-models/02:221` 那个错的【现场】—— 把 **`P_os+g+p` 的账"
          "算到【三阶段整体】头上**（计划 §5 v3 的 `P24` 现场：那一句其实**基本是对的**，"
          "唯一的问题是措辞「接近…以上」略有夸张）。")
    base = comm_units("DP-baseline")
    right_p_os = comm_units("P_os")
    wrong_p_os = comm_units("P_os", reschedule_param=True)        # 错法：把 `P_os` 记成 `P_os+g+p`
    contradiction = abs(wrong_p_os - base)
    print("  ★★★ **本节的错法**：把 `P_os+g+p` 的 `3Ψ` **套到 `P_os` 上**（= 记错 ZeRO 阶段）——")
    print("      `P_os` 正确是 **%dΨ**（§7 开头「不增通信」），错记成 `P_os+g+p` 后变 **%dΨ**"
          % (right_p_os, wrong_p_os))
    print("      ⇒ ★★ 它与 baseline `%dΨ`（§7.1 逐字 `%s`）**直接矛盾**。"
          % (base, specs.ZERO_BASELINE_HEAD))
    print()
    print("     错法：`P_os`（错记成 `P_os+g+p`）= **%dΨ** · baseline = **%dΨ** ⇒ 矛盾量 = **%d**"
          % (wrong_p_os, base, contradiction))
    print("     ★★ 若把这一档当成 `P_os` 报出去，**读者会以为「不增通信」是错的**。")
    print("     ★ 引文（§7.1 已含一次 all-gather —— 这正是 `3Ψ` 增量的归属依据）：`%s`"
          % specs.ZERO_RS_AG_QUOTE)
    check("C.1", contradiction > TOL_MACHINE and wrong_p_os != base
          and base == specs.ZERO_BASELINE_MULTIPLE == 2,
          "★★★ **判据成立**：把 `3Ψ` 套到 `P_os` 上 ⇒ 算得 **%dΨ**，与 baseline `%dΨ` 的矛盾量 "
          "= **%d > 0** ⇒ ★★ **「记错 ZeRO 阶段」必然失败**；★ 且 `3Ψ` 相对 baseline 的增量"
          "**恰好是「多一次 all-gather（参数）」**（§7.3 定 `3Ψ` = `2Ψ × %s` = `%s`）"
          % (wrong_p_os, base, contradiction,
             specs.ZERO_1_5X_VERBATIM, specs.ZERO_POGP_1_5X_QUOTE),
          "**「把 `3Ψ` 套到 `P_os`」居然不与 baseline 矛盾** —— ★ 先查 `reschedule_param` 开关")

    # ── C.1g ★★★ 守卫：把「错法」换回「对法」⇒ 矛盾量归零 ⇒ C.1 的不等式**必然失败** ──
    #   ★★★ 这就是「这条 C 类真的会失败」那个验证的**常驻形式**。
    right_contradiction = abs(right_p_os - base)
    print()
    print("     ★★ 对法：`P_os` 正确记成 **%dΨ** ⇒ 与 baseline 的矛盾量 = **%d**"
          % (right_p_os, right_contradiction))
    check("C.1g", right_contradiction <= TOL_MACHINE,
          "★★ **守卫成立**：把 `C.1` 里的**错法**换回**对法**（`P_os` 正确记成 `%dΨ`）"
          " ⇒ 矛盾量归零（**%d**）⇒ ★★ C.1 的那个不等式**不再成立**"
          " ⇒ **C.1 对「记哪个 ZeRO 阶段」敏感**（它不是恒真的）"
          % (right_p_os, right_contradiction),
          "**守卫失效** —— 用对法后与 baseline 竟然仍「矛盾」（那说明 C.1 判的不是这个方法）")


# ===========================================================================
# 5. 前哨：口径 / 阈值 / 标记自检（判据 D.1 / D.2 / D.2b / D.3）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**推导值 + 引文**：通信量是**算术**（不是本机测出来的性能）。")
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · **集合通信原语的算法与复杂度**（`V(N) = 2(N−1)/N × S`、阶段划分）→ "
          "`10-foundations/performance-models/02-amdahl-and-scaling.md:209,428`、"
          "`110-communication/collectives`（")
    print("      · **拓扑受限的集合通信**（bisection 下界 / 分层）→ "
          "`20-gpu-architecture/interconnect/03-collectives-on-topology.md`；"
          "**链路带宽的三种口径** → 同域 `interconnect/01`")
    print("      · **通信-计算重叠的机制** → `110-communication/overlap/`（`overlap`，")
    print("      · **`16Ψ` 的单卡口径与它的字节分解** → "
          "`90-training/checkpointing/01-state-and-bytes.md`（本域只写「分片后」的账）")
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
    check("D.2", ASSERT_FAIL_MARK == "?? FAIL: zero-fsdp" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `?? FAIL: zero-fsdp`、且含 `FAIL`\n"
          "        ★★ 那几个 `?` 是 shell / make 的元字符 ⇒ 它**只出现在 `print()` 里**。")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                                        "|| FAIL: expert-parallel"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住")
    check("D.3", specs.ZERO_BASELINE_MULTIPLE == 2 and specs.ZERO_1_5X_VERBATIM == 1.5
          and specs.ZERO_S_IS_GRAD_BYTES_MULT == 2 and specs.ZERO_K_MEMORY_MULTIPLE == 12
          and specs.TOL_MACHINE == 1.0e-12,
          "★ 逐字常量：基线 `%dΨ` · 倍数 `%s` · `S = %dΨ` · `K = %d`（**内存乘子，不是通信量**）· "
          "阈 `%g`" % (specs.ZERO_BASELINE_MULTIPLE, specs.ZERO_1_5X_VERBATIM,
                       specs.ZERO_S_IS_GRAD_BYTES_MULT, specs.ZERO_K_MEMORY_MULTIPLE,
                       specs.TOL_MACHINE))
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：引文逐条钉住 · `K` 不进通信 · 记号桥 · 出处纪律
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐条钉住 · `K` 不进通信 · 记号桥 · 出处纪律")
    print("=" * 78)

    # ── D.4 ★ 出处纪律①：`P_os` 的「不增通信」只能引 §7 开头 ───────────────
    check("D.4", specs.ZERO_NO_EXTRA_COMM_QUOTE == "incurs no additional communication using"
          and specs.ZERO_S7_OPEN_HEAD == "The answer is in two parts: i)",
          "★★★ **出处纪律①**：`P_os` 的「不增通信」**只能引 §7 开头的总结句** ——"
          " 逐字 `%s`（载体片段，原始字节 1 次）。\n"
          "        ★★★ **原文【没有】`P_os` 单独的通信量等式** ⇒ **不得自己造一个 `P_os = 2Ψ` 并标成"
          "「论文式(N)」**。★ 本程序的 `comm_units()` 走的是 **baseline 就有的结构**（`ZERO_RS_AG_QUOTE`）。"
          % specs.ZERO_NO_EXTRA_COMM_QUOTE)

    # ── D.5 ★★★ §7 的各条引文逐字钉住 ──────────────────────────────────────
    check("D.5", specs.ZERO_S7_OPEN_QUOTE == (
              "The answer is in two parts: i) ZeRO-DP incurs no additional communication using "
              "P_os and P_g, while enabling up to 8x memory reduction, ii) ZeRO-DP incurs a "
              "maximum of 1.5x communication when using P_p in addition to P_os and P_g, while "
              "further reducing the memory footprint by N_d times.")
          and specs.ZERO_UP_TO_8X_QUOTE == "while enabling up to 8x memory reduction"
          and specs.ZERO_FOOTPRINT_BY_ND_QUOTE == "while further reducing the memory footprint by"
          and specs.ZERO_BASELINE_QUOTE == ("the standard DP incurs 2Ψ data movement "
                                            "during each training step.")
          and specs.ZERO_RS_AG_QUOTE == ("Both reduce-scatter and all-gather are implemented "
                                         "using a pipelined approach")
          and specs.ZERO_POG_SAME_QUOTE == "exactly the same as the baseline DP."
          and specs.ZERO_POGP_1_5X_QUOTE == "which is 1.5x compared to the baseline.",
          "★★★ §7 的**七条**引文逐字钉住 —— ★★ §7 开头是**两句完整的**（v1/v2 曾把 `ii)` 的从句"
          "切下来贴到 `i)` 上，**删掉了 `i)` 真正的结尾**（`%s`）⇒ 本模块**两种计数都写明**：\n"
          "        整句（跨 `<math>`）原始字节 **0** 次；不跨标记的载体片段各 **1** 次。\n"
          "        ★★★ **别把 0 次读成「原文没有这句」**（那是本库 v6.11 的**假否定**）。"
          % specs.ZERO_UP_TO_8X_QUOTE,
          "**§7 的引文常量被动过** —— ★ 先查它是不是跨了 `<math>`（那样整句会计到 0）")

    # ── D.6 ★★ 载体片段 + `2` 与 `1.5` 的逐字依据 + 两种口径 ─────────────────
    check("D.6", specs.ZERO_BASELINE_HEAD == "the standard DP incurs 2"
          and specs.ZERO_BASELINE_TAIL == "data movement during each training step."
          and specs.ZERO_POG_SUM_HEAD == "So the total communication volume per training step is"
          and specs.ZERO_POGP_HEAD == "The total volume is therefore"
          and specs.ZERO_S1_50PCT_QUOTE == "modest 50% increase in communication volume",
          "★★ `2` 的逐字依据 = `%s`（原始字节 1 次）；`1.5x` 的逐字依据 = §7.3 的 "
          "`which is 1.5x compared to the baseline.`；\n"
          "        ★★ 而 §1 列表把它写成 `%s` ⇒ **「多了 50%%」与「是原本的 1.5 倍」是同一件事的"
          "两种口径**（计划 `P28` 家族）—— 报「增了多少」时必须写明是哪一种。"
          % (specs.ZERO_BASELINE_HEAD, specs.ZERO_S1_50PCT_QUOTE))

    # ── D.6b ★★ 「2 次」是**两处出处** ──────────────────────────────────────
    check("D.6b", specs.ZERO_S1_SAME_COMM_QUOTE == "same communication volume as DP",
          "★★ `same communication volume as DP` 在 §1 列表里出现 **2 次**（`P_os` 与 `P_os+g` 各一）"
          " ⇒ ★★ **这「2 次」是【两处出处】**（★ 与 90 域的一处教训**相反**：那边有一处「2 次」"
          "来自同一个 `<math>` 的两份注释，**不是两处出处**）。⇒ 它与 §7 开头那句**互相印证**。")

    # ── D.7 ★★★ 记号的桥（`S = 2Ψ`）────────────────────────────────────────
    ndim_bridge = specs.ZERO_S_IS_GRAD_BYTES_MULT == specs.ZERO_BYTES_PER_PARAM
    check("D.7", ndim_bridge and specs.ZERO_S_IS_GRAD_BYTES_MULT == 2,
          "★★★ **记号的桥**：`performance-models/02` 的 `S` = 梯度**字节**、ZeRO 的 `Ψ` = **参数量**，"
          "梯度 fp16（每参数 `%d` 字节）⇒ **`S = %dΨ`（字节）**\n"
          "        ⇒ 该文的 `mS`（字节）= `m × %d × Ψ` = 论文的 `mΨ`（元素，每元素 %d 字节）"
          " ⇒ ★★★ **`2S → 3S` 就是 `2Ψ → 3Ψ`**（倍数 `m` 是**共用的**）。\n"
          "        ★★★ **那句话【是对的】，一个字都不许改** —— 计划 §5 v3 已**撤回**「订正」判定"
          "（成因是 v1/v2 没引 baseline 的定义 ⇒ 计划新立 `P30`）。"
          % (specs.ZERO_BYTES_PER_PARAM, specs.ZERO_S_IS_GRAD_BYTES_MULT,
             specs.ZERO_BYTES_PER_PARAM, specs.ZERO_BYTES_PER_PARAM),
          "**记号桥被动过** —— ★ 先查 `S` 是否仍等于「每参数字节 × Ψ」")

    # ── D.8 ★★ `K` 不进通信模型（源码级线索）───────────────────────────────
    src_comm = textwrap.dedent(inspect.getsource(comm_units))
    src_gather = textwrap.dedent(inspect.getsource(param_all_gathers))
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：通信模型里**有没有 `K`** ──")
    print("     `comm_units()` 的源码里出现 `K`（`ZERO_K_MEMORY_MULTIPLE`）吗：%s"
          % ("**没有** ✓" if "ZERO_K_MEMORY_MULTIPLE" not in src_comm else "**有** ✗"))
    check("D.8", "ZERO_K_MEMORY_MULTIPLE" not in src_comm
          and "ZERO_K_MEMORY_MULTIPLE" not in src_gather,
          "★★★ **出处纪律③**：`K` = `%d` 是**优化器状态的内存乘子**（属于**显存**那一侧的账）"
          " ⇒ ★★ **它不出现在通信模型里**（`comm_units()` 与 `param_all_gathers()` 的源码里都没有它）"
          " —— `%s`" % (specs.ZERO_K_MEMORY_MULTIPLE, "源码级线索成立"),
          "**`K` 混进了通信模型** —— ★ 它是内存乘子，不是通信量")

    # ── D.9 ★★ `[C]` 论文的级与同行评审 + `UNSOURCED` 留空 ──────────────────
    check("D.9", specs.UNSOURCED == (),
          "★ ZeRO 论文 = **[C] 论文**、**同行评审：未确认**（★ arXiv abs 页无 `Comments` / "
          "`journal-ref` 字段 ⇒ 本库不推断）；★ `specs.UNSOURCED` **留空**。\n"
          "        ★★★ `P_os` **单独的通信量等式原文【本来就没有】** —— 那是**禁写**，"
          "**不是「未取到」**（依据只有 §7 开头的**总结句**）。")


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
    print("★ `P_os`/`P_g` 不增通信（§7 开头）；`P_os+g` = `2Ψ`（§7.2）；"
          "`P_os+g+p` = `3Ψ` = `1.5x`（§7.3）。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_communication.py —— ZeRO 三个阶段的通信量（不增 / 2Ψ / 3Ψ）")
    print("=" * 78)
    print("  ★ 真值标注：**推导值 + 论文引文**（★ 通信量是算术，不是本机测出来的性能）。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **出处纪律**：`P_os` 的「不增通信」**只引** §7 开头的总结句（原文**没有**"
          "`P_os` 单独的等式）；`K = 12` **不是通信量**（它是内存乘子）。")
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
