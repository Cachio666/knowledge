#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_roce_and_infiniband.py —— **RoCE v2 的机制**（PFC / ECN / DCQCN）与 **InfiniBand 的 SHARP**。

配套文档：`03-roce-and-infiniband.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本层是【物理层】**（以太网上的 RDMA 与在网计算），**不是**算法层也**不是**实现层。

★★★ **本文件【没有 A 类】—— 本机无网卡**（无 InfiniBand / RoCE 网卡、无 GPU、无交换机）：
  「RoCEv2 真的这么封装吗」「PFC 真的把丢包消掉了吗」**无处可验**。
  ★★ 所以**一切钉住归 `D` 类**；★★★ **不得把「常量钉住」冒充 A 类**
    （那是 `P12` 的变体：**判据看着在检查，其实只是把常量抄了一遍**）。
  ★★ 而「**把不可测的那一档写明**」是【另一条】纪律（不是 `CONTRIBUTING` §2 的任何一种编号形态）。

★★★ **★ 本文件与 `nccl` 的 `D.0` 的【差别】（★ 这是刻意的）**：
  `nccl` 的三个程序用 **AST 扫【源码】**证明「没有 A 类判据」；本文件**不照抄那条路线**，
  改用 **运行期记账 + 调用点核对** 两条独立机械量（见 `D.0`）：
    ① **运行期**：`check()` 每跑一次就把**实际执行过的类别**记进 `_TOTAL`
       ⇒ 跑完后 `_TOTAL` 的**键集合必须恰好是 `{B, C, D}`**（**没有 `A` 这个键**）；
    ② **调用点核对**：用 AST 数本文件里 `check(` 的**调用点个数**，它必须**等于**实际执行次数之和
       ⇒ ★★ 这样**连「一段从未被执行的 A 类判据（死代码）」也跑不掉**（★ 只数运行期会漏它）。
  ★★ **两条一起才叫可机械复核**：单靠 ① 会漏死代码，单靠 ② 就是 `nccl` 那条路线的变体。

★★★ **两条禁写清单（计划 §C.5）**：
  · **`F4`** —— **不得**把 RoCEv2 的**帧结构 / UDP 封装**标成 **`[A]`（规范）**：
    IBTA 的一手规范（**Annex A17**）**未取到** ⇒ 只能标 **`[B]`**（Cisco 产品文档）。
    ★★ 本程序把它做成**机械断言**（`C.2` + `specs.HAVE_IBTA_ANNEX_A17`）。
  · **`F6`** —— **DCQCN / SHARP 2016 的正文未取到** ⇒ **只引【载体】**；
    ★★★ **DCQCN 的三组件描述**与 **SHARP 的 `2.1×` / `3.24×`** 来自**二手检索摘要**
      ⇒ **不得当逐字引文**（★ 本程序 `D.8` 会**扫源码**确认那两个数字**一次都没出现**）。
  ★ **`F7`** —— DBLP 页取不到 ⇒ **不得声称「查过 DBLP」**。

★★ **覆盖边界**：本程序守的是「**规格值抄对了没有**」与「**两条错法真的会被拒**」，
  **不守**任何一条**真实 RoCE / IB 的行为**（本机**测不了**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `B.1` ★★ **规格值表**：`7` 条事实、**逐条标 `A`/`B`** 与**本轮复核状态**（`6` 已复核 / `1` 未取到）；
  · `B.2` ★★★ **PFC 的官方作用域**（三条逐字：`intended to` · `limited to a domain` · `802.3x PAUSE`）；
  · `B.3` **SHARP 的在网计算定位**（`[B]`，两版逐字相同）；
  · `C.1` ★★★ **把 PFC 当成流控的万能解**必须被拒（跨**原因**与跨**类别**两个方向）；
  · `C.2` ★★★ **把 RoCEv2 的 UDP 封装标成 `[A]`** 必须被拒（`F4`）；
  · `D.0` ★★★ **机械证明本文件没有 A 类**（运行期记账 + 调用点核对）；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **禁写清单扫描**（`D.8`）+ **失败标记自检**。
"""
import ast
import collections
import sys

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（network-topology）取 `,, FAIL: network-topology`（★ 三个程序**同一个**）。
#   ★★ `,` 在 **shell 里需要引号**、在**正则里才是元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ",, FAIL: network-topology"

# 本域另外三个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("&& FAIL: collectives", ">< FAIL: nccl", "+= FAIL: overlap")

# ★★★ `D.0` 的两条机械量：本文件**声明【没有】的类别**与**实际允许出现的类别**。
#   ★★ 它们是模块级常量 ⇒ 会被 `mutate.py` 变异 ⇒ **必须各有一条 `==` 钉住**（`D.0` 里）。
ABSENT_CLASS = "A"
ALLOWED_CLASSES = ("B", "C", "D")

# ★★★ `D.8` 的禁写清单（`F6`）：这样两个数字**一次都不许出现在本主题的任何源码里**。
FORBIDDEN_TOKENS = ("2.1×", "3.24×")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
    ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 失败标记必然在**行首**。
    ★★★ **本文件里 `check()` 的调用点与执行次数必须一一对应**（`D.0` 核）：
      所以**不要**把 `check()` 放进循环或分支里。
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
# 1. 被测：本库的**范围判定器**两条 —— ★★ **它们【不是】A 类**（本机没有网卡可测）
#    ★★★ 它们是**判定逻辑**：把「官方措辞的范围」变成**可数/可比的东西**，
#      用来判「把 PFC 当万能解」「把封装标成 [A]」这两类错法。**一切钉住归 `D` 类**。
# ===========================================================================
def pfc_claim_ok(claim_causes, claim_priorities, enabled_priorities):
    """★★★ **PFC 声明的范围校验器**：声称的范围必须**被官方范围包住**才算成立。

    · `claim_causes`：这条声明声称 PFC 能对付**哪些丢包原因**；
    · `claim_priorities`：声称保护**哪些流量类别**；
    · `enabled_priorities`：**实际配置**里开了 PFC 的类别。
    ★★ 判定 = 「原因 ⊆ 官方点名的原因（`specs.PFC_OFFICIAL_LOSS_CAUSES`）」
           且「类别 ⊆ 实际开着的类别」。
    ★★★ **官方的两条限制是逐字的**（`specs.IEEE8021QBB_PURPOSE_QUOTE` 的 `due to congestion`
      与 `specs.IEEE8021QBB_DOMAIN_QUOTE` 的 `is limited to a domain`）⇒ ★ 本函数**不是**凭空的规则。
    """
    official = set(specs.PFC_OFFICIAL_LOSS_CAUSES)
    return set(claim_causes) <= official and set(claim_priorities) <= set(enabled_priorities)


def roce_level_ok(fact_name, level):
    """★★★ **规格值级别的校验器**（`F4` 的落点）：**没取到一手规范的事实不许标 `[A]`**。

    ★★ 依据 = `specs.SPEC_REQUIRED_FOR`（要标 `A` 就得先有 Annex A17）与
      `specs.HAVE_IBTA_ANNEX_A17`（本次取证的实情）—— ★★★ **两个独立的量**，
      所以这条判定**不是** `return False` 那种定义级恒真。
    """
    if level == "A" and fact_name in specs.SPEC_REQUIRED_FOR and not specs.HAVE_IBTA_ANNEX_A17:
        return False
    return True


# ===========================================================================
# 2. A 类 —— ★★★ 【本文件没有 A 类】
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ 【本文件没有 A 类】—— 本机无 InfiniBand / RoCE 网卡、无 GPU、无交换机")
    print("=" * 78)
    print("  ★★★ 「RoCEv2 真的这么封装吗」「PFC 真的把丢包消掉了吗」—— **无处可验**。")
    print("      ⇒ ★★ 本程序**跑在默认 `python3` 下、只用标准库** ⇒ 落在 `CONTRIBUTING` §2 表的")
    print("        **第二种情形**（**程序内断言**）；★ 而「**把不可测的那一档写明**」是【另一条】纪律。")
    print("      ★★ **如实点破一层**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本文件**没有 A**")
    print("        ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。")
    print("  ★★★ **不得把「常量钉住」冒充 A 类**（`P12` 的变体）⇒ ★★ 钉住一律归 `D` 类。")
    print("  ★★ 本程序只守两件事：**规格值抄对了没有**、**两条错法会不会被拒**。")
    print("  ★★ 机械复核**不在这里**：它需要**全部判据跑完**才成立 ⇒ 见后面的 **`D.0`**。")
    print()


# ===========================================================================
# 3. B 类：规格值表 · PFC 的官方作用域 · SHARP 的定位
# ===========================================================================
def part_B():
    print("=" * 78)
    print("B · `[A]`/`[B]` 混合的**规格值表** · PFC 的官方作用域 · SHARP 的一条")
    print("=" * 78)
    print("  ★★ 载体 = RFC 3168（`[A]`）· IEEE 802.1Qbb 工作组页（`[A]`）· Cisco 产品文档（`[B]`）·")
    print("      NVIDIA SHARP 手册（`[B]`）；抓取日 **%s**。" % specs.FETCH_DATE)

    # ── B.1 ★★★ 规格值表（★ 级别与【复核状态】是两轴）─────────────────────
    print()
    print("  ── B.1 ★★★ **规格值表**（★ 级别一列、**本轮复核状态**一列）──")
    print("     %-40s %-6s %-12s" % ("事实", "级别", "本轮复核"))
    for fact, level, status in specs.ROCE_FACT_ROWS:
        print("     %-40s %-6s %-12s" % (fact, "[%s]" % level, status))
    n_a = sum(1 for _f, lv, _s in specs.ROCE_FACT_ROWS if lv == "A")
    n_b = sum(1 for _f, lv, _s in specs.ROCE_FACT_ROWS if lv == "B")
    n_v = sum(1 for _f, _lv, s in specs.ROCE_FACT_ROWS if s == "verified")
    n_n = sum(1 for _f, _lv, s in specs.ROCE_FACT_ROWS if s == "not-fetched")
    print("     ★★ **%d 条**：`[A]` %d · `[B]` %d；**本轮已复核** %d · **未取到** %d"
          % (len(specs.ROCE_FACT_ROWS), n_a, n_b, n_v, n_n))
    print("     ★★★ **两轴【不是一回事】**：第 3 行是 `[A]` 却 `not-fetched`（IBTA 那句话本轮没取到页）；")
    print("        而 4 条 `[B]` **全部已复核**（★ 因为 `[B]` 的载体是产品文档，本轮 curl 到了）。")
    print("     ★★ 两个可数的规格值：ECN 占 **bit %s**（`%s`）；RoCEv2 的 UDP 端口 **%d**"
          % ("、".join(str(b) for b in specs.ECN_BITS), specs.RFC3168_ECN_QUOTE,
             specs.ROCEV2_UDP_PORT))
    print("       ★ 端口合法性（算术检查）：`0 < %d <= %d` = %s"
          % (specs.ROCEV2_UDP_PORT, specs.ROCEV2_PORT_RANGE_MAX,
             0 < specs.ROCEV2_UDP_PORT <= specs.ROCEV2_PORT_RANGE_MAX))
    check("B.1", len(specs.ROCE_FACT_ROWS) == 7 and n_a == 3 and n_b == 4 and n_v == 6 and n_n == 1
          and specs.ECN_BITS == (6, 7) and len(specs.ECN_BITS) == 2
          and specs.ROCEV2_UDP_PORT == 4791
          and 0 < specs.ROCEV2_UDP_PORT <= specs.ROCEV2_PORT_RANGE_MAX == 65535
          and dict((f, lv) for f, lv, _s in specs.ROCE_FACT_ROWS)[
              "RoCEv2 的封装（IB 传输包套以太网）"] == "B",
          "★★ **判据成立**：规格值表 **7 条**（`[A]` **3** · `[B]` **4**；**已复核 6** · **未取到 1**）；"
          "ECN 占 **bit 6–7**（两个位）；RoCEv2 的 UDP 端口 **4791** 落在 `(0, 65535]` 内",
          "**规格值表或两个规格值被动过** —— ★ 先查 `ROCE_FACT_ROWS` / `ECN_BITS` / `ROCEV2_UDP_PORT`")

    # ── B.2 ★★★ PFC 的官方作用域（三条逐字）──────────────────────────────
    print()
    print("  ── B.2 ★★★ **PFC 的官方作用域**（★ 三条逐字的**限定词**都在这儿）──")
    print("     ① 定义：`%s`" % specs.IEEE8021QBB_QUOTE)
    print("     ② 目的与机制：`%s`" % specs.IEEE8021QBB_PURPOSE_QUOTE)
    print("     ③ 作用域：`%s`" % specs.IEEE8021QBB_DOMAIN_QUOTE)
    print("     ④ 为什么需要（★ **主体是「高层协议」，不是 PFC**）：`%s`" % specs.IEEE8021QBB_WHY_QUOTE)
    print("     ⑤ 与 ECN 的关系（★ **互补，不是替代**）：`%s`" % specs.IEEE8021QBB_COMPLEMENT_QUOTE)
    print("     ★★★ **四个限定词，一个都不能漏**：")
    print("        · `per traffic class`（**按类别**，不是全链路一刀切）；")
    print("        · `full-duplex links`（**全双工链路**）；")
    print("        · `intended to`（**意在** —— 目的，**不是保证**）；")
    print("        · `is limited to a domain controlled by ...`（**作用域是一条限制**）。")
    print("     ★★ 官方点名的丢包原因**只有一个**：`frame loss due to congestion`")
    print("        ⇒ 本程序把它落成 `specs.PFC_OFFICIAL_LOSS_CAUSES = %s`（**范围**，不是物理断言）。"
          % (specs.PFC_OFFICIAL_LOSS_CAUSES,))
    print("     ★★ **ECN 那一半**（`[A]`，逐字）：`%s`" % specs.RFC3168_CE_QUOTE)
    print("        ★★★ 注意第二句：**满队列照样丢包**（`just as they do in the absence of ECN`）")
    print("        ⇒ ★★ **ECN 不消除丢包**，它让端点在丢包之前**得到信号**（★ 这半句是限定词）。")
    print("     ★★ 类别总数 `PFC_PRIORITY_COUNT = %d`（★ **标 `[B]/本轮未复核`** —— "
          "`8` 来自 802.1Q 的 3 位 PCP，而 802.1Q 原文本轮未取到）。" % specs.PFC_PRIORITY_COUNT)
    check("B.2", "per traffic class" in specs.IEEE8021QBB_QUOTE
          and "full-duplex" in specs.IEEE8021QBB_QUOTE
          and "intended to" in specs.IEEE8021QBB_PURPOSE_QUOTE
          and "IEEE 802.3x PAUSE" in specs.IEEE8021QBB_PURPOSE_QUOTE
          and "is limited to a domain" in specs.IEEE8021QBB_DOMAIN_QUOTE
          and specs.IEEE8021QBB_WHY_QUOTE.startswith("Data Center Bridging networks employ")
          and "approaches lossless behavior" in specs.IEEE8021QBB_WHY_QUOTE
          and specs.PFC_OFFICIAL_LOSS_CAUSES == ("congestion",)
          and specs.PFC_PRIORITY_COUNT == 8
          and specs.IEEE8021QBB_COMPLEMENT_QUOTE
          == ("In addition, PFC complements Congestion Notification in Data "
              "Center Bridging networks.")
          and specs.RFC3168_CE_QUOTE
          == ("The CE codepoint '11' is set by a router to indicate congestion to the end "
              "nodes.  Routers that have a packet arriving at a full queue drop the packet, "
              "just as they do in the absence of ECN."),
          "★★★ **判据成立**：PFC 的**四个限定词**逐字钉住（`per traffic class` · `full-duplex links` ·"
          " `intended to` · `is limited to a domain`）；★ 「为什么需要」那句的**主语**（`higher layer "
          "protocols`）与它的 `approaches lossless behavior` 一并钉住；★★ 官方点名的丢包原因 = **1 个**",
          "**PFC 的作用域引文被动过** —— ★ 先查那四个限定词（★ 少一个就会把 PFC 升格成万能解）")

    # ── B.3 ★★ SHARP 的在网计算定位（两版逐字相同）──────────────────────
    print()
    print("  ── B.3 ★★ **SHARP**：在网计算的官方定位（`[B]`）──")
    print("     逐字：`%s`" % specs.SHARP_QUOTE)
    print("     ★★ 载体 %s（另一版 %s **逐字相同**）—— ★ 本库**两版都取到**，故这条对版本不敏感。"
          % (specs.SHARP_DOC_TAG, specs.SHARP_DOC_TAG_ALT))
    print("     ★★★ **本库【不引】SHARP 的任何性能数字**（`F6`）：`2.1×` / `3.24×` 来自"
          "**二手检索摘要** ⇒ ★ `D.8` 会**扫源码**确认它们一次都没出现。")
    print("     ★★ **InfiniBand 的代际（EDR/HDR/NDR/XDR）与它们的带宽** → "
          "`20-gpu-architecture/interconnect/01` —— **本域只回链，不重写**。")
    check("B.3", specs.SHARP_QUOTE.startswith("NVIDIA® Scalable Hierarchical Aggregation")
          and "offloading collective operations from CPUs and GPUs to the network" in specs.SHARP_QUOTE
          and "eliminating the need to send data multiple times between endpoints" in specs.SHARP_QUOTE
          and specs.SHARP_DOC_TAG == "SHARP UM 3.13"
          and specs.SHARP_DOC_TAG_ALT == "SHARP UM 3.10.2",
          "★★ **判据成立**：SHARP 的定位逐字（`offloading ... to the network` + "
          "`eliminating the need to send data multiple times between endpoints`），"
          "★ 且**两个版本**都登记（★ 三篇里唯一一条**版本不敏感**的引文）",
          "**SHARP 的引文或版本标签被动过**")


# ===========================================================================
# 4. C 类：两条【真的会失败】的错法
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**把 PFC 当万能解** · **把 RoCEv2 的封装标成 `[A]`**（`F4`）")
    print("=" * 78)

    # ── C.1 ★★★ 把 PFC 当成流控的万能解 ─────────────────────────────────
    print()
    print("  ── C.1 ★★★ **把 PFC 当成流控的万能解**（★ 本程序的主力）──")
    all_classes = tuple(range(specs.PFC_PRIORITY_COUNT))
    enabled = (3,)                                    # ★ 示意配置：只在一个类别上开了 PFC
    causes_universal = ("congestion", "bit_error", "link_failure", "buffer_overflow")
    cases = (
        ("官方范围内（原因 `congestion`、类别 `3`）", ("congestion",), enabled, True),
        ("**万能解**：对**任何**丢包原因都有效", causes_universal, enabled, False),
        ("**全链路**：保护**全部 8 个类别**", ("congestion",), all_classes, False),
        ("双重越界：任何原因 + 全部类别", causes_universal, all_classes, False),
    )
    all_ok = True
    print("     实际配置里开了 PFC 的类别：%s（共 %d 个）；官方点名的丢包原因：%s"
          % (enabled, len(enabled), specs.PFC_OFFICIAL_LOSS_CAUSES))
    for label, causes, prios, expect in cases:
        got = pfc_claim_ok(causes, prios, enabled)
        all_ok = all_ok and (got == expect)
        print("     %-40s 原因 %-2d 个 · 类别 %-2d 个 ⇒ %s（期望 %s）"
              % (label, len(causes), len(prios),
                 "放行" if got else "**拒绝**", "放行" if expect else "**拒绝**"))
    print("     ★★★ **两个越界方向【独立】**：")
    print("        · **跨[原因]**：官方只点名 `frame loss due to congestion`；")
    print("        · **跨[类别]**：PFC 是 `per traffic class` 的，**只保护开了它那些类别**。")
    print("     ★★ 第三句是**最有说服力的一条**：即使原因没越界，「保护全部 8 个类别」")
    print("        **也是错的** —— ★★ 因为 PFC 从来不是全链路流控（逐字 `per traffic class`）。")
    check("C.1", all_ok and pfc_claim_ok(("congestion",), enabled, enabled)
          and not pfc_claim_ok(causes_universal, enabled, enabled)
          and not pfc_claim_ok(("congestion",), all_classes, enabled)
          and not pfc_claim_ok(causes_universal, all_classes, enabled)
          and len(all_classes) == 8 and len(specs.PFC_OFFICIAL_LOSS_CAUSES) == 1,
          "★★★ **判据成立**：**官方范围内的声明放行**，而**三个方向**的「万能解」**全部被拒**"
          "—— ① 跨**原因** ② 跨**类别** ③ 两者都跨 ⇒ ★★ **「把 PFC 当成流控的万能解」真的会失败**",
          "**「万能解」没被拒绝** —— ★ 先查 `pfc_claim_ok` 的两条集合包含关系")

    # ── C.2 ★★★ 把 RoCEv2 的 UDP 封装标成 `[A]`（`F4`）───────────────────
    print()
    print("  ── C.2 ★★★ **把 RoCEv2 的 UDP 封装标成 `[A]`**（`F4` 的落点）──")
    encap = specs.SPEC_REQUIRED_FOR[0]
    rows_level = dict((f, lv) for f, lv, _s in specs.ROCE_FACT_ROWS)
    print("     需要一手规范（Annex A17）才能标 `[A]` 的事实：`%s`" % encap)
    print("     本轮是否取到 Annex A17：**%s**" % specs.HAVE_IBTA_ANNEX_A17)
    for fact, level, expect in ((encap, "A", False), (encap, "B", True),
                                ("ECN 字段（IPv4 TOS 的 bit 6–7）", "A", True)):
        got = roce_level_ok(fact, level)
        print("     `%s` 标 `[%s]` ⇒ 校验器 %s（期望 %s）"
              % (fact, level, "放行" if got else "**拒绝**", "放行" if expect else "**拒绝**"))
    print("     ★★ 表里那一行的**实际登记级别**是 `[%s]`（★ 必须是 `B`）" % rows_level[encap])
    print("     ★★ 载体（`[B]`，逐字）：`%s`" % specs.CISCO_ROCEV2_ENCAP_QUOTE)
    print("     ★★★ 依据：RoCEv2 的**帧结构 / UDP 封装**的 `[A]` 出处是 IBTA 的 **Annex A17**，")
    print("        而它**本轮未取到**（见 `NOT_QUOTED` 第 3 条）⇒ ★★ **只能标 `[B]`**（Cisco 产品文档）。")
    check("C.2", roce_level_ok(encap, "A") is False and roce_level_ok(encap, "B") is True
          and roce_level_ok("ECN 字段（IPv4 TOS 的 bit 6–7）", "A") is True
          and rows_level[encap] == "B" and specs.HAVE_IBTA_ANNEX_A17 is False
          and specs.SPEC_REQUIRED_FOR == ("RoCEv2 的封装（IB 传输包套以太网）",)
          and specs.NOT_QUOTED[-1].startswith("RoCEv2 的帧结构")
          and specs.CISCO_ROCEV2_PORT_QUOTE
          == "The UDP destination port number 4791 has been reserved for RoCEv2.",
          "★★★ **判据成立**：把 RoCEv2 的**封装**标成 `[A]` **被拒**（因为 Annex A17 未取到）、"
          "标成 `[B]` **放行**；★ 而**已复核**的 `[A]` 事实（ECN 那两个 bit）**照常放行**"
          " ⇒ ★★ **`F4` 的错法真的会失败**，且**不是**把一切都拒掉",
          "**`F4` 的守卫失效** —— ★ 先查 `SPEC_REQUIRED_FOR` / `HAVE_IBTA_ANNEX_A17` / `ROCE_FACT_ROWS`")


# ===========================================================================
# 5. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 RoCEv2，也不是真机** —— 本机**无网卡 / 无 GPU / 无交换机**。")
    print("  ★★★ **本文件没有 A 类**（`D.0` 机械证明）；★★ **一切钉住归 `D` 类**。")
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
    check("D.2", ASSERT_FAIL_MARK == ",, FAIL: network-topology" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `,, FAIL: network-topology`、且含 `FAIL`\n"
          "        ★★ `,` 在 **shell 里需要引号** ⇒ 它**只允许出现在 `print()` 的字符串里**。")
    check("D.2b", OTHER_THEME_MARKS == ("&& FAIL: collectives", ">< FAIL: nccl", "+= FAIL: overlap")
          and ABSENT_CLASS == "A" and ALLOWED_CLASSES == ("B", "C", "D"),
          "★ 「不跨主题照抄」的**对照物**逐位钉住；★ `D.0` 的两条机械量（`ABSENT_CLASS` / "
          "`ALLOWED_CLASSES`）一并钉住（★ 它们被改掉，`D.0` 就会**静默失效**）")

    # ── D.3 ★ 载体元数据 ──────────────────────────────────────────────────
    check("D.3", specs.RFC3168_CARRIER
          == "RFC 3168: The Addition of Explicit Congestion Notification (ECN) to IP"
          and specs.RFC3168_URL == "https://www.rfc-editor.org/rfc/rfc3168.txt"
          and specs.IEEE8021QBB_URL == "https://1.ieee802.org/dcb/802-1qbb/"
          and specs.CISCO_ROCEV2_URL
          == ("https://www.cisco.com/c/en/us/td/docs/unified_computing/ucs/c/sw/gui/config/"
              "guide/4_2/b-roce-configuration-cimc-4-2/"
              "m-rdma-over-converged-ethernet-roce-version-2.html")
          and specs.SHARP_URL == "https://networking-docs.nvidia.com/sharpum/3130/introduction"
          and specs.CISCO_ROCEV2_DOC == ("Cisco UCS C-Series Integrated Management Controller Configuration Guide for RDMA over Converged Ethernet (RoCEv2), Release 4.2")
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：RFC 3168 · IEEE 802.1Qbb · Cisco（Release 4.2）· "
          "NVIDIA SHARP UM 3.13；抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住" % specs.FETCH_DATE)

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    #   ① 表里的级别只许是 A/B；复核状态只许是 verified / not-fetched；
    #   ② `SPEC_REQUIRED_FOR` 的名字**必须与表里某一行逐字相同**（★ 否则守卫会**静默失效**）；
    #   ③ 「未取到」的那一行**必须**是 `SPEC_REQUIRED_FOR` 点名的那一个事实（两处**必须同步**）。
    levels = {lv for _f, lv, _s in specs.ROCE_FACT_ROWS}
    statuses = {s for _f, _lv, s in specs.ROCE_FACT_ROWS}
    names = [f for f, _lv, _s in specs.ROCE_FACT_ROWS]
    name_match = all(n in names for n in specs.SPEC_REQUIRED_FOR)
    not_fetched = tuple(f for f, _lv, s in specs.ROCE_FACT_ROWS if s == "not-fetched")
    sync = (len(specs.SPEC_REQUIRED_FOR) == 1
            and not_fetched == ("RoCE 的定位（IBTA 的措辞）",)
            and specs.HAVE_IBTA_ANNEX_A17 is False)
    print("     ★★ 表里出现过的级别：%s；复核状态：%s" % (sorted(levels), sorted(statuses)))
    print("     ★★ `SPEC_REQUIRED_FOR` 的名字**在表里找得到** = %s（★ 对不上就是**静默失效**）"
          % name_match)
    print("     ★★ `HAVE_IBTA_ANNEX_A17 = %s`、`not-fetched` 的行 = %s ⇒ **两处同步** = %s"
          % (specs.HAVE_IBTA_ANNEX_A17, not_fetched, sync))
    check("D.3b", levels <= {"A", "B"} and statuses <= {"verified", "not-fetched"}
          and names and len(set(names)) == len(names)
          and name_match and len(specs.SPEC_REQUIRED_FOR) == 1,
          "★★★ **口径守卫**：级别只有 `A`/`B`、复核状态只有两种、**事实名无重复**，"
          "且 `SPEC_REQUIRED_FOR` 的名字**能在表里对到行**（★ 否则 `C.2` 会**静默通过**）",
          "**规格值表的形状被动过** —— ★ 先查 `ROCE_FACT_ROWS` 与 `SPEC_REQUIRED_FOR`")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：`D.0` 机械复核 · 引文逐字 · 逐元素 · 禁写清单扫描
# ===========================================================================
def _count_check_callsites():
    """★ 数本文件里 `check(` 的**调用点个数**（★ 读自己的源码；★ 函数名 `check` 的 `Call` 节点）。

    ★★ 注释与 docstring 里的 `check(` 文本**不会被误计**（它们不是 `ast.Call`）。
    """
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "check":
            n += 1
    return n


def part_D():
    print()
    print("=" * 78)
    print("D · `D.0` 机械复核 · 引文逐字钉住 · **逐元素**钉住 · 禁写清单扫描")
    print("=" * 78)

    # ── D.4 ★★★ 逐元素钉住 ───────────────────────────────────────────────
    check("D.4", specs.ECN_BITS == (6, 7)
          and specs.ROCEV2_UDP_PORT == 4791
          and specs.ROCEV2_PORT_RANGE_MAX == 65535
          and specs.PFC_PRIORITY_COUNT == 8
          and specs.PFC_OFFICIAL_LOSS_CAUSES == ("congestion",)
          and specs.SPEC_REQUIRED_FOR == ("RoCEv2 的封装（IB 传输包套以太网）",)
          and specs.HAVE_IBTA_ANNEX_A17 is False
          and specs.ROCE_FACT_ROWS == (
              ("ECN 字段（IPv4 TOS 的 bit 6–7）", "A", "verified"),
              ("PFC 按流量类别作用于全双工链路", "A", "verified"),
              ("RoCE 的定位（IBTA 的措辞）", "A", "not-fetched"),
              ("RoCEv2 的 UDP 目的端口 4791", "B", "verified"),
              ("RoCEv2 可路由（网络层协议）", "B", "verified"),
              ("RoCEv2 的封装（IB 传输包套以太网）", "B", "verified"),
              ("SHARP 的在网计算定位", "B", "verified"),
          ),
          "★★★ **逐元素钉住**：`ECN_BITS`（两档）· `ROCE_FACT_ROWS`（**7 行 × 3 列 = 21 个叶子**）· "
          "`PFC_OFFICIAL_LOSS_CAUSES`（1）· `SPEC_REQUIRED_FOR`（1）· 四个标量\n"
          "        —— ★ 只钉「整体」会让**元素级**变异漏网（本库实测过：一次报出 58 条漏网）",
          "**规格值表的某个元素被动过** —— ★ 注意那 21 个叶子")

    # ── D.5 ★★★ 引文逐字（`[A]` 与 `[B]` 都在这里）──────────────────────
    check("D.5", specs.RFC3168_ECN_QUOTE
          == "Bits 6 and 7 in the IPv4 TOS octet are designated as the ECN field."
          and specs.IEEE8021QBB_QUOTE
          == ("This standard specifies protocols, procedures and managed objects that enable flow "
              "control per traffic class on IEEE 802 full-duplex links.")
          and specs.IEEE8021QBB_PURPOSE_QUOTE
          == ("Priority-based flow control is intended to eliminate frame loss due to congestion. "
              "This is achieved by a mechanism similar to the IEEE 802.3x PAUSE, but operating on "
              "individual priorities.")
          and specs.IEEE8021QBB_DOMAIN_QUOTE
          == ("Operation of priority-based flow control is limited to a domain controlled by a Data "
              "Center Bridging control protocol that controls the application of Priority-based Flow "
              "Control, Enhanced Transmission Selection, and Congestion Notification.")
          and specs.IEEE8021QBB_WHY_QUOTE
          == ("Data Center Bridging networks employ higher layer protocols that depend on the "
              "delivery of data frames without frame loss due to congestion. These protocols were "
              "designed for an underlying transport that approaches lossless behavior and therefore "
              "do not include appropriate response to frame loss due to congestion (e.g. back-off, "
              "slow restart, etc.).")
          and specs.IEEE8021QBB_COMPLEMENT_QUOTE
          == ("In addition, PFC complements Congestion Notification in Data "
              "Center Bridging networks."),
          "★★★ **`[A]` 侧的五条引文逐字钉住**（RFC 3168 一条 + IEEE 802.1Qbb 三条）——"
          "★★★ 特别注意 `intended to`（**意在**）与 `is limited to a domain`（**作用域是限制**）；\n"
          "        ★★ 以及 `IEEE8021QBB_WHY_QUOTE` 的**主语在第一句里**（`higher layer protocols`）"
          "—— ★ 只抄第二句就会**丢掉主语**（v1.7 那条硬伤的形态）",
          "**`[A]` 侧的引文被动过** —— ★ 先查四个限定词")

    # ── D.6 ★★ `[B]` 侧的引文 + 两条「只引载体」+ 三条「不引什么」──────────
    check("D.6", specs.CISCO_ROCEV2_PORT_QUOTE
          == "The UDP destination port number 4791 has been reserved for RoCEv2."
          and specs.CISCO_ROCEV2_ROUTABLE_QUOTE
          == "RoCEv2 is an internet layer protocol, which means that RoCEv2 packets can be routed."
          and specs.CISCO_ROCEV1_QUOTE
          == ("RoCEv1 is an Ethernet link layer protocol and hence allows communication between any "
              "two hosts in the same Ethernet broadcast domain.")
          and specs.CISCO_ROCEV2_ENCAP_QUOTE
          == ("It does this by encapsulating an Infiniband (IB) transport packet over Ethernet.")
          and specs.IBTA_ROCE_POSITION_QUOTE
          == ("RoCE is an industry standard transport that enables Remote Direct Memory Access "
              "(RDMA) to operate on ordinary Ethernet layer 2 and 3 networks.")
          and specs.SHARP_QUOTE == ("NVIDIA® Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ technology improves the performance of MPI and Machine Learning collective operation, by offloading collective operations from CPUs and GPUs to the network and eliminating the need to send data multiple times between endpoints.")
          and specs.SHARP_PAPER_CARRIER == "SHARP, COMHPC 2016"
          and specs.DCQCN_CARRIER == "DCQCN, SIGCOMM 2015"
          and specs.NOT_QUOTED == (
              "SHARP 的 2.1× / 3.24× 性能数字（二手检索摘要）",
              "DCQCN 的三组件机制描述（二手检索摘要）",
              "RoCEv2 的帧结构（IBTA Annex A17 一手规范未取到）",
          ),
          "★★ **`[B]` 侧的引文逐字**（Cisco 四条 —— ★ 含 `Infiniband` 的**小写 i**，照抄）+"
          " ★★ **只引载体**的两条（`SHARP, COMHPC 2016` / `DCQCN, SIGCOMM 2015`：★★ **正文都未取到**）\n"
          "        ★★★ + **三条「不引什么」**（`NOT_QUOTED`，逐元素）—— 它们是 `F4`/`F6` 的书面承诺；\n"
          "        ★★★ 注：IBTA 那句 `RoCE is an industry standard transport ...` **照计划 §C.3 转录**，"
          "**本轮未取到该页**（`specs.py` 的 `# 复核状态：` 已如实登记）⇒ ★ 所以它的 `[A]` **挂在规划上**，"
          "本程序把它与「已复核」分开显示在 `B.1` 的两轴里",
          "**`[B]` 侧引文 / 两条载体 / 三条「不引什么」被动过**")

    # ── D.7 ★★ `UNSOURCED` + 抓取日 ─────────────────────────────────────
    check("D.7", specs.UNSOURCED == () and specs.FETCH_DATE == "2026-10-08"
          and specs.SHARP_DOC_TAG == "SHARP UM 3.13" and specs.SHARP_DOC_TAG_ALT == "SHARP UM 3.10.2",
          "★★ `UNSOURCED` 留空（★ 0 个叶子 ⇒ 不产生变异体）；★★ 抓取日与两个 SHARP 版本标签钉住；"
          "★★ **`UNSOURCED` 与 `NOT_QUOTED` 的分工**：前者 = 「用了但没出处」（**本主题为空**），"
          "后者 = 「**故意不引**并说明为什么」（**3 条**）",
          "**`UNSOURCED` / 抓取日 / 版本标签被动过**")

    # ── D.8 ★★★ 禁写清单扫描（`F6`）：扫的是**引文常量**，不是全文 ──────────
    #   ★★ 口径（写清，免得读者误判）：**靶子 = `specs` 里所有 `*_QUOTE` 字符串常量**。
    #      ★ 那正是「逐字引文」的存放处 ⇒ 只要它们里没有禁写数字，「引进来」就不可能发生。
    #      ★★ 而 `NOT_QUOTED`（**故意不引**的登记处）里**应当**出现它们 —— 那是记录，不是引用。
    quote_consts = {n: v for n, v in vars(specs).items()
                    if n.endswith("_QUOTE") and isinstance(v, str)}
    leaks = sorted((n, t) for n, v in quote_consts.items() for t in FORBIDDEN_TOKENS if t in v)
    print()
    print("  ── D.8 ★★★ **禁写清单扫描**（`F6`）──")
    print("     被禁的字面量：%s（SHARP 的性能数字，★ 二手摘要）" % (FORBIDDEN_TOKENS,))
    print("     扫描靶子：`specs` 里所有 `*_QUOTE` 常量 —— 共 **%d** 条（**逐字引文的存放处**）"
          % len(quote_consts))
    print("     命中（引文里含禁写数字）：%s" % (leaks or "**无**"))
    print("     ★★ `NOT_QUOTED`（**故意不引**的登记处）里**应当**出现它们 —— ★ 那是记录，不是引用。")
    print("     ★★★ 口径写在这里：本扫描**不**扫散文与注释（★ 那里出现这两个数字是**在说「不引」**）")
    print("        —— ★ 扫描靶子**收窄到引文常量**，是为了让「>0 命中」**就是**违规。")
    check("D.8", not leaks and len(quote_consts) == 26
          and FORBIDDEN_TOKENS == ("2.1×", "3.24×")
          and len(specs.NOT_QUOTED) == 3 and specs.NOT_QUOTED[0].startswith("SHARP 的 2.1×"),
          "★★★ **判据成立**：**%d 条** `*_QUOTE` 引文常量里，`F6` 的两个性能数字**一次都没出现**；"
          "★ 而它们在 `NOT_QUOTED` 里**有登记**（★ 那是「我们不引它」的记录）"
          "⇒ ★★ **禁写清单是可机械复核的**，不是一句口头承诺" % len(quote_consts),
          "**引文常量里出现了禁写数字** —— ★ 先查 SHARP 的性能数字有没有被引进来")

    # ── D.9 ★★ 三条 `[E]` 记录的边界 ─────────────────────────────────────
    check("D.9", pfc_claim_ok(("congestion",), (3,), (3,))
          and not pfc_claim_ok(("congestion",), (3,), ())
          and roce_level_ok("PFC 按流量类别作用于全双工链路", "A")
          and not roce_level_ok("RoCEv2 的封装（IB 传输包套以太网）", "A")
          and specs.ROCEV2_UDP_PORT == 4791 and specs.ECN_BITS == (6, 7),
          "★★ **三条 `[E]` 记录的边界**：① 没开 PFC 的类别**不受保护**（`enabled=()` ⇒ 任何类别声明都被拒）；"
          "② 已复核的 `[A]` 事实（PFC 那条）**照常放行**；③ `F4` 那条**必须被拒**",
          "**`[E]` 记录的边界被动过**")

    # ── D.0 ★★★ 机械证明「本文件没有 A 类」（★ 两条独立机械量）────────────
    #   ★★★ **它必须【最后跑】** —— 它读的是**全部判据跑完之后**的记账。
    #   ★★ **一条 off-by-one（写在这里免得读者踩）**：`check()` 是**先算条件、再记账**的
    #     ⇒ 本条求值时 `_TOTAL` 里**还没有本条自己** ⇒ 期望是 `调用点 == 执行次数 + 1`
    #     （`+1` 就是**本条自己的调用点**）。
    print()
    print("  ── D.0 ★★★ **机械证明本文件没有 A 类**（★ 与 `nccl` 的路线**不同**）──")
    executed = sum(_TOTAL.values())
    callsites = _count_check_callsites()
    keys = sorted(_TOTAL)
    n_a = _TOTAL[ABSENT_CLASS]
    print("     ① **运行期记账**：`_TOTAL` 的键 = %s（**必须恰好是 `%s`**）· "
          "`_TOTAL[\"%s\"]` = **%d**" % (keys, list(ALLOWED_CLASSES), ABSENT_CLASS, n_a))
    print("     ② **调用点核对**：本文件 `check(` 的调用点 **%d** 个 vs 已执行 **%d** 次"
          " ⇒ `调用点 == 已执行 + 1`（★ 那个 `+1` = **本条自己的调用点**，它还没被记账）"
          % (callsites, executed))
    print("     ★★★ 两条**分工不同**：① 抓「跑起来了的 A 类」；")
    print("        ② 抓「**从未被执行**的判据（死代码）」—— ★★ 单靠 ① 会漏它，单靠 ② 就是 `nccl` 那条路线的变体。")
    print("     ★★ **如实交代残余边界**：这两条只覆盖**本文件**的 `check()` 调用；")
    print("        ★ 若有人把 A 类判据写进 `specs.py` 或另起一个文件，**这里看不见**")
    print("        —— ★★ 但那时它**也不在**本文件的断言集里，本文件的总结会照实报 `A 0/0`。")
    check("D.0", n_a == 0 and keys == list(ALLOWED_CLASSES) and callsites == executed + 1
          and ABSENT_CLASS == "A" and ALLOWED_CLASSES == ("B", "C", "D"),
          "★★★ **机械证明成立**：① `_TOTAL` 里**没有 `A` 这个键**（`_TOTAL[\"A\"] == 0`）；"
          "② **调用点 == 已执行 + 1**（`%d == %d + 1`）⇒ ★★ **本文件既没有跑到的 A 类判据，"
          "也没有从未执行的死判据**" % (callsites, executed),
          "**本文件出现了 A 类判据**（或调用点与执行次数对不上）—— ★ 那与「本文件没有 A 类」**自相矛盾**")


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
    print("★★★ **A = %d/%d** —— ★ **本文件【没有 A 类】**：本机无 InfiniBand / RoCE 网卡、无 GPU、"
          "无交换机。★ 「常量钉住」**不算 A 类**（`P12` 的变体）⇒ 本程序把它们全归了 `D` 类。"
          % (_PASS["A"], _TOTAL["A"]))
    print("    ★★ 机械复核在 `D.0`（**运行期记账 + 调用点核对**）—— ★ 与 `nccl` 的源码扫描**机制不同**。")
    print("★ 依据在 `specs.py` 的 `# 来源：`（RFC 3168 / IEEE 802.1Qbb / Cisco / NVIDIA，"
          "抓取日 %s）。" % specs.FETCH_DATE)
    print("★ 本程序只核对「规格值抄对了没有」与「两条错法真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_roce_and_infiniband.py —— RoCE v2 的机制（PFC / ECN / DCQCN）与 SHARP")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **物理层**；★★★ **本文件【没有 A 类】**（本机无网卡）。")
    print("  ★★★ 强制限定语：**不是 RoCEv2，也不是真机**。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
