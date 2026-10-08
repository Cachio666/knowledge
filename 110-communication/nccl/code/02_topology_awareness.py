#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_topology_awareness.py —— **NCCL 的拓扑感知**：官方暴露了哪几条抓手。

配套文档：`02-topology-awareness.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题【没有 A 类】**：本机**无 GPU、无 NCCL、无真网卡** ⇒「算法正确性」**无处可验**。
  ★ 本程序**跑在默认 `python3` 下、只用标准库** ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**
  （**程序内断言**）。
  ★★ **如实点破一层**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**
  ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。
  ★★ 而「**把不可测的那一档写明**」是【另一条】纪律（**不是** §2 的任何一种编号形态）。
  ★★★ **不得把「常量钉住」冒充 A 类**（那是 `P12` 的变体）⇒ ★★ 本程序的一切钉住归 `D` 类。
  ★ `D.0` **机械地**证明「本文件里一条 A 类判据都没有」（AST 扫自己）。

★★★ **与 `20-gpu-architecture/interconnect/03:243-249` 的【分工】—— 本程序一行都不重写它**：
  那一处是一段 **7 行**的**定性描述**（ring / tree / collnet-NVLS + 「由探测拓扑决定」），
  ★ **而且不止它一处**（`interconnect/04:222` · `interconnect/02:192` ·
  `performance-models/02:124` 也各有一句，**都停在定性**）。
  ⇒ ★★★ **本主题只写它【没写】的**：**变量名 · XML · DEBUG 子系统**这三条**官方抓手**。
  ★★ 主线不同：那边讲「**有哪些算法、各对应什么拓扑假设**」，
  本程序讲「**官方把哪几条旋钮露出来了、每条的能力是什么**」。

★★★ **本程序最要紧的一条区分**：**「感知拓扑」≠「控制选路」** ——
  官方暴露的**三条**抓手（`NCCL_TOPO_FILE` / `NCCL_DEBUG_SUBSYS=GRAPH` / `=TUNING`）里，
  **没有一条**的能力是「控制选路」（`C.1`/`C.2` 判它）。
  ★★★ **这是本主题最容易写错的一条**：把「NCCL 会读拓扑」顺手写成「你可以让它走某条路」。

★★ **覆盖边界**：本程序守的是「**抓手有哪几条、各自的能力标签对不对、错法会不会被拒**」。
  **不守**任何一条**真实 NCCL 的选路行为**（本机**测不了**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `B.1`  ★★ **官方暴露的抓手【数】= 3**（逐条可数），且**没有一条**的能力是「控制选路」；
  · `B.2`  ★★ 两条 **offload 开关**与它们的**默认值**（★ 默认**不是** `unset`）；
  · `C.1`  ★★★ **把「感知拓扑」写成「能控制选路」**必须被拒；
  · `C.2`  ★★★ **把 `NCCL_DEBUG_SUBSYS=GRAPH` 当成「能改选路」**必须被拒；
  · `D.0`  ★★★ **机械证明「本文件没有 A 类判据」**；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **引文逐字**钉住 + **失败标记自检**。
"""
import ast
import collections
import sys

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（nccl）取 `>< FAIL: nccl`（★ 三个程序**同一个**标记）。
#   ★ 本域另外三个主题各用**不同的前缀**（`&&` / `,,` / `+=`），四个标记**两两不互为前缀**。
#   ★★ `>` 是 **shell 的重定向操作符** ⇒ 它**只允许出现在 `print()` 的字符串里**。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ">< FAIL: nccl"

# 本域另外三个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("&& FAIL: collectives", ",, FAIL: network-topology", "+= FAIL: overlap")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
    ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 失败标记必然在**行首**。
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
# 1. 被测的「能力模型」—— ★★ **它【不是】A 类**（本机没有 NCCL 可测）
#    ★★★ 它是**本库的一段校验器**：把官方暴露的抓手按**能力**分类，用来判
#      「把感知写成控制」这类错法。★ 一切钉住归 `D` 类。
# ===========================================================================
def cap_of(handle):
    """取某条抓手的**能力标签**（`INPUT` / `OBSERVE`）—— ★ 未知抓手抛 `KeyError`。

    ★★ `D.3b` 守着「抓手表与能力表**同名同序**」这个前提，所以查表是安全的。
    """
    return dict(specs.NCCL_TOPO_HANDLE_CAPS)[handle]


def cap_matches(handle, claimed):
    """★ 某条抓手的**能力**是否**就是** `claimed` —— `True` = 这个断言是对的。"""
    return cap_of(handle) == claimed


def can_control_routing(handle):
    """★★★ **官方暴露的抓手【没有】一条是「控制选路」的能力** —— 本函数对**全部**抓手返回 `False`。

    ★★ 官方给的三条能力只有两种：`INPUT`（检测的输入）与 `OBSERVE`（观测窗口）
      ⇒ **`ROUTE` 这个能力在官方暴露面上【不存在】**（`NCCL_ALGO` 的默认是 `unset`
      ⇒ 选路是 NCCL **自己算**的，见 `01` 的 `B.2`）。
    """
    return cap_of(handle) == "ROUTE"


# ===========================================================================
# 2. A 类 —— ★★★ 【本主题没有 A 类】
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ 【本主题没有 A 类】—— 本机无 GPU / 无 NCCL / 无真网卡")
    print("=" * 78)
    print("  ★★★ 「NCCL 到底怎么感知拓扑、感知到拓扑之后怎么选」**无处可验**：")
    print("      本机既没有 NCCL 也没有 GPU、更没有一张真网卡。")
    print("      ⇒ 本程序**跑在默认 `python3` 下** ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**")
    print("        （**程序内断言**）；★ 而「**把不可测的那一档写明**」是【另一条】纪律（不是 §2 的形态）。")
    print("      ★★ **如实点破**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**")
    print("        ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。")
    print("  ★★★ **不得把「常量钉住」冒充 A 类**（`P12` 的变体）⇒ ★★ 钉住一律归 `D` 类。")
    print("  ★★ 本程序只守一件事：**官方暴露的抓手有哪几条、各自的能力标签对不对**。")
    print()
    n_a = _count_a_checks_in_source()
    print("  ── D.0 ★★★ **机械证明**：AST 扫本文件，`check(\"A...\")` 的调用数 = **%d** ──" % n_a)
    check("D.0", n_a == 0,
          "★★★ **判据成立**：本文件里**一条 A 类判据都没有**（AST 逐个数 `check()` 的"
          "第一实参，字面量以 `A.` 开头者 = **0**）",
          "★★ 本文件里出现了 A 类判据 —— ★ 那与「本主题没有 A 类」**自相矛盾**")


def _count_a_checks_in_source():
    """★ 数本文件里 `check("A...", ...)` 的调用数（**读自己的源码**，不是写死的数）。"""
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "check" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                    and a0.value.startswith("A."):
                n += 1
    return n


# ===========================================================================
# 3. B 类：**官方暴露的抓手【数】** 与 offload 开关
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `[B]` 级的账：**官方暴露的抓手【数】= 3** · offload 开关")
    print("=" * 78)
    print("  ★★ 载体 = NCCL 官方产品文档 `env.html` / `troubleshooting/logging.html`，")
    print("      版本 **%s**，抓取日 **%s**。" % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))
    print("  ★★ **分工**：`interconnect/03:243-249`（7 行定性描述）讲「有哪些算法、各假设什么拓扑」；")
    print("      ★★★ **本程序只写它没写的**：**变量名 · XML · DEBUG 子系统**这三条**官方抓手**。")

    # ── B.1 ★★ 抓手数 = 3（逐条可数）+ 能力表里没有「控制选路」─────────────
    print()
    print("  ── B.1 ★★ **官方暴露的抓手（逐条可数）** ──")
    for h in specs.NCCL_TOPO_HANDLES:
        print("     %-26s 能力 = **%s**" % (h, cap_of(h)))
    no_route = all(cap_of(h) != "ROUTE" for h in specs.NCCL_TOPO_HANDLES)
    caps_seen = tuple(cap_of(h) for h in specs.NCCL_TOPO_HANDLES)
    print("     ★★ **抓手数 = %d**；★★ **能力取值域 = %s**（★ **没有** `ROUTE`）"
          % (len(specs.NCCL_TOPO_HANDLES), sorted(set(caps_seen))))
    print("     ★★★ 读法：官方给了**两条观测**（`GRAPH` / `TUNING`）与**一条输入**"
          "（`TOPO_FILE`），★ **没有一条**是「指定某对设备走哪条路」。")
    check("B.1", len(specs.NCCL_TOPO_HANDLES) == specs.NCCL_TOPO_HANDLE_COUNT == 3
          and caps_seen == ("INPUT", "OBSERVE", "OBSERVE")
          and no_route
          and specs.NCCL_DEBUG_SUBSYS_GRAPH in specs.NCCL_TOPO_HANDLES[1]
          and specs.NCCL_DEBUG_SUBSYS_TUNING in specs.NCCL_TOPO_HANDLES[2],
          "★★★ **判据成立**：官方暴露的抓手**恰好 3 条**（`NCCL_TOPO_FILE` ·"
          " `NCCL_DEBUG_SUBSYS=GRAPH` · `=TUNING`），★★ **没有一条的能力是「控制选路」**"
          "（能力只有 `INPUT` / `OBSERVE`）",
          "**抓手表或能力表被动过** —— ★ 先查 `NCCL_TOPO_HANDLES` / `NCCL_TOPO_HANDLE_CAPS`")

    # ── B.2 ★★ offload 开关（默认值不是 unset）───────────────────────────
    print()
    print("  ── B.2 ★★ **offload 开关**与它们的**默认值** ──")
    for i, nm in enumerate(specs.NCCL_OFFLOAD_SWITCHES):
        since = specs.NCCL_NVLS_ENABLE_SINCE if i == 0 else specs.NCCL_COLLNET_ENABLE_SINCE
        print("     %-22s 默认 = `%s` · since = `%s`"
              % (nm, specs.NCCL_OFFLOAD_DEFAULTS[i], since))
    print("     ★★ 读法：默认 `0`（`COLLNET` 关闭）与默认 `2`（`NVLS` 允许）**都不是 `unset`**")
    print("       ⇒ ★★ **与 `NCCL_ALGO` 的默认（`unset`）形态不同** —— 别把两类变量的默认混为一谈。")
    print("     ★★ 这两条是**开关**（允许不允许用某条路径），★ **不决定**哪条路径承载哪笔消息。")
    print("     ★★★ **修正**：`NCCL_COLLNET_ENABLE` 的 `since` 官方**逐字给了 `2.6`** ——")
    print("       `%s`" % specs.NCCL_COLLNET_ENABLE_QUOTE)
    print("       ★ 初稿把它当成「**官方未给**」（空串占位）⇒ ★★ 那是把**「规划没记」当成"
          "「官方没有」**。")
    check("B.2", specs.NCCL_OFFLOAD_SWITCHES == ("NCCL_NVLS_ENABLE", "NCCL_COLLNET_ENABLE")
          and specs.NCCL_OFFLOAD_DEFAULTS == ("2", "0")
          and specs.NCCL_NVLS_ENABLE_SINCE == "2.17"
          and specs.NCCL_COLLNET_ENABLE_SINCE == "2.6",
          "★★ **判据成立**：两条 offload 开关（`NCCL_NVLS_ENABLE` 默认 `2`，since `2.17`；"
          "`NCCL_COLLNET_ENABLE` 默认 `0`，since `2.6`）—— "
          "★★ **默认都不是 `unset`**（与 `NCCL_ALGO` 形态不同）；★ 两条的 `since` 官方**都给了**",
          "**offload 开关或默认值被动过**")


# ===========================================================================
# 4. C 类：两条【真的会失败】的错法（把感知写成控制）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**把「感知拓扑」写成「能控制选路」**")
    print("=" * 78)
    print("  ★★★ 本主题最容易写错的一条就在这里 —— 官方暴露的三条抓手**没有一条**能控制选路。")

    # ── C.1 ★★★ 把 `NCCL_TOPO_FILE` 写成「能控制选路」──────────────────
    print()
    print("  ── C.1 ★★★ **把 `NCCL_TOPO_FILE` 写成「能控制选路」** ──")
    print("     声称 `NCCL_TOPO_FILE` 的能力 = `ROUTE` ⇒ 能力表**拒绝**？ **%s**（★ 必须是 `False`）"
          % cap_matches("NCCL_TOPO_FILE", "ROUTE"))
    print("     而文档逐字说它是 **`%s`**" % specs.NCCL_TOPO_FILE_QUOTE)
    print("     ★★★ 关键在措辞 **`load before detecting the topology`** —— 它是**检测的输入**，")
    print("       ★ **不是**「指定每对设备走哪条路」。★ **选路是 NCCL 自己算的**"
          "（`NCCL_ALGO` 默认 `unset`）。")
    check("C.1", (not cap_matches("NCCL_TOPO_FILE", "ROUTE"))
          and (not can_control_routing("NCCL_TOPO_FILE"))
          and cap_matches("NCCL_TOPO_FILE", "INPUT"),
          "★★★ **判据成立**：把 `NCCL_TOPO_FILE` 说成「能控制选路」（`ROUTE`）会被能力表**拒绝**；"
          "它的能力是 **`INPUT`**（检测的输入）⇒ ★★ **「感知拓扑 ≠ 控制选路」真的会失败**",
          "**「控制选路」这个错法没被拒绝** —— ★ 先查 `NCCL_TOPO_HANDLE_CAPS` 的能力取值域")

    # ── C.2 ★★★ 把 `NCCL_DEBUG_SUBSYS=GRAPH` 当成「能改选路」────────────
    print()
    print("  ── C.2 ★★★ **把 `NCCL_DEBUG_SUBSYS=GRAPH` 当成「能改选路」** ──")
    g = "NCCL_DEBUG_SUBSYS=GRAPH"
    print("     声称 `%s` 的能力 = `ROUTE` ⇒ 能力表**拒绝**？ **%s**（★ 必须是 `False`）"
          % (g, cap_matches(g, "ROUTE")))
    print("     ★★★ 它是一条**观测**（能力 = `%s`）：`NCCL_DEBUG_SUBSYS` 是按子系统**输出日志**的"
          "设施" % cap_of(g))
    print("       ★★ 逐字旁证：`troubleshooting/logging.html` 的用法是 **开子系统【看】** ——")
    print("         `%s`" % specs.NCCL_HANG_DIAG_QUOTE)
    check("C.2", (not cap_matches(g, "ROUTE")) and (not can_control_routing(g))
          and cap_matches(g, "OBSERVE")
          and cap_matches("NCCL_DEBUG_SUBSYS=TUNING", "OBSERVE"),
          "★★★ **判据成立**：把 `DEBUG_SUBSYS=GRAPH` / `=TUNING` 说成「能改选路」会被能力表**拒绝**；"
          "它们的能力是 **`OBSERVE`**（观测窗口，★ 逐字旁证：官方用 `DEBUG_SUBSYS` **开子系统看**）",
          "**「用 DEBUG 变量改选路」没被拒绝** —— ★ 先查能力表")


# ===========================================================================
# 5. 前哨：口径 / 阈值 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机** —— 本机无 GPU / 无 NCCL / 无真网卡。")
    print("  ★★ 本程序**没有 A 类**（`D.0` 机械证明），**一切钉住归 `D` 类**。")
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
    check("D.2", ASSERT_FAIL_MARK == ">< FAIL: nccl" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `>< FAIL: nccl`、且含 `FAIL`\n"
          "        ★★ `>` 是 **shell 的重定向操作符** ⇒ 它**只允许出现在 `print()` 的字符串里**。")
    check("D.2b", OTHER_THEME_MARKS == ("&& FAIL: collectives", ",, FAIL: network-topology",
                                        "+= FAIL: overlap"),
          "★ 「不跨主题照抄」的**对照物**（本域另外三个主题的标记）逐位钉住；"
          "★ 四个标记**两两不互为前缀**")

    # ── D.3 ★ 载体元数据 ───────────────────────────────────────────────────
    check("D.3", specs.NCCL_DOC_VERSION == "2.32.3"
          and specs.NCCL_DOC_BASE_URL
          == "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/"
          and specs.NCCL_FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：文档版本 `%s` · URL · 抓取日 `%s`"
          "（★ `P19` 的第一道防线：**不写版本与抓取日的引文不算引文**）；★ `UNSOURCED == ()` 一并钉住"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))

    # ── D.3b ★★★ 口径守卫：抓手表与能力表**同名同序**（否则 `cap_of` 会 KeyError）──
    #   ★★★ 它**必须跑在任何计算之前**（本库 `P23`）：两表不同名时，`cap_of()` 会
    #     `KeyError` ⇒ 那是**崩溃捕获**（不算真捕获）⇒ 有了这一条，同一个变异就是**断言捕获**。
    check("D.3b", tuple(h for h, _c in specs.NCCL_TOPO_HANDLE_CAPS) == tuple(specs.NCCL_TOPO_HANDLES)
          and len(specs.NCCL_TOPO_HANDLES) == len(specs.NCCL_TOPO_HANDLE_CAPS),
          "★★★ **口径守卫**：`NCCL_TOPO_HANDLES` 与 `NCCL_TOPO_HANDLE_CAPS` **同名同序**"
          "⇒ `cap_of()` 的查表安全（★ 任一被改，**本条先失败**，而不是让 `cap_of` 抛 `KeyError`）",
          "**抓手表与能力表不同名/不同序** —— ★ 先查两条常量")

    # ── D.4 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）────────────
    check("D.4", specs.NCCL_TOPO_HANDLES == ("NCCL_TOPO_FILE", "NCCL_DEBUG_SUBSYS=GRAPH",
                                             "NCCL_DEBUG_SUBSYS=TUNING")
          and specs.NCCL_TOPO_HANDLE_CAPS == (("NCCL_TOPO_FILE", "INPUT"),
                                              ("NCCL_DEBUG_SUBSYS=GRAPH", "OBSERVE"),
                                              ("NCCL_DEBUG_SUBSYS=TUNING", "OBSERVE"))
          and specs.NCCL_TOPO_HANDLE_COUNT == 3
          and specs.NCCL_OFFLOAD_SWITCHES == ("NCCL_NVLS_ENABLE", "NCCL_COLLNET_ENABLE")
          and specs.NCCL_OFFLOAD_DEFAULTS == ("2", "0")
          and specs.NCCL_NVLS_ENABLE_SINCE == "2.17"
          and specs.NCCL_COLLNET_ENABLE_SINCE == "2.6",
          "★★★ **逐元素钉住**：抓手表（3 档）· 能力表（3 对）· 抓手计数 · offload 开关（2 档）· "
          "默认值（2 档）· 两条开关的 since（`NVLS` = `2.17` · `COLLNET` = `2.6`）\n"
          "        —— ★ 只钉「整体」会让**元素级**变异漏网",
          "**抓手表 / 能力表 / offload 开关被动过**")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：引文逐字钉住 · 消歧记录钉住
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · 消歧记录钉住")
    print("=" * 78)

    # ── D.5 ★★ `NCCL_TOPO_FILE` / `DUMP` / `DEBUG_SUBSYS` 的引文与抓手 ────
    check("D.5", specs.NCCL_TOPO_FILE_QUOTE
          == ("Path to an XML file to load before detecting the topology. By default, NCCL will "
              "load /var/run/nvidia-topologyd/virtualTopology.xml if present.")
          and specs.NCCL_TOPO_FILE_SINCE == "2.6"
          and specs.NCCL_TOPO_DUMP_FILE == "NCCL_TOPO_DUMP_FILE"
          and specs.NCCL_DEBUG_SUBSYS_GRAPH == "GRAPH"
          and specs.NCCL_DEBUG_SUBSYS_TUNING == "TUNING"
          # ★★★ A3 修正：`GRAPH` / `TUNING` 的语义**逐字就在** `logging.html`（★ 不再是「非逐字」）
          and specs.NCCL_DEBUG_SUBSYS_GRAPH_QUOTE == "Topology detection and graph search"
          and specs.NCCL_DEBUG_SUBSYS_TUNING_QUOTE == "Algorithm and protocol selection"
          # ★★★ A2 修正：`NCCL_COLLNET_ENABLE` 的 since 官方逐字给了 `2.6`（★ 不是「官方未给」）
          and specs.NCCL_COLLNET_ENABLE_QUOTE
          == "NCCL_COLLNET_ENABLE (since 2.6) Enable the use of the CollNet plugin.",
          "★★ **`NCCL_TOPO_FILE` 的引文逐字钉住**（含默认路径 "
          "`/var/run/nvidia-topologyd/virtualTopology.xml`）+ since `2.6` + "
          "`NCCL_TOPO_DUMP_FILE` + `GRAPH` / `TUNING`\n"
          "        ★★ 载体 = `env.html` / `troubleshooting/logging.html`（版本 %s，抓取日 %s），"
          "★ 全部标 `[B]`\n"
          "        ★★★ **`GRAPH` / `TUNING` 的语义【逐字可引】**（载体 = `logging.html`）—— "
          "初稿标 `[B]·非逐字`（「本库未取到」）是**把「没去找」当成「不存在」**，"
          "★ 而**该页本来就在手上**；★★ `NCCL_COLLNET_ENABLE` 的 since 官方**逐字给了 `2.6`**"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE),
          "**拓扑抓手的引文被动过**")

    # ── D.6 ★★★ 消歧记录 + 那个**不是**引文的描述句 ───────────────────────
    check("D.6", specs.NCCL_TOPO_NO_ROUTE_CAP_NOTE
          == ("★ NCCL 官方暴露的拓扑抓手（NCCL_TOPO_FILE / DEBUG_SUBSYS=GRAPH / =TUNING）"
              "里没有一条的能力是「控制选路」——感知拓扑 ≠ 能控制选路")
          and specs.NCCL_PLAN_VS_DOC_NOTE
          == ("★「规划没记」≠「官方没有」——本库实测：NCCL_COLLNET_ENABLE 的 since 官方"
              "逐字给了 2.6，初稿却当成了「官方未给」")
          and specs.UNSOURCED == (),
          "★★★ **两条 `[E]` 记录逐字钉住**：**感知拓扑 ≠ 能控制选路**（★ 不是引文）＋ "
          "**「规划没记」≠「官方没有」**（★ A2 的方法论，不是引文）；★ `UNSOURCED == ()` 一并钉住\n"
          "        ★★ 注意 `NCCL_HANG_DIAG_QUOTE`（`C.2` 用的逐字旁证）在 `03` 里钉住 —— "
          "★ 同一条常量**只在一处钉**，避免「整体钉住代替不了元素钉住」的反面（重复钉）",
          "**消歧记录被动过**")


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
    print("★★★ **A = %d/%d** —— ★ **本主题【没有 A 类】**：本机无 GPU / 无 NCCL / 无真网卡。"
          % (_PASS["A"], _TOTAL["A"]))
    print("    ★★ 「常量钉住」**不算 A 类**（那是 `P12` 的变体）⇒ 本程序把它们全归了 `D` 类。")
    print("★ 依据在 `specs.py` 的 `# 来源：`（NCCL 官方产品文档 %s，抓取日 %s）。"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))
    print("★ 本程序只核对「官方抓手的条数与能力标签」与「两条错法真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_topology_awareness.py —— NCCL 的拓扑感知：官方暴露的抓手")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机** —— 本机无 GPU / 无 NCCL / 无真网卡。")
    print("  ★★★ **本主题没有 A 类**（`D.0` 机械证明）；★★ **载体全靠 `[B]` 级**产品文档。")
    print("  ★★★ **与 `interconnect/03:243-249` 分工**：本程序**一行都不重写**那段定性描述，")
    print("      只写它没写的：**变量名 · XML · DEBUG 子系统**这三条官方抓手。")
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
