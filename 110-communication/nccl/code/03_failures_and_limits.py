#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_failures_and_limits.py —— **NCCL 的报错归因与它的边界**：官方归因覆盖到哪、没覆盖到哪。

配套文档：`03-failures-and-limits.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题【没有 A 类】**：本机**无 GPU、无 NCCL、无真网卡** ⇒「算法正确性」/「真实报错行为」
  **无处可验**。★ 本程序**跑在默认 `python3` 下、只用标准库** ⇒ 落在 `CONTRIBUTING` §2 表的
  **第二种情形**（**程序内断言**）。
  ★★ **如实点破一层**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**
  ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。
  ★★ 而「**把不可测的那一档写明**」是【另一条】纪律（**不是** §2 的任何一种编号形态）。
  ★★★ **不得把「常量钉住」冒充 A 类**（那是 `P12` 的变体）⇒ ★★ 本程序的一切钉住归 `D` 类。
  ★ `D.0` **机械地**证明「本文件里一条 A 类判据都没有」（AST 扫自己）。

★★★ **本程序的两条禁写清单（计划 `§C.5`）**：
  · **`F2`** —— **不得写**「NCCL 文档说 `unhandled system error` 是……」：
    那个字面串**只在官方【源码】**（`v2.26.2-1` 的 `src/init.cc:2198`）里，
    用户指南 8 页 grep ⇒ **0 命中** ⇒ ★★ 要引**必须注明「官方源码」**并标 **`[B']`**
    （`C.2`/`D.6` 判它）。
  · **`F3`** —— **不得**把「阈值表」当官方规则（见 `01` 的 `B.2`/`C.2`；本程序不重复）。

★★★ **本程序的主力是 `C.1`**：**把官方归因当成通用规律** 必须失败 ——
  官方**逐字点名**的错误码只有 **2 个**（`ncclUnhandledCudaError` / `ncclSystemError`），
  而错误码表**共 7 个** ⇒ ★★ 「所有 NCCL 错误都表示外部组件失败」是**错的**。

★★ **覆盖边界**：本程序守的是「**归因覆盖了哪几个码、覆盖之外有几条入口、错法会不会被拒**」。
  **不守**任何一条**真实 NCCL 的报错行为**（本机**测不了**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `B.1`  ★★★ **归因的【可判定范围】**：错误码 **7 个**、被归因的 **2 个** ⇒ **5 个没给归因**；
    ★★ 而 **hang 官方【没有】给出单一归因**，只给「开这些子系统看」；
  · `B.2`  ★★ **异步错误**经 `ncclCommGetAsyncError` 报出（★ 返回值**不是**唯一信号）；
  · `C.1`  ★★★ **把官方归因当成通用规律**必须被拒（★ 本程序的主力）；
  · `C.2`  ★★★ **把 `unhandled system error` 说成「NCCL 文档说的」**必须被拒（`F2`）；
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
# 1. 被测的「归因范围校验器」—— ★★ **它【不是】A 类**（本机没有 NCCL 可测）
#    ★★★ 它是**本库的一段校验器**：把「官方归因覆盖了哪几个码」变成**可数**的量，
#      用来判「把归因当通用规律」这类错法。★ 一切钉住归 `D` 类。
# ===========================================================================
def code_of(name):
    """取某错误码的**数值**（`ncclUnhandledCudaError` ⇒ `1`）—— ★ 未知名字抛 `ValueError`。"""
    d = dict(specs.NCCL_ERROR_CODES)
    if name not in d:
        raise ValueError("不是 NCCL 错误码表里的名字：%r" % (name,))
    return d[name]


def attributed_codes():
    """★ **被官方归因覆盖的**错误码 —— ★ 从**归因引文点名的**那两个，**不是**猜的。"""
    return tuple(n for n, _c in specs.NCCL_ERROR_CODES if n in specs.NCCL_ATTRIBUTED_CODES)


def unattributed_codes():
    """★ **官方【没有】给归因的**错误码（全表减去被覆盖的）—— 本程序报出它。"""
    return tuple(n for n, _c in specs.NCCL_ERROR_CODES if n not in specs.NCCL_ATTRIBUTED_CODES)


def attribution_is_general():
    """★★★ **「归因是一条通用规律」这个论断成立吗？** —— 只有当它覆盖**全表**才成立。

    ★★★ 官方点名 **2 个**、全表 **7 个** ⇒ 本函数返回 `False`
      ⇒ ★★ **把它当通用规律是错的**（`C.1` 判它）。
    """
    return len(attributed_codes()) == specs.NCCL_ERROR_CODE_COUNT


def carrier_of_syserr_quote():
    """★ 那条 `unhandled system error` 字面串的**载体**：`source`（官方源码）还是 `doc`（文档）。

    ★★★ `NCCL_SYSERR_SOURCE_TAG`（`v2.26.2-1`）**≠** `NCCL_DOC_VERSION`（`2.32.3`）
      ⇒ 载体是 **`source`** ⇒ ★★ **说它来自文档是错的**（`C.2` 判它，`F2`）。
    """
    return "doc" if specs.NCCL_SYSERR_SOURCE_TAG == specs.NCCL_DOC_VERSION else "source"


# ===========================================================================
# 2. A 类 —— ★★★ 【本主题没有 A 类】
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ 【本主题没有 A 类】—— 本机无 GPU / 无 NCCL / 无真网卡")
    print("=" * 78)
    print("  ★★★ 「NCCL 报什么错、为什么报」**无处可验**：本机既没有 NCCL 也没有 GPU。")
    print("      ⇒ 本程序**跑在默认 `python3` 下** ⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**")
    print("        （**程序内断言**）；★ 而「**把不可测的那一档写明**」是【另一条】纪律（不是 §2 的形态）。")
    print("      ★★ **如实点破**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而本主题**没有 A**")
    print("        ⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。")
    print("  ★★★ **不得把「常量钉住」冒充 A 类**（`P12` 的变体）⇒ ★★ 钉住一律归 `D` 类。")
    print("  ★★ 本程序只守一件事：**官方归因覆盖到哪几个码、覆盖之外有几条入口**。")
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
# 3. B 类：**归因的可判定范围** · 异步错误的报出通道
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `[B]` 级的账：**归因的【可判定范围】**（2 / 7）· 异步错误的通道")
    print("=" * 78)
    print("  ★★ 载体 = NCCL 官方产品文档 `api/types.html`（码表）/ `usage/communicators.html`"
          "（归因与异步段）/ `troubleshooting/logging.html`，")
    print("      版本 **%s**，抓取日 **%s**。" % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))

    # ── B.1 ★★★ 归因的【可判定范围】＋ hang 没有单一归因 ─────────────────
    print()
    print("  ── B.1 ★★★ **归因的可判定范围** ──")
    for n, c in specs.NCCL_ERROR_CODES:
        tag = "★ **有归因**" if n in specs.NCCL_ATTRIBUTED_CODES else "（未给归因）"
        print("     %-2d  %-24s %s" % (c, n, tag))
    att, unat = attributed_codes(), unattributed_codes()
    print("     ★★ **全表 %d 个码** · **被归因 %d 个** · **未给归因 %d 个**"
          % (specs.NCCL_ERROR_CODE_COUNT, len(att), len(unat)))
    print("     ★★ 注：官方 `api/types.html` 原表**共 8 项**，多的一项 `ncclSuccess (0)` **不是错误**")
    print("        ⇒ **本文不计它**，「全表」一律指这 **7 个错误码**。")
    print("     ★★★ 那条归因（逐字）：%s" % specs.NCCL_ERROR_ATTRIBUTION_QUOTE)
    print()
    print("     ★★★ **hang 的处置不同**：官方**没有**给出单一归因 —— 只给了「**开这些子系统看**」：")
    print("       `%s`" % specs.NCCL_HANG_DIAG_QUOTE)
    check("B.1", len(specs.NCCL_ERROR_CODES) == specs.NCCL_ERROR_CODE_COUNT == 7
          and att == ("ncclUnhandledCudaError", "ncclSystemError")
          and len(att) == specs.NCCL_ATTRIBUTED_CODE_COUNT == 2
          and len(unat) == 5
          and "external component failed" in specs.NCCL_ERROR_ATTRIBUTION_QUOTE
          and specs.NCCL_HANG_DIAG_QUOTE
          == ("If your application hangs during NCCL initialization: NCCL_DEBUG=INFO "
              "NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET ./my_app"),
          "★★★ **判据成立**：错误码 **7 个**、官方**逐字点名**被归因的 **2 个**"
          "（`ncclUnhandledCudaError` / `ncclSystemError`）⇒ **5 个未给归因**；"
          "★★ **而 hang 连归因都没有** —— 只有一条**诊断入口**",
          "**错误码表或归因范围被动过** —— ★ 先查 `NCCL_ERROR_CODES` / `NCCL_ATTRIBUTED_CODES`")

    # ── B.2 ★★ 异步错误的报出通道 ────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **异步错误**的报出通道 ──")
    print("     接口：`%s`（★ 异步错误**不走返回值**）" % specs.NCCL_ASYNC_ERROR_API)
    print("     官方逐字：`%s`" % specs.NCCL_ASYNC_ERROR_QUOTE)
    print("     ★★ 读法：**返回值为 0 不代表 NCCL 没出错** —— 内核里跑出来的错误"
          "要**查询**这个接口。")
    print("       ★★ **本库【不写】它的用法**（库 API 的用法是全库禁区）。")
    check("B.2", specs.NCCL_ASYNC_ERROR_API == "ncclCommGetAsyncError"
          and specs.NCCL_ASYNC_ERROR_QUOTE == "will usually not progress and never complete"
          and specs.NCCL_ASYNC_ERROR_API != specs.NCCL_ERROR_ATTRIBUTION_QUOTE,
          "★★ **判据成立**：异步错误经 `%s` 报出（★ **返回值不是唯一信号**）——"
          "载体 = `usage/communicators.html`（版本 %s，抓取日 %s）"
          % (specs.NCCL_ASYNC_ERROR_API, specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE),
          "**异步错误的接口或引文被动过**")


# ===========================================================================
# 4. C 类：两条【真的会失败】的错法（★ `C.1` 是本程序的主力）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**把归因当通用规律** · **把源码串说成文档说的**")
    print("=" * 78)

    # ── C.1 ★★★ 把官方归因当成【通用规律】 ──────────────────────────────
    print()
    print("  ── C.1 ★★★ **把官方归因当成【通用规律】**（★ 本程序的主力）──")
    print("     官方归因逐字只点了 **2 个**码；错误码表**共 7 个** ⇒")
    print("     论断「所有 NCCL 错误都表示外部组件失败」⇒ 校验器判**成立**？ **%s**"
          "（★ 必须是 `False`）" % attribution_is_general())
    print("     被归因：%s" % (attributed_codes(),))
    print("     **未被归因**：%s" % (unattributed_codes(),))
    print("     ★★★ 关键：那 **%d** 个未被归因的码**在官方文档里【没有】这条归因** ⇒"
          % len(unattributed_codes()))
    print("       ★★ 把「`ncclUnhandledCudaError`/`ncclSystemError` 表示外部组件失败」"
          "**升格成通用规律**，就把 5 个码**也**盖住了。")
    check("C.1", (not attribution_is_general())
          and len(attributed_codes()) == 2 and specs.NCCL_ERROR_CODE_COUNT == 7
          and len(unattributed_codes()) == 5
          and specs.NCCL_ATTRIBUTION_SCOPE_NOTE
          == ("★ 官方对错误码的归因只覆盖 ncclUnhandledCudaError 与 ncclSystemError 两个"
              "（共 7 个码）⇒ 不得把它当成通用规律"),
          "★★★ **判据成立**：把官方归因当**通用规律**会被拒 —— 它只覆盖 **2 / 7** 个码，"
          "**5 个未被归因** ⇒ ★★ **「把官方归因当通用规律」真的会失败**",
          "**「把归因当通用规律」没被拒绝** —— ★ 先查 `NCCL_ATTRIBUTED_CODES` / "
          "`NCCL_ERROR_CODE_COUNT`")

    # ── C.2 ★★★ 把 `unhandled system error` 说成「NCCL 文档说的」（`F2`）──
    print()
    print("  ── C.2 ★★★ **把源码串说成「NCCL 文档说的」**（`F2`）──")
    print("     那条字面串：`%s`" % specs.NCCL_SYSERR_SOURCE_QUOTE)
    print("     载体判定 ⇒ **`%s`**（★ 必须是 `source`，**不是** `doc`）" % carrier_of_syserr_quote())
    print("     ★★★ 依据：载体是**官方源码** `%s` @ `%s`，而**文档版本**是 `%s` ——"
          % (specs.NCCL_SYSERR_SOURCE_TAG, specs.NCCL_SYSERR_SOURCE_LOC, specs.NCCL_DOC_VERSION))
    print("       **两者【不同】**；★★ 该字面串在**用户指南 8 页 grep ⇒ 0 命中**。")
    check("C.2", carrier_of_syserr_quote() == "source"
          and specs.NCCL_SYSERR_SOURCE_TAG != specs.NCCL_DOC_VERSION
          and specs.NCCL_SYSERR_SOURCE_TAG == "v2.26.2-1"
          and specs.NCCL_SYSERR_SOURCE_LOC == "src/init.cc:2198"
          and specs.NCCL_SYSERR_SOURCE_NOTE
          == ("★「unhandled system error」的载体是官方【源码】(v2.26.2-1, src/init.cc:2198)，"
              "不是 NCCL 2.32.3 的文档"),
          "★★★ **判据成立**：把 `unhandled system error` 说成「**NCCL 文档说的**」会被拒 —— "
          "它的载体是**官方源码** `%s` @ `%s`（★ 文档版本 `%s`，**两者不同**）⇒ "
          "★★ **`F2` 的错法真的会失败**"
          % (specs.NCCL_SYSERR_SOURCE_TAG, specs.NCCL_SYSERR_SOURCE_LOC, specs.NCCL_DOC_VERSION),
          "**「源码串说成文档」没被拒绝** —— ★ 先查 `NCCL_SYSERR_SOURCE_TAG` 与 "
          "`NCCL_DOC_VERSION`（★ 两者**必须不同**）")


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

    # ── D.3 ★ 载体元数据（★ 含 `[B']` 源码载体）────────────────────────────
    check("D.3", specs.NCCL_DOC_VERSION == "2.32.3"
          and specs.NCCL_DOC_BASE_URL
          == "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/"
          and specs.NCCL_FETCH_DATE == "2026-10-08"
          and specs.NCCL_SYSERR_SOURCE_TAG == "v2.26.2-1"
          and specs.NCCL_SYSERR_SOURCE_LOC == "src/init.cc:2198"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：文档版本 `%s` · URL · 抓取日 `%s`；"
          "★★ **`[B']` 源码载体** `%s` @ `%s`（★ **与文档版本不同**）；★ `UNSOURCED == ()` 一并钉住"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE,
             specs.NCCL_SYSERR_SOURCE_TAG, specs.NCCL_SYSERR_SOURCE_LOC))

    # ── D.3b ★★★ 口径守卫：错误码表与「被归因表」**一致**（否则 `code_of`/计数会跑偏）──
    #   ★★★ 它**必须跑在任何计算之前**（本库 `P23`）：保证
    #     ① 码表元素数与 `NCCL_ERROR_CODE_COUNT` 相等；② 被归因的名字**都在码表里**
    #     （否则 `attributed_codes()` 会少算、甚至恒为空 ⇒ 判据退化）。
    names = tuple(n for n, _c in specs.NCCL_ERROR_CODES)
    check("D.3b", len(names) == specs.NCCL_ERROR_CODE_COUNT
          and all(n in names for n in specs.NCCL_ATTRIBUTED_CODES)
          and len(set(names)) == len(names),
          "★★★ **口径守卫**：码表 **%d** 个、`NCCL_ERROR_CODE_COUNT` = %d、**名字无重复**，"
          "且**被归因的两个名字都在码表里**（★ 否则 `attributed_codes()` 会少算 ⇒ 判据退化）"
          % (len(names), specs.NCCL_ERROR_CODE_COUNT),
          "**错误码表或计数被动过** —— ★ 先查 `NCCL_ERROR_CODES` / `NCCL_ATTRIBUTED_CODES`")

    # ── D.4 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）────────────
    check("D.4", specs.NCCL_ERROR_CODES == (("ncclUnhandledCudaError", 1),
                                            ("ncclSystemError", 2),
                                            ("ncclInternalError", 3),
                                            ("ncclInvalidArgument", 4),
                                            ("ncclInvalidUsage", 5),
                                            ("ncclRemoteError", 6),
                                            ("ncclInProgress", 7))
          and specs.NCCL_ERROR_CODE_COUNT == 7
          and specs.NCCL_ATTRIBUTED_CODES == ("ncclUnhandledCudaError", "ncclSystemError")
          and specs.NCCL_ATTRIBUTED_CODE_COUNT == 2,
          "★★★ **逐元素钉住**：错误码表（7 对 = 14 个叶子）· 码计数 · 被归因表（2 档）· 被归因计数\n"
          "        —— ★ 只钉「整体」会让**元素级**变异漏网（本库实测过：一次报出 58 条漏网）",
          "**错误码表 / 被归因表被动过**")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：引文逐字钉住 · 两条 `[E]` 消歧记录
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · 两条 `[E]` 消歧记录")
    print("=" * 78)

    # ── D.5 ★★ 归因引文 · hang 入口 · `[B']` 源码串（逐字）───────────────
    check("D.5", specs.NCCL_ERROR_ATTRIBUTION_QUOTE
          == ("ncclUnhandledCudaError and ncclSystemError indicate that a call NCCL made to an "
              "external component failed, which caused the NCCL operation to fail. The error "
              "message should explain which component the user should look at and try to fix")
          and specs.NCCL_HANG_DIAG_QUOTE
          == ("If your application hangs during NCCL initialization: NCCL_DEBUG=INFO "
              "NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET ./my_app")
          and specs.NCCL_SYSERR_SOURCE_QUOTE
          == ('case ncclSystemError : return "unhandled system error (run with NCCL_DEBUG=INFO '
              'for details)";'),
          "★★ **三条引文逐字钉住**：归因句（`usage/communicators.html`）· hang 入口"
          "（`troubleshooting/logging.html`）· `[B']` 源码串（`v2.26.2-1` 的 `src/init.cc:2198`）\n"
          "        ★★★ **归因句给出的范围**：只有 `ncclUnhandledCudaError` / `ncclSystemError` "
          "两个（★ 那正是 `C.1` 的依据）\n"
          "        ★★ 载体版本 %s，抓取日 %s；★ 全部标 `[B]`（源码那条标 `[B']`）"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE),
          "**归因句 / hang 入口 / 源码串被动过**")

    # ── D.6 ★★ 异步错误接口 + 两条 `[E]` 记录 + `UNSOURCED` ───────────────
    check("D.6", specs.NCCL_ASYNC_ERROR_API == "ncclCommGetAsyncError"
          and specs.NCCL_ASYNC_ERROR_QUOTE == "will usually not progress and never complete"
          and specs.NCCL_ATTRIBUTION_SCOPE_NOTE != ""
          and "通用规律" in specs.NCCL_ATTRIBUTION_SCOPE_NOTE
          and specs.NCCL_SYSERR_SOURCE_NOTE != ""
          and "官方【源码】" in specs.NCCL_SYSERR_SOURCE_NOTE
          and specs.UNSOURCED == (),
          "★★ 钉住：异步错误接口 `%s` 与它的引文；★★ 两条 `[E]` 消歧记录"
          "（归因**不是**通用规律 · 源码串**不是**文档）—— ★ 它们是 `C.1`/`C.2` 的判据依据，"
          "**没有外部出处** ⇒ **不得**当引文；★ `UNSOURCED == ()` 一并钉住"
          % specs.NCCL_ASYNC_ERROR_API,
          "**异步错误接口或两条消歧记录被动过**")


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
    print("★ 依据在 `specs.py` 的 `# 来源：`（NCCL 官方产品文档 %s，抓取日 %s；"
          "另一条 `[B']` 是官方源码）。" % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))
    print("★ 本程序只核对「归因覆盖了哪几个码」与「两条错法真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_failures_and_limits.py —— NCCL 的报错归因与它的边界")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机** —— 本机无 GPU / 无 NCCL / 无真网卡。")
    print("  ★★★ **本主题没有 A 类**（`D.0` 机械证明）；★★ **载体全靠 `[B]` 级**产品文档。")
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
