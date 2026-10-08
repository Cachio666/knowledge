#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_architecture_and_selection.py —— **NCCL 的算法取值与协议取值，以及它怎么选**。

配套文档：`01-architecture-and-selection.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题【没有 A 类】—— 这是本程序的第一条事实，不是遗漏**：
  本机**无 GPU、无 NCCL、无真网卡** ⇒ NCCL 的「**算法正确性**」**无处可验**。
  按 `CONTRIBUTING` §2 的**第三种形态**处理：**把不可测的那一档写明**，
  ★★★ **不得把「常量钉住」冒充 A 类** —— 那是 `P12` 的变体
  （**判据看着在检查，其实只是把常量抄了一遍**）。
  ⇒ ★★ 本程序的一切「钉住」**一律归 `D` 类**（照同域 `collectives` 的做法）。
  ★ 本程序用 `D.0` **机械地**证明「本文件里一条 A 类判据都没有」（AST 扫自己的源码）。

★★★ **【载体】全是 `[B]` 级** —— NCCL 官方**产品文档**（**不是规范**），
  版本一律 `NCCL 2.32.3`，抓取日 **2026-10-08**。
  ★★ 唯一的 `[B']`（**官方源码**）是 `03` 的主题，本程序不引。
  ★★★ **本域的头号错误模式是 `P19`（把 B 级当 A 级）** ⇒ 本程序**每一条**外部事实
  都带 **URL + 版本 + 抓取日**，并**显式标 `[B]`**。

★★★ **本程序修的三条禁写清单（计划 `§C.5`）**：
  · **`F1`** —— 不得把 `"Communication is always asynchronous with respect to the host"`
    当 NCCL 文档的逐字引文（8 个文档页 grep ⇒ **0 命中**）；
    ★ 可用的**已验证**替代表述只有 `specs.py` 的三条（`D.6` 逐字钉住）。
  · **`F2`** —— 见 `03`（本程序不引那条字面串）。
  · **`F3`** —— ★★★ **不得写**「NCCL 在 X 大小以下用 tree、以上用 ring」这类**阈值表**：
    `NCCL_TREE_THRESHOLD` 已 `removed in 2.5`、`NCCL_SINGLE_RING_THRESHOLD` 已 `removed in 2.3`
    （`B.2`/`C.2` 判它）。

★★ **覆盖边界**（`CONTRIBUTING` §2「不得声称验证了更多」）：
  本程序守的是「**取值表与版本账有没有抄对、错法会不会被拒**」。
  **不守**任何一条**真实 NCCL 的行为**（选哪个算法、跑多快）—— 本机**测不了**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `B.1`  ★★ **取值表的【版本账】**：8 个取值、每个带 `since`、**恰好 1 个闭区间**（`Collnet`），
    且 `2.14` 上**同时出现两个**替代取值；
  · `B.2`  ★★★ **默认 = `unset`（自动选）**＋**两个阈值变量已移除** ⇒ **官方没有阈值表**（`F3`）；
  · `B.3`  ★★ `NCCL_PROTO` 三个取值 + **官方劝阻**（含 `data corruption` 的理由）；
  · `B.4`  ★★ **「不跨标记」计数**：`NCCL_ALGO` 那句 **raw=0 / plain=1**（跨标记）、
    `usage/streams.html` 那句 **raw=1 / plain=1**（不跨）；
  · `C.1`  ★★★ **编一个不存在的 `NCCL_ALGO` 取值**必须被拒；
  · `C.2`  ★★★ **把 `since` 版本记错**必须被版本账拒绝；
  · `D.0`  ★★★ **机械证明「本文件没有 A 类判据」**（AST 扫自己的源码）；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **引文逐字**钉住 + **失败标记自检**。
"""
import ast
import collections
import sys

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（nccl）取 `>< FAIL: nccl`。
#   ★ 本域另外三个主题各用**不同的前缀**（`&&` / `,,` / `+=`），四个标记**两两不互为前缀**
#     ⇒ 跨主题照抄会被下面的 `D.2b` 当场抓住。
#   ★★ `>` 是 **shell 的重定向操作符** ⇒ 它**只允许出现在 `print()` 的字符串里**，
#     **不得**出现在 `Makefile` 的 recipe 行或任何 shell 命令行里。
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

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
      ⇒ 所以 `cond` 为假时走**另一版文案**。
      ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 本行打印的失败标记必然在**行首**。
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
# 1. 被测的「取值表校验器」—— ★★ **它们【不是】A 类**
#    ★★★ 它们是**本库写的一段校验收敛器**，被测对象是**常量表**（`specs.py` 的 [B] 级引文）
#      ⇒ 它们**不测任何实现**（本机没有 NCCL 可测）⇒ **一切钉住归 `D` 类**。
#      这里放它们，是为了让 `C` 类的**反例**（编造的取值 / 记错的版本）有一个**可判定的**判据。
# ===========================================================================
def algo_value_known(name):
    """`name` 是否是 `NCCL_ALGO` 文档列出的取值（★ **大小写敏感** —— 文档写 `Ring` 不是 `ring`）。"""
    return name in specs.NCCL_ALGO_NAMES


def since_of(name):
    """把一个**已知**取值映射到它的 `since`（★ 未知取值抛 `ValueError`）。

    ★★ `D.3b` 守着「两张表等长」这个前提，所以下标是安全的。
    """
    try:
        i = specs.NCCL_ALGO_NAMES.index(name)
    except ValueError:
        raise ValueError("不是 NCCL_ALGO 的已知取值：%r" % (name,))
    return specs.NCCL_ALGO_SINCE[i]


def rejects_since(name, claimed):
    """★ 版本账**是否拒绝**这个 `since` 断言 —— `True` = 记错了、账不认。"""
    return since_of(name) != claimed


def is_removed_var(name):
    """该名字是否是**已被移除**的阈值变量（`F3`：它们**不能**用来预测算法选择）。"""
    return name in dict(specs.NCCL_REMOVED_VARS)


def predict_algo_by_threshold(_msg_bytes, _n_dev):
    """★★★ **一个【必须返回「未知」的】函数** —— 但它**不允许**用阈值表回答。

    ★★★ 官方**没有公开**算法选择的阈值规则（`NCCL_TREE_THRESHOLD` / `NCCL_SINGLE_RING_THRESHOLD`
      **都已移除**）⇒ 任何「按消息大小分档」的表都是**编的** ⇒ 本函数**只**返回 `"unknown"`。
      ★ `C.2` 判「谁要是把它写成一张表，那条断言必须失败」。
    """
    return "unknown"


# ===========================================================================
# 2. A 类 —— ★★★ 【本主题没有 A 类】
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ 【本主题没有 A 类】—— 本机无 GPU / 无 NCCL / 无真网卡")
    print("=" * 78)
    print("  ★★★ NCCL 的**算法正确性**【无处可验】：本机既没有 NCCL 也没有 GPU。")
    print("      ⇒ 按 `CONTRIBUTING` §2 的**第三种形态**：把不可测的那一档**写明**。")
    print("  ★★★ **不得把「常量钉住」冒充 A 类** —— 那是 `P12` 的变体：")
    print("      判据**看着在检查**，其实只是把常量**抄了一遍**（`abs(2V − 2V) == 0`）。")
    print("      ⇒ ★★ 本程序的「钉住」**一律归 `D` 类**（见下面的 `D.*`）。")
    print("  ★★ 本程序只守两件事：**取值表与版本账抄对了没有** · **错法会不会被拒**。")
    print("  ★★ 它**不守**任何一条真实 NCCL 的行为（选哪个算法 / 跑多快）—— 本机**测不了**。")
    print()
    # ── D.0 ★★★ 机械证明「本文件没有 A 类判据」───────────────────────────
    n_a = _count_a_checks_in_source()
    print("  ── D.0 ★★★ **机械证明**：AST 扫本文件，`check(\"A...\")` 的调用数 = **%d** ──" % n_a)
    check("D.0", n_a == 0,
          "★★★ **判据成立**：本文件里**一条 A 类判据都没有**（AST 逐个数 `check()` 的"
          "第一实参，字面量以 `A.` 开头者 = **0**）⇒ ★★ 「本主题没有 A 类」是"
          "**可机械复核**的，不是一句声明",
          "★★ 本文件里出现了 A 类判据 —— ★ 那与「本主题没有 A 类」**自相矛盾**"
          "（本机没有 NCCL 可测，任何 A 类都是**假的 A 类**）")
    print("      ★★ 做法：`check()` 的第一实参是**字符串字面量**且以 `A.` 开头者计数。")
    print("      ★ 哪天本机有了 GPU 与 NCCL，要加 A 类，**这一条会先失败** —— 那正是设计意图。")


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
# 3. B 类：代价/事实模型 —— ★★ 全部是 **`[B]` 级产品文档**上的**账**
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `[B]` 级的**账**：取值表的版本账 · 默认与阈值 · 协议 · 不跨标记")
    print("=" * 78)
    print("  ★★ 载体 = NCCL 官方产品文档 `env.html` / `usage/streams.html`，")
    print("      版本 **%s**，抓取日 **%s**。" % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))

    # ── B.1 ★★ 取值表的【版本账】──────────────────────────────────────────
    print()
    print("  ── B.1 ★★ `NCCL_ALGO` 的**版本账**（8 个取值，逐条可数）──")
    for i, nm in enumerate(specs.NCCL_ALGO_NAMES):
        print("     %-14s since %s" % (nm, specs.NCCL_ALGO_SINCE[i]))
    closed = [(n, s) for n, s in zip(specs.NCCL_ALGO_NAMES, specs.NCCL_ALGO_SINCE)
              if "–" in s]                       # ★ en-dash ⇒ 闭区间（★ ASCII `-` 不算）
    at214 = [n for n, s in zip(specs.NCCL_ALGO_NAMES, specs.NCCL_ALGO_SINCE) if s == "2.14+"]
    print("     ★★ **闭区间（有移除）的取值**：%s ⇒ **恰好 %d 个**" % (closed, len(closed)))
    print("     ★★ **`2.14` 上出现的取值**：%s ⇒ **%d 个**" % (at214, len(at214)))
    print("     ★★★ 读法：`Collnet`（`2.5–2.13`）在 **`2.14` 被拆成两条**"
          "（`CollnetChain` / `CollnetDirect`），★ **不是**「还在、又多了两条」。")
    check("B.1", len(specs.NCCL_ALGO_NAMES) == specs.NCCL_ALGO_COUNT == 8
          and len(closed) == 1 and closed[0][0] == "Collnet"
          and closed[0][1] == "2.5–2.13"
          and at214 == ["CollnetChain", "CollnetDirect"]
          and specs.NCCL_ALGO_SINCE[0] == "2.5+" and specs.NCCL_ALGO_SINCE[7] == "2.23+",
          "★★★ **判据成立**：取值表 **8 个**、每个都带 `since`；**恰好 1 个闭区间**"
          "（`Collnet` = `2.5–2.13`）；**`2.14` 上恰好 2 个取值**（`CollnetChain`/`CollnetDirect`）"
          "⇒ ★★ 版本账**读得通**（拆而非增）",
          "**取值表或版本账被动了** —— ★ 先查 `NCCL_ALGO_NAMES` / `NCCL_ALGO_SINCE` 的同下标对应")

    # ── B.2 ★★★ 默认 unset ＋ 两个阈值变量已移除 ⇒ 官方没有阈值表（`F3`）───────
    print()
    print("  ── B.2 ★★★ 默认 = `unset`（自动选）＋ 两个阈值变量**已移除**（`F3`）──")
    print("     默认句：%s" % specs.NCCL_ALGO_DEFAULT_QUOTE)
    for nm, ver in specs.NCCL_REMOVED_VARS:
        print("     ★ `%s` —— **removed in %s**" % (nm, ver))
    print("     ★★★ **`F3` 的落点**：**不得写**「NCCL 在 X 大小以下用 tree、以上用 ring」这类"
          "**阈值表** ——")
    print("       那两个变量**已被移除**，★ **官方未公开替代规则**。")
    print("     ★★ **复核陷阱**：**只看文档 TOC 会误判「那两个变量还在用」**"
          "（TOC 里**仍列着**，**正文**才写 `removed`）")
    print("       ⇒ ★★ **凡引「某变量存在/已移除」，去看【正文】而不是 TOC**。")
    check("B.2", specs.NCCL_ALGO_DEFAULT_QUOTE.startswith("The default is unset")
          and specs.NCCL_REMOVED_COUNT == 2
          and is_removed_var("NCCL_TREE_THRESHOLD") and is_removed_var("NCCL_SINGLE_RING_THRESHOLD")
          and predict_algo_by_threshold(1048576, 8) == "unknown",
          "★★★ **判据成立**：默认是 **`unset`**（⇒ NCCL **自动选**）、"
          "`NCCL_TREE_THRESHOLD` 与 `NCCL_SINGLE_RING_THRESHOLD` **都已移除** ⇒ "
          "★ **官方没有阈值表** ⇒ 任何「按大小分档」的表都是**编的**（`F3`）",
          "**默认句或移除账被动过** —— ★ 那会让「官方没有阈值表」这条结论失去依据（`F3`）")

    # ── B.3 ★★ `NCCL_PROTO`：三个取值 + 官方劝阻 ─────────────────────────
    print()
    print("  ── B.3 ★★ `NCCL_PROTO`：三个取值 + **官方劝阻** ──")
    print("     定义句：%s" % specs.NCCL_PROTO_QUOTE)
    print("     取值：%s" % (specs.NCCL_PROTO_VALUES,))
    print("     ★★★ 官方警告（逐字）：%s" % specs.NCCL_PROTO_WARNING_QUOTE)
    print("       ★★ 读法：官方**劝阻**用户设置它，且给了理由（`data corruption`）")
    print("         ⇒ **协议不是给用户调的** —— 这是与 `NCCL_ALGO` 的明显差别"
          "（后者只给可选项，不给劝告）。")
    check("B.3", specs.NCCL_PROTO_VALUES == ("LL", "LL128", "Simple")
          and "data corruption" in specs.NCCL_PROTO_WARNING_QUOTE
          and specs.NCCL_PROTO_WARNING_QUOTE.startswith("Users are discouraged from setting"),
          "★★ **判据成立**：`NCCL_PROTO` 三个取值（`LL` / `LL128` / `Simple`）＋"
          "官方**劝阻**（含 `data corruption` 的理由）—— ★★ **本程序不写 LL/LL128 的机制**"
          "（`F9`：机制要写就必须带载体；本库未取到 `prims_ll128.h` 的正文 ⇒ **不写**）",
          "**协议取值或官方警告被动过**")

    # ── B.4 ★★ 「不跨标记」计数（引文的【可引性】判据）────────────────────
    print()
    print("  ── B.4 ★★ **「不跨标记」计数**：raw HTML 的 `count()` vs 去标记文本的 `count()` ──")
    print("     `NCCL_ALGO` 那句（`env.html`）        ：raw **%d** / plain **%d** ⇒ **跨标记**"
          % (specs.NCCL_ALGO_QUOTE_RAW_COUNT, specs.NCCL_ALGO_QUOTE_PLAIN_COUNT))
    print("     `usage/streams.html` 那句            ：raw **%d** / plain **%d** ⇒ **不跨标记**"
          % (specs.NCCL_ASYNC_DEVICE_RAW_COUNT, specs.NCCL_ASYNC_DEVICE_PLAIN_COUNT))
    print("     ★★★ 判据：**「逐字」这个标签要看 raw HTML 的计数** —— "
          "只看去标记文本会把**跨标记的拼接**当成逐字。")
    check("B.4", specs.NCCL_ALGO_QUOTE_RAW_COUNT == 0 and specs.NCCL_ALGO_QUOTE_PLAIN_COUNT == 1
          and specs.NCCL_ASYNC_DEVICE_RAW_COUNT == 1
          and specs.NCCL_ASYNC_DEVICE_PLAIN_COUNT == 1
          and specs.NCCL_ALGO_QUOTE_PLAIN_COUNT > specs.NCCL_ALGO_QUOTE_RAW_COUNT
          and specs.NCCL_ASYNC_DEVICE_PLAIN_COUNT == specs.NCCL_ASYNC_DEVICE_RAW_COUNT,
          "★★ **判据成立**：`NCCL_ALGO` 那句在**原始 HTML 上 `count()==0`**、**去标记文本上 `1`**"
          "（`NCCL_ALGO` 被 `<code><span class=\"pre\">` 包裹 ⇒ **跨标记**）；"
          "而 `usage/streams.html` 那句 **raw=1 / plain=1**（**不跨标记**）",
          "**引文的标记计数被动过** —— ★ 那会让「逐字」的判定失去依据")


# ===========================================================================
# 4. C 类：两条【真的会失败】的错法（编造取值 / 记错版本）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**编一个不存在的取值** · **把 `since` 记错**")
    print("=" * 78)
    print("  ★★ 本主题的 C 类钉在**可判定的错法**上（计划 `§3 Chunk 2` 的指示）：")
    print("     ① 编一个**不存在的** `NCCL_ALGO` 取值；② 把 `since` 版本**记错**。")

    # ── C.1 ★★★ 编一个不存在的 `NCCL_ALGO` 取值 ─────────────────────────
    print()
    print("  ── C.1 ★★★ **编一个不存在的 `NCCL_ALGO` 取值** ──")
    fab = specs.NCCL_ALGO_FABRICATED
    known = [algo_value_known(n) for n in specs.NCCL_ALGO_NAMES]
    print("     编造的取值 `%r` ⇒ 校验器判**已知**？ **%s**（★ 必须是 `False`）"
          % (fab, algo_value_known(fab)))
    print("     ★ 而 8 个**真**取值**全部**判**已知** = %s" % all(known))
    print("     ★★★ 它**不报错**：`RingAllReduce` 是个**合法标识符**、看起来很像真的 —— "
          "唯一的信号是**校验器拒绝**。")
    check("C.1", (not algo_value_known(fab)) and all(known)
          and fab not in specs.NCCL_ALGO_NAMES,
          "★★★ **判据成立**：编造的取值 `%r` 被**拒绝**（不在 8 个真取值里），"
          "而 8 个真取值**全部**被接受 ⇒ ★★ 「编一个不存在的取值」**真的会失败**" % fab,
          "**编造的取值被接受了** —— ★ 那说明取值表被污染，或校验器写成了「永远返回真」")

    # ── C.2 ★★★ 把 `since` 版本记错 ─────────────────────────────────────
    print()
    print("  ── C.2 ★★★ **把 `since` 版本记错** ──")
    wrong = specs.NCCL_BAD_SINCE_CLAIM
    print("     断言 `NVLS` 的 `since` = `%s` ⇒ 版本账**拒绝**？ **%s**（★ 必须是 `True`）"
          % (wrong, rejects_since("NVLS", wrong)))
    print("     ★★★ 这个错法**看起来合理**：`%s` 是**真的存在**的一个版本档"
          "（`CollnetChain` / `CollnetDirect` 就在那）" % wrong)
    print("     正确答案：`NVLS` 的 `since` = **`%s`**" % since_of("NVLS"))
    check("C.2", rejects_since("NVLS", wrong) and since_of("NVLS") == "2.17+"
          and since_of("NVLS") != wrong
          and wrong in specs.NCCL_ALGO_SINCE,      # ★ 那个版本档**确实存在于表里**（不是瞎编的）
          "★★★ **判据成立**：把 `NVLS` 的 `since` 记成 `%s`（一个**真实存在的**版本档）"
          "会被版本账**拒绝** —— 正确值是 `2.17+` ⇒ ★★ **「记错版本」真的会失败**" % wrong,
          "**记错的版本没被拒绝** —— ★ 先查 `NCCL_ALGO_SINCE` 与 `NCCL_ALGO_NAMES` 的下标对应")


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

    # ── D.3 ★ 载体元数据（版本 / URL / 抓取日 / `[B']` 源码载体）───────────
    check("D.3", specs.NCCL_DOC_VERSION == "2.32.3"
          and specs.NCCL_DOC_BASE_URL
          == "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/"
          and specs.NCCL_FETCH_DATE == "2026-10-08"
          and specs.NCCL_SYSERR_SOURCE_TAG == "v2.26.2-1"
          and specs.NCCL_SYSERR_SOURCE_LOC == "src/init.cc:2198"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：文档版本 `%s` · URL · 抓取日 `%s`；"
          "★★ **`[B']` 源码载体** `%s` @ `%s` —— ★★★ **它与文档版本【不同】**"
          "（`P19` 的第一道防线：**不写版本与抓取日的引文不算引文**）；★ `UNSOURCED == ()` 一并钉住"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE,
             specs.NCCL_SYSERR_SOURCE_TAG, specs.NCCL_SYSERR_SOURCE_LOC))

    # ── D.3b ★★★ 口径守卫：两张表**等长**（否则 `since_of` 的下标会越界）──
    #   ★★★ 它**必须跑在任何计算之前**（本库 `P23`）：把 `NCCL_ALGO_SINCE` 少写一项时，
    #     `since_of()` 会 `IndexError` ⇒ 那是**崩溃捕获**（不算真捕获）。
    #     有了这一条，同一个变异就变成**断言捕获**。
    check("D.3b", len(specs.NCCL_ALGO_NAMES) == len(specs.NCCL_ALGO_SINCE)
          == specs.NCCL_ALGO_COUNT,
          "★★★ **口径守卫**：`NCCL_ALGO_NAMES` / `NCCL_ALGO_SINCE` / `NCCL_ALGO_COUNT`"
          "**三者等长（都是 %d）** ⇒ `since_of()` 的下标安全"
          "（★ 任一被改，**本条先失败**，而不是让 `since_of` 抛 `IndexError`）"
          % specs.NCCL_ALGO_COUNT,
          "**取值表与 `since` 表不等长** —— ★ 先查 `NCCL_ALGO_NAMES`/`NCCL_ALGO_SINCE`")

    # ── D.4 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）────────────
    check("D.4", specs.NCCL_ALGO_NAMES == ("Ring", "Tree", "Collnet", "CollnetChain",
                                           "CollnetDirect", "NVLS", "NVLSTree", "PAT")
          and specs.NCCL_ALGO_SINCE == ("2.5+", "2.5+", "2.5–2.13", "2.14+", "2.14+",
                                        "2.17+", "2.18+", "2.23+")
          and specs.NCCL_ALGO_COUNT == 8,
          "★★★ **两个元组逐元素 + 一个标量**钉住（`NCCL_ALGO_NAMES` 八档 · "
          "`NCCL_ALGO_SINCE` 八档 · `NCCL_ALGO_COUNT`）"
          "—— ★ 只钉「整体」会让**元素级**变异漏网（本库实测过：一次报出 58 条漏网）",
          "**取值表 / `since` 表 / 计数被动过**")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：引文逐字钉住 · 逐元素钉住
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · **逐元素**钉住")
    print("=" * 78)

    # ── D.5 ★★ `NCCL_ALGO` / `NCCL_PROTO` 的定义句与默认句 ────────────────
    check("D.5", specs.NCCL_ALGO_QUOTE
          == "The NCCL_ALGO variable defines which algorithms NCCL will use."
          and specs.NCCL_ALGO_DEFAULT_QUOTE
          == ("The default is unset, which causes NCCL to automatically choose the available "
              "algorithms based on the node topology and architecture.")
          and specs.NCCL_PROTO_QUOTE
          == ("The NCCL_PROTO variable defines which protocol(s) NCCL will be allowed to use.")
          and specs.NCCL_PROTO_VALUES == ("LL", "LL128", "Simple")
          and specs.NCCL_PROTO_WARNING_QUOTE
          == ("Users are discouraged from setting this variable … enabling LL128 on platforms "
              "that don't support it can lead to data corruption."),
          "★★ **五条引文逐字钉住**（`NCCL_ALGO` 定义句 / 默认句 · `NCCL_PROTO` 定义句 / "
          "三取值 / 官方警告）\n"
          "        ★★ 载体 = `env.html`（版本 %s，抓取日 %s），★ 全部标 `[B]`\n"
          "        ★★ 警告里的 `…` 是**计划自己的省略标记**（本库未取到原文），照抄不改"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE),
          "**`NCCL_ALGO` / `NCCL_PROTO` 的引文常量被动过**")

    # ── D.6 ★★★ 异步语义三条（`F1` 的落点）+ `F1` 的否定记录 ───────────────
    check("D.6", specs.NCCL_ASYNC_DEVICE_QUOTE
          == "The collective operation is then executed asynchronously on the CUDA device."
          and specs.NCCL_GROUP_NONBLOCK_QUOTE
          == ("All subsequent calls to NCCL until ncclGroupEnd will not block due to "
              "inter-CPU synchronization.")
          and specs.NCCL_GROUP_RETURN_QUOTE
          == "can return without having enqueued the operation on the stream"
          and specs.NCCL_NO_F1_QUOTE_NOTE
          == ("★「Communication is always asynchronous with respect to the host」不是 NCCL 文档的"
              "逐字引文（8 个文档页逐页 grep ⇒ 0 命中）")
          and "Communication is always asynchronous with respect to the host"
          not in specs.NCCL_ASYNC_DEVICE_QUOTE
          and "Communication is always asynchronous with respect to the host"
          not in specs.NCCL_GROUP_NONBLOCK_QUOTE
          and "Communication is always asynchronous with respect to the host"
          not in specs.NCCL_GROUP_RETURN_QUOTE,
          "★★★ **`F1` 的落点**：异步语义**只有这三条**已验证的表述 "
          "（`usage/streams.html` · `api/group.html` · `usage/groups.html`）；\n"
          "        ★★★ **`\"Communication is always asynchronous with respect to the host\"` "
          "【不在】任何一条里**（8 个文档页 grep ⇒ 0 命中）⇒ **不得**把它当引文",
          "**异步语义引文被动过，或被塞进了 F1 那条字面串**")

    # ── D.7 ★★★ 逐元素钉住（移除账 · 两条 `[E]` 反例输入 · `UNSOURCED`）────
    check("D.7", specs.NCCL_REMOVED_VARS == (("NCCL_TREE_THRESHOLD", "2.5"),
                                             ("NCCL_SINGLE_RING_THRESHOLD", "2.3"))
          and specs.NCCL_REMOVED_COUNT == 2
          and specs.NCCL_ALGO_FABRICATED == "RingAllReduce"
          and specs.NCCL_BAD_SINCE_CLAIM == "2.14+"
          and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：移除账两个变量（名字 + 版本，四个叶子）· 两条 `[E]` 反例输入"
          "（`NCCL_ALGO_FABRICATED` / `NCCL_BAD_SINCE_CLAIM`）· `UNSOURCED == ()`\n"
          "        ★★ `NCCL_ALGO_FABRICATED` = `%r` —— ★ 它是 `C.1` 的输入，**不是**任何来源里的值"
          % specs.NCCL_ALGO_FABRICATED,
          "**移除账或两条反例输入被动过**")


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
    print("★★★ **A = %d/%d** —— ★ **本主题【没有 A 类】**：本机无 GPU / 无 NCCL / 无真网卡，"
          "「算法正确性」**无处可验**；" % (_PASS["A"], _TOTAL["A"]))
    print("    ★★ 「常量钉住」**不算 A 类**（那是 `P12` 的变体）⇒ 本程序把它们全归了 `D` 类。")
    print("★ 依据在 `specs.py` 的 `# 来源：`（NCCL 官方产品文档 %s，抓取日 %s）。"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_FETCH_DATE))
    print("★ 本程序只核对「取值表 / 版本账 / 引文的字面」与「两条错法真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_architecture_and_selection.py —— NCCL 的算法取值、协议取值、与它怎么选")
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
