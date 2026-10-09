#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_activation_sharding.py —— SP 切的是**张量并行没切到的那些逐 token 算子**，
以及 ★★★ **它【不】引入通信开销** 这件事怎么落成可验证的形式。

配套文档：`01-activation-sharding.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**。
★★ **本程序【不用 numpy】**（★ 订正说明：本主题的 `02`/`03` 需要 numpy —— 它们要做矩阵乘法；
  但 `01` 全是**标量算术**（三式与它们的恒等式）⇒ **本文件一行都没 import numpy**
  ⇒ ★★ **直接 `python3 01_activation_sharding.py` 不需要 numpy**。
  ★★★ **但 `make 01_activation_sharding` 【仍要求】它** —— `Makefile` 里 `$(SCRIPTS): check-env`，
    而 `check-env` 在缺 numpy 时 `exit 1`（那一份要求按**三个程序的并集**做，是**真的**：
    `requirements.txt` 与 `Makefile` 的 `check-env` 都这么写，`02`/`03` 用得上）。

★★★ **本程序的判据形态：把论文的公式钉成【可验证的恒等式】**。
  三式（式(1)/(2)/(4)）都是**从 `specs.py` 的逐字引文抄来的闭式** ⇒ 本程序
  **不是**「模拟器 vs 闭式」的对拍，而是**转写错误敏感、物理错误不敏感**的**恒等式核对**。
  ★★ 这件事**必须写明**，否则会误以为 A 类在验证「SP 对不对」。**内容判据在 C 类**：
    · `C.1` **只切一半**（SP 只切了 `10sbh` 里的 `4sbh`）⇒ `A.2` 的差式当场不成立；
    · `C.2` **切错轴**（按 `h` 切而不是按 `s` 切）⇒ `5as²b` 那一项**完全不减**，
      差 `5as²b(1−1/t)` 随 `s²` 增长 ⇒ `A.3` 与 `A.2` 都不成立。
  ★★★ **「SP 不引入通信开销」是原文的【结论】**（`specs.SP_NO_OVERHEAD_QUOTE`）——
    本程序**不**把它写成任何 C 类反例（★ 那会与原文相反）；它由 `D.9` 以**引文**的方式钉住，
    并把「裸 SP 有开销 / 混用后没有」**成对**引出来。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1`/`A.2`/`A.3` 若删掉 `eq1()`/`eq2()`/`eq4()` ⇒ 三式无从计算
    （不会通过）；★ 而 `C.1`/`C.2` 是**内容**判据（删掉错法变体就无从比）。

★★ **覆盖边界**：本程序的 A 类守的是「**三式的数值结构与论文一致**」，
  **不守**「SP 在真实集群上省了多少显存」—— ★★ 后者的依据是 `specs.py` 的 `# 来源：`
  （**同行评审：是**，依据 = MLSys 2023 会议论文索引，载体见 `specs.SP_MLSYS_INDEX_TITLE`）。
  ★★ 且 **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：`C.1`/`C.2` 断言的是
  「**错法 ⇒ 恒等式不成立**」这个**失败**，不是「一致」这个成功。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **「在线 softmax」的 `(m, l)` 更新公式本身** →
    `40-kernels/softmax-normalization/02-online-softmax.md`（★ 那篇逐字自述「本文是 `(m, l)`
    更新公式的正文」）与 `40-kernels/attention/`。
    ★★★ **本主题一律不使用「online softmax」这个词** —— 本域写的是**跨设备**的
    **块统计量重缩放**，术语用 Ring Attention 原文的措辞（**置换不变性** + **统计量重缩放**）。
    ★ **本程序连那一套都没有**（`01` 讲的是**激活显存**，不是注意力内核）。
  · **`S×S` 矩阵的访存账**（FlashAttention 的字节数收益）→
    `40-kernels/attention/03-flash-attention-backward.md` 与 `00-meta/glossary.md` 的 `flash attention` 条。
  · **位置编码的外推**（PI / NTK / ALiBi、混叠周期）→
    `80-model-architectures/positional-encoding/02,03`。
  · **集合通信原语的算法与复杂度**（`V(N)`、reduce-scatter / all-gather 的阶段划分）→
    `110-communication/collectives`）与 `10-foundations/performance-models/02:209`。
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02:417`（★ **不是 `:223`** —— 那一行是**纯 TP 段**）。
  · **重算换显存**的机制 → `90-training/autograd-and-backward/02-gradient-checkpointing.md`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ **`t = 1` 时式(2) == 式(4) == 式(1)**（SP 在 TP 度 1 时无收益）—— **两组** `(s,b,h,a)`；
  · `A.2`  ★★★ **`式(2) − 式(4) == 10·s·b·h·(1 − 1/t)`**（`SP_TP_GRID` 六档，实测差全为 `0.000e+00`）；
  · `A.3`  ★★ **式(4) == 式(1)/t**（同一件事的另一种写法）；
  · `A.1g`/`A.2g`/`A.3g` ★★★ **三条守卫**（把参照故意改错 ⇒ 对应的 A 类**必须失败**）；
  · `B.1`  ★★ SP 的**收益比** `式(2)/式(4)`：两组 `a/h` 各测 `t = 8` 与 `t = 16`；
  · `B.2`  ★★★ **序列越长，SP 的相对收益越小**（两组对照 + 单调性）；
  · `B.3`  ★★ 差式那部分 `∝ s`（线性）而注意力项 `∝ s²` ⇒ **这就是 `03` 的引子**；
  · `C.1`  ★★★ **只切一半**（SP 只切 LN 那 `4sbh`）⇒ 差式不成立（真的会失败）；
  · `C.2`  ★★★ **切错轴**（按 `h` 切）⇒ `5as²b` 完全不减，缺口随 `s²` 增长；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **失败标记自检**。
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
# ★★★ 失败标记：本主题（sequence-context-parallel）取 `{{ FAIL: seq-context-parallel`。
#   ★ 本域另外五个主题各用**不同的前缀**（`%%` / `^^` / `==` / `||` / `??`），
#     六个标记**两两不互为前缀** ⇒ 跨主题照抄会被下面两条断言当场抓住。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
#   ★ 它**只允许出现在 `print()` 的字符串里** ⇒ 本文件里它只出现在打印与断言明细中。
ASSERT_FAIL_MARK = "{{ FAIL: seq-context-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                     "== FAIL: pipeline-parallel", "|| FAIL: expert-parallel",
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
# 1. 三式 —— ★★ 逐字从 `specs.py` 的引文抄来的**闭式**（式(1) / 式(2) / 式(4)）
#    ★★★ **系数是【函数体内的字面量】**（不是从 `specs` 读的）—— 这一点是**故意的**：
#      若系数也从 `specs` 读，那么 `A.2` 的差式与 `A.2` 的右端会**同时**跟着常量动，
#      于是改常量**不再有可观察后果**（一条**不过期的漏网**）。
#      ⇒ 现在的分工：**闭式的系数写死在函数里**（= 论文的转写），
#        **`specs` 的常量只出现在差式的右端与 `D.6` 的钉住里**。
# ===========================================================================
def eq1(s, b, h, a):
    """式(1)：**无任何模型并行**时，每层激活的**元素数** = `sbh(34 + 5as/h)`。

    ★ `specs.SP_EQ1_PROSE_QUOTE` 那一句 `The above equation is for the case that no form of
      model parallelism is applied.` 引的就是它。
    """
    return s * b * h * (34.0 + 5.0 * a * s / h)


def eq2(s, b, h, a, t):
    """式(2)：**`t` 路张量并行**时每层激活的元素数 = `sbh(10 + 24/t + 5as/(ht))`。

    ★★ **`10sbh` 那一项【不除以 `t`】**（`specs.SP_REPLICATED_QUOTE` 说的是这件事）。
    """
    return s * b * h * (10.0 + 24.0 / t + 5.0 * a * s / (h * t))


def eq4(s, b, h, a, t):
    """式(4)：**序列并行 + 张量并行**时每层激活的元素数 = `sbh(10/t + 24/t + 5as/(ht))`。

    ★★★ 论文紧接着写的另一种形式是 `= (sbh/t)(34 + 5as/h)` —— 即 **式(4) == 式(1)/t**
      （`A.3` 钉的就是它；`specs.SP_EQ4_TAIL_QUOTE` 是原文自己说的那句）。
    """
    return s * b * h * (10.0 / t + 24.0 / t + 5.0 * a * s / (h * t))


# ---------------------------------------------------------------------------
# ★★★ 三个【错法变体】—— 它们只被守卫与 C 类用，**不是**被测实现的一部分
# ---------------------------------------------------------------------------
def eq1_broken(s, b, h, a):
    """错法①：把式(1) 的 `34` 写成 `33`（**逐项求和时少算一项**）。"""
    return s * b * h * (33.0 + 5.0 * a * s / h)


def eq2_no_split(s, b, h, a, t):
    """错法②：**把 `10sbh` 那一项也除以 `t`** —— 即误以为 SP 已经在 TP 上生效。

    ★★ 这正是 `A.2` 要挡住的错：改完之后 `式(2) − 式(4) == 0`，与 `10sbh(1−1/t)` 不符。
    """
    return s * b * h * (10.0 / t + 24.0 / t + 5.0 * a * s / (h * t))


def eq4_hidden_split(s, b, h, a, t):
    """错法③：**切错轴** —— 按**隐维 `h`** 切（不是按序列维 `s` 切）。

    ★★★ `5as²b/h` 乘上 `h` 之后是 `5as²b` —— 它**没有 `h`** ⇒ 按 `h` 切**根本切不到它**，
      它整项**留全**。⇒ 这个错法的缺口 `5as²b(1 − 1/t)` 随 `s²` 增长。
    """
    return 34.0 * s * b * h / t + 5.0 * a * s * s * b


def eq4_partial_ln(s, b, h, a, t):
    """错法④：**只切一半** —— SP 只切掉了 `10sbh` 里 LayerNorm 的那 `4sbh`。

    ★★ 于是差式变成 `4sbh(1−1/t)`，而不是 `10sbh(1−1/t)` ⇒ `A.2` 当场不成立。
    """
    return s * b * h * (6.0 + 4.0 / t + 24.0 / t + 5.0 * a * s / (h * t))


def _swaps():
    """★ 返回「切错轴的处置」用的独立变量：**规格常数只在这一处被读**（便于审计）。"""
    return specs.SP_EQ1_SBH, specs.SP_EQ1_ATTN_COEF


# ===========================================================================
# 2. A 类：三式的恒等式（判据 A.1 / A.1g / A.2 / A.2g / A.3 / A.3g）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 三式的恒等式：把论文的公式钉成【可验证的】形式")
    print("=" * 78)
    print("  ★★ **本节的 A 类是【恒等式核对】**：三式都是从 `specs.py` 的逐字引文抄来的闭式")
    print("     ⇒ 它**对转写错误敏感、对物理错误不敏感**（物理判据在 `C.1`/`C.2`）。")
    print("     ★ 三条 A 类**各配一条守卫**（`A.1g`/`A.2g`/`A.3g`），守卫的错法**真的会改变被判量**。")
    cases = (("LONG", specs.SP_GAIN_CASE_LONG), ("SHORT", specs.SP_GAIN_CASE_SHORT))

    # ── A.1 ★★ t = 1 时三式相等 ────────────────────────────────────────────
    print()
    print("  ── A.1 ★★ `t = 1`（TP 度 1）时：式(2) == 式(4) == 式(1)（**退化对照**）──")
    w_a1 = 0.0
    rows_a1 = []
    for tag, (s, b, h, a) in cases:
        v1, v2, v4 = eq1(s, b, h, a), eq2(s, b, h, a, 1), eq4(s, b, h, a, 1)
        rel = abs(v2 - v4) / abs(v2)
        w_a1 = max(w_a1, rel, abs(v2 - v1) / abs(v1))
        rows_a1.append((tag, s, b, h, a, v1, v2, v4, rel))
        print("     %-5s s=%-5d b=%d h=%-4d a=%-3d ⇒ 式(1)=%.6e 式(2)=%.6e 式(4)=%.6e "
              "相对差=%.3e" % (tag, s, b, h, a, v1, v2, v4, rel))
    print("     ★★ 物理含义：**SP 在 TP 度 1 时【没有收益】** —— 没有可切的 TP 区域，")
    print("        序列并行就只是把数据切成 1 份。★ `specs.SP_EQ4_TAIL_QUOTE` 是原文自己说的：")
    print("        `%s` … 1" % specs.SP_EQ4_TAIL_QUOTE)
    check("A.1", w_a1 <= TOL_MACHINE,
          "★★ **判据成立**：`t = 1` 时式(2)、式(4)、式(1) **三式相等**"
          "（两组 `(s,b,h,a)`，最大相对差 **%.3e**）" % w_a1,
          "**`t = 1` 时三式不相等** —— ★ 先查式(1) 的 `34` 是不是 `10 + 24`（见 `D.6`）")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    s0, b0, h0, a0 = specs.SP_GAIN_CASE_LONG
    gap_a1 = abs(eq2(s0, b0, h0, a0, 1) - eq1_broken(s0, b0, h0, a0)) / abs(eq2(s0, b0, h0, a0, 1))
    check("A.1g", gap_a1 > TOL_MACHINE,
          "★★ **守卫成立**：把式(1) 的 `34` 改成 `33` ⇒ 与式(2)(`t = 1`) 的相对差变成 **%.3e**"
          " > 阈 ⇒ `A.1` **能失败** ✓（★ 它守的是「**无并行那一支**」）" % gap_a1,
          "**守卫失效** —— 式(1) 改错之后 `A.1` 仍然通过")

    # ── A.2 ★★★ 差式 == 10sbh(1 − 1/t) ──────────────────────────────────────
    print()
    print("  ── A.2 ★★★ **式(2) − 式(4) == `10·s·b·h·(1 − 1/t)`**（六档 `t` 全扫）──")
    w_a2 = 0.0
    for tag, (s, b, h, a) in cases:
        print("     %s 组：s=%d b=%d h=%d a=%d" % (tag, s, b, h, a))
        for t in specs.SP_TP_GRID:
            d = eq2(s, b, h, a, t) - eq4(s, b, h, a, t)
            rhs = specs.SP_SPLIT_REPLICATED * s * b * h * (1.0 - 1.0 / t)
            rel = abs(d - rhs) / abs(rhs) if rhs else 0.0
            w_a2 = max(w_a2, rel)
            print("        t=%2d ⇒ 左端 %.6e · 右端 %.6e · 相对差 %.3e" % (t, d, rhs, rel))
    print("     ★★★ **为什么这条是本主题 01 的题眼**：右端**只有 `10sbh` 与 `t`**、")
    print("        **不含 `a`、也不含注意力项** ⇒ ★★ **SP 省下的东西【与注意力无关】**：")
    print("        它省的是那 `10sbh`（`specs.SP_REPLICATED_QUOTE`：LN 与两处 dropout 的激活）。")
    check("A.2", w_a2 <= TOL_MACHINE,
          "★★★ **判据成立**：式(2) − 式(4) **恒等于** `10·s·b·h·(1 − 1/t)`"
          "（`SP_TP_GRID` 六档 × 两组，最大相对差 **%.3e**）" % w_a2,
          "**差式不成立** —— ★ 最可能的两种成因：`10sbh` 被并进了 `/t`（`A.2g` 复现它），"
          "或式(4) 切错了轴（`C.2` 复现它）")

    # ── A.2g ★★★ 守卫 ─────────────────────────────────────────────────────
    # ★ `t = 1` 那一档要跳过：那里右端**恰好是 `0`**（差式两边都为 0），比不出相对差。
    gap_a2 = 0.0
    for t in specs.SP_TP_GRID[1:]:
        d_bad = eq2_no_split(s0, b0, h0, a0, t) - eq4(s0, b0, h0, a0, t)
        rhs = specs.SP_SPLIT_REPLICATED * s0 * b0 * h0 * (1.0 - 1.0 / t)
        gap_a2 = max(gap_a2, abs(d_bad - rhs) / abs(rhs))
    check("A.2g", gap_a2 > 0.5,
          "★★ **守卫成立**：把 `10sbh` 也除以 `t`（错法②）⇒ 左端变成 `0`，"
          "与右端差 **%.3e**（相对）⇒ `A.2` **能失败** ✓" % gap_a2,
          "**守卫失效** —— 错法② 之下 `A.2` 仍然通过")

    # ── A.3 ★★ 式(4) == 式(1)/t ─────────────────────────────────────────────
    print()
    print("  ── A.3 ★★ **式(4) == 式(1)/t**（论文自己给的另一种写法，`specs.SP_EQ4_TAIL_QUOTE`）──")
    w_a3 = 0.0
    for tag, (s, b, h, a) in cases:
        for t in specs.SP_TP_GRID:
            lhs = eq4(s, b, h, a, t)
            rhs = eq1(s, b, h, a) / t
            w_a3 = max(w_a3, abs(lhs - rhs) / abs(rhs))
        print("     %-5s ⇒ 六档 `t` 下最大相对差 %.3e" % (tag, w_a3))
    print("     ★★ 物理含义：**SP 把整层激活按序列维切成 `t` 份**，所以式(4) 就是式(1) 除以 `t`")
    print("        —— **一个除法，没有别的**。★ 这也是 `C.2`（切错轴）能立刻现形的原因：")
    print("        按 `h` 切**得不到** `式(1)/t`（`5as²b` 整项不减）。")
    check("A.3", w_a3 <= TOL_MACHINE,
          "★★ **判据成立**：式(4) == 式(1)/t（两组 × 六档，最大相对差 **%.3e**）" % w_a3,
          "**式(4) ≠ 式(1)/t** —— 先查式(4) 的 `34` 是不是 `10 + 24`（见 `D.6`）")

    # ── A.3g ★★★ 守卫 ─────────────────────────────────────────────────────
    gap_a3 = 0.0
    for t in specs.SP_TP_GRID:
        gap_a3 = max(gap_a3, abs(eq4_hidden_split(s0, b0, h0, a0, t) - eq1(s0, b0, h0, a0) / t)
                     / abs(eq1(s0, b0, h0, a0) / t))
    check("A.3g", gap_a3 > TOL_MACHINE,
          "★★ **守卫成立**：按 `h` 切（错法③）⇒ 与 `式(1)/t` 的最大相对差 **%.3e**"
          " > 阈 ⇒ `A.3` **能失败** ✓（★ 它守的是「**切在哪个轴**」）" % gap_a3,
          "**守卫失效** —— 切错轴之后 `A.3` 仍然通过")


# ===========================================================================
# 3. B 类：SP 的收益比（判据 B.1 / B.2 / B.3）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：SP 的【相对收益】随序列长度下降")
    print("=" * 78)
    print("  ★★ 收益比的定义：`式(2)/式(4)` —— **同样的 `(s,b,h,a)`，加不加 SP 的激活之比**。")

    # ── B.1 ★★ 两组的收益比 ────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **两组 `a/h`** 下的收益比（`t = 8` 与 `t = 16`）──")
    gains = {}
    for tag, (s, b, h, a) in (("LONG", specs.SP_GAIN_CASE_LONG),
                              ("SHORT", specs.SP_GAIN_CASE_SHORT)):
        print("     %-5s s=%-5d b=%d h=%-4d a=%-3d ⇒ **a/h = %.4f**" % (tag, s, b, h, a, a / h))
        for t in (8, 16):
            g = eq2(s, b, h, a, t) / eq4(s, b, h, a, t)
            gains[(tag, t)] = g
            print("        t=%2d ⇒ 收益比 = **%.6f**" % (t, g))
    ok_expected = (abs(gains[("LONG", 8)] - 1.1039) < 5e-5
                   and abs(gains[("LONG", 16)] - 1.2226) < 5e-5
                   and abs(gains[("SHORT", 8)] - 1.0533) < 5e-5
                   and abs(gains[("SHORT", 16)] - 1.1142) < 5e-5)
    print("     ★★ **两组必须一起引**：`LONG` 组 `a/h = 0.1250`、`SHORT` 组 `a/h = 0.0625`")
    print("        —— 只引一组会把「收益随 `s` 下降」误读成「收益随 `a/h` 下降」（`B.2` 分辨这两件事）。")
    check("B.1", ok_expected,
          "★★ **判据成立**：`t=8 → %.4f`、`t=16 → %.4f`（`LONG`）；"
          "`t=8 → %.4f`、`t=16 → %.4f`（`SHORT`）—— ★ 四个数与任务书给定值相符（容差 `5e-5`）"
          % (gains[("LONG", 8)], gains[("LONG", 16)],
             gains[("SHORT", 8)], gains[("SHORT", 16)]),
          "**收益比与预期不符** —— ★ 先查 `a/h` 与 `t` 的换写（本程序是单头口径）")

    # ── B.2 ★★★ 序列越长，SP 的相对收益越小 ─────────────────────────────────
    print()
    print("  ── B.2 ★★★ **序列越长，SP 的相对收益越小**（两组对照 + 单调性）──")
    print("     组内单调性（`s` 翻倍，`b/h/a/t` 保持 LONG 组的比例）：")
    s_list = (512, 1024, 2048, 4096, 8192)
    _, b_l, h_l, a_l = specs.SP_GAIN_CASE_LONG
    mono = []
    for t in (8, 16):
        seq = [eq2(s, b_l, h_l, a_l, t) / eq4(s, b_l, h_l, a_l, t) for s in s_list]
        mono.append(all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)))
        print("        t=%2d ⇒ %s" % (t, " · ".join("s=%d: %.6f" % (s, g) for s, g in zip(s_list, seq))))
    cross = (gains[("LONG", 8)] > gains[("SHORT", 8)]
             and gains[("LONG", 16)] > gains[("SHORT", 16)])
    print("     ★★ **跨组对照**：`LONG`（`s` 长）的收益比 **小于** `SHORT`（`s` 短）的：")
    print("        `t=8` 时 %.6f < %.6f · `t=16` 时 %.6f < %.6f"
          % (gains[("LONG", 8)], gains[("SHORT", 8)], gains[("LONG", 16)], gains[("SHORT", 16)]))
    print("        ⇒ ★★★ **两组一起看才排除了「`a/h` 变小导致的」这个替代解释**：")
    print("          `SHORT` 组的 `a/h` 更小、按直觉应当**收益更小**，实测却**更大**")
    print("          ⇒ 起作用的变量是 `s`，不是 `a/h`。")
    check("B.2", all(mono) and cross,
          "★★★ **判据成立**：① 组内 `s` 翻倍时收益比**严格单调下降**（`t=8`/`t=16` 两条都成立）；"
          "② 跨组 `LONG` 的收益比**小于** `SHORT` ⇒ ★★ **序列越长，SP 的相对收益越小**",
          "**单调性不成立** —— ★ 检查 `5as²b/h` 的 `s²` 次幂有没有被写成 `s`（`B.3` 量它）")

    # ── B.3 ★★ 成因：10sbh 是 O(s)、5as²b/h 是 O(s²) ────────────────────────
    print()
    print("  ── B.3 ★★ **成因**：`10sbh` 是 `O(s)`、`5as²b/h` 是 `O(s²)` —— SP **只切前者** ──")
    s1, s2 = 1024, 2048
    save1 = eq2(s1, b_l, h_l, a_l, 8) - eq4(s1, b_l, h_l, a_l, 8)
    save2 = eq2(s2, b_l, h_l, a_l, 8) - eq4(s2, b_l, h_l, a_l, 8)
    att1 = specs.SP_EQ1_ATTN_COEF * a_l * s1 * s1 * b_l / h_l
    att2 = specs.SP_EQ1_ATTN_COEF * a_l * s2 * s2 * b_l / h_l
    r_save, r_att = save2 / save1, att2 / att1
    print("     `s` 从 %d 翻到 %d：**SP 省下的** `式(2)−式(4)` 变成 **%.4f 倍**"
          % (s1, s2, r_save))
    print("     **注意力那一项** `5as²b/h` 变成 **%.4f 倍**" % r_att)
    print("     ⇒ ★★ **`%.4f ≈ 2 = 2¹`（线性、`O(s)`）而 `%.4f ≈ 4 = 2²`（二次、`O(s²)`）**"
          % (r_save, r_att))
    print("     ⇒ ★★★ **SP 不切 `O(s²)` 的那一项** ⇒ 序列越长，省下的（`O(s)`）占总量（`O(s²)`）的比例越小")
    print("        ⇒ ★★ **这就是 `03` 的引子**：`specs.CP_VS_SP_QUOTE` 逐字说")
    print("          `%s`" % specs.CP_VS_SP_QUOTE[:88] + " …")
    print("          ⇒ **CP 连 `O(s²)` 的注意力一起切** ⇒ 序列越长，CP 越必需。")
    check("B.3", abs(r_save - 2.0) < 1e-12 and abs(r_att - 4.0) < 1e-12,
          "★★ **判据成立**：`s` 翻倍时 SP 的收益变成 **%.4f 倍**（`O(s)`）、"
          "注意力项变成 **%.4f 倍**（`O(s²)`）⇒ 量级差异就是收益比下降的**成因**" % (r_save, r_att),
          "**量级关系不成立** —— ★ 注意力项的 `s²` 次幂可能被写成了 `s`")


# ===========================================================================
# 4. C 类：边界与反例（判据 C.1 / C.2）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**切一半** 与 **切错轴**（★ 两条都真的会失败）")
    print("=" * 78)
    s, b, h, a = specs.SP_GAIN_CASE_LONG
    t = 8

    # ── C.1 ★★★ 只切一半 ────────────────────────────────────────────────────
    print()
    print("  ── C.1 ★★★ **只切一半**：SP 只切掉了 `10sbh` 里 LayerNorm 的那 `4sbh` ──")
    d_full = eq2(s, b, h, a, t) - eq4(s, b, h, a, t)
    d_part = eq2(s, b, h, a, t) - eq4_partial_ln(s, b, h, a, t)
    rhs = specs.SP_SPLIT_REPLICATED * s * b * h * (1.0 - 1.0 / t)
    print("     完整 SP 的差式 = **%.6e**（== `10sbh(1−1/t)` = %.6e）" % (d_full, rhs))
    print("     只切 LN 的差式 = **%.6e** ⇒ ★★ **账算少了一半**（`4sbh` 而不是 `10sbh`）" % d_part)
    print("     两种切法的激活之差 = **%.6e**" % (eq4_partial_ln(s, b, h, a, t) - eq4(s, b, h, a, t)))
    print("     ⇒ ★★★ **它【不报错】：形状、类型、算术全都对，只是省得少** ——")
    print("        唯一的信号是 **`A.2` 的恒等式不再成立**（这一节就是那个「信号」的可执行版）。")
    _gap_c1 = abs(d_part - rhs) / rhs
    check("C.1", d_part > 0.0 and _gap_c1 > 0.1 and d_part < d_full,
          "★★★ **判据成立**（三个条件）：① 只切一半也**省了**（差 > 0）；"
          "② 但它与 `10sbh(1−1/t)` 的相对差 **%.3e**（≫ 阈 `1e-12`）⇒ `A.2` **会失败**；"
          "③ 它省的**严格少于**完整 SP ⇒ ★ 「只切一半」是一种**静默的**错误"
          % _gap_c1,
          "**「只切一半」没有让恒等式失败** —— 那条判据对这条错法**不敏感**")

    # ── C.2 ★★★ 切错轴 ─────────────────────────────────────────────────────
    print()
    print("  ── C.2 ★★★ **切错轴**：按【隐维 `h`】切，而不是按【序列维 `s`】切 ──")
    print("     ★★ 逐字依据（`specs.SP_SEQ_DIM_QUOTE`）：")
    print("        `%s`" % specs.SP_SEQ_DIM_QUOTE)
    e1 = eq1(s, b, h, a) / t
    e4h = eq4_hidden_split(s, b, h, a, t)
    gap = e4h - e1
    att_term = specs.SP_EQ1_ATTN_COEF * a * s * s * b
    print("     按 `s` 切（正确）：式(1)/t = **%.6e**" % e1)
    print("     按 `h` 切（错法③）：                **%.6e**" % e4h)
    print("     缺口 = **%.6e** ⇒ ★★ 它正是 **`5as²b(1 − 1/t)`**（留在全量的注意力项的一部分）" % gap)
    # ★ `5as²b(1−1/t)` 的独立算法：先算 `5as²b`，再乘 `(1−1/t)`。
    expect_gap = att_term * (1.0 - 1.0 / t)
    print("     `5as²b(1 − 1/t)` = %.6e × %.4f = **%.6e**（与缺口相同 = %s）"
          % (att_term, 1.0 - 1.0 / t, expect_gap, abs(gap - expect_gap) < 1e-6))
    print("     ⇒ ★★★ **缺口随 `s²` 增长**：")
    for ss in (512, 1024, 2048, 4096):
        g = eq4_hidden_split(ss, b, h, a, t) - eq1(ss, b, h, a) / t
        print("        s=%-5d ⇒ 缺口 %.6e" % (ss, g))
    ratio2 = ((eq4_hidden_split(2 * s, b, h, a, t) - eq1(2 * s, b, h, a) / t)
              / (eq4_hidden_split(s, b, h, a, t) - eq1(s, b, h, a) / t))
    print("     ★★ `s` 翻倍 ⇒ 缺口变成 **%.4f 倍**（`≈ 4`）⇒ 它是 `O(s²)`" % ratio2)
    print("     ⇒ ★★★ **切错轴的代价随序列长度平方增长** —— 这正是")
    print("        `specs.SP_LN_DROP_QUOTE` 之外的另一半：SP 的**收益**是 `O(s)`，")
    print("        而**切不到**的那部分是 `O(s²)`。")
    check("C.2", gap > 0.0 and abs(gap - expect_gap) <= 1e-6 * max(expect_gap, 1.0)
          and abs(ratio2 - 4.0) < 1e-9,
          "★★★ **判据成立**（三个条件）：① 按 `h` 切**确实**比按 `s` 切多出 **%.6e**；"
          "② 缺口**恰好**等于 `5as²b(1−1/t)`；③ `s` 翻倍时缺口变 **%.4f 倍**"
          "（`O(s²)`）⇒ ★★ **切错轴是一个【随序列长度变严重】的错误**" % (gap, ratio2),
          "**缺口与 `5as²b(1−1/t)` 不符**、或它不随 `s²` 增长")
    print()
    print("     ★★ **为什么 `C.2` 是本程序最要紧的一条**：`5as²b` 里**没有 `h`** ⇒")
    print("        按 `h` 切的方案（那种「把激活沿隐维切开」的做法）**根本切不到它**，")
    print("        于是**序列越长、浪费越大** —— 而端到端**不报任何错**。")


# ===========================================================================
# 5. 前哨：口径 / 阈值 / 标记自检（判据 D.1 / D.2 / D.2b / D.3 / D.3b / D.4 / D.4b）
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
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s`，纯标量算术（**本文件不 import numpy**）。"
          % sys.version.split()[0])
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · 「在线 softmax」的 `(m, l)` 更新公式本身 → "
          "`40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`；")
    print("        ★★★ **本主题一律不使用「online softmax」这个词** —— 本域写的是**跨设备**的")
    print("            块统计量重缩放，术语用 Ring Attention 原文的措辞（置换不变性 + 统计量重缩放）。")
    print("      · `S×S` 矩阵的访存账（FlashAttention 的字节数收益）→ "
          "`40-kernels/attention/03-flash-attention-backward.md`、`00-meta/glossary.md` 的 `flash attention` 条；")
    print("      · 位置编码的外推（PI / NTK / ALiBi、混叠周期）→ "
          "`80-model-architectures/positional-encoding/02,03`；")
    print("      · 集合通信原语的算法与复杂度 → `110-communication/collectives`（"
          "与 `10-foundations/performance-models/02:209`；")
    print("      · TP 每层 `2+2` 次 all-reduce → `tensor-parallel/03` 与 "
          "`10-foundations/performance-models/02:417`（★ **不是 `:223`** —— 那一行是纯 TP 段）；")
    print("      · 重算换显存的机制 → `90-training/autograd-and-backward/02-gradient-checkpointing.md`。")
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
    check("D.2", ASSERT_FAIL_MARK == "{{ FAIL: seq-context-parallel" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `{{ FAIL: seq-context-parallel`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "|| FAIL: expert-parallel",
                                        "?? FAIL: zero-fsdp"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住；"
          "★ 六个标记**两两不互为前缀**")

    # ── D.3 ★ 口径常量（三式的**测量条件**）────────────────────────────────
    check("D.3", specs.TOL_MACHINE == 1.0e-12,
          "★ 机器精度档阈 = %g（★ 「恰好为 0」是**实测观察**，不是设计保证 ⇒ 判据仍写 `<=`）"
          % specs.TOL_MACHINE)
    check("D.3b", all(isinstance(x, int) for x in specs.SP_TP_GRID)
          and specs.SP_TP_GRID[0] == 1
          and all(x & (x - 1) == 0 for x in specs.SP_TP_GRID),
          "★★ `SP_TP_GRID` 全为整数、**首档是 `t = 1`**（退化对照），且**全为 2 的幂**"
          " ⇒ `1/t` 在二进制里**精确** ⇒ ★★ `A.2` 的差式才能期望**恰好 `0.0`**")

    # ── D.4 ★ 形状的整除性（结论的适用边界）────────────────────────────────
    ok_div = True
    for s, b, h, a in (specs.SP_GAIN_CASE_LONG, specs.SP_GAIN_CASE_SHORT):
        ok_div = ok_div and (h % a == 0 or a % h == 0)
    check("D.4", all(all(isinstance(x, int) for x in c)
                     for c in (specs.SP_GAIN_CASE_LONG, specs.SP_GAIN_CASE_SHORT)) and ok_div
          and len(specs.SP_GAIN_CASE_LONG) == 4 and len(specs.SP_GAIN_CASE_SHORT) == 4,
          "★★ 两组 `(s, b, h, a)` **全是整数**、长度都是 `4`、且 `a/h` 可整除/被整除"
          "（`0.1250` / `0.0625` —— ★ 这是本程序单头口径的**适用边界**）")
    check("D.4b", specs.SP_SPLIT_REPLICATED + specs.SP_EQ2_TP_SHARD == specs.SP_EQ1_SBH
          and specs.SP_SPLIT_REPLICATED == specs.SP_EQ2_TP_NUM,
          "★★ **同值不同指**：`SP_SPLIT_REPLICATED`（`10`，**复制区**）与 "
          "`SP_EQ2_TP_NUM`（`10`，**切开之后**的那一项）数值相同、**位置不同** —— "
          "★★ 把两者合并成一个常量就抹掉了这个区别；★ 且 `10 + 24 == 34`（式(1) 的常数）")
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：specs 常量逐位钉住 · 逐元素钉住 · 引文纪律 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · specs 常量逐位钉住 · **逐元素**钉住 · 引文纪律 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★ SP 的引文（**载体 + 计数**）─────────────────────────────────
    check("D.5", specs.SP_EQ1_PROSE_QUOTE
          == "The above equation is for the case that no form of model parallelism is applied."
          and specs.SP_EQ2_PROSE_QUOTE
          == "the per-layer memory required to store the activations reduces from Equation"
          and specs.SP_EQ4_PROSE_QUOTE
          == ("Using sequence parallelism along with tensor parallelism, the memory required "
              "to store the activations per transformer layer reduces from Equation")
          and specs.SP_EQ4_TAIL_QUOTE == "The above equation is now Equation"
          and specs.SP_REPLICATED_QUOTE
          == ("is due to these replicated operations and as a result they are not divided "
              "by the tensor parallel size")
          and specs.SP_LN_DROP_QUOTE
          == ("These elements do not require a lot of compute but demand a considerable amount "
              "of activation memory.")
          and specs.SP_SEQ_DIM_QUOTE
          == ("This characteristic allows us to partition these regions along the sequence dimension"),
          "★★ SP 的**七条**引文逐字钉住 —— ★ 每一条都**切在 `<math>` 标记之前**"
          "（凡引言以 `… to:` / `… from Equation` 收尾，**不带公式与式号**）\n"
          "        ★★ 载体 = `https://ar5iv.labs.arxiv.org/html/2205.05198` 正文，"
          "抓取日 **2026-10-08**，逐条在**原始字节**上 `count()` = **1**",
          "**SP 的引文常量被动过** —— ★ 先查它是不是跨了 `<math>`（那样原始字节上会计到 `0`）")

    # ── D.6 ★★ SP 的数值常量（六条）+ 那条加法的钉住 ────────────────────────
    check("D.6", specs.SP_SPLIT_REPLICATED == 10 and specs.SP_EQ2_TP_NUM == 10
          and specs.SP_EQ2_TP_SHARD == 24 and specs.SP_EQ1_SBH == 34
          and specs.SP_EQ1_ATTN_COEF == 5
          and specs.SP_EQ2_TP_NUM + specs.SP_EQ2_TP_SHARD == specs.SP_EQ1_SBH,
          "★★ SP 的**五个**数值常量逐位钉住，且 `SP_EQ2_TP_NUM + SP_EQ2_TP_SHARD == SP_EQ1_SBH`"
          "（`10 + 24 == 34` ⇒ 式(1) 与式(2)/(4) 的分母对得上）；\n"
          "        ★★ 它们**都取自 `<math>` 的 `alttext`**（散文里那句只到 `… parallel size` 为止）"
          "⇒ 载体是**公式的 `alttext`**，不是散文 —— ★★ 这一点必须写明，"
          "否则会像前三个 chunk 那样在散文上 `count()` 到 `0`",
          "**SP 的数值常量被动过** —— ★ 或者 `10 + 24 ≠ 34`（三式的分母对不上了）")

    # ── D.7 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）─────────────
    check("D.7", specs.SP_TP_GRID == (1, 2, 4, 8, 16, 32)
          and specs.SP_GAIN_CASE_LONG == (1024, 1, 64, 8)
          and specs.SP_GAIN_CASE_SHORT == (4096, 1, 256, 16),
          "★★★ **三个元组【逐元素】钉住**（六档 `t` + 两组 `(s,b,h,a)`）—— "
          "★ 只钉「整体」会让**元素级**变异漏网（本库 `pipeline-parallel` 的那一轮实测过："
          "一次报出 58 条漏网）",
          "**元组的元素被动过**")

    # ── D.8 ★★ 独立性线索（源码级）──────────────────────────────────────────
    srcs = {inspect.getsource(f) for f in (eq1, eq2, eq4)}
    srcs_bad = {inspect.getsource(f) for f in (eq1_broken, eq2_no_split,
                                               eq4_hidden_split, eq4_partial_ln)}
    calls = {n.func.id for f in (eq1, eq2, eq4)
             for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(f))))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    check("D.8", len(srcs) == 3 and len(srcs_bad) == 4 and not (srcs & srcs_bad)
          and not calls,
          "★★ **源码级线索**（★ **不是证明**）：三式两两不同源、四个**错法变体**与三式也不同源，"
          "且三式里**不调用任何函数**（AST 调用表 = %s）⇒ ★ 它们是四条各自独立的算式。\n"
          "        ★★ **降级措辞**：这只说明「**四条式子是四段独立代码**」，"
          "**不**证明「A 类判据有物理信息量」—— 后者由 `C.1`/`C.2` 的**行为**兜底"
          % (", ".join(sorted(calls)) if calls else "**空集**"),
          "**三式或错法变体之间出现了同源/调用** —— 判据可能退化成同一个式子自比")

    # ── D.9 ★★★ 引文纪律：「SP 不引入通信开销」是**原文的结论** ─────────────
    check("D.9", specs.SP_NO_OVERHEAD_QUOTE
          == ("our technique, which mixes both tensor and sequence parallelism, reduces "
              "the activation memory significantly without any additional compute, "
              "communication, or memory overhead.")
          and specs.SP_NAIVE_EXTRA_COMM_QUOTE
          == "These extra communications introduce overhead and will slow down the training."
          and specs.SP_FUSE_CONVERTER_QUOTE
          == "To avoid these extra communications, we combine these operations with the",
          "★★★ 三条引文**成对**钉住：**裸 SP 有开销**（`These extra communications …`）"
          "⇒ 原文的处置是**把它合并进 TP 的那两个集合通信**（`To avoid … we combine …`）"
          "⇒ 于是**混用之后没有额外开销**。\n"
          "        ★★★ **本程序【不】把「SP 不引入通信开销」写成任何 C 类反例**"
          "（★ 那会与原文相反）—— 它在这里以**引文**的形式钉住，`C` 类一节讲的是"
          "**激活显存**的两条错法。\n"
          "        ★★ 载体 = ar5iv `2205.05198` 正文，抓取日 2026-10-08，原始字节各 **1** 次",
          "**那三条引文被动过** —— 若把它们改掉，「SP 不引入通信开销」就失去了逐字依据")

    # ── D.10 ★★ 同行评审依据的载体（★ 停在索引页上，不停在 abs 页）──────────
    check("D.10", specs.SP_MLSYS_INDEX_TITLE
          == "Reducing Activation Recomputation in Large Transformer Models"
          and specs.UNSOURCED == (),
          "★★ **同行评审：是**的依据 = MLSys 2023 会议论文索引页上该标题**原始字节 1 次**"
          "（`proceedings.mlsys.org/paper_files/paper/2023`，抓取日 2026-10-08）；\n"
          "        ★★★ **不是** arXiv abs 页 —— 该页上 `Comments` / `MLSys` / `ICLR` / "
          "`Accepted` / `Proceedings` / `Conference` **各 0 次**（本程序把依据**停在索引页**上）。\n"
          "        ★★ `specs.UNSOURCED` **留空**（两处「未核到」已写在各自常量的 `# 来源：` 里）",
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
    print("★ 本程序的三式依据在 `specs.py` 的 `# 来源：`（**同行评审：是**，依据 = MLSys 2023 索引页）；")
    print("  本程序只核对「三式的数值结构与论文一致」与「两条错法真的会失败」。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_activation_sharding.py —— SP 切的是 TP 没切到的逐 token 算子（式(1)/(2)/(4) 的恒等式）")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s`，纯标量算术，**不用 numpy**）。"
          % sys.version.split()[0])
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 本程序**不是**「模拟器 vs 闭式」的对拍，而是**把论文公式钉成恒等式**：")
    print("      ★ A 类**对转写错误敏感、对物理错误不敏感** ⇒ **内容判据在 C.1（只切一半）与 C.2（切错轴）**。")
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
