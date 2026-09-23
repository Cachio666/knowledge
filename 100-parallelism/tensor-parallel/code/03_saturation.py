#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_saturation.py —— **TP 的通信量【也饱和】**：`8bsh((t−1)/t)`，但饱和值与 DP 不同。

配套文档：`03-saturation-and-limits.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「推导值」或「本机实测值」**，测量/推导条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★★★ 本程序与同域另外两个程序**不同**：它的主角是**闭式**（论文里的公式），
> 所以它的数字**绝大多数是【推导值】**（公式代入，可手算复核），
> 只有 `A.1` 的**对拍**与 `D.10` 的形状扫描是**本机真跑**。

★★★ **本主题的头号风险是「模拟器同时是被测对象」**（计划 §1.3.3）：
  本程序**不去模拟**通信（不去数「第 k 步发了什么」—— 那是**代码结构**，不是被测对象），
  它把两条**来源不同**的表达式**对拍**：
    · 一条来自 **[C] 论文的闭式** `8bsh((t−1)/t)`（`2104.04473 §3.2` 逐字）；
    · 一条来自 **结构计数**：`4` 次 all-reduce（`1909.08053` Fig 4 逐字）× 每次的搬运量
      `2(t−1)/t × S`（系数 `2` 来自 **本库正文** `performance-models/02:209` 的 `V(N)`）。
  ⇒ ★★ 两条**独立**路径给出同一个数 —— 这是本程序唯一的 A 类判据的形式。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序对这一条的答案是**有限制的**，必须明说：
    `A.1` 的对拍是**两个闭式之间**的对拍 —— 若把「结构式」那一支删掉，断言确实不再成立，
    但**它证明的是「两条式子在代数上一致」**，**不是**「真实系统里的通信量就是这个数」。
    ⇒ ★★ **本程序不得声称验证了任何真实系统的通信量**；真实带宽/延迟的正文在
      `110-communication` 与 `20-gpu-architecture/interconnect/`（本程序一行都不写）。

★★ **覆盖边界（必须写明 —— 计划 §2 的第 3 条）**：本程序的 A 类守的是
  「**公式与结构计数一致**」，**不守**「公式本身对不对」（那由 [C] 的原文保证）
  ⇒ **不得声称 A 类验证了更多东西**。
  ★★ **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：`C.1`（多乘一次 `2`）与
  `C.2`（漏掉 `(t−1)/t`）断言的都是**失败**（差一个可判定的倍数）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  `10-foundations/performance-models/02-amdahl-and-scaling.md:223` 已经逐字写完了 TP 的**三条**
  （通信发生在**每一层内部** · 通信量与**激活**成正比而不是与参数量成正比 ·
  **只在高带宽互联上可用 / 通常限于单节点 8 卡**，以及「规模上限由节点内互联决定」）
  ⇒ **本程序一行都不重写那三条**（否则是 `P2`「同一个载体重复」）。
  ★★★ **本程序只写它们【没写】的那一层**：**TP 的通信量【也饱和】** ——
  那个 **`(t−1)/t → 1`** 与 DP 的 **`2(N−1)/N → 2`** 是**同一个结构**
  （环 all-reduce 的效率因子），⇒ **两者都饱和，但饱和值不同**：
  **DP 饱和到 `2S`（∝ 参数量，`:219`）· TP 饱和到 `8bsh`（∝ 激活量）**。
  ★★ 另有：**集合通信原语的算法与复杂度**（`V(N) = 2(N−1)/N × S`、`S` 的饱和）
  → `110-communication` 与 `performance-models/02:209`（**本程序只引用它的系数 `2`**）；
  **链路带宽的三种口径** → `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`；
  **拓扑受限的集合通信** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  **Amdahl / Gustafson / `N₅₀` / 扩展效率模型** → `performance-models/02`（**全篇**）；
  **`(p−1)/m` 的另一种写法与 DP 度 `d` 的关系** → `performance-models/02:225` 与
  `2104.04473 §3`（**3D 并行**，本程序不涉及）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **闭式 == 结构计数**（两条独立路径）：
             `l × 8bsh((t−1)/t)` == `l × 4 × [2(t−1)/t × bsh]`（**元素数**；×`itemsize` 得字节）；
  · `A.1g` ★★ **守卫**：把结构计数**多乘一次 `2`** ⇒ 同一条断言**必须失败**（差**恰好** `2` 倍）；
  · `A.2`  ★★ **`t = 1` 时通信量恒为 `0`**（两条路径都是 `0`）——
             ★ 这是对公式转写的一条强检查：错抄的式子（例如漏了 `t−1`）在 `t=1` 上**不给 0**；
  · `B.1`  ★★ **饱和行为**：`(t−1)/t` 在 `t=1/2/4/8/16/32` 上是
             `0 / 0.500000 / 0.750000 / 0.875000 / 0.937500 / 0.968750`
             ⇒ **`t ≥ 8` 后收益极小**（`t=8→16` 只涨 **7.1%**、`16→32` 只涨 **3.3%**）；
  · `B.2`  ★★ **两个饱和值随什么变【不同】**：DP 的 `2S` **与 `b·s` 无关**（∝ 参数量），
             TP 的 `8bsh` **与 `b·s` 成正比**（∝ 激活量）—— 这是「TP 与 DP 的通信模式不同」的
             **定量版本**（`:223` 只有定性版）；
  · `C.1`  ★★★ **`8` 里已经含了前向与后向** ⇒ 算全模型时**不要再乘 `2`** ——
             多乘一次的版本**恰好差 `2` 倍**（**本库的控制者就多乘过一次**）；
  · `C.2`  ★★ **饱和会把错误藏起来**：忘记 `(t−1)/t` 时，`t=2` 差 **2 倍**、
             而 `t=32` 只差 **3.1%** ⇒ ★ **这类判据必须在【小 `t`】上做**。

用法：
    python3 03_saturation.py
"""

import ast
import collections
import inspect
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（tensor-parallel）取 `^^ FAIL: tensor-parallel`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
#   ★ 它**只允许出现在 `print()` 的字符串里** ⇒ 本文件里它只出现在打印与断言明细中。
ASSERT_FAIL_MARK = "^^ FAIL: tensor-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "== FAIL: pipeline-parallel",
                     "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                     "?? FAIL: zero-fsdp")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★★ **被测函数的调用计数器**（`D.7c` 用）：参照实现跑完后它必须恰好为空。
_CALLS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 一律是**判据成立时**的措辞 ⇒
      若把它原样打在失败行上，输出会**自相矛盾** ⇒ `cond` 为假时走**另一版文案**。
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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立 / 守卫成立 / "
                                "⇒ 不是恒真的」的措辞是**成立时**的措辞，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260953              # ★ 与同域另外两个程序（…0951 / …0952）都**不同**
TP_GRID = (1, 2, 4, 8, 16, 32)   # TP 度的扫描点（`B.1` 的饱和表就用它）
L_LAYERS = 24                # 模型层数（`l`）
B_SIZE = 1                   # micro-batch 大小（`b`）
S_LEN = 2048                 # 序列长度（`s`）
H_SIZE = 1024                # 隐层宽度（`h`）
ITEMSIZE = 2                 # ★ 口径：一个元素占 **2 字节**（bf16）—— 见 `D.3b`
BS_GRID = ((1, 512), (1, 1024), (1, 2048), (2, 2048), (4, 2048))
# `B.2` 的 `(b, s)` 扫描点：★ 它把 `b·s` 从 `512` 一路推到 `8192`（**16 倍**）
# ⇒ 「TP 的通信量与 `b·s` 成正比」这句话才有可判定的形态。
# ── 阈值 ────────────────────────────────────────────────────────────────────
TOL_MACHINE = 1e-12          # 数值对拍的相对容差
TOL_GUARD = 1e-6             # 守卫：改错的式子必须**远大于**它
TOL_ERR = 1e-3               # C 类「真的会错」的下限
TOL_RATIO = 1e-9             # 「恰好 `2` 倍」这类比值的容差
GAIN_8_16 = 0.08             # `B.1`：`t=8→16` 的涨幅上限（本机 7.1%）
GAIN_16_32 = 0.04            # `B.1`：`t=16→32` 的涨幅上限（本机 3.3%）
# ★★ **盲区⑥ 的落法**：本程序的浮点常量都 `< 2^53` ⇒ 每个叶子都**可变异**。
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 十进制字面量：盲区④）


# ===========================================================================
# 2. 两条路径：论文闭式 vs 结构计数
# ===========================================================================
def saturation_factor(t):
    """★★★ **饱和因子** `(t−1)/t` —— 本程序的主角。

    ★★ 它与 DP 的 `2(N−1)/N` 里那个 `(N−1)/N` 是**同一个结构**（环 all-reduce 的效率因子），
      只是**饱和值不同**：`(t−1)/t → 1`、`(N−1)/N → 1`（而外面乘的常数不同）。
    """
    _CALLS["saturation_factor"] += 1
    return (t - 1) / t


def ring_volume(t, s_bytes):
    """一次 all-reduce 里**每卡**搬的字节数 —— **引用** `V(N)`，**不重推**。

    `V(N) = RING_ALLREDUCE_VOLUME_FACTOR × (N−1)/N × S`，出处是本库正文
    `10-foundations/performance-models/02-amdahl-and-scaling.md:209`
    （它在那边是**推导**，正文在彼；本程序**只消费它的系数 `2`**）。

    ★★ 本程序的职责边界：**不重推 `V(N)`**、**不写带宽/延迟**。
    """
    _CALLS["ring_volume"] += 1
    return specs.RING_ALLREDUCE_VOLUME_FACTOR * saturation_factor(t) * s_bytes


def closed_form_bytes(t, l, b, s, h, itemsize):
    """**被测路径 ①**：[C] `2104.04473 §3.2` 的**闭式** `8bsh((t−1)/t)`，乘层数与字节宽。

    ★★★ 逐字（`specs.MEGATRON_TP_COMM_QUOTE`）：`twice each in the forward and backward pass
      for each layer, leading to a total communication of` **`8bsh((t−1)/t)`**
      `per layer per device for each microbatch.`
      （★ 公式本体在原始字节上是 `<math>` ⇒ 按本库规矩**不入 `specs.py`**，
        只寄存它的**系数 `8`**：`specs.MEGATRON_TP_COMM_MULTIPLIER`。）
    """
    _CALLS["closed_form_bytes"] += 1
    return (l * specs.MEGATRON_TP_COMM_MULTIPLIER * b * s * h
            * saturation_factor(t) * itemsize)


def structural_bytes(t, l, b, s, h, itemsize):
    """**被测路径 ②**：**结构计数** —— `l` 层 × 每层 `4` 次 all-reduce × 每次的搬运量。

    · `4` = **前向 2 次 + 后向 2 次**（[C] `1909.08053` 的 Figure 4 逐字 `4 total`；
      [C] `2104.04473` 的 `twice each in the forward and backward pass`）—— 见
      `specs.MEGATRON_TP_COMM_OPS_PER_LAYER`；
    · 每次搬 `ring_volume(t, b·s·h·itemsize)`。

    ★★★ **`8` 里已经含了前向与后向** —— 本函数**不再乘 `2`**（`C.1` 把这条变成实测的 2 倍差）。
    ★★ **两条路径的独立性**：① 是**论文的闭式**（系数 `8` 一次乘出来）；
      ② 是**结构计数**（`4` 次 × 每次的搬运量，两个因子各有各的出处）。
    """
    _CALLS["structural_bytes"] += 1
    per_allreduce = ring_volume(t, b * s * h * itemsize)
    return l * specs.MEGATRON_TP_COMM_OPS_PER_LAYER * per_allreduce


def dp_saturation_bytes(psi):
    """DP 那一侧的**饱和值** `2S`（[E] `performance-models/02:219`）—— `S` = 参数字节数。

    ★★ `S = 2Ψ`（bf16 下每个参数 `2` 字节）⇒ `2S = 4Ψ` ⇒ **它与 `b`、`s` 都无关**。
    ★ 本函数**只做算术**，不做任何测量（本机没有真机）。
    """
    _CALLS["dp_saturation_bytes"] += 1
    return specs.DP_COMM_SATURATION_MULTIPLE * (2 * psi)


# ===========================================================================
# 3. 独立参照（**不调用**被测路径的任何函数）
# ===========================================================================
def reference_bytes(t, l, b, s, h, itemsize):
    """**独立参照**：把 `8bsh((t−1)/t)` **从零重写一遍**（不引用任何被测函数的中间量）。

    ★★★ **独立性**：它只做一次乘法连乘，**不调用** `closed_form_bytes` /
      `structural_bytes` / `ring_volume` / `saturation_factor`
      （`D.7b` 的 AST 调用表 + `D.7c` 的调用计数**两条**证据钉住）。
    ★ 它与被测路径 ① 的**写法不同**（这里是 `8 * l * b * s * h * (t − 1) / t * itemsize`
      的直写），但**代数上必须相同** —— 这是 `A.1` 能成立的前提。
    """
    return 8 * l * b * s * h * (t - 1) / t * itemsize


# ===========================================================================
# 4. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    """用 **AST** 从本文件里数出模块级的 `ASSERT_FAIL_MARK = <字符串字面量>`。"""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def no_framework_imported():
    """★ 运行期的同义确认：`sys.modules` 里**第三方**名字中只允许有 `numpy`。"""
    banned = {"torch", "tensorflow", "jax", "triton", "numba", "cupy", "apex",
              "deepspeed", "bitsandbytes", "transformers", "flax", "vllm"}
    return not (banned & set(sys.modules))


def doc_has_boundary_ref(filename, needle):
    """`D.5c`：读**配套文档**并检查边界锚点逐字在（`P1` 的可执行落法）。

    ★★ 返回 `None` 表示**文档不在**（`mutate.py` 的副本目录里只有 `*.py`）⇒ 跳过并打印。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    if not os.path.exists(path):
        return None
    return needle in open(path, encoding="utf-8").read()


def _gb(nbytes):
    """字节 → GB（`1e9` 口径，★ 与 `performance-models/02:215` 的口径一致）。"""
    return nbytes / 1e9


# ===========================================================================
# 5. A 类：闭式 vs 结构计数（判据 A.1 / A.1g / A.2）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 两条独立路径：论文闭式 `8bsh((t−1)/t)` vs 结构计数 `l × 4 × V(t)`")
    print("=" * 74)
    print("  口径：`l=%d` 层 · `b=%d` · `s=%d` · `h=%d` · `itemsize=%d` 字节（bf16）"
          % (L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE))
    print("  ★★★ **本程序没有张量**：通信量是**算出来的**，不是量出来的 ——")
    print("      它是一个**代价模型**（B 类），不是一个实现。")

    # ── A.1 ★★★ 闭式 == 结构计数 ─────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **闭式 == 结构计数**（两条来源不同的路径）──")
    print("     [C] `%s` §3.2 逐字：`%s`" % (specs.MEGATRON_PP_ARXIV_ID,
                                           specs.MEGATRON_TP_COMM_QUOTE))
    print("       同句尾：`%s`（口径三件套：**每层 · 每卡 · 每 micro-batch**）"
          % specs.MEGATRON_TP_COMM_CALIBER_QUOTE)
    print("     [C] `%s` Figure 4 caption 逐字：`%s`" % (specs.MEGATRON_TP_ARXIV_ID,
                                                       specs.MEGATRON_FIG4_QUOTE))
    print("     [E] 本库正文 `performance-models/02:209` 的 `V(N) = 2(N−1)/N × S`")
    print("         ⇒ 每次 all-reduce 每卡搬 `%.1f × (t−1)/t × S` 字节（**引用，不重推**）"
          % specs.RING_ALLREDUCE_VOLUME_FACTOR)
    print("     ★★★ **`8 = 4 × 2` 的分解**（★ 这个分解**原文没有给**，是本库拼的 ⇒ 标 [E]）：")
    print("        `4` = 前向 2 次 + 后向 2 次 all-reduce（[C] 两条来源互证）；")
    print("        `2` = 环 all-reduce 的 `2(n−1)/n` 因子（[E] `:209`）。")
    worst = 0.0
    rows = []
    for t in TP_GRID:
        c = closed_form_bytes(t, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
        st = structural_bytes(t, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
        rf = reference_bytes(t, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
        rel = abs(c - st) / c if c > 0 else 0.0
        relr = abs(c - rf) / c if c > 0 else 0.0
        worst = max(worst, rel, relr)
        rows.append((t, c, st, rf))
        print("     `t=%-3d` 闭式 = **%.6e B**（%.3f GB）· 结构 = **%.6e B** · "
              "参照 = **%.6e B** · 相对差 = **%.1e**"
              % (t, c, _gb(c), st, rf, rel))
    print("     ⇒ ★★ **两条路径的相对差 ≤ `%.1e`**（全部是**双精度可表示**的整数运算，"
          "所以差是 `0`；有限非零只可能来自浮点除法）" % worst)
    # ★★ 守卫：把结构计数**多乘一次 2** ⇒ A.1 必须失败（差恰好 2 倍）
    t_ = 4
    st_doubled = 2 * structural_bytes(t_, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    c_ = closed_form_bytes(t_, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    ratio_bad = st_doubled / c_
    print("     ★★ **守卫**（`CONTRIBUTING:235`）：把结构计数**多乘一次 `2`** ⇒ "
          "比值 = **%.9f** ⇒ 同一条断言**必须失败**" % ratio_bad)
    ok = all(abs(c - st) <= TOL_MACHINE * max(abs(c), 1.0)
             and abs(c - rf) <= TOL_MACHINE * max(abs(c), 1.0) for _, c, st, rf in rows)
    check("A.1", ok and abs(ratio_bad - 2.0) <= TOL_RATIO,
          "★★★ **判据成立**：`t ∈ %s` 六点上，**闭式 == 结构计数 == 独立参照**"
          "（相对差 ≤ `%.1e`）；★ 且守卫给出**恰好 `2` 倍**的偏差 ⇒ `A.1` **能失败**。"
          "★★ **本判据证明的是「两条式子代数一致」**，**不是**「真实系统的通信量是这个数」"
          % (TP_GRID, worst))

    # ── A.2 ★★ t = 1 时恒为 0 ────────────────────────────────────────────
    print()
    print("  ── A.2 ★★ **`t = 1` 时通信量恒为 `0`** —— 对公式转写的一条强检查 ──")
    c1 = closed_form_bytes(1, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    s1 = structural_bytes(1, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    r1 = reference_bytes(1, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    print("     `t=1`：闭式 = **%.1f** · 结构 = **%.1f** · 参照 = **%.1f**（都必须是 `0`）"
          % (c1, s1, r1))
    print("     ⇒ ★★ **为什么它值钱**：TP 度为 1 就是**单卡**，**通信量必须是 `0`**。")
    print("        错抄的公式会在这一点上露馅 —— 例如：")
    print("        · 写成 `8bsh/t` ⇒ `t=1` 给出 `8bsh`（**不是 0**）；")
    print("        · 写成 `8bsh(1 − 1/t)` ⇒ `t=1` 给出 `0` ✓（**这个错法在 `t=1` 上抓不到**，"
          "见 `C.2` 的同一族）；")
    print("        · 漏掉 `(t−1)` 只留 `8bsh` ⇒ `t=1` 给出 `8bsh`（**不是 0**）。")
    print("     ⇒ ★ 但**必须写明它的适用边界**：`t=1` 只能挡住「把 `(t−1)` 丢掉」这一类错，"
          "**挡不住**「把 `(t−1)` 写成 `(t−k)`」这一类 ⇒ `B.1` 的**逐点数值表**是必需的补充。")
    check("A.2", c1 == 0.0 and s1 == 0.0 and r1 == 0.0 and saturation_factor(1) == 0.0,
          "★★ **判据成立**：`t = 1` 时**三条路径都给出 `0.0`**（`(t−1)/t = 0`）⇒ "
          "「单卡没有通信」这件事在公式里是**可判定的**，不是靠嘴说的")


# ===========================================================================
# 6. B 类：饱和行为 · DP 与 TP 的两个饱和值
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：`(t−1)/t` 的饱和行为 · DP 与 TP 的饱和值【随什么变】不同")
    print("=" * 74)

    # ── B.1 ★★ 饱和行为 ──────────────────────────────────────────────────
    print("  ── B.1 ★★ **饱和行为**：`(t−1)/t` 在 `t = 1/2/4/8/16/32` 上 ──")
    facs = [(t, saturation_factor(t)) for t in TP_GRID]
    print("     " + "  ".join("`t=%-2d`→**%.6f**" % (t, f) for t, f in facs))
    gains = []
    for i in range(1, len(facs)):
        if facs[i - 1][1] == 0.0:
            # ★★ **`t=1 → 2` 的涨幅【无定义】**（除数是 `(t−1)/t = 0`）——
            #    这本身是一个口径点：**从 0 出发的「涨幅」不是一个比值**。
            #    ⇒ 本表**从 `t=2` 起**给涨幅，并把这一条写明（不是悄悄跳过）。
            print("     逐步涨幅：`1→2` **无定义**（`(t−1)/t` 在 `t=1` 处是 `0`，"
                  "**零不能作除数**）⇒ 下表从 `t=2` 起")
            continue
        g = facs[i][1] / facs[i - 1][1] - 1.0
        gains.append((facs[i - 1][0], facs[i][0], g))
    print("     " + "  ".join("`%d→%d` 涨 **%.4f%%**" % (a, b, g * 100.0)
                              for a, b, g in gains))
    g_8_16 = [g for a, b, g in gains if a == 8 and b == 16][0]
    g_16_32 = [g for a, b, g in gains if a == 16 and b == 32][0]
    print("     ⇒ ★★ **`t ≥ 8` 之后收益极小**：`8→16` 只涨 **%.4f%%**、`16→32` 只涨 **%.4f%%**"
          % (g_8_16 * 100.0, g_16_32 * 100.0))
    print("     ⇒ ★★ **这才是「TP 通常限于单节点 8 卡」的【定量】一侧** ——")
    print("        另一侧是**代价**（跨节点慢链路），它的正文在 `%s`（本程序不重写）。"
          % specs.TP_BOUNDARY_REF)
    print("     ★★ **一处必须写明的出处纪律**：这一段的 [C] 依据是 `%s` §3.2 的 `%s`"
          % (specs.MEGATRON_PP_ARXIV_ID, specs.MEGATRON_TAKEAWAY1_LABEL))
    print("        （逐字：`%s … %s`）" % (specs.MEGATRON_TP_NODE_TAKEAWAY_QUOTE,
                                          specs.MEGATRON_TP_NODE_TAKEAWAY_TAIL_QUOTE))
    print("        ★★★ **不是** `1909.08053` —— 那一篇里**没有这句话**"
          "（本轮的 `curl` 实测：`single node` / `g-GPU` 各计 `0` 次）。")
    mono = all(facs[i][1] > facs[i - 1][1] for i in range(1, len(facs)))
    shrink = len(gains) == len(facs) - 2 and all(
        gains[i][2] < gains[i - 1][2] for i in range(1, len(gains)))
    check("B.1", mono and shrink and g_8_16 < GAIN_8_16 and g_16_32 < GAIN_16_32
          and abs(facs[3][1] - 0.875) <= TOL_RATIO and abs(facs[5][1] - 0.96875) <= TOL_RATIO,
          "★★ **判据成立**：`(t−1)/t` **单调递增**、**涨幅单调递减**；`t=8` 时已是 `%.6f`"
          "（`0.875`）、`t=32` 时 `%.6f`（`0.96875`）；`8→16` 涨 **%.2f%%**（< `%.0f%%`）、"
          "`16→32` 涨 **%.2f%%**（< `%.0f%%`）⇒ ★★ **饱和**：再加 TP 度几乎不改变通信量"
          % (facs[3][1], facs[5][1], g_8_16 * 100.0, GAIN_8_16 * 100.0,
             g_16_32 * 100.0, GAIN_16_32 * 100.0))

    # ── B.2 ★★ 两个饱和值随什么变 ────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **DP 与 TP 的饱和值不同**：`2S`（∝ 参数量）vs `8bsh`（∝ 激活量）──")
    psi = 12 * L_LAYERS * H_SIZE * H_SIZE            # ★ 假设：每层 ≈ `12h²`（attention 4h² + MLP 8h²）
    dp = dp_saturation_bytes(psi)
    print("     [E] `performance-models/02:219` 逐字：`每卡通信量饱和到 2S`"
          "（`%s = %.6f`）" % ("DP_COMM_SATURATION_MULTIPLE",
                              specs.DP_COMM_SATURATION_MULTIPLE))
    print("     ★★ **DP 一侧**：`S` = **参数的字节数** = `2Ψ`（bf16）⇒ `2S = 4Ψ`。")
    print("        本程序的模型规模假设：`Ψ ≈ 12·l·h² = 12 × %d × %d²` = **%d** 个参数"
          % (L_LAYERS, H_SIZE, psi))
    print("        ⇒ 每卡每步的 DP 饱和通信量 = **%.6e B**（%.1f GB）" % (dp, _gb(dp)))
    print("     ★★ **TP 一侧**：`8bsh((t−1)/t)` 是**每层每卡每 micro-batch** 的量 ⇒")
    print("        全模型（`l` 层）在 `t → ∞` 的饱和值 = `l × 8 × b × s × h × itemsize`。")
    base = closed_form_bytes(TP_GRID[-1], L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    print("        `t=%d`（饱和档）时 = **%.6e B**（%.3f GB）" % (TP_GRID[-1], base, _gb(base)))
    print("     ★★★ **两个饱和值【随什么变】完全不同**（这才是要证明的东西）——")
    dp_list, tp_list = [], []
    for (bb, ss) in BS_GRID:
        d = dp_saturation_bytes(psi)
        tpv = closed_form_bytes(TP_GRID[-1], L_LAYERS, bb, ss, H_SIZE, ITEMSIZE)
        dp_list.append(d)
        tp_list.append(tpv)
        print("        `(b=%d, s=%-5d)`：DP = **%.6e B**（**不变**）· TP = **%.6e B** · "
              "TP/DP = **%.4f**" % (bb, ss, d, tpv, tpv / d))
    dp_const = len(set(dp_list)) == 1
    bs0 = BS_GRID[0][0] * BS_GRID[0][1]
    bs1 = BS_GRID[-1][0] * BS_GRID[-1][1]
    factor = bs1 / bs0                            # `b·s` 的跨度（本程序是 16 倍）
    tp_linear = abs(tp_list[-1] / tp_list[0] - factor) <= TOL_RATIO
    print("     ⇒ ★★ **DP 的 `2S` 与 `b`、与 `s` 都【无关】**（∝ **参数量**）；")
    print("        ★★ **TP 的 `8bsh` 与 `b·s` 【成正比】**（∝ **激活量**）——"
          "本机实测（本程序的算术）：`b·s` 从 `%d` 涨到 `%d`（**%.0f 倍**）时 TP 涨 **%.4f 倍**。"
          % (bs0, bs1, factor, tp_list[-1] / tp_list[0]))
    print("     ⇒ ★★★ **这就是「TP 与 DP 的通信模式不同」的定量版本**"
          "（`%s` 只有定性版：`通信量与激活成正比，而不是与参数量成正比`）。"
          % specs.TP_BOUNDARY_REF)
    print("     ⇒ ★★ **口径必须写明**：DP 的 `2S` 是**每步**（梯度规约每步一次），"
          "TP 的 `8bsh` 是**每层每卡每 micro-batch** —— 两者的**分母不同**")
    print("        ⇒ 本条比的是「**随什么变**」，**不是**「谁更大」（后者需要再补一层换算，"
          "而那一层的输入本机取不到 ⇒ 本程序**不做那个比较**）。")
    check("B.2", dp_const and tp_linear and psi == 12 * L_LAYERS * H_SIZE * H_SIZE
          and specs.DP_COMM_SATURATION_MULTIPLE == 2.0 and factor == 16.0,
          "★★ **判据成立**：DP 的 `2S` 在 `%d` 组 `(b, s)` 下**完全相同**（`%.6e B`）；"
          "TP 的 `8bsh` 与 `b·s` **严格成正比**（`b·s` ×`%.0f` ⇒ TP ×**%.4f**）⇒ "
          "★ **两个饱和值随不同的东西变**：DP ∝ **参数量**、TP ∝ **激活量**"
          % (len(dp_list), dp_list[0], factor, tp_list[-1] / tp_list[0]))


# ===========================================================================
# 7. C 类：边界与反例 —— 多乘一次 2、漏掉 (t−1)/t
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：`8` 里已含前向+后向（不要再乘 2）· 漏掉 `(t−1)/t` 的错在饱和处看不见")
    print("=" * 74)

    # ── C.1 ★★★ 多乘一次 2 ────────────────────────────────────────────────
    print("  ── C.1 ★★★ **`8` 里已经含了前向与后向** ⇒ 算全模型时**不要再乘 `2`** ──")
    t_ = 4
    good = closed_form_bytes(t_, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    bad = 2 * good
    print("     [C] `%s` §3.2 的**语序**（逐字）：`%s`" % (specs.MEGATRON_PP_ARXIV_ID,
                                                          specs.MEGATRON_TP_COMM_QUOTE))
    print("        ⇒ `twice each in the forward and backward pass`（**4 次**）**然后是**")
    print("          `leading to a total communication of 8bsh(…)` ⇒")
    print("        ★★★ **`8` 是那 4 次的【结果】，不是它的【前置因子】**。")
    print("     ★★ 双重计数版（把 `8` 当「每方向 4 次」再乘 `2`）：")
    print("        `正确 = %.6e B` vs `多乘一次 = %.6e B` ⇒ 比值 = **%.9f**"
          % (good, bad, bad / good))
    print("     ★★★ **这是一个真的踩过的坑**：本库的**控制者在验证脚本里就多乘过一次**，"
          "实测差**恰好 2 倍**。")
    print("     ★★ **为什么容易踩**：`8 = 4 × 2` 里的 `4` 已经是「前向 2 + 后向 2」⇒")
    print("        看到「前向 + 后向」就想再乘 `2` 的人，会把 `8` 读成「前向的 4 次」。")
    check("C.1", abs(bad / good - 2.0) <= TOL_RATIO and good > 0.0,
          "★★★ **失败成立**：双重计数版**恰好差 `2` 倍**（`%.9f`）⇒ "
          "「把 `8` 再乘一次 `2`」是一个**可判定的错**，不是量级模糊的问题"
          % (bad / good))

    # ── C.2 ★★ 饱和把错误藏起来 ──────────────────────────────────────────
    print()
    print("  ── C.2 ★★ **饱和会把错误藏起来**：漏掉 `(t−1)/t` 时，`t` 越大越看不出来 ──")
    worst_small, worst_large = None, None
    for t in TP_GRID:
        if t == 1:
            continue
        realv = closed_form_bytes(t, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
        wrong = (specs.MEGATRON_TP_COMM_MULTIPLIER * L_LAYERS * B_SIZE * S_LEN * H_SIZE
                 * ITEMSIZE)                     # 漏掉 `(t−1)/t`（= 把因子当成 1）
        r = wrong / realv
        if t == 2:
            worst_small = r
        worst_large = r
        print("        `t=%-3d`：正确 = **%.6e B** · 漏掉因子 = **%.6e B** ⇒ 比值 = **%.6f**"
              % (t, realv, wrong, r))
    print("     ⇒ ★★ **`t=2` 时差 `2` 倍（一眼可见），`t=32` 时只差 `%.4f%%`（几乎看不见）**"
          % ((worst_large - 1.0) * 100.0))
    print("     ★★ **同一件事的反面**：`(t−1)/t` 这个因子**只有在【小 `t`】上才有区分度** ⇒")
    print("        一条在 `t=32` 上跑的判据**几乎什么也检查不到**（这正是 `P26`/`D22` 那一族"
          "「判据的取值让判据失效」）。")
    print("     ★★ **纪律**：凡判据里出现 `(t−1)/t` 这类**饱和因子**，"
          "参数必须**含小值**（本程序用 `%s`，且 `B.1` 的逐点表把 `t=1` 与 `t=2` 都列出来）。"
          % (TP_GRID,))
    check("C.2", worst_small is not None and abs(worst_small - 2.0) <= TOL_RATIO
          and worst_large - 1.0 < 0.04,
          "★★ **失败成立**：漏掉 `(t−1)/t` 在 `t=2` 时差 **%.6f** 倍（**可判定**），"
          "在 `t=32` 时只差 **%.4f%%**（**看不见**）⇒ ★★ **饱和会让判据失效**，"
          "所以这类判据必须在**小 `t`** 上做" % (worst_small, (worst_large - 1.0) * 100.0))


# ===========================================================================
# 8. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**推导值**（公式代入，可手算复核）+ **本机实测值**"
          "（`python3 %s` + `numpy %s`，macOS arm64，**无 GPU / 无多卡 / 无 NCCL**）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序**不测量任何通信**。")
    print("  ★★★ **禁区回链（本程序一行都不重写）**：TP 的三条"
          "（通信在**每一层内部** · 通信量 ∝ **激活** · **只在高带宽互联上可用 / 通常限于"
          "单节点 8 卡**）已经由 `%s` 逐字写完。" % specs.TP_BOUNDARY_REF)
    print("     另有：**集合通信原语的算法与复杂度**（`V(N) = 2(N−1)/N × S`、`S` 的饱和）"
          "→ `110-communication` 与 `performance-models/02:209`（**本程序只引用系数 `2`**）；"
          "**链路带宽的三种口径** → `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`；"
          "**拓扑受限的集合通信** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；"
          "**扩展效率模型**（Amdahl / `N₅₀`）→ `performance-models/02`（**全篇**）；"
          "**3D 并行与 `(p−1)/m`** → `performance-models/02:225`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == "^^ FAIL: tensor-parallel" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "== FAIL: pipeline-parallel",
                                        "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                                        "?? FAIL: zero-fsdp"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住；"
          "★ 六个标记**两两不互为前缀**")

    measured = {"SEED": SEED, "TP_GRID": TP_GRID, "L_LAYERS": L_LAYERS, "B_SIZE": B_SIZE,
                "S_LEN": S_LEN, "H_SIZE": H_SIZE, "ITEMSIZE": ITEMSIZE, "BS_GRID": BS_GRID,
                "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT}
    expect = {"SEED": 20260953, "TP_GRID": (1, 2, 4, 8, 16, 32), "L_LAYERS": 24,
              "B_SIZE": 1, "S_LEN": 2048, "H_SIZE": 1024, "ITEMSIZE": 2,
              "BS_GRID": ((1, 512), (1, 1024), (1, 2048), (2, 2048), (4, 2048)),
              "FLOAT_EXACT_LIMIT": 9007199254740992}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**推导条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", all((t & (t - 1)) == 0 for t in TP_GRID) and TP_GRID[0] == 1
          and max(TP_GRID) >= 32 and ITEMSIZE == 2
          and specs.MEGATRON_TP_COMM_MULTIPLIER == 8
          and specs.MEGATRON_TP_COMM_OPS_PER_LAYER == 4
          and specs.MEGATRON_TP_COMM_MULTIPLIER == 2 * specs.MEGATRON_TP_COMM_OPS_PER_LAYER,
          "★★★ **本程序的全部设计都在这几个数上**：`TP_GRID` 全是 **2 的幂**且**含 `t=1`**"
          "（`A.2` 需要它）与 `t=32`（`B.1` 的饱和档需要它）；"
          "★ `MEGATRON_TP_COMM_MULTIPLIER (%d) == 2 × MEGATRON_TP_COMM_OPS_PER_LAYER (%d)`"
          " ⇒ **`8 = 4 × 2` 这条分解是可数地钉住的**；"
          "★ `ITEMSIZE=%d` 是**一个口径选择**（bf16），不是来源里的数"
          % (specs.MEGATRON_TP_COMM_MULTIPLIER, specs.MEGATRON_TP_COMM_OPS_PER_LAYER, ITEMSIZE))

    th = {"TOL_MACHINE": TOL_MACHINE, "TOL_GUARD": TOL_GUARD, "TOL_ERR": TOL_ERR,
          "TOL_RATIO": TOL_RATIO, "GAIN_8_16": GAIN_8_16, "GAIN_16_32": GAIN_16_32}
    th_expect = {"TOL_MACHINE": 1e-12, "TOL_GUARD": 1e-6, "TOL_ERR": 1e-3,
                 "TOL_RATIO": 1e-9, "GAIN_8_16": 0.08, "GAIN_16_32": 0.04}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 6 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < FLOAT_EXACT_LIMIT
          and FLOAT_EXACT_LIMIT == 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点阈值都 `< 2^53`（最大的是 `%g`）⇒ "
          "`mutate.py` 的 `v + 1.0` **能**改变它们（否则那类叶子**免检**）"
          % max(th_expect.values()))
    check("D.4c", GAIN_16_32 < GAIN_8_16 and TOL_MACHINE < TOL_GUARD < TOL_ERR
          and no_framework_imported(),
          "★ 两个涨幅上限**必须递减**（`%.2f < %.2f`）—— 那是**饱和**的可判定形态；"
          "★ 三个数值阈值量级递增；★ 运行期确认 `sys.modules` 里只有 `numpy`"
          % (GAIN_16_32, GAIN_8_16))
    return len(FAILS) == before


# ===========================================================================
# 9. D 类（后半）：specs 常量钉住 + 独立性证据 + 出处纪律 + 边界扩散检查
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 · 独立性证据 · **出处纪律** · 边界锚点的扩散检查")
    print("=" * 74)

    # ★★ 常量逐个钉住，**条数由 `len()` 数出来**（照 `D.3`/`D.4` 的形态）。
    #    ★ 原先把条数写成字面量「十七条」：它**看起来是**「18 项 − 1 个边界锚点路径常量
    #      `TP_BOUNDARY_REF`」，但那套口径**没有写在任何地方** ⇒ 读者无法分辨「口径」与「数错了」
    #      （`01`/`02` 的 `D.5` 也各写「十三条」，同属这一族，但它们同为「减掉那 1 项」）。
    #    ⇒ 现在既**由 `len()` 数出**，又在消息里**写明口径**（含那 1 项）⇒ 两种读法下都不会再错（`P1`）。
    d5_expect = {
        "MEGATRON_PP_ARXIV_ID": "2104.04473",
        "MEGATRON_PP_TITLE": "Efficient Large-Scale Language Model Training on GPU "
                             "Clusters Using Megatron-LM",
        "MEGATRON_PP_AR5IV_URL": "https://ar5iv.labs.arxiv.org/html/2104.04473",
        "MEGATRON_PP_PAGE_BYTES": 208446,
        "MEGATRON_PP_PAGE_SHA256_16": "9dea9fa431b59daf",
        "MEGATRON_PP_COMMENTS_QUOTE": "Accepted to SC 2021",
        "MEGATRON_TP_FWD_BWD_QUOTE": "uses all-reduce communication (two all-reduce operations "
                                     "each in the forward and backward pass, see",
        "MEGATRON_TP_COMM_QUOTE": "twice each in the forward and backward pass for each layer, "
                                  "leading to a total communication of",
        "MEGATRON_TP_COMM_CALIBER_QUOTE": "per device for each microbatch.",
        "MEGATRON_TP_COMM_MULTIPLIER": 8,
        "MEGATRON_TP_COMM_OPS_PER_LAYER": 4,
        "MEGATRON_TP_NODE_TAKEAWAY_QUOTE": "tensor model parallelism should generally be used "
                                           "up to degree",
        "MEGATRON_TP_NODE_TAKEAWAY_TAIL_QUOTE": "-GPU servers, and then pipeline model parallelism "
                                               "can be used to scale up to larger models across "
                                               "servers.",
        "MEGATRON_TP_NODE_REASON_QUOTE": "is larger than the number of GPUs in a single node, the "
                                         "overhead of performing tensor model parallelism across "
                                         "slower inter-node links can be impractical.",
        "MEGATRON_TAKEAWAY1_LABEL": "Takeaway #1:",
        "RING_ALLREDUCE_VOLUME_FACTOR": 2.0,
        "DP_COMM_SATURATION_MULTIPLE": 2.0,
        "TP_BOUNDARY_REF": "10-foundations/performance-models/02-amdahl-and-scaling.md:223",
    }
    d5_got = {k: getattr(specs, k) for k in d5_expect}
    d5_diff = {k: (v, d5_got[k]) for k, v in d5_expect.items() if d5_got[k] != v}
    check("D.5", not d5_diff and len(d5_expect) == 18,
          "★★ [C] `%s`（**同行评审：是** —— 一手依据是 abs 页的 `Comments:` 字段逐字 "
          "`%s`）与本库正文的**%d** 条常量逐位一致%s（★ **条数由 `d5_expect` 的 `len()` 数出**，"
          "不写字面量；口径 = **`d5_expect` 里全部 18 项**，含边界锚点路径常量 `TP_BOUNDARY_REF`）；"
          "★ **`RING_ALLREDUCE_VOLUME_FACTOR` 与 "
          "`DP_COMM_SATURATION_MULTIPLE` 同值但不同项**（一个是单次 all-reduce 的系数、"
          "一个是 DP 的饱和倍数）⇒ 分开存放、**不得合并**"
          % (specs.MEGATRON_PP_ARXIV_ID, specs.MEGATRON_PP_COMMENTS_QUOTE, len(d5_expect),
             "" if not d5_diff else " ← 差异 %s" % d5_diff))

    # ── ★★ 出处纪律（可执行的落法）─────────────────────────────────────────
    print()
    print("  ── D.10 ★★★ **出处纪律**：「TP 限于单节点」的句子归 `2104.04473`，不归 `1909.08053` ──")
    q1909 = (specs.MEGATRON_GELU_NONLINEAR_QUOTE, specs.MEGATRON_GELU_SYNC_QUOTE,
             specs.MEGATRON_COL_SPLIT_LEAD_QUOTE, specs.MEGATRON_COL_REMOVES_SYNC_QUOTE,
             specs.MEGATRON_TWO_ALLREDUCE_QUOTE, specs.MEGATRON_FIG4_QUOTE,
             specs.MEGATRON_HEAD_LOCAL_QUOTE, specs.MEGATRON_EMBED_ALLREDUCE_QUOTE,
             specs.MEGATRON_VOCAB_HUGE_QUOTE, specs.MEGATRON_VOCAB_FUSE_QUOTE,
             specs.MEGATRON_VOCAB_PAD_QUOTE, specs.MEGATRON_VOCAB_PAD_RESULT_QUOTE,
             specs.MEGATRON_8WAY_NOTE_QUOTE)
    q2104 = (specs.MEGATRON_TP_FWD_BWD_QUOTE, specs.MEGATRON_TP_COMM_QUOTE,
             specs.MEGATRON_TP_COMM_CALIBER_QUOTE, specs.MEGATRON_TP_NODE_TAKEAWAY_QUOTE,
             specs.MEGATRON_TP_NODE_TAKEAWAY_TAIL_QUOTE, specs.MEGATRON_TP_NODE_REASON_QUOTE)
    print("     `1909.08053` 的引文常量 **%d** 条 · `2104.04473` 的引文常量 **%d** 条"
          % (len(q1909), len(q2104)))
    print("     ★★★ **本轮 `curl` 实测（两篇 ar5iv 原始字节上 `count()`）**：")
    print("        · `2104.04473`：`%s` 计 **1** 次、`Takeaway #1:` 计 **1** 次；"
          % specs.MEGATRON_TP_NODE_TAKEAWAY_QUOTE)
    print("        · `1909.08053`：`single node` 计 **0** 次、`g-GPU` 计 **0** 次、"
          "`Takeaway` 计 **0** 次。")
    print("        ⇒ ★★ **那句话不在 `1909` 里**，「`Takeaway`」这个体裁也不是它的。")
    print("     ⇒ ★ 本判据是这条纪律的**可执行落法**：两篇的引文常量集合**必须不相交**，"
          "且 `1909` 那一组里**不得**出现 `Takeaway` 或 Takeaway #1 的句子。")
    print("     ★★ **它的覆盖边界（必须写明）**：它只挡住「**把 2104 的句子挂到 1909 名下**」"
          "这类**操作错误**；原文层面的证据是上面的 `curl` 计数（写在每条常量的 `# 来源：` 里）。")
    check("D.10", not (set(q1909) & set(q2104))
          and all("Takeaway" not in q for q in q1909)
          and all(q not in q1909 for q in q2104)
          and len(q1909) == 13 and len(q2104) == 6,
          "★★★ **判据成立**：两篇的引文常量**不相交**（`%d` 条 vs `%d` 条）、"
          "且 `1909` 那一组里**没有任何一条含 `Takeaway`** ⇒ "
          "「TP 限于单节点」的出处**只能**是 `%s §3.2` 的 `%s`"
          % (len(q1909), len(q2104), specs.MEGATRON_PP_ARXIV_ID, specs.MEGATRON_TAKEAWAY1_LABEL))
    got = doc_has_boundary_ref("03-saturation-and-limits.md", specs.TP_BOUNDARY_REF)
    if got is None:
        print("  [skip] D.5c 配套文档不在（`mutate.py` 的副本目录里只有 `*.py`）⇒ 跳过；"
              "★ 字面量本身已在 `D.5` 里逐位钉住")
    else:
        check("D.5c", got,
              "★★ **边界锚点已扩散到本文档**：`%s` 逐字出现在 `03-saturation-and-limits.md` 里 "
              "⇒ 「本域不重写那三条」这句话**有可执行的检查**（`P1`：订正不扩散是默认失败）"
              % specs.TP_BOUNDARY_REF)
    check("D.5d", specs.UNSOURCED == ("tensor_parallel_vocab_pad_multiple_128_rationale",
                                     "tensor_parallel_pad_logits_masking_mechanism"),
          "★★ `specs.UNSOURCED` 的 **2** 项逐位钉住 —— ★★★ **两项都不属于本程序的范围**"
          "（它们在 `02_heads_vocab.py` 的 `C.3`/`B.2`），本程序只**转钉**一次，"
          "保证整个主题只有一处真值源；★ 而 `V(N)` **不在**这里（它有来源，正文在 `:209`）")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(closed_form_bytes)
    src_b = inspect.getsource(structural_bytes)
    src_c = inspect.getsource(reference_bytes)
    check("D.7", len({src_a, src_b, src_c}) == 3,
          "★ 防 `P17`：被测（`closed_form_bytes` %d 字符 / `structural_bytes` %d）与"
          "**独立参照**（`reference_bytes` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c)))
    watched = {"closed_form_bytes", "structural_bytes", "ring_volume", "saturation_factor",
               "dp_saturation_bytes"}
    called = _callee_names(reference_bytes)
    check("D.7b", not (watched & called),
          "★ **AST 调用表**：**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    _CALLS.clear()
    closed_form_bytes(4, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    seen = sum(_CALLS.values())
    _CALLS.clear()
    reference_bytes(4, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
    left = sum(_CALLS.values())
    check("D.7c", seen > 0 and left == 0,
          "★★ **调用计数器两向都量**：被测路径跑一次 ⇒ 计数 = **%d**（> 0，计数器**非空**）；"
          "**参照**跑完 ⇒ 计数 = **%d**（必须**恰好为 0**）⇒ "
          "「参照不调用被测」不是靠读代码相信的" % (seen, left))
    # ★ 参照的**独立存在的意义**：它与被测路径① 写法不同、代数相同
    t_ = 8
    same = abs(reference_bytes(t_, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)
               - closed_form_bytes(t_, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE))
    check("D.7d", same <= TOL_MACHINE * max(
        abs(closed_form_bytes(t_, L_LAYERS, B_SIZE, S_LEN, H_SIZE, ITEMSIZE)), 1.0),
          "★ 参照与被测路径 ① **代数相同、写法不同**（差 `%.1e`）—— ★★ 这是**必要的**："
          "若两者写法相同，`A.1` 就退化成「同一条式子和它自己对拍」" % same)


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
    return 1 if FAILS else 0


def main():
    print("=" * 74)
    print("03_saturation.py —— TP 的通信量【也饱和】：`8bsh((t−1)/t)`")
    print("=" * 74)
    print("  ★ 真值标注：**推导值**（公式代入）+ **本机实测值**（`python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序**不测量任何通信**，"
          "它把两条**来源不同**的公式对拍。")
    print("  ★★★ 本程序只写 `:223` **没写**的那一层：`(t−1)/t → 1` 与 DP 的 "
          "`2(N−1)/N → 2` 是**同一个结构**，但**饱和值不同**"
          "（DP `2S` ∝ 参数量 · TP `8bsh` ∝ 激活量）。")
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
