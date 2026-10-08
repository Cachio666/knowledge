#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_three_stages.py —— **三个 ZeRO 阶段各切什么、省多少**。

配套文档：`01-three-stages.md`。

★★★ **本主题的主线：分片不是除法。**
  `16Ψ` 是**单卡**的口径（参数 `2Ψ` + 梯度 `2Ψ` + 优化器状态 `12Ψ`）——
  但**把状态切到 `N` 张卡上，每卡占用的字节【不是】 `16Ψ/N`**：
  每个阶段**只切它自己声明的那几项**，没切的项**留在每张卡上**。
  ⇒ `P_os`（只切优化器状态）在 `N → ∞` 时仍趋近 **`4Ψ`**（参数 `2Ψ` + 梯度 `2Ψ` 没切），
    只有 `P_os+g+p`（三项全切）才到 **`16Ψ/N`**。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**
> （切了几项 ⇒ 每卡还剩几项 ⇒ 合起来几个字节）。

★★★ **A 类判据的落点（本程序最要紧的一处设计）**：
  · **✗ 不能写成** `assert per_card == ZERO_TOTAL_MULTIPLE * Psi / N` 这种 ——
    那是**乘法自证**（右边就是把左边再算一遍）。本程序**一个这样的断言都没有**。
  · **✓ A 类**：**真的建出四种分片状态、量 `nbytes`**，断言
    **「每卡字节 ÷ (16Ψ/N) 只有【三项全切】才 `== 1`」** ——
    它把「**名字与切项要对上**」这件事钉住（`ZeRO-0` / `P_os` / `P_os+g` / `P_os+g+p`
    各自切哪几项），且 `A.1g` 的守卫证明**改错配置它真的会失败**。
  · ★★★ **参照（闭式）不调用被测（建状态）** —— `closed_form_bytes()` 只用 `specs.py` 的常量
    与算术，不调用 `sharded_state_bytes()`（`D.8` 给**源码级线索**，★ **不是证明**）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：把 `sharded_state_bytes()` 里的「按 `shard_*` 标志分片」删掉
    （即**三档都不切**）⇒ 四种状态全是 `16Ψ` ⇒ `A.1` 的「只有全切才 `== 1`」**必然失败**
    ⇒ ★ `A.1` **不是自证**（`A.1g` 把这个反例真的跑了一遍）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **优化器状态的更新式**（SGD / momentum / Adam / AdamW）→ `90-training/optimizers/01,02`；
  · ★★★ **`16Ψ` 的组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）→
    `90-training/checkpointing/01-state-and-bytes.md:97-107`（§1.3 + `:229-230` 判据表）——
    ★ **本程序只写「分片后」的账**，**不重推分解**；
  · **`4x`/`8x` 与 `P_os`/`P_os+g` 的对应关系** →
    `90-training/checkpointing/02-sharded-save-and-load.md:156`（`:268` 判据表）；
  · **混合精度的类型选择**（主权重 / 梯度 / 优化器状态三者的类型表）→
    `90-training/mixed-precision/01-type-selection.md`；
  · **迭代精化 / 主权重为什么必须 fp32 的【数值】理由** →
    `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ 四种分片状态下「每卡字节 ÷ (16Ψ/N)」**只有【三项全切】才 `== 1`**，
         其余三档都 `> 1`（真的建状态、量 `nbytes`）；
  · `A.1g` ★★★ **一条守卫**：把「全切」那一档的**参数不分片**（配置改错）⇒
         归一化值 `1.875 ≠ 1` ⇒ `A.1` 的判据**必须失败**；
  · `A.2`  ★★ 四种状态真的**建出来了**（`nbytes` 之和 > 0，且四种两两不同）；
  · `B.1`  ★★ `P_os` 在 `ZERO_N_GRID` 六档下的「每卡字节 ÷ Ψ」逐档与闭式对拍（本机实测）；
  · `B.2`  ★★ `P_os+g` 在六档下同样逐档对拍；两档都**严格单调下降**；
  · `C.1`  ★★★ **收益饱和（`P_os` 的下界是 `4Ψ`，不是 `0`）** —— `P_os(N) ≥ 4.0` 恒成立，
         且 `N = 4096` 时 `< 4.003`；`C.1g` 守卫：若参数也切，下界掉到 `2Ψ` ⇒ 判据失败；
  · `C.2`  ★★★ **`P_os+g` 的下界是 `2Ψ`**（只有参数没切）—— 同样恒成立且已验证；
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


# ===========================================================================
# 1. 被测：一个「分片后每卡还剩几个字节」的状态构造器
#    ★★★ 它是**真的建 numpy 数组**、**真的切**、**真的量 `nbytes`** ——
#      **不是**把 `16*Psi/N` 算一遍。
#    ★★ 种子/值写在函数体内（**不是模块级常量**）：本程序只量**字节数**，不看数组内容
#      ⇒ 改它们对任何断言都没有可观察后果（放进变异分母只会制造不过期漏网）。
# ===========================================================================
def sharded_state_bytes(Psi, N, shard_param, shard_grad, shard_opt):
    """建出**一张卡上**的那份分片状态，返回它的**总字节数**（`nbytes` 之和）。

    · 每参数字节：参数 `2`（fp16）、梯度 `2`（fp16）、优化器状态 `12`（= fp32 主权重 `4` +
      `m` `4` + `v` `4`；★ `K = 12` 见 `specs.ZERO_K_MEMORY_MULTIPLE`，那是**内存乘子**）。
    · `shard_*` 为真 ⇒ 这一项在 `N` 张卡上**均分**（本卡拿 `Psi // N` 个，**要求整除**）；
      为假 ⇒ 本卡留**完整**的一份。
    ★★★ **只有三项全 `True` 时它才 `== 16Ψ/N`** —— 这就是本主题的主线。
    """
    def held(n_arrays, dtype, sharded):
        total = 0
        for _ in range(n_arrays):
            arr = np.zeros(Psi, dtype=dtype)
            if sharded:
                arr = np.array_split(arr, N)[0]      # 本卡（第 0 号）拿到的那一片
            total += arr.nbytes
        return total

    p = held(1, np.float16, shard_param)             # 参数： 1 个 Ψ 长的 fp16 数组
    g = held(1, np.float16, shard_grad)              # 梯度： 1 个 Ψ 长的 fp16 数组
    opt = held(3, np.float32, shard_opt)             # 优化器状态：主权重 + m + v（各 Ψ 长 fp32）
    return p + g + opt


def closed_form_bytes(Psi, N, shard_param, shard_grad, shard_opt):
    """**独立闭式**（参照）：只用 `specs.py` 的常量与算术，**不调用** `sharded_state_bytes()`。

    ★★★ 与上一条是**两条来源不同的路径**：一条走 `numpy.nbytes`（由**形状与 dtype** 决定），
    一条走 `specs.py` 的 `[C]`/`[E]` 常量与除法 ⇒ 两者相等**不是恒真**（本库 `90-training`
    的判据① 用的同一个形态）。
    """
    bpp = specs.ZERO_BYTES_PER_PARAM
    k = specs.ZERO_K_MEMORY_MULTIPLE
    per_param_bytes = (bpp / N) if shard_param else bpp
    per_grad_bytes = (bpp / N) if shard_grad else bpp
    per_opt_bytes = (k / N) if shard_opt else k
    return (per_param_bytes + per_grad_bytes + per_opt_bytes) * Psi


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
# 2. A 类：四种分片状态真的建出来了，且**只有三项全切**才归一化到 1
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 四种分片状态（**真的建、量 nbytes**）：**只有三项全切**才 `== 16Ψ/N`")
    print("=" * 78)
    Psi, N = specs.ZERO_A_PSI, specs.ZERO_A_N
    unit = specs.ZERO_TOTAL_MULTIPLE * Psi / N        # 16Ψ/N（**参照口径，不是被测**）
    print("  Ψ = %d（元素）· N = %d · 参照口径 16Ψ/N = %d 字节/卡"
          % (Psi, N, unit))
    print("  ★★ 四种状态切哪几项：ZeRO-0 不切 · P_os 切优化器状态 · "
          "P_os+g 再切梯度 · P_os+g+p 再切参数")

    # (名字, shard_param, shard_grad, shard_opt)
    configs = [(specs.ZERO_STAGE_NAMES[0], False, False, False),
               (specs.ZERO_STAGE_NAMES[1], False, False, True),
               (specs.ZERO_STAGE_NAMES[2], False, True, True),
               (specs.ZERO_STAGE_NAMES[3], True, True, True)]
    rows = []
    for name, sp, sg, so in configs:
        b = sharded_state_bytes(Psi, N, sp, sg, so)
        rows.append((name, b, b / unit))
    print()
    for name, b, norm in rows:
        print("    %-10s ⇒ 每卡 **%6d B** · ÷ (16Ψ/N) = **%.4f**" % (name, b, norm))
    print("  ★★★ 只有最后一档（`%s`）的归一化值是 **1.0000** —— 因为**只有它三项全切**；"
          % specs.ZERO_STAGE_NAMES[3])
    print("     `P_os` 与 `P_os+g` **永远到不了 1**（参数、或参数+梯度留在每张卡上）。")

    full_norm = rows[3][2]
    others = [norm for _n, _b, norm in rows[:3]]
    check("A.1", abs(full_norm - 1.0) < TOL_MACHINE and all(v > 1.0 + TOL_MACHINE for v in others),
          "★★★ **判据成立**：四种状态**真的建出来**了；归一化值 = %s ⇒ "
          "**只有三项全切那一档 `== 1.0000`**，其余三档都 `> 1`（`P_os` = %.4f · `P_os+g` = %.4f）"
          % (" / ".join("%.4f" % x for x in [r[2] for r in rows]), others[0], others[1]),
          "**四种状态的字节账被动过** —— ★ 先查「名字 ↔ 切了哪几项」是否还对得上")

    # ── A.1g ★★★ 守卫 ─────────────────────────────────────────────────────
    # ★★ 守卫的错法：把「全切」那一档的**参数漏掉**（`shard_param=False`）——
    #    ★★★ 这条改动**真的改变了被判量**：归一化值从 1.0 变成 1.875（参数那一份没切）。
    miswired = sharded_state_bytes(Psi, N, False, True, True) / unit
    check("A.1g", abs(miswired - 1.0) > TOL_MACHINE,
          "★★ **守卫成立**：「全切」那一档若**漏切参数**（配置改错）⇒ 归一化值 = **%.4f ≠ 1**"
          " ⇒ ★★ 换掉「参数切不切」这个条件**真的改变了被判量**，`A.1` **能失败** ✓" % miswired,
          "**守卫失效** —— 参数漏切后归一化值居然仍是 1（`A.1` 对配置错误不敏感）")

    # ── A.2 ★★ 四种状态真的建出来了、且两两不同 ────────────────────────────
    vals = [b for _n, b, _r in rows]
    check("A.2", all(v > 0 for v in vals) and len(set(vals)) == 4,
          "★★ **判据成立**：四种状态的每卡字节 = %s ⇒ 都 `> 0` 且**两两不同** "
          "（★ 说明它们是**真的建出来的四种状态**，不是同一个占位值）" % vals,
          "**四种状态没有真的建出来** —— ★ 有重复或非正值（建状态的逻辑被动过）")


# ===========================================================================
# 3. B 类：`P_os` / `P_os+g` 在六档 N 下的字节账（闭式对拍）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：`P_os` 与 `P_os+g` 的收益随 N 【饱和】")
    print("=" * 78)
    Psi = specs.ZERO_B_PSI
    print("  Ψ = %d（元素，★ 被 N_GRID 的最大档 %d 整除 ⇒ 能真的建分片）"
          % (Psi, specs.ZERO_N_GRID[-1]))

    # ── B.1 ★★ P_os（只切优化器状态）────────────────────────────────────────
    print()
    print("  ── B.1 ★★ `P_os`：每卡字节 ÷ Ψ，逐档与**独立闭式**对拍 ──")
    ok1, ratios1, prev = True, [], None
    for n in specs.ZERO_N_GRID:
        measured = sharded_state_bytes(Psi, n, False, False, True)
        closed = closed_form_bytes(Psi, n, False, False, True)
        ratio = measured / Psi
        ok1 = ok1 and (measured == closed) and (ratio > 4.0 - 1e-12)
        if prev is not None:
            ok1 = ok1 and (ratio < prev)             # 严格单调下降
        prev = ratio
        ratios1.append(ratio)
        print("     N=%-5d ⇒ 每卡 **%7d B** · ÷Ψ = **%.4f** · 闭式 %.4f ⇒ %s"
              % (n, measured, ratio, closed, "**一致** ✓" if measured == closed else "**不一致** ✗"))
    check("B.1", ok1 and len(specs.ZERO_N_GRID) == 6,
          "★★ **判据成立**：`P_os` 六档「每卡字节 ÷ Ψ」= %s ⇒ 逐档与闭式 `2+2+12/N` **精确一致**，"
          "且**严格单调下降**、且**始终 `> 4`**（★ 下界在 `C.1`）"
          % " / ".join("%.4f" % r for r in ratios1),
          "**`P_os` 的账被动过** —— ★ 先查每档是不是仍与闭式一致、是否仍单调")

    # ── B.2 ★★ P_os+g（再切梯度）───────────────────────────────────────────
    print()
    print("  ── B.2 ★★ `P_os+g`：再切梯度；每卡字节 ÷ Ψ 逐档对拍 ──")
    ok2, ratios2, prev = True, [], None
    for n in specs.ZERO_N_GRID:
        measured = sharded_state_bytes(Psi, n, False, True, True)
        closed = closed_form_bytes(Psi, n, False, True, True)
        ratio = measured / Psi
        ok2 = ok2 and (measured == closed) and (ratio > 2.0 - 1e-12)
        if prev is not None:
            ok2 = ok2 and (ratio < prev)
        prev = ratio
        ratios2.append(ratio)
        print("     N=%-5d ⇒ 每卡 **%7d B** · ÷Ψ = **%.4f** · 闭式 %.4f ⇒ %s"
              % (n, measured, ratio, closed, "**一致** ✓" if measured == closed else "**不一致** ✗"))
    print("  ★★ **`P_os+g` 比 `P_os` 少的那一块正是梯度**（`2Ψ` 那一份不切时不省）"
          "—— 见 `C.2` 的下界 `2Ψ`。")
    check("B.2", ok2,
          "★★ **判据成立**：`P_os+g` 六档「每卡字节 ÷ Ψ」= %s ⇒ 逐档与闭式 `2+2/N+12/N` **精确一致**，"
          "且**严格单调下降**、且**始终 `> 2`**（★ 下界在 `C.2`）"
          % " / ".join("%.4f" % r for r in ratios2),
          "**`P_os+g` 的账被动过** —— ★ 先查每档是不是仍与闭式一致、是否仍单调")


# ===========================================================================
# 4. C 类：收益【饱和】—— 下界是 4Ψ / 2Ψ，**不是 0**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**N 大到某值后收益消失** —— 写出那个下界")
    print("=" * 78)
    Psi, Nmax = specs.ZERO_B_PSI, specs.ZERO_N_GRID[-1]
    # ★★ 经验下界 = 每卡字节 − （被切那一项摊到的字节）/ Ψ
    pos_now = sharded_state_bytes(Psi, Nmax, False, False, True)
    pos_floor = (pos_now - specs.ZERO_K_MEMORY_MULTIPLE * Psi / Nmax) / Psi
    pog_now = sharded_state_bytes(Psi, Nmax, False, True, True)
    pog_floor = (pog_now - specs.ZERO_K_MEMORY_MULTIPLE * Psi / Nmax
                 - specs.ZERO_BYTES_PER_PARAM * Psi / Nmax) / Psi
    print("  `P_os` 在 N=%d 时：每卡 %d B ⇒ ÷Ψ = %.4f；减去被切的优化器状态那一摊 ⇒ **下界 %.4fΨ**"
          % (Nmax, pos_now, pos_now / Psi, pos_floor))
    print("  `P_os+g` 在 N=%d 时：每卡 %d B ⇒ ÷Ψ = %.4f；再减去被切的梯度那一摊 ⇒ **下界 %.4fΨ**"
          % (Nmax, pog_now, pog_now / Psi, pog_floor))

    # ── C.1 ★★★ P_os 的下界是 4Ψ（参数 2 + 梯度 2）──────────────────────────
    # ★★ 参照：`16 − K = 2 × 每参数字节` —— **两条来源不同的路径**：左边是「总量减优化器状态」，
    #    右边是「参数 + 梯度两项」。它们相等**不是恒真**（一个是 specs 的两个常量相减，
    #    一个是 bpp 乘 2）；★ 而经验下界 `pos_floor` 又独立地量了一次。
    pos_floor_expected = specs.ZERO_TOTAL_MULTIPLE - specs.ZERO_K_MEMORY_MULTIPLE
    check("C.1", abs(pos_floor - pos_floor_expected) < TOL_MACHINE
          and abs(pos_floor_expected - 2 * specs.ZERO_BYTES_PER_PARAM) < TOL_MACHINE
          and pos_floor > 0.0,
          "★★★ **判据成立**：`P_os` 的下界**实测** %.4fΨ == `16 − K` = %d == `2 × 每参数字节` = %d "
          "⇒ ★★★ **它不是 0**（参数与梯度不切）—— **「分片不是除法」的直接证据**"
          % (pos_floor, pos_floor_expected, 2 * specs.ZERO_BYTES_PER_PARAM),
          "**`P_os` 的下界被动过** —— ★ 先查「参数与梯度在 `P_os` 里是否仍留完整一份」")

    # ── C.1g ★★★ 守卫：参数也切 ⇒ 下界掉到 2Ψ ─────────────────────────────
    # ★★ 守卫的错法：把 `P_os` 也当成「参数全切」—— ★★★ 这条改动**真的改变了被判量**：
    #    下界从 4Ψ 掉到 2Ψ（参数那一份被切走了）。
    wrong_now = sharded_state_bytes(Psi, Nmax, True, False, True)
    wrong_floor = (wrong_now - specs.ZERO_K_MEMORY_MULTIPLE * Psi / Nmax
                   - specs.ZERO_BYTES_PER_PARAM * Psi / Nmax) / Psi
    check("C.1g", abs(wrong_floor - pos_floor) > TOL_MACHINE,
          "★★ **守卫成立**：若把 `P_os` 改成**参数也切**（配置改错）⇒ 下界从 **%.4fΨ** 掉到 "
          "**%.4fΨ** ⇒ ★★ 换掉「参数切不切」真的改变了被判量，`C.1` **能失败** ✓"
          % (pos_floor, wrong_floor),
          "**守卫失效** —— 参数也切后下界没变（`C.1` 对「参数是否分片」不敏感）")

    # ── C.2 ★★★ P_os+g 的下界是 2Ψ（只有参数没切）──────────────────────────
    pog_floor_expected = specs.ZERO_BYTES_PER_PARAM
    check("C.2", abs(pog_floor - pog_floor_expected) < TOL_MACHINE and pog_floor > 0.0,
          "★★★ **判据成立**：`P_os+g` 的下界**实测** %.4fΨ == 每参数字节 = %d "
          "⇒ ★★★ **只有参数没切** ⇒ 它也不是 0" % (pog_floor, pog_floor_expected),
          "**`P_os+g` 的下界被动过** —— ★ 先查「参数在 `P_os+g` 里是否仍留完整一份」")

    # ── C.3 ★★ 「收益消失」的量化：4096 → 8192 的边际收益 ───────────────────
    # ★★ 这里**不用 `ZERO_N_GRID` 之外的档**做「真的建」——只做闭式外推（标注为推导值），
    #    因为 8192 > `ZERO_B_PSI` 会破坏整除（那条纪律是 A 类的）。
    pos_4096 = sharded_state_bytes(Psi, Nmax, False, False, True) / Psi
    pos_8192 = closed_form_bytes(Psi, 2 * Nmax, False, False, True) / Psi   # 推导值（闭式外推）
    gain = (pos_4096 - pos_8192) / pos_4096 * 100.0
    check("C.3", gain < 0.05 and pos_8192 < pos_4096,
          "★★ **判据成立**：`N` 从 %d 翻倍到 %d，`P_os` 只从 %.4fΨ 降到 %.4fΨ ⇒ "
          "**再翻一倍 N 也只省 %.4f%%** ⇒ ★★ **收益已经饱和**（" % (Nmax, 2 * Nmax, pos_4096, pos_8192, gain) +
          "★ 后者是**闭式外推的推导值**，不是本机建出来的)",
          "**收益没有饱和** —— ★ 先查下界是不是还是 4Ψ")


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
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` / `numpy %s`。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · **`16Ψ` 的组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）→ "
          "`90-training/checkpointing/01-state-and-bytes.md:97-107`（§1.3 + `:229-230` 判据表）")
    print("      · **`4x`/`8x` 与 `P_os`/`P_os+g` 的对应关系** → "
          "`90-training/checkpointing/02-sharded-save-and-load.md:156`")
    print("      · **优化器状态的更新式** → `90-training/optimizers/01,02`；"
          "**混合精度的类型选择** → `90-training/mixed-precision/01-type-selection.md`")
    print("      · **分片检查点的落盘与恢复** → `90-training/checkpointing/02,03`"
          "（★ 本域只写「分片（放进显存）」，不写「落盘」）")
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
    check("D.2", ASSERT_FAIL_MARK == "?? FAIL: zero-fsdp" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `?? FAIL: zero-fsdp`、且含 `FAIL`\n"
          "        ★★ 那几个 `?` 在 shell / make 里都是元字符 ⇒ 它**只出现在 `print()` 里**，"
          "**不出现在任何命令行或 recipe 行**；★ Makefile 的注释里也照原样写（注释不执行）。")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                                        "|| FAIL: expert-parallel"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住；"
          "★ 六个标记**两两不互为前缀**")

    # ── D.3 ★ 口径常量（本程序的**输入**与阈）──────────────────────────────
    check("D.3", specs.TOL_MACHINE == 1.0e-12
          and specs.ZERO_BYTES_PER_PARAM == 2 and specs.ZERO_K_MEMORY_MULTIPLE == 12
          and specs.ZERO_TOTAL_MULTIPLE == 16
          and specs.ZERO_TOTAL_MULTIPLE == 2 * specs.ZERO_BYTES_PER_PARAM + specs.ZERO_K_MEMORY_MULTIPLE,
          "★ 每参数字节 = %d · `K`（内存乘子）= %d · `16Ψ` = %d ⇒ `2×2 + 12 = 16` ✓；"
          "★ 机器精度档阈 = %g" % (specs.ZERO_BYTES_PER_PARAM, specs.ZERO_K_MEMORY_MULTIPLE,
                                   specs.ZERO_TOTAL_MULTIPLE, specs.TOL_MACHINE))
    check("D.3b", all(isinstance(v, int) for v in specs.ZERO_N_GRID)
          and specs.ZERO_A_PSI % specs.ZERO_A_N == 0
          and all(specs.ZERO_B_PSI % n == 0 for n in specs.ZERO_N_GRID),
          "★★ `ZERO_N_GRID` 全整数；`ZERO_A_PSI=%d` 被 `ZERO_A_N=%d` 整除；`ZERO_B_PSI=%d` "
          "被六档 N **全整除** ⇒ 分片边界整齐（★ 不整除的处置**不是** 01 的题目）"
          % (specs.ZERO_A_PSI, specs.ZERO_A_N, specs.ZERO_B_PSI))
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：specs 常量逐位钉住 · 逐元素钉住 · 引文纪律 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · specs 常量逐位钉住 · **逐元素**钉住 · 引文纪律 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★ `[C]` 常量逐位钉住（本程序用到的日志面）──────────────────────
    check("D.4", specs.ZERO_K_MEMORY_MULTIPLE == 12 and specs.ZERO_BYTES_PER_PARAM == 2
          and specs.ZERO_A_PSI == 1024 and specs.ZERO_A_N == 8
          and specs.ZERO_B_PSI == 4096,
          "★ `[C]`/`[E]` 常量逐位钉住：`K = 12`（★ **内存乘子，不是通信量**，那是 02 的事）· "
          "每参数字节 2 · `ZERO_A_PSI = 1024` · `ZERO_A_N = 8` · `ZERO_B_PSI = 4096`")

    # ── D.7 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）─────────────
    check("D.7", specs.ZERO_N_GRID == (1, 2, 8, 64, 512, 4096)
          and specs.ZERO_STAGE_NAMES == ("ZeRO-0", "P_os", "P_os+g", "P_os+g+p"),
          "★★★ **两个元组【逐元素】钉住**（`ZERO_N_GRID` 六档 · `ZERO_STAGE_NAMES` 四档）——"
          " ★ 只钉「整体」会让**元素级**变异漏网（本库 `pipeline-parallel` 那一轮实测过："
          " 一次报出 58 条漏网，其中 38 条出自「只读整体」）")

    # ── D.8 ★★ 独立性线索（源码级，★ **不是证明**）────────────────────────
    def _calls(fn):
        """收集 `fn` 源码里的**调用名**（`f(...)` 与 `mod.f(...)` 的根名）。"""
        out = set()
        for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    out.add(f.id)
                elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                    out.add(f.value.id)
        return out
    calls_test = _calls(sharded_state_bytes)
    calls_ref = _calls(closed_form_bytes)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照的**调用表** ──")
    print("     `sharded_state_bytes()` 里调用的函数：%s"
          % (", ".join(sorted(calls_test)) if calls_test else "**空集**"))
    print("     `closed_form_bytes()`   里调用的函数：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    print("     ⇒ ★★ `closed_form_bytes()` **不调用** `sharded_state_bytes()`"
          "（也不调用 `np.zeros` / `np.array_split`）——")
    print("        **参照不调用被测**（域 README 的硬要求）在此**在源码层面成立**。")
    print("     ★★★ **降级措辞**：这只是「两段代码不互相调用」的**源码级线索**，"
          "**不证明**「A/C 类判据有物理信息量」—— 后者由 `A.1g`/`C.1g`/`C.2` 的**行为**兜底。")
    check("D.8", "sharded_state_bytes" not in calls_ref and "np" not in calls_ref
          and "sharded_state_bytes" not in calls_test,
          "★★ **源码级线索成立**：`closed_form_bytes()` 的调用表里**没有** `sharded_state_bytes` "
          "⇒ ★ **参照不调用被测**（源码层面）",
          "**参照与分片版互相调用** —— §A/§C 的判据可能退化成自证")

    # ── D.9 ★★ 记号的桥（`S = 2Ψ`）与 P30 的现场 ────────────────────────────
    check("D.9", specs.ZERO_S_IS_GRAD_BYTES_MULT == 2,
          "★★ **记号的桥**：`performance-models/02` 的 `S` = 梯度**字节**、ZeRO 的 `Ψ` = **参数量**，"
          "梯度 fp16 ⇒ **`S = 2Ψ`** ⇒ 该文的 `2S → 3S` **就是**论文的 `2Ψ → 3Ψ`。\n"
          "        ★★★ **那句话【是对的】，一个字都不许改**（计划 §5 v3 已**撤回**「订正」判定；"
          "成因是 v1/v2 没引 baseline 的定义 ⇒ 计划新立 `P30`）。★ 详见 `02_communication.py` 的 `D.7`。")

    # ── D.10 ★ `UNSOURCED` 留空 ────────────────────────────────────────────
    check("D.10", specs.UNSOURCED == (),
          "★ `specs.UNSOURCED` **留空**（与同域另三个主题一致）——★ 本主题的「未复核」写在"
          "各自常量的 `# 来源：` 里（★ FSDP 那批**未在原始字节上复核**），不欠一条记录。")


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
    print("★ `16Ψ` 的单卡口径在 `90-training/checkpointing/01-state-and-bytes.md`；"
          "本程序只写「**分片后**」的账。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_three_stages.py —— 三个 ZeRO 阶段各切什么、省多少（**分片不是除法**）")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **主线**：每卡字节 **不是** `16Ψ/N` —— 每个阶段只切它声明的那几项，"
          "没切的项**留在每张卡上**。")
    print("      ⇒ `P_os` 的下界是 **4Ψ**、`P_os+g` 是 **2Ψ**；只有三项全切才到 `16Ψ/N`。")
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
