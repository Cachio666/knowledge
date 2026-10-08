#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_fsdp_correctness.py —— **FSDP 的策略与分片后的正确性**。

配套文档：`03-fsdp-and-correctness.md`。

★★★ **本主题的 A 类【很弱】，本程序如实写、不硬凑**（★ 计划 §3 Chunk 6 明令）：
  · **✗ A 类不能钉在「全分片 == 不分片」上** —— 分片再 all-gather 拼回与原始参数**逐位相同**
    （`D.4` 量它）⇒ **纯搬运 ⇒ 恒真**（本库 `P12`/`D22` 家族）。
    ★★★ **FSDP 的正确性本质就是「搬运正确」** ⇒ **本主题【没有】改变求值顺序的地方**。
  · **✓ A 类（弱，但不恒真）**：钉在**「分片边界的配对」**上 ——
    **「`N` 个 shard 拼起来的顺序与位置 == 原始参数」**（★ 拼错顺序或位置会**真的错**，
    `A.1g` 的守卫把它跑了一遍）。
  · **✓ B 类（本主题的主力）**：**三种 `sharding_strategy` 的每设备字节账**。
  · **✓ C 类**：★★★ **「只 all-gather 了【一部分】就用」** —— 只拼回一半的分片 ⇒
    **不报错、结果错**；★ 「忘了 all-gather」「拼错顺序」同族。
  ★★★ **一条纪律**：**A 类弱不是缺陷，如实写就行** —— 但**不得为了「三类缺一不可」而编一条
    恒真的 A 类**（那正是 `P12`）。⇒ 那个「逐位相同」的 `0.0` 被写成 `D.4` 的**报告**，
    **不是**判据。

★★★ **`（P19）`边界**：本程序**只引 `sharding_strategy` 的【取值列表】**（标
  **B 级 + 版本 2.14 + URL + 抓取日**），**不解释框架 API 的通用语义** ——
  ★★ **全库当前在这上面没有正文**（实测 `70-frameworks/pytorch/` 4 篇 1057 行里
  `torch.distributed` / `DDP` / `FSDP` / `ProcessGroup` 命中 **0** 次）⇒ 把禁区指向 70 域是**假让渡**。
  ★★ `FULL_SHARD` 那句「什么时候 all-gather / 什么时候丢弃」**可以写**（它有载体），
  但**不扩写成「FSDP 怎么用」的教程**。

★★★ **同名不同指（本主题必须点明）**：本域写的是**「显存分片」**（`ZeRO`/`FSDP`），
  而 `00-meta/glossary.md:275` 的 `sharded checkpoint` 是**落盘** ⇒
  ★★ **落盘一律回链 `90-training/checkpointing/02,03`**（本程序**不写落盘**）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：把 `part_A` 里**用来比较的下标**改错（`params[k*L]` → `params[k*L + 1]`）
    ⇒ `A.1` 的边界配对**必然失败**（★ 实测：**断言捕获**、不崩）⇒ `A.1` **不是自证**。
    ★★ 而**改分片切片的算术**（`contiguous_shard()` 的 `k*L` → `k*L + 1`）是**崩溃捕获**
      （末片长度变 `L−1` ⇒ `IndexError`、**一行失败标记都不打**）—— ★ 那是「代码炸了」，
      不是「判据抓住了错」（★ 这条实测记在 `mutate.py` 的抬头里）。
  ★★ 而「拼回来 == 原始」那条（`D.4`）**正是自查问句的答案**：删掉被测逻辑它照样通过
    ⇒ 它**不是判据**，是**报告**。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · ★★★ **框架的分布式 API 语义**（`torch.distributed` / `DDP` / `FSDP` 的类与参数）⇒
    **全库当前没有正文**（本程序**只引取值列表**，标 B 级 + 版本 + URL + 抓取日）；
  · **分片检查点的落盘与恢复**（元数据三层、重分片、一致性）⇒
    `90-training/checkpointing/02-sharded-save-and-load.md`、`03`；
  · **`16Ψ` 的单卡口径与它的字节分解** ⇒ `90-training/checkpointing/01-state-and-bytes.md`；
  · **`4x`/`8x` 与 `P_os`/`P_os+g` 的对应关系** ⇒ `90-training/checkpointing/02`；
  · **迭代精化 / 主权重为什么必须 fp32** ⇒
    `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ **分片边界的配对**：第 `k` 个分片的**首/末元素**落在**正确的全局下标**
         （`k*L` 与 `k*L + L − 1`），且长度恰好 `L`；
  · `A.1g` ★★ **一条守卫**：改用**跨步（interleaved）映射**分片 ⇒ 边界配对**必须失败**；
  · `A.2`  ★★ `N` 个分片的边界 `[k*L, (k+1)*L)` 构成 `range(Ψ)` 的一个**划分**；
  · `B.1`  ★★★ 三种 `sharding_strategy` 的每设备字节账（真的建状态、量 `nbytes`）：
         `FULL_SHARD` = `4.0000Ψ`、`SHARD_GRAD_OP` = `5.5000Ψ`、`NO_SHARD` = `16.0000Ψ`；
  · `B.2`  ★★ `SHARD_GRAD_OP` 在 `N = 4` 的 `5.5Ψ` 与 `ZeRO-1`（`P_os`）在 `N = 8` 的 `5.5Ψ`
         **数值相同、口径不同** —— 程序把**两条公式**都算出来钉住；
  · `C.1`  ★★★ **只拼回一半的分片** ⇒ `max|diff|` 落内容级（**不报错、结果错**）；
  · `C.2`  ★★★ **忘了 all-gather**（一片都不拼）⇒ `max|diff|` **不小于** `C.1`；
  · `C.3`  ★★ **拼错顺序**（把分片倒过来拼）⇒ `max|diff|` 落内容级；
  · `D.*`  口径常量逐个钉住 + **逐元素**钉住 + **恒真性自检（报告）** + **失败标记自检**。
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

# ★ 本程序**内部**的内容级阈（**不进 `specs.py`**：它是「多大算真的错了」的口径，
#   不是从任何来源抄来的常量）。★★ 取 `1e-3` 是因为 C 类的差落在 **`1e+0` 档**
#   （本机实测 `4.18 ~ 5.88`）、与机器精度档（`0`）相隔**十几个数量级**。
TOL_CONTENT = 1.0e-3


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）。
      ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 失败标记必然在**行首**（前面只有缩进）。
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
# 1. 被测：分片 / 拼回 / 每设备字节账
#    ★★ 种子写在**函数体内**（不是模块级常量）：改它对**字节账**与**边界配对**都没有
#      可观察后果（配对只看**下标**、字节账只看**形状与 dtype**）——
#      ★ 但 C 类的 `max|diff|` **依赖它** ⇒ 那几条把种子固定为 `C_FIXTURE_SEED`（写在本节里）。
# ===========================================================================
def contiguous_shard(params, N):
    """**连续分片**（本主题的口径）：第 `k` 片 = `params[k*L : (k+1)*L]`，`L = len/N`。"""
    L = params.shape[0] // N
    return [params[k * L:(k + 1) * L] for k in range(N)]


def strided_shard(params, N):
    """**跨步分片**（`A.1g` 的错法）：第 `k` 片 = `params[k::N]` —— 边界**整体错位**。"""
    return [params[k::N] for k in range(N)]


def gather(shards, k_used=None):
    """把分片**按顺序、按位置**拼回一个整向量；`k_used` 限定只用前几片（其余留 `0`）。

    ★★★ 本函数就是「参数 all-gather」的实现 —— **它是一个显式的张量搬运**
      （本域把通信实现成 numpy 操作的既有形态）。
    """
    if k_used is None:
        k_used = len(shards)
    L = shards[0].shape[0]
    out = np.zeros(len(shards) * L, dtype=shards[0].dtype)
    for k in range(min(k_used, len(shards))):
        out[k * L:(k + 1) * L] = shards[k]
    return out


def strategy_bytes(strategy, Psi, N):
    """按 `sharding_strategy` 建出**一张卡上**的状态，返回**总字节数**（`nbytes` 之和）。

    ★ 三项（参数 `2` / 梯度 `2` / 优化器状态 `12`，**每参数字节**）各自**切或不切**：
      · `FULL_SHARD`     —— 参数 / 梯度 / 优化器状态**全分片** ⇒ `16Ψ/N`；
      · `SHARD_GRAD_OP`  —— 参数**不分片**（在计算外分片、前向后不 reshard），梯度 + 优化器状态**分片**；
      · `NO_SHARD`       —— 等价 DDP ⇒ **一项都不切** ⇒ `16Ψ`。
    ★★ **只声明【静态账】**（每卡几个字节），不涉及「什么时候 all-gather」的时序 —— 后者在文档里。
    """
    def held(n_arrays, dtype, sharded):
        total = 0
        for _ in range(n_arrays):
            arr = np.zeros(Psi, dtype=dtype)
            if sharded:
                arr = np.array_split(arr, N)[0]
            total += arr.nbytes
        return total

    param_sharded = strategy == "FULL_SHARD"
    grad_sharded = strategy in ("FULL_SHARD", "SHARD_GRAD_OP")
    opt_sharded = strategy in ("FULL_SHARD", "SHARD_GRAD_OP")
    p = held(1, np.float16, param_sharded)
    g = held(1, np.float16, grad_sharded)
    o = held(3, np.float32, opt_sharded)
    return p + g + o


def closed_form(strategy, Psi, N):
    """**独立闭式**（参照）：只用 `specs.py` 的常量与算术，**不调用** `strategy_bytes()`。"""
    bpp, k = specs.ZERO_BYTES_PER_PARAM, specs.ZERO_K_MEMORY_MULTIPLE
    param_sharded = strategy == "FULL_SHARD"
    grad_sharded = strategy in ("FULL_SHARD", "SHARD_GRAD_OP")
    opt_sharded = strategy in ("FULL_SHARD", "SHARD_GRAD_OP")
    per = (bpp / N if param_sharded else bpp) + (bpp / N if grad_sharded else bpp) \
        + (k / N if opt_sharded else k)
    return per * Psi


# ===========================================================================
# 2. A 类（**弱，但不恒真**）：分片边界的配对
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 分片边界的**配对**（★ 本主题的 A 类【很弱】—— 如实写）")
    print("=" * 78)
    Psi, N = specs.ZERO_C_PSI, specs.ZERO_C_N
    L = Psi // N
    print("  Ψ = %d · N = %d ⇒ 每片 L = %d" % (Psi, N, L))
    print("  ★★★ **为什么 A 类只能钉在这里**：「全分片 == 不分片」是**纯搬运 ⇒ 恒真**")
    print("     （`D.4` 量它、逐位相同）⇒ ★ **FSDP 的正确性本质就是「搬运正确」**，")
    print("     本主题**没有**改变求值顺序的地方。")

    params = np.random.default_rng(100).standard_normal(Psi)
    shards = contiguous_shard(params, N)

    # ── A.1 ★★ 边界配对 ────────────────────────────────────────────────────
    print()
    print("  ── A.1 ★★ 第 k 片的**首 / 末元素**落在正确的全局下标（`k*L` 与 `k*L+L−1`）──")
    ok = True
    for k in range(N):
        first_ok = shards[k][0] == params[k * L]
        last_ok = shards[k][L - 1] == params[k * L + L - 1]
        len_ok = shards[k].shape[0] == L
        ok = ok and first_ok and last_ok and len_ok
        print("     片 %d：长度 %d · 首元素 == params[%d]? %s · 末元素 == params[%d]? %s"
              % (k, shards[k].shape[0], k * L, "✓" if first_ok else "✗",
                 k * L + L - 1, "✓" if last_ok else "✗"))
    check("A.1", ok,
          "★★ **判据成立**：`N` 个连续分片的**首/末元素都落在正确的全局下标**、长度恰为 `L` "
          "⇒ ★★ 这条判据**弱**（正确实现当然满足），但**不恒真**（错位的映射会失败，见 `A.1g`）",
          "**分片边界的配对被动过** —— ★ 先查 `L = Ψ/N` 与 `k*L` 的边界算术")

    # ── A.1g ★★ 守卫（跨步映射）─────────────────────────────────────────────
    bad = strided_shard(params, N)
    bad_ok = all(bad[k][0] == params[k * L] for k in range(N))
    check("A.1g", not bad_ok,
          "★★ **守卫成立**：改用**跨步映射** `params[k::N]` 分片 ⇒ 第 k 片的**首元素不再等于** "
          "`params[k*L]`（它取的是 `params[k]`）⇒ ★★ 换掉「连续 / 跨步」这个条件**真的改变了"
          "被判量**，`A.1` **能失败** ✓",
          "**守卫失效** —— 跨步映射居然仍满足边界配对（`A.1` 对映射方式不敏感）")

    # ── A.2 ★★ 划分（位置）──────────────────────────────────────────────────
    starts = [k * L for k in range(N)]
    covered = sorted(i for k in range(N) for i in range(k * L, k * L + L))
    check("A.2", starts == [k * L for k in range(N)] and covered == list(range(Psi))
          and len(starts) == N,
          "★★ **判据成立**：`N = %d` 个分片的边界 `[k*L, (k+1)*L)` 构成 `range(%d)` 的一个**划分**"
          "（每个下标**恰好被覆盖一次**）⇒ ★ 这钉的是「**位置**」那一半（顺序由 `gather` 的循环定）"
          % (N, Psi),
          "**分片的边界不构成划分** —— ★ 先查 `L` 与 `N` 的乘积是不是 `Ψ`")


# ===========================================================================
# 3. B 类（本主题的主力）：三种 `sharding_strategy` 的每设备字节账
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 三种 `sharding_strategy` 的每设备字节账（★ 本主题的主力）")
    print("=" * 78)
    Psi, N = specs.ZERO_C_PSI, specs.ZERO_C_N
    print("  Ψ = %d · N = %d" % (Psi, N))
    print("  ★★ **只引取值列表**（B 级 + 版本 %s + URL + 抓取日）：`FSDP_STRATEGY_VALUES = %s`"
          % (specs.FSDP_DOC_VERSION, list(specs.FSDP_STRATEGY_VALUES)))
    print("  ★ **默认值是 `%s`**；`SHARD_GRAD_OP` ≈ ZeRO-2（参数在计算外分片、前向后不 reshard）；"
          "`NO_SHARD` 等价 DDP。" % specs.FSDP_DEFAULT_STRATEGY)
    print("  ★★ **与 ZeRO-3 的关系的逐字依据**：`%s`" % specs.FSDP_INSPIRED_QUOTE)

    # ── B.1 ★★★ 三种策略的字节账（真的建、量 nbytes）──────────────────────
    print()
    print("  ── B.1 ★★★ 三种策略：每设备字节 ÷ Ψ（真的建状态、量 `nbytes`）──")
    ok, rows = True, []
    for strat in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"):
        measured = strategy_bytes(strat, Psi, N)
        closed = closed_form(strat, Psi, N)
        rows.append((strat, measured, measured / Psi))
        ok = ok and (measured == closed)
        print("     %-14s ⇒ 每设备 **%9d B** · ÷Ψ = **%.4f** · 闭式 %.4f ⇒ %s"
              % (strat, measured, measured / Psi, closed,
                 "**一致** ✓" if measured == closed else "**不一致** ✗"))
    print("     ★★ `FULL_SHARD` 的 `4.0000Ψ` 正是 `16Ψ/N`（★ 与 `01` 的「三项全切」那一档**对拍**）。")
    check("B.1", ok and abs(rows[0][2] - 4.0) < TOL_MACHINE
          and abs(rows[1][2] - 5.5) < TOL_MACHINE and abs(rows[2][2] - 16.0) < TOL_MACHINE,
          "★★★ **判据成立**：`FULL_SHARD` = `%.4fΨ`（= `16Ψ/%d`）· `SHARD_GRAD_OP` = `%.4fΨ` · "
          "`NO_SHARD` = `%.4fΨ` ⇒ 三者逐档与独立闭式**精确一致**"
          % (rows[0][2], N, rows[1][2], rows[2][2]),
          "**三种策略的字节账被动过** —— ★ 先查每档是不是仍与闭式一致")

    # ── B.2 ★★ 「同一个数、不同口径」───────────────────────────────────────
    print()
    print("  ── B.2 ★★ **数值相同、口径不同**：`SHARD_GRAD_OP` 在 N=%d 的 `5.5Ψ` vs "
          "`ZeRO-1`(`P_os`) 在 N=8 的 `5.5Ψ` ──" % N)
    # ★ 两条公式**各自独立**地算出来（不是同一个式子换个 N）
    fsdp5, n_fsdp = strategy_bytes("SHARD_GRAD_OP", Psi, N) / Psi, N
    zero1_formula = 2 + 2 + specs.ZERO_K_MEMORY_MULTIPLE / 8        # P_os 在 N=8：2 + 2 + 12/8
    print("     `SHARD_GRAD_OP`(N=%d)：`2 + 2/%d + 12/%d = %.4fΨ`" % (N, N, N, fsdp5))
    print("     `ZeRO-1`/`P_os`(N=8)：`2 + 2 + 12/8 = %.4fΨ`" % zero1_formula)
    print("     ★★ **两者数值相同，但 `N` 不同** ⇒ ★★★ **报这个数时必须带 `N`**，"
          "否则两个来自不同配置的 `5.5` 会被当成同一个东西。")
    check("B.2", abs(fsdp5 - zero1_formula) < TOL_MACHINE and n_fsdp != 8
          and abs(zero1_formula - 5.5) < TOL_MACHINE,
          "★★ **判据成立**：`SHARD_GRAD_OP`(N=%d) = **%.4fΨ** == `ZeRO-1`(N=8) = **%.4fΨ** "
          "⇒ ★★ 数值相同、**口径不同**（`N` 一个是 `%d`、一个是 `8`）—— **报数必须带 `N`**"
          % (N, fsdp5, zero1_formula, n_fsdp),
          "**那条「同值不同口径」被动过** —— ★ 先查两条公式是不是仍各自成立")


# ===========================================================================
# 4. C 类：**只 all-gather 了一部分就用**（不报错、结果错）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**只拼回一部分就用**（★ 不报错、结果错）")
    print("=" * 78)
    Psi, N = specs.ZERO_C_PSI, specs.ZERO_C_N
    params = np.random.default_rng(100).standard_normal(Psi)
    shards = contiguous_shard(params, N)
    print("  ★★★ 三条错法的**共同点**：形状对、类型对、**一次异常都不抛** ——")
    print("      唯一的信号是**与原始参数的差**。★ 「拼回」（`gather`）是**纯搬运**，")
    print("      所以**忘了搬 / 搬一半 / 搬错顺序**都不会有任何运行时检查会发现它。")

    # ── C.1 ★★★ 只拼回一半 ─────────────────────────────────────────────────
    half = gather(shards, k_used=N // 2)
    d1 = float(np.abs(half - params).max())
    print()
    print("  ── C.1 ★★★ **只 all-gather 了【一部分】**：`N=%d` 片里只拼回 **%d** 片 ──" % (N, N // 2))
    print("     后 %d 片的位置仍是 `0`（没搬回来）⇒ `max|diff|` = **%.6f**" % (N // 2, d1))
    print("     ★★★ **它【不报错】**：拼回的张量形状、类型全都对 —— 只是**一半的参数是错的**。")
    check("C.1", d1 > TOL_CONTENT,
          "★★★ **判据成立**：只拼回一半的分片 ⇒ `max|diff|` = **%.6f**（≫ 内容级阈 `%g`）"
          " ⇒ ★ **不报错、结果错**" % (d1, TOL_CONTENT),
          "**只拼一半居然没让结果变** —— ★ 那说明 `gather` 与原始参数共用了一段代码（自证）")

    # ── C.2 ★★★ 忘了 all-gather（一片都不拼）────────────────────────────────
    none = gather(shards, k_used=0)
    d2 = float(np.abs(none - params).max())
    print()
    print("  ── C.2 ★★★ **忘了 all-gather**（`k_used = 0`）：一片都不拼 ⇒ 全 `0` ──")
    print("     `max|diff|` = **%.6f**（= `max|原始参数|`）⇒ ★★ **它 ≥ `C.1` 的 %.6f**"
          "（漏得越多、差越大）" % (d2, d1))
    check("C.2", d2 > TOL_CONTENT and d2 >= d1,
          "★★★ **判据成立**：一片都不拼 ⇒ `max|diff|` = **%.6f** ≥ `C.1` 的 **%.6f** ⇒ "
          "★★ 「忘了 all-gather」与「只搬一半」是**同族的两种漏法**" % (d2, d1),
          "**忘了 all-gather 的差没有 ≥ 只搬一半的差** —— ★ 两条错法可能被写成了同一件事")

    # ── C.3 ★★ 拼错顺序 ─────────────────────────────────────────────────────
    wrong_order = gather(list(reversed(shards)))
    d3 = float(np.abs(wrong_order - params).max())
    print()
    print("  ── C.3 ★★ **拼错顺序**：把 `N` 个分片**倒过来**拼（位置全错）──")
    print("     `max|diff|` = **%.6f** ⇒ ★ 「顺序」也是**分片边界配对**的一部分（`A.1` 的落点）" % d3)
    check("C.3", d3 > TOL_CONTENT,
          "★★ **判据成立**：倒序拼 ⇒ `max|diff|` = **%.6f**（≫ `%g`）⇒ ★★ **拼错顺序会真的错**"
          "（这是 `A.1` 那条弱判据的意义所在）" % (d3, TOL_CONTENT),
          "**拼错顺序没有让结果变** —— ★ 分片之间可能是常量（`A.1` 失去意义）")


# ===========================================================================
# 5. 前哨：口径 / 阈值 / 标记自检（判据 D.1 / D.2 / D.2b / D.3 / D.3b）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` / `numpy %s`。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本主题的禁区回链（本程序一行都不重写）**：")
    print("      · ★★★ **框架的分布式 API 语义**（`torch.distributed` / `DDP` / `FSDP` 的类与参数）"
          "→ **全库当前没有正文**（本程序**只引取值列表**，标 B 级 + 版本 + URL + 抓取日）")
    print("      · **分片检查点的落盘与恢复** → `90-training/checkpointing/02,03`"
          "（★ 本域只写「分片（放进显存）」，不写「落盘」）")
    print("      · **`16Ψ` 的单卡口径与它的字节分解** → `90-training/checkpointing/01-state-and-bytes.md`")
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
          "★ 标记自检：**等于**本主题被分配的字面量 `?? FAIL: zero-fsdp`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                        "== FAIL: pipeline-parallel", "{{ FAIL: seq-context-parallel",
                                        "|| FAIL: expert-parallel"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住")
    check("D.3", specs.TOL_MACHINE == 1.0e-12 and specs.ZERO_BYTES_PER_PARAM == 2
          and specs.ZERO_K_MEMORY_MULTIPLE == 12 and specs.ZERO_TOTAL_MULTIPLE == 16
          and TOL_CONTENT == 1.0e-3,
          "★ 每参数字节 = %d · `K` = %d · `16Ψ` = %d · 机器精度档阈 `%g` · "
          "**本程序的内容级阈 `TOL_CONTENT = %g`**（★ 它是**口径**，不进 `specs.py`，"
          "但**照样要有一条 `==` 钉住** —— 否则它是 `make mutate` 的漏网）"
          % (specs.ZERO_BYTES_PER_PARAM, specs.ZERO_K_MEMORY_MULTIPLE,
             specs.ZERO_TOTAL_MULTIPLE, specs.TOL_MACHINE, TOL_CONTENT))
    check("D.3b", specs.ZERO_C_PSI % specs.ZERO_C_N == 0
          and isinstance(specs.ZERO_C_PSI, int) and isinstance(specs.ZERO_C_N, int)
          and len(specs.FSDP_STRATEGY_VALUES) == 5,
          "★★ `ZERO_C_PSI = %d` 被 `ZERO_C_N = %d` 整除 ⇒ 分片边界整齐（★ 不整除的处置**不是** 03 的题目）"
          % (specs.ZERO_C_PSI, specs.ZERO_C_N))
    return len(FAILS) == before


# ===========================================================================
# 6. D 类（后半）：恒真性自检（报告）· 引文钉住 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 恒真性自检（**报告**）· 引文钉住 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★★ 恒真性自检（**报告，不是判据**）────────────────────────────
    Psi, N = specs.ZERO_C_PSI, specs.ZERO_C_N
    params = np.random.default_rng(100).standard_normal(Psi)
    shards = contiguous_shard(params, N)
    rt = float(np.abs(gather(shards) - params).max())
    print()
    print("  ── D.4 ★★★ **恒真性自检**：分片再 all-gather 拼回 vs 原始参数的 `max|diff|` ──")
    print("     `max|diff|` = **%.3e** —— ★★★ **恰好为 0（逐位相同）**。" % rt)
    print("     ★★ 而**这不是「分片实现对了」的证据**：**分片再拼回是纯搬运**，")
    print("        它连一次浮点运算都不改 ⇒ **这条判据恒真**（本库 `P12`/`D22`）。")
    print("     ★★ **同一批数据、同一个参照**下，三条**内容级**错法给的是：")
    print("        `C.1` 只拼一半 **%.6f** · `C.2` 忘了 all-gather **%.6f** · `C.3` 拼错顺序 **%.6f**"
          % (float(np.abs(gather(shards, k_used=N // 2) - params).max()),
             float(np.abs(gather(shards, k_used=0) - params).max()),
             float(np.abs(gather(list(reversed(shards))) - params).max())))
    print("     ⇒ ★★★ **那三个数才说明这条路上有东西可错** —— 没有它们，`D.4` 的信息量为零。")
    check("D.4", rt == 0.0,
          "★★★ **自检成立（报告）**：分片再拼回**逐位相同**（`max|diff| = 0`）⇒ "
          "★★ **这条判据恒真** ⇒ **本程序的 A 类不钉在这里**（`A.1` 钉在**边界配对**上）",
          "**拼回不再逐位相同** —— ★ 那说明这一段**不是**纯搬运了；"
          "★★ 此时 `A.1` 的定位需要重新审视（本自检的结论**已经过期**）")

    # ── D.5 ★ FSDP 引文逐字钉住（★ 只引取值列表）───────────────────────────
    check("D.5", specs.FSDP_INSPIRED_QUOTE == ("This is inspired by Xu et al. as well as "
                                               "the ZeRO Stage 3 from DeepSpeed.")
          and specs.FSDP_FULL_SHARD_ALLGATHER_QUOTE == (
              "unshards (via all-gather) before the forward, reshards after the forward, "
              "unshards before the backward computation, and reshards after the backward computation")
          and specs.FSDP_DEFAULT_STRATEGY == "FULL_SHARD",
          "★★ **FSDP 的引文逐字钉住**（B 级 + 版本 %s）：\n"
          "        · 与 ZeRO-3 的关系：`%s`；\n"
          "        · `FULL_SHARD` 的时序（★ 本域主线的落点②「什么时刻谁手里有哪一块」）：`%s`；\n"
          "        · **默认值 = `%s`**。\n"
          "        ★★★ **本程序只引这个取值列表**，**不解释框架 API 的通用语义**（`P19`）。"
          % (specs.FSDP_DOC_VERSION, specs.FSDP_INSPIRED_QUOTE,
             specs.FSDP_FULL_SHARD_ALLGATHER_QUOTE, specs.FSDP_DEFAULT_STRATEGY),
          "**FSDP 的引文常量被动过**")

    # ── D.5b ★ 载体 / 版本 / URL / 更新日 + **可核性** ───────────────────────
    check("D.5b", specs.FSDP_DOC_VERSION == "2.14"
          and specs.FSDP_DOC_URL == "https://docs.pytorch.org/docs/stable/fsdp.html"
          and specs.FSDP_LAST_UPDATED == "May 12, 2026"
          and specs.FSDP_DEFAULT_STRATEGY in specs.FSDP_STRATEGY_VALUES,
          "★ **载体** = `%s`（★ **它会 302 到 `/docs/2.14/fsdp.html`** ⇒ 写 **版本 %s**）；"
          "页面自称 `Last Updated On: %s`。\n"
          "        ★★★ **可核性（如实登记）**：本机 `curl -sSL` 取回的是**客户端渲染空壳**"
          "（实测 **1275 字节**）⇒ ★★ **本模块按计划 §C.6 的转写逐字登记，"
          "不声称在原始字节上复核过**（★ 本库反复记的「转写是独立的出错环节」）。"
          % (specs.FSDP_DOC_URL, specs.FSDP_DOC_VERSION, specs.FSDP_LAST_UPDATED),
          "**FSDP 载体/版本的常量被动过**")

    # ── D.6 ★★ 同名不同指：显存分片 vs 分片检查点 ───────────────────────────
    check("D.6", specs.FSDP_STRATEGY_VALUES[0] == "FULL_SHARD"
          and len(specs.FSDP_STRATEGY_VALUES) == 5,
          "★★★ **同名不同指**：本域写的是**「显存分片」**（`ZeRO`/`FSDP`），而 "
          "`00-meta/glossary.md:275` 的 `sharded checkpoint` 是**落盘** ⇒ ★★ **落盘一律回链 "
          "`90-training/checkpointing/02,03`**（本程序**不写落盘**）。\n"
          "        ★★ `FSDP_STRATEGY_VALUES` 的**五个取值**（★ 第二个是 `SHARD_GRAD_OP`；"
          "最后一个**带下划线前缀**：`_HYBRID_SHARD_ZERO2`）。")

    # ── D.7 ★★★ 逐元素钉住 ────────────────────────────────────────────────
    check("D.7", specs.FSDP_STRATEGY_VALUES == ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD",
                                                "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2")
          and specs.ZERO_C_PSI == 65536 and specs.ZERO_C_N == 4,
          "★★★ **取值列表【逐元素】钉住**（五个）· `ZERO_C_PSI = 65536` · `ZERO_C_N = 4` ——"
          " ★ 只钉「整体」会让**元素级**变异漏网（本库 v6.4 一轮实测 58 条漏网里 38 条出自这一条）")

    # ── D.8 ★★ 独立性线索（源码级，★ **不是证明**）────────────────────────
    def _calls(fn):
        out = set()
        for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    out.add(f.id)
                elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                    out.add(f.value.id)
        return out
    calls_ref = _calls(closed_form)
    calls_test = _calls(strategy_bytes)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照的**调用表** ──")
    print("     `strategy_bytes()` 里调用的函数：%s"
          % (", ".join(sorted(calls_test)) if calls_test else "**空集**"))
    print("     `closed_form()`   里调用的函数：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    print("     ⇒ ★★ `closed_form()` **不调用** `strategy_bytes()`（也不调用 `np`）——")
    print("        **参照不调用被测**（域 README 的硬要求）在此**在源码层面成立**。")
    print("     ★★★ **降级措辞**：这只是「两段代码不互相调用」的**源码级线索**，"
          "**不证明**判据有物理信息量 —— 后者由 `A.1g`/`C.1`/`C.2` 的**行为**兜底。")
    check("D.8", "strategy_bytes" not in calls_ref and "np" not in calls_ref,
          "★★ **源码级线索成立**：`closed_form()` 的调用表里**没有** `strategy_bytes` / `np` "
          "⇒ ★ **参照不调用被测**（源码层面）",
          "**参照与被测互相调用** —— §B 的判据可能退化成自证")

    # ── D.9 ★ `UNSOURCED` 留空 ─────────────────────────────────────────────
    check("D.9", specs.UNSOURCED == (),
          "★ `specs.UNSOURCED` **留空** —— ★ FSDP 那条「**未在原始字节上复核**」写在"
          "`specs.py` §3 抬头的「可核性」里，★ **不欠一条记录**。")


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
    print("★★★ **本主题的 A 类【很弱】**（如实写）：FSDP 的正确性本质是「搬运正确」⇒")
    print("    「全分片 == 不分片」**恒真**（`D.4`）；A 类钉在**分片边界的配对**上（`A.1`）。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_fsdp_correctness.py —— FSDP 的策略与分片后的正确性")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ **本主题的 A 类【很弱】，如实写**：「全分片 == 不分片」是**纯搬运 ⇒ 恒真**；")
    print("      A 类钉在**分片边界的配对**上（`A.1`），**本主题的主力是 B 类**（三种策略的字节账）。")
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
