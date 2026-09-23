#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_grad_averaging.py —— DP 的**语义**：为什么梯度是**平均**不是求和。

配套文档：`01-gradient-averaging.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**，
> **不是**任何关于真实集群性能的结论。

★★★ **本主题的头号风险是「模拟器同时是被测对象」**（计划 §1.3.3）：
  本程序里的 `all_reduce` 就是 `sum()`、`dp_average_gradient` 就是几步算术
  ⇒ 「N 卡平均 == 单卡大 batch」这类判据**极易写成恒真**。
  ★ 本程序的处置有两条：
    · **每一条 A 类判据都带一个「把参照实现故意改错」的守卫**（`CONTRIBUTING:235`）——
      改错之后**该断言必须失败**（`A.1` 的 `A.1g` / `A.3` 的四格 / `A.2` 的零与非零）；
    · **参照实现不得调用被测实现的任何函数** —— 本程序给被测函数装了**调用计数器**
      （`_CALLS`），参照跑完后必须**恰好为 `0`**（`D.7b`），且**先证明这个计数器非空**
      （被测路径跑一次必须 > 0；否则那是一条恒真的「没调用」）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  > 答「不会」⇒ 它至少钉住了某段逻辑；**再问下一步**：那段逻辑**是不是就是断言自己写的那行**。
  ★ 本程序的答案落点：`A.1` 若删掉 `dp_average_gradient`（被测逻辑）⇒ `g_dp` 无从算起
    （不会通过）；`A.2`/`A.3`/`C.1`/`C.2` 同理。
  ★★ **一处对答案的反面样本（本轮已改）**：`A.3` 原来还多钉了一条
    `abs(mean(cells[0][1] / base) - 1.0) <= TOL_RATIO` —— 而 `cells[0]` 与 `base` 是
    **同参数同函数的两次调用**（`pooled_gradient(..., False, True)`）⇒ 比值**恒为 `1.0`**
    ⇒ 删掉被测逻辑它照样「成立」⇒ 已从 `cond` 移除（`ok` 那四项仍然有效）。

★★ **覆盖边界（必须写明 —— 计划 §2 的第 3 条）**：本程序的 A 类守的是
  「**切分做对了没有**」（除数在哪一边、损失口径与除数配不配对、累加顺序变了多少），
  **不守**「算子本身对不对」（那由 `numpy` 保证）⇒ **不得声称 A 类验证了更多东西**。
  ★★ **订正（2026-09-23）**：原句在这里引了 `CONTRIBUTING:238`，**引错了** ——
  `:238` 讲的是**另一件事**：「C 类断言成立时程序应当失败 ⇒ 凡断言「成功行为」的
  （如「变换前后语义等价」）归 A 类或 B 类，不要塞进 C 类」。两件事都在
  `CONTRIBUTING`/计划里有出处，但**不是同一条**（★ 这正是「引文作用域不匹配」的形态）。

★★★ **两条必须落在程序里的口径纪律**（计划 §0.4.1 与 §D.4①）：
  1. **凡报「N/M 次非零」这类计数，必须同时报三件事**：**除法的位置**、
     **累加的实现**、**被归约的轴**。`A.2` 的三行 `print` 逐条报出；
  2. **`(Σg)/N` 与 `Σ(g/N)` 的「无差别」只在 `N` 是 2 的幂时成立**（`P26`）——
     ★ 这是**取值造成的恒真**，不是结论 ⇒ `A.2` 把**非 2 的幂**那一列也跑出来。

★ **本主题的禁区**（回链，不重写）：
  · **梯度规约的算法**（ring allreduce / `V(N) = 2(N−1)/N × S`）与**扩展效率模型**
    （Amdahl / Gustafson / `N₅₀`）⇒ `110-communication` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md`；
  · **梯度累积与 batch 放大的等价性（含归一化层时的失效）**已经由
    `90-training/autograd-and-backward/03-gradient-lifecycle.md` 写过
    （`:71` 与判据表 `:223-224` —— ★ 失效条件的**实测值**与**机制**都在那边）
    ⇒ 本程序**只写「无归一化层」那一支并回链**，**不重写**那一条及其失效条件；
  · **`16Ψ` 的组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）已经由
    `90-training/checkpointing/01-state-and-bytes.md` 写过（§1.3 + 判据表 +
    `code/01_state_bytes.py:310` 的 `A.3`）⇒ 本程序只**消费**那个系数（`B.1`），不重推分解。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ `N` 卡各自算梯度再平均 == 单卡一次算大 batch 的梯度（**机器精度档**）；
             ★ **前提**：无按片统计的归一化层（有 BN 时该等价性**失效**，回链 `90` 的 `03:71`）；
  · `A.1g` ★★ **守卫**：把参照实现的输入**故意改错**（少喂最后一张卡的行）⇒ 同一条断言**必须失败**；
  · `A.2`  ★★★ `(Σg)/N` vs `Σ(g/N)`：`N` 是 **2 的幂** ⇒ **逐位相同**（计数 == `0`）；
             `N` **非** 2 的幂 ⇒ **每次试验都不同**（计数 == `TRIALS`，`D ≥ 16`）⇒ **结论不可外推**；
  · `A.3`  ★★ `loss` 的 reduction 口径与卡间除数**必须配对**：四种组合只**两种**正确，
             另两种的比值**恰好**是 `N·B` 与 `B`；
  · `B.1`  ★ 梯度的字节账 == `ZERO_GRAD_FP16_MULTIPLE × Ψ`（**两条来源不同**的路径）；
  · `B.2`  ★ 每卡持有的梯度字节**与 `N`、与每卡行数都无关**（常量 `2Ψ`）；变的是**全局 batch**；
  · `C.1`  ★★ 用**求和**代替平均 ⇒ 更新**恰好差 `N` 倍**（本机 `4.000000`）——
             ★ 而 `torch.distributed.all_reduce` 的**默认 `op` 就是 `ReduceOp.SUM`**
               （[B] PyTorch 2.14）⇒ **这个错法不是假想的**；
  · `C.2`  ★ 在 `N = 1` 上，「求和」与「平均」**逐位相同** ⇒ **用 `N=1` 测不出口径错误**
             （这是 `C.1` 那条判据的**适用边界**）。

用法：
    python3 01_grad_averaging.py
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
# ★★ 失败标记：本主题（data-parallel）取 `%% FAIL: data-parallel`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
#   ★ 它**只允许出现在 `print()` 的字符串里** ⇒ 本文件里它只出现在打印与断言明细中。
ASSERT_FAIL_MARK = "%% FAIL: data-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("^^ FAIL: tensor-parallel", "== FAIL: pipeline-parallel",
                     "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                     "?? FAIL: zero-fsdp")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★★ **被测函数的调用计数器**（`D.7b` 用）：参照实现跑完后它必须恰好为空。
_CALLS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（2026-09-23 订正，本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞（里面满是「判据成立」「必须失败」「⇒ 这证明
      `A.1` 不是恒真的」这类断言）⇒ 若把它原样打在失败行上，输出会**自相矛盾**：
        `%% FAIL: data-parallel A.1   ★★ 判据成立：N 卡平均 == 单卡大 batch（…）`
      ⇒ 所以 `cond` 为假时走**另一版文案**：默认只报「这条判据【不成立】」；
        调用方若真的想给失败版文案，用 `detail_fail=` 显式给一份。
      ★ 这是**输出层**的纪律，改动**不碰**记账（`FAILS` / `_TOTAL` / `_PASS` 三条路径照旧）。
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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立 / 必须失败 / "
                                "⇒ 不是恒真的」的措辞是**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260941              # ★ 与同域另外两个程序（…0942 / …0943）都**不同**
N_CARDS = 4                  # `A.1` 的卡数
PER_CARD_ROWS = 8            # `A.1` 每卡的行数
DIM = 64                     # 玩具模型的**参数个数** `Ψ`（= 权重向量的长度）
LR = 0.05                    # `C.1` 的玩具步长（只需要它是**确定**的）
# ── `A.2` 的取值集合（★★ 这一对「2 的幂 / 非 2 的幂」就是 `P26` 的全部内容）──
POW2_N = (2, 4, 8, 64)       # `N` 是 2 的幂 ⇒ 除法是**精确的指数偏移**
NONPOW2_N = (3, 5, 7, 10, 100)   # `N` 非 2 的幂 ⇒ 逐次试验几乎全不同
DIVORDER_DIMS = (2, 6, 16, 64)   # 维度 `D`：`D` 越小，非零比例越低
DIVORDER_TRIALS = 200        # 每个 `(N, D)` 组合的试验次数
# ── 阈值 ────────────────────────────────────────────────────────────────────
TOL_EXACT = 0.0              # 逐位相等（`A.2` 的 2 的幂那一列）
TOL_MACHINE = 1e-12          # 「机器精度档」（`A.1` 的等价性）
TOL_GUARD = 1e-6             # 守卫：改错的参照必须**远大于**它（`A.1g`）
TOL_RATIO = 1e-9             # 「恰好 `N` 倍」的容差（`A.3` / `C.1`）
BATCH_SPREAD = 8             # `B.2`：全局 batch 的跨度（最大 / 最小）
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   （`1e300 + 1.0 == 1e300`）⇒ 那样的叶子变异后**逐位不变**、任何断言都抓不到。
#   本程序的浮点常量都 `< 2^53`（最大的阈值是 `1e-6`）⇒ 每一个叶子都**可变异**。
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 十进制字面量：盲区④）


# ===========================================================================
# 2. 被测路径 vs 参照路径
# ===========================================================================
def local_gradient(x_block, y_block, rows):
    """**被测**：一块数据上的**平均**梯度（`loss` 用 `mean` 口径）—— 它**真的计入调用计数**。

    ★ 这是 DP 里「每张卡在自己那一份数据上算梯度」的可执行形态。
    """
    _CALLS["local_gradient"] += 1
    return (x_block.T @ y_block) / rows


def dp_average_gradient(x, y, n_cards, rows):
    """**被测**：`N` 卡各自算梯度，再在卡间**平均**（`(Σ g_i)/N`）。

    ★★ 这条路径**改变了累加顺序**（分块累加 vs 一次算）⇒ 它与单卡大 batch 的差
      **有信息量**（不是「纯搬运」，见计划 `D22`：本机实测 `8.882e-16` 档）。
    """
    parts = [local_gradient(x[i * rows:(i + 1) * rows], y[i * rows:(i + 1) * rows], rows)
             for i in range(n_cards)]
    return sum(parts) / n_cards


def single_card_gradient(x, y):
    """**独立参照**：**一次**在大 batch 上算平均梯度（`(Xᵀy)/rows`）。

    ★★★ **独立性**：这条路径**不调用** `local_gradient` / `dp_average_gradient` 中的任何一行
      （`D.7b` 用调用计数与 AST 调用表**两条**证据钉住它）。
      ★ 它也**不是**「把同一个值换个写法」：它是**一次矩阵乘**，而被测路径是
        **`N` 次矩阵乘 + `N−1` 次加法 + 一次除法** ⇒ 浮点路径**不同**。
    """
    return (x.T @ y) / x.shape[0]


def pooled_gradient(x, y, n_cards, rows, loss_is_sum, divide_by_cards):
    """`A.3` 的四格：`loss` 的 reduction 口径 × 卡间是否再除以 `N`。

    ★ `loss_is_sum=True` 表示每卡的 `loss` 用 **sum** reduction（不除以本卡的行数）；
      否则用 **mean**（除以 `rows`）。
    ★ `divide_by_cards=True` 表示卡间**再**除以 `N`。
    """
    parts = []
    for i in range(n_cards):
        gp = x[i * rows:(i + 1) * rows].T @ y[i * rows:(i + 1) * rows]
        if not loss_is_sum:
            gp = gp / rows
        parts.append(gp)
    g = sum(parts)
    return g / n_cards if divide_by_cards else g


def reduce_two_ways(shards, n):
    """`A.2` 的两种写法。★★ **口径写在这里，三条缺一不可**：

      · **除法的位置**：`a` 是「**先加后除**」（除以 `N` 发生在求和之后），
        `b` 是「**先除后加**」（除以 `N` 发生在每一次加法之前）；
      · **累加的实现**：两边都用 **Python 内建 `sum()`**（对整个列表**顺序累加**）——
        ★★ **不是** `np.sum()`（它在连续内轴上走 pairwise）⇒ 换实现数会变（计划 `D5b`）；
      · **被归约的轴**：被归约的是**卡（rank）轴**（`shards` 这个**列表的顺序**）；
        每个梯度向量的**元素轴**是连续的，但**元素之间不互相归约**
        （`+` 是一次向量化运算，没有顺序自由度）⇒ 这里的「顺序」完全由列表序决定。
    """
    a = sum(shards) / n
    b = sum(g / n for g in shards)
    return a, b


def count_divergence(n, d, trials, seed):
    """`A.2`：`trials` 次试验里，两种写法**逐位不同**的次数。"""
    rng = np.random.default_rng(seed)
    nz = 0
    for _ in range(trials):
        shards = [rng.standard_normal(d) for _ in range(n)]
        a, b = reduce_two_ways(shards, n)
        if not np.array_equal(a, b):
            nz += 1
    return nz


def toy_batch(rows, dim, seed):
    """给定行数/维度造一份确定的数据（★ 只用于把**形状**摆出来，不训练任何东西）。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, dim)), rng.standard_normal(rows)


# ===========================================================================
# 3. 工具
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
    """★ 运行期的同义确认：`sys.modules` 里**第三方**名字中只允许有 `numpy`。

    ★ `make lint` 用 **AST** 在**静态**上做同一件事（注释里提到框架名不算违规）；
      这一条是它在**运行期**的影子 —— 两条都留着，因为 lint 是 `make` 目标、不随程序走。
    """
    banned = {"torch", "tensorflow", "jax", "triton", "numba", "cupy", "apex",
              "deepspeed", "bitsandbytes", "transformers", "flax", "vllm"}
    return not (banned & set(sys.modules))


# ===========================================================================
# 4. A 类：平均的语义（判据 A.1 / A.1g / A.2 / A.3）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 语义：`N` 卡平均 == 单卡大 batch（**无归一化层**）· 除法的位置 · 口径配对")
    print("=" * 74)

    # ── A.1 ★★ N 卡平均 vs 单卡大 batch ────────────────────────────────────
    print("  ── A.1 ★★ `N=%d` 卡各自算梯度再平均 vs 单卡一次算 `N×B=%d` 行的梯度 ──"
          % (N_CARDS, N_CARDS * PER_CARD_ROWS))
    x, y = toy_batch(N_CARDS * PER_CARD_ROWS, DIM, SEED)
    print("     ★★ **前提（必须写出来）**：**没有按片统计的归一化层**。")
    print("        ★★ 该等价性与它的**失效条件**的正文在 "
          "`90-training/autograd-and-backward/03-gradient-lifecycle.md`（`:71` / 判据表 "
          "`:223-224`）⇒ **本程序只有「无归一化层」这一支，机制不回述**"
          "（★ 订正 2026-09-23：原来这里抄了 `90` 的机制从句与它的实测值 `4.173e-01`，"
          "违反 `CONTRIBUTING` §7.2 的「只放链接、不复制内容」）。")
    g_dp = dp_average_gradient(x, y, N_CARDS, PER_CARD_ROWS)
    g_one = single_card_gradient(x, y)
    diff = float(np.abs(g_dp - g_one).max())
    scale = float(np.abs(g_one).max())
    # ★★ 守卫：**把参照实现的输入故意改错**（少喂最后一张卡的行）⇒ A.1 必须失败
    g_bad = single_card_gradient(x[:-(PER_CARD_ROWS)], y[:-(PER_CARD_ROWS)])
    diff_bad = float(np.abs(g_dp - g_bad).max())
    print("     `max|N 卡平均 − 单卡大 batch|` = **%.3e**（`g` 的量级 %.3f）"
          % (diff, scale))
    print("     `[B] NCCL %s` 逐字：`%s` 与 `%s`"
          % (specs.NCCL_DOC_VERSION, specs.NCCL_ALLREDUCE_REDUCES_QUOTE,
             specs.NCCL_ALLREDUCE_SUM_ELEM_QUOTE))
    print("     ★★ **这不是「纯搬运」**：被测路径是 `N` 次矩阵乘 + `N−1` 次加法 + 一次除法，"
          "参照是**一次**矩阵乘 ⇒ **累加顺序真的变了**（差应落在机器精度档，不是 `0.0`）。")
    print("     ★★ **守卫**（`CONTRIBUTING:235`）：把参照的输入少喂一张卡的行 ⇒ "
          "差 = **%.3e** ⇒ 同一条断言**必须失败**" % diff_bad)
    check("A.1", diff <= TOL_MACHINE * max(scale, 1.0) and np.isfinite(diff),
          "★★ **判据成立**：`N` 卡平均 == 单卡大 batch（差 `%.3e` ≤ 机器精度档）。"
          "★ 前提：**无按片统计的归一化层**（失效条件回链 `90` 的 `03:71`）" % diff)
    check("A.1g", diff_bad > TOL_GUARD,
          "★★ **守卫成立**：故意改错的参照给出 `%.3e` > `%g` ⇒ `A.1` **能失败**"
          "（它守的是「除数在哪一边、累加顺序对不对」）" % (diff_bad, TOL_GUARD))

    # ── A.2 ★★★ (Σg)/N vs Σ(g/N)：2 的幂 vs 非 2 的幂 ─────────────────────
    print()
    print("  ── A.2 ★★★ `(Σg)/N` vs `Σ(g/N)`：**除法的位置**决定它是不是恒真 ──")
    print("     ★★ 本行的三条口径（缺一不可，`P26` 的落法）：")
    print("        · **除法的位置**：先加后除（`(Σg)/N`）vs 先除后加（`Σ(g/N)`）；")
    print("        · **累加的实现**：**Python 内建 `sum()`**（顺序累加）—— "
          "**不是** `np.sum()`（连续内轴上走 pairwise，数会变，见 `D5b`）；")
    print("        · **被归约的轴**：**卡（rank）轴**（列表顺序）；元素之间**不互相归约**。")
    rows_tbl = []
    worst_nonpow2 = 0
    for d in DIVORDER_DIMS:
        p2 = [(n, count_divergence(n, d, DIVORDER_TRIALS, SEED)) for n in POW2_N]
        n2 = [(n, count_divergence(n, d, DIVORDER_TRIALS, SEED)) for n in NONPOW2_N]
        rows_tbl.append((d, p2, n2))
        worst_nonpow2 = max(worst_nonpow2, min(c for _, c in n2))
        print("     `D=%-3d`｜**2 的幂**：" % d
              + "  ".join("N=%d→%d/%d" % (n, c, DIVORDER_TRIALS) for n, c in p2))
        print("            ｜**非 2 的幂**：" + "  ".join("N=%d→%d/%d" % (n, c, DIVORDER_TRIALS)
                                                       for n, c in n2))
    pow2_all_zero = all(c == 0 for _, p2, _ in rows_tbl for _, c in p2)
    nonpow2_all_nz = all(c > 0 for _, _, n2 in rows_tbl for _, c in n2)
    wide_dims_full = all(c == DIVORDER_TRIALS for d, _, n2 in rows_tbl if d >= 16 for _, c in n2)
    print("     ⇒ ★★ **`N` 是 2 的幂：`%d/%d` 格全部逐位相同（计数 == `0`）**；"
          "**非 2 的幂：`%d/%d` 格全部出现过非零**（其中最保守的一格是 `%d/%d`）"
          % (sum(1 for _, p2, _ in rows_tbl for _, c in p2 if c == 0),
             sum(len(p2) for _, p2, _ in rows_tbl),
             sum(1 for _, _, n2 in rows_tbl for _, c in n2 if c > 0),
             sum(len(n2) for _, _, n2 in rows_tbl), worst_nonpow2, DIVORDER_TRIALS))
    check("A.2", pow2_all_zero and nonpow2_all_nz and wide_dims_full,
          "★★★ **「2 的幂下无差别」是【取值造成的】，不是结论**：`N` 是 2 的幂时逐位相同"
          "（除法是**精确的指数偏移**），非 2 的幂时 `D ≥ 16` 的三列是 **`%d/%d` 全不同** "
          "⇒ 用 2 的幂做除法测试会得到「无差别」的**假结论**（`P26`）"
          % (DIVORDER_TRIALS, DIVORDER_TRIALS))

    # ── A.3 ★★ loss 的 reduction 口径 × 卡间除数：四格 ─────────────────────
    print()
    print("  ── A.3 ★★ `loss` 的 reduction 口径与卡间除数**必须配对**：四格只两格对 ──")
    base = pooled_gradient(x, y, N_CARDS, PER_CARD_ROWS, False, True)
    cells = [(name, pooled_gradient(x, y, N_CARDS, PER_CARD_ROWS, ls, dv))
             for name, ls, dv in (("mean + 卡间平均", False, True),
                                 ("mean + 再除以 N", False, False),
                                 ("sum + 卡间平均", True, True),
                                 ("sum + 不再除", True, False))]
    expect = {"mean + 卡间平均": 1.0, "mean + 再除以 N": float(N_CARDS),
              "sum + 卡间平均": float(PER_CARD_ROWS),
              "sum + 不再除": float(N_CARDS * PER_CARD_ROWS)}
    ok = True
    for name, g in cells:
        ratio = float(np.mean(g / base))
        ok = ok and abs(ratio - expect[name]) <= TOL_RATIO * max(abs(expect[name]), 1.0)
        print("     %-16s ⇒ 比值（相对正确解）= **%.6f**（应为 `%.0f` = %s）"
              % (name, ratio, expect[name],
                 {"mean + 卡间平均": "基准", "mean + 再除以 N": "`N`",
                  "sum + 卡间平均": "`B`", "sum + 不再除": "`N·B`"}[name]))
    print("     ⇒ ★★ **「平均」不是可选的写法**：`loss` 的 reduction 口径（除 `B` 还是不除）"
          "与卡间的除数（除 `N` 还是不除）**乘起来必须等于 `1/(N·B)`**，"
          "四格只有两格满足；另两格分别差 `N·B` 与 `B` 倍。")
    check("A.3", ok,
          "★★ **四格的比值恰好是 `{1, N, B, N·B}`**（本机 `1.000000 / %.0f / %.0f / %.0f`）"
          "⇒ 这条判据**能失败**（把口径写错，比值就离开 1）。★ 订正（2026-09-23）："
          "原来还多钉了一条 `abs(mean(cells[0][1]/base) − 1) <= TOL_RATIO` —— "
          "`cells[0]` 与 `base` 是**同参数同函数的两次调用** ⇒ 它**恒真**，已移除"
          % (N_CARDS, PER_CARD_ROWS, N_CARDS * PER_CARD_ROWS))


# ===========================================================================
# 5. B 类：代价模型 —— 梯度的字节账 `2Ψ`、每卡常量而全局 batch 变
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：每卡的梯度字节账 = `2Ψ`（与 `N`、与每卡行数都无关）")
    print("=" * 74)

    # ── B.1 ★ 梯度的字节账 == 2Ψ ───────────────────────────────────────────
    print("  ── B.1 ★ 梯度的字节账：实测 `Σ nbytes` vs `specs.py` 的 [C] 常量 ──")
    g = dp_average_gradient(*toy_batch(N_CARDS * PER_CARD_ROWS, DIM, SEED),
                            n_cards=N_CARDS, rows=PER_CARD_ROWS)
    psi = int(g.size)                       # ★ `Ψ` = **参数个数**（不是张量数、不是字节数）
    g16 = g.astype(np.float16)
    measured = int(g16.nbytes)
    closed = int(specs.ZERO_GRAD_FP16_MULTIPLE * psi)
    print("     玩具模型：`Ψ` = **%d 个参数**（`int(size)`）· 梯度按 **fp16** 计 ⇒ "
          "实测 `nbytes` = **%d B**" % (psi, measured))
    print("     `specs.ZERO_GRAD_FP16_MULTIPLE × Ψ` = `%.0f × %d` = **%d B**" %
          (specs.ZERO_GRAD_FP16_MULTIPLE, psi, closed))
    print("     ★★ `Ψ` 的口径（[C] ZeRO Figure 1 逐字）：`%s`" % specs.ZERO_FIGURE_MODEL_QUOTE)
    check("B.1", measured == closed and psi == g.size
          and measured == 2 * psi
          and specs.ZERO_FIGURE_MODEL_QUOTE == "denotes model size (number of parameters), K",
          "★★ **两条来源不同的路径给出同一个数**：左边是 NumPy 的 `size × itemsize`"
          "（由**形状与 dtype** 决定）、右边是 `specs.py` 的 **[C] 常量 × 参数个数** ⇒ "
          "**不是**恒真。`2Ψ` = `%d B`（本机）" % measured)

    # ── B.2 ★ 每卡常量、全局 batch 变 ──────────────────────────────────────
    print()
    print("  ── B.2 ★ 每卡的梯度字节与 `N`、每卡行数都无关；变的是**全局 batch** ──")
    combos = [(2, 4), (4, 8), (8, 8), (16, 4)]
    per_card_bytes, batches = [], []
    for nc, rows_ in combos:
        gg = dp_average_gradient(*toy_batch(nc * rows_, DIM, SEED), n_cards=nc, rows=rows_)
        per_card_bytes.append(int(gg.astype(np.float16).nbytes))
        batches.append(nc * rows_)
        print("     `N=%-3d` 每卡 `B=%-3d` ⇒ 全局 batch = **%-4d** 行 · "
              "每卡梯度 = **%d B**（= `2Ψ`，`Ψ=%d`）"
              % (nc, rows_, nc * rows_, per_card_bytes[-1], DIM))
    print("     ⇒ ★★ 全局 batch 从 `%d` 涨到 `%d`（**%.1f 倍**），"
          "而**每卡持有的梯度字节一个都没变**（`%.0fΨ` 与 `N`、与 `B` 都无关）"
          % (min(batches), max(batches), max(batches) / min(batches),
             specs.ZERO_GRAD_FP16_MULTIPLE))
    check("B.2", len(set(per_card_bytes)) == 1
          and per_card_bytes[0] == int(specs.ZERO_GRAD_FP16_MULTIPLE * DIM)
          and max(batches) // min(batches) == BATCH_SPREAD,
          "★★ **DP 的代价落在「每卡一份完整梯度」上**（`2Ψ` = `%d B`，**与 `N` 无关**）："
          "加卡**不减少**每卡的梯度字节，它买到的是**更大的全局 batch**（`%d → %d`）"
          % (per_card_bytes[0], min(batches), max(batches)))


# ===========================================================================
# 6. C 类：边界与反例 —— 求和代替平均、以及「N=1 看不出差别」
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：用**求和**代替平均 ⇒ 差 `N` 倍；而 `N=1` 时两者逐位相同")
    print("=" * 74)

    # ── C.1 ★★ 求和代替平均 ⇒ 更新恰好差 N 倍 ──────────────────────────────
    print("  ── C.1 ★★ 把「平均」写成「求和」：更新**恰好差 `N` 倍** ──")
    x, y = toy_batch(N_CARDS * PER_CARD_ROWS, DIM, SEED)
    g_mean = dp_average_gradient(x, y, N_CARDS, PER_CARD_ROWS)
    g_sum = sum(local_gradient(x[i * PER_CARD_ROWS:(i + 1) * PER_CARD_ROWS],
                               y[i * PER_CARD_ROWS:(i + 1) * PER_CARD_ROWS], PER_CARD_ROWS)
                for i in range(N_CARDS))
    d_mean, d_sum = -LR * g_mean, -LR * g_sum
    ratios = d_sum / d_mean
    r_min, r_max = float(np.min(ratios)), float(np.max(ratios))
    same = bool(np.allclose(d_sum, d_mean))
    print("     `[B] PyTorch 2.14` 逐字：`torch.distributed.all_reduce(tensor, op=_ReduceOp = "
          "%s, ...)`" % specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE)
    print("     ⇒ ★★ **`all_reduce` 的默认 `op` 就是【求和】** ⇒ 「平均」那一步"
          "（除以 `N`）**必须由 DP 自己补**，文档不替你做。")
    print("     更新（`w − lr·g`）：求和版 / 平均版 的逐元素比值 ∈ "
          "**[%.6f, %.6f]**（`N=%d`）· max|差| = **%.6f**"
          % (r_min, r_max, N_CARDS, float(np.abs(d_sum - d_mean).max())))
    check("C.1", abs(r_min - N_CARDS) <= TOL_RATIO * N_CARDS
          and abs(r_max - N_CARDS) <= TOL_RATIO * N_CARDS
          and not same
          and specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE == "ReduceOp.SUM",
          "★★ **用求和代替平均 ⇒ 更新恰好差 `N = %d` 倍**（本机比值 `%.6f`）；"
          "★ 断言「两者相同」**会失败**（`allclose` = `%s`）⇒ **这条判据有效**。"
          "★★ 而且这个错法**不是假想的**：[B] PyTorch 2.14 的 `all_reduce` 默认就是 `%s`"
          % (N_CARDS, r_max, same, specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE))

    # ── C.2 ★ N=1 时两者逐位相同 ⇒ 适用边界 ───────────────────────────────
    print()
    print("  ── C.2 ★ 适用边界：`N = 1` 时「求和」与「平均」**逐位相同** ──")
    x1, y1 = toy_batch(PER_CARD_ROWS, DIM, SEED)
    g1m = dp_average_gradient(x1, y1, 1, PER_CARD_ROWS)
    g1s = sum(local_gradient(x1, y1, PER_CARD_ROWS) for _ in range(1))
    same1 = bool(np.array_equal(g1m, g1s))
    print("     `N=1`：平均版 == 求和版 ⇒ **逐位相同** = **%s**（浮点也算得一样）" % same1)
    print("     ⇒ ★★ **在单卡（或 `N=1` 的仿真）上，这条口径错误【测不出来】** —— "
          "`C.1` 的判据必须以 `N ≥ 2` 为前提。")
    check("C.2", same1 and N_CARDS >= 2,
          "★★ **这是一条适用边界，不是一条「通过」**：`N=1` 时两种写法**恒等** ⇒ "
          "任何在 `N=1` 上做的口径测试都**没有信息量**（`C.1` 必须用 `N = %d`）" % N_CARDS)


# ===========================================================================
# 7. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**本机实测值**（玩具规模的真跑）："
          "`python3 %s` + `numpy %s`，macOS arm64，**无 GPU / 无多卡 / 无 NCCL**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 验证的是**切分的代数**。")
    print("  ★★ **禁区回链**（本程序一行都不重写）：梯度规约的算法（ring allreduce / "
          "`V(N) = 2(N−1)/N × S`）与扩展效率模型（Amdahl / `N₅₀`）在 `110-communication` "
          "与 `10-foundations/performance-models/02-amdahl-and-scaling.md`；"
          "梯度累积与 batch 放大的等价性（含归一化层时的失效）在 "
          "`90-training/autograd-and-backward/03-gradient-lifecycle.md:71,223-224`；"
          "`16Ψ` 的组成与分解在 `90-training/checkpointing/01-state-and-bytes.md`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == "%% FAIL: data-parallel" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("^^ FAIL: tensor-parallel", "== FAIL: pipeline-parallel",
                                        "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                                        "?? FAIL: zero-fsdp"),
          "★ 「不跨主题照抄」的**对照物**（本域另外五个主题的标记）逐位钉住；"
          "★ 六个标记**两两不互为前缀**")

    measured = {"SEED": SEED, "N_CARDS": N_CARDS, "PER_CARD_ROWS": PER_CARD_ROWS,
                "DIM": DIM, "LR": LR, "POW2_N": POW2_N, "NONPOW2_N": NONPOW2_N,
                "DIVORDER_DIMS": DIVORDER_DIMS, "DIVORDER_TRIALS": DIVORDER_TRIALS,
                "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT}
    expect = {"SEED": 20260941, "N_CARDS": 4, "PER_CARD_ROWS": 8, "DIM": 64, "LR": 0.05,
              "POW2_N": (2, 4, 8, 64), "NONPOW2_N": (3, 5, 7, 10, 100),
              "DIVORDER_DIMS": (2, 6, 16, 64), "DIVORDER_TRIALS": 200,
              "FLOAT_EXACT_LIMIT": 9007199254740992}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", not (set(POW2_N) & set(NONPOW2_N))
          and all((n & (n - 1)) == 0 for n in POW2_N)
          and all((n & (n - 1)) != 0 for n in NONPOW2_N),
          "★★ `A.2` 的两列**必须真的分开**：`POW2_N` 全部满足 `n & (n−1) == 0`、"
          "`NONPOW2_N` 全部不满足（★ 这是 `P26` 的**机械**落法：拿 2 的幂做除法测试 = 假结论）")
    check("D.3c", DIVORDER_TRIALS >= 200 and max(DIVORDER_DIMS) >= 16,
          "★ `A.2` 的试验次数 ≥ `200`、维度必须够大（`D ≥ 16` 时非零比例拉满）——"
          "★ 这是「判据的小 n 稳定性」的反面：**小 n 会让计数看起来「无差别」**")

    th = {"TOL_EXACT": TOL_EXACT, "TOL_MACHINE": TOL_MACHINE, "TOL_GUARD": TOL_GUARD,
          "TOL_RATIO": TOL_RATIO, "BATCH_SPREAD": BATCH_SPREAD}
    th_expect = {"TOL_EXACT": 0.0, "TOL_MACHINE": 1e-12, "TOL_GUARD": 1e-6,
                 "TOL_RATIO": 1e-9, "BATCH_SPREAD": 8}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 5 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < FLOAT_EXACT_LIMIT
          and FLOAT_EXACT_LIMIT == 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点阈值都 `< 2^53`（最大的是 `%g`）⇒ "
          "`mutate.py` 的 `v + 1.0` **能**改变它们（否则那类叶子**免检**）"
          % max(th_expect.values()))
    check("D.4c", no_framework_imported(),
          "★ 运行期确认：`sys.modules` 里的第三方名字中只有 `numpy`"
          "（静态那一半在 `make lint` 的 AST 检查里）")
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", (specs.ZERO_K_OPT_STATE, specs.ZERO_PARAM_FP16_MULTIPLE,
                  specs.ZERO_GRAD_FP16_MULTIPLE, specs.ZERO_TOTAL_MULTIPLE)
          == (12, 2.0, 2.0, 16.0)
          and specs.ZERO_ARXIV_ID == "1910.02054"
          and specs.ZERO_TITLE
          == "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
          and specs.ZERO_FIGURE_MODEL_QUOTE
          == "denotes model size (number of parameters), K"
          and specs.ZERO_K12_QUOTE == "Mixed-precision Adam has K = 12"
          and specs.ZERO_TOTAL_QUOTE == "In total, this results in 2"
          and specs.ZERO_TOTAL_TAIL_QUOTE == "bytes of memory requirement",
          "★★ ZeRO 的**七项**逐位一致（[C] `%s`，**不确认同行评审**）——"
          "★ `ZERO_GRAD_FP16_MULTIPLE` 与 `ZERO_PARAM_FP16_MULTIPLE` **同值但不同项**，"
          "分开存放、**不得合并**；★ `Ψ` 的口径由 `ZERO_FIGURE_MODEL_QUOTE` 钉住" %
          specs.ZERO_ARXIV_ID)
    check("D.5b", specs.ZERO_FIGURE_CALIBER_QUOTE
          == "based on mixed-precision training with Adam optimizer."
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径",
          "★★ `K = 12` 的**口径**（mixed-precision + Adam 这一组假设）与那句中文限定语"
          "**成对存放** ⇒ 正文里说 `16Ψ` / `12Ψ` 时必须带上它（`P1` 的可执行落法）")
    check("D.6", specs.NCCL_DOC_VERSION == "2.31.2"
          and specs.NCCL_USAGE_COLLECTIVES_URL
          == "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html"
          and specs.NCCL_PAGE_BYTES == 86584
          and specs.NCCL_PAGE_SHA256_16 == "d7d2b6cff49d9f79"
          and specs.NCCL_ALLREDUCE_REDUCES_QUOTE
          == "The AllReduce operation performs reductions on data (for example, sum, min, max) "
             "across devices and stores the result in the receive buffer of every rank."
          and specs.NCCL_ALLREDUCE_SUM_ELEM_QUOTE
          == "each rank will provide an array in of N values, and receive identical results "
             "in array out of N values,"
          and specs.TORCH_DIST_DOC_URL
          == "https://docs.pytorch.org/docs/2.14/distributed.html"
          and specs.TORCH_DIST_PAGE_BYTES == 1509715
          and specs.TORCH_DIST_PAGE_SHA256_16 == "1a4a7e398b5a7744"
          and specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE == "ReduceOp.SUM"
          and specs.TORCH_ALLREDUCE_REDUCES_QUOTE
          == "Reduces the tensor data across all machines in a way that all get the final result.",
          "★★ [B] 两份产品文档的**版本 + URL + 字节数 + 摘要 + 逐字引文**逐位一致"
          "（NCCL `%s`（精确到补丁号）/ PyTorch 2.14）—— ★★ 其中 `%s` 是**求和口径的默认值**，"
          "它**不是 NCCL 的**（NCCL 的文档没有默认值 ⇒ `UNSOURCED`：`nccl_allreduce_default_op`）"
          % (specs.NCCL_DOC_VERSION, specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE))

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(local_gradient)
    src_b = inspect.getsource(single_card_gradient)
    src_c = inspect.getsource(dp_average_gradient)
    check("D.7", len({src_a, src_b, src_c}) == 3,
          "★ 防 `P17`：被测（`local_gradient` %d 字符）、**独立参照**"
          "（`single_card_gradient` %d）、被测的组合（`dp_average_gradient` %d）"
          "**源码两两不同**" % (len(src_a), len(src_b), len(src_c)))
    watched = {"local_gradient", "dp_average_gradient"}
    called = _callee_names(single_card_gradient)
    check("D.7b", not (watched & called),
          "★ **AST 调用表**：**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    # ★★ **调用计数器**：先证明它非空（被测路径跑一次必须 > 0），再证明参照跑完为 0。
    _CALLS.clear()
    dp_average_gradient(*toy_batch(N_CARDS * PER_CARD_ROWS, DIM, SEED),
                        n_cards=N_CARDS, rows=PER_CARD_ROWS)
    seen = sum(_CALLS.values())
    _CALLS.clear()
    single_card_gradient(*toy_batch(N_CARDS * PER_CARD_ROWS, DIM, SEED))
    left = sum(_CALLS.values())
    check("D.7c", seen > 0 and left == 0,
          "★★ **调用计数器两向都量**：被测路径跑一次 ⇒ 计数 = **%d**（> 0，计数器**非空**）；"
          "**参照路径**跑完 ⇒ 计数 = **%d**（必须**恰好为 0**）⇒ "
          "「参照不调用被测」不是靠读代码相信的" % (seen, left))
    check("D.8", specs.UNSOURCED == ("nccl_allreduce_default_op",
                                     "data_parallel_redundancy_ratio_definition"),
          "★★ `specs.UNSOURCED` 的 **2** 项逐位钉住 —— ★ 第 1 项是本轮**真的撞到**的"
          "来源错配（NCCL 的文档**没有** `op` 的默认值）。★★ 订正（2026-09-23）："
          "原来的第 3 项 `data_parallel_ring_allreduce_volume`（`V(N)`）**已移出** —— "
          "它的性质是「**不在本主题**」而不是「找不到来源」（它有来源），"
          "留着它就违反 `CONTRIBUTING:65` 的「不得既保留数字又记 `UNSOURCED`」")
    check("D.9", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))


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
    print("01_grad_averaging.py —— DP 的语义：为什么是**平均**不是求和")
    print("=" * 74)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 头号风险是「模拟器同时是被测对象」：本程序里的 `all_reduce` 就是 `sum()` "
          "⇒ 每条 A 类都配一个「把参照故意改错」的**守卫**。")
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
