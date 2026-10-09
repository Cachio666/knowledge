#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_grad_sync.py —— DP 的**工程**：梯度怎么与参数**对齐**（**梯度分桶**）。

配套文档：`02-gradient-synchronization.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**。

★★★ **本程序的核心判据是「梯度分桶的划分必须【完备且不重叠】」**，而它**不是**一条
  「桶划分后的结果 == 单桶的结果」—— 那一条**实测三种划分（`1` 桶 / `8` 桶等分 / 不等分 `3` 桶）
  与「不再分桶」的结果全部**逐位相同**（`np.array_equal(...) == True`）**
  （纯搬运 ⇒ **恒真**，计划 `D22`）⇒ **不能当 A 类**。
  ★ 本程序的落法有两条：
    · **A 类**钉在「**与单卡大 batch 一致**」上（机器精度档）—— ★★ 这是 `A.1` **唯一**的判据：
      漏桶会让它**变大** ⇒ 它**能失败**；
    · ★★ **反向那一半才是信息量所在**（`D22` 的教训）：**漏一个桶**（有未覆盖区）或
      **重叠一个桶**（那部分变 `N` 倍），判据**必须失败** —— 这就是 `CONTRIBUTING:235`
      的守卫，`A.3` 逐条量它。
  ★★★ **本轮订正的一处【恒真当成判据】**：跨 rank 极差（`cross_rank_ptp`）在 `A.1`/`A.2`/`A.3`
    **都**被写进了 `cond`，而它**恒为 `0.0`** —— 理由**不是**「划分完备」，而是
    `bucket_sync` 的每个桶都是 `out[:, lo:hi] = Σ_rank out[:, lo:hi]`：
    **只要某元素被【至少一个桶】覆盖，各 rank 在该元素上算的就是同一个表达式**。
    实测：`PLAN_OVERLAP`（**错误**的划分）的极差**也是 `0.0`**，只有 `PLAN_MISS`
    （**有未覆盖区**）才 `> 0`。⇒ ★ 极差的**全部**信息量 = 「有没有未覆盖区」，
    与「重叠」「划分对不对」都无关 ⇒ 现在它**只被打印**（是 `C.1` 那条判据的量），
    **不再进任何 `cond`**。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  > 答「不会」⇒ 它至少钉住了某段逻辑；**再问下一步**：那段逻辑**是不是就是断言自己写的那行**。
  ★ 本程序的答案落点：`A.1` 若删掉 `bucket_sync`（被测逻辑）⇒ `d_full`/`d_eq8` 无从算起
    （不会通过）；★ 但**极差那两项**删掉 `bucket_sync` 也照样「成立」——
    因为它量的是**模拟器自己的广播语义**（本轮已移出判据）。
  ★ 同理：`A.1` 原来那个 `and same_as_unbucketed` 就是自查问句的**反面样本** ——
    `sync_eq8` 与 `sync_full` 都是 `bucket_sync` 的输出 ⇒ 那一项**答「会」**
    ⇒ 本轮已把它从 `cond` 里移到**只打印**（与文档的说法一致；见 `A.1` 的注释）。

★★ **一处必须写明的口径**（本程序报的每一个「极差 / 比值」都挂在这三件事上）：
  · **累加的实现**：`allreduce_sum()` 用 **Python 内建 `sum()`**（对 rank 列表**顺序累加**）；
  · **被归约的轴**：**rank 轴**（列表顺序）—— 桶内的**参数轴**只被**切片**，不被归约；
  · **除法的位置**：本程序取 **`SUM` 之后统一除以 `N`** 这一支（不是每个桶各自除）
    —— 这正是 [B] NCCL 的 AllReduce **求和**语义（`ReduceOp.SUM`）的落法，`C.2` 量它的后果。
  ★★ 而「除法的位置」在本程序里有一个**已知的 1 ulp 级依赖**（`(Σg)/N` vs `Σ(g/N)`）
    —— 那一条的正文在 `01_grad_averaging.md` 的 `A.2`（`P26`），本程序**不重写**。

★ **本主题的禁区**（回链，不重写）：
  · **梯度规约的算法**（ring / tree allreduce、`V(N) = 2(N−1)/N × S`、reduce-scatter 与
    all-gather 的阶段划分）⇒ `110-communication` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md`；
  · ★★★ **「通信能藏多少」不给判据**（计划 §1.3.4）：那是**通信-计算重叠**，
    在单进程形态下**没有可跑的判据**（写下去就是「要求实现者复现跑不了的东西」= `P3`）
    ⇒ 本程序**只写接口**（桶的作用是让通信能**尽早发起**），回链 `110-communication` 的 `overlap`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ 划分**完备且不重叠** ⇒ 结果与**单卡大 batch** 的梯度一致（机器精度档）；
             ★★ **只有这一条进 `cond`**。另有两条**恒真**的只打印、不当判据：
             「与**不再分桶**的结果逐位相同」（纯搬运）与「跨 rank 极差 == `0.0`」
             （**覆盖的产物**，见头部订正）；
  · `A.2`  ★ **不等分但完备**的划分（`[100,60,32]`）**也通过** —— 桶**不必等大**；
  · `A.3`  ★★★ **守卫**：同一个判据在**漏一个桶**与**重叠一个桶**的划分上**必须失败**；
  · `B.1`  ★ 桶的峰值缓冲 == `最大桶的元素数 × itemsize`（本机 `192` 元素下
             `1/8/48` 桶 ⇒ `768/96/16 B`）；
  · `C.1`  ★★ **漏桶** ⇒ 那部分梯度**不参与归约** ⇒ 各卡**发散**（跨 rank 极差不为 0）；
  · `C.2`  ★★ **重叠** ⇒ 那部分被**重复计数** ⇒ 在求和口径下变成 **`N` 倍**（本机 `8.000000`）；
  · `C.3`  ★ **错一个下标**（`[0:96]` 与 `[95:192]`）⇒ **恰好一个元素**被重复计数。

用法：
    python3 02_grad_sync.py
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
ASSERT_FAIL_MARK = "%% FAIL: data-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("^^ FAIL: tensor-parallel", "== FAIL: pipeline-parallel",
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

    ★★★ **失败路径【不复述结论】**（2026-09-23 订正，本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞（里面满是「判据成立」「必须失败」「⇒ 这证明
      `A.1` 不是恒真的」这类断言）⇒ 若把它原样打在失败行上，输出会**自相矛盾**：
        `%% FAIL: data-parallel A.3   ★★★ 守卫成立：同一个判据在坏划分上真的会失败`
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
SEED = 20260942              # ★ 与同域另外两个程序（…0941 / …0943）都**不同**
N_SYNC = 8                   # rank 数（= 副本数）
PER_RANK_ROWS = 8            # 每个 rank 的数据行数
DIM = 192                    # 梯度向量的长度（= 参数个数；★ 与任务书的 `192` 元素同口径）
LR = 0.05                    # `C` 类里量「更新差几倍」用的步长
# ── 五种**梯度分桶**的划分（★★ 每一个都是**真的会参与断言**的常量）──────────
# ★★ 判据是「划分必须**完备且不重叠**」⇒ 下面每一个划分的**每一个端点**都在断言的作用范围内：
#    改掉任一端点都会让 `A.1`/`A.2`/`A.3`/`C.1`/`C.2`/`C.3` 中的某一条**失败**。
PLAN_FULL = ((0, 192),)                                       # 1 桶：完备且不重叠
PLAN_EQ8 = ((0, 24), (24, 48), (48, 72), (72, 96),
            (96, 120), (120, 144), (144, 168), (168, 192))    # 8 桶**等分**：完备且不重叠
PLAN_UNEQ = ((0, 100), (100, 160), (160, 192))                # 3 桶**不等分**：完备且不重叠
PLAN_MISS = ((0, 128),)                                       # 1 桶，**漏掉后 64 个元素**
PLAN_OVERLAP = ((0, 96), (95, 192))                           # 2 桶，第 95 个元素**被覆盖两次**
# ── `B.1` 的桶数（★ 三者都能整除 `DIM = 192`）───────────────────────────────
BUCKET_COUNTS = (1, 8, 48)
PEAK_BYTES_EXPECT = (768, 96, 16)     # `192` 元素 **fp32** 下的峰值缓冲（实测值）
# ── 阈值 ────────────────────────────────────────────────────────────────────
TOL_EXACT = 0.0              # 跨 rank 极差必须是**逐位** `0.0`
TOL_REL = 1e-6               # 「与单卡大 batch 一致」的相对档（**fp32** 的机器精度档）
TOL_GUARD = 1e-3             # 守卫：改坏的划分必须**远大于**它
TOL_RATIO = 1e-6             # 「恰好 `N` 倍」「恰好 1 个元素」的容差
DUP_COUNT = 1                # `C.3`：错一个下标 ⇒ **恰好** 1 个元素被重复计数
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   （`1e300 + 1.0 == 1e300`）⇒ 那样的叶子变异后**逐位不变**、任何断言都抓不到。
#   本程序的浮点常量都 `< 2^53`（最大的阈值是 `1e-3`）⇒ 每一个叶子都**可变异**。
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 十进制字面量：盲区④）


# ===========================================================================
# 2. 被测路径 vs 参照路径
# ===========================================================================
def allreduce_sum(parts):
    """**被测**：把各 rank 的局部张量做一次 `all_reduce`（`SUM` 口径）—— 它**计入调用计数**。

    ★★★ 这就是本域的模拟器立场（域 README 的「代码约定」表）：**通信被实现为一个显式的
      张量搬运**，`all_reduce(parts) = Σ parts` —— 依据是 [B] NCCL 的逐字语义
      （`specs.NCCL_ALLREDUCE_SUM_ELEM_QUOTE`）与 [B] PyTorch 的默认 `op = ReduceOp.SUM`。
      ★★ **本函数是「被测对象」**：本程序量的正是「用它按桶搬运时，划分对不对」。
    """
    _CALLS["allreduce_sum"] += 1
    return sum(parts)                       # ★ 内建 sum：对 rank 列表**顺序累加**


def bucket_sync(local, buckets, n_ranks):
    """**被测**：按 `buckets` 逐桶做 `all_reduce(SUM)`，最后统一除以 `N`。

    ★★ 语义（三件事必须一起读）：
      · **缓冲区的初值 = 本地梯度**，每个桶把它**改写成** `Σ_rank` —— 这是
        「**in-place** all-reduce 一个桶」的模型（桶之间**依次**发生）；
      · **除法的位置**：所有桶做完之后**统一**除以 `N`（不是每桶各自除）；
      · ★★★ 正因为桶是**依次**发生的，**重叠的桶会把已经归约过的值再归约一次**
        ⇒ 那部分变成 `N × 均值`（`C.2` 量它）。**漏掉的桶**则**保持本地值**（`C.1` 量它）。
    """
    _CALLS["bucket_sync"] += 1
    out = local.copy()
    for lo, hi in buckets:
        out[:, lo:hi] = allreduce_sum([out[r, lo:hi] for r in range(n_ranks)])
    return out / n_ranks


def single_card_gradient(x, y):
    """**独立参照**：**一次**在大 batch 上算平均梯度（`(Xᵀy)/rows`）。

    ★★★ **独立性**：它**不调用** `allreduce_sum` / `bucket_sync` 中的任何一行
      （`D.7b` 用调用计数与 AST 调用表**两条**证据钉住它）。
      ★ 它也不是「换个写法」：它是**一次矩阵乘**，而被测路径是
        `N` 次矩阵乘 + `N` 次桶搬运 + `N−1` 次减法/加法 ⇒ **浮点路径不同**。
    """
    return (x.T @ y) / x.shape[0]


def even_plan(n_buckets, dim):
    """把 `dim` 个参数**等分**成 `n_buckets` 个桶（`B.1` 用；★ 只做形状，不参与 A/C 类判据）。"""
    step = dim // n_buckets
    return tuple((i * step, (i + 1) * step) for i in range(n_buckets))


def peak_buffer_bytes(buckets, itemsize):
    """**峰值缓冲**：一次搬运**最大的那个桶** ⇒ 缓冲 = `max(桶大小) × itemsize`。"""
    return max(hi - lo for lo, hi in buckets) * itemsize


def cross_rank_ptp(synced):
    """跨 rank 极差（`max − min`，逐元素）—— ★★ 本程序的**完备性量**。

    ★★★ **为什么不用 `np.std`**（本机实测的口径坑）：当所有 rank 的值**逐位相同**时，
      `np.std(..., axis=0)` 仍会因 `mean` 那一步的舍入给出一个**极小的非零值**
      ⇒ 「已同步」与「未同步」的分界就**不再是 `0` 与非 `0`**，判据要多引一个容差。
      **极差在「全同」时【恰好为 `0.0`】** ⇒ 分界干净。
      ★ 这是一处**量法口径**：同一个现象（各卡是否一致），两种统计量给出不同的数
      （★ 本例的实测值在 `D.8` 里**当场量出来**，不写死在注释里）。
    """
    return float(np.ptp(synced, axis=0).max())


def toy_gradients(n_ranks, rows, dim, seed):
    """造一份**能真的对应到「单卡大 batch」**的数据，并返回 `(全局数据, 各 rank 的局部梯度)`。

    ★ 局部梯度取 **`mean` 口径**（除以本卡的行数）——与 `01` 的 `A.3` 配对表一致。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_ranks * rows, dim)).astype(np.float32)
    y = rng.standard_normal(n_ranks * rows).astype(np.float32)
    local = np.stack([(x[i * rows:(i + 1) * rows].T @ y[i * rows:(i + 1) * rows]) / rows
                      for i in range(n_ranks)])
    return x, y, local


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
    """★ 运行期的同义确认：`sys.modules` 里**第三方**名字中只允许有 `numpy`。"""
    banned = {"torch", "tensorflow", "jax", "triton", "numba", "cupy", "apex",
              "deepspeed", "bitsandbytes", "transformers", "flax", "vllm"}
    return not (banned & set(sys.modules))


# ===========================================================================
# 4. A 类：桶划分的**完备性**（判据 A.1 / A.2 / A.3）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 完备性：**桶的划分必须【完备且不重叠】**")
    print("=" * 74)
    x, y, local = toy_gradients(N_SYNC, PER_RANK_ROWS, DIM, SEED)
    g_one = single_card_gradient(x, y)                 # 独立参照：单卡大 batch
    scale = float(np.abs(g_one).max())
    print("  玩具规模：`N=%d` 个 rank、每个 `B=%d` 行、`Ψ=%d` 个参数（**fp32**）"
          % (N_SYNC, PER_RANK_ROWS, DIM))
    print("  ★★ 口径：**累加实现** = Python 内建 `sum()`（对 rank 列表**顺序累加**）；"
          "**被归约的轴** = **rank 轴**（桶内的参数轴只被切片）；")
    print("     **除法的位置** = 全部桶做完后**统一除以 `N`**（[B] NCCL 的 `SUM` 口径）。")

    # ── A.1 ★★ 完备且不重叠 ⇒ 极差 0 + 与单卡大 batch 一致 ─────────────────
    print()
    print("  ── A.1 ★★ 划分**完备且不重叠**（1 桶 / 8 桶等分）──")
    sync_full = bucket_sync(local, PLAN_FULL, N_SYNC)
    sync_eq8 = bucket_sync(local, PLAN_EQ8, N_SYNC)
    ptp_full, ptp_eq8 = cross_rank_ptp(sync_full), cross_rank_ptp(sync_eq8)
    d_full = float(np.abs(sync_full[0] - g_one).max())
    d_eq8 = float(np.abs(sync_eq8[0] - g_one).max())
    # ★★ 「与不再分桶的结果」逐位相同 —— **这一半是纯搬运 ⇒ 恒真**（D22 的教训），
    #    本程序把它**写出来**当作反面教材，**不当判据**。
    same_as_unbucketed = bool(np.array_equal(sync_eq8, sync_full))
    print("     1 桶：跨 rank 极差 = **%.3e** · `max|− 单卡大 batch|` = **%.3e**"
          % (ptp_full, d_full))
    print("     8 桶等分：跨 rank 极差 = **%.3e** · `max|− 单卡大 batch|` = **%.3e**"
          % (ptp_eq8, d_eq8))
    print("     ★★ 口径提醒：本程序用**跨 rank 极差**（`max−min`）而不是 `np.std` —— "
          "后者在「全同」时仍给 **%.3e**（`mean` 那一步的舍入），"
          % float(np.std(sync_full, axis=0).max()))
    print("        会把「已同步」与「未同步」的分界从 `0`/非 `0` 变成一个带容差的比较。")
    print("     ★★★ 而「**与不再分桶的结果**逐位相同」= **%s** ⇒ **那一半是纯搬运、恒真**，"
          "**不能当判据**（计划 `D22`）" % same_as_unbucketed)
    # ★★ 两个**当场量出来**的对照（不写成注释里的断言）：坏划分的极差与有未覆盖区的极差
    ptp_ovl_probe = cross_rank_ptp(bucket_sync(local, PLAN_OVERLAP, N_SYNC))
    ptp_miss_probe = cross_rank_ptp(bucket_sync(local, PLAN_MISS, N_SYNC))
    print("     ★★★ **【本轮订正】跨 rank 极差也是恒真的 —— 但它恒真的理由【不是】"
          "「划分完备」**：")
    print("        `bucket_sync` 的每个桶都是 `out[:, lo:hi] = Σ_rank out[:, lo:hi]` ⇒")
    print("        **只要某个元素被【至少一个桶覆盖】，各 rank 在该元素上算的就是【同一个"
          "表达式】** ⇒ 逐元素极差**恒为 `0.0`**。")
    print("        ★★ **实测**：`PLAN_OVERLAP`（第 `95` 个元素被覆盖**两次**、错误的划分）"
          "的极差**同样是 `%.1f`** —— 而 `PLAN_MISS`（有未覆盖区）才给出 `%.3e`。"
          % (ptp_ovl_probe, ptp_miss_probe))
    print("        ⇒ ★ `ptp` 的**全部**信息量 = 「**有没有未覆盖区**」，"
          "与「重叠」「划分对不对」**都无关**；")
    print("          **它不是「完备性」的证据，也不是 `A.1` 的判据**（本程序只把它**打印**"
          "出来，它是 `C.1` 那条判据的量）。")
    print("        ★★ 订正前 `A.1`/`A.2`/`A.3` 都把 `ptp == TOL_EXACT` 写进了 `cond`"
          " —— 那是把**恒真**当判据（`P12`），且失败行还会打出「判据成立」⇒ 本轮一并改了。")
    check("A.1", d_full <= TOL_REL * max(scale, 1.0) and d_eq8 <= TOL_REL * max(scale, 1.0),
          "★★ **完备且不重叠 ⇒ 结果与【单卡大 batch】一致**（`%.3e` / `%.3e`，`g` 量级 %.3f）"
          "—— ★★ **这是 `A.1` 唯一的判据**（`d_full`/`d_eq8` 会因漏桶而变大 ⇒ **能失败**）。"
          "★ 另两半（跨 rank 极差 `%.1f`、与「不再分桶」逐位相同）**都是恒真的**，"
          "只作打印（见上面两段反面教材）" % (d_full, d_eq8, scale, ptp_full))

    # ── A.2 ★ 不等分但完备也通过 ───────────────────────────────────────────
    print()
    print("  ── A.2 ★ 桶**不必等大**：`[100, 60, 32]` 也是完备且不重叠 ──")
    sync_uneq = bucket_sync(local, PLAN_UNEQ, N_SYNC)
    ptp_uneq = cross_rank_ptp(sync_uneq)
    d_uneq = float(np.abs(sync_uneq[0] - g_one).max())
    sizes = [hi - lo for lo, hi in PLAN_UNEQ]
    same_uneq = bool(np.array_equal(sync_uneq, bucket_sync(local, PLAN_FULL, N_SYNC)))
    print("     桶大小 = **%s**（`Σ = %d` = `Ψ`）· 跨 rank 极差 = **%.3e** · "
          "`max|− 单卡大 batch|` = **%.3e**" % (sizes, sum(sizes), ptp_uneq, d_uneq))
    print("     ★★ 与**不再分桶**的结果逐位相同 = **%s**（★ 三种划分下都是 —— "
          "这就是「那一半恒真」的第三种证据；**只作打印**）" % same_uneq)
    print("     ★★ 跨 rank 极差 = **%.1f** —— ★ 它**同样是恒真的**（见 `A.1` 那段订正），"
          "**不进 `A.2` 的判据**" % ptp_uneq)
    print("     ⇒ ★★ 判据只要求**完备且不重叠**，**不要求等大** ⇒ 分桶是**实现自由**，"
          "不是数学约束（这一条正是 `B.1` 的立足点：可以为了峰值缓冲去改桶大小）。")
    check("A.2", d_uneq <= TOL_REL * max(scale, 1.0)
          and sum(sizes) == DIM and len(set(sizes)) > 1,
          "★★ **不等分但完备的划分同样通过**（`%s`，与单卡大 batch 差 `%.3e`）⇒ 桶的**大小**"
          "不是判据，**覆盖**才是。★ 订正（2026-09-23）：原来这里写着 `and same_uneq and "
          "same_uneq`（同一个恒真条件**写了两遍**）与 `ptp_uneq == TOL_EXACT`（恒真）"
          "⇒ 两条都已移出判据，只留打印" % (sizes, d_uneq))

    # ── A.3 ★★★ 守卫：漏一个桶 / 重叠一个桶 ⇒ 同一判据必须失败 ─────────────
    print()
    print("  ── A.3 ★★★ 守卫（`CONTRIBUTING:235`）：把划分改坏，`A.1` 的那条判据**必须失败** ──")
    sync_miss = bucket_sync(local, PLAN_MISS, N_SYNC)
    sync_ovl = bucket_sync(local, PLAN_OVERLAP, N_SYNC)
    ptp_miss, ptp_ovl = cross_rank_ptp(sync_miss), cross_rank_ptp(sync_ovl)
    ratio_ovl = float(np.mean(sync_ovl[0, 95] / g_one[95]))
    miss_lo = max(hi for _, hi in PLAN_MISS)
    print("     · **漏一个桶**（只同步前 `%d/%d` 个元素）：覆盖区极差 `%.3e`、"
          "**漏掉区极差 `%.3e`**" % (miss_lo, DIM, cross_rank_ptp(sync_miss[:, :miss_lo]), ptp_miss))
    print("     · **重叠一个桶**（`[0:96]` 与 `[95:192]`）：第 95 个元素被归约两次 ⇒ "
          "该元素 = **%.6f 倍**均值" % ratio_ovl)
    print("       ★★ 而它的跨 rank 极差 = **%.1f**（**不是**判据 —— 见 `A.1` 那段订正："
          "覆盖到的元素各 rank 算的是同一个表达式 ⇒ 极差恒 `0.0`）" % ptp_ovl)
    check("A.3", ptp_miss > TOL_GUARD and abs(ratio_ovl - N_SYNC) <= TOL_RATIO * N_SYNC,
          "★★★ **守卫成立**：同一个判据在坏划分上**真的会失败** —— 漏桶 ⇒ **有未覆盖区**、"
          "极差 `%.3e` > `%g`；重叠 ⇒ 那部分变成 `N` 倍（`%.6f`）。★ 这证明 `A.1` 不是恒真的"
          % (ptp_miss, TOL_GUARD, ratio_ovl))


# ===========================================================================
# 5. B 类：代价模型 —— 桶的峰值缓冲（**只算字节，不算时间**）
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：桶的**峰值缓冲** ∝ 最大桶；桶数 = 通信**启动次数**")
    print("=" * 74)
    _, _, local = toy_gradients(N_SYNC, PER_RANK_ROWS, DIM, SEED)
    itemsize = int(local.dtype.itemsize)
    print("  dtype = `%s`（`itemsize = %d B`）· `Ψ = %d` ⇒ 单桶缓冲 = `%d × %d` = **%d B**"
          % (local.dtype, itemsize, DIM, DIM, itemsize, DIM * itemsize))

    # ── B.1 ★ 峰值缓冲 ∝ 最大桶的元素数 ────────────────────────────────────
    print()
    print("  ── B.1 ★ 峰值缓冲（**一次搬运最大的那个桶**）──")
    peak = []
    for k in BUCKET_COUNTS:
        plan = even_plan(k, DIM)
        pb = peak_buffer_bytes(plan, itemsize)
        peak.append(pb)
        print("     `%d` 桶（每桶 `%d` 个元素）⇒ 峰值缓冲 = **%d B**"
              % (k, DIM // k, pb))
    print("     ⇒ ★★ 桶数从 `1` 到 `48`（**48 倍**），峰值缓冲从 `%d B` 降到 `%d B`"
          "（**1/%d**）—— **两者恰好成反比**：`峰值 × 桶数 == Ψ × itemsize`"
          % (peak[0], peak[-1], BUCKET_COUNTS[-1]))
    check("B.1", tuple(peak) == PEAK_BYTES_EXPECT
          and all(pb * k == DIM * itemsize for pb, k in zip(peak, BUCKET_COUNTS))
          and peak[0] * BUCKET_COUNTS[0] == peak[-1] * BUCKET_COUNTS[-1],
          "★★ **峰值缓冲 = `最大桶元素数 × itemsize`**（本机 `%s` B）⇒ 桶越小、"
          "峰值越低；两条路径同给一个数（`峰值 × 桶数 == Ψ × itemsize` 是**整数恒等**）"
          % (list(peak),))

    # ── B.2 ★ 桶数 = 通信启动次数（**只写接口，不给判据**）──────────────────
    print()
    print("  ── B.2 ★ 桶数 = 通信的**启动次数**（★ 接口，回链，**本程序不给判据**）──")
    print("     ★★★ **为什么不给判据**（计划 §1.3.4）：桶的意义是让「某段梯度就绪就发」"
          "—— 那是**通信-计算重叠**，")
    print("        在**单进程**形态下**没有可跑的判据**（写成判据就是「要求实现者复现跑不了的东西」= `P3`）")
    print("        ⇒ 本程序**只写接口**：桶数 `k` ⇒ 通信**启动 `k` 次**、每次搬 `Ψ/k` 个元素；")
    print("        ★ **能藏多少** 的正文在 `110-communication` 的 `overlap` 主题，此处回链。")
    print("     ⇒ 一个可算的**接口级**关系（不是判据）：`启动次数 × 每次元素数 == Ψ` = `%d × %d == %d`"
          % (BUCKET_COUNTS[-1], DIM // BUCKET_COUNTS[-1], DIM))
    check("B.2", BUCKET_COUNTS[-1] * (DIM // BUCKET_COUNTS[-1]) == DIM
          and specs.NCCL_RS_AG_EQ_QUOTE
          == "Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.",
          "★ **接口**：桶数 `k` ⇒ 启动 `k` 次（`%d × %d == %d`，整数恒等）。"
          "★★ 而 all-reduce 能被拆成**分段搬运**这件事本身是 [B] NCCL 的逐字结论"
          "（`%s`）—— **本程序不写它的算法与复杂度**（回链 `110-communication`）"
          % (BUCKET_COUNTS[-1], DIM // BUCKET_COUNTS[-1], DIM, specs.NCCL_RS_AG_EQ_QUOTE[:48] + "…"))


# ===========================================================================
# 6. C 类：边界与反例 —— 漏桶 ⇒ 发散；重叠 ⇒ 重复计数；错一个下标 ⇒ 恰好一个元素
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：漏桶 / 重叠 / 错一个下标的后果")
    print("=" * 74)
    x, y, local = toy_gradients(N_SYNC, PER_RANK_ROWS, DIM, SEED)
    g_one = single_card_gradient(x, y)
    scale = float(np.abs(g_one).max())

    # ── C.1 ★★ 漏桶 ⇒ 那部分不参与归约 ⇒ 各卡发散 ─────────────────────────
    print("  ── C.1 ★★ **漏一个桶**：只同步前 `128/192` 个元素 ──")
    sync_miss = bucket_sync(local, PLAN_MISS, N_SYNC)
    lo = max(hi for _, hi in PLAN_MISS)
    ptp_cov = cross_rank_ptp(sync_miss[:, :lo])
    ptp_unc = cross_rank_ptp(sync_miss[:, lo:])
    d_unc = float(np.abs(sync_miss[0, lo:] - g_one[lo:]).max())
    print("     **已同步部分**（`[0:%d)`）：跨 rank 极差 = **%.3e**（= `%.0f`）"
          % (lo, ptp_cov, TOL_EXACT))
    print("     **漏掉部分**（`[%d:%d)`）：跨 rank 极差 = **%.3e** · "
          "`max|− 正确值|` = **%.3e**" % (lo, DIM, ptp_unc, d_unc))
    print("     ⇒ ★★ **各卡不再持有同一个梯度** —— 下一步 `w ← w − lr·g` 之后，"
          "**每个 rank 的参数都不同**（副本**发散**）。")
    print("       ★ 已同步部分那一半（`%.3e`）是**恒真**的（覆盖到的元素各 rank 算的是同一个"
          "表达式，见 `A.1` 那段订正）—— **不进判据**，只作对照" % ptp_cov)
    check("C.1", ptp_unc > TOL_GUARD and d_unc > TOL_REL * max(scale, 1.0),
          "★★ **漏桶的后果是可量的**：**漏掉部分**（**有未覆盖区**）的极差 **`%.3e`** "
          "⇒ 那部分梯度**根本没参与归约**，各卡保留各自的本地值 ⇒ **复制品发散**"
          "（★ 判据只用**未覆盖区**那一段 —— 覆盖区的极差恒 `0.0`，没有信息量）" % (ptp_unc,))

    # ── C.2 ★★ 重叠 ⇒ 重复计数（求和口径下 = N 倍）────────────────────────
    print()
    print("  ── C.2 ★★ **重叠一个桶**：`[0:96]` 与 `[95:192]` ──")
    sync_ovl = bucket_sync(local, PLAN_OVERLAP, N_SYNC)
    ratios = sync_ovl[0, 95] / g_one[95]
    ptp_ovl = cross_rank_ptp(sync_ovl)
    print("     重叠的那个元素（下标 `95`）：`同步结果 / 正确值` = **%.6f**（`N = %d`）"
          % (float(ratios), N_SYNC))
    print("     ⇒ ★★ **原因**：桶是**依次**发生的，第二次 `all_reduce` 读到的是**已经归约过的值**"
          "（每个 rank 上都是 `Σ`）")
    print("        ⇒ 再求和一次 = `N × Σ`，最后统一除以 `N` ⇒ 那部分变成 **`N × 均值`**。")
    print("        ★★ 这是 **[B] NCCL 的 `SUM` 口径**（`%s`）的直接后果："
          "**重叠在 `SUM` 下会重复计数，在 `AVE` 下不会**。" % specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE)
    check("C.2", abs(float(ratios) - N_SYNC) <= TOL_RATIO * N_SYNC
          and specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE == "ReduceOp.SUM",
          "★★ **重叠 ⇒ 重复计数：那部分恰好是 `N = %d` 倍**（本机 `%.6f`）——"
          "★ 注意**各卡仍是一致的**（极差 `%.3e`：**这一条恒真**，见 `A.1` 那段订正）："
          "这个错**不会**让副本发散，它让**所有副本一起朝错的方向走**（比 `C.1` 更隐蔽）"
          % (N_SYNC, float(ratios), ptp_ovl))

    # ── C.3 ★ 错一个下标 ⇒ 恰好一个元素被重复计数 ─────────────────────────
    print()
    print("  ── C.3 ★ **错一个下标**：两个桶的端点各偏一位（`[0:96]` 与 `[95:192]`）──")
    diff = np.abs(sync_ovl[0] - g_one)
    bad = np.nonzero(diff > TOL_REL * max(scale, 1.0))[0]
    print("     与正确值不同的元素：**共 %d 个**，下标 = **%s**" % (len(bad), bad.tolist()))
    print("     ⇒ ★★ **只错一个下标，就只有【一个】元素错** —— 而且它**不报错、不崩**："
          "桶的边界错位是**静默**的（这正是「完备且不重叠」必须被**断言**、不能被指望的原因）。")
    check("C.3", len(bad) == DUP_COUNT and list(bad) == [95]
          and abs(float(sync_ovl[0, 95] / g_one[95]) - N_SYNC) <= TOL_RATIO * N_SYNC,
          "★★ **错一个端点 ⇒ 恰好 `%d` 个元素被重复计数**（下标 `%s`）⇒ "
          "桶划分的错误**不会自我暴露**，只能靠「完备且不重叠」这条判据抓"
          % (DUP_COUNT, bad.tolist()))


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
    print("  ★★ **禁区回链**（本程序一行都不重写）：梯度规约的算法（ring / tree allreduce、"
          "`V(N) = 2(N−1)/N × S`、reduce-scatter 与 all-gather 的阶段划分）在 `110-communication` "
          "与 `10-foundations/performance-models/02-amdahl-and-scaling.md`；")
    print("     ★★★ **「通信能藏多少」在本程序【不给判据】**（计划 §1.3.4）——"
          "那是**通信-计算重叠**，单进程形态下没有可跑的判据 ⇒ 只写接口，回链 `110-communication` 的 `overlap`。")
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

    measured = {"SEED": SEED, "N_SYNC": N_SYNC, "PER_RANK_ROWS": PER_RANK_ROWS, "DIM": DIM,
                "LR": LR, "PLAN_FULL": PLAN_FULL, "PLAN_EQ8": PLAN_EQ8, "PLAN_UNEQ": PLAN_UNEQ,
                "PLAN_MISS": PLAN_MISS, "PLAN_OVERLAP": PLAN_OVERLAP,
                "BUCKET_COUNTS": BUCKET_COUNTS, "PEAK_BYTES_EXPECT": PEAK_BYTES_EXPECT,
                "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT}
    expect = {"SEED": 20260942, "N_SYNC": 8, "PER_RANK_ROWS": 8, "DIM": 192, "LR": 0.05,
              "PLAN_FULL": ((0, 192),),
              "PLAN_EQ8": ((0, 24), (24, 48), (48, 72), (72, 96),
                           (96, 120), (120, 144), (144, 168), (168, 192)),
              "PLAN_UNEQ": ((0, 100), (100, 160), (160, 192)),
              "PLAN_MISS": ((0, 128),), "PLAN_OVERLAP": ((0, 96), (95, 192)),
              "BUCKET_COUNTS": (1, 8, 48), "PEAK_BYTES_EXPECT": (768, 96, 16),
              "FLOAT_EXACT_LIMIT": 9007199254740992}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", all(sum(hi - lo for lo, hi in p) == DIM for p in (PLAN_FULL, PLAN_EQ8, PLAN_UNEQ))
          and sum(hi - lo for lo, hi in PLAN_MISS) < DIM
          and sum(hi - lo for lo, hi in PLAN_OVERLAP) > DIM,
          "★★ `A`/`C` 类依赖的**三种划分在算术上真的不同**：完备的两条 `Σ = Ψ = %d`；"
          "`PLAN_MISS` 的 `Σ < Ψ`（漏）；`PLAN_OVERLAP` 的 `Σ > Ψ`（叠）——"
          "★ 这是**在任何计算之前**检查「被测的三种输入真的不一样」（`P23` 的同一条理由）" % DIM)
    check("D.3c", all(DIM % k == 0 for k in BUCKET_COUNTS)
          and peak_buffer_bytes(even_plan(BUCKET_COUNTS[-1], DIM), 4) == PEAK_BYTES_EXPECT[-1],
          "★ `B.1` 的桶数必须**整除** `Ψ`（`%s`）—— 不整除时峰值那一项的口径会变"
          "（**`P25`**：不整除是**常态**，本程序把它的处置**限制在 `B.1`**、并显式要求整除）"
          % (BUCKET_COUNTS,))

    th = {"TOL_EXACT": TOL_EXACT, "TOL_REL": TOL_REL, "TOL_GUARD": TOL_GUARD,
          "TOL_RATIO": TOL_RATIO, "DUP_COUNT": DUP_COUNT}
    th_expect = {"TOL_EXACT": 0.0, "TOL_REL": 1e-6, "TOL_GUARD": 1e-3,
                 "TOL_RATIO": 1e-6, "DUP_COUNT": 1}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 5 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < FLOAT_EXACT_LIMIT
          and FLOAT_EXACT_LIMIT == 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点阈值都 `< 2^53`（最大的是 `%g`）⇒ "
          "`mutate.py` 的 `v + 1.0` **能**改变它们（否则那类叶子**免检**）"
          % max(float(v) for v in th_expect.values()))
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

    check("D.5", specs.TORCH_ALLREDUCE_DEFAULT_OP_QUOTE == "ReduceOp.SUM"
          and specs.NCCL_RS_AG_EQ_QUOTE
          == "Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation."
          and specs.NCCL_ALLREDUCE_REDUCES_QUOTE
          == "The AllReduce operation performs reductions on data (for example, sum, min, max) "
             "across devices and stores the result in the receive buffer of every rank."
          and specs.NCCL_DOC_VERSION == "2.31.2"
          and specs.ZERO_ARXIV_ID == "1910.02054",
          "★★ 本程序消费的四条外部引文逐位一致 —— ★ 其中 [B] NCCL 的 `%s` 是「all-reduce "
          "**可以被拆成分段搬运**」的逐字依据（★ 它的**算法与复杂度不在本主题**，回链 `110-communication`）"
          % specs.NCCL_RS_AG_EQ_QUOTE[:44] + "…")
    check("D.5b", specs.ZERO_FIGURE_MODEL_QUOTE
          == "denotes model size (number of parameters), K"
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径",
          "★★ `Ψ` 的口径（**参数个数**）与 `16Ψ` 的**中文限定语**成对钉住 ——"
          "本程序把 `Ψ` 当**参数个数**用（`DIM = %d`），与那两条一致" % DIM)
    check("D.6", specs.UNSOURCED == ("nccl_allreduce_default_op",
                                     "data_parallel_redundancy_ratio_definition"),
          "★★ `specs.UNSOURCED` 的 **2** 项逐位钉住。★★ 订正（2026-09-23）："
          "原来还钉着第 3 项 `data_parallel_ring_allreduce_volume` —— 它**已移出**，"
          "因为它的性质是「**不在本主题**」（`V(N)` 的那一块，本程序**必须不写**）"
          "而不是「找不到来源」：它有来源，只是不在本域")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(allreduce_sum)
    src_b = inspect.getsource(bucket_sync)
    src_c = inspect.getsource(single_card_gradient)
    check("D.7", len({src_a, src_b, src_c}) == 3,
          "★ 防 `P17`：**搬运**（`allreduce_sum` %d 字符）、**分桶**（`bucket_sync` %d）、"
          "**独立参照**（`single_card_gradient` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c)))
    watched = {"allreduce_sum", "bucket_sync"}
    called = _callee_names(single_card_gradient)
    check("D.7b", not (watched & called),
          "★ **AST 调用表**：**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    # ★★ **调用计数器**：先证明它非空，再证明参照跑完为 0。
    _, _, local = toy_gradients(N_SYNC, PER_RANK_ROWS, DIM, SEED)
    _CALLS.clear()
    bucket_sync(local, PLAN_EQ8, N_SYNC)
    seen = sum(_CALLS.values())
    _CALLS.clear()
    single_card_gradient(*toy_gradients(N_SYNC, PER_RANK_ROWS, DIM, SEED)[:2])
    left = sum(_CALLS.values())
    check("D.7c", seen > 0 and left == 0,
          "★★ **调用计数器两向都量**：被测路径（8 桶）跑一次 ⇒ 计数 = **%d**（= 1 次 `bucket_sync` "
          "+ 8 次 `allreduce_sum`）· **参照路径**跑完 ⇒ 计数 = **%d**（必须**恰好为 0**）"
          % (seen, left))
    # ★ 两种量法的对照：极差 vs 标准差（`cross_rank_ptp` 的 docstring 是它的记录）
    _, _, loc2 = toy_gradients(N_SYNC, PER_RANK_ROWS, DIM, SEED)
    synced = bucket_sync(loc2, PLAN_FULL, N_SYNC)
    std_v = float(np.std(synced, axis=0).max())
    ptp_v = cross_rank_ptp(synced)          # ★ 订正：原来是写死的字面量 `0.0`，现在**真的量**
    check("D.8", ptp_v == TOL_EXACT and std_v > TOL_EXACT,
          "★★ **量法口径的现场**：同一个「各卡是否一致」的**全同**结果 —— "
          "**极差** = `%.1f`（**当场量出来的**：`cross_rank_ptp(synced)`，干净），"
          "而 `np.std` = **`%.3e`**（`mean` 那一步的舍入）⇒ "
          "用 `std` 就必须给判据加容差、于是「已同步 / 未同步」的**分界变模糊**"
          "（★ 订正 2026-09-23：原来那个 `0.0` 是**写死的字面量**，不是 `cross_rank_ptp` 的结果"
          " ⇒ 读者分不出哪一半是量的）" % (ptp_v, std_v))
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
    print("02_grad_sync.py —— DP 的工程：**梯度分桶**（与参数对齐）")
    print("=" * 74)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 核心判据：**梯度分桶的划分必须【完备且不重叠】**；"
          "★ 而「分桶后的结果 == 单桶的结果」是**纯搬运 ⇒ 恒真**，本程序不拿它当判据。")
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
