#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_algorithm_family.py —— **集合通信的算法族**：有哪几种、各自的**步数**与**每卡搬运量**。

配套文档：`01-algorithm-family.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**算法的代数**
> （步数、每卡搬运量、块索引），**不是**任何真实网络上的性能。

★★★ **本程序最重要的一条设计纪律（也是规划自己踩过的坑）**：
  **A 类的参照必须是【真消息序列模拟】** —— 逐步穷举「**哪一步、谁发给谁、哪个块、多少元素**」，
  把每一步的 `(phase, sender, receiver, block, n_elem)` 真的**枚举出来**，
  **再**与闭式对拍。
  ★★★ **不许把闭式用循环写一遍** —— 规划 `§D` 的 `D2` 就是这么写的
    （`for _ in range(N-1): sent += PER`），那样的「相对差 `0.0e+00`」是**恒真**的
    （`P12` 的现场：模拟器同时是被测对象）。
  ★★ 本程序的模拟器**驱动真实数据**（`numpy` 整数数组按块搬运、按块累加），
    ⇒ 它同时给出**结果正确性**与**搬运量**两个量。

★★★ **一条本程序自己发现、并写进正文的【结构性事实】**：
  · **每卡搬运量**对「块索引写错」是**不变量**（`±` 交换只是重排块，块数不变 ⇒ 搬运量不变）；
  · **结果正确性**对块索引**敏感**。
  ⇒ ★★ 所以「**搬运量 == 闭式**」这一条**过不了** `CONTRIBUTING:235` 的守卫
    （把块索引改错，它**照样通过**）；**过守卫的是结果正确性那条**。
  ★ 本程序把两件事**分别**记成 `A.1`（含正确性）与 `D.4b`（纯结构），并在 `D.4` 里
    用**同一个模拟器的错法变体**证明「换掉块索引**真的**改变了被判量」。
    ★★ **为什么这两条不叫 `A.2`/`A.1g`**：它们**不测任何实现** —— 任何算法变异都不会让
    它们失败（一条是**实验守卫**、一条是**整除口径**）⇒ 按 `CONTRIBUTING` §4「A 类 = 算法
    正确性」的含义，它们**必须归 D 类**（否则 A 类的计数会把「非实现判据」算进去）。

★★ **覆盖边界**（`CONTRIBUTING` §2「不得声称验证了更多」）：
  本程序守的是「**算法写对了没有**」（步数、块索引、累加目标、搬运量）+
  「**闭式的边界项有没有漏**」。
  **不守**任何一个**真实网络的性能**（那是实现层与物理层，本机**测不了**）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · ★★★ **`V(N) = 2(N−1)/N × S` 的推导与 `S` 的饱和**
    → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`
    （★ 本程序**不重推**它 —— 只把它当 **ring allreduce 的闭式**去与消息序列对拍）；
  · **bisection 下界 `t ≥ 2S/B`** · **all-to-all 跨切口量 `N·V/4`** · **分层 all-reduce 三段式**
    → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **`algbw` vs `busbw` 与修正系数表** → `interconnect/04-measuring-bandwidth.md`；
  · **带宽口径**（线速率 / 载荷 / 营销取整）→ `interconnect/01-bandwidth-conventions.md`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ ring allreduce 的**真消息序列模拟**：① 结果与**单进程参照**逐位相同；
    ② 步数 == `2(N−1)`；③ 每卡搬运量 == `2(N−1)/N × S`（`DEV_GRID` 五档）；
  · `D.4`  ★★★ **守卫**（`CONTRIBUTING:235`）：把**块索引**换成 `(d+k)%N` ⇒
    结果**必须变**、而**搬运量不变** ⇒ 证明「正确性那条**真的**改变了被判量」
    （★ 它**不测实现** ⇒ 归 D 类，**不占 A 类**）；
  · `D.4b` ★★ **整除口径**：`RING_ELEMS` 被 `DEV_GRID` 五档**全部整除** ⇒ 块大小是整数
    （★ 若这条不成立，A 类的块大小会有余数、搬运量闭式**前提失效**；
     ★ 它同样**不测实现** ⇒ 归 D 类）；
  · `A.3`  ★★ allgather 的 **ring 与递归倍增**：ring `p−1` 步、递归倍增 `lg p` 步，
    而**每卡搬运量两者相同**（`(p−1)/p × n`）；
  · `A.4`  ★★★ allgather 的 **Bruck**：`⌈lg p⌉` 步（★ 含 `GATHER_ODD = 6` 这个**非 2 的幂**档，
    它就落在「`⌈⌉` 与 `⌊⌋` 分得开」的地方）；搬运量仍是 `(p−1)/p × n`；
  · `A.5`  ★★ **二项树广播**：`⌈lg p⌉` 步、总消息数 `p−1`、总搬运量 `(p−1) × n`；
  · `A.6`  ★★★ **halving-doubling（Rabenseifner）allreduce**：`2 lg p` 步，
    搬运量 `2(p−1)/p × n`；★ 结果与单进程参照逐位相同；
  · `B.1`  ★★ `V(N)/S` 的**饱和**（`SAT_GRID` 六档，含 `4096` 档）；
  · `B.2`  ★★★ **三算法对照**：ring / 递归倍增 / Rabenseifner 的**步数**与**每卡搬运量**；
    ★★★ **数值那半是【跨模拟器】的真对照** —— `ring allreduce` 与 `Rabenseifner` 两个
    **独立消息序列模拟器**枚举出来的每卡搬运量必须相等（★ 拿两条**逐字同形**的闭式互比
    是**恒真**的，原写法已删）；★ 式子本身的载体是 `specs.py` 的 [C] 级逐字常量（`D.6` 钉住）；
  · `C.1`  ★★★ **环形块索引写错（发哪块）** ⇒ 结果错、**不报错**；
  · `C.2`  ★★★ **环形块索引写错（累加到哪块）** ⇒ 结果错、**不报错**（与 `C.1` 不同值）；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **引文逐字**钉住 + **失败标记自检**。
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
# ★★★ 失败标记：本主题（collectives）取 `&& FAIL: collectives`。
#   ★ 本域另外三个主题各用**不同的前缀**（`><` / `,,` / `+=`），四个标记**两两不互为前缀**
#     ⇒ 跨主题照抄会被下面两条断言当场抓住。
#   ★★ `&&` 是 **shell 元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**，
#     **不得**出现在 `Makefile` 的 recipe 行或任何 shell 命令行里。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = "&& FAIL: collectives"

# 本域另外三个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (">< FAIL: nccl", ",, FAIL: network-topology", "+= FAIL: overlap")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★ 口径常量**从 `specs.py` 读**（本行不产生可变异叶子：`_walk()` 不下钻 `Attribute`）
TOL_MACHINE = specs.TOL_MACHINE

# ★ 一条消息的形态：**这正是 `§3.2` 要求「真的枚举出来」的那个四元组**（外加**第几步**）。
#   ★★ 带上 `step` 是为了让**步数也是【枚举出来的】**，而不是照着闭式写死在循环上界里。
#   ★ 它是 `Call`（`collections.namedtuple(...)`）⇒ **不产生可变异叶子**。
MSG = collections.namedtuple("MSG", "step phase sender receiver block n_elem")


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
      ⇒ 所以 `cond` 为假时走**另一版文案**：默认只报「这条判据【不成立】」。
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
# 1. 被测的闭式（`[C]` 级载体见 `specs.py`）+ 一个单进程参照
#    ★★★ 这些函数是【被测】：模拟器（参照）**不得**调用它们（`D.8` 用 AST 调用表核）。
# ===========================================================================
def cf_ring_steps(n_dev):
    """ring allreduce 的步数闭式：reduce-scatter `N−1` 步 + all-gather `N−1` 步。"""
    return 2 * (n_dev - 1)


def cf_ring_volume(n_dev, n_elem):
    """ring allreduce 的**每卡搬运量**闭式 `2(N−1)/N × S`
    —— ★★ 这就是**上游已推导好的** `V(N)`（`performance-models/02:147-155`），
    本程序**不重推**它，只拿它与消息序列模拟对拍。"""
    return 2 * (n_dev - 1) / n_dev * n_elem


def cf_allgather_volume(p, n):
    """allgather 三个算法**共用**的每卡搬运量闭式 `(p−1)/p × n`
    （依据：`Tring` / `Trec dbl` / `Tbruck` 的带宽项逐字相同，见 `specs.py`）。"""
    return (p - 1) / p * n


def cf_allgather_ring_steps(p):
    """ring allgather 的步数闭式 `p − 1`（`MPICH_AG_RING_STEPS_QUOTE`）。"""
    return p - 1


def cf_allgather_lg_steps(p):
    """**裸 `lg p`** 的档 —— 递归倍增的步数（`MPICH_AG_RECDUB_QUOTE`，**无取整符号**）。"""
    return int(np.log2(p))


def cf_allgather_ceil_steps(p):
    """**`⌈lg p⌉`** 的档 —— Bruck 的步数（`MPICH_AG_BRUCK_STEPS_QUOTE`）。"""
    return int(np.ceil(np.log2(p)))


def cf_allreduce_rabenseifner_steps(p):
    """halving-doubling（Rabenseifner）allreduce 的步数闭式 `2 lg p`。"""
    return 2 * int(np.log2(p))


def cf_allreduce_rabenseifner_volume(p, n):
    """它的每卡搬运量闭式 `2(p−1)/p × n`（两段各 `(p−1)/p × n`）。"""
    return 2 * (p - 1) / p * n


def cf_allreduce_recdub_volume(p, n):
    """**递归倍增 allreduce** 的每卡搬运量闭式 `n lg p`
    —— ★★★ 这是 Rabenseifner 要压掉的那一项（`MPICH_ALLREDUCE_RECDUB_TIME_QUOTE`）。"""
    return n * int(np.log2(p))


def ref_allreduce_sum(x):
    """**单进程参照**：把每台设备的输入直接加起来。

    ★★ 它与消息序列模拟器**不共享任何代码**（一个是「谁发给谁」的逐步模拟，
      一个是 `numpy` 的 `sum`）⇒ 两条**独立路径**（`CONTRIBUTING:232`）。
    """
    return x.sum(axis=0)


# ===========================================================================
# 2. 参照：**六个**真消息序列模拟器（逐步枚举 `(phase, sender, receiver, block, n_elem)`）
#    ★★★ 它们**不调用**上节的任何 `cf_*`（`D.8` 核 —— ★★ **逐个审这六个**，不再只审一个）。
#    ★★ 每个模拟器都**驱动真实数据**，最后返回 `(每个设备手里的结果, 消息清单)`。
# ===========================================================================
def _split(n_elem, p):
    """把 `range(n_elem)` 均分成 `p` 块（★ 调用方保证整除 —— `D.4b` 守这件事）。"""
    w = n_elem // p
    return [(i * w, (i + 1) * w) for i in range(p)]


def ring_send_block(sender, k, n_dev):
    """★★★ **ring allreduce 的块索引**（reduce-scatter 段）—— **本程序最要紧的一行**。

    ★ `(sender - k) % n_dev`：第 `k` 步，设备 `sender` 发出的是**它自己往前数第 `k` 块**。
    ★★★ **把这里的 `-` 改成 `+` 就是 §3.2 那条守卫实验** ——
      它**不改变搬运量**（`±` 只是重排块），但**会改变结果**（`A.1` 必须失败）。
    """
    return (sender - k) % n_dev


def sim_ring_allreduce(n_dev, n_elem, x=None, corrupt=None):
    """★★★ **ring allreduce 的真消息序列模拟**（reduce-scatter + all-gather）。

    · `corrupt=None`   ⇒ 正确实现；
    · `corrupt="send"`  ⇒ `C.1`：**发错块**（发 `(d+k)%N`）；
    · `corrupt="accum"` ⇒ `C.2`：**累加错目标**（把收到的加到 `(块+1)%N` 上）。

    ★★ 每一步都记录 `MSG(phase, sender, receiver, block, n_elem)` ——
      **这就是「谁发给谁、哪个块、多少元素」的逐步枚举**，
      ★★★ **而不是** `for _ in range(N-1): sent += PER` 那种把闭式抄一遍的写法。
    ★★★ **错法变体的读法是【宽容的】**（`_get` 缺块给零、`pop` 缺块给 `None`）——
      这是**刻意**的：`D1b` 的现场就是「**累加目标越界（空切片）** ⇒
      **不报错、只是结果错**」。★ 若让错法变体抛 `KeyError`，那它就变成**崩溃捕获**，
      **不是**本程序要演示的「**不报错、结果错**」。
    返回 `(parts, msgs)`：`parts[d]` 是设备 `d` 最后手里的**完整向量**。
    """
    parts = _split(n_elem, n_dev)
    if x is None:
        x = np.arange(n_dev * n_elem, dtype=np.int64).reshape(n_dev, n_elem)
    # 每台设备手里：块号 -> 数组（RS 段里是**部分和**）
    held = [{i: x[d][s:e].copy() for i, (s, e) in enumerate(parts)} for d in range(n_dev)]
    msgs = []

    def _get(slot, blk, like):
        """★ 宽容读：缺块给**同形状的零数组**（模拟「越界得到空缓冲区」）。"""
        return slot[blk] if blk in slot else np.zeros_like(like)

    # ── 第一段：reduce-scatter ────────────────────────────────────────────
    # ★★★ **步数【涌现】自【一个工作队列】**（**不是** `for k in range(n_dev - 1)`）：
    #   `todo[d]` = 设备 `d` 还要转发几块 —— 初始值**取自数据**（`len(held[d]) - 1`，
    #   即「每台设备初始持有几块」减 1）；每转发一块减 1，队列空了就停。
    #
    # ★★★ 【现场】为什么这里**不写** `while max(len(h))>1`（`01` 其余四个模拟器 + AG 段用的
    #   那种【状态】条件）—— ★ 这是本库一个**实测顶掉处方**的现场，留给下一个域：
    #
    #     做法                              `C.1` 每卡搬运量（正确版 vs `corrupt="send"`）  判据/D.4
    #     ────────────────────────────────  ──────────────────────────────────────────  ────────
    #     `for k in range(n_dev - 1)`        `96 vs 96`    ✓                                通过
    #     `while max(len(h) for h in held)>1`**`96 vs 1088`** ✗（跑飞撞 `k>64` 护栏）        **失败**
    #     本行的工作队列 `while max(todo)>0` `96 vs 96`    ✓                                通过
    #
    #   ★★★ **可证的理由**（不是"看起来会"）：本段**唯一**两条错法
    #   （`corrupt="send"`/`"accum"`）**扰动的【正是】`held` 的块数** ——
    #   `corrupt="send"` 下，设备 `d` 第 `k` 步发出的块 `(d+k)%N` **恰好就是它第 `k−1` 步
    #   刚被腾空的那个槽**（`(d−1+k)%N == (d+(k−1))%N`）⇒ 设备 `d` **每步丢掉一块、又收回一块**
    #   ⇒ **它的块数【永不下降】** ⇒ `max(len(h))>1` **永不成立** ⇒ 循环跑到 `k>64` 的护栏
    #   ⇒ 搬运量从 `96` 变 `1088`、`A.1g`/`D.4` 的「**搬运量对块索引不变**」当场失效。
    #   ★ **转发次数不随错法变**（错法只改「**发哪块 / 加到哪块**」，**不改「发几次」**）
    #   ⇒ ★ 队列驱动的读法对**正确版与两条错法都成立**。
    #   ★★ 一句话：**环上的 RS 段是【固定调度】（每台转发 `N−1` 次），它的完成不能由
    #      `held` 的块数判定**；AG 段（与其余五个模拟器）**能**用状态条件（已独立实测）。
    todo = [len(h) - 1 for h in held]      # ★ 取自数据：每台要转发「初始块数 − 1」块
    k = 0
    while max(todo) > 0:
        batch = []
        for d in range(n_dev):
            blk = ring_send_block(d, k, n_dev) if corrupt != "send" else (d + k) % n_dev
            src = _get(held[d], blk, np.zeros(n_elem // n_dev, dtype=np.int64))
            batch.append((d, (d + 1) % n_dev, blk, src.copy(), len(src)))
        for d, dst, blk, src, ne in batch:                 # 先全发，再全收（lockstep）
            msgs.append(MSG(k, "reduce-scatter", d, dst, blk, ne))
            tgt = blk if corrupt != "accum" else (blk + 1) % n_dev
            held[dst][tgt] = _get(held[dst], tgt, src) + src
            held[d].pop(blk, None)                          # 发过的块本设备不再需要
            todo[d] -= 1
        k += 1
        if k > 64:
            break
    n_rs_steps = k                                     # ★ 第一段【枚举出来】的步数

    # ── 第二段：all-gather ────────────────────────────────────────────────
    # ★★★ **步数【涌现】自【数据】**：还有设备没拿全（`min(len(h)) < n_dev`）就再来一步
    #   —— 这一段是**状态驱动**的，且**对两条错法都成立**（实测 `C.1`/`C.2` 全过）。
    k = 0
    while min(len(h) for h in held) < n_dev:
        batch = []
        for d in range(n_dev):
            blk = (d + 1 - k) % n_dev
            src = _get(held[d], blk, np.zeros(n_elem // n_dev, dtype=np.int64))
            batch.append((d, (d + 1) % n_dev, blk, src.copy()))
        for d, dst, blk, src in batch:
            msgs.append(MSG(n_rs_steps + k, "all-gather", d, dst, blk, len(src)))
            held[dst][blk] = src                            # 收下
            held[d].setdefault(blk, src)                    # ★ 发起方保留自己那份
        k += 1
        if k > 64:
            break

    order = list(range(n_dev))
    out = [np.concatenate([held[d][i] for i in order]) for d in range(n_dev)]
    return out, msgs


def sim_allgather_ring(p, n):
    """ring allgather 的消息序列模拟：**逐步枚举**每步谁把哪一块传给谁。

    ★★★ **步数是【涌现的】**：还有设备没拿全（`min(len(h)) < p`）就再来一步 ——
      **不是** `for k in range(p - 1)`（那会把「`p−1`」预先塞进控制流，让
      「步数 == 闭式」变成**准恒真**，`P12`）。
    """
    w = n // p
    base = [np.arange(i * w, (i + 1) * w, dtype=np.int64) for i in range(p)]
    held = [{d: base[d].copy()} for d in range(p)]
    msgs = []
    k = 0
    while min(len(h) for h in held) < p:            # ★ 涌现：还缺块就再来一步
        batch = []
        for d in range(p):
            blk = (d - k) % p
            batch.append((d, (d + 1) % p, blk, held[d][blk].copy()))
        for d, dst, blk, src in batch:
            msgs.append(MSG(k, "allgather-ring", d, dst, blk, len(src)))
            held[dst][blk] = src
        k += 1
        if k > 64:
            break
    out = [np.concatenate([held[d][i] for i in range(p)]) for d in range(p)]
    return out, msgs


def sim_allgather_recdub(p, n):
    """递归倍增 allgather 的消息序列模拟：第 `k` 步与 `d XOR 2^k` 交换**手里全部的块**。

    ★★★ **步数是【涌现的】**：还有设备没拿全（`min(len(h)) < p`）就再来一步 ——
      **不是** `for k in range(cf_allgather_lg_steps(p))`（那会把**被测的闭式**当控制流上界 ⇒
      「步数 == 闭式」变成**准恒真**，`P12`）。
    """
    w = n // p
    base = [np.arange(i * w, (i + 1) * w, dtype=np.int64) for i in range(p)]
    held = [{d: base[d].copy()} for d in range(p)]
    msgs = []
    k = 0
    while min(len(h) for h in held) < p:            # ★ 涌现：还缺块就再来一步
        batch = []
        for d in range(p):
            partner = d ^ (1 << k)
            for blk, arr in held[d].items():
                batch.append((d, partner, blk, arr.copy()))
        for d, partner, blk, src in batch:
            msgs.append(MSG(k, "allgather-recdub", d, partner, blk, len(src)))
            held[partner][blk] = src
        k += 1
        if k > 64:                                  # ★ 防跑飞（本主题只用到 `p <= 16`）
            break
    out = [np.concatenate([held[d][i] for i in range(p)]) for d in range(p)]
    return out, msgs


def sim_allgather_bruck(p, n):
    """Bruck allgather 的消息序列模拟：第 `k` 步与 `(d − 2^k) mod p` 交换、
    ★★ **只发目标缺的那些块**（这正是它能在**任意 `p`** 上保持 `(p−1)/p × n` 的原因）。

    ★★★ **步数【涌现】**（`min(len(h)) < p`），不是 `range(cf_allgather_ceil_steps(p))`。
    """
    w = n // p
    base = [np.arange(i * w, (i + 1) * w, dtype=np.int64) for i in range(p)]
    held = [{d: base[d].copy()} for d in range(p)]
    msgs = []
    k = 0
    while min(len(h) for h in held) < p:            # ★ 涌现：任意 `p`（含 GATHER_ODD）都靠它
        batch = []
        for d in range(p):
            partner = (d - (1 << k)) % p
            need = set(held[d]) - set(held[partner])        # ★★ 只发目标缺的
            for blk in sorted(need):
                batch.append((d, partner, blk, held[d][blk].copy()))
        for d, partner, blk, src in batch:
            msgs.append(MSG(k, "allgather-bruck", d, partner, blk, len(src)))
            held[partner][blk] = src
        k += 1
        if k > 64:
            break
    out = [np.concatenate([held[d][i] for i in range(p)]) for d in range(p)]
    return out, msgs


def sim_broadcast_binomial(p, n, root=0):
    """二项树广播的消息序列模拟：第 `k` 步，**已有数据的** `r < 2^k` 发给 `r + 2^k`。

    ★★★ **步数【涌现】**（`len(have) < p`），不是 `range(cf_allgather_ceil_steps(p))`。
    """
    msg = np.arange(n, dtype=np.int64)
    have = {root: msg.copy()}
    msgs = []
    k = 0
    while len(have) < p:                            # ★ 涌现：还没铺满就再来一步
        batch = []
        for r in sorted(have):
            if r < (1 << k) and r + (1 << k) < p:
                batch.append((r, r + (1 << k)))
        for r, dst in batch:
            msgs.append(MSG(k, "broadcast-tree", r, dst, 0, n))
            have[dst] = have[r].copy()
        k += 1
        if k > 64:
            break
    return have, msgs


def sim_allreduce_rabenseifner(p, n):
    """halving-doubling（Rabenseifner）allreduce = **recursive halving 的 reduce-scatter**
    + **recursive doubling 的 all-gather**（★ 两段，各自逐步枚举）。

    ★★★ **两段的步数都是【涌现的】**：第一段问「**还有设备手里多于一块吗**」、
      第二段问「**还有设备没拿全吗**」—— **不是** `range(lg)`（那又是把闭式抄一遍）。
    """
    parts = _split(n, p)
    x = np.arange(p * n, dtype=np.int64).reshape(p, n)
    held = [{i: x[d][s:e].copy() for i, (s, e) in enumerate(parts)} for d in range(p)]
    msgs = []

    # ── 第一段：recursive halving 的 reduce-scatter（每步减半）─────────────
    k = 0
    while max(len(h) for h in held) > 1:            # ★ 涌现：减到每台只剩一块为止
        batch = []
        for d in range(p):
            partner = d ^ (1 << k)
            send = [i for i in sorted(held[d]) if ((i >> k) & 1) != ((d >> k) & 1)]
            for blk in send:
                batch.append((d, partner, blk, held[d][blk].copy()))
        for d, partner, blk, src in batch:
            msgs.append(MSG(k, "rs-halving", d, partner, blk, len(src)))
            held[partner][blk] = held[partner][blk] + src   # ★ 累加进**保留的那一半**
        for d, partner, blk, src in batch:
            del held[d][blk]
        k += 1
        if k > 64:
            break
    n_rs = k                                        # ★ 第一段【枚举出来】的步数

    # ── 第二段：recursive doubling 的 all-gather ──────────────────────────
    k = 0
    while min(len(h) for h in held) < p:            # ★ 涌现：还没拿全就再来一步
        batch = []
        for d in range(p):
            partner = d ^ (1 << k)
            for blk, arr in held[d].items():
                batch.append((d, partner, blk, arr.copy()))
        for d, partner, blk, src in batch:
            msgs.append(MSG(n_rs + k, "ag-recdub", d, partner, blk, len(src)))
            held[partner][blk] = src
        k += 1
        if k > 64:
            break
    out = [np.concatenate([held[d][i] for i in range(p)]) for d in range(p)]
    return out, msgs


def vol_of(msgs, n_dev):
    """**从枚举出来的消息清单**算每台设备的发出元素数 —— ★ 逐条求和，不是乘出来的。"""
    return [sum(m.n_elem for m in msgs if m.sender == d) for d in range(n_dev)]


# ===========================================================================
# 3. A 类：消息序列模拟 vs 闭式（★ 与上游推导是**两条独立路径**）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 算法族：**真消息序列模拟**（逐步枚举谁发给谁）vs **闭式**")
    print("=" * 78)
    print("  ★★★ 参照 = 消息序列模拟器（驱动真实数据、逐步枚举四元组）；")
    print("      被测 = 闭式（`cf_*`）。★ 两条**独立路径**，不是把闭式抄一遍。")

    # ── A.1 ★★★ ring allreduce ────────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ ring allreduce（reduce-scatter + all-gather）──")
    x = np.arange(max(specs.DEV_GRID) * specs.RING_ELEMS, dtype=np.int64) \
        .reshape(max(specs.DEV_GRID), specs.RING_ELEMS)
    ok, rows = True, []
    for n_dev in specs.DEV_GRID:
        sub = x[:n_dev]
        got, msgs = sim_ring_allreduce(n_dev, specs.RING_ELEMS, x=sub)
        exact = all(np.array_equal(g, ref_allreduce_sum(sub)) for g in got)
        steps = max(m.step for m in msgs) + 1      # ★ 步数**枚举出来**（两段**连号**）
        vol = vol_of(msgs, n_dev)[0]
        cf_v = cf_ring_volume(n_dev, specs.RING_ELEMS)
        rel = abs(vol - cf_v) / cf_v
        rows.append((n_dev, steps, len(msgs), vol, cf_v, rel, exact))
        ok = ok and exact and (steps == cf_ring_steps(n_dev)) and (rel == 0.0) \
            and (len(msgs) == n_dev * cf_ring_steps(n_dev))
        print("     N=%-3d 总步数 %-3d（= `2(N−1)`）· 每卡搬运量 %-4d（闭式 %.1f，相对差 %.1e）"
              " ⇒ 结果%s" % (n_dev, steps, vol, cf_v, rel, "逐位相同 ✓" if exact else "不同 ✗"))
    check("A.1", ok,
          "★★★ **判据成立**（`DEV_GRID` 五档）：① 每个设备手里的结果与**单进程参照**"
          "`x.sum(axis=0)` **逐位相同**；② 每段步数 == `N−1`（共 `2(N−1)`）；"
          "③ **每卡搬运量 == `2(N−1)/N × S`**（相对差**恰好 `0`**）；④ 消息总数 == `N × 2(N−1)`",
          "**ring allreduce 的消息序列模拟与闭式对不上** —— ★ 先查块索引 `ring_send_block`")

    # ── D.4 ★★★ 守卫（★ 归 D 类：它**不测实现**，任何算法变异都不会让它失败）──────
    print()
    print("  ── D.4 ★★★ **守卫**（`CONTRIBUTING:235`）：把块索引改成 `(d+k)%N` ──")
    n_g = specs.DEV_GRID[1]
    sub_g = x[:n_g]
    good, m_good = sim_ring_allreduce(n_g, specs.RING_ELEMS, x=sub_g)
    bad, m_bad = sim_ring_allreduce(n_g, specs.RING_ELEMS, x=sub_g, corrupt="send")
    ref_g = ref_allreduce_sum(sub_g)
    same_good = all(np.array_equal(g, ref_g) for g in good)
    bad_differs = any(not np.array_equal(g, ref_g) for g in bad)
    vol_same = vol_of(m_good, n_g)[0] == vol_of(m_bad, n_g)[0]
    print("     N=%d：正确索引 ⇒ 结果逐位相同 = %s；改成 `(d+k)%%N` ⇒ 结果变 = %s；"
          "而**搬运量不变** = %s" % (n_g, same_good, bad_differs, vol_same))
    check("D.4", same_good and bad_differs and vol_same,
          "★★★ **守卫成立**：换掉块索引 ⇒ ① 正确版仍逐位相同；② 错法版结果**真的变了**；"
          "③ **搬运量两者相同** ⇒ ★★ **证明「正确性」那条真的改变了被判量**，"
          "而「搬运量」那条**对块索引不敏感**（★ 这正是本程序把两件事分成 `A.1`/`D.4b` 的原因）"
          "\n        ★★ **它归 D 类而不是 A 类**：它是**实验守卫**，不测任何实现。",
          "**守卫失效** —— 块索引改成 `(d+k)%%N` 后结果居然不变（`A.1` 对块索引不敏感）")

    # ── D.4b ★★ 整除口径（★ 归 D 类：不测实现）──────────────────────────────
    print()
    print("  ── D.4b ★★ **整除口径**：`RING_ELEMS` 被 `DEV_GRID` 五档全部整除 ──")
    div = [specs.RING_ELEMS % n for n in specs.DEV_GRID]
    print("     `RING_ELEMS=%d` 对 `DEV_GRID=%s` 取余 ⇒ `%s`（全为 0 ⇒ 块大小是整数）"
          % (specs.RING_ELEMS, specs.DEV_GRID, div))
    check("D.4b", all(r == 0 for r in div) and len(specs.DEV_GRID) == 5,
          "★★ **口径成立**：`RING_ELEMS = %d` 是 `DEV_GRID` **五档的最小公倍数**的倍数 ⇒"
          "**每块的元素数都是整数** ⇒ `A.1` 的闭式 `2(N−1)/N × S` 的**整除前提**成立"
          "\n        ★★ **它归 D 类而不是 A 类**：它是**口径检查**，不测任何实现。"
          % specs.RING_ELEMS,
          "**整除被破坏** —— ★ 那意味着块大小有余数，`A.1` 的闭式前提失效")

    # ── A.3 ★★ allgather：ring vs 递归倍增 ────────────────────────────────
    print()
    print("  ── A.3 ★★ allgather：**ring 与递归倍增**（同一原语、两种算法）──")
    ok3, rows3 = True, []
    for p in specs.GATHER_GRID:
        vols_here = {}
        for name, fn, cf_s in (("ring", sim_allgather_ring, cf_allgather_ring_steps),
                               ("recdub", sim_allgather_recdub, cf_allgather_lg_steps)):
            got, msgs = fn(p, specs.GATHER_ELEMS)
            full = np.arange(specs.GATHER_ELEMS, dtype=np.int64)
            exact = all(np.array_equal(g, full) for g in got)
            vol = vol_of(msgs, p)[0]
            cf_v = cf_allgather_volume(p, specs.GATHER_ELEMS)
            steps = max(m.step for m in msgs) + 1      # ★ 步数是**枚举出来的**，不是写死的
            ok3 = ok3 and exact and (vol == cf_v) and (steps == cf_s(p))
            vols_here[name] = vol
            rows3.append((p, name, cf_s(p), vol, cf_v, exact))
            print("     p=%-3d %-7s 步数 %-3d · 每卡搬运量 %-4d（闭式 %.1f）⇒ 结果%s"
                  % (p, name, cf_s(p), vol, cf_v, "相同 ✓" if exact else "不同 ✗"))
        # ★★ 同一个 p 下两个算法的搬运量必须相同（★ 这是本判据的核心）
        ok3 = ok3 and vols_here["ring"] == vols_here["recdub"]
    # ★★ 真对照：**跨算法**比 —— 同一个 `p` 下 `rows3` 里相邻两行是 (ring, recdub)，
    #   ★ 它们的**枚举出来的搬运量**必须相等（★ 原写法 `r[3] == r[3]` 是恒真，已删）。
    same_vol = all(rows3[i][3] == rows3[i + 1][3] for i in range(0, len(rows3), 2)) \
        and len(rows3) == 8
    check("A.3", ok3 and same_vol,
          "★★ **判据成立**：ring `p−1` 步、递归倍增 `lg p` 步；★★★ **而【同一个 p 下】两者的"
          "每卡搬运量完全相同**（`GATHER_GRID` 四档逐一核过，都是 `(p−1)/p × n`）"
          "⇒ **步数与搬运量是两条独立的轴**",
          "**allgather 的步数或搬运量与闭式不符** —— ★ 先查 `sim_allgather_*` 的配对式")

    # ── A.4 ★★★ Bruck（含非 2 的幂）──────────────────────────────────────
    print()
    print("  ── A.4 ★★★ Bruck：`⌈lg p⌉` 步，**任意 p**（含非 2 的幂）──")
    ok4 = True
    for p in list(specs.GATHER_GRID) + [specs.GATHER_ODD]:
        n = specs.GATHER_ELEMS if specs.GATHER_ELEMS % p == 0 else p
        got, msgs = sim_allgather_bruck(p, n)
        full = np.arange(n, dtype=np.int64)
        exact = all(np.array_equal(g, full) for g in got)
        vol = vol_of(msgs, p)[0]
        cf_v = cf_allgather_volume(p, n)
        steps = max(m.step for m in msgs) + 1     # ★ 步数**枚举出来**
        ok4 = ok4 and exact and (vol == cf_v) \
            and (steps == cf_allgather_ceil_steps(p)) and (len(msgs) == p * (p - 1))
        print("     p=%-3d（%s）步数 %-3d = `⌈lg p⌉` · 每卡搬运量 %-4d（闭式 %.1f）⇒ 结果%s"
              % (p, "2 的幂" if p & (p - 1) == 0 else "**非 2 的幂**",
                 steps, vol, cf_v, "相同 ✓" if exact else "不同 ✗"))
    print("     ★★ `GATHER_ODD = %d` 这一档上 **`⌈lg 6⌉ = 3` 而 `⌊lg 6⌋ = 2`** ——"
          " 两个取整符号在这一档**分得开**（2 的幂上它们相等）。" % specs.GATHER_ODD)
    check("A.4", ok4,
          "★★★ **判据成立**：Bruck 在 **2 的幂与 `GATHER_ODD`（非 2 的幂）**上都用 `⌈lg p⌉` 步"
          "**收全全部块**，且每卡搬运量仍是 `(p−1)/p × n`（★ 靠「只发目标缺的块」这条剪枝）",
          "**Bruck 的步数或搬运量与闭式不符** —— ★ 先查「只发目标缺的那些块」这条剪枝")

    # ── A.5 ★★ 二项树广播 ────────────────────────────────────────────────
    print()
    print("  ── A.5 ★★ 二项树广播：`⌈lg p⌉` 步、总消息 `p−1` 条 ──")
    ok5 = True
    for p in specs.GATHER_GRID:
        have, msgs = sim_broadcast_binomial(p, specs.GATHER_ELEMS)
        moved = have[p - 1] is not None and np.array_equal(have[p - 1],
                                                           np.arange(specs.GATHER_ELEMS))
        tot_e = sum(m.n_elem for m in msgs)
        per_send = max(m.n_elem for m in msgs)
        steps = max(m.step for m in msgs) + 1     # ★ 步数**枚举出来**
        ok5 = ok5 and moved and len(msgs) == p - 1 \
            and tot_e == (p - 1) * specs.GATHER_ELEMS \
            and per_send == specs.GATHER_ELEMS \
            and steps == cf_allgather_ceil_steps(p)
        print("     p=%-3d 步数 `⌈lg p⌉` = %-2d · 总消息 %-3d 条 · 总搬运量 %-4d = `(p−1)×n`"
              % (p, steps, len(msgs), tot_e))
    check("A.5", ok5,
          "★★ **判据成立**：二项树广播 `⌈lg p⌉` 步走完、总消息数 `p−1`、"
          "总搬运量 `(p−1) × n`（每设备**最多发一次**，且发的都是**整条消息**）",
          "**二项树广播的步数/消息数与闭式不符** —— ★ 先查「已有数据的 `r < 2^k` 才发」这条")

    # ── A.6 ★★★ Rabenseifner allreduce ───────────────────────────────────
    print()
    print("  ── A.6 ★★★ halving-doubling（Rabenseifner）allreduce：`2 lg p` 步 ──")
    ok6 = True
    for p in specs.GATHER_GRID:
        n = specs.STAGE_ELEMS * p                       # ★ 每块的元素数恒为 STAGE_ELEMS
        sub = np.arange(p * n, dtype=np.int64).reshape(p, n)
        got, msgs = sim_allreduce_rabenseifner(p, n)
        exact = all(np.array_equal(g, ref_allreduce_sum(sub)) for g in got)
        vol = vol_of(msgs, p)[0]
        cf_v = cf_allreduce_rabenseifner_volume(p, n)
        steps = max(m.step for m in msgs) + 1           # ★ 步数**枚举出来**
        ok6 = ok6 and exact and (vol == cf_v) and (steps == cf_allreduce_rabenseifner_steps(p))
        print("     p=%-3d 步数 `2 lg p` = %-2d · 每卡搬运量 %-4d（闭式 %.1f）⇒ 结果%s"
              % (p, steps, vol, cf_v, "逐位相同 ✓" if exact else "不同 ✗"))
    check("A.6", ok6,
          "★★★ **判据成立**：Rabenseifner allreduce 在 `GATHER_GRID` 四档上 "
          "① 结果与单进程参照**逐位相同**；② 步数 == `2 lg p`；③ 每卡搬运量 == `2(p−1)/p × n`",
          "**Rabenseifner 的步数/搬运量/结果与闭式不符** —— ★ 先查两段的累加目标")


# ===========================================================================
# 4. B 类：饱和 · 三算法对照（★ 用 `specs.py` 的 [C] 级逐字式复核）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：`V(N)/S` 的**饱和** · 三个 allreduce 算法的**对照**")
    print("=" * 78)

    # ── B.1 ★★ 饱和 ──────────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ `V(N)/S = 2(N−1)/N` 的**饱和**（★ 推导在上游，本程序只回链）──")
    vals = []
    for n_dev in specs.SAT_GRID:
        r = 2 * (n_dev - 1) / n_dev
        vals.append((n_dev, r))
        print("     N=%-5d `V(N)/S` = **%.6f**" % (n_dev, r))
    mono = all(vals[i][1] < vals[i + 1][1] for i in range(len(vals) - 1))
    sat = abs(vals[-1][1] - 2.0) < 0.001 and all(v < 2.0 for _n, v in vals)
    print("     ★★ **单调上升、恒 `< 2`、且最末档 `%.6f` 已饱和到 `2`** ——"
          " ★★★ 这个式子的**推导与饱和**在 `performance-models/02:147-155,209`，"
          "**本主题不重推**（此处只做边界检查）。" % vals[-1][1])
    check("B.1", mono and sat and len(vals) == 6,
          "★★ **判据成立**：`SAT_GRID` 六档上 `V(N)/S` **严格单调上升**、**恒 `< 2`**、"
          "且 `4096` 档已到 `%.6f`（差 `< 1e-3`）⇒ **饱和**" % vals[-1][1],
          "**饱和的边界不成立** —— ★ 注意本程序**只做边界检查**，闭式本身回链上游")

    # ── B.2 ★★★ 三算法对照 ───────────────────────────────────────────────
    print()
    print("  ── B.2 ★★★ **三个 allreduce 算法**：步数 × 每卡搬运量（同一个 `n`）──")
    n = specs.STAGE_ELEMS
    tbl = []
    for p in specs.GATHER_GRID:
        ring_s, ring_v = cf_ring_steps(p), cf_ring_volume(p, n)
        rd_s, rd_v = cf_allgather_lg_steps(p), cf_allreduce_recdub_volume(p, n)
        ra_s, ra_v = cf_allreduce_rabenseifner_steps(p), \
            cf_allreduce_rabenseifner_volume(p, n)
        tbl.append((p, ring_s, ring_v, rd_s, rd_v, ra_s, ra_v))
        print("     p=%-3d │ ring: %-3d 步 / %-6.2f │ 递归倍增: %-2d 步 / %-6.2f │ "
              "Rabenseifner: %-2d 步 / %-6.2f"
              % (p, ring_s, ring_v, rd_s, rd_v, ra_s, ra_v))
    print("     ★★★ **三条结构性事实**（全部可手算复核）：")
    print("       · Rabenseifner 的**搬运量与 ring 相同**、而**步数是对数级** ⇒ 它**两头都占**；")
    print("       · 递归倍增的**步数最少**（`lg p`）、代价是**搬运量 `n lg p` 最大**；")
    print("       · ring 的**搬运量最小**、代价是**步数 `2(p−1)` 线性**。")
    print("     ★★ **注意**：闭式里 Rabenseifner 与 ring 的每卡搬运量**逐字同形**"
          "（都是 `2(p−1)/p × n`）")
    print("        ⇒ 拿**两条闭式**互比是**恒真**的（★ 原写法 `abs(t[2]-t[6]) < 1e-12` 已删）；")
    print("        ★ 所以这里比的是**两个模拟器【枚举出来】的每卡搬运量**（真对照）。")
    # ★★★ **真对照**：`ring allreduce` 与 `Rabenseifner allreduce` 是**两条独立的**消息序列路径；
    #   同一个 `p`、同一个每设备输入 `ne = STAGE_ELEMS × p`（块大小是整数）下，两者的枚举
    #   搬运量必须相等 —— ★ 这才是有信息量的那一条（闭式同形不算）。
    enum_ok = True
    for p in specs.GATHER_GRID:
        ne = specs.STAGE_ELEMS * p
        _o, m_ring = sim_ring_allreduce(p, ne)
        _o2, m_ra = sim_allreduce_rabenseifner(p, ne)
        v_ring = vol_of(m_ring, p)[0]
        v_ra = vol_of(m_ra, p)[0]
        enum_ok = enum_ok and (v_ring == v_ra)
        print("     p=%-3d **枚举**：ring allreduce %-4d · Rabenseifner %-4d ⇒ 相等 = %s"
              % (p, v_ring, v_ra, v_ring == v_ra))
    ok = enum_ok and all(t[6] <= t[4] for t in tbl) and len(tbl) == 4
    check("B.2", ok,
          "★★★ **判据成立**（`GATHER_GRID` 四档）：**两个模拟器【枚举出来】的每卡搬运量相等**"
          "（`ring allreduce` 与 `Rabenseifner` —— ★ 两条**独立**的消息序列路径，**不是**两条"
          "同形的闭式互比），且 **Rabenseifner 的搬运量恒不超过递归倍增的**（`n lg p`）—— "
          "★ 三条式子的载体是 `specs.py` 的 [C] 级逐字常量（`D.6` 钉住它们）",
          "**三算法的对照关系不成立** —— ★ 先查两个模拟器（不是闭式）的枚举量")


# ===========================================================================
# 5. C 类：两条【真的会失败】的块索引错法（★ 都**不报错**，只是结果错）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**环形块索引写错**（发哪块 / 累加到哪块）")
    print("=" * 78)
    print("  ★★★ 两条错法的**共同点**：块数、字节数、形状**全都对**、**一次异常都不抛** ——")
    print("      唯一的信号是**与参照的差**。★ 参照（单进程 `sum`）与模拟器**不共享代码**。")
    n_dev = specs.DEV_GRID[1]
    x = np.arange(n_dev * specs.RING_ELEMS, dtype=np.int64) \
        .reshape(n_dev, specs.RING_ELEMS)
    ref = ref_allreduce_sum(x)

    print()
    print("  ── C.1 ★★★ **发错块**：把 `(d−k)%%N` 写成 `(d+k)%%N` ──")
    good, mg = sim_ring_allreduce(n_dev, specs.RING_ELEMS)
    bad1, mb1 = sim_ring_allreduce(n_dev, specs.RING_ELEMS, corrupt="send")
    d1 = float(np.abs(good[0] - bad1[0]).max())
    print("     `max|diff|` = **%.1f**（**整数**差，不是舍入）；搬运量：%d vs %d（**不变**）"
          % (d1, vol_of(mg, n_dev)[0], vol_of(mb1, n_dev)[0]))
    print("     ★★★ **它【不报错】**：块号合法、形状合法、字节数合法 —— 错的是**哪个块去了哪**。")
    check("C.1", d1 > 0 and vol_of(mg, n_dev)[0] == vol_of(mb1, n_dev)[0],
          "★★★ **判据成立**：发错块 ⇒ `max|diff|` = **%.1f**（≫ 内容级阈 `1e-6`），"
          "★ 而**搬运量逐元素不变** ⇒ ★★ 「搬运量对拍」**看不见**这类错，"
          "「正确性对拍」才看得见" % d1,
          "**发错块没有让结果变** —— ★ 那说明参照与模拟器共用了一段代码（自证）")

    print()
    print("  ── C.2 ★★★ **累加到错误的目标**：把收到的加到 `(块+1)%%N` 上 ──")
    bad2, mb2 = sim_ring_allreduce(n_dev, specs.RING_ELEMS, corrupt="accum")
    d2 = float(np.abs(good[0] - bad2[0]).max())
    print("     `max|diff|` = **%.1f** ⇒ 与 `C.1` 的 **%.1f** %s"
          % (d2, d1, "**不同**" if abs(d2 - d1) > 0 else "**相同**"))
    print("     ★★★ **它【不报错】**：累加的目标是一块**存在的**块 —— 只是**不是该累加的那一块**。")
    check("C.2", d2 > 0 and abs(d2 - d1) > 0,
          "★★★ **判据成立**：累加错目标 ⇒ `max|diff|` = **%.1f**（≫ `1e-6`），"
          "且它与 `C.1` 的 **%.1f** **不同** ⇒ ★★ **两条错法是两种不同的失败**" % (d2, d1),
          "**累加错目标没有让结果变**，或它与 `C.1` 给同一个数（★ 两条错法可能是同一个）")


# ===========================================================================
# 6. 前哨：口径 / 阈值 / 标记自检（判据 D.1 / D.2 / D.2b / D.3）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 A/B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` / `numpy %s`。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**算法的代数**。")
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
    check("D.2", ASSERT_FAIL_MARK == "&& FAIL: collectives" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `&& FAIL: collectives`、且含 `FAIL`\n"
          "        ★★ `&&` 是 **shell 元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**。")
    check("D.2b", OTHER_THEME_MARKS == (">< FAIL: nccl", ",, FAIL: network-topology",
                                        "+= FAIL: overlap"),
          "★ 「不跨主题照抄」的**对照物**（本域另外三个主题的标记）逐位钉住；"
          "★ 四个标记**两两不互为前缀**")

    # ── D.3 ★ 口径常量与书目字段 ───────────────────────────────────────────
    check("D.3", specs.TOL_MACHINE == 1.0e-12
          and specs.MPICH_CITATION
          == ("Thakur, Rabenseifner, Gropp, Optimization of Collective Communication "
              "Operations in MPICH, IJHPCA 19(1):49-66, 2005")
          and specs.UNSOURCED == (),
          "★ `TOL_MACHINE` = %g；★★ **它的唯一消费者【就是这条 `D.3` 的钉住】** ——"
          "★ **本程序【不用】它做浮点边界**（`A.1` 用**精确相等** `rel == 0.0`，"
          "因为纯两段式的搬运量是**整数**）；★ 另两个程序的 `TOL_MACHINE` **确有**实质消费者"
          "（`02` 的 `B.2` / `03` 的 `B.1`）；"
          "★★ **[C] 级载体的书目字段用 `==` 逐字钉住**（★ 不用 `startswith` ——"
          "那抓不到「末尾追加一个 `X`」这类变异，`mutate.py` 的盲区③）；"
          "★ `UNSOURCED == ()` 一并钉住" % specs.TOL_MACHINE)

    # ── D.3b ★★★ **口径守卫**：档位必须是 2 的幂（否则递归倍增的 `XOR` 配对会跑飞）──
    #   ★★★ 它**必须跑在任何计算之前**（本库 `P23`）：把 `GATHER_GRID` 的某一档改成
    #     非 2 的幂时，模拟器会 `KeyError` ⇒ 那是**崩溃捕获**（不算真捕获）。
    #     有了这一条，同一个变异就变成**断言捕获**。
    pw = all((v & (v - 1)) == 0 for v in tuple(specs.DEV_GRID) + tuple(specs.GATHER_GRID))
    odd = (specs.GATHER_ODD & (specs.GATHER_ODD - 1)) != 0
    check("D.3b", pw and odd,
          "★★★ **口径守卫**：`DEV_GRID` 五档与 `GATHER_GRID` 四档**全是 2 的幂**"
          "（递归倍增的 `d XOR 2^k` 配对要求）；★ `GATHER_ODD = %d` **刻意不是**"
          "（它是 Bruck 那一条「任意 `p`」的档）⇒ 任一档被改成非 2 的幂，**本条先失败**"
          "（而不是让模拟器崩溃）" % specs.GATHER_ODD,
          "**档位的 2 的幂性质被破坏** —— ★ 先查 `DEV_GRID`/`GATHER_GRID`/`GATHER_ODD`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字钉住 · 逐元素钉住 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★ MPICH 的步数/时间引文（allgather 与广播）──────────────────
    check("D.5", specs.MPICH_BCAST_TREE_STEPS_QUOTE
          == "This communication takes a total of ⌈lg p⌉ steps."
          and specs.MPICH_BCAST_TREE_TIME_QUOTE == "Ttree = ⌈lg p⌉(α + nβ)."
          and specs.MPICH_AG_RING_STEPS_QUOTE == "the entire algorithm takes p − 1 steps."
          and specs.MPICH_AG_RING_TIME_QUOTE == "Tring = (p − 1)α + (p−1)/p · nβ."
          and specs.MPICH_AG_RECDUB_QUOTE == "all processes get all the data in lg p steps."
          and specs.MPICH_AG_RECDUB_TIME_QUOTE == "Trec dbl = lg p α + (p−1)/p · nβ."
          and specs.MPICH_AG_BRUCK_STEPS_QUOTE
          == "Both algorithms take ⌈lg p⌉ steps in all cases"
          and specs.MPICH_AG_BRUCK_TIME_QUOTE == "Tbruck = ⌈lg p⌉α + (p−1)/p · nβ.",
          "★★ **八条引文逐字钉住** —— ★★★ 特别注意**取整符号的三种形态**：\n"
          "        `Ttree` / `Tbruck` **有** `⌈lg p⌉`；`Trec dbl`（allgather）**是裸 `lg p`**；\n"
          "        `Tring` 里 `(p − 1)α` 的减号**两侧有空格**、`(p−1)/p` 的**没有**。\n"
          "        ★★ 载体 = 一手 PDF（`mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`），"
          "抓取日 2026-10-08；`pdftotext -layout` 提取形态见 `specs.py` 各条的 `# 来源：`",
          "**MPICH 的引文常量被动过** —— ★ 先查取整符号（`dlg pe` / 裸 `lg p`）")

    # ── D.6 ★★★ 同名不同指 + 三算法对照的式子 + 清单 ─────────────────────
    check("D.6", specs.MPICH_REDUCE_ALGOS == ("binomial tree", "recursive doubling",
                                              "recursive halving and doubling",
                                              "binary blocks", "ring")
          and specs.MPICH_REDUCE_ALGO_COUNT == 5
          and specs.MPICH_ALLREDUCE_RECDUB_TIME_QUOTE
          == "Trec−dbl = lg p α + n lg p β + n lg p γ."
          and specs.MPICH_RABENSEIFNER_TIME_QUOTE
          == "Trabenseifner = 2 lg p α + 2(p−1)/p · nβ + (p−1)/p · nγ."
          and specs.MPICH_BCAST_VDG_QUOTE.startswith("the message to be broadcast is first")
          and specs.MPICH_BRUCK_SCOPE_NOTE != "",
          "★★★ **reduce / allreduce【共用】的算法清单**逐元素钉住（5 个名字，`MPICH_REDUCE_ALGOS`；"
          "★ 论文原句是 `five different algorithms for implementing allreduce and reduce` —— "
          "**两原语共列**）+\n"
          "        ★★★ **同名不同指**：`MPICH_ALLREDUCE_RECDUB_TIME_QUOTE`（allreduce，"
          "带宽项 `n lg p`）与 `MPICH_AG_RECDUB_TIME_QUOTE`（allgather，带宽项 `(p−1)/p · n`）"
          "**同名 `Trec dbl`** —— 差一个 `lg p`；\n"
          "        ★★ Rabenseifner 的式子（`2 lg p` 步 + `2(p−1)/p · nβ`）"
          "与 Van de Geijn 的**两段式**散文引文一并钉住；\n"
          "        ★★★ **Bruck【不是】allreduce 的算法**（`MPICH_BRUCK_SCOPE_NOTE`）",
          "**算法清单或两条 allreduce 式子被动过** —— ★ 特别注意别把两条 `Trec dbl` 合并")

    # ── D.7 ★★★ 逐元素钉住（★ 只钉「整体」会让元素级变异漏网）────────────
    check("D.7", specs.DEV_GRID == (2, 4, 8, 16, 64)
          and specs.SAT_GRID == (2, 4, 8, 64, 512, 4096)
          and specs.GATHER_GRID == (2, 4, 8, 16)
          and specs.RING_ELEMS == 64 and specs.GATHER_ELEMS == 32
          and specs.GATHER_ODD == 6 and specs.STAGE_ELEMS == 8,
          "★★★ **四个元组逐元素 + 三个标量**钉住（`DEV_GRID` 五档 · `SAT_GRID` 六档 ·"
          " `GATHER_GRID` 四档 · `GATHER_ODD`；标量 `RING_ELEMS` / `GATHER_ELEMS` / `STAGE_ELEMS`）"
          "—— ★ 只钉「整体」会让**元素级**变异漏网（本库实测过：一次报出 58 条漏网）",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★ 独立性线索（源码级）—— ★★ **遍历全部 6 个模拟器**（不再只审一个）──
    sims = (sim_ring_allreduce, sim_allgather_ring, sim_allgather_recdub,
            sim_allgather_bruck, sim_broadcast_binomial, sim_allreduce_rabenseifner)
    srcs = {inspect.getsource(f) for f in sims + (ref_allreduce_sum,)}
    call_tables = {}
    for f in sims:
        call_tables[f.__name__] = {n.func.id for n in ast.walk(
            ast.parse(textwrap.dedent(inspect.getsource(f))))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_ref = {n.func.id for n in ast.walk(
        ast.parse(textwrap.dedent(inspect.getsource(ref_allreduce_sum))))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：**六个模拟器 + 单进程参照**的调用表 ──")
    for _name, _tbl in call_tables.items():
        print("     `%s()` 里调用的函数：%s"
              % (_name, ", ".join(sorted(_tbl)) if _tbl else "**空集**"))
    print("     `ref_allreduce_sum()`（单进程参照）里调用的函数：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    cf_hits = sorted({c for _tbl in call_tables.values() for c in _tbl if c.startswith("cf_")})
    print("     ⇒ ★★ **六个模拟器都不调用任何 `cf_*`（闭式）** —— `CONTRIBUTING:232` 的"
          "「参照不得用被测的输出」在此**在源码层面成立**。")
    print("        ★★ **本轮修正**：原写法只审 `sim_ring_allreduce` 一个，却把结论写成"
          "「全体」；现在**逐个打印六个的调用表**。")
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，")
    print("        **不证明**判据有物理信息量 —— 后者由 `C.1`/`C.2` 的**行为**兜底。")
    check("D.8", not cf_hits and len(srcs) == 7
          and all(n not in calls_ref for n in call_tables),
          "★★ **源码级线索成立**：**七个函数两两不同源**；★★ **六个模拟器的调用表里都没有**"
          " `cf_*`（★ **全体**结论，不是只审一个）⇒ ★ **参照不调用被测**（源码层面）",
          "**参照与闭式互相调用** —— A 类可能退化成自证")

    # ── D.9 ★★ 浮点边界与 `UNSOURCED` ──────────────────────────────────────
    check("D.9", specs.UNSOURCED == ()
          and cf_ring_volume(2, specs.RING_ELEMS) == specs.RING_ELEMS
          and abs(cf_ring_volume(specs.SAT_GRID[-1], 1.0) - 2.0) < 1.0e-3,
          "★★ `UNSOURCED` 留空（★ 0 个叶子 ⇒ 不产生变异体）；★ 两条**闭式边界**："
          "① `N = 2` 时 `V(2) == S`（**恰好**，整数）；② `N = %d` 时 `V(N)/S` 与 `2` 的差 `< 1e-3`"
          "（★ **不是**机器精度 —— 饱和的残差是 `2/N`，不是 `eps`）" % specs.SAT_GRID[-1],
          "**`UNSOURCED` 不再是空的，或闭式的两条边界不成立**")


# ===========================================================================
# 8. 汇总 / 主程序
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
    print("★ 算法族的依据在 `specs.py` 的 `# 来源：`（Thakur / Rabenseifner / Gropp 2005）；")
    print("  本程序只核对「消息序列模拟 vs 闭式」与「两条块索引错法真的会失败」。")
    print("★ **不是 NCCL，也不是真机**：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_algorithm_family.py —— 集合通信的算法族：步数与每卡搬运量")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**算法的代数**。")
    print("  ★★★ **参照是【真消息序列模拟】**（逐步枚举 `(phase, sender, receiver, block, n_elem)`），")
    print("      **不是**把闭式用循环写一遍（★ 规划 `§D` 的 `D2` 犯过那个错，那是 `P12`）。")
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
