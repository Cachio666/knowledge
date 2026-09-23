#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_column_row.py —— TP 的**切法**：列并行切输出维、行并行切输入维，以及【配对约束】。

配套文档：`01-column-and-row.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**，
> **不是**任何关于真实集群性能的结论。

★★★ **本主题的头号风险是「模拟器同时是被测对象」**（计划 §1.3.3）：
  本程序里的 `all_reduce` 就是 `sum()`、`all_gather` 就是 `concat`
  ⇒ 「切分后 == 单卡」这类判据**极易写成恒真**。★ 本程序的处置有三条：
    · **纯搬运处的等价性判据【不当 A 类】**（`D.10` 把这件事变成一个**机械的检查**：
      列并行与单卡的差**恒为 `0.0`** ⇒ 它只重述定义）；
    · **每一条 A 类判据都带一个「把参照实现故意改错」的守卫**（`CONTRIBUTING:235`）——
      改错之后**该断言必须失败**（`A.1g` / `A.4g` / `A.3` 的跨格比较）；
    · **参照实现不得调用被测实现的任何函数** —— 本程序给被测函数装了**调用计数器**
      （`_CALLS`），参照跑完后必须**恰好为 `0`**（`D.7c`），且**先证明这个计数器非空**
      （被测路径跑一次必须 > 0；否则那是一条恒真的「没调用」）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  > 答「不会」⇒ 它至少钉住了某段逻辑；**再问下一步**：那段逻辑**是不是就是断言自己写的那行**。
  ★ 本程序的答案落点：`A.1` 若删掉 `mlp_column_row` ⇒ 切分结果无从算起（不会通过）；
    `A.2`/`A.3`/`A.4`/`C.1`/`C.3` 同理。★ 而 `D.10` **正是一条「恒真」的机械检查** ——
    它**故意**去证明某条判据恒真，并因此**不把它当 A 类**用。

★★ **覆盖边界（必须写明 —— 计划 §2 的第 3 条）**：本程序的 A 类守的是
  「**切分做对了没有**」（切在哪个轴、两块 GEMM 的配对对不对、索引有没有错位），
  **不守**「算子本身对不对」（那由 `numpy` 保证）⇒ **不得声称 A 类验证了更多东西**。
  ★★ 且 **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：本程序的 `C.3` 断言的是
  「**形状相同却数值不同**」这个**失败**，不是「一致」这个成功。

★★ 一条**口径声明**（它必须在正文里也写明）：本程序用一个**确定的非线性**
  `relu(x) = max(x, 0)` 代替论文里的 GeLU —— **机制只要求「非线性」**，
  不要求具体形状（`A.2` 用一个**线性对照**证明：不可交换性**来自非线性本身**）。
  ★★★ **那是本库的替身，不是「GeLU 在本机上的实测」** ⇒ 不得据它声称任何关于 GeLU 的结论。

★★★ **本主题的禁区（回链，一行都不重写）**：
  `10-foundations/performance-models/02-amdahl-and-scaling.md:223` 已经逐字写完了 TP 的**三条**
  （通信发生在**每一层内部** · 通信量与**激活**成正比而不是与参数量成正比 ·
  **只在高带宽互联上可用 / 通常限于单节点 8 卡**，以及「规模上限由节点内互联决定」）
  ⇒ **本程序一行都不重写那三条**（否则是 `P2`「同一个载体重复」）。
  ★ 本程序只写它**没写**的：**切法本身**（列/行的**配对**）与**「列并行为什么非得在前面」的机制**。
  ★★ 另有两条边界：**集合通信原语的算法与复杂度**（`V(N) = 2(N−1)/N × S`、`S` 的饱和）
  → `110-communication` 与 `performance-models/02:209`；**GEMM 的 tiling / 分块 / 层级**
  → `40-kernels/gemm/` 与 `30-gpu-programming/cutlass/02-hierarchical-tiling.md`
  （本程序只写「**矩阵怎么切到设备上**」，不写「一个 GEMM 怎么切到 block 上」）。
  ★★ **注意力变体的结构**（MHA / MQA / GQA 的定义、KV head 与 Q head 的配比）
  → `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`（本程序不涉及）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ MLP 的「列 + 行」组合 == 单卡一次算完（**机器精度档**）；
  · `A.1g` ★★ **守卫**：把参照实现的第二层权重**故意改错**（行序反转）⇒ 同一条断言**必须失败**；
  · `A.2`  ★★★ **非线性不可交换**：`relu(Σ_i x_iW_i) ≠ Σ_i relu(x_iW_i)`（实测非零）；
             ★ **线性对照**：把非线性换成**恒等** ⇒ 两边**逐位相同**（`0.0`）
             ⇒ **不可交换性来自非线性本身**，这就是「列并行必须在前面」的算术证据；
  · `A.3`  ★★ **配对约束**：四格（第一层切法 × 第二层切法）的**结果全对**，
             而**块内通信次数**是 `{1, 2, 2, 2}` —— 只有「列 + 行」是 `1`
             ⇒ ★★ **配对是【代价】约束，不是正确性约束**（这条结论本身是可证伪的）；
  · `A.4`  ★ 行并行（单独一层）== 单卡（机器精度档）+ `A.4g` 守卫；
  · `B.1`  ★★ 局部部分和的量级：**每一片都不是零、也不是全量**（实测占比 ~`0.5`）
             ⇒ 证伪「一片是零、一片是全」的直觉；
  · `B.2`  ★ 每层 all-reduce **次数**：配对后 `2` 个块各 `1` 次 ⇒ 前向 `2` 次，
             与 [C] `1909.08053` 的 `two all-reduces in the forward path` 对拍；
             配对改成另外三种 ⇒ 每方向 `4` 次（**两倍**）；
  · `C.1`  ★★ **忘记 all-reduce**（只拿一片的部分和当结果）⇒ 差 ~`2.07`（**真的会错**）；
  · `C.2`  ★★ **切错轴** ⇒ `ValueError`（形状不匹配）⇒ 「它必须抛错」是一条判据；
  · `C.3`  ★★ **交错切**（形状对、索引错位）⇒ 差 ~`3.35` ⇒ 形状检查**挡不住**它。

用法：
    python3 01_column_row.py
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

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立 / 守卫成立 / "
                                "⇒ 不是恒真的」的措辞是**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260951              # ★ 与同域 data-parallel 的 20260941 **不同**
B_SIZE = 2                   # micro-batch 大小（`b`）
S_LEN = 8                    # 序列长度（`s`）
D_IN = 64                    # 输入维
D_OUT = 32                   # 输出维（= 论文记号里的 `h`，激活的宽度）
D_FF = 128                   # MLP 中间维（`4 × D_OUT`）
TP = 4                       # TP 度（`t`）
TP_GRID = (1, 2, 4, 8)       # `A.4`/`B.1` 扫描用的 TP 度
# ── 阈值 ────────────────────────────────────────────────────────────────────
TOL_MACHINE = 1e-12          # 「机器精度档」（`A.1`/`A.3`/`A.4` 的等价性）
TOL_GUARD = 1e-6             # 守卫：改错的参照必须**远大于**它
TOL_ERR = 1e-3               # C 类「真的会错」的下限（实测差比它大三个数量级）
SHARE_LO = 0.35              # `B.1` 局部部分和占比的下界（证伪「一片是零」）
SHARE_HI = 0.75              # `B.1` 的上界（证伪「一片是全量」）
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   ⇒ 本程序的浮点常量都 `< 2^53`（最大的阈值是 `1e-3`）⇒ 每一个叶子都**可变异**。
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 十进制字面量：盲区④）


# ===========================================================================
# 2. 被测路径（列 / 行并行、MLP 组合、四格配对）
# ===========================================================================
def relu(t):
    """本程序的**非线性**（论文里是 GeLU）—— 它**真的计入调用计数**。

    ★★ **口径声明**：机制只要求「非线性」，不要求具体形状 ⇒ 本程序用**确定的** `relu`。
      ★ 那是本库的**替身**，**不是**「GeLU 在本机上的实测」。
    """
    _CALLS["relu"] += 1
    return np.maximum(t, 0.0)


def all_reduce_sum(parts):
    """**被测**：`all_reduce` = **显式的张量搬运**（逐元素求和）。

    ★★ 依据是 [B] NCCL 的逐字结果语义（`data-parallel/01` 已引）与
      [C] `1909.08053` 的 `g` 算子名字；本域的知识对象是**切分策略**，
      **不是**通信原语的实现（通信算法的正文在 `110-communication`）。
    """
    _CALLS["all_reduce_sum"] += 1
    return sum(parts)


def all_gather_concat(shards):
    """**被测**：`all_gather` = **显式的张量搬运**（沿最后一维拼接）。"""
    _CALLS["all_gather_concat"] += 1
    return np.concatenate(shards, axis=-1)


def column_parallel_linear(x, w, tp):
    """**被测**：列并行 —— `W` 按**输出维**切（`axis=1`），`x` **完整复制**。

    ★ 每卡算出一段**输出**，再 `all_gather`（`concat`）拼回来。
      ★★ 这条路径**不改变任何一个元素的求和顺序**（每个输出元素只由一张卡算，
        且算它用的那次点积与单卡完全一样）⇒ 它是**纯搬运**（`D.10` 会把它变成一条检查）。
    """
    _CALLS["column_parallel_linear"] += 1
    return all_gather_concat([x @ wi for wi in np.split(w, tp, axis=1)])


def row_parallel_partials(x, w, tp):
    """**被测**：行并行 —— `x` 按**输入维**切（`axis=-1`）、`W` 按**行**切（`axis=0`）。

    ★ 每卡拿到的是**局部部分和**（形状**就是**输出形状）⇒ 相减/相加是合法运算。
    """
    _CALLS["row_parallel_partials"] += 1
    return [xi @ wi for xi, wi in zip(np.split(x, tp, axis=-1), np.split(w, tp, axis=0))]


def row_parallel_linear(x, w, tp):
    """**被测**：行并行 = 局部部分和 + **一次 all-reduce**。

    ★★ 这条路径**改变了求和顺序**（`tp` 次部分和再相加 vs 一次点积）
      ⇒ 它与单卡的差**有信息量**（不是「纯搬运」）。
    """
    _CALLS["row_parallel_linear"] += 1
    return all_reduce_sum(row_parallel_partials(x, w, tp))


def mlp_column_row(x, w1, w2, tp):
    """**被测**：MLP 的「**列 + 行**」组合（Megatron 的标准切法）。

    ★ 第一层 GEMM **列并行**（切 `W1` 的输出维）⇒ 每卡拿到激活的一段，**非线性本地算**；
      第二层 GEMM **行并行**（切 `W2` 的输入维，恰好与上一段的宽度**配对**）
      ⇒ 局部部分和，最后**一次** all-reduce。
    ★★ 这条路径**改变了求和顺序**（第二层的部分和相加）⇒ 差落在机器精度档。
    """
    _CALLS["mlp_column_row"] += 1
    w1s = np.split(w1, tp, axis=1)          # 列并行：切输出维
    w2s = np.split(w2, tp, axis=0)          # 行并行：切输入维（与上一段的宽度配对）
    return all_reduce_sum([relu(x @ a) @ b for a, b in zip(w1s, w2s)])


def mlp_pair(x, w1, w2, tp, first, second):
    """**`A.3` 的四格**：第一层 GEMM 的切法 × 第二层 GEMM 的切法。

    `first` / `second` 取 `"col"`（列并行）或 `"row"`（行并行）。
    返回 `(结果, 块内通信总次数, 非线性【之前】的通信次数)`。

    ★★ **本函数是「机制的可执行形态」**，两个计数的含义：
      · **非线性之前的通信** = [C] `1909.08053` 说的那个 `a synchronization point before the
        GeLU function` —— 行并行先做时，部分和**必须先规约**才能在非线性上用
        （因为 `relu(Σ) ≠ Σ relu`，见 `A.2`）；
      · **块内通信总次数** = 两块 GEMM 之间与之后的所有集合通信。
    ★ **两个计数都是【代码结构】的计数，不是实测的通信次数** ——
      「通信发生了几次」在单进程模拟里**测不出来**（计划 §1.3.2），
      它的**次数依据**是 [C] 的外部事实（`1909` 的 Figure 4）与 [C] 的 `removes a
      synchronization point`；本函数只把那个**结构**摆成一个可数的东西。
    """
    _CALLS["mlp_pair"] += 1
    comm_total = 0
    comm_before_nl = 0
    if first == "col":
        a_shards = [x @ a for a in np.split(w1, tp, axis=1)]      # 切输出维：免通信
    else:
        a_shards = [xi @ ai for xi, ai in zip(np.split(x, tp, axis=-1),
                                              np.split(w1, tp, axis=0))]
        # ★★ 非线性**不可交换** ⇒ 行并行先做时，这里**必须**先规约（那就是那个「同步点」）
        a_shards = [all_reduce_sum(a_shards)]
        comm_total += 1
        comm_before_nl += 1
    ys = [relu(v) for v in a_shards]
    if second == "row":
        w2s = np.split(w2, tp, axis=0)
        if first == "col":
            parts = [y @ b for y, b in zip(ys, w2s)]     # 配对：宽度就是行块
        else:
            parts = [y @ b for y, b in zip(np.split(ys[0], tp, axis=-1), w2s)]  # 本地切片
        out = all_reduce_sum(parts)
        comm_total += 1
    else:
        w2s = np.split(w2, tp, axis=1)
        if first == "col":
            full = all_gather_concat(ys)                 # 列并行的第二层要**全量**激活
            comm_total += 1
        else:
            full = ys[0]
        out = all_gather_concat([full @ b for b in w2s])
        comm_total += 1
    return out, comm_total, comm_before_nl


def interleaved_split(x, tp):
    """`C.3` 的**错误切法**：按**交错**（`i::tp`）切输入维 —— 形状和分块切**完全一样**。

    ★★ 它是一个**索引错位**的错法：形状检查**挡不住**它（`B_SIZE, S_LEN, D_IN` 不变），
      只有数值判据能抓（差 ~`3.35` ⇒ 它不是精度问题）。
    """
    _CALLS["interleaved_split"] += 1
    return np.concatenate([x[..., i::tp] for i in range(tp)], axis=-1)


# ===========================================================================
# 3. 独立参照（**不调用**被测路径的任何函数）
# ===========================================================================
def reference_linear(x, w):
    """**独立参照**：单卡一次算完一层线性层（`x @ w`）。

    ★★★ **独立性**：这条路径**不调用** `column_parallel_linear` / `row_parallel_partials` /
      `row_parallel_linear` / `all_reduce_sum` / `all_gather_concat` 中的任何一行
      （`D.7b` 用 AST 调用表、`D.7c` 用调用计数**两条**证据钉住它）。
    """
    return x @ w


def reference_mlp(x, w1, w2):
    """**独立参照**：单卡一次算完两层（`relu(x @ w1) @ w2`）。

    ★★ **非线性是内联写的**（`np.maximum(..., 0.0)`）—— 它**不是**调用被测的 `relu`
      ⇒ 「参照不调用被测」这条不是靠读代码相信的，`D.7c` 会数。
      ★ 它与被测路径的**浮点路径不同**：被测是 `tp` 次矩阵乘 + `tp−1` 次加法 + 一次除法以外的
        一切，参照是**一次**矩阵乘 —— 所以两者的差**有信息量**。
    """
    return np.maximum(x @ w1, 0.0) @ w2


def toy_batch(b, s, d_in, d_out, d_ff, seed):
    """给定形状造一份确定的数据（★ 只用于把**形状**摆出来，不训练任何东西）。"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((b, s, d_in)),
            rng.standard_normal((d_in, d_ff)) * 0.1,
            rng.standard_normal((d_ff, d_out)) * 0.1,
            rng.standard_normal((d_in, d_out)) * 0.1)


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

    ★★ 返回 `None` 表示**文档不在**（`mutate.py` 的副本目录里只有 `*.py`）⇒ 调用方跳过并打印。
      ★ 那一支**不会**造成漏网：同一个字面量在 `D.5` 里另有**逐位**钉住。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    if not os.path.exists(path):
        return None
    return needle in open(path, encoding="utf-8").read()


# ===========================================================================
# 5. A 类：切法与配对（判据 A.1 / A.1g / A.2 / A.3 / A.4）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 切法：列切输出维 / 行切输入维 / MLP 的「列 + 行」配对 / 非线性的不可交换性")
    print("=" * 74)
    x, w1, w2, w = toy_batch(B_SIZE, S_LEN, D_IN, D_OUT, D_FF, SEED)
    print("  形状：`x=%s` · `W1=%s` · `W2=%s` · `W=%s` · `TP=%d`"
          % (x.shape, w1.shape, w2.shape, w.shape, TP))

    # ── A.1 ★★ MLP 的「列 + 行」== 单卡 ────────────────────────────────────
    print()
    print("  ── A.1 ★★ MLP：第一层**列并行**、第二层**行并行** vs 单卡一次算完 ──")
    y_split = mlp_column_row(x, w1, w2, TP)
    y_single = reference_mlp(x, w1, w2)
    diff = float(np.abs(y_split - y_single).max())
    scale = float(np.abs(y_single).max())
    # ★★ 守卫：**把参照实现的第二层权重故意改错**（行序反转）⇒ A.1 必须失败
    y_bad = reference_mlp(x, w1, w2[::-1, :])
    diff_bad = float(np.abs(y_split - y_bad).max())
    print("     ★★★ [C] `%s` §3 逐字（**本主题的机制理由，不是本库编的从句**）：" % specs.MEGATRON_TP_ARXIV_ID)
    print("        `%s`" % specs.MEGATRON_GELU_NONLINEAR_QUOTE)
    print("        `%s`" % specs.MEGATRON_GELU_SYNC_QUOTE)
    print("        `%s` … `%s`" % (specs.MEGATRON_COL_SPLIT_LEAD_QUOTE,
                                  specs.MEGATRON_COL_REMOVES_SYNC_QUOTE))
    print("     `max|列+行 − 单卡|` = **%.3e**（`y` 的量级 %.3f）" % (diff, scale))
    print("     ★★ **这不是「纯搬运」**：被测路径是 `TP` 次矩阵乘 + `TP−1` 次加法"
          "（第二层的部分和），参照是**一次**矩阵乘 ⇒ 差落在机器精度档（不是 `0.0`）。")
    print("     ★★ **守卫**（`CONTRIBUTING:235`）：把参照的第二层权重行序反转 ⇒ "
          "差 = **%.3e** ⇒ 同一条断言**必须失败**" % diff_bad)
    check("A.1", diff <= TOL_MACHINE * max(scale, 1.0) and np.isfinite(diff),
          "★★ **判据成立**：MLP 的「列 + 行」== 单卡（差 `%.3e` ≤ 机器精度档）"
          "⇒ 这一刀切对了（切在输出维、第二层与它配对切在输入维）" % diff)
    check("A.1g", diff_bad > TOL_GUARD,
          "★★ **守卫成立**：故意改错的参照给出 `%.3e` > `%g` ⇒ `A.1` **能失败**"
          "（它守的是「切在哪个轴、配对对不对、索引有没有错位」）" % (diff_bad, TOL_GUARD))

    # ── A.2 ★★★ 非线性不可交换 + 线性对照 ──────────────────────────────────
    print()
    print("  ── A.2 ★★★ **非线性不可交换** ⇒ 这就是「列并行必须在前面」的算术证据 ──")
    xs = np.split(x, TP, axis=-1)
    ws = np.split(w, TP, axis=0)
    partials_w = [xi @ wi for xi, wi in zip(xs, ws)]
    ref = reference_linear(x, w)                     # ★ **独立参照**：单卡一次点积
    # ★★ 四格：**函数在规约的哪一侧** × **函数是不是非线性**
    #    口径：`after` = 先把部分和加起来、再上函数（= **列并行先做**）
    #          `before` = 先给每份部分和上函数、再相加（= **行并行先做**）
    #    ★★★ 每一格都**与独立参照比**（`3·(XW)` 或 `relu(XW)`），
    #        而**不是**两个格子互比 —— 后者会退化成「同一个表达式和自己比」。
    d_nl_after = float(np.abs(np.maximum(sum(partials_w), 0.0) - np.maximum(ref, 0.0)).max())
    d_nl_before = float(np.abs(sum(np.maximum(p, 0.0) for p in partials_w)
                                - np.maximum(ref, 0.0)).max())
    d_lin_after = float(np.abs(3.0 * sum(partials_w) - 3.0 * ref).max())
    d_lin_before = float(np.abs(sum(3.0 * p for p in partials_w) - 3.0 * ref).max())
    cross_nl = float(np.abs(np.maximum(sum(partials_w), 0.0)
                            - sum(np.maximum(p, 0.0) for p in partials_w)).max())
    cross_lin = float(np.abs(3.0 * sum(partials_w)
                             - sum(3.0 * p for p in partials_w)).max())
    print("     ★★ **口径**：`after` = **函数在规约之后**（**列并行先做**）· "
          "`before` = **函数在规约之前**（**行并行先做**）；")
    print("     ★★ 每一格都与**独立参照** `φ(XW)` 比（`XW` 是单卡的一次点积），"
          "**不是两格互比** ——")
    print("        后者会退化成「同一个表达式和自己比」（本行**曾**这么写过，2026-09-23 已改）。")
    print("     ★★★ **一处必须写明的性质**：`A.2` 是**机制的可执行说明**，"
          "**不是**「对被测实现的判据」——")
    print("        它检查的是「非线性不可交换」这个**数学事实**，"
          "被测的那段逻辑**就在这几行里**。")
    print("        ⇒ 按 `§1.3.3` 的自查问句，它答「不会通过」（删掉就无从比），"
          "但**第二问**（「那段逻辑是不是就是断言自己写的那行」）的答案是**是**")
    print("     `φ = 3u`（**线性**）：`after` 与参照差 **%.3e** · `before` 与参照差 **%.3e**"
          % (d_lin_after, d_lin_before))
    print("        ⇒ ★★ **两种位置都对**（都在机器精度档内）—— 这就是**对照**。")
    print("     `φ = relu`（**非线性**）：`after` 与参照差 **%.3e** · `before` 与参照差 "
          "**%.6f**" % (d_nl_after, d_nl_before))
    print("        ⇒ ★★★ **只有「函数在规约之后」那一支是对的**。")
    print("     **两格互差**：线性 **%.3e** · 非线性 **%.6f**" % (cross_lin, cross_nl))
    print("     ⇒ ★★★ **不可交换性来自【非线性本身】**：换成线性 `φ = 3u` 之后，"
          "两种位置**都在机器精度档内**（且互差 `%.1e`，**不是同一个表达式**）。" % cross_lin)
    print("     ⇒ 这与 [C] 的逐字一致：`%s`" % specs.MEGATRON_GELU_SYNC_QUOTE)
    check("A.2", d_nl_before > TOL_GUARD and d_nl_after <= TOL_MACHINE * max(
        float(np.abs(np.maximum(ref, 0.0)).max()), 1.0)
        and d_lin_after <= TOL_MACHINE * max(float(np.abs(ref).max()), 1.0)
        and d_lin_before <= TOL_MACHINE * max(float(np.abs(ref).max()), 1.0),
        "★★★ **判据成立**（四格一起）：非线性下「函数在规约**之前**」与参照差 **%.6f**（≫ 阈值），"
        "而「之后」差 **%.1e**；★ **线性对照**（`φ = 3u`）下**两种位置都 ≤ 机器精度档** "
        "⇒ ★★ **列并行的位置由「非线性在哪里」决定**。"
        "★★ **这条判据【能失败】的那个变异是哪一个（已实测）**：把「行并行先做」那一格"
        "**改成与参照同构**（即把错法改对）⇒ 本断言当场失败 ✓。"
        "★★★ **而一个【杀不死它】的变异也要写明**：把 `partials_w` 的 `np.maximum(p, 0.0)` "
        "换成 `3.0 * p`（只换一边）**不会**让它失败 —— 因为那时两边**仍然不等**"
        "（一边上了非线性、一边没有）⇒ ★ 这正说明本判据测的是「**两边是否同构**」，"
        "不是「哪一边用了什么函数」" % (d_nl_before, d_nl_after))

    # ── A.3 ★★ 配对约束：四格 ─────────────────────────────────────────────
    print()
    print("  ── A.3 ★★ **配对约束**：四格（第一层切法 × 第二层切法）都算对，差别在**通信** ──")
    cells = []
    for first, second in (("col", "row"), ("row", "row"), ("col", "col"), ("row", "col")):
        out, ct, cb = mlp_pair(x, w1, w2, TP, first, second)
        dd = float(np.abs(out - y_single).max())
        cells.append((first, second, ct, cb, dd))
        print("     `%-3s + %-3s` ⇒ 块内通信 **%d** 次（非线性之前 **%d** 次）· "
              "与单卡的差 **%.3e**" % (first, second, ct, cb, dd))
    all_ok = all(dd <= TOL_MACHINE * max(scale, 1.0) for *_, dd in cells)
    totals = [ct for _, _, ct, _, _ in cells]
    before = [cb for _, _, _, cb, _ in cells]
    print("     ⇒ ★★ **四格的结果【全对】**（都在机器精度档）⇒ **配对是【代价】约束，"
          "不是正确性约束**。")
    print("     ⇒ ★★ 块内通信次数 `{1, 2, 2, 2}`、非线性**之前**的通信 `{0, 1, 0, 1}`：")
    print("        · 只有 **列 + 行** 的总次数是 `1` —— 且那一次在**块的出口**（不在非线性之前）；")
    print("        · **列 + 列** 的总次数也是 `2`，但它的那次通信在**非线性之后**"
          "（一次 all-gather 把各段的激活拼回全量）⇒ ★ **「有没有同步点在非线性之前」"
          "与「块内一共几次通信」是两件事**，四格正好把它们分开。")
    print("     ⇒ 「行并行先做」的那两格**都在非线性之前多一次通信** —— 与 [C] 逐字一致：")
    print("        `%s`" % specs.MEGATRON_GELU_SYNC_QUOTE)
    print("     ⇒ 两个块（attention / MLP）各 `1` 次 ⇒ 前向 `2` 次 —— 与 [C] 逐字对拍：")
    print("        `%s`" % specs.MEGATRON_TWO_ALLREDUCE_QUOTE)
    check("A.3", all_ok and totals == [1, 2, 2, 2] and before == [0, 1, 0, 1]
          and totals[0] == min(totals) and before[0] == 0,
          "★★ **判据成立**（五个条件）：① 四格结果**全对**；② 块内通信次数 == `{1,2,2,2}`；"
          "③ 非线性之前的通信 == `{0,1,0,1}`；④ 「列 + 行」是**唯一**的最小值；"
          "⑤ 它在非线性之前**没有**通信 ⇒ ★★ **「列并行后面必须跟一个行并行」是一条"
          "【代价】约束**（只有这一对把块内通信压到 `1` 次且不在非线性前加同步点），"
          "**不是正确性约束**。"
          "★ 这条判据**能失败**：把某一格的切法配错（例如让第二层也用列并行）⇒ ② 立刻不成立")

    # ── A.4 ★ 行并行（单独一层）== 单卡 ────────────────────────────────────
    print()
    print("  ── A.4 ★ 行并行（单独一层）vs 单卡 —— 与列并行的**误差档位**不同 ──")
    yrow = row_parallel_linear(x, w, TP)
    ycol = column_parallel_linear(x, w, TP)
    yref = reference_linear(x, w)
    d_row = float(np.abs(yrow - yref).max())
    d_col = float(np.abs(ycol - yref).max())
    d_bad4 = float(np.abs(row_parallel_linear(x, w, TP) - reference_linear(np.roll(x, 1, axis=-1), w)).max())
    print("     行并行 `max|diff|` = **%.3e**（改了求和顺序 ⇒ 非零）" % d_row)
    print("     列并行 `max|diff|` = **%.3e**（纯搬运 ⇒ **恒为 0**，见 `D.10`）" % d_col)
    print("     ★★ **同一个 TP 度、同一份数据，误差档位不同** —— 这一条是本主题主线的微观证据：")
    print("        「切在哪个轴」决定的不是精度，而是**这一刀改不改变求值顺序**。")
    check("A.4", d_row <= TOL_MACHINE * max(float(np.abs(yref).max()), 1.0) and d_row > 0.0,
          "★★ **判据成立**：行并行 == 单卡（差 `%.3e`），**且这个差【不是 `0`】**"
          "（它是「改变了求值顺序」的那一类）" % d_row)
    check("A.4g", d_bad4 > TOL_GUARD,
          "★★ **守卫成立**：把参照的输入**故意改错**（列循环移位）⇒ `%.3e` > `%g` ⇒ "
          "`A.4` **能失败**" % (d_bad4, TOL_GUARD))


# ===========================================================================
# 6. B 类：代价模型 —— 局部部分和的量级、每层的 all-reduce 次数
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：局部部分和的量级（不是「一片零一片全」）· 每层的 all-reduce 次数")
    print("=" * 74)
    x, w1, w2, w = toy_batch(B_SIZE, S_LEN, D_IN, D_OUT, D_FF, SEED)

    # ── B.1 ★★ 局部部分和的量级 vs 全局 ───────────────────────────────────
    print("  ── B.1 ★★ 局部部分和的**量级**：「一片是零、一片是全」这个直觉成立吗？ ──")
    parts = row_parallel_partials(x, w, TP)
    g_full = float(np.linalg.norm(reference_linear(x, w)))
    norms = [float(np.linalg.norm(p)) for p in parts]
    shares = [v / g_full for v in norms]
    zeroed = [0.0] * len(parts)
    shares_zero = [v / g_full for v in zeroed]
    print("     各卡局部部分和的 Frobenius 范数：" + "  ".join("%.6f" % v for v in norms))
    print("     合并后（单卡）的范数 = **%.6f**" % g_full)
    print("     占比：" + "  ".join("%.6f" % v for v in shares) +
          "（均值 **%.6f**）" % (sum(shares) / len(shares)))
    print("     ★★ **判据的两侧**（这一条是**两侧**的，不是「大于」）：")
    print("        · 下界 `%.2f`：证伪「**一片是零**」—— 每一片都真的贡献了可观的一份；"
          % SHARE_LO)
    print("        · 上界 `%.2f`：证伪「**一片是全量**」—— 没有哪一片能单独代表结果。" % SHARE_HI)
    print("     ★ 对照（**把一片置零**）：占比 = %.1f ⇒ 会**落在下界之外**（这就是守卫）"
          % shares_zero[0])
    check("B.1", all(SHARE_LO <= v <= SHARE_HI for v in shares)
          and not any(SHARE_LO <= v <= SHARE_HI for v in shares_zero),
          "★★ **判据成立**（两侧）：`TP=%d` 片的占比全在 `[%.2f, %.2f]`（实测均值 `%.6f`）"
          "⇒ **既不是「某一片没贡献」、也不是「某一片代表全部」**；"
          "★ 守卫：把一片置零 ⇒ 占比 `%.1f` **落在区间外** ⇒ 这条判据能失败"
          % (TP, SHARE_LO, SHARE_HI, sum(shares) / len(shares), shares_zero[0]))

    # ── B.2 ★ 每层的 all-reduce 次数 ───────────────────────────────────────
    print()
    print("  ── B.2 ★ 每层 all-reduce 的**次数**：配对 `2` 次 vs 不配对 `4` 次（前向） ──")
    per_block = mlp_pair(x, w1, w2, TP, "col", "row")[1]        # 配对时的块内次数
    per_block_other = mlp_pair(x, w1, w2, TP, "row", "row")[1]  # 不配对时的块内次数
    blocks = 2      # 一个 transformer 层 = 1 个 attention 块 + 1 个 MLP 块
    fwd_paired = blocks * per_block
    print("     一个 **MLP 块**在「列 + 行」配对下：块内 **%d** 次 all-reduce" % per_block)
    print("     一个 transformer 层 = `%d` 个块（attention + MLP）⇒ 前向 **%d** 次"
          % (blocks, fwd_paired))
    print("     ★★ [C] `%s` 逐字：`%s`" % (specs.MEGATRON_TP_ARXIV_ID,
                                            specs.MEGATRON_TWO_ALLREDUCE_QUOTE))
    print("     ★★★ **两条独立路径给出同一个数**：左边是**代码结构**（块数 × 每块次数），"
          "右边是 **[C] 论文的逐字** ⇒ 不是恒真。")
    print("     ★★ 若把配对改成另外三种之一 ⇒ 每块 `%d` 次 ⇒ 前向 **%d** 次（**两倍**）"
          % (per_block_other, blocks * per_block_other))
    print("     ★★ 而 `:223` 只写了「attention 后、FFN 后**各一次**」（= 前向 2 次），"
          "**漏了 backward 那 2 次** ⇒ 全模型的账在 `03_saturation.py`（回链）。")
    check("B.2", per_block == 1 and per_block_other == 2 and fwd_paired == 2
          and specs.MEGATRON_TP_COMM_OPS_PER_LAYER == 4
          and specs.MEGATRON_TWO_ALLREDUCE_QUOTE
          == "two all-reduces in the forward path and two in the backward path",
          "★★ **判据成立**：配对后每块 `1` 次、每层前向 `%d` 次，与 [C] 的逐字"
          "（`two all-reduces in the forward path and two in the backward path`）对拍 ✓；"
          "★ 不配对 ⇒ 每块 `%d` 次（**两倍**）；"
          "★ [C] 的 `4` 是**前向 + 后向**的合计（`MEGATRON_TP_COMM_OPS_PER_LAYER`）"
          % (fwd_paired, per_block_other))


# ===========================================================================
# 7. C 类：边界与反例 —— 忘记 all-reduce、切错轴、交错切
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：忘记 all-reduce（真的会错）· 切错轴（抛错）· 交错切（形状对、索引错）")
    print("=" * 74)
    x, w1, w2, w = toy_batch(B_SIZE, S_LEN, D_IN, D_OUT, D_FF, SEED)
    y_single = reference_mlp(x, w1, w2)

    # ── C.1 ★★ 忘记 all-reduce ────────────────────────────────────────────
    print("  ── C.1 ★★ **忘记 all-reduce**：把一片的部分和当成结果 ──")
    parts = row_parallel_partials(x, w, TP)
    d_keep1 = float(np.abs(parts[0] - reference_linear(x, w)).max())
    d_times = float(np.abs(parts[0] * TP - reference_linear(x, w)).max())
    print("     只取 `device 0` 的局部部分和：`max|diff|` = **%.6f**" % d_keep1)
    print("     拿了「差 `TP` 倍」的补偿、但**仍然没规约**（`partials[0] × TP`）："
          "`max|diff|` = **%.6f**" % d_times)
    print("     ★★ 与 `A.4` 的机器精度档比：**大了 15 个数量级** ⇒ 这是**可观测的失败**，"
          "不是精度问题。")
    check("C.1", d_keep1 > TOL_ERR and d_times > TOL_ERR,
          "★★ **失败成立**（这是 C 类：断言的是「会出问题」）：忘记 all-reduce ⇒ 差 "
          "**%.6f**（≫ `%g`）⇒ 拿它当结果**会错**" % (d_keep1, TOL_ERR))

    # ── C.2 ★★ 切错轴 ─────────────────────────────────────────────────────
    print()
    print("  ── C.2 ★★ **切错轴**：把 `x` 按输入维切开、却仍与**整个** `W` 相乘 ──")
    raised = None
    try:
        _ = np.split(x, TP, axis=-1)[0] @ w          # (b,s,D_in/TP) @ (D_in,D_out)
    except ValueError as e:
        raised = str(e)
    print("     ⇒ `ValueError: %s`" % (raised if raised is not None else "**没有报错**"))
    check("C.2", raised is not None and "mismatch" in raised and "core dimension" in raised,
          "★★ **失败成立**：形状不匹配**立刻抛 `ValueError`**（`mismatch in its core "
          "dimension`）⇒ 「切错轴」在**类型层**就被挡住（★ 但见 `C.3`：**形状对**的错法挡不住）")

    # ── C.3 ★★ 交错切（形状对、索引错位）────────────────────────────────────
    print()
    print("  ── C.3 ★★ **交错切**：形状**完全一样**、索引**错位** —— 形状检查挡不住它 ──")
    x_int = interleaved_split(x, TP)
    y_int = x_int @ w
    d_int = float(np.abs(y_int - reference_linear(x, w)).max())
    print("     `x` 的形状：原 `%s` vs 交错切后 `%s` ⇒ **相同**（形状检查无信息量）"
          % (x.shape, x_int.shape))
    print("     交错切后乘 `W`：`max|diff|` = **%.6f**" % d_int)
    print("     ★★★ **这一条是 C 类里最值钱的**：它证明「形状对」**不等于**「切对了」——")
    print("        只有**数值判据**（`A.1`/`A.4`）能抓住它，而它比机器精度档大 15 个数量级。")
    check("C.3", x_int.shape == x.shape and d_int > TOL_ERR,
          "★★ **失败成立**：交错切的**形状与分块切完全相同**（`%s`），而数值差 **%.6f** ⇒ "
          "「形状对」不等于「切对了」（`CONTRIBUTING:238`：C 类装的是**失败**，不是成功）"
          % (tuple(x_int.shape), d_int))


# ===========================================================================
# 8. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
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
    print("  ★★ **本程序的非线性是 `relu`（论文里是 GeLU）**：机制只要求「非线性」，"
          "不要求具体形状 ⇒ 那是本库的**替身**，不是「GeLU 的实测」。")
    print("  ★★★ **禁区回链（本程序一行都不重写）**：TP 的三条"
          "（通信在**每一层内部** · 通信量 ∝ **激活** · **只在高带宽互联上可用 / 通常限于"
          "单节点 8 卡**）已经由 `%s` 逐字写完。" % specs.TP_BOUNDARY_REF)
    print("     另有：集合通信原语的算法与复杂度（`V(N) = 2(N−1)/N × S`）→ `110-communication`"
          " 与 `performance-models/02:209`；GEMM 的 tiling → `40-kernels/gemm/`；"
          "注意力变体的结构（MHA/MQA/GQA、KV head 配比）→ "
          "`80-model-architectures/attention-variants/01-mha-mqa-gqa.md`。")
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

    measured = {"SEED": SEED, "B_SIZE": B_SIZE, "S_LEN": S_LEN, "D_IN": D_IN,
                "D_OUT": D_OUT, "D_FF": D_FF, "TP": TP, "TP_GRID": TP_GRID,
                "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT}
    expect = {"SEED": 20260951, "B_SIZE": 2, "S_LEN": 8, "D_IN": 64,
              "D_OUT": 32, "D_FF": 128, "TP": 4, "TP_GRID": (1, 2, 4, 8),
              "FLOAT_EXACT_LIMIT": 9007199254740992}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", D_FF == 4 * D_OUT and TP in TP_GRID
          and all(D_IN % t == 0 and D_OUT % t == 0 and D_FF % t == 0 for t in TP_GRID),
          "★★ 形状的**整除性**是被断言的：`D_FF == 4 × D_OUT`、且 `TP_GRID` 里每个 `t` 都整除"
          " `D_IN`/`D_OUT`/`D_FF` ⇒ ★ **不整除的处置在 `02_heads_vocab.py`**（本程序不涉及，"
          "「整除了」是本程序结论的**适用边界**）")

    th = {"TOL_MACHINE": TOL_MACHINE, "TOL_GUARD": TOL_GUARD, "TOL_ERR": TOL_ERR,
          "SHARE_LO": SHARE_LO, "SHARE_HI": SHARE_HI}
    th_expect = {"TOL_MACHINE": 1e-12, "TOL_GUARD": 1e-6, "TOL_ERR": 1e-3,
                 "SHARE_LO": 0.35, "SHARE_HI": 0.75}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 5 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < FLOAT_EXACT_LIMIT
          and FLOAT_EXACT_LIMIT == 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点阈值都 `< 2^53`（最大的是 `%g`）⇒ "
          "`mutate.py` 的 `v + 1.0` **能**改变它们（否则那类叶子**免检**）"
          % max(th_expect.values()))
    check("D.4c", SHARE_LO < 0.5 < SHARE_HI and TOL_MACHINE < TOL_GUARD < TOL_ERR,
          "★ `B.1` 的区间**跨过 `0.5`**（本机实测均值就在它附近）⇒ 这是一条**两侧**判据；"
          "★ 三个阈值**量级递增**（`%.0e < %.0e < %.0e`）⇒ 三者不会互相冒充"
          % (TOL_MACHINE, TOL_GUARD, TOL_ERR))
    check("D.4d", no_framework_imported(),
          "★ 运行期确认：`sys.modules` 里的第三方名字中只有 `numpy`"
          "（静态那一半在 `make lint` 的 AST 检查里）")
    return len(FAILS) == before


# ===========================================================================
# 9. D 类（后半）：specs 常量钉住 + 独立性证据 + 恒真检查 + 边界扩散检查
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 · 独立性证据 · **恒真检查** · 边界锚点的扩散检查")
    print("=" * 74)

    check("D.5", specs.MEGATRON_TP_ARXIV_ID == "1909.08053"
          and specs.MEGATRON_TP_TITLE
          == "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
          and specs.MEGATRON_TP_AR5IV_URL
          == "https://ar5iv.labs.arxiv.org/html/1909.08053"
          and specs.MEGATRON_TP_PAGE_BYTES == 203970
          and specs.MEGATRON_TP_PAGE_SHA256_16 == "e29667b94577fae1"
          and specs.MEGATRON_GELU_NONLINEAR_QUOTE == "Since GeLU is a nonlinear function,"
          and specs.MEGATRON_GELU_SYNC_QUOTE
          == "and this approach will require a synchronization point before the GeLU function."
          and specs.MEGATRON_COL_SPLIT_LEAD_QUOTE == "Another option is to split"
          and specs.MEGATRON_COL_REMOVES_SYNC_QUOTE
          == "This is advantageous as it removes a synchronization point."
          and specs.MEGATRON_TWO_ALLREDUCE_QUOTE
          == "two all-reduces in the forward path and two in the backward path"
          and specs.MEGATRON_FIG4_QUOTE
          == "There are 4 total communication operations in the forward and backward pass of a "
             "single model parallel transformer layer."
          and specs.MEGATRON_TP_COMM_OPS_PER_LAYER == 4
          and specs.RING_ALLREDUCE_VOLUME_FACTOR == 2.0
          and specs.TP_BOUNDARY_REF
          == "10-foundations/performance-models/02-amdahl-and-scaling.md:223",
          "★★ [C] `%s`（**不确认同行评审**，abs 页 `Comments:` 计 0 次）的**十三条**常量"
          "逐位一致 —— ★ 其中 `MEGATRON_COL_REMOVES_SYNC_QUOTE`（`removes`）是本主题"
          "**机制理由的结论句**；★ `MEGATRON_TWO_ALLREDUCE_QUOTE` 是**次数**的判定句"
          "（★ `:223` 只写了前向那 2 次，**后向 2 次的逐字出处在这里**）" % specs.MEGATRON_TP_ARXIV_ID)
    check("D.5b", specs.MEGATRON_GELU_NONLINEAR_QUOTE in
          "Since GeLU is a nonlinear function, and this approach will require a "
          "synchronization point before the GeLU function."
          and specs.MEGATRON_GELU_SYNC_QUOTE.endswith("function.")
          and specs.MEGATRON_COL_SPLIT_LEAD_QUOTE in
          "Another option is to split A along its columns"
          and specs.MEGATRON_FIG4_QUOTE.count("4") == 1,
          "★★ **机制从句的两半【都引自原文】**（不是本库编的）：前半是「非线性」那句、"
          "后半是「同步点」那句；★ 且 `MEGATRON_FIG4_QUOTE` 里**只有一个 `4`**"
          "（`4 total`）⇒ 它是 `8 = 4 × 2` 里那个 `4` 的可数依据")
    got = doc_has_boundary_ref("01-column-and-row.md", specs.TP_BOUNDARY_REF)
    if got is None:
        print("  [skip] D.5c 配套文档不在（`mutate.py` 的副本目录里只有 `*.py`）⇒ 跳过；"
              "★ 字面量本身已在 `D.5` 里逐位钉住")
    else:
        check("D.5c", got,
              "★★ **边界锚点已扩散到本文档**：`%s` 逐字出现在 `01-column-and-row.md` 里 ⇒ "
              "「本域不重写那三条」这句话**有可执行的检查**（`P1`：订正不扩散是默认失败）"
              % specs.TP_BOUNDARY_REF)
    check("D.5d", specs.UNSOURCED == ("tensor_parallel_vocab_pad_multiple_128_rationale",
                                     "tensor_parallel_pad_logits_masking_mechanism"),
          "★★ `specs.UNSOURCED` 的 **2** 项逐位钉住 —— ★ 两项都是**真的取不到**的东西"
          "（`128` 为何是 128 · Megatron 怎么屏蔽补齐位置），**而不是「本主题不做」的东西**"
          "（后者**不得**进这里：那是「既保留数字又记 `UNSOURCED`」）")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(column_parallel_linear)
    src_b = inspect.getsource(row_parallel_linear)
    src_c = inspect.getsource(reference_linear)
    src_d = inspect.getsource(reference_mlp)
    check("D.7", len({src_a, src_b, src_c, src_d}) == 4,
          "★ 防 `P17`：被测（`column_parallel_linear` %d 字符 / `row_parallel_linear` %d）与"
          "**两条独立参照**（`reference_linear` %d / `reference_mlp` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c), len(src_d)))
    watched = {"column_parallel_linear", "row_parallel_partials", "row_parallel_linear",
               "all_reduce_sum", "all_gather_concat", "relu", "mlp_column_row", "mlp_pair"}
    called = _callee_names(reference_linear) | _callee_names(reference_mlp)
    check("D.7b", not (watched & called),
          "★ **AST 调用表**：**两条参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    # ★★ **调用计数器**：先证明它非空，再证明参照跑完为 0。
    _CALLS.clear()
    mlp_column_row(*toy_batch(B_SIZE, S_LEN, D_IN, D_OUT, D_FF, SEED)[:3], TP)
    seen = sum(_CALLS.values())
    _CALLS.clear()
    xx, ww1, ww2, _ = toy_batch(B_SIZE, S_LEN, D_IN, D_OUT, D_FF, SEED)
    reference_mlp(xx, ww1, ww2)
    reference_linear(xx, ww1[:, :D_OUT])
    left = sum(_CALLS.values())
    check("D.7c", seen > 0 and left == 0,
          "★★ **调用计数器两向都量**：被测路径跑一次 ⇒ 计数 = **%d**（> 0，计数器**非空**）；"
          "**两条参照**跑完 ⇒ 计数 = **%d**（必须**恰好为 0**）⇒ "
          "「参照不调用被测」不是靠读代码相信的" % (seen, left))

    # ── ★★ 恒真检查（本域判据设计的第一依据）────────────────────────────────
    print()
    print("  ── D.10 ★★★ **恒真检查**：列并行的「与单卡逐位相同」是无用的判据 ──")
    x, w1, w2, w = toy_batch(B_SIZE, S_LEN, D_IN, D_OUT, D_FF, SEED)
    d_col = float(np.abs(column_parallel_linear(x, w, TP) - reference_linear(x, w)).max())
    d_row = float(np.abs(row_parallel_linear(x, w, TP) - reference_linear(x, w)).max())
    print("     列并行 `max|diff|` = **%.3e**（**恒为 0**）· 行并行 = **%.3e**（非零）"
          % (d_col, d_row))
    print("     ⇒ ★★★ **列并行的那条判据恒真**：它只重述了「把输出列分开算」这个**定义**，"
          "**不能当 A 类**（计划 `D22`：本域最自然的 A 类判据在「纯搬运」处恒为 `0.0`）。")
    print("     ⇒ ★ 本程序的 A 类因此落在**改变了求值顺序**的地方：`A.1`（列+行的组合）、"
          "`A.4`（行并行）、`A.3`（四格的**通信次数**）、`A.2`（非线性不可交换）。")
    print("     ⇒ ★ 而列并行的**位置**仍然有信息量 —— 但那是**机制**（`A.2`/`A.3`），"
          "不是「等价性」。")
    print("     ★★ **一处必须写明的边界**：上面那个 `0.0` 是**本组形状**"
          "（`x=(2,8,64)`、`W=(64,32)`、`TP=%d`）下的结果 —— "
          "★ `02_heads_vocab.py` 的 `D.10` 会给出**反例**：换一组形状（`(3,4,64)`），"
          "vocab 维的列切差是 `1.4e-14`（**不是 `0.0`**，BLAS 路径变了）。" % TP)
    print("        ⇒ ★★ **纪律**：「列切 == 单卡」**不是一条恒真的判据，而是一条【形状依赖】的**："
          "它在某些形状下恒真、在另一些形状下给出机器精度档的噪声 —— "
          "**两种情形都不能当 A 类**（前者无信息量，后者的差与「切分对不对」无关）。")
    print("     ★★ **一处可移植性（必须写明）**：上面这条判据（`d_col == 0.0`）**是本机 BLAS 依赖的**"
          " —— 换 BLAS / 换机器时分块与向量化的选择会变 ⇒ ★ **本判据与阈值的对应关系可能需要"
          "重新标定**（在别的机器上可能误报失败）。这正是本域「本机实测值」限定语的一个具体落法。")
    check("D.10", d_col == 0.0 and d_row > 0.0,
          "★★★ **恒真性已被机械证明**（**本组形状下**）：列并行的差**为 `0.0`**、"
          "行并行的差**非零** ⇒ 「切分 == 单卡」这条判据**只在改变了求值顺序的地方**有信息量。"
          "★ 这条断言本身**能失败**：若列并行被改成改变求和顺序的实现，`d_col` 就不再是 `0.0`；"
          "★★ 而**反例**在 `02` 的 `D.10`（另一组形状下不是 `0.0`）⇒ "
          "**「恒真」这个词本身也要带形状**")


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
    print("01_column_row.py —— TP 的切法：列切输出维、行切输入维、以及【配对约束】")
    print("=" * 74)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 头号风险是「模拟器同时是被测对象」：本程序的 `all_reduce` 就是 `sum()` "
          "⇒ 纯搬运处的等价性判据**恒真**（`D.10` 机械地证明了这件事），不能当 A 类。")
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
