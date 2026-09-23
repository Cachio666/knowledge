#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_heads_vocab.py —— TP 的三处「不整」：**头切分**、**vocab 并行**、**不整除的三种处置**。

配套文档：`02-heads-and-vocab.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**。

★★★ **本主题的头号风险是「模拟器同时是被测对象」**（计划 §1.3.3）：
  本程序的 `all_reduce` 就是 `sum()`、`all_gather` 就是 `concat`
  ⇒ 「切分后 == 单卡」这类判据**极易写成恒真**。★ 处置有三条：
    · **两处「纯搬运」的等价性判据不当 A 类**：**头切分**（每卡算自己的 head）
      与 **vocab 列切 + all-gather**。`D.10` 把「头切分的差恒为 `0.0`」变成一条**机械检查**；
    · **每一条 A 类判据都带守卫**（`CONTRIBUTING:235`）——
      `A.1g` 把参照故意改错、`A.2` 的两侧、`A.3` 的**成因**（恰有一片非零）都是可失败的；
    · **参照实现不得调用被测实现的任何函数** —— `D.7b`（AST 调用表）与 `D.7c`（调用计数）两条。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 `padded_vocab_logits` ⇒ 补齐路径无从算起（不会通过）；
    `C.1`/`C.2`/`C.3` 断言的是**失败行为**，删掉被测逻辑它们同样不成立。

★★ **覆盖边界（必须写明 —— 计划 §2 的第 3 条）**：本程序的 A 类守的是
  「**切分做对了没有**」（头怎么分、vocab 切在哪一维、不整除时怎么处置、索引有没有错位），
  **不守**「算子本身对不对」（那由 `numpy` 保证）⇒ **不得声称 A 类验证了更多东西**。
  ★★ **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：本程序的 `C.1` 断言「**必须抛错**」、
  `C.2` 断言「**部分 token 恒不可预测**」、`C.3` 断言「**概率算错**」—— 三条都是失败。

★★★ **本主题的禁区（回链，一行都不重写）**：
  `10-foundations/performance-models/02-amdahl-and-scaling.md:223` 已经逐字写完了 TP 的**三条**
  （通信发生在**每一层内部** · 通信量与**激活**成正比 · **只在高带宽互联上可用 /
  通常限于单节点 8 卡**，以及「规模上限由节点内互联决定」）
  ⇒ **本程序一行都不重写那三条**（否则是 `P2`「同一个载体重复」）。
  ★★ **另一条边界（本主题特有）**：**注意力变体的结构**（MHA / MQA / GQA 的定义、
  KV head 与 Q head 的**配比**）→ `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`
  ⇒ **本程序只写「头怎么分给设备」**，不写「有几种 head 结构」、也不写 KV head 的配比。
  ★ 另有：**GEMM 的 tiling** → `40-kernels/gemm/`；**通信原语的算法与复杂度** → `110-communication`
  与 `performance-models/02:209`（本程序**只引用、不重推**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **补齐**之后前 `V` 列的 logits == 单卡（**机器精度档**，**不是逐位**）；
  · `A.1g` ★★ **守卫**：把参照的词表**故意改错**（两行互换）⇒ 同一条断言**必须失败**；
  · `A.2`  ★★★ **「逐位相等」这条更强的判据在本机【不成立】**（实测差 > `0`）——
             成因是**补齐改变了张量与内存布局 ⇒ BLAS 的路径变** ⇒ 判据只能写机器精度档；
  · `A.3`  ★★ **输入 embedding 列切 + all-reduce == 单卡，且【逐位】相同** ——
             成因可验证：每个 token 在每张卡的分片里**恰有一项非零**（其余是结构性的零）；
  · `B.1`  ★★ **vocab 并行的通信量**：`b×s×v` vs **与交叉熵融合后的** `b×s` ⇒ 差 `v` 倍；
  · `B.2`  ★ **补齐的代价**：`50257 → 51200`（多 `943` 个），且 `51200 % 1024 == 0` 而
             `50257 % 1024 = 97` —— 「补齐」的可机械判定含义；
  · `C.1`  ★★★ **`np.split` 在不整除时【抛错】**（逐字 `ValueError`）⇒
             这**是一道防线**：用了它的地方，不整除**不可能静默通过**；
  · `C.2`  ★★ **静默截断** ⇒ 输出形状 `(B, S, 28)` 而非 `(B, S, 30)`，
             那 **`2/30`** 个 token **恒不可预测**（断言「全部 token 都能被预测」会失败）；
  · `C.3`  ★★★ **补齐但忘记把 pad 位置屏蔽掉** ⇒ softmax 的概率**算错**
             （概率质量漏到 pad 列上 ⇒ 前 `V` 列的和 **< 1**）。

用法：
    python3 02_heads_vocab.py
"""

import ast
import collections
import inspect
import math
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
                                "⇒ 不是恒真的」的措辞是**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260952              # ★ 与同域另外两个程序（…0951 / …0953）都**不同**
B_SIZE = 3                   # micro-batch 大小（`b`）—— ★ 与 `C.2` 的输出形状 `(3, 28)` 对得上
S_LEN = 4                    # 序列长度（`s`）
N_HEADS = 8                  # 注意力头数
D_HEAD = 8                   # 每个头的宽度
D_MODEL = 64                 # 模型宽度 = `N_HEADS × D_HEAD`（`D.3b` 会断言这个等式）
TP = 4                       # TP 度
VOCAB = 30                   # ★★★ **词表大小：故意取一个【不被 `TP` 整除】的数**
VOCAB_PER_GPU_MULTIPLE = 4   # ★ 本程序对 [C] `128` 的**玩具替身**（见 `D.3b` 的说明）
VOCAB_PAD = 32               # `30` 补齐到 `VOCAB_PER_GPU_MULTIPLE × TP = 16` 的倍数 ⇒ `32`
# ── 阈值 ────────────────────────────────────────────────────────────────────
TOL_MACHINE = 1e-12          # 「机器精度档」（`A.1`/`A.2` 的等价性）
TOL_GUARD = 1e-6             # 守卫：改错的参照必须**远大于**它
TOL_ERR = 1e-3               # C 类「真的会错」的下限（实测差比它大几个数量级）
# ★★ **盲区⑥ 的落法**：本程序的浮点常量都 `< 2^53`（最大的阈值是 `1e-3`）⇒ 每个叶子**可变异**。
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 十进制字面量：盲区④）


# ===========================================================================
# 2. 被测路径（头切分 · vocab 列切 · 补齐 / 截断）
# ===========================================================================
def all_reduce_sum(parts):
    """**被测**：`all_reduce` = **显式的张量搬运**（逐元素求和）。

    ★★ 依据是 [C] `1909.08053` 的 `g` 算子与 [B] NCCL 的逐字结果语义；
      本域的知识对象是**切分策略**，不是通信原语的实现。
    """
    _CALLS["all_reduce_sum"] += 1
    return sum(parts)


def all_gather_concat(shards, axis=-1):
    """**被测**：`all_gather` = **显式的张量搬运**（沿 `axis` 拼接）。"""
    _CALLS["all_gather_concat"] += 1
    return np.concatenate(shards, axis=axis)


def head_parallel_attention(q, k, v, tp):
    """**被测**：**头切分** —— 每个 head 的矩阵乘**在本卡本地算**。

    ★★ [C] `1909.08053` §3 逐字：`%s`

    ★★ **不改变任何一个元素的求值顺序**：每张卡拿到的 head **整块**就是单卡时那几个 head，
      用的还是同一个 `einsum` ⇒ 这是**纯搬运**（`D.10` 会把这件事变成一条检查）。
    ★★★ **作用域**：原文用的是 `doesn't require any immediate communication` ——
      「**立即**」这个词是关键：**紧接着的 output linear 层是按行并行的** ⇒
      那次 all-reduce 仍然发生（只是可以晚一点）⇒ ★ **不得写成「头切分不需要通信」**。
    """
    _CALLS["head_parallel_attention"] += 1
    outs = []
    for qh, kh, vh in zip(np.split(q, tp, axis=2), np.split(k, tp, axis=2),
                          np.split(v, tp, axis=2)):
        s = np.einsum('bshd,bthd->bhst', qh, kh) / math.sqrt(D_HEAD)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        outs.append(np.einsum('bhst,bthd->bshd', p, vh))
    return all_gather_concat(outs, axis=2)


def vocab_parallel_logits(x, emb, tp):
    """**被测**：**vocab 并行**的输出侧 —— `emb` 按**词表维**切，算完 `all_gather` 拼回。

    ★★ 这是 [C] 的 `parallel GEMM` + `all-gather` 那一支（**未融合**交叉熵）；
      ★★ 它是**纯搬运**（每个输出元素只由一张卡算，点积顺序不变）。
    """
    _CALLS["vocab_parallel_logits"] += 1
    return all_gather_concat([x @ w for w in np.split(emb, tp, axis=1)], axis=-1)


def padded_vocab_logits(x, emb, tp, pad_to):
    """**被测**：**补齐**的处置 —— 把词表按列补到 `pad_to`，再按词表维切、算完拼回。

    ★★★ **返回的是【补齐后】的 logits**（形状 `(b, s, pad_to)`）—— **不在这里切回前 `V` 列**：
      调用方（`A.1`）自己切，因为「取前 `V` 列」这一步**本身就是判据要检查的索引操作**，
      藏进被测函数里会让它**无法被单独断言**。
      ★★ 而 `B.1`/`C.3` 需要的正是**补齐后**的张量（通信量 / 概率质量都发生在那一层）。

    ★★ [C] `1909.08053` §5.1 逐字（出处见 `specs.MEGATRON_VOCAB_PAD_QUOTE` /
      `specs.MEGATRON_VOCAB_PAD_RESULT_QUOTE`）：**把词表补齐到能被整除**。

    ★★★ **补齐改变了张量与内存布局** ⇒ BLAS 可能走**不同的路径**
      ⇒ 「补齐后与单卡**逐位**相同」这条更强的判据**不成立**（`A.2` 把它量出来）。
    """
    _CALLS["padded_vocab_logits"] += 1
    emb_pad = np.zeros((emb.shape[0], pad_to))
    emb_pad[:, :emb.shape[1]] = emb
    return all_gather_concat([x @ w for w in np.split(emb_pad, tp, axis=1)], axis=-1)


def embed_column_split_lookup(idx, emb, tp):
    """**被测**：**输入 embedding 的列切** + all-reduce（`A.3` 的被测路径）。

    ★ 每张卡只持有词表的一段（`emb` 的**列**）⇒ 它只能查出落在自己那一段里的 token，
      其余的 token 在它这里是**结构性的零**（不是「算出 0」）。
    """
    _CALLS["embed_column_split_lookup"] += 1
    per = emb.shape[1] // tp
    parts = []
    for r in range(tp):
        local = np.zeros((len(idx), emb.shape[0]))
        for i, t in enumerate(idx):
            if r * per <= t < (r + 1) * per:
                local[i] = emb[:, t]
        parts.append(local)
    return all_reduce_sum(parts)


def silent_truncate(emb, tp):
    """**被测**：**静默截断** —— 只留能被 `TP` 整除的那一段。

    ★★ 这是**最危险**的一种处置：**它不报错**，只是把最后 `VOCAB − per×TP` 个 token
      **从词表里删掉**了（`C.2` 把这个后果量出来）。
    ★ 要写出它**必须自己写 `[:per*tp]`** —— `np.split` 在同样输入下**会抛错**（`C.1`）。
    """
    _CALLS["silent_truncate"] += 1
    per = emb.shape[1] // tp
    return emb[:, :per * tp]


def softmax_unmasked(logits_pad):
    """**被测**（`C.3` 的错法）：对**补齐后**的 logits 直接做 softmax —— **不屏蔽 pad 列**。

    ★★ 后果：概率质量**漏到 pad 列上** ⇒ 前 `V` 列的和 **< 1**。
    """
    _CALLS["softmax_unmasked"] += 1
    e = np.exp(logits_pad - logits_pad.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def softmax_masked(logits_pad, real_vocab):
    """**被测**（正解）：把 pad 列置 `-inf` 之后再做 softmax。

    ★★ 这是「补齐」能成立的前提 —— ★ 而 [C] `1909.08053` **没有写**这一步的具体做法
      ⇒ 已记 `UNSOURCED`：`tensor_parallel_pad_logits_masking_mechanism`
      （本程序里的这一支是**本库的构造**，不是「Megatron 的做法」）。
    """
    _CALLS["softmax_masked"] += 1
    m = logits_pad.copy()
    m[..., real_vocab:] = -np.inf
    e = np.exp(m - m.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ===========================================================================
# 3. 独立参照（**不调用**被测路径的任何函数）
# ===========================================================================
def reference_attention(q, k, v):
    """**独立参照**：单卡、**一次**算完全部 head 的注意力。

    ★★★ **独立性**：这条路径**不调用**任何被测函数（`D.7b` 的 AST 调用表 +
      `D.7c` 的调用计数**两条**证据）。
    """
    s = np.einsum('bshd,bthd->bhst', q, k) / math.sqrt(D_HEAD)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum('bhst,bthd->bshd', p, v)


def reference_logits(x, emb):
    """**独立参照**：单卡、**一次**矩阵乘（`x @ emb`）—— 不做任何切分。"""
    return x @ emb


def toy_heads(b, s, n_heads, d_head, seed):
    """造 `(b, s, h, d)` 的 `q/k/v`（★ 只用于把**形状**摆出来）。"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((b, s, n_heads, d_head)) * 0.5,
            rng.standard_normal((b, s, n_heads, d_head)) * 0.5,
            rng.standard_normal((b, s, n_heads, d_head)) * 0.5)


def toy_vocab(b, s, d_model, vocab, seed):
    """造隐藏态 `(b, s, d_model)` 与词表矩阵 `(d_model, vocab)`。"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((b, s, d_model)),
            rng.standard_normal((d_model, vocab)))


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


# ===========================================================================
# 5. A 类：不整除的三种处置（判据 A.1 / A.1g / A.2 / A.3）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 不整除的三种处置：补齐（A.1/A.2）· 逐位相同（A.2）· 输入 embedding 列切（A.3）")
    print("=" * 74)
    x, emb = toy_vocab(B_SIZE, S_LEN, D_MODEL, VOCAB, SEED)
    print("  形状：`x=%s` · `emb=%s` · `TP=%d` · **`VOCAB=%d` 不被 `TP` 整除**"
          " ⇒ `%d %% %d = %d`" % (x.shape, emb.shape, TP, VOCAB, VOCAB, TP, VOCAB % TP))
    print("  ★★★ **本程序的正题就在这里**：`VOCAB` 故意取 `%d`（不被 `%d` 整除）——"
          " 这不是笔误，是**要考的那个情形**。" % (VOCAB, TP))

    # ── A.1 ★★★ 补齐后前 V 列 == 单卡（机器精度档）────────────────────────
    print()
    print("  ── A.1 ★★★ **补齐**：把词表补到 `%d`（`%d × %d = %d` 的倍数）再切、算完取前 `%d` 列 ──"
          % (VOCAB_PAD, VOCAB_PER_GPU_MULTIPLE, TP, VOCAB_PER_GPU_MULTIPLE * TP, VOCAB))
    y_pad_full = padded_vocab_logits(x, emb, TP, VOCAB_PAD)     # (b, s, 32)
    y_pad = y_pad_full[..., :VOCAB]                             # ★ **判据要检查的那一刀**：取前 V 列
    y_single = reference_logits(x, emb)
    md = float(np.abs(y_pad - y_single).max())
    scale = float(np.abs(y_single).max())
    same_bits = bool(np.array_equal(y_pad, y_single))
    # ★★ 守卫：把参照的词表**故意改错**（两行互换）⇒ A.1 必须失败
    emb_bad = emb.copy()
    emb_bad[[0, 1]] = emb_bad[[1, 0]]
    md_bad = float(np.abs(y_pad - reference_logits(x, emb_bad)).max())
    print("     [C] `%s` §5.1 逐字：`%s`" % (specs.MEGATRON_TP_ARXIV_ID,
                                             specs.MEGATRON_VOCAB_PAD_QUOTE))
    print("       以及 `%s`（同一段）" % specs.MEGATRON_VOCAB_PAD_RESULT_QUOTE)
    print("     补齐后的张量 `%s` ⇒ 取前 `%d` 列 ⇒ `%s`" % (y_pad_full.shape, VOCAB, y_pad.shape))
    print("     `max|补齐后前 V 列 − 单卡|` = **%.3e**（`y` 的量级 %.3f）" % (md, scale))
    print("     ★★ **守卫**（`CONTRIBUTING:235`）：把参照的词表两行互换 ⇒ "
          "差 = **%.3e** ⇒ 同一条断言**必须失败**" % md_bad)
    check("A.1", md <= TOL_MACHINE * max(scale, 1.0) and np.isfinite(md),
          "★★★ **判据成立**：补齐之后**前 `%d` 列的 logits 就是真 logits**（差 `%.3e` ≤ "
          "机器精度档）⇒ 补齐**不改变数学**，只有代价（`B.2`）。"
          "★★ 注意**判据的档位**：这里**不能**写「逐位相等」（见 `A.2`）" % (VOCAB, md))
    check("A.1g", md_bad > TOL_GUARD,
          "★★ **守卫成立**：故意改错的参照给出 `%.3e` > `%g` ⇒ `A.1` **能失败**"
          "（它守的是「补的对不对、切在哪一维、取前 `V` 列的索引有没有错位」）"
          % (md_bad, TOL_GUARD))

    # ── A.2 ★★★ 「逐位相等」在本机【不成立】──────────────────────────────
    print()
    print("  ── A.2 ★★★ **「补齐后逐位相同」这条更强的判据【不成立】**（本机实测）──")
    print("     `np.array_equal(补齐后前 V 列, 单卡)` = **%s** · `max|diff|` = **%.3e**"
          % (same_bits, md))
    print("     ⇒ ★★★ **成因**：补齐改变了**张量的形状与内存布局**（`%s` → `%s`，"
          "`%d` 列 → `%d` 列）⇒ BLAS 可能走**不同的路径**（分块/向量化不同）"
          % ((B_SIZE, S_LEN, VOCAB), (B_SIZE, S_LEN, VOCAB_PAD), VOCAB, VOCAB_PAD))
    print("     ⇒ ★★ **纪律**：这条等价性的判据**只能写「机器精度档」** ——")
    print("        写「逐位相等」会在**本机**上直接失败（而它是**正确的**实现！）。")
    print("     ⇒ ★ **本程序的立场**：`A.2` 钉的不是「实现错了」，而是"
          "「**这条判据的档位必须放宽**」这个事实 —— 它**能失败**：")
    print("        若某个 BLAS 恰好给出逐位相同的结果，本断言会失败 ⇒ 那说明**档位的来源变了**。")
    check("A.2", md > 0.0 and md <= TOL_MACHINE * max(scale, 1.0) and not same_bits,
          "★★★ **判据成立**（三个条件）：① 差 **> 0**（**逐位相等不成立**）；"
          "② 差 ≤ 机器精度档（**是 1 ulp 级，不是算错**）；③ `array_equal` 为 `False` "
          "⇒ ★★ **「补齐后逐位相同」不是一条可用的判据**，原因写在上面（**形状变了 ⇒ BLAS "
          "路径变**）。★ 这条判据**能失败**：把它改成逐位相等 ⇒ 立刻失败")

    # ── A.3 ★★ 输入 embedding 列切 + all-reduce ───────────────────────────
    print()
    print("  ── A.3 ★★ **输入 embedding 列切 + all-reduce**：为什么它是【逐位】精确的 ──")
    print("     [C] `%s` §3 逐字：`%s`" % (specs.MEGATRON_TP_ARXIV_ID,
                                           specs.MEGATRON_EMBED_ALLREDUCE_QUOTE))
    # 输入 embedding：`emb` 按**词表维**切 ⇒ 每卡只持有一部分词表（列切）
    per = VOCAB // TP
    rng = np.random.default_rng(SEED + 1)
    idx = rng.integers(0, per * TP, size=16)      # ★ 只看能被均匀切到的那些 token
    y_split_in = embed_column_split_lookup(idx, emb, TP)
    y_ref_in = emb[:, idx].T
    md_in = float(np.abs(y_split_in - y_ref_in).max())
    pieces = [np.zeros((len(idx), D_MODEL)) for _ in range(TP)]
    for r in range(TP):
        for i, t in enumerate(idx):
            if r * per <= t < (r + 1) * per:
                pieces[r][i] = emb[:, t]
    nz_per_token = [sum(1 for r in range(TP) if np.count_nonzero(pieces[r][i]) > 0)
                    for i in range(len(idx))]
    print("     `emb` 被切成 %d 段（每段 `%s`）⇒ **每个 token 只落在【一段】里**"
          % (TP, (D_MODEL, per)))
    print("     实测（%d 个随机 token）：每张卡对每个 token 的贡献里"
          " **非零的段数** ∈ [%d, %d]" % (len(idx), min(nz_per_token), max(nz_per_token)))
    print("     `max|列切 + all-reduce − 单卡查表|` = **%.3e**（**逐位相同** = %s）"
          % (md_in, bool(np.array_equal(y_split_in, y_ref_in))))
    print("     ⇒ ★★★ **成因**：all-reduce 加的是 `%d` 个分片，其中**只有 1 个非零**、"
          "其余是**结构性的零**（不是「算出 0」，是「这张卡根本没有这一列」）" % TP)
    print("     ⇒ ★★ 「加 `t` 个分片」与「直接查表」**逐位相同** ⇒ 这条判据"
          "**在该形态下【恒真】** ⇒ ★ **不能当 A 类**（它是定义的重述，不是发现）。")
    print("     ⇒ 本程序把它钉成一条**事实**（非零段数 `min == max == 1`），"
          "而不是把它当正确性证据。")
    check("A.3", min(nz_per_token) == 1 and max(nz_per_token) == 1 and md_in == 0.0
          and specs.MEGATRON_EMBED_ALLREDUCE_QUOTE.startswith("Since each partition"),
          "★★ **判据成立**：每个 token 的**非零分片数恰好是 `1`**（实测 `%d` 个 token 全是 1）、"
          "且差**恒为 `0.0`** ⇒ **输入 embedding 列切的 all-reduce 是【逐位】精确的**，"
          "★ 而这正是「用『与单卡相同』当判据」在**这一处恒真**的**成因**"
          "（`D.10` 里对头切分与 vocab 列切也证明了同一件事）" % len(idx))


# ===========================================================================
# 6. B 类：代价模型 —— vocab 并行的通信量、补齐的代价
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：vocab 并行的通信量（`b×s×v` vs 融合后的 `b×s`）· 补齐的代价")
    print("=" * 74)
    x, emb = toy_vocab(B_SIZE, S_LEN, D_MODEL, VOCAB, SEED)

    # ── B.1 ★★ vocab 并行的通信量 ─────────────────────────────────────────
    print("  ── B.1 ★★ **vocab 并行的通信量**：未融合 vs 与交叉熵融合 ──")
    # ★★★ **本路径必须走「补齐」那一支**：`VOCAB=30` 不被 `TP=4` 整除 ⇒
    #   `np.split` 会**立刻抛错**（见 `C.1`）—— 这本身就是「不整除」的第一手后果。
    logits_pad = padded_vocab_logits(x, emb, TP, VOCAB_PAD)      # (B,S,32)：**补齐后的全量**
    n_gather = int(logits_pad.size)
    n_logits = int(logits_pad[..., :VOCAB].size)
    fused = logits_pad[..., :VOCAB].max(-1)          # 融合后只需 `b×s` 个元素的中间量
    n_fused = int(fused.size)
    ratio = n_logits / n_fused
    print("     [C] `%s` §3 逐字（all-gather 的量）：`%s`" % (specs.MEGATRON_TP_ARXIV_ID,
                                                             specs.MEGATRON_VOCAB_HUGE_QUOTE))
    print("       以及（融合之后）：`%s`" % specs.MEGATRON_VOCAB_FUSE_QUOTE)
    print("     未融合：all-gather 搬的是 `b×s×v` 个元素 ⇒ 实测真 logits 的 `size` = **%d**"
          "（`%d×%d×%d`）" % (n_logits, B_SIZE, S_LEN, VOCAB))
    print("       ★ 本程序走的是**补齐**那一支 ⇒ 实际搬的是 `b×s×%d` = **%d** 个元素"
          "（多 `%d` 个 pad 列）" % (VOCAB_PAD, n_gather, n_gather - n_logits))
    print("     与交叉熵融合：只需 `b×s` 个元素 ⇒ 实测 `size` = **%d**（`%d×%d`）"
          % (n_fused, B_SIZE, S_LEN))
    print("     ⇒ 比值 = **%.0f**（= `v`）⇒ ★★ **量级差 `v` 倍**：`v` 越大，未融合越不可接受"
          % ratio)
    print("     ★★ **口径**：这里量的是**元素个数**（`size`），不是字节数 ——")
    print("        换精度要乘 `itemsize`，但**比值与精度无关**（同一种 dtype 下）。")
    print("     ★★ **两条独立路径**：左边是 `numpy` 的张量形状（**程序真的建了那两个张量**），"
          "右边是 [C] 的公式口径 `b×s×v` / `b×s` ⇒ 不是恒真。")
    check("B.1", n_logits == B_SIZE * S_LEN * VOCAB and n_fused == B_SIZE * S_LEN
          and n_gather == B_SIZE * S_LEN * VOCAB_PAD and abs(ratio - VOCAB) < 1e-9
          and specs.MEGATRON_VOCAB_FUSE_QUOTE
          == "with the cross entropy loss which reduces the dimension to",
          "★★ **判据成立**：未融合 `%d` 个元素 vs 融合后 `%d` 个 ⇒ 比值 **%.0f** == `v` "
          "⇒ ★★★ **vocab 并行的代价在【输出侧】**，融合交叉熵把它从 `b×s×v` 压到 `b×s`；"
          "★ 补齐那一支实际搬 `%d` 个（多 `%d` 个 pad 列，不影响这个量级结论）"
          % (n_logits, n_fused, ratio, n_gather, n_gather - n_logits))

    # ── B.2 ★ 补齐的代价 ──────────────────────────────────────────────────
    print()
    print("  ── B.2 ★ **补齐的代价**：`51200` vs `50257`（[C] §5.1 的真实数）──")
    mult = specs.MEGATRON_VOCAB_PER_GPU_MULTIPLE * specs.MEGATRON_VOCAB_PAD_TP_DEGREE
    extra = specs.MEGATRON_VOCAB_PAD_SIZE - specs.MEGATRON_VOCAB_ORIGINAL
    rem_orig = specs.MEGATRON_VOCAB_ORIGINAL % mult
    print("     [C] `%s` §5.1 逐字：`%s`" % (specs.MEGATRON_TP_ARXIV_ID,
                                             specs.MEGATRON_VOCAB_PAD_QUOTE))
    print("       同段：`%s`" % specs.MEGATRON_8WAY_NOTE_QUOTE)
    print("     每卡词表大小的倍数 `%d` × TP 度 `%d` = **%d**"
          % (specs.MEGATRON_VOCAB_PER_GPU_MULTIPLE, specs.MEGATRON_VOCAB_PAD_TP_DEGREE, mult))
    print("     原词表 `%d`：`%d %% %d` = **%d**（不整除）" % (specs.MEGATRON_VOCAB_ORIGINAL,
                                                              specs.MEGATRON_VOCAB_ORIGINAL,
                                                              mult, rem_orig))
    print("     补齐后 `%d`：`%d %% %d` = **%d**（整除）⇒ 多算了 **%d** 个 token 的 logits"
          % (specs.MEGATRON_VOCAB_PAD_SIZE, specs.MEGATRON_VOCAB_PAD_SIZE, mult,
             specs.MEGATRON_VOCAB_PAD_SIZE % mult, extra))
    print("     ★★ 本程序在玩具规模上做同一件事：`%d` 补到 `%d`（`%d` 的倍数）⇒ 多 **%d** 个"
          % (VOCAB, VOCAB_PAD, VOCAB_PER_GPU_MULTIPLE * TP, VOCAB_PAD - VOCAB))
    print("     ★★ **代价的性质**：多算的是 **pad 列**的 logits（在真实规模上是 `.11%` 量级的"
          "多余计算），换来的是**能被整除** —— 而 [C] 明写这个倍数是 `for efficient GEMMs`。")
    check("B.2", rem_orig != 0 and specs.MEGATRON_VOCAB_PAD_SIZE % mult == 0
          and extra == 943 and VOCAB_PAD % (VOCAB_PER_GPU_MULTIPLE * TP) == 0
          and VOCAB_PAD - VOCAB == 2,
          "★★ **判据成立**：`%d %% %d = %d ≠ 0`（原词表不整除）、`%d %% %d = 0`（补齐后整除）、"
          "多算 **%d** 个 ⇒ 「补齐」这个词有了**可机械判定的含义**；"
          "★ 玩具规模同一件事：`%d → %d`（+`%d`）"
          % (specs.MEGATRON_VOCAB_ORIGINAL, mult, rem_orig,
             specs.MEGATRON_VOCAB_PAD_SIZE, mult, extra, VOCAB, VOCAB_PAD, VOCAB_PAD - VOCAB))


# ===========================================================================
# 7. C 类：边界与反例 —— 抛错的防线、静默截断、忘记屏蔽 pad
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：`np.split` 抛错（一道防线）· 静默截断（token 消失）· 忘记屏蔽 pad")
    print("=" * 74)
    x, emb = toy_vocab(B_SIZE, S_LEN, D_MODEL, VOCAB, SEED)

    # ── C.1 ★★★ np.split 在不整除时抛错 ───────────────────────────────────
    print("  ── C.1 ★★★ **`np.split` 在不整除时【抛错】** —— 这是一道**防线** ──")
    raised = None
    try:
        _ = np.split(emb, TP, axis=1)
    except ValueError as e:
        raised = str(e)
    print("     `VOCAB=%d`、`TP=%d`、`VOCAB // TP = %d`（余 `%d`）" % (VOCAB, TP, VOCAB // TP,
                                                                     VOCAB % TP))
    print("     ⇒ `ValueError: %s`" % (raised if raised is not None else "**没有报错**"))
    print("     ★★★ **为什么这是一道防线**：`np.split` 把「不整除」变成**立刻可见的失败** ⇒")
    print("        凡是**用了它**的地方，不整除**不可能静默通过** —— 后面两种处置"
          "（截断 / 补齐）**都是【主动】写出来的**。")
    print("     ★★ 一句话：**「不整除」在这条路径上不是风险，是异常**。")
    check("C.1", raised is not None and "array split does not result in an equal division" in raised,
          "★★★ **失败成立**（C 类断言的是「会出问题」）：`np.split` 逐字给出 "
          "`array split does not result in an equal division` ⇒ ★★ **它是一道防线**"
          "（用了它的地方，不整除**不可能静默通过**）")

    # ── C.2 ★★ 静默截断 ⇒ 部分 token 恒不可预测 ──────────────────────────
    print()
    print("  ── C.2 ★★ **静默截断**：`emb[:, :per*TP]` ⇒ 输出形状变了、token 消失了 ──")
    emb_tr = silent_truncate(emb, TP)
    y_tr = reference_logits(x, emb_tr)
    print("     截断后的词表形状：`%s`（原 `%s`）⇒ logits 形状 `%s`（原 `%s`）"
          % (emb_tr.shape, emb.shape, y_tr.shape, (B_SIZE, S_LEN, VOCAB)))
    gone = [t for t in range(VOCAB) if t >= y_tr.shape[-1]]
    # ★ 直接演示后果：把某个被删掉的 token 做成**最可能的**那一个
    t_bad = VOCAB - 1                                     # `29`（最后一个，被删掉了）
    probe = np.zeros((1, 1, VOCAB))
    probe[0, 0, t_bad] = 999.0                            # 单卡下 argmax 必然是它
    probe_tr = probe[..., :y_tr.shape[-1]]
    argmax_single = int(np.argmax(probe[0, 0]))
    argmax_tr = int(np.argmax(probe_tr[0, 0]))
    print("     ⇒ 消失的 token 编号：%s（共 **%d** 个，`%d/%d`）" % (gone, len(gone), len(gone), VOCAB))
    print("     ★★ **后果的直接演示**：造一个「第 `%d` 个 token 的 logit 是 999」的输入 ——"
          % t_bad)
    print("        单卡下 argmax = **%d**（正确）；截断后 argmax = **%d**（**永远不可能是 %d**）"
          % (argmax_single, argmax_tr, t_bad))
    print("     ⇒ ★★ 那 `%d` 个 token **恒不可预测**（它们的 logits **根本不存在**）⇒"
          " 「全部 token 都能被预测」这条断言**会失败**（它现在就是假的）。" % len(gone))
    check("C.2", y_tr.shape == (B_SIZE, S_LEN, VOCAB - 2)
          and argmax_single == t_bad and argmax_tr != t_bad
          and len(gone) == 2 and VOCAB % TP == 2,
          "★★ **失败成立**：静默截断把词表从 `%d` 变成 `%d`（`%d/%d` 个 token **恒不可预测**："
          "它们的 logits 根本不存在）⇒ ★ **这条路径不报错、只在训练后表现为「有些 token "
          "从来预测不出来」** —— 这正是它比 `C.1` 危险的地方" % (VOCAB, y_tr.shape[-1],
                                                                 len(gone), VOCAB))

    # ── C.3 ★★★ 补齐但忘记屏蔽 pad 位置 ──────────────────────────────────
    print()
    print("  ── C.3 ★★★ **补齐但忘记把 pad 列屏蔽掉** ⇒ softmax 的概率**算错** ──")
    logits_pad = padded_vocab_logits(x, emb, TP, VOCAB_PAD)
    # ★★★ **pad 位置的 logits 取成与真 logits 同量级的值** —— 这一步是**必需的**，
    #   不是修饰：Megatron 之所以要补，正是因为**那些位置会继续参与训练**
    #   （补齐出来的词表行带着任意的权重）。★ 若把 pad 位置取成 `0`，概率漏失只有 `1e-13`
    #   量级 ⇒ **这个失败会被「太小」掩盖掉**（本程序把两种都打出来对照）。
    rngm = np.random.default_rng(SEED + 3)
    logits_pad_real = logits_pad.copy()
    pad_scale = float(np.std(logits_pad[..., :VOCAB]))     # ★ 与**真 logits 同量级**（取标准差）
    logits_pad_real[..., VOCAB:] = (
        rngm.standard_normal((B_SIZE, S_LEN, VOCAB_PAD - VOCAB)) * pad_scale)
    p_bad = softmax_unmasked(logits_pad_real)
    p_ok = softmax_masked(logits_pad_real, VOCAB)
    p_zero = softmax_unmasked(logits_pad)            # 对照：pad 位置恰好是 `0`
    s_bad = p_bad[..., :VOCAB].sum(-1)
    s_ok = p_ok[..., :VOCAB].sum(-1)
    leak = float(p_bad[..., VOCAB:].sum(-1).max())
    leak0 = float(p_zero[..., VOCAB:].sum(-1).max())
    md3 = float(np.abs(p_bad[..., :VOCAB] - p_ok[..., :VOCAB]).max())
    print("     补齐到 `%d` 列后（**pad 位置的 logits 与真 logits 同量级**）：" % VOCAB_PAD)
    print("       **不屏蔽** ⇒ 前 `%d` 列的概率和 = **%.6f**（< 1）· 漏到 pad 列上的质量（最大）"
          " = **%.6f**" % (VOCAB, float(s_bad.min()), leak))
    print("       **屏蔽**（pad 列置 `-inf`）⇒ 前 `%d` 列的概率和 = **%.6f**（= 1）"
          % (VOCAB, float(s_ok.min())))
    print("       两种做法的概率差（`max|·|`）= **%.6f**" % md3)
    print("     ★★ **对照（一个必须写出来的量法口径）**：若 pad 位置的 logits 恰好是 `0`，"
          "漏失只有 **%.3e** 量级（`max` 口径）⇒ **这个失败会被「太小」掩盖**。"
          % leak0)
    print("        本程序因此**显式地把 pad 位置取成同量级的值**（见代码注释）——"
          "否则 `C.3` 会变成一条**看起来成立、实际什么也没检查**的判据。")
    print("     ⇒ ★★★ **「补齐」不是免费的**：它**多出来 `%d` 列**，"
          "不屏蔽就会**分走一部分概率** ⇒ 训练时表现为「loss 偏高、但看不出哪里错」。"
          % (VOCAB_PAD - VOCAB))
    print("     ★★ **一处出处纪律**：Megatron **具体怎么**屏蔽（置 `-inf`？换 loss 的 index？）"
          "在 `%s` 里**没有取到** ⇒ 已记 `UNSOURCED`：`%s`；"
          "本程序这一支是**本库的构造**（`UNSOURCED` 第 2 项正是这件事）。"
          % (specs.MEGATRON_TP_ARXIV_ID, specs.UNSOURCED[1]))
    check("C.3", float(s_bad.mean()) < 1.0 - TOL_ERR
          and abs(float(s_ok.mean()) - 1.0) <= TOL_MACHINE
          and md3 > TOL_ERR and leak0 < TOL_ERR,
          "★★★ **失败成立**：不屏蔽 pad ⇒ 前 `%d` 列的概率和**平均**降到 **%.6f**（< 1）"
          "（最大漏失 **%.6f**）、两种做法的概率差 **%.6f** ⇒ ★ **「补齐」必须配一个「屏蔽」**，"
          "否则概率是错的。★★ 而**对照**（pad 位置恰好为 `0`）的漏失只有 `%.3e`"
          "（< `%g`）⇒ ★★ **这个失败模式会被「pad 值恰好是 0」掩盖** ⇒ "
          "判据的**量法口径**必须写明（本条把两种口径**都**钉住）"
          % (VOCAB, float(s_bad.mean()), leak, md3, leak0, TOL_ERR))


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
    print("  ★★★ **禁区回链（本程序一行都不重写）**：TP 的三条"
          "（通信在**每一层内部** · 通信量 ∝ **激活** · **只在高带宽互联上可用 / 通常限于"
          "单节点 8 卡**）已经由 `%s` 逐字写完。" % specs.TP_BOUNDARY_REF)
    print("     另有：**注意力变体的结构**（MHA / MQA / GQA 的定义、KV head 与 Q head 的配比）"
          "→ `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`"
          "（**本程序只写「头怎么分给设备」**）；GEMM 的 tiling → `40-kernels/gemm/`；"
          "通信原语的算法与复杂度 → `110-communication` 与 `performance-models/02:209`。")
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

    measured = {"SEED": SEED, "B_SIZE": B_SIZE, "S_LEN": S_LEN, "N_HEADS": N_HEADS,
                "D_HEAD": D_HEAD, "D_MODEL": D_MODEL, "TP": TP, "VOCAB": VOCAB,
                "VOCAB_PER_GPU_MULTIPLE": VOCAB_PER_GPU_MULTIPLE, "VOCAB_PAD": VOCAB_PAD,
                "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT}
    expect = {"SEED": 20260952, "B_SIZE": 3, "S_LEN": 4, "N_HEADS": 8,
              "D_HEAD": 8, "D_MODEL": 64, "TP": 4, "VOCAB": 30,
              "VOCAB_PER_GPU_MULTIPLE": 4, "VOCAB_PAD": 32,
              "FLOAT_EXACT_LIMIT": 9007199254740992}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", D_MODEL == N_HEADS * D_HEAD and N_HEADS % TP == 0
          and VOCAB % TP != 0 and VOCAB_PAD % (VOCAB_PER_GPU_MULTIPLE * TP) == 0
          and VOCAB_PAD // (VOCAB_PER_GPU_MULTIPLE * TP) == 2,
          "★★★ **两组「整除性」是【故意相反】的，这是本程序的全部设计**："
          "**头**能整除（`N_HEADS=%d %% TP=%d == 0`）而**词表**不能（`VOCAB=%d %% TP=%d == %d`）"
          "⇒ ★ **同一篇里「整除了」与「不整除」是两种不同的处境**，处理方式也不同"
          "；★ `VOCAB_PER_GPU_MULTIPLE=%d` 是 [C] 的 `128` 的**玩具替身**"
          "（真值见 `B.2`），`VOCAB_PAD` 是 `%d × %d` 的整数倍"
          % (N_HEADS, TP, VOCAB, TP, VOCAB % TP, VOCAB_PER_GPU_MULTIPLE,
             VOCAB_PER_GPU_MULTIPLE, TP))

    th = {"TOL_MACHINE": TOL_MACHINE, "TOL_GUARD": TOL_GUARD, "TOL_ERR": TOL_ERR}
    th_expect = {"TOL_MACHINE": 1e-12, "TOL_GUARD": 1e-6, "TOL_ERR": 1e-3}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 3 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
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
          and specs.MEGATRON_HEAD_LOCAL_QUOTE
          == "in a column parallel fashion such that the matrix multiply corresponding to each "
             "attention head is done locally on one GPU"
          and specs.MEGATRON_EMBED_ALLREDUCE_QUOTE
          == "Since each partition now only contains a portion of the embedding table, "
             "an all-reduce ("
          and specs.MEGATRON_VOCAB_HUGE_QUOTE
          == "which is huge due to vocabulary size being large."
          and specs.MEGATRON_VOCAB_FUSE_QUOTE
          == "with the cross entropy loss which reduces the dimension to"
          and specs.MEGATRON_VOCAB_PER_GPU_MULTIPLE == 128
          and specs.MEGATRON_VOCAB_PAD_TP_DEGREE == 8
          and specs.MEGATRON_VOCAB_PAD_QUOTE
          == "pad the vocabulary such that it is divisible by"
          and specs.MEGATRON_VOCAB_PAD_RESULT_QUOTE
          == ", resulting in a padded vocabulary size of 51,200."
          and specs.MEGATRON_VOCAB_ORIGINAL == 50257
          and specs.MEGATRON_VOCAB_PAD_SIZE == 51200
          and specs.MEGATRON_8WAY_NOTE_QUOTE
          == "The configuration with 1.2 billion parameters fits on a single GPU whereas the "
             "8 billion parameter model requires 8-way model parallelism (8 GPUs)."
          and specs.TP_BOUNDARY_REF
          == "10-foundations/performance-models/02-amdahl-and-scaling.md:223",
          "★★ [C] `%s`（**不确认同行评审**）的**十三条**常量逐位一致 —— ★ `128 × 8 = 1024` "
          "与 `50257 → 51200` 是**真实规模**上的那一组数（本程序的玩具规模是 `4 × %d` 与 "
          "`%d → %d`）；★ `MEGATRON_8WAY_NOTE_QUOTE` **不是**「TP 限于单节点」的出处"
          "（那句在 `2104.04473 §3.2`，见 `03_saturation.py` 的 `D.5`）"
          % (specs.MEGATRON_TP_ARXIV_ID, TP, VOCAB, VOCAB_PAD))
    check("D.5b", specs.MEGATRON_HEAD_LOCAL_QUOTE.count("locally on one GPU") == 1
          and specs.MEGATRON_EMBED_ALLREDUCE_QUOTE.endswith("all-reduce (")
          and specs.MEGATRON_VOCAB_PAD_QUOTE in
          "we pad the vocabulary such that it is divisible by"
          and specs.MEGATRON_VOCAB_PAD_RESULT_QUOTE.startswith(", resulting in")
          and specs.MEGATRON_VOCAB_PAD_SIZE // 1024 == 50,
          "★★ **头切分那句的作用域**：它只说「每个 head 本地算」+「**不立即**通信」——"
          "★ 不得外推成「头切分不需要通信」（原文后面紧接着的 output linear 层是按行并行的，"
          "那次 all-reduce 仍然发生）；★ `51200 // 1024 == 50` ⇒ 补齐后的词表是 `1024` 的整数倍")
    got = doc_has_boundary_ref("02-heads-and-vocab.md", specs.TP_BOUNDARY_REF)
    if got is None:
        print("  [skip] D.5c 配套文档不在（`mutate.py` 的副本目录里只有 `*.py`）⇒ 跳过；"
              "★ 字面量本身已在 `D.5` 里逐位钉住")
    else:
        check("D.5c", got,
              "★★ **边界锚点已扩散到本文档**：`%s` 逐字出现在 `02-heads-and-vocab.md` 里 ⇒ "
              "「本域不重写那三条」这句话**有可执行的检查**（`P1`：订正不扩散是默认失败）"
              % specs.TP_BOUNDARY_REF)
    check("D.5d", specs.UNSOURCED == ("tensor_parallel_vocab_pad_multiple_128_rationale",
                                     "tensor_parallel_pad_logits_masking_mechanism")
          and "tensor_parallel_pad_logits_masking_mechanism" == specs.UNSOURCED[1],
          "★★ `specs.UNSOURCED` 的 **2** 项逐位钉住 —— ★ 第 2 项（**怎么屏蔽 pad 位置**）"
          "正是本程序 `C.3` 用到的那件事：**本程序只有「不屏蔽会错」这个本库构造的反例**，"
          "**没有**声称 Megatron 的做法")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(head_parallel_attention)
    src_b = inspect.getsource(reference_attention)
    src_c = inspect.getsource(vocab_parallel_logits)
    src_d = inspect.getsource(reference_logits)
    check("D.7", len({src_a, src_b, src_c, src_d}) == 4,
          "★ 防 `P17`：被测（`head_parallel_attention` %d 字符 / `vocab_parallel_logits` %d）与"
          "**两条独立参照**（`reference_attention` %d / `reference_logits` %d）"
          "**源码两两不同**" % (len(src_a), len(src_c), len(src_b), len(src_d)))
    watched = {"head_parallel_attention", "vocab_parallel_logits", "padded_vocab_logits",
               "silent_truncate", "all_reduce_sum", "all_gather_concat",
               "softmax_unmasked", "softmax_masked", "embed_column_split_lookup"}
    called = _callee_names(reference_attention) | _callee_names(reference_logits)
    check("D.7b", not (watched & called),
          "★ **AST 调用表**：**两条参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    _CALLS.clear()
    q, k, v = toy_heads(B_SIZE, S_LEN, N_HEADS, D_HEAD, SEED)
    head_parallel_attention(q, k, v, TP)
    seen = sum(_CALLS.values())
    _CALLS.clear()
    reference_attention(q, k, v)
    xx, ee = toy_vocab(B_SIZE, S_LEN, D_MODEL, VOCAB, SEED)
    reference_logits(xx, ee)
    left = sum(_CALLS.values())
    check("D.7c", seen > 0 and left == 0,
          "★★ **调用计数器两向都量**：被测路径跑一次 ⇒ 计数 = **%d**（> 0，计数器**非空**）；"
          "**两条参照**跑完 ⇒ 计数 = **%d**（必须**恰好为 0**）⇒ "
          "「参照不调用被测」不是靠读代码相信的" % (seen, left))

    # ── ★★ 恒真检查 ────────────────────────────────────────────────────────
    print()
    print("  ── D.10 ★★★ **恒真检查**：头切分与 vocab 列切的「与单卡逐位相同」，性质**不同** ──")
    q, k, v = toy_heads(B_SIZE, S_LEN, N_HEADS, D_HEAD, SEED)
    d_head = float(np.abs(head_parallel_attention(q, k, v, TP)
                          - reference_attention(q, k, v)).max())
    xx, ee = toy_vocab(B_SIZE, S_LEN, D_MODEL, VOCAB, SEED)
    # ★★ vocab 列切**只能**在整除时跑（`VOCAB=30` 时 `np.split` 会抛错，见 `C.1`）
    #    ⇒ 这里用**补齐后**的词表（`VOCAB_PAD=32` 能被 `TP=4` 整除）来量那一刀。
    ee_pad = np.zeros((D_MODEL, VOCAB_PAD))
    ee_pad[:, :VOCAB] = ee
    rows = []
    for bb, ss in ((B_SIZE, S_LEN), (2, 8), (4, 8), (1, 16), (8, 4)):
        xe = np.random.default_rng(SEED + bb * 100 + ss).standard_normal((bb, ss, D_MODEL))
        colv = np.concatenate([xe @ w for w in np.split(ee_pad, TP, axis=1)], axis=-1)
        rows.append((bb, ss, float(np.abs(colv - xe @ ee_pad).max())))
    print("     **头切分** `max|diff|` = **%.3e**（★ 每卡做的**就是**单卡时那几个 head 的"
          "同一批运算：形状没变 ⇒ 逐位相同）" % d_head)
    print("     **vocab 维的列切**：差**随形状变** ——")
    for bb, ss, dd in rows:
        print("        `(b=%d, s=%d)` ⇒ `max|diff|` = **%.3e**%s"
              % (bb, ss, dd, "（**恰好为 0**）" if dd == 0.0 else ""))
    dmax = max(dd for _, _, dd in rows)
    dmin = min(dd for _, _, dd in rows)
    print("     ⇒ ★★★ **两件事的性质不同**：")
    print("        · **头切分**：`0.0` 是**结构性的**（同一批运算被重新分配）⇒ **恒真**；")
    print("        · **vocab 列切**：`GEMM` 的 `N`（与 `M`）变了 ⇒ BLAS 可能换**分块/向量化**"
          "路径 ⇒ 差**有时 `0`、有时非零**（本机实测落在 `[%.1e, %.1e]`）。" % (dmin, dmax))
    print("     ★★★ **因此两条判据【都不能当 A 类】，但理由不同**：")
    print("        头切分是**恒真**（只重述定义）；vocab 列切是**差值来自 BLAS 路径噪声**"
          "（与「切分对不对」无关，且**不保证非零**）。")
    print("     ★★ **一处必须写明的复核口径**：本库的 `probes/probe_100a.py` 与 `probe_100h.py`"
          "在**它们自己那组形状**上测到的是 `0.000e+00` —— **两者都对**，"
          "但它们共同支持的是「**在该组形状下**的观测」，**不是**「列切恒等于单卡」。")
    check("D.10", d_head == 0.0 and dmax > 0.0 and dmax <= TOL_MACHINE * 1e2 and dmin >= 0.0,
          "★★★ **恒真性的性质已被机械证明**：**头切分**的差**恒为 `0.0`**（结构性）；"
          "**vocab 列切**的差**随形状变**（本机 `[%.1e, %.1e]`，含 `0.0` 与非零两种）⇒ "
          "★ 它**不是恒真，而是「BLAS 路径依赖」** ⇒ 两条都**不能当 A 类**。"
          "★ 这条断言**能失败**：若某处切分被改成改变求和顺序的实现（例如把 head 交错分给设备），"
          "`d_head` 就不再是 `0.0`" % (dmin, dmax))


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
    print("02_heads_vocab.py —— 头切分 · vocab 并行 · **不整除的三种处置**")
    print("=" * 74)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 本程序的正题：`VOCAB=%d` **不被** `TP=%d` 整除 —— 三种处置"
          "（抛错 / 截断 / 补齐）各自的后果都要跑出来。" % (VOCAB, TP))
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
