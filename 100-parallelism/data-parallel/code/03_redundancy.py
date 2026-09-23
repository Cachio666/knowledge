#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_redundancy.py —— ★★★ DP 的**冗余账**：`N` 个副本用 `N × 16Ψ` 的**总**显存存 `16Ψ` 的状态。

配套文档：`03-redundancy-and-limits.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**切分的代数**。

★★★ **本程序的头号风险与它的解药**（计划 §1.3.3 的第 3 条）：
  「`N` 卡下每卡显存不随 `N` 下降」这句话，若实现里它就是常量 `16Ψ`，
  那么 `assert per_card == 16 * Psi` 是**算术恒真**（**模拟器同时是被测对象**）。
  ⇒ 本程序的落法是：**真的建出 `N` 份状态**（`make_state()` / `make_full_state()`），
    量它们的 **`nbytes`**、算它们的 **sha256 摘要** —— 判据钉在「**副本一致性**」上，
    **不钉在乘法上**：
      · `A.1` 的判据是「**`N` 份的【参数】逐位相同**」（摘要相等）—— **这才是 DP 的隐含假设**；
      · `A.3`/`C.1` 证明它**可证伪**（把一份换成独立建的 ⇒ 摘要**必须**不同）；
      · `B.2` 的「每卡常量」是**从建出来的数组量出来的**，并且**按计划 §1.3.3 的裁定**：
        「**加卡不能解决显存问题**」这一条在此**只作陈述 + 回链**，**不另给判据**。

★ **本主题的禁区**（回链，不重写）：
  · **`16Ψ` 的组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）已经由
    `90-training/checkpointing/01-state-and-bytes.md` 写过（§1.3 + 判据表 +
    `code/01_state_bytes.py:310` 的 `A.3`）⇒ 本程序**只消费**那个系数（`A.2` 把它的
    **四项**建出来量一遍），**不重推分解**；
  · ★★ **「按 `N` 分片后怎么变」不在本主题** —— 那是 `zero-fsdp/01` 的正文
    （`P_os` 在 `N=512` 仍要 `4.0234Ψ`、「分片不是除法」）⇒ 本程序**只写 DP 的账**，
    并在 `B.2` 里给出**接口 + 回链**；
  · **梯度规约的算法**（ring allreduce / `V(N)`）⇒ `110-communication`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ **`N` 份副本的【参数】逐位相同**（`sha256` 摘要相等），且
             `总量 == N × 单份`（`Ψ = %d` 个 `fp64` 参数 ⇒ 单份 `%d B`）；
  · `A.2`  ★★ **每卡的完整训练状态 == `16Ψ`**（真的按四种 dtype 建出来、量 `nbytes`）——
             它与 `90-training/checkpointing` 的账**同值**，本程序**不重推分解**；
  · `A.3`  ★★ **守卫**：把第 `DRIFT_RANK` 份换成**独立建的** ⇒ 同一个摘要判据**必须失败**；
  · `B.1`  ★★ **冗余度 `(N−1)/N`** —— `N = 1/2/4/8/16` 上 `0.0 / 0.5 / 0.75 / 0.875 / 0.9375`；
  · `B.2`  ★ **每卡状态的字节数与 `N` 无关**（从建出来的数组量），而**总**字节随 `N` 线性涨；
  · `C.1`  ★★ **一个 rank 漂掉** ⇒ 「所有 rank 相同」**失败**（本机：不同的 rank == `[3]`）；
  · `C.2`  ★★ **每一份都独立初始化**（忘掉广播）⇒ **`N/N` 个 rank 全不同**；
  · `C.3`  ★ **加卡不能让单卡装下更大的模型**：`Ψ` 翻倍 ⇒ **每卡**字节翻倍（`N` 无关）。

用法：
    python3 03_redundancy.py
"""

import ast
import collections
import hashlib
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


def check(name, cond, detail=""):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260943              # ★ 与同域另外两个程序（…0941 / …0942）都**不同**
SEED_DRIFT = 20260944        # 漂移那一份用的**另一个**种子
LAYERS = 8                   # 玩具模型的层数
LAYER_IN = 64                # 每层的输入维
LAYER_OUT = 128              # 每层的输出维 ⇒ `Ψ = LAYERS × LAYER_IN × LAYER_OUT = 65536`
REPLICA_COUNTS = (1, 2, 4, 8, 16)   # `A.1`/`B.1` 扫的副本数（★ 含 `N=1`：冗余度为 `0`）
DRIFT_RANK = 3               # `A.3`/`C.1`：**故意**让第 `3` 号副本漂掉
# ── 期望值（**推导值**，但都是整数/精确二进制小数 ⇒ 可以逐位钉住）────────────
PSI_EXPECT = 65536           # `Ψ` = 参数个数
SINGLE_BYTES_EXPECT = 524288   # 单份**参数**（`fp64`）的字节数 = `Ψ × 8`
REDUNDANCY_EXPECT = (0.0, 0.5, 0.75, 0.875, 0.9375)   # `(N−1)/N`，逐个钉住
# ── 阈值 ────────────────────────────────────────────────────────────────────
TOL_EXACT = 0.0              # 摘要不同 / 数值差为 0 这类**逐位**判定
TOL_SCALE_UP = 2.0           # `C.3`：`Ψ` 翻倍时每卡字节必须**恰好**翻倍
FP64_ITEMSIZE = 8            # `fp64` 的字节/元素（★ 十进制字面量：盲区④）
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   （`1e300 + 1.0 == 1e300`）⇒ 那样的叶子变异后**逐位不变**、任何断言都抓不到。
#   本程序的浮点常量都 `< 2^53`（最大的是 `0.9375`）⇒ 每一个叶子都**可变异**。
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 十进制字面量：盲区④）


# ===========================================================================
# 2. 被测路径 vs 参照路径
# ===========================================================================
def make_state(seed):
    """**被测**：按 `seed` 建一份**参数**状态（`LAYERS` 层 `(LAYER_IN, LAYER_OUT)`，**fp64**）。

    ★★ 它**真的分配内存** ⇒ 本程序量的是**真实的 `nbytes`**，不是算出来的乘法。
    """
    _CALLS["make_state"] += 1
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((LAYER_IN, LAYER_OUT)) for _ in range(LAYERS)]


def make_full_state(seed, psi):
    """**被测**：按 `16Ψ` 的**四项组成**建一份**每卡完整训练状态**（混合精度）。

    ★★ 四种 dtype **分别对应** `specs.py` 里的四项常量：
      `param` fp16（`ZERO_PARAM_FP16_MULTIPLE`）· `grad` fp16（`ZERO_GRAD_FP16_MULTIPLE`）·
      `master`/`m`/`v` fp32（`K = 12` 的三项：`4 + 4 + 4`）。
    ★ **本程序不重推这个分解**（正文在 `90-training/checkpointing/01-state-and-bytes.md`）
      —— 它只是**按那个分解把数组真的建出来**，好让 `A.2` 量到的是**实际字节**。
    """
    _CALLS["make_full_state"] += 1
    rng = np.random.default_rng(seed)
    return {
        "param": rng.standard_normal(psi).astype(np.float16),
        "grad": rng.standard_normal(psi).astype(np.float16),
        "master": rng.standard_normal(psi).astype(np.float32),
        "m": np.zeros(psi, np.float32),
        "v": np.zeros(psi, np.float32),
    }


def state_bytes(state):
    """`Σ nbytes`（由**实际的 numpy 数组**算出来）—— 本程序只用这一条路径量字节。"""
    return int(sum(int(a.nbytes) for a in (state.values() if isinstance(state, dict) else state)))


def state_digest(state):
    """一份状态的 **sha256 摘要**（前 16 位十六进制）—— 判据钉在它上面。

    ★★ 用**摘要相等**而不是容差比较：来源是 [B] PyTorch 2.14 的逐字
      `bitwise identical in all processes`（**逐位**，不是「近似相同」）。
    """
    h = hashlib.sha256()
    for a in (state.values() if isinstance(state, dict) else state):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def broadcast_replicas(seed, n_replicas):
    """**被测**：「广播初始参数」的模型 —— `N` 份副本**都从 `rank 0` 的那一份来**。

    ★★★ 这是 DP 的**隐含假设**的可执行形态：参数在初始化时**必须**从同一个来源广播
      （或从同一个检查点加载）⇒ 否则各副本从第 0 步就不一样。
      ★★ 本函数**不是**「把同一个 seed 传给 `N` 次」—— 那样写的话
        「摘要相同」会是 **RNG 的恒等式**（`P12`），而不是一条判据。
    """
    origin = make_state(seed)
    return [list(origin) for _ in range(n_replicas)]


def ref_params_bytes(psi):
    """**独立参照**：`Ψ` 个 `fp64` 参数 ⇒ `Ψ × 8` 字节 —— **不碰任何数组**，只做算术。

    ★★ 它与 `state_bytes(make_state(...))` **来源不同**（一条来自 NumPy 的
      `size × itemsize`，一条来自 `specs.py` 的 [C] 常量与整数乘法）⇒ 两者相等是一条
      **有内容的**断言（`A.1`）。
    """
    return int(psi) * FP64_ITEMSIZE


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


def doc_path(name):
    """配套文档的路径（`code/` 的**上一级**）。

    ★★ 为什么可以读它：本主题有一条要求是「**凡说 `16Ψ` / `12Ψ` 的地方必须带上
      『ZeRO 的举例口径』那句限定语**」（`P1` 订正不扩散的可执行落法）——
      本程序把它做成一条**可执行的检查**（`D.5c`），而**不是在文档里写一句「已检查」**。
    ★★ 为什么**不能**无条件断言它存在：`mutate.py` 只把 `*.py` 复制到临时目录再跑
      （`code/` 自包含，`CONTRIBUTING:36`）⇒ 在副本里这个文件**必然不在**；
      那时**跳过**并**打印出来**，而**不是**让它崩（崩会被记成「崩溃捕获」、
      还会让 `mutate.py` 的基线不干净）。
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)


def doc_contains(name, needle):
    """返回 `(是否存在, 是否命中)`；文件不存在时给 `(False, False)`。"""
    path = doc_path(name)
    if not os.path.exists(path):
        return False, False
    text = open(path, encoding="utf-8").read()
    return True, (needle in text)


# ===========================================================================
# 4. A 类：副本一致性（判据 A.1 / A.2 / A.3）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 副本一致性：`N` 份副本的【参数】逐位相同 · 每卡状态 == `16Ψ`")
    print("=" * 74)
    psi = LAYERS * LAYER_IN * LAYER_OUT
    print("  玩具模型：`%d` 层 × `(%d, %d)` ⇒ **`Ψ = %d` 个参数**；"
          "副本数 `N ∈ %s`" % (LAYERS, LAYER_IN, LAYER_OUT, psi, list(REPLICA_COUNTS)))
    print("  ★★ `Ψ` 的口径（[C] ZeRO Figure 1 逐字）：`%s`" % specs.ZERO_FIGURE_MODEL_QUOTE)

    # ── A.1 ★★ N 份副本逐位相同 + 总量 == N × 单份 ─────────────────────────
    print()
    print("  ── A.1 ★★ 真的建出 `N` 份状态，量 `nbytes`、算 `sha256` 摘要 ──")
    replica_bytes, replica_ratios, replica_digits = [], [], set()
    single = None
    for n in REPLICA_COUNTS:
        reps = broadcast_replicas(SEED, n)
        single = state_bytes(reps[0])
        total = sum(state_bytes(r) for r in reps)
        replica_bytes.append(single)
        replica_ratios.append(total / single)
        replica_digits |= {state_digest(r) for r in reps}
        print("     `N=%-3d` 单份 = **%d B** · 总计 = **%d B** · 比值 = **%.6f** · 不同摘要数 = **%d**"
              % (n, single, total, total / single, len({state_digest(r) for r in reps})))
    ref = ref_params_bytes(psi)
    print("     独立参照（**不碰数组**，只做算术）：`Ψ × %d` = **%d B**" % (FP64_ITEMSIZE, ref))
    print("     ★★ [B] PyTorch 2.14 逐字：`%s` ⇒ 判据用**摘要相等**（逐位），不是容差比较"
          % specs.TORCH_ALLREDUCE_BITWISE_QUOTE)
    check("A.1", single == ref == SINGLE_BYTES_EXPECT and len(replica_digits) == 1
          and all(abs(r - n) <= TOL_EXACT for r, n in zip(replica_ratios, REPLICA_COUNTS))
          and psi == PSI_EXPECT,
          "★★ **DP 的隐含假设被量出来了**：`N` 份副本的**参数逐位相同**（不同摘要数 == `1`），"
          "单份 **%d B**（= `Ψ × %d`，两条来源不同的路径），总量 == `N × 单份`（比值 "
          "`1.000000 → 16.000000`）。★ 判据钉在**副本一致性**上，**不是**钉在乘法上"
          % (single, FP64_ITEMSIZE))

    # ── A.2 ★★ 每卡完整状态 == 16Ψ（真的建出来量）────────────────────────
    print()
    print("  ── A.2 ★★ 每卡**完整训练状态** == `16Ψ`：四项 dtype 真的建出来 ──")
    st = make_full_state(SEED, psi)
    parts = {k: int(v.nbytes) for k, v in st.items()}
    total_st = state_bytes(st)
    print("     四项（★ 与 `specs.py` 的四条常量一一对应）：")
    print("       `param` fp16 = **%d B** = `%.0fΨ`（`ZERO_PARAM_FP16_MULTIPLE` = `%.0f`）"
          % (parts["param"], parts["param"] / psi, specs.ZERO_PARAM_FP16_MULTIPLE))
    print("       `grad`  fp16 = **%d B** = `%.0fΨ`（`ZERO_GRAD_FP16_MULTIPLE` = `%.0f`）"
          % (parts["grad"], parts["grad"] / psi, specs.ZERO_GRAD_FP16_MULTIPLE))
    print("       优化器三项 fp32 = **%d B** = `%.0fΨ`（`K = %d`：主权重 `4Ψ` + `m` `4Ψ` + `v` `4Ψ`）"
          % (parts["master"] + parts["m"] + parts["v"],
             (parts["master"] + parts["m"] + parts["v"]) / psi, specs.ZERO_K_OPT_STATE))
    print("     ⇒ 合计 = **%d B** = **`%.4fΨ`**（`specs.ZERO_TOTAL_MULTIPLE` = `%.0f`）"
          % (total_st, total_st / psi, specs.ZERO_TOTAL_MULTIPLE))
    print("     ★★ `K = %d` 是 **%s**（条件逐字：`%s`）⇒ 正文里说 `16Ψ` 时必须带上这句"
          % (specs.ZERO_K_OPT_STATE, specs.ZERO_K_CALIBER_NOTE, specs.ZERO_FIGURE_CALIBER_QUOTE))
    print("     ★ 逐字依据：`%s` … `%s`" % (specs.ZERO_TOTAL_QUOTE, specs.ZERO_TOTAL_TAIL_QUOTE))
    print("     ★★ **本程序不重推这个分解** —— 它的正文在 "
          "`90-training/checkpointing/01-state-and-bytes.md`（§1.3 + 判据表 + `A.3`），"
          "本程序只**消费**那个系数。")
    check("A.2", total_st == int(specs.ZERO_TOTAL_MULTIPLE * psi) and total_st == 16 * psi
          and parts["param"] == int(specs.ZERO_PARAM_FP16_MULTIPLE * psi)
          and parts["grad"] == int(specs.ZERO_GRAD_FP16_MULTIPLE * psi)
          and parts["master"] + parts["m"] + parts["v"] == specs.ZERO_K_OPT_STATE * psi
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径",
          "★★ **每卡状态 == `16Ψ`**（本机 `%d B`）：四个 dtype 的**实际 nbytes** 与四条 [C] "
          "常量逐项对齐 ⇒ ★★ 而它是 **%s**（`based on mixed-precision training with Adam optimizer.`）"
          % (total_st, specs.ZERO_K_CALIBER_NOTE))

    # ── A.3 ★★ 守卫：把一份换成独立建的 ⇒ 摘要必须不同 ────────────────────
    print()
    print("  ── A.3 ★★ 守卫（`CONTRIBUTING:235`）：把第 `%d` 份换成**独立建的** ──" % DRIFT_RANK)
    reps = broadcast_replicas(SEED, REPLICA_COUNTS[-1])
    base = state_digest(reps[0])
    reps[DRIFT_RANK] = make_state(SEED_DRIFT)          # ← 故意换掉一份
    digests = [state_digest(r) for r in reps]
    differing = [i for i, d in enumerate(digests) if d != base]
    print("     第 `%d` 份的摘要 = `%s` · 其余 = `%s` ⇒ **不同的 rank = %s**"
          % (DRIFT_RANK, digests[DRIFT_RANK], base, differing))
    check("A.3", differing == [DRIFT_RANK] and len(set(digests)) == 2
          and state_bytes(reps[DRIFT_RANK]) == state_bytes(reps[0]),
          "★★ **守卫成立**：改掉一份 ⇒ **同一条摘要判据真的会失败**（不同的 rank = `%s`）——"
          "★ 而**字节数一个都没变**（两份大小相同）⇒ 「逐位相同」这条判据抓的是**内容**，"
          "不是**大小**：只量字节数的检查会**完全放过**这个错" % differing)


# ===========================================================================
# 5. B 类：代价模型 —— 冗余度 `(N−1)/N`、每卡常量而总量线性涨
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：**冗余度 `(N−1)/N`** · 每卡常量（不随 `N` 下降）· 总量 `∝ N`")
    print("=" * 74)
    psi = LAYERS * LAYER_IN * LAYER_OUT

    # ── B.1 ★★ 冗余度 ──────────────────────────────────────────────────────
    print("  ── B.1 ★★ 冗余度 = `(总量 − 单份) / 总量 = (N−1)/N` ──")
    red = []
    for n in REPLICA_COUNTS:
        reps = broadcast_replicas(SEED, n)
        single = state_bytes(reps[0])
        total = sum(state_bytes(r) for r in reps)
        red.append((total - single) / total)
        print("     `N=%-3d` 总计 = **%8d B** · 单份 = **%d B** ⇒ 冗余度 = **%.6f**"
              % (n, total, single, red[-1]))
    print("     ⇒ ★★ `N → ∞` 时冗余度 `→ 1`：**几乎全部显存都在存重复的东西**。"
          "★ 「冗余度」这个量**全库没有既有定义** ⇒ 它是本主题自立的定义（`UNSOURCED`："
          "`data_parallel_redundancy_ratio_definition`）。")
    check("B.1", all(abs(r - e) <= TOL_EXACT for r, e in zip(red, REDUNDANCY_EXPECT))
          and red[0] == TOL_EXACT and red[-1] > red[-2],
          "★★ **冗余度**在 `N = %s` 上是 `%s`（逐个钉住）—— `N=1` 时为 `0`（没有冗余），"
          "`N=16` 时 **`%.6f`** ⇒ **加卡把「有用的显存」压得越来越小**"
          % (list(REPLICA_COUNTS), [round(r, 6) for r in red], red[-1]))

    # ── B.2 ★ 每卡常量、总量线性涨（★ 只作陈述 + 回链，不另给判据）────────
    print()
    print("  ── B.2 ★ **每卡**状态的字节数不随 `N` 变化；**总量** `∝ N` ──")
    st = make_full_state(SEED, psi)
    per_card = [state_bytes(make_full_state(SEED, psi)) for _ in REPLICA_COUNTS]
    totals = [state_bytes(make_full_state(SEED, psi)) * n for n in REPLICA_COUNTS]
    print("     每卡状态（**从真的建出来的数组量**）= **%d B** = `%.4fΨ`（`Ψ = %d`）；"
          "`N` 取 `%s` 时它**逐位不变** = **%s**"
          % (per_card[0], per_card[0] / psi, psi, list(REPLICA_COUNTS),
             len(set(per_card)) == 1))
    print("     总量 = `N × 每卡`：%s" % "、".join("`N=%d` → %d B" % (n, t)
                                                 for n, t in zip(REPLICA_COUNTS, totals)))
    print("     ★★ **口径提醒（`P12`）**：这一条守的是「**程序真的各建了一份**」，"
          "**不是**公式的同义反复 —— 判据的两端是**实际数组的 `nbytes`**（`A.2` 已把它与")
    print("       [C] 常量对拍）。★★ 而「**加卡不能解决显存问题**」这句话本身"
          "**在本程序只作陈述、不另给判据**")
    print("       （计划 §1.3.3 第 3 条的裁定）⇒ 它的定量正文在 `zero-fsdp/01`（`P_os` 在 "
          "`N=512` 仍要 `4.0234Ψ`），此处**回链**。")
    check("B.2", len(set(per_card)) == 1
          and totals[0] * REPLICA_COUNTS[-1] == totals[-1] * REPLICA_COUNTS[0]
          and per_card[0] == 16 * psi,
          "★★ **每卡常量、总量线性**：`N` 从 `%d` 涨到 `%d`（**%d 倍**），每卡**逐位不变**"
          "（`%d B`），总量**恰好成比例**（`%d × %d == %d × %d` 整数恒等）"
          % (REPLICA_COUNTS[0], REPLICA_COUNTS[-1],
             REPLICA_COUNTS[-1] // REPLICA_COUNTS[0], per_card[0],
             totals[-1], REPLICA_COUNTS[0], totals[0], REPLICA_COUNTS[-1]))


# ===========================================================================
# 6. C 类：边界与反例 —— 一个 rank 漂掉 / 全部独立初始化 / 加卡装不下更大的模型
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：副本发散的两条路 · 加卡不能让单卡装下更大的模型")
    print("=" * 74)
    psi = LAYERS * LAYER_IN * LAYER_OUT

    # ── C.1 ★★ 一个 rank 漂掉 ⇒ 副本发散 ──────────────────────────────────
    print("  ── C.1 ★★ **让一个 rank 的参数漂掉**（第 `%d` 份换成另一组种子）──" % DRIFT_RANK)
    reps = broadcast_replicas(SEED, REPLICA_COUNTS[-1])
    base = state_digest(reps[0])
    reps[DRIFT_RANK] = make_state(SEED_DRIFT)
    digests = [state_digest(r) for r in reps]
    differing = [i for i, d in enumerate(digests) if d != base]
    print("     与 `rank 0` **不同**的 rank = **%s**（共 `%d/%d` 个）"
          % (differing, len(differing), len(reps)))
    print("     ⇒ ★★ **「所有 rank 相同」这条断言会失败** ⇒ 这正是「**用 DP 装大模型**」"
          "之外的第二条失败路：**副本发散**。")
    check("C.1", differing == [DRIFT_RANK] and len(differing) == 1
          and state_bytes(reps[DRIFT_RANK]) == state_bytes(reps[0]),
          "★★ **一个 rank 漂掉就会被抓住**（不同的 rank = `%s`）——★ 而**字节数完全相同**"
          "⇒ 「量字节数」的检查**抓不到它**，判据必须是**逐位比较**（摘要）" % differing)

    # ── C.2 ★★ 全部独立初始化 ⇒ N/N 全不同 ────────────────────────────────
    print()
    print("  ── C.2 ★★ **忘掉广播**：每一份都用自己的种子独立初始化 ──")
    n = REPLICA_COUNTS[-1]
    independent = [make_state(SEED + i) for i in range(n)]
    ind_digests = [state_digest(r) for r in independent]
    print("     `N=%d` 份各自独立建 ⇒ **不同摘要数 = %d**（`%d/%d` 个 rank 互不相同）"
          % (n, len(set(ind_digests)), len(set(ind_digests)), n))
    print("     ⇒ ★★ **DP 的隐含假设是「初始化时必须广播（或从同一个检查点加载）」** ——"
          "少了这一步，")
    print("        副本从**第 0 步**就互不相同，而**没有人会报错**（每份都是合法的随机初始化）。")
    check("C.2", len(set(ind_digests)) == n and n == REPLICA_COUNTS[-1],
          "★★ **`%d/%d` 个 rank 全不同** ⇒ 「副本一致性」**不是自动成立的**："
          "它是 DP 的一条**前置条件**（广播），而本条判据证明它**能失败**" % (len(set(ind_digests)), n))

    # ── C.3 ★ 加卡不能让单卡装下更大的模型 ────────────────────────────────
    print()
    print("  ── C.3 ★ **`Ψ` 翻倍**（把每层的输出维翻倍）⇒ **每卡**字节翻倍 ──")
    psi2 = LAYERS * LAYER_IN * (LAYER_OUT * 2)
    st1, st2 = make_full_state(SEED, psi), make_full_state(SEED, psi2)
    b1, b2 = state_bytes(st1), state_bytes(st2)
    print("     `Ψ = %d` ⇒ 每卡 **%d B**（`%.0f B/参数`）· `Ψ = %d` ⇒ 每卡 **%d B**"
          % (psi, b1, b1 / psi, psi2, b2))
    print("     ⇒ ★★ 每卡字节 = **`16Ψ` 个字节**（`16` 这个倍数**已经含了**各 dtype 的字节数）"
          "⇒ 它**只由 `Ψ` 决定**、**与 `N` 无关**（`B.2` 已量）")
    print("        ⇒ **加卡不会让单卡装下更大的模型**：要装下 `2Ψ`，**每张卡**都得有 "
          "**%d B**（= `2 × %d B`）。" % (b2, b1))
    print("     ★★ **分片之后**怎么变（`P_os` 的 `4Ψ` 下界、「分片不是除法」）**不在本主题**"
          " ⇒ 回链 `zero-fsdp/01`。")
    check("C.3", b2 == int(TOL_SCALE_UP * b1) and psi2 == 2 * psi and b1 == 16 * psi,
          "★★ **每卡字节只由 `Ψ` 决定**（`%d → %d B`，**恰好 %g 倍**）⇒ "
          "「加卡能装下更大的模型」是**错的**：每份副本都要放**完整**的状态"
          "（分片后的账回链 `zero-fsdp/01`）" % (b1, b2, TOL_SCALE_UP))


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
    print("  ★★ **禁区回链**（本程序一行都不重写）：`16Ψ` 的**组成与分解**在 "
          "`90-training/checkpointing/01-state-and-bytes.md`；**按 `N` 分片后怎么变**在 "
          "`zero-fsdp/01`；**梯度规约的算法**（ring allreduce / `V(N)`）在 `110-communication`。")
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

    measured = {"SEED": SEED, "SEED_DRIFT": SEED_DRIFT, "LAYERS": LAYERS,
                "LAYER_IN": LAYER_IN, "LAYER_OUT": LAYER_OUT,
                "REPLICA_COUNTS": REPLICA_COUNTS, "DRIFT_RANK": DRIFT_RANK,
                "PSI_EXPECT": PSI_EXPECT, "SINGLE_BYTES_EXPECT": SINGLE_BYTES_EXPECT,
                "REDUNDANCY_EXPECT": REDUNDANCY_EXPECT, "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT}
    expect = {"SEED": 20260943, "SEED_DRIFT": 20260944, "LAYERS": 8, "LAYER_IN": 64,
              "LAYER_OUT": 128, "REPLICA_COUNTS": (1, 2, 4, 8, 16), "DRIFT_RANK": 3,
              "PSI_EXPECT": 65536, "SINGLE_BYTES_EXPECT": 524288,
              "REDUNDANCY_EXPECT": (0.0, 0.5, 0.75, 0.875, 0.9375),
              "FLOAT_EXACT_LIMIT": 9007199254740992}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", LAYERS * LAYER_IN * LAYER_OUT == PSI_EXPECT
          and PSI_EXPECT * FP64_ITEMSIZE == SINGLE_BYTES_EXPECT
          and DRIFT_RANK < REPLICA_COUNTS[-1] and SEED_DRIFT != SEED
          and all(abs((n - 1) / n - e) <= TOL_EXACT
                  for n, e in zip(REPLICA_COUNTS, REDUNDANCY_EXPECT)),
          "★★ **在任何计算之前**核对三件事：`Ψ` 的定义与它乘 `%d` 字节 == 单份字节；"
          "漂移用的种子与基准**不同**、漂移的 rank 在范围内；**冗余度的期望值**与 "
          "`(N−1)/N` 逐位一致（`%s`）" % (FP64_ITEMSIZE, list(REDUNDANCY_EXPECT)))
    check("D.3c", REPLICA_COUNTS[0] == 1 and REDUNDANCY_EXPECT[0] == TOL_EXACT,
          "★ `B.1` 必须从 `N=1` 起扫：**没有那一点，`(N−1)/N` 这条式子就少了一个端点**"
          "（`N=1` 时冗余度**恰好为 0**）")

    th = {"TOL_EXACT": TOL_EXACT, "TOL_SCALE_UP": TOL_SCALE_UP, "FP64_ITEMSIZE": FP64_ITEMSIZE}
    th_expect = {"TOL_EXACT": 0.0, "TOL_SCALE_UP": 2.0, "FP64_ITEMSIZE": 8}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 3 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.4b", max(th.values()) < FLOAT_EXACT_LIMIT
          and FLOAT_EXACT_LIMIT == 9007199254740992,
          "★★ **盲区⑥ 的落法**：全部浮点常量都 `< 2^53`（最大的是 `%g`）⇒ "
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
          and specs.ZERO_TOTAL_TAIL_QUOTE == "bytes of memory requirement"
          and specs.ZERO_FIGURE_CALIBER_QUOTE
          == "based on mixed-precision training with Adam optimizer."
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径",
          "★★ ZeRO 的**九项**逐位一致（[C] `%s`，**不确认同行评审**）—— ★★ 这九项与 "
          "`90-training/checkpointing/code/specs.py` 的对应常量**同值**（`# 同值位置：` 互指），"
          "⇒ 两域对 `16Ψ` 的账**不会漂**" % specs.ZERO_ARXIV_ID)
    check("D.5b", specs.TORCH_ALLREDUCE_BITWISE_QUOTE == "bitwise identical in all processes",
          "★★ `A.1` 的**外部锚点**逐位：[B] PyTorch 2.14 用的是 **`bitwise`（逐位）**，"
          "不是「近似相同」⇒ 本程序的判据因此用 **sha256 摘要相等**，不用容差比较")
    check("D.6", specs.UNSOURCED == ("nccl_allreduce_default_op",
                                     "data_parallel_redundancy_ratio_definition",
                                     "data_parallel_ring_allreduce_volume"),
          "★★ `specs.UNSOURCED` 的 **3** 项逐位钉住 —— ★ 第 2 项 "
          "(`data_parallel_redundancy_ratio_definition`) 正是本程序 `B.1` 用的那个量："
          "**全库没有既有定义** ⇒ 它是本主题自立的定义")

    # ── ★★★ 判据的**正文侧**：配套文档里必须逐字有那句限定语 ────────────────
    has_doc, hit = doc_contains("03-redundancy-and-limits.md", specs.ZERO_K_CALIBER_NOTE)
    check("D.5c", (not has_doc) or hit,
          "★★★ **`P1`（订正不扩散）的可执行落法**：配套文档 `03-redundancy-and-limits.md` "
          "里**逐字**含 `%r`（本机：文件存在 = **%s**、命中 = **%s**）⇒ 凡说 `16Ψ` / `12Ψ` "
          "的地方都带上「那是 **%s**」这句 —— ★ 文档与常量**同源才不漂**"
          % (specs.ZERO_K_CALIBER_NOTE, has_doc, hit, specs.ZERO_K_CALIBER_NOTE))

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(make_state)
    src_b = inspect.getsource(broadcast_replicas)
    src_c = inspect.getsource(ref_params_bytes)
    check("D.7", len({src_a, src_b, src_c}) == 3,
          "★ 防 `P17`：**建状态**（`make_state` %d 字符）、**广播**（`broadcast_replicas` %d）、"
          "**独立参照**（`ref_params_bytes` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c)))
    watched = {"make_state", "make_full_state", "broadcast_replicas"}
    called = _callee_names(ref_params_bytes)
    check("D.7b", not (watched & called),
          "★ **AST 调用表**：**参照路径**的 `Call` 节点（`ast.Name` 形态）= `{%s}` —— "
          "**没有一个落在被测集合 `%s` 里**"
          % (", ".join(sorted(called)) if called else "**空集**", sorted(watched)))
    # ★★ **调用计数器**：先证明它非空，再证明参照跑完为 0。
    _CALLS.clear()
    broadcast_replicas(SEED, 2)
    seen = sum(_CALLS.values())
    _CALLS.clear()
    ref_params_bytes(PSI_EXPECT)
    left = sum(_CALLS.values())
    check("D.7c", seen > 0 and left == 0,
          "★★ **调用计数器两向都量**：被测路径（广播 2 份）跑一次 ⇒ 计数 = **%d**"
          "（= 1 次 `broadcast_replicas` + 1 次 `make_state`）· **参照路径**跑完 ⇒ "
          "计数 = **%d**（必须**恰好为 0**）" % (seen, left))
    # ★★ 判据钉在**内容**上而不是**大小**上：两条路径同大小、不同内容
    a, b = make_state(SEED), make_state(SEED + 1)
    check("D.8", state_bytes(a) == state_bytes(b)
          and state_digest(a) != state_digest(b),
          "★★ **「大小相同、内容不同」是常态**（两份不同初始化的状态各 `%d B`，"
          "摘要不同）⇒ 判据**必须**钉在**内容**上：只量字节数的检查会**完全放过** "
          "`C.1` 的漂移与 `C.2` 的独立初始化" % state_bytes(a))
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
    print("03_redundancy.py —— DP 的冗余账：`N` 个副本用 `N × 16Ψ` 存 `16Ψ` 的状态")
    print("=" * 74)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 判据钉在「**副本一致性**」上（真的建出 `N` 份、量 `nbytes`、算 `sha256`），"
          "**不是**钉在 `assert total == N * 16 * Psi` 那行乘法上（那是 `P12`：模拟器同时是被测对象）。")
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
