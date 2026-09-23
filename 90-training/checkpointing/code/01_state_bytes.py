#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_state_bytes.py —— 训练状态的**组成**与它的**字节账**（参数 / 梯度 / 优化器状态 / RNG）。

配套文档：`01-state-and-bytes.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★★ **本主题最要紧的一条口径区分（`C.3` 把它量出来）**：
  ZeRO 的 `2Ψ + 2Ψ + KΨ = 16Ψ` 是「**训练时的显存**」口径（原文用的是
  `memory requirement`，Figure 1 说的是 `per-device memory consumption of model states`），
  **不是「检查点文件字节」** ⇒ **两笔账差的那一项正是【梯度】**（`2Ψ`），
  而梯度**通常不入存档**（下一步就被覆盖）。
  ⇒ 本程序把**两笔账分开算、并把差量出来**；`02_sharded.py` 才算**文件**的账。

★★ **判据 ①**（计划 §3 Chunk 5 表 A）：**优化器状态的字节账按精度算** ——
  fp32 的 `m` + `v` = `(4 + 4)Ψ = 8Ψ` 字节。★ 值**取自 `specs.py` 的常量**，
  与**实测的数组 `nbytes`** 对拍（两条**来源不同**的路径 ⇒ 判据不是恒真）。
  ★ `16Ψ` 必须注明是 **ZeRO 的举例口径**（计划 **Z4**，`specs.ZERO_K_CALIBER_NOTE`）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  实测 `m.nbytes + v.nbytes == (ZERO_MOMENTUM_MULTIPLE + ZERO_VARIANCE_MULTIPLE) × Ψ`（= `8Ψ`）；
  · `A.2`  同一批实测对上 `K = 12` 的**三项分解**（`4 + 4 + 4`），且 `K == ZERO_K_OPT_STATE`；
  · `A.3`  `2Ψ + 2Ψ + KΨ == 16Ψ`（ZeRO §5.1 的算术**复现**，逐字依据在 `specs.py`）；
  · `A.4`  RNG 的**内部状态**往返之后，后续 1000 次抽样**逐位**一致；★ 且**反面对照**：
            一个**新的**发生器给出**不同**的流（否则这条判据什么也没检查）；
  · `A.5`  PaLM 的两句逐字（判据 ④ 的**外部依据**，不是本库自造的目标）；
  · `B.1`  ★ **张量部分的字节数与步数无关**（第 1 步与第 200 步**逐位相等**），而内容不同；
  · `B.2`  ★ RNG 状态的序列化字节数**有上界**（跨 400 个状态实测 `[165, 176]`）；
  · `B.3`  ★ 参数字节**正比于 `Ψ`**（两个规模差 ~60 倍），而 RNG 字节**落在同一个带里**；
  · `B.4`  ★ **非张量部分**在玩具规模上是**可观的占比**、在真实规模上可以忽略 ⇒ 口径提醒；
  · `C.1`  ★★ PCG64 的 `has_uint32`：**缺键 → `KeyError`（有守卫）**；
            **归一化成 `0` → 静默不一致**，且**只在 32 位抽样路径上**（64 位路径仍一致）；
  · `C.2`  ★★ 把 `m`/`v` 按 **fp16** 存 ⇒ `4Ψ`（**少一半**）而**不报错** ⇒ 精度选择是载荷性的；
  · `C.3`  ★★★ **两笔账的差 == 梯度那一项**：`16Ψ − 14Ψ == ZERO_GRAD_FP16_MULTIPLE × Ψ`；
  · `C.4`  ★ `Ψ` 的口径 = **参数个数**（不是张量数、不是字节数）—— 三个数两两不同。

用法：
    python3 01_state_bytes.py
"""

import ast
import collections
import inspect
import json
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（checkpointing）取 `;; FAIL: checkpointing`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ";; FAIL: checkpointing"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", ":: FAIL: mixed-precision",
                     "++ FAIL: optimizers", "// FAIL: init-stability")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


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
SEED = 20260953              # ★ 与同域另外两个程序（…0954 / …0955）都**不同**
# ★ 两个玩具模型：`(n_in, n_hidden, n_out)` ⇒ `Ψ = n_in·n_hidden + n_hidden + n_hidden·n_out + n_out`
SMALL_SHAPE = (16, 8, 4)     # Ψ = 128 + 8 + 32 + 4 = 172
LARGE_SHAPE = (128, 64, 32)  # Ψ = 8192 + 64 + 2048 + 32 = 10336（60.1 倍）
STEPS_LATE = 200             # `B.1`：第二个存档点在**第 200 步**（不是第 1 步）
LR = 0.05                    # 玩具目标函数的步长（本程序不需要它正确，只需要它**确定**）
RNG_STATE_DRAWS = 1000       # `A.4`：往返之后要比多少个抽样
RNG_STATE_SWEEP = 400        # `B.2`：扫多少个 RNG 状态来找字节数的上下界
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   （`1e300 + 1.0 == 1e300`）⇒ 那样的叶子变异后**逐位不变**、任何断言都抓不到。
#   本程序的浮点常量都 `< 2^53`（最大的阈值是 `200.0`）⇒ 每一个叶子都**可变异**。
TOL_EXACT = 0.0              # 逐位相等
TOL_IDENT = 1e-15            # 「同一个量」的机器精度档
RNG_STATE_BYTES_MIN = 160    # `B.2`：RNG 状态序列化字节数的下界（**本机实测 165**）
RNG_STATE_BYTES_MAX = 200    # `B.2`：上界（**本机实测 176**）
SMALL_RATIO_MIN = 8.0        # `B.3`：小模型上「参数字节 / RNG 字节」的下界（实测 11.7）
LARGE_RATIO_MIN = 500.0      # `B.3`：大模型上的下界（实测 704.7）
TENSOR_FRACTION_MIN = 0.99   # `B.4`：大模型上张量部分的占比下界（实测 0.99858）
SMALL_FRACTION_MAX = 0.95    # `B.4`：小模型上张量部分的占比**上界**（实测 0.92143）
FP16_MULTIPLE = 2.0          # `C.2`：fp16 的**字节/参数**（同值：`specs.ZERO_PARAM_FP16_MULTIPLE`）
UINT32_BOUND = 4294967296    # `C.1`：32 位路径的上界（`2^32`，★ 写成十进制字面量：盲区④）
FLOAT_EXACT_LIMIT = 9007199254740992   # `D.4b`：`2^53`（★ 同上，十进制字面量）


# ===========================================================================
# 2. 路径（被测）：状态的构造、字节账、RNG 状态的往返
# ===========================================================================
def build_state(shape, seed, steps):
    """**被测**：按 `shape` 建一个玩具模型的状态，并跑 `steps` 步 Adam。

    返回 `(tensors, rng, step)`：
      · `tensors`：`{"w": params, "m": …, "v": …}`（**全部 fp32**，形状由 `shape` 定）；
      · `rng`：这一步的 `np.random.Generator`（**它的内部状态要被存档**）；
      · `step`：步号（**Adam 的偏置校正要用它**）。

    ★ 目标函数是**逐元素**的（`g = w − w_target`），**不用矩阵乘** ——
      本程序量的是**字节**，不是数值；逐元素使 `Ψ` 大 60 倍时它仍然快到可以忽略。
    """
    n_in, n_hid, n_out = shape
    rng = np.random.default_rng(seed)
    tensor_shapes = ((n_in, n_hid), (n_hid,), (n_hid, n_out), (n_out,))
    w_parts = [rng.normal(size=s).astype(np.float32) for s in tensor_shapes]
    target = [g + 0.5 for g in w_parts]
    m = [np.zeros_like(g) for g in w_parts]
    v = [np.zeros_like(g) for g in w_parts]
    for t in range(1, int(steps) + 1):
        for i, g in enumerate(w_parts):
            grad = g - target[i]
            m[i] = specs.ADAM_DEFAULT_BETA1 * m[i] + (1 - specs.ADAM_DEFAULT_BETA1) * grad
            v[i] = specs.ADAM_DEFAULT_BETA2 * v[i] + (1 - specs.ADAM_DEFAULT_BETA2) * grad * grad
            mh = m[i] / (1 - specs.ADAM_DEFAULT_BETA1 ** t)
            vh = v[i] / (1 - specs.ADAM_DEFAULT_BETA2 ** t)
            w_parts[i] = g - LR * mh / (np.sqrt(vh) + specs.ADAM_DEFAULT_EPS)
    tensors = {
        "w": np.concatenate([g.reshape(-1) for g in w_parts]),
        "m": np.concatenate([g.reshape(-1) for g in m]),
        "v": np.concatenate([g.reshape(-1) for g in v]),
    }
    return tensors, rng, int(steps)


def tensor_bytes(tensors):
    """**被测路径 A**：`Σ` 每个张量的 `nbytes`（由**实际的 numpy 数组**算出来）。

    ★ `nbytes = size × itemsize`：`size` 来自**形状**、`itemsize` 来自**dtype** ⇒
      它与 `specs.py` 里那条「每参数 4 字节」的**来源不同**（一条来自 NumPy 的 dtype 表，
      一条来自论文的字节账）⇒ 两者相等是一条**有内容的**断言（`A.1`）。
    """
    return int(sum(int(a.nbytes) for a in tensors.values()))


def state_multiple_bytes(psi, multiple):
    """**参照路径 B**：`multiple × Ψ` —— 全部输入都是 `specs.py` 的常量与参数个数。

    ★★ 这是判据 ① 要求的**另一条路径**：**不碰任何数组**，只做算术。
    """
    if psi <= 0:
        raise ValueError("Ψ 必须为正（Ψ = %r）" % (psi,))
    if multiple <= 0.0:
        raise ValueError("每参数字节数必须为正（multiple = %r）" % (multiple,))
    return int(multiple * psi)


def rng_state_bytes(rng):
    """RNG 内部状态的**序列化字节数**（本程序的口径：`json.dumps(..., sort_keys=True)`）。

    ★ 口径写在函数里：**这是本程序选的序列化方式**，不是 NumPy 的规格
      （NumPy 没有为 `bit_generator.state` 这个字典给出 schema ⇒ 已登记 `UNSOURCED`：
       `numpy_pcg64_state_json_schema`）。
    """
    return len(json.dumps(rng.bit_generator.state, sort_keys=True).encode("utf-8"))


def roundtrip_rng(rng):
    """**被测**：把 RNG 的**内部状态**序列化再从字节读回来（**不调用任何 `seed()`**）。"""
    blob = json.dumps(rng.bit_generator.state, sort_keys=True).encode("utf-8")
    fresh = np.random.default_rng(0)
    fresh.bit_generator.state = json.loads(blob.decode("utf-8"))
    return blob, fresh


def ref_bytes(tensors):
    """**独立参照**：把全部张量拼成一个**连续的一维数组**再取 `nbytes`。

    ★ 它与 `tensor_bytes` 是**两条不同的算式**（一条逐张量累加、一条拼接后再量），
      数值必须**逐位**一致 ⇒ 这条一致性同时说明「张量之间没有隐藏的对齐填充」。
    """
    flat = np.concatenate([a.reshape(-1) for a in tensors.values()])
    return int(flat.nbytes)


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


def psi_of(shape):
    """`Ψ` = **参数个数**（`specs.ZERO_FIGURE_MODEL_QUOTE` 逐字：`number of parameters`）。"""
    n_in, n_hid, n_out = shape
    return n_in * n_hid + n_hid + n_hid * n_out + n_out


def no_framework_imported():
    """★ 运行期的同义确认：`sys.modules` 里**第三方**名字中只允许有 `numpy`。

    ★ `make lint` 用 **AST** 在**静态**上做同一件事（注释里提到框架名不算违规）；
      这一条是它在**运行期**的影子 —— 两条都留着，因为 lint 是 `make` 目标、不随程序走。
    """
    banned = {"torch", "tensorflow", "jax", "triton", "numba", "cupy", "apex",
              "deepspeed", "bitsandbytes", "transformers", "flax", "vllm"}
    return not (banned & set(sys.modules))


# ★ `D.3e` 的对齐检查放在模块级（`preflight()` 在任何计算之前读它）。
ZERO_ALIGNED_OK = (
    specs.ZERO_MASTER_WEIGHT_MULTIPLE + specs.ZERO_MOMENTUM_MULTIPLE
    + specs.ZERO_VARIANCE_MULTIPLE == float(specs.ZERO_K_OPT_STATE)
    and specs.ZERO_PARAM_FP16_MULTIPLE + specs.ZERO_GRAD_FP16_MULTIPLE
    + float(specs.ZERO_K_OPT_STATE) == specs.ZERO_TOTAL_MULTIPLE
    and specs.ZERO_MOMENTUM_MULTIPLE == specs.ZERO_VARIANCE_MULTIPLE == 4.0
)


# ===========================================================================
# 4. A 类：字节账（判据 ①）与 RNG 状态的往返
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 字节账：优化器状态按**精度**算（判据 ①）· RNG 状态的往返")
    print("=" * 74)
    tensors, rng, _ = build_state(SMALL_SHAPE, SEED, 1)
    psi = psi_of(SMALL_SHAPE)
    print("  玩具模型 `%s` ⇒ **Ψ = %d 个参数**；全部张量是 **fp32**；"
          "Adam 超参从 `specs.py` 读（[C] 1412.6980，**已确认 ICLR 2015**）"
          % (SMALL_SHAPE, psi))
    print("  ★★ 判据 ①：「优化器状态的字节账**按精度算**」—— 值取自 `specs.py` 的常量"
          "（**不是**本程序自己分配的长度，那样是恒真）")

    # ── A.1 ★★ fp32 的 m + v = 8Ψ（两条来源不同的路径对拍）─────────────────
    m_b = int(tensors["m"].nbytes)
    v_b = int(tensors["v"].nbytes)
    mv_measured = m_b + v_b
    mv_closed = state_multiple_bytes(psi, specs.ZERO_MOMENTUM_MULTIPLE) \
        + state_multiple_bytes(psi, specs.ZERO_VARIANCE_MULTIPLE)
    print("  ── A.1 ★★ fp32 的 `m` + `v`：**实测**（`Σ nbytes`）vs **闭式**（`4Ψ + 4Ψ`）──")
    print("     `m` = **%d B**（闭式 `4Ψ` = %d）· `v` = **%d B**（闭式 `4Ψ` = %d）"
          % (m_b, state_multiple_bytes(psi, specs.ZERO_MOMENTUM_MULTIPLE),
             v_b, state_multiple_bytes(psi, specs.ZERO_VARIANCE_MULTIPLE)))
    print("     `m + v` = **%d B** = **%.1fΨ**（闭式 `(4+4)Ψ` = %d）"
          % (mv_measured, mv_measured / psi, mv_closed))
    check("A.1", mv_measured == mv_closed
          and m_b == state_multiple_bytes(psi, specs.ZERO_MOMENTUM_MULTIPLE)
          and v_b == state_multiple_bytes(psi, specs.ZERO_VARIANCE_MULTIPLE)
          and mv_measured == 8 * psi
          and abs(mv_measured / psi - 8.0) < TOL_IDENT,
          "★★ **判据 ① 成立**：fp32 的 `m` + `v` = **`8Ψ` 字节**（本机 `%d B`）——"
          "★ 左边来自 NumPy 的 `size × itemsize`、右边来自 `specs.py` 的 [C] 常量"
          "（`4 + 4`）⇒ 两条**来源不同**的路径给出同一个数，**不是**恒真"
          % mv_measured)

    # ── A.2 ★★ K = 12 的三项分解 ───────────────────────────────────────────
    print()
    print("  ── A.2 ★★ `K = 12` 的**三项分解**（主权重 + `m` + `v`）──")
    k_terms = (specs.ZERO_MASTER_WEIGHT_MULTIPLE, specs.ZERO_MOMENTUM_MULTIPLE,
               specs.ZERO_VARIANCE_MULTIPLE)
    opt_measured = int(tensors["w"].nbytes) + m_b + v_b
    opt_closed = sum(state_multiple_bytes(psi, t) for t in k_terms)
    print("     三项：主权重（fp32 副本）`%.0fΨ` + `m` `%.0fΨ` + `v` `%.0fΨ` = **`%.0fΨ`**"
          % (k_terms[0], k_terms[1], k_terms[2], sum(k_terms)))
    print("     实测 `Σ nbytes(w, m, v)` = **%d B**（= `%.0fΨ`）· `specs.ZERO_K_OPT_STATE` = **%d**"
          % (opt_measured, opt_measured / psi, specs.ZERO_K_OPT_STATE))
    print("     ★★ `K = 12` 是 **%s** —— 它的成立条件是 `%s`"
          % (specs.ZERO_K_CALIBER_NOTE, specs.ZERO_FIGURE_CALIBER_QUOTE))
    check("A.2", sum(k_terms) == specs.ZERO_K_OPT_STATE and opt_measured == opt_closed
          and opt_closed == specs.ZERO_K_OPT_STATE * psi,
          "★★ **`K = 12` 复现了**：`%.0f + %.0f + %.0f = %.0f`，"
          "且**实测**的三项字节数逐位等于它（`%d B`）—— ★★ 但它是 **%s**（Z4）"
          % (k_terms[0], k_terms[1], k_terms[2], sum(k_terms), opt_measured,
             specs.ZERO_K_CALIBER_NOTE))

    # ── A.3 ★★ ZeRO 的 16Ψ 算术复现 ────────────────────────────────────────
    print()
    print("  ── A.3 ★★ ZeRO §5.1 的算术：`2Ψ + 2Ψ + KΨ = 16Ψ` ──")
    total_closed = state_multiple_bytes(psi, specs.ZERO_PARAM_FP16_MULTIPLE) \
        + state_multiple_bytes(psi, specs.ZERO_GRAD_FP16_MULTIPLE) \
        + state_multiple_bytes(psi, specs.ZERO_K_OPT_STATE)
    print("     逐字依据（`specs.py`）：`%s`；`%s` … `%s`"
          % (specs.ZERO_K12_QUOTE, specs.ZERO_TOTAL_QUOTE, specs.ZERO_TOTAL_TAIL_QUOTE))
    print("     `2Ψ + 2Ψ + 12Ψ` = **%d B** = `%.0fΨ`（`specs.ZERO_TOTAL_MULTIPLE` = `%.0f`）"
          % (total_closed, total_closed / psi, specs.ZERO_TOTAL_MULTIPLE))
    check("A.3", total_closed == specs.ZERO_TOTAL_MULTIPLE * psi
          and total_closed == 16 * psi
          and specs.ZERO_PARAM_FP16_MULTIPLE + specs.ZERO_GRAD_FP16_MULTIPLE
          + float(specs.ZERO_K_OPT_STATE) == specs.ZERO_TOTAL_MULTIPLE,
          "★★ **算术复现**：`%.0fΨ + %.0fΨ + %dΨ = %.0fΨ`（本机 `%d B`）—— "
          "★★★ 但原文用的词是 `%s` ⇒ **这是「训练显存」的口径，不是「检查点文件」的口径**"
          "（`C.3` 量两者的差）"
          % (specs.ZERO_PARAM_FP16_MULTIPLE, specs.ZERO_GRAD_FP16_MULTIPLE,
             specs.ZERO_K_OPT_STATE, specs.ZERO_TOTAL_MULTIPLE, total_closed,
             specs.ZERO_TOTAL_TAIL_QUOTE))

    # ── A.4 ★★ RNG 内部状态的往返 ─────────────────────────────────────────
    print()
    print("  ── A.4 ★★ RNG 的**内部状态**往返（**不调用任何 `seed()`**）──")
    blob, fresh = roundtrip_rng(rng)
    ref_draws = rng.random(RNG_STATE_DRAWS).tolist()
    got_draws = fresh.random(RNG_STATE_DRAWS).tolist()
    other = np.random.default_rng(0).random(RNG_STATE_DRAWS).tolist()
    print("     状态的序列化字节 = **%d B**（`json.dumps(..., sort_keys=True)` 的口径）"
          % len(blob))
    print("     往返后 %d 次 `random()`：与原来**逐位**一致 = **%s**"
          % (RNG_STATE_DRAWS, got_draws == ref_draws))
    print("     ★ 反面对照：一个**新的**发生器给出**不同**的流 = **%s**"
          % (other != ref_draws))
    check("A.4", got_draws == ref_draws and other != ref_draws
          and len(blob) > 0 and RNG_STATE_DRAWS >= 1000,
          "★★ **「RNG 的内部状态」是可往返的**（%d 次抽样逐位一致）—— "
          "★ 而**反面对照**（一个全新的发生器）**不一致** ⇒ 这条判据**不是**恒真："
          "它真的在检查「状态被存下来了」" % RNG_STATE_DRAWS)

    # ── A.5 ★ 判据 ④ 的外部依据（PaLM 逐字）───────────────────────────────
    print()
    print("  ── A.5 ★ 判据 ④ 的**外部依据**：PaLM 逐字说自己做到了 ──")
    print("     [C] PaLM 逐字：`%s` / `%s`" % (specs.PALM_BITWISE_SECTION_QUOTE,
                                             specs.PALM_BITWISE_QUOTE))
    print("     ★ 它给的两条做法之一：`%s` … `%s`"
          % (specs.PALM_DATASET_RA_QUOTE, specs.PALM_DATASET_STEP_QUOTE))
    check("A.5", specs.PALM_BITWISE_QUOTE == "fully bitwise reproducible from any checkpoint"
          and specs.PALM_DATASET_STEP_QUOTE
          == "the contents of a given training batch is only a function of the step number",
          "★★ **判据 ④ 不是本库自造的目标**：PaLM 明写 `%s`；★ 而它给的第二条做法"
          "（数据按步号索引）正是 `03_consistency.py` 的**第三条路径** —— "
          "⇒ 那条路径**不是错的**，它只是**不检查存档**" % specs.PALM_BITWISE_QUOTE)


# ===========================================================================
# 5. B 类：代价模型 —— 字节数**与步数无关**、RNG 状态是**常数级**
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：张量字节数与**步数**无关 · RNG 状态是**常数级**（与 Ψ 无关）")
    print("=" * 74)
    early, _, _ = build_state(LARGE_SHAPE, SEED, 1)
    late, _, _ = build_state(LARGE_SHAPE, SEED, STEPS_LATE)
    psi_s, psi_l = psi_of(SMALL_SHAPE), psi_of(LARGE_SHAPE)

    # ── B.1 ★★ 张量部分的字节数与步数无关 ──────────────────────────────────
    print("  ── B.1 ★★ 张量部分：第 1 步 vs 第 %d 步 ──" % STEPS_LATE)
    b_early, b_late = tensor_bytes(early), tensor_bytes(late)
    print("     第 1 步 `Σ nbytes` = **%d B** · 第 %d 步 = **%d B**（逐位相等 = **%s**）"
          % (b_early, STEPS_LATE, b_late, b_early == b_late))
    same_content = early["w"].tobytes() == late["w"].tobytes()
    print("     ★ 而**内容不同**（`w` 的字节相同 = %s）⇒ 变的是**值**，不是**字节数**"
          % same_content)
    check("B.1", b_early == b_late and not same_content and b_early == 12 * psi_l,
          "★★ **检查点的张量字节数与步数无关**（`%d B` = `12Ψ`，`Ψ = %d`）：它是 `Ψ` 的函数，"
          "**不是**训练时长的函数 ⇒ 「存得越勤 ⇒ 单次越贵」是**错的**；"
          "贵的是**次数**（`Σ = 次数 × W`）" % (b_early, psi_l))

    # ── B.2 ★★ RNG 状态的字节数有上界 ─────────────────────────────────────
    print()
    print("  ── B.2 ★★ RNG 状态的序列化字节数：扫 %d 个状态 ──" % RNG_STATE_SWEEP)
    sizes = []
    for s in range(RNG_STATE_SWEEP):
        r = np.random.default_rng(s)
        k = s % 17
        if k:
            r.integers(0, 1000, size=k)
        sizes.append(rng_state_bytes(r))
    lo, hi = min(sizes), max(sizes)
    print("     跨度 = **[%d, %d] B**；★ 它**不随 Ψ 增长**（带子 `[RNG_STATE_BYTES_MIN, "
          "RNG_STATE_BYTES_MAX]` = `[%d, %d]`）" % (lo, hi, RNG_STATE_BYTES_MIN,
                                                  RNG_STATE_BYTES_MAX))
    check("B.2", RNG_STATE_BYTES_MIN <= lo <= hi <= RNG_STATE_BYTES_MAX
          and len(set(sizes)) > 1,
          "★★ **RNG 状态是常数级**：序列化字节 ∈ `[%d, %d]`（本机实测），"
          "★ 它**不是**定值（`set(sizes)` 有 %d 个不同值）⇒ 正文里写「约 170 字节」可以、"
          "写「恰好 175 字节」不行" % (lo, hi, len(set(sizes))))

    # ── B.3 ★ 参数字节 ∝ Ψ，RNG 字节落在同一个带里 ────────────────────────
    print()
    print("  ── B.3 ★ 两个规模上的比值（`Ψ` 差 %.1f 倍）──" % (psi_l / psi_s))
    rows = []
    for label, shape in (("SMALL", SMALL_SHAPE), ("LARGE", LARGE_SHAPE)):
        tens, rng_m, _ = build_state(shape, SEED, 1)
        tb, rb = tensor_bytes(tens), rng_state_bytes(rng_m)
        rows.append((label, psi_of(shape), tb, rb, tb / rb))
        print("     %-6s Ψ=%-7d 张量 = **%d B** · RNG = **%d B** ⇒ 比值 **%.1f**"
              % (label, psi_of(shape), tb, rb, tb / rb))
    print("     ★ 张量字节之比 = **%.2f 倍**（= `Ψ` 之比 %.2f ⇒ **正比**）；"
          "RNG 字节 = **%d B** 与 **%d B**（**落在同一个 12 字节的带里**）"
          % (rows[1][2] / rows[0][2], psi_l / psi_s, rows[0][3], rows[1][3]))
    check("B.3", rows[1][2] * psi_s == rows[0][2] * psi_l
          and max(rows[0][3], rows[1][3]) <= RNG_STATE_BYTES_MAX
          and rows[0][4] > SMALL_RATIO_MIN and rows[1][4] > LARGE_RATIO_MIN,
          "★★ **两笔账的标度完全不同**：张量**精确正比于 `Ψ`**"
          "（`%d × %d == %d × %d`，整数恒等），而 RNG 状态的两个值只差 %d 字节 —— "
          "★ 把两者混进一个「O(Ψ) 的账」里，在小模型上会**错一个数量级**"
          % (rows[1][2], psi_s, rows[0][2], psi_l, abs(rows[1][3] - rows[0][3])))

    # ── B.4 ★ 非张量部分：玩具规模上占比可观 ──────────────────────────────
    print()
    print("  ── B.4 ★ 「张量 vs 非张量」的占比随 Ψ 变（[B] DCP 逐字 `%s`）──"
          % specs.DCP_NON_TENSOR_QUOTE)
    fracs = []
    for label, p, tb, rb, _ in rows:
        fracs.append(tb / (tb + rb))
        print("     %-6s 张量占比 = **%.5f**（非张量 %d B / 总 %d B）"
              % (label, fracs[-1], rb, tb + rb))
    check("B.4", fracs[1] > TENSOR_FRACTION_MIN and fracs[0] < SMALL_FRACTION_MAX,
          "★★ **口径提醒**：DCP 逐字说非张量数据**走另一条路**（`torch.load()` 侧）——"
          "★ 而它的**占比**在玩具规模上是 **%.3f%%**、在 `Ψ = %d` 上是 **%.5f%%** ⇒ "
          "「检查点几乎全是张量」这句话**在玩具规模上不成立**，"
          "它是**关于规模的一句话**" % ((1 - fracs[0]) * 100, psi_l, (1 - fracs[1]) * 100))


# ===========================================================================
# 6. C 类：边界与反例 —— RNG 内部状态的完整性、精度选择、两笔账的差
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：`has_uint32` 的三态 · fp16 的静默减半 · **两笔账的差**")
    print("=" * 74)

    # ── C.1 ★★ PCG64 的 `has_uint32`：守卫 vs 静默 ─────────────────────────
    print("  ── C.1 ★★ RNG 内部状态是**四个字段**；`has_uint32` 有两个坏法（一个报错、一个静默）──")
    r = np.random.default_rng(1)
    r.integers(0, UINT32_BOUND, size=3)          # ★ 奇数个 32 位抽样 ⇒ 缓冲区里留了半字
    st = r.bit_generator.state
    keys = tuple(sorted(st.keys()))
    print("     四个字段 = `%s`（`has_uint32` = %d：**缓冲区里有一个半字**）"
          % (keys, st["has_uint32"]))
    # ★★ 口径：**先在同一个状态上取参照、再从这个状态出发做两种坏法**
    #   （顺序反了就会把「参照自己已经消耗掉的部分」当成差异 —— 本程序第一次写错的就是这一处）。
    ref32 = r.integers(0, UINT32_BOUND, size=8).tolist()
    missing = {"bit_generator": st["bit_generator"], "state": dict(st["state"])}
    guard = None
    g = np.random.default_rng(0)
    try:
        g.bit_generator.state = missing
    except KeyError as e:
        guard = str(e)
    normalized = json.loads(json.dumps(st))
    normalized["has_uint32"] = 0
    g2 = np.random.default_rng(0)
    g2.bit_generator.state = normalized
    silent32 = g2.integers(0, UINT32_BOUND, size=8).tolist() == ref32
    # 64 位路径：**同一份** `st`（同样带 `has_uint32 = 1`）上再取一次参照。
    r2 = np.random.default_rng(0)
    r2.bit_generator.state = json.loads(json.dumps(st))
    ref64 = r2.random(8).tolist()
    n2 = json.loads(json.dumps(st))
    n2["has_uint32"] = 0
    g3 = np.random.default_rng(0)
    g3.bit_generator.state = n2
    silent64 = g3.random(8).tolist() == ref64
    print("     (a) **缺** `has_uint32` 键 ⇒ `KeyError%s`（**有守卫**，不是静默）" % guard)
    print("     (b) `has_uint32` **归一化成 0** ⇒ 32 位路径（`integers`）一致 = **%s**"
          % silent32)
    print("     (c) 同一份被归一化的状态 ⇒ 64 位路径（`random()`）一致 = **%s**" % silent64)
    check("C.1", keys == ("bit_generator", "has_uint32", "state", "uinteger")
          and guard is not None and "has_uint32" in guard
          and not silent32 and silent64,
          "★★★ **三态全部量出来了**：`(a)` 缺键 ⇒ **报错**（NumPy 自己守卫）；"
          "`(b)` 归一化 ⇒ **静默不一致**；`(c)` ★★ 而**只在 32 位路径上**不一致 —— "
          "64 位路径（`random()`）**照样一致** ⇒ 「用 `random()` 做的恢复测试」**抓不到**这一类错；"
          "判据 ④ 的玩具循环因此必须含**真正用到的那个抽样路径**")

    # ── C.2 ★★ 精度选择是载荷性的（静默减半）───────────────────────────────
    print()
    print("  ── C.2 ★★ 把 `m`/`v` 按 **fp16** 存：`4Ψ`（少一半），**不报错**──")
    small, _, _ = build_state(SMALL_SHAPE, SEED, 1)
    psi = psi_of(SMALL_SHAPE)
    mv32 = int(small["m"].nbytes) + int(small["v"].nbytes)
    mv16 = state_multiple_bytes(psi, FP16_MULTIPLE) * 2
    print("     fp32：`m + v` = **%d B**（`8Ψ`）· fp16：`2Ψ + 2Ψ` = **%d B**（`4Ψ`）"
          % (mv32, mv16))
    print("     ⇒ 差 **%d B**（= `4Ψ`）；★ 两种都不会抛异常、两种都能跑通" % (mv32 - mv16))
    check("C.2", mv16 * 2 == mv32 and mv16 == 4 * psi and mv32 == 8 * psi,
          "★★ **精度选择是载荷性的**：同一个优化器、同一组超参，`m`/`v` 的**字节数差一倍**"
          "（`%d` vs `%d`）而**没有任何报错** ⇒ 字节账必须**写明精度**，"
          "「两个长度 Ψ 的数组」这种说法**不带载荷**" % (mv32, mv16))

    # ── C.3 ★★★ 两笔账的差 == 梯度那一项 ──────────────────────────────────
    print()
    print("  ── C.3 ★★★ 「训练显存」的账 vs 「检查点文件」的账：差**恰好**是梯度那一项 ──")
    mem_total = state_multiple_bytes(psi, specs.ZERO_TOTAL_MULTIPLE)
    # 检查点存什么由**用户的 state_dict 决定**（[B] DCP 逐字：save 会先调 state_dict）⇒
    # 本程序取「fp16 参数副本 + `KΨ` 优化器状态」这一支。
    ckpt = state_multiple_bytes(psi, specs.ZERO_K_OPT_STATE) \
        + state_multiple_bytes(psi, specs.ZERO_PARAM_FP16_MULTIPLE)
    diff = mem_total - ckpt
    grad_term = state_multiple_bytes(psi, specs.ZERO_GRAD_FP16_MULTIPLE)
    print("     [B] DCP 逐字：`%s` … `%s`"
          % (specs.DCP_NON_TENSOR_QUOTE, "save will call state_dict before serialization"))
    print("     · **训练显存**（ZeRO §5.1 的举例口径）= `%.0fΨ` = **%d B**"
          % (specs.ZERO_TOTAL_MULTIPLE, mem_total))
    print("     · **检查点**（`state_dict` 里放 `2Ψ` 参数 + `KΨ` 优化器状态）= `%.0fΨ` = **%d B**"
          % (specs.ZERO_K_OPT_STATE + specs.ZERO_PARAM_FP16_MULTIPLE, ckpt))
    print("     ⇒ 差 = **%d B** = `%.0fΨ` —— 与 `ZERO_GRAD_FP16_MULTIPLE × Ψ` = **%d B** "
          "逐位相等 = **%s**" % (diff, diff / psi, grad_term, diff == grad_term))
    check("C.3", diff == grad_term and diff > 0 and mem_total == 16 * psi
          and ckpt == 14 * psi,
          "★★★ **两笔账的差就是【梯度】那一项**（`%d B` = `2Ψ`）⇒ "
          "把 ZeRO 的 `16Ψ` 当检查点大小，**恰好**多算了一份梯度；"
          "★ 而原文从没说过检查点含梯度（已登记 `UNSOURCED`：`dcp_checkpoint_excludes_gradients`）"
          "⇒ 检查点的字节由**你的 `state_dict` 里放了什么**决定" % diff)

    # ── C.4 ★ `Ψ` 的口径 = 参数个数 ────────────────────────────────────────
    print()
    print("  ── C.4 ★ `Ψ` 的三种数法：**参数个数** / **张量个数** / 单张量的**字节数**──")
    n_tensors = len(small)
    n_params = psi
    n_bytes = int(small["w"].nbytes)
    print("     同一个模型：参数个数 = **%d** · 张量个数 = **%d** · `w` 的字节数 = **%d**"
          % (n_params, n_tensors, n_bytes))
    print("     [C] ZeRO Figure 1 逐字：`%s` ⇒ **`Ψ` = 参数个数**"
          % specs.ZERO_FIGURE_MODEL_QUOTE)
    check("C.4", len({n_params, n_tensors, n_bytes}) == 3
          and specs.ZERO_FIGURE_MODEL_QUOTE
          == "denotes model size (number of parameters), K",
          "★★ **三个数两两不同**（`%d` / `%d` / `%d`）⇒ 字节账里的 `Ψ` **必须**是"
          "**参数个数**（ZeRO 的逐字定义）；★ 用张量个数当 `Ψ` 会让账**差 %d 倍**，"
          "而它**不会报错**" % (n_params, n_tensors, n_bytes, psi // n_tensors))


# ===========================================================================
# 7. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ **禁区回链**：导出/推理格式的选择（`.keras` vs SavedModel）在 "
          "`70-frameworks/tensorflow/03-custom-ops-and-export.md:180,188`；"
          "优化器状态的分片与通信量在 `100-parallelism/README.md:14`；"
          "重算换显存（`O(√n)` 是**内存**、`≈ 2n` 是**前向次数**）在 "
          "`90-training/autograd-and-backward/02-gradient-checkpointing.md`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ";; FAIL: checkpointing" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", ":: FAIL: mixed-precision",
                                        "++ FAIL: optimizers", "// FAIL: init-stability"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")

    measured = {"SEED": SEED, "SMALL_SHAPE": SMALL_SHAPE, "LARGE_SHAPE": LARGE_SHAPE,
                "STEPS_LATE": STEPS_LATE, "LR": LR, "RNG_STATE_DRAWS": RNG_STATE_DRAWS,
                "RNG_STATE_SWEEP": RNG_STATE_SWEEP, "UINT32_BOUND": UINT32_BOUND,
                "FLOAT_EXACT_LIMIT": FLOAT_EXACT_LIMIT,
                "ADAM_DEFAULTS": (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                                  specs.ADAM_DEFAULT_EPS)}
    expect = {"SEED": 20260953, "SMALL_SHAPE": (16, 8, 4), "LARGE_SHAPE": (128, 64, 32),
              "STEPS_LATE": 200, "LR": 0.05, "RNG_STATE_DRAWS": 1000,
              "RNG_STATE_SWEEP": 400, "UINT32_BOUND": 4294967296,
              "FLOAT_EXACT_LIMIT": 9007199254740992,
              "ADAM_DEFAULTS": (0.9, 0.999, 1e-8)}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 %d 个逐位一致（结论的**测量条件**）%s"
          % (len(measured), "" if not diff else " ← 差异 %s" % diff))
    check("D.3b", psi_of(SMALL_SHAPE) == 172 and psi_of(LARGE_SHAPE) == 10336
          and psi_of(LARGE_SHAPE) // psi_of(SMALL_SHAPE) == 60,
          "★ 两个玩具模型的 `Ψ` 与它们的比例（%d / %d，比例 %d）——"
          "它们是 `B.3` 那两个比值的**取值范围**"
          % (psi_of(LARGE_SHAPE), psi_of(SMALL_SHAPE),
             psi_of(LARGE_SHAPE) // psi_of(SMALL_SHAPE)))
    check("D.3c", UINT32_BOUND == 4294967296 and RNG_STATE_DRAWS >= 1000
          and RNG_STATE_SWEEP >= 400,
          "★ `C.1` 要走到**32 位路径**（上界 `2^32` = %d）★ 它是**十进制字面量**、"
          "不是 `2 ** 32`：盲区④ 会让幂次字面量**整体不进变异分母**" % UINT32_BOUND)
    check("D.3d", (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2)
          == (0.9, 0.999),
          "★★ **Adam 的 `β` 必须先钉住**（在 `preflight()` 里）：`build_state` 用它们跑递推 ——"
          "★ 这是 **P23** 的同一条理由（口径型常量的钉住要跑在任何计算之前）")
    check("D.3e", ZERO_ALIGNED_OK,
          "★ `specs.py` 的字节账四项**互相对齐**：`4 + 4 + 4 == K`、`2 + 2 + K == 16`"
          "（在**任何计算之前**查一遍：否则后面那些算术断言会一起崩）")

    th = {"TOL_EXACT": TOL_EXACT, "TOL_IDENT": TOL_IDENT,
          "RNG_STATE_BYTES_MIN": RNG_STATE_BYTES_MIN, "RNG_STATE_BYTES_MAX": RNG_STATE_BYTES_MAX,
          "SMALL_RATIO_MIN": SMALL_RATIO_MIN, "LARGE_RATIO_MIN": LARGE_RATIO_MIN,
          "TENSOR_FRACTION_MIN": TENSOR_FRACTION_MIN, "SMALL_FRACTION_MAX": SMALL_FRACTION_MAX,
          "FP16_MULTIPLE": FP16_MULTIPLE}
    th_expect = {"TOL_EXACT": 0.0, "TOL_IDENT": 1e-15,
                 "RNG_STATE_BYTES_MIN": 160.0, "RNG_STATE_BYTES_MAX": 200.0,
                 "SMALL_RATIO_MIN": 8.0, "LARGE_RATIO_MIN": 500.0,
                 "TENSOR_FRACTION_MIN": 0.99, "SMALL_FRACTION_MAX": 0.95,
                 "FP16_MULTIPLE": 2.0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 9 个逐位一致%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
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

    check("D.5", (specs.ZERO_K_OPT_STATE, specs.ZERO_MOMENTUM_MULTIPLE,
                  specs.ZERO_VARIANCE_MULTIPLE, specs.ZERO_MASTER_WEIGHT_MULTIPLE,
                  specs.ZERO_PARAM_FP16_MULTIPLE, specs.ZERO_GRAD_FP16_MULTIPLE,
                  specs.ZERO_TOTAL_MULTIPLE) == (12, 4.0, 4.0, 4.0, 2.0, 2.0, 16.0),
          "★ ZeRO 的七项字节账常量逐位一致（[C] 1910.02054 §5.1）——"
          "★ 本程序的**判据 ①** 与 `C.3` 的全部结论都挂在这七个数上")
    check("D.6", specs.ZERO_ARXIV_ID == "1910.02054"
          and specs.ZERO_TITLE
          == "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
          "★★ **P21 的落法**：ZeRO 的**编号 + 标题**成对钉住（`%s`，**不确认同行评审** ——"
          "abs 页 `Comments:` 字段不存在）" % specs.ZERO_ARXIV_ID)
    check("D.7", specs.ZERO_K12_QUOTE == "Mixed-precision Adam has K = 12"
          and specs.ZERO_K_DEFINITION_QUOTE
          == "to denote the memory multiplier of the optimizer states"
          and specs.ZERO_ADDITIONAL_QUOTE
          == "i.e., the additional memory required to store them is K"
          and specs.ZERO_TOTAL_QUOTE == "In total, this results in 2"
          and specs.ZERO_TOTAL_TAIL_QUOTE == "bytes of memory requirement"
          and specs.DCP_NON_TENSOR_QUOTE
          == "All non-tensor data is loaded using torch.load()",
          "★★ **`K = 12` 的五条逐字**（`specs.py` 里各有 1 次 / 剥标签后 1 次）——"
          "★ 其中特别钉住 `memory requirement` 那个词：**它是「显存」口径的字面证据**")
    check("D.7b", specs.ZERO_FIGURE_CALIBER_QUOTE
          == "based on mixed-precision training with Adam optimizer."
          and specs.ZERO_K_CALIBER_NOTE == "ZeRO 的举例口径",
          "★★★ **Z4 的落法**：`K = 12` 的**口径**（mixed-precision + Adam 这一组假设）"
          "与那句中文限定语**成对存放** ⇒ 正文里说 `16Ψ` 时必须带上它")
    check("D.7c", specs.ZERO_OPT_4X_QUOTE
          == "4x memory reduction, same communication volume as DP"
          and specs.ZERO_GRAD_8X_QUOTE
          == "8x memory reduction, same communication volume as DP",
          "★★ **Y14 的落法**：`4x` 属 `P_os`、`8x` 属 `P_os+g` —— 两条**分开存放**，"
          "不得合并成「ZeRO 省 8 倍」")
    check("D.8", specs.PALM_ARXIV_ID == "2204.02311"
          and specs.PALM_TITLE == "PaLM: Scaling Language Modeling with Pathways"
          and specs.PALM_BITWISE_SECTION_QUOTE == "Bitwise determinism"
          and specs.PALM_BITWISE_QUOTE == "fully bitwise reproducible from any checkpoint"
          and specs.PALM_RUN_PAIR_QUOTE
          == "if the model has been trained up to step 17,000 in a single run"
          and specs.PALM_DATASET_RA_QUOTE
          == "a deterministic dataset pipeline where the shuffled data is written out in a random-access format"
          and specs.PALM_DATASET_STEP_QUOTE
          == "the contents of a given training batch is only a function of the step number",
          "★★ **判据 ④ 的七条引文逐位**（[C] PaLM，`%s`，**不确认同行评审**）——"
          "★ 其中 `PALM_DATASET_STEP_QUOTE` 是**第三条路径的正面依据**"
          % specs.PALM_ARXIV_ID)
    check("D.9", specs.UNSOURCED == ("dcp_per_shard_header_bytes",
                                      "dcp_metadata_file_format",
                                      "dcp_partial_load_state",
                                      "dcp_checkpoint_excludes_gradients",
                                      "palm_pipeline_materialized_bytes",
                                      "numpy_pcg64_state_json_schema",
                                      "zero_16psi_checkpoint_caliber"),
          "★★ `specs.UNSOURCED` 的 %d 项**逐位钉住** —— ★ 其中 "
          "`numpy_pcg64_state_json_schema` 就是 `C.1` 的处境（那个四字段字典**没有官方 schema**）、"
          "`dcp_checkpoint_excludes_gradients` 是 `C.3` 的处境" % len(specs.UNSOURCED))

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_a = inspect.getsource(tensor_bytes)
    src_b = inspect.getsource(state_multiple_bytes)
    src_c = inspect.getsource(ref_bytes)
    check("D.10", len({src_a, src_b, src_c}) == 3,
          "★ 防 P17：被测（`tensor_bytes` %d 字符）、参照（`state_multiple_bytes` %d）、"
          "反面对照（`ref_bytes` %d）**源码两两不同**"
          % (len(src_a), len(src_b), len(src_c)))
    watched = {"tensor_bytes", "state_multiple_bytes", "build_state"}
    called = _callee_names(state_multiple_bytes) | _callee_names(ref_bytes)
    check("D.10b", not (watched & called),
          "★ **AST 调用表**：参照路径的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    worst = 0
    for shape in (SMALL_SHAPE, LARGE_SHAPE):
        t, _, _ = build_state(shape, SEED, 1)
        worst = max(worst, abs(tensor_bytes(t) - ref_bytes(t)))
    check("D.10c", worst == TOL_EXACT,
          "★ **两条算式**（逐张量累加 vs 拼接后再量）在两个规模上**逐位一致**（差 %d）⇒ "
          "`D.10` 的「源码两两不同」不是装饰，且**张量之间没有隐藏的对齐填充**" % worst)
    check("D.11", MARKS_USED <= {ASSERT_FAIL_MARK},
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
    print("01_state_bytes.py —— 训练状态的组成与字节账（参数 / 梯度 / 优化器状态 / RNG）")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ **两笔账的口径区分**：ZeRO 的 `16Ψ` 是**训练显存**（原文 `memory requirement`），"
          "**不是检查点文件字节** —— 两者**恰好差一个梯度项 `2Ψ`**（`C.3` 断言这个恒等式）。")
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
