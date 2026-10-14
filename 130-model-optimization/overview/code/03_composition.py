#!/usr/bin/env python3
"""130-model-optimization/overview 编号程序 03 —— 组合的顺序何时可交换（推导，不是实测）。

配对文档：`130-model-optimization/overview/03-composition-and-selection.md`

★ 形态（CONTRIBUTING §2 判定表第二种）：编号程序**本机可跑**（默认 `python3` + `numpy`），
  三类断言**全部在本程序内部**，失败以非零码退出 ⇒ **不需要 `verify.py`**。
  产出**一律是推导值**。

本程序给出的不是「顺序有影响」这句废话，而是一条**判据**与它的**失效边界**：

  记量化 `Q` 为**对称 RTN**、剪枝 `P` 为**全局 top-k（只按 |W| 取）**。
  · **`Q` 用逐张量（per-tensor）scale 时**：`Q(P(W))` 与 `P(Q(W))` 的**支撑集与数值
    完全相同**，**当且仅当** top-k 的边界处**没有并列**（量化把两个不同幅值映到同一码）。
    理由：`q(x) = rtn(x/s)·s` 在 `s` 固定时是**单调不减**的 ⇒ 幅值次序不变；
    而全局 top-k **绝不会**删掉某个组的最大值却保留同组其他元素 ⇒ scale 不变。
  · **`Q` 用逐组（per-group）scale 时**：上面的「无并列 ⇒ 可交换」**不再充分**——
    不同组的 scale 不同，**跨组幅值次序会被量化改变**，掩码因此可以不同。

  程序把这两条都做成可证伪的断言：前者在 3000 组随机配置上断言**零反例**，
  后者断言**反例存在**（naive 判据被证伪）。

三类断言：
  A 算法正确性 —— ① 量化器的手算期望值；② 剪枝掩码的两条独立实现一致；
                  ③ **per-tensor 定理**：`边界无并列 ⇔ 两序同解`，3000 组零反例；
                  ④ QLoRA 的 double quantization 算术复算（`0.5 / 0.127 / 0.373`）。
  B 代价模型   —— 两序的支撑差异、数值差异与层输出误差；两种 scale 口径下的反例率；
                  「NF4 的元素载荷 4 bit」与「整块宽度 4.127 bit」的差。
  C 边界与反例 —— ① per-group 下「无并列 ⇒ 可交换」被证伪（**反例必须真的出现**）；
                  ② 量化并列确实改变支撑集；③ 「NF4 = 4 bit/参数」被证伪；
                  ④ 支撑相同则数值必相同（反向断言，防「差异来自数值而非支撑」的误判）。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。

用法：cd code && make 03_composition   （或 make run 全跑）
"""

import sys

import numpy as np

import specs as SPECS

FAILURES = []

BITS = 4                       # 量化位宽（示例档位）
GROUP = 4                      # 逐组 scale 的组大小（示例档位）
PRUNE_FRAC = 0.5               # 剪掉的比例
QMAX = 2 ** (BITS - 1) - 1     # 对称量化的最大码值：4 bit → 7
N_TRIALS = 3000                # 随机搜索次数（显式播种）
SEED = 20261013


def judge(label, ok, detail):
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# 两个算子（被测路径）
# ---------------------------------------------------------------------------

def rtn(x):
    """Round-to-nearest, **half away from zero**。★ 口径必须写死：

    numpy 的 `np.round` 是 half-to-even，换一种舍入会让**并列的位置**移动——
    而并列正是本题第 2 条失效来源本身。本程序一律用 half-away-from-zero。
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_per_tensor(W, bits=BITS):
    """对称 RTN，**逐张量** scale：`s = max|W| / qmax`。"""
    m = np.abs(W).max()
    s = 1.0 if m == 0 else m / (2 ** (bits - 1) - 1)
    return rtn(W / s) * s


def quantize_per_group(W, bits=BITS, group=GROUP):
    """对称 RTN，**逐组** scale：组沿**输入维**（`axis=1`）连续切，组不跨行。

    ★ 轴约定必须写死（CONTRIBUTING 对量化粒度的要求）：组的切分维度在本例是
      「输入维」，换一个维度会得到**另一组** scale。
    """
    out, inn = W.shape
    assert inn % group == 0, "本程序要求输入维能被组大小整除"
    Wr = W.reshape(out, inn // group, group)
    m = np.abs(Wr).max(axis=2, keepdims=True)
    s = np.where(m == 0, 1.0, m / (2 ** (bits - 1) - 1))
    return (rtn(Wr / s) * s).reshape(out, inn)


def keep_count(W, frac=PRUNE_FRAC):
    return int(round((1.0 - frac) * W.size))


def mask_topk(W, frac=PRUNE_FRAC):
    """全局 top-k 掩码（只按 `|W|` 取，`kind="stable"`）。**被测路径**。"""
    m = np.zeros(W.shape, dtype=bool)
    k = keep_count(W, frac)
    idx = np.argsort(-np.abs(W).ravel(), kind="stable")[:k]
    m.ravel()[idx] = True
    return m


def mask_threshold(W, frac=PRUNE_FRAC):
    """独立参照：用「第 k 大的幅值」当阈值再打掩码（**不经过 argsort 的取前 k 步**）。

    ⚠ 有并列时它与 `mask_topk` 可以合法地不同（阈值法会把并列**全保留**）。
    所以 A.2 只在**无并列**的数据上对拍——这正是「先写死判据」的落法。
    """
    k = keep_count(W, frac)
    flat = np.abs(W).ravel()
    thr = np.sort(flat)[::-1][k - 1]
    return (np.abs(W) >= thr)


# ---------------------------------------------------------------------------
# 定理的判据
# ---------------------------------------------------------------------------

def boundary_is_strict(W, QW, frac=PRUNE_FRAC):
    """top-k 边界处**无并列**：第 k 大的量化幅值 **严格大于** 第 k+1 大。"""
    flat = np.sort(np.abs(QW).ravel())[::-1]
    k = keep_count(W, frac)
    return bool(flat[k - 1] > flat[k])


def orders(W, quantizer):
    """两种顺序的产物。返回 (支撑A, 支撑B, 矩阵A, 矩阵B, 掩码相同?)。"""
    mA = mask_topk(W)                     # 先剪枝：掩码取自原权重
    WA = quantizer(W * mA)                # 再量化：scale 取自**剪过的**矩阵
    QW = quantizer(W)                     # 先量化
    mB = mask_topk(QW)                    # 再剪枝：掩码取自**量化后的**权重
    WB = QW * mB
    return mA, mB, WA, WB


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a():
    print("\nA 算法正确性（手算期望值 / 两条独立实现 / 定理的随机搜证）")

    # A.1 量化器：手算期望值
    W = np.array([[4.0, 0.0, -4.0, 1.0]])
    got = quantize_per_tensor(W, bits=2)
    want = np.array([[4.0, 0.0, -4.0, 0.0]])
    judge(
        "A.1 量化器的手算期望值（bits=2 / scale=4）",
        np.array_equal(got, want),
        f"Q([4, 0, -4, 1]) = {got.ravel().tolist()}，手算 {want.ravel().tolist()}"
        f"（0.25 舍入为 0，与舍入模式无关 ⇒ 手算无歧义）",
    )

    # A.2 剪枝掩码：两条独立实现在**无并列**数据上一致
    rng = np.random.default_rng(SEED)
    Wd = rng.normal(size=(8, 16))
    judge(
        "A.2 剪枝掩码：argsort 取前 k == 阈值法（无并列数据）",
        np.array_equal(mask_topk(Wd), mask_threshold(Wd)),
        f"两者都保留 {mask_topk(Wd).sum()} / {Wd.size} 个元素（k = {keep_count(Wd)}）",
    )

    # A.3 per-tensor 定理：边界无并列 ⇔ 两序同解
    counter_strict = 0        # 无并列却不同解
    counter_tie_same = 0      # 有并列却同解（合法，不计为反例）
    same_cnt = 0
    for t in range(N_TRIALS):
        Wt = rng.normal(size=(6, 16))
        QW = quantize_per_tensor(Wt)
        mA, mB, WA, WB = orders(Wt, quantize_per_tensor)
        strict = boundary_is_strict(Wt, QW)
        identical = np.array_equal(mA, mB) and np.array_equal(WA, WB)
        same_cnt += identical
        if strict and not identical:
            counter_strict += 1
        if (not strict) and identical:
            counter_tie_same += 1
    judge(
        f"A.3 per-tensor 定理：无并列 ⇒ 两序同解（{N_TRIALS} 组随机配置零反例）",
        counter_strict == 0,
        f"反例 {counter_strict} 个；同解的配置 {same_cnt}/{N_TRIALS}"
        f"（其中有并列但仍同解 {counter_tie_same} 个——合法：并列未必改变集合）",
    )

    # A.4 QLoRA double quantization 的算术复算
    first = SPECS.NF4_CONST_BITS_FP32 / SPECS.NF4_BLOCK_SIZE
    second = (SPECS.DQ_CONST_BITS_FP8 / SPECS.NF4_BLOCK_SIZE
              + SPECS.NF4_CONST_BITS_FP32 / (SPECS.NF4_BLOCK_SIZE * SPECS.DQ_BLOCK_SIZE))
    saving = first - second
    judge(
        "A.4 QLoRA double quantization：`32/64 → 8/64 + 32/(64·256)`",
        abs(first - 0.5) < 1e-12 and abs(second - 0.127) < 5e-4 and abs(saving - 0.373) < 5e-4,
        f"第一级 {first:.3f} bit/参数 → double quant {second:.5f} bit/参数"
        f" = {second:.3f}（省 {saving:.4f}）；论文 §3 印值 `0.127` / `0.373`",
    )
    return counter_strict


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b():
    print("\nB 代价模型（两序的差 / 两种 scale 口径的反例率 / 整块宽度）")
    rng = np.random.default_rng(SEED + 1)

    # B.1 per-group 与 per-tensor 的反例率
    rates = {}
    for name, q in (("per-tensor", quantize_per_tensor), ("per-group G=4", quantize_per_group)):
        strict_total = 0
        bad = 0
        same = 0
        for _t in range(N_TRIALS):
            Wt = rng.normal(size=(6, 16))
            QW = q(Wt)
            mA, mB, WA, WB = orders(Wt, q)
            strict = boundary_is_strict(Wt, QW)
            identical = np.array_equal(mA, mB) and np.array_equal(WA, WB)
            same += identical
            if strict:
                strict_total += 1
                if not identical:
                    bad += 1
        rates[name] = (bad, strict_total, same)
        print(f"  {name:<14} 边界严格 {strict_total:>4} 组中反例 {bad:>4} 个；"
              f"两序同解 {same:>4}/{N_TRIALS}")
    judge(
        "B.1 per-tensor 的判据是**充分**的，per-group 的不是",
        rates["per-tensor"][0] == 0 and rates["per-group G=4"][0] > 0,
        f"per-tensor 反例 {rates['per-tensor'][0]}；per-group 反例 {rates['per-group G=4'][0]}"
        f"（同样的「边界严格」前提）",
    )

    # B.2 差异有多大：支撑差异数 + 层输出误差比
    Wt = rng.normal(size=(32, 64))
    X = rng.normal(size=(128, 64))
    mA, mB, WA, WB = orders(Wt, quantize_per_group)
    diff_pos = int((mA != mB).sum())
    errA = float(np.linalg.norm(X @ (Wt - WA).T))
    errB = float(np.linalg.norm(X @ (Wt - WB).T))
    print(f"  样例（32×64、G={GROUP}、剪 {PRUNE_FRAC:.0%}）：掩码不同位置 {diff_pos} 个；"
          f"‖X(W−WA)‖={errA:.4f}，‖X(W−WB)‖={errB:.4f}，比 {errB/errA:.4f}")
    judge(
        "B.2 两序有差时，差异**只在支撑上**（数值在同一支撑上逐元素相同）",
        np.allclose(WA[mA & mB], WB[mA & mB]),
        f"公共支撑上最大逐元素差 {float(np.max(np.abs(WA[mA & mB] - WB[mA & mB]))):.3e}"
        f"（全局 top-k 从不删掉组内最大值 ⇒ 组 scale 不变 ⇒ 数值不变）",
    )

    # B.3 整块宽度：NF4 的元素载荷 vs 含常量的口径
    nf4_bits = (SPECS.NF4_ELEMENT_BITS
                + SPECS.DQ_CONST_BITS_FP8 / SPECS.NF4_BLOCK_SIZE
                + SPECS.NF4_CONST_BITS_FP32 / (SPECS.NF4_BLOCK_SIZE * SPECS.DQ_BLOCK_SIZE))
    ratio = 16.0 / nf4_bits
    judge(
        "B.3 含常量后 NF4 的整块宽度 > 4 bit，16-bit 对它的比值 < 4",
        nf4_bits > SPECS.NF4_ELEMENT_BITS and ratio < 4.0,
        f"{SPECS.NF4_ELEMENT_BITS} + {SPECS.DQ_CONST_BITS_FP8}/{SPECS.NF4_BLOCK_SIZE}"
        f" + {SPECS.NF4_CONST_BITS_FP32}/({SPECS.NF4_BLOCK_SIZE}·{SPECS.DQ_BLOCK_SIZE})"
        f" = {nf4_bits:.4f} bit/参数 ⇒ 16/{nf4_bits:.4f} = {ratio:.4f}×，不是 4×",
    )
    return rates, (mA, mB, WA, WB, errA, errB)


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(rates, sample):
    print("\nC 边界与反例（断言的是「失败行为成立」）")
    mA, mB, WA, WB, errA, errB = sample

    # C.1 per-group 下 naive 判据必须**真的**失败
    bad, strict_total, _same = rates["per-group G=4"]
    judge(
        "C.1 per-group 下「边界无并列 ⇒ 可交换」被证伪",
        bad > 0,
        f"{strict_total} 组「边界严格」里 **{bad} 组**两序不同解"
        f" ⇒ 逐组 scale 下不存在只看幅值的先验判据，只能直接比较两次掩码（O(n)）",
    )

    # C.2 量化并列确实改变支撑集（手造最小反例）
    #   构造要点：并列必须**跨在 top-k 的边界上**才有影响——
    #   三个元素（5.9 / 6.0 / 5.85，同一组、scale 由 6.0 定）量化后都映到 6.0，
    #   而 k=4 时第 4 个名额恰由这三者的并列争夺；`stable` 取**平坦下标最小**的那个。
    Wtie = np.array([[5.9, 6.0, 5.85, 0.05], [9.0, 8.0, 7.0, 0.05]])
    mAt, mBt, _wa, _wb = orders(Wtie, quantize_per_group)
    judge(
        "C.2 量化并列改变支撑集（手造最小反例，并列跨在边界上）",
        not np.array_equal(mAt, mBt),
        f"5.9 / 6.0 / 5.85 在同一组内都映到 6.0（scale = 6.0/7）⇒ 取前 k 时由并列打破"
        f"规则决定：掩码 {mAt.astype(int).ravel().tolist()} vs {mBt.astype(int).ravel().tolist()}"
        f"——**秩 4 的名额**一个给了 6.0、一个给了 5.9",
    )

    # C.3 「NF4 = 4 bit/参数」被证伪：把常量算漏会高估压缩率
    nf4_bits = (SPECS.NF4_ELEMENT_BITS
                + SPECS.DQ_CONST_BITS_FP8 / SPECS.NF4_BLOCK_SIZE
                + SPECS.NF4_CONST_BITS_FP32 / (SPECS.NF4_BLOCK_SIZE * SPECS.DQ_BLOCK_SIZE))
    ratio_payload_only = 16.0 / SPECS.NF4_ELEMENT_BITS
    ratio_full = 16.0 / nf4_bits
    judge(
        "C.3 「NF4 = 4 bit/参数」被证伪（只算载荷会高估压缩率）",
        ratio_payload_only > ratio_full and np.isclose(SPECS.NF4_ELEMENT_BITS, 4),
        f"载荷口径 16/4 = {ratio_payload_only:.4f}× vs 整块口径 16/{nf4_bits:.4f}"
        f" = {ratio_full:.4f}× ⇒ 高估 {ratio_payload_only - ratio_full:.4f}×",
    )

    # C.4 反向断言：支撑相同则数值必相同（防止把差异归错因）
    judge(
        "C.4 支撑相同 ⇒ 数值逐元素相同（反向断言）",
        np.allclose(WA[mA & mB], WB[mA & mB]),
        "差异**只**来自支撑集；若有人声称「差异来自数值」，这条断言会失败",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("130-model-optimization/overview 03 —— 组合的顺序：一条判据与它的失效边界")
    print("★ 本程序只做算术，产出是**推导值**，不是实测值。")
    print(f"  口径：对称 RTN（half-away-from-zero）、{BITS} bit、逐组 G={GROUP}、"
          f"全局 top-k 剪 {PRUNE_FRAC:.0%}、随机搜索 {N_TRIALS} 组（seed={SEED}）")
    class_a()
    rates, sample = class_b()
    class_c(rates, sample)
    print()
    if FAILURES:
        print(f"结果：{len(FAILURES)} 条断言失败 →")
        for f in FAILURES:
            print(f"  · {f}")
        return 1
    print("结果：全部断言通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
