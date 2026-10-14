#!/usr/bin/env python3
"""130-model-optimization/overview 编号程序 02 —— 五本账与三种峰值（推导，不是实测）。

配对文档：`130-model-optimization/overview/02-compression-accounting.md`

★ 形态（CONTRIBUTING §2 判定表第二种）：编号程序**本机可跑**（默认 `python3` + `numpy`），
  三类断言**全部在本程序内部**，失败以非零码退出 ⇒ **不需要 `verify.py`**。
  产出**一律是推导值**。

本程序要证明的那句话是：**「省了多少显存」没有唯一答案**，必须先问两件事——
**哪一本账**（五本）与**哪个阶段**（训练 / prefill / decode）。因为

    · 五本账**不同时**到达峰值；
    · 六族优化各自只动其中一本或两本，**没有一族能同时动五本**；
    · 所以「把账加起来」与「取各阶段最大值」是两个不同的数，而后者才是峰值。

三类断言：
  A 算法正确性 —— ① 逐张量参数量 vs 闭式；② `16Ψ` 分解的复现（**消费** `90-training`
                  的结论，不重推）；③ 峰值 = 各阶段取 max ≠ 各账相加；
                  ④ **跨域一致性**：2:4 的「含 metadata 整块宽度」复算 == `36/64`。
  B 代价模型   —— 六族的**归因表**（每族对五本账各降多少、动到哪几本）。
  C 边界与反例 —— ① 「int4 权重 ⇒ 训练显存减半」被证伪；② 「五本账相加 == 峰值」被证伪；
                  ③ 把 `16Ψ` 当检查点大小**恰好多算一份梯度 `2Ψ`**；
                  ④ 「量化能省优化器状态」被证伪（量化那一列的差必须**恰好为 0**）。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。

用法：cd code && make 02_accounting   （或 make run 全跑）
"""

import sys

import numpy as np

import specs as SPECS

FAILURES = []

TRAIN_BATCH, TRAIN_SEQ = 8, 4096
PREFILL_BATCH, PREFILL_SEQ = 1, 4096
DECODE_BATCH, DECODE_SEQ = 1, 4096
LOW_RANK_R = 256
DISTILL_STUDENT_RATIO = 0.5
# 上面前三行是三个阶段的**示例工作负载**，后两行是两族的**示例档位**。
# **本文构造的假设**，不是任何实测的配置，也不是任何模型的规格。

ACCOUNT_NAMES = ("weights", "grads", "optimizer", "activations", "kv")
PHASES = ("train", "prefill", "decode")


def judge(label, ok, detail):
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# 参数账
# ---------------------------------------------------------------------------

def weight_matrices(cfg):
    d, d_ff = cfg["d_model"], cfg["d_ff"]
    heads, head_dim, kv_heads = cfg["n_heads"], cfg["head_dim"], cfg["n_kv_heads"]
    vocabulary = cfg["vocab"]
    per_layer = [
        ("Wq", heads * head_dim, d),
        ("Wk", kv_heads * head_dim, d),
        ("Wv", kv_heads * head_dim, d),
        ("Wo", d, heads * head_dim),
        ("W_gate", d_ff, d),
        ("W_up", d_ff, d),
        ("W_down", d, d_ff),
    ]
    return per_layer, ("embedding", vocabulary, d), ("lm_head", vocabulary, d)


def psi(cfg):
    """参数量。**闭式**路径：`L·(4d² + 3·d·d_ff) + 2·vocab·d`。"""
    d, d_ff = cfg["d_model"], cfg["d_ff"]
    return cfg["layers"] * (4 * d * d + 3 * d * d_ff) + 2 * cfg["vocab"] * d


def psi_reference(cfg):
    """独立参照：逐张量数元素（含 embedding 与 LM head）。"""
    per_layer, emb, lm = weight_matrices(cfg)
    lay = sum(int(np.prod(s, dtype=np.int64)) for _n, *s in per_layer)
    return (cfg["layers"] * lay + int(np.prod(emb[1:], dtype=np.int64))
            + int(np.prod(lm[1:], dtype=np.int64)))


def psi_low_rank(cfg, r):
    """把每层的 4 个 `d×d` 与 3 个 `d×d_ff` 换成秩 `r` 的双因子分解。"""
    d, d_ff = cfg["d_model"], cfg["d_ff"]
    per_layer = 4 * r * (d + d) + 3 * r * (d + d_ff)
    return cfg["layers"] * per_layer + 2 * cfg["vocab"] * d


def kv_bytes_per_token(bits):
    return SPECS.KV_BYTES_PER_TOKEN_FP16 * bits / 16.0


def sparse_2to4_bytes(count, elem_bits):
    """2:4 的**整块**字节：载荷（一半元素）+ metadata（4 bit / 4 元素）。

    ★ 稀疏粒度的正文在 `sparsity/01-sparsity-granularity.md`；本程序只**消费**
      「4 bit/组、一组 4 元素」这一个原子事实。
    """
    payload = (count / 2.0) * (elem_bits / 8.0)
    metadata = count * (SPECS.SPARSE_2_4_METADATA_BITS_PER_GROUP / SPECS.SPARSE_2_4_GROUP) / 8.0
    return payload + metadata


# ---------------------------------------------------------------------------
# 五本账
# ---------------------------------------------------------------------------

def accounts(cfg, phase, setting=None):
    """某个阶段上**同时活着**的账。

    `setting` 可含：
      · `w_bits`        权重的存储位宽（默认 16）
      · `kv_bits`       KV 的存储位宽（默认 16）
      · `trainable`     可训练参数占比 ρ（PEFT 用，默认 1.0）
      · `psi_override`  **结构性**改变参数量（低秩 / 蒸馏用；默认 `psi(cfg)`）
      · `weights_bytes` 直接给权重字节（稀疏化用；默认由位宽算）
    ★ 不在该阶段活着的账**不出现**（不是 0）——这正是「账不同时到顶」的来源。
    """
    st = dict(setting or {})
    P = st.get("psi_override", psi(cfg))
    trainable = st.get("trainable", 1.0)
    out = {"weights": st.get("weights_bytes", P * st.get("w_bits", 16) / 8.0)}
    if phase == "train":
        out["grads"] = P * 2.0 * trainable
        out["optimizer"] = P * SPECS.OPTIMIZER_STATE_BYTES_PER_PARAM * trainable
        out["activations"] = SPECS.ACTIVATION_BYTES_TRAIN_EXAMPLE
    elif phase == "prefill":
        out["activations"] = SPECS.ACTIVATION_BYTES_PREFILL_EXAMPLE
        out["kv"] = PREFILL_BATCH * PREFILL_SEQ * kv_bytes_per_token(st.get("kv_bits", 16))
    elif phase == "decode":
        out["kv"] = DECODE_BATCH * DECODE_SEQ * kv_bytes_per_token(st.get("kv_bits", 16))
    else:
        raise ValueError(phase)
    return out


def peak(cfg, phase, setting=None):
    """一个阶段的**峰值** = 该阶段活着的账之和。"""
    return sum(accounts(cfg, phase, setting).values())


def sum_of_all_accounts(cfg, setting=None):
    """「把五本账各自的最大值加起来」——这是「加总当峰值」那个错误做法的量。"""
    total = 0.0
    for name in ACCOUNT_NAMES:
        total += max(accounts(cfg, ph, setting).get(name, 0.0) for ph in PHASES)
    return total


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a(cfg):
    print("\nA 算法正确性（独立参照 / 跨域一致性）")
    P = psi(cfg)

    judge(
        "A.1 参数量：数元素 == 闭式",
        psi_reference(cfg) == P,
        f"{psi_reference(cfg)} == {P}（{P/1e9:.3f}B）",
    )

    comp = 2 + 2 + SPECS.OPTIMIZER_STATE_BYTES_PER_PARAM
    judge(
        "A.2 `16Ψ` 的分解复现（**消费** 90-training 的正文，不重推）",
        comp == SPECS.TRAINING_STATE_BYTES_PER_PARAM
        and SPECS.TRAINING_STATE_BYTES_PER_PARAM - SPECS.CHECKPOINT_BYTES_PER_PARAM == 2,
        f"2Ψ 权重 + 2Ψ 梯度 + {SPECS.OPTIMIZER_STATE_BYTES_PER_PARAM}Ψ 优化器 = {comp}Ψ"
        f" == specs {SPECS.TRAINING_STATE_BYTES_PER_PARAM}Ψ；"
        f"训练显存 − 检查点 = {SPECS.TRAINING_STATE_BYTES_PER_PARAM - SPECS.CHECKPOINT_BYTES_PER_PARAM}Ψ"
        f" = 梯度那一项",
    )

    peaks = {ph: peak(cfg, ph) for ph in PHASES}
    total = sum_of_all_accounts(cfg)
    judge(
        "A.3 「峰值」= 各阶段取 max，而不是五本账相加",
        max(peaks.values()) < total,
        f"train {peaks['train']/2**30:.2f} / prefill {peaks['prefill']/2**30:.2f}"
        f" / decode {peaks['decode']/2**30:.2f} GiB ⇒ 峰值 {max(peaks.values())/2**30:.2f} GiB"
        f" < 加总 {total/2**30:.2f} GiB",
    )

    ratio = sparse_2to4_bytes(P, 16) / (P * 2.0)
    judge(
        "A.4 2:4 稀疏的整块宽度复算 == 36/64（跨域一致性）",
        abs(ratio - 36.0 / 64.0) < 1e-12,
        f"({P/2:.0f}×2 B 载荷 + {P/4:.0f} B metadata) / ({P:.0f}×2 B) = {ratio:.6f}"
        f" == 36/64 = {36/64:.6f}（**不是** 1/2）",
    )
    return P, peaks, total


# ---------------------------------------------------------------------------
# B 代价模型：六族的归因表
# ---------------------------------------------------------------------------

def scenarios(cfg, P):
    """六族、八个档位。每项 = (名字, 训练设置, decode 设置)。"""
    w_2to4 = sparse_2to4_bytes(P, 16)
    psi_lr = psi_low_rank(cfg, LOW_RANK_R)
    psi_student = int(P * DISTILL_STUDENT_RATIO)
    rho = SPECS.QLORA_LORA_PARAM_FRACTION
    return [
        ("量化 · 权重 W4", {"w_bits": 4}, {"w_bits": 4}),
        ("量化 · KV4", {"kv_bits": 4}, {"kv_bits": 4}),
        ("稀疏化 · 2:4 结构化", {"weights_bytes": w_2to4}, {"weights_bytes": w_2to4}),
        ("剪枝 · 50%（稀疏存储）", {"weights_bytes": P * 2.0 * 0.5}, {"weights_bytes": P * 2.0 * 0.5}),
        ("剪枝 · 50%（稠密存储）", {}, {}),
        (f"低秩分解 · r={LOW_RANK_R}", {"psi_override": psi_lr}, {"psi_override": psi_lr}),
        (f"蒸馏 · 学生 {DISTILL_STUDENT_RATIO:.0%}", {"psi_override": psi_student},
         {"psi_override": psi_student}),
        (f"PEFT · ρ={rho:.4f}", {"trainable": rho}, {}),
    ]


def class_b(cfg, P, peaks):
    print("\nB 代价模型（六族的归因表）")
    base_train, base_decode = accounts(cfg, "train"), accounts(cfg, "decode")
    print(f"  {'族 / 档位':<26}{'训练峰值':>10}{'降幅':>8}{'decode':>10}{'降幅':>8}  {'动到的账'}")
    rows = []
    for name, st_tr, st_de in scenarios(cfg, P):
        tr, de = accounts(cfg, "train", st_tr), accounts(cfg, "decode", st_de)
        p_tr, p_de = sum(tr.values()), sum(de.values())
        moved = sorted(k for k in ACCOUNT_NAMES
                       if abs(tr.get(k, 0.0) - base_train.get(k, 0.0)) > 1e-6
                       or abs(de.get(k, 0.0) - base_decode.get(k, 0.0)) > 1e-6)
        rows.append({"name": name, "train": p_tr, "decode": p_de, "moved": moved})
        print(f"  {name:<26}{p_tr/2**30:>9.2f}G{100*(1-p_tr/peaks['train']):>7.1f}%"
              f"{p_de/2**30:>9.2f}G{100*(1-p_de/peaks['decode']):>7.1f}%  "
              f"{','.join(moved) if moved else '（一本都没动）'}")

    by_name = {r["name"].split(" · ")[0] + r["name"].split(" · ")[1][:4]: r for r in rows}
    judge(
        "B.1 六族里**没有一族**同时动到全部五本账",
        all(len(r["moved"]) < 5 for r in rows),
        "各族动到 "
        + "；".join(f"{r['name'].split(' · ')[0]}→{len(r['moved'])} 本" for r in rows),
    )
    peft = next(r for r in rows if r["name"].startswith("PEFT"))
    prune_dense = next(r for r in rows if "稠密存储" in r["name"])
    judge(
        "B.2 PEFT 动「梯度 + 优化器状态」，而**稠密存储的剪枝一本都没动**",
        peft["moved"] == ["grads", "optimizer"] and prune_dense["moved"] == [],
        f"PEFT → {peft['moved']}；剪枝（稠密存储）→ {prune_dense['moved'] or '[]'}"
        f" ⇒ 「剪枝省显存」**以改成稀疏存储为前提**",
    )
    _ = by_name
    return rows


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(cfg, P, peaks, total):
    print("\nC 边界与反例（断言的是「失败行为成立」）")

    tr_int4 = accounts(cfg, "train", {"w_bits": 4})
    red = 1 - sum(tr_int4.values()) / peaks["train"]
    judge(
        "C.1 「int4 权重 ⇒ 训练显存减半」被证伪",
        red < 0.10,
        f"训练峰值 {peaks['train']/2**30:.2f} GiB → {sum(tr_int4.values())/2**30:.2f} GiB，"
        f"只降 {red*100:.2f}%（优化器状态 {SPECS.OPTIMIZER_STATE_BYTES_PER_PARAM}Ψ 没动）",
    )

    judge(
        "C.2 「五本账相加 == 峰值」被证伪",
        total > peaks["train"],
        f"加总 {total/2**30:.2f} GiB > 峰值 {peaks['train']/2**30:.2f} GiB"
        f"（多出来的是**别的阶段**才活着的账）",
    )

    over = (SPECS.TRAINING_STATE_BYTES_PER_PARAM - SPECS.CHECKPOINT_BYTES_PER_PARAM) * P
    judge(
        "C.3 把 `16Ψ` 当检查点大小会**恰好多算一份梯度**",
        over == 2 * P,
        f"({SPECS.TRAINING_STATE_BYTES_PER_PARAM} − {SPECS.CHECKPOINT_BYTES_PER_PARAM})Ψ = 2Ψ"
        f" = {over/2**30:.2f} GiB",
    )

    base_opt = accounts(cfg, "train")["optimizer"]
    judge(
        "C.4 「量化能省优化器状态」被证伪",
        tr_int4["optimizer"] == base_opt,
        f"int4 权重的优化器状态 {tr_int4['optimizer']/2**30:.2f} GiB"
        f" == fp16 的 {base_opt/2**30:.2f} GiB（差 **恰好 0**）⇒ 要省这一本只能走 PEFT 或蒸馏",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("130-model-optimization/overview 02 —— 五本账与三种峰值")
    print("★ 本程序只做算术，产出是**推导值**，不是实测值。")
    cfg = SPECS.EXAMPLE_MODEL
    P, peaks, total = class_a(cfg)
    print(f"  三种峰值（示例工作负载：train B={TRAIN_BATCH}/S={TRAIN_SEQ}，"
          f"prefill B={PREFILL_BATCH}/S={PREFILL_SEQ}，decode B={DECODE_BATCH}/S={DECODE_SEQ}）：")
    for ph in PHASES:
        detail = " + ".join(f"{k} {v/2**30:.2f}G" for k, v in accounts(cfg, ph).items())
        print(f"    {ph:<8} {peaks[ph]/2**30:>7.2f} GiB = {detail}")
    class_b(cfg, P, peaks)
    class_c(cfg, P, peaks, total)
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
