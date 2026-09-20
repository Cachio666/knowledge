#!/usr/bin/env python3
r"""01-model-to-operator-inventory —— 从 config 推出算子清单。

对应文档：`../01-model-to-operator-inventory.md`

★ 本程序的可证伪断言（改错即报错，见文件末的 `assert_all_ran`）：
  A 类  从 config 反算的参数量落在公开参数量上（Llama 8.03 B / Qwen 30.5 B）
        —— 若我漏了某个矩阵（如 o_proj、或 down_proj 的转置宽度），这条会失败；
        `L` 层的 q_proj 每 token 调用 L 次、FLOPs = L·2·d·d_q —— 与手算逐位对照
  B 类  总 FLOPs 与推理口径的 `2ND` 对照，偏差被**逐项分解**（不是「大致吻合」）；
        并给出 attention 二次项超过专家 FFN 的**临界上下文长度 S\***
  C 类  ① MoE 按 dense 全专家算 ⇒ 专家部分虚高 **E/k 倍**（精确，不是「数量级」）；
        ② 该口径错误**真的翻转**「谁是第一 FLOPs 消费者」的排序；
        ③ 忽略 GQA ⇒ k_proj/v_proj 虚高 **h/h_kv 倍**

跑法：`make 01_operator_inventory`（或 `python3 01_operator_inventory.py`）。
纯 python3 + 标准库（连 numpy 都不需要），不 import torch，不碰 GPU。
"""

import sys

import specs as SP

FAILED = []


def check(name, cond, detail=""):
    print(f"    [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def sub(title):
    print()
    print(f"  ── {title}")


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# 模型：从 config 到「每 token 的算子清单」
# ===========================================================================
#
# 单位约定（**写死，否则同一张表内部会混两种口径**）：
#   · FLOPs 按 **FMA 计 2 次浮点运算**（一次乘 + 一次加）；
#   · 矩阵乘 M×K 乘 K×N 记 `2·M·N·K`；
#   · 每 token 的调用次数按 **decode（batch = 1，一次一个 token）**计；
#   · 上下文长度（已生成的 KV 长度）记作 `S`，**只影响 attention 的两个核**。


def derived_dims(cfg):
    """从 config 字段推出「每 token 的算子清单」所需的派生量。"""
    d = cfg["hidden_size"]
    hq = cfg["num_attention_heads"]
    hkv = cfg["num_key_value_heads"]
    dh = cfg["head_dim"]
    return {
        "d": d,
        # ★ d_q 用 `hq · dh`，**不用 `d`**。两者只在 hq·dh == d 时相等。
        "d_q": hq * dh,
        "d_kv": hkv * dh,
        "d_ff": cfg["moe_intermediate_size"] or cfg["intermediate_size"],
        "L": cfg["num_hidden_layers"],
        "V": cfg["vocab_size"],
        "E": cfg["num_experts"],
        "k": cfg["num_experts_per_tok"],
    }


def n_params(cfg):
    """从 config 反算总参数量。**这是 A 类断言的对照对象。**"""
    g = derived_dims(cfg)
    d, dq, dkv, dff, L, V = g["d"], g["d_q"], g["d_kv"], g["d_ff"], g["L"], g["V"]
    per_layer_attn = d * dq + 2 * d * dkv + dq * d
    if g["E"]:                      # MoE：E 个专家各 3 个矩阵
        per_layer_ffn = g["E"] * 3 * d * dff
    else:                           # dense：gate/up/down
        per_layer_ffn = 3 * d * dff
    n_embed = V * d
    n_lm_head = V * d               # 按 tie_word_embeddings=False 计（见 specs §2 的注）
    return {
        "total": L * (per_layer_attn + per_layer_ffn) + n_embed + n_lm_head,
        "per_layer_attn": per_layer_attn,
        "per_layer_ffn": per_layer_ffn,
        "embed": n_embed,
        "lm_head": n_lm_head,
    }


def inventory(cfg, S, esize=SP.FLOAT16_BYTES, moe_as_dense=False):
    """每 token 的算子清单。

    返回 list，每项 `(算子名, 每 token 调用次数, 每 token FLOPs, 每 token 字节)`。
    字节口径：**必须往返 DRAM 的唯一字节** —— 权重 + 激活的读入与写出。
    ⚠ 这是**理想下界**：它假设每个权重元素只从 DRAM 取一次。朴素实现会重复读
      （见 `02_roofline_verdict.py` 的 C 类），本函数**不**做那个口径。
    """
    g = derived_dims(cfg)
    d, dq, dkv, dff = g["d"], g["d_q"], g["d_kv"], g["d_ff"]
    L, V, E, k = g["L"], g["V"], g["E"], g["k"]
    rows = []

    def matmul(name, calls, m, nn, kk):
        """一次 `(m×kk) @ (kk×nn)`：FLOPs = 2mNk，字节 = 权重 + 激活。"""
        fl = 2.0 * m * nn * kk
        by = (kk * nn + m * kk + m * nn) * esize   # 权重 + 读入 + 写出，唯一字节
        rows.append((name, calls, calls * fl, calls * by))

    # ---- attention 的四个投影 -------------------------------------------
    matmul("q_proj",   L, 1, dq,  d)
    matmul("k_proj",   L, 1, dkv, d)
    matmul("v_proj",   L, 1, dkv, d)
    matmul("o_proj",   L, 1, d,   dq)
    # ---- attention 的两个核（**随 S 变**）--------------------------------
    # 每 token：scores = Q(1×dq) @ Kᵀ(dq×S) → 2·S·dq；PV = P(1×S) @ V(S×dq) → 2·S·dq
    # 字节：读 K、V 各 S·dkv 个元素；Q、O 各 dq 个
    atn_fl = 2.0 * S * dq * 2
    atn_by = (2 * S * dkv + 2 * dq) * esize
    rows.append(("attention_scores", L, L * atn_fl / 2, L * atn_by / 2))
    rows.append(("attention_output", L, L * atn_fl / 2, L * atn_by / 2))
    # ---- FFN -------------------------------------------------------------
    if E:
        ne = E if moe_as_dense else k        # ★ C 类的开关就在这一行
        matmul("moe_gate_proj", L * ne, 1, dff, d)
        matmul("moe_up_proj",   L * ne, 1, dff, d)
        matmul("moe_down_proj", L * ne, 1, d,   dff)
    else:
        matmul("gate_proj", L, 1, dff, d)
        matmul("up_proj",   L, 1, dff, d)
        matmul("down_proj", L, 1, d,   dff)
    # ---- 词表侧 -----------------------------------------------------------
    matmul("lm_head", 1, 1, V, d)
    rows.append(("embedding_lookup", 1, 0.0, d * esize))   # 查表：0 FLOPs
    return rows


def total(rows, idx):
    return sum(r[idx] for r in rows)


def family(name):
    """把算子名归到**算子族**。

    ★ 为什么必须聚合：FFN 被拆成 gate/up/down **三个名字**，而 attention 的二次项被拆成
      scores/output 两个名字。**按名字取最大值是在比「名字」，不是比「工作量」**——
      FFN 的三个名字各自都小于 attention_scores，但三者之和远大于它。
      这是本程序第一版犯过的错（当时 C.2 的「翻转」有一半是命名造成的伪象）。
    """
    if name.startswith("attention_"):
        return "attention_core（scores + output）"
    if name.endswith("_proj") and name[0] in "qkvo":
        return "attention_proj（q/k/v/o）"
    if "gate_proj" in name or "up_proj" in name or "down_proj" in name:
        return "FFN / MoE 专家"
    if name == "lm_head":
        return "lm_head"
    return "embedding"


def by_family(rows, idx=2):
    acc = {}
    for r in rows:
        acc[family(r[0])] = acc.get(family(r[0]), 0.0) + r[idx]
    return sorted(acc.items(), key=lambda kv: -kv[1])


def fmt(x):
    return f"{x:,.0f}"


# ===========================================================================
head("§A  算法正确性：从 config 反算的参数量与手算的调用次数")
# ===========================================================================

CFGS = [SP.LLAMA3_8B_CONFIG, SP.QWEN3_30B_A3B_CONFIG]

sub("A.1  从 config 反算参数量 vs 公开参数量")
for cfg in CFGS:
    n = n_params(cfg)["total"]
    pub = SP.PUBLISHED_PARAM_COUNT_B[cfg["name"]] * 1e9
    dev = (n - pub) / pub * 100
    print(f"    {cfg['name']:16s} 反算 = {n/1e9:8.4f} B   公开 = {pub/1e9:6.2f} B   "
          f"偏差 = {dev:+.2f}%")
    # ★ 判据：偏差 < 1%。松了这一条，A 类就变成了「只要差不多就行」。
    check(f"A.1 {cfg['name']} 反算参数量落在公开值的 1% 内", abs(dev) < 1.0,
          f"{n/1e9:.4f} B vs {pub/1e9:.2f} B")

sub("A.2  手算 `L` 层的 q_proj：每 token 调用 L 次，FLOPs = L·2·d·d_q")
for cfg in CFGS:
    g = derived_dims(cfg)
    L, d, dq = g["L"], g["d"], g["d_q"]
    rows = inventory(cfg, S=1024)
    got = [r for r in rows if r[0] == "q_proj"][0]
    hand_calls = L
    hand_flops = L * 2.0 * d * dq
    print(f"    {cfg['name']:16s} L={L:3d} d={d:5d} d_q={dq:5d}(= {cfg['num_attention_heads']}"
          f"×{cfg['head_dim']})")
    print(f"      手算：调用 {hand_calls} 次，FLOPs = {L}×2×{d}×{dq} = {fmt(hand_flops)}")
    print(f"      脚本：调用 {got[1]} 次，FLOPs = {fmt(got[2])}")
    check(f"A.2 {cfg['name']} q_proj 调用次数 == L", got[1] == hand_calls,
          f"{got[1]} vs {hand_calls}")
    check(f"A.2 {cfg['name']} q_proj FLOPs 与手算逐位一致",
          abs(got[2] - hand_flops) < 1e-6 * max(1.0, hand_flops),
          f"{got[2]!r} vs {hand_flops!r}")

sub("A.3  GQA：k_proj 的宽度是 h_kv·d_h，不是 h_q·d_h")
for cfg in CFGS:
    g = derived_dims(cfg)
    hq, hkv, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    rows = inventory(cfg, S=1024)
    got = [r for r in rows if r[0] == "k_proj"][0]
    hand_wrong = g["L"] * 2.0 * g["d"] * (hq * dh)
    hand_right = g["L"] * 2.0 * g["d"] * (hkv * dh)
    print(f"    {cfg['name']:16s} h_q={hq} h_kv={hkv} d_h={dh} ⇒ 比值 {hq/hkv:.0f}:1")
    print(f"      正确 k_proj FLOPs = {fmt(hand_right)}；误用 h_q 得 {fmt(hand_wrong)}"
          f"（×{hand_wrong/hand_right:.0f}）")
    check(f"A.3 {cfg['name']} k_proj FLOPs 用的是 h_kv", abs(got[2] - hand_right) < 1e-6 * hand_right,
          f"{fmt(got[2])}")

sub("A.4  代数恒等式：清单 == 2N − 2·嵌入 + attention 项（逐项恒等，独立于清单的生成路径）")
for cfg in CFGS:
    g = derived_dims(cfg)
    n = n_params(cfg)
    rows = inventory(cfg, S=4096)
    got = total(rows, 2)
    attn = sum(r[2] for r in rows if r[0].startswith("attention_"))
    # 2N 把「嵌入 + lm_head + 全部专家」都算成矩阵乘；清单里嵌入是查表（0 FLOPs），
    # 而且 MoE 只算 k 个专家。所以恒等式只在 dense 模型上干净。
    distinct = len({r[0] for r in rows})
    rhs = 2.0 * n["total"] - 2.0 * n["embed"] + attn
    if g["E"]:
        print(f"    {cfg['name']:16s} MoE：恒等式**不成立**（2N 含全部 E 个专家）"
              f" —— 见 §C.1")
        # MoE 上把「全部专家」换成「k 个专家」后恒等式恢复：
        dense_rows = inventory(cfg, S=4096, moe_as_dense=True)
        got_dense = total(dense_rows, 2)
        attn_d = sum(r[2] for r in dense_rows if r[0].startswith("attention_"))
        rhs_d = 2.0 * n["total"] - 2.0 * n["embed"] + attn_d
        print(f"      改按 dense 全专家算后：清单 = {fmt(got_dense)}，"
              f"2N − 2·嵌入 + attn = {fmt(rhs_d)}")
        check("A.4 Qwen3 dense 口径下恒等式成立",
              abs(got_dense - rhs_d) < 1e-6 * rhs_d, f"差 {fmt(got_dense - rhs_d)}")
    else:
        print(f"    {cfg['name']:16s} 清单 = {fmt(got)}；2N − 2·嵌入 + attn = {fmt(rhs)}")
        check(f"A.4 {cfg['name']} 恒等式成立（floating point 内精确）",
              abs(got - rhs) < 1e-6 * rhs, f"差 {fmt(got - rhs)}")
    print(f"      清单含 {distinct} 种算子、{len(rows)} 行（同种算子按调用点分行）")


# ===========================================================================
head("§B  模型与公式：与 `2ND` 对照，并逐项分解偏差")
# ===========================================================================
#
# ★ 口径（本主题最容易写错的一处，P9）：**`6ND` 是训练口径**（前向 `2ND` + 反向 `4ND`），
#   **推理的前向只有 `2ND`**。本主题是推理场景，**用 `2ND`**。
#   来源：C ≈ 6ND 的分解（2ND 前向 + 4ND 反向）是 Kaplan 式 FLOPs 记账的标准形式，
#   见本模块 §B 的说明与 `specs.py` §2 的注（[C]）。

sub("B.1  每 token FLOPs vs 2N（D = 1 token）")
S_B = 4096
for cfg in CFGS:
    g = derived_dims(cfg)
    n = n_params(cfg)
    rows = inventory(cfg, S=S_B)
    got = total(rows, 2)
    two_n = 2.0 * n["total"]
    dev = (got - two_n) / two_n * 100
    attn = sum(r[2] for r in rows if r[0].startswith("attention_"))
    print()
    print(f"    {cfg['name']}（S={S_B}, fp16）")
    print(f"      清单合计 FLOPs/token = {fmt(got)}")
    print(f"      2N（N = 反算总参数 {n['total']/1e9:.4f} B） = {fmt(two_n)}"
          f"   偏差 = {dev:+.2f}%")
    print(f"      偏差的三项来源（**这是分解，不是解释**）：")
    print(f"        + attention 二次项（2N 不含）= {fmt(attn)}")
    print(f"        − 2×嵌入（2N 计入，但查表 0 FLOPs）= {fmt(2*n['embed'])}")
    if g["E"]:
        dense_rows = inventory(cfg, S=S_B, moe_as_dense=True)
        dense_fl = total(dense_rows, 2)
        gap = dense_fl - got
        print(f"        − MoE 未激活的 {g['E']-g['k']} 个专家"
              f"（2N 计入，但每 token 不读）= {fmt(gap)}")
    residual = got - (two_n - 2 * n["embed"] + attn) - (0 if not g["E"] else -gap)
    print(f"      残差（分解后应为 0）= {fmt(residual)}")
    check(f"B.1 {cfg['name']} 偏差被三项分解后残差为 0",
          abs(residual) < 1e-6 * two_n, f"残差 {residual:.3e}")

sub("B.2  attention 二次项超过 MoE 专家 FFN 的临界上下文长度 S*")
for cfg in CFGS:
    g = derived_dims(cfg)
    dq = g["d_q"]
    L = g["L"]
    if g["E"]:
        expert_per_layer = g["k"] * 3 * 2.0 * g["d"] * g["d_ff"]
        label = f"{g['k']} 个激活专家"
    else:
        expert_per_layer = 3 * 2.0 * g["d"] * g["d_ff"]
        label = "FFN"
    # 每 token 的 attention 二次项（两层核）= L · 4·S·d_q；FFN = L · expert_per_layer
    s_star = expert_per_layer / (4.0 * dq)
    print(f"    {cfg['name']:16s} {label} 每层 = {fmt(expert_per_layer)} FLOPs；"
          f"attention 每层 = 4·S·{dq}")
    print(f"      ⇒ 临界 S* = {fmt(expert_per_layer)} ÷ {4*dq} = **{s_star:.1f}**"
          f"（S 超过它，attention 二次项更大）")
    for S in (1024, 4096, 8192):
        a = L * 4.0 * S * dq
        e = L * expert_per_layer
        who = "attention" if a > e else label
        print(f"      S={S:5d}: attention = {fmt(a):>14s}   {label} = {fmt(e):>14s}"
              f"   ⇒ 更大的是 **{who}**")
    check(f"B.2 {cfg['name']} S* 处两者相等",
          abs(L * 4.0 * s_star * dq - L * expert_per_layer) < 1e-6 * expert_per_layer)


# ===========================================================================
head("§C  边界与反例：三种「会得出错误结论」的口径，必须真的跑出那个错误")
# ===========================================================================

cfg = SP.QWEN3_30B_A3B_CONFIG
g = derived_dims(cfg)
E, k = g["E"], g["k"]

sub(f"C.1  MoE 按 dense 全专家算 ⇒ 专家部分虚高 E/k = {E}/{k}（精确，不是「数量级」）")
right = inventory(cfg, S=S_B, moe_as_dense=False)
wrong = inventory(cfg, S=S_B, moe_as_dense=True)
exp_r = sum(r[2] for r in right if r[0].startswith("moe_"))
exp_w = sum(r[2] for r in wrong if r[0].startswith("moe_"))
ratio = exp_w / exp_r
print(f"    正确的专家侧 FLOPs/token = {fmt(exp_r)}")
print(f"    按 dense 全专家算        = {fmt(exp_w)}")
print(f"    虚高倍数 = {ratio:.6f}   ← E/k = {E}/{k} = {E/k:.6f}")
check(f"C.1 专家侧虚高倍数精确等于 E/k = {E/k}", abs(ratio - E / k) < 1e-12, f"{ratio!r}")
tot_r, tot_w = total(right, 2), total(wrong, 2)
print(f"    ★ **但整模型的虚高倍数不是 E/k**：")
print(f"      正确合计 = {fmt(tot_r)}；dense 口径合计 = {fmt(tot_w)}"
      f"；倍数 = {tot_w/tot_r:.4f}  (**小于** {E/k:.0f})")
print(f"      原因：attention 与 lm_head **与专家数无关**，它们把比率拉下来了。")
print(f"      ⇒ 「MoE 虚高 = E/k」这句话**只对专家那一部分成立**，"
      f"对整模型不成立。")

sub("C.2  该口径错误**真的翻转**「第一 FLOPs 消费者」的排序（按**算子族**聚合）")
# ★ 先说清一件事：在 S=4096 上，**两个口径的第一名是同一个族**（MoE 专家）。
#   所以「dense 口径会翻转排序」这句话**不是无条件成立的**——它有一个临界 S。
#   本程序把两个 S 都跑出来，让翻转**发生在明面上**，而不是挑一个能翻转的 S 交差。
for S in (S_B, 8192):
    rr, ww = inventory(cfg, S=S), inventory(cfg, S=S, moe_as_dense=True)
    fr, fw = by_family(rr), by_family(ww)
    print(f"    S = {S}")
    print(f"      正确口径   #1 = {fr[0][0]:28s} {fmt(fr[0][1]):>16s}"
          f"   #2 = {fr[1][0]:28s} {fmt(fr[1][1]):>16s}")
    print(f"      dense 口径 #1 = {fw[0][0]:28s} {fmt(fw[0][1]):>16s}"
          f"   #2 = {fw[1][0]:28s} {fmt(fw[1][1]):>16s}")
    print(f"      ⇒ {'**翻转**' if fr[0][0] != fw[0][0] else '不翻转（第一名同族）'}")
    if S == 8192:
        check("C.2 S=8192 时两口径的第一名**不同族**（真的翻转）",
              fr[0][0] != fw[0][0], f"{fr[0][0]} vs {fw[0][0]}")
    else:
        check("C.2 S=4096 时两口径的第一名**同族**（翻转不是无条件的）",
              fr[0][0] == fw[0][0], f"{fr[0][0]}")
print(f"    ★ 翻转的临界值就是 §B.2 的 S* = {g['k']*3*2*g['d']*g['d_ff']/(4*g['d_q']):.0f}"
      f"：**S 超过它才翻转**。")
print(f"      ⇒ 正确的说法是「**在长上下文下**，忽略路由会把第一目标指错」，"
      f"不是「忽略路由一定会指错」。")

sub("C.3  忽略 GQA ⇒ k_proj/v_proj 虚高 h_q/h_kv 倍")
for c in CFGS:
    gg = derived_dims(c)
    rows = inventory(c, S=S_B)
    right_kv = sum(r[2] for r in rows if r[0] in ("k_proj", "v_proj"))
    hq, hkv, dh, d, L = (c["num_attention_heads"], c["num_key_value_heads"],
                         c["head_dim"], gg["d"], gg["L"])
    wrong_kv = 2 * L * 2.0 * d * (hq * dh)
    print(f"    {c['name']:16s} h_q/h_kv = {hq}/{hkv} = {hq//hkv}×，"
          f"k+v 正确 = {fmt(right_kv)}，误用 h_q = {fmt(wrong_kv)}"
          f"（虚高 {wrong_kv/right_kv:.0f}×）")
    check(f"C.3 {c['name']} 忽略 GQA 的虚高倍数 == h_q/h_kv",
          abs(wrong_kv / right_kv - hq / hkv) < 1e-12, f"{wrong_kv/right_kv} vs {hq/hkv}")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 4 组 · B 类 2 组 · C 类 3 组）")
print()
print("  ★ 本程序的数**全部是从 config 推出的推导值**，不是实测：")
print("    本机 macOS 无 GPU，本主题不产出任何实测性能数字。")
print(f"    python {sys.version.split()[0]} / 纯标准库（未 import numpy、未 import torch）")
