#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_depth_width.py —— 深度与宽度怎么换；参数量在 embedding / attention / FFN 间的分布；
以及 ★ **架构规格表：逐字段标注「config 原生」还是「推导」**。

配套文档：`03-depth-and-width.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 真值标注（逐处都要带）：

  · **外部规格值**：两个真实模型的 config 来自 `specs.py`（[B]，转记自
    `45-kernel-lifecycle/requirement-discovery/code/specs.py`）；公开参数量 `8.03 B` 同理。
  · **推导值**：本程序算出的每一个参数量、占比、替换关系。
    测量条件：`python3 3.13.5` + `numpy 2.4.4`（本程序只用标准库与整数运算），机器 macOS arm64，**无 GPU**。

★★ `tie_word_embeddings` 的口径（计划 §4 **R15**）：`8.030 B` 是 **untied** 口径；
`probes/probe_80g.py`【C】的默认是 **tied（7.505 B）**。**两个数不可混用** ⇒
本程序把 `TIE_EMBEDDINGS = False` 写成**显式的模块级常量**，并在 `C.2` 把两个口径都算出来。

★★ 可证伪断言（把这四段改错，本程序必须非零退出）：

  · `A.1`  两条**独立**路径算 Llama-3-8B 的参数（闭式 vs **逐个模块枚举形状**）给出同一个整数；
  · `B.1`  反算总参数量落在公开值 `8.03 B` 的 **±1%** 内，**且**与
           `45-…/code/01_operator_inventory.py:549` 的逐位断言 `8029995008` 相同；
  · `B.4`  ★ 规格表的机械判据：标「**原生**」的字段必须在来源字典里；
           标「**推导**」的必须**给出推导式且复现出该值**（`head_dim` 就是这一条）；
  · `C.1`  **反例**：Qwen3-30B-A3B 用 `hidden_size // num_attention_heads` 代替 `head_dim`
           会把 attention 的宽度**少算一半**。

用法：
    python3 03_depth_width.py
"""

import ast
import collections
import inspect
import os
import sys

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（transformer）取 `!! FAIL: transformer` —— 与同域另外四个主题
#   写死的取值不同（域 README 规定了五个不同形态），**不跨主题照抄**是本库纪律。
ASSERT_FAIL_MARK = "!! FAIL: transformer"

OTHER_THEME_MARKS = ("@@ FAIL: attention-variants", "## FAIL: moe",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

FAILS = []
MARKS_USED = set()
# ★ 计数器用 `Counter()` 而**不是**字面量字典：`mutate.py` 的分母是「模块级常量的
#   **字面量叶子**」，写成 `{"A": 0, ...}` 会把 8 个**不承载任何判据**的初值塞进分母
#   （改它们只动打印出来的汇总行，不动任何结论）⇒ 那是给分母灌水。
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def _c(v):
    """千分位（★ Python 的 `%` 格式化**没有** `%,d`，那是 C 的写法）。"""
    return format(int(v), ",")


def check(name, cond, detail=""):
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-5s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-5s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值（★ 全部在 `D.3`–`D.6` 被钉住）
# ===========================================================================
# ★★ 计划 §4 R15 点名要求：**必须显式指定**，不得沿用任何工具的默认值。
TIE_EMBEDDINGS = False       # ⇒ 嵌入与 `lm_head` **各算一次**（这才是 8.030 B 的口径）

PARAM_ANCHOR_B = 8.03e9      # 公开参数量（`specs.PUBLISHED_PARAM_COUNT_B` 的那个 8.03 B）
PARAM_ANCHOR_TOL = 0.01      # 表 A ⑥ 的阈值：**±1%**（★ 阈值必须写出来）
BIT_EXACT_FROM_45 = 8029995008   # ★ 45-…/code/01_operator_inventory.py:549 的**逐位断言**
NOMINAL_SIZE_B = 30e9        # Qwen3-**30B**-A3B 的**名字**里那个 30B（C.4 只用它做量级对照）

# ★★ 规格表的机械判据（判据 ⑦）：来源组 `"upstream"` / `"mirror"` 都算**原生**
#   （真实 `config.json` 里真有这个字段），只有 `"derived"` 是**推导**。
NATIVE_GROUPS = ("upstream", "mirror")

# ★ 推导字段的**推导式**：判据 ⑦ 要求「标『推导』的必须给出推导式」。
#   写成 `字段名 -> (说明, 求值函数)`，求值函数拿 config 算出该字段的值。
DERIVATIONS = {
    "head_dim": ("hidden_size // num_attention_heads",
                 lambda c: c["hidden_size"] // c["num_attention_heads"]),
}

# ★ 判据 ⑦ 与计划 §2 的边界：`head_dim` 那个例子的**正文已经在**
#   `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276`（§4.5），
#   判据在 `:288`。⇒ 本域**只做「逐字段标注」这一新增部分，不重推那个例子**。
HEAD_DIM_EXAMPLE_BACKLINK = "45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276"

# ★★ 规格表的**逐位钉住**（判据 ⑦ 的另一半）。
#   先例与理由：`45-kernel-lifecycle/requirement-discovery/code/01_operator_inventory.py:526-541`
#   的「C6：逐个字段钉住」—— 只断言「落在 1% 内」会放过任何一个字段的小改动
#   （`intermediate_size` 14336 → 14343 只移动 0.03%），所以规格表**逐位比**。
#   ★ 这些字面量在**函数体**里（不是模块级常量），因此**不在** `mutate.py` 的分母里 ——
#     本库口径把「断言调用里的字面量」当**期望值**排除，而这里正是期望值。
EXPECTED_CONFIG = {
    "Meta-Llama-3-8B": {
        "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
        "tie_word_embeddings": False, "hidden_act": "silu",
        # ★ 这一条是**推导值**（Llama 的 config 里没有该字段），其余都是原生：
        "head_dim": 128,
    },
    "Qwen3-30B-A3B": {
        "num_hidden_layers": 48, "hidden_size": 2048, "num_attention_heads": 32,
        "num_key_value_heads": 4, "intermediate_size": 6144, "vocab_size": 151936,
        "tie_word_embeddings": False, "hidden_act": "silu",
        # ★ 这一条是**原生**（Qwen3 的 config 里真有该字段），且 ≠ hidden/heads = 64：
        "head_dim": 128,
    },
}

# ★ 深度/宽度替换的实验点（B.4 用）
BASE_LAYERS, BASE_HIDDEN = 32, 4096
# ★ GQA 的比值（`num_attention_heads / num_key_value_heads`）在宽度缩放时**必须保持**，
#   否则 k/v 的宽度不缩、attention 的参数量会随宽度**涨**（B.4 的口径说明）。
GQA_RATIO = 4


# ===========================================================================
# 2. 参数量：闭式（被测路径）
# ===========================================================================
def params_closed_form(cfg, tie):
    """按**部件的闭式**算参数量（SwiGLU 的 FFN 是**三个**矩阵 —— 见 `02-ffn-shapes`）。

    `attn` 一层的三块：`q`（`d → h_q·d_h`）、`k`/`v`（各 `d → h_kv·d_h`）、`o`（`h_q·d_h → d`）。
    `ffn` 一层三个矩阵：`gate`/`up`（`d → d_ff`）、`down`（`d_ff → d`）。
    """
    d = cfg["hidden_size"]
    V = cfg["vocab_size"]
    L = cfg["num_hidden_layers"]
    dh = cfg["head_dim"]                       # ★ **用字段**，不做那个除法（见判据 ⑦ 与 C.1）
    dq = cfg["num_attention_heads"] * dh
    dkv = cfg["num_key_value_heads"] * dh
    dff = cfg["intermediate_size"]
    emb = V * d
    lm_head = 0 if tie else V * d
    attn = L * (d * dq + 2 * d * dkv + dq * d)
    ffn = L * 3 * d * dff
    return {"embed": emb, "attention": attn, "ffn": ffn, "lm_head": lm_head,
            "total": emb + attn + ffn + lm_head}


def param_shapes(cfg, tie):
    """★ **独立参照**：不套闭式，**逐个模块枚举 (名字, 形状)**，最后把元素数加起来。

    与闭式是两条不同的代码路径：闭式把 32 层写成 `L * (...)`，这里**真的展开 32 层**。
    它**不调用** `params_closed_form()`（`D.7`/`D.8` 用 AST 守着）。
    """
    d, V, L = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    dh = cfg["head_dim"]
    dq = cfg["num_attention_heads"] * dh
    dkv = cfg["num_key_value_heads"] * dh
    dff = cfg["intermediate_size"]
    out = [("model.embed_tokens.weight", (V, d))]
    if not tie:
        out.append(("lm_head.weight", (V, d)))
    for l in range(L):
        p = "model.layers.%d." % l
        out += [(p + "self_attn.q_proj.weight", (dq, d)),
                (p + "self_attn.k_proj.weight", (dkv, d)),
                (p + "self_attn.v_proj.weight", (dkv, d)),
                (p + "self_attn.o_proj.weight", (d, dq)),
                (p + "mlp.gate_proj.weight", (dff, d)),
                (p + "mlp.up_proj.weight", (dff, d)),
                (p + "mlp.down_proj.weight", (d, dff))]
    return out


def total_from_shapes(shapes):
    return sum(m * n for _, (m, n) in shapes)


# ===========================================================================
# 3. A 类：两条独立路径
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 参数量：闭式 与 「逐个模块枚举形状」是两条独立路径")
    print("=" * 74)
    print("  口径：`TIE_EMBEDDINGS = %s`（★ 计划 §4 R15：**必须显式指定**，"
          "不得沿用工具默认值）" % TIE_EMBEDDINGS)
    ok = True
    for name in ("Meta-Llama-3-8B", "Qwen3-30B-A3B"):
        cfg = specs.LLAMA3_8B_CONFIG if name.startswith("Meta") else specs.QWEN3_30B_A3B_CONFIG
        closed = params_closed_form(cfg, tie=TIE_EMBEDDINGS)["total"]
        shapes = param_shapes(cfg, tie=TIE_EMBEDDINGS)
        enum = total_from_shapes(shapes)
        ok = ok and (closed == enum)
        print("  %-16s 闭式 = %s ；枚举 %d 个模块 = %s"
              % (name, _c(closed), len(shapes), _c(enum)))
    check("A.1", ok,
          "★ 两条**独立**路径给出同一个整数（整数计数没有舍入 ⇒ 这里 `==` 是对的，"
          "不是 P18 那种浮点情形）")

    lb = specs.LLAMA3_8B_CONFIG
    p = params_closed_form(lb, tie=TIE_EMBEDDINGS)
    check("A.2", p["total"] == sum(p[k] for k in ("embed", "attention", "ffn", "lm_head")),
          "分项之和 == 总数（部分-整体自洽）")
    check("A.3", p["lm_head"] == lb["vocab_size"] * lb["hidden_size"],
          "`lm_head` 那一项恰好是 `V·d` = %s（untied 口径的定义）" % _c(p["lm_head"]))
    return lb


# ===========================================================================
# 4. B 类：代价模型 —— 反算、分布、深度/宽度替换、★ 规格表
# ===========================================================================
def part_B(lb):
    print()
    print("=" * 74)
    print("B. 代价模型：反算、参数分布、深度/宽度的替换、★ 规格表的逐字段标注")
    print("=" * 74)

    # ── ⑥ 反算 Llama-3-8B，与公开参数量对照 ────────────────────────────────
    p = params_closed_form(lb, tie=TIE_EMBEDDINGS)
    total = p["total"]
    anchor = specs.PUBLISHED_PARAM_COUNT_B["Meta-Llama-3-8B"] * 1e9
    rel = abs(total - anchor) / anchor
    print("  Llama-3-8B（untied）：embed %s + attention %s + ffn %s + lm_head %s"
          % (_c(p["embed"]), _c(p["attention"]), _c(p["ffn"]), _c(p["lm_head"])))
    print("    = **%s**（= %.6f B）" % (_c(total), total / 1e9))
    print("    公开参数量 %.2f B ⇒ 相对差 %.2e（阈值 ±%g）"
          % (anchor / 1e9, rel, PARAM_ANCHOR_TOL))
    check("B.1", rel < PARAM_ANCHOR_TOL,
          "反算落在公开值 8.03 B 的 ±%g%% 内（实际 %.2e）" % (PARAM_ANCHOR_TOL * 100, rel))
    print("    ⚠ 阈值为什么这么松：公开值本身只给到 3 位有效数字 ⇒ 「±1%」这条判据"
          "**分辨不出** 8.029995008 与 7.95。真正的强对照是下面这条：")
    print("    回链：`45-kernel-lifecycle/requirement-discovery/code/01_operator_inventory.py:549`")
    print("          那里的逐位断言是 %s —— 本域算出的 %s"
          % (_c(BIT_EXACT_FROM_45), _c(total)))
    check("B.2", total == BIT_EXACT_FROM_45,
          "★ 与 45 域那条**逐位断言**相同（%s）—— 本域**只把它当一次对照，"
          "不重做整套算子清单**（那是 45 的正文，见域 README 的边界表）" % _c(total))

    # ── 参数分布 ───────────────────────────────────────────────────────────
    print()
    print("  参数在四个部件间的分布（Llama-3-8B，untied）：")
    for key, label in (("embed", "嵌入"), ("attention", "attention"), ("ffn", "FFN"),
                       ("lm_head", "lm_head")):
        print("    %-10s %14s   %6.2f%%" % (label, _c(p[key]), p[key] / total * 100))
    share = {k: p[k] / total for k in ("embed", "attention", "ffn", "lm_head")}
    check("B.3", abs(sum(share.values()) - 1.0) < 1e-12 and share["ffn"] > 0.6,
          "占比之和 == 1（浮点档 < 1e-12），且 **FFN 一项就占 %.1f%%** ——"
          "「参数主要在 FFN 里」这句话在本程序的设定下是**可算出来的**，不是转述"
          % (share["ffn"] * 100))

    # ── 深度/宽度替换：等参数预算下 d ∝ L^(−1/2) ──────────────────────────
    print()
    print("  ★ 深度与宽度怎么换：固定 (attn + ffn) 的预算时，宽度按 `d ∝ L^(−1/2)` 缩")
    print("    （口径：`head_dim` 与 GQA 的比值**保持不变**，头数随宽度缩放 —— 这是")
    print("      Llama-3 那一族的做法：d=4096、32 头、d_h=128、GQA 4:1；`d_ff` 也跟着 d 缩）")
    base = p["attention"] + p["ffn"]
    rows = []
    for mult in (1, 4, 16, 64):
        # ★ 取 mult 为 4 的幂：这样 `4096/√mult` 恰好落在 head_dim=128 的网格上，
        #   而且头数仍是 4 的倍数（GQA 4:1 成立）—— 见 C.4 的离散化边界。
        L = BASE_LAYERS * mult
        d = BASE_HIDDEN // int(mult ** 0.5)
        cfg = dict(lb)
        cfg["num_hidden_layers"] = L
        cfg["hidden_size"] = d
        cfg["num_attention_heads"] = d // lb["head_dim"]
        # ★ GQA 的比值保持 4:1（`num_key_value_heads` 必须跟着头数缩，否则 k/v 的宽度不变、
        #   attn 项会随深度**涨**而不是持平 —— 本骨架第一次跑就把它打出来了）
        cfg["num_key_value_heads"] = max(1, cfg["num_attention_heads"] // GQA_RATIO)
        cfg["intermediate_size"] = int(round(cfg["intermediate_size"] * d / BASE_HIDDEN))
        q = params_closed_form(cfg, tie=TIE_EMBEDDINGS)
        sub = q["attention"] + q["ffn"]
        rows.append((mult, L, d, sub, q["embed"]))
        print("    L=%4d（%2d×）、d=%4d（÷√%2d）：attn+ffn = %14s （基准的 %.4f×）"
              " · 嵌入 = %12s （基准的 %.4f×）"
              % (L, mult, d, mult, _c(sub), sub / base, _c(q["embed"]),
                 q["embed"] / p["embed"]))
    sub_ok = all(r[3] == base for r in rows)
    emb_ok = all(abs(r[4] * (r[0] ** 0.5) / p["embed"] - 1.0) < 1e-9 for r in rows)
    check("B.4", sub_ok and emb_ok,
          "★ 「加深 = 变窄」的账：层数 ×4 配宽度 ÷2（= `d ∝ L^(−1/2)`），"
          "**attn+ffn 逐位不变**（整数意义下完全相同，不是近似）；"
          "代价落在**嵌入**上：`V·d` 恰好按 `1/√mult` 缩（×1、1/2、1/4、1/8）")
    print("    ⇒ 这就是「深度/宽度怎么换」的定量形态：**FFN 与 attention 的参数 ∝ L·d²，"
          "嵌入 ∝ V·d** ⇒ 等预算下深化的代价是**词表那两项**（也正因如此，"
          "MoE 那条路选择「加宽 FFN 而不激活它」，见 `moe` 主题）。")

    # ── ★ 规格表：逐字段标注 ───────────────────────────────────────────────
    print()
    print("  ★★ 架构规格表：**逐字段**标注「config 原生」/「推导」（判据 ⑦）")
    ok = True
    # ★ 先验一道**完整性**：来源组的标签只能是「已知的两类原生 + `derived`」。
    #   ★ 少了这一条，一个被改坏的标签会走到 `DERIVATIONS[f]` 上抛 `KeyError` ——
    #     那是**崩溃**，`mutate.py` 会把它单独计数、**不算真捕获**（口径见其模块头）。
    known_labels = set(NATIVE_GROUPS) | {"derived"}
    bad_labels = sorted({g for name in ("Meta-Llama-3-8B", "Qwen3-30B-A3B")
                         for g in specs.MODEL_FIELD_GROUP[name].values()
                         if g not in known_labels})
    check("B.5", not bad_labels,
          "★ 规格表的标签完整性：全部字段的来源组 ∈ %s（越界标签会在这里被抓住，"
          "而不是在后面崩成 traceback）%s"
          % (sorted(known_labels), "" if not bad_labels else " ← 越界 %s" % bad_labels))
    for name in ("Meta-Llama-3-8B", "Qwen3-30B-A3B"):
        cfg = specs.LLAMA3_8B_CONFIG if name.startswith("Meta") else specs.QWEN3_30B_A3B_CONFIG
        group = specs.MODEL_FIELD_GROUP[name]
        print("    【%s】" % name)
        for f in sorted(cfg):
            g = group[f]
            mark = ("config 原生" if g in NATIVE_GROUPS
                    else "推导（%s）" % (DERIVATIONS[f][0] if f in DERIVATIONS
                                       else "**未登记**"))
            print("      %-24s = %-10s  [%s]  %s"
                  % (f, cfg[f], g, mark))
            if g in NATIVE_GROUPS:
                # 机械判据之一：标「原生」的**必须在来源字典里**
                src = (specs.LLAMA3_8B_FIELDS_FROM_UPSTREAM if name.startswith("Meta")
                       else specs.QWEN3_30B_A3B_FIELDS_FROM_UPSTREAM)
                src = src if g == "upstream" else (
                    specs.LLAMA3_8B_FIELDS_FROM_MIRROR if name.startswith("Meta")
                    else specs.QWEN3_30B_A3B_FIELDS_FROM_MIRROR)
                ok = ok and (f in src)
            else:
                # 机械判据之二：标「推导」的**必须给出推导式，且复现出该值**
                ok = ok and (f in DERIVATIONS and DERIVATIONS[f][1](cfg) == cfg[f])
        ok = ok and set(group) == set(cfg)
    check("B.5b", ok,
          "★ 判据 ⑦ 的两条机械判据都成立：标「原生」的字段都在来源字典里；"
          "标「推导」的字段都有推导式且**复现出该值**（本域只有 `head_dim` 一个推导字段，"
          "且只出现在 Llama 上）")

    # 推理组的划分不变量（`specs.py` 自己声明的）
    lbg = specs.MODEL_FIELD_GROUP["Meta-Llama-3-8B"]
    qwg = specs.MODEL_FIELD_GROUP["Qwen3-30B-A3B"]
    parts_ok = True
    for name, grp, dic in (("Meta-Llama-3-8B", lbg,
                            (specs.LLAMA3_8B_FIELDS_FROM_UPSTREAM,
                             specs.LLAMA3_8B_FIELDS_FROM_MIRROR,
                             specs.LLAMA3_8B_FIELDS_DERIVED)),
                           ("Qwen3-30B-A3B", qwg,
                            (specs.QWEN3_30B_A3B_FIELDS_FROM_UPSTREAM,
                             specs.QWEN3_30B_A3B_FIELDS_FROM_MIRROR,
                             specs.QWEN3_30B_A3B_FIELDS_DERIVED))):
        sets = [set(x) for x in dic]
        pairwise = not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        covers = (sets[0] | sets[1] | sets[2]) == set(grp)
        parts_ok = parts_ok and pairwise and covers
        print("    %-16s 三组的规模：upstream %d + mirror %d + derived %d = %d；"
              "两两不交 %s；并集 == config 的字段集 %s"
              % (name, len(sets[0]), len(sets[1]), len(sets[2]), len(grp), pairwise, covers))
    check("B.6", parts_ok,
          "★ `specs.py` 声明的那个**可机械判定的不变量**（三组必须构成一个划分）成立 ——"
          "它挡住「某个字段被静默换组」这类漂移")
    return p


# ===========================================================================
# 5. C 类：边界与反例
# ===========================================================================
def part_C(lb):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)

    qw = specs.QWEN3_30B_A3B_CONFIG
    _lbp = params_closed_form(lb, tie=TIE_EMBEDDINGS)
    base = _lbp["attention"] + _lbp["ffn"]          # 基准（L=32、d=4096）的 attn+ffn

    # ── C.1 ★ head_dim：Qwen3 上「除以头数」会少算一半 ─────────────────────
    print("  C.1：`hidden_size // num_attention_heads` 代替 `head_dim` 的代价")
    print("    （★ 这个例子的**正文已在** `%s`，" % HEAD_DIM_EXAMPLE_BACKLINK)
    print("       判据在 `:288`；本程序只把它做成**可执行的对照**，不重推那个例子）")
    for name, cfg in (("Meta-Llama-3-8B", lb), ("Qwen3-30B-A3B", qw)):
        d = cfg["hidden_size"]
        hq = cfg["num_attention_heads"]
        dh = cfg["head_dim"]
        wrong = d // hq
        print("    %-16s hidden/heads = %4d/%2d = %-4d · config 的 head_dim = %-4d · 相等？ %s"
              % (name, d, hq, wrong, dh, wrong == dh))
    cfg_bad = dict(qw)
    cfg_bad["head_dim"] = qw["hidden_size"] // qw["num_attention_heads"]
    good = params_closed_form(qw, tie=TIE_EMBEDDINGS)["attention"]
    bad = params_closed_form(cfg_bad, tie=TIE_EMBEDDINGS)["attention"]
    print("    Qwen3-30B-A3B 的 attention 项：正确 %s · 用除法 %s（少 %.1f%%）"
          % (_c(good), _c(bad), (1 - bad / good) * 100))
    check("C.1", (qw["head_dim"] != qw["hidden_size"] // qw["num_attention_heads"])
          and (lb["head_dim"] == lb["hidden_size"] // lb["num_attention_heads"])
          and bad < good * 0.75,
          "★ Qwen3 上两者**不等**（128 vs 64）⇒ 用除法会把 attention 的宽度**少算一半**、"
          "参数量少 %.1f%%；而 Llama 上两者**恰好相等**（128）—— "
          "**「两个模型都这么算」这种说法只在其中一个上碰巧成立**"
          % ((1 - bad / good) * 100))

    # ── C.2 ★ tie 口径：7.505 与 8.030 不可混用（R15）─────────────────────
    print()
    print("  C.2：`tie_word_embeddings` 两个口径 —— 计划的 R15 点名的现场")
    untied = params_closed_form(lb, tie=False)["total"]
    tied = params_closed_form(lb, tie=True)["total"]
    print("    untied（本域口径，D21 的 8.030 B）: %s = %.6f B" % (_c(untied), untied / 1e9))
    print("    tied  （`probe_80g.py`【C】的默认）  : %s = %.6f B" % (_c(tied), tied / 1e9))
    print("    差 = V·d = %s（占 untied 的 %.2f%%）"
          % (_c(untied - tied), (untied - tied) / untied * 100))
    check("C.2", untied - tied == lb["vocab_size"] * lb["hidden_size"]
          and (untied - tied) / untied > 0.05,
          "★ 两个口径差**恰好一项 `V·d`**（%.2f%%）⇒ 混用它们会让「反算 vs 公开值」"
          "的对照直接失真。★ 本程序**显式**写死 `TIE_EMBEDDINGS = False`（R15）"
          % ((untied - tied) / untied * 100))
    print("    ⚠ 该字段的限定语：`tie_word_embeddings` **不在** `45/…/code/specs.py` 里，")
    print("      本域经 `hf-mirror.com` 镜像取得（`false`），**未与官方 repo 比对 sha256**；")
    print("      上游 `45-…/code/specs.py:128` 也逐字把它标为「**未核实**」⇒ **不静默升格**。")

    # ── C.3 反例：把 dense 公式套在 MoE 上 ────────────────────────────────
    print()
    print("  C.3：反例 —— 把 dense 的 FFN 公式套在 **MoE** 上（计划 §0.1 的 P15）")
    dense = params_closed_form(qw, tie=TIE_EMBEDDINGS)["total"]
    print("    Qwen3-30B-A3B（它是 MoE）：按 dense 公式算 = %s = %.3f B"
          % (_c(dense), dense / 1e9))
    print("    而模型名的 nominal 规模是 %.0f B ⇒ 差了 %.1f×"
          % (NOMINAL_SIZE_B / 1e9, NOMINAL_SIZE_B / dense))
    print("    ⇒ **不是 dense 的公式写错了，是把 MoE 当成了 dense**：")
    print("      `intermediate_size = %d` 只是「**非 MoE 的 FFN 宽度**」，"
          "专家宽度在 `moe_intermediate_size` 里。" % qw["intermediate_size"])
    print("      ★ 正式处置在 `moe` 主题（P15 的现场在那一侧）；本域**不下结论**，")
    print("        只把这条反例做成可执行的量级对照。")
    check("C.3", dense * 5 < NOMINAL_SIZE_B,
          "★ 用 dense 公式得到 %.3f B，比模型名的 nominal 规模小 %.1f×（>5×）"
          "⇒ 这条错法的量级足够大、不会看不出来"
          % (dense / 1e9, NOMINAL_SIZE_B / dense))

    # ── C.4 离散化边界：`d ∝ L^(−1/2)` 只能在 head_dim 的网格上取 ───────────
    print()
    print("  C.4：`d ∝ L^(−1/2)` 的**离散化边界** —— 网格是 head_dim=128 与 GQA 的 4:1")
    ideal = BASE_HIDDEN / (2 ** 0.5)
    print("    层数 ×2（L=64）时理想宽度 = 4096/√2 = %.2f；但宽度必须落在 128 的网格上，" % ideal)
    print("    且 GQA 4:1 要求头数是 4 的倍数（`num_key_value_heads` 得是整数）：")
    print("    （对照量 = **基准本身**（L=32、d=4096 的 attn+ffn）；"
          "「等参数」的理想比 = 1.0）")
    grid = {}
    for hq in (21, 22, 23, 24):
        d = hq * lb["head_dim"]
        cfg = dict(lb)
        cfg["num_hidden_layers"] = 2 * BASE_LAYERS
        cfg["hidden_size"] = d
        cfg["num_attention_heads"] = hq
        cfg["num_key_value_heads"] = max(1, hq // GQA_RATIO)
        cfg["intermediate_size"] = int(round(lb["intermediate_size"] / BASE_HIDDEN * d))
        q = params_closed_form(cfg, tie=TIE_EMBEDDINGS)
        sub = q["attention"] + q["ffn"]
        grid[hq] = (d, sub / base, hq % 4 == 0)
        print("      头数 %2d（%s）⇒ d=%4d、d_ff=%5d：attn+ffn = 基准的 %.4f×"
              % (hq, "4 的倍数" if hq % 4 == 0 else "**不是** 4 的倍数", d,
                 cfg["intermediate_size"], sub / base))
    ok = all(x[1] != 1.0 for x in grid.values()) and any(not x[2] for x in grid.values())
    check("C.4", ok,
          "★ 理想宽度 %.2f **不在**网格上：L=64 时最近四个可实现的宽度给出 "
          "%.4f~%.4f× 的偏离（理想 1.0），且其中三个头数不是 4 的倍数"
          "（GQA 4:1 不成立）⇒ **「等参数加深」是一条连续近似，"
          "落到真实网格上必然有离散误差**"
          % (ideal, min(x[1] for x in grid.values()), max(x[1] for x in grid.values())))


# ===========================================================================
# 6. D 类：失败标记自检 + 来源/口径钉住
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正调用的函数名（★ 不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def part_D():
    print()
    print("=" * 74)
    print("D. 失败标记自检 与 来源/口径钉住")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1 and uniq == {ASSERT_FAIL_MARK}
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    # ★ 「等于该值」：把**本主题被分配的那个字面量**写死在这里。
    #   ★ 这不是同义反复：`mutate.py` 只变异**模块级**常量，函数体里的这个字面量
    #     不在它的分母里 ⇒ 它正是用来挡住「标记被改掉 / 被跨主题照抄」的那把锁。
    check("D.2", ASSERT_FAIL_MARK == "!! FAIL: transformer"
          and "FAIL" in ASSERT_FAIL_MARK and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行用的标记 ⊆ {%r}（本次用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.2b", OTHER_THEME_MARKS == ("@@ FAIL: attention-variants", "## FAIL: moe",
                                        "~~ FAIL: normalization",
                                        "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    check("D.3", TIE_EMBEDDINGS is False and PARAM_ANCHOR_TOL == 0.01
          and PARAM_ANCHOR_B == 8.03e9 and BIT_EXACT_FROM_45 == 8029995008
          and NOMINAL_SIZE_B == 30e9,
          "★ 计划 §4 R15 点名的口径：`TIE_EMBEDDINGS = False`（**显式**）、阈值 ±1%、"
          "公开值 8.03 B、45 的逐位断言 8029995008、模型名里的 nominal 30 B —— 五项逐位一致")
    check("D.4", specs.PUBLISHED_PARAM_COUNT_B == {"Meta-Llama-3-8B": 8.03},
          "公开参数量表**只有** Llama-3-8B 一项（Qwen3 的是 MoE，按 `E` 算，属 `moe` 主题 —— P15）")
    # ★ D.4c（2026-09-22 控制者加）：`UNSOURCED` 是**记录**而不是真值 —— 它没有任何
    #   数值消费者，所以 R17 修复后剩下的**唯一 3 条漏网**正是它。
    #   控制者没有把它从分母里静默排除（那正是 `mutate.py` 反对的做法），
    #   而是**给它一个消费者**：把三项逐位钉住。改任何一项（或改名），这里立刻失败。
    check("D.4c", specs.UNSOURCED == ("llama2_70b_config",
                                      "qwen3_30b_a3b_shared_expert_size",
                                      "llama3_8b_head_dim_native_field"),
          "★ `UNSOURCED` 的三项**逐位钉住**（%d 项）：它是「本域不写、但记下为什么」的清单，"
          "钉住它使它不再是一份「改掉也无人发现」的待办 —— 见计划 §4 R17 的修复说明"
          % len(specs.UNSOURCED))
    check("D.5", HEAD_DIM_EXAMPLE_BACKLINK
          == "45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276",
          "★ 判据 ⑦ 的**表 B** 那一条：引用 `head_dim` 时的回链目标写死在这里（%s）"
          % HEAD_DIM_EXAMPLE_BACKLINK)
    check("D.6", DERIVATIONS["head_dim"][0] == "hidden_size // num_attention_heads"
          and DERIVATIONS["head_dim"][1](specs.LLAMA3_8B_CONFIG) == 128
          and "head_dim" in specs.MODEL_FIELD_GROUP["Meta-Llama-3-8B"]
          and specs.MODEL_FIELD_GROUP["Meta-Llama-3-8B"]["head_dim"] == "derived"
          and specs.MODEL_FIELD_GROUP["Qwen3-30B-A3B"]["head_dim"] == "upstream",
          "★ 判据 ⑦ 要求「标『推导』的必须**给出推导式**」⇒ 那条推导式本身（"
          "「hidden_size // num_attention_heads」）也逐位钉住；\n"
          "        ★ `head_dim` 的**分组随模型而变**：Llama 上是 `derived`（真实 config 里没有该字段）、"
          "Qwen3 上是 `upstream`（原生）—— 这条正是上游那处内部矛盾（计划 §4 R13）**在本域这一侧"
          "不成立**的办法：把「原生 / 推导」逐字段标出来")

    # ── ★★ 规格表：18 个字段值 + 字段名集合，**逐位钉住** ────────────────────
    diffs, setdiff = {}, {}
    for name, cfg in (("Meta-Llama-3-8B", specs.LLAMA3_8B_CONFIG),
                      ("Qwen3-30B-A3B", specs.QWEN3_30B_A3B_CONFIG)):
        # ★ 用 `.get()` 而不是 `[name]`：若上面那张表的**模型名**被改坏，
        #   直接下标会抛 `KeyError` —— 那是**崩溃**，`mutate.py` 会把它单独计数、
        #   **不算真捕获**（口径见其模块头）。这里把它变成一条**断言失败**。
        exp = EXPECTED_CONFIG.get(name)
        if exp is None:
            diffs[name] = "**规格表里没有这个模型名**"
            continue
        d = {k: (v, cfg.get(k)) for k, v in exp.items() if cfg.get(k) != v}
        if d:
            diffs[name] = d
        if set(exp) != set(cfg):
            setdiff[name] = sorted(set(cfg) ^ set(exp))
    check("D.4b", not diffs and not setdiff,
          "★ 规格表 **9 字段 × 2 模型逐位一致**（值改动、或字段被改名，这里都会失败）%s%s"
          % ("" if not diffs else " ← 值差异 %s" % diffs,
             "" if not setdiff else " ← 字段名差异 %s" % setdiff))

    check("D.7", inspect.getsource(params_closed_form)
          != inspect.getsource(param_shapes),
          "★ 防 P17：闭式与「逐个模块枚举」的**源码不同**")
    called = _callee_names(param_shapes)
    check("D.8", not (called & {"params_closed_form", "total_from_shapes"}),
          "参照的 **AST 调用表**里没有被测侧的函数（它的调用表 = %s）" % sorted(called))


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
    print("03_depth_width.py —— 深度与宽度怎么换；参数量在四个部件间的分布；架构规格表")
    print("=" * 74)
    print("  ★ 真值标注：config 与公开参数量是**外部规格值**（`specs.py`，[B]）；")
    print("    其余全部是**推导值**（本机 `python3 %s`，纯标准库 + 整数运算）。"
          % sys.version.split()[0])
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实训练的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★ 禁区：本程序**不重做**算子清单 / 每 token FLOPs / `2ND` vs `6ND` /")
    print("    参数量的反算核对（正文在 `45-kernel-lifecycle/requirement-discovery`，此处回链）；")
    print("    `head_dim` 那个例子的正文在 `%s`（只做逐字段标注这一新增部分）。"
          % HEAD_DIM_EXAMPLE_BACKLINK)
    lb = part_A()
    part_B(lb)
    part_C(lb)
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
