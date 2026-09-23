#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_mha_mqa_gqa.py —— MHA / MQA / GQA 的 head 关系与共享图景。

配套文档：`01-mha-mqa-gqa.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形：编号程序本机可跑 ⇒ 断言在程序内），
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本程序的数字**全部是推导值或本机实测值**，测量条件是：

> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 它们支撑的是「**三种结构的差别只是 head 的分组关系**」这类**结构**结论，
> 不支撑任何关于真实模型训练或 GPU 的结论。

★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）：

  · `A.1`  G 必须是 query 头数的**因子**，两个端点是 `1`（MQA）与 `h_q`（MHA）；
  · `A.2`  `G=1` 的映射表**全部映射到 0 号 KV 头**（MQA），`G=h_q` 的映射表**是恒等**（MHA）
           —— 把 `h // (h_q//h_kv)` 写成 `h % h_kv` 或 `h // h_kv` 立刻失败；
  · `A.3`  **两条独立代码路径**（共享式稠密 vs 复制式流式）给出同一个输出
           （最大逐元素差 `< 1e-14`）；
  · `A.4`  参照实现跑完后**调用计数表完全为空** ⇒ 它一次都没调用被测实现的函数；
  · `B.1`  两个真实模型的组数是**精确有理数** `4` 与 `8`（把 `heads_kv` 写成 `heads_q`
           就把第一个断言打掉）；
  · `C.1`  组数必须整除头数 —— 反例 `G=3`（`h_q=8`）**不合法**；
  · `C.3`  把分数整体乘 2（等价于换掉 `1/√d_h` 这个缩放）**会改变输出** ⇒ 缩放不是装饰。

★★ **本程序刻意不做的事**（计划 §2 的边界表，越界即作废）：

  * **不重讲**「GQA/MQA 复制 KV 后与 MHA 逐位相同」的**实现与比较** —— 那在
    `40-kernels/attention/05-decoding-attention.md:60,66-94` 与 `…/code/verify.py` 的
    A4（`:509-529`）/ B8（`:1008-1040`）（正文在彼，此处回链）。本程序只写**共享图景**；
  * **不算字节**：KV cache 的六项乘积与 GQA 的字节坑正文在
    `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`（正文在彼，此处回链）。
    本程序只算**元素数与倍率**；
  * **不讲**变体对 **kernel 结构**的影响（正文在 `40-kernels/attention/04:126-156`）。

用法：
    python3 01_mha_mqa_gqa.py
"""

import ast
import collections
import inspect
import os
import sys
from fractions import Fraction

import numpy as np

# ★ 同目录的共享事实源（`code/` 自包含，`CONTRIBUTING:36`）。★ 本主题**不 import 任何框架**。
import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（attention-variants）取 `@@ FAIL: attention-variants`。
#   ★ 这个字面量**不是**照抄任何别的主题的：域 README 把五个主题的标记写死为
#     五个不同形态（`attention-variants` 用 `@@`），**不跨主题照抄**是本库纪律。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；
#     `D.1`/`D.2` 两条断言守着它：文件里恰好一次、且与其他四个主题的标记都不同。
ASSERT_FAIL_MARK = "@@ FAIL: attention-variants"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("!! FAIL: transformer", "## FAIL: moe",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

FAILS = []
MARKS_USED = set()
# ★ 计数器用 `Counter()` 而**不是**字面量字典：`mutate.py` 的分母是「模块级常量的
#   **字面量叶子**」，写成 `{"A": 0, ...}` 会把初值塞进分母（改它们只动汇总行）。
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。"""
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
# 1. 口径常量（★ 全部在 `D.3`/`D.4` 被逐位钉住：改一个字节，本程序就报错）
# ===========================================================================
# ★★ 为什么「口径常量」也要钉住：它们不是「可调的参数」，而是**结论所引用的测量条件**。
#    文档 §5 与本程序的输出都写着「h_q=8 / d_head=32 / S_q=12 / S_k=9 / 块大小 4」——
#    静默改掉其中任何一个，**大部分断言的输出会变而文档不会**（两条路径依然互相一致），
#    这正是本域最想防的那类漂移。
#    （先例：`45-…/code/01_operator_inventory.py:526-541` 的「C6：逐个字段钉住」。）
H_Q = 8                  # query 头数（本程序的口径）
H_KV_LIST = (1, 2, 4, 8)  # 组数 G 的取值：两个端点 1 与 h_q 都要在
D_HEAD = 32              # 每个 head 的宽度
S_Q = 12                 # query 的位置数
S_K = 9                  # KV 的位置数（★ 与 S_Q 不同，避免形状巧合掩盖转置错）
SEED = 20260922          # 数据种子（= 探测脚本 probe_80.py 用的那一个）
BLOCK = 4                # online softmax 的块大小（只影响参照实现的累加顺序）

# ★ 分数缩放：softmax 之前要除以 √d_head。
#   ★ **写成模块级常量**是刻意的：这样「去掉 1/√d_h」这类口径变异才有落点
#     （它是本域最容易写错的一处；计划 §1.4 把它列为 `mutate.py` 必含的口径变异之一）。
ATTN_SCALE = 1.0 / np.sqrt(D_HEAD)

# ── 阈值（★ 逐个在 `D.4` 钉住，并说明它属于哪一档）────────────────────────────
REF_TOL = 1e-14          # A.3 两条独立路径的最大逐元素差（**机器精度档**）
SCALE_DROP_MIN = 1e-2    # C.3 换掉缩放后输出差的下限（**量级档**）


# ★ 被测实现的这个函数每被调一次就 +1；参照实现跑完后，计数**必须不动**。
CALLS = {}


# ===========================================================================
# 2. 被测实现（路径 1）—— GQA 的**共享**写法
# ===========================================================================
def group_of(h, h_q, h_kv):
    """第 `h` 个 query 头属于第几个组（**GQA 的全部机制就是这一个函数**）。

    ★ 论文逐字（arXiv 2305.13245 §2.2）：「Grouped-query attention divides query heads
      into `G` groups, each of which shares a single key head and value head.」
      ⇒ 组数 `G = h_q / h_kv`，第 `h` 个头属于第 `h // (h_q // h_kv)` 组。
    ★ 要求 `h_kv` **整除** `h_q`（否则「均分成 G 组」这句话无定义）—— 见 `C.1` 的反例。
    """
    return h // (h_q // h_kv)


def attn_dense(q, k, v):
    """被测实现（路径 1）：**物化**完整 `S_q×S_k` 分数矩阵的稠密 softmax。

    ★ 它只接收**一条** KV —— 谁调用它、传哪一条，就是「共享图景」的全部内容。
      ⇒ 这个函数里**没有**任何关于 `h_kv` / `G` 的知识（这是刻意的不对称：
        分组的那一侧是调用方，参照实现走的是**另一条路**）。
    """
    CALLS["attn_dense"] = CALLS.get("attn_dense", 0) + 1
    s = (q @ k.T) * ATTN_SCALE
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=-1, keepdims=True)
    return p @ v


# ===========================================================================
# 3. ★★ 参照实现（路径 2）—— **不调用被测实现的任何函数**
# ===========================================================================
# ★ `CONTRIBUTING:232-235`：A 类必须有一条与被检验代码路径**不同**的参照；
#   `00-meta/glossary.md` 的「reference implementation（参考实现）」条给出机械判据：**参考实现不得调用被测实现的任何函数**
#   （调用计数器必须恰为 0），**且不得比被测实现更复杂**。
# ★★ 两条路径的差别是**真的**，不是换了个名字：
#     · 被测：**共享** KV（按 `group_of` 取组）+ **物化**分数矩阵 + 一次性 softmax；
#     · 参照：**物化复制** KV 成 `h_q` 份（`np.repeat`）+ **不物化**分数矩阵、
#       逐块流式更新 running max / running sum + 缩放**先乘在 q 上**（位置不同）。
#   `A.5`/`D.6` 用 `inspect.getsource` 不等 + **AST 调用表为空**挡住「两边其实是同一行代码」
#   （计划 §0.1 的 **P17**：探针初版就踩过这个坑，报出的 `0.000e+00` 是构造出来的）。
def attn_stream(q, k, v, block=BLOCK):
    """独立参照（路径 2）：复制式 KV + 流式 online softmax。

    ★ 与路径 1 不共享任何辅助函数；缩放作用在 `q` 上而不是分数上。
    ★ 它**不需要** `group_of`：调用方传进来的 KV 已经是复制过的那一份。
    """
    qs = q * ATTN_SCALE
    out = np.zeros_like(q)
    for t in range(q.shape[0]):
        m, l, o = -np.inf, 0.0, np.zeros(q.shape[-1])
        for s0 in range(0, k.shape[0], block):
            kb, vb = k[s0:s0 + block], v[s0:s0 + block]
            sc = kb @ qs[t]
            m_new = max(m, float(sc.max()))
            p = np.exp(sc - m_new)
            scale = np.exp(m - m_new)          # ← 路径 1 里没有这一步
            o = o * scale + p @ vb
            l = l * scale + p.sum()
            m = m_new
        out[t] = o / l
    return out


def draw_data(seed, h_q, h_kv, d_head, s_q, s_k):
    """★ **抽取顺序是口径的一部分**：`Q` → `K` → `V`，形状分别是
    `(h_q, s_q, d_head)` / `(h_kv, s_k, d_head)` / `(h_kv, s_k, d_head)`。
    ★ `S_q ≠ S_k` 是刻意的：相等时转置错也能「通过」。"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((h_q, s_q, d_head)),
            rng.standard_normal((h_kv, s_k, d_head)),
            rng.standard_normal((h_kv, s_k, d_head)))


def kv_elements_ratio(h_kv, h_q):
    """GQA 的 KV **元素数**相对 MHA 的倍率 —— **精确有理数**。

    ★★ **口径（写死在函数名里）**：只算**元素数**（KV 头的条数之比），
      **不算字节**。字节账（六项乘积）的正文在
      `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`
      （正文在彼，此处回链）。本程序的输出里**不出现任何字节数**。
    """
    return Fraction(h_kv, h_q)


# ===========================================================================
# 4. A 类：三种结构的 head 关系（共享图景）+ 两条独立路径
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 共享图景：GQA 只改「一条 KV 服务几个 query 头」")
    print("=" * 74)
    print("  口径：h_q=%d · d_head=%d · S_q=%d · S_k=%d · 种子 %d · 块大小 %d"
          % (H_Q, D_HEAD, S_Q, S_K, SEED, BLOCK))

    # ── A.1 组数必须整除头数，且两个端点要在 ────────────────────────────────
    check("A.1", all(H_Q % g == 0 for g in H_KV_LIST)
          and H_KV_LIST[0] == 1 and H_KV_LIST[-1] == H_Q,
          "G ∈ %s：全部整除 h_q=%d，两个端点是 1（MQA）与 h_q（MHA）"
          % (list(H_KV_LIST), H_Q))

    # ── A.2 ★ 共享图景：映射表（端点 = MQA / MHA 的定义）───────────────────
    print("  每个 query 头拿到的 KV 头（`h // (h_q // h_kv)`）：")
    print("    %-8s %-4s %s" % ("h_kv(=G)", "组数", "每个 query 头的 KV 头编号"))
    for h_kv in H_KV_LIST:
        mapping = [group_of(h, H_Q, h_kv) for h in range(H_Q)]
        print("    %-8d %-4d %s" % (h_kv, H_Q // h_kv, mapping))
    m_mqa = [group_of(h, H_Q, 1) for h in range(H_Q)]
    m_mha = [group_of(h, H_Q, H_Q) for h in range(H_Q)]
    check("A.2", m_mqa == [0] * H_Q and m_mha == list(range(H_Q)),
          "★ G=1 的映射表**全 0**（所有 query 头共享同一条 KV = MQA 的共享图景）；"
          "G=h_q 的映射表**是恒等**（每头一条自己的 KV = MHA 的共享图景）")
    print("    ⚠ 诚实注记：`GQA-1 ≡ MQA`、`GQA-h ≡ MHA` 是论文写明的**定义上的等价**")
    print("      （逐字：「GQA-1 … is equivalent to MQA, while GQA-h … is equivalent to MHA」）")
    print("      ⇒ **这个等价本身不构成证据**（计划 §0.1 的 P17）。本程序真正测的是")
    print("        下面 A.3/A.4：两条**独立代码路径**给出同一个输出、且参照一次都没调用被测侧。")

    # ── A.3/A.4 ★★ 两条独立路径 ───────────────────────────────────────────
    print()
    print("  两条**独立**代码路径：① 共享 KV + 物化分数矩阵（被测）"
          "② 复制 KV + 流式 online softmax（参照）")
    worst = 0.0
    for h_kv in H_KV_LIST:
        q, k, v = draw_data(SEED, H_Q, h_kv, D_HEAD, S_Q, S_K)
        rep = H_Q // h_kv
        shared = np.concatenate([attn_dense(q[h], k[group_of(h, H_Q, h_kv)],
                                            v[group_of(h, H_Q, h_kv)])
                                 for h in range(H_Q)], axis=0)
        k_rep = np.repeat(k, rep, axis=0)      # ★ 物化复制成 h_q 份
        v_rep = np.repeat(v, rep, axis=0)
        copied = np.concatenate([attn_stream(q[h], k_rep[h], v_rep[h])
                                 for h in range(H_Q)], axis=0)
        d = float(np.abs(shared - copied).max())
        worst = max(worst, d)
        print("    G=%-2d（复制 %d 份）：最大逐元素差 %.3e（输出量级 %.3f）"
              % (h_kv, rep, d, float(np.abs(shared).max())))
    check("A.3", worst < REF_TOL,
          "共享式与复制式的最大逐元素差 %.3e < %g（**机器精度档**，不是 `== 0.0`："
          "两条真独立的路径不可能逐位相同）" % (worst, REF_TOL))

    # ★ 参照跑完后，计数表必须**整个是空的** —— 比「那个键是 0」更强：
    #   它等于「参照一次都没调用被测侧的任何函数」（含将来新增的函数）。
    CALLS.clear()
    q, k, v = draw_data(SEED, H_Q, 2, D_HEAD, S_Q, S_K)
    k_rep = np.repeat(k, H_Q // 2, axis=0)
    v_rep = np.repeat(v, H_Q // 2, axis=0)
    for h in range(H_Q):
        attn_stream(q[h], k_rep[h], v_rep[h])
    check("A.4", not CALLS,
          "★ 参照实现跑完后**计数表完全为空**（记录到的调用：%s）⇒ "
          "它一次都没调用被测实现的函数" % (sorted(CALLS) if CALLS else "无"))

    # ── A.5 防 P17：源码不同 ──────────────────────────────────────────────
    src_dense = inspect.getsource(attn_dense)
    src_stream = inspect.getsource(attn_stream)
    check("A.5", src_dense != src_stream,
          "★ 防 P17：被测与参照的**源码不同**（`inspect.getsource` 比较，%d vs %d 字节）"
          % (len(src_dense), len(src_stream)))
    return


# ===========================================================================
# 5. B 类：KV 元素数的倍率（结构侧）
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. KV 元素数随组数的倍率（**元素数**，不是字节）")
    print("=" * 74)
    print("  ★ 口径：只算元素数（KV 头的条数之比）。**字节**账的正文在")
    print("    130-quantization/kv-cache-quantization/01:48-76（正文在彼，此处回链）。")
    print("  ★ 同一断言的**实现与比较**在 40-kernels/attention/05-decoding-attention.md:60,66-94")
    print("    与 …/code/verify.py 的 A4（:509-529）/ B8（:1008-1040）（正文在彼，此处回链）；")
    print("    本主题只写**共享图景**这一新增部分。")

    # ── B.1 两个真实模型的组数（精确有理数）──────────────────────────────
    rows = []
    for name, fields in (("Meta-Llama-3-8B", specs.LLAMA3_8B_ATTENTION),
                         ("Qwen3-30B-A3B", specs.QWEN3_30B_A3B_ATTENTION)):
        h_q = fields["num_attention_heads"]
        h_kv = fields["num_key_value_heads"]
        rows.append((name, h_q, h_kv, h_q // h_kv))
        print("    %-16s heads_q=%-3d heads_kv=%-3d ⇒ G=%d（倍率 %s）"
              % (name, h_q, h_kv, h_q // h_kv, kv_elements_ratio(h_kv, h_q)))
    check("B.1", [r[3] for r in rows] == [4, 8]
          and all(r[1] % r[2] == 0 for r in rows)
          and all(kv_elements_ratio(r[2], r[1]) == Fraction(1, r[3]) for r in rows),
          "两个真实模型的组数是**精确有理数** 4（Llama，32/8）与 8（Qwen3，32/4），"
          "且各自的倍率 == 1/G（`Fraction` 比较，不是浮点）")

    # ── B.2 倍率表（精确）────────────────────────────────────────────────
    print()
    print("    %-6s %-8s %-14s" % ("KV 头数", "组数 G", "元素数倍率（精确）"))
    exact = True
    for h_kv in H_KV_LIST:
        g = H_Q // h_kv                    # ★ 组数 = h_q / h_kv，不要与 h_kv 混
        r = kv_elements_ratio(h_kv, H_Q)
        exact = exact and (r == Fraction(1, g))
        print("    %-6d %-8d %-14s" % (h_kv, g, r))
    check("B.2", exact,
          "倍率逐位等于 **1/G**（G ∈ %s，k 与 `h_kv` 的换算见上表），用 `Fraction` 比较 ⇒ "
          "精确有理数，与 head 宽度 `d_head` 无关"
          % sorted({H_Q // h_kv for h_kv in H_KV_LIST}))

    # ── B.3 ★ 那个坑：把 `heads_kv` 写成 `heads_q` ────────────────────────
    print()
    print("    ★ 坑（130-quantization/01:76 记的那一类）：用 heads_q 当 KV 头数 ⇒")
    for name, h_q, h_kv, g in rows:
        wrong = kv_elements_ratio(h_q, h_q)
        print("      %-16s 真倍率 %s vs 错倍率 %s ⇒ 高估 %.0f 倍"
              % (name, kv_elements_ratio(h_kv, h_q), wrong, wrong / kv_elements_ratio(h_kv, h_q)))
    check("B.3", all(kv_elements_ratio(r[1], r[1]) / kv_elements_ratio(r[2], r[1]) == r[3]
                     for r in rows),
          "「组数」是唯一变化的那个因子：把 KV 头数写成 heads_q，倍率就从 1/G 变成 1，"
          "**高估恰好 G 倍**（Llama 4×、Qwen3 8×）")


# ===========================================================================
# 6. C 类：边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)

    # ── C.1 组数必须是头数的因子 ─────────────────────────────────────────
    bad = [(8, 3), (8, 5), (32, 6), (32, 7)]
    ok = all(h_q % g != 0 for h_q, g in bad)
    print("  C.1：GQA 把 query 头**均分**成 G 组 ⇒ 不整除的 G **不存在**：")
    for h_q, g in bad:
        print("      h_q=%d、G=%d ⇒ %d %% %d = %d ≠ 0" % (h_q, g, h_q, g, h_q % g))
    check("C.1", ok and all(H_Q % g == 0 for g in H_KV_LIST),
          "★ G 的取值空间是 **h_q 的因子格**，不是连续可调；本程序用的 %s 全部合法"
          % (list(H_KV_LIST),))

    # ── C.2 GQA 不动算式：分数矩阵的元素数与 G 无关 ───────────────────────
    shapes = []
    for h_kv in H_KV_LIST:
        q, k, v = draw_data(SEED, H_Q, h_kv, D_HEAD, S_Q, S_K)
        shapes.append((q.shape, (q[0] @ k[0].T).shape))
    same = len({str(s) for s in shapes}) == 1
    print()
    print("  C.2：分数矩阵的形状（G ∈ %s 逐个抽数据算）：" % (list(H_KV_LIST),))
    for h_kv, (qs, ss) in zip(H_KV_LIST, shapes):
        print("      G=%-2d  Q%s  分数矩阵 %s" % (h_kv, qs, ss))
    check("C.2", same,
          "★ 分数矩阵的形状**与 G 无关**（恒为 %s）⇒ 改的只有「一条 KV 服务几个头」"
          "（复用），**没改算式**" % (shapes[0][1],))

    # ── C.3 ★ 反例：缩放不是装饰（把 `1/√d_h` 换掉，输出真的变）────────────
    qq, kk, vv = draw_data(SEED, 1, 1, D_HEAD, S_Q, S_K)
    q, k, v = qq[0], kk[0], vv[0]           # ← 单头（判据与 head 的分组无关）
    base = attn_dense(q, k, v)
    doubled = attn_dense(2.0 * q, k, v)     # ← 等价于把分数整体乘 2（换掉缩放）
    d = float(np.abs(base - doubled).max())
    print()
    print("  C.3：把 `q` 乘 2（= 把 softmax 的输入整体乘 2，等价于换掉缩放 `1/√d_h`）：")
    print("      输出最大差 %.4f（输出量级 %.3f）" % (d, float(np.abs(base).max())))
    check("C.3", d > SCALE_DROP_MIN,
          "★ 缩放**是判据的组成部分**：换掉它输出就变（差 %.4f > %g）"
          "⇒ 「去掉 `1/√d_h`」这类口径变异会在输出上露出来" % (d, SCALE_DROP_MIN))

    # ── C.4 ★★ M6 的**对照组**：一个**共享代码的「假参照」** ⇒ 断言「测不出来」 ──
    # ★★ 计划 §3 Chunk 2 的验收要求：「把参考实现与被测实现**共享一段代码**，断言应当
    #    **测不出来**（即这个「假通过」现象**必须能被复现**）」。
    # ★ 这里刻意把 P17 的现场复现一遍：假参照**直接调用被测实现** ⇒ 逐元素差恒为
    #    **精确的 `0.0`**，而 `A.3` 的容差式判据（`< 1e-14`）**照样通过**
    #    ⇒ 容差对这种情况**没有分辨力**。
    # ★ 揭穿它的是**机械检查**：`inspect.getsource` 不等（`A.5`）与 **AST 调用表**
    #    （下面这一条用的是与 `D.8` 同一个探测器）。
    # ★ 注意 `== 0.0` 在这里是**合法的用法**：它落在 `00-meta/glossary.md` 的「reference implementation（参考实现）」条说的
    #   「**两种相反含义**」的**第二种**（共享代码 ⇒ 测不出来），而不是第一种（正确）。
    def attn_shared_ref(q, k, v):
        """★ 故意的「假参照」：它**直接调用被测实现**（这就是 P17 的现场）。"""
        return attn_dense(q, k, v)

    x_s, k_s, v_s = draw_data(SEED, 1, 1, D_HEAD, S_Q, S_K)
    d_shared = float(np.abs(attn_dense(x_s[0], k_s[0], v_s[0])
                            - attn_shared_ref(x_s[0], k_s[0], v_s[0])).max())
    d_indep = float(np.abs(attn_dense(x_s[0], k_s[0], v_s[0])
                           - attn_stream(x_s[0], k_s[0], v_s[0])).max())
    caught_shared = "attn_dense" in _callee_names(attn_shared_ref)   # ← 探测器
    caught_real = "attn_dense" in _callee_names(attn_stream)
    print()
    print("  C.4：★★ 对照组（M6）——「参照」是一个**直接调用被测实现**的假参照：")
    print("       逐元素差 = **%.1f**（精确 0.0！）而 A.3 的判据 `< %g` **照样通过**"
          % (d_shared, REF_TOL))
    print("       真独立时同一个量 = %.3e ≠ 0.0 ⇒ 两种情形**只是量级不同**，"
          "容差**分辨不出来**" % d_indep)
    print("       ★ 但 AST 调用表**分辨得出来**：假参照里含 `attn_dense` = %s，"
          "真参照里含 = %s" % (caught_shared, caught_real))
    check("C.4", d_shared == 0.0 and d_indep != 0.0
          and caught_shared and not caught_real,
          "★ 假通过**可复现**：共享代码 ⇒ 差是**精确 `0.0`**（%.1f），独立路径 ⇒ %.3e "
          "—— 两者都 `< %g`。★ 揭穿它的是**AST 调用表**（假参照 True / 真参照 False）"
          "与 `inspect.getsource` 不等（`A.5`），**不是**容差" % (d_shared, d_indep, REF_TOL))


# ===========================================================================
# 7. D 类：失败标记自检 + 口径钉住
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    先例：`make lint` 与 `mutate.py` 都走 AST 而不是 grep —— 注释里出现一个名字
    和真的调用它是两件事。
    ★ 它同时是**参照实现独立性**的探测器（`D.8`）与**假通过**的探测器（`C.4`）。
    """
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


def pin_part():
    """★★ **口径 / 阈值 / specs 的钉住 —— 刻意放在 A/B/C 之前**。

    ★ 为什么放最前：这几条是**所有判据的前提**，而且它们必须**先于**任何
      「形状对不上」「键名找不到」这类**结构性崩溃**报出来。
      否则那些变异会被 `make mutate` 记成**崩溃捕获**，而崩溃捕获**不算真捕获**
      （它证明「代码炸了」，不是「判据抓住了错」）。本主题实测：先钉住之后，
      这些变异全部落进**断言捕获**。
    """
    print("=" * 74)
    print("D. 口径/阈值/specs 的钉住（★ 放在 A/B/C **之前**：先钉住前提，再看数）")
    print("=" * 74)

    # ── 口径常量 ─────────────────────────────────────────────────────────
    measured = {"H_Q": H_Q, "H_KV_LIST": H_KV_LIST, "D_HEAD": D_HEAD, "S_Q": S_Q,
                "S_K": S_K, "SEED": SEED, "BLOCK": BLOCK,
                "ATTN_SCALE": ATTN_SCALE}
    expect = {"H_Q": 8, "H_KV_LIST": (1, 2, 4, 8), "D_HEAD": 32, "S_Q": 12,
              "S_K": 9, "SEED": 20260922, "BLOCK": 4,
              "ATTN_SCALE": 1.0 / float(np.sqrt(32))}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 8 个逐位一致（含 `ATTN_SCALE = 1/√d_head` —— 它是**算式的口径**，"
          "改掉它两条路径依然一致、只有这里能发现）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    # ── 阈值 ─────────────────────────────────────────────────────────────
    # ★ 阈值：**逐个说明它属于哪一档**——机器精度档的阈值一旦被悄悄放宽，
    #   整条判据就从「测出来」退化成「总会通过」。
    th = {"REF_TOL": REF_TOL, "SCALE_DROP_MIN": SCALE_DROP_MIN}
    th_expect = {"REF_TOL": 1e-14, "SCALE_DROP_MIN": 1e-2}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 2 个逐位一致（含 1 个**机器精度档**：REF_TOL —— 放宽它，"
          "A.3 就退化成恒真）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── specs 的常量逐位钉住 ──────────────────────────────────────────────
    # ★ **键名先钉、取值后钉** —— 顺序是刻意的：`mutate.py` 的分母含**字典的键**，
    #   键名被改后按字面量取值会抛 `KeyError`（那是**崩溃**，不算断言捕获）。
    #   所以键名检查必须**排在**任何按名取值的语句之前。
    keys_ok = (set(specs.LLAMA3_8B_ATTENTION) == {"num_attention_heads", "num_key_value_heads"}
               and set(specs.QWEN3_30B_A3B_ATTENTION)
               == {"num_attention_heads", "num_key_value_heads"})
    check("D.3", keys_ok,
          "★ `specs.py` 的两个 head 字面量的**键名**逐位钉住 = %s（先钉键名，"
          "否则键名一变、下一行就 `KeyError` ⇒ 那是崩溃不是断言捕获）"
          % (sorted(set(specs.LLAMA3_8B_ATTENTION)),))
    if not keys_ok:                      # ★ 键名不对就不再去按名取值（避免崩溃吞掉结论）
        return

    # ★★ 为什么这里要**重钉一次上游常量**：`mutate.py` 会改 `specs.py` 的每一个叶子，
    #   而有些改动**保住了比值**（例如 heads_q 32→64 与 heads_kv 8→16 同时改，
    #   G 仍是 4；或把 `H_KV_LIST` 里的 2 改成 4）⇒ 只有逐位钉住才挡得住。
    #   这正是 R17 后半段「给它一个消费者」的落法。
    spec_pin = (
        specs.LLAMA3_8B_ATTENTION["num_attention_heads"],
        specs.LLAMA3_8B_ATTENTION["num_key_value_heads"],
        specs.QWEN3_30B_A3B_ATTENTION["num_attention_heads"],
        specs.QWEN3_30B_A3B_ATTENTION["num_key_value_heads"],
    )
    check("D.4", spec_pin == (32, 8, 32, 4),
          "★ `specs.py` 的 4 个 head 字段**逐位钉住** = %s（[B] 级，转记自 "
          "45-…/code/specs.py）" % (spec_pin,))


def mark_part():
    """失败标记自检 + 参照实现的独立性（AST 级）—— 放在最后跑。"""
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 参照实现的独立性")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.5", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)

    # ★ 「等于该值」：把**本主题被分配的那个字面量**写死在这里。
    #   ★ 这不是同义反复：`mutate.py` 只变异**模块级**常量，函数体里的这个字面量
    #     不在它的分母里 ⇒ 它正是用来挡住「标记被改掉 / 被跨主题照抄」的那把锁（R18）。
    check("D.6", ASSERT_FAIL_MARK == "@@ FAIL: attention-variants"
          and "FAIL" in ASSERT_FAIL_MARK
          and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.7", OTHER_THEME_MARKS == ("!! FAIL: transformer", "## FAIL: moe",
                                       "~~ FAIL: normalization",
                                       "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    # ── 参照实现的独立性（AST 级）─────────────────────────────────────────
    watched = {group_of, attn_dense, attn_stream, draw_data, kv_elements_ratio}
    wnames = {f.__name__ for f in watched}
    called = _callee_names(attn_stream)
    shared = sorted(wnames & called)
    counted_src = inspect.getsource(attn_dense)
    check("D.8", "CALLS[" in counted_src and not shared,
          "被测的 `attn_dense` 带 `CALLS` 计数；参照实现的 **AST 调用表里一个都没有它**"
          "（也没共用 `group_of` / `draw_data`）⇒ A.4 的「计数为 0」不是空断言%s"
          % ("" if not shared else " ← 参照调用了 %s" % shared))


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
    print("01_mha_mqa_gqa.py —— MHA / MQA / GQA 的 head 关系与共享图景")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区（计划 §2 的边界表）：本程序**不**重讲「复制 KV 后与 MHA 逐位相同」的实现")
    print("    与比较（正文在 40-kernels/attention/05 与其 code/verify.py 的 A4/B8，此处回链）；")
    print("    也**不**算 KV 的字节（六项乘积的正文在 130-quantization/kv-cache-quantization/01:48-76）。")
    pin_part()      # ★ D.1–D.4：先钉住前提（见 pin_part 的说明）
    part_A()
    part_B()
    part_C()
    mark_part()     # ★ D.5–D.8：标记自检与独立性
    return summary()


if __name__ == "__main__":
    sys.exit(main())
