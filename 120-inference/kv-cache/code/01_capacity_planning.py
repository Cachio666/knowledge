#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_capacity_planning.py —— **从「账」到「容量」**：可用显存 → KV 块数 → 最大并发。

配套文档：`01-capacity-planning.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题不写「账」**（那是 `130-quantization/kv-cache-quantization/01` 的正文，本主题**只回链**）——
  本程序把「每 token 字节数」当**输入**（从 `specs` 取用，**不重推**六项乘积，`P37`），
  只算 130-01 **没算的那条链**：**可用显存 → KV 块数 → 最大并发 / 最大 batch**。

★★★ **本程序的 A 类有【两条独立路径】**（照 `§1.3.2b` / `CONTRIBUTING:235`）：
  · **被测** = 闭式除法 `池块数 = 可用显存 // (块大小 × 每 token 字节)`，再 `//` 每序列块数；
  · **参照** = **逐块填池 + 逐 token 取块**的**模拟分配**（`sim_*`）：
      `sim_fill_pool` 一块一块往池里放，放到下一块放不下为止；
      `sim_admit_all` **逐条序列、逐 token** 地从池里取块，取不到就停。
    ★★ 参照的**控制流由数据驱动**（`while` 试放一块 / 逐 token 数满块），
      **既没有除法、也不调用任何 `cf_*`**（`D.8` 在源码层核）。

★★ **覆盖边界**：本程序守的是「**容量规划的算术**」（块数 / 并发 / 极值退化）与
  「**两条独立路径会不会会合**」。**不守**任何真实引擎的并发数（本机**无 GPU / 无引擎**）。
  ★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **闭式 == 模拟**（`KV_AVAIL_GIB_GRID` × `KV_BLOCK_TOKEN_GRID` × `KV_SEQ_LEN_GRID`）；
  · `A.2` ★★ 两条路径在**池块数**上也一致（`sim_fill_pool` vs 闭式除法）；
  · `A.3` ★★ **免费守卫**：预算↑ ⇒ 并发不降；序列长度↑ ⇒ 并发不增；预算 `0` ⇒ 并发 `0`；
  · `B.1/B.2/B.3` ★ 敏感度表（Llama-2-7B 的预算 × 序列长度；换模型差 4 倍）；
  · `C.1` ★★★ **极小块的退化**：需块数越过块表上界 ⇒ 计划**不可实现**（而**并发数看不出来**）；
  · `C.2` ★★★ **极大块的退化**：一个块就装不下 ⇒ 池空 ⇒ 并发 **`0`**；
  · `D.4` ★★★ **守卫实验**：把参照的分配规则改坏 ⇒ 闭式与模拟**当场对不上**；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **失败标记自检** + `UNSOURCED`。
"""
import ast
import collections
import inspect
import sys
import textwrap

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（kv-cache）取 `-- FAIL: kv-cache`
#   （★ 三个程序**同一个**标记；本域另外四个主题各有自己的，**两两不互为前缀**）。
#   ★★ `--` 在 shell 里是**选项前缀** ⇒ 它**只允许出现在 `print()` 的字符串里**，
#     **不得**出现在 Makefile 的 recipe 行或任何命令行里。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = "-- FAIL: kv-cache"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序**不打印**它们。
OTHER_THEME_MARKS = ("-- FAIL: batching-and-scheduling", "-- FAIL: decoding-strategies",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

GIB = 1073741824          # 2^30（十进制字面量，绝不写成幂）


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
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
# 1. 被测：**闭式除法**（**只有这些**；它们**不得**出现在参照里 —— `D.8` 核）
# ===========================================================================
def cf_pool_blocks(avail_bytes, block_tokens, per_token_bytes):
    """**闭式除法**：一段可用显存能装下多少个 KV 块（**被测路径之一**）。

    ★ 它是本主题最直白的那条算式：`池块数 = avail // (块大小 × 每 token 字节)`。
    """
    return avail_bytes // (block_tokens * per_token_bytes)


def cf_blocks_per_seq(seq_len, block_tokens):
    """一条长度为 `seq_len` 的序列占多少个 KV 块（**向上取整**，末尾不满一块也算一块）。"""
    return (seq_len + block_tokens - 1) // block_tokens


def cf_max_concurrency(avail_bytes, block_tokens, per_token_bytes, seq_len):
    """**最大并发**（闭式）：`池块数 // 每序列块数`。"""
    return cf_pool_blocks(avail_bytes, block_tokens, per_token_bytes) \
        // cf_blocks_per_seq(seq_len, block_tokens)


# ===========================================================================
# 2. 参照：**逐块填池 + 逐 token 取块**（★ **不调用任何 `cf_*`**，`D.8` 核）
# ===========================================================================
def sim_fill_pool(avail_bytes, block_tokens, per_token_bytes):
    """★★★ **逐块填池**：一块一块往池里放，直到**下一块放不下**为止，返回池里的块数。

    ★★ 循环是 `while used + 一块字节 <= avail` —— **拍数由数据驱动**，**没有除法**。
    ★★ `corrupt="overfill"` 只用于 `D.4` 的守卫实验（多塞一块，模拟「把分配规则改坏」）。
    """
    one_block = block_tokens * per_token_bytes
    n = 0
    used = 0
    while used + one_block <= avail_bytes:
        used += one_block
        n += 1
    return n


def sim_admit_all(avail_bytes, block_tokens, per_token_bytes, seq_len, corrupt=None):
    """★★★ **逐条序列、逐 token 地从池里取块**，返回**被接受的序列数**（**参照**）。

    ★★ 分配规则（逐条来自题设，**不是**从闭式反推的）：
      · 一条序列每写满 `block_tokens` 个 token 就**取走一个块**；
      · 池里**没有空块**时，这一条**取不到** ⇒ 停（它不占已取的块，全部**退回**）。
    ★★ `corrupt="drop_partial"`（守卫实验用）：**末尾不满的一块不取**（把分配规则改坏）——
      于是每序列块数从 `ceil(S/B)` 掉到 `floor(S/B)`，**闭式与模拟当场对不上**。
    ★★★ 它**不与 `cf_*` 共享任何一行**：这里数的是**取走的块数**，不是算出来的商。
    """
    free = sim_fill_pool(avail_bytes, block_tokens, per_token_bytes)
    admitted = 0
    while True:
        taken = 0
        ok = True
        for t in range(seq_len):
            if t % block_tokens == 0:
                if corrupt == "drop_partial" and t + block_tokens > seq_len:
                    break                # ★ 坏法：末尾不满的一块不取
                if free == 0:
                    ok = False
                    break
                free -= 1
                taken += 1
        if not ok:
            free += taken            # ★ 退回这一条已取的块
            break
        admitted += 1
    return admitted


def both_paths(avail_gib, block_tokens, per_token_bytes, seq_len):
    """把两条路径的**并发数**一起返回：`(闭式, 模拟)`。★ 只在这里并置，便于对拍。"""
    avail = avail_gib * GIB
    return (cf_max_concurrency(avail, block_tokens, per_token_bytes, seq_len),
            sim_admit_all(avail, block_tokens, per_token_bytes, seq_len))


# ===========================================================================
# 3. A 类 —— ★★★ **两条独立路径会不会会合**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **容量规划的算术**：闭式除法（代数）vs 模拟分配（逐块 / 逐 token）")
    print("=" * 78)
    print("  ★★★ 参照 = `sim_fill_pool` + `sim_admit_all`（**调度式的分配**，拍数由数据驱动）；")
    print("      被测 = `cf_pool_blocks` / `cf_blocks_per_seq`（**代数**）。")
    print("  ★★ 两条**独立路径**；★★ 参照**不调用**任何 `cf_*`（`D.8` 在源码层核）。")

    # ── A.1 ★★★ 闭式 == 模拟（网格）─────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **闭式 == 模拟**（三网格的笛卡尔积）──")
    print("     %-8s %-8s %-10s %-10s %-10s %-6s"
          % ("预算GiB", "块tok", "序列长", "闭式并发", "模拟并发", "一致"))
    ok1 = True
    n_rows = 0
    a_seq = (512, 4096)          # ★ 对拍只用两档序列长（模拟是逐 token 的，取小网格保证快）
    for avail_gib in specs.KV_AVAIL_GIB_GRID:
        for bt in specs.KV_BLOCK_TOKEN_GRID:
            for sl in a_seq:
                cf, sm = both_paths(avail_gib, bt, specs.LLAMA2_7B_KV_BYTES_PER_TOKEN, sl)
                ok1 = ok1 and (cf == sm)
                n_rows += 1
                print("     %-8d %-8d %-10d %-10d %-10d %-6s"
                      % (avail_gib, bt, sl, cf, sm, cf == sm))
    print("     ★★★ **%d 档全部一致**（★ 闭式是**代数**，模拟是**逐块取走** —— 两条独立路径）。"
          % n_rows)
    check("A.1", ok1 and n_rows == 24
          and specs.KV_AVAIL_GIB_GRID == (7, 23, 63)
          and specs.KV_BLOCK_TOKEN_GRID == (1, 16, 64, 256)
          and specs.KV_SEQ_LEN_GRID == (512, 1024, 2048, 4096),
          "★★★ **判据成立**：**%d 档**上「闭式除法给出的最大并发」**逐位等于**"
          "「模拟分配真的塞进去的序列数」（★ 一面是除法、一面是逐块取走 —— 两条独立路径会合）"
          % n_rows,
          "**闭式与模拟对不上** —— ★ 先查分配规则（末尾不满一块是否也占一块），再看闭式取整")

    # ── A.2 ★★ 池块数：逐块填池 vs 闭式除法 ──────────────────────────────
    print()
    print("  ── A.2 ★★ **池块数**：`sim_fill_pool`（逐块放）vs 闭式除法 ──")
    ok2 = True
    for avail_gib in specs.KV_AVAIL_GIB_GRID:
        for bt in specs.KV_BLOCK_TOKEN_GRID:
            avail = avail_gib * GIB
            got = sim_fill_pool(avail, bt, specs.LLAMA2_7B_KV_BYTES_PER_TOKEN)
            exp = cf_pool_blocks(avail, bt, specs.LLAMA2_7B_KV_BYTES_PER_TOKEN)
            ok2 = ok2 and (got == exp)
            print("     %-8d %-8d 逐块填池 %-10d 闭式除法 %-10d 一致=%s"
                  % (avail_gib, bt, got, exp, got == exp))
    check("A.2", ok2,
          "★★ **判据成立**：**逐块填池得到的块数**（一个 `while`，每转一圈放一块）"
          "**逐位等于**闭式除法 `avail // (块 × 每token字节)` —— ★ 这是**同一条路径以外的第二条**",
          "**填池与除法对不上** —— ★ 先查 `<=` 还是 `<`（边界那一块）")

    # ── A.3 ★★ 免费守卫（单调性 + 零预算）────────────────────────────────
    print()
    print("  ── A.3 ★★ **免费守卫**：预算↑ ⇒ 并发不降；长度↑ ⇒ 并发不增；预算 0 ⇒ 并发 0 ──")
    BT = specs.VLLM_DEFAULT_BLOCK_SIZE
    t = specs.LLAMA2_7B_KV_BYTES_PER_TOKEN
    by_budget = [cf_max_concurrency(g * GIB, BT, t, 4096) for g in specs.KV_AVAIL_GIB_GRID]
    by_len = [cf_max_concurrency(63 * GIB, BT, t, sl) for sl in specs.KV_SEQ_LEN_GRID]
    zero = cf_max_concurrency(0, BT, t, 4096)
    mono_b = all(by_budget[i] <= by_budget[i + 1] for i in range(len(by_budget) - 1))
    mono_l = all(by_len[i] >= by_len[i + 1] for i in range(len(by_len) - 1))
    print("     预算 %s GiB ⇒ 并发 %s（单调不降 = %s）"
          % (list(specs.KV_AVAIL_GIB_GRID), by_budget, mono_b))
    print("     序列长 %s ⇒ 并发 %s（单调不增 = %s）"
          % (list(specs.KV_SEQ_LEN_GRID), by_len, mono_l))
    print("     ★★ 预算 `0` ⇒ 并发 **%d**（池里一块都放不下）" % zero)
    check("A.3", mono_b and mono_l and zero == 0
          and by_budget == [3, 11, 31] and by_len == [252, 126, 63, 31],
          "★★ **判据成立**：三条免费守卫都过 —— ★ **预算↑ ⇒ 并发不降**（%s）；"
          "★ **序列长度↑ ⇒ 并发不增**（%s）；★ **预算 `0` ⇒ 并发 `0`**（无块可用，不是负数、不是幻觉）"
          % (by_budget, by_len),
          "**某条守卫被破坏** —— ★ 先查取整方向（`//` 的方向决定单调性）")


# ===========================================================================
# 4. B 类：敏感度表（给定模型与预算 → 最大并发）
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **敏感度**：同一个模型、不同预算与序列长度下的最大并发")
    print("=" * 78)
    print("  ★★★ **账的部分不在这里** ——「每 token %d B = %s KiB」取自 %s"
          % (specs.LLAMA2_7B_KV_BYTES_PER_TOKEN,
             specs.LLAMA2_7B_KV_BYTES_PER_TOKEN // 1024,
             "`130-quantization/kv-cache-quantization/01`（本主题只取用，不重推）"))

    # ── B.1 ★ 预算 × 序列长度 ⇒ 并发（Llama-2-7B）─────────────────────────
    print()
    print("  ── B.1 ★ **预算 × 序列长度 ⇒ 最大并发**（Llama-2-7B，块 = %d token）──"
          % specs.VLLM_DEFAULT_BLOCK_SIZE)
    t = specs.LLAMA2_7B_KV_BYTES_PER_TOKEN
    BT = specs.VLLM_DEFAULT_BLOCK_SIZE
    tbl = {}
    print("     %-10s %-10s %-12s %-12s %-10s"
          % ("总显存GiB", "可用GiB", "池块数", "序列长", "最大并发"))
    for gpu in specs.KV_GPU_MEM_GIB_GRID:
        avail_gib = gpu - specs.LLAMA2_7B_WEIGHTS_GIB - specs.KV_RESERVE_GIB
        blocks = cf_pool_blocks(avail_gib * GIB, BT, t)
        for sl in specs.KV_SEQ_LEN_GRID:
            c = cf_max_concurrency(avail_gib * GIB, BT, t, sl)
            tbl[(gpu, sl)] = c
            print("     %-10d %-10d %-12d %-12d %-10d" % (gpu, avail_gib, blocks, sl, c))
    print("     ★★ **读法**：总显存 %d GiB ⇒ 可用 %d GiB（减权重 %d / 预留 %d）⇒ 池 %d 块。"
          % (specs.KV_GPU_MEM_GIB_GRID[2],
             specs.KV_GPU_MEM_GIB_GRID[2] - specs.LLAMA2_7B_WEIGHTS_GIB - specs.KV_RESERVE_GIB,
             specs.LLAMA2_7B_WEIGHTS_GIB, specs.KV_RESERVE_GIB,
             cf_pool_blocks(63 * GIB, BT, t)))
    check("B.1", tbl[(80, 4096)] == 31 and tbl[(80, 512)] == 252
          and tbl[(40, 4096)] == 11 and tbl[(24, 4096)] == 3
          and tbl[(24, 512)] == 28,
          "★ **判据成立**：80 GiB 上 `S = 512 / 4096` ⇒ 并发 **252 / 31**；"
          "40 GiB、`S = 4096` ⇒ **11**；24 GiB ⇒ **3** —— ★★ 全部由「可用显存 → 池块数 → //每序列块数」这条链给出",
          "**敏感度表的某一格被动过** —— ★ 先查「可用显存」有没有减掉权重与预留")

    # ── B.2 ★ 敏感度：并发 ≈ 与序列长度成反比 ────────────────────────────
    print()
    print("  ── B.2 ★ **敏感度**：并发随序列长度近似按 `1/S` 掉（`80 GiB` 一档）──")
    c512 = tbl[(80, 512)]
    c4096 = tbl[(80, 4096)]
    print("     并发(512) = %d · 并发(4096) = %d · 比值 = %.2f"
          % (c512, c4096, c512 / c4096))
    print("     ★★ **读法**：序列长 8 倍 ⇒ 并发**约**掉 8 倍（`%d ≈ 8 × %d`）——"
          % (c512, c4096))
    print("        ★ 不精确的**唯一原因**是每序列块数的**向上取整**（末尾不满一块也算一块）。")
    check("B.2", c512 >= 8 * c4096 and c4096 == 31
          and tbl[(80, 1024)] == 126 and tbl[(80, 2048)] == 63,
          "★ **判据成立**：`S × 8 ⇒ 并发 ÷ 8` 在**取整误差内**成立"
          "（`%d ≥ 8 × %d`，四个 `S` 给出 `252 / 126 / 63 / 31`）"
          " ★★ 敏感度是**近似线性的**，偏差全部来自末尾不满块的取整",
          "**反比关系被破坏** —— ★ 先查每序列块数是否真的按 `ceil(S/B)` 变")

    # ── B.3 ★ 换模型：Mistral-7B（每 token 128 KiB）⇒ 并发 ×4 ────────────
    print()
    print("  ── B.3 ★ **换模型**：Mistral-7B（%d KiB/token）vs Llama-2-7B（%d KiB/token）──"
          % (specs.LLAMA2_7B_MISTRAL_CONTRAST_KIB, specs.LLAMA2_7B_KV_BYTES_PER_TOKEN // 1024))
    t_mistral = specs.LLAMA2_7B_MISTRAL_CONTRAST_KIB * 1024
    c_llama = tbl[(80, 4096)]
    c_mistral = cf_max_concurrency(63 * GIB, BT, t_mistral, 4096)
    print("     同一 80 GiB、`S = 4096`：Llama-2-7B %d · Mistral-7B %d · 倍数 %.2f"
          % (c_llama, c_mistral, c_mistral / c_llama))
    check("B.3", c_mistral >= 4 * c_llama and c_mistral == 126,
          "★ **判据成立**：每 token 字节掉 4 倍 ⇒ 并发涨约 4 倍（`%d → %d`）"
          " ★★ **容量规划的输入随模型变**，而**块的粒度（%d token）不变**"
          % (c_llama, c_mistral, BT),
          "**换模型后并发没有按字节数反比变** —— ★ 先查每 token 字节数有没有被替换掉")


# ===========================================================================
# 5. C 类：**块大小取极值**时的退化（★ 必须真的跑出问题）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**块大小取极值**（`1` 与极大）时的退化")
    print("=" * 78)
    t = specs.LLAMA2_7B_KV_BYTES_PER_TOKEN
    avail = 63 * GIB
    S = 4096

    # ── C.1 ★★★ 极小块（B=1）：计划**不可实现**（而并发数看不出来）────────
    print()
    print("  ── C.1 ★★★ 块大小 = `1`（每个 token 一个块）──")
    pool_b1 = cf_pool_blocks(avail, 1, t)
    pool_b16 = cf_pool_blocks(avail, specs.VLLM_DEFAULT_BLOCK_SIZE, t)
    conc_b1 = cf_max_concurrency(avail, 1, t, S)
    conc_b16 = cf_max_concurrency(avail, specs.VLLM_DEFAULT_BLOCK_SIZE, t, S)
    print("     块 = `1`  ⇒ 池 %d 块 · 并发 %d" % (pool_b1, conc_b1))
    print("     块 = `%d` ⇒ 池 %d 块 · 并发 %d" % (specs.VLLM_DEFAULT_BLOCK_SIZE, pool_b16, conc_b16))
    print("     ★★★ **问题**：块 = `1` 时池里要装 **%d** 个块，而块表最多能索引 **%d** 个"
          % (pool_b1, specs.KV_BLOCK_TABLE_CAP))
    print("        ⇒ ★★ 这份容量规划**不可实现** —— ★ 而**并发数（%d）与块 = `%d` 时一样**，"
          % (conc_b1, specs.VLLM_DEFAULT_BLOCK_SIZE))
    print("           ★★★ **只看并发数根本看不出来**：代价不在并发上，在**块表的条目数**上。")
    check("C.1", pool_b1 > specs.KV_BLOCK_TABLE_CAP
          and pool_b16 <= specs.KV_BLOCK_TABLE_CAP
          and conc_b1 == conc_b16
          and pool_b1 == 129024 and pool_b16 == 8064,
          "★★★ **判据成立**：块 = `1` 时**池要 %d 块 > 块表上界 %d** ⇒ **计划不可实现**；"
          "★★ **而并发数与块 = `%d` 时**逐位相同（都是 `%d`）⇒ ★★★ **退化只在块表上，不在并发上**"
          % (pool_b1, specs.KV_BLOCK_TABLE_CAP, specs.VLLM_DEFAULT_BLOCK_SIZE, conc_b1),
          "**极小块的退化没跑出来** —— ★ 先查「池块数」是否随块大小真的变了")

    # ── C.2 ★★★ 极大块：一个块就装不下 ⇒ 池空 ⇒ 并发 0 ──────────────────
    print()
    print("  ── C.2 ★★★ 块大小 = `%d`（极大，一个块就超过整个预算）──"
          % specs.KV_EXTREME_BLOCK_TOKENS)
    one = specs.KV_EXTREME_BLOCK_TOKENS * t
    print("     一个块占 %d 字节 = %.1f GiB，而可用显存只有 %d GiB"
          % (one, one / GIB, avail // GIB))
    pool_x = cf_pool_blocks(avail, specs.KV_EXTREME_BLOCK_TOKENS, t)
    conc_x = cf_max_concurrency(avail, specs.KV_EXTREME_BLOCK_TOKENS, t, S)
    sim_x = sim_admit_all(avail, specs.KV_EXTREME_BLOCK_TOKENS, t, S)
    print("     ⇒ 池 **%d** 块 · 并发 **%d**（★ 两条路径一致：模拟给 %d）" % (pool_x, conc_x, sim_x))
    check("C.2", pool_x == 0 and conc_x == 0 and sim_x == 0,
          "★★★ **判据成立**：块大到**一个都放不进池** ⇒ 池 `0` 块 ⇒ 并发 **`0`**"
          " —— ★ 这不是「并发很低」，是**这份规划连一条序列都开不起来**（两条路径一致）",
          "**极大块没有退化成 0** —— ★ 先查「一个块的字节数」是否真的超过了池预算")


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 A/B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无引擎**；")
    print("      ★★ 但**本程序的 A 类是真的**：它做的是**闭式除法 vs 模拟分配**（两条独立路径）。")
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
    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: kv-cache" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: kv-cache`、且含 `FAIL`\n"
          "        ★★ `--` 是 **shell 的选项前缀** ⇒ 它**只允许出现在 `print()` 的字符串里**。")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies",
                                        "-- FAIL: engines", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住；"
          "★ 五个标记**两两不互为前缀**")

    # ── D.3 ★ 载体元数据 ──────────────────────────────────────────────────
    check("D.3", specs.KV_FETCH_DAY == "2026-10-09"
          and specs.VLLM_DEFAULT_BLOCK_SIZE == 16
          and specs.SGLANG_PAGE_TOKENS == 1
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：抓取日 `%s` · vLLM 块 = `%d` · SGLang page = `%d`（★ **两者同名"
          "不同指、差 16 倍**）；★ `UNSOURCED == ()` 一并钉住" % (specs.KV_FETCH_DAY, 16, 1))

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    #   ① 两个网格里的值与「每 token 字节」全 `> 0`（否则会 `ZeroDivisionError`
    #      ⇒ **崩溃捕获**而不是断言捕获）；
    #   ② 块网格里**必须有 `1`**（`C.1` 靠它）；序列长度网格**全 ≥ 块大小**（否则每序列 0 块的假象）。
    vals_ok = (all(v > 0 for v in specs.KV_AVAIL_GIB_GRID)
               and all(v > 0 for v in specs.KV_BLOCK_TOKEN_GRID)
               and all(v > 0 for v in specs.KV_SEQ_LEN_GRID)
               and specs.LLAMA2_7B_KV_BYTES_PER_TOKEN > 0)
    has_b1 = 1 in specs.KV_BLOCK_TOKEN_GRID
    print("     ★★ 网格取值全 `> 0` = %s；块网格含 `1` = %s" % (vals_ok, has_b1))
    check("D.3b", vals_ok and has_b1
          and specs.KV_AVAIL_GIB_GRID[0] == 7 and specs.KV_BLOCK_TOKEN_GRID[0] == 1,
          "★★★ **口径守卫**：网格全为正（★ 否则 `// 0` 会抛 `ZeroDivisionError` ⇒ **崩溃捕获**，"
          "把「断言捕获」的读数污染掉），★ 且块网格**从 `1` 起**（`C.1` 的极值）",
          "**网格或每 token 字节被动过** —— ★ 先查 `KV_*_GRID` 与 `LLAMA2_7B_KV_BYTES_PER_TOKEN`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★★ 守卫实验（`CONTRIBUTING:235`）：把参照的【分配规则】改坏 ────
    print()
    print("  ── D.4 ★★★ **守卫实验**：把参照的分配规则改坏（末尾不满的一块不取）──")
    avail = 7 * GIB
    BTg, Sg = 4096, 5000
    t = specs.LLAMA2_7B_KV_BYTES_PER_TOKEN
    good = sim_admit_all(avail, BTg, t, Sg)
    bad = sim_admit_all(avail, BTg, t, Sg, corrupt="drop_partial")
    closed = cf_max_concurrency(avail, BTg, t, Sg)
    print("     `corrupt=\"drop_partial\"`：每序列只取 `floor(S/B)` 块（★ 坏法）")
    print("     预算 7 GiB、块 %d、序列长 %d：正确模拟 %d · 坏法模拟 %d · 闭式 %d ⇒ **变了 = %s**"
          % (BTg, Sg, good, bad, closed, good != bad))
    print("     ★★★ **读法**：坏法把每序列块数从 `ceil(%d/%d) = %d` 掉到 `floor = %d` ⇒"
          % (Sg, BTg, cf_blocks_per_seq(Sg, BTg), Sg // BTg))
    print("        **池里多塞得下几条** ⇒ 与闭式当场对不上。")
    print("     ★★ **它归 D 类而不是 A 类**：它是**实验守卫**（改坏参照，看断言会不会失败），不测任何实现。")
    check("D.4", (good != bad) and good == closed and bad > good,
          "★★★ **守卫成立**：把参照的分配规则改坏后，模拟**从 %d 变成 %d**（★ 而正确的模拟 == 闭式 %d）"
          "⇒ ★★ **`A.1` 的等式对「末尾不满一块也占一块」这条规则是敏感的**"
          "（★ 若它不敏感，`A.1` 就没有信息量）\n        ★★ **它归 D 类而不是 A 类**：它是**实验守卫**。"
          % (good, bad, closed),
          "**守卫失效** —— ★ 改坏分配规则后模拟结果居然不变（那 `A.1` 对取整规则不敏感 ⇒ 需要加判据）")

    # ── D.5 ★★ 引文逐字（★ 块粒度的两处外部依据）────────────────────────
    check("D.5", specs.VLLM_BLOCK_SIZE_SOURCE_QUOTE == "DEFAULT_BLOCK_SIZE: ClassVar[int] = 16"
          and specs.VLLM_BLOCK_SIZE_DOC_QUOTE
          == 'Size of a contiguous cache block in number of tokens. Accepts None (meaning '
             '"use default"). After construction, always int.'
          and specs.SGLANG_PAGE_LAYOUT_QUOTE
          == "These KV cache tensors are stored on the GPU in a paged layout, where the size of "
             "each page is equivalent to one token.",
          "★★ **两条引文逐字钉住**（vLLM 源码常量一行 + vLLM 文档一句 + SGLang 博客一句）——"
          " ★★★ **特别注意**：vLLM 的 `16` 来自**源码常量**（文档只写 `Accepts None (meaning "
          "\"use default\")`，**不给数字**）；★★ SGLang 的 `page = 1 token` 与它是**同名不同指**；\n"
          "        ★★ 载体 = 源码 `vllm/config/cache.py` + 文档 + LMSYS 博客 `.md`，抓取日 %s"
          % specs.KV_FETCH_DAY,
          "**引文常量被动过** —— ★ 先查内层引号 `\"use default\"` 与 `page is equivalent to one token`")

    # ── D.6 ★★★ 消歧记录 + 级别 ──────────────────────────────────────────
    check("D.6", specs.PAGE_BLOCK_DISAMBIGUATION
          == "block / page 是同名不同指：vLLM 的 KV 块 = 16 token，SGLang 的 page = 1 token，"
             "两者差 16 倍，不可互相套用",
          "★★★ **消歧记录逐字钉住**：★ 它是 **`[E]` 级的本库记录**（**不是任何来源里的话**）——"
          "把 vLLM 的 `block` 与 SGLang 的 `page` 当成同一个粒度会**写错正文**；"
          "★ 它**没有外部出处** ⇒ **不得写成引文**（本库纪律：**自造真值**是一种错误模式）",
          "**消歧记录被动过** —— ★ 先查「16 倍」与「同名不同指」两个词")

    # ── D.7 ★★★ 逐元素钉住 + `UNSOURCED` ──────────────────────────────────
    check("D.7", specs.KV_AVAIL_GIB_GRID == (7, 23, 63)
          and specs.KV_SEQ_LEN_GRID == (512, 1024, 2048, 4096)
          and specs.KV_BLOCK_TOKEN_GRID == (1, 16, 64, 256)
          and specs.KV_GPU_MEM_GIB_GRID == (24, 40, 80)
          and specs.KV_EXTREME_BLOCK_TOKENS == 1048576
          and specs.KV_BLOCK_TABLE_CAP == 65536
          and specs.KV_RESERVE_GIB == 4
          and specs.LLAMA2_7B_KV_BYTES_PER_TOKEN == 524288
          and specs.LLAMA2_7B_WEIGHTS_GIB == 13
          and specs.LLAMA2_7B_MISTRAL_CONTRAST_KIB == 128
          and specs.KV_FETCH_DAY == "2026-10-09"
          and GIB == 1073741824
          and specs.UNSOURCED == (),
          "★★★ **五个元组逐元素钉住**（`KV_AVAIL_GIB_GRID` 3 · `KV_SEQ_LEN_GRID` 4 · "
          "`KV_BLOCK_TOKEN_GRID` 4 · `KV_GPU_MEM_GIB_GRID` 3）+ **六个标量**"
          "（极值块 · 块表上界 · 预留 · 每 token 字节 · 权重 · 对照模型）钉住；"
          "★★ `UNSOURCED == ()` 一并钉住（★ 0 个叶子 ⇒ 不产生变异体）",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★★ 独立性（源码级）── ★★ 遍历【全部】参照实现 ───────────────
    refs = (sim_fill_pool, sim_admit_all)
    cfs = (cf_pool_blocks, cf_blocks_per_seq, cf_max_concurrency)
    call_tables = {}
    for f in refs:
        srcf = textwrap.dedent(inspect.getsource(f))
        call_tables[f.__name__] = {n.func.id for n in ast.walk(ast.parse(srcf))
                                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    cf_calls = set()
    for f in cfs:
        srcf = textwrap.dedent(inspect.getsource(f))
        cf_calls |= {n.func.id for n in ast.walk(ast.parse(srcf))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：**参照**侧两个函数的调用表 ──")
    for nm, tbl in call_tables.items():
        print("     `%s()` 里调用的函数：%s" % (nm, ", ".join(sorted(tbl)) if tbl else "**空集**"))
    print("     **被测**侧三个 `cf_*` 里调用的函数：%s"
          % (", ".join(sorted(cf_calls)) if cf_calls else "**空集**"))
    hits = sorted({c for tbl in call_tables.values() for c in tbl if c.startswith("cf_")})
    print("     ⇒ ★★ **参照侧不调用任何 `cf_*`**；★ `sim_admit_all` 调用 `sim_fill_pool`（**参照内部**）")
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4` 的**行为**兜底。")
    check("D.8", not hits and not any(c.startswith("sim_") for c in cf_calls),
          "★★ **源码级线索成立**：**参照侧的调用表里一个 `cf_*` 都没有，`cf_*` 也不调用参照**"
          "（★ **两侧都审**）—— ★★ 于是「闭式 == 模拟」**不是恒真**，而是**两条独立路径的会合**",
          "**参照与闭式互相调用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 闭式的边界 + 两处「同值不同写」──────────────────────────
    t = specs.LLAMA2_7B_KV_BYTES_PER_TOKEN
    check("D.9", cf_pool_blocks(GIB, 16, t) == 128
          and cf_blocks_per_seq(1, 16) == 1
          and cf_blocks_per_seq(16, 16) == 1
          and cf_blocks_per_seq(17, 16) == 2
          and cf_max_concurrency(0, 16, t, 16) == 0,
          "★★ **闭式的边界**：`1 GiB / (16 × 512 KiB) = 128` 块（整）；"
          "★ 每序列块数在 `S = 16` 与 `17` 之间**从 1 跳到 2**（★ 末尾不满一块也占一块）；"
          "★ 预算 `0` ⇒ 并发 `0`（**不是负数**）",
          "**闭式的边界值被动过**")


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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：**闭式除法** vs **模拟分配**（逐块填池 / 逐 token 取块）"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：本主题的模型是 **`[E]` 级（本库推导 / 假设）**；「每 token 字节」取自 "
          "`130-quantization/kv-cache-quantization/01`（**只取用，不重推**）。")
    print("★ 本程序只核对「**容量规划的算术**」与「**极值时的退化**」——**不是**任何真实引擎的并发数。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_capacity_planning.py —— 从「账」到「容量」：可用显存 → KV 块数 → 最大并发")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **系统侧（容量规划）**；★★ 模型是 **`[E]` 级**，不是任何规范。")
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无引擎。")
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
