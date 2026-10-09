#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_inference_parallelism_and_pd.py —— ★★★ **承接两处让渡**：**推理侧的并行形态** + **PD 分离**。

配套文档：`02-inference-parallelism-and-pd.md`。

★★★ **本篇是【全库唯一「无人接」的让渡】的落点**：
  `100-parallelism/README.md:90`（推理的 TP/PP/EP、PD 分离、KV 缓存切分）与
  `110-communication/README.md:61`（KV 缓存的**传输**、PD 分离、推理的 TP/PP/EP）
  **都把「推理侧」让渡给了本域** ⇒ 本文**必须接住**。

★★★ **但本文【不重推】上游的正文**（`P37`）—— ★ 只写**推理与训练的【实质差异】**：
  · ★★ **PP 在推理里没有反向** ⇒ 时间表上只有 `t_f`（★ 气泡率**公式相同**、**绝对量与上界不同**）；
  · ★★ **TP 在推理里要多切一样东西 —— KV 缓存本身**（★ `100-parallelism/tensor-parallel` 的正文只管权重与激活）
    ⇒ ★ **KV 切分的粒度下限是 KV 头的个数**（GQA 的边界）；
  · ★ **PD 分离**：为什么拆（prefill 与 decode 的 `AI` 差 120 倍）+ **KV 的传输量与时延**。

★★★ **一行的硬边界（只回链，不重写）**：
  · ★★★ **并行策略的六轴本身**（数据/张量/流水/序列/专家/状态）→ `100-parallelism` 全域；
  · ★★★ **通信的机制**（集合通信原语 / NCCL / 拓扑 / 重叠）→ `110-communication` 全域；
  · **训练侧的气泡推导**（`(m+p−1)(t_f+t_b)` / 两个口径）→ `100-parallelism/pipeline-parallel/01`；
  · **chunked prefill 的分块数权衡** → 本域 `batching-and-scheduling/02`；
  · **KV 缓存的「账」** → `130-quantization/kv-cache-quantization/01`；
  · **prefill / decode 的 `AI` 差 120 倍** → `40-kernels/attention/05`。

★★★ **本程序【有真 A 类】**（三条，各自两条独立路径）：
  · `A.1` **前向-only 流水线**：离散事件模拟器 vs 闭式 `(m+p−1)·t_f`；
  · `A.2` **KV 头切分**：逐头分配模拟 vs 闭式 `h_kv / min(t, h_kv)`；
  · `A.3` **PD 分离的临界**：逐用户时间线模拟 vs 阈值的闭式 `n*`（★ 两条路径算的是**不同形式**）。
  ★★ **守卫**（`D.4`）：给流水线模拟器加上反向 ⇒ `A.1` 必须失败。

★ **一条写明的口径**：本机的 **P→D 链路带宽是【推导用的规格值】**（`PD_LINK_BYTES_PER_MS`），
  **不是本机 socket 的任何数字**（`P34`：本机单机、无真网络 ⇒ 残留 `T75`）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ 前向-only 流水线的总时间 == `(m+p−1)·t_f`（模拟器 vs 闭式）；
  · `A.2` ★★★ **KV 头切分**：逐头分配的每设备头数 == 闭式，且 `h_kv < t` 时**切不动**；
  · `A.3` ★★★ **PD 分离的临界**：逐用户时间线的裁决 == 阈值闭式，且无一次性开销时 `n*` 与 `P` 无关；
  · `B.1` ★★ 气泡率 vs `(p, m)`（★ 公式相同、绝对量不同）；
  · `B.2` ★★ 每设备 KV 头数 vs `(h_kv, t)`（★ 下限是 1 头）；
  · `B.3` ★★ PD 裁决网格 + `n*` 随 `P` 的走向；
  · `C.1` ★★★ **气泡率相同、绝对量不同**（`t_f/(t_f+t_b)`）；
  · `C.2` ★★★ **KV 切分的下限**：`h_kv < t` 时线性缩小是错的；
  · `C.3` ★★ **PD 的反例格**：`n = 1` 时分离永不更优；
  · `D.4` ★★★ 守卫实验：给流水线模拟器加反向 ⇒ 断言必须失败。
"""
import ast
import collections
import inspect
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（serving-infra）取 `-- FAIL: serving-infra`；`mutate.py` 用 AST 读出（**不写死**）。
ASSERT_FAIL_MARK = "-- FAIL: serving-infra"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: decoding-strategies", "-- FAIL: engines")

# ★★ **术语纪律**（本域 README §同名不同指）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
TERM_KV_CACHE = "KV 缓存"
TERM_SCHEDULER = "请求调度器"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ 形态 A 的根治：计数 == 豁免清单）：
BARE_ASCII_WORDS = ("ca" + "che", "sched" + "uler")
TERM_BARE_EXEMPT = (
    "本域一律写「**KV 缓存**」全称（不裸写 `cache`，那是 L1/L2 与 kernel cache）",
    "本域一律写「**请求调度器**」全称（不裸写 `scheduler`，那是 GPU 的 warp scheduler）",
)


def _is_cjk(ch):
    """★ 是否 CJK 表意字（★ **不含** CJK 标点 —— 「，」「）」不算「中文语境」）。"""
    return ch != "" and "一" <= ch <= "鿿"


def _count_bare_ascii(src):
    """★★ 数「**裸写**」：被禁 ASCII 词**紧邻中文**（中间只隔空白 / 反引号）的出现次数。"""
    n = 0
    for w in BARE_ASCII_WORDS:
        i = src.find(w)
        while i != -1:
            l = i
            while l > 0 and src[l - 1] in " `\t":
                l -= 1
            r = i + len(w)
            while r < len(src) and src[r] in " `\t":
                r += 1
            lc = src[l - 1] if l > 0 else ""
            rc = src[r] if r < len(src) else ""
            if _is_cjk(lc) or _is_cjk(rc):
                n += 1
            i = src.find(w, i + 1)
    return n


FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

# ★★ **本程序【有 A 类】** —— `D.0` 断言本文件里 A 类判据的条数 **> 0**。
EXPECTED_MIN_A = 3


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D）。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail_fail if detail_fail is not None else "★★ 这条判据【不成立】"))
    return bool(cond)


# ===========================================================================
# 1. A.1 的被测：**前向-only 流水线**（离散事件）
# ===========================================================================
def sim_pipeline(p, m, tf, with_bwd=False):
    """★★★ **被测实现（I）**：前向-only 流水线的**离散事件模拟**。

    ★ 两级串行约束：① 同一级的微批串行；② 同一微批的级间串行（它只能在前一级完成后进下一级）。
    ★★★ `with_bwd=True` 只用于 `D.4` 的守卫实验：加上**反向**（训练侧才有）⇒ 总时间必然变长。

    ★★ **闭式的任何形式都不出现在本函数里**（`D.8` 在源码层核）。
    """
    free = [0.0] * p               # 每一级「空闲」的时刻
    last_end = 0.0
    for _i in range(m):
        t = 0.0
        for j in range(p):         # ★ 前向：级 0 → 级 p−1
            s = free[j] if free[j] > t else t
            t = s + tf
            free[j] = t
        if with_bwd:
            for j in range(p - 1, -1, -1):   # ★ 反向：级 p−1 → 级 0（推理侧**没有**这一段）
                s = free[j] if free[j] > t else t
                t = s + tf
                free[j] = t
        if t > last_end:
            last_end = t
    return last_end


def pipe_closed_form(p, m, tf):
    """★★★ **参照实现（I）**：闭式 —— `(m + p − 1) · t_f`（★ 纯代数，不含任何事件循环）。"""
    return (m + p - 1) * tf


def bubble_rate_megatron(p, m):
    """★ 气泡率（**Megatron 口径**：除以**理想时间** `m·t_f`）—— ★★ 与 `100-parallelism` 的公式**同形**。

    ★★★ **本条【不重推】它的推导**（正文在 `100-parallelism/pipeline-parallel/01`）——
      这里只用它的**值**做对照。
    """
    return (p - 1) / m


# ===========================================================================
# 2. A.2 的被测：**KV 头到设备的分配**（逐头，不用闭式）
# ===========================================================================
def assign_kv_heads(h_kv, t):
    """★★★ **被测实现（II）**：把 `h_kv` 个 **KV 头**逐个分给 `t` 个设备 —— **逐头分配**，数每设备几个。

    ★★★ **切分粒度下限**：一个 KV 头**不可再分** ⇒ 能切出的份数最多 `h_kv`
      ⇒ `t > h_kv` 时**只能复制**、每设备仍要装 **1 个头**。
    ★ 返回 `(每设备头数列表, 每设备最大头数)`。
    """
    s = h_kv if h_kv < t else t          # ★ 可用的「份数」= min(头数, 设备数)
    per = [0] * t
    for h in range(h_kv):
        per[h % s] += 1
    return per, max(per)


def kv_heads_per_device_closed(h_kv, t):
    """★★★ **参照实现（II）**：闭式 —— `h_kv / min(h_kv, t)`（★ 纯代数）。"""
    return h_kv // (h_kv if h_kv < t else t)


# ===========================================================================
# 3. A.3 的被测：**PD 分离的逐用户时间线**
# ===========================================================================
def sim_pd(workload, n_users, cp, b_kv, bw, overhead, mode):
    """★★★ **被测实现（III）**：**逐用户**推进时间线，累加每个用户被**推迟**的时间。

    `workload` = `[(到达时刻, prefill 长度 P), …]`；★ 返回**全体用户被推迟的总时间**。
      · `mode="coupled"`：prefill 在**同一个**引擎上跑 ⇒ 它**阻塞** decode 循环 `cp·P` ⇒ **每个** decode 用户都被推后；
      · `mode="split"`  ：prefill 在**另一个**池上跑 ⇒ decode 不受影响，但 KV 要**传输** `P·b_kv/bw` + 一次性开销
                          ⇒ ★ **只有那一个** prefill 请求被推后。
    ★★ 它**逐用户循环**；闭式走的是**聚合代数**（`D.8` 核两条路径不互相调用）。
    """
    total = 0.0
    for arr, P in workload:
        for _u in range(n_users):
            if mode == "coupled":
                total += cp * P                       # ★ 每个用户都等这一整段
        if mode == "split":
            total += (P * b_kv / bw + overhead)        # ★ 只有一个请求多等这一段
        _ = arr
    return total


def pd_threshold_closed(cp, b_kv, bw, overhead, P):
    """★★★ **参照实现（III）**：临界 decode 用户数 `n*` 的**闭式**。

    ★ 分离更优 ⟺ `n · cp · P > P · b_kv/bw + overhead` ⟺ `n > (b_kv/bw + overhead/P) / cp`。
    ★★★ **`overhead = 0` 时 `n* = (b_kv/bw)/cp`，与 `P` 无关** —— 这是本文一条**可证伪**的结论。
    """
    return (b_kv / bw + overhead / P) / cp


# ===========================================================================
# 4. A 类 —— ★★★ 三条，各自两条独立路径
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **推理侧的并行**：三条判据，各自两条独立路径")
    print("=" * 78)

    # ── A.1 ★★★ 前向-only 流水线 ────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **前向-only 流水线**：模拟器 vs 闭式 `(m+p−1)·t_f` ──")
    p, tf = specs.PP_N_STAGES, specs.PP_FWD_COST
    tot = {}
    for m in specs.PP_M_MICRO_GRID:
        tot[m] = sim_pipeline(p, m, tf)
    cf = {m: pipe_closed_form(p, m, tf) for m in specs.PP_M_MICRO_GRID}
    bad = [m for m in specs.PP_M_MICRO_GRID if abs(tot[m] - cf[m]) > 1e-12]
    print("     `p = %d` · `t_f = %.1f` · `m ∈ %s`" % (p, tf, list(specs.PP_M_MICRO_GRID)))
    print("     %-6s %-14s %-14s %-14s" % ("m", "模拟器", "闭式", "气泡(绝对)"))
    for m in specs.PP_M_MICRO_GRID:
        print("     %-6d %-14.2f %-14.2f %-14.2f" % (m, tot[m], cf[m], tot[m] - m * tf))
    check("A.1", not bad and tot[1] == p * tf and tot[specs.PP_M_MICRO_GRID[-1]] == pipe_closed_form(p, specs.PP_M_MICRO_GRID[-1], tf),
          "★★★ **判据成立**：**%d 档 `m`** 上，离散事件模拟的总时间与闭式 `(m+p−1)·t_f` **逐位一致**"
          "（`m=1` 时 = `%d×%.1f = %.1f`，气泡吃满）—— ★★ 一面按级推进、一面纯代数，两条独立路径会合"
          % (len(specs.PP_M_MICRO_GRID), p, tf, p * tf),
          "**流水线总时间不一致** —— ★ 先查 `sim_pipeline` 的级间串行与 `pipe_closed_form`")

    # ── A.2 ★★★ KV 头切分 ──────────────────────────────────────────────
    print()
    print("  ── A.2 ★★★ **KV 头切分**：逐头分配 vs 闭式 `h_kv / min(h_kv, t)` ──")
    print("     %-10s %-8s %-14s %-14s %-14s" % ("(h_kv,t)", "份数", "逐头分配", "闭式", "线性 `h_kv/t`"))
    bad2 = []
    for h_kv, t in specs.TP_KV_HEAD_GRID:
        per, mx = assign_kv_heads(h_kv, t)
        cl = kv_heads_per_device_closed(h_kv, t)
        lin = h_kv / t
        if mx != cl:
            bad2.append((h_kv, t))
        print("     %-10s %-8d %-14d %-14d %-14.4f" % ("(%d,%d)" % (h_kv, t), min(h_kv, t), mx, cl, lin))
    print("     ★★★ **读法**：`h_kv < t` 的两格（`(2,8)` / `(8,16)`）—— 闭式给出 **1 头/设备**，")
    print("        而「线性 `h_kv/t`」给出 `0.25` / `0.5`（**不可能**：一个头不可再分）⇒ ★ **线性缩小是错的**。")
    check("A.2", not bad2 and len(specs.TP_KV_HEAD_GRID) == 5
          and assign_kv_heads(2, 8)[1] == 1 and assign_kv_heads(8, 4)[1] == 2
          and assign_kv_heads(8, 8)[1] == 1,
          "★★★ **判据成立**：**5 格**上逐头分配与闭式**逐格一致**；`(2,8)` 的每设备头数 = **1**"
          "（线性会写成 `0.25`）—— ★★ 再次两条独立路径会合",
          "**KV 头分配不一致** —— ★ 先查 `assign_kv_heads` 的份数与 `kv_heads_per_device_closed`")

    # ── A.3 ★★★ PD 分离的临界 ──────────────────────────────────────────
    print()
    print("  ── A.3 ★★★ **PD 分离的临界**：逐用户时间线 vs 阈值闭式 ──")
    cp = specs.PD_PREFILL_MS_PER_TOKEN
    b_kv = specs.PD_KV_BYTES_PER_TOKEN
    bw = specs.PD_LINK_BYTES_PER_MS
    ov = specs.PD_SPLIT_OVERHEAD_MS
    verdict_sim = {}
    for P in specs.PD_PREFILL_LEN_GRID:
        wl = [(0.0, P)]
        for n in specs.PD_DECODE_USERS_GRID:
            c = sim_pd(wl, n, cp, b_kv, bw, ov, "coupled")
            s = sim_pd(wl, n, cp, b_kv, bw, ov, "split")
            verdict_sim[(P, n)] = s < c
    verdict_cf = {}
    for P in specs.PD_PREFILL_LEN_GRID:
        nstar = pd_threshold_closed(cp, b_kv, bw, ov, P)
        for n in specs.PD_DECODE_USERS_GRID:
            verdict_cf[(P, n)] = n > nstar
    mismatch = [k for k in verdict_sim if verdict_sim[k] != verdict_cf[k]]
    nstars = {P: pd_threshold_closed(cp, b_kv, bw, ov, P) for P in specs.PD_PREFILL_LEN_GRID}
    nstar0 = {P: pd_threshold_closed(cp, b_kv, bw, 0.0, P) for P in specs.PD_PREFILL_LEN_GRID}
    print("     `b_kv/bw = %.5f` ms/token · `c_p = %.3f` ms/token · 一次性开销 = %.2f ms" % (b_kv / bw, cp, ov))
    print("     %-10s %-14s %-14s" % ("P", "临界 n*", "（无开销时 n*）"))
    for P in specs.PD_PREFILL_LEN_GRID:
        print("     %-10d %-14.4f %-14.5f" % (P, nstars[P], nstar0[P]))
    print("     ★★ **裁决不一致的格数 = %d**（应 `0`）" % len(mismatch))
    print("     ★★★ **读法**：`n*` **随 `P` 上升而下降**（趋于 `(b_kv/bw)/c_p = %.4f`）；" % (b_kv / bw / cp))
    print("        ★★ **`overhead = 0` 时 `n*` 与 `P` 无关**（恒为 %.4f）—— 那是本文一条可证伪的结论。" % (b_kv / bw / cp))
    check("A.3", not mismatch and abs(nstar0[specs.PD_PREFILL_LEN_GRID[0]] - nstar0[specs.PD_PREFILL_LEN_GRID[-1]]) < 1e-12
          and nstars[specs.PD_PREFILL_LEN_GRID[0]] > nstars[specs.PD_PREFILL_LEN_GRID[-1]]
          and not verdict_sim[(16, 1)] and verdict_sim[(1024, 8)],
          "★★★ **判据成立**：**%d 格**的裁决两条路径**逐格一致**；★ 带一次性开销时 `n*` **随 `P` 下降**"
          "（`%.3f → %.3f`），★ 而无开销时**恒为 %.4f**（与 `P` 无关）—— ★★ 「与 `P` 无关」这条结论**可证伪且成立**"
          % (len(verdict_sim), nstars[specs.PD_PREFILL_LEN_GRID[0]], nstars[specs.PD_PREFILL_LEN_GRID[-1]], b_kv / bw / cp),
          "**PD 裁决不一致（或 `n*` 的走向不对）** —— ★ 先查 `sim_pd` 与 `pd_threshold_closed`")


# ===========================================================================
# 5. B 类：三张结构账
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 三张结构账：**气泡率** · **每设备 KV 头数** · **PD 裁决网格**")
    print("=" * 78)

    # ── B.1 ★★ 气泡率 ──────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **气泡率 vs `(p, m)`**（★ 公式与训练侧**同形**，绝对量不同）──")
    p, tf, tb = specs.PP_N_STAGES, specs.PP_FWD_COST, specs.PP_BWD_COST
    print("     %-6s %-14s %-14s %-14s %-14s" % ("m", "推理总时间", "训练总时间", "气泡率", "训练/推理"))
    rowsm = []
    for m in specs.PP_M_MICRO_GRID:
        inf = pipe_closed_form(p, m, tf)
        trn = (m + p - 1) * (tf + tb)
        br = bubble_rate_megatron(p, m)
        rowsm.append((m, inf, trn, br))
        print("     %-6d %-14.2f %-14.2f %-14.6f %-14.4f" % (m, inf, trn, br, trn / inf))
    mono = all(rowsm[i][3] > rowsm[i + 1][3] for i in range(len(rowsm) - 1))
    print("     ★★ 气泡率**随 `m` 单调降** = %s；`m → ∞` 时 → `0`（★ 推理侧**可以把 `m` 提得很高**）" % mono)
    print("     ★★★ **读法**：**气泡率的公式两边一样** `(p−1)/m`；**差别在** ——")
    print("        ① 绝对气泡：推理 `(p−1)·t_f` vs 训练 `(p−1)·(t_f+t_b)`；")
    print("        ② ★★ **`m` 的上限**：训练要为每个在飞微批**存激活**（`90-training` 的 checkpointing 侧），"
          "而推理**没有反向、不必存激活** ⇒ `m` 可以取得大得多。")
    check("B.1", mono and len(rowsm) == len(specs.PP_M_MICRO_GRID)
          and abs(rowsm[0][3] - (p - 1) / 1) < 1e-12 and abs(rowsm[0][1] - p * tf) < 1e-12
          and abs(rowsm[-1][3] - (p - 1) / specs.PP_M_MICRO_GRID[-1]) < 1e-12,
          "★★ **判据成立**：气泡率 `(p−1)/m` 从 **%.4f**（`m=1`）降到 **%.4f**（`m=%d`）；"
          "★ 同一 `m` 下推理总时间 = 训练的 **%.4f 倍**（`t_f/(t_f+t_b)`）"
          % (rowsm[0][3], rowsm[-1][3], specs.PP_M_MICRO_GRID[-1], rowsm[0][1] / rowsm[0][2]),
          "**气泡率或端值被动过** —— ★ 先查 `bubble_rate_megatron` / `pipe_closed_form`")

    # ── B.2 ★★ 每设备 KV 头数 ───────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **每设备 KV 头数 vs `(h_kv, t)`**（★ 下限是 1 头）──")
    print("     %-10s %-14s %-16s %-16s" % ("(h_kv,t)", "每设备头数", "KV 相对单卡", "线性会写成"))
    rows2 = []
    for h_kv, t in specs.TP_KV_HEAD_GRID:
        per, mx = assign_kv_heads(h_kv, t)
        rows2.append((h_kv, t, mx, mx / h_kv))
        print("     %-10s %-14d %-16.4f %-16.4f" % ("(%d,%d)" % (h_kv, t), mx, mx / h_kv, (h_kv / t) / h_kv))
    print("     ★★ **读法**：`KV 相对单卡` = 每设备头数 / 单卡头数 —— 它**只在 `h_kv ≥ t` 时**才等于 `1/t`；")
    print("        `h_kv < t` 时**卡在 `1/h_kv`**（★ 复制）⇒ ★★ **再加 TP 度也压不下去**。")
    floor_ok = all(r[2] >= 1 for r in rows2)
    check("B.2", floor_ok and len(rows2) == 5
          and abs(rows2[2][3] - 0.5) < 1e-12 and abs(rows2[3][3] - 0.5) < 1e-12,
          "★★ **判据成立**：**5 格**的每设备头数**全部 ≥ 1**（★ 下限是 1 头）；`(4,2)` 与 `(2,8)` 的"
          "「KV 相对单卡」都是 **%.2f** —— ★ `(2,8)` 用 8 张卡却只把 KV 压到一半（★ 线性会说 `1/8`）"
          % rows2[3][3],
          "**每设备头数或下限被动过** —— ★ 先查 `assign_kv_heads`")

    # ── B.3 ★★ PD 裁决网格 ─────────────────────────────────────────────
    print()
    print("  ── B.3 ★★ **PD 裁决网格**（行 = prefill 长度 `P`，列 = 被阻塞的 decode 用户数 `n`）──")
    cp = specs.PD_PREFILL_MS_PER_TOKEN
    b_kv = specs.PD_KV_BYTES_PER_TOKEN
    bw = specs.PD_LINK_BYTES_PER_MS
    ov = specs.PD_SPLIT_OVERHEAD_MS
    print("     %-10s %s" % ("P \\ n", "  ".join("%-8d" % n for n in specs.PD_DECODE_USERS_GRID)))
    grid_split = {}
    for P in specs.PD_PREFILL_LEN_GRID:
        cells = []
        for n in specs.PD_DECODE_USERS_GRID:
            c = sim_pd([(0.0, P)], n, cp, b_kv, bw, ov, "coupled")
            s = sim_pd([(0.0, P)], n, cp, b_kv, bw, ov, "split")
            w = s < c
            grid_split[(P, n)] = w
            cells.append("%-8s" % ("分离" if w else "耦合"))
        print("     %-10d %s" % (P, "  ".join(cells)))
    n_split = sum(1 for v in grid_split.values() if v)
    print("     ★★ 分离更优的格数 = **%d / %d**；★★ **`n = 1` 那一列全是「耦合」**（只有一个人被阻塞）"
          % (n_split, len(grid_split)))
    check("B.3", n_split == 7 and all(not grid_split[(P, 1)] for P in specs.PD_PREFILL_LEN_GRID)
          and all(grid_split[(P, 8)] for P in specs.PD_PREFILL_LEN_GRID)
          and grid_split[(16, 4)] is False and grid_split[(64, 4)] is True,
          "★★ **判据成立**：**16 格**里 **7 格**分离更优；★ `n=1` 列**全耦合**（反例），`n=8` 列**全分离**；"
          "★ `n=4` 时 `P=16` 耦合、`P=64` 分离 ⇒ **临界随 `P` 移动**",
          "**PD 网格被动过** —— ★ 先查 `PD_*` 与 `sim_pd`")


# ===========================================================================
# 6. C 类：**两条硬边界**与一条**反例**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**气泡率同、绝对量不同** · **KV 切分下限** · **PD 的反例格**")
    print("=" * 78)

    # ── C.1 ★★★ 气泡率相同、绝对量不同 ─────────────────────────────────
    print()
    print("  ── C.1 ★★★ **气泡率的公式两边一样、绝对量不同** ──")
    p, tf, tb = specs.PP_N_STAGES, specs.PP_FWD_COST, specs.PP_BWD_COST
    p_, m_ = 4, 8
    br_inf = bubble_rate_megatron(p_, m_)
    br_trn = (p_ - 1) / m_            # ★ 训练侧 Megatron 口径（**同形**）
    abs_inf = (p_ - 1) * tf
    abs_trn = (p_ - 1) * (tf + tb)
    print("     `(p, m) = (%d, %d)`：气泡率 **推理 %.6f == 训练 %.6f** ⇒ **公式相同** = %s"
          % (p_, m_, br_inf, br_trn, abs(br_inf - br_trn) < 1e-12))
    print("     绝对气泡：推理 `(p−1)·t_f = %.2f` vs 训练 `(p−1)·(t_f+t_b) = %.2f`" % (abs_inf, abs_trn))
    print("     ★★★ **比值 = `t_f/(t_f+t_b)` = %.4f** —— ★ 推理侧的绝对气泡**更小**；" % (tf / (tf + tb)))
    print("     而**真正拉开差距的是 `m` 的上限**（训练要存激活，推理不必）—— ★ 公式相同、可达区不同。")
    check("C.1", abs(br_inf - br_trn) < 1e-12 and abs_trn > abs_inf
          and abs(abs_inf / abs_trn - tf / (tf + tb)) < 1e-12,
          "★★★ **判据成立**：`(p,m)=(4,8)` 下**气泡率两边都是 %.6f**（公式相同），"
          "而绝对气泡之比 = `t_f/(t_f+t_b)` = **%.4f < 1** —— ★★ 「公式相同、绝对量不同」**逐位对上**"
          % (br_inf, tf / (tf + tb)),
          "**气泡率或比值被动过** —— ★ 先查 `bubble_rate_megatron` 与 `PP_FWD_COST`/`PP_BWD_COST`")

    # ── C.2 ★★★ KV 切分的下限 ──────────────────────────────────────────
    print()
    print("  ── C.2 ★★★ **KV 切分的下限**：`h_kv < t` 时「线性缩小」是错的 ──")
    mis = []
    for h_kv, t in specs.TP_KV_HEAD_GRID:
        _per, mx = assign_kv_heads(h_kv, t)
        if h_kv < t and abs(mx / h_kv - 1.0 / t) > 1e-12:
            mis.append((h_kv, t))
    print("     ★★ 线性会说「`t` 张卡 ⇒ 每设备 `1/t` 的 KV」；★★★ 实际**卡在 `1/h_kv`**（`h_kv < t` 时）。")
    print("     ★★ 违反线性预测的格数 = **%d**（★ 应是 `(2,8)` 与 `(8,16)` 两格）" % len(mis))
    print("     ★★★ **读法**：这正是「**TP 在推理里要多切一样东西**」的边界 ——")
    print("        权重与激活可以切到 `t` 份，但 **KV 缓存的切分粒度下限是 KV 头的个数**。")
    print("        ★★ 要往下切，只能改架构（MQA `h_kv=1` 就**完全切不动**）或减少 TP 度。")
    check("C.2", len(mis) == 2 and (2, 8) in mis and (8, 16) in mis
          and assign_kv_heads(1, 8)[1] == 1,
          "★★★ **判据成立**：**%d 格**违反线性预测（`h_kv < t`）；极端 `h_kv = 1`（MQA）时**每设备必装 1 个头**"
          "（8 张卡也压不下去）—— ★★ 这是**推理侧 TP 的独有约束**" % len(mis),
          "**KV 下限被动过** —— ★ 先查 `assign_kv_heads` 的 `min(h_kv, t)`")

    # ── C.3 ★★ PD 的反例格 ─────────────────────────────────────────────
    print()
    print("  ── C.3 ★★ **PD 的反例格**：`n = 1` 时分离**永不更优** ──")
    cp = specs.PD_PREFILL_MS_PER_TOKEN
    b_kv = specs.PD_KV_BYTES_PER_TOKEN
    bw = specs.PD_LINK_BYTES_PER_MS
    ov = specs.PD_SPLIT_OVERHEAD_MS
    c1 = sim_pd([(0.0, 1024)], 1, cp, b_kv, bw, ov, "coupled")
    s1 = sim_pd([(0.0, 1024)], 1, cp, b_kv, bw, ov, "split")
    print("     `n = 1`、`P = 1024`：耦合的代价 = **%.4f**（一人等 `c_p·P`）· 分离的代价 = **%.4f**"
          "（传输 + 开销）" % (c1, s1))
    print("     ★★ 分离更优 = **%s**（应 `False`）—— ★ 只阻塞一个人时，**传输反而更贵**。" % (s1 < c1))
    print("     ★★★ **读法**：PD 分离**不是普适更优** —— 它换的是「**少阻塞 N 个人**」对「**多一次传输**」，")
    print("        当被阻塞的人**不够多**时，这笔交换是亏的。")
    check("C.3", (not (s1 < c1)) and c1 > 0 and s1 > 0 and c1 != s1,
          "★★ **判据成立**：`n = 1` 时分离的代价（**%.4f**）**大于**耦合（**%.4f**）⇒ ★ **反例成立**："
          "PD 分离**有适用域**（需要「被阻塞的人够多」）" % (s1, c1),
          "**反例格被动过** —— ★ 先查 `sim_pd` 的两种模式")


# ===========================================================================
# 7. 前哨：口径 / 回链 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 回链锚点 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/回链/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无多机 / 无真网络**。")
    print("      ★★ 但**本程序的 A 类是真的**：三条判据各有**两条独立路径**。")
    before = len(FAILS)

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

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: serving-infra" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: serving-infra`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies", "-- FAIL: engines"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    # ── D.3 ★★★ 回链锚点（**本程序【不重推】它们的正文**）────────────────
    check("D.3", specs.PIPELINE_BUBBLE_REF == "100-parallelism/pipeline-parallel/01-microbatches-and-bubbles"
          and specs.PARALLELISM_LETDOWN_REF == "100-parallelism/README.md:90"
          and specs.COMMUNICATION_LETDOWN_REF == "110-communication/README.md:61"
          and specs.BATCHING_CHUNKED_REF == "120-inference/batching-and-scheduling/02-chunked-prefill"
          and specs.KV_ACCOUNT_REF == "130-quantization/kv-cache-quantization/01"
          and specs.KERNEL_AI_REF == "40-kernels/attention/05"
          and specs.SERVING_INFRA_FETCH_DAY == "2026-10-09",
          "★★★ **回链锚点逐字钉住**（★ **正文在彼处，本文一行都不重写**）：训练侧气泡 `%s` · "
          "两处让渡 `%s` / `%s` · 同域前驱 `%s` · KV 账 `%s` · `AI` 账 `%s`；抓取日 `%s`"
          % (specs.PIPELINE_BUBBLE_REF, specs.PARALLELISM_LETDOWN_REF, specs.COMMUNICATION_LETDOWN_REF,
             specs.BATCHING_CHUNKED_REF, specs.KV_ACCOUNT_REF, specs.KERNEL_AI_REF, specs.SERVING_INFRA_FETCH_DAY))

    ok_grid = (specs.PP_N_STAGES >= 2 and len(specs.PP_M_MICRO_GRID) >= 2
               and 1 in specs.PP_M_MICRO_GRID and any(h < t for h, t in specs.TP_KV_HEAD_GRID)
               and any(h >= t for h, t in specs.TP_KV_HEAD_GRID)
               and len(specs.PD_PREFILL_LEN_GRID) >= 2 and 1 in specs.PD_DECODE_USERS_GRID)
    print("     ★★ `p ≥ 2`、`m` 含 1、KV 网格**既有 `h ≥ t` 也有 `h < t`**、PD 网格含 `n = 1` = %s" % ok_grid)
    check("D.3b", ok_grid and specs.UNSOURCED == ()
          and specs.PP_FWD_COST == 3.0 and specs.PP_BWD_COST == 3.0
          and specs.PD_KV_BYTES_PER_TOKEN == 131072 and specs.PD_LINK_BYTES_PER_MS == 2621440
          and specs.PD_PREFILL_MS_PER_TOKEN == 0.02 and specs.PD_SPLIT_OVERHEAD_MS == 0.5
          and specs.TP_KV_HEAD_GRID == ((8, 8), (8, 4), (4, 2), (2, 8), (8, 16))
          and specs.PD_PREFILL_LEN_GRID == (16, 64, 256, 1024)
          and specs.PD_DECODE_USERS_GRID == (1, 2, 4, 8)
          and specs.PP_M_MICRO_GRID == (1, 2, 4, 8, 16),
          "★★★ **口径守卫 + 参数逐元素钉住**：PD 参数（`b_kv` / 带宽 / `c_p` / 开销）· KV 头网格 5 行 · "
          "PD 的 `P`/`n` 网格 · `m` 网格；★ 网格**同时含 `h ≥ t` 与 `h < t`**（否则 `A.2`/`C.2` 退化成恒真）",
          "**网格或 PD 参数被动过** —— ★ 先查 `TP_KV_HEAD_GRID` / `PD_*`")
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：A 类存在性 · 守卫实验 · 独立性 · 术语纪律
# ===========================================================================
def _func_names(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


def _count_a_checks_in_source():
    """★ 数本文件里 `check("A...", ...)` 的调用数（**读自己的源码**，不是写死的数）。"""
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "check" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                    and a0.value.startswith("A."):
                n += 1
    return n


def part_D():
    print()
    print("=" * 78)
    print("D · **A 类存在性** · 守卫实验 · 独立性 · 术语纪律")
    print("=" * 78)

    n_a = _count_a_checks_in_source()
    print("  ── D.0 ★★★ **机械证明**：AST 扫本文件，`check(\"A...\")` 的调用数 = **%d** ──" % n_a)
    check("D.0", n_a >= EXPECTED_MIN_A,
          "★★★ **判据成立**：本文件里 **A 类判据有 %d 条**（≥ %d）—— ★ 本主题**有真 A 类**"
          "（流水线 / KV 头切分 / PD 临界，各两条独立路径）" % (n_a, EXPECTED_MIN_A),
          "★★ 本文件里 A 类判据少于 %d 条 —— ★ 与「本主题有 A 类」**自相矛盾**" % EXPECTED_MIN_A)

    # ── D.4 ★★★ 守卫实验：给流水线加上反向 ──────────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验**：给前向-only 流水线加上**反向**（训练侧才有）──")
    p, tf = specs.PP_N_STAGES, specs.PP_FWD_COST
    m = 4
    fwd = sim_pipeline(p, m, tf)
    bwd = sim_pipeline(p, m, tf, with_bwd=True)
    print("     构造：`(p, m) = (%d, %d)`、`t_f = %.1f`；前向-only = **%.2f** · 加反向 = **%.2f**"
          % (p, m, tf, fwd, bwd))
    print("     ★★★ **读法**：加上反向 ⇒ 每级要做 `2` 倍工作 ⇒ 总时间**变成 `2` 倍**，闭式 `(m+p−1)·t_f` 不再成立。")
    check("D.4", abs(fwd - pipe_closed_form(p, m, tf)) < 1e-12 and bwd == m * 2 * p * tf and bwd > fwd,
          "★★★ **守卫成立**：加上反向以后总时间**从 %.2f 变成 %.2f**（= `m·2p·t_f`，各级都要做两遍）⇒ "
          "★★ **`A.1` 对「推理侧【没有】反向」这条规则是敏感的**" % (fwd, bwd),
          "**守卫失效** —— ★ 加反向总时间居然不变（那 `A.1` 抓不到它）")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.PP_N_STAGES == 4 and specs.PP_FWD_COST == 3.0 and specs.PP_BWD_COST == 3.0
          and specs.PP_M_MICRO_GRID == (1, 2, 4, 8, 16)
          and specs.TP_KV_HEAD_GRID == ((8, 8), (8, 4), (4, 2), (2, 8), (8, 16))
          and specs.PD_KV_BYTES_PER_TOKEN == 131072 and specs.PD_LINK_BYTES_PER_MS == 2621440
          and specs.PD_PREFILL_MS_PER_TOKEN == 0.02 and specs.PD_SPLIT_OVERHEAD_MS == 0.5
          and specs.PD_PREFILL_LEN_GRID == (16, 64, 256, 1024)
          and specs.PD_DECODE_USERS_GRID == (1, 2, 4, 8)
          and specs.SERVING_INFRA_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：流水线两个代价 + `m` 网格（**逐元素**）· KV 头网格 5 行 × 2 列 · "
          "PD 四个参数 · PD 两个网格（**逐元素**）· 抓取日；★ `UNSOURCED == ()`",
          "**某个常量被动过** —— ★ 先查 `PP_*` / `TP_KV_HEAD_GRID` / `PD_*`")

    # ── D.8 ★★★ 独立性（源码级）──────────────────────────────────────────
    s1 = _func_names(sim_pipeline)
    c1 = _func_names(pipe_closed_form)
    s2 = _func_names(assign_kv_heads)
    c2 = _func_names(kv_heads_per_device_closed)
    s3 = _func_names(sim_pd)
    c3 = _func_names(pd_threshold_closed)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：三条判据的「被测」与「参照」互不引用 ──")
    print("     A.1: `sim_pipeline`→`pipe_closed_form` = %s ；反向 = %s"
          % ("pipe_closed_form" in s1, "pipe_closed_form" in c1))
    print("     A.2: `assign_kv_heads`→`kv_heads_per_device_closed` = %s ；反向 = %s"
          % ("kv_heads_per_device_closed" in s2, "assign_kv_heads" in c2))
    print("     A.3: `sim_pd`→`pd_threshold_closed` = %s ；反向 = %s"
          % ("pd_threshold_closed" in s3, "sim_pd" in c3))
    print("     ★★★ **降级措辞**：这只是「**几段代码不互相引用**」的**源码级线索**，**不证明**判据有信息量 ——"
          " 后者由每一条 A 与 `D.4` 的**行为**兜底。")
    check("D.8", "pipe_closed_form" not in s1 and "sim_pipeline" not in c1
          and "kv_heads_per_device_closed" not in s2 and "assign_kv_heads" not in c2
          and "pd_threshold_closed" not in s3 and "sim_pd" not in c3,
          "★★ **源码级线索成立**：三条判据的**被测与参照两两互不引用**（★ **两侧都审**）——"
          " ★★ 于是「结果相同」**不是恒真**，而是**两条独立路径的会合**",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★★ 术语纪律 ───────────────────────────────────────────────
    src = open(__file__, encoding="utf-8").read()
    n_bare_terms = _count_bare_ascii(src)
    n_exempt_terms = sum(src.count(s) * _count_bare_ascii(s) for s in TERM_BARE_EXEMPT)
    print()
    print("  ── D.9 ★★★ **术语纪律**：一律写「%s」/「%s」全称；限定语逐字出现 ──"
          % (TERM_KV_CACHE, TERM_SCHEDULER))
    print("     ★★★ 源码里**裸写那两个被禁 ASCII 词**（★ 紧邻中文）= **%d** 处；豁免清单覆盖 = **%d**"
          % (n_bare_terms, n_exempt_terms))
    check("D.9", TERM_KV_CACHE == "KV 缓存" and TERM_SCHEDULER == "请求调度器"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_KV_CACHE in src
          and BARE_ASCII_WORDS == ("ca" + "che", "sched" + "uler")
          and TERM_BARE_EXEMPT == (
              "本域一律写「**KV 缓存**」全称（不裸写 `" + BARE_ASCII_WORDS[0] + "`，那是 L1/L2 与 kernel "
              + BARE_ASCII_WORDS[0] + "）",
              "本域一律写「**请求调度器**」全称（不裸写 `" + BARE_ASCII_WORDS[1] + "`，那是 GPU 的 warp "
              + BARE_ASCII_WORDS[1] + "）")
          and n_bare_terms == n_exempt_terms
          and all(s in src for s in TERM_BARE_EXEMPT),
          "★★ **判据成立**：两条术语常量 `%r` / `%r`；★ 限定语 `%r` **逐字出现**；"
          " ★★★ **被禁 ASCII 词的裸写计数 `%d` 恰好等于其豁免清单覆盖的 `%d`**"
          % (TERM_KV_CACHE, TERM_SCHEDULER, LIMIT_QUOTE, n_bare_terms, n_exempt_terms),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写**")


# ===========================================================================
# 9. 汇总 / 主程序
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
    print("★★★ **A = %d/%d** —— ★★ **本程序有真 A 类**：流水线 / KV 头切分 / PD 临界，各两条独立路径。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：模型与算例是 **`[E]` 级（本库实现）**；★ PD 的链路带宽是**推导用的规格值**（**不是本机 socket 数字**）。")
    print("★ 本程序只核对「**推理与训练的实质差异**」与「**PD 分离的临界**」——")
    print("  ★★ **不重推** `100-parallelism` 的六轴与气泡推导、**不重推** `110-communication` 的通信机制。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_inference_parallelism_and_pd.py —— 推理侧的并行形态 + PD 分离")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无多机 / 无真网络。")
    print("  ★★★ **本程序有真 A 类**（三条）；★★ 承接 `100-parallelism:90` 与 `110-communication:61` 的让渡。")
    if not preflight():
        print()
        print("!! 口径/回链钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
