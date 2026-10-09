#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_latency_and_slo.py —— **两个延迟**（TTFT 与 inter-token latency）与它们的**冲突**。

配套文档：`01-latency-and-slo.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本程序【有真 A 类】**（★ 与 `engines/01` / `engines/03` 的「机械证明没有 A 类」互为镜像）：
  ★ **被测** = 一个**离散事件模拟器** `simulate()`：按**逐步推进**的方式跑引擎，从**事件时间表**里
    **量出**每个请求的 TTFT 与 inter-token latency；
  ★ **参照** = `closed_form()`：**纯代数** —— 从相位数与步耗时模型**直接算出**同样几个量。
  ★★★ **两条路径不共享一行代码**（`D.8` 在源码层核）—— 模拟器不 import 参照、参照不 import 模拟器。
  ★★ **判据**：任意 `(B, 工作负载)` 下，**逐请求**的 TTFT 与 inter-token latency、以及 makespan，
    两条路径**逐位一致**（容差 `1e-12`）。
  ★★ **守卫**（`D.4`）：把模拟器的**准入上限**改成 `B + 1` ⇒ **断言必须失败**（`CONTRIBUTING:235`）。

★★★ **本程序只写【两个延迟与它们的冲突】** —— ★ **别处的正文一律回链**（`P37`）：
  · ★★ **chunked prefill 的分块数权衡 `k*`** → 本域 `batching-and-scheduling/02-chunked-prefill`；
  · ★★ **continuous batching 的收益来源（长度方差）** → 本域 `batching-and-scheduling/01-iteration-level-scheduling`；
  · **KV 缓存的「账」** → `130-quantization/kv-cache-quantization/01`；
  · **prefill / decode 的 `AI` 差 120 倍** → `40-kernels/attention/05-decoding-attention`。

★★ **一条写明的简化（★ 必须连同结论一起写）**：本模型把 **prefill 记成一次独立的 prefill 步**
  （耗时 `LAT_PREFILL_COST`，**不占 decode 步的时间**）。★★★ **真实里 prefill 与 decode 共享算力**
  —— 那套耦合模型（「取较慢者」）的正文在 `batching-and-scheduling/02-chunked-prefill`，本程序**不重推**。

★★★ **本程序的禁写清单（计划 `§0.7 C.5`）**：
  · `F9`（**不裸写那个社区缩写当 vLLM 的官方词** —— vLLM 页面上它 **0 处**，见 `C.1`）；
  · ★★ **不得把 TTFT 的口径混起来**：vLLM 的 TTFT 从「request was scheduled」起（**不含排队**），
    本库口径**含排队** ⇒ 两个量**不同**（`C.2`）。

★★★ **本程序的判据纪律（★ 形态 A 的根治）**：★ 抬头写了「不裸写某某」这条纪律，
  本程序里**必须有一条断言能抓住「没做到」** —— 见 `D.9`：
  ★★ 判据是**等式**「裸写计数 == 显式豁免清单覆盖的计数」，**不是「≤ N」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **inter-token latency**：模拟器**量出**的逐请求 ITL == 参照的 `α + β·n_相位`；
  · `A.2` ★★★ **makespan**：模拟器**量出**的总时间 == 参照的 `Σ_k (prefill + (D−1)·步耗时)`；
  · `A.3` ★★★ **TTFT**：模拟器**量出**的逐请求 TTFT == 参照的 `相位起点 + prefill`；
  · `B.1` ★★ **两个延迟的冲突**：`B ↑` ⇒ ITL **单调升**、吞吐**单调升**（⇒ 没有一个 `B` 两头都好）；
  · `B.2` ★ **两个端点的闭式**（`B = 1` 与 `B = N` 各有唯一解；★ **不守 `1 / β`** —— 那是**步吞吐**的上界，与本程序的 `throughput()` 是两个量）；
  · `B.3` ★★ **TTFT 的分解**（排队 + prefill + 首 token）与**两个工作负载下方向不同**（数值驱动）；
  · `C.1` ★★★ **把社区缩写当 vLLM 官方词**必须被拒（`F9`）；
  · `C.2` ★★★ **两个 TTFT 口径**：vLLM 的从 `scheduled` 起（不含排队），本库的含排队；
  · `C.3` ★★ **边界**：`B = 1`（ITL 最小）与 `B ≥ N`（单相位）；
  · `D.4` ★★★ **守卫实验**：把准入上限改成 `B + 1` ⇒ 断言必须失败；
  · `D.5` ★★★ **六条引文逐字钉住**（★ 一律 `==`，不用 `in`）。
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
# ★★★ 失败标记：本主题（serving-infra）取 `-- FAIL: serving-infra`；`mutate.py` 用 AST 读出（**不写死**）。
ASSERT_FAIL_MARK = "-- FAIL: serving-infra"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: decoding-strategies", "-- FAIL: engines")

# ★★ **术语纪律**（本域 README §同名不同指）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
TERM_KV_CACHE = "KV 缓存"
TERM_SCHEDULER = "请求调度器"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ 形态 A：「纪律宣称了而断言没盯住」，本域已栽过三次）：
#   本程序源码里**裸写那两个被禁 ASCII 词**的【计数】必须**恰好等于**下面这份**显式豁免清单**覆盖的计数。
#   ★★★ 判据是【计数 == 豁免覆盖数】，**不是「≤ N」**（「≤ N」挡不住**新增**一处裸写）。
#   ★★ **裸写 = 该 ASCII 词紧邻中文**（中间只隔空白 / 反引号）；★ 计数码用**拼接**避免自计。
BARE_ASCII_WORDS = ("ca" + "che", "sched" + "uler")
TERM_BARE_EXEMPT = (
    "本域一律写「**KV 缓存**」全称（不裸写 `cache`，那是 L1/L2 与 kernel cache）",
    "本域一律写「**请求调度器**」全称（不裸写 `scheduler`，那是 GPU 的 warp scheduler）",
)

# ★★ **引文里的口径常量**（`C.2` 的判据）：
VLLM_TTFT_START_TOKEN = "scheduled"
VLLM_TTFT_EXCLUDES_QUEUE_NOTE = "vLLM 的 TTFT 从「request was scheduled」起算，不含排队；本库口径含排队"


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
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name,
                                detail_fail if detail_fail is not None else
                                "★★ 这条判据【不成立】"))
    return bool(cond)


# ===========================================================================
# 1. 被测（I）：**离散事件模拟器** —— 按步推进，从事件时间表里【量】延迟
# ===========================================================================
def step_cost(n, alpha, beta):
    """★ 一步（一次 iteration）的耗时模型：`α + β · 在飞请求数`。

    ★★ 它是**本文的模型假设**（不是任何引擎的实测）：`α` = 与批无关的固定开销，`β·n` = 随批线性增长的部分。
    """
    return alpha + beta * n


def simulate(arrivals, B, D, pref, alpha, beta, corrupt=None):
    """★★★ **被测实现（I）**：离散事件模拟器。

    ★ 事件粒度 = **一步**：先准入（队首最多 `B` 条），再跑**一个** prefill 步或**一个** decode 步。
    ★ 「按步推进 + 记录时间戳」⇒ 延迟是**量出来的**，不是算出来的。
    ★★★ `corrupt="overadmit"` 只用于 `D.4` 的守卫实验：准入上限变成 `B + 1`。

    返回 `(rec, makespan, n_steps)`；`rec[i]` 含 `ttft` / `itl`（列表）/ `done`。
    """
    lim = B + 1 if corrupt == "overadmit" else B
    N = len(arrivals)
    nxt = 0
    queue = []
    active = []
    rec = [{"arrival": arrivals[i], "first": None, "last": None, "itl": [], "done": None}
           for i in range(N)]
    t = 0.0
    n_steps = 0
    while nxt < N or queue or active:
        while nxt < N and arrivals[nxt] <= t + 1e-12:
            queue.append(nxt)
            nxt += 1
        while queue and len(active) < lim:
            active.append(queue.pop(0))
        if not active and not queue:
            if nxt < N:
                t = arrivals[nxt]
                continue
            break
        need = [i for i in active if rec[i]["first"] is None]
        if need:
            t += pref
            n_steps += 1
            for i in need:
                rec[i]["first"] = t
                rec[i]["last"] = t
            continue
        n = len(active)
        t += step_cost(n, alpha, beta)
        n_steps += 1
        for i in list(active):
            rec[i]["itl"].append(t - rec[i]["last"])
            rec[i]["last"] = t
            rec[i]["decoded"] = rec[i].get("decoded", 0) + 1
            if rec[i]["decoded"] >= D - 1:
                rec[i]["done"] = t
                active.remove(i)
    for i in range(N):
        rec[i]["ttft"] = rec[i]["first"] - rec[i]["arrival"]
    return rec, t, n_steps


# ===========================================================================
# 2. 参照（II）：**闭式（纯代数）** —— 与模拟器**无共享代码**
# ===========================================================================
def phases_for(N, B):
    """★ 相位的**大小序列**（每相位最多 `B` 条）。★ 纯整数运算，不碰模拟器。"""
    out = []
    rem = N
    while rem > 0:
        k = B if rem >= B else rem
        out.append(k)
        rem -= k
    return out


def closed_form(arrivals, B, D, pref, alpha, beta):
    """★★★ **参照实现（II）**：闭式 —— 从相位数与步耗时模型**直接算**。

    ★★ 它**不 import 模拟器、不调用模拟器的任何函数、不含任何事件循环**（`D.8` 核）。
    ★ 前提（**只对「同时到达」成立**）：所有请求同时到达 ⇒ 引擎按相位满批推进，相位内 `n` 恒定。
    """
    N = len(arrivals)
    ph = phases_for(N, B)
    ttft = {}
    itl = {}
    t0 = arrivals[0] if arrivals else 0.0
    cursor = t0
    idx = 0
    for n in ph:
        c = step_cost(n, alpha, beta)
        for _ in range(n):
            ttft[idx] = (cursor + pref) - arrivals[idx]
            itl[idx] = c
            idx += 1
        cursor += pref + (D - 1) * c
    return ttft, itl, cursor


def throughput(n_req, D, makespan):
    """★ 吞吐 = **总 token 数 / 总时间**（★ 口径必须一起写：分母是 makespan）。"""
    return n_req * D / makespan


# ===========================================================================
# 3. A 类 —— ★★★ 两条独立路径（模拟器 vs 闭式）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **两个延迟**：离散事件模拟器（被测）vs 闭式（参照）")
    print("=" * 78)
    N, D = specs.LAT_N_REQ, specs.LAT_DEC_TOKENS
    pref, a, b = specs.LAT_PREFILL_COST, specs.LAT_STEP_FIXED, specs.LAT_STEP_PER_REQ
    arrivals = [0.0] * N
    B = specs.LAT_BATCH_GRID[2]
    print("  ★★★ 被测 = `simulate()`（按步推进，量事件时间戳）；参照 = `closed_form()`（纯代数）。")
    print("  ★★ 两者**无共享代码**（`D.8` 在源码层核）；★ prefill 记为一次独立步（简化，见模块头）。")
    rec, mk, ns = simulate(arrivals, B, D, pref, a, b)
    ttft_cf, itl_cf, mk_cf = closed_form(arrivals, B, D, pref, a, b)

    # ── A.1 ★★★ inter-token latency ─────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **inter-token latency**：模拟器量出的逐请求 ITL == 参照的 `α + β·n_相位` ──")
    bad = [i for i in range(N) if rec[i]["itl"] and any(abs(x - itl_cf[i]) > 1e-12 for x in rec[i]["itl"])]
    nsamp = sum(len(rec[i]["itl"]) for i in range(N))
    print("     B = %d · 请求 %d 条 · ITL 样本 %d 个 · 不一致 = %d 条" % (B, N, nsamp, len(bad)))
    print("     ★★ 样例（前 3 条）：模拟器 %s · 参照 %s"
          % ([round(rec[i]["itl"][0], 4) for i in range(3)], [round(itl_cf[i], 4) for i in range(3)]))
    check("A.1", not bad and nsamp == N * (D - 1) and abs(rec[0]["itl"][0] - (a + b * B)) < 1e-12,
          "★★★ **判据成立**：**%d 条请求 × %d 个间隔 = %d 个 ITL 样本**上，"
          "模拟器量出的值与参照的 `α + β·n = %.4f + %.4f×%d = %.4f` **逐位一致** —— "
          "★★ 一面按事件推进、一面纯代数，两条独立路径会合"
          % (N, D - 1, nsamp, a, b, B, a + b * B),
          "**ITL 不一致（或样本数不是 %d）** —— ★ 先查 `simulate` 的准入与 `step_cost`" % (N * (D - 1)))

    # ── A.2 ★★★ makespan ───────────────────────────────────────────────
    print()
    print("  ── A.2 ★★★ **makespan**：模拟器量出的总时间 == 参照的 `Σ_k (prefill + (D−1)·步耗时)` ──")
    ph = phases_for(N, B)
    print("     相位数 = %d · 每相位 %d 条 · 模拟器 makespan = %.6f · 参照 = %.6f"
          % (len(ph), B, mk, mk_cf))
    print("     ★★ 调度步数 = %d（= 相位数 × (1 个 prefill 步 + %d 个 decode 步)）" % (ns, D - 1))
    check("A.2", abs(mk - mk_cf) < 1e-12 and ns == len(ph) * D
          and abs(mk - (len(ph) * (pref + (D - 1) * (a + b * B)))) < 1e-12,
          "★★★ **判据成立**：makespan **%.6f**（模拟器）== **%.6f**（参照）—— ★★ 调度步数 %d 也对上"
          % (mk, mk_cf, ns),
          "**makespan 不一致（或步数不是 %d）** —— ★ 先查 `phases_for` / `simulate` 的推进" % (len(ph) * D))

    # ── A.3 ★★★ TTFT ───────────────────────────────────────────────────
    print()
    print("  ── A.3 ★★★ **TTFT**：模拟器量出的逐请求 TTFT == 参照的 `相位起点 + prefill` ──")
    b3 = [i for i in range(N) if abs(rec[i]["ttft"] - ttft_cf[i]) > 1e-12]
    print("     ★★ 首条请求 TTFT = %.6f（模拟器）· 末条 = %.6f" % (rec[0]["ttft"], rec[-1]["ttft"]))
    print("     ★★ 不一致 = %d 条；★ TTFT 只依赖**相位起点**，与相位内的步耗时无关" % len(b3))
    check("A.3", not b3 and abs(rec[0]["ttft"] - pref) < 1e-12,
          "★★★ **判据成立**：%d 条请求的 TTFT 两条路径**逐位一致**（首条 = `prefill` = %.4f —— "
          "★ 它排在第一相位，没有前序相位要等）" % (N, pref),
          "**TTFT 不一致** —— ★ 先查 `phases_for` 与模拟器的 prefill 记账")


# ===========================================================================
# 4. B 类：**两个延迟的冲突**
# ===========================================================================
def _table(B_grid, arrivals_kind):
    N, D = specs.LAT_N_REQ, specs.LAT_DEC_TOKENS
    pref, a, b = specs.LAT_PREFILL_COST, specs.LAT_STEP_FIXED, specs.LAT_STEP_PER_REQ
    if arrivals_kind == "all_at_once":
        arr = [0.0] * N
    else:
        arr = [i * specs.LAT_ARRIVE_PERIOD for i in range(N)]
    rows = []
    for B in B_grid:
        rec, mk, _ns = simulate(arr, B, D, pref, a, b)
        itl = [x for i in range(N) for x in rec[i]["itl"]]
        rows.append((B, sum(itl) / len(itl), throughput(N, D, mk), sum(rec[i]["ttft"] for i in range(N)) / N, mk))
    return rows


def part_B():
    print()
    print("=" * 78)
    print("B · **两个延迟的冲突**：批越大 ⇒ ITL 越差、吞吐越好（同一个旋钮，两个方向）")
    print("=" * 78)
    N, D = specs.LAT_N_REQ, specs.LAT_DEC_TOKENS
    a, b = specs.LAT_STEP_FIXED, specs.LAT_STEP_PER_REQ
    rows = _table(specs.LAT_BATCH_GRID, "all_at_once")

    # ── B.1 ★★ ITL vs 吞吐 ─────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **同一个旋钮**：在飞上限 `B` 同时决定 ITL 与吞吐 ──")
    print("     %-8s %-14s %-14s %-14s" % ("B", "ITL(均值)", "吞吐(token/单位时间)", "makespan"))
    for B, itl, thr, _ttft, mk in rows:
        print("     %-8d %-14.4f %-14.4f %-14.2f" % (B, itl, thr, mk))
    itl_m = all(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1))
    thr_m = all(rows[i][2] < rows[i + 1][2] for i in range(len(rows) - 1))
    # ★★★ **整表逐格钉住**（D1）：正文 §2 的表把 7 行当「本机实测值」列出 ⇒ ★ **每一格都要有 `==` 复核**，
    #   只查单调性与端点等于**给错的那几格开后门**。★ `rows` 来自 `simulate()`，这里是**闭式**复核（两个量）。
    pref = specs.LAT_PREFILL_COST
    bad_cells = []
    for (B, itl, thr, _ttft, mk) in rows:
        nph = N // B                     # ★ 每个 `B` 都整除 `N` ⇒ 相位满、相位内 `n` 恒为 `B`
        c = a + b * B
        mk_exp = nph * (pref + (D - 1) * c)
        if not (abs(itl - c) < 1e-12 and abs(thr - N * D / mk_exp) < 1e-12 and abs(mk - mk_exp) < 1e-12):
            bad_cells.append(B)
    print("     ★★ ITL **单调升** = %s · 吞吐 **单调升** = %s · **整表 %d 行 × 3 列逐格 == 闭式** = %s"
          % (itl_m, thr_m, len(rows), not bad_cells))
    print("     ★★★ **读法**：这两行**同向变动** ⇒ ★ **没有任何一个 `B` 能同时让两头都最优** ——")
    print("        ITL 最小在 `B = %d`（%.4f），吞吐最大在 `B = %d`（%.4f）。"
          % (rows[0][0], rows[0][1], rows[-1][0], rows[-1][2]))
    check("B.1", itl_m and thr_m and not bad_cells and len(rows) == len(specs.LAT_BATCH_GRID)
          and abs(rows[0][1] - (a + b * 1)) < 1e-12
          and abs(rows[-1][1] - (a + b * specs.LAT_BATCH_GRID[-1])) < 1e-12,
          "★★ **判据成立**：ITL 从 **%.4f**（`B=1`）升到 **%.4f**（`B=%d`），吞吐同时从 **%.4f** 升到 **%.4f**"
          " —— ★★ 两行**同向** ⇒ **冲突成立**；★★ **整表 %d 行 × 3 列（ITL/吞吐/makespan）逐格 == 闭式**"
          % (rows[0][1], rows[-1][1], specs.LAT_BATCH_GRID[-1], rows[0][2], rows[-1][2], len(rows)),
          "**单调性、端值或表中某一格被动过** —— ★ 先查 `LAT_BATCH_GRID` / `LAT_STEP_FIXED` / `LAT_STEP_PER_REQ`")

    # ── B.2 ★ **两端的闭式**（免费守卫：端点各有唯一解）──────────────────
    print()
    print("  ── B.2 ★ **两个端点的闭式**（`B = 1` 与 `B = N` 各有唯一解）──")
    pref = specs.LAT_PREFILL_COST
    mk1 = N * (pref + (D - 1) * (a + b * 1))
    mkN = pref + (D - 1) * (a + b * N)
    print("     `B = 1` ⇒ makespan = `N·(prefill + (D−1)(α+β))` = %d × (%.1f + %d×%.1f) = **%.2f**" % (N, pref, D - 1, a + b, mk1))
    print("     `B = N` ⇒ makespan = `prefill + (D−1)(α+β·N)` = %.1f + %d×%.1f = **%.2f**" % (pref, D - 1, a + b * N, mkN))
    thr1, thrN = throughput(N, D, mk1), throughput(N, D, mkN)
    print("     ⇒ 吞吐范围 `[%.4f, %.4f]`（★ **跨 %.2f 倍**）" % (thr1, thrN, thrN / thr1))
    meas1 = dict((r[0], r) for r in rows)
    check("B.2", abs(meas1[1][2] - thr1) < 1e-12 and abs(meas1[N][2] - thrN) < 1e-12
          and abs(meas1[1][4] - mk1) < 1e-12 and abs(meas1[N][4] - mkN) < 1e-12,
          "★ **判据成立**：两端的吞吐/makespan **逐位等于闭式**（`B=1`: %.4f / %.2f；`B=N`: %.4f / %.2f）"
          " —— ★★ 端点各有唯一解，且模拟器与闭式在**两端同时**对上"
          % (thr1, mk1, thrN, mkN),
          "**端点闭式被动过** —— ★ 先查 `step_cost` / `throughput` / `phases_for`")

    # ── B.3 ★★ TTFT 的分解与【两个工作负载】 ───────────────────────────
    print()
    print("  ── B.3 ★★ **TTFT 的分解**（排队 + prefill + 首 token）与两个工作负载 ──")
    arr_all = [0.0] * N
    rec, _mk, _ns = simulate(arr_all, specs.LAT_BATCH_GRID[2], D, pref, a, b)
    comp_ok = all(abs(rec[i]["ttft"] - (rec[i]["first"] - rec[i]["arrival"])) < 1e-12 for i in range(N))
    ge_pref = all(rec[i]["ttft"] >= pref - 1e-12 for i in range(N))
    print("     ★★★ TTFT = **从到达**到**首个输出 token** 的时间（vLLM 的官方定义只算后两段，见 `C.2`）。")
    print("     ★★ 分解自洽 = %s ；★★ **免费守卫**：`TTFT ≥ prefill` = %s（★ 首 token 不可能早于 prefill 走完）"
          % (comp_ok, ge_pref))
    rows_cont = _table(specs.LAT_BATCH_GRID, "periodic")
    print("     %-8s %-18s %-18s" % ("B", "TTFT(同时到达)", "TTFT(持续到达)"))
    for r1, r2 in zip(rows, rows_cont):
        print("     %-8d %-18.4f %-18.4f" % (r1[0], r1[3], r2[3]))
    all_noninc = all(rows[i][3] >= rows[i + 1][3] - 1e-12 for i in range(len(rows) - 1))
    cont_noninc = all(rows_cont[i][3] >= rows_cont[i + 1][3] - 1e-12 for i in range(len(rows_cont) - 1))
    print("     ★★★ **本文模型的实际方向（数值驱动，不是预设）**：同时到达 ⇒ TTFT **非增** = %s；" % all_noninc)
    print("        持续到达 ⇒ TTFT **非增** = %s。" % cont_noninc)
    print("     ★★★ **一条【与规划不同】的如实报告**：**本模型【没有】给出「批越大 TTFT 越差」** ——")
    print("        因为本文把 **prefill 记成独立的一步**（不与 decode 争算力）⇒ **批越大 = 槽位越多 = 排队越少**，")
    print("        TTFT 反而下降。★ **TTFT 上升需要 prefill 与 decode 【争算力】** ——")
    print("        那套耦合模型（「取较慢者」）的正文在 `batching-and-scheduling/02-chunked-prefill`，本文**不重推**。")
    check("B.3", comp_ok and ge_pref and all_noninc and cont_noninc,
          "★★ **判据成立**：TTFT 分解自洽、**且 `TTFT ≥ prefill` 逐条成立**（免费守卫）；"
          "★ **两个工作负载下 TTFT 都【非增】**（同时到达 `%.2f → %.2f`；持续到达 `%.2f → %.2f`）——"
          "★★ **方向是量出来的，与规划预设的「升」不同**，原因已在正文写明（prefill 不与 decode 争算力）"
          % (rows[0][3], rows[-1][3], rows_cont[0][3], rows_cont[-1][3]),
          "**TTFT 的分解、下界或方向被动过** —— ★ 先查 `LAT_ARRIVE_PERIOD` 与 prefill 记账")



# ===========================================================================
# 5. C 类：**禁写的真现场**（`F9`）与两个口径 / 边界
# ===========================================================================
def is_official_vllm_term(term):
    """★ **`term` 是 vLLM 的【官方词】吗？** —— ★ 只有 **inter-token latency** 是。

    ★★★ 那个社区缩写是**社区常用词**（★ 在 vLLM 的 metrics 页面上**出现 0 处**）
      ⇒ 把它当成官方词 ⇒ **被拒**（`F9`）。
    """
    return term in specs.VLLM_ITL_DEFINITION_QUOTE


def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**社区缩写当官方词** · **两个 TTFT 口径** · **两端边界**")
    print("=" * 78)

    # ── C.1 ★★★ 把社区缩写当 vLLM 官方词（`F9`）────────────────────────
    print()
    print("  ── C.1 ★★★ **把社区缩写当成 vLLM 的官方词**（`F9`）──")
    abbr = "TP" + "OT"
    print("     vLLM 页面上该缩写的出现次数 = **%d**（★ 必须是 `0`）"
          % specs.VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES)
    print("     vLLM 的官方词逐字：`%s`" % specs.VLLM_ITL_DEFINITION_QUOTE)
    print("     ★★★ **反例（把社区缩写当官方词）**：`is_official_vllm_term(%r)` ⇒ **%s**（应 `False` —— **被拒**）"
          % (abbr, is_official_vllm_term(abbr)))
    print("     ★ **正例**：`is_official_vllm_term(\"inter-token latency\")` ⇒ **%s**（应 `True`）"
          % is_official_vllm_term("inter-token latency"))
    print("     ★★ **读法**：`%s`（`specs.TPOT_COMMUNITY_ABBREVIATION_NOTE` 逐字）" % "社区缩写")
    check("C.1", (not is_official_vllm_term(abbr)) and is_official_vllm_term("inter-token latency")
          and specs.VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES == 0
          and "inter-token latency" in specs.VLLM_ITL_DEFINITION_QUOTE
          and "社区常用缩写" in specs.TPOT_COMMUNITY_ABBREVIATION_NOTE,
          "★★★ **判据成立**：把社区缩写当 **vLLM 的官方词**会被拒（**反例 → 被拒**）—— "
          "该词在 vLLM 页面上 **%d 处**；官方词是 **inter-token latency**（`F9`）"
          % specs.VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES,
          "**缩写计数或官方词被动过** —— ★ 先查 `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES`（须为 0）")

    # ── C.2 ★★★ 两个 TTFT 口径 ─────────────────────────────────────────
    print()
    print("  ── C.2 ★★★ **两个 TTFT 口径**：vLLM 的从 `scheduled` 起，本库的含排队 ──")
    print("     vLLM 逐字：`%s`" % specs.VLLM_TTFT_DEFINITION_QUOTE)
    print("     ★★★ **起算点是「request was scheduled」** ⇒ **不含排队**；而本库口径是「**从到达**到首 token」"
          " ⇒ **含排队**。")
    print("     ★★ 另一条 vLLM 逐字（`generation_time`）：`%s`" % specs.VLLM_GENERATION_TIME_DEFINITION_QUOTE)
    q = specs.VLLM_TTFT_DEFINITION_QUOTE
    genq = specs.VLLM_GENERATION_TIME_DEFINITION_QUOTE
    print("     ★★ 判定：vLLM 的 TTFT 定义**含** `%s`、**不含**「到达」；两个口径**不是同一个量** = %s"
          % (VLLM_TTFT_START_TOKEN, ("arriv" not in q.lower())))
    check("C.2", VLLM_TTFT_START_TOKEN in q and "arriv" not in q.lower()
          and "Excludes both queue wait and prefill" in genq
          and VLLM_TTFT_EXCLUDES_QUEUE_NOTE.endswith("本库口径含排队"),
          "★★★ **判据成立**：vLLM 的 TTFT 起算于 `%s`（**不含排队**）；本库口径从**到达**起（**含排队**）——"
          "★ **两个同名不同指的量**（`generation_time` 还显式排除了排队与 prefill/TTFT）" % VLLM_TTFT_START_TOKEN,
          "**TTFT 口径引文被动过** —— ★ 先查 `VLLM_TTFT_DEFINITION_QUOTE`")

    # ── C.3 ★★ 两端边界 ────────────────────────────────────────────────
    print()
    print("  ── C.3 ★★ **两端边界**：`B = 1`（逐条）与 `B ≥ N`（单相位）──")
    N, D = specs.LAT_N_REQ, specs.LAT_DEC_TOKENS
    pref, a, b = specs.LAT_PREFILL_COST, specs.LAT_STEP_FIXED, specs.LAT_STEP_PER_REQ
    arr = [0.0] * N
    r1, _m1, _n1 = simulate(arr, 1, D, pref, a, b)
    rN, _mN, _nN = simulate(arr, N, D, pref, a, b)
    itl1 = [x for i in range(N) for x in r1[i]["itl"]][0]
    itlN = [x for i in range(N) for x in rN[i]["itl"]][0]
    print("     `B = 1`  ⇒ ITL = **%.4f**（`α + β`，最小）；每相位 1 条 ⇒ 相位数 = %d" % (itl1, N))
    print("     `B = N`  ⇒ ITL = **%.4f**（`α + β·N`，最大）；**单相位** ⇒ 首条 TTFT = prefill = **%.4f**"
          % (itlN, rN[0]["ttft"]))
    print("     ★★ 两端都落在模型端点：`B = 1` 时 ITL 最小、`B ≥ N` 时相位数 = 1（TTFT 最小）")
    check("C.3", abs(itl1 - (a + b)) < 1e-12 and abs(itlN - (a + b * N)) < 1e-12
          and abs(rN[0]["ttft"] - pref) < 1e-12 and abs(r1[0]["ttft"] - pref) < 1e-12
          and len(phases_for(N, 1)) == N and len(phases_for(N, N)) == 1,
          "★★ **判据成立**：`B=1` ⇒ ITL **%.4f**（最小）；`B=N` ⇒ ITL **%.4f**（最大）；"
          "两端 TTFT 恰为 `prefill` = **%.4f**（各自的**首相位**都没有前序要等）" % (itl1, itlN, pref),
          "**端点值被动过** —— ★ 先查 `step_cost` 与 `phases_for`")


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无任何引擎**。")
    print("      ★★ 但**本程序的 A 类是真的**：**模拟器 vs 闭式**两条独立路径。")
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

    ok_model = (specs.LAT_N_REQ > 1 and specs.LAT_DEC_TOKENS > 1
                and specs.LAT_PREFILL_COST > 0 and specs.LAT_STEP_FIXED > 0
                and specs.LAT_STEP_PER_REQ > 0 and specs.LAT_ARRIVE_PERIOD > 0)
    print("     ★★ 模型参数为正、`N > 1`、`D > 1` = %s" % ok_model)
    check("D.3", specs.SERVING_INFRA_FETCH_DAY == "2026-10-09" and ok_model
          and specs.LAT_N_REQ == 60 and specs.LAT_DEC_TOKENS == 12
          and specs.LAT_BATCH_GRID == (1, 2, 4, 10, 20, 30, 60)
          and all(specs.LAT_N_REQ % B == 0 for B in specs.LAT_BATCH_GRID)
          and specs.UNSOURCED == (),
          "★ **载体元数据 + 模型参数逐字钉住**：抓取日 `%s`；`N=%d` · `D=%d` · 网格 `%s`"
          "（★ 每个 `B` 都整除 `N` ⇒ 相位是满的）；★ `UNSOURCED == ()` 一并钉住"
          % (specs.SERVING_INFRA_FETCH_DAY, specs.LAT_N_REQ, specs.LAT_DEC_TOKENS, specs.LAT_BATCH_GRID))

    check("D.3b", specs.LAT_PREFILL_COST == 5.0 and specs.LAT_STEP_FIXED == 2.0
          and specs.LAT_STEP_PER_REQ == 0.5 and specs.LAT_ARRIVE_PERIOD == 12.0,
          "★ **模型参数逐字**：`prefill=%.1f` · `α=%.1f` · `β=%.1f` · 到达间隔 `%.1f`"
          % (specs.LAT_PREFILL_COST, specs.LAT_STEP_FIXED, specs.LAT_STEP_PER_REQ, specs.LAT_ARRIVE_PERIOD),
          "**模型参数被动过** —— ★ 先查 `LAT_*`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：A 类存在性 · 守卫实验 · 独立性 · 术语纪律
# ===========================================================================
def _class_names(cls):
    src = textwrap.dedent(inspect.getsource(cls))
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


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
    print("D · **A 类存在性** · 守卫实验 · 独立性 · 引文逐字 · 术语纪律")
    print("=" * 78)

    # ── D.0 ★★★ 机械证明「本文件【有】A 类」─────────────────────────────
    n_a = _count_a_checks_in_source()
    print("  ── D.0 ★★★ **机械证明**：AST 扫本文件，`check(\"A...\")` 的调用数 = **%d** ──" % n_a)
    check("D.0", n_a >= EXPECTED_MIN_A,
          "★★★ **判据成立**：本文件里 **A 类判据有 %d 条**（≥ %d）—— ★ 本主题**有真 A 类**"
          "（模拟器 vs 闭式，两条独立路径）" % (n_a, EXPECTED_MIN_A),
          "★★ 本文件里 A 类判据少于 %d 条 —— ★ 那与「本主题有 A 类」**自相矛盾**" % EXPECTED_MIN_A)

    # ── D.4 ★★★ 守卫实验：准入上限改成 B+1 ──────────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验**：把模拟器的准入上限改成 `B + 1` ──")
    N, D = specs.LAT_N_REQ, specs.LAT_DEC_TOKENS
    pref, a, b = specs.LAT_PREFILL_COST, specs.LAT_STEP_FIXED, specs.LAT_STEP_PER_REQ
    B = specs.LAT_BATCH_GRID[1]
    arr = [0.0] * N
    good, _mg, _ng = simulate(arr, B, D, pref, a, b)
    bad, _mb, _nb = simulate(arr, B, D, pref, a, b, corrupt="overadmit")
    itl_good = [x for i in range(N) for x in good[i]["itl"]]
    itl_bad = [x for i in range(N) for x in bad[i]["itl"]]
    print("     构造：`B = %d`、`N = %d`；正确版首个 ITL = %.4f · 坏法首个 ITL = %.4f"
          % (B, N, itl_good[0], itl_bad[0]))
    print("     ★★★ **读法**：多准入一条 ⇒ 第一步的在飞数从 `%d` 变成 `%d` ⇒ 步耗时变大 ⇒ `A.1` 的等式不成立。"
          % (B, B + 1))
    check("D.4", itl_good != itl_bad and itl_bad[0] != itl_good[0]
          and abs(itl_bad[0] - (a + b * (B + 1))) < 1e-12,
          "★★★ **守卫成立**：准入上限 `%d → %d` 后，首个 ITL **从 %.4f 变成 %.4f** ⇒ "
          "★★ **`A.1` 对「准入上限 = B」这条规则是敏感的**" % (B, B + 1, itl_good[0], itl_bad[0]),
          "**守卫失效** —— ★ 多准入一条后 ITL 居然不变（那 `A.1` 抓不到它）")

    # ── D.5 ★★★ 逐字引文钉住（★ 关键引文一律 `==`，不用 `in`）──────────────
    check("D.5", specs.VLLM_TTFT_DEFINITION_QUOTE
          == ("Time from when the request was scheduled until the first output token was generated (TTFT).")
          and specs.VLLM_ITL_DEFINITION_QUOTE
          == ("Mean inter-token latency (average time between successive output tokens) "
              "during the decode phase.")
          and specs.VLLM_ITL_METRIC_QUOTE == "Histogram of inter-token latency in seconds."
          and specs.VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE
          == "Histogram of time_per_output_token_seconds per request."
          and specs.VLLM_GENERATION_TIME_DEFINITION_QUOTE
          == ("Decode time: time from the first output token to the last output token. "
              "Excludes both queue wait and prefill/TTFT.")
          and specs.TPOT_COMMUNITY_ABBREVIATION_NOTE
          == ("TPOT 是社区常用缩写（time per output token）；vLLM 官方页面不用 TPOT，其词是 "
              "inter-token latency（mean_itl_ms / request_time_per_output_token_seconds）"),
          "★★★ **六条引文逐字钉住**（★ 三条指标的说明 + TTFT 的起算点 + `generation_time` 的排除项 + "
          "社区缩写的消歧记录）—— ★★ **每一条都用 `==`，不用 `in`**（`in` 抓不到**末尾追加**）\n"
          "        ★ 载体 = vLLM 官方文档（`usage/metrics/` 与 `features/per_request_metrics/`），抓取日 %s"
          % specs.SERVING_INFRA_FETCH_DAY,
          "**某条引文被动过** —— ★ 先查 `VLLM_*_QUOTE` / `TPOT_COMMUNITY_ABBREVIATION_NOTE`")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.LAT_N_REQ == 60 and specs.LAT_DEC_TOKENS == 12
          and specs.LAT_PREFILL_COST == 5.0 and specs.LAT_STEP_FIXED == 2.0
          and specs.LAT_STEP_PER_REQ == 0.5 and specs.LAT_ARRIVE_PERIOD == 12.0
          and specs.LAT_BATCH_GRID == (1, 2, 4, 10, 20, 30, 60)
          and specs.VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES == 0
          and VLLM_TTFT_START_TOKEN == "scheduled"
          and specs.SERVING_INFRA_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：六个模型参数 · 批网格 7 档（**逐元素**）· 缩写计数 `0` · "
          "TTFT 起算点字面量 · 抓取日；★ `UNSOURCED == ()`",
          "**某个常量被动过** —— ★ 先查 `LAT_*` / `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES`")

    # ── D.8 ★★★ 独立性（源码级）：模拟器与参照互不引用 ─────────────────
    sim_names = _func_names(simulate)
    cf_names = _func_names(closed_form)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     `simulate` 里出现 `closed_form` = %s ；`closed_form` 里出现 `simulate` = %s"
          % ("closed_form" in sim_names, "simulate" in cf_names))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1`/`A.2`/`A.3` 与 `D.4` 的**行为**兜底。")
    print("     ★★ **如实标注（S1）**：`A.1`/`A.2`/`A.3` **共用同一对**被测/参照"
          "（都是 `simulate()` vs 同一个 `closed_form()`）—— ★ **不是三条独立参照**，"
          "而是**同一个被测**量了**三个不同的量**（ITL / makespan / TTFT）；★ 真正兜底的是 `D.4` 的**行为守卫**。")
    check("D.8", "closed_form" not in sim_names and "simulate" not in cf_names
          and "step_cost" in sim_names and "step_cost" in cf_names
          and "phases_for" not in sim_names,
          "★★ **源码级线索成立**：**被测（事件模拟器）与参照（闭式）互不引用**（★ **两侧都审**）——"
          " ★★ 于是「结果相同」**不是恒真**，而是**两条独立路径的会合**"
          "（★ 两者**共享** `step_cost` —— 那是**模型假设**，不是路径）",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★★ 术语纪律（★ 形态 A 的根治：计数 == 豁免清单）──────────────
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
          "（★ 清单之外的任何一处裸写都会让等式不等）"
          % (TERM_KV_CACHE, TERM_SCHEDULER, LIMIT_QUOTE, n_bare_terms, n_exempt_terms),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写**")


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
    print("★★★ **A = %d/%d** —— ★★ **本程序有真 A 类**：**离散事件模拟器 vs 闭式**（两条独立路径）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：模型与算例是 **`[E]` 级（本库实现）**；TTFT / inter-token latency 的口径取自 vLLM 官方文档（`[B]`）。")
    print("★ 本程序只核对「**两个延迟的算术与它们的关系**」—— **不是**任何真实引擎的延迟；")
    print("  ★★ **chunked prefill 的分块数权衡**在 `batching-and-scheduling/02`，本程序**不重推**。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_latency_and_slo.py —— 两个延迟（TTFT / inter-token latency）与它们的冲突")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无任何引擎。")
    print("  ★★★ **本程序有真 A 类**（模拟器 vs 闭式）；★ 模型是一条**写明的简化**。")
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
