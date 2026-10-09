#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_scaling_and_observability.py —— 扩缩容 / 路由（**前缀感知的路由**）/ 可观测性（**指标口径**）。

配套文档：`03-scaling-and-observability.md`。

★★★ **本程序承接 `50-performance/README.md:49` 明文让渡的「serving 栈」** ——
  那里逐字写「**不重讲 serving 栈**」，而 serving 栈就是本主题。

★★★ **本程序【有真 A 类】**（两条，各自两条独立路径）—— ★★ **不能整主题标「无 A」**：
  ★ 「**排队与负载均衡本质是可独立参照的算法**」：
  · `A.1` **前缀感知的路由**：一个**事件驱动**的路由模拟（逐请求）vs **闭式** `N − 不同模板数`（纯计数）；
  · `A.2` **排队**：离散事件排队模拟（逐到达、逐服务）vs **M/M/1 的闭式**（`W_q = ρ/(μ(1−ρ))`）。
  ★★ **只有「真实网络 / 多机的行为」不可测**（本机单机、无真网络 —— `§0.7 D7`）。
  ★★ **守卫**（`D.4`）：把路由策略的「优先发给持有者」改坏（改成随机）⇒ `A.1` 必须失败。

★★★ **本程序只写【路由与观测的算术】** —— ★ 别处的正文一律回链（`P37`）：
  · ★★ **KV 缓存侧的「`hash → 块` / 命中率 / LRU」** → 本域 `kv-cache/03-prefix-and-eviction`
    （★★ **本程序的「路由命中率」是【另一个量】**：它是**请求被分配到哪个副本**这一动作的属性，
      不是缓存内部的查找结果 —— **同名不同指**，`C.2` 会把它写成断言）；
  · **请求调度器的机制** → 本域 `batching-and-scheduling/01`；
  · **两个延迟的定义** → 本域 `01-latency-and-slo`。

★★★ **本程序的禁写清单**（★ **每条纪律都配一条断言** —— 形态 A 的根治）：
  · `F5`（**不写**那个前缀复用开关「默认开启」—— 取数到的页面**未声明**默认值）⇒ ★ **由 `D.6` 机械守住**
    （`specs.APC_DEFAULT_DECLARED_IN_SOURCE is False`）；
  · ★★ **不裸写那个社区缩写当 vLLM 的官方词**（该词在 vLLM 页面上 **0 处**）⇒ ★ 由 `C.1` 守住。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **前缀感知的路由**：逐请求模拟的命中数 == 闭式 `N − 不同模板数`（★ 前提 `R·C ≥ T`）；
  · `A.2` ★★★ **排队**：离散事件排队的平均等待 == M/M/1 闭式 `ρ/(μ(1−ρ))`（容差 `3%`）；
  · `B.1` ★★ 路由命中率 vs 缓存容量（两种策略）；
  · `B.2` ★★ 排队等待 vs 利用率（`ρ → 1` 时发散）；
  · `B.3` ★★ 扩缩容：达到目标利用率所需的副本数；
  · `C.1` ★★★ **指标口径**：vLLM 的 TTFT 不含排队、`generation_time` 排除排队+prefill；
  · `C.2` ★★★ **前缀感知路由的两个反例**：无共享前缀 ⇒ 无收益；单一前缀 ⇒ **负载集中**；
  · `C.3` ★★ **均值 vs 分位**：均值达标 ≠ 尾部达标；
  · `D.4` ★★★ 守卫实验：把路由改回随机 ⇒ 断言必须失败。
"""
import ast
import collections
import inspect
import math
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

# ★★ **路由命中率的定义**（★ 与 `kv-cache/03` 的**缓存命中率**是**两个量**）：
ROUTE_HIT_DEF = "路由命中 = 请求的前缀在被【分配到的那个副本】的 KV 缓存里"


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

EXPECTED_MIN_A = 2


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
# 1. A.1：**前缀感知的路由**（逐请求模拟）vs **随机路由**（参照）
# ===========================================================================
class Replica:
    """★ 一个副本的**前缀缓存**（容量 = 条数，LRU）—— ★ 机制细节**在 `kv-cache/03`**，此处只用它的**接口**。"""

    def __init__(self, cap):
        self.cap = cap
        self.cache = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
        self.served = 0

    def has(self, t):
        return t in self.cache

    def serve(self, t):
        """★ 返回 `True` = **路由命中**（前缀已在**这个**副本里，prefill 可以省）。"""
        self.served += 1
        if t in self.cache:
            self.cache.move_to_end(t)
            self.hits += 1
            return True
        self.misses += 1
        self.cache[t] = True
        self.cache.move_to_end(t)
        while len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return False


def route_trace(n_req, n_template, rng):
    """★ 请求流：每条请求取一个**模板**（= 它的共享前缀）。★ 用固定种子 ⇒ 可复现。"""
    return [int(x) for x in rng.integers(0, n_template, size=n_req)]


def run_routing(trace, n_replica, cap, policy, corrupt=False):
    """★★★ **被测实现（I）**：**逐请求**的路由模拟。

    · `policy="prefix_aware"`：优先发给**已经持有**该前缀的副本（★ 由路由器的轻量查找给出）；
    · `policy="random"`：**参照** —— 与请求内容无关地轮转分配（★ 与 `prefix_aware` 无共享代码）。

    ★★★ `corrupt=True` 只用于 `D.4`：把 `prefix_aware` **改坏成随机**。
    """
    reps = [Replica(cap) for _ in range(n_replica)]
    rr = 0
    hits = 0
    for t in trace:
        if policy == "random" or corrupt:
            r = rr % n_replica
            rr += 1
        else:
            holders = [i for i in range(n_replica) if reps[i].has(t)]
            if holders:
                r = holders[0]
            else:
                r = min(range(n_replica), key=lambda i: (len(reps[i].cache), i))
        if reps[r].serve(t):
            hits += 1
    return hits, [x.served for x in reps], reps


def prefix_hits_closed(trace, n_replica, cap):
    """★★★ **参照实现（I）**：**前缀感知路由**的命中数**闭式**。

    ★★ 前提 `R·C ≥ T`（`R` = 副本数、`C` = 每副本容量、`T` = **不同**模板数）：
      ⇒ ★★★ **永无驱逐** —— 若 `R` 台副本都装满 `C` 个，则**已被持有的不同模板数 = `R·C ≥ T`**
      ⇒ **所有模板都有主** ⇒ 不会出现「无主的模板」触发**回退分配** ⇒ **不会挤掉任何条目**。
    ⇒ ★★★ 于是**每个模板的首次出现是唯一的缺失** ⇒ 命中数 = `N − (#不同模板)`。
    ★ 纯计数，**不含任何 LRU / 事件循环**（`D.8` 在源码层核两条路径不互相引用）。
    """
    return len(trace) - len(set(trace))


# ===========================================================================
# 2. A.2：**排队**（离散事件）vs **M/M/1 闭式**
# ===========================================================================
def sim_queue(lam, mu, n_sample, seed):
    """★★★ **被测实现（II）**：`M/M/1` 的**离散事件**排队模拟 —— **逐到达、逐服务**推进。

    ★ 指数到达间隔（率 `λ`）与指数服务时间（率 `μ`）；★ 返回 `(平均等待, 平均系统时间, 最大等待)`。
    ★★ **闭式不出现在本函数里**（`D.8` 在源码层核）。
    """
    rng = np.random.default_rng(seed)
    ia = rng.exponential(1.0 / lam, size=n_sample)
    sv = rng.exponential(1.0 / mu, size=n_sample)
    arrive = np.cumsum(ia)
    waits = np.empty(n_sample)
    free_at = 0.0
    for i in range(n_sample):
        start = arrive[i] if arrive[i] > free_at else free_at
        waits[i] = start - arrive[i]
        free_at = start + sv[i]
    burn = 1000
    return float(waits[burn:].mean()), float(waits[burn:].mean() + sv[burn:].mean()), float(waits.max())


def mm1_wait_closed(lam, mu):
    """★★★ **参照实现（II）**：`M/M/1` 的平均等待闭式 `W_q = ρ/(μ(1−ρ))`（★ 纯代数）。"""
    rho = lam / mu
    return rho / (mu * (1.0 - rho))


def mm1_wait_quantile_closed(lam, mu, p):
    """★ `M/M/1` 等待时间的**分位**闭式：`P(W_q > t) = ρ·e^{−μ(1−ρ)t}` ⇒ 解出 `t`。"""
    rho = lam / mu
    return -math.log(p / rho) / (mu * (1.0 - rho))


# ===========================================================================
# 3. A 类 —— ★★ 两条，各自两条独立路径
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **路由与排队**：两条判据，各自两条独立路径")
    print("=" * 78)

    # ── A.1 ★★★ 前缀感知的路由：逐请求模拟 vs 闭式 ───────────────────────
    print()
    print("  ── A.1 ★★★ **前缀感知的路由**：逐请求模拟（被测）vs **闭式** `N − 不同模板数`（参照）──")
    R, N, T = specs.ROUTE_N_REPLICA, specs.ROUTE_N_REQ, specs.ROUTE_N_TEMPLATE
    cap = 2
    rng = np.random.default_rng(specs.ROUTE_SEED)
    trace = route_trace(N, T, rng)
    hit_pa, share_pa, _ = run_routing(trace, R, cap, "prefix_aware")
    hit_rr, share_rr, _ = run_routing(trace, R, cap, "random")
    hit_pa_cf = prefix_hits_closed(trace, R, cap)
    print("     副本 %d 个 · 每副本缓存 `%d` 条前缀 · 请求 %d 条 · 模板 %d 个（★ 模板 = 共享前缀）"
          % (R, cap, N, T))
    print("     %-20s %-12s %-14s %s" % ("策略", "命中数", "命中率", "各副本份额"))
    print("     %-20s %-12d %-14.4f %s" % ("前缀感知（被测）", hit_pa, hit_pa / N, share_pa))
    print("     %-20s %-12d %-14.4f %s" % ("前缀感知（闭式）", hit_pa_cf, hit_pa_cf / N, "-"))
    print("     %-20s %-12d %-14.4f %s" % ("随机（对照）", hit_rr, hit_rr / N, share_rr))
    print("     ★★ `%s`" % ROUTE_HIT_DEF)
    print("     ★★★ **读法**：被测（逐请求事件驱动）的命中数与**闭式** `N − 不同模板数 = %d − %d = %d`"
          % (N, len(set(trace)), hit_pa_cf))
    print("         **逐位一致**（★ 前提 `R·C = %d ≥ T = %d` ⇒ **永无驱逐**，首次出现是唯一的缺失）；" % (R * cap, len(set(trace))))
    print("         ★★ 随机（对照）只命中 **%d** —— 它**完全不看请求内容**，与前述两条路径拉开差距。" % hit_rr)
    check("A.1", hit_pa == hit_pa_cf and hit_pa == 792 and hit_rr == 209
          and abs(hit_pa / N - 0.99) < 1e-9 and abs(hit_rr / N - 0.26125) < 1e-9
          and R * cap >= len(set(trace)) and len(trace) == N and hit_pa > hit_rr,
          "★★★ **判据成立**：前缀感知命中 **%d**（= 闭式 `N − 不同模板数 = %d`，**逐位一致**）· "
          "随机 **%d** —— ★★ 一面逐请求事件驱动、一面纯计数，两条**独立路径**会合"
          % (hit_pa, hit_pa_cf, hit_rr),
          "**前缀感知命中数（或闭式、或 `R·C ≥ T` 前提）被动过** —— ★ 先查 `run_routing` 与 `prefix_hits_closed`")

    # ── A.2 ★★★ 排队模拟 vs M/M/1 闭式 ─────────────────────────────────
    print()
    print("  ── A.2 ★★★ **排队**：离散事件模拟 vs `M/M/1` 闭式 `W_q = ρ/(μ(1−ρ))` ──")
    lam, mu, m = specs.QUEUE_ARRIVAL_RATE, specs.QUEUE_SERVICE_RATE, specs.QUEUE_N_SAMPLE
    wq, ws, wmax = sim_queue(lam, mu, m, specs.ROUTE_SEED)
    wq_cf = mm1_wait_closed(lam, mu)
    rel = abs(wq - wq_cf) / wq_cf
    print("     `λ = %.2f` · `μ = %.2f` · `ρ = %.2f` · 样本 %d（前 1000 条预热丢弃）" % (lam, mu, lam / mu, m))
    print("     模拟平均等待 = **%.4f** · 闭式 = **%.4f** · 相对差 = **%.4f**（容差 `3%%`）" % (wq, wq_cf, rel))
    print("     ★★ 标准误 ≈ `1/√%d` ≈ `%.4f` ⇒ ★ **容差不是随便定的**（判据容差应与标准误同量级）"
          % (m, 1.0 / math.sqrt(m)))
    print("     ★★ Little 定律 `L = λW`：平均系统时间 %.4f ⇒ `L = %.4f`（★ 由闭式 `ρ/(1−ρ)` 得 %.4f）"
          % (ws, lam * ws, (lam / mu) / (1 - lam / mu)))
    check("A.2", rel < 0.03 and abs(wq - 9.0871) < 1e-3 and abs(ws - (wq + 1.0 / mu)) < 0.05
          and abs(wq_cf - 9.0) < 1e-9 and abs(lam * ws - (lam / mu) / (1 - lam / mu)) < 0.2,
          "★★★ **判据成立**：离散事件模拟的平均等待 **%.4f** 与闭式 **%.4f** 相对差 **%.4f** < `3%%` —— "
          "★★ 一面逐到达推进、一面纯代数；★ Little 定律也逐位对上" % (wq, wq_cf, rel),
          "**排队模拟与闭式不符** —— ★ 先查 `sim_queue` 的到达/服务与 `mm1_wait_closed`")


# ===========================================================================
# 4. B 类：路由 · 排队 · 扩缩容
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 三张账：**命中率 vs 容量** · **等待 vs 利用率** · **副本数 vs 负载**")
    print("=" * 78)
    R, N, T = specs.ROUTE_N_REPLICA, specs.ROUTE_N_REQ, specs.ROUTE_N_TEMPLATE

    # ── B.1 ★★ 命中率 vs 容量 ──────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **路由命中率 vs 每副本缓存容量**（两种策略）──")
    print("     %-10s %-16s %-16s" % ("容量(条)", "前缀感知", "随机"))
    caps = (1, 2, 3, 4)
    rates = {}
    for c in caps:
        rng = np.random.default_rng(specs.ROUTE_SEED)
        tr = route_trace(N, T, rng)
        hpa, _s, _r = run_routing(tr, R, c, "prefix_aware")
        hrr, _s2, _r2 = run_routing(tr, R, c, "random")
        rates[c] = (hpa / N, hrr / N)
        print("     %-10d %-16.4f %-16.4f" % (c, hpa / N, hrr / N))
    pa_mono = all(rates[caps[i]][0] <= rates[caps[i + 1]][0] for i in range(len(caps) - 1))
    print("     ★★ 前缀感知**随容量单调不降** = %s；★ 随机也随容量上升（但**始终更低**）" % pa_mono)
    check("B.1", pa_mono and all(rates[c][0] > rates[c][1] for c in caps)
          and abs(rates[2][0] - rates[2][1]) > 0.3,
          "★★ **判据成立**：**4 档容量**下前缀感知**始终高于**随机，且随容量**单调不降**"
          "（`cap=2` 时 **%.4f vs %.4f**）" % (rates[2][0], rates[2][1]),
          "**命中率单调性或两策略的次序被动过** —— ★ 先查 `run_routing` 与 `Replica.serve`")

    # ── B.2 ★★ 等待 vs 利用率 ─────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **平均等待 vs 利用率**（`ρ → 1` 时发散）──")
    mu = specs.QUEUE_SERVICE_RATE
    print("     %-10s %-16s %-16s %-16s" % ("ρ", "λ", "W_q(闭式)", "p95(闭式)"))
    rows = []
    for rho in (0.5, 0.7, 0.9, 0.95):
        lam = rho * mu
        rows.append((rho, lam, mm1_wait_closed(lam, mu), mm1_wait_quantile_closed(lam, mu, 0.05)))
        print("     %-10.2f %-16.2f %-16.4f %-16.4f" % rows[-1])
    print("     ★★ 从 `ρ=0.5` 到 `ρ=0.95`：`W_q` **%.2f → %.2f（%.1f 倍）** —— 利用率越接近 1 越发散"
          % (rows[0][2], rows[-1][2], rows[-1][2] / rows[0][2]))
    check("B.2", all(rows[i][2] < rows[i + 1][2] for i in range(len(rows) - 1))
          and abs(rows[0][2] - 1.0) < 1e-12 and abs(rows[-1][2] - 19.0) < 1e-12,
          "★★ **判据成立**：`W_q` 随 `ρ` **单调上升**（`ρ=0.5` 时 **%.1f**，`ρ=0.95` 时 **%.1f**）—— "
          "★ 两端都是闭式端点" % (rows[0][2], rows[-1][2]),
          "**等待曲线被动过** —— ★ 先查 `mm1_wait_closed`")

    # ── B.3 ★★ 副本数 vs 负载 ─────────────────────────────────────────
    print()
    print("  ── B.3 ★★ **扩缩容**：达到目标利用率所需的副本数 ──")
    mu = specs.QUEUE_SERVICE_RATE
    target = 0.9
    print("     %-14s %-14s %-16s %-16s" % ("总到达率", "ρ 上限", "最少副本数", "实际 ρ"))
    rows3 = []
    for lam in (1.8, 3.6, 5.4, 9.0):
        need = 1
        while lam / (need * mu) > target:
            need += 1
        rows3.append((lam, need, lam / (need * mu)))
        print("     %-14.2f %-14.2f %-16d %-16.4f" % (lam, target, need, rows3[-1][2]))
    print("     ★★ **读法**：副本数 = `⌈λ / (μ · ρ_target)⌉` —— ★ 是**整数**，所以**实际利用率 ≤ 目标**。")
    check("B.3", all(r[2] <= target + 1e-12 for r in rows3) and rows3[1][1] == 4
          and rows3[3][1] == 10 and all(rows3[i][1] <= rows3[i + 1][1] for i in range(len(rows3) - 1)),
          "★★ **判据成立**：**4 档负载**下副本数 `= ⌈λ/(μ·ρ_target)⌉`（`λ=3.6` ⇒ **4** 个，`λ=9.0` ⇒ **10** 个），"
          "★ **实际利用率始终 ≤ 目标**（整数副本的保守性）",
          "**副本数被动过** —— ★ 先查上取整逻辑与 `QUEUE_SERVICE_RATE`")


# ===========================================================================
# 5. C 类：**指标口径** · **路由的两个反例** · **均值 vs 分位**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**指标口径** · **路由的两个反例** · **均值 vs 分位**")
    print("=" * 78)

    # ── C.1 ★★★ 指标口径 ──────────────────────────────────────────────
    print()
    print("  ── C.1 ★★★ **指标口径**：vLLM 的 TTFT **不含排队** ──")
    q = specs.VLLM_TTFT_DEFINITION_QUOTE
    genq = specs.VLLM_GENERATION_TIME_DEFINITION_QUOTE
    itl = specs.VLLM_ITL_DEFINITION_QUOTE
    print("     `time_to_first_token_ms`：`%s`" % q)
    print("     `generation_time_ms`    ：`%s`" % genq)
    print("     `mean_itl_ms`           ：`%s`" % itl)
    print("     ★★★ **三个量是同一条时间轴上的【三段】**：")
    print("        ① 排队（**谁都不算它** —— vLLM 的 TTFT 从 `scheduled` 起）；")
    print("        ② TTFT（`scheduled` → 首 token，**含 prefill**）；")
    print("        ③ `generation_time`（首 token → 末 token，**排除排队与 prefill**）。")
    print("     ★★ **读法**：把「排队」漏掉，**用户的端到端首 token 延迟会被系统性低估** ——")
    print("        这正是「两个都叫延迟的量，作用域不同」（`01` 的 `C.2` 是同一件事的另一面）。")
    check("C.1", "scheduled" in q and "arriv" not in q.lower()
          and "Excludes both queue wait and prefill" in genq
          and specs.VLLM_ITL_NULL_FOR_SINGLE_TOKEN_QUOTE == "null for single-token responses"
          and itl.endswith("during the decode phase.")
          and ROUTE_HIT_DEF.endswith("副本】的 KV 缓存里"),
          "★★★ **判据成立**：vLLM 的 TTFT 起于 `scheduled`（**不含排队**）、`generation_time` **显式排除**"
          "排队与 prefill、`mean_itl_ms` 对**单 token 响应为 `null`** —— ★★ 三段口径**逐字钉住**",
          "**指标口径引文被动过** —— ★ 先查 `VLLM_TTFT_DEFINITION_QUOTE` / `VLLM_GENERATION_TIME_DEFINITION_QUOTE`")

    # ── C.2 ★★★ 前缀感知路由的两个反例 ────────────────────────────────
    print()
    print("  ── C.2 ★★★ **前缀感知路由的两个反例** ──")
    R, N = specs.ROUTE_N_REPLICA, specs.ROUTE_N_REQ
    cap = 2
    # 反例 1：无共享前缀（模板数 ≫ 总容量）
    tr_uniq = list(range(N))            # ★ **每条请求一个各不相同的模板** ⇒ 零共享
    h1, _s1, _r1 = run_routing(tr_uniq, R, cap, "prefix_aware")
    h1r, _s1r, _r1r = run_routing(tr_uniq, R, cap, "random")
    # 反例 2：单一前缀 ⇒ 全打到同一个副本
    tr_one = [0] * N                    # ★ **所有请求共享同一个前缀**（单一模板）
    h2, share2, _r2 = run_routing(tr_one, R, cap, "prefix_aware")
    print("     ★ **反例 1（零共享前缀）**：%d 条请求**各一个不同模板** ⇒ 前缀感知命中 **%d**、随机 **%d**"
          % (N, h1, h1r))
    print("        ⇒ ★★ **优势消失**（两个都 ≈ 强制缺失率）—— 前缀感知**依赖前缀共享这个前提**。")
    print("     ★ **反例 2（单一前缀）**：模板 `1` 个 ⇒ 各副本份额 `%s`" % share2)
    print("        ⇒ ★★ **全部请求被打到同一个副本**（`max = %d`）—— **路由命中率高 ≠ 系统好**：" % max(share2))
    print("        ★★★ 它把**负载集中**这一个副作用买来了（★ 真实系统要在「命中」与「均衡」之间取折中）。")
    check("C.2", h1 == 0 and h1r == 0
          and max(share2) == N and h2 == N - 1,
          "★★★ **判据成立**：反例 1 里两策略命中数**都是 `0`**（零共享 ⇒ **无收益**）；"
          "反例 2 里**全部 %d 条**都落到一个副本（`max` = %d）⇒ ★ **负载集中**"
          % (N, max(share2)),
          "**两个反例之一被动过** —— ★ 先查 `ROUTE_N_TEMPLATE` 的用法与 `run_routing`")

    # ── C.3 ★★ 均值 vs 分位 ───────────────────────────────────────────
    print()
    print("  ── C.3 ★★ **均值 vs 分位**：均值达标 ≠ 尾部达标 ──")
    lam, mu = 0.9, specs.QUEUE_SERVICE_RATE
    wq = mm1_wait_closed(lam, mu)
    p95 = mm1_wait_quantile_closed(lam, mu, 0.05)
    p50 = mm1_wait_quantile_closed(lam, mu, 0.50)
    print("     `ρ = %.2f`：平均等待 **%.4f** · 中位 **%.4f** · `p95` **%.4f** · `p95/均值` = **%.2f**"
          % (lam / mu, wq, p50, p95, p95 / wq))
    print("     ★★ **读法**：若 SLO 写「平均等待 < `%.1f`」，则**看似达标**；" % (wq + 1.0))
    print("        但 `p95 = %.2f` ⇒ ★ **5%% 的请求等了三倍于均值的倍数** —— ★★ **均值掩盖尾部**。" % p95)
    print("     ★★★ 这与 `01` 的「**两个延迟**」是同一件事的度量侧：**报一个均值，必须先说清它是什么口径**。")
    check("C.3", abs(wq - 9.0) < 1e-12 and abs(p95 - 28.9037) < 1e-3
          and p95 > 3 * wq and p50 < wq,
          "★★ **判据成立**：`ρ=0.9` 时**均值 `%.2f` 但 `p95` = `%.2f`**（`p95/均值` = **%.2f**），"
          "且**中位数【小于】均值**（`%.2f < %.2f`，★ **右偏**的重尾）—— ★★ 均值达标**不能**推出尾部达标"
          % (wq, p95, p95 / wq, p50, wq),
          "**分位或均值被动过** —— ★ 先查 `mm1_wait_quantile_closed`")


# ===========================================================================
# 6. 前哨：口径 / 回链 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 回链锚点 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/回链/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无多机 / 无真网络**。")
    print("      ★★ 但**本程序的 A 类是真的**：**路由**与**排队**各自两条独立路径。")
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

    check("D.3", specs.SERVING_STACK_LETDOWN_REF == "50-performance/README.md:49"
          and specs.KV_HIT_RATE_REF == "120-inference/kv-cache/03-prefix-and-eviction"
          and specs.SERVING_INFRA_FETCH_DAY == "2026-10-09"
          and ROUTE_HIT_DEF == "路由命中 = 请求的前缀在被【分配到的那个副本】的 KV 缓存里",
          "★★★ **回链锚点逐字钉住**：`50-performance` 的 serving 栈让渡 `%s` · 同域前驱（缓存侧命中率）`%s`；"
          "★ **路由命中**的定义逐字钉住（★ 与缓存命中率**同名不同指**）；抓取日 `%s`"
          % (specs.SERVING_STACK_LETDOWN_REF, specs.KV_HIT_RATE_REF, specs.SERVING_INFRA_FETCH_DAY))

    ok = (0.0 < specs.QUEUE_ARRIVAL_RATE < specs.QUEUE_SERVICE_RATE
          and specs.QUEUE_N_SAMPLE >= 10000 and specs.ROUTE_N_REPLICA >= 2
          and specs.ROUTE_N_TEMPLATE >= 2 and specs.ROUTE_N_REQ >= 100)
    print("     ★★ 队列**稳定**（`λ < μ`）、样本数 ≥ `1e4`、副本数 ≥ 2、模板数 ≥ 2、请求数 ≥ 100 = %s" % ok)
    check("D.3b", ok and specs.UNSOURCED == ()
          and specs.ROUTE_N_REPLICA == 4 and specs.ROUTE_N_REQ == 800
          and specs.ROUTE_N_TEMPLATE == 8 and specs.ROUTE_SEED == 20261009
          and specs.QUEUE_ARRIVAL_RATE == 0.9 and specs.QUEUE_SERVICE_RATE == 1.0
          and specs.QUEUE_N_SAMPLE == 120000,
          "★★★ **口径守卫 + 参数逐字钉住**：`λ=%.1f` · `μ=%.1f`（**稳定** `ρ=%.1f`）· 样本 `%d` · "
          "副本 `%d` · 请求 `%d` · 模板 `%d` · 种子 `%d`；★ `UNSOURCED == ()`"
          % (specs.QUEUE_ARRIVAL_RATE, specs.QUEUE_SERVICE_RATE,
             specs.QUEUE_ARRIVAL_RATE / specs.QUEUE_SERVICE_RATE, specs.QUEUE_N_SAMPLE,
             specs.ROUTE_N_REPLICA, specs.ROUTE_N_REQ, specs.ROUTE_N_TEMPLATE, specs.ROUTE_SEED),
          "**队列不稳定或参数被动过** —— ★ 先查 `QUEUE_*` / `ROUTE_*`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：A 类存在性 · 守卫实验 · 独立性 · 术语纪律
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
          "★★★ **判据成立**：本文件里 **A 类判据有 %d 条**（≥ %d）—— ★★ **本主题不能整主题标「无 A」**："
          "**排队与负载均衡是可独立参照的算法**" % (n_a, EXPECTED_MIN_A),
          "★★ 本文件里 A 类判据少于 %d 条 —— ★ 与「排队/路由有 A 类」**自相矛盾**" % EXPECTED_MIN_A)

    # ── D.4 ★★★ 守卫实验：把路由改回随机 ────────────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验**：把前缀感知**改坏成随机** ──")
    R, N, T = specs.ROUTE_N_REPLICA, specs.ROUTE_N_REQ, specs.ROUTE_N_TEMPLATE
    rng = np.random.default_rng(specs.ROUTE_SEED)
    trace = route_trace(N, T, rng)
    good, _sg, _rg = run_routing(trace, R, 2, "prefix_aware")
    bad, _sb, _rb = run_routing(trace, R, 2, "prefix_aware", corrupt=True)
    print("     构造：副本 %d · 容量 2 · 请求 %d 条；正确版命中 = **%d** · 改坏版命中 = **%d**"
          % (R, N, good, bad))
    print("     ★★★ **读法**：把「优先发给持有者」去掉 ⇒ 命中数掉到与随机**完全一样** ⇒ `A.1` 的差距消失。")
    check("D.4", good > bad and bad < good * 0.6,
          "★★★ **守卫成立**：改坏后命中**从 %d 掉到 %d** ⇒ ★★ **`A.1` 对「优先发给持有者」这条规则是敏感的**"
          % (good, bad),
          "**守卫失效** —— ★ 改回随机后命中居然没掉（那 `A.1` 抓不到它）")

    # ── D.6 ★★ 禁写挡板 `F5`（★ 抬头宣称的纪律必须有一条断言盯住）────────────
    print()
    print("  ── D.6 ★★ **禁写挡板 `F5`**：APC「默认开启」**未取到** ──")
    print("     `specs.APC_DEFAULT_DECLARED_IN_SOURCE` = %s（★ 只能是 `False`）"
          % specs.APC_DEFAULT_DECLARED_IN_SOURCE)
    check("D.6", specs.APC_DEFAULT_DECLARED_IN_SOURCE is False,
          "★★ **禁写挡板成立**：取数到的页面**未声明** APC 默认开启 ⇒ ★ 写「APC **默认开启**」会与它"
          "**自相矛盾**（`F5`）；★ 抬头那条禁写与**本断言成对**",
          "**`F5` 挡板被动过** —— ★ `APC_DEFAULT_DECLARED_IN_SOURCE` 只能是 `False`（未取到默认值声明）")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.ROUTE_N_REPLICA == 4 and specs.ROUTE_N_REQ == 800
          and specs.ROUTE_N_TEMPLATE == 8 and specs.ROUTE_SEED == 20261009
          and specs.QUEUE_ARRIVAL_RATE == 0.9 and specs.QUEUE_SERVICE_RATE == 1.0
          and specs.QUEUE_N_SAMPLE == 120000
          and specs.APC_DEFAULT_DECLARED_IN_SOURCE is False
          and specs.VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES == 0
          and specs.SERVING_INFRA_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：路由四个参数 · 队列三个参数 · `F5` 挡板 · 缩写计数 `0` · 抓取日；★ `UNSOURCED == ()`",
          "**某个常量被动过** —— ★ 先查 `ROUTE_*` / `QUEUE_*` / `APC_*`")

    # ── D.8 ★★★ 独立性（源码级）──────────────────────────────────────────
    s1 = _func_names(run_routing)
    c1 = _func_names(prefix_hits_closed)
    s2 = _func_names(sim_queue)
    c2 = _func_names(mm1_wait_closed)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     A.1: `run_routing` 里出现 `prefix_hits_closed` = %s ；反向 = %s"
          % ("prefix_hits_closed" in s1, "run_routing" in c1))
    print("     A.2: `sim_queue` 里出现 `mm1_wait_closed` = %s ；反向 = %s"
          % ("mm1_wait_closed" in s2, "sim_queue" in c2))
    print("     ★★★ **降级措辞**：这只是「**几段代码不互相引用**」的**源码级线索**，**不证明**判据有信息量 ——")
    print("       后者由 `A.1`/`A.2` 与 `D.4` 的**行为**兜底（★ `A.1` 的守卫是 `D.4`：改坏后闭式不再对上）。")
    check("D.8", "prefix_hits_closed" not in s1 and "run_routing" not in c1
          and "mm1_wait_closed" not in s2 and "sim_queue" not in c2,
          "★★ **源码级线索成立**：两条判据的**被测与参照两两互不引用**（★ **两侧都审**）——"
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
    print("★★★ **A = %d/%d** —— ★★ **本主题不能整主题标「无 A」**：路由与排队各两条独立路径。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：模型与算例是 **`[E]` 级（本库实现）**；★ 指标口径取自 vLLM 官方文档（`[B]`）。")
    print("★ 本程序只核对「**路由与排队的算术**」——**不是**任何真实部署的行为；")
    print("  ★★ **真实网络 / 多机的行为本机测不了**（单机、无真网络）。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_scaling_and_observability.py —— 扩缩容 / 路由（前缀感知）/ 可观测性（指标口径）")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无多机 / 无真网络。")
    print("  ★★★ **本程序有真 A 类**（路由 / 排队）；★ 承接 `50-performance:49` 让渡的 serving 栈。")
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
