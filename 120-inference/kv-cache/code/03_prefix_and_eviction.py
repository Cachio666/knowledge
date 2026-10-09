#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_prefix_and_eviction.py —— **前缀共享与淘汰的机制**：`hash → 块`、命中率、LRU 驱逐。

配套文档：`03-prefix-and-eviction.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题不写**「KV cache 量化」「paged KV 的 kernel 侧」（那是 `130-quantization/.../02,03` 与
  `40-kernels/attention/05:130-179,283-290` 的正文，本主题**只回链**）。
  本程序只写 **前缀共享的系统侧机制**：**`hash → 块` / 命中率的定义 / LRU 驱逐**。

★★★ **A 类有两条独立路径**（照 `§1.3.2b` / `CONTRIBUTING:235`）：
  · **被测** = `hash → 块` 的映射（每条请求逐个**整块**算键，键**含前缀链** —— 这是 vLLM 文档
    的哈希口径，`[B]`，见 `specs.VLLM_FULL_BLOCKS_QUOTE`）；
  · **参照** = **逐 token 朴素比对**：拿当前请求与**之前每一条**请求**逐 token** 比最长公共前缀，
    再除以块大小取整（★ **完全不碰哈希表**）。
  ★★ 判据：在**构造的前缀结构**上，两条路径给出的**命中块数必须一致**。

★★ **覆盖边界**：本程序守的是「**前缀共享的算术**」（命中块数 / 命中率 / LRU 的收益 / 碰撞与误逐出的后果）。
  **不守**任何**真实引擎**的前缀缓存行为（本机**无引擎**）。
  ★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **`hash → 块` == 逐 token 朴素比对**（构造的 `PREFIX_N_REQ` 条请求上逐条一致）；
  · `A.2` ★★ 两条路径给出**同一个命中率**（同一个分子分母口径）；
  · `A.3` ★★ **免费守卫**：重复的请求 ⇒ **整条命中**；首个请求 ⇒ 命中 `0`；
  · `B.1/B.2` ★ **LRU 的收益账**：命中率 vs 缓存容量（单调不降 + 容量 ≥ 去重键数 ⇒ 只剩强制缺失）；
  · `C.1` ★★★ **哈希碰撞** ⇒ 命中到**错的块** ⇒ 复用的内容与真前缀**不一致**；
  · `C.2` ★★★ **驱逐掉仍在用的块** ⇒ 活请求的块**消失**；
  · `D.4/D.5` ★★★ **守卫实验**：把哈希改坏 / 把「只逐出 `ref == 0`」改坏 ⇒ 断言必须失败；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **失败标记自检** + 前沿分支（KDA）的三概念。
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
ASSERT_FAIL_MARK = "-- FAIL: kv-cache"

OTHER_THEME_MARKS = ("-- FAIL: batching-and-scheduling", "-- FAIL: decoding-strategies",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

LRU_N_KEYS = 16          # 访问序列里**不同块键**的个数（**本主题算例**）
LRU_TRACE_LEN = 400      # 访问序列的长度（**本主题算例**）
KV_MIX_A = 131           # ★ 本主题的**确定性**哈希：一次乘加（**不是** Python 的 `hash()` ——
KV_MIX_B = 7             #   后者对字符串会随 `PYTHONHASHSEED` 变，不可复现）
KV_MOD = 1000000007      # 取模


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D）。★ 失败路径**不复述结论**。"""
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
# 1. 被测：**`hash → 块`**（键含前缀链；只对整整块）
# ===========================================================================
def mk_key(prefix_key, block):
    """**确定性的块键**：把「前缀键」与「本块的 token」混成一个整数（★ 单行确定性）。

    ★★ 这是**本主题的实现**（不是 vLLM 的原函数）；它照的**口径**是 vLLM 文档那一条
      「块键 = 本块 token + **本块之前的前缀**」。
    """
    h = prefix_key
    for t in block:
        h = (h * KV_MIX_A + t + KV_MIX_B) % KV_MOD
    return h


def mk_key_constant(prefix_key, block):
    """★ **坏法**（只用于 `C.1` / `D.4` 的守卫实验）：**常数键** ⇒ 所有块撞在一起。"""
    return 0


def measured_match(req, cache, mk):
    """★★★ **被测路径**：逐**整块**算键、查缓存，返回 `(命中块数, 复用到的 token)`。

    ★★ 口径（`[B]`，`specs.VLLM_FULL_BLOCKS_QUOTE`）：
      · **只有整块**参与（末尾不满一块**既不入缓存、也不参与命中**）；
      · 键**含前缀链**（同一个块内容在不同前缀下是**不同的键**）；
      · **只有【连续的开头**命中的那一段**才算命中**（遇到第一个未命中就停止计命中，但仍继续入缓存）。
    """
    prefix_key = 0
    matched = 0
    reused = []
    all_hit = True
    nblocks = len(req) // specs.PREFIX_BLOCK_TOKENS
    for bi in range(nblocks):
        blk = tuple(req[bi * specs.PREFIX_BLOCK_TOKENS:(bi + 1) * specs.PREFIX_BLOCK_TOKENS])
        k = mk(prefix_key, blk)
        if all_hit and k in cache:
            matched += 1
            reused.extend(cache[k])
        else:
            all_hit = False
        cache[k] = blk                     # ★ 每一块都入缓存（含未命中的）
        prefix_key = k
    return matched, tuple(reused)


# ===========================================================================
# 2. 参照：**逐 token 朴素比对**（★ **完全不碰哈希表**）
# ===========================================================================
def lcp_len(a, b):
    """两条 token 序列的**最长公共前缀长度**（逐 token 比，**不用任何哈希**）。"""
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def reference_match(reqs, idx):
    """★★★ **参照路径**：与**之前每一条**请求逐 token 比对，取**最大公共前缀**再按【整块】取整。

    ★★ 它判的是与 `measured_match` **同一个量**（开头的**整块**命中数），
      但走的是**完全不同的路**：没有哈希、没有缓存表，只有 `lcp_len` 的逐 token 比较。
    """
    best = 0
    for p in range(idx):
        best = max(best, lcp_len(reqs[p], reqs[idx]) // specs.PREFIX_BLOCK_TOKENS)
    return best


def build_requests(seed, n):
    """构造一棵**共同前缀**的小结构：`(share, extra)` 表示「共享 base 的前 `share` 个 token + `extra` 个新 token」。

    ★ 所有请求长度都是块大小（`%d`）的整数倍 ⇒ **每个块都是整块**，命中口径无歧义。
    """
    rng = np.random.default_rng(seed)
    base = tuple(int(x) for x in rng.integers(1, 50, size=12))
    plans = [(12, 0), (8, 4), (8, 4), (4, 8), (12, 0), (4, 4), (8, 0), (12, 0)]
    reqs = []
    for (share, extra) in plans[:n]:
        reqs.append(base[:share] + tuple(int(x) for x in rng.integers(50, 100, size=extra)))
    return reqs


# ===========================================================================
# 3. A 类 —— ★★★ **`hash → 块` vs 逐 token 朴素比对**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **前缀共享**：`hash → 块`（被测）vs 逐 token 朴素比对（参照）")
    print("=" * 78)
    print("  ★★★ 被测 = `measured_match`（逐整块算键查缓存，键**含前缀链**）；")
    print("      参照 = `reference_match`（与之前每条**逐 token** 比 LCP，**不碰哈希表**）。")
    print("  ★★ 块大小 = `%d` token；★ 口径：**只有整块**参与命中（`[B]`，vLLM：`%s`）"
          % (specs.PREFIX_BLOCK_TOKENS, specs.VLLM_FULL_BLOCKS_QUOTE))

    reqs = build_requests(specs.PREFIX_SEED, specs.PREFIX_N_REQ)
    cache = {}
    meas = []
    for i, r in enumerate(reqs):
        m, reused = measured_match(r, cache, mk_key)
        meas.append((m, reused))

    # ── A.1 ★★★ 逐条一致 ───────────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **命中块数：两条路径逐条一致** ──")
    print("     %-6s %-10s %-30s %-10s %-10s %-6s"
          % ("#req", "长度", "前 8 个 token", "hash命中", "朴素比对", "一致"))
    ok1 = True
    for i, r in enumerate(reqs):
        ref = reference_match(reqs, i)
        ok1 = ok1 and (meas[i][0] == ref)
        print("     %-6d %-10d %-30s %-10d %-10d %-6s"
              % (i, len(r), str(r[:8]), meas[i][0], ref, meas[i][0] == ref))
    check("A.1", ok1 and len(reqs) == 8 and meas[0][0] == 0 and meas[4][0] == 3,
          "★★★ **判据成立**：**%d 条请求**上，「`hash → 块` 给出的命中块数」**逐条等于**"
          "「逐 token 朴素比对给出的命中块数」—— ★★ 一面走**哈希表**、一面**逐 token 比**，"
          "两条独立路径会合" % len(reqs),
          "**两条路径对不上** —— ★ 先查「整块」口径（末尾不满一块是否被算进去了）与键是否含前缀链")

    # ── A.2 ★★ 同一个命中率 ─────────────────────────────────────────────
    print()
    print("  ── A.2 ★★ **命中率**：同一条分子分母口径 ──")
    total_blocks = sum(len(r) // specs.PREFIX_BLOCK_TOKENS for r in reqs)
    m_hit = sum(m for m, _ in meas)
    r_hit = sum(reference_match(reqs, i) for i in range(len(reqs)))
    print("     命中块数：hash %d · 朴素 %d ／ 总块数 %d ⇒ 命中率 hash %.4f · 朴素 %.4f"
          % (m_hit, r_hit, total_blocks, m_hit / total_blocks, r_hit / total_blocks))
    print("     ★★ **命中率的定义**：`Σ_请求 命中块数 / Σ_请求 总块数`（★ **分母是全部块**，不是 `len(reqs)`）——")
    print("        换一个分母就会得到另一个数（★ 这是本库「**报数必报口径**」的直接应用）。")
    check("A.2", m_hit == r_hit and total_blocks == 22 and m_hit == 14,
          "★★ **判据成立**：两条路径给出**同一个命中块数 %d / 同一个总块数 %d ⇒ 命中率 %.4f**"
          "（★ 分子分母都是同一条口径）" % (m_hit, total_blocks, m_hit / total_blocks),
          "**命中率对不上** —— ★ 先查分子分母是不是同一条口径")

    # ── A.3 ★★ 免费守卫 ────────────────────────────────────────────────
    print()
    print("  ── A.3 ★★ **免费守卫**：首个请求命中 0；重复的请求整条命中 ──")
    first = meas[0][0]
    dup = meas[4][0]                      # req4 与 req0 完全相同
    print("     req0（第一个）命中 **%d**；req4（与 req0 逐 token 相同）命中 **%d**（其整块数 = %d）"
          % (first, dup, len(reqs[4]) // specs.PREFIX_BLOCK_TOKENS))
    check("A.3", first == 0 and dup == len(reqs[4]) // specs.PREFIX_BLOCK_TOKENS == 3
          and meas[1][0] == 2 and meas[3][0] == 1,
          "★★ **判据成立**：**第一个请求命中 `0`**（缓存是空的 —— ★ 没有「凭空命中」）；"
          "★ 与 req0 **逐 token 相同**的 req4 命中**整条 3 块**；★ 共享 8 个 token 的 req1 命中 `2`、"
          "共享 4 个的 req3 命中 `1`",
          "**守卫被破坏** —— ★ 首个请求居然命中了，说明缓存没清空或键退化了")


# ===========================================================================
# 4. B 类：**LRU 驱逐的收益账**（命中率 vs 缓存大小）
# ===========================================================================
def gen_trace(seed, n_keys, length):
    """生成访问序列：**热键集中**（80% 落在前 `n_keys//4` 个键上），其余均匀散布。"""
    rng = np.random.default_rng(seed)
    hot = max(1, n_keys // 4)
    trace = []
    for _ in range(length):
        if rng.random() < 0.80:
            trace.append(int(rng.integers(0, hot)))
        else:
            trace.append(int(rng.integers(0, n_keys)))
    return trace


def lru_hits(trace, cap):
    """★★ **LRU 缓存**：容量 `cap` 块，返回**命中次数**。"""
    cache = collections.OrderedDict()
    hits = 0
    for k in trace:
        if k in cache:
            hits += 1
            cache.move_to_end(k)
        else:
            cache[k] = None
            if len(cache) > cap:
                cache.popitem(last=False)
    return hits


def fifo_hits(trace, cap):
    """★ **对照**：FIFO（与 LRU 只差一件事 —— 命中时**不刷新**顺序）。"""
    cache = collections.OrderedDict()
    hits = 0
    for k in trace:
        if k in cache:
            hits += 1
        else:
            cache[k] = None
            if len(cache) > cap:
                cache.popitem(last=False)
    return hits


def part_B():
    print()
    print("=" * 78)
    print("B · **LRU 驱逐的收益账**：命中率 vs 缓存容量")
    print("=" * 78)

    trace = gen_trace(specs.LRU_SEED, LRU_N_KEYS, LRU_TRACE_LEN)
    distinct = len(set(trace))

    # ── B.1 ★ 命中率 vs 容量（单调不降）──────────────────────────────────
    print()
    print("  ── B.1 ★ **命中率 vs 容量**（访问序列 %d 步 · 去重键 %d 个）──"
          % (LRU_TRACE_LEN, distinct))
    print("     %-10s %-10s %-10s %-14s %-10s" % ("容量(块)", "LRU命中", "FIFO命中", "LRU命中率", "全命中?"))
    rows = {}
    for cap in specs.LRU_CAP_GRID:
        h = lru_hits(trace, cap)
        f = fifo_hits(trace, cap)
        rows[cap] = (h, f)
        full = "是" if cap >= distinct else "否"
        print("     %-10d %-10d %-10d %-14.4f %-10s" % (cap, h, f, h / LRU_TRACE_LEN, full))
    mono = all(rows[specs.LRU_CAP_GRID[i]][0] <= rows[specs.LRU_CAP_GRID[i + 1]][0]
               for i in range(len(specs.LRU_CAP_GRID) - 1))
    print("     ★★ **单调不降 = %s**（★ 缓存越大，命中不会变少）" % mono)
    check("B.1", mono and rows[16][0] == 384 and rows[8][0] == 350 and rows[2][0] == 136
          and rows[4][0] == 266
          and rows[2][1] == 138 and rows[4][1] == 250 and rows[8][1] == 327 and rows[16][1] == 384,
          "★ **判据成立**：命中率随容量**单调不降**（`2 → 4 → 8 → 16` 块：`%d → %d → %d → %d`）——"
          " ★★ 这正是「**缓存大小是命中率的上界**」这条单调性的可算形态。\n"
          "        ★★ **FIFO 对照（逐格钉住）**：`2 / 4 / 8 / 16` 块上 FIFO = `%d / %d / %d / %d` ——"
          "        ★★★ **注意 `cap = 2`：FIFO `%d` > LRU `%d`**（★ **两者无普遍次序**，不是「LRU 恒优」）"
          % (rows[2][0], rows[4][0], rows[8][0], rows[16][0],
             rows[2][1], rows[4][1], rows[8][1], rows[16][1], rows[2][1], rows[2][0]),
          "**单调性或某一格（LRU / FIFO）被动过** —— ★ 先查 LRU 的 `move_to_end`（命中时有没有刷新顺序）")

    # ── B.2 ★ 免费守卫：容量 ≥ 去重键数 ⇒ 只剩强制缺失 ──────────────────
    print()
    print("  ── B.2 ★ **免费守卫**：容量 ≥ 去重键数 ⇒ 命中 = 长度 − 去重数（只剩**强制缺失**）──")
    big = lru_hits(trace, distinct)
    print("     容量 = 去重键数 = %d：LRU 命中 **%d**；而 `长度 − 去重数` = %d − %d = **%d**"
          % (distinct, big, LRU_TRACE_LEN, distinct, LRU_TRACE_LEN - distinct))
    print("     ★★ **读法**：缓存装得下所有键时，**唯一**的缺失是每个键的**第一次**（强制缺失）——")
    print("        ★★ 这条下界与**任何**替换策略无关（LRU / FIFO 都给同一个数）。")
    check("B.2", big == LRU_TRACE_LEN - distinct and fifo_hits(trace, distinct) == big
          and rows[16][0] == big and distinct == 16,
          "★ **判据成立**：容量 `≥ %d` 时 LRU 命中 **%d == 长度 − 去重数**，且 **FIFO 给同一个数** ——"
          " ★★ 那是**策略无关的强制缺失下界**" % (distinct, big),
          "**强制缺失下界不成立** —— ★ 先查去重键数与序列长度")


# ===========================================================================
# 5. C 类：**哈希碰撞** 与 **驱逐掉仍在用的块**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**哈希碰撞** 与 **驱逐掉仍在用的块**")
    print("=" * 78)

    # ── C.1 ★★★ 哈希碰撞 ⇒ 命中到错的块 ────────────────────────────────
    print()
    print("  ── C.1 ★★★ **哈希碰撞**（常数键）：命中到**别人的块** ──")
    reqs = build_requests(specs.PREFIX_SEED, specs.PREFIX_N_REQ)
    cache_good = {}
    cache_bad = {}
    bad_req = None
    ok_good = True
    for i, r in enumerate(reqs):
        _, reused_good = measured_match(r, cache_good, mk_key)
        # ★★ 正确路径的复用内容必须与真前缀**逐字相同** —— ★ 并进 `C.1`（**不是**裸 `assert`）
        ok_good = ok_good and (reused_good == tuple(r[:len(reused_good)]))
        m_bad, reused_bad = measured_match(r, cache_bad, mk_key_constant)
        want = tuple(r[:m_bad * specs.PREFIX_BLOCK_TOKENS])
        if reused_bad != want and bad_req is None:
            bad_req = (i, want, reused_bad)
    b0 = bad_req[0] if bad_req else -1
    b1 = bad_req[1] if bad_req else ()
    b2 = bad_req[2] if bad_req else ()
    print("     正确路径：**每一条**请求「复用到的内容」都等于它自己的真前缀（★ 逐字）= %s" % ok_good)
    print("     碰撞路径：req%d 期望前缀 %s，却复用到了 %s"
          % (b0, b1, b2) if bad_req else "     碰撞路径：未出现错配")
    check("C.1", ok_good and bad_req is not None and b1 != b2,
          "★★★ **判据成立**：① **正确路径**：**每一条**请求「复用到的内容」都等于它自己的**真前缀**（★ 逐字）；"
          "② 常数键下，req%d 复用到的是 **`%s`**，而它**真正**的前缀是 `%s` ⇒"
          " ★★★ **命中到错的块**（`hash → 块` 的前提是**键唯一标识内容**，碰撞就把它打破了）"
          % (b0, b2, b1),
          "**碰撞没有跑出后果、或正确路径没跑对** —— ★ 先查坏法键是否真的把所有块撞在一起")

    # ── C.2 ★★★ 驱逐掉仍在用的块 ⇒ 活请求的块消失 ─────────────────────
    print()
    print("  ── C.2 ★★★ **驱逐掉仍在用的块**（忽略了 `ref > 0`）──")
    cap = 4

    def scenario(corrupt):
        pool = {}       # key -> [tokens, ref, age]
        tick = [0]

        def acquire(key, tokens, hold):
            tick[0] += 1
            if key in pool:
                pool[key][1] += 1
                pool[key][2] = tick[0]
                return True
            if len(pool) >= cap:
                cands = [k for k, v in pool.items()
                         if (corrupt == "evict_in_use" or v[1] == 0)]
                if not cands:
                    return False
                victim = min(cands, key=lambda k: pool[k][2])
                del pool[victim]
            pool[key] = [tokens, 1 if hold else 0, tick[0]]
            return True

        def release(key):
            if key in pool:
                pool[key][1] -= 1
                tick[0] += 1

        acquire("K0", ("a", "b", "c", "d"), True)             # ★ 一条活请求**在用**这个块
        for i in range(cap - 1):
            acquire("F%d" % i, ("f", i), True)               # 填满池
            release("F%d" % i)                               # ★ 这些块**用完就还**（ref → 0）
        ok = acquire("NEW", ("z", "z"), False)               # 需要逐出一个
        return pool, ok

    good_pool, good_ok = scenario(None)
    bad_pool, bad_ok = scenario("evict_in_use")
    print("     池容量 %d；先用一个块 `K0`（**ref = 1，正在被用**），再用 `F*` 填满并归还" % cap)
    print("     正确驱逐：`K0` 还在 = %s（逐出的是 `ref == 0` 的 `F*`）" % ("K0" in good_pool))
    print("     坏法驱逐：`K0` 还在 = %s（★ 忽略了 `ref > 0`，把**正在用的块**逐掉了）"
          % ("K0" in bad_pool))
    check("C.2", ("K0" in good_pool) and good_ok and ("K0" not in bad_pool) and bad_ok,
          "★★★ **判据成立**：正确驱逐**保留了 `ref = 1` 的 `K0`**（逐出的是已归还的 `F*`）；"
          "★★ **坏法把 `K0` 逐掉了** —— 而它**还在被一条活请求使用** ⇒ 那条请求的块**消失了**"
          "（后果：要么读到错的块、要么 **KV 重算**）",
          "**误逐出没有跑出后果** —— ★ 先查驱逐候选是否真的按 `ref == 0` 过滤")


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无引擎**；")
    print("      ★★ 但**本程序的 A 类是真的**：**`hash → 块` vs 逐 token 朴素比对**（两条独立路径）。")
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

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: kv-cache" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: kv-cache`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies",
                                        "-- FAIL: engines", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.KV_FETCH_DAY == "2026-10-09"
          and specs.PREFIX_BLOCK_TOKENS == 4 and specs.PREFIX_SEED == 20261009
          and specs.PREFIX_N_REQ == 8 and specs.LRU_SEED == 20261009
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：抓取日 `%s` · 块 = `%d` · 种子 `%d` · 请求数 `%d` · LRU 种子 `%d`；"
          "★ `UNSOURCED == ()` 一并钉住"
          % (specs.KV_FETCH_DAY, 4, 20261009, 8, 20261009))

    # ★★★ 口径守卫：块大小必须**整除**计划里所有请求长度（否则「整块」口径会有边界歧义），
    #   且 `LRU_CAP_GRID` 里**必须有 ≥ `LRU_N_KEYS` 的一档**（`B.2` 的强制缺失下界靠它）。
    reqs = build_requests(specs.PREFIX_SEED, specs.PREFIX_N_REQ)
    mod_ok = all(len(r) % specs.PREFIX_BLOCK_TOKENS == 0 for r in reqs)
    cap_ok = max(specs.LRU_CAP_GRID) >= LRU_N_KEYS
    print("     ★★ 所有请求长度都是块大小的整数倍 = %s；容量网格含 ≥ 去重键数的一档 = %s"
          % (mod_ok, cap_ok))
    check("D.3b", mod_ok and cap_ok and specs.PREFIX_BLOCK_TOKENS == 4
          and LRU_N_KEYS == 16 and LRU_TRACE_LEN == 400,
          "★★★ **口径守卫**：**每个请求长度都是块大小的整数倍**（★ 否则「末尾不满一块」的口径分歧会把"
          "「断言捕获」污染成假失败）；★ LRU 容量网格里有一档 `≥` 去重键数（`B.2` 靠它）",
          "**块大小 / 请求长度 / 容量网格被动过** —— ★ 先查 `PREFIX_BLOCK_TOKENS` 与 `LRU_CAP_GRID`")

    # ── D.3c ★★★ **先钉住【会被索引】的网格**（★ 跑在任何计算之前）──────────
    #   `B.1` 用 `rows[16]` / `rows[8]` / `rows[2]` 作**字典键**取出 LRU/FIFO 命中数 ——
    #   ★★ 若 `LRU_CAP_GRID` 的元素被动过（如 `2 → 3`），会在**逐元素钉住（`D.7`）之前**就 `KeyError`
    #      ⇒ 那是**崩溃捕获**，而本库明文「**崩溃捕获不算真捕获**」。⇒ 把会被索引的网格**先钉在这里**。
    check("D.3c", specs.LRU_CAP_GRID == (2, 4, 8, 16),
          "★★★ **先钉住会被【索引】的网格**：`LRU_CAP_GRID` = `(2, 4, 8, 16)`"
          "（`B.1` 用 `rows[2]` / `rows[8]` / `rows[16]` 索引它）；★ 钉在**计算之前** ⇒ "
          "网格被动过是**断言捕获**，不是 `KeyError`（崩溃捕获不算真捕获）",
          "**被索引的容量网格被动过** —— ★ 它原本会先崩在 `rows[2]` 上")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 前沿分支 · 独立性
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 前沿分支 · 独立性线索")
    print("=" * 78)

    reqs = build_requests(specs.PREFIX_SEED, specs.PREFIX_N_REQ)

    # ── D.4 ★★★ 守卫实验 ①：把哈希改成常数 ─────────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：把「块键」改成**常数** ──")
    cache_good = {}
    cache_bad = {}
    good0, _ = measured_match(reqs[1], cache_good, mk_key)
    measured_match(reqs[0], cache_good, mk_key)          # 先把 req0 放进好缓存
    cache_good = {}
    measured_match(reqs[0], cache_good, mk_key)
    good1, _ = measured_match(reqs[1], cache_good, mk_key)
    cache_c = {}
    measured_match(reqs[0], cache_c, mk_key_constant)
    bad1, _ = measured_match(reqs[1], cache_c, mk_key_constant)
    ref1 = reference_match(reqs, 1)
    print("     req1 命中：正确键 %d · 参照(逐 token) %d · **常数键 %d**" % (good1, ref1, bad1))
    print("     ★★★ **读法**：常数键把 req1 的首块也「命中」了（缓存里有 req0 留下的键 `0`）——")
    print("        于是它命中到一个**内容完全不同**的块。")
    check("D.4", good1 == ref1 and bad1 != ref1
          and measured_match(reqs[0], {}, mk_key)[0] == 0,
          "★★★ **守卫成立**：把块键改成常数后，req1 的命中块数**从 %d 变成 %d**（参照 = %d）⇒"
          " ★★ **`A.1` 的「两条路径一致」对「键唯一标识内容」这条前提是敏感的**"
          % (good1, bad1, ref1),
          "**守卫失效** —— ★ 常数键下命中数居然不变（那 `A.1` 抓不到碰撞 ⇒ 需要加判据）")

    # ── D.5 ★★ 引文逐字（前缀缓存的两条口径）──────────────────────────
    print()
    check("D.5", specs.VLLM_FULL_BLOCKS_QUOTE == "We only cache full blocks."
          and specs.SGLANG_PAGE_LAYOUT_QUOTE
          == "These KV cache tensors are stored on the GPU in a paged layout, where the size of "
             "each page is equivalent to one token.",
          "★★ **两条引文逐字钉住**：★ vLLM 前缀缓存「**只缓存整块**」（`We only cache full blocks.`）；"
          "★ SGLang 的 `page = 1 token`。★★ 两者**不是同一个对象**：一条讲**缓存粒度**、一条讲**分页粒度**；"
          "★★ 载体 = vLLM 文档页 + LMSYS 博客，抓取日 %s" % specs.KV_FETCH_DAY,
          "**引文常量被动过** —— ★ 先查 `We only cache full blocks.` 与 `equivalent to one token`")

    # ── D.6 ★★★ 前沿分支：循环模型（KDA / SSM）⇒ 三概念必须拆开 ────────
    check("D.6", specs.KDA_THREE_CONCEPTS == ("物理块大小", "调度器对齐", "前缀匹配单元")
          and specs.KDA_NOTE_QUOTE
          == "vLLM 把三个原本绑在一起的概念拆开——物理块大小 / 调度器对齐 / 前缀匹配单元，"
             "并规定命中部分块时先把缓存状态拷进私有目的地再扩展（copy-on-write）",
          "★★★ **前沿分支的三概念钉住**：对**循环模型**（KDA / SSM，**不存 per-token KV 对**），"
          "vLLM 把**物理块大小 / 调度器对齐 / 前缀匹配单元**三个概念**拆开**（★ 命中部分块时"
          "**先拷进私有目的地再扩展**，即 copy-on-write）；\n"
          "        ★★★ **来源级别**：**`[B]` 且【经检索转述】**（本库 `45-kernel-lifecycle/"
          "requirement-discovery/01:187` 自标「经检索转述」，**非逐字**）⇒ **不得**写成逐字引文、"
          "**不得**升格为 `[A]`",
          "**三概念或转述记录被动过** —— ★ 先查 `KDA_THREE_CONCEPTS` 与 `KDA_NOTE_QUOTE`")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.PREFIX_BLOCK_TOKENS == 4 and specs.PREFIX_SEED == 20261009
          and specs.PREFIX_N_REQ == 8 and specs.LRU_SEED == 20261009
          and specs.LRU_CAP_GRID == (2, 4, 8, 16)
          and specs.KDA_THREE_CONCEPTS == ("物理块大小", "调度器对齐", "前缀匹配单元")
          and specs.VLLM_DEFAULT_BLOCK_SIZE == 16 and specs.SGLANG_PAGE_TOKENS == 1
          and specs.KV_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == ()
          and LRU_N_KEYS == 16 and LRU_TRACE_LEN == 400
          and KV_MIX_A == 131 and KV_MIX_B == 7 and KV_MOD == 1000000007,
          "★★★ **两个元组逐元素 + 十个标量**钉住（块 · 两个种子 · 请求数 · 容量网格 · 三概念 · "
          "vLLM 块 · SGLang page · 抓取日 · LRU 键数/长度 · 哈希三个参数）；★★ `UNSOURCED == ()` 一并钉住",
          "**某个常量被动过**")

    # ── D.8 ★★★ 独立性（源码级）：参照不碰哈希 ──────────────────────────
    ref_src = textwrap.dedent(inspect.getsource(lcp_len)) + \
        textwrap.dedent(inspect.getsource(reference_match))
    ref_names = set()
    for n in ast.walk(ast.parse(ref_src)):
        if isinstance(n, ast.Name):
            ref_names.add(n.id)
        elif isinstance(n, ast.Attribute):
            ref_names.add(n.attr)
    meas_names = set()
    for f in (mk_key, measured_match):
        for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(f)))):
            if isinstance(n, ast.Name):
                meas_names.add(n.id)
            elif isinstance(n, ast.Attribute):
                meas_names.add(n.attr)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                meas_names.add(n.name)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：参照侧不碰哈希 ──")
    print("     参照侧（`lcp_len` + `reference_match`）里出现 `mk_key` = %s ；出现 `cache` = %s"
          % ("mk_key" in ref_names, "cache" in ref_names))
    print("     ★★★ **降级措辞**：这只是「**参照不调用被测的键函数、也不碰缓存表**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4` 的**行为**兜底。")
    check("D.8", ("mk_key" not in ref_names) and ("mk_key_constant" not in ref_names)
          and ("cache" not in ref_names) and ("measured_match" not in ref_names)
          and ("mk_key" in meas_names),
          "★★ **源码级线索成立**：**参照侧（逐 token 比对）不调用任何被测的键函数、也不碰缓存表**"
          "（★ 两侧都审）—— ★★ 于是「两条路径一致」**不是恒真**，而是**两条独立路径的会合**",
          "**参照引用了被测** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 边界 ─────────────────────────────────────────────────────
    B = specs.PREFIX_BLOCK_TOKENS
    cache = {}
    z = measured_match((), cache, mk_key)
    oneblock = measured_match((1, 2, 3, 4), cache, mk_key)
    same = measured_match((1, 2, 3, 4), cache, mk_key)
    short = measured_match((1, 2, 3), {}, mk_key)
    check("D.9", z[0] == 0 and oneblock[0] == 0 and same[0] == 1 and short[0] == 0,
          "★★ **边界**：**空请求**命中 `0`；第一个**恰好一整块**的请求命中 `0`、**重复它**则命中 `1`；"
          "★ 长度**不足一块**（`%d` 个 token < `%d`）的请求命中 `0`（★ 末块不满 ⇒ 不入缓存、也不命中）"
          % (3, B),
          "**边界值被动过**")


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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：`hash → 块` vs **逐 token 朴素比对**"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：本主题的机制是 **`[E]` 级（本库实现）**，口径取自 vLLM 文档（`[B]`，"
          "「只缓存整块」/「键含前缀链」）；★ 前沿分支的三概念来自 `45-kernel-lifecycle`（**经检索转述**）。")
    print("★ 本程序只核对「**前缀共享的算术**」——**不是**任何真实引擎的前缀缓存。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_prefix_and_eviction.py —— 前缀共享与淘汰：hash → 块 / 命中率 / LRU 驱逐")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **系统侧（前缀共享与淘汰）**；★★ 本程序**不碰** kernel 侧与量化（均只回链）。")
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
