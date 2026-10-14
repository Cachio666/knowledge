#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_sglang_and_radix.py —— **RadixAttention 的数据结构**（radix tree / 前缀查找 / LRU 淘汰）
+ 它与 vLLM 的**两处口径差**（`page = 1 token` vs `block = 16`；**默认行为** vs 需 `--enable-prefix-caching`）。

配套文档：`02-sglang-and-radix.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本程序【有真 A 类】**（★ 这是 `engines` 主题里**唯一**有 A 类的一篇 —— 见计划 `§1.3.4`）：
  ★ 引擎的**【行为与性能】**本机无处可验（无 GPU、无 SGLang）——
  ★★ **但 RadixAttention 的【数据结构】是可写独立参照的**：
  · **被测** = `RadixTree`（**压缩前缀树**）—— 插入 / **最长前缀查找** / 成员判定；
  · **参照** = `NaivePrefixStore`（**逐条序列、逐 token 朴素比对**，★ **完全不碰树结构**）；
  · **参照（空间）** = `UncompressedTrie`（**未压缩 trie**，一条边一个 token）；
  · **参照（淘汰）** = `ListCache`（**列表线性扫描**判成员 + 扫最小时间戳选受害者）。
  ★★★ **判据**：任意构造下，`RadixTree` 与两条参照路径的**结果必须一致**；★★ 且断言
    **radix 的节点数 < 未压缩 trie 的节点数**（★ 这是官方博客「space-efficient alternative」的**量化**）。
  ★★ **守卫**（`D.4`/`D.5`）：把 radix 的**切分**改坏（不切分）/ 把 LRU 改成**逐出最近用过的**
    ⇒ **断言必须失败**（`CONTRIBUTING:235` 的要求）。

★★★ **本程序【只写数据结构与两处口径差】** —— ★ **机制一律回链**（`P37`）：
  · ★★ **前缀共享的系统侧机制**（`hash → 块` / 命中率 / 驱逐）→ `120-inference/kv-cache/03`；
  · **分页 / 块管理** → `120-inference/kv-cache/02`；
  · **PagedAttention 的算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84`。
  ★★★ 本程序**不重算** `kv-cache/03` 的「命中块数 / 命中率」—— 那是**块粒度（16）**的世界；
    本程序的对象是**token 粒度（1）的 radix tree**，两者是**同一个知识对象的两个不同数据结构**。

★★★ **本程序的禁写清单（计划 `§0.7 C.5`）**：
  · `F1`（**不得写那类「没有第二个独立主体」的断言**）—— ★ `page` / `block` 必须**写明是哪一家**；
  · `F5`（**不写** vLLM APC 默认开启 —— 取数到的页面**未声明**默认开/关）。

★★★ **本程序的判据纪律（★ 形态 A 的根治）**：★ 抬头写了「不裸写某某」这条纪律，
  本程序里**必须有一条断言能抓住「没做到」** —— 见 `D.9`：
  ★★ 判据是**等式**「裸写计数 == 显式豁免清单覆盖的计数」，**不是「≤ N」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **最长前缀查找**：`RadixTree` == `NaivePrefixStore`（★ 两条独立路径会合）；
  · `A.2` ★★★ **空间**：radix 节点数 **<** 未压缩 trie 节点数（压缩真的发生）；
  · `A.3` ★★★ **LRU 淘汰**：`RadixCache` 的命中/受害序列 == `ListCache`（列表扫描参照）；
  · `B.1` ★ **空间账**：`节点数 / token 数`（压缩比）；
  · `B.2` ★ **命中率随容量单调不降**（★ 容量 ≥ 相异键数 ⇒ 只剩强制缺失）；
  · `C.1` ★★★ **同名不同指**：`page = 1` vs `block = 16` ⇒ 同一个 token 数的块数**差 16 倍**；
  · `C.2` ★★★ **默认行为**：SGLang 默认复用 vs vLLM 需开关（★ 且 vLLM 的默认值**未取到**，`F5`）；
  · `C.3` ★★ **前缀查找的边界**（空查询 / 恰好命中 / 查询长于一切）；
  · `D.4` / `D.5` ★★★ **守卫实验**：把切分改坏 / 把 LRU 改成逐出最近用过的 ⇒ 断言必须失败。
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
# ★★★ 失败标记：本主题（engines）取 `-- FAIL: engines`；`mutate.py` 用 AST 读出（**不写死**）。
ASSERT_FAIL_MARK = "-- FAIL: engines"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                     "-- FAIL: decoding-strategies", "-- FAIL: serving-infra")

# ★★ **术语纪律**（本域 README §同名不同指）—— ★ 全部转成**可机械判定的常量**，由 `D.9` 守住：
TERM_KV_CACHE = "KV 缓存"
TERM_SCHEDULER = "请求调度器"
LIMIT_QUOTE = "不是任何真实引擎，也不是真机"

# ★★★ **术语纪律的第二半**（★ 形态 A：「纪律宣称了而断言没盯住」，本域已栽过两次）：
#   本程序源码里**裸写那个被禁的裸词**（`F1` 明文禁的那类断言用词）的【计数】
#   必须**恰好等于**下面这份**显式豁免清单**覆盖的计数 ——
#   ★★★ 判据是【计数 == 豁免覆盖数】，**不是「≤ N」**（「≤ N」挡不住**新增**一处裸写）。
#   ★ 每一项都写明【为什么豁免】：
#     ① 抬头注释里**提到**这条纪律本身 —— 那是 **mention**（它写该词**恰恰因为在规定不许写它**）：
#        · ★★★ 本主题一律【写明是哪一家】：不裸写「业界」这类无第二个独立主体的断言（`F1`）。
#     ② `D.9` 的判据文案里**复述**这条纪律 —— ★ 同为 mention。
BARE_ATTRIBUTION_EXEMPT = (
    "不裸写「业界」这类无第二个独立主体的断言",
    "**不裸写「业界」**",
)

# ★★★ **术语纪律的第三半**（★ `D3` 根治：本域抬头宣称「不裸写那两个 ASCII 词」而此前**没有任何断言盯住**，
#   本主题在自己身上栽了第三次 —— 见计划 `§5` v1.10 ⇒ 现在对它建**同形态的等式**）：
#   判据 = 【**等式**】「裸写计数 == 显式豁免清单覆盖数」（**不是「≤ N」**）；★ 计数码用**拼接**避免自计。
#   ★ 裸写 = 被禁 ASCII 词**紧邻中文**（中间只隔空白 / 反引号）。
#   ★ 豁免清单（**逐项带理由**）：① 下面两条字符串**本身就是**抬头那条纪律的 **mention** ——
#     它们写该 ASCII 词**恰恰因为在规定不许写它**。★ 它们是**唯一**被豁免的裸写；
#     ★★ 任何**新增**的裸写都会让等式 `n_bare == n_exempt` 不成立。
BARE_ASCII_WORDS = ("ca" + "che", "sched" + "uler")
TERM_BARE_EXEMPT = (
    "本域一律写「**KV 缓存**」全称（不裸写 `cache`，那是 L1/L2 与 kernel cache）",
    "本域一律写「**请求调度器**」全称（不裸写 `scheduler`，那是 GPU 的 warp scheduler）",
)


def _is_cjk(ch):
    """★ 是否 CJK 表意字（★ **不含** CJK 标点 —— 「，」「）」不算「中文语境」）。"""
    return ch != "" and "一" <= ch <= "鿿"


def _count_bare_ascii(src):
    """★★ 数「**裸写**」：被禁 ASCII 词**紧邻中文**（中间只隔空白 / 反引号）的出现次数。

    ★★★ 这是 `D.9` 的**被测量** —— ★ 判据是**等式**（`_count_bare_ascii(src) == 豁免覆盖数`）。
    """
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

# ★★ **本程序【有 A 类】** —— `D.0` 断言本文件里 A 类判据的条数 **> 0**（★ 与 `01`/`03` 的
#   「机械证明没有 A 类」互为镜像；★ 它是**结构性**的，不是钉住某个数）。
EXPECTED_MIN_A = 3


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
# 1. 被测（I）：**压缩前缀树（radix tree）** —— 边上挂的是 token 子序列
# ===========================================================================
def _mk_node(edge):
    return {"edge": tuple(edge), "children": {}, "terms": set()}


class RadixTree:
    """★★★ **被测实现（I）**：压缩前缀树 —— 插入 / 最长前缀查找 / 成员判定。

    ★★ 与未压缩 trie 的差别**只在插入时的【切分】与【合并】**：一条边可以挂**多个** token，
      ★ 于是「链」被压成一个节点（这正是官方博客说的 `space-efficient alternative to a trie`）。
    ★★★ `corrupt` 只用于 `D.4` 的失败实验：
      · `"no_split"`  —— 插入遇到**部分匹配**时**不切分**边（把不同的序列焊在一起）；
      · `"off_by_one"` —— 最长前缀查找**少算一个 token**。
    """

    def __init__(self, corrupt=None):
        self.corrupt = corrupt
        self.root = _mk_node(())
        self.n_nodes = 1

    def _add(self, edge):
        self.n_nodes += 1
        return _mk_node(edge)

    def insert(self, tokens, sid):
        tokens = tuple(tokens)
        node = self.root
        i = 0
        while i < len(tokens):
            t = tokens[i]
            child = node["children"].get(t)
            if child is None:
                nd = self._add(tokens[i:])
                nd["terms"].add(sid)
                node["children"][t] = nd
                return
            edge = child["edge"]
            j = 0
            lim = min(len(edge), len(tokens) - i)
            while j < lim and edge[j] == tokens[i + j]:
                j += 1
            if j == len(edge):
                if i + j == len(tokens):
                    child["terms"].add(sid)
                    return
                node = child
                i += j
                continue
            # ★ 部分匹配：j < len(edge) —— 正解是**在这里切分**这条边
            if self.corrupt == "no_split":
                node = child                     # ★ 坏法：不切分，直接往下塞
                i += j
                continue
            mid = self._add(edge[:j])
            mid["children"][edge[j]] = child
            child["edge"] = edge[j:]
            node["children"][t] = mid
            if i + j == len(tokens):
                mid["terms"].add(sid)
                return
            nd = self._add(tokens[i + j:])
            nd["terms"].add(sid)
            mid["children"][tokens[i + j]] = nd
            return

    def longest_prefix_len(self, q):
        q = tuple(q)
        node = self.root
        i = 0
        matched = 0
        while i < len(q):
            child = node["children"].get(q[i])
            if child is None:
                break
            edge = child["edge"]
            j = 0
            lim = min(len(edge), len(q) - i)
            while j < lim and edge[j] == q[i + j]:
                j += 1
            if self.corrupt == "off_by_one":
                matched += max(j - 1, 0)
                break
            matched += j
            if j < len(edge):
                break
            node = child
            i += j
        return matched

    def has(self, tokens):
        """★ 成员判定：**整条 token 序列**是否被存过（★ 走压缩边）。"""
        tokens = tuple(tokens)
        node = self.root
        i = 0
        while i < len(tokens):
            child = node["children"].get(tokens[i])
            if child is None:
                return False
            edge = child["edge"]
            if tuple(tokens[i:i + len(edge)]) != edge:
                return False
            i += len(edge)
            node = child
        return len(node["terms"]) > 0


# ===========================================================================
# 2. 参照（I）：**逐条序列、逐 token 朴素比对**（★ 完全不碰树结构）
# ===========================================================================
class NaivePrefixStore:
    """★★★ **参照实现（I）**：把每条序列**原样存进一个列表**；
    最长前缀 = 对**每一条**已存序列**逐 token** 比对后取最大。

    ★★ 它**没有树、没有边、没有节点** —— 与 `RadixTree` **不共享任何一行**（`D.8` 核）。
    """

    def __init__(self):
        self.seqs = []

    def insert(self, tokens, sid):
        self.seqs.append(tuple(tokens))

    def longest_prefix_len(self, q):
        q = tuple(q)
        best = 0
        for s in self.seqs:
            j = 0
            while j < len(s) and j < len(q) and s[j] == q[j]:
                j += 1
            if j > best:
                best = j
        return best

    def has(self, tokens):
        return tuple(tokens) in self.seqs


# ===========================================================================
# 3. 参照（II）：**未压缩 trie**（一条边一个 token）—— 只用来量化「压缩」
# ===========================================================================
class UncompressedTrie:
    """★★ **参照实现（II）**：未压缩 trie —— **每条边恰好一个 token**（★ 节点数可数）。"""

    def __init__(self):
        self.root = {}
        self.n_nodes = 1
        self.seqs = set()

    def insert(self, tokens, sid):
        node = self.root
        for t in tokens:
            if t not in node:
                node[t] = {}
                self.n_nodes += 1
            node = node[t]
        self.seqs.add(tuple(tokens))


# ===========================================================================
# 4. LRU 缓存（被测 vs 参照）—— 淘汰策略 = LRU（官方博客的逐字表述）
# ===========================================================================
class RadixCache:
    """★★★ **被测实现（III）**：成员判定走 **`RadixTree`**；淘汰 = **LRU**（最小时间戳）。

    ★★ 淘汰一个键之后，**用存活键重建树** —— 这是**本主题的实现**（不是 SGLang 的内部结构）。
    ★★★ `corrupt` 只用于 `D.5`：`"evict_mru"` ⇒ 逐出**最近用过**的那个（★ 与 LRU 相反）。
    """

    def __init__(self, cap, corrupt=None):
        self.cap = cap
        self.corrupt = corrupt
        self.tree = RadixTree()
        self.last = {}
        self.clock = 0
        self.contents = {}
        self.victims = []

    def _rebuild(self):
        self.tree = RadixTree()
        for k in self.last:
            self.tree.insert(self.contents[k], self.contents[k])

    def access(self, tokens):
        self.clock += 1
        key = tuple(tokens)
        if self.tree.has(key):
            self.last[key] = self.clock
            return "hit"
        if self.cap <= 0:
            return "miss"
        if len(self.last) >= self.cap:
            if self.corrupt == "evict_mru":
                victim = max(self.last, key=lambda k: self.last[k])   # ★ 坏法：逐出最近用过的
            else:
                victim = min(self.last, key=lambda k: self.last[k])
            self.victims.append(victim)
            del self.last[victim]
            del self.contents[victim]
            self._rebuild()
        self.last[key] = self.clock
        self.contents[key] = key
        self.tree.insert(key, key)
        return "miss"


class ListCache:
    """★★★ **参照实现（III）**：成员判定走**列表线性扫描**；淘汰 = 扫最小时间戳（★ 不碰树）。"""

    def __init__(self, cap):
        self.cap = cap
        self.stored = []          # [(key, last)]
        self.clock = 0
        self.victims = []

    def _find(self, key):
        for k, _t in self.stored:
            if k == key:
                return True
        return False

    def access(self, tokens):
        self.clock += 1
        key = tuple(tokens)
        if self._find(key):
            self.stored = [(k, self.clock if k == key else t) for k, t in self.stored]
            return "hit"
        if self.cap <= 0:
            return "miss"
        if len(self.stored) >= self.cap:
            victim = min(self.stored, key=lambda kt: kt[1])[0]
            self.victims.append(victim)
            self.stored = [(k, t) for k, t in self.stored if k != victim]
        self.stored.append((key, self.clock))
        return "miss"


# ===========================================================================
# 5. 构造：序列集 / 查询集 / 访问轨迹（★ 两实现**共用同一条**）
# ===========================================================================
def gen_seqs(seed, n_seq, len_grid, alphabet, n_seeds=2):
    """构造 `n_seq` 条序列：★ 前几个 token 从 `n_seeds` 个「种子」里取（**保证共享前缀**）。

    ★★ **共享前缀是故意的**：没有它，`A.2`（压缩）与 `A.1`（前缀查找）都会退化成恒真。
    """
    rng = np.random.default_rng(seed)
    seqs = []
    seeds = [tuple(int(x) for x in rng.integers(0, alphabet, size=2)) for _ in range(n_seeds)]
    for sid in range(n_seq):
        L = int(len_grid[int(rng.integers(0, len(len_grid)))])
        head = seeds[int(rng.integers(0, n_seeds))]
        tail = tuple(int(x) for x in rng.integers(0, alphabet, size=max(L - len(head), 0)))
        seqs.append(((head + tail)[:L], sid))
    return seqs


def gen_queries(seed, seqs, alphabet):
    """构造查询集：★ 已存序列的**前缀** + 随机串（★ 覆盖「恰好命中 / 查不到 / 部分命中」）。"""
    rng = np.random.default_rng(seed + 1)
    qs = []
    for s, _sid in seqs:
        for cut in range(len(s) + 1):
            qs.append(s[:cut])                       # ★ 已存序列的所有前缀
    for _ in range(20):
        L = int(rng.integers(0, 6))
        qs.append(tuple(int(x) for x in rng.integers(0, alphabet, size=L)))
    return qs


def gen_trace(seed, seqs, alphabet, length=120):
    """访问轨迹：从**一个小键池**里取（★ 保证既有命中也有未命中 ⇒ LRU 真的在动）。"""
    rng = np.random.default_rng(seed + 2)
    pool = [s for s, _ in seqs]
    trace = []
    for _ in range(length):
        if float(rng.random()) < 0.75:
            trace.append(pool[int(rng.integers(0, len(pool)))])
        else:
            L = int(rng.integers(1, 5))
            trace.append(tuple(int(x) for x in rng.integers(0, alphabet, size=L)))
    return trace


# ===========================================================================
# 6. A 类 —— ★★★ 三条**独立路径**（前缀查找 / 空间 / 淘汰）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **RadixAttention 的数据结构**：压缩前缀树（被测）vs 三条独立参照")
    print("=" * 78)
    print("  ★★★ 被测 = `RadixTree`（压缩前缀树）；参照 = `NaivePrefixStore`（逐 token 朴素比对）·")
    print("      `UncompressedTrie`（未压缩 trie）· `ListCache`（列表扫描 LRU）。")
    print("  ★★ 被测与三条参照**无共享代码**（`D.8` 在源码层核）。")

    seqs = gen_seqs(specs.RADIX_SEED, specs.RADIX_N_SEQ, specs.RADIX_SEQ_LEN_GRID, specs.RADIX_ALPHABET)
    queries = gen_queries(specs.RADIX_SEED, seqs, specs.RADIX_ALPHABET)
    radix, naive = RadixTree(), NaivePrefixStore()
    for s, sid in seqs:
        radix.insert(s, sid)
        naive.insert(s, sid)

    # ── A.1 ★★★ 最长前缀查找：压缩树 == 逐 token 朴素比对 ────────────────
    print()
    print("  ── A.1 ★★★ **最长前缀查找**：`RadixTree` == `NaivePrefixStore` ──")
    bad = [q for q in queries if radix.longest_prefix_len(q) != naive.longest_prefix_len(q)]
    print("     ★ 序列 %d 条 · 查询 %d 条（含**所有**已存序列的前缀）· 不一致 = %d 条"
          % (len(seqs), len(queries), len(bad)))
    print("     ★★ 样例（前 3 条已存序列）：%s" % "; ".join("%s⇒%d" % (s, radix.longest_prefix_len(s)) for s, _ in seqs[:3]))
    check("A.1", not bad and len(queries) == 85 and len(seqs) == specs.RADIX_N_SEQ,
          "★★★ **判据成立**：**%d 条查询**（★ **等式钉住 `== 85`**，`D2` 订正）上，"
          "压缩前缀树与**逐 token 朴素比对**给出的最长前缀**逐条一致**"
          " —— ★★ 一面走**切分后的边**，一面**逐 token 数**，两条独立路径会合" % len(queries),
          "**两条路径的最长前缀不一致（或查询数不是 85）** —— ★ 先查 `gen_queries` 与 `RadixTree.insert` 的**切分**逻辑")

    # ── A.2 ★★★ 空间：radix 节点数 < 未压缩 trie 节点数 ─────────────────
    print()
    print("  ── A.2 ★★★ **空间**：radix 节点数 **<** 未压缩 trie 节点数 ──")
    trie = UncompressedTrie()
    for s, sid in seqs:
        trie.insert(s, sid)
    total_tokens = sum(len(s) for s, _ in seqs)
    identical = all(radix.has(s) == naive.has(s) == (s in trie.seqs) for s, _ in seqs)
    print("     总 token 数 = %d · **radix 节点数** = %d · **未压缩 trie 节点数** = %d"
          % (total_tokens, radix.n_nodes, trie.n_nodes))
    print("     ★★ 成员判定在三条路径上一致 = %s" % identical)
    check("A.2", radix.n_nodes == 16 and trie.n_nodes == 31 and total_tokens == 53 and identical,
          "★★★ **判据成立**：radix **%d 个节点** < 未压缩 trie **%d 个**（总 token %d）—— "
          "★★ **三格都用等式钉住**（`D2` 订正：此前只有 `<` 不等式，正文的 `16`/`31`/`53` 未被断言钉住）；"
          "★★ 差的就是「一条边挂多个 token」压掉的**链**；★★ 且三条路径的**成员判定**一致"
          % (radix.n_nodes, trie.n_nodes, total_tokens),
          "**压缩没有发生（或三个整数不是 16 / 31 / 53）** —— ★ 那 `A.1` 的相等就退化成「两条路径都是同一种结构」了")

    # ── A.3 ★★★ LRU：RadixCache == ListCache ────────────────────────────
    print()
    print("  ── A.3 ★★★ **LRU 淘汰**：`RadixCache`（树判成员）== `ListCache`（列表扫描）──")
    trace = gen_trace(specs.RADIX_SEED, seqs, specs.RADIX_ALPHABET)
    cap = specs.RADIX_CAP_GRID[1]
    rc, lc = RadixCache(cap), ListCache(cap)
    rhit = [rc.access(t) for t in trace]
    lhit = [lc.access(t) for t in trace]
    print("     容量 = %d 条 · 轨迹 %d 步 · 命中数 = %d" % (cap, len(trace), rhit.count("hit")))
    print("     ★★ 命中/未命中序列一致 = %s · 受害序列一致 = %s"
          % (rhit == lhit, rc.victims == lc.victims))
    check("A.3", rhit == lhit and rc.victims == lc.victims
          and len(trace) == 120 and len(rc.victims) == 85 and rhit.count("hit") == 31,
          "★★★ **判据成立**：`RadixCache`（成员判定走**压缩树**）与 `ListCache`（**列表扫描**）在 **%d 步**"
          "轨迹上**命中序列与受害序列都一致**（受害 **%d** 次、命中 **%d** 次）—— "
          "★★ **三个整数都用等式钉住**（`D2` 订正）；★★ 又两条独立路径"
          % (len(trace), len(rc.victims), rhit.count("hit")),
          "**两条路径的淘汰不一致（或 120 / 85 / 31 被改）** —— ★ 先查 `RadixCache._rebuild` 与最小时间戳的选择")


# ===========================================================================
# 7. B 类：空间账 · 命中率 vs 容量
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · `[B]` 级的账：**压缩的收益**（节点数 / token 数）· **命中率 vs 容量**")
    print("=" * 78)
    print("  ★★ 载体 = LMSYS Org 官方博客 `https://lmsys.org/blog/2024-01-17-sglang/`，"
          "抓取日 **%s**。" % specs.ENGINES_FETCH_DAY)
    print("     逐字：`%s`" % specs.SGLANG_RADIX_DEF_QUOTE)

    # ── B.1 ★ 空间账 ─────────────────────────────────────────────────────
    print()
    print("  ── B.1 ★ **空间账**：`节点数 / 总 token 数`（越小 = 压缩越狠）──")
    seqs = gen_seqs(specs.RADIX_SEED, specs.RADIX_N_SEQ, specs.RADIX_SEQ_LEN_GRID, specs.RADIX_ALPHABET)
    radix, trie = RadixTree(), UncompressedTrie()
    for s, sid in seqs:
        radix.insert(s, sid)
        trie.insert(s, sid)
    total_tokens = sum(len(s) for s, _ in seqs)
    print("     %-22s %-14s %-18s" % ("结构", "节点数", "节点/token"))
    print("     %-22s %-14d %.4f" % ("压缩前缀树 (radix)", radix.n_nodes, radix.n_nodes / total_tokens))
    print("     %-22s %-14d %.4f" % ("未压缩 trie", trie.n_nodes, trie.n_nodes / total_tokens))
    print("     ★★ **读法**：radix 的 `节点/token` **小于** trie 的 —— 被压掉的是**只有单孩子**的链。")
    check("B.1", round(radix.n_nodes / total_tokens, 4) == 0.3019
          and round(trie.n_nodes / total_tokens, 4) == 0.5849
          and radix.n_nodes == 16 and trie.n_nodes == 31 and total_tokens == 53,
          "★ **判据成立**：radix `节点/token` = **%.4f** < trie 的 **%.4f**（同一个序列集）—— "
          "★★ **两个比值都用等式钉住**（`0.3019` / `0.5849`，`D2` 订正：此前只有 `<`）"
          % (radix.n_nodes / total_tokens, trie.n_nodes / total_tokens),
          "**压缩比被动过** —— ★ 先查 `RadixTree` 的切分与 `UncompressedTrie` 的节点计数")

    # ── B.2 ★ 命中率 vs 容量（单调不降）──────────────────────────────────
    print()
    print("  ── B.2 ★ **命中率 vs 容量**（★ 单调不降 + 大容量只剩强制缺失）──")
    seqs = gen_seqs(specs.RADIX_SEED, specs.RADIX_N_SEQ, specs.RADIX_SEQ_LEN_GRID, specs.RADIX_ALPHABET)
    trace = gen_trace(specs.RADIX_SEED, seqs, specs.RADIX_ALPHABET)
    rates = {}
    hits_by_cap = {}
    for cap in specs.RADIX_CAP_GRID:
        lc = ListCache(cap)
        hits = sum(1 for t in trace if lc.access(t) == "hit")
        hits_by_cap[cap] = hits
        rates[cap] = hits / len(trace)
    print("     %-10s %-8s %-12s" % ("容量(条)", "命中数", "命中率"))
    for cap in specs.RADIX_CAP_GRID:
        print("     %-10d %-8d %.4f" % (cap, hits_by_cap[cap], rates[cap]))
    monotone = all(rates[specs.RADIX_CAP_GRID[i]] <= rates[specs.RADIX_CAP_GRID[i + 1]]
                   for i in range(len(specs.RADIX_CAP_GRID) - 1))
    print("     ★★ 单调不降 = %s ；最大容量命中率 = %.4f" % (monotone, rates[specs.RADIX_CAP_GRID[-1]]))
    check("B.2", monotone and hits_by_cap == {2: 17, 4: 31, 8: 46, 16: 65}
          and {c: round(rates[c], 4) for c in specs.RADIX_CAP_GRID}
          == {2: 0.1417, 4: 0.2583, 8: 0.3833, 16: 0.5417},
          "★ **判据成立**：命中率随容量**单调不降**（`%.4f → %.4f`）—— ★★ **四档命中数与命中率都用等式钉住**"
          "（`D2` 订正：此前只有「单调不增」）；★ 大容量下仍有未命中的是**强制缺失**（首次出现）"
          % (rates[specs.RADIX_CAP_GRID[0]], rates[specs.RADIX_CAP_GRID[-1]]),
          "**命中率非单调（或四档命中数不是 17/31/46/65）** —— ★ 先查 `ListCache` 的淘汰与轨迹生成")


# ===========================================================================
# 8. C 类：同名不同指 · 默认行为 · 前缀查找边界
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**同名不同指** · **默认行为** · **前缀查找的边界**")
    print("=" * 78)

    # ── C.1 ★★★ 同名不同指：page = 1 vs block = 16 ─────────────────────
    print()
    print("  ── C.1 ★★★ **同名不同指**：`page = 1`（SGLang） vs `block = 16`（vLLM）──")
    L = 4096
    blocks_sglang = L // specs.SGLANG_PAGE_TOKENS
    blocks_vllm = -(-L // specs.VLLM_BLOCK_SIZE_TOKENS)
    print("     `%s` = %d（SGLang 的 `page`）· `%s` = %d（vLLM 的 `block`）"
          % (specs.SGLANG_PAGE_LAYOUT_QUOTE.split(" where")[0], specs.SGLANG_PAGE_TOKENS,
             specs.VLLM_BLOCK_SIZE_SOURCE_QUOTE, specs.VLLM_BLOCK_SIZE_TOKENS))
    print("     对 %d 个 token：SGLang 数出 **%d 页** · vLLM 数出 **%d 块** ⇒ **差 %d 倍**"
          % (L, blocks_sglang, blocks_vllm, blocks_sglang // blocks_vllm))
    print("     ★★★ 消歧记录：`%s`" % specs.PAGE_BLOCK_DISAMBIGUATION)
    print("     ★★ **读法**：把两家粒度的数**互相套用**会**差 16 倍** ⇒ 引用时必须**写明是哪一家**。")
    check("C.1", blocks_sglang // blocks_vllm == 16
          and specs.SGLANG_PAGE_TOKENS == 1 and specs.VLLM_BLOCK_SIZE_TOKENS == 16
          and specs.SGLANG_PAGE_TOKENS != specs.VLLM_BLOCK_SIZE_TOKENS,
          "★★★ **判据成立**：同一段 KV，SGLang 的 `page=1` 数出 **%d 页**、vLLM 的 `block=16` 数出 **%d 块**"
          " ⇒ **差 %d 倍** —— ★★ 两家的粒度**同名不同指**（`F1`：不可互相套用）"
          % (blocks_sglang, blocks_vllm, blocks_sglang // blocks_vllm),
          "**块粒度被动过** —— ★ 先查 `SGLANG_PAGE_TOKENS` 与 `VLLM_BLOCK_SIZE_TOKENS`")

    # ── C.2 ★★★ 默认行为：SGLang 默认 vs vLLM 需开关 ────────────────────
    print()
    print("  ── C.2 ★★★ **默认行为**：SGLang 默认复用 vs vLLM 需开关 ──")
    sglang_default = specs.SGLANG_PREFIX_REUSE_IS_DEFAULT
    vllm_default = None            # ★ 未取到（`F5`）—— 用 `None` 表示「未声明」
    print("     SGLang：`%s`" % specs.SGLANG_RADIX_RETENTION_QUOTE)
    print("     vLLM 的开关：`%s`" % specs.VLLM_ENABLE_PREFIX_CACHING_QUOTE)
    print("     ★★★ 判定：SGLang 的「默认即复用」= **%s**（应 `True`）；vLLM 的默认值 = **%s**（应 `None`，"
          "★ **未取到 ⇒ 不得臆断**）" % (sglang_default, vllm_default))
    print("     ★★ 读法：这一条**不是**「谁更好」，而是「**同一个能力，一家的默认值是另一家的开关**」。")
    check("C.2", sglang_default is True and vllm_default is None
          and specs.SGLANG_PREFIX_REUSE_IS_DEFAULT != (vllm_default is True),
          "★★★ **判据成立**：SGLang 的**默认行为**是复用（`%s`）；vLLM 需 `--enable-prefix-caching`，"
          "且其默认值**未取到**（`F5`）⇒ ★ **两家的默认值截然不同**" % specs.SGLANG_PREFIX_REUSE_IS_DEFAULT,
          "**默认行为被动过** —— ★ 先查 `SGLANG_PREFIX_REUSE_IS_DEFAULT` 与 vLLM 开关引文")

    # ── C.3 ★★ 前缀查找的边界 ───────────────────────────────────────────
    print()
    print("  ── C.3 ★★ **前缀查找的边界**（空查询 / 恰好命中 / 查询长于一切）──")
    rt, nv = RadixTree(), NaivePrefixStore()
    rt.insert((1, 2, 3), 0)
    nv.insert((1, 2, 3), 0)
    rt.insert((1, 2, 9), 1)
    nv.insert((1, 2, 9), 1)
    cases = {"空查询": (), "恰好命中": (1, 2, 3), "共享前二": (1, 2, 4), "首 token 不匹配": (7,), "查询长于一切": (1, 2, 3, 4, 5)}
    got = {k: (rt.longest_prefix_len(v), nv.longest_prefix_len(v)) for k, v in cases.items()}
    print("     %-16s %-8s %-8s" % ("查询类型", "radix", "朴素"))
    for k, (a, b) in got.items():
        print("     %-16s %-8d %-8d" % (k, a, b))
    print("     ★★ 读法：空查询 ⇒ `0`；恰好命中 ⇒ 全长；共享前二 ⇒ `2`；首 token 不匹配 ⇒ `0`；")
    print("        查询长于一切 ⇒ **停在已存的最长**（不会越界）。")
    check("C.3", got["空查询"] == (0, 0) and got["恰好命中"] == (3, 3) and got["共享前二"] == (2, 2)
          and got["首 token 不匹配"] == (0, 0) and got["查询长于一切"] == (3, 3),
          "★★ **判据成立**：五类边界上 radix 与朴素**逐条一致**（空 `0` / 命中全长 / 共享 `2` / 不匹配 `0` / 超长停在 `3`）",
          "**边界值被动过** —— ★ 先查 `RadixTree.longest_prefix_len` 的循环终止条件")


# ===========================================================================
# 9. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无 SGLang**。")
    print("      ★★ 但**本程序的 A 类是真的**：**三条独立参照**（朴素比对 / 未压缩 trie / 列表扫描 LRU）。")
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

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: engines" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: engines`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: batching-and-scheduling",
                                        "-- FAIL: decoding-strategies", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.ENGINES_FETCH_DAY == "2026-10-09"
          and specs.RADIX_ALPHABET == 8 and specs.RADIX_N_SEQ == 12
          and specs.RADIX_SEQ_LEN_GRID == (2, 3, 4, 5) and specs.RADIX_SEED == 20261009
          and specs.RADIX_CAP_GRID == (2, 4, 8, 16) and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ 算例：字母表 `%d` · 序列 `%d` 条 · 长度网格 `%s` · "
          "种子 `%d` · 容量网格 `%s`；★ `UNSOURCED == ()` 一并钉住"
          % (specs.ENGINES_FETCH_DAY, specs.RADIX_ALPHABET, specs.RADIX_N_SEQ,
             specs.RADIX_SEQ_LEN_GRID, specs.RADIX_SEED, specs.RADIX_CAP_GRID))

    # ★★★ 口径守卫（★ 跑在任何计算之前）：长度网格必须**含 ≥ 2 的长度**（否则没有可压缩的链，
    #   `A.2` 退化成恒真）；容量网格必须**含 ≥ 2 个不同值**（否则 `B.2` 的单调性无意义）。
    ok_grid = min(specs.RADIX_SEQ_LEN_GRID) >= 2 and len(set(specs.RADIX_CAP_GRID)) >= 2 \
        and min(specs.RADIX_CAP_GRID) >= 1 and min(specs.RADIX_CAP_GRID) < specs.RADIX_N_SEQ \
        and specs.RADIX_N_SEQ >= 4 and specs.RADIX_ALPHABET > 1
    print("     ★★ 长度网格最小值 ≥ 2（有可压缩的链）、容量网格有 ≥ 2 个不同值且**最小容量 < 序列数** = %s" % ok_grid)
    check("D.3b", ok_grid,
          "★★★ **口径守卫**：长度网格最小 `%d` ≥ 2（★ 否则 `A.2` 的压缩**退化成恒真**）；"
          "容量网格 `%s` 有 ≥ 2 个不同值且**最小容量 < 序列数**（★ 否则容量从第一档就装得下全部键，"
          "`B.2` 的单调性会退化成恒真）"
          % (min(specs.RADIX_SEQ_LEN_GRID), specs.RADIX_CAP_GRID),
          "**网格被动过** —— ★ 先查 `RADIX_SEQ_LEN_GRID` / `RADIX_CAP_GRID` / `RADIX_N_SEQ`")
    return len(FAILS) == before


# ===========================================================================
# 10. D 类（后半）：A 类存在性 · 守卫实验 · 逐元素 · 独立性 · 术语纪律
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


def part_D():
    print()
    print("=" * 78)
    print("D · **A 类存在性** · 守卫实验 · 逐元素钉住 · 术语纪律")
    print("=" * 78)

    # ── D.0 ★★★ 机械证明「本文件【有】A 类」（与 01/03 的镜像）───────────
    n_a = _count_a_checks_in_source()
    print("  ── D.0 ★★★ **机械证明**：AST 扫本文件，`check(\"A...\")` 的调用数 = **%d** ──" % n_a)
    check("D.0", n_a >= EXPECTED_MIN_A,
          "★★★ **判据成立**：本文件里 **A 类判据有 %d 条**（≥ %d）—— ★ 这是 `engines` 主题里"
          "**唯一**有真 A 类的一篇（引擎的**行为**无处可验，但**数据结构**可写独立参照）"
          % (n_a, EXPECTED_MIN_A),
          "★★ 本文件里 A 类判据少于 %d 条 —— ★ 那与「`02` 有真 A 类」**自相矛盾**" % EXPECTED_MIN_A)

    # ── D.4 ★★★ 守卫实验 ①：插入时**不切分**边 ──────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：插入遇到部分匹配时**不切分**边 ──")
    guard = [((1, 2), 0), ((1, 3), 1)]
    good, bad, nv = RadixTree(), RadixTree(corrupt="no_split"), NaivePrefixStore()
    for t, sid in guard:
        good.insert(t, sid)
        bad.insert(t, sid)
        nv.insert(t, sid)
    q = (1, 3)
    print("     构造：insert(1,2) → insert(1,3)；查询 `%s`" % (q,))
    print("     最长前缀：正确 %d · 坏法 %d · 参照(朴素) %d"
          % (good.longest_prefix_len(q), bad.longest_prefix_len(q), nv.longest_prefix_len(q)))
    print("     ★★★ **读法**：不切分 ⇒ 把 `(1,2)` 与 `(1,3)` **焊在一条边上** ⇒ 第二次插入被塞进子节点，")
    print("        于是查 `(1,3)` 只前缀匹配到 **1**（而不是 **2**）。")
    check("D.4", good.longest_prefix_len(q) == nv.longest_prefix_len(q) == 2
          and bad.longest_prefix_len(q) != nv.longest_prefix_len(q),
          "★★★ **守卫成立**：不切分后前缀长度**从 %d 变成 %d**（参照 = %d）⇒ "
          "★★ **`A.1` 对「部分匹配要切分」这条规则是敏感的**"
          % (good.longest_prefix_len(q), bad.longest_prefix_len(q), nv.longest_prefix_len(q)),
          "**守卫失效** —— ★ 不切分后前缀居然不变（那 `A.1` 抓不到它）")

    # ── D.5 ★★★ 守卫实验 ②：LRU 改成逐出**最近用过**的 ─────────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：把 LRU 改成**逐出最近用过的**（MRU）──")
    seqs = gen_seqs(specs.RADIX_SEED, specs.RADIX_N_SEQ, specs.RADIX_SEQ_LEN_GRID, specs.RADIX_ALPHABET)
    trace = gen_trace(specs.RADIX_SEED, seqs, specs.RADIX_ALPHABET)
    cap = specs.RADIX_CAP_GRID[1]
    good_c, bad_c = RadixCache(cap), RadixCache(cap, corrupt="evict_mru")
    for t in trace:
        good_c.access(t)
        bad_c.access(t)
    print("     容量 = %d · 轨迹 %d 步 · 正确版受害数 = %d · 坏法受害数 = %d"
          % (cap, len(trace), len(good_c.victims), len(bad_c.victims)))
    print("     受害序列一致 = %s（★ 应 `False`）" % (good_c.victims == bad_c.victims))
    check("D.5", good_c.victims != bad_c.victims and len(good_c.victims) > 0
          and bad_c.victims != [],
          "★★★ **守卫成立**：把 LRU 改成**逐出最近用过的**之后，受害序列**变了**（正确 %d 次 vs 坏法 %d 次）"
          " —— ★★ `A.3` 对「受害者是最小时间戳」这条规则是敏感的"
          % (len(good_c.victims), len(bad_c.victims)),
          "**守卫失效** —— ★ 改成 MRU 后受害序列不变（那 `A.3` 抓不到它）")

    # ── D.6 ★★ 引文逐字（RadixAttention 三条）───────────────────────────
    check("D.6", specs.SGLANG_RADIX_RETENTION_QUOTE
          == ("Instead of discarding the KV cache after finishing a generation request, our approach "
              "retains the KV cache for both prompts and generation results in a radix tree.")
          and specs.SGLANG_RADIX_DEF_QUOTE
          == ("A radix tree is a data structure that serves as a space-efficient alternative to a "
              "trie (prefix tree).")
          and specs.SGLANG_LRU_QUOTE
          == ("We implement a Least Recently Used (LRU) eviction policy, complemented by a "
              "cache-aware scheduling policy, to enhance the cache hit rate.")
          and specs.SGLANG_PAGE_LAYOUT_QUOTE
          == ("These KV cache tensors are stored on the GPU in a paged layout, where the size of "
              "each page is equivalent to one token."),
          "★★ **四条引文逐字钉住**（SGLang 官方博客）：radix tree 保留 KV · 「省空间的 trie 替代」· "
          "**LRU 淘汰** · `page = 1 token`\n"
          "        ★★ 载体抓取日 %s；★ **机制正文在 `kv-cache/03`**（本主题只写数据结构与口径差）"
          % specs.ENGINES_FETCH_DAY,
          "**SGLang 引文之一被动过**")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.SGLANG_PAGE_TOKENS == 1 and specs.VLLM_BLOCK_SIZE_TOKENS == 16
          and specs.VLLM_BLOCK_SIZE_SOURCE_QUOTE == "DEFAULT_BLOCK_SIZE: ClassVar[int] = 16"
          and specs.SGLANG_PREFIX_REUSE_IS_DEFAULT is True
          and specs.VLLM_ENABLE_PREFIX_CACHING_QUOTE == "Whether to enable prefix caching."
          and specs.PAGE_BLOCK_DISAMBIGUATION
          == ("block / page 是同名不同指：vLLM 的 KV 块 = 16 token，SGLang 的 page = 1 token，"
              "两者差 16 倍，不可互相套用")
          and specs.RADIX_ALPHABET == 8 and specs.RADIX_N_SEQ == 12
          and specs.RADIX_SEQ_LEN_GRID == (2, 3, 4, 5) and specs.RADIX_SEED == 20261009
          and specs.RADIX_CAP_GRID == (2, 4, 8, 16)
          and specs.ENGINES_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == (),
          "★★★ **逐元素钉住**：两家块粒度（`1` / `16`）· 源码引文 · 两家默认（`True` / 开关引文）· "
          "消歧记录前缀 · 算例五个（字母表 / 序列数 / 长度网格 / 种子 / 容量网格）· 抓取日；★ `UNSOURCED == ()`",
          "**某个常量被动过** —— ★ 先查 `SGLANG_*` / `VLLM_BLOCK_*` / `RADIX_*`")

    # ── D.8 ★★★ 独立性（源码级）：被测与三条参照互不引用 ────────────────
    radix_names = _class_names(RadixTree)
    naive_names = _class_names(NaivePrefixStore)
    trie_names = _class_names(UncompressedTrie)
    cache_names = _class_names(RadixCache)
    list_names = _class_names(ListCache)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     `RadixTree` 里出现 `NaivePrefixStore` = %s ；`NaivePrefixStore` 里出现 `RadixTree` = %s"
          % ("NaivePrefixStore" in radix_names, "RadixTree" in naive_names))
    print("     `RadixCache` 里出现 `ListCache` = %s ；`ListCache` 里出现 `RadixCache` = %s"
          % ("ListCache" in cache_names, "RadixCache" in list_names))
    print("     ★★★ **降级措辞**：这只是「**几段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1`/`A.2`/`A.3` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", "NaivePrefixStore" not in radix_names and "RadixTree" not in naive_names
          and "UncompressedTrie" not in radix_names and "ListCache" not in cache_names
          and "RadixCache" not in list_names and "has" in radix_names and "has" in naive_names
          and "n_nodes" in trie_names,
          "★★ **源码级线索成立**：**被测（压缩前缀树 / 树判成员的缓存）与三条参照互不引用**（★ **两侧都审**）——"
          " ★★ 于是「结果相同」**不是恒真**，而是**两条独立路径的会合**",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★★ 术语纪律（★ 形态 A 的根治：计数 == 豁免清单）──────────────
    src = open(__file__, encoding="utf-8").read()
    WORD = "业" + "界"
    n_bare = src.count(WORD)
    n_exempt = sum(src.count(s) * s.count(WORD) for s in BARE_ATTRIBUTION_EXEMPT)
    # ★★★ 第三半：**被禁 ASCII 词**的裸写计数（★ `D3` 的根治）—— 判据同样是**等式**。
    n_bare_terms = _count_bare_ascii(src)
    n_exempt_terms = sum(src.count(s) * _count_bare_ascii(s) for s in TERM_BARE_EXEMPT)
    print()
    print("  ── D.9 ★★★ **术语纪律**：一律写「%s」/「%s」全称；限定语逐字出现 ──"
          % (TERM_KV_CACHE, TERM_SCHEDULER))
    print("     ★★ 源码里**裸写那个被禁裸词** = **%d** 处；豁免清单覆盖 = **%d**（清单 %d 项，逐项理由见模块头）"
          % (n_bare, n_exempt, len(BARE_ATTRIBUTION_EXEMPT)))
    print("     ★★★ 源码里**裸写那两个被禁 ASCII 词**（★ 紧邻中文）= **%d** 处；豁免清单覆盖 = **%d**"
          % (n_bare_terms, n_exempt_terms))
    check("D.9", TERM_KV_CACHE == "KV 缓存" and TERM_SCHEDULER == "请求调度器"
          and LIMIT_QUOTE == "不是任何真实引擎，也不是真机"
          and LIMIT_QUOTE in src and TERM_KV_CACHE in src
          and BARE_ATTRIBUTION_EXEMPT
          == ("不裸写「" + WORD + "」这类无第二个独立主体的断言", "**不裸写「" + WORD + "」**")
          and n_bare == n_exempt
          and all(s in src for s in BARE_ATTRIBUTION_EXEMPT)
          and BARE_ASCII_WORDS == ("ca" + "che", "sched" + "uler")
          and TERM_BARE_EXEMPT == (
              "本域一律写「**KV 缓存**」全称（不裸写 `" + BARE_ASCII_WORDS[0] + "`，那是 L1/L2 与 kernel "
              + BARE_ASCII_WORDS[0] + "）",
              "本域一律写「**请求调度器**」全称（不裸写 `" + BARE_ASCII_WORDS[1] + "`，那是 GPU 的 warp "
              + BARE_ASCII_WORDS[1] + "）")
          and n_bare_terms == n_exempt_terms
          and all(s in src for s in TERM_BARE_EXEMPT),
          "★★ **判据成立**：两条术语常量 `%r` / `%r`；★ 限定语 `%r` **逐字出现**；"
          " ★★ **被禁裸词的裸写计数 `%d` 恰好等于豁免清单覆盖的 `%d`**；"
          " ★★★ **被禁 ASCII 词的裸写计数 `%d` 恰好等于其豁免清单覆盖的 `%d`**"
          % (TERM_KV_CACHE, TERM_SCHEDULER, LIMIT_QUOTE, n_bare, n_exempt, n_bare_terms, n_exempt_terms),
          "**术语或限定语被动过，或源码里出现了【豁免清单之外】的裸写**")


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


# ===========================================================================
# 11. 汇总 / 主程序
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
    print("★★★ **A = %d/%d** —— ★★ **本程序有真 A 类**（`engines` 主题里唯一的一篇）："
          "压缩前缀树 vs 逐 token 朴素比对 / 未压缩 trie / 列表扫描 LRU。" % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：radix tree 与 LRU 是 **`[E]` 级（本库实现）**；结构口径取自 SGLang 博客（`[B]`）。")
    print("★ 本程序只核对「**数据结构的不变量**」——**不是**任何真实引擎的前缀缓存行为；")
    print("  ★★ **机制（`hash → 块` / 命中率）在 `kv-cache/03`**，本程序只写数据结构与两处口径差。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_sglang_and_radix.py —— RadixAttention 的数据结构 + 与 vLLM 的两处口径差")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无 SGLang。")
    print("  ★★★ **本程序有真 A 类**（数据结构可写独立参照）；★ 机制一律回链 `kv-cache/02,03`。")
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
