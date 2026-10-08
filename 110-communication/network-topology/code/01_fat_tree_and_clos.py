#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_fat_tree_and_clos.py —— **胖树 / 三层 Clos 的组合结构**：把拓扑建成一张图，再**数**。

配套文档：`01-fat-tree-and-clos.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本层是【物理层】**（DC 网络的组合结构），**不是**算法层（`collectives`）也**不是**实现层（`nccl`）。

★★★ **本主题的 A 类【可以真做】，而 `nccl` 相反（它没有 A 类）**：
  `01` 的 A 类钉在**【组合结构】**上，并且是**与外部工件对拍**：
    · **被测** = 闭式（`cf_hosts(k) = k³/4` · `cf_cores(k) = (k/2)²` · `cf_paths(k) = (k/2)²`）；
    · **参照** = 把胖树**建成一张显式的图**，再用**图遍历**数出来
      （数节点、数度、BFS 数**最短路**）—— ★★ **两条独立路径**；
    · **外部锚点** = SIGCOMM 2008 的**论文值**（`k = 48`：`27,648` 主机 · `576` 核心 ·
      `1,152` 子网 · `576` 条等价路径）。★★ **与论文对拍是「外部锚点」，不是自证**。

★★★ **一条 §1.3.2b 的自查**（本程序最要紧的设计说明）：
  「把闭式当参照」是本域的系统性诱惑。本程序的建图**是【端口驱动】**的：
    · 主机是**填满边缘交换机的下行端口**长出来的（不是 `k³/4` 个）；
    · 核心交换机是**按需创建的**（贪心分配器：找**编号最小**的、还没连过这个 pod 的核心）——
      ★★ 它的**条数是【涌现】的**（有多少 to-core 端口，就长出多少核心），**不是 `(k/2)²` 个**；
    · 路径数是 **BFS** 数出来的。
  ★★ **残余的诚实交代**：建图的循环上界里**有 `k//2`** —— 那是**拓扑参数**
    （`two layers of k/2 switches`，见 `specs.SIGCOMM08_KARY_QUOTE`），**不是**被测的三个闭式；
    被测的三个闭式（`k³/4` / `(k/2)²` / 路径数）**一个都不出现在建图里**（`D.8` 在源码层核）。

★★ **覆盖边界**：本程序守的是「**拓扑的计数**」与「**错公式会不会被拒**」。
  **不守**任何一条**真实网络的性能**（本机**无网卡**）；★ **bisection 下界 `t ≥ 2S/B`** 与
  **`bisection(ring)`** 的正文在 `20-gpu-architecture/interconnect/02,03`（**本域只回链**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★ **枚举**出的规模（主机 / 边缘 / 聚合 / 核心 / 各层度）== 闭式（`FATTREE_K_GRID` 四档）；
  · `A.2` ★★★ **等价路径数**：BFS 数出的最短路条数 == `(k/2)²`，且 `k = 48` 时 **== 论文值 `576`**
    （★ 同档的**论文实例值**一并核：`27,648` 主机 · `1,152` 子网 · `24` 主机/边缘）；
  · `B.1` 参数化规模表（含 `k = 48` 的**论文实例**）；`B.2` **bisection 的定义式**与结构计数；
    `B.3` 三层 / 两层的 `tier` 表；
  · `C.1` ★★ **把 `k-ary` 的规模公式写错**必须被拒（`k³/8` / `k/2` 核心 …）；
  · `C.2` ★★★ **把等价路径数写成 `k/2`**（漏掉一个 `k/2` 因子）必须被拒；
  · `D.4` ★★★ **守卫实验**：改坏**连接规则** ⇒ **节点计数**不变（无感）· **核心层的度**变（有感）
    · **路径数**变（`A.2` 失败）⇒ ★★ 「节点计数」对「接线」不敏感，**度与路径数**那两条才敏感；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **失败标记自检** + `UNSOURCED`。
"""
import ast
import collections
import inspect
import sys
import textwrap
from collections import deque

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（network-topology）取 `,, FAIL: network-topology`
#   （★ 三个程序**同一个**标记；本域另外三个主题各用 `&&` / `><` / `+=`，**两两不互为前缀**）。
#   ★★ `,` 在 **shell 里需要引号**、在**正则里才是元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ",, FAIL: network-topology"

# 本域另外三个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("&& FAIL: collectives", ">< FAIL: nccl", "+= FAIL: overlap")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
    ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 失败标记必然在**行首**。
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
# 1. 被测：闭式（**只有这四个**；它们**不得**出现在建图里 —— `D.8` 在源码层核）
# ===========================================================================
def cf_hosts(k):
    """闭式：`k³/4` 台主机（`specs.SIGCOMM08_HOSTS_QUOTE`）。"""
    return k ** 3 // 4


def cf_switches_per_layer(k):
    """闭式：每层的交换机数 = `k pods × k/2`（`specs.SIGCOMM08_INSTANCE_QUOTE` 的 `24 switches each`）。"""
    return k * (k // 2)


def cf_cores(k):
    """闭式：核心交换机数 = `(k/2)²`（`specs.SIGCOMM08_CORE_QUOTE`）。"""
    return (k // 2) ** 2


def cf_paths(k):
    """闭式：不同 pod 的两台主机之间的**等代价路径数** = `(k/2)²`（`specs.SIGCOMM08_PATHCOUNT_QUOTE`）。"""
    return (k // 2) ** 2


# ===========================================================================
# 2. 参照：**端口驱动**地把胖树建成一张图（★ 不调用上面的 `cf_*`；`D.8` 核）
#    ★★ 一张图 = `(邻接表, 节点的角色)`；所有计数都从图上**读回来**。
# ===========================================================================
class FatTree(object):
    """胖树的**端口驱动**构造。

    ★★ 构造规则**逐条来自引文**（`specs.py` 的 `# 来源：`）：
      · `k` 个 pod，每 pod **两层、每层 `k/2` 台交换机**（`SIGCOMM08_KARY_QUOTE`）；
      · 每台**边缘**交换机的 `k` 个端口：**`k/2` 下接主机**、**`k/2` 上接本 pod 的聚合层**；
      · 每台**聚合**交换机的 `k` 个端口：**`k/2` 下接本 pod 的边缘层**、**`k/2` 上接核心层**；
      · 每台**核心**交换机**对每个 pod 各占一个端口**（`SIGCOMM08_CORE_QUOTE`）——
        ★★★ 核心交换机的**条数不预设**：它由「聚合层的 to-core 端口有多少」**涌现**出来。
    """

    def __init__(self, k, corrupt=None):
        if k < 2 or k % 2:
            raise ValueError("k 必须是 >= 2 的偶数（每层 k/2 台交换机）：%r" % (k,))
        self.k = k
        self.half = k // 2          # ★ 拓扑参数 k/2（**不是**被测的三个闭式）
        self.corrupt = corrupt
        self.adj = {}               # 节点 -> set(邻居)
        self.role = {}              # 节点 -> "host"/"edge"/"agg"/"core"
        self._build()

    # ── 小工具 ────────────────────────────────────────────────────────────
    def _node(self, role, *key):
        n = (role,) + key
        self.adj.setdefault(n, set())
        self.role[n] = role
        return n

    def _link(self, a, b):
        self.adj[a].add(b)
        self.adj[b].add(a)

    def nodes_of(self, role):
        return [n for n, r in self.role.items() if r == role]

    # ── 建图 ──────────────────────────────────────────────────────────────
    def _build(self):
        half = self.half
        pods = list(range(self.k))

        # ① 每 pod：k/2 台边缘 + k/2 台聚合（**层数由引文给**，不是被测量）
        for p in pods:
            for e in range(half):
                self._node("edge", p, e)
            for a in range(half):
                self._node("agg", p, a)

        # ② 主机：**填满每台边缘交换机的下行端口** —— ★ 主机数**涌现**自端口预算
        #    （边缘交换机的 k 个端口里，k/2 被上行占掉 ⇒ 剩下的填主机）
        for p in pods:
            for e in range(half):
                edge = ("edge", p, e)
                for h in range(half):                    # ★ k/2 个下行端口
                    host = self._node("host", p, e, h)
                    self._link(edge, host)

        # ③ 边缘 ↔ 聚合（同 pod 内**全连**：每台边缘的 k/2 个上行端口接本 pod 的 k/2 台聚合）
        for p in pods:
            for e in range(half):
                for a in range(half):
                    self._link(("edge", p, e), ("agg", p, a))

        # ④ 聚合 ↔ 核心：★★★ **贪心分配器** —— 核心交换机**按需创建**
        #    ★ 规则：为 `(pod, agg, 该 agg 的第 t 个上行端口)` 找**编号最小**的、
        #      **还没连过这个 pod** 的核心；一个都没有就**新建**一台。
        #    ★★ 于是核心台数 = 「为了让每台核心对每个 pod 至多一条链路，最少要几台」
        #       —— **它涌现自「聚合层的 to-core 端口总数」**，**不是** `(k/2)²`。
        self.cores = []            # 每台核心：{pod: agg_index}
        for p in pods:
            for a in range(half):
                for _t in range(half):                   # ★ 每个 agg 有 k/2 个上行端口
                    if self.corrupt == "half_pods" and p >= half:
                        continue                         # ★ D.4 的坏接线：上半 pod 不接核心
                    c = 0
                    while c < len(self.cores) and p in self.cores[c]:
                        c += 1
                    if c == len(self.cores):
                        self.cores.append({})            # ★ 按需新建
                    self.cores[c][p] = a
                    core = self._node("core", c)
                    self._link(core, ("agg", p, a))

    # ── 从图上**读回来**的量（全部是枚举 / 遍历，不是闭式）─────────────────
    def counts(self):
        return {r: len(self.nodes_of(r)) for r in ("host", "edge", "agg", "core")}

    def degrees(self, role):
        return sorted(len(self.adj[n]) for n in self.nodes_of(role))

    def n_edges(self):
        return sum(len(v) for v in self.adj.values()) // 2

    def shortest_path_count(self, src, dst):
        """★ **BFS 数最短路条数**（`cnt[v] += cnt[u]` 的 DP）—— 一条**图遍历**，与闭式无关。"""
        dist, cnt = {src: 0}, {src: 1}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    cnt[v] = cnt[u]
                    q.append(v)
                elif dist[v] == dist[u] + 1:
                    cnt[v] += cnt[u]
        return cnt.get(dst, 0), dist.get(dst, -1)

    def cross_pod_pairs(self, n_pairs):
        """挑 `n_pairs` 对**不同 pod** 的主机对（`(src, dst)`），**按确定的顺序**挑（可复现）。"""
        hosts = {}
        for n in self.nodes_of("host"):
            hosts.setdefault(n[1], []).append(n)
        out = []
        for p in range(min(self.k, n_pairs + 1)):
            for q in range(self.k):
                if p == q:
                    continue
                out.append((hosts[p][0], hosts[q][0]))
                if len(out) >= n_pairs:
                    return out
        return out

    def host_of_pod(self, p, idx=0):
        """取 pod `p` 下的第 `idx` 台主机（用于 `D.4` 的**跨半区**主机对）。"""
        return [n for n in sorted(self.nodes_of("host")) if n[1] == p][idx]

    def bisection_links(self):
        """★★ **bisection 的【定义式】**（本模型里的**结构计数**）：

        · **切法**：把 `k` 个 pod **对半分**（`[0, k/2)` 一侧、`[k/2, k)` 另一侧），
          并把**全部核心交换机**划到**第一侧**（★ 这是一个**显式的**划分规则，不是最优性声明）；
        · **跨切口链路** = 两个端点**分属两侧**的边 —— 逐条边枚举出来的。
        ★★ **通用推导**（`bisection(ring)` / 下界 `t ≥ 2S/B`）在
          `20-gpu-architecture/interconnect/02,03` —— **本域只给定义与这一个切法的计数**。
        """
        side = {}
        for n in self.adj:
            if n[0] == "core":
                side[n] = 0
            elif n[0] == "host":
                side[n] = 0 if n[1] < self.half else 1
            else:                                        # edge / agg 跟着 pod 走
                side[n] = 0 if n[1] < self.half else 1
        cross = 0
        for a, nb in self.adj.items():
            for b in nb:
                if a < b and side[a] != side[b]:
                    cross += 1
        hosts_per_side = collections.Counter(side[n] for n in self.nodes_of("host"))
        return cross, dict(hosts_per_side)


# ===========================================================================
# 3. A 类 —— ★★★ 本主题【有】A 类：**枚举 vs 闭式 vs 论文值**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 胖树的**组合结构**：端口驱动的枚举 vs 闭式 vs SIGCOMM 2008 的论文值")
    print("=" * 78)
    print("  ★★★ 参照 = 把拓扑建成图再**数**（节点数 / 度 / BFS 的最短路条数）；")
    print("      被测 = 闭式（`cf_hosts` / `cf_switches_per_layer` / `cf_cores` / `cf_paths`）。")
    print("  ★★ 两条**独立路径**；★★ 建图是**端口驱动**的（主机填下行端口、核心**按需新建**）。")

    # ── A.1 ★★ 规模：枚举 vs 闭式（四档）──────────────────────────────────
    print()
    print("  ── A.1 ★★ **规模**：枚举 == 闭式（`k³/4` 主机 · `k×(k/2)` 每层 · `(k/2)²` 核心）──")
    print("     %-4s %-8s %-8s %-8s %-8s %-9s %-9s %-9s"
          % ("k", "主机", "闭式", "边缘", "聚合", "核心", "闭式核", "各层度"))
    ok1 = True
    for k in specs.FATTREE_K_GRID:
        ft = FatTree(k)
        c = ft.counts()
        n_layer = cf_switches_per_layer(k)
        deg_e = ft.degrees("edge")
        deg_a = ft.degrees("agg")
        deg_c = ft.degrees("core")
        ok1 = ok1 and (c["host"] == cf_hosts(k)) and (c["edge"] == n_layer) \
            and (c["agg"] == n_layer) and (c["core"] == cf_cores(k)) \
            and set(deg_e) == {k} and set(deg_a) == {k} and set(deg_c) == {k} \
            and c["edge"] == c["agg"] == n_layer
        print("     %-4d %-8d %-8d %-8d %-8d %-8d %-9d %-9d"
              % (k, c["host"], cf_hosts(k), c["edge"], c["agg"], c["core"], cf_cores(k), deg_e[0]))
    print("     ★★ 三层的**度全是 `k`** —— ★ 这是**端口预算**的红利：")
    print("        边缘 `k/2` 主机 + `k/2` 聚合；聚合 `k/2` 边缘 + `k/2` 核心；核心**每 pod 一个端口**。")
    check("A.1", ok1 and len(specs.FATTREE_K_GRID) == 4
          and specs.FATTREE_K_GRID == (4, 6, 8, 16),
          "★★ **判据成立**（`FATTREE_K_GRID` 四档）：枚举出的**主机数 == `k³/4`**、"
          "**每层交换机数 == `k×(k/2)`**、**核心数 == `(k/2)²`**，且三层的**度全 == `k`**",
          "**枚举与闭式对不上** —— ★ 先查建图（端口预算 / 按需新建核心），再看闭式")

    # ── A.2 ★★★ 等价路径数：BFS vs 闭式 vs **论文值** ────────────────────
    print()
    print("  ── A.2 ★★★ **等价路径数**：BFS 数出的最短路条数 vs `(k/2)²` vs **论文值** ──")
    ok2 = True
    for k in specs.FATTREE_K_GRID:
        ft = FatTree(k)
        rows = []
        for src, dst in ft.cross_pod_pairs(3):
            n, hops = ft.shortest_path_count(src, dst)
            rows.append((n, hops))
        n_paths = rows[0][0]
        equal_cost = len({n for n, _h in rows}) == 1    # ★ 不同主机对之间**条数相同**
        ok2 = ok2 and (n_paths == cf_paths(k)) and equal_cost and rows[0][1] == 6
        print("     k=%-3d 路径数 %-4d（闭式 `(k/2)²` = %-4d）· 跳数 %d · 三对主机**条数全同** = %s"
              % (k, n_paths, cf_paths(k), rows[0][1], equal_cost))
    print("     ★★ **跳数恒为 6 条链路**（主机 → 边缘 → 聚合 → 核心 → 聚合 → 边缘 → 主机）；")
    print("        ★★ 但**条数**是 `(k/2)²` —— ★ 那是**核心交换机**台数：**每条等价路径恰好由一个核心决定**。")

    # ── `k = 48`：与**论文实例值**对拍（★ 外部锚点）────────────────────────
    print()
    print("     ── `k = %d`（**论文自设上限**）：与**论文实例值**对拍 ──" % specs.SIGCOMM08_K48)
    k48 = specs.SIGCOMM08_K48
    ft48 = FatTree(k48)
    c48 = ft48.counts()
    src48, dst48 = ft48.cross_pod_pairs(1)[0]
    paths48, hops48 = ft48.shortest_path_count(src48, dst48)
    print("        %-28s 枚举 %-8d 论文 %-8d 一致 = %s"
          % ("主机数", c48["host"], specs.SIGCOMM08_HOSTS_AT_K48,
             c48["host"] == specs.SIGCOMM08_HOSTS_AT_K48))
    print("        %-28s 枚举 %-8d 论文 %-8d 一致 = %s"
          % ("边缘交换机数（= 子网数）", c48["edge"], specs.SIGCOMM08_SUBNETS_AT_K48,
             c48["edge"] == specs.SIGCOMM08_SUBNETS_AT_K48))
    print("        %-28s 枚举 %-8d 论文 %-8d 一致 = %s"
          % ("核心交换机数", c48["core"], specs.SIGCOMM08_CORES_AT_K48,
             c48["core"] == specs.SIGCOMM08_CORES_AT_K48))
    print("        %-28s 枚举 %-8d 论文 %-8d 一致 = %s"
          % ("等价路径数", paths48, specs.SIGCOMM08_PATHS_AT_K48,
             paths48 == specs.SIGCOMM08_PATHS_AT_K48))
    per_edge = c48["host"] // c48["edge"]
    print("        %-28s 枚举 %-8d 论文 %-8d 一致 = %s"
          % ("每台边缘下的主机数", per_edge, specs.SIGCOMM08_HOSTS_PER_EDGE_AT_K48,
             per_edge == specs.SIGCOMM08_HOSTS_PER_EDGE_AT_K48))
    ok2 = ok2 and c48["host"] == specs.SIGCOMM08_HOSTS_AT_K48 \
        and c48["edge"] == specs.SIGCOMM08_SUBNETS_AT_K48 \
        and c48["core"] == specs.SIGCOMM08_CORES_AT_K48 \
        and paths48 == specs.SIGCOMM08_PATHS_AT_K48 \
        and hops48 == 6 and per_edge == specs.SIGCOMM08_HOSTS_PER_EDGE_AT_K48
    print("     ★★★ **五条全部一致** ⇒ 本程序的建图**复现了论文的拓扑性质**"
          "（★ 论文值是**外部锚点**，不是本程序算出来的）。")
    check("A.2", ok2,
          "★★★ **判据成立**：`FATTREE_K_GRID` 四档上 **BFS 数出的等价路径数 == `(k/2)²`**"
          "（且**不同主机对条数相同**、跳数恒为 **6 条链路**）；"
          "★★ **`k = %d` 那一档与论文的实例值逐条一致**：`27,648` 主机 · `1,152` 子网 · "
          "`576` 核心 · **`576` 条等价路径** · `24` 主机/边缘"
          % specs.SIGCOMM08_K48,
          "**路径数或 `k=48` 的实例值与论文不符** —— ★ 先查连接规则（核心是否覆盖全部 pod）")

    # ── D.4 ★★★ 守卫实验（★ 归 D 类：它是**实验守卫**，不测任何实现）──────
    print()
    print("  ── D.4 ★★★ **守卫实验**（`CONTRIBUTING:235`）：把**接线**改坏 ──")
    k_g = specs.FATTREE_K_GRID[2]                     # k = 8
    good, bad = FatTree(k_g), FatTree(k_g, corrupt="half_pods")
    cg, cb = good.counts(), bad.counts()
    src_g, dst_g = good.host_of_pod(0), good.host_of_pod(k_g - 1)   # ★ 跨半区的一对
    src_b, dst_b = bad.host_of_pod(0), bad.host_of_pod(k_g - 1)
    n_ok = good.shortest_path_count(src_g, dst_g)[0]
    n_bad = bad.shortest_path_count(src_b, dst_b)[0]
    scale_same = all(cg[r] == cb[r] for r in ("host", "edge", "agg", "core"))
    deg_same = good.degrees("core") == bad.degrees("core")
    print("     `corrupt=\"half_pods\"`：**上半区 pod 的聚合交换机上行的核心链路被剪断**")
    print("     k=%d：主机 %d vs %d · 边缘 %d vs %d · 聚合 %d vs %d · 核心 %d vs %d ⇒"
          " **四类节点数全同 = %s**"
          % (k_g, cg["host"], cb["host"], cg["edge"], cb["edge"],
             cg["agg"], cb["agg"], cg["core"], cb["core"], scale_same))
    print("     **核心层的度**：正确 %s vs 坏接线 %s ⇒ 同 = %s"
          % (sorted(set(good.degrees("core"))), sorted(set(bad.degrees("core"))), deg_same))
    print("     跨半区的**等价路径数**：正确 %d vs 坏接线 %d ⇒ **变了 = %s**"
          % (n_ok, n_bad, n_ok != n_bad))
    print("     ★★★ **读法（如实写清两半）**：")
    print("        · **节点计数**这一半**完全无感**（四类节点数一字不变）；")
    print("        · **度**这一半**有感**（核心度从 `k` 掉到 `k/2`）—— ★★ 也就是说 `A.1` 里")
    print("          那条「三层的度全是 `k`」**也会**发现这个坏接线；")
    print("        · **路径数**这一半有感（`A.2` 失败）。")
    print("     ★★ 结论不是「A.1 无感」而是：**同一段判据内部有敏感与不敏感的两半**。")
    check("D.4", scale_same and (n_ok != n_bad) and n_ok == cf_paths(k_g) and not deg_same,
          "★★★ **守卫成立**：改坏接线后 ① **四类节点的【计数】一字不变**；"
          "② **核心层的【度】变了**（★ 所以 `A.1` 的「度」那一小条**也**会发现问题）；"
          "③ **等价路径数真的变了** ⇒ ★★ 「节点计数」对「接线」不敏感，**路径数**才是敏感的"
          "（★ 与 `collectives` 的「搬运量对块索引不敏感」**同型**）"
          "\n        ★★ **它归 D 类而不是 A 类**：它是**实验守卫**，不测任何实现。",
          "**守卫失效** —— ★ 改坏接线后路径数居然不变（那 `A.2` 对连接规则不敏感 ⇒ 需要加判据）")


# ===========================================================================
# 4. B 类：参数化规模表 · **bisection 的定义式** · 三层/两层的 tier
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 规模表 · **bisection 的【定义式】** · `tier` 的两层 / 三层")
    print("=" * 78)
    print("  ★★ 载体 = SIGCOMM 2008（一手 PDF，抓取日 **%s**）。" % specs.FETCH_DATE)

    # ── B.1 ★★ 参数化规模表（含 k=48 的论文实例）──────────────────────────
    print()
    print("  ── B.1 ★★ **参数化规模表**（`k` 越大越接近论文实例）──")
    print("     %-4s %-10s %-10s %-10s %-10s %-10s" % ("k", "主机", "边缘", "聚合", "核心", "交换机总数"))
    rows = []
    for k in (4, 6, 8, 16, specs.SIGCOMM08_K48):
        ft = FatTree(k)
        c = ft.counts()
        total_sw = c["edge"] + c["agg"] + c["core"]
        rows.append((k, c["host"], c["edge"], c["agg"], c["core"], total_sw))
        print("     %-4d %-10d %-10d %-10d %-10d %-10d" % (k, c["host"], c["edge"], c["agg"],
                                                            c["core"], total_sw))
    k48_row = rows[-1]
    print("     ★★ `k = %d` 那一行正是论文的实例：主机 `%d` · 子网/边缘 `%d` · 核心 `%d`"
          % (specs.SIGCOMM08_K48, k48_row[1], k48_row[2], k48_row[4]))
    print("     ★★ 论文 §2.2 逐字（两个数在这一句里）：`%s`" % specs.SIGCOMM08_SUBNETS_QUOTE)
    print("     ★★ 而层内交换机数是 `k/2`（论文逐字 `%s`）" % specs.SIGCOMM08_INSTANCE_QUOTE)
    check("B.1", rows[0][1] == 16 and rows[0][4] == 4
          and rows[1][1] == 54 and rows[2][1] == 128
          and k48_row[1] == specs.SIGCOMM08_HOSTS_AT_K48
          and k48_row[2] == specs.SIGCOMM08_SUBNETS_AT_K48
          and k48_row[4] == specs.SIGCOMM08_CORES_AT_K48
          and k48_row[5] == specs.SIGCOMM08_SUBNETS_AT_K48 * 2 + specs.SIGCOMM08_CORES_AT_K48
          and len(rows) == 5
          and specs.SIGCOMM08_SUBNETS_QUOTE == ("The network supports 27,648 hosts, made up of 1,152 subnets with 24 hosts each.")
          and specs.SIGCOMM08_INSTANCE_QUOTE == ("a fat-tree built from 48-port GigE switches would consist of 48 pods, each containing an edge layer and an aggregation layer with 24 switches each."),
          "★★ **判据成立**：`k = 4` ⇒ **16** 主机 / **4** 核心（论文 Figure 3 的**最简实例**）；"
          "`k = 6` ⇒ **54**；`k = 8` ⇒ **128**；★★ `k = 48` ⇒ `27,648` 主机 / `1,152` 子网 / "
          "`576` 核心，且**交换机总数 = `2×1152 + 576 = 2880`**",
          "**规模表被动过** —— ★ 先查 `FatTree.counts()` 与闭式")

    # ── B.2 ★★★ bisection 的**定义式** + 本切法的结构计数 ────────────────
    print()
    print("  ── B.2 ★★★ **bisection 的【定义式】**（★ 只给定义与**一个显式切法**的计数）──")
    print("     定义：把网络切成两半时的**跨切口链路总容量**。")
    print("     ★★ **本程序只做一件事**：按**显式规则**划一刀，把**跨切口的边**逐条枚举出来。")
    print("        · 切法：pod `[0, k/2)` 一侧、`[k/2, k)` 另一侧；**核心全部划到第一侧**。")
    ok_b2 = True
    for k in (4, 8, 16):
        ft = FatTree(k)
        cross, hs = ft.bisection_links()
        hosts_per_side = k ** 3 // 8
        ok_b2 = ok_b2 and cross == hosts_per_side and set(hs.values()) == {hosts_per_side}
        print("     k=%-3d 跨切口链路 %-6d（`k³/8` = %-6d）· 两侧主机 %s（各 `k³/8`）"
              % (k, cross, hosts_per_side, dict(sorted(hs.items()))))
    print("     ★★ 两侧主机**相等**，且跨切口链路数 **== 一侧主机数** ⇒ 在这个切法下")
    print("        **每台主机对应一条跨切口链路**（等价于论文说的 `1:1` oversubscription）。")
    print("     ★★★ **但论文自己给了一句限定语**（逐字，必须一起引）：`%s`"
          % specs.SIGCOMM08_OVERSUBSCRIPTION_QUOTE)
    print("     ★★ **通用推导**（`bisection(ring) = L×b`、下界 `t ≥ 2S/B`）在 "
          "`20-gpu-architecture/interconnect/02,03` —— **本域不重推**。")
    check("B.2", ok_b2,
          "★★ **判据成立**：本模型里，**显式切法**下的跨切口链路数 == `k³/8` == **一侧的主机数**"
          "（`k = 4/8/16` 三档），两侧主机数**相等**；"
          "★★ 且**限定语一并登记**（论文逐字：`1:1` **实践上可能很难**）",
          "**bisection 的结构计数被动过** —— ★ 先查 `bisection_links()` 的显式切法")

    # ── B.3 ★★ 三层 / 两层（tier）────────────────────────────────────────
    print()
    print("  ── B.3 ★★ `tier` 的两层 / 三层 ──")
    print("     论文逐字：`%s`" % specs.SIGCOMM08_TIER_QUOTE)
    print("     ★★ **本库一律写「三层 Clos」全称**，保留英文 `tier`；")
    print("        ★★★ **不得裸写 `stage`** —— 库内 `stage` / 阶段已有**四个**含义"
          "（TVM 计算阶段 / cutlass 流水级数 / 流水线并行的 stage / ZeRO 阶段，见域 README 的「同名不同指」表）。")
    print("     ★★ 本文的「层」是**拓扑层**：核心层 / 聚合层 / 边缘层（★ 与 `collectives` 的"
          "「通信阶段」**不是一回事**）。")
    check("B.3", "A three-tiered design" in specs.SIGCOMM08_TIER_QUOTE
          and "5K to 8K hosts" in specs.SIGCOMM08_TIER_QUOTE
          and all(w in specs.SIGCOMM08_TIER_QUOTE
                  for w in ("core tier", "aggregation tier", "edge tier", "edge tiers"))
          and specs.SIGCOMM08_TIER_QUOTE.count("two-tiered") == 2,
          "★★ **判据成立**：`tier` 的一级表述**逐字钉住**（三层 = core / aggregation / edge；"
          "两层 = core + edge；★ 两层规模区间 `5K to 8K hosts`）⇒ ★★ 本库**不裸写 `stage`**",
          "**tier 的引文被动过**")


# ===========================================================================
# 5. C 类：两条【真的会失败】的错法
# ===========================================================================
def scale_claim_ok(k, hosts_claim, cores_claim):
    """★★★ **规模声明的校验器**：把「某个 `k` 下的 (主机数, 核心数)」与**枚举值**比对。

    ★★ 它**不**是 A 类（A 类在 `part_A`）：它是**一段判定逻辑**，用来判「错公式会不会被拒」。
    ★★★ **它必须对【真值】放行、对【错公式】拒绝** —— `C.1` 同时验这两侧。
    """
    c = FatTree(k).counts()
    return (hosts_claim, cores_claim) == (c["host"], c["core"])


def path_claim_ok(k, n_paths_claim):
    """★ **等价路径数**的校验器（同上：真值放行、错法拒绝）。"""
    ft = FatTree(k)
    n, _hops = ft.shortest_path_count(*ft.cross_pod_pairs(1)[0])
    return n_paths_claim == n


def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**规模公式写错** · **等价路径数漏一个因子**")
    print("=" * 78)

    # ── C.1 ★★ 把 k-ary 的规模公式写错 ─────────────────────────────────
    print()
    print("  ── C.1 ★★ **把 `k-ary` 的规模公式写错**（`k = 8`）──")
    k = 8
    truth = FatTree(k).counts()
    cases = (
        ("真值（`k³/4` 主机 · `(k/2)²` 核心）", truth["host"], truth["core"], True),
        ("把主机数写成 `k³/8`（漏一半）", k ** 3 // 8, truth["core"], False),
        ("把主机数写成 `k²/4`（少一个 `k`）", k ** 2 // 4, truth["core"], False),
        ("把核心数写成 `k/2`（漏乘 `k/2`）", truth["host"], k // 2, False),
        ("把核心数写成 `k²`（多乘 4）", truth["host"], k ** 2, False),
    )
    print("     k=%d 的枚举真值：主机 %d · 核心 %d" % (k, truth["host"], truth["core"]))
    all_ok = True
    for label, h, c, expect in cases:
        got = scale_claim_ok(k, h, c)
        all_ok = all_ok and (got == expect)
        print("     %-34s (主机 %-6d, 核心 %-5d) ⇒ 校验器 %s（期望 %s）"
              % (label, h, c, "放行" if got else "**拒绝**", "放行" if expect else "**拒绝**"))
    print("     ★★★ 关键：**真值被放行、四种错公式全部被拒** —— 校验器**两侧都验了**。")
    check("C.1", all_ok and scale_claim_ok(k, truth["host"], truth["core"])
          and not scale_claim_ok(k, k ** 3 // 8, truth["core"])
          and not scale_claim_ok(k, truth["host"], k // 2),
          "★★ **判据成立**：`k = 8` 下 **真值 `(128, 16)` 放行**，而 **`k³/8` / `k²/4`（主机）**"
          "与 **`k/2` / `k²`（核心）** 四种错法**全部被拒** ⇒ ★★ **写错规模公式真的会失败**",
          "**错公式没被拒绝**（或真值被误拒）—— ★ 先查 `scale_claim_ok`（它比对的是**枚举值**）")

    # ── C.2 ★★★ 把等价路径数写成 `k/2`（漏一个因子）────────────────────
    print()
    print("  ── C.2 ★★★ **把等价路径数写成 `k/2`**（漏掉一个 `k/2` 因子）──")
    all_ok2 = True
    for k in (4, 8, 16):
        ft = FatTree(k)
        truth_paths = ft.shortest_path_count(*ft.cross_pod_pairs(1)[0])[0]
        wrong = k // 2                                # ★ 少一个 k/2 因子
        all_ok2 = all_ok2 and path_claim_ok(k, truth_paths) and not path_claim_ok(k, wrong)
        print("     k=%-3d 真值 %-4d ⇒ 放行；写成 `k/2` = %-3d ⇒ **拒绝**" % (k, truth_paths, wrong))
    print("     ★★★ **为什么这个错法可信**：`k/2` 与 `(k/2)²` 只差一个因子，")
    print("        而 `k/2` 在**同一个拓扑**里到处都是（每层交换机数、每交换机的主机数、端口的一半）")
    print("        ⇒ ★★ 这正是「**同一个数在不同指上的传染**」（本库 `no-fabricated-numbers` 的孪生形态）。")
    check("C.2", all_ok2,
          "★★★ **判据成立**：`k = 4/8/16` 三档上 **`(k/2)²` 被放行**，"
          "**`k/2` 被拒** ⇒ ★★ 「把等价路径数写成 `k/2`」**真的会失败**",
          "**漏因子的写法没被拒绝** —— ★ 先查 `path_claim_ok`（它比对的是 **BFS 的数**）")


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
    print("  ★★★ 限定语：**不是 NCCL，也不是真机** —— 本机**无 GPU / 无 NCCL / 无网卡**；")
    print("      ★★ 但**本程序的 A 类是真的**：它算的是**组合结构**（论文值当外部锚点）。")
    print("  ★★ 本层 = **物理层**（DC 网络的组合结构）。")
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
    check("D.2", ASSERT_FAIL_MARK == ",, FAIL: network-topology" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `,, FAIL: network-topology`、且含 `FAIL`\n"
          "        ★★ `,` 在 **shell 里需要引号** ⇒ 它**只允许出现在 `print()` 的字符串里**。")
    check("D.2b", OTHER_THEME_MARKS == ("&& FAIL: collectives", ">< FAIL: nccl", "+= FAIL: overlap"),
          "★ 「不跨主题照抄」的**对照物**（本域另外三个主题的标记）逐位钉住；"
          "★ 四个标记**两两不互为前缀**")

    # ── D.3 ★ 载体元数据 ──────────────────────────────────────────────────
    check("D.3", specs.SIGCOMM08_CARRIER
          == ("Al-Fares, Loukissas, Vahdat, A Scalable, Commodity Data Center Network "
              "Architecture, SIGCOMM 2008")
          and specs.SIGCOMM08_DOI == "10.1145/1402958.1402967"
          and specs.SIGCOMM08_PDF_URL == "https://cseweb.ucsd.edu/~vahdat/papers/sigcomm08.pdf"
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：作者 + 标题 + 会议 · DOI `%s` · 一手 PDF URL ·"
          "抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住（★ 本程序**没有**「用了但没出处」的常量）"
          % (specs.SIGCOMM08_DOI, specs.FETCH_DATE))

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    #   ① `FATTREE_K_GRID` 全为偶数且 >= 2（否则建图会 `ValueError` ⇒ **崩溃捕获**而不是断言）；
    #   ② `k = 48` 档的四个论文值**内部自洽**（`1152 = 48 × 24`、`27648 = 1152 × 24`）。
    grid_ok = all(k >= 2 and k % 2 == 0 for k in specs.FATTREE_K_GRID)
    consist = (specs.SIGCOMM08_SUBNETS_AT_K48
               == specs.SIGCOMM08_K48 * specs.SIGCOMM08_SWITCHES_PER_LAYER_PER_POD_AT_K48) \
        and (specs.SIGCOMM08_HOSTS_AT_K48
             == specs.SIGCOMM08_SUBNETS_AT_K48 * specs.SIGCOMM08_HOSTS_PER_EDGE_AT_K48) \
        and specs.SIGCOMM08_HOSTS_PER_EDGE_AT_K48 == specs.SIGCOMM08_SWITCHES_PER_LAYER_PER_POD_AT_K48
    print("     ★★ `FATTREE_K_GRID = %s`：全为偶数 = %s" % (specs.FATTREE_K_GRID, grid_ok))
    print("     ★★ 论文实例值的内部自洽：`1152 = 48×24` 且 `27648 = 1152×24` = %s" % consist)
    check("D.3b", grid_ok and consist and specs.SIGCOMM08_K48 == 48
          and specs.SIGCOMM08_SUBNETS_AT_K48 == 1152 and specs.SIGCOMM08_HOSTS_AT_K48 == 27648,
          "★★★ **口径守卫**：`k` 网格**全为偶数（`k/2` 才是整数台交换机）**，"
          "且 `k = 48` 的四个论文值**逐个自洽**（★ 否则建图会抛 `ValueError` ⇒ **崩溃捕获**）",
          "**`k` 网格或 `k=48` 的论文值被动过** —— ★ 先查 `FATTREE_K_GRID` 与三个 `*_AT_K48`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字 · 书目载体 · 逐元素 · 独立性 · UNSOURCED
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · 书目载体 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★ SIGCOMM 的七条引文（逐字）────────────────────────────────
    check("D.5", specs.SIGCOMM08_KARY_QUOTE
          == ("We organize a k-ary fat-tree as shown in Figure 3. "
              "There are k pods, each containing two layers of k/2 switches.")
          and specs.SIGCOMM08_CORE_QUOTE
          == ("There are (k/2)2 k-port core switches. "
              "Each core switch has one port connected to each of k pods.")
          and specs.SIGCOMM08_HOSTS_QUOTE
          == ("In general, a fat-tree built with k-port switches supports k3 /4 hosts. "
              "In this paper, we focus on designs up to k = 48.")
          and specs.SIGCOMM08_PATHCOUNT_QUOTE
          == ("there are (k/2)2 such shortest-paths between any two hosts on different pods, "
              "but only one is chosen")
          and specs.SIGCOMM08_576_QUOTE
          == ("There are 576 equal-cost paths between any given pair of hosts in different pods.")
          and specs.SIGCOMM08_NONBLOCKING_QUOTE
          == ("fat-trees are rearrangeably non-blocking, meaning that for arbitrary communication "
              "patterns, there is some set of paths that will saturate all the bandwidth available "
              "to the end hosts in the topology."),
          "★★ **六条引文逐字钉住** —— ★★★ **特别注意上标的【提取形态】**："
          "`(k/2)2`（上标 `2` 压平、**无空格**）与 `k3 /4`（上标 `3` 压平、**`/4` 前有空格**）；"
          "\n        ★★ 且 `NONBLOCKING` 那句里的限定词是 **`rearrangeably`** 与 **`some set of paths`**"
          "（★ 漏掉就把「可重排」升格成「无阻塞」）；\n"
          "        ★★ 载体 = 一手 PDF，抓取日 %s" % specs.FETCH_DATE,
          "**SIGCOMM 的引文常量被动过** —— ★ 先查上标的提取形态（`(k/2)2` / `k3 /4`）")

    # ── D.6 ★★ 书目载体 + 限定语引文 ──────────────────────────────────────
    check("D.6", specs.SIGCOMM08_CLOS_CARRIER
          == ("C. Clos. A Study of Non-blocking Switching Networks. "
              "Bell System Technical Journal, 32(2), 1953.")
          and specs.SIGCOMM08_LEISERSON_CARRIER
          == ("C. E. Leiserson. Fat-Trees: Universal Networks for Hardware-Efficient "
              "Supercomputing. IEEE Transactions on Computers, 34(10):892–901, 1985.")
          and specs.SIGCOMM08_OVERSUBSCRIPTION_QUOTE
          == ("Achieving an oversubscription ratio of 1:1 in practice may be difficult because of "
              "the need to prevent packet reordering for TCP flows.")
          and specs.SIGCOMM08_TIER_QUOTE == ("A three-tiered design (see Figure 1) has a core tier in the root of the tree, an aggregation tier in the middle and an edge tier at the leaves of the tree. A two-tiered design has only the core and the edge tiers. Typically, a two-tiered design can support between 5K to 8K hosts."),
          "★★ **两条书目载体逐字**（Clos 1953 与 Leiserson 1985 —— ★★ **它们是从同一份一手 PDF 的"
          "参考文献里读到的**，所以「只引载体」这一层**本轮取到了内容**）；"
          "★★ Leiserson 那条的页码短横是 **en-dash `–`**（U+2013），**不是 ASCII 连字符**；\n"
          "        ★★ **限定语引文**：`1:1` 的「**实践上可能很难**」与 `tier` 的 `5K to 8K` 一并钉住",
          "**书目载体或限定语被动过** —— ★ 先查 en-dash 与 `5K to 8K`")

    # ── D.7 ★★★ 逐元素钉住 + 三条「不引什么」+ `UNSOURCED` ────────────────
    check("D.7", specs.FATTREE_K_GRID == (4, 6, 8, 16)
          and specs.SIGCOMM08_K48 == 48
          and specs.SIGCOMM08_HOSTS_AT_K48 == 27648
          and specs.SIGCOMM08_SUBNETS_AT_K48 == 1152
          and specs.SIGCOMM08_HOSTS_PER_EDGE_AT_K48 == 24
          and specs.SIGCOMM08_SWITCHES_PER_LAYER_PER_POD_AT_K48 == 24
          and specs.SIGCOMM08_CORES_AT_K48 == 576
          and specs.SIGCOMM08_PATHS_AT_K48 == 576
          and specs.UNSOURCED == (),
          "★★★ **一个元组逐元素（四档）+ 七个标量**钉住（`k=48` 实例的六个值 + 一个 `k`）；"
          "★★ **特别提示**：`CORES_AT_K48` 与 `PATHS_AT_K48` **同值 `576` 但不同指**"
          "（一个是设备数、一个是路径数）⇒ ★ 必须**两个名字分开钉**，别合并成一个常量；\n"
          "        ★★ `UNSOURCED == ()` 一并钉住（★ 0 个叶子 ⇒ 不产生变异体）",
          "**元组的元素或某个标量被动过** —— ★ 注意 `576` 那两个**同值不同指**的常量")

    # ── D.8 ★★★ 独立性（源码级）── ★★ 遍历【全部】参照实现 ───────────────
    refs = (FatTree._build, FatTree.counts, FatTree.degrees,
            FatTree.shortest_path_count, FatTree.cross_pod_pairs, FatTree.bisection_links)
    cfs = (cf_hosts, cf_switches_per_layer, cf_cores, cf_paths)
    ref_srcs = {textwrap.dedent(inspect.getsource(f)) for f in refs}
    cf_srcs = {textwrap.dedent(inspect.getsource(f)) for f in cfs}
    call_tables = {}
    for f in refs:
        t = textwrap.dedent(inspect.getsource(f))
        call_tables[f.__name__] = {n.func.id for n in ast.walk(ast.parse(t))
                                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    cf_calls = {n.func.id for s in cf_srcs for n in ast.walk(ast.parse(s))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：**参照**侧六个函数的调用表 ──")
    for nm, tbl in call_tables.items():
        print("     `FatTree.%s()` 里调用的函数：%s"
              % (nm, ", ".join(sorted(tbl)) if tbl else "**空集**"))
    print("     **被测**侧四个 `cf_*` 里调用的函数：%s"
          % (", ".join(sorted(cf_calls)) if cf_calls else "**空集**"))
    hits = sorted({c for tbl in call_tables.values() for c in tbl
                   if c.startswith("cf_") or c == "FatTree"})
    print("     ⇒ ★★ **参照侧不调用任何 `cf_*`**（★★ 也不在 `cf_*` 的调用表里回环）"
          " ⇒ 「参照不得用被测」**在源码层面成立**。")
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，")
    print("        **不证明**判据有物理信息量 —— 后者由 `A.2` 与 `D.4` 的**行为**兜底。")
    check("D.8", not hits
          and len(ref_srcs) == 6 and len(cf_srcs) == 4
          and not any(c.startswith("cf_") for s in ref_srcs
                      for c in _call_names(s))
          and not any(c in call_tables for s in cf_srcs for c in _call_names(s)),
          "★★ **源码级线索成立**：**六个**参照函数两两不同源、**四个** `cf_*` 两两不同源；"
          "★★ **参照侧的调用表里一个 `cf_*` 都没有，`cf_*` 也不调用参照**（★ **两侧都审**）",
          "**参照与闭式互相调用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 闭式的边界 + 与 `k` 的整除 ───────────────────────────────
    check("D.9", cf_hosts(4) == 16 and cf_cores(4) == 4 and cf_paths(4) == 4
          and cf_hosts(6) == 54 and cf_cores(6) == 9
          and all(cf_hosts(k) * 4 % (k ** 3) == 0 for k in specs.FATTREE_K_GRID),
          "★★ **闭式的边界**：`k = 4` ⇒ `16 / 4 / 4`（论文 Figure 3 的最简实例）；"
          "`k = 6` ⇒ `54 / 9`（★ **非 2 的幂也整除**，因为 `k` 是偶数）；"
          "★ 且 `k³/4` 在四档上都是整数",
          "**闭式的边界值被动过**")


def _call_names(src):
    return {n.func.id for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


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
    print("★★ **A = %d/%d** —— ★★★ **本主题【有】A 类**（与 `nccl` 相反）："
          "它把胖树**建成图再数**，并与 **SIGCOMM 2008 的论文值**对拍（外部锚点）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 依据在 `specs.py` 的 `# 来源：`（SIGCOMM 2008 一手 PDF，抓取日 %s）。" % specs.FETCH_DATE)
    print("★ 本程序只核对「组合结构的计数」与「错公式真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_fat_tree_and_clos.py —— 胖树 / 三层 Clos 的组合结构")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **物理层**；★★ **本主题的 A 类锚点是 SIGCOMM 2008 的论文值**。")
    print("  ★★★ 限定语：**不是 NCCL，也不是真机** —— 本机无 GPU / 无 NCCL / 无网卡。")
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
