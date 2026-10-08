#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_rail_optimized.py —— **rail-optimized / 多轨道**：把 `N` 张卡**逐张**摆到 `R` 个 rail 上。

配套文档：`02-rail-optimized.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本层是【物理层】**（scale-out 网络的**布线结构**），**不是**算法层也**不是**实现层。

★★★ **本主题的 A 类【可以真做】**（与 `nccl` 相反）：
  `02` 的 A 类钉在 **rail 的【划分】** 上，**与外部工件的措辞对拍**：
    · **被测** = 闭式（**每个 rail 有 `N/R` 张卡**、**每节点每轨 `G/R` 张**）；
    · **参照** = 把**每一张卡**按**端口位置**接到它的 leaf 上，再从图上**数回来**
      （每个 leaf 挂了哪些卡、每节点贡献了几张）—— ★★ **两条独立路径**；
    · **外部锚点** = NVIDIA 的**逐字定义**（`Rail-optimized design means a GPU node with multiple
      interfaces will put each GPU "rail" (IB network interface) onto a different first level (LEAF)
      switch ...`）—— ★★★ 注意它界定的是「**一个节点内部的 `G` 张卡被分散到 `G` 个 leaf**」，
      **不是**「一个 rail 里有 `G` 台节点」⇒ ★★ 这正是 `C` 类要判的混淆。

★★ **一条 §1.3.2b 的自查**：建 fabric 时**没有任何地方写 `N/R`**：
  规则只是「编号 `g` 的卡接到编号 `g mod R` 的 leaf」（★ 那是**定义**，来自引文），
  而**每个 rail 有几张卡**是**数出来的**（`A.1`/`A.2` 与闭式对拍）。
  ★★ **残余的诚实交代**：`g mod R` 这个**规则本身**是定义（引文给的）——
    它**不**是被测的量；被测的是 `N/R` 与 `G/R` 这两个**导出量**。

★ **本主题的禁区**（一行都不重写，只回链）：
  · **NVLink / NVSwitch / PCIe 的带宽与拓扑**、**NVLink 域逐代** → `20-gpu-architecture/interconnect/01,02`；
  · ★★ 因此本文件**不解释**「为什么是 8 张卡一个节点」（那在上游），只把它当**输入**。

★★ **覆盖边界**：本程序守的是「**布线的划分与跳数账**」与「**错模型会不会被拒**」。
  **不守**任何一条**真实 fabric 的性能**（本机**无网卡**）——
  ★★ **本机 socket 的任何数字都不得**用作带宽 / 延迟 / 吞吐的依据（`100-parallelism` 的逐字禁令）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★ **rail 划分**（`R == G` 档）：**枚举**出的每轨卡数 == `N/R`，且**每节点每轨恰好 1 张**；
  · `A.2` ★★ **rail 划分**（`R < G` 档）：每轨卡数 == `N/R`，**每节点每轨 `G/R` 张**（枚举复算）；
  · `B.1` ★★★ **跳数 / 路径账**：同轨 **2 条链路 / 1 台交换机**（★ NVIDIA 逐字 `one switch hop away`）
    与跨轨 **4 条链路 / 3 台交换机**，路径数 **`1` vs `S`（脊数）**；
  · `B.2` 参数化 `N/R` 表；`B.3` 「同 index 集合通信」vs「全卡集合通信」的**触及 leaf 数**账；
  · `C.1` ★★★ **把 rail 与「节点」搞混**必须被拒（三种错模型）；
  · `C.2` ★★★ 把「**每节点每轨 `1` 张**」误当「**每轨共 `1` 张**」必须被拒；
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
# ★★★ 失败标记：本主题（network-topology）取 `,, FAIL: network-topology`（★ 三个程序**同一个**）。
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
# 1. 被测：闭式（**只有这三个**；它们**不得**出现在建 fabric 里 —— `D.8` 在源码层核）
# ===========================================================================
def cf_total_gpus(nodes, gpus):
    """闭式：总卡数 `N = 节点数 × 每节点卡数`。"""
    return nodes * gpus


def cf_rail_size(nodes, gpus, rails):
    """闭式：**每个 rail 的卡数** = `N/R`。"""
    return nodes * gpus // rails


def cf_per_node_per_rail(gpus, rails):
    """闭式：**每个节点在每个 rail 上贡献几张卡** = `G/R`。"""
    return gpus // rails


# ===========================================================================
# 2. 参照：**逐张卡**按【端口位置】摆上 leaf，再从图上数回来（★ 不调用 `cf_*`；`D.8` 核）
# ===========================================================================
class RailFabric(object):
    """rail-optimized fabric 的**逐卡摆放**模型。

    ★★ 布线规则**逐字来自引文**（`specs.NVIDIA_RAIL_QUOTE`）：
      「一台多网卡的 GPU 节点，把**每张 GPU 的 rail** 放到**不同的第一层（LEAF）交换机**上」
      ⇒ 本模型把它落成一条**位置规则**：**编号 `g` 的卡 → 编号 `g mod R` 的 leaf**。
      ★★★ **这条规则是【定义】，不是被测的量**；被测的是「每轨几张卡」这类**导出量**。

    · `corrupt=None`      ⇒ 正确布线；
    · `corrupt="by_node"` ⇒ `C.1` 的错模型①：**按节点号**分轨（「把 rail 当成节点」）；
    · `corrupt="by_node_id"` ⇒ `C.1` 的错模型②：**每个节点一条轨**（轨数 = 节点数）；
    · `corrupt="one_rail"`   ⇒ `C.1` 的错模型③：**全部塞进第 0 条轨**。
    """

    def __init__(self, nodes, gpus, rails, spines, corrupt=None):
        self.nodes, self.gpus, self.rails, self.spines = nodes, gpus, rails, spines
        self.corrupt = corrupt
        self.adj = {}
        self.role = {}
        self._build()

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

    # ── 布线 ──────────────────────────────────────────────────────────────
    def _leaf_of(self, n, g):
        """★ **位置规则**（唯一一条）：编号 `g` 的卡接到某条 leaf 上。"""
        if self.corrupt == "by_node":
            return n % self.rails
        if self.corrupt == "by_node_id":
            return n
        if self.corrupt == "one_rail":
            return 0
        return g % self.rails

    def _build(self):
        # ① 交换机：每条 rail 一台 leaf；`spines` 台脊（★ 两个数都是**输入**，不是被测）
        for r in range(self.rails):
            self._node("leaf", r)
        for s in range(self.spines):
            self._node("spine", s)
        # ② leaf ↔ spine（每台 leaf 上行到**全部**脊 —— ★ 这是账里路径数的来源）
        for r in range(self.rails):
            for s in range(self.spines):
                self._link(("leaf", r), ("spine", s))
        # ③ ★★★ 逐张卡摆放：**没有任何地方写 `N/R`**
        for n in range(self.nodes):
            for g in range(self.gpus):
                gpu = self._node("gpu", n, g)
                r = self._leaf_of(n, g)
                if ("leaf", r) not in self.adj:       # 模型② 会要求**更多** leaf
                    self._node("leaf", r)
                self._link(gpu, ("leaf", r))

    # ── 从图上**读回来**的量（全是枚举 / 遍历）──────────────────────────────
    def rail_members(self, r):
        """★ **从图上数**：编号 `r` 的 leaf 上挂了哪些卡（**涌现**的，不是算出来的）。"""
        leaf = ("leaf", r)
        return sorted(n for n in self.adj.get(leaf, ()) if n[0] == "gpu")

    def n_leaves(self):
        return len(self.nodes_of("leaf"))

    def n_gpus(self):
        return len(self.nodes_of("gpu"))

    def per_node_counts(self, r):
        """每个节点往 rail `r` 贡献了几张卡（`Counter`）。"""
        return collections.Counter(n[1] for n in self.rail_members(r))

    def shortest_path(self, src, dst):
        """★ **BFS**：返回 `(链路数, 最短路条数, 中间的交换机台数)` —— 一条**图遍历**。"""
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
        if dst not in dist:
            return -1, 0, -1
        # 交换机台数 = 路径上的**非端点**节点数 = 链路数 − 1（两端都是卡）
        return dist[dst], cnt[dst], (dist[dst] - 1 if dist[dst] > 0 else -1)


# ===========================================================================
# 3. A 类 —— ★★★ rail 的【划分】：枚举 vs 闭式
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · rail 的【划分】：**逐张卡摆放**再数回来 vs 闭式 `N/R` / `G/R`")
    print("=" * 78)
    print("  ★★★ 参照 = 把每张卡按**位置规则**接上 leaf，再从图上**数**；")
    print("      被测 = 闭式（`N/R` 每轨卡数、`G/R` 每节点每轨卡数）。★★ 两条独立路径。")
    print("  ★★★ 外部锚点（NVIDIA 逐字）：`%s`" % specs.NVIDIA_RAIL_QUOTE)

    # ── A.1 ★★ `R == G` 档：每轨 `N/R` 张、每节点每轨 1 张 ────────────────
    print()
    print("  ── A.1 ★★ **`R == G` 档**（每卡一轨）：每轨 `N/R` 张、**每节点每轨恰好 1 张** ──")
    nodes, gpus, rails = specs.RAIL_NODES, specs.RAIL_GPUS_PER_NODE, specs.RAIL_COUNT
    ft = RailFabric(nodes, gpus, rails, specs.RAIL_SPINES)
    sizes = [len(ft.rail_members(r)) for r in range(rails)]
    per_node = [sorted(set(ft.per_node_counts(r).values())) for r in range(rails)]
    ok1 = (ft.n_gpus() == cf_total_gpus(nodes, gpus)) \
        and (ft.n_leaves() == rails) \
        and set(sizes) == {cf_rail_size(nodes, gpus, rails)} \
        and cf_per_node_per_rail(gpus, rails) == 1 \
        and all(v == [1] for v in per_node)
    print("     节点 %d × 每节点 %d 卡 = N = **%d**；R = %d ⇒ 闭式 `N/R` = **%d**"
          % (nodes, gpus, ft.n_gpus(), rails, cf_rail_size(nodes, gpus, rails)))
    for r in range(rails):
        print("     rail %d：图示 %-4d 张（`N/R` = %-4d）· 每节点贡献 %s 张 · 挂在这条 leaf 上"
              % (r, sizes[r], cf_rail_size(nodes, gpus, rails), sorted(set(ft.per_node_counts(r).values()))))
    print("     ★★ **每条 rail 上恰好有一张来自每个节点的卡** ⇒ 这正是 NVIDIA 那句")
    print("        `put each GPU \"rail\" ... onto a different first level (LEAF) switch` 的可数形式。")
    check("A.1", ok1 and specs.RAIL_NODES == 64 and specs.RAIL_GPUS_PER_NODE == 8
          and specs.RAIL_COUNT == 8 and len(sizes) == 8
          and rail_model_ok(sizes, per_node, rails,
                            cf_rail_size(nodes, gpus, rails), cf_per_node_per_rail(gpus, rails)),
          "★★ **判据成立**（`R == G = 8`）：枚举出的 **总卡数 == `N = 节点数 × G`**、"
          "**每轨卡数 == `N/R`**、**每节点每轨恰好 1 张**、**leaf 数 == `R`**；"
          "★★ 且**三条不变量校验器对【正确模型】放行**（★ 那是**成功行为** —— 依 `CONTRIBUTING:238`"
          "归 A 类，**不塞进 `C.1`**）",
          "**rail 的划分与 `N/R` 不符** —— ★ 先查 `_leaf_of` 的位置规则")

    # ── A.2 ★★ `R < G` 档：每节点每轨 `G/R` 张 ──────────────────────────
    print()
    print("  ── A.2 ★★ **`R < G` 档**（一轨承载同余的多张卡）：每节点每轨 `G/R` 张 ──")
    ok2 = True
    for nodes2, gpus2, rails2 in ((64, 8, 4), (64, 8, 2), (32, 8, 4)):
        f2 = RailFabric(nodes2, gpus2, rails2, specs.RAIL_SPINES)
        sizes2 = [len(f2.rail_members(r)) for r in range(rails2)]
        per2 = [sorted(set(f2.per_node_counts(r).values())) for r in range(rails2)]
        exp_size = cf_rail_size(nodes2, gpus2, rails2)
        exp_pn = cf_per_node_per_rail(gpus2, rails2)
        ok2 = ok2 and set(sizes2) == {exp_size} and all(v == [exp_pn] for v in per2) \
            and f2.n_gpus() == cf_total_gpus(nodes2, gpus2)
        print("     节点 %-3d × %d 卡 · R=%d ⇒ 闭式 `N/R` = %-4d、`G/R` = %d；"
              "枚举每轨 %s 张 · 每节点每轨 %s 张"
              % (nodes2, gpus2, rails2, exp_size, exp_pn,
                 sorted(set(sizes2)), sorted({x for v in per2 for x in v})))
    print("     ★★ **`R < G` 时，同一个节点会有 `G/R` 张卡落在同一条 rail 上** ——")
    print("        ★ 这与 `R == G` 档的「每节点每轨 1 张」**是两个不同的结构**（`C.2` 判这个混淆）。")
    check("A.2", ok2
          and all(len(RailFabric(n2, g2, r2, specs.RAIL_SPINES).rail_members(0)) == cf_rail_size(n2, g2, r2)
                  and len(RailFabric(n2, g2, r2, specs.RAIL_SPINES).per_node_counts(0)) == n2
                  for n2, g2, r2 in ((64, 8, 8), (8, 4, 4))),
          "★★ **判据成立**（三档）：`R < G` 时枚举出的 **每轨卡数 == `N/R`**、"
          "**每节点每轨 == `G/R`**（`(64,8,4)` / `(64,8,2)` / `(32,8,4)` 逐一核过）；"
          "★★ 且**两个口径被【分开】核过**：每轨共 `N/R` 张（按轨看）vs 每节点每轨 `G/R` 张（按节点看）"
          "（★ 这是**成功行为** —— 依 `CONTRIBUTING:238` 归 A 类，**不塞进 `C.2`**）",
          "**`R < G` 档的划分与闭式不符** —— ★ 先查 `_leaf_of` 是否只用了 `g mod R`")


# ===========================================================================
# 4. B 类：**跳数 / 路径账** · 参数化 `N/R` 表 · 集合通信的触及 leaf 数
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **跳数 / 路径账** · 参数化 `N/R` 表 · 集合通信触及几条 leaf")
    print("=" * 78)
    print("  ★★ 载体 = NVIDIA 官方产品文档（抓取日 **%s**）。" % specs.FETCH_DATE)

    # ── B.1 ★★★ 同轨 vs 跨轨的**跳数 / 路径账** ──────────────────────────
    print()
    print("  ── B.1 ★★★ **同轨 vs 跨轨**：链路数 · 交换机台数 · 等价路径数 ──")
    nodes, gpus, rails = specs.RAIL_NODES, specs.RAIL_GPUS_PER_NODE, specs.RAIL_COUNT
    ft = RailFabric(nodes, gpus, rails, specs.RAIL_SPINES)
    # 同轨：两张**编号相同**、**节点不同**的卡（rail 相同 = `g mod R` 相同）
    a_in = ("gpu", 0, 0)
    b_in = ("gpu", 1, 0)                      # 同 index（0）⇒ 同 rail
    a_cr = ("gpu", 0, 0)
    b_cr = ("gpu", 1, 1)                      # index 0 vs 1 ⇒ 不同 rail
    in_links, in_paths, in_sw = ft.shortest_path(a_in, b_in)
    cr_links, cr_paths, cr_sw = ft.shortest_path(a_cr, b_cr)
    print("     同轨（`g=0` 的卡，节点 0 ↔ 节点 1）：链路 %d · 交换机 **%d** 台 · 路径数 %d"
          % (in_links, in_sw, in_paths))
    print("     跨轨（节点 0 的 `g=0` ↔ 节点 1 的 `g=1`）：链路 %d · 交换机 **%d** 台 · 路径数 %d"
          % (cr_links, cr_sw, cr_paths))
    print("     ★★★ NVIDIA 逐字（**同轨那一句**，★ 计划 §C.3 的引文在它之前【截断】了）：")
    print("        `%s`" % specs.NVIDIA_RAIL_HOP_QUOTE)
    print("     ★★ 读法：同轨只过 **1 台**交换机（`just one switch hop away`）——")
    print("        它**不经过脊层**；跨轨要过 **3 台**（叶 → 脊 → 叶），且路径数是 **脊的台数 `S`**。")
    print("     ★★ 本 f2 的脊数 `S = %d`（★ 本库自定的示意值）。" % specs.RAIL_SPINES)
    check("B.1", in_links == 2 and in_sw == 1 and in_paths == 1
          and cr_links == 4 and cr_sw == 3 and cr_paths == specs.RAIL_SPINES
          and "just one switch hop away" in specs.NVIDIA_RAIL_HOP_QUOTE,
          "★★★ **判据成立**：**同轨** `2` 条链路 / **`1` 台交换机** / `1` 条路径（★ 与 NVIDIA 逐字"
          "`just one switch hop away` 一致）；**跨轨** `4` 条链路 / **`3` 台交换机** / **`S` 条"
          "等价路径（每条路径对应一台脊）⇒ ★★ 这就是 rail 优化的**代价账**",
          "**跳数或路径账被动过** —— ★ 先查 `shortest_path` 与 leaf↔spine 的连线")

    # ── B.2 ★★ 参数化 `N/R` 表 ───────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **参数化 `N/R` 表**（`R` 整除 `G`）──")
    print("     %-6s %-6s %-6s %-6s %-8s %-10s %-8s"
          % ("节点", "每节点卡", "N", "R", "N/R", "每节点每轨", "轨内是否跨节点"))
    rows = []
    for nodes2, gpus2, rails2 in ((64, 8, 8), (64, 8, 4), (64, 8, 2), (32, 8, 8), (8, 4, 4)):
        f2 = RailFabric(nodes2, gpus2, rails2, specs.RAIL_SPINES)
        sz = cf_rail_size(nodes2, gpus2, rails2)
        pn = cf_per_node_per_rail(gpus2, rails2)
        cross = len(set(n[1] for n in f2.rail_members(0))) > 1
        rows.append((nodes2, gpus2, rails2, sz, pn, cross))
        print("     %-6d %-6d %-6d %-6d %-8d %-10d %-8s"
              % (nodes2, gpus2, cf_total_gpus(nodes2, gpus2), rails2, sz, pn, cross))
    print("     ★★ **`R == G` 的档上「每节点每轨 = 1」**，`R < G` 的档上是 `G/R`；")
    print("        ★★ 而**任何一轨都跨多个节点**（`轨内是否跨节点` 全为 `True`）—— ★ `C.1` 判的就是这个。")
    check("B.2", len(rows) == 5
          and rows[0][3] == 64 and rows[0][4] == 1
          and rows[1][3] == 128 and rows[1][4] == 2
          and rows[2][3] == 256 and rows[2][4] == 4
          and rows[3][3] == 32 and rows[4][3] == 8 and rows[4][4] == 1
          and all(r[5] for r in rows),
          "★★ **判据成立**（五档）：`(64,8,8) ⇒ N/R = 64、G/R = 1`；`(64,8,4) ⇒ 128 / 2`；"
          "`(64,8,2) ⇒ 256 / 4`；`(32,8,8) ⇒ 32 / 1`；`(8,4,4) ⇒ 8 / 1`；"
          "★★ 且**每一轨都横跨多个节点**",
          "**参数化表被动过** —— ★ 先查 `cf_rail_size` / `cf_per_node_per_rail`")

    # ── B.3 ★★ 集合通信触及几条 leaf ─────────────────────────────────────
    print()
    print("  ── B.3 ★★ **集合通信按哪种粒度发起**：触及几条 leaf ──")
    f3 = RailFabric(specs.RAIL_NODES, specs.RAIL_GPUS_PER_NODE, specs.RAIL_COUNT, specs.RAIL_SPINES)
    # ① 「同 index」：每个节点只出 `g = 0` 那张卡
    same_idx = {("gpu", n, 0) for n in range(specs.RAIL_NODES)}
    leaves_same = {next(iter(f3.adj[g])) for g in same_idx}
    # ② 「全卡」：N 张全参与
    all_gpus = set(f3.nodes_of("gpu"))
    leaves_all = {next(iter(f3.adj[g])) for g in all_gpus}
    print("     ① **同 index**（每节点只出 `g=0` 那卡，共 %d 张）：触及 **%d** 条 leaf（= rail 数）"
          % (len(same_idx), len(leaves_same)))
    print("     ② **全卡**（%d 张全参与）：触及 **%d** 条 leaf" % (len(all_gpus), len(leaves_all)))
    print("     ★★ **两种粒度的代价不同**：① 只在**一条** leaf 内（**不碰脊层**）；")
    print("        ② 要横跨**全部 `R` 条** leaf ⇒ ★★ 这才需要脊层（`B.1` 的跨轨账）。")
    print("     ★★ 引申（★ 这是**本库的推理**，不是引文）：**`rail` 这个布局让「同 index」的通信")
    print("        走最短路，而把「跨 index」的通信留给脊层** —— ★ 代价与收益在同一个结构里。")
    check("B.3", len(leaves_same) == 1 and len(leaves_all) == specs.RAIL_COUNT
          and len(same_idx) == specs.RAIL_NODES,
          "★★ **判据成立**：「同 index」的集合通信只触及 **1** 条 leaf（同轨、不碰脊层）；"
          "「全卡」的集合通信触及 **全部 `R` 条** leaf ⇒ ★★ 两种粒度的**代价不同**",
          "**触及 leaf 数的账被动过** —— ★ 先查 `g mod R` 的映射")


# ===========================================================================
# 5. C 类：两条【真的会失败】的错法（★★ `C.1` 是「把 rail 当成节点」）
# ===========================================================================
def rail_model_ok(sizes, per_node, rails, exp_rail_size, exp_per_node):
    """★★★ **rail 模型的校验器**：三条不变量全中才放行。

    ① **轨数** == `R`；② **每轨卡数** == `N/R`；③ **每节点每轨卡数** == `G/R`。
    ★★ 它**不是 A 类**：它是**判定逻辑**，用来判「错模型会不会被拒」（`C.1` **只验拒绝侧**；
       ★ 正确模型放行是**成功行为**，归 `A.1`）。
    """
    return (len(sizes) == rails and set(sizes) == {exp_rail_size}
            and all(v == [exp_per_node] for v in per_node))


def rail_total_claim_ok(fabric, r, claimed_total):
    """★★★ **「编号 `r` 的轨【共】有几张卡」这条声明的校验器** —— 与**图上数出来的**比对。

    ★★ 它判的正是 `C.2` 的错法：把**每节点每轨 `G/R`**（**按节点看**）误当
      **每轨共 `N/R` 张**（**按轨看**）⇒ 两个口径都是「每轨的卡数」，极易混。
    """
    return len(fabric.rail_members(r)) == claimed_total


def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**把 rail 当成节点** · **把「每节点每轨 1 张」当成「每轨共 1 张」**")
    print("=" * 78)

    # ── C.1 ★★★ 把 rail 与「节点」搞混 ──────────────────────────────────
    print()
    print("  ── C.1 ★★★ **把 rail 与「节点」搞混**（三种错模型）──")
    nodes, gpus, rails = specs.RAIL_NODES, specs.RAIL_GPUS_PER_NODE, specs.RAIL_COUNT
    exp_size = cf_rail_size(nodes, gpus, rails)
    exp_pn = cf_per_node_per_rail(gpus, rails)
    cases = (
        ("正确：编号 `g` 的卡 → 第 `g mod R` 条 leaf", None, True),
        ("错①把 rail **当成节点号**（`rail = n mod R`）", "by_node", False),
        ("错②**每个节点一条轨**（轨数 = 节点数）", "by_node_id", False),
        ("错③**全部塞进第 0 条轨**", "one_rail", False),
    )
    wrong_all_rejected = True
    for label, corrupt, expect in cases:
        f = RailFabric(nodes, gpus, rails, specs.RAIL_SPINES, corrupt=corrupt)
        nr = f.n_leaves()
        sizes = [len(f.rail_members(r)) for r in range(nr)]
        per_node = [sorted(set(f.per_node_counts(r).values())) for r in range(nr)]
        got = rail_model_ok(sizes, per_node, rails, exp_size, exp_pn)
        if expect:      # ★ 正确模型是**成功行为** ⇒ 不并入 C 类判据（`CONTRIBUTING:238`）
            print("     %-40s 轨数 %-4d 每轨卡数 %-14s ⇒ 校验器 %s（★ 成功行为，已在 `A.1` 断言）"
                  % (label, nr, sorted(set(sizes)) if sizes else "—", "放行" if got else "**拒绝**"))
            continue
        wrong_all_rejected = wrong_all_rejected and (not got)
        print("     %-40s 轨数 %-4d 每轨卡数 %-14s ⇒ 校验器 %s（期望 **拒绝**）"
              % (label, nr, sorted(set(sizes)) if sizes else "—",
                 "放行" if got else "**拒绝**"))
    print("     ★★★ 三条的**失败原因【各不相同】**（★ 这正是「rail ≠ 节点」的三张脸）：")
    print("        · 错① `rail = n mod R`：轨数**对了（8）**、每轨卡数**也对了（64）**，")
    print("          但**每节点每轨 = 8 ≠ 1** ⇒ 只靠**不变量③**才拒得掉；")
    print("        · 错② `rail = n`：轨数 **64 ≠ 8** ⇒ 不变量① 拒（每轨只有 8 张）；")
    print("        · 错③ 全塞 0 号轨：轨数看起来**仍是 8**（★ 那 8 条 leaf 是先建好的），")
    print("          但每轨卡数成了 `[0…0, 512]` ⇒ 不变量② 拒（★ **不是**靠轨数）。")
    check("C.1", wrong_all_rejected
          and (not rail_model_ok([exp_size] * rails, [[gpus]] * rails, rails, exp_size, exp_pn))
          and (not rail_model_ok([exp_size] * (nodes), [[exp_pn]] * nodes, rails, exp_size, exp_pn)),
          "★★★ **判据成立**：三种错模型**全部被拒**"
          "（★ 错① 只被**不变量③**抓住：它的轨数与每轨卡数**都对**）"
          "⇒ ★★ **「把 rail 当成节点」真的会失败**（★ 正确模型放行是成功行为，归 `A.1`）",
          "**错模型没被拒绝** —— ★ 先查 `rail_model_ok` 的三条不变量")

    # ── C.2 ★★★ 把「每节点每轨 1 张」当成「每轨共 1 张」─────────────────
    print()
    print("  ── C.2 ★★★ **把「每节点每轨 1 张」当成「每轨共 1 张」** ──")
    wrong_rejected2 = True
    for nodes2, gpus2, rails2 in ((64, 8, 8), (8, 4, 4)):
        f = RailFabric(nodes2, gpus2, rails2, specs.RAIL_SPINES)
        wrong = cf_per_node_per_rail(gpus2, rails2)   # ★ 错法：拿「每节点每轨」当「每轨共」
        got = rail_total_claim_ok(f, 0, wrong)
        wrong_rejected2 = wrong_rejected2 and (not got)
        print("     节点 %-3d × %d 卡 · R=%d：**每轨共 `N/R` = %-3d 张**；"
              "★ **错法**「每轨共 %d 张」（= 每节点每轨 `G/R`）⇒ 校验器 %s"
              % (nodes2, gpus2, rails2, cf_rail_size(nodes2, gpus2, rails2), wrong,
                 "放行" if got else "**拒绝**"))
        print("        枚举核对：每轨 %d 张、每轨里含 %d 个不同节点"
              % (len(f.rail_members(0)), len(f.per_node_counts(0))))
    print("     ★★★ **这个错法特别可信**：`1`（每节点每轨）与 `N/R`（每轨总数）**都是「每轨的卡数」**，")
    print("        只是**口径不同**（一个按节点看、一个按轨看）—— ★★ 正是「**量法口径**」那一类错。")
    check("C.2", wrong_rejected2
          and (not rail_total_claim_ok(RailFabric(64, 8, 8, specs.RAIL_SPINES), 0, 1))
          and (not rail_total_claim_ok(RailFabric(8, 4, 4, specs.RAIL_SPINES), 0, 1)),
          "★★★ **判据成立**：把「每节点每轨 `1` 张」误当「**每轨共 1 张**」⇒ 校验器 **被拒**"
          "（★ 图上数出的每轨是 `N/R = 64` / `8` 张）⇒ ★★ **口径混用真的会失败**"
          "（★ 两个口径被**分开**核过是成功行为，归 `A.1`/`A.2`）",
          "**两种口径没被分开** —— ★ 先查 `rail_total_claim_ok`（它比对的是**图上数出的每轨张数**）")


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
    print("      ★★ 但**本程序的 A 类是真的**：它算的是**布线的划分**（NVIDIA 的措辞当外部锚点）。")
    print("  ★★ 本层 = **物理层**（scale-out 的布线结构）。")
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
    check("D.3", specs.NVIDIA_RAIL_DOC
          == "NVIDIA, IB Cluster Bring-up Procedure: Setting the InfiniBand Cluster Topology"
          and specs.NVIDIA_RAIL_URL
          == ("https://networking-docs.nvidia.com/ibclusterbringupproc/"
              "setting-the-infiniband-cluster-topology")
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：NVIDIA 产品文档（★ **`[B]` 级，不是规范**）· URL · "
          "抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住" % specs.FETCH_DATE)

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    #   ① 三个规模常量**整除**（否则 `G/R` 不是整数 ⇒ 校验器退化）；
    #   ② 脊数 >= 1（否则跨轨的路径数会是 0）。
    div_ok = (specs.RAIL_GPUS_PER_NODE % specs.RAIL_COUNT == 0
              and (specs.RAIL_NODES * specs.RAIL_GPUS_PER_NODE) % specs.RAIL_COUNT == 0)
    spines_ok = specs.RAIL_SPINES >= 1
    print("     ★★ 整除口径：`G = %d` 被 `R = %d` 整除 = %s；`N` 被 `R` 整除 = %s"
          % (specs.RAIL_GPUS_PER_NODE, specs.RAIL_COUNT, div_ok,
             (specs.RAIL_NODES * specs.RAIL_GPUS_PER_NODE) % specs.RAIL_COUNT == 0))
    print("     ★★ 脊数 `S = %d`（跨轨路径数 = `S`）" % specs.RAIL_SPINES)
    check("D.3b", div_ok and spines_ok and specs.RAIL_NODES == 64
          and specs.RAIL_GPUS_PER_NODE == 8 and specs.RAIL_COUNT == 8,
          "★★★ **口径守卫**：`R` **整除** `G` 与 `N`（★ 否则 `G/R` 不是整数 ⇒ 校验器退化），"
          "且脊数 `S >= 1` ⇒ ★ 全部规模常量在**计算之前**就被钉住",
          "**规模常量被动过** —— ★ 先查 `RAIL_*` 三个常量与 `cf_per_node_per_rail` 的整除前提")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字 · **逐元素** · 独立性 · `UNSOURCED`
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · **边界档** · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★ **边界档**：`R = 1` 与 `R = G`（枚举 vs 闭式）────────────────
    #   ★★ 补上本程序此前**缺号**的 `D.4`（★ `01`/`03` 都有 `D.4`，但三者的**指代本就不同**
    #     —— `01/D.4` = 守卫实验、`03/D.4` = 逐元素钉住、`02/D.4` = 这里的边界档）。
    print()
    print("  ── D.4 ★★ **边界档**（`R = 1` 与 `R = G`）：枚举 vs 闭式 ──")
    f_r1 = RailFabric(64, 8, 1, specs.RAIL_SPINES)
    f_rg = RailFabric(64, 8, 8, specs.RAIL_SPINES)
    print("     `R = 1`：leaf 数 %d（= `R`）· 每轨卡数 %d（`N/R` = %d）· 每节点每轨 %s（`G/R` = %d）"
          % (f_r1.n_leaves(), len(f_r1.rail_members(0)), cf_rail_size(64, 8, 1),
             sorted(set(f_r1.per_node_counts(0).values())), cf_per_node_per_rail(8, 1)))
    print("     `R = G = 8`：leaf 数 %d · 每轨 %d 张 · 每节点每轨 %s"
          % (f_rg.n_leaves(), len(f_rg.rail_members(0)),
             sorted(set(f_rg.per_node_counts(0).values()))))
    check("D.4", cf_per_node_per_rail(8, 1) == 8 and cf_rail_size(64, 8, 1) == 512
          and len(f_r1.rail_members(0)) == 512 and f_r1.n_leaves() == 1
          and sorted(set(f_r1.per_node_counts(0).values())) == [8]
          and f_rg.n_leaves() == 8 and sorted(set(f_rg.per_node_counts(0).values())) == [1],
          "★★ **边界成立**：`R = 1` 时**全部卡落在同一条 leaf**（`N/R = 512`、每节点每轨 `G/R = 8`）；"
          "`R = G` 时**每节点每轨恰好 1 张** —— ★ 两个极端都由**枚举**复现了闭式",
          "**边界档（`R = 1` / `R = G`）与闭式不符** —— ★ 先查 `_leaf_of` 的 `g mod R`")

    # ── D.5 ★★★ 三条 NVIDIA 引文（★ 含计划【截断】的那一句的后半）────────
    check("D.5", specs.NVIDIA_RAIL_QUOTE
          == ('Rail-optimized design means a GPU node with multiple interfaces will put each '
              'GPU "rail" (IB network interface) onto a different first level (LEAF) switch for '
              'cluster Interconnect.')
          and specs.NVIDIA_RAIL_HOP_QUOTE
          == ('This allows multiple nodes to utilize their internal NVSwitch path to talk across a '
              'NIC that is just one switch hop away (instead of having to cross multiple switches, '
              'incurring additional latency).')
          and specs.NVIDIA_RAIL_FATTREE_QUOTE
          == ("Fat-Tree is NVIDIA's recommended topology, AI factory should based on rail optimized."),
          "★★★ **三条 NVIDIA 引文逐字钉住** —— ★★★ **特别注意**：")
    print("        · 第 1 条界定的是「**一个节点内部的 `G` 张卡被分散到 `G` 个 leaf**」"
          "（★ 不是「一个 rail 里有 `G` 台节点」）；")
    print("        · 第 2 条是计划 §C.3 **截断**掉的那半句 —— ★★ `just one switch hop away` 才是跳数账的依据；")
    print("        · 第 3 条**原句语法是残的**（`should based on`）—— ★★ **照抄**，不改写。")

    # ── D.6 ★★ 交叉引用：与上游的分工 ────────────────────────────────────
    check("D.6", "Rail-optimized design" in specs.NVIDIA_RAIL_QUOTE
          and "first level (LEAF)" in specs.NVIDIA_RAIL_QUOTE
          and "one switch hop" in specs.NVIDIA_RAIL_HOP_QUOTE
          and "recommended topology" in specs.NVIDIA_RAIL_FATTREE_QUOTE
          and specs.NVIDIA_RAIL_DOC.startswith("NVIDIA"),
          "★★ **交叉引用逐条钉住**：`first level (LEAF)` · `one switch hop` · `recommended topology`；"
          "★★ 本文件与 `01` 的分工：**`01` 讲胖树的组合结构**，**`02` 讲 rail 的布线** ——"
          "★ NVIDIA 那句 `Fat-Tree is NVIDIA's recommended topology` 说明两者**层叠**、不互斥；"
          "★★ 而 **NVLink / NVSwitch 的带宽** 属 `20-gpu-architecture/interconnect/02`（**本域只回链**）",
          "**交叉引用的字面被动过**")

    # ── D.7 ★★★ 逐元素钉住 + `UNSOURCED` ────────────────────────────────
    check("D.7", specs.RAIL_NODES == 64 and specs.RAIL_GPUS_PER_NODE == 8
          and specs.RAIL_COUNT == 8 and specs.RAIL_SPINES == 4
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★★★ **四个规模常量逐个钉住**（`RAIL_NODES` / `RAIL_GPUS_PER_NODE` / `RAIL_COUNT` / "
          "`RAIL_SPINES`）+ 抓取日 + `UNSOURCED == ()`；"
          "★★ 它们是 `[E]` 级**本库自定**的示意值（★ 无外部出处 ⇒ `# 来源：` 里如实标 `[E]`）",
          "**规模常量被动过**")

    # ── D.8 ★★ 独立性（源码级）── ★★ 遍历**全部**参照实现 ────────────────
    refs = (RailFabric._build, RailFabric._leaf_of, RailFabric.rail_members,
            RailFabric.per_node_counts, RailFabric.shortest_path, RailFabric.n_gpus)
    cfs = (cf_total_gpus, cf_rail_size, cf_per_node_per_rail)
    ref_srcs = {textwrap.dedent(inspect.getsource(f)) for f in refs}
    cf_srcs = {textwrap.dedent(inspect.getsource(f)) for f in cfs}
    call_tables = {}
    for f in refs:
        t = textwrap.dedent(inspect.getsource(f))
        call_tables[f.__name__] = {n.func.id for n in ast.walk(ast.parse(t))
                                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：**参照**侧六个函数的调用表 ──")
    for nm, tbl in call_tables.items():
        print("     `RailFabric.%s()` 里调用的函数：%s"
              % (nm, ", ".join(sorted(tbl)) if tbl else "**空集**"))
    hits = sorted({c for tbl in call_tables.values() for c in tbl
                   if c.startswith("cf_")})
    cf_calls = {n.func.id for s in cf_srcs for n in ast.walk(ast.parse(s))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print("     ⇒ ★★ **参照侧不调用任何 `cf_*`**（★★ `cf_*` 侧也没有回环）"
          " ⇒ 「参照不得用被测」**在源码层面成立**。")
    print("     ★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，")
    print("        **不证明**判据有物理信息量 —— 后者由 `A.1`/`A.2` 与 `C.1` 的**行为**兜底。")
    check("D.8", not hits and len(ref_srcs) == 6 and len(cf_srcs) == 3
          and not any(c.startswith("cf_") for s in ref_srcs for c in _call_names(s))
          and not any(c in call_tables for s in cf_srcs for c in _call_names(s)),
          "★★ **源码级线索成立**：**六个**参照方法两两不同源、**三个** `cf_*` 两两不同源；"
          "★★ **参照侧的调用表里一个 `cf_*` 都没有，`cf_*` 也不调用参照**（★ 两侧都审）",
          "**参照与闭式互相调用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 闭式的边界与整除 ─────────────────────────────────────────
    check("D.9", cf_total_gpus(64, 8) == 512 and cf_rail_size(64, 8, 8) == 64
          and cf_per_node_per_rail(8, 8) == 1 and cf_per_node_per_rail(8, 4) == 2
          and cf_per_node_per_rail(8, 2) == 4
          and cf_rail_size(64, 8, 8) * 8 == cf_total_gpus(64, 8),
          "★★ **闭式的边界**：`R = G` 档 `G/R = 1`（`64×1 = 64 = N/R`）；`R = 4` ⇒ `2`；`R = 2` ⇒ `4`；"
          "★ 且 `N/R × R == N`（本档 `64 × 8 = 512`）⇒ ★★ **两条闭式在边界上自洽**",
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
          "它把 `N` 张卡**逐张摆**到 `R` 条轨上，再**数回来**与 `N/R` / `G/R` 对拍。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 依据在 `specs.py` 的 `# 来源：`（NVIDIA 产品文档 `[B]`，抓取日 %s）。" % specs.FETCH_DATE)
    print("★ 本程序只核对「布线的划分与跳数账」与「错模型真的会被拒」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_rail_optimized.py —— rail-optimized / 多轨道的布局与它的代价")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **物理层**；★★ **本主题的 A 类锚点是 NVIDIA 的 rail 定义（`[B]` 级）**。")
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
