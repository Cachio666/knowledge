# -*- coding: utf-8 -*-
"""03 layout conversion 的泄漏：判定哪里必须插转换，并给出代价模型。

★ 这一篇的全部**代价**结论都是**模型值**（本机无 GPU、`triton` 包全机器不存在）。强的一档只有两处：
  · **A 类**：两张「同数学、不同转换次数」的图，输出**逐位相同**（用 `graph_execute` 跑 numpy）
  · **C 类**：故意拒绝插入转换 → 必须抛 `LayoutConflict`；故意改错免费/昂贵的判定 → 断言必须失败

★★ 本文件负责四条语义锚点（`specs.ANCHORS` 的 `G1` / `G3` / `G4` / `G5`，A 级：官方文档逐字）：

    `G1` layout 转换可能很贵，因为跨 warp 要走**共享内存**
    `G3` 存在「**只重排线程内寄存器**」的代价为零的转换（原文括号：`which is free`）
    `G4` 「内维不再连续 ⇒ **没有合并访存**」（官方实测量 `0.774 TB/s`，★ **不是本机实测**）
    `G5` 「先转成高效布局再做归约通常**更贵**」——★ 原文末词拼作 **`reeduction`**（官方拼写错误，
        **逐字保留，引用时不得「修好」**）

★★ 本文件还负责**在线再抽取**（`--anchors-net`）：重新下载官方页面、用 `specs.py` 里写明的
   抽取口径重算词频与引文，并**逐数比对** `specs.WORD_FREQ` 与 `specs.ANCHORS`。
   ⇒ 「主 tutorial 里 `coalesc` 是 0 次」这句话**不是转述**，是可复算的。

★ 四处限定语（计划 §0.7 F10）：迷你实现不是 Triton；`specs.MODEL_OWN_RULES` 的 5 条规则自造；
  B 类全是模型值、本主题无实测；不是硬件也不是 Triton 编译器。
"""

import itertools
import os
import re
import sys

import numpy as np

import checks as C
import minitriton as mt
import specs

W = 4                       # num_warps
SH = (128, 128)             # 方形 tile，方便转置
COVERED_ANCHORS = ("G1", "G3", "G4", "G5")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

chk = C.Checks("03_layout")

X = (np.arange(SH[0] * SH[1], dtype=np.float32).reshape(SH) % 251) * 0.5 - 60.0
Y = ((np.arange(SH[0] * SH[1], dtype=np.float32).reshape(SH) * 7) % 97) * 0.25

OFFICIAL_LITERAL = "gl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])"


def print_env():
    print("  解释器 = %s" % sys.executable)
    print("  python %s / numpy %s（**无 GPU**；`triton` 包全机器不存在）"
          % (sys.version.split()[0], np.__version__))


def print_anchors():
    """把本文件负责的四条锚点**逐字**打出来（常量 ↔ 输出的双向绑定）。"""
    for aid in COVERED_ANCHORS:
        a = specs.ANCHORS[aid]
        print("  · %s（页面 %s｜%s）" % (aid, a["page"], a["section"]))
        print("      「%s」" % a["quote"])


# ---------------------------------------------------------------------------
# 9. 在线再抽取（`make anchors` / `--anchors-net`）—— 把「转述」变成「可复算」
#    ★ 定义放在前面、**紧跟一次 dispatch**：带 `--anchors-net` 时**不跑**离线那套断言
#      （离线与在线是两条入口，混在一起会让「哪一档失败了」说不清）。
# ---------------------------------------------------------------------------

def _fetch(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "knowledge-base-anchors/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _stripped(raw_text):
    """`specs.py` 模块头写明的抽取口径，**逐条实现**（口径不写下来就没法复现）。"""
    import html as H
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw_text)   # strip(script|style)
    t = re.sub(r"(?s)<[^>]+>", " ", t)                           # strip(tags)
    t = H.unescape(t)                                            # html.unescape
    return re.sub(r"\s+", " ", t)                                # collapse(\s+)


def anchors_net():
    print("=" * 74)
    print("在线再抽取：从官方页面重算词频与引文，逐数比对 specs.py")
    print("=" * 74)
    bad = []
    for page in (specs.VECTOR_ADD_PAGE, specs.GLUON_PAGE):
        url = specs.BASE_URL + page
        print()
        print("· %s" % url)
        t = _stripped(_fetch(url))
        rec = specs.WORD_FREQ[page]
        got = {
            "coalesc_cs": t.count("coalesc"),
            "layout_conversions_cs": t.count("Layout conversions"),
            "warp_cs": t.count("warp"),
            "warp_ci": len(re.findall("warp", t, re.I)),
            "stripped_len": len(t),
        }
        for k, v in got.items():
            mark = "OK" if v == rec[k] else "**MISMATCH**"
            print("    %-22s 再抽取 %-7d specs 记录 %-7d %s" % (k, v, rec[k], mark))
            if v != rec[k]:
                bad.append((page, k, v, rec[k]))
    seen = {}
    for aid, a in specs.ANCHORS.items():
        url = specs.BASE_URL + a["page"]
        t = seen.get(a["page"]) or _stripped(_fetch(url))
        seen[a["page"]] = t
        for s in a["must_contain"]:
            ok = s in t
            print("    %-4s must_contain 逐字命中 = %-5s %r" % (aid, ok, s[:60]))
            if not ok:
                bad.append((aid, "must_contain", s, None))
    for aid, a in specs.LOAD_QUOTES.items():
        t = _stripped(_fetch(specs.BASE_URL + a["page"]))
        for s in a["must_contain"]:
            ok = s in t
            print("    %-4s must_contain 逐字命中 = %-5s %r" % (aid, ok, s[:60]))
            if not ok:
                bad.append((aid, "must_contain", s, None))
    print()
    if bad:
        print("!! 在线再抽取有 %d 处对不上 specs.py：%r" % (len(bad), bad[:8]))
        print("   （这**不是**「文档变了」的自动结论：先确认抽取口径与页面，再决定改哪一边 ——")
        print("     不得静默放宽 specs.py，也不得不记这笔账。）")
        return 1
    print("✓ 两个页面的 5 个量 + 7 条锚点的全部 must_contain 逐字命中 —— 与 specs.py 完全一致")
    return 0


if __name__ == "__main__" and "--anchors-net" in sys.argv:
    sys.exit(anchors_net())


# ---------------------------------------------------------------------------
# 1. 两种规范布局：规则算出来的，正好复现官方文档逐字给出的那一条
# ---------------------------------------------------------------------------

C.rule("1 两种规范布局（coalesced_layout 的规则 vs 官方文档逐字给出的一条）")
print_env()
print()

L = mt.coalesced_layout(SH, W, order=[1, 0], dtype="f32")     # 沿最内维合并访存
Lrow = mt.coalesced_layout(SH, W, order=[0, 1], dtype="f32")  # 沿第 0 维合并访存
print("L   （内维合并）    : %r" % (L,))
print("Lrow（第 0 维合并） : %r" % (Lrow,))
print("Lrow 的参数字面量   : %s" % OFFICIAL_LITERAL)
print("官方文档原文那一行  : gl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])")
chk.B(Lrow.spt == (1, 128) and Lrow.tpw == (32, 1) and Lrow.wpc == (4, 1)
      and Lrow.order == (0, 1),
      "B.1 本模型的 coalesced_layout 规则在 shape=(128,128)/num_warps=4/order=[0,1] 上"
      "**逐参数复现**了官方文档给出的 BlockedLayout —— 但**复现同一个结果 ≠ 同一个规则**"
      "（`specs.MODEL_OWN_RULES['R1']` 明写摊维规则是本模型自造的）",
      "本模型 spt/tpw/wpc/order = %r == 官方 [1,128],[32,1],[4,1],[0,1]"
      % (list(Lrow.spt) + list(Lrow.tpw) + list(Lrow.wpc) + list(Lrow.order),))

owners = L.owners()
print("L 的属主分布：%d 个元素映射到 %d 个线程（每线程 %d 个）"
      % (owners.size, len(np.unique(owners)), owners.size // len(np.unique(owners))))
print("L 里第 0 行的 128 个元素分别属于 %d 个不同线程"
      % len(np.unique(owners[0])))
print("Lrow 里第 0 行的 128 个元素分别属于 %d 个不同线程"
      % len(np.unique(Lrow.owners()[0])))
chk.B(len(np.unique(owners[0])) == 128 and len(np.unique(Lrow.owners()[0])) == 1,
      "B.2 两种布局的差别就是「一行归谁」：L 里一行散在 128 个线程上，Lrow 里一行归 1 个线程"
      "——这正是后面所有归约代价差异的**根源**",
      "散在 128 个线程 vs 归 1 个线程")

# A.8：把「精确切分」这条**简化**摆出来（`specs.MODEL_SIMPLIFICATIONS['S1']`）
NON_EXACT = mt.coalesced_layout((128, 128), 4, order=[1, 0])
chk.A(NON_EXACT.spt[0] * NON_EXACT.tpw[0] * NON_EXACT.wpc[0] == 128
      and NON_EXACT.spt[1] * NON_EXACT.tpw[1] * NON_EXACT.wpc[1] == 128,
      "A.8 本模型构造出来的每个布局都满足 **spt×tpw×wpc == shape**（精确切分）——"
      "而真实 Triton 的 layout **可以覆盖不足**（官方第一个例子 BlockedLayout([1,1],[1,32],[1,4],[1,0]) "
      "在 128×128 上只覆盖 1×128，要沿第 0 维复制）",
      "本模型覆盖的是真实 layout 空间的**一个子集**；这不改变归约的结论（见 `04`），但必须写明")


# ---------------------------------------------------------------------------
# 2. 泄漏 L1：同一个 tl.sum(x, axis=1)，两种布局，2 barrier vs 0
# ---------------------------------------------------------------------------

C.rule("2 泄漏 L1：同一行 `tl.sum(x, axis=1)`，代价随布局从 2 barrier 变成 0")

pL = mt.ReducePlan(L, 1)
pR = mt.ReducePlan(Lrow, 1)
print("在 L 上归约 axis=1   ：%r" % (pL,))
print("   线程内 %d ops ｜ 蝶形 %d 步 / %d ops ｜ 跨 warp %d ops / %dB / %d barrier"
      % (pL.phase1_ops, pL.warp_steps, pL.phase2_ops,
         pL.phase3_ops, pL.phase3_smem_bytes, pL.phase3_barriers))
print("在 Lrow 上归约 axis=1：%r" % (pR,))
print("   线程内 %d ops ｜ 蝶形 %d 步 / %d ops ｜ 跨 warp %d ops / %dB / %d barrier"
      % (pR.phase1_ops, pR.warp_steps, pR.phase2_ops,
         pR.phase3_ops, pR.phase3_smem_bytes, pR.phase3_barriers))
print("广播回代价：L 上 %r ｜ Lrow 上 %r" % (pL.broadcast_back(), pR.broadcast_back()))
chk.B(pL.barriers == 2 and pR.barriers == 0 and pL.smem_bytes > 0 and pR.smem_bytes == 0,
      "B.3 [128,128] tile、num_warps=4、**同一行源码** `reduce(axis=1)`："
      "在 L 上是「蝶形 + 跨 warp 经共享内存」（锚点 `G2`），在 Lrow 上"
      "「不需要任何线程间通信」",
      "L: %d ops / %dB / %d barrier ｜ Lrow: %d ops / %dB / %d barrier"
      % (pL.total_ops, pL.smem_bytes, pL.barriers, pR.total_ops, pR.smem_bytes, pR.barriers))
chk.B(pL.broadcast_back().barriers == 2 and pR.broadcast_back().free,
      "B.3b 归约后**广播回**也一样：L 上还要再经一次共享内存（2 barrier），"
      "Lrow 上值本来就在本线程里（0 字节、0 barrier）",
      "L: %r ｜ Lrow: %r" % (pL.broadcast_back(), pR.broadcast_back()))


# ---------------------------------------------------------------------------
# 3. 泄漏 L2：trans 免费，但它**不**让归约变便宜（负面结论）
# ---------------------------------------------------------------------------

C.rule("3 泄漏 L2：`trans` 是免费的，但它**不**让归约变便宜（负面结论，锚点 G5）")

g1 = [
    mt.Op("load", SH, name="x", buff="X", order=[1, 0]),
    mt.Op("reduce", (SH[1],), inputs=(0,), name="rowofsum", axis=1, fn="sum"),
]
g2 = [
    mt.Op("load", SH, name="x", buff="X", order=[1, 0]),
    mt.Op("trans", SH, inputs=(0,), name="xt", perm=[1, 0]),
    mt.Op("reduce", (SH[1],), inputs=(1,), name="colofsumT", axis=0, fn="sum"),
]
r1 = mt.layout_pass(g1, W)
r2 = mt.layout_pass(g2, W)
print("G1 = load -> reduce(axis=1)")
print("G2 = load -> trans -> reduce(axis=0)      ← 多一次 trans，源码看着更贵")
print("G1 事件：")
print(r1.report())
print("G2 事件：")
print(r2.report())
print("G1 合计：ops=%d  smem=%d B  barriers=%d" % (r1.ops, r1.smem_bytes, r1.barriers))
print("G2 合计：ops=%d  smem=%d B  barriers=%d" % (r2.ops, r2.smem_bytes, r2.barriers))

env1, env2 = {"X": X, "OUT": None}, {"X": X, "OUT": None}
v1 = mt.graph_execute(g1, env1)[1]
v2 = mt.graph_execute(g2, env2)[2]
same = bool(np.array_equal(v1, v2))
print("G1 的第 3 个行和 = %.1f ｜ G2 的第 3 个列和 = %.1f ｜ 逐位相同 = %s"
      % (float(v1[3]), float(v2[3]), same))
chk.A(same, "A.1 G1 与 G2 的值**逐位相同**（都是 X 的逐行和）——它们真的是同一份数学",
      "(128,) 个输出逐位相等 = %s" % same)

Lp = L.permute([1, 0])
print("L.permute([1,0]) = %r" % (Lp,))
chk.B(Lp.spt == Lrow.spt and Lp.tpw == Lrow.tpw and Lp.wpc == Lrow.wpc
      and Lp.order == Lrow.order,
      "B.4 把 L 转置得到的布局，**参数逐项等于 Lrow**（转置只是重新贴标签，不搬数据）",
      "两者参数逐项相同：spt %r / tpw %r / wpc %r / order %r"
      % (Lp.spt, Lp.tpw, Lp.wpc, Lp.order))

print("但是：`reduce(axis=0)` 在 Lp 上的代价 = %r" % (mt.ReducePlan(Lp, 0),))
print("      与 `reduce(axis=1)` 在 L 上的代价 = %r" % (pL,))
chk.B(r2.barriers == r1.barriers and r2.smem_bytes == r1.smem_bytes,
      "B.5 **负面结论**：多写的那个 `trans` **一点没省**。转置改变的只是「跨线程摊开的是哪一条轴」"
      "（`axis_threads(0)` 与 `axis_threads(1)` 都是 128），**不改变**「跨线程摊开了多少」。"
      "⇒ `trans` 免费 ≠ 归约变便宜（与锚点 `G5`「先转成高效布局再做归约通常更贵」**方向一致，但不同源**："
      "本库这条是从两个数推出来的，`G5` 是官方的一句话）",
      "barriers %d vs %d，smem %d B vs %d B，ops %d vs %d"
      % (r1.barriers, r2.barriers, r1.smem_bytes, r2.smem_bytes, r1.ops, r2.ops))
print("  → 结论：想省掉跨 warp 相位，唯一的办法是让**被归约的那条轴**在跨 warp 方向上是 1，"
      "而那由 **load 的 order** 决定（见 §4），加 trans 改不了它。")


# ---------------------------------------------------------------------------
# 4. 泄漏 L3（最像「无害」的一处）：两次 load 的 order 不同
# ---------------------------------------------------------------------------

C.rule("4 泄漏 L3：两次 load 只差一个 order，代价 0 -> 2 barrier")

G_cheap = [
    mt.Op("load", SH, name="a", buff="X", order=[1, 0]),
    mt.Op("load", SH, name="b", buff="Y", order=[1, 0]),
    mt.Op("elementwise", SH, inputs=(0, 1), name="a*b", fn=np.multiply),
]
G_dear = [
    mt.Op("load", SH, name="a", buff="X", order=[1, 0]),
    mt.Op("load", SH, name="b", buff="Y", order=[0, 1]),      # ← 只改了这一个关键字
    mt.Op("elementwise", SH, inputs=(0, 1), name="a*b", fn=np.multiply),
]
rc = mt.layout_pass(G_cheap, W)
rd = mt.layout_pass(G_dear, W)
print("G_cheap：a=load(order=[1,0])，b=load(order=[1,0])")
print("G_dear ：a=load(order=[1,0])，b=load(order=[0,1])   ← 唯一的差别")
print("G_cheap 事件：")
print(rc.report())
print("G_dear  事件：")
print(rd.report())
print("G_cheap 合计 barriers=%d smem=%d B ｜ G_dear 合计 barriers=%d smem=%d B"
      % (rc.barriers, rc.smem_bytes, rd.barriers, rd.smem_bytes))

vc = mt.graph_execute(G_cheap, {"X": X, "Y": Y})[2]
vd = mt.graph_execute(G_dear, {"X": X, "Y": Y})[2]
same_ab = bool(np.array_equal(vc, vd))
chk.A(same_ab, "A.2 两张图的输出**逐位相同**（%d 个元素）——差别纯粹在布局，不在数学"
      % vc.size, "逐位相等 = %s" % same_ab)
chk.B(rc.barriers == 0 and rc.smem_bytes == 0 and rd.barriers == 2 and rd.smem_bytes > 0,
      "B.6 一个**离得很远的关键字参数**（第二次 load 的 order）把逐元素乘法从 0 转换"
      "变成 1 次共享内存往返（%d B）—— 而源码里 `a * b` 那一行**完全看不出差别**" % rd.smem_bytes,
      "barriers %d -> %d ｜ smem %d B -> %d B"
      % (rc.barriers, rd.barriers, rc.smem_bytes, rd.smem_bytes))


# ---------------------------------------------------------------------------
# 4b. 泄漏 L3b：合并访存的要求（锚点 G4）
# ---------------------------------------------------------------------------

C.rule("4b 泄漏 L3b：合并访存的要求 —— 把转置后的值写回要付一次转换")

G_store_ok = [
    mt.Op("load", SH, name="x", buff="X", order=[1, 0]),
    mt.Op("store", SH, inputs=(0,), name="store_x", buff="OUT", coalesced_dim=1),
]
G_store_t = [
    mt.Op("load", SH, name="x", buff="X", order=[1, 0]),
    mt.Op("trans", SH, inputs=(0,), name="xt", perm=[1, 0]),
    mt.Op("store", SH, inputs=(1,), name="store_xt", buff="OUT", coalesced_dim=1),
]
rs_ok = mt.layout_pass(G_store_ok, W)
rs_t = mt.layout_pass(G_store_t, W)
print("G_store_ok：load -> store（要求沿维 1 合并）")
print("G_store_t ：load -> trans -> store（同样要求沿维 1 合并）")
print("G_store_ok 事件：")
print(rs_ok.report() or "    （无事件）")
print("G_store_t 事件：")
print(rs_t.report() or "    （无事件）")
print("G_store_ok 合计 barriers=%d smem=%d B ｜ G_store_t 合计 barriers=%d smem=%d B"
      % (rs_ok.barriers, rs_ok.smem_bytes, rs_t.barriers, rs_t.smem_bytes))
print("  → store 的布局要求：`contiguous_along(coalesced_dim)`；trans 之后是 %s"
      % ("否" if not rs_t.layouts[1].contiguous_along(1) else "是"))
chk.B(rs_ok.barriers == 0 and rs_t.barriers == 2 and rs_t.smem_bytes > 0,
      "B.7b 把「转过置的结果」写回一个要求合并访存的位置，必须付 %d 字节、2 barrier 的转换 ——"
      "要么付它，要么放弃合并访存（本模型里是**拒绝**，见 C.1c）。"
      "这一条锚的是 `G4` 的后半句「内维不再连续 ⇒ 没有合并访存」" % rs_t.smem_bytes,
      "barriers %d -> %d ｜ smem %d B -> %d B"
      % (rs_ok.barriers, rs_t.barriers, rs_ok.smem_bytes, rs_t.smem_bytes))
chk.expect_raises(lambda: mt.layout_pass(G_store_t, W, allow_convert=False),
                  mt.LayoutConflict,
                  "C.1c 对同一张图声明「不许插转换」→ **拒绝**（非合并访存不能静默放过；"
                  "锚点 `G4` 的可证伪形式）")


# ---------------------------------------------------------------------------
# 5. 泄漏 L4：K 次归约 —— 一次显式转换 vs 什么都不做，两个交叉点
# ---------------------------------------------------------------------------

C.rule("5 泄漏 L4：K 次归约的循环，一次显式转换 vs 不转换（两个交叉点）")


def graph_loop(K, convert_once):
    ops = [mt.Op("load", SH, name="x", buff="X", order=[1, 0])]
    src = 0
    if convert_once:
        ops.append(mt.Op("convert", SH, inputs=(0,), name="x->Lrow", target=Lrow))
        src = 1
    for k in range(K):
        ops.append(mt.Op("reduce", (SH[1],), inputs=(src,), name="rowsum%d" % k, axis=1, fn="sum"))
    return ops


print("  K   不转换(barrier / smem B)      转换一次(barrier / smem B)")
rows = []
for K in [1, 2, 4, 8, 16, 32, 64]:
    ra = mt.layout_pass(graph_loop(K, False), W)
    rb = mt.layout_pass(graph_loop(K, True), W)
    rows.append((K, ra.barriers, ra.smem_bytes, rb.barriers, rb.smem_bytes))
    print("  %-3d  %-4d / %-8d            %-4d / %-8d"
          % (K, ra.barriers, ra.smem_bytes, rb.barriers, rb.smem_bytes))

k_tie_b = next((K for K, _, a, _, b in rows if b == a), None)
k_tie_r = next((K for K, a, _, b, _ in rows if b == a), None)
k_bar = next(K for K, a, _, b, _ in rows if b < a)
k_byte = next(K for K, _, a, _, b in rows if b < a)
print("按 barrier 口径：K=%s 时两者相等，K=%d 起转换版**严格更省**" % (k_tie_r, k_bar))
print("按字节口径  ：K=%s 时两者相等，K=%d 起转换版**严格更省**" % (k_tie_b, k_byte))
chk.B(k_bar == 2 and k_byte == 64,
      "B.10 **同一次转换，按两个代价口径给出的交叉点不同**：barrier 口径 K*=2，字节口径 K*=64"
      "——「这次转换值不值」在模型里**根本不是一个数**"
      "（★ 这两个数是 `specs.MODEL_OWN_RULES['R5']` 列的自造结论）",
      "barrier：每次归约 2 barrier vs 转换 2 barrier ⇒ K≥2 即打平后更省；"
      "字节：每次归约 4096 B vs 转换 131072 B ⇒ K=32 正好相等（4096×32=131072），K=64 才严格更省")

ga = graph_loop(4, False)
gb = graph_loop(4, True)
va = [mt.graph_execute(ga, {"X": X})[i] for i in (1, 2, 3, 4)]
vb = [mt.graph_execute(gb, {"X": X})[i] for i in (2, 3, 4, 5)]
allsame = all(bool(np.array_equal(a, b)) for a, b in zip(va, vb))
chk.A(allsame, "A.3 K=4 时两张图算出的 4 个行和向量逐位相同——**同一份数学**",
      "4 个 (128,) 向量逐位相等 = %s" % allsame)


# ---------------------------------------------------------------------------
# 6. 代价模型的非退化性：存在**真的免费**的转换（锚点 G3）
# ---------------------------------------------------------------------------

C.rule("6 代价模型不能退化成「转换一律昂贵」：存在真的免费的转换")

fs, fd = mt.free_conversion_demo()
own_eq = bool(np.array_equal(fs.owners(), fd.owners()))
reg_diff = [idx for idx in itertools.product(range(64), range(2))
            if fs.register_index(idx) != fd.register_index(idx)]
cfree = mt.insert_conversion([64, 2], fs, fd)
print("免费对：shape=%r" % (fs.shape,))
print("  属主映射逐位相同 = %s；寄存器编号不同的下标个数 = %d / %d"
      % (own_eq, len(reg_diff), 64 * 2))
print("  转换判定：%r" % (cfree,))
chk.A(own_eq and len(reg_diff) > 0,
      "A.4 这一对布局的**属主逐位相同、寄存器编号不同**——所以「转换」只是线程内寄存器重排",
      "不同寄存器编号的下标 = %d 个" % len(reg_diff))
chk.B(cfree.free and cfree.smem_bytes == 0 and cfree.barriers == 0,
      "B.8 它被判为免费：0 字节共享内存、0 barrier（口径锚点是 `G3` 的 "
      "「at most only require reordering registers within a thread (which is free)」；"
      "★ 本模型的免费判据 =「逐元素属主相同」，那是**本库自定的**）",
      "%s ｜ %dB ｜ %d barrier" % (cfree.kind, cfree.smem_bytes, cfree.barriers))

cpaid = mt.insert_conversion(SH, L, Lrow)
print("昂贵对：L -> Lrow：%r" % (cpaid,))
chk.B(not cpaid.free and cpaid.smem_bytes == 2 * SH[0] * SH[1] * 4 and cpaid.barriers == 2,
      "B.9 换一对布局（L -> Lrow）就不免费了：写一遍 + 读一遍，%d 字节、2 barrier"
      "（公式是本模型自定的，`specs.MODEL_OWN_RULES['R3']`）" % cpaid.smem_bytes,
      "字节数 = 2 × %d 个元素 × 4 B = %d" % (SH[0] * SH[1], cpaid.smem_bytes))

chk.A(mt.insert_conversion(SH, L, L) is None,
      "A.9 两个参数**逐项相同**的布局之间，连转换节点都不建（返回 `None`）——"
      "这条把「恒等转换」与「免费转换」**分开**：前者不产生事件，后者产生一个 0 代价事件",
      "insert_conversion(L, L) -> None")


# ---------------------------------------------------------------------------
# 7. C 类：判定必须可证伪
# ---------------------------------------------------------------------------

C.rule("7 C 类：把判定改错，断言必须失败")

chk.expect_raises(lambda: mt.layout_pass(G_dear, W, allow_convert=False),
                  mt.LayoutConflict,
                  "C.1 对 G_dear 声明「不许插转换」→ pass 拒绝（它确实需要转换）")
chk.expect_ok(lambda: mt.layout_pass(G_cheap, W, allow_convert=False),
              "C.1b 对照：G_cheap 在同一个「不许插转换」的声明下通过（它确实不需要）",
              "G_cheap barriers = %d" % rc.barriers)
n_conv_ok = len(mt.layout_pass(G_dear, W).conversions)
n_conv_no = 0
try:
    mt.layout_pass(G_dear, W, allow_convert=False)
except mt.LayoutConflict:
    n_conv_no = 1
chk.C(n_conv_ok == 1 and n_conv_no == 1,
      "C.1d 同**一张图**的两个分支必须不同：allow_convert=True 产出 %d 个转换，"
      "allow_convert=False 抛错 —— 两条分支真的分开了" % n_conv_ok,
      "转换节点数 = %d vs 拒绝 = %d" % (n_conv_ok, n_conv_no))

_orig_is_free = mt.Conversion._is_free
mt.Conversion._is_free = lambda self: True
try:
    bad_paid = mt.Conversion(SH, L, Lrow, "f32")
    broke = (bad_paid.smem_bytes == 0) != (cpaid.smem_bytes == 0)
finally:
    mt.Conversion._is_free = _orig_is_free
chk.C(broke,
      "C.2 把免费判据改成「一律免费」，昂贵转换的字节数从 %d 掉到 0 —— "
      "B.9 的判据有判别力（改错就失配）" % cpaid.smem_bytes,
      "改错后 L->Lrow 的 smem_bytes = %d" % bad_paid.smem_bytes)

mt.Conversion._is_free = lambda self: False
try:
    bad_free = mt.Conversion([64, 2], fs, fd, "f32")
    broke2 = (bad_free.smem_bytes != 0) != (cfree.smem_bytes != 0)
finally:
    mt.Conversion._is_free = _orig_is_free
chk.C(broke2,
      "C.2b 把免费判据改成「一律昂贵」，免费转换的字节数从 0 涨到 %d —— "
      "B.8 的判据**两个方向**都有判别力（锚点 `G3` 的存在性因此可证伪）" % bad_free.smem_bytes,
      "改错后 shape=(64,2) 那一对的 smem_bytes = %d" % bad_free.smem_bytes)


class LyingLayout:
    """在 wpc 上说谎：全说成 1，同时改 spt 把切分恒等式补回来（所以仍是个合法布局）。

    这样 `ReducePlan` 会认为「没有跨 warp 的摊开」，跨 warp 相位就消失。
    """

    def __init__(self, real):
        self._r = real
        self.shape = real.shape
        self.rank = real.rank
        self.wpc = tuple(1 for _ in range(real.rank))                     # ← 谎话
        self.tpw = real.tpw
        self.order = real.order
        self.spt = tuple(real.shape[d] // (real.tpw[d] * 1) for d in range(real.rank))
        self.dtype = real.dtype
        self.name = "lying"

    @property
    def numel(self):
        return self._r.numel

    @property
    def num_warps(self):
        return 1

    @property
    def n_threads(self):
        return 32


lying = LyingLayout(L)
fake = mt.ReducePlan(lying, 1)
chk.C(fake.barriers == 0 and pL.barriers == 2,
      "C.3 把布局里的 warpsPerCTA 谎报成全 1，跨 warp 相位就从 2 barrier 掉到 0 —— "
      "B.3 的数是从布局参数**真算出来的**，不是一句声明",
      "谎报后 barriers = %d，真布局 %d" % (fake.barriers, pL.barriers))

chk.expect_raises(lambda: L.squeeze(1), mt.LayoutError,
                  "C.4 对有跨线程摊开的维调 squeeze → 拒绝（归约结果必须用 SliceLayout 表示）")
chk.expect_ok(lambda: Lrow.squeeze(1),
              "C.4b 对照：Lrow 的 axis=1 只归一个线程所有 → squeeze 合法",
              "得到 %r" % (Lrow.squeeze(1),))

# C.5：★ 本模型的**已记录简化**是可证伪的 —— 非精确切分直接抛错
chk.expect_raises(lambda: mt.Layout([128, 128], [1, 1], [32, 1], [1, 4], order=[1, 0]),
                  mt.LayoutError,
                  "C.5 非精确切分（spt×tpw×wpc = 1×32×1 = 32 ≠ 128）**直接抛 LayoutError** ——"
                  "这条简化是本模型自定的（`S1`），真实 Triton 允许这种布局（要沿第 0 维复制）")


# ---------------------------------------------------------------------------
# 8. 被引为「官方规定」的四条 + 本模型自造的 5 条
# ---------------------------------------------------------------------------

C.rule("8 官方锚点（逐字）与本模型自造的 5 条规则")
print_anchors()

# A.5：★ 锚点 G5 的原词 `reeduction` 是官方**拼写错误**，引用时**不得修好**
g5 = specs.ANCHORS["G5"]["quote"]
chk.A(g5.count("reeduction") == 1 and "then perform the reeduction" in g5,
      "A.5 锚点 `G5` 的引文里 `reeduction` 出现**恰好 1 次**，且不以 `reduction` 的形式出现"
      "——原文是官方页面上的拼写错误，**逐字保留**",
      "`reeduction` 计数 = %d；把它「修好」会让这条断言与在线再抽取同时失败"
      % g5.count("reeduction"))

# A.6：`must_contain` 与本程序负责的四条锚点逐条自洽（引文与断言作用域的第一道检查）
bad_mc = [(aid, s[:40]) for aid in COVERED_ANCHORS for s in specs.ANCHORS[aid]["must_contain"]
          if s not in specs.ANCHORS[aid]["quote"]]
chk.A(not bad_mc, "A.6 本程序负责的四条锚点，每一条 `must_contain` 都是它自己 `quote` 的逐字子串",
      "不合规 = %r" % (bad_mc[:5],))

# A.7：★ 本模型自造的 5 条规则 —— **打印给读者看**，并把「可证伪的那部分」说清楚
#
#   ★★ 这里必须克制（P5「断言形式正确但不可证伪」）：**不要把「常量 = 它自己」写成断言**。
#     第一版写的是「`specs.MODEL_OWN_RULES[k]['rule']` 出现在从**同一个常量**拼出来的
#     `printed` 列表里」—— 那是**恒真**的，变异测试把它如实报成了漏网（5 条）。
#     ⇒ 现在只断言**真正可判定的部分**：
#       ① 条数是 5；② 5 条**两两不同**（不是复制粘贴出来的同一个句子）；
#       ③ 每条都带 `covered_by` 且**标签真实存在**（由 `01_block_model.py` 的 `C.8` 统一校验）；
#       ④ 这 5 条**没有一条**是官方引文的字面量（不得把自造规则冒充成官方措辞）。
#     ★ `rule` / `basis` 的**散文正文**本身**不是行为判据的载体** —— 这一点如实写在这里，
#       并如实出现在 `make mutate` 的漏网清单里（不粉饰）。
print()
print("  ★ 本模型**没有**声称下列任意一条是 Triton 的规定（**是本模型自造的**）：")
for k in sorted(specs.MODEL_OWN_RULES):
    print("      [%s] %s" % (k, specs.MODEL_OWN_RULES[k]["rule"]))
rules = [specs.MODEL_OWN_RULES[k]["rule"] for k in sorted(specs.MODEL_OWN_RULES)]
official = " ".join([a["quote"] for a in specs.ANCHORS.values()] +
                    [a["quote"] for a in specs.LOAD_QUOTES.values()])
fake = [r for r in rules if r in official]
chk.A(len(rules) == 5 and len(set(rules)) == 5 and not fake
      and all(specs.MODEL_OWN_RULES[k]["covered_by"] for k in specs.MODEL_OWN_RULES),
      "A.7 本模型自造的规则恰好 **5 条**、**两两不同**、**每条都带 `covered_by`**，"
      "且**没有一条**是官方引文的字面量 —— 这是「这些规则是自造的」这句话**可判定的那一半**"
      "（`rule` 的散文正文不是判据载体，如实列在 `make mutate` 的漏网里）",
      "条数 %d ｜ 去重后 %d ｜ 冒充官方引文的 = %r"
      % (len(rules), len(set(rules)), [f[:30] for f in fake]))

print()
print("  ★ 本模型**已记录的简化**（写出来是因为不写就是「比实际更强的声明」）：")
for k in sorted(specs.MODEL_SIMPLIFICATIONS):
    s = specs.MODEL_SIMPLIFICATIONS[k]
    print("      [%s] %s" % (k, s["item"]))
chk.A(len(specs.MODEL_SIMPLIFICATIONS) == 3,
      "A.10 三条已记录的简化（精确切分 / 越界抛错 / 毒值）都在 `specs` 里，"
      "并由 `A.8`、`C.1`、`C.3` 各自锚住",
      "简化条目 = %d" % len(specs.MODEL_SIMPLIFICATIONS))

# B.11：★ F4 的离线结论 —— 主 tutorial 里这三样东西是 **0** 次
vf = specs.WORD_FREQ[specs.VECTOR_ADD_PAGE]
gf = specs.WORD_FREQ[specs.GLUON_PAGE]
print()
print("  ★ 词频表（口径见 `specs.py` 模块头；在线可用 `make anchors` 复算）：")
for page in (specs.VECTOR_ADD_PAGE, specs.GLUON_PAGE):
    f = specs.WORD_FREQ[page]
    print("      %-52s coalesc=%d  Layout conversions=%d  warp=%d/%d  len=%d"
          % (page, f["coalesc_cs"], f["layout_conversions_cs"], f["warp_cs"], f["warp_ci"],
             f["stripped_len"]))
chk.B(vf["coalesc_cs"] == 0 and vf["layout_conversions_cs"] == 0 and vf["warp_cs"] == 0
      and vf["warp_ci"] == 0 and gf["coalesc_cs"] > 0 and gf["layout_conversions_cs"] > 0
      and gf["warp_ci"] > 0,
      "B.11 **主 tutorial（01-vector-add）里 `coalesc` / `Layout conversions` / `warp` 都是 0 次**，"
      "而 Gluon「Tensor Layouts」那一页有 —— ⇒ 要引「转换昂贵」与「归约跨 warp」这两条机制，"
      "**出处没有选择余地**",
      "vector-add 三个口径全 0；gluon coalesc=%d、Layout conversions=%d、warp(不敏感)=%d"
      % (gf["coalesc_cs"], gf["layout_conversions_cs"], gf["warp_ci"]))


# A.11/A.12：布局代数自身的结构性质（与代价无关，但它们是后面所有数的基础）
sl = L.slice_along(1)
live, dead = sl.live_threads(), sl.dead_threads()
print("L 沿 axis=1 归约后的 SliceLayout：持有结果的线程 %d 个（%d 个线程持有）"
      % (len(live), L.n_threads))
chk.A(len(live) == L.n_threads // L.axis_threads(1) and len(dead) == L.n_threads - len(live),
      "A.11 归约后的值只落在**少数线程**手里：%d 个线程存活、%d 个线程闲置"
      "（=%d 减去沿 axis 摊开的 %d 的商）—— 这就是「广播回不免费」的**根源**"
      % (len(live), len(dead), L.n_threads, L.axis_threads(1)),
      "live=%d dead=%d（合计 %d）" % (len(live), len(dead), L.n_threads))

Lperm = L.permute([1, 0])
chk.A(Lperm.shape == (SH[1], SH[0]) and sorted(Lperm.order) == [0, 1]
      and Lperm.spt == (L.spt[1], L.spt[0]),
      "A.12 `permute` 只重排维标签：形状变成 %r，order 仍是合法排列，spt 逐维跟着换 —— "
      "**没有数据被搬动**（这就是 `trans` 免费的原因）" % (Lperm.shape,),
      "permute([1,0]) 后 shape=%r order=%r spt=%r" % (Lperm.shape, Lperm.order, Lperm.spt))

chk.expect_raises(lambda: L.permute([1, 1]), mt.LayoutError,
                  "C.7 非法的 `perm`（[1,1]，不是 0..rank−1 的排列）→ 抛 LayoutError")
other_shape_layout = mt.Layout([64, 128], [1, 128], [32, 1], [2, 1], order=[0, 1])
chk.A(other_shape_layout.shape == (64, 128),
      "A.13 另造一个**合法**的异构布局（shape=%r，逐维 spt×tpw×wpc 仍等于 shape）"
      "—— 用来试「形状不同的两个布局能不能插转换」" % (other_shape_layout.shape,),
      "spt=%r tpw=%r wpc=%r" % (other_shape_layout.spt, other_shape_layout.tpw,
                               other_shape_layout.wpc))
chk.expect_raises(lambda: mt.insert_conversion((128, 128), L, other_shape_layout),
                  mt.LayoutConflict,
                  "C.8 两个**形状不同**的布局之间不能插转换 → LayoutConflict（不静默产出错计划）")

# B.12：代价的**记账自洽** —— PassResult 的合计等于它自己事件列表的逐项和
ev_sum = sum(getattr(e, "smem_bytes", 0) for _, e in rd.events)
chk.B(ev_sum == rd.smem_bytes == 131072,
      "B.12 `G_dear` 的合计共享内存 %d B **等于**它事件列表里逐项之和 %d B —— "
      "代价是**逐事件记账**出来的，不是一笔总账" % (rd.smem_bytes, ev_sum),
      "事件 %d 条：%s" % (len(rd.events), [n for n, _ in rd.events]))

sys.exit(chk.summary())
