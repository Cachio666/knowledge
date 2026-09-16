"""02 融合与调度：判据链、门槛默认值，以及「融合不是越多越好」。

对应文档：60-compilers/inductor/02-fusion-scheduling.md

★ 本程序跑的是**本仓库的迷你融合调度器**（minifuse.py），不是 Inductor 的
  `Scheduler` / `V.choices`。判据与默认值逐条抄自固定版本的 torch 源码
  （见 specs.py，每条带来源），**实现形态是为教学重写的**。
  输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。

用法：python3 02_fusion_groups.py
"""

import numpy as np

import minifuse as F
import minigraph as M
import specs as SPECS
from minigraph import Checks, Dep, Graph

checks = Checks()


def capture(fn, inputs):
    d = M.MiniDynamo()
    d.register(fn)
    return d.capture(fn, inputs).subgraphs[0]


# ---- 地形一：垂直融合（生产者 → 消费者） -----------------------------------------------------


def t_vertical(x, w):
    a = x * 2.0
    b = a + w
    c = b.relu()
    return c


def ref_vertical(x, w):
    return np.maximum(x * 2.0 + w, 0.0)


def t_mismatch(x):
    """把生产者**转置**后再消费：读索引 (@1,@0) ≠ 写索引 (@0,@1)。"""
    a = x * 2.0
    t = a.transpose()
    return t + 1.0


def ref_mismatch(x):
    return (x * 2.0).T + 1.0


# ---- 地形二：水平融合（消费者 ↔ 消费者）——阈值以**字节**计 -----------------------------------------


def t_horiz(x):
    a = x * 2.0
    b = x + 1.0
    return [a, b]


def ref_horiz(x):
    return [x * 2.0, x + 1.0]


X2 = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]], dtype=np.float32)
W3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

M.rule("1. 判据与默认值（逐条来自固定版本源码，见 specs.py）")
print("%-36s %-18s %s" % ("项", "默认值", "核的是什么载体"))
print("-" * 92)
_loc = {
    "aggressive_fusion": "config.py（2.6.0:437 / 2.7.1:500 / 2.13.0:916）",
    "score_fusion_memory_threshold": "config.py（2.6.0:457 / 2.7.1:520 / 2.13.0:953）",
    "max_fusion_size": "config.py（2.6.0:468 / 2.7.1:531 / 2.13.0:964）",
}
for k, v in SPECS.INDUCTOR_DEFAULTS.items():
    print("%-36s %-18s %s" % (k, v, _loc[k]))
print("%-36s %-18s %s" % ("max_fusion_unique_io_buffers",
                          SPECS.MAX_FUSION_UNIQUE_IO_BUFFERS["2.13.0"],
                          "2.13.0 config.py:975（2.6.0/2.7.1 无此名）"))
print("%-36s %-18s %s" % ("mix_order_reduction_max_reads",
                          SPECS.MIX_ORDER_CONFIG["mix_order_reduction_max_reads"],
                          "2.13.0 config.py:2137"))
print("%-36s %-18s %s" % ("spill_threshold", "HIP 32 / 其他 16",
                          "2.13.0 config.py:2048"))
print("%-36s %-18s %s" % ("max_horizontal_fusion_size",
                          SPECS.MAX_HORIZONTAL_FUSION_SIZE["value"],
                          SPECS.MAX_HORIZONTAL_FUSION_SIZE["scope"]))
print()
print("★ 分数的单位：%s" % SPECS.SCORE_UNIT)
print("★ %s" % SPECS.AGGRESSIVE_FUSION_CONDITION)
print("★ 垂直融合：%s" % SPECS.VERTICAL_INDEX_RULE)
print("  源码注释给的反例：%s" % " · ".join(SPECS.VERTICAL_INDEX_EXAMPLES))
print()
print("拒绝融合的理由字符串（逐条抄自源码，顺序即源码顺序）：")
for why, loc in SPECS.REJECT_REASONS:
    print("   %-62s %s" % ('"%s"' % why, loc))

# ---- 地形一 ---------------------------------------------------------------------

M.rule("2. 垂直融合：索引必须逐字匹配")

g_v = capture(t_vertical, [X2, W3])
print("图：%s" % [(n.buf, n.op, [str(d) for d in n.reads]) for n in g_v.nodes])
plan_v = F.schedule(g_v)
print("调度结果：%s" % [[n.op for n in g] for g in plan_v.groups])
for d in plan_v.decisions:
    print("   [%s] %-8s %s" % ("融" if d.ok else "拒", d.kind, d.why))

g_m = capture(t_mismatch, [X2])
print()
print("转置地形：%s" % [(n.buf, n.op, [str(d) for d in n.reads]) for n in g_m.nodes])
plan_m = F.schedule(g_m)
print("调度结果：%s" % [[n.op for n in g] for g in plan_m.groups])
for d in plan_m.decisions:
    print("   [%s] %-8s %s" % ("融" if d.ok else "拒", d.kind, d.why))

checks.C(any(not d.ok and "memory deps did not match" in d.why
             for d in plan_m.decisions),
         "C：转置消费的读索引与生产者的写索引不匹配 → 真的被拒（理由串是 memory deps did not match）",
         [d.why for d in plan_m.decisions if not d.ok][0])
checks.expect_ok(lambda: all(d.ok for d in plan_v.decisions),
                 "对照：索引逐字匹配的点式链全部融合（换掉不匹配的读法就放行）",
                 "%d 条决策全部通过" % len(plan_v.decisions))

M.rule("2b. B 代价模型：全融 vs 不融，字节差在哪")

plan_all = [[n] for n in g_v.nodes if n.kind not in ("placeholder", "effect")]
t_fused = F.plan_traffic(g_v, plan_v.groups)
t_split = F.plan_traffic(g_v, plan_all)
print("点式链 `x*2 → +w → relu`（x 是 2×3 float32）：")
print("   全融 ：%d 个核，读 %d + 写 %d = %d 字节，物化 %s"
      % (t_fused["kernels"], t_fused["read_bytes"], t_fused["write_bytes"],
         t_fused["total"], t_fused["materialized"]))
print("   不融 ：%d 个核，读 %d + 写 %d = %d 字节，物化 %s"
      % (t_split["kernels"], t_split["read_bytes"], t_split["write_bytes"],
         t_split["total"], t_split["materialized"]))
print("   本机代价模型的结论：融合把字节流量从 %d 降到 %d（省下的是中间结果"
      "那几次往返）" % (t_split["total"], t_fused["total"]))
checks.B(t_split["total"] == 156 and t_fused["total"] == 60,
         "B：文档 §0 引用的两个数字（全融 60 / 不融 156 字节）都出自本程序",
         "全融 %d / 不融 %d" % (t_fused["total"], t_split["total"]))
checks.B(len(t_split["materialized"]) == 3 and len(t_fused["materialized"]) == 1,
         "B：不融时物化 3 个缓冲、全融时只剩 1 个（图输出）",
         "%d vs %d" % (len(t_split["materialized"]),
                       len(t_fused["materialized"])))

# ---- 地形二：水平融合的字节门槛 -----------------------------------------------------------

M.rule("3. 水平融合：门槛是**字节**，不是「有没有共享」")

for n_el in (2, 4):
    x = np.arange(n_el, dtype=np.float32) + 1.0
    g_h = capture(t_horiz, [x])
    a, b = g_h.by_buf("buf0"), g_h.by_buf("buf1")
    score = F.shared_data_score(g_h, a, b)
    plan_h = F.schedule(g_h)
    dec = [d for d in plan_h.decisions if d.kind == "horizontal"]
    print("x 有 %d 个 float32（%d 字节）：shared_data_score=%d 阈值=%d → %s"
          % (n_el, x.nbytes, score, F.DEFAULTS["score_fusion_memory_threshold"],
             "融" if (dec and dec[0].ok) else "拒"))
    print("    工程结论：%s" % (dec[0].why if dec else "（没有横向候选）"))
    if n_el == 2:
        checks.C(score < F.DEFAULTS["score_fusion_memory_threshold"]
                 and dec and not dec[0].ok,
                 "C：共享 8 字节 < 阈值 10 → 水平融合被拒（分数以字节计）",
                 "score=%d" % score)
    else:
        checks.expect_ok(lambda: score >= F.DEFAULTS["score_fusion_memory_threshold"]
                         and dec and dec[0].ok,
                         "对照：把共享区域放大到 16 字节 ≥ 10 → 同一个判据放行",
                         "score=%d" % score)

# ---- A 类：上面这些地形本身必须算对 --------------------------------------------------------

M.rule("3b. A 算法正确性：图解释器 vs 纯 numpy 参考实现")

g_h4 = capture(t_horiz, [np.arange(4, dtype=np.float32) + 1.0])
for label, g, ref, ins in (("垂直地形", g_v, ref_vertical, [X2, W3]),
                           ("转置地形", g_m, ref_mismatch, [X2]),
                           ("水平地形", g_h4, ref_horiz,
                            [np.arange(4, dtype=np.float32) + 1.0])):
    env = {n.buf: n.value for n in g.nodes if n.kind == "placeholder"}
    got_env = M.eval_graph(g, dict(env))
    want = ref(*[np.asarray(a) for a in ins])
    want = want if isinstance(want, list) else [want]      # 单输出地形包成列表
    errs = [M.max_abs_err(got_env[o.buf], w) for o, w in zip(g.outputs, want)]
    print("%s：图解释器 vs 参考实现，最大绝对误差 %.3e" % (label, max(errs)))
    checks.A(max(errs) == 0.0, "A：%s 的解释结果 == 纯 numpy 参考实现" % label,
             "最大绝对误差 %.3e" % max(errs))

# ---- reduction ↔ reduction：不是通用能力 --------------------------------------------

M.rule("4. reduction ↔ reduction：只有 mix-order 特例放行")


def mix_order_pair(nrow, ncol, n_extra_reads=0, sibling=True):
    """**构造规则先于结果冻结**：nrow / ncol / 额外读缓冲数都是显式入参。

    两个归约都读同一个 (nrow, ncol) 张量，一个归约最后一维（循环序 PR）、
    一个归约第一维（循环序 RP）——这正是 `has_mix_reduction_orders` 要的镜像对。
    """
    g = Graph("mix_%d_%d_%d_%s" % (nrow, ncol, n_extra_reads, sibling))
    x = g.placeholder((nrow, ncol), "arg")
    common = Dep(x.buf, ("@0", "@1"))
    # 额外读缓冲**逐个不同**：一对候选融合的唯一读缓冲数 = 1 + n_extra_reads
    a_rd, b_rd = [common], [common]
    for k in range(n_extra_reads):
        extra = g.placeholder((16,), "extra")
        (a_rd if k % 2 == 0 else b_rd).append(Dep(extra.buf, ("@0", "@1")))
    a = g.add("sum", [x], (nrow,), "reduction", reads=a_rd, rloops=1,
              loop_order="PR", attrs={"axis": (1,), "red_shape": (ncol,)})
    a.writes = Dep(a.buf, tuple("@0" for _ in a.shape))
    if sibling:
        b = g.add("sum", [x], (ncol,), "reduction", reads=b_rd, rloops=1,
                  loop_order="RP", attrs={"axis": (0,), "red_shape": (nrow,)})
    else:
        b = g.add("sum", [a], (ncol,), "reduction",
                  reads=[Dep(a.buf, ("@0",))], rloops=1, loop_order="RP",
                  attrs={"axis": (0,), "red_shape": (nrow,)})
    b.writes = Dep(b.buf, tuple("@0" for _ in b.shape))
    return g, a, b


cfg = F.FuseConfig()
print("mix-order 的全部条件（来源：%s）" % SPECS.MIX_ORDER_SIZE_THRES_LOC)
for k, v in SPECS.MIX_ORDER_CONDITIONS.items():
    print("   %-24s %s" % (k, v))
print("%s" % SPECS.MIX_ORDER_NROW_LOC)
print("%s" % SPECS.MIX_ORDER_NCOL_LOC)
print()

MIX_CASES = [
    # (nrow, ncol, 额外读缓冲数, 是否兄弟关系, 期望, 说明)
    (32, 64, 0, True, False, "镜像 + 兄弟 + 有公共读，但 nrow*ncol=2048 < 5*2**20"),
    (4096, 1280, 0, True, True, "刚好越过 nrow*ncol >= 5*2**20 与 nrow >= 4096"),
    (2048, 4096, 0, True, False, "nrow=2048 < 4096"),
    (4096, 1280, 9, True, True, "唯一读缓冲 10 个 == mix_order_reduction_max_reads=10（边界内）"),
    (4096, 1280, 10, True, False, "唯一读缓冲 11 个 > mix_order_reduction_max_reads=10"),
    (4096, 1280, 0, False, False, "改成生产者-消费者（祖先关系）→ 出局"),
]
for nrow, ncol, extra, sib, want, note in MIX_CASES:
    g, a, b = mix_order_pair(nrow, ncol, extra, sib)
    ok, why = F.mix_order_can_fuse(g, a, b, cfg)
    print("%-46s → %s   %s" % (note, "放行" if ok else "拒", why))
    checks.B(ok == want, "mix-order 判定：%s == %s" % (note, want),
             "本机 %s（%s）" % (ok, why))

# 通用路径对普通的 reduction ↔ reduction 必须拒
g_ord = Graph("two_reductions")
x = g_ord.placeholder((8, 8), "arg")
r1 = g_ord.add("sum", [x], (8,), "reduction", reads=[Dep(x.buf, ("@0", "@1"))],
               rloops=1, loop_order="PR", attrs={"axis": (1,), "red_shape": (8,)})
r1.writes = Dep(r1.buf, ("@0",))
r2 = g_ord.add("sum", [x], (8,), "reduction", reads=[Dep(x.buf, ("@0", "@1"))],
               rloops=1, loop_order="PR", attrs={"axis": (1,), "red_shape": (8,)})
r2.writes = Dep(r2.buf, ("@0",))
ok, why = F.mix_order_can_fuse(g_ord, r1, r2, cfg)
checks.C(not ok, "C：循环序**不镜像**的两个归约 → 通用路径拒绝（reduction↔reduction 不是通用能力）",
         why)

# ---- C3：融合越界反而更慢——重算 vs 物化 ---------------------------------------------------

M.rule("5. C3 边界与反例：融合越界反而更慢（地形常量先于结果冻结）")

# 两个地形的常量是写死的，不看结果再调：
#   地形 R：P 是**归约**（读 (4,32)=512 字节，写 (4,)=16 字节，R/W = 32）
#   地形 P：P 是**点式**（读 2 个 (4,)=16 字节 → 32 字节，写 16 字节，R/W = 2）
LAMBDA1 = [np.full((4, 32), 2.0, np.float32), np.ones((4, 32), np.float32),
           np.ones((4, 2), np.float32)]
LAMBDA2 = [np.full((4,), 2.0, np.float32), np.ones((4,), np.float32),
           np.ones((4, 2), np.float32)]


def terrain_reduction(x, w, m):
    s = (x * w).sum(axis=1)      # 归约：读得多、写得少
    e = s.matmul(m)
    c = s + 1.0
    return [e, c]


def terrain_pointwise(x, w, m):
    p = x * w                    # 点式：读得和写得一样多
    e = p.matmul(m)
    c = p + 1.0
    return [e, c]


def ref_reduction(x, w, m):
    s = (x * w).sum(axis=1)
    return [s @ m, s + 1.0]


def ref_pointwise(x, w, m):
    p = x * w
    return [p @ m, p + 1.0]


def two_plans(g, mid_buf):
    """两个候选计划：

    · 「物化」：P 自己一组，它的输出落到存储，两个消费者各自读它；
    · 「内联」：P 被并进**每一个读了它的**消费者组，输出不落存储、被重算。
    """
    mid = g.by_buf(mid_buf)
    others = [n for n in g.nodes
              if n.kind in ("pointwise", "reduction", "extern") and n is not mid]
    plan_mat = sorted([[n] for n in others] + [[mid]],
                      key=lambda grp: g.nodes.index(grp[0]))
    plan_inl = [[n] for n in others]
    for grp in plan_inl:
        if any(d.buf == mid.buf for n in grp for d in n.reads):
            grp.append(mid)
    plan_inl = sorted([sorted(grp, key=lambda n: g.nodes.index(n))
                       for grp in plan_inl],
                      key=lambda grp: g.nodes.index(grp[0]))
    return plan_mat, plan_inl, mid


for label, fn, ref, inputs, mid_buf in (
        ("地形 R（P 是归约，R/W=32）", terrain_reduction, ref_reduction, LAMBDA1, "buf1"),
        ("地形 P（P 是点式，R/W=2）", terrain_pointwise, ref_pointwise, LAMBDA2, "buf0")):
    g = capture(fn, inputs)
    plan_mat, plan_inl, mid = two_plans(g, mid_buf)
    t_mat = F.plan_traffic(g, plan_mat)
    t_inl = F.plan_traffic(g, plan_inl)
    print()
    print("%s  中间结果 = %s（%d 元素）" % (label, mid.op, mid.numel))
    print("   计划「物化」：组 %s → 总字节 %6d（重算 %d 个节点）"
          % ([[n.op for n in grp] for grp in plan_mat], t_mat["total"],
             t_mat["recomputed_nodes"]))
    print("   计划「内联」：组 %s → 总字节 %6d（重算 %d 个节点）"
          % ([[n.op for n in grp] for grp in plan_inl], t_inl["total"],
             t_inl["recomputed_nodes"]))
    better = "内联" if t_inl["total"] < t_mat["total"] else "物化"
    print("   本机代价模型的结论：应当 **%s**" % better)
    if mid.kind == "reduction":
        checks.C(t_inl["total"] > t_mat["total"],
                 "C3：中间结果是归约（读 ≫ 写）时，把 P 内联进两个消费者**反而更贵**",
                 "内联 %d > 物化 %d 字节" % (t_inl["total"], t_mat["total"]))
        checks.C(t_inl["recomputed_nodes"] == 1,
                 "C3：内联计划里 P 真的被重算了一次（不是纸面说法）",
                 "recomputed_nodes=%d" % t_inl["recomputed_nodes"])
    else:
        checks.expect_ok(lambda: t_inl["total"] < t_mat["total"],
                         "对照：换成点式中间结果（读 ≈ 写）时，同一个模型改判「应当内联」",
                         "内联 %d < 物化 %d 字节" % (t_inl["total"], t_mat["total"]))

# —— 可证伪性反证：把「按地形判断」换成常数策略，两条断言里必有一条失败 ——
print()
print("   —— 可证伪性反证：把策略换成常数 ——")
g_r = capture(terrain_reduction, LAMBDA1)
g_p = capture(terrain_pointwise, LAMBDA2)
pairs = {}
for name, g, mid_buf in (("R", g_r, "buf1"), ("P", g_p, "buf0")):
    pairs[name] = two_plans(g, mid_buf)[:2]

always_inline = all(
    F.plan_traffic(g, inl)["total"] < F.plan_traffic(g, mat)["total"]
    for (mat, inl), g in ((pairs["R"], g_r), (pairs["P"], g_p)))
always_mat = all(
    F.plan_traffic(g, mat)["total"] <= F.plan_traffic(g, inl)["total"]
    for (mat, inl), g in ((pairs["R"], g_r), (pairs["P"], g_p)))
print("   常数策略「永远内联」在两个地形上都成立吗？%s" % always_inline)
print("   常数策略「永远物化」在两个地形上都成立吗？%s" % always_mat)
checks.C(not always_inline,
         "反证：常数策略「永远内联」在地形 R 上失败 → 判定确实依赖地形，不是在检验地形",
         "always_inline=%s" % always_inline)
checks.C(not always_mat,
         "反证：常数策略「永远物化」在地形 P 上失败",
         "always_mat=%s" % always_mat)

# ---- A 类：两种分组都必须算出同一个结果 ------------------------------------------------------

M.rule("6. A 算法正确性：分组不影响语义（生成代码真的跑）")

for label, fn, ref, inputs, mid_buf in (
        ("地形 R", terrain_reduction, ref_reduction, LAMBDA1, "buf1"),
        ("地形 P", terrain_pointwise, ref_pointwise, LAMBDA2, "buf0")):
    g = capture(fn, inputs)
    want = ref(*[np.asarray(a) for a in inputs])
    for name, groups in zip(("物化", "内联"), two_plans(g, mid_buf)[:2]):
        src, meta = F.emit_python(F.Plan(g, groups, []))
        env = {n.buf: n.value for n in g.nodes if n.kind == "placeholder"}
        got_env = F.exec_kernels(src, meta, env)
        errs = [M.max_abs_err(got_env[o.buf], w)
                for o, w in zip(g.outputs, want)]
        print("%s · %s 计划：生成 %d 个核，最大绝对误差 %.3e"
              % (label, name, meta and len(meta), max(errs)))
        checks.A(max(errs) == 0.0,
                 "A：%s 的「%s」分组，生成代码执行结果 == 纯 numpy 参考实现"
                 % (label, name), "最大绝对误差 %.3e" % max(errs))

M.rule("7. 断言汇总")
checks.summary()
