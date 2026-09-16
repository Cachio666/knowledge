"""02 融合：两套 fusion kind 与「预算 + 合法性 + 代价模型」三层判据。

对应文档：60-compilers/xla/02-fusion.md

★ 本程序跑的是本仓库的**迷你融合实现**（`minifuse.py` + `minilayout.py` + `minihlo.py`），
  **不是 XLA**。输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。
  真实 XLA 的融合判据（`FusionFitsInBudget` / `GpuPerformanceModel` / … ）
  只在文档与 `specs.py` 里，逐条带一手来源，**本程序不对它们下任何断言**。

三类断言：
  A 算法正确性 —— 融合后的执行 == 未融合的执行 == **手写的 numpy 参照表达式**
  B 代价模型   —— 预算计数 / 流量计数 / 融合前后算子数，须与文档 §5 表格逐行一致
  C 边界与反例 —— 文档 §4 声称「会被拒绝 / 反而变多」的情形必须**真的发生**

用法：python3 02_fusion_kinds.py
"""

import numpy as np

import specs as SPECS
from minifuse import (BUDGET_LIMIT, EmitterFusionKind, FusionKind, Group, Plan,
                      check_budget, check_legality, execute_plan,
                      layout_conversions, ops_after_fusion, perfect_cost_model,
                      try_fuse)
from minihlo import Checks, Module, Shape, execute, rule
import minilayout as ML

SEED = 20260916
rng = np.random.default_rng(SEED)
S = 64                                  # 小方阵边长
S2 = 16

# ---------------------------------------------------------------------------
# ★★ 地形冻结区（§0.4 第 5 条）
#
# 下面每一张图、每一个候选融合组、以及各层的判定，都在**写下任何运行结果之前**
# 就固定在这里。种子固定为 SEED。**不得看到结果再回来调地形**。
# 每张图都配一条**手写的 numpy 参照表达式**（A 类的独立参考实现）。
# ---------------------------------------------------------------------------


def terrain_transpose():
    """F1：`transpose` 后面跟一个 `add`。**物理转置型算子**参加融合。"""
    m = Module("f1_transpose")
    p0 = m.parameter(Shape([2, 3], "f32"), "p0")
    p1 = m.parameter(Shape([3, 2], "f32"), "p1")
    t = m.emit("transpose", [p0], Shape([3, 2], "f32"), permutation=[1, 0])
    m.emit("add", [t, p1], Shape([3, 2], "f32"), broadcast_dimensions=[])
    return m, [Group(["transpose.0", "add.0"])], \
        lambda f: np.transpose(f["p0"], (1, 0)) + f["p1"]


def terrain_elementwise():
    """F2：逐元素链 `(p0*p1)+p2`。融合只省一次中间结果的往返。"""
    m = Module("f2_elementwise")
    p0 = m.parameter(Shape([S, S], "f32"), "p0")
    p1 = m.parameter(Shape([S, S], "f32"), "p1")
    p2 = m.parameter(Shape([S, S], "f32"), "p2")
    mu = m.emit("multiply", [p0, p1], Shape([S, S], "f32"), broadcast_dimensions=[])
    m.emit("add", [mu, p2], Shape([S, S], "f32"), broadcast_dimensions=[])
    return m, [Group(["multiply.0", "add.0"])], lambda f: f["p0"] * f["p1"] + f["p2"]


def terrain_shared_producer():
    """F3：`dot` 的结果**同时**被组内和组外消费——融合要把 `dot` 再算一遍。

    这就是本域主线上那个权衡的另一面：**不物化就得重算**。
    """
    m = Module("f3_shared")
    p0 = m.parameter(Shape([S2, S2], "f32"), "p0")
    p1 = m.parameter(Shape([S2, S2], "f32"), "p1")
    p2 = m.parameter(Shape([S2, S2], "f32"), "p2")
    d = m.emit("dot_general", [p0, p1], Shape([S2, S2], "f32"),
               lhs_contracting_dims=[1], rhs_contracting_dims=[0])
    m.emit("add", [d, p2], Shape([S2, S2], "f32"), broadcast_dimensions=[])
    m.emit("multiply", [d, p2], Shape([S2, S2], "f32"), broadcast_dimensions=[])
    return m, [Group(["dot_general.0", "add.0"])], lambda f: (
        np.dot(f["p0"], f["p1"]) + f["p2"], np.dot(f["p0"], f["p1"]) * f["p2"])


def terrain_siblings():
    """F4：两个**兄弟**算子共享输入 p0。multi-output 融合的形态。"""
    m = Module("f4_siblings")
    p0 = m.parameter(Shape([S, S], "f32"), "p0")
    p1 = m.parameter(Shape([S, S], "f32"), "p1")
    p2 = m.parameter(Shape([S, S], "f32"), "p2")
    m.emit("multiply", [p0, p1], Shape([S, S], "f32"), broadcast_dimensions=[])
    m.emit("add", [p0, p2], Shape([S, S], "f32"), broadcast_dimensions=[])
    grp = Group(["multiply.0", "add.0"], multi_output=True, already_fusion=True,
                member_kinds=[FusionKind.kInput, FusionKind.kInput])
    return m, [grp], lambda f: (f["p0"] * f["p1"], f["p0"] + f["p2"])


def budget_terrain(n_extra_constants):
    """预算层的边界地形：若干互不相干的 `add`，外加 N 个常量。"""
    m = Module("budget")
    for k in range(32):
        a = m.parameter(Shape([2], "f32"), "a%d" % k)
        b = m.parameter(Shape([2], "f32"), "b%d" % k)
        m.emit("add", [a, b], Shape([2], "f32"), broadcast_dimensions=[])
    for k in range(n_extra_constants):
        m.emit("constant", [], Shape([1], "f32"), value=np.zeros(1, np.float32))
    members = [o.name for o in m.ops]
    return m, Group(members)


# ---------------------------------------------------------------------------
# 迷你 pipeline：对每个候选组依次过三层，把放行的组装进 Plan
# ---------------------------------------------------------------------------


def run_pipeline(module, groups, strict_layout=True):
    rows = []
    plan_groups = []
    for g in groups:
        d, trace = try_fuse(g, module, LAYOUTS[module.name], strict_layout=strict_layout)
        rows.append((g, d, trace))
        if d.allowed:
            plan_groups.append(g)
    # 没被任何组收走的 op 各自成单元素组，保证 Plan 覆盖全部 op
    taken = {n for g in plan_groups for n in g.members}
    for op in module.ops:
        if op.name not in taken:
            plan_groups.append(Group([op.name]))
    # 按模块序排列组，保证拓扑序
    order = {o.name: i for i, o in enumerate(module.ops)}
    plan_groups.sort(key=lambda g: min(order[n] for n in g.members))
    return rows, Plan(plan_groups)


LAYOUTS = {}
TERRAINS = {}


def prepare():
    """冻结的地形 + 各自的布局指派结果。全部在跑之前算好、存起来。"""
    f1, g1, r1 = terrain_transpose()
    f2, g2, r2 = terrain_elementwise()
    f3, g3, r3 = terrain_shared_producer()
    f4, g4, r4 = terrain_siblings()
    for m in (f1, f2, f3, f4):
        lays, copies, notes = ML.assign_layouts(m)
        LAYOUTS[m.name] = lays
        TERRAINS[m.name] = (m, lays, copies, notes)
    return (f1, g1, r1), (f2, g2, r2), (f3, g3, r3), (f4, g4, r4)


# ---------------------------------------------------------------------------


def main():
    checks = Checks()
    t1, t2, t3, t4 = prepare()

    # ------------------------------------------------------------------
    rule("1. 两套 fusion kind：名字像，但不是一回事")
    print("  FusionKind（%d 个）        : %s"
          % (len(FusionKind), [k.value for k in FusionKind]))
    print("  EmitterFusionKind（%d 个）: %s"
          % (len(EmitterFusionKind), [k.value for k in EmitterFusionKind]))
    print()
    only_emitter = sorted({k.value for k in EmitterFusionKind}
                          - {k.value for k in FusionKind})
    print("  只在 EmitterFusionKind 里有的成员：%s" % only_emitter)
    print("  ↑ 这些**不在** FusionKind 里——两套枚举描述的是两件事：")
    print("    FusionKind 说「primary node 是不是 root」；")
    print("    EmitterFusionKind 说「这个 fusion 由哪个 emitter 出码」。")
    print()
    print("  ★ 来自 XLA 的一手原文（来源见 specs.py，本程序不对它下断言）：")
    print("     %s" % SPECS.FUSION_KIND_TODO)
    print()
    checks.A(len(only_emitter) > 0,
             "迷你实现里两套枚举的成员集合不同（不是同一套枚举换名）",
             "EmitterFusionKind 多出 %d 个成员" % len(only_emitter))
    checks.A("kTuple" not in SPECS.HLO_FUSION_KIND_ENUM,
             "`FusionKind` 的枚举体里**没有** kTuple —— multi-output fusion 不是 kind，"
             "而是「root 是 kTuple」的一种**组形态**",
             "枚举体 4 个成员，kTuple 不在其中")

    # ------------------------------------------------------------------
    rule("2. 三层判据在四张地形上的判定（每层给出理由字符串）")
    plan_rows = []
    for tag, (m, groups, _ref) in (("F1", t1), ("F2", t2), ("F3", t3), ("F4", t4)):
        rows, plan = run_pipeline(m, groups)
        for g, d, trace in rows:
            print("  [%s] %-42s → %s" % (tag, g, d))
            for step in trace[:-1]:
                print("        层「%s」：%s" % (step.layer, step.reason))
        plan_rows.append((tag, m, plan, _ref))
    print()
    print("  说明：判定是**顺序**的——前一层否决，后面两层根本不会被问到。")

    # ------------------------------------------------------------------
    rule("3. A 算法正确性：融合执行 == 未融合执行 == 手写 numpy 参照")
    for tag, m, plan, ref in plan_rows[:2]:
        feeds = {p.name: rng.standard_normal(p.shape.dims).astype(np.float32)
                 for p in m.params}
        env_plain = execute(m, feeds)
        env_plan = execute_plan(m, plan, feeds)
        want = ref(feeds)
        got_plan = env_plan[m.root().name]
        got_plain = env_plain[m.root().name]
        e1 = float(np.max(np.abs(got_plan - want)))
        e2 = float(np.max(np.abs(got_plan - got_plain)))
        print("  [%s] 分组执行 vs numpy 参照：误差 %.3g；vs 未融合执行：误差 %.3g"
              % (tag, e1, e2))
        checks.A(e1 == 0.0 and e2 == 0.0,
                 "[%s] 融合（分组执行）与 numpy 参照、未融合执行三方全等" % tag,
                 "误差 %.3g / %.3g" % (e1, e2))

    # F3：代价模型否决 → plan 里没有融合组，仍要能正确执行两条输出
    _tag3, m3, plan3, ref3 = plan_rows[2]
    feeds3 = {p.name: rng.standard_normal(p.shape.dims).astype(np.float32)
              for p in m3.params}
    env3 = execute_plan(m3, plan3, feeds3)
    w_add, w_mul = ref3(feeds3)
    e3 = max(float(np.max(np.abs(env3["add.0"] - w_add))),
             float(np.max(np.abs(env3["multiply.0"] - w_mul))))
    print("  [F3] 被否决后仍按未融合执行：与 numpy 参照误差 %.3g" % e3)
    checks.A(e3 == 0.0 and len(plan3.groups) == 3,
             "[F3] 代价模型否决后 plan 里是 3 个单算子组，执行仍与 numpy 全等",
             "误差 %.3g，组数 %d" % (e3, len(plan3.groups)))

    _tag4, m4, plan4, ref4 = plan_rows[3]
    feeds4 = {p.name: rng.standard_normal(p.shape.dims).astype(np.float32)
              for p in m4.params}
    missing, dup = plan4.check_covers(m4)
    env4 = execute_plan(m4, plan4, feeds4)
    w_mu, w_ad = ref4(feeds4)
    e4 = max(float(np.max(np.abs(env4["multiply.0"] - w_mu))),
             float(np.max(np.abs(env4["add.0"] - w_ad))))
    print("  [F4] multi-output（兄弟）融合：与 numpy 参照误差 %.3g" % e4)
    checks.A(e4 == 0.0 and not missing and not dup,
             "[F4] 兄弟 multi-output 分组的执行与 numpy 全等，且 Plan 恰好覆盖全部 op",
             "误差 %.3g，缺 %s，重 %s" % (e4, missing, dup))

    # ------------------------------------------------------------------
    rule("4. B 代价模型：与文档 §5 的表格逐行核对")
    f1, g1, _ = t1
    f2, g2, _ = t2
    ext1 = g1[0].external_operands(f1)
    ext2 = g2[0].external_operands(f2)
    b_rows = [
        ("F1 组的外部操作数个数", len(ext1), 2),
        ("F1 组的 root 个数", len(g1[0].roots(f1)), 1),
        ("F1 组的 operands+outputs", len(ext1) + 1, 3),
        ("F2 组的外部操作数字节数", sum(f2.by_name(n).shape.nbytes()
                                    for n in ext2), 3 * S * S * 4),
        ("F1 严格 layout 下的转换条数",
         layout_conversions(g1[0], f1, LAYOUTS[f1.name], strict=True), 2),
        ("F1 放松 layout 下的转换条数",
         layout_conversions(g1[0], f1, LAYOUTS[f1.name], strict=False), 0),
        ("BUDGET_LIMIT", BUDGET_LIMIT, 96),
    ]
    print("%-30s %12s %10s" % ("项", "本机值", "文档值"))
    print("-" * 56)
    for name, got, want in b_rows:
        ok = (got == want)
        print("%-30s %12s %10s  %s" % (name, got, want, "ok" if ok else "MISMATCH"))
        checks.B(ok, "文档表：%s == %s" % (name, want), "本机 %s" % got)

    # 预算边界：96 恰好放行、97 被拒
    m96, grp96 = budget_terrain(0)
    m97, grp97 = budget_terrain(1)
    d96 = check_budget(grp96, m96)
    d97 = check_budget(grp97, m97)
    print()
    print("  预算边界：32 个 add（64 操作数 + 32 root = 96）→ %s" % d96)
    print("            再加 1 个常量 → 97 → %s" % d97)
    checks.B(d96.allowed and not d97.allowed,
             "预算层在 96 处恰好放行、97 处拒绝（边界是闭区间）",
             "96: %s / 97: %s" % (d96.allowed, d97.allowed))

    # ------------------------------------------------------------------
    rule("5. C 边界与反例：这些情形**必须真的发生**")

    _m3, g3, _r3 = t3

    # C1（地形冻结）：融合引入了额外的 transpose，算子数**反而变多**
    plan1 = Plan([Group(["transpose.0", "add.0"])])
    n_before = len(f1.ops)
    n_strict = ops_after_fusion(f1, plan1, LAYOUTS[f1.name], strict_layout=True)
    n_relax = ops_after_fusion(f1, plan1, LAYOUTS[f1.name], strict_layout=False)
    print("  构造函数 = transpose(p0) 再 add；转置是**物理转置型**算子")
    print("  融合前算子数 %d → 严格 layout 约束下 %d；放松约束后 %d"
          % (n_before, n_strict, n_relax))
    checks.C(n_strict > n_before,
             "★ C1：严格 layout 约束下，融合**反而让算子数变多**"
             "（断言的是**本仓库迷你模型**在什么条件下会这样，不是「XLA 会这样」）",
             "%d > %d" % (n_strict, n_before))
    checks.C(n_relax < n_before,
             "★ C1 的反证：把 layout 约束**放松**（等价于「布局指派时已知会被融合」），"
             "同一条判据不再成立——所以上面那条断言**是**关于 layout 约束的，"
             "不是关于地形的",
             "放松后 %d < %d" % (n_relax, n_before))

    # C2：合法性层——兄弟融合缺 kFusion
    sib_no_fusion = Group(["multiply.0", "add.0"], multi_output=True)
    d = check_legality(sib_no_fusion, m4)
    print("  兄弟融合、组内无 kFusion → %s" % d)
    checks.C(not d.allowed and "at least one op to be a kFusion" in d.reason,
             "C2a：兄弟融合缺 kFusion → 拒绝（理由逐字来自 XLA 的硬约束）",
             d.reason)

    sib_kind_mismatch = Group(["multiply.0", "add.0"], multi_output=True,
                              already_fusion=True,
                              member_kinds=[FusionKind.kInput, FusionKind.kLoop])
    d = check_legality(sib_kind_mismatch, m4)
    print("  兄弟融合、一个是 kInput 一个是 kLoop → %s" % d)
    checks.C(not d.allowed and "fusion kinds must match" in d.reason,
             "C2b：kInput 与 kLoop 的兄弟融合 → 拒绝（「the fusion kinds must match」）",
             d.reason)

    bad_param = Group(["p0"])
    d = check_legality(bad_param, m4)
    checks.C(not d.allowed, "C2c：把 parameter 放进融合组 → 拒绝", d.reason)

    # C3：代价模型否决一个**已经合法**的融合
    d3, trace3 = try_fuse(g3[0], m3, LAYOUTS[m3.name])
    print("  F3 的三层走完：%s" % d3)
    for step in trace3:
        print("        层「%s」：%s" % (step.layer, step.reason))
    checks.C(not d3.allowed and d3.layer == "代价模型"
             and "will execute slower if fused" in d3.reason,
             "★ C3：`dot` 的结果被组外也消费时，融合要**重算**一遍 —— "
             "前两层放行，第三层（代价模型）拒绝",
             d3.reason)
    checks.C(trace3[0].allowed and trace3[1].allowed,
             "★ C3 的对照：同一条融合在前两层（预算 / 合法性）**是放行的**——"
             "拒绝只发生在代价模型层",
             "%s / %s" % (trace3[0].allowed, trace3[1].allowed))

    # C 的可证伪性：把代价模型换成「完美模型」，C3 的拒绝就不再发生
    d3_perfect, _ = try_fuse(g3[0], m3, LAYOUTS[m3.name],
                             cost_fn=perfect_cost_model)
    print()
    print("  反证：把代价模型换成「永远放行」的完美模型 → %s" % d3_perfect)
    checks.C(d3_perfect.allowed != d3.allowed,
             "★ C3 的可证伪性：换掉代价模型后判定**翻转**——"
             "上面那条拒绝确实是代价模型给出的，不是别处顺带产生的",
             "%s → %s" % (d3.allowed, d3_perfect.allowed))

    # ------------------------------------------------------------------
    rule("6. 断言汇总")
    checks.summary()


if __name__ == "__main__":
    main()
