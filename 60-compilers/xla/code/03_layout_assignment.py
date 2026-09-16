"""03 布局与内存：layout assignment 与 memory space assignment。

对应文档：60-compilers/xla/03-layout-and-memory.md

★ 本程序跑的是本仓库的**迷你 layout / 内存实现**（`minilayout.py` + `minihlo.py`），
  **不是 XLA**。输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。
  真实 XLA 的行为（`kCopy` 的插入、MSA 的方向、GPU 侧同名不同物的着色机制、
  GPU pipeline 不跑 MSA …）只在文档与 `specs.py` 里，逐条带一手来源，
  **本程序不对它们下任何断言**。

三类断言：
  A 算法正确性 —— 布局表与**手写的期望值**逐项比对，并用**独立的校验器**重走一遍图
  B 代价模型   —— kCopy 条数 / alt 与 default 的字节数 / prefetch 与 eviction 条数
  C 边界与反例 —— 文档 §4 声称「会插入 kCopy / 会被拒绝 / 会被赶回 default」的情形
                 必须**真的发生**

用法：python3 03_layout_assignment.py
"""

import numpy as np

import specs as SPECS
from minihlo import Checks, Module, Shape, rule
import minilayout as ML
from minilayout import (COLOR_NAMES, DEFAULT_LIVE_RANGE_LIMIT, Layout, MemorySpaceColor,
                        ValueInfo, assign_layouts, check_layouts, color_value,
                        host_offload_custom_calls, memory_space_assignment,
                        op_output_layout, pipeline_passes, required_operand_layout)

SEED = 20260916
rng = np.random.default_rng(SEED)

ALT_CAPACITY = 4096          # 迷你 alternate memory 容量（字节）
MINI_GPU_PASS_MAP = {        # 把 specs 里从 XLA 源码抄来的 pass 名映射成本迷你 pipeline 的名字
    "RunLayoutAssignmentPasses": "layout_assignment",
    "RunLayoutNormalizationPasses": "layout_normalization",
    "OptimizeHloPostLayoutAssignment": "host_offloader",     # 见 specs：HostOffloader 在这条里
    "RunDynamicSliceFusionPasses": "dynamic_slice_fusion",
    "RunFusionPasses": "fusion",
    "RunPostFusionPasses": "post_fusion",
}

# ---------------------------------------------------------------------------
# ★★ 地形冻结区（§0.4 第 5 条）
# 下面两张图与那段 MSA 的值表，都在**写下任何运行结果之前**就固定在这里。
# ---------------------------------------------------------------------------


def build_dot_reduce(reduce_dim):
    """`dot` 之后接一个 `reduce`。`dot` 的输出布局被**钉死**在 `{n-1,...,1,0}`。

    reduce_dim=0 → 归约维要求它**最次**，与钉死的布局冲突 → 需要一条 kCopy。
    reduce_dim=1 → 归约维本来就最次，与钉死布局一致 → 不需要 copy。
    """
    m = Module("lr%d" % reduce_dim)
    p0 = m.parameter(Shape([8, 16], "f32"), "p0")
    p1 = m.parameter(Shape([16, 4], "f32"), "p1")
    d = m.emit("dot_general", [p0, p1], Shape([8, 4], "f32"),
               lhs_contracting_dims=[1], rhs_contracting_dims=[0])
    m.emit("reduce", [d], Shape([8] if reduce_dim == 1 else [4], "f32"),
           dimensions=[reduce_dim])
    return m


# MSA 的值表： (名字, def, last_use, 字节数)
MSA_VALUES = [
    ("v0", 0, 3, 1024),
    ("v1", 1, 6, 1024),      # 活跃区间 5 ≥ limit 4 → 不适合 no-copy 的 alt 分配
    ("v2", 2, 3, 2048),
    ("v3", 3, 4, 4096),      # 单独就超过 alt 容量
    ("v4", 4, 5, 512),       # ★ 前一个没塞进去，这一个仍要能塞进去
]


def main():
    checks = Checks()

    # ------------------------------------------------------------------
    rule("1. layout assignment：约束沿图传播，冲突处插 kCopy")
    m_bad = build_dot_reduce(0)
    lays_bad, copies_bad, notes_bad = assign_layouts(m_bad)
    print(m_bad.text())
    print()
    print("  dot 的输出布局被**钉死**在 %s（本迷你版的口径，见 minilayout.op_output_layout）"
          % op_output_layout(m_bad.by_name("dot_general.0"), []))
    print("  reduce(dim=0) 要求它的操作数最次维是第 0 维 → %s"
          % required_operand_layout(m_bad.by_name("reduce.0"),
                                    lays_bad["reduce.0"], 0))
    print()
    for op in m_bad.ops:
        print("  %-16s 布局 %s" % (op.name, lays_bad[op.name]))
    print("  插入的 kCopy：%s" % (copies_bad or "无"))
    print()
    print("  来源（一手，见 specs.py）：%s" % SPECS.LAYOUT_ASSIGNMENT_MECHANISM)

    # ------------------------------------------------------------------
    rule("2. A 算法正确性：与**手写的期望布局表**比对 + 独立校验器重走")
    # ★ 期望值手写在这里，不由 assign_layouts 产生
    EXPECTED_BAD = {"dot_general.0": Layout([1, 0]), "reduce.0": Layout([0])}
    got_bad = {k: lays_bad[k] for k in EXPECTED_BAD}
    same = all(got_bad[k] == EXPECTED_BAD[k] for k in EXPECTED_BAD)
    print("  期望 %s" % {k: str(v) for k, v in EXPECTED_BAD.items()})
    print("  本机 %s" % {k: str(v) for k, v in got_bad.items()})
    checks.A(same, "布局表与手写期望值逐项一致",
             "%d 项" % len(EXPECTED_BAD))
    checks.A(len(copies_bad) == 1 and copies_bad[0].op_name == "dot_general.0",
             "恰好插入 1 条 kCopy，且落在被钉死的那个值上",
             str(copies_bad))
    problems = check_layouts(m_bad, lays_bad, copies_bad)
    print("  独立校验器（只看最终的布局表 + copy 列表，重走一遍图）：%d 个问题"
          % len(problems))
    checks.A(not problems, "独立校验器确认每条边「布局一致 或 有 kCopy」",
             "问题：%s" % problems)

    m_ok = build_dot_reduce(1)
    lays_ok, copies_ok, _ = assign_layouts(m_ok)
    print()
    print("  对照：reduce(dim=1) 的图 → 布局 %s，kCopy %d 条"
          % ({k: str(v) for k, v in lays_ok.items()}, len(copies_ok)))
    checks.A(not copies_ok and not check_layouts(m_ok, lays_ok, copies_ok),
             "对照：归约维与钉死布局**一致**时，一条 kCopy 都不需要",
             "copies=%d" % len(copies_ok))

    # ------------------------------------------------------------------
    rule("3. A 算法正确性：把被测实现故意改错，A 类判据必须报错")

    def broken_assign(module):
        """故意写错：**忽略算子自身钉死的布局**，让消费者的要求直接覆盖它。"""
        out, requested = {}, {}
        for op in reversed(module.ops):
            out[op.name] = requested.get(op.name, Layout.row_major(op.shape.rank))
            for i, operand in enumerate(op.operands):
                need = required_operand_layout(op, out[op.name], i)
                if need is not None:
                    requested[operand.name] = need
        return out

    broken = broken_assign(m_bad)
    mismatch = {k: str(broken[k]) for k in EXPECTED_BAD
                if broken[k] != EXPECTED_BAD[k]}
    print("  改错后的布局表：%s" % {k: str(v) for k, v in broken.items()})
    print("  与手写期望不一致的项：%s" % mismatch)
    checks.C(bool(mismatch),
             "反证：忽略「钉死布局」的实现在 A 类的期望表比对下会失败",
             "不一致 %d 项" % len(mismatch))

    # ------------------------------------------------------------------
    rule("4. B 代价模型：kCopy 与内存空间的计数（与文档 §5 表格逐行核对）")
    vals = [ValueInfo(n, d, u, b) for n, d, u, b in MSA_VALUES]
    res = memory_space_assignment(vals, ALT_CAPACITY)
    print("  %s" % res.report())
    print("    alternate: %s" % res.alt)
    print("    default  : %s" % res.default)
    print("    prefetch : %s" % res.prefetches)
    print("    eviction : %s" % res.evictions)
    b_rows = [
        ("alt 里的值个数", len(res.alt), 3),
        ("alt 字节数", res.alt_bytes, 3584),
        ("default 里的值个数", len(res.default), 2),
        ("default 字节数", res.default_bytes, 5120),
        ("prefetch 条数", len(res.prefetches), 3),
        ("eviction 条数", len(res.evictions), 1),
        ("kCopy 条数（上一步的图）", len(copies_bad), 1),
    ]
    print()
    print("%-28s %10s %8s" % ("项", "本机值", "文档值"))
    print("-" * 50)
    for name, got, want in b_rows:
        ok = (got == want)
        print("%-28s %10s %8s  %s" % (name, got, want, "ok" if ok else "MISMATCH"))
        checks.B(ok, "文档表：%s == %s" % (name, want), "本机 %s" % got)

    # ------------------------------------------------------------------
    rule("5. C 边界与反例：这些情形**必须真的发生**")

    # C1：default memory「假定足够大」——落进 default 的字节数**超过了 alt 容量**
    checks.C(res.default_bytes > ALT_CAPACITY,
             "★ C1：default memory 没有容量上限——落在里面的字节数（%d）"
             "**超过了 alt 的容量**（%d），程序不报错"
             % (res.default_bytes, ALT_CAPACITY),
             "%d > %d" % (res.default_bytes, ALT_CAPACITY))

    # C2：方向 —— 目标是「尽量多塞进 alt」，不是「把大的赶去慢存储」
    small_placed = "v4" in res.alt
    big_rejected = "v3" in res.default
    checks.C(small_placed and big_rejected,
             "★ C2：贪心**继续**扫描——先遇到塞不下的大值（v3 → default），"
             "后面更小的值（v4）仍然进了 alternate。"
             "这正是「greedily try placing as many values in the alternate memory "
             "space as possible」的方向",
             "v4 in alt=%s, v3 in default=%s" % (small_placed, big_rejected))

    # C2 的可证伪性：把贪心改成「一失败就停」，上面那条断言就不再成立
    def stop_at_first_failure(values, capacity):
        placed, used = [], 0
        for v in values:
            if used + v.nbytes > capacity:
                break                      # ★ 一失败就停
            placed.append(v.name)
            used += v.nbytes
        return placed

    stopped = stop_at_first_failure(vals, ALT_CAPACITY)
    print("  反证：把贪心改成「一失败就停」→ alt 里只有 %s" % stopped)
    checks.C("v4" not in stopped,
             "★ C2 的可证伪性：换成「一失败就停」的实现在 v4 上会失败——"
             "所以上面那条断言检验的确实是**贪心的方向**",
             "停版 alt=%s" % stopped)

    # C3：活跃区间过长的值不做 no-copy 的 alt 分配
    checks.C("v1" in res.default,
             "★ C3：活跃区间长度 %d ≥ 本迷你版的 limit %d → 该值不进 alternate"
             % (dict((n, u - d) for n, d, u, _ in MSA_VALUES)["v1"],
                DEFAULT_LIVE_RANGE_LIMIT),
             "v1 lr=%d" % (6 - 1))
    checks.C(6 - 1 >= DEFAULT_LIVE_RANGE_LIMIT,
             "★ C3 的前提成立：v1 的活跃区间确实越过了 limit（否则这条断言是空的）",
             "%d >= %d" % (6 - 1, DEFAULT_LIVE_RANGE_LIMIT))

    # ------------------------------------------------------------------
    rule("6. 两个坑：同名不同物的着色 + GPU pipeline 不跑 MSA")

    color_mod = Module("color")
    c1 = color_mod.parameter(Shape([8, 8], "f32"), "p0")
    col = color_mod.emit("all-reduce", [c1], Shape([8, 8], "f32"), collective=True)
    color_mod.emit("add", [col, c1], Shape([8, 8], "f32"), broadcast_dimensions=[])
    colors = {o.name: COLOR_NAMES[color_value(o)] for o in color_mod.ops}
    print("  本迷你 colorer 的着色：%s" % colors)
    print("  着色枚举：%s" % COLOR_NAMES)
    print("  ↑ 这三个成员与 MSA 的 default/alternate **不是一回事**：")
    print("    着色发生在 **buffer assignment** 阶段（对应 XLA 的 CreateColorer），")
    print("    MSA 决定的是「值放哪个 memory space」。两者正交。")
    checks.C(color_value(color_mod.by_name("all-reduce.0"))
             == MemorySpaceColor.kCollective
             and len(COLOR_NAMES) == 3,
             "★ 坑①：存在**同名但不同**的 buffer 着色机制（3 个成员），"
             "与本程序的 MSA 是两件事——一个值可以被着色成 kCollective，"
             "同时被 MSA 放进 default",
             "all-reduce.0 着色 %s" % colors["all-reduce.0"])

    mini_gpu = [MINI_GPU_PASS_MAP[p] for p in SPECS.GPU_PIPELINE_ORDER]
    print()
    print("  XLA 侧的 pass 顺序（从 gpu_compiler.cc 抄来，见 specs.py）：")
    for p in SPECS.GPU_PIPELINE_ORDER:
        print("    %s" % p)
    print("  → 映射成本迷你 pipeline：%s" % mini_gpu)
    print("  本迷你 CPU pipeline：%s" % pipeline_passes("cpu"))
    print("  host 卸载用的 custom call target：%s（PascalCase，来源见 specs.py）"
          % host_offload_custom_calls())
    checks.C("memory_space_assignment" not in mini_gpu,
             "★ 坑②：本迷你 GPU pipeline **不含** MSA pass（host 卸载走 host_offloader + "
             "MoveToHost/MoveToDevice）——所以把 MSA 当成 GPU 的机制是错的",
             "gpu=%s" % mini_gpu)
    checks.A("memory_space_assignment" in pipeline_passes("cpu"),
             "对照：本迷你 CPU pipeline **含** MSA pass",
             "cpu=%s" % pipeline_passes("cpu"))
    checks.A(host_offload_custom_calls() == ["MoveToHost", "MoveToDevice"],
             "host 卸载的 custom call target 是 PascalCase（照抄 specs 的一手值）",
             str(host_offload_custom_calls()))

    # ------------------------------------------------------------------
    rule("7. 断言汇总")
    checks.summary()


if __name__ == "__main__":
    main()
