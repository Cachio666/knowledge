"""01 计算与调度分离：同一份计算，默认调度把它**全部物化**。

对应文档：60-compilers/tvm/01-compute-and-schedule.md

★ 本程序跑的是本仓库的**迷你 TIR**（minitir.py），不是 TVM。
  输出的一切数字都是**本机实测值**，只支撑关于机制的结论。

本程序要证的机制只有一条：
  「计算」与「调度」是两件事——**计算定义了算什么，调度决定中间结果物化不物化**。
  而在 v0.26.0 的默认调度下，**每一个中间结果都被物化在根作用域**（不是 inline）。

用法：python3 01_compute_schedule.py
"""

import numpy as np

import specs as SPECS
from minitir import (Axis, Buffer, Checks, InfeasibleSchedule, Interp, Plan,
                     PrimFunc, Schedule, ScheduleFailure, Stage, ESIZE, max_err,
                     rule, table)

# ---- 地形（构造规则写死在这里，不因结果调整） --------------------------------

M, RW = 8, 512                       # 形状 M × RW
LEVEL = M * RW                       # 每一级的元素数
K = 5                                # 中间 stage 的级数（s1..s5）
I = Axis("i", M)
J = Axis("j", RW)
C_MUL, C_ADD, C_OUT = 1.5, 0.25, 2.0


def build_chain():
    """一条 K 级链：a,b -> s1 -> s2 -> ... -> s5 -> out。**只写计算，不写调度。**"""
    inputs = [Buffer("a", [I, J], role="input"), Buffer("b", [I, J], role="input")]
    stages = []
    prev = inputs[0]
    for n in range(1, K + 1):
        out = Buffer("s%d" % n, [I, J], role="intermediate")
        if n == 1:
            fn = lambda a, b: a + b                                 # noqa: E731
            ins = [inputs[0], inputs[1]]
        else:
            fn = lambda p: p * C_MUL + C_ADD                        # noqa: E731
            ins = [prev]
        stages.append(Stage("s%d" % n, [I, J], ins, out, fn))
        prev = out
    out_buf = Buffer("out", [I, J], role="output")
    stages.append(Stage("out", [I, J], [prev], out_buf,
                        lambda p: p * C_OUT))
    return PrimFunc("chain%d" % K, inputs, stages)


# ---- 独立参考实现（纯 numpy，与迷你 TIR 没有任何共享代码） -------------------


def reference_chain(a, b, k=K):
    x = a + b
    for _ in range(k - 1):
        x = x * C_MUL + C_ADD
    return x * C_OUT


def make_inputs():
    a = (np.arange(M * RW).reshape(M, RW) + 1.0) / 1024.0
    b = (np.arange(M * RW).reshape(M, RW) % 7) * 0.125
    return {"a": a, "b": b}


def describe(func, plan):
    """按调度打印每个 stage 的循环结构（迷你实现的打印格式，用于说明结构）。"""
    lines = []
    for st in func.topo():
        p = plan.place[st.name]
        body = "%s[%s] = %s(%s)" % (
            st.out.name, ",".join(a.name for a in st.axes), st.name,
            ",".join(b.name for b in st.ins))
        if p == ("root",):
            head = "根位置："
            nest = "".join("for %s in [0,%d) { " % (a.name, a.extent)
                           for a in st.axes)
            lines.append("%-4s %s%s %s %s" % (st.name, head, nest, body, "}" * len(st.axes)))
        elif p[0] == "at":
            lines.append("%-4s 物化在 %s 的 %s 循环内（切片 %d 个元素）"
                         % (st.name, p[1], p[2], st.out.size // st.axes[0].extent))
        else:
            lines.append("%-4s 不物化（inline 进消费者，重算）" % st.name)
    return "\n".join(lines)


# ---- 跑 ---------------------------------------------------------------------

checks = Checks()
func = build_chain()

rule("1. 计算：只写「算什么」，一个字都没提「放哪儿」")
print("prim_func %s：%d 个 stage，%d 个入参，%d 个输出"
      % (func.name, len(func.stages), len(func.inputs), len(func.sinks())))
print("每级形状 %d×%d = %d 个元素（%.1f KiB/级）"
      % (M, RW, LEVEL, LEVEL * ESIZE / 1024.0))
print()
print("中间结果：%s" % ", ".join(st.name for st in func.stages
                              if st.out.role == "intermediate"))

sch = Schedule(func)
default_plan = Plan(func, sch)

print()
print("默认调度（没有调用任何原语）：")
print(describe(func, default_plan))
print()
print("★ 默认位置不是 inline：迷你 Schedule 里没有 `compute_root` 这个方法"
      "（has_compute_root() = %s），因为**默认就在根位置**。"
      % sch.has_compute_root())
print("  真实 s_tir（%s）的两态与默认位置的对应关系见 code/specs.py："
      % SPECS.TVM_TAG)
print("  %s" % SPECS.DEFAULT_MATERIALIZE_HEADLINE)

# ==== A 类 ====

rule("2. A 算法正确性：解释器 vs 独立 numpy 参考实现")

vals = make_inputs()
ref = reference_chain(vals["a"], vals["b"])
got = Interp(default_plan).run(vals)["out"]
e = max_err(got, ref)
print("参考实现 reference_chain = ((a+b)*1.5+0.25)×4 级，再 ×2")
print("解释器结果 vs 参考：最大误差 %.3e" % e)
print("结果样例：%s" % np.array2string(ref.ravel()[:4], precision=4))
checks.A(e == 0.0, "默认调度的执行结果 == 独立 numpy 参考实现", "最大误差 %.3e" % e)

# ==== B 类 ====

rule("3. B 代价模型：解析式 vs 解释器实测（两条独立路径）")

analy = dict(default_plan.summary_rows())
interp = Interp(default_plan)
interp.run(vals)
observed = dict(interp.report())

print("%-24s %12s %12s  %s" % ("项", "解析式", "实测", "判定"))
print("-" * 62)
for name, want in default_plan.summary_rows():
    gotv = observed[name]
    ok = (gotv == want)
    print("%-24s %12d %12d  %s" % (name, want, gotv, "ok" if ok else "MISMATCH"))
    checks.B(ok, "解析式 %s == 实测（%s）" % (name, want), "实测 %s" % gotv)

# 文档 §1 的推导：默认全部物化 ⟹ 物化次数 = 中间 stage 数
n_inter = len([st for st in func.stages if st.out.role == "intermediate"])
checks.B(default_plan.n_buffers == n_inter,
         "★ 默认调度下**每个中间结果都被物化**（物化次数 == 中间 stage 数）",
         "物化 %d 个，中间 stage %d 个" % (default_plan.n_buffers, n_inter))
checks.B(default_plan.peak_elems == n_inter * LEVEL,
         "峰值中间存储 == 级数 × 每级元素数（全部经由根作用域）",
         "%d == %d × %d" % (default_plan.peak_elems, n_inter, LEVEL))
checks.B(default_plan.writes == n_inter * LEVEL and
         default_plan.reads == n_inter * LEVEL,
         "每级写一次、读一次 ⟹ 流量 == 2 × 级数 × 每级元素数",
         "写 %d + 读 %d = %d" % (default_plan.writes, default_plan.reads,
                                default_plan.writes + default_plan.reads))
checks.B(default_plan.times["s1"] == 1,
         "物化的中间结果每元素只被求值一次（times == 1，重算为 0）",
         "times(s1) = %d" % default_plan.times["s1"])

# ==== C 类 ====

rule("4. C 边界与反例")

# C1：违反 compute_at 的前置条件必须被拒绝（消费者不在目标 loop 下）
steps = [st for st in func.stages if st.name.startswith("s")]
out_stage = func.stage("out")
bad_loop = sch.get_loops(out_stage)[0]
checks.expect_raises(
    lambda: Schedule(func).compute_at(func.stage("s1"), bad_loop),
    ScheduleFailure,
    "C1 违反 compute_at 前置条件（s1 的消费者 s2 不在 out 的循环下）→ 拒绝")
# C1 的对照：换成满足前置条件的调用，同一判据放行


def _ok_call():
    s = Schedule(func)
    s.compute_at(func.stage("s5"), bad_loop)
    return s.placement("s5")


checks.expect_ok(_ok_call,
                 "C1 对照：把最后一级 s5 放在 out 的 i 循环下 → 接受（它是 out 的直接生产者）")

# C2：容量超限 —— 按代价模型不可行，而且实测峰值**真的**超
CAP = 32 * 1024
print()
print("可用容量 CAP = %d 字节（%d KiB）" % (CAP, CAP // 1024))
checks.expect_raises(lambda: default_plan.check_capacity(CAP),
                     InfeasibleSchedule,
                     "C2 默认调度在 CAP 下被判不可行（峰值 %d > %d）"
                     % (default_plan.peak_elems * ESIZE, CAP))
checks.C(default_plan.peak_elems * ESIZE > CAP,
         "C2 实测峰值确实超过 CAP——不可行判定不是空话",
         "实测峰值 %d 字节 > CAP %d" % (default_plan.peak_elems * ESIZE, CAP))
# C2 的对照：把全部中间结果搬进 out 的 i 循环，峰值降到 CAP 以内
# ★ 次序是**从后往前**：compute_at 的前置条件看的是**当前**放置状态，
#   先搬 s5 才谈得上 s4 的消费者「已经在 loop 下」——这条过程式次序是本迷你实现
#   有意复现的真实性质（见 code/specs.py 的 PROCEDURAL_QUOTE）。
sch2 = Schedule(func)
for n in range(K, 0, -1):
    sch2.compute_at(func.stage("s%d" % n), bad_loop)
tight = Plan(func, sch2)
checks.expect_ok(lambda: tight.check_capacity(CAP),
                 "C2 对照：全部 compute_at 到 out 的 i 循环后，同一 CAP 下可行",
                 "峰值 %d 字节 ≤ %d" % (tight.peak_elems * ESIZE, CAP))

# C3：反例 —— 数值正确性**不能**用来判断调度写没写对
sch3 = Schedule(func)
for n in range(1, K + 1):
    sch3.compute_inline(func.stage("s%d" % n))
inline_plan = Plan(func, sch3)
got_inline = Interp(inline_plan).run(vals)["out"]
same = max_err(got, got_inline) == 0.0
print()
print("默认调度物化 %d 个中间 buffer；全 inline 调度物化 %d 个"
      % (default_plan.n_buffers, inline_plan.n_buffers))
print("两者的数值最大差：%.3e" % max_err(got, got_inline))
checks.C(same and default_plan.n_buffers != inline_plan.n_buffers,
         "C3 反例：两份代价完全不同的调度，数值**逐元素相同**——"
         "A 类正确性检查对调度错误是盲的",
         "buffers %d vs %d，数值差 %.3e"
         % (default_plan.n_buffers, inline_plan.n_buffers, max_err(got, got_inline)))

# C4：可证伪性反证 —— 把 B 类的解析公式故意改错，必须失配
wrong = dict(default_plan.summary_rows())
wrong["中间流量（元素）"] = default_plan.writes          # 只算写、漏掉读
observed_traffic = observed["中间流量（元素）"]
print()
print("故意把解析公式改错（流量只算写、漏掉读）：%d" % wrong["中间流量（元素）"])
checks.C(wrong["中间流量（元素）"] != observed_traffic,
         "C4 反证：把解析公式改错后，B 类比对**必须失配**——"
         "这条比对有判别力，不是恒真的",
         "改错值 %d != 实测 %d" % (wrong["中间流量（元素）"], observed_traffic))

rule("5. 断言汇总")
checks.summary()
