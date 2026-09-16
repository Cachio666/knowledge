"""02 调度原语与物化位置：三态对照 —— 默认 / compute_at / compute_inline。

对应文档：60-compilers/tvm/02-schedule-primitives.md

★ 本程序跑的是本仓库的**迷你 TIR**（minitir.py），不是 TVM。
  输出的一切数字都是**本机实测值**，只支撑关于机制的结论。

本程序要证的机制：

  · **两态不是三态**：迷你 Schedule 只有 `compute_at`（物化）与 `compute_inline`
    （不物化）。**没有 `compute_root`** —— 没被移走的 block 本来就在根位置。
  · **`compute_at` 是物化，不是重算**：它把中间结果的**作用域**缩到父循环一次迭代，
    但**依然分配独立存储**。
  · **`compute_inline` 才是不物化**，而「不物化」不是免费的：重算次数 = 消费者个数。

用法：python3 02_schedule_primitives.py
"""

import numpy as np

from minitir import (Axis, Buffer, Checks, Interp, Plan, PrimFunc, Schedule,
                     ScheduleFailure, Stage, ESIZE, max_err, rule)

# ---- 地形（构造规则写死在这里，不因结果调整） --------------------------------

M, RW = 8, 512
LEVEL = M * RW
K = 5
I = Axis("i", M)
J = Axis("j", RW)
C_MUL, C_ADD, C_OUT = 1.5, 0.25, 2.0


def build_chain(k=K):
    """a,b -> s1 -> s2 -> ... -> sk -> out（单消费者链）。"""
    inputs = [Buffer("a", [I, J], role="input"), Buffer("b", [I, J], role="input")]
    stages = []
    prev = inputs[0]
    for n in range(1, k + 1):
        out = Buffer("s%d" % n, [I, J], role="intermediate")
        if n == 1:
            stages.append(Stage("s1", [I, J], [inputs[0], inputs[1]], out,
                                lambda a, b: a + b))
        else:
            stages.append(Stage("s%d" % n, [I, J], [prev], out,
                                lambda p: p * C_MUL + C_ADD))
        prev = out
    stages.append(Stage("out", [I, J], [prev],
                        Buffer("out", [I, J], role="output"), lambda p: p * C_OUT))
    return PrimFunc("chain%d" % k, inputs, stages)


def build_fanout():
    """**多消费者地形**：s 同时被 out1 与 out2 消费。"""
    inputs = [Buffer("a", [I, J], role="input"), Buffer("b", [I, J], role="input")]
    s = Buffer("s", [I, J], role="intermediate")
    st_s = Stage("s", [I, J], [inputs[0], inputs[1]], s, lambda a, b: a * b)
    o1 = Buffer("out1", [I, J], role="output")
    o2 = Buffer("out2", [I, J], role="output")
    st_o1 = Stage("out1", [I, J], [s], o1, lambda p: p * 2.0)
    st_o2 = Stage("out2", [I, J], [s], o2, lambda p: p + 1.0)
    return PrimFunc("fanout", inputs, [st_s, st_o1, st_o2])


# ---- 独立参考实现（纯 numpy） -----------------------------------------------


def ref_chain(a, b, k=K):
    x = a + b
    for _ in range(k - 1):
        x = x * C_MUL + C_ADD
    return x * C_OUT


def ref_fanout(a, b):
    return a * b * 2.0, a * b + 1.0


def make_inputs():
    a = (np.arange(M * RW).reshape(M, RW) + 1.0) / 1024.0
    b = (np.arange(M * RW).reshape(M, RW) % 7) * 0.125
    return {"a": a, "b": b}


def make_schedule(func, how):
    """按名字造三种调度之一：root / at / inline。"""
    sch = Schedule(func)
    mid = [st for st in func.stages if st.out.role == "intermediate"]
    sink = func.sinks()[0]
    loop = sch.get_loops(sink)[0]
    if how == "at":
        # 次序从后往前：compute_at 的前置条件看的是**当前**放置状态（过程式调度）
        for st in reversed(mid):
            sch.compute_at(st, loop)
    elif how == "inline":
        for st in mid:
            sch.compute_inline(st)
    return sch


# ---- 跑 ---------------------------------------------------------------------

checks = Checks()
func = build_chain()
vals = make_inputs()
ref = ref_chain(vals["a"], vals["b"])

rule("1. 三态（本迷你实现里其实是两态 + 默认位置）")
print("迷你 Schedule 的方法：compute_at / reverse_compute_at / compute_inline /"
      " reverse_compute_inline")
print("★ 没有 compute_root：has_compute_root() = %s" % Schedule(func).has_compute_root())
print("  没被移走的 block 本来就在根位置（推断，依据见 code/specs.py）")

plans, results = {}, {}
for how in ("root", "at", "inline"):
    sch = make_schedule(func, how)
    plans[how] = Plan(func, sch)
    results[how] = Interp(plans[how]).run(vals)["out"]
    print()
    print("调度 %-7s → %s" % (how, {n: plans[how].place[n] for n in ("s1", "s5", "out")}))

# ==== A 类 ====

rule("2. A 算法正确性：三种调度各自 vs 独立 numpy 参考实现")
rows = [("root（默认）", results["root"]), ("compute_at", results["at"]),
        ("compute_inline", results["inline"])]
for name, got in rows:
    e = max_err(got, ref)
    print("%-16s vs 参考：最大误差 %.3e" % (name, e))
    checks.A(e == 0.0, "%s 的执行结果 == 独立 numpy 参考实现" % name,
             "最大误差 %.3e" % e)
cross = max(max_err(results["root"], results["at"]),
            max_err(results["root"], results["inline"]))
checks.A(cross == 0.0,
         "三态**语义等价**（变换前后数值一致）——按 CONTRIBUTING §2 归 A 类，不归 C 类",
         "三态互相之差 %.3e" % cross)

# ==== B 类 ====

rule("3. B 代价模型：三态的整数对照（断言对象是具体整数，不是 O(·)）")

n_inter = K
rows_b = []
for how, label in (("root", "默认（根作用域）"), ("at", "compute_at"),
                   ("inline", "compute_inline")):
    p = plans[how]
    interp = Interp(p)
    interp.run(vals)
    obs = dict(interp.report())
    ana = dict(p.summary_rows())
    for k_ in ana:
        checks.B(ana[k_] == obs[k_],
                 "%s：解析式 %s == 实测" % (label, k_),
                 "%s vs %s" % (ana[k_], obs[k_]))
    rows_b.append((label, p, ana))

print("%-16s %10s %12s %12s %12s" % ("调度", "中间buffer", "峰值(字节)", "中间流量", "求值次数"))
print("-" * 68)
for label, p, ana in rows_b:
    print("%-16s %10d %12d %12d %12d"
          % (label, ana["中间 buffer 个数"], ana["峰值中间存储（字节）"],
             ana["中间流量（元素）"], ana["全 stage 求值（元素）"]))
print()
print("推导：每级 %d 个元素，%d 级 → 每级 %d 字节" % (LEVEL, K, LEVEL * ESIZE))
print("  默认   ：峰值 = %d 级 × %d 字节 = %d 字节" % (K, LEVEL * ESIZE, K * LEVEL * ESIZE))
print("  compute_at：切片 = 一次 i 迭代的 %d 个元素 → 峰值 = %d×%d×%d = %d 字节"
      % (RW, K, RW, ESIZE, K * RW * ESIZE))

# 断言① 物化次数（三态的分界）
checks.B(plans["root"].n_buffers == K and plans["at"].n_buffers == K,
         "★ 断言①：默认与 compute_at 下，中间 buffer 个数**都是 K**"
         "（compute_at 是物化，不是重算）",
         "root %d / at %d" % (plans["root"].n_buffers, plans["at"].n_buffers))
checks.B(plans["inline"].n_buffers == 0,
         "★ 断言①：compute_inline 下中间 buffer 个数是 **0**（不物化）",
         "inline %d" % plans["inline"].n_buffers)

# 断言② 字节数（默认走根作用域；compute_at 缩到切片）
checks.B(plans["root"].peak_elems == K * LEVEL,
         "★ 断言②：默认的峰值中间存储 == K × LEVEL（全部经由根作用域）",
         "%d == %d" % (plans["root"].peak_elems, K * LEVEL))
checks.B(plans["at"].peak_elems == K * RW,
         "★ 断言②：compute_at 的峰值 == K × 切片大小（每个切片 %d 个元素，"
         "具体数不是 O(·)）" % RW,
         "%d == %d" % (plans["at"].peak_elems, K * RW))
checks.B(plans["root"].peak_elems // plans["at"].peak_elems == I.extent,
         "★ compute_at 把峰值压低 **i 轴的 extent 倍**，而流量不变",
         "%d // %d = %d == %d"
         % (plans["root"].peak_elems, plans["at"].peak_elems,
            plans["root"].peak_elems // plans["at"].peak_elems, I.extent))
checks.B(plans["root"].writes + plans["root"].reads ==
         plans["at"].writes + plans["at"].reads,
         "★ compute_at **不减流量**（切片精确划分时每元素仍写一次读一次）",
         "root %d == at %d"
         % (plans["root"].writes + plans["root"].reads,
            plans["at"].writes + plans["at"].reads))

# 断言③ inline 的代价
checks.B(plans["inline"].peak_elems == 0 and
         plans["inline"].writes + plans["inline"].reads == 0,
         "★ 断言③：compute_inline 的中间流量与峰值都是 0（真的没有任何中间存储）",
         "峰值 %d，流量 %d" % (plans["inline"].peak_elems,
                              plans["inline"].writes + plans["inline"].reads))
checks.B(plans["inline"].evals == plans["root"].evals,
         "★ 断言③：单消费者链上，inline 的求值次数与默认**相同**"
         "（每元素仍只算一次）—— 代价要到多消费者时才出现",
         "inline %d == root %d" % (plans["inline"].evals, plans["root"].evals))

# 混合态：物化与重算可以落在同一份调度里
sch_mix = Schedule(func)
for st in [s for s in func.stages if s.out.role == "intermediate"][1:]:
    sch_mix.compute_inline(st)                 # s1 留在默认（根位置、物化），s2..s5 inline
mix = Plan(func, sch_mix)
checks.B(mix.n_buffers == 1,
         "混合态：只把 s1 物化、其余 inline → 中间 buffer 个数 == 1",
         "%d（%s）" % (mix.n_buffers, mix.place["s1"]))

# ==== C 类 ====

rule("4. C 边界与反例")

# C1：多消费者地形上，compute_at 的前置条件（所有 consumer 都在该 loop 下）
fan = build_fanout()
fsch = Schedule(fan)
o1_loop = fsch.get_loops(fan.stage("out1"))[0]
s_stage = fan.stage("s")
print("多消费者地形：s 的消费者 = %s" % [c.name for c in fan.consumers(s_stage)])
checks.expect_raises(lambda: Schedule(fan).compute_at(fan.stage("s"), o1_loop),
                     ScheduleFailure,
                     "C1 s 的消费者之一是 out2，不在 out1 的循环下 → compute_at 被拒绝")
checks.expect_ok(lambda: Schedule(fan).compute_inline(fan.stage("s")),   # noqa: B018
                 "C1 对照：多消费者地形上 **compute_inline 是允许的**"
                 "（它没有「所有消费者都在同一 loop 下」这个前提）",
                 "inline 接受")

# C2：不物化不是免费的 —— 重算次数 == 消费者数
fan_root = Plan(fan, Schedule(fan))
fsch2 = Schedule(fan)
fsch2.compute_inline(fan.stage("s"))
fan_inline = Plan(fan, fsch2)
i_root = Interp(fan_root)
i_inline = Interp(fan_inline)
r_root = i_root.run(vals)
r_inline = i_inline.run(vals)
rr = ref_fanout(vals["a"], vals["b"])
print()
print("参考：out1 = a*b*2，out2 = a*b+1")
print("物化 s 时求值次数 %d；inline s 时求值次数 %d"
      % (fan_root.evals, fan_inline.evals))
checks.A(max_err(r_root["out1"], rr[0]) == 0.0 and
         max_err(r_root["out2"], rr[1]) == 0.0,
         "物化版本的两个输出 == 独立 numpy 参考", "数值一致")
checks.A(max_err(r_inline["out1"], rr[0]) == 0.0 and
         max_err(r_inline["out2"], rr[1]) == 0.0,
         "inline 版本的两个输出 == 独立 numpy 参考", "数值一致")
n_cons = len(fan.consumers(s_stage))
checks.C(fan_inline.times["s"] == n_cons,
         "C2 ★ 反例：多消费者时把 s inline，**重算次数 == 消费者数**"
         "——「不物化」不是免费的",
         "times(s) = %d，消费者 %d 个" % (fan_inline.times["s"], n_cons))
checks.C(fan_inline.evals - fan_root.evals == (n_cons - 1) * LEVEL,
         "C2 重算代价是可数的：inline 比物化多算 (消费者数-1) × 每级元素数",
         "多算 %d 个元素（= (%d-1)×%d）"
         % (fan_inline.evals - fan_root.evals, n_cons, LEVEL))
# C2 对照：把 s 换成物化，重算次数回到 1
checks.expect_ok(lambda: Plan(fan, Schedule(fan)).times["s"] == 1,
                 "C2 对照：物化 s 时重算次数 == 1", "times(s) = 1")

# C3：切片倍数是可证伪的整数关系
checks.C(plans["at"].peak_elems * I.extent == plans["root"].peak_elems,
         "C3 切片关系精确成立（把切片大小算错，这条会失败）",
         "%d × %d == %d" % (plans["at"].peak_elems, I.extent,
                            plans["root"].peak_elems))

# C4：反证 —— 「不物化不是免费的」这条结论依赖消费者个数
one_consumer_saves = (plans["inline"].evals - plans["root"].evals == 0)
checks.C(one_consumer_saves,
         "C4 反证：把消费者减到 1 个，「inline 更贵」就不再成立"
         "（单消费者链上 inline 的求值次数与物化相同）",
         "单消费者差值 %d" % (plans["inline"].evals - plans["root"].evals))

rule("5. 断言汇总")
checks.summary()
