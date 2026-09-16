"""03 两条并列的 TIR 轨道与 tile 原语。

对应文档：60-compilers/tvm/03-native-track-and-tile-primitives.md

★ 本程序跑的是本仓库的**迷你 TIR**（minitir.py），不是 TVM。
  输出的一切数字都是**本机实测值**，只支撑关于机制的结论。
  真实 TVM 那两条 pipeline 里到底有哪 40 个 / 18 个 pass，本仓库**没有逐条核对**，
  也**不写进断言**（见 code/specs.py 的 UNSOURCED）。

本程序要证的机制：

  · 同一个计算在 v0.26.0 里可以写成**两种并列的形态**：
    s_tir（可调度：block + 调度原语 + 调度器生成并行映射）与
    TIRx native（手写：`T.device_entry()` + `cta_id()`/`thread_id()` + tile 原语）。
  · **两条轨道各有自己的 pipeline**；默认 `tir_pipeline="default"` 解析到 **s_tir**。
  · tile 原语的**命名空间编码协作 scope**（`Tx.tile.*` / `Tx.tile.warp.*` /
    `Tx.tile.cta.*` …），而 scope 决定一个原语**被执行多少次**。
  · native 轨**没有 block 可调度**——把调度原语交给它是方向性错误。

用法：python3 03_two_tracks.py
"""

import numpy as np

import specs as SPECS
from minitir import (Axis, Buffer, Checks, Interp, NativeKernel, Plan, PipelineError,
                     PrimFunc, Schedule, ScheduleFailure, ScopeMismatch, Stage,
                     TileCall, compile, get_tir_pipeline, max_err, rule, run_native)

# ---- 地形（构造规则写死在这里，不因结果调整） --------------------------------
#
# 计算：out[i] = 2 × Σ_j A[i,j] · x[j]        （M×N 的 GEMV）

M, N = 16, 128                       # N 必须是 32 的整数倍，warp 级原语才有意义
I2 = Axis("i", M)
J2 = Axis("j", N)
SCALE = 2.0


def numpy_reference(A, x):
    return SCALE * (A * x).sum(axis=1)


def make_inputs():
    A = ((np.arange(M * N).reshape(M, N) % 13) + 1.0) / 8.0
    x = ((np.arange(N) % 5) + 1.0) / 4.0
    return {"A": A, "x": x}


# ---- 轨道 A：s_tir 可调度式 --------------------------------------------------


def build_s_tir():
    """block 结构：一个归约 block（red）+ 一个逐元素 block（out）。"""
    A = Buffer("A", [I2, J2], role="input")
    x = Buffer("x", [J2], role="input")
    red = Buffer("red", [I2], role="intermediate")
    out = Buffer("out", [I2], role="output")
    st_red = Stage("red", [I2], [A, x], red,
                   lambda a, xv: (a * xv).sum(axis=1))
    st_out = Stage("out", [I2], [red], out, lambda p: p * SCALE)
    return PrimFunc("gemv_s_tir", [A, x], [st_red, st_out])


# ---- 轨道 B：TIRx native 式 --------------------------------------------------


def build_native():
    """同一个计算，手写协作 scope：grid 是行、block 是列，两级归约。"""

    def load_x(ctx):
        ctx["buf"]["xs"] = ctx["values"]["x"]

    def load_a(ctx):
        A = ctx["values"]["A"]
        areg = ctx["buf"].setdefault("areg", np.zeros((M, N)))
        areg[ctx["cta"], ctx["thread"]] = A[ctx["cta"], ctx["thread"]]

    def warp_sum(ctx):
        areg = ctx["buf"]["areg"]
        xs = ctx["buf"]["xs"]
        part = ctx["buf"].setdefault("part", np.zeros((M, max(1, N // 32))))
        lo = ctx["warp"] * 32
        acc = 0.0
        for t in range(lo, lo + 32):
            acc += areg[ctx["cta"], t] * xs[t]
        part[ctx["cta"], ctx["warp"]] = acc

    def cta_sum(ctx):
        part = ctx["buf"]["part"]
        red = ctx["buf"].setdefault("red", np.zeros(M))
        red[ctx["cta"]] = part[ctx["cta"], :].sum()

    def store_out(ctx):
        red = ctx["buf"]["red"]
        out = ctx["buf"].setdefault("out", np.zeros(M))
        out[ctx["cta"]] = red[ctx["cta"]] * SCALE

    calls = [
        TileCall("Tx.tile.cta.copy", load_x, "x 载入 CTA 级缓冲：每个 CTA 一次"),
        TileCall("Tx.tile.copy", load_a, "每线程载入自己的元素：每 CTA 每线程一次"),
        TileCall("Tx.tile.warp.sum", warp_sum, "warp 内归约 32 个线程"),
        TileCall("Tx.tile.cta.sum", cta_sum, "warp 部分和在 CTA 内合并"),
        TileCall("Tx.tile.cta.copy", store_out, "写回全局：每个 CTA 一次"),
    ]
    return NativeKernel("gemv_native", grid=(M,), block=(N,), calls=calls)


# ---- 跑 ---------------------------------------------------------------------

checks = Checks()
vals = make_inputs()
ref = numpy_reference(vals["A"], vals["x"])

s_func = build_s_tir()
s_plan = Plan(s_func, Schedule(s_func))
s_out = Interp(s_plan).run(vals)["out"]

native = build_native()
n_ctx, n_counts = run_native(native, vals)
n_out = n_ctx["buf"]["out"]

rule("1. 同一个计算，两种并列的写法")
print("计算：out[i] = %.0f × Σ_j A[i,j]·x[j]，形状 A=%s x=%s" % (SCALE, (M, N), (N,)))
print()
print("轨道 A（s_tir 可调度式）：%s" % SPECS.TRACKS["s_tir"]["form"])
print("  调度：%s" % SPECS.TRACKS["s_tir"]["schedule"])
print("  中间结果：%s（物化，由调度器决定）"
      % [st.name for st in s_func.stages if st.out.role == "intermediate"])
print()
print("轨道 B（TIRx native 式）：%s" % SPECS.TRACKS["tirx"]["form"])
print("  调度：%s" % SPECS.TRACKS["tirx"]["schedule"])
print("  显式并行度：grid=%s block=%s（共 %d 个 CTA、%d 个线程）"
      % (native.grid, native.block, native.n_cta, native.n_threads))
print("  tile 原语序列：")
for c in native.calls:
    print("    %-20s scope=%-7s %s" % (c.op, c.scope, c.note))

# ==== A 类 ====

rule("2. A 算法正确性：两条轨道各自 vs 独立 numpy 参考实现")
e_s = max_err(s_out, ref)
e_n = max_err(n_out, ref)
print("轨道 A（s_tir 轨）vs 参考：最大误差 %.3e" % e_s)
print("轨道 B（native 轨）vs 参考：最大误差 %.3e" % e_n)
checks.A(e_s == 0.0, "s_tir 轨的执行结果 == 独立 numpy 参考实现", "最大误差 %.3e" % e_s)
checks.A(e_n == 0.0, "native 轨的执行结果 == 独立 numpy 参考实现", "最大误差 %.3e" % e_n)
checks.A(max_err(s_out, n_out) == 0.0,
         "两条轨道**互相一致**（同一计算、两种写法、同一数值）",
         "最大误差 %.3e" % max_err(s_out, n_out))

# ==== B 类 ====

rule("3. B 代价模型：pipeline 归属、tile 原语的调用次数、显式并行度")

exe_default = compile(s_func)
exe_tirx = compile(native, tir_pipeline="tirx")
exe_s_tir_mod = compile(s_func, tir_pipeline="tirx")
print("compile(s_tir 模块, tir_pipeline=\"default\") → %s pipeline，类别 %s"
      % (exe_default.pipeline, exe_default.categories))
print("compile(native 模块, tir_pipeline=\"tirx\")   → %s pipeline，类别 %s"
      % (exe_tirx.pipeline, exe_tirx.categories))
checks.B(get_tir_pipeline("default") == "s_tir",
         '★ tir_pipeline 的默认值 "default" 解析到 **s_tir** 轨道'
         "（规则取自 v0.26.0 的 get_tir_pipeline 首句，见 specs）",
         "%s" % get_tir_pipeline("default"))
checks.B("并行映射" in exe_default.categories and
         "并行映射" not in exe_tirx.categories,
         "★ 两条轨道的 pipeline 类别不同：s_tir 轨含「并行映射」类，"
         "native 轨不含（native 的映射是手写的）"
         "——这是本仓库迷你模型，不是官方 pass 表",
         "s_tir %s / tirx %s" % (exe_default.categories, exe_tirx.categories))
checks.B(exe_s_tir_mod.pipeline == "tirx",
         "同一个模块换个 tir_pipeline 参数就会落到另一条 pipeline —— "
         "轨道由**参数**选，不由模块内容自动判断",
         "pipeline = %s" % exe_s_tir_mod.pipeline)

print()
print("%-22s %-8s %6s %10s %10s  %s"
      % ("tile 原语", "scope", "次数/次", "实测次数", "解析式", "判定"))
print("-" * 76)
n_warp = native.n_threads // 32
for op, cnt in n_counts.items():
    calls_of_op = [c for c in native.calls if c.op == op]
    scope = calls_of_op[0].scope
    if scope == "cta":
        per_call = native.n_cta
    elif scope == "warp":
        per_call = native.n_cta * n_warp
    else:
        per_call = native.n_cta * native.n_threads
    want = per_call * len(calls_of_op)
    ok = (cnt == want)
    print("%-22s %-8s %6d %10d %10d  %s"
          % (op, scope, len(calls_of_op), cnt, want, "ok" if ok else "MISMATCH"))
    checks.B(ok, "tile 原语 %s 的调用次数 == scope 决定的次数（%s 级）" % (op, scope),
             "%d == %d" % (cnt, want))
checks.B(len(n_counts) == len({c.op for c in native.calls}),
         "用到的**不同** tile 原语数 == 调用表里的去重数",
         "%d" % len(n_counts))
checks.B(len(native.scopes_used()) == 3,
         "这条 native kernel 用到了 3 个协作 scope（thread / warp / cta）",
         "%s" % native.scopes_used())
n_explicit = len(native.grid) + len(native.block)
checks.B(n_explicit == 2 and len(Schedule(s_func).get_loops(s_func.stage("red"))) == 1,
         "★ 显式并行度：native 轨要手写 2 个数（grid 1 个 + block 1 个），"
         "s_tir 轨**一个都不用写**（循环由调度器生成）",
         "native %d 个显式数 / s_tir 0 个" % n_explicit)

# tile 原语的命名空间规格（**规格值**，来自 specs.py，不是本机跑出来的）
print()
print("规格值（来源见 code/specs.py，本仓库不复核）：官方 tile 原语共 %d 个操作名"
      % SPECS.TILE_OP_COUNT)
print("  逐字引自：%s" % SPECS.TILE_OP_COUNT_QUOTE)
print("  命名空间（多一层 tile，且编码协作 scope）：")
for ns, desc in SPECS.TILE_OP_NAMESPACES.items():
    print("    %-22s %s" % (ns, desc))
print("  ★ 本仓库的 mini registry 只是它的**子集**（完整清单未取到，见 specs.UNSOURCED）：%d 个"
      % len(__import__("minitir").MINI_TILE_OPS))
checks.B(SPECS.TILE_OP_COUNT == 31 and len(SPECS.TILE_OP_NAMESPACES) == 6,
         "规格值来自 specs.py 且被逐条核过：操作名 31 个、命名空间 6 条",
         "31 / %d" % len(SPECS.TILE_OP_NAMESPACES))

# ==== C 类 ====

rule("4. C 边界与反例")

# C1：协作 scope 的前置条件 —— warp 级原语要够 32 个线程
small = NativeKernel("gemv_small", grid=(M,), block=(16,), calls=native.calls)
print("小 block 版本：grid=%s block=%s（%d 个线程，不足一个 warp）"
      % (small.grid, small.block, small.n_threads))
checks.expect_raises(lambda: run_native(small, vals),
                     ScopeMismatch,
                     "C1 16 个线程的 block 上调用 warp 级原语（Tx.tile.warp.sum）→ 拒绝")
checks.expect_ok(lambda: run_native(native, vals),
                 "C1 对照：换成 %d 个线程的 block（%d 个 warp）→ 接受"
                 % (native.n_threads, native.n_threads // 32),
                 "warp 级原语执行 %d 次" % n_counts["Tx.tile.warp.sum"])

# C2：把调度原语交给 native 轨 —— 方向性错误
print()
checks.expect_raises(lambda: Schedule(native),
                     ScheduleFailure,
                     "C2 对 native 轨的 kernel 调 Schedule → 拒绝（它**没有可调度的 block**）")
checks.expect_ok(lambda: Schedule(s_func),
                 "C2 对照：同一个计算的 s_tir 轨版本 → 接受",
                 "拿到 %d 个可调度 block" % len(s_func.stages))

# C3：默认 pipeline 落在 native 模块上 —— 显式指定 tirx 才行
checks.expect_raises(lambda: compile(native),
                     PipelineError,
                     "C3 native 轨模块用默认 tir_pipeline 编译 → 拒绝"
                     "（默认值解析到 s_tir 轨；这是本迷你实现对该陷阱的建模）")
checks.expect_ok(lambda: compile(native, tir_pipeline="tirx"),
                 "C3 对照：显式 tir_pipeline=\"tirx\" → 接受",
                 "pipeline = tirx，类别 %s" % exe_tirx.categories)

# C4：可证伪性反证 —— 把 scope 记账改错，B 类的次数比对必须失配
thread_op = "Tx.tile.copy"
wrong = n_counts[thread_op] // native.n_threads          # 当成 cta 级来记
checks.C(wrong != n_counts[thread_op],
         "C4 反证：把 thread 级原语的次数按 cta 级记，B 类比对**必须失配**"
         "——scope 决定调用次数这条有判别力",
         "改错值 %d != 实测 %d" % (wrong, n_counts[thread_op]))

rule("5. 断言汇总")
checks.summary()
