"""01 图捕获：帧入口决策、graph break、以及「切出来的那段跑普通 Python」。

对应文档：60-compilers/inductor/01-graph-capture.md

★ 本程序跑的是**本仓库的迷你追踪器**（minigraph.MiniDynamo），不是 TorchDynamo。
  输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。

★ **自曝实现方式**（计划 §3 的 A9b 要求）：
  迷你追踪器用标准库 `sys.settrace` 的 **`call` 事件**做「帧入口决策」，
  用代理对象的算子重载记录节点，用**重放**模拟 resume function。
  **真实的 TorchDynamo 用的不是这些**——它装的是 CPython 的 **PEP 523 帧求值回调**
  （`_PyInterpreterState_SetEvalFrameFunc`，见 specs.PEP523_FILE），
  靠字节码分析建图，并在切图点生成 resume function。
  两者在「每个帧入口做一次决策」「切出的区域跑普通 Python」这两点是同构的，
  **实现层次与触发时机完全不同**。

用法：python3 01_graph_capture.py
"""

import numpy as np

import minigraph as M
import specs as SPECS
from minigraph import Checks

# ---------------------------------------------------------------------------
# 地形 A：不切图的函数（对照组）
# ---------------------------------------------------------------------------


def forward_plain(x, w):
    a = x * 2.0
    b = a + w
    c = b.relu()
    return c


def ref_plain(x, w):
    """**独立参考实现**：纯 numpy，不经代理、不进图。"""
    a = x * 2.0
    b = a + w
    c = np.maximum(b, 0.0)
    return c


def ref_plain_wrong(x, w):
    """故意写错的参考实现：给 A 类断言做可证伪性反证。"""
    a = x * 2.0
    b = a + w
    return np.maximum(b, 0.0) * 2.0


# ---------------------------------------------------------------------------
# 地形 B：数据依赖控制流 → 切图
# ---------------------------------------------------------------------------


def forward_with_break(x, w):
    a = x * 2.0
    s = a.sum()
    if s.item() > 0.0:          # ★ 数据依赖控制流：切图
        b = a + w
    else:
        b = a - w
    return b


def ref_break(x, w):
    """独立参考实现，同时返回断点处**存活的那个中间值** a。"""
    a = x * 2.0
    s = a.sum()
    b = a + w if float(s) > 0.0 else a - w
    return a, b


# ---------------------------------------------------------------------------
# 地形 C：print / logging
# ---------------------------------------------------------------------------


def forward_with_print(x):
    a = x * 2.0
    M.mprint(a)                 # 未登记 → 切图；登记 → 重排，不切图
    return a + 1.0


# ---------------------------------------------------------------------------
# 地形 D：原地修改（mutation）
# ---------------------------------------------------------------------------


def forward_inplace(x, w):
    a = x * 2.0
    a.add_(w)                   # ★ 原地修改**不是**切图触发条件
    return a


INPUT_X = np.array([[0.5, -1.0, 2.0],
                    [1.5, 0.25, -0.5]], dtype=np.float32)
INPUT_W = np.array([1.0, 2.0, 3.0], dtype=np.float32)


def capture(fn, inputs, reorderable=()):
    d = M.MiniDynamo(reorderable_logging=reorderable)
    d.register(fn)
    return d.capture(fn, inputs)


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------

checks = Checks()

M.rule("1. 帧入口决策：钩子看到每个帧，只放行登记过的那些")
print("机制：%s（%s）" % (SPECS.PEP523_SYMBOL, SPECS.PEP523_FILE))
print("官方原文：「%s」" % SPECS.PEP523_DOC_QUOTE)
print("⚠ 本仓库用的是 sys.settrace 的 call 事件**模拟**这个决策，见文件头。")
print("%s" % SPECS.SYS_MONITORING)
print()

r_plain = capture(forward_plain, [INPUT_X, INPUT_W])
print("地形 A（不切图）：%s" % r_plain.summary())
print("  被放行的帧：%s" % [n for n, ok in r_plain.frame_entries if ok])

M.rule("2. 子图结构")
for i, g in enumerate(r_plain.subgraphs):
    print("子图 %d：%s" % (i, g))
    for n in g.nodes:
        print("   %-6s %-12s reads=%s writes=%s"
              % (n.buf, n.op, [str(d) for d in n.reads], n.writes))

# ---------------------------------------------------------------------------
# A 类：解释器 vs 独立参考实现
# ---------------------------------------------------------------------------

M.rule("3. A 算法正确性：图解释器 vs 纯 numpy 参考实现")

g_plain = r_plain.subgraphs[0]
env0 = {n.buf: n.value for n in g_plain.nodes if n.kind == "placeholder"}
got = M.eval_graph(g_plain, dict(env0))[g_plain.outputs[0].buf]
want = ref_plain(INPUT_X, INPUT_W)
print("解释器输出：\n%s" % got)
print("参考实现  ：\n%s" % want)
checks.A(M.max_abs_err(got, want) == 0.0,
         "地形 A：图解释器 == 纯 numpy 参考实现",
         "最大绝对误差 %.3e" % M.max_abs_err(got, want))

# 可证伪性反证：把参照实现故意改错，同一条判据必须失败
bad = ref_plain_wrong(INPUT_X, INPUT_W)
checks.C(M.max_abs_err(got, bad) > 0,
         "反证：把参考实现故意改错，A 类判据立刻报不一致",
         "错误参考实现的最大绝对误差 %.3e" % M.max_abs_err(got, bad))

# ---------------------------------------------------------------------------
# C1：真的制造 graph break
# ---------------------------------------------------------------------------

M.rule("4. C1 边界与反例：数据依赖控制流真的把图切断")

r_brk = capture(forward_with_break, [INPUT_X, INPUT_W])
print("地形 B（.item() 触发）：%s" % r_brk.summary())
for i, g in enumerate(r_brk.subgraphs):
    print("  子图 %d：%s  %s" % (i, g, [n.op for n in g.nodes]))
for b in r_brk.breaks:
    print("  切图点：%s" % b)
print("候选触发条件（官方列举，非穷举表）：")
for t in SPECS.GRAPH_BREAK_TRIGGERS:
    print("  · %s" % t)

checks.C(r_brk.n_subgraphs == 2,
         "C1：命中 .item() 后图被切成 2 张子图（**不是静默产出一张错的图**）",
         "子图数 %d" % r_brk.n_subgraphs)
checks.C(len(r_brk.breaks) == 1 and "item" in r_brk.breaks[0].reason,
         "C1：切图原因确实是数据依赖控制流",
         r_brk.breaks[0].reason)

# —— S11 要求的可证伪性反证：把「切图检测」换成不检测的版本，同一条判据必须失败 ——
_orig_item = M.P.item


def capture_without_break_detection():
    """故意写坏的追踪器：.item() 照常返回，不切图。"""
    M.P.item = lambda self: float(self._val)
    try:
        return capture(forward_with_break, [INPUT_X, INPUT_W])
    finally:
        M.P.item = _orig_item


silent = capture_without_break_detection()
print()
print("  —— 可证伪性反证：把切图检测换成「不检测」的版本 ——")
print("  坏追踪器给出：%s" % silent.summary())
checks.C(silent.n_subgraphs == 1,
         "反证：不检测 .item() 的追踪器只会产出 1 张子图，C1 立刻判失败",
         "坏追踪器子图数 %d（正确实现是 %d）" % (silent.n_subgraphs,
                                            r_brk.n_subgraphs))

# —— 断点之后的那张子图：它的输入是「断点处存活的值」——
g2 = r_brk.subgraphs[1]
mids = [n for n in g2.nodes if n.kind == "placeholder" and n.buf.startswith("mid")]
print()
print("切出来的那段（断点前）跑的是普通 Python：重放 %d 轮，第 1 轮在 .item() 处中断；"
      % r_brk.passes)
print("第 2 轮里前 %d 个算子走 eager，之后重新开始记录。" % r_brk.breaks[0].op_index)
print("子图 2 的输入是断点处存活的中间值：%s" % [n.buf for n in mids])

def placeholder_env(graph, ref_inputs, ref_mids):
    """给子图的占位符配值：`mid*` 取**参考实现**算出的中间值，`arg*` 取参考输入。

    这一步是 A 类「独立参照」的关键：输入不取自被测追踪器留下的 eager 值，
    而取自另一条实现路径（ref_break）。
    """
    env = {}
    for n in graph.nodes:
        if n.kind != "placeholder":
            continue
        if n.buf.startswith("mid"):
            env[n.buf] = ref_mids[n.buf]
        else:
            env[n.buf] = ref_inputs[n.buf]
    return env


REF_INPUTS = {"arg0": INPUT_X, "arg1": INPUT_W}
a_ref, b_ref = ref_break(INPUT_X, INPUT_W)
env2 = placeholder_env(g2, REF_INPUTS, {mids[0].buf: a_ref})
got2 = M.eval_graph(g2, env2)[g2.outputs[0].buf]
ref_a, ref_b = ref_break(INPUT_X, INPUT_W)
checks.A(M.max_abs_err(got2, ref_b) == 0.0,
         "地形 B：子图 2 的解释结果 == 参考实现（输入取自参考实现自己的中间值）",
         "最大绝对误差 %.3e" % M.max_abs_err(got2, ref_b))
checks.A(M.max_abs_err(mids[0].value, ref_a) == 0.0,
         "断点处存活的中间值 == 参考实现的同一个中间值",
         "最大绝对误差 %.3e" % M.max_abs_err(mids[0].value, ref_a))

# ---------------------------------------------------------------------------
# §4 的核心：print/logging 与 mutation 的区别
# ---------------------------------------------------------------------------

M.rule("5. §4 的核心：有副作用就 break 是**错的**")

r_p = capture(forward_with_print, [INPUT_X])
print("print 未登记：%s" % r_p.summary())
for b in r_p.breaks:
    print("  切图点：%s" % b)

r_p2 = capture(forward_with_print, [INPUT_X], reorderable=("print",))
print("print 已登记进 reorderable_logging_functions：%s" % r_p2.summary())
print("  重排出来的 effect 节点：%s"
      % [(e.op, [a.shape for a in e.attrs["args"]]) for e in r_p2.subgraphs[0].effects])
print("  默认值：%s（%s）" % (SPECS.REORDERABLE_LOGGING_DEFAULT,
                          "torch/_dynamo/config.py"))

r_m = capture(forward_inplace, [INPUT_X, INPUT_W])
print()
print("原地修改（add_）：%s" % r_m.summary())
print("  记录到的算子：%s" % [n.op for n in r_m.subgraphs[0].nodes])

checks.C(r_p.n_subgraphs == 2,
         "C1 补充：**未登记**的 print 触发切图",
         "子图数 %d" % r_p.n_subgraphs)
checks.expect_ok(lambda: r_p2.n_subgraphs == 1,
                 "对照：**登记后**的 print 被重排，不再切图",
                 "子图数 %d，effect 节点 %d 个"
                 % (r_p2.n_subgraphs, len(r_p2.subgraphs[0].effects)))
checks.expect_ok(lambda: r_m.n_subgraphs == 1,
                 "对照：原地修改（add_）**不**切图——「有副作用就 break」是错的",
                 "子图数 %d，算子 %s" % (r_m.n_subgraphs,
                                     [n.op for n in r_m.subgraphs[0].nodes]))
checks.A(any(n.op == "add_" for n in r_m.subgraphs[0].nodes),
         "原地修改被正常记录成算子（而不是被丢掉）",
         "算子序列 %s" % [n.op for n in r_m.subgraphs[0].nodes])

# ---------------------------------------------------------------------------
# B 类：与文档 §5 的表格逐行核对
# ---------------------------------------------------------------------------

M.rule("6. B 代价模型：与文档 §5 的表格逐行核对")

plain_ops = [n for n in g_plain.nodes if n.kind not in ("placeholder",)]
b_rows = [
    ("地形 A · 图输入数", len(env0), 2),
    ("地形 A · 被计数的算子数", len(plain_ops), 3),
    ("地形 A · 子图数", r_plain.n_subgraphs, 1),
    ("地形 A · 切图次数", len(r_plain.breaks), 0),
    ("地形 B · 子图数", r_brk.n_subgraphs, 2),
    ("地形 B · 切图次数", len(r_brk.breaks), 1),
    ("地形 B · 重放轮数", r_brk.passes, 2),
    ("地形 B · 子图 1 的算子数（不含占位符）",
     len([n for n in r_brk.subgraphs[0].nodes if n.kind != "placeholder"]), 2),
    ("地形 B · 子图 2 的 mid 占位符数", len(mids), 1),
    ("地形 C · 未登记 print 的子图数", r_p.n_subgraphs, 2),
    ("地形 C · 登记 print 后的子图数", r_p2.n_subgraphs, 1),
    ("地形 C · 登记 print 后的 effect 数", len(r_p2.subgraphs[0].effects), 1),
    ("地形 D · 原地修改的子图数", r_m.n_subgraphs, 1),
]
print("%-34s %8s %8s" % ("项", "本机值", "文档值"))
print("-" * 54)
for name, got_v, want_v in b_rows:
    ok = (got_v == want_v)
    print("%-34s %8s %8s  %s" % (name, got_v, want_v, "ok" if ok else "MISMATCH"))
    checks.B(ok, "文档表：%s == %s" % (name, want_v), "本机 %s" % got_v)

checks.A(len(r_plain.frame_entries) == 1,
         "帧入口钩子在 1 轮捕获里只放行 1 次登记帧",
         "放行 %d 次 / 跳过 %d 次"
         % (len(r_plain.frame_entries), r_plain.skipped_frames))

M.rule("7. 断言汇总")
checks.summary()
