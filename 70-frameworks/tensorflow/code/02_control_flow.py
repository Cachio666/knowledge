#!/usr/bin/env python3
"""02 控制流与 XLA：**固化分支，还是把两条都放进图**。

对应文档：70-frameworks/tensorflow/02-control-flow-and-xla.md

★★ 本程序跑的是**本仓库的迷你 TF 追踪器**（`minitf.py`），**不是 TensorFlow** ★★
  本机没有任何 TF 环境。所有数字只支撑**机制**结论；关于真实 TF 的行为断言
  以 `specs.py` 里带 `# 来源：` 的一手 URL 为准。

挂主线（§0.8）：本文处理的是「固化什么」的**控制流那一面**——
  Python `if` 把**当时走到的那条分支**固化进图；`tf.cond` 反过来，**两条都放进图**，
  把「选哪条」推迟到运行时。前者图更小、假设更硬；后者图更大、假设更软。

三类断言（CONTRIBUTING §2：编号程序跑在默认 python3 上，故三类都在程序内部）：
  A —— 与**独立参考实现**比对：分支取值、循环取值都用一个另写的 numpy 参照函数算一遍。
  B —— 代价模型：图为多少节点、追踪几次，都与模型逐项核对。
  C —— 失败行为必须真的跑出来：符号张量取 bool / 迭代必须抛错，
       cond 分支 dtype 不一致必须抛错，含 tf.print 的图必须编不过 XLA。

用法：cd code && make 02_control_flow   （或 make run 全跑）
"""

import sys
import time
import warnings

import numpy as np

import minitf as T
import specs as SPECS

F32 = "float32"


# ---------------------------------------------------------------------------
# 独立参考实现：不看图，直接用 numpy 把「应该等于几」算一遍
# ---------------------------------------------------------------------------

def ref_select(x, threshold, plus, minus):
    """`x > threshold ? x + plus : x - minus` 的 numpy 直接实现。"""
    return np.where(x > threshold, x + plus, x - minus)


def ref_countdown(start, stop, step=1.0):
    """`while v > stop: v -= step` 的 numpy 直接实现（就是一句一句照着写）。"""
    v = np.asarray(start, dtype=np.float32)
    guard = 0
    while bool(v > np.float32(stop)):
        v = v - np.float32(step)
        guard += 1
        if guard > 100000:
            raise RuntimeError("参照实现未收敛")
    return v


# ---------------------------------------------------------------------------
# §1 AutoGraph 的触发条件：条件是 Tensor 才转，否则按 Python 执行
# ---------------------------------------------------------------------------

def section_1():
    T.rule("§1 条件是 Tensor 才转成 tf.cond；否则就是一个 Python 条件")
    print("官方逐字（%s）：" % "site/en/guide/function.ipynb §Conditionals")
    print("  %s" % SPECS.AUTOGRAPH_IF_VERBATIM)
    print("  %s" % SPECS.PYTHON_IF_VERBATIM)
    print()
    print("官方逐字（同页）：")
    print("  %s" % SPECS.COND_BOTH_BRANCHES_VERBATIM)

    hits = {"then": 0, "else": 0}

    def then_branch(x):
        hits["then"] += 1                  # Python 侧副作用：追踪时就会发生
        return T.add(x, 1.0)

    def else_branch(x):
        hits["else"] += 1
        return T.sub(x, 1.0)

    @T.tf_function
    def with_cond(x):
        return T.cond(T.greater(x, 0.0), lambda: then_branch(x),
                      lambda: else_branch(x))

    print()
    print("追踪一次 `tf.cond` 版本（条件 `x > 0` 是**图里的张量**）：")
    with_cond(T.Tensor(np.float32(3.0)))
    print("  追踪计数 = %d" % with_cond.experimental_get_tracing_count())
    print("  两个分支的副作用计数：%s" % hits)
    print("  ★ 注意两点：①**两个分支的 Python 代码都跑了**（副作用各 1 次）；")
    print("             ②图里两个分支的算子**都在**：")
    print(with_cond._function_cache[list(with_cond._function_cache)[0]].graph)
    return with_cond, hits


# ---------------------------------------------------------------------------
# §A 算法正确性
# ---------------------------------------------------------------------------

def section_A(with_cond):
    T.rule("§A 算法正确性：与另写的 numpy 参照逐项比对")
    c = T.Checks()

    xs = [-5.0, -0.5, 0.0, 0.5, 5.0, 123.25]
    worst = 0.0
    for v in xs:
        got = float(with_cond(T.Tensor(np.float32(v))).numpy)
        want = float(ref_select(np.float32(v), np.float32(0.0), 1.0, 1.0))
        worst = max(worst, abs(got - want))
    c.A(worst == 0.0, "cond 的取值与 numpy 参照逐点一致",
        "%d 个输入，max|Δ| = %g" % (len(xs), worst))

    # 循环：与另写的参照比对
    @T.tf_function
    def countdown(x):
        return T.while_loop(lambda v: T.greater(v, 1.0), lambda v: T.sub(v, 1.0), x)

    worst_loop = 0.0
    for start in [1.0, 2.0, 5.0, 17.0, 100.0]:
        got = float(countdown(T.Tensor(np.float32(start))).numpy)
        want = float(ref_countdown(start, 1.0))
        worst_loop = max(worst_loop, abs(got - want))
    c.A(worst_loop == 0.0, "while_loop 的收敛值与 numpy 参照一致",
        "5 个起点，max|Δ| = %g" % worst_loop)

    # 分支的选择确实发生在运行时（同一张图上换输入 → 换分支）
    a = float(with_cond(T.Tensor(np.float32(2.0))).numpy)
    b = float(with_cond(T.Tensor(np.float32(-2.0))).numpy)
    c.A(abs(a - 3.0) < 1e-6 and abs(b + 3.0) < 1e-6,
        "同一张图、不同输入 ⇒ 走了不同分支（分支选择在运行时）",
        "pred=+2 → %.1f（+1 分支）；pred=-2 → %.1f（−1 分支）" % (a, b))
    c.A(with_cond.experimental_get_tracing_count() == 1,
        "上面这批调用**一次都没重追踪**（分支选择不进 TraceType）",
        "追踪计数 = %d" % with_cond.experimental_get_tracing_count())
    return c


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def section_B():
    T.rule("§B 代价模型：图的节点数、追踪次数、XLA 的门槛")
    c = T.Checks()

    # B1 tf.cond 的图 = then + else + 1
    def t1(x):
        return T.add(x, 1.0)

    def t2(x):
        return T.mul(x, 2.0)

    def f1(x):
        return T.sub(x, 1.0)

    def f2(x):
        return T.reduce_sum(x)

    @T.tf_function
    def two_and_two(x):
        return T.cond(T.greater(x, 0.0), lambda: T.add(t1(x), t2(x)),
                      lambda: T.reduce_sum(T.sub(f1(x), f2(x))))

    two_and_two(T.Tensor(np.float32(1.0)))
    cf = list(two_and_two._function_cache.values())[0]
    cond_op = [op for op in cf.graph.ops if op.name == "tf.cond"][0]
    then_sg, else_sg = cond_op.subgraphs["then"], cond_op.subgraphs["else"]
    pred_ops = len([op for op in cf.graph.ops if op.name != "tf.cond"])
    model = pred_ops + 1 + len(then_sg.ops) + len(else_sg.ops)
    c.B(cf.graph.size() == model,
        "cond 图的总节点数 == 谓词算子 + cond 自己 + then + else",
        "实测 %d，模型 %d（谓词 %d + 1 + then %d + else %d）"
        % (cf.graph.size(), model, pred_ops, len(then_sg.ops), len(else_sg.ops)))
    print("    ★ 这张图**两个分支的算子同时存在**——节点数是「两条都算」的结果，")
    print("      而不是「哪条被走到算哪条」。")

    # B2 while_loop 的图**不随迭代次数增长**
    @T.tf_function
    def looped(x):
        return T.while_loop(lambda v: T.greater(v, 1.0), lambda v: T.sub(v, 1.0), x)

    looped(T.Tensor(np.float32(5.0)))
    size_small = list(looped._function_cache.values())[0].graph.size()
    looped(T.Tensor(np.float32(500.0)))
    size_big = list(looped._function_cache.values())[0].graph.size()
    c.B(size_small == size_big,
        "★ while_loop 的图**与迭代次数无关**（5 次与 500 次是同一张图）",
        "init=5 → %d 个节点；init=500 → %d 个节点" % (size_small, size_big))
    c.B(looped.experimental_get_tracing_count() == 1,
        "★ 500 次迭代**也只追踪一次**（body 是符号执行一遍）",
        "追踪计数 = %d" % looped.experimental_get_tracing_count())

    # B3 对照：Python for 循环会**展开**，图随 N 线性增长
    def make_unrolled(n):
        @T.tf_function
        def f(x):
            v = x
            for _ in range(n):            # Python 循环：追踪时展开
                v = T.sub(v, 1.0)
            return v
        return f

    u3, u6 = make_unrolled(3), make_unrolled(6)
    u3(T.Tensor(np.float32(5.0)))
    u6(T.Tensor(np.float32(5.0)))
    s3 = list(u3._function_cache.values())[0].graph.size()
    s6 = list(u6._function_cache.values())[0].graph.size()
    c.B(s6 - s3 == 3, "对照：Python for 循环展开后，图**随 N 线性增长**",
        "N=3 → %d 个节点；N=6 → %d 个节点；差 %d" % (s3, s6, s6 - s3))
    print("    ★ 这就是主线上的取舍：`while_loop` 图小（不固化迭代次数），")
    print("      Python 循环把**具体次数**固化了进去——次数变一次就得重追一次。")

    # B4 XLA 的门槛：整函数必须可编译
    @T.tf_function
    def clean(x):
        return T.cond(T.greater(x, 0.0), lambda: T.add(x, 1.0), lambda: T.sub(x, 1.0))

    clean(T.Tensor(np.float32(1.0)))
    cf_clean = list(clean._function_cache.values())[0]
    T.xla_compile(cf_clean)
    c.B(cf_clean.xla_compiled, "整函数可编译时，XLA 编译成功（本模型）",
        "官方规则逐字见 specs.JIT_COMPILE_DOC：%s"
        % SPECS.JIT_COMPILE_DOC[:64] + " …")

    t0 = time.perf_counter()
    clean(T.Tensor(np.float32(1.0)))
    warm = (time.perf_counter() - t0) * 1e6
    print("    XLA 编译模型跑通；一次 replay ≈ %.1f μs（本仓库迷你实现）" % warm)
    print("    ★ 上限：这里的「编译」只是**本模型的标记**，没有任何代码生成发生；")
    print("      XLA 的 HLO / 融合 / layout 正文在 60-compilers/xla，本文只回链。")
    return c


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def section_C():
    T.rule("§C 边界与反例：每条失败行为都要真的跑出来")
    c = T.Checks()

    T.subsection("C1 tf.cond **两个分支都进图**；Python if 只进一个")
    @T.tf_function
    def both(x):
        return T.cond(T.greater(x, 0.0), lambda: T.add(x, 1.0), lambda: T.sub(x, 1.0))

    both(T.Tensor(np.float32(1.0)))
    g = list(both._function_cache.values())[0].graph
    names = g.all_op_names()
    c.C("tf.add" in names and "tf.subtract" in names,
        "条件在图里（x > 0）时，**两个分支的算子都在同一张图里**",
        "图里出现：%s" % sorted(set(n for n in names if n != "tf.cond")))
    c.C(both.experimental_get_tracing_count() == 1,
        "对照：追踪**只有一次**（不像 Python if 那样每种取值追一次）",
        "追踪计数 = %d" % both.experimental_get_tracing_count())

    T.subsection("C2 Python if + 布尔参数 ⇒ 每种取值**各追一次**，图里各只有一个分支")
    @T.tf_function
    def pyif(x, flag):
        if flag:                          # flag 是 Python 原生值 ⇒ LiteralTraceType<True/False>
            return T.add(x, 1.0)
        return T.sub(x, 1.0)

    x = T.Tensor(np.float32(1.0))
    r_true = float(pyif(x, True).numpy)
    n_after_true = pyif.experimental_get_tracing_count()
    r_false = float(pyif(x, False).numpy)
    n_after_false = pyif.experimental_get_tracing_count()
    c.C(n_after_true == 1 and n_after_false == 2,
        "布尔参数 True / False ⇒ **两次追踪**（每个分支各被固化一次）",
        "追踪计数 %d → %d（对照 tf.cond 的 1）" % (n_after_true, n_after_false))
    c.C(abs(r_true - 2.0) < 1e-6 and abs(r_false - 0.0) < 1e-6,
        "但两种取值的结果都是**对的**——代价是重追踪，不是算错",
        "True → %.1f，False → %.1f" % (r_true, r_false))
    graphs = [cf.graph.all_op_names() for cf in pyif._function_cache.values()]
    one_branch_each = all(
        ("tf.add" in g) ^ ("tf.subtract" in g) for g in graphs)
    c.C(one_branch_each, "★ 每张图里**只有一个分支**（另一个分支的 Python 代码根本没跑）",
        "两张图的算子：%s" % [sorted(n for n in g if n != "tf.cond") for g in graphs])

    T.subsection("C3 对符号张量取 bool / 迭代 —— 必须抛错")
    @T.tf_function
    def bool_trap(x):
        if x:                             # 对 Tracer 取 bool
            return T.add(x, 1.0)
        return T.sub(x, 1.0)

    c.expect_raises(lambda: bool_trap(T.Tensor(np.float32(1.0))),
                    T.OperatorNotAllowedInGraphError,
                    "对符号张量取 bool 抛 OperatorNotAllowedInGraphError")
    print("    ★ 真实 TF 的同名报错逐字（%s）：" % SPECS.src(
        "tensorflow/python/framework/tensor.py", (302, 319)))
    print("      %r + %r" % (SPECS.ERR_TASK_BOOL_CAST, SPECS.ERR_DISALLOW_SUFFIX[:60] + " …"))

    @T.tf_function
    def iter_trap(x):
        total = 0.0
        for v in x:                       # 对 Tracer 迭代
            total = T.add(total, v)
        return total

    c.expect_raises(lambda: iter_trap(T.Tensor(np.zeros(3, dtype=np.float32))),
                    T.OperatorNotAllowedInGraphError,
                    "对符号张量迭代抛 OperatorNotAllowedInGraphError")

    T.subsection("C4 cond 两个分支的 dtype 必须一致")
    @T.tf_function
    def dtype_clash(x):
        return T.cond(T.greater(x, 0.0), lambda: T.add(x, 1.0),
                      lambda: T.cast(x, "int32"))

    c.expect_raises(lambda: dtype_clash(T.Tensor(np.float32(1.0))),
                    T.InvalidArgumentError,
                    "分支 dtype 不一致 ⇒ InvalidArgumentError")

    T.subsection("C5 含 tf.print 的图编不过 XLA（**本模型**的规则）")
    @T.tf_function
    def with_print(x):
        T.tf_print(x, "C5")
        return T.add(x, 1.0)

    with_print(T.Tensor(np.float32(1.0)))
    c.expect_raises(lambda: T.xla_compile(
        list(with_print._function_cache.values())[0]),
        T.InvalidArgumentError,
        "图里有 tf.print ⇒ XLA 编译被拒（本模型）")
    print("    ★⚠ 这条的**规则**有官方来源（整函数必须可编译，否则 InvalidArgumentError），")
    print("      但「tf.print 编不了」是**本仓库模型自己定的**——本环境打不开官方 sharp-corners 页，")
    print("      **不得**把它写成 TF 的事实（见 specs.py 文件头的 UNSOURCED 段）。")

    T.subsection("C6 已废弃的 tf.xla.experimental.compile：仍然能跑，但会告警")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = T.xla_experimental_compile(lambda: 41 + 1)
    msgs = [str(w.message) for w in caught]
    c.C(out == 42 and len(msgs) == 1 and msgs[0] == SPECS.XLA_EXPERIMENTAL_COMPILE_MESSAGE
        and caught[0].category is DeprecationWarning,
        "tf.xla.experimental.compile **仍在但已废弃**：照常返回结果 + 一条 DeprecationWarning",
        "结果 %r；告警类别 %s" % (out, caught[0].category.__name__))
    print("    官方逐字（%s）：" % SPECS.src("tensorflow/python/compiler/xla/xla.py",
                                          (62, 67)))
    print("      %s" % SPECS.XLA_EXPERIMENTAL_COMPILE_DEPRECATION.replace("\n", "\n      "))
    print("    ★ 装饰器第一个参数是 `None` ⇒ **官方没给移除日期**（也不要说它「从某个版本起废弃」，")
    print("      本条没取到版本归属的源码证据）。")

    return c


def main():
    T.header("02", "控制流与 XLA —— 固化分支，还是把两条都放进图")
    with_cond, _ = section_1()
    parts = [section_A(with_cond), section_B(), section_C()]
    return T.Checks.merge(parts).summary()


if __name__ == "__main__":
    sys.exit(main())
