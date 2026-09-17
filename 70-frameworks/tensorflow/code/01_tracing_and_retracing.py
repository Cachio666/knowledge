#!/usr/bin/env python3
"""01 graph 与 eager：**`TraceType` 就是被固化进图的那份假设清单**。

对应文档：70-frameworks/tensorflow/01-graph-and-eager.md

★★ 本程序跑的是**本仓库的迷你 TF 追踪器**（`minitf.py`），**不是 TensorFlow** ★★
  本机没有任何 TF 环境。下面所有数字都只支撑关于**机制**的结论，
  **不支撑任何关于真实 TensorFlow 的结论**；关于真实 TF 的每条行为断言
  都以 `specs.py` 里带 `# 来源：` 的一手 URL 为准。

三类断言（CONTRIBUTING §2：编号程序跑在默认 python3 上，故三类都写在程序内部）：
  A —— 与**独立参考实现**逐项比对（形状子类型、缓存复用决策各一条另写的路径），
       再附一条「把实现故意改错必须被抓到」的反证。
  B —— 代价模型：追踪次数 = 不同 FunctionType 的个数；重追踪告警阈值；
       trace / replay / eager 三条路的耗时量级。
  C —— 失败行为必须真的发生：换 shape 重追踪、Python 标量按值特化、
       自由变量改动**不**重追踪（于是结果错）、副作用只在追踪时发生一次。

用法：cd code && make 01_tracing_and_retracing   （或 make run 全跑）
"""

import functools
import sys
import time

import numpy as np

import minitf as T
import specs as SPECS

F32 = "float32"


def measure(fn, n):
    """跑 n 次取**最快**一组（微秒）。取最小值以压掉调度抖动。"""
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    return best


def prebuilt_once(a, b):
    """把 `reduce_sum(mul(a, b))` 预解析成一次 numpy 组合调用：图被编译掉之后的每回合成本。
    ⚠ 这是本仓库迷你实现里的手写对照，**不是 XLA 的模型**。
    """
    return np.sum(np.multiply(a.numpy, b.numpy))


# ---------------------------------------------------------------------------
# 独立参考实现 1：形状的子类型关系（不调用 minitf 的方法）
# ---------------------------------------------------------------------------

def ref_shape_subtype(a, b):
    """把官方规则**从文字直接翻译**成独立谓词（见 specs.TRACETYPE_RULES）。

    规则原文：ranked shapes are a subtype of unranked shapes;
    fixed dimensions are a subtype of unknown dimensions。`None` 表示未知秩。
    """
    if b is None:                       # b 未知秩 ⇒ 任何 a 都是它的子类型
        return True
    if a is None:                       # a 未知秩、b 已知秩 ⇒ 不成立
        return False
    if len(a) != len(b):                # 秩不同 ⇒ 不成立
        return False
    for da, db in zip(a, b):
        if da is None and db is not None:
            return False                # 未知维 不是 定长维 的子类型
        if da is not None and db is not None and da != db:
            return False
    return True


def ref_reuse_ok(cached_dims, call_dims):
    """缓存复用判据的独立重算：**调用方的类型必须是缓存键的子类型**。

    这正是官方那段话的机制（见 specs.TRACETYPE_DISPATCH_VERBATIM）：
    命中就派发过去，不命中就新追一个。
    """
    return ref_shape_subtype(call_dims, cached_dims)


def enumerate_shapes(max_rank=2, dims=(None, 1, 2)):
    """枚举一批形状（含未知秩），用来把两条路径逐项比完。"""
    out = [None]
    def rec(prefix, k):
        if k == 0:
            out.append(tuple(prefix))
            return
        for d in dims:
            rec(prefix + [d], k - 1)
    for r in range(1, max_rank + 1):
        rec([], r)
    return out


# ---------------------------------------------------------------------------
# 独立参考实现 2：一段调用序列应当追踪几次
# ---------------------------------------------------------------------------

def ref_trace_count(sequence, cached=None):
    """按「子类型匹配 + 最具体者优先」的规则从零重算：一段调用序列要追踪几次。

    **独立于 minitf 的缓存实现**——它只认形状元组，不碰 Tracer / Graph。
    """
    cached = list(cached or [])
    traces = 0
    for call in sequence:
        if not any(ref_reuse_ok(c, call) for c in cached):
            cached.append(call)
            traces += 1
    return traces, cached


# ---------------------------------------------------------------------------
# §1 追踪出来的东西长什么样
# ---------------------------------------------------------------------------

def section_1():
    T.rule("§1 一次调用经过什么：FunctionType → 查表 → ConcreteFunction")

    @T.tf_function
    def affine(x, k):
        return T.add(T.mul(x, k), 1.0)

    x = T.Tensor(np.arange(4, dtype=np.float32).reshape(2, 2))
    r1 = affine(x, 3)
    print("第一次 affine(x, 3) → %s，追踪计数 = %d"
          % (r1.numpy.tolist(), affine.experimental_get_tracing_count()))
    print("缓存里的键（FunctionType）：%s"
          % [str(k) for k in affine._function_cache])
    r2 = affine(x, 3)
    print("第二次同一个键 → %s，追踪计数仍 = %d"
          % (r2.numpy.tolist(), affine.experimental_get_tracing_count()))
    print()
    print("同一段图的两种读法——**注意 Python 字面量 3 不在 inputs 里**（它被烧成了字面量）：")
    print(affine._function_cache[list(affine._function_cache)[0]].graph)
    print("pretty_printed_concrete_signatures()（官方同名 API）：")
    print(affine.pretty_printed_concrete_signatures())
    return affine, x


# ---------------------------------------------------------------------------
# §A 算法正确性：与独立参考实现比对
# ---------------------------------------------------------------------------

def section_A(affine, x):
    T.rule("§A 算法正确性：两条另写的参照路径")
    c = T.Checks()

    shapes = enumerate_shapes()
    mismatch = 0
    total = 0
    for a in shapes:
        for b in shapes:
            total += 1
            got = T.TensorShape(a).is_subtype_of(T.TensorShape(b))
            if got != ref_shape_subtype(a, b):
                mismatch += 1
    c.A(mismatch == 0, "形状子类型关系与独立参考实现逐项一致",
        "枚举 %d 对，差异 %d" % (total, mismatch))

    c.A(T.TensorShape([1, 2]).is_subtype_of(T.TensorShape([None, None])),
        "TensorShape([1,2]) 是 TensorShape([None,None]) 的子类型")
    c.A(not T.TensorShape([None, None]).is_subtype_of(T.TensorShape([1, 2])),
        "反过来**不成立**：TensorShape([None,None]) 不是 TensorShape([1,2]) 的子类型")
    c.A(T.TensorShape([1, 2]).is_subtype_of(T.TensorShape(None))
        and not T.TensorShape(None).is_subtype_of(T.TensorShape([1, 2])),
        "已知秩 是 未知秩 的子类型，反之不成立")

    seq = [(2, 2), (2, 2), (3, 3), (2, 2), (None, None), (7, 7), (3, 3)]
    expected, _ = ref_trace_count(seq)

    @T.tf_function
    def probe(t):
        return T.reduce_sum(t)

    for dims in seq:
        data = np.zeros(tuple(2 if d is None else d for d in dims), dtype=np.float32)
        probe(T.Tensor(data))
    got = probe.experimental_get_tracing_count()
    c.A(got == expected, "一段调用序列的追踪次数与独立重算一致",
        "实测 %d，独立重算 %d，序列 %s" % (got, expected, seq))

    # 反证：把实现故意改错，上面那条 A 必须失败（证明它不是恒真断言）
    def wrong_is_subtype(self_dims, other_dims):
        """故意把方向反过来：未知维当作定长维的子类型。"""
        if other_dims is None:
            return True
        if self_dims is None:
            return False
        return [(None if d is None else d) for d in self_dims] == \
               [(None if d is None else d) for d in other_dims]

    broken = sum(1 for a in shapes for b in shapes
                 if wrong_is_subtype(a, b) != ref_shape_subtype(a, b))
    c.A(broken > 0, "反证：把子类型方向改错后，与参照实现的差异必须 > 0",
        "差异 %d 项（说明 A1 会因实现改错而失败，不是恒真）" % broken)

    return c


# ---------------------------------------------------------------------------
# §2 / §3 追踪与副作用
# ---------------------------------------------------------------------------

def section_2_3():
    T.rule("§2 Python 副作用只在 tracing 时发生一次")
    counter = {"n": 0}

    def body(x):
        counter["n"] += 1                 # Python 侧副作用
        return T.add(x, 1.0)

    traced = T.tf_function(body)
    t = T.Tensor(np.zeros(3, dtype=np.float32))
    for _ in range(5):
        traced(t)
    with_tf = counter["n"]

    counter["n"] = 0                      # 对照：**eager** 每次都真的跑 Python
    for _ in range(5):
        body(t)
    eager_n = counter["n"]
    print("同一个函数体调用 5 次：套 tf.function ⇒ 副作用 %d 次；eager ⇒ %d 次"
          % (with_tf, eager_n))
    print("官方逐字：%s" % SPECS.SIDE_EFFECTS_VERBATIM[:110] + " …")

    T.rule("§3 诊断工具：只存在于源码 docstring 的两件")
    print("experimental_get_tracing_count() → %d" % traced.experimental_get_tracing_count())
    print("官方 docstring 逐字：%r" % SPECS.TRACING_COUNT_DOC)
    print("★ 注意：这两个诊断入口**在 tf.function 手册正文里没有**，只存在于源码 docstring")
    print("  （见 specs.py 的 `# 来源：` 指到的 polymorphic_function.py 行号）。")
    return with_tf, eager_n


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def section_B():
    T.rule("§B 代价模型：追踪次数、告警阈值、三条路的耗时")

    c = T.Checks()

    @T.tf_function
    def m(x, k):
        return T.add(x, k)

    keys = []
    for dims, k in [((2,), 1), ((2,), 1), ((3,), 1), ((2,), 2), ((3,), 2), ((3,), 2)]:
        keys.append((dims, k))
        m(T.Tensor(np.zeros(dims, dtype=np.float32)), k)
    distinct = len(set(keys))
    c.B(m.experimental_get_tracing_count() == distinct,
        "追踪次数 == 调用的不同 (shape, 字面量) 组合数",
        "实测 %d，模型 %d" % (m.experimental_get_tracing_count(), distinct))

    det = T.FrequentTracingDetector(SPECS.FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY,
                                    SPECS.FREQUENT_TRACING_WARNING_THRESHOLD)
    for _ in range(4):
        det.called_with_tracing("f")
    four = len(det.warnings)
    det.called_with_tracing("f")
    five = len(det.warnings)
    c.B(four == 0 and five == 1,
        "重追踪告警的阈值模型：10 次窗口内 4 次不报、5 次报",
        "常量取 (%d, %d)（来源：polymorphic_function.py:110-111）"
        % (SPECS.FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY,
           SPECS.FREQUENT_TRACING_WARNING_THRESHOLD))
    c.B("different shapes" in SPECS.FREQUENT_TRACING_WARNING_TEXT,
        "官方告警文案里直接点名了 passing tensors with different shapes",
        "这正是「TF 对 shape 敏感」的一手证据")

    x = T.Tensor(np.random.rand(64, 64).astype(np.float32))
    w = T.Tensor(np.random.rand(64, 64).astype(np.float32))

    def eager_run(n):
        t0 = time.perf_counter()
        for _ in range(n):
            T.reduce_sum(T.mul(x, w))
        return (time.perf_counter() - t0) / n * 1e6

    @T.tf_function
    def compiled(xa, wa):
        return T.reduce_sum(T.mul(xa, wa))

    def trace_once():
        @T.tf_function
        def fresh(a, b):
            return T.reduce_sum(T.mul(a, b))
        t0 = time.perf_counter()
        fresh(x, w)
        return (time.perf_counter() - t0) * 1e6

    n = 300
    e_us = eager_run(n)
    cf = compiled.get_concrete_function(T.TensorSpec(F32, (64, 64)),
                                        T.TensorSpec(F32, (64, 64)))
    t0 = time.perf_counter()
    for _ in range(n):
        cf(x, w)
    r_us = (time.perf_counter() - t0) / n * 1e6
    traces = [trace_once() for _ in range(5)]
    t_us = min(traces)
    c_us = measure(functools.partial(prebuilt_once, x, w), n)
    print("    每次调用耗时（μs，本仓库迷你实现）：eager %.2f / replay %.2f / "
          "预解析闭包 %.2f / trace %.2f" % (e_us, r_us, c_us, t_us))
    c.B(t_us > 3 * r_us,
        "trace 一次比 replay 一次贵一个量级以上——这就是「固化假设」要付的前期代价",
        "trace %.2f μs vs replay %.2f μs（%.1f×）" % (t_us, r_us, t_us / r_us))
    c.B(0.5 <= r_us / e_us <= 2.0,
        "★ 反直觉：本迷你实现里 **replay 并不比 eager 便宜**（两者同量级）",
        "replay %.2f μs / eager %.2f μs = %.2f×"
        % (r_us, e_us, r_us / e_us))
    print("    ★ 这是第一版被实跑推翻的断言：「折叠成图 ⇒ 每次调用更省」在本迷你实现里不成立——")
    print("      图是被**解释执行**的，每回合仍要 dict 查表 + 逐节点分派；"
          "「折叠」真正省下的是可序列化、可交给编译器。")

    ref = prebuilt_once(x, w)
    got = cf(x, w).numpy
    c.A(np.allclose(ref, got), "预解析闭包与图重放的数值一致",
        "max|Δ| = %.3g" % float(np.max(np.abs(ref - got))))

    before = T.STATS["eager_dispatch_layers"]
    T.mul(x, w)
    c.B(T.STATS["eager_dispatch_layers"] - before == T.EAGER_LAYERS_PER_OP,
        "eager 一次算子调用记 %d 层（本仓库模型：python 绑定 → dispatcher → kernel）"
        % T.EAGER_LAYERS_PER_OP,
        "★ 这个层数是**本仓库模型参数**，不是对 TF eager 路径的测量")
    return c


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def section_C():
    T.rule("§C 边界与反例：每条失败行为都要真的跑出来")
    c = T.Checks()

    T.subsection("C1 换 shape → 重追踪；用 None 维 trace → 复用")
    @T.tf_function
    def plain(t):
        return T.reduce_sum(t)

    plain(T.Tensor(np.zeros((2, 2), dtype=np.float32)))
    n1 = plain.experimental_get_tracing_count()
    plain(T.Tensor(np.zeros((2, 3), dtype=np.float32)))
    n2 = plain.experimental_get_tracing_count()
    c.C(n1 == 1 and n2 == 2, "shape 从 (2,2) 换到 (2,3) 触发了**重追踪**",
        "追踪计数 %d → %d" % (n1, n2))

    @T.tf_function
    def relaxed(t):
        return T.reduce_sum(t)

    cf = relaxed.get_concrete_function(T.TensorSpec(F32, (None, None)))
    relaxed(T.Tensor(np.zeros((2, 2), dtype=np.float32)))
    relaxed(T.Tensor(np.zeros((5, 7), dtype=np.float32)))
    relaxed(T.Tensor(np.zeros((1, 9), dtype=np.float32)))
    c.C(relaxed.experimental_get_tracing_count() == 1,
        "先用 `None` 维 trace，再喂三种具体 shape —— **一次都没重追踪**",
        "追踪计数 = %d（对照上一段的 2）" % relaxed.experimental_get_tracing_count())
    print("    ★ 方向性只在这一个方向上成立：(1,2) <: (None,None) = %s，而反过来 = %s"
          % (T.TensorShape([1, 2]).is_subtype_of(T.TensorShape([None, None])),
             T.TensorShape([None, None]).is_subtype_of(T.TensorShape([1, 2]))))

    T.subsection("C2 Python 标量按**值**特化：3 与 4 是两次追踪")
    @T.tf_function
    def with_k(x, k):
        return T.add(x, k)

    x = T.Tensor(np.zeros(3, dtype=np.float32))
    with_k(x, 3)
    a = with_k.experimental_get_tracing_count()
    with_k(x, 4)
    b = with_k.experimental_get_tracing_count()
    c.C(a == 1 and b == 2, "同一个函数用 3 与 4 调用 ⇒ **两次追踪**（LiteralTraceType<3> / <4>）",
        "追踪计数 %d → %d；键 = %s" % (a, b, [str(k) for k in with_k._function_cache]))
    g = with_k._function_cache[list(with_k._function_cache)[0]].graph
    lits = [op.attrs.get("rhs_literal") for op in g.ops if "rhs_literal" in op.attrs]
    c.C(lits == [3.0], "字面量真的被**烧进了图**（不是占位符）", "图里的字面量 = %s" % lits)

    T.subsection("C3 自由变量改动**不**重追踪 —— 于是图里留着旧值")
    flag = {"k": 1.0}

    @T.tf_function
    def uses_closure(x):
        return T.add(x, flag["k"])       # 自由变量：不进 TraceType

    r_before = uses_closure(x).numpy
    traces_before = uses_closure.experimental_get_tracing_count()
    flag["k"] = 100.0                    # 改自由变量
    r_after = uses_closure(x).numpy
    traces_after = uses_closure.experimental_get_tracing_count()
    c.C(traces_before == traces_after == 1 and abs(float(r_after[0]) - 1.0) < 1e-6,
        "改自由变量后**没有重追踪**，返回的仍是旧值（图里冻着 1.0）",
        "追踪计数 %d → %d；结果 %.1f → %.1f（期望若重追踪会是 100.0）"
        % (traces_before, traces_after, float(r_before[0]), float(r_after[0])))
    print("    ★ 官方逐字：%s" % SPECS.GLOBALS_NO_RETRACE_VERBATIM[:110] + " …")

    T.subsection("C4 reduce_retracing 的效果与它的边界")
    @T.tf_function
    def rr(t):
        return T.reduce_sum(t)

    rr.reduce_retracing = True
    for dims in [(2, 2), (3, 3), (4, 5), (7, 9)]:
        rr(T.Tensor(np.zeros(dims, dtype=np.float32)))
    c.C(rr.experimental_get_tracing_count() == 1,
        "reduce_retracing=True 下，四种 shape 只追踪一次（松弛成了 (None, None)）",
        "追踪计数 = %d" % rr.experimental_get_tracing_count())
    @T.tf_function
    def rr_rank(t):
        return T.reduce_sum(t)

    rr_rank.reduce_retracing = True
    rr_rank(T.Tensor(np.zeros((2, 2), dtype=np.float32)))
    rr_rank(T.Tensor(np.zeros((8,), dtype=np.float32)))          # 秩变了
    c.C(rr_rank.experimental_get_tracing_count() == 2,
        "★ 边界：reduce_retracing **救不了秩变化**（(2,2) → (8,) 仍是两次追踪）",
        "追踪计数 = %d（松弛只把已知维换成未知维，不跨秩）"
        % rr_rank.experimental_get_tracing_count())

    T.subsection("C5 dtype 变化也重追踪")
    @T.tf_function
    def dtype_sensitive(t):
        return T.reduce_sum(t)

    dtype_sensitive(T.Tensor(np.zeros(3, dtype=np.float32)))
    dtype_sensitive(T.Tensor(np.zeros(3, dtype=np.float64)))
    c.C(dtype_sensitive.experimental_get_tracing_count() == 2,
        "dtype 从 float32 换到 float64 ⇒ 重追踪（键里含 dtype）",
        "追踪计数 = %d" % dtype_sensitive.experimental_get_tracing_count())

    return c


def main():
    T.header("01", "graph 与 eager —— TraceType 就是被固化进图的那份假设清单")
    affine, x = section_1()
    checks = [section_A(affine, x), section_B(), section_C()]
    return T.Checks.merge(checks).summary()


if __name__ == "__main__":
    sys.exit(main())
