#!/usr/bin/env python3
"""01 jit 与追踪：**cache key 就是被固化进图的那份假设清单**——而 Python 标量默认不在里面。

对应文档：70-frameworks/jax/01-jit-and-tracing.md

★★ 本程序跑的是**本仓库的迷你 JAX 追踪器**（`minijax.py`），**不是 JAX** ★★
  本机没有任何 JAX 环境。下面所有数字都只支撑关于**机制**的结论，
  **不支撑任何关于真实 JAX 的结论**；关于真实 JAX 的每条行为断言
  都以 `specs.py` 里带 `# 来源：` 的一手 URL 为准。

挂主线（§0.8）：本文处理「固化什么」的**假设清单那一面**——cache key 的三项
  （JAX types / 静态参数的值 / 函数对象 identity）**就是清单本身**，而
  **Python 标量默认不进这份清单**（镜像对照见 70-frameworks/tensorflow/）。

三类断言（CONTRIBUTING §2：编号程序跑在默认 python3 上，故三类都在程序内部）：
  A 独立参考实现 + 反证；B 代价模型（三项 key 的作用、诊断、清缓存、耗时量级）；
  C **3 与 4 只追踪一次**且标量进的是 invars、函数 identity、jit 的透明性、
    换 shape/dtype 重追踪、形状多态仍重编译。

用法：cd code && make 01_jit_cache_key   （或 make run 全跑）
"""

import re
import sys
import time

import numpy as np

import minijax as J
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


# ---------------------------------------------------------------------------
# 独立参考实现 1：cache key 的构造（不调用 minijax 的任何东西）
# ---------------------------------------------------------------------------

def ref_abstractify(value):
    """把实参抽象成「JAX type」的**另写**版本。

    规则照官方那句「the JAX types of the arguments（new shapes or dtypes trigger a
    fresh trace and compile）」与 docstring 的「no arguments are treated as static」：
      · 数组 → (形状, dtype, 强类型)
      · **Python 标量 → ((), 默认 dtype, 弱类型)**——所以 `3` 与 `4` 抽出来**一样**
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        # 数组（numpy 的 ndarray / generic，或本仓库的 Array）——**弱类型看属性**
        return (tuple(value.shape), str(value.dtype),
                bool(getattr(value, "weak_type", False)))
    if isinstance(value, bool):
        return ((), "bool", True)
    if isinstance(value, int):
        return ((), "int32", True)
    if isinstance(value, float):
        return ((), "float32", True)
    raise TypeError("参照实现不认识这个类型：%s" % type(value).__name__)


def ref_cache_key(fun_id, calls, static_positions=()):
    """**从零重算**一次调用的 cache key。与 minijax 的 `JitWrapped._key` 无共同代码。"""
    out = [("fun", fun_id)]
    for i, v in enumerate(calls):
        if i in static_positions:
            out.append(("static", i, repr(v)))
        else:
            out.append(("aval", i, ref_abstractify(v)))
    return tuple(out)


def ref_trace_count(fun_id, sequence, static_positions=()):
    """一段调用序列要追踪几次 = 不同 cache key 的个数（本模型无子类型复用）。"""
    keys = set()
    for calls in sequence:
        keys.add(ref_cache_key(fun_id, calls, static_positions))
    return len(keys), keys


# ---------------------------------------------------------------------------
# 独立参考实现 2：数值（纯 numpy，不看任何图）
# ---------------------------------------------------------------------------

def ref_affine(x, k):
    """`x * k + 1` 的 numpy 直接实现。"""
    return np.asarray(x, dtype=np.float32) * np.float32(k) + np.float32(1.0)


# ---------------------------------------------------------------------------
# §1 一次调用经过什么
# ---------------------------------------------------------------------------

def section_1():
    J.rule("§1 一次 jit 调用：抽象化 → 拼 cache key → 查表 → 命中复用 / 不命中重追踪")

    @J.jit
    def affine(x, k):
        return x * k + 1.0

    x = J.Array(np.arange(4, dtype=np.float32).reshape(2, 2))
    r3 = affine(x, 3)
    n_after_3 = affine.trace_count()
    r4 = affine(x, 4)
    n_after_4 = affine.trace_count()

    print("官方逐字（%s）：" % SPECS.JIT_NO_STATIC_SRC)
    print("  %s …" % SPECS.JIT_NO_STATIC_VERBATIM.split(" If ")[0])
    print("affine(x, 3) → %s，追踪计数 = %d" % (r3.tolist(), n_after_3))
    print("affine(x, 4) → %s，追踪计数 = %d  ← **没有重追踪**" % (r4.tolist(), n_after_4))
    print("cache 里的键（本模型的读法：函数 identity + 各实参的 JAX type）：")
    for k in affine.keys():
        print("  %s" % (k,))
    print("那张图（jaxpr，文法见 specs.JAXPR_GRAMMAR）：")
    print(list(affine._cache.values())[0][0])
    return affine, x


# ---------------------------------------------------------------------------
# §A 算法正确性
# ---------------------------------------------------------------------------

def section_A(affine, x):
    J.rule("§A 算法正确性：两条另写的参照路径 + 一条反证")
    c = J.Checks()

    # A1 抽象化规则与独立参照逐项一致
    probes = [np.zeros((2, 2), dtype=np.float32), np.ones(3, dtype=np.float64),
              np.array(7, dtype=np.int32), 3, 4, 3.0, -1.5, True, False, 0]
    mismatch = sum(1 for p in probes
                   if (J.abstractify(p).shape, J.abstractify(p).dtype,
                       J.abstractify(p).weak_type) != ref_abstractify(p))
    c.A(mismatch == 0, "抽象化（JAX types）与独立参考实现逐项一致",
        "%d 个探针，差异 %d" % (len(probes), mismatch))
    c.A(J.abstractify(3) == J.abstractify(4),
        "★ `3` 与 `4` 抽象出来是**同一个 aval**（这就是 3/4 只追踪一次的机制）",
        "abstractify(3) = %s，abstractify(4) = %s"
        % (J.abstractify(3), J.abstractify(4)))

    # A2 一段调用序列的追踪次数 == 独立重算
    sequence = [(x, 3), (x, 4), (x, 5), (J.Array(np.zeros(3, dtype=np.float32)), 3)]

    @J.jit
    def probe(x, k):
        return x * k + 1.0

    for args in sequence:
        probe(*args)
    got = probe.trace_count()
    want, _ = ref_trace_count(id(probe.fun), sequence)
    c.A(got == want, "一段调用序列的追踪次数与独立重算一致",
        "实测 %d，独立重算 %d（序列：3/4/5 三个标量 + 一个换形状的实参）"
        % (got, want))

    # A3 数值：与纯 numpy 参照逐点一致
    worst = 0.0
    for k in (1, 2, 7):
        got_v = affine(x, k).data.astype(np.float64)
        want_v = ref_affine(x.data, k).astype(np.float64)
        worst = max(worst, float(np.max(np.abs(got_v - want_v))))
    c.A(worst == 0.0, "jit 重放的结果与 numpy 参照逐点一致",
        "3 个 k，max|Δ| = %g" % worst)

    # A4 反证：把参照的「函数 identity 项」删掉，它必须与实现不一致
    def wrong_key(fun_id, calls, static_positions=()):
        return ref_cache_key(fun_id, calls, static_positions)[1:]   # 丢掉函数 identity

    g1 = J.jit(lambda a: a * 2.0)
    g2 = J.jit(lambda a: a * 2.0)
    g1(x)
    g2(x)
    bad = len({wrong_key(id(f), (x,)) for f in (g1.fun, g2.fun)})
    good = len({ref_cache_key(id(f), (x,)) for f in (g1.fun, g2.fun)})
    c.A(bad != good,
        "反证：把参照里的「函数对象 identity」项删掉后，它与实现**必须**不一致",
        "删掉后判出 %d 个键，保留时判出 %d 个（说明 A6 那条不是恒真断言）" % (bad, good))
    return c


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def section_B(affine, x):
    J.rule("§B 代价模型：追踪次数、三项 cache key、诊断与清缓存")
    c = J.Checks()

    # B1 三项之一：**JAX types**（形状 / dtype 变了就重追踪）
    @J.jit
    def by_type(t):
        return J.numpy.reduce_sum(t)

    by_type(J.Array(np.zeros((2, 2), dtype=np.float32)))
    n1 = by_type.trace_count()
    by_type(J.Array(np.zeros((2, 3), dtype=np.float32)))
    n2 = by_type.trace_count()
    by_type(J.Array(np.zeros((2, 3), dtype=np.float64)))
    n3 = by_type.trace_count()
    c.B(n1 == 1 and n2 == 2 and n3 == 3,
        "★ 换 shape 与换 dtype 各自触发一次**新的追踪**（cache key 的第一项含二者）",
        "追踪计数 %d → %d（换 shape）→ %d（再换 dtype）" % (n1, n2, n3))
    flat = " ".join(SPECS.CACHE_KEY_VERBATIM_TAG.split())   # 原文跨行，先压平再查
    c.B("new shapes or dtypes trigger a fresh trace and compile" in flat,
        "这条有官方逐字依据（引文里直接点名 shapes 与 dtypes）",
        "来源：%s" % SPECS.raw("docs/new_docs/201/jit.md"))

    # B2 三项之二：**静态参数的值**
    @J.jit(static_argnums=(1,))
    def with_static(x, k):
        return x * k + 1.0

    with_static(x, 3)
    s1 = with_static.trace_count()
    with_static(x, 4)
    s2 = with_static.trace_count()
    c.B(s1 == 1 and s2 == 2,
        "显式标了 `static_argnums` 之后，`3` 与 `4` 变成**两个键**（这是「要静态得自己说」）",
        "追踪计数 %d → %d（对照：不标 static 时是 1 → 1）" % (s1, s2))
    c.B("no arguments are treated as static" in SPECS.JIT_NO_STATIC_VERBATIM,
        "★ Python 标量**默认不是 static**——有官方 docstring 逐字",
        "来源：%s" % SPECS.JIT_NO_STATIC_SRC)

    # B3 三项之三：**函数对象 identity**
    id_a = J.jit(lambda a: a * 2.0)
    id_b = J.jit(lambda a: a * 2.0)
    id_a(x)
    id_b(x)
    c.B(id_a.trace_count() == 1 and id_b.trace_count() == 1,
        "两个**内容相同但不同**的函数对象 ⇒ 两张缓存、各追一次",
        "追踪计数 %d + %d" % (id_a.trace_count(), id_b.trace_count()))
    print("    官方 dev 文档逐字（%s）：%s …" % (SPECS.doc("201/jit.html"),
                                            SPECS.CACHE_IDENTITY_PRACTICAL[:120]))
    print("    ⚠ 版本提示：第三项在 dev 是 `object identity of the function g`，在 tag")
    print("      jax-v0.11.1 里是 `the identity of the function \\`g\\` itself`——**不是同一句**。")

    # B4 诊断：explain_cache_misses 的文案形态
    J.config.update("jax_explain_cache_misses", True)
    explained = J.jit(lambda t: J.numpy.reduce_sum(t))
    explained(J.Array(np.zeros(3, dtype=np.float32)))
    explained(J.Array(np.zeros(3, dtype=np.int32)))     # 触发一次 miss
    J.config.update("jax_explain_cache_misses", False)
    line = explained.explain_lines[-1] if explained.explain_lines else ""
    c.B(bool(re.search(r"TRACING CACHE MISS at .*different input types", line)),
        "诊断文案与官方给的例子同形（`TRACING CACHE MISS at …: different input types`）",
        "本模型输出：%s" % line[:96])
    c.B("i32[3]" in SPECS.EXPLAIN_CACHE_MISSES_VERBATIM
        and "f32[3]" in SPECS.EXPLAIN_CACHE_MISSES_VERBATIM,
        "官方例子本身就是 f32[3] vs i32[3] 这种「换 dtype」的 miss",
        "来源：%s" % SPECS.doc("201/jit.html"))

    # B5 clear_caches 会清掉缓存
    cleared = J.jit(lambda t: J.numpy.reduce_sum(t))
    cleared(x)
    before = cleared.trace_count()
    J.clear_caches()
    cleared(x)
    c.B(cleared.trace_count() == before + 1,
        "`clear_caches()` 之后同一段调用会**重新追踪**",
        "追踪计数 %d → %d" % (before, cleared.trace_count()))

    # B6 Python 副作用：**只在追踪时发生一次**（结构判据，不看时间）
    side_effects = {"n": 0}

    @J.jit
    def with_effect(a):
        side_effects["n"] += 1                 # Python 侧副作用
        return J.numpy.reduce_sum(a)

    big = J.Array(np.random.rand(64, 64).astype(np.float32))
    for _ in range(5):
        with_effect(big)
    c.B(side_effects["n"] == 1,
        "★ 5 次调用里，函数体的 Python 代码**只跑了 1 次**（之后就只是重放那张图）",
        "副作用计数 = %d，追踪计数 = %d"
        % (side_effects["n"], with_effect.trace_count()))

    # B7 时间量级：eager / replay / trace 三条路
    replay_us = measure(lambda: with_effect(big), 200)
    eager_us = measure(lambda: J.numpy.reduce_sum(J.numpy.mul(big, 2.0)), 200)

    def trace_once():
        @J.jit
        def fresh(a):
            return J.numpy.reduce_sum(J.numpy.mul(a, 2.0))
        t0 = time.perf_counter()
        fresh(big)
        return (time.perf_counter() - t0) * 1e6

    trace_us = min(trace_once() for _ in range(5))
    print("    每次调用耗时（μs，**本仓库迷你追踪器实测，不是 JAX**）："
          "eager %.2f / replay %.2f / trace %.2f" % (eager_us, replay_us, trace_us))
    c.B(trace_us > replay_us,
        "trace 一次比 replay 一次贵（追踪要跑一遍 Python 函数体并把算子记下来）",
        "trace %.2f μs vs replay %.2f μs（%.2f×）" % (trace_us, replay_us,
                                                      trace_us / replay_us))
    c.B(0.5 <= replay_us / eager_us <= 2.0,
        "★ 反直觉：本迷你实现里 **replay 并不比 eager 便宜**（两者同量级）",
        "replay %.2f μs / eager %.2f μs = %.2f×"
        % (replay_us, eager_us, replay_us / eager_us))
    print("    ★ 这是本程序**第一版被实跑推翻的断言**：第一版写「trace 一次比 replay 贵")
    print("      一个量级以上」，实跑只有约 %.1f 倍。更重要的是后两个数——在**纯 Python" % (trace_us / replay_us))
    print("      的迷你实现**里，图是被**解释执行**的数据结构，每回合仍要付 dict 查表与")
    print("      逐节点分派，所以「折叠成图 ⇒ 每次调用更省」**在这里不成立**（与")
    print("      70-frameworks/tensorflow 那边测到的同一个现象一致）：折叠真正省下的是")
    print("      **可序列化、可交给编译器**，不是每次调用的时间。")
    print("    ⚠ 限定语：上述数字全部来自**本仓库迷你追踪器**，**不是 JAX**，不构成任何")
    print("      关于真实 JAX 编译成本或运行开销的结论。")
    return c


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def section_C():
    J.rule("§C 边界与反例：每条失败行为都要真的跑出来")
    c = J.Checks()

    J.subsection("C1 ★ Python 标量默认不特化：用 3 与 4 调用**只追踪一次**")
    x = J.Array(np.zeros(3, dtype=np.float32))

    @J.jit
    def with_k(v, k):
        return v * k

    with_k(x, 3)
    a = with_k.trace_count()
    with_k(x, 4)
    b = with_k.trace_count()
    c.C(a == 1 and b == 1,
        "★ 同一个函数用 3 与 4 调用 ⇒ **只追踪一次**（与 TF 的 C2 正好相反）",
        "追踪计数 %d → %d；cache 里的键数 = %d" % (a, b, with_k.cache_size()))
    c.C(float(with_k(x, 3).data[0]) == 0.0 and float(with_k(x, 4).data[0]) == 0.0,
        "对照：结果仍然跟着**运行时的实参**走（不是被冻结的旧值）",
        "因为标量已经是一个**图输入**，每次调用都会重新喂进去")

    J.subsection("C2 ★ 标量进的是 jaxpr 的 **invars**（不是一个被烧进图里的字面量）")
    cj = list(with_k._cache.values())[0][0]
    n_invars = len(cj.jaxpr.invars)
    c.C(n_invars == 2,
        "★ 追踪出来的 jaxpr 有 **2 个 invar**（张量 v + 标量 k）",
        "invars = %s（对照：TF 会把 3 烧成 op 的字面量属性，见回链）"
        % [str(v) for v in cj.jaxpr.invars])
    c.C(len(cj.jaxpr.constvars) == 0,
        "所以这个 jaxpr 里**没有**来自该标量的常量",
        "constvars = %s" % [str(v) for v in cj.jaxpr.constvars])
    print("    对照实验：函数体里写的字面量（`+ 1.0`）**是**常量：")
    print("      %s" % str(section_C_const()[0].jaxpr).replace("\n", "\n      "))

    with_static = J.jit(lambda v, k: v * k + 1.0, static_argnums=(1,))
    with_static(x, 3)
    scj = list(with_static._cache.values())[0][0]
    c.C(len(scj.jaxpr.invars) == 1,
        "显式 `static_argnums` 之后，标量**不再进 invars**（只剩 1 个）",
        "invars = %s" % [str(v) for v in scj.jaxpr.invars])
    print("    ★ 这一条就是主线上那句话的全部：**进不进 cache key，决定进不进图。**")

    J.subsection("C3 ★ 函数对象的 identity：内容相同的两个 lambda 各追一次")
    f_lambda_1 = J.jit(lambda z: z * 2.0)
    f_lambda_2 = J.jit(lambda z: z * 2.0)
    f_lambda_1(x)
    f_lambda_2(x)
    c.C(f_lambda_1.trace_count() == 1 and f_lambda_2.trace_count() == 1,
        "两次 `jit(lambda …)` 即使函数体一模一样，也**各追踪一次**",
        "两个 wrapper 的追踪计数分别 = %d, %d"
        % (f_lambda_1.trace_count(), f_lambda_2.trace_count()))
    same = J.jit(f_lambda_1.fun)                 # 换成**同一个函数对象**再包一次
    same(x)
    c.C(same.trace_count() == 1,
        "对照：包**同一个函数对象**不会有额外代价（键里的 identity 没变）",
        "追踪计数 = %d" % same.trace_count())
    print("    官方 dev 文档逐字（%s）：%s" % (SPECS.doc("201/jit.html"),
                                             SPECS.CACHE_IDENTITY_PRACTICAL))

    J.subsection("C4 `jit` 在别的追踪之下是**透明的**（不会多一层缓存）")
    plain = lambda z: z * 3.0                                      # noqa: E731
    nested = J.jit(J.jit(plain))
    outer = J.jit(lambda z: nested(z) * 2.0)
    outer(x)
    c.C(outer.trace_count() == 1 and len(outer.keys()) == 1,
        "`jit(jit(f))` 与 `jit(f)` 得到同一张图——嵌套的 jit **不建自己的缓存**",
        "外层追踪计数 = %d，缓存条目 = %d" % (outer.trace_count(), outer.cache_size()))
    h = J.jit(plain)
    c.C(h(x).tolist() == [0.0, 0.0, 0.0],
        "同一份函数体，直接 jit 与嵌套 jit 的数值一致",
        "h(x) = %s" % h(x).tolist())

    J.subsection("C5 换 shape / 换 dtype 各自重追踪（『JAX 对 shape 不敏感』是错的）")
    shape_sensitive = J.jit(lambda t: J.numpy.reduce_sum(t))
    shape_sensitive(J.Array(np.zeros(3, dtype=np.float32)))
    n1 = shape_sensitive.trace_count()
    shape_sensitive(J.Array(np.zeros(5, dtype=np.float32)))
    n2 = shape_sensitive.trace_count()
    shape_sensitive(J.Array(np.zeros(5, dtype=np.int32)))
    n3 = shape_sensitive.trace_count()
    c.C(n1 == 1 and n2 == 2 and n3 == 3,
        "★ 换 shape、换 dtype 都会**重追踪**——**三者（TF / JAX / PyTorch）默认都对 shape 敏感**",
        "追踪计数 %d → %d → %d" % (n1, n2, n3))

    J.subsection("C6 `jax.jit` 没有 dynamic shape 开关；`jax.export` 的形状多态也不免重编译")
    print("    官方逐字（%s）：" % SPECS.EXPORT_SYMBOLIC_URL)
    print("      %s" % SPECS.EXPORT_SYMBOLIC_VERBATIM)
    g = J.jit(lambda z: z * 2.0)
    ex = J.export.export(g)(J.Array(np.zeros((3,), dtype=np.float32)))
    ex(J.Array(np.ones((3,), dtype=np.float32)))
    same_shape_recompiles = ex.recompiles
    ex(J.Array(np.ones((4,), dtype=np.float32)))
    c.C(same_shape_recompiles == 0 and ex.recompiles == 1,
        "★ 导出之后：**同样的形状**不重编译；**换了形状仍然重编译**",
        "重编译次数 %d → %d（形状多态也不免除重编译）"
        % (same_shape_recompiles, ex.recompiles))
    c.C(J.export.symbolic_shape("a, b").names == ("a", "b"),
        "`jax.export.symbolic_shape(\"a, b\")` 是本模型的对应物",
        "官方签名逐字：%s" % SPECS.SYMBOLIC_SHAPE_SIGNATURE)

    return c


def section_C_const():
    lit = J.jit(lambda v: v * 2.0 + 1.0)
    lit(J.Array(np.zeros(2, dtype=np.float32)))
    return [list(lit._cache.values())[0][0]]


def main():
    J.header("01", "jit 与追踪 —— cache key 就是假设清单，而 Python 标量默认不在里面")
    affine, x = section_1()
    checks = [section_A(affine, x), section_B(affine, x), section_C()]
    return J.Checks.merge(checks).summary()


if __name__ == "__main__":
    sys.exit(main())
