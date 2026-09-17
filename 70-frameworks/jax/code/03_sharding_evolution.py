#!/usr/bin/env python3
"""03 分片与心智差异：**多设备那条路的演进**，以及它与 PyTorch 的差别。

对应文档：70-frameworks/jax/03-sharding-and-differences.md

★★ 本程序跑的是**本仓库的迷你 JAX 追踪器**（`minijax.py`），**不是 JAX** ★★
  本机没有任何 JAX 环境，也没有多设备。下面所有数字都只支撑**机制**结论；
  关于真实 JAX 的行为断言以 `specs.py` 里带 `# 来源：` 的一手 URL 为准。

挂主线（§0.8）：本文处理的是「固化什么」的**多设备那一面**。
  `pmap` → `pjit` → 并入 `jax.jit` → `shard_map` 这条演进线，
  每一步换的都是「**谁来把假设写下来**」：用户显式写（pmap / shard_map），
  还是编译器替你推（jit 的自动分片）。**假设仍然要有人写，只是换了个作者。**

三类断言（CONTRIBUTING §2：编号程序跑在默认 python3 上，故三类都在程序内部）：
  A —— 与**独立参考实现**比对：`lax` 四件套的取值、`psum` 的全和设备求和、
       结构化循环的梯度，全部用另写的 numpy 路径重算，再附一条反证。
  B —— 代价模型：自动分片插了几个集合通信、`shard_map` 插了几个（0）、
       设备数变了会怎样。
  C —— 失败行为必须真的跑出来：对 traced 值取 bool 抛错、`while_loop` 反向模式抛错、
       未注册的 primitive 没法 vmap、`pmap`/`pjit` 各自发废弃告警。

用法：cd code && make 03_sharding_evolution   （或 make run 全跑）
"""

import sys
import warnings

import numpy as np

import minijax as J
import specs as SPECS
from minijax import lax

F32 = "float32"

# 一组**让「累乘」与「累加」差别足够大**的扫描输入（1·2·3 恰好等于 1+2+3，会掩盖错误）
SCAN_XS = (2.0, 3.0, 4.0)
SCAN_PROD = 24.0


# ---------------------------------------------------------------------------
# 独立参考实现（纯 numpy，不看任何图）
# ---------------------------------------------------------------------------

def ref_cond(pred, x, plus, minus):
    """`pred > 0 ? x + plus : x - minus` 的 numpy 直接实现。"""
    return np.asarray(x) + plus if bool(pred > 0) else np.asarray(x) - minus


def ref_countdown(start, stop=1.0, step=1.0):
    """`while v > stop: v -= step` 的 numpy 直接实现。"""
    v = np.float64(start)
    guard = 0
    while v > np.float64(stop):
        v -= np.float64(step)
        guard += 1
        if guard > 100000:
            raise RuntimeError("参照实现未收敛")
    return v


def ref_psum_shards(shards):
    """`psum` 的语义：把各设备的**局部值**按元素加起来。"""
    total = np.zeros_like(np.asarray(shards[0], dtype=np.float64))
    for s in shards:
        total = total + np.asarray(s, dtype=np.float64)
    return total


def ref_scan_product(xs):
    """`scan(lambda c, xi: (c*xi, c*xi), init, xs)` 的 carry 终值。"""
    c = 1.0
    for x in xs:
        c = c * x
    return c


# ---------------------------------------------------------------------------
# §1 结构化控制流四件套
# ---------------------------------------------------------------------------

def section_1():
    J.rule("§1 结构化控制流：lax.cond / while_loop / fori_loop / scan")

    @J.jit
    def pick(p, x):
        return lax.cond(p > 0, lambda v: v + 1.0, lambda v: v - 1.0, x)

    x = J.Array(np.float32(0.5))
    print("cond 的第一条性质：**谓词可以是 traced 值**。")
    print("  pick(+3.0, x) = %.1f ；pick(-3.0, x) = %.1f ；追踪计数 = %d"
          % (float(pick(J.Array(np.float32(3.0)), x).data),
             float(pick(J.Array(np.float32(-3.0)), x).data), pick.trace_count()))
    print("  ★ 同一张图、不同输入 ⇒ 走了不同分支，且**一次都没重追踪**。")
    print()
    print("官方逐字（dev 文档 %s）：" % SPECS.LAX_COND_URL)
    print("  %s" % SPECS.LAX_COND_VERBATIM_DEV)
    print("  （tag jax-v0.11.1 里同一句用的是破折号，见 specs.LAX_COND_VERBATIM_TAG）")
    print()
    print("第二个性质：**两支都进图**。这张图的形态：")
    cj = list(pick._cache.values())[0][0]
    print("  %s" % str(cj).replace("\n", "\n  "))

    @J.jit
    def loopy(v):
        return lax.while_loop(lambda t: t > 1.0, lambda t: t - 1.0, v)

    print()
    print("while_loop：body 只进图一次，**图的语句数与迭代次数无关**。")
    v5 = float(loopy(J.Array(np.float32(5.0))).data)
    v500 = float(loopy(J.Array(np.float32(500.0))).data)
    cj_loop = list(loopy._cache.values())[0][0]
    print("  init=5 → %.1f ；init=500 → %.1f ；追踪计数 = %d"
          % (v5, v500, loopy.trace_count()))
    print("  图的**语句数**（本层方程 + 子 jaxpr 的方程，递归）= %d"
          % cj_loop.jaxpr.statement_count())
    print("  ★ init 从 5 变到 500（迭代次数差 100 倍），语句数**一个也没多**；")
    print("    而且两次调用共享**同一张图**（标量 f32 的 aval 相同 ⇒ 同一个 cache key）。")
    print()
    print("⚠ 但 `while_loop` 有一条硬限制（见 §C2）：**它不是反向模式可微的**。")
    print("  官方 docstring 逐字（%s）：" % SPECS.WHILE_FORWARD_ONLY_SRC)
    print("  %s" % SPECS.WHILE_FORWARD_ONLY_VERBATIM)
    return pick, x


# ---------------------------------------------------------------------------
# §A 算法正确性
# ---------------------------------------------------------------------------

def section_A(pick, x):
    J.rule("§A 算法正确性：与另写的 numpy 参照逐项比对 + 一条反证")
    c = J.Checks()

    worst = 0.0
    for pred in (-5.0, -0.5, 0.5, 5.0, 123.25):
        got = float(pick(J.Array(np.float32(pred)), x).data)
        want = float(ref_cond(np.float32(pred), x.data, 1.0, 1.0))
        worst = max(worst, abs(got - want))
    c.A(worst == 0.0, "cond 的取值与 numpy 参照逐点一致",
        "5 个谓词，max|Δ| = %g" % worst)
    c.A(pick.trace_count() == 1,
        "上面这批调用**一次都没重追踪**（谓词的值不进 cache key）",
        "追踪计数 = %d" % pick.trace_count())

    @J.jit
    def countdown(v):
        return lax.while_loop(lambda t: t > 1.0, lambda t: t - 1.0, v)

    worst_loop = 0.0
    for start in (1.0, 2.0, 5.0, 17.0, 100.0):
        got = float(countdown(J.Array(np.float32(start))).data)
        want = float(ref_countdown(start))
        worst_loop = max(worst_loop, abs(got - want))
    c.A(worst_loop < 1e-5, "while_loop 的收敛值与 numpy 参照一致",
        "5 个起点，max|Δ| = %g" % worst_loop)

    @J.jit
    def scanned(v):
        carry, _ys = lax.scan(lambda cc, xi: (cc * xi, cc * xi), v,
                              J.Array(SCAN_XS, dtype=np.float32))
        return carry

    got_scan = float(scanned(J.Array(np.float32(1.0))).data)
    want_scan = ref_scan_product(SCAN_XS)
    c.A(abs(got_scan - want_scan) < 1e-5, "scan 的 carry 终值与 numpy 参照一致",
        "实测 %.6f，参照 %.6f" % (got_scan, want_scan))

    # 梯度：scan 有反向规则，与解析导数一致
    g = J.grad(lambda v: scanned(v))(J.Array(np.float32(2.0)))
    c.A(abs(float(g.data) - SCAN_PROD) < 1e-4,
        "★ `scan` **有反向规则**：d(carry)/d(init) = %s = %g"
        % ("·".join(str(v) for v in SCAN_XS), SCAN_PROD),
        "实测 %.6f" % float(g.data))

    # 反证：把参照的「乘」改成「加」，A 必须失败
    def wrong_scan(xs):
        c0 = 1.0
        for x in xs:
            c0 = c0 + x
        return c0

    c.A(abs(wrong_scan(SCAN_XS) - want_scan) > 1.0,
        "反证：把参照的「累乘」改成「累加」，与实现的差立刻拉开",
        "错版 %.1f vs 正确 %.1f（说明上面那条不是恒真断言）"
        % (wrong_scan(SCAN_XS), want_scan))
    return c


# ---------------------------------------------------------------------------
# §2 多设备那条路的演进
# ---------------------------------------------------------------------------

def section_2():
    J.rule("§2 演进：pmap → pjit → 并入 jax.jit → shard_map")

    print("① **`pmap` 已被官方劝退**（docstring 逐字，%s）：" % SPECS.PMAP_DOC_SRC)
    print("   %s" % SPECS.PMAP_DOC_VERBATIM)
    print("   渲染后：%s" % SPECS.PMAP_DOC_RENDERED)
    print("   并且（同文件 :79-81）：%s" % SPECS.PMAP_IMPL_RENDERED)
    print("   ★ 注意「劝退」的准确含义：**pmap 本体还在**，只是改成用别的实现——")
    print("     与 `pjit` 的处置**不同**（下面 ②）。")
    print()
    print("   与 vmap 的官方对照（同文件 :88-91，**完整句，别拆**）：")
    print("   %s" % SPECS.PMAP_VMAP_CONTRAST_VERBATIM)
    print("   ⚠ 本项目订正：计划把这句话引成「vmap() vectorizes … , pmap() instead")
    print("     replicates …」——**拆掉了原文的 `but where … , … instead` 关联结构**，")
    print("     那不是逐字的原句。")
    print()
    print("② **`pjit` 已被并进 `jax.jit`**（%s）：" % SPECS.PJIT_MERGE_SRC)
    for line in SPECS.PJIT_MERGE_VERBATIM.splitlines():
        print("   %s" % line)
    print("   ★ 源码注释里的精确日期：**%s**。" % SPECS.PJIT_DEPRECATION_DATE)
    print("   落点：`jax.jit` 的**返回类型**就叫 `%s`（%s）。"
          % (SPECS.JIT_RETURN_TYPE, SPECS.JIT_RETURN_TYPE_SRC))
    print()
    print("③ **`shard_map` 的定位**（%s）：" % SPECS.SHARD_MAP_URL)
    print("   %s" % SPECS.SHARD_MAP_COMPOSABLE)
    print()
    mesh = J.Mesh(["d0", "d1"], ("x",))
    xs = J.Array(np.arange(8, dtype=np.float32))

    @J.jit
    def total(v):
        return J.numpy.reduce_sum(v)

    total(xs)
    cj = list(total._cache.values())[0][0]
    auto = J.Partitioned(J.ClosedJaxpr(cj.jaxpr), [J.PartitionSpec("x")],
                         J.PartitionSpec(), mesh)
    auto_out = auto(xs)

    sm = J.shard_map(lambda v: lax.psum(J.numpy.reduce_sum(v), "x"), mesh,
                     [J.PartitionSpec("x")], [J.PartitionSpec()])
    sm_out = sm(xs)

    print("两条路都算「所有元素之和」（8 个元素的数组，2 个设备）：")
    print("  · `jit(..., in_shardings=P('x'))`：分区器**自动**插了 %d 个集合通信，"
          "结果 %.1f" % (auto.inserted_collectives, float(auto_out.data)))
    print("  · `shard_map(...)`：包装器自己插了 %d 个，**用户**写了 %d 个，结果 %.1f"
          % (sm.inserted_collectives, sm.user_collectives, float(sm_out.data)))
    print("  ★ 这就是两条路唯一可数的差别：**假设由谁写下来**——")
    print("    编译器替你推（自动分片），还是你在函数体里显式写（shard_map）。")
    print("    官方说两者「complementary to, and composable with」。")
    return mesh, xs, auto, sm, auto_out, sm_out


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def section_B(mesh, xs, auto, sm, auto_out, sm_out):
    J.rule("§B 代价模型：集合通信个数、设备数、分片切法")
    c = J.Checks()

    c.B(auto.inserted_collectives == 1,
        "`jit` 的自动分片：**用户写了 0 个集合通信，分区器插 1 个 `psum`**",
        "auto.inserted_collectives = %d" % auto.inserted_collectives)
    c.B(sm.inserted_collectives == 0 and sm.user_collectives == 1,
        "`shard_map`：**包装器插 0 个**，那 1 个 `psum` 是**用户写的**",
        "wrapper 插 %d 个，用户写 %d 个" % (sm.inserted_collectives,
                                            sm.user_collectives))
    c.B(sm.trace_count == 1,
        "★ `shard_map` 的函数体**只 trace 一次**（不是每设备一份）",
        "trace_count = %d" % sm.trace_count)

    # 设备数变了会怎样
    for n_dev in (2, 4):
        m = J.Mesh(["d%d" % i for i in range(n_dev)], ("x",))
        data = J.Array(np.arange(8, dtype=np.float32))
        s = J.shard_map(lambda v: lax.psum(J.numpy.reduce_sum(v), "x"), m,
                        [J.PartitionSpec("x")], [J.PartitionSpec()])
        out = s(data)
        c.B(abs(float(out.data) - float(np.sum(data.data))) < 1e-5,
            "设备数 = %d 时，`psum` 的结果仍是**全局和**" % n_dev,
            "%.1f（参照 %.1f）" % (float(out.data), float(np.sum(data.data))))

    # 切法：沿轴切，每台一份
    parts = J._split_on_axis(xs.data, 2)
    c.B(all(p.shape == (4,) for p in parts) and
        np.array_equal(np.concatenate(parts), xs.data),
        "本模型的分片切法：沿轴切成 2 份、拼回来等于原数组",
        "分片形状 %s" % [p.shape for p in parts])

    # A 类：与独立参照的设备内求和比对
    want = ref_psum_shards(parts)
    got = np.asarray(sm_out.data, dtype=np.float64)
    c.A(abs(float(got) - float(np.sum(want))) < 1e-5,
        "`psum` 的结果与独立参照「把各分片按元素加起来再求和」一致",
        "实测 %.1f，参照 %.1f" % (float(got), float(np.sum(want))))
    return c


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def section_C():
    J.rule("§C 边界与反例：每条失败行为都要真的跑出来")
    c = J.Checks()

    J.subsection("C1 对 traced 值取 bool（Python `if`）必须抛错")
    @J.jit
    def bool_trap(p, x):
        if p > 0:                       # 谓词是 traced 值
            return x + 1.0
        return x - 1.0

    c.expect_raises(lambda: bool_trap(J.Array(np.float32(1.0)),
                                      J.Array(np.float32(0.5))),
                    J.TracerBoolConversionError,
                    "对 traced 值取 bool ⇒ TracerBoolConversionError")
    print("    ⚠ 诚实标注：**类名沿用 JAX，报错措辞是本模型自写的**——")
    print("      本仓库**没有取到** `TracerBoolConversionError` 报错的逐字一手来源，")
    print("      所以这里只作**机制**演示，不作原文引用（见 specs 的 UNSOURCED 段）。")
    print("    ★ 这是 `lax.cond` 存在的**理由**：它的谓词**可以**是 traced 值——")
    print("      「the choice of branch happens on the device, at run time, inside the")
    print("       compiled program.」（官方逐字，见 specs.LAX_COND_VERBATIM_DEV）")

    J.subsection("C2 ★ `while_loop` 的反向模式**必须**失败（scan 则成功）")
    @J.jit
    def w(v):
        return lax.while_loop(lambda t: t > 1.0, lambda t: t - 1.0, v)

    # 前向模式：可以
    _p, t = J.jvp(lambda z: w(z), (J.Array(np.float32(5.3)),),
                  (J.Array(np.float32(1.0)),))
    c.C(abs(float(t.data) - 1.0) < 1e-6,
        "`while_loop` 的**前向**模式微分是通的（切向 1.0 原样穿过）",
        "jvp 切向 = %.6f" % float(t.data))
    c.expect_raises(lambda: J.grad(lambda z: w(z))(J.Array(np.float32(5.3))),
                    NotImplementedError,
                    "★ `grad`（反向模式）穿过 `while_loop` ⇒ NotImplementedError")
    print("    官方逐字（%s）：" % SPECS.WHILE_FORWARD_ONLY_SRC)
    print("      %s" % SPECS.WHILE_FORWARD_ONLY_VERBATIM)
    print("    ⚠ 措辞提示：官方 docstring 用的是**反向排除**（not reverse-mode")
    print("      differentiable）；「only forward-mode differentiable」那句在 **dev 文档**里")
    print("      （%s）——**两者不是同一句**。" % SPECS.WHILE_FORWARD_ONLY_DEV_URL)

    @J.jit
    def s(v):
        carry, _ys = lax.scan(lambda cc, xi: (cc * xi, cc),
                              v, J.Array(np.array([1.0, 2.0, 3.0], dtype=np.float32)))
        return carry

    g = J.grad(lambda z: s(z))(J.Array(np.float32(1.0)))
    c.C(abs(float(g.data) - 6.0) < 1e-5,
        "对照：同一件事用 `scan` 写 ⇒ **反向模式可用**（官方指的就是这条路）",
        "d(carry)/d(init) = %.6f（应为 6）" % float(g.data))

    J.subsection("C3 `pmap` 与 `pjit` 各自发废弃告警（但处置不同）")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pm = J.pmap(lambda v: v * 2.0, axis_name="i")
    msgs = [str(w.message) for w in caught]
    c.C(len(msgs) == 1 and msgs[0] == SPECS.PMAP_DEPRECATION_MESSAGE
        and caught[0].category is DeprecationWarning,
        "`pmap` 发一条 DeprecationWarning，文案与官方 docstring 逐字一致",
        "告警 = %r" % (msgs[0] if msgs else None))
    print("    ★ 注意：它**还能用**（只是被劝退）——官方没给移除日期：")
    print("      %s" % SPECS.PMAP_DOC_RENDERED)

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        J.pjit(lambda v: v * 2.0)
    msgs2 = [str(w.message) for w in caught2]
    c.C(len(msgs2) == 1 and msgs2[0] == SPECS.PJIT_DEPRECATION_MESSAGE
        and caught2[0].category is DeprecationWarning,
        "`pjit` 发的告警与源码里的消息逐字一致",
        "告警 = %r" % (msgs2[0] if msgs2 else None))
    c.C(msgs != msgs2,
        "★ 两者**不是同一条消息**：pmap 说「换个 API」，pjit 说「用 jax.jit」",
        "pmap = %r ；pjit = %r" % (msgs[0][:38], msgs2[0][:38]))

    J.subsection("C4 缺 batching rule 的 primitive 不能 vmap（与 02 的同一机制）")
    c.expect_raises(lambda: J.vmap(lambda v: J.numpy.stack([v, v]))(
        J.Array(np.zeros((4, 3), dtype=np.float32))),
        NotImplementedError,
        "`stack` 没有 batcher ⇒ vmap 抛 NotImplementedError")
    print("    官方源码逐字：%s" % SPECS.BATCHING_RULE_MISSING_SOURCE)

    J.subsection("C5 `jit` 没有 dynamic shape 开关——形状多态在 export 那一层")
    print("    官方逐字（%s）：" % SPECS.EXPORT_SYMBOLIC_URL)
    print("      %s" % SPECS.EXPORT_SYMBOLIC_VERBATIM)
    g2 = J.jit(lambda z: z * 2.0)
    ex = J.export.export(g2)(J.Array(np.zeros((3,), dtype=np.float32)))
    ex(J.Array(np.ones((3,), dtype=np.float32)))
    before = ex.recompiles
    ex(J.Array(np.ones((5,), dtype=np.float32)))
    c.C(before == 0 and ex.recompiles == 1,
        "★ 「形状多态」的导出产物**换了形状仍要重编译**（只省下 tracing 与 lowering）",
        "重编译次数 %d → %d" % (before, ex.recompiles))
    return c


# ---------------------------------------------------------------------------
# §3 与 PyTorch 的心智差异（本文的收束）
# ---------------------------------------------------------------------------

def section_3():
    J.rule("§3 与 PyTorch 的心智差异：三者默认都对 shape 敏感")
    rows = [
        ("缓存/失效机制",
         "按 FunctionType（各实参 TraceType 拼成）查表",
         "cache key = JAX types + 静态参数值 + 函数对象 identity",
         "guards：trace 时装布尔断言，调用时逐个校验"),
        ("shape 变化", "retrace", "重编译", "重编译"),
        ("dtype 变化", "retrace", "重编译", "guard 失败 → 重编译"),
        ("Python 标量",
         "**按值特化**（LiteralTraceType<3>）",
         "**默认不特化**（weak-typed traced array）",
         "看用法（depends on its usage）"),
        ("缓解手段",
         "input_signature / None 维 / reduce_retracing",
         "无 jit 级开关；padding 到固定 bucket；export 的形状多态**仍重编译**",
         "mark_dynamic / mark_static；dynamic=True 官方标 Not recommended"),
        ("诊断",
         "experimental_get_tracing_count()、自动重追踪告警",
         "jax_explain_cache_misses / jax_log_compiles / JAX_DUMP_IR_TO",
         "TORCH_LOGS=guards|recompiles、tlparse"),
    ]
    w1, w2, w3 = 14, 30, 34
    print("  %-*s | %-*s | %-*s | %s" % (w1, "维度", w2, "TF tf.function", w3,
                                         "JAX jax.jit", "PyTorch (Dynamo)"))
    print("  " + "-" * (w1 + w2 + w3 + 30))
    for a, b, cc, d in rows:
        print("  %-*s | %-*s | %-*s | %s" % (w1, a, w2, b, w3, cc, d))
    print()
    print("  ★ 三列的证据强度不一样，引用时要分清：")
    print("    · **JAX 这一列**：本仓库有 specs.py 的一手 URL（本文件与 01/02 都在引）；")
    print("    · **TF 那一列**：见 70-frameworks/tensorflow 的 specs.py（本仓库不复核其口径）；")
    print("    · **PyTorch 那一列**：本机有真 torch，属「真框架实测」档——**但本域不重复测它**。")
    print()
    print("  ★ 一条最容易记错的：**「Python 标量是静态的」这句话在 TF 里对、在 JAX 里错。**")
    print("    本例的 C1（03 的 C2 与 01 的 C1）已经把它跑出来了。")
    return None


def main():
    J.header("03", "分片与心智差异 —— 多设备那条路的演进，以及它与 PyTorch 的差别")
    pick, x = section_1()
    section_2_result = section_2()
    parts = [section_A(pick, x), section_B(*section_2_result), section_C()]
    section_3()
    return J.Checks.merge(parts).summary()


if __name__ == "__main__":
    sys.exit(main())
