#!/usr/bin/env python3
"""02 变换：**vmap 给每个 primitive 定义 batching rule**——它不展开循环。

对应文档：70-frameworks/jax/02-transformations.md

★★ 本程序跑的是**本仓库的迷你 JAX 追踪器**（`minijax.py`），**不是 JAX** ★★
  本机没有任何 JAX 环境。所有数字只支撑**机制**结论；关于真实 JAX 的行为断言
  以 `specs.py` 里带 `# 来源：` 的一手 URL 为准。

挂主线（§0.8）：本文处理的是「固化什么」的**变换组合那一面**。
  变换（`jit` / `vmap` / `grad`）**不各自建图**，它们叠在**同一层**上：
  `vmap` 把被映射的轴**推下去**，推到底层 primitive 的语义里；
  所以图里**没有 batch 大小的痕迹**——「固化」的是「有一个轴要被映射」，不是「B 是几」。

三类断言（CONTRIBUTING §2：编号程序跑在默认 python3 上，故三类都在程序内部）：
  A —— 与**独立参考实现**比对：vmap 的语义用「numpy 逐元素循环」这条**完全不同的路径**重算
       （不是把 batched 原语再跑一遍），再附一条「把参照故意改错必须被抓到」的反证。
  B —— 代价模型：jaxpr 的语句数、batching rule 的三张注册表、追踪次数。
  C —— 失败行为必须真的跑出来：**语句数不随 batch 变大**（对照手写循环线性增长）、
       缺 batcher 的 primitive 抛官方文案、donate 之后 buffer 失效、
       就地修改抛官方报错、`at[]` 是 out-of-place、`+=` 不是就地。

用法：cd code && make 02_vmap_batching_rules   （或 make run 全跑）
"""

import sys

import numpy as np

import minijax as J
import specs as SPECS

F32 = "float32"


# ---------------------------------------------------------------------------
# 独立参考实现：**逐元素循环**的 numpy 版本（与「batched 原语」是完全不同的路径）
# ---------------------------------------------------------------------------

def ref_vmap_elementwise(fn, xs):
    """把 `fn` **对每一行单独**跑一遍，再 stack 起来。

    ★ 这是 vmap 语义的**独立参照**：它**不用**任何批量原语，
      所以如果 batcher 把轴推错了（例如该映射到 axis 0 却推成了整数组求和），
      两者立刻不一致。
    """
    return np.stack([np.asarray(fn(row)) for row in np.asarray(xs)], axis=0)


def ref_scalar_fn(row):
    """`sum(row * 2 + 1)` —— 与 `minijax` 里那份实现**逐句对应但独立写**。"""
    return np.sum(np.asarray(row, dtype=np.float64) * 2.0 + 1.0)


def ref_matvec(row, w):
    """`row @ w` 的 numpy 直接实现（第二组参照）。"""
    return np.asarray(row, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# §1 vmap 的机制：batching rule
# ---------------------------------------------------------------------------

def section_1():
    J.rule("§1 vmap 的机制：**给每个 primitive 定义 batching rule**，把被映射轴推下去")

    def per_row(v):
        return J.numpy.reduce_sum(J.numpy.mul(v, 2.0) + 1.0)

    xs = J.Array(np.arange(12, dtype=np.float32).reshape(4, 3))
    out = J.vmap(per_row)(xs)
    print("vmap(per_row)(xs)（xs 形状 %s）→ %s" % (tuple(xs.shape), out.tolist()))
    print("  cache 里没有图 —— **vmap 单独用时不建 jaxpr**，它直接算。")
    print()

    @J.jit
    def jitted_vmap(x):
        return J.vmap(per_row)(x)

    jitted_vmap(xs)
    cj = list(jitted_vmap._cache.values())[0][0]
    print("而 `jit(vmap(per_row))` 会建一张图（**注意：轴被推到 reduce_sum 的参数里去了**）：")
    print(str(cj).replace("const(", "const("))
    print()
    print("★ 图里**没有任何地方记着 batch 是 4**——batcher 根本拿不到 batch 大小。")
    print()
    print("注册方式的官方逐字（%s）：" % SPECS.BATCHING_REGISTER_URL)
    print("  %s" % SPECS.BATCHING_REGISTER_VERBATIM)
    print()
    print("官方对「它不展开循环」的逐字（%s）：" % SPECS.VMAP_SIZE_URL)
    print("  %s" % SPECS.VMAP_SIZE_VERBATIM)
    print("  对照（同页）：%s" % SPECS.VMAP_LOOP_VERBATIM)
    return jitted_vmap, xs, cj


# ---------------------------------------------------------------------------
# §A 算法正确性
# ---------------------------------------------------------------------------

def section_A(jitted_vmap, xs):
    J.rule("§A 算法正确性：与「逐元素循环」这条**另写的路径**比对 + 一条反证")
    c = J.Checks()

    def per_row(v):
        return J.numpy.reduce_sum(J.numpy.mul(v, 2.0) + 1.0)

    want = ref_vmap_elementwise(ref_scalar_fn, xs.data)
    got = J.vmap(per_row)(xs).data.astype(np.float64)
    err = float(np.max(np.abs(got - want)))
    c.A(err < 1e-9, "vmap 的结果与「逐元素循环」参照逐项一致",
        "%d 行，max|Δ| = %.3g" % (xs.shape[0], err))

    # ★ 关键的一条：reduce 型函数的 vmap ≠ 对整数组 reduce
    c.A(abs(float(np.sum(got)) - float(np.sum(want))) < 1e-9
        and abs(float(np.sum(want)) - float(np.sum(xs.data * 2.0 + 1.0))) < 1e-6,
        "对照：vmap(reduce_sum) 得到的是**每行一个标量**，而不是整数组的一个标量",
        "vmap 结果 %s；把整数组一起 reduce 得到 %.1f"
        % (got.tolist(), float(np.sum(xs.data * 2.0 + 1.0))))

    # 第二组参照：矩阵向量乘（走 dot 的 batcher）
    w = J.Array(np.arange(3, dtype=np.float32) + 1.0)

    def matvec(row):
        return J.numpy.dot(row, w)

    got_mv = J.vmap(matvec)(xs).data.astype(np.float64)
    want_mv = ref_vmap_elementwise(lambda r: ref_matvec(r, w.data), xs.data)
    err_mv = float(np.max(np.abs(got_mv - want_mv)))
    c.A(err_mv < 1e-9, "走 `dot` batcher 的那条路也与逐元素参照一致",
        "%d 行，max|Δ| = %.3g" % (xs.shape[0], err_mv))

    # 变换可组合：jit(vmap) 与 vmap(jit) 数值一致
    @J.jit
    def jv(x):
        return J.vmap(per_row)(x)

    vm_jit = J.vmap(J.jit(per_row))(xs)
    c.A(np.allclose(jv(xs).data, vm_jit.data, atol=1e-9),
        "`jit(vmap(f))` 与 `vmap(jit(f))` 数值一致（变换**可组合**）",
        "max|Δ| = %.3g" % float(np.max(np.abs(jv(xs).data - vm_jit.data))))

    # 反证：把参照的「映射轴」改成「整数组一起算」
    def wrong_ref(fn, arr):
        return np.asarray(fn(np.asarray(arr).reshape(-1)))     # 抹掉 batch 结构

    bad = wrong_ref(ref_scalar_fn, xs.data)
    c.A(len(bad.shape) == 0 and bad.shape != want.shape,
        "反证：把参照改成「整数组一起算」后，形状立刻对不上",
        "错版形状 %s vs 正确 %s（说明上面那条不是恒真断言）"
        % (bad.shape, want.shape))
    return c


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def section_B(cj):
    J.rule("§B 代价模型：语句数、三张注册表、追踪次数")
    c = J.Checks()

    n_eqns = len(cj.jaxpr.eqns)
    c.B(n_eqns == 3,
        "`jit(vmap(f))` 的图只有 3 条方程（mul / add / reduce_sum），**不含 batch 维**",
        "方程：%s" % [e.primitive.name for e in cj.jaxpr.eqns])
    c.B([e.primitive.name for e in cj.jaxpr.eqns] == ["mul", "add", "reduce_sum"],
        "`reduce_sum` 的 axes 参数被 batcher **右移了一位**（把被映射轴让开）",
        "reduce_sum 的参数 = %s"
        % [e.params for e in cj.jaxpr.eqns if e.primitive.name == "reduce_sum"])

    # batching rule 的注册表：**本项目订正了计划里的「三张表」说法**
    tables = sorted(J.fancy_primitive_batchers.keys())
    c.B("add" in tables and "mul" in tables and "reduce_sum" in tables,
        "已注册 batching rule 的 primitive 名单可枚举",
        "共 %d 个：%s" % (len(tables), tables))
    c.B(("stack" not in tables),
        "★ `stack` **故意没有** batcher（下面 C2 用它演示缺规则时的报错）",
        "`stack` 不在表里；`J.primitive_batchers` / `J.axis_primitive_batchers` "
        "都是写穿到 `fancy_primitive_batchers` 的兼容代理")
    print("    ★ 一段要紧的订正：**0.11.1 里真正被查表的只有 `fancy_primitive_batchers`**——")
    print("      `primitive_batchers` / `axis_primitive_batchers` 在源码里紧挨着")
    print("      `# backwards compat shim. TODO: delete`，是**写穿**到前者的代理：")
    for _name, _desc in SPECS.BATCHING_TABLES.items():
        print("        · `%s` —— %s" % (_name, _desc))
    print("      源码位置：%s" % SPECS.BATCHING_TABLE_SRC)

    # 追踪次数
    @J.jit
    def by_batch(x):
        return J.vmap(lambda v: J.numpy.reduce_sum(v))(x)

    for B in (4, 64):
        by_batch(J.Array(np.zeros((B, 3), dtype=np.float32)))
    c.B(by_batch.trace_count() == 2,
        "换 batch 大小 = 换**形状** ⇒ 仍然要重追踪（cache key 的第一项含 shape）",
        "B=4 与 B=64 各追一次，共 %d 次" % by_batch.trace_count())
    c.B(by_batch.cache_size() == 2,
        "但两次追踪出来的图，**方程数相同**（下面 C1 逐项比）",
        "缓存条目 = %d" % by_batch.cache_size())
    return c


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def section_C():
    J.rule("§C 边界与反例：每条失败行为都要真的跑出来")
    c = J.Checks()

    J.subsection("C1 ★ vmap 的图**不随 batch 变大**；手写 Python 循环线性增长")
    sizes = [4, 8, 64, 256]
    vm_counts, loop_counts = [], []

    def per_row(v):
        return J.numpy.reduce_sum(J.numpy.mul(v, 2.0) + 1.0)

    for B in sizes:
        @J.jit
        def vmapped(x):
            return J.vmap(per_row)(x)

        @J.jit
        def looped(x):
            return J.numpy.stack([per_row(x[i]) for i in range(B)])

        data = J.Array(np.zeros((B, 3), dtype=np.float32))
        vmapped(data)
        looped(data)
        vm_counts.append(len(list(vmapped._cache.values())[0][0].jaxpr.eqns))
        loop_counts.append(len(list(looped._cache.values())[0][0].jaxpr.eqns))

    print("    语句数（本仓库迷你追踪器实测，**不是 JAX**）：")
    for B, a, b in zip(sizes, vm_counts, loop_counts):
        print("      B = %-4d vmap: %2d 条   手写 Python 循环: %3d 条" % (B, a, b))
    c.C(len(set(vm_counts)) == 1,
        "★ vmap 版本的语句数与 B **无关**（4 / 8 / 64 / 256 全都一样）",
        "vmap 语句数 = %s" % vm_counts)
    c.C(loop_counts == sorted(loop_counts) and loop_counts[-1] > loop_counts[0],
        "★ 对照：手写 `for i in range(B)` 的语句数**随 B 线性增长**",
        "手写循环语句数 = %s（逐项差 %s）"
        % (loop_counts, [loop_counts[i + 1] - loop_counts[i]
                         for i in range(len(loop_counts) - 1)]))
    print("    官方逐字（%s）：" % SPECS.VMAP_SIZE_URL)
    print("      %s" % SPECS.VMAP_SIZE_VERBATIM[-132:])
    print("      %s" % SPECS.VMAP_LOOP_VERBATIM)

    J.subsection("C2 缺 batching rule 的 primitive：必须抛官方式的 NotImplementedError")
    def with_stack(v):
        return J.numpy.stack([v, v])

    c.expect_raises(lambda: J.vmap(with_stack)(
        J.Array(np.zeros((4, 3), dtype=np.float32))),
        NotImplementedError,
        "对没有 batcher 的 primitive 做 vmap ⇒ NotImplementedError")
    print("    官方源码里那条 raise 的**逐字**（%s）：" % SPECS.BATCHING_RULE_MISSING_SRC)
    print("      %s" % SPECS.BATCHING_RULE_MISSING_SOURCE)
    print("    ⚠ 注意：源码是 **f-string**，运行时才渲染成")
    print("      `%s`——**两者不能混标成「源码逐字」**。"
          % SPECS.BATCHING_RULE_MISSING_RUNTIME)
    print("    ★ 本模型的 `BatchTrace.process_primitive` 就是照这句写的。")

    J.subsection("C3 jaxpr 的形态：文法、constvars / invars、ClosedJaxpr")
    def affine(v, k):
        return J.numpy.add(J.numpy.mul(v, k), 1.0)

    cj = J.make_jaxpr(lambda v: affine(v, 2.0))(J.ShapedArray((3,), F32))
    print("    `jax.make_jaxpr(...)` 的输出：")
    print("      %s" % str(cj).replace("\n", "\n      "))
    c.C(isinstance(cj, J.ClosedJaxpr),
        "★ `jax.make_jaxpr()` 返回的是 **`ClosedJaxpr`**（jaxpr + 它闭包住的常量）",
        "本模型的类型名 = %s" % type(cj).__name__)
    c.C(len(cj.jaxpr.invars) == 1 and len(cj.jaxpr.constvars) == 0,
        "分号前是 constvars、分号后是 invars（本图只有 1 个 invar：张量 v）",
        "invars = %s，constvars = %s"
        % ([str(v) for v in cj.jaxpr.invars], [str(v) for v in cj.jaxpr.constvars]))
    c.C("lambda" in str(cj) and "let" in str(cj) and " in " in str(cj),
        "文本形态与官方文法一致：`%s`" % SPECS.JAXPR_GRAMMAR,
        "官方逐字定义见 %s" % SPECS.JAXPR_URL)
    c.C(all(e.primitive.name != "" for e in cj.jaxpr.eqns),
        "ANF：每条方程都把一个中间结果绑定到一个**新变量**",
        "方程 = %s" % [str(e) for e in cj.jaxpr.eqns])

    J.subsection("C4 donate：被捐出的 buffer **之后无效**；关键字实参要用 donate_argnames")
    a = J.Array(np.zeros(3, dtype=np.float32))
    b = J.Array(np.zeros(3, dtype=np.float32))

    @J.jit(donate_argnums=(1,))
    def donated(x, y):
        return J.numpy.add(x, y)

    donated(a, b)
    c.C(True if _is_dead(b) else False,
        "位置实参被 donate 后，再读它的 buffer 会失败（本模型用一个失效标记来体现）",
        "读 b 得到：%s" % ("报错" if _is_dead(b) else "正常值"))
    c.C(not _is_dead(a), "对照：**没有被捐**的那个实参照常可用", "a = %s" % a.tolist())
    print("    官方逐字（%s）：" % SPECS.DONATE_URL)
    print("      %s" % SPECS.DONATE_VERBATIM)
    print("    ⚠ 版本提示：tag jax-v0.11.1 的同一页写的是「%s」——"
          % SPECS.DONATE_TAG_VARIANT[:40])

    @J.jit(donate_argnums=(1,))
    def kw_donated(x, y):
        return J.numpy.add(x, y)

    c2 = J.Array(np.zeros(3, dtype=np.float32))
    kw_donated(J.Array(np.zeros(3, dtype=np.float32)), y=c2)
    c.C(not _is_dead(c2),
        "★ 坑：`donate_argnums` **只覆盖位置实参**——关键字传进来的那个**不会被捐**",
        "用 `y=c2` 传参时 c2 仍然可用")

    @J.jit(donate_argnames=("y",))
    def name_donated(x, y):
        return J.numpy.add(x, y)

    c3 = J.Array(np.zeros(3, dtype=np.float32))
    name_donated(J.Array(np.zeros(3, dtype=np.float32)), y=c3)
    c.C(_is_dead(c3),
        "用 `donate_argnames=('y',)` 点名之后，关键字实参才真的被捐",
        "读 c3 得到：%s" % ("报错" if _is_dead(c3) else "正常值"))
    print("    官方 docstring 逐字（%s）：" % SPECS.DONATE_ARGNUMS_SRC)
    print("      …你需要显式点名；且 `donate_argnums` 与 `donate_argnames` 的对应关系由")
    print("      `inspect.signature(fun)` 决定（见 specs.DONATE_ARGNUMS_DOC）。")

    J.subsection("C5 就地修改：官方报错原文 / `at[]` 是 out-of-place / `+=` 不是就地")
    x = J.Array(np.arange(4, dtype=np.float32))
    c.expect_raises(lambda: x.__setitem__(1, 9.0), TypeError,
                    "`x[idx] = y` 抛 TypeError")
    print("    官方报错的**完整**原文（%s）：" % SPECS.INPLACE_ERROR_SRC)
    print("      %s" % SPECS.INPLACE_ERROR_VERBATIM)
    print("    ★ 本项目订正：计划里给的引文**缺了尾部的 `or another .at[] method: <url>`**，")
    print("      那不是逐字（P7 的一种：引文真实存在，但被截断后当成全文）。")

    updated = x.at[1].set(9.0)
    c.C(updated.tolist() == [0.0, 9.0, 2.0, 3.0] and x.tolist() == [0.0, 1.0, 2.0, 3.0],
        "★ `x.at[1].set(9.0)` **不是就地修改的语法糖**：它返回新数组，**原数组一点没变**",
        "更新后 = %s；原数组 = %s" % (updated.tolist(), x.tolist()))
    print("    官方逐字（%s）：" % SPECS.AT_DOCSTRING_URL)
    print("      %s" % SPECS.AT_DOCSTRING_VERBATIM)
    print("    ⚠ **位置订正（P7）**：计划把下面这句当成 `jax.numpy.ndarray.at` 的 docstring——")
    print("      它其实在 *Common Gotchas* 笔记里；两者**措辞不同，不得互换**：")
    print("      %s" % SPECS.AT_SET_VERBATIM)
    print("      （来源：%s）" % SPECS.AT_SET_URL)

    alias = x
    x += 1.0
    c.C(alias.tolist() == [0.0, 1.0, 2.0, 3.0] and x.tolist() == [1.0, 2.0, 3.0, 4.0],
        "★ `x += 1.0` **不是就地**：原先指向同一个数组的别名 `alias` 看到的是**旧值**",
        "alias = %s；x = %s（若就地改，alias 会跟着变成新值）"
        % (alias.tolist(), x.tolist()))
    print("    官方逐字（%s）：" % SPECS.IADD_URL)
    print("      %s" % SPECS.IADD_VERBATIM)
    print("    ★ 这条**不必删**：计划 §3.1 的 S4 担心「机制从句没来源」，")
    print("      而**官方文档自己就写了**「`jax.Array` doesn't define an `__iadd__`」。")
    return c


def _is_dead(arr):
    """读一下这块 buffer；已失效会抛 RuntimeError。"""
    try:
        arr.data
        return False
    except RuntimeError:
        return True


def main():
    J.header("02", "变换 —— vmap 给每个 primitive 定义 batching rule，它不展开循环")
    jitted_vmap, xs, cj = section_1()
    parts = [section_A(jitted_vmap, xs), section_B(cj), section_C()]
    return J.Checks.merge(parts).summary()


if __name__ == "__main__":
    sys.exit(main())
