"""JAX 的外部事实源：每条带一行 `# 来源：` 与**可打开的 URL**。

★ 本主题的特殊约束（计划 §0.3）：**本机没有 JAX**。
  所以本文件里**没有一条本机实测值**——本机能跑出来的数（迷你追踪器）只支撑机制结论，
  一律不入本文件，只出现在各编号程序与文档 §5 的「迷你实现实测」档里。

★ 版本坐标（钉死，且有取用日期）：

  · **JAX 本体钉 `jax-v0.11.1`**（源码 tag）。源码行号即该 tag 下的行号；
    取用方式：官方 tarball `https://github.com/jax-ml/jax/archive/refs/tags/jax-v0.11.1.tar.gz`
    解包后本地 `grep -n`，与 `raw.githubusercontent.com/jax-ml/jax/jax-v0.11.1/…` 一致。

  · **官方文档用的是 `https://docs.jax.dev/en/latest/`——它跟的是 `main`，不是 0.11.1 的
    冻结快照。** `docs.jax.dev` **没有 0.11.1 版本页**（`/en/v0.11.1/…` 与 `/en/stable/…`
    都返回 404），所以**拿不到「0.11.1 版的线上文档 URL」**。
    → 凡引「dev 文档」的，本文件一律**显式标注**，并在同一处给出 **tag 内源码文档**
      （`docs/new_docs/…` 或 `docs/…`）里那句话的**另一版措辞**。
      **⚠ 两者不是同一句，不得混标。**（已确证至少三处漂移，见 DOC_DRIFT。）

  · 取用日期：2026-09-17。

★★ 本项目对计划 §0.7.C 的**三处订正**（都属 P7「引文真实存在但贴错位置」）：

  1. **「源码里的三张注册表」**：0.11.1 里只有 `fancy_primitive_batchers` 是真 dict，
     `primitive_batchers` / `axis_primitive_batchers` 是标注 `# backwards compat shim.
     TODO: delete` 的**代理对象**，`__setitem__` 最终都写进前者。
  2. **`x[idx] = y` 的报错原文被截断了**：完整句尾部还有
     「or another `.at[]` method: <url>」。见 INPLACE_ERROR_VERBATIM。
  3. **「inside `jit`-compiled code … optimize the array update to occur in-place」这句不在
     `jax.numpy.ndarray.at` 的 docstring 里**，而在 *Common Gotchas* 笔记；
     `.at` 的 docstring 说的是另一句（「are **guaranteed** to be applied in-place」）。
     两句措辞不同，**不得互相替代**。见 AT_SET_VERBATIM / AT_DOCSTRING_VERBATIM。

UNSOURCED / NOT FOUND（查不到一手来源，**正文不使用**）：

  · **「`weak_type` 参与 jit 的 cache key」**——`ShapedArray.key()` 把 weak_type 并进去是
    **本仓库模型的取舍**；本次没有取到「weak/strong 之分是否另算一个 cache key」的
    官方逐字。**本主题的断言都不依赖它**（C1 只用「`3` 与 `4` 抽象出同一个 aval」）。
  · **「`lax.fori_loop` 何时降级成 `scan`、何时降级成 `while_loop`」**——没有取到一手来源。
    编号程序里 `fori_loop` 落成 `scan` 是**本仓库迷你实现自己定的模型**，
    代码与文档都必须标明，**不得写成 JAX 的事实**。
  · **`pmap` 的废弃起始版本 / 移除日期**——docstring 只说「Old way …」，
    **没有给日期**，故不写。
  · **`jax.export` 是否有「不重编译」的开关**——本次只核实了「形状多态仍会重编译」这一面。
"""

# ---------------------------------------------------------------------------
# 版本坐标
# ---------------------------------------------------------------------------

JAX_VERSION = "0.11.1"                       # 源码 tag：jax-v0.11.1
DOC_HOST = "https://docs.jax.dev/en/latest/"
DOC_KIND = "dev（跟 main，不是 0.11.1 的冻结快照；0.11.1 无线上文档页）"
FETCH_DATE = "2026-09-17"

RAW = "https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.1/"
BLOB = "https://github.com/jax-ml/jax/blob/jax-v0.11.1/"


def src(path, lines=None):
    url = BLOB + path
    if lines:
        url += "#L%d-L%d" % lines
    return url


def raw(path):
    return RAW + path


def doc(path):
    return DOC_HOST + path


# ★ 已确证的**文档漂移**（dev 与 tag 之间措辞不同）——引用时必须二选一并注明：
DOC_DRIFT = [
    "201/jit 的 donation 措辞：dev = 'Donated buffers are invalid afterwards.'；"
    "tag(jax-v0.11.1, docs/new_docs/201/jit.md:581) = 'Donated means gone.'",
    "201/jit 的 cache key 第三项：dev = 'the object identity of the function g itself'；"
    "tag(docs/new_docs/201/jit.md:233) = 'the identity of the function `g` itself'",
    "201/control-flow 的 lax.cond 那句：dev 用句号断句；"
    "tag(docs/new_docs/201/control-flow.md:256-257) 用破折号 '—'",
    "201/control-flow 的 while_loop 微分措辞：dev 说 'only forward-mode differentiable'；"
    "tag 里这句话**不存在**（tag 只有 docstring 里的反向排除措辞）",
]

# ---------------------------------------------------------------------------
# 一、jit 与 cache key（文档 01）
# ---------------------------------------------------------------------------

# 来源：**dev 文档** https://docs.jax.dev/en/latest/201/jit.html（§JIT and caching）
# tag 内的同一篇：docs/new_docs/201/jit.md:228-234（第三项措辞不同，见下面两段）
CACHE_KEY_VERBATIM_DEV = (
    "Suppose we define f = jax.jit(g). When f is first invoked, it gets compiled and "
    "the resulting executable is cached. The cache key includes: the JAX types of the "
    "arguments — new shapes or dtypes trigger a fresh trace and compile; the values of "
    "any static arguments; and the object identity of the function g itself."
)

# 来源：**tag 内文档** docs/new_docs/201/jit.md:228-234（jax-v0.11.1）
# ★ 这是 0.11.1 里那句话的**逐字**形态——第三项是 "the identity of the function `g` itself"，
#   **不是** dev 的 "object identity"。★ 这条也说明「形状/dtype 变了会重追踪」有官方逐字。
CACHE_KEY_VERBATIM_TAG = (
    "Suppose we define `f = jax.jit(g)`. When `f` is first invoked, it gets compiled "
    "and the resulting executable is cached. The cache key includes:\n"
    "- the JAX types of the arguments — new shapes or dtypes trigger a fresh\n"
    "  trace and compile (this is what made the rank-1/rank-2 example above safe);\n"
    "- the values of any static arguments; and\n"
    "- the identity of the function `g` itself."
)

# 来源：**dev 文档**同上，紧随其后的一段
# ★ 这一段的「practical consequence」正是 JAX-C2 的官方依据：**两个内容相同的
#   lambda 是两次编译**。
CACHE_IDENTITY_PRACTICAL = (
    "That last point has a practical consequence: avoid calling jax.jit on temporary "
    "functions defined inside loops or other inner scopes. The cache relies on the "
    "object identity of the function, so freshly-created lambdas and partial objects "
    "— even ones wrapping the same underlying code — look like new functions every "
    "time, and recompile every time:"
)

# 来源：jax/_src/api.py:253-254（`jit` 的 `static_argnums` 参数段，docstring）
# ★ 这是「JAX 的 Python 标量**默认不是 static**」的**一手来源**。
#   ⚠ 计划里把它加粗成 `**no arguments are treated as static**`——**原文没有加粗**，
#     加粗是引用者自己加的（本项目照实去掉）。
JIT_NO_STATIC_VERBATIM = (
    "If neither ``static_argnums`` nor ``static_argnames`` is provided, no arguments "
    "are treated as static. If ``static_argnums`` is not provided but "
    "``static_argnames`` is, or vice versa, JAX uses :code:`inspect.signature(fun)` to "
    "find any positional arguments that correspond to ``static_argnames`` (or vice "
    "versa). If both ``static_argnums`` and ``static_argnames`` are provided, "
    "``inspect.signature`` is not used, and only actual parameters listed in either "
    "``static_argnums`` or ``static_argnames`` will be treated as static."
)
JIT_NO_STATIC_SRC = src("jax/_src/api.py", (253, 254))
JIT_NO_STATIC_DOC = doc("_autosummary/jax.jit.html")

# 来源：jax/_src/api.py:249-251（同一参数段，紧邻上一句）
STATIC_RECOMPILE_VERBATIM = (
    "Static arguments should be hashable, meaning both ``__hash__`` and ``__eq__`` are "
    "implemented, and immutable. Otherwise, they can be arbitrary Python objects. "
    "Calling the jitted function with different values for these constants will "
    "trigger recompilation. Arguments that are not array-like or containers thereof "
    "must be marked as static."
)

# 来源：jax/_src/api.py:226-228（docstring，`fun` 参数段）——弱引用做编译缓存键
JIT_WEAKREF_VERBATIM = (
    "hash and equality operators must be defined. JAX keeps a weak reference to "
    "``fun`` for use as a compilation cache key, so the object ``fun`` must be "
    "weakly-referenceable."
)

# 来源：**dev 文档** https://docs.jax.dev/en/latest/201/jit.html（诊断）
# ★ 这句是「输入类型变了」的诊断入口的官方例子；本模型的 explain 输出照它写。
#   ⚠ 计划 §0.7.C 把它写成「官方诊断手段」是**准确的**，但它属于 **dev 文档**。
EXPLAIN_CACHE_MISSES_VERBATIM = (
    "TRACING CACHE MISS at ...: different input types: * at x, now f32[3] and before "
    "i32[3]"
)

# 来源：**dev 文档** https://docs.jax.dev/en/latest/export/shape_poly.html
# tag 内同文：docs/export/shape_poly.md:46-48 与 docs/new_docs/501/shape-polymorphism.md:49-51
# ★ **「jax.jit 没有 dynamic shape 开关」**这条的落点：形状多态是 `jax.export` 这一层的事，
#   而且**它也不免除重编译**。
EXPORT_SYMBOLIC_VERBATIM = (
    "Note that such functions are still re-compiled on demand for each concrete input "
    "shape they are invoked on. Only the tracing and the lowering are saved."
)
EXPORT_SYMBOLIC_URL = doc("export/shape_poly.html")
EXPORT_SYMBOLIC_URL_ALT = doc("501/shape-polymorphism.html")

# 来源：jax/_src/export/shape_poly.py:1383（签名）
# 公开导出点在 **jax/export.py**（tag 里没有 `jax/export/` 目录）。
SYMBOLIC_SHAPE_SIGNATURE = (
    "jax.export.symbolic_shape(shape_spec, *, constraints=(), scope=None, like=None)"
)
SYMBOLIC_SHAPE_SRC = src("jax/_src/export/shape_poly.py", (1383, 1383))

# ---------------------------------------------------------------------------
# 二、vmap / jaxpr / 就地修改（文档 02）
# ---------------------------------------------------------------------------

# 来源：**dev 文档** https://docs.jax.dev/en/latest/101/transformations.html
# tag 内同文：docs/new_docs/101/transformations.md:366-375（jax-v0.11.1）
# ★ 计划 §3.1 的 **S3**：这两句在 `jax.vmap` docstring 与 `jax/_src/api.py` 里
#   **grep 不到**（已逐处排除），**它们在这篇教程的正文里**。页面 URL 见下。
#   注意 tag 原文里 `per batch element` 带 markdown 强调符（`*per batch element*`）。
VMAP_SIZE_VERBATIM = (
    "This is the same program, operation for operation. The only change is that every "
    "operation gained a batch axis: each `f32[5]` became `f32[2,5]`, and each "
    "`dot_general` picked up a batch dimension. That's what \"`vmap` replaces each "
    "intercepted operation with a batched version\" means concretely: the batching "
    "happens *inside* each operation, where array-level hardware parallelism lives, "
    "and the program stays the same size no matter the batch."
)
VMAP_LOOP_VERBATIM = (
    "Compare the Python-loop version, `manually_batched_convolve`, whose recording "
    "contains a full copy of the body *per batch element*: six `dot_general`s, not "
    "three, and growing linearly with the batch size:"
)
VMAP_SIZE_URL = doc("101/transformations.html")
VMAP_SIZE_SRC = raw("docs/new_docs/101/transformations.md")

# 来源：**dev 文档** https://docs.jax.dev/en/latest/601/jax-primitives.html
# tag 内同文：docs/jax-primitives.md:581
# ★ 注册方式逐字（`batching.primitive_batchers[…] = …`）
BATCHING_REGISTER_VERBATIM = "batching.primitive_batchers[multiply_add_p] = multiply_add_batch"
BATCHING_REGISTER_URL = doc("601/jax-primitives.html")

# 来源：jax/_src/interpreters/batching.py:585 / :593 / :608（源码）
# ★ **本项目订正（P7）**：0.11.1 里只有 `fancy_primitive_batchers` 是真 dict；
#   `primitive_batchers` 与 `axis_primitive_batchers` 都紧挨着
#   `# backwards compat shim. TODO: delete` 注释，是**写穿到前者的代理**。
#   真正被查表的地方是 `:256` 的 `if p in fancy_primitive_batchers:`。
BATCHING_TABLES = {
    "fancy_primitive_batchers": "真 dict（源码 :585：`fancy_primitive_batchers: dict[core.Primitive, Callable] = {}`）",
    "primitive_batchers": "兼容代理（源码 :596 起，注释 `# backwards compat shim. TODO: delete`，写穿到 fancy_primitive_batchers）",
    "axis_primitive_batchers": "兼容代理（源码 :588 起，同上；老签名带 axis_data）",
}
BATCHING_TABLE_SRC = src("jax/_src/interpreters/batching.py", (585, 608))

# 来源：jax/_src/interpreters/batching.py:275（源码）
# ★ 源码里是 **f-string**：`raise NotImplementedError(f"Batching rule for '{p}' not implemented")`。
#   运行时渲染成 `Batching rule for '<primitive>' not implemented`。
#   **两者不能混标成「源码逐字」**——本文件把源码形态与渲染形态分开写。
BATCHING_RULE_MISSING_SOURCE = 'raise NotImplementedError(f"Batching rule for \'{p}\' not implemented")'
BATCHING_RULE_MISSING_RUNTIME = "Batching rule for '<primitive>' not implemented"
BATCHING_RULE_MISSING_SRC = src("jax/_src/interpreters/batching.py", (275, 275))

# 来源：**dev 文档** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
# tag 内同文：docs/notebooks/Common_Gotchas_in_JAX.md:216-219
# ★★ 计划 §3.1 的 **S4**：「`jax.Array` 没定义 `__iadd__`」——**官方文档自己就写了这句**，
#    所以**不需要删**这条机制从句，只需补来源。
#    另一路独立核实：`grep -rn "iadd" jax/` 在 0.11.1 的 `jax/` 包内 **0 处命中**；
#    `jax/_src/numpy/array_methods.py` 的 `_array_operators` 注册表（:1409-1449）里
#    **没有任何 `iadd` 键**。
IADD_VERBATIM = (
    "That's because NumPy defines `__iadd__` to perform in-place mutation. In "
    "contrast, `jax.Array` doesn't define an `__iadd__`, so Python treats "
    "`jax_array_new += 10` as syntactic sugar for `jax_array_new = jax_array_new + 10`, "
    "rebinding the variable without mutating any arrays."
)
IADD_URL = doc("notebooks/Common_Gotchas_in_JAX.html")

# 来源：jax/_src/numpy/array_methods.py:971-975（源码，`_unimplemented_setitem`）
# ★ **本项目订正**：计划 §0.7.C 给的引文**缺了尾部的半句**——完整运行期消息是下面这条。
#   （源码里 msg 由三个相邻字符串字面量拼成。）
INPLACE_ERROR_VERBATIM = (
    "JAX arrays are immutable and do not support in-place item assignment. Instead of "
    "x[idx] = y, use x = x.at[idx].set(y) or another .at[] method: "
    "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html"
)
INPLACE_ERROR_SRC = src("jax/_src/numpy/array_methods.py", (971, 975))
INPLACE_ERROR_URL = IADD_URL

# 来源：**dev 文档**，*Common Gotchas* 笔记（tag 内 docs/notebooks/Common_Gotchas_in_JAX.md:260）
# ⚠ **位置订正（P7）**：这句**不在** `jax.numpy.ndarray.at` 的 docstring 里。
AT_SET_VERBATIM = (
    "However, inside __jit__-compiled code, if the __input value__ `x` of "
    "`x.at[idx].set(y)` is not reused, the compiler will optimize the array update to "
    "occur _in-place_."
)
AT_SET_URL = IADD_URL

# 来源：**dev 文档** https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html
# ★ 与上一句**措辞不同**（「保证就地」vs「编译器会优化成就地」）——**两句不得互相替代**。
AT_DOCSTRING_VERBATIM = (
    "None of the x.at expressions modify the original x; instead they return a "
    "modified copy of x. However, inside a jit() compiled function, expressions like "
    "x = x.at[idx].set(y) are guaranteed to be applied in-place."
)
AT_DOCSTRING_URL = doc("_autosummary/jax.numpy.ndarray.at.html")

# 来源：**dev 文档** https://docs.jax.dev/en/latest/201/jit.html（§Buffer donation）
# tag 内**同一页的另一版措辞**：`docs/new_docs/201/jit.md:581` 是
# 「**Donated means gone.** After the call, the donated input is invalid, and using it
# is an error:」。**两者不是同一句**——引 dev 版就不能说它是 0.11.1。
DONATE_VERBATIM = (
    "Donated buffers are invalid afterwards. Using the donated input after the call "
    "is an error:"
)
DONATE_URL = doc("201/jit.html")
DONATE_TAG_VARIANT = ("Donated means gone. After the call, the donated input is "
                      "invalid, and using it is an error:")
DONATE_TAG_SRC = raw("docs/new_docs/201/jit.md")

# 来源：jax/_src/api.py:267-288（`jit` 的 `donate_argnums` 参数段，docstring）
DONATE_ARGNUMS_DOC = (
    "donate_argnums: optional, collection of integers to specify which positional "
    "argument buffers can be overwritten by the computation and marked deleted in the "
    "caller. … You should not reuse buffers that you donate to a computation; JAX will "
    "raise an error if you try to. By default, no argument buffers are donated. If "
    "neither ``donate_argnums`` nor ``donate_argnames`` is provided, no arguments are "
    "donated. … If both ``donate_argnums`` and ``donate_argnames`` are provided, "
    "``inspect.signature`` is not used, and only actual parameters listed in either "
    "``donate_argnums`` or ``donate_argnames`` will be donated."
)
DONATE_ARGNUMS_SRC = src("jax/_src/api.py", (267, 288))

# 来源：jax/_src/lax/control_flow/loops.py:1641-1643（`while_loop` 的 docstring，源码）
# ★ **while_loop 只支持前向模式微分**的一手来源——tag 里用的是**反向排除**的措辞。
#   ⚠ 计划 §0.7.C 写的「只支持前向模式微分」那句**在 tag 里不存在**，
#     只在 **dev 文档**（201/control-flow）里有；本文件把两版都记下来：
WHILE_FORWARD_ONLY_VERBATIM = (
    "Another difference from using Python-native loop constructs is that ``while_loop`` "
    "is not reverse-mode differentiable because XLA computations require static bounds "
    "on memory requirements."
)
WHILE_FORWARD_ONLY_SRC = src("jax/_src/lax/control_flow/loops.py", (1641, 1643))
WHILE_FORWARD_ONLY_DOC = doc("_autosummary/jax.lax.while_loop.html")
# dev 文档的正面措辞（**与上面不是同一句**）：
WHILE_FORWARD_ONLY_DEV_VERBATIM = (
    "Note the differentiability annotation above: while_loop is only forward-mode "
    "differentiable. Reverse-mode autodiff needs to run the loop backwards, saving "
    "each iteration's intermediates on the way forward, which requires a bound on the "
    "number of iterations, and a while_loop's trip count is dynamic and unbounded. For "
    "reverse-mode differentiation through a loop, use scan (fixed length), or "
    "fori_loop with static bounds (which lowers to scan)."
)
WHILE_FORWARD_ONLY_DEV_URL = doc("201/control-flow.html")

# 来源：jax/_src/api.py / jax/_src/lax/control_flow/conditionals.py 无此句；
#   实际在 **dev 文档** https://docs.jax.dev/en/latest/201/control-flow.html
# tag 内同文（**破折号版**）：docs/new_docs/201/control-flow.md:256-257
LAX_COND_VERBATIM_DEV = (
    "Unlike a Python if, the predicate here can be a traced value. The choice of "
    "branch happens on the device, at run time, inside the compiled program."
)
LAX_COND_VERBATIM_TAG = (
    "Unlike a Python `if`, the predicate here can be a traced value — the choice of "
    "branch happens on the device, at run time, inside the compiled program."
)
LAX_COND_URL = doc("201/control-flow.html")
LAX_COND_TAG_SRC = raw("docs/new_docs/201/control-flow.md")

# ---------------------------------------------------------------------------
# 三、jaxpr（文档 02 §2）
# ---------------------------------------------------------------------------

# 来源：**dev 文档** https://docs.jax.dev/en/latest/601/jaxpr.html（jaxpr 专页）
# （tag 内的对应文件是 docs/jaxpr.md，同一段定义。）
JAXPR_VERBATIM = (
    "JAXPR is a representation of a functional program, made up of a sequence of "
    "typed equations. It is explicitly typed, functional, first-order, and in "
    "algebraic normal form (ANF)."
)
JAXPR_GRAMMAR = "jaxpr ::= { lambda Var* ; Var+. let Eqn* in [ Expr+ ] }"
JAXPR_URL = doc("601/jaxpr.html")
JAXPR_MAKE_NOTE = ("`jax.make_jaxpr()` 的返回值是 **`ClosedJaxpr`**（jaxpr + 它闭包住的常量）。")

# ---------------------------------------------------------------------------
# 四、多设备：pmap / pjit / shard_map（文档 03 §2）
# ---------------------------------------------------------------------------

# 来源：jax/_src/pmap.py:70（docstring，源码）
# 渲染后逐字：`Old way of doing parallel map. Use jax.shard_map() instead.`
# 文档页：https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html
PMAP_DOC_VERBATIM = "Old way of doing parallel map. Use :py:func:`jax.shard_map` instead."
PMAP_DOC_RENDERED = "Old way of doing parallel map. Use jax.shard_map() instead."
PMAP_DOC_SRC = src("jax/_src/pmap.py", (70, 70))
PMAP_DOC_URL = doc("_autosummary/jax.pmap.html")

# 来源：jax/_src/pmap.py:79-81（docstring，源码）
# 渲染后逐字：`pmap() is now implemented in terms of jit() and shard_map().`
PMAP_IMPL_VERBATIM = (
    ":py:func:`pmap` is now implemented in terms of :py:func:`jit` and "
    ":py:func:`shard_map`. Please see the `migration guide "
    "<https://docs.jax.dev/en/latest/migrate_pmap.html>`_ for more information."
)
PMAP_IMPL_RENDERED = ("pmap() is now implemented in terms of jit() and shard_map(). "
                      "Please see the migration guide for more information.")
PMAP_IMPL_SRC = src("jax/_src/pmap.py", (79, 81))

# 来源：jax/_src/pmap.py:88-91（docstring，源码）
# ★ **本项目订正**：计划 §0.7.C 把这句话引成
#   「`vmap()` vectorizes … , `pmap()` instead replicates …」——**拆掉了原文的
#   `but where … , … instead` 关联结构**，那不是逐字的原句。下面给**完整句**。
PMAP_VMAP_CONTRAST_VERBATIM = (
    "Semantically it is comparable to :py:func:`vmap` because both transformations map "
    "a function over array axes, but where :py:func:`vmap` vectorizes functions by "
    "pushing the mapped axis down into primitive operations, :py:func:`pmap` instead "
    "replicates the function and executes each replica on its own XLA device in "
    "parallel."
)
PMAP_VMAP_CONTRAST_SRC = src("jax/_src/pmap.py", (88, 91))

# 本模型用的告警文案（**渲染后的形态**，照 docstring 的语义写；本模型自己发这条告警）
PMAP_DEPRECATION_MESSAGE = PMAP_DOC_RENDERED

# 来源：jax/experimental/pjit.py:20-26（tag jax-v0.11.1 下是 37 行的纯壳，逐字）
# ★ `pjit` **已并入 `jax.jit`**；源码注释里给了精确日期 **Added Oct 13, 2025**。
PJIT_MERGE_VERBATIM = (
    "_deprecations = {\n"
    "    # Added Oct 13, 2025\n"
    '    "pjit": (\n'
    "        (\n"
    '            "jax.experimental.pjit.pjit has been deprecated. Please use"\n'
    '            " `jax.jit` instead."\n'
    "        ),\n"
    "        _deprecated_pjit,\n"
    "    )\n"
    "}"
)
PJIT_DEPRECATION_MESSAGE = ("jax.experimental.pjit.pjit has been deprecated. "
                            "Please use `jax.jit` instead.")
PJIT_DEPRECATION_DATE = "Oct 13, 2025"
PJIT_MERGE_SRC = src("jax/experimental/pjit.py", (20, 26))

# 来源：jax/_src/api.py:184 / :201 / :217（三处返回类型注解，含两个 overload）
# ★ 这是「`jax.jit` 的返回类型就叫 `pjit.JitWrapped`」的源码依据。
JIT_RETURN_TYPE = "-> pjit.JitWrapped"
JIT_RETURN_TYPE_SRC = src("jax/_src/api.py", (184, 217))

# 来源：jax/_src/pjit.py:708（真正的实现留在 `_src/pjit.py`）
PJIT_IMPL_DOC = "\"\"\"`jax.experimental.pjit.pjit` has been deprecated. Please use `jax.jit`.\"\"\""

# 来源：**dev 文档** https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html
# （同文也在 jax/_src/shard_map.py 的 docstring 里。）
SHARD_MAP_VERBATIM = (
    "shard_map is a single-program multiple-data (SPMD) multi-device parallelism API "
    "… Mapped function applications, or instances, communicate with each other via "
    "explicit collective communication operations."
)
SHARD_MAP_COMPOSABLE = (
    "shard_map is complementary to, and composable with, the automatic "
    "compiler-based parallelization built into jit."
)
SHARD_MAP_URL = doc("_autosummary/jax.shard_map.html")

# ---------------------------------------------------------------------------
# 五、与 TF 的镜像对照（文档 01 §1.3 / 文档 03 §3）
# ---------------------------------------------------------------------------
# 本主题能一手核实的是 **JAX 这一列**；TF 那一列请回链 70-frameworks/tensorflow。
# ★ 对照的结论：**TF 与 JAX 对 Python 标量的处理正好相反**。
#   JAX 侧有来源：官方 docstring 逐字 "no arguments are treated as static"
#   （见 JIT_NO_STATIC_VERBATIM）⇒ Python 标量是一个**被 trace 的实参**，
#     于是它进 jaxpr 的 **invars**，而且 `3` 与 `4` 共享同一个 cache key。
#   TF 侧见 70-frameworks/tensorflow/code/specs.py 的 TRACETYPE_RULES["Python 原生值"]
#   （`LiteralTraceType<3>`，按**值**进键、并被烧进图）。
TF_VS_JAX_SCALAR_NOTE = (
    "JAX 侧有来源：no arguments are treated as static ⇒ 标量是被 trace 的实参。"
    "TF 侧见 70-frameworks/tensorflow —— 本文件不替它下断言。")
