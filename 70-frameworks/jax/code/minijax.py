"""本仓库的**迷你 JAX 追踪器**：把 `jax.jit` 的「按 cache key 查表 → 命中复用，
不命中重追踪」、`jax.vmap` 的「给每个 primitive 定义 batching rule」、
以及 `jit` / `vmap` / `grad` 三者的**组合**用一千多行 Python 重写一遍。

★★ 这不是 JAX ★★
  本文件是**本仓库自己写的模型**，只依赖 `numpy` 与标准库。它借用了 JAX 的**术语**
  （jit / Tracer / ShapedArray / jaxpr / primitive / batching rule / shard_map …）
  以便对照，**结构是我们自己简化的**。凡由它跑出的数字，只能支撑
  「这套机制可以怎样被计数与观测」这类**机制**结论，**不支持任何关于真实 JAX 的结论**。
  关于真实 JAX 的每一条行为断言，见 `specs.py`（每条带 `# 来源：` 与可打开的 URL）。

★★ 与 `70-frameworks/tensorflow/code/minitf.py` 的关系 ★★
  两者是**镜像**：同一套「追踪成图 / 按假设查表」的骨架，但**假设清单的内容不同**。
  最要紧的一条对照是 **Python 标量**：
    · minitf：Python 原生值按**值**进 TraceType，并被**烧成图上的字面量**；
    · minijax（本文件）：Python 标量**默认不是 static**（官方 docstring 逐字
      "no arguments are treated as static"，见 specs.JIT_NO_STATIC_VERBATIM），
      它会变成一个 **weak-typed 的 traced 实参** ⇒ 进 jaxpr 的 **invars**，
      并且 **`3` 与 `4` 共享同一个 cache key**。
  两边都在各自程序的 C 类断言里把这条跑出来。

本文件提供的对象：

  · `ShapedArray` / `abstractify`      —— 「JAX types」：形状 + dtype + weak_type
  · `Array`                            —— eager 值（**不可变**，没有 view）
  · `Primitive` / `bind`               —— primitive 注册表
  · `Jaxpr` / `Eqn` / `Var` / `Literal` / `ClosedJaxpr`
                                       —— 官方文法的 jaxpr（ANF）
  · `Tracer` / `JaxprTracer` / `BatchTracer`
  · `jit` / `vmap` / `grad` / `jvp` / `vjp` / `make_jaxpr`
  · `lax`                              —— `cond` / `while_loop` / `fori_loop` / `scan` / `psum`
  · `Mesh` / `PartitionSpec` / `shard_map` / `pmap` / `pjit`
  · `config` / `clear_caches`          —— `jax_explain_cache_misses` 等诊断
  · `export`                           —— `symbolic_shape` 与「只省下 tracing 与 lowering」
  · `Checks`                           —— 三类断言（A / B / C）的记账器

用法：`python3 <编号程序>.py`（见同目录 Makefile）
"""

import os
import sys
import time
import weakref
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# 一、JAX types：形状 + dtype + weak_type
# ---------------------------------------------------------------------------

_DTYPE_NAMES = {
    "float32": np.dtype("float32"), "float64": np.dtype("float64"),
    "int32": np.dtype("int32"), "int64": np.dtype("int64"),
    "bool": np.dtype("bool"),
}


def _dtype_name(dt):
    dt = np.dtype(dt)
    for name, d in _DTYPE_NAMES.items():
        if d == dt:
            return name
    return str(dt)


def _numpy_dtype(name):
    if name not in _DTYPE_NAMES:
        raise TypeError("mini: 不认识 dtype %r" % (name,))
    return _DTYPE_NAMES[name]


class ShapedArray:
    """JAX 的抽象值（aval）：**形状 + dtype + weak_type**。

    它就是本主题里「cache key 的第一项」——官方把这一项叫 **the arguments' JAX types**
    （见 specs.CACHE_KEY_VERBATIM / specs.SHAPES_DTYPES_VERBATIM）。

    `weak_type` 的来历：Python 标量（`3` / `3.0`）在 JAX 里是**弱类型**的——
    它不与别的 dtype 争类型，而是让位给对面的强类型。本仓库按 aval 的相等性把它并进
    `key()`（**⚠ 这条是本仓库模型的取舍，没有单独取到「weak_type 参与 jit cache key」的
    官方逐字来源**，见 specs 文件头的 UNSOURCED 段；本主题的断言都不依赖 weak/strong 之分）。
    """

    __slots__ = ("shape", "dtype", "weak_type")

    def __init__(self, shape, dtype, weak_type=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.weak_type = bool(weak_type)

    def key(self):
        return (self.shape, self.dtype, self.weak_type)

    def __eq__(self, other):
        return isinstance(other, ShapedArray) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return "%s%s%s" % (self.dtype,
                           "[" + ",".join(str(d) for d in self.shape) + "]",
                           "(weak)" if self.weak_type else "")

    __repr__ = __str__


def _promote_dtype(a, b):
    """dtype 提升：**弱类型让位给强类型**（两边都弱/都强时走 numpy 规则）。"""
    if a.weak_type and not b.weak_type:
        return b.dtype
    if b.weak_type and not a.weak_type:
        return a.dtype
    return _dtype_name(np.promote_types(_numpy_dtype(a.dtype), _numpy_dtype(b.dtype)))


# ---------------------------------------------------------------------------
# 二、eager 值 Array（**不可变**）
# ---------------------------------------------------------------------------

class Array:
    """eager 的 JAX 数组。

    ★ 三条与真实 JAX 一致的性质（本主题反复用到）：
      ① **不可变**：`x[idx] = y` 直接抛 `TypeError`（文案见 specs.INPLACE_ERROR_VERBATIM）；
      ② **没有 view**：`x[i]` 返回的是一个**新数组**，改它不影响 `x`；
      ③ 更新只能 out-of-place：`x.at[idx].set(y)` 返回新数组，**原数组不动**
         （⚠ 官方对这条的说明是「编译器**可能**把它优化成原地」，见 specs.AT_SET_VERBATIM）。

    ★ 本类**不定义 `__iadd__`**——所以 `x += y` 走的是 `__add__`，
      绑定到一个**新对象**，原有别名看到的仍是旧值。这条断言的**可观测面**就是「别名不变」，
      **不依赖**「真实 `jax.Array` 有没有 `__iadd__`」这个机制从句（见 specs.UNSOURCED 段）。
    """

    __slots__ = ("_data", "_invalid", "weak_type")

    def __init__(self, data, dtype=None, weak_type=False):
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(_numpy_dtype(dtype) if isinstance(dtype, str) else dtype)
        self._data = arr
        self._invalid = False
        self.weak_type = bool(weak_type)

    # -- 数据访问 ---------------------------------------------------------
    @property
    def data(self):
        if self._invalid:
            raise RuntimeError(
                "mini: 这块 buffer 已被 donate 掉，之后不能再用了"
                "（官方逐字见 specs.DONATE_VERBATIM：Donated buffers are invalid "
                "afterwards.）——⚠ 这条报错是**本模型加的保护**，真实 JAX 里读到的是"
                "未定义的值，不会替你报错。")
        return self._data

    @property
    def shape(self):
        return tuple(self._data.shape)

    @property
    def dtype(self):
        return _dtype_name(self._data.dtype)

    @property
    def aval(self):
        return ShapedArray(self.shape, self.dtype, self.weak_type)

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def T(self):
        return Array(self._data.T)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter([Array(r) for r in self._data])

    # -- 与 numpy 互操作 --------------------------------------------------
    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def tolist(self):
        return self._data.tolist()

    def __repr__(self):
        return "Array(%s, dtype=%s)" % (
            np.array2string(self._data, separator=", "), self.dtype)

    # -- 取下标：返回**新数组**（JAX 没有 view，改它不影响原数组）------------
    def __getitem__(self, idx):
        return Array(np.array(self._data[idx], copy=True))

    # -- 算术与比较：与 Tracer 走同一条分派 --------------------------------
    def __add__(self, o):
        return bind(PRIMITIVES["add"], self, o)

    def __radd__(self, o):
        return bind(PRIMITIVES["add"], o, self)

    def __sub__(self, o):
        return bind(PRIMITIVES["sub"], self, o)

    def __rsub__(self, o):
        return bind(PRIMITIVES["sub"], o, self)

    def __mul__(self, o):
        return bind(PRIMITIVES["mul"], self, o)

    def __rmul__(self, o):
        return bind(PRIMITIVES["mul"], o, self)

    def __truediv__(self, o):
        return bind(PRIMITIVES["div"], self, o)

    def __rtruediv__(self, o):
        return bind(PRIMITIVES["div"], o, self)

    def __neg__(self):
        return bind(PRIMITIVES["neg"], self)

    def __gt__(self, o):
        return bind(PRIMITIVES["gt"], self, o)

    def __lt__(self, o):
        return bind(PRIMITIVES["lt"], self, o)

    # -- 就地修改：**JAX 数组不允许** --------------------------------------
    def __setitem__(self, idx, value):
        import specs
        raise TypeError(specs.INPLACE_ERROR_VERBATIM)

    # -- 更新语法：out-of-place -------------------------------------------
    @property
    def at(self):
        return _AtHelper(self)


class _AtHelper:
    """`x.at[idx]` —— 官方叫「out-of-place update」的那套语法。

    ⚠ 它不是「就地修改的语法糖」：`.set()` 返回**新数组**。官方给的那半句是
      「inside `jit`-compiled code, if the input value `x` … is not reused, the compiler
       will optimize the array update to occur in-place」——**优化是编译器的可能行为，
       不是这套 API 的语义**（见 specs.AT_SET_VERBATIM）。
    """

    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, idx):
        return _AtIndexUpdate(self._arr, idx)


class _AtIndexUpdate:
    __slots__ = ("_arr", "_idx")

    def __init__(self, arr, idx):
        self._arr = arr
        self._idx = idx

    def get(self):
        return Array(self._arr.data[self._idx])

    def set(self, values):
        out = self._arr.data.copy()
        out[self._idx] = np.asarray(values)
        return Array(out)

    def add(self, values):
        out = self._arr.data.copy()
        out[self._idx] = out[self._idx] + np.asarray(values)
        return Array(out)


def array(value, dtype=None, weak_type=False):
    return Array(value, dtype=dtype, weak_type=weak_type)


def abstractify(value):
    """把实参抽象成 **JAX type**（`ShapedArray`）。这正是 jit cache key 的第一项。

    ★ 关键：Python 标量 → `ShapedArray((), 'int32'/'float32'/'bool', weak_type=True)`。
      **`3` 与 `4` 抽象出来是同一个 `ShapedArray`**——这就是 C1 的全部机制。
    """
    if isinstance(value, (Tracer, Array)):
        return value.aval
    if isinstance(value, np.ndarray):
        return ShapedArray(value.shape, _dtype_name(value.dtype), False)
    if isinstance(value, np.generic):
        return ShapedArray((), _dtype_name(value.dtype), False)
    if isinstance(value, bool):
        return ShapedArray((), "bool", weak_type=True)
    if isinstance(value, int):
        return ShapedArray((), "int32", weak_type=True)
    if isinstance(value, float):
        return ShapedArray((), "float32", weak_type=True)
    raise TypeError(
        "mini: 不支持把 %s 抽象成 JAX type（真实 JAX 对未知类型也会报错）"
        % type(value).__name__)


# ---------------------------------------------------------------------------
# 三、jaxpr：**explicitly typed, functional, first-order, ANF**
# ---------------------------------------------------------------------------

class Var:
    """jaxpr 里的变量（ANF：每个中间结果都绑定到一个名字）。"""

    __slots__ = ("count", "aval")

    def __init__(self, count, aval):
        self.count = count
        self.aval = aval

    def __str__(self):
        return "v%d" % self.count


class Literal:
    """jaxpr 里的常量：**烧进图**的那种（对照：Python 标量实参不是常量，是 invar）。"""

    __slots__ = ("val", "aval")

    def __init__(self, val, aval):
        self.val = np.asarray(val)
        self.aval = aval

    def __str__(self):
        if self.aval.shape:
            return "const(%s)" % np.array2string(self.val, separator=",")
        return "const(%r)" % (self.val.item() if self.val.ndim == 0 else self.val,)


class Eqn:
    """一条方程：`outvars = primitive invars [params]`。"""

    __slots__ = ("primitive", "invars", "outvars", "params", "subjaxprs")

    def __init__(self, primitive, invars, outvars, params, subjaxprs=()):
        self.primitive = primitive
        self.invars = tuple(invars)
        self.outvars = tuple(outvars)
        self.params = dict(params)
        self.subjaxprs = tuple(subjaxprs)      # cond / while / scan 的子 jaxpr

    def __str__(self):
        lhs = ", ".join(str(v) for v in self.outvars)
        rhs = "%s %s" % (self.primitive.name,
                         " ".join(str(v) for v in self.invars))
        extra = ""
        for sub in self.subjaxprs:
            extra += " [子 jaxpr %s: %d 条方程]" % (sub.name, len(sub.eqns))
        if self.params:
            extra += " %s" % _fmt_params(self.params)
        return "%s = %s%s" % (lhs, rhs, extra)


def _fmt_params(params):
    bits = []
    for k in sorted(params):
        if k.startswith("_"):
            continue
        v = params[k]
        if isinstance(v, (list, tuple)):
            v = "[" + ",".join(str(x) for x in v) + "]"
        bits.append("%s=%s" % (k, v))
    return " ".join(bits)


class Jaxpr:
    """`jaxpr ::= { lambda Var* ; Var+. let Eqn* in [ Expr+ ] }`

    官方的逐字定义：「**explicitly typed, functional, first-order, and in algebraic
    normal form (ANF)**」（见 specs.JAXPR_VERBATIM）。文法里 **分号前是 constvars、
    分号后是 invars**——这个区分在本主题里很要紧：
      · **invars** 是「每次调用都要喂进来的东西」（于是 Python 标量实参落在这里）；
      · **constvars** 是「在 jit 时就冻进图里的东西」（`static_argnums` 的值落在这里）。
    """

    __slots__ = ("invars", "constvars", "eqns", "outvars", "name", "_counter")

    def __init__(self, name="jaxpr"):
        self.invars = []
        self.constvars = []
        self.eqns = []
        self.outvars = []
        self.name = name
        self._counter = 0

    def new_var(self, aval):
        v = Var(self._counter, aval)
        self._counter += 1
        return v

    def new_invar(self, aval):
        v = self.new_var(aval)
        self.invars.append(v)
        return v

    def new_constvar(self, aval):
        v = self.new_var(aval)
        self.constvars.append(v)
        return v

    def __str__(self):
        head = "{ lambda %s ; %s. let" % (
            " ".join(str(v) for v in self.constvars),
            " ".join(str(v) for v in self.invars))
        body = "\n    %s" % "\n    ".join(str(e) for e in self.eqns)
        tail = "\n  in (%s) }" % ", ".join(str(v) for v in self.outvars)
        return head + body + tail

    # -- 两个可数的量（C 类断言直接看它们）---------------------------------
    def statement_count(self):
        """**语句数**：本层方程 + 所有子 jaxpr 的方程（递归）。"""
        n = len(self.eqns)
        for e in self.eqns:
            for sub in e.subjaxprs:
                n += sub.statement_count()
        return n

    def op_histogram(self):
        hist = OrderedDict()
        for e in self.eqns:
            hist[e.primitive.name] = hist.get(e.primitive.name, 0) + 1
            for sub in e.subjaxprs:
                for k, v in sub.op_histogram().items():
                    hist[k] = hist.get(k, 0) + v
        return hist


class ClosedJaxpr:
    """`jax.make_jaxpr()` 返回的就是它：**jaxpr + 它闭包住的常量**。

    （官方把 `jax.make_jaxpr` 的返回值叫 `ClosedJaxpr`，见 specs.MAKE_JAXPR_NOTE。）
    """

    __slots__ = ("jaxpr", "consts")

    def __init__(self, jaxpr, consts=()):
        self.jaxpr = jaxpr
        self.consts = list(consts)

    def __str__(self):
        if self.consts:
            return "{ closed over %d consts:\n%s }" % (len(self.consts), self.jaxpr)
        return str(self.jaxpr)


# ---------------------------------------------------------------------------
# 四、Primitive：一张注册表，五套规则（eager / infer / jvp / vjp / batch）
# ---------------------------------------------------------------------------

class Primitive:
    """一个 primitive 携带五套规则；**缺哪套就在用到时抛错**——
    这正是 vmap 与 grad 的报错来源（`Batching rule for '…' not implemented` /
    `Reverse-mode differentiation rule for '…' not implemented`）。
    """

    __slots__ = ("name", "eager", "infer", "jvp", "vjp")

    def __init__(self, name, eager, infer, jvp=None, vjp=None):
        self.name = name
        self.eager = eager            # 具体值上的实现（numpy）
        self.infer = infer            # aval 上的类型推导
        self.jvp = jvp                # 前向模式微分规则
        self.vjp = vjp                # 反向模式（transpose）规则


PRIMITIVES = OrderedDict()


# ★ vmap 的注册表——**名字照 `jax-v0.11.1` 的源码写**（见 specs.BATCHING_TABLES）。
#   ⚠ 本项目订正：计划 §0.7.C 把 `fancy_primitive_batchers` / `primitive_batchers` /
#     `axis_primitive_batchers` 并列为「源码里的三张注册表」，**在 0.11.1 里已不准确**：
#     真正被查表的只有 `fancy_primitive_batchers`；另外两个在源码里紧挨着
#     `# backwards compat shim. TODO: delete` 注释，`__setitem__` 最终都写进前者。
#     本模型照这个真实结构写：一张真表 + 两个写穿（write-through）的兼容代理。
#   ⚠ 另一处模型简化：真实的表以 **Primitive 对象**为键，本模型以**名字字符串**为键。
fancy_primitive_batchers = {}


class _PrimitiveBatchersProxy:
    """`primitive_batchers` 的模型：兼容代理，写穿到 `fancy_primitive_batchers`。"""

    def __init__(self, real):
        self._real = real

    def __setitem__(self, prim, batcher):
        self._real[prim] = batcher

    def __getitem__(self, prim):
        return self._real[prim]

    def __delitem__(self, prim):
        del self._real[prim]

    def __contains__(self, prim):
        return prim in self._real

    def get(self, prim, default=None):
        return self._real.get(prim, default)

    def keys(self):
        return self._real.keys()


class _AxisPrimitiveBatchersProxy(_PrimitiveBatchersProxy):
    """`axis_primitive_batchers` 的模型：老签名（带 `axis_data`）的代理。"""


primitive_batchers = _PrimitiveBatchersProxy(fancy_primitive_batchers)
axis_primitive_batchers = _AxisPrimitiveBatchersProxy(fancy_primitive_batchers)


def defprim(name, eager, infer, jvp=None, vjp=None, batcher=None):
    p = Primitive(name, eager, infer, jvp, vjp)
    PRIMITIVES[name] = p
    if batcher is not None:
        fancy_primitive_batchers[name] = batcher
    return p


def bind(prim, *args, **params):
    """分派：有 tracer 就交给**最顶层**的 trace；没有 tracer 就落到**最底层**的 trace。

    ★ 后一半是 vmap 能工作的关键：batching rule 是在**低一层**的语言里写的，
      它拿到的操作数常常是「没有 tracer 的具体值」；这些值必须交给**第 0 层**的
      trace 去处理（真实 JAX 的 `find_top_trace` 在没有 tracer 时也是回落到
      `EvalTrace` 这个 bottom trace）。**直接跑 numpy 会让值的类型不一致。**
    """
    tracers = [a for a in args if isinstance(a, Tracer)]
    if tracers:
        top = max(t.trace.level for t in tracers)
        return _TRACE_STACK[top].process_primitive(prim, args, params)
    if _TRACE_STACK:
        return _TRACE_STACK[0].process_primitive(prim, args, params)
    out = prim.eager(*args, **params)
    if isinstance(out, (tuple, list)):
        return tuple(Array(o) for o in out)
    return Array(out)


# ---------------------------------------------------------------------------
# 五、Tracer 与两层可组合的 trace：Jaxpr（jit） / Batch（vmap）
# ---------------------------------------------------------------------------

_TRACE_STACK = []


def current_level():
    return len(_TRACE_STACK) - 1 if _TRACE_STACK else None


class Tracer:
    __slots__ = ("trace", "aval")

    def __init__(self, trace, aval):
        self.trace = trace
        self.aval = aval

    # -- 算子重载：一律走 bind，落到当前顶层 trace --------------------------
    def __add__(self, o):
        return bind(PRIMITIVES["add"], self, o)

    def __radd__(self, o):
        return bind(PRIMITIVES["add"], o, self)

    def __sub__(self, o):
        return bind(PRIMITIVES["sub"], self, o)

    def __rsub__(self, o):
        return bind(PRIMITIVES["sub"], o, self)

    def __mul__(self, o):
        return bind(PRIMITIVES["mul"], self, o)

    def __rmul__(self, o):
        return bind(PRIMITIVES["mul"], o, self)

    def __truediv__(self, o):
        return bind(PRIMITIVES["div"], self, o)

    def __rtruediv__(self, o):
        return bind(PRIMITIVES["div"], o, self)

    def __neg__(self):
        return bind(PRIMITIVES["neg"], self)

    def __getitem__(self, idx):
        return bind(PRIMITIVES["static_index"], self, idx=idx)

    def __gt__(self, o):
        return bind(PRIMITIVES["gt"], self, o)

    def __lt__(self, o):
        return bind(PRIMITIVES["lt"], self, o)

    def __len__(self):
        if not self.aval.shape:
            raise TracerBoolConversionError(
                "mini: 对标量取 len() 在 JAX 里也是错的（本句是本模型的措辞）")
        return self.aval.shape[0]

    def __bool__(self):
        """★ **值相关的 Python 控制流**在这里断掉。

        追踪期拿到的是**符号值**，它没有「真值」。官方给的解法就是 `lax.cond`
        ——它的谓词**可以**是 traced 值（见 specs.LAX_COND_VERBATIM_DEV）。
        """
        raise TracerBoolConversionError(
            "mini: 不能把 traced 值当 Python 布尔用（本句是本模型的措辞，"
            "不是 JAX 的原文；真实 JAX 会抛 TracerBoolConversionError）")


class TracerBoolConversionError(Exception):
    """**值相关的 Python 控制流**的下场。

    ⚠ 诚实标注：类名沿用 JAX 的 `TracerBoolConversionError`（本项目的计划 §0.7.D
      转述为「Attempted boolean conversion of traced array…」），但**本仓库没有取到
      该报错的逐字一手来源**（见 specs 文件头的 UNSOURCED 段），所以：
        · 本模型的**措辞**是自写的，**不得当成 JAX 的原文引用**；
        · 能坐实的只有一件事——**对 traced 值取 bool 会失败**，而 `lax.cond`
          的谓词**可以**是 traced 值（后者有官方逐字）。
    """


class JaxprTracer(Tracer):
    """jit 追踪期的符号值：它的 `val` 是 jaxpr 里的一个变量或常量。"""

    __slots__ = ("val",)

    def __init__(self, trace, val, aval):
        Tracer.__init__(self, trace, aval)
        self.val = val

    def __repr__(self):
        return "<JaxprTracer %s %s>" % (self.val, self.aval)


class BatchTracer(Tracer):
    """vmap 追踪期的符号值：`val` 是**低一层**的值，`batch_dim` 是被映射的轴。

    ★ 机制要点：**batch 的大小不在这张图里**——batcher 是照着「有一个轴要被映射」
      这件事写的一般性规则，它不需要知道 B 是几。这正是「图不随 batch 变大」的原因。
    """

    __slots__ = ("val", "batch_dim")

    def __init__(self, trace, val, batch_dim, aval):
        Tracer.__init__(self, trace, aval)
        self.val = val
        self.batch_dim = batch_dim

    def __repr__(self):
        return "<BatchTracer dim=%s %s>" % (self.batch_dim, self.aval)


class MainTrace:
    __slots__ = ("level", "kind")

    def __init__(self, level, kind):
        self.level = level
        self.kind = kind

    def process_primitive(self, prim, args, params):
        raise NotImplementedError


class EvalTrace(MainTrace):
    """第 0 层：没有 jaxpr，直接 numpy 算（`vmap` 单独用时的底座）。"""

    def __init__(self, level):
        MainTrace.__init__(self, level, "eval")

    def process_primitive(self, prim, args, params):
        vals = [_unwrap(a) for a in args]
        out_aval = prim.infer(*[_aval_of(v) for v in vals], **params)
        return Array(prim.eager(*vals, **params), dtype=out_aval.dtype)


def _unwrap(a):
    if isinstance(a, BatchTracer):
        return _unwrap(a.val)
    if isinstance(a, JaxprTracer):
        return _unwrap(a.val)
    return a


def _np(v):
    if isinstance(v, Array):
        return v.data
    if isinstance(v, (BatchTracer, JaxprTracer)):
        return _np(_unwrap(v))
    if isinstance(v, Literal):
        return v.val
    if isinstance(v, Var):
        raise RuntimeError("mini: 变量 %s 还没有被求值" % v)
    return np.asarray(v)


def _aval_of(v):
    if isinstance(v, (Literal, Var)):
        return v.aval
    if isinstance(v, (Array, Tracer)):
        return v.aval
    return abstractify(v)


class JaxprTrace(MainTrace):
    """jit 追踪期：往一张 jaxpr 里记方程。"""

    def __init__(self, level, jaxpr, name="jit"):
        MainTrace.__init__(self, level, "jaxpr")
        self.jaxpr = jaxpr
        self.name = name

    def process_primitive(self, prim, args, params):
        invals = []
        for a in args:
            if isinstance(a, JaxprTracer) and a.trace is self:
                invals.append(a.val)
            else:
                invals.append(self._const(a))
        out_aval = prim.infer(*[_aval_of(v) for v in invals], **params)
        outvar = self.jaxpr.new_var(out_aval)
        self.jaxpr.eqns.append(Eqn(prim, invals, [outvar], params))
        return JaxprTracer(self, outvar, out_aval)

    def _const(self, a):
        """把非本层的值变成 jaxpr 里的**常量**。

        ★ 注意区分两件事（本主题的一条要害）：
          · 函数体里写的字面量（`x + 1.0` 里的 `1.0`）→ 走这里 → **Literal**
          · 从外面**传进来的 Python 标量实参**（`f(x, 3)` 里的 `3`）→ **不是**常量，
            它有自己的 invar（因为它是个被 trace 的实参）。
          在 TF 里这两者都会变成图上的字面量；在 JAX 里**只有前者是**。
        """
        v = _unwrap(a)
        if isinstance(v, Var):
            raise ValueError("mini: 不能把低层变量直接嵌进 jaxpr")
        if isinstance(v, (bool, int, float)):
            # ★ Python 标量是**弱类型**的：它按 JAX 的默认 dtype 走（int→int32 /
            #   float→float32），**不**变成 numpy 的 float64。丢了这一条，
            #   `x * 3.0`（x 是 float32）就会算成 float64——与真实 JAX 不符。
            av = abstractify(v)
            return Literal(np.asarray(v, dtype=_numpy_dtype(av.dtype)), av)
        arr = _np(v)
        return Literal(arr, abstractify(arr))

    def process_control_flow(self, prim, subjaxprs, invals, params):
        out_avals = params.pop("_out_avals")
        outvars = [self.jaxpr.new_var(av) for av in out_avals]
        self.jaxpr.eqns.append(
            Eqn(prim, invals, outvars, params, subjaxprs=subjaxprs))
        outs = [JaxprTracer(self, v, av) for v, av in zip(outvars, out_avals)]
        return outs[0] if len(outs) == 1 else tuple(outs)


class BatchTrace(MainTrace):
    """vmap 追踪期。

    ★ 机制：**本层不生成方程**——它把每个 primitive 交给该 primitive 注册的
      **batching rule（batcher）**，batcher 在**低一层**调用同一个 primitive。
      所以「一层 vmap」在图上**不多出任何节点**，只是把被映射的轴推了下去。
      这就是官方那句「the program stays the same size no matter the batch」的机制面
      （见 specs.VMAP_SIZE_VERBATIM）。
    """

    def __init__(self, level, parent, batch_size, axis_name=None):
        MainTrace.__init__(self, level, "batch")
        self.parent = parent
        self.batch_size = batch_size
        self.axis_name = axis_name

    def process_primitive(self, prim, args, params):
        vals, dims, avals = [], [], []
        for a in args:
            if isinstance(a, BatchTracer) and a.trace is self:
                vals.append(a.val)
                dims.append(a.batch_dim)
                # ★ 交给 batcher 的是**去掉被映射轴**之后的抽象值——
                #   因为 batching rule 是在「低一层」的语言里写的。
                avals.append(_unbatched_aval(a.val, a.batch_dim))
            else:
                vals.append(a)
                dims.append(None)
                avals.append(_aval_of(a) if isinstance(a, (Tracer, Array))
                             else abstractify(a))
        rule = fancy_primitive_batchers.get(prim.name)
        if rule is None:
            raise NotImplementedError(
                "Batching rule for '%s' not implemented" % prim.name)
        out_val, out_dim = rule(vals, dims, params, self.batch_size, avals)
        return BatchTracer(self, out_val, out_dim, _aval_of(out_val))


# ---------------------------------------------------------------------------
# 六、primitive 的实现：eager / infer / jvp / vjp / batcher
# ---------------------------------------------------------------------------

def _binop_infer(name):
    def infer(a, b):
        try:
            shape = np.broadcast_shapes(tuple(a.shape), tuple(b.shape))
        except ValueError:
            raise TypeError("mini: %s 的形状对不上：%s 与 %s" % (name, a.shape, b.shape))
        return ShapedArray(shape, _promote_dtype(a, b))
    return infer


def _binop_batcher(name):
    """逐元素二元算子的 batching rule。

    ★ 这里有一个容易写错、但正是机制所在的细节：**`dim is None` 的操作数原样传下去**，
      不给它加 batch 轴——因为真实的 JAX batching rule 也是这么做的：它靠 primitive
      自己的 broadcast 语义把「没被映射」的那个操作数铺开。
      （如果给它加一根长度为 B 的**前导**轴，`(B,3) * (B,)` 反而会撞上形状冲突。）
    """
    def rule(vals, dims, params, batch_size, avals):
        a, b = vals
        da, db = dims
        return bind(PRIMITIVES[name], a, b), (da if da is not None else
                                              (db if db is not None else 0))
    return rule


def _broadcast_to_batch(v, batch_size):
    """把一个「没有被映射」的操作数在前面补一维（`dot` 这类不能自动 broadcast 的算子用）。"""
    arr = _np(v)
    operand = v if isinstance(v, (Tracer, Array)) else Array(arr)
    return bind(PRIMITIVES["broadcast_in_dim"], operand,
                shape=(batch_size,) + tuple(arr.shape),
                broadcast_dimensions=tuple(range(1, 1 + arr.ndim)))


def _mk_binop(name, npfn, jvp, vjp):
    defprim(name, lambda a, b: npfn(_np(a), _np(b)), _binop_infer(name),
            jvp=jvp, vjp=vjp, batcher=_binop_batcher(name))


# jvp/vjp 的签名统一为：
#   jvp(p_args, t_args, params) -> tuple[tangent_out]
#   vjp(p_args, ct_outs, params)  -> tuple[ct_in]
_mk_binop("add", np.add,
          lambda pa, ta, p: (ta[0] + ta[1],),
          lambda pa, ct, p: (ct[0], ct[0]))
_mk_binop("sub", np.subtract,
          lambda pa, ta, p: (ta[0] - ta[1],),
          lambda pa, ct, p: (ct[0], -ct[0]))
_mk_binop("mul", np.multiply,
          lambda pa, ta, p: (ta[0] * pa[1] + pa[0] * ta[1],),
          lambda pa, ct, p: (ct[0] * pa[1], ct[0] * pa[0]))
_mk_binop("div", np.divide,
          lambda pa, ta, p: ((ta[0] * pa[1] - pa[0] * ta[1]) / (pa[1] * pa[1]),),
          lambda pa, ct, p: (ct[0] / pa[1], -ct[0] * pa[0] / (pa[1] * pa[1])))
_mk_binop("maximum", np.maximum, None, None)


def _cmp_infer(a, b):
    return ShapedArray(np.broadcast_shapes(tuple(a.shape), tuple(b.shape)), "bool")


def _zero_tangent(pa, ta, p):
    """比较运算的导数恒为 0——但**仍要有规则**，否则 `grad` 会在这里断掉。"""
    return (np.zeros(np.broadcast_shapes(np.shape(pa[0]), np.shape(pa[1])),
                     dtype=bool),)


for _nm, _fn in (("gt", np.greater), ("lt", np.less)):
    defprim(_nm, (lambda fn: (lambda a, b: fn(_np(a), _np(b))))(_fn), _cmp_infer,
            jvp=_zero_tangent, vjp=lambda pa, ct, p: (np.zeros_like(pa[0]),
                                                      np.zeros_like(pa[1])),
            batcher=_binop_batcher(_nm))


def _mk_unop(name, npfn, jvp, vjp, infer=None):
    defprim(name, lambda a: npfn(_np(a)),
            infer or (lambda a: ShapedArray(a.shape, a.dtype)),
            jvp=jvp, vjp=vjp,
            batcher=lambda vals, dims, params, bs, avals: (
                bind(PRIMITIVES[name], vals[0]), dims[0]))


_mk_unop("neg", np.negative, lambda pa, ta, p: (-ta[0],),
         lambda pa, ct, p: (-ct[0],))
_mk_unop("exp", np.exp, lambda pa, ta, p: (np.exp(pa[0]) * ta[0],),
         lambda pa, ct, p: (ct[0] * np.exp(pa[0]),))
_mk_unop("log", np.log, lambda pa, ta, p: (ta[0] / pa[0],),
         lambda pa, ct, p: (ct[0] / pa[0],))
_mk_unop("sqrt", np.sqrt,
         lambda pa, ta, p: (ta[0] / (2.0 * np.sqrt(pa[0])),),
         lambda pa, ct, p: (ct[0] / (2.0 * np.sqrt(pa[0])),))
_mk_unop("abs", np.abs, None, None)
_mk_unop("tanh", np.tanh,
         lambda pa, ta, p: ((1.0 - np.tanh(pa[0]) ** 2) * ta[0],),
         lambda pa, ct, p: (ct[0] * (1.0 - np.tanh(pa[0]) ** 2),))


def _reduce_sum_infer(a, axes=None):
    if axes is None:
        return ShapedArray((), a.dtype)
    axes = tuple(axes)
    return ShapedArray(tuple(d for i, d in enumerate(a.shape) if i not in axes),
                       a.dtype)


def _reduce_sum_batcher(vals, dims, params, batch_size, avals):
    (a,), (d,), (av,) = vals, dims, avals
    if d is None:
        return bind(PRIMITIVES["reduce_sum"], a, **params), None
    # ★ batching rule 的一步：`axes=None`（归约掉所有轴）在 vmap 之下**不能照抄**——
    #   被映射的那根轴必须留下，所以要把「低一层」的全部轴显式列出来、再整体右移一位。
    axes = params.get("axes")
    if axes is None:
        axes = tuple(range(len(av.shape)))
    return bind(PRIMITIVES["reduce_sum"], a,
                axes=tuple(ax + 1 for ax in axes)), d


defprim("reduce_sum",
        lambda a, axes=None: np.sum(_np(a), axis=(None if axes is None else tuple(axes))),
        _reduce_sum_infer,
        jvp=lambda pa, ta, p: (np.sum(ta[0], axis=p.get("axes")),),
        vjp=lambda pa, ct, p: (np.broadcast_to(
            np.asarray(ct[0]),
            _avalanche_shape(pa[0], p.get("axes"))).copy(),),
        batcher=_reduce_sum_batcher)


def _avalanche_shape(x, axes):
    """reverse 时把标量广播回原形状（本模型专供 reduce_sum 的 transpose）。"""
    arr = np.asarray(x)
    if axes is None:
        return arr.shape
    return arr.shape


def _bcast_infer(x, shape, broadcast_dimensions):
    return ShapedArray(shape, x.dtype)


def _bcast_eager(x, shape, broadcast_dimensions):
    a = _np(x)
    if a.shape == tuple(shape):
        return a
    return np.broadcast_to(a, tuple(shape)).copy()


def _bcast_batcher(vals, dims, params, batch_size, avals):
    (a,), (d,) = vals, dims
    if d is None:
        base = bind(PRIMITIVES["broadcast_in_dim"], a, shape=tuple(params["shape"]),
                    broadcast_dimensions=tuple(params["broadcast_dimensions"]))
        return bind(PRIMITIVES["broadcast_in_dim"], base,
                    shape=(batch_size,) + tuple(params["shape"]),
                    broadcast_dimensions=(0,) + tuple(
                        x + 1 for x in params["broadcast_dimensions"])), 0
    return bind(PRIMITIVES["broadcast_in_dim"], a,
                shape=(batch_size,) + tuple(params["shape"]),
                broadcast_dimensions=(0,) + tuple(
                    x + 1 for x in params["broadcast_dimensions"])), 0


defprim("broadcast_in_dim", _bcast_eager, _bcast_infer, batcher=_bcast_batcher)


def _dot_infer(a, b):
    if len(a.shape) == 2 and len(b.shape) == 2:
        return ShapedArray((a.shape[0], b.shape[1]), _promote_dtype(a, b))
    if len(a.shape) == 2 and len(b.shape) == 1:
        return ShapedArray((a.shape[0],), _promote_dtype(a, b))
    raise TypeError("mini: dot 只支持 2D×2D / 2D×1D（本模型的限制）")


def _dot_batcher(vals, dims, params, batch_size, avals):
    a, b = vals
    da, db = dims
    if da is None and db is None:
        raise ValueError("mini: dot 的两个操作数都没有被映射（vmap 之下不该发生）")
    return bind(PRIMITIVES["dot"], a, b), (da if da is not None else db)


defprim("dot", lambda a, b: np.matmul(_np(a), _np(b)), _dot_infer,
        batcher=_dot_batcher)


def _reshape_infer(x, new_sizes):
    return ShapedArray(tuple(new_sizes), x.dtype)


def _reshape_batcher(vals, dims, params, batch_size, avals):
    (a,), (d,) = vals, dims
    ns = tuple(params["new_sizes"])
    if d is None:
        return bind(PRIMITIVES["reshape"], a, new_sizes=ns), None
    return bind(PRIMITIVES["reshape"], a, new_sizes=(batch_size,) + ns), 0


defprim("reshape", lambda x, new_sizes: np.reshape(_np(x), tuple(new_sizes)),
        _reshape_infer,
        jvp=lambda pa, ta, p: (np.reshape(ta[0], tuple(p["new_sizes"])),),
        vjp=lambda pa, ct, p: (np.reshape(ct[0], np.asarray(pa[0]).shape),),
        batcher=_reshape_batcher)


def _static_index_infer(x, idx):
    if not isinstance(idx, int):
        raise TypeError("mini: static_index 只支持整数下标（本模型）")
    return ShapedArray(x.shape[1:], x.dtype)


def _static_index_batcher(vals, dims, params, batch_size, avals):
    (a,), (d,) = vals, dims
    idx = params["idx"]
    if d is None:
        return bind(PRIMITIVES["static_index"], a, idx=idx), None
    # ★ 关键一步：被映射轴在 0 位，静态下标要先 +1 才落到数据轴上
    return bind(PRIMITIVES["static_index"], a, idx=idx + 1), d


defprim("static_index", lambda x, idx: np.asarray(_np(x)[idx]),
        _static_index_infer, batcher=_static_index_batcher)


def _convert_infer(x, new_dtype):
    return ShapedArray(x.shape, new_dtype)


defprim("convert_element_type",
        lambda x, new_dtype: _np(x).astype(_numpy_dtype(new_dtype)),
        _convert_infer,
        jvp=lambda pa, ta, p: (np.asarray(ta[0]).astype(
            _numpy_dtype(p["new_dtype"])),),
        batcher=lambda vals, dims, params, bs, avals: (
            bind(PRIMITIVES["convert_element_type"], vals[0],
                 new_dtype=params["new_dtype"]), dims[0]))


def _select_n_infer(pred, a, b):
    return ShapedArray(a.shape, _promote_dtype(a, b))


def _select_n_batcher(vals, dims, params, batch_size, avals):
    p, a, b = vals
    dp, da, db = dims
    out_dim = next(d for d in (dp, da, db) if d is not None)
    return bind(PRIMITIVES["select_n"], p, a, b), out_dim


defprim("select_n", lambda p, a, b: np.where(np.asarray(_np(p)).astype(bool),
                                             _np(a), _np(b)),
        _select_n_infer, batcher=_select_n_batcher)


def _stack_infer(*xs):
    if not xs:
        raise TypeError("mini: stack 至少要有一个操作数")
    return ShapedArray((len(xs),) + tuple(xs[0].shape), xs[0].dtype)


def _stack_eager(*xs):
    return np.stack([_np(x) for x in xs], axis=0)


# ★ `stack` **故意没有 batching rule**——这正是官方那条报错的来源：
#   `raise NotImplementedError(f"Batching rule for '{p}' not implemented")`
#   （jax/_src/interpreters/batching.py:275，见 specs.BATCHING_RULE_MISSING）。
#   在程序里用它来演示「primitive 缺 batcher 时 vmap 会怎么失败」。
defprim("stack", _stack_eager, _stack_infer)


# ---------------------------------------------------------------------------
# 七、命名空间 `numpy`：一套够用的算子
# ---------------------------------------------------------------------------

class _NumpyNS:
    """`jax.numpy` 的迷你对应物。"""

    def add(self, a, b):
        return bind(PRIMITIVES["add"], a, b)

    def sub(self, a, b):
        return bind(PRIMITIVES["sub"], a, b)

    def mul(self, a, b):
        return bind(PRIMITIVES["mul"], a, b)

    def div(self, a, b):
        return bind(PRIMITIVES["div"], a, b)

    def neg(self, a):
        return bind(PRIMITIVES["neg"], a)

    def exp(self, a):
        return bind(PRIMITIVES["exp"], a)

    def log(self, a):
        return bind(PRIMITIVES["log"], a)

    def sqrt(self, a):
        return bind(PRIMITIVES["sqrt"], a)

    def tanh(self, a):
        return bind(PRIMITIVES["tanh"], a)

    def abs(self, a):
        return bind(PRIMITIVES["abs"], a)

    def maximum(self, a, b):
        return bind(PRIMITIVES["maximum"], a, b)

    def greater(self, a, b):
        return bind(PRIMITIVES["convert_element_type"],
                    bind(PRIMITIVES["sub"], a, b), new_dtype="bool")

    def reduce_sum(self, a, axes=None):
        if axes is None:
            return bind(PRIMITIVES["reduce_sum"], a)
        return bind(PRIMITIVES["reduce_sum"], a, axes=tuple(axes))

    def dot(self, a, b):
        return bind(PRIMITIVES["dot"], a, b)

    def matmul(self, a, b):
        return bind(PRIMITIVES["dot"], a, b)

    def reshape(self, a, new_sizes):
        return bind(PRIMITIVES["reshape"], a, new_sizes=tuple(new_sizes))

    def where(self, pred, a, b):
        return bind(PRIMITIVES["select_n"], pred, a, b)

    def broadcast_in_dim(self, a, shape, broadcast_dimensions):
        return bind(PRIMITIVES["broadcast_in_dim"], a, shape=tuple(shape),
                    broadcast_dimensions=tuple(broadcast_dimensions))

    def convert(self, a, dtype):
        return bind(PRIMITIVES["convert_element_type"], a, new_dtype=dtype)

    def array(self, v, dtype=None, weak_type=False):
        return Array(v, dtype=dtype, weak_type=weak_type)

    def zeros(self, shape, dtype="float32"):
        return Array(np.zeros(shape, dtype=_numpy_dtype(dtype)), weak_type=True)

    def ones(self, shape, dtype="float32"):
        return Array(np.ones(shape, dtype=_numpy_dtype(dtype)), weak_type=True)

    def full(self, shape, fill_value, dtype="float32"):
        return Array(np.full(shape, fill_value, dtype=_numpy_dtype(dtype)))

    def arange(self, n, dtype="int32"):
        return Array(np.arange(n, dtype=_numpy_dtype(dtype)), weak_type=True)

    def stack(self, xs, axis=0):
        return bind(PRIMITIVES["stack"], *list(xs))

    def add_all(self, xs):
        acc = xs[0]
        for x in xs[1:]:
            acc = bind(PRIMITIVES["add"], acc, x)
        return acc


numpy = _NumpyNS()


# ---------------------------------------------------------------------------
# 八、jit：cache key、static 实参、donate、以及诊断
# ---------------------------------------------------------------------------

CONFIG = {
    "jax_explain_cache_misses": False,
    "jax_log_compiles": False,
    "jax_dump_ir_to": None,
}

# 全局的追踪事件日志（**本模型的诊断设施**；真实 JAX 用 `jax_log_compiles` 等）
TRACE_LOG = []
_LIVE_JITS = []


class _ConfigNS:
    def update(self, name, value):
        if name not in CONFIG:
            raise ValueError("mini: 本模型不认识配置项 %r" % (name,))
        CONFIG[name] = value

    def __getattr__(self, name):
        if name in CONFIG:
            return CONFIG[name]
        raise AttributeError(name)


config = _ConfigNS()


def _ordered_args(static_argnums, static_argnames, args, kwargs):
    """实参的统一顺序：位置实参（按序）+ 关键字实参（按名排序）。

    返回 `[(pos_or_name, value, is_static)]`。
    """
    items = [(i, a, i in static_argnums) for i, a in enumerate(args)]
    items += [(k, kwargs[k], k in static_argnames) for k in sorted(kwargs)]
    return items


class JitWrapped:
    """`jax.jit(fun)` 的返回物。

    ★ 真实 JAX 里它的类型就叫 **`pjit.JitWrapped`**——因为 `pjit` 已经被并进 `jax.jit`
      （`jax/_src/api.py` 里 `def jit(...) -> pjit.JitWrapped`，见 specs.JIT_RETURN_TYPE）。

    ★ **cache key 的三项**（官方逐字见 specs.CACHE_KEY_VERBATIM）：
      ① 参数的 **JAX types**（形状 + dtype，`abstractify` 的结果）；
      ② **静态参数的值**（只有显式 `static_argnums` / `static_argnames` 点名了才有）；
      ③ **函数对象的 identity**。
    第 ③ 项在本模型里落在「每个 `JitWrapped` 各有一张缓存」这件事上：
    两次 `jit()` 即使包的是**内容相同**的函数对象，也是两张缓存、各追一次。
    """

    def __init__(self, fun, static_argnums=(), static_argnames=(),
                 donate_argnums=(), donate_argnames=(), name=None):
        self.fun = fun
        self.name = name or getattr(fun, "__name__", "jit_fun")
        self.static_argnums = tuple(static_argnums)
        self.static_argnames = tuple(static_argnames)
        self.donate_argnums = tuple(donate_argnums)
        self.donate_argnames = tuple(donate_argnames)
        self._cache = OrderedDict()          # key -> (ClosedJaxpr, arg_specs)
        self._hits = 0
        self._misses = 0
        self.explain_lines = []
        _LIVE_JITS.append(weakref.ref(self))

    # -- cache key --------------------------------------------------------
    def _key(self, items):
        """① JAX types + ② 静态参数的值 + ③ 函数对象的 identity。

        ★ 这里是本主题最要紧的一处：**Python 标量默认按 ① 处理**——
          `3` 与 `4` 抽象出来是同一个 `ShapedArray((), 'int32', weak)`，
          于是**同一个 key**、**只追踪一次**。
          （要它进 ②，必须显式点名 `static_argnums` / `static_argnames`。）
        """
        key = [("fun", id(self.fun))]
        for name, v, is_static in items:
            if is_static:
                key.append(("static", name, _freeze(v)))
            else:
                key.append(("aval", name, abstractify(v).key()))
        return tuple(key)

    def _explain(self, key, items):
        """`jax_explain_cache_misses=True` 时打印 miss 原因。

        官方给的例子逐字：「TRACING CACHE MISS at ...: different input types: * at x,
        now f32[3] and before i32[3]」（见 specs.EXPLAIN_CACHE_MISSES_VERBATIM）。
        """
        if not CONFIG["jax_explain_cache_misses"]:
            return
        if not self._cache:
            line = ("TRACING CACHE MISS at %s: cache is empty" % self.name)
        else:
            prev = next(iter(self._cache))
            line = ("TRACING CACHE MISS at %s: different input types: %s, now %s "
                    "and before %s" % (self.name, "*", _sig_of(key), _sig_of(prev)))
        self.explain_lines.append(line)
        print("      [explain_cache_misses] %s" % line)

    # -- 追踪 -------------------------------------------------------------
    def _trace(self, items):
        jaxpr = Jaxpr(self.name)
        trace = JaxprTrace(len(_TRACE_STACK) if _TRACE_STACK else 0, jaxpr, self.name)
        arg_specs = []
        _TRACE_STACK.append(trace)
        try:
            tracers = []
            for name, v, is_static in items:
                aval = abstractify(v)
                # ★ `donate_argnums` **只覆盖位置实参**；关键字实参要用 `donate_argnames`
                #   （官方 docstring 的规则，见 specs.DONATE_ARGNUMS_DOC）。
                donated = (name in self.donate_argnums if isinstance(name, int)
                           else name in self.donate_argnames)
                arg_specs.append((name, aval, donated))
                if is_static:
                    tracers.append(v)               # 静态：原样传进去，会烧成常量
                else:
                    tracers.append(JaxprTracer(trace, jaxpr.new_invar(aval), aval))
            out = self.fun(*tracers)
        finally:
            _TRACE_STACK.pop()
        for o in (out if isinstance(out, tuple) else (out,)):
            jaxpr.outvars.append(o.val if isinstance(o, JaxprTracer)
                                 else Literal(_np(o), abstractify(o)))
        self._misses += 1
        TRACE_LOG.append({"fun": id(self.fun), "name": self.name,
                          "at": time.perf_counter()})
        if CONFIG["jax_log_compiles"]:
            print("      [jax_log_compiles] compiling %s" % self.name)
        if CONFIG["jax_dump_ir_to"]:
            _dump_ir(CONFIG["jax_dump_ir_to"], self.name, jaxpr)
        return jaxpr, arg_specs

    def __call__(self, *args, **kwargs):
        # ★ 在别的变换（jit / vmap / grad）之下，**`jit` 是透明的**：
        #   它不建自己的缓存，只把被包的 Python 函数交给外层的 trace。
        #   这就是为什么 `jax.grad(jax.jit(f))` 与 `jax.grad(f)` 得到同一张 jaxpr。
        if _TRACE_STACK:
            return self.fun(*args, **kwargs)
        items = _ordered_args(self.static_argnums, self.static_argnames, args, kwargs)
        key = self._key(items)
        if key in self._cache:
            self._hits += 1
        else:
            self._explain(key, items)
            jaxpr, arg_specs = self._trace(items)
            self._cache[key] = (ClosedJaxpr(jaxpr, []), arg_specs)
        cjaxpr, arg_specs = self._cache[key]
        values = [_np(v) for name, v, _st in items if not _is_static(self, name)]
        # donate：被捐出的 buffer **在调用之后**无效（官方逐字见 specs.DONATE_VERBATIM）
        by_name = {name: v for name, v, _st in items}
        for name, _aval, donated in arg_specs:
            if donated:
                _donate(by_name.get(name))
        out = _eval_closed(cjaxpr, values)
        return out[0] if len(out) == 1 else tuple(out)

    # -- 诊断 -------------------------------------------------------------
    def trace_count(self):
        return self._misses

    def cache_size(self):
        return len(self._cache)

    def keys(self):
        return list(self._cache)


def _is_static(wrapper, name):
    if isinstance(name, int):
        return name in wrapper.static_argnums
    return name in wrapper.static_argnames


def _sig_of(key):
    out = []
    for it in key:
        if it[0] == "aval":
            av = it[2]
            out.append("%s[%s]" % (av[1], ",".join(str(d) for d in av[0])))
        elif it[0] == "static":
            out.append("static(%r)" % (it[2],))
    return ", ".join(out)


def _freeze(v):
    if isinstance(v, Array):
        return ("Array", v.tolist())
    if isinstance(v, np.ndarray):
        return ("ndarray", v.tolist())
    return v


def _donate(a):
    """`donate_argnums`：被捐出的实参**之后就不能再用了**。

    官方逐字：「**Donated buffers are invalid afterwards.**」
    （`https://docs.jax.dev/en/latest/201/jit.html`，见 specs.DONATE_VERBATIM）
    ⚠ 本模型用一个「已失效」标记来体现这件事；真实 JAX 里那块显存会被下一个 kernel
      覆写，读到的是**未定义的值**，不会替你报错——**这里的报错是本模型加的保护**。
    """
    if isinstance(a, Array):
        a._invalid = True


def _dump_ir(directory, name, jaxpr):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + ".jaxpr")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(jaxpr) + "\n")
    return path


def jit(fun=None, static_argnums=(), static_argnames=(), donate_argnums=(),
        donate_argnames=()):
    """`jax.jit`。装饰器与直接调用两种用法都支持。

    ⚠ **`jax.jit` 没有 dynamic shape 开关**——形状多态在 `jax.export` 那一层，
      而且它**也不免除重编译**（见 `export` 类与 specs.EXPORT_SYMBOLIC_VERBATIM）。
    """
    if fun is None:
        return lambda f: JitWrapped(f, static_argnums, static_argnames,
                                    donate_argnums, donate_argnames)
    return JitWrapped(fun, static_argnums, static_argnames, donate_argnums,
                      donate_argnames)


def clear_caches():
    """`jax.clear_caches()`：清掉**所有** jit 缓存（于是下一次调用必然重追踪）。"""
    n = 0
    for ref in list(_LIVE_JITS):
        w = ref()
        if w is not None:
            n += len(w._cache)
            w._cache.clear()
    TRACE_LOG.clear()
    return n


def traces_of(fun):
    """**本模型的诊断入口**：某个函数对象被追踪了几次（不是 JAX 的 API）。"""
    return sum(1 for e in TRACE_LOG if e["fun"] == id(fun))


# ---------------------------------------------------------------------------
# 九、jaxpr 解释器（重放时按图执行，**不再跑 Python 函数体**）
# ---------------------------------------------------------------------------

def _as_list(v):
    """primitive 的 eager/jvp/vjp 结果统一成元组（`reduce_sum` 全归约会返回标量）。"""
    if isinstance(v, (tuple, list)):
        return tuple(np.asarray(x) for x in v)
    return (np.asarray(v),)


def _cast_to_aval(aval, arr):
    """按 jaxpr 的**显式类型**对齐结果 dtype。

    ★ 这一步是必需的，不是修饰：jaxpr 是「**explicitly typed**」的，而 numpy 的
      dtype 提升规则与 JAX 的**弱类型**规则不同——`float32 数组 * Python int`
      在 numpy 里会升到 float64，在 JAX 里仍是 float32。不按 aval 对齐就会算错类型。
    """
    a = np.asarray(arr)
    if aval.dtype not in _DTYPE_NAMES or a.dtype == _numpy_dtype(aval.dtype):
        return a
    if np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_:
        return a.astype(_numpy_dtype(aval.dtype))
    return a


def _cast_outs(eqn, outs):
    return [_cast_to_aval(_aval_of(v), o) for v, o in zip(eqn.outvars, outs)]


def _eval_eqn(eqn, env):
    return _cast_outs(eqn, _eval_eqn_raw(eqn, env))


def _eval_eqn_raw(eqn, env):
    prim = eqn.primitive
    args = [env[v] if isinstance(v, Var) else v.val for v in eqn.invars]
    params = dict(eqn.params)
    if prim.name == "cond":
        idx = 0 if bool(np.asarray(args[0])) else 1
        sub = eqn.subjaxprs[idx]
        senv = {}
        _eval_jaxpr(sub, args[1:], senv)
        return [senv[v] for v in sub.outvars]
    if prim.name == "while":
        return _eval_while(eqn, args)
    if prim.name == "scan":
        return _eval_scan(eqn, args)
    if prim.name == "psum":
        return [_psum_all_reduce(args[0])]
    return list(_as_list(prim.eager(*args, **params)))


def _eval_jaxpr(jaxpr, in_vals, env):
    for v, val in zip(jaxpr.invars, in_vals):
        env[v] = val
    for eqn in jaxpr.eqns:
        outs = _eval_eqn(eqn, env)
        for v, o in zip(eqn.outvars, outs):
            env[v] = o
    return env


def _eval_while(eqn, args):
    cond_sub, body_sub = eqn.subjaxprs
    state = list(args)
    guard = 0
    while True:
        cenv = {}
        _eval_jaxpr(cond_sub, state, cenv)
        if not bool(np.asarray(cenv[cond_sub.outvars[0]])):
            break
        benv = {}
        _eval_jaxpr(body_sub, state, benv)
        state = [benv[v] for v in body_sub.outvars]
        guard += 1
        if guard > 100000:
            raise RuntimeError("mini: while_loop 未收敛")
    return state


def _eval_scan(eqn, args):
    body = eqn.subjaxprs[0]
    n = int(eqn.params["length"])
    carry = args[0]
    xs = args[1:]
    ys = []
    for i in range(n):
        benv = {}
        in_vals = [carry] + [np.asarray(x)[i] for x in xs]
        _eval_jaxpr(body, in_vals, benv)
        outs = [benv[v] for v in body.outvars]
        carry = outs[0]
        ys.append(outs[1:])
    stacked = ([np.stack([y[j] for y in ys]) for j in range(len(ys[0]))]
               if ys and ys[0] else [])
    return [carry] + stacked


def _eval_closed(cjaxpr, values):
    env = {}
    for var, val in zip(cjaxpr.jaxpr.invars, values):
        env[var] = _np(val)
    for eqn in cjaxpr.jaxpr.eqns:
        outs = _eval_eqn(eqn, env)
        for v, o in zip(eqn.outvars, outs):
            env[v] = o
    return [Array(env[v] if isinstance(v, Var) else v.val)
            for v in cjaxpr.jaxpr.outvars]


def eval_jaxpr(jaxpr, values):
    return _eval_closed(ClosedJaxpr(jaxpr, []), values)


def make_jaxpr(fun, static_argnums=()):
    """`jax.make_jaxpr(fun)`：返回 **`ClosedJaxpr`**（jaxpr + consts）。

    官方把它的返回值叫 `ClosedJaxpr`，并给了 `jaxpr ::= { lambda Var* ; Var+. let Eqn*
    in [ Expr+ ] }` 这条文法（见 specs.JAXPR_VERBATIM / specs.MAKE_JAXPR_NOTE）。
    本模型的 `make_jaxpr` 接受一批 `ShapedArray` 作为抽象实参。
    """
    def wrapper(*avals):
        jaxpr = Jaxpr(getattr(fun, "__name__", "fun"))
        trace = JaxprTrace(0, jaxpr, jaxpr.name)
        _TRACE_STACK.append(trace)
        try:
            tracers = []
            for i, av in enumerate(avals):
                if i in static_argnums:
                    raise NotImplementedError("mini: make_jaxpr 不支持 static_argnums")
                tracers.append(JaxprTracer(trace, jaxpr.new_invar(av), av))
            out = fun(*tracers)
        finally:
            _TRACE_STACK.pop()
        for o in (out if isinstance(out, tuple) else (out,)):
            jaxpr.outvars.append(o.val if isinstance(o, JaxprTracer)
                                 else Literal(_np(o), abstractify(o)))
        return ClosedJaxpr(jaxpr, [])
    return wrapper


# ---------------------------------------------------------------------------
# 十、vmap：**给每个 primitive 定义 batching rule**
# ---------------------------------------------------------------------------

def vmap(fun, in_axes=0, out_axes=0, axis_name=None):
    """`jax.vmap`。

    ★ 机制（官方术语）：vmap **给每个 primitive 定义 batching rule（batcher）**，
      把被映射的轴**推下去**（push the mapped axis down into primitive operations）。
      它**不是**把循环展开：官方逐字「the program **stays the same size no matter
      the batch**」，对比手写 Python 循环「… contains **a full copy of the body
      per batch element**: six `dot_general`s, not three, and growing linearly with
      the batch size」（见 specs.VMAP_SIZE_VERBATIM，**具体页面 URL 也在那里**）。

    这里一个重要实现事实：**batcher 拿不到 batch 大小**——它只被告知「有一个轴要映射」。
    所以图里不会有 B 的痕迹，方程数自然与 B 无关。
    """

    def wrapped(*args):
        created = not _TRACE_STACK
        if created:
            _TRACE_STACK.append(EvalTrace(0))
        try:
            level = len(_TRACE_STACK)
            axes = _expand_axes(in_axes, len(args))
            batch_size = None
            for a, ax in zip(args, axes):
                if ax is not None:
                    batch_size = a.aval.shape[ax]
                    break
            if batch_size is None:
                raise ValueError("mini: vmap 至少要有一个被映射的实参")
            bt = BatchTrace(level, _TRACE_STACK[-1], batch_size, axis_name)
            _TRACE_STACK.append(bt)
            try:
                in_vals = []
                for a, ax in zip(args, axes):
                    if ax is None:
                        in_vals.append(a)
                    else:
                        in_vals.append(BatchTracer(bt, a, 0, _unbatched_aval(a, ax)))
                out = fun(*in_vals)
            finally:
                _TRACE_STACK.pop()
            return _unbatch(out)
        finally:
            if created:
                _TRACE_STACK.pop()

    wrapped.__name__ = getattr(fun, "__name__", "vmapped")
    return wrapped


def _expand_axes(in_axes, n):
    if in_axes is None:
        return tuple(0 for _ in range(n))
    if isinstance(in_axes, int):
        return tuple(in_axes for _ in range(n))
    axes = tuple(in_axes)
    if len(axes) < n:
        raise ValueError("mini: in_axes 的个数少于实参个数")
    return axes


def _unbatched_aval(a, ax):
    """被映射轴拿走之后的 aval（batcher 看到的是「低一层」的抽象值）。"""
    av = a.aval if hasattr(a, "aval") else abstractify(a)
    shape = tuple(d for i, d in enumerate(av.shape) if i != ax)
    return ShapedArray(shape, av.dtype, av.weak_type)


def _unbatch(out):
    if isinstance(out, (tuple, list)):
        return tuple(_unbatch(o) for o in out)
    if isinstance(out, BatchTracer):
        return out.val
    return out


# ---------------------------------------------------------------------------
# 十一、微分：前向（jvp）与反向（vjp / grad）
#
# 本模型的微分直接作用在**具体值**上：先把函数 trace 成 jaxpr，再对每个方程查
# `primitive.jvp` / `primitive.vjp` 规则。**缺规则就抛错**——本主题要用的正是这一点：
# `while` 有前向规则、**故意没有反向规则**（官方对它的说明见 specs.WHILE_FORWARD_ONLY）。
# ---------------------------------------------------------------------------

def _jvp_while(eqn, args, targs):
    """`while` 的**前向**规则：把「值 + 切向」一起迭代下去（这是它有、反向没有的那条规则）。"""
    cond_sub, body_sub = eqn.subjaxprs
    state, tstate = list(args), list(targs)
    guard = 0
    while True:
        cenv = {}
        _eval_jaxpr(cond_sub, state, cenv)
        if not bool(np.asarray(cenv[cond_sub.outvars[0]])):
            break
        benv, bt = {}, {}
        _eval_jaxpr_with_tangents(body_sub, state, tstate, benv, bt)
        state = [benv[v] for v in body_sub.outvars]
        tstate = [bt[v] for v in body_sub.outvars]
        guard += 1
        if guard > 100000:
            raise RuntimeError("mini: while_loop 未收敛")
    return state, tstate


def _eval_jaxpr_with_tangents(jaxpr, in_vals, in_ts, env, tenv):
    for v, val, t in zip(jaxpr.invars, in_vals, in_ts):
        env[v] = np.asarray(val)
        tenv[v] = np.asarray(t)
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "while":
            args = [env[v] if isinstance(v, Var) else v.val for v in eqn.invars]
            targs = [tenv.get(v, np.zeros_like(np.asarray(env[v])))
                     if isinstance(v, Var) else np.zeros_like(np.asarray(v.val))
                     for v in eqn.invars]
            p_outs, t_outs = _jvp_while(eqn, args, targs)
            for v, p, t in zip(eqn.outvars, p_outs, t_outs):
                env[v] = p
                tenv[v] = t
            continue
        args = [env[v] if isinstance(v, Var) else v.val for v in eqn.invars]
        targs = [tenv.get(v, np.zeros_like(np.asarray(env[v])))
                 if isinstance(v, Var) else np.zeros_like(np.asarray(v.val))
                 for v in eqn.invars]
        if eqn.primitive.jvp is None:
            raise NotImplementedError(
                "mini: primitive %r 没有注册 jvp 规则" % eqn.primitive.name)
        p_out = _as_list(eqn.primitive.eager(*args, **eqn.params))
        t_out = _as_list(eqn.primitive.jvp(args, targs, eqn.params))
        for v, p, t in zip(eqn.outvars, p_out, t_out):
            env[v] = p
            tenv[v] = t


def jvp(fun, primals, tangents):
    """前向模式微分：每个值带一个切向分量，与值一起走完整张图。"""
    cjaxpr = make_jaxpr(fun)(*[abstractify(p) for p in primals])
    env = {v: _np(p) for v, p in zip(cjaxpr.jaxpr.invars, primals)}
    tenv = {v: _np(t) for v, t in zip(cjaxpr.jaxpr.invars, tangents)}
    for eqn in cjaxpr.jaxpr.eqns:
        args = [env[v] if isinstance(v, Var) else v.val for v in eqn.invars]
        targs = [tenv.get(v, np.zeros_like(np.asarray(env[v])))
                 if isinstance(v, Var) else np.zeros_like(np.asarray(v.val))
                 for v in eqn.invars]
        if eqn.primitive.name == "while":
            p_outs, t_outs = _jvp_while(eqn, args, targs)
            for v, p, t in zip(eqn.outvars, p_outs, t_outs):
                env[v] = p
                tenv[v] = t
            continue
        if eqn.primitive.jvp is None:
            raise NotImplementedError(
                "mini: primitive %r 没有注册 jvp 规则" % eqn.primitive.name)
        p_out = _as_list(eqn.primitive.eager(*args, **eqn.params))
        t_out = _as_list(eqn.primitive.jvp(args, targs, eqn.params))
        for v, p, t in zip(eqn.outvars, p_out, t_out):
            env[v] = p
            tenv[v] = t
    outs = [env[v] if isinstance(v, Var) else v.val for v in cjaxpr.jaxpr.outvars]
    touts = [tenv.get(v, np.zeros_like(np.asarray(env[v])))
             if isinstance(v, Var) else np.zeros_like(np.asarray(v.val))
             for v in cjaxpr.jaxpr.outvars]
    pack = lambda xs: (Array(xs[0]) if len(xs) == 1
                       else tuple(Array(x) for x in xs))
    return pack(outs), pack(touts)


_PULLBACK_REVERSE_OK = {"add", "sub", "mul", "div", "neg", "exp", "log", "sqrt",
                        "tanh", "abs", "reduce_sum", "reshape",
                        "convert_element_type", "static_index", "dot", "select_n",
                        "broadcast_in_dim"}


def vjp(fun, primals):
    """反向模式微分：先正向记录每个中间值，再从输出往回扫一遍（transpose 规则）。

    ★ `while` **没有反向规则**——走到它就抛 `NotImplementedError`。
      官方对这条的说明见 specs.WHILE_FORWARD_ONLY。
    """
    cjaxpr = make_jaxpr(fun)(*[abstractify(p) for p in primals])
    env = {v: _np(p) for v, p in zip(cjaxpr.jaxpr.invars, primals)}
    for eqn in cjaxpr.jaxpr.eqns:
        args = [env[v] if isinstance(v, Var) else v.val for v in eqn.invars]
        outs = _eval_eqn(eqn, env)
        for v, o in zip(eqn.outvars, outs):
            env[v] = o
    outs = [env[v] if isinstance(v, Var) else v.val for v in cjaxpr.jaxpr.outvars]

    def pullback(ct):
        cts = {}
        ct_list = ct if isinstance(ct, (tuple, list)) else (ct,)
        for i, v in enumerate(cjaxpr.jaxpr.outvars):
            cts[v] = np.asarray(ct_list[i])
        for eqn in reversed(cjaxpr.jaxpr.eqns):
            contribs = _pullback_eqn(eqn, env, cts)
            for v, c in zip(eqn.invars, contribs):
                if isinstance(v, Var):
                    cts[v] = cts.get(v, np.zeros_like(np.asarray(c))) + np.asarray(c)
        return [cts.get(v, np.zeros_like(np.asarray(env[v])))
                for v in cjaxpr.jaxpr.invars]

    return tuple(Array(x) for x in outs), pullback


def _pullback_eqn(eqn, env, cts):
    """一条方程的反向规则分派。

    ★ **`while` 是唯一一条没有反向规则的**——走到它就抛 `NotImplementedError`。
      这正是官方那条限制的落点：`while_loop` **只支持前向模式微分**
      （反传要静态边界，官方指的路是用 `scan` / `fori_loop`，见 specs.WHILE_FORWARD_ONLY）。
      `cond` 与 `scan` 在本模型里**都有**反向规则（`scan` 是逐步回扫）。
    """
    name = eqn.primitive.name
    args = [env[v] if isinstance(v, Var) else v.val for v in eqn.invars]
    ct_outs = [cts.get(v, np.zeros_like(np.asarray(env[v]))) for v in eqn.outvars]

    if name == "while":
        raise NotImplementedError(
            "Reverse-mode differentiation rule for 'while' not implemented")

    if name == "cond":
        idx = 0 if bool(np.asarray(args[0])) else 1
        sub = eqn.subjaxprs[idx]
        grads = _vjp_jaxpr(sub, args[1:], ct_outs)
        return [np.zeros_like(np.asarray(args[0]))] + list(grads)

    if name == "scan":
        body = eqn.subjaxprs[0]
        n = int(eqn.params["length"])
        xs = args[1:]
        carries, c = [], np.asarray(args[0])
        for i in range(n):
            carries.append(c)
            benv = {}
            _eval_jaxpr(body, [c] + [np.asarray(x)[i] for x in xs], benv)
            c = np.asarray(benv[body.outvars[0]])
        ct_carry = np.asarray(ct_outs[0])
        ct_xs = [np.zeros_like(np.asarray(x)) for x in xs]
        for i in range(n - 1, -1, -1):
            ct_y = [np.asarray(ct_outs[1 + j])[i] for j in range(len(ct_outs) - 1)]
            grads = _vjp_jaxpr(body, [carries[i]] + [np.asarray(x)[i] for x in xs],
                               [ct_carry] + ct_y)
            ct_carry = grads[0]
            for j, gg in enumerate(grads[1:]):
                ct_xs[j][i] = gg
        return [ct_carry] + ct_xs

    if eqn.primitive.vjp is None:
        raise NotImplementedError(
            "Reverse-mode differentiation rule for '%s' not implemented" % name)
    return list(_as_list(eqn.primitive.vjp(args, ct_outs, eqn.params)))


def _vjp_jaxpr(jaxpr, in_vals, ct_outs):
    """对一张（不含 while 的）jaxpr 做「正向记录 + 反向回扫」，返回各入参的 cotangent。

    这是 `scan` 反向规则的基础：**每一步单独做一次**，carry 的 cotangent 往下一步传。
    """
    env = {}
    for v, val in zip(jaxpr.invars, in_vals):
        env[v] = np.asarray(val)
    for eqn in jaxpr.eqns:
        outs = _eval_eqn(eqn, env)
        for v, o in zip(eqn.outvars, outs):
            env[v] = o
    cts = {}
    for v, ct in zip(jaxpr.outvars, ct_outs):
        cts[v] = np.asarray(ct)
    for eqn in reversed(jaxpr.eqns):
        contribs = _pullback_eqn(eqn, env, cts)
        for v, c in zip(eqn.invars, contribs):
            if isinstance(v, Var):
                cts[v] = cts.get(v, np.zeros_like(np.asarray(c))) + np.asarray(c)
    return [cts.get(v, np.zeros_like(np.asarray(env[v]))) for v in jaxpr.invars]


def grad(fun, argnums=0):
    """`jax.grad`：vjp 的 `argnums` 版包装（只看标量输出的情形）。"""
    def wrapped(*primals):
        outs, pullback = vjp(fun, primals)
        ct = np.ones_like(np.asarray(outs[0]))
        grads = pullback(ct)
        if isinstance(argnums, int):
            return Array(grads[argnums])
        return tuple(Array(grads[i]) for i in argnums)
    return wrapped


def value_and_grad(fun, argnums=0):
    def wrapped(*primals):
        outs, pullback = vjp(fun, primals)
        ct = np.ones_like(np.asarray(outs[0]))
        grads = pullback(ct)
        g = grads[argnums] if isinstance(argnums, int) else tuple(
            grads[i] for i in argnums)
        return outs[0], (Array(g) if not isinstance(g, tuple)
                         else tuple(Array(x) for x in g))
    return wrapped


# ---------------------------------------------------------------------------
# 十二、控制流：lax.cond / while_loop / fori_loop / scan
# ---------------------------------------------------------------------------

def _trace_subjaxpr(fn, avals, name):
    """把分支/循环体 trace 成一张**子 jaxpr**。

    ⚠ 本模型的限制：子 jaxpr 里**只能引用它自己的入参**（不能闭包外层的 traced 值）——
      真实 JAX 会把外层值当常量 lift 进来。
    """
    jaxpr = Jaxpr(name)
    trace = JaxprTrace(len(_TRACE_STACK), jaxpr, name)
    _TRACE_STACK.append(trace)
    try:
        tracers = [JaxprTracer(trace, jaxpr.new_invar(av), av) for av in avals]
        out = fn(*tracers)
    finally:
        _TRACE_STACK.pop()
    for o in (out if isinstance(out, tuple) else (out,)):
        jaxpr.outvars.append(o.val if isinstance(o, JaxprTracer)
                             else Literal(_np(o), abstractify(o)))
    return jaxpr


def _placeholder(name):
    if name not in PRIMITIVES:
        defprim(name, None, lambda *a, **k: None)


for _cf in ("cond", "while", "scan", "psum"):
    _placeholder(_cf)


class _LaxNS:
    """`jax.lax` 的迷你对应物：结构化控制流四件套 + `psum`。"""

    def cond(self, pred, true_fun, false_fun, *operands):
        """`lax.cond`。

        ★ 关键性质（官方逐字）：「**the predicate here can be a traced value.
          The choice of branch happens on the device, at run time, inside the compiled
          program.**」（见 specs.LAX_COND_VERBATIM）

        于是它与 Python `if` 的区别是双重的：
          ① 谓词**可以是 traced 值**（Python `if` 对 traced 值直接抛错）；
          ② **两支都进图**，选哪支推迟到运行时。
        """
        if not isinstance(pred, Tracer):
            return true_fun(*operands) if bool(_np(pred)) else false_fun(*operands)
        op_avals = [o.aval for o in operands]
        tj = _trace_subjaxpr(lambda *a: true_fun(*a), op_avals, "cond_true")
        fj = _trace_subjaxpr(lambda *a: false_fun(*a), op_avals, "cond_false")
        if len(tj.outvars) != len(fj.outvars):
            raise TypeError("mini: cond 两支的返回值个数必须一致")
        for a, b in zip(tj.outvars, fj.outvars):
            if _aval_of(a).dtype != _aval_of(b).dtype:
                raise TypeError(
                    "mini: cond 两支的 dtype 必须一致，得到 %s 与 %s"
                    % (_aval_of(a).dtype, _aval_of(b).dtype))
        out_avals = [_aval_of(v) for v in tj.outvars]
        invals = [pred.val if isinstance(pred, JaxprTracer)
                  else Literal(_np(pred), pred.aval)]
        invals += [o.val if isinstance(o, JaxprTracer)
                   else Literal(_np(o), o.aval) for o in operands]
        return pred.trace.process_control_flow(
            PRIMITIVES["cond"], (tj, fj), invals, {"_out_avals": out_avals})

    def while_loop(self, cond_fun, body_fun, init_val):
        """`lax.while_loop`：body **只进图一次**，循环在运行时发生。

        ⚠ 官方另有一条限制：**`while_loop` 只支持前向模式微分**（见
          specs.WHILE_FORWARD_ONLY）。反传的规则在本模型里**故意不注册**，
          于是 `grad` 走到它时会抛 `NotImplementedError`。
        """
        if not isinstance(init_val, Tracer):
            return _eager_while(cond_fun, body_fun, init_val)
        aval = init_val.aval
        cj = _trace_subjaxpr(lambda v: cond_fun(v), [aval], "while_cond")
        bj = _trace_subjaxpr(lambda v: body_fun(v), [aval], "while_body")
        return init_val.trace.process_control_flow(
            PRIMITIVES["while"], (cj, bj), [init_val.val],
            {"_out_avals": [_aval_of(bj.outvars[0])]})

    def fori_loop(self, lower, upper, body_fun, init_val):
        """`lax.fori_loop`：**本模型把它落成 `scan`**（于是它有反向规则）。

        ⚠ 真实 JAX 对 `fori_loop` 的降级规则（什么时候走 `scan`）**本仓库没有取到
          一手来源**，所以这里只把它当作「循环的另一种写法」来对照，
          **不对真实 JAX 的降级做任何断言**（见 specs 文件头的 UNSOURCED 段）。
        """
        n = int(upper) - int(lower)
        if not isinstance(init_val, Tracer):
            v = init_val
            for i in range(int(lower), int(upper)):
                v = body_fun(i, v)
            return v
        aval = init_val.aval
        bj = _trace_subjaxpr(
            lambda c, i: body_fun(_LoopIndex(i, aval), c),
            [aval, ShapedArray((), "int32")], "fori_body")
        return init_val.trace.process_control_flow(
            PRIMITIVES["scan"], (bj,), [init_val.val],
            {"length": n, "_out_avals": [_aval_of(bj.outvars[0])]})

    def scan(self, body_fun, init, xs):
        """`lax.scan`：与 `while_loop` 一样，body **只进图一次**。"""
        if not isinstance(init, Tracer):
            carry = init
            ys = []
            for i in range(xs.shape[0]):
                carry, y = body_fun(carry, Array(np.asarray(xs.data)[i]))
                ys.append(_np(y))
            return carry, Array(np.stack(ys))
        aval = init.aval
        bj = _trace_subjaxpr(lambda c, x: body_fun(c, x),
                             [aval, _unbatched_aval(xs, 0)], "scan_body")
        invals = [init.val, xs.val if isinstance(xs, JaxprTracer)
                  else Literal(_np(xs), xs.aval)]
        return init.trace.process_control_flow(
            PRIMITIVES["scan"], (bj,), invals,
            {"length": int(xs.aval.shape[0]),
             "_out_avals": [_aval_of(v) for v in bj.outvars]})

    def psum(self, x, axis_name):
        """`lax.psum`：**显式的**跨设备求和——`shard_map` 里要你自己写。"""
        if not isinstance(x, Tracer):
            return _psum_all_reduce(x)
        return bind(PRIMITIVES["psum"], x, axis_name=axis_name)


class _LoopIndex:
    """`fori_loop` 体里的循环变量：**本模型把它当成常量**（真实 JAX 里是 traced 值）。"""

    __slots__ = ("i", "aval")

    def __init__(self, i, aval):
        self.i = i
        self.aval = ShapedArray((), "int32")


def _eager_while(cond_fun, body_fun, init):
    v = init
    guard = 0
    while bool(np.asarray(_np(cond_fun(v)))):
        v = body_fun(v)
        guard += 1
        if guard > 100000:
            raise RuntimeError("mini: while_loop 未收敛")
    return v


lax = _LaxNS()

# `psum` 在**没有设备网格**时的语义：单设备上就是一个恒等（all-reduce 一台上等于自己）
_PSUM_SCOPE = []


def _psum_all_reduce(x):
    if _PSUM_SCOPE:
        raise RuntimeError("mini: psum 在设备网格里必须由网格求值器处理")
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 十三、多设备：Mesh / PartitionSpec / shard_map / pmap / pjit
# ---------------------------------------------------------------------------

class Mesh:
    """设备网格。本模型只支持一条轴。"""

    def __init__(self, devices, axis_names):
        self.devices = tuple(devices)
        self.axis_names = tuple(axis_names)
        self.shape = (len(self.devices),)

    def __enter__(self):
        _MESH_STACK.append(self)
        return self

    def __exit__(self, *exc):
        _MESH_STACK.pop()
        return False

    def __repr__(self):
        return "Mesh(%s, %s)" % (list(self.devices), list(self.axis_names))


_MESH_STACK = []


class PartitionSpec:
    __slots__ = ("dims",)

    def __init__(self, *dims):
        self.dims = tuple(dims)

    def __eq__(self, other):
        return isinstance(other, PartitionSpec) and self.dims == other.dims

    def __hash__(self):
        return hash(("PartitionSpec", self.dims))

    def __repr__(self):
        return "P(%s)" % ", ".join("None" if d is None else str(d) for d in self.dims)


def _split_on_axis(arr, n, axis=0):
    return [np.asarray(p) for p in np.array_split(np.asarray(arr), n, axis=axis)]


def _concat_on_axis(parts, axis=0):
    if all(np.ndim(p) == 0 for p in parts):
        return np.asarray(parts)
    return np.concatenate([np.asarray(p) for p in parts], axis=axis)


class Partitioned:
    """`jit(..., in_shardings=…, out_shardings=…)` 的产物：**分区器自动插集合通信**。

    本模型实现的规则很少但足够说明问题：如果某个 `reduce_sum` 要跨越**分片轴**归约，
    分区器会把它拆成「本地归约 + 跨设备 `psum`」——**用户一个 collective 都没写**。
    这就是「自动分片」那条路的形态；对照 `shard_map`（用户自己写 `lax.psum`）。
    """

    def __init__(self, closed_jaxpr, in_spec, out_spec, mesh):
        self.closed_jaxpr = closed_jaxpr
        self.in_spec = in_spec
        self.out_spec = out_spec
        self.mesh = mesh
        self.inserted_collectives = 0
        self.auto_partitioned_program = None

    def __call__(self, *args):
        n = len(self.mesh.devices)
        axis_name = self.mesh.axis_names[0]
        cj = self.closed_jaxpr
        envs = []
        for d in range(n):
            env = {}
            for i, (var, a) in enumerate(zip(cj.jaxpr.invars, args)):
                spec = self.in_spec[i] if i < len(self.in_spec) else PartitionSpec()
                dim = spec.dims[0] if spec.dims else None
                shard = _split_on_axis(_np(a), n)[d] if dim == axis_name else _np(a)
                env[var] = shard
            envs.append(env)
        self.inserted_collectives = 0
        for eqn in cj.jaxpr.eqns:
            if eqn.primitive.name == "reduce_sum":
                # 本地归约 + **自动**跨设备 psum
                for d in range(n):
                    src = envs[d][eqn.invars[0]] if isinstance(eqn.invars[0], Var) \
                        else eqn.invars[0].val
                    envs[d][eqn.outvars[0]] = np.sum(np.asarray(src))
                total = sum(np.asarray(envs[d][eqn.outvars[0]]) for d in range(n))
                for d in range(n):
                    envs[d][eqn.outvars[0]] = total
                self.inserted_collectives += 1
                continue
            for d in range(n):
                outs = _eval_eqn(eqn, envs[d])
                for v, o in zip(eqn.outvars, outs):
                    envs[d][v] = o
        outs = []
        for v in cj.jaxpr.outvars:
            parts = [e[v] for e in envs]
            out_dim = (self.out_spec.dims[0] if self.out_spec is not None
                       and self.out_spec.dims else None)
            outs.append(parts[0] if out_dim != axis_name else _concat_on_axis(parts))
        packed = [Array(o) for o in outs]
        return packed[0] if len(packed) == 1 else tuple(packed)


class ShardMap:
    """`jax.shard_map`：**显式的** SPMD 接口。

    官方定位逐字：「a single-program multiple-data (SPMD) multi-device parallelism
    API… **Mapped function applications, or instances, communicate with each other via
    explicit collective communication operations.**」，并且是
    「**complementary to, and composable with, the automatic compiler-based
    parallelization built into `jit`**」（见 specs.SHARD_MAP_VERBATIM）。

    与 `jit(..., in_shardings=…)` 的分工就是本模型里**可数**的那个差：
      · `jit` 那条路：**分区器替你插 `psum`**（`Partitioned.inserted_collectives`）；
      · `shard_map` 这条路：**你自己在函数体里写 `lax.psum`**，包装器一个都不插
        （`ShardMap.inserted_collectives` 恒为 0）。
    """

    def __init__(self, fun, mesh, in_specs, out_specs):
        self.fun = fun
        self.mesh = mesh
        self.in_specs = in_specs
        self.out_specs = out_specs
        self.trace_count = 0
        self.inserted_collectives = 0          # ★ 恒为 0：shard_map 不替你插任何东西
        self.user_collectives = 0
        self.jaxpr = None

    def _local_jaxpr(self, local_avals):
        self.trace_count += 1
        self.jaxpr = _trace_subjaxpr(lambda *a: self.fun(*a), local_avals,
                                     "shard_map_body")
        self.user_collectives = self.jaxpr.op_histogram().get("psum", 0)
        return ClosedJaxpr(self.jaxpr, [])

    def __call__(self, *args):
        n = len(self.mesh.devices)
        axis_name = self.mesh.axis_names[0]
        local_avals = []
        for i, a in enumerate(args):
            spec = self.in_specs[i]
            dim = spec.dims[0] if spec.dims else None
            av = a.aval
            shape = tuple(d // n if (j == 0 and dim == axis_name) else d
                          for j, d in enumerate(av.shape))
            local_avals.append(ShapedArray(shape, av.dtype, av.weak_type))
        cj = self._local_jaxpr(local_avals)
        # ★ 设备间**同步推进**（psum 要看到所有设备的局部值）
        envs = []
        for d in range(n):
            env = {}
            for i, (var, a) in enumerate(zip(cj.jaxpr.invars, args)):
                spec = self.in_specs[i]
                dim = spec.dims[0] if spec.dims else None
                env[var] = (_split_on_axis(_np(a), n)[d] if dim == axis_name
                            else _np(a))
            envs.append(env)
        for eqn in cj.jaxpr.eqns:
            if eqn.primitive.name == "psum":
                srcs = [envs[d][eqn.invars[0]] for d in range(n)]
                total = np.sum([np.asarray(s) for s in srcs], axis=0)
                for d in range(n):
                    envs[d][eqn.outvars[0]] = total
                continue
            for d in range(n):
                outs = _eval_eqn(eqn, envs[d])
                for v, o in zip(eqn.outvars, outs):
                    envs[d][v] = o
        outs = []
        for i, v in enumerate(cj.jaxpr.outvars):
            parts = [e[v] for e in envs]
            spec = self.out_specs[i] if i < len(self.out_specs) else PartitionSpec()
            dim = spec.dims[0] if spec.dims else None
            outs.append(_concat_on_axis(parts) if dim == axis_name else parts[0])
        return Array(outs[0]) if len(outs) == 1 else tuple(Array(o) for o in outs)


def shard_map(fun, mesh, in_specs, out_specs):
    return ShardMap(fun, mesh, in_specs, out_specs)


# -- pmap：**已被官方劝退** -------------------------------------------------

def pmap(fun, axis_name="i", **kwargs):
    """`jax.pmap`：**旧的多设备写法**。

    官方 docstring 逐字：「**Old way of doing parallel map. Use `jax.shard_map()`
    instead.**」，并且「**`pmap()` is now implemented in terms of `jit()` and
    `shard_map()`**」（见 specs.PMAP_DOC_VERBATIM / specs.PMAP_IMPL_VERBATIM）。

    本模型如实体现这两条：它**发一条 DeprecationWarning**，然后**转交给
    jit + shard_map 那条路**——也就是说 pmap 自己没有独立实现。
    """
    import warnings
    try:
        import specs
        msg = specs.PMAP_DEPRECATION_MESSAGE
    except Exception:
        msg = "jax.pmap is deprecated; use jax.shard_map instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    def wrapped(*args):
        mesh = Mesh(tuple("dev:%d" % i for i in range(len(_DEVICES))), (axis_name,))
        spec = PartitionSpec(axis_name)
        sm = shard_map(lambda *a: fun(*a), mesh, [spec] * len(args),
                       [spec] * len(args))
        return sm(*args)

    wrapped.__name__ = getattr(fun, "__name__", "pmapped")
    wrapped.__wrapped__ = fun
    return wrapped


_DEVICES = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]


def pjit(fun=None, **kwargs):
    """`jax.experimental.pjit.pjit`：**已并入 `jax.jit`**。

    源码注释里给了精确日期「**Added Oct 13, 2025**」，消息逐字
    「`jax.experimental.pjit.pjit` has been deprecated. Please use `jax.jit`
    instead.」（见 specs.PJIT_MERGE_VERBATIM）。
    ⚠ 注意这点与 `pmap` 不同：`pmap` 只是被劝退（本体还在，改成用别的实现），
      **`pjit` 是被并进 `jax.jit`**——所以 `jax.jit` 的返回类型就叫
      `pjit.JitWrapped`（见 specs.JIT_RETURN_TYPE）。
    """
    import warnings
    try:
        import specs
        msg = specs.PJIT_DEPRECATION_MESSAGE
    except Exception:
        msg = ("jax.experimental.pjit.pjit has been deprecated. "
           "Please use jax.jit instead.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if fun is None:
        return lambda f: jit(f, **kwargs)
    return jit(fun, **kwargs)


# ---------------------------------------------------------------------------
# 十四、export：**只有 tracing 与 lowering 被省下**
# ---------------------------------------------------------------------------

class SymbolicShape:
    __slots__ = ("names",)

    def __init__(self, names):
        self.names = tuple(names)

    def __repr__(self):
        return "symbolic_shape(%s)" % ", ".join(self.names)


class Exported:
    """导出产物：带的是 **jaxpr + 降低后的程序**，不是 Python 函数。

    ★ 模型里的两条表现（对应官方那句「**Only the tracing and the lowering are
      saved.**」）：
      · **形状与记录一致** → 直接用记录下来的 jaxpr 重放，**不重新 tracing**；
      · **形状变了** → **重新 tracing**（于是「形状多态」**也不免除重编译**）。
    """

    __slots__ = ("closed_jaxpr", "avals", "source", "recompiles", "traces",
                 "symbolic")

    def __init__(self, closed_jaxpr, avals, source, symbolic=None):
        self.closed_jaxpr = closed_jaxpr
        self.avals = avals
        self.source = source
        self.symbolic = symbolic
        self.recompiles = 0
        self.traces = 0

    def serialize(self):
        return repr(self.closed_jaxpr).encode()

    def __call__(self, *args):
        got = tuple(abstractify(a) for a in args)
        if got != self.avals:
            self.recompiles += 1
            self.traces += 1
            cj = make_jaxpr(self.source.fun)(*got)
            outs = _eval_closed(cj, list(args))
        else:
            outs = _eval_closed(self.closed_jaxpr, list(args))
        return outs[0] if len(outs) == 1 else tuple(outs)


class _ExportNS:
    """`jax.export` 的模型。

    官方对形状多态的逐字说明：「such functions are **still re-compiled on demand for
    each concrete input shape**… **Only the tracing and the lowering are saved.**」
    （见 specs.EXPORT_SYMBOLIC_VERBATIM）

    重要推论（**本主题的一条结论**）：**`jax.jit` 没有 dynamic shape 开关**——
    形状多态是 `jax.export` 这一层的事，而且**它也不免除重编译**。
    """

    def symbolic_shape(self, spec):
        """`jax.export.symbolic_shape("a, b")` → 一串**符号维**。"""
        names = [s.strip() for s in spec.split(",") if s.strip()]
        return SymbolicShape(names)

    def export(self, jitted_fn, platforms=("cpu",), **kwargs):
        if not isinstance(jitted_fn, JitWrapped):
            raise TypeError("mini: export 需要一个 jit 过的函数")
        def do(*example_args):
            avals = tuple(abstractify(a) for a in example_args)
            cj = make_jaxpr(jitted_fn.fun)(*avals)
            return Exported(cj, avals, jitted_fn, symbolic=kwargs.get("symbolic"))
        return do

    def deserialize(self, blob):
        return blob


export = _ExportNS()


# ---------------------------------------------------------------------------
# 十五、三类断言的记账器（与 minitf.py 同形）
# ---------------------------------------------------------------------------

class Checks:
    """A 算法正确性 / B 代价模型 / C 边界与反例。失败累计，`summary()` 以非零码退出。"""

    def __init__(self):
        self.counts = {"A": 0, "B": 0, "C": 0}
        self.failures = []

    def _record(self, cls, ok, msg, detail=""):
        self.counts[cls] += 1
        line = "  [%s] %s: %s" % ("ok  " if ok else "FAIL", cls, msg)
        if detail:
            line += "  | " + detail
        print(line)
        if not ok:
            self.failures.append((cls, msg))
        return ok

    def A(self, ok, msg, detail=""):
        return self._record("A", ok, msg, detail)

    def B(self, ok, msg, detail=""):
        return self._record("B", ok, msg, detail)

    def C(self, ok, msg, detail=""):
        return self._record("C", ok, msg, detail)

    def expect_raises(self, fn, exc, msg, detail=""):
        try:
            fn()
        except exc as e:
            return self._record("C", True, msg,
                                detail or ("抛出 %s: %s" % (type(e).__name__, e)))
        except Exception as e:                                    # noqa: BLE001
            return self._record("C", False, msg,
                                detail or "抛了非预期的 %s: %s" % (type(e).__name__, e))
        return self._record("C", False, msg,
                            detail or "**没有抛异常**——失败行为没发生")

    @classmethod
    def merge(cls, parts):
        total = cls()
        for part in parts:
            for k, n in part.counts.items():
                total.counts[k] += n
            total.failures.extend(part.failures)
        return total

    def summary(self):
        total = sum(self.counts.values())
        print()
        print("-" * 78)
        if self.failures:
            for cls, msg in self.failures:
                print("  失败项 [%s] %s" % (cls, msg))
            print("断言汇总：通过 %d 条，失败 %d 条（共 %d 条）"
                  % (total - len(self.failures), len(self.failures), total))
            sys.exit(1)
        print("断言汇总：全部通过 %d 条（A %d / B %d / C %d）"
              % (total, self.counts["A"], self.counts["B"], self.counts["C"]))
        return 0


def rule(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def subsection(title):
    print()
    print("--- %s ---" % title)


def header(number, title):
    print("=" * 76)
    print("%s %s" % (number, title))
    print("=" * 76)
    print("★ 本程序跑的是**本仓库的迷你 JAX 追踪器**，不是 JAX。")
    print("  下面每个数字都只支撑关于**机制**的结论，不支撑任何关于真实 JAX 的结论。")
    print("  迷你实现只用 python3 %s + numpy %s。"
          % (sys.version.split()[0], np.__version__))
