"""本仓库的**迷你 TF 追踪器**：把 `tf.function` 的「按 TraceType 查表 → 命中则复用，
不命中则重追踪」这套机制用几百行 Python 重写一遍。

★★ 这不是 TensorFlow ★★
  本文件是**本仓库自己写的模型**，只依赖 `numpy` 与标准库。它借用了 TensorFlow 的**术语**
  （`TraceType` / `FunctionType` / `ConcreteFunction` / `tf.cond` / SavedModel …）以便对照，
  **结构是我们自己简化的**。凡由它跑出的数字，只能支撑「这套机制可以怎样被计数与观测」
  这类**机制**结论，**不支持任何关于真实 TensorFlow 的结论**。
  关于真实 TF 的每一条行为断言，见 `specs.py`（每条带 `# 来源：` 与可打开的 URL）。

本文件提供的对象：

  · `TensorShape` / `TensorSpec` —— 形状与「子类型方向」（`[1,2] <: [None,None]`）
  · `TensorTraceType` / `LiteralTraceType` / `ListTraceType` / `MappingTraceType` / `GenericTraceType`
  · `FunctionType`                —— **派发键**：各参数 TraceType 的元组
  · `Tensor` / `Tracer`           —— eager 值与追踪期的符号值
  · `Graph` / `Op`                —— 追踪出来的图（顺序表 + 子图）
  · `AtomicFunction` / `ConcreteFunction` / `PolymorphicFunction`
                                  —— 层级：`tf.function`（缓存层）→ ConcreteFunction → 图
  · `tf_function` / `cond` / `while_loop` / `autograph_if`
  · `register_custom_op` / `custom_gradient` / `xla_compile` / `export_saved_model`
  · `Checks`                      —— 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）的记账器

用法：`python3 <编号程序>.py`（见同目录 Makefile）
"""

import json
import os
import sys
import time
import weakref
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# 一、形状与 TraceType
# ---------------------------------------------------------------------------

UNRANKED = None          # `TensorShape(None)` = 未知秩（unranked）


class TensorShape:
    """形状。`dims=None` 表示**未知秩**（unranked）；秩内的 `None` 表示**未知维**。

    ★ 子类型方向（来自 TF 官方 TraceType 规则，见 specs.TRACETYPE_RULES）：
        · 已知秩 是 未知秩 的子类型；
        · 定长维 是 未知维 的子类型；
        · 反过来**不成立**。
      所以 `TensorShape([1, 2]) <: TensorShape([None, None])`，而反向为假。
    """

    __slots__ = ("dims",)

    def __init__(self, dims):
        if dims is None:
            self.dims = None
        else:
            self.dims = tuple(int(d) if d is not None else None for d in dims)

    @property
    def rank(self):
        return None if self.dims is None else len(self.dims)

    def is_subtype_of(self, other):
        """self <: other ？"""
        if not isinstance(other, TensorShape):
            return False
        if other.dims is None:                 # 任何形状都是 unranked 的子类型
            return True
        if self.dims is None:                  # unranked 只是 unranked 的子类型
            return False
        if len(self.dims) != len(other.dims):
            return False
        for a, b in zip(self.dims, other.dims):
            if a is None:
                if b is not None:
                    return False               # 未知维 <: 定长维 不成立
            elif b is not None and a != b:
                return False
        return True

    def relax(self):
        """把所有已知维换成未知维（保留秩）。`reduce_retracing` 的模型。"""
        if self.dims is None:
            return TensorShape(None)
        return TensorShape(tuple(None for _ in self.dims))

    def __eq__(self, other):
        return isinstance(other, TensorShape) and self.dims == other.dims

    def __hash__(self):
        return hash(("TensorShape", self.dims))

    def __str__(self):
        if self.dims is None:
            return "TensorShape(None)"
        return "TensorShape([%s])" % ", ".join(
            "None" if d is None else str(d) for d in self.dims)

    __repr__ = __str__


class TensorSpec:
    """dtype + shape。"""

    __slots__ = ("dtype", "shape")

    def __init__(self, dtype, shape):
        self.dtype = str(dtype)
        self.shape = shape if isinstance(shape, TensorShape) else TensorShape(shape)

    def __eq__(self, other):
        return (isinstance(other, TensorSpec) and self.dtype == other.dtype
                and self.shape == other.shape)

    def __hash__(self):
        return hash(("TensorSpec", self.dtype, self.shape))

    def __str__(self):
        return "TensorSpec(%s, %s)" % (self.shape, self.dtype)


class TraceType:
    """所有 TraceType 的基类。`key()` 给出**查缓存用的相等键**，`__str__` 给出可读名。"""

    def key(self):
        raise NotImplementedError

    def is_subtype_of(self, other):
        """self <: other ？子类型方向决定「能不能复用对方 trace 出来的图」。"""
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        raise NotImplementedError


class TensorTraceType(TraceType):
    """`Tensor` 的 TraceType：由 **dtype + shape** 参数化（官方逐字，见 specs.TRACETYPE_RULES）。"""

    __slots__ = ("spec",)

    def __init__(self, dtype, shape):
        self.spec = TensorSpec(dtype, shape)

    @property
    def dtype(self):
        return self.spec.dtype

    @property
    def shape(self):
        return self.spec.shape

    def key(self):
        return ("Tensor", self.spec.dtype, self.spec.shape)

    def is_subtype_of(self, other):
        return (isinstance(other, TensorTraceType)
                and self.dtype == other.dtype
                and self.shape.is_subtype_of(other.shape))

    def __str__(self):
        return "TensorSpec(%s, %s)" % (self.shape, self.dtype)


class LiteralTraceType(TraceType):
    """Python 原生值的 TraceType：**类型就是值本身**。

    官方逐字：the `TraceType` of the value `3` is `LiteralTraceType<3>`, **not `int`**。
    所以 `3` 与 `4` 是**两个不同的键**——这是 TF 与 JAX 分道扬镳的地方（见 specs.TF_VS_JAX_SCALAR）。
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def key(self):
        return ("Literal", type(self.value).__name__, self.value)

    def is_subtype_of(self, other):
        return (isinstance(other, LiteralTraceType)
                and type(self.value) is type(other.value)
                and self.value == other.value)

    def __str__(self):
        return "LiteralTraceType<%r>" % (self.value,)


class ListTraceType(TraceType):
    """有序容器（list / tuple）：**按元素逐个参数化，且保序**。"""

    __slots__ = ("elements",)

    def __init__(self, elements):
        self.elements = tuple(elements)

    def key(self):
        return ("List", tuple(e.key() for e in self.elements))

    def is_subtype_of(self, other):
        return (isinstance(other, ListTraceType)
                and len(self.elements) == len(other.elements)
                and all(a.is_subtype_of(b)
                        for a, b in zip(self.elements, other.elements)))

    def __str__(self):
        return "ListTraceType<%s>" % ", ".join(str(e) for e in self.elements)


class MappingTraceType(TraceType):
    """映射（dict）：键相同、值换成值的类型；**且与键序无关**。

    官方逐字：`{1: 2, 3: 4}` 与 `{3: 4, 1: 2}` 的类型**相等**（与有序容器相反）。
    """

    __slots__ = ("items",)

    def __init__(self, items):
        self.items = frozenset(items)          # (键, 值的 TraceType)

    def key(self):
        return ("Mapping", frozenset((k, vt.key()) for k, vt in self.items))

    def is_subtype_of(self, other):
        if not isinstance(other, MappingTraceType):
            return False
        mine = dict(self.items)
        theirs = dict(other.items)
        return (set(mine) == set(theirs)
                and all(mine[k].is_subtype_of(theirs[k]) for k in mine))

    def __str__(self):
        return "MappingTraceType<%s>" % ", ".join(
            "<KeyValue<%r, %s>>" % (k, vt) for k, vt in sorted(self.items, key=repr))


class GenericTraceType(TraceType):
    """兜底：其他 Python 对象。按**对象身份 / 相等**匹配（官方给出的顺序）。"""

    __slots__ = ("obj_id", "obj_repr", "_ref")

    def __init__(self, obj):
        self.obj_id = id(obj)
        self.obj_repr = repr(obj)
        try:
            self._ref = weakref.ref(obj)
        except TypeError:
            self._ref = None

    def key(self):
        return ("Generic", self.obj_id)

    def is_subtype_of(self, other):
        if not isinstance(other, GenericTraceType):
            return False
        if self.obj_id == other.obj_id:
            return True                        # 同一个对象
        mine = self._ref() if self._ref else None
        theirs = other._ref() if other._ref else None
        return mine is not None and theirs is not None and mine == theirs

    def __str__(self):
        return "GenericTraceType<%s>" % self.obj_repr


def trace_type_of(value):
    """把一次调用的实参映射成 TraceType。规则照 TF 的官方列表（见 specs.TRACETYPE_RULES）。"""
    if isinstance(value, Tracer):
        return TensorTraceType(value.dtype, value.shape)
    if isinstance(value, Tensor):
        return TensorTraceType(value.dtype, value.shape)
    if isinstance(value, Variable):
        return VariableTraceType(value)
    if isinstance(value, TensorSpec):
        return TensorTraceType(value.dtype, value.shape)
    if isinstance(value, bool) or isinstance(value, (int, float, str, bytes)) or value is None:
        return LiteralTraceType(value)
    if isinstance(value, (list, tuple)):
        return ListTraceType([trace_type_of(v) for v in value])
    if isinstance(value, dict):
        return MappingTraceType([(k, trace_type_of(v)) for k, v in value.items()])
    return GenericTraceType(value)


class VariableTraceType(TraceType):
    """`Variable`：与 Tensor 类似，但**还带一个唯一的 resource ID**（官方逐字）。"""

    __slots__ = ("dtype", "shape", "resource_id")

    def __init__(self, var):
        self.dtype = var.dtype
        self.shape = var.shape
        self.resource_id = var.resource_id

    def key(self):
        return ("Variable", self.dtype, self.shape, self.resource_id)

    def is_subtype_of(self, other):
        return (isinstance(other, VariableTraceType)
                and self.resource_id == other.resource_id
                and self.dtype == other.dtype)

    def __str__(self):
        return "VariableSpec(%s, %s, id=%d)" % (self.shape, self.dtype, self.resource_id)


class FunctionType:
    """**派发键**：各参数 TraceType 组成的元组。

    官方逐字：`tf.function` 先算出每个实参的 TraceType，用它拼出 `FunctionType`，
    再拿它与已有 `ConcreteFunction` 的 `FunctionType` 比对；命中就派发过去，不命中就新追一个。
    """

    __slots__ = ("types",)

    def __init__(self, types):
        self.types = tuple(types)

    def key(self):
        return tuple(t.key() for t in self.types)

    def is_subtype_of(self, other):
        return (len(self.types) == len(other.types)
                and all(a.is_subtype_of(b) for a, b in zip(self.types, other.types)))

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return "FunctionType(%s)" % ", ".join(str(t) for t in self.types)


# ---------------------------------------------------------------------------
# 二、值：eager 的 Tensor / Variable 与追踪期的 Tracer
# ---------------------------------------------------------------------------

STATS = {"eager_ops": 0, "eager_dispatch_layers": 0, "traced_ops": 0, "replays": 0}

# 本仓库迷你实现给「eager 一次算子调用」定的层数（**模型，不是 TF 的实测值**）。
# 真实 TF 的 eager 路径分层见 specs.EAGER_PATH_LAYERS 的说明；这里只是让代价模型有对象可算。
EAGER_LAYERS_PER_OP = 3          # python 绑定 → dispatcher → kernel

_UID = [0]


def _next_uid():
    _UID[0] += 1
    return _UID[0]


_DTYPE_OF = {np.dtype("float32"): "float32", np.dtype("float64"): "float64",
             np.dtype("int32"): "int32", np.dtype("int64"): "int64",
             np.dtype("bool"): "bool"}


def _dtype_name(arr):
    return _DTYPE_OF.get(arr.dtype, str(arr.dtype))


class Tensor:
    """eager 值：numpy 数组 + dtype。每次算子调用都付 `EAGER_LAYERS_PER_OP` 层代价。"""

    __slots__ = ("data",)

    def __init__(self, data, dtype=None):
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype)
        self.data = arr

    @property
    def dtype(self):
        return _dtype_name(self.data)

    @property
    def shape(self):
        return TensorShape(self.data.shape)

    @property
    def numpy(self):
        return self.data

    def __repr__(self):
        return "<Tensor: shape=%s, dtype=%s, numpy=%s>" % (
            tuple(self.data.shape), self.dtype, np.array2string(self.data))


class Tracer:
    """追踪期的符号值：只有 dtype / shape / 在图里的引用，没有数据。"""

    __slots__ = ("uid", "dtype", "shape", "ctx")

    def __init__(self, uid, dtype, shape, ctx):
        self.uid = uid
        self.dtype = dtype
        self.shape = shape
        self.ctx = ctx

    def __bool__(self):
        raise OperatorNotAllowedInGraphError(
            "Using a symbolic mini-Tensor as a Python `bool` is not allowed in this "
            "mini implementation. Use cond() instead."
            " ——本句是本仓库迷你实现的措辞，不是 TensorFlow 的原文；"
            "真实 TF 的对应报错见 specs.ERR_TASK_BOOL_CAST。")

    def __iter__(self):
        raise OperatorNotAllowedInGraphError(
            "Iterating over a symbolic mini-Tensor is not allowed in this mini "
            "implementation. ——本句是本仓库迷你实现的措辞，不是 TensorFlow 的原文；"
            "真实 TF 的对应报错见 specs.ERR_TASK_ITERATION。")

    def __repr__(self):
        return "<Tracer#%d: shape=%s, dtype=%s>" % (self.uid, self.shape, self.dtype)


class Variable:
    """可变量。TraceType 里带 resource ID（官方规则）。"""

    __slots__ = ("value", "resource_id", "name")

    _ids = [0]

    def __init__(self, initial, name="v"):
        self.value = initial if isinstance(initial, Tensor) else Tensor(initial)
        Variable._ids[0] += 1
        self.resource_id = Variable._ids[0]
        self.name = "%s:%d" % (name, self.resource_id)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    def assign(self, new):
        self.value = new
        return self


# ---------------------------------------------------------------------------
# 三、图与算子
# ---------------------------------------------------------------------------

class Op:
    """图里的一个节点。`inputs` 是 uid 列表；`subgraphs` 给 cond / while 用。"""

    __slots__ = ("name", "inputs", "attrs", "out_uids", "subgraphs")

    def __init__(self, name, inputs=(), attrs=None, out_uids=(),
                 subgraphs=None):
        self.name = name
        self.inputs = tuple(inputs)
        self.attrs = dict(attrs or {})
        self.out_uids = tuple(out_uids)
        self.subgraphs = dict(subgraphs or {})

    def __repr__(self):
        return "%s(in=%s)" % (self.name, list(self.inputs))


class Graph:
    """顺序表 + 子图。迷你实现里没有块/控制依赖，只有「谁在谁前面」。"""

    __slots__ = ("name", "ops")

    def __init__(self, name):
        self.name = name
        self.ops = []

    def add(self, op):
        self.ops.append(op)
        STATS["traced_ops"] += 1
        return op

    def op_names(self):
        return [op.name for op in self.ops]

    def name_counts(self):
        counts = OrderedDict()
        for op in self.ops:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def size(self):
        """含子图的**总节点数**（结构大小，与数据无关）。"""
        return len(self.ops) + sum(
            sg.size() for op in self.ops for sg in op.subgraphs.values())

    def all_op_names(self):
        names = list(self.op_names())
        for op in self.ops:
            for sg in op.subgraphs.values():
                names.extend(sg.all_op_names())
        return names

    def __str__(self):
        lines = ["graph %r {" % self.name]
        for op in self.ops:
            extra = ""
            for label, sg in op.subgraphs.items():
                extra += "  [子图 %s: %d 个节点]" % (label, len(sg.ops))
            for k, v in sorted(op.attrs.items()):
                if k.endswith("_literal"):
                    extra += "  [字面量 %s=%s]" % (k, v)
            lhs = "%s = " % ", ".join("%%%d" % u for u in op.out_uids) \
                if op.out_uids else ""
            lines.append("  %s%s(%s)%s" % (
                lhs, op.name, ", ".join("%%%d" % u for u in op.inputs), extra))
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 四、追踪上下文
# ---------------------------------------------------------------------------

_CONTEXT_STACK = []


class TraceContext:
    """追踪期的当前图。进/出用 `with`。"""

    __slots__ = ("graph", "function_type")

    def __init__(self, graph, function_type):
        self.graph = graph
        self.function_type = function_type

    def __enter__(self):
        _CONTEXT_STACK.append(self)
        return self

    def __exit__(self, *exc):
        _CONTEXT_STACK.pop()
        return False


def current_context():
    return _CONTEXT_STACK[-1] if _CONTEXT_STACK else None


class OperatorNotAllowedInGraphError(Exception):
    """对应 TF 的 `errors.OperatorNotAllowedInGraphError`（**同名，不同实现**）。"""


class InvalidArgumentError(Exception):
    """对应 TF 的 `errors.InvalidArgumentError`（**同名，不同实现**）。"""


class FatalError(Exception):
    pass


# ---------------------------------------------------------------------------
# 五、算子
#
# 每个算子两副面孔：
#   · 没有激活的 TraceContext → eager：真的用 numpy 算，并记 `EAGER_LAYERS_PER_OP` 层代价；
#   · 有激活的 TraceContext → 往当前图里记一个节点，返回一个 Tracer（带形状推导）。
# 这就是本域主线「eager 每次走全链 / 图只走一次」在迷你实现里的落点。
# ---------------------------------------------------------------------------

def _as_array(v):
    if isinstance(v, Tensor):
        return v.data
    if isinstance(v, Variable):
        return v.value.data
    return np.asarray(v)


def _bcast_shape(a, b):
    try:
        return TensorShape(np.broadcast_shapes(tuple(a) if a else (), tuple(b) if b else ()))
    except ValueError:
        raise InvalidArgumentError("mini: incompatible shapes %s 与 %s" % (a, b))


def _emit(name, inputs, dtype, shape, attrs=None, subgraphs=None):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return None
    out = Tracer(_next_uid(), dtype, shape, ctx)
    ctx.graph.add(Op(name, [t.uid for t in inputs], attrs, [out.uid], subgraphs))
    return out


def _binary(name, a, b, np_fn, force_dtype=None):
    ctx = current_context()
    if ctx is None:
        x, y = _as_array(a), _as_array(b)
        r = np_fn(x, y)
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(r, dtype=force_dtype)
    a_is_t = isinstance(a, (Tensor, Tracer))
    b_is_t = isinstance(b, (Tensor, Tracer))
    if a_is_t and b_is_t and a.dtype != b.dtype:
        raise InvalidArgumentError(
            "mini: cannot compute %s of tensors with different dtypes %s vs %s"
            % (name, a.dtype, b.dtype))
    # 一边是 Python 原生值时，按另一边的 dtype 走（真实 TF 也是这么做的）
    dtype = a.dtype if a_is_t else (b.dtype if b_is_t else _dtype_name(np.asarray(a)))
    sa = a.shape if a_is_t else TensorShape(np.asarray(a).shape)
    sb = b.shape if b_is_t else TensorShape(np.asarray(b).shape)
    inputs = [v for v in (a, b) if isinstance(v, (Tensor, Tracer))]
    attrs = {}
    if not a_is_t:
        attrs["lhs_literal"] = float(np.asarray(a))
    if not b_is_t:
        attrs["rhs_literal"] = float(np.asarray(b))
    return _emit(name, inputs, force_dtype or dtype, _bcast_shape(sa.dims, sb.dims),
                 attrs=attrs)


def add(a, b):
    return _binary("tf.add", a, b, np.add)


def sub(a, b):
    return _binary("tf.subtract", a, b, np.subtract)


def mul(a, b):
    return _binary("tf.multiply", a, b, np.multiply)


def maximum(a, b):
    return _binary("tf.maximum", a, b, np.maximum)


def matmul(a, b):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(np.matmul(_as_array(a), _as_array(b)))
    sa, sb = a.shape.dims, b.shape.dims
    if sa is None or sb is None:
        out = TensorShape(None)
    else:
        out = TensorShape((sa[0], sb[1]))
    return _emit("tf.matmul", [a, b], a.dtype, out)


def greater(a, b):
    out = _binary("tf.greater", a, b, np.greater, force_dtype="bool")
    return out


def reduce_sum(x):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(np.sum(_as_array(x)))
    return _emit("tf.reduce_sum", [x], x.dtype, TensorShape(()))


def relu(x):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(np.maximum(_as_array(x), 0.0))
    return _emit("tf.nn.relu", [x], x.dtype, x.shape)


def cast(x, dtype):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(_as_array(x), dtype=dtype)
    return _emit("tf.cast", [x], dtype, x.shape)


def sqrt(x):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(np.sqrt(_as_array(x)))
    return _emit("tf.sqrt", [x], x.dtype, x.shape)


def exp(x):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(np.exp(_as_array(x)))
    return _emit("tf.exp", [x], x.dtype, x.shape)


def log1p(x):
    ctx = current_context()
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(np.log1p(_as_array(x)))
    return _emit("tf.math.log1p", [x], x.dtype, x.shape)


def tf_print(x, note="print"):
    """`tf.print` 的迷你对应物。**XLA 编不了它**——见 `xla_compile` 的 UNSUPPORTED_OPS。"""
    ctx = current_context()
    if ctx is None:
        print("    [tf.print] %s: %s" % (note, np.array2string(_as_array(x))))
        return None
    return _emit("tf.print", [x], "bool", TensorShape(()))


def const(value, dtype=None):
    ctx = current_context()
    arr = np.asarray(value)
    if ctx is None:
        return Tensor(arr, dtype=dtype)
    out = Tracer(_next_uid(), _dtype_name(arr) if dtype is None else dtype,
                 TensorShape(arr.shape), ctx)
    ctx.graph.add(Op("tf.constant", [], {"value": np.asarray(value).tolist()},
                     [out.uid]))
    return out


# ---------------------------------------------------------------------------
# 六、控制流：cond / while_loop
# ---------------------------------------------------------------------------

def cond(pred, true_fn, false_fn):
    """`tf.cond`：**两个分支都被 trace 进图**，运行时才选（官方逐字，见 specs.COND_BOTH_BRANCHES）。

    eager 下调（没有激活的 TraceContext）时退回 Python `if`，与 TF 一致。
    """
    ctx = current_context()
    if ctx is None:
        return true_fn() if bool(_as_array(pred)) else false_fn()

    then_graph, else_graph = Graph("cond/then"), Graph("cond/else")
    with TraceContext(then_graph, ctx.function_type):
        r_then = true_fn()
    with TraceContext(else_graph, ctx.function_type):
        r_else = false_fn()

    if isinstance(r_then, Tracer) and isinstance(r_else, Tracer):
        if r_then.dtype != r_else.dtype:
            raise InvalidArgumentError(
                "mini: tf.cond branches must return the same dtype; got %s and %s"
                % (r_then.dtype, r_else.dtype))
        out_shape = r_then.shape if r_then.shape.rank is None else r_else.shape
        dtype = r_then.dtype
    elif isinstance(r_then, Tracer):
        dtype, out_shape = r_then.dtype, r_then.shape
    elif isinstance(r_else, Tracer):
        dtype, out_shape = r_else.dtype, r_else.shape
    else:
        # 两个分支都没产生符号值：cond 只是副作用，无返回值
        _emit("tf.cond", [pred], "bool", TensorShape(()), attrs={
            "then_out": None, "else_out": None},
            subgraphs={"then": then_graph, "else": else_graph})
        return None

    return _emit("tf.cond", [pred], dtype, out_shape, attrs={
        "then_out": r_then.uid if isinstance(r_then, Tracer) else None,
        "else_out": r_else.uid if isinstance(r_else, Tracer) else None,
    }, subgraphs={"then": then_graph, "else": else_graph})


def while_loop(cond_fn, body_fn, init):
    """`tf.while_loop`：**body 只 trace 一次**（循环变量是符号值），循环在运行时发生。"""
    ctx = current_context()
    if ctx is None:
        v = init
        it = 0
        while bool(_as_array(cond_fn(v))):
            v = body_fn(v)
            it += 1
            if it > 10000:
                raise FatalError("mini: while_loop 未收敛")
        return v

    if not isinstance(init, Tracer):
        init = const(_as_array(init))
    loop_var = Tracer(_next_uid(), init.dtype, init.shape, ctx)
    cond_graph = Graph("while/cond")
    with TraceContext(cond_graph, ctx.function_type):
        c_out = cond_fn(loop_var)
    body_graph = Graph("while/body")
    with TraceContext(body_graph, ctx.function_type):
        b_out = body_fn(loop_var)
    return _emit("tf.while_loop", [init], b_out.dtype, b_out.shape, attrs={
        "iv_uid": loop_var.uid, "cond_out": c_out.uid, "body_out": b_out.uid,
    }, subgraphs={"cond": cond_graph, "body": body_graph})


def autograph_if(pred, true_fn, false_fn):
    """AutoGraph 的模型：**条件是符号值就转成 `tf.cond`；否则按 Python 条件执行**。

    官方逐字：This substitution is made **if `<condition>` is a Tensor**.
    Otherwise, the `if` statement is executed as a Python conditional.（见 specs.AUTOGRAPH_IF）
    """
    if isinstance(pred, Tracer):
        return cond(pred, true_fn, false_fn)
    return true_fn() if bool(pred) else false_fn()


# ---------------------------------------------------------------------------
# 七、图解释器：让 ConcreteFunction 可以被「重放」
# ---------------------------------------------------------------------------

def _eval_binary(name, x, y):
    if name == "tf.add":
        return np.add(x, y)
    if name == "tf.subtract":
        return np.subtract(x, y)
    if name == "tf.multiply":
        return np.multiply(x, y)
    if name == "tf.maximum":
        return np.maximum(x, y)
    if name == "tf.greater":
        return np.greater(x, y)
    if name == "tf.matmul":
        return np.matmul(x, y)
    raise FatalError("mini: 解释器不认识算子 %s" % name)


def _eval_op(op, env):
    name = op.name
    if name == "tf.constant":
        return np.asarray(op.attrs["value"])
    if name in ("tf.add", "tf.subtract", "tf.multiply", "tf.maximum",
                "tf.greater", "tf.matmul"):
        if len(op.inputs) == 2:
            x, y = env[op.inputs[0]], env[op.inputs[1]]
        elif len(op.inputs) == 1:
            # 另一边在 trace 时就被烧成了字面量（C2 的机制：Python 值按值进图）
            if "lhs_literal" in op.attrs:
                x, y = np.asarray(op.attrs["lhs_literal"]), env[op.inputs[0]]
            else:
                x, y = env[op.inputs[0]], np.asarray(op.attrs["rhs_literal"])
            x = np.asarray(x, dtype=y.dtype) if hasattr(y, "dtype") else np.asarray(x)
        else:
            x = np.asarray(op.attrs["lhs_literal"])
            y = np.asarray(op.attrs["rhs_literal"])
        return _eval_binary(name, x, y)
    if name == "tf.reduce_sum":
        return np.sum(env[op.inputs[0]])
    if name == "tf.nn.relu":
        return np.maximum(env[op.inputs[0]], 0.0)
    if name == "tf.sqrt":
        return np.sqrt(env[op.inputs[0]])
    if name == "tf.exp":
        return np.exp(env[op.inputs[0]])
    if name == "tf.math.log1p":
        return np.log1p(env[op.inputs[0]])
    if name == "tf.cast":
        return np.asarray(env[op.inputs[0]])
    if name == "tf.print":
        print("    [tf.print@replay] %s" % np.array2string(env[op.inputs[0]]))
        return np.asarray(True)
    if name == "tf.cond":
        chosen = "then" if bool(env[op.inputs[0]]) else "else"
        sg = op.subgraphs[chosen]
        _eval_graph(sg, env)
        out_uid = op.attrs.get("then_out" if chosen == "then" else "else_out")
        return env[out_uid] if out_uid is not None else np.asarray(True)
    if name == "tf.while_loop":
        iv = op.attrs["iv_uid"]
        env[iv] = np.asarray(env[op.inputs[0]])
        # 循环体一次都不跑时，结果就是初值（官方参考实现同样这么处理）
        env[op.attrs["body_out"]] = env[iv]
        cond_g, body_g = op.subgraphs["cond"], op.subgraphs["body"]
        it = 0
        while True:
            _eval_graph(cond_g, env)
            if not bool(env[op.attrs["cond_out"]]):
                break
            _eval_graph(body_g, env)
            env[iv] = env[op.attrs["body_out"]]
            it += 1
            if it > 10000:
                raise FatalError("mini: while_loop 未收敛")
        return env[op.attrs["body_out"]]
    if name.endswith("__custom"):
        impl = _CUSTOM_OPS[name[: -len("__custom")]]
        return impl(*[env[u] for u in op.inputs])
    raise FatalError("mini: 解释器不认识算子 %s" % name)


def _eval_graph(graph, env):
    for op in graph.ops:
        val = _eval_op(op, env)
        for uid in op.out_uids:
            env[uid] = val


# ---------------------------------------------------------------------------
# 八、层级：AtomicFunction → ConcreteFunction → PolymorphicFunction（`tf.function`）
# ---------------------------------------------------------------------------

class AtomicFunction:
    """官方定位：**A Python callable for functions in the TF Runtime.**（见 specs.HIERARCHY）

    它持有**图的缓存**（`_cached_graph`）——图在真实 TF 里也落在这一层。
    """

    __slots__ = ("name", "function_type", "_cached_graph", "arg_uids",
                 "arg_positions", "result_uid")

    def __init__(self, name, function_type, graph, arg_uids, result_uid,
                 arg_positions=()):
        self.name = name
        self.function_type = function_type
        self._cached_graph = graph
        self.arg_uids = tuple(arg_uids)          # 被符号化的实参（uid）
        self.arg_positions = tuple(arg_positions)  # 它们在实参列表里的位置
        self.result_uid = result_uid


class ConcreteFunction:
    """「一次 trace 的产物」。

    ⚠ 本项目订正：**计划里那句「can be thought of as a wrapper around a `tf.Graph`」
      在 v2.21.0 的源码里逐字查不到**（见 specs.CONCRETE_FUNCTION_DOC_2_21 的说明）。
      官方 docstring 逐字是「Base class for differentiable graph functions.」
    """

    __slots__ = ("function_type", "atomic_function", "traced_from", "trace_index",
                 "xla_compiled")

    def __init__(self, function_type, atomic_function, traced_from, trace_index):
        self.function_type = function_type
        self.atomic_function = atomic_function          # inference_fn: AtomicFunction
        self.traced_from = traced_from                  # 被 trace 的实参形状（可读）
        self.trace_index = trace_index
        self.xla_compiled = False

    @property
    def graph(self):
        return self.atomic_function._cached_graph

    @property
    def name(self):
        return self.atomic_function.name

    def __call__(self, *values):
        """重放：按图执行，**不再跑 Python 函数体**。"""
        STATS["replays"] += 1
        env = {}
        for uid, pos in zip(self.atomic_function.arg_uids,
                            self.atomic_function.arg_positions):
            env[uid] = np.asarray(_as_array(values[pos]))
        _eval_graph(self.graph, env)
        return Tensor(env[self.atomic_function.result_uid])

    def pretty_printed_signature(self):
        return "%s(%s) -> %s" % (self.name, self.function_type,
                                 self.traced_from)

    def __repr__(self):
        return "<ConcreteFunction %s>" % self.function_type


class FrequentTracingDetector:
    """官方告警器算法的**等价重写**：最近 `MAX_CALL_HISTORY` 次调用里
    `>= THRESHOLD` 次触发 tracing 就告警。常量与文案见 specs（`# 来源：` 指到源码行）。
    """

    def __init__(self, max_call_history, threshold, max_warnings=2):
        self.max_call_history = max_call_history
        self.threshold = threshold
        self.max_warnings = max_warnings
        self.calls_per_tracings = []
        self.call_count = 0
        self.warning_count = 0
        self.warnings = []

    def called_with_tracing(self, function_name):
        self.call_count += 1
        self.calls_per_tracings.append(1)
        while self.calls_per_tracings:
            if self.call_count - self.calls_per_tracings[0] > self.max_call_history:
                self.call_count -= self.calls_per_tracings.pop(0)
            else:
                break
        if self.warning_count >= self.max_warnings:
            return
        if len(self.calls_per_tracings) >= self.threshold:
            self.warning_count += 1
            msg = WARNING_TEMPLATE.format(len(self.calls_per_tracings),
                                          self.call_count, function_name)
            self.warnings.append(msg)
            print("    **重追踪告警** %s" % msg)

    def called_without_tracing(self):
        if not self.calls_per_tracings:
            self.calls_per_tracings = [0]


# 告警文案的**骨架**取自 TF 源码；`{}` 顺序为（触发次数, 窗口内调用数, 函数名）。
# 逐字原文见 specs.FREQUENT_TRACING_WARNING_TEXT（另含固定的首尾措辞）。
WARNING_TEMPLATE = ("{} out of the last {} calls to {} triggered tf.function retracing. "
                    "Tracing is expensive and the excessive number of tracings could "
                    "be due to (1) creating @tf.function repeatedly in a loop, "
                    "(2) passing tensors with different shapes, (3) passing Python "
                    "objects instead of tensors.")


class PolymorphicFunction:
    """`tf.function` 本体：**缓存层**。

    缓存键 = `FunctionType`；命中就复用 `ConcreteFunction`，不命中就重追踪。
    """

    def __init__(self, python_function, input_signature=None, reduce_retracing=False,
                 autograph=False, jit_compile=None, name=None):
        self.python_function = python_function
        self.name = name or getattr(python_function, "__name__", "function")
        self.input_signature = input_signature
        self.reduce_retracing = reduce_retracing
        self.autograph = autograph
        self.jit_compile = jit_compile
        self._function_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._dispatch_log = []           # 每次调用命中了哪个键
        self.detector = FrequentTracingDetector(*specs_warning_constants())

    # -- 内部 ---------------------------------------------------------------
    def _bind(self, args, kwargs):
        if kwargs:
            raise NotImplementedError("mini: 只支持位置实参")
        values = list(args)
        if self.input_signature is not None:
            if len(values) != len(self.input_signature):
                raise ValueError("mini: input_signature 与实参个数不符")
        return values

    def _call_type(self, values):
        return FunctionType([trace_type_of(v) for v in values])

    def _relax(self, ft):
        return FunctionType([
            TensorTraceType(t.dtype, t.shape.relax()) if isinstance(t, TensorTraceType)
            else t for t in ft.types])

    def _lookup(self, call_type):
        if call_type in self._function_cache:
            return self._function_cache[call_type], "exact"
        cands = [(k, cf) for k, cf in self._function_cache.items()
                 if call_type.is_subtype_of(k)]
        if not cands:
            return None, None
        # 官方：If multiple matches are found, **the most specific signature is chosen**
        most_specific = [(k, cf) for k, cf in cands
                         if all(k.is_subtype_of(k2) for k2, _ in cands)]
        k, cf = most_specific[0]
        kind = "exact" if k == call_type else "supertype"
        return cf, kind

    def _trace(self, values, key, call_type):
        """跑一遍 Python 函数体，把算子记进图。

        ★ 关键点（与真实 TF 一致）：**只有张量类实参才被换成符号值**；
          Python 原生值原样传进去——所以它会被**按值烧进图里**（C2 的机制）。
        """
        graph = Graph(self.name)
        ctx = TraceContext(graph, key)
        tracers = []
        arg_uids, arg_positions = [], []
        with ctx:
            for pos, v in enumerate(values):
                if isinstance(v, (Tensor, TensorSpec)):
                    t = Tracer(_next_uid(), v.dtype, v.shape, ctx)
                    arg_uids.append(t.uid)
                    arg_positions.append(pos)
                else:
                    t = v                      # 字面量 / 容器：原样传入，不进图
                tracers.append(t)
            out = self.python_function(*tracers)
        result_uid = out.uid if isinstance(out, Tracer) else None
        atomic = AtomicFunction(self.name, key, graph, arg_uids, result_uid,
                                arg_positions)
        self._cache_misses += 1
        cf = ConcreteFunction(key, atomic, [str(t) for t in key.types],
                              self._cache_misses)
        self._function_cache[key] = cf
        self.detector.called_with_tracing(self.name)
        return cf

    # -- 公开接口 -----------------------------------------------------------
    def __call__(self, *args, **kwargs):
        values = self._bind(args, kwargs)
        call_type = self._call_type(values)
        cf, kind = self._lookup(call_type)
        if cf is None:
            key = self._relax(call_type) if self.reduce_retracing else call_type
            # 松弛后的键必须仍然接受本次实参，否则回退到精确键
            if not call_type.is_subtype_of(key):
                key = call_type
            cf = self._trace(values, key, call_type)
            self._dispatch_log.append((str(call_type), "TRACE"))
        else:
            self._cache_hits += 1
            self.detector.called_without_tracing()
            self._dispatch_log.append((str(call_type), kind.upper()))
        return cf(*values)

    def get_concrete_function(self, *args, **kwargs):
        """接受 `TensorSpec` 而不是真张量（官方 docstring 的行为）。"""
        values = self._bind(args, kwargs)
        call_type = self._call_type(values)
        cf, _ = self._lookup(call_type)
        if cf is None:
            key = self._relax(call_type) if self.reduce_retracing else call_type
            if not call_type.is_subtype_of(key):
                key = call_type
            cf = self._trace(values, key, call_type)
        return cf

    def experimental_get_tracing_count(self):
        """官方 docstring：Returns the number of times the function has been traced."""
        return len(self._function_cache)

    def pretty_printed_concrete_signatures(self):
        return "\n\n".join(cf.pretty_printed_signature()
                           for cf in self._function_cache.values())

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return _BoundPolymorphicFunction(self, instance)


class _BoundPolymorphicFunction:
    """`@tf.function` 用在方法上时的描述符包装（真实 TF 里由 `tf_method_target` 承担）。"""

    def __init__(self, pf, instance):
        self.pf = pf
        self.instance = instance

    def __call__(self, *args, **kwargs):
        return self.pf(self.instance, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.pf, item)


def tf_function(func=None, **kwargs):
    """`tf.function` 的装饰器形态。"""
    if func is not None:
        return PolymorphicFunction(func)
    return lambda f: PolymorphicFunction(f, **kwargs)


def specs_warning_constants():
    """告警阈值：-- 与 TF 源码常量一致（见 specs）。写成函数以免 specs 循环导入。"""
    try:
        import specs
        return (specs.FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY,
                specs.FREQUENT_TRACING_WARNING_THRESHOLD)
    except Exception:
        return (10, 5)


# ---------------------------------------------------------------------------
# 九、自定义算子、梯度与导出
# ---------------------------------------------------------------------------

_CUSTOM_OPS = {}


def register_custom_op(name, impl):
    """注册一个「一个节点进图」的算子。

    真实 TF 的官方次序是**先试着用现有 Python 算子组合出来**，不行才写 C++ op
    （官方逐字，见 specs.CREATE_OP_RECOMMENDATION）。本模型把两条路的**图大小**摆在一起：
    组合出来的一串算子会逐个进图，注册成单个算子则只进一个节点。
    """
    _CUSTOM_OPS[name] = impl
    return name


def custom_op_call(name, *args):
    ctx = current_context()
    if name not in _CUSTOM_OPS:
        raise FatalError(
            "mini: 算子 %r 没有注册实现（模型里对应「这个 op 在运行时无处可调」）" % (name,))
    impl = _CUSTOM_OPS[name]
    if ctx is None:
        STATS["eager_ops"] += 1
        STATS["eager_dispatch_layers"] += EAGER_LAYERS_PER_OP
        return Tensor(impl(*[_as_array(a) for a in args]))
    out = Tracer(_next_uid(), args[0].dtype, args[0].shape, ctx)
    ctx.graph.add(Op(name + "__custom", [t.uid for t in args], {}, [out.uid]))
    return out


def custom_gradient(fn):
    """`tf.custom_gradient` 的模型。

    官方 docstring 逐字：「Decorator to define a function with a custom gradient.」
    （见 specs.TF_CUSTOM_GRADIENT_DOC）——它给**已有的 Python 组合**换一个梯度函数，
    **不注册新 op**；注册新 op 是 `register_custom_op` 那条路。
    """
    def wrapper(*args):
        return fn(*args)
    wrapper.grad_fn = getattr(fn, "grad_fn", None)
    return wrapper


def xla_compile(concrete_function, unsupported=("tf.print", "tf.numpy_function")):
    """XLA 编译的模型。

    官方 docstring 逐字：If set to `True`, **the whole function needs to be compilable
    by XLA, or an `errors.InvalidArgumentError` is thrown**（见 specs.JIT_COMPILE_DOC）。
    本模型给出一份「编不了」的算子集合，命中就抛 `InvalidArgumentError`。
    ⚠ 这个集合是**本仓库约定的模型**，不是「TF 的 XLA 不支持清单」——
      真实 TF 的 sharp corners 见 docstring 里给的链接。
    """
    names = set(concrete_function.graph.all_op_names())
    bad = sorted(names & set(unsupported))
    if bad:
        raise InvalidArgumentError(
            "mini: the whole function needs to be compilable, but the graph contains %s"
            % ", ".join(bad))
    STATS["xla_compiles"] = STATS.get("xla_compiles", 0) + 1
    concrete_function.xla_compiled = True
    return concrete_function


def xla_experimental_compile(computation, inputs=None):
    """`tf.xla.experimental.compile` 的模型：**仍在，但已废弃**。

    真实 TF 的装饰器逐字：`@deprecated(None, 'xla.experimental.compile is deprecated.
    Consider using `@tf.function(jit_compile=True)`.', warn_once=True)`——第一个参数是
    `None`，**官方没给移除日期**（见 specs.XLA_EXPERIMENTAL_COMPILE_DEPRECATION）。
    """
    import warnings
    try:
        import specs
        msg = specs.XLA_EXPERIMENTAL_COMPILE_MESSAGE
    except Exception:
        msg = "xla.experimental.compile is deprecated."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return computation(*inputs) if inputs is not None else computation()


# ---------------------------------------------------------------------------
# 十、导出：SavedModel 的模型
# ---------------------------------------------------------------------------

SAVED_MODEL_FORMAT_DEFAULT = "tf_saved_model"     # Keras 3 `Model.export()` 的默认值，见 specs


def export_saved_model(pf, path, signatures=None):
    """把已 trace 出来的 `ConcreteFunction` 落盘。

    模型要点：**落盘的是已经 trace 好的图**，不是 Python 函数——所以再载入时**不需要重追踪**。
    """
    signatures = signatures or {"serving_default": list(pf._function_cache.values())[0]}
    payload = {"format": SAVED_MODEL_FORMAT_DEFAULT, "functions": {}}
    for sig_name, cf in signatures.items():
        payload["functions"][sig_name] = {
            "function_type": str(cf.function_type),
            "op_names": cf.graph.all_op_names(),
            "arg_uids": list(cf.atomic_function.arg_uids),
            "arg_positions": list(cf.atomic_function.arg_positions),
            "arg_specs": [[t.dtype, None if t.shape.dims is None else list(t.shape.dims)]
                          for t in cf.function_type.types
                          if isinstance(t, TensorTraceType)],
            "result_uid": cf.atomic_function.result_uid,
            "graph": _serialize_graph(cf.graph),
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _serialize_graph(g):
    return {
        "name": g.name,
        "ops": [{
            "name": op.name,
            "inputs": list(op.inputs),
            "attrs": dict(op.attrs),
            "out_uids": list(op.out_uids),
            "subgraphs": {k: _serialize_graph(v) for k, v in op.subgraphs.items()},
        } for op in g.ops],
    }


def _deserialize_graph(d):
    g = Graph(d["name"])
    for o in d["ops"]:
        g.ops.append(Op(o["name"], o["inputs"], o["attrs"], o["out_uids"],
                        {k: _deserialize_graph(v) for k, v in o["subgraphs"].items()}))
    return g


class LoadedFunction:
    """从磁盘载回来的函数：**调用它不产生任何 tracing**。

    ★ 它也**带着导出时固化的那条签名**：实参的 dtype / shape 必须落在签名允许的范围内，
      否则拒绝——这正是「导出把假设冻住了」的代价。
    """

    def __init__(self, payload, sig_name):
        entry = payload["functions"][sig_name]
        self.sig_name = sig_name
        self.function_type = entry["function_type"]
        self.graph = _deserialize_graph(entry["graph"])
        self.arg_uids = entry["arg_uids"]
        self.arg_positions = entry["arg_positions"]
        self.arg_specs = [TensorSpec(d, TensorShape(None if s is None else s))
                          for d, s in entry["arg_specs"]]
        self.result_uid = entry["result_uid"]
        self.op_names = entry["op_names"]
        self.calls = 0

    def _check(self, values):
        for spec, pos in zip(self.arg_specs, self.arg_positions):
            got = trace_type_of(values[pos])
            if not isinstance(got, TensorTraceType) or not got.is_subtype_of(
                    TensorTraceType(spec.dtype, spec.shape)):
                raise InvalidArgumentError(
                    "mini: 载入的函数只接受 %s，收到 %s（导出把 shape 假设冻住了）"
                    % (spec, got))

    def __call__(self, *values):
        self._check(values)               # 被签名拒掉的调用**不算一次 replay**
        self.calls += 1
        env = {}
        for uid, pos in zip(self.arg_uids, self.arg_positions):
            env[uid] = np.asarray(_as_array(values[pos]))
        _eval_graph(self.graph, env)
        return Tensor(env[self.result_uid])


def load_saved_model(path):
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return SavedModelArtifact(payload)


class SavedModelArtifact:
    def __init__(self, payload):
        self.payload = payload
        self.format = payload["format"]
        self.signatures = {k: LoadedFunction(payload, k) for k in payload["functions"]}


# ---------------------------------------------------------------------------
# 十一、三类断言的记账器（沿用 60-compilers 的形态）
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
        """把几段 §的记账器合并成一个，由它统一出汇总行与退出码。"""
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
    """（注意：不要再定义一个叫 `sub` 的函数——那会遮蔽同名的减法算子。）"""
    print()
    print("--- %s ---" % title)


def header(number, title):
    print("=" * 76)
    print("%s %s" % (number, title))
    print("=" * 76)
    print("★ 本程序跑的是**本仓库的迷你 TF 追踪器**，不是 TensorFlow。")
    print("  下面每个数字都只支撑关于**机制**的结论，不支撑任何关于真实 TF 的结论。")
    print("  迷你实现只用 python3 %s + numpy %s。"
          % (sys.version.split()[0], np.__version__))
