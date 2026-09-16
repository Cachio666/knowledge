"""迷你 HLO：形状与 shape inference、numpy 解释器、三类断言框架。

★ 本模块是**本仓库自己的几百行实现**，不是 XLA。它借用了 XLA 的术语
  （HLO / parameter / dot_general / broadcast_dimensions / FusionKind / kCopy …）
  以便与真实系统对照，**结构与判据都是我们自己简化的**。
  XLA 的真实行为只出现在 `specs.py` 与文档里，逐条带一手来源，
  **不作为任何程序的断言对象**（§0.3 规矩 4）。

★ shape inference 是 HLO 这一层最容易出错的地方：同一个 `dot_general` 在
  HLO proto、HLO 文本格式、StableHLO op attribute 三处用的是**三套不同的字段名**，
  而 batch 维的语义在 `jnp.dot` 与 `jnp.matmul` 之间又不一样。本模块把这两件事
  都**显式化**，好让编号程序能拿 numpy 当独立参照物来对拍。

对外的三个概念（与 HLO 一致）：
  Op      —— 一条 HLO 指令，**恰好产生一个值**（HLO 是 SSA，每个指令一个结果）
  Module  —— 一个计算（本迷你版只有一个 entry computation）
  Shape   —— `f32[3,2,4]` 这样的形状（本迷你版不建模 layout，layout 在 minilayout.py）
"""

import sys

import numpy as np

# ---------------------------------------------------------------------------
# 错误类型：shape inference 的失败必须是**可捕获的异常**，否则 C 类断言写不出来
# ---------------------------------------------------------------------------


class HloError(Exception):
    """迷你 HLO 的基类错误。"""


class ShapeInferenceError(HloError):
    """形状推断失败。"""


class DimNumberError(ShapeInferenceError):
    """维度编号越界或重复（对应 StableHLO 的 C3–C8）。"""


class DimSizeError(ShapeInferenceError):
    """对应维度的大小不相等（对应 StableHLO 的 C9 / C10）。"""


class BroadcastError(ShapeInferenceError):
    """broadcast_dimensions 与操作数形状不兼容。"""


class ExecutionError(HloError):
    """解释器执行失败。"""


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------

BITS = {"f32": 32, "f16": 16, "bf16": 16, "s32": 32, "s8": 8, "pred": 8, "f64": 64}
NP_DT = {"f32": np.float32, "f16": np.float16, "bf16": np.float32,
         "s32": np.int32, "s8": np.int8, "pred": np.bool_, "f64": np.float64}


class Shape:
    """一个 HLO 形状：维度列表 + 元素类型。"""

    __slots__ = ("dims", "dtype")

    def __init__(self, dims, dtype="f32"):
        self.dims = tuple(int(d) for d in dims)
        self.dtype = dtype

    @property
    def rank(self):
        return len(self.dims)

    @property
    def num_elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    def nbytes(self):
        return self.num_elements * BITS[self.dtype] // 8

    def np_dtype(self):
        return NP_DT[self.dtype]

    def __eq__(self, other):
        return (isinstance(other, Shape) and self.dims == other.dims
                and self.dtype == other.dtype)

    def __hash__(self):
        return hash((self.dims, self.dtype))

    def __repr__(self):
        return "%s[%s]" % (self.dtype, ",".join(str(d) for d in self.dims))


# ---------------------------------------------------------------------------
# Op / Module
# ---------------------------------------------------------------------------


class Op:
    """一条 HLO 指令。**恰好一个结果值**，值就是它自己。"""

    def __init__(self, name, operands=(), shape=None, attrs=None, index=0,
                 opcode=None):
        self.name = name                 # 如 "dot_general.2"
        self.opcode = opcode or name.split(".")[0]  # 如 "dot_general"
        self.operands = list(operands)
        self.shape = shape
        self.attrs = dict(attrs or {})
        self.index = index
        self.users = []
        for o in self.operands:
            o.users.append(self)

    @property
    def num_operands(self):
        return len(self.operands)

    def __repr__(self):
        return "%%%s" % self.name


class Module:
    """一个迷你 HLO 模块：参数 + 一串按序排列的指令 + root。"""

    def __init__(self, name="mini_module"):
        self.name = name
        self.params = []
        self.ops = []
        self._counter = {}

    def add(self, op):
        self.ops.append(op)
        return op

    def parameter(self, shape, name=None):
        idx = len(self.params)
        nm = name or ("p%d" % idx)
        op = Op(nm, operands=[], shape=shape, attrs={"parameter_number": idx},
                index=-1 - idx, opcode="parameter")
        self.params.append(op)
        return op

    def emit(self, opcode, operands, shape=None, **attrs):
        n = self._counter.get(opcode, 0)
        self._counter[opcode] = n + 1
        return self.add(Op("%s.%d" % (opcode, n), operands, shape, attrs,
                           index=len(self.ops)))

    def root(self):
        return self.ops[-1] if self.ops else None

    def by_name(self, name):
        for op in self.params + self.ops:
            if op.name == name:
                return op
        raise HloError("没有名为 %s 的 op" % name)

    def __iter__(self):
        return iter(self.ops)

    def text(self):
        """打印成**模仿 HLO 文本格式**的样子（本仓库自己的简化格式）。"""
        lines = ["HloModule %s" % self.name, ""]
        sig = ", ".join("%%%s: %s" % (p.name, p.shape) for p in self.params)
        lines.append("ENTRY %%main (%s) -> %s {" % (sig, self.root().shape))
        for p in self.params:
            lines.append("  %%%s = %s parameter(%d)"
                         % (p.name, p.shape, p.attrs["parameter_number"]))
        for i, op in enumerate(self.ops):
            root = "ROOT " if i == len(self.ops) - 1 else ""
            ops = ", ".join("%%%s" % o.name for o in op.operands)
            lines.append("  %s%%%s = %s %s(%s)%s"
                         % (root, op.name, op.shape, op.opcode, ops,
                            _attr_suffix(op)))
        lines.append("}")
        return "\n".join(lines)


def _attr_suffix(op):
    """把 attrs 打印成 HLO 那样跟在指令后面的 `, key={v}`。"""
    if not op.attrs:
        return ""
    skip = {"parameter_number", "value"}
    parts = []
    for k in sorted(op.attrs):
        if k in skip:
            continue
        v = op.attrs[k]
        if isinstance(v, (list, tuple)):
            parts.append("%s={%s}" % (k, ",".join(str(x) for x in v)))
        else:
            parts.append("%s=%s" % (k, v))
    return (", " + ", ".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# 语义维度的**三套命名**（文档事实；程序里只用其中一套，见编号程序的说明）
#
# 三套写法对照（来源见 specs.py）：
#   HLO/StableHLO proto      : lhs_contracting_dimensions / lhs_batch_dimensions
#   HLO 文本格式             : lhs_contracting_dims       / lhs_batch_dims
#   StableHLO op attribute   : lhs_contracting_dimensions / lhs_batching_dimensions
#
# 本迷你实现**统一采用 HLO 文本格式那一套**（`*_dims`），并在 `dot_general_attrs()`
# 里提供从另外两套改名进来的辅助函数——**改名只是名字，语义完全一样**：
# 维度编号都是**该操作数自身 shape 的 0-based 下标**。
# ---------------------------------------------------------------------------

DIM_NAME_STYLES = {
    "hlo_proto": ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
                  "lhs_batch_dimensions", "rhs_batch_dimensions"),
    "hlo_text": ("lhs_contracting_dims", "rhs_contracting_dims",
                 "lhs_batch_dims", "rhs_batch_dims"),
    "stablehlo": ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
                  "lhs_batching_dimensions", "rhs_batching_dimensions"),
}


def rename_dims(attrs, src_style, dst_style="hlo_text"):
    """把一套命名下的 dot_general attrs 改写成另一套命名。

    **纯改名**：值不变。存在的意义是让「三套写法指的是同一件事」在代码里可见，
    并让读者能一眼看到 `batch` 与 `batching` 的差别。
    """
    src = DIM_NAME_STYLES[src_style]
    dst = DIM_NAME_STYLES[dst_style]
    return {dst[i]: attrs[src[i]] for i in range(4) if src[i] in attrs}


# ---------------------------------------------------------------------------
# shape inference
# ---------------------------------------------------------------------------


def _check_dims(dims, rank, who):
    """StableHLO 的 C3/C5/C6/C7/C8：编号在 [0, rank) 内且互不重复。"""
    for d in dims:
        if not (0 <= d < rank):
            raise DimNumberError("%s 的维度编号 %d 越界（rank=%d）" % (who, d, rank))
    if len(set(dims)) != len(dims):
        raise DimNumberError("%s 的维度编号有重复：%s" % (who, list(dims)))


def _check_unique_together(b, c, who):
    """StableHLO 的 C3/C4：batching 与 contracting 在同一操作数内不得重复。"""
    if set(b) & set(c):
        raise DimNumberError(
            "%s 的 batching 与 contracting 维有重叠：%s ∩ %s"
            % (who, sorted(b), sorted(c)))


def infer_broadcast_in_dim(operand_shape, bd, out_shape, who="binary"):
    """HLO 的 `broadcast_dimensions`：把 operand 的第 i 维放到结果的 bd[i] 维。

    ★ 本迷你版的一条自有约定：`broadcast_dimensions` **为空**时表示
      「两个操作数形状完全相同，不做任何广播」（HLO 文本里同形状的二元运算就是这么印的）。
    返回 operand 在结果 rank 下的形状（大小为 1 的维度保持 1，交给 numpy 广播）。
    """
    bd = list(bd)
    if not bd:
        if operand_shape != out_shape:
            raise BroadcastError(
                "%s：未给 broadcast_dimensions 时，两个操作数形状必须完全相同"
                "（%s vs %s）" % (who, operand_shape, out_shape))
        return Shape(out_shape.dims, operand_shape.dtype)
    if operand_shape.rank != len(bd):
        raise BroadcastError(
            "%s：operand rank %d 与 broadcast_dimensions 长度 %d 不符"
            % (who, operand_shape.rank, len(bd)))
    if list(bd) != sorted(bd) or len(set(bd)) != len(bd):
        raise BroadcastError("%s：broadcast_dimensions 必须严格递增且不重复：%s"
                             % (who, list(bd)))
    out_rank = out_shape.rank
    _check_dims(bd, out_rank, who + " 的 broadcast_dimensions")
    packed = [1] * out_rank
    for i, d in enumerate(bd):
        sd = operand_shape.dims[i]
        if sd != 1 and sd != out_shape.dims[d]:
            raise BroadcastError(
                "%s：operand 的维 %d（大小 %d）与结果第 %d 维（大小 %d）不兼容"
                % (who, i, sd, d, out_shape.dims[d]))
        packed[d] = sd
    return Shape(packed, operand_shape.dtype)


def infer_broadcast(operand_shape, sizes, dimensions):
    """HLO 的 `broadcast`：结果 = sizes ++ (operand 的各维放到 dimensions 位置上)。"""
    out_rank = len(sizes) + len(dimensions)
    if operand_shape.rank != len(dimensions):
        raise BroadcastError(
            "broadcast：operand rank %d 与 dimensions 长度 %d 不符"
            % (operand_shape.rank, len(dimensions)))
    _check_dims(dimensions, out_rank, "broadcast 的 dimensions")
    dims = [0] * out_rank
    for i, s in enumerate(sizes):
        dims[i] = int(s)
    for i, d in enumerate(dimensions):
        if d < len(sizes):
            raise BroadcastError(
                "broadcast：dimensions 指向了前 %d 维（那几维由 sizes 决定）" % len(sizes))
        dims[d] = operand_shape.dims[i]
    return Shape(dims, operand_shape.dtype)


def infer_dot_general(lhs, rhs, attrs):
    """`dot_general` 的结果形状。

    这是 StableHLO spec 里 C12 的直接实现：
      shape(result) = dim(lhs, lhs_batching) + dim(lhs, lhs_result)
                      + dim(rhs, rhs_result)
    其中 lhs_result = lhs 里既不是 batching 也不是 contracting 的维（**保持原顺序**）。
    """
    lb = list(attrs.get("lhs_batch_dims", []))
    rb = list(attrs.get("rhs_batch_dims", []))
    lc = list(attrs.get("lhs_contracting_dims", []))
    rc = list(attrs.get("rhs_contracting_dims", []))

    if len(lb) != len(rb):                                   # C1
        raise DimSizeError("dot_general：两侧 batch 维个数不等（%d vs %d）"
                           % (len(lb), len(rb)))
    if len(lc) != len(rc):                                   # C2
        raise DimSizeError("dot_general：两侧 contracting 维个数不等（%d vs %d）"
                           % (len(lc), len(rc)))
    _check_dims(lb, lhs.rank, "lhs 的 batch")                # C5
    _check_dims(lc, lhs.rank, "lhs 的 contracting")          # C6
    _check_dims(rb, rhs.rank, "rhs 的 batch")                # C7
    _check_dims(rc, rhs.rank, "rhs 的 contracting")          # C8
    _check_unique_together(lb, lc, "lhs")                    # C3
    _check_unique_together(rb, rc, "rhs")                    # C4

    for i in range(len(lb)):                                 # C9
        if lhs.dims[lb[i]] != rhs.dims[rb[i]]:
            raise DimSizeError(
                "dot_general：第 %d 个 batch 维大小不等（lhs[%d]=%d vs rhs[%d]=%d）"
                % (i, lb[i], lhs.dims[lb[i]], rb[i], rhs.dims[rb[i]]))
    for i in range(len(lc)):                                 # C10
        if lhs.dims[lc[i]] != rhs.dims[rc[i]]:
            raise DimSizeError(
                "dot_general：第 %d 个 contracting 维大小不等（lhs[%d]=%d vs rhs[%d]=%d）"
                % (i, lc[i], lhs.dims[lc[i]], rc[i], rhs.dims[rc[i]]))

    lhs_free = [d for d in range(lhs.rank) if d not in lb and d not in lc]
    rhs_free = [d for d in range(rhs.rank) if d not in rb and d not in rc]
    dims = ([lhs.dims[d] for d in lb] + [lhs.dims[d] for d in lhs_free]
            + [rhs.dims[d] for d in rhs_free])
    if lhs.dtype != rhs.dtype:
        raise ShapeInferenceError("dot_general：两侧元素类型不同（%s vs %s）"
                                  % (lhs.dtype, rhs.dtype))
    return Shape(dims, lhs.dtype)


def infer_shape(op):
    """按 opcode 分派。**所有**形状推断都从这里走，没有第二条路径。"""
    opshapes = [o.shape for o in op.operands]
    a = op.attrs
    if op.opcode == "parameter":
        return op.shape
    if op.opcode == "constant":
        return op.shape
    if op.opcode in ("add", "subtract", "multiply", "divide", "maximum", "minimum"):
        lhs, rhs = opshapes
        if lhs.rank < rhs.rank:
            raise BroadcastError("%s：HLO 要求 lhs 的 rank >= rhs 的 rank" % op.opcode)
        infer_broadcast_in_dim(rhs, a.get("broadcast_dimensions", []), lhs, op.opcode)
        return Shape(lhs.dims, lhs.dtype)
    if op.opcode == "broadcast":
        return infer_broadcast(opshapes[0], a.get("sizes", []),
                               a.get("dimensions", []))
    if op.opcode == "dot_general":
        return infer_dot_general(opshapes[0], opshapes[1], a)
    if op.opcode == "transpose":
        p = list(a["permutation"])
        _check_dims(p, opshapes[0].rank, "transpose 的 permutation")
        return Shape([opshapes[0].dims[d] for d in p], opshapes[0].dtype)
    if op.opcode == "reshape":
        new = Shape(a["dims"], opshapes[0].dtype)
        if new.num_elements != opshapes[0].num_elements:
            raise ShapeInferenceError(
                "reshape：元素数不等（%d -> %d）"
                % (opshapes[0].num_elements, new.num_elements))
        return new
    if op.opcode == "reduce":
        d = list(a["dimensions"])
        _check_dims(d, opshapes[0].rank, "reduce 的 dimensions")
        if len(d) != 1:
            raise ShapeInferenceError("本迷你实现只支持单维 reduce")
        return Shape([x for i, x in enumerate(opshapes[0].dims) if i not in d],
                     opshapes[0].dtype)
    raise ShapeInferenceError("未知 opcode：%s" % op.opcode)


def verify(module):
    """对全模块做一遍 shape inference，返回失败信息列表（空 = 通过）。"""
    problems = []
    for op in list(module.params) + list(module.ops):
        try:
            got = infer_shape(op)
        except HloError as e:
            problems.append("%s：%s" % (op.name, e))
            continue
        if op.shape is not None and got != op.shape:
            problems.append("%s：声明 %s，推断 %s" % (op.name, op.shape, got))
    return problems


def verify_or_raise(module):
    """与 `verify` 同样的检查，但有问题时**抛原始异常**——C 类断言要用这个形态。

    ★ 不把异常重新包成基类：C 类断言要能区分 `BroadcastError` 与 `DimNumberError`，
      否则「断言抛出的是哪一类」就写不出来了（这是第一版跑出来的教训）。
    """
    for op in list(module.params) + list(module.ops):
        got = infer_shape(op)          # 原始异常直接抛出
        if op.shape is not None and got != op.shape:
            raise ShapeInferenceError("%s：声明 %s，推断 %s" % (op.name, op.shape, got))
    return True


# ---------------------------------------------------------------------------
# numpy 解释器 —— 语义参照物，不是 shape 推断的第二条路径
# ---------------------------------------------------------------------------


def _as_np(shape):
    return np.zeros(shape.dims, dtype=shape.np_dtype())


def _broadcast_operand(arr, bd, out_rank):
    packed = [1] * out_rank
    for i, d in enumerate(bd):
        packed[d] = arr.shape[i]
    return arr.reshape(packed)


def eval_op(op, env, feed=None):
    """求**单条** op 的值。`env` 是 {op.name: ndarray}，必须已含它的操作数。

    拆成单条是为了让融合后的**分组执行**（minifuse.execute_plan）能复用同一套
    逐算子语义——分组执行与整体执行走的是**同一条算式**，但**分组顺序是另一条路径**。
    """
    feed = feed or {}
    a = op.attrs
    src = [env[o.name] for o in op.operands]
    if op.opcode == "parameter":
        if op.name not in feed:
            raise ExecutionError("参数 %s 没有喂值" % op.name)
        return np.asarray(feed[op.name], dtype=op.shape.np_dtype())
    if op.opcode == "constant":
        return np.asarray(a["value"], dtype=op.shape.np_dtype())
    if op.opcode in ("add", "subtract", "multiply", "divide", "maximum", "minimum"):
        lhs, rhs = src
        bd = list(a.get("broadcast_dimensions", []))
        if bd:
            rhs = _broadcast_operand(rhs, bd, lhs.ndim)
        # bd 为空 = 两个操作数同形状，直接用
        fn = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
              "divide": np.divide, "maximum": np.maximum,
              "minimum": np.minimum}[op.opcode]
        return fn(lhs, rhs)
    if op.opcode == "broadcast":
        x = src[0]
        sizes = [int(s) for s in a.get("sizes", [])]
        dims = list(a.get("dimensions", []))
        out_rank = len(sizes) + len(dims)
        target = [1] * out_rank
        for i, s in enumerate(sizes):
            target[i] = s
        for i, d in enumerate(dims):
            target[d] = x.shape[i]
        return np.broadcast_to(_broadcast_operand(x, dims, out_rank),
                               tuple(target)).copy()
    if op.opcode == "dot_general":
        return _exec_dot_general(src[0], src[1], a)
    if op.opcode == "transpose":
        return np.transpose(src[0], a["permutation"])
    if op.opcode == "reshape":
        return np.reshape(src[0], a["dims"])
    if op.opcode == "reduce":
        return np.sum(src[0], axis=tuple(a["dimensions"]))
    raise ExecutionError("未知 opcode：%s" % op.opcode)


def execute(module, feed=None):
    """按序执行，返回 {op.name: ndarray}。参数先入 env，再按序跑 module.ops。"""
    feed = feed or {}
    env = {p.name: eval_op(p, {}, feed) for p in module.params}
    for op in module.ops:
        env[op.name] = eval_op(op, env, feed)
    return env


def _exec_dot_general(lhs, rhs, attrs):
    """把 dot_general 化成一批 batched matmul —— 与 StableHLO spec 的参考算法同构。

    步骤：转置成 [batch..., contracting..., free...] → reshape 成 3D → matmul → reshape。
    **这条路径与 `infer_dot_general` 的索引算术是分开写的**，所以两者互为旁证。
    """
    lb = list(attrs.get("lhs_batch_dims", []))
    rb = list(attrs.get("rhs_batch_dims", []))
    lc = list(attrs.get("lhs_contracting_dims", []))
    rc = list(attrs.get("rhs_contracting_dims", []))
    lhs_free = [d for d in range(lhs.ndim) if d not in lb and d not in lc]
    rhs_free = [d for d in range(rhs.ndim) if d not in rb and d not in rc]

    # ★ 两边的转置顺序**不一样**，这是因为 reshape 按 C 序展开：
    #   lhs 要摊成 (M, K)，K 必须最**快**变 → 转置成 [batch, free, contracting]
    #   rhs 要摊成 (K, N)，N 必须最**快**变 → 转置成 [batch, contracting, free]
    l = np.transpose(lhs, lb + lhs_free + lc)
    r = np.transpose(rhs, rb + rc + rhs_free)
    b = int(np.prod([lhs.shape[d] for d in lb], dtype=np.int64)) if lb else 1
    k = int(np.prod([lhs.shape[d] for d in lc], dtype=np.int64)) if lc else 1
    m = int(np.prod([lhs.shape[d] for d in lhs_free], dtype=np.int64)) if lhs_free else 1
    n = int(np.prod([rhs.shape[d] for d in rhs_free], dtype=np.int64)) if rhs_free else 1
    out = np.matmul(np.ascontiguousarray(l).reshape(b, m, k),
                    np.ascontiguousarray(r).reshape(b, k, n))
    batch_dims = [lhs.shape[d] for d in lb]
    free_dims = ([lhs.shape[d] for d in lhs_free] + [rhs.shape[d] for d in rhs_free])
    return out.reshape(tuple(batch_dims + free_dims))


# ---------------------------------------------------------------------------
# 三类断言的记账器（与仓库其它主题同一形态）
# ---------------------------------------------------------------------------

A_NAME = "A 算法正确性"
B_NAME = "B 代价模型"
C_NAME = "C 边界与反例"


class Checks:
    """A / B / C 三类断言的记账器。失败累计，`summary()` 以非零码退出。"""

    def __init__(self, title=""):
        self.title = title
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
        """跑 fn，断言它抛出 exc。**没抛 = 这条 C 类断言失败。**"""
        try:
            fn()
        except exc as e:
            return self._record("C", True, msg,
                                detail or ("抛出 %s: %s" % (type(e).__name__, e)))
        except Exception as e:                                   # noqa: BLE001
            return self._record("C", False, msg,
                                detail or ("抛了非预期的 %s: %s"
                                           % (type(e).__name__, e)))
        return self._record("C", False, msg,
                            detail or "**没有抛异常**——失败行为没发生")

    def expect_ok(self, fn, msg, detail=""):
        """断言 fn 能跑通。给 C 类配一个「换成正确做法就通过」的对照（记 A 类）。"""
        try:
            r = fn()
        except Exception as e:                                   # noqa: BLE001
            return self._record("A", False, msg,
                                "抛了 %s: %s" % (type(e).__name__, e))
        return self._record("A", True, msg, detail if detail else str(r))

    def summary(self):
        total = sum(self.counts.values())
        print()
        print("-" * 78)
        if self.failures:
            for cls, msg in self.failures:
                print("  失败项 [%s] %s" % (cls, msg))
            print("断言汇总：通过 %d 条断言，失败 %d 条（共 %d 条）"
                  % (total - len(self.failures), len(self.failures), total))
            sys.exit(1)
        print("通过 %d 条断言（%s %d / %s %d / %s %d）"
              % (total, A_NAME, self.counts["A"], B_NAME, self.counts["B"],
                 C_NAME, self.counts["C"]))
        return 0


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
