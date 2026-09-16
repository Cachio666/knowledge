"""迷你图捕获骨架：图 / 代理张量 / 帧入口追踪器 / 解释器 / 三类断言框架。

★ 本模块是**本仓库的迷你实现**，不是 TorchDynamo、不是 PyTorch Inductor。
  它借用了真实系统的术语（proxy、graph break、guard、fusion group）以便对照，
  但结构是为教学重写的：**任何输出数字都只支持关于「机制」的结论**
  （见 60-compilers/inductor/*.md §5 的强制限定语）。

★ 关于捕获机制（对应 01 号文档的 §1）
  探针用 Python 标准库 `sys.settrace` 的 **`call` 事件**做「帧入口决策」：
  每进入一个函数帧，钩子被调用一次，由它决定这个帧是否进入捕获。
  真实的 TorchDynamo **不是这样做的**——它装的是 CPython 的 **PEP 523 帧求值回调**
  （`_PyInterpreterState_SetEvalFrameFunc`，见 specs.PEP523_SYMBOL）。
  两者在「每个帧入口做一次决策」这一点上同构，**触发时机与实现层次都不同**。
  `sys.monitoring` 不是 Dynamo 的机制，且与 torch.compile 已知不兼容
  （见 specs.SYS_MONITORING）。

★ 关于 graph break（对应 01 号文档的 §4）
  迷你追踪器只认两类切图触发条件，**都与真实系统官方列举的一致**：
    1. **数据依赖控制流**：对代理调用 `.item()`；
    2. **print / logging**：对代理调用 `mprint()`，除非它被登记进
       `reorderable_logging_functions`（登记后改为「重排」——记录成一个 effect
       节点，**不切图**）。
  **原地修改（mutation）不是触发条件**：`P.add_()` / `P.copy_()` 会被正常记录成节点。

★ 关于「重放」（对应 01 号文档的 §3）
  真实的 Dynamo 在切图点生成一个 **resume function**，从断点继续捕获。
  本仓库没有字节码层，用**重放**模拟：切图后重跑整个函数，前 k 个算子走 eager、
  从第 k+1 个算子起重新开始记录。这样「被切出的区域跑普通 Python、前后子图仍被编译」
  这件事在迷你实现里也是真的（子图数 > 1，且中间那段确实以普通 Python 执行）。
"""

import sys

import numpy as np

A_NAME, B_NAME, C_NAME = "A 算法正确性", "B 代价模型", "C 边界与反例"
ITEMSIZE = np.dtype(np.float32).itemsize


# ---------------------------------------------------------------------------
# 三类断言的记账器（与 40-kernels 的 verify.py / mlir / tvm 同一形态）
# ---------------------------------------------------------------------------


class Checks:
    """A 算法正确性 / B 代价模型 / C 边界与反例。失败累计，`summary()` 非零退出。"""

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
        """跑 fn，断言它抛出 exc。**没抛 = 这条 C 类断言失败。**"""
        try:
            fn()
        except exc as e:
            return self._record("C", True, msg,
                                detail or ("抛出 %s: %s" % (type(e).__name__, e)))
        except Exception as e:                                   # noqa: BLE001
            return self._record("C", False, msg, detail or
                                ("抛了非预期的 %s: %s" % (type(e).__name__, e)))
        return self._record("C", False, msg, detail or "**没有出问题**——失败行为没发生")

    def expect_ok(self, fn, msg, detail=""):
        """断言 fn 能跑通。给 C 类配一个「换成正确做法就通过」的对照。"""
        try:
            r = fn()
        except Exception as e:                                   # noqa: BLE001
            return self._record("A", False, msg, "抛了 %s: %s" % (type(e).__name__, e))
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


# ---------------------------------------------------------------------------
# 图的数据结构
# ---------------------------------------------------------------------------


class Dep:
    """一次缓冲访问：`(缓冲名, 索引表达式)`。

    索引表达式是本仓库的简化模型：按被访问缓冲的维度逐个给出该维依赖「访问者循环」
    的哪一个，用 `@k` 表示访问者的第 k 个循环（**按访问者的循环序**编号）。
    两个不同节点只要访问模式一致，表达式就逐字相等。

    真实 Inductor 用的是 `MemoryDep(name, index, size, mode)`，索引是 sympy 表达式，
    比较前还要做循环归一化；本仓库把它降成一个字符串元组，**不能声称等价**。
    """

    __slots__ = ("buf", "index")

    def __init__(self, buf, index):
        self.buf = buf
        self.index = tuple(index)

    def __eq__(self, other):
        return isinstance(other, Dep) and self.buf == other.buf \
            and self.index == other.index

    def __hash__(self):
        return hash((self.buf, self.index))

    def __repr__(self):
        return "%s[%s]" % (self.buf, ",".join(self.index))


class Node:
    """图里的一个算子。"""

    __slots__ = ("op", "inputs", "shape", "kind", "reads", "writes",
                 "rloops", "loop_order", "buf", "value", "attrs")

    def __init__(self, op, inputs, shape, kind, reads=(), writes=None,
                 rloops=0, loop_order=None, attrs=None):
        self.op = op
        self.inputs = list(inputs)          # list[Node]
        self.shape = tuple(shape)
        self.kind = kind                    # placeholder|pointwise|reduction|extern|effect
        self.reads = list(reads)            # list[Dep]
        self.writes = writes                # Dep | None
        self.rloops = rloops
        # loop_order：本节点的循环序，"P"=并行、"R"=归约。
        # sum(axis=1) 得到 "PR"（并行在外），sum(axis=0) 得到 "RP"（归约在外）。
        self.loop_order = loop_order if loop_order is not None \
            else "P" * (len(self.shape) + rloops)
        self.buf = None
        self.value = None
        self.attrs = attrs or {}

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def numbytes(self):
        return self.numel * ITEMSIZE

    @property
    def red_numel(self):
        n = 1
        for d in self.attrs.get("red_shape", ()):
            n *= d
        return n or 1

    def readers_of(self, buf):
        return [d for d in self.reads if d.buf == buf]

    def __repr__(self):
        return "%s(%s) -> %s" % (self.op, ",".join(i.op for i in self.inputs),
                                 "x".join(map(str, self.shape)))


class Graph:
    """一张被捕获的子图：节点按拓扑序排列。"""

    def __init__(self, name="g0"):
        self.name = name
        self.nodes = []
        self.effects = []                   # 被「重排」出来的 print/logging
        self.outputs = []                   # list[Node]
        self._n_buf = 0

    def add(self, op, inputs, shape, kind, reads=(), rloops=0, loop_order=None,
            attrs=None):
        node = Node(op, inputs, shape, kind, reads=reads, rloops=rloops,
                    loop_order=loop_order, attrs=attrs)
        node.buf = "buf%d" % self._n_buf
        self._n_buf += 1
        self.nodes.append(node)
        return node

    def placeholder(self, shape, hint="arg"):
        node = Node(hint, [], shape, "placeholder")
        node.buf = "%s%d" % (hint, len([n for n in self.nodes
                                        if n.kind == "placeholder"]))
        self.nodes.append(node)
        return node

    def add_effect(self, name, args):
        """登记过的 print/logging：**重排**到计算之外，不参与图与 codegen。"""
        node = Node(name, [], (), "effect", attrs={"args": args})
        self.effects.append(node)
        return node

    def by_buf(self, name):
        for n in self.nodes:
            if n.buf == name:
                return n
        return None

    def __repr__(self):
        return "Graph(%s, %d 节点)" % (self.name, len(self.nodes))


# ---------------------------------------------------------------------------
# 代理张量：算子重载即记录
# ---------------------------------------------------------------------------


def _par_index(rank):
    """一张秩 rank 的输出缓冲，写访问按访问者循环序逐维展开。"""
    return tuple("@%d" % i for i in range(rank))


def _read_index(in_rank, n_par):
    """输入（秩 in_rank）被访问时各维依赖访问者的哪个循环——**右对齐**。

    右对齐是元素级算子的语义：形状 (n,) 的 bias 加进形状 (m,n) 的张量时，
    bias 的第 0 维依赖访问者的**最后一个**并行循环。
    """
    out = []
    for j in range(in_rank):
        pos = n_par - in_rank + j
        out.append("@%d" % pos if pos >= 0
                   else "@b%d" % j)      # operand 自带的高阶循环（比输出秩还高）
    return tuple(out)


class GraphBreak(Exception):
    """切图信号。携带原因与「断在第几个算子上」。"""

    def __init__(self, reason, op_index):
        Exception.__init__(self, "%s（第 %d 个算子）" % (reason, op_index))
        self.reason = reason
        self.op_index = op_index


class _TraceState:
    def __init__(self):
        self.recording = False
        self.counter = 0
        self.eager_prefix = 0
        self.graph = None
        self.frame_entries = []
        self.skipped_frames = 0
        self.reorderable_logging = set()


_S = _TraceState()


def _emit(op, kind, args, value, rloops=0, red_input=False, transposed=False,
          attrs=None, loop_order=None):
    """记录一个算子（或在不记录时直接透传）。`args` 是 P 或 None。

    ★ 本仓库的简化：**代理始终携带 eager 的 numpy 值**，所以「记录」与「求值」
    在同一次执行里完成——真实 Dynamo 也是边追踪边求值（fake tensor + 实际执行）。

    ★ 断点处「存活的中间值」在这里被补成占位符：切图后重放时，前缀跑的是普通
    Python，产出的代理没有节点；当捕获在断点之后重新开始时，第一个消费它们的算子
    会为它们建一个占位符——这正是真实系统里 resume function 的入参。
    """
    _S.counter += 1
    recording = _S.recording and _S.counter > _S.eager_prefix
    if not recording:
        return P(value, None)
    nodes = []
    for a in args:
        if a is None:
            continue
        if a._node is None:                       # 断点处存活下来的中间值
            ph = _S.graph.placeholder(a.shape, "mid")
            ph.value = a._val
            a._node = ph
        nodes.append(a._node)
    k = len(value.shape)
    reads = []
    for nd in nodes:
        if red_input:
            idx = tuple("@%d" % j for j in range(len(nd.shape)))
        elif transposed:
            idx = tuple(reversed(_read_index(len(nd.shape), k)))
        else:
            idx = _read_index(len(nd.shape), k)
        reads.append(Dep(nd.buf, idx))
    node = _S.graph.add(op, nodes, value.shape, kind, reads=reads, rloops=rloops,
                        loop_order=loop_order, attrs=attrs)
    node.value = value
    node.writes = Dep(node.buf, _par_index(len(value.shape)))
    return P(value, node)


class P:
    """迷你代理张量。带 eager 值 + 记录到的节点（None 表示未记录）。"""

    __slots__ = ("_val", "_node")

    def __init__(self, val, node):
        self._val = val
        self._node = node

    @property
    def shape(self):
        return self._val.shape

    @property
    def node(self):
        return self._node

    def numpy(self):
        return self._val

    def __repr__(self):
        return "P%s" % (self._val.shape,)

    # ---- 元素级算子 ----

    def _bin(self, other, name, fn):
        ov = other._val if isinstance(other, P) else other
        ins = [self, other if isinstance(other, P) else None]
        # 标量常量不是缓冲：它进节点属性，不进图输入（真实 Inductor 同此——
        # 常量被写进 kernel 源码，`shared_data_score` 也不把它算作共享依赖）。
        attrs = None
        if not isinstance(other, P):
            attrs = {"const": float(np.asarray(ov).reshape(-1)[0])}
        return _emit(name, "pointwise", ins, fn(self._val, ov), attrs=attrs)

    def __add__(self, o):
        return self._bin(o, "add", lambda a, b: a + b)

    def __radd__(self, o):
        return self._bin(o, "add", lambda a, b: b + a)

    def __mul__(self, o):
        return self._bin(o, "mul", lambda a, b: a * b)

    def __rmul__(self, o):
        return self._bin(o, "mul", lambda a, b: b * a)

    def __sub__(self, o):
        return self._bin(o, "sub", lambda a, b: a - b)

    def __truediv__(self, o):
        return self._bin(o, "div", lambda a, b: a / b)

    def __neg__(self):
        return _emit("neg", "pointwise", [self], -self._val)

    def relu(self):
        return _emit("relu", "pointwise", [self], np.maximum(self._val, 0.0))

    def exp(self):
        return _emit("exp", "pointwise", [self], np.exp(self._val))

    def sqrt(self):
        return _emit("sqrt", "pointwise", [self], np.sqrt(self._val))

    # ---- 形状类算子：索引不再逐字相等 ----

    def transpose(self):
        return _emit("transpose", "pointwise", [self],
                     np.ascontiguousarray(self._val.T), transposed=True)

    # ---- 归约 ----

    def sum(self, axis=None, order=None):
        """按维归约。`axis=None` 归约到标量。`order` 可显式给出循环序。

        循环序默认按 axis 推：归约的是**最后一维** → "PR"（并行循环在外）；
        归约的是**第一维** → "RP"（归约循环在外）。这两个正是 mix-order 的镜像对。
        """
        if axis is None:
            axis_t = tuple(range(len(self._val.shape)))
            order = order or "R" * len(self._val.shape)
        else:
            axis_t = (axis,) if isinstance(axis, int) else tuple(axis)
            if order is None:
                order = "PR" if axis_t[-1] == len(self._val.shape) - 1 else "RP"
        v = self._val.sum(axis=axis_t)
        red_shape = tuple(self._val.shape[a] for a in axis_t)
        return _emit("sum", "reduction", [self], v, rloops=len(axis_t),
                     red_input=True, loop_order=order,
                     attrs={"axis": None if axis is None else axis_t,
                            "red_shape": red_shape})

    # ---- 外部库算子（默认不搜的那一类）----

    def matmul(self, other):
        """外部库算子（GEMM：默认**不**参与模板搜索的那一类）。"""
        if not isinstance(other, P):
            raise TypeError("迷你实现的 matmul 要求右操作数也是代理张量"
                            "（这样它才会进图；常量矩阵不是缓冲）")
        return _emit("mm", "extern", [self, other], self._val @ other._val,
                     attrs={"lib": "extern_atomlib"})

    # ---- 原地修改：**不是**切图触发条件 ----

    def add_(self, other):
        """原地加法。**不是切图触发条件**——官方能处理大量原地/mutation。"""
        ov = other._val if isinstance(other, P) else other
        self._val[...] = self._val + ov
        ins = [self, other if isinstance(other, P) else None]
        r = _emit("add_", "pointwise", ins, self._val, attrs={"inplace": True})
        self._node = r._node            # 原地语义：代理本身仍然指向同一个缓冲
        return self

    def copy_(self, other):
        """原地拷贝。同样**不是**切图触发条件。"""
        ov = other._val if isinstance(other, P) else other
        self._val[...] = ov
        r = _emit("copy_", "pointwise", [other if isinstance(other, P) else None],
                  self._val, attrs={"inplace": True})
        self._node = r._node
        return self

    # ---- 数据依赖控制流：**切图** ----

    def item(self):
        _S.counter += 1
        if _S.recording and _S.counter > _S.eager_prefix:
            raise GraphBreak("数据依赖控制流（.item()）", _S.counter)
        return float(self._val)


def mprint(*args):
    """模拟 `print`。**未登记时切图**；登记后改为重排（不切图）。"""
    _S.counter += 1
    if _S.recording and _S.counter > _S.eager_prefix:
        if "print" in _S.reorderable_logging:
            _S.graph.add_effect("print", args)
            return None
        raise GraphBreak("print / logging（未登记进 reorderable_logging_functions）",
                         _S.counter)
    return None


# ---------------------------------------------------------------------------
# 迷你 Dynamo：帧入口钩子 + 捕获 + 重放
# ---------------------------------------------------------------------------


class CaptureResult:
    def __init__(self, subgraphs, breaks, frame_entries, passes, skipped_frames):
        self.subgraphs = subgraphs
        self.breaks = breaks
        self.frame_entries = frame_entries
        self.passes = passes
        self.skipped_frames = skipped_frames

    @property
    def n_subgraphs(self):
        return len(self.subgraphs)

    def summary(self):
        return "子图 %d 张 · 切图 %d 次 · 重放 %d 轮 · 帧入口放行 %d 次 / 跳过 %d 次" % (
            self.n_subgraphs, len(self.breaks), self.passes,
            len(self.frame_entries), self.skipped_frames)


class MiniDynamo:
    """迷你 Dynamo：决定哪些帧进入捕获，并把它们切成若干张子图。"""

    MAX_PASSES = 8

    def __init__(self, reorderable_logging=()):
        self.registry = set()
        self.reorderable_logging = set(reorderable_logging)

    def register(self, fn):
        """登记「要编译」的函数。真实 Dynamo 也在帧入口做这个决定。"""
        self.registry.add(fn.__code__)
        return fn

    def _hook(self, frame, event, arg):
        # ★ 这就是「帧入口决策」在迷你实现里的位置。
        #   真实 Dynamo 的同名决策发生在 PEP 523 的帧求值回调里，**不是** sys.settrace。
        if event == "call":
            if frame.f_code in self.registry:
                _S.frame_entries.append((frame.f_code.co_name, True))
            else:
                _S.skipped_frames += 1
        return None      # 不要行级事件：本仓库靠代理记录算子

    def capture(self, fn, inputs, name="fwd"):
        """跑 fn(*inputs)，返回 CaptureResult。

        `inputs` 是 numpy 数组列表。每次切图后**重放**整个函数：
        前 k 个算子走 eager，从第 k+1 个起重新记录。
        """
        old_hook = sys.gettrace()
        _S.recording = True
        _S.eager_prefix = 0
        _S.counter = 0
        _S.frame_entries = []
        _S.skipped_frames = 0
        _S.reorderable_logging = set(self.reorderable_logging)
        subgraphs, breaks, passes = [], [], 0
        sys.settrace(self._hook)
        try:
            for _ in range(1, self.MAX_PASSES + 1):
                passes += 1
                _S.counter = 0
                _S.graph = Graph("%s_s%d" % (name, len(subgraphs)))
                args = [P(np.asarray(a, dtype=np.float32), None) for a in inputs]
                for i, a in enumerate(args):
                    ph = _S.graph.placeholder(a.shape, "arg")
                    ph.value = a.numpy()
                    a._node = ph
                try:
                    out = fn(*args)
                except GraphBreak as gb:
                    breaks.append(gb)
                    subgraphs.append(_close(_S.graph, []))
                    _S.eager_prefix = gb.op_index
                    continue
                outs = out if isinstance(out, (list, tuple)) else [out]
                subgraphs.append(_close(
                    _S.graph,
                    [o._node for o in outs
                     if isinstance(o, P) and o._node is not None]))
                return CaptureResult(subgraphs, breaks, list(_S.frame_entries),
                                     passes, _S.skipped_frames)
            raise RuntimeError("重放轮数超过上限 %d——迷你追踪器没能收敛"
                               % self.MAX_PASSES)
        finally:
            sys.settrace(old_hook)
            _S.recording = False
            _S.reorderable_logging = set()


def _close(graph, outputs):
    """收尾：剪掉没有任何读者的占位符（函数签名里没用到的入参就在这一类）。"""
    read = {d.buf for n in graph.nodes for d in n.reads}
    graph.nodes = [n for n in graph.nodes
                   if n.kind != "placeholder" or n.buf in read]
    graph.outputs = outputs
    return graph


# ---------------------------------------------------------------------------
# 解释器：按拓扑序逐节点求值（独立于「捕获时的 eager 路径」的一条参照）
# ---------------------------------------------------------------------------


class InterpError(Exception):
    pass


def eval_graph(graph, env):
    """在 `env`（缓冲名 -> numpy 数组）上把整张图算出来，返回 env。"""
    for node in graph.nodes:
        if node.kind in ("placeholder", "effect"):
            continue
        try:
            vals = [env[d.buf] for d in node.reads]
        except KeyError as e:
            raise InterpError("缓冲 %s 没有输入值（缺 %s）" % (e, node.buf)) from None
        env[node.buf] = _apply(node, vals)
    return env


def _apply(node, vals):
    op = node.op
    const = node.attrs.get("const")
    if op in ("add", "mul", "sub", "div"):
        b = const if const is not None else (vals[1] if len(vals) > 1 else vals[0])
        return {"add": vals[0] + b, "mul": vals[0] * b,
                "sub": vals[0] - b, "div": vals[0] / b}[op]
    if op == "neg":
        return -vals[0]
    if op == "relu":
        return np.maximum(vals[0], 0.0)
    if op == "exp":
        return np.exp(vals[0])
    if op == "sqrt":
        return np.sqrt(vals[0])
    if op == "transpose":
        return np.ascontiguousarray(vals[0].T)
    if op == "sum":
        ax = node.attrs["axis"]
        return vals[0].sum() if ax is None else vals[0].sum(axis=ax)
    if op == "mm":
        return vals[0] @ vals[1] if len(vals) > 1 else vals[0] @ vals[0]
    if op == "add_":
        return vals[0] + vals[1]
    if op == "copy_":
        return vals[0]
    raise InterpError("解释器不认识算子 %r" % op)


def max_abs_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0
