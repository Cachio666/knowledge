"""迷你 IR：Op / Region / Block / Value 四层递归结构 + 迷你解释器。

★ 本模块是本仓库的**迷你实现，不是 MLIR**。
  术语（Op / Region / Block / Value / tensor semantics / memref semantics /
  bufferization / one-shot）取自 MLIR 官方文档以便对照，**结构与语义都是本仓库
  自己简化的**，与真实 MLIR 不同。引用本模块输出的结论只能关于「机制」，
  不能关于 MLIR 本身（见各篇 §5 的限定语）。

官方依据（逐条 URL 见 specs.py）：
  · 「Operations are contained in Blocks and Blocks are contained in Regions.
    Operations may also contain regions, enabling hierarchical structures to be
    represented.」                                    —— LangRef High-Level Structure
  · 「Each Value is the result of exactly one Operation or Block Argument」
                                                     —— LangRef High-Level Structure
  · region      ::= "{" entry-block? block* "}"      —— Region **可以是空的**
  · block       ::= block-label operation+           —— Block **非空**
  · region-list ::= "(" region ("," region)* ")"     —— Operation 有 **0 个到多个** region
  · bufferization 是「the process of converting ops with tensor semantics to ops
    with memref semantics」                           —— Bufferization 文档

本模块提供五组东西：
  1. 结构：Ty / Value / Op / Block / Region / Module + IRBuilder
  2. 校验：verify() —— SSA 唯一定义、Block 非空、Block 以 terminator 结尾、operand 已定义
  3. 统计与打印：walk_ops / count_ops / max_depth / all_values / print_ir
  4. 解释器与运行时：MemRef / Interp
  5. 断言框架：Checks（三类断言 A/B/C + 末尾汇总 + 失败非零退出）
"""

import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class IRError(Exception):
    """本模块所有 IR 相关错误的基类。"""


class SSAError(IRError):
    """违反 SSA：同一个 Value 被定义两次，或引用了未定义的 Value。"""


class IRVerificationError(IRError):
    """IR 结构不合法：Block 为空、Block 没有 terminator 等。"""


class TransformFailure(IRError):
    """变换（pass / transform op）拒绝执行。对应真实 MLIR 里 transform 的 failure。"""


class NonConvergenceError(IRError):
    """rewrite driver 在迭代上限内没有到达不动点。"""


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

_SCALAR_ELEMS = frozenset({"f16", "f32", "f64", "i32", "i64", "index"})


class Ty:
    """迷你类型：scalar / tensor / memref。

    `tensor<2x3xf32>` 是 tensor 语义（未指定存储），`memref<2x3xf32>` 是 memref 语义
    （已指定存储）。形状部分相同，差别只在 kind——这正是 bufferization 要消掉的那一维。
    """

    def __init__(self, kind, shape=(), elem="f32"):
        self.kind = kind                       # "scalar" | "tensor" | "memref"
        self.shape = tuple(int(d) for d in shape)
        self.elem = elem

    @property
    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def is_tensor(self):
        return self.kind == "tensor"

    @property
    def is_memref(self):
        return self.kind == "memref"

    @property
    def is_scalar(self):
        return self.kind == "scalar"

    @property
    def rank(self):
        return len(self.shape)

    def with_kind(self, kind):
        return Ty(kind, self.shape, self.elem)

    def __str__(self):
        if self.kind == "scalar":
            return self.elem
        body = ("x".join(str(d) for d in self.shape) + "x" + self.elem) \
            if self.shape else self.elem
        return "%s<%s>" % (self.kind, body)

    def __repr__(self):
        return "Ty(%s)" % self


def parse_ty(text):
    """解析 'tensor<2x3xf32>' / 'memref<2x3xf32>' / 'f32' / 'index'。"""
    text = text.strip()
    m = re.fullmatch(r"(tensor|memref)<(.+)>", text)
    if not m:
        if text in _SCALAR_ELEMS:
            return Ty("scalar", (), text)
        raise IRError("无法解析类型：%r" % text)
    kind, body = m.group(1), m.group(2)
    parts = body.split("x")
    elem, dims = parts[-1], parts[:-1]
    for d in dims:
        if not d.isdigit():
            raise IRError("迷你实现只支持静态形状：%r" % text)
    return Ty(kind, [int(d) for d in dims], elem)


# ---------------------------------------------------------------------------
# 结构：Value / Op / Block / Region / Module
# ---------------------------------------------------------------------------

TERMINATORS = frozenset({
    "func.return", "scf.yield", "gpu.return", "cf.br", "cf.cond_br", "func.call",
})


def _owner_name(owner):
    if isinstance(owner, Op):
        return owner.name
    if isinstance(owner, Block):
        return "block-arg@%s" % owner.label
    return repr(owner)


def _bind(value, owner, what):
    """把 Value 绑定到它的定义者。**第二次绑定即为违反 SSA。**"""
    if value.owner is not None:
        raise SSAError("违反 SSA：值 %s 已被 (%s) 定义过，不能再被 (%s) 定义"
                       % (value.name, _owner_name(value.owner), what))
    value.owner = owner


class Value:
    """一个 SSA 值：恰是一个 Operation 的结果，或一个 Block 参数。"""

    _counter = [0]

    def __init__(self, ty, name=None):
        Value._counter[0] += 1
        self.id = Value._counter[0]
        self.name = name or ("%%v%d" % self.id)
        self.ty = ty if isinstance(ty, Ty) else parse_ty(ty)
        self.owner = None            # Op 或 Block；None = 尚未定义
        self.uses = []

    @property
    def defined(self):
        return self.owner is not None

    @property
    def num_uses(self):
        return len(self.uses)

    def __repr__(self):
        return "Value(%s : %s)" % (self.name, self.ty)


class Op:
    """一个 Operation：0..n 个 operand、0..n 个 result、0..n 个 region。"""

    def __init__(self, name, operands=(), result_types=(), attrs=None, num_regions=0,
                 result_values=None, no_terminator=False):
        if not name:
            raise IRError("Operation 必须有名字")
        self.name = name
        self.operands = list(operands)
        self.attrs = dict(attrs or {})
        self.parent = None                                      # 所属 Block
        # NoTerminator：官方 LangRef §Blocks——「The last operation in a block must be a
        # terminator operation. A region with a single block may opt out of this
        # requirement by attaching the NoTerminator on the enclosing op. The top-level
        # ModuleOp is an example of such an operation…」
        self.no_terminator = no_terminator
        if result_values is not None:
            # 显式注入 Value 对象：给「同一个 Value 定义两次」这条反例用
            self.results = list(result_values)
        else:
            self.results = [Value(t) for t in result_types]
        self.regions = [Region(self) for _ in range(num_regions)]
        for v in self.operands:
            v.uses.append(self)
        for v in self.results:
            _bind(v, self, self.name)

    @property
    def num_regions(self):
        return len(self.regions)

    @property
    def num_results(self):
        return len(self.results)

    def is_terminator(self):
        return self.name in TERMINATORS

    def __repr__(self):
        return "Op(%s, %d ops, %d regions)" % (self.name, self.num_results,
                                               self.num_regions)


class Block:
    """一个 Block：`block-label operation+` —— **非空**，且以 terminator 结尾。"""

    _counter = [0]

    def __init__(self, label=None):
        Block._counter[0] += 1
        self.label = label or ("^bb%d" % Block._counter[0])
        self.args = []
        self.ops = []
        self.parent = None                                      # 所属 Region

    def add_arg(self, ty, name=None):
        v = Value(ty if isinstance(ty, Ty) else parse_ty(ty), name)
        _bind(v, self, "block-arg")
        self.args.append(v)
        return v

    def append(self, op):
        if op.parent is not None:
            raise IRVerificationError("Operation %s 已属于另一个 Block" % op.name)
        op.parent = self
        self.ops.append(op)
        return op

    @property
    def terminated(self):
        return bool(self.ops) and self.ops[-1].is_terminator()

    def __repr__(self):
        return "Block(%s, %d ops)" % (self.label, len(self.ops))


class Region:
    """一个 Region：0..n 个 Block。**可以是空的**。"""

    def __init__(self, parent=None):
        self.blocks = []
        self.parent = parent                                    # 所属 Op

    def new_block(self, label=None):
        b = Block(label)
        b.parent = self
        self.blocks.append(b)
        return b

    @property
    def num_blocks(self):
        return len(self.blocks)

    def __repr__(self):
        return "Region(%d blocks)" % len(self.blocks)


class Module:
    """一个 module op：`builtin.module` 包一个 region，region 里一个 entry block。

    ★ module 的 block **不需要 terminator**——官方 LangRef 明写 top-level ModuleOp 是
      带 `NoTerminator` 的那个例子。这不是「近似」，是 IR 里真实存在的第二种 region 形态。
    """

    def __init__(self, name="builtin.module"):
        self.op = Op(name, num_regions=1, no_terminator=True)
        self.region = self.op.regions[0]
        self.body = self.region.new_block("^entry")

    def __repr__(self):
        return "Module(%s)" % self.op.name


class IRBuilder:
    """薄封装：把 op 追加到「当前 block」。"""

    def __init__(self, block):
        self.block = block

    def emit(self, name, operands=(), result_types=(), attrs=None, num_regions=0,
             result_values=None):
        op = Op(name, operands, result_types, attrs, num_regions,
                result_values=result_values)
        self.block.append(op)
        return op

    def func(self, sym, arg_types, result_types=()):
        """建一个 func.func，返回 (op, body builder)。entry block 的参数 = 函数参数。"""
        op = self.emit("func.func", (), result_types, attrs={"sym": sym}, num_regions=1)
        body = op.regions[0].new_block("^entry")
        for i, t in enumerate(arg_types):
            body.add_arg(t, "%%arg%d" % i)
        return op, IRBuilder(body)

    def terminate(self, name, operands=()):
        return self.emit(name, operands)


# ---------------------------------------------------------------------------
# 遍历 / 校验 / 统计 / 打印
# ---------------------------------------------------------------------------


def _as_op(node):
    return node.op if isinstance(node, Module) else node


def walk_ops(node):
    """先序递归遍历，产出 Op。node 可以是 Module / Op / Region / Block / list。"""
    node = _as_op(node)
    if isinstance(node, Op):
        yield node
        for r in node.regions:
            yield from walk_ops(r)
        return
    if isinstance(node, Region):
        for b in node.blocks:
            yield from walk_ops(b)
        return
    if isinstance(node, Block):
        for op in node.ops:
            yield from walk_ops(op)
        return
    if isinstance(node, (list, tuple)):
        for x in node:
            yield from walk_ops(x)
        return
    raise TypeError("不能遍历 %r" % (node,))


def all_regions(node):
    """所有 Region（含空的）。"""
    node = _as_op(node)
    if isinstance(node, Op):
        for r in node.regions:
            yield r
            yield from all_regions(r)
    elif isinstance(node, Region):
        for b in node.blocks:
            yield from all_regions(b)
    elif isinstance(node, Block):
        for op in node.ops:
            yield from all_regions(op)
    elif isinstance(node, (list, tuple)):
        for x in node:
            yield from all_regions(x)
    else:
        raise TypeError("不能遍历 %r" % (node,))


def all_blocks(node):
    for r in all_regions(node):
        for b in r.blocks:
            yield b


def count_ops(node, name=None):
    """按名字统计 op 数。name=None 返回 dict，否则返回单个计数。"""
    if name is None:
        out = {}
        for op in walk_ops(node):
            out[op.name] = out.get(op.name, 0) + 1
        return out
    return sum(1 for op in walk_ops(node) if op.name == name)


def max_depth(node):
    """嵌套深度：module 那一层记为 0；op 里嵌 op 才 +1。"""
    node = _as_op(node)
    if isinstance(node, Op):
        best = 0
        for r in node.regions:
            for b in r.blocks:
                for op in b.ops:
                    best = max(best, 1 + max_depth(op))
        return best
    if isinstance(node, Region):
        return max((max_depth(b) for b in node.blocks), default=0)
    if isinstance(node, Block):
        return max((max_depth(op) for op in node.ops), default=0)
    if isinstance(node, (list, tuple)):
        return max((max_depth(x) for x in node), default=0)
    raise TypeError("不能求深度的 %r" % (node,))


def all_values(node):
    """所有 Value：op 的结果 + block 的参数（LangRef 的 Value 定义只有这两种来源）。"""
    out = []
    for op in walk_ops(node):
        out.extend(op.results)
    for b in all_blocks(node):
        out.extend(b.args)
    return out


def _needs_terminator(block):
    """单块 region 的父 op 带 NoTerminator 时，该 block 免于「必须以 terminator 结尾」。"""
    region = block.parent
    if region is None or region.parent is None:
        return False
    return not (region.parent.no_terminator and region.num_blocks == 1)


def is_const_op(v):
    """v 是不是一个 `arith.constant` 的结果。"""
    return v.owner is not None and getattr(v.owner, "name", None) == "arith.constant"


def const_value(v):
    return v.owner.attrs["value"]


def mk_const(value, ty="f32"):
    return Op("arith.constant", (), [ty], attrs={"value": value})


def remove_op(block, op):
    if op in block.ops:
        block.ops.remove(op)
    op.parent = None


def replace_uses(block, old_val, new_val):
    """把 block 内所有对 old_val 的使用换成 new_val（同时维护 uses 表）。"""
    for op in list(block.ops):
        if old_val not in op.operands:
            continue
        op.operands = [new_val if v is old_val else v for v in op.operands]
        new_val.uses.append(op)
    old_val.uses = [op for op in old_val.uses
                    if any(v is old_val for v in op.operands)]


def splice_op(block, op, new_ops, new_result=None, drop_old=True):
    """把 op 换成 new_ops（插在同一位置），并把 op 结果的**所有使用者**改指向
    new_result。`drop_old=False` 用于「新 op 建立在旧 op 结果之上」那种不删旧的改写。"""
    i = block.ops.index(op)
    for k, o in enumerate(new_ops):
        block.ops.insert(i + k, o)
        o.parent = block
    if new_result is not None:
        replace_uses(block, op.results[0], new_result)
    if drop_old:
        remove_op(block, op)


def dead_code_elimination(block):
    """删掉结果没有任何使用者的 op（terminator 除外）。返回删除数。

    官方 applyPatternsGreedily 的注释：「Also performs simple dead-code elimination
    **before attempting to match** any of the provided patterns.」
    """
    removed = 0
    for op in list(block.ops):
        if op.is_terminator() or not op.results:
            continue
        if not any(op.results[0] in other.operands
                   for other in block.ops if other is not op):
            remove_op(block, op)
            removed += 1
    return removed


def verify(node):
    """结构校验。返回问题列表；空列表 = 通过。

    检查项（每条对应官方语法的一处约束）：
      1. Block **非空**（`block ::= block-label operation+`）
      2. Block **以 terminator 结尾**——但带 `NoTerminator` 的 op 的单块 region 例外
         （top-level ModuleOp 就是官方举的那个例子）
      3. 每个 operand 都已被定义
    Region 可以是空的——**不报错**。
    """
    problems = []
    for b in all_blocks(node):
        if not b.ops:
            problems.append("%s: Block 为空（官方语法要求 operation+）" % b.label)
        elif _needs_terminator(b) and not b.terminated:
            problems.append("%s: 最后一个 op 是 %s，不是 terminator"
                            % (b.label, b.ops[-1].name))
    for op in walk_ops(node):
        for v in op.operands:
            if not v.defined:
                problems.append("%s: operand %s 未被定义" % (op.name, v.name))
    return problems


def verify_or_raise(node):
    problems = verify(node)
    if problems:
        raise IRVerificationError("；".join(problems))
    return True


def _attr_str(v):
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def format_op(op, indent):
    pad = "  " * indent
    lhs = ""
    if op.results:
        lhs = ", ".join("%s: %s" % (v.name, v.ty) for v in op.results) + " = "
    line = "%s%s%s" % (pad, lhs, op.name)
    if op.operands:
        line += "(" + ", ".join(v.name for v in op.operands) + ")"
    if op.results:
        line += " : (%s) -> %s" % (", ".join(str(v.ty) for v in op.operands),
                                   ", ".join(str(v.ty) for v in op.results))
    elif op.operands:
        line += " : (%s)" % ", ".join(str(v.ty) for v in op.operands)
    if op.attrs:
        kv = ", ".join("%s = %s" % (k, _attr_str(v)) for k, v in sorted(op.attrs.items()))
        line += " {%s}" % kv
    return line


def format_region(region, indent):
    lines = []
    for b in region.blocks:
        head = "  " * indent + b.label
        if b.args:
            head += "(" + ", ".join("%s: %s" % (v.name, v.ty) for v in b.args) + ")"
        lines.append(head + ":")
        for op in b.ops:
            lines.append(format_op(op, indent + 1))
            for r in op.regions:
                lines.append("  " * (indent + 1) + "{")
                lines.extend(format_region(r, indent + 2))
                lines.append("  " * (indent + 1) + "}")
    return lines


def print_ir(node, indent=0):
    """打印成类 MLIR 文本。**格式是本仓库自己定的，不是 MLIR 的打印格式。**"""
    node = _as_op(node)
    lines = ["  " * indent + node.name + " {"]
    for r in node.regions:
        lines.extend(format_region(r, indent + 1))
    lines.append("  " * indent + "}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 解释器
# ---------------------------------------------------------------------------


class MemRef:
    """缓冲语义的运行时对象：**可变** numpy 数组 + 分配点编号（用于看别名）。"""

    _counter = [0]

    def __init__(self, data, ty):
        MemRef._counter[0] += 1
        self.id = MemRef._counter[0]
        self.data = data
        self.ty = ty
        self.freed = False

    def __repr__(self):
        return "MemRef#%d<%s>" % (self.id, self.ty.shape)


class Interp:
    """迷你解释器。

    **串行执行 scf.parallel / scf.forall / gpu.launch**——本仓库只用它检查各
    lowering 阶段的**语义等价**，不检查并行性；真实并行性由 gpu 映射的类型决定。
    """

    def __init__(self):
        self.allocs = 0
        self.frees = 0
        self.executed = []          # 实际执行过的 op 名字序列（含重复）
        self.parallel_ops = 0
        self._ret = None
        self.nan_reads = 0          # 读到未初始化（NaN）元素的次数

    # -- 入口 -------------------------------------------------------------

    def run(self, node, args=()):
        env = {}
        node = _as_op(node)
        if node.name != "func.func":
            funcs = [op for op in walk_ops(node) if op.name == "func.func"]
            if len(funcs) != 1:
                raise IRError("模块里有 %d 个 func.func，无法确定入口" % len(funcs))
            node = funcs[0]
        if node.name == "func.func":
            body = node.regions[0].blocks[0]
            if len(args) != len(body.args):
                raise IRError("需要 %d 个实参，给了 %d 个" % (len(body.args), len(args)))
            for v, a in zip(body.args, args):
                env[v] = self._coerce(v.ty, a)
            self.exec_block(body, env)
            return self._ret
        for b in node.regions[0].blocks:
            self.exec_block(b, env)
        return None

    def _coerce(self, ty, a):
        if ty.is_scalar:
            return float(a)
        if ty.is_memref:
            if isinstance(a, MemRef):
                return a
            return MemRef(np.array(a, dtype=np.float64), ty)
        return np.array(a, dtype=np.float64)

    @staticmethod
    def _val(x):
        """取值：MemRef 取它的 numpy 数组，其余原样。"""
        return x.data if isinstance(x, MemRef) else x

    # -- 执行 -------------------------------------------------------------

    def exec_block(self, block, env):
        for op in block.ops:
            self.exec_op(op, env)

    def exec_region(self, region, env):
        for b in region.blocks:
            self.exec_block(b, env)

    def exec_op(self, op, env):
        name = op.name
        self.executed.append(name)

        if name == "func.func":
            self.exec_region(op.regions[0], env)
        elif name == "func.return":
            self._ret = env[op.operands[0]] if op.operands else None
        elif name == "scf.yield":
            pass
        elif name == "arith.constant":
            self._put(op, 0, env, float(op.attrs["value"]))
        elif name == "tensor.empty":
            # 内容未初始化：迷你实现填 NaN，读到它就能被发现
            self._put(op, 0, env, np.full(op.results[0].ty.shape, np.nan))
        elif name == "memref.alloc":
            ty = op.results[0].ty
            self.allocs += 1
            self._put(op, 0, env, MemRef(np.zeros(ty.shape), ty))
        elif name == "memref.dealloc":
            env[op.operands[0]].freed = True
            self.frees += 1
        elif name == "memref.load":
            m = env[op.operands[0]]
            idx = tuple(int(env[i]) for i in op.operands[1:])
            val = float(m.data[idx])
            if repr(val) == "nan":
                self.nan_reads += 1
            self._put(op, 0, env, val)
        elif name == "memref.store":
            m = env[op.operands[1]]
            idx = tuple(int(env[i]) for i in op.operands[2:])
            m.data[idx] = env[op.operands[0]]
        elif name in ("arith.addf", "arith.mulf", "arith.subf", "arith.divf"):
            a, b = env[op.operands[0]], env[op.operands[1]]
            f = {"arith.addf": np.add, "arith.mulf": np.multiply,
                 "arith.subf": np.subtract, "arith.divf": np.divide}[name]
            self._put(op, 0, env, float(f(a, b)))
        elif name in LINALG_OPS:
            self._exec_linalg(op, env)
        elif name == "scf.for":
            lb, ub, step = (int(env[o]) for o in op.operands)
            body = op.regions[0].blocks[0]
            for iv in range(lb, ub, step):
                env[body.args[0]] = float(iv)
                self.exec_block(body, env)
        elif name in ("scf.parallel", "scf.forall"):
            self.parallel_ops += 1
            self._exec_index_space(op, env)
        elif name == "gpu.launch":
            self.parallel_ops += 1
            self._exec_index_space(op, env)
        else:
            raise IRError("迷你解释器不认识 op：%s" % name)

    def _put(self, op, i, env, value):
        env[op.results[i]] = value

    def _exec_linalg(self, op, env):
        dest = env[op.operands[-1]]            # 约定：linalg op 的最后一个 operand 是 outs
        if op.name == "linalg.fill":
            res = np.full(dest.data.shape if isinstance(dest, MemRef) else dest.shape,
                          env[op.operands[0]])
        elif op.name in ("linalg.add", "linalg.mul"):
            a, b = self._val(env[op.operands[0]]), self._val(env[op.operands[1]])
            res = (np.add if op.name == "linalg.add" else np.multiply)(a, b)
        elif op.name == "linalg.matmul":
            res = self._val(env[op.operands[0]]) @ self._val(env[op.operands[1]])
        elif op.name == "linalg.reduce_sum":
            res = self._val(env[op.operands[0]]).sum(axis=0, keepdims=True)
        else:
            raise IRError("未知 linalg op：%s" % op.name)

        if isinstance(dest, MemRef):
            dest.data[...] = res
            if op.results:
                self._put(op, 0, env, dest)    # destination-style：结果就是同一个缓冲
        else:
            self._put(op, 0, env, res)

    def _exec_index_space(self, op, env):
        """串行执行多维索引空间。scf.parallel / scf.forall / gpu.launch 共用。"""
        dims = []
        for i in range(0, len(op.operands), 3):
            dims.append((int(env[op.operands[i]]), int(env[op.operands[i + 1]]),
                         int(env[op.operands[i + 2]])))
        body = op.regions[0].blocks[0]

        def rec(d, idx):
            if d == len(dims):
                for arg, i in zip(body.args, idx):
                    env[arg] = float(i)
                self.exec_block(body, env)
                return
            lb, ub, step = dims[d]
            for i in range(lb, ub, step):
                rec(d + 1, idx + [i])
        rec(0, [])


LINALG_OPS = frozenset({"linalg.fill", "linalg.add", "linalg.mul",
                        "linalg.matmul", "linalg.reduce_sum"})


# ---------------------------------------------------------------------------
# 断言框架：三类断言 + 末尾汇总
# ---------------------------------------------------------------------------

A_NAME = "A 算法正确性"
B_NAME = "B 代价模型"
C_NAME = "C 边界与反例"


class Checks:
    """三类断言的记账器。

    A 算法正确性：与**独立参考实现**逐项比对，打印最大误差 / 差异数。
    B 代价模型  ：按文档公式算出的数值，须与文档表格逐行一致。
    C 边界与反例：文档 §4 声称「会出问题」的输入**真的跑出问题**——
                  断言的是**失败行为成立**，不是成功。

    失败累计，`summary()` 以非零码退出。
    """

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
        except Exception as e:                                  # noqa: BLE001
            return self._record("C", False, msg, detail or
                                ("抛了非预期的 %s: %s" % (type(e).__name__, e)))
        return self._record("C", False, msg, detail or "**没有抛异常**——失败行为没发生")

    def expect_ok(self, fn, msg, detail=""):
        """断言 fn 能跑通。用来给 C 类配一个「换成正确做法就通过」的对照。"""
        try:
            r = fn()
        except Exception as e:                                  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def max_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0
