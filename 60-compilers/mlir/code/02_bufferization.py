"""02 tensor 与 memref：bufferization 把 tensor 语义换成 memref 语义。

对应文档：60-compilers/mlir/02-tensor-vs-memref.md

★ 本程序跑的是本仓库的**迷你 bufferization**（miniir.py + 本文件的 analyze/bufferize），
  不是 MLIR 的 `-one-shot-bufferize`。输出的一切数字都是**本机实测值**。

本程序要证的机制只有一条：
  tensor 语义下每个结果都可能被物化；memref 语义把它变成显式的 `memref.alloc`，
  而「要不要真的分配」由**一个就地判据**决定——判据写错，程序就会静默算错。

用法：python3 02_bufferization.py
"""

import sys

import numpy as np

import specs as SPECS
from miniir import (IRBuilder, Interp, MemRef, Module, Checks, TransformFailure,
                    count_ops, max_err, print_ir, rule)

SHAPE = (2, 3)
TENSOR_TY = "tensor<%dx%dxf32>" % SHAPE
MEMREF_TY = "memref<%dx%dxf32>" % SHAPE
K = 1.5

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

LINALG_WITH_OUTS = frozenset({"linalg.fill", "linalg.add", "linalg.mul",
                              "linalg.matmul", "linalg.reduce_sum"})


# ==== 造 IR ====


def build_tensor_ir():
    """链式 elementwise：t1 = k；t2 = a + t1；t3 = t2 * b。

    每个中间结果在 tensor 语义下都是一个 SSA 值，**存储完全没被指定**。
    """
    m = Module()
    outer = IRBuilder(m.body)
    fop, b = outer.func("main", [TENSOR_TY, TENSOR_TY])
    a, bb = fop.regions[0].blocks[0].args
    cst = b.emit("arith.constant", (), ["f32"], attrs={"value": K})
    e0 = b.emit("tensor.empty", (), [TENSOR_TY])
    t1 = b.emit("linalg.fill", [cst.results[0], e0.results[0]], [TENSOR_TY])
    t2 = b.emit("linalg.add", [a, t1.results[0], t1.results[0]], [TENSOR_TY])
    t3 = b.emit("linalg.mul", [t2.results[0], bb, t2.results[0]], [TENSOR_TY])
    b.terminate("func.return", [t3.results[0]])
    return m


def build_conflict_ir():
    """**故意的反例地形**：`%t1` 在被就地写之后还被读到。

    t1 = k；t2 = a + b（outs 是 %t1）；t3 = t1 * t2。
    `%t1` 的最后一次使用在 t3 而不是 t2 —— 判据必须因此拒绝就地。
    地形的构造规则写死在这里，不因结果调整。
    """
    m = Module()
    outer = IRBuilder(m.body)
    fop, b = outer.func("main", [TENSOR_TY, TENSOR_TY])
    a, bb = fop.regions[0].blocks[0].args
    cst = b.emit("arith.constant", (), ["f32"], attrs={"value": K})
    e0 = b.emit("tensor.empty", (), [TENSOR_TY])
    t1 = b.emit("linalg.fill", [cst.results[0], e0.results[0]], [TENSOR_TY])
    t2 = b.emit("linalg.add", [a, bb, t1.results[0]], [TENSOR_TY])
    t3 = b.emit("linalg.mul", [t1.results[0], t2.results[0], t2.results[0]],
                [TENSOR_TY])
    b.terminate("func.return", [t3.results[0]])
    return m


def build_double_write_ir():
    """另一个反例：同一个 tensor 值被**两个 op** 当作 outs —— 写冲突。"""
    m = Module()
    outer = IRBuilder(m.body)
    fop, b = outer.func("main", [TENSOR_TY, TENSOR_TY])
    a, bb = fop.regions[0].blocks[0].args
    cst = b.emit("arith.constant", (), ["f32"], attrs={"value": K})
    e0 = b.emit("tensor.empty", (), [TENSOR_TY])
    t1 = b.emit("linalg.fill", [cst.results[0], e0.results[0]], [TENSOR_TY])
    b.emit("linalg.add", [a, bb, t1.results[0]], [TENSOR_TY])
    t3 = b.emit("linalg.mul", [a, bb, t1.results[0]], [TENSOR_TY])   # %t1 又当 outs
    b.terminate("func.return", [t3.results[0]])
    return m


# ==== 判据（迷你 one-shot 的就地分析） ====


class Decision:
    """一个 tensor 运算的物化决策。"""

    def __init__(self, idx, op, outs, result, kind):
        self.idx = idx                 # 在原函数 body 里的位置
        self.op = op
        self.outs = outs               # outs 那个 tensor 值
        self.result = result           # 该 op 的结果 tensor 值
        self.kind = kind               # "fresh-empty" | "in-place" | "out-of-place"

    @property
    def allocs(self):
        return 1 if self.kind == "out-of-place" else 0


def body_ops(func_op):
    return func_op.regions[0].blocks[0].ops


def last_use_index(body, v):
    """v 在 body 里最后一次被当作 operand 的位置。没有使用则返回 None。"""
    idx = None
    for i, op in enumerate(body):
        if v in op.operands:
            idx = i
    return idx


def times_as_outs(body, v):
    return sum(1 for op in body
               if op.name in LINALG_WITH_OUTS and op.operands and op.operands[-1] is v)


def analyze(func_op, force_inplace=False):
    """迷你就地分析。

    规则（**判据是这个程序被测的东西**）：
      1. outs 是 `tensor.empty` 的结果 → 这次分配归 empty（alloc 1 次，内容无关）
      2. outs 的**最后一次使用就是这个 op** → 就地复用它的 buffer（alloc 0 次）
      3. 否则（outs 在本 op 之后还有读者）→ 必须新分配（alloc 1 次）
    另外：同一个值被两个 op 当 outs → 写冲突，直接拒绝。
    """
    body = body_ops(func_op)
    decs = []
    for i, op in enumerate(body):
        if op.name not in LINALG_WITH_OUTS or not op.results:
            continue
        outs = op.operands[-1]
        if times_as_outs(body, outs) > 1:
            raise TransformFailure(
                "写冲突：%s 被多个 op 当作 outs，就地分析无法给出唯一决策" % outs.name)
        producer = outs.owner
        if getattr(producer, "name", None) == "tensor.empty":
            kind = "fresh-empty"
        elif force_inplace:
            kind = "in-place"                 # 反例开关：无视判据，一律就地
        elif last_use_index(body, outs) == i:
            kind = "in-place"
        else:
            kind = "out-of-place"
        decs.append(Decision(i, op, outs, op.results[0], kind))
    return decs


# ==== bufferize：tensor 语义 → memref 语义 ====


def bufferize(func_op, force_inplace=False, naive=False):
    """产出 memref 语义的模块。

    naive=True 时**不做就地分析**：每个 tensor 结果各分配一块缓冲（朴素做法）。
    两种模式都**不插入 dealloc**——真实 MLIR 的 `-one-shot-bufferize` 同样不做 dealloc，
    它把这步交给 ownership-based buffer deallocation（见 specs.BUFFERIZATION_DEPRECATED_PASSES 旁注）。
    """
    body = body_ops(func_op)
    decs = {d.idx: d for d in analyze(func_op, force_inplace)} if not naive else {}
    src_args = func_op.regions[0].blocks[0].args

    m = Module()
    outer = IRBuilder(m.body)
    _, b = outer.func("main", [str(v.ty.with_kind("memref")) for v in src_args])
    new_args = b.block.args
    valmap = dict(zip(src_args, new_args))

    for i, op in enumerate(body):
        if op.name == "arith.constant":
            new = b.emit("arith.constant", (), [str(op.results[0].ty)], attrs=op.attrs)
            valmap[op.results[0]] = new.results[0]
        elif op.name == "tensor.empty":
            # tensor.empty 就是分配点：真实 MLIR 里它 bufferize 成 alloc_tensor → memref.alloc
            new = b.emit("memref.alloc", (), [str(op.results[0].ty.with_kind("memref"))])
            valmap[op.results[0]] = new.results[0]
        elif op.name in LINALG_WITH_OUTS:
            outs = op.operands[-1]
            if naive:
                new_buf = b.emit("memref.alloc", (), [MEMREF_TY])
                outs_new = new_buf.results[0]
            elif decs[i].kind == "out-of-place":
                new_buf = b.emit("memref.alloc", (), [MEMREF_TY])
                outs_new = new_buf.results[0]
            else:
                outs_new = valmap[outs]                    # 就地：复用已有 buffer
            ins_new = [valmap[v] for v in op.operands[:-1]]
            b.emit(op.name, ins_new + [outs_new])          # memref 语义：没有结果
            valmap[op.results[0]] = outs_new
        elif op.name == "func.return":
            b.emit("func.return", [valmap[op.operands[0]]])
        else:
            raise TransformFailure("迷你 bufferize 不认识 op：%s" % op.name)
    return m


# ==== 参考实现（纯 numpy，与被测代码路径无关） ====


def reference_chain(a, b, k=K):
    return (a + k) * b


def reference_conflict(a, b, k=K):
    t1 = np.full(a.shape, k)
    t2 = a + b
    return t1 * t2


def run_func(mod, args):
    return Interp().run(mod, args)


# ==== 跑 ====

checks = Checks()
tensor_mod = build_tensor_ir()
tensor_fn = tensor_mod.body.ops[0]

rule("1. tensor 语义：存储没有被指定")
print(print_ir(tensor_mod))
print()
print("官方定义（来源见 code/specs.py）：")
print("  「%s」" % SPECS.BUFFERIZATION_DEF)

decisions = analyze(tensor_fn)
print()
print("%-4s %-14s %-10s %s" % ("pos", "op", "outs", "决策"))
print("-" * 56)
for d in decisions:
    print("%-4d %-14s %-10s %s" % (d.idx, d.op.name, d.outs.name, d.kind))

memref_mod = bufferize(tensor_fn)
print()
print("bufferize 之后：")
print(print_ir(memref_mod))

# ==== A 类 ====

rule("2. A 算法正确性：解释器 vs 独立 numpy 参考实现")

ref = reference_chain(A, B)
got_tensor = run_func(tensor_mod, [A, B])
got_memref = run_func(memref_mod, [A, B])
got_memref = got_memref.data if isinstance(got_memref, MemRef) else got_memref

e_tensor = max_err(got_tensor, ref)
e_memref = max_err(got_memref, ref)
e_cross = max_err(got_tensor, got_memref)

print("参考实现 reference_chain = (a + k) * b")
print("tensor 语义结果 vs 参考：最大误差 %.3e" % e_tensor)
print("memref 语义结果 vs 参考：最大误差 %.3e" % e_memref)
print("两个语义互相之差：      %.3e" % e_cross)
print("结果样例：%s" % np.array2string(np.asarray(ref).ravel(), precision=3))

checks.A(e_tensor == 0.0, "tensor 语义执行结果 == 独立 numpy 参考实现",
         "最大误差 %.3e" % e_tensor)
checks.A(e_memref == 0.0, "memref 语义执行结果 == 独立 numpy 参考实现",
         "最大误差 %.3e" % e_memref)
# 「变换前后语义等价」断言的是**成功**行为，按 CONTRIBUTING §2 归 A 类，不归 C 类
checks.A(e_cross == 0.0, "bufferize 前后语义等价（tensor 结果 == memref 结果）",
         "最大误差 %.3e" % e_cross)

naive_mod = bufferize(tensor_fn, naive=True)
got_naive = run_func(naive_mod, [A, B])
checks.A(max_err(got_naive.data, ref) == 0.0,
         "朴素（每个结果各分配一块）的版本语义同样正确",
         "最大误差 %.3e" % max_err(got_naive.data, ref))

# ==== B 类 ====

rule("3. B 代价模型：物化次数与字节数")

ta = count_ops(tensor_mod.region)          # 不含 module op 自己
ma = count_ops(memref_mod.region)
na = count_ops(naive_mod.region)
elem_bytes = 4

n_tensor_results = sum(1 for op in body_ops(tensor_fn)
                       if op.results and op.name != "arith.constant")
alloc_oneshot = ma.get("memref.alloc", 0)
alloc_naive = na.get("memref.alloc", 0)
alloc_outofplace = sum(d.allocs for d in decisions) + 1      # +1 是 tensor.empty 那次

print("%-34s %10s %10s" % ("项", "本机值", "文档值"))
print("-" * 58)
b_rows = [
    ("tensor 语义下的 tensor 结果数", n_tensor_results, 4),
    ("朴素 bufferize 的 alloc 次数", alloc_naive, 4),
    ("one-shot 的 alloc 次数", alloc_oneshot, 1),
    ("就地决策数（in-place）", sum(1 for d in decisions if d.kind == "in-place"), 2),
    ("tensor IR 的 op 数", sum(ta.values()), 7),
    ("one-shot memref IR 的 op 数", sum(ma.values()), 7),
    ("朴素 memref IR 的 op 数", sum(na.values()), 10),
    ("单块缓冲字节数", elem_bytes * int(np.prod(SHAPE)), 24),
    ("朴素总分配字节数", alloc_naive * elem_bytes * int(np.prod(SHAPE)), 96),
    ("one-shot 总分配字节数", alloc_oneshot * elem_bytes * int(np.prod(SHAPE)), 24),
]
for name, got, want in b_rows:
    ok = (got == want)
    print("%-34s %10s %10s  %s" % (name, got, want, "ok" if ok else "MISMATCH"))
    checks.B(ok, "文档表：%s == %s" % (name, want), "本机 %s" % got)

print()
print("就地判据省下的分配：%d → %d（省 %d 次，%.0f%%）"
      % (alloc_naive, alloc_oneshot, alloc_naive - alloc_oneshot,
         100.0 * (alloc_naive - alloc_oneshot) / alloc_naive))
checks.B(alloc_outofplace == alloc_oneshot,
         "物化次数可由决策表独立算出（不依赖 bufferize 的实现）",
         "决策表算出 %d，bufferize 结果 %d" % (alloc_outofplace, alloc_oneshot))

# ==== C 类 ====

rule("4. C 边界与反例：判据必须拒绝就地，且拒绝是有后果的")

conflict_fn = build_conflict_ir().body.ops[0]
cbody = body_ops(conflict_fn)
c_dec = {d.idx: d for d in analyze(conflict_fn)}
c_t2 = next(d for d in c_dec.values() if d.op.name == "linalg.add")
print("反例地形（t1 在 t3 里还被读到）：")
for i, op in enumerate(cbody):
    print("  %d: %-14s ins=%s outs=%s" % (
        i, op.name, [v.name for v in op.operands[:-1]],
        op.operands[-1].name if op.operands else "-"))

# C1：判据必须把 t2 判为 out-of-place
checks.C(c_t2.kind == "out-of-place",
         "C1 判据拒绝就地：outs 的值在本 op 之后还有读者",
         "linalg.add 的决策 = %s" % c_t2.kind)

# C1 的对照：把那个后续读者去掉，同一判据必须放行
fixed = build_tensor_ir().body.ops[0]
checks.expect_ok(lambda: analyze(fixed),
                 "C1 对照：没有后续读者时，就地分析正常完成并判为 in-place",
                 "%s" % [d.kind for d in analyze(fixed)])

# C2：无视判据强制就地 → 数值真的算错
unsafe_mod = bufferize(conflict_fn, force_inplace=True)
got_unsafe = run_func(unsafe_mod, [A, B])
got_unsafe = got_unsafe.data if isinstance(got_unsafe, MemRef) else got_unsafe
ref_conf = reference_conflict(A, B)
e_unsafe = max_err(got_unsafe, ref_conf)
print()
print("强制就地后：最大误差 %.3e（参考值首元素 %.3f，算出来 %.3f）"
      % (e_unsafe, ref_conf[0, 0], got_unsafe[0, 0]))
checks.C(e_unsafe > 0.0,
         "C2 强制就地**真的**算错——判据不是摆设",
         "最大误差 %.3e > 0" % e_unsafe)

# C2 的对照：不强制就地时，同一地形算对
safe_mod = bufferize(conflict_fn)
got_safe = run_func(safe_mod, [A, B])
got_safe = got_safe.data if isinstance(got_safe, MemRef) else got_safe
checks.A(max_err(got_safe, ref_conf) == 0.0,
         "C2 对照：走判据这条路，同一地形算对",
         "最大误差 %.3e" % max_err(got_safe, ref_conf))

# C3：写冲突必须被拒
checks.expect_raises(lambda: analyze(build_double_write_ir().body.ops[0]),
                     TransformFailure, "C3 一个值被两个 op 当 outs → 拒绝")

# C4：可证伪性反证——把 a 类对照的「正确实现」换成「永远就地」的实现，
#     A 类判据必须报错。这就是「故意改错，程序会报错吗」的直接答案。
rule("5. 可证伪性反证")


def a_class_would_pass(mod, args, ref):
    out = run_func(mod, args)
    out = out.data if isinstance(out, MemRef) else out
    return max_err(out, ref) == 0.0


checks.C(not a_class_would_pass(unsafe_mod, [A, B], ref_conf),
         "反证：把判据改成「永远就地」，A 类等价性判据会失败",
         "等价性检查 %s" % ("通过（反证失败）" if a_class_would_pass(
             unsafe_mod, [A, B], ref_conf) else "失败（反证成立）"))
checks.A(a_class_would_pass(safe_mod, [A, B], ref_conf),
         "同一判据在正确实现上放行", "最大误差 %.3e" % max_err(got_safe, ref_conf))

rule("6. 断言汇总")
checks.summary()
