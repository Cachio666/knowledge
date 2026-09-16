"""03 渐进式 lowering：每一层保留什么、丢什么。对应文档 03-progressive-lowering.md

★ 本程序跑的是本仓库的**迷你 lowering 管线**，不是 MLIR 的 `mlir-opt`。
  pass 名借用官方名字以便对照，实现是本仓库的简化版（见 miniir.py 开头）。
  输出的一切数字都是**本机实测值**。

主线：**tensor → memref 的转换不是预处理，而是并行化的前置条件。**
链路：S0 tensor 语义 → S1 memref 语义 → S2 `scf.for` 嵌套 / S3 `scf.parallel`
→ S4 `scf.forall` → S5 映射到 block。每步都解释执行一次并与独立 numpy
参考实现比对。用法：python3 03_lowering_pipeline.py
"""

import sys

import numpy as np

import specs as SPECS
from miniir import (IRBuilder, Interp, MemRef, Module, Op, TransformFailure, Checks,
                    count_ops, max_depth, max_err, print_ir, rule, walk_ops)

SHAPE = (2, 3)
TENSOR_TY = "tensor<2x3xf32>"
MEMREF_TY = "memref<2x3xf32>"
K = 1.5
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

LINALG = frozenset({"linalg.fill", "linalg.add", "linalg.mul"})
ARITH_BIN = {"linalg.add": "arith.addf", "linalg.mul": "arith.mulf"}


def reference(a, b, k=K):        # 独立参考实现：纯 numpy，不经过任何 IR
    return (a + k) * b

# ==== 小工具 ====


class Emitter:
    """把新造的 op 收进一个列表（不立刻挂到 block 上），便于整体替换。"""

    def __init__(self, pool):
        self.ops, self.pool = [], pool

    def emit(self, name, operands=(), result_types=(), attrs=None, num_regions=0):
        op = Op(name, operands, result_types, attrs, num_regions)
        self.ops.append(op)
        return op

    def index(self, n):
        """index 常量按值复用——真实管线里由 canonicalize / CSE 干这件事。"""
        if n not in self.pool:
            self.pool[n] = self.emit("arith.constant", (), ["index"],
                                     attrs={"value": n}).results[0]
        return self.pool[n]


def func_of(mod):
    return [op for op in walk_ops(mod) if op.name == "func.func"][0]


def body_of(mod):
    return func_of(mod).regions[0].blocks[0]     # 迷你实现只处理单块函数体


def replace_op(block, old, new_ops):
    """block 里的 old 换成 new_ops（一个或多个）。"""
    i = block.ops.index(old)
    block.ops[i:i + 1] = new_ops
    old.parent = None
    for op in new_ops:
        op.parent = block


# ==== S0：tensor 语义 ====


def stage0_tensor():
    m = Module()
    fop, b = IRBuilder(m.body).func("main", [TENSOR_TY, TENSOR_TY])
    a, bb = fop.regions[0].blocks[0].args
    cst = b.emit("arith.constant", (), ["f32"], attrs={"value": K})
    e0 = b.emit("tensor.empty", (), [TENSOR_TY])
    t1 = b.emit("linalg.fill", [cst.results[0], e0.results[0]], [TENSOR_TY])
    t2 = b.emit("linalg.add", [a, t1.results[0], t1.results[0]], [TENSOR_TY])
    t3 = b.emit("linalg.mul", [t2.results[0], bb, t2.results[0]], [TENSOR_TY])
    b.terminate("func.return", [t3.results[0]])
    return m


# ---------------------------------------------------------------------------
# S1：bufferize（这里用**朴素**版本：每个 tensor 结果各一块缓冲）
#
# 为什么这里不用 02 篇的就地分析：本文的主线是 lowering 链本身，不是物化次数的
# 优化。朴素版让每一层「保留了几个张量」这件事看得更清楚。
# ---------------------------------------------------------------------------


def stage1_memref(tensor_mod):
    src = body_of(tensor_mod)
    args = func_of(tensor_mod).regions[0].blocks[0].args
    m = Module()
    _, b = IRBuilder(m.body).func("main", [str(v.ty.with_kind("memref")) for v in args])
    valmap = dict(zip(args, b.block.args))
    for op in src.ops:
        if op.name == "arith.constant":
            new = b.emit("arith.constant", (), [str(op.results[0].ty)], attrs=op.attrs)
            valmap[op.results[0]] = new.results[0]
        elif op.name == "tensor.empty":
            valmap[op.results[0]] = b.emit("memref.alloc", (), [MEMREF_TY]).results[0]
        elif op.name in LINALG:
            buf = b.emit("memref.alloc", (), [MEMREF_TY]).results[0]
            ins = [valmap[v] for v in op.operands[:-1]]
            b.emit(op.name, ins + [buf])              # memref 语义：没有结果
            valmap[op.results[0]] = buf
        elif op.name == "func.return":
            b.terminate("func.return", [valmap[op.operands[0]]])
        else:
            raise TransformFailure("stage1 不认识 op：%s" % op.name)
    return m

# ==== S2 / S3 / S4：linalg → 循环 ====


def _scalar_body(blk, op, ivs):
    """把 elementwise linalg op 的内层标量运算写进 blk。"""
    b = IRBuilder(blk)
    out = op.operands[-1]
    if op.name == "linalg.fill":
        b.emit("memref.store", [op.operands[0], out] + list(ivs))
    else:
        l0 = b.emit("memref.load", [op.operands[0]] + list(ivs), ["f32"]).results[0]
        l1 = b.emit("memref.load", [op.operands[1]] + list(ivs), ["f32"]).results[0]
        s = b.emit(ARITH_BIN[op.name], [l0, l1], ["f32"]).results[0]
        b.emit("memref.store", [s, out] + list(ivs))
    blk.append(Op("scf.yield"))


def _loop_nest(em, op, kind, pool):
    """kind ∈ {"for", "parallel", "forall"}：三种循环结构，语义相同。"""
    shp = op.operands[-1].ty.shape
    ivs = []
    if kind in ("parallel", "forall"):
        # 常量全部先发，保证它们支配循环体（迷你实现不做自动提升）
        bounds = []
        for d in shp:
            bounds += [em.index(0), em.index(d), em.index(1)]
        loop = em.emit("scf." + kind, bounds, num_regions=1)
        blk = loop.regions[0].new_block("^bb_%s" % kind)
        ivs = [blk.add_arg("index", "%%i%d" % d) for d in range(len(shp))]
        _scalar_body(blk, op, ivs)
        return
    bounds = [(em.index(0), em.index(d), em.index(1)) for d in shp]
    first, cur_block = None, None
    for d, (lb, ub, st) in enumerate(bounds):
        loop = Op("scf.for", [lb, ub, st], num_regions=1)
        blk = loop.regions[0].new_block("^bb_for%d" % d)
        ivs.append(blk.add_arg("index", "%%k%d" % d))
        if d == 0:
            first = loop
        else:
            cur_block.append(loop)
        cur_block = blk
    _scalar_body(cur_block, op, ivs)
    em.ops.append(first)


def lower_linalg_to_loops(mod, kind, check_precondition=True):
    """S2 = `-convert-linalg-to-loops`（scf.for）；
    S3 = `-convert-linalg-to-parallel-loops`；S4 = transform 方言的 tile_using_forall。

    ★ 官方 `-convert-linalg-to-loops` 页写明：**Pre-condition: the operands used by
      the linalg ops have buffer semantics**。本函数先检查这一点，不满足就失败。

    `check_precondition=False` 只给 C 类反例用：它造出「忘了 bufferize 就急着 lower」
    的那份畸形 IR，用来证明检查本身在起作用（见 C1 / C4）。
    """
    body = body_of(mod)
    pool = {}
    for op in list(body.ops):
        if op.name not in LINALG:
            continue
        if check_precondition:
            for v in op.operands:
                if v.ty.is_tensor:
                    raise TransformFailure(
                        "前条件是 buffer 语义，但 %s 的 operand %s 还是 %s"
                        % (op.name, v.name, v.ty))
        em = Emitter(pool)
        _loop_nest(em, op, kind, pool)
        replace_op(body, op, em.ops)
    return mod


def pipeline(kind):
    """每次从 S0 重新跑一遍——真实管线也是从源 IR 重跑，不是接着上一阶段的内存改。

    kind ∈ {"for", "parallel", "forall", "gpu"}：前者是三种循环 lowering，
    "gpu" 是「forall 之后再映射到 block」的两段式。
    """
    if kind == "gpu":
        return map_forall_to_blocks(
            lower_linalg_to_loops(stage1_memref(stage0_tensor()), "forall"))
    return lower_linalg_to_loops(stage1_memref(stage0_tensor()), kind)

# ==== S5：`transform.gpu.map_forall_to_blocks` ====


def map_forall_to_blocks(mod):
    """把顶层 `scf.forall` 改写成 `gpu.launch`（映射到 block）。

    ★ 官方原文：「**Only bufferized scf.forall are currently supported.**
      If any scf.forall with **tensors** is found, the transform definitely fails.」
      —— buffer 语义是这条变换的硬前置条件，不是优化选项。
    """
    found = False
    for op in list(walk_ops(mod)):
        if op.name != "scf.forall":
            continue
        found = True
        for inner in walk_ops(op.regions[0]):
            for v in list(inner.operands) + list(inner.results):
                if v.ty.is_tensor:
                    raise TransformFailure(
                        "map_forall_to_blocks 只支持 bufferized scf.forall；"
                        "发现 %s 里有 tensor 值 %s" % (inner.name, v.name))
        launch = Op("gpu.launch", list(op.operands), num_regions=1)
        launch.regions[0].blocks = op.regions[0].blocks
        for blk in launch.regions[0].blocks:
            blk.parent = launch.regions[0]
        launch.regions[0].parent = launch
        replace_op(op.parent, op, [launch])
    if not found:
        raise TransformFailure("模块里没有 scf.forall，map_forall_to_blocks 无事可做")
    return mod


def gpu_lower_to_nvvm(mod):
    """`-gpu-lower-to-nvvm-pipeline`：把已有的并行结构 lower 到 NVVM。

    ★ 官方原文：「This pipeline **specifically requires explicitly parallel IR and
      doesn't do GPU parallelization**. To enable parallelism, necessary
      transformations must be applied before utilizing this pipeline.」
      因此它**检查**模块里已经有显式并行结构；只有顺序 `scf.for` 时它做不了事。
    """
    parallel = [op for op in walk_ops(mod)
                if op.name in ("scf.parallel", "scf.forall", "gpu.launch")]
    if not parallel:
        raise TransformFailure(
            "gpu-lower-to-nvvm-pipeline requires explicitly parallel IR and "
            "doesn't do GPU parallelization")
    return mod, sorted({op.name for op in parallel})

# ==== 执行与检查 ============================================================


def run_traced(mod):
    """跑一遍迷你解释器，返回 (结果数组, 解释器状态)。"""
    it = Interp()  # noqa: E501
    out = it.run(mod, [A, B])
    return (out.data if isinstance(out, MemRef) else out), it


def n_ops(mod, name=None):
    return count_ops(mod.region, name)


def total_ops(mod):
    return sum(count_ops(mod.region).values())


checks = Checks()
ref = reference(A, B)

# ==== 逐层打印 ==============================================================
rule("1. S0 tensor 语义")
s0 = stage0_tensor()
print(print_ir(s0))
print("官方前置条件原文：\n  「%s」" % SPECS.LINALG_TO_LOOPS)

rule("2. S1 memref 语义（bufferize 之后）")
s1 = stage1_memref(s0)
print(print_ir(s1))

rule("3. S2 / S3：两种循环结构")
s2, s3 = pipeline("for"), pipeline("parallel")
print("S2（-convert-linalg-to-loops）:\n%s" % print_ir(s2))
print("\nS3（-convert-linalg-to-parallel-loops）——**并行循环，不是 scf.forall**：")
print(print_ir(s3))

rule("4. S4 / S5：for-all 与映射到 block")
s4, s5 = pipeline("forall"), pipeline("gpu")
print("S4（transform.structured.tile_using_forall 产出的 scf.forall）:\n%s"
      % print_ir(s4))
print("\nS5（transform.gpu.map_forall_to_blocks → gpu.launch）:\n%s" % print_ir(s5))

# ==== A 类：每一层都解释执行一次，与独立参考实现比对 ========================
rule("5. A 算法正确性：六个阶段 vs 独立 numpy 参考实现")
stages = [("S0 tensor 语义", s0), ("S1 memref 语义", s1), ("S2 scf.for 嵌套", s2),
          ("S3 scf.parallel", s3), ("S4 scf.forall", s4), ("S5 gpu.launch", s5)]
print("%-22s %14s %12s %s\n%s"
      % ("阶段", "最大误差", "store 次数", "执行 op 数", "-" * 66))
errors = {}
for name, mod in stages:
    got, it = run_traced(mod)
    errors[name] = max_err(got, ref)
    print("%-22s %14.3e %12d %d" % (name, errors[name],
                                    it.executed.count("memref.store"), len(it.executed)))
    checks.A(errors[name] == 0.0,
             "阶段 %s 的执行结果 == 独立 numpy 参考实现" % name,
             "最大误差 %.3e" % errors[name])

checks.A(all(v == 0.0 for v in errors.values()),
         "六个阶段全部与参考实现一致（顺序路径与两条并行路径结果相同）",
         "最大误差 %s" % {k: ("%.1e" % v) for k, v in errors.items()})
checks.A(max_depth(s3) == max_depth(s4),
         "S3 与 S4 的嵌套深度相同：单层多维循环（只是种类不同）",
         "S3=%d S4=%d" % (max_depth(s3), max_depth(s4)))
checks.A(max_depth(s2) > max_depth(s3),
         "S2 比 S3 深：`scf.for` 是**嵌套**的，`scf.parallel` 是**单层多维**的",
         "S2=%d S3=%d" % (max_depth(s2), max_depth(s3)))

# ==== B 类 ==================================================================
rule("6. B 代价模型：各阶段的 op 数与循环结构")
print("%-32s %10s %10s\n%s" % ("项", "本机值", "文档值", "-" * 56))
for name, got, want in [
        ("S0 tensor 语义的 op 数", total_ops(s0), 7),
        ("S1 memref 语义的 op 数", total_ops(s1), 10),
        ("S1 的 memref.alloc 次数", n_ops(s1, "memref.alloc"), 4),
        ("S2 的 scf.for 个数", n_ops(s2, "scf.for"), 6),
        ("S3 的 scf.parallel 个数", n_ops(s3, "scf.parallel"), 3),
        ("S4 的 scf.forall 个数", n_ops(s4, "scf.forall"), 3),
        ("S5 的 gpu.launch 个数", n_ops(s5, "gpu.launch"), 3),
        ("最内层循环体迭代次数（单次）", int(np.prod(SHAPE)), 6),
        ("S5 的 op 嵌套深度", max_depth(s5), 3)]:
    print("%-32s %10s %10s  %s" % (name, got, want, "ok" if got == want else "MISMATCH"))
    checks.B(got == want, "文档表：%s == %s" % (name, want), "本机 %s" % got)

n_store2 = run_traced(s2)[1].executed.count("memref.store")
n_store5 = run_traced(s5)[1].executed.count("memref.store")
checks.B(n_store2 == A.size * 3,
         "顺序路径执行 3×A.size 次 store（3 个 linalg op 各一趟 2×3 的循环）",
         "%d 次 vs %d" % (n_store2, A.size * 3))
checks.B(n_store5 == n_store2,
         "gpu 映射路径执行同样多次 store（并行性不改变元素级工作量）",
         "%d 次 vs %d 次" % (n_store5, n_store2))

# ==== C 类 ==================================================================
rule("7. C 边界与反例：三条前置条件都必须真的挡住")


def make_tensor_forall():
    """造一份**没 bufferize 过就 lower** 的畸形 IR：region 里还带着 tensor 值。
    这就是「忘了先 bufferize」会得到的东西——正是官方警告的那份 IR。"""
    return lower_linalg_to_loops(stage0_tensor(), "forall", check_precondition=False)



checks.expect_raises(lambda: map_forall_to_blocks(make_tensor_forall()),
                     TransformFailure,
                     "C1 tensor 版 scf.forall 交给 map_forall_to_blocks → 拒绝")
checks.expect_ok(lambda: map_forall_to_blocks(pipeline("forall")),
                 "C1 对照：先 bufferize 再映射 → 成功（顺序不能颠倒）")
checks.expect_raises(lambda: gpu_lower_to_nvvm(s2), TransformFailure,
                     "C2 纯 scf.for（无显式并行）交给 gpu-lower-to-nvvm → 拒绝")
checks.expect_ok(lambda: gpu_lower_to_nvvm(s5),
                 "C2 对照：S5 有显式 gpu.launch → 通过",
                 "识别到的并行构造 %s" % gpu_lower_to_nvvm(s5)[1])
checks.expect_raises(lambda: lower_linalg_to_loops(stage0_tensor(), "for"),
                     TransformFailure,
                     "C3 tensor 语义的 linalg 交给 -convert-linalg-to-loops → 拒绝")
checks.expect_ok(lambda: lower_linalg_to_loops(stage1_memref(stage0_tensor()), "for"),
                 "C3 对照：bufferize 之后的同一 linalg → 通过")

# ==== 可证伪性反证 ==========================================================
rule("8. 可证伪性反证")


def map_forall_without_check(mod):
    """故意去掉 buffer 语义检查的版本——用来证明 C1 抓的正是那个检查。"""
    for op in list(walk_ops(mod)):
        if op.name == "scf.forall":
            launch = Op("gpu.launch", list(op.operands), num_regions=1)
            launch.regions[0].blocks = op.regions[0].blocks
            replace_op(op.parent, op, [launch])
    return mod


try:
    map_forall_without_check(make_tensor_forall())
    checks.C(True, "反证：去掉 buffer 语义检查后，同一份 tensor 版 forall **不会**失败",
             "说明 C1 抓的正是那个检查，不是别的东西")
except Exception as e:                                     # noqa: BLE001
    checks.C(False, "反证：去掉 buffer 语义检查后，同一份 tensor 版 forall **不会**失败",
             "仍然抛了 %s: %s" % (type(e).__name__, e))

checks.A(n_ops(s5, "gpu.launch") == 3,
         "S5 把 3 个 scf.forall 全部换成了 gpu.launch", "%d 个" % n_ops(s5, "gpu.launch"))

rule("9. 断言汇总")
checks.summary()
