"""04 编写 pass：pattern rewrite 与不动点。

对应文档：60-compilers/mlir/04-writing-a-pass.md

★ 本程序跑的是本仓库的**迷你 greedy rewrite driver**，不是 MLIR 的
  `applyPatternsGreedily`。规则表与收益值是本仓库自己的设定；只有**默认值**
  （maxIterations=10 / fold=true / useTopDownTraversal=false /
  regionSimplificationLevel=Aggressive）与官方一致，逐条出处见 specs.py。
  输出的一切数字都是**本机实测值**。

核心失败模式在这里被真的跑出来：**达到迭代上限而停 = 没收敛**，
driver 必须报「未收敛」，不能静默返回。用法：python3 04_pattern_rewrite.py
"""

import collections
import random
import sys

import exprlang as EL
import specs as SPECS
from miniir import (IRBuilder, Interp, Module, NonConvergenceError, Op, Checks,
                    const_value, count_ops, dead_code_elimination, is_const_op,
                    max_err, mk_const, print_ir, rule, splice_op)

# 独立参照物（求值器 / 参考简化器 / 随机地形）在 exprlang.py 里，
# 它与本文件的 IR 路径完全解耦。
X, SEED, ev, simplify_expr, rand_expr = EL.X, EL.SEED, EL.ev, EL.simplify_expr, \
    EL.rand_expr

# ==== 规则表 ================================================================


class Pattern:
    """一条 rewrite 规则：benefit 决定它在贪心驱动里被挑中的优先级。"""

    def __init__(self, name, benefit, match, apply):
        self.name = name
        self.benefit = benefit
        self.match = match          # (body, op) -> bool
        self.apply = apply          # (body, op) -> None

    def __repr__(self):
        return "Pattern(%s, benefit=%d)" % (self.name, self.benefit)



def build_patterns(with_fold=True, with_wrap=False, with_oscillate=False):
    """规则表。IR 编辑原语（splice_op / replace_uses / dead_code_elimination 等）
    来自共享模块 miniir.py——它们是通用 IR 工具，不是本文的被测对象。

    ★ 地形开关：with_wrap 与 with_oscillate 只用于 C 类反例，
      它们的构造规则**先于结果写死**，不看结果调参。
    """
    pats = []

    def fold(kind):
        op_name = "arith.addf" if kind == "add" else "arith.mulf"

        def match(body, op):
            return (op.name == op_name and is_const_op(op.operands[0])
                    and is_const_op(op.operands[1]))

        def apply(body, op):
            a, b = (const_value(v) for v in op.operands)
            c = mk_const(a + b if kind == "add" else a * b)
            splice_op(body, op, [c], c.results[0])
        return Pattern("fold_" + kind, 20, match, apply)

    def to_const_or_keep(op_name, val, benefit, name, mode):
        """x <op> val → x（mode="keep"）或 → val（mode="const"）。两侧都查。"""

        def match(body, op):
            return op.name == op_name and any(
                is_const_op(v) and const_value(v) == val for v in op.operands)

        def apply(body, op):
            if mode == "const":
                c = mk_const(val)
                splice_op(body, op, [c], c.results[0])
            else:
                keep = next(v for v in op.operands
                            if not (is_const_op(v) and const_value(v) == val))
                splice_op(body, op, [], keep)
        return Pattern(name, benefit, match, apply)

    if with_fold:
        pats += [fold("add"), fold("mul")]
    pats.append(to_const_or_keep("arith.mulf", 0.0, 15, "mul_zero", "const"))
    pats.append(to_const_or_keep("arith.addf", 0.0, 10, "add_zero", "keep"))
    pats.append(to_const_or_keep("arith.mulf", 1.0, 10, "mul_one", "keep"))

    if with_wrap:
        # C1 地形：与 add_zero 抵消的规则「x → x + 0」。benefit 更高（30 > 10），
        # 造一个新 op 包住旧结果并把使用改指向它（**旧 op 保留**，否则悬空）。
        def wrap_match(body, op):
            return op.name == "arith.addf" and bool(op.results)

        def wrap_apply(body, op):
            zero = mk_const(0.0)
            new = Op("arith.addf", [op.results[0], zero.results[0]], ["f32"])
            splice_op(body, op, [zero, new], new.results[0], drop_old=False)
        pats.append(Pattern("add_zero_reverse", 30, wrap_match, wrap_apply))

    if with_oscillate:
        # C2 地形：两条**互逆且规模不增长**的规则（benefit 5 > 4）——
        # mulf(x,2) ↔ addf(x,x) 来回换，永远不收敛，但 IR 不会撑爆。
        def to_add_match(body, op):
            return (op.name == "arith.mulf" and is_const_op(op.operands[0])
                    and const_value(op.operands[0]) == 2.0)

        def to_add_apply(body, op):
            new = Op("arith.addf", [op.operands[1], op.operands[1]], ["f32"])
            splice_op(body, op, [new], new.results[0])

        def to_mul_match(body, op):
            return op.name == "arith.addf" and op.operands[0] is op.operands[1]

        def to_mul_apply(body, op):
            two = mk_const(2.0)
            new = Op("arith.mulf", [two.results[0], op.operands[0]], ["f32"])
            splice_op(body, op, [two, new], new.results[0])
        pats.append(Pattern("mul2_to_add", 5, to_add_match, to_add_apply))
        pats.append(Pattern("add_to_mul2", 4, to_mul_match, to_mul_apply))
    return pats


# ==== 贪心驱动 ==============================================================


class GreedyRewriteConfig:
    """默认值取自官方源码（GreedyPatternRewriteDriver.h，取用日期见 specs.py）。"""

    def __init__(self, max_iterations=None, fold=None, use_top_down=None):
        d = SPECS.GREEDY_CONFIG_DEFAULTS
        self.maxIterations = d["maxIterations"] if max_iterations is None else max_iterations
        self.fold = d["fold"] if fold is None else fold
        self.useTopDownTraversal = (d["useTopDownTraversal"] if use_top_down is None
                                    else use_top_down)
        self.regionSimplificationLevel = d["regionSimplificationLevel"]


RewriteResult = collections.namedtuple(
    "RewriteResult", "converged iterations applied")


def rstr(r):
    return "converged=%s iterations=%d applied=%d" % r


def apply_patterns_greedily(body, patterns, config=None):
    """官方语义：贪心地反复套用 benefit 最高的规则，直到**到达不动点**
    或**迭代数用尽**。用尽而停 = **没收敛**，返回值是 failure。"""
    config = config or GreedyRewriteConfig()
    patterns = [p for p in patterns if config.fold or not p.name.startswith("fold_")]
    applied_total = 0
    for it in range(1, config.maxIterations + 1):
        dead_code_elimination(body)          # 官方：匹配前先做一次简单 DCE
        applied = 0
        order = list(body.ops) if config.useTopDownTraversal else list(reversed(body.ops))
        for op in order:
            if op not in body.ops:
                continue                     # 已被本轮前面的改写删掉
            cands = [p for p in patterns if p.match(body, op)]
            if not cands:
                continue
            cands.sort(key=lambda p: (-p.benefit, p.name))
            cands[0].apply(body, op)
            applied += 1
            applied_total += 1
        if applied == 0:
            return RewriteResult(True, it, applied_total)
    return RewriteResult(False, config.maxIterations, applied_total)


# ==== 表达式 ↔ IR ===========================================================


def emit(b, e, x):
    if e[0] == "const":
        return b.emit("arith.constant", (), ["f32"], attrs={"value": e[1]}).results[0]
    if e[0] == "var":
        return x
    return b.emit("arith.addf" if e[0] == "add" else "arith.mulf",
                  [emit(b, e[1], x), emit(b, e[2], x)], ["f32"]).results[0]


def build_module(expr):
    m = Module()
    fop, b = IRBuilder(m.body).func("f", ["f32"])
    b.terminate("func.return", [emit(b, expr, fop.regions[0].blocks[0].args[0])])
    return m


def ir_to_expr(v):
    """从 IR 读回表达式树。Value 只有两种来源：某个 op 的结果，或 block 参数
    （后者就是函数形参 %x）——这是 LangRef 对 Value 的定义。"""
    owner, name = v.owner, getattr(v.owner, "name", None)
    if owner is not None and name is None:
        return ("var", "x")
    if name == "arith.constant":
        return ("const", owner.attrs["value"])
    if owner is None:
        raise NonConvergenceError("值 %s 没有定义者" % v.name)
    return ("add" if name == "arith.addf" else "mul",
            ir_to_expr(owner.operands[0]), ir_to_expr(owner.operands[1]))


def body_of(mod):
    return [op for op in mod.body.ops if op.name == "func.func"][0].regions[0].blocks[0]


def result_value(mod):
    return [op for op in body_of(mod).ops if op.name == "func.return"][0].operands[0]


def count_ir_ops(mod):
    return sum(count_ops(mod.region).values())


def eval_ir(mod):
    return Interp().run(mod, [X])            # 被测路径：跑迷你解释器


# ==== 跑 ====================================================================

checks = Checks()
cfg = GreedyRewriteConfig()

rule("1. 规则表与官方默认值")
for p in build_patterns():
    print("  %-20s benefit=%d" % (p.name, p.benefit))
print("\nGreedyRewriteConfig 默认值（逐条来源见 code/specs.py）:")
for k, v in sorted(SPECS.GREEDY_CONFIG_DEFAULTS.items()):
    print("  %-28s %s" % (k, v))
print("\n  「%s」\n  「%s」" % (SPECS.GREEDY_PREMATURE, SPECS.GREEDY_RETURN))

# ---- A 类：随机地形上的一致性 ----
rule("2. A 算法正确性：driver vs 独立参考简化器（种子 %d 冻结）" % SEED)
rng = random.Random(SEED)
terms = [rand_expr(rng, 3) for _ in range(12)]
terms.append(("add", ("mul", ("var", "x"), ("const", 0.0)), ("var", "x")))
terms.append(("mul", ("add", ("var", "x"), ("const", 0.0)), ("const", 1.0)))

ok_sem = ok_struct = ok_idem = 0
worst = 0.0
for t in terms:
    mod = build_module(t)
    apply_patterns_greedily(body_of(mod), build_patterns())
    want = simplify_expr(t)
    e = max_err(eval_ir(mod), ev(want, X))
    worst = max(worst, e)
    ok_sem += (e == 0.0)
    ok_struct += (ir_to_expr(result_value(mod)) == want)
    again = apply_patterns_greedily(body_of(mod), build_patterns())
    ok_idem += (again.applied == 0 and again.converged)

print("地形数：%d（12 个由种子 %d 生成 + 2 个手工构造）" % (len(terms), SEED))
print("语义一致（求值）：%d / %d，最大误差 %.3e" % (ok_sem, len(terms), worst))
print("结构一致（读回表达式树逐节点比对）：%d / %d" % (ok_struct, len(terms)))
checks.A(ok_sem == len(terms), "driver 输出与独立参考简化器**求值相同**",
         "%d/%d，最大误差 %.3e" % (ok_sem, len(terms), worst))
checks.A(ok_struct == len(terms), "driver 输出与独立参考简化器**结构相同**",
         "%d/%d" % (ok_struct, len(terms)))
checks.A(ok_idem == len(terms), "幂等：对第一遍的输出再跑一次，改写次数为 0",
         "%d/%d" % (ok_idem, len(terms)))
checks.A(all(max_err(eval_ir(build_module(t)), ev(t, X)) == 0.0 for t in terms),
         "改写前的 IR 与表达式树求值器一致（参照物不经过 IR）", "%d 个地形" % len(terms))

# ---- B 类 ----
rule("3. B 代价模型：迭代次数、改写次数、IR 规模")
t_big = ("add", ("add", ("add", ("mul", ("var", "x"), ("const", 1.0)), ("const", 0.0)),
                 ("const", 0.0)), ("mul", ("const", 2.0), ("const", 3.0)))
mod_big = build_module(t_big)
n_before = count_ir_ops(mod_big)
res_big = apply_patterns_greedily(body_of(mod_big), build_patterns())
n_after = count_ir_ops(mod_big)
print(print_ir(mod_big))
print("\n%-26s %10s %10s" % ("项", "本机值", "文档值"))
for name, got, want in [
        ("改写前 op 数", n_before, 12), ("改写后 op 数", n_after, 4),
        ("改写后收敛", int(res_big.converged), 1),
        ("用到的迭代数", res_big.iterations, 2),
        ("累计改写次数", res_big.applied, 4),
        ("maxIterations 默认值", cfg.maxIterations, 10)]:
    print("%-26s %10s %10s  %s" % (name, got, want, "ok" if got == want else "MISMATCH"))
    checks.B(got == want, "文档表：%s == %s" % (name, want), "本机 %s" % got)
checks.B(n_after < n_before, "改写让 IR 变小", "%d → %d" % (n_before, n_after))


# ---- C 类 ----
def c1_predicate(r):
    """C1 判据的形式化：**未收敛**，且在 maxIterations 处停下。"""
    return (not r.converged) and r.iterations == cfg.maxIterations


def strict_driver(mod, patterns, config=None):
    r = apply_patterns_greedily(body_of(mod), patterns, config)
    if not r.converged:
        raise NonConvergenceError("贪心改写在上限 %d 轮内没有到达不动点" % r.iterations)
    return r


rule("4. C 边界与反例：不收敛必须被检出，且不是上限造出来的")
c1 = ("add", ("var", "x"), ("const", 0.0))
mod_c1 = build_module(c1)
n_c1a = count_ir_ops(mod_c1)
res_c1 = apply_patterns_greedily(body_of(mod_c1), build_patterns(with_wrap=True))
checks.C(c1_predicate(res_c1), "C1 `x+0→x` 与 `x→x+0` 同时存在 → 报「未收敛」",
         "%s；IR op 数 %d → %d（不收敛的 pass 还会持续改写）"
         % (rstr(res_c1), n_c1a, count_ir_ops(mod_c1)))
res_c1b = apply_patterns_greedily(body_of(build_module(c1)), build_patterns())
checks.A(res_c1b.converged, "C1 对照：去掉 `x→x+0` 后，同一地形收敛",
         "%s" % rstr(res_c1b))

mod_c2 = build_module(("add", ("var", "x"), ("var", "x")))
n_c2a = count_ir_ops(mod_c2)
res_c2 = apply_patterns_greedily(body_of(mod_c2), build_patterns(with_oscillate=True),
                                 GreedyRewriteConfig(max_iterations=500))
print("\nC2 地形 `addf(x, x)`：%s；IR op 数 %d → %d（规模基本不变）"
      % (rstr(res_c2), n_c2a, count_ir_ops(mod_c2)))
checks.C(not res_c2.converged,
         "C2 上限抬到 500 仍不收敛——不收敛是规则集本身的性质，不是上限太小",
         "%s" % rstr(res_c2))

fake = RewriteResult(True, 0, 0)          # 「永远声称收敛」的实现会返回的东西
checks.C(not c1_predicate(fake),
         "C3 反证：把收敛判定改成永远 True，C1 的判据立即判为失败",
         "假 driver %s → c1_predicate = %s；真 driver → %s"
         % (rstr(fake), c1_predicate(fake), c1_predicate(res_c1)))
checks.expect_raises(lambda: strict_driver(build_module(c1),
                                           build_patterns(with_wrap=True)),
                     NonConvergenceError,
                     "C4 把「未收敛」升级成异常后，同一地形真的抛异常")
checks.expect_ok(lambda: strict_driver(build_module(c1), build_patterns()),
                 "C4 对照：去掉 `x→x+0` 后，同一个 driver 正常返回")

# ---- 边界：不动点恰好落在迭代上限两侧 ----
rule("5. 边界：迭代上限两侧各站一个地形")


def peel_patterns():
    """`addf(a, 0) → a`，但**只有当 a 本身不是 addf 时**才匹配。配合默认的
    bottom-up 遍历，左嵌套链每轮只能剥掉一层——这才让「用掉几轮」可测。"""
    def match(body, op):
        if op.name != "arith.addf":
            return False
        a, b = op.operands
        return (is_const_op(b) and const_value(b) == 0.0
                and getattr(a.owner, "name", None) != "arith.addf")

    def apply(body, op):
        splice_op(body, op, [], op.operands[0])
    return [Pattern("peel_one_layer", 10, match, apply)]


def peel_chain(n):
    e = ("var", "x")
    for _ in range(n):
        e = ("add", e, ("const", 0.0))
    return e


boundary = {}
for n in (8, 9, 10, 11):
    boundary[n] = apply_patterns_greedily(
        body_of(build_module(peel_chain(n))), peel_patterns(),
        GreedyRewriteConfig(use_top_down=False))
print("%-10s %-8s %-8s %s" % ("嵌套层数", "收敛", "轮数", "改写次数"))
print("-" * 38)
for n, r in sorted(boundary.items()):
    print("%-10d %-8s %-8d %d" % (n, "是" if r.converged else "否", r.iterations, r.applied))
ok_side = sorted(n for n, r in boundary.items() if r.converged)
bad_side = sorted(n for n, r in boundary.items() if not r.converged)
checks.C(bool(bad_side) and max(ok_side) + 1 == min(bad_side),
         "C5 边界两侧相邻：收敛的最大层数与不收敛的最小层数差 1",
         "收敛 %s / 不收敛 %s" % (ok_side, bad_side))
checks.C(min(bad_side) == cfg.maxIterations,
         "C5 不收敛的起点恰好是 maxIterations——上限不是装饰",
         "min(不收敛层数) = %d，maxIterations = %d" % (min(bad_side), cfg.maxIterations))
checks.B(min(bad_side) == cfg.maxIterations,
         "文档表：不收敛的起点 == maxIterations（%d）" % cfg.maxIterations,
         "不收敛从 %d 层开始" % min(bad_side))

rule("6. 断言汇总")
checks.summary()
