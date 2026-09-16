"""独立参照物：表达式树上的求值器与简化器。

与 IR 路径**完全无关**——本文件不认识 miniir 的任何对象，只认嵌套元组。
它负责两件事：按固定种子生成随机地形，以及作为 04 号程序 A 类断言的
**独立参考实现**（判定方法：把这里的化简规则与被测 driver 故意改成不一致，
A 类断言必须失败）。

★★ 本文件同样属于本仓库的迷你实现，不是 MLIR 的任何组件。
"""

X = 1.25          # 求值时给变量 x 的取值
SEED = 20260916   # 地形种子：**先于结果冻结**（计划 §0.4 第 5 条）


def ev(e, x):
    """独立参考实现：直接在**表达式树**上求值，完全不碰 IR。"""
    if e[0] == "const":
        return e[1]
    if e[0] == "var":
        return x
    a, b = ev(e[1], x), ev(e[2], x)
    return a + b if e[0] == "add" else a * b


def _one_pass(e):
    """自底向上走一遍：先化简子式，再套恒等式。"""
    if e[0] in ("const", "var"):
        return e
    a, b = _one_pass(e[1]), _one_pass(e[2])
    if a[0] == "const" and b[0] == "const":
        return ("const", a[1] + b[1] if e[0] == "add" else a[1] * b[1])
    zero, one = ("const", 0.0), ("const", 1.0)
    if e[0] == "mul":
        if a == zero or b == zero:
            return zero
        if a == one:
            return b
        if b == one:
            return a
    if e[0] == "add":
        if a == zero:
            return b
        if b == zero:
            return a
    return (e[0], a, b)


def simplify_expr(e, cap=1000):
    """独立参考简化器：**递归**走表达式树，与 worklist driver 是两条路径。"""
    for _ in range(cap):
        new = _one_pass(e)
        if new == e:
            return e
        e = new
    raise RuntimeError("参考简化器在 %d 轮内没有到不动点" % cap)


def rand_expr(rng, depth):
    """按固定种子生成随机地形。"""
    if depth == 0 or rng.random() < 0.25:
        return ("var", "x") if rng.random() < 0.6 else ("const",
                                                        float(rng.randint(0, 3)))
    return ("add" if rng.random() < 0.5 else "mul",
            rand_expr(rng, depth - 1), rand_expr(rng, depth - 1))
