#!/usr/bin/env python3
"""03 自定义算子与导出：**怎么把算子接进这条链，以及图最终落到什么格式**

对应文档：70-frameworks/tensorflow/03-custom-ops-and-export.md

★★ 本程序跑的是**本仓库的迷你 TF 追踪器**（`minitf.py`），**不是 TensorFlow** ★★
  本机没有任何 TF 环境。所有数字只支撑**机制**结论；关于真实 TF 的行为断言
  以 `specs.py` 里带 `# 来源：` 的一手 URL 为准。

挂主线（§0.8）：本文处理「固化什么」的**注册接口与落地形态**那一面——
  自定义算子决定「这一步在图上占**几个节点**」；导出决定「固化下来的图**以什么格式离开进程**」。

三类断言（CONTRIBUTING §2：编号程序跑在默认 python3 上，故三类都在程序内部）：
  A —— 与**独立参考实现**比对：同一算子的三条实现路径（numpy 直算 / Python 组合 / 注册算子）、
       自定义梯度与**有限差分**参照、导出前后图与数值的一致性。
  B —— 代价模型：三条路径的图节点数、导出产物的规模、载入后的追踪次数。
  C —— 失败行为必须真的跑出来：导出把 shape 假设冻住（换 shape 即被拒）、
       未注册的算子不可调、数值不稳的梯度公式**真的**给出 NaN。

用法：cd code && make 03_custom_ops_and_export   （或 make run 全跑）
"""

import math
import os
import sys
import tempfile

import numpy as np

import minitf as T
import specs as SPECS

F32 = "float32"


# ---------------------------------------------------------------------------
# 独立参考实现：直接用 numpy 把算子算一遍（不看任何图）
# ---------------------------------------------------------------------------

def ref_swish(x):
    """`x * sigmoid(x)` 的 numpy 直接实现。sigmoid 用数值稳定的分情形写法。"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = x[~pos] * ex / (1.0 + ex)
    return out


def ref_finite_difference(fn, x, h=1e-5):
    """中心差分：**与解析梯度完全独立的**参照。"""
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def log1pexp_forward(x):
    """`log(1 + exp(x))`：本程序用来演示自定义梯度的对象。"""
    return float(np.log1p(np.exp(np.float64(x))))


def log1pexp_naive_grad(x):
    """链式法则直接写出来的梯度 `exp(x) / (1 + exp(x))`——**在 float32 下会溢出**。"""
    x = np.float32(x)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x)
        return float(e / (1.0 + e))


def log1pexp_custom_grad(x):
    """官方示例里那个化简过的梯度 `1 - 1 / (1 + exp(x))`。"""
    x = np.float64(x)
    return float(1.0 - 1.0 / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# §1 官方次序：先 Python 组合，不行再写 C++ op
# ---------------------------------------------------------------------------

def section_1():
    T.rule("§1 官方推荐的次序：先组合，再考虑写自定义 C++ 算子")
    print("官方逐字（%s）：" % SPECS.doc("guide/create_op.md"))
    print("  %s" % SPECS.CREATE_OP_RECOMMENDATION)
    print()
    print("★ 同一文件里 `raw_ops` 出现的次数（本仓库核对）= %d ⇒ "
          "**不得**写「tf.raw_ops 是官方推荐的 X」。" % SPECS.CREATE_OP_RAW_OPS_MENTIONS)
    print("  （`tf.raw_ops` 的官方定义原文本次没取到：API 页被拦；且 `raw_ops.py`")
    print("    在 v2.21.0 的目录列表里**不存在**，只有 raw_ops_test.py ⇒ 生成物。见 specs.py 文件头。）")

    # 三条路径：numpy 直算 / Python 组合 / 注册成单个算子
    def swish_composed(x):
        """Python 组合：sigmoid(x) = 1/(1+e^{-x})，用现有算子拼出来（5 个节点）。"""
        neg = T.mul(x, -1.0)
        e = T.exp(neg)
        denom = T.add(e, 1.0)
        return T.mul(x, _recip(denom))

    @T.tf_function
    def composed(x):
        return swish_composed(x)

    T.register_custom_op("swish", lambda a: ref_swish(a))

    @T.tf_function
    def single(x):
        return T.custom_op_call("swish", x)

    print()
    print("同一算子的两条落图路径（本仓库迷你实现）：")
    xt = T.Tensor(np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32))
    composed(xt)
    single(xt)
    g_comp = list(composed._function_cache.values())[0].graph
    g_single = list(single._function_cache.values())[0].graph
    print("  Python 组合 ⇒ %d 个节点：%s" % (len(g_comp.ops), g_comp.op_names()))
    print("  注册成算子 ⇒ %d 个节点：%s" % (len(g_single.ops), g_single.op_names()))
    return composed, single, xt


def _recip(denom):
    """`1 / denom`——mini 里没有 div，用 mul + 一个自定义算子表达。"""
    return T.custom_op_call("__recip__", denom)


T.register_custom_op("__recip__", lambda a: 1.0 / a)


# ---------------------------------------------------------------------------
# §2 tf.custom_gradient 的定位
# ---------------------------------------------------------------------------

def section_2():
    T.rule("§2 tf.custom_gradient 的定位：**给已有的组合换梯度**，不是注册新算子")
    print("官方 docstring 首句逐字（%s）：" % SPECS.TF_CUSTOM_GRADIENT_SOURCE)
    print("  %s" % SPECS.TF_CUSTOM_GRADIENT_DOC)
    print()
    print("也就是说：它作用在**已经能用现有算子写出来的函数**上，只替换反向那一半——")
    print("与 §1 那条「先组合」的次序是同一件事的两面（真注册新 op 是另一条路）。")
    print()
    print("数值动机（本仓库用 numpy 现场算的推导，不是 TF 的实测）：")
    for x in (1.0, 20.0, 100.0):
        naive = log1pexp_naive_grad(x)
        custom = log1pexp_custom_grad(x)
        print("  x = %-6s 链式法则直接写 exp(x)/(1+exp(x)) → %-8s   化简式 1-1/(1+exp(x)) → %.6f"
              % (x, ("NaN" if math.isnan(naive) else "%.6f" % naive), custom))
    return None


# ---------------------------------------------------------------------------
# §3 导出
# ---------------------------------------------------------------------------

def section_3(composed):
    T.rule("§3 导出：图以什么格式离开进程")
    print("官方对 SavedModel 的原话（%s，**含结尾那半句**）："
          % SPECS.doc("guide/saved_model.ipynb"))
    print("  %s" % SPECS.SAVED_MODEL_DEPRECATION_VERBATIM)
    print()
    print("★ 所以「SavedModel 被废弃」是**过度概括**：")
    print("  · `tf.saved_model.save` 在 v2.21.0 源码里**没有任何 deprecated 装饰器**（只有 @tf_export）：")
    print("      %s" % SPECS.SAVED_MODEL_SAVE_DECORATOR.splitlines()[0])
    print("  · Keras 3 的 `Model.export()` **默认格式就是 SavedModel**（%s）："
          % SPECS.KERAS_EXPORT_DOC_URL)
    print("      %s" % SPECS.KERAS_EXPORT_SIGNATURE)
    print("  被取代的是「把 SavedModel 当作 Keras 模型的**保存/再训练**格式」这件事（那用 `.keras`）。")

    path = os.path.join(tempfile.mkdtemp(prefix="mini_sm_"), "saved_model.json")
    cf = list(composed._function_cache.values())[0]
    T.export_saved_model(composed, path, {"serving_default": cf})
    size = os.path.getsize(path)
    art = T.load_saved_model(path)
    print()
    print("本仓库的导出产物：%s" % path)
    print("  格式标记 = %r（本模型借用了 Keras 的默认值这个名字）" % art.format)
    print("  字节数 = %d；签名 = %s" % (size, list(art.signatures)))
    return path, art, size


# ---------------------------------------------------------------------------
# §A 算法正确性
# ---------------------------------------------------------------------------

def section_A(composed, single, xt):
    T.rule("§A 算法正确性：三条独立路径 + 有限差分参照")
    c = T.Checks()

    want = ref_swish(xt.numpy)
    got_comp = composed(xt).numpy
    got_single = single(xt).numpy
    e1 = float(np.max(np.abs(got_comp.astype(np.float64) - want)))
    e2 = float(np.max(np.abs(got_single.astype(np.float64) - want)))
    c.A(e1 < 1e-6, "Python 组合路径与 numpy 参照一致", "max|Δ| = %.3g" % e1)
    c.A(e2 < 1e-6, "注册算子路径与 numpy 参照一致", "max|Δ| = %.3g" % e2)
    c.A(float(np.max(np.abs(got_comp - got_single))) < 1e-6,
        "两条落图路径互相一致（同一算子、两种表示）",
        "max|Δ| = %.3g" % float(np.max(np.abs(got_comp - got_single))))

    # 自定义梯度 vs 有限差分（独立参照）
    xs = [0.0, 0.5, 1.0, 3.0, 10.0]
    worst = 0.0
    for x in xs:
        analytic = log1pexp_custom_grad(x)
        numeric = float(ref_finite_difference(log1pexp_forward, np.float64(x)))
        worst = max(worst, abs(analytic - numeric))
    c.A(worst < 1e-5,
        "化简后的梯度与**中心差分**参照一致（自定义梯度的正确性）",
        "%d 个点，max|Δ| = %.3g" % (len(xs), worst))

    # 反证：把化简式写错（丢掉 1 -），A 类必须失败
    def wrong_grad(x):
        return 1.0 / (1.0 + np.exp(np.float64(x)))          # 少了那个 1 -

    worst_wrong = max(abs(wrong_grad(x) - float(
        ref_finite_difference(log1pexp_forward, np.float64(x)))) for x in xs)
    c.A(worst_wrong > 1e-3,
        "反证：梯度公式写错时，与有限差分的差异必须被检出",
        "错版 max|Δ| = %.3g（说明上面那条不是恒真断言）" % worst_wrong)
    return c


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def section_B(composed, single, art, size):
    T.rule("§B 代价模型：节点数、导出规模、载入后的追踪次数")
    c = T.Checks()

    g_comp = list(composed._function_cache.values())[0].graph
    g_single = list(single._function_cache.values())[0].graph
    c.B(len(g_comp.ops) == 5,
        "Python 组合：图节点数 == 组合里用到的算子数（本模型把 swish 拆成 5 步）",
        "实测 %d 个节点：%s" % (len(g_comp.ops), g_comp.op_names()))
    c.B(len(g_single.ops) == 1,
        "★ 注册成算子：**整个算子只占 1 个节点**（图更小、但语义被藏进那个 op 里）",
        "实测 %d 个节点：%s" % (len(g_single.ops), g_single.op_names()))
    print("    ★ 主线上的一句话：**算子粒度决定「固化了多少细节」**——")
    print("      %d 个节点的图，编译器看得见每一步；1 个节点的图，只有一条边界。"
          % len(g_comp.ops))

    sig = art.signatures["serving_default"]
    op_names = list(sig.op_names)
    c.B(len(op_names) == len(g_comp.ops),
        "导出产物里的算子数 == 导出前那张图的节点数",
        "产物 %d 个：%s" % (len(op_names), op_names))
    c.B(size > 0 and sig.function_type == str(
        list(composed._function_cache.values())[0].function_type),
        "导出产物里带着**签名**（正是它把 shape 假设一起冻住了）",
        "签名 = %s" % sig.function_type)
    print("    产物字节数 = %d（本仓库的 JSON 形态，**不是 SavedModel 的 protobuf**）" % size)

    composed(T.Tensor(np.zeros((5,), dtype=np.float32)))
    n_after_more_calls = composed.experimental_get_tracing_count()
    c.B(n_after_more_calls == 1,
        "导出**不改变**原函数的行为：继续调用也不重追踪",
        "追踪计数 = %d" % n_after_more_calls)
    return c


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def section_C(art):
    T.rule("§C 边界与反例：每条失败行为都要真的跑出来")
    c = T.Checks()

    T.subsection("C1 导出把 shape 假设冻住了：换个 shape 就进不去")
    sig = art.signatures["serving_default"]
    probe = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    ok = sig(T.Tensor(probe))
    err = float(np.max(np.abs(ok.numpy.astype(np.float64) - ref_swish(probe))))
    c.C(err < 1e-6, "对照：**同样的 shape** 调载入的函数，结果与参照一致",
        "shape (5,) 通过，max|Δ| = %.3g；载入后调用次数 = %d" % (err, sig.calls))
    c.expect_raises(
        lambda: sig(T.Tensor(np.zeros((7,), dtype=np.float32))),
        T.InvalidArgumentError,
        "换一个 shape 调载入的函数 ⇒ 被签名拒掉（这就是固化假设的代价）")
    c.C(len(sig.op_names) > 0 and sig.calls == 1,
        "★ 载入后的调用**不产生任何 tracing**（图是死的，Python 函数体不在了）",
        "载入函数只带图与签名，没有 python_function 字段")

    T.subsection("C2 未注册的算子调不动")
    T.register_custom_op("known_op", lambda a: a * 2.0)
    c.expect_raises(lambda: T.custom_op_call("swish_v2", T.Tensor(np.float32(1.0))),
                    T.FatalError, "调用未注册的算子名 ⇒ FatalError")
    print("    ★ 反面：注册过的名字可以调：%s"
          % T.custom_op_call("known_op", T.Tensor(np.float32(2.0))).numpy)

    T.subsection("C3 数值不稳的梯度公式真的会给 NaN")
    bad = log1pexp_naive_grad(100.0)
    good = log1pexp_custom_grad(100.0)
    c.C(math.isnan(bad) and abs(good - 1.0) < 1e-9,
        "x=100 处链式法则直接写的公式给 NaN，化简式给 1.0",
        "naive = %s，custom = %.9f（float32 下 exp(100) 已溢出为 inf）" % (bad, good))

    T.subsection("C4 组合路径把中间量也暴露给了图")
    neg_inf = T.Tensor(np.array([-100.0], dtype=np.float32))
    out = T.custom_op_call("swish", neg_inf)
    c.C(float(out.numpy[0]) == 0.0 or abs(float(out.numpy[0])) < 1e-6,
        "注册算子版本在 x=-100 上不产生 NaN（实现内部做了稳定化）",
        "swish(-100) = %r；对照：组合路径用 e^{-x} 会在 x 很大时溢出"
        % float(out.numpy[0]))
    return c


def main():
    T.header("03", "自定义算子与导出 —— 把算子接进这条链，以及图的落地格式")
    composed, single, xt = section_1()
    section_2()
    path, art, size = section_3(composed)
    parts = [section_A(composed, single, xt), section_B(composed, single, art, size),
             section_C(art)]
    return T.Checks.merge(parts).summary()


if __name__ == "__main__":
    sys.exit(main())
