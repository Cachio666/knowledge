"""01 HLO 与 shape inference。

对应文档：60-compilers/xla/01-hlo-and-shape-inference.md

★ 本程序跑的是本仓库的**迷你 HLO 实现**（`minihlo.py`），**不是 XLA**。
  输出的一切数字都是**本机实测值**，只能支撑关于「机制」的结论。
  关于真实 XLA 的事实（`dot_general` 的三套维度命名、`jnp.dot` 的语义 …）
  只在文档与 `specs.py` 里，逐条带来源，**本程序不对它们下任何断言**。

三类断言：
  A 算法正确性 —— shape inference 的结果与**手写的 numpy 参照表达式**逐项比对
  B 代价模型   —— 值的字节数 / 元素数 / op 数，须与文档 §5 的表格逐行一致
  C 边界与反例 —— 文档 §4 声称「会被拒绝」的构造必须**真的抛异常**

用法：python3 01_hlo_shape_inference.py
"""

import numpy as np

import specs as SPECS
from minihlo import (BroadcastError, Checks, DimNumberError, DimSizeError, Module,
                     Shape, execute, infer_dot_general, infer_shape, rename_dims,
                     rule, verify, verify_or_raise)

SEED = 20260916
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 地形 1：一张含广播 / broadcast / transpose / reduce 的小图
# ★ 地形在写下任何运行结果**之前**就冻结在这里，种子固定，不因结果调整。
# ---------------------------------------------------------------------------


def build_graph_module():
    m = Module("graph")
    p0 = m.parameter(Shape([2, 3], "f32"))
    c0 = m.emit("constant", [], Shape([3], "f32"), value=np.array([1.0, 2.0, 3.0],
                                                                  np.float32))
    add = m.emit("add", [p0, c0], Shape([2, 3], "f32"), broadcast_dimensions=[1])
    # `broadcast` 的结果形状 = sizes ++ （操作数的各维放到 dimensions 指示的位置）。
    # 这里 sizes={4}、dimensions={1,2} → 操作数 f32[2,3] 落到结果的第 1、2 维。
    b = m.emit("broadcast", [add], Shape([4, 2, 3], "f32"), sizes=[4],
               dimensions=[1, 2])
    t = m.emit("transpose", [b], Shape([3, 4, 2], "f32"), permutation=[2, 0, 1])
    m.emit("reduce", [t], Shape([3, 4], "f32"), dimensions=[2])
    return m


# 地形 2 / 3：同样是 `(3,2,4) dot (3,4,1)`，**只差 batch 维**。
#   地形 2 没有 batch 维 → 前导维全都是自由维（对应 numpy / jnp 的 `dot` 语义）
#   地形 3 有 batch 维   → 前导维成对匹配（对应 `matmul` 语义）


def build_dot_module(with_batch):
    m = Module("dot_batched" if with_batch else "dot_stacked")
    p0 = m.parameter(Shape([3, 2, 4], "f32"), name="p0")
    p1 = m.parameter(Shape([3, 4, 1], "f32"), name="p1")
    attrs = dict(lhs_contracting_dims=[2], rhs_contracting_dims=[1])
    if with_batch:
        attrs.update(lhs_batch_dims=[0], rhs_batch_dims=[0])
        shape = Shape([3, 2, 1], "f32")
    else:
        shape = Shape([3, 2, 3, 1], "f32")
    m.emit("dot_general", [p0, p1], shape, **attrs)
    return m


# ---------------------------------------------------------------------------
# A 类的**独立参考实现**：手写 numpy 表达式，逐 op 给出期望值。
#   这条路径不看 minihlo 的 infer_shape，也不看它的执行结果——
#   它是「同一个计算用 numpy 再写一遍」。
# ---------------------------------------------------------------------------

# `broadcast` 的 dimensions={1,2}：操作的 f32[2,3] 落到结果 (4,2,3) 的第 1、2 维，
# 所以 numpy 这边要先在**最前面**补一维（不是中间）。
REFERENCE_GRAPH = {
    "add.0": lambda f: f["p0"] + f["c0"],
    "broadcast.0": lambda f: np.broadcast_to(
        (f["p0"] + f["c0"])[None, :, :], (4, 2, 3)),
    "transpose.0": lambda f: np.transpose(
        np.broadcast_to((f["p0"] + f["c0"])[None, :, :], (4, 2, 3)), (2, 0, 1)),
    "reduce.0": lambda f: np.sum(np.transpose(
        np.broadcast_to((f["p0"] + f["c0"])[None, :, :], (4, 2, 3)), (2, 0, 1)),
        axis=2),
}


def main():
    checks = Checks()

    # ------------------------------------------------------------------
    rule("1. 一张小图的 HLO 文本（本仓库自己的简化打印格式）")
    g = build_graph_module()
    print(g.text())
    print()
    _probs = verify(g)
    print("verify（全模块重推一遍形状）：%d 个问题" % len(_probs))
    for p in _probs:
        print("   ", p)
    print()
    print("★ 术语来自 XLA，实现是本仓库的迷你版；打印格式也是我们自己定的。")

    # ------------------------------------------------------------------
    rule("2. A 算法正确性：shape inference vs 手写 numpy 参照")
    feed = {"p0": rng.standard_normal((2, 3)).astype(np.float32)}
    env = execute(g, feed)
    feed_plus = dict(feed, c0=np.array([1.0, 2.0, 3.0], np.float32))

    a_rows = []
    for op in g.ops:
        if op.name not in REFERENCE_GRAPH:
            continue                       # 常量没有「计算」，不参与对拍
        got = env[op.name]
        want = REFERENCE_GRAPH[op.name](feed_plus)
        same_shape = got.shape == want.shape
        max_err = float(np.max(np.abs(got - want))) if same_shape else float("inf")
        a_rows.append((op.name, str(tuple(got.shape)), str(tuple(want.shape)),
                       max_err, same_shape and max_err == 0.0))
        print("  %-14s 迷你 %-12s numpy %-12s 最大误差 %.3g"
              % (op.name, tuple(got.shape), tuple(want.shape), max_err))
    checks.A(all(r[4] for r in a_rows),
             "%d 个 op 的**形状与数值**都与手写 numpy 参照逐项一致" % len(a_rows),
             "最大误差 %.3g" % max(r[3] for r in a_rows))

    # 形状推断（不看执行）也必须一致
    infer_rows = []
    for op in g.ops:
        if op.name not in REFERENCE_GRAPH:
            continue
        want = REFERENCE_GRAPH[op.name](feed_plus)
        got = infer_shape(op)
        infer_rows.append(got.dims == want.shape)
    checks.A(all(infer_rows),
             "infer_shape 的结果与 numpy 实际算出的 ndarray.shape 逐项一致",
             "%d/%d" % (sum(infer_rows), len(infer_rows)))

    # ------------------------------------------------------------------
    rule("3. A 算法正确性：`(3,2,4) dot (3,4,1)` 的两种语义 vs numpy")
    a = rng.standard_normal((3, 2, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4, 1)).astype(np.float32)

    ms = build_dot_module(False)
    mb = build_dot_module(True)
    print(ms.text())
    print()
    print(mb.text())
    print()

    got_s = execute(ms, {"p0": a, "p1": b})["dot_general.0"]
    got_b = execute(mb, {"p0": a, "p1": b})["dot_general.0"]
    ref_dot = np.dot(a, b)
    ref_matmul = np.matmul(a, b)

    print("  np.dot(a,b)    的 shape = %-12s   ← jnp.dot 的语义（官方文档，见 specs）"
          % (ref_dot.shape,))
    print("  np.matmul(a,b) 的 shape = %-12s   ← matmul 的语义（batch 维广播）"
          % (ref_matmul.shape,))
    print("  迷你 dot_general（无 batch 维） = %s" % (got_s.shape,))
    print("  迷你 dot_general（有 batch 维） = %s" % (got_b.shape,))
    err_s = float(np.max(np.abs(got_s - ref_dot)))
    err_b = float(np.max(np.abs(got_b - ref_matmul)))
    print("  与 numpy 的最大绝对误差：无 batch %.3g，有 batch %.3g" % (err_s, err_b))
    print()
    # 容差取 1e-5：fp32 下「先 reshape/transpose 再 matmul」与 numpy 的直接实现
    # **累加次序不同**，末位会差 1 个 ulp 量级——这不是实现错误，是浮点累加次序的差异。
    TOL = 1e-5
    checks.A(got_s.shape == ref_dot.shape and err_s <= TOL,
             "无 batch 维的 dot_general == np.dot（形状一致且数值在 1e-5 内）",
             "%s vs %s，误差 %.3g" % (got_s.shape, ref_dot.shape, err_s))
    checks.A(got_b.shape == ref_matmul.shape and err_b <= TOL,
             "有 batch 维的 dot_general == np.matmul（形状一致且数值在 1e-5 内）",
             "%s vs %s，误差 %.3g" % (got_b.shape, ref_matmul.shape, err_b))
    checks.A(ref_dot.shape != ref_matmul.shape,
             "同一对操作数，两种语义**结果形状不同**——这就是「stacked 而非 broadcast」",
             "%s vs %s" % (ref_dot.shape, ref_matmul.shape))

    # ------------------------------------------------------------------
    rule("4. 三套维度命名指的是同一件事（**文档事实，不是断言对象**）")
    print("  HLO proto     :", SPECS.HLO_PROTO_DIM_NAMES)
    print("  HLO 文本      :", SPECS.HLO_TEXT_DIM_NAMES)
    print("  StableHLO attr:", SPECS.STABLEHLO_DIM_NAMES)
    print()
    attrs = {"lhs_contracting_dimensions": [2], "rhs_contracting_dimensions": [1]}
    print("  以 proto 写法给出 %s" % attrs)
    print("  改名为 HLO 文本   → %s" % rename_dims(attrs, "hlo_proto", "hlo_text"))
    print("  改名为 StableHLO  → %s" % rename_dims(attrs, "hlo_proto", "stablehlo"))
    print("  ↑ 只是**改名**，语义完全一样；本迷你实现内部统一用 `*_dims` 这一套。")
    print("  来源见 code/specs.py。三套命名**不进断言**（纯 Python + numpy 下没有 "
          "HLO proto 可断言）。")

    # ------------------------------------------------------------------
    rule("5. B 代价模型：与文档 §5 的表格逐行核对")
    n_ops = len(g.ops)
    n_params = len(g.params)
    total_bytes = sum(p.shape.nbytes() for p in g.params) + sum(
        o.shape.nbytes() for o in g.ops)
    b_rows = [
        ("op 总数（含常量）", n_ops, 5),
        ("参数个数", n_params, 1),
        ("全部值的字节数", total_bytes, 300),
        ("root 的元素数", g.root().shape.num_elements, 12),
        ("broadcast.0 的元素数", g.by_name("broadcast.0").shape.num_elements, 24),
        ("dot 无 batch 的元素数", ms.root().shape.num_elements, 18),
        ("dot 有 batch 的元素数", mb.root().shape.num_elements, 6),
    ]
    print("%-26s %10s %8s" % ("项", "本机值", "文档值"))
    print("-" * 48)
    for name, got, want in b_rows:
        ok = (got == want)
        print("%-26s %10s %8s  %s" % (name, got, want, "ok" if ok else "MISMATCH"))
        checks.B(ok, "文档表：%s == %d" % (name, want), "本机 %s" % got)
    checks.B(ms.root().shape.num_elements == 3 * mb.root().shape.num_elements,
             "同一对操作数：stacked 语义的结果是 batched 语义的 3 倍大",
             "%d vs %d" % (ms.root().shape.num_elements,
                           mb.root().shape.num_elements))

    # ------------------------------------------------------------------
    rule("6. C 边界与反例：这些构造**必须被拒绝**")

    # C1：广播退化——rhs 的维与结果对应维既不相等也不是 1
    def bad_broadcast_mismatch():
        m = Module("bad1")
        p0 = m.parameter(Shape([2, 3], "f32"))
        bad = m.emit("constant", [], Shape([4], "f32"), value=np.zeros(4, np.float32))
        m.emit("add", [p0, bad], Shape([2, 3], "f32"), broadcast_dimensions=[1])
        return verify_or_raise(m)

    # C2：broadcast_dimensions 不是严格递增（HLO 要求有序）
    def bad_broadcast_unsorted():
        m = Module("bad2")
        p0 = m.parameter(Shape([2, 3], "f32"))
        rhs = m.emit("constant", [], Shape([3, 2], "f32"), value=np.zeros((3, 2),
                                                                          np.float32))
        m.emit("add", [p0, rhs], Shape([2, 3], "f32"), broadcast_dimensions=[1, 0])
        return verify_or_raise(m)

    # C2b：`broadcast` 的 dimensions 指向了由 sizes 决定的那几维
    def bad_broadcast_overlap():
        m = Module("bad2b")
        p0 = m.parameter(Shape([2, 3], "f32"))
        m.emit("broadcast", [p0], Shape([4, 2, 3], "f32"), sizes=[4],
               dimensions=[0, 1])          # 应写 {1,2}
        return verify_or_raise(m)

    # C3：dot_general 的 contracting 维与 batch 维重叠（StableHLO C3）
    def bad_dup_dims():
        return [infer_dot_general(Shape([2, 3], "f32"), Shape([3, 4], "f32"),
                                  {"lhs_contracting_dims": [1], "rhs_contracting_dims": [0],
                                   "lhs_batch_dims": [1], "rhs_batch_dims": [0]})]

    # C4：两侧 batch 维个数不等（StableHLO C1）
    def bad_batch_count():
        return [infer_dot_general(Shape([2, 3, 4], "f32"), Shape([3, 4], "f32"),
                                  {"lhs_contracting_dims": [2], "rhs_contracting_dims": [0],
                                   "lhs_batch_dims": [0], "rhs_batch_dims": []})]

    # C5：contracting 维大小不等（StableHLO C10）
    def bad_contract_size():
        return [infer_dot_general(Shape([2, 5], "f32"), Shape([3, 4], "f32"),
                                  {"lhs_contracting_dims": [1], "rhs_contracting_dims": [0]})]

    # C6：batch 维编号越界（StableHLO C5）
    def bad_dim_range():
        return [infer_dot_general(Shape([2, 3], "f32"), Shape([3, 4], "f32"),
                                  {"lhs_contracting_dims": [1], "rhs_contracting_dims": [0],
                                   "lhs_batch_dims": [2], "rhs_batch_dims": [0]})]

    # C7：★ batch 维大小不等——**dot_general 拒绝，而 matmul 会广播**
    def bad_batch_size():
        return [infer_dot_general(Shape([3, 2, 4], "f32"), Shape([1, 4, 5], "f32"),
                                  {"lhs_contracting_dims": [2], "rhs_contracting_dims": [1],
                                   "lhs_batch_dims": [0], "rhs_batch_dims": [0]})]

    # C8：标量操作数——rank 0 上没有任何合法的 contracting 维
    def bad_scalar_dot():
        return [infer_dot_general(Shape([], "f32"), Shape([4], "f32"),
                                  {"lhs_contracting_dims": [0],
                                   "rhs_contracting_dims": [0]})]

    checks.expect_raises(bad_broadcast_mismatch, BroadcastError,
                         "广播退化（rhs 维既不等于 1 也不等于结果维）→ 拒绝")
    checks.expect_raises(bad_broadcast_unsorted, BroadcastError,
                         "broadcast_dimensions 非严格递增 → 拒绝")
    checks.expect_raises(bad_broadcast_overlap, BroadcastError,
                         "broadcast 的 dimensions 指向 sizes 覆盖的前几维 → 拒绝")
    checks.expect_raises(bad_dup_dims, DimNumberError,
                         "contracting 与 batch 维重叠（C3）→ 拒绝")
    checks.expect_raises(bad_batch_count, DimSizeError,
                         "两侧 batch 维个数不等（C1）→ 拒绝")
    checks.expect_raises(bad_contract_size, DimSizeError,
                         "contracting 维大小不等（C10）→ 拒绝")
    checks.expect_raises(bad_dim_range, DimNumberError,
                         "维度编号越界（C5）→ 拒绝")
    checks.expect_raises(bad_batch_size, DimSizeError,
                         "batch 维大小不等（C9）→ 拒绝：**这里是 dot_general 与 "
                         "matmul 分道扬镳的地方**（matmul 会把它广播掉）")
    checks.expect_raises(bad_scalar_dot, DimNumberError,
                         "标量操作数进 dot_general（rank 0 无合法 contracting 维）→ 拒绝")

    print()
    print("  —— 对照：把上面的构造换成合法的，同一条判据必须放行 ——")

    def good_broadcast():
        return len(verify(g)) == 0

    def good_batch_one():
        # 把 batch 维改成 1 之外的合法值：两侧都是 3
        s = infer_dot_general(Shape([3, 2, 4], "f32"), Shape([3, 4, 5], "f32"),
                              {"lhs_contracting_dims": [2], "rhs_contracting_dims": [1],
                               "lhs_batch_dims": [0], "rhs_batch_dims": [0]})
        return s.dims

    checks.expect_ok(good_broadcast, "对照：地形 1 的图 verify 无问题")
    checks.expect_ok(good_batch_one,
                     "对照：两侧 batch 维都是 3 时，dot_general 通过",
                     "结果 (3,2,5)")

    # ------------------------------------------------------------------
    rule("7. 可证伪性反证：把被测实现故意改错，A 类判据必须报错")

    def broken_infer_dot(lhs, rhs, attrs):
        """故意写错：把 lhs/rhs 的 contracting 维**对调**（一个经典的笔误）。"""
        swapped = dict(attrs)
        swapped["lhs_contracting_dims"] = attrs.get("rhs_contracting_dims", [])
        swapped["rhs_contracting_dims"] = attrs.get("lhs_contracting_dims", [])
        return infer_dot_general(lhs, rhs, swapped)

    bad_desc = ""
    try:
        bad_shape = broken_infer_dot(Shape([3, 2, 4], "f32"), Shape([3, 4, 1], "f32"),
                                     {"lhs_contracting_dims": [2],
                                      "rhs_contracting_dims": [1]})
        bad_desc = str(bad_shape)
        broken = (bad_shape.dims != ref_dot.shape)
    except Exception as e:                                       # noqa: BLE001
        bad_desc = "抛出 %s: %s" % (type(e).__name__, e)
        broken = True
    print("  改错后的推断结果：%s（手写 numpy 参照给的是 shape=%s）"
          % (bad_desc, ref_dot.shape))
    checks.C(broken,
             "反证：把 contracting 维对调后，A 类的「与 numpy 一致」判据会失败",
             "改错实现给出 %s，numpy 给出 %s" % (bad_desc, ref_dot.shape))

    # ------------------------------------------------------------------
    rule("8. 断言汇总")
    checks.summary()


if __name__ == "__main__":
    main()
