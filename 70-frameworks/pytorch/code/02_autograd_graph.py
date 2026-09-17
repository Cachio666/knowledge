#!/usr/bin/env python3
"""02 autograd：反向图**固化了什么**——saved tensors 就是被固化下来的假设。

对应文档：70-frameworks/pytorch/02-autograd.md

★ 本程序跑的是**真 torch**（`import torch`）——输出是**本机实测值**，
  **必须绑定版本**（本机 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 **五个**都跑过；2.9/2.10/2.12 本机无解释器）。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 `verify.py`）：
  A 算法正确性 —— 用**有限差分**（纯 numpy，与 autograd 引擎完全无关的数学路径）
                  复核 torch 自动求出的梯度；再加 `is_leaf` 的形式判据逐例核对。
  B 代价模型   —— 一次前向**保存了什么、多少字节**：按「反向公式需要什么」手写模型，
                  再用 `saved_tensors_hooks` 实测对拍。
  C 边界与反例 —— 失败行为必须真的跑出来（inference_mode 的边界、`retain_graph`、
                  `autograd.grad()` 不写 `.grad` 等）。

用法：cd code && make 02_autograd_graph   （或 make run 全跑）
"""

import sys

import numpy as np
import torch

import specs as SPECS

VERSIONS = "2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0"

_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append(f"[{cls}] {name}")
    tail = f"   ← {detail}" if detail else ""
    print(f"    [{'通过' if ok else '**不通过**'}] {name}{tail}")
    return ok


def rule(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def sub(title):
    print()
    print(f"--- {title} ---")


# --- §1 反向图怎么建：grad_fn / next_functions / is_leaf

def section_1():
    rule("§1  反向图是挂在**输出**上的节点链")
    w = torch.randn(3, requires_grad=True)
    y = (w * 2 + 1).relu().sum()
    print("计算：w (shape %s, requires_grad=True) → w*2 → +1 → relu → sum → y"
          % (tuple(w.shape),))
    print()
    print("y.grad_fn =", type(y.grad_fn).__name__)
    print()
    print("沿 next_functions[0] 一路走（每个节点打印 [(子节点, 输入序号)]）：")
    chain = []
    node = y.grad_fn
    while node is not None:
        nxt = [(type(n).__name__ if n is not None else None, idx)
               for n, idx in node.next_functions]
        chain.append((type(node).__name__, nxt))
        print("    %-18s → %s" % (type(node).__name__, nxt))
        kids = [n for n, _ in node.next_functions if n is not None]
        node = kids[0] if kids else None
    print()
    print("★ `None` 表示**该输入不需要梯度**（这里是常量 1 与标量 2）；")
    print("  叶子输入对应的是 `AccumulateGrad`——它就是把梯度累加进 `.grad` 的那个节点。")
    expected_chain = [
        ("SumBackward0", [("ReluBackward0", 0)]),
        ("ReluBackward0", [("AddBackward0", 0)]),
        ("AddBackward0", [("MulBackward0", 0), (None, 0)]),
        ("MulBackward0", [("AccumulateGrad", 0), (None, 0)]),
        ("AccumulateGrad", []),
    ]
    check("A", "实测的 grad_fn 链与手写的期望链逐项一致", chain == expected_chain, f"实测 {[c[0] for c in chain]}")
    v = w.view(-1)
    print()
    print("is_leaf 的对照（形式判据：`grad_fn is None or not requires_grad`）：")
    print("    w.is_leaf = %s（用户创建的叶子）" % w.is_leaf)
    print("    w.view(-1).is_leaf = %s，_is_view() = %s，_base is w = %s"
          % (v.is_leaf, v._is_view(), v._base is w))
    check("A", "view 出来的张量不是 leaf（尽管它是「用户创建的」）", v.is_leaf is False and v._is_view() is True)


# --- §A 独立参照：有限差分

def finite_diff_grad(f, xs, eps=1e-6):
    """中心差分的独立参照实现（纯 numpy，不碰 autograd 引擎）。
    这是与本主题**完全不同的一条数学路径**：autograd 走的是反向图里的链式法则，
    这里走的是 f(x+eps) - f(x-eps) 的数值微分。
    """
    grads = []
    for i, x in enumerate(xs):
        g = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            xp = x.copy()
            xm = x.copy()
            xp[idx] += eps
            xm[idx] -= eps
            g[idx] = (f(*[xp if j == i else y for j, y in enumerate(xs)])
                      - f(*[xm if j == i else y for j, y in enumerate(xs)])) / (2 * eps)
        grads.append(g)
    return grads


def section_A():
    rule("§A  A 算法正确性：有限差分（纯 numpy）复核 autograd 的梯度")
    rng = np.random.default_rng(20260917)
    X0 = rng.standard_normal((3, 4))
    W0 = rng.standard_normal((4, 5))
    B0 = rng.standard_normal((5,))
    def f_np(x, w, b):
        h = np.maximum(x @ w + b, 0.0)          # relu
        return float(h.sum())
    def f_torch(x, w, b):
        return torch.relu(x @ w + b).sum()
    xt = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
    wt = torch.tensor(W0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(B0, dtype=torch.float64, requires_grad=True)
    f_torch(xt, wt, bt).backward()
    fd = finite_diff_grad(f_np, [X0, W0, B0])
    names = ["d/dX", "d/dW", "d/db"]
    for name, ref, got in zip(names, fd, [xt.grad, wt.grad, bt.grad]):
        err = float(np.max(np.abs(ref - got.detach().numpy())))
        scale = max(1.0, float(np.max(np.abs(ref))))
        check("A", f"{name}：有限差分与 autograd 一致（相对误差 < 1e-5）", err / scale < 1e-5, f"最大绝对误差 {err:.3e}，量级 {scale:.3f}")
    sub("A2  反证：把一侧故意改错，**同一条 A 类判据**必须失败")
    fd_x = fd[0].copy()
    broken = fd[0].copy()
    broken[:, 0] = 0.0                     # 故意改错参照的一列
    err_broken = float(np.max(np.abs(broken - fd_x)))
    truth = xt.grad.detach().numpy()
    rel_broken = err_broken / max(1.0, float(np.max(np.abs(truth))))
    print("    把有限差分参照的第 0 列置零后，同一条「相对误差 < 1e-5」判定：")
    print("        该列差异 = %.3f ⇒ 相对误差 %.3e" % (err_broken, rel_broken))
    check("A", "反证：改错的参照被同一条判据判为不通过（A 类因此可证伪）", rel_broken >= 1e-5, f"相对误差 {rel_broken:.3e} ≥ 1e-5")
    sub("A3  is_leaf 的形式判据逐例核对（判据写在程序里，不读 torch 的字段）")
    def is_leaf_rule(grad_fn_is_none, requires_grad):
        return grad_fn_is_none or (not requires_grad)
    leaf = torch.randn(3, requires_grad=True)
    nol = torch.randn(3, requires_grad=False)
    cases = [
        ("用户创建的叶子 w", leaf, leaf.grad_fn is None),
        ("不需要梯度的张量", nol, nol.grad_fn is None),
        ("w * 2", leaf * 2, (leaf * 2).grad_fn is None),
        ("w.view(-1)", leaf.view(-1), (leaf.view(-1)).grad_fn is None),
        ("w.detach()", leaf.detach(), (leaf.detach()).grad_fn is None),
        ("with no_grad 下的 w*2", None, None),
    ]
    with torch.no_grad():
        ng = leaf * 2
    cases[-1] = ("no_grad 下的 w*2", ng, ng.grad_fn is None)
    for name, t, gfn_none in cases:
        predicted = is_leaf_rule(gfn_none, bool(t.requires_grad))
        check("A", f"is_leaf 判据：{name}", predicted == bool(t.is_leaf), f"判据预测 {predicted}，torch 报 {bool(t.is_leaf)}（requires_grad={t.requires_grad}）")
    sub("A4  `w.view(-1)` 的判据必须用形式判据，不能用「用户创建的」这个口语表述")
    print("    官方 docstring 逐字：「they will be leaf Tensors if they were **created by the user**」")
    print("    但 `w.view(-1)` 正是用户创建的，torch 却报 is_leaf=False。")
    print("    口语表述与形式判据在 view 上分道扬镳——这就是必须用形式判据的理由。")
    check("A", "口语表述（假设「用户创建的 ⇒ leaf」）在 view 上与 torch 冲突", (leaf.view(-1).is_leaf is False), "若按口语表述会预测 True，实测 False")


# --- §B 代价模型：一次前向保存了多少

def measure_saved(fn):
    """用 `saved_tensors_hooks` 数一次前向保存了几个张量、共多少字节。
    ★ 这是**唯一通用的观测手段**：`grad_fn.saved_tensors` 在 C++ 定义的节点上
      **不存在**（§C4 会真的把它跑出 AttributeError）。
    """
    rec = []
    def pack(t):
        rec.append((tuple(t.shape), t.numel() * t.element_size()))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return rec


def section_B():
    rule("§B  B 代价模型：反向图固化了什么——按「反向公式需要什么」预测，再实测对拍")
    print("模型（手写，不是从 torch 读的）：")
    print("    · add / sub / sum / 与**标量**相乘与相加：反向公式不需要任何输入 ⇒ 保存 0 个")
    print("    · mul(A, B) 两个**张量**相乘：反向需要另一个操作数 ⇒ 保存 1 个/每个操作数")
    print("      （即使两个操作数是同一个张量，也各存一份 ⇒ `x*x` 是 2 个）")
    print("    · mm(A, B)：反向需要 A 与 B ⇒ 保存 2 个")
    print("    · relu / sigmoid / exp：反向需要**输出** ⇒ 保存 1 个（输出大小）")
    print("    · pow(x, 2)：反向需要底数 ⇒ 保存 1 个")
    print("    · 逐元素一元算子若反向只用输出，保存字节数 = 输出字节数")
    print()
    print("%-26s %-8s %-8s %-10s %s" % ("算例", "模型:packs", "实测", "模型:字节", "实测"))
    print("-" * 78)
    x = torch.randn(3, requires_grad=True)
    A = torch.randn(3, 4, requires_grad=True)
    B = torch.randn(4, 5, requires_grad=True)
    q = torch.randn(4, requires_grad=True)
    r = torch.randn(4, requires_grad=True)
    s = torch.randn(4, requires_grad=True)
    m = torch.randn(4, requires_grad=True)
    n_ = torch.randn(4, requires_grad=True)
    o = torch.randn(4, 4, requires_grad=True)
    p = torch.randn(4, requires_grad=True)
    k = torch.randn(3, requires_grad=True)
    cases = [
        ("(x*x).sum()", lambda: (x * x).sum().backward(), 2, 2 * 3 * 4),
        ("(k*2).sum()（标量乘）", lambda: (k * 2).sum().backward(), 0, 0),
        ("(A@B).sum()", lambda: (A @ B).sum().backward(), 2, 3 * 4 * 4 + 4 * 5 * 4),
        ("(q*2+1).relu().sum()", lambda: (q * 2 + 1).relu().sum().backward(), 1, 4 * 4),
        ("(r*s).sum()", lambda: (r * s).sum().backward(), 2, 2 * 4 * 4),
        ("m.pow(2).sum()", lambda: m.pow(2).sum().backward(), 1, 4 * 4),
        ("n.exp().sum()", lambda: n_.exp().sum().backward(), 1, 4 * 4),
        ("o.sum(1).sum()", lambda: o.sum(1).sum().backward(), 0, 0),
        ("p.sigmoid().sum()", lambda: p.sigmoid().sum().backward(), 1, 4 * 4),
    ]
    for name, fn, want_packs, want_bytes in cases:
        rec = measure_saved(fn)
        got_packs = len(rec)
        got_bytes = sum(r[1] for r in rec)
        print("%-26s %-8d %-8d %-10d %d" % (name, want_packs, got_packs,
                                            want_bytes, got_bytes))
        check("B", f"{name}：保存个数 == 模型预测的 {want_packs}", got_packs == want_packs, f"实测 {got_packs}")
        check("B", f"{name}：保存字节 == 模型预测的 {want_bytes}", got_bytes == want_bytes, f"实测 {got_bytes}")
    print()
    print("★ 与 specs.py 的基线表逐行核对（那三个算例是 §5 引用的数字）：")
    base = {r[0]: r[1] for r in SPECS.SAVED_TENSOR_BASELINE}
    for name, fn, _p, _b in cases:
        if name not in base:
            continue
        got = len(measure_saved(fn))
        check("B", f"基线表：{name} 的 pack 数 == {base[name]}", got == base[name], f"实测 {got}")
    print()
    print("★ 不得给「一次前向保存 N 个中间结果」这类通用值——")
    print("  它的正确答案是「取决于算子的反向公式」，上表每个算例都不一样。")


# --- §C 失败模式

def section_C():
    rule("§C  C 边界与反例：这些失败行为必须真的发生")
    sub("C1  inference_mode 张量：出作用域后**读**正常，**原地改**被拒")
    x = torch.randn(3)
    with torch.inference_mode():
        z = x * 2
    reads = {
        "shape": tuple(z.shape), "sum": float(z.sum()), "item": z[0].item(),
        "numpy": z.numpy().shape, "clone": tuple(z.clone().shape),
        "detach": tuple(z.detach().shape),
    }
    print("    出作用域后的读操作：", reads)
    check("C", "inference_mode 张量出作用域后**读**全部正常", True)
    try:
        z.add_(1)
        check("C", "出作用域后原地改必须被拒", False, "它没有被拒")
    except RuntimeError as exc:
        check("C", "出作用域后原地改被拒（原文见 specs.py）", "Inplace update to inference tensor outside InferenceMode" in str(exc), str(exc).splitlines()[0][:70])
    sub("C2  inference_mode 张量：`_version` **在 mode 内就已不可读**")
    with torch.inference_mode():
        z2 = x * 2
        try:
            z2._version
            check("C", "读 inference 张量的 _version 必须报错", False, "它没有报错")
        except RuntimeError as exc:
            check("C", "读 _version 在 mode 内就报错（不是出作用域才失效）", "do not track version counter" in str(exc), str(exc)[:60])
    try:
        z2._version
        check("C", "出作用域后读 _version 同样报错", False, "它没有报错")
    except RuntimeError:
        check("C", "出作用域后读 _version 同样报错", True)
    sub("C3  save-for-backward：用 inference 张量参与要被记录的计算，必须被拒")
    with torch.inference_mode():
        inf = torch.randn(3)
    class Saver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(b)
            return a * 2
        @staticmethod
        def backward(ctx, g):
            return g * 2, None
    zt = torch.randn(3, requires_grad=True)
    try:
        out = Saver.apply(zt, inf)
        out.sum().backward()
        check("C", "保存 inference 张量必须被拒", False, "它没有被拒")
    except RuntimeError as exc:
        msg = str(exc)
        print("    错误原文：", " ".join(msg.split())[:150])
        check("C", "保存 inference 张量被拒（措辞与 specs.py 一致）", "Inference tensors cannot be saved for backward" in msg, "")
    sub("C4  `grad_fn.saved_tensors` 在 C++ 定义的节点上**不存在**")
    A2 = torch.randn(3, 4, requires_grad=True)
    B2 = torch.randn(4, 5, requires_grad=True)
    mm = (A2 @ B2).sum()
    try:
        mm.grad_fn.saved_tensors
        check("C", "访问 MmBackward0 的 saved_tensors 必须 AttributeError", False, "它没有抛")
    except AttributeError as exc:
        check("C", "C++ 节点上没有 saved_tensors（只有自定义 Function 才有）", True, str(exc)[:60])
    class MyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return a * b
        @staticmethod
        def backward(ctx, g):
            a, b = ctx.saved_tensors
            return g * b, g * a
    aa = torch.randn(3, requires_grad=True)
    bb = torch.randn(3, requires_grad=True)
    out = MyFn.apply(aa, bb)
    check("C", "对照：自定义 Function 的节点**有** saved_tensors（长度 2）", len(out.grad_fn.saved_tensors) == 2, f"{len(out.grad_fn.saved_tensors)} 个")
    sub("C5  `autograd.grad()` **不写** `.grad`；`backward()` 写且**累加**")
    g1 = torch.randn(3, requires_grad=True)
    gr = torch.autograd.grad(g1.sum(), g1)
    print("    autograd.grad(x.sum(), x) 之后 x.grad 是", g1.grad,
          "；返回的梯度是", tuple(gr[0].shape))
    check("C", "autograd.grad() 不写 .grad", g1.grad is None)
    g2 = torch.randn(3, requires_grad=True)
    g2.sum().backward()
    first = g2.grad.clone()
    g2.sum().backward()
    check("C", "backward() 写 .grad 且**累加**", torch.allclose(g2.grad, first * 2), f"{first.tolist()} → {g2.grad.tolist()}")
    sub("C6  `retain_graph` 的默认值是 `create_graph`，不是 `False`")
    import inspect
    sig = str(inspect.signature(torch.autograd.grad))
    print("    torch.autograd.grad 的签名：", sig[:120], "...")
    print("    源码里的那一行：\"%s\"" % SPECS.RETAIN_GRAPH_SOURCE)
    src_lines = open(torch.autograd.__file__, encoding="utf-8").read().splitlines()
    hits = [(i, l.strip()) for i, l in enumerate(src_lines, 1)
            if "retain_graph = create_graph" in l]
    print("    实测命中行：", hits)
    check("C", "源码里确有 `retain_graph = create_graph`", any("retain_graph = create_graph" == l for _i, l in hits), str(hits))
    check("C", "签名里两个参数的默认值不同（retain_graph=None，create_graph=False）", "retain_graph: bool | None = None" in sig.replace("Optional[bool]", "bool | None") or "retain_graph=None" in sig.replace(" ", ""), "见上面的签名原文")
    sub("C7  不写 retain_graph 就连着 backward 两次：第二次必须报错")
    g3 = torch.randn(3, requires_grad=True)
    y3 = (g3 * 2).sum()
    y3.backward()
    try:
        y3.backward()
        check("C", "第二次 backward 必须报错（图已被释放）", False, "它没有报错")
    except RuntimeError as exc:
        check("C", "第二次 backward 报错（图已被释放）", "backward through the graph a second time" in str(exc), " ".join(str(exc).split())[:70])
    sub("C8  ★ 版本差异：inference_mode 里的 forward-mode AD（**五个版本逐一实测**）")
    def jvp_in_inference():
        def f(t):
            return t * 2
        with torch.inference_mode():
            _, tangent = torch.func.jvp(f, (torch.ones(3),), (torch.ones(3),))
            return tangent.tolist()
    tangent = jvp_in_inference()
    print("    torch.func.jvp 在 inference_mode 里的 tangent =", tangent)
    print("    specs.py 记的五版实测结果：")
    for ver, val in SPECS.FORWARD_AD_IN_INFERENCE_MODE.items():
        print("        %-8s %s" % (ver, val))
    table = SPECS.FORWARD_AD_IN_INFERENCE_MODE
    if torch.__version__ in table:
        want = "正确" if table[torch.__version__].startswith("[2") else "全零"
        print("    本版本（%s）在表里，记的是「%s」。" % (torch.__version__, want))
        check("C", "实测与 specs 表里本版本那一行一致",
              (tangent == [2.0, 2.0, 2.0]) if want == "正确" else (tangent == [0.0, 0.0, 0.0]),
              f"实测 {tangent}")
    else:
        print("    ⚠ 本版本不在实测表内（表里只有 %s）——本程序只断言「必须是两种已知形态之一」。"
              % ", ".join(table))
        check("C", "不在表内的版本：实测值必须是已知的两种形态之一", tangent in ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), f"实测 {tangent}")
    check("C", "2.6/2.7/2.8/2.11 的形态是**静默全零**（不抛异常——最容易漏掉的失败形态）",
          all(v.startswith("[0.0") for k, v in table.items() if k != "2.13.0"),
          "见上表")
    print("    ★ 这一行**不能跨版本外推**：2.6/2.7/2.8/2.11 与 2.13 的行为相反。")
    print("      变更点只能定位到 **2.11 与 2.13 之间**；specs.py 里记的分界 2.12 是**假设**，")
    print("      因为本机**没有 2.12 的解释器**——这条不得写成已核实。")


# --- 主流程

def main():
    print("=" * 76)
    print("02 autograd —— 本机实测（真 torch，非迷你实现）")
    print("=" * 76)
    print()
    print("测量条件：torch %s，python %s，numpy %s，CPU-only"
          % (torch.__version__, sys.version.split()[0], np.__version__))
    print("版本绑定：本机 " + VERSIONS + " **五个**解释器都跑过；**2.9/2.10/2.12 本机无解释器**。")
    print("★ 本程序里有一处**版本行为相反**（§C8 的 forward-mode AD：2.6/2.7/2.8/2.11 全零，2.13 正确），"
          "文档必须照着实测写，不能外推。")
    section_1()
    section_A()
    section_B()
    section_C()
    rule("断言汇总")
    print(f"  共 {_STATE['n']} 条：A 类 {_STATE['A']} 条，B 类 {_STATE['B']} 条，"
          f"C 类 {_STATE['C']} 条")
    if _STATE["fail"]:
        print(f"  **不通过 {len(_STATE['fail'])} 条**：")
        for f in _STATE["fail"]:
            print(f"    - {f}")
        return 1
    print("  全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
