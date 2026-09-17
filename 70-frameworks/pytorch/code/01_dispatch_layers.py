#!/usr/bin/env python3
"""01 dispatch 与 aten：eager 那条路的**层数是可数的**。

对应文档：70-frameworks/pytorch/01-dispatch-and-aten.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 的结论，
  但**必须绑定版本**（本机 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 **五个**都跑过；2.9/2.10/2.12 本机无解释器）。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  A —— 两条互不依赖的观察通道（Python dispatch mode / C++ profiler）+ 第三个来源
       （_dispatch_dump_table 的注册层级）互相印证；再加一条「改错预测器必须失败」的反证。
  B —— eager 每次调用要付的 dispatcher 层开销（微秒级），与模型逐项核对。
  C —— 失败行为必须真的跑出来（不是「理论上会」）。

用法：cd code && make 01_dispatch_layers   （或 make run 全跑）
"""

import contextlib
import os
import sys
import tempfile
import time

import torch

import specs as SPECS


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


# ---------------------------------------------------------------------------
# 两条独立的观察通道
# ---------------------------------------------------------------------------

class Recorder(torch.utils._python_dispatch.TorchDispatchMode):
    """通道 1：Python 层的 `TorchDispatchMode`（走 Python key / 线程局部栈）。"""
    def __init__(self):
        self.calls = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append(str(func))
        return func(*args, **(kwargs or {}))


def observe_mode(fn):
    r = Recorder()
    with r:
        fn()
    return list(r.calls)


@contextlib.contextmanager
def silence_usdt():
    """吞掉 profiler 启停时打到 fd 2 的 `USDT: ... profiler_start/stop` 两行。
    窗口内**其它**写入 fd 2 的内容会被原样打印回来——不静默丢弃信息，
    只是不让每条表达式都带两行 kineto 噪音。
    """
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            text = tmp.read().decode("utf-8", "replace")
        keep = [ln for ln in text.splitlines() if not ln.startswith("USDT:")]
        if keep:
            print("\n".join(keep), file=sys.stderr)


def observe_profiler(fn):
    """通道 2：C++ 的 RecordFunction 层（profiler）。
    它与通道 1 **不是同一套机制**：profiler 记录的是 C++ 里真正被调用的 aten op，
    而 mode 记录的是 dispatcher 在 Python key 上的拦截。
    两者在 CompositeImplicitAutograd 的算子上会**系统性地不一致**——见 §A。
    """
    with silence_usdt():
        with torch.autograd.profiler.profile(use_device=None) as prof:
            fn()
        return [ev.name for ev in prof.function_events if ev.name.startswith("aten::")]


def registration_level(aten_name):
    """通道 3：注册表本身（`_dispatch_dump_table`）。
    返回注册行的关键词：'math kernel' 表示它是 CompositeImplicitAutograd
    （被 dispatcher 内联，**不进 Python key**）；'kernel' 表示有真实现。
    """
    try:
        text = torch._C._dispatch_dump_table(aten_name)
    except Exception:
        return "?"
    for line in text.splitlines():
        if line.startswith("Undefined:"):
            return "math kernel" if "math kernel" in line else "kernel"
        if line.startswith("CPU:"):
            return "math kernel" if "math kernel" in line else "kernel"
    return "?"


def bench(fn, n=2000, reps=7):
    """返回每次调用的**最快**一组耗时（微秒）。取最小值以压掉调度抖动。"""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    return best



A = torch.randn(4, 4)
B = torch.randn(4, 4)
AG = torch.randn(4, 4, requires_grad=True)
V1, V2 = torch.randn(4), torch.randn(4)

EXPRESSIONS = [
    ("a + b", lambda: A + B),
    ("torch.add(a, b)", lambda: torch.add(A, B)),
    ("a.add(b)", lambda: A.add(B)),
    ("a.T", lambda: A.T),
    ("a[0]", lambda: A[0]),
    ("a.sum()", lambda: A.sum()),
    ("(ag + b).sum()", lambda: (AG + B).sum()),
    ("一维 v1 @ v2", lambda: V1 @ V2),
    ("二维 a @ b", lambda: A @ B),
    ("+a", lambda: +A),
    ("a.float()（已同 dtype）", lambda: A.float()),
    ("a.shape", lambda: A.shape),
]


def section_1():
    rule("§1  eager 那条路的层数是可数的：一次表达式 = 几次 dispatcher 拦截")
    print("被观察的表达式共用一批张量：a/b 是 (4,4) float32 CPU，ag 同形但 requires_grad=True。")
    print("%-24s %-8s %-42s %s" % ("表达式", "mode 拦截", "mode 观察到的算子", "profiler 看到的 aten op"))
    print("-" * 120)
    rows = []
    for name, fn in EXPRESSIONS:
        mode_calls = observe_mode(fn)
        prof_calls = observe_profiler(fn)
        rows.append((name, mode_calls, prof_calls))
        print("%-24s %-8d %-42s %s" % (name, len(mode_calls),
                                       ", ".join(mode_calls) or "-",
                                       ", ".join(prof_calls) or "-"))
    return rows



def section_A(rows):
    rule("§A  A 算法正确性：三条互不依赖的来源互相印证")
    sub("A1  第三条来源预测第二条通道：注册层级决定 mode 能不能看见「它自己」")
    print("判据（先用注册表预测，再看 mode 的拦截名单）：")
    print("   'math kernel'（CompositeImplicit）⇒ 那个算子**自己**不在拦截名单里；'kernel' ⇒ 在。")
    print("   ⚠ 判据**不是**「拦截数 == 0」：composite 算子会内联展开，展开出来的子算子照样被拦")
    print("     （`reshape` 在连续输入下被拦 1 次，拦到的是 `view`）。第一版就错在这里。")
    # (注册表要查的名字, 期望被拦到的重载名, 调用)
    # ⚠ `dump_table` 对**有重载的**算子必须给重载名（裸 `aten::add` 查出来是空的，
    #   本机实测：2.11.0 / 2.13.0 上 `aten::add` → "?"，`aten::add.Tensor` → "default backend kernel"）。
    probes = [
        ("aten::add.Tensor", "aten.add.Tensor", lambda: torch.ops.aten.add.Tensor(A, B)),
        ("aten::sum", "aten.sum.default", lambda: torch.ops.aten.sum.default(A)),
        ("aten::relu", "aten.relu.default", lambda: torch.ops.aten.relu.default(A)),
        ("aten::reshape", "aten.reshape.default", lambda: torch.ops.aten.reshape.default(A, [16])),
        ("aten::contiguous", "aten.contiguous.default", lambda: torch.ops.aten.contiguous.default(A)),
        ("aten::positive", "aten.positive.default", lambda: torch.ops.aten.positive.default(A)),
    ]
    agree = 0
    for dump_name, overload, fn in probes:
        level = registration_level(dump_name)
        calls = observe_mode(fn)
        self_seen = overload in calls
        predicted = (level != "math kernel")
        ok = (self_seen == predicted) and level != "?"
        agree += ok
        check("A", f"{dump_name} 的可见性被注册层级正确预测", ok, f"注册层级={level} → 预测 '自己可见={predicted}'，实测 {calls}")
    print("  A1 小结：%d/%d 个算子被第三条来源正确预测——「CompositeImplicitAutograd 的算子对" % (agree, len(probes)))
    print("  dispatch mode 不可见」不是传闻，而是**注册层级**决定的可预测行为。")
    sub("A2  反证：把「可见性预测器」故意改成恒 True，A 类判据必须失败")
    def broken_predictor(_level):
        return True     # 故意写错：以为所有算子都「自己可见」
    bad = sum(1 for dump_name, overload, fn in probes
              if broken_predictor(registration_level(dump_name)) != (overload in observe_mode(fn)))
    check("A", "恒 True 的预测器会与实测冲突（A 类因此可证伪）", bad > 0, f"被它判错的算子有 {bad} 个")
    sub("A3  两条通道对「真 kernel」算子必须指向同一个算子")
    for name, fn, base in [("a + b", lambda: A + B, "aten::add"),
                           ("a.sum()", lambda: A.sum(), "aten::sum")]:
        mode_calls = observe_mode(fn)
        prof_calls = observe_profiler(fn)
        # mode 报的是带重载名的 `aten.add.Tensor`；profiler 报的是不带重载名的 `aten::add`
        base_mode = [f"aten::{c.split('.')[1]}" for c in mode_calls if c.count(".") >= 1]
        check("A", f"{name}：mode 与 profiler 都指向 {base}", base in base_mode and base in prof_calls, f"mode={mode_calls} profiler={prof_calls}")
    print("  （profiler 一侧会多出 as_strided/fill_/resolve_conj 等实现细节：它是 C++ 层的完整")
    print("   记录，而 mode 只记 Python key 上的拦截。**两条通道本来就不等**；这里要求的是")
    print("   「同一个算子名双方都在」，不是「输出相同」。）")
    sub("A4  逐表达式核对 specs 的基线表（表在 specs.py，§5 与它逐行一致）")
    baseline = {r[0]: (r[1], r[2]) for r in SPECS.DISPATCH_BASELINE}
    bad = []
    for name, mode_calls, _prof in rows:
        if name not in baseline:
            continue
        want_n, want_ops = baseline[name]
        got_ops = ", ".join(mode_calls) if mode_calls else "-"
        if (len(mode_calls), got_ops) != (want_n, want_ops):
            bad.append((name, (len(mode_calls), got_ops), (want_n, want_ops)))
    check("A", "§1 表的每一行都与 specs.py 的基线表一致（%d 行）" % len(baseline), not bad, f"不一致：{bad}")



def section_2():
    rule("§2  DispatchKeySet 是一个 64 位整数；mode 走的是另一套（线程局部栈）")
    plain = torch.randn(3)
    class Empty(torch.Tensor):
        """空子类：**没有**覆盖 `__torch_dispatch__`。"""
    class WithDispatch(torch.Tensor):
        """覆盖了 `__torch_dispatch__` 的子类。"""
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            with torch._C._DisableTorchDispatch():
                return func(*args, **(kwargs or {}))
    e = torch.Tensor._make_subclass(Empty, torch.randn(3), False)
    w = torch.Tensor._make_subclass(WithDispatch, torch.randn(3), False)
    k_plain, k_e, k_w = (torch._C._dispatch_keys(t) for t in (plain, e, w))
    print("普通 CPU 张量的 keyset            ：", k_plain)
    print("空子类（未覆盖 __torch_dispatch__）：", k_e)
    print("覆盖了 __torch_dispatch__ 的子类   ：", k_w)
    print()
    print("★ 计划里订正过的一条（T2）：**不是「张量子类」就会加 Python key**，而是**只有覆盖了")
    print("  `__torch_dispatch__` 的子类**才会。上面两行就是对照。")
    check("A", "普通 CPU 张量 / 空子类的 keyset 里都没有 Python key", "Python" not in str(k_plain) and "Python" not in str(k_e), f"{k_e}")
    check("A", "覆盖 __torch_dispatch__ 的子类里有 Python + PythonTLSSnapshot", "Python" in str(k_w) and "PythonTLSSnapshot" in str(k_w), str(k_w))
    n_before = torch._C._len_torch_dispatch_stack()
    with Recorder():
        n_inside = torch._C._len_torch_dispatch_stack()
    n_after = torch._C._len_torch_dispatch_stack()
    print("线程局部 dispatch 栈深度：无 mode 时 %d，进 mode 后 %d，出 mode 后 %d" % (n_before, n_inside, n_after))
    check("A", "mode 走线程局部栈：进 mode 深度 +1，出 mode 复原", (n_inside == n_before + 1) and (n_after == n_before), f"{n_before}/{n_inside}/{n_after}")
    # 两个 mode 叠加：后进先出
    order = []
    class Tag(torch.utils._python_dispatch.TorchDispatchMode):
        def __init__(self, tag):
            self.tag = tag
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            order.append(self.tag)
            return func(*args, **(kwargs or {}))
    with Tag("outer"):
        with Tag("inner"):
            A + B
    print("两个 mode 叠加时的调用序列：", order)
    check("A", "两个 mode 叠加是后进先出（内层先看到）", order == ["inner", "outer"], str(order))



def section_3():
    rule("§3  mode 优先于子类 __torch_dispatch__ —— ★ 依据只能是实测")
    order = []
    class Sub(torch.Tensor):
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            order.append("subclass")
            # 必须在关掉 Python key 之后才调用，否则无限递归（见 §C3）
            with torch._C._DisableTorchDispatch():
                return func(*args, **(kwargs or {}))
    s = torch.Tensor._make_subclass(Sub, torch.randn(3), False)
    class Mode(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            order.append("mode")
            return func(*args, **(kwargs or {}))
    with Mode():
        s + 1
    print("两者同时在栈上时，调用序列：", order)
    check("A", "mode 先于子类被调用", order[:2] == SPECS.MEASURED_DISPATCH_ORDER, f"实测 {order[:2]}，specs 记的是 {SPECS.MEASURED_DISPATCH_ORDER}")
    del order[:]
    s2 = torch.Tensor._make_subclass(Sub, torch.randn(3), False)
    r = s2 + 1
    check("A", "无 mode 时子类被调用 1 次且结果正确", order == ["subclass"] and tuple(r.shape) == (3,), f"{order}")



def section_B():
    rule("§B  B 代价模型：不给 dispatch 定价，就说不出「固化进图」省了什么")
    t_add = bench(lambda: A + B)
    t_torchadd = bench(lambda: torch.add(A, B))
    t_sum = bench(lambda: A.sum())
    t_noop = bench(lambda: +A)
    def mode_call():
        with Recorder():
            A + B
    t_mode = bench(mode_call, n=500)
    r = Recorder()
    with r:
        t_inside_mode = bench(lambda: A + B, n=1000)
    print("每次调用的最快单次耗时（本机实测，单位 μs）：")
    for label, val in [("a + b", t_add), ("torch.add(a, b)", t_torchadd),
                       ("a.sum()", t_sum), ("+a（0 次拦截）", t_noop),
                       ("进/出 mode 的开销（含建栈）", t_mode),
                       ("mode 内的 a + b", t_inside_mode)]:
        print("    %-26s %8.3f" % (label, val))
    print()
    print("模型（推导；写出来是为了让每一项都可被上面的数字否证）：")
    print("    T(表达式) ≈ T(空调用基线) + Σ_每个被拦截的算子 T(一层 dispatcher)")
    print("    ⇒ ① 同一条 dispatcher 入口的两个绑定应同价：T(a+b) ≈ T(torch.add)，且差价")
    print("         远小于一个 dispatcher 层（用 mode 内/外的差价当标尺）")
    print("    ⇒ ② 加一层 Python key（mode 内）应显著更贵；③ 0 次拦截的 +a 应更便宜")
    gap = abs(t_add - t_torchadd)
    layer = t_inside_mode - t_add
    print()
    print("    实测差价 |T(a+b) - T(torch.add)| = %.3f μs；一层 Python key 的标尺 = %.3f μs"
          % (gap, layer))
    check("B", "① a+b 与 torch.add 同价（差价小于一层 dispatcher 的量级）", gap < max(layer, 1.0), f"差价 {gap:.3f} μs，标尺 {layer:.3f} μs")
    check("B", "② 多一层 Python key 显著更贵（> 1.5×）", t_inside_mode > t_add * 1.5, f"{t_inside_mode:.3f} vs {t_add:.3f} μs")
    check("B", "③ 0 次拦截的 +a 比 1 次拦截的 a+b 便宜", t_noop < t_add, f"+a {t_noop:.3f} μs vs a+b {t_add:.3f} μs")
    print()
    print("    ★ 口径：这组数只说明**本机 CPU 上 eager 路径的相对开销**，不是 GPU 耗时，也不能")
    print("      外推到别的机器（测量条件见文档 §5 与 specs.py）。")



def section_C():
    rule("§C  C 边界与反例：这些失败行为必须真的发生")
    sub("C1  CompositeImplicitAutograd 的算子对 dispatch mode 不可见")
    nc = torch.randn(4, 4).t()
    cases = [
        ("aten.reshape.default(连续)", lambda: torch.ops.aten.reshape.default(A, [16])),
        ("aten.reshape.default(非连续)", lambda: torch.ops.aten.reshape.default(nc, [16])),
        ("aten.ravel(非连续)", lambda: torch.ops.aten.ravel(nc)),
        ("aten.ravel(连续)", lambda: torch.ops.aten.ravel(A)),
        ("aten.contiguous(非连续)", lambda: torch.ops.aten.contiguous.default(nc)),
        ("aten.contiguous(连续)", lambda: torch.ops.aten.contiguous.default(A)),
    ]
    seen = {}
    for name, fn in cases:
        seen[name] = observe_mode(fn)
        print("    %-32s mode 看到：%s" % (name, ", ".join(seen[name]) or "（0 次）"))
    check("C", "reshape/ravel/contiguous 全部观察不到自己（只看到被内联的替代实现）", not any(any(c.startswith(("aten.reshape", "aten.ravel", "aten.contiguous")) for c in v) for v in seen.values()))
    check("C", "对照：真 kernel 的 relu/sum 是**可见**的（说明不是 mode 坏了）", len(observe_mode(lambda: torch.ops.aten.relu.default(A))) == 1 and len(observe_mode(lambda: torch.ops.aten.sum.default(A))) == 1)
    check("C", "连续输入下 contiguous() **连 dispatcher 都不进**（拦截 0 次）", len(seen["aten.contiguous(连续)"]) == 0)
    print("    ★ 后果：『用 dispatch mode 观察算子』这个流行做法对一大类算子**静默失效**——")
    print("      它不报错，只是让你看不到。第二条通道（profiler）看得到 C++ 层的调用：")
    print("      +a 的 profiler 输出 =", observe_profiler(lambda: +A))
    sub("C2  `_dispatch_has_kernel` 的「必须写重载名」**只对 aten 成立**")
    check("C", "aten 算子：裸名 False、重载名 True", torch._C._dispatch_has_kernel("aten::add") is False and torch._C._dispatch_has_kernel("aten::add.Tensor") is True)
    sub("C3  子类 `__torch_dispatch__` 里直接调 `func(*args)` 会无限递归")
    print("    （torch 自己的 LoggingTensor 之所以没事，是因为它**先 unwrap 再调**：见")
    print("     torch/testing/_internal/logging_tensor.py 的 `func(*tree_map(unwrap, args))`；")
    print("     而同文件的 LoggingTensorMode 就直接调 `func(*args)`。）")
    def make_subclass(handler_body, data=None):
        class S(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return handler_body(func, args, kwargs or {})
        return torch.Tensor._make_subclass(
            S, torch.randn(3) if data is None else data, False)
    def naive(func, args, kwargs):
        return func(*args, **kwargs)                       # 故意不关掉 Python key
    def via_as_subclass(func, args, kwargs):
        plain = [a.as_subclass(torch.Tensor) if isinstance(a, torch.Tensor)
                 and type(a) is not torch.Tensor else a for a in args]
        return func(*plain, **kwargs)
    def disabled(func, args, kwargs):
        with torch._C._DisableTorchDispatch():
            return func(*args, **kwargs)
    for label, body in [("直接 func(*args)", naive),
                        ("用 a.as_subclass(torch.Tensor) 拆包", via_as_subclass)]:
        try:
            make_subclass(body) + 1
            check("C", f"{label} 应当递归到 RecursionError", False, "它没有抛")
        except RecursionError:
            check("C", f"{label} 真的递归到 RecursionError（as_subclass 自己也要进 dispatcher）", True)
    ref = torch.tensor([1.0, 2.0, 3.0])
    got = make_subclass(disabled, ref.clone()) + 1
    check("C", "对照：`torch._C._DisableTorchDispatch()` 之后同一运算正常且数值正确", tuple(got.shape) == (3,) and torch.allclose(got, ref + 1), f"got={got.tolist()}")
    sub("C4  `__torch_function__` 在 API 层、`__torch_dispatch__` 在 aten 层")
    tf_calls = []
    class TFOnly(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            tf_calls.append(str(func))
            return super().__torch_function__(func, types, args, kwargs or {})
    t = torch.Tensor._make_subclass(TFOnly, torch.randn(3), False)
    del tf_calls[:]; _ = t.shape
    n_shape_tf = len(tf_calls)
    del tf_calls[:]; _ = torch.Tensor.dim(t)
    n_dim_tf = len(tf_calls)
    n_dispatch = len(observe_mode(lambda: (t.shape, torch.Tensor.dim(t))))
    print("    `x.shape` 触发 __torch_function__ %d 次；`torch.Tensor.dim(x)` %d 次；"
          % (n_shape_tf, n_dim_tf))
    print("    两者合起来触发 __torch_dispatch__ %d 次。" % n_dispatch)
    check("C", "x.shape / dim(x) 只触发 __torch_function__，根本不进 dispatcher", n_shape_tf >= 1 and n_dim_tf >= 1 and n_dispatch == 0, f"tf={n_shape_tf}/{n_dim_tf} dispatch={n_dispatch}")
    sub("C5  `__add__` 不经过 Python 的 `torch.Tensor.add`")
    same = torch.Tensor.add is torch._C.TensorBase.add
    src = open(torch._tensor.__file__, encoding="utf-8").read()
    has_def_add = "\n    def add(" in src
    print("    `torch.Tensor.add is torch._C.TensorBase.add` →", same)
    print("    `torch/_tensor.py` 里有 `def add` 吗 →", has_def_add)
    check("C", "两者是同一个 C 绑定，且 Python 层没有 def add", same and not has_def_add)



def main():
    print("=" * 76)
    print("01 dispatch 与 aten —— 本机实测（真 torch，非迷你实现）")
    print("=" * 76)
    print("测量条件：torch %s，python %s，CPU-only（torch.cuda.is_available() = %s）"
          % (torch.__version__, sys.version.split()[0], torch.cuda.is_available()))
    print("版本绑定：本程序在哪个版本上跑，§5 的数字就只属于那个版本；")
    print("          本机另有 2.6.0 / 2.7.1 两个解释器（见 Makefile 的探测）；")
    print("          **本机无 2.9/2.10/2.12 的解释器，不声称覆盖这三个 minor。**")
    rows = section_1()
    section_A(rows)
    section_2()
    section_3()
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
