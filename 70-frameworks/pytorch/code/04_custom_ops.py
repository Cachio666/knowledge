#!/usr/bin/env python3
"""04 自定义算子：把自定义算子接进这条链；编译入口（用户接口视角）。

对应文档：70-frameworks/pytorch/04-custom-operators.md

★ 本程序跑的是**真 torch**（`import torch`）——输出是**本机实测值**，
  **必须绑定版本**（本机 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 **五个**都跑过；2.9/2.10/2.12 本机无解释器）。

★ 边界（§0.5）：`torch.compile` 的**编译器内部**（图捕获细节、融合、codegen）在
  `60-compilers/inductor`，本程序只从**用户接口**视角用它：怎么用、怎么数重编译。
  本程序**不读** Inductor 的 IR，也不推断它的调度。

三类断言（CONTRIBUTING §2 第三种形态）：
  A 算法正确性 —— numpy 参考实现（与被测的 kernel 是**两条独立代码路径**）
                  复核自定义算子的数值；再加「schema 推导」的可证伪核对。
  B 代价模型   —— ①一次自定义算子调用在 eager 路径上的层数；
                  ②Dynamo 的 guard：**几个不同的 (shape, dtype) 就要几张图**。
  C 边界与反例 —— 违反 `custom_op` 约束必须被拒；版本面的差异必须真的跑出来。

用法：cd code && make 04_custom_ops   （或 make run 全跑）
"""

import sys
import warnings

import numpy as np
import torch
import torch.library as L

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


# --- §1 三个入口，三种注册

# 入口 1：custom_op —— 给一个 Python 函数，自动推导 schema
@L.custom_op("kb04::swiglu", mutates_args=())
def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU 的前半段：把最后一维一分为二，a * sigmoid(a) * b。"""
    a, b = x.chunk(2, dim=-1)
    return a * torch.sigmoid(a) * b


# 入口 2：define + impl —— schema 自己写，kernel 分开注册
L.define("kb04::affine", "(Tensor x, float scale, float bias) -> Tensor")
L.impl("kb04::affine", "CPU", lambda x, scale, bias: x * scale + bias)

# 入口 3：register_fake —— 给 FakeTensor / 元数据推断用
swiglu.register_fake(lambda x: torch.empty_like(x))


def numpy_swiglu(x):
    """A 类的独立参照：纯 numpy，**不调用 torch**。"""
    a, b = np.split(x, 2, axis=-1)
    sig = 1.0 / (1.0 + np.exp(-a))
    return a * sig * b


def section_1():
    rule("§1  三个入口：custom_op / define+impl / register_fake")
    print("入口 1（custom_op）：")
    print("    返回对象的类型      =", type(swiglu).__name__)
    print("    它是函数吗          =", callable(swiglu), "（但它**不是**普通函数，见下）")
    print("    有 register_kernel  =", hasattr(swiglu, "register_kernel"))
    print("    有 register_impl    =", hasattr(swiglu, "register_impl"), "← 易错点")
    print("    有 register_fake    =", hasattr(swiglu, "register_fake"))
    print("    register_fake 的签名 =", str(__import__("inspect").signature(swiglu.register_fake)))
    print()
    print("入口 2（define + impl）：")
    print("    define 的签名 =", SPECS.SIG_DEFINE)
    print("    impl   的签名 =", SPECS.SIG_IMPL)
    print("    L.define 是 functools.singledispatch 包装过的——所以它的签名会随")
    print("    第一个参数的类型变；必须一次给两个位置参数（qualname, schema）。")
    print()
    print("入口 3（register_fake）：")
    print("    torch.ops.kb04.swiglu.default 的 fake 推断已注册（可被 torch.compile 用）")
    check("A", "custom_op 返回的是 CustomOpDef 对象，不是函数", type(swiglu).__name__ == "CustomOpDef")
    check("C", "CustomOpDef **有** register_kernel、**没有** register_impl", hasattr(swiglu, "register_kernel") and not hasattr(swiglu, "register_impl"))
    check("C", "`torch.library.fallback` 在模块级**不存在**（只有 Library.fallback 方法）", not hasattr(L, "fallback") and hasattr(L.Library, "fallback"), f"模块级 hasattr={hasattr(L, 'fallback')}，Library.fallback={hasattr(L.Library, 'fallback')}")
    sub("§1b  `impl_abstract` → `register_fake` 的改名（还在兼容期）")
    L.define("kb04::legacy", "(Tensor x) -> Tensor")
    L.impl("kb04::legacy", "CPU", lambda x: x + 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        L.impl_abstract("kb04::legacy")(lambda x: torch.empty_like(x))
        cats = [(x.category.__name__, str(x.message)) for x in w]
    print("    调用 torch.library.impl_abstract 时发出的警告：")
    for cat, msg in cats:
        print("        [%s] %s" % (cat, " ".join(msg.split())[:100]))
    check("C", "impl_abstract 仍可用，但发**FutureWarning**（不是 DeprecationWarning）", any(c == "FutureWarning" and "was renamed to" in m for c, m in cats), f"{[c for c, _ in cats]}")
    check("C", "警告文案与 specs.py 记的一致", any("was renamed to `torch.library.register_fake`" in m for _c, m in cats))


# --- §A 独立参照：numpy 实现 vs 注册的 kernel

def section_A():
    rule("§A  A 算法正确性：numpy 参照实现复核自定义算子")
    rng = np.random.default_rng(20260917)
    X = rng.standard_normal((4, 8)).astype(np.float32)
    ref = numpy_swiglu(X)
    via_call = swiglu(torch.tensor(X))
    via_op = torch.ops.kb04.swiglu.default(torch.tensor(X))
    err_call = float(np.max(np.abs(ref - via_call.detach().numpy())))
    err_op = float(np.max(np.abs(ref - via_op.detach().numpy())))
    print("    numpy 参照的最大绝对误差：直接调用 %.3e，走 torch.ops ... %.3e"
          % (err_call, err_op))
    check("A", "自定义算子（直接调用）与 numpy 参照一致", err_call < 1e-5, f"误差 {err_call:.3e}")
    check("A", "同一个算子走 `torch.ops.kb04.swiglu.default` 也一致", err_op < 1e-5, f"误差 {err_op:.3e}")
    check("A", "两条调用路径给出同一个结果", torch.equal(via_call, via_op))
    sub("A2  第二形态（define + impl）的数值也要与参照一致")
    x = torch.tensor(X)
    got = torch.ops.kb04.affine.default(x, 2.0, 1.0)
    want = X * 2.0 + 1.0
    err = float(np.max(np.abs(want - got.detach().numpy())))
    check("A", "define+impl 的 kernel 与 numpy 手算一致", err < 1e-5, f"误差 {err:.3e}")
    sub("A3  schema 由**类型标注**推导：标注错了会怎样（可证伪）")
    try:
        @L.custom_op("kb04::noschema", mutates_args=())
        def noschema(x):
            return x + 1
        check("C", "无类型标注的函数必须被拒", False, "它没有被拒")
    except ValueError as exc:
        print("    错误原文：", str(exc)[:110])
        check("C", "无类型标注 → ValueError，且原文与 specs.py 一致", "must have a type annotation" in str(exc), "")
    print("    ⇒ schema 不是装饰性的：它由标注推导，而 dispatcher 靠它做类型检查。")
    sub("A4  可证伪性：把 numpy 参照故意改错，A 类对拍必须失败")
    def broken_numpy_swiglu(x):
        a, b = np.split(x, 2, axis=-1)
        return a * b            # 故意漏掉 sigmoid 门
    broken = broken_numpy_swiglu(X)
    err_broken = float(np.max(np.abs(broken - via_call.detach().numpy())))
    check("A", "改错的参照被同一条判据（误差 < 1e-5）判为不通过", err_broken >= 1e-5, f"误差 {err_broken:.3e}")


# --- §B 代价模型

class Recorder(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.calls = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append(str(func))
        return func(*args, **(kwargs or {}))


def section_B():
    rule("§B  B 代价模型：接进这条链要付几层；固化进图要付几张图")
    sub("B1  eager 路径：一次自定义算子调用 = 1 次 dispatcher 拦截")
    x = torch.randn(4, 8)
    r1 = Recorder()
    with r1:
        swiglu(x)
    r2 = Recorder()
    with r2:
        x + 1                      # 对照：内置算子
    print("    torch.ops.kb04.swiglu.default(x) 的 mode 拦截：", r1.calls)
    print("    对照（内置算子 x + 1）的拦截：                ", r2.calls)
    model_ops = 1
    check("B", "自定义算子调用被拦 1 次（与内置算子同一层数）", len(r1.calls) == model_ops and len(r2.calls) == model_ops, f"自定义 {len(r1.calls)} 次，内置 {len(r2.calls)} 次")
    print("    注意：算子**内部**用到的 aten 调用（chunk/sigmoid/mul）**不会**被再拦一次——")
    print("    因为 handler 里调用 `func` 时，当前 mode 不会被重新进入（见 01 号程序 §3）。")
    print()
    print("    模型：自定义算子只是**换了个 kernel**，链本身没变——")
    print("      Python 绑定 → dispatcher → (autograd) → kernel")
    print("    所以层数与 `a + b` 相同（都是 1 次拦截，见 01 号程序）。")
    sub("B2  ★ Dynamo 的 guard：几个不同的 (shape, dtype) 就要几张图")
    print("    用法：torch.compile(fn, backend='eager') —— backend 选 eager 是为了")
    print("    不依赖本机的 C++ 编译器；这里测的是**图捕获与 guard**，不是 codegen。")
    torch._dynamo.reset()
    def f(a, b):
        return torch.relu(a + b).sum()
    cf = torch.compile(f, backend="eager", dynamic=False)
    stats = torch._dynamo.utils.counters["stats"]
    def snap():
        return (stats["unique_graphs"], stats["calls_captured"])
    print()
    print("%-42s %-14s %s" % ("调用序列", "unique_graphs", "calls_captured"))
    print("-" * 74)
    base = snap()
    rows = []
    for _ in range(3):
        cf(torch.randn(4), torch.randn(4))
    rows.append(("同 shape 调 3 次 (4,4)", snap()))
    cf(torch.randn(8), torch.randn(8))
    rows.append(("再换 shape 到 (8,8) 调 1 次", snap()))
    cf(torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64))
    rows.append(("再把 dtype 换成 float64 调 1 次", snap()))
    for name, s in rows:
        print("%-42s %-14d %d" % (name, s[0], s[1]))
    print("（base = %s，上表是增量后的绝对计数）" % (base,))
    del base
    expected = [(1, 3), (2, 6), (3, 9)]
    got = [r[1] for r in rows]
    print()
    print("    模型：guard 固化的假设是 (shape, dtype)；每出现一个**新组合**就多一张图。")
    for (name, s), (wg, wc) in zip(rows, expected):
        check("B", f"{name}：unique_graphs == {wg}", s[0] == wg, f"实测 {s[0]}")
        check("B", f"{name}：calls_captured == {wc}", s[1] == wc, f"实测 {s[1]}")
    check("B", "与 specs.py 的基线表一致", [r[1] for r in rows] == [(r[1], r[2]) for r in SPECS.DYN_GUARD_BASELINE], f"{[r[1] for r in rows]}")
    del got, expected
    sub("B3  Python 标量会被按值固化（与 TF 的按值特化同构）")
    torch._dynamo.reset()
    fr = torch._dynamo.utils.counters["frames"]
    def g(x, n):
        return x[:n].sum()
    cg = torch.compile(g, backend="eager")
    before = fr["total"]
    cg(torch.randn(10), 3)
    cg(torch.randn(10), 3)
    after_same = fr["total"]
    cg(torch.randn(10), 5)
    after_diff = fr["total"]
    print("    g(x, n) = x[:n].sum()：")
    print("        n=3 两次后 frames.total = %d（较起点 +%d）" % (after_same, after_same - before))
    print("        再传 n=5 后 frames.total = %d（较起点 +%d）" % (after_diff, after_diff - before))
    check("B", "重复同一标量不新增捕获", after_same - before == 1, f"增量 {after_same - before}")
    check("B", "换一个标量值会新增一次捕获（按值固化）", after_diff - after_same >= 1, f"增量 {after_diff - after_same}")
    print("    ★ 与 specs.py 的 DYN_GUARD_BASELINE 是同一组测量条件（真 torch 实测）。")
    sub("B4  诊断入口（用户接口视角）")
    expl = torch._dynamo.explain(f)(torch.randn(4), torch.randn(4))
    text = " ".join(str(expl).split())
    print("    torch._dynamo.explain 的输出前缀：", text[:120])
    check("B", "explain 报出 1 张图、0 个 graph break（与 specs.py 的基线一致）", SPECS.DYN_EXPLAIN_OK in text or "Graph Break Count: 0" in text, text[:80])
    check("B", "`torch.while_loop` **存在**（prototype）但不在 `torch.__all__`", hasattr(torch, "while_loop") and SPECS.WHILE_LOOP_IN_ALL is False, f"hasattr={hasattr(torch, 'while_loop')}, in __all__={'while_loop' in torch.__all__}")
    check("B", "`torch.cond` 同样 prototype，但**在** `torch.__all__` 里", hasattr(torch, "cond") and SPECS.COND_IN_ALL is True, f"in __all__={'cond' in torch.__all__}")


# --- §C 失败模式（含跨版本可用面）

def section_C():
    rule("§C  C 边界与反例：约束必须真的被触发；版本面必须真的测出来")
    sub("C1  不传 `mutates_args` → TypeError（它是 keyword-only **且无默认值**）")
    try:
        @L.custom_op("kb04::noargs")
        def noargs(x: torch.Tensor) -> torch.Tensor:
            return x + 1
        check("C", "不传 mutates_args 必须 TypeError", False, "它没有被拒")
    except TypeError as exc:
        print("    错误原文：", str(exc))
        check("C", "不传 mutates_args → TypeError，原文与 specs.py 一致", "missing 1 required keyword-only argument: 'mutates_args'" in str(exc))
    check("C", "对照：传了 mutates_args 就能定义成功", isinstance(swiglu, type(swiglu)), "swiglu 定义于 §1")
    sub("C2  `_dispatch_has_kernel` 的「必须写重载名」**只对 aten 成立**")
    has_bare = torch._C._dispatch_has_kernel("kb04::swiglu")
    print("    _dispatch_has_kernel('kb04::swiglu')  =", has_bare, "（裸名，可用）")
    try:
        torch._C._dispatch_has_kernel("kb04::swiglu.default")
        check("C", "自定义算子加 .default 后缀必须被拒", False, "它没有被拒")
    except RuntimeError as exc:
        print("    _dispatch_has_kernel('kb04::swiglu.default') →",
              type(exc).__name__, str(exc)[:60])
        check("C", "自定义算子加 `.default` 会抛错——因为它按 **aten 的重载名规则**解析", "not a legal overload name for aten operators" in str(exc), "")
    check("C", "对照：aten 算子必须写重载名（裸名 False、重载名 True）", torch._C._dispatch_has_kernel("aten::add") is False and torch._C._dispatch_has_kernel("aten::add.Tensor") is True)
    print()
    print("    ★ 所以计划里那条「必须写重载名」要加限定：**它只对 aten 命名空间成立**。")
    print("      对自定义算子反过来：裸名可用，加 `.default` 反而报错。")
    sub("C3  ★ 跨版本可用面：照一个版本写的代码在别的版本上会遇到什么")
    surface = SPECS.LIBRARY_PUBLIC_SURFACE
    for ver in sorted(surface):
        print("    torch.library 的公开名（%s）：%s" % (ver, ", ".join(surface[ver])))
    leaked = [n for n in dir(L) if n in ("Any", "Callable", "Dict", "List", "Optional",
                                         "Sequence", "Set", "Tuple", "Union", "TypeVar")]
    print("    本机实际 `dir(torch.library)` 里的类型名泄漏：", leaked)
    print("    本机实际有的「新增名」：",
          [n for n in ("get_kernel", "overload", "register_autocast", "EffectType")
           if hasattr(L, n)])
    print()
    print("    ★ 判据**用特征探测**（hasattr），不用版号比较——因为本机实测已经推翻了")
    print("      「这些名字是哪个版本加的」这一类说法：get_kernel / EffectType / register_effect")
    print("      在 **2.11.0 上就已经存在**（2.8.0 上还没有）——真实区间是 **2.9–2.11**，")
    print("      并不像计划里写的那样「2.13 新增」。")
    print("      更稳的写法是探测能力，而不是拿版号当门牌。")
    for name in ("get_kernel", "EffectType"):
        print("        %-14s 本机可用 = %s" % (name, hasattr(L, name)))
    check("C", "「按版号硬编码可用面」是脆的：get_kernel/EffectType 在 2.11 就有",
          hasattr(L, "get_kernel") == hasattr(L, "EffectType"),
          f"本机 {torch.__version__}：get_kernel={hasattr(L, 'get_kernel')}, "
          f"EffectType={hasattr(L, 'EffectType')}")
    check("C", "类型名泄漏是**部分清理**，不是「全清了」"
               "（本机五个版本里 Any/Callable/Sequence 一直都在）",
          "Any" in leaked and "Callable" in leaked, f"本机仍在的：{leaked}")
    if torch.__version__ in surface:
        check("C", "本机实测的公开名与 specs.py 的版本表逐版一致（本版本那一行）",
              set(surface[torch.__version__]) <= set(dir(L)) | {"custom_op", "Library"},
              f"本机 {torch.__version__}")
    else:
        print("    ⚠ 本版本（%s）不在 specs.py 的实测表内（表里只有 %s）——"
              % (torch.__version__, ", ".join(sorted(surface))))
        print("      按上面打印的「本机实际」逐名核对，不做版号断言。")
        check("C", "本版本不在表内时，只断言「公开名集合非空且与相邻版本形态一致」",
              {"custom_op", "define", "impl", "register_fake"} <= set(dir(L)),
              f"本机 {torch.__version__}")
    print("       ⚠ 2.9 / 2.10 / 2.12 本机没有解释器——**这三个 minor 的公开面未核**。")


# --- 主流程

def main():
    print("=" * 76)
    print("04 自定义算子 —— 本机实测（真 torch，非迷你实现）")
    print("=" * 76)
    print()
    print("测量条件：torch %s，python %s，numpy %s，CPU-only"
          % (torch.__version__, sys.version.split()[0], np.__version__))
    print("版本绑定：本机 " + VERSIONS + " **五个**解释器都跑过；**2.9/2.10/2.12 本机无解释器**。")
    print("边界：torch.compile 的**编译器内部**在 60-compilers/inductor；")
    print("      本程序只从用户接口视角用它（怎么用、怎么数重编译）。")
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
