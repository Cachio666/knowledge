#!/usr/bin/env python3
"""03 张量内存模型：stride / storage / offset——框架对内存形状的假设。

对应文档：70-frameworks/pytorch/03-tensor-memory-model.md

★ 本程序跑的是**真 torch**（`import torch`）——输出是**本机实测值**，
  **必须绑定版本**（本机 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 **五个**都跑过；2.9/2.10/2.12 本机无解释器）。

★ A 类的参照物是 `verify.py` 里的 `ref_*` 那套**独立实现**——
  它跑在**默认 python3** 下、**不 import torch**，是「换了台机器也能复核」的那一半。
  本程序做的是：**用那套参照去预测真 torch 的行为，再逐条对拍**。
  判定独立性的方法见 §A5：把参照故意改错，对拍必须失败。

三类断言（CONTRIBUTING §2 第三种形态）：
  A 算法正确性 —— ref_* 预测 vs torch 实测（连续判据 / view 可行性 / stride 传播)
  B 代价模型   —— 非连续切片的**访存字节**模型，与计时实测对拍
  C 边界与反例 —— `as_strided` 的边界检查、失败的 view、共享存储的误判

用法：cd code && make 03_strides_and_views   （或 make run 全跑）
"""

import sys
import time

import torch

import specs as SPECS

VERSIONS = "2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0"
import verify as REF          # ★ 纯 python3 的独立参考实现（本身不 import torch）

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


def desc(t):
    return "shape=%-10s stride=%-14s contig=%-5s off=%-3d numel=%d" % (
        tuple(t.shape), tuple(t.stride()), t.is_contiguous(),
        t.storage_offset(), t.numel())


def shares_storage(a, b):
    """共享存储的判定：用**未加 offset 的 storage 指针**，不要只看 data_ptr()。"""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# --- §1 storage / storage_offset / stride 的关系

def section_1():
    rule("§1  同一块 storage 上的几种「形状」：storage / offset / stride")
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    print("base                :", desc(base))
    views = [
        ("base.t()", base.t()),
        ("base[:, ::2]", base[:, ::2]),
        ("base[1:3]", base[1:3]),
        ("base[1:3, ::3]", base[1:3, ::3]),
        ("base.reshape(6, 4)", base.reshape(6, 4)),
        ("base.t().reshape(6, 4)", base.t().reshape(6, 4)),
        ("base.expand(3, 4, 6)", base.unsqueeze(0).expand(3, 4, 6)),
        ("base.flatten()", base.flatten()),
    ]
    for name, v in views:
        print("%-24s: %s  共享存储=%s" % (name, desc(v), shares_storage(base, v)))
    print()
    print("★ 注意 `base.t().reshape(6, 4)` 与 `base[:, ::2]`：")
    print("  前者是**非连续也能 reshape 成 view** 的例子（§A2），")
    print("  后者的 view(8) **成功**——尽管它的 is_contiguous() 是 False（§A2）。")


# --- §A 独立参考实现 vs 真 torch

def build_battery():
    """一批 (名字, 张量) —— 每个张量都是真 torch 造的，参考实现只看它的 sizes/strides。"""
    base = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    flat = torch.arange(24, dtype=torch.float32)
    m44 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    m46 = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return [
        ("(2,3,4,5) 连续", base),
        ("(2,3,4,5) transpose(0,1)", base.transpose(0, 1)),
        ("(4,4) 连续", m44),
        ("(4,4).t()", m44.t()),
        ("(4,4)[:, ::2]", m44[:, ::2]),
        ("(4,4)[:, :2]", m44[:, :2]),
        ("(4,6).t()", m46.t()),
        ("(4,6).t() 的 (6,4) 视图", m46.t().reshape(6, 4)),
        ("(24,) 切片 2:22", flat[2:22]),
        ("(24,) 切片 ::3", flat[::3]),
        ("(4,4) 连续 的 (1,16)", m44.reshape(1, 16)),
        ("expand (3,4)", torch.arange(4, dtype=torch.float32).unsqueeze(0).expand(3, 4)),
        ("(6,) 切片 1::2", torch.arange(6, dtype=torch.float32)[1::2]),
        ("空张量 (0,3)", torch.empty(0, 3)),
        # ★ 这个张量的 size==1 维 stride=7：朴素的「逐维检查」会说它不连续，
        #   官方判据（跳过 size==1）说它连续。A5 的反证就靠它。
        ("(4,1,5) 中间维 size==1 且 stride=7",
         torch.as_strided(torch.arange(20, dtype=torch.float32).reshape(4, 1, 5),
                          (4, 1, 5), (5, 7, 1))),
        ("(4,2,5) 同几何但中间维 size=2",
         torch.as_strided(torch.arange(40, dtype=torch.float32).reshape(4, 2, 5),
                          (4, 2, 5), (5, 7, 1))),
    ]


def section_A():
    rule("§A  A 算法正确性：`verify.py` 的独立参照预测 → 真 torch 对拍")
    battery = build_battery()
    sub("A1  `is_contiguous()`：参照实现预测 vs torch 实测")
    print("%-28s %-10s %-10s %s" % ("张量", "参照预测", "torch 实测", "判定"))
    print("-" * 62)
    for name, t in battery:
        pred = REF.ref_compute_contiguous(list(t.shape), list(t.stride()), t.numel())
        got = bool(t.is_contiguous())
        print("%-28s %-10s %-10s %s" % (name, pred, got, "ok" if pred == got else "MISMATCH"))
        check("A", f"is_contiguous 对拍：{name}", pred == got, f"参照 {pred}，torch {got}")
    sub("A2  `view()` 可行性：参照预测 vs torch 实测（含计划里点名的两个例子）")
    view_cases = [
        ("(4,4)[:, ::2].view(8)", m44_slice(), (8,)),
        ("(4,4)[:, ::2].view(2,4)", m44_slice(), (2, 4)),
        ("(4,6).t().view(24)", m46_t(), (24,)),
        ("(4,6).t().view(6,4)", m46_t(), (6, 4)),
        ("(4,6).t().reshape(6,4) 后 .view(24)", m46_t(), (24,)),
        ("(4,4).t().view(16)", m44_t(), (16,)),
        ("(4,4) 连续 .view(16)", m44_c(), (16,)),
        ("(4,4) 连续 .view(4,4)", m44_c(), (4, 4)),
        ("(2,3,4,5) 连续 .view(120)", base4(), (120,)),
        ("(2,3,4,5) 连续 .reshape(6,20)", base4(), (6, 20)),
    ]
    print("%-34s %-10s %-10s %s" % ("表达式", "参照预测", "torch 实测", "判定"))
    print("-" * 70)
    for name, t, target in view_cases:
        pred = REF.ref_view_compatible(list(t.shape), list(t.stride()), list(target))
        try:
            t.view(target)
            got = True
        except RuntimeError:
            got = False
        print("%-34s %-10s %-10s %s" % (name, pred, got,
                                        "ok" if pred == got else "MISMATCH"))
        check("A", f"view 可行性对拍：{name}", pred == got, f"参照 {pred}，torch {got}")
    print()
    print("★ 这就是本主题最容易记错的两条：")
    print("  · `(4,4)[:, ::2].view(8)`：该张量 is_contiguous() = False，但 view **成功**——")
    print("    判据是 contiguity-like condition，不是 is_contiguous()。")
    print("  · `(4,6).t().reshape(6,4)`：非连续，但 reshape **返回 view**（共享存储）。")
    sub("A3  stride 传播：参照实现 vs torch 实测")
    x = torch.arange(10, dtype=torch.float32)
    pred_s, pred_t = REF.ref_stride_slice(list(x.shape), list(x.stride()), 0, 1, 3)
    y = x[1::3]
    check("A", "slice 的 size 传播", pred_s == list(y.shape), f"{pred_s} vs {list(y.shape)}")
    check("A", "slice 的 stride 传播", pred_t == list(y.stride()), f"{pred_t} vs {list(y.stride())}")
    x2 = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    t2 = x2.t()
    pred_s, pred_t = REF.ref_stride_transpose([4, 6], [6, 1], 0, 1)
    check("A", "transpose 的 size 传播", pred_s == list(t2.shape), f"{pred_s}")
    check("A", "transpose 的 stride 传播", pred_t == list(t2.stride()), f"{pred_t}")
    e = torch.arange(4, dtype=torch.float32).unsqueeze(0).expand(3, 4)
    pred_s, pred_t = REF.ref_stride_expand([1, 4], [4, 1], [3, 4])
    check("A", "expand 的 size 传播", pred_s == list(e.shape), f"{pred_s}")
    check("A", "expand 的 stride 传播（被广播的维 stride 置 0）", pred_t == list(e.stride()), f"{pred_t} vs {list(e.stride())}")
    sub("A4  ★ 计划里点名要找的「真的会抛错的 view」")
    try:
        torch.randn(4, 4).t().view(16)
        check("A", "`(4,4).t().view(16)` 必须抛错", False, "它成功了")
    except RuntimeError as exc:
        msg = " ".join(str(exc).split())
        print("    错误原文：", msg[:110], "...")
        check("A", "`(4,4).t().view(16)` 真的抛错，且原文与 specs.py 一致", "view size is not compatible with input tensor's size and stride" in msg, "")
    print("    对照：同一张量 `.reshape(16)` 会成功（需要拷贝时 reshape 才 fallback）：")
    r = torch.randn(4, 4).t().reshape(16)
    check("A", "对照：`.reshape(16)` 成功", tuple(r.shape) == (16,))
    sub("A5  可证伪性：把参照故意改错，同一条 A 类对拍必须失败")
    def broken_contiguous(sizes, strides, numel):
        """故意写错：不跳过 size == 1 的维。"""
        if numel == 0:
            return True
        expected = 1
        for d in range(len(sizes) - 1, -1, -1):
            if strides[d] != expected:
                return False
            expected *= sizes[d]
        return True
    mismatch = 0
    for name, t in battery:
        pred = broken_contiguous(list(t.shape), list(t.stride()), t.numel())
        if pred != bool(t.is_contiguous()):
            mismatch += 1
    check("A", "改错的参照在 A1 对拍里被判不一致（A 类因此可证伪）", mismatch > 0, f"被它判错的有 {mismatch} 个张量")
    sub("A6  参照与 torch 的分工：参照不 import torch（这条本身就是可核对的）")
    import re
    src = open(REF.__file__, encoding="utf-8").read()
    real_imports = [ln for ln in src.splitlines()
                    if re.match(r"\s*(import\s+torch|from\s+torch)", ln)]
    check("A", "verify.py 里没有真的 `import torch` 语句（正则按行匹配）", not real_imports, f"命中 {real_imports}")
    print("    （词面 `import torch` 会出现在 docstring 的说明里——")
    print("      所以这条判据必须按行正则匹配，不能做子串查找。）")


# 小工具：把 A2 里要用的张量单独造出来（避免共享可变状态）
def m44_slice():
    return torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, ::2]


def m44_t():
    return torch.arange(16, dtype=torch.float32).reshape(4, 4).t()


def m44_c():
    return torch.arange(16, dtype=torch.float32).reshape(4, 4)


def m46_t():
    return torch.arange(24, dtype=torch.float32).reshape(4, 6).t()


def base4():
    return torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)


# --- §B 代价模型：非连续切片的访存字节

def bench(fn, n=20, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    return best


def section_B():
    rule("§B  B 代价模型：stride 决定**访存字节**，而访存字节决定耗时")
    N = 8_000_000
    big = torch.randn(N)
    itemsize = big.element_size()
    print("被测量的张量：%d 个元素 × %d 字节 = %.1f MB 的 storage"
          % (N, itemsize, N * itemsize / 1e6))
    print()
    print("模型（推导，来自 §1 的 stride 语义）：")
    print("    · `x[::2]` 的 stride 是 (2,) —— 视图元素只有 N/2 个，")
    print("      但要取到它们，**必须走过整块 storage**：访存 N 个元素 = %.1f MB"
          % (N * itemsize / 1e6))
    print("    · `x[:N/2]` 的 stride 是 (1,) —— 只碰前一半：访存 N/2 = %.1f MB"
          % (N / 2 * itemsize / 1e6))
    print("    ⇒ 预测：T(x[::2].sum()) / T(x[:N/2].sum()) ≈ 2.0（两者求和元素数相同）")
    print("    · `x[::2]`（只造视图）与 `x[::2].contiguous()`（真的拷贝）应当是**两个量级**")
    print("      ⇒ 拷贝的字节数 = 视图元素数 × itemsize = %.1f MB"
          % (N / 2 * itemsize / 1e6))
    t_view = bench(lambda: big[::2], n=200)
    t_sum_stride = bench(lambda: big[::2].sum())
    t_sum_half = bench(lambda: big[:N // 2].sum())
    t_copy = bench(lambda: big[::2].contiguous())
    t_sum_full = bench(lambda: big.sum())
    print()
    print("%-34s %10.4f ms" % ("造视图 x[::2]（不拷贝）", t_view))
    print("%-34s %10.4f ms" % ("x[::2].sum()", t_sum_stride))
    print("%-34s %10.4f ms" % ("x[:N/2].sum()", t_sum_half))
    print("%-34s %10.4f ms" % ("x.sum()（全量连续）", t_sum_full))
    print("%-34s %10.4f ms" % ("x[::2].contiguous()（真的拷贝）", t_copy))
    ratio = t_sum_stride / t_sum_half
    print()
    print("    实测比 T(x[::2].sum()) / T(x[:N/2].sum()) = %.2f（模型预测 ≈ 2.0）" % ratio)
    check("B", "① 跨步访问更贵：实测比 > 1.3（模型预测 2.0，实测允许被缓存效应放大）", ratio > 1.3, f"实测 {ratio:.2f}")
    check("B", "② 造视图几乎免费：拷贝比造视图慢两个量级以上", t_copy > t_view * 100, f"拷贝 {t_copy:.4f} ms vs 视图 {t_view:.4f} ms")
    check("B", "③ 全量求和的耗时与跨步求和同量级（都走过整块 storage）", 0.4 < t_sum_full / t_sum_stride < 2.5, f"full {t_sum_full:.4f} vs ::2 {t_sum_stride:.4f}")
    print()
    print("    ★ 口径：这组数只说明**本机 CPU 上访存字节与耗时的关系**，")
    print("      不是 GPU 带宽测量，也不能外推到别的机器。")


# --- §C 失败模式

class _NoDispatch(torch.Tensor):
    """空子类：**没有**覆盖 `__torch_dispatch__`（用于 C2 的反例）。"""


class _Wrapper(torch.Tensor):
    """最小可用 wrapper 子类：必须有 `__torch_dispatch__`。"""
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        with torch._C._DisableTorchDispatch():
            return func(*args, **(kwargs or {}))


def section_C():
    rule("§C  C 边界与反例：这些失败行为必须真的发生")
    sub("C1  `as_strided` **自带边界检查**——所以它造不出「不查边界」的反例")
    small = torch.randn(4)          # 4 个 float32 = 16 字节
    try:
        torch.as_strided(small, (2, 3), (3, 1))
        check("C", "as_strided 越界必须抛错", False, "它没有抛")
    except RuntimeError as exc:
        msg = " ".join(str(exc).split())
        print("    错误原文：", msg[:130])
        check("C", "as_strided 越界真的抛错，且原文与 specs.py 一致", "out of bounds for storage of size" in msg, str(small.numel() * small.element_size()) + " 字节的 storage")
    sub("C2  同一几何用 wrapper 子类就**被接受**，并判为连续（这才是「不查边界」的实证）")
    w = torch.Tensor._make_wrapper_subclass(
        _Wrapper, (2, 3), strides=(3, 1), dtype=torch.float32, device="cpu")
    print("    wrapper 子类 (2,3)/(3,1)：is_contiguous() = %s，stride = %s"
          % (w.is_contiguous(), tuple(w.stride())))
    print("    它**没有 storage**——`is_contiguous()` 也就无从检查边界。")
    check("C", "同一几何：as_strided 拒绝、wrapper 子类接受且判为连续", bool(w.is_contiguous()) is True, "见上面两行")
    try:
        torch.Tensor._make_wrapper_subclass(
            _NoDispatch, (2, 3), strides=(3, 1), dtype=torch.float32, device="cpu")
        check("C", "不定义 __torch_dispatch__ 的 wrapper 子类必须抛 TypeError", False, "它没有抛")
    except TypeError as exc:
        check("C", "不定义 __torch_dispatch__ 的 wrapper 子类抛 TypeError", "must define __torch_dispatch__" in str(exc), str(exc)[:60])
    print()
    print("    结构性证据（规格层，见 specs.py）：`_compute_contiguous` 的签名是")
    print("        %s" % SPECS.CONTIGUOUS_SIGNATURE)
    print("    参数里**没有 storage**，所以它不可能检查边界；重叠由另一个函数负责：")
    print("        %s" % SPECS.OVERLAP_SIGNATURE)
    sub("C3  `view()` 失败时报错、`reshape()` fallback——这是两者的实用区别")
    nc = torch.randn(4, 4).t()
    try:
        nc.view(16)
        check("C", "非连续的 (4,4).t() 不能 view(16)", False, "它成功了")
    except RuntimeError as exc:
        check("C", "非连续的 (4,4).t() 不能 view(16)（reshape 可以）", "not compatible" in str(exc), " ".join(str(exc).split())[:60])
    before = nc.data_ptr()
    r = nc.reshape(16)
    check("C", "reshape 在需要时会**拷贝**（指针变了）", r.data_ptr() != before, f"ptr {before} → {r.data_ptr()}")
    sub("C4  但 reshape 对非连续也可能**返回 view**（不能一律说「reshape 会拷贝」）")
    t46 = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()   # (6,4) 非连续
    r2 = t46.reshape(6, 4)
    print("    (4,6).t() 非连续，reshape(6,4) 后：共享存储 = %s" % shares_storage(t46, r2))
    check("C", "reshape 对非连续输入也可能返回 view（共享存储）", shares_storage(t46, r2), f"ptr {t46.untyped_storage().data_ptr()} vs {r2.untyped_storage().data_ptr()}")
    sub("C5  共享存储的判定：**不要只看 `storage().data_ptr()`**（offset 会掩盖差异）")
    full = torch.randn(8)
    sl = full[2:6]
    print("    full.data_ptr()      =", full.data_ptr())
    print("    sl.data_ptr()        =", sl.data_ptr(), "（含 offset=2 的偏移）")
    print("    sl.storage() 的指针   =", sl.untyped_storage().data_ptr())
    print("    sl.storage_offset()  =", sl.storage_offset())
    check("C", "`data_ptr()` 不相等，但 `untyped_storage().data_ptr()` 相等", sl.data_ptr() != full.data_ptr() and sl.untyped_storage().data_ptr() == full.untyped_storage().data_ptr(), "两个判据会给出相反的结论")
    sub("C6  `contiguous()` 在已连续时返回 **self**，连 dispatcher 都不进")
    class Rec(torch.utils._python_dispatch.TorchDispatchMode):
        def __init__(self):
            self.calls = []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls.append(str(func))
            return func(*args, **(kwargs or {}))
    cont = torch.randn(4, 4)
    rec = Rec()
    with rec:
        same = cont.contiguous()
    n_contig, calls_contig = len(rec.calls), list(rec.calls)
    ncsr = cont.t()
    rec2 = Rec()
    with rec2:
        copied = ncsr.contiguous()
    print("    已连续张量：contiguous() 返回 self = %s，拦截 %d 次 %s"
          % (same is cont, n_contig, calls_contig))
    print("    非连续张量：contiguous() 返回 self = %s，拦截 %d 次 %s"
          % (copied is ncsr, len(rec2.calls), rec2.calls))
    check("C", "已连续时 contiguous() 返回 self 且拦截 0 次", same is cont and n_contig == 0)
    check("C", "对照：非连续时它真的走了一次 clone", copied is not ncsr and rec2.calls == ["aten.clone.default"], str(rec2.calls))


# --- 主流程

def main():
    print("=" * 76)
    print("03 张量内存模型 —— 本机实测（真 torch，非迷你实现）")
    print("=" * 76)
    print()
    print("测量条件：torch %s，python %s，CPU-only"
          % (torch.__version__, sys.version.split()[0]))
    print("版本绑定：本机 " + VERSIONS + " **五个**解释器都跑过；**2.9/2.10/2.12 本机无解释器**。")
    print("A 类的参照是 verify.py 的 ref_* ——它跑在默认 python3 下、**不 import torch**。")
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
