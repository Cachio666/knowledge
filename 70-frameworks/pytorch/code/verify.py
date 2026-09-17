#!/usr/bin/env python3
"""70-frameworks/pytorch 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`**（CONTRIBUTING §2 第三种形态）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件**不做** B。

  ★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
    期望值是手写的常量（不是从 torch 抄的输出）。§D 节把这条**机械地验证一遍**：
    把参考实现故意改错，A 类断言必须失败。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import sys

# ===========================================================================
# 口径
# ===========================================================================

VERSIONS_BOUND = ("2.6.0", "2.7.1", "2.13.0")
UNCOVERED = "2.8 – 2.12"

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append(f"[{cls}] {name}")
    tail = f"   ← {detail}" if detail else ""
    print(f"    [{'通过' if ok else '**不通过**'}] {name}{tail}")
    return ok


def section(tag, title):
    print()
    print("=" * 76)
    print(f"§{tag}  {title}")
    print("=" * 76)


def sub(title):
    print()
    print(f"--- {title} ---")


# ===========================================================================
# §1 独立参考实现（全部手写，不含 torch 的任何代码或输出）
# ===========================================================================

def ref_contiguous_strides(sizes):
    """C 连续布局的标准 stride：最后一维 1，往前逐维乘后一维的 size。"""
    strides = [0] * len(sizes)
    acc = 1
    for d in range(len(sizes) - 1, -1, -1):
        strides[d] = acc
        acc *= sizes[d]
    return strides


def ref_compute_contiguous(sizes, strides, numel):
    """`c10/core/Contiguity.h` 的 `_compute_contiguous` 的独立重写。

    逐字语义（来源：本机 torch/include/c10/core/Contiguity.h:15-35，2.13.0 文件）：
        if numel == 0: return True
        expected = 1
        for d from last to first:
            if size[d] == 1: continue          # ★ 跳过 size==1 的维
            if stride[d] != expected: return False
            expected *= size[d]
        return True
    **签名里没有 storage** —— 所以它不可能检查边界或重叠；
    「重叠」由同文件另一个函数 `_compute_non_overlapping_and_dense` 负责。
    """
    if numel == 0:
        return True
    expected = 1
    for d in range(len(sizes) - 1, -1, -1):
        if sizes[d] == 1:
            continue
        if strides[d] != expected:
            return False
        expected *= sizes[d]
    return True


def ref_numel(sizes):
    n = 1
    for s in sizes:
        n *= s
    return n


def ref_runs(sizes, strides):
    """把原张量按 contiguity-like condition 压成**最大连续段**列表（外层在前）。

    每段是 `(段内元素数, 段内最内维的 stride, 已连续拼起来的块大小)`；
    外层的维 d 能与当前段合并，当且仅当 `stride[d] == 内部 stride * 块大小`，
    也就是官方那条 `stride[i] = stride[i+1] * size[i+1]`。
    `size == 1` 的维**先跳过**——它不产生边界（这正是 `is_contiguous` 判据里
    「跳过 size==1 的维」的同一条约定）。
    """
    runs = []
    for d in range(len(sizes) - 1, -1, -1):
        sz, st = sizes[d], strides[d]
        if sz == 1:
            continue
        if runs and st == runs[-1][1] * runs[-1][2]:
            total, inner_stride, block = runs[-1]
            runs[-1] = (total * sz, inner_stride, block * sz)
        else:
            runs.append((sz, st, sz))
    runs.reverse()
    return runs


def _prefix_products(seq):
    """前缀积（不含总和），即这一串维在**元素序**上切出来的边界位置。"""
    out, acc = set(), 1
    for x in seq[:-1]:
        acc *= x
        out.add(acc)
    return out


def ref_view_compatible(sizes, strides, new_sizes):
    """`Tensor.view` 的判据：contiguity-like condition（见 specs.py 的逐字引文）。

    判据的一句话形式（本项目自己的表述，**不是** torch 的代码）：

      > 目标形状在**元素序**上切出的每一条边界，都必须落在原张量「连续段」的边界上——
      > 即 `原张量的段边界 ⊆ 目标形状的维边界`。

    为什么是这个：
      · 目标张量的每一个维都只有**一个** stride，所以任何一个目标维都**不能横跨**
        原张量里两个不连续的段（那样元素序里的偏移就不可能是等差数列）；
      · 反过来，**段内部**可以随便切（一段是等 stride 的等差数列，切成几维后
        stride 依次乘起来仍是同一条等差数列）——这就是官方说的「新维是原维的子空间」；
      · 于是合法当且仅当：原张量的段边界集合是目标维边界集合的**子集**。
    """
    if ref_numel(new_sizes) != ref_numel(sizes):
        return False
    if ref_numel(sizes) == 0:
        return True                      # 空张量：任何同 numel 的形状都能 view
    src_boundaries = _prefix_products([r[0] for r in ref_runs(sizes, strides)])
    dst_boundaries = _prefix_products([t for t in new_sizes if t != 1])
    return src_boundaries <= dst_boundaries


def ref_stride_slice(sizes, strides, dim, start, step):
    """`t[..., start::step, ...]` 之后的 (sizes, strides)。"""
    sizes = list(sizes)
    strides = list(strides)
    if step <= 0:
        raise ValueError("本参考实现只支持 step > 0")
    n = sizes[dim]
    new_n = max(0, (n - start + step - 1) // step)
    sizes[dim] = new_n
    strides[dim] = strides[dim] * step
    return sizes, strides


def ref_stride_transpose(sizes, strides, d0, d1):
    sizes = list(sizes)
    strides = list(strides)
    sizes[d0], sizes[d1] = sizes[d1], sizes[d0]
    strides[d0], strides[d1] = strides[d1], strides[d0]
    return sizes, strides


def ref_stride_expand(sizes, strides, target):
    """expand：被广播的维 stride 置 0。"""
    sizes = list(sizes)
    strides = list(strides)
    for d, t in enumerate(target):
        if sizes[d] == 1 and t != 1:
            sizes[d] = t
            strides[d] = 0
        elif sizes[d] != t:
            raise ValueError("expand 目标维不合法")
    return sizes, strides


def ref_is_leaf(grad_fn_is_none, requires_grad):
    """`is_leaf` 的**形式判据**（不是「用户创建的」这个口语表述）。

    来源：本机 `torch.Tensor.is_leaf.__doc__` 的语义拆解 + 官方在 `view` 上的实测行为。
    形式判据：`grad_fn is None or not requires_grad`。
    对 `w.view(-1)` 这类 view：它由操作产生（grad_fn 不为 None）且 requires_grad，
    所以 **不是 leaf**——尽管它是「用户创建的」。
    """
    return grad_fn_is_none or (not requires_grad)


# ===========================================================================
# §A 判据复核：手写期望值 vs 参考实现
# ===========================================================================

CASES_CONTIGUOUS = [
    # (名字, sizes, strides, numel, 手写期望)
    ("(4,4) 连续", (4, 4), (4, 1), 16, True),
    ("(4,4) 转置", (4, 4), (1, 4), 16, False),
    ("(4,2) 切片 ::2", (4, 2), (4, 2), 8, False),
    ("(6,4) 转置后 reshape", (6, 4), (1, 6), 24, False),
    ("(4,1,5) 中间维 size==1 且 stride 乱写", (4, 1, 5), (5, 7, 1), 20, True),
    ("(4,1,5) size==1 维 stride 极不连续", (4, 1, 5), (5, 999, 1), 20, True),
    ("numel==0 且 stride 荒谬", (0, 3), (999, 999), 0, True),
    ("(3,) 连续", (3,), (1,), 3, True),
    ("(2,3) stride (3,1)（恰是连续）", (2, 3), (3, 1), 6, True),
    ("(2,3) stride (1,1)（重叠）", (2, 3), (1, 1), 6, False),
    ("(2,3) stride (2,1)（有洞）", (2, 3), (2, 1), 6, False),
]

CASES_VIEW = [
    # (名字, sizes, strides, new_sizes, 手写期望)
    ("(4,4) 连续 → (16,)", (4, 4), (4, 1), (16,), True),
    ("(4,4) 连续 → (2,8)", (4, 4), (4, 1), (2, 8), True),
    ("(4,2) 切片 ::2 → (8,)", (4, 2), (4, 2), (8,), True),
    ("(4,2) 切片 ::2 → (2,4)", (4, 2), (4, 2), (2, 4), True),
    ("(3,2) 切片 ::2 → (6,)", (3, 2), (4, 2), (6,), True),
    ("(4,4) 转置 → (16,)", (4, 4), (1, 4), (16,), False),
    ("(4,4) 转置 → (4,4)", (4, 4), (1, 4), (4, 4), True),
    ("(4,4) 转置 → (2,8)", (4, 4), (1, 4), (2, 8), False),
    ("(6,4) 转置 → (6,4)", (6, 4), (1, 6), (6, 4), True),
    ("(6,4) 转置 → (24,)", (6, 4), (1, 6), (24,), False),
    ("(2,3,4) 全连续 → (120,) 形态（4 维连续）", (2, 3, 4, 5), (60, 20, 5, 1), (120,), True),
    ("(2,3,4,5) 连续 → (6,20)", (2, 3, 4, 5), (60, 20, 5, 1), (6, 20), True),
    ("expand (3,4)/(0,1) → (12,)", (3, 4), (0, 1), (12,), False),
    ("numel 不匹配 (4,4) → (15,)", (4, 4), (4, 1), (15,), False),
    ("空张量 (0,3) → (3,0)", (0, 3), (3, 1), (3, 0), True),
]

CASES_SLICE = [
    # (名字, sizes, strides, dim, start, step, 期望 sizes, 期望 strides)
    ("x[:, ::2]", (4, 4), (4, 1), 1, 0, 2, [4, 2], [4, 2]),
    ("x[1::3]", (10,), (1,), 0, 1, 3, [3], [3]),
    ("x[2:]", (5,), (1,), 0, 2, 1, [3], [1]),
]

CASES_LEAF = [
    # (名字, grad_fn_is_none, requires_grad, 手写期望)
    ("用户创建的叶子 w", True, True, True),
    ("常量张量（不需要梯度）", True, False, True),
    ("w.view(-1)：由操作产生", False, True, False),
    ("no_grad 下的输出（不需要梯度）", None, False, True),
]


def run_A():
    section("A", "判据复核：手写期望值 vs 独立参考实现")

    sub("A1  contiguity 判据（含 size==1 跳过与 numel==0）")
    for name, sizes, strides, numel, want in CASES_CONTIGUOUS:
        got = ref_compute_contiguous(sizes, strides, numel)
        check("A", f"contiguity：{name}", got == want, f"期望 {want}，得 {got}")

    sub("A2  view 可行性（contiguity-like condition）")
    for name, sizes, strides, new_sizes, want in CASES_VIEW:
        got = ref_view_compatible(sizes, strides, new_sizes)
        check("A", f"view：{name}", got == want, f"期望 {want}，得 {got}")

    sub("A3  stride 传播")
    for name, sizes, strides, dim, start, step, ws, wt in CASES_SLICE:
        gs, gt = ref_stride_slice(sizes, strides, dim, start, step)
        check("A", f"slice：{name} 的 size", gs == ws, f"期望 {ws}，得 {gs}")
        check("A", f"slice：{name} 的 stride", gt == wt, f"期望 {wt}，得 {gt}")
    gs, gt = ref_stride_transpose((4, 4), (4, 1), 0, 1)
    check("A", "transpose：(4,4)/(4,1) → (4,4)/(1,4)", (gs, gt) == ([4, 4], [1, 4]),
          f"得 {gs}/{gt}")
    gs, gt = ref_stride_expand((1, 4), (4, 1), (3, 4))
    check("A", "expand：(1,4)/(4,1) → (3,4)/(0,1)", (gs, gt) == ([3, 4], [0, 1]),
          f"得 {gs}/{gt}")
    check("A", "expand 后 is_contiguous 为假（stride 0 不是连续）",
          ref_compute_contiguous([3, 4], [0, 1], 12) is False)

    sub("A4  is_leaf 的形式判据（不是「用户创建的」）")
    for name, gfn_none, req, want in CASES_LEAF:
        got = ref_is_leaf(gfn_none, req)
        check("A", f"is_leaf：{name}", got == want, f"期望 {want}，得 {got}")

    sub("A5  两条路径必须一致：连续布局的 stride 一定判为连续")
    for sizes in [(4,), (3, 5), (2, 3, 4), (1, 1, 7)]:
        st = ref_contiguous_strides(sizes)
        check("A", f"自洽：{sizes} 的标准 stride 判为连续",
              ref_compute_contiguous(sizes, st, ref_numel(sizes)) is True,
              f"stride={st}")


# ===========================================================================
# §C 边界与反例：这些构造**必须被拒**
# ===========================================================================

def run_C():
    section("C", "边界与反例：以下构造必须被拒绝（断言「失败行为成立」）")

    sub("C1  numel 不匹配的 view 必须被拒")
    check("C", "view (4,4) → (15,) 被拒",
          ref_view_compatible((4, 4), (4, 1), (15,)) is False)
    check("C", "view (4,4) → (3,5) 被拒",
          ref_view_compatible((4, 4), (4, 1), (3, 5)) is False)

    sub("C2  真非连续的几何必须被拒（对照：同一几何的另一侧必须放行）")
    check("C", "(4,4)/(1,4) → (16,) 被拒",
          ref_view_compatible((4, 4), (1, 4), (16,)) is False)
    check("C", "对照：(4,2)/(4,2)（::2 切片）→ (8,) 放行",
          ref_view_compatible((4, 2), (4, 2), (8,)) is True)

    sub("C3  非法 stride 参数必须抛异常（不是静默返回）")
    for bad in [lambda: ref_stride_slice((4,), (1,), 0, 0, 0),
                lambda: ref_stride_slice((4,), (1,), 0, 0, -1),
                lambda: ref_stride_expand((2, 4), (4, 1), (3, 4))]:
        try:
            bad()
            check("C", "非法参数应抛 ValueError", False, "它没有抛")
        except ValueError:
            check("C", "非法参数应抛 ValueError", True)

    sub("C4  重叠与非连续是两件事：重叠的布局也可能被判「非连续」")
    check("C", "(2,3)/(1,1) 重叠 → 判为非连续",
          ref_compute_contiguous((2, 3), (1, 1), 6) is False)
    check("C", "(2,3)/(3,1) 无重叠 → 判为连续（说明判据管的是「布局」不是「重叠」）",
          ref_compute_contiguous((2, 3), (3, 1), 6) is True)

    sub("C5  size==1 的维是「免检」的——这是判据的有意行为，不是漏检")
    check("C", "(4,1,5)/(5,7,1) 判为连续（中间维 stride=7 被跳过）",
          ref_compute_contiguous((4, 1, 5), (5, 7, 1), 20) is True)
    check("C", "对照：(4,2,5)/(5,7,1) 同样几何但中间维不是 1 → 判为非连续",
          ref_compute_contiguous((4, 2, 5), (5, 7, 1), 40) is False)


# ===========================================================================
# §D 可证伪性：把参考实现故意改错，A 类断言必须失败
# ===========================================================================

def broken_not_skipping_size_one(sizes, strides, numel):
    """故意写错：**不跳过 size==1 的维**（这正是官方判据与朴素直觉的差别所在）。"""
    if numel == 0:
        return True
    expected = 1
    for d in range(len(sizes) - 1, -1, -1):
        if strides[d] != expected:
            return False
        expected *= sizes[d]
    return True


def broken_numel_zero_false(sizes, strides, numel):
    """故意写错：numel == 0 时返回 False。"""
    if numel == 0:
        return False
    expected = 1
    for d in range(len(sizes) - 1, -1, -1):
        if sizes[d] == 1:
            continue
        if strides[d] != expected:
            return False
        expected *= sizes[d]
    return True


def broken_contiguous_ignoring_strides(sizes, strides, numel):
    """故意写错：只看 size，不看 stride（很多人的第一直觉）。"""
    return numel >= 0


def run_D():
    section("D", "可证伪性：把参考实现故意改错，A 类判据必须失败")

    bad_cases = [c for c in CASES_CONTIGUOUS
                 if broken_not_skipping_size_one(c[1], c[2], c[3]) != c[4]]
    check("A", "改错 1（不跳过 size==1）会与手写期望冲突", len(bad_cases) > 0,
          f"找到 {len(bad_cases)} 个反例，例如 {bad_cases[0][0] if bad_cases else '-'}")

    bad_cases2 = [c for c in CASES_CONTIGUOUS
                  if broken_numel_zero_false(c[1], c[2], c[3]) != c[4]]
    check("A", "改错 2（numel==0 返回 False）会与手写期望冲突", len(bad_cases2) > 0,
          f"找到 {len(bad_cases2)} 个反例")

    bad_cases3 = [c for c in CASES_CONTIGUOUS
                  if broken_contiguous_ignoring_strides(c[1], c[2], c[3]) != c[4]]
    check("A", "改错 3（完全不看 stride）会与手写期望冲突", len(bad_cases3) > 0,
          f"找到 {len(bad_cases3)} 个反例，例如 {bad_cases3[0][0] if bad_cases3 else '-'}")

    ok_cases = [c for c in CASES_CONTIGUOUS
                if ref_compute_contiguous(c[1], c[2], c[3]) == c[4]]
    check("A", "对照：正确实现与手写期望**全部**一致",
          len(ok_cases) == len(CASES_CONTIGUOUS),
          f"{len(ok_cases)}/{len(CASES_CONTIGUOUS)}")


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    print("=" * 76)
    print("70-frameworks/pytorch —— 纯 python3 复核（**不 import torch**）")
    print("=" * 76)
    print()
    print("口径：本文件全部输出是**推导**，不是本机实测值。")
    print("      它跑的是本文件自己的参考实现与手写期望值，不碰任何框架、不碰硬件。")
    print("      编号程序（01–04）在装了 torch 的环境里跑真系统，那才是本机实测值。")
    print()
    print(f"版本绑定：本主题的实测基线锚在 torch {VERSIONS_BOUND[0]} / "
          f"{VERSIONS_BOUND[1]} / {VERSIONS_BOUND[2]}（本机三个解释器）；")
    print(f"          **{UNCOVERED} 本机测不到，不声称覆盖**。")
    print(f"          python3 = {sys.version.split()[0]}（本文件的唯一测量条件）")

    run_A()
    run_C()
    run_D()

    section("汇总", "断言汇总")
    print(f"  共 {_STATE['n']} 条：A 类 {_STATE['A']} 条，C 类 {_STATE['C']} 条")
    if _STATE["fail"]:
        print(f"  **不通过 {len(_STATE['fail'])} 条**：")
        for f in _STATE["fail"]:
            print(f"    - {f}")
        return 1
    print("  全部通过。")
    print()
    print("  提醒：本文件通过与「编号程序跑出的实测数字」是**两档不同的证据**，")
    print("        不得互相背书（CONTRIBUTING §2）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
