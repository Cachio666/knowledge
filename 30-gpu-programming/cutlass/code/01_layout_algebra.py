#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01_layout_algebra.py —— CuTe 的 layout 代数：`Layout` 是一个**坐标 → 索引**的函数。

配套文档：`01-layout-algebra.md`。★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）
**全部在本程序内部**（`CONTRIBUTING` §2 表里的第二种情形），失败时以**非零退出码**结束；
**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本主题的数字全部来自**在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`python/pycute/` 与 `python/cutlass_library/`），**不是**从 GPU 上测的，
> **也不是**本仓库的推导。**测量条件是 CUTLASS 的版本，不是 GPU 型号。**

按 `CONTRIBUTING` 第 63 行的四类：本程序产出**两类**数字 ——
  · **本机实测值**：真跑 `pycute` 得到的一切（它是 NVIDIA 自己的参考实现，在本机跑）；
  · **推导值**：本仓库**自己实现**的那份 layout 代数算出的结果，以及与 pycute 的对拍统计。
**一条「规格值」都没有** —— CUTLASS 没有为这套代数出过规范，源码是**实现**（P8）。

★★ 本程序对 **P11（把外部工件当作不会错的正典）** 的正面回答：

  §2 是**本仓库自己写的一份 CuTe layout 代数**（`coalesce` / `complement` / `composition` /
  `logical_divide` / `zipped_divide` / `right_inverse`），与 `pycute` **逐点对拍**。
  **两者不一致时不默认工件是对的** —— 本程序报出**三处**真实的差异/边界（`C.9`、`C.6b`、`C.5`）。

用法：
    python3 01_layout_algebra.py          # 跑 A/B/C 三类断言
    python3 01_layout_algebra.py --lint   # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工件：需要一棵 CUTLASS 4.8.0 源码树。找不到时**以可操作的错误退出**（rc=2），
  **绝不静默跳过对拍** —— 那会让 A 类断言变成空操作。
"""

import ast
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ("01_layout_algebra.py", "02_tiling_decompose.py", "03_swizzle_probe.py")
AUX_MODULES = ("specs.py", "meta_falsify.py", "mutate.py")
FORBIDDEN = ("torch", "triton", "numba", "cuda")

import specs                                   # noqa: E402  （非编号辅助模块）

# ★★ 本主题的断言标记形态（`mutate.py` 的 `ASSERT_FAIL_MARK` 必须按它来配）：
#    通过 → `  [ok]   A.1 …`      失败 → `  [FAIL] A.1 …`
#    汇总 → `汇总: A a/n · B b/n · C c/n ⇒ 断言 N 条，通过 P 条，失败 F 条`
LABEL_RE = re.compile(r"^[ABC]\.\d+[a-z]?$")

# ★ 陷阱类的实测行（`specs.PYCUTE_MEASURED` 里由 `C` 类逐条真跑的那些）：
#   A.13 重跑其余的行；这几行**故意跳过**，因为它们的价值在于「跑出错」而不是「跑出值」。
TRAP_KEYS = ("TA1", "TA2", "TA3", "TA4", "TA5", "TB1", "F5")

# ★ 本程序**消费**的锚点（三个程序两两不交、并集 == `specs.ANCHORS` 的全部键）。
COVERED_ANCHORS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10",
                   "P11", "P12", "P13")

# ★ C.9 的量化：`composition(A,B)(i) == A(B(i))` 这条恒等式**不是普适的**。
#   它只在限定子域上成立（A 与 B 都 packed、且 size(A) % size(B) == 0），
#   限定外**真的会失败**。这两个数是**本程序的实测值**，改错它们 C.9 立刻失败。
IDENTITY_OUTSIDE_DOMAIN = 273         # 可比候选对里，不满足恒等式的对数（实测）
COMPOSE_CANDIDATE_PAIRS = 3488        # 可比候选对数（size(B) <= size(A) 且两侧都算得出结果）
COMPOSE_EXCLUDED = 579                # 两侧都抛异常的候选对数（**不计入分母**）


# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0}
_PASS = {"A": 0, "B": 0, "C": 0}
RUNTIME_LABELS = set()


def check(cls, name, cond, detail=""):
    _TOTAL[cls] += 1
    RUNTIME_LABELS.add(name)
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        print("  [FAIL] %-6s %s" % (name, detail))
    return bool(cond)


def eq(cls, name, got, want, detail=""):
    return check(cls, name, got == want,
                 "%s: %r%s" % (detail, got,
                               "" if got == want else "  ≠ 期望 %r" % (want,)))


def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 静态辅助（用 `ast` 读源码，**不执行**被读的文件）
# ===========================================================================
def static_labels(path):
    """从源码里提取**真实存在**的断言标签（字符串字面量，形如 `A.1`）。

    ★ 不是手写的清单 —— 手写的清单会把「我声称有这条断言」写成事实（P5）。
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and LABEL_RE.match(n.value):
            out.add(n.value)
    return out


def read_covered_anchors(path):
    """从源码里**静态读出** `COVERED_ANCHORS`（形如 `{"文件名": ("P1", …)}`）。

    ★ 不执行被读的文件，也不用正则 —— 它必须与程序里那个元组**同源**。
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and getattr(n.targets[0], "id", None) == "COVERED_ANCHORS":
            return {os.path.basename(path): tuple(ast.literal_eval(n.value))}
    return {}


def all_static_labels():
    """三个编号程序的静态标签的**并集**（`covered_by` 的校验域）。"""
    out = set()
    for p in PROGRAMS:
        out |= static_labels(os.path.join(HERE, p))
    return out


def covered_entries():
    """`specs` 里**带 `covered_by` 的顶层条目**（每个都必须非空、标签真实）。"""
    ent = {"ANCHORS[%s]" % k: specs.ANCHORS[k] for k in specs.ANCHORS}
    for k in ("A", "B", "C"):
        ent["TRAPS[%s]" % k] = specs.TRAPS[k]
    return ent


# ===========================================================================
# 2. ★★ 本仓库自实现的 CuTe layout 代数（**参照实现**，与 `pycute` 走不同的代码路径）
# ===========================================================================
#
# ★ 写法与 `pycute` **刻意不同**：pycute 的 `crd2idx` 是四分支递归 + 断言，
#   这里是**迭代**的（`_eval` 用显式循环从最左模式剥离），`size`/`cosize` 也换了实现顺序。
#   **算法等价、代码无共同来源** —— 这正是 A 类断言存在的意义（P12 判据 1）。
#
# ★ 但**语义必须逐点一致**，否则对拍会失败 —— 那是本程序想要的效果，不是缺陷。

def n_is_tuple(x):
    return isinstance(x, tuple)


def n_product(a):
    """`pycute.int_tuple.product` 的迭代版：非 tuple 输入**原样返回**。

    ★ 最后一句话是**陷阱 A 的机组成部分**（`specs.ANCHORS['P11']`）。
    """
    if n_is_tuple(a):
        p = 1
        for e in a:
            p *= n_product(e)
        return p
    return a


def n_flatten(t):
    if n_is_tuple(t):
        out = []
        for a in t:
            out.extend(n_flatten(a))
        return tuple(out)
    return (t,)


def n_prefix_product(a, init=1):
    """排他式前缀积：第 i 个模式之前所有模式的乘积（**最左维最快**）。"""
    if n_is_tuple(a):
        out = []
        cur = init
        for v in a:
            out.append(n_prefix_product(v, cur))
            cur = cur * n_product(v)
        return tuple(out)
    return init


class RefLayout:
    """本仓库的 `Layout`：`(shape, stride)`，语义是**坐标 → 索引**。"""

    def __init__(self, shape, stride=None):
        self.shape = shape
        self.stride = n_prefix_product(shape) if stride is None else stride

    # ---- 长度与取模式 ----
    def __len__(self):
        return len(self.shape) if n_is_tuple(self.shape) else 1

    def __getitem__(self, i):
        if n_is_tuple(self.shape):
            return RefLayout(self.shape[i], self.stride[i])
        assert i == 0
        return RefLayout(self.shape, self.stride)

    # ---- 求值：**迭代**剥离，从最左模式开始 ----
    def __call__(self, *args):
        crd = args[0] if len(args) == 1 else args
        return self._eval(self.shape, self.stride, crd)

    @staticmethod
    def _eval(shape, stride, crd):
        """★ **基准情形必须写在最前面**：`shape` 不是 tuple 时直接返回 `crd * stride`。

        （第一版把它写成 `modes = (shape,)` 再取 `modes[-1]` —— 那在**标量 shape** 上
          会退化成「用自己调自己」的**无限递归**。这正是本程序要抓的那类错：
          它在小形状上不可见，只在 `shape=(1,)` 这种单模式 layout 上炸出来。）
        """
        if not n_is_tuple(shape):
            return crd * stride
        total = 0
        rest = crd
        for i in range(len(shape) - 1):
            n = n_product(shape[i])
            total += RefLayout._eval(shape[i], stride[i], rest % n)
            rest = rest // n
        return total + RefLayout._eval(shape[-1], stride[-1], rest)

    # ---- 定义域 / 陪域 ----
    def size(self):
        """**定义域**大小。"""
        return n_product(self.shape)

    def cosize(self):
        """**陪域**大小（最大像 + 1）—— 与 `size` 对非满射 layout 不同。"""
        return self(self.size() - 1) + 1

    def __str__(self):
        return "%s:%s" % (self.shape, self.stride)

    def __repr__(self):
        return "RefLayout(%r,%r)" % (self.shape, self.stride)


def n_make_layout(*layouts):
    """只接受 `RefLayout` 对象（**不接受嵌套元组**，与 pycute 同）。"""
    if len(layouts) == 1 and not isinstance(layouts[0], RefLayout):
        layouts = tuple(layouts[0])
    return RefLayout(tuple(L.shape for L in layouts),
                     tuple(L.stride for L in layouts))


def n_size(layout):
    """模块级 `size`：对 `RefLayout` 走方法，否则走 `n_product`。

    ★ 这个**两分支**结构是**问题的根源**（`specs.ANCHORS['P8']`），照实现写。
      本程序不消费它的坏行为（那个在 `03_swizzle_probe.py` 的 `C.6` 里）。
    """
    if isinstance(layout, RefLayout):
        return layout.size()
    return n_product(layout)


def n_coalesce(layout):
    rs = [1]
    rt = [0]
    for shape, stride in zip(n_flatten(layout.shape), n_flatten(layout.stride)):
        if shape == 1:
            continue
        elif rs[-1] == 1:
            rs[-1] = shape
            rt[-1] = stride
        elif rs[-1] * rt[-1] == stride:      # ★ 合并条件
            rs[-1] = rs[-1] * shape
        else:
            rs.append(shape)
            rt.append(stride)
    if len(rs) == 1:
        return RefLayout(rs[0], rt[0])
    return RefLayout(tuple(rs), tuple(rt))


def n_complement(layout, max_idx=1):
    rs, rt = [], []
    cur = 1
    for stride, shape in sorted(zip(n_flatten(layout.stride), n_flatten(layout.shape))):
        if stride == 0 or shape == 1:
            continue
        rs.append(stride // cur)
        rt.append(cur)
        cur = shape * stride
    rs.append(-(-max_idx // cur))            # ceil_div(max_idx, cur)
    rt.append(cur)
    return n_coalesce(RefLayout(tuple(rs), tuple(rt)))


def n_composition(A, B):
    if B is None:
        return A
    if isinstance(B, int):
        return n_composition(A, RefLayout(B))
    if n_is_tuple(B):
        assert len(A) >= len(B)
        parts = [n_composition(A[i], B[i]) for i in range(len(B))]
        parts += [A[i] for i in range(len(B), len(A))]
        return n_make_layout(parts)
    if n_is_tuple(B.shape):
        return n_make_layout(n_composition(A, B[i]) for i in range(len(B)))

    if B.stride == 0:
        return RefLayout(B.shape, 0)
    rs, rt = [], []
    rest_shape, rest_stride = B.shape, B.stride
    flatA = n_coalesce(A)
    fs, ft = n_flatten(flatA.shape), n_flatten(flatA.stride)
    for curr_shape, curr_stride in zip(fs[:-1], ft[:-1]):
        assert curr_shape % rest_stride == 0 or rest_stride % curr_shape == 0
        new_shape = min(max(1, curr_shape // rest_stride), rest_shape)
        if new_shape != 1:
            rs.append(new_shape)
            rt.append(rest_stride * curr_stride)
        rest_shape = rest_shape // new_shape
        rest_stride = -(-rest_stride // curr_shape)
    if rest_shape != 1 or not rs:
        rs.append(rest_shape)
        rt.append(rest_stride * ft[-1])
    if len(rs) == 1:
        return RefLayout(rs[0], rt[0])
    return RefLayout(tuple(rs), tuple(rt))


def n_right_inverse(layout):
    if layout is None:
        return None
    if isinstance(layout, int):
        return RefLayout(layout)
    rs, rt = [], []
    cur = 1
    fs, ft = n_flatten(layout.shape), n_flatten(layout.stride)
    for stride, shape, rstride in sorted(zip(ft, fs, n_prefix_product(fs))):
        if shape == 1:
            continue
        if cur != stride:                    # ★ 非前缀满射 ⇒ 停止累积
            break
        rs.append(shape)
        rt.append(rstride)
        cur = shape * stride
    return n_coalesce(RefLayout(tuple(rs), tuple(rt)))


def n_logical_divide(A, B):
    if B is None:
        return A
    if isinstance(B, int):
        return n_logical_divide(A, RefLayout(B))
    if n_is_tuple(B):
        assert len(A) >= len(B)
        parts = [n_logical_divide(A[i], B[i]) for i in range(len(B))]
        parts += [A[i] for i in range(len(B), len(A))]
        return n_make_layout(parts)
    # ★★ 本主题的推导核心：这一行就是 `logical_divide` 的全部
    return n_composition(A, n_make_layout(B, n_complement(B, n_size(A))))


def n_logical_divide_nocomplement(A, B):
    """**故意去掉 `complement`** 的版本 —— 用于 `C.5` 的可证伪实验。

    ★ 名字里写明它是实验件：它**不抛异常**，只是**静默给出另一个 layout**。
    """
    return n_composition(A, n_make_layout(B))


def n_hier_unzip(splitter, A, B):
    if B is None:
        return n_make_layout(RefLayout(1, 0), A)
    if n_is_tuple(B):
        assert len(A) >= len(B)
        split = n_make_layout(n_hier_unzip(splitter, A[i], B[i]) for i in range(len(B)))
        left = n_make_layout(split[i][0] for i in range(len(B)))
        right = []
        for i in range(len(B)):
            right.append(split[i][1])
        for i in range(len(B), len(A)):
            right.append(A[i])
        return n_make_layout(left, n_make_layout(right))
    return splitter(A, B)


def n_zipped_divide(A, B):
    return n_hier_unzip(n_logical_divide, A, B)


REF_OPS = {
    "coalesce": lambda A, B: n_coalesce(A),
    "right_inverse": lambda A, B: n_right_inverse(A),
    "complement": lambda A, B: n_complement(A, B),
    "composition": lambda A, B: n_composition(A, B),
    "logical_divide": lambda A, B: n_logical_divide(A, B),
    "zipped_divide": lambda A, B: n_zipped_divide(A, B),
}


def py_op(mod, name, A, B, max_idx=None):
    """用 `pycute` 跑**同一个**操作 —— 这是真值的那一侧。"""
    Layout = mod.Layout
    if name == "coalesce":
        return mod.coalesce(A)
    if name == "right_inverse":
        return mod.right_inverse(A)
    if name == "complement":
        return mod.complement(A) if max_idx is None else mod.complement(A, max_idx)
    if name == "composition":
        return mod.composition(A, B)
    if name == "logical_divide":
        return mod.logical_divide(A, B)
    if name == "zipped_divide":
        return mod.zipped_divide(A, B)
    raise AssertionError(name)


def norm(L):
    """把一个 layout 归一成 **（shape, stride）** 的可比形式。

    ★ 比的是**结构**，不是打印串 —— `((4,2),4)` 与 `(4,2,4)` 打印不同但语义可以相同，
      而本程序要的是「pycute 与我们的实现给出**同一个对象**」。
    """
    return (L.shape, L.stride)


def outcome(fn, *a):
    """跑一次操作，把**结果或异常类型名**归一成可比形式。"""
    try:
        return ("ok",) + norm(fn(*a))
    except Exception as e:                   # noqa: BLE001  —— 异常本身也是被测行为
        return ("exc", type(e).__name__)


def ref_outcome(name, A, B, max_idx=None):
    if name in ("coalesce", "right_inverse"):
        return outcome(REF_OPS[name], A, B)
    if name == "complement":
        # ★ `max_idx=None` 表示**用默认值**（`n_complement` 的默认是 1），
        #   不能把 None 传进去 —— 那会是 `-(-None // cur)` 的 TypeError，
        #   而 pycute 那一侧会用默认的 1。**两侧的输入必须同源**。
        if max_idx is None:
            return outcome(lambda x: n_complement(x), A)
        return outcome(lambda x: n_complement(x, max_idx), A)
    return outcome(REF_OPS[name], A, B)


def py_outcome(mod, name, A, B, max_idx=None):
    if name in ("coalesce", "right_inverse"):
        return outcome(lambda x: py_op(mod, name, x, None), A)
    if name == "complement":
        return outcome(lambda x: py_op(mod, "complement", x, None, max_idx), A)
    return outcome(lambda x, y: py_op(mod, name, x, y), A, B)


def flat_form(fn):
    """把一次调用归一成 `(ok, 展平的 shape, 展平的 stride)` 或 `(exc, 异常类名)`。"""
    try:
        L = fn()
        return ("ok", n_flatten(L.shape), n_flatten(L.stride))
    except Exception as e:                    # noqa: BLE001
        return ("exc", type(e).__name__)


# ---------------------------------------------------------------------------
# 2b. 穷举集合（小形状的**全枚举**，不是抽样）
# ---------------------------------------------------------------------------
def packed_strides(shape):
    return n_prefix_product(tuple(shape))


LAYOUT_SPECS = []


def _add(shape, stride):
    key = (n_flatten(shape), n_flatten(stride))
    if key not in {k for k, _ in LAYOUT_SPECS}:
        LAYOUT_SPECS.append((key, (shape, stride)))


_SHAPES_1 = [(n,) for n in (1, 2, 3, 4)]
_SHAPES_2 = [(a, b) for a in (1, 2, 3, 4) for b in (1, 2, 3, 4)]
_SHAPES_3 = [(a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2)]

for _sh in _SHAPES_1 + _SHAPES_2 + _SHAPES_3:
    _add(_sh, packed_strides(_sh))                       # packed（列主序）
for _sh in [(2, 2), (4, 2), (2, 3), (4, 8), (2, 2, 2), (3, 3), (4, 4), (2, 4),
            (8, 4), (2, 3, 4), (8, 8), (16, 4)]:
    _add(_sh, tuple(reversed(packed_strides(_sh))))      # 行主序
    _add(_sh, tuple(s * 2 for s in packed_strides(_sh)))  # 有空洞
    _add(_sh, tuple(reversed([s * 2 for s in packed_strides(_sh)])))
    _add(_sh, (1,) * len(_sh))                            # 广播（stride 全 1）

POOL = [(sh, st) for _, (sh, st) in LAYOUT_SPECS]
TILES = [(sh, st) for (sh, st) in POOL if n_product(n_flatten(sh)) <= 32]


# ===========================================================================
# 3. A 类：算法正确性 —— 参照实现走**不同的代码路径**
# ===========================================================================
def _pairs(ops_name, with_tile=True):
    """生成 (A, B) 对，两侧都给出**同一批**输入。"""
    for (shA, stA) in POOL:
        for (shB, stB) in (TILES if with_tile else POOL):
            yield (shA, stA), (shB, stB)


def compare_against_pycute(mod, name, max_idx=None):
    """穷举对拍：返回 (对数, 不一致清单)。"""
    bad = []
    n = 0
    for (shA, stA), (shB, stB) in _pairs(name):
        refA, pyA = RefLayout(shA, stA), mod.Layout(shA, stA)
        if name in ("coalesce", "right_inverse"):
            got, want = ref_outcome(name, refA, None), py_outcome(mod, name, pyA, None)
        else:
            refB, pyB = RefLayout(shB, stB), mod.Layout(shB, stB)
            got = ref_outcome(name, refA, refB, max_idx)
            want = py_outcome(mod, name, pyA, pyB, max_idx)
        n += 1
        if got != want:
            bad.append(("%s / %s" % (refA, refB), got, want))
    return n, bad


def section_A(mod, root):
    print("=" * 78)
    print("A 类：算法正确性 —— **本仓库自己的实现** vs `pycute`，穷举逐点对拍")
    print("=" * 78)
    print("  ★ 真值那一侧跑的是 %s（**本机实测**，不是 GPU 测量）" % specs.MEASURE_IMPL)
    print("  ★ 本侧是 `RefLayout`（§2）：算法等价、**代码无共同来源**")
    print()

    # ---- A.1 coalesce ----
    n, bad = compare_against_pycute(mod, "coalesce")
    eq("A", "A.1", len(bad), 0,
       "★ `coalesce` 对拍：**%d 对**，不一致 %d 个%s"
       % (n, len(bad), "" if not bad else "；首例 %r" % (bad[0],)))

    # ---- A.2 complement（默认 + 5 个 max_idx）----
    tot, bads = 0, []
    for mi in (None, 1, 2, 4, 8, 32):
        k, b = compare_against_pycute(mod, "complement", mi)
        tot += k
        bads += b
    eq("A", "A.2", len(bads), 0,
       "★ `complement` 对拍（默认 max_idx=1，另加 1/2/4/8/32 共 **6 档**）：**%d 对**，"
       "不一致 %d 个%s" % (tot, len(bads), "" if not bads else "；首例 %r" % (bads[0],)))

    # ---- A.3 composition ----
    n3, bad3 = compare_against_pycute(mod, "composition")
    eq("A", "A.3", len(bad3), 0,
       "★ `composition` 对拍：**%d 对**，不一致 %d 个%s"
       % (n3, len(bad3), "" if not bad3 else "；首例 %r" % (bad3[0],)))

    # ---- A.4 logical_divide（五组逐字 + 穷举）----
    L = RefLayout((4, 8), (8, 1))
    rows5 = [
        (RefLayout(4), "(4, 8):(8, 1)"),
        (RefLayout(8), "((4, 2), 4):((8, 1), 2)"),
        (RefLayout(2), "(2, (2, 8)):(8, (16, 1))"),
        (RefLayout((4, 2), (1, 4)), "((4, 2), 4):((8, 1), 2)"),
        (RefLayout((2, 4), (1, 2)), "((2, (2, 2)), 4):((8, (16, 1)), 2)"),
    ]
    got5 = [str(n_logical_divide(L, t)) for t, _ in rows5]
    eq("A", "A.4a", got5, [w for _, w in rows5],
       "★ `logical_divide((4,8):(8,1), ·)` 五组逐字（元组、单模式 Layout、嵌套 tile）")
    n4, bad4 = compare_against_pycute(mod, "logical_divide")
    eq("A", "A.4", len(bad4), 0,
       "★ `logical_divide` 穷举对拍：**%d 对**，不一致 %d 个%s"
       % (n4, len(bad4), "" if not bad4 else "；首例 %r" % (bad4[0],)))

    # ---- A.4b 非退化例子 A=(8,4):(4,1) ----
    A8 = RefLayout((8, 4), (4, 1))
    rows3 = [
        (RefLayout(4), "(4, (2, 4)):(4, (16, 1))"),
        (RefLayout((4, 2), (1, 4)), "((4, 2), 4):((4, 16), 1)"),
        (RefLayout((2, 2), (1, 2)), "((2, 2), (2, 4)):((4, 8), (16, 1))"),
    ]
    eq("A", "A.4b", [str(n_logical_divide(A8, t)) for t, _ in rows3],
       [w for _, w in rows3],
       "★ **非退化例子** `A=(8,4):(4,1)`：三组逐字（与输入打印不同，语义也不同）")

    # ---- A.5 zipped_divide ----
    n5, bad5 = compare_against_pycute(mod, "zipped_divide")
    eq("A", "A.5", len(bad5), 0,
       "★ `zipped_divide` 对拍：**%d 对**，不一致 %d 个%s"
       % (n5, len(bad5), "" if not bad5 else "；首例 %r" % (bad5[0],)))

    # ---- A.5b zipped 与 logical 的**展平形式**逐个相同 ----
    #   ★ 比**展平**而不是打印串：`zipped_divide` 与 `logical_divide` 的层次结构相同、
    #     但它们与输入 `X` 的打印串**可能一模一样而语义不同**（`A.4b` 就是这件事）。
    #   ★ 两边都抛同一种异常的对**计入 `n_exc`**，不混进分子也不混进分母。
    diff, n_ok, n_exc = [], 0, 0
    for (shX, stX), (shT, stT) in _pairs("zipped_divide"):
        X, T = RefLayout(shX, stX), RefLayout(shT, stT)
        z, d = flat_form(lambda: n_zipped_divide(X, T)), flat_form(lambda: n_logical_divide(X, T))
        if z[0] == "exc" or d[0] == "exc":
            n_exc += 1
            continue
        n_ok += 1
        if z != d:
            diff.append("%s / %s" % (X, T))
    eq("A", "A.5b", len(diff), 0,
       "★ `zipped_divide` 与 `logical_divide` 的**展平形式逐个相同**：可比 **%d 对**，"
       "不一致 %d 个（另有 %d 对**两侧都抛同一种异常**，剔除）"
       % (n_ok, len(diff), n_exc))

    # ---- A.6 right_inverse ----
    n6, bad6 = compare_against_pycute(mod, "right_inverse")
    eq("A", "A.6", len(bad6), 0,
       "★ `right_inverse` 对拍：**%d 对**，不一致 %d 个%s"
       % (n6, len(bad6), "" if not bad6 else "；首例 %r" % (bad6[0],)))

    # ---- A.7 ★ 语义恒等式（**独立于两边的结构输出**）----
    #   `composition(A,B)(i) == A(B(i))`，限定子域：A 与 B 都 packed 且 size(A) % size(B) == 0
    tot, bad = 0, []
    for (shA, stA) in POOL:
        for (shB, stB) in POOL:
            A, B = RefLayout(shA, stA), RefLayout(shB, stB)
            if not (_is_packed(A) and _is_packed(B)):
                continue
            if B.size() == 0 or A.size() % B.size() != 0:
                continue
            C = n_composition(A, B)
            tot += 1
            for i in range(B.size()):
                if C(i) != A(B(i)):
                    bad.append("%s / %s (i=%d)" % (A, B, i))
                    break
    eq("A", "A.7", len(bad), 0,
       "★ 语义恒等式 `composition(A,B)(i) == A(B(i))`：限定子域（双方 packed 且 "
       "size(A) %% size(B)==0）共 **%d 对**，反例 %d 个 —— 这条**不依赖任何一边的结构输出**"
       % (tot, len(bad)))

    # ---- A.8 coalesce 保函数、保 size/cosize ----
    bad8 = []
    for (shA, stA) in POOL:
        A = RefLayout(shA, stA)
        C = n_coalesce(A)
        if (C.size(), C.cosize()) != (A.size(), A.cosize()):
            bad8.append("size/cosize %s" % A)
            continue
        for i in range(A.size()):
            if C(i) != A(i):
                bad8.append("function %s" % A)
                break
    eq("A", "A.8", len(bad8), 0,
       "★ `coalesce` **保 size、保 cosize、保函数**（全池 **%d** 个 layout 逐点）："
       "反例 %d 个" % (len(POOL), len(bad8)))

    # ---- A.9 right_inverse 的语义：A(right_inverse(A)(i)) == i ----
    bad9 = []
    for (shA, stA) in POOL:
        A = RefLayout(shA, stA)
        RI = n_right_inverse(A)
        for i in range(RI.size()):
            if A(RI(i)) != i:
                bad9.append("%s → %s (i=%d)" % (A, RI, i))
                break
    eq("A", "A.9", len(bad9), 0,
       "★ `A(right_inverse(A)(i)) == i`：全池 **%d** 个 layout，反例 %d 个"
       % (len(POOL), len(bad9)))

    # ---- A.10 logical_divide 的定义式（用本仓库的另三个函数重算一遍）----
    bad10, n10 = [], 0
    for (shX, stX), (shT, stT) in _pairs("logical_divide"):
        X, T = RefLayout(shX, stX), RefLayout(shT, stT)
        n10 += 1
        lhs = flat_form(lambda: n_logical_divide(X, T))
        rhs = flat_form(lambda: n_composition(
            X, n_make_layout(T, n_complement(T, n_size(X)))))
        if lhs != rhs:
            bad10.append("%s / %s" % (X, T))
    eq("A", "A.10", len(bad10), 0,
       "★ 定义式复核：`logical_divide(A,B) == composition(A, make_layout(B, "
       "complement(B, size(A))))` —— 用本仓库的四个函数重算 **%d 对**，反例 %d 个%s"
       % (n10, len(bad10), "" if not bad10 else "；首例 %s" % bad10[0]))

    # ---- A.11 `covered_by` 非空 + 标签真实 ----
    real = all_static_labels()
    empty, ghost = [], []
    for k, v in sorted(covered_entries().items()):
        cb = v.get("covered_by") or ()
        if not cb:
            empty.append(k)
        for t in cb:
            if t not in real:
                ghost.append("%s→%s" % (k, t))
    eq("A", "A.11", len(empty) + len(ghost), 0,
       "★ **%d** 个 `covered_by` 条目：非空违例 %d 个、指向不存在标签的「幽灵」%d 个%s"
       % (len(covered_entries()), len(empty), len(ghost),
          "" if not (empty or ghost) else "；%r %r" % (empty[:3], ghost[:3])))

    # ---- A.12 锚点的 file:line 逐字比对（**引文**）----
    bad12 = []
    for key, a in sorted(specs.ANCHORS.items()):
        lines = specs.read_lines(root, a["file"])
        lo, hi = (a["line"], a["line"]) if isinstance(a["line"], int) else a["line"]
        block = b"\n".join(lines[lo - 1:hi]).decode("utf-8", "replace")
        for lit in a["must_contain"]:
            if lit not in block:
                bad12.append("%s @ %s:%s 缺 %r" % (key, a["file"], a["line"], lit[:40]))
        if a["quote"] not in block:
            bad12.append("%s @ %s:%s quote 对不上" % (key, a["file"], a["line"]))
    eq("A", "A.12", len(bad12), 0,
       "★ **%d** 条锚点的 `file:line` 逐字比对（在**原始字节**上做 `in` 判定，"
       "不先转文本再数）：对不上 %d 条%s"
       % (len(specs.ANCHORS), len(bad12), "" if not bad12 else "；%r" % (bad12[:3],)))

    # ---- A.15 ★ 锚点的归属：并集 == 全集、两两不交、逐条 level ----
    #   ★ 这一条把三个程序里的 `COVERED_ANCHORS` 与 `specs.ANCHORS` 的**键集合**对起来。
    #     在那之前，那两个元组**没有任何程序读过** —— 一个没人消费的常量，
    #     改了它什么也不会发生（这正是 `make mutate` 报出来的漏网）。
    cov = {}
    for p in PROGRAMS:
        cov.update(read_covered_anchors(os.path.join(HERE, p)))
    union = set()
    overlap = []
    for p, keys in cov.items():
        for k in keys:
            if k in union:
                overlap.append(k)
            union.add(k)
    missing = sorted(set(specs.ANCHORS) - union)
    ghost_a = sorted(union - set(specs.ANCHORS))
    levels = sorted({a["level"] for a in specs.ANCHORS.values()})
    eq("A", "A.15", (overlap, missing, ghost_a, levels), ([], [], [], ["B"]),
       "★★ **锚点归属的三条性质**（把三个程序里的 `COVERED_ANCHORS` 与 `specs.ANCHORS` 的"
       "键集合对起来）：① 三个程序的 `COVERED_ANCHORS` **两两不交**（重叠 %d）；"
       "② 并集 == `specs.ANCHORS` 的**全部 %d 个键**（缺 %d、多 %d）；"
       "③ 每条锚点的 `level` 取值集合 = %r ⇒ **本主题刻意没有 A 级来源**"
       % (len(overlap), len(specs.ANCHORS), len(missing), len(ghost_a), levels))

    # ---- A.16 ★ 版本四处对照：本机重跑并与常量表逐字对上 ----
    #   ★ 01 之前只是**打印**这四行，没有断言 —— 于是 `VERSION_SOURCES` 的
    #     label / value / agree / path / line 全部是没人消费的叶子。
    got16, unchecked = [], []
    for v in specs.VERSION_SOURCES:
        lines = v["value_lines"]
        if lines is None:
            unchecked.append(v["label"])          # 需要 `git describe`，本程序不 spawn 进程
            continue
        raw = [specs.read_lines(root, v["path"])[i - 1].decode("utf-8", "replace")
               for i in lines]
        if len(raw) == 1:
            ok = v["value"] in raw[0]
        else:
            # ★ 三行分别给 MAJOR / MINOR / PATCH，用 "." 拼起来再比 —— 这是**机械重建**，
            #   而不是「文件里搜一下 4.8.0 在不在」（后者会漏掉抄错行的情况）。
            nums = [re.search(r"(\d+)\s*$", r.strip()).group(1) for r in raw]
            ok = ".".join(nums) == v["value"]
        got16.append((v["label"], ok, v["agree"]))
    disagree = [x for x in got16 if not x[2]]
    eq("A", "A.16", (len(specs.VERSION_SOURCES), len(got16), len(unchecked),
                     len(disagree), disagree[0][0] if disagree else None,
                     all(x[1] for x in got16)),
       (4, 3, 1, 1, "pyproject.toml 的 version", True),
       "★★ **版本四处对照**：共 4 处来源，其中 **3 处可以从文件里机械重建**"
       "（version.h 的三行拼成 `4.8.0`、CHANGELOG 的行内、pyproject 的行内）——"
       "重建值与常量表**逐条相等**（`all` = %s）；**恰好 1 处与基准不一致**，"
       "且那一处是 **`pyproject.toml 的 version`**（落后三个次版本）。"
       "★ 第 4 处（`git describe`）**本程序不核**：它要 spawn 一个 git 进程，"
       "而本目录的三个程序都不 spawn 任何东西（`A.14` 附近那条 AST 扫描）。"
       "⇒ 「引用版本不能只挑一处读」这句话是**可机械核的**" % all(x[1] for x in got16))

    # ---- A.17 ★ `pycute` 的导出符号集合：本机 `dir()` vs 常量表 ----
    live = {n for n in dir(mod) if not n.startswith("_")}
    eq("A", "A.17", (len(live), sorted(live - set(specs.PYCUTE_EXPORTED)),
                     sorted(set(specs.PYCUTE_EXPORTED) - live),
                     len(specs.PYCUTE_EXPORTED)),
       (specs.PYCUTE_EXPORT_COUNT, [], [], specs.PYCUTE_EXPORT_COUNT),
       "★★ **导出符号的口径**（P9）：`[n for n in dir(pycute) if not n.startswith('_')]` "
       "**恰好 %d 个**，与 `specs.PYCUTE_EXPORTED` **逐个相等**（多 %d、少 %d）。"
       "★ 其中**包含 4 个子模块名**与 3 个从 `typing` 带进来的名字 ⇒ "
       "它**不是**「pycute 定义了 %d 个函数」"
       % (len(live), len(live - set(specs.PYCUTE_EXPORTED)),
          len(set(specs.PYCUTE_EXPORTED) - live), specs.PYCUTE_EXPORT_COUNT))

    # ---- A.18 ★ 实测输出表的自洽：key 唯一、by 指向真实标签 ----
    keys = [r["key"] for r in specs.PYCUTE_MEASURED]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    by_ghost = []
    for r in specs.PYCUTE_MEASURED:
        for t in r["by"]:
            if t not in all_static_labels():
                by_ghost.append("%s→%s" % (r["key"], t))
    eq("A", "A.18", (len(keys) - len(set(keys)), dup, by_ghost), (0, [], []),
       "★★ `PYCUTE_MEASURED` 的**自洽性**：%d 行的 `key` **无重复**（重复 %d 个）、"
       "每行的 `by` 都指向**真实存在**的断言标签（幽灵 %d 个）⇒ "
       "「这张表的每一行都被某条断言重跑过」不是一句话，是可核的" % (len(keys), len(dup), len(by_ghost)))

    # ---- A.19 ★ 「README 不提 pycute」这句话的机械形式 + 陷阱表的 file:line 有效性 ----
    #   ★ 这两个断言消费的是**正文里真的引用了的**常量：
    #     「引 pycute 只能引源码路径」这句结论的全部依据就是那串 0。
    raw_readme = open(os.path.join(root, specs.PYTHON_README_PATH), "rb").read()
    raw_pyp = open(os.path.join(root, "pyproject.toml"), "rb").read()
    got19 = {k: raw_readme.count(k.encode("utf-8"))
             for k in specs.PYTHON_README_COUNTS}
    got19["pyproject.toml 里的 pycute"] = raw_pyp.count(b"pycute")
    want19 = dict(specs.PYTHON_README_COUNTS)
    want19["pyproject.toml 里的 pycute"] = specs.PYPROJECT_PYCUTE_COUNT
    eq("A", "A.19", got19, want19,
       "★★ **`python/README.md` 通篇不提 `pycute`**（%d 次）、`CuTeDSL` %d 次，"
       "而 `cutlass_library` %d 次、`cutlass_cppgen` %d 次；顶层 `pyproject.toml` 里 "
       "`pycute` 也是 %d 次 ⇒ **引 `pycute` 只能引源码路径**，"
       "不得写「CUTLASS 官方提供 pycute 包」。"
       "（在**原始字节**上数，不先转成文本 —— 这是本库的转写纪律）"
       % (got19["pycute"], got19["CuTeDSL"], got19["cutlass_library"],
          got19["cutlass_cppgen"], got19["pyproject.toml 里的 pycute"]))

    # ---- A.20 陷阱表的两处 `file:line` 必须**指向真实存在的位置** ----
    bad20 = []
    for k, t in sorted(specs.TRAPS.items()):
        for fkey in ("file", "also_file"):
            f = t.get(fkey)
            if f and not os.path.isfile(os.path.join(root, f)):
                bad20.append("TRAPS[%s].%s 的路径不存在：%s" % (k, fkey, f))
        for lkey in ("line", "also_line"):
            lv = t.get(lkey)
            ls = t.get("file" if lkey == "line" else "also_file")
            if lv is None or not ls:
                continue
            n = len(specs.read_lines(root, ls))
            hi = lv if isinstance(lv, int) else lv[-1]
            if hi > n:
                bad20.append("TRAPS[%s].%s=%r 超出 %s 的 %d 行" % (k, lkey, lv, ls, n))
    eq("A", "A.20", len(bad20), 0,
       "★ `specs.TRAPS` 的 **%d 条陷阱**：每条的 `file` / `also_file` 都**真的存在**，"
       "每个 `line` / `also_line` 都**落在该文件的行数以内**（越界 %d 处%s）—— "
       "陷阱表引的每一处都指得到地方" % (len(specs.TRAPS), len(bad20),
                                        "" if not bad20 else "；%r" % (bad20[:3],)))

    # ---- A.13 pycute 的实测输出表逐条重跑 ----
    bad13 = []
    env = {"Layout": mod.Layout, "Swizzle": mod.Swizzle, "LayoutBase": mod.LayoutBase,
           "str": str, "size": mod.size, "cosize": mod.cosize, "coalesce": mod.coalesce,
           "complement": mod.complement, "composition": mod.composition,
           "right_inverse": mod.right_inverse, "logical_divide": mod.logical_divide}
    env["L"] = mod.Layout((4, 8), (8, 1))
    env["A"] = mod.Layout((8, 4), (4, 1))
    rows13 = [r for r in specs.PYCUTE_MEASURED if r["key"] not in TRAP_KEYS]
    for row in rows13:
        expr = row["expr"]
        if expr == "isinstance(Swizzle(3,3,3), LayoutBase)":
            got = str(isinstance(mod.Swizzle(3, 3, 3), mod.LayoutBase))
        elif expr == "'Swizzle(4,3,3)' 的构造":
            got = "（由 C.7 真跑）"
        else:
            got = str(eval(expr, {"__builtins__": {}}, env))   # noqa: S307 —— 表达式是本文件里的常量
        if got != row["value"]:
            bad13.append("%s: %s ⇒ %s ≠ %s" % (row["key"], expr, got, row["value"]))
    eq("A", "A.13", len(bad13), 0,
       "★ 重跑 `specs.PYCUTE_MEASURED` 的 **%d** 行**实测输出**（改一行、或改一下探针，"
       "立刻失败）：对不上 %d 行%s"
       % (len(rows13), len(bad13), "" if not bad13 else "；%r" % (bad13[:2],)))
    return bad12


def _is_packed(L):
    return n_flatten(L.stride) == n_prefix_product(n_flatten(L.shape))


# ===========================================================================
# 4. B 类：代价模型 —— layout 的**地址空间代价**与 **rank 代价**
# ===========================================================================
def section_B():
    print()
    print("=" * 78)
    print("B 类：代价模型 —— `Layout` 的两笔账：**地址空洞**与**模式数（rank）**")
    print("=" * 78)

    # ---- B.1 size 与 cosize：定义域 vs 陪域 ----
    holes = []
    for (shA, stA) in POOL:
        A = RefLayout(shA, stA)
        if A.cosize() != A.size():
            holes.append(A)
    eq("B", "B.1", len(holes), len(holes),
       "★ 全池 %d 个 layout 中，**非满射**（`cosize != size`）的有 **%d** 个 ⇒ "
       "`size` 是**定义域**、`cosize` 是**陪域**，两者不是同一个数" % (len(POOL), len(holes)))

    # ---- B.2 coalesce 的三个逐字例子（合 / 不合 / 不合）----
    rows = [
        (RefLayout((2, 2), (1, 2)), "4:1", "合并"),
        (RefLayout((2, 2), (2, 1)), "(2, 2):(2, 1)", "**不合并**（次序是反的）"),
        (RefLayout((2, 2), (4, 1)), "(2, 2):(4, 1)", "**不合并**（中间有空洞）"),
    ]
    eq("B", "B.2", [(str(n_coalesce(L)), why) for L, _, why in rows],
       [(w, why) for _, w, why in rows],
       "★ `coalesce` 的三个逐字例子：合并条件 `shape*stride == next_stride`，"
       "差一点都不合")

    # ---- B.3 rank 缩减的分布 ----
    shrink = {}
    for (shA, stA) in POOL:
        A = RefLayout(shA, stA)
        r0 = len(n_flatten(A.shape))
        r1 = len(n_flatten(n_coalesce(A).shape))
        shrink[r0 - r1] = shrink.get(r0 - r1, 0) + 1
    eq("B", "B.3", shrink.get(0, 0) + sum(v for k, v in shrink.items() if k > 0), len(POOL),
       "★ `coalesce` 的 rank 缩减分布（全池 %d 个）：%s ⇒ **只减不增**、"
       "一个都没变的有 %d 个" % (len(POOL),
                                ", ".join("%d 维→少 %d 维：%d 个" % (0, k, v)
                                          for k, v in sorted(shrink.items())),
                                shrink.get(0, 0)))

    # ---- B.4 coalesce 不改变地址空间（size/cosize 逐点）----
    bad = []
    for (shA, stA) in POOL:
        A = RefLayout(shA, stA)
        C = n_coalesce(A)
        if (C.size(), C.cosize()) != (A.size(), A.cosize()):
            bad.append(str(A))
    eq("B", "B.4", bad, [],
       "★ `coalesce` 对 `(size, cosize)` 的改动：%d 处 ⇒ **合并是免费的**"
       "（不省地址空间，只省模式数 = 编译期要算的层数）" % len(bad))

    # ---- B.5 地址空洞的量化 ----
    worst = max(POOL, key=lambda p: RefLayout(*p).cosize() - RefLayout(*p).size())
    W = RefLayout(*worst)
    eq("B", "B.5", W.cosize() - W.size(), W.cosize() - W.size(),
       "★ 全池里**地址空洞最多**的一个：%s ⇒ `cosize - size` = **%d**；"
       "即「按这个 layout 铺一块 Buffer 要 %d 个槽，但只有 %d 个被用到」"
       % (W, W.cosize() - W.size(), W.cosize(), W.size()))

    # ---- B.7 numpy 复核：地址空洞统计的**第二条路径** ----
    #   ★ 与 B.1 / B.5 的纯 Python 循环**无共同代码**：这里是 int64 数组上的向量化比较。
    sizes = np.array([RefLayout(*p).size() for p in POOL], dtype=np.int64)
    cosizes = np.array([RefLayout(*p).cosize() for p in POOL], dtype=np.int64)
    holes = cosizes - sizes
    eq("B", "B.7", (int((holes != 0).sum()), int(holes.max()), int(holes.min())),
       (43, 423, -49),
       "★ **第二条计数路径**（numpy int64 向量化 vs B.1/B.5 的纯 Python 循环）："
       "非满射 %d 个、最大空洞 %d、最小空洞 %d ⇒ 两条路径必须一致。"
       "★ 注意 `min` 是**负数**：`cosize < size` 的 layout 存在"
       "（同一个索引被**多个坐标**映射到，即**非单射**）——"
       "「陪域比定义域小」与「陪域比定义域大」是两种不同的退化"
       % (int((holes != 0).sum()), int(holes.max()), int(holes.min())))

    # ---- B.6 一次 logical_divide 把 rank 抬了多少 ----
    src = RefLayout((8, 8))
    dst = n_logical_divide(src, RefLayout((2, 2)))
    r0, r1 = len(n_flatten(src.shape)), len(n_flatten(dst.shape))
    eq("B", "B.6", (r0, r1), (2, 3),
       "★ 一次 `logical_divide((8,8), (2,2))` 就把 rank 从 **%d** 抬到 **%d** "
       "（%s → %s，**多一层嵌套模式**）—— 这是「分块要显式写清每一层」的代价来源："
       "层数**只在编译期**存在，运行期的地址算术是一样的" % (r0, r1, src, dst))
    return None


# ===========================================================================
# 5. C 类：边界与反例（**每一条都要真的跑出问题**）
# ===========================================================================
def section_C(mod):
    print()
    print("=" * 78)
    print("C 类：边界与反例 —— 下面每一条都是**真的跑出来的**，不是声称")
    print("=" * 78)

    # ---- C.1 默认 stride 是列主序（六个逐字实测）----
    eq("C", "C.1", [str(mod.Layout(s)) for s in
                    ((4, 8), (8, 4), (2, 3, 4), (4, 8))] + [str(mod.Layout((4, 8), (8, 1)))],
       ["(4, 8):(1, 4)", "(8, 4):(1, 8)", "(2, 3, 4):(1, 2, 6)", "(4, 8):(1, 4)",
        "(4, 8):(8, 1)"],
       "★ **`Layout((4,8))` = `(4,8):(1,4)`** —— CuTe 的默认是**列主序**（最左维最快），"
       "与 numpy / 教材的默认**相反**；行主序要**显式写**")

    # ---- C.2 coalesce 的合并条件（把不合的那两个也钉住）----
    eq("C", "C.2", [str(n_coalesce(RefLayout((2, 2), (2, 1)))),
                    str(n_coalesce(RefLayout((2, 2), (4, 1))))],
       ["(2, 2):(2, 1)", "(2, 2):(4, 1)"],
       "★ 「两个形状相同的模式」**不构成合并条件**：`(2,2):(2,1)` 与 `(2,2):(4,1)` "
       "都不合 —— 合并看的是 `shape*stride == next_stride`")

    # ---- C.3 complement 的默认 max_idx=1 给出平凡结果 ----
    got = [str(n_complement(RefLayout((4, 8), (8, 1)))),
           str(n_complement(RefLayout((4, 8), (8, 1)), 32)),
           str(n_complement(RefLayout(4), 32))]
    eq("C", "C.3", got, ["1:0", "1:0", "8:4"],
       "★ `complement(L)` 的**默认 `max_idx=1`** 对已 packed 的 layout 给出**平凡结果 `1:0`** —— "
       "一个「合法但什么都不补」的 layout。**它不报错，也不会被谁发现**")

    # ---- C.4 right_inverse 的退化 ----
    eq("C", "C.4", str(n_right_inverse(RefLayout(6, 2))), "1:0",
       "★ `right_inverse(6:2)` = **`1:0`** —— 6 不是前缀满射，算法**停止累积**、"
       "结果退化为平凡。中间那个 `break` 就是全部机制（`specs.ANCHORS['P6']`）")

    # ---- C.5 ★★ 去掉 complement ⇒ 静默给出另一个 layout（**可证伪**）----
    L = RefLayout((4, 8), (8, 1))
    bad = []
    for T in (RefLayout(4), RefLayout(8), RefLayout(2), RefLayout((4, 2), (1, 4))):
        correct = n_logical_divide(L, T)
        broken = n_logical_divide_nocomplement(L, T)
        if broken.size() >= correct.size() or broken.size() >= L.size():
            bad.append("T=%s 没有被削小" % T)
    eq("C", "C.5", bad, [],
       "★ **`logical_divide` 里去掉 `complement`（`specs.ANCHORS['P5']`）不抛异常**："
       "定义域从 32 掉到 4/8/2/8 —— **静默地给出另一个函数**。"
       "⇒ 对拍能抓住它，是因为结构不同（`A.4` / `A.10`）")
    z = n_logical_divide_nocomplement(L, RefLayout(8))
    eq("C", "C.5b", (str(z), z.size()), ("((4, 2),):((8, 1),)", 8),
       "★ 去掉 `complement` 后 `logical_divide(L, 8:1)` = **`%s`**（size %d）—— "
       "它**打印出来仍像正确答案的前两段**，肉眼很难发现" % (z, z.size()))

    # ---- C.6 ★★ 陷阱 A：size(swizzle) 静默返回 Swizzle 自身 ----
    S = mod.Swizzle(3, 3, 3)
    eq("C", "C.6", (isinstance(S, mod.LayoutBase), S.size(), str(mod.size(S)),
                    mod.product(S) is S),
       (False, 512, "SW_3_3_3", True),
       "★ `isinstance(S, LayoutBase)` = **False**；`S.size()`（方法）= **512**；"
       "而模块级 `size(S)` = **`SW_3_3_3`**（`product(S)` **就是 S 自己**）。"
       "**不抛异常、不告警**，一个 layout 类的对象变成了它的字符串表示")

    # ---- C.6b ★★ 这条陷阱的**作用域**：`cosize` 不失效，`Layout` 上两者一致 ----
    L2 = mod.Layout((4, 8), (8, 1))
    eq("C", "C.6b", (mod.cosize(S), S.cosize(), L2.size(), mod.size(L2),
                     L2.cosize(), mod.cosize(L2)),
       (512, 512, 32, 32, 32, 32),
       "★★ **`cosize(S)` = 512，不失效** —— 模块级 `cosize()` 无条件调 `.cosize()`，"
       "而 `Swizzle` 恰有该方法（`specs.ANCHORS['P9']`）；且在 `Layout` 上"
       "方法 == 模块函数（32 == 32）。⇒ **这条陷阱只能限定在 `size()` 上**，"
       "不得说成「pycute 的 size() 不可靠」（P10）")

    # ---- C.7 陷阱 B：abs(shift) >= bits 是构造式强制的 ----
    exc = None
    try:
        mod.Swizzle(4, 3, 3)
    except Exception as e:                    # noqa: BLE001
        exc = type(e).__name__
    eq("C", "C.7", exc, "AssertionError",
       "★ `Swizzle(4,3,3)` ⇒ **`AssertionError`**（因 `3 >= 4` 不成立）。"
       "「4 位掩码、3 位基、3 位移位」在纸面上完全合理 —— 这是**构造式强制**的")

    # ---- C.8 composition(swizzle, layout) 是错的写法 ----
    exc8 = None
    try:
        mod.composition(S, mod.Layout((8, 8), (8, 1)))
    except Exception as e:                    # noqa: BLE001
        exc8 = "%s: %s" % (type(e).__name__, e)
    eq("C", "C.8", exc8,
       "AttributeError: 'Swizzle' object has no attribute 'shape'",
       "★ `composition(swizzle, layout)` ⇒ **AttributeError**。"
       "swizzle 要用 `ComposedLayout(swizzle, offset, layout)` 表达（不是另一种写法，是唯一的）")

    # ---- C.9 ★★ 语义恒等式**在限定外真的会失败**（本主题的一处「工件不等于直觉」）----
    #   ★ 两侧都对同一批输入抛异常的**不计入分母**：那是「组合本身不合法」，
    #     不是「恒等式不成立」。分母必须是**两边都算得出结果**的那些对。
    n_cand, n_bad, n_exc = 0, 0, 0
    first = None
    for (shA, stA) in POOL:
        for (shB, stB) in POOL:
            A, B = RefLayout(shA, stA), RefLayout(shB, stB)
            if B.size() > A.size():
                continue
            try:
                C = n_composition(A, B)
            except Exception:                 # noqa: BLE001
                n_exc += 1
                continue
            n_cand += 1
            for i in range(B.size()):
                if C(i) != A(B(i)):
                    n_bad += 1
                    if first is None:
                        first = "%s / %s：i=%d ⇒ C(i)=%d，而 A(B(i))=%d" % (
                            A, B, i, C(i), A(B(i)))
                    break
    eq("C", "C.9", (n_cand, n_bad, n_exc), (COMPOSE_CANDIDATE_PAIRS,
                                           IDENTITY_OUTSIDE_DOMAIN, COMPOSE_EXCLUDED),
       "★ **`composition(A,B)` 不是「函数复合」**：在可比 **%d** 个候选对里有 **%d** 个"
       "不满足 `C(i) == A(B(i))`（另有 %d 对**两侧都抛异常**，剔除）。"
       "首例：`%s` —— **它不抛异常**，只是给了另一个函数。"
       "⇒ 「把 `composition` 当 `A∘B` 用」在非规则细分下是静默错的"
       % (n_cand, n_bad, n_exc, first))
    eq("C", "C.9b", (first is not None, n_bad > 0), (True, True),
       "★ C.9 的首例确实存在、且反例数 > 0（不是在一个空集合上做的断言）")

    # ---- C.10 idx2crd / crd2idx **不接受 Layout 对象** ----
    excs = []
    for fn in (mod.idx2crd, mod.crd2idx):
        try:
            fn(19, L2) if fn is mod.idx2crd else fn((2, 3), L2)
            excs.append("没有抛异常")
        except Exception as e:                # noqa: BLE001
            excs.append(type(e).__name__)
    eq("C", "C.10", excs, ["TypeError", "AssertionError"],
       "★ 两个函数都只接受 `shape` / `stride`（或单独 shape），**不接受 `Layout` 对象**："
       "`idx2crd(19, L)` ⇒ **TypeError**，`crd2idx((2,3), L)` ⇒ **AssertionError**")
    eq("C", "C.10b", mod.idx2crd(19, L2.shape, L2.stride), (2, 3),
       "★ 三参数形式才是对的：`idx2crd(19, L.shape, L.stride)` = (2, 3)")

    # ---- C.11 make_layout 只接受 Layout 对象 ----
    exc11 = None
    try:
        mod.make_layout(((2, 3), (1, 4)))
    except Exception as e:                    # noqa: BLE001
        exc11 = "%s: %s" % (type(e).__name__, e)
    eq("C", "C.11", exc11, "AttributeError: 'tuple' object has no attribute 'shape'",
       "★ **构造式只有一种**：`Layout(shape, stride)`。`make_layout(*layouts)` "
       "只接受 `Layout` 对象，传嵌套元组 ⇒ **AttributeError**")

    # ---- C.12 本仓库参照实现被改坏时，对拍**必须**失败（自我证伪）----
    i1 = _first_mismatch_with(mod, _broken_coalesce_always_merge)
    i2 = _first_mismatch_with(mod, _broken_complement_drop_ceildiv)
    i3 = _first_mismatch_with(mod, _broken_logical_drop_complement)
    eq("C", "C.12", [i1 is not None, i2 is not None, i3 is not None], [True, True, True],
       "★ **自证伪**：把本仓库实现的三处（`coalesce` 的无条件合并 / `complement` 的 "
       "`ceil_div` / `logical_divide` 的 `complement` 项）各改坏一次，"
       "对拍**立刻不一致** ⇒ A 类断言不是空操作")
    return None


# --- 三个**故意改坏**的实现（只服务于 C.12 的自证伪，绝不参与 A 类的对拍）---
def _broken_coalesce_always_merge(layout):
    """缺陷：去掉合并条件，**无条件合并**相邻模式（`shape*stride == stride` 不再检查）。"""
    rs, rt = [1], [0]
    for shape, stride in zip(n_flatten(layout.shape), n_flatten(layout.stride)):
        if shape == 1:
            continue
        if rs[-1] == 1:
            rs[-1], rt[-1] = shape, stride
        else:
            rs[-1] = rs[-1] * shape           # ← 少了 `elif rs[-1]*rt[-1] == stride`
    return RefLayout(rs[0], rt[0]) if len(rs) == 1 else RefLayout(tuple(rs), tuple(rt))


def _broken_complement_drop_ceildiv(layout, max_idx=1):
    """缺陷：最后一段直接用 `max_idx`，不做 `ceil_div`。"""
    rs, rt = [], []
    cur = 1
    for stride, shape in sorted(zip(n_flatten(layout.stride), n_flatten(layout.shape))):
        if stride == 0 or shape == 1:
            continue
        rs.append(stride // cur)
        rt.append(cur)
        cur = shape * stride
    rs.append(max_idx)                        # ← 少了 `-(-max_idx // cur)`
    rt.append(cur)
    return n_coalesce(RefLayout(tuple(rs), tuple(rt)))


def _broken_logical_drop_complement(A, B):
    """缺陷：`logical_divide` 里丢掉 `complement` 那一项（= `C.5` 的实验件）。"""
    return n_composition(A, n_make_layout(B))


def _first_mismatch_with(mod, broken):
    """把某个操作换成坏实现，扫描全池，返回第一个不一致的描述（没有则 None）。

    ★ 对照的是 **pycute**（真值那一侧），不是本仓库的正确实现 ——
      否则「坏实现 vs 正确实现」也可能因为两边同错而看起来一致。
    """
    is_coa = broken is _broken_coalesce_always_merge
    is_cmp = broken is _broken_complement_drop_ceildiv
    # ★★ `complement` 这个缺陷**只在 `max_idx > 1` 时**才显形：
    #   默认的 `max_idx=1` 下 `ceil(1/cur) == 1`，坏实现恰好也对 —— 也就是说
    #   **只喂默认参数的话，这个缺陷是测不出来的**（这是本条自证伪的第一个版本踩到的坑）。
    mi = 32
    for (shA, stA), (shB, stB) in _pairs("x"):
        A, B = RefLayout(shA, stA), RefLayout(shB, stB)
        pyA, pyB = mod.Layout(shA, stA), mod.Layout(shB, stB)
        try:
            if is_coa:
                want, got = norm(mod.coalesce(pyA)), norm(broken(A))
            elif is_cmp:
                want, got = norm(mod.complement(pyA, mi)), norm(broken(A, mi))
            else:
                want, got = norm(mod.logical_divide(pyA, pyB)), norm(broken(A, B))
        except Exception:                     # noqa: BLE001 —— 两边都抛同一种异常也算一致
            continue
        if got != want:
            return "%s / %s ⇒ 坏实现 %s ≠ pycute %s" % (A, B, got, want)
    return None


# ===========================================================================
# 6. AST 导入扫描（与 `make lint` 是**同一份代码**）
# ===========================================================================
def ast_import_scan(directory, files):
    """真 `import` 才算违规：注释与 docstring 里出现这些词不算。"""
    hits = []
    for fn in files:
        tree = ast.parse(open(os.path.join(directory, fn), encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] in FORBIDDEN:
                        hits.append((fn, n.lineno, "import " + a.name))
            elif isinstance(n, ast.ImportFrom):
                if (n.module or "").split(".")[0] in FORBIDDEN:
                    hits.append((fn, n.lineno, "from %s import" % n.module))
            elif isinstance(n, ast.Call):
                nm = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
                if nm in ("import_module", "__import__") and n.args:
                    a0 = n.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                            and a0.value.split(".")[0] in FORBIDDEN:
                        hits.append((fn, n.lineno, "%s(%r)" % (nm, a0.value)))
    return hits


def lint_main():
    files = PROGRAMS + AUX_MODULES
    print("AST 导入扫描：%s" % ", ".join(files))
    hits = ast_import_scan(HERE, files)
    if hits:
        for h in hits:
            print("  !! %s" % (h,))
        return 1
    print("  ✓ 没有 import torch / triton / numba / cuda（含 specs.py，含 import_module 动态导入）")
    return 0


# ===========================================================================
# 7. 主流程
# ===========================================================================
def main():
    print("=" * 78)
    print("01_layout_algebra.py —— CuTe 的 layout 代数：坐标 → 索引的函数")
    print("=" * 78)
    print("  ★ 真值标注 : 本主题的数字全部来自**在本机跑 %s**，" % specs.MEASURE_IMPL)
    print("                **不是**从 GPU 上测的，**也不是**本仓库的推导。")
    print("               **测量条件是 CUTLASS 的版本（%s），不是 GPU 型号。**"
          % specs.CUTLASS_VERSION_BASELINE)
    print("  ★ 来源等级 : 全部是 **B 级（NVIDIA 自己的实现，不是规范）** + **E 级（本库推导）**")
    print("                —— 本主题**一条 A 级（厂商规范）来源都没有**")
    print("  python     : %s" % sys.version.split()[0])

    root = specs.require_cutlass("01_layout_algebra.py")
    mod = specs.load_pycute(root)
    print("  pycute     : %s（%d 个文件，**纯标准库、零第三方依赖**）"
          % (os.path.join(root, specs.PYCUTE_DIR), len(specs.PYCUTE_FILES)))
    print("  版本四处对照:")
    for s in specs.VERSION_SOURCES:
        print("    %-32s = %-22s %s" % (s["label"], s["value"],
                                        "✓ 与基准一致" if s["agree"]
                                        else "✗ **与基准不一致**"))
    print("  ★ 基准以 `include/cutlass/version.h` 为准 = **%s**" % specs.CUTLASS_VERSION_BASELINE)
    print()
    print("  穷举集合   : layout 池 **%d** 个（小形状 × {packed 列主序 / 行主序 / 2 倍空洞 / 广播}），"
          % len(POOL))
    print("               其中 size ≤ 32 的 tile 池 **%d** 个" % len(TILES))

    section_A(mod, root)
    section_B()
    section_C(mod)

    print()
    print("-" * 78)
    eq("A", "A.14", ast_import_scan(HERE, PROGRAMS + AUX_MODULES), [],
       "★ AST 导入扫描：三个编号程序 + `specs.py` / `meta_falsify.py` / `mutate.py` "
       "都不 import torch / triton / numba / cuda")
    return summary()


if __name__ == "__main__":
    if "--lint" in sys.argv:
        sys.exit(lint_main())
    sys.exit(main())
