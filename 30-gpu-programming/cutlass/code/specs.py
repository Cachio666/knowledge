#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""specs.py —— 非编号辅助模块：**外部事实常量**（`30-gpu-programming/cutlass`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★★ 本主题的测量条件 —— 逐处都要带，不许省
===============================================================================

本主题的数字**全部**来自：

> **在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`cutlass/python/pycute/` 与 `cutlass/python/cutlass_library/`），
> **不是**从 GPU 上测的，**也不是**本仓库的推导。
> **测量条件是 CUTLASS 的版本，不是 GPU 型号。**

按 `CONTRIBUTING` 第 63 行的四类口径：

| 类 | 本主题里是什么 |
|---|---|
| **规格值** | **无**。本主题**没有**任何规范级来源——CUTLASS 的源码是**实现**，不是规范（P8）。|
| **推导值** | 本仓库程序当场算出的（layout 代数的对拍结果、rank 缩减统计、像集合……）|
| **引用值** | CUTLASS 源码里的常量与其 `file:line`（本仓库**不复核**它们的测量口径，只逐字引）|
| **本机实测值** | **真跑 `pycute` / `cutlass_library` 的输出**——测的是 **CUTLASS 4.8.0 的 Python 实现**，不是硬件 |

⇒ 本主题的数字**只能支撑关于 layout 代数与生成器策略的结论，不能支撑关于 GPU 硬件的结论。**

===============================================================================
★★ 来源等级（沿用 `45-kernel-lifecycle` 的 A–E，本模块只用其中两级）
===============================================================================

  · **B 级：某一实现的官方材料** —— `pycute` / `cutlass_library` 是 **NVIDIA 自己写的实现**，
    它们的**源码与常量**属此类。**B 级只能支撑「NVIDIA 的这个实现是这么做的」，
    不能支撑「CUTLASS 规定必须这么做」，更不能支撑「硬件如此」**（P8 规范与实现不分）。
    ★ 本主题**刻意没有 A 级**：CUTLASS 没有给这套 layout 代数出过规范文本。
  · **E 级：本库推导** —— 本模块**不含推导值**。所有推导（对拍、像集合、rank 统计）
    都在编号程序内部当场算出并断言，**不搬进 `specs.py`**（`CONTRIBUTING` 第 48 行）。
  · **D 级（社区项目 / 二手转述）：0 条。**

★ **本模块的引文是怎么来的**：`quote` 字段是**从源码文件里按 `file` + `line` 抄下来的**，
  不是凭印象写的英文。**转写层是最容易出错又最难自查的一层**（计划 §2「本域特有的两条」第 3 条），
  所以 `must_contain` 的每一条都是**短而无歧义的字面量**，由程序在**原始字节**上
  `str.count()` 计数 —— **不在「我记得是这样」上做判断**。
  `01_layout_algebra.py` 的 `A.12` 会把每个锚点的 `file` + `line` 上的**真实行内容**
  读出来，逐字比对 `quote`（**改了源码或抄错了引文，断言立刻失败**）。

★★ **`# 来源：` 的机器可核形式就是 `ANCHORS[*]` 的 `file` + `line` 两个字段**
  （`cuda/code/specs.py` 的先例是同一做法）：
  · `file` + `line`  出处 —— 由 `A.12` 在**原始字节**上逐字核对（31 条锚点全部覆盖）；
  · `level`          来源等级 —— 由 `A.15` 断言**整张表的取值集合恰好是 `{'B'}`**
                     （本主题**刻意没有 A 级**：CUTLASS 没有为这套 layout 代数出过规范）；
  · **没有 `file` + `line` 的 CUTLASS 源码引用一律不写进本模块**（见 `UNSOURCED`）。

★ **本模块不放什么**（计划 §0.7 C2 的硬要求）：
  **任何没有文件路径 + 行号的 CUTLASS 源码引用，一律不写进来。**
  找不到 `file:line` 的项记入 `UNSOURCED`，**正文与常量名里都不出现**。

★ **本模块的路径探测**：本域代码形态规定「需要探测外部工件路径时，不得写死构造者的目录布局；
  找到后把选中路径打印出来，找不到以可操作的错误退出」。
  探测逻辑放在本文件（三个编号程序共用同一份），候选列表见 `CUTLASS_CANDIDATES`。
"""

import os
import sys

# ===========================================================================
# 0. 测量条件（每处引用都要带的那句话）
# ===========================================================================
MEASURE_IMPL = "CUTLASS 4.8.0 的 Python 参考实现（python/pycute、python/cutlass_library）"
MEASURE_NOTE = ("测量条件是 **CUTLASS 的版本**，不是 GPU 型号；"
                "不是从 GPU 上测的，也不是本仓库的推导")
PROBE_DATE = "2026-09-20"

# ===========================================================================
# 1. CUTLASS 版本 —— **四处来源不一致**，必须逐处列出（不得只挑一处读）
# ===========================================================================
#
# ★ 出处：计划 §0.7 D1。控制者已逐处实跑，实施者（本轮）已**独立复算**：
#   version.h 41–43 行 = 4.8.0；CHANGELOG 第 5 行 = 4.8.0（2026-08-25）；
#   `git describe --tags --always` = v4.8.0dev-2-gdc45f979；
#   pyproject.toml 第 6 行 = 4.2.0.0（**陈旧**）。四处来源**互相矛盾**。
CUTLASS_VERSION_BASELINE = "4.8.0"     # 以 include/cutlass/version.h 为准

VERSION_SOURCES = (
    # (来源标签, 相对 CUTLASS 根的路径或命令, 行号, 值, 是否与基准一致)
    {"label": "include/cutlass/version.h", "kind": "file",
     "path": "include/cutlass/version.h", "line": 41, "value": "4.8.0",
     "value_lines": (41, 42, 43),   # 三行分别给 MAJOR / MINOR / PATCH，用 "." 拼起来
     "agree": True,
     "quote": "#define CUTLASS_MAJOR 4",
     "note": "CUTLASS_MAJOR 4 / CUTLASS_MINOR 8 / CUTLASS_PATCH 0（41/42/43 行）⇒ 这是**基准**"},
    {"label": "CHANGELOG.md 首条标题", "kind": "file",
     "path": "CHANGELOG.md", "line": 5, "value": "4.8.0",
     "value_lines": (5,),
     "agree": True,
     "quote": "## [4.8.0](https://github.com/NVIDIA/cutlass) (2026-08-25)"},
    {"label": "git describe --tags --always", "kind": "git",
     "path": None, "line": None, "value": "v4.8.0dev-2-gdc45f979",
     "value_lines": None,           # ★ 需要 `git describe`，**本程序不 spawn 任何进程** ⇒ 不核
     "agree": True,
     "note": "tag 后 2 个 commit ⇒ 与 4.8.0 一致"},
    {"label": "pyproject.toml 的 version", "kind": "file",
     "path": "pyproject.toml", "line": 7, "value": "4.2.0.0",
     "value_lines": (7,),
     "agree": False,
     "quote": 'version = "4.2.0.0"',
     "note": "★ **陈旧，落后三个次版本**。照它写「本库基于 CUTLASS 4.2」，"
             "后面所有 sm_100 的 atom 都会变成「这个版本不该有的东西」"},
)

CUTLASS_HEAD = "dc45f979ae336a235da1676b311f35efeb30149a"
CUTLASS_HEAD_DATE = "2026-08-28"

# ★ pyproject.toml 的 dependencies 共 **6** 条，其中 networkx / pydot / treelib
#   **本机都没装**，而 cutlass_library 的 12/12 模块导入**全部成功**
#   ⇒ 它们是**懒加载**的，只有生成器的 CLI 路径才需要。
CUTLASS_PYPROJECT_DEPENDENCIES = ("cuda-python>=11.8.0", "networkx", "numpy",
                                 "pydot", "scipy", "treelib")
CUTLASS_PYPROJECT_DEP_LINES = 17          # `dependencies = [` 所在行
NOT_INSTALLED_LOCALLY = ("networkx", "pydot", "treelib", "cuda-python")

# ===========================================================================
# 2. `pycute` 的源码事实（B 级：NVIDIA 官方**实现**，不是规范）
# ===========================================================================
PYCUTE_DIR = "python/pycute"
PYCUTE_FILES = ("__init__.py", "int_tuple.py", "layout.py", "swizzle.py", "typing.py")
PYCUTE_SETUP = "python/setup_pycute.py"

# ★「导出 48 个符号」的口径必须写明（P9）：是
#   `[n for n in dir(pycute) if not n.startswith('_')]` 的**元素个数**，
#   其中**包含 4 个子模块名**（layout / swizzle / int_tuple / typing 被 `dir()` 看见）
#   与 3 个从 `typing` 里带进来的名字（ABC / Union / chain）。
#   ⇒ 它**不是**「pycute 定义了 48 个函数」。
PYCUTE_EXPORT_COUNT = 48
PYCUTE_EXPORTED = (
    'ABC', 'ComposedLayout', 'Integer', 'Layout', 'LayoutBase', 'Swizzle', 'Union',
    'chain', 'coalesce', 'complement', 'composition', 'cosize', 'crd2crd', 'crd2idx',
    'elem_scale', 'filter', 'flatten', 'has_none', 'hier_unzip', 'idx2crd',
    'inner_product', 'int_tuple', 'is_int', 'is_layout', 'is_tuple', 'layout',
    'left_inverse', 'logical_divide', 'logical_product', 'make_layout', 'prefix_product',
    'product', 'reduce', 'right_inverse', 'shape_div', 'shiftl', 'shiftr', 'signum',
    'size', 'slice_', 'slice_and_offset', 'swizzle', 'tiled_divide', 'tiled_product',
    'tuple_max', 'typing', 'zipped_divide', 'zipped_product',
)

# ★ `cutlass/python/README.md` 通篇不提 `pycute`（0 次）⇒ 引 `pycute` 只能引**源码路径**，
#   **不得写「CUTLASS 官方提供 pycute 包」**。下列计数是在**原始字节**上数的。
PYTHON_README_PATH = "python/README.md"
PYTHON_README_COUNTS = {"pycute": 0, "PyCute": 0, "CuTeDSL": 0, "CuTe DSL": 0,
                        "cutlass_library": 7, "cutlass_cppgen": 1}
PYPROJECT_PYCUTE_COUNT = 0

# ===========================================================================
# 3. `pycute` 的实测输出表（**本机实测值**；每条都带探针表达式）
# ===========================================================================
#
# ★ 这些不是源码引用，是**跑出来的输出**。每条带：
#   expr   —— 在 `01_layout_algebra.py` 里当场重跑的探针表达式（字符串，供人核对）
#   value  —— 规划阶段（2026-09-20）记录的值，逐字
#   by     —— 哪条断言重跑并比对它
# ⇒ 程序会把 expr **真跑一遍**，与 value 比对；**改错任一方，断言失败**。

PYCUTE_MEASURED = (
    # ---- 事实 1：默认 stride 是【列主序】 ----
    {"key": "F1a", "expr": "str(Layout((4,8)))", "value": "(4, 8):(1, 4)",
     "by": ("A.1",),
     "note": "★ **最左维最快** ⇒ 与 numpy / 教材的默认**相反**"},
    {"key": "F1b", "expr": "str(Layout((8,4)))", "value": "(8, 4):(1, 8)", "by": ("A.1",)},
    {"key": "F1c", "expr": "str(Layout((2,3,4)))", "value": "(2, 3, 4):(1, 2, 6)", "by": ("A.1",)},
    {"key": "F1d", "expr": "str(Layout((4,8),(8,1)))", "value": "(4, 8):(8, 1)",
     "by": ("A.1",), "note": "行主序要**显式写**"},
    # ---- 事实 2：complement 的默认 max_idx=1 ----
    {"key": "F2a", "expr": "str(complement(Layout((4,8),(8,1))))", "value": "1:0",
     "by": ("A.2",), "note": "默认 max_idx=1 ⇒ 对已 packed 的 layout 给出**平凡结果**"},
    {"key": "F2b", "expr": "str(complement(Layout((4,8),(8,1)), 32))", "value": "1:0", "by": ("A.2",)},
    {"key": "F2c", "expr": "str(complement(Layout(4), 32))", "value": "8:4", "by": ("A.2",)},
    # ---- 事实 3：logical_divide ----
    {"key": "F3a", "expr": "str(logical_divide(L, Layout(4)))", "value": "(4, 8):(8, 1)",
     "by": ("A.4",), "note": "L = (4,8):(8,1)"},
    {"key": "F3b", "expr": "str(logical_divide(L, Layout(8)))", "value": "((4, 2), 4):((8, 1), 2)",
     "by": ("A.4",)},
    {"key": "F3c", "expr": "str(logical_divide(L, Layout(2)))", "value": "(2, (2, 8)):(8, (16, 1))",
     "by": ("A.4",)},
    {"key": "F3d", "expr": "str(logical_divide(L, Layout((4,2),(1,4))))",
     "value": "((4, 2), 4):((8, 1), 2)", "by": ("A.4",)},
    {"key": "F3e", "expr": "str(logical_divide(L, Layout((2,4),(1,2))))",
     "value": "((2, (2, 2)), 4):((8, (16, 1)), 2)", "by": ("A.4",)},
    # ---- 事实 4：非退化例子（行主序 A=(8,4):(4,1)）----
    {"key": "F4a", "expr": "str(logical_divide(A, Layout(4)))",
     "value": "(4, (2, 4)):(4, (16, 1))", "by": ("A.4b",), "note": "A = (8,4):(4,1)"},
    {"key": "F4b", "expr": "str(logical_divide(A, Layout((4,2),(1,4))))",
     "value": "((4, 2), 4):((4, 16), 1)", "by": ("A.4b",)},
    {"key": "F4c", "expr": "str(logical_divide(A, Layout((2,2),(1,2))))",
     "value": "((2, 2), (2, 4)):((4, 8), (16, 1))", "by": ("A.4b",)},
    # ---- 事实 6：right_inverse ----
    {"key": "F6a", "expr": "str(right_inverse(Layout((4,8),(8,1))))", "value": "(8, 4):(4, 1)",
     "by": ("A.6",)},
    {"key": "F6b", "expr": "str(right_inverse(Layout((8,4),(1,8))))", "value": "32:1", "by": ("A.6",)},
    {"key": "F6c", "expr": "str(right_inverse(Layout(6,2)))", "value": "1:0",
     "by": ("A.6", "C.4"), "note": "非前缀满射 ⇒ **退化为平凡**"},
    # ---- 事实 7：coalesce ----
    {"key": "F7a", "expr": "str(coalesce(Layout((4,8),(8,1))))", "value": "(4, 8):(8, 1)",
     "by": ("A.3",)},
    {"key": "F7b", "expr": "str(coalesce(Layout((2,2),(1,2))))", "value": "4:1",
     "by": ("A.3", "B.2"), "note": "★ **合并**（shape*stride 相接）"},
    {"key": "F7c", "expr": "str(coalesce(Layout((2,2),(2,1))))", "value": "(2, 2):(2, 1)",
     "by": ("A.3", "B.2"), "note": "★ **不合并**（次序是反的）"},
    {"key": "F7d", "expr": "str(coalesce(Layout((2,2),(4,1))))", "value": "(2, 2):(4, 1)",
     "by": ("A.3", "B.2"), "note": "★ **不合并**（中间有空洞）"},
    # ---- 陷阱 A：模块级 size() 对 Swizzle 失效（**只在 Swizzle 上**）----
    {"key": "TA1", "expr": "isinstance(Swizzle(3,3,3), LayoutBase)", "value": "False",
     "by": ("C.6",)},
    {"key": "TA2", "expr": "Swizzle(3,3,3).size()", "value": "512", "by": ("C.6",),
     "note": "方法，**正确**"},
    {"key": "TA3", "expr": "str(size(Swizzle(3,3,3)))", "value": "SW_3_3_3",
     "by": ("C.6",), "note": "★ 模块函数走 `product()` 分支 ⇒ **把 Swizzle 自己返回了**"},
    {"key": "TA4", "expr": "cosize(Swizzle(3,3,3))", "value": "512", "by": ("C.6b",),
     "note": "★★ **`cosize` 不失效**：模块级 `cosize()` 无条件调 `.cosize()`，"
             "而 `Swizzle` 恰有该方法（见 A 锚点的 `layout.py:132-133`）。"
             "⇒ 这条陷阱**只能限定在 `size()` 上**"},
    # ---- 陷阱 A 的**作用域反面**：在 Layout 上两者一致 ----
    {"key": "TA5", "expr": "(Layout((4,8),(8,1)).size(), size(Layout((4,8),(8,1))))",
     "value": "(32, 32)", "by": ("C.6b",),
     "note": "★ 在 `Layout` 上方法 == 模块函数 ⇒ **不得说成「pycute 的 size() 不可靠」**（P10）"},
    # ---- 陷阱 B：abs(shift) >= bits 是构造式强制的 ----
    {"key": "TB1", "expr": "'Swizzle(4,3,3)' 的构造", "value": "AssertionError",
     "by": ("C.7",), "note": "因 `3 >= 4` 不成立"},
    # ---- 事实 5：composition(swizzle, layout) 是错的写法 ----
    {"key": "F5", "expr": "composition(Swizzle(3,3,3), Layout((8,8),(8,1)))",
     "value": "AttributeError", "by": ("C.8",),
     "note": "★ 要用 `ComposedLayout(swizzle, offset, layout)` 表达"},
)

# ===========================================================================
# 4. ★★ ANCHORS —— 外部源码引用（**每条都有 file + line**，没有的一律不进本表）
# ===========================================================================
#
# 字段：
#   file        相对 CUTLASS 根的路径
#   line        行号（单行）或 (起, 止)
#   quote       该行（或该块的首行）的**逐字内容**，由程序在原始字节上比对
#   level       来源等级（本主题只有 B 与 E；见模块头）
#   must_contain  必须逐字出现的短字面量（改一个字，程序就报错）
#   note        作用域说明。★ 它**不单独充当依据**（P5）
#   covered_by  被哪些断言支撑（标签必须**真实存在**于某个编号程序的断言清单里）
#
# ★★ `covered_by` 的三条规矩（照抄 `cuda/code/specs.py`）：
#   ① 每个锚点的 `covered_by` **非空**；
#   ② 里面每个标签都必须**真实存在**于某个编号程序的断言清单里（不是编的）；
#   ③ `note` 里凡属**承重论断**的句子，必须能被 `covered_by` 指向的断言机械判定；
#      做不到的，`note` 里必须**当场写明这一点**，不得写成结论。

ANCHORS = {

    # -----------------------------------------------------------------------
    # 4.1 `pycute/layout.py` —— `01_layout_algebra.py` 消费
    # -----------------------------------------------------------------------
    "P1": {
        "file": "python/pycute/layout.py", "line": (52, 57), "level": "B",
        "quote": "def __init__(self, _shape, _stride=None):",
        "must_contain": ("def __init__(self, _shape, _stride=None):",
                         "self.stride = prefix_product(self.shape)"),
        "note": "★ `Layout(shape)` 的默认 stride 走 `prefix_product`（**最左维最快**）"
                "⇒ 这就是「**CuTe 的默认是列主序**」的源码出处（`C.1` 逐点对拍它）。"
                "★ 第 55 行 `self.stride = prefix_product(self.shape)` 是本条的全部内容。",
        "covered_by": ("A.1", "C.1"),
    },
    "P2": {
        "file": "python/pycute/layout.py", "line": (99, 104), "level": "B",
        "quote": "def size(self):",
        "must_contain": ("def size(self):", "return product(self.shape)",
                         "def cosize(self):", "return self(self.size() - 1) + 1"),
        "note": "★ `size` 是**定义域**大小（shape 的乘积），`cosize` 是**陪域**大小"
                "（最大像 + 1）—— 两者对**非满射** layout 不同（`B.1` 穷举统计差异面）。",
        "covered_by": ("B.1", "A.9"),
    },
    "P3": {
        "file": "python/pycute/layout.py", "line": (137, 164), "level": "B",
        "quote": "def coalesce(layout, profile=None):",
        "must_contain": ("def coalesce(layout, profile=None):",
                         "result_shape[-1] * result_stride[-1] == stride",
                         "result_shape[-1] = result_shape[-1] * shape"),
        "note": "★ 合并条件**逐字**是 `result_shape[-1] * result_stride[-1] == stride`："
                "两个模式相接（前一个的 shape×stride 恰好等于后一个的 stride）才合并。"
                "⇒ `(2,2):(1,2)` 合、`(2,2):(2,1)` **不合**、`(2,2):(4,1)` **不合**"
                "（`A.3`/`B.2` 逐点对拍这三个）。",
        "covered_by": ("A.3", "B.2", "C.2"),
    },
    "P4": {
        "file": "python/pycute/layout.py", "line": (232, 256), "level": "B",
        "quote": "def complement(layout, max_idx=1):",
        "must_contain": ("def complement(layout, max_idx=1):",
                         "result_shape.append((max_idx + current_idx - 1) // current_idx)",
                         "return coalesce(Layout(tuple(result_shape), tuple(result_stride)))"),
        "note": "★ **默认 `max_idx=1`** ⇒ 最后一段的 shape 是 `ceil(1/current_idx) == 1`，"
                "对已 packed 的 layout 给出**平凡结果 `1:0`**（`A.2`/`C.3` 两个方向都断言）。"
                "★ 返回前**过一道 `coalesce`** —— 这是「结果与输入打印相同但语义不同」的来源之一。",
        "covered_by": ("A.2", "C.3"),
    },
    "P5": {
        "file": "python/pycute/layout.py", "line": 307, "level": "B",
        "quote": "return composition(layoutA, make_layout(layoutB, complement(layoutB, size(layoutA))))",
        "must_contain": ("return composition(layoutA, make_layout(layoutB,"
                         " complement(layoutB, size(layoutA))))",),
        "note": "★★ **本主题的推导核心**：`logical_divide(A,B)` **就是**这一行。"
                "`complement(B, size(A))` 必须在那里，否则 `make_layout(B, …)` 的坐标域"
                "覆盖不满 `A` 的定义域 ⇒ `composition` 会**静默给出另一个函数**"
                "（`C.5` 把这一点做成可证伪的：去掉 `complement`，对拍必须失败）。",
        "covered_by": ("A.4", "A.5", "C.5"),
    },
    "P6": {
        "file": "python/pycute/layout.py", "line": (260, 283), "level": "B",
        "quote": "def right_inverse(layout):",
        "must_contain": ("def right_inverse(layout):", "if current_idx != stride:",
                         "break"),
        "note": "★ 那个 `break` 是「**非前缀满射 ⇒ 退化为平凡**」的全部机制："
                "只要排序后的某个 stride 与累积值对不上，就**停止累积**，"
                "结果退化成 `1:0`（`A.6` 对拍 + `C.4` 断言 `right_inverse(6:2) == 1:0`）。",
        "covered_by": ("A.6", "C.4"),
    },
    "P7": {
        "file": "python/pycute/layout.py", "line": (343, 350), "level": "B",
        "quote": "def zipped_divide(layoutA, layoutB):",
        "must_contain": ("def zipped_divide(layoutA, layoutB):",
                         "return hier_unzip(logical_divide, layoutA, layoutB)",
                         "result = zipped_divide(layoutA, layoutB)"),
        "note": "★ `zipped_divide` = `hier_unzip(logical_divide, …)`，`tiled_divide` 再包一层。"
                "⇒ 计划 §0.7 C 事实 3 里「zipped_divide 与 logical_divide 逐行相同」"
                "**只在二维 R^2 的展平打印上成立**，二者在**层次结构**上是不同的对象 —— "
                "`A.5` 用 `make_layout` 的展平形式逐点对拍，而不是比较打印串。",
        "covered_by": ("A.5",),
    },
    "P8": {
        "file": "python/pycute/layout.py", "line": (125, 128), "level": "B",
        "quote": "def size(layout):",
        "must_contain": ("def size(layout):", "if is_layout(layout):",
                         "return layout.size()", "return product(layout)"),
        "note": "★★ **陷阱 A 的全部机制**（只有两行）：`is_layout` 走的是"
                "`isinstance(x, LayoutBase)`，而 `Swizzle` **不是** `LayoutBase` 的子类"
                "（见锚点 `S2`）⇒ 对 Swizzle，模块级 `size()` 落到 `product(layout)` 分支，"
                "`product` 对非 tuple 输入**原样返回**（锚点 `P11`）⇒ 它把 Swizzle **自己**返回了。",
        "covered_by": ("C.6",),
    },
    "P9": {
        "file": "python/pycute/layout.py", "line": (132, 133), "level": "B",
        "quote": "def cosize(layout):",
        "must_contain": ("def cosize(layout):", "return layout.cosize()"),
        "note": "★★ **与 `size()` 的关键差别**：模块级 `cosize()` **无条件**调 `layout.cosize()`，"
                "**没有** `is_layout` 分支。而 `Swizzle` 恰有 `cosize()` 方法"
                "（锚点 `S3`）⇒ **`cosize(Swizzle)` 给出正确的 512**。"
                "⇒ 「模块级 size()/cosize() 对 Swizzle 失效」这句话里，"
                "**只有 `size()` 那一半成立**（`C.6b` 双向钉住）。",
        "covered_by": ("C.6b",),
    },
    "P10": {
        "file": "python/pycute/layout.py", "line": (190, 228), "level": "B",
        "quote": "def composition(layoutA, layoutB):",
        "must_contain": ("def composition(layoutA, layoutB):", "flat_A = coalesce(layoutA)",
                         "assert curr_shape % rest_stride == 0 or rest_stride % curr_shape == 0",
                         "rest_stride = -(-rest_stride // curr_shape)"),
        "note": "★ `composition` **先 `coalesce(layoutA)`**，再逐模式切分 —— 所以它**不是**"
                "「函数复合」的一般实现，而是「按 A 的模式切分 B」的专用算法。"
                "⇒ `C(i) == A(B(i))` **只在 A 与 B 都是 packed 且 `size(A) % size(B) == 0` 时成立**；"
                "`A.7` 就是在这条**限定后的子域**上断言它，`C.9` 断言**限定外真的会失败**。",
        "covered_by": ("A.7", "C.9"),
    },
    "P11": {
        "file": "python/pycute/int_tuple.py", "line": (65, 69), "level": "B",
        "quote": "def product(a):",
        "must_contain": ("def product(a):", "if is_tuple(a):", "return a"),
        "note": "★ 非 tuple 输入**原样返回**。这就是陷阱 A 里 `size(Swizzle)` "
                "返回 Swizzle 自身的**最后一步**（`C.6` 直接断言 `product(S) is S`）。",
        "covered_by": ("C.6",),
    },
    "P12": {
        "file": "python/pycute/int_tuple.py", "line": (142, 160), "level": "B",
        "quote": "def idx2crd(idx, shape, stride=None):",
        "must_contain": ("def idx2crd(idx, shape, stride=None):",
                         "def crd2idx(crd, shape, stride=None):"),
        "note": "★ 两个函数都只接受 **shape / stride**（或单独的 shape），"
                "**不接受 `Layout` 对象**（`C.10` 真的把它跑出 `TypeError`）。",
        "covered_by": ("C.10",),
    },
    "P13": {
        "file": "python/pycute/layout.py", "line": (116, 121), "level": "B",
        "quote": "def make_layout(*layouts):",
        "must_contain": ("def make_layout(*layouts):", "(a.shape,a.stride) for a in layouts"),
        "note": "★ 构造式**只有一种**：`Layout(shape, stride)`。`make_layout(*layouts)` "
                "**只接受 `Layout` 对象**，传嵌套元组会 `AttributeError`（`C.11` 真跑）。",
        "covered_by": ("C.11",),
    },

    # -----------------------------------------------------------------------
    # 4.2 `cutlass_library` —— `02_tiling_decompose.py` 消费
    # -----------------------------------------------------------------------
    "L1": {
        "file": "python/cutlass_library/library.py", "line": (1455, 1461), "level": "B",
        "quote": "class MathInstruction:",
        "must_contain": ("class MathInstruction:", "instruction_shape,",
                         "element_a, element_b, element_accumulator,"),
        "note": "★ **首参是 `instruction_shape`，是一个元组**。"
                "`MathInstruction(16,8,16,f16,f16,f32)`（**错误写法**）**不报错**，"
                "首参被当成 `instruction_shape=16` ⇒ `C.12` 真的把它跑出来。",
        "covered_by": ("C.1",),
    },
    "L2": {
        "file": "python/cutlass_library/library.py", "line": (1472, 1483), "level": "B",
        "quote": "class TileDescription:",
        "must_contain": ("class TileDescription:",
                         "def __init__(self, threadblock_shape, stages, warp_count, math_instruction,",
                         "self.threadblock_shape = threadblock_shape",
                         "self.tile_shape = threadblock_shape"),
        "note": "★★ **参数是 `warp_count`，属性里没有 `warp_shape`**（`C.13` 用 `hasattr` 双向断言）。"
                "★★ `tile_shape` 与 `threadblock_shape` 是**同一个对象挂在两个名字下** —— "
                "这是 **P9「同名不同指」的镜像：同指不同名**（`C.14` 断言 `is` 且改一个两个都变）。",
        "covered_by": ("C.2", "C.3"),
    },
    "L3": {
        "file": "python/cutlass_library/library.py", "line": (1485, 1496), "level": "B",
        "quote": "def procedural_name(self):",
        "must_contain": ('if self.minimum_compute_capability >= 90:',
                         'return "%dx%d_%dx%d" % (self.threadblock_shape[0],'
                         ' self.threadblock_shape[1], self.threadblock_shape[2], self.stages)'),
        "note": "★ **sm90 是命名方案的分界点**：`min_compute >= 90` 走含 cluster 的三段式，"
                "否则走不含 cluster 的四段式（`B.4` 用六行逐字输出钉住）。"
                "★★ 断言它的输入必须与输出**同源**：`else` 分支的第三个数就是"
                "`threadblock_shape[2]`，**同一个对象的 tbk 不可能同时是 32 与 64**。",
        "covered_by": ("B.4",),
    },
    "L4": {
        "file": "python/cutlass_library/sm90_utils.py", "line": (385, 396), "level": "B",
        "quote": "tile_desc = TileDescription(",
        "must_contain": ("tile_desc = TileDescription(", "math_inst.instruction_shape[0] * mma_mul[0],",
                         "math_inst.instruction_shape[1] * mma_mul[1],",
                         "math_inst.instruction_shape[2] * mma_mul[2]"),
        "note": "★ **CTA 形状 = 指令形状 × 乘数（逐元素）**。`B.3`/`B.5` 在**枚举结果**上逐条验证它："
                "本仓库自己重跑这个乘积循环，断言发射出的每个 `threadblock_shape` "
                "都落在乘积集合里。",
        "covered_by": ("A.2", "B.3"),
    },
    "L5": {
        "file": "python/cutlass_library/sm90_utils.py", "line": 392, "level": "B",
        "quote": "warp_count=[4, 1, 1],",
        "must_contain": ("warp_count=[4, 1, 1],",),
        "note": "★ SM90 的 `warp_count` 是**写死的**，不随形状变（`B.6` 在**全部**枚举结果上断言）。",
        "covered_by": ("B.6",),
    },
    "L6": {
        "file": "python/cutlass_library/sm90_utils.py", "line": (351, 353), "level": "B",
        "quote": "    # CTA shape upper bound: <256, 256, 256>",
        "must_contain": ("# CTA shape upper bound: <256, 256, 256>",
                         "if cta_shape[0] > 256 or cta_shape[1] > 256 or cta_shape[2] > 256:"),
        "note": "★ CTA 各维**上限 256**，超出即 `is_tile_desc_valid` 返回 False。"
                "★ 注意注释写的是 `<256, 256, 256>` 而代码判的是 `> 256`（**含 256**）——"
                "本库按**代码**断言（`B.7`），不按注释（P8：注释是作者写的，不是规范）。",
        "covered_by": ("B.7",),
    },
    "L7": {
        "file": "python/cutlass_library/sm90_utils.py", "line": (378, 383), "level": "B",
        "quote": "        # generator can stamp out duplicate kernels, because it doesn't explicitly set instruction",
        "must_contain": ("# generator can stamp out duplicate kernels, because it doesn't explicitly set instruction",
                         "# shape for SM90 kernels, and the 3.X collective API doesn't directly expose them when using",
                         "# the auto kernel schedule.",
                         "math_inst_stub.instruction_shape = [0, 0, 0]"),
        "note": "★★ **一条从工件里长出来的一手泄漏**（作者原话，逐字引）："
                "生成器**故意把 `instruction_shape` 置零** ⇒ "
                "**同一个 `threadblock_shape` 可以由不同的 (指令形状, 乘数) 对产生，"
                "而发射出的 kernel 不携带这个信息**。"
                "`B.8` 用**实测**把它量化：见 `SM90_LEAK`。"
                "★ 这段注释是**某个实现的作者写的**，不是规范（P8）——它**只能**支撑"
                "「CUTLASS 的这个生成器这么做」，不能支撑任何关于性能或硬件的断言（P7）。",
        "covered_by": ("B.8",),
    },
    "L8": {
        "file": "python/cutlass_library/sm90_utils.py", "line": (128, 133), "level": "B",
        "quote": "def generate_tf32_math_instruction_shapes_sm90(level: int):",
        "must_contain": ("wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_TF32_DENSE.items() if level >= min_level"),
        "note": "★★ **形状表的 value 是「生成档位」**（`min_level`）—— 筛选条件是 `level >= min_level`。"
                "⇒ 它不是指令数、不是代价、不是权重（P4 高危点）。"
                "`B.9` 用**表本身**算出「档位 0 时入选的键集」并断言它恰好等于"
                "`value == 0` 的那些键 —— 这是**可机械判定**的，不需要解释。",
        "covered_by": ("B.2",),
    },
    "L9": {
        "file": "python/cutlass_library/sm90_shapes.py", "line": (33, 41), "level": "B",
        "quote": "These shape and level pairs are defined as dicts, where keys are shapes and values are their",
        "must_contain": ("where keys are shapes and values are their",
                         "associated levels", "it will be excluded, and otherwise, included.",
                         "Level 0 is always emitted."),
        "note": "★ 与 `L8` **互为正反**：`L8` 是代码，`L9` 是同一件事的 docstring。两条必须同时引（P7）。"
                "★ 这段 docstring 讲的是**生成器策略**，**不能**拿来支持任何关于硬件吞吐的断言（P7）。",
        "covered_by": ("B.2",),
    },
    "L10": {
        "file": "python/cutlass_library/library.py", "line": (1191, 1200), "level": "B",
        "quote": "ArchitectureNames = {",
        "must_contain": ("ArchitectureNames = {", "90: 'hopper'", "89: 'ada'"),
        "note": "★ **`ArchitectureNames` 只到 90**（没有 sm_100 / sm_107 / sm_120），"
                "而同一棵树里 `include/cute/arch/` 已有 sm100/sm107/sm120 的文件 ⇒ "
                "**「CUTLASS 支持哪些架构」必须指明问的是哪个子系统**（`B.12` 双向断言）。",
        "covered_by": ("B.10",),
    },
    "L11": {
        "file": "python/cutlass_library/sm100_shapes.py", "line": (343, 366), "level": "B",
        "quote": "SM100_MMA_SHAPES_MXF4NVF4_DENSE_1SM = {",
        "must_contain": ("SM100_MMA_SHAPES_MXF4NVF4_DENSE_1SM = {",
                         "SM100_MMA_SHAPES_MXF4NVF4_DENSE_2SM = {"),
        "note": "★★ **限定语（P10）**：在这**一族**上，2SM 的 M 恰好是 1SM 的两倍（128 → 256），"
                "N/K 不变。**不得推广成「SM100 的 2SM 让 M 翻倍」** —— "
                "`SM100_MMA_SHAPES_TF32_DENSE_1SM` 的 M 就是 **64**。"
                "`B.10` 把两个方向都断言（该族成立、TF32 反例存在）。",
        "covered_by": ("B.10",),
    },
    "L12": {
        "file": "python/cutlass_library/sm90_utils.py", "line": (90, 103), "level": "B",
        "quote": "def get_wgmma_level_from_global_level(global_level: int):",
        "must_contain": ("def get_wgmma_level_from_global_level(global_level: int):",
                         "return global_level % 10",
                         "def get_mma_level_from_global_level(global_level: int):",
                         "return (global_level // 10) % 10",
                         "def get_cluster_level_from_global_level(global_level: int):",
                         "return (global_level // 100) % 10",
                         "def get_pruning_level_from_global_level(global_level: int):",
                         "return (global_level // 1000) % 10"),
        "note": "★★ **P9「同名不同指」的载体**：`level` 这个词有两个层级 —— "
                "形状表里的 `min_level` 是**某个分类内部的档位**，"
                "而这四个函数解码的是**全局档位**的四个十进制位（个/十/百/千）。"
                "⇒ 正文凡出现 `level` 必须**当场说明是哪一个**。`B.11` 逐位断言解码。",
        "covered_by": ("A.1",),
    },

    # -----------------------------------------------------------------------
    # 4.3 `pycute/swizzle.py` —— `03_swizzle_probe.py` 消费
    # -----------------------------------------------------------------------
    "S1": {
        "file": "python/pycute/swizzle.py", "line": (48, 58), "level": "B",
        "quote": "## A generic Swizzle functor",
        "must_contain": ("## A generic Swizzle functor", "0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx",
                         "Base is the number of least-sig bits to keep constant",
                         "Bits is the number of bits in the mask",
                         "Shift is the distance to shift the YYY mask",
                         "the result is", "where AA = ZZ xor YY"),
        "note": "★ **逐字引用**作者画的位图。它说明的是 `Swizzle` 的**位运算定义**，"
                "**不是一个关于 bank 冲突的规范**（P8）。"
                "`03` 篇 §1 直接引这一段。",
        "covered_by": ("A.1", "A.2"),
    },
    "S2": {
        "file": "python/pycute/swizzle.py", "line": (60, 70), "level": "B",
        "quote": "class Swizzle:",
        "must_contain": ("class Swizzle:", "def __init__(self, bits, base, shift):",
                         "assert abs(shift) >= bits",
                         "self.yyy_msk = bit_msk << (base + max(0,shift))",
                         "self.zzz_msk = bit_msk << (base - min(0,shift))"),
        "note": "★★ **两个承重事实同在这一段**："
                "① `assert abs(shift) >= bits` 是**构造式强制的**（`C.7` 真跑 `Swizzle(4,3,3)`）；"
                "② yyy/zzz 掩码的**位置**由 `base + max(0,shift)` / `base - min(0,shift)` 决定 "
                "⇒ 这直接决定**恒等区间**（`03` 篇 §2 就是从这个式子推的）。"
                "★ **`class Swizzle:` 没有基类** —— 这正是陷阱 A 的根源（锚点 `P8` 消费它）。",
        "covered_by": ("A.1", "A.2", "A.3", "C.7"),
    },
    "S3": {
        "file": "python/pycute/swizzle.py", "line": (73, 82), "level": "B",
        "quote": "  def __call__(self, offset):",
        "must_contain": ("def __call__(self, offset):",
                         "return offset ^ shiftr(offset & self.yyy_msk, self.shift)",
                         "def size(self):", "return 1 << (self.bits + self.base + abs(self.shift))",
                         "def cosize(self):", "return self.size()"),
        "note": "★ `__call__` 是**一行**：`offset ^ shiftr(offset & yyy_msk, shift)`。"
                "`A.1` 用本仓库自己的位运算实现逐点对拍它。"
                "★ `size()` = `1 << (bits+base+|shift|)`、`cosize()` **就是** `size()` "
                "⇒ 这就是 `cosize(Swizzle)` 不失效的原因（锚点 `P9`）。",
        "covered_by": ("A.1", "A.2", "C.6b"),
    },
    "S4": {
        "file": "python/pycute/swizzle.py", "line": (93, 129), "level": "B",
        "quote": "class ComposedLayout(LayoutBase):",
        "must_contain": ("class ComposedLayout(LayoutBase):",
                         "def __init__(self, layoutB, offset, layoutA):",
                         "return self.layoutB(self.offset + self.layoutA(*args))",
                         "return f\"{self.layoutB} o {self.offset} o {self.layoutA}\""),
        "note": "★★ **`ComposedLayout` 是 `LayoutBase` 的子类，而 `Swizzle` 不是** —— "
                "同一段代码里两个类的基类不同，这是陷阱 A 的直接对照。"
                "★★ 三段复合的语义**逐字**是 `layoutB(offset + layoutA(crd))`；"
                "`A.4` 用**本仓库自己的**三段复合公式逐点对拍它（两条不同路径）。",
        "covered_by": ("A.4", "C.6"),
    },
    "S5": {
        "file": "python/pycute/swizzle.py", "line": 37, "level": "B",
        "quote": "from .layout import *",
        "must_contain": ("from .layout import *",),
        "note": "★ `swizzle.py` **只从 `layout.py` 里拿名字**，它自己没有 `size()` 这个模块级函数 —— "
                "所以 `Swizzle` 身上那个 `size()` 是**方法**，不是模块函数。"
                "★ 两个同名 `size` 的**全部差别**在锚点 `P8` 与 `P9`。",
        "covered_by": ("C.6",),
    },
    "S6": {
        "file": "python/pycute/swizzle.py", "line": (115, 121), "level": "B",
        "quote": "  # size(layout)   Size of the domain",
        "must_contain": ("return size(self.layoutA)", "return cosize(self.layoutB)"),
        "note": "★ `ComposedLayout.size()` 调的是**模块级** `size()`（即 `P8` 那个有分支的函数），"
                "`cosize()` 调的是模块级 `cosize()`。"
                "⇒ 对一个 `ComposedLayout(swizzle, …, layout)`，`size()` 看的是 **layoutA** 那一侧，"
                "**不经过 Swizzle** —— 所以陷阱 A 在 ComposedLayout 上不出现。",
        "covered_by": ("A.4",),
    },
}

# ===========================================================================
# 5. `cutlass_library` 的枚举与形状表（B 级 + 本机实测计数）
# ===========================================================================
LIB_MODULES = (
    "cutlass_library", "cutlass_library.library", "cutlass_library.generator",
    "cutlass_library.gemm_operation", "cutlass_library.manifest",
    "cutlass_library.heuristics", "cutlass_library.heuristics_provider",
    "cutlass_library.sm90_shapes", "cutlass_library.sm90_utils",
    "cutlass_library.sm100_shapes", "cutlass_library.sm100_utils",
    "cutlass_library.emit_kernel_listing",
)
# ★ 12 个 cutlass_library 子模块 + 1 个 pycute，本机默认 python3（3.13.5）下**全部导入成功**。

ENUM_DEFS = (
    # (枚举名, 定义行, 实测取值数, 备注)
    ("OpcodeClass", 1137, 6, "Simt / TensorOp / WmmaTensorOp / SparseTensorOp / "
                             "BlockScaledTensorOp / BlockScaledSparseTensorOp"),
    ("DataType", 69, 47, "含 e4m3 e5m2 e2m1 f4 f6 ue8m0 ue4m3 tf32 bf16；invalid 是哨兵"),
    ("LayoutType", 401, 20, "含 ColumnMajor / RowMajor / *Interleaved2,32,64 / TensorNHWC"),
    ("GemmKind", 1234, 16, "含 BlockScaledMoeGroupedUniversal3x（MoE + block-scale 合成一个 kind）"),
    ("GemmUniversalMode", 1624, 4, "Gemm / GemmSplitKParallel / Batched / Array"),
    ("MathOperation", 368, 12, "含 multiply_add_fast_accum（即 fp16 快累加）"),
)

# ★ `procedural_name()` 的 sm90 断点，**六行逐字**（同一对象只改 min_compute；
#   stages=3、cluster_shape=[2,1,1]）。规划阶段（2026-09-20）实测，本程序重跑比对。
#   ★★ 两行必须来自**同一个 tbs** —— `else` 分支的第三个数就是 threadblock_shape[2]。
TILE_NAME_CASES = (
    # (threadblock_shape, stages, cluster_shape, min_compute, 期望的 procedural_name)
    ((128, 128, 32), 3, (2, 1, 1), 75, "128x128_32x3"),
    ((128, 128, 32), 3, (2, 1, 1), 80, "128x128_32x3"),
    ((128, 128, 32), 3, (2, 1, 1), 90, "128x128x32_2x1x1_3"),
    ((128, 128, 64), 3, (2, 1, 1), 75, "128x128_64x3"),
    ((128, 128, 64), 3, (2, 1, 1), 80, "128x128_64x3"),
    ((128, 128, 64), 3, (2, 1, 1), 90, "128x128x64_2x1x1_3"),
)

# ★ 全局档位的四级解码（`L12`）。实测：level=29 ⇒ wgmma 9 / mma 2 / cluster 0 / pruning 0。
GLOBAL_LEVEL_DECODE_CASES = (
    (0, 0, 0, 0, 0), (1, 1, 0, 0, 0), (9, 9, 0, 0, 0),
    (10, 0, 1, 0, 0), (20, 0, 2, 0, 0), (29, 9, 2, 0, 0),
    (100, 0, 0, 1, 0), (101, 1, 0, 1, 0), (1000, 0, 0, 0, 1),
    #     level, wgmma(个位), mma(十位), cluster(百位), pruning(千位)
)

# ★ 生成数的实测（fp16/bf16 一族，is_aligned=True，global level 见下）：
#   level 0 → 3 tds / level 10 → 6 / level 20 → 9 / level 30 → 12
#   （3 个 math instruction 每个 shape 一份：f16×f16→f16 / f16×f16→f32 / bf16×bf16→f32）
GEN_COUNT_CASES = ((0, 3), (10, 6), (20, 9), (30, 12))

# ★ SM100 的 1SM/2SM 成对（`L11`；**限定在 MXF4NVF4 一族**）
SM100_MXF4NVF4_1SM = {(128, 64, 64): 1, (128, 128, 64): 0, (128, 192, 64): 1, (128, 256, 64): 0}
SM100_MXF4NVF4_2SM = {(256, 64, 64): 1, (256, 128, 64): 0, (256, 192, 64): 1, (256, 256, 64): 0}
# ★★ 限定语的反例（**本轮实测订正了计划 §0.7 C2 事实 10 的措辞**）：
#   计划写「`SM100_MMA_SHAPES_TF32_DENSE_1SM` 的 M 就是 64」。实测该表里
#   **M 有 64 与 128 两种**（是「**含** M=64 的形状」，不是「就是 64」）。
#   结论（不得把 MXF4NVF4 的配对推广成通则）不变，但**理由**必须按实测写。
SM100_TF32_1SM_MS = (64, 128)
SM100_MXF4NVF4_1SM_MS = (128,)

# ★ 泄漏的量（`L7`；实测于 fp16/bf16 一族 + `is_aligned=True`）：
#   · 一个 `threadblock_shape` 可以被**多于一组** (指令形状, 乘数) 对产生；
#   · 而发射出的 `TileDescription` 携带的 `instruction_shape` 恒为 `[0, 0, 0]`。
#   实测：level=39 时 tds=327，**唯一 procedural_name 只有 93 个**，最多的一个名字出现 **6 次**；
#         乘积集合 112 个 tbs 中，原像 > 1 的有 **16** 个（最大原像 2）。
#   例：tbs=(128, 16, 64) ← ((64,8,16),(2,2,4)) 与 ((64,16,16),(2,1,4)) 两组不同的 (指令, 乘数)。
SM90_LEAK_LEVEL = 39
SM90_LEAK_TDS = 327
SM90_LEAK_UNIQUE_NAMES = 93
SM90_LEAK_MAX_NAME_MULT = 6
SM90_LEAK_MULTI_PREIMAGE = 16
SM90_LEAK_EXAMPLE_TBS = (128, 16, 64)
SM90_LEAK_EXAMPLE_PREIMAGES = (((64, 8, 16), (2, 2, 4)), ((64, 16, 16), (2, 1, 4)))

# ★ 架构名表到 90 为止；同一棵树里 `include/cute/arch/` 的实际内容（`B.12`）。
#   ★ 目录里**没有** `mma_sm107.hpp` —— 只有 `mma_sm107_desc.hpp` 与 `mma_sm107_umma.hpp`。
ARCH_NAMES_MAX = 90
CUTE_ARCH_DIR = "include/cute/arch"
CUTE_ARCH_PRESENT = ("mma_sm100.hpp", "mma_sm100_umma.hpp", "copy_sm100.hpp",
                     "mma_sm107_desc.hpp", "mma_sm107_umma.hpp",
                     "mma_sm120.hpp", "mma_sm120_sparse.hpp")
CUTE_ARCH_ABSENT = ("mma_sm107.hpp",)
CUTE_ARCH_FILE_COUNT = 35

# ===========================================================================
# 6. 两条陷阱（逐条带 `file:line`，故可进本模块）
# ===========================================================================
TRAPS = {
    "A": {
        "what": "`size(swizzle)` 静默返回 Swizzle 自身",
        "file": "python/pycute/layout.py", "line": (125, 128),
        "also_file": "python/pycute/swizzle.py", "also_line": 60,
        "observed": ("isinstance(S, LayoutBase) = False；S.size() = 512；"
                     "size(S) 的字符串形式 = SW_3_3_3；**cosize(S) = 512（不失效）**"),
        "scope": "★ **只能限定在 `size()` 上**。`cosize()` **不失效**（锚点 `P9`）；"
                 "而且在 `Layout` 上方法 == 模块函数（32 == 32）。"
                 "**不得说成「pycute 的 size() 不可靠」**（P10）",
        "covered_by": ("C.6", "C.6b"),
    },
    "B": {
        "what": "`Swizzle(bits,base,shift)` 强制 `abs(shift) >= bits`",
        "file": "python/pycute/swizzle.py", "line": 64,
        "observed": "`Swizzle(4,3,3)` 直接 `AssertionError`（因 3 >= 4 不成立）",
        "scope": "★ 「4 位掩码、3 位基、3 位移位」在纸面上看起来完全合理 —— "
                 "这条断言是**构造式**强制的，不是文档建议",
        "covered_by": ("C.7",),
    },
    "C": {
        "what": "`composition(swizzle, layout)` 抛 `AttributeError`",
        "file": "python/pycute/layout.py", "line": (190, 228),
        "observed": "AttributeError: 'Swizzle' object has no attribute 'shape'",
        "scope": "★ swizzle 要用 `ComposedLayout(swizzle, offset, layout)` 表达"
                 "（锚点 `S4`）——它不是「另一种写法」，是**唯一**能参与 composition 的写法",
        "covered_by": ("C.8",),
    },
}

# ===========================================================================
# 7. 找不到可定位一手来源的项（**只出现在这里**，正文与常量名里都不出现）
# ===========================================================================
UNSOURCED = (
    # (项, 为什么不用它)
    ("CUTLASS 对 layout 代数的规范文本", "CUTLASS 没有为 CuTe 的 layout 代数出过规范；"
     "本主题因此**一条 A 级来源都没有**，全部按 B 级（NVIDIA 的实现）标注"),
    ("`pycute` 的公开文档页", "`cutlass/python/README.md` 通篇不提 `pycute`（0 次），"
     "顶层 `pyproject.toml` 也不含 `pycute`（0 次）⇒ 引它**只能引源码路径**"),
)

# ===========================================================================
# 8. 外部工件路径探测（`Makefile` 与本模块共用一份候选列表）
# ===========================================================================
# ★ 规定：**不得写死构造者的目录布局**（`CONTRIBUTING` 第 140 行）。
#   候选顺序：环境变量 → 相对本目录的兄弟目录 → 家目录下的常见位置。
#   找到后**打印选中路径**；找不到**以可操作的错误退出**（rc 非 0），**不静默跳过对拍**。
CUTLASS_ENV = "CUTLASS_PATH"
CUTLASS_CANDIDATES = (
    "../../../../cutlass",      # knowledge/30-gpu-programming/cutlass/code → 上溯四级的兄弟目录
    "../../../cutlass",
    "../../cutlass",
    "../cutlass",
    "~/cutlass",
    "~/src/cutlass",
)

# ★★ 相对候选的**解析基准** = `specs.py` 自己所在的目录。**不是** `os.getcwd()`。
#
#   为什么这一条是承重的（本机实测，修复前）：
#
#     cd <本目录>            && python3 02_tiling_decompose.py  → rc=0（选中 /…/CUDA/cutlass）
#     cd <仓库根 knowledge/> && python3 …/code/02_tiling_decompose.py → rc=0  ← 同一个答案
#     cd /tmp                && python3 …/code/02_tiling_decompose.py → rc=2（一个候选都没命中）
#
#   中间那行之所以「碰巧也对」，是因为从仓库根看 `../cutlass` 恰好落在
#   `/Users/cachio/gitProjects/CUDA/cutlass` —— **本机目录布局的巧合**，
#   换一台机器、换一个检出位置就没了。这正是 `CONTRIBUTING` 第 140 行禁止的
#   「工作靠的是构造者本机的目录布局」。⇒ 基准钉死到**本文件所在目录**，
#   于是「同一个候选永远解析到同一个地方」与调用者的 cwd 无关。
CUTLASS_CANDIDATE_BASE = os.path.dirname(os.path.abspath(__file__))
# ★ 这个 `abspath` **不会**读 cwd：`__file__` 在 python ≥ 3.9 恒为**绝对路径**
#   （`Makefile` 的 `check-env` 就要求 ≥ 3.9），`abspath` 对绝对路径是恒等变换。
#   它只在 **import 时**算一次，之后 `find_cutlass()` 全程不碰 `os.getcwd()`。
CUTLASS_REQUIRED = ("python/pycute/__init__.py",
                    "python/pycute/layout.py",
                    "python/pycute/swizzle.py",
                    "python/cutlass_library/library.py",
                    "include/cutlass/version.h")

# ★ 探测结果（由 `require_cutlass()` 写入）。**初始为 None**：
#   本模块在 import 时**不做任何文件系统探测**，也不打印任何东西。
CUTLASS_ROOT = None


def is_cutlass_root(path):
    """判定一个目录是不是可用的 CUTLASS 源码树（**全部**必需文件都在才算）。"""
    if not path or not os.path.isdir(path):
        return False
    return all(os.path.isfile(os.path.join(path, f)) for f in CUTLASS_REQUIRED)


def resolve_candidate(c):
    """把一个**候选**解析成绝对路径：`~` 展开，**相对项以本目录为基准**。

    ★ 这里**不碰 `os.getcwd()`**（不用 `os.path.abspath`）——
      用了它，「同一个候选」在不同 cwd 下就会解析到不同地方（见上面的实测）。
      `meta_falsify.py` 把这一条做成可证伪的：把 `os.getcwd` 换成会抛异常的桩，
      再调 `find_cutlass()`，**不得**有任何异常逃出来。
    ★ `~/cutlass` 是**允许**的：它展开成**读者自己**的家目录，不是构造者的绝对路径
      （判据是「字面量是不是绝对路径」，不是「展开后是不是绝对路径」）。
    """
    c = os.path.expanduser(c)
    if os.path.isabs(c):
        return os.path.normpath(c)
    return os.path.normpath(os.path.join(CUTLASS_CANDIDATE_BASE, c))


def resolve_env_path(v):
    """把 `CUTLASS_PATH` 的值解析成绝对路径：**相对值按调用者的 cwd 解析**。

    ★ 与候选列表**故意不同**：那是用户当场敲进去（或 `make run CUTLASS_PATH=…` 传进来）
      的路径，语义就是「相对我现在所在的地方」——它是**输入**，不是本库写死的布局假设，
      所以不适用「不得写死构造者的目录布局」那一条。
    ★ 想让行为与 cwd 完全无关，就传**绝对路径**（错误提示里给的就是绝对路径形态）。
    """
    return os.path.normpath(os.path.abspath(os.path.expanduser(v)))


def find_cutlass(verbose=False):
    """按候选列表探测 CUTLASS 源码树，返回第一个命中的**绝对路径**；找不到返回 None。

    ★ 与 `Makefile` 的探测是**同一份候选列表**（都从这里读）。
    ★ **相对候选以 `CUTLASS_CANDIDATE_BASE`（本目录）为基准，与调用者的 cwd 无关**：
      于是「任意 cwd 下调用」的行为一致（要么都命中同一棵树，要么都以 rc=2 退出）。
    ★ 命中即**打印它选中的路径**（无声命中等于没说清用的是哪棵树）。
    """
    tried = []
    env = os.environ.get(CUTLASS_ENV, "")
    env_rel = bool(env) and not os.path.isabs(os.path.expanduser(env))
    # (标签, 解析后的绝对路径) —— 顺序即优先级：env 第一顺位，然后是候选列表。
    plan = ([(CUTLASS_ENV, resolve_env_path(env))] if env else [])
    plan += [("候选", resolve_candidate(c)) for c in CUTLASS_CANDIDATES if c]
    for label, p in plan:
        tried.append(p)
        if is_cutlass_root(p):
            if verbose:
                print("  探测        : %s=%r" % (CUTLASS_ENV, env))
                print("  选中路径    : %s" % p)
                print("  （来自 %s，在候选列表里排在第 %d 位）"
                      % ("环境变量" if label == CUTLASS_ENV else "候选列表的兄弟目录",
                         len(tried)))
                if env_rel and label != CUTLASS_ENV:
                    print("    ★ 注意：`%s=%r` 是**相对值** ⇒ 按**你当前的 cwd** 没命中；"
                          % (CUTLASS_ENV, env))
                    print("      选中这棵树的是**候选列表**（与 cwd 无关）。"
                          "要指定你自己的那棵就写**绝对路径**。")
            return p
    if verbose:
        print("  探测        : %s=%r" % (CUTLASS_ENV, env))
        print("  解析基准    : %s（**相对候选以它为基准**，与 cwd 无关）"
              % CUTLASS_CANDIDATE_BASE)
        if env_rel:
            print("    ★ `%s` 这次给的是**相对值** ⇒ 上面那一条是按**你当前的 cwd** 解析的"
                  "（不是本目录）：" % CUTLASS_ENV)
            print("      那是**你当场敲的路径**，不属于「写死构造者的目录布局」；"
                  "要完全与 cwd 无关就给**绝对路径**。")
        for t in tried:
            print("    未命中    : %s" % t)
    return None


def invariant_rows(cands=None, required=None):
    """对候选列表 / 必需文件表做**可机械判定**的性质检查。

    返回 `[(标签, 观测值, 期望值), …]`，调用方逐行比对（与 `meta_falsify` 的其它自检同形）。

    ★ 三条都来自 `CONTRIBUTING` 第 140 行「不得写死构造者本机的目录布局」，
      是**该规定对这两张表的实际要求**（不是附会上去的）：
        · ① 候选里出现**绝对路径** ⇒ 那张表只对构造者的机器成立；
        · ② `CUTLASS_REQUIRED` 出现**绝对路径** ⇒ `os.path.join(root, f)` 会**丢掉 root**
          （join 的第二项是绝对路径时第一项被丢弃）⇒ `is_cutlass_root()` 与传进来的
          那棵树**无关**，「本机碰巧有那个文件」就成了判据；
        · ③ 候选里必须**至少有一条相对本目录**的（`~` 家目录项不算）——
          没有它，那张表就只剩「**读者自己**家目录里的副本」，再也表达不出
          「**本仓库旁边的那棵树**」这个位置（本主题的默认共享方式），
          而这一条**不需要**先定位仓库根（`~` 与「相对本目录」是两种可分辨的写法）。
    ★ 三张（可传参的）表都能由调用方换成**合成的坏表**，用来证明这些检查**有区分力**
      （恒真的检查不是判据，只是一句声明）。
    """
    cands = CUTLASS_CANDIDATES if cands is None else tuple(cands)
    required = CUTLASS_REQUIRED if required is None else tuple(required)
    bad_c = [c for c in cands if os.path.isabs(c)]
    bad_r = [f for f in required if os.path.isabs(f)]
    has_rel = bool([c for c in cands if not c.startswith("~") and not os.path.isabs(c)])
    return [
        ("① `CUTLASS_CANDIDATES` 里不得含绝对路径", bad_c, []),
        ("② `CUTLASS_REQUIRED` 里每项必须是相对路径", bad_r, []),
        ("③ 候选里至少有一条**相对本目录**的（不是 `~` 家目录项）", has_rel, True),
    ]


def require_cutlass(prog):
    """找不到就**以可操作的错误退出**（rc=2），**绝不静默回落**。

    ★ 「静默跳过对拍」会把 A 类断言变成**空操作** —— 那是本域最不能接受的一种通过。
    ★ 命中后把路径写进模块级 `CUTLASS_ROOT`，供后续的 `read_lines` 等使用。
    """
    global CUTLASS_ROOT
    root = find_cutlass(verbose=True)
    if root is None:
        print()
        print("!! 找不到 CUTLASS 源码树。**本程序不做任何对拍就退出**（不静默跳过）。")
        print("   本主题的真值来源是 CUTLASS 4.8.0 自带的 Python 参考实现：")
        print("     python/pycute/          —— CuTe layout 代数的纯 Python 实现")
        print("     python/cutlass_library/ —— CUTLASS 自己的生成器")
        print("   需要下列文件存在（相对 CUTLASS 根）：")
        for f in CUTLASS_REQUIRED:
            print("     %s" % f)
        print()
        print("   请显式指定（两种方式任选）：")
        print("     make run %s=/path/to/cutlass" % CUTLASS_ENV)
        print("     export %s=/path/to/cutlass" % CUTLASS_ENV)
        print("   （★ `%s` 给**相对值**时按**你当前的 cwd** 解析 —— 那是你当场敲的路径；"
              % CUTLASS_ENV)
        print("     要完全与 cwd 无关就给绝对路径。下面的**候选**不受 cwd 影响。）")
        print("   或把 CUTLASS 源码树放在下列任一位置（**相对项以本目录为基准**：")
        print("     %s）—— 与你在哪个 cwd 调用**无关**：" % CUTLASS_CANDIDATE_BASE)
        for c in CUTLASS_CANDIDATES:
            print("     %-20s ⇒ %s" % (c, resolve_candidate(c)))
        print()
        print("   ★ 本库**不复刻** CUTLASS 的源码树，也不内置它的副本。")
        sys.exit(2)
    CUTLASS_ROOT = root
    return root


def load_pycute(root):
    """把 `<root>/python` 加进 `sys.path` 后 `import pycute`，返回模块。

    ★ 这是**真的 import 并跑 NVIDIA 自己的实现** —— 按 `CONTRIBUTING` 第 63 行的四类，
      产出的数字是**本机实测值**，必须写明测的是哪个实现哪个版本。
    """
    py_dir = os.path.join(root, "python")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    import pycute
    return pycute


def load_cutlass_library(root):
    """把 `<root>/python` 加进 `sys.path` 后返回 `cutlass_library.library` 模块等。"""
    py_dir = os.path.join(root, "python")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    from cutlass_library import library, sm90_utils, sm90_shapes, sm100_shapes
    return library, sm90_utils, sm90_shapes, sm100_shapes


def read_lines(root, relpath):
    """读外部源码的**原始字节**并按行返回（`\\n` 切分）。

    ★ 用 bytes 而不是先转成文本再数（计划 §2「本域特有的两条」第 3 条）：
      **转写层是最容易出错又最难自查的一层**。
    """
    with open(os.path.join(root, relpath), "rb") as f:
        return f.read().split(b"\n")
