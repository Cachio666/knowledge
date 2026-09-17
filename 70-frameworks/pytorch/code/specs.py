"""PyTorch 的外部事实源：版本锚点、官方文案、源码位置、本机实测基线表。

本文件只放两类东西：
  1. **从一手来源抄来的**事实（本机 torch 安装里的 docstring / C++ 头文件 / Python 源码行），
     每条带一行 `# 来源：`；
  2. **本机实测基线表**（与文档 §5 逐行一致的数字），每条注明测它的版本与测量条件。

★ 核查范围（错误模式 P6 的对策）：逐条写明核的是**哪个载体**——docstring / 头文件行 /
  Python 源码行 / 实跑输出。**不写「全库」「已核实全部」这类大于实际核查范围的措辞。**
  `torch/csrc/**/*.cpp` **不随 wheel 分发**，所以 dispatcher 的 C++ 实现本体读不到；
  可读的只有 `include/c10/**/*.h`（定义层）+ Python 层 + 实测行为。

★ 版本坐标（★ 本项目最重要的约束）：
  本机只有 2.6.0 / 2.7.1 / 2.13.0 三个解释器，**跨了 7 个 minor**——
  **本机实际有五个装了 torch 的解释器**（2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）——
  计划只列了三个，漏了 base conda 那个（见 MEASURED_VERSIONS 的订正说明）。
  **2.9 / 2.10 / 2.12 本机没有解释器，这四个 minor 不声称覆盖。**
  凡引「行号」一律是 **2.13.0 的行号**（2.6/2.7 的行号不同，见 VERSION_LINE_SHIFT）。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · 「Dynamo 的一次 guard 校验耗时 xx 微秒」——本机只能测「重编译发生了几次」，
    **测不到 guard 的逐次开销**（这是编译器的内部计数，不是可分离的计时对象）。
  · 「`torch._C._dispatch_*` 函数一共 67 个」——计划里给过这个数，
    本机按 `"dispatch" in name` 数出来是 **68 / 68 / 70**（2.6.0 / 2.7.1 / 2.13.0），
    **数不出 67 这个口径**，故正文改用本程序实际数出的口径与数字。
"""

# ---------------------------------------------------------------------------
# 版本锚点
# ---------------------------------------------------------------------------

# 本机已安装并可 import torch 的三个解释器（探测顺序见 code/Makefile）
# 来源：本机实测 `import torch; torch.__version__`（2026-09-17）
# ★★ 订正（2026-09-17 实施时实测，两次）：本机其实有**五个**装了 torch 的解释器——
#   计划 §0.4 只列了三个（它只扫了 `envs/*/bin/python`），漏掉的是：
#     · `/Users/cachio/miniconda3/bin/python`（base conda）  → torch **2.11.0**
#     · `/usr/bin/python3`（macOS 自带 py3.9，用户 site-packages）→ torch **2.8.0**
#   后果：计划里「2.8–2.12 本机测不到」是**错的**——2.8.0 与 2.11.0 本机都测得到。
#   真正没有解释器的只有 **2.9 / 2.10 / 2.12** 三个 minor。
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")

# ★ 本机确实没有解释器的三个 minor（不得声称覆盖）
UNCOVERED_MINORS = "2.9.x / 2.10.x / 2.12.x"

# 计划 §0.7.A 的全部行号引用都是 2.13.0 的行号；本表给出同名位置的实测行号差异
# 来源：本机读源码（`sed -n '<行号>p'`），三个版本逐条打印核对（2026-09-17）
VERSION_LINE_SHIFT = {
    "torch/_tensor.py: __torch_dispatch__ = _C._disabled_torch_dispatch_impl": {
        "2.13.0": 1536, "2.7.1": 1674, "2.6.0": 1654},
    "torch/autograd/__init__.py: retain_graph = create_graph": {
        "2.13.0": 390, "2.7.1": 348, "2.6.0": 342},
    "c10/core/DispatchKeySet.h: uint64_t repr_ = 0;": {
        "2.13.0": 513, "2.7.1": 510, "2.6.0": 510},   # 2.11.0 未逐行核
    "c10/core/DispatchKey.h: enum class DispatchKey : uint16_t": {
        "2.13.0": 137, "2.7.1": 135, "2.6.0": 135},
}

# 来源：本机 `torch/include/c10/core/DispatchKey.h`，grep "BELOW Python"
# 实测：**只有 2.13.0 有**（2.6.0 / 2.7.1 里 0 命中；2.6.0 同一位置是另一条讲 torchdistx 的注释）
# ⚠ 这条注释讲的是 **C++ 的 `Fake` 这个 dispatch key 在枚举里的次序**，
#   不是「TorchDispatchMode 优先于子类 __torch_dispatch__」的判据。
#   本项目对「mode 优先」的**唯一依据是实测**（见 MEASURED_DISPATCH_ORDER）。
FAKE_KEY_COMMENT_2_13_ONLY = (
    "Fake dispatch key for C++ FakeTensor mode. Must be BELOW Python so that "
    "TorchDispatchModes (e.g. ProxyTorchDispatchMode, FakeTensorMode) fire "
    "first, matching the Python FakeTensor dispatch order."
)

# ---------------------------------------------------------------------------
# dispatch（文档 01）
# ---------------------------------------------------------------------------

# 来源：本机 `torch.Tensor.view.__doc__`（逐字，三个版本一致）
# 判据名「contiguity-like condition」出自这句原文；文档 03 §1 直接引它
VIEW_DOC_CONTIGUITY_LIKE = (
    "For a tensor to be viewed, the new view size must be compatible with its "
    "original size and stride, i.e., each new view dimension must either be a "
    "subspace of an original dimension, or only span across original dimensions "
    "d, d+1, ..., d+k that satisfy the following contiguity-like condition that "
    "for all i = d, ..., d+k-1, stride[i] = stride[i+1] * size[i+1]"
)

# 来源：本机 `torch.Tensor.reshape.__doc__`（逐字）
RESHAPE_DOC = (
    "This method returns a view if shape is compatible with the current shape. "
    "See torch.Tensor.view on when it is possible to return a view."
)

# 来源：本机 `torch.Tensor.is_leaf.__doc__`（逐字，三个版本一致）
# ⚠ T7：官方**恰恰用了**「created by the user」这个口语表述；
#   形式判据是 `grad_fn is None or not requires_grad`（见 verify.py 的独立实现）
IS_LEAF_DOC = (
    "All Tensors that have requires_grad which is False will be leaf Tensors by "
    "convention. For Tensors that have requires_grad which is True, they will be "
    "leaf Tensors if they were created by the user. This means that they are not "
    "the result of an operation and so grad_fn is None."
)

# 来源：本机 `torch.inference_mode.__doc__`
# 2.7.1 逐字：「entering inference_mode also disables to forward-mode AD」
# 2.13.0 **把这句删掉了**，改为：
INFERENCE_MODE_DOC_2_13 = (
    "Compared to no_grad, it removes additional overhead by disabling view "
    "tracking and version counter bumps. It is also more restrictive, in that "
    "tensors created in this mode cannot be used in computations recorded by "
    "autograd."
)

# 来源：本机实跑 `torch._dynamo.utils.counters`（**不是官方文案，是本机观测**）
# 见 DYN_GUARD_BASELINE

# 版本差异：`torch.library` 的公开面（本机 `dir(torch.library)` 实测，五版本）
# ★ 两处订正（实测推翻计划）：
#   ① 泄漏的类型名**不是 2.13 才清的**：2.11.0 就已经是 Any/Callable/Sequence/TypeVar/Union，
#      与 2.13.0 完全相同——`Optional` 的清理发生在 2.8 与 2.11 之间（2.7/2.8 都还有它），
#      而 `Dict/List/Set/Tuple` 的清理发生在 2.6 与 2.7 之间。**没有全清**。
#   ② get_kernel / EffectType **不是 2.13 新增**：2.11.0 上就有了，
#      2.8.0 上还没有 ⇒ 真实区间是 **2.9–2.11**（2.9/2.10 本机无解释器）。
#      ⇒ 可用面判据应当用**特征探测**（hasattr），不要拿版号当门牌。
LIBRARY_PUBLIC_SURFACE = {
    "2.6.0": ("custom_op", "define", "impl", "register_fake", "impl_abstract",
              "infer_schema", "Library", "triton_op"),
    "2.7.1": ("custom_op", "define", "impl", "register_fake", "impl_abstract",
              "overload", "register_autocast", "infer_schema", "Library", "triton_op"),
    "2.8.0": ("custom_op", "define", "impl", "register_fake", "impl_abstract",
              "overload", "register_autocast", "infer_schema", "Library", "triton_op"),
    "2.11.0": ("custom_op", "define", "impl", "register_fake", "impl_abstract",
               "get_kernel", "overload", "register_autocast", "EffectType",
               "infer_schema", "Library", "triton_op"),
    "2.13.0": ("custom_op", "define", "impl", "register_fake", "impl_abstract",
               "get_kernel", "overload", "register_autocast", "EffectType",
               "infer_schema", "Library", "triton_op"),
}
# 来源：本机 `dir(torch.library)` 的类型名泄漏（逐版本实测）
LIBRARY_LEAKED_TYPENAMES = {
    "2.6.0": ("Any", "Callable", "Dict", "List", "Optional", "Sequence", "Set", "Tuple", "Union"),
    "2.7.1": ("Any", "Callable", "Optional", "Sequence", "TypeVar", "Union"),
    "2.8.0": ("Any", "Callable", "Optional", "Sequence", "TypeVar", "Union"),
    "2.11.0": ("Any", "Callable", "Sequence", "TypeVar", "Union"),
    "2.13.0": ("Any", "Callable", "Sequence", "TypeVar", "Union"),
}

# 来源：本机 `torch.library.define` / `impl` 的 `inspect.signature`（三版一致）
SIG_DEFINE = "(qualname, schema, *, lib=None, tags=())"
SIG_IMPL = "(qualname, types, func=None, *, lib=None)"

# ---------------------------------------------------------------------------
# 本机实测基线表（与文档 §5 逐行一致）
# ---------------------------------------------------------------------------

# 测量条件：三个解释器各跑一遍 `04_custom_ops.py --probe`（2026-09-17），
# 语义见 04 号程序的 §B；本表是**真 torch 实测**（版本见 KEY），不是推导。
#
# 表 1：一次表达式触发的 `__torch_dispatch__` 拦截次数
# 观测手段：TorchDispatchMode（Python key）+ profiler（C++ RecordFunction 层）两条独立通道
DISPATCH_BASELINE = [
    # (表达式, mode 拦截数, mode 观察到的 op, profiler 观察到的 aten op)
    ("a + b",              1, "aten.add.Tensor", "aten::add"),
    ("torch.add(a, b)",    1, "aten.add.Tensor", "aten::add"),
    ("a.add(b)",           1, "aten.add.Tensor", "aten::add"),
    ("(ag + b).sum()",     2, "aten.add.Tensor, aten.sum.default", "aten::add, aten::sum"),
    ("a.sum()",            1, "aten.sum.default", "aten::sum"),
    ("a.T",                1, "aten.permute.default", "aten::permute, aten::as_strided"),
    ("a[0]",               1, "aten.select.int", "aten::select, aten::as_strided"),
    ("一维 v @ w",          1, "aten.dot.default", "aten::matmul, aten::dot"),
    ("二维 A @ B",          1, "aten.mm.default", "aten::matmul, aten::mm"),
    ("+a",                 0, "-", "aten::positive"),
    ("a.float()（同 dtype）", 0, "-", "aten::to"),
    ("a.shape",            0, "-", "-"),
    ("torch.Tensor.dim(a)", 0, "-", "-"),
]

# 表 2：CompositeImplicitAutograd 的算子对 dispatch mode 不可见（C1）
# 观测手段：TorchDispatchMode 拦截 + `torch._C._dispatch_dump_table` 的注册层级
COMPOSITE_INVISIBLE = [
    # (调用的算子, mode 观察到的 op 序列, 注册层级（dump_table 首行关键词）)
    ("aten.reshape.default(非连续, [16])", "aten.clone.default, aten._unsafe_view.default", "math kernel"),
    ("aten.reshape.default(连续, [16])", "aten.view.default", "math kernel"),
    ("aten.ravel(非连续)", "aten.clone.default, aten.view.default", "math kernel"),
    ("aten.ravel(连续)", "aten.view.default", "math kernel"),
    ("aten.contiguous(非连续)", "aten.clone.default", "math kernel"),
    ("aten.contiguous(连续)", "（0 次）", "math kernel"),
    ("aten.relu(非连续)", "aten.relu.default", "kernel"),
    ("aten.sum.default(非连续)", "aten.sum.default", "kernel"),
]

# 表 3：mode 与子类 __torch_dispatch__ 的顺序（实测，三版一致）
# 观测：两者同时在栈上时，调用序列以 mode 开头
MEASURED_DISPATCH_ORDER = ["mode", "subclass"]

# ---------------------------------------------------------------------------
# autograd（文档 02）
# ---------------------------------------------------------------------------

# 来源：本机 `torch/autograd/__init__.py` 源码行（三版都有这一行，行号不同，见 VERSION_LINE_SHIFT）
RETAIN_GRAPH_SOURCE = "if retain_graph is None: retain_graph = create_graph"

# 来源：本机实跑 `torch.inference_mode` 下 `torch.func.jvp`（**四版**逐版本实测）
# ★ 四版**不一致**：2.6.0 / 2.7.1 / **2.11.0** 静默返回全零，2.13.0 正确。
# ⚠ 变更点**只能定位到 2.11 与 2.13 之间**——2.12 本机没有解释器，**不得声称是 2.12 改的**。
#   （2.8.0 实测也是「全零」，所以变更点被夹在 2.11 与 2.13 之间，不是 2.8。）
FORWARD_AD_IN_INFERENCE_MODE = {
    "2.6.0": "[0.0, 0.0, 0.0]（静默全零，不抛异常）",
    "2.7.1": "[0.0, 0.0, 0.0]（静默全零，不抛异常）",
    "2.8.0": "[0.0, 0.0, 0.0]（静默全零，不抛异常）",
    "2.11.0": "[0.0, 0.0, 0.0]（静默全零，不抛异常）",
    "2.13.0": "[2.0, 2.0, 2.0]（正确）",
}
# 关于分界的**假设**（可证伪）：变更发生在 2.12；本仓库没有 2.12 的解释器，
# 所以 02 号程序对不在上表里的版本只断言「实测值必须是上面两种形态之一」。
FORWARD_AD_CHANGE_BOUNDARY_HYPOTHESIS = "2.12（未实测，仅假设）"

# 来源：本机实跑，一次前向**保存的中间结果个数**（saved_tensors_hooks 的 pack 计数）
# ★ 没有通用数字——逐 op 不同；本表是三个算例的实测值
SAVED_TENSOR_BASELINE = [
    ("(x*x).sum()", 2),
    ("(A@B).sum()", 2),
    ("(q*2+1).relu().sum()", 1),
]

# 来源：本机实跑的错误原文（**逐版本措辞不同，注意**）
ERR_INFERENCE_SAVE_FOR_BWD = {
    "2.6.0": ("Inference tensors cannot be saved for backward. To work around you "
              "can make a clone to get a normal tensor and use it in autograd."),
    "2.13.0": ("Inference tensors cannot be saved for backward. Please do not use "
               "Tensors created in inference mode in computation tracked by autograd. "
               "To work around this, you can make a clone to get a normal tensor and "
               "use it in autograd, or use `torch.no_grad()`"),
}
ERR_INPLACE_OUTSIDE_INFERENCE = (
    "Inplace update to inference tensor outside InferenceMode is not allowed."
)
ERR_VERSION_ON_INFERENCE_TENSOR = "Inference tensors do not track version counter."

# ---------------------------------------------------------------------------
# 张量内存模型（文档 03）
# ---------------------------------------------------------------------------

# 来源：本机 `torch/include/c10/core/Contiguity.h`（2.13.0 文件，行号见 VERSION_LINE_SHIFT 之外的实测）
# 实测行号：`_compute_contiguous` 定义在 :15，`_compute_non_overlapping_and_dense` 在 :275
# ★ 结构性证据：签名里**没有 storage**，所以它不可能检查边界或重叠
CONTIGUOUS_SIGNATURE = "bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel)"
OVERLAP_SIGNATURE = "bool _compute_non_overlapping_and_dense(ArrayRef<T> sizes, ArrayRef<T> strides)"

# 来源：同上头文件 :15-35 的算法（逐字结构：先判 numel == 0，再倒序扫、跳过 size == 1）
# verify.py 的 `ref_compute_contiguous` 是这份算法的**独立重写**（不看 torch 的输出）
CONTIGUOUS_RULE = (
    "if numel == 0: return True; expected = 1; "
    "for d from last to first: if size[d] == 1: continue; "
    "if stride[d] != expected: return False; expected *= size[d]; return True"
)

# 来源：本机实跑 `torch.as_strided` 的报错原文（4 个 float32 元素的 storage 上要 (2,3)/(3,1)）
# ★ 后果：**as_strided 自带边界检查**，不能用它造「is_contiguous 不查边界」的反例
ERR_AS_STRIDED_OOB = (
    "setStorage: sizes [2, 3], strides [3, 1], storage offset 0, and itemsize 4 "
    "requiring a storage size of 24 are out of bounds for storage of size 16"
)
ERR_WRAPPER_NO_DISPATCH = "must define __torch_dispatch__"

# 来源：本机实跑 `Tensor.view` 失败的报错原文（`torch.randn(4,4).t().view(16)`）
ERR_VIEW_INCOMPATIBLE = (
    "view size is not compatible with input tensor's size and stride (at least one "
    "dimension spans across two contiguous subspaces). Use .reshape(...) instead."
)

# ---------------------------------------------------------------------------
# 自定义算子（文档 04）
# ---------------------------------------------------------------------------

# 来源：本机实跑（三版一致）
ERR_CUSTOM_OP_NO_MUTATES = "custom_op() missing 1 required keyword-only argument: 'mutates_args'"
ERR_CUSTOM_OP_NO_ANNOTATION = (
    "infer_schema(func): Parameter x must have a type annotation. Got func with signature (x)"
)
# impl_abstract → register_fake 的改名：**发的是 FutureWarning**（不是 DeprecationWarning），
# 2.6.0 / 2.7.1 / 2.13.0 三版都还在发
IMPL_ABSTRACT_RENAME_WARNING = (
    "`torch.library.impl_abstract` was renamed to `torch.library.register_fake`. "
    "Please use that instead; we will remove `torch.library.impl_abstract`"
)
IMPL_ABSTRACT_WARNING_CATEGORY = "FutureWarning"

# ---------------------------------------------------------------------------
# 图捕获的代价模型（文档 04 §3）：本机实测，非官方数字
# ---------------------------------------------------------------------------

# 测量条件：`torch.compile(fn, backend="eager")` + `torch._dynamo.utils.counters["stats"]`
# 三个版本实测 `unique_graphs` 完全一致（2026-09-17）
DYN_GUARD_BASELINE = [
    # (调用序列, unique_graphs, calls_captured)
    ("同 shape 调 3 次 (4,4)", 1, 3),
    ("再换 shape 到 (8,8) 调 1 次", 2, 6),
    ("再把 dtype 换成 float64 调 1 次", 3, 9),
]

# 来源：本机实跑 `torch._dynamo.explain`（0 graph break 的小函数）
DYN_EXPLAIN_OK = "Graph Count: 1 Graph Break Count: 0"

# 来源：本机 `hasattr(torch, "while_loop")` / `"while_loop" in torch.__all__`（四版一致）
# ★ 「PyTorch 没有 while 版结构化控制流」是错的：
#   torch.while_loop **存在**，docstring 逐字：「while_loop is a structured control flow
#   operator. It preserves the loop semantic across the torch.compile and torch.export.」
#   但**不在 torch.__all__**（prototype）；torch.cond 同样 prototype **但在 __all__ 里**
WHILE_LOOP_IN_ALL = False
COND_IN_ALL = True
