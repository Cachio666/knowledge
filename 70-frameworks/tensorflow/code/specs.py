"""TensorFlow 的外部事实源：每条带一行 `# 来源：` 与**可打开的 URL**。

★ 本主题的特殊约束（计划 §0.3）：**本机没有 TensorFlow**。
  所以本文件里**没有一条本机实测值**——本机能跑出来的数（迷你追踪器）只支撑机制结论，
  一律不入本文件，只出现在各编号程序与文档 §5 的「迷你实现实测」档里。

★ 版本坐标（钉死，且有取用日期）：
  · TF 本体钉 **v2.21.0**（tag），源码行号即该 tag 下的行号。
  · TF 文档取自 `tensorflow/docs` 仓库 `master`——**这是移动目标**，
    故本文件所有文档类引文钉在 commit
    `35e0922e059d7bc6d515a83e03a7494f0640c314`（取用于 2026-09-17，commit 日期 2026-07-09）。
    引用时须写明这一点，不得写成「TF 文档里就是这么说的」。
  · `tensorflow.org` 在调查环境被网络拦死，**未核对线上渲染版与仓库版本是否一致**。

UNSOURCED / NOT FOUND（查不到一手来源，**正文不使用**）：
  · **「`ConcreteFunction` can be thought of as a wrapper around a `tf.Graph`」**——
    计划 §0.7.B 把它写成官方逐字，**在 v2.21.0 里查不到**。
    实际查到的是 `tensorflow/python/types/core.py:170` 的
    「Base class for differentiable graph functions.」（见 CONCRETE_FUNCTION_DOC_2_21）。
  · **「那句 Deprecated 在**两处**文档里出现」**——本次只核实到**一处**
    （`site/en/guide/saved_model.ipynb`）；`tf.saved_model.save` 的源码 docstring 里没有这句
    （`save.py` 全文只有 1 次 `deprecated`，是一句关于 SaveableObject 的注释）。
    **正文只说「官方文档里逐字有这句」，不说「两处」。**
  · 「`tf.raw_ops` 的官方定义原文」——查不到：
    ① `create_op.md` 全文 **0 次**提及 `raw_ops`（本仓库 `grep -c` 核实）；
    ② **`tensorflow/python/ops/raw_ops.py` 在 v2.21.0 这个 tag 下不存在**——
       本仓库列过该目录（GitHub contents API，`ref=v2.21.0`），只有 `raw_ops_test.py`，
       没有 `raw_ops.py` ⇒ 生成物，定义原文随构建产生；
    ③ API 文档页在调查环境被网络拦死。
    **正文因此不写「官方推荐 `tf.raw_ops`」这类话。**
  · 「`tf.xla.experimental.compile` 的废弃起始版本 / 移除日期」——源码里第一个参数是 `None`，
    **官方没给日期**。正文只说「仍在但已废弃，且官方未给移除日期」。
  · 「`tf.print` 在 XLA 下会失败」——**未取到一手来源**。
    docstring 只给了通则（整函数必须可编译，否则 `InvalidArgumentError`）与
    一个进一步阅读的链接（`https://tensorflow.org/xla/known_issues`，本环境打不开）。
    故编号程序里「XLA 编不了 tf.print」是**本仓库迷你实现自己定的模型**，
    代码与文档都必须标明这一点，**不得写成 TF 的事实**。
"""

# ---------------------------------------------------------------------------
# 版本坐标
# ---------------------------------------------------------------------------

TF_VERSION = "2.21.0"                       # 源码 tag：v2.21.0
DOCS_REPO_COMMIT = "35e0922e059d7bc6d515a83e03a7494f0640c314"
DOCS_REPO_COMMIT_DATE = "2026-07-09"
DOCS_FETCH_DATE = "2026-09-17"

_SRC = "https://github.com/tensorflow/tensorflow/blob/v2.21.0/"
_DOC = "https://github.com/tensorflow/docs/blob/" + DOCS_REPO_COMMIT + "/site/en/"


def src(path, lines=None):
    url = _SRC + path
    if lines:
        url += "#L%d-L%d" % lines
    return url


def doc(path):
    return _DOC + path


# ---------------------------------------------------------------------------
# 一、追踪与重追踪（文档 01）
# ---------------------------------------------------------------------------

# 来源：site/en/guide/function.ipynb（§Tracing 一节）
# 载体：笔记本的 markdown 单元（本仓库用 json 解析后逐字抄出）
TRACETYPE_DISPATCH_VERBATIM = (
    "When called, a `tf.function` first evaluates the type of each input argument "
    "using the `tf.types.experimental.TraceType` of each argument. This is used to "
    "construct a `tf.types.experimental.FunctionType` describing the signature of the "
    "desired `ConcreteFunction`. We compare this `FunctionType` to the `FunctionType`s "
    "of existing `ConcreteFunction`s. If a matching `ConcreteFunction` is found, the "
    "call is dispatched to it. If no match is found, a new `ConcreteFunction` is traced "
    "for the desired `FunctionType`."
)

# 来源：同上（紧接上一段的子类型规则）
SHAPE_SUBTYPE_VERBATIM = (
    "If multiple matches are found, the most specific signature is chosen. Matching is "
    "done by [subtyping](https://en.wikipedia.org/wiki/Subtyping), much like normal "
    "function calls in C++ or Java, for instance. For example, `TensorShape([1, 2])` is "
    "a subtype of `TensorShape([None, None])` and so a call to the tf.function with "
    "`TensorShape([1, 2])` can be dispatched to the `ConcreteFunction` produced with "
    "`TensorShape([None, None])` but if a `ConcreteFunction` with `TensorShape([1, None])` "
    "also exists then it will be prioritized since it is more specific."
)

# 来源：同上（TraceType 规则的清单，逐字，只摘与本主题有关的三条）
TRACETYPE_RULES = {
    "Tensor": ("For `Tensor`, the type is parameterized by the `Tensor`'s `dtype` and "
               "`shape`; ranked shapes are a subtype of unranked shapes; fixed "
               "dimensions are a subtype of unknown dimensions"),
    "Variable": ("For `Variable`, the type is similar to `Tensor`, but also includes a "
                 "unique resource ID of the variable, necessary to correctly wire "
                 "control dependencies"),
    "Python 原生值": ("For Python primitive values, the type corresponds to the **value** "
                      "itself. For example, the `TraceType` of the value `3` is "
                      "`LiteralTraceType<3>`, not `int`."),
    "有序容器": ("For Python ordered containers such as `list` and `tuple`, etc., the type "
                 "is parameterized by the types of their elements; for example, the type "
                 "of `[1, 2]` is `ListTraceType<LiteralTraceType<1>, LiteralTraceType<2>>` "
                 "and the type for   `[2, 1]` is `ListTraceType<LiteralTraceType<2>, "
                 "LiteralTraceType<1>>` which is different."),
    "映射": ("For Python mappings such as `dict`, the type is also a mapping from the same "
             "keys but to the types of values instead of the actual values. … However, "
             "unlike ordered containers, `{1: 2, 3: 4}` and `{3: 4, 1: 2}` have equivalent "
             "types."),
}

# 来源：同上（自由变量那一节）
GLOBALS_NO_RETRACE_VERBATIM = (
    "Note: `TraceType` is based on the `tf.function` input parameters so changes to "
    "global and free variables alone will not create a new trace.")

# 来源：同上（§Controlling retracing / Use `reduce_retracing`）
REDUCE_RETRACING_VERBATIM = (
    "When `reduce_retracing` is enabled, `tf.function` automatically identifies "
    "supertypes of the input types it is observing and chooses to trace more generalized "
    "graphs automatically. It is less efficient than setting the `input_signature` "
    "directly but useful when many types need to be supported.")

# 来源：同上（Pass a fixed `input_signature` to `tf.function`）
INPUT_SIGNATURE_VERBATIM = (
    "This forces `tf.function` to constrain itself to only one "
    "`tf.types.experimental.FunctionType` composed of the types enumerated by the "
    "`input_signature`. Calls that cannot be dispatched to this `FunctionType` will throw "
    "an error.")

# 来源：同上（§Executing Python side effects）
SIDE_EFFECTS_VERBATIM = (
    "Side effects, like printing, appending to lists, and mutating globals, can behave "
    "unexpectedly inside a `tf.function`, sometimes executing twice or not all. They only "
    "happen the first time you call a `tf.function` with a set of inputs. Afterwards, the "
    "traced `tf.Graph` is reexecuted, without executing the Python code.")

# 来源：tensorflow/python/eager/polymorphic_function/polymorphic_function.py:110-111
# 载体：源码常量（**注意：手册里查不到这两个数**，它们只存在于源码）
FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY = 10
FREQUENT_TRACING_WARNING_THRESHOLD = 5
FREQUENT_TRACING_WARNING_MAX_WARNING_PER_DETECTOR = 2

# 来源：同文件 :157-167 的 logging.warning(...) 载荷（逐字；`{}` 依次为
#       （触发次数, 窗口内调用数, 函数名））
FREQUENT_TRACING_WARNING_TEXT = (
    "{} out of the last {} calls to {} triggered tf.function retracing. Tracing is "
    "expensive and the excessive number of tracings could be due to (1) creating "
    "@tf.function repeatedly in a loop, (2) passing tensors with different shapes, "
    "(3) passing Python objects instead of tensors. For (1), please define your "
    "@tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True "
    "option that can avoid unnecessary retracing. For (3), please refer to "
    "https://www.tensorflow.org/guide/function#controlling_retracing and "
    "https://www.tensorflow.org/api_docs/python/tf/function for  more details.")

# ★ 上面这段文案是本主题**唯一一条直接否证「TF 对 shape 不敏感」的一手证据**：
#   官方自己在告警里点名 `passing tensors with different shapes`。

# 来源：polymorphic_function.py:770 的 docstring（**guide 正文里没有这个 API**）
TRACING_COUNT_DOC = "Returns the number of times the function has been traced."

# 来源：同文件 :1068（方法体）+ 手册 §Inspecting traces 里的用法
PRETTY_PRINTED = "pretty_printed_concrete_signatures(self, verbose=True)"

# ---------------------------------------------------------------------------
# 二、层级（文档 01 §1）
# ---------------------------------------------------------------------------

# 来源：tensorflow/python/types/core.py:193-200（`PolymorphicFunction` 的 docstring，逐字）
POLYMORPHIC_FUNCTION_DOC = (
    "Base class for polymorphic graph functions.\n\n"
    "Graph functions are Python callable objects that dispatch calls to a TensorFlow "
    "graph. Polymorphic graph functions can be backed by multiple TF graphs, and "
    "automatically select the appropriate specialization based on the type of input they "
    "were called with. They may also create specializations on the fly if necessary, for "
    "example by tracing.")

# 来源：tensorflow/python/types/core.py:170-177（`ConcreteFunction` 的 docstring，逐字）
# ⚠ 计划 §0.7.B 声称官方说的是「can be thought of as a wrapper around a `tf.Graph`」——
#   **v2.21.0 里查不到这句**。下面是实际存在的文本。
CONCRETE_FUNCTION_DOC_2_21 = (
    "Base class for differentiable graph functions.\n\n"
    "A `ConcreteFunction` encapsulates the original graph function definition with "
    "support for differentiability under `tf.GradientTape` contexts. In the process, it "
    "may generate new graph functions (using the original) to efficiently perform "
    "forwards and backwards passes.")

# 来源：tensorflow/python/eager/polymorphic_function/atomic_function.py:66-76
# ★ v2.21.0 里 `ConcreteFunction` 与图之间**还夹着一层 `AtomicFunction`**
#   （`ConcreteFunction.inference_fn` 返回的就是它），图缓存在它的 `_cached_graph`。
ATOMIC_FUNCTION_DOC = (
    "A Python callable for functions in the TF Runtime.\n\n"
    "Provides core functionality for tf.function including:\n"
    "  - automatic lifecycle management of runtime functions\n"
    "  - structured inputs (including captures) and structured outputs\n"
    "  - calls from both eager and graph mode\n"
    "  - dependency tracking of children functions\n"
    "  - runtime error interpolation to identify user code stack traces\n"
    "  - control dependencies (including automatic)")

# 来源：tensorflow/c/c_api.h:1147 与 :1159（两个 C API 符号的声明）
# `FunctionDef` 是 proto 的序列化形态，**不是运行时对象**——进出运行时要过一次这两个函数。
FUNCTION_DEF_C_API = {
    "to_proto": "TF_CAPI_EXPORT extern void TF_FunctionToFunctionDef(TF_Function* func,",
    "from_proto": "TF_CAPI_EXPORT extern TF_Function* TF_FunctionImportFunctionDef(",
}

# ---------------------------------------------------------------------------
# 三、控制流（文档 02）
# ---------------------------------------------------------------------------

# 来源：site/en/guide/function.ipynb（§Conditionals）
AUTOGRAPH_IF_VERBATIM = (
    "AutoGraph will convert some `if <condition>` statements into the equivalent "
    "`tf.cond` calls. This substitution is made if `<condition>` is a Tensor. Otherwise, "
    "the `if` statement is executed as a Python conditional.")

PYTHON_IF_VERBATIM = (
    "A Python conditional executes during tracing, so exactly one branch of the "
    "conditional will be added to the graph. Without AutoGraph, this traced graph would "
    "be unable to take the alternate branch if there is data-dependent control flow.")

COND_BOTH_BRANCHES_VERBATIM = (
    "`tf.cond` traces and adds both branches of the conditional to the graph, "
    "dynamically selecting a branch at execution time.")

# 来源：同笔记本（§Loops 的转换规则清单）
LOOP_RULES = {
    "for": ("`for x in y`: if `y` is a Tensor, convert to `tf.while_loop`. In the special "
            "case where `y` is a `tf.data.Dataset`, a combination of `tf.data.Dataset` "
            "ops are generated."),
    "while": "`while <condition>`: if `<condition>` is a Tensor, convert to `tf.while_loop`.",
}

# 来源：tensorflow/python/autograph/g3doc/reference/control_flow.md（§Effects of the tracing process）
TRACING_ALL_PATHS_VERBATIM = (
    "When constructing a graph, TensorFlow _traces_ the code. The tracing of control flow "
    "requires visiting _every possible code path_ (usually once).")
TRACING_BOTH_BRANCHES_VERBATIM = (
    "In other words, tracing executes both branches of an if statement. Similarly, the "
    "body of loops is executed once (even if the loop would otherwise not iterate at all).")

# 来源：tensorflow/python/framework/tensor.py:302-319（`_disallow` / `_disallow_bool_casting`）
# 逐字：任务串 + 固定后缀（后缀在源码里是同一个 f-string 的其余部分）
ERR_TASK_BOOL_CAST = "Using a symbolic `tf.Tensor` as a Python `bool`"
ERR_TASK_ITERATION = "Iterating over a symbolic `tf.Tensor`"
ERR_DISALLOW_SUFFIX = (
    " is not allowed. You can attempt the following resolutions to the problem: If you "
    "are running in Graph mode, use Eager execution mode or decorate this function with "
    "@tf.function. If you are using AutoGraph, you can try decorating this function with "
    "@tf.function. If that does not work, then you may be using an unsupported feature or "
    "your source code may not be visible to AutoGraph. See "
    "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/"
    "g3doc/reference/limitations.md#access-to-source-code for more information.")

# ---------------------------------------------------------------------------
# 四、XLA（文档 02 §3）
# ---------------------------------------------------------------------------

# 来源：tensorflow/python/eager/polymorphic_function/polymorphic_function.py:1591-1602
# 载体：`tf.function` 的 docstring（jit_compile 参数段）
JIT_COMPILE_DOC = (
    "jit_compile: If `True`, compiles the function using [XLA](https://tensorflow.org/xla). "
    "XLA performs compiler optimizations, such as fusion, and attempts to emit more "
    "efficient code. This may drastically improve the performance. If set to `True`, the "
    "whole function needs to be compilable by XLA, or an `errors.InvalidArgumentError` is "
    "thrown. If `None` (default), compiles the function with XLA when running on TPU and "
    "goes through the regular function execution path when running on other devices. If "
    "`False`, executes the function without XLA compilation.  Set this value to `False` "
    "when directly running a multi-device function on TPUs (e.g. two TPU cores, one TPU "
    "core and its host CPU). Not all functions are compilable, see a list of [sharp "
    "corners](https://tensorflow.org/xla/known_issues).")

# 来源：tensorflow/python/compiler/xla/xla.py:62-67
# 逐字：装饰器 + 它的消息。第一个参数是 `None` ⇒ **官方没给移除日期**。
XLA_EXPERIMENTAL_COMPILE_DEPRECATION = (
    "@tf_export('xla.experimental.compile')\n"
    "@deprecated(\n"
    "    None, 'xla.experimental.compile is deprecated. Consider using '\n"
    "    '`@tf.function(jit_compile=True)`.',\n"
    "    warn_once=True)\n"
    "def compile(computation, inputs=None):")
XLA_EXPERIMENTAL_COMPILE_MESSAGE = (
    "xla.experimental.compile is deprecated. Consider using `@tf.function(jit_compile=True)`.")

# ---------------------------------------------------------------------------
# 五、自定义算子与导出（文档 03）
# ---------------------------------------------------------------------------

# 来源：site/en/guide/create_op.md（首段，逐字）
CREATE_OP_RECOMMENDATION = (
    "If you'd like to create an op that isn't covered by the existing TensorFlow library, "
    "we recommend that you first try writing the op in Python as a composition of existing "
    "Python ops or functions. If that isn't possible, you can create a custom C++ op.")
# ★ 同一文件的实测：`grep -c raw_ops create_op.md` = **0**。
#   所以**不得**写「`tf.raw_ops` 是官方推荐的 X」。
CREATE_OP_RAW_OPS_MENTIONS = 0

# 来源：tensorflow/python/ops/custom_gradient.py:46-47（docstring 首句，逐字）
# ★ 这条补上了计划 §3.1 的 **S1**（原计划说「§0.7.B 里没研究过它」）：
#   它的定位是**给已有的 Python 组合换一个梯度函数**，不是注册新 op。
TF_CUSTOM_GRADIENT_DOC = "Decorator to define a function with a custom gradient."
TF_CUSTOM_GRADIENT_SOURCE = src("tensorflow/python/ops/custom_gradient.py", (46, 47))

# 来源：site/en/guide/saved_model.ipynb（逐字，**含结尾那半句**）
SAVED_MODEL_DEPRECATION_VERBATIM = (
    "Deprecated: For Keras objects, it's recommended to use the new high-level `.keras` "
    "format and `tf.keras.Model.export`, as demonstrated in the guide here. The low-level "
    "SavedModel format continues to be supported for existing code.")
# ⚠ 本次只核实到**一处**；计划说「两处」，未复现（见文件头的 UNSOURCED 段）。

# 来源：tensorflow/python/saved_model/save.py:1241-1244
# 载体：源码装饰器——**只有 `@tf_export`，没有任何 `@deprecated`**
SAVED_MODEL_SAVE_DECORATOR = (
    '@tf_export(\n    "saved_model.save",\n'
    '    v1=["saved_model.save", "saved_model.experimental.save"])\ndef save(')

# 来源：https://keras.io/api/models/model_saving_apis/export/（Keras 3 API 文档，逐字）
# `Model.export(filepath, format="tf_saved_model", …)` —— 默认值就是 SavedModel
KERAS_EXPORT_SIGNATURE = 'Model.export(filepath, format="tf_saved_model", verbose=None, input_signature=None, **kwargs)'
KERAS_EXPORT_FORMAT_DEFAULT = "tf_saved_model"
KERAS_EXPORT_DOC_URL = "https://keras.io/api/models/model_saving_apis/export/"

# ---------------------------------------------------------------------------
# 六、三者对照（文档 01/03 的镜像，来自计划 §0.7.D，此处只放**有来源的那半**）
# ---------------------------------------------------------------------------

# 本主题能一手核实的是 TF 这一列；JAX 那一列请回链 70-frameworks/jax（Chunk 3）。
# ★ 对照的结论：**TF 与 JAX 对 Python 标量的处理正好相反**。
#   TF：按**值**特化（LiteralTraceType<3>，见 TRACETYPE_RULES["Python 原生值"]）。
#   JAX：默认**不**特化（本文件不含 JAX 的来源，**不得在此处替 JAX 下断言**）。
TF_VS_JAX_SCALAR_NOTE = (
    "TF 侧有来源：Python 原生值的 TraceType 是 LiteralTraceType<值本身>。"
    "JAX 侧见 70-frameworks/jax —— 本文件不替它下断言。")
