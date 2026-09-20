"""45-kernel-lifecycle/operator-design 的外部事实常量源。

本模块是 CONTRIBUTING §2 约定的**共享事实源**（`specs.py`）：三条编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ 本主题的常量**只有三类来源**（域 README 的 A–E 五级）：

  · **A 级：规范与厂商文档**——PTX ISA 的 `wmma` 语义、CUDA C++ Programming Guide 的
    wmma 对齐与占用限制、NVIDIA H100 白皮书的 SMEM 容量、cuDNN 的 FP8 约束、
    IEEE 754 的 `u`。本主题的**主体**是这一类。
  · **B 级：某一实现的官方规格**——vLLM 的 `vllm.ir.op` / `vllm/config/kernel`（IR 与
    compile key）。**只能支撑「这一家这么做」**（P10）。
  · **D 级：社区仓库**——只有一条（Dynamo 的 issue）用来做**反方向**的对照，
    正文出现时必须写明它是 D 级。

★ **本模块刻意不含散文。** 上一轮（`requirement-discovery`）的 C6 教训：把来源里的
  **描述性段落**整段搬进 `specs.py`，程序**没法计算散文 ⇒ 只能 `print` ⇒ 必然漏网**，
  而且会让模块「看起来很充实」，掩盖「这些常量没人查」。
  ⇒ **规则：每个模块级常量必须是可被计算的值**（数、名、布尔、阈值、元组）。
    描述性文字归**文档**；来源里的每个数字以**结构化字段**的形式落在这里。

★ 本模块的常量**全部**被三个程序断言钉住。**判据是机械的**：`01_contract_check.py`
  用 `ast` 读本目录另外两个程序里的 `COVERED_*` 元组，断言**三者的并集 == 本模块的
  全部模块级常量**（见该程序 §D.9）。⇒ 在这里加一个常量而没人断言它，`make run` 会失败。
"""

# ===========================================================================
# 1. 契约的落点：注册、dispatch、kernel cache key（**B 级 · vLLM 的做法**）
# ===========================================================================
#
# ★ P10 的正面用法：本节全部常量只能支撑「**vLLM 这一家**这么做」。
#   正文必须写成「vLLM 的做法是……」，**不得**写成「业界通常」「主流做法」。
#
# ★ 本节**没有**任何描述性段落。上一轮把整段来源文字搬进来的做法已废止——
#   下面每一条都是**可以被断言的东西**（布尔、名字、元组）。

VLLM_IR_RESERVED_PROVIDERS = ("native", "unfused")
# 来源：vLLM 官方文档 `vllm.ir.op`（`RESERVED_PROVIDERS`）  [B]（**经检索转述**）
# 口径：`native` 是**保留名**，指 `@register_op` 自带的那份实现 ⇒ 优先级列表
#       **永远有兜底**，不会因为所有 provider 都不支持而选不出实现。

VLLM_IR_REGISTER_IMPL_KWARGS = ("supported", "supports_args", "inplace")
# 来源：同上（`register_impl(provider, *, supported=True, supports_args=None, inplace=False)`）  [B]
# 口径：★ 这是**两层支持性检查**的落点，本篇 §2 的论点就是「**两层**不是一层」。

VLLM_IR_TWO_LEVEL_SUPPORT_CHECKS = ("supported", "supports_args")
# 来源：同上  [B]
# 口径：★ **两层检查的分工是本主题最可操作的一条事实**：
#       `supported` 是**静态布尔**（import 时求值一次，只用于**平台/库是否存在**）；
#       `supports_args` 是**动态谓词**（按类型与形状判，编译期拿 fake tensor 调，
#       eager 路径拿真 tensor 调）。
#       ⚠ 两者**都不是**「这个 kernel 快不快」的判断——它们只回答「能不能用」。

VLLM_IR_SUPPORTED_IS_IMPORT_TIME_ONLY = True
# 来源：同上  [B]
# 口径：`supported` 在 import 时求值一次 ⇒ **它不能依赖运行期状态**。
#       把「运行期才定的值」放进 `supported` 是一个具体的错误（§4 的坑 ③）。

VLLM_IR_SUPPORTS_ARGS_MUST_NOT_CHECK_BATCH = True
# 来源：同上  [B]
# 口径：★★ **反直觉的一条，也是本篇最可证伪的一条**：官方文档明确要求
#       `supports_args` **不应**检查 batch size，也不应加「基于取值」的守卫
#       （"Should NOT check batch sizes or add value-based guards"）。
#       ⇒ 「运行期才定的值不进编译期决策」这条规则**在实现里是被写下来的**，
#         不是本库的推论。
#       ⚠ 这与「凡改变构建结果的字段都要进 key」**不矛盾**：batch size 不改变
#         **构建出哪个 kernel**，它只改变**同一个 kernel 的调用参数**。

VLLM_IR_PRIORITY_CLI_PREFIX = "--ir-op-priority"
# 来源：同上  [B]（与 `requirement-discovery` 的 `VLLM_IR_PRIORITY_CLI_PREFIX`
#       **同值**，此处按「跨主题常量复制」纪律重抄一份，并在正文注明与该主题一致）
# 口径：选择规则 = 「**优先级列表中第一个 `supported` 且 `supports_args` 通过的实现**」。
#       ⇒ 选择本身是一个**显式配置项**，不是隐式回退。

VLLM_IR_IMPL_UUID_IS_SOURCE_HASH = True
# 来源：同上（`IrOpImpl.uuid()` = `hash_source([Path(inspect.getfile(self.impl_fn))])`）  [B]
# 口径：★ `uuid` 是**编译期哈希**，用来判断「实现变了没有」。

VLLM_IR_DYNAMO_HIDES_IMPL_SOURCES = True
# 来源：vLLM `vllm/config/kernel.py` 的源码注释（"Implementations are hidden from Dynamo,
#       so they don't show up in the traced files list."）  [B]（**经检索转述**）
# 口径：★★ **这是本篇 B 类的核心**：一个**真的会改变构建结果**的输入（实现源文件），
#       **不在**被追踪的文件列表里——它必须被**手工**加进 hash factors。
#       ⇒ 现实的形态不是「有人忘了写」，而是「**机制上看不见它**」。

VLLM_IR_IMPL_UUID_MANUALLY_IN_HASH = True
# 来源：同上（`IrOpPriorityConfig.compute_hash()` 把各 provider 的 `uuid()` 收进
#       `factors["_impls"]`）  [B]
# 口径：★ 与上一条配对：**看得见的东西会被自动收集，看不见的必须手工加**。

VLLM_KERNELCONFIG_HASH_INCLUDES_COMPILE_AFFECTING_FIELDS = True
VLLM_KERNELCONFIG_HASH_EXCLUDES_NON_COMPILE_FIELDS = True
# 来源：vLLM `KernelConfig.compute_hash()` 的 docstring（"Any new fields that affect
#       compilation should be added to the hash. Any future fields that don't affect
#       compilation should be excluded."）  [B]
# 口径：★ 这条规则是**双向**的：**漏加** ⇒ 该失效的没失效；**多加** ⇒ 永远不复用。
#       两句话在本模块里拆成两个布尔——**散文归文档，判据留在这里**。
VLLM_KERNELCONFIG_HASH_RULE_IS_VERBATIM = False
# 口径：★ **引用纪律**：上面那句 docstring 本实现者**只经一次检索读到**
#       ⇒ 按域 README 的规则（「逐字」须**两次独立检索一致**且出处到文件级），
#       它只能标**「经检索转述」**。正文**不得**加引号、不得写「原文如此」。
#       ⇒ 这个布尔就是那条纪律的机器化：它的值是 `False`，改它是会被断言的。

VLLM_DYNAMO_BUSTING_FIELD = "fpm_worker_id"
VLLM_DYNAMO_BUSTING_FIELD_IS_COMPILE_INPUT = False
# 来源：ai-dynamo/dynamo 的 issue #10119  [D]（**社区仓库的 issue，单一主体**）
# 口径：★★ **反方向的证据，而且它比正方向更容易被忽略**：
#       一个**不改变构建结果**的运行期字段（worker id）进了 config hash ⇒
#       每次 rerun 都算出新 hash ⇒ **torch.compile 缓存一次都不命中**。
#       ⚠ D 级：**正文出现这两个常量时必须写明它是 D 级、单一项目的报告、未被独立复现**，
#         且**不得单独支撑结论**（它是与 B 级那条 docstring 规则**同向**的独立一例，
#         两者方向一致才允许写）。


# ===========================================================================
# 2. 数值设计：**规范规定什么、不规定什么**（A 级）
# ===========================================================================
#
# ★★ 本节是本主题最硬的事实源。分两组：
#    (a) **规定了的**（"at least …"）——PTX ISA 对 `wmma.mma` 的精度下限；
#    (b) **没规定的**（"unspecified"）——累加顺序、舍入、次正规处理。
#    **本篇的核心论断就是 (b) 的存在**：同一个 kernel 的浮点结果**不是规范保证的**。

PTX_WMMA_MUL_PRECISION_AT_LEAST = "single"
# 来源：NVIDIA *Parallel Thread Execution ISA* 的 `wmma.mma` 条目，
#       "Precision and rounding for .f16 floating point operations" 段：
#       "Element-wise multiplication of matrix A and B is performed with at least
#        single precision."  [A]
# 口径：★★ **逐字**——本实现者**两次独立检索**读到同一段文字（PTX ISA 6.4/8.0/8.1/9.1
#       四个版本的归档页 + 一份 PDF 检索片段）。出处可给到**文档级**（PTX ISA 的
#       `wmma.mma` 一节）。
#       ⚠ 「at least」是**下限**：允许更精确，不允许更差。**它不是「就是单精度」**——
#         把「至少」读成「恰好」是本篇 §4 的坑 ①。

PTX_WMMA_ACC_F32_AT_LEAST = "single"
# 来源：同上："When .ctype or .dtype is .f32, accumulation of the intermediate values
#       is performed with at least single precision."  [A]（**逐字**，同上两处检索）
# 口径：★ 注意条件是 **`.ctype` `.dtype` 二者之一**为 `.f32`，不是二者都为 `.f32`。

PTX_WMMA_ACC_F16_AT_LEAST = "half"
# 来源：同上："When both .ctype and .dtype are specified as .f16, the accumulation is
#       performed with at least half precision."  [A]（**逐字**，同上两处检索）
# 口径：★ 条件反过来：**两者都是** `.f16`。⇒ **只有这一档允许半精度累加**。

PTX_WMMA_UNSPECIFIED_ASPECTS = ("accumulation order", "rounding", "subnormal input handling")
# 来源：同上："The accumulation order, rounding and handling of subnormal inputs is
#       unspecified."  [A]（★ **逐字**，两次独立检索一致）
# 口径：★★ **本主题最重要的一条事实**：它不是「精度不够」，而是**「规范不说」**。
#       三者未规定 ⇒ **同一个 kernel 在同一份输入上的结果不保证逐位可复现**
#       （对同一份二进制、同一块 GPU 都可能随 tile 划分/scoring 变化）。
#       ⇒ 任何**依赖逐位可复现**的设计，都必须**自己保证顺序**，而不能引用规范。
#       ⚠ 这一条是**规范层的**事实（P8）：它说的是 PTX 允许什么，
#         **不是**某块具体 GPU 的实际行为（实际行为可以被实测，但那不是规范）。

PTX_WMMA_UNSPECIFIED_QUOTE_IS_VERBATIM = True
# 口径：★ 上面三条引文的**引用等级标记**。按域 README：`逐字` 须**两次独立检索一致**
#       且出处到**文件级**——三处都满足（PTX ISA 的 `wmma.mma` 精度段）。
#       ⚠ 与 §1 的 `VLLM_KERNELCONFIG_HASH_RULE_IS_VERBATIM = False` **成对读**：
#         一边够格标逐字，一边不够格——**这个对照本身就是引用纪律的执行记录**。

CUDNN_FP8_COMPUTE_TYPES = ("CUDNN_DATA_FAST_FLOAT_FOR_FP8", "CUDNN_DATA_FLOAT")
# 来源：NVIDIA cuDNN Developer Guide，"Recommended compute type for FP8 tensor
#       computations for Hopper architecture"（表 11）与 cuDNN 8.8/8.9 的卷积约束表  [A]
#       （**经检索转述**：本实现者只读到检索摘要，未逐字读到该表原文）
# 口径：**FP8 张量核的累加精度由 compute type 两档控制**。

CUDNN_FP8_FORWARD_MAY_USE_FAST_FLOAT = True
# 来源：同上  [A]（经检索转述）
# 口径：`CUDNN_DATA_FAST_FLOAT_FOR_FP8` **更快**、对**推理**与**训练前向**足够精确。

CUDNN_FP8_BACKWARD_REQUIRES_FLOAT = True
# 来源：同上  [A]（经检索转述）
# 口径：★★ **反直觉且可证伪**：**同一个算子**（FP8 卷积）在**前向**与**反向**上
#       **允许的累加精度不同**——反向算权重/激活梯度时 cuDNN 建议用精度更高的
#       `CUDNN_DATA_FLOAT`（且在 runtime fusion engine 的约束里，ConvolutionBwData /
#       ConvolutionBwFilter 的 compute type **只列了 `CUDNN_DATA_FLOAT`**）。
#       ⇒ **「这个算子该用哪档累加」不是算子的属性，是「算子 × 方向」的联合属性**——
#         与 `requirement-discovery` 02 篇「受限类型是「算子 × 精度路径」的联合属性」
#         **同构**（跨主题的同一条结构性认识）。

FP16_UNIT_ROUNDOFF_U = 2.0 ** -11
FP32_UNIT_ROUNDOFF_U = 2.0 ** -24
# 来源：IEEE 754 binary16 / binary32 的**单位舍入误差** `u = 2^-p`，`p` = 有效位数
#       （binary16 的 1+5+10 ⇒ p = 11；binary32 的 1+8+23 ⇒ p = 24）  [A]
# 口径：★ `u` 是**相对**量（一半是 `u/2` 的界之常写法，本主题一律用 `u`，与
#       `10-foundations/floating-point/04-low-precision-formats` 的 `γ_K = K·u/(1-K·u)` 同口径）。
#       ⇒ **两个精度之间只差一个常数因子**：`u16 / u32 == 2^13 == 8192`。
#       ⚠ 这个 8192 **不是**「约三个数量级」——`log10(8192) = 3.91`。
#         正文写数量级时必须写 **≈ 3.9 个数量级**（或直接给 8192 倍）。
#         本模块的 `U16_OVER_U32_ORDERS_OF_MAGNITUDE` 就是防止再写错的那条常量。

FP16_MAX_FINITE = 65504.0
# 来源：IEEE 754 binary16 的最大有限数（=(2-2^-10)·2^15 = 65504）  [A]
# 口径：★ 它是**累加的第二个天花板**：即使误差界没问题，`K` 个数的**和本身**
#       也不能溢出 fp16。本主题用它给出「K 的第三个上界」（03 节的 B 类）。

FP16_FRAGMENT_SHAPES = ((16, 16, 16), (32, 8, 16), (8, 32, 16))
# 来源：CUDA C++ Programming Guide 的 Warp Matrix Functions（`__half` 支持三种 shape）
#       与 PTX ISA 的 `wmma`（`.m16n16k16` PTX ISA 6.0 引入；`.m8n32k16` / `.m32n8k16`
#       在 6.1）  [A]（**两次独立检索一致**）
# 口径：★ 三种 shape 是**同一套 fragment 规则的不同布局**（寄存器数相同，见下）。
#       本主题的分块只能是这三种之一 ⇒ **tile 选型是离散的**（03 篇 §0 的论点）。
#       ⚠ `tf32` 只有 `16x16x8`、`double` 只有 `8x8x4`——**与本主题的 fp16 路径不是同一组**。

WMMA_A_FRAGMENT_REGS = 8
WMMA_B_FRAGMENT_REGS = 8
WMMA_ACC_F32_FRAGMENT_REGS = 8
WMMA_ACC_F16_FRAGMENT_REGS = 4
# 来源：PTX ISA 的 `wmma` 矩阵 fragment 表（fp16 的三种 shape 每线程寄存器数）：
#       A = 8、B = 8、`.f32` 累加器 = 8、`.f16` 累加器 = 4  [A]（经检索转述）
# 口径：★★ **寄存器的计数单位是「一个 fragment」而不是「一个元素」**——
#       A 的 fragment 是 16×16 个 `.f16` = 32 B/线程 = 8 个 32-bit 寄存器。
#       ⇒ 三种 shape **寄存器数相同、只有布局不同**（这是它们的定义方式）。
#       ★ 累加器 `.f16` 只要 **4** 个寄存器（两个 `.f16` 打包进一个 `.b32`）——
#         **用 `.f16` 累加可以省一半累加器寄存器**，代价是 §2 的精度（这是
#         03 篇 §1「寄存器预算」与 02 篇「精度预算」**打架**的地方，两篇各说一半）。

U16_OVER_U32 = 8192.0
U16_OVER_U32_ORDERS_OF_MAGNITUDE = 3.91
# 来源：**本库推导值 [E]**（`FP16_UNIT_ROUNDOFF_U / FP32_UNIT_ROUNDOFF_U` 与它的
#       `log10`）。放在这里是因为**同一个数被两次写错过**（本域规格 §Chunk2 的
#       逐篇表写「容差相差约三个数量级」）。
# 口径：★ 数量级要按**十进制**算：8192 = 10^3.91。写「三」是**低估**，
#       写「四」是**高估**；给 **3.9** 或直接给 **8192 倍**。
#       ⚠ 这个常量存在的唯一目的是**防止那个措辞错误回流**——它被 02 篇程序断言。


# ===========================================================================
# 3. 分块与边界：**约束方程的四条腿**（A 级）
# ===========================================================================
#
# ★ 本节把 03 篇的约束方程拆成**四组**：寄存器 / 共享内存 / 对齐 / warp 数。
#   「四者取最紧」的可计算形态是：对每个候选 tile，四条各算一个**紧度**，
#   合法 tile 是**四条全部满足**的那些，而**最紧的那条**决定它为什么合法。

WMMA_MPTR_ALIGN_BYTES = 32
# 来源：CUDA C++ Programming Guide §7.24「Warp Matrix Functions」（`mptr` 必须是
#       32 字节 / 256 bit 对齐）  [A]
# 口径：★ 本库**已引用过**同一句：`20-gpu-architecture/tensor-core/02-fragment-layout.md`
#       第 234 行（"`mptr` 必须按 32 字节（256 bit）对齐"）与其「参考」一节。
#       ⇒ 本主题**照抄口径、不重推**（回链该篇）。

WMMA_LDM_ELEMENT_MULTIPLE = {"fp16": 8, "fp32": 4}
WMMA_LDM_BYTE_MULTIPLE = 16
# 来源：CUDA C++ Programming Guide §7.24（`ldm` 必须是 8（`__half`）/ 4（`float`）的倍数）  [A]
# 口径：★★ 这里有一个**必须写清楚、否则一定会写错**的换算：
#       `ldm` 的单位是**元素**，8 个 `__half` = 16 B、4 个 `float` = **也是 16 B**。
#       ⇒ 「`ldm` 是 8 或 4 的倍数」与「**行距的字节数是 16 的倍数**」是**同一件事的
#         两种写法**（本库 `02-fragment-layout.md` 第 232 行的原话：
#         「官方表格给的两个数字（8 与 4）是**同一个约束在不同元素宽度下的写法**」）。
#       ⚠ **但规律是「16 字节」不是「ldm ≥ 8」**：`ldm = 8` 与 `ldm = 4` 本身
#         **不是** 16 的倍数——是**它们的字节积**才是。把 `ldm` 直接说成
#         「16 字节的倍数」是**措辞错误**（本域规格初稿就是这么写的，已订正）。

WMMA_FRAGMENT_BYTES = 32
# 来源：PTX ISA 的 `wmma.load` 对齐要求——「起始地址按**对应 fragment 的字节大小**对齐」，
#       m16n16k16 的 f16 fragment = `16 行 × 16 B` 中的**一个 32 B 的 fragment**
#       （`wmma.load.a...m16n16k16.f16` 的 `p` 必须是 32 的倍数，`2*s` 也是）  [A]
#       （**经检索转述**：单次检索）
# 口径：★★ **这是与上一条不同的规则，两个数都等于 16 却不是一个东西**——
#       本库 `02-fragment-layout.md` 第 236 行逐字警告过：
#       「**两个数都等于 16 字节（`8 × 2 B` 与 `4 × 4 B`），但不是同一条规则，
#         别互相换算。**」
#       ⇒ 本模块**故意把两条规则各留一个常量**（`WMMA_LDM_BYTE_MULTIPLE = 16` 与
#         `WMMA_FRAGMENT_BYTES = 32`），并在 03 篇程序里断言**它们不相等**——
#         防止有人「顺手统一」成一个数。

WMMA_ALIGN_RULES_ARE_DISTINCT_LEVELS = True
# 来源：本库 `20-gpu-architecture/tensor-core/02-fragment-layout.md:236`  [E]（本库既有口径）
# 口径：★ 一句话：**CUDA C++ API 层**说「`mptr` 32 B 对齐 + `ldm` 是 8/4 的倍数」；
#       **PTX `wmma.load` 层**说「起始地址按 fragment 的字节大小对齐」。
#       前者是**按位对齐的基数**，后者是**按（元素数算的）stride**。
#       ⇒ 本文的约束方程**在 C++ API 这一层**写（因为设计者接触的是这一层），
#         并把 PTX 那一条**单独列出**、标明层级。

CUDNN_FP8_HOPPER_CHANNELS_MULTIPLE = 16
CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE = 16
CUDNN_FP8_HOPPER_FWD_BATCH_ANY = True
# 来源：NVIDIA cuDNN Developer Guide 的 Hopper FP8 约束表（I/O channels 必须是 **16 的倍数**；
#       `ConvolutionFwd` 的 batch size **任意**，而 `ConvolutionBwFilter` / `ConvolutionBwData`
#       要求 batch size 是 **16 的倍数**）  [A]（**经检索转述**）
# 口径：★★ **本库此前未引用过这一条**（本域规格标注「实施时须自行核到 cuDNN 文档原文」，
#       本实现者做到了两处独立检索一致）。
#       ★ 它是**问题形状**的约束，不是 tile 参数：**它直接判某些问题形状无法走这条快路**，
#         无论 tile 怎么选。⇒ 约束方程里必须把「tile 参数约束」与「问题形状约束」
#         **分开写**——它们的修复方式不同（前者换参数，后者改数据布局/换算法）。
#       ★ 前向 batch 任意 / 反向必须 16 的倍数，是**又一处「算子 × 方向」的不对称**，
#         与 `CUDNN_FP8_BACKWARD_REQUIRES_FLOAT` 同向。

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 CC 的 "warp size" 恒为 32）  [A]
# 同值位置：20-gpu-architecture/architecture-evolution/code/specs.py:281 等。

MAX_THREADS_PER_BLOCK = 1024
# 来源：CUDA C++ Programming Guide 13.4 §5.1（"Maximum number of threads per block"，
#       CC 2.0 起为 1024）  [A]
# 同值位置：20-gpu-architecture/architecture-evolution/code/specs.py:285。

MAX_REGISTERS_PER_THREAD = 255
# 来源：CUDA C++ Programming Guide 13.4 §5.1（"Maximum number of 32-bit registers per
#       thread" 一栏，CC 3.0 起为 255）  [A]
# 同值位置：20-gpu-architecture/tensor-core/code/specs.py:46、
#          40-kernels/{gemm:36, attention:51, elementwise-fusion:48}/code/specs.py。

REGISTER_ALLOC_GRANULARITY = 8
# 来源：CUDA C++ Programming Guide 13.4 §5.1 的寄存器分配粒度说明（每线程寄存器数
#       按 8 个一档向上取整）  [A]
# 口径：★★ **它把 255 变成 248**：上取整到 8 的倍数后 ≤ 255 的最大值是 **248**
#       （256 > 255）。⇒ **寄存器预算的真实上限是 248，不是 255**。
#       这是一个**纯从两条 A 级常量推出来、却极容易算错**的数——
#       03 篇的 B 类把它做成「恰好合法 / 恰好非法」的那一对。

SMEM_PER_SM_MAX_KB = 228
# 来源：NVIDIA H100 Tensor Core GPU Architecture Whitepaper（每 SM 统一池 256 KB，
#       其中**至多 228 KB 可切给 shared memory**）  [A]
# 口径：★ 本主题**把它当作「每 block 的共享内存预算」用**——这**忽略了两件真事**：
#       ① carve-out 有保留量（每 block 可用的最大值略小于 228 KB）；
#       ② 每 SM 上还要驻留多个 block 时，预算是**被分摊**的。
#       ⇒ **得到的 tile 上界是乐观上界**（真实可用的更小）。正文必须写出这句限定。
#       同值位置：20-gpu-architecture/tensor-core/code/specs.py:53、
#                40-kernels/{gemm:43, attention:58, elementwise-fusion:63,
#                reduce-scan:53, softmax-normalization:45, moe-kernels:40}/code/specs.py。

MAX_WARPS_PER_SM = 64
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 7.0/8.0/9.0/10.0 的
#       "Maximum number of resident warps per SM" = 64）  [A]
# 口径：★ 64 × 32 = 2048，与「每 SM 最多 2048 线程」是**同一条约束的两种写法**。
#       本主题用它给出**warp 数约束**的 SM 侧上限；block 侧上限来自
#       `MAX_THREADS_PER_BLOCK`。**两者取最紧**。
#       同值位置：40-kernels/elementwise-fusion/code/specs.py:38、reduce-scan:39。

ADDRESS_AND_INDEX_REGS_ASSUMED = 0
# 来源：**本库的设计假设 [E]**，不是来源值。
# 口径：★★ 它是**故意取 0** 的：真实的 kernel 除 A/B/累加器 fragment 之外还要
#       地址、索引、循环变量等寄存器（数十个）。取 0 使本主题算出的
#       **最大合法 tile 是一个「乐观上界」**——真实能用的一定更小。
#       ★ 这样处理的好处是**结论不依赖一个编出来的数**：`ADDRESS_AND_INDEX_REGS_ASSUMED`
#         是一个**明写的、可改的输入**，任何读者都能把它改成 40 重算一遍。
#       ⚠ **不得**把它换成一个「看起来更真实」的常数——那会是本域最严重的 P4
#         （编机制从句：没有任何来源能给出「地址寄存器要几个」）。

FLOAT16_BYTES = 2
# 来源：IEEE 754 binary16 的存储宽度  [A]
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py（**同域内的上一主题**，
#           此处按「跨主题常量复制」纪律重抄，正文注明与该主题一致）。


# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ §0.6 硬规则 5：**不得「既保留数字又记 UNSOURCED」**。下表每一项都**没有**在
#   本模块的正文常量里出现。

UNSOURCED = (
    # ── 本清单的口径 ───────────────────────────────────────────────────────
    # ★ 每一项是一个**稳定标识符**（不是散文）。三个程序对本元组做**逐条比对**
    #   （01 篇 §D.8），所以改任何一个标识符都会让 `make run` 失败。

    # 1. 「地址/索引 registers 到底占多少」——本主题唯一真正的缺口。
    #    状态：**无来源**。没有任何 A/B/C 级材料给出「一个 m×n×k tile 的 kernel
    #    需要多少个非 fragment 寄存器」。处置：不写数字，改为
    #    `ADDRESS_AND_INDEX_REGS_ASSUMED = 0` 这个**明写的假设**，并断言
    #    「取 0 得到的是乐观上界」。⇒ 缺口被**标注**而不是被**填平**。
    "address_and_index_register_budget",

    # 2. 「大多数 kernel 在 80% 带宽处饱和」这类**编机制从句**（P4）。
    #    状态：**无来源，也测不了**（本机无 GPU）。处置：不写。
    "ungrounded_saturation_claims",

    # 3. 「FP8 累加用 FAST_FLOAT_FOR_FP8 会带来多少精度损失」。
    #    状态：**未找到**。cuDNN 只说「更快、对推理与前向足够准确」，
    #    **没有给出**任何误差数字。⇒ 正文**不得**出现这类数字。
    #    处置：只写「允许哪一档」，不写「差多少」。
    "cudnn_fast_float_for_fp8_error_magnitude",

    # 4. 「wmma 的 fragment 在 sm_70 上的**具体元素归属**」。
    #    状态：**规范明写为 unspecified**（"unspecified and target-architecture
    #    dependent"），本库 `20-gpu-architecture/tensor-core/02-fragment-layout.md`
    #    已用穷举法给出过**一个**具体布局。⇒ 本主题**不回链重推**（§0.5 边界），
    #    也不把它写成规范保证。处置：只引用「布局未规定」这个**规范事实**。
    "wmma_fragment_element_mapping_is_unspecified",

    # 5. 「vLLM 的 compile key 在一次真实坏缓存里导致了哪个 kernel 被选错」。
    #    状态：**未找到任何一例**。检索只找到两个**机制性**证据（实现源文件被
    #    Dynamo 隐藏 ⇒ 必须手工进 hash；以及一个**不**影响编译的字段进了 hash
    #    ⇒ 缓存从不命中）。**没有**「key 漏字段 ⇒ 静默返回错 kernel」的公开案例。
    #    ★ 处置：01 篇的 C 类是**本库构造的仿真**，正文必须**明写它是构造的**，
    #      不得暗示它是 vLLM 的真实故障。
    "real_world_wrong_kernel_by_missing_key_field",

    # 6. 「只在最后一 tile 出错」的**真实缺陷清单**。
    #    状态：**未找到**。检索没有给出一份可引用的、真实的「tail-only」bug 清单
    #    （既没有 CVE 式的编号，也没有一个可复核的复现记录）。
    #    ★ 处置：03 篇 §C 的三个缺陷模型（丢 M tail / 夹回起点写错位置 / 丢 K tail）
    #      是**本库构造的**，正文必须**明写它们是构造的**。
    #    ⚠ 连带的一个缺口：`2·TM+1` 那一类「余数自己还要再分块」的形状，
    #      **本程序没有为它建模任何缺陷** ⇒ 它对连续 tiling 是**冗余**的。
    #      正文不得声称它「补上了单靠 +1 抓不到的那类」——那个说法**没有对象**。
    "real_world_tail_defect_inventory",
)
# ★ `UNSOURCED` 是**待办清单**：上面每一项都在等一个可定位的一手来源，
#   在它被找到之前，**正文里不得出现对应的数字或案例**。
