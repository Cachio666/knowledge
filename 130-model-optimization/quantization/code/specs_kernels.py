"""130-model-optimization/quantization 的外部规格常量源。

本文件只放两类东西：
  1. **外部常量**（规范明文、判据阈值、外部实现的设计选择），每条带一行 `# 来源：`；
  2. **本机实测基线**（与三篇文档 §5 逐行一致的数字），每条注明测它的版本与判据。

★ 核查范围（错误模式 P6 的对策）——**本主题的来源档位逐条写明**：

  · 标 `（本机实测）` 的：实现者 2026-09-18 在本机 5 个解释器上真跑过，
    数字与 `specs_kernels.py` 逐字一致，可用文档里的复跑命令复现。
  · 标 `（源码直取）` 的：实现者本次用 python `urllib` 把该文件**逐字读下来**，
    并记下**取文件的 URL 与字节数**（vLLM 的 machete `Readme.md`），结论可在同一位置复核。
  · 标 `（规范/论文印值）` 的：外部规范或论文的原文，本次**未逐字读到**，
    只有计划基线与检索摘要两条间接通道 —— 按**引用值**处理，不得说成「已核实」。
  · 标 `（计划基线，本次未独立核实）` 的：只来自计划文件 §0.7.E 的事实基线，
    本实现者**本次没有拿到任何独立佐证**（本轮 `raw.githubusercontent.com` 全程 404 / 超时；
    `WebFetch` 被策略拦截）。**这类条目在正文里一律写「引用值」并注明未核实。**

  ⚠ 三档在正文里**都**写成「引用值」或「本机实测值」，因为「引用值」这个标签隐含的是
    「来源存在」，不是「本仓库复核过其测量口径」（CONTRIBUTING §2）。
    **不得把 `（计划基线，本次未独立核实）` 说成「已核实」。**

★ 「规范」与「实现」分开标注（错误模式 P8）。本主题已确证两处：
  · **W4A16 的 zero-point 修正形式**：官方文档里只有 **W8A8 激活侧**的「列和」版本
    （`J_a·B̂` = B̂ 沿归约轴的列和，可离线预计算）；
    而**主流 W4A16 kernel 逐元素做 `(q − z)`** —— vLLM 的 machete `Readme.md` 逐字写着
    `out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a`。
    **不得把「Σx 列和修正」写成 W4A16 的标准做法。**
  · **int32 累加器**：PTX 规范**在指令层面**把整数 `wmma.mma` 的累加器类型钉死为 `.s32`
    （不存在 int8 的 s16 累加变体）；而**本机 CPU 的 `_weight_int4pack_mm_for_cpu` /
    `_weight_int8pack_mm` 实测不是精确整数累加**（见下面 §5、§6 的实测）。
    **规范说指令，实测说这个具体实现——两者不是同一件事，写文档时必须分开。**

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（int4 打包布局、int32 溢出阈值、fp16 的 ulp/可表示整数上界、
  epilogue 的列和恒等式）**一律在 `verify.py` 里另写** —— 理由见 CONTRIBUTING §2
  「参考实现必须与被检验代码路径不同」。

★ **A6 约定（本主题自加）**：本文件里**每一条常量都被 `verify.py` 或某个编号程序的断言实际读过**。
  判据是机械的：逐个改掉一个值，`make run` 或 `make verify` **必须报错**。
  **因此本文件不收「没人查的抄件」** —— 纯引用、无法参与任何断言的长句
  （例如「ExLlamaV2 仓库已归档」「CUTLASS 的 Hopper 4-bit example 在 examples/55」）
  写在三篇文档的正文与「参考」里，**不落成常量**。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · **ExLlamaV2 源码注释 `"Use FP32 accumulator to avoid potential overflow since unscaled
    weights are in the range -128..127"` 的逐字原文** —— 计划基线里有这句，
    但本实现者本次**拿不到 exllamav2 的源码**（`raw.githubusercontent.com` 全部 404 / 超时，
    `github.com/.../blob/...` 返回 404）。**→ 不落成常量**；正文只按「计划基线、未独立核实」引用，
    并**不声称逐字读到**。（检索摘要能印证「ExLlamaV2 的内核有时用 FP32 累加防溢出」这一**定性**说法，
    但**不是那句注释**。）
  · **ExLlamaV2 源码的 `grep mma = 0` / 全部 `__hfma2` 这一计数** —— 同上，本次无法复核。
    **不落成常量。**

UNRESOLVED（来源冲突或本仓库无法判定，正文按「未解决」写，不落成常量）：
  · **`_weight_int8pack_mm` 的累加器**——实测**不是**精确整数累加（§6），
    但**它究竟落在 fp32 还是别的宽度，本仓库没有独立通道可以判定**。
    **只写下界（「不是精确整数累加」），不下断言说是哪一种。**
  · **fp16 累加的「顺序 unspecified」在本机 CPU 路径上的具体表现** —— 本机 CPU 的 `torch.mm`
    用 fp32 累加（§7），**碰不到 fp16 累加这条路径**，无从谈起。**不得外推。**
  · **CUTLASS Hopper 4-bit 快路径的四条硬约束**（只支持 TMA / scale 与 zero-point 类型 ≥ 2 字节 /
    group size 必须是 gemm-k 或 threadblock-k 的倍数 / narrow type 必须 K-major）——
    计划基线有，本次**未拿到 README 原文**。**不落成常量**，正文按「引用值、未核实」写。
"""

# ===========================================================================
# 0. 测量条件（三篇文档的 §5 直接引用这一节）
# ===========================================================================

# 本主题的三个编号程序在真 torch 上跑；下面五个解释器都跑过全部三个程序。
# 来源：（本机实测）`python3 -c 'import torch; print(torch.__version__)'`，2026-09-18。
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
UNCOVERED_MINORS = "2.9.x / 2.10.x / 2.12.x"
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
# 下面这批数字**冻结在哪个版本上**——凡本主题的「本机实测值」都出自这一个版本。
MEASURED_TORCH = "2.11.0"
MEASURED_PYTHON = "3.12.9"
# 解释器路径：`make run` 的自动探测在本机选中的就是它（通用位置候选里的
# `$$HOME/miniconda3/envs/*/bin/python` 之后的那个——见 Makefile 的候选顺序）。
MEASURED_INTERPRETER = "$HOME/miniconda3/bin/python"

# ★ 本主题的版本下限**不是依赖，是对齐的选择**（避免 P4「编机制从句」）。
#   来源：（本机实测）三个编号程序只用到 `_weight_int4pack_mm_for_cpu` / `_convert_weight_to_int4pack_for_cpu`
#   / `_weight_int8pack_mm` / 以及 fp8（`torch.float8_e4m3fn`）的 `torch.mm`——
#   **这四样在 2.6.0 上就全部存在**，五个解释器实测三个程序全部通过
#   （本主题**不需要** `torch.float8_e8m0fnu`，那个是 `numerical-formats` 的依赖）。
#   ⚠ 照抄 `numerical-formats` 的「≥2.8 是真依赖」那句理由会说出一句假话。
TORCH_MIN_FOR_THIS_TOPIC = (2, 8)
TORCH_MIN_REASON = ("对齐选择（不是依赖）：使 make run 在装多个 torch 的机器上选中与本域"
                    "其余主题同一个解释器，§5 的测量条件全域可比")

# ===========================================================================
# 1. int8×int8 的累加器算术（01-w8a8）
# ===========================================================================

# 来源：（规范/论文印值）int8 的值域是 [−128, 127]，于是单个乘积的**最大绝对值**有两个候选：
#   127×127 = 16129（两个操作数都在正端取到最大）；
#   (−128)×(−128) = 16384（两个都在负端）。
#   计划基线选 **16129** 作为溢出阈值的判据，本文件沿用并在 `verify.py` 的 A 节里
#   **把这三种取法都算出来**（两个候选值 + 各自的溢出阈值 K），因为
#   **阈值 K 依赖你取哪一个**——这是一个必须写出来的「判据」（P9）。
INT8_PRODUCT_MAX_POSPOS = 127 * 127        # = 16129
INT8_PRODUCT_MAX_NEGNEG = (-128) * (-128)  # = 16384

# 来源：（规范/论文印值）补码整数的上界。INT16_MAX = 2^15 − 1、INT32_MAX = 2^31 − 1。
INT16_MAX = 32767
INT32_MAX = 2147483647

# 来源：（推导，`verify.py` A.2 逐条算出）按 **16129** 计：
#   · int16 累加器：K=2 → 32258 ≤ 32767 不溢出；K=3 → 48387 > 32767 **溢出** ⇒ 首次溢出在 K=3。
#   · int32 累加器：K=133144 → 2147488... ≤ INT32_MAX 不溢出；K=133145 → **溢出**。
#     ⇒ 最后一次安全的 K = 133144，首次溢出的 K = 133145。
INT8_ACC_INT16_LAST_SAFE_K = 2
INT8_ACC_INT16_FIRST_OVERFLOW_K = 3
INT8_ACC_INT32_LAST_SAFE_K = 133144
INT8_ACC_INT32_FIRST_OVERFLOW_K = 133145

# 来源：（推导）现实 K 下 int32 的最坏占用 = K × 16129 ÷ INT32_MAX。
#   判据：**假定全部乘积同号且都取到 16129**——这是上界，不是典型值。
#   K=4096 → 3.08%；8192 → 6.15%；16384 → 12.31%（`verify.py` A.2 逐条核）。
INT8_ACC_IDEAL_K_FILL_PCT = {4096: 3.08, 8192: 6.15, 16384: 12.31}

# 来源：（规范/论文印值）fp16 = S1E5M10，11 位有效位 ⇒ **能精确表示的整数上界是 2^11 = 2048**。
#   判据：2048 = 2^11 是「尾数恰好用完」的那个点；2049 起需要 11 位尾数中的第 12 位。
#   ⇒ **int32 累加器必须在 epilogue 先转 fp32 再乘 scale**；直接塞进 fp16 会丢整数位。
FP16_EXACT_INT_MAX = 2048

# 来源：（规范/论文印值，四条独立通道）整数矩阵乘的累加器**在指令层面**就是 32 位：
#   · PTX 规范：整数 `wmma.mma` 的 `.ctype` 与 `.dtype` **必须是 `.s32`**——
#     **指令集里根本不存在 int8 的 s16 累加变体**；
#   · cuBLAS：`CUBLAS_COMPUTE_32I`；
#   · cuSPARSELt：`CUSPARSE_COMPUTE_32I`；
#   · DP4A 的文档：`"with 32-bit integer accumulation"`。
#   ⚠ 这是**规范侧**的断言；本机 CPU 的那两个算子**实测不是精确整数累加**（见 §5/§6），
#     两者**不是同一件事**，写文档必须分开（P8）。
PTX_INT_MMA_ACC_CTYPE = ".s32"
CUBLAS_INT_COMPUTE_TYPE = "CUBLAS_COMPUTE_32I"
CUSPARSELT_INT_COMPUTE_TYPE = "CUSPARSE_COMPUTE_32I"
DP4A_ACCUM_QUOTE = "with 32-bit integer accumulation"

# 来源：（规范/论文印值）W8A8 的标准 epilogue 形式（scale/zero-point 在 GEMM 内融合应用）：
#   `D = s_a·s_b·(Â·B̂ − z_a·J_a·B̂)`，其中 `J_a·B̂` **等于 B̂ 沿归约轴的列和**，
#   与 batch 无关、可离线预计算；per-token 的 `z_a` 退化成 `z_a`（列向量）与
#   `J_a·B̂`（行向量）的**外积 / rank-1 更新**。
#   ⚠ **这是 W8A8（激活侧 zero-point）的版本，不是 W4A16 的。** 见 §4 的 P8 条目。
EPILOGUE_W8A8_FORM = "D = s_a*s_b*(A_hat.B_hat - z_a*(J_a.B_hat))"
EPILOGUE_COLSUM_IS_BATCH_INVARIANT = True
EPILOGUE_PERTOKEN_IS_RANK1 = True

# 来源：（规范/论文印值）CUTLASS 的 epilogue 是**独立阶段**；`LinearCombination` 即
#   `D = alpha·accumulator + beta·source`；per-channel 向量 scale 走 `EpilogueWithBroadcast`。
CUTLASS_LINEAR_COMBINATION = "D = alpha*accumulator + beta*source"
CUTLASS_EPILOGUE_BROADCAST_OP = "EpilogueWithBroadcast"

# 来源：（规范/论文印值）vLLM 的 W8A8 epilogue 文档：`scale_a` = per-tensor 或 **per-token（列向量）**；
#   `scale_b` = per-tensor 或 **per-channel（行向量）**；`bias` **永远是 per-channel（行向量）**。4 个变体。
VLLM_W8A8_SCALE_A_AXES = ("per-tensor", "per-token")
VLLM_W8A8_SCALE_B_AXES = ("per-tensor", "per-channel")
VLLM_W8A8_BIAS_AXIS = "per-channel"
VLLM_W8A8_VARIANTS = 4

# ===========================================================================
# 2. `_weight_int4pack_mm_for_cpu` 的完整调用契约（02-w4a16-packing）
# ===========================================================================

# 来源：（本机实测）torch 2.11.0。转换算子只做 `输入 & 0xF`，而 mm 算子解包时**减 8**，
#   所以喂给转换器的必须是**有符号值 + 8**（uint8 nibble 0..15）。
INT4_PACK_ZERO_POINT = 8
# 来源：（本机实测）torch 2.11.0：`_convert_weight_to_int4pack_for_cpu(w, 2)` 的第二个实参。
#   实测 innerKTiles ∈ {1, 2, 4, 8} 对同一个 [64, 256] 权重给出**逐字节相同**的输出
#   （`0x88` 基线全等 ⇒ 见编号程序 02 的 A.4）。文档里沿用计划基线的 `2`。
INT4_PACK_INNER_K_TILES = 2

# 来源：（本机实测）torch 2.11.0，逐个触发异常并抄下原文。
#   `_convert_weight_to_int4pack_cpu : expect N to be dividable by 16`
#   `_weight_int4pack_mm_cpu: expect qGroupSize to be 32, 64, 128 or 256, got 48`
#   `_weight_int4pack_mm_cpu: expect K to be divisible by qGroupSize, got K:192, qGroupSize:128`
#   `_weight_int4pack_mm_cpu: expect qScaleAndZeros to be 3d tensor with sizes [:, N, 2]`
#   `_weight_int4pack_mm_cpu : expect B to be uint8 tensor.`
INT4_PACK_N_MULTIPLE = 16
INT4_PACK_GROUP_SIZES = (32, 64, 128, 256)
# ★ 异常原文里的 `[:, N, 2]` 是把**实际 N** 代进去打印的（实测 N=32 时打印 `[:, 32, 2]`）。
#   ⇒ 文档引用这句话时**不能把 N 当成字面量**（这是 P9 的形状：同一句话里两个 N 不是同一个量）。
INT4_PACK_QSZ_SHAPE_MSG = "expect qScaleAndZeros to be 3d tensor with sizes [:,%d,2]"

# 来源：（本机实测）torch 2.11.0，逐字节反解 `_convert_weight_to_int4pack_for_cpu(v+8, 2)` 的输出。
#   判据：全矩阵 (N=64, K=256) **每个字节的低/高半字节都逐位对上**，且覆盖恰好 N×K 个元素。
#   `wp` 的形状是 `[N, K//2]`，下标 (i, j) 处的字节装两个元素：
#     低半字节 ← n = 16·(i//16) + 2·(j%8) + 0,  k = (K//16)·(i%16) + (j//8)
#     高半字节 ← n = 16·(i//16) + 2·(j%8) + 1,  k 同上
#   ★ **低/高半字节沿 N（输出通道）配对，不是沿 K 相邻配对**（后者在 N=64,K=256 时
#     需要 n 上到 127，**直接越界**——这就是它不可能的那条证明）。
INT4_PACK_LAYOUT_N = "n = 16*(i//16) + 2*(j%8) + off"
INT4_PACK_LAYOUT_K = "k = (K//16)*(i%16) + (j//8)"
INT4_PACK_LAYOUT_PAIRS_ALONG = "N"
INT4_PACK_LAYOUT_NAIVE_PAIRS_ALONG = "K"
# 解包数学：`q` 是无符号 nibble（0..15），`dequant = scale·(q − 8) + zero_point`。
INT4_PACK_DEQUANT_FORM = "scale*(q-8) + zero_point"

# --- 2.x 本机实测基线（与 02-w4a16-packing.md §5 逐行一致）------------------
# 来源：（本机实测）torch 2.11.0 / Python 3.12.9 / Apple M1 Pro，2026-09-18。
#   判据（**把参照钉死，这是 maxdiff = 0 能成立的全部原因**）：
#     · `x` = `torch.randint(-4, 5, (4, K)).float()` —— **整数值的 fp32**；
#     · `scale = 1.0`、`zero_point = 0.0` ⇒ 反量化后的权重是**小整数**；
#     · 参照 = `x.long() @ ((v+8)&0xF).t().long() - 8·x.long().sum(1)` 的 **int64 精确值**。
#   此时每一项乘积与每个部分和都是**绝对值 < 2^24 的整数**，在 fp32 里**逐位精确**，
#   于是**任何累加顺序都给出同一个结果** ⇒ maxdiff 恒为 0，与形状无关。
#   ⚠ **换成 `x ~ N(0,1)`、参照用 fp32 逐组 matmul，maxdiff 就是 ~1e-6 而不是 0**
#     （编号程序 02 的 A 节把两种判据都跑出来，说明「0」不是「算子更准」，
#     而是**这一组输入让参照与实现同时退化成同一个整数运算**）。
INT4PACK_EXACT_REF_MAXDIFF = 0.0
# 不 +8 的反证：误差**逐元素恰好 ±8**（因为 `(v&0xF)−8` 与 `v` 对每个 v 都差 8 的整数倍符号翻转）。
#   同一个整数参照下，maxdiff 由「x 与 scale」决定 —— 值是**setup 相关的**，不是常量。
#   判据：下面每个形状的 x 与 v 都由 `torch.manual_seed(20260918)` 固定。
INT4PACK_NO_PLUS8_MEASURED = {   # (N, K, gs): (maxdiff_+8, maxdiff_no_plus8)
    # ★ 判据：**每个形状独立播种 `torch.manual_seed(K)`**（见下方注释），
    #   否则这些数字会随调用顺序漂移——第一版就漂过。
    (16, 32, 32): (0.0, 263.0),
    (64, 128, 32): (0.0, 656.0),
    (64, 128, 64): (0.0, 656.0),
    (128, 256, 128): (0.0, 1048.0),
    (256, 512, 256): (0.0, 1648.0),
    (64, 1024, 256): (0.0, 2209.0),
}
# 计划基线写的是「不 +8 时 maxdiff = 3.389e+01」——那是**控制者自己那组 x/scale** 的值。
# 本文件的这几条用的是整数值 x（更大的量级），所以绝对值更大。
# **两者都是对的，但不是同一个量**（P9）：判据不同 ⇒ 不得并列。
INT4PACK_NO_PLUS8_PLAN_BASELINE = 3.389e+01

# ===========================================================================
# 3. `_weight_int8pack_mm`（01-w8a8）
# ===========================================================================

# 来源：（本机实测）torch 2.11.0，`torch.ops.aten._weight_int8pack_mm(x, w_int8, scales)`。
#   判据：`w` = `torch.randint(-128, 128, (N, K), dtype=torch.int8)`、`x ~ N(0,1)`、
#   `scales = torch.rand(N)*0.02 + 0.005`（**per-N 的 1D 向量，不是 `[N,1]` 的 2D**），
#   参照 = `x @ (w.float()*scales.unsqueeze(1)).t()`（**逐元素反量化后另走一条 matmul**）；
#   ★ **每个形状独立播种 `torch.manual_seed(K)`**（K 在表里唯一）——这样数字不依赖调用顺序。
#     ⚠ 第一版曾用「一个种子按顺序取随机数」，结果同一个形状在 `specs_kernels.py` 与
#       编号程序里对不上（程序先跑了别的实验，RNG 状态不同）。**这是一个真实的坑。**
#   ★ 这张表本身就是计划要求的那句「maxdiff 随 setup 变化」的证据：
#     **同一张表里它就跨了三个数量级**（K=32 的 1.9e-06 → K=131072 的 1.7e-03）。
#   ⚠ **不得给单一固定值**。写文档用「与浮点参照一致到 ~1e-3 量级（取决于 K 与 scale）」。
INT8PACK_MEASURED = {           # (M, K, N): maxdiff
    (4, 32, 32): 1.907349e-06,
    (4, 128, 64): 2.288818e-05,
    (4, 256, 128): 3.051758e-05,
    (8, 512, 256): 6.866455e-05,
    (4, 1024, 64): 1.220703e-04,
    (2, 8192, 32): 6.866455e-04,
    (1, 131072, 16): 1.739502e-03,
}
INT8PACK_MEASURED_ORDER = "maxdiff 随 K 单调增长（表内三个数量级）"

# ★ 判据：**「是不是精确整数累加」这个问题，只有把精确和推过 2^24 才能回答。**
#   判据（本机实测，torch 2.11.0）：取 `x = randint(1,5,(1,K)).float()`（**全正、不抵消**）、
#   `w = full((4,K), 127, int8)`、`scales = ones(4)`，比较对象是 **`float32(int64 精确和)`**。
#   · 精确和 ≤ 2^24 时，任何累加顺序在 fp32 里都精确 ⇒ 差为 0，**这一档区分不了什么**；
#   · 精确和 > 2^24 时差距出现并随 K 增长：
#       K=2^18（和 2.08e7）：0；K=2^20（和 3.33e8）：5.5072e+05（相对 1.655e-03）；
#       K=2^21（和 6.66e8）：3.1724e+06（相对 4.765e-03）。
#   ★ 3.33e8 ≪ INT32_MAX(2.147e9) —— **一个 int32 累加器在 K=2^20 上仍然完全精确**，
#     所以这个偏差**排除了「精确整数累加」**。
#   ⚠ **只排除得掉这一种**：具体是 fp32 还是别的宽度，本仓库没有独立通道，**不得断言**。
INT8PACK_ACCUM_IS_EXACT_INT = False
INT8PACK_ACC_PROBE = {          # K: (精确和, |out - float32(精确和)|, 相对偏差)
    # ★ 判据：**每个 K 独立播种 `torch.manual_seed(5)`**，再 `randint(1, 5, (1, K))`。
    (1 << 16): (2.08326e+07, 0.0, 0.0),          # 和已 > 2^24，但**偏差仍是 0**——
    #   ⇒ 这一档说明「超过 2^24」只是**必要条件**，不是「必然出偏差」的充分条件。
    (1 << 18): (8.32369e+07, 4.8000e+01, 5.767e-07),
    (1 << 20): (3.32851e+08, 5.5072e+05, 1.655e-03),
    (1 << 21): (6.65803e+08, 3.1724e+06, 4.765e-03),
    # (1 << 22): (1.331796e+09, 8.4164e+06, 6.320e-03)  # 本机也测过，未放进上表
}
INT8PACK_ACC_PROBE_X_MAX = 4      # x = randint(1, 5, ...) ⇒ 取值 1..4
INT8PACK_ACC_PROBE_W = 127
# 计划基线里研究员测的「K=2^20 偏差 0.039%」——**与上表的 1.655e-03 不是同一个量**：
# 他的 0.039% 是另一种相对口径（分母不同）。**判据没写出来的数字不得并列**（P9）。
INT8PACK_ACC_PLAN_BASELINE_REL = 0.039e-2

# ===========================================================================
# 4. fp8 的 `torch.mm`（01-w8a8 与 03-kernel-design 共用）
# ===========================================================================

# 来源：（本机实测）torch 2.11.0，CPU-only。`torch.mm(a_fp8, b_fp8)` **在 CPU 上可用**，
#   输出**仍是 fp8**（`torch.float8_e4m3fn`），不是 fp32。
#   ⚠ 但 fp8 的**逐元素算术不可用**：`f8([448.]) + f8([448.])` 抛
#     `NotImplementedError: "add_stub" not implemented for 'Float8_e4m3fn'`（2.6.0 上是 RuntimeError）。
#     ⇒ **本机 CPU 上碰得到 fp8 的唯一入口是 `torch.mm`（与 `.to()` 转换）。**
FP8_MM_OUTPUT_DTYPE = "torch.float8_e4m3fn"
FP8_MM_AVAILABLE_ON_CPU = True
FP8_ELEMENTWISE_ADD_AVAILABLE = False

# 来源：（本机实测）torch 2.11.0。判据：`a, b ~ N(0,1)` 的 **256×256 fp32**，
#   先 `.to(float8_e4m3fn)` 再 `torch.mm`，比较对象 = **把两个已量化输入升到 fp64 精确累加、
#   再把结果舍回 e4m3**。
#   · **逐元素全等比例 = 1.0000** ⇒ 累加不丢精度、**只有输出被舍入一次**。
#   · 输出 vs 量化输入后的精确积的**平均相对误差 = 0.022674**；
#     输入 cast 到 e4m3 的**逐元素平均相对误差 = 0.024157** ⇒ 两者同量级
#     ⇒ **误差来自量化，不是来自累加**。
#   · 全链路（vs fp64 原矩阵）平均相对误差 = 0.323179；**这不是上面那个量**——
#     矩阵乘的输出在随机符号下有大量接近 0 的元素，相对误差在那里被放大。
#     **判据不同 ⇒ 0.323 与 0.024 不得并列成「误差多大」的两种说法**（P9）。
#   · 对照：同矩阵的 fp16 mm 是 0.002525。
FP8_MM_MEASURED = {
    "out_exact_fraction": 1.0,
    "rel_vs_quantized_input": 0.022674,
    "input_cast_rel": 0.024157,
    "rel_full_chain": 0.323179,
    "fp16_same_matrix_rel": 0.002525,
    "shape": (256, 256),
}
FP8_E4M3_MAX = 448.0
# ★ 由「输出要被舍回 e4m3」直接推出的溢出陷阱：
#   累加器不溢出（它是高精度的），但**输出 cast 回 e4m3 会溢出成 nan**。
#   判据：`x = [448×8, −448×7, 100]`，精确和 = **548 > 448** ⇒ `torch.mm` 输出 `nan`。
FP8_OUTPUT_OVERFLOW_EXACT_SUM = 548.0
FP8_OUTPUT_OVERFLOW_IS_NAN = True
# ★ 排除「累加在 fp8 里做」的判据：`[448]×4 + [−448]×3` 如果按 fp8 累加，
#   第一个部分和就是 448+448 = 896 > 448（e4m3 无 Inf，越界即 NaN），
#   结果**不可能**是 448。实测 = 448.0（精确和）。
FP8_ACC_PROBE = [([448.0] * 4 + [-448.0] * 3, 448.0),
                 ([448.0, 448.0, -448.0, -448.0], 0.0)]

# ===========================================================================
# 5. fp16 累加（03-kernel-design）
# ===========================================================================

# 来源：（规范/论文印值）fp16 = S1E5M10：最大有限值 **65504**，最小正规数 2^-14。
FP16_MAX = 65504.0
# 来源：（推导 + 本机实测）只有 10 位显式尾数 ⇒ 区间 [1024, 2048) 的 ulp = **1**、
#   [2048, 4096) 的 ulp = **2**。**⇒ 把 1.0 加到 2048.0 上会直接丢失。**
#   `verify.py` 的 A.4 用 `struct` 的 `'e'`（半精度）**逐点算出这两段 ulp**，
#   编号程序 03 在真 torch 上核 `float16(2048.0) + float16(1.0) == 2048.0`。
FP16_ULP_LOW_BINADE = 1.0       # [1024, 2048)
FP16_ULP_HIGH_BINADE = 2.0      # [2048, 4096)
FP16_ULP_LOW_RANGE = (1024.0, 2048.0)
FP16_ULP_HIGH_RANGE = (2048.0, 4096.0)

# ★ 本机 CPU 的 fp16 `torch.mm` **用 fp32 累加**（实测，torch 2.11.0）。
#   判据（这条判据是**设计出来的、能区分开两种解释的**）：
#     · `[2048, 1, 1, …, 1]`（K=2049 个元素）@ ones → **4096.0**。
#       若累加在 fp16 里做，2048+1 先舍成 2048，一路加下去仍是 2048；
#       实测 4096 = 2048 + 2048×1，**只有 fp32 累加能得到它**。
#     · `[2048, 1, 1]`（K=3）@ ones → **2050.0**。
#       纯 fp16 顺序累加会得 2048（第一步 2048+1→2048）；**2050 排除了 fp16 累加**。
#   ⚠ **不得外推到 GPU**：PTX 规范里 `.ctype` 与 `.dtype` 都是 `.f16` 的累加变体**确实存在**，
#     只是本机 CPU 路径不走它。
FP16_MM_CPU_ACCUM = "fp32"
FP16_MM_CPU_PROBE = {2049: 4096.0, 3: 2050.0}
FP16_MM_CPU_PROBE_FP16_WOULD_GIVE = {2049: 2048.0, 3: 2048.0}
# 溢出：30000+30000 = 60000 ≤ 65504（可表示）；40000+40000 = 80000 > 65504 → inf。
FP16_MM_OVERFLOW_INPUT = 40000.0
FP16_MM_OK_INPUT = 30000.0
FP16_MM_OK_EXPECTED = 60000.0

# 来源：（规范/论文印值）PTX 对 fp16 累加的明文：`.ctype` 与 `.dtype` 都是 `.f16` 时
#   `"accumulation is performed with at least half precision"`，
#   且 `"The accumulation order, rounding and handling of subnormal inputs is unspecified"`。
#   ⇒ **指令层面 fp16 累加变体确实存在**，且**顺序不作保证**——本机的 CPU 行为**不构成**反例。
PTX_F16_ACC_QUOTE = "accumulation is performed with at least half precision"
PTX_F16_ACC_ORDER_QUOTE = ("The accumulation order, rounding and handling of subnormal inputs "
                           "is unspecified")

# 来源：（计划基线，本次未独立核实）fp16 累加数值行为的形式化研究：
#   Fasi / Higham / Mikaitis / Pranesh, **PeerJ Computer Science 7:e330 (2021)**：
#   fp16 乘积精确、加法在 binary32 中以 **round-towards-zero** 做、
#   内部累加器有效位宽 **27 bit**、加数按**绝对值从大到小**累加、部分和不归一化 ⇒ **点积非单调**。
FP16_FORMAL_STUDY = "PeerJ CS 7:e330 (2021)"
FP16_FORMAL_STUDY_INTERNAL_BITS = 27

# 来源：（规范/论文印值）cuBLAS 官方对 split-K 的说明：compute type 精度高于 output type 时，
#   split-K 的分块和可能产生**本不该有的中间溢出**，由
#   `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` 控制。
CUBLAS_SPLITK_OVERFLOW_ENV = "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION"

# ===========================================================================
# 6. 主流量化 GEMM kernel 的设计要点（03-kernel-design，**全部是引用值**）
# ===========================================================================

# 来源：（规范/论文印值）Marlin: **arXiv:2408.11743**（*MARLIN: Mixed-Precision Auto-Regressive
#   Parallel Inference on Large Language Models*），PPoPP'25。
#   本实现者**逐字读到了 arXiv abstract**（urllib 直取，43783 字节）：
#     · "batchsizes up to 16-32 can be supported with close to **maximum (4×)** quantization speedup,
#        and larger batchsizes up to 64-128 with gradually decreasing, but still significant,
#        acceleration"；
#     · "end-to-end LLM inference speedups (of up to **2.8×**) when integrated with the popular
#        vLLM serving engine"。
#   ⚠ 「A10 上 batch 128 降到约 **1.5×**」这一条**不在 abstract 里**——
#     它来自检索摘要（**未逐字读到原文**），按引用值标注。
MARLIN_ARXIV = "arXiv:2408.11743"
MARLIN_MAX_SPEEDUP = "4x"
MARLIN_BATCH_SWEET = (16, 32)
MARLIN_BATCH_DECAY = (64, 128)
MARLIN_END_TO_END_VLLM = 2.8
MARLIN_BATCH128_A10 = 1.5       # （计划基线/检索摘要，未逐字核实）
MARLIN_AUTHOR = "IST-DASLab + Neural Magic（共同作者）"

# 来源：（源码直取）vLLM 的 machete `Readme.md`（vllm 0.18.1 源码镜像，
#   `csrc/quantization/machete/Readme.md`，urllib 直取 **2672 字节**，逐字读过）。
#   逐字原文（这是本主题最重要的一条 P8 证据）：
#     ```python
#     scale_type = w_s.dtype
#     compute_type = a.dtype
#     out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a
#     ```
#     "Where `w_q` is a quantized weight matrix, `w_s` is the quantization scales, and
#      `w_z` is the quantization zeropoints."
#     NOTE: "`w_z` is added after the scales so we can use FMA operations, but this means they
#      must have the scales pre-applied if the supplied zeropoints assume that they will be
#      subtracted before the scales are applied."
#   ⇒ **Machete（W4A16）是逐元素 `(q·s − z)`，不是「Σx 列和修正」。**
#     同 Readme 另一句："Machete is a spiritual successor to the Marlin kernel but optimized
#     for Hopper architectures and based on Cutlass."
MACHETE_DEQUANT_FORM = "out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a"
MACHETE_IS_ZERO_POINT_ELEMENTWISE = True
MACHETE_ZP_ORDER_NOTE = "w_z is added after the scales so we can use FMA operations"
MACHETE_AUTHOR = "Neural Magic"
MACHETE_README_BYTES = 2672
# ⚠ **Machete 没有 arXiv 论文。上面那个 Readme 里没有任何 arXiv 编号。**
#   计划基线明令「不得编造论文号」，`verify.py` 的 A.6 把这一条做成可断言的常量。
MACHETE_HAS_ARXIV = False
# ⚠ 研究员**否证了**「Machete 用混合 tile / SIMT 双路径」这个说法。
#   本次独立复核：上面那个 Readme 与 Red Hat 的官方介绍里**都没有 SIMT 字样**
#   （Red Hat 文的机制说的是 TMA + WGMMA + prepacking + warp specialization 之外的东西按 tile 走）。
MACHETE_USES_SIMT_DUAL_PATH = False

# 来源：（源码直取 + 检索摘要）Red Hat Developer 2024-10-14 的 Machete 官方介绍
#   （urllib 直取 148111 字节，逐字读过）。逐字原文：
#     "**Machete is Neural Magic's latest advancement in mixed-input quantization performance.**
#      This kernel is the spiritual successor to the Marlin kernels created by Elias Frantar and
#      integrated into vLLM by Neural Magic."
#     "For standard data types (16, and 32-bit), NVIDIA provides an efficient ldmatrix instruction …
#      **However, this instruction isn't available for 4-bit types.**"
#     "…we can perform 128-bit loads—the widest shared load instruction currently available on CUDA devices."
REDHAT_MACHETE_ARTICLE_BYTES = 148111
MACHETE_NO_4BIT_LDMATRIX = True

# 来源：（计划基线，本次未独立核实）ExLlamaV2 的设计：反量化融进内积、
#   **走 SIMT 不走 tensor core**（`q_gemm_kernel.cuh` 里 `mma` 零命中、全部 `__hfma2`）、
#   group scale 融进点积，且**特意用 FP32 累加防溢出**。
#   ⚠ 本次**拿不到该源码**（raw.githubusercontent 全程 404/超时），**逐字原文未能复核**；
#     只在检索摘要里得到**定性**印证（「有时用 FP32 防溢出」）。
#   ⇒ 按引用值写，**且不得声称逐字读到**。
EXLLAMA_V2_PATH_USES_TENSOR_CORE = False
EXLLAMA_V2_ACCUM_TYPE = "fp32"
EXLLAMA_V2_REPO_ARCHIVED = True
EXLLAMA_V2_SOURCE_VERIFIED_THIS_RUN = False

# 来源：（计划基线，本次未独立核实）CUTLASS 的 int4 快路径：
#   Ampere(Sm80) 有 `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite`（累加器固定 int32）；
#   Hopper 的 4-bit 快路径在 `examples/55_hopper_mixed_dtype_gemm/`。
#   本次只**独立证实了「CUTLASS 3.5.1 example #55 是 Machete 的起点」**（Red Hat 文逐字）。
CUTLASS_INT4_MMA = "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite"
CUTLASS_HOPPER_INT4_EXAMPLE = "examples/55_hopper_mixed_dtype_gemm/"
CUTLASS_MACHETE_STARTING_POINT = "CUTLASS 3.5.1 example #55"

# 来源：（计划基线，本次未独立核实）量化 × 2:4 稀疏：
#   · A100 白皮书峰值表：`Peak INT8 TC: 624 | 1,248 TOPS (sparsity)`；
#     `Peak INT4 TC: 1,248 | 2,496 TOPS (sparsity)` ⇒ **稀疏在每个精度上都恰好 2×**，
#     INT4 相对 INT8 再 2×。
#   · 理论：sparse INT8 = 2×（稀疏）× 2×（INT8 vs FP16）= **4× vs dense FP16**；
#     **dense INT4 也是 4×** ⇒ 两者理论加速相同。
#   · Sparse-Marlin（**IST-DASLab/Sparse-Marlin**，不是 Neural Magic）：
#     Readme 逐字 "provides near-optimal speedups of around **5.3x**, while being more robust to
#     larger batch sizes"、"Marlin is able to preserve close-to-ideal 4x performance up to a batch
#     size of 16, Sparse-Marlin … batch size twice as large (i.e., 32)"、
#     "consistently deliver 2x over the dense FP16 counterpart"。
#     ⚠ 「dense Marlin 之上再快最多 **65%**」与「端到端最多 **3.2×**」**不在 Readme 里**——
#       来自检索摘要，按引用值标注、未逐字核实。
#   · cuSPARSELt 的稀疏格式按类型不同：`e2m1` 用 **paired 4:8**；
#     `half/bfloat16/int/int8/e4m3/e5m2` 用 **2:4**；`float` 用 **1:2**。
A100_INT8_TC_TOPS = (624, 1248)          # (dense, with sparsity)
A100_INT4_TC_TOPS = (1248, 2496)
A100_SPARSITY_FACTOR = 2
SPARSE_INT8_THEORETICAL_VS_DENSE_FP16 = 4
DENSE_INT4_THEORETICAL_VS_DENSE_FP16 = 4
SPARSE_MARLIN_REPO = "IST-DASLab/Sparse-Marlin"
SPARSE_MARLIN_SPEEDUP = 5.3
SPARSE_MARLIN_BATCH = 32
SPARSE_MARLIN_OVER_DENSE_FP16_LARGE_BATCH = 2
SPARSE_MARLIN_DELTA_VS_DENSE_MARLIN_PCT = 65     # （检索摘要，未逐字核实）
SPARSE_MARLIN_END_TO_END = 3.2                   # （检索摘要，未逐字核实）
CUSPARSELT_SPARSITY_BY_TYPE = {"e2m1": "paired 4:8",
                               "half/bfloat16/int/int8/e4m3/e5m2": "2:4",
                               "float": "1:2"}

# 来源：（计划基线，本次未独立核实）W4A16 是**访存受限**：
#   独立反量化 pass 的带宽预算 = **读 K·N/2 字节（int4）但写 K·N·2 字节（fp16）——写是读的 4 倍**，
#   因此独立 kernel 被**写侧带宽**卡死；融合进 GEMM 后反量化发生在寄存器里，
#   fp16 权重永远不落回 HBM；**收益随 batch 增大而衰减**。
W4A16_DECODE_BYTES_PER_ELEM = 0.5      # int4 载荷
W4A16_DEQUANT_WRITE_BYTES_PER_ELEM = 2.0   # fp16 写回
W4A16_SEPARATE_PASS_WRITE_OVER_READ = 4    # = 2.0 / 0.5

# 来源：（本机实测 + 规范/论文印值）**W4A16 的 zero-point 用法**：
#   · 本机实测（torch 2.11.0，判据：`zp ~ randint(-3,4)`、`q = (v+8)&0xF`，
#     `scale` 与 `zp` 的形状都是 `[K//gs, N]`）：
#     `_weight_int4pack_mm_for_cpu` 的输出与 **`s·(q−8)+zp` 逐元素**形式的差值 = 7.629395e-05，
#     而与「zp 每组只加一次」形式的差值 = 2.08e+02 ⇒ **逐元素 / 非列和**。
#   · 源码直取（同 §6 的 machete Readme）：`out = (w_q*w_s − w_z) @ a` —— 也是**逐元素**。
#   ⇒ **不得把「Σx 列和修正」写成 W4A16 的标准做法**：那个形式在官方文档里只有 W8A8 的
#     **激活侧**版本（见 §1 的 `EPILOGUE_W8A8_FORM`）。
W4A16_ZP_ELEMENTWISE_MAXDIFF = 3.814697e-05
W4A16_ZP_ONCE_PER_GROUP_MAXDIFF = 1.48e+02
W4A16_ZP_IS_ELEMENTWISE = True
W4A16_OFFICIAL_DOC_COLSUM_FORM_IS_W8A8_ONLY = True

# ===========================================================================
# 7. `_dyn_quant_pack_4bit_weight`（**不写进文档**，见 §3 残留 3）
# ===========================================================================

# 来源：（本机实测）torch 2.11.0 可调用；**2.6.0 没有**（`hasattr` 为 False）。
#   ⚠ 排布约定未定（研究员试两种都误差很大）⇒ **不得写进文档**。
#   本文件把这个「不写」的决定本身落成常量，由编号程序 02 的 F 节断言它。
DYN_QUANT_IN_DOCS = False
DYN_QUANT_AVAILABLE_IN = ("2.7.1", "2.8.0", "2.11.0", "2.13.0")
DYN_QUANT_GROUP_SIZE_MULTIPLE = 32


# ===========================================================================
# 8. 版本断点（**本主题实测到的三处**；跨版本对拍是本主题的必做项）
# ===========================================================================
#
# ★ 对拍方式：**单程序**显式指定解释器（见 Makefile 末尾）。
#   五个解释器、三个程序，全部跑过。下面三处是**行为真的变了**的地方——
#   其余结论（int4pack 的 maxdiff=0、逐元素 zp、不 +8 的错、fp16 的 CPU 累加宽度）
#   五版本**完全一致**。

# --- 8.1 `_weight_int8pack_mm` 的对拍值**随 torch 版本变**，但**变的是参照那一侧** ---
# 来源：（本机实测）2026-09-18，五版本对拍。**同一种子**下：
#   · `_weight_int8pack_mm` 的输出**五个版本逐位相同**（`out.sum` 到 12 位小数一致）；
#   · 参照 `x @ (w.float()*sc.unsqueeze(1)).t()` 的 `ref.sum` 变了：
#       **2.6.0 / 2.7.1** → 20.066647015512
#       **2.8.0 / 2.11.0 / 2.13.0** → 20.066647142172
#   ⇒ 断点在 **2.7.1 与 2.8.0 之间**，落在 `torch.matmul` 的 fp32 累加顺序上。
#   ★ **这是本主题最值得写出来的一条**：「maxdiff 随 setup 变化」这句话，
#     连「换个 torch 版本」都算换 setup —— 而被测算子本身没变。
INT8PACK_REF_IS_VERSION_DEPENDENT = True
INT8PACK_OUT_IS_VERSION_STABLE = True
INT8PACK_REF_SPLIT_BREAK = ("2.7.1", "2.8.0")          # 断点落在这一对之间
INT8PACK_REF_SUM_OLD = 20.066647015512                 # ≤ 2.7.1
INT8PACK_REF_SUM_NEW = 20.066647142172                 # ≥ 2.8.0
INT8PACK_OUT_SUM_STABLE = 20.066649161279              # 五版本一致
# ★ 把参照**再拆一步**，定位断点到底在哪一半（判据：同一组输入，分别看两步）：
#   · `wq = w.float()*sc.unsqueeze(1)` 的 `.sum()` = **3.714967675507** —— **五版本一致**；
#   · `(x @ wq.t())` 的 `.sum()` —— **2.6.0/2.7.1 是 20.066647015512、2.8.0 起是 20.066647142172**。
#   ⇒ 断点**就在 `torch.matmul` 本身**（它的 fp32 累加顺序），**不在反量化那一步**。
#   **这一步拆分是必需的**：不拆就只能说「参照那条路径变了」，拆了才能说清是哪一步。
INT8PACK_REF_WQ_SUM_STABLE = 3.714967675507
INT8PACK_REF_BREAK_IS_IN = "torch.matmul 的 fp32 累加" 
# 受影响的三个形状（**同一种子、同一输入，只是参照的累加顺序不同**）：
#   判据：`INT8PACK_MEASURED` 记的是 **≥2.8.0** 那一档。
INT8PACK_MEASURED_OLD_REGIME = {   # (M, K, N): maxdiff @ torch ≤ 2.7.1
    (4, 32, 32): 2.861023e-06,
    (2, 8192, 32): 4.730225e-04,
    (1, 131072, 16): 2.304077e-03,
}
INT8PACK_OLD_REGIME_VERSIONS = ("2.6.0", "2.7.1")
INT8PACK_NEW_REGIME_VERSIONS = ("2.8.0", "2.11.0", "2.13.0")

# --- 8.2 `K % qGroupSize` 这条校验：**2.6.0 还没有** ---
# 来源：（本机实测）同一段调用在五个版本上：
#   2.6.0 → **NO ERROR**（静默跑过）；2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 → 抛
#   `_weight_int4pack_mm_cpu: expect K to be divisible by qGroupSize, got K:192, qGroupSize:128`。
#   ⇒ 断点在 **2.6.0 与 2.7.1 之间**。
# ⚠ 其余四条校验（N 的倍数、gs 取值、qsz 的 ndim、w 的 dtype）**五版本都在**。
INT4PACK_K_DIVISIBLE_CHECK_SINCE = "2.7.1"
INT4PACK_K_DIVISIBLE_CHECK_ABSENT_IN = ("2.6.0",)
INT4PACK_K_DIVISIBLE_MSG = ("expect K to be divisible by qGroupSize, got K:192, "
                            "qGroupSize:128")

# --- 8.3 fp8 输出越界：**2.13.0 起饱和，之前是 NaN** ---
# 来源：（本机实测）五版本对拍，输入完全相同（`x = [448×8, −448×7, 100]`，精确和 548）：
#   2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 → **nan**；2.13.0 → **448.0**（饱和）。
#   ⇒ 断点在 **2.11.0 与 2.13.0 之间**。
# ★ 这与同域 `numerical-formats` 已记录的断点**是同一次变更**
#   （「2.13 起 e4m3fn 越界一律饱和」，动因 pytorch PR #178817）。
#   **本主题独立复现了它，且是在一个不同的算子上复现的**（那里是 `.to()` 转换，
#   这里是 `torch.mm` 的**输出 cast**）。
FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0")
FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS = ("2.13.0",)
FP8_OUTPUT_OVERFLOW_SATURATED_VALUE = 448.0
FP8_OUTPUT_OVERFLOW_BREAK = ("2.11.0", "2.13.0")
# ⚠ 因此 `FP8_OUTPUT_OVERFLOW_IS_NAN = True` 那句是**版本相关的**——
#   它记的是 ≥2.8.0 与 ≤2.11.0 那一档（即被探测选中那个版本的档）。
#   **不得写成「torch 的行为是 NaN」。**
FP8_OUTPUT_OVERFLOW_IS_NAN_IS_VERSION_SCOPED = True

# `_weight_int4pack_mm_for_cpu` 的同一判据（**判据同 §6，只是把算子换掉**）：
#   x = randint(1,5,(1,K))、nibble 恒为 15（⇒ 反量化 = 15 − 8 = 7）、scale=1、zp=0。
#   ⚠ 这里喂的是**已经无符号的 nibble**，所以**不能**再 `+8` —— 加了以后 nibble 变成 7、
#     反量化成 −1，参照会全错（差 8·Σx，差值比精确和还大）。**第一版就是这么错的。**
INT4PACK_ACC_PROBE_REL = {
    (1 << 20): 2.2893099038816884e-05,
    (1 << 21): 0.0004756684782543534,
}
INT4PACK_ACC_PROBE_SUM = {
    (1 << 20): 18346140.0,
    (1 << 21): 36697828.0,
}
