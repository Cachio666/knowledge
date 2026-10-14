#!/usr/bin/env python3
"""130-model-optimization/overview 主题的事实源与示例配置。

★ 本主题是 CONTRIBUTING §2 判定表里的**第二种情形**：编号程序**本机跑得了**
  （`.py`，默认 `python3` + `numpy` 即可），所以三类断言**全部在编号程序内部**，
  失败时以非零码退出。因此本目录**不需要 `verify.py`**。
  三个编号程序产出的是**推导值**——它们只跑算术与 numpy，不碰硬件、不加载权重。

★ 本文件只放**事实前提**，不放推导值。分三类：

  ① 示例模型配置 —— **本文构造的示例假设**，不是任何已发布模型的规格；
  ② 本库别处已经推导过、本主题**只消费不重推**的量（每 token KV 字节、`16Ψ` 的分解）；
  ③ 各族档位的外部常量（2:4 metadata 位宽、NF4 与 double quantization 的块参数）。

★ 引用分级（CONTRIBUTING §2）：[A] 厂商规范 · [B] 产品官方文档 · [C] 论文/技术报告 ·
  [D] 社区项目 · [E] 本库推导值。每条常量都带 `# 来源：` 与**口径注释**。
"""

# ===========================================================================
# 1. 示例模型配置（本文构造的示例假设）
# ===========================================================================

EXAMPLE_MODEL = {
    "layers": 32,          # 只数 transformer block，**不含** embedding
    "d_model": 4096,
    "d_ff": 11008,         # SwiGLU 的中间维 ⇒ 每层三个矩阵 gate / up / down
    "n_heads": 32,
    "head_dim": 128,
    "n_kv_heads": 32,      # = n_heads ⇒ MHA
    "vocab": 32000,
}
# 来源：**本文构造的示例假设**。形状取 Llama-2-7B 的量级，但**本文不对任何已发布模型
#   作规格声明**——配置是假设，配置之上的字节账才是本主题的推导。
# 口径：`layers` 不含 embedding 层；`d_ff` 是 SwiGLU 的中间维（每层 `3·d·d_ff`）；
#   `n_kv_heads = n_heads` ⇒ MHA（GQA 会改变 KV 那一本账，本文不展开）。

ELEMENT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int4": 0.5}
# 来源：位宽 ÷ 8。
# 口径：`int4` 的 0.5 B 是**纯载荷**——含 scale/zero 的「整块宽度」另算，
#   本文在需要时把 metadata 单列（见 `SPARSE_2_4_METADATA_BITS_PER_GROUP` 与
#   `NF4_*` 那一组常量），**不把它混进元素位宽**。

ACTIVATION_BYTES_TRAIN_EXAMPLE = 6.0e9
ACTIVATION_BYTES_PREFILL_EXAMPLE = 3.0e9
# 来源：**示例假设**。本主题讲的是**归因**（哪一族动哪一本账），不是激活公式的推导——
#   激活账的正文在 `90-training/autograd-and-backward/02-gradient-checkpointing.md`
#   （`O(√n)` 那条链），本文**不重推**，只把它当作一个**给定的输入**。
# 口径：两个数都是「batch × seq × d × layers 量级」的示例值，**不是任何实测**；
#   训练态比 prefill 大，因为它还要保反向用的中间量。

# ===========================================================================
# 2. 只消费、不重推的量（正文在别处）
# ===========================================================================

KV_BYTES_PER_TOKEN_FP16 = 524288            # = 512 KiB
# 来源：[E] 本库 `130-model-optimization/quantization/18-kv-cache-basics.md`
#   ——`每 token 字节 = 2 × layers × kv_heads × head_dim × bytes_per_element` 的
#   **正文在那里**，本文不重推。
# 口径：该值对应 LLaMA-2-7B（MHA / 32 层 / 32 个 KV 头 / head_dim 128 / fp16）。
#   ★ 上面 `EXAMPLE_MODEL` 的形状恰好给出同一个数（`2·32·32·128·2 = 524288`）——
#   `01_taxonomy.py` 的 A.3 会**独立复算**一遍并与本常量对拍：那是 **A 类**
#   （独立路径复核 + 跨域一致性），**不是**把正文再写一遍。

TRAINING_STATE_BYTES_PER_PARAM = 16
# 来源：[E] 本库 `90-training/checkpointing/01-state-and-bytes.md` §1.3：
#   `2Ψ + 2Ψ + KΨ = 16Ψ`（参数 fp16、**梯度 fp16**、优化器状态 `K = 12`：
#   fp32 主权重 + fp32 `m` + fp32 `v`）。
# 口径：★ 这是**「训练显存」口径、含梯度**，**不是**检查点大小。
#   本文只**消费**这个分解（分解与逐项字节的正文在那篇），不重推。

CHECKPOINT_BYTES_PER_PARAM = 14
# 来源：同上 §1.3：`2Ψ + 12Ψ = 14Ψ`（参数 + 优化器状态，**不含梯度**）。
# 口径：与训练显存**恰好差 `2Ψ`**——`02_accounting.py` 的 C.3 用这个差做反例。

OPTIMIZER_STATE_BYTES_PER_PARAM = 12
# 来源：同上（`K = 12`）。本文在归因表里单列它，因为**六族里只有 PEFT 直接动这一项**。

# ===========================================================================
# 3. 各族档位的外部常量
# ===========================================================================

SPARSE_2_4_GROUP = 4
SPARSE_2_4_METADATA_BITS_PER_GROUP = 4
# 来源：[A] PTX ISA 稀疏矩阵存储一节：2:4 的 metadata 是 `.b32` 寄存器里 16 个 2-bit
#   向量，每两个向量存一个 4 元素组里两个非零的索引 ⇒ **4 bit/组**。
# 口径：**一组 = 4 个元素**。⚠ 同一个 4 bit 有两个分母（占原数据 `4/(4·w)`、
#   占压缩后载荷 `4/(2·w)`，`w` = 元素位宽）——本文件只给「位/组」这一个原子事实，
#   分母由程序显式写出。稀疏粒度的正文在 `sparsity/01-sparsity-granularity.md`。

NF4_ELEMENT_BITS = 4
NF4_BLOCK_SIZE = 64
NF4_CONST_BITS_FP32 = 32
DQ_CONST_BITS_FP8 = 8
DQ_BLOCK_SIZE = 256
# 来源：[C] QLoRA, arXiv:2305.14314v1 §3（NeurIPS 2023；**逐字读到**）：
#   「using 32-bit constants and a blocksize of 64 … quantization constants add
#   `32/64 = 0.5` bits per parameter on average」；
#   「We use 8-bit Floats with a blocksize of 256 for the second quantization」；
#   「reduces the memory footprint per parameter from `32/64 = 0.5` bits, to
#   `8/64 + 32/(64·256) = 0.127` bits, a reduction of **0.373** bits per parameter」。
# 口径：`NF4_ELEMENT_BITS = 4` 是**元素载荷**；上式的两项分别是
#   **第二级常量的位宽摊到第一级块**（`8/64`）与**第二级自己的 scale 摊到两层块**
#   （`32/(64·256)`）。★ 论文摘要另写「about 0.37 bits」（**引用值**，
#   四舍五入），程序里对拍的是 §3 的 `0.373`。

QLORA_DQ_PAPER_SAVING_BITS = 0.37
# 来源：[C] 同上，摘要逐字：`saving an average of about 0.37 bits per parameter
#   (approximately 3 GB for a 65B model)`。
# 口径：**引用值**，只用于「论文摘要与 §3 是否自洽」的对拍（`03_composition.py` 的 B.3），
#   **不参与**本库的推导链。

QLORA_16BIT_65B_MEMORY_GB = 780
QLORA_4BIT_65B_MEMORY_GB = 48
# 来源：[C] 同上，§1 逐字：`regular 16-bit finetuning of a LLaMA 65B parameter model
#   requires more than 780 GB of GPU memory`；`reduces the average memory requirements
#   of finetuning a 65B parameter model from >>780GB of GPU memory to <48GB`。
# 口径：**引用值**，且是两个**不等式**（`>` 与 `<`）——引用时必须保留不等号，
#   不得写成「从 780 GB 降到 48 GB」这样的确数。

QLORA_LORA_PARAM_FRACTION = 0.002
# 来源：[C] 同上 §2「Memory Requirement of Parameter-Efficient Finetuning」逐字：
#   `with LoRA weights equivalent to commonly used 0.2% of the original model weights`。
# 口径：**引用值**，是论文引用的**常用比例**，不是 QLoRA 的规定值。
