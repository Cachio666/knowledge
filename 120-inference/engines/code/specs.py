"""120-inference/engines 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 同值位置：` + `# 口径：` + `# 消费者：`。
★★ **本域的头号错误模式是「把 B 级当 A 级」** —— ★ 本主题**跨 A / B / C 三级**：
  TensorRT-LLM 的术语是 `[A]`（厂商第一方）、vLLM 的是 `[B]`、论文的是 `[C]` ⇒ **引用必须带级别**。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）；
  ★★ **字符串一律【单行】** —— 跨行隐式拼接会被 `ast` 折成一个 `Constant`、`end_lineno != lineno`
    ⇒ `mutate.py` 会**静默**把它计入「跳过」（`110-communication` 的现场）。

★★★ **本主题唯一的载体口径（★ 这是本文件最要紧的一条）**：
  **「各家引擎的事实」必须带【级别 + 版本/分支 + 它是规范还是该实现的选择】**（`§0.7 C` 的题眼）。
  两条最危险的坑：
    · ★★★ **vLLM 的 `design/paged_attention` 页自标为历史文档** ⇒ **不得拿它当 vLLM 现状**
      （这是「**把外部工件当不会错的正典**」的现场，见 `VLLM_PAGED_ATTENTION_HISTORICAL_QUOTE`）；
    · ★ **TensorRT-LLM 的顺序约束是该实现当前版本的** ⇒ **不得说成 continuous batching 的通用约束**
      （见 `TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE` / `TRTLLM_INFLIGHT_ORDER_QUALIFIER`）。

★★ **禁写清单（本主题）**：
  · 不得引 `design/paged_attention` 当 vLLM 现状（`F7`）；
  · 不得把 ORCA 的 `36:9×` 或 vLLM 首页的 `23×` 当本库实测（它们是**引用值**，基线各自不同）。

★ **本主题的禁区**（本主题**一行都不重写**，只回链；★ 机制细节一律回链）：
  · **分页** → `120-inference/kv-cache/02`；**前缀树** → `kv-cache/03`；**调度器** → `batching-and-scheduling/01`；
  · ★★ **PagedAttention 的定义与算式** → `130-quantization/kv-cache-quantization/01:68-84`
    + `40-kernels/attention/05:130-179`；
  · **vLLM 的模型接入与算子注册** → `45-kernel-lifecycle/requirement-discovery/01:158-214`。

★ **本主题的判据纪律**：
  ★★ **「没有 A 类」只适用于「引擎的【行为/性能】」** —— `RadixAttention` 的前缀树 / 块池 / LRU 是
  **可写独立参照实现**的数据结构 ⇒ `02-sglang-and-radix` **有真 A 类**；
  ★ 只有「真实引擎的行为」不可验（沿用 `nccl` 的 `D` 类处置：一切钉住归 `D`）。

★ **本机的限定语**：本机**无 GPU、无 vLLM / SGLang / TensorRT-LLM / llama.cpp**
  ⇒ ★★★ **本主题最强的一条限定语：**「**不是任何真实引擎，也不是真机**」。
"""


# ===========================================================================
# 1. PagedAttention 的机制定义（原始论文）—— [C] 论文
#    载体 = `https://arxiv.org/abs/2309.06180`（SOSP 2023 vLLM 论文）；抓取日 **2026-10-09**。
#    ★★ 这是**论文作者的算法命名与设计意图**，**不是任何实现的行为保证**。
# ===========================================================================

PAGEDATTENTION_PAPER_CITATION = "Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, Efficient Memory Management for Large Language Model Serving with PagedAttention, arXiv:2309.06180"
# 来源：[C] 论文 —— 载 `https://arxiv.org/abs/2309.06180`，arXiv 提交 2023-09-12（即 SOSP 2023 vLLM 论文），
#       抓取日 2026-10-09。
# 口径：★★ 本常量是**书目字段**（作者 + 标题 + 载体）—— 它只用来支撑「本主题的 PagedAttention 定义来自这篇论文」。
# 消费者：Chunk 4 的 `01-vllm`（程序 `code/01_*.py` 尚未落盘）—— 载体声明。
# 同值位置：无。

PAGEDATTENTION_DEFINITION_QUOTE = "we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems."
# 来源：[C] 论文 —— 逐字（`https://arxiv.org/abs/2309.06180`），抓取日 2026-10-09。
#       ★ 完整性：`grep -c 'PagedAttention, an attention algorithm inspired by the classical virtual memory'` → 1。
# 口径：★★★ 这是**论文的命名与设计意图**，**不是任何实现的行为保证**（`§0.7 C` 事实 1）⇒
#       引用时**必须**写明它是「论文的定义」，**不得**当成「vLLM 的现状」。
# 消费者：Chunk 4 的 `01-vllm`（PagedAttention 的名称来源）。
# 同值位置：无。

VLLM_PAGED_ATTENTION_HISTORICAL_QUOTE = "This is a historical document based on the original paper for vLLM. It no longer describes the code used in vLLM today."
# 来源：[B] vLLM 官方文档 —— `https://docs.vllm.ai/en/latest/design/paged_attention/`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'no longer describes the code used in vLLM today'` → 1；该页 HTTP 200、251859 字节。
# 口径：★★★ **vLLM 官方【自标】该页为历史文档** ⇒ **不得**拿它当 vLLM 0.31.0 的行为来源（`F7`）。
#       这是「**把外部工件当不会错的正典**」的教科书现场（`§0.7 C` 事实 2）。
# 消费者：Chunk 4 的 `01-vllm`（禁写清单的逐字依据）。
# 同值位置：无。


# ===========================================================================
# 2. TensorRT-LLM 的 in-flight batching —— [A] 厂商官方文档（NVIDIA 第一方）
#    载体 = `https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html`；抓取日 **2026-10-09**。
#    ★ 注：`.../performance/optimization.html` 实测 **HTTP 404**，有效来源改用 `advanced/gpt-attention.html`。
# ===========================================================================

TRTLLM_INFLIGHT_BATCHING_QUOTE = "TensorRT-LLM supports in-flight batching of requests (also known as continuous batching or iteration-level batching) for higher serving throughput."
# 来源：[A] NVIDIA 官方文档 —— `https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html`，抓取日 2026-10-09。
#       ★ 完整性：`grep -c 'in-flight batching'` 于原始 HTML → 1。
# 口径：★★★ 这是 **[A] 级的术语同义声明**（`= continuous batching = iteration-level batching`），
#       ★★ **是 NVIDIA 对 TensorRT-LLM 自己**的表述 ⇒ 引用时写明级别与产品。
#       ★ 同段还说「sequences in context phase can be processed together with sequences in generation phase」。
# 消费者：Chunk 4 的 `03-tensorrt-llm-and-llamacpp`（NVIDIA 的术语口径）。
# 同值位置：无。

TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE = "In the current implementation, the sequences that are going through the context phase must be before the sequences in the generation phase in the input tensor."
# 来源：[A] NVIDIA 官方文档 —— 同上页（`advanced/gpt-attention.html`），抓取日 2026-10-09。
# 口径：★★★ **这是【该实现当前版本】的顺序约束** —— 文档明说
#       「The constraint may or may not be relaxed in a future version」⇒
#       ★★ **不得**把这句说成「continuous batching 的**通用约束**」（`§0.7 C` 事实 8）。
# 消费者：Chunk 4 的 `03-tensorrt-llm-and-llamacpp`（顺序约束的口径）。
# 同值位置：无。

TRTLLM_INFLIGHT_ORDER_QUALIFIER = "该顺序约束（context 段必须排在 generation 段之前）是 TensorRT-LLM 当前实现的约束，文档自述可能在未来版本放宽，不得说成 continuous batching 的通用约束"
# 来源：**[E] 本库的限定语记录**（★ 不是文档原文的逐字句，而是对 `TRTLLM_INFLIGHT_ORDER_CONSTRAINT_QUOTE`
#       的一句**必须并置**的限定 —— 依据是文档自述「may or may not be relaxed in a future version」，`§0.7 C` 事实 8）。
# 口径：★★★ 它**没有外部出处**（是限定语而非引文）⇒ 不得把它写成引文（本库纪律：**自造真值**是一种错误模式）。
#       ★ 它的存在理由是**可复核**：引用该顺序约束时「有没有带限定」变成一条可被变异测试抓住的断言。
# 消费者：Chunk 4 的 `03-tensorrt-llm-and-llamacpp`（与顺序约束**同一条**断言）。
# 同值位置：无。


# ===========================================================================
# 3. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在本模块的正文常量里。
# ★★ **本元组【留空】**，理由与 `110-communication/collectives` 逐字相同：**每一项的字符串都会成为 `make mutate`
#   的叶子**，而它们**没有消费者** ⇒ 会全部记成**漏网**。★★ **而【空元组本身】有消费者**：
#   编号程序的 `preflight()` 用 `specs.UNSOURCED == ()` 钉住它。
# ★ **本主题的「未核到」写在注释里，不立常量**：
#   ① **vLLM `--enable-prefix-caching` 的默认开关值** —— 取数到的文档页**未声明**默认开/关 ⇒ **不得臆断**（`F5`）；
#   ② **各家引擎的版本相关参数**（如 `max_num_seqs` 随设备/场景变）—— **会漂移** ⇒ ★ **锚点用「文件路径 + 常量名」，不写行号**。
UNSOURCED = ()
