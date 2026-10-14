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
  · 不得把 ORCA 的 `36:9×` 或 vLLM 首页引的那篇博客的 `23x` 当本库实测（它们是**引用值**，基线各自不同）。

★ **本主题的禁区**（本主题**一行都不重写**，只回链；★ 机制细节一律回链）：
  · **分页** → `120-inference/kv-cache/02`；**前缀树** → `kv-cache/03`；**调度器** → `batching-and-scheduling/01`；
  · ★★ **PagedAttention 的定义与算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84`
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
# 消费者：`01-vllm`（程序 `code/01_vllm.py`，已落盘）—— 载体声明。
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
#   ② **各家引擎的版本相关参数**（如 `max_num_seqs` 随设备/场景变）—— **会漂移** ⇒ ★ **锚点用「文件路径 + 常量名」，不写行号**；
#   ③ ★★ **llama.cpp 的 CPU-first 取向** —— ★★★ **研究原始报告的 13 条里【没有】llama.cpp 的一手来源**
#      （唯一的 llama.cpp 条目是「本机没有它」）⇒ ★ 本主题**不得**据印象写它的实现细节；`03` 只把它写成
#      **边界项**（`LLAMACPP_FIRST_SOURCE_OBTAINED = False`），★ 该 bool **有消费者**，不是无主常量。
UNSOURCED = ()


# ===========================================================================
# 4. vLLM 的架构选择 —— [B] 产品官方文档 + 产品源码（`01-vllm` 的消费者）
#    载体 = 文档站 `https://docs.vllm.ai/en/latest/`（latest developer preview，页脚 2026-10-08）
#           + GitHub `vllm/config/{cache,scheduler}.py` / `vllm/engine/arg_utils.py`（**main 分支**）；
#    抓取日 **2026-10-09**。★ 行号会漂移 ⇒ 锚点一律用「**文件路径 + 常量名**」，**不写行号**。
# ===========================================================================

VLLM_ARCH_BUSY_LOOP_QUOTE = "a busy loop that continuously schedules requests and dispatches work to the GPU workers"
# 来源：[B] vLLM 官方文档 —— `design/arch_overview`（抓取日 2026-10-09）。
# 口径：★★★ 这是 **vLLM 一家**对**引擎循环**的措辞；★ 该页**没有**使用「continuous batching」这个词
#       ⇒ **不得**据此写「vLLM 定义 continuous batching 为……」（本批取数找不到逐字支撑）。
# 消费者：`01` 的 `B.1` / `D.5`（引擎循环的逐字依据）。
# 同值位置：无。

VLLM_IR_REF = "45-kernel-lifecycle/requirement-discovery/01:205-214"
# 来源：**[E] 本库的跨域回链** —— 本库 `45-kernel-lifecycle/requirement-discovery/01:205-214`
#       写了 vLLM 的 `@register_op` / `register_impl` / 优先级列表 / `native` 兜底（`vllm.ir`）。
# 口径：★★ 它是**回链目标**，**不是**引文 —— 本主题**一行都不重写**那个机制（`P37`）。
# 消费者：`01` 的 `D.3`（回链锚点逐字钉住）。
# 同值位置：`45-kernel-lifecycle/requirement-discovery/01`（**那是正文，这里是回链**）。

VLLM_BLOCK_SIZE_TOKENS = 16
# 来源：[B] vLLM 官方源码 —— `vllm/config/cache.py`（main 分支，抓取日 2026-10-09），常量 `DEFAULT_BLOCK_SIZE`。
# 口径：★★ 单位 = **token / 块**；★★ **不得写成「业界通常 16」**（`F1`）：SGLang 的 `page = 1 token`（见第 5 节）。
# 消费者：`02` 的 `C.1`（同名不同指：16 vs 1）—— 消费者的消费者链见下一条。
# 同值位置：`120-inference/kv-cache` 的 `VLLM_DEFAULT_BLOCK_SIZE`（同一条事实的两处引用）。

VLLM_BLOCK_SIZE_SOURCE_QUOTE = "DEFAULT_BLOCK_SIZE: ClassVar[int] = 16"
# 来源：[B] vLLM 官方源码 —— `vllm/config/cache.py`（main 分支，抓取日 2026-10-09）。
# 口径：★ 它是上面那个 `16` 的**逐字依据**（**源码**，不是文档 —— 文档只写 `Accepts None (meaning "use default")`）。
# 消费者：`02` 的 `D.6`（引文逐字钉住）。
# 同值位置：`120-inference/kv-cache` 的 `VLLM_BLOCK_SIZE_SOURCE_QUOTE`。

VLLM_MAX_NUM_SEQS_FALLBACK = 128
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py` 的 `DEFAULT_MAX_NUM_SEQS`（main 分支，抓取日 2026-10-09）。
# 口径：★★★ 这是**类常量兜底**，**不是**「vLLM 的默认值」—— 真实默认由 `get_batch_defaults()` 按
#       设备 + 场景覆盖（见 `VLLM_BATCH_DEFAULTS`）⇒ **「vLLM 的 max_num_seqs 默认值」没有单一答案**（`F2`）。
# 消费者：`01` 的 `B.2`（参数面）/ `C.1`（把兜底当权威的后果）。
# 同值位置：无。

VLLM_MAX_NUM_BATCHED_TOKENS_FALLBACK = 2048
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py` 的 `DEFAULT_MAX_NUM_BATCHED_TOKENS`（同上）。
# 口径：★ 同上：**兜底值**，会被 `get_batch_defaults()` 覆盖。
# 消费者：`01` 的 `B.2` / `D.7`。
# 同值位置：无。

VLLM_BATCH_DEFAULTS = (
    ("B200/B300", "OPENAI_API_SERVER", 16384, 1024),
    ("B200/B300", "LLM_CLASS", 16384, 1024),
    ("H100/H200", "OPENAI_API_SERVER", 8192, 1024),
    ("H100/H200", "LLM_CLASS", 16384, 1024),
    ("A100/其余", "OPENAI_API_SERVER", 2048, 256),
    ("A100/其余", "LLM_CLASS", 8192, 256),
)
# 来源：[B] vLLM 官方源码 —— `vllm/engine/arg_utils.py` 的 `get_batch_defaults()`（main 分支，抓取日 2026-10-09）。
#       每行 = `(设备档, 使用场景, max_num_batched_tokens, max_num_seqs)`。
# 口径：★★★ 这张表是**同一对参数随设备/场景变**的现场 ⇒ 只引兜底 `128` 会把 A100 上
#       API server 的真实值（`256`）写成 `128`（「**规范与实现不分**」，`§0.7 C` 事实 5）。
#       ★ 表按研究原始报告的**三个分支**转录（`>=160 GiB` / `>=70 GiB 且非 a100` / else）。
# 消费者：`01` 的 `B.2` / `C.1` —— ★★ `D.7` **逐元素**钉住（六行 × 四列）。
# 同值位置：无。

VLLM_THROUGHPUT_MULTIPLIER = 2
# 来源：[B] vLLM 官方源码 —— `get_batch_defaults()`：`performance_mode == "throughput"` 且用户未显式给值时，
#       两个参数**各自翻倍**（抓取日 2026-10-09）。
# 口径：★★ 它让「默认值」的多样性再翻一倍 ⇒ 参数面**不是一个数**。
# 消费者：`01` 的 `B.2`（翻倍后的取值集合）。
# 同值位置：无。

VLLM_MAX_NUM_SEQS_QUOTE = "Maximum number of sequences to be processed in a single iteration. The default value here is mainly for convenience when testing. In real usage, this should be set in EngineArgs.create_engine_config."
# 来源：[B] vLLM 官方文档 —— `configuration/engine_args/` 的 `--max-num-seqs`（抓取日 2026-10-09）。
# 口径：★★ 文档自己说这个 default 是「**mainly for convenience when testing**」⇒ **不是生产默认**。
# 消费者：`01` 的 `D.5`（引文逐字钉住）。
# 同值位置：`--max-num-batched-tokens` 的**同句式**（★ **≥ 2 处** —— 今日原始 HTML 上 `grep -c`
#       得 3，**第三处属 `--enable-chunked-prefill`**；★★ **该句式会随文档改版漂移 ⇒ 计数会失效**，
#       故这里只作**定性提示**、**不写死条数**（与「锚点用名字、不用行号」同一处置）。

VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE = "chunked prefill is enabled by default whenever possible"
# 来源：[B] vLLM 官方文档 —— `configuration/optimization/`（抓取日 2026-10-09）。
# 口径：★★★ 「默认开启」**必须带**「**In V1**」与「**whenever possible**」两个限定（`F6`）；
#       `SchedulerConfig` 的 docstring 明说 encoder-decoder 模型会**禁用** ⇒ **不是无条件默认**。
# 消费者：`01` 的 `C.2`（缺限定即错）/ `D.5`。
# 同值位置：源码 `enable_chunked_prefill: bool = True`（同一条事实的**值**与**引文**两写）。

VLLM_CHUNKED_PREFILL_QUALIFIERS = ("In V1", "whenever possible")
# 来源：**[E] 本库的限定语清单** —— 依据是 `VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE`（含 `whenever possible`）
#       与同一文档页的 `In V1`（`§0.7 C` 事实 10）。
# 口径：★★★ **两个限定缺一不可**（`F6`）⇒ 「默认开启」这句话**不带这两个词就是不完整的**。
# 消费者：`01` 的 `C.2`（缺限定的断言必须失败）。
# 同值位置：无。

VLLM_ENABLE_PREFIX_CACHING_QUOTE = "Whether to enable prefix caching."
# 来源：[B] vLLM 官方文档 —— `configuration/engine_args/` 的 `--enable-prefix-caching`（抓取日 2026-10-09）。
# 口径：★★ 取数到的页面**未声明开/关默认值** ⇒ **不得断言「APC 默认开启」**（`F5`）。
# 消费者：`02` 的 `C.2`（默认行为：SGLang 默认 vs vLLM 需开关）/ `D.7`。
# 同值位置：无。

VLLM_HOME_FEATURES_QUOTE = "Continuous batching of incoming requests, chunked prefill, prefix caching"
# 来源：[B] vLLM 官方文档首页（抓取日 2026-10-09）。
# 口径：★★ 首页把 continuous batching 只当**特性名**列出，**没有机制定义**（`§0.7 C` 事实 6）。
# 消费者：`01` 的 `D.5`（逐字钉住）。
# 同值位置：无。

ORCA_ITERATION_LEVEL_QUOTE = "schedules execution at the granularity of iteration (instead of request)"
# 来源：[C] 论文 —— ORCA（OSDI '22，`https://www.usenix.org/conference/osdi22/presentation/yu`，抓取日 2026-10-09）。
# 口径：★★★ 原始术语是 **iteration-level scheduling**（ORCA，`§0.7 C` 事实 7）⇒
#       **不得**写成「vLLM 造的词」（`F3`）。
# 消费者：`01` 的 `C.3` / `D.5`。
# 同值位置：无。

ORCA_THROUGHPUT_QUOTE = "36:9× throughput improvement at the same level of latency"
# 来源：[C] 论文 —— ORCA（OSDI '22，同上，抓取日 2026-10-09）。
# 口径：★★★ 它是**引用值**（基线 = NVIDIA FasterTransformer）⇒ **不得当作本库的测量**（`F4`）；
#       ★ 原文用的是**冒号**（`36:9×`），**不是**小数点。
# 消费者：`01` 的 `D.5`（引用值钉住）/ `C.4`。
# 同值位置：无。

VLLM_HOME_23X_BLOG_QUOTE = "How continuous batching enables 23x throughput in LLM inference while reducing p50 latency"
# 来源：★★★ **双标注**（★ 与 `batching-and-scheduling/01:177` 的标法**统一**，`F1`）——
#        **载体 = `[B]`**（vLLM 官方文档首页的「进一步阅读」栏目，抓取日 2026-10-09）
#        + **数字出处 = `[D]`**（该栏目引的 Cade Daniel et al. 的**博客**）。
# 口径：★★★ 它与 ORCA 的 `36:9×` 是**两个不同来源**（`§0.7 D4` 的订正）⇒ 引用必须**连出处**写（`F4`）。
# 消费者：`01` 的 `D.5`。
# 同值位置：无。

VLLM_INTER_TOKEN_LATENCY_QUOTE = "Mean inter-token latency (average time between successive output tokens) during the decode phase."
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `mean_itl_ms`（抓取日 2026-10-09）。
# 口径：★★★ vLLM 的官方词是 **inter-token latency**；**`TPOT` 在 vLLM 页面上 0 处**（`F9`）⇒
#       「vLLM 把每输出 token 时间叫 TPOT」这句话**可被机械判定为错**（见 `VLLM_TPOT_OCCURRENCES`）。
# 消费者：`01` 的 `C.4` / `D.5`。
# 同值位置：无。

VLLM_TPOT_OCCURRENCES = 0
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 与 `features/per_request_metrics/` 两页的**原始 HTML 上**
#       `grep -c 'TPOT'` 均为 **0**（抓取日 2026-10-09，见研究原始报告事实 12 的验证行）。
# 口径：★★★ 读法：**「vLLM 把每输出 token 时间叫 TPOT」是错的** —— 那是**社区缩写**（`F9`）。
# 消费者：`01` 的 `C.4`（裸写 `TPOT` 当官方词必须被拒）。
# 同值位置：无。


# ===========================================================================
# 5. SGLang 的 RadixAttention —— [B] 项目官方博客 + 本主题的算例（`02-sglang-and-radix` 的消费者）
#    载体 = `https://lmsys.org/blog/2024-01-17-sglang/`（LMSYS Org，2024-01-17）；抓取日 **2026-10-09**。
#    ★ 注：早期猜测的 URL `.../2023-12-18-sglang/` 实测 **HTTP 404**（研究原始报告的「未取到」节）。
# ===========================================================================

SGLANG_RADIX_RETENTION_QUOTE = "Instead of discarding the KV cache after finishing a generation request, our approach retains the KV cache for both prompts and generation results in a radix tree."
# 来源：[B] LMSYS Org 官方博客 —— `https://lmsys.org/blog/2024-01-17-sglang/`（抓取日 2026-10-09）。
# 口径：★★★ 它同时给出**两条**差异化事实：(a) 用 **radix tree** 保留 KV；(b) 是**默认行为**
#       （`Instead of discarding ...`，与 vLLM 需 `--enable-prefix-caching` 迥异，`§0.7 C` 事实 11）。
# 消费者：`02` 的 `C.2` / `D.5`。
# 同值位置：无。

SGLANG_RADIX_DEF_QUOTE = "A radix tree is a data structure that serves as a space-efficient alternative to a trie (prefix tree)."
# 来源：[B] LMSYS Org 官方博客 —— 同上（抓取日 2026-10-09）。
# 口径：★★ 它是「**radix tree 相对未压缩 trie 省空间**」这条论断的逐字依据 ⇒ `02` 的 `A.2`
#       把「省空间」**量化**（节点数对比），★ 而不只是复述这句话。
# 消费者：`02` 的 `A.2` / `D.5`。
# 同值位置：无。

SGLANG_LRU_QUOTE = "We implement a Least Recently Used (LRU) eviction policy, complemented by a cache-aware scheduling policy, to enhance the cache hit rate."
# 来源：[B] LMSYS Org 官方博客 —— 同上（抓取日 2026-10-09）。
# 口径：★★ 它是「**淘汰策略 = LRU**」的逐字依据 ⇒ `02` 的 `A.3` 用**独立参照**（列表扫描）核对它。
# 消费者：`02` 的 `A.3` / `D.5`。
# 同值位置：无。

SGLANG_PAGE_TOKENS = 1
# 来源：[B] LMSYS Org 官方博客 —— 同上（抓取日 2026-10-09）。逐字见 `SGLANG_PAGE_LAYOUT_QUOTE`。
# 口径：★★★ 这是 **SGLang** 的存储布局口径（`page = 1 token`），★ **不是** vLLM 的 `block = 16`
#       （`VLLM_BLOCK_SIZE_TOKENS`）—— 两者**同名不同指**、**差 16 倍**（`§0.7 C` 事实 11）。
# 消费者：`02` 的 `C.1`（同名不同指）/ `D.6`。
# 同值位置：`120-inference/kv-cache` 的 `SGLANG_PAGE_TOKENS`（同一条事实的两处引用）。

SGLANG_PAGE_LAYOUT_QUOTE = "These KV cache tensors are stored on the GPU in a paged layout, where the size of each page is equivalent to one token."
# 来源：[B] LMSYS Org 官方博客 —— 同上（抓取日 2026-10-09）。
#       ★ 完整性：`curl .../2024-01-17-sglang/ | grep -c 'RadixAttention'` → 10（原始 HTML 字节）。
# 口径：★★ 「page」在这里是 **SGLang 的词**；vLLM 用「block」（`VLLM_BLOCK_SIZE_TOKENS`）。
# 消费者：`02` 的 `D.6`。
# 同值位置：`SGLANG_PAGE_TOKENS`（同一条事实的「引文」与「值」两写）。

PAGE_BLOCK_DISAMBIGUATION = "block / page 是同名不同指：vLLM 的 KV 块 = 16 token，SGLang 的 page = 1 token，两者差 16 倍，不可互相套用"
# 来源：**[E] 本库的消歧记录**（★ 不是任何来源里的话）—— 依据是 `VLLM_BLOCK_SIZE_TOKENS`
#       与 `SGLANG_PAGE_TOKENS` 两条 [B] 级事实的**并列**。
# 口径：★★★ 它是一条**同名不同指**的记录：把 vLLM 的 `block` 与 SGLang 的 `page` 当成同一个粒度会**写错正文**
#       （`§2.2` 与 `§0.7 C` 事实 3/11 都点名这一条）。★ 它**没有外部出处** ⇒ 不得写成引文（本库纪律：
#       **自造真值**是一种错误模式）。★ 它的存在理由是**可复核**（引用块粒度时「有没有写明是哪一家」）。
# 消费者：`02` 的 `C.1` / `D.6`。
# 同值位置：`120-inference/kv-cache` 的 `PAGE_BLOCK_DISAMBIGUATION`。

SGLANG_PREFIX_REUSE_IS_DEFAULT = True
# 来源：**[E] 本库的记录** —— 依据是 `SGLANG_RADIX_RETENTION_QUOTE` 的
#       「Instead of discarding ... our approach retains ...」（**默认保留**），
#       对照 `VLLM_ENABLE_PREFIX_CACHING_QUOTE`（**需显式开启**的一个开关）。
# 口径：★★★ **两条必须分开写**：(a) SGLang 的前缀复用**是默认行为**；(b) vLLM 的需 `--enable-prefix-caching`。
#       ★ 它**不是**引文（是 [E] 记录）⇒ 不得写成引文。★ 它的理由是**可机械判定**：`02` 的 `C.2` 断言
#       「按引擎名取的『默认即复用』布尔」在两家的取值**不同**。
# 消费者：`02` 的 `C.2`。
# 同值位置：无。


# ===========================================================================
# 6. `02` 的 radix tree 与 LRU —— **模型参数**（[E] 本主题的算例，不是外部事实）
# ===========================================================================

RADIX_ALPHABET = 8
# 来源：**[E] 本主题的算例** —— token 的取值范围 `[0, 8)`（**小**到让前缀频繁共享）。
# 口径：★★ 取小值是为了让 `A.1`（最长前缀）与 `A.2`（压缩）**真的有信息量**；★ **不是**任何真实词表。
# 消费者：`02` 的 `A.1` / `A.2` / `B.1`。
# 同值位置：无。

RADIX_N_SEQ = 12
# 来源：**[E] 本主题的算例** —— 构造的序列条数。
# 口径：★ 12 条足够造出**多层共享前缀**（多条序列共享同一段开头）。
# 消费者：`02` 的 `A.1` / `A.2` / `B.1`。
# 同值位置：无。

RADIX_SEQ_LEN_GRID = (2, 3, 4, 5)
# 来源：**[E] 本主题的算例** —— 序列长度在 `[2, 5]` 上取（★ 含**极值**短序列）。
# 口径：★ 长度 ≥ 2 是**故意的**：这样才有**可压缩的链**（长度 1 的序列压不出节点）⇒ `A.2` 才有信息量。
# 消费者：`02` 的 `A.1` / `A.2` / `D.7`（逐元素）。
# 同值位置：无。

RADIX_SEED = 20261009
# 来源：**[E] 本主题的算例**（伪随机种子，`numpy.random.default_rng`）。
# 口径：★ 钉死 ⇒ `A.1`/`A.2` 的结论**可复现**。
# 消费者：`02` 的 `A.1` / `A.2` / `B.1`。
# 同值位置：`120-inference/kv-cache` 的 `PREFIX_SEED`（同一天，**不同主题的不同流**）。

RADIX_CAP_GRID = (2, 4, 8, 16)
# 来源：**[E] 本主题的算例** —— LRU 缓存的**容量网格**（条数）。
# 口径：★ `B.2` 断言命中率**随容量单调不降**；★ 容量单位是**条序列**，不是字节。
# 消费者：`02` 的 `B.2` / `D.7`（逐元素）。
# 同值位置：无。


# ===========================================================================
# 7. TensorRT-LLM 与 llama.cpp —— [A] 厂商文档 + [E] 边界记录（`03-...` 的消费者）
#    载体 = `https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html`（NVIDIA 第一方）；抓取日 **2026-10-09**。
#    ★ 注：`.../performance/optimization.html` 实测 **HTTP 404**（研究原始报告的「未取到」节）。
# ===========================================================================

TRTLLM_INFLIGHT_SYNONYM_QUOTE = "in-flight batching of requests (also known as continuous batching or iteration-level batching)"
# 来源：[A] NVIDIA 官方文档 —— `advanced/gpt-attention.html`（抓取日 2026-10-09）。
# 口径：★★★ 这是 **[A] 级的术语同义声明**（`= continuous batching = iteration-level batching`），
#       ★★ **是 NVIDIA 对 TensorRT-LLM 自己**的表述 ⇒ 引用时写明**级别与产品**。
# 消费者：`03` 的 `C.2` / `D.5`。
# 同值位置：`TRTLLM_INFLIGHT_BATCHING_QUOTE`（**同一条事实的完整句 vs 同义子句两写**）。

TRTLLM_PACKED_QUOTE = "the support for inflight batching requires the input tensors to be packed (no padding)"
# 来源：[A] NVIDIA 官方文档 —— `advanced/gpt-attention.html`（抓取日 2026-10-09）。
# 口径：★★ 它是**顺序约束**的**前提**（`padded` 张量没有「顺序」可言）⇒ `03` 的 `B.2` 用它把
#       「结构后果」讲清：**紧凑张量 + 顺序要求 ⇒ 布局被唯一确定**。
# 消费者：`03` 的 `B.2` / `D.5`。
# 同值位置：无。

TRTLLM_ORDER_RELAX_QUOTE = "The constraint may or may not be relaxed in a future version"
# 来源：[A] NVIDIA 官方文档 —— `advanced/gpt-attention.html`（抓取日 2026-10-09）。
# 口径：★★★ 它是 `TRTLLM_INFLIGHT_ORDER_QUALIFIER` 的**逐字依据** ⇒ 该顺序约束是
#       **该实现当前版本**的，**不得**说成 continuous batching 的通用约束（`§0.7 C` 事实 8）。
# 消费者：`03` 的 `C.1` / `D.5`。
# 同值位置：无。

TRTLLM_CONFIG_AXES = (3, 3, 2)
# 来源：**[E] 本主题的算例**（不是外部事实）—— 三个配置轴各取几个值（`max_batch_size` / `max_input_len` /
#       `max_output_len` ⇒ 各 `3/3/2` 档）。
# 口径：★★★ 它是**编译式路线的结构账**：本模型假设「每个（batch, 输入长, 输出长）组合**各编一份**」⇒
#       份数 = 三轴**乘积**（`3×3×2 = 18`）。★ **这是假设，不是任何引擎的实测**（本机无引擎）。
# 消费者：`03` 的 `B.1`。
# 同值位置：无。

LLAMACPP_FIRST_SOURCE_OBTAINED = False
# 来源：**[E] 本库的边界记录** —— ★★★ **研究原始报告的 13 条里【没有】llama.cpp 的一手来源**
#       （唯一的 llama.cpp 相关条目是「本机没有它」）。
# 口径：★★★ 它记的是「**本库**有没有取到 llama.cpp 的一手来源」，**不是**「llama.cpp 有没有这个性质」——
#       ★ **「数不到」≠「不存在」**。★ 它的用途是把「不写 llama.cpp 的实现细节」变成**可被断言的边界**，
#       而不是凭印象写一段社区转述（那是「把 B 级当 A 级」的现场）。
# 消费者：`03` 的 `C.3` / `D.7`。
# 同值位置：无。


# ===========================================================================
# 8. 抓取日（★ 全主题共用一条）—— [E] 本库自记，不是外部事实
# ===========================================================================

ENGINES_FETCH_DAY = "2026-10-09"
# 来源：**[E] 本库自记** —— 它记的是「本主题所有外部条目的抓取日」（计划 `§0.7 C` 的题头 + 本文件各节的 `# 来源：`）。
# 口径：★★ 不写抓取日的引文不算引文；★ 本主题的 vLLM / SGLang / TensorRT-LLM / ORCA / 论文条目**同日抓取**。
# 消费者：`01` 的 `D.3` · `02` 的 `D.3` · `03` 的 `D.3`（三个程序各钉一次）。
# 同值位置：本文件第 4/5/7 节各条目的 `# 来源：` 注释。
