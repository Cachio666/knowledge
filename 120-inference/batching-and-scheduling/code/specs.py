"""120-inference/batching-and-scheduling 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 同值位置：` + `# 口径：` + `# 消费者：`。
★★ **本域的头号错误模式是「把 B 级当 A 级」**（真实引擎的一切都是 `[B]` 或以下）。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）；
  ★★ **字符串一律【单行】** —— 跨行隐式拼接会被 `ast` 折成一个 `Constant`、`end_lineno != lineno`
    ⇒ `mutate.py` 会**静默**把它计入「跳过」（`110-communication` 的现场）。

★★★ **本主题唯一的载体口径（★ 这是本文件最要紧的一条）**：
  **`max_num_seqs` / `max_num_batched_tokens` 【没有单一默认值】** ——
  类常量 `2048` / `128` 只是**兜底**，`EngineArgs.get_batch_defaults()` 会按 **显存 + 场景** 覆盖
  （见 `VLLM_MAX_NUM_SEQS_CLASS_DEFAULT` 下方的分档常量）。
  ★★★ **「vLLM 的 `max_num_seqs` 默认值」这句话没有单一答案** ⇒ 引用**必须带设备与场景**
  （只引 `128` 会把 A100 上 API server 的真实值 **256** 写成 128 —— 这是**规范与实现不分**的教科书现场）。

★ **本主题的禁区**（本主题**一行都不重写**，只回链）：
  · **continuous batching / chunked prefill 的【定义】** → `10-foundations/performance-models/01:251`；
  · **prefill/decode 的 kernel 差异** → `40-kernels/attention/05:229-236`；
  · ★★ **静态 batching 的填充浪费 `1 − 均值/最大值`** → `40-kernels/moe-kernels/02-grouped-gemm.md:89`
    （★ 本域**只写「同式在调度里的含义不同」**，**不重算** —— `P37` / `§0.7 D5`）。

★ **本主题的判据纪律**：
  ★★ **收益来自【长度方差】，不来自【到达过程】**（`§0.7 D3`）；
  ★★ 任何「X 带来 N 倍」的句子**必须写明【基线】**（`§0.7 D4`：同一件事的两个加速比差 **17 倍**）。

★ **本机的限定语**：本机**无 GPU、无 vLLM / SGLang / TensorRT-LLM**
  ⇒ ★★★ **本主题最强的一条限定语：**「**不是任何真实引擎，也不是真机**」。
"""


# ===========================================================================
# 1. vLLM 的 chunked prefill —— [B] 产品官方文档 + 产品源码
#    载体 = 文档 `configuration/optimization/` + 源码 `vllm/config/scheduler.py`（main 分支）；
#    抓取日 **2026-10-09**。★ 行号会漂移 ⇒ 锚点用「**文件路径 + 常量名**」，**不写行号**。
# ===========================================================================

VLLM_ENABLE_CHUNKED_PREFILL = True
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py`（main 分支，抓取日 2026-10-09），
#       常量 `SchedulerConfig.enable_chunked_prefill`。逐字见 `VLLM_CHUNKED_PREFILL_SOURCE_QUOTE`。
#       环境口径：vLLM 发布版 0.31.0；文档站 = latest developer preview。
# 口径：★★ 这是 **vLLM 的实现默认值**（`True`）——
#       ★★ **不得裸写「chunked prefill 默认开启」**：必须带 **「V1 / whenever possible」** 两个限定
#       （`VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE`），否则在 V0 语义下是错的（`F6`）。
#       ★ encoder-decoder 模型会**禁用** chunked prefill 与 prefix caching（「whenever possible」的落点）。
# 消费者：Chunk 2 的 `02-chunked-prefill`（程序 `code/02_*.py` 尚未落盘）—— 默认开关这一事实。
# 同值位置：`VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE`（同一条事实的「值」与「引文」两写）。

VLLM_CHUNKED_PREFILL_SOURCE_QUOTE = "enable_chunked_prefill: bool = True"
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py`（main 分支，抓取日 2026-10-09）。
#       ★ 完整性：`enable_chunked_prefill: bool = True`（`SchedulerConfig` 原文）。
# 口径：★★ 它是 `VLLM_ENABLE_CHUNKED_PREFILL` 的**逐字依据**（源码）。
# 消费者：Chunk 2 的 `02-chunked-prefill`（默认值的来源佐证）。
# 同值位置：`VLLM_ENABLE_CHUNKED_PREFILL`（同一条事实的「值」与「引文」两写）。

VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE = "In V1, chunked prefill is enabled by default whenever possible."
# 来源：[B] vLLM 官方文档 —— `configuration/optimization/`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'chunked prefill is enabled by default whenever possible'` 于原始 HTML → 1。
#       ★ 转写注：原始 HTML 在 `possible` 与句点间夹了一个排版空格；本常量按语义转写为常规句读。
# 口径：★★★ **「默认开启」必须带「V1 / whenever possible」两个限定**（`F6`）——
#       这是**句子的一部分**，不是可选注释。★ 「whenever possible」的落点是 encoder-decoder 模型**例外**。
# 消费者：Chunk 2 的 `02-chunked-prefill`（默认值口径的逐字依据）。
# 同值位置：`VLLM_ENABLE_CHUNKED_PREFILL`（同一条事实的「引文」与「值」两写）。


# ===========================================================================
# 2. vLLM 的 `max_num_seqs` / `max_num_batched_tokens` —— [B] 产品官方文档 + 产品源码
#    载体 = 文档 `configuration/engine_args/` + 源码 `vllm/config/scheduler.py` 与
#      `vllm/engine/arg_utils.py`（`get_batch_defaults`，main 分支）；抓取日 **2026-10-09**。
#    ★★★ 本节的核心事实：**这两个参数【没有单一默认值】**。
# ===========================================================================

VLLM_MAX_NUM_SEQS_CLASS_DEFAULT = 128
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py`（main 分支，抓取日 2026-10-09），
#       常量 `DEFAULT_MAX_NUM_SEQS`。`grep -c 'DEFAULT_MAX_NUM_SEQS: ClassVar\\[int\\] = 128'` → 1。
# 口径：★★★ **这只是【兜底类常量】，不是生产默认** —— `EngineArgs` 构造时会按**显存 + 场景**覆盖
#       （见 `VLLM_API_SERVER_H100_MAX_SEQS` / `VLLM_API_SERVER_DEFAULT_MAX_SEQS` 等分档）。
#       ★★ **不得裸写「vLLM 的 `max_num_seqs` 默认是 128」**（`F2`）。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（「兜底值 vs 覆盖值」对照）。
# 同值位置：无（★ 与分档常量**值不同**，正是本节的题眼）。

VLLM_MAX_NUM_BATCHED_TOKENS_CLASS_DEFAULT = 2048
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py`（main 分支，抓取日 2026-10-09），
#       常量 `DEFAULT_MAX_NUM_BATCHED_TOKENS`。`grep -c '... = 2048'` → 1。
# 口径：★★ **同样只是兜底类常量**；生产默认按显存 + 场景覆盖（见下）。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（同 `VLLM_MAX_NUM_SEQS_CLASS_DEFAULT`）。
# 同值位置：无。

VLLM_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP = 256
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py`（main 分支，抓取日 2026-10-09），
#       常量 `DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`。
# 口径：★★ 这是 **batched-DP 场景**的另一个兜底常量（★ 与上面两条**并列，不是它的子项**）。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（说明「兜底值有多个」）。
# 同值位置：无。

VLLM_API_SERVER_H100_MAX_TOKENS = 8192
# 来源：[B] vLLM 官方源码 —— `vllm/engine/arg_utils.py` 的 `get_batch_defaults()`（main 分支，抓取日 2026-10-09）。
#       分支条件：「device_memory >= 70 * GiB_bytes and "a100" not in device_name」（如 H100/H200），
#       `OPENAI_API_SERVER` → `max_num_batched_tokens`。
# 口径：★★ 这是 **该设备 + API server 场景**下的**覆盖值**（不是类常量）。★ 引用**必须带设备与场景**。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（「无单一默认值」的证据之一）。
# 同值位置：无。

VLLM_API_SERVER_H100_MAX_SEQS = 1024
# 来源：[B] 同上（`get_batch_defaults`，H100/H200 分支，`OPENAI_API_SERVER` → `max_num_seqs`）。抓取日 2026-10-09。
# 口径：★★ 覆盖值；★ 引用必须带设备与场景。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`。
# 同值位置：`VLLM_API_SERVER_B200_MAX_SEQS`（**值相同**（都是 1024），但**分属不同显存分支** —— 不得合并）。

VLLM_API_SERVER_DEFAULT_MAX_TOKENS = 2048
# 来源：[B] 同上（`get_batch_defaults`，**else 分支**「其余，含 A100」，`OPENAI_API_SERVER` → `max_num_batched_tokens`）。抓取日 2026-10-09。
# 口径：★★★ 这是 **A100 / 其余设备**在 API server 场景下的 `max_num_batched_tokens` **覆盖值** ——
#       ★★ 它与类常量 `VLLM_MAX_NUM_BATCHED_TOKENS_CLASS_DEFAULT` **恰好都是 2048**，
#       ★★★ **数值巧合 ≠ 同一个东西**（一个是兜底、一个是覆盖）—— 不得据此说「A100 用的就是兜底值」。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`。
# 同值位置：`VLLM_MAX_NUM_BATCHED_TOKENS_CLASS_DEFAULT`（**同值不同指** —— 兜底 vs 覆盖）。

VLLM_API_SERVER_DEFAULT_MAX_SEQS = 256
# 来源：[B] 同上（`get_batch_defaults`，**else 分支**「其余，含 A100」，`OPENAI_API_SERVER` → `max_num_seqs`）。抓取日 2026-10-09。
# 口径：★★★ **A100 上 API server 的真实默认值是 256，不是类常量的 128** ——
#       ★ 只引 `128` 就会写错（这正是 `F2` 要挡的那件事）。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（「无单一默认值」的关键反例）。
# 同值位置：`VLLM_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`（**值相同**（都是 256），**不同指** —— 不得合并）。

VLLM_API_SERVER_B200_MAX_TOKENS = 16384
# 来源：[B] 同上（`get_batch_defaults`，分支「device_memory >= 160 * GiB_bytes」，如 B200/B300，`OPENAI_API_SERVER` → `max_num_batched_tokens`）。抓取日 2026-10-09。
# 口径：★★ 覆盖值；★ 引用必须带设备与场景。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`。
# 同值位置：无。

VLLM_API_SERVER_B200_MAX_SEQS = 1024
# 来源：[B] 同上（B200/B300 分支，`OPENAI_API_SERVER` → `max_num_seqs`）。抓取日 2026-10-09。
# 口径：★★ 覆盖值；★ 与 `VLLM_API_SERVER_H100_MAX_SEQS` **值相同**（1024）但**分支不同**。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`。
# 同值位置：`VLLM_API_SERVER_H100_MAX_SEQS`（**同值不同指**）。

VLLM_TEST_CONVENIENCE_QUOTE = "The default value here is mainly for convenience when testing. In real usage, this should be set in EngineArgs.create_engine_config."
# 来源：[B] vLLM 官方文档 —— `configuration/engine_args/` 的 `--max-num-seqs` 与
#       `--max-num-batched-tokens`（**两条各一次**）。抓取日 2026-10-09。
#       ★ 完整性：`grep -c 'The default value here is mainly for convenience when testing'` → 2。
# 口径：★★★ 这是**文档自己**把两个「default」标注为**非生产默认**的逐字依据 ——
#       它证明「文档默认值」与「生产默认值」**不是同一件事**。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（默认值口径的逐字依据）。
# 同值位置：无。

VLLM_MAX_NUM_SEQS_NO_SINGLE_DEFAULT = "vLLM 的 max_num_seqs / max_num_batched_tokens 没有单一默认值：类常量 2048/128 只是兜底，get_batch_defaults() 按显存+场景覆盖，throughput 模式再各自翻倍"
# 来源：**[E] 本库的消歧记录**（★ 不是任何来源里的话）—— 依据是本节的 [B] 级分档常量与
#       `VLLM_TEST_CONVENIENCE_QUOTE` 的**并列**。
# 口径：★★★ 它是一条**「规范与实现不分」的挡板**：把类常量当成生产默认会**写错正文**（`F2`）。
#       ★ 它**没有外部出处** ⇒ 不得写成引文（本库纪律：**自造真值**是一种错误模式）。
#       ★ 它的存在理由是**可复核**：引用默认值时「有没有带设备与场景」变成一条可被变异测试抓住的断言。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（首次出现默认值处点明）。
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
#   ① **vLLM 文档对 `continuous batching` 的机制定义** —— 未取到逐字定义（首页只把它列为特性名）；
#      ★★ 机制所在的 `design/arch_overview` 用「a busy loop that continuously schedules requests」，
#      **未用该术语** ⇒ **不得写「vLLM 定义 continuous batching 为……」**。
UNSOURCED = ()
