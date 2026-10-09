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
    （★ 本域**只写「同式在调度里的含义不同」**，**不重算** —— **错误模式 `P37`** / `§0.7 D5`）。

★ **本主题的判据纪律**：
  ★★ **收益来自【长度方差】，不来自【到达过程】**（`§0.7 D3`）；
  ★★ 任何「X 带来 N 倍」的句子**必须写明【基线】**（`§0.7 D4`：同一件事的两个加速比差 **16.9 倍**）。

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
# 同值位置：`VLLM_API_SERVER_DEFAULT_MAX_TOKENS`（**值相同**（都是 2048），**不同指** —— 兜底 vs 覆盖）。

VLLM_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP = 256
# 来源：[B] vLLM 官方源码 —— `vllm/config/scheduler.py`（main 分支，抓取日 2026-10-09），
#       常量 `DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`。
# 口径：★★ 这是 **batched-DP 场景**的另一个兜底常量（★ 与上面两条**并列，不是它的子项**）。
# 消费者：Chunk 2 的 `01-iteration-level-scheduling`（说明「兜底值有多个」）。
# 同值位置：`VLLM_API_SERVER_DEFAULT_MAX_SEQS`（**值相同**（都是 256），**不同指** —— batched-DP 兜底 vs A100 覆盖）。

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
#       ★ 完整性：抓取日（2026-10-09）上 `grep -c 'The default value here is mainly for convenience when testing'` → 2。
#       ★★ **该计数会漂移**（文档站是 developer preview）—— ★ 所以这里记作「**≥ 2，且至少含
#          `--max-num-seqs` 与 `--max-num-batched-tokens` 两条各一次**」，不写成硬编码的常数。
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
# 3. `01-iteration-level-scheduling` 的**调度算例** —— ★★ 它们是【本主题的模型假设】
#    ★★★ 本机无 GPU / 无 vLLM / 无 ORCA ⇒ 这一节**没有一个数来自真实引擎**，
#        **不得**写成「vLLM 的调度是这样」「业界通常如此」（`§0.7 C.5` 的 `F1`/`F2`）。
#    ★★ 本节的 A 类判据（间隙闭式 vs 模拟器）**只用** `SCHED_EQUAL_LEN` / `SCHED_N_GRID` /
#        `SCHED_B_GRID` 三条；其余是**收益来源实验**（`D3`/`D4`）的输入。
# ===========================================================================

SCHED_N_REQ = 200
# 来源：**[E] 本主题的算例**（不是外部事实）—— 一次调度实验里的**请求条数**。
# 口径：★ 200 条足够让「最后一批没填满」与「长度方差」两类效应都显现；
#       ★ **不是**任何引擎的并发上限（那类参数见本文件第 2 节，且**无单一默认值**）。
# 消费者：`01` 的 `A.1` / `B.1` / `B.3` / `B.4` / `C.3`。
# 同值位置：无。

SCHED_BATCH_CAP = 32
# 来源：**[E] 本主题的算例** —— **批上限**（同时**在飞**的请求数上限）。
# 口径：★★ 它是本主题的「**在飞请求数**」（`§2.2`：本域不裸写 `batch`）——
#       ★ **不是**任何引擎的 `max_num_seqs`（后者**无单一默认值**，见第 2 节）。
#       ★ 取 2 的幂使闭式在**二进制下精确**（`§0.7 D9` 的「逐位一致」靠这一点）。
# 消费者：`01` 的 `A.1` / `B.*` / `C.*`；`SCHED_GUARD_CAP` 必须等于它。
# 同值位置：`SCHED_GUARD_CAP`（同一条事实的两写，用于显式守卫）；
#           `SCHED_BURST_SIZE` / `SCHED_BURST_PERIOD` / `CHUNK_CONCURRENT_DECODE`
#           （**值相同**（都是 `32`），**不同指** —— 批上限 / 突发粒度 / 突发周期 / 并发 decode）。

SCHED_EQUAL_LEN = 100
# 来源：**[E] 本主题的算例** —— **等长**请求的长度（token）。
# 口径：★★ 闭式的**适用域**是「等长 + 同时到达」；`L = 100` 使 `N=1, B=32` 的间隙
#       = `31×100/32 = 96.875`（`A.2` 钉住它）。
# 消费者：`01` 的 `A.1` / `A.2` / `B.1` / `B.3` / `C.*` / `D.4` / `D.5` / `D.9`。
# 同值位置：无。

SCHED_LEN_LO = 20
# 来源：**[E] 本主题的算例** —— 均匀长度分布的**下界**（token）。
# 口径：★ 与 `SCHED_LEN_HI` 一起给出**长度方差**；★ 两者的均值 = `SCHED_EXP_MEAN`。
# 消费者：`01` 的 `B.1` / `B.2` / `B.4`（`gen_lens("uniform")`）。
# 同值位置：无。

SCHED_LEN_HI = 200
# 来源：**[E] 本主题的算例** —— 均匀长度分布的**上界**（token）。
# 口径：★ `[20, 200]` 的均值 = `110` = `SCHED_EXP_MEAN`（★ 两个分布**同均值、不同方差**，
#       这是「收益来自方差」这条结论的**控制变量**）。
# 消费者：`01` 的 `B.1` / `B.2` / `B.4`。
# 同值位置：无。

SCHED_EXP_MEAN = 110
# 来源：**[E] 本主题的算例** —— 指数分布长度的**均值**（token），取 `(20+200)/2 = 110`。
# 口径：★★ **它与 `SCHED_LEN_LO`/`HI` 的均值相同**（`110 = (20+200)/2`）——
#       ★★ 于是「均匀 vs 指数」只差**方差**（**错误模式 `P38`** 的隔离手段：一次只改一个因素）。
# 消费者：`01` 的 `B.2`（`gen_lens("exp")`）。
# 同值位置：无（★ 与 `SCHED_LEN_LO`/`HI` 是**同一条链的上下游**，值不同）。

SCHED_SEED = 20261009
# 来源：**[E] 本主题的算例** —— 长度分布与到达过程的伪随机种子（`numpy.random.default_rng`）。
# 口径：★★ 种子**钉死**使 `B.2`/`B.3` 的比值**可复现**；★ 改变它会得到**另一组**长度/到达
#       （结论仍同向，但**具体的比值会变** ⇒ 正文里的数必须与它绑定）。
# 消费者：`01` 的 `B.1` / `B.2` / `B.3` / `B.4` / `C.1` / `C.3`。
# 同值位置：无。

SCHED_GUARD_CAP = 32
# 来源：**[E] 本主题的算例** —— `A.2` 两端守卫**显式**使用的批上限。
# 口径：★★ **必须等于 `SCHED_BATCH_CAP`**（`D.3b` 的口径守卫钉住这一点）——
#       ★ 它的作用是让「`B | N` ⇒ 间隙 0」与「`N=1` ⇒ 间隙 `(B−1)L/B`」两条守卫
#       **不依赖网格的遍历顺序**。
# 消费者：`01` 的 `A.2` / `D.4` / `D.5` / `D.9`。
# 同值位置：`SCHED_BATCH_CAP`（同一条事实的两写）。

SCHED_N_GRID = (1, 7, 32, 100, 200)
# 来源：**[E] 本主题的算例网格** —— 闭式对拍用的请求数网格。
# 口径：★★★ **覆盖必须两端都有**（`D.3b` 守卫）：`1`（间隙的**极端**，一批只装一个）、
#       `32`/`200`（与 `SCHED_B_GRID` 里的 `8`/`16`/`32` 构成 **`B | N`** 的三种情形）
#       —— ★ 缺了任一端，`A.2` 的两端守卫就**退化成恒真**（Chunk 1 的 `H1` 形态）。
# 消费者：`01` 的 `A.1` / `A.3` / `D.3b`。
# 同值位置：无。

SCHED_B_GRID = (8, 16, 32, 64)
# 来源：**[E] 本主题的算例网格** —— 批上限网格（**全是 2 的幂**）。
# 口径：★★ 取 2 的幂使闭式的除法在**二进制下精确** ⇒ 「逐位一致」可以用 `==`（`§0.7 D9`）。
#       ★ 含 `SCHED_BATCH_CAP`（`32`）与小于/大于它的档。
# 消费者：`01` 的 `A.1` / `A.3` / `D.3b`。
# 同值位置：无。

SCHED_POISSON_LAMBDA_FAST = 8.0
# 来源：**[E] 本主题的算例** —— 「**远快于处理**」的到达速率（条 / 时间单位）。
# 口径：★★ 处理速率 ≈ `BATCH_CAP / EQUAL_LEN = 0.32` ⇒ `λ=8` 是它的 **25 倍**
#       ⇒ 系统**始终饱和** ⇒ ★ 到达**节奏**几乎不起作用（`B.3`）。
# 消费者：`01` 的 `B.3` / `D.7`。
# 同值位置：无。

SCHED_POISSON_LAMBDA_SLOW = 1.2
# 来源：**[E] 本主题的算例** —— 「**约为处理速度**」的到达速率（`1.2 ≈ 4 × 0.32`）。
# 口径：★★ 仍**高于**处理速率（系统饱和），但比 `λ=8` 慢得多 ⇒ ★ 用来检验
#       「到达过程」（而非到达**速率**）是否影响收益。
# 消费者：`01` 的 `B.3` / `C.1` / `D.7`。
# 同值位置：无。

SCHED_BURST_SIZE = 32
# 来源：**[E] 本主题的算例** —— 突发到达的**每次突发条数**。
# 口径：★★★ **它必须与批上限一起读**：`S = 32 = SCHED_BATCH_CAP` ⇒ **一次突发恰好填满一批**
#       ⇒ 静态批**不必等**⇒ 收益**恰为 `1.0000×`**（`C.3` 的守卫）。
#       ★ `burst_small` 用它的**一半**（`16 < 32`）来做反例。
# 消费者：`01` 的 `B.3` / `C.3` / `D.7`。
# 同值位置：`SCHED_BATCH_CAP`（**值相同、不同指** —— 一个是批上限，一个是突发粒度）。

SCHED_BURST_PERIOD = 32
# 来源：**[E] 本主题的算例** —— 突发的**周期**（时间单位）。
# 口径：★ 与 `SCHED_BURST_SIZE` 相等 ⇒ 平均到达速率 = `1.0` / 时间单位（> 处理速率 `0.32`）
#       ⇒ 系统**饱和**（★ 否则测到的是「到达太慢」而不是「到达过程本身」）。
# 消费者：`01` 的 `B.3` / `C.3` / `D.7`。
# 同值位置：`SCHED_BATCH_CAP` / `SCHED_GUARD_CAP` / `SCHED_BURST_SIZE` / `CHUNK_CONCURRENT_DECODE`
#           （**值相同**（都是 `32`），**不同指** —— 不得合并）。

SCHED_FETCH_DAY = "2026-10-09"
# 来源：**[E] 本库自记** —— 它记的是「本主题所有外部条目的抓取日」（计划 `§0.7 C` / 本文件第 1、2 节）。
# 口径：★★ 不写抓取日的引文不算引文；★ 本主题的 vLLM 条目**同日抓取**。
# 消费者：`01` 的 `D.3` · `02` 的 `D.3`（两个程序各钉一次）。
# 同值位置：本文件第 1、2 节各条目的 `# 来源：` 注释。


# ===========================================================================
# 4. `02-chunked-prefill` 的**分块数算例** —— ★★ 【本主题的模型假设】（不是外部事实）
#    ★★★ 模型：**一步耗时 = `max(t_d, t_p/k)`**、`TTFT = k × 每步耗时`。
#      ★★ **这是简化**：真实里 prefill 与 decode **共享算力**（不是简单取 `max`），
#      且 `C` 个 decode 请求与 prefill 块的相对成本随实现变（`[B]` 级，本机**无引擎**）。
#      ⇒ ★ 本节的数**不得**写成「某引擎的 chunked prefill 参数」。
# ===========================================================================

CHUNK_PREFILL_MS = 400
# 来源：**[E] 本主题的模型假设** —— 整段 **prefill 的工作量**（时间单位，量级 = ms）。
# 口径：★★ 它被切成 `k` 块 ⇒ **每块 `t_p/k`**；★ 与 `CHUNK_DECODE_STEP_MS` **整除**
#       （`400 / 20 = 20`）⇒ `k*` 处 TTFT **恰好**回落到 `t_p`（`A.2` 的落点）。
# 消费者：`02` 的 `A.*` / `B.*` / `C.*` / `D.*`。
# 同值位置：`CHUNK_PREFILL_NONDIV_MS`（**同一含义的另一个取值** —— 用于「不整除」的边界，`C.4`）。

CHUNK_DECODE_STEP_MS = 20
# 来源：**[E] 本主题的模型假设** —— 一步里**全部并发 decode** 的耗时（= `t_d`）。
# 口径：★★ `k* = ⌈t_p/t_d⌉ = 20`；★ 它是「每步耗时的地板」——`k ≥ k*` 时每步耗时不再下降。
# 消费者：`02` 的 `A.*` / `B.*` / `C.*` / `D.*`。
# 同值位置：无。

CHUNK_CONCURRENT_DECODE = 32
# 来源：**[E] 本主题的模型假设** —— 与 prefill 块**混跑**的**并发 decode 请求数**。
# 口径：★ 它只用于说明「一批里有什么」（`B.3`）—— ★ 本模型把这 `C` 个请求合起来的耗时
#       记作 `t_d`（不再拆到每个请求）；真实的每人延迟取决于 kernel，属 `[B]`（本机无引擎）。
# 消费者：`02` 的 `B.3` / `D.7`。
# 同值位置：`SCHED_BATCH_CAP` / `SCHED_GUARD_CAP` / `SCHED_BURST_SIZE` / `SCHED_BURST_PERIOD`
#           （**值相同**（都是 `32`），**不同指** —— 并发 decode vs 批上限 / 突发粒度 / 突发周期）。

CHUNK_K_GRID = (1, 2, 4, 8, 16, 20, 32, 64, 128)
# 来源：**[E] 本主题的算例网格** —— 分块数 `k` 的网格。
# 口径：★★★ **必须同时含** `k<k*`（`1..16`）、`k=k*`（`20`）、`k>k*`（`32/64/128`）三段
#       （`D.3b` 守卫）—— ★ 缺任一段，`A.2` 的三段性质就**退化成恒真**。
#       ★ 含 `k=1`（不分块的退化守卫）。
# 消费者：`02` 的 `A.1` / `A.2` / `A.4` / `B.1` / `C.1` / `D.3b`。
# 同值位置：无。

CHUNK_SCAN_MAX = 4096
# 来源：**[E] 本主题的算例** —— `k*` 的**第二条路径**（在模拟器上扫描）的搜索上界。
# 口径：★ 必须 `> k*`（否则扫描找不到）—— `4096 ≫ 20`。
# 消费者：`02` 的 `A.3`。
# 同值位置：无。

CHUNK_PREFILL_NONDIV_MS = 401
# 来源：**[E] 本主题的算例** —— **不被 `t_d` 整除**的 prefill 工作量（`401 / 20 = 20.05`）。
# 口径：★★ 它是「不整除 ⇒ `k*` 仍取 `⌈·⌉`，但 TTFT **略高于** `t_p`」这一边界的落点
#       （`C.4`）；★ 也是 `D.5` 守卫（`ceil` vs `floor`）的**唯一**能暴露取整方向的取值。
#       ★ 必须**不被** `CHUNK_DECODE_STEP_MS` 整除 —— 否则该守卫退化成恒真。
# 消费者：`02` 的 `A.3` / `C.4` / `D.5` / `D.7`。
# 同值位置：`CHUNK_PREFILL_MS`（**同一条链的另一个取值** —— 整除 vs 不整除）。

CHUNK_HUGE_K = 1000000
# 来源：**[E] 本主题的算例** —— 「`k` 极大」的极值（`10^6`，写成十进制字面量）。
# 口径：★★ 它演示 `k→∞ ⇒ TTFT` **无界**（`A.4`/`C.3`）：`TTFT = k·t_d = 2×10^7`。
#       ★ 这个数是**构造出来的极值**，**不是**任何实现的配置范围。
# 消费者：`02` 的 `A.4` / `C.3` / `D.7`。
# 同值位置：无。


# ===========================================================================
# 5. `03-preemption-and-fairness` 的**抢占代价账** —— ★★ 【本主题的模型假设】
#    ★★★ 本机无 GPU / 无引擎 ⇒ 这一节**没有一个数来自真实系统**；
#        ★ 换出/重算的**公式**是本文的知识，**具体取值是可调的旋钮**。
# ===========================================================================

PREEMPT_KV_BYTES_PER_TOKEN = 524288
# 来源：**[E] 本库** —— `130-quantization/kv-cache-quantization/01`：Llama-2-7B（MHA）**每 token `512 KiB`**。
# 口径：★★★ 本主题**取用**这个值，**不重推**它的六项乘积（那条正文在 130-01）——
#       ★ 它是 `[E]` 级（本库推导值），**不是**任何引擎的规格、**也不是**任何真实硬件的测量。
#       ★★ **口径 = KiB**（`512 × 1024`）。
#       ★ 换成别的模型会改变「换出」的**绝对值**，但**不改公式的形状**（`s*` 的闭式）。
# 消费者：`03` 的 `A.1` / `A.2` / `A.3` / `B.1`。
# 同值位置：`130-quantization/kv-cache-quantization/01` 的表 1（**那是正文，这里是引用**）；
#           `120-inference/kv-cache/code/specs.py` 的 `LLAMA2_7B_KV_BYTES_PER_TOKEN`（**同值**，
#           分属两个主题的 specs —— ★ **不是同一条事实**，各自只服务本主题）。

PREEMPT_SWAP_BW_BYTES_PER_MS = 32000000.0
# 来源：**[E] 本主题的模型假设** —— 换出时**主机侧单向搬运带宽**（字节 / 毫秒），取 `32 GB/s` 量级。
# 口径：★★★ **方向与口径必须写清**：本值是「**单向、载荷口径**」（真正能搬数据的字节率）——
#       ★★ 三种口径（线速率 / 载荷 / 双向合计）的正文在 `20-gpu-architecture/interconnect/01`，
#       **本主题只回链、不重述**；★ 换出代价里的 **`× 2`** 就是因为要**搬出再搬回**（出 + 入）。
#       ★ **不指代任何具体硬件**（本机**无 GPU**）⇒ **不得**写成「PCIe Gen5 就是 32 GB/s」。
#       ★★ 本主题的知识是**公式** `s* = (2·bpt/BW − τ_lin)/τ_quad`，**不是这个具体值**。
# 消费者：`03` 的 `A.1` / `A.2` / `A.3` / `B.1` / `C.2` / `C.3`。
# 同值位置：无。

PREEMPT_SWAP_BLOCK_TOKENS = 16
# 来源：**[E] 本主题的算例** —— 换出账本**按块搬运**时的块大小（token 数）。
# 口径：★★ 它只影响**账本的循环粒度**，**不影响合计数**（`Σ == 2·s·bpt/BW`）——
#       这正是 `A.1` 要断言的「**两条路径一致**」。★ 取 `16` 与 KV 块粒度的常见值一致
#       （**同名不同指**：这里的 `16` 只是本账本的搬运粒度）。
# 消费者：`03` 的 `A.1` / `D.7`。
# 同值位置：无（★ 与 vLLM 的块大小 `16` **不是**同一条事实）。

PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN = 0.02
# 来源：**[E] 本主题的模型假设** —— **KV 重算**的**线性项**（每 token 的 GEMM / MLP 成本）。
# 口径：★★ 重算 = 把这段 prefill **重跑一遍**；★ 线性项代表与 `s` 成正比的那部分
#       （矩阵乘 / MLP）。★ **不得**写成「某引擎实测 per-token prefill 耗时」。
# 消费者：`03` 的 `A.1` / `A.2` / `A.3` / `B.1` / `D.7`。
# 同值位置：无。

PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ = 0.000003192
# 来源：**[E] 本主题的模型假设** —— **KV 重算**的**二次项**（每 token² 的自注意力成本）。
# 口径：★★★ **它是「换出 vs 重算」出现【长度临界点】的唯一来源**：
#       ★ 自注意力对 `s` 是**二次**的 ⇒ 重算对**长序列**变得不可行，而换出只**线性**涨。
#       ★ 取 `3.192e-6` 使临界长度 `s* = (0.032768 − 0.02)/3.192e-6 = **4000**`（一个整数）。
#       ★ 若把它取 `0`（纯线性），则**没有长度临界点** —— 两条线**恒不成交叉**或只在一个点相遇。
# 消费者：`03` 的 `A.1` / `A.2` / `A.3` / `B.1` / `C.1` / `C.3` / `D.5` / `D.7`。
# 同值位置：无。

PREEMPT_LEN_GRID = (256, 512, 1024, 2048, 4000, 8192, 16384)
# 来源：**[E] 本主题的算例网格** —— 被抢占序列的**长度网格**。
# 口径：★★ **必须横跨临界点 `s* = 4000`**（`D.3b` 守卫）：`4000` 恰好在网格内 ⇒
#       「`s<s*` 重算更便宜 / `s>s*` 换出更便宜」两侧都有落点。
# 消费者：`03` 的 `A.1` / `A.2` / `B.1` / `D.3b`。
# 同值位置：无。

PREEMPT_SCAN_MAX = 65536
# 来源：**[E] 本主题的算例** —— 账本**二分找临界点**的搜索上界（`≫ s* = 4000`）。
# 口径：★ 必须 `> s*`，否则二分落在搜索区间外。
# 消费者：`03` 的 `A.2`。
# 同值位置：无。

FAIR_SERVICE_TOKENS = (7, 2, 9, 4, 1, 5)
# 来源：**[E] 本主题的算例** —— **公平性对照**里 6 个作业的**服务时间**（按**到达顺序**给出）。
# 口径：★★ 顺序**故意打乱**（不是升序）⇒ FCFS 与 SJF 的结果**不同**；
#       ★ 用来演示「SJF 降平均完成时间（`A.4` 穷举证明）而 FCFS 不最优」。
# 消费者：`03` 的 `A.4` / `B.2` / `D.7`。
# 同值位置：无。

FAIR_STARVE_SHORT = 3
# 来源：**[E] 本主题的算例** —— **动态饥饿**演示里「短作业」的服务时间。
# 口径：★★ 长作业（服务 `10^5`）在**持续到达**的短作业洪流下**永远轮不到** ⇒ 等待**无界**
#       （`C.4`）；★ 这是**静态集合上不出现**的现象（那时 FCFS 的最大等待 ≥ SJF 的）。
# 消费者：`03` 的 `C.4` / `D.7`。
# 同值位置：无。

FAIR_STARVE_LONG = 100000
# 来源：**[E] 本主题的算例** —— 动态饥饿演示里的「**长作业**」服务时间。
# 口径：★ 与 `FAIR_STARVE_SHORT` 形成**两个数量级**的差 ⇒ 饥饿效应可量化。
# 消费者：`03` 的 `C.4` / `D.7`。
# 同值位置：无。


# ===========================================================================
# 6. UNSOURCED —— 待办清单，不是免责声明
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
