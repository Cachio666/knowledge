"""120-inference/serving-infra 的外部事实常量源。

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
  **vLLM 官方页面上【没有 `TPOT` 这个词】** —— 它用的是 **inter-token latency** 与
  **time_per_output_token**（见 `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES` 与 `VLLM_ITL_DEFINITION_QUOTE`）。
  ⇒ ★★ **本域写 `TPOT` 时必须注明它是社区常用缩写**（`TPOT_COMMUNITY_ABBREVIATION_NOTE`），
  **不得**把它当成 vLLM 的官方词（`F9`）。

★ **本主题的边界**（本主题**一行都不重写**，只回链）：
  · **通信的机制**（集合通信原语 / NCCL / 拓扑 / 重叠）→ ★ `110-communication` 全域；
  · **并行策略本身的正文**（数据/张量/流水/序列/专家/状态六轴）→ ★ `100-parallelism` 全域。

★ **本主题的判据纪律**：
  ★★ **不能整主题标「无 A」** —— **排队与负载均衡本质是可独立参照的算法** ⇒ **有 A 类**；
  ★ **只有**「**真实网络 / 多机的行为**」**不可测**（本机单机，无真网络 —— `§0.7 D7`）。

★ **本机的限定语**：本机**无 GPU、无真网络、单机**
  ⇒ ★★★ **本主题最强的一条限定语：**「**不是任何真实引擎，也不是真机**」；
  ★ 本机 socket 的任何数字都**不得**用作依据（`P34`）。
"""


# ===========================================================================
# 1. TTFT / 逐 token 延迟（vLLM 官方定义）—— [B] 产品官方文档
#    载体 = 文档 `usage/metrics/` 与 `features/per_request_metrics/`；抓取日 **2026-10-09**。
# ===========================================================================

VLLM_TTFT_DEFINITION_QUOTE = "Time from when the request was scheduled until the first output token was generated"
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `time_to_first_token_ms` 说明（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'Time from when the request was scheduled until the first output token was generated'` → 1。
# 口径：★★★ 这是 TTFT 的**官方逐字定义** —— ★ 注意起算点是「**request was scheduled**」（**不含排队**）⇒
#       ★★ 与本库「TTFT 含排队 + prefill」的口径**要分开写**（排队时间在 vLLM 里是另一个量）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（程序 `code/01_*.py` 尚未落盘）—— TTFT 的定义。
# 同值位置：无。

VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES = 0
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 与 `features/per_request_metrics/`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'TPOT'` 在这两页的**原始 HTML 上均为 0**。
# 口径：★★★ **这条「0」本身就是一个可断言的常量** —— 它证明 **vLLM 官方页面不用 `TPOT` 这个词**。
#       ⇒ ★★ 本域写 `TPOT` 时**必须注明它是社区常用缩写**，其逐字定义在 vLLM 里对应 inter-token latency
#       （`VLLM_ITL_DEFINITION_QUOTE`）；★ **不得**裸写 `TPOT` 当成 vLLM 的官方词（`F9`）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（禁写清单的逐字依据）。
# 同值位置：无。

VLLM_ITL_DEFINITION_QUOTE = "Mean inter-token latency (average time between successive output tokens) during the decode phase."
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `mean_itl_ms` 说明（抓取日 2026-10-09）。
#       ★ 完整性：`mean_itl_ms` 是「Mean inter-token latency ... during the decode phase. null for single-token responses.」。
# 口径：★★★ 这是 **vLLM 对「逐 token 时间」的官方口径** —— ★ **vLLM 的词是 inter-token latency，
#       不是 `TPOT`**（见 `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES`）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（第二个延迟的定义）。
# 同值位置：`VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE`（**同一个量的另一处名字**「time per output token」）。

VLLM_ITL_METRIC_QUOTE = "Histogram of inter-token latency in seconds."
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 的 `vllm:inter_token_latency_seconds`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'Histogram of inter-token latency in seconds'` → 1。
# 口径：★★ 这是 **Prometheus 指标侧**的 inter-token latency 口径（★ 与 per-request 的 `mean_itl_ms`
#       **同一个量的两处写法** —— 一个是指标、一个是每请求）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（指标口径）。
# 同值位置：`VLLM_ITL_DEFINITION_QUOTE`（**同值不同指**：指标 vs per-request）。

VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE = "Histogram of time_per_output_token_seconds per request."
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 的 `vllm:request_time_per_output_token_seconds`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'time_per_output_token_seconds per request'` → 1。
# 口径：★★★ 这是 **vLLM 对「逐 token 时间」的【另一个名字】**（`time per output token`）——
#       ★★ **`TPOT` 这个缩写就是它 + inter-token latency 的社区归并**（官方页面不拼 `TPOT`）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（`TPOT` 与官方词的对应关系）。
# 同值位置：`VLLM_ITL_DEFINITION_QUOTE`（**同一个量的不同名字**）。

VLLM_GENERATION_TIME_DEFINITION_QUOTE = "Decode time: time from the first output token to the last output token. Excludes both queue wait and prefill/TTFT."
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `generation_time_ms` 说明（抓取日 2026-10-09）。
#       ★ 完整性：该句在该页原文。
# 口径：★★ 它**显式排除排队与 prefill** ⇒ ★★ 它与 TTFT（`VLLM_TTFT_DEFINITION_QUOTE`）是
#       **两个不重叠的时段**（一个从 scheduled 到首 token，一个从首 token 到末 token）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（两个时段的边界）。
# 同值位置：无。

TPOT_COMMUNITY_ABBREVIATION_NOTE = "TPOT 是社区常用缩写（time per output token）；vLLM 官方页面不用 TPOT，其词是 inter-token latency（mean_itl_ms / request_time_per_output_token_seconds）"
# 来源：**[E] 本库的消歧记录**（★ 不是任何来源里的话）—— 依据是 `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES`
#       与 `VLLM_ITL_DEFINITION_QUOTE` / `VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE` 的**并列**（`§0.7 C` 事实 12 / `F9`）。
# 口径：★★★ 它是一条**「同名不同指 / 规范与实现不分」的挡板**：把 `TPOT` 当成 vLLM 的官方词会**写错正文**。
#       ★ 它**没有外部出处**（是对事实的记载而非引文）⇒ 不得把它写成引文（本库纪律：**自造真值**是一种错误模式）。
#       ★ 它的存在理由是**可复核**：写 `TPOT` 时「有没有注明是社区缩写」变成一条可被变异测试抓住的断言。
# 消费者：Chunk 5 的 `01-latency-and-slo`（首次出现 `TPOT` 处点明）。
# 同值位置：无。


# ===========================================================================
# 2. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在本模块的正文常量里。
# ★★ **本元组【留空】**，理由与 `110-communication/collectives` 逐字相同：**每一项的字符串都会成为 `make mutate`
#   的叶子**，而它们**没有消费者** ⇒ 会全部记成**漏网**。★★ **而【空元组本身】有消费者**：
#   编号程序的 `preflight()` 用 `specs.UNSOURCED == ()` 钉住它。
# ★ **本主题的「未核到」写在注释里，不立常量**：
#   ① **`TPOT` 缩写在 vLLM 官方 metrics 页的展开** —— 未取到（因为该缩写根本不出现，见上）；
#   ② **多机 / 真网络下的 PD 分离开销** —— 本机单机、无真网络 ⇒ ★ 只能用**规格带宽推导**（标为推导值，`T75`）。
UNSOURCED = ()
