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
# 消费者：Chunk 5 的 `01-latency-and-slo`（程序 `code/01_latency_and_slo.py`，已落盘）—— TTFT 的定义。
# 同值位置：无。

VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES = 0
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 与 `features/per_request_metrics/`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'TPOT'` 在这两页的**原始 HTML 上均为 0**。
# 口径：★★★ **这条「0」本身就是一个可断言的常量** —— 它证明 **vLLM 官方页面不用 `TPOT` 这个词**。
#       ⇒ ★★ 本域写 `TPOT` 时**必须注明它是社区常用缩写**，其逐字定义在 vLLM 里对应 inter-token latency
#       （`VLLM_ITL_DEFINITION_QUOTE`）；★ **不得**裸写 `TPOT` 当成 vLLM 的官方词（`F9`）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（禁写清单的逐字依据；程序 `code/01_latency_and_slo.py`）。
# 同值位置：无。

VLLM_ITL_DEFINITION_QUOTE = "Mean inter-token latency (average time between successive output tokens) during the decode phase."
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `mean_itl_ms` 说明（抓取日 2026-10-09）。
#       ★ 完整性：`mean_itl_ms` 是「Mean inter-token latency ... during the decode phase. null for single-token responses.」。
# 口径：★★★ 这是 **vLLM 对「逐 token 时间」的官方口径** —— ★ **vLLM 的词是 inter-token latency，
#       不是 `TPOT`**（见 `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES`）。
# 消费者：Chunk 5 的 `01-latency-and-slo`（第二个延迟的定义；程序 `code/01_latency_and_slo.py`）。
# 同值位置：`VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE`（**同一个量的另一处名字**「time per output token」）。

VLLM_ITL_NULL_FOR_SINGLE_TOKEN_QUOTE = "null for single-token responses"
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `mean_itl_ms` 说明的后半句（抓取日 2026-10-09）。
# 口径：★★ 它是「该指标在**单 token 响应**上**不可定义**」的逐字依据 ⇒ 写「每输出 token 时间」时
#       **必须说明它的分母口径**（响应只有 1 个 token 时它无意义）。
# 消费者：Chunk 5 的 `03-scaling-and-observability`（`C.1` 的指标口径）。
# 同值位置：`VLLM_ITL_DEFINITION_QUOTE`（同一条说明的**前半句与后半句**）。

VLLM_ITL_METRIC_QUOTE = "Histogram of inter-token latency in seconds."
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 的 `vllm:inter_token_latency_seconds`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'Histogram of inter-token latency in seconds'` → 1。
# 口径：★★ 这是 **Prometheus 指标侧**的 inter-token latency 口径（★ 与 per-request 的 `mean_itl_ms`
#       **同一个量的两处写法** —— 一个是指标、一个是每请求）。
# 消费者：Chunk 5 的 `01-latency-and-slo` / `03-scaling-and-observability`（指标口径；程序 `code/01_*.py` 与 `code/03_*.py`）。
# 同值位置：`VLLM_ITL_DEFINITION_QUOTE`（**同值不同指**：指标 vs per-request）。

VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE = "Histogram of time_per_output_token_seconds per request."
# 来源：[B] vLLM 官方文档 —— `usage/metrics/` 的 `vllm:request_time_per_output_token_seconds`（抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'time_per_output_token_seconds per request'` → 1。
# 口径：★★★ 这是 **vLLM 对「逐 token 时间」的【另一个名字】**（`time per output token`）——
#       ★★ **`TPOT` 这个缩写就是它 + inter-token latency 的社区归并**（官方页面不拼 `TPOT`）。
# 消费者：Chunk 5 的 `01-latency-and-slo` / `03-scaling-and-observability`（`TPOT` 与官方词的对应关系）。
# 同值位置：`VLLM_ITL_DEFINITION_QUOTE`（**同一个量的不同名字**）。

VLLM_GENERATION_TIME_DEFINITION_QUOTE = "Decode time: time from the first output token to the last output token. Excludes both queue wait and prefill/TTFT."
# 来源：[B] vLLM 官方文档 —— `features/per_request_metrics/` 的 `generation_time_ms` 说明（抓取日 2026-10-09）。
#       ★ 完整性：该句在该页原文。
# 口径：★★ 它**显式排除排队与 prefill** ⇒ ★★ 它与 TTFT（`VLLM_TTFT_DEFINITION_QUOTE`）是
#       **两个不重叠的时段**（一个从 scheduled 到首 token，一个从首 token 到末 token）。
# 消费者：Chunk 5 的 `01-latency-and-slo` / `03-scaling-and-observability`（两个时段的边界）。
# 同值位置：无。

TPOT_COMMUNITY_ABBREVIATION_NOTE = "TPOT 是社区常用缩写（time per output token）；vLLM 官方页面不用 TPOT，其词是 inter-token latency（mean_itl_ms / request_time_per_output_token_seconds）"
# 来源：**[E] 本库的消歧记录**（★ 不是任何来源里的话）—— 依据是 `VLLM_TPOT_OCCURRENCES_ON_DOC_PAGES`
#       与 `VLLM_ITL_DEFINITION_QUOTE` / `VLLM_TIME_PER_OUTPUT_TOKEN_METRIC_QUOTE` 的**并列**（`§0.7 C` 事实 12 / `F9`）。
# 口径：★★★ 它是一条**「同名不同指 / 规范与实现不分」的挡板**：把 `TPOT` 当成 vLLM 的官方词会**写错正文**。
#       ★ 它**没有外部出处**（是对事实的记载而非引文）⇒ 不得把它写成引文（本库纪律：**自造真值**是一种错误模式）。
#       ★ 它的存在理由是**可复核**：写 `TPOT` 时「有没有注明是社区缩写」变成一条可被变异测试抓住的断言。
# 消费者：Chunk 5 的 `01-latency-and-slo`（首次出现 `TPOT` 处点明）。
# 同值位置：`02` / `03` 也各有一条「`TPOT` 是社区缩写」的断言，**共用这一条记录**。


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


# ===========================================================================
# 3. 「推理的并行 / 通信 / serving 栈」的硬边界 —— [E] 本库的跨域回链锚点（不重推，只回链）
#    ★★★ 本主题**一行都不重写**这些正文；下面每条都只是**回链目标**，由 `02` 的 `D.3b` 逐字钉住。
# ===========================================================================

PIPELINE_BUBBLE_REF = "100-parallelism/pipeline-parallel/01-microbatches-and-bubbles"
# 来源：[E] 本库的跨域回链 —— **训练侧**的气泡模型与 `(m+p−1)(t_f+t_b)` / 两个口径（GPipe 的
#       `(K−1)/(M+K−1)`、Megatron 的 `(p−1)/m`）的**正文在彼处**。
# 口径：★★★ 本主题**不得重推**那条推导（`P37`）—— 只写**推理侧的差别**：无反向 ⇒ 只有 `t_f`。
# 消费者：`02` 的 `D.3b`（回链锚点逐字钉住）。
# 同值位置：`100-parallelism/pipeline-parallel/01-microbatches-and-bubbles`（**那是正文，这里是回链**）。

PARALLELISM_LETDOWN_REF = "100-parallelism/README.md:90"
# 来源：[E] 本库的**反向契约** —— 该行把「推理的 TP/PP/EP、PD 分离、KV cache 切分」让渡给本域。
# 口径：★★★ 建成后**该行要去掉「未建成」并反向回链到本主题的 `02`**（收尾办，计划 `§0.7 B.1`）。
# 消费者：`02` 的 `D.3b`。
# 同值位置：`100-parallelism/README.md:90`（**那是让渡声明，这里是对它的引用**）。

COMMUNICATION_LETDOWN_REF = "110-communication/README.md:61"
# 来源：[E] 本库的**反向契约** —— 该行把「KV cache 的**传输**、PD 分离、推理的 TP/PP/EP」让渡给本域。
# 口径：★★★ 通信的**机制**（集合通信原语 / NCCL / 拓扑 / 重叠）**仍在 `110-communication`** ——
#       本域只写**推理侧的通信需求**（KV 的传输量与时延），机制一律回链。
# 消费者：`02` 的 `D.3b`。
# 同值位置：`110-communication/README.md:61`（**那是让渡声明，这里是对它的引用**）。

SERVING_STACK_LETDOWN_REF = "50-performance/README.md:49"
# 来源：[E] 本库的**反向契约** —— 该行逐字写「**不重讲 serving 栈**」（把 serving 栈划给本域）。
# 口径：★★★ **这正是本主题的定位** —— `50-performance` 明说不重讲的，就是 `03-scaling-and-observability` 要讲的。
# 消费者：`03` 的 `D.3b`。
# 同值位置：`50-performance/README.md:49`（**那是让渡声明，这里是对它的引用**）。

KERNEL_AI_REF = "40-kernels/attention/05"
# 来源：[E] 本库的跨域回链 —— **prefill 与 decode 的 `AI` 差 120 倍**那条账的**正文在彼处**
#       （`05-decoding-attention.md:229-236`），`120-inference/batching-and-scheduling/01` 也只是回链。
# 口径：★★★ 本主题**不得重算** `AI`（`P37`）—— 只**引用**「两者在数量级上不同」这个已被写下的结论。
# 消费者：`02` 的 `D.3b`。
# 同值位置：`40-kernels/attention/05-decoding-attention`（**那是正文，这里是回链**）。

BATCHING_CHUNKED_REF = "120-inference/batching-and-scheduling/02-chunked-prefill"
# 来源：[E] 本库的**同域前驱** —— chunked prefill 的**分块数权衡**（`k* = ⌈t_p/t_d⌉`）与它「取较慢者」的
#       耦合模型的**正文在彼处**（★ 同域前驱用**普通链接**，不是跨域回链）。
# 口径：★★★ 本主题**不得重推** `k*`（`P37`）—— `02` 只把「不切块的耦合（`k = 1`）」当作 **PD 分离的对照基线**。
# 消费者：`02` 的 `D.3b`。
# 同值位置：`120-inference/batching-and-scheduling/02-chunked-prefill`（**那是正文，这里是回链**）。

KV_HIT_RATE_REF = "120-inference/kv-cache/03-prefix-and-eviction"
# 来源：[E] 本库的**同域前驱** —— **KV 缓存**侧的「`hash → 块`、命中率定义（分母 = 全部块）、LRU」的正文在彼处。
# 口径：★★★ 本主题的「**路由命中率**」是**另一个量**（★ 请求被**分配到哪个副本**这一动作的属性，
#       不是缓存内部的查找结果）—— 两者**同名不同指**，`03` 必须写清并回链。
# 消费者：`03` 的 `D.3b`。
# 同值位置：`120-inference/kv-cache/03-prefix-and-eviction`（**那是正文，这里是回链**）。

KV_ACCOUNT_REF = "130-quantization/kv-cache-quantization/01"
# 来源：[E] 本库的跨域回链 —— **KV 缓存的【账】**（六项乘积 / `AI` / GQA 的坑）的正文在彼处。
# 口径：★★★ 本主题**不得重算**那笔账（`P37`）—— `02` 只把「每 token 的 KV 字节数」当作**给定的模型参数**
#       （见 `PD_KV_BYTES_PER_TOKEN`），**不重推它的来源**。
# 消费者：`02` 的 `D.3b`。
# 同值位置：`130-quantization/kv-cache-quantization/01`（**那是正文，这里是回链**）。


# ===========================================================================
# 4. `01-latency-and-slo` 的模型参数 —— [E] 本主题的算例（**不是任何真实引擎的实测**）
#    ★★ 模型（写在正文里）：一步的耗时 = `LAT_STEP_FIXED + LAT_STEP_PER_REQ × 在飞请求数`。
#       ⇒ **ITL（步耗时）随批增大而变差**，而**吞吐 `n / 步耗时` 随批增大而变好** —— 这就是两延迟的冲突。
# ===========================================================================

SERVING_INFRA_FETCH_DAY = "2026-10-09"
# 来源：**[E] 本库自记** —— 它记的是「本主题所有外部条目的抓取日」（计划 `§0.7 C` 的题头）。
# 口径：★★ 不写抓取日的引文不算引文；★ 本主题的 vLLM 条目与全库其余各域同日抓取。
# 消费者：`01` / `02` / `03` 的 `D.3b`（三个程序各钉一次）。
# 同值位置：本文件第 1 节各条目的 `# 来源：` 注释。

LAT_N_REQ = 60
# 来源：**[E] 本主题的算例** —— 一次性到达的请求条数。
# 口径：★ 取 60 是为了让 `LAT_BATCH_GRID` 的每一档都产生**整数个相位**（60 能被 1/2/4/10/20/30/60 整除）。
# 消费者：`01` 的 `A.1` / `A.2` / `A.3` / `B.1` / `B.2` / `D.7`。
# 同值位置：无。

LAT_DEC_TOKENS = 12
# 来源：**[E] 本主题的算例** —— 每条请求要**自回归解码**出的 token 数。
# 口径：★ 每条请求的 decode 步数（等长 ⇒ 让「ITL vs 批大小」成为唯一变量）。
# 消费者：`01` 的 `A.1` / `A.2` / `A.3` / `B.1` / `B.2` / `D.7`。
# 同值位置：无。

LAT_PREFILL_COST = 5.0
# 来源：**[E] 本主题的算例** —— **一次 prefill 步的耗时**（简化：记为一次性的固定延迟，不占调度步的时间）。
# 口径：★★★ **这是一条写明的简化** —— 真实里 prefill 与 decode **共享算力**（耦合模型正文在
#       `120-inference/batching-and-scheduling/02-chunked-prefill`）；本模型把 prefill 记成**独立的 prefill 步**。
# 消费者：`01` 的 `A.2` / `A.3` / `B.4` / `D.7`。
# 同值位置：无。

LAT_STEP_FIXED = 2.0
# 来源：**[E] 本主题的算例** —— 步耗时模型里的**固定项** `α`（与在飞请求数无关的开销）。
# 口径：★★ 它是「批处理能省掉的那部分」—— `β` 才是随批线性增长的那部分（两者之比决定吞吐的上界）。
# 消费者：`01` 的 `A.1` / `B.1` / `B.2` / `D.7`。
# 同值位置：无。

LAT_STEP_PER_REQ = 0.5
# 来源：**[E] 本主题的算例** —— 步耗时模型里的**每请求项** `β`。
# 口径：★★★ **吞吐的渐近上界 = `1 / β`**（`n → ∞`）—— 这是一个**免费守卫**（`01` 的 `B.2`）。
# 消费者：`01` 的 `A.1` / `B.1` / `B.2` / `D.7`。
# 同值位置：无。

LAT_BATCH_GRID = (1, 2, 4, 10, 20, 30, 60)
# 来源：**[E] 本主题的算例** —— **在飞请求数上限**的网格（★ 每档都整除 `LAT_N_REQ`）。
# 口径：★ 含两端：`1`（逐条）与 `60`（= 全部请求一次在飞）⇒ `B.1`/`B.2` 的单调性**两端都有守卫**。
# 消费者：`01` 的 `A.1` / `B.1` / `B.2` / `B.3` / `D.7`。
# 同值位置：无。

LAT_ARRIVE_PERIOD = 12.0
# 来源：**[E] 本主题的算例** —— **持续到达**模型里相邻两次到达的间隔（固定间隔，不用随机 ⇒ 可复现）。
# 口径：★★ 它是 `B.4` 的第二个工作负载：**到达发生在服务过程中** ⇒ 排队时间真的出现（与一次到齐的 `B.1` 对照）。
# 消费者：`01` 的 `B.4` / `D.7`。
# 同值位置：无。


# ===========================================================================
# 5. `02-inference-parallelism-and-pd` 的模型参数 —— [E] 本主题的算例
# ===========================================================================

PP_N_STAGES = 4
# 来源：**[E] 本主题的算例** —— 流水线级数 `p`。
# 口径：★★ 训练侧与推理侧**用同一组 `(p, m)`**，才看得出「气泡率公式相同、绝对量与上界不同」。
# 消费者：`02` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：无。

PP_M_MICRO_GRID = (1, 2, 4, 8, 16)
# 来源：**[E] 本主题的算例** —— 微批个数 `m` 的网格。
# 口径：★ 含 `m = 1`（气泡最重）⇒ `A.1`/`B.1` 的守卫。
# 消费者：`02` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：无。

PP_FWD_COST = 3.0
# 来源：**[E] 本主题的算例** —— 一个微批走**一级**的**前向**耗时 `t_f`（推理侧**没有** `t_b`）。
# 口径：★★★ **推理侧没有反向** ⇒ 时间表上只有 `t_f` —— 这正是「与训练的实质差别」的载体。
# 消费者：`02` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：无。

PP_BWD_COST = 3.0
# 来源：**[E] 本主题的算例** —— 训练侧一级的**反向**耗时 `t_b`（★ **推理侧用不到**，只用来做「训练 vs 推理」的对照）。
# 口径：★★ 取 `t_b = t_f` 是**算例的对称选择**（★ 不是任何实测）—— 它让「同一组 `(p,m)` 下推理比训练省多少」有一个具体数。
# 消费者：`02` 的 `B.1` / `C.1` / `D.7`。
# 同值位置：无。

TP_KV_HEAD_GRID = ((8, 8), (8, 4), (4, 2), (2, 8), (8, 16))
# 来源：**[E] 本主题的算例** —— 每行 `(KV 头数 h_kv, TP 度 t)`。
# 口径：★★★ **含 `h_kv < t` 的两格**（`(2, 8)` / `(8, 16)`）—— 那时 KV **不能**按 `t` 线性缩小，只能**按 `h_kv` 分**（复制）。
#       这就是 GQA + TP 的边界：**KV cache 的切分粒度下限是 KV 头的个数**。
# 消费者：`02` 的 `A.2` / `B.2` / `D.7`。
# 同值位置：无。

PD_KV_BYTES_PER_TOKEN = 131072
# 来源：**[E] 本主题的算例**（★ 一个**给定的模型参数**，**不是**本库重推的 KV 账 —— 账的正文在 `KV_ACCOUNT_REF`）。
# 口径：★ 单位 = 字节 / token；`131072` = `128 KiB` —— 一个便于读数、且**远小于 `2^53`** 的值。
# 消费者：`02` 的 `A.3` / `B.3` / `D.7`。
# 同值位置：无。

PD_LINK_BYTES_PER_MS = 2621440
# 来源：**[E] 本主题的算例** —— P→D 那条链路的**可用带宽**（★ `2621440` B/ms = `2.62 GB/s`，一条**慢链路**档）。
# 口径：★★★ **这是【推导用的规格值】，不是本机 socket 的任何数字**（`P34`：本机 socket 的数字不得作依据）——
#       本机**单机、无真网络** ⇒ PD 的传输开销**只能用规格带宽推导**（残留 `T75`）。
# 消费者：`02` 的 `A.3` / `B.3` / `D.7`。
# 同值位置：无。

PD_PREFILL_MS_PER_TOKEN = 0.02
# 来源：**[E] 本主题的算例** —— **prefill 每 token 在目标硬件上占用的时间**（耦合系统里**阻塞** decode 的那部分）。
# 口径：★ 单位 = ms / token；`0.02` 与 `PD_LINK_BYTES_PER_MS` 一起决定 PD 分离的**临界在飞请求数**。
# 消费者：`02` 的 `A.3` / `B.3` / `D.7`。
# 同值位置：无。

PD_SPLIT_OVERHEAD_MS = 0.5
# 来源：**[E] 本主题的算例** —— **PD 分离的一次性开销**（建连 / 握手 / 元数据）。
# 口径：★★★ 它让「分离更优」的条件**依赖 prefill 长度 `P`**（否则两侧都 ∝ `P`、条件与 `P` 无关）——
#       这是一个**可被断言钉住**的结构事实（`A.3` 的临界点随 `P` 移动）。
# 消费者：`02` 的 `A.3` / `B.3` / `D.7`。
# 同值位置：无。

PD_PREFILL_LEN_GRID = (16, 64, 256, 1024)
# 来源：**[E] 本主题的算例** —— prefill 长度 `P` 的网格（跨 **64 倍**）。
# 口径：★ 跨两个数量级，才看得出「临界随 `P` 移动」与「大 `P` 时分离必胜」。
# 消费者：`02` 的 `A.3` / `B.3` / `D.7`。
# 同值位置：无。

PD_DECODE_USERS_GRID = (1, 2, 4, 8)
# 来源：**[E] 本主题的算例** —— 被阻塞的 decode 用户数 `n` 的网格。
# 口径：★ `n = 1` 时「分离」几乎总不如「耦合」（只有一个人被阻塞）⇒ 是一个**反例格**。
# 消费者：`02` 的 `A.3` / `B.3` / `D.7`。
# 同值位置：无。


# ===========================================================================
# 6. `03-scaling-and-observability` 的模型参数 —— [E] 本主题的算例
# ===========================================================================

ROUTE_N_REPLICA = 4
# 来源：**[E] 本主题的算例** —— 副本（replica）个数。
# 口径：★ 4 个副本足够让「把请求发给谁」这件事真的有选择余地（随机路由的命中率约 `1/4` 量级）。
# 消费者：`03` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：无。

ROUTE_N_REQ = 800
# 来源：**[E] 本主题的算例** —— 请求条数。
# 口径：★ 800 条足够让「路由命中率」的差异稳定到千分位。
# 消费者：`03` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：无。

ROUTE_N_TEMPLATE = 8
# 来源：**[E] 本主题的算例** —— **共享前缀模板**的个数（★ 每个模板有一批请求复用它的前缀）。
# 口径：★★ 它是「前缀共享」这个前提的载体：`N_TEMPLATE = 1` 时就退化成「所有请求共享一个前缀」（`C.3` 的反例）。
# 消费者：`03` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：无。

ROUTE_SEED = 20261009
# 来源：**[E] 本主题的算例**（伪随机种子，`numpy.random.default_rng`）。
# 口径：★ 钉死 ⇒ `A.1`/`A.2` 的结论**可复现**。
# 消费者：`03` 的 `A.1` / `B.1` / `D.7`。
# 同值位置：`120-inference/kv-cache` 的 `PREFIX_SEED` 与 `engines` 的 `RADIX_SEED`（**同一天，不同主题的不同流**）。

QUEUE_ARRIVAL_RATE = 0.9
# 来源：**[E] 本主题的算例** —— **到达率** `λ`（单位时间内的请求数）。
# 口径：★★ 配合 `QUEUE_SERVICE_RATE = 1.0` ⇒ `ρ = 0.9`（**< 1，稳定**，但排队显著）。
# 消费者：`03` 的 `A.2` / `B.2` / `D.7`。
# 同值位置：无。

QUEUE_SERVICE_RATE = 1.0
# 来源：**[E] 本主题的算例** —— **服务率** `μ`。
# 口径：★ `ρ = λ / μ`；★ `ρ ≥ 1` 时排队无界（`C.4` 的反例格）。
# 消费者：`03` 的 `A.2` / `B.2` / `D.7`。
# 同值位置：无。

QUEUE_N_SAMPLE = 120000
# 来源：**[E] 本主题的算例** —— 排队模拟的**样本数**（`120000` ⇒ 标准误约 `1 / √120000 ≈ 0.29%`）。
# 口径：★★ 样本数决定 `A.2` 的**判据容差**（本文取 `3%`）—— 容差不是随便定的，是标准误的量级。
# 消费者：`03` 的 `A.2` / `D.7`。
# 同值位置：无。
