"""120-inference/kv-cache 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 同值位置：` + `# 口径：` + `# 消费者：`。
★★ **本域的头号错误模式是「把 B 级当 A 级」**（真实引擎的一切都是 `[B]` 或以下）。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）；
  ★★ **字符串一律【单行】** —— 跨行隐式拼接会被 `ast` 折成一个 `Constant`、`end_lineno != lineno`
    ⇒ `mutate.py` 会**静默**把它计入「跳过」（`110-communication` 的现场，29 个叶子被跳过）。

★★★ **本主题唯一的载体口径（★ 这是本文件最要紧的一条）**：
  **`block` / `page` 是【同名不同指】** ——
    · **vLLM 的 KV 块 = 16 token**（源码常量 `DEFAULT_BLOCK_SIZE`，见 `VLLM_DEFAULT_BLOCK_SIZE`）；
    · **SGLang 的 page = 1 token**（LMSYS 博客逐字，见 `SGLANG_PAGE_TOKENS`）。
  ★★ 两者**差 16 倍**，**不可互相套用**；引用时必须**写明是哪一家**（`§2.2`）。

★ **本主题的禁区**（本主题**一行都不重写**，只回链 —— 这是 `P37` 的现场）：
  · ★★★ **KV cache 的【账】**（六项乘积 / `512 KiB` / `2 GiB` / GQA 的坑 / `AI = 2ST/(bpe(S+T))`）
    → `130-quantization/kv-cache-quantization/01`；
  · ★★★ **paged KV 的 kernel 侧【全段】**（页表延迟 / 碎片 `16/2 → 0.049%` / 布局 /
    **块大小的双向权衡及其结论「真正约束是查表次数，不是碎片」**）
    → `40-kernels/attention/05:130-179,283-290`；
  · **KV cache 量化** → `130-quantization/kv-cache-quantization/02,03`；
  · **vLLM 的模型接入与算子注册** → `45-kernel-lifecycle/requirement-discovery/01:158-214`。

★ **本主题的判据纪律**：
  本主题的 A 类钉在【**容量规划与块管理**】上（可用显存 → 块数 → 最大并发 · 块池 / 引用计数 / CoW），
  ★ **不碰**「页表 / 碎片 / 布局 / 块大小权衡」（那是 `40-kernels/05` 的 kernel 侧正文，`P37`）。

★ **本机的限定语**：本机**无 GPU、无 vLLM / SGLang / TensorRT-LLM**
  ⇒ ★★★ **本主题最强的一条限定语：**「**不是任何真实引擎，也不是真机**」。
"""


# ===========================================================================
# 1. vLLM 的 KV 块大小 —— [B] 产品官方文档 + 产品源码
#    载体 = GitHub `vllm/config/cache.py`（main 分支）+ 文档 `configuration/engine_args/`；
#    抓取日 **2026-10-09**。★ 行号会随维护漂移 ⇒ 锚点用「**文件路径 + 常量名**」，**不写行号**。
# ===========================================================================

VLLM_DEFAULT_BLOCK_SIZE = 16
# 来源：[B] vLLM 官方源码 —— `vllm/config/cache.py`（main 分支，抓取日 2026-10-09），
#       常量 `DEFAULT_BLOCK_SIZE`（`CacheConfig`）。逐字见 `VLLM_BLOCK_SIZE_SOURCE_QUOTE`。
#       环境口径：vLLM 发布版 0.31.0；文档站 = latest developer preview。
# 口径：★★ 这是 **vLLM 的实现默认值**，**单位 = token 数 / 块** ——
#       ★★ **不得写成「业界通常 16」**：SGLang 的对应实现是 `page = 1 token`（见下，差 16 倍）。
#       ★ 文档页**不给数字**（只写 `Accepts None (meaning "use default")`）⇒ 16 这个数**来自源码常量**。
#       ★ 另有独立的 `--mamba-block-size`（`mamba_block_size`），**不要与 `block_size` 混同**。
# 消费者：Chunk 1 的 `01-capacity-planning`（程序 `code/01_*.py` 尚未落盘）——
#         容量规划里「一个 KV 块装多少 token」这一因子。
# 同值位置：无（★ 本域唯一的 `16`；`40-kernels/attention/05` 的块大小讨论**回链，不重算**）。

VLLM_BLOCK_SIZE_SOURCE_QUOTE = "DEFAULT_BLOCK_SIZE: ClassVar[int] = 16"
# 来源：[B] vLLM 官方源码 —— `vllm/config/cache.py`（main 分支，抓取日 2026-10-09）。
#       ★ 完整性：`DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`（`grep -c` 于原始字节 → 1）。
# 口径：★★ 它是 `VLLM_DEFAULT_BLOCK_SIZE` 的**逐字依据**（源码，不是文档 —— 文档不给数字）。
# 消费者：Chunk 1 的 `01-capacity-planning`（`VLLM_DEFAULT_BLOCK_SIZE` 的来源佐证）。
# 同值位置：`VLLM_DEFAULT_BLOCK_SIZE`（同一条事实的「值」与「引文」两写）。

VLLM_BLOCK_SIZE_DOC_QUOTE = 'Size of a contiguous cache block in number of tokens. Accepts None (meaning "use default"). After construction, always int.'
# 来源：[B] vLLM 官方文档 —— `configuration/engine_args/` 的 `--block-size`（抓取日 2026-10-09）。
#       ★★ **内层引号 `"use default"` 是原文的一部分**（第一版转写曾漏掉它 —— `§5` v1.3 口径 10）。
#       ★ 完整性：该句式在原始 HTML 出现 **2 次**（另一处属 `--mamba-block-size`）
#         ⇒ **不能只凭该句式唯一确定 `--block-size`**（这正是「同值位置」要写清的地方）。
# 口径：★★ 它说明**文档页不给出 `16` 这个数**（`Accepts None (meaning "use default")`）——
#       `16` 来自源码常量（见 `VLLM_BLOCK_SIZE_SOURCE_QUOTE`）。★ 这是「规范与实现不分」的反面教材。
# 消费者：Chunk 1 的 `01-capacity-planning`（说明为何块大小取自源码而非文档）。
# 同值位置：`--mamba-block-size`（同一句式，**不同参数** —— 不得混同）。

SGLANG_PAGE_TOKENS = 1
# 来源：[B] LMSYS Org 官方博客 —— `https://lmsys.org/blog/2024-01-17-sglang/`（2024-01-17，抓取日 2026-10-09）。
#       逐字见 `SGLANG_PAGE_LAYOUT_QUOTE`。★ 注：早期猜测的 URL `.../2023-12-18-sglang/` 实测 **HTTP 404**。
# 口径：★★★ 这是 **SGLang** 的存储布局口径（`page = 1 token`），
#       ★★ **不是** vLLM 的 `block = 16`（`VLLM_DEFAULT_BLOCK_SIZE`）—— 两者**同名不同指**、**差 16 倍**。
#       ★ 引用时必须**写明是哪一家**；**不得互相套用**。
# 消费者：Chunk 1 的 `01-capacity-planning` / `03-prefix-and-eviction`（块粒度对照）。
# 同值位置：`VLLM_DEFAULT_BLOCK_SIZE`（**同名不同指**的那一条 —— 值不同）。

SGLANG_PAGE_LAYOUT_QUOTE = "These KV cache tensors are stored on the GPU in a paged layout, where the size of each page is equivalent to one token."
# 来源：[B] LMSYS Org 官方博客 —— `https://lmsys.org/blog/2024-01-17-sglang/`（2024-01-17，抓取日 2026-10-09）。
#       ★ 完整性：`grep -c 'RadixAttention'` 于原始 HTML → 10（同页）。
# 口径：★★ 它是 `SGLANG_PAGE_TOKENS` 的**逐字依据**（`the size of each page is equivalent to one token`）。
#       ★★ 「page」在这里是 **SGLang 的词**；vLLM 用「block」（`VLLM_DEFAULT_BLOCK_SIZE`）。
# 消费者：Chunk 1 的 `01-capacity-planning`（块粒度的外部依据）。
# 同值位置：`SGLANG_PAGE_TOKENS`（同一条事实的「引文」与「值」两写）。

PAGE_BLOCK_DISAMBIGUATION = "block / page 是同名不同指：vLLM 的 KV 块 = 16 token，SGLang 的 page = 1 token，两者差 16 倍，不可互相套用"
# 来源：**[E] 本库的消歧记录**（★ 不是任何来源里的话）—— 依据是 `VLLM_DEFAULT_BLOCK_SIZE`
#       与 `SGLANG_PAGE_TOKENS` 两条 [B] 级事实的**并列**。
# 口径：★★★ 它是一条**同名不同指**的记录：把 vLLM 的 `block` 与 SGLang 的 `page` 当成同一个粒度会**写错正文**
#       （`§2.2` 与 `§0.7 C` 事实 3/11 都点名这一条）。★ 它**没有外部出处** ⇒ 不得写成引文（本库纪律：
#       **自造真值**是一种错误模式）。
#       ★ 它的存在理由是**可复核**：引用块粒度时「有没有写明是哪一家」变成一条可被变异测试抓住的断言。
# 消费者：Chunk 1 的 `01-capacity-planning`（首次出现块粒度处点明）。
# 同值位置：无。


# ===========================================================================
# 2. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在本模块的正文常量里。
# ★★ **本元组【留空】**，理由与 `110-communication/collectives` 逐字相同：**每一项的字符串都会成为 `make mutate`
#   的叶子**，而它们**没有消费者** ⇒ 会全部记成**漏网**（本库实测：只用于打印的模块级常量表制造过漏网）。
#   ★★ **而【空元组本身】有消费者**：编号程序的 `preflight()` 用 `specs.UNSOURCED == ()` 钉住它。
# ★ **本主题的两处「未核到」写在注释里，不立常量**：
#   ① **vLLM 文档对 `continuous batching` 的机制定义** —— 未取到逐字定义（属 `batching-and-scheduling`，此处只记）；
#   ② **vLLM `--enable-prefix-caching` 的默认开关值** —— 取数到的文档页**未声明**默认开/关 ⇒ **不得臆断**
#      （属 `kv-cache/03` 与 `engines`）。
UNSOURCED = ()
