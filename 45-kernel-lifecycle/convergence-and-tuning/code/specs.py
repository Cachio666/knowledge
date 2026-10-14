"""45-kernel-lifecycle/convergence-and-tuning 的外部事实常量源。

本模块是 CONTRIBUTING §2 约定的**共享事实源**（`specs.py`）：三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ 本主题的来源强度是**全库最弱的一处**（域规格 Chunk 4 首句）：
   本主题的核心数字——噪声带、冷启动、收益门限——**大部分来自 D 级来源**
   （个人或小团队仓库）。因此本模块的常量命名**把来源编进去**
   （`HIPFIRE_*` / `TRITON_*` / `VLLM_*` / `ARXIV_*` / `MYTKOWICZ_*` / `CUDA_*`），
   使**误用可见**：看到一个以别人项目名开头的常量，就知道它只支撑「那个项目这么做」。

★ 来源等级（`CONTRIBUTING.md` §2 的 A–E 五级，本模块用到 A/B/C/D/E）：

  · **A 级：厂商规范**——`cudaEventElapsedTime` 的分辨率与语义（CUDA Runtime API / Best
    Practices Guide 两个官方页面**两次检索一致**）。
  · **B 级：某一实现的官方规格**——Triton 官方的 `do_bench` 默认值与其官方仓库 issue #2306；
    vLLM 官方的 JIT Kernel Warmup 贡献者文档与冷启动清单。**只能支撑「这一家这么做」**（P10）。
  · **C 级：论文/技术报告**——ASPLOS 2009 的测量偏差论文（**已确认同行评审**）、
    IEEE-HPEC 的 Robust benchmarking（**已确认同行评审**）、NIST TN 1826（官方技术报告）、
    以及 **arXiv:2607.02541（预印本，未确认同行评审，正文必须写明）**、
    Welch 检验的小样本模拟研究（同行评审期刊论文）。
    ★ 本模块**刻意收录它们的原因**：域 README 硬规则 2 说「D 级来源不得单独支撑一条论断，
    须有 A/B/C 级佐证」。**上面这几条就是那张 D 级表的 C/B 级佐证**——
    本主题的第一条论断（不 warmup 测出来的数字是错的）**最终由 B 级来源撑住**，
    不是靠 D 级的经验数字。
  · **D 级：社区材料**——hipfire（`warpfront/hipfire`）的 `docs/methodology/perf-benchmarking.md`。
    本模块保留它，因为它是**唯一给出「噪声带 / 门限」具体数值**的来源；
    但它**只支撑「这个项目这么做」**，正文必须写明它的测量条件
    （gfx1100 = RX 7900 XTX、新进程、受 DPM/thermal/firmware 影响）。
  · **E 级：本库推导/建模**——解析式、模型参数、以及 §8 里**被本库实跑推翻的规格条目**。

★ **本模块刻意不含散文。** 规则来自前两个主题的教训：把来源里的**描述性段落**搬进
  `specs.py`，程序**没法计算散文 ⇒ 只能 `print` ⇒ 必然漏网**。
  ⇒ **每个模块级常量必须是可被计算的值**（数、名、布尔、阈值、元组、字典）。
    描述性文字归**文档**；来源里的每个数字以**结构化字段**的形式落在这里。

★ **本模块的常量全部被三个程序断言钉住**，判据是机械的：
  `01_benchmark_hygiene.py` 用 `ast` 读另外两个程序的 `COVERED_*` 元组，断言
  **三者的并集 == 本模块的全部模块级常量**、且**两两不交**（见该程序 §D.8）。
  ⇒ 在这里加一个常量而没人断言它，`make run` 会立刻失败。

★ **§0.6 硬规则 5 的执行**：找不到来源的数字**只出现在 `UNSOURCED` 里**，
  **正文与常量名里都不出现**——不得「既保留数字又记 UNSOURCED」。
"""

# ===========================================================================
# 1. 噪声带与回归门限：hipfire（**D 级：个人/小团队仓库，单一来源**）
# ===========================================================================
#
# ★ 本节的每一条都只能支撑「**这个项目在 gfx1100 上这么做**」。
#   它的测量条件**必须随数字一起搬**（域 README 硬规则 3）：
#   GPU 是 gfx1100（RX 7900 XTX）、**新进程**、且带宽受 DPM 状态/热余量/固件版本影响。
#   ⇒ 正文**不得**写成「会话内噪声带通常是 ±10–15%」（那是 P10）。

HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100 = (10.0, 15.0)   # 来源等级：[D]
# 来源：**hipfire（D 级）** `docs/methodology/perf-benchmarking.md`，**两次独立检索一致**。
# 口径：「同一会话内的 A/B 噪声带」，**且是在新进程上测的**（见下一条）；
#       条件 = gfx1100（RX 7900 XTX）、DPM 状态 / 热余量 / 固件版本。
# ★ **这个带子比同一文档里的门限（±5%）还宽** —— 两档数字并列是这节的要害。

HIPFIRE_NOISE_BAND_IS_MEASURED_ON_FRESH_PROCESS = True   # 来源等级：[D]
# 来源：同上（D 级）。口径：该文档明写噪声带是「on a fresh process」的。
# ★ 它的用途是**否掉一个常见的错误补救**：「重开一个进程再测就好了」——
#   重开进程**本身就是**这条噪声带的测量条件，换进程不是它的解药。

HIPFIRE_SPEED_GATE_TOLERANCE_PCT = 5.0   # 来源等级：[D]
# 来源：**hipfire（D 级）**，两次独立检索读到同一句：Tolerance is ±5% from the committed baseline.
# 口径：相对**已提交基线**的容差；作为「防回归门限」的一个示例值。

HIPFIRE_SPEED_BASELINE_FILE_PATTERN = "tests/speed-baselines/<arch>.txt"   # 来源等级：[D]
# 来源：同上（D 级）。口径：**基线按架构分文件** —— 这是「跨架构基线各自维护」的
#       第一个独立实例（第二个实例是 §4 的 arXiv 预印本，两者独立 ⇒ 满足 P10 的检查）。

HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT = 8.0   # 来源等级：[D]
# 来源：同上（D 级）。口径：该文档明写「会话内的 +8% 是落在噪声里的」。
# ★ **这个数与它自己的门限 ±5% 并列时最好用**：一个超过门限的读数仍然可能是噪声。

HIPFIRE_GRAPH_AB_ENV_VAR = "HIPFIRE_GRAPH"   # 来源等级：[D]
# 来源：同上（D 级），**经检索转述**。口径：graph 开/关的一等对照实验用 `HIPFIRE_GRAPH=0/1`。
# ★ 本常量存在的**主要目的**是记录一处**规格订正**：规格原文写的是
#   `HIPFIRE_VERIFY_GRAPH`（见 §8 的 `SPEC_CLAIM_GRAPH_AB_ENV_VAR`），
#   两次检索都没有命中那个名字。

HIPFIRE_MAX_REL_SPREAD_BEFORE_UNSTABLE = 0.20   # 来源等级：[D]
# 来源：同上（D 级），**经检索转述**。口径：`(max − min) / median > 0.20` ⇒ 状态判为
#       `unstable`，**按「测量失败」处理，不当作一次胜/负**。
# ★ 这是一条**方向与「紧得可疑」相反的**机械判据：散得太开 ⇒ 无效。
#   两条合起来才是完整判据（见 02 篇 §3）。

HIPFIRE_BASELINE_UPDATE_MUST_BE_IN_SAME_COMMIT = True   # 来源等级：[D]
# 来源：同上（D 级）。口径：以一处架构的回归换别处的收益时，**基线更新必须与改动同一次提交**，
#       让评审看到这笔交易；且不得用 `--no-verify` 绕过门限。

HIPFIRE_BASELINE_ROW_PROVENANCE_FIELDS = ("binary_md5", "git_dirty_state", "diff_md5")   # 来源等级：[D]
# 来源：同上（D 级）。口径：**行级来源信息**——脏工作区与已提交基线不可直接比。
# ★ 三个字段**逐个元素**进变异测试（元组内部元素也要被断言，见 01 篇 §D.9）。


# ===========================================================================
# 2. warmup 的官方默认值：Triton（**B 级：某一实现的官方规格**）
# ===========================================================================
#
# ★ 这是本主题**证据最强的一组数字**，也是 01 篇那条核心论断的落点。
#   规格原文（§0.7.G）把「冷启动」的证据挂在 D 级来源上；
#   实施时检索到 **B 级来源直接把同一件事量化了**（官方文档的默认值 + 官方仓库的 issue），
#   所以 01 篇的正文以这一组为主、D 级为辅。

TRITON_DO_BENCH_DEFAULT_WARMUP_MS = 25.0   # 来源等级：[B]
# 来源：**Triton 官方文档** `triton.testing.do_bench`（B 级），**经检索转述**。
# 口径：★★ **单位是毫秒，不是次数**——`warmup=25` 是「先跑 25 ms」而不是「先跑 25 次」，
#       默认值是**时间去**估计成迭代次数的（`n_warmup = max(1, int(warmup / estimate_ms))`）。
#       ⇒ 「跳过前 N 次」这个说法在真实工具里是**由时间换算出来的**。

TRITON_DO_BENCH_DEFAULT_REP_MS = 100.0   # 来源等级：[B]
# 来源：同上（B 级）。口径：计时窗口同样是**时间**（100 ms），不是固定次数。

TRITON_DO_BENCH_WARMUP_UNIT_IS_MILLISECONDS = True   # 来源等级：[B]
# 来源：同上（B 级）。★ 单列这一条是因为**把 25 当成次数会给出完全不同的量**。

TRITON_DO_BENCH_DEFAULT_RETURN_MODE = "mean"   # 来源等级：[B]
# 来源：同上（B 级）。口径：官方默认返回**均值**。
# ★★ 与本主题 01 篇的结论**正面相撞**：本库实算（§7 的模型）在 `k/n = 0.10` 时
#    **均值偏差 +0.400、中位数偏差 +0.007**（稳态尺度 1.0、σ = 0.05）——
#    官方默认返回的正是**对冷启动最敏感的那个统计量**。这不是说 Triton 错了，
#    而是说：**默认值的前提是 warmup 真的跑够了**（它的 warmup 单位是毫秒，见上）。

TRITON_DO_BENCH_AUTOTUNE_QUANTILES = (0.5, 0.2, 0.8)   # 来源等级：[B]
# 来源：同上（B 级）。口径：autotune 路径改用**分位数**（中位 + 20%/80%）而不是均值。
# ★ 两个统计量在同一个工具里并存，用途不同——引用时必须说清是哪一个（P9）。

TRITON_DO_BENCH_CLEARS_L2_BEFORE_EACH_ITER = True   # 来源等级：[B]
# 来源：同上（B 级）。口径：每次计时前清 L2 —— 这是「固定测量条件」的一个官方实现。

TRITON_DO_BENCH_IS_OFFICIAL_API = True   # 来源等级：[B]
# 来源：同上（B 级）。★ 单列它是为了让「这是谁的做法」这句话在常量层面可见（P10 的机械检查）。

TRITON_ISSUE_2306_TITLE_UNDERESTIMATE_PCT = 30.0   # 来源等级：[B]
# 来源：**Triton 官方仓库 issue #2306**（B 级），**经检索转述**：
#       "do_bench underestimates the performance by 30% due to too short warmup time"。
# 口径：**默认 warmup（25 ms）不够长**导致的低估量级；`triton.jit` 且**不用 autotune 装饰器**
#       的 kernel 最容易撞上（autotune 会多跑几次 do_bench，等于加长了 warmup）。

TRITON_ISSUE_2306_TIME_WITH_DEFAULT_WARMUP_MS = 11.23   # 来源等级：[B]
# 来源：同上（B 级）。口径：同一个 kernel 在初始估计下测到的量级（单位 ms，是**单次调用**的量级）。

TRITON_ISSUE_2306_TIME_WITH_EXTENDED_WARMUP_MS = 8.79   # 来源等级：[B]
# 来源：同上（B 级）。口径：把 warmup 拉长到 **> 150 ms** 之后的同一个 kernel。

TRITON_ISSUE_2306_REQUIRED_WARMUP_OVER_MS = 150.0   # 来源等级：[B]
# 来源：同上（B 级）。口径：issue 里使读数稳定的 warmup 门槛。

TRITON_ISSUE_2306_IMPLIED_FAKE_GAIN_PCT = (11.23 / 8.79 - 1.0) * 100.0   # 来源等级：[E]
# 来源：**本库推导 [E]**，输入是上面三条 B 级常量（口径相同：同一 kernel、同一台机器）。
# ★★ 用途：**一个纯粹由 warmup 造成的「+27.8% 加速」**。这是 01 篇 C 类断言的现实锚点——
#    本库的模型放大它（冷启动 5× ⇒ 人为收益 200%+），但**符号与机制是真的**。

TRITON_ISSUE_2306_AFFECTS_JIT_WITHOUT_AUTOTUNE = True   # 来源等级：[B]
# 来源：同上（B 级）。口径：受害对象是**不用 autotune 装饰器**的 `triton.jit` kernel。

TRITON_ISSUE_2306_IS_AN_OFFICIAL_REPO_ISSUE = True   # 来源等级：[B]
# 来源：同上（B 级）。★ 单列它是为了标注**证据形态**：它是一个**官方仓库里尚未关闭的
#   issue**，不是一篇经过评审的结论——正文引用时必须写成「官方仓库的 issue 里报告了」。


# ===========================================================================
# 3. 冷启动的时间尺度：vLLM（**B 级：某一实现的官方文档**）
# ===========================================================================
#
# ★ 规格 §0.7.G 把「JIT 冷缓存首次编译 30 s–2 min；模型重载冷启动 2–5 s」标成 **D 级**，
#   并注明「Chunk 1 未复核，留给本主题去核，核不到就删」。
#   实施时**核不到那两个数**（见 §8 的两条 SPEC_CLAIM_*），但**核到了更细的 B 级数据**：
#   vLLM 官方把冷启动拆成了**有条件的**分量。本节就是这个替换的结果。
#   ★ 处置：规格那两个数**进 UNSOURCED、正文不用**；正文用本节这一组。

VLLM_JIT_FIRST_REQUEST_SPIKE_S = (2.0, 3.0)   # 来源等级：[B]
# 来源：**vLLM 官方文档** `contributing/jit_kernel_warmup`（B 级），**经检索转述**。
# 口径：**没有 warmup 时，第一个真实用户请求**为 JIT 编译多付 2–3 s。
# ★ 注意它**不是**「模型重载时间」，是「首个请求的编译尖峰」——两者常被混为一谈（P9）。

VLLM_GRAPH_CAPTURE_PER_BATCH_SIZE_DELAY_S = (1.0, 2.0)   # 来源等级：[B]
# 来源：同上（B 级）。口径：**每个 batch size 的首次**请求触发 CUDA graph capture，约 1–2 s。
# ★ 这与 01 篇的机制同源：**代价按「新形状」付**，而不是只付一次。

VLLM_FLASHINFER_FIRST_USE_NO_CACHE_MIN = 10.0   # 来源等级：[B]
# 来源：同上（B 级）。口径：未装 `flashinfer-jit-cache` 时，首次使用按「**最长约 10 分钟且没有日志**」
#       计——**表现为疑似挂死**。★ 这条对「怎么看冷启动」很有用：**没有输出不等于没有在干活**。

VLLM_FLASHINFER_LOGGED_FIRST_USE_S = 312.47   # 来源等级：[B]
# 来源：**vLLM PR #54529**（B 级，官方仓库）新增的日志行，**经检索转述**：
#       "finished first use in 312.47 seconds"。口径：一次真实的首次编译耗时（约 5.2 min）。

VLLM_DEEPGEMM_KERNEL_COUNTS = {"glm5_744b": 2259, "deepseek_v3_2_685b": 1827}   # 来源等级：[B]
# 来源：同上（B 级）。口径：**首次启动需要编译的 kernel 数**（FP8 GEMM / CUTLASS 后端）。
# ★ 这是「冷启动的代价可以有多大」的**计数形态**：不是倍数，是**几千个编译单元**。

VLLM_DEEPGEMM_FIRST_LAUNCH_MIN = 5.0   # 来源等级：[B]
# 来源：同上（B 级）。口径：上述几千个 kernel 的首次编译量级（约 5 min），之后走缓存。

VLLM_COLD_START_TOTAL_CACHED_S = 182.0   # 来源等级：[B]
VLLM_COLD_START_TOTAL_FRESH_MIN = (10.0, 15.0)   # 来源等级：[B]
# 来源：同上（B 级）。口径：一个具体模型（H200 / 31B dense）的**端到端**冷启动总时长：
#       权重已缓存时约 182 s，权重需下载时 10–15 min。**其中含权重加载，不只是编译。**
#       ★ 排除项：这一条是**端到端**口径，与上面各条「单次编译/单请求」口径**不可比**（P9）。

VLLM_COLD_START_LISTED_COMPONENTS_S = {   # 来源等级：[B]
    "container_init": 30.0,
    "weights_load_cached": 32.0,
    "torch_compile_cached": 8.0,
    "graph_capture": 14.0,
    "engine_init_rest": 10.0,
    "warmup_query": 7.0,
}
VLLM_COLD_START_COMPONENT_SUM_S = 101.0   # 来源等级：[B]
VLLM_COLD_START_BREAKDOWN_IS_INCOMPLETE_IN_EXCERPT = True   # 来源等级：[B]
# 来源：同上（B 级），**经检索转述**；后两条是**本库 [E] 的一致性检查**。
# ★★ 检查结果：列出的分量**加起来只有 101 s，而总时长写的是 182 s** ⇒ 检索到的清单
#    **是不完整的**（有未列出的项）。⇒ **处置：正文只引用总时长与逐个分量，禁止把「182 − 101 = 81 s」
#    当成「其余」来用**（残差不是测量值）。这是本主题在数字纪律上的一个具体动作。

VLLM_TORCH_COMPILE_COLD_S_MIN = 60.0   # 来源等级：[B]
# 来源：同上（B 级）。口径：`torch.compile` 冷缓存（权重已缓存但编译缓存未命中）时的量级（约 60 s+），
#       对照缓存命中时的 8 s。★ **同一个阶段的两档相差约 7.5 倍**，取决于缓存是否命中。

VLLM_ENFORCE_EAGER_SAVES_STARTUP_S = (20.0, 40.0)   # 来源等级：[B]
# 来源：同上（B 级）。口径：`--enforce-eager`（跳过 CUDA graph capture）省下的启动时间，代价是每 token 吞吐。

VLLM_SKIP_WARMUP_AFFECTS_ONLY_FIRST_REQUEST_LATENCY = True   # 来源等级：[B]
# 来源：同上（B 级）。口径：官方明写 `VLLM_SKIP_WARMUP` **不影响正确性**，只影响最早请求的延迟。
# ★ 这一条把「冷启动」与「正确性门」分开了：它是 `implementation` 主题之外的代价，不是缺陷。

VLLM_WARMUP_MUST_NOT_ALLOCATE_REAL_TENSORS = True   # 来源等级：[B]
# 来源：同上（B 级）。口径：预热契约要求**用假张量或后端 compile-only 描述符**完成物化，
#       **不得**启动真实推理或分配真实张量。★ 「预热」不等于「跑一遍」。

VLLM_JIT_MONITOR_WARNS_ON_INFERENCE_TIME_COMPILE = True   # 来源等级：[B]
# 来源：同上（B 级）。口径：`jit_monitor.py` 在推理期发生 JIT 编译时打警告，并建议
#       「考虑扩大 warmup 覆盖这个 shape/config」。★ 这说明**预热覆盖不了所有形状**，
#       它是一个**下界**，不是「一次预热终身免疫」。


# ===========================================================================
# 4. 跨架构不可迁移（**C 级：arXiv 预印本，未确认同行评审**）
# ===========================================================================
#
# ★ 这是 03 篇的核心论断的证据：**同一个改动在不同架构上可以反向**。
#   ★★ 措辞的硬约束（规格 Chunk 4 末段）：只能说「**已有报告显示**可以反向」，
#      **不得**说「同一改动在不同架构上一定反向」。

ARXIV_2607_02541_IS_PREPRINT_NOT_PEER_REVIEWED = True   # 来源等级：[C]
ARXIV_2607_02541_TITLE = "Static PTX Metrics Track Structural Kernel Regressions but Miss Semantic Ones"   # 来源等级：[C]
# 来源：**arXiv:2607.02541**（C 级），**两次独立检索一致**且命中 PDF 页面级出处。
# ★ 本模块**单列「是预印本」这个布尔量**，是为了让它在正文里**不可能被漏标**。

ARXIV_2607_02541_GPUS = ("RTX 3060", "A10", "L40S", "A100 SXM4", "H100 NVL")   # 来源等级：[C]
# 来源：同上（C 级）。口径：五类 GPU（sm_86 / sm_86 / sm_89 / sm_80 / sm_90）。

ARXIV_2607_02541_SAME_CHANGE_L40S_PCT = 9.4   # 来源等级：[C]
ARXIV_2607_02541_SAME_CHANGE_A100_SXM4_PCT = -50.0   # 来源等级：[C]
# 来源：同上（C 级），Table 4（**两次独立检索一致**）。口径：**同一个 (correct, buggy) 结构改动**
#       的实测 Δperf%：L40S 上 +9.4%、A100 SXM4 上 −50%。
# ★★ 两个数**符号相反、量级相差 5 倍以上**——这是「加速比不可跨硬件搬运」的最强单例。

ARXIV_2607_02541_ARCHS_WITHIN_PCT_OF_ZERO = 4   # 来源等级：[C]
ARXIV_2607_02541_WITHIN_BAND_PCT = 20.0   # 来源等级：[C]
# 来源：同上（C 级），Table 4。口径：五类中**四类**落在零的 ±20% 以内，**A100 SXM4 被共享宿主机的
#       启动方差支配**。★ 规格 §0.7.G 已订正过这条：**用原文的 ±20%，不要用来源不明的「±5%」**。

ARXIV_2607_02541_STRUCTURAL_BUG_DELTA_PCT_WITHIN_NOISE = 3.2   # 来源等级：[C]
# 来源：同上（C 级）。口径：`gelu_triton_buggy`（丢掉一个 0.5 因子，少 8 条指令、8 个寄存器）
#       在 RTX 3060 上的实测 Δperf% = **+3.2%**，**落在噪声带内**。
# ★★ 这是「**真**坏了一个功能，性能读数却反向变好」的现成实例——比本库的仿真更硬。

ARXIV_2607_02541_STATIC_METRICS_ARE_ARCH_INDEPENDENT = True   # 来源等级：[C]
# 来源：同上（C 级）。口径：同一对的 Δregs / Δinstrs **在五类 GPU 上完全相同**（编译期决定）。

ARXIV_2607_02541_PERF_GATE_NEEDS_PER_ARCH_CALIBRATION = True   # 来源等级：[C]
# 来源：同上（C 级）。口径：原文结论——**基于性能的门限必须在每个架构上重新标定**，
#       否则只能当冒烟检查。★ 与 §1 的 `HIPFIRE_SPEED_BASELINE_FILE_PATTERN` 是**两个独立主体**
#       的同一结论 ⇒ 满足 P10 的机械检查（≥2 个独立主体）。

ARXIV_2607_02541_NOISE_IS_SUB_MILLISECOND_SCALE = True   # 来源等级：[C]
# 来源：同上（C 级）。口径：噪声支配现象的**前提是 kernel 处在亚毫秒尺度**。
# ★ 这一条是**判据的作用域**（P7）：换成长时间的 kernel，本条不成立。


# ===========================================================================
# 5. 测量偏差与小样本统计（**C 级：已确认同行评审** + 官方技术报告）
# ===========================================================================
#
# ★ 本节是域规格 §0.7.G「优先寻找 A/B/C 级佐证」的**执行结果**：
#   规格把「紧得可疑的标准差反而可疑」标成 D 级，实施时**没有在 hipfire 里找到那句话**，
#   但找到了**两篇同行评审/官方技术报告**支撑同一机制（计时器分辨率把方差压平）。

MYTKOWICZ_ENV_SIZE_PERF_CHANGE_PCT = 33.0   # 来源等级：[C]
MYTKOWICZ_ENV_SIZE_MAX_PERF_CHANGE_PCT = 300.0   # 来源等级：[C]
# 来源：**ASPLOS 2009**（C 级，**已确认同行评审**）"Producing Wrong Data Without Doing Anything
#       Obviously Wrong!"，**经检索转述**。口径：只改一个**未被使用的环境变量的大小**
#       （改变栈起始地址 ⇒ 改变局部变量对齐），性能**经常变化约 33%，最多近 300%**。
# ★ 规格注：检索**不支持**「40%」这个数，本模块不收录它。

MYTKOWICZ_LINK_ORDER_LUCKY_O3_ADVANTAGE_PCT = 8.0   # 来源等级：[C]
MYTKOWICZ_LINK_ORDER_UNLUCKY_O3_DEFICIT_PCT = 7.0   # 来源等级：[C]
# 来源：同上（C 级）。口径：只改**链接参数顺序**，「幸运」顺序让 -O3 看起来比 -O2 快 8%，
#       「不幸」顺序让它看起来慢 7% ⇒ **结论翻转**。
# ★★ 这是「同一改动可测出相反符号」的**同行评审实例**（03 篇用），且**不需要换硬件**。

MYTKOWICZ_SURVEYED_PAPERS = 133   # 来源等级：[C]
MYTKOWICZ_PAPERS_THAT_ADDRESSED_MEASUREMENT_BIAS = 0   # 来源等级：[C]
# 来源：同上（C 级）。口径：作者统计了 ASPLOS/PACT/PLDI/CGO 的 **133 篇**近期论文，
#       **没有一篇**充分考虑了测量偏差。★ 这是「这件事被系统性忽略」的量化证据。

MYTKOWICZ_TECHNIQUES = ("causal_analysis", "setup_randomization")   # 来源等级：[C]
# 来源：同上（C 级）。口径：两个对策——**因果分析**（检出偏差）与**设置随机化**（规避它）。

HPEC_ROBUST_BENCH_TITLE = "Robust benchmarking in noisy environments"   # 来源等级：[C]
HPEC_ROBUST_BENCH_IS_PEER_REVIEWED = True   # 来源等级：[C]
# 来源：**IEEE-HPEC / arXiv:1608.04295**（C 级，**已确认同行评审**），**经检索转述**。

HPEC_TIMER_ACCURACY_TERM_EXISTS = True   # 来源等级：[C]
HPEC_OPTIMAL_REPS_BOUNDED_BY_ACCURACY_OVER_PRECISION = True   # 来源等级：[C]
# 来源：同上（C 级）。口径：该文把每次计时建模为 `T_i = tP0 + E_i`，误差项里含一个**计时器准确度**
#       分量；要点是**最优重复次数受「准确度 / 精度」之比约束**——
#       ⇒ **重复次数不是想加就加的，它有一个由计时器决定的界**。

NIST_TN1826_NEEDS_CONFIDENCE_INTERVAL = True   # 来源等级：[C]
NIST_TN1826_FREQUENCY_SCALING_BREAKS_IID = True   # 来源等级：[C]
NIST_TN1826_SMALL_SAMPLES_INVALIDATED_BY_OUTLIERS = True   # 来源等级：[C]
# 来源：**NIST TN 1826**（C 级：官方技术报告），**经检索转述**。
# 口径三条：① 重复到「能给出可接受的置信区间」为止；② Turbo Boost / SpeedStep 之类的
# **动态频率特性会破坏「观测独立同分布」这个前提**；③ 小样本容易被离群点毁掉。
# ★ 第②条与本主题直接相关：**时钟爬升**既是 01 篇的现象，也是**统计前提的破坏者**。

WELCH_SATTERTHWAITE_DF_IS_CRUCIAL = True   # 来源等级：[C]
WELCH_HAS_NO_ANALYTIC_ERROR_BOUND = True   # 来源等级：[C]
WELCH_LIBERAL_VARIANCE_RATIO_GT = 2.0   # 来源等级：[C]
WELCH_LIBERAL_SAMPLE_SIZE_BOUND = 30   # 来源等级：[C]
WELCH_MAX_TYPE_I_INFLATION_PCT = 58.0   # 来源等级：[C]
# 来源：**Welch 检验小样本行为的已发表模拟研究**（C 级，同行评审期刊），**经检索转述**。
# 口径：① 用合并自由度配 Welch 统计量、或用 Welch 自由度配合并统计量，**都不稳健**
#       ⇒ **Satterthwaite 自由度是必需的**；② Welch 的近似**没有解析误差界**，只能靠模拟评估；
#       ③ n ≤ 20–30 且两组方差比 > 2:1 时 Welch **变自由（liberal）**；
#       ④ 模拟里观察到的最**大**第一类错误膨胀 **+58%**。
# ★★ **本节第 ③④ 条与本库实算的关系**（口径必须写清）：本库实算的是**等 n、等方差**的情形，
#     那种情形下 Welch 统计量与合并统计量**代数恒等**，所以本节这些数**不用于**解释本库的实算值；
#     它们支撑的是「Welch 近似在小样本下没有保证」这句话本身。

MLPERF_ALIGNED_WARMUP_RUNS = 5   # 来源等级：[D]
MLPERF_ALIGNED_MEASUREMENT_RUNS = 10   # 来源等级：[D]
MLPERF_ALIGNED_PRIMARY_METRIC = "median"   # 来源等级：[D]
MLPERF_OFFICIAL_RULEBOOK_NOT_RETRIEVED = True   # 来源等级：[D]
# 来源：**MLPerf 对齐的方法学描述**（D 级：二手转述），**经检索转述**。
# ★★ **作用域必须写死**（P7）：这几条**不是** MLCommons 的规则原文——检索**没有**取到官方
#    rulebook，所以本模块**不写「MLPerf 规定……」**，只写「一套 MLPerf 对齐的方法学用……」。
#   最后一条布尔量把这件事**变成可断言的常量**，避免正文滑回「MLPerf 规定」。

BENCHMARK_MIN_RUNS_RECOMMENDED = 10   # 来源等级：[D]
BENCHMARK_MIN_RUNS_HIGH_VARIANCE = 30   # 来源等级：[D]
# 来源：同上（D 级）。口径：「至少 10 次测量；高方差算子加到 30+」。
# ★ 这是**练习建议**，不是推导结论——本库自己的实算（§7）给出的是**公式**，不是固定次数。

WARMUP_DISCARD_IS_REQUIRED_BY_ALIGNED_METHODOLOGY = True   # 来源等级：[D]
# 来源：同上（D 级）。口径：MLPerf 对齐的方法学里，warmup 阶段的结果**一律丢弃**。
# ★ 这与 01 篇的核心论断同向：**前几次测量不是「差一点」，是不该进统计**。


# ===========================================================================
# 6. 计时器的分辨率（**A 级：厂商规范**）
# ===========================================================================
#
# ★ 本节是「紧得可疑的标准差反而可疑」的**机械来源**：分辨率先天会把方差压平。
#   两个官方页面（CUDA Runtime API 与 CUDA C++ Best Practices Guide）**两次检索一致**。

CUDA_EVENT_TIMER_RESOLUTION_US = 0.5   # 来源等级：[A]
# 来源：**NVIDIA 官方**（A 级）：`cudaEventElapsedTime` 的文档原文
#       "in milliseconds with a resolution of around 0.5 microseconds"，
#       Best Practices Guide 亦写 "resolution of approximately half a microsecond"。
# 口径：分辨率约 **0.5 µs**；返回值单位是**毫秒**（两个单位不要混）。

CUDA_EVENT_TIME_RETURN_UNIT = "milliseconds"   # 来源等级：[A]
# 来源：同上（A 级）。★ 单列它是因为「0.5」常被当成秒/毫秒误用。

CUDA_EVENT_RECORD_IS_ASYNC_MAY_OVERCOUNT = True   # 来源等级：[A]
# 来源：同上（A 级）。口径：若两个 event 落在非 NULL 流上，`cudaEventElapsedTime` 的结果
#       **可能比预期大**（record 是异步的，中间可能插入了别的流操作）。
# ★ 这是一条**方向固定**的偏差：**只会偏大**。

CUDA_EVENT_NEEDS_ENOUGH_WORK_TO_BRACKET = True   # 来源等级：[A]
# 来源：同上（A 级）。口径：两个 event 之间要夹**足够的工作量**，否则测的是开销不是 kernel。
# ★ 与 `KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL`（§7）连读：亚毫秒尺度上这两件事是同一个问题。


# ===========================================================================
# 7. 本库推导与建模常量（**E 级**）
# ===========================================================================
#
# ★★ 本节的常量是**程序的可调参数**（域规格 Chunk 4「每篇必须做到」的第 2 条硬要求）：
#   它们**不得被硬编码进逻辑**——证据强度不足以支撑硬编码。
#   三个程序的做法是：把这些值作为**函数参数**传入，默认值只在本模块里出现一次。

STEADY_TIME_NOMINAL = 1.0   # 来源等级：[E]
STEADY_SIGMA_DEFAULT = 0.05   # 来源等级：[E]
COLD_START_SLOWDOWN_DEFAULT = 5.0   # 来源等级：[E]
N_MEASUREMENTS_DEFAULT = 100   # 来源等级：[E]
MC_REPLICATES_DEFAULT = 3000   # 来源等级：[E]
# 来源：**本库建模 [E]**。口径：稳态时间取无量纲 1.0、稳态 σ = 0.05、冷启动值 = 稳态 × 5、
#       一次会话取 n = 100 次测量、统计量用 3000 次重复取均值。
# ★ `STEADY_SIGMA_DEFAULT = 0.05` 是**模型参数**，不是「业界噪声带」——它与 §1 的 D 级 D 级
#   数字**没有换算关系**（此处刻意不做换算，见 `UNSOURCED` 的 `noise_band_vs_sigma_mapping`）。

K_OVER_N_GRID = (0.0, 0.10, 0.40, 0.49, 0.50, 0.90)   # 来源等级：[E]
# 来源：规格 Chunk 4 的预跑表用它；**本库 [E]**。★ 四点 0.10/0.49/0.50/0.90 是**判据的关键点**。

ORDER_STATISTIC_QUANTILE_RULE = "blom"   # 来源等级：[E]
# 来源：**本库 [E]**。口径：解析式用 `E[z_(i:m)] ≈ Φ⁻¹((i − 0.375) / (m + 0.25))`（Blom 近似）。
# ★ 单列这一条是**方法可见性**：解析值依赖这个近似，换近似会改变解析值（程序里作 A 类对拍）。

Z_ALPHA_OVER_2_95 = 1.9599639845400534   # 来源等级：[E]
Z_BETA_AT_POWER_80 = 0.8416212335729144   # 来源等级：[E]
MDE_COEFFICIENT_AT_POWER_80 = 2.801585218112968   # 来源等级：[E]
# 来源：**本库 [E] 推导**（标准正态分位数，由 `statistics.NormalDist().inv_cdf` 算出，
#       程序里再独立算一遍做 A 类对拍）。
# 口径：`z_{0.975} + z_{0.80} = 1.9599639845400534 + 0.8416212335729144 = 2.801585218112968`。

MDE_FORMULA_DENOMINATOR_IS_SQRT_2_OVER_N = True   # 来源等级：[E]
# 来源：规格订正（见 §8 的 `SPEC_CLAIM_MDE_USED_SQRT_1_OVER_N`）[E]。
# ★★ 两组**独立**样本之差的标准误是 `σ·√(2/n)`，**不是** `σ·√(1/n)`——
#    规格初稿漏了 `√2`，解析值系统性偏小 `√2 ≈ 1.414` 倍。

MDE_REDUCED_COEFFICIENT_EXACT = 2.801585218112968   # 来源等级：[E]
MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED = 4.0   # 来源等级：[E]
MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT = (4.0 / 2.801585218112968 - 1.0) * 100.0   # 来源等级：[E]
# 来源：**本库 [E]**。规格 Chunk 4 把最小可检出差异的约简形式写成 `≈ 4σ/√n`；
#       精确系数是 `2.80`。⇒ 那个「≈」是一个 **+42.8% 的保守放大**。
# ★★ **它不是无害的**：03 篇的程序量到，用 4.0 与用 2.80 会给出**不同的停止步数**（4 vs 5）。

STOP_RULE_USES_MDE_AS_THRESHOLD = True   # 来源等级：[E]
# 来源：**本库 [E]**（规格 Chunk 4 §Chunk4 文档 03 的 §1）。口径：停止判据 = 「下一步的预期增益
#       低于最小可检出差异 ⇒ 不可判定 ⇒ 停手」。★ 它是一个**判据**，不是一个感觉。

GAIN_FIRST_STEP_PCT = 30.0   # 来源等级：[E]
GAIN_DECAY_RATIO_DEFAULT = 0.6   # 来源等级：[E]
SIGMA_PCT_DEFAULT = 3.0   # 来源等级：[E]
N_PER_GROUP_DEFAULT = 5   # 来源等级：[E]
# 来源：**本库 [E] 建模**。口径：收益递减用几何序列 `g_i = 30% × 0.6^i`；每组 n = 5、σ = 3%。

FP_RATE_PCT_AT_T8_CORRECT_DF = 2.50   # 来源等级：[E]
FP_RATE_PCT_AT_T4_WRONG_DF = 3.28   # 来源等级：[E]
# 来源：**本库推导 [E]**，且**与 02 篇同一口径**（在那里是实测 + 解析值 `P(t₈ > 临界值)`）：
#       n = 5 每组、单侧检验、真实差异为 0 时，用 t(8)=2.306 ⇒ **2.50%**（名义值），
#       用 t(4)=2.132 ⇒ **3.28%**（多出 1.31 倍）。
# ★★ 用途：03 篇的「假胜利累积」表直接用这两个**解析率**相乘，而不是用带蒙特卡洛噪声的
#   估计值（否则 0.35 与 0.327 之间的差会被误读成模型差异）。
FALSE_VICTORY_STEP_COUNTS = (10, 50, 200)   # 来源等级：[E]
SINGLE_MEASUREMENT_WIN_PROB_PCT = 50.0   # 来源等级：[E]
# 来源：**本库 [E] 推导**。口径：真实差异为 0 时，**一次 A/B 各测一次**就宣判胜负，
#       假阳性率是 **50%**（对称分布下 P(B > A) = 1/2）——不是 5%，也不是 1%。
# ★★ 这个数是 02 篇 C 类断言的靶心：**「跑一次看快了就是快了」= 抛硬币**。

LOWER_IS_BETTER_METRIC_CONVENTION = "lower_is_better"   # 来源等级：[E]
# 来源：**本库 [E] 约定**。口径：本主题全部模型里，「时间/延迟」是**越小越好**；
#       所以「b < a」= b 更快。★ 单列它是为了**口径可见**（P9 的高发处：加速比 vs 时间比）。

QUANT_MUTATION_FIRST_ROUND_CAPTURED = 66   # 来源等级：[E]
QUANT_MUTATION_FIRST_ROUND_ESCAPED = 29   # 来源等级：[E]
QUANT_MUTATION_FINAL_CAPTURED = 95   # 来源等级：[E]
QUANT_FIRST_100PCT_WAS_FAKE = True   # 来源等级：[E]
META_CLUE_SOURCE_TIERS = ("E", "C")   # 来源等级：[E]
# 来源：**本库既有记录 [E]**，出处可核查：
#       `docs/superpowers/plans/2026-09-18-quantization-domain.md` 第 1791 与 1799–1801 行。
# 口径：`130-model-optimization` 变异测试的真实轨迹——**第一轮跑出「95/95 全捕获」，全是假的**
#       （`verify.py` 里有个 `KeyError`，崩溃被算成「捕获」，于是每一次变异都「被捕获」）；
#       真实轨迹是 **66 捕获 / 29 漏网**（66 + 29 = 95）→ 逐条补强 → **95/95**。
# ★★ 用途：`02` 篇 §3 的**元线索**——它与外部的「计时器分辨率把方差压平」
#    （`META_CLUE_SOURCE_TIERS` 里的 `"C"`）是**两条独立来源**，指向同一件事：
#    **过于干净的结果通常是采集方式的问题，不是对象本身的性质。**
#    ★ 这是本域**唯一**一处「本库自身经验与外部经验互相印证」的地方（规格已点明）。


# ===========================================================================
# 8. 规格预跑值 与 **被本库实跑推翻的规格条目**（**E 级**）
# ===========================================================================
#
# ★★ 本节是本主题对域规格 §0.1「每个数字默认是错的」的**机器化回应**：
#   规格给出的**预跑值**原样抄进来（供三个程序复算），**已被推翻的条目**单列成
#   `SPEC_CLAIM_*` 常量，使「本库在哪里、为什么不同意规格」**在常量层面可见**。

SPEC_PRERUN_MEDIAN_BIAS = {0.0: 0.000, 0.10: 0.007, 0.40: 0.048, 0.49: 0.103, 0.50: 2.000, 0.90: 3.993}   # 来源等级：[E]
SPEC_PRERUN_MEAN_BIAS = {0.0: 0.000, 0.10: 0.400, 0.40: 1.600, 0.49: 1.960, 0.50: 2.000, 0.90: 3.600}   # 来源等级：[E]
# 来源：规格 Chunk 4 文档 01 的预跑表（**E 级：规格作者的建模结果**）。
# 口径：`n = 100`、冷启动 5×、稳态 σ = 0.05、**保留全部数据**、3000 次重复取均值。
# ★ 本库复算的**唯一差别**在 `k/n = 0.90` 那一格：规格 3.993、本库 4.000 —
#   成因是规格的冷启动值带了一点噪声，而本库模型里冷启动值**是确定值**
#   （0.90 时中位数落在两个冷启动值中间 ⇒ 每个重复都恰好 4.000，标准差为 0）。

SPEC_MEDIAN_MECHANISM_IS_K_OVER_HALF = False   # 来源等级：[E]
SPEC_MEDIAN_MECHANISM_NAME = "position_pushed_into_steady_upper_tail"   # 来源等级：[E]
# 来源：规格已自订正（Chunk 4 文档 01）；本库 [E] 复算确认。
# ★★ 机制：保留冷启动值时，**中位数在合并样本里的名次被推高**，落到稳态子样本的**上尾**
#    （不是「k 超过半数才污染」）。判据是两个点都要断言：
#    `k/n = 0.49` 偏差 > 0.05（**超过一个稳态 σ**）、`k/n = 0.10` 偏差 < 0.02。

SPEC_PRERUN_MDE = {5: (0.0531, 0.0548), 25: (0.0238, 0.0239), 100: (0.0119, 0.0119)}   # 来源等级：[E]
# 来源：规格 Chunk 4 文档 02 的预跑表（E 级）。口径：`(解析式, 蒙特卡洛)`，σ = 3%，两组各 n 次。

SPEC_CLAIM_MDE_USED_SQRT_1_OVER_N = False   # 来源等级：[E]
# 来源：规格已自订正。★ 初稿用 `t·σ/√n`（**漏 √2**）；代回 `σ = 3%, n = 25` 得 1.7%，
#   而正确式给 2.4%、蒙特卡洛给 2.4% ⇒ **初稿的式子偏小 1.41 倍**。本库复算确认。

SPEC_PRERUN_FP_RATE_PCT = {1.645: 6.95, 1.960: 4.28, 2.132: 3.29, 2.776: 1.20}   # 来源等级：[E]
SPEC_CLAIM_T4_CRIT_IS_ONE_SIDED_2_5PCT = False   # 来源等级：[E]
SPEC_CLAIM_WELCH_DF_CAUSES_FP_INFLATION = False   # 来源等级：[E]
SPEC_WELCH_DF_EFFECT_DIRECTION = "conservative"   # 来源等级：[E]
# 来源：**本库实跑推翻规格的两条**（★★ 本主题最深的一处订正）。
# ★ 事实①：`2.132` 是 **t(4) 在 0.95** 的分位（单侧 5%），**不是** t(4) 的单侧 2.5%
#   （那是 `2.776`）。规格把它标成「t(4) 单侧 2.5%」——**标签错**。
# ★★ 事实②：那 3.29% 与 Welch 自由度**无关**。两组等 n、等方差时，
#   `√(s²ₐ/n + s²_b/n) ≡ √((s²ₐ+s²_b)/n)` 与合并版 `sp·√(2/n)` **代数恒等**，
#   所以统计量**恰好**是 t(2n−2) 分布；固定的临界值下，假阳性率**只由临界值决定**：
#   `P(t₈ > 2.132) = 3.280%` —— 与规格实测的 3.29% 逐位吻合。
#   ⇒ 规格那句「因为自由度由 Welch–Satterthwaite 决定、小于 2n−2」**既没进入这个计算、
#     方向也反了**：用 Welch 自由度当临界值会让检验**更保守**（本库实测 2.25% < 2.5%）。
# ★ 三点合起来的判据（进 02 篇 §5）：**报显著必须写明临界值与自由度的算法**；
#   本设计的正确临界值是 `t(2n−2)`（n = 5 时 2.306 → 恰好 2.5%）。

SPEC_PRERUN_FALSE_VICTORIES = {10: 0.35, 50: 1.75, 200: 6.90}   # 来源等级：[E]
SPEC_PRERUN_FALSE_VICTORY_NOMINAL = {10: 0.25, 50: 1.25, 200: 5.00}   # 来源等级：[E]
# 来源：规格 Chunk 4 文档 03 的预跑表（E 级）。口径：真实增益恒为 0、每组 n = 5、σ = 3%、
#       用**单侧**临界值累计「假胜利」次数/试验；右列是名义 2.5% 的理论值。
# ★ 本库复算：用规格那个临界值 2.132（每步 3.28%）得 0.33 / 1.64 / 6.56，
#   与规格的 0.35 / 1.75 / 6.90 相差 ≤ 5%——**残差超出本库 20000 次重复的噪声**
#   （每步 SE ≈ 0.0013）⇒ 规格那组数应在**更低重复次数**下取得。
#   用**正确**的临界值 2.306 则恰好回到名义值（0.25 / 1.25 / 5.00）。

SPEC_CLAIM_MEDIAN_MECHANISM_IS_K_OVER_HALF_LABEL = "k_over_half_only"   # 来源等级：[E]
SPEC_CLAIM_GRAPH_AB_ENV_VAR = "HIPFIRE_VERIFY_GRAPH"   # 来源等级：[E]
SPEC_CLAIM_GRAPH_AB_ENV_VAR_IS_REPRODUCIBLE = False   # 来源等级：[E]
SPEC_CLAIM_GRAPH_OFF_SPREAD_PCT = 0.1   # 来源等级：[E]
SPEC_CLAIM_GRAPH_ON_SPREAD_PCT = (1.5, 3.0)   # 来源等级：[E]
SPEC_CLAIM_GRAPH_SPREAD_VALUES_ARE_REPRODUCIBLE = False   # 来源等级：[E]
# 来源：**本库两轮独立检索推翻**（E 级记录）。§0.7.G 声称「Chunk 1 已核到原文逐字：
#       `HIPFIRE_VERIFY_GRAPH=0` gives ~0.1% spread versus 1.5–3% with graph capture on」。
# ★ 事实①：两轮检索都**没有**命中 `HIPFIRE_VERIFY_GRAPH` 这个名字；命中的是
#   `HIPFIRE_GRAPH=0/1`（graph-ab 对照）。
# ★★ 事实②：两轮检索都**没有**命中 graph 开/关的两个 spread 数值（0.1% / 1.5–3%）。
#   ⇒ 处置：**这四个字面量进 `UNSOURCED`，正文不得使用**，连「关掉 graph 更稳」这个
#     **反直觉方向**也不能写——方向也需要来源。

SPEC_CLAIM_COLD_COMPILE_TIME_S = (30.0, 120.0)   # 来源等级：[E]
SPEC_CLAIM_COLD_COMPILE_IS_REPRODUCIBLE = False   # 来源等级：[E]
# 来源：规格 §0.7.G 的 D 级行「JIT 冷缓存首次编译 30 s–2 min」。★ 两轮检索**未命中**该区间；
#   命中的是**形态完全不同**的一组 B 级数：首请求编译尖峰 2–3 s、FlashInfer 首次最长约 10 min、
#   DeepGEMM 约 5 min。⇒ **处置：该区间进 UNSOURCED、正文不用**（规格原本就写了「核不到就删」）。
#   ★ 保留这条常量的目的是让**「我们核过、核不到」这件事本身可复核**。

SPEC_CLAIM_MODEL_RELOAD_COLD_START_S = (2.0, 5.0)   # 来源等级：[E]
SPEC_CLAIM_MODEL_RELOAD_IS_REPRODUCIBLE = False   # 来源等级：[E]
# 来源：规格 §0.7.G 的 D 级行「模型重载冷启动 2–5 s」。★ 检索**未命中**与「模型重载」配对的
#   2–5 s；最接近的 B 级数是「**首个请求的 JIT 编译尖峰** 2–3 s」（对象不同，见 P9）
#   与「**权重加载** 约 32 s（缓存）/ 5–10 min（需下载）」（量级不同）。
#   ⇒ 处置：该区间进 UNSOURCED、正文不用。

SPEC_CLAIM_NOISE_BAND_IS_GENERAL = False   # 来源等级：[E]
SPEC_CLAIM_TIGHT_SPREAD_SOURCE_TIER = "D"   # 来源等级：[E]
RETRIEVED_TIGHT_SPREAD_SOURCE_TIER = "C"   # 来源等级：[E]
# 来源：**本库对规格来源分级的订正**（E 级记录）。
# ★ 规格 §0.7.G 把「紧得可疑的标准差反而可疑」标成 **D 级（hipfire）**；
#   两轮检索**没有**在 hipfire 文档里找到该说法（找到的是**方向相反**的一条：
#   散得太开 `(max−min)/median > 0.20` 判为 unstable）。
#   ⇒ 处置：**换来源**——该机制由 §5 的两篇 C 级文献与 §6 的 A 级分辨率事实支撑。
#     规格的「D 级」标签不成立（不是数字错，是**归属错**：P7 的一种）。

SPEC_CLAIM_MEASUREMENT_STATISTIC_DEFAULT_IS_MEAN = True   # 来源等级：[B]
# 来源：B 级（Triton 默认 `return_mode='mean'`）+ 本库 [E] 建模。
# ★ 单列它是为了让 01 篇有**一条可反驳的论断**：在本库模型下（`k/n = 0.10`），
#   均值偏差是 0.400、中位数偏差是 0.007 ⇒ **官方默认返回的统计量对冷启动最敏感**。


# ===========================================================================
# 9. UNSOURCED —— **待办清单，不是免责声明**
# ===========================================================================
#
# ★ 域 README 硬规则 5：**不得「既保留数字又记 UNSOURCED」**，二者只能取其一。
#   下面每一条都是**本库主动选择不用**的东西（正文与常量名里都不出现）。

UNSOURCED = (   # 来源等级：[待办清单]（不是来源，是本库主动不用的清单]
    # 规格声称逐字核到、本库两轮检索都核不到（见 §8 的三条 SPEC_CLAIM_*）
    "hipfire_graph_on_off_spread_values_0p1_vs_1p5_3pct",
    "hipfire_verify_graph_env_var_name",
    "jit_cold_compile_30s_to_2min_interval",
    "model_reload_cold_start_2s_to_5s_interval",
    # 本库自己的模型参数与外部噪声带之间没有换算关系，故刻意不换算
    "noise_band_vs_sigma_mapping",
    # 检索到但证据强度不够、因此不进正文的
    "hipfire_dpm_warmup_secs_value",
    "hipfire_warmup_equals_128_low_bias_claim",
    "tight_spread_suspicion_in_hipfire_doc",
    "mytkowicz_40pct_figure",
    "mlcommons_official_rulebook_warmup_and_run_counts",
    "cold_start_component_residual_81s",
    "real_system_fake_speedup_attributable_to_warmup",
)
# 来源：**本库检索记录 [E]**。★ 逐条说明为什么不用：
#   · 前四条：规格声称有、本库核不到 ⇒ **数字与名字都不用**（含「关掉 graph 更稳」的方向）。
#   · `noise_band_vs_sigma_mapping`：§1 的 D 级噪声带是**相对值的区间**，
#     本库 §7 的 σ 是**模型参数**——把两者换算成同一个数是**本库编的换算**，故不做。
#   · `hipfire_dpm_warmup_secs_value` / `hipfire_warmup_equals_128_low_bias_claim`：
#     规格已订正过一次（warmup=128 那条数字在原文里检索不到）⇒ 本库也不使用。
#   · `tight_spread_suspicion_in_hipfire_doc`：见 §8 的 `RETRIEVED_TIGHT_SPREAD_SOURCE_TIER`。
#   · `mytkowicz_40pct_figure`：检索**明确不支持** 40% 这个数（支持的是 33% / 近 300%）。
#   · `mlcommons_official_rulebook_warmup_and_run_counts`：没取到官方 rulebook ⇒
#     §5 的三条只写成「MLPerf 对齐的方法学」，不写成规范。
#   · `cold_start_component_residual_81s`：分量清单不完整（101 vs 182）⇒ **残差不作数**。
#   · `real_system_fake_speedup_attributable_to_warmup`：本库只有「模型 + 一个官方 issue」，
#     没有「某真实系统因 warmup 导致对外宣称的加速比是假的」的可引用案例 ⇒ **不写这种话**。
