"""auto-tuning 主题的外部事实源：每条常量带一行 `# 来源：` 与**可打开的 URL**。

本文件只放**从一手来源抄来的**东西（论文 PDF 原文 / 固定 tag 的源码），不放推导值，
不放本仓库自己跑出来的数。找不到可定位载体的项**不写进来**，记入文件末尾的 UNSOURCED。

★ 版本坐标（本主题强制）
  · **Triton 侧全部钉在 tag `v3.6.0`**。这是**会漂移的坐标**——最新的 tag 是 `v3.8.0`，
    「磁盘缓存默认关闭」在 v3.4.0–v3.8.0 全部成立（3.3.0 及更早根本没有磁盘缓存），
    v3.8.0 另新增了「剪枝后只剩一个 config 时的短路」。**凡写 Triton 默认行为必须带版本。**
  · **Ansor 侧钉在论文本身**（OSDI '20），不是任何代码仓库——Ansor 从来不是 TVM 里的模块。

★ ★ 本文件的性质（**必须如实说明，错误模式 P6 的对策**）
  本主题执行时，本仓库所在环境**没有出网**：`curl` 与网页抓取都被挡。
  因此下面的引文与默认值**不是本次执行现场读源码得到的**，而是**照抄控制者基线**
  （计划 §0.7.E，由**两路独立研究**对照一手来源核实后写入）。逐条标了
  `BASELINE` 或 `BASELINE_VERIFIED`，并给出**该来源的可打开 URL**（URL 的形态是标准的
  `github.com/<org>/<repo>/blob/<tag>/<path>` 与论文官方 PDF 链接）。
  **本仓库不复核这些 URL 的内容**——复核它们的测量口径与文字超出了本仓库的范围。
  凡引文，正文引用时都按「转抄自控制者基线」处理，**不声称是本仓库现场核对过的**。

★ 本文件的任何条目**都不得作为编号程序的断言对象**（§0.3 规矩 4）：
  程序断言的是本仓库的迷你实现，真实 Ansor / Triton 的行为只出现在文档与这里。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · 「搜到多少个配置之后收益饱和」——**无定论**。研究代理明确报告找不到任何官方文档或
    论文给出**通用**结论；能查到的只有绑定具体 workload / 硬件 / 实现的经验曲线。
    **拿 GCC flag 调优的收敛结论类推张量编译器是无效的**（机制不同）。
    正文只能写「无定论」，并给出 ANSOR_TRIALS_SETTING 作为**单篇论文在其基准上的设置**。
    **不得给出任何区间或拐点数字。**
  · 「MetaSchedule 是 Ansor 的工程化 / 实现」——**推断，TVM 源码无明文**。
    能核到的是「MetaSchedule 由一份独立的 RFC（AutoTensorIR）提出，第三方文献用
    `absorbs` / `unifies` 这类**等价**表述」。**不得写成直接引用，也不得写成「改名」。**
    （本条目只有控制者基线，**没有落到可打开的 URL**，故正文降级为推断表述。）
  · Ansor 特征集里「**不存在**网格规模 / 问题规模特征」——论文列出的是**逐语句**的
    算术特征与访存特征两类（见 ANSOR_FEATURES）。「块数 / 网格规模是 program 级属性、
    不在逐语句特征里」是**本文的推断**，不是论文原句。正文引用时必须标注为推断。
  · 「TVM 默认 pipeline 的 pass 数」「AutoTVM 的历史默认值」等属于 `60-compilers/tvm`
    主题，正文位置在那里，本文件不重复收录（CONTRIBUTING §7.1）。
"""

# ---------------------------------------------------------------------------
# 一、Ansor 论文（OSDI '20）
# ---------------------------------------------------------------------------

ANSOR_PAPER_TITLE = "Ansor: Generating High-Performance Tensor Programs for Deep Learning"

# 来源：https://www.usenix.org/system/files/osdi20-zheng.pdf
#       （14th USENIX OSDI，2020-11-04/06；以下简写为 ANSOR_PDF）
ANSOR_PDF = "https://www.usenix.org/system/files/osdi20-zheng.pdf"
ANSOR_VENUE = "14th USENIX OSDI, 2020-11-04 .. 2020-11-06"

# 来源：ANSOR_PDF，§3 开头，逐字
ANSOR_HIERARCHICAL_SPACE = (
    "we define a hierarchical search space with two levels: sketch and annotation. "
    "At the top level, we generate sketches by recursively applying a few derivation "
    "rules. At the bottom level, we randomly annotate these sketches to get complete "
    "programs."
)

# 来源：ANSOR_PDF，Table 1 的规则名（**只抄名字**；除下面那条 fusion 规则外，
#       本文档不逐字引用这些规则的公式——引用别的公式需要另外核一手）
ANSOR_DERIVATION_RULE_NAMES = (
    "Skip",
    "AlwaysInline",
    "MultiLevelTiling",
    "MultiLevelTilingWithFusion",
    "AddCacheStage",
    "ReductionFactorization",
    "UserDefinedRule",
)

# 来源：ANSOR_PDF，derivation rule 原文（这是本主题**唯一一条逐字的规则公式**——
#       它证明 sketch 阶段**已经包含 fusion 决策**）
ANSOR_FUSION_RULE_VERBATIM = (
    "HasDataReuse ∧ HasFusibleConsumer → FuseConsumer(MultiLevelTiling(S,i), i)"
)

# 来源：ANSOR_PDF，§5 代价模型一节，逐字（**这条常被说反**）
ANSOR_NO_INCREMENTAL_UPDATES = (
    "we train a new model every time instead of doing incremental updates"
)

# 来源：ANSOR_PDF，§5.2 特征一节（两类特征；本仓库据此标注「逐语句」）
ANSOR_FEATURES = (
    "arithmetic features and memory access features; features are extracted for each "
    "innermost non-loop statement, and the score of a whole program is the sum of the "
    "predicted scores of its statements"
)

# 来源：ANSOR_PDF，§5.3 损失函数，逐字
ANSOR_LOSS_VERBATIM = "loss(f,P,y) = y(Σ_{s∈S(P)} f(s) − y)²"
ANSOR_LOSS_NOTE = ("weighted squared error；**权重直接取吞吐 y**，不是常数 1")

# 来源：ANSOR_PDF，引文 [9]（XGBoost）
ANSOR_COST_MODEL_TYPE = "gradient boosting decision tree (GBDT)"
ANSOR_XGBOOST_REF = "XGBoost (Chen & Guestrin, KDD 2016)"
ANSOR_XGBOOST_URL = "https://arxiv.org/abs/1603.02754"

# 来源：ANSOR_PDF，§4 训练数据规模，逐字口径
ANSOR_TRAINING_DATA_SCALE = "less than 30,000 measured programs"

# 来源：ANSOR_PDF，§7.5 模型质量（25,000 个程序，20,000 训练 / 5,000 测试）
#       ★ 这组数字是**代价模型自身的性质**，正文位置在本主题 doc02
ANSOR_MODEL_QUALITY = {
    "programs": 25000,
    "train": 20000,
    "test": 5000,
    "rmse": 0.079,
    "r2": 0.958,
    "pairwise_accuracy": 0.851,
    "recall_at_30": 0.624,
}

# 来源：ANSOR_PDF，§7.4 Table 3 的**标题原文**（★ 口径是「追平 AutoTVM 所需成本」，
#       **不是**「达到最优所需成本」——不写明会把上界说成下界）
ANSOR_TABLE3_CAPTION = (
    "The number of measurements and wall-clock time used for Ansor to match the "
    "performance of AutoTVM on the Intel CPU (batch size=1)"
)

# 来源：ANSOR_PDF，§7.4 Table 3 全表
#   (网络, AutoTVM 测量次数, Ansor 测量次数, 测量次数节省, AutoTVM 墙钟秒, Ansor 墙钟秒, 墙钟节省)
ANSOR_TABLE3 = (
    ("ResNet-50", 21220, 6403, "3.3x", 39250, 4540, "8.6x"),
    ("Mobilenet-V2", 31272, 1892, "16.5x", 58468, 660, "88.6x"),
    ("3D-ResNet", 5158, 1927, "2.7x", 7594, 2296, "3.3x"),
    ("DCGAN", 3003, 298, "10.1x", 4914, 420, "11.7x"),
    ("BERT", 6220, 496, "12.5x", 12007, 266, "45.1x"),
)

# 来源：ANSOR_PDF，§7.3 正文原句（★ 引用**必须用正文口径**，不要用 abstract 口径——
#       abstract 写的是 "relative to the state-of-the-art"，正文限定更严：
#       "Compared with **the best alternative**" 且是 **up to**）
ANSOR_SPEEDUP_VERBATIM = (
    "Compared with search-based AutoTVM, Ansor matches or outperforms it in all cases "
    "with 1.0–21.8× speedup. Compared with the best alternative, Ansor improves "
    "the execution performance of DNNs on the Intel CPU, ARM CPU, and NVIDIA GPU by up "
    "to 3.8×, 2.6×, and 1.7×"
)

# 来源：ANSOR_PDF，§7.4 正文
ANSOR_SEARCH_COST_VERBATIM = (
    "Ansor can match the performance of AutoTVM with an order of magnitude less "
    "search time"
)
ANSOR_SEARCH_COST_HOURS = (
    "Typically, it takes several hours for Ansor to generate fully-optimized programs "
    "for a DNN on a single machine"
)

# 来源：ANSOR_PDF，§7.4 实验设置，逐字——**这是本主题唯一可引的「搜多少个配置」
#       的数字，且必须写明它是该论文在其基准上的设置，不是普适定律**
ANSOR_TRIALS_SETTING = (
    "We let both AutoTVM and Ansor run auto-tuning until they use to 1000 × n "
    "measurement trials on each DNN, where n is the number of subgraphs in the DNN. "
    "This is typically enough for them to converge."
)
ANSOR_TRIALS_SETTING_NOTE = (
    "**单篇论文在其基准上的实验设置**，不是「搜到 N 个就饱和」的普适定律；"
    "n 是该 DNN 的子图个数，所以它是**按图缩放**的一个数，不是一个常数"
)

# ---------------------------------------------------------------------------
# 二、Triton autotuner（固定 tag v3.6.0）
# ---------------------------------------------------------------------------

TRITON_REPO = "https://github.com/triton-lang/triton"
TRITON_TAG = "v3.6.0"
TRITON_TAG_LATEST = "v3.8.0"
TRITON_AUTOTUNER_PATH = "python/triton/runtime/autotuner.py"
TRITON_AUTOTUNER_URL = "%s/blob/%s/%s" % (TRITON_REPO, TRITON_TAG, TRITON_AUTOTUNER_PATH)

# 来源：TRITON_AUTOTUNER_URL
#       `Autotuner.__init__` 的默认值（BASELINE_VERIFIED，计划 §0.7.E 第（五）节）
TRITON_DEFAULTS = {
    "cache_results": False,       # ★ 磁盘缓存**默认关闭**
    "perf_model": None,           # ★ 默认无性能模型
    "configs_top_k": 1.0,         # ★ 属性名是 configs_top_k（`top_k` 只是
                                  #    `prune_configs_by` 字典的键，别混）
    "early_config_prune": None,   # ★ 默认无早期剪枝
    "prune_configs_by": None,
    "do_bench": "quantiles=(0.5, 0.2, 0.8)：用**中位数**比较",
}

# 来源：同上，调优路径的守卫（逐字）
TRITON_SINGLE_CONFIG_GUARD = "if len(self.configs) > 1:"
TRITON_SINGLE_CONFIG_NOTE = (
    "只有一个 config 时**根本不走调优路径**；configs 为空时默认 "
    "Config({}, num_warps=4, num_stages=3, num_ctas=1)"
)

# 来源：同上，默认路径逐字：对（剪枝后）**全部** config 逐个 bench，再取中位数最小者。
#       ★ 结论：**默认不是「搜索」而是穷举，且默认无任何剪枝**
TRITON_DEFAULT_SELECTION = "min(timings, key=timings.get)"

# 来源：同上，编译失败的处理
#       ★ 返回的是 **[inf, inf, inf] 三元组**，不是单个 inf（计划 §3 A9 订正）
TRITON_COMPILE_FAILURE_RETURN = "[inf, inf, inf]"
TRITON_COMPILE_FAILURE_NOTE = (
    "OutOfResources / CompileTimeAssertionFailure / PTXASError 都走这条路，"
    "等价于自动出局（中位数 inf 不会被 min 选中）"
)

# 来源：同上，磁盘缓存
TRITON_DISK_CACHE_FILE = "{kernel_name}.autotune.json"
TRITON_DISK_CACHE_NOTE = (
    "内存缓存 self.cache: Dict[Tuple, Config] 一直有；**磁盘缓存默认关闭**，"
    "需显式 cache_results=True 或 knobs.autotuning.cache；"
    "**带 pre_hook 的 config 一律不落盘**"
)

# 来源：同上，key 的构成
TRITON_KEY_NOTE = (
    "key 是参数名列表；运行时 key = 这些参数本次调用的取值元组，"
    "**且每个带 dtype 的参数会把 dtype 字符串追加进 key**；key 不在 cache 里才重新调优"
)

# 来源：同上，已废弃参数
TRITON_DEPRECATED = "warmup / rep / use_cuda_graph（会发 DeprecationWarning）"

# 来源：计划 §3 A9（控制者基线）——版本区间上的成立范围
TRITON_VERSION_NOTES = (
    "「磁盘缓存默认关闭」在 v3.4.0–v3.8.0 **全部成立**；3.3.0 及更早**没有磁盘缓存**；"
    "v3.8.0 新增了「剪枝后只剩一个 config 时的短路」（v3.6.0 没有这一步）"
)

# ---------------------------------------------------------------------------
# 三、交叉引用（正文位置在别的主题，本文件只登记，不复制）
# ---------------------------------------------------------------------------

# 来源：计划 §0.7.C 第（四）（七）节（控制者基线）
CROSS_REF_NOTES = (
    "· MetaSchedule 的默认组件（PostOrderApply / EvolutionarySearch / XGBModel / "
    "JSONDatabase / GradientBasedScheduler）与 Ansor 不是模块这件事："
    "**正文在 `60-compilers/tvm/04-auto-scheduling`**，本主题只回链。"
    "· `tvm.s_tir.Schedule` 的 API 断言（含 `compute_root` 不在 API 里）："
    "正文在 `60-compilers/tvm`，钉在 TVM tag v0.26.0。"
    "· 融合的字节数收益上界（链长 k）的推导：正文在 `40-kernels/elementwise-fusion/02`。"
)

# ---------------------------------------------------------------------------
# 四、本主题「引用值」的唯一正文位置
# ---------------------------------------------------------------------------

# ANSOR_TABLE3 / ANSOR_SPEEDUP_VERBATIM / ANSOR_SEARCH_COST_* / ANSOR_TRIALS_SETTING
# 的**正文位置只有一处**：`60-compilers/auto-tuning/03-search-algorithms` §5。
# CONTRIBUTING §7.1：一个知识点只有一处正文，其它地方只回链、不复制。
# ANSOR_MODEL_QUALITY 的正文位置是 `60-compilers/auto-tuning/02-cost-models`
# （那是代价模型**自身**的性质，不是搜索结果）。
REFERENCE_VALUE_HOME = {
    "search_cost_table": "auto-tuning/03-search-algorithms §5",
    "speedup": "auto-tuning/03-search-algorithms §5",
    "convergence_setting": "auto-tuning/03-search-algorithms §5",
    "model_quality": "auto-tuning/02-cost-models §2/§5",
}

# 全部引用值统一标注（§0.6：引用值隐含来源存在，且必须注明不复核测量口径）
REFERENCE_VALUE_CAVEAT = "引用值，本仓库不复核其测量口径"
