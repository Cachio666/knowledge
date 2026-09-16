"""迷你自动调优基础设施（非编号辅助模块）。

本模块是 60-compilers/auto-tuning 三个编号程序共用的**本仓库迷你实现**：
搜索空间构造 / 代价地形 / 迷你 GBDT / 搜索算法 / 迷你 autotuner / 三类断言框架。

★ 这里没有 Ansor、没有 TVM、没有 Triton、不碰硬件、不产生任何时间数字。
  全部数字都是**本仓库自己定的**量纲（「周期」「字节」）。
  真实系统的行为只在 `specs.py` 与文档里，逐条带来源，**不作为本模块的断言对象**。

★ 主线口径（计划 §0.8）：**物化 = 为中间结果分配独立的存储**（寄存器 / 共享内存 /
  全局内存都算），**重算 = 根本不分配存储，把计算替换进消费者的表达式**。
  本模块里 `matmul(A,B) -> C; relu(C) -> D` 的 C 就是那个决策点：物化在 gmem、
  物化在 smem、还是融合进消费者（不物化）。

★ 地形构造的纪律（计划 §0.4 第 5 条）：下面的 `true_cost` 与两个 DAG 的尺寸、
  以及全部搜索超参数，都在**跑出任何结果之前**写死在本文件里；
  编号程序不得「看到结果再调地形」。
"""

import math
import sys

import numpy as np

INF = float("inf")

# ===========================================================================
# 0. 断言框架（与 60-compilers/mlir、tvm、xla 的 Checks 同形）
# ===========================================================================


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


class Checks:
    """A / B / C 三类断言的记账器。失败累计，`summary()` 以非零码退出。"""

    def __init__(self, title=""):
        self.title = title
        self.counts = {"A": 0, "B": 0, "C": 0}
        self.failures = []

    def _record(self, cls, ok, msg, detail=""):
        self.counts[cls] += 1
        line = "  [%s] %s: %s" % ("ok  " if ok else "FAIL", cls, msg)
        if detail:
            line += "  | " + detail
        print(line)
        if not ok:
            self.failures.append((cls, msg))
        return ok

    def A(self, ok, msg, detail=""):
        return self._record("A", ok, msg, detail)

    def B(self, ok, msg, detail=""):
        return self._record("B", ok, msg, detail)

    def C(self, ok, msg, detail=""):
        return self._record("C", ok, msg, detail)

    def summary(self):
        total = sum(self.counts.values())
        print()
        print("-" * 78)
        if self.failures:
            for cls, msg in self.failures:
                print("  失败项 [%s] %s" % (cls, msg))
            print("断言汇总：通过 %d 条断言，失败 %d 条（共 %d 条）"
                  % (total - len(self.failures), len(self.failures), total))
            sys.exit(1)
        print("通过 %d 条断言（A %d / B %d / C %d）"
              % (total, self.counts["A"], self.counts["B"], self.counts["C"]))
        print("-" * 78)


# ===========================================================================
# 1. 搜索空间：两级结构（sketch + annotation）
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.1 两个 mini DAG
#
# ★ 尺寸在跑任何结果之前写死。两个 DAG 的 **K 相同、M/N 不同**，这是故意的：
#   Ansor 的代价模型特征是**逐语句**的（见 statements_of），而「网格规模 = 块数」
#   是 **program 级**属性——于是同一个 tile 在两个 DAG 上抽出**同一组特征**，
#   共享模型无法区分它们。doc02 的 C1 断言用的就是这个机制。
# ---------------------------------------------------------------------------


class DAG:
    def __init__(self, name, M, N, K, predicates=()):
        self.name = name
        self.M, self.N, self.K = M, N, K
        self.predicates = frozenset(predicates)

    def __repr__(self):
        return "DAG(%s, M=%d, N=%d, K=%d, %s)" % (
            self.name, self.M, self.N, self.K, sorted(self.predicates))


# 「大图」：512x512x512。tile 大 → 块数仍 >= SM 数 → 大 tile 真的更好。
DAG_BIG = DAG("big", 512, 512, 512, ("HasDataReuse", "HasFusibleConsumer"))
# 「小图」：128x128x512。同一个 tile 只有 1 个块 → 网格填不满 → 大 tile 反而糟。
DAG_SMALL = DAG("small", 128, 128, 512, ("HasDataReuse", "HasFusibleConsumer"))
# 「无消费者」的图：没有可融合的消费者 → 那条 fusion 规则不适用。
DAG_NO_CONSUMER = DAG("nofuse", 512, 512, 512, ("HasDataReuse",))

# ---------------------------------------------------------------------------
# 1.2 推导规则（Ansor Table 1 的规则名；**只有 fusion 那条是逐字公式**）
# ---------------------------------------------------------------------------

DERIVATION_RULES = (
    dict(name="Skip", requires=(),
         formula="（本迷你实现的口径）恒适用：不施加任何循环变换",
         creates_loops=False),
    dict(name="AlwaysInline", requires=(),
         formula="（本迷你实现的口径）恒适用：消费点直接重算，不分配存储",
         creates_loops=False),
    dict(name="MultiLevelTiling", requires=("HasDataReuse",),
         formula="（本迷你实现的口径）HasDataReuse → MultiLevelTiling(S, i)",
         creates_loops=True),
    dict(name="MultiLevelTilingWithFusion",
         requires=("HasDataReuse", "HasFusibleConsumer"),
         formula="HasDataReuse ∧ HasFusibleConsumer → "
                 "FuseConsumer(MultiLevelTiling(S,i), i)   ← 论文逐字，见 specs",
         creates_loops=True),
    dict(name="AddCacheStage", requires=("HasDataReuse",),
         formula="（本迷你实现的口径）HasDataReuse → AddCacheStage(MultiLevelTiling(S,i))",
         creates_loops=True),
    dict(name="ReductionFactorization", requires=("HasDataReuse",),
         formula="（本迷你实现的口径）HasDataReuse → ReductionFactorization(S, k)",
         creates_loops=True),
)

RULE_BY_NAME = dict((r["name"], r) for r in DERIVATION_RULES)

# ★ sketch 层的 fusion 决策：这两个 sketch 一旦被选中，融合/重算就已经定了，
#   annotation 改不了它。（`AlwaysInline` = 不物化 = 重算；`...WithFusion` = 融合消费者。）
FUSION_DECIDED_AT_SKETCH = ("AlwaysInline", "MultiLevelTilingWithFusion")
# ★ 这两个 sketch 不创建循环嵌套，因此**忽略全部 tile/线程标注**
ANNOTATION_FREE_SKETCHES = ("Skip", "AlwaysInline")


def available_sketches(dag):
    """按规则的适用条件筛出该 DAG 上可用的 sketch。"""
    out = []
    for r in DERIVATION_RULES:
        if all(p in dag.predicates for p in r["requires"]):
            out.append(r["name"])
    return out


# ---------------------------------------------------------------------------
# 1.3 annotation 的取值网格（名义空间 = 这些范围的**笛卡尔积**）
# ---------------------------------------------------------------------------

TILE_CHOICES = (16, 32, 64, 128, 256, 512)
TK_CHOICES = (8, 16, 32, 64, 128)
THREAD_CHOICES = (32, 64, 128, 256, 512, 1024)
VEC_CHOICES = (1, 2, 4, 8)


class Config:
    """一个完整程序 = 一个 sketch + 一组 annotation。"""

    __slots__ = ("sketch", "tm", "tn", "tk", "threads", "vec")

    def __init__(self, sketch, tm, tn, tk, threads, vec):
        self.sketch = sketch
        self.tm, self.tn, self.tk = tm, tn, tk
        self.threads = threads
        self.vec = vec

    def __repr__(self):
        return "%s(tm=%s,tn=%s,tk=%s,T=%s,v=%s)" % (
            self.sketch, self.tm, self.tn, self.tk, self.threads, self.vec)

    def as_tuple(self):
        return (self.sketch, self.tm, self.tn, self.tk, self.threads, self.vec)


def nominal_space(dag):
    """名义搜索空间：可用 sketch × annotation 网格的笛卡尔积（**不去重、不过滤**）。"""
    out = []
    for s in available_sketches(dag):
        for tm in TILE_CHOICES:
            for tn in TILE_CHOICES:
                for tk in TK_CHOICES:
                    for th in THREAD_CHOICES:
                        for v in VEC_CHOICES:
                            out.append(Config(s, tm, tn, tk, th, v))
    return out


# ---------------------------------------------------------------------------
# 1.4 有效化：等价类（生成同一段代码）+ 不可行
# ---------------------------------------------------------------------------

def effective_tiles(dag, cfg):
    """实际生成的 tile。

    · `ANNOTATION_FREE_SKETCHES` 不创建循环嵌套 → 标注全部无效，tile 就是整个问题；
    · 其余 sketch 的 tile 会被**裁剪到 extent**（tile 比 extent 大不产生新代码）。
    """
    if cfg.sketch in ANNOTATION_FREE_SKETCHES:
        return (dag.M, dag.N, dag.K)
    return (min(cfg.tm, dag.M), min(cfg.tn, dag.N), min(cfg.tk, dag.K))


def canonical_key(dag, cfg):
    """**代码等价**的规范键：生成同一段代码的配置共享同一个键。

    三处等价来源（都是编译器里真实存在的）：
      1. 忽略标注的 sketch（Skip / AlwaysInline）——把全部 tile 选择压成一类；
      2. tile 被裁剪到 extent——tile 与 extent 取同一个上界时是同一段代码；
      3. 线程数被 tile 面积掐住——多于 tile 点数的线程不产生任何额外工作。
    """
    tm, tn, tk = effective_tiles(dag, cfg)
    if cfg.sketch in ANNOTATION_FREE_SKETCHES:
        return (cfg.sketch, None, None, None, min(cfg.threads, dag.M * dag.N),
                min(cfg.vec, dag.N))
    return (cfg.sketch, tm, tn, tk, min(cfg.threads, tm * tn), min(cfg.vec, tn))


def infeasible_reason(dag, cfg):
    """不可行的理由（`[类别] 详情`）；可行返回 None。

    四条硬约束（**全部是本迷你实现的口径**，不是任何真实编译器的规格）：
      · 向量宽度受 f32 的 128 位访存上界限制；
      · 每线程至少要摊到 4 个输出元素（否则寄存器分块退化成无分块）；
      · 线程数上界；
      · 一个块的共享内存占用要能放进一个 SM。
    """
    tm, tn, _ = effective_tiles(dag, cfg)
    if cfg.vec > MINI_MAX_VEC_WIDTH:
        return "[vec] vec=%d > f32 的 128 位访存上界 %d" % (cfg.vec, MINI_MAX_VEC_WIDTH)
    if cfg.threads > MINI_MAX_THREADS:
        return "[threads] threads=%d > %d" % (cfg.threads, MINI_MAX_THREADS)
    if cfg.threads * MINI_MIN_ELEMS_PER_THREAD > tm * tn:
        return "[register-block] 每线程不足 %d 个输出元素（tile 面积 %d < %d×%d）" % (
            MINI_MIN_ELEMS_PER_THREAD, tm * tn, cfg.threads,
            MINI_MIN_ELEMS_PER_THREAD)
    sb = smem_bytes(dag, cfg)
    if sb > MINI_SMEM_BYTES_PER_SM:
        return "[smem] 共享内存 %d 字节 > 每 SM %d 字节" % (sb, MINI_SMEM_BYTES_PER_SM)
    return None


def feasible(dag, cfg):
    return infeasible_reason(dag, cfg) is None


def effective_space(dag, space=None):
    """有效空间：**去等价 + 去不可行**之后每个等价类留一个代表。

    返回 (代表配置列表, 统计字典)。
    """
    if space is None:
        space = nominal_space(dag)
    reps, seen = [], {}
    n_infeasible = 0
    for c in space:
        if not feasible(dag, c):
            n_infeasible += 1
            continue
        k = canonical_key(dag, c)
        if k in seen:
            continue
        seen[k] = c
        reps.append(c)
    stats = dict(nominal=len(space),
                 infeasible=n_infeasible,
                 n_classes=len(seen),
                 effective=len(reps))
    return reps, stats


# ===========================================================================
# 2. 代价地形（**人为构造，不是任何真实硬件或编译器的测量**）
# ===========================================================================

# ★ 下面全部是**本仓库自己定的**常数。它们不是任何型号的规格，不在 specs.py 里。
#   命名前缀 MINI_ 是为了让「误当成硬件规格」这件事在阅读时可见（CONTRIBUTING §2）。
MINI_ELEM_BYTES = 4                 # f32
MINI_MAX_VEC_WIDTH = 4              # f32 的 128 位访存上界（本迷你实现的口径）
MINI_MIN_ELEMS_PER_THREAD = 4       # 每线程至少摊到的输出元素数
MINI_MAX_THREADS = 1024
MINI_SMEM_BYTES_PER_SM = 262144     # 每 SM 可用的共享内存字节
MINI_MAX_BLOCKS_PER_SM = 8          # 达到满占用所需的同时驻留块数
MINI_NUM_SMS = 8                    # 迷你 GPU 的 SM 数
MINI_GMEM_BYTES_PER_CYCLE = 64.0
MINI_SMEM_BYTES_PER_CYCLE = 256.0

# C 的物化轮数：C 落 gmem 时走「写 C → 读 C → 写 D」三轮，融合/重算时只写 D。
C_GMEM_ROUNDS = {
    "Skip": 3,
    "AlwaysInline": 1,                 # 不分配存储（重算）
    "MultiLevelTiling": 3,
    "MultiLevelTilingWithFusion": 1,   # 融合：C 不落 gmem
    "AddCacheStage": 1,                # C 物化在 smem
    "ReductionFactorization": 3,
}
SKETCH_FACTOR = {
    "Skip": 1.50,
    "AlwaysInline": 0.95,
    "MultiLevelTiling": 1.00,
    "MultiLevelTilingWithFusion": 0.90,
    "AddCacheStage": 1.05,
    "ReductionFactorization": 1.10,
}


def ceil_div(a, b):
    return (a + b - 1) // b


def num_blocks(dag, cfg):
    tm, tn, _ = effective_tiles(dag, cfg)
    return ceil_div(dag.M, tm) * ceil_div(dag.N, tn)


def smem_bytes(dag, cfg):
    """一个块占用的共享内存字节。Skip / AlwaysInline 不物化到 smem。"""
    if cfg.sketch == "Skip":
        return 0
    if cfg.sketch == "AlwaysInline":
        return 0
    tm, tn, tk = effective_tiles(dag, cfg)
    b = (tm * tk + tk * tn) * MINI_ELEM_BYTES
    if cfg.sketch == "AddCacheStage":
        b += tm * tn * MINI_ELEM_BYTES            # C 物化在 smem
    return b


def gmem_bytes(dag, cfg):
    tm, tn, tk = effective_tiles(dag, cfg)
    M, N, K = dag.M, dag.N, dag.K
    nb = num_blocks(dag, cfg)
    slabs = nb * (tm * K + K * tn) * MINI_ELEM_BYTES     # 每块重读 A/B 的 K 切片
    rounds = C_GMEM_ROUNDS[cfg.sketch]
    total = slabs + rounds * M * N * MINI_ELEM_BYTES
    if cfg.sketch == "ReductionFactorization":
        total += M * N * MINI_ELEM_BYTES                  # 部分和的额外往返
    return total


def smem_traffic_bytes(dag, cfg):
    """经共享内存搬运的字节数（每块每个 k 迭代搬一次操作数切片）。"""
    if smem_bytes(dag, cfg) == 0:
        return 0
    tm, tn, tk = effective_tiles(dag, cfg)
    iters_k = ceil_div(dag.K, tk)
    return num_blocks(dag, cfg) * iters_k * (tm * tk + tk * tn) * MINI_ELEM_BYTES


def parallelism(dag, cfg):
    """并行度 = 占用率 × 网格填充率。两者都**不是**逐语句特征（见 statements_of）。"""
    nb = num_blocks(dag, cfg)
    sb = smem_bytes(dag, cfg)
    blocks_per_sm = MINI_MAX_BLOCKS_PER_SM if sb == 0 else max(1, MINI_SMEM_BYTES_PER_SM // sb)
    occ = min(1.0, blocks_per_sm / float(MINI_MAX_BLOCKS_PER_SM))
    grid_fill = min(1.0, nb / float(MINI_NUM_SMS))
    return max(occ * grid_fill, 1.0 / MINI_NUM_SMS)


def true_cost(dag, cfg):
    """本仓库**人为构造**的代价地形，单位是「周期」。

    ★ 这不是任何真实硬件或编译器的测量。它支持的是「一次代价评估可以怎样被构造、
      以及它的盲点会怎样改变搜索结果」这类**机制**结论。
    返回 inf 表示不可行（相当于编译失败）。
    """
    if not feasible(dag, cfg):
        return INF
    tm, tn, _ = effective_tiles(dag, cfg)
    M, N, K = dag.M, dag.N, dag.K
    threads = min(cfg.threads, tm * tn)
    vec = min(cfg.vec, tn)

    flops = 2.0 * M * N * K + float(M * N)              # matmul + 逐元素消费
    c_compute = flops / (2.0 * threads * vec)
    c_gmem = gmem_bytes(dag, cfg) / MINI_GMEM_BYTES_PER_CYCLE
    c_smem = smem_traffic_bytes(dag, cfg) / MINI_SMEM_BYTES_PER_CYCLE

    pad = ceil_div(M, tm) * tm * ceil_div(N, tn) * tn
    waste = pad / float(M * N)
    return ((c_compute + c_gmem + c_smem) * (1.0 + waste)
            * SKETCH_FACTOR[cfg.sketch] / parallelism(dag, cfg))


def all_costs(dag, space=None):
    """地形上每个（可行）配置的真实代价，给暴力枚举当参照。"""
    if space is None:
        space = effective_space(dag)[0]
    return np.array([true_cost(dag, c) for c in space], dtype=float)


# ===========================================================================
# 3. 逐语句特征与迷你 GBDT（Ansor 代价模型的形状）
# ===========================================================================


def statements_of(dag, cfg):
    """该配置的**最内层非循环语句**列表，每条给出 Ansor 式的两类特征。

    ★ 特征全部是**语句局部**的：算术（flops / 归约行程）、访存（字节数 / 向量宽度）、
      复用比、以及该语句的并行宽度。
      **没有**块数 / 网格规模 / 问题规模——那是 **program 级**属性，不是语句级属性。
      doc02 的 C1 断言的机制就在这里：Ansor 论文列出的特征类别是
      「arithmetic features + memory access features，对每条 innermost non-loop
      statement 抽取」，据此**推断**（本文标注为推断）它同样看不见网格规模。
    """
    tm, tn, tk = effective_tiles(dag, cfg)
    v = min(cfg.vec, tn)
    pw = float(min(cfg.threads, tm * tn))          # 语句的并行宽度（同一时刻的实例数）
    st = [dict(flops=2.0 * tm * tn * tk, bytes_in=(tm * tk + tk * tn) * MINI_ELEM_BYTES,
               vec=float(v), trip=float(tk), parallel_width=pw)]
    if cfg.sketch == "Skip":
        st.append(dict(flops=float(tm * tn), bytes_in=2.0 * tm * tn * MINI_ELEM_BYTES,
                       vec=float(v), trip=1.0, parallel_width=pw))
    elif cfg.sketch == "MultiLevelTiling":
        st.append(dict(flops=float(tm * tn), bytes_in=2.0 * tm * tn * MINI_ELEM_BYTES,
                       vec=float(v), trip=1.0, parallel_width=pw))
    elif cfg.sketch == "AddCacheStage":
        st.append(dict(flops=float(tm * tn), bytes_in=2.0 * tm * tn * MINI_ELEM_BYTES,
                       vec=float(v), trip=1.0, parallel_width=pw))
    elif cfg.sketch == "ReductionFactorization":
        st.append(dict(flops=2.0 * tm * tn * tk, bytes_in=2.0 * tm * tn * MINI_ELEM_BYTES,
                       vec=float(v), trip=1.0, parallel_width=pw))
    return st


FEATURE_NAMES = ("flops", "bytes_in", "vec", "trip", "reuse", "parallel_width")


def feature_vector(stmt):
    b = max(stmt["bytes_in"], 1.0)
    return [stmt["flops"], stmt["bytes_in"], stmt["vec"], stmt["trip"],
            stmt["flops"] / b, stmt["parallel_width"]]


def program_matrix(dag, cfgs):
    """把一批配置摊平成 (语句数, 5) 的特征矩阵 + 每条语句属于哪个程序。"""
    rows, owner = [], []
    for i, c in enumerate(cfgs):
        for s in statements_of(dag, c):
            rows.append(feature_vector(s))
            owner.append(i)
    return np.array(rows, dtype=float), np.array(owner, dtype=int)


class Tree:
    """最小二乘回归树（CART，平方误差）。叶子取**加权**均值。"""

    def __init__(self, max_depth=3, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y, w):
        self.root = self._build(X, y, w, 0)
        return self

    def _leaf(self, y, w):
        sw = w.sum()
        return dict(value=float((y * w).sum() / sw) if sw > 0 else float(y.mean()))

    def _build(self, X, y, w, depth):
        n = len(y)
        if (depth >= self.max_depth or n < 2 * self.min_samples_leaf
                or float(y.std()) < 1e-12):
            return self._leaf(y, w)
        parent = float((w * (y - (y * w).sum() / w.sum()) ** 2).sum())
        best = None
        for j in range(X.shape[1]):
            thr = np.unique(X[:, j])
            if len(thr) < 2:
                continue
            cand = (thr[:-1] + thr[1:]) / 2.0
            if len(cand) > 16:                      # 限候选阈值个数，控制规模
                cand = cand[np.linspace(0, len(cand) - 1, 16).astype(int)]
            for t in cand:
                m = X[:, j] <= t
                if m.sum() < self.min_samples_leaf or (~m).sum() < self.min_samples_leaf:
                    continue
                gain = parent - self._sse(y[m], w[m]) - self._sse(y[~m], w[~m])
                if best is None or gain > best[0]:
                    best = (gain, j, float(t), m)
        if best is None or best[0] <= 1e-12:
            return self._leaf(y, w)
        _, j, t, m = best
        return dict(feature=j, threshold=t,
                    left=self._build(X[m], y[m], w[m], depth + 1),
                    right=self._build(X[~m], y[~m], w[~m], depth + 1))

    @staticmethod
    def _sse(y, w):
        if len(y) == 0:
            return 0.0
        sw = w.sum()
        mu = (y * w).sum() / sw if sw > 0 else y.mean()
        return float((w * (y - mu) ** 2).sum())

    def predict(self, X):
        return np.array([self._walk(self.root, row) for row in X], dtype=float)

    def _walk(self, node, row):
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] \
                else node["right"]
        return node["value"]


class GBDT:
    """迷你梯度提升树。

    ★ 与 Ansor 对齐的两点（其它都是本迷你实现的简化）：
      1. **损失是 weighted squared error，权重直接取吞吐 y**（`fit(..., weight=y)`）；
      2. `fit()` **每次从头重训**——不保留上一轮的树，**不做增量更新**。
         对应论文逐字 "we train a new model every time instead of doing
         incremental updates"（见 specs.ANSOR_NO_INCREMENTAL_UPDATES）。
    """

    def __init__(self, n_estimators=40, learning_rate=0.20, max_depth=3,
                 min_samples_leaf=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.base = 0.0

    def fit(self, X, y, weight=None):
        # ★ 从头重训：清空全部状态，不读上一轮的任何东西
        self.trees = []
        w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
        self.base = float((y * w).sum() / w.sum()) if w.sum() > 0 else float(y.mean())
        pred = np.full(len(y), self.base)
        for _ in range(self.n_estimators):
            t = Tree(self.max_depth, self.min_samples_leaf).fit(X, y - pred, w)
            pred = pred + self.learning_rate * t.predict(X)
            self.trees.append(t)
        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.base)
        for t in self.trees:
            pred = pred + self.learning_rate * t.predict(X)
        return pred


class GBDTModel:
    """把 GBDT 包成「程序级打分器」：整程序得分 = 各语句预测之和。

    ★ 用**吞吐**当目标（越高越好），并**按 DAG 归一化到 [0,1]**
      —— 对应 Ansor「所有 DAG 共用一个模型、同一 DAG 内吞吐归一化」。
    """

    def __init__(self, n_estimators=40, learning_rate=0.20, max_depth=3,
                 min_samples_leaf=4):
        self.gbdt = GBDT(n_estimators, learning_rate, max_depth, min_samples_leaf)

    def fit_from_costs(self, dag, cfgs, throughput):
        X, owner = program_matrix(dag, cfgs)
        y = np.asarray(throughput, dtype=float)
        ys = y[owner]
        self.gbdt.fit(X, ys, weight=ys)
        return self

    def fit_from_dag(self, dag, cfgs=None):
        """在某个 DAG 上「现场调优」得到训练集：吞吐按该 DAG 归一化到 [0,1]。"""
        if cfgs is None:
            cfgs = effective_space(dag)[0]
        cost = np.array([true_cost(dag, c) for c in cfgs])
        ok = np.isfinite(cost)
        cfgs = [c for c, o in zip(cfgs, ok) if o]
        cost = cost[ok]
        thr = 1.0 / cost
        thr = thr / thr.max()
        return self.fit_from_costs(dag, cfgs, thr), cfgs, cost

    def score(self, dag, cfgs):
        X, owner = program_matrix(dag, cfgs)
        per = self.gbdt.predict(X)
        out = np.zeros(len(cfgs))
        np.add.at(out, owner, per)
        return out


class PerfectModel:
    """把地形本身当模型的**对照物**（计划 §0.4 第 5 条要求的「换成正确的」）。

    它不是「更好的模型」，而是「完美模型」——用来检验一条 C 类断言
    究竟在检验机制、还是在检验地形。
    """

    def score(self, dag, cfgs):
        return np.array([-true_cost(dag, c) for c in cfgs])   # 越大越好


# ===========================================================================
# 4. 测量与搜索算法
# ===========================================================================

# ★ 冻结：本迷你地形**关闭测量噪声**。真实调优里 do_bench 的中位数本身有抖动，
#   把它加进来会让「模型误导搜索」与「测量噪声」两件事混在一起，无法归因。
MEASURE_NOISE = 0.0


def measure(dag, cfg, rng=None):
    """一次 measurement trial：返回实测代价（越小越好）。不可行 = 编译失败 = inf。"""
    c = true_cost(dag, cfg)
    if not np.isfinite(c) or MEASURE_NOISE == 0.0:
        return c
    return c * (1.0 + MEASURE_NOISE * float(rng.normal()))


class Result:
    """一次搜索的记账：最好解、best-so-far 轨迹、以及**真的测过**的全部试次。"""

    def __init__(self, name, budget):
        self.name = name
        self.budget = budget
        self.best = INF
        self.best_cfg = None
        self.trace = []
        self.trials = []

    def offer(self, dag, cfg):
        cost = measure(dag, cfg)
        self.trials.append((cfg, cost))
        if cost < self.best:
            self.best, self.best_cfg = cost, cfg
        self.trace.append(self.best)
        return cost

    def __repr__(self):
        return "%s(budget=%d, best=%.1f, cfg=%r)" % (
            self.name, self.budget, self.best, self.best_cfg)


def brute_force(dag, space):
    """A 类的独立参照：**暴力枚举**全部可行配置，取地形上的真实最优。"""
    cost = all_costs(dag, space)
    i = int(np.argmin(cost))
    return space[i], float(cost[i])


def random_search(dag, space, budget, seed=0):
    rng = np.random.default_rng(seed)
    res = Result("random", budget)
    for _ in range(budget):
        res.offer(dag, space[int(rng.integers(len(space)))])
    return res


class ModelGuided:
    """模型引导的搜索：**每轮取模型打分最高的 k 个去测**，测完从头重训模型。

    ★ 这是「利用」端的**退化形态**：不采样、不加探索项。真实 Ansor 的取样器里还有
      随机性与按 sketch 分层的探索；本文把探索项拿掉，是为了让「模型排序错」这件事
      单独可见（有探索项时它会掩盖一部分误导）。带采样池的版本见 `pool=` 参数。
    ★ 训练集 = **源 DAG 上量到的全部数据 + 本 DAG 上已测到的数据**——Ansor
      「所有 DAG 共用一个模型」，所以新 DAG 的起点是别的 DAG 上量出的数据。
      重训是**从头重训**，不做增量更新。
    """

    def __init__(self, base_model=None, base_data=None, pool=None, k=1):
        self.base_model = base_model
        self.base_data = base_data          # (dag, cfgs, thr) 或 None
        self.pool = pool                    # None = 在全空间上取 argmax
        self.k = k

    def run(self, dag, space, budget, seed=0, model=None):
        rng = np.random.default_rng(seed)
        model = model if model is not None else self.base_model
        res = Result("model-guided", budget)
        seen = set()                       # ★ 同一个配置不重复 bench（真实 tuner 都是这样）
        while len(res.trials) < budget:
            pool = space if self.pool is None else \
                [space[int(rng.integers(len(space)))] for _ in range(self.pool)]
            scores = np.asarray(model.score(dag, pool))
            order = np.argsort(-scores, kind="stable")
            picked = 0
            for i in order:
                if len(res.trials) >= budget or picked >= self.k:
                    break
                cfg = pool[int(i)]
                if cfg.as_tuple() in seen:
                    continue
                seen.add(cfg.as_tuple())
                res.offer(dag, cfg)
                picked += 1
            model = self._refit(dag, model, res.trials)
        return res

    def _refit(self, dag, model, trials):
        if isinstance(model, PerfectModel):
            return model
        cfgs, thr = [], []
        if self.base_data is not None:
            bdag, bcfgs, bthr = self.base_data
            cfgs.extend(bcfgs)
            thr.extend(list(bthr))
        for cfg, cost in trials:
            if np.isfinite(cost) and cost > 0:
                cfgs.append(cfg)
                thr.append(1.0 / cost)
        thr = np.array(thr)
        thr = thr / max(thr.max(), 1e-30)     # Ansor：按 DAG 归一化到 [0,1]
        return _refit(model, dag, cfgs, thr)


def _refit(model, dag, cfgs, thr):
    """从头重训（不是增量更新）。PerfectModel 没有可训的参数，原样返回。"""
    if isinstance(model, PerfectModel):
        return model
    return GBDTModel(model.gbdt.n_estimators, model.gbdt.learning_rate,
                     model.gbdt.max_depth, model.gbdt.min_samples_leaf) \
        .fit_from_costs(dag, cfgs, thr)


def evolutionary_search(dag, space, budget, seed=0, pop=8, restarts=4):
    """演化搜索（Ansor 的搜索是演化式的）：在**已测最好个体**附近变异。

    本迷你版的变异算子：动 tile / 线程 / 向量宽度 / 换 sketch——都是空间里真实存在的
    邻域。它**不查模型**，所以是 C1 里「模型引导 vs 不看模型」之外的第三条路。

    ★ `pop` 必须**远小于** `budget`，否则整个预算被初始种群吃光、变异一次都不发生，
      这个算法会**静默退化成随机搜索**（第一版 `pop = budget` 与 `pop = 8 > budget = 6`
      两次都踩到了：实跑里它与 random 的结果逐位相同——见 §5）。
      下面把种群大小夹到 `budget // 3`，使「演化」这件事在任何预算下都真的发生。
    """
    pop = max(2, min(pop, budget // 3))
    rng = np.random.default_rng(seed)
    res = Result("evolutionary", budget)
    popl = [space[int(rng.integers(len(space)))] for _ in range(pop)]
    for c in popl:
        if len(res.trials) >= budget:
            return res
        res.offer(dag, c)
    while len(res.trials) < budget:
        if rng.random() < 1.0 / max(restarts, 1):
            cand = space[int(rng.integers(len(space)))]
        else:
            parent = res.best_cfg
            cand = Config(parent.sketch,
                          _pick(rng, TILE_CHOICES, parent.tm),
                          _pick(rng, TILE_CHOICES, parent.tn),
                          _pick(rng, TK_CHOICES, parent.tk),
                          _pick(rng, THREAD_CHOICES, parent.threads),
                          _pick(rng, VEC_CHOICES, parent.vec))
            if not feasible(dag, cand):
                continue
        res.offer(dag, cand)
    return res


def _pick(rng, choices, current):
    """在当前值附近取一个邻居（一半概率原地不动）。"""
    if rng.random() < 0.5:
        return current
    i = choices.index(current) if current in choices else 0
    j = int(np.clip(i + int(rng.integers(-1, 2)), 0, len(choices) - 1))
    return choices[j]


def regret(dag, cfg, ref_cost):
    """regret = 搜索到的最好真实代价 / 暴力枚举最优真实代价（>= 1）。"""
    if cfg is None:
        return INF
    return true_cost(dag, cfg) / ref_cost


# ===========================================================================
# 5. 迷你 autotuner（照 Triton v3.6.0 的**默认**路径）
# ===========================================================================


class KernelConfig:
    __slots__ = ("name", "num_warps", "num_stages")

    def __init__(self, name, num_warps, num_stages):
        self.name, self.num_warps, self.num_stages = name, num_warps, num_stages

    def key(self):
        return (self.name, self.num_warps, self.num_stages)

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return isinstance(other, KernelConfig) and self.key() == other.key()

    def __repr__(self):
        return "%s(w=%d,s=%d)" % (self.name, self.num_warps, self.num_stages)


class MiniAutotuner:
    """照 Triton v3.6.0 `python/triton/runtime/autotuner.py` 的**默认**路径写的迷你版。

    默认路径的四个要点（见 specs.TRITON_DEFAULTS，全部钉在 tag v3.6.0）：
      1. **没有搜索策略——是穷举**：对（剪枝后的）全部 config 逐个 bench，
         再 `min(timings, key=timings.get)`（比较的是中位数）；
      2. **默认无任何剪枝**：`perf_model=None` / `configs_top_k=1.0` /
         `early_config_prune=None` / `prune_configs_by=None`；
      3. **只有一个 config 时不走调优路径**（`if len(self.configs) > 1:`）；
      4. **磁盘缓存默认关闭**（`cache_results=False`）；带 `pre_hook` 的 config
         **一律不落盘**。

    ★ 本类只复现**计数与分支**，不产出任何时间数字：`bench_fn` 由调用方提供，
      三个分量是 [median, p20, p80]，编译失败返回 `[inf, inf, inf]`。
    """

    def __init__(self, configs, bench_fn, cache_results=False, perf_model=None,
                 configs_top_k=1.0, early_config_prune=None, prune_configs_by=None,
                 pre_hook=None, cache_dir=None, name="mini_kernel"):
        self.configs = list(configs)
        self.bench_fn = bench_fn
        self.cache_results = cache_results
        self.name = name
        self.perf_model = perf_model
        self.configs_top_k = configs_top_k
        self.early_config_prune = early_config_prune
        self.prune_configs_by = prune_configs_by
        self.pre_hook = pre_hook
        self.cache_dir = cache_dir
        self.cache = {}                    # 内存缓存：Dict[Tuple, Config]
        self.calls = 0                     # bench 次数（本仓库的计数器）
        self.pruned_out = 0

    # --- 剪枝：默认什么都不剪 -------------------------------
    def _prune(self):
        confs = list(self.configs)
        if self.prune_configs_by:
            for fn in self.prune_configs_by.values():
                before = len(confs)
                confs = fn(confs)
                self.pruned_out += before - len(confs)
        return confs

    def run(self, key, key_has_dtype=True):
        """返回 (选中的 config, 本次新增的 bench 次数)。

        `key` 照 Triton 的口径：参数名列表对应的取值元组，**带 dtype 的参数会把
        dtype 字符串追加进 key**——本参数只用来说明「key 不同就要重新调优」。
        """
        if key in self.cache:
            return self.cache[key], 0
        if len(self.configs) == 1:                 # ★ v3.6.0：只有一个 config 时不调优
            self.cache[key] = self.configs[0]
            return self.configs[0], 0
        confs = self._prune()
        if len(confs) == 0:
            raise RuntimeError("剪枝后没有 config 可测")
        timings, n = {}, 0
        for c in confs:
            timings[c] = self.bench_fn(c)          # 逐 median/p20/p80 三元组
            n += 1
        self.calls += n
        best = min(timings, key=timings.get)       # ★ 默认选择：中位数最小者
        self.cache[key] = best
        if self.cache_results and self.pre_hook is None:
            self._dump_cache()
        return best, n

    def _dump_cache(self):
        import json
        import os
        if self.cache_dir is None:
            return
        path = os.path.join(self.cache_dir, "%s.autotune.json" % self.name)
        with open(path, "w") as fh:
            json.dump(dict((str(k), str(v)) for k, v in self.cache.items()), fh)
        return path


def compile_failure():
    """★ 编译失败返回的是 **[inf, inf, inf] 三元组**，不是单个 inf。"""
    return [INF, INF, INF]


def timings_from(cost, p20_ratio=1.10, p80_ratio=0.92):
    """把一个真实代价变成 bench 返回的三元组 [median, p20, p80]。"""
    if not np.isfinite(cost):
        return compile_failure()
    return [cost, cost * p20_ratio, cost * p80_ratio]
