"""03 搜索算法与预算：策略、真实 autotuner 的默认行为、以及「搜得多就更好」为什么错。

对应文档：60-compilers/auto-tuning/03-search-algorithms.md

★ 本程序跑的是本仓库的**迷你搜索算法与迷你 autotuner**（`mini_autotune.py` 里照
  Triton v3.6.0 的**默认路径**写的那一个），**不是 Triton、不是 Ansor**，
  也不产生任何时间数字（timings 是代价量纲，不是秒）。
  输出的一切数字都是**本机实测值**，只能支撑关于「搜索策略与预算的权衡」这类
  **机制**结论。真实 Triton / Ansor 的行为只在文档与 `specs.py` 里，逐条带来源。

三类断言：
  A 算法正确性 —— ① **暴力枚举最优**是独立参照：任何搜索找到的 best 都不可能低于它；
                   ② 搜索的记账自洽：返回的 best_cfg 真的被 bench 过、且代价与 best 一致；
                   ③ 迷你 autotuner 选中的 config 与**独立重算**的中位数 argmin 相同
  B 代价模型   —— bench 次数 / 剪枝后配置数 / 各策略在各预算上的 regret，与文档 §5 表核对
  C 边界与反例 —— ① 编译失败必须返回 **[inf, inf, inf] 三元组**并自动出局；
                   ② 磁盘缓存**默认关闭**、带 `pre_hook` 的 config 一律不落盘；
                   ③ **C3：regret 的冻结阈值**——演化搜索不超过它，而被模型带偏的搜索越过它；
                   ④ 把「搜得多就更好」当单调关系是错的：4× 预算花错方向，不如 1× 预算

用法：python3 03_search_trace.py
"""

import os
import tempfile

import numpy as np

import specs as SPECS
from mini_autotune import (INF, Checks, DAG_BIG, DAG_SMALL, GBDTModel, KernelConfig,
                           MINI_SMEM_BYTES_PER_SM, MiniAutotuner, ModelGuided,
                           compile_failure, effective_space, evolutionary_search,
                           brute_force, random_search, regret, rule, timings_from)

# ---------------------------------------------------------------------------
# ★★ 地形与阈值冻结区（计划 §0.4 第 5 条）
# ---------------------------------------------------------------------------
SEED = 20260916
BUDGETS = (6, 12, 24, 48)          # 预算扫描点
SEARCH_BUDGET = 48                 # 主轨迹用的预算
SEEDS = (0, 1, 2)                  # 冻结的种子
MG_BUDGET = 12                     # 模型引导搜索的预算（它每轮要重训模型，代价高）
N_EST, LR, DEPTH, MIN_LEAF = 40, 0.25, 3, 6

# ★★ C3：**冻结阈值**（计划 §3 给的两个可证伪写法里的 **(a) 冻结阈值**，本程序选的就是它）
# 取值来源（2026-09-16 的实跑）：演化搜索在冻结种子 0/1/2、预算 48、小图上的 regret 为
# 1.00 / 1.33 / 1.11，**最差 1.33**；阈值取 1.40（比最差值高约 5%）。
# **任何把搜索改坏**的改动都会越过它——下面的反例（模型引导的搜索）就越过了。
# REGRET 的口径：搜索到的最好真实代价 / 暴力枚举最优。**不引用任何文献给出的界**——
# 「搜多少配置后收益饱和」无定论（见 specs.py 的 UNSOURCED 第一条）。
REGRET_MAX = 1.40

# 迷你 autotuner 的配置表：num_warps × num_stages（照 Triton 的形参口径）
TRITON_WARPS = (2, 4, 8, 16)
TRITON_STAGES = (2, 3, 4)

# 文档 §5 的计数表（首次实跑值）
TRITON_TABLE = {
    "n_configs": 12,
    "benches_default": 12,
    "benches_single": 0,
    "n_compile_fail": 5,
}


def triton_bench(cfg):
    """迷你 bench：把 (num_warps, num_stages) 映射成代价三元组。

    资源不够时走**编译失败**这条路（返回 `[inf, inf, inf]`，等价自动出局）——
    对应真实 Triton 里 OutOfResources / CompileTimeAssertionFailure / PTXASError 的待遇。
    """
    threads = cfg.num_warps * 32
    if threads > 1024:
        return compile_failure()
    smem = cfg.num_stages * threads * 512
    if smem > MINI_SMEM_BYTES_PER_SM:
        return compile_failure()
    base = 2.0e7 / (threads * 4.0)
    cost = base * (1.0 + 1.0 / cfg.num_stages) + 400.0 * (16.0 / cfg.num_warps)
    return timings_from(cost)


def reference_selection(configs):
    """独立参照：不看 autotuner 的选择，自己按 bench 结果重算中位数最小者。"""
    best, best_t = None, INF
    for c in configs:
        t = triton_bench(c)[0]
        if t < best_t:
            best, best_t = c, t
    return best, best_t


def main():
    checks = Checks()

    # ------------------------------------------------------------------
    rule("1. 四类搜索策略与它们各自的前提")
    print("  · 穷举（Triton 的默认）：不需要模型、不需要先验；代价 = 空间基数 × 一次 bench。")
    print("  · 随机：同上的前提，但只花预算那么多；没有任何「学到了什么」的保证。")
    print("  · 模型引导：前提是一个**在新图上仍然排得对**的模型；否则利用得越狠越糟。")
    print("  · 演化：前提是一个能产生「邻近好解」的变异算子；不依赖模型。")
    print("  Ansor 论文把搜索空间分成两层（sketch + annotation），搜索在**注释层**做——")
    print("  逐字：%s" % SPECS.ANSOR_HIERARCHICAL_SPACE[:78] + " ...")
    print("  来源（一手，见 specs.py）：%s" % SPECS.ANSOR_PDF)
    print("  ★ Triton 侧的前提完全不同（见 §2）：它**默认不搜索**，是穷举。")

    # ------------------------------------------------------------------
    rule("2. 迷你 autotuner：Triton v3.6.0 的**默认**路径")
    confs = [KernelConfig("mini_kernel", w, s)
             for w in TRITON_WARPS for s in TRITON_STAGES]
    print("  钉住的版本：%s（最新 %s）。默认值：%s"
          % (SPECS.TRITON_TAG, SPECS.TRITON_TAG_LATEST, SPECS.TRITON_DEFAULTS))
    print("  默认选择式：`min(timings, key=timings.get)` —— **穷举后取中位数最小者**")
    print("  单 config 守卫：`%s`" % SPECS.TRITON_SINGLE_CONFIG_GUARD)
    print("  编译失败返回值：%s" % SPECS.TRITON_COMPILE_FAILURE_RETURN)
    print()

    at = MiniAutotuner(confs, triton_bench)
    picked, n_bench = at.run(key=("x", "f32"))
    ref_cfg, ref_t = reference_selection(confs)
    print("  配置 %d 个；一次 run 的 bench 次数 = %d" % (len(confs), n_bench))
    print("  autotuner 选中：%r；独立重算的中位数 argmin：%r" % (picked, ref_cfg))
    fail = [c for c in confs if not np.isfinite(triton_bench(c)[0])]
    print("  其中编译失败的 config：%d 个（%s）"
          % (len(fail), [str(c) for c in fail]))
    checks.A(picked == ref_cfg,
             "③ 选中的 config 与**独立重算**的中位数最小者一致",
             "%r == %r" % (picked, ref_cfg))
    checks.B(n_bench == TRITON_TABLE["benches_default"],
             "默认无剪枝 → bench 次数 == 配置数（%d）" % TRITON_TABLE["benches_default"],
             "本机 %d" % n_bench)
    checks.B(len(confs) == TRITON_TABLE["n_configs"],
             "配置数 == 文档表 %d" % TRITON_TABLE["n_configs"], "本机 %d" % len(confs))
    checks.B(len(fail) == TRITON_TABLE["n_compile_fail"],
             "编译失败的 config 数 == 文档表 %d" % TRITON_TABLE["n_compile_fail"],
             "本机 %d" % len(fail))

    # --- 只有一个 config 时不走调优路径 ---
    at1 = MiniAutotuner([confs[0]], triton_bench)
    one, n1 = at1.run(key=("x", "f32"))
    checks.B(n1 == TRITON_TABLE["benches_single"],
             "★ 只有一个 config 时**不调优**（bench 次数 %d）——`if len(self.configs) > 1:` 的后果"
             % TRITON_TABLE["benches_single"], "本机 %d 次 bench" % n1)
    checks.A(one == confs[0], "单 config 时原样返回那个 config", str(one))

    # --- 剪枝是显式开启的 ---
    atp = MiniAutotuner(confs, triton_bench,
                        prune_configs_by={"early_config_prune":
                                          lambda cs: [c for c in cs if c.num_stages <= 3]})
    pk, np_bench = atp.run(key=("x", "f32"))
    print("  显式开启剪枝（num_stages <= 3）后：配置 %d → %d，bench 次数 %d"
          % (len(confs), len(confs) - atp.pruned_out, np_bench))
    checks.A(np_bench == len(confs) - atp.pruned_out and np_bench < n_bench,
             "显式剪枝真的少 bench（%d < %d）——所以「默认无剪枝」不是空话"
             % (np_bench, n_bench), "剪掉 %d 个" % atp.pruned_out)

    # --- 内存缓存 vs 磁盘缓存 ---
    at2 = MiniAutotuner(confs, triton_bench)
    at2.run(key=("x", "f32"))
    _, again = at2.run(key=("x", "f32"))
    _, other = at2.run(key=("y", "f32"))
    checks.A(again == 0 and other == n_bench,
             "内存缓存命中 → 0 次 bench；key 换了 → 重新调优（%d 次）" % n_bench,
             "same-key %d / new-key %d" % (again, other))
    print("  key 口径（specs.TRITON_KEY_NOTE）：%s" % SPECS.TRITON_KEY_NOTE[:60] + " ...")

    tmp = tempfile.mkdtemp(prefix="miniautotune_")
    off_file = os.path.join(tmp, "%s.autotune.json" % "mini_kernel")
    at3 = MiniAutotuner(confs, triton_bench, cache_dir=tmp, cache_results=False)
    at3.run(key=("x", "f32"))
    existed_off = os.path.exists(off_file)
    at4 = MiniAutotuner(confs, triton_bench, cache_dir=tmp, cache_results=True)
    at4.run(key=("x", "f32"))
    existed_on = os.path.exists(off_file)
    os.remove(off_file)
    at5 = MiniAutotuner(confs, triton_bench, cache_dir=tmp, cache_results=True,
                        pre_hook=lambda *a, **k: None)
    at5.run(key=("x", "f32"))
    existed_hook = os.path.exists(off_file)
    print("  落盘名 `%s`；cache_results=False → 文件存在？%s；True → %s；"
          "带 pre_hook → %s" % (SPECS.TRITON_DISK_CACHE_FILE, existed_off, existed_on,
                                existed_hook))
    checks.C(not existed_off and existed_on,
             "★ **磁盘缓存默认关闭**（`cache_results=False` → 不落盘），"
             "显式打开才落盘——不得写「Triton 会把 autotune 结果缓存到磁盘」",
             "off=%s on=%s" % (existed_off, existed_on))
    checks.C(existed_on and not existed_hook,
             "★ 带 `pre_hook` 的 config **一律不落盘**（specs.TRITON_DISK_CACHE_NOTE）",
             "有 pre_hook 时文件存在？%s" % existed_hook)

    # --- 编译失败的三元组 ---
    bad = fail[0]
    tb = triton_bench(bad)
    print("  一个编译失败的 config %r → bench 返回 %s（%d 个分量）"
          % (bad, tb, len(tb)))
    checks.C(len(tb) == 3 and all(v == INF for v in tb),
             "★ 编译失败返回 **[inf, inf, inf] 三元组**（不是单个 inf），"
             "且正因为它不是有限数，`min(timings, key=timings.get)` 会把它跳过",
             "返回 %s" % tb)
    checks.C(tb[0] > max(triton_bench(c)[0] for c in confs if np.isfinite(triton_bench(c)[0])),
             "★ 出局是真的：编译失败的代价大于**任何**能编译的 config，"
             "所以选中的那个一定不是它",
             "选中的是 %r（不在失败名单 %s 里）" % (picked, [str(c) for c in fail]))

    # ------------------------------------------------------------------
    rule("3. 预算扫描：regret 与 best-so-far 轨迹")
    small_reps, small_st = effective_space(DAG_SMALL)
    bf_cfg, bf_cost = brute_force(DAG_SMALL, small_reps)
    print("  小图有效空间 %d 个配置；**暴力枚举最优** %.0f（%r）——这是 A 类的独立参照"
          % (small_st["effective"], bf_cost, bf_cfg))
    print()

    # ① A：任何搜索的 best 都不可能低于暴力枚举最优；且记账自洽
    res_rand = random_search(DAG_SMALL, small_reps, SEARCH_BUDGET, seed=SEEDS[0])
    res_evo = evolutionary_search(DAG_SMALL, small_reps, SEARCH_BUDGET, seed=SEEDS[0])
    for res in (res_rand, res_evo):
        checks.A(res.best >= bf_cost,
                 "① %s 的 best（%.0f）≥ 暴力枚举最优（%.0f）——没有任何搜索能赢过全枚举"
                 % (res.name, res.best, bf_cost), "")
        measured = dict((c.as_tuple(), v) for c, v in res.trials)
        checks.A(res.best_cfg.as_tuple() in measured
                 and abs(measured[res.best_cfg.as_tuple()] - res.best) < 1e-9,
                 "② %s 的记账自洽：返回的 best_cfg 真的被 bench 过，且代价 == best"
                 % res.name, "best=%.1f" % res.best)
        checks.A(res.trace[-1] == res.best and len(res.trace) == len(res.trials),
                 "② %s 的 best-so-far 轨迹末尾 == best，长度 == 试次数" % res.name,
                 "%d 个点" % len(res.trace))

    # ② B：预算扫描（三种策略 × 四个预算，取 3 个种子的**最差** regret）
    print("%-10s %s" % ("预算", "  ".join("%-22s" % n for n in
                                         ("随机", "演化", "模型引导(仅 12)"))))
    print("-" * 78)
    table = {}
    for bud in BUDGETS:
        cells = []
        for name, fn in (("random", random_search), ("evolutionary", evolutionary_search)):
            worst = max(regret(DAG_SMALL, fn(DAG_SMALL, small_reps, bud, seed=s).best_cfg,
                               bf_cost) for s in SEEDS)
            table[(name, bud)] = worst
            cells.append("%-22s" % ("%.2f" % worst))
        if bud == MG_BUDGET:
            mg = _model_guided(small_reps, bud)
            table[("model-guided", bud)] = regret(DAG_SMALL, mg.best_cfg, bf_cost)
            cells.append("%-22s" % ("%.2f" % table[("model-guided", bud)]))
        else:
            cells.append("%-22s" % "—")
        print("%-10d %s" % (bud, "  ".join(cells)))
    for bud in BUDGETS:
        checks.B(abs(table[("random", bud)] - DOC_TABLE[("random", bud)]) < 0.005,
                 "文档表：随机 预算 %d 的最差 regret == %.2f"
                 % (bud, DOC_TABLE[("random", bud)]),
                 "本机 %.2f" % table[("random", bud)])
        checks.B(abs(table[("evolutionary", bud)] - DOC_TABLE[("evolutionary", bud)]) < 0.005,
                 "文档表：演化 预算 %d 的最差 regret == %.2f"
                 % (bud, DOC_TABLE[("evolutionary", bud)]),
                 "本机 %.2f" % table[("evolutionary", bud)])

    # ③ best-so-far 轨迹（预算 48，种子 0）
    print()
    print("  best-so-far 轨迹（预算 %d，种子 %d，每 8 个试次取一点）："
          % (SEARCH_BUDGET, SEEDS[0]))
    for res in (res_rand, res_evo):
        pts = [res.trace[i] for i in range(7, len(res.trace), 8)]
        print("    %-14s %s" % (res.name, " ".join("%.0f" % v for v in pts)))
    checks.B(repr([round(res_rand.trace[i]) for i in (7, 15, 23, 31, 39, 47)])
             == repr(DOC_TRACE_RANDOM),
             "文档表：随机搜索的 best-so-far 轨迹（8/16/24/32/40/48 试次）",
             "%s" % [round(res_rand.trace[i]) for i in (7, 15, 23, 31, 39, 47)])
    checks.B(repr([round(res_evo.trace[i]) for i in (7, 15, 23, 31, 39, 47)])
             == repr(DOC_TRACE_EVO),
             "文档表：演化搜索的 best-so-far 轨迹（8/16/24/32/40/48 试次）",
             "%s" % [round(res_evo.trace[i]) for i in (7, 15, 23, 31, 39, 47)])

    # ④ C3：冻结阈值（选项 a）——本程序选的就是 (a)
    print()
    print("  ★ C3 判据选的是计划 §3 给的两个写法里的 **(a) 冻结阈值**：")
    print("    REGRET_MAX = %.2f（来源：种子 0/1/2 在预算 %d 上演化搜索的 regret 为"
          " 1.00 / 1.33 / 1.11，最差 1.33）" % (REGRET_MAX, SEARCH_BUDGET))
    worst_evo = max(regret(DAG_SMALL, evolutionary_search(DAG_SMALL, small_reps,
                                                         SEARCH_BUDGET, seed=s).best_cfg,
                           bf_cost) for s in SEEDS)
    print("    本次实跑：演化搜索在三个种子上的**最差** regret = %.2f" % worst_evo)
    checks.C(worst_evo < REGRET_MAX,
             "★ C3：演化搜索的最差 regret %.2f **严格小于**冻结阈值 %.2f"
             % (worst_evo, REGRET_MAX), "%.2f < %.2f" % (worst_evo, REGRET_MAX))
    mg12 = _model_guided(small_reps, MG_BUDGET)
    rg = regret(DAG_SMALL, mg12.best_cfg, bf_cost)
    print("    反例：模型引导的搜索（预算 %d）regret = %.2f" % (MG_BUDGET, rg))
    checks.C(rg > REGRET_MAX,
             "★ C3 的可证伪性：被代价模型带偏的搜索 regret %.2f **越过**同一个阈值"
             "（%.2f）——所以上面那条阈值断言不是空的，它真的能分辨「搜得对」与"
             "「搜得多但方向错」" % (rg, REGRET_MAX),
             "%.2f > %.2f" % (rg, REGRET_MAX))

    # ⑤ 「搜得多就更好」不是单调关系：同一个策略翻倍预算，收益取决于方向
    print()
    mg24 = _model_guided(small_reps, 2 * MG_BUDGET, seed=SEEDS[0])
    r_mg12 = regret(DAG_SMALL, mg12.best_cfg, bf_cost)
    r_mg24 = regret(DAG_SMALL, mg24.best_cfg, bf_cost)
    r_ev12 = regret(DAG_SMALL, evolutionary_search(DAG_SMALL, small_reps,
                                                   MG_BUDGET, seed=SEEDS[0]).best_cfg, bf_cost)
    r_ev24 = regret(DAG_SMALL, evolutionary_search(DAG_SMALL, small_reps,
                                                   2 * MG_BUDGET, seed=SEEDS[0]).best_cfg, bf_cost)
    imp_mg = (r_mg12 - r_mg24) / r_mg12
    imp_ev = (r_ev12 - r_ev24) / r_ev12
    print("    预算 %2d → %2d（种子 %d）：模型引导 %.2f → %.2f（改善 %.0f%%）；"
          "演化 %.2f → %.2f（改善 %.0f%%）"
          % (MG_BUDGET, 2 * MG_BUDGET, SEEDS[0], r_mg12, r_mg24, 100 * imp_mg,
             r_ev12, r_ev24, 100 * imp_ev))
    checks.C(imp_mg < 0.01 and imp_ev > 0.05,
             "★ 把「搜得多就更好」当单调关系是错的：**把预算翻倍**，"
             "模型引导的搜索 regret 一点没改善（%.2f → %.2f，改善 %.0f%%），"
             "而同样翻倍的演化搜索改善了 %.0f%%——"
             "**方向错了，预算买不到东西**"
             % (r_mg12, r_mg24, 100 * imp_mg, 100 * imp_ev),
             "模型引导 %.0f%% vs 演化 %.0f%% 的改善" % (100 * imp_mg, 100 * imp_ev))

    rule("4. 断言汇总")
    checks.summary()


def _model_guided(space, budget, seed=SEEDS[0]):
    """模型引导的搜索：模型在**大图**上训练，拿到小图上来用（共享模型的用法）。"""
    big_reps, _ = effective_space(DAG_BIG)
    rng = np.random.default_rng(SEED)
    src = [big_reps[i] for i in rng.permutation(len(big_reps))[:1200]]
    model, _, cost = GBDTModel(N_EST, LR, DEPTH, MIN_LEAF).fit_from_dag(DAG_BIG, src)
    thr = 1.0 / cost
    thr = thr / thr.max()
    return ModelGuided(base_model=model, base_data=(DAG_BIG, src, thr),
                       pool=256).run(DAG_SMALL, space, budget, seed=seed)


# 文档表：首次实跑值（2026-09-16，三个冻结种子的**最差** regret；轨迹取种子 0）
DOC_TABLE = {
    ("random", 6): 2.65, ("random", 12): 2.08, ("random", 24): 2.08, ("random", 48): 1.65,
    ("evolutionary", 6): 3.83, ("evolutionary", 12): 2.09,
    ("evolutionary", 24): 1.80, ("evolutionary", 48): 1.33,
}
DOC_TRACE_RANDOM = [167765, 98927, 98927, 98927, 75784, 75784]
DOC_TRACE_EVO = [167765, 86659, 71899, 64519, 64519, 64519]


if __name__ == "__main__":
    main()
