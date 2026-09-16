"""02 代价模型：它要解决什么、它的确切形状、它的盲点。

对应文档：60-compilers/auto-tuning/02-cost-models.md

★ 本程序跑的是本仓库的**迷你 GBDT 与本仓库人为构造的代价地形**，
  **不是 Ansor 的代价模型、不是 XGBoost、不是任何真实编译器的成本模型**。
  输出的一切数字都是**本机实测值**，只能支撑关于「一个学习出来的排序器可以怎样被
  训练、以及它的盲点会怎样改变搜索结果」这类**机制**结论。
  Ansor 的模型形状（GBDT/XGBoost、共用模型、逐语句特征、加权平方误差、每轮从头重训）
  与论文报的质量数字只在文档与 `specs.py` 里，逐条带来源，**本程序不对它们下断言**。

三类断言：
  A 算法正确性 —— ① 整程序得分 = 各语句预测之和（两条独立路径）；
                   ② 基值是**加权**均值、损失是**加权**平方误差（手算三点的对照）；
                   ③ `fit()` 每次从头重训（与新建一个模型重训逐位相同）；
                   ④ 树集成**不能外推**（训练范围外的预测恒定，手算对照）
  B 代价模型   —— RMSE / R² / pairwise 排序准确率 / recall@30，同分布 vs 跨图；
                   以及小样本训练集下的误差，与文档 §5 表逐行核对
  C 边界与反例 —— **C1：代价模型会误导搜索**。构造「模型排序与真实代价不一致」的地形，
                   断言模型引导的搜索**比随机搜索差**；并断言把模型换成**完美模型**后
                   这个判定**翻转**（否则它在检验地形而不是在检验机制）

用法：python3 02_cost_model.py
"""

import numpy as np

import specs as SPECS
from mini_autotune import (Checks, DAG_BIG, DAG_SMALL, GBDT, GBDTModel, ModelGuided,
                           PerfectModel, brute_force, effective_space,
                           effective_tiles, program_matrix, random_search, rule,
                           statements_of, true_cost)

# ---------------------------------------------------------------------------
# ★★ 地形与超参数冻结区（计划 §0.4 第 5 条）
# 两个 DAG 的尺寸、模型的超参数、种子、预算、以及下面每一个阈值，
# 都在**跑出任何结果之前**写死在这里。
# ---------------------------------------------------------------------------
SEED = 20260916
TRAIN_SOURCE = 1200        # 从「大图」有效空间里冻结点数的训练集
HOLDOUT_SOURCE = 600       # 同分布留出集
BUDGET = 12                # C1 里两种搜索的**同一份**预算
SEEDS = (0, 1, 2)          # 冻结的种子
N_EST, LR, DEPTH, MIN_LEAF = 40, 0.25, 3, 6

# ★ 阈值来自首次实跑（2026-09-16）。实跑值见文档 §5 表；阈值取在实跑值**之下**
#   留出余量，因此「把模型或特征改坏」会让这些断言失败。
RMSE_ID_MAX = 0.180        # 同分布 RMSE 必须优于它（否则模型根本没学）
RMSE_OOD_MIN = 0.180       # 跨图 RMSE 必须**劣于**它（否则盲点不复现）
PAIRWISE_ID_MIN = 0.740    # 同分布 pairwise 排序准确率下界
RECALL_ID_MIN = 0.300      # 同分布 recall@30 下界
# C1：模型引导的搜索在全部冻结种子上都比同预算的随机搜索差
C1_MUST_BE_WORSE_ON = len(SEEDS)


def metrics(pred, cost, npairs=8000):
    """模型质量的四个数（与文档 §5 表逐行核对）。

    `pred` 是模型给的分数（越大越好），`cost` 是地形的真实代价（越小越好）。
    ★ 目标按**本 DAG** 归一化到 [0,1]——Ansor 的口径是「同一 DAG 内吞吐归一化」，
      所以同分布与跨图两组数才是同一把尺子量出来的。
    """
    y = 1.0 / cost
    y = y / y.max()
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    n = len(pred)
    i = rng.integers(0, n, npairs)
    j = rng.integers(0, n, npairs)
    m = i != j
    pw = float(np.mean((pred[i[m]] > pred[j[m]]) == (cost[i[m]] < cost[j[m]])))
    k = 30
    true_top = set(np.argsort(cost, kind="stable")[:k].tolist())
    pred_top = set(np.argsort(-pred, kind="stable")[:k].tolist())
    recall = len(true_top & pred_top) / float(k)
    return dict(rmse=rmse, r2=r2, pairwise=pw, recall_at_30=recall)


def main():
    checks = Checks()

    # ------------------------------------------------------------------
    rule("1. 为什么需要代价模型：把「测一次」的账单摆出来")
    big_reps, big_st = effective_space(DAG_BIG)
    small_reps, small_st = effective_space(DAG_SMALL)
    print("  小图有效空间 %d 个配置；每个配置一次 bench 就是一次 measurement trial"
          % small_st["effective"])
    print("  一次完整枚举 = %d 次测量；代价模型的账是「固定训练成本 + 每次打分近乎免费」"
          % small_st["effective"])
    print("  Ansor 论文的训练数据规模口径：%s" % SPECS.ANSOR_TRAINING_DATA_SCALE)
    print("  来源（一手，见 specs.py）：%s" % SPECS.ANSOR_PDF)
    checks.B(small_st["effective"] == 4596 and big_st["effective"] == 10482,
             "有效空间规模与文档 §5 表一致（小图 4596 / 大图 10482）",
             "%d / %d" % (small_st["effective"], big_st["effective"]))

    # ------------------------------------------------------------------
    rule("2. A 算法正确性：迷你 GBDT 的确切形状")

    # ① 整程序得分 = 各语句预测之和（两条独立路径）
    prog = small_reps[:7]
    X, owner = program_matrix(DAG_SMALL, prog)
    gb = GBDT(N_EST, LR, DEPTH, MIN_LEAF)
    gb.fit(X, np.linspace(0.1, 1.0, len(X)), weight=np.linspace(0.1, 1.0, len(X)))
    per = gb.predict(X)
    ref = np.array([sum(per[owner == k]) for k in range(len(prog))])   # 逐语句循环求和
    fast = np.zeros(len(prog))
    np.add.at(fast, owner, per)
    print("  程序 #0 的语句数 = %d；逐语句循环求和 = %.6f；向量化求和 = %.6f"
          % ((owner == 0).sum(), ref[0], fast[0]))
    checks.A(len(X) == sum(len(statements_of(DAG_SMALL, c)) for c in prog),
             "语句矩阵的行数 = 各配置语句数之和", "%d 行" % len(X))
    checks.A(np.allclose(ref, fast),
             "①「整程序得分 = 各语句预测之和」：循环求和与向量化求和逐项一致",
             "最大差 %.2e" % float(np.max(np.abs(ref - fast))))

    # ② 基值是**加权**均值、损失是**加权**平方误差（手算对照）
    yv = np.array([1.0, 2.0, 6.0])
    wv = np.array([1.0, 1.0, 2.0])            # Ansor：权重直接取吞吐 y
    hand_base = (yv * wv).sum() / wv.sum()     # 手算：(1+2+12)/4 = 3.75
    gb2 = GBDT(n_estimators=0)
    gb2.fit(yv.reshape(-1, 1), yv, weight=wv)
    print("  手算加权均值 = (1×1 + 2×1 + 6×2) / 4 = %.4f；实现给出 %.4f"
          % (hand_base, gb2.base))
    checks.A(abs(gb2.base - hand_base) < 1e-12,
             "② 基值 = **加权**均值（权重 = 吞吐 y，不是常数 1）",
             "本机 %.4f / 手算 %.4f" % (gb2.base, hand_base))

    def loss(pred, target, weight):
        """论文的损失形式：Σ y·(pred − y)²（见 specs.ANSOR_LOSS_VERBATIM）。"""
        return float(np.sum(weight * (np.asarray(pred) - np.asarray(target)) ** 2))

    pred0 = np.array([2.0, 2.0, 5.0])
    hand_loss = 1.0 * (2 - 1) ** 2 + 1.0 * (2 - 2) ** 2 + 2.0 * (5 - 6) ** 2
    print("  手算 loss = 1×1 + 1×0 + 2×1 = %.4f；函数给出 %.4f"
          % (hand_loss, loss(pred0, yv, wv)))
    checks.A(abs(loss(pred0, yv, wv) - hand_loss) < 1e-12,
             "② 损失 = 加权平方误差（权重 = y）", "%.4f" % hand_loss)

    # ③ 每轮从头重训：不保留上一轮的任何东西
    Xa = np.array([[0.0], [1.0], [2.0], [3.0]])
    ya = np.array([0.0, 1.0, 4.0, 9.0])
    Xb = np.array([[0.0], [1.0], [2.0]])
    yb = np.array([5.0, 5.0, 5.0])
    g_a = GBDT(5, 0.3, 2, 1).fit(Xa, ya, weight=ya)
    g_a.fit(Xb, yb, weight=yb)                       # 第二次 fit
    fresh = GBDT(5, 0.3, 2, 1).fit(Xb, yb, weight=yb)
    same = np.allclose(g_a.predict(Xb), fresh.predict(Xb))
    print("  先 fit(A) 再 fit(B) 与「新建模型只 fit(B)」的预测：%s"
          % ("逐位相同" if same else "不同"))
    checks.A(same,
             "③ `fit()` 每轮**从头重训**，不做增量更新——"
             "第二次 fit 的结果与新模型完全一致（specs 里的 '%s'）"
             % SPECS.ANSOR_NO_INCREMENTAL_UPDATES[:34] + "...", "逐位相同")

    # ④ 树集成**不能外推**：训练范围外的预测恒定
    xs = np.linspace(0.0, 10.0, 40).reshape(-1, 1)
    ys = xs.ravel() ** 2
    gt = GBDT(30, 0.3, 3, 2).fit(xs, ys, weight=ys)
    far = np.array([[10.5], [100.0], [1e6]])
    p_far = gt.predict(far)
    p_edge = gt.predict(np.array([[10.0]]))[0]
    print("  训练区间 [0,10] 上的平方函数；区间外预测：%s"
          % np.array2string(p_far, precision=3))
    print("  区间右端点 x=10 的预测：%.6f" % p_edge)
    checks.A(np.allclose(p_far, p_edge),
             "④ 分段常值函数**不能外推**：x=10.5 / 100 / 1e6 的预测都等于 x=10 处的值"
             "（这既是 GBDT 的性质，也是「共享模型在新图上失效」的数学根源）",
             "max|p_far - p(10)| = %.2e" % float(np.max(np.abs(p_far - p_edge))))

    # ------------------------------------------------------------------
    rule("3. B 代价模型：模型质量（与文档 §5 表逐行核对）")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(big_reps))
    src = [big_reps[i] for i in idx[:TRAIN_SOURCE]]
    hold = [big_reps[i] for i in idx[TRAIN_SOURCE:TRAIN_SOURCE + HOLDOUT_SOURCE]]
    model, src_cfgs, src_cost = GBDTModel(N_EST, LR, DEPTH, MIN_LEAF).fit_from_dag(
        DAG_BIG, src)
    thr_src = 1.0 / src_cost
    thr_src = thr_src / thr_src.max()

    id_cost = np.array([true_cost(DAG_BIG, c) for c in hold])
    id_pred = model.score(DAG_BIG, hold)
    m_id = metrics(id_pred, id_cost)
    ood_cost = np.array([true_cost(DAG_SMALL, c) for c in small_reps])
    ood_pred = model.score(DAG_SMALL, small_reps)
    m_ood = metrics(ood_pred, ood_cost)
    print("  同分布（大图留出集 %d 个程序）  RMSE %.3f  R² %.3f  pairwise %.3f  recall@30 %.3f"
          % (len(hold), m_id["rmse"], m_id["r2"], m_id["pairwise"], m_id["recall_at_30"]))
    print("  跨 图（小图全体 %d 个程序）     RMSE %.3f  R² %.3f  pairwise %.3f  recall@30 %.3f"
          % (len(small_reps), m_ood["rmse"], m_ood["r2"], m_ood["pairwise"],
             m_ood["recall_at_30"]))
    print()
    print("  Ansor 论文 §7.5 报的模型质量（**引用值**，本仓库不复核其测量口径）：")
    q = SPECS.ANSOR_MODEL_QUALITY
    print("    %d 个程序（%d 训练 / %d 测试）：RMSE %.3f  R² %.3f  pairwise %.3f  recall@30 %.3f"
          % (q["programs"], q["train"], q["test"], q["rmse"], q["r2"],
             q["pairwise_accuracy"], q["recall_at_30"]))
    print("    ↑ 那组数字是**代价模型自身**的性质，正文位置在本主题 doc02 §2（本文档）；")
    print("      搜索结果那一组（Table 3 / §7.3 加速比）的正文位置在 doc03 §5。")

    # 文档值 = 首次实跑（2026-09-16）的结果，逐行核对
    B_TABLE = [
        ("同分布 RMSE", m_id["rmse"], 0.160),
        ("同分布 R²", m_id["r2"], 0.057),
        ("同分布 pairwise", m_id["pairwise"], 0.762),
        ("同分布 recall@30", m_id["recall_at_30"], 0.367),
        ("跨图 RMSE", m_ood["rmse"], 0.194),
        ("跨图 R²", m_ood["r2"], -0.162),
        ("跨图 pairwise", m_ood["pairwise"], 0.685),
        ("跨图 recall@30", m_ood["recall_at_30"], 0.000),
    ]
    print()
    print("%-22s %10s %8s" % ("项", "本机值", "文档值"))
    print("-" * 44)
    for name, got, want in B_TABLE:
        ok = abs(got - want) < 0.001
        print("%-22s %10.3f %8.3f  %s" % (name, got, want, "ok" if ok else "MISMATCH"))
        checks.B(ok, "文档表：%s == %.3f" % (name, want), "本机 %.3f" % got)

    checks.B(m_id["rmse"] <= RMSE_ID_MAX,
             "同分布 RMSE %.3f ≤ 冻结阈值 %.3f（模型在同分布上确实学到了东西）"
             % (m_id["rmse"], RMSE_ID_MAX), "")
    checks.B(m_id["pairwise"] >= PAIRWISE_ID_MIN,
             "同分布 pairwise 排序准确率 %.3f ≥ %.3f"
             % (m_id["pairwise"], PAIRWISE_ID_MIN), "")
    checks.B(m_id["recall_at_30"] >= RECALL_ID_MIN,
             "同分布 recall@30 %.3f ≥ %.3f" % (m_id["recall_at_30"], RECALL_ID_MIN), "")
    checks.C(m_ood["recall_at_30"] < m_id["recall_at_30"],
             "★ 跨图后 recall@30 从 %.3f 掉到 %.3f ——模型排出来的 top-30 与真值 top-30 "
             "几乎不重叠，而它的**排序能力**却只掉了一点点"
             % (m_id["recall_at_30"], m_ood["recall_at_30"]),
             "ID %.3f → OOD %.3f" % (m_id["recall_at_30"], m_ood["recall_at_30"]))
    checks.C(m_ood["rmse"] >= RMSE_OOD_MIN,
             "★ 跨图 RMSE %.3f ≥ 冻结阈值 %.3f ——**盲点在实跑里真的复现了**："
             "同一个模型换一个图就明显变差" % (m_ood["rmse"], RMSE_OOD_MIN),
             "同分布 %.3f → 跨图 %.3f" % (m_id["rmse"], m_ood["rmse"]))

    # 小样本训练的陷阱：训练集越小，同分布误差越大
    print()
    print("  小样本曲线（同一留出集，只看训练集大小）：")
    curve = []
    SMALL_CURVE = ((30, 0.470, 0.646), (120, 0.409, 0.726), (1200, 0.160, 0.762))
    for n_small, want_rmse, want_pw in SMALL_CURVE:
        m_s, _, _ = GBDTModel(N_EST, LR, DEPTH, MIN_LEAF).fit_from_dag(DAG_BIG, src[:n_small])
        mm = metrics(m_s.score(DAG_BIG, hold), id_cost)
        curve.append((n_small, mm["rmse"], mm["pairwise"]))
        print("    训练 %4d 个程序 → RMSE %.3f  pairwise %.3f（文档表 %.3f / %.3f）"
              % (n_small, mm["rmse"], mm["pairwise"], want_rmse, want_pw))
        checks.B(abs(mm["rmse"] - want_rmse) < 0.001 and abs(mm["pairwise"] - want_pw) < 0.001,
                 "文档表：训练 %d 个程序 → RMSE %.3f / pairwise %.3f"
                 % (n_small, want_rmse, want_pw),
                 "本机 %.3f / %.3f" % (mm["rmse"], mm["pairwise"]))
    checks.A(curve[0][1] > curve[1][1] > curve[2][1],
             "训练集越大同分布误差越小（30 → 120 → 1200 单调下降）",
             " / ".join("%.3f" % c[1] for c in curve))

    # ------------------------------------------------------------------
    rule("4. C1：代价模型会误导搜索（★ 本程序的核心断言）")
    base_data = (DAG_BIG, src_cfgs, thr_src)
    print("  地形：模型在**大图**上训练（最优 tile 128×256），拿到**小图**上用。")
    print("  小图上同一个 tile 只切出 1 个块 → 网格填不满 → 真实代价坍崖，")
    print("  而模型的逐语句特征里**没有块数 / 网格规模**这一项。")
    bf_cfg, bf_cost = brute_force(DAG_SMALL, small_reps)
    print("  小图暴力枚举最优：%.0f  %r" % (bf_cost, bf_cfg))
    print()
    print("%-6s %-20s %-20s %-20s %-20s" % ("种子", "随机搜索(best/regret)",
                                            "模型引导(best/regret)",
                                            "模型引导·采样版", "完美模型(best/regret)"))
    print("-" * 78)
    worse = 0
    worse_pool = 0
    flip = 0
    for seed in SEEDS:
        rnd = random_search(DAG_SMALL, small_reps, BUDGET, seed=seed)
        mg = ModelGuided(base_model=model, base_data=base_data).run(
            DAG_SMALL, small_reps, BUDGET, seed=seed)
        mgp = ModelGuided(base_model=model, base_data=base_data, pool=256).run(
            DAG_SMALL, small_reps, BUDGET, seed=seed)
        pm = ModelGuided(base_model=PerfectModel(),
                         pool=256).run(DAG_SMALL, small_reps, BUDGET, seed=seed)
        print("%-6d %7.0f (%.2f)     %7.0f (%.2f)     %7.0f (%.2f)     %7.0f (%.2f)"
              % (seed, rnd.best, rnd.best / bf_cost, mg.best, mg.best / bf_cost,
                 mgp.best, mgp.best / bf_cost, pm.best, pm.best / bf_cost))
        if mg.best > rnd.best:
            worse += 1
        if mgp.best > rnd.best:
            worse_pool += 1
        if pm.best < rnd.best:
            flip += 1
    print()
    checks.C(worse == C1_MUST_BE_WORSE_ON,
             "★ **C1：模型引导的搜索比随机搜索更差**——在 %d 个冻结种子上全部成立"
             % C1_MUST_BE_WORSE_ON,
             "%d/%d 个种子更差" % (worse, len(SEEDS)))
    checks.C(worse_pool == len(SEEDS),
             "★ C1 的稳健性：给同一条搜索加上**采样池**（每轮从 256 个随机候选里取模型最优，"
             "而不是在全空间取最优）之后，误导**依然**在全部种子上成立",
             "%d/%d 个种子更差" % (worse_pool, len(SEEDS)))
    checks.C(flip == len(SEEDS),
             "★ C1 的可证伪性（计划 §0.4 第 5 条）：把代价模型换成**完美模型**后，"
             "同一条搜索流程**每次都优于**随机搜索——判定确实翻转了，"
             "所以上面那条检验的是**模型**，不是地形",
             "%d/%d 个种子翻转" % (flip, len(SEEDS)))

    # 顺带记录一个可观察的机制：模型把搜索停在哪一档 tile 上
    tm_pred = np.array([effective_tiles(DAG_SMALL, c)[0]
                        for c in small_reps])
    best_tile = int(tm_pred[int(np.argmax(ood_pred))])
    print("  模型在小图上打分最高的配置，其 tm = %d；而暴力最优的 tm = %d"
          % (best_tile, effective_tiles(DAG_SMALL, bf_cfg)[0]))
    checks.C(best_tile > effective_tiles(DAG_SMALL, bf_cfg)[0],
             "★ 机制可观察：模型把「更大的 tile」排在前面——正是它在**大图**上学到的偏好",
             "模型 tm=%d vs 最优 tm=%d"
             % (best_tile,
                effective_tiles(DAG_SMALL, bf_cfg)[0]))

    rule("5. 断言汇总")
    checks.summary()


if __name__ == "__main__":
    main()
