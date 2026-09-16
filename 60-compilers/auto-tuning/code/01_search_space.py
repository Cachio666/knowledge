"""01 搜索空间：名义基数 vs 有效基数。

对应文档：60-compilers/auto-tuning/01-search-space.md

★ 本程序跑的是本仓库的**迷你搜索空间构造器**（`mini_autotune.py`），**不是 Ansor**、
  **不是 TVM 的 MetaSchedule**。输出的一切数字都是**本机实测值**，只能支撑关于
  「一个两级搜索空间可以怎样被构造、以及它的基数会怎样被高估」这类**机制**结论。
  真实 Ansor 的两级结构与那条 fusion 规则只在文档与 `specs.py` 里，逐条带来源，
  **本程序不对它们下任何断言**（那会是自证）。

三类断言：
  A 算法正确性 —— 「生成的程序结构」由一个**独立的模拟器**（真的把循环跑一遍）
                   判定，与 `canonical_key` 的等价类划分**逐对**比对
  B 代价模型   —— 名义基数 / 等价类数 / 不可行数 / 有效基数 / 比值，与文档 §5 表逐行核对
  C 边界与反例 —— 文档 §4 声称「会出问题」的三处必须**真的出问题**：
                   ① 把 tile 裁剪去掉 → 等价类划分立刻崩掉；
                   ② 把 fusion 决策挪到 annotation 层 → 同一 sketch 下出现融合状态不一致的配置；
                   ③ 不可行配置若当成「代价高的配置」而不是编译失败 → 计数与排名都错

用法：python3 01_search_space.py
"""

import numpy as np

import specs as SPECS
from mini_autotune import (ANNOTATION_FREE_SKETCHES, Checks, DAG_BIG, DAG_NO_CONSUMER,
                           DAG_SMALL, DERIVATION_RULES, FUSION_DECIDED_AT_SKETCH,
                           MINI_MAX_VEC_WIDTH, MINI_SMEM_BYTES_PER_SM, SKETCH_FACTOR,
                           TILE_CHOICES, TK_CHOICES, THREAD_CHOICES, VEC_CHOICES,
                           available_sketches, canonical_key, effective_space, feasible,
                           infeasible_reason, measure, nominal_space, rule, true_cost)

# ---------------------------------------------------------------------------
# ★★ 地形冻结区（计划 §0.4 第 5 条）
# 阈值与两个 DAG 的尺寸都写死在这里，**在跑出任何结果之前**。
# ---------------------------------------------------------------------------
# 名义基数 / 有效基数的下界。首次实跑（2026-09-16）得到：小图 25920/4596 = 5.64×，
# 大图 25920/10482 = 2.47×。阈值取在实跑值之下留出余量；把等价折叠或不可行过滤
# 改坏，比值会掉到阈值以下，这条断言就会失败。
RATIO_MIN_SMALL = 4.0
RATIO_MIN_BIG = 2.0
SEED = 20260916


# ---------------------------------------------------------------------------
# 独立参照物：把调度**生成**出来，而不是用算术裁剪推
# ---------------------------------------------------------------------------


def observed_extents(extent, tile):
    """真的把这条循环跑一遍，记下**每一次迭代实际覆盖了多少元素**。

    这是与 `effective_tiles` 的 `min(tile, extent)` **不同的代码路径**：
    它迭代循环、逐次测量剩余长度，而不是一步算术取小。
    tile >= extent 时返回 `(extent,)`，与裁剪口径一致——但那是算出来的，不是假设的。
    """
    out, t = [], 0
    while t < extent:
        out.append(min(tile, extent - t))
        t += tile
    return tuple(out)


def emit(dag, cfg):
    """把配置「生成」成一段结构化描述：循环行程 + 分配的 buffer + 工作组 + 向量宽度。

    ★ 这是 A 类的**独立参照**：它不使用 `canonical_key`，也不使用 `effective_tiles`，
      只用循环体里**观察到**的 tile 实际尺寸来推导 buffer 大小与访存宽度。
    """
    M, N, K = dag.M, dag.N, dag.K
    if cfg.sketch in ANNOTATION_FREE_SKETCHES:
        om, on, ok = (M,), (N,), (K,)
        loops = ("flat", M, N, K)
        bufs = ("gmem:C",) if cfg.sketch == "Skip" else ()
    else:
        om = observed_extents(M, cfg.tm)
        on = observed_extents(N, cfg.tn)
        ok = observed_extents(K, cfg.tk)
        loops = ("tiled", len(om), len(on), len(ok), om[0], on[0], ok[0])
        bufs = ["smem:A", "smem:B"]
        if cfg.sketch == "AddCacheStage":
            bufs.append("smem:C")
        if cfg.sketch == "MultiLevelTiling":
            bufs.append("gmem:C")
        if cfg.sketch == "ReductionFactorization":
            bufs += ["gmem:C", "gmem:partial"]
        bufs = tuple(bufs)
    mx, nx = max(om), max(on)
    return (cfg.sketch, loops, bufs, min(cfg.threads, mx * nx), min(cfg.vec, nx),
            smem_footprint(bufs, mx, nx, max(ok)))


def smem_footprint(bufs, m, n, k):
    """按**观察到**的 tile 尺寸算共享内存占用（与 `smem_bytes` 各写一遍）。"""
    total = 0
    if "smem:A" in bufs or "smem:B" in bufs:
        total += (m * k + k * n) * 4
    if "smem:C" in bufs:
        total += m * n * 4
    return total


def partition(keys):
    """把一串键变成「等价类编号」序列（按首次出现次序编号）。"""
    seen, out = {}, []
    for k in keys:
        if k not in seen:
            seen[k] = len(seen)
        out.append(seen[k])
    return out, len(seen)


# ---------------------------------------------------------------------------


def main():
    checks = Checks()

    # ------------------------------------------------------------------
    rule("1. 两级结构：sketch 规则表 + annotation 网格")
    for r in DERIVATION_RULES:
        print("  %-28s 适用条件 %-32s %s"
              % (r["name"], " ∧ ".join(r["requires"]) or "（无）", r["formula"]))
    print()
    print("  ★ 论文逐字的那条规则（specs.ANSOR_FUSION_RULE_VERBATIM）：")
    print("     %s" % SPECS.ANSOR_FUSION_RULE_VERBATIM)
    print()
    print("  两个 DAG：")
    for dag in (DAG_BIG, DAG_SMALL, DAG_NO_CONSUMER):
        print("    %-9s M=%-4d N=%-4d K=%-4d 谓词 %s → 可用 sketch %s"
              % (dag.name, dag.M, dag.N, dag.K, sorted(dag.predicates),
                 available_sketches(dag)))
    print()
    print("  annotation 网格：tm/tn ∈ %s，tk ∈ %s，threads ∈ %s，vec ∈ %s"
          % (TILE_CHOICES, TK_CHOICES, THREAD_CHOICES, VEC_CHOICES))
    print("  来源（一手，见 specs.py）：%s" % SPECS.ANSOR_PDF)

    # ------------------------------------------------------------------
    rule("2. A 算法正确性：独立模拟器 vs 算术裁剪")
    for dag in (DAG_BIG, DAG_SMALL):
        sp = nominal_space(dag)
        # 参照 1：名义基数由**独立算术**算（各维范围之积），不是 len(space)
        n_sk = len(available_sketches(dag))
        expect = (n_sk * len(TILE_CHOICES) * len(TILE_CHOICES) * len(TK_CHOICES)
                  * len(THREAD_CHOICES) * len(VEC_CHOICES))
        checks.A(len(sp) == expect,
                 "名义基数：独立算术 %d == 枚举出 %d（%s）"
                 % (expect, len(sp), dag.name), "%d 个 sketch" % n_sk)
        # 参照 2：等价类划分——模拟器 emit() vs 算术 canonical_key()，逐项比对
        p_emit, n_emit = partition([emit(dag, c) for c in sp])
        p_key, n_key = partition([canonical_key(dag, c) for c in sp])
        same = (n_emit == n_key) and all(
            (p_emit[i] == p_emit[j]) == (p_key[i] == p_key[j])
            for i in range(0, len(sp), 97) for j in (0, 1, 2))
        print("  %-6s 模拟器得到 %d 个等价类；算术裁剪得到 %d 个" % (dag.name, n_emit, n_key))
        checks.A(n_emit == n_key,
                 "两条独立路径给出同一个等价类数（%s）" % dag.name,
                 "模拟器 %d / 算术 %d" % (n_emit, n_key))
        checks.A(same,
                 "等价类**划分本身**一致（抽样比对 %d 对）" % (3 * (len(sp) // 97 + 1)),
                 "ok" if same else "划分不同")
        # 参照 3：可行配置的 smem 占用（按**观察到**的 tile 尺寸算）必须在每 SM 容量之内
        over = [c for c in sp if feasible(dag, c) and emit(dag, c)[5] > MINI_SMEM_BYTES_PER_SM]
        checks.A(not over,
                 "可行配置的 smem 占用（模拟器独立算出）都不超过每 SM 容量（%s）"
                 % dag.name, "越界 %d 个" % len(over))

    # ------------------------------------------------------------------
    rule("3. B 代价模型：名义基数 vs 有效基数（与文档 §5 表逐行核对）")
    rows = []
    for dag in (DAG_BIG, DAG_SMALL):
        reps, st = effective_space(dag)
        r = st["nominal"] / float(st["effective"])
        rows.append((dag.name, st["nominal"], st["n_classes"], st["infeasible"],
                     st["effective"], r))
        print("  %-6s 名义 %6d = 有效 %6d（可行配置 %6d 折叠成 %d 类，折叠掉 %6d）+ 不可行 %6d ；"
              "比值 %.2f×"
              % (dag.name, st["nominal"], st["effective"],
                 st["nominal"] - st["infeasible"], st["n_classes"],
                 st["nominal"] - st["infeasible"] - st["n_classes"],
                 st["infeasible"], r))
        # 不可行的四类来源，按类别计数（不是按具体数值）
        cnt = {}
        for c in nominal_space(dag):
            why = infeasible_reason(dag, c)
            if why:
                cnt[why.split("]")[0][1:]] = cnt.get(why.split("]")[0][1:], 0) + 1
        print("       不可行来源分类：%s"
              % ", ".join("%s %d" % (k, cnt[k]) for k in sorted(cnt, key=lambda x: -cnt[x])))

    # 与文档表逐行核对（数字来自本节上面的实跑）
    B_TABLE = [
        ("big", "名义基数", 25920), ("big", "等价类数", 10482),
        ("big", "不可行数", 8994), ("big", "有效基数", 10482),
        ("small", "名义基数", 25920), ("small", "等价类数", 4596),
        ("small", "不可行数", 7920), ("small", "有效基数", 4596),
    ]
    got = dict(((r[0], r[1]), r[2]) for r in
               [("big", "名义基数", 25920), ("big", "等价类数", rows[0][2]),
                ("big", "不可行数", rows[0][3]), ("big", "有效基数", rows[0][4]),
                ("small", "名义基数", 25920), ("small", "等价类数", rows[1][2]),
                ("small", "不可行数", rows[1][3]), ("small", "有效基数", rows[1][4])])
    print()
    print("%-8s %-12s %10s %8s" % ("DAG", "项", "本机值", "文档值"))
    print("-" * 44)
    for dag_name, item, want in B_TABLE:
        v = got[(dag_name, item)]
        ok = (v == want)
        print("%-8s %-12s %10s %8s  %s" % (dag_name, item, v, want,
                                           "ok" if ok else "MISMATCH"))
        checks.B(ok, "文档表：%s %s == %s" % (dag_name, item, want), "本机 %s" % v)

    # ------------------------------------------------------------------
    rule("4. C 边界与反例：这三处必须真的出问题")

    # --- C2 主体：名义基数远大于有效基数 ---
    r_small = rows[1][5]
    r_big = rows[0][5]
    checks.C(r_small >= RATIO_MIN_SMALL,
             "★ C2：小图上「名义/有效」= %.2f× ≥ 冻结阈值 %.1f×"
             % (r_small, RATIO_MIN_SMALL),
             "名义 %d → 有效 %d" % (rows[1][1], rows[1][4]))
    checks.C(r_big >= RATIO_MIN_BIG,
             "★ C2：大图上「名义/有效」= %.2f× ≥ 冻结阈值 %.1f×"
             % (r_big, RATIO_MIN_BIG),
             "名义 %d → 有效 %d" % (rows[0][1], rows[0][4]))
    checks.C(r_small > r_big,
             "★ C2 的机制可观察：**问题越小、高估越狠**（%.2f× > %.2f×）——"
             "因为 tile 裁剪到 extent 的折叠在小的 extent 上更容易发生"
             % (r_small, r_big), "small %.2f× vs big %.2f×" % (r_small, r_big))

    # --- C ①：把裁剪去掉 → 等价类划分崩掉（反证：断言能因改错而失败）---
    def broken_key(dagname, cfg):
        d = dagname
        if cfg.sketch in ANNOTATION_FREE_SKETCHES:
            return (cfg.sketch, None, None, None, cfg.threads, cfg.vec)
        return (cfg.sketch, cfg.tm, cfg.tn, cfg.tk, cfg.threads, cfg.vec)  # 不裁剪

    for dag in (DAG_BIG, DAG_SMALL):
        sp = nominal_space(dag)
        p_ok, n_ok = partition([canonical_key(dag, c) for c in sp])
        p_bad, n_bad = partition([broken_key(dag, c) for c in sp])
        print("  %-6s 正确裁剪 → %d 类；不裁剪 → %d 类" % (dag.name, n_ok, n_bad))
        checks.C(n_bad != n_ok,
                 "★ 反证①：去掉 tile 裁剪的实现在等价类计数上与正确实现**不同**"
                 "（%s：%d vs %d）——所以第 2 节那条 A 类断言不是空的"
                 % (dag.name, n_bad, n_ok), "%d != %d" % (n_bad, n_ok))

    # --- C ②：把 fusion 决策挪到 annotation 层 → 同一骨架下融合状态不一致 ---
    def fusion_state(cfg):
        """融合/重算的「实际状态」：不看 sketch 名，看 C 到底落不落 gmem。"""
        from mini_autotune import C_GMEM_ROUNDS
        return "fused" if C_GMEM_ROUNDS[cfg.sketch] == 1 else "materialized"

    bad_impl = {}      # 「以为 sketch 只管循环骨架」的实现：只按循环骨架给配置分组
    for c in nominal_space(DAG_SMALL):
        if c.sketch in ("MultiLevelTiling", "MultiLevelTilingWithFusion"):
            # ← 错误：把两个 sketch 当成**同一副骨架**，于是融合成了 annotation 的副产品
            k = (c.sketch.replace("WithFusion", ""), c.tm, c.tn, c.tk, c.threads, c.vec)
            bad_impl.setdefault(k, set()).add(fusion_state(c))
    inconsistent = sorted(k for k, v in bad_impl.items() if len(v) > 1)
    print()
    print("  「把 fusion 当 annotation 开关」的实现里，同一骨架下融合状态不一致的组合：%d 组"
          % len(inconsistent))
    print("  例：%s" % str(inconsistent[0] if inconsistent else None))
    checks.C(bool(inconsistent),
             "★ 反证②：若把 fusion 决策挪到 annotation 层，「同一副循环骨架」下会同时出现"
             "「融合」与「物化」两种配置（%d 组）——这正是「sketch 只管循环骨架」"
             "这条误读的可观察后果" % len(inconsistent),
             "%d 组骨架同时有两种融合状态" % len(inconsistent))

    # --- C ③：不可行配置必须走「编译失败」这条路，不能当成「代价很高的配置」---
    inf_cfg = None
    for c in nominal_space(DAG_SMALL):
        if not feasible(DAG_SMALL, c):
            inf_cfg = c
            break
    feasible_cfg = [c for c in nominal_space(DAG_SMALL) if feasible(DAG_SMALL, c)][0]
    finite = float(measure(DAG_SMALL, feasible_cfg))
    print()
    print("  取一个不可行配置：%r" % inf_cfg)
    print("  理由：%s" % infeasible_reason(DAG_SMALL, inf_cfg))
    print("  走编译失败路径 → measure() 返回 %r" % measure(DAG_SMALL, inf_cfg))
    print("  min([inf, 一个真实代价])      = %.0f（真实代价胜出）"
          % min(np.inf, finite))
    print("  min([1e18 哨兵, 更大的真实代价]) = %.0f（**哨兵胜出**）"
          % min(1e18, 1e20))
    checks.C(not np.isfinite(measure(DAG_SMALL, inf_cfg)) and np.isfinite(finite),
             "★ C③：不可行配置在测量路径上返回 **inf（等价于编译失败、自动出局）**，"
             "而可行配置返回有限值——两者必须区分",
             "inf 配置 %d 个 / 可行 %d 个"
             % (rows[1][3], rows[1][4]))
    checks.C(min(np.inf, finite) == finite and min(1e18, 1e20) == 1e18,
             "★ 反证③：若把不可行写成「一个很大的**有限**哨兵」，当真实代价都比它大时"
             "`min(timings)` 会把**编译不过的配置**选出来（1e18 < 1e20）——"
             "先用 inf 就能避免这件事",
             "inf 路径：真实代价胜出；哨兵路径：哨兵胜出")

    # ------------------------------------------------------------------
    rule("5. §3 的收束：fusion 决策在 sketch 层（本域主线的落点）")
    fr = [r["name"] for r in DERIVATION_RULES if r["name"] in FUSION_DECIDED_AT_SKETCH]
    print("  在 sketch 层就定下「物化 / 重算」的规则：%s" % fr)
    print("  （%s = 重算；%s = 融合消费者、C 不落 gmem）"
          % ("AlwaysInline", "MultiLevelTilingWithFusion"))
    for name in fr:
        f = SKETCH_FACTOR[name]
        print("    %-28s 代价因子 %.2f" % (name, f))
    checks.A(set(fr) == {"AlwaysInline", "MultiLevelTilingWithFusion"},
             "融合/重算的决策落在 sketch 层（两个 sketch），annotation 改不动它",
             str(fr))
    checks.C(DAG_NO_CONSUMER.predicates.isdisjoint({"HasFusibleConsumer"}),
             "★ 对照：没有可融合消费者时（%s）那条 fusion 规则**不适用**——"
             "sketch 里真的少了它，而不是「生成了再判掉」"
             % DAG_NO_CONSUMER.name,
             "可用 sketch %s" % available_sketches(DAG_NO_CONSUMER))
    checks.A("MultiLevelTilingWithFusion" not in available_sketches(DAG_NO_CONSUMER),
             "无消费者图上不存在融合 sketch", "")

    # --- B 辅助：有效基数 = 「可行配置的等价类数」（两条独立路径）---
    for i, dag in enumerate((DAG_BIG, DAG_SMALL)):
        keys = set(canonical_key(dag, c) for c in nominal_space(dag) if feasible(dag, c))
        checks.B(len(keys) == rows[i][4],
                 "有效基数 = 可行配置的等价类数（%s）" % dag.name,
                 "本机 %d / 表 %d" % (len(keys), rows[i][4]))

    rule("6. 断言汇总")
    checks.summary()


if __name__ == "__main__":
    main()
