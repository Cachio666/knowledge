#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_diagnosis_walkthrough.py —— 一次**完整**的诊断：四步，顺序不能乱。

配套文档：`03-diagnosis-walkthrough.md`。★ 三类断言（A/B/C）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 四步（每一步的**前置条件**与「跳过它会怎样」写在 `part_A` 的段注释里）：

  ① 端到端计时（否则不知道值不值得查）
  ② `cProfile` 定位到函数（否则不知道改哪）
  ③ 对热点判 bound —— **用 `microbench` 的 C 微基准**测该访问模式的延迟/带宽（否则不知道怎么改）
  ④ 改一处，重测，**用 `45/02` 的 δ\* 判定改动是否落在噪声外**（否则会把噪声当成果）

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` /
> `python3 3.13.5` / `numpy 2.4.4` / `Apple clang 21.0.0`），**不是 GPU、不是 Nsight Compute**。

★★ 本程序的可证伪断言：

  · `A.1`  **②定位**：`cProfile` 报的最热点**就是**那个循环函数，且它的 exclusive 时间
           占比 > 50%（改坏热点函数的名字/结构，这条必须失败）；
  · `A.2`  **③判 bound**：慢版本耗时 > **10 倍**延迟限（微基准 K=1 在同一规模上的耗时）
           ⇒ 它的瓶颈**不在内存**（本机实测约 42 倍）；
  · `A.3`  **④改一处**：向量化后耗时 < 慢版本的 **1/10**（本机实测约 1/880）；
  · `A.4`  **④δ\* 判定（真实改动）**：`|Δ|/δ* > 3`（本机 7 次实测 20.3–69.6）⇒ **可判定**；
  · `A.5`  **④δ\* 判定（安慰剂改动）**：`|Δ|/δ* < 2`（本机 7 次实测落在 0.03–0.39）
           ⇒ **不构成成果**。★ 窗开得宽是**有意的**，理由见 `part_A` 的注；
  · `C.1`  **δ\* 的系数**与 `MDE_ANCHORS` 引文里那个数**相等**（改错常数立刻失败）
           ← **`make falsify` 的「把 δ\* 系数改错」注入项就打在这一点上**；
  · `C.3`  失败标记与 `01` / `02` / `meta_falsify` / `mutate` 五处一致。

用法：
    python3 03_diagnosis_walkthrough.py            # 默认档
    PERF_QUICK=1 python3 03_diagnosis_walkthrough.py   # 快档（诊断工具用，断言相同）
    python3 03_diagnosis_walkthrough.py --lint     # AST 扫描

★ 外部工具：`clang`（**必需**，见 `microbench.require_clang`）。
★ **与 `45` 的分工写死**：第 ④ 步**代入** `45/02` 的
  `δ* = 2.801585218112968 · σ · √(2/n)`，**本域不重推**——
  正文在 `45-kernel-lifecycle/convergence-and-tuning/02-significance`（此处回链）。
"""

import ast
import cProfile
import io
import os
import pstats
import shutil
import statistics
import sys
import time

import microbench
import numpy as np
import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题自定的字面形态是 `*** FAIL ***`。
#   判定侧必须【行首锚定】—— 程序**成功时也会打印这个字面量**（`C.3` 的 detail 里就有）。
ASSERT_FAIL_MARK = "*** FAIL ***"

FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0}
_PASS = {"A": 0, "B": 0, "C": 0}
SKIPPED = []


def check(cls, name, cond, detail=""):
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]     %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


def skip(cls, name, why):
    SKIPPED.append(name)
    print("  [SKIP]   %-6s %s" % (name, why))


def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d ⇒ "
          "断言 %d 条，通过 %d 条，失败 %d 条；跳过 %d 项（跳过不计入通过）"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], t, p, len(FAILS), len(SKIPPED)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 那个**故意写慢的**被测程序（纯 python + numpy，全部在本文件里）
# ===========================================================================
# ★ 「瘦身后的替代版」与「值」无关，只与**代价**有关：被测对象是**耗时**，不是结果。
def make_input(n):
    """冷路径：造一个 float64 数组。8 MiB 的规模**超过本机 L2（4 MiB）**。"""
    return (np.arange(n, dtype=np.float64) % 7.0) - 3.0


def slow_sum_sq(x):
    """热函数（v1）：Python 层循环 —— **每一次取元素都要付解释器的钱**。"""
    s = 0.0
    for i in range(len(x)):
        s += x[i] * x[i]
    return s


def slow_sum_sq_sham(x):
    """**安慰剂改动**：只把局部变量 `s` 改名成 `acc`。

    ★★ 它**不可能**改变机器行为 —— 这是第 ④ 步要展示的那一半：
      「我改了一处，重测快了 0.3%」在 δ* 判据下**不可报**。
      与「真实改动」（向量化）配成**一对**：同一个判据给出相反结论。
    """
    acc = 0.0
    for i in range(len(x)):
        acc += x[i] * x[i]
    return acc


def fast_sum_sq(x):
    """改一处之后的版本（v2）：把循环交给 numpy（内部的 BLAS/向量化内核）。"""
    return float(np.dot(x, x))


def params():
    """★ 默认档是正文引用的口径（`specs`）；`PERF_QUICK=1` 是**诊断工具**用的快档。

    ★ 断言全是**结构性**的（谁是热点、比值落在哪个窗内），所以两档走**同一条代码路径**。
    """
    if os.environ.get("PERF_QUICK") == "1":
        return dict(specs.DIAG_QUICK, quick=True)
    return {"reps": specs.DIAG_REPS, "warmup_rounds": specs.DIAG_WARMUP_ROUNDS,
            "fast_inner": specs.DIAG_FAST_INNER, "quick": False}


# ===========================================================================
# 2. 每一步 ①：端到端计时（**交替编排**）
# ===========================================================================
def measure(versions, reps, warmup, x):
    r"""交替编排（A/B/C/A/B/C…），返回 `{名字: [每次测量的单次耗时]}`。

    ★★ 为什么**交替**而不是「先跑完 A 再跑完 B」：同域 `tuning-practice/03` 的实测
      （计划 §0.7 D-12）——A 与 B **逐字等价**（真实差异为 0）时，顺序编排仍稳定给出
      约 **+0.86%** 的虚假差异（8/8 轮同号），交错把它压到 **0.015%**。
      ⇒ 顺序编排会把「两个时间段的系统状态差异」记到「改动效果」头上。
    ★ 每轮**轮换顺序**（偶数轮正序、奇数轮倒序）：把「处在第二个被测」这件事也抵消掉。
    ★★ 短操作要**在测量内部重复 R 次再除以 R**（`inner`）：
      一次 0.18 ms 的操作，相对噪声被调度抖动支配（实测 σ 可达 50%），
      而一次 160 ms 的操作 σ 只有 1% 量级 —— **两组的相对噪声不可比，δ\* 就没有意义**。
      这是本域「重复次数不是自由参数」在**比较两个版本**时的形态。
    """
    series = {name: [] for name, _, _ in versions}
    for r in range(reps):
        order = versions if r % 2 == 0 else list(reversed(versions))
        for name, fn, inner in order:
            t0 = time.perf_counter()
            for _ in range(inner):
                fn(x)
            series[name].append((time.perf_counter() - t0) / inner)
    for name in series:
        series[name] = series[name][warmup:]
    return series


# ===========================================================================
# 3. 第 ④ 步的判据：**代入** `45/02` 的 δ*
# ===========================================================================
def robust_sigma(vals, med):
    """**稳健**的组内尺度估计：`1.4826 × MAD`（相对量，先减掉本组中位数）。

    ★★ 为什么不用样本标准差 —— **这是口径选择，不是放宽门槛**：
      本域 §0.7 **D-8/D-10** 的实测：同一份固定工作量的耗时分布是**重尾**的
      （`max/中位 = 601.66×`、偏度 **+62.8**、超额峰度 **+4469.5**），
      **样本标准差会被那 0.1% 的尾部定义**（D-10：`σ/中位 = 45%`，而 `p50→p90` 只差 `4.6%`）。
      把一个**被尾部定义**的 σ 代进 `δ*`，等于让「这一轮恰好撞上一次调度抖动」
      决定这条断言的红绿 —— 实施者实测：同一次实验里 σ 从 2.4% 漂到 **25.9%**，
      于是 `|Δ|/δ*` 从 41 掉到 **3.5**（`|Δ|` 一点没变，变的全是 δ*）。
    ★ `1.4826` 是「MAD 在正态下的无偏换算系数」；对重尾分布它给出的是一个
      **不被尾部支配**的量级。**它不是 `45` 的公式的一部分** —— `δ*` 的公式照抄 `45/02`，
      本域只负责给它一个**可复现的 σ 估计量**（`45` 把 σ 列为「表外输入」，见 `specs.UPSTREAM_CLAIMS`）。
    """
    n = len(vals)
    if n < 2:
        return 0.0
    dev = sorted(abs((t - med) / med) for t in vals)
    mad = statistics.median(dev)
    return 1.4826 * mad


def judge(va, vb):
    r"""两组比较的 δ\* 判定 —— **只代入、不重推**。

    σ 用**合并的稳健组内尺度**（`robust_sigma`）：每组先减掉自己组的中位数，
    再把两组的自由度合起来 ⇒ 合并后的 σ 里**不含组间差**。
    """
    ma, mb = statistics.median(va), statistics.median(vb)
    dof = (len(va) - 1) + (len(vb) - 1)
    sa, sb = robust_sigma(va, ma), robust_sigma(vb, mb)
    sigma = ((sa * sa * (len(va) - 1) + sb * sb * (len(vb) - 1)) / dof) ** 0.5 if dof > 0 else 0.0
    n = len(va)
    delta_star = specs.mde_delta_star(sigma, n)
    gap = abs(mb - ma) / ma
    ratio = (gap / delta_star) if delta_star > 0 else float("inf")
    n_req = specs.mde_required_n(sigma, gap) if gap > 0 else float("inf")
    return {"median_a": ma, "median_b": mb, "sigma": sigma, "delta_star": delta_star,
            "gap": gap, "ratio": ratio, "n": n, "n_required": n_req,
            "sigma_a": sa, "sigma_b": sb}


# ===========================================================================
# 4. A 类：四步诊断
# ===========================================================================
def part_A(bpaths, par):
    x = make_input(specs.DIAG_N_ELEMS)
    slow_inner, fast_inner = 1, par["fast_inner"]
    versions = (("v1_loop", slow_sum_sq, slow_inner),
                ("v1_loop_sham", slow_sum_sq_sham, slow_inner),
                ("v2_vec", fast_sum_sq, fast_inner))

    print("=" * 74)
    print("A 类 · 一次完整的诊断：端到端 → 定位 → 判 bound → 改一处 → 判 δ*")
    print("=" * 74)
    print("  ★ 被测程序就在本文件里：一个**故意写慢的** python + numpy 程序。")
    print("     数据规模 = %s 个 float64 = **%s**（★ 超过本机 L2 = %s）"
          % (specs.DIAG_N_ELEMS, microbench.describe(specs.DIAG_BYTES),
             microbench.describe(specs.MACHINE["l2_bytes"])))
    print("  ★ 测量条件：%s；python %s / numpy %s / clang %s"
          % (specs.MEASURE_IMPL, sys.version.split()[0], np.__version__,
             microbench.clang_version()))
    print("  ★ 编排：**交替**（每轮都把三个版本各跑一次，且偶数轮正序、奇数轮倒序），")
    print("     重复 %d 轮、丢掉前 %d 轮（`45/01` 讲 warmup，正文在那边）。"
          % (par["reps"], par["warmup_rounds"]))

    # ---- ① 端到端 ----
    series = measure(versions, par["reps"], par["warmup_rounds"], x)
    print()
    print("  ① 端到端计时（每次测量的**单次耗时**，n=%d）" % len(series["v1_loop"]))
    for name, fn, inner in versions:
        v = series[name]
        print("     %-14s 中位 %.6f s（min %.6f · max %.6f）  测量内部重复 %d 次"
              % (name, statistics.median(v), min(v), max(v), inner))

    # ---- ② 定位 ----
    print()
    print("  ② `cProfile` 定位热点 —— 跳过它的后果：**不知道该改哪一行**。")
    pr = cProfile.Profile()
    pr.enable()
    slow_sum_sq(x)
    pr.disable()
    st = pstats.Stats(pr, stream=io.StringIO())
    st.sort_stats("tottime")
    items = sorted(st.stats.items(), key=lambda kv: -kv[1][2])      # 按 tottime 降序
    total_tt = sum(v[2] for v in st.stats.values())
    top_name, top_tt = items[0][0][2], items[0][1][2]
    for k, v in items[:4]:
        print("     %-34s ncalls=%-8d tottime=%.4f cumtime=%.4f"
              % (k[2][:34], v[0], v[2], v[3]))
    share = top_tt / total_tt if total_tt > 0 else 0.0
    check("A", "A.1", top_name == "slow_sum_sq" and share > 0.50,
          "★ ②定位：`cProfile` 的最热点**就是** `%s`，exclusive 时间占 **%.1f%%**（> 50%%）"
          % (top_name, 100.0 * share))

    # ---- ③ 判 bound：**用 microbench 的 C 微基准**测同一规模上的延迟/带宽 ----
    print()
    print("  ③ 判 bound —— 跳过它的后果：**不知道怎么改**（改并发度？还是换算法？）。")
    print("     做法：拿 `microbench` 的 C 微基准，在**同一个规模**上测这条访问模式的两端——")
    print("       · K=1（**延迟限**：一次只有一个未完成的独立读）")
    print("       · K=%d（**带宽限**：并发度打满）" % microbench.DEP_K_FAST)
    t_lat = microbench.dep_time(bpaths, specs.DIAG_BYTES, 1, microbench.DEP_REPS)
    t_bw = microbench.dep_time(bpaths, specs.DIAG_BYTES, microbench.DEP_K_FAST,
                               microbench.DEP_REPS)
    t_slow = statistics.median(series["v1_loop"])
    r_lat, r_bw = t_slow / t_lat, t_slow / t_bw
    print("     %s 上同一访问模式：延迟限（K=1）**%.6f s** · 带宽限（K=%d）**%.6f s** ⇒ 差 %.1f 倍"
          % (microbench.describe(specs.DIAG_BYTES), t_lat, microbench.DEP_K_FAST,
             t_bw, t_lat / t_bw))
    print("     慢版本（v1_loop）中位 **%.6f s** ⇒ 是延迟限的 **%.1f 倍**、带宽限的 **%.0f 倍**"
          % (t_slow, r_lat, r_bw))
    check("A", "A.2", r_lat > 10.0,
          "★ ③判 bound：慢版本是**延迟限**的 %.1f 倍（> 10）⇒ **即便按最慢的访存方式算，"
          "内存也只能解释总耗时的 %.1f%%** ⇒ 瓶颈**不在内存**" % (r_lat, 100.0 / r_lat))

    # ---- ④ 改一处，重测，判 δ* ----
    print()
    print("  ④ 改一处（把 Python 循环换成向量化），重测，**用 `45/02` 的 δ\\* 判**")
    print("     —— 跳过它的后果：**把噪声当成果**。")
    t_fast = statistics.median(series["v2_vec"])
    check("A", "A.3", t_fast < t_slow / 10.0,
          "★ ④改一处：向量化后中位 %.6f s < 慢版本的 1/10（慢 %.6f s ⇒ **%.0f 倍**）"
          % (t_fast, t_slow, t_slow / max(t_fast, 1e-12)))

    j_real = judge(series["v1_loop"], series["v2_vec"])
    j_sham = judge(series["v1_loop"], series["v1_loop_sham"])
    for tag, j in (("真实改动 v1→v2", j_real), ("安慰剂改动 v1→v1'", j_sham)):
        print("     %-16s 合并组内 σ=%.4f%%（两组 %.4f%% / %.4f%%）· δ*=%.4f%%（n=%d）"
              "· |Δ|=%.4f%% ⇒ **|Δ|/δ* = %.2f**"
              % (tag, 100.0 * j["sigma"], 100.0 * j["sigma_a"], 100.0 * j["sigma_b"],
                 100.0 * j["delta_star"], j["n"], 100.0 * j["gap"], j["ratio"]))
        if j["n_required"] != float("inf"):
            print("       ⇒ 要把这个 |Δ| 判到 80%% 检定力，每组需要约 %.0f 次重复（现跑 %d 次）"
                  % (j["n_required"], j["n"]))
    print("     ★ δ\\* 是**从 `45/02` 代入**的：`δ* = %.15g · σ · √(2/n)`，本域不重推。"
          % specs.MDE_COEFFICIENT_AT_POWER_80)

    check("A", "A.4", j_real["ratio"] > 3.0,
          "★ ④δ\\* 判定（**真实改动**）：|Δ|/δ* = %.2f > 3 ⇒ **可判定**——"
          "这个改动落在分辨极限之外（本机 7 次实测 20.3–69.6）" % j_real["ratio"])
    # ★★ A.5 的窗开得宽是**有意的**（判据来自实测，不是拍脑袋）：
    #   安慰剂改动**逐字等价**（只把局部变量改名），所以它的 |Δ| 是**纯噪声**：
    #   · 6 次独立实测里 |Δ|/δ* 落在 **0.03–0.39**；
    #   · 窗取 **2.0** 而不是 1.0，是为了不让「这一轮恰好撞上一次调度抖动」
    #     决定这条断言的红绿（`|Δ|` 本身可以到 0.7%，而稳健 σ 可以小到 0.5%）。
    #   ⇒ 这一条是**冒烟检查**，不是精确判据；真正有区分力的是 **A.4 与 A.5 的对照**：
    #     同一条判据对真实改动给「可判定」、对噪声给「不可判定」，两者差 60 倍以上。
    check("A", "A.5", j_sham["ratio"] < 2.0,
          "★ ④δ\\* 判定（**安慰剂改动**）：|Δ|/δ* = %.2f < 2 ⇒ **不构成成果**——"
          "这正是第 ④ 步要挡住的那件事（本机 7 次实测的区间 0.03–0.39）" % j_sham["ratio"])
    check("A", "A.6", j_real["ratio"] > 5.0 * j_sham["ratio"],
          "★ **对照**：同一条 δ\\* 判据对两种改动给出**相反结论**（%.1f vs %.2f，差 %.0f 倍）"
          % (j_real["ratio"], j_sham["ratio"],
             j_real["ratio"] / max(j_sham["ratio"], 1e-9)))

    # ---- 改完之后，它在哪儿？----
    print()
    print("  ★ 改完之后的版本落在哪儿（**只打印，不作为跨实现的断言**）：")
    print("     v2_vec 中位 %.6f s ÷ 带宽限 %.6f s = **%.2f**" % (t_fast, t_bw, t_fast / t_bw))
    print("     ⚠ **这个比值不是「超过了物理极限」**：`microbench` 的那段 C 与 numpy 的")
    print("       BLAS 内核是**两个不同的实现**（numpy 版实测 %.1f GB/s、C 探针 %.1f GB/s）——"
          % (specs.DIAG_BYTES / t_fast / 1e9, specs.DIAG_BYTES / t_bw / 1e9))
    print("       比较的是**量级**，不是同一个数的两次测量（P13 判据第③句）。")
    return {"t_slow": t_slow, "t_fast": t_fast, "t_lat": t_lat, "t_bw": t_bw,
            "j_real": j_real, "j_sham": j_sham, "share": share}


# ===========================================================================
# 5. B 类：微基准给出的两端（延迟限 / 带宽限）
# ===========================================================================
def part_B(bpaths, res):
    print()
    print("=" * 74)
    print("B 类 · 第 ③ 步用的那台仪器：延迟限与带宽限在同一规模上的差")
    print("=" * 74)
    ratio = res["t_lat"] / res["t_bw"]
    print("     %s：K=1（延迟限）%.6f s · K=%d（带宽限）%.6f s ⇒ **%.1f 倍**"
          % (microbench.describe(specs.DIAG_BYTES), res["t_lat"],
             microbench.DEP_K_FAST, res["t_bw"], ratio))
    print("     ★ 这两个数是**同一段 C 代码**（`microbench.DEP_SRC`）的两个 K 值 ——")
    print("       不是两段不同的代码（同域 `01` 的 C 类用的是**另一段**，两者不得并列比较）。")
    check("B", "B.1", ratio > 5.0,
          "★ 同一规模上，**延迟限比带宽限慢 %.1f 倍**（> 5）—— 这就是第 ③ 步"
          "「判 bound」要用的那把尺子的量程" % ratio)


# ===========================================================================
# 6. C 类：δ* 的系数、上游契约、标记自检
# ===========================================================================
def read_mark_from(path):
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        tgts = n.targets if isinstance(n, ast.Assign) else (
            [n.target] if isinstance(n, ast.AnnAssign) else [])
        for t in tgts:
            if isinstance(t, ast.Name) and t.id == "ASSERT_FAIL_MARK" \
                    and isinstance(n.value, ast.Constant):
                return n.value.value
    return None


def part_C(doc_dir):
    print()
    print("=" * 74)
    print("C 类 · δ* 的系数：**从 45 代入，本域不重推**")
    print("=" * 74)
    ok, lits = specs.mde_coefficient_selfcheck()
    print("     `MDE_COEFFICIENT_AT_POWER_80` = %.15g" % specs.MDE_COEFFICIENT_AT_POWER_80)
    print("     从 `MDE_ANCHORS` 的引文里抽出的数     = %s" % lits)
    check("C", "C.1", ok,
          "★ 该常数与**引文里那个数**相等 ⇒ 改错常数立刻失败（**不需要上游文件在场**）")

    rows = specs.mde_anchor_check(specs.REPO_ROOT)
    if rows is None:
        skip("C", "C.2", "上游 `45/02-significance` 不在场 ⇒ δ* 系数的**逐字核对** SKIP")
    else:
        bad = [(lbl, g, w) for (lbl, g, w) in rows if g < w]
        for lbl, g, w in bad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (lbl, g, w))
        check("C", "C.2", not bad,
              "★ δ* 的系数 **%.15g** 在 `45/02` 原文里**逐字在场**（%d 处锚点）"
              % (specs.MDE_COEFFICIENT_AT_POWER_80, len(rows)))

    urows = specs.upstream_check(specs.REPO_ROOT)
    if urows is None:
        skip("C", "C.4", "上游文档不在场 ⇒ 契约逐字核对 SKIP")
    else:
        ubad = [(l, g, w) for (l, g, w) in urows if g < w]
        check("C", "C.4", not ubad,
              "上游契约 %d 处**逐字在场**（含 `10/03:303` 给本主题的指名指派）" % len(urows))

    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_latency_bandwidth.py", "02_reading_counters.py",
                       "03_diagnosis_walkthrough.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    print("     %s" % ", ".join("%s=%r" % (k, v) for k, v in sorted(marks.items())))
    check("C", "C.3",
          all(v is not None for v in marks.values())
          and len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK,
          "★ 失败标记**五个文件全部在场且一致**：%r" % (uniq,))


# ===========================================================================
def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "microbench.py",
                                          "meta_falsify.py", "mutate.py")]
    bad = []
    for p in files:
        if not p.exists():
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                bad += [(p.name, a.name) for a in n.names
                        if a.name.split(".")[0] in ("torch", "triton", "numba", "cuda")]
            elif isinstance(n, ast.ImportFrom) and n.module:
                if n.module.split(".")[0] in ("torch", "triton", "numba", "cuda"):
                    bad.append((p.name, n.module))
    print("AST 检查：%d 个文件，import 目标框架的 %d 个" % (len(files), len(bad)))
    for b in bad:
        print("   !! %s: %s" % b)
    return 1 if bad else 0


def main():
    if "--lint" in sys.argv:
        return lint()
    print("=" * 74)
    print("03_diagnosis_walkthrough.py —— 一次完整的诊断：四步，顺序不能乱")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 第 ④ 步**代入** `45/02` 的 δ*，**不重推**（正文在 `45`，此处回链）。")
    print("  ★ 测量条件：%s / python %s / numpy %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], np.__version__, sys.platform))
    par = params()
    print("  ★ 档位：%s" % ("**快档**（PERF_QUICK=1，诊断工具用）" if par["quick"]
                             else "默认档（正文引用的就是这一档）"))
    clang = microbench.require_clang()
    doc_dir = specs.find_doc_dir(verbose=True)

    tmpdir = microbench.make_tmpdir("perf03b_")
    try:
        bpaths = microbench.build_all(tmpdir, clang=clang)
        res = part_A(bpaths, par)
        part_B(bpaths, res)
        part_C(doc_dir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
