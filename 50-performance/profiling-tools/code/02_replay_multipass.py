#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_replay_multipass.py —— 比值型指标的分子与分母可能来自**不同的执行**。

配套文档：`02-replay-and-multipass.md`。★ 三类断言（A/B/C）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` / `python3 3.13.5`），
> **不是 GPU、不是 Nsight Compute**。GPU 侧的那条因果链（硬件计数器有限 → replay →
> 跨 pass 取分子分母）是**规格事实**（`specs.FACTS` 的 F1–F4/F10，A 级），
> 本程序**不试图在 CPU 上复现它**——它复现的是**那条链的算术后果**：
> **分子来自一次执行、分母来自另一次执行时，那个比值不是任何一次执行的测量结果。**

★★ 本程序的可证伪断言：

  · `A.1`–`A.3` 三次**各自真实**的执行，hits/queries 恰好是规格里那三组数；
  · `A.4` **四个值两两互不相同**（28.5714% / 0.0244% / 0.0976% / 7.1429%）；
  · `A.5` **两个交叉值都不是任何一次真实执行的测量结果** ← 本节的核心断言；
  · `A.6` **400% > 100%**：程序**不 clamp**（对 F10 的正面回答）；
  · `B.2` **同一个 pass 跑 20 次，「真实」hit rate 就不止一个值**（F3 逐字的 `spin loops`）；
  · `C.1` 失败标记与 01/03 及两个诊断工具一致。

用法：
    python3 02_replay_multipass.py            # 跑 A/B/C 三类断言
    python3 02_replay_multipass.py --lint     # AST 扫描：不 import torch/triton/numba/cuda
"""

import ast
import os
import sys
import time

import specs                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（profiling-tools）自定的字面形态是 `**FAIL**`，
#   **不是** `30-gpu-programming/cutlass` 的 `[FAIL]`，也**不是**
#   `45-kernel-lifecycle/convergence-and-tuning` 的 `[FAIL]|✗ N 条断言失败`。
#   照抄任何一个都会让「断言捕获」与「崩溃捕获」的区分失效。
ASSERT_FAIL_MARK = "**FAIL**"

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
# 1. 状态依赖工作负载：**自旋查找**（F3 逐字点名的 `spin loop`）
# ===========================================================================
def make_ring(target_slots):
    """造一个环形数组：默认全 0，只有 `target_slots` 上是 `TARGET_KEY`。"""
    arr = bytearray(specs.N_SLOTS)
    for s in target_slots:
        arr[s] = specs.TARGET_KEY
    return bytes(arr)


def spin_lookup(arr, stop_path, cap, start_path=0):
    """在环形数组上**自旋探测**，返回 `(hits, queries)`。

    · 消费者从 `start_path` 起按**路径位置** `p` 推进（**每探一个位置记 1 次 query**）；
    · `arr[p % len(arr)]` 等于 `TARGET_KEY` 时记 **1 次 hit**；
    · 路径位置越过 `stop_path`，或 query 数达到 `cap`，就停。

    ★ 这是**状态依赖**的：要探多少次，取决于「目标在路径的哪个位置」这个状态。
    ★ **`p % len(arr)` 就是「自旋」的字面含义**：冷 pass 的 `stop_path` 越过数组末尾
      ⇒ 消费者**转进第二圈**（F3：「workload is inherently variable, as e.g. in the
      case of spin loops」）。
    ★ 计数里**没有任何 clamp** —— 这是对 F10 的正面回答（工具「故意不 clamp，
      以保证报告其余部分自洽」；本程序照做，并**把超范围的值原样打印出来**）。
    """
    n = len(arr)
    hits = queries = 0
    p = start_path
    while p <= stop_path and queries < cap:
        if arr[p % n] == specs.TARGET_KEY:
            hits += 1
        queries += 1
        p += 1
    return hits, queries


def run_passes():
    """按 `specs.PASSES` 跑三次**各自真实**的执行，返回 `{名称: 结果}`。"""
    out = {}
    for (name, slots, stop_path, cap) in specs.PASSES:
        arr = make_ring(slots)
        t0 = time.perf_counter()
        hits, queries = spin_lookup(arr, stop_path, cap)
        dt = time.perf_counter() - t0
        out[name] = {"hits": hits, "queries": queries, "seconds": dt,
                     "slots": slots, "stop_path": stop_path, "cap": cap,
                     "rate": hits / queries if queries else float("nan")}
    return out


# ★★ 跨执行组合的**配对表**：分子取自前者那次执行、分母取自后者那次执行。
#   ★ 同一次执行的配对（分子分母来自同一个 pass）**不是**跨 pass 组合 ——
#     那是一次**真实**的测量。`A.5` 是**守护式**断言：只对「两次执行不同」的行生效。
#   ★ 这一点由 `make falsify` 的注入项 3 **反向验证**：把下面两行的配对改成同一次执行，
#     断言被它的前置条件挡掉 ⇒ 程序**照样退出 0**。若改成同一次执行后程序反而失败，
#     说明 `A.5` 被写成了无条件断言（那它就脱离了「跨 pass」这个机制）。
CROSS_PAIRING = (("hot", "cold"), ("cold", "hot"))


# ===========================================================================
# 2. A 类：跨执行取分子分母
# ===========================================================================
def part_A(res):
    print("=" * 74)
    print("A 类 · 比值型指标跨执行组合：那个比值不是任何一次执行的测量结果")
    print("=" * 74)
    print("  ★ 真值标注：本程序数字**全部是 CPU 上的本机实测值**，不是 GPU。")
    print("  ★ 被测负载：环形数组（%d 槽）上的**自旋查找** —— F3 逐字点名的那一类" % specs.N_SLOTS)

    print()
    print("  ── 三次**各自真实**的执行 ──")
    print("     %-6s %-34s %-8s %-8s %-10s %s"
          % ("pass", "目标槽位 / 停止路径 / cap", "hits", "queries", "真实 hit rate", "时长"))
    for name in ("hot", "cold", "tiny"):
        r = res[name]
        print("     %-6s %-34s %-8d %-8d %-10s %.6f s"
              % (name, "%s / %d / %d" % (r["slots"], r["stop_path"], r["cap"]),
                 r["hits"], r["queries"], "%.4f%%" % (100.0 * r["rate"]), r["seconds"]))

    exp = {"hot": (4, 14), "cold": (1, 4100), "tiny": (1, 1)}
    for name in ("hot", "cold", "tiny"):
        h, q = exp[name]
        r = res[name]
        check("A", {"hot": "A.1", "cold": "A.2", "tiny": "A.3"}[name],
              (r["hits"], r["queries"]) == (h, q),
              "`%s` 的真实测量 = %d/%d = %.4f%%（期望 %d/%d = %.4f%%）"
              % (name, r["hits"], r["queries"], 100.0 * r["rate"], h, q, 100.0 * h / q))

    real = {"hot": res["hot"]["rate"], "cold": res["cold"]["rate"]}
    print()
    print("  ── 真实值（各自执行内取分子分母）──")
    for k, v in real.items():
        print("     %-6s %.6f%%" % (k, 100.0 * v))

    print()
    print("  ── 按 F3 的机制**跨执行**取分子分母（分子来自前一次、分母来自后一次）──")
    cross = {}
    for (num, den) in CROSS_PAIRING:
        if num == den:
            # ★ 守护：同一次执行 ⇒ 那是一次**真实**测量，本断言不适用（见 CROSS_PAIRING 的注）
            print("     (%s/%s) 分子分母来自**同一次**执行 ⇒ 真实测量，不在跨 pass 组合内"
                  % (num, den))
            continue
        v = res[num]["hits"] / res[den]["queries"]
        cross[(num, den)] = v
        print("     %-6s hits(%d) / %-6s queries(%d) = %.6f%%"
              % (num, res[num]["hits"], den, res[den]["queries"], 100.0 * v))

    four = sorted(set([real["hot"], real["cold"],
                       res["hot"]["hits"] / res["cold"]["queries"],
                       res["cold"]["hits"] / res["hot"]["queries"]]))
    check("A", "A.4", len(four) == 4,
          "★ **四个值两两互不相同**：%s ⇒ 跨执行组合**造出了两个新数**"
          % " / ".join("%.4f%%" % (100.0 * v) for v in four))

    real_set = set(real.values())
    bad = {k: v for k, v in cross.items() if v in real_set}
    check("A", "A.5", not bad,
          "★ **%d 对跨执行组合里，没有一个等于任何一次真实执行的测量结果**"
          "（真实值集合 %s）"
          % (len(cross), ", ".join("%.4f%%" % (100.0 * v) for v in sorted(real_set))))

    over = res["hot"]["hits"] / min(r["queries"] for r in res.values())
    print()
    print("  ── 超范围的构造（F10：工具**故意不 clamp**）──")
    print("     热 hits(%d) / 最小 pass 的 queries(%d) = **%.0f%%**"
          % (res["hot"]["hits"], min(r["queries"] for r in res.values()), 100.0 * over))
    print("     ★ 物理上不可能，而本程序**照原样打印、不做 clamp** ——")
    print("       这正是 F10 要你看到的形态：%s" % specs.FACTS[9]["must_contain"][1])
    check("A", "A.6", over > 1.0,
          "★ 跨执行组合可以造出 **%.0f%% > 100%%**（比值超出物理范围，程序不 clamp）"
          % (100.0 * over))


# ===========================================================================
# 3. B 类：每个 pass 的执行条件都不同
# ===========================================================================
VAR_RUNS = 20
VAR_SLOTS = (100, 900, 1700, 2500, 3300, 3900)


def part_B(res):
    print()
    print("=" * 74)
    print("B 类 · 每个 pass 的执行条件都不同 ⇒ 两个 pass 的数不可比")
    print("=" * 74)
    print("  ★ F3 逐字：steady state **generally > 20 µs** —— 没到稳态的 kernel，")
    print("    跨 pass 误差显著。本程序没有 GPU，所以这里量的是**本机 CPU 上**")
    print("    三次执行的**时长差**：它们相差几个数量级 ⇒ 执行条件根本不同。")

    d = {k: res[k]["seconds"] for k in res}
    for k, v in d.items():
        print("     %-6s 时长 %.9f s  (hits=%d, queries=%d)"
              % (k, v, res[k]["hits"], res[k]["queries"]))
    ratio = d["cold"] / max(d["tiny"], 1e-12)
    check("B", "B.1", ratio > 100.0,
          "★ 同一个「热点」的两次执行，时长差 **%.0f 倍**（cold %.6f s vs tiny %.9f s）"
          "⇒ 它们**不是同一次执行的两次采样**" % (ratio, d["cold"], d["tiny"]))

    # ---- F3 的第二句：工作量本身可变（spin loop） ----
    print()
    print("  ── F3 的第二句：「workload is inherently variable, as e.g. in the case of "
          "spin loops」 ──")
    print("     同一个 pass（同一份代码、同一个数组构造方式），只让**目标槽位随运行变化**，")
    print("     跑 %d 次，看「真实」的 hit rate 变不变：" % VAR_RUNS)
    rates, qs = [], []
    for i in range(VAR_RUNS):
        slot = VAR_SLOTS[i % len(VAR_SLOTS)] + (i // len(VAR_SLOTS))
        arr = make_ring((slot,))
        h, q = spin_lookup(arr, slot, specs.N_SLOTS)
        rates.append(h / q if q else float("nan"))
        qs.append(q)
    print("     queries 取值集合（前 8 个）= %s" % sorted(set(qs))[:8])
    print("     hit rate 最小 %.6f%% · 最大 %.6f%% ⇒ 极差 %.1f 倍"
          % (100.0 * min(rates), 100.0 * max(rates), max(rates) / min(rates)))
    check("B", "B.2", len(set(qs)) >= 5 and (max(rates) / min(rates)) > 3.0,
          "★ **同一个 pass 的「真实」hit rate 就不止一个值**：queries 有 %d 个不同取值，"
          "hit rate 极差 **%.1f 倍**（%d 次运行）"
          % (len(set(qs)), max(rates) / min(rates), VAR_RUNS))
    print("     ★ ⇒ 「工作量本身可变」时连**单次执行内部**的比值都不稳，"
          "跨执行组合只会更不可靠。")

    print()
    print("  ── 每个 pass 的查询序列都是**确定**的（这一点也要写清楚，别把两件事混起来）──")
    same = all(spin_lookup(make_ring(sl), sp, cp) == spin_lookup(make_ring(sl), sp, cp)
               for (_n, sl, sp, cp) in specs.PASSES)
    check("B", "B.3", same,
          "同一份输入下 `spin_lookup` 是**确定性**的（可复现）——")
    print("     所以上面 B.2 的变化来自**状态**（目标位置），不是来自随机数。")


# ===========================================================================
# 4. C 类：specs 核对 + 标记自检
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


def part_C():
    print()
    print("=" * 74)
    print("C 类 · 规格事实核对与标记自检")
    print("=" * 74)
    f3 = specs.FACTS[2]
    check("C", "C.1", "20 µs" in f3["quote"] and "spin loops" in f3["quote"],
          "F3 的引文逐字含「generally > 20 µs」与「spin loops」（两者都是本节的承重句）")
    check("C", "C.2",
          all(m in specs.FACTS[9]["quote"] for m in specs.FACTS[9]["must_contain"]),
          "F10 的引文逐字含「%s」" % specs.FACTS[9]["must_contain"][1][:56])
    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_collection_mechanisms.py", "02_replay_multipass.py",
                       "03_defaults_attribution.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    check("C", "C.3",
          all(v is not None for v in marks.values())
          and len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK,
          "★ 失败标记**五个文件全部在场且一致**：%s" % ", ".join("%r" % v for v in uniq))


# ===========================================================================
def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "meta_falsify.py", "mutate.py")]
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
    print("02_replay_multipass.py —— 比值型指标的分子分母可能来自不同的执行")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 测量条件：%s / python %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], sys.platform))
    res = run_passes()
    part_A(res)
    part_B(res)
    part_C()
    return summary()


if __name__ == "__main__":
    sys.exit(main())
