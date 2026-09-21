#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""meta_falsify.py —— **元检验**（`make falsify`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 它回答什么问题
===============================================================================

本主题讲的是「**拿到一个数之后，怎么判断卡在哪**」。于是有一个致命质疑：

  > **如果那台「仪器」本身坏了呢？它还会报「通过」吗？**

更具体地说，本域最危险的错误模式是 **P13（量法口径错，伪装成关于被测对象的发现）**：
**测量者自己量错了，会产出一条关于被测对象的发现**；而且它比「结论错」隐蔽——
结论错了复核者重推一遍能发现，**量法错了，「原始输出」看起来完全正常**。

本文件用**最直接的方式**回答它：**往测量引擎里注入缺陷，看断言会不会叫。**

  · **期望「失败」**的：改坏它之后，那条断言**必须**让程序非零退出；
    若程序照样退出 0，说明那条断言是**空操作**，本文件把它标成
    `**没失败，这条断言无效**` 并计一次失败。

===============================================================================
★ 五个注入项（**逐条对应正文里的一个承重结论**；★ 与同域 `profiling-tools` 的**不同**）
===============================================================================

  | # | 注入的缺陷 | 作用于 | **期望结果** |
  |---|---|---|---|
  | 1 | 把 C 微基准的**累加器个数写死为 1** | 01 | **失败** |
  | 2 | 把 `clang -O2` 改成 `-O0` | 01 | **失败** |
  | 3 | 把「归因」断言改成「符号名**出现**在输出里」（而非出现 **0** 次） | 02 | **失败** |
  | 4 | 把 δ\* 的系数 `2.801585218112968` 改错 | 03 | **失败** |
  | 5 | 采样标记自检被破坏（`ASSERT_FAIL_MARK` 改错） | 三个 | **失败** |

===============================================================================
★ 与计划 §2.0 的**两处偏差**（实施者实测后报告，**未就地修正**）
===============================================================================

  **偏差一：注入项 2 的实现方式。** 计划 §2.0 写的是
  「去掉 `volatile`，**或**把 `-O2` 改成 `-O0`」。实施者实测：
  **去掉 `volatile` 在本机 clang 21.0.0 上并不触发死代码消除** ——
  `dep_sink` 只被写、从不被读，但 `-O2` **仍然保留了整个循环**
  （实测 K=1 的耗时 30.1 ms → 30.0 ms，曲线一点没变，比值仍是 16.2 倍）。
  ⇒ 本文件取计划的**后一个选项**：`-O2` → `-O0`（实测 K=32/K=1 从 16.7 掉到 **3.0**，
    `01` 的 `A.2` 立刻失败）。**这是偏差，不是修正**：计划的「或」两字让它可选，
    但**实测只有其中一支真的打坏了仪器**，另一支是空操作。

  **偏差二：`microbench.py` 不在编号程序名单里。** 它不是编号程序（无对应文档），
  但注入项 1/2 的靶子都在它里面。本文件把它列进 `AUX`（复制进每个副本），
  并允许注入项指向它 —— 否则「往测量引擎里注入缺陷」这句话就是空的。

===============================================================================
★ 四条硬纪律
===============================================================================

1. **不修改**仓库里的任何原文件 —— 每个变异体都在自己的 `tempfile` 目录里跑，
   并在收尾时校验一次 sha256。
2. **基线必须全过**：不打补丁时三个编号程序必须都退出 0，否则读数毫无意义。
3. **锚点必须恰好出现 1 次**：锚点漂了（`src.count(old) != 1`）补丁打不上，
   那会让元检验**静默变成空操作** —— 这里直接把它记成失败。
4. **临时目录里要放齐全部支撑模块**（含 `specs.py` 与 `microbench.py`）——
   编号程序要 import 它们，副本里少一个，基线就会 rc≠0，整个元检验的读数作废。
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

PROGRAMS = ("01_latency_bandwidth.py", "02_reading_counters.py",
            "03_diagnosis_walkthrough.py")
AUX = ("specs.py", "microbench.py", "meta_falsify.py", "mutate.py")

# ★★ 失败标记：本主题（`bottleneck-diagnosis`）自定的字面形态是 `*** FAIL ***`。
#   ★ 它**不是**照抄任何别的主题的：`30-gpu-programming/cutlass` 用 `[FAIL]`，
#     `45-kernel-lifecycle/convergence-and-tuning` 用 `[FAIL]|✗ N 条断言失败`，
#     同域 `50-performance/profiling-tools` 用 `**FAIL**`。抄它们都会匹配到。
ASSERT_FAIL_MARK = "*** FAIL ***"
# ★★ 找「证据行」时必须**行首锚定**：编号程序**成功时也会打印这个字面量**
#   （标记自检那一条的 detail 里就写着它），用 `ASSERT_FAIL_MARK in line`
#   会把一条**通过**的行当成失败证据。判决本身用的是**退出码**，这里只影响证据展示，
#   但证据看错了与判决看错了是同一类错（`CONTRIBUTING:280`）。
ASSERT_FAIL_LINE = re.compile(r"(?m)^\s*" + re.escape(ASSERT_FAIL_MARK))

# (编号, 说明, 被改的文件, 原文, 替换成什么, 期望失败的程序, 期望结果)
DEFECTS = [
    ("D1", "把 C 微基准的**累加器个数写死为 1**（依赖链曲线塌平）",
     "microbench.py",
     "        for (size_t i = 0; i + (size_t)K <= n; i += (size_t)K) {\n"
     "            for (int k = 0; k < K; k++) s[k] += a[i + (size_t)k];\n"
     "        }",
     "        for (size_t i = 0; i < n; i++) {\n"
     "            s[0] += a[i];   /* DEFECT: 累加器个数被写死为 1 */\n"
     "        }",
     ["01_latency_bandwidth.py"], "fail"),

    ("D2", "把 `clang -O2` 改成 `-O0`（并发度被打回内存：K=32/K=1 从 16.7 掉到 3.0）",
     "microbench.py",
     'DEFAULT_OPT = "O2"',
     'DEFAULT_OPT = "O0"      # DEFECT: -O2 → -O0',
     ["01_latency_bandwidth.py"], "fail"),

    ("D3", "把「归因」断言改成「符号名**出现**在输出里」（而非出现 **0** 次）",
     "02_reading_counters.py",
     "all(v == 0 for v in own_hits.values()),",
     "all(v > 0 for v in own_hits.values()),   # DEFECT: 断言写反了",
     ["02_reading_counters.py"], "fail"),

    ("D4", "把 δ* 的系数 `2.801585218112968` 改错（它是**从 45 代入**的）",
     "specs.py",
     # ★ 锚点必须把**定义那一段的上下文**带上：这个常数在 `MDE_ANCHORS` 的引文里
     #   也出现过一次（那是**有意的**：`mde_coefficient_selfcheck()` 就靠它钉住转写），
     #   所以裸的 `NAME = <数>` 在文件里出现 **2 次**，补丁会打不上（元检验直接记失败）。
     "MDE_COEFFICIENT_AT_POWER_80 = 2.801585218112968\n\nMDE_ANCHORS = (",
     "MDE_COEFFICIENT_AT_POWER_80 = 4.0    # DEFECT: 系数被改错\n\nMDE_ANCHORS = (",
     ["03_diagnosis_walkthrough.py"], "fail"),

    ("D5", "采样标记自检被破坏（`ASSERT_FAIL_MARK` 改错）",
     "01_latency_bandwidth.py",
     'ASSERT_FAIL_MARK = "*** FAIL ***"',
     'ASSERT_FAIL_MARK = "FAILED:"      # DEFECT: 标记改错',
     ["01_latency_bandwidth.py", "02_reading_counters.py",
      "03_diagnosis_walkthrough.py"], "fail"),
]

# ★ 子进程的 env：`PERF_QUICK=1` 走快档（**断言全是结构性的，走同一条代码路径**），
#   否则每次注入都要等 `03` 的十几轮受测循环，元检验会慢到不可用。
PERF_QUICK = True
RUN_TIMEOUT = 900


def child_env():
    env = dict(os.environ)
    if PERF_QUICK:
        env["PERF_QUICK"] = "1"
    return env


def env_report():
    """报出三个编号程序各自需要的**外部工具**在不在（**不静默跳过**）。"""
    return [("clang", shutil.which(os.environ.get("CLANG") or "clang")),
            ("sample", shutil.which("sample"))]


def run_case(tmp, which, patches, env):
    """把全部支撑模块复制到 `tmp`，按 `patches` 打补丁，逐个程序跑。"""
    for f in AUX + PROGRAMS:
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tmp, f))
    for (fn, old, new) in patches:
        path = os.path.join(tmp, fn)
        text = open(path, encoding="utf-8").read()
        if text.count(old) != 1:
            return None, "锚点在 %s 里出现 %d 次（必须恰好 1 次）" % (fn, text.count(old))
        open(path, "w", encoding="utf-8").write(text.replace(old, new))
    out = {}
    for p in which:
        try:
            r = subprocess.run([sys.executable, p], cwd=tmp, capture_output=True,
                               text=True, env=env, timeout=RUN_TIMEOUT)
            out[p] = (r.returncode, r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            out[p] = (99, "超时 > %ds" % RUN_TIMEOUT)
    return out, None


def main():
    print("=" * 78)
    print("元检验：往**测量引擎**里注入缺陷，断言编号程序按「期望结果」反应")
    print("=" * 78)
    print("  ★ 本主题讲的是「拿到数之后怎么判」，所以本文件回答的质疑是：")
    print("    **如果那台判据所用的仪器本身坏了呢？它还会报「通过」吗？**")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自己的输出形态，**不是**照抄别域）"
          % ASSERT_FAIL_MARK)
    print("  ★ 子进程档位：%s" % ("PERF_QUICK=1（快档）" if PERF_QUICK else "默认档"))
    print()
    print("  ── 外部工具 ──")
    for name, path in env_report():
        print("     %-8s %s" % (name, path or "**不在**"))
    print("     ★ `clang` 不在 ⇒ 编号程序会以 rc=2 退出（**不回落、不静默跳过**），")
    print("       那时基线会红；本文件会把它报成「跑不起来」，不是「元检验失败」。")
    if not shutil.which("sample"):
        print("     ⚠ `sample` 不在 ⇒ `02` 的 A 类断言 SKIP（**不计入通过**），基线仍能过；")
        print("       此时注入项 D3 的靶子（`A.3`「自己的函数出现 0 次」）不会被触及。")
    print()

    files = AUX + PROGRAMS
    before = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
              for f in files if os.path.exists(os.path.join(HERE, f))}
    env = child_env()

    # ---- 步骤 2/4：基线 ----
    print("[步骤 2/4] 基线（不打补丁）")
    with tempfile.TemporaryDirectory(prefix="pt_base_") as tmp:
        base, err = run_case(tmp, PROGRAMS, [], env)
    if base is None:
        print("!! [步骤 2/4] %s" % err)
        return 2
    bad = [p for p, (rc, _) in base.items() if rc != 0]
    n_fail = 0
    if bad:
        print("     基线：**有程序失败** → %r" % bad)
        for p in bad:
            print("     %s 的尾部输出：" % p)
            for line in base[p][1].strip().splitlines()[-6:]:
                print("        %s" % line[:110])
        print("     ⇒ 先修基线，再跑注入项。**这不是元检验的结论。**")
        n_fail += 1
    else:
        tot = sum(int(m.group(1)) for m in
                  re.finditer(r"断言 (\d+) 条", "".join(v[1] for v in base.values())))
        print("     基线：三个编号程序全部退出 0 ✓（断言总数约 %d 条）" % tot)
    print()

    # ---- 步骤 3/4：注入项 ----
    print("[步骤 3/4] %d 个注入项（每个都在自己的副本里跑）" % len(DEFECTS))
    n_ok = 0
    for (code, name, fn, old, new, progs, expect) in DEFECTS:
        src = open(os.path.join(HERE, fn), encoding="utf-8").read()
        print()
        print("[%s] %s" % (code, name))
        print("     改的是：%s    期望结果：**%s**" % (fn, expect.upper()))
        if src.count(old) != 1:
            print("     ★ 锚点在 %s 里出现 %d 次（必须恰好 1 次）→ **补丁无效**，记一次失败"
                  % (fn, src.count(old)))
            n_fail += 1
            continue
        with tempfile.TemporaryDirectory(prefix="pt_def_") as tmp:
            res, err = run_case(tmp, PROGRAMS, [(fn, old, new)], env)
        if res is None:
            print("     ★ %s" % err)
            n_fail += 1
            continue
        for p in PROGRAMS:
            rc, txt = res[p]
            mark = next((l.strip() for l in txt.splitlines()
                         if ASSERT_FAIL_LINE.match(l)), None)
            if p not in progs:
                print("     %-30s 退出 %-4d （未注入本缺陷）" % (p, rc))
                continue
            if expect == "fail":
                ok = rc != 0
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                print("     %-30s 退出 %-4d 期望失败 → %s"
                      % (p, rc, "OK" if ok else "**没失败，这条断言无效**"))
                if mark:
                    print("         证据（断言捕获）：%s" % mark[:104])
                elif rc != 0:
                    print("         ⚠ 退出非 0 但**没有失败标记** ⇒ 这是**崩溃捕获**，"
                          "不是断言捕获（两类必须分开数）")
            else:
                ok = rc == 0
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                print("     %-30s 退出 %-4d 期望**通过** → %s"
                      % (p, rc, "OK" if ok else "**反而失败了 ⇒ 那条断言写反了**"))

    # ---- 步骤 4/4：原文件完整性 ----
    after = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
             for f in files if os.path.exists(os.path.join(HERE, f))}
    changed = [f for f in before if before[f] != after.get(f)]

    print()
    print("=" * 78)
    if changed:
        print("!! **仓库原文件被改动了**：%r —— 读数不可信" % changed)
        n_fail += 1
    else:
        print("原文件完整性：跑前 / 跑后 sha256 全部相同 ✓（%d 个文件）" % len(before))
    if n_fail:
        print("元检验失败 %d 项：至少有一条断言是「只演示、不检查」" % n_fail)
    else:
        print("元检验通过：**%d 个注入项全部按「期望结果」反应**，且不打补丁时三个程序全过。"
              % len(DEFECTS))
        print("→ 「那台仪器自己坏掉会怎样」这个质疑，被这 %d 条反证正面回答。" % len(DEFECTS))
        print("★ 计划 §2.0 说注入项 2 是「去掉 volatile **或** -O2→-O0」；")
        print("  实测**只有后者**有效（去掉 volatile 时 clang 21.0.0 仍保留整个循环）——见模块头。")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
