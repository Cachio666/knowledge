#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""meta_falsify.py —— **元检验**（`make falsify`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 它回答什么问题
===============================================================================

本主题讲的是「**σ 与 τ 怎么量、噪声下怎么决策**」。于是有一个致命质疑：

  > **如果那台「仪器」本身坏了呢？它还会报「通过」吗？**

本域最危险的错误模式是 **P13（量法口径错，伪装成关于被测对象的发现）**：
**测量者自己量错了，会产出一条关于被测对象的假发现**；而且它比「结论错」隐蔽 ——
结论错了复核者重推一遍能发现，**量法错了，「原始输出」看起来完全正常**。

本文件用**最直接的方式**回答它：**往测量引擎里注入缺陷，看断言会不会叫。**

===============================================================================
★ 五个注入项（**逐条对应正文里的一个承重结论**；★ 与前两个主题的**都不同**）
===============================================================================

  | # | 注入的缺陷 | 作用于 | **期望结果** |
  |---|---|---|---|
  | 1 | 把 `τ_q` 换成写死的常数 **250 ns**（= 计划 v1 已撤回的那个错值） | 01 | **失败** |
  | 2 | 把重复次数数组 `B_NS` 砍到只剩 `(1,)` | 01 | **失败** |
  | 3 | 把 `Measurer.measure()` 的噪声**关掉**（直接返回解析真值） | 02 | **失败** |
  | 4 | 把 B 段的**交错编排改回顺序编排**（= 把「修好」的那一侧改坏） | 03 | **失败** |
  | 5 | 采样标记自检被破坏（`ASSERT_FAIL_MARK` 改错） | 三个 | **失败** |

===============================================================================
★ 与计划 §3.0 的**两处偏差**（实施者实测后报告，**未就地修正**）
===============================================================================

  **偏差一：注入项 4 的靶子。** 计划 §3.0 写的是「**关掉 B 段的周期性干扰线程** |
  03（断言①失败）」。实施者实测：**那个周期性干扰本来就不是机制** ——
  `03` 的 `B.5` 用**独立的实测**证明了「干扰开着时交错编排照样准」。
  ⇒ 关掉它**不会**让任何断言失败（它是被否掉的那一项，不是承重的那一项）。
  **本文件改用真正的承重项**：把交错编排改回顺序编排 ⇒ `B.1`（顺序/交错之比 > 4）
  立刻失败。**这是偏差，不是修正** —— 计划把「被否掉的东西」当成了靶子。

  **偏差二：注入项 2 的靶子从「断言②③④」变成 `B.1`/`B.3`。**
  本主题 `01` 把 D-4b 的曲线重排成了 B.1–B.4（左端偏差 / 固定开销 / 谷底位置 / 右端回升），
  编号与计划不是一一对应。注入后 `B.1`（N=1 与平台之比）与 `B.3`（谷底在内部）失败。
  ★ 同时给 `01` 的 `part_B` 加了**空集保护**：否则 `B.1` 会抛 `ValueError`
  （= 崩溃捕获）而不是干净地断言失败。

  **偏差三（立项理由）**：计划 §3.0 的注入项 1 写的是「把
  `get_clock_info("perf_counter").resolution` 换成写死的常数」——本文件照做，
  **但写死的值取 250 ns**（而不是随便一个数），因为**那正是计划 v1 撤回过的错值**：
  于是这一条同时检验「程序能不能抓住本域最该引以为戒的那次 P13」。

===============================================================================
★ 四条硬纪律
===============================================================================

1. **不修改**仓库里的任何原文件 —— 每个变异体都在自己的 `tempfile` 目录里跑，
   并在收尾时校验一次 sha256。
2. **基线必须全过**：不打补丁时三个编号程序必须都退出 0，否则读数毫无意义。
3. **锚点必须恰好出现 1 次**：锚点漂了（`src.count(old) != 1`）补丁打不上，
   那会让元检验**静默变成空操作** —— 这里直接把它记成失败。
4. **临时目录里要放齐全部支撑模块**（含 `specs.py` 与 `mini_search.py`）——
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

PROGRAMS = ("01_sigma_tau.py", "02_noise_and_search.py", "03_ab_orchestration.py")
AUX = ("specs.py", "mini_search.py", "meta_falsify.py", "mutate.py")

# ★★ 失败标记：本主题（`tuning-practice`）自定的字面形态是 `>>> FAIL <<<`。
#   ★ 它**不是**照抄任何别的主题的：`30-gpu-programming/cutlass` 用 `[FAIL]`，
#     `45-kernel-lifecycle/convergence-and-tuning` 用 `[FAIL]|✗ N 条断言失败`，
#     同域 `50-performance/profiling-tools` 用 `**FAIL**`、
#     `50-performance/bottleneck-diagnosis` 用 `*** FAIL ***`。抄它们都会匹配到。
ASSERT_FAIL_MARK = ">>> FAIL <<<"
# ★★ 找「证据行」时必须**行首锚定**：编号程序**成功时也会打印这个字面量**
#   （标记自检那一条的 detail 里就写着它），用 `ASSERT_FAIL_MARK in line`
#   会把一条**通过**的行当成失败证据。判决本身用的是**退出码**，这里只影响证据展示，
#   但证据看错了与判决看错了是同一类错（`CONTRIBUTING:280`）。
ASSERT_FAIL_LINE = re.compile(r"(?m)^\s*" + re.escape(ASSERT_FAIL_MARK))

# (编号, 说明, 被改的文件, 原文, 替换成什么, 期望失败的程序, 期望结果)
DEFECTS = [
    ("D1", "把 `τ_q` 换成写死的常数 **250 ns**（= 计划 v1 已撤回的那个错值）",
     "01_sigma_tau.py",
     "    tau_q = info.resolution",
     "    tau_q = 250e-9      # DEFECT: 换成了已撤回的错值",
     ["01_sigma_tau.py"], "fail"),

    ("D2", "把重复次数数组 `B_NS` 砍到只剩 `(1,)`（曲线跑不出来）",
     "01_sigma_tau.py",
     "B_NS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)",
     "B_NS = (1,)      # DEFECT: 重复次数数组被砍到只剩 1",
     ["01_sigma_tau.py"], "fail"),

    ("D3", "把 `Measurer.measure()` 的**噪声关掉**（直接返回解析真值 = `60` 域的做法）",
     "mini_search.py",
     "        if not self.noise:\n            return self.true_seconds(x)",
     "        if True:      # DEFECT: 噪声被关掉\n            return self.true_seconds(x)",
     ["02_noise_and_search.py"], "fail"),

    ("D4", "把 B 段的**交错编排改回顺序编排**（把「修好」的那一侧改坏）",
     "03_ab_orchestration.py",
     "    intl = [inter_round(B_N) for _ in range(B_ROUNDS)]",
     "    intl = [seq_round(B_N) for _ in range(B_ROUNDS)]   # DEFECT: 交错被改成顺序",
     ["03_ab_orchestration.py"], "fail"),

    ("D5", "采样标记自检被破坏（`ASSERT_FAIL_MARK` 改错）",
     "01_sigma_tau.py",
     'ASSERT_FAIL_MARK = ">>> FAIL <<<"',
     'ASSERT_FAIL_MARK = "FAILED:"      # DEFECT: 标记改错',
     ["01_sigma_tau.py", "02_noise_and_search.py",
      "03_ab_orchestration.py"], "fail"),
]

# ★ 子进程的 env：`PERF_QUICK=1` 走快档（**断言全是结构性的，走同一条代码路径**），
#   否则每次注入都要等 `03` 的二十几秒受测循环 + A 段的负载，元检验会慢到不可用。
PERF_QUICK = True
RUN_TIMEOUT = 900


def child_env():
    env = dict(os.environ)
    if PERF_QUICK:
        env["PERF_QUICK"] = "1"
    return env


def env_report():
    """报出三个编号程序各自需要的**外部工具**在不在（**不静默跳过**）。"""
    return [("python3", sys.executable)]


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
    print("  ★ 本主题讲的是「σ 与 τ 怎么量、噪声下怎么决策」，所以本文件回答的质疑是：")
    print("    **如果那台用来量 σ 与 τ 的仪器本身坏了呢？它还会报「通过」吗？**")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自己的输出形态，**不是**照抄别域）"
          % ASSERT_FAIL_MARK)
    print("  ★ 子进程档位：%s" % ("PERF_QUICK=1（快档）" if PERF_QUICK else "默认档"))
    print()
    print("  ── 外部工具 ──")
    for name, path in env_report():
        print("     %-8s %s" % (name, path or "**不在**"))
    print("     ★ 本主题的三个编号程序**不需要任何外部工具**（纯 python3 + 标准库）：")
    print("       `01`/`02` 只读时钟；`03` 的 A 段会**临时拉起** `cpu_count()-1` 个忙进程")
    print("       并在结束时全部终止。没有 clang / sample / 网络依赖。")
    print()

    files = AUX + PROGRAMS
    before = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
              for f in files if os.path.exists(os.path.join(HERE, f))}
    env = child_env()

    # ---- 步骤 2/4：基线 ----
    print("[步骤 2/4] 基线（不打补丁）")
    with tempfile.TemporaryDirectory(prefix="tp_base_") as tmp:
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
        with tempfile.TemporaryDirectory(prefix="tp_def_") as tmp:
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
        print("★ 计划 §3.0 写注入项 4 是「关掉 B 段的周期性干扰线程」；")
        print("  实测**那不是承重项**（`B.5` 本来就证明干扰不是机制），已改用")
        print("  「把交错编排改回顺序编排」——见模块头「与计划 §3.0 的两处偏差」。")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
