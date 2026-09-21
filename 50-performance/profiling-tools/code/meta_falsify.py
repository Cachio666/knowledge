#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""meta_falsify.py —— **元检验**（`make falsify`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 它回答什么问题
===============================================================================

本主题是**一个专讲测量的域**：三个编号程序自己就是「仪器」。于是有一个致命质疑：

  > **如果那台「仪器」本身坏了呢？它还会报「通过」吗？**

更具体地说，本域最危险的错误模式是 **P13（量法口径错，伪装成关于被测对象的发现）**：
**测量者自己量错了，会产出一条关于被测对象的假发现**；而且它比「结论错」隐蔽——
结论错了复核者重推一遍能发现，**量法错了，「原始输出」看起来完全正常**。

本文件用**最直接的方式**回答它：**往测量引擎里注入缺陷，看断言会不会叫。**

  · **期望「失败」**的：改坏它之后，那条断言**必须**让程序非零退出；
    若程序照样退出 0，说明那条断言是**空操作**，本文件把它标成
    `**没失败，这条断言无效**` 并计一次失败。
  · **期望「通过」**的（★ **反向测试**，`D3`）：注入后程序**必须仍然退出 0**；
    若反而失败，说明那条断言**写反了**（脱离了它要守护的那个机制）。

===============================================================================
★ 六个注入项（**逐条对应正文里的一个承重结论**）
===============================================================================

  | # | 注入的缺陷 | 作用于 | **期望结果** |
  |---|---|---|---|
  | 1 | 采样标记自检被破坏（`ASSERT_FAIL_MARK` 改错） | 三个 | **失败** |
  | 2 | 插桩对照里去掉「不插桩」那一组 | 01 | **失败** |
  | 3 | 跨 pass 组合改成**同一次** pass 内取分子分母 | 02 | **通过**（反向测试） |
  | 4 | 默认值常量由 `all` 改成 `none` | 03 | **失败** |
  | 5 | 观测者效应的对照结构改回「两组 subprocess 不同」 | 03 | **失败** |
  | 6 | `clang -O2` 改成 `-O0` | 01 | **失败** |

★★ **`D3` 为什么期望「通过」——把这件事写清楚，因为它是本表唯一一行反的**：

  `02_replay_multipass.py` 的 `CROSS_PAIRING` 是一张**配对表**：分子取自哪次执行、
  分母取自哪次执行。`A.5` 是一条**守护式**断言，它只对「两次执行不同」的那些行生效
  （分母分子来自同一次执行 ⇒ 那是一次**真实**测量，不在跨 pass 组合的范围内）。

  `D3` 把配对表改成同一次执行的配对 ⇒ 守护条件成立 ⇒ `A.5` **被跳过** ⇒ 程序退出 0。
  **这正是期望的结果**：它证明 `A.5` 检查的是「**跨执行**」这个机制，
  而不是「数字看起来很怪」这种无条件的怪癖。
  **若 `D3` 让程序失败**，就说明 `A.5` 被写成了无条件断言 —— 那样它对
  **v1 那个失败的设计**（两个 pass 的 hits 都是 1，交叉组合恰好等于两个真实值）
  也一样无能为力：它检查的东西和它声称检查的东西不是同一个（P7）。

  ⚠ **本项的期望结果与计划 §1.5 的表一致**（v1 的表头只写「必须失败」，
    而表内第 3 行自己写着「反向测试」——两者互斥，v2 补上了「期望结果」列）。
    本文件把两个方向**分开计数**，所以「通过」不是「没发现问题」。

===============================================================================
★ 与计划 §1.5 的**一处偏差**（实施者报告，**未就地修正**）
===============================================================================

  计划 §1.5 的注入项 6 写的是「`clang -O2` 改成 `-O0`（或去掉 `volatile`）**作用于三个**」。
  **只有 `01_collection_mechanisms.py` 编译 C**：它内嵌 C 源码、用 `clang` 编译，
  正是为了复现「内联让函数边界消失」。
  `02` 的被测对象是**纯 Python 的状态依赖负载**，`03` 的是**纯 Python 子进程**，
  两者**都不碰 `clang`**——本主题「访存微基准用 C」那条约定属于 Chunk 2
  （`bottleneck-diagnosis`），不属于本主题。
  ⇒ 本文件把 `D6` 只打在 `01` 上，并在报告里写明这一点。
  **这是偏差，不是修正**：计划的表说「三个」，实测只有「一个」。

===============================================================================
★ 四条硬纪律
===============================================================================

1. **不修改**仓库里的任何原文件 —— 每个变异体都在自己的 `tempfile` 目录里跑，
   并在收尾时校验一次 sha256。
2. **基线必须全过**：不打补丁时三个编号程序必须都退出 0，否则读数毫无意义。
3. **锚点必须恰好出现 1 次**：锚点漂了（`src.count(old) != 1`）补丁打不上，
   那会让元检验**静默变成空操作** —— 这里直接把它记成失败。
4. **临时目录里要放齐全部支撑模块**（含 `specs.py`）—— 编号程序要 import 它，
   副本里少一个，基线就会 rc≠0，整个元检验的读数作废。
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

PROGRAMS = ("01_collection_mechanisms.py", "02_replay_multipass.py",
            "03_defaults_attribution.py")
AUX = ("specs.py", "meta_falsify.py", "mutate.py")

# ★★ 失败标记：本主题三个程序统一打印 `  **FAIL** A.1 …`。
#   ★ 这条字面量**不是**照抄任何别的主题的：`30-gpu-programming/cutlass` 用 `[FAIL]`，
#     `45-kernel-lifecycle/convergence-and-tuning` 用 `[FAIL]|✗ N 条断言失败`。
#     抄它们**都会匹配到** —— 那正是本域禁止的「跨主题照抄」，
#     而且会让「断言捕获」与「崩溃捕获」的区分失效（cutlass 的 mutate.py 记着这个教训）。
ASSERT_FAIL_MARK = "**FAIL**"
# ★★ 找「证据行」时必须**行首锚定**：编号程序**成功时也会打印这个字面量**
#   （标记自检那一条的 detail 里就写着它），用 `ASSERT_FAIL_MARK in line`
#   会把一条**通过**的行当成失败证据。判决本身用的是**退出码**，这里只影响证据展示，
#   但证据看错了与判决看错了是同一类错（`CONTRIBUTING:280`）。
ASSERT_FAIL_LINE = re.compile(r"(?m)^\s*" + re.escape(ASSERT_FAIL_MARK))

# (编号, 说明, 被改的文件, 原文, 替换成什么, 期望失败的程序, 期望结果)
#   ★ `expect` 取 "fail" 或 "pass"（后者是**反向测试**）。
DEFECTS = [
    ("D1", "采样标记自检被破坏（`ASSERT_FAIL_MARK` 改错）",
     "01_collection_mechanisms.py",
     'ASSERT_FAIL_MARK = "**FAIL**"',
     'ASSERT_FAIL_MARK = "FAILED:"      # DEFECT: 标记改错',
     ["01_collection_mechanisms.py"], "fail"),

    ("D2", "插桩对照里去掉「不插桩」那一组（两组都插桩 ⇒ 测不出开销）",
     "01_collection_mechanisms.py",
     "        a, b = timed_pair(fn)",
     "        a, b = timed_pair(fn)\n"
     "        a = b                     # DEFECT: 「不插桩」那一组被删掉",
     ["01_collection_mechanisms.py"], "fail"),

    ("D3", "（★ 反向测试）跨 pass 组合改成**同一次** pass 内取分子分母",
     "02_replay_multipass.py",
     'CROSS_PAIRING = (("hot", "cold"), ("cold", "hot"))',
     'CROSS_PAIRING = (("hot", "hot"), ("cold", "cold"))   # DEFECT: 不再是跨执行',
     ["02_replay_multipass.py"], "pass"),

    ("D4", "默认值常量由 `all` 改成 `none`",
     "specs.py",
     'F15_DEFAULTS = {\n    "cache-control": "all",       # td[2] 逐字',
     'F15_DEFAULTS = {\n    "cache-control": "none",      # DEFECT',
     ["03_defaults_attribution.py"], "fail"),

    ("D5", "观测者效应的对照结构改回「两组 subprocess 不同」",
     "03_defaults_attribution.py",
     '    argv, shell, kwargs = spec["argv"], spec["shell"], spec["kwargs"]\n'
     '    # ★ 注入项 5（`make falsify`）就打在**上面这一行**之后：给 attach 组换一套启动结构。',
     '    argv, shell, kwargs = spec["argv"], spec["shell"], spec["kwargs"]\n'
     '    if attach:                       # DEFECT: 两组 subprocess 结构不同\n'
     '        shell = True\n'
     '        argv = " ".join(argv)\n'
     '    # ★ 注入项 5（`make falsify`）就打在**上面这一行**之后：给 attach 组换一套启动结构。',
     ["03_defaults_attribution.py"], "fail"),

    ("D6", "`clang -O2` 改成 `-O0`（内联不再发生 ⇒ 符号重新出现在采样报告里）",
     "01_collection_mechanisms.py",
     '    _, b_o2 = build_c(tmpdir, C_WORKLOAD_SRC, "O2", clang)',
     '    _, b_o2 = build_c(tmpdir, C_WORKLOAD_SRC, "O0", clang)   # DEFECT: -O2 → -O0',
     ["01_collection_mechanisms.py"], "fail"),
]

# ★ 子进程的 env：`PERF_QUICK=1` 走快档（**断言全是结构性的，走同一条代码路径**），
#   否则每次注入都要等 01 的三次 `sample` + 03 的十次子进程，元检验会慢到不可用。
PERF_QUICK = True
RUN_TIMEOUT = 600


def child_env():
    env = dict(os.environ)
    if PERF_QUICK:
        env["PERF_QUICK"] = "1"
    return env


def env_report():
    """报出三个编号程序各自需要的**外部工具**在不在（**不静默跳过**）。"""
    rows = [("clang", shutil.which(os.environ.get("CLANG") or "clang")),
            ("sample", shutil.which("sample"))]
    return rows


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
    print("  ★ 本主题的三个编号程序自己就是仪器 —— 本文件回答的质疑是：")
    print("    **如果那台仪器本身坏了呢？它还会报「通过」吗？**")
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
        print("     ⚠ `sample` 不在 ⇒ 相关断言 SKIP（**不计入通过**），基线仍然能过；")
        print("       此时 D6 的靶子（C.2「-O2 版 hot_loop 出现 0 次」）不会被触及。")
    print()

    # ---- 步骤 1/4：原文件哈希（跑前） ----
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
    n_pass_fail_expected = 0
    n_pass_pass_expected = 0
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
                    n_pass_fail_expected += 1
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
                    n_pass_pass_expected += 1
                else:
                    n_fail += 1
                print("     %-30s 退出 %-4d 期望**通过** → %s"
                      % (p, rc, "OK" if ok else "**反而失败了 ⇒ 那条断言写反了**"))
                if not ok:
                    print("         末行：%s" % (txt.strip().splitlines() or ["(无)"])[-1][:104])

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
        print("元检验通过：**%d 个注入项全部按「期望结果」反应**"
              "（%d 项期望失败 + %d 项期望通过的反向测试），"
              % (len(DEFECTS), n_pass_fail_expected, n_pass_pass_expected))
        print("           且不打补丁时三个程序全过。")
        print("→ 「本主题那台测量仪器自己坏掉会怎样」这个质疑，被这 %d 条反证正面回答。"
              % len(DEFECTS))
        print("★ `D3` 是**唯一一行期望「通过」**的：它证明 `A.5` 检查的是「**跨执行**」")
        print("  这个机制，而不是「数字看起来很怪」这种无条件怪癖 —— 若它失败，")
        print("  说明断言脱离了自己的作用域（P7）。")
        print("★ 计划 §1.5 说注入项 6 作用于「三个」；实测只有 `01` 编译 C（见模块头）。")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
