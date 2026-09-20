#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""元检验（`make falsify`）：**故意把解释器改错，断言编号程序必须真的失败**。

★ 这是本主题对 **P12（自造真值：模拟器同时是被测对象）** 的正面回答。
  本主题的三个编号程序**全部**跑在同一个自己写的解释器（`ptxsub.py`）上：
  发射数、活跃 lane 数、bank 冲突路数、屏障到达量，**都是它算的**。
  于是有一个致命质疑：

  > **被测对象和参照实现是同一段代码。** 如果文档说「这样写会多 2 条发射」，
  > 而程序用自己写的解释器「证明」了它，那么**这什么也没证明**。

  本文件用**最直接的方式**回答它：**把解释器改坏，看断言会不会叫。**
  做法是给 `ptxsub.py` 打补丁写成副本 → 连同三个编号程序与其余支撑模块复制到临时目录
  → 逐个跑 → **断言退出码非 0**。若某个缺陷下程序仍然退出 0，
  则那条断言是**无效的**（只演示、不检查），本文件会把它标成
  `**没失败，这条断言无效**` 并计一次失败。

★ 三条硬纪律：
  1. **不修改**仓库里的任何原文件 —— 每个变异体都在自己的 `tempfile` 目录里跑。
  2. **基线必须全过**：不打补丁时三个程序必须都退出 0，否则读数毫无意义。
  3. **锚点必须恰好出现 1 次**：锚点漂了（`src.count(old) != 1`）补丁打不上，
     那会让元检验**静默变成空操作** —— 这里直接把它记成失败。
     （`01_virtual_isa.py` 的 `C.3`/`C.4` 也把「表外指令必须报错」钉在 `make run` 里。）

★ 本文件是**非编号支撑模块**，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ["01_virtual_isa.py", "02_warp_execution.py", "03_smem_and_sass.py"]
# ★ `meta_falsify.py` **也要复制**：`01_virtual_isa.py` 的 `A.3`/`A.7`/`A.8` 会用 `ast`
#   读**同目录里的其它文件**（另两个程序 + `specs.py` + `ptxsub.py`）。副本里少一个，
#   基线就会 rc=1，整个元检验的读数作废（`triton` 主题的第一版就是这么失败的）。
AUX = ["ptxsub.py", "specs.py", "meta_falsify.py"]

# (缺陷名, ptxsub.py 里要被替换的原文, 替换成什么, 期望失败的程序)
# ★ 每一条都对应一个**正文里的承重结论** —— 改坏它，那条结论的载体必须叫。
DEFECTS = [
    ("D1 bfly 的异或方向被改成加一（源 lane 算错）",
     "                j = l ^ (src & 0x1F)",
     "                j = l + (src & 0x1F)  # DEFECT: 异或改成加法",
     ["02_warp_execution.py"]),

    ("D2 bar.sync 不阻塞（直接放行）",
     "        self.blocked.add(w)\n        return 'blocked'",
     "        return 'ok'  # DEFECT: 屏障不阻塞",
     ["02_warp_execution.py"]),

    ("D3 到达计数按**活跃 lane** 记（而不是按 warp 记）",
     "        self.bar_count[bid] = self.bar_count.get(bid, 0) + 32",
     "        self.bar_count[bid] = self.bar_count.get(bid, 0) + len(act)  # DEFECT",
     ["02_warp_execution.py"]),

    ("D4 未声明的寄存器不再报错（越界静默取 0）",
     "        if i >= len(self.regs[0][0]):",
     "        if False:  # DEFECT: 未声明的寄存器不再报错",
     ["01_virtual_isa.py"]),

    ("D5 bank 冲突统计不去重（广播被算成冲突）",
     "        banks.setdefault(word % nbank, set()).add(word)",
     "        banks.setdefault(word % nbank, []).append(word)  # DEFECT: 用 list",
     ["03_smem_and_sass.py"]),

    ("D6 表外指令静默跳过（覆盖范围不再闭合）",
     "        bad = sorted({i.op for i in self.prog.instrs if i.op not in SUPPORTED_OPS})",
     "        bad = []  # DEFECT: 表外指令静默跳过",
     ["01_virtual_isa.py"]),
]

# ★ 另外一条**方向相反**的缺陷：把 `.up` 的实现改成**照抄 ISA 原文**（`maxLane`）。
#   它不是「改错」——**它是改成了另一个候选语义**。它必须让 `02` 的 `C.3` 叫，
#   因为那条断言的作用正是**把「本库的偏离」钉成可检测的**：
#   如果换成原文式子而程序照样通过，那说明「本实现的偏离」根本没被任何断言抓住。
DEFECT_UP_LITERAL = (
    "D7 `.up` 改成照抄 ISA 原文的 maxLane（本库的「有意偏离」应当被检测到）",
    "                pval = (j >= minLane)          # ★ 有意偏离 ISA 文本（原文是 maxLane）",
    "                pval = (j >= maxLane)  # DEFECT: 照抄 ISA 原文",
    ["02_warp_execution.py"],
)


def run_case(tmp, src_text, programs):
    for f in AUX:
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    for f in PROGRAMS:
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    with open(os.path.join(tmp, "ptxsub.py"), "w", encoding="utf-8") as fh:
        fh.write(src_text)
    out = {}
    for p in programs:
        r = subprocess.run([sys.executable, p], cwd=tmp, capture_output=True, text=True)
        out[p] = (r.returncode, r.stdout + r.stderr)
    return out


def main():
    src = open(os.path.join(HERE, "ptxsub.py"), encoding="utf-8").read()
    print("=" * 74)
    print("元检验：往解释器里注入缺陷，断言编号程序必须失败（P12 的正面回答）")
    print("=" * 74)
    print("原始 ptxsub.py：%d 行 / %d 字节" % (len(src.splitlines()), len(src.encode())))
    print("接入的支撑模块：%r" % (AUX,))
    print("★ 被测对象与参照实现是**同一段代码**（都跑 `ptxsub`）——本文件就是对这个质疑的回答")

    with tempfile.TemporaryDirectory(prefix="ps_base_") as tmp:
        base = run_case(tmp, src, PROGRAMS)
    bad = [p for p, (rc, _) in base.items() if rc != 0]
    print()
    print("[基线] 不打补丁：%s" % ("三个程序全部退出 0 ✓" if not bad
                                else "有失败 → %r" % bad))
    n_fail = 0
    if bad:
        n_fail += 1

    defects = DEFECTS + [DEFECT_UP_LITERAL]
    for name, old, new, progs in defects:
        if src.count(old) != 1:
            print()
            print("[%s] 锚点在 ptxsub.py 里出现 %d 次（必须恰好 1 次）→ **补丁无效**"
                  % (name, src.count(old)))
            n_fail += 1
            continue
        with tempfile.TemporaryDirectory(prefix="ps_def_") as tmp:
            res = run_case(tmp, src.replace(old, new), PROGRAMS)
        print()
        print("[%s]" % name)
        print("    锚点：%s" % old.strip()[:96])
        for p in PROGRAMS:
            rc, txt = res[p]
            mark = "退出 %d" % rc
            if p in progs:
                ok = rc != 0
                first_fail = next((l.strip() for l in txt.splitlines()
                                   if l.strip().startswith("[FAIL]")), None)
                print("    %-24s %-8s 期望失败 → %s"
                      % (p, mark, "OK" if ok else "**没失败，这条断言无效**"))
                if first_fail:
                    print("        证据：%s" % first_fail[:110])
                if not ok:
                    n_fail += 1
            else:
                print("    %-24s %-8s （未注入本缺陷）" % (p, mark))

    print()
    print("=" * 74)
    if n_fail:
        print("元检验失败 %d 项：至少有一条断言是「只演示、不检查」" % n_fail)
    else:
        print("元检验通过：**%d 种缺陷全部被断言抓住**，且不打补丁时三个程序全过。"
              % len(defects))
        print("→ 「解释器与被测对象是同一个东西」这个质疑，在本主题上被这 %d 条反证正面回答。"
              % len(defects))
        print("★ 其中 D7 是**方向相反**的一条：把 `.up` 改成照抄 ISA 原文也必须被抓住 ——")
        print("  它证明的是「本库那处有意偏离**是可检测的**」，而不是「本库是对的」。")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
