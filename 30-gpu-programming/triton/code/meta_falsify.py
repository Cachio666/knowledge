# -*- coding: utf-8 -*-
"""元检验（`make falsify`）：**故意把解释器改错，断言编号程序必须真的失败**。

★ 这是本主题对 **P12（自造真值：模拟器同时是被测对象）** 的正面回答。
  自造 DSL 最大的隐患是**解释器与被测对象是同一个东西**：
  解释器算出来的数，参照也是同一套代码算出来的，于是断言必然通过。
  本文件用**最直接的方式**回答这个质疑 —— **把解释器改坏，看断言会不会叫**。

  做法：把 `minitriton.py` 打上补丁写成副本 → 连同四个编号程序与其余支撑模块复制到临时目录 →
  逐个跑 → **断言退出码非 0**。若某个缺陷下程序仍然退出 0，则那条断言是**无效的**
  （只演示、不检查），本文件会把它标成 `**没失败，这条断言无效**` 并计一次失败。

★ 三条硬纪律：
  1. **不修改**仓库里的任何原文件 —— 每个变异体都在自己的 `tempfile` 目录里跑。
  2. **基线必须全过**：不打补丁时四个程序必须都退出 0，否则读数毫无意义。
  3. **锚点必须恰好出现 1 次**：锚点漂了（`src.count(old) != 1`）补丁打不上，
     那会让元检验**静默变成空操作** —— 这里直接把它记成失败。
     （`01_block_model.py` 的 `C.10` 也把这条钉在 `make run` 里。）

★ 本文件是**非编号支撑模块**，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ["01_block_model.py", "02_masking.py", "03_layout.py", "04_warp_reduce.py"]
# ★ `meta_falsify.py` **也要复制**：`01_block_model.py` 的 `C.10` 会用 `ast` 读它自己的
#   `DEFECTS` 列表来核对「锚点在 minitriton.py 里恰好出现 1 次」——副本里少了它，
#   基线（不打补丁）就会 rc=1，整个元检验的读数作废（第一版就是这么失败的）。
AUX = ["checks.py", "refs.py", "minitriton.py", "specs.py", "meta_falsify.py"]

# (缺陷名, minitriton.py 里要被替换的原文, 替换成什么, 期望失败的程序)
# ★ 这 5 条与 `specs.META_DEFECTS` 是同一份账（`01_block_model.py` 的 `C.10`/`A.21` 钉住条数）。
DEFECTS = [
    ("D1 掩码被忽略（读了越界位置）",
     "        vals = np.where(m, vals, fill)",
     "        vals = vals  # DEFECT: 掩码被忽略",
     ["02_masking.py"]),
    ("D2 所有布局转换都判成免费",
     "        return bool(np.array_equal(self.src.owners(), self.dst.owners()))",
     "        return True  # DEFECT: 一律免费",
     ["03_layout.py"]),
    ("D3 跨 warp 相位被抹掉（wpc>1 也不认）",
     "        self.cross_warp = wpc > 1",
     "        self.cross_warp = False  # DEFECT",
     ["03_layout.py", "04_warp_reduce.py"]),
    ("D4 三相归约的跨 warp 相位写错（只取 warp 0 的局部和）",
     "            combined = smem.sum(axis=axis, keepdims=True)   # 某个 warp 汇总",
     "            combined = smem.take(0, axis=axis)  # DEFECT: 丢掉其余 warp 的部分和",
     ["04_warp_reduce.py"]),
    ("D5 coalesced_layout 把 lane 摊到错的维上",
     "    rem_lane, rem_warp = 32, num_warps\n    for d in order:",
     "    rem_lane, rem_warp = 32, num_warps\n    for d in order[::-1]:  # DEFECT",
     ["03_layout.py"]),
]


def run_case(tmp, src_text, programs):
    for f in AUX:
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    for f in PROGRAMS:
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    with open(os.path.join(tmp, "minitriton.py"), "w", encoding="utf-8") as fh:
        fh.write(src_text)
    out = {}
    for p in programs:
        r = subprocess.run([sys.executable, "-W", "ignore::RuntimeWarning", p],
                           cwd=tmp, capture_output=True, text=True)
        out[p] = (r.returncode, r.stdout + r.stderr)
    return out


def main():
    src = open(os.path.join(HERE, "minitriton.py"), encoding="utf-8").read()
    print("=" * 74)
    print("元检验：往解释器里注入缺陷，断言编号程序必须失败（P12 的正面回答）")
    print("=" * 74)
    print("原始 minitriton.py：%d 行 / %d 字节"
          % (len(src.splitlines()), len(src.encode())))
    print("接入的支撑模块：%r" % (AUX,))

    with tempfile.TemporaryDirectory(prefix="tp_base_") as tmp:
        base = run_case(tmp, src, PROGRAMS)
    bad = [p for p, (rc, _) in base.items() if rc != 0]
    print()
    print("[基线] 不打补丁：%s" % ("四个程序全部退出 0 ✓" if not bad
                                else "有失败 → %r" % bad))
    n_fail = 0
    if bad:
        n_fail += 1

    for name, old, new, progs in DEFECTS:
        if src.count(old) != 1:
            print()
            print("[%s] 锚点在 minitriton.py 里出现 %d 次（必须恰好 1 次）→ **补丁无效**"
                  % (name, src.count(old)))
            n_fail += 1
            continue
        with tempfile.TemporaryDirectory(prefix="tp_def_") as tmp:
            res = run_case(tmp, src.replace(old, new), PROGRAMS)
        print()
        print("[%s]" % name)
        print("    锚点：%s" % old.strip())
        for p in PROGRAMS:
            rc, txt = res[p]
            mark = "退出 %d" % rc
            if p in progs:
                ok = rc != 0
                first_fail = next((l.strip() for l in txt.splitlines()
                                   if l.strip().startswith("[FAIL]")), None)
                print("    %-22s %-8s 期望失败 → %s"
                      % (p, mark, "OK" if ok else "**没失败，这条断言无效**"))
                if first_fail:
                    print("        证据：%s" % first_fail[:110])
                if not ok:
                    n_fail += 1
            else:
                print("    %-22s %-8s （未注入本缺陷）" % (p, mark))

    print()
    print("=" * 74)
    if n_fail:
        print("元检验失败 %d 项：至少有一条断言是「只演示、不检查」" % n_fail)
    else:
        print("元检验通过：**5 种缺陷全部被断言抓住**，且不打补丁时四个程序全过。")
        print("→ 「解释器与被测对象是同一个东西」这个质疑，在本主题上被这 5 条反证正面回答。")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
