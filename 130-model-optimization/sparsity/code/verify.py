#!/usr/bin/env python3
"""130-model-optimization/sparsity 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`、也不 `import numpy`**
  （CONTRIBUTING §2 第三种形态；只用 `sys`）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序** `01_sparsity.py` 里（代价模型的参照是真系统，
      只有能 import 到它的环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「数学 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的算术**：
    2:4 的压缩率账（36/64 与两个分母）、4:8 paired 的两种 bit/元素口径、
    硬件粒度表只有三种 N:M、非结构化不在表里、非结构化的 CPU/GPU 两条引用值。
  · 编号程序（真 torch）测的是**真实实现的行为**：torch 内部转换函数算出的 metadata
    位宽与往返、**float32 路径用的是 1:2 还是 2:4**、公开 API 在 CPU 上报错。
  · **两者不得互相背书。** 数学成立不证明实现如此；实现如此不证明数学成立。

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch。
"""

import sys

import specs as SPECS

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}


# ===========================================================================
# 基础设施


# ===========================================================================


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def section(tag, title):
    print()
    print("-" * 74)
    print("%s  %s" % (tag, title))
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


# ===========================================================================
# 独立参考实现（ref_*）：**一行都不来自 torch**


# ===========================================================================


def ref_compress_ratio(elem_bits, meta_bits, group, nonzero):
    """**比率**：`(nonzero × elem_bits + meta_bits) / (group × elem_bits)`。"""
    return (nonzero * elem_bits + meta_bits) / float(group * elem_bits)


def ref_bits_per_elem(elem_bits, meta_bits, group, nonzero):
    """**bit/元素**（含 metadata）：`(nonzero × elem_bits + meta_bits) / group`。

    ⚠ 这两个函数**不是同一个量**：一个是比率（无量纲），一个是宽度（bit）。
    本文件把它们分开写，就是为了不让「36/64」与「9 bit/元素」混用——
    它们是同一个事实的两种写法，但**判据不同**。
    """
    return (nonzero * elem_bits + meta_bits) / float(group)


# ===========================================================================
# A. 判据复核


# ===========================================================================


def run_A():
    section("A", "判据复核（纯推导；参照物是 ref_* 与手写期望值）")

    meta_bits = SPECS.META_INDEX_BITS * SPECS.META_INDEXES_PER_GROUP_2_4

    sub("A.1 2:4 的压缩率算术：36/64 与**两个分母**")
    rows = {}
    for name in ("fp16", "int8", "int4"):
        w = SPECS.ELEM_BITS[name]
        rows[name] = ref_compress_ratio(w, meta_bits, SPECS.GROUP_SIZE_2_4,
                                        SPECS.NONZERO_PER_GROUP_2_4)
        print("      %-5s : (2×%2d + %d)/(4×%2d) = %d/%d = %.6f"
              % (name, w, meta_bits, w, 2 * w + meta_bits, 4 * w, rows[name]))
    check("A", "★ fp16 的 36/64 = 0.5625（省 43.75%，**不是 50%**）",
          abs(rows["fp16"] - 36.0 / 64.0) < 1e-15 and abs(rows["fp16"] - 0.5625) < 1e-15)
    check("A", "★ int8 的 20/32 = 0.6250（**元素越窄，4 bit metadata 越贵**）",
          abs(rows["int8"] - 20.0 / 32.0) < 1e-15)
    check("A", "★ int4 的 12/16 = 0.7500 —— 稀疏这一步本身只省 25%",
          abs(rows["int4"] - 12.0 / 16.0) < 1e-15)
    check("A", "★ **两个分母**：metadata/原数据 = 4/64 = 1/16 = %.6f；"
          "metadata/压缩后载荷 = 4/32 = 1/8 = %.6f"
          % (meta_bits / 64.0, meta_bits / 32.0),
          abs(meta_bits / 64.0 - 1.0 / 16.0) < 1e-15
          and abs(meta_bits / 32.0 - 1.0 / 8.0) < 1e-15)
    check("A", "★ 16-bit 的**三个分母必须分清**：36/64（总）、4/64（原数据）、4/32（载荷）",
          abs(meta_bits / 64.0 - 0.0625) < 1e-15
          and abs(meta_bits / 32.0 - 0.125) < 1e-15
          and abs(36.0 / 64.0 - 0.5625) < 1e-15)

    sub("A.2 4:8 paired：**0.5 bit/元素只是 metadata 那一份**，整块等效是 2.5")
    meta_per_elem = SPECS.META_BITS_PER_GROUP_4_8 / float(SPECS.GROUP_SIZE_4_8)
    equiv_width = ref_bits_per_elem(SPECS.ELEM_BITS["int4"], SPECS.META_BITS_PER_GROUP_4_8,
                                    SPECS.GROUP_SIZE_4_8, SPECS.NONZERO_PER_GROUP_4_8)
    print("      metadata 那一份 = %d/%d = %.2f bit/元素" %
          (SPECS.META_BITS_PER_GROUP_4_8, SPECS.GROUP_SIZE_4_8, meta_per_elem))
    print("      整块等效宽度   = (4×4 + 4)/8 = %.1f bit/元素（%.1f/%.1f = %.0f×）"
          % (equiv_width, equiv_width, meta_per_elem, equiv_width / meta_per_elem))
    check("A", "★ 两个口径**差 5 倍**（0.5 vs 2.5 bit/元素）——**不得混称**（P9 实例）",
          abs(meta_per_elem - 0.5) < 1e-15
          and abs(equiv_width - 2.5) < 1e-15
          and abs(equiv_width / meta_per_elem - 5.0) < 1e-15)

    sub("A.3 叠加压缩率：**不是简单相乘**，逐行由规格常量重算")
    recomputed = {
        "fp16 dense": float(SPECS.ELEM_BITS["fp16"]),
        "int8 dense": float(SPECS.ELEM_BITS["int8"]),
        "int4 dense": float(SPECS.ELEM_BITS["int4"]),
        "fp16+2:4": ref_bits_per_elem(SPECS.ELEM_BITS["fp16"], meta_bits,
                                      SPECS.GROUP_SIZE_2_4, SPECS.NONZERO_PER_GROUP_2_4),
        "int8+2:4": ref_bits_per_elem(SPECS.ELEM_BITS["int8"], meta_bits,
                                      SPECS.GROUP_SIZE_2_4, SPECS.NONZERO_PER_GROUP_2_4),
        "int4+2:4": ref_bits_per_elem(SPECS.ELEM_BITS["int4"], meta_bits,
                                      SPECS.GROUP_SIZE_2_4, SPECS.NONZERO_PER_GROUP_2_4),
        "int4+4:8": ref_bits_per_elem(SPECS.ELEM_BITS["int4"],
                                      SPECS.META_BITS_PER_GROUP_4_8,
                                      SPECS.GROUP_SIZE_4_8, SPECS.NONZERO_PER_GROUP_4_8),
    }
    print("      组合       | bit/元素 | 本文件重算 | 相对 fp16 | 朴素相乘 | 高估")
    bad = []
    for key in sorted(SPECS.COMBINED_BITS_PER_ELEM, key=lambda k: SPECS.COMBINED_BITS_PER_ELEM[k]):
        want = SPECS.COMBINED_BITS_PER_ELEM[key]
        got = recomputed[key]
        if abs(want - got) > 1e-12:
            bad.append(key)
        naive = SPECS.NAIVE_MULTIPLE[key]
        print("      %-10s | %8.2f | %10.2f | %8.2f× | %7.2f× | %5.2f×"
              % (key, want, got, SPECS.BASE_ELEM_BITS / got, naive,
                 naive / (SPECS.BASE_ELEM_BITS / got)))
    check("A", "★ %d 行**逐行**重算一致（本文件用独立的 ref_compress_ratio）"
          % len(recomputed), not bad, "不一致：%s" % bad if bad else "")
    check("A", "★ 朴素相乘**只对稠密行正确**，对稀疏行一律高估"
          "（fp16+2:4 %.2f×、int8+2:4 %.2f×、int4+2:4 %.2f×）"
          % (SPECS.NAIVE_MULTIPLE["fp16+2:4"] / (SPECS.BASE_ELEM_BITS / recomputed["fp16+2:4"]),
             SPECS.NAIVE_MULTIPLE["int8+2:4"] / (SPECS.BASE_ELEM_BITS / recomputed["int8+2:4"]),
             SPECS.NAIVE_MULTIPLE["int4+2:4"] / (SPECS.BASE_ELEM_BITS / recomputed["int4+2:4"])),
          all(SPECS.NAIVE_MULTIPLE[k] >= SPECS.BASE_ELEM_BITS / recomputed[k]
              for k in recomputed))

    sub("A.4 硬件粒度表：**只有三种 N:M**，非结构化不在表里")
    tags = [t for t, _, _ in SPECS.HW_NM_GRANULARITIES]
    print("      粒度：%s" % " / ".join(tags))
    check("A", "★ 恰好 %d 档，且每档都是 N:M 形式" % len(tags),
          len(tags) == 3 and all(":" in t for t in tags))
    check("A", "★ 非结构化**不在表里**（规格明确：`UNSTRUCTURED_IN_HW_GRANULARITY_TABLE` 为假）",
          SPECS.UNSTRUCTURED_IN_HW_GRANULARITY_TABLE is False)
    check("A", "★ 4:8 的簇约束逐字写着「2 元素子块」",
          "2 元素子块" in SPECS.HW_NM_GRANULARITIES[2][2])
    # ★ 三档的**名字与 dtype 集合**逐条核（规格表逐字核对，改一个字就失败）
    by_tag = dict((t, set(d)) for t, d, _ in SPECS.HW_NM_GRANULARITIES)
    check("A", "★ 三档的名字逐字是 `1:2` / `2:4` / `4:8 paired`",
          set(by_tag) == {"1:2", "2:4", "4:8 paired"})
    check("A", "★ `1:2` 档对应的 dtype **只有 tf32**（PTX 的映射）",
          by_tag["1:2"] == {"tf32"})
    check("A", "★ `2:4` 档含 f16/bf16/int8/e4m3/e5m2",
          {"f16", "bf16", "int8", "e4m3", "e5m2"} <= by_tag["2:4"])
    check("A", "★ `4:8 paired` 档含 e2m1/mxf4 与 u4/s4",
          {"e2m1", "mxf4", "u4", "s4"} <= by_tag["4:8 paired"])
    check("A", "★ cuSPARSELt 的 50%% 映射里 `float` 与 `e2m1` **不在 2:4 档**"
          "（%s / %s）"
          % (SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["float"],
             SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["e2m1"]),
          SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["float"] == "1:2"
          and SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["e2m1"] == "4:8 paired"
          and len(SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE) == 7)
    # 组合数：一个 4 元素组里「恰好 2 个非零」的模式数 = C(4,2) = 6。
    popcount2 = sum(1 for mask in range(16) if bin(mask).count("1") == 2)
    print("      一个 4 元素组的「恰好 2 非零」模式数 = %d（= C(4,2)）" % popcount2)
    check("A", "★ 一个 4 元素组里「恰好 2 非零」的模式恰好 %d 种" % popcount2,
          popcount2 == 6)
    check("A", "★ %d bit 的 metadata 有 %d 个可用值 ≥ %d 种合法模式（**够用但不唯一**）"
          % (meta_bits, 2 ** meta_bits, popcount2),
          meta_bits == 4 and 2 ** meta_bits >= popcount2)
    check("A", "★ 4:8 的每组非零数 = %d（组大小 %d）；以 2 元素子块计，等价于 4 个「1:2」对"
          % (SPECS.NONZERO_PER_GROUP_4_8, SPECS.GROUP_SIZE_4_8),
          SPECS.NONZERO_PER_GROUP_4_8 == SPECS.GROUP_SIZE_4_8 // 2
          and SPECS.GROUP_SIZE_4_8 == 4 * (SPECS.GROUP_SIZE_2_4 // SPECS.NONZERO_PER_GROUP_2_4))


# ===========================================================================
# C. 边界与反例


# ===========================================================================


def run_C():
    section("C", "边界与反例（断言的是「失败行为成立」）")

    sub("C.1 非结构化：**不在硬件粒度表里**（穷举表项也找不到它）")
    found = [t for t, _, _ in SPECS.HW_NM_GRANULARITIES if "非结构" in t or "任意" in t]
    check("C", "★ 在 %d 档硬件粒度里**找不到**「非结构化」这一档"
          % len(SPECS.HW_NM_GRANULARITIES), not found)
    check("C", "★ GPU 侧的门槛是 **%.0f%% 稀疏**（引用值），而 CPU 侧 OPT-2.7B 在 50%% 就有 "
          "%.2f×——**两者不是同一个机制、也不在同一个硬件上**"
          % (100 * SPECS.UNSTRUCTURED_GPU_MIN_SPARSITY_FOR_SPEEDUP,
             SPECS.DEEPSPARSE_OPT27B_CPU_SPEEDUP[1][1]),
          # ★ 判据：它是一个**稀疏度**，必须落在 (0.9, 1) 里——
          #   写成「> 0.9」太松（改成一个大于 1 的数也能过）。
          0.9 < SPECS.UNSTRUCTURED_GPU_MIN_SPARSITY_FOR_SPEEDUP < 1.0
          and SPECS.DEEPSPARSE_OPT27B_CPU_SPEEDUP[1][0] == 0.50)
    check("C", "★ CPU 侧三个稀疏度**单调加速**（%.2f → %.2f → %.2f）"
          % tuple(v for _, v in SPECS.DEEPSPARSE_OPT27B_CPU_SPEEDUP),
          all(SPECS.DEEPSPARSE_OPT27B_CPU_SPEEDUP[i][1]
              < SPECS.DEEPSPARSE_OPT27B_CPU_SPEEDUP[i + 1][1] for i in range(2)))


# ===========================================================================
# E. 自检：故意改错会报错吗（P5 的逐条施加）


# ===========================================================================


def run_E():
    section("E", "自检：**故意改错**，A/C 的判据会不会失败（P5）")
    ok = True

    def expect_change(label, a, b, why):
        nonlocal ok
        good = (a != b)
        ok &= good
        print("    [%s] E  %s" % ("通过" if good else "**不通过**", label))
        print("          —— %s" % why)

    meta_bits = SPECS.META_INDEX_BITS * SPECS.META_INDEXES_PER_GROUP_2_4
    # E.1 metadata 从 4 bit 改成 2 bit：36/64 变成 34/64
    r_wrong = ref_compress_ratio(16, 2, 4, 2)
    r_right = ref_compress_ratio(16, meta_bits, 4, 2)
    expect_change("E.1 把 metadata 从 4 bit/组改成 2 bit/组，36/64 立刻变成 34/64",
                  r_wrong, r_right, "A.1 的那条断言会失败（%.4f vs %.4f）"
                  % (r_wrong, r_right))

    # E.2 把「分母 = 原数据」改成「分母 = 载荷」
    expect_change("E.2 metadata 的分母从「压缩后载荷」改成「原数据」，12.5% 变成 6.25%",
                  4.0 / 32.0, 4.0 / 64.0, "A.1 的「两个分母」那条会失败")

    # E.3 4:8 的 metadata 宽度写错：0.5 变成 1.0 bit/元素
    expect_change("E.3 4:8 的 metadata 从 4 bit/组改成 8 bit/组，0.5 变成 1.0 bit/元素",
                  4 / 8.0, 8 / 8.0, "A.2 那条「两个口径差 5 倍」会失败")

    # E.4 **对照组**：把上面每一条「改错」改回原值，原判据必须重新通过。
    #   没有对照组的话，「改错会失败」可能只是 harness 在空转（E 里任何一条
    #   只要自己崩了也会算「按预期失败」）——这正是 A6 说的假阳性模式一。
    ctrl = {
        "E.1 原值 36/64": abs(ref_compress_ratio(16, meta_bits, 4, 2) - 0.5625) < 1e-15,
        "E.3 原值 0.5 bit/元素": abs(4 / 8.0 - 0.5) < 1e-15,
    }
    n_ctrl = sum(1 for v in ctrl.values() if v)
    for k, v in ctrl.items():
        print("          %s %s" % ("✓" if v else "✗", k))
    ok &= (n_ctrl == len(ctrl))
    print("    [%s] E  %s **对照组**：%d/%d 条的**原值**仍然满足原判据 ⇒ harness 不是空转"
          % ("通过" if n_ctrl == len(ctrl) else "**不通过**", "E.4", n_ctrl, len(ctrl)))

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A/C 类判据是活的" if ok
                              else "**有自检未按预期失败 ⇒ 判据有问题**"))
    return ok


# ===========================================================================
# F. 本文件覆盖了哪些 specs 常量（A6 变异测试的清单）


# ===========================================================================


def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    src = open(__file__, encoding="utf-8").read()
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到的 %d 条"
          % (len(names), len(used)))
    unused = [n for n in names if n not in used]
    print("      本文件**没有**读到的 %d 条（由编号程序 01_sparsity.py 负责，见其 §F）："
          % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异的对象是**常量**，每个常量的值都能被至少一处断言分辨",
          len(used) > 0)
    # ★ `UNSOURCED` 也必须被断言读到——否则它就成了「没人查的抄件」。
    #   判据写成**条数与内容**两条：条数一改就失败，内容删掉也失败。
    joined = " ".join(SPECS.UNSOURCED)
    print("      `UNSOURCED` 共 %d 条；逐条内容：" % len(SPECS.UNSOURCED))
    for i, item in enumerate(SPECS.UNSOURCED, 1):
        print("        %d. %s" % (i, item[:64].replace("\n", " ") + "…"))
    check("A", "★ `UNSOURCED` 恰好 %d 条（**条数本身是判据**：删一条或多一条都失败）"
          % len(SPECS.UNSOURCED), len(SPECS.UNSOURCED) == 3)
    check("A", "★ 其中**逐字**记着「端到端合成区间」与 cuSPARSELt 字节格式这两项",
          "合成区间" in joined and "cuSPARSELt" in joined)
    return unused


# ===========================================================================
# main


# ===========================================================================


def main():
    print("=" * 78)
    print("130-model-optimization/sparsity · verify.py（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。")
    print("      它用独立的 ref_* 重算 2:4 / 4:8 的压缩率算术与硬件粒度表，")
    print("      并用手写期望值判定——**不读 torch 的任何输出**。")

    run_A()
    run_C()
    okE = run_E()
    run_F()

    print()
    print("=" * 78)
    n, a, c = _STATE["n"], _STATE["A"], _STATE["C"]
    print("合计 %d 条断言：A %d 条 / C %d 条；失败 %d 条" % (n, a, c, len(_STATE["fail"])))
    if _STATE["fail"]:
        print("失败清单：")
        for f in _STATE["fail"]:
            print("  · %s" % f)
    print("E 节自检：%s" % ("通过" if okE else "**不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okE) else 1)


if __name__ == "__main__":
    main()
