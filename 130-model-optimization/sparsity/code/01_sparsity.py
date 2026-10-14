#!/usr/bin/env python3
"""01 稀疏粒度：硬件只定义三种 N:M；以及 ★ torch 的 float32 路径其实是 1:2。

对应文档：130-model-optimization/sparsity/01-sparsity-granularity.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：稀疏省下的字节去哪了。压缩率的三笔账（载荷、metadata、两个分母）
       与「元素越窄 metadata 越贵」的算术。**这一节全部由 specs 的规格常量驱动，
       一个数都不硬编码。**
  A —— **两条互不依赖的观察通道**：
       ① 规格侧（PTX 的三种 N:M 粒度、cuSPARSELt 的按类型映射）—— **引用/规格值**；
       ② 本机侧（`torch.sparse._semi_structured_conversions` 的 metadata 位宽）——
          **本机实测值**，与规格侧的 4 bit/组**逐位吻合**。
  C —— 失败行为必须**真的跑出来**：torch 的 **float32** 路径不是 2:4 而是 1:2，
       构造 6 种 2:4 模式，其中 2 种的往返**静默丢数据**（不报错）。

★★ 本程序最重要的一条：**「4 bit/组」这个数不能跨 dtype 混用。**
   · fp16 / bf16 / int8：一组 = 4 个元素，metadata 4 bit/组；
   · **float32：编码单元是 2 元素对**（源码 `ksparse = 2`），metadata 8 bit/
     「4 元素」——但那是**两个**编码单元。用「4 元素组」做分母会算错。
   判据是源码第 87–95 行的分支，实测的判别方法是那 6 种 2:4 模式的往返。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 01_sparsity   （或 make run 全跑）
"""

import platform
import subprocess
import sys

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def cpu_model():
    """测量条件里的「CPU 型号」。**这不是算法**。"""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          stderr=subprocess.DEVNULL)
            return out.decode().strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "%s（型号未取到）" % platform.machine()


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def rule(title):
    print()
    print("-" * 74)
    print(title)
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


def banner():
    print("=" * 74)
    print("01 稀疏粒度：硬件只定义三种 N:M；torch 的 float32 路径其实是 1:2")
    print("=" * 74)
    print("测量条件（§5 直接引用这一段）：")
    print("  CPU        ：%s" % cpu_model())
    print("  主机       ：%s" % SPECS.MEASURE_HOST)
    print("  python     ：%s" % sys.version.split()[0])
    print("  torch      ：%s" % torch.__version__)
    print("  CUDA 可用  ：%s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  本主题跑过的版本：%s" % " / ".join(SPECS.MEASURED_VERSIONS))
    print("  未覆盖的 minor  ：%s" % SPECS.UNCOVERED_MINORS)
    print("  版本下限        ：%s（%s）"
          % (".".join(str(v) for v in SPECS.TORCH_MIN_FOR_THIS_TOPIC), SPECS.TORCH_MIN_REASON))
    print()
    print("  ⚠ 本程序不跑 GPU、不跑模型；外部来源的加速比一律按**引用值**处理。")
    print("  ⚠ 公开 API `torch.sparse.to_sparse_semi_structured` 在 **CPU 上直接报错**")
    print("     （\"Only CUDA tensors are currently supported\"），所以本程序用的是")
    print("     **内部**函数 `torch.sparse._semi_structured_conversions` 的两个函数。")


# ---------------------------------------------------------------------------
# 被测对象：torch 内部的半结构化转换
# ---------------------------------------------------------------------------

def _enc_dec():
    from torch.sparse._semi_structured_conversions import (
        sparse_semi_structured_from_dense_cutlass as enc,
        sparse_semi_structured_to_dense_cutlass as dec)
    return enc, dec


def measure_layout(dtype_name, m=64, k=256):
    """量一种 dtype 的压缩布局。返回一个字典。**这是本程序的被测对象**。

    判据（三条，缺一不可）：
      · 分母 = **原始稠密张量的位宽** `numel × element_size × 8`（压缩后的矩阵列数
        已经减半，拿它当分母是另一个量）；
      · 「组」= 4 个元素（对 float32 是**两个** 2 元素对，见 §C）；
      · 元素值必须**真的非零**——全零矩阵的 metadata 走的是「按原样取前两个」的
        退化分支，位宽一样，但那样就没有验证非零选取的机会。
    """
    enc, dec = _enc_dec()
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "int8": torch.int8, "float32": torch.float32}[dtype_name]
    if dtype_name == "int8":
        dense = torch.randint(1, 100, (m, k), dtype=dt)
    else:
        dense = torch.randn(m, k, dtype=dt)

    sparse, meta = enc(dense)
    back = dec(sparse, meta)

    dense_bits = dense.numel() * dense.element_size() * 8
    comp_bits = sparse.numel() * sparse.element_size() * 8
    meta_bits = meta.numel() * meta.element_size() * 8
    return {
        "dense": dense, "sparse": sparse, "meta": meta, "back": back,
        "dense_bits": dense_bits, "comp_bits": comp_bits, "meta_bits": meta_bits,
        "meta_per_4elem": meta_bits / (dense.numel() // 4.0),
        "total_over_dense": (comp_bits + meta_bits) / dense_bits,
        "meta_over_dense": meta_bits / dense_bits,
        "meta_dtype": str(meta.dtype).replace("torch.", ""),
    }


def pattern_dense(m, k, dtype, p0, p1):
    """造一个固定 2:4 模式的矩阵：每组 4 个元素里第 p0、p1 位非零（值 1.0 与 2.0）。

    判据：**非零值的相对大小（1.0 vs 2.0）用于识别「谁被丢了」**——
    往返失败时 maxdiff = 2.0 说明丢掉的是那个 2.0。
    """
    base = torch.zeros(m, k, dtype=torch.float32)
    idx = torch.arange(k)
    base[:, idx % 4 == p0] = 1.0
    base[:, idx % 4 == p1] = 2.0
    return base.to(dtype)


# ---------------------------------------------------------------------------
# B. 代价模型：压缩率的三笔账
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：稀疏省下的字节去哪了（全部由 specs 的规格常量算出）")

    meta_bits = SPECS.META_INDEX_BITS * SPECS.META_INDEXES_PER_GROUP_2_4
    sub("B.1 2:4 的三笔账 ——「组」= %d 个元素，metadata = %d bit/组"
        % (SPECS.GROUP_SIZE_2_4, meta_bits))
    print("      规格：一组里 %d 个非零，每个非零一个 %d-bit 索引 ⇒ %d × %d = %d bit"
          % (SPECS.NONZERO_PER_GROUP_2_4, SPECS.META_INDEX_BITS,
             SPECS.NONZERO_PER_GROUP_2_4, SPECS.META_INDEX_BITS, meta_bits))
    check("B", "★ 规格推出的 metadata 位宽 = %d bit/组" % meta_bits, meta_bits == 4)

    rows = []
    for name in ("fp16", "int8", "int4"):
        w = SPECS.ELEM_BITS[name]
        dense_bits = SPECS.GROUP_SIZE_2_4 * w
        payload_bits = SPECS.NONZERO_PER_GROUP_2_4 * w
        total_bits = payload_bits + meta_bits
        rows.append((name, dense_bits, payload_bits, total_bits,
                     total_bits / dense_bits, meta_bits / dense_bits,
                     meta_bits / payload_bits))
    print()
    print("      元素位宽 | 稠密/组 | 载荷/组 | metadata | 合计 | 合计/稠密 | meta/原数据 | meta/载荷")
    for (name, d, p, t, r, md, mp) in rows:
        print("      %-9s | %7d | %7d | %8d | %4d | %9.4f | %10.4f | %9.4f"
              % (name, d, p, meta_bits, t, r, md, mp))
    fp16 = [r for r in rows if r[0] == "fp16"][0]
    check("B", "★ fp16：36/64 = %.4f（省 %.2f%%）" % (fp16[4], 100 * (1 - fp16[4])),
          abs(fp16[4] - 36.0 / 64.0) < 1e-12)
    check("B", "★ 两个分母：meta/原数据 = 1/16 = %.4f，meta/载荷 = 1/8 = %.4f"
          % (fp16[5], fp16[6]),
          abs(fp16[5] - 4.0 / 64.0) < 1e-12 and abs(fp16[6] - 4.0 / 32.0) < 1e-12)
    int8 = [r for r in rows if r[0] == "int8"][0]
    check("B", "★ 8-bit 元素：(2×8+4)/(4×8) = 20/32 = %.4f（元素越窄 metadata 越贵）" % int8[4],
          abs(int8[4] - 20.0 / 32.0) < 1e-12)
    check("B", "★ 两列 meta 比值差 2 倍（1/16 vs 1/8）——**不得混用**",
          abs(fp16[6] / fp16[5] - 2.0) < 1e-12)

    sub("B.2 4:8 paired：**0.5 bit/元素只是 metadata 那一份**，整块等效是 2.5")
    print("      metadata：每 %d 元素组 %d bit ⇒ %.2f bit/元素"
          % (SPECS.GROUP_SIZE_4_8, SPECS.META_BITS_PER_GROUP_4_8,
             SPECS.META_BITS_PER_GROUP_4_8 / float(SPECS.GROUP_SIZE_4_8)))
    eq_int4_4_8 = (SPECS.NONZERO_PER_GROUP_4_8 * SPECS.ELEM_BITS["int4"]
                   + SPECS.META_BITS_PER_GROUP_4_8) / float(SPECS.GROUP_SIZE_4_8)
    check("B", "★ int4 + 4:8 的**整块等效宽度** = %.1f bit/元素（不是 0.5）" % eq_int4_4_8,
          abs(eq_int4_4_8 - 2.5) < 1e-12)
    check("B", "★ 两个口径差 %.1f 倍——**不得混称**"
          % (eq_int4_4_8 / (SPECS.META_BITS_PER_GROUP_4_8 / float(SPECS.GROUP_SIZE_4_8))),
          abs(eq_int4_4_8 / (SPECS.META_BITS_PER_GROUP_4_8 / float(SPECS.GROUP_SIZE_4_8)) - 5.0) < 1e-12)

    sub("B.3 叠加压缩率：**不是简单相乘**（判据：含 metadata 的 bit/元素）")
    print("      组合            | bit/元素 | 相对 fp16 | 朴素相乘 | 朴素高估")
    worst = 0.0
    for key, bits in sorted(SPECS.COMBINED_BITS_PER_ELEM.items(), key=lambda kv: kv[1]):
        naive = SPECS.NAIVE_MULTIPLE[key]
        ratio = SPECS.BASE_ELEM_BITS / bits
        over = naive / ratio
        worst = max(worst, over)
        print("      %-15s | %8.2f | %8.2f× | %7.2f× | %8.2f×"
              % (key, bits, ratio, naive, over))
    # 逐行用规格常量重算，与表核对
    recomputed = {
        "fp16 dense": float(SPECS.ELEM_BITS["fp16"]),
        "int8 dense": float(SPECS.ELEM_BITS["int8"]),
        "int4 dense": float(SPECS.ELEM_BITS["int4"]),
        "fp16+2:4": (SPECS.NONZERO_PER_GROUP_2_4 * SPECS.ELEM_BITS["fp16"] + meta_bits)
                    / float(SPECS.GROUP_SIZE_2_4),
        "int8+2:4": (SPECS.NONZERO_PER_GROUP_2_4 * SPECS.ELEM_BITS["int8"] + meta_bits)
                    / float(SPECS.GROUP_SIZE_2_4),
        "int4+2:4": (SPECS.NONZERO_PER_GROUP_2_4 * SPECS.ELEM_BITS["int4"] + meta_bits)
                    / float(SPECS.GROUP_SIZE_2_4),
        "int4+4:8": (SPECS.NONZERO_PER_GROUP_4_8 * SPECS.ELEM_BITS["int4"]
                     + SPECS.META_BITS_PER_GROUP_4_8) / float(SPECS.GROUP_SIZE_4_8),
    }
    bad = [k for k, v in recomputed.items()
           if abs(v - SPECS.COMBINED_BITS_PER_ELEM[k]) > 1e-12]
    check("B", "★ 表里 %d 行**逐行**由规格常量重算一致" % len(recomputed), not bad,
          "不一致：%s" % bad if bad else "")
    check("B", "★ 朴素相乘**总是高估**（最大 %.2f×）" % worst, worst > 1.0)
    over_42 = (SPECS.NAIVE_MULTIPLE["int4+2:4"]
               / (SPECS.BASE_ELEM_BITS / SPECS.COMBINED_BITS_PER_ELEM["int4+2:4"]))
    over_24 = (SPECS.NAIVE_MULTIPLE["int4+4:8"]
               / (SPECS.BASE_ELEM_BITS / SPECS.COMBINED_BITS_PER_ELEM["int4+4:8"]))
    check("B", "★ int4+2:4 的朴素高估最大（%.2f×；对照 int4+4:8 只有 %.2f×）"
          % (over_42, over_24), over_42 > over_24 and over_42 > 1.4)
    # ★ 把 `NAIVE_MULTIPLE` **全表**重算一遍：稠密行必须恰好等于稠密倍数（高估 = 1.00×），
    #   稀疏行必须恰好等于它的二倍。**逐行**核，不只核两行。
    naive_bad = []
    for key, naive in SPECS.NAIVE_MULTIPLE.items():
        elem = key.replace(" dense", "").split("+")[0]      # "fp16+2:4" → "fp16"
        want = SPECS.BASE_ELEM_BITS / float(SPECS.ELEM_BITS[elem]) * (2.0 if "+" in key else 1.0)
        if abs(naive - want) > 1e-12:
            naive_bad.append(key)
    check("B", "★ `NAIVE_MULTIPLE` %d 行**逐行**重算一致（稠密行 = 稠密倍数，稀疏行 = 其 2 倍）"
          % len(SPECS.NAIVE_MULTIPLE), not naive_bad,
          "不一致：%s" % naive_bad if naive_bad else "")
    dense_keys = [k for k in SPECS.NAIVE_MULTIPLE if "+" not in k]
    check("B", "★ 三个稠密行的「朴素高估」必须**恰好 1.00×**（朴素相乘对稠密行是对的）",
          all(abs(SPECS.NAIVE_MULTIPLE[k]
                  / (SPECS.BASE_ELEM_BITS / SPECS.COMBINED_BITS_PER_ELEM[k]) - 1.0) < 1e-12
              for k in dense_keys))

    sub("B.4 代价模型的结论：**访存受限算子的收益上限 = 1/压缩率**")
    caps = []
    for name in ("fp16", "int8", "int4"):
        w = SPECS.ELEM_BITS[name]
        r = ((SPECS.NONZERO_PER_GROUP_2_4 * w + meta_bits)
             / float(SPECS.GROUP_SIZE_2_4 * w))
        caps.append((name, r, 1.0 / r))
        print("      %-5s + 2:4：A 的字节压到 %.4f 倍 ⇒ 访存受限时上限 %.4f×（不是 2×）"
              % (name, r, 1.0 / r))
    # 判据：上限 = 稠密位宽 / 含 metadata 的位宽。**它必须严格小于 2**——
    # 这正是「2× 只属于算力受限算子」的算术形式；等号成立当且仅当 metadata 免费。
    check("B", "★ 三个位宽下**都严格小于 2×**（fp16 上限 %.4f×、int8 %.4f×、int4 %.4f×）"
          % tuple(c[2] for c in caps),
          all(1.0 < c[2] < 2.0 for c in caps))
    # ★ 方向容易说反，所以断言写成「越宽上限越高」：
    #   元素越窄，同一个 4-bit metadata 占的比例越大 ⇒ **稀疏这一步本身**的收益越小。
    #   （注意这与「int4+2:4 相对 fp16 稠密的压缩倍数更大」**不矛盾**——
    #     那是另一个分母：5.33× 是相对 fp16 稠密，1.33× 是相对 int4 稠密。）
    check("B", "★ 方向：**元素越宽、上限越高**（fp16 %.4f× > int8 %.4f× > int4 %.4f×）"
          % (caps[0][2], caps[1][2], caps[2][2]),
          caps[0][2] > caps[1][2] > caps[2][2])
    check("B", "★ 两个分母**都成立且方向相反**：int4+2:4 相对 fp16 稠密是 %.2f×"
          "（压缩最多），相对 int4 稠密只有 %.4f×（稀疏这一步收益最小）"
          % (SPECS.BASE_ELEM_BITS / SPECS.COMBINED_BITS_PER_ELEM["int4+2:4"], caps[2][2]),
          (SPECS.BASE_ELEM_BITS / SPECS.COMBINED_BITS_PER_ELEM["int4+2:4"])
          > (SPECS.BASE_ELEM_BITS / SPECS.COMBINED_BITS_PER_ELEM["fp16+2:4"])
          and caps[2][2] < caps[0][2])


# ---------------------------------------------------------------------------
# A. 规格侧 × 本机侧
# ---------------------------------------------------------------------------

def run_A():
    rule("A 两条互不依赖的通道：规格（PTX/cuSPARSELt）× 本机实测（torch）")

    sub("A.1 规格侧：硬件粒度表**只定义 N:M 三种**")
    for tag, dtypes, note in SPECS.HW_NM_GRANULARITIES:
        print("      %-12s %-46s %s" % (tag, "/".join(dtypes), note))
    check("A", "★ 表里恰好 %d 档，且**每一档都是 N:M 形式**"
          % len(SPECS.HW_NM_GRANULARITIES),
          len(SPECS.HW_NM_GRANULARITIES) == 3
          and all(":" in tag for tag, _, _ in SPECS.HW_NM_GRANULARITIES))
    check("A", "★ **非结构化不在硬件粒度表里**（规格明确）",
          SPECS.UNSTRUCTURED_IN_HW_GRANULARITY_TABLE is False)
    check("A", "★ 4:8 的说明里写明「零与非零以 2 元素子块成簇」",
          "2 元素子块" in SPECS.HW_NM_GRANULARITIES[2][2])

    sub("A.2 规格侧：cuSPARSELt 的「50% 稀疏」按 dtype 落到**不同粒度**")
    for k in sorted(SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE):
        print("      %-10s → %s" % (k, SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE[k]))
    check("A", "★ float → 1:2、e2m1 → 4:8 paired、其余 → 2:4（三档全在一张表里）",
          SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["float"] == "1:2"
          and SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["e2m1"] == "4:8 paired"
          and SPECS.CUSPARSELT_50PCT_FORMAT_BY_DTYPE["half"] == "2:4")

    sub("A.3 本机侧：实测 metadata 位宽（**与规格的 4 bit/组逐位吻合**）")
    print("      dtype    | meta.dtype | meta 形状     | meta bit/4元素 | meta/原数据 | 合计/原数据")
    meas = {}
    for name in ("float16", "bfloat16", "int8", "float32"):
        r = measure_layout(name)
        meas[name] = r
        print("      %-8s | %-10s | %-13s | %14.4f | %11.4f | %11.4f"
              % (name, r["meta_dtype"], "%d×%d" % tuple(r["meta"].shape),
                 r["meta_per_4elem"], r["meta_over_dense"], r["total_over_dense"]))
    for name in ("float16", "bfloat16", "int8", "float32"):
        r = meas[name]
        check("A", "★ %s：metadata = %.2f bit/「4 元素」、合计/原数据 = %.4f（与 specs 基线一致）"
              % (name, r["meta_per_4elem"], r["total_over_dense"]),
              abs(r["meta_per_4elem"] - SPECS.MEASURED_META_BITS_PER_GROUP[name]) < 1e-9
              and abs(r["total_over_dense"] - SPECS.MEASURED_TOTAL_OVER_DENSE[name]) < 1e-9
              and abs(r["meta_over_dense"] - SPECS.MEASURED_META_OVER_DENSE[name]) < 1e-9)
    check("A", "★ fp16 / bf16 / int8 三者的 metadata 都是 4 bit/组（**逐位吻合规格推导**）",
          all(abs(meas[n]["meta_per_4elem"] - 4.0) < 1e-9
              for n in ("float16", "bfloat16", "int8")))
    # ⚠ 这里**不能**断言「随机数据的往返无损」——随机数据本来就不满足 2:4，
    #   编码器只保留每组两个，剩下的被丢掉。**无损只对「恰好 2:4」的模式成立**，
    #   判据见 C.2 的受控模式表。这一条之前写错过（断言随机往返相等），
    #   跑出来才发现假设是错的。
    check("A", "★ metadata 的 dtype 按元素类型选：fp16/bf16/float32 → int16，int8 → int32",
          meas["float16"]["meta_dtype"] == "int16"
          and meas["bfloat16"]["meta_dtype"] == "int16"
          and meas["float32"]["meta_dtype"] == "int16"
          and meas["int8"]["meta_dtype"] == "int32")
    check("A", "★ 随机稠密数据在 fp16 上**也**不是无损往返（它本来就不满足 2:4）",
          not bool(torch.equal(meas["float16"]["dense"], meas["float16"]["back"])))
    check("A", "★⚠ **float32 的 8 bit/「4 元素」不是「4 bit/组」**——它的编码单元是 %d 元素对"
          % SPECS.MEASURED_FP32_ENCODING_UNIT,
          abs(meas["float32"]["meta_per_4elem"] - 8.0) < 1e-9
          and SPECS.MEASURED_FP32_ENCODING_UNIT == 2)
    check("A", "★ 但两种口径下 **meta/原数据 都是 6.25%**（4 bit per 64 bit 稠密载荷）",
          abs(meas["float32"]["meta_over_dense"] - meas["float16"]["meta_over_dense"]) < 1e-12)

    sub("A.4 公开 API 在 CPU 上直接报错（判据：异常类型与消息）")
    try:
        torch.sparse.to_sparse_semi_structured(torch.zeros(32, 64, dtype=torch.float16))
        ok = False
        msg = ""
    except Exception as exc:                       # noqa: BLE001 —— 就是要抓它
        ok = isinstance(exc, RuntimeError)
        msg = str(exc)
    print("      %s" % msg[:110])
    check("A", "★ `to_sparse_semi_structured` 在 CPU 上抛 RuntimeError（所以本程序用内部函数）",
          ok and "CUDA" in msg)


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：**torch 的 float32 路径是 1:2，不是 2:4**（静默丢数据）")

    pattern_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    enc, dec = _enc_dec()

    sub("C.1 用 6 种 2:4 模式做**判别实验**")
    print("      判据：假设 A（2:4）⇒ 6 种全过；假设 B（1:2，每对至多一个非零）")
    print("            ⇒ 只有同对两个非零的 (0,1) 与 (2,3) 失败。")
    print("      下表是**实测**（float32 / 32×64 / 值 1.0 与 2.0）：")
    print()
    print("      模式 (p0,p1) | 往返相等 | maxdiff | 假设 A 预言 | 假设 B 预言")
    got = {}
    lost = {}
    for (p0, p1) in pattern_pairs:
        dense = pattern_dense(32, 64, torch.float32, p0, p1)
        sparse, meta = enc(dense)
        back = dec(sparse, meta)
        eq = bool(torch.equal(dense, back))
        md = (dense - back).abs().max().item()
        got[(p0, p1)] = eq
        lost[(p0, p1)] = int((dense != back).sum())
        print("      %-12s | %-8s | %7.1f | %-11s | %s"
              % ("(%d,%d)" % (p0, p1), eq, md, "True",
                 "True" if p0 // 2 != p1 // 2 else "**False**"))
    check("C", "★ 实测结果是**假设 B**（1:2）：只有 (0,1) 与 (2,3) 失败",
          got[(0, 1)] is False and got[(2, 3)] is False
          and all(got[p] for p in pattern_pairs if p != (0, 1) and p != (2, 3)))
    check("C", "★ 与 specs 的基线表**逐格一致**（6/6）",
          got == SPECS.MEASURED_FP32_PATTERN_ROUNDTRIP)
    # 判据：每次「同对两个非零」丢掉**恰好一个**元素 ⇒ 每行丢 k/4 个、整块丢 numel/4。
    # （这是可证伪的定量预言；若编码器改成「保住某个、丢掉另一个」，数量仍然成立。）
    check("C", "★ **失败是静默的、且丢的数量精确**：两种失败模式都丢 %d 个元素，"
          "= 2048/4（每行 16 个）；**不抛任何异常**"
          % lost[(0, 1)],
          lost[(0, 1)] == 2048 // 4 and lost[(2, 3)] == 2048 // 4
          and all(lost[p] == 0 for p in pattern_pairs
                  if p != (0, 1) and p != (2, 3)))

    sub("C.2 同样 6 种模式在 fp16 与 int8 上**全部无损**（对照组）")
    print("      判据：若 C.1 的失败是「模式本身非法」造成的，fp16 也该失败。")
    rows = []
    for name, dt in (("float16", torch.float16), ("int8", torch.int8)):
        oks = []
        for (p0, p1) in pattern_pairs:
            dense = pattern_dense(32, 64, dt, p0, p1)
            if name == "int8":
                dense = dense.to(torch.int8)
            sparse, meta = enc(dense)
            back = dec(sparse, meta)
            oks.append(bool(torch.equal(dense, back)))
        rows.append((name, oks))
        print("      %-8s : %s" % (name, " ".join("T" if o else "F" for o in oks)))
    check("C", "★ 对照组：fp16 / int8 六种模式**全过** ⇒ 失败只与 dtype=float32 有关",
          all(all(o) for _, o in rows))

    sub("C.3 随机稠密矩阵的往返**必然失败**——但它**证明不了**是 1:2")
    torch.manual_seed(0)
    dense = (torch.randn(32, 64) > 0.5).float() * torch.randn(32, 64)
    sparse, meta = enc(dense)
    back = dec(sparse, meta)
    same = bool(torch.equal(dense, back))
    n_bad = int((dense != back).sum())
    # 判据：把每行按 1:2 的「对」切开，数出**两元素都非零**的对数。
    pairs = dense.view(dense.shape[0], dense.shape[1] // 2, 2)
    both = int(((pairs[:, :, 0] != 0) & (pairs[:, :, 1] != 0)).sum())
    print("      随机（每元素 p=0.5 非零）的往返：相等 = %s，失配 %d / %d 个元素"
          % (same, n_bad, dense.numel()))
    print("      其中「同一对内两个都非零」的对数 = %d（= 2048 个元素里的 1024 对）" % both)
    # ★ 判据写成不等式而不是等式：本机实测失配 **1044 > 1024**——多出来的 20 个说明
    #   还有第二种丢法（同一对里只有一个非零、但被选错）。**等式会假**，不等式是真的。
    check("C", "★ 随机矩阵**必然**失配（相等 = %s），且失配数 ≥ 「同对两非零」的对数" % same,
          not same and n_bad >= both)
    print()
    print("      ⚠ 所以这条**不能**用来判定「1:2 还是 2:4」：随机数据下它 100% 失败，")
    print("        失败本身不指向任何具体假设。能区分的只有 C.1 那 6 种**受控模式**。")
    print("        ——这正是「C 类样本要专门挑两种候选解释会给出不同答案的点」的含义。")
    check("C", "★ 设计判据的方法论自检：受控模式表能区分两假设（6 种里 4 过 2 败），"
          "随机矩阵不能（0 过 100% 败）",
          (not same) and len(pattern_pairs) == 6
          and sum(1 for p in pattern_pairs if got[p]) == 4)


# ---------------------------------------------------------------------------
# F. 变异测试清单
# ---------------------------------------------------------------------------

def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    print("      未读到的（由 verify.py 负责）：%s"
          % ", ".join(n for n in names if n not in used))
    check("B", "本程序读到的常量数 ≥ 20", len(used) >= 20)
    print()
    print("  · 测量条件常量自检（把打印出来的东西变成**能被查**的东西）")
    check("B", "★ 本机 torch 版本被 specs 的「跑过的版本」清单覆盖",
          torch.__version__ in SPECS.MEASURED_VERSIONS)
    # ★ 清单的**条数**也是判据：多一条少一条都会让「跑过哪些版本」这句话失真。
    check("B", "★ 清单恰好 %d 条（%s）——**条数本身是判据**"
          % (len(SPECS.MEASURED_VERSIONS), " / ".join(SPECS.MEASURED_VERSIONS)),
          len(SPECS.MEASURED_VERSIONS) == 5)
    # ⚠ 这里**不能**断言「当前版本 ≥ 版本下限」：那个下限是**对齐选择**，不是依赖。
    #   本仓库实测：显式指定 PYTHON= 时，2.6.0 / 2.7.1 上本程序也全部通过。
    below = tuple(int(x) for x in torch.__version__.split(".")[:2]) \
        < tuple(SPECS.TORCH_MIN_FOR_THIS_TOPIC)
    print("      当前版本 %s %s 版本下限 %s —— %s"
          % (torch.__version__, "低于" if below else "不低于",
             SPECS.TORCH_MIN_FOR_THIS_TOPIC,
             "（显式指定解释器，本仓库实测本程序仍全部通过）" if below else ""))
    # ★ 转录锚点：这是一句**说明文字**，作用是让读者知道「为什么不能只 find_spec」。
    #   除逐字对之外，没有别的机械办法保证它没被改过（A6 要求每条常量都能被分辨）。
    check("B", "★ 转录锚点：版本下限的理由逐字对上（%s）" % SPECS.TORCH_MIN_REASON,
          SPECS.TORCH_MIN_REASON ==
          "对齐选择，不是依赖（本主题的编号程序在 2.6.0 上也能跑通）"
          and "对齐选择" in SPECS.TORCH_MIN_REASON
          and "不是依赖" in SPECS.TORCH_MIN_REASON)
    # ★ 下限必须是**两元组**：Makefile 里比较的两边都取 `[:2]`，
    #   **长度不等会让元组比较静默判负**（`(2,8) >= (2,8,0)` 是 False）。
    check("B", "★ 版本下限是**两元组**（%s）——与 Makefile 里 `[:2]` 的那一边等长"
          % (SPECS.TORCH_MIN_FOR_THIS_TOPIC,),
          len(SPECS.TORCH_MIN_FOR_THIS_TOPIC) == 2
          and all(isinstance(v, int) for v in SPECS.TORCH_MIN_FOR_THIS_TOPIC))
    check("B", "★ 测量条件写明「CPU-only、无 GPU 数字」，未覆盖的 minor 也列出",
          "CPU-only" in SPECS.MEASURE_HOST and bool(SPECS.UNCOVERED_MINORS))
    # ★ 转录锚点：这两行是**要写进文档 §5 的测量条件**，逐字对齐才有意义
    #   （只查「含 CPU-only」太松——后面追加任何字符都能过）。
    check("B", "★ 转录锚点：测量条件那一行逐字对上", 
          SPECS.MEASURE_HOST == "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**")
    check("B", "★ 转录锚点：未覆盖的 minor 那一行逐字对上（%s）" % SPECS.UNCOVERED_MINORS,
          SPECS.UNCOVERED_MINORS == "2.9 / 2.10 / 2.12（本机无解释器）")


def main():
    banner()
    run_B()
    run_A()
    run_C()
    run_F()
    print()
    print("=" * 74)
    print("合计 %d 条断言：A %d / B %d / C %d；失败 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"], len(_STATE["fail"])))
    for f in _STATE["fail"]:
        print("  · %s" % f)
    sys.exit(0 if not _STATE["fail"] else 1)


if __name__ == "__main__":
    main()
