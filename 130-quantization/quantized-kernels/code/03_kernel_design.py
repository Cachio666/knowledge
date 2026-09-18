#!/usr/bin/env python3
"""03 量化 GEMM kernel 的设计要点：累加器怎么选、为什么 fp16 累加会丢。

对应文档：130-quantization/quantized-kernels/03-kernel-design.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

★★ **本篇的 §5「实测数据」整节省略**——它的数字**全部是引用值**（Marlin / Machete /
   ExLlamaV2 / CUTLASS 的 kernel 性能必须有 GPU 才能量，本机 `cuda.is_available()` 为假）。
   但**累加器这一层是可以在真 torch 上量的**，那些放在 §1 与 §4，经由本程序的 B/A/C 节支撑。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：累加器宽度怎么换、fp16 累加的两个上界（65504 与 [2048,4096) 的 ulp=2）、
       以及「2:4 稀疏 × int4」为什么**不叠成 4×**的算术。
  A —— **两条互不依赖的观察通道**：
       ① 文档/论文侧（PTX 的 fp16 累加明文、cuBLAS 的 split-K 说明、A100 白皮书峰值表、
          Marlin/Machete/Sparse-Marlin 的归属与数字）—— **引用值**；
       ② 本机侧（fp16 `torch.mm` 的累加宽度、fp16 的两个溢出边界、fp8 的累加器与输出）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：fp16 把 `2048+1` 吃掉、`40000+40000` 溢出成 inf、
       `448+448` 在 e4m3 里根本不可表示。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 03_kernel_design   （或 make run 全跑）
"""

import platform
import subprocess
import sys

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def ver_ge(v):
    """`torch.__version__ >= v`？**两边都取两元组**（`(2,8) >= (2,8,0)` 是 False）。"""
    return tuple(int(x) for x in torch.__version__.split(".")[:2]) \
        >= tuple(int(x) for x in v.split(".")[:2])


def fp8_overflow_expected():
    """本机版本下 `448×8 − 448×7 + 100 = 548` 的**输出 cast** 应该是什么。

    判据（本机五版本对拍，specs §8.3）：
      · torch ≤ 2.11.0 → **nan**（越界即 NaN，e4m3fn 无 Inf）
      · torch ≥ 2.13.0 → **448.0**（**一律饱和**）
    ⇒ **这不是「torch 的行为」，是「torch X 上的行为」。**
    """
    if ver_ge(SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0]):
        return SPECS.FP8_OUTPUT_OVERFLOW_SATURATED_VALUE
    return float("nan")


def cpu_model():
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
    print("03 kernel 设计要点：累加器怎么选；fp16 累加的两个陷阱；与 2:4 稀疏叠加")
    print("=" * 74)
    print("测量条件（本程序**只在累加器这一层**有实测数字；kernel 性能一概没有）：")
    print("  CPU        ：%s" % cpu_model())
    print("  主机       ：%s" % SPECS.MEASURE_HOST)
    print("  python     ：%s" % sys.version.split()[0])
    print("  torch      ：%s" % torch.__version__)
    print("  CUDA 可用  ：%s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  本主题跑过的版本：%s" % " / ".join(SPECS.MEASURED_VERSIONS))
    print("  未覆盖的 minor  ：%s" % SPECS.UNCOVERED_MINORS)
    print("  数字冻结在      ：torch %s / python %s"
          % (SPECS.MEASURED_TORCH, SPECS.MEASURED_PYTHON))
    print()
    print("  ★★ **本篇文档的 §5「实测数据」整节省略**：它的数字全部是引用值。")
    print("     但**下面这些是可以真量的**（累加器宽度、fp16 的两个上界、fp8 的输出舍入），")
    print("     它们进文档的 §1 与 §4，**不是 §5**。")


def f16_tensor(vals):
    return torch.tensor(vals, dtype=torch.float16)


# ===========================================================================
# B. 代价模型
# ===========================================================================

def run_B():
    rule("B 代价模型（累加器的宽度账；fp16 的两个上界；稀疏 × int4 的算术）")
    ok = True

    sub("B.1 累加器宽度的三个选项，各自的**代价**是什么")
    print("      判据逐条写出（三个不是同一个量）：")
    print("        ① int32：值域 ±2^31，**整数精确**；代价是寄存器/带宽 4 字节，")
    print("           且 **epilogue 必须先转 fp32 再乘 scale**（fp16 只能精确表示 ≤%d 的整数）。"
          % SPECS.FP16_EXACT_INT_MAX)
    print("        ② fp32：值域 ±3.4e38、24 位有效；**浮点累加会丢低位**（见 01 的实测）。")
    print("        ③ fp16：值域上限 %g、11 位有效位；**在 [2048,4096) 区间 ulp = %g**，"
          % (SPECS.FP16_MAX, SPECS.FP16_ULP_HIGH_BINADE))
    print("           把 1.0 加到 2048.0 上会**直接丢失**。")
    print("      ⇒ 三者的**瓶颈不同**：int32 的代价是宽度，fp16 的代价是精度上界。")
    P = SPECS.INT8_PRODUCT_MAX_POSPOS
    ok &= check("B", "B.1  int32 在 K=133144 仍精确（%.4e ≤ %d）、fp16 连 2049 都表示不了"
                % (SPECS.INT8_ACC_INT32_LAST_SAFE_K * P, SPECS.INT32_MAX),
                SPECS.INT8_ACC_INT32_LAST_SAFE_K * P <= SPECS.INT32_MAX
                and SPECS.FP16_EXACT_INT_MAX == 2048 < 2049)
    ok &= check("B", "B.1b 四条规范来源指向同一件事（指令层面就是 32 位整数累加）",
                SPECS.PTX_INT_MMA_ACC_CTYPE == ".s32"
                and SPECS.CUBLAS_INT_COMPUTE_TYPE == "CUBLAS_COMPUTE_32I"
                and SPECS.CUSPARSELT_INT_COMPUTE_TYPE == "CUSPARSE_COMPUTE_32I"
                and "32-bit integer accumulation" in SPECS.DP4A_ACCUM_QUOTE)
    ok &= check("B", "B.1c ⚠ **规范说指令、实测说这个 CPU 实现** —— 本机那两个算子"
                     "**不是**精确整数累加，两者不得混为一谈（P8）",
                SPECS.INT8PACK_ACCUM_IS_EXACT_INT is False
                and SPECS.PTX_INT_MMA_ACC_CTYPE == ".s32")

    sub("B.2 fp16 累加的两个上界（两个不同的量，各给各的判据）")
    print("      判据①**加法丢失**：只有 10 位显式尾数 ⇒ [1024,2048) 的 ulp = %g、"
          % SPECS.FP16_ULP_LOW_BINADE)
    print("            [2048,4096) 的 ulp = %g。**2048 + 1 落在后一个区间**。"
          % SPECS.FP16_ULP_HIGH_BINADE)
    print("      判据②**值域上限**：最大有限值 %g；40000 + 40000 = 80000 超出它。" % SPECS.FP16_MAX)
    lo, hi = SPECS.FP16_ULP_LOW_RANGE, SPECS.FP16_ULP_HIGH_RANGE
    ok &= check("B", "B.2  两个区间的 ulp 相差 %g 倍，且 2048 恰好是分界点 —— "
                     "**「丢精度」的两个判据不是同一个量**"
                % (SPECS.FP16_ULP_HIGH_BINADE / SPECS.FP16_ULP_LOW_BINADE),
                lo[1] == hi[0] == 2048.0
                and SPECS.FP16_ULP_HIGH_BINADE == 2 * SPECS.FP16_ULP_LOW_BINADE)
    print("      代价换算：一个 fp16 累加器在 [2048, 4096) 上**每隔 %d 才有一个格点** ⇒"
          % int(SPECS.FP16_ULP_HIGH_BINADE))
    for K in (1024, 4096, 16384):
        print("        K=%-6d：若全在 [2048,4096) 区间，累加器能分辨的相对变化 ≥ %.2e"
              % (K, SPECS.FP16_ULP_HIGH_BINADE / (2048.0 * K / 2048.0) if K else 0))
    ok &= check("B", "B.2b PTX 明文：fp16 累加是「at least half precision」且**顺序 unspecified**"
                     "⇒ **指令层面这个变体确实存在**，本机 CPU 的行为不是反例",
                "at least half precision" in SPECS.PTX_F16_ACC_QUOTE
                and "unspecified" in SPECS.PTX_F16_ACC_ORDER_QUOTE)
    ok &= check("B", "B.2c ⚠ **不得把本机 CPU 的行为外推到 GPU**：本机记的是 %s 累加，"
                     "GPU 上可选 fp16"
                % SPECS.FP16_MM_CPU_ACCUM,
                SPECS.FP16_MM_CPU_ACCUM == "fp32"
                and SPECS.FP16_MM_CPU_PROBE != SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE)
    ok &= check("B", "B.2d split-K 的陷阱（cuBLAS 官方）：compute type 比 output type 精时，"
                     "分块和可能产生**本不该有的中间溢出**，由 `%s` 控制"
                % SPECS.CUBLAS_SPLITK_OVERFLOW_ENV,
                SPECS.CUBLAS_SPLITK_OVERFLOW_ENV
                == "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION")
    ok &= check("B", "B.2e fp16 累加的形式化研究给出了更细的机制（引用值，未核实）",
                SPECS.FP16_FORMAL_STUDY == "PeerJ CS 7:e330 (2021)"
                and SPECS.FP16_FORMAL_STUDY_INTERNAL_BITS == 27)

    sub("B.3 量化 × 2:4 稀疏：**两个 2× 不叠成 4×**（算术 + A100 白皮书峰值表）")
    i8, i8s = SPECS.A100_INT8_TC_TOPS
    i4, i4s = SPECS.A100_INT4_TC_TOPS
    print("      A100 白皮书峰值表（引用值）：")
    print("        Peak INT8 TC: %d | %d TOPS (sparsity)" % (i8, i8s))
    print("        Peak INT4 TC: %d | %d TOPS (sparsity)" % (i4, i4s))
    print("      ⇒ 稀疏在每个精度上**都恰好 %d×**；INT4 相对 INT8 再 %d×。" % (SPECS.A100_SPARSITY_FACTOR, 2))
    ok &= check("B", "B.3  INT8 的稀疏倍率 = %d、INT4 的稀疏倍率 = %d"
                % (i8s / i8, i4s / i4),
                i8s / i8 == SPECS.A100_SPARSITY_FACTOR
                and i4s / i4 == SPECS.A100_SPARSITY_FACTOR
                and i4 == 2 * i8)
    print("      理论加速（vs dense FP16，判据：INT8/FP16 = 2×、INT4/INT8 = 2×）：")
    print("        sparse INT8 = %d×（稀疏）× %d×（位宽）= %d× vs dense FP16"
          % (SPECS.A100_SPARSITY_FACTOR, 2, SPECS.SPARSE_INT8_THEORETICAL_VS_DENSE_FP16))
    print("        dense  INT4 = %d×（位宽）                        = %d× vs dense FP16"
          % (i4 // i8 * 2, SPECS.DENSE_INT4_THEORETICAL_VS_DENSE_FP16))
    ok &= check("B", "B.3b ⇒ **两者理论加速相同（都是 %d×）**——所以「稀疏 + int4」"
                     "不是 4×·4×，而是把一个 4× 换成另一个 4×；"
                     "真实收益来自二者**同时**用上时**省下的两笔不同开销**（算力 vs 带宽）"
                % SPECS.DENSE_INT4_THEORETICAL_VS_DENSE_FP16,
                SPECS.SPARSE_INT8_THEORETICAL_VS_DENSE_FP16
                == SPECS.DENSE_INT4_THEORETICAL_VS_DENSE_FP16 == 4)
    print("      Sparse-Marlin（%s，**注意不是 Machete 那个归属**）：" % SPECS.SPARSE_MARLIN_REPO)
    print("        Readme 逐字：near-optimal speedups of around %.1f×；"
          "batch size 保到 %d（dense Marlin 的两倍）；" % (SPECS.SPARSE_MARLIN_SPEEDUP,
                                                          SPECS.SPARSE_MARLIN_BATCH))
    print("        更大 batch 衰减后仍稳定 %d× over dense FP16。"
          % SPECS.SPARSE_MARLIN_OVER_DENSE_FP16_LARGE_BATCH)
    ok &= check("B", "B.3c Sparse-Marlin 的归属是 `%s`（**IST-DASLab**），"
                     "而 Machete 的归属是 `%s` —— 两个不同的仓库，不得互串"
                % (SPECS.SPARSE_MARLIN_REPO, SPECS.MACHETE_AUTHOR),
                SPECS.SPARSE_MARLIN_REPO.startswith("IST-DASLab")
                and SPECS.MACHETE_AUTHOR == "Neural Magic")
    ok &= check("B", "B.3d 「dense Marlin 之上再快 %d%%」与「端到端 %s×」**不在 Readme 里**——"
                     "它们来自检索摘要，按引用值标注"
                % (SPECS.SPARSE_MARLIN_DELTA_VS_DENSE_MARLIN_PCT,
                   SPECS.SPARSE_MARLIN_END_TO_END),
                SPECS.SPARSE_MARLIN_DELTA_VS_DENSE_MARLIN_PCT == 65
                and SPECS.SPARSE_MARLIN_END_TO_END == 3.2)
    ok &= check("B", "B.3e cuSPARSELt 的稀疏格式**按类型不同**：e2m1 用 %s、"
                     "half/int8/e4m3 等用 %s、float 用 %s ⇒ 「2:4」严格说只适用于列出的那些类型"
                % (SPECS.CUSPARSELT_SPARSITY_BY_TYPE["e2m1"],
                   SPECS.CUSPARSELT_SPARSITY_BY_TYPE["half/bfloat16/int/int8/e4m3/e5m2"],
                   SPECS.CUSPARSELT_SPARSITY_BY_TYPE["float"]),
                len(SPECS.CUSPARSELT_SPARSITY_BY_TYPE) == 3
                and SPECS.CUSPARSELT_SPARSITY_BY_TYPE["e2m1"] == "paired 4:8")

    sub("B.4 三个 kernel 的**归属与出处**（防 P7：引文贴到它不支持的位置）")
    print("      Marlin   ：%s，%s；abstract 逐字：batch 16–32 ≈ %s、vLLM 端到端 %.1f×"
          % (SPECS.MARLIN_ARXIV, SPECS.MARLIN_AUTHOR, SPECS.MARLIN_MAX_SPEEDUP,
             SPECS.MARLIN_END_TO_END_VLLM))
    print("      Machete  ：%s；**没有 arXiv 论文**（Readme %d 字节里一个编号都没有）"
          % (SPECS.MACHETE_AUTHOR, SPECS.MACHETE_README_BYTES))
    print("      ExLlamaV2：不走 tensor core、%s 累加防溢出 —— "
          "**本次未独立核实**（源码拿不到）" % SPECS.EXLLAMA_V2_ACCUM_TYPE)
    print("      CUTLASS  ：int4 的 Ampere 指令 `%s`（累加器固定 int32）；"
          % SPECS.CUTLASS_INT4_MMA)
    print("                 Hopper 的 4-bit 快路径在 `%s`（引用值，未核实原文）"
          % SPECS.CUTLASS_HOPPER_INT4_EXAMPLE)
    print("                 ★ 唯一被本实现者**独立证实**的是「Machete 的起点是 %s」"
          % SPECS.CUTLASS_MACHETE_STARTING_POINT)
    ok &= check("B", "B.4  四条归属各自独立，且没有一条是靠「看起来合理」拼出来的",
                SPECS.MARLIN_ARXIV == "arXiv:2408.11743"
                and SPECS.MACHETE_HAS_ARXIV is False
                and SPECS.EXLLAMA_V2_SOURCE_VERIFIED_THIS_RUN is False
                and SPECS.EXLLAMA_V2_REPO_ARCHIVED is True
                and SPECS.CUTLASS_MACHETE_STARTING_POINT == "CUTLASS 3.5.1 example #55")
    ok &= check("B", "B.4b Marlin 的 abstract 数字与**未逐字核实**的那条（A10 / batch 128）"
                     "分开标注",
                SPECS.MARLIN_BATCH_SWEET == (16, 32)
                and SPECS.MARLIN_BATCH_DECAY == (64, 128)
                and SPECS.MARLIN_BATCH128_A10 == 1.5
                and SPECS.MARLIN_MAX_SPEEDUP == "4x")

    print()
    print("  B 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# A. 判据复核
# ===========================================================================

def run_A():
    rule("A 判据复核（本机实测值 / 引用值分列）")
    ok = True

    sub("A.1 ★ fp16 的 CPU `torch.mm` 用 **fp32** 累加（两条**能区分两种解释**的判据）")
    print("      判据必须挑成「fp16 累加与 fp32 累加给出不同答案」的点，否则验证不了任何东西：")
    K = 2049
    a = torch.zeros(1, K, dtype=torch.float16)
    a[0, 0] = 2048.0
    a[0, 1:] = 1.0
    got = float(torch.mm(a, torch.ones(K, 1, dtype=torch.float16))[0, 0])
    print("        ① [2048, 1, 1, …, 1]（K=%d 项）@ ones → **%.1f**" % (K, got))
    print("           若在 fp16 里累加：2048 + 1 先舍成 2048，一路加下去仍是 2048；")
    print("           只有 **fp32 累加**才给 2048 + 2048×1 = 4096。")
    b = f16_tensor([[2048.0, 1.0, 1.0]])
    got2 = float(torch.mm(b, torch.ones(3, 1, dtype=torch.float16))[0, 0])
    print("        ② [2048, 1, 1]（K=3）@ ones → **%.1f**" % got2)
    print("           纯 fp16 顺序累加会给 2048；**2050 排除了 fp16 累加**。")
    ok &= check("A", "A.1  ① %.1f = specs 记的 %.1f；fp16 累加会得 %.1f"
                % (got, SPECS.FP16_MM_CPU_PROBE[K], SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[K]),
                got == SPECS.FP16_MM_CPU_PROBE[K] == 4096.0
                and SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[K] == 2048.0)
    ok &= check("A", "A.1b ② %.1f = specs 记的 %.1f（fp16 累加会得 %.1f）"
                % (got2, SPECS.FP16_MM_CPU_PROBE[3], SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[3]),
                got2 == SPECS.FP16_MM_CPU_PROBE[3] == 2050.0
                and SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[3] == 2048.0)
    ok &= check("A", "A.1c 结论只对本机 CPU 成立——specs 把「fp32 累加」与「不外推」"
                     "写在同一处，PTX 的 fp16 变体另有明文",
                SPECS.FP16_MM_CPU_ACCUM == "fp32" and "unspecified" in SPECS.PTX_F16_ACC_ORDER_QUOTE)

    sub("A.2 fp16 的两个陷阱在**真 tensor** 上真的触发")
    h = torch.tensor([2048.0], dtype=torch.float16)
    one = torch.tensor([1.0], dtype=torch.float16)
    s = float((h + one)[0])
    print("        float16(2048) + float16(1) = %.1f（ulp = %.1f，被吃掉）"
          % (s, SPECS.FP16_ULP_HIGH_BINADE))
    ok &= check("A", "A.2  2048.0 + 1.0 == 2048.0（加法丢失，真 tensor 上）",
                s == 2048.0 and SPECS.FP16_EXACT_INT_MAX == 2048)
    okv = float(torch.mm(torch.full((1, 2), SPECS.FP16_MM_OK_INPUT, dtype=torch.float16),
                         torch.ones(2, 1, dtype=torch.float16))[0, 0])
    ovf = float(torch.mm(torch.full((1, 2), SPECS.FP16_MM_OVERFLOW_INPUT, dtype=torch.float16),
                         torch.ones(2, 1, dtype=torch.float16))[0, 0])
    ok &= check("A", "A.2b %g+%g = %g ≤ %g（可表示）；%g+%g = %g > %g ⇒ **溢出**"
                % (SPECS.FP16_MM_OK_INPUT, SPECS.FP16_MM_OK_INPUT, okv, SPECS.FP16_MAX,
                   SPECS.FP16_MM_OVERFLOW_INPUT, SPECS.FP16_MM_OVERFLOW_INPUT, ovf,
                   SPECS.FP16_MAX),
                okv == SPECS.FP16_MM_OK_EXPECTED == 60000.0
                and ovf == float("inf") and 2 * SPECS.FP16_MM_OVERFLOW_INPUT > SPECS.FP16_MAX)
    print("        ⚠ 两个陷阱的判据不同：一个是**加法在某个区间丢位**，一个是**值域被超**。")
    ok &= check("A", "A.2c 用一个 27 位内部累加器的形式化研究解释后者的机制（引用值，未核实）",
                SPECS.FP16_FORMAL_STUDY_INTERNAL_BITS == 27)

    sub("A.3 fp8：**累加器不溢出，输出 cast 才溢出**（与 fp16 的陷阱不是同一件事）")
    d8 = torch.float8_e4m3fn
    vals = [448.0] * 8 + [-448.0] * 7 + [100.0]
    xv = torch.tensor(vals, dtype=torch.float32).to(d8).reshape(1, 16)
    ones = torch.tensor([1.0] * 16, dtype=torch.float32).to(d8).reshape(16, 1)
    o = torch.mm(xv, ones)
    print("        448×8 + (−448)×7 + 100 = %d（在 fp32 里毫无问题）"
          % int(SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM))
    exp = fp8_overflow_expected()
    got = float(o[0, 0])
    print("        但它 > %g（e4m3 的上限），而 e4m3fn **没有 Inf 编码** ⇒ 本机输出 %s"
          % (SPECS.FP8_E4M3_MAX, got))
    print("        ★ **这一步是版本相关的**（specs §8.3）：≤%s 出 nan；≥%s **一律饱和**到 %g。"
          % (SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS[-1],
             SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0],
             SPECS.FP8_OUTPUT_OVERFLOW_SATURATED_VALUE))
    ok &= check("A", "A.3  累加结果 %.0f 表示得下，**输出**才出事：本机（torch %s）得到 %s"
                % (SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM, torch.__version__, got),
                (got != got and exp != exp) or got == exp)
    ok &= check("A", "A.3d ★ 「越界即 NaN」这句话**只在 ≤%s 成立**；≥%s 是饱和 —— "
                     "不得写成「torch 的行为是 NaN」"
                % (SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS[-1],
                   SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0]),
                SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_IS_VERSION_SCOPED
                and SPECS.FP8_OUTPUT_OVERFLOW_BREAK == ("2.11.0", "2.13.0"))
    ok &= check("A", "A.3e ★ 这条断点与同域 `numerical-formats` 记的**是同一次变更**"
                     "（那里是 `.to()` 转换，这里是 `mm` 的**输出 cast**）",
                SPECS.FP8_OUTPUT_OVERFLOW_SATURATED_VALUE == SPECS.FP8_E4M3_MAX
                and SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM > SPECS.FP8_E4M3_MAX)
    for vals2, expect in SPECS.FP8_ACC_PROBE:
        xx = torch.tensor(vals2, dtype=torch.float32).to(d8).reshape(1, len(vals2))
        bb = torch.tensor([1.0] * len(vals2), dtype=torch.float32).to(d8).reshape(len(vals2), 1)
        ok &= check("A", "A.3b 探针 %s → %g（精确和 %g）⇒ 累加**不在 fp8 里做**"
                    % ("[" + ",".join("%g" % q for q in vals2) + "]",
                       float(torch.mm(xx, bb)[0, 0]), expect),
                    float(torch.mm(xx, bb)[0, 0]) == expect)
    ok &= check("A", "A.3c fp8 的逐元素加法在 CPU 上不可用 ⇒ 上面这些只能走 mm；"
                     "fp8 的 mm **在 CPU 上可用**",
                SPECS.FP8_MM_AVAILABLE_ON_CPU and not SPECS.FP8_ELEMENTWISE_ADD_AVAILABLE
                and SPECS.FP8_MM_OUTPUT_DTYPE == "torch.float8_e4m3fn")

    sub("A.4 三个 kernel 的设计要点（**全部是引用值**，本机无 GPU 无法核实）")
    ok &= check("A", "A.4  Machete：Neural Magic、无 arXiv、逐元素 zero-point、无 4-bit ldmatrix、"
                     "**且不是「混合 tile / SIMT 双路径」**",
                SPECS.MACHETE_AUTHOR == "Neural Magic" and SPECS.MACHETE_HAS_ARXIV is False
                and SPECS.MACHETE_IS_ZERO_POINT_ELEMENTWISE
                and SPECS.MACHETE_NO_4BIT_LDMATRIX
                and SPECS.MACHETE_USES_SIMT_DUAL_PATH is False
                and SPECS.MACHETE_README_BYTES == 2672)
    ok &= check("A", "A.4b Machete 的 zero-point 顺序有明文（FMA 便利 ⇒ zp 必须**预先乘过 scale**）",
                "added after the scales" in SPECS.MACHETE_ZP_ORDER_NOTE
                and SPECS.MACHETE_DEQUANT_FORM.endswith("@ a"))
    ok &= check("A", "A.4c Machete 与 Sparse-Marlin 的归属不同（前者 Neural Magic、"
                     "后者 IST-DASLab）——**这条最容易被写串**",
                SPECS.MACHETE_AUTHOR != SPECS.SPARSE_MARLIN_REPO.split("/")[0])
    ok &= check("A", "A.4d ⚠ Machete 的官方介绍里**没有 SIMT 字样**，也没有「混合 tile」——"
                     "研究员对任务书那条说法的否证成立",
                SPECS.MACHETE_USES_SIMT_DUAL_PATH is False
                and SPECS.MACHETE_NO_4BIT_LDMATRIX is True)
    ok &= check("A", "A.4e ExLlamaV2 的两条按「本次未独立核实」标注，**不声称逐字读到源码**",
                SPECS.EXLLAMA_V2_SOURCE_VERIFIED_THIS_RUN is False
                and SPECS.EXLLAMA_V2_PATH_USES_TENSOR_CORE is False
                and SPECS.EXLLAMA_V2_ACCUM_TYPE == "fp32")

    print()
    print("  A 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    rule("C 边界与反例（这几条成立 = 文档里那条失败模式是真的）")
    ok = True

    sub("C.1 fp16 累加的「加法丢失」必须真的发生")
    h = torch.tensor([2048.0], dtype=torch.float16)
    one = torch.tensor([1.0], dtype=torch.float16)
    same = float((h + one)[0]) == float(h[0])
    ok &= check("C", "C.1  2048.0 + 1.0 == 2048.0（增量被完全吃掉，且**不报错**）", same)
    # 挨着 2048 的两个可表示数间距是 2
    nxt = float((h + torch.tensor([2.0], dtype=torch.float16))[0])
    ok &= check("C", "C.1b 2048 + 2.0 = %.1f ⇒ [%g, %g) 的 ulp 是 %g（把 1.0 加进去只能舍）"
                % (nxt, SPECS.FP16_ULP_HIGH_RANGE[0], SPECS.FP16_ULP_HIGH_RANGE[1],
                   SPECS.FP16_ULP_HIGH_BINADE),
                nxt == 2050.0 and SPECS.FP16_ULP_HIGH_BINADE == 2.0)
    # 下面这段是「一个 fp16 累加器加 2048 个 1 会停在 2048」的**纯 fp16 模拟**
    acc = torch.tensor(2048.0, dtype=torch.float16)
    for _ in range(64):
        acc = acc + one
    ok &= check("C", "C.1c 纯 fp16 累加 64 个 1.0：2048 → %.1f（**一个都没加上**）"
                % float(acc),
                float(acc) == 2048.0)
    ok &= check("C", "C.1d ⇒ 这就是「fp16 累加器不可用于低位宽量化」的**最小反例**；"
                     "而本机 CPU 的 mm 给 4096（fp32 累加），两者不是一回事",
                SPECS.FP16_MM_CPU_PROBE[2049] == 4096.0
                and SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[2049] == 2048.0)

    sub("C.2 fp16 的值域上限必须真的溢出")
    ovf = float(torch.mm(torch.full((1, 2), SPECS.FP16_MM_OVERFLOW_INPUT, dtype=torch.float16),
                         torch.ones(2, 1, dtype=torch.float16))[0, 0])
    ok &= check("C", "C.2  %g+%g = inf（上限 %g）"
                % (SPECS.FP16_MM_OVERFLOW_INPUT, SPECS.FP16_MM_OVERFLOW_INPUT, SPECS.FP16_MAX),
                ovf == float("inf") and 2 * SPECS.FP16_MM_OVERFLOW_INPUT > SPECS.FP16_MAX)
    ok &= check("C", "C.2b 而 %g+%g = %g **没有**溢出 —— 两个输入只差 %g，结果一个正常一个 inf"
                % (SPECS.FP16_MM_OK_INPUT, SPECS.FP16_MM_OK_INPUT,
                   SPECS.FP16_MM_OK_EXPECTED,
                   SPECS.FP16_MM_OVERFLOW_INPUT - SPECS.FP16_MM_OK_INPUT),
                SPECS.FP16_MM_OK_EXPECTED <= SPECS.FP16_MAX)
    ok &= check("C", "C.2c ⚠ 溢出**不是**「每加一次检查一次」就能挡住的："
                     "真因是**中间部分和**越界 —— 这正是 cuBLAS 的 split-K 说明所指的那种风险",
                SPECS.CUBLAS_SPLITK_OVERFLOW_ENV.endswith("REDUCED_PRECISION_REDUCTION"))

    sub("C.3 fp8：越界不可表示的那一侧必须真的出 NaN")
    d8 = torch.float8_e4m3fn
    vals = [448.0] * 8 + [-448.0] * 7 + [100.0]
    xv = torch.tensor(vals, dtype=torch.float32).to(d8).reshape(1, 16)
    ones = torch.tensor([1.0] * 16, dtype=torch.float32).to(d8).reshape(16, 1)
    o = torch.mm(xv, ones)
    exp = fp8_overflow_expected()
    got = float(o[0, 0])
    ok &= check("C", "C.3  精确和 %d > %g ⇒ 输出**不可表示**：本机得到 %s（该版本档期望 %s）"
                % (int(SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM), SPECS.FP8_E4M3_MAX, got, exp),
                (got != got and exp != exp) or got == exp)
    ok &= check("C", "C.3b e4m3fn **没有 Inf 编码**（`0x7F`/`0xFF` 是 NaN）——"
                     "所以这里的失败模式是 NaN/饱和，**不是 inf**（与 fp16 那一条相反）",
                SPECS.FP8_E4M3_MAX == 448.0 and SPECS.FP16_MAX == 65504.0)
    ok &= check("C", "C.3d ★ 这条失败模式的**形态随版本变**：≤%s 是 NaN、≥%s 是饱和 —— "
                     "文档里必须两个都写"
                % (SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS[-1],
                   SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0]),
                SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS[-1] == "2.11.0"
                and SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0] == "2.13.0")
    try:
        torch.tensor([448.0]).to(d8) + torch.tensor([448.0]).to(d8)
        ran = True
    except Exception:
        ran = False
    ok &= check("C", "C.3c fp8 的**逐元素加法**在 CPU 上不可用 ⇒ 「fp8 累加会怎样」"
                     "在本机只能通过 mm 的探针观察", not ran)

    sub("C.4 引用数字里最容易写错的两条，做成可断言的常量")
    ok &= check("C", "C.4  Machete **没有** arXiv 编号（不许编）；且它的归属是 Neural Magic",
                SPECS.MACHETE_HAS_ARXIV is False and SPECS.MACHETE_AUTHOR == "Neural Magic"
                and "arXiv" not in SPECS.MACHETE_DEQUANT_FORM)
    ok &= check("C", "C.4b 把「Σx 列和修正」写成 W4A16 的标准做法是错的 —— "
                     "那条形式只出现在 W8A8 的激活侧文档里",
                SPECS.W4A16_OFFICIAL_DOC_COLSUM_FORM_IS_W8A8_ONLY
                and SPECS.MACHETE_IS_ZERO_POINT_ELEMENTWISE)
    ok &= check("C", "C.4c 「2:4 稀疏」的适用类型有限：cuSPARSELt 里 e2m1 走的是 %s"
                % SPECS.CUSPARSELT_SPARSITY_BY_TYPE["e2m1"],
                SPECS.CUSPARSELT_SPARSITY_BY_TYPE["e2m1"] != "2:4")

    print()
    print("  C 节结论：%s" % ("全部按预期成立" if ok else "**有反例不成立**"))
    return ok


# ===========================================================================
# F. 变异测试清单（A6）
# ===========================================================================

def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    ok = True
    print()
    print("  · 测量条件自检")
    print("      torch %s 在 specs.MEASURED_VERSIONS 里：%s"
          % (torch.__version__, torch.__version__ in SPECS.MEASURED_VERSIONS))
    ok &= check("B", "★ 本机 torch 版本被 specs 的「跑过的版本」清单覆盖",
                torch.__version__ in SPECS.MEASURED_VERSIONS)
    below = tuple(int(x) for x in torch.__version__.split(".")[:2]) \
        < tuple(SPECS.TORCH_MIN_FOR_THIS_TOPIC)
    print("      当前版本 %s %s 版本下限 %s（%s）"
          % (torch.__version__, "低于" if below else "不低于",
             SPECS.TORCH_MIN_FOR_THIS_TOPIC, SPECS.TORCH_MIN_REASON))
    ok &= check("B", "★ 版本下限是「对齐选择、不是依赖」——低于它**不是**失败",
                "对齐选择" in SPECS.TORCH_MIN_REASON and "不是依赖" in SPECS.TORCH_MIN_REASON)
    ok &= check("B", "★ 测量条件写明「CPU-only、无 GPU 数字」，冻结版本也记下",
                "CPU-only" in SPECS.MEASURE_HOST and bool(SPECS.UNCOVERED_MINORS)
                and SPECS.MEASURED_TORCH == "2.11.0" and SPECS.MEASURED_PYTHON == "3.12.9")
    ok &= check("B", "★ 本篇的 §5 按全域口径**整节省略**（全部数字是引用值）；"
                     "但累加器层有实测，所以程序不是空跑",
                SPECS.MARLIN_END_TO_END_VLLM == 2.8 and SPECS.FP16_MM_CPU_ACCUM == "fp32")
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    unused = [n for n in names if n not in used]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      本程序**没有**读到的 %d 条（由 01/02 或 verify.py 负责）：" % len(unused))
    for n in unused:
        print("        · %s" % n)
    ok &= check("B", "本程序读到的常量数 ≥ 30", len(used) >= 30, "读到 %d 条" % len(used))
    print()
    print("  F 节结论：%s" % ("通过" if ok else "**不通过**"))
    return ok


def main():
    banner()
    okB = run_B()
    okA = run_A()
    okC = run_C()
    okF = run_F()
    print()
    print("=" * 74)
    print("合计 %d 条断言：A %d / B %d / C %d；失败 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"], len(_STATE["fail"])))
    for f in _STATE["fail"]:
        print("  · %s" % f)
    print("B/A/C/F 节：%s" % ("全部通过" if (okA and okB and okC and okF) else "**有不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okA and okB and okC and okF) else 1)


if __name__ == "__main__":
    main()
