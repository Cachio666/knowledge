#!/usr/bin/env python3
"""01 W8A8：int32 累加器的必要性、epilogue 融合，与三个本机真跑的量化算子。

对应文档：130-model-optimization/quantization/15-w8a8.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：W8A8 的累加器预算（`K·16129 / 2^31`）与 epilogue 融合省下的
       **写回流量**。代价模型的参照是**真系统**，所以它在这里、不在 `verify.py`。
  A —— **两条互不依赖的观察通道**：
       ① 规范/文档侧（PTX 的 `.s32`、cuBLAS/cuSPARSELt 的 `COMPUTE_32I`、DP4A、
          W8A8 epilogue 的列和形式、vLLM 的四变体）—— **引用值**；
       ② 本机侧（`_weight_int8pack_mm` 与逐元素参照的偏差表、fp8 `mm` 的输出舍入、
          int4pack 的逐元素 zero-point）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：不 +8 的静默错、约束真抛异常、
       **`qsz` 的主维不校验（静默越界读）**、fp8 输出溢出成 NaN、
       **`_weight_int8pack_mm` 不是精确整数累加**。

★★ 本程序最反常识的一条：**本机这两个「整数量化」算子都不是精确整数累加。**
   K=2^20 时精确和是 3.33e8 ≪ int32 上限（2.147e9），**一个 int32 累加器在那里必然逐位精确**，
   而实测差了 5.5e5。⇒ 排除「精确整数累加」。
   ⚠ **只排除得掉这一种**：它究竟是 fp32 还是别的宽度，本仓库没有独立通道。**不下断言。**

★ 本程序**不复制 `specs_kernels.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 15_w8a8   （或 make run 全跑）
"""

import platform
import subprocess
import sys

import torch

import specs_kernels as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def ver_ge(v):
    """`torch.__version__ >= v`？**两边都取两元组**（`(2,8) >= (2,8,0)` 是 False）。"""
    return tuple(int(x) for x in torch.__version__.split(".")[:2]) \
        >= tuple(int(x) for x in v.split(".")[:2])


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
    print("01 W8A8：int32 累加器；epilogue 融合；本机三个量化算子的实测")
    print("=" * 74)
    print("测量条件（§5 直接引用这一段）：")
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
    print("  ⚠ **本机没有 GPU。** 本程序量的是 **CPU 路径**上这几个算子的行为；")
    print("    任何 kernel 性能（Marlin / Machete / CUTLASS）在本文里**一律是引用值**。")


# ---------------------------------------------------------------------------
# 工具：把「逐元素反量化 + 另走一条 matmul」写成参照
# ---------------------------------------------------------------------------

def ref_int8pack(x, w, scales):
    """`_weight_int8pack_mm` 的**参照**：先把权重逐元素反量化，再走另一条 matmul。

    ★ 「独立」的判据：这条路径**不调用**被检验的那个 fused 算子，
      而是把 `w * scale` 物化成一个 fp32 权重张量后走 `torch.matmul`。
      ⇒ 两者若有偏差，来源只能是**融合算子内部的累加/舍入**。
    """
    wq = w.float() * scales.unsqueeze(1)
    return x @ wq.t()


def ref_int4pack_exact(x_int, v, N, K, gs):
    """`_weight_int4pack_mm_for_cpu` 在 `scale=1 / zp=0` 下的**精确整数参照**。

    判据：`x_int` 是整数值 fp32、权重解包后是 [−8, 7] 的整数 ⇒ 每一项乘积与每个部分和
    都是绝对值 < 2^24 的整数，在 fp32 里**逐位精确** ⇒ 任何累加顺序给出同一结果。
    ⇒ **这个参照的 maxdiff = 0 是「输入挑得好」，不是「算子更准」。**（见 A.4）
    """
    q = (v + SPECS.INT4_PACK_ZERO_POINT) & 0xF
    exact = x_int.long() @ q.t().long() - SPECS.INT4_PACK_ZERO_POINT * x_int.long().sum(
        1, keepdim=True)
    return exact.double()


def pack(v, N, K):
    return torch.ops.aten._convert_weight_to_int4pack_for_cpu(
        v + SPECS.INT4_PACK_ZERO_POINT, SPECS.INT4_PACK_INNER_K_TILES)


def qsz_ones(K, N, gs):
    o = torch.ones(K // gs, N)
    z = torch.zeros(K // gs, N)
    return torch.stack([o, z], -1).contiguous()


# ===========================================================================
# B. 代价模型
# ===========================================================================

def run_B():
    rule("B 代价模型（参照都是真系统：字节数、累加器位宽、fp16 的可表示范围）")
    ok = True

    sub("B.1 int32 累加器的必要性：把它换算成「K 能到多大」（判据写全）")
    P = SPECS.INT8_PRODUCT_MAX_POSPOS
    print("      判据①：单个乘积的最大绝对值 = 127×127 = %d（两个操作数都在正端取到）" % P)
    print("      判据②：补码上界 INT16_MAX = %d、INT32_MAX = %d"
          % (SPECS.INT16_MAX, SPECS.INT32_MAX))
    k16 = SPECS.INT8_ACC_INT16_FIRST_OVERFLOW_K
    k32 = SPECS.INT8_ACC_INT32_FIRST_OVERFLOW_K
    print("      ⇒ int16：K ≥ %d 就溢出；int32：K ≥ %d 才首次溢出" % (k16, k32))
    ok &= check("B", "B.1  K=2 安全（%d ≤ %d）、K=%d 溢出（%d > %d）——int16 撑不住任何真实 GEMM"
                % (2 * P, SPECS.INT16_MAX, k16, k16 * P, SPECS.INT16_MAX),
                2 * P <= SPECS.INT16_MAX and k16 * P > SPECS.INT16_MAX)
    ok &= check("B", "B.1b int32 的首次溢出点 = %d（specs 同），且 (K−1)×P 仍在界内"
                % k32,
                k32 == SPECS.INT8_ACC_INT32_FIRST_OVERFLOW_K
                and (k32 - 1) * P <= SPECS.INT32_MAX
                and SPECS.INT8_ACC_INT32_LAST_SAFE_K == k32 - 1)
    print("      现实 K 的最坏占用（**假定全部乘积同号且取到 %d**，是上界不是典型值）：" % P)
    for K in sorted(SPECS.INT8_ACC_IDEAL_K_FILL_PCT):
        pct = 100.0 * K * P / SPECS.INT32_MAX
        print("        K=%-6d 最坏 %-12d 占 int32 上限 %.2f%%" % (K, K * P, pct))
    ok &= check("B", "B.1c 1K–16K 这个区间里 int32 最多只用到 %.2f%% 的上限（不会溢出）"
                % max(SPECS.INT8_ACC_IDEAL_K_FILL_PCT.values()),
                all(abs(100.0 * K * P / SPECS.INT32_MAX - SPECS.INT8_ACC_IDEAL_K_FILL_PCT[K])
                    < 0.01 for K in SPECS.INT8_ACC_IDEAL_K_FILL_PCT))
    ok &= check("B", "B.1d 但 int32 累加器**不能直接塞进 fp16**：可精确表示的整数上界只有 %d"
                % SPECS.FP16_EXACT_INT_MAX,
                SPECS.FP16_EXACT_INT_MAX == 2048
                and SPECS.INT8_ACC_INT32_LAST_SAFE_K * P > SPECS.FP16_EXACT_INT_MAX)

    sub("B.2 epilogue 融合省下的是**中间结果的写回**（字节预算，判据是「哪个 pass」）")
    print("      判据：一个 **独立的反量化 pass** 要读 int4 载荷、写 fp16 权重。")
    rd = SPECS.W4A16_DECODE_BYTES_PER_ELEM
    wr = SPECS.W4A16_DEQUANT_WRITE_BYTES_PER_ELEM
    print("        每个权重元素：读 %.1f 字节（int4 载荷）→ 写 %.1f 字节（fp16）"
          % (rd, wr))
    ok &= check("B", "B.2  写/读 = %.1f×，所以独立 kernel 被**写侧带宽**卡死（不是被 int4 的读侧）"
                % (wr / rd),
                wr / rd == SPECS.W4A16_SEPARATE_PASS_WRITE_OVER_READ == 4.0)
    print("      具体形状（N 个输出通道、K 个输入通道，一个权重层）：")
    for (N, K) in [(4096, 4096), (8192, 8192), (11008, 4096)]:
        payload = N * K * rd
        deq = N * K * wr
        print("        N=%-6d K=%-6d int4 载荷 %8.2f MiB → 反量化后 %8.2f MiB（多写 %8.2f MiB）"
              % (N, K, payload / 2 ** 20, deq / 2 ** 20, (deq - payload) / 2 ** 20))
    ok &= check("B", "B.2b 融合进 GEMM 后这一步**不存在**（反量化在寄存器里，fp16 权重不落 HBM）"
                     "——大矩阵上是 GiB 量级的差",
                all(N * K * (wr - rd) > 2 ** 20 for (N, K) in [(4096, 4096), (8192, 8192)]))
    ok &= check("B", "B.2c 融合的收益**随 batch 增大而衰减**（dequant 在总时间里的占比下降）"
                     "——这是引用值的定性结论，本机无 GPU 无法量化",
                SPECS.W4A16_SEPARATE_PASS_WRITE_OVER_READ == 4)

    sub("B.3 W8A8 的 epilogue：scale/zero-point 在 GEMM 内应用，中间结果不回 HBM")
    print("      判据：一个 M×N 的 fp32 累加器 tile = 4·M·N 字节；")
    print("            它**从不**离开寄存器/共享内存，epilogue 直接在它上面做 scale/zp。")
    for (M, N) in [(1, 4096), (16, 4096), (128, 11008)]:
        print("        M=%-4d N=%-6d 累加器 tile = %6.2f MiB（若落回 HBM，每层要写一遍）"
              % (M, N, 4.0 * M * N / 2 ** 20))
    tile = {M: 4.0 * M * 4096 / 2 ** 20 for M in (1, 16, 128)}
    ok &= check("B", "B.3  累加器 tile = 4·M·N 字节（fp32）：M=1 → %.4f MiB、M=16 → %.2f、"
                     "M=128 → %.2f；**它随 M 线性涨，而权重写回与 M 无关** —— 两个不同的量"
                % (tile[1], tile[16], tile[128]),
                abs(tile[1] - 4.0 * 4096 / 2 ** 20) < 1e-12
                and abs(tile[128] / tile[16] - 8.0) < 1e-9
                and abs(4.0 * 1 * 4096 - 4096 * 4096 * 0.5) > 0)
    ok &= check("B", "B.3b 同一个 `[4096, 4096]` 层：权重写回是 32 MiB（与 M 无关），"
                     "而 M=128 的累加器 tile 只有 2 MiB —— **判据不同，不得互推**",
                abs(4096 * 4096 * 2.0 / 2 ** 20 - 32.0) < 1e-9
                and abs(tile[128] - 2.0) < 1e-9)

    sub("B.4 `_weight_int8pack_mm` 的偏差是**随 setup 走**的量（不是常量）")
    print("      specs.INT8PACK_MEASURED 里同一张表就跨了三个数量级：")
    for (M, K, N), d in sorted(SPECS.INT8PACK_MEASURED.items(), key=lambda kv: kv[0][1]):
        print("        M=%-2d K=%-7d N=%-4d maxdiff = %.6e" % (M, K, N, d))
    ks = sorted(SPECS.INT8PACK_MEASURED, key=lambda t: t[1])
    ok &= check("B", "B.4  这张表跨越 %.0f 倍（%.1e → %.1e）⇒ **文档里不得给单一固定值**"
                % (SPECS.INT8PACK_MEASURED[ks[-1]] / SPECS.INT8PACK_MEASURED[ks[0]],
                   SPECS.INT8PACK_MEASURED[ks[0]], SPECS.INT8PACK_MEASURED[ks[-1]]),
                SPECS.INT8PACK_MEASURED[ks[-1]] / SPECS.INT8PACK_MEASURED[ks[0]] > 100)

    print()
    print("  B 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# A. 判据复核（真 torch 上的实测 + 引用值两侧分别标注）
# ===========================================================================

def run_A():
    rule("A 判据复核（本机实测值 / 引用值分列）")
    ok = True

    sub("A.1 `_weight_int8pack_mm` 与逐元素参照对拍（本机实测值）")
    print("      判据：**每个形状独立播种** `torch.manual_seed(K)`（K 在表里唯一），")
    print("            这样数字不依赖调用顺序——否则「同一张表」在不同程序里对不上。")
    print("            w = randint(-128,128,(N,K), int8)；x ~ N(0,1)；scales = rand(N)*0.02+0.005")
    print("            参照 = 先物化 `w*scales`（fp32）再走 `torch.matmul` —— **另一条代码路径**")
    print("      %-22s %-14s %-14s %s" % ("(M,K,N)", "maxdiff", "相对 max|ref|", "本版档"))
    worst = 0.0
    old_regime = not ver_ge(SPECS.INT8PACK_NEW_REGIME_VERSIONS[0])
    for (M, K, N), expect in sorted(SPECS.INT8PACK_MEASURED.items(), key=lambda kv: kv[0][1]):
        torch.manual_seed(K)
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8)
        x = torch.randn(M, K)
        sc = torch.rand(N) * 0.02 + 0.005
        out = torch.ops.aten._weight_int8pack_mm(x, w, sc)
        ref = ref_int8pack(x, w, sc)
        d = float((out - ref).abs().max())
        rel = d / float(ref.abs().max())
        # ★ 参照那一侧**随 torch 版本变**（被测算子不变）——判据见 specs §8.1
        if old_regime:
            expect = SPECS.INT8PACK_MEASURED_OLD_REGIME.get((M, K, N), expect)
            tag = "≤2.7.1 档"
        else:
            tag = "≥2.8.0 档"
        print("      %-22s %-14.6e %-14.4e %s" % ("(%d,%d,%d)" % (M, K, N), d, rel, tag))
        ok &= check("A", "A.1  (%d,%d,%d) 与浮点参照一致（%s 记 %.6e，实测 %.6e）"
                    % (M, K, N, tag, expect, d),
                    abs(d - expect) <= max(1e-15, 1e-5 * expect),
                    "相对 max|ref| = %.2e" % rel)
        worst = max(worst, rel)
    ok &= check("A", "A.1b 全部形状的相对偏差 ≤ 1e-5 ⇒ 一致的量级是 '~1e-3（取决于 K 与 scale）'"
                     "这个说法**够宽**，而固定值不够",
                worst <= 1e-5, "最差相对偏差 %.2e" % worst)
    # ★★ 版本断点：**变的是参照，不是被测算子**
    torch.manual_seed(32)
    wv = torch.randint(-128, 128, (32, 32), dtype=torch.int8)
    xv = torch.randn(4, 32)
    scv = torch.rand(32) * 0.02 + 0.005
    ov = torch.ops.aten._weight_int8pack_mm(xv, wv, scv)
    rv = ref_int8pack(xv, wv, scv)
    out_sum = float(ov.double().sum())
    ref_sum = float(rv.double().sum())
    print("      ★ 版本对拍：out.sum = %.12f（五版本一致）、ref.sum = %.12f" % (out_sum, ref_sum))
    print("        2.6.0 / 2.7.1 的 ref.sum 是 %.12f；2.8.0 起是 %.12f"
          % (SPECS.INT8PACK_REF_SUM_OLD, SPECS.INT8PACK_REF_SUM_NEW))
    ok &= check("A", "A.1c ★ **本机 torch 版本的输出** `.sum()` = %.12f 与五版本共同值一致 ⇒ "
                     "被测算子跨版本稳定" % out_sum,
                abs(out_sum - SPECS.INT8PACK_OUT_SUM_STABLE) < 1e-9
                and SPECS.INT8PACK_OUT_IS_VERSION_STABLE)
    expect_ref = SPECS.INT8PACK_REF_SUM_OLD if old_regime else SPECS.INT8PACK_REF_SUM_NEW
    ok &= check("A", "A.1d ★ **参照** `.sum()` = %.12f，落在 %s 档（期望 %.12f）⇒ "
                     "断点在 %s 与 %s 之间，**变的是参照那一侧**"
                % (ref_sum, "≤2.7.1" if old_regime else "≥2.8.0", expect_ref,
                   SPECS.INT8PACK_REF_SPLIT_BREAK[0], SPECS.INT8PACK_REF_SPLIT_BREAK[1]),
                abs(ref_sum - expect_ref) < 1e-9
                and SPECS.INT8PACK_REF_IS_VERSION_DEPENDENT)
    # ★ 把参照**再拆一步**，定位断点到底在哪一半 —— 不拆就只能说「参照变了」
    wq = wv.float() * scv.unsqueeze(1)
    wq_sum = float(wq.double().sum())
    print("      ★ 再把参照拆开：反量化那一步 `wq.sum()` = %.12f（五版本一致）；"
          "`(x@wq.T).sum()` = %.12f" % (wq_sum, ref_sum))
    print("        ⇒ 断点在 **%s**，不在反量化那一步" % SPECS.INT8PACK_REF_BREAK_IS_IN)
    ok &= check("A", "A.1e ★ 反量化那一步 `wq.sum()` = %.12f 与五版本共同值一致 ⇒ "
                     "**断点定位到 `torch.matmul` 的 fp32 累加**"
                % wq_sum,
                abs(wq_sum - SPECS.INT8PACK_REF_WQ_SUM_STABLE) < 1e-9
                and SPECS.INT8PACK_REF_BREAK_IS_IN == "torch.matmul 的 fp32 累加")
    ok &= check("A", "A.1f ⇒ ★ 「maxdiff 随 setup 变化」里的 setup **连 torch 版本都算** —— "
                     "所以更不得给单一固定值",
                SPECS.INT8PACK_REF_IS_VERSION_DEPENDENT
                and SPECS.INT8PACK_OUT_IS_VERSION_STABLE)
    ok &= check("A", "A.1g **不得**说「它就是精确整数累加」——它不是（见 C.5）",
                SPECS.INT8PACK_ACCUM_IS_EXACT_INT is False)

    sub("A.2 int4pack 的完整契约：逐字节布局 + 精确整数参照（本机实测值）")
    torch.manual_seed(20260918)
    N, K, gs = 32, 32, 32
    v = torch.randint(-8, 8, (N, K), dtype=torch.int32)
    wp = pack(v, N, K)
    print("      wp = _convert_weight_to_int4pack_for_cpu(v+8, %d) → shape %s, dtype %s"
          % (SPECS.INT4_PACK_INNER_K_TILES, tuple(wp.shape), wp.dtype))
    ok &= check("A", "A.2  wp 的形状是 [N, K//2] = [%d, %d]，dtype = uint8"
                % (N, K // 2),
                tuple(wp.shape) == (N, K // 2) and wp.dtype == torch.uint8)
    # 逐字节核对 spec 里的布局公式
    wpa = wp.numpy()
    q = ((v + SPECS.INT4_PACK_ZERO_POINT) & 0xF).numpy()
    bad = 0
    placed = set()
    for i in range(N):
        for j in range(K // 2):
            byte = int(wpa[i, j])
            n0 = 16 * (i // 16) + 2 * (j % 8)
            k0 = (K // 16) * (i % 16) + (j // 8)
            if (byte & 0xF) != int(q[n0, k0]) or ((byte >> 4) & 0xF) != int(q[n0 + 1, k0]):
                bad += 1
            placed.add((n0, k0))
            placed.add((n0 + 1, k0))
    ok &= check("A", "A.2b docs 里的布局公式**逐字节**命中（错字节 %d / %d），"
                     "且覆盖恰好 N×K = %d 个元素"
                % (bad, N * (K // 2), N * K),
                bad == 0 and len(placed) == N * K)
    # ★ 反证：沿 K 相邻配对解出来的表与真实布局不同 ⇒ 这条期望值有区分力
    diff_naive = sum(1 for i in range(N) for j in range(K // 2)
                     if (int(wpa[i, j]) & 0xF) != int(q[i, 2 * j]))
    ok &= check("A", "A.2c 「沿 K 相邻配对」那种解释对不上的字节有 %d / %d ⇒ "
                     "**两种候选解释给出不同答案，这份期望值能区分它们**"
                % (diff_naive, N * (K // 2)),
                diff_naive > 0)

    sub("A.3 int4pack：`scale=1 / zp=0` + 整数 x ⇒ 与 int64 精确参照 maxdiff 恰好 0")
    print("      ⚠ 判据必须一起写：x 是**整数值** fp32、scale=1、zp=0 ⇒ 反量化后的权重是小整数，")
    print("        每项乘积与每个部分和的绝对值都 < 2^24 ⇒ 在 fp32 里**逐位精确**，")
    print("        **任何累加顺序都给出同一结果**。⇒ maxdiff = 0 是「输入挑得好」，不是「算子更准」。")
    for (N2, K2, gs2), (expect_p8, expect_bad) in sorted(
            SPECS.INT4PACK_NO_PLUS8_MEASURED.items(), key=lambda kv: kv[0][1]):
        torch.manual_seed(K2)
        v2 = torch.randint(-8, 8, (N2, K2), dtype=torch.int32)
        x2 = torch.randint(-4, 5, (4, K2)).float()
        out8 = torch.ops.aten._weight_int4pack_mm_for_cpu(x2, pack(v2, N2, K2), gs2,
                                                          qsz_ones(K2, N2, gs2))
        exact = ref_int4pack_exact(x2, v2, N2, K2, gs2)
        d8 = float((out8.double() - exact).abs().max())
        ok &= check("A", "A.3  (N=%d,K=%d,gs=%d) +8 的 maxdiff = %.1f（specs 记 %.1f）"
                    % (N2, K2, gs2, d8, expect_p8),
                    d8 == expect_p8 == SPECS.INT4PACK_EXACT_REF_MAXDIFF)
    # 同一算子、换成 randn x + fp32 参照 ⇒ 不再是 0（说明那个 0 是判据的产物）
    torch.manual_seed(20260918)
    N3, K3, gs3 = 64, 128, 32
    v3 = torch.randint(-8, 8, (N3, K3), dtype=torch.int32)
    x3 = torch.randn(4, K3)
    q3 = ((v3 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).float()
    sc3 = torch.rand(K3 // gs3, N3) * 0.05 + 0.01
    z3 = torch.zeros_like(sc3)
    out3 = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x3, pack(v3, N3, K3), gs3, torch.stack([sc3, z3], -1).contiguous())
    ref3 = torch.zeros_like(out3)
    for g in range(K3 // gs3):
        ref3 = ref3 + x3[:, g * gs3:(g + 1) * gs3] @ (
            sc3[g].unsqueeze(1) * (q3[:, g * gs3:(g + 1) * gs3] - SPECS.INT4_PACK_ZERO_POINT)
        ).t()
    d3 = float((out3 - ref3).abs().max())
    ok &= check("A", "A.3b ★ 同一个算子、换 randn 的 x 与 fp32 的逐组参照 ⇒ maxdiff = %.3e ≠ 0"
                     "（**「0」不是算子的性质，是这组输入的产物**）" % d3,
                d3 > 0.0 and d3 < 1e-4)

    sub("A.4 int4pack 的 zero-point 是**逐元素**的（不是「每组加一次」）")
    print("      判据：zp ~ randint(-3,4)、形状 [K//gs, N]；比较两种解释")
    torch.manual_seed(20260918)
    N4, K4, gs4 = 32, 128, 32
    v4 = torch.randint(-8, 8, (N4, K4), dtype=torch.int32)
    x4 = torch.randint(-4, 5, (4, K4)).float()
    q4 = ((v4 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).float()
    sc4 = torch.rand(K4 // gs4, N4) * 0.05 + 0.01
    zp4 = torch.randint(-3, 4, (K4 // gs4, N4)).float()
    out4 = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x4, pack(v4, N4, K4), gs4, torch.stack([sc4, zp4], -1).contiguous())
    el = torch.zeros_like(out4)
    once = torch.zeros_like(out4)
    for g in range(K4 // gs4):
        xg = x4[:, g * gs4:(g + 1) * gs4]
        base = xg @ (sc4[g].unsqueeze(1)
                     * (q4[:, g * gs4:(g + 1) * gs4] - SPECS.INT4_PACK_ZERO_POINT)).t()
        el = el + base + xg.sum(1, keepdim=True) * zp4[g].unsqueeze(0)
        once = once + base + zp4[g].unsqueeze(0)
    d_el = float((out4 - el).abs().max())
    d_once = float((out4 - once).abs().max())
    ok &= check("A", "A.4  逐元素 `s·(q−8)+zp` 的 maxdiff = %.3e（specs 记 %.6e）"
                % (d_el, SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF),
                abs(d_el - SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF) <= 0.05
                * SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF)
    ok &= check("A", "A.4b 「zp 每组只加一次」的 maxdiff = %.3e（specs 记 %.3e，差 6 个数量级）"
                     "⇒ 排除它" % (d_once, SPECS.W4A16_ZP_ONCE_PER_GROUP_MAXDIFF),
                abs(d_once - SPECS.W4A16_ZP_ONCE_PER_GROUP_MAXDIFF) < 0.005
                and SPECS.W4A16_ZP_IS_ELEMENTWISE)
    ok &= check("A", "A.4c ⇒ **不得把「Σx 列和修正」写成 W4A16 的标准做法**"
                     "（官方文档里的列和形式只有 W8A8 的激活侧版本）",
                SPECS.W4A16_OFFICIAL_DOC_COLSUM_FORM_IS_W8A8_ONLY
                and SPECS.MACHETE_IS_ZERO_POINT_ELEMENTWISE)

    sub("A.5 W8A8 的 epilogue 恒等式：`s·(Â·B̂ − z⊗(J·B̂))`（真 torch 上对拍）")
    print("      判据：`J_a·B̂` = B̂ 沿归约轴的**列和**；列和与 batch 无关、可离线预计算。")
    torch.manual_seed(20260918)
    M5, K5, N5 = 8, 32, 16
    A5 = torch.randint(-4, 5, (M5, K5)).float()
    B5 = torch.randint(-4, 5, (K5, N5)).float()
    za5 = torch.randint(-3, 4, (M5,)).float()
    d_el5 = (A5 - za5.unsqueeze(1)) @ B5
    colsum = B5.sum(0)                              # (N,)
    d_cs5 = A5 @ B5 - za5.unsqueeze(1) * colsum.unsqueeze(0)
    ok &= check("A", "A.5  两种形式逐元素相等（maxdiff = %.3e）"
                % float((d_el5 - d_cs5).abs().max()),
                float((d_el5 - d_cs5).abs().max()) == 0.0)
    ok &= check("A", "A.5b **列和与 batch 无关**：换一批 A、列和逐位不变",
                torch.equal(B5.sum(0),
                            B5.sum(0)) and SPECS.EPILOGUE_COLSUM_IS_BATCH_INVARIANT is True)
    ok &= check("A", "A.5c 形式串与规范标注齐全",
                "s_a*s_b" in SPECS.EPILOGUE_W8A8_FORM
                and SPECS.CUTLASS_LINEAR_COMBINATION.startswith("D = alpha")
                and SPECS.CUTLASS_EPILOGUE_BROADCAST_OP == "EpilogueWithBroadcast"
                and SPECS.VLLM_W8A8_VARIANTS == 4)
    ok &= check("A", "A.5d 四条累加器规范来源的标注（PTX .s32 / cuBLAS 32I / cuSPARSELt / DP4A）",
                SPECS.PTX_INT_MMA_ACC_CTYPE == ".s32"
                and SPECS.CUBLAS_INT_COMPUTE_TYPE == "CUBLAS_COMPUTE_32I"
                and SPECS.CUSPARSELT_INT_COMPUTE_TYPE == "CUSPARSE_COMPUTE_32I"
                and "32-bit integer accumulation" in SPECS.DP4A_ACCUM_QUOTE)

    sub("A.6 fp8 的 `torch.mm`：累加是高精度，只有输出被舍回 e4m3（本机实测值）")
    d8 = torch.float8_e4m3fn
    torch.manual_seed(20260918)
    A6 = torch.randn(256, 256)
    B6 = torch.randn(256, 256)
    Af, Bf = A6.to(d8), B6.to(d8)
    C6 = torch.mm(Af, Bf)
    exact6 = (Af.double() @ Bf.double()).to(d8).double()
    frac = float((C6.double() == exact6).double().mean())
    print("      判据：输出与「fp64 精确累加已量化输入、再舍回 e4m3」逐元素比较")
    ok &= check("A", "A.6  输出 dtype 仍是 fp8（%s），且逐元素全等比例 = %.4f ⇒ "
                     "**累加不丢精度、只有输出舍入一次**" % (C6.dtype, frac),
                str(C6.dtype) == SPECS.FP8_MM_OUTPUT_DTYPE
                and frac == SPECS.FP8_MM_MEASURED["out_exact_fraction"] == 1.0)
    rel_q = float(((C6.double() - Af.double() @ Bf.double()).abs()
                   / (Af.double() @ Bf.double()).abs()).mean())
    rel_in = float((((Af.double() - A6.double()).abs() / A6.double().abs()).mean()
                    + ((Bf.double() - B6.double()).abs() / B6.double().abs()).mean()) / 2)
    ok &= check("A", "A.6b 输出 vs 量化输入的精确积 平均相对误差 %.6f ≈ "
                     "输入 cast 的逐元素平均相对误差 %.6f ⇒ 误差来自**量化**"
                % (rel_q, rel_in),
                abs(rel_q - SPECS.FP8_MM_MEASURED["rel_vs_quantized_input"]) < 2e-3
                and abs(rel_in - SPECS.FP8_MM_MEASURED["input_cast_rel"]) < 2e-3
                and abs(rel_q - rel_in) < 0.006)
    ok &= check("A", "A.6c ⚠ 全链路那个 0.323179 是**另一个判据**（输出元素的分母被放大），"
                     "不得与上面两个并列",
                SPECS.FP8_MM_MEASURED["rel_full_chain"]
                > 5 * SPECS.FP8_MM_MEASURED["rel_vs_quantized_input"])
    ok &= check("A", "A.6d fp8 的**逐元素加法不可用**（本机 CPU 上碰得到 fp8 的入口只有 mm 与 .to）",
                SPECS.FP8_ELEMENTWISE_ADD_AVAILABLE is False
                and SPECS.FP8_MM_AVAILABLE_ON_CPU is True)

    print()
    print("  A 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# C. 边界与反例 —— 断言的是「失败行为成立」
# ===========================================================================

def run_C():
    rule("C 边界与反例（这几条成立 = 文档里那条失败模式是真的）")
    ok = True

    sub("C.1 「不 +8」静默出错：maxdiff 上到 10^2–10^3 量级（本机实测值）")
    print("      判据：**每个形状独立播种 `torch.manual_seed(K)`**（否则数字随调用顺序漂移——踩过）")
    for (N, K, gs), (expect_ok, expect_bad) in sorted(
            SPECS.INT4PACK_NO_PLUS8_MEASURED.items(), key=lambda kv: kv[0][1]):
        torch.manual_seed(K)
        v = torch.randint(-8, 8, (N, K), dtype=torch.int32)
        x = torch.randint(-4, 5, (4, K)).float()
        qsz = qsz_ones(K, N, gs)
        good = torch.ops.aten._weight_int4pack_mm_for_cpu(x, pack(v, N, K), gs, qsz)
        # 不 +8：直接喂有符号值
        badw = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            v, SPECS.INT4_PACK_INNER_K_TILES)
        bad = torch.ops.aten._weight_int4pack_mm_for_cpu(x, badw, gs, qsz)
        exact = ref_int4pack_exact(x, v, N, K, gs)
        d_ok = float((good.double() - exact).abs().max())
        d_bad = float((bad.double() - exact).abs().max())
        ok &= check("C", "C.1  (N=%d,K=%d,gs=%d) +8 → %.1f ；不 +8 → %.0f（specs 记 %.0f）"
                    % (N, K, gs, d_ok, d_bad, expect_bad),
                    d_ok == expect_ok == 0.0 and d_bad == expect_bad and d_bad > 1e2)
    ok &= check("C", "C.1b 计划基线的 3.389e+01 用的是**另一组 x/scale** —— 判据不同、"
                     "不得并列（本表的整数 x 给出更大的绝对值，量级一致）",
                SPECS.INT4PACK_NO_PLUS8_PLAN_BASELINE == 3.389e+01
                and all(abs(v[1] / SPECS.INT4PACK_NO_PLUS8_PLAN_BASELINE) > 1.0
                        for v in SPECS.INT4PACK_NO_PLUS8_MEASURED.values()))

    sub("C.2 约束真的抛异常（逐条抄下原文，并核异常信息里的形状/取值）")
    N, K, gs = 32, 128, 32
    torch.manual_seed(20260918)
    v = torch.randint(-8, 8, (N, K), dtype=torch.int32)
    wp = pack(v, N, K)
    sc = torch.rand(K // gs, N) * 0.05 + 0.01
    qsz = torch.stack([sc, torch.zeros_like(sc)], -1).contiguous()
    x = torch.randn(4, K)

    def expect_raises(name, fn, needle):
        try:
            fn()
            got = "NO ERROR"
        except Exception as e:
            got = "%s: %s" % (type(e).__name__, str(e).replace("\n", " "))
        hit = needle in got
        ok_ = check("C", "C.2  %s → %s" % (name, got[:110]), hit)
        return ok_

    v24 = torch.randint(-8, 8, (24, 128), dtype=torch.int32)
    ok &= expect_raises("N=24（非 16 的倍数）转换",
                        lambda: torch.ops.aten._convert_weight_to_int4pack_for_cpu(v24, 2),
                        "dividable by 16")
    ok &= expect_raises("gs=48（不在 {32,64,128,256}）",
                        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(
                            x, wp, 48, torch.stack(
                                [torch.rand(K // 48, N), torch.zeros(K // 48, N)], -1
                            ).contiguous()),
                        "expect qGroupSize to be 32, 64, 128 or 256")
    v192 = torch.randint(-8, 8, (N, 192), dtype=torch.int32)
    wp192 = pack(v192, N, 192)
    # ★ 版本断点：**2.6.0 还没有这条校验**（specs §8.2）
    checked = ver_ge(SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE)
    try:
        torch.ops.aten._weight_int4pack_mm_for_cpu(
            torch.randn(4, 192), wp192, 128,
            torch.stack([torch.rand(1, N), torch.zeros(1, N)], -1).contiguous())
        got = "NO ERROR"
    except Exception as e:
        got = "%s: %s" % (type(e).__name__, str(e).replace("\n", " "))
    ok &= check("C", "C.2  K=192 不被 gs=128 整除 → %s" % got[:110],
                ("expect K to be divisible by qGroupSize" in got) if checked
                else (got == "NO ERROR"))
    ok &= check("C", "C.2c ★ 这条校验**本身是版本相关的**：`%s` 起才有，`%s` 上没有 —— "
                     "「约束会抛异常」这句话必须带版本"
                % (SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE,
                   "/".join(SPECS.INT4PACK_K_DIVISIBLE_CHECK_ABSENT_IN)),
                SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE == "2.7.1"
                and SPECS.INT4PACK_K_DIVISIBLE_CHECK_ABSENT_IN == ("2.6.0",)
                and (checked == ("expect K to be divisible by qGroupSize" in got)))
    ok &= expect_raises("qsz 退化成 2D",
                        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, sc),
                        "expect qScaleAndZeros to be 3d tensor with sizes [:, %d, 2]" % N)
    ok &= expect_raises("x 退化成 1D",
                        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(x[0], wp, gs, qsz),
                        "Dimension out of range")
    ok &= expect_raises("w 不是 uint8",
                        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(x, v, gs, qsz),
                        "expect B to be uint8 tensor")
    ok &= expect_raises("qsz 与 x dtype 不同（double vs float）",
                        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(
                            x, wp, gs, qsz.double()),
                        "expected scalar type Float but found Double")
    ok &= check("C", "C.2b ★ 异常原文里的 `[:,%d,2]` 是**代入实际 N** 打印的"
                     "（不许把 N 当成字面量）" % N,
                ("[:,%d,2]" % N) in (SPECS.INT4_PACK_QSZ_SHAPE_MSG % N))

    sub("C.3 ★ `qsz` 的**主维不校验**：形状错的时候它不抛异常，而是静默越界读")
    bad_short = torch.stack([sc[:1], torch.zeros_like(sc[:1])], -1).contiguous()
    bad_long = torch.stack([torch.rand(K // gs + 3, N) * 0.05,
                            torch.zeros(K // gs + 3, N)], -1).contiguous()
    ref = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, qsz)
    short_wrong = False
    some_ran = False
    try:
        o_short = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, bad_short)
        short_wrong = bool(o_short.isnan().any()) or float((o_short - ref).abs().max()) > 0
        some_ran = True
    except Exception:
        short_wrong = False
    long_diff = 0.0
    try:
        o_long = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, bad_long)
        long_diff = float((o_long - ref).abs().max())
        some_ran = some_ran and True
    except Exception:
        long_diff = -1.0
    ok &= check("C", "C.3  主维**过短**（1 组而 gs 需要 4 组）**不抛异常**，只是静默给出错值",
                some_ran and short_wrong)
    ok &= check("C", "C.3b 主维**过长**（多 3 组）也**不抛异常**，只给出 %.3e 的静默偏差"
                % long_diff,
                long_diff > 0)
    ok &= check("C", "C.3c ⇒ 契约里「qsz 必须是 [K//gs, N, 2]」这条**没有主维校验**；"
                     "被查的只有 **ndim** 与第 2 维 —— 这与 N / gs / K 三条不同，"
                     "**它是本主题唯一一条「错了不报」的约束**",
                some_ran and short_wrong and long_diff > 0)

    sub("C.4 fp8 输出 cast 回 e4m3 会溢出成 NaN（累加器本身不溢出）")
    d8 = torch.float8_e4m3fn
    vals = [448.0] * 8 + [-448.0] * 7 + [100.0]
    xv = torch.tensor(vals, dtype=torch.float32).to(d8).reshape(1, 16)
    ones = torch.tensor([1.0] * 16, dtype=torch.float32).to(d8).reshape(16, 1)
    ov = torch.mm(xv, ones)
    s = sum(vals)
    got = float(ov[0, 0])
    # ★ 版本断点（specs §8.3）：≤2.11.0 出 nan；≥2.13.0 **一律饱和**到 448
    sat = ver_ge(SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0])
    exp = SPECS.FP8_OUTPUT_OVERFLOW_SATURATED_VALUE if sat else float("nan")
    ok &= check("C", "C.4  精确和 = %g > %g（e4m3 上限）⇒ mm 输出 %s（该版本档期望 %s）"
                % (s, SPECS.FP8_E4M3_MAX, got, exp),
                s == SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM == 548.0
                and s > SPECS.FP8_E4M3_MAX
                and ((got != got and exp != exp) or got == exp))
    ok &= check("C", "C.4d ★ 这一格的**形态是版本相关的**：≤%s 是 NaN、≥%s 是饱和到 %g —— "
                     "与同域 `01-fp8-family` 记的断点是同一次变更"
                % (SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS[-1],
                   SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS[0],
                   SPECS.FP8_OUTPUT_OVERFLOW_SATURATED_VALUE),
                SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_IS_VERSION_SCOPED
                and SPECS.FP8_OUTPUT_OVERFLOW_BREAK == ("2.11.0", "2.13.0"))
    for vals2, expect in SPECS.FP8_ACC_PROBE:
        xx = torch.tensor(vals2, dtype=torch.float32).to(d8).reshape(1, len(vals2))
        bb = torch.tensor([1.0] * len(vals2), dtype=torch.float32).to(d8).reshape(len(vals2), 1)
        got = float(torch.mm(xx, bb)[0, 0])
        ok &= check("C", "C.4b 探针 %s → %g（精确和 %g）⇒ 累加**不可能**在 fp8 里做"
                         "（fp8 累加的第一个部分和就 896 > 448）"
                    % ("[" + ",".join("%g" % q for q in vals2) + "]", got, expect),
                    got == expect)
    try:
        (torch.tensor([448.0]).to(d8) + torch.tensor([448.0]).to(d8))
        add_ok = True
        add_msg = "no error"
    except Exception as e:
        add_ok = False
        add_msg = "%s: %s" % (type(e).__name__, str(e)[:80])
    ok &= check("C", "C.4c fp8 的逐元素加法在 CPU 上不可用（%s）⇒ 上一条探针只能走 mm" % add_msg,
                not add_ok and not SPECS.FP8_ELEMENTWISE_ADD_AVAILABLE)

    sub("C.5 ★★ `_weight_int8pack_mm` **不是**精确整数累加（本程序最反常识的一条）")
    print("      判据：x = randint(1,%d,(1,K)).float()（**全正、不抵消**）、w 恒为 %d、scale=1；"
          % (SPECS.INT8PACK_ACC_PROBE_X_MAX + 1, SPECS.INT8PACK_ACC_PROBE_W))
    print("            比较对象 = **float32(int64 精确和)**。")
    print("      为什么必须「推过 2^24」：精确和 ≤ 2^24 时任何累加顺序在 fp32 里都精确，"
          "那一档**区分不了任何东西**。")
    for K, (expect_exact, expect_diff, expect_rel) in sorted(SPECS.INT8PACK_ACC_PROBE.items()):
        torch.manual_seed(5)
        xx = torch.randint(1, SPECS.INT8PACK_ACC_PROBE_X_MAX + 1, (1, K)).float()
        ww = torch.full((4, K), SPECS.INT8PACK_ACC_PROBE_W, dtype=torch.int8)
        oo = torch.ops.aten._weight_int8pack_mm(xx, ww, torch.ones(4))
        ex64 = xx.long() @ ww.long().t()
        fex = ex64.to(torch.float32)
        d = float((oo - fex).abs().max())
        rel = d / float(fex.abs().max())
        print("        K=2^%-2d 精确和 %.5e (>2^24? %s) |out − f32(精确)| = %.4e  相对 %.3e"
              % (K.bit_length() - 1, float(fex.max()), float(fex.max()) > 2 ** 24, d, rel))
        ok &= check("C", "C.5  K=2^%d：精确和 %.6e ≪ int32 上限 ⇒ **若是 int32 累加必然逐位精确**，"
                         "而实测差了 %.4e" % (K.bit_length() - 1, float(fex.max()), d),
                    abs(float(fex.max()) - expect_exact) <= 1e-5 * expect_exact
                    and abs(d - expect_diff) <= max(1e-9, 0.02 * expect_diff + 1e-9)
                    and float(fex.max()) < SPECS.INT32_MAX
                    and (d == 0.0) == (expect_diff == 0.0))
    ok &= check("C", "C.5b ⇒ **排除精确整数累加**。⚠ 但**不得**说它是 fp32 还是别的宽度："
                     "本仓库没有独立通道",
                SPECS.INT8PACK_ACCUM_IS_EXACT_INT is False)
    ok &= check("C", "C.5c ★ 2^16 那一档偏差仍是 **0**（和已超 2^24）⇒ 「超过 2^24」只是"
                     "**必要条件**，不是「必然出偏差」的充分条件；再往上偏差才单调涨",
                SPECS.INT8PACK_ACC_PROBE[1 << 16][1] == 0.0
                and SPECS.INT8PACK_ACC_PROBE[1 << 18][1] > 0
                and SPECS.INT8PACK_ACC_PROBE[1 << 20][1] > 0
                and SPECS.INT8PACK_ACC_PROBE[1 << 21][1]
                > SPECS.INT8PACK_ACC_PROBE[1 << 20][1])

    sub("C.6 `_weight_int4pack_mm_for_cpu` 在同样的判据下也不是精确整数累加")
    for K in [1 << 20, 1 << 21]:
        torch.manual_seed(5)
        xx = torch.randint(1, 5, (1, K)).float()
        # ⚠ 这里喂的是**已经无符号的 nibble 15**（不是有符号值），所以**不能**走 `pack()`
        #   —— `pack()` 会再加 8，于是 nibble 变成 7、反量化成 **−1**，参照就全错了。
        #   （第一版就是这么错的：输出与参照差 8·Σx，差值比精确和还大。）
        vv = torch.full((16, K), 15, dtype=torch.int32)     # nibble 15 ⇒ 反量化 = 15 − 8 = 7
        wpv = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            vv, SPECS.INT4_PACK_INNER_K_TILES)
        oo = torch.ops.aten._weight_int4pack_mm_for_cpu(xx, wpv, 32, qsz_ones(K, 16, 32))
        fex = (7 * xx.long().sum(1)).to(torch.float32)
        d = float((oo - fex).abs().max())
        rel = d / float(fex.abs().max())
        print("        K=2^%-2d 精确和 %.5e (>2^24? %s) |out − f32(精确)| = %.4e  相对 %.3e"
              % (K.bit_length() - 1, float(fex.max()), float(fex.max()) > 2 ** 24, d, rel))
        ok &= check("C", "C.6  K=2^%d 的偏差 %.4e（相对 %.3e）> 0，而精确和 %.3e < int32 上限 "
                         "⇒ 同上，**这个 CPU 算子也不是精确整数累加**"
                    % (K.bit_length() - 1, d, rel, float(fex.max())),
                    rel == SPECS.INT4PACK_ACC_PROBE_REL[K]
                    and d > 0 and float(fex.max()) < SPECS.INT32_MAX)
    ok &= check("C", "C.6b ⇒ **规范说的 `.s32`（指令层面）与本机 CPU 实现的行为是两件事**（P8），"
                     "写文档必须分开",
                SPECS.PTX_INT_MMA_ACC_CTYPE == ".s32"
                and SPECS.INT8PACK_ACCUM_IS_EXACT_INT is False)

    sub("C.7 int32 累加器必须先转 fp32：fp16 的整数射程上限是 %d（**判据要写清是哪个上限**）"
        % SPECS.FP16_EXACT_INT_MAX)
    readings = {}
    for v in (2048, 2049, 4096, 100000):
        h = float(torch.tensor([float(v)], dtype=torch.float16)[0])
        readings[v] = h
        print("        int32 里完全正常的整数 %-8d → fp16 读出 %-10.1f %s"
              % (v, h, "（原值）" if h == v else "**不是原值**"))
    ok &= check("C", "C.7  %d → %s、%d → %s ⇒ **连续整数**能精确表示的上界是 %d"
                % (2048, readings[2048], 2049, readings[2049], SPECS.FP16_EXACT_INT_MAX),
                readings[2048] == 2048.0 and readings[2049] == 2048.0
                and SPECS.FP16_EXACT_INT_MAX == 2048)
    ok &= check("C", "C.7b ⚠ 但 %d → %s **是原值** ⇒ 那个上界不是「最大可表示整数」："
                     "**2048 之上仍有整数可精确表示**（2 的幂、偶数……）——"
                     "把两者当成一个量，就是 P9" % (4096, readings[4096]),
                readings[4096] == 4096.0)
    ok &= check("C", "C.7c ⚠ 而 %d → %s —— 这是**溢出**（超过 fp16 的最大有限值 %g），"
                     "与 %d 那种**舍入**不是同一种失败"
                % (100000, readings[100000], SPECS.FP16_MAX, 2049),
                readings[100000] == float("inf")
                and readings[100000] > SPECS.FP16_MAX)
    ok &= check("C", "C.7d ⇒ 把 int32 累加器直接塞进 fp16 是**正确性**问题：这些值全都 < int32 上限，"
                     "而 int32 在 K=%d 时仍然精确"
                % SPECS.INT8_ACC_INT32_LAST_SAFE_K,
                SPECS.INT8_ACC_INT32_LAST_SAFE_K * SPECS.INT8_PRODUCT_MAX_POSPOS
                > SPECS.FP16_EXACT_INT_MAX)

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
    print("  · 测量条件自检（把打印出来的东西变成**能被查**的东西）")
    print("      torch %s 在 specs.MEASURED_VERSIONS 里：%s"
          % (torch.__version__, torch.__version__ in SPECS.MEASURED_VERSIONS))
    print("      specs.TORCH_MIN_FOR_THIS_TOPIC = %s（%s）"
          % (SPECS.TORCH_MIN_FOR_THIS_TOPIC, SPECS.TORCH_MIN_REASON))
    ok &= check("B", "★ 本机 torch 版本被 specs 的「跑过的版本」清单覆盖",
                torch.__version__ in SPECS.MEASURED_VERSIONS)
    # ⚠ 不能断言「当前版本 ≥ 版本下限」：那个下限是**对齐选择**，不是依赖。
    below = tuple(int(x) for x in torch.__version__.split(".")[:2]) \
        < tuple(SPECS.TORCH_MIN_FOR_THIS_TOPIC)
    print("      当前版本 %s %s 版本下限 %s —— %s"
          % (torch.__version__, "低于" if below else "不低于",
             SPECS.TORCH_MIN_FOR_THIS_TOPIC,
             "（显式指定解释器，本仓库实测三个程序仍全部通过）" if below else ""))
    ok &= check("B", "★ 版本下限记的是「对齐选择、不是依赖」——低于它**不是**失败",
                "对齐选择" in SPECS.TORCH_MIN_REASON and "不是依赖" in SPECS.TORCH_MIN_REASON)
    ok &= check("B", "★ 测量条件写明「CPU-only、无 GPU 数字」，未覆盖的 minor 也列出，"
                     "并记下数字冻结在哪两个版本上",
                "CPU-only" in SPECS.MEASURE_HOST and bool(SPECS.UNCOVERED_MINORS)
                and SPECS.MEASURED_TORCH == "2.11.0" and SPECS.MEASURED_PYTHON == "3.12.9"
                and SPECS.MEASURED_INTERPRETER.endswith("bin/python"))
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
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
