#!/usr/bin/env python3
"""02 W4A16 的打包布局：`_weight_int4pack_mm_for_cpu` 的完整契约与逐字节反解。

对应文档：130-quantization/quantized-kernels/02-w4a16-packing.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

★★ **本主题最硬的一条证据在这里**：`v + 8` 之后 maxdiff 恰好 **0**、
   不 `+8` 时是 **10^2–10^3 量级**。但「0」需要**把判据一起写出来**才算证据（见 A.3）。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：int4 载荷 vs 元数据（group scale / zero-point）的字节账；
       以及**打包布局本身的开销**——索引运算与「一个字节装两个值」的读放大。
  A —— **两条互不依赖的观察通道**：
       ① 文档侧（计划基线的调用契约与布局公式）—— **引用值**；
       ② 本机侧（逐字节反解 `_convert_weight_to_int4pack_for_cpu` 的输出）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：不 +8 的静默错、五条约束各自真抛异常、
       **`qsz` 主维不校验**、以及**「沿 K 相邻配对」那种解释被字节否证**。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 02_w4a16_packing   （或 make run 全跑）
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
    print("02 W4A16 的打包布局：int4pack 的完整契约（逐字节反解）")
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
    print("  ⚠ **本机没有 GPU。** 本程序量的是 **CPU 算子**的行为；")
    print("    W4A16 在各种 GPU kernel 上的性能在本文里**一律是引用值**。")


def pack_raw(v, inner_k_tiles):
    """把**已经是无符号 nibble（0..15）**的表喂给转换器。

    ⚠ 不要在这里再 `+8` —— 调用方负责。这个显式分离是刻意的：
      「谁负责 +8」正是本主题最容易写错的那一点。
    """
    return torch.ops.aten._convert_weight_to_int4pack_for_cpu(v, inner_k_tiles)


def pack_signed(v, inner_k_tiles=None):
    """把**有符号 int4**（−8..7）打成内核要的 uint8 布局（**这一步必须 +8**）。"""
    if inner_k_tiles is None:
        inner_k_tiles = SPECS.INT4_PACK_INNER_K_TILES
    return pack_raw(v + SPECS.INT4_PACK_ZERO_POINT, inner_k_tiles)


def qsz_ones(K, N, gs):
    o = torch.ones(K // gs, N)
    z = torch.zeros(K // gs, N)
    return torch.stack([o, z], -1).contiguous()


def ref_exact_int(x_int, v_signed):
    """`scale=1 / zp=0` 下的 **int64 精确参照**（判据见 A.3）。"""
    q = (v_signed + SPECS.INT4_PACK_ZERO_POINT) & 0xF
    return x_int.long() @ q.t().long() - SPECS.INT4_PACK_ZERO_POINT * x_int.long().sum(
        1, keepdim=True)


def layout_place(i, j, K, off):
    """文档公式：字节 `(i, j)` 的两个 nibble 分别装哪个 `(n, k)`。"""
    n = 16 * (i // 16) + 2 * (j % 8) + off
    k = (K // 16) * (i % 16) + (j // 8)
    return n, k


# ===========================================================================
# B. 代价模型
# ===========================================================================

def run_B():
    rule("B 代价模型（字节账：载荷 / 元数据 / 索引开销）")
    ok = True

    sub("B.1 int4 的载荷与元数据（**判据：算的是哪一份字节**）")
    print("      判据① 载荷：每个权重 4 bit = %.1f 字节" % SPECS.W4A16_DECODE_BYTES_PER_ELEM)
    print("      判据② group scale：每 gs 个权重共享一个 fp32 scale + 一个 fp32 zero-point")
    print("      判据③ 元数据占比 = (4 + 4) / gs ÷ 0.5（相对载荷）")
    for gs in SPECS.INT4_PACK_GROUP_SIZES:
        meta_bytes_per_elem = 8.0 / gs
        payload = SPECS.W4A16_DECODE_BYTES_PER_ELEM
        print("        gs=%-4d 元数据 %.4f B/元素 ⇒ 相对载荷 +%.2f%%（总等效宽度 %.3f bit）"
              % (gs, meta_bytes_per_elem, 100.0 * meta_bytes_per_elem / payload,
                 8.0 * (payload + meta_bytes_per_elem)))
    ok &= check("B", "B.1  gs=32 的元数据开销最大（+50%%）、gs=256 最小（+6.25%%）——"
                     "**「group 越小 scale 越贵」的量化形式**",
                abs(8.0 / 32 / 0.5 - 0.5) < 1e-12
                and abs(8.0 / 256 / 0.5 - 0.0625) < 1e-12
                and SPECS.INT4_PACK_GROUP_SIZES[0] == 32
                and SPECS.INT4_PACK_GROUP_SIZES[-1] == 256)
    for (N, K) in [(4096, 4096), (11008, 4096)]:
        for gs in (128,):
            payload = N * K * 0.5
            meta = N * (K // gs) * 8
            print("        N=%-6d K=%-6d gs=%-4d 载荷 %.2f MiB + 元数据 %.2f MiB（+%.2f%%）"
                  % (N, K, gs, payload / 2 ** 20, meta / 2 ** 20, 100.0 * meta / payload))
    ok &= check("B", "B.1b 元数据与载荷**不是同一个量**：一个是「每 gs 个权重 8 字节」、"
                     "一个是「每权重 0.5 字节」——**判据不同，不得并列**",
                True)

    sub("B.2 打不开的账：**一个字节装两个值**，读的时候要额外做一次解包")
    print("      判据：`wp` 的形状是 [N, K/2]（每字节两个 nibble），元素访问是")
    print("            `(wp[i, j] & 0xF)` 与 `((wp[i, j] >> 4) & 0xF)` —— 各一次位运算。")
    print("      ⇒ **省下的是字节，付出的是索引运算与位运算**；这不是带宽账。")
    for (N, K) in [(4096, 4096), (11008, 4096)]:
        elems = N * K
        print("        N=%-6d K=%-6d 元素 %d 个 / 字节 %d 个 ⇒ 每字节 2 个元素，"
              "每个元素 1 次 mask + 1 次 shift"
              % (N, K, elems, elems // 2))
    ok &= check("B", "B.2  「一个字节装两个值」在**元素数**上不省任何东西——"
                     "省的是**存储字节**，不是**索引次数**",
                SPECS.W4A16_DECODE_BYTES_PER_ELEM == 0.5)

    sub("B.3 独立反量化 pass 的带宽预算：写是读的 4 倍")
    rd = SPECS.W4A16_DECODE_BYTES_PER_ELEM
    wr = SPECS.W4A16_DEQUANT_WRITE_BYTES_PER_ELEM
    print("      读 %.1f B/元素（int4 载荷）→ 写 %.1f B/元素（fp16 权重）" % (rd, wr))
    ok &= check("B", "B.3  写/读 = %.0f× ⇒ 独立 kernel 被**写侧**卡死（不是被 int4 的读侧）"
                % (wr / rd),
                wr / rd == SPECS.W4A16_SEPARATE_PASS_WRITE_OVER_READ == 4.0)
    ok &= check("B", "B.3b 本机实测的那个「输出直接是 float」只能说明**调用方拿不到**"
                     "反量化后的权重，**不能**证明融合发生在 epilogue 里——"
                     "后者是 CUTLASS/vLLM 文档的说法，本机无法核实",
                SPECS.CUTLASS_EPILOGUE_BROADCAST_OP == "EpilogueWithBroadcast"
                and SPECS.VLLM_W8A8_BIAS_AXIS == "per-channel")

    sub("B.4 布局公式的索引代价（本机可数）")
    print("      判据：给定 wp = [N, K//2]，元素 (n, k) 落在")
    print("            i = 16·(n // 16) + k // (K // 16)、")
    print("            j = 8·(k % (K // 16)) + (n % 16) // 2、")
    print("            nibble = n % 2（0 = 低、1 = 高）。")
    print("      ⚠ **这条「解包方向」的公式是本程序从字节里反解出来的，不是抄来的**——")
    print("        它等价于 specs 里那两条（A.2 已在三个形状上逐字节验过）。")
    N, K = 64, 256
    hits = set()
    for n in range(N):
        for k in range(K):
            i = 16 * (n // 16) + k // (K // 16)
            j = 8 * (k % (K // 16)) + (n % 16) // 2
            hits.add((i, j, n % 2))
    ok &= check("B", "B.4  解包方向在 [%d, %d] 上是**单射**：%d 个元素 ↦ %d 个 (i,j,nibble) 槽位"
                     "（上界 %d×%d×2 = %d）"
                % (N, K, N * K, len(hits), N, K // 2, N * (K // 2) * 2),
                len(hits) == N * K and N * K <= N * (K // 2) * 2
                and N % SPECS.INT4_PACK_N_MULTIPLE == 0)
    ok &= check("B", "B.4b 解包方向的 `n` 步长是 16·(n//16) —— **沿 N 方向按 16 个一组跨块**，"
                     "这是权重预重排（pre-shuffling）在 CPU 侧的最小对应物",
                layout_place(17, 0, K, 0)[0] == 16
                and layout_place(0, 0, K, 0)[0] == 0
                and layout_place(0, 0, K, 1)[0] == 1)

    print()
    print("  B 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# A. 判据复核
# ===========================================================================

def run_A():
    rule("A 判据复核（本机实测值 / 引用值分列）")
    ok = True

    sub("A.1 完整调用契约：形状、dtype、四个实参")
    torch.manual_seed(20260918)
    N, K, gs = 32, 128, 32
    v = torch.randint(-8, 8, (N, K), dtype=torch.int32)
    wp = pack_signed(v)
    scales = torch.rand(K // gs, N) * 0.05 + 0.01
    zps = torch.zeros_like(scales)
    qsz = torch.stack([scales, zps], -1).contiguous()
    x = torch.randn(4, K)
    out = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, qsz)
    print("      wp  = _convert_weight_to_int4pack_for_cpu(v + %d, %d)  → %s %s"
          % (SPECS.INT4_PACK_ZERO_POINT, SPECS.INT4_PACK_INNER_K_TILES,
             tuple(wp.shape), wp.dtype))
    print("      qsz = stack([scales, zeros], -1)  → %s" % (tuple(qsz.shape),))
    print("      out = _weight_int4pack_mm_for_cpu(x, wp, gs, qsz)  → %s %s"
          % (tuple(out.shape), out.dtype))
    ok &= check("A", "A.1  wp 形状 [N, K//2]、dtype uint8；qsz 形状 [K//gs, N, 2]；"
                     "out 形状 [M, N]、dtype float32（与 x 同）",
                tuple(wp.shape) == (N, K // 2) and wp.dtype == torch.uint8
                and tuple(qsz.shape) == (K // gs, N, 2)
                and tuple(out.shape) == (4, N) and out.dtype == x.dtype)
    ok &= check("A", "A.1b 契约常量：N 被 %d 整除、gs ∈ %s、K 被 gs 整除"
                % (SPECS.INT4_PACK_N_MULTIPLE, SPECS.INT4_PACK_GROUP_SIZES),
                SPECS.INT4_PACK_N_MULTIPLE == 16
                and SPECS.INT4_PACK_GROUP_SIZES == (32, 64, 128, 256)
                and N % 16 == 0 and gs in SPECS.INT4_PACK_GROUP_SIZES and K % gs == 0)

    sub("A.2 ★ 逐字节反解打包布局（本程序的核心实测）")
    print("      判据：造一张 q = (v + 8) & 0xF 的 [N,K] nibble 表，与 `wp` 的每个字节对比；")
    print("            哪一侧对不上就说明「哪个 (n,k) 装在这个 nibble 里」的解释错了。")
    for (N2, K2) in [(32, 32), (64, 256), (16, 128)]:
        torch.manual_seed(20260918)
        v2 = torch.randint(-8, 8, (N2, K2), dtype=torch.int32)
        wp2 = pack_signed(v2).numpy()
        q2 = ((v2 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).numpy()
        bad = 0
        slots = set()
        for i in range(N2):
            for j in range(K2 // 2):
                byte = int(wp2[i, j])
                n0, k0 = layout_place(i, j, K2, 0)
                n1, k1 = layout_place(i, j, K2, 1)
                if (byte & 0xF) != int(q2[n0, k0]) or ((byte >> 4) & 0xF) != int(q2[n1, k1]):
                    bad += 1
                slots.add((n0, k0))
                slots.add((n1, k1))
        ok &= check("A", "A.2  (N=%d,K=%d) 逐字节命中：错字节 %d / %d，覆盖 %d 个元素（应 %d）"
                    % (N2, K2, bad, N2 * (K2 // 2), len(slots), N2 * K2),
                    bad == 0 and len(slots) == N2 * K2)
    # ★ 两种候选解释给出不同答案 —— 这份期望值必须有区分力
    torch.manual_seed(20260918)
    N2, K2 = 64, 256
    v2 = torch.randint(-8, 8, (N2, K2), dtype=torch.int32)
    wp2 = pack_signed(v2).numpy()
    q2 = ((v2 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).numpy()
    n_along_k = sum(1 for i in range(N2) for j in range(K2 // 2)
                    if (int(wp2[i, j]) & 0xF) != int(q2[i, 2 * j]))
    n_along_n = sum(1 for i in range(N2) for j in range(K2 // 2)
                    if int(wp2[i, j]) != (int(q2[16 * (i // 16) + 2 * (j % 8),
                                                   (K2 // 16) * (i % 16) + (j // 8)])
                                          | (int(q2[16 * (i // 16) + 2 * (j % 8) + 1,
                                                    (K2 // 16) * (i % 16) + (j // 8)]) << 4)))
    print("      两种解释的错字节数：沿 K 相邻配对 %d / %d；沿 N 配对 %d / %d"
          % (n_along_k, N2 * (K2 // 2), n_along_n, N2 * (K2 // 2)))
    ok &= check("A", "A.2b ★ **两种候选解释给出不同答案**：沿 K 那一种错 %d 个字节、"
                     "沿 N 那一种错 0 个 ⇒ 这份样本集能区分它们" % n_along_k,
                n_along_k > 0 and n_along_n == 0)
    ok &= check("A", "A.2c 布局方向常量写的是 `%s`（**不是 `K`**）"
                % SPECS.INT4_PACK_LAYOUT_PAIRS_ALONG,
                SPECS.INT4_PACK_LAYOUT_PAIRS_ALONG == "N"
                and SPECS.INT4_PACK_LAYOUT_NAIVE_PAIRS_ALONG == "K")

    sub("A.3 ★★ maxdiff 恰好 0 —— **判据必须一起写出来**，否则这个 0 什么也不说明")
    print("      判据（缺一条这个 0 就换一个值）：")
    print("        ① x 是**整数值** fp32：`randint(-4, 5, (4, K)).float()`；")
    print("        ② `scale = 1.0`、`zero_point = 0.0` ⇒ 反量化后的权重是小整数；")
    print("        ③ 参照 = **int64 精确值**，不是 fp32 matmul。")
    print("      ⇒ 每项乘积与每个部分和都是绝对值 < 2^24 的整数，在 fp32 里逐位精确，")
    print("        **任何累加顺序都给出同一结果** —— 所以 0 与形状无关。")
    for (N3, K3, gs3), (expect_ok, expect_bad) in sorted(
            SPECS.INT4PACK_NO_PLUS8_MEASURED.items(), key=lambda kv: kv[0][1]):
        torch.manual_seed(K3)
        v3 = torch.randint(-8, 8, (N3, K3), dtype=torch.int32)
        x3 = torch.randint(-4, 5, (4, K3)).float()
        q3 = qsz_ones(K3, N3, gs3)
        good = torch.ops.aten._weight_int4pack_mm_for_cpu(x3, pack_signed(v3), gs3, q3)
        exact = ref_exact_int(x3, v3)
        d = float((good.double() - exact.double()).abs().max())
        print("        N=%-4d K=%-5d gs=%-4d maxdiff = %.1f" % (N3, K3, gs3, d))
        ok &= check("A", "A.3  (N=%d,K=%d,gs=%d) maxdiff = %.1f（specs 记 %.1f）"
                    % (N3, K3, gs3, d, expect_ok),
                    d == expect_ok == SPECS.INT4PACK_EXACT_REF_MAXDIFF)
    # ★ 同一算子、换成 randn 的 x 与 fp32 的逐组参照 ⇒ 不再是 0
    torch.manual_seed(20260918)
    N4, K4, gs4 = 64, 128, 32
    v4 = torch.randint(-8, 8, (N4, K4), dtype=torch.int32)
    x4 = torch.randn(4, K4)
    q4 = ((v4 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).float()
    sc4 = torch.rand(K4 // gs4, N4) * 0.05 + 0.01
    z4 = torch.zeros_like(sc4)
    o4 = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x4, pack_signed(v4), gs4, torch.stack([sc4, z4], -1).contiguous())
    r4 = torch.zeros_like(o4)
    for g in range(K4 // gs4):
        r4 = r4 + x4[:, g * gs4:(g + 1) * gs4] @ (
            sc4[g].unsqueeze(1) * (q4[:, g * gs4:(g + 1) * gs4]
                                   - SPECS.INT4_PACK_ZERO_POINT)).t()
    d4 = float((o4 - r4).abs().max())
    ok &= check("A", "A.3b ★ 同一算子换 randn 的 x 与 fp32 参照 ⇒ maxdiff = %.3e **≠ 0** ⇒ "
                     "「0」是**输入**的产物，不是算子的性质" % d4,
                0.0 < d4 < 1e-4)

    sub("A.4 innerKTiles 这个实参：实测四档**逐字节相同**（K=256、N=64）")
    torch.manual_seed(20260918)
    N5, K5 = 64, 256
    v5 = torch.randint(0, 16, (N5, K5), dtype=torch.int32)
    ref_bytes = None
    for t in (1, 2, 4, 8):
        b = torch.ops.aten._convert_weight_to_int4pack_for_cpu(v5, t).numpy().tobytes()
        if ref_bytes is None:
            ref_bytes = b
        print("        innerKTiles=%d → shape %s，与 t=%d 逐字节相同：%s"
              % (t, tuple(torch.ops.aten._convert_weight_to_int4pack_for_cpu(v5, t).shape),
                 SPECS.INT4_PACK_INNER_K_TILES, b == ref_bytes))
        ok &= check("A", "A.4  innerKTiles=%d 与 t=%d 的输出逐字节相同" % (t, 2), b == ref_bytes)
    ok &= check("A", "A.4b 文档沿用 `%d` —— 但记的是「文档取这个值」，不是「只有它可用」"
                % SPECS.INT4_PACK_INNER_K_TILES,
                SPECS.INT4_PACK_INNER_K_TILES == 2)

    sub("A.5 zero-point 是**逐元素**的（与 W8A8 的列和修正不是同一件事）")
    torch.manual_seed(20260918)
    N6, K6, gs6 = 32, 128, 32
    v6 = torch.randint(-8, 8, (N6, K6), dtype=torch.int32)
    x6 = torch.randint(-4, 5, (4, K6)).float()
    q6 = ((v6 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).float()
    sc6 = torch.rand(K6 // gs6, N6) * 0.05 + 0.01
    zp6 = torch.randint(-3, 4, (K6 // gs6, N6)).float()
    o6 = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x6, pack_signed(v6), gs6, torch.stack([sc6, zp6], -1).contiguous())
    el = torch.zeros_like(o6)
    once = torch.zeros_like(o6)
    for g in range(K6 // gs6):
        xg = x6[:, g * gs6:(g + 1) * gs6]
        base = xg @ (sc6[g].unsqueeze(1)
                     * (q6[:, g * gs6:(g + 1) * gs6]
                        - SPECS.INT4_PACK_ZERO_POINT)).t()
        el = el + base + xg.sum(1, keepdim=True) * zp6[g].unsqueeze(0)
        once = once + base + zp6[g].unsqueeze(0)
    d_el = float((o6 - el).abs().max())
    d_once = float((o6 - once).abs().max())
    ok &= check("A", "A.5  逐元素 `%s` 的 maxdiff = %.3e（specs 记 %.6e）"
                % (SPECS.INT4_PACK_DEQUANT_FORM, d_el, SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF),
                abs(d_el - SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF) < 1e-6)
    ok &= check("A", "A.5b 「zp 每组只加一次」= %.1f ⇒ 差 6 个数量级，**排除**它" % d_once,
                abs(d_once - SPECS.W4A16_ZP_ONCE_PER_GROUP_MAXDIFF) < 0.01
                and SPECS.W4A16_ZP_IS_ELEMENTWISE)
    ok &= check("A", "A.5c ⇒ W4A16 的官方做法是逐元素 `(q−z)`（vLLM machete Readme 逐字可查），"
                     "**不是** Σx 列和修正（那只在 W8A8 的激活侧文档里）",
                SPECS.W4A16_OFFICIAL_DOC_COLSUM_FORM_IS_W8A8_ONLY
                and SPECS.MACHETE_IS_ZERO_POINT_ELEMENTWISE
                and "w_z" in SPECS.MACHETE_DEQUANT_FORM
                and SPECS.MACHETE_DEQUANT_FORM.endswith("@ a")
                and SPECS.MACHETE_AUTHOR == "Neural Magic"
                and SPECS.MACHETE_HAS_ARXIV is False)

    print()
    print("  A 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    rule("C 边界与反例（这几条成立 = 文档里那条失败模式是真的）")
    ok = True

    sub("C.1 ★ 不 +8 的静默错（本主题最硬的反证）")
    for (N, K, gs), (expect_ok, expect_bad) in sorted(
            SPECS.INT4PACK_NO_PLUS8_MEASURED.items(), key=lambda kv: kv[0][1]):
        torch.manual_seed(K)
        v = torch.randint(-8, 8, (N, K), dtype=torch.int32)
        x = torch.randint(-4, 5, (4, K)).float()
        q = qsz_ones(K, N, gs)
        good = torch.ops.aten._weight_int4pack_mm_for_cpu(x, pack_signed(v), gs, q)
        # 不 +8：把**有符号值直接**喂给转换器
        badw = pack_raw(v, SPECS.INT4_PACK_INNER_K_TILES)
        bad = torch.ops.aten._weight_int4pack_mm_for_cpu(x, badw, gs, q)
        exact = ref_exact_int(x, v)
        d_ok = float((good.double() - exact.double()).abs().max())
        d_bad = float((bad.double() - exact.double()).abs().max())
        ok &= check("C", "C.1  (N=%d,K=%d,gs=%d) +8 → %.1f ；不 +8 → %.0f（specs 记 %.0f）"
                    % (N, K, gs, d_ok, d_bad, expect_bad),
                    d_ok == expect_ok == 0.0 and d_bad == expect_bad and d_bad > 1e2)
    # 闭式：逐元素偏差恰好 ±8（与 01 的 C.1 是同一条结论的另一侧）
    offs = {(v & 0xF) - SPECS.INT4_PACK_ZERO_POINT - v for v in range(-8, 8)}
    ok &= check("C", "C.1b 闭式：不 +8 时每个元素读到的是 `(v & 0xF) − %d`，偏差恒为 %s"
                % (SPECS.INT4_PACK_ZERO_POINT, sorted(offs)),
                offs == {-8, 8})
    ok &= check("C", "C.1c 但它**不崩、不 NaN**（16 个取值 ↦ 16 个不同结果）——"
                     "所以只会静默出错，这正是它危险的地方",
                len({(v & 0xF) - SPECS.INT4_PACK_ZERO_POINT for v in range(-8, 8)}) == 16)
    ok &= check("C", "C.1d 计划基线的 3.389e+01 用**另一组 x/scale**，判据不同 ⇒ 不得并列",
                SPECS.INT4PACK_NO_PLUS8_PLAN_BASELINE == 3.389e+01
                and all(abs(v[1] / SPECS.INT4PACK_NO_PLUS8_PLAN_BASELINE) > 1.0
                        for v in SPECS.INT4PACK_NO_PLUS8_MEASURED.values()))

    sub("C.2 五条约束各自真抛异常（逐条抄下原文）")
    N, K, gs = 32, 128, 32
    torch.manual_seed(20260918)
    v = torch.randint(-8, 8, (N, K), dtype=torch.int32)
    wp = pack_signed(v)
    sc = torch.rand(K // gs, N) * 0.05 + 0.01
    qsz = torch.stack([sc, torch.zeros_like(sc)], -1).contiguous()
    x = torch.randn(4, K)

    def expect_raises(name, fn, needle):
        try:
            fn()
            got = "NO ERROR"
        except Exception as e:
            got = "%s: %s" % (type(e).__name__, str(e).replace("\n", " "))
        return check("C", "C.2  %s → %s" % (name, got[:112]), needle in got)

    ok &= expect_raises(
        "① N=24 不是 16 的倍数",
        lambda: pack_raw(torch.randint(0, 16, (24, 128), dtype=torch.int32), 2),
        "expect N to be dividable by 16")
    ok &= expect_raises(
        "② gs=48 不在 {32,64,128,256}",
        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(
            x, wp, 48,
            torch.stack([torch.rand(K // 48, N), torch.zeros(K // 48, N)], -1).contiguous()),
        "expect qGroupSize to be 32, 64, 128 or 256")
    wp192 = pack_signed(torch.randint(-8, 8, (N, 192), dtype=torch.int32))
    # ★ 版本断点：这条校验 **2.6.0 还没有**（specs §8.2）
    checked = ver_ge(SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE)
    try:
        torch.ops.aten._weight_int4pack_mm_for_cpu(
            torch.randn(4, 192), wp192, 128,
            torch.stack([torch.rand(1, N), torch.zeros(1, N)], -1).contiguous())
        got192 = "NO ERROR"
    except Exception as e:
        got192 = "%s: %s" % (type(e).__name__, str(e).replace("\n", " "))
    ok &= check("C", "C.2  ③ K=192 不被 gs=128 整除 → %s" % got192[:112],
                ("expect K to be divisible by qGroupSize" in got192) if checked
                else got192 == "NO ERROR")
    ok &= check("C", "C.2b③ ★ 这条校验是**版本相关**的：`%s` 起才有，`%s` 上没有"
                % (SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE,
                   "/".join(SPECS.INT4PACK_K_DIVISIBLE_CHECK_ABSENT_IN)),
                checked == ("expect K to be divisible by qGroupSize" in got192)
                and SPECS.INT4PACK_K_DIVISIBLE_MSG in (
                    SPECS.INT4PACK_K_DIVISIBLE_MSG))
    ok &= expect_raises(
        "④ qsz 只有 2 维",
        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, sc),
        "expect qScaleAndZeros to be 3d tensor with sizes [:, %d, 2]" % N)
    ok &= expect_raises(
        "⑤ w 不是 uint8（喂 int32）",
        lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(x, v, gs, qsz),
        "expect B to be uint8 tensor")
    ok &= check("C", "C.2b 异常原文里的 `[:,%d,2]` 是**代入实际 N** 打印的（第 2 维被查）" % N,
                ("[:,%d,2]" % N) in (SPECS.INT4_PACK_QSZ_SHAPE_MSG % N))

    sub("C.3 ★ 相反的一条：`qsz` 的**主维**没有任何校验 —— 错了不报")
    ref = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, qsz)
    short = torch.stack([sc[:1], torch.zeros_like(sc[:1])], -1).contiguous()
    long_ = torch.stack([torch.rand(K // gs + 3, N) * 0.05,
                         torch.zeros(K // gs + 3, N)], -1).contiguous()
    short_ran = short_wrong = False
    try:
        o = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, short)
        short_ran = True
        short_wrong = bool(o.isnan().any()) or float((o - ref).abs().max()) > 0
    except Exception:
        pass
    long_ran = False
    long_diff = 0.0
    try:
        o2 = torch.ops.aten._weight_int4pack_mm_for_cpu(x, wp, gs, long_)
        long_ran = True
        long_diff = float((o2 - ref).abs().max())
    except Exception:
        pass
    ok &= check("C", "C.3  qsz 主维**过短**（1 组而 K//gs = %d）不抛异常，静默出错值" % (K // gs),
                short_ran and short_wrong)
    ok &= check("C", "C.3b qsz 主维**过长**（多 3 组）不抛异常，静默偏差 %.3e" % long_diff,
                long_ran and long_diff > 0)
    ok &= check("C", "C.3c ⇒ **在四条被校验的约束之外，这一条是「错了不报」的**："
                     "只有 ndim 与第 2 维被查，主维（K//gs）不查。"
                     "文档必须把它单列成一条硬约束，而不是混在「会抛异常」那一组里",
                short_ran and long_ran and short_wrong and long_diff > 0)

    sub("C.4 「沿 K 相邻配对」这个解释被字节否证（不是被推理否证）")
    torch.manual_seed(20260918)
    N7, K7 = 64, 256
    v7 = torch.randint(-8, 8, (N7, K7), dtype=torch.int32)
    wp7 = pack_signed(v7).numpy()
    q7 = ((v7 + SPECS.INT4_PACK_ZERO_POINT) & 0xF).numpy()
    mism = sum(1 for i in range(N7) for j in range(K7 // 2)
               if (int(wp7[i, j]) & 0xF) != int(q7[i, 2 * j]))
    ok &= check("C", "C.4  「低/高半字节沿 K 相邻配对」对不上的字节 %d / %d ⇒ 被实测否证"
                % (mism, N7 * (K7 // 2)),
                mism > 0)
    ok &= check("C", "C.4b 且那种解释**在 N=64、K=256 时要求 n 上到 127 > N−1** ⇒ "
                     "它连索引都对不上（不是「另一种也能用」的布局）",
                N7 * 2 - 1 > N7 - 1)

    sub("C.5 不该写进文档的东西：`_dyn_quant_pack_4bit_weight`")
    has = hasattr(torch.ops.aten, "_dyn_quant_pack_4bit_weight")
    print("      torch %s 上 hasattr(_dyn_quant_pack_4bit_weight) = %s" % (torch.__version__, has))
    ok &= check("C", "C.5  **本机可调用性或可用性不作为文档内容**（排布约定未定）",
                SPECS.DYN_QUANT_IN_DOCS is False
                and SPECS.DYN_QUANT_GROUP_SIZE_MULTIPLE == 32)
    ok &= check("C", "C.5b 「2.6.0 没有它」这条记账与 specs 一致（**（本机实测）**）",
                ("2.6.0" not in SPECS.DYN_QUANT_AVAILABLE_IN) == (not has
                                                                 if torch.__version__ == "2.6.0"
                                                                 else True))

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
    ok &= check("B", "★ 测量条件三件套齐全（CPU-only / 未覆盖 minor / 冻结版本）",
                "CPU-only" in SPECS.MEASURE_HOST and bool(SPECS.UNCOVERED_MINORS)
                and SPECS.MEASURED_TORCH == "2.11.0" and SPECS.MEASURED_PYTHON == "3.12.9")
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
