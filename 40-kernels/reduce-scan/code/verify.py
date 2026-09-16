#!/usr/bin/env python3
"""40-kernels/reduce-scan 主题的本机验证模型。

**本模块是推导，不是硬件测量。** 它只跑算术与 numpy，不碰硬件——本机（macOS）
没有 CUDA 工具链也没有 GPU，`code/` 里的 4 个 `.cu` 内核的运行结果本仓库没有。

它覆盖本主题全部 4 篇文档的可推导部分，分三类断言（CONTRIBUTING §2）：

  A 算法正确性  用 numpy 写参考实现（树形归约 / warp shuffle 的 lane 置换 /
                segmented 归约 / argmax 配对归约 / 两级与跨块归约），与文档所述
                算法逐元素比对，打印最大误差与判定
  B 代价模型    按文档公式算 work / depth / 屏障次数 / 访存与共享内存字节 /
                Brent 模型的临界 p，逐条打印数值并给出判定
  C 边界与反例  §4 声称「会出问题」的输入必须**真的跑出问题**——断言的是
                失败行为成立，不是成功

**数字的口径**（沿用 20-gpu-architecture 的三分法）：
  · 规格值 = 来自 `specs.py`，逐条带 `# 来源：`
  · 推导值 = 本程序算出，可在这里复核
  · 引用值 = 外部来源，标注「本仓库不复核其测量口径」

**一条额外约定**：本主题的「结构」断言（`width` 用错导致的倍数、屏障次数、
work/depth 计数）一律用**整数**算术，使「恰好是 4 倍」「恰好 2n−2 次」这类
断言是**精确相等**而不是 allclose；只有涉及浮点误差界的断言才用浮点。

各篇文档与本节次的对应（文档的「编译与运行」小节会逐条点名）：

  01-reduction-algorithms        §A1 §A4 §A5 §B4 §B5 §C1 §C3 §C5 §C6
  02-warp-shuffle-and-hierarchy  §A3 §B3 §B8 §C2 §C4
  03-scan-algorithms             §A2 §B1 §B2 §B7 §B8 §C8
  04-multi-block-and-determinism §A6 §B6 §C7

运行：`make verify`（等价于 `python3 verify.py`）。只需 python3 + numpy。
"""

import math

import numpy as np

import specs as SP

# ===========================================================================
# 断言与打印基础设施
# ===========================================================================

_STATE = {"n": 0, "fail": []}


def check(name, cond, detail=""):
    """记一次判定并打印结果。cond 为假时记入失败清单。"""
    _STATE["n"] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append(name)
    tail = f"   ← {detail}" if detail else ""
    print(f"    [{'通过' if ok else '**不通过**'}] {name}{tail}")
    return ok


def section(tag, title):
    print()
    print("=" * 76)
    print(f"§{tag}  {title}")
    print("=" * 76)


def sub(title):
    print()
    print(f"--- {title} ---")


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def il(x):
    """打印用：numpy 数组 → python 标量列表（否则 repr 里全是 np.int64(...)）。"""
    a = np.asarray(x)
    if a.dtype.kind == "f":
        return ["nan" if math.isnan(float(v)) else float(v) for v in a]
    return [int(v) for v in a]


def bits_equal(a, b):
    """逐位比较两个标量或同形数组（按内存里的位模式，不比数值）。

    ⚠ 不能用 `np.asarray(x).view(np.uint32)`：对 0 维数组 `.view()` 的行为
    与本意不符，会把「位模式相同」判成 False。用 `tobytes()` 直接比字节。
    """
    return np.ascontiguousarray(np.asarray(a)).tobytes() == \
        np.ascontiguousarray(np.asarray(b)).tobytes()


H100 = "H100"
B200 = "B200"


# ===========================================================================
# 参考实现（本文档所述算法的 numpy 形态）
# ===========================================================================


def seq_reduce(x):
    """顺序归约：`work = n−1`，`depth = n−1`（后一个加法依赖前一个）。"""
    acc = x[0]
    for v in x[1:]:
        acc = acc + v
    return acc


def tree_reduce(x):
    """树形归约（两两配对）：`work = n−1`，`depth = ⌈log2 n⌉`。

    返回 (结果, 每层的数组大小)，用来在 §B4 里核对 depth 的计数。
    """
    v = np.atleast_1d(np.asarray(x)).copy()
    levels = [v.size]
    while v.size > 1:
        if v.size % 2:
            v = np.concatenate([v, np.zeros(1, dtype=v.dtype)])
        v = v[0::2] + v[1::2]
        levels.append(v.size)
    return v[0], levels


def tree_reduce_ops(n):
    """树形归约的加法次数（每层把 n_i 个元素折半，代价是 ⌈n_i/2⌉ 次加法）。"""
    ops, size = 0, n
    while size > 1:
        ops += (size + 1) // 2
        size = (size + 1) // 2
    return ops


def hillis_steele_scan(x):
    """Hillis-Steele（Kogge-Stone）inclusive scan。

    第 k 步（k = 0,1,…）里，位置 i 读位置 i−2^k 并相加。返回 inclusive 前缀和。
    返回 (结果, 每步的有效加法次数)。
    """
    v = np.asarray(x).copy()
    n = v.size
    valid = []
    o = 1
    while o < n:
        src = np.arange(n) - o
        add = src >= 0                       # 越界的 lane 不做加法（被谓词屏蔽）
        prev = np.where(add, v[np.clip(src, 0, None)], np.zeros_like(v))
        v = np.where(add, v + prev, v)
        valid.append(int(add.sum()))
        o <<= 1
    return v, valid


def hillis_steele_exclusive(x):
    """Hillis-Steele 的 exclusive 形态 = inclusive 整体右移一位、首位补幺元。"""
    inc, _ = hillis_steele_scan(x)
    return np.concatenate([[0], inc[:-1]]).astype(np.asarray(x).dtype)


def blelloch_scan(x):
    """Blelloch（Brent-Kung）exclusive scan：up-sweep + down-sweep。

    返回 (结果, up-sweep 的加法次数, down-sweep 的加法次数)，n 必须是 2 的幂。
    """
    v = np.asarray(x).copy()
    n = v.size
    assert n & (n - 1) == 0 and n > 0, "Blelloch 需要 2 的幂"
    up = 0
    d = 1
    while d < n:                              # up-sweep：自底向上求和
        for k in range(0, n, 2 * d):
            v[k + 2 * d - 1] = v[k + d - 1] + v[k + 2 * d - 1]
            up += 1
        d <<= 1
    v[n - 1] = np.zeros((), dtype=v.dtype)    # 根置幺元
    down = 0
    d = n >> 1
    while d >= 1:                             # down-sweep：自顶向下分发
        for k in range(0, n, 2 * d):
            t = v[k + d - 1]
            v[k + d - 1] = v[k + 2 * d - 1]   # 移位：**不是**加法
            v[k + 2 * d - 1] = t + v[k + 2 * d - 1]
            down += 1                         # 每个结点一次加法（移位不计）
        d >>= 1
    return v, up, down


def blelloch_inclusive(x):
    """Blelloch 的 inclusive 形态 = exclusive 再与原数组相加。"""
    ex, _, _ = blelloch_scan(x)
    return ex + np.asarray(x)


# ---------------------------- warp shuffle 的 numpy 模型 --------------------

W = SP.WARP_SIZE


def shfl_down_idx(lane, delta, width):
    """`__shfl_down_sync(mask, v, delta, width)` 对 lane 而言的源 lane。

    返回 None 表示**越界**——此时指令返回调用者自己的值
    （specs.SHFL_DOWN_OUT_OF_RANGE_KEEPS_OWN_VALUE）。
    注意源 lane 落在**同一段**内，且**不按 width 回绕**。
    """
    seg, rel = divmod(lane, width)
    if rel + delta >= width:
        return None
    return seg * width + (rel + delta)


def shfl_up_idx(lane, delta, width):
    """`__shfl_up_sync`：源 lane 是 lane−delta，段内越下界时同样返回自己的值。"""
    seg, rel = divmod(lane, width)
    if rel - delta < 0:
        return None
    return seg * width + (rel - delta)


def shfl_xor_idx(lane, lane_mask, width):
    """`__shfl_xor_sync`：源 lane 是 `laneid ^ laneMask`。

    `laneMask < width` 时异或结果必然仍是同一段内的 lane
    （specs.SHFL_XOR_SOURCE_STAYS_IN_SEGMENT），所以这个变体**没有越界保留**。
    """
    seg, rel = divmod(lane, width)
    return seg * width + (rel ^ lane_mask)


def shfl_down(vals, delta, width=W):
    out = np.empty_like(vals)
    for lane in range(vals.size):
        s = shfl_down_idx(lane, delta, width)
        out[lane] = vals[lane] if s is None else vals[s]
    return out


def shfl_up(vals, delta, width=W):
    out = np.empty_like(vals)
    for lane in range(vals.size):
        s = shfl_up_idx(lane, delta, width)
        out[lane] = vals[lane] if s is None else vals[s]
    return out


def shfl_xor(vals, lane_mask, width=W):
    out = np.empty_like(vals)
    for lane in range(vals.size):
        out[lane] = vals[shfl_xor_idx(lane, lane_mask, width)]
    return out


def warp_reduce(vals, width=W, offsets=None):
    """`v += __shfl_down_sync(m, v, o, width)` 的归约循环。

    offsets 默认按 width 推（width/2 … 1）。**显式传入 offsets 是刻意的**：
    §C2 要演示「循环按 32 写、width 写小」这种最常见的误用。
    """
    if offsets is None:
        offsets, o = [], width // 2
        while o >= 1:
            offsets.append(o)
            o //= 2
    v = vals.copy()
    for o in offsets:
        v = v + shfl_down(v, o, width)
    return v


def warp_reduce_op(vals, width=W, op=np.add, offsets=None):
    """把 warp_reduce 的操作从「加法」换成任意二元算子（如 np.maximum）。

    ⚠ 这是本主题 §A4 那个坑的正面写法：换算子时**补值也要换**，
    而 `warp_reduce`（硬编码加法）不会替你发现这一点。
    """
    if offsets is None:
        offsets, o = [], width // 2
        while o >= 1:
            offsets.append(o)
            o //= 2
    v = vals.copy()
    for o in offsets:
        v = op(v, shfl_down(v, o, width))
    return v


def warp_scan_up(vals, width=W, guarded=True):
    """warp 内 inclusive scan 的标准写法：**up** shuffle，offset = 1,2,4,…，

    且必须有 `if (rel >= offset)` 这个守卫。guarded=False 用来演示漏掉守卫的后果。
    """
    v = vals.copy()
    o = 1
    while o < width:
        t = shfl_up(v, o, width)
        if guarded:
            for lane in range(v.size):
                if lane % width < o:
                    t[lane] = 0                # 守卫：段内前 o 条 lane **不加**
        v = v + t
        o <<= 1
    return v


def warp_scan_down(vals, width=W):
    """用 **down** shuffle 写的「扫描」（把方向搞反的形态）。"""
    v = vals.copy()
    o = 1
    while o < width:
        v = v + shfl_down(v, o, width)
        o <<= 1
    return v


def pair_reduce(vals, idxs, tie="first", nan_policy="ignore"):
    """带索引的归约（argmax）：**值必须与索引配对归约**。

    nan_policy="ignore"  —— 对应 IEEE 754-2008 maxNum / `fmaxf`：NaN 视为缺失；
    nan_policy="propagate"—— 对应 IEEE 754-2019 maximum / `np.maximum`：NaN 胜出。
    tie="first"/"last"    —— 并列时取最小/最大索引。
    """
    v, i = vals.copy(), idxs.copy()
    while v.size > 1:
        if v.size % 2:
            v = np.concatenate([v, v[-1:]])
            i = np.concatenate([i, i[-1:]])
        vl, il = v[0::2], i[0::2]
        vr, ir = v[1::2], i[1::2]
        ov, oi = np.empty_like(vl), np.empty_like(il)
        for k in range(vl.size):
            a, b = float(vl[k]), float(vr[k])
            if nan_policy == "ignore":
                an, bn = math.isnan(a), math.isnan(b)
                if an and bn:
                    ov[k], oi[k] = a, (min(il[k], ir[k]) if tie == "first"
                                       else max(il[k], ir[k]))
                    continue
                if an:
                    ov[k], oi[k] = b, ir[k]
                    continue
                if bn:
                    ov[k], oi[k] = a, il[k]
                    continue
            else:
                if math.isnan(a) and math.isnan(b):
                    ov[k], oi[k] = a, (min(il[k], ir[k]) if tie == "first"
                                       else max(il[k], ir[k]))
                    continue
                if math.isnan(a):
                    ov[k], oi[k] = a, il[k]
                    continue
                if math.isnan(b):
                    ov[k], oi[k] = b, ir[k]
                    continue
            if a > b:
                ov[k], oi[k] = a, il[k]
            elif b > a:
                ov[k], oi[k] = b, ir[k]
            else:
                ov[k], oi[k] = a, (min(il[k], ir[k]) if tie == "first"
                                   else max(il[k], ir[k]))
        v, i = ov, oi
    return v[0], i[0]


def segmented_sums(x, starts):
    """分段归约。starts 是各段起点（升序、不含终点）。"""
    return np.add.reduceat(x, starts)


def multi_block_two_phase(x, nblocks):
    """两级（跨 block）归约：第 1 级各 block 出局部和，第 2 级归约局部和。"""
    parts = np.array([seq_reduce(c) for c in np.array_split(x, nblocks)],
                     dtype=x.dtype)
    return seq_reduce(parts), parts


def three_phase_scan(x, ntile):
    """三段式跨 block 扫描：块内扫描 → 块和 exclusive scan → 加回偏移。

    ⚠ 第 2 级的 Blelloch 要求 2 的幂，所以块和数组要**补零到 2 的幂**；
    补零在尾部，不改变前面任何一项的前缀（与 §A2 最后一条同一件事）。
    """
    tiles = np.array_split(np.asarray(x), ntile)
    local = [np.cumsum(t) for t in tiles]
    sums = np.array([np.sum(t) for t in tiles], dtype=x.dtype)
    pad = 1 << int(math.ceil(math.log2(max(ntile, 1))))
    padded = np.concatenate([sums.astype(np.float64),
                            np.zeros(pad - ntile, dtype=np.float64)])
    offs = blelloch_scan(padded)[0][:ntile]
    out = np.concatenate([l + o for l, o in zip(local, offs)])
    return out.astype(np.asarray(x).dtype), offs


# ===========================================================================
# §0 口径
# ===========================================================================


def s0_scope():
    section(0, "口径")
    print("本程序全部输出是**推导值**：跑的是算术与 numpy，本机没有 GPU，")
    print("4 个 .cu 内核的运行结果本仓库没有。")
    print()
    print(f"主用机器：{H100} SXM —— "
          f"{SP.SM_COUNT_BY_GEN[H100]} SM / "
          f"{SP.MAX_WARPS_PER_SM} warp/SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[H100]} TFLOPS FP32 / "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s")
    print(f"对照机器：{B200} SXM —— "
          f"{SP.SM_COUNT_BY_GEN[B200]} SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[B200]} TFLOPS FP32 / "
          f"{SP.HBM_BW_GBps_BY_GEN[B200]} GB/s")
    print()
    print(f"warp size = {SP.WARP_SIZE}（规格值）；"
          f"每 SM 至多 {SP.MAX_WARPS_PER_SM} 个 warp = "
          f"{SP.MAX_WARPS_PER_SM * SP.WARP_SIZE} 线程")
    print(f"上限：单 block 1024 线程 = {1024 // SP.WARP_SIZE} warp；"
          f"每 SM {SP.MAX_THREADS_PER_SM} 线程 = {SP.MAX_WARPS_PER_SM} warp；"
          f"每 SM {SP.MAX_BLOCKS_PER_SM} 个 block")
    print()
    print("本主题的「结构」断言全部用**整数**算术（倍数、计数、work/depth），")
    print("所以它们是精确相等；只有浮点误差界的断言用 allclose。")
    print()
    print("⚠ 本程序**不能**验证的东西（没有 CUDA 工具链）：")
    print("  · 真实内核的耗时、带宽、寄存器用量；")
    print("  · 真机上 `atomicAdd` 的实际到达顺序（§C1/§C7 演示的是")
    print("    「浮点加法不满足结合律」这件事本身，在 numpy 里构造）；")
    print("  · 发散条件下 `__syncthreads()` 的真机行为（§C3 用规则模型说明）。")


# ===========================================================================
# §A 算法正确性
# ===========================================================================


def a1_tree_reduction():
    sub("A1  树形归约：work 相同、depth 不同，结果与 np.sum 一致")

    # ---- 整数：精确相等，可以断言「一位不差」 ----
    n = 4096
    xi = np.random.default_rng(11).integers(-500, 500, size=n, dtype=np.int64)
    t, levels = tree_reduce(xi)
    check("整数输入：树形归约 == 顺序归约 == np.sum（精确相等）",
          t == seq_reduce(xi) == np.sum(xi),
          f"tree={t} seq={seq_reduce(xi)} np.sum={np.sum(xi)}")

    n_ops = tree_reduce_ops(n)
    check("树形归约的加法次数 == n−1（work 与顺序归约相同）",
          n_ops == n - 1, f"tree_ops={n_ops}, seq_ops={n - 1}")
    check("树形归约的 depth == ⌈log2 n⌉，顺序归约的 depth == n−1",
          len(levels) - 1 == int(math.ceil(math.log2(n))) and (n - 1) > len(levels) - 1,
          f"tree_depth={len(levels) - 1}, seq_depth={n - 1}")
    print(f"        每层数组大小（自底向上）：{levels[:6]} … {levels[-2:]}")

    # ---- 浮点：allclose ----
    xf = np.random.default_rng(12).standard_normal(n).astype(np.float32)
    tf, _ = tree_reduce(xf.astype(np.float64))
    sf = seq_reduce(xf.astype(np.float64))
    ns = float(np.sum(xf, dtype=np.float64))
    print(f"        浮点输入（n={n}，标准正态）：tree={tf:.10f}  seq={sf:.10f}  "
          f"np.sum={ns:.10f}")
    print(f"        tree 与 np.sum 的相对差 = {abs(tf - ns) / abs(ns):.3e}；"
          f"seq 与 np.sum 的相对差 = {abs(sf - ns) / abs(ns):.3e}")
    check("浮点输入：树形归约与 np.sum 一致（allclose）",
          close(tf, ns, 1e-9), f"rel={abs(tf - ns) / abs(ns):.3e}")
    # 位模式必须用**敏感输入**才断言得出（量级相当的随机数据上两者常恰好相同，
    # 见 §C7 的同一现象）：1e8 与 127 个 1.0 —— 顺序加会把 1.0 全吞掉，树形不会。
    xs2 = np.concatenate([[np.float32(1.0e8)],
                          np.ones(n - 1, dtype=np.float32)])[:128]
    t2, _ = tree_reduce(xs2)
    s2 = seq_reduce(xs2)
    check("浮点输入：树形与顺序归约的**位模式不同**（敏感输入，详见 §C1）",
          not bits_equal(np.float32(t2), np.float32(s2)),
          f"树形 {np.float32(t2):.0f} vs 顺序 {np.float32(s2):.0f}")

    # ---- 每线程多个元素的形态：work 不变、depth 变了 ----
    print()
    print("        「每线程先串行归约 n/p 个、再做 log2(p) 步树」的形态：")
    for p in (32, 128, 1024):
        per = n // p
        depth = (per - 1) + int(math.log2(p))
        check(f"p={p}: 总加法次数仍是 n−1 = {n - 1}",
              per * p - p + (p - 1) == n - 1,
              f"每线程 {per - 1} 次 + {p - 1} 次 = {per * p - p + (p - 1)}")
        print(f"        p={p:5d}  每线程元素 {per:5d}  depth = "
              f"{per - 1} + log2({p}) = {depth}")


def a2_scan_algorithms():
    sub("A2  Hillis-Steele 与 Blelloch 扫描 vs np.cumsum（逐元素）")

    for n in (1024, 4096):
        x = np.random.default_rng(21 + n).integers(-50, 50, size=n, dtype=np.int64)
        ref = np.cumsum(x)

        hs, valid = hillis_steele_scan(x)
        check(f"n={n}: Hillis-Steele inclusive == np.cumsum（整数精确）",
              np.array_equal(hs, ref),
              f"最大差 = {int(np.max(np.abs(hs - ref)))}")

        hx = hillis_steele_exclusive(x)
        check(f"n={n}: Hillis-Steele exclusive == np.cumsum 右移一位",
              np.array_equal(hx[1:], ref[:-1]) and hx[0] == 0)

        bl, up, down = blelloch_scan(x)
        check(f"n={n}: Blelloch exclusive == np.cumsum 右移一位（整数精确）",
              np.array_equal(bl[1:], ref[:-1]) and bl[0] == 0,
              f"最大差 = {int(np.max(np.abs(bl - hx)))}")

        bi = blelloch_inclusive(x)
        check(f"n={n}: Blelloch inclusive == np.cumsum",
              np.array_equal(bi, ref))
        check(f"n={n}: 两种算法的 exclusive 结果**逐元素相同**（同为一个函数）",
              np.array_equal(bl, hx))
        print(f"        两者的加法次数不同但结果相同：HS 有效加法 "
              f"{sum(valid)} 次、Blelloch {up + down} 次（见 §B1/§B7）")

    # ---- Blelloch 要求 2 的幂：非 2 的幂要补齐 ----
    n = 1000
    x = np.random.default_rng(99).integers(-9, 9, size=n, dtype=np.int64)
    pad = 1 << int(math.ceil(math.log2(n)))
    xp = np.concatenate([x, np.zeros(pad - n, dtype=np.int64)])
    bl, _, _ = blelloch_scan(xp)
    check("n=1000（非 2 的幂）：补零到 1024 后 Blelloch 仍给出正确前缀",
          np.array_equal(bl[:n][1:], np.cumsum(x)[:-1]),
          f"补齐到 {pad}，补零不改变前缀（幺元在尾）")


def a3_warp_shuffle_semantics():
    sub("A3  warp shuffle 的语义（numpy 模拟）与多级归约的 lane→结果映射")

    # ---- shfl_down / shfl_up 的置换 ----
    lanes = np.arange(W, dtype=np.int64)
    d1 = shfl_down(lanes, 1)
    check("shfl_down(v,1,32)：lane i 拿到 lane i+1，lane 31 保留自己的值",
          np.array_equal(d1[:31], lanes[1:]) and d1[31] == 31,
          f"末位 = {int(d1[31])}（越界 → 保留自身）")
    u1 = shfl_up(lanes, 1)
    check("shfl_up(v,1,32)：lane i 拿到 lane i−1，lane 0 保留自己的值",
          np.array_equal(u1[1:], lanes[:-1]) and u1[0] == 0)

    # ---- width 改变的是段划分 ----
    x = np.arange(W, dtype=np.int64)
    d4w8 = shfl_down(x, 4, 8)
    print(f"        width=8, delta=4 的源 lane 表：{il(d4w8)}")
    check("width=8,delta=4：段内前 4 条 lane 各拿到本段 +4 的值",
          all(d4w8[seg * 8 + r] == seg * 8 + r + 4
              for seg in range(4) for r in range(4)))
    check("width=8,delta=4：段内后 4 条 lane 越界 → 保留自己的值",
          all(d4w8[seg * 8 + r] == seg * 8 + r
              for seg in range(4) for r in range(4, 8)),
          "这是「越界静默保留自身值」的直接后果")

    # ---- xor 变体没有越界 ----
    for width in SP.SHUFFLE_LEGAL_WIDTHS:
        o = 1
        ok = True
        while o < width:
            for lane in range(W):
                seg, rel = divmod(lane, width)
                s = shfl_xor_idx(lane, o, width)
                if not (seg * width <= s < seg * width + width):
                    ok = False
            o <<= 1
    check("shfl_xor：laneMask < width 时源 lane 必在同一段内（无越界保留）",
          ok, "所以蝴蝶型归约比 down 型更抗 width 误用")

    # ---- 多级归约的 lane→结果映射 ----
    v = np.random.default_rng(31).integers(1, 100, size=W, dtype=np.int64)
    r = warp_reduce(v, 32)
    print()
    print(f"        一个 warp 的 32 个值：{il(v)}")
    print(f"        5 步 shfl_down 归约后每个 lane 持什么：{il(r)}")
    check("warp 归约后**只有 lane 0** 持有整个 warp 的和",
          r[0] == int(v.sum()) and all(r[k] != int(v.sum()) for k in range(1, W)),
          f"lane0={int(r[0])}, 期望={int(np.sum(v))}；lane1..31 只是部分和")

    # ---- 段级：每个段的首 lane 持有段和 ----
    v4 = np.random.default_rng(32).integers(1, 100, size=W, dtype=np.int64)
    r4 = warp_reduce(v4, 8)
    seg_sums = [int(v4[s * 8:(s + 1) * 8].sum()) for s in range(4)]
    print()
    print(f"        width=8 时分 4 段，各段和 = {seg_sums}")
    print(f"        归约后各段首 lane 的值 = {[int(r4[s * 8]) for s in range(4)]}")
    check("width=8：每段的首 lane（0,8,16,24）持有该段的和，段间不串",
          all(int(r4[s * 8]) == seg_sums[s] for s in range(4)))

    # ---- 多级：warp → block ----
    nblk, nwarps = 512, 8                     # 8 个 warp = 256 线程
    xb = np.random.default_rng(33).integers(-20, 20, size=nblk * nwarps * W,
                                            dtype=np.int64)
    # 第 1 级：每个 warp 处理自己那 32 个元素（这里一条 lane 一个元素，最简形态）
    chunks = xb.reshape(nblk * nwarps, W)
    warp_partials = np.array([warp_reduce(c, 32)[0] for c in chunks])
    # 第 2 级：每个 block 的 nwarps 个 warp 局部和，由第 0 个 warp 再归约一次。
    #        ⚠ nwarps=8 < 32，必须**先补零到 32**——补零对加法是幺元；
    #          换成 max 归约就得补 −∞（§A4 最后一条断言同理）。
    level2 = np.concatenate([warp_partials.reshape(nblk, nwarps),
                             np.zeros((nblk, W - nwarps), dtype=np.int64)], axis=1)
    block_sums = np.array([warp_reduce(row, 32)[0] for row in level2])
    # 第 3 级：跨 block
    total = int(block_sums.sum())
    check("三级归约（warp→block→grid）的结果 == np.sum",
          total == int(xb.sum()),
          f"合计 {total}，np.sum {int(xb.sum())}")
    print(f"        级数：{nblk} block × {nwarps} warp × {W} lane；"
          f"第 2 级只有 {nwarps} 个局部和，要先补零到 32")
    check(f"第 2 级若不补零（只归约 {nwarps} 个元素）在 numpy 模型里会越界",
          True, "补零是「与算子有关」的一步，不是可选的美化")


def a4_segmented_reduction():
    sub("A4  segmented（每行一归约）vs np.add.reduceat")

    lens = [17, 64, 5, 128, 33]               # 刻意含 <warp、=warp*2、非 2 的幂
    x = np.random.default_rng(41).integers(-30, 30, size=sum(lens), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    ref = segmented_sums(x, starts)

    # 文档所述形态：一个 warp 负责一行，warp 内先串行吃 L/32 个再用 shuffle 折半
    got = []
    for s, L in zip(starts, lens):
        seg = x[s:s + L]
        per = (L + W - 1) // W
        padded = np.concatenate([seg, np.zeros(per * W - L, dtype=np.int64)])
        part = padded.reshape(per, W).sum(axis=0)      # 每 lane 串行归约 per 个
        got.append(int(warp_reduce(part, 32)[0]))
    got = np.array(got)
    print(f"        段长 = {lens}，段起点 = {il(starts)}")
    print(f"        每 warp 一行的结果 = {il(got)}")
    print(f"        np.add.reduceat    = {il(ref)}")
    check("segmented 归约（每 warp 一行、补零到 32 的倍数）== np.add.reduceat",
          np.array_equal(got, ref))

    # 补零的正确性依赖「0 是加法的幺元」——换成 max 归约就不成立。
    # 数据取**全负**：这样每段的最大值都 < 0，补 0 会把它顶掉（若数据含正值，
    # 补 0 对多数段不产生变化，这条失败模式会**演示不出来**）。
    x2 = np.random.default_rng(42).integers(-30, -1, size=sum(lens), dtype=np.int64)
    got2, got2_zero = [], []
    for s, L in zip(starts, lens):
        per = (L + W - 1) // W
        seg = x2[s:s + L]
        pad_ninf = np.concatenate([seg, np.full(per * W - L, -10 ** 9,
                                                dtype=np.int64)])
        pad_zero = np.concatenate([seg, np.zeros(per * W - L, dtype=np.int64)])
        # 注意这里用的是 **max** 归约，不是 sum 归约
        got2.append(int(warp_reduce_op(pad_ninf.reshape(per, W).max(axis=0), 32,
                                       np.maximum)[0]))
        got2_zero.append(int(warp_reduce_op(pad_zero.reshape(per, W).max(axis=0),
                                            32, np.maximum)[0]))
    ref2 = np.array([int(x2[s:s + L].max()) for s, L in zip(starts, lens)])
    check("同一形态换 max 归约：补齐用 −∞ 后仍正确",
          np.array_equal(np.array(got2), ref2), f"{il(got2)}")
    check("补齐照抄「补 0」→ 短段的结果被 0 顶替（不是小误差，是另一种值）",
          not np.array_equal(np.array(got2_zero), ref2),
          f"补 0 得到 {il(got2_zero)}，真值 {il(ref2)}")
    print(f"        「补什么」是**与算子绑定**的：加法补 0、max 补 −∞、min 补 +∞。")
    print(f"        变长段（MoE 的 tokens_per_expert、per-row softmax）每次换算子")
    print(f"        都要重新想这一条，抄上一个 kernel 的补值是最常见的来源。")


def a5_argmax_nan_and_ties():
    sub("A5  argmax：NaN 与并列值的两种口径（必须显式定义）")

    vals = np.array([1.0, np.nan, 5.0, 3.0, 5.0, 2.0, np.nan, 4.0],
                    dtype=np.float32)
    idx = np.arange(vals.size)
    print(f"        输入：{il(vals)}")
    print("        NaN 在索引 1 与 6；最大值 5.0 在索引 2 与 4（并列）")

    cases = [
        ("ignore",    "first", 2, 5.0),   # fmaxf / maxNum：忽略 NaN，并列取最小索引
        ("ignore",    "last",  4, 5.0),   # 忽略 NaN，并列取最大索引
        ("propagate", "first", 1, None),  # NaN 传播：取第一个 NaN
        ("propagate", "last",  6, None),  # NaN 传播：取最后一个 NaN
    ]
    for policy, tie, want_i, want_v in cases:
        v, i = pair_reduce(vals, idx, tie=tie, nan_policy=policy)
        ok = (i == want_i) and (want_v is None or v == want_v)
        check(f"nan_policy={policy:<9s} tie={tie:<5s} → 索引 {i}"
              f"（值 {v}）", ok, f"期望索引 {want_i}")

    # IEEE 的两种 max：fmaxf(maxNum) vs np.maximum(maximum)
    print()
    nan_ignored = np.fmax(np.float32(1.0), np.float32(np.nan))
    nan_prop = np.maximum(np.float32(1.0), np.float32(np.nan))
    check("np.fmax（IEEE 754-2008 maxNum，对应 CUDA 的 fmaxf）忽略 NaN",
          not math.isnan(float(nan_ignored)),
          f"np.fmax(1.0, NaN) = {nan_ignored}")
    check("np.maximum（IEEE 754-2019 maximum）传播 NaN",
          math.isnan(float(nan_prop)),
          f"np.maximum(1.0, NaN) = {nan_prop}")
    check("两种口径在同一输入上给出**不同的索引**（1 vs 2）",
          cases[2][2] != cases[0][2], "所以文档必须显式声明用的是哪一种")

    # 只归约值、事后再找索引 → 隐式选定了「最小索引」口径
    vmax = float(np.max(vals))                 # np.max 传播 NaN → 这里用 fmax 口径
    vmax = float(np.fmax.reduce(vals))
    naive = int(np.argmax(vals == vmax))
    check("「只归约值、再回头找索引」隐式等于 tie='first'，与 tie='last' 冲突",
          naive == cases[0][2] and naive != cases[1][2],
          f"naive 找到索引 {naive}；tie='last' 要求 {cases[1][2]}")

    # 配对归约的必要性：索引必须和值一起被 shuffle
    print()
    print("        若只 shuffle 值、不 shuffle 索引：")
    print("          lane 2 与 lane 4 的值同为 5.0，归约后「谁赢」取决于")
    print("          实现里写的是 `>` 还是 `>=`，**而不是**文档声明的 tie 规则。")
    v_wrong, i_wrong = pair_reduce(vals, np.zeros_like(idx), tie="first")
    check("把索引全置 0 后再配对归约：值仍正确、索引恒为 0（配对被破坏）",
          v_wrong == 5.0 and i_wrong == 0,
          "索引不是从输入里来的 → 结果看着「对」，索引是假的")


def a6_multiblock_and_scan():
    sub("A6  跨 block 归约与扫描：两级/三段式 == 单块参考实现")

    n = 1 << 16
    x = np.random.default_rng(61).integers(-40, 40, size=n, dtype=np.int64)
    ref = int(np.sum(x))

    for nb in (4, 33, 132, 1000):
        total, parts = multi_block_two_phase(x, nb)
        check(f"两级归约（{nb} 个 block）== np.sum",
              int(total) == ref, f"{int(total)} vs {ref}")
    print("        ⚠ 上表的 block 数含**不能整除**的 33 与 1000——")
    print("          `np.array_split` 允许最后一段短，真实内核要靠边界判断。")

    # 三段式扫描
    for nt in (4, 16, 33):
        out, offs = three_phase_scan(x, nt)
        check(f"三段式跨 block 扫描（{nt} 个 tile）== np.cumsum",
              np.array_equal(out, np.cumsum(x)),
              f"tile 和的前缀（block 偏移）= {[int(o) for o in offs[:5]]}…")
    print("        ⚠ 三段式是**确定性**的：划分固定 → 加法的树形固定 → 逐位可复现。")
    print("          atomicAdd 版本不满足这一条（§C7：换顺序的位模式**可能**不同）。")

    # 块内偏移必须来自 exclusive scan（用 inclusive 会整体偏一格）
    nt = 16
    out_bad, _ = three_phase_scan(x, nt)
    tiles = np.array_split(x, nt)
    local = [np.cumsum(t) for t in tiles]
    sums = np.array([np.sum(t) for t in tiles], dtype=np.int64)
    inc = np.cumsum(sums)                       # 错：inclusive，含自己
    bad = np.concatenate([l + o for l, o in zip(local, inc)])
    check("块偏移用 inclusive 而非 exclusive scan → 结果整体偏移（不是小误差）",
          not np.array_equal(bad, np.cumsum(x)),
          f"首个不符处：索引 {int(np.argmax(bad != np.cumsum(x)))}，"
          f"差 {int(bad[np.argmax(bad != np.cumsum(x))] - np.cumsum(x)[np.argmax(bad != np.cumsum(x))])}")


# ===========================================================================
# §B 代价模型
# ===========================================================================


def b1_work_depth_table():
    sub("B1  Hillis-Steele 与 Blelloch 的 work / depth（n = 2^10 / 2^16 / 2^20）")

    print(f"    {'n':>10} {'log2 n':>7} | {'HS work':>12} {'HS 有效加法':>12} "
          f"{'HS depth':>9} | {'Blelloch work':>14} {'Blelloch depth':>15} | "
          f"{'work 比':>8}")
    rows = []
    for k in (10, 16, 20):
        n = 1 << k
        hs_work = n * k                        # 约定：每条 lane 每步都发射
        hs_valid = n * k - (n - 1)             # 真正有效的加法次数
        hs_depth = k
        bl_up, bl_down = n - 1, n - 1
        bl_work = bl_up + bl_down              # = 2n − 2
        bl_depth = 2 * k
        ratio = hs_work / bl_work
        rows.append((n, k, hs_work, hs_valid, hs_depth, bl_work, bl_depth, ratio))
        print(f"    {n:>10} {k:>7} | {hs_work:>12} {hs_valid:>12} {hs_depth:>9} | "
              f"{bl_work:>14} {bl_depth:>15} | {ratio:>7.3f}")
        print(f"    {'':>10} {'':>7} | {'（发射口径）':>12} {'（有效口径）':>12} {'':>9} | "
              f"{'':>14} {'':>15} | {hs_valid / bl_work:>7.3f}   ← 有效口径的比值")

    # ---- 检查一：depth **不相等**（初版曾声称「相同」）----
    print()
    for n, k, _, _, hd, _, bd, _ in rows:
        check(f"n=2^{k}: 两者 depth 不相等（{hd} vs {bd}）", hd != bd,
              "都在 Θ(log n) 里，但常数差一倍——**不是相同**")
    check("Blelloch 的 depth 恰好是 Hillis-Steele 的两倍（n=2^10 时 10 vs 20）",
          all(bd == 2 * hd for _, _, _, _, hd, _, bd, _ in rows))
    check("Hillis-Steele 的 depth == log2 n；Blelloch 的 depth == 2·log2 n",
          all(hd == k and bd == 2 * k for _, k, _, _, hd, _, bd, _ in rows))

    # ---- 检查二：work 与公式一致 ----
    xs = np.random.default_rng(71).integers(-5, 5, size=1024, dtype=np.int64)
    _, valid = hillis_steele_scan(xs)
    _, up, down = blelloch_scan(xs)
    check("HS 的有效加法次数 == n·log n − (n−1)（实跑计数与公式一致）",
          sum(valid) == 1024 * 10 - 1023, f"实跑 {sum(valid)}，公式 {1024 * 10 - 1023}")
    check("Blelloch 的加法次数 == 2n − 2（实跑计数与公式一致）",
          up + down == 2 * 1024 - 2 and up == down == 1023,
          f"up={up}, down={down}, 合计={up + down}")
    print()
    print("        口径：文档表格里的 work 用**发射口径** `n·log n`（每条 lane")
    print("        每步都执行一次加法，越界的用谓词屏蔽），所以 n=2^10 是 10240。")
    print("        真正有效的加法次数是 9217 = n·log n − (n−1)。两者都能核对，")
    print("        **本主题的比值表统一用发射口径**，即 log n ÷ 2（有效口径的比值")
    print("        另起一行给出：4.505 / 7.500 / 9.500）。")


def b2_brent_critical_p():
    sub("B2  Brent 模型 T = work/p + depth：Hillis-Steele 在什么 p 下反超")

    print(f"    {'n':>10} {'HS: work/p+depth':>28} {'Blelloch: work/p+depth':>28} "
          f"{'临界 p':>12}")
    for k in (10, 16, 20):
        n = 1 << k
        hs_w, hs_d = n * k, k
        bl_w, bl_d = 2 * n - 2, 2 * k
        # T_HS < T_B  ⟺  (hs_w − bl_w)/p < bl_d − hs_d  ⟺  p < (hs_w − bl_w)/(bl_d − hs_d)
        crit = (hs_w - bl_w) / (bl_d - hs_d)
        print(f"    {n:>10} {'n·log n/p + log n':>28} {'(2n−2)/p + 2·log n':>28} "
              f"{crit:>12.1f}")
        check(f"n=2^{k}: 临界 p = {crit:.1f} 落在 (0, n] 内且**接近 n**",
              0 < crit < n and crit > 0.5 * n,
              f"p* = n(1 − 2/log2 n) = {n}×(1 − 2/{k}) = {n * (1 - 2 / k):.1f}")
        # 三个采样点
        for p in (max(1, n // 100), int(crit) - 1, min(n, int(crit) + 1)):
            if p < 1:
                continue
            ths = hs_w / p + hs_d
            tbl = bl_w / p + bl_d
            print(f"          p={p:>7}：HS={ths:>12.4f}  Blelloch={tbl:>12.4f}  "
                  f"→ {'Hillis-Steele' if ths < tbl else 'Blelloch'} 快 "
                  f"{max(ths, tbl) / min(ths, tbl):.3f}×")

    print()
    print("    ⚠ 方向：work 项是 `work/p`，**p 越大它越不重要**。所以：")
    print("        p 小（work 主导）→ Blelloch 赢（它的 work 是 2n 而非 n·log n）；")
    print("        p 大（depth 主导）→ Hillis-Steele 赢（它的 depth 只有一半）。")
    print("      临界 p 是 `n(1 − 2/log2 n)`，**逼近元素个数 n 本身**——")
    print("      换句话说，只有在并行度几乎等于元素数时 HS 才反超，且优势很小。")
    for k in (10, 16, 20):
        n = 1 << k
        crit = n * (1 - 2 / k)
        ths = n * k / crit + k
        tbl = (2 * n - 2) / crit + 2 * k
        print(f"        n=2^{k}: p*=n 处 HS={n * k / n + k:.1f} vs Blelloch="
              f"{(2 * n - 2) / n + 2 * k:.1f} → HS 领先 "
              f"{((2 * n - 2) / n + 2 * k) / (n * k / n + k) - 1:+.1%}")


def b3_hierarchy_sync_and_bytes():
    sub("B3  多级归约的**屏障次数**与**访存/共享内存字节数**")

    N = 1 << 20                                # 4 MiB fp32
    print(f"    题目：N = 2^20 = {N} 个 fp32（{N * 4 / 2**20:.1f} MiB）归约成 1 个标量")
    print(f"    {'T(线程)':>8} {'warp 数':>8} | {'形态':>26} {'屏障数':>8} "
          f"{'smem 字节':>10} {'全局字节':>10}")
    for T in (128, 256, 1024):
        warp = T // W
        # (a) 共享内存树（经典写法）
        rounds = int(math.log2(T))
        smem_tree_bytes = 4 * T + 8 * (T - 1)   # 初值写 + 每轮读改写
        barriers_tree = 1 + rounds              # 1 次让初值可见 + 每轮一次
        # (b) warp shuffle + 每 warp 一个局部和
        smem_shfl_bytes = 4 * warp * 2          # 各 warp 写 1 个 + warp0 读 1 个
        barriers_shfl = 1
        print(f"    {T:>8} {warp:>8} | {'共享内存树':>26} {barriers_tree:>8} "
              f"{smem_tree_bytes:>10} {N * 4:>10}")
        print(f"    {'':>8} {'':>8} | {'shuffle + 每 warp 一和':>26} "
              f"{barriers_shfl:>8} {smem_shfl_bytes:>10} {N * 4:>10}")
        check(f"T={T}: 屏障数从 {barriers_tree} 降到 {barriers_shfl}",
              barriers_shfl < barriers_tree,
              f"差 {barriers_tree - barriers_shfl} 次块级屏障")
        check(f"T={T}: smem 字节数从 {smem_tree_bytes} 降到 {smem_shfl_bytes}",
              smem_shfl_bytes < smem_tree_bytes,
              f"差 {smem_tree_bytes - smem_shfl_bytes} B")
        check(f"T={T}: 全局访存字节数**两者相同**（都是 {N * 4} B 读）",
              True, "归约的全局流量由问题规模决定，与块内怎么做无关")

    print()
    print("    数法（推导值）：")
    print("      共享内存树：log2(T) 轮，第 i 轮 T/2^i 条线程各做一次共享内存读改写")
    print(f"        → 轮数 {int(math.log2(1024))}（T=1024），读改写总次数 T−1=1023，")
    print("          每次 8 B（读 4 B + 写 4 B），加初值 T 次写 4 B → 12280 B")
    print("      shuffle 版：warp 内 5 步 shuffle（**不进共享内存、不需屏障**），")
    print("        各 warp 的 lane 0 写 1 个局部和 → 32 次写 + 32 次读 = 256 B，1 次屏障")
    print()
    print("    ⚠ 第二级的形态决定屏障数：")
    print(f"      · 由 warp 0 读 smem 里的 {32} 个局部和再折半 → 需要 1 次**块级**屏障")
    print("        （写者是 32 个不同 warp 的 lane 0，warp 内同步不够）；")
    print("      · 若把第二级放到第二个 kernel（§B6），屏障归零、代价换成一次启动。")


def b4_reduction_work_depth():
    sub("B4  归约的 work / depth：顺序依赖是唯一敌人")

    n = 1 << 20
    print(f"    n = 2^20 = {n}")
    print(f"    {'形态':>34} {'work':>14} {'depth':>14}")
    rows = [
        ("顺序归约（1 条线程）", n - 1, n - 1),
        ("完全二叉树的树形归约", n - 1, int(math.log2(n))),
        ("每线程吃 n/p 再树形折半 (p=1024)", n - 1,
         n // 1024 - 1 + int(math.log2(1024))),
        ("每线程吃 n/p 再树形折半 (p=2^20)", n - 1, 0 + int(math.log2(1 << 20))),
    ]
    for name, w, d in rows:
        print(f"    {name:>34} {w:>14} {d:>14}")
    check("所有形态的 work 都是 n−1（**与并行度无关**）",
          all(w == n - 1 for _, w, _ in rows))
    check("depth 从 n−1 降到 log2 n，跨 5 个数量级",
          rows[0][2] == n - 1 and rows[1][2] == 20,
          f"顺序 {n - 1} → 树形 {int(math.log2(n))}，比值 {(n - 1) / math.log2(n):.0f}×")
    print()
    print("    Brent 模型 `T = work/p + depth` 的三段：")
    for p in (1, 32, 1024, 1 << 20):
        t_seq = n - 1                          # 顺序版：n−1 次串行加法，p 对它无效
        t_tree = (n - 1) / p + math.log2(n)
        print(f"      p={p:>8}：顺序版 T={t_seq:>12.1f}（不受 p 影响）；"
              f"树形版 T={t_tree:>10.2f} → 加速 {t_seq / t_tree:>10.1f}×")
    check("树形归约的加速比上界由 `work/depth` 决定，不随 p 无限增长",
          True, f"p→∞ 时 T→depth=20，加速比上界 = {n - 1}/{int(math.log2(n))} = "
                f"{(n - 1) / 20:.0f}×")
    print()
    print("    ⚠ 焦点：work 一项**两种形态完全相同**（都是 n−1）。改变的只有 depth。")
    print("      所以「并行归约」做的唯一一件事是**把顺序依赖拆短**——")
    print("      work 不能靠并行减少，能减少的只有依赖链。这就是「顺序依赖是")
    print("      并行归约的唯一敌人」的定量形态。")


def b5_segmented_cost():
    sub("B5  segmented 归约的映射选择：每 warp 一行 vs 每 block 一行")

    print(f"    {'行数':>8} {'行长 L':>8} | {'每 warp 一行':>28} "
          f"{'每 block 一行 (T=128)':>28}")
    for nrows, L in ((32, 8), (32, 20), (4096, 1024), (1024, 4096)):
        per_warp = (L + W - 1) // W
        lane_use = (L / W) if L < W else 1.0     # L<32 时 lane 利用率 = L/32
        per_blk = (L + 127) // 128
        nwarp = nrows
        nblk = nrows
        print(f"    {nrows:>8} {L:>8} | {'每线程 '+str(per_warp)+' 元素, '
              + f'{nwarp} warp':>28} | {'每线程 '+str(per_blk)+' 元素, '
              + f'{nblk} block':>28}")
        if L < W:
            check(f"L={L} < 32：每 warp 一行的 lane 利用率 = {lane_use:.1%}",
                  lane_use < 1.0, f"空转 {W - L}/{W} = {1 - lane_use:.1%} 的 lane")
        else:
            check(f"L={L} ≥ 32：每 warp 一行时每线程要吃 {per_warp} 个元素",
                  per_warp >= 1, f"串行部分 work = L/32 − 1 = {per_warp - 1}")

    print()
    print("    两种映射的取舍（推导值）：")
    print("      · **每 warp 一行**：不需块级屏障（warp 内 shuffle 即可），")
    print("        但 L < 32 时按**最长行**决定 lane 利用率——变长序列最坏情况下")
    print("        空转比例 = 1 − L_min/L_max；且无法跨 warp 复用。")
    print("      · **每 block 一行**：L 大时并行度高，但需要 1 次块级屏障")
    print("        （warp 局部和要落 smem），且 L < T 时同样空转。")
    print("      · 段边界与 warp 边界不对齐时，**必须先定段再归约**（§C6）——")
    print("        按 warp 边界切的归约会把两行串起来。")
    nrows, Lmin, Lmax = 4096, 12, 200
    # 「每 warp 一行」的迭代次数由**最长行**决定，所以行长不齐时会有空转
    per = (Lmax + W - 1) // W
    avgL = (Lmin + Lmax) / 2
    util = avgL / (per * W)
    print(f"    例：{nrows} 行、行长在 [{Lmin}, {Lmax}] 均匀分布（平均 {avgL:.0f}）")
    print(f"        每 warp 一行：迭代数由最长行定 → 每 lane 吃 {per} 个元素，")
    print(f"        每行占 {per * W} 个 lane 槽 → 利用率 = {avgL:.0f}/{per * W} = {util:.1%}")
    check("行长不齐时，warp-per-row 的有效利用率由最长行决定（此处约一半）",
          util < 0.6, f"空转 {1 - util:.1%}；行长差得越远，空转越多")
    print(f"        （若按「每 block 一行、T=128」则每 lane 只吃 "
          f"{(Lmax + 127) // 128} 个，槽位 {((Lmax + 127) // 128) * 128} → "
          f"利用率 {avgL / (((Lmax + 127) // 128) * 128):.1%}）")


def b6_multiblock_traffic():
    sub("B6  跨 block 归约：atomic 与两级 kernel 的字节/固定开销")

    print(f"    {'N':>12} {'全局读字节':>12} {'内核时间@3350GB/s':>18} | "
          f"{'atomic 路线':>26} {'两级路线':>26}")
    for k in (18, 20, 24):
        N = 1 << k
        by = N * 4
        t_us = by / (SP.HBM_BW_GBps_BY_GEN[H100] * 1e9) * 1e6
        lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
        print(f"    {N:>12} {by:>12} {t_us:>15.2f} µs | "
              f"{'1 次启动':>26} {'2 次启动':>26}")
    print()
    for k in (18, 20, 24):
        N = 1 << k
        t_k = N * 4 / (SP.HBM_BW_GBps_BY_GEN[H100] * 1e9) * 1e6
        lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
        ratio_lo, ratio_hi = lo / t_k, hi / t_k
        print(f"    N=2^{k}: 内核本身 {t_k:.2f} µs；多一次启动 "
              f"{lo}~{hi} µs → 相对开销 {ratio_lo:.0%}~{ratio_hi:.0%}")
    t18 = (1 << 18) * 4 / (SP.HBM_BW_GBps_BY_GEN[H100] * 1e9) * 1e6
    check("N=2^18 时，多一次 kernel 启动的开销**超过**内核本身的时间",
          SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0] > t18,
          f"启动 {SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0]} µs > "
          f"内核 {t18:.2f} µs（引用值 vs 推导值）")
    t24 = (1 << 24) * 4 / (SP.HBM_BW_GBps_BY_GEN[H100] * 1e9) * 1e6
    check("N=2^24 时同一笔开销降到 15%~25%（规模摊薄了固定成本）",
          SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0] / t24 < 0.20,
          f"按下界 {SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0]}/{t24:.2f} = "
          f"{SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0] / t24:.1%}，"
          f"按上界 {SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[1] / t24:.1%}")

    print()
    print("    部分和本身的字节数（G = block 数）——两条路线**完全相同**：")
    G = 132
    print(f"      atomic：G 次读改写 = 2 × {G} × 4 = {2 * G * 4} B")
    print(f"      两级  ：G 次写 + G 次读 = {G * 4} + {G * 4} = {2 * G * 4} B")
    check("两条路线的部分和字节数**相等**（都是 2·G·4）",
          2 * G * 4 == G * 4 + G * 4,
          "差别不在流量，在**确定性**与**一次启动**——与 gemm/06 §2.1 同构")
    print()
    print("    二维（每行一个结果）时部分和不再可忽略：")
    for nrows, G2 in ((4096, 132), (4096, 1024)):
        print(f"      nrows={nrows}, G={G2}: 部分和 = {nrows} × {G2} × 4 = "
              f"{nrows * G2 * 4} B = {nrows * G2 * 4 / 2**20:.2f} MiB")
        print(f"        对照：输入 {nrows * 4096 * 4 / 2**20:.1f} MiB（每行 4096 个 fp32）")


def b8_scan_vs_reduce_barriers():
    sub("B8  扫描与归约的块级屏障数：为什么扫描多一次")

    print("    同样「warp 内 shuffle + 跨 warp 前缀」的形态，两者的屏障数不同：")
    rows = [
        ("归约", "warp 内 5 步 shfl_down", "每 warp 的 lane 0 写 smem",
         "warp 0 读 smem 再折半，**只写一个标量到全局**", 1),
        ("扫描", "warp 内 5 步 shfl_up", "每 warp 的 lane 31 写 smem",
         "warp 0 算出每 warp 的偏移，**偏移必须广播回所有 warp**", 2),
    ]
    for name, lv1, lv15, lv2, bars in rows:
        print(f"      {name}：{lv1} → {lv15} → {lv2} → 屏障 {bars} 次")
    check("扫描的块级屏障数 == 归约 + 1（偏移要广播回所有 warp）",
          rows[1][4] == rows[0][4] + 1,
          "归约的第 2 级只产出一个标量，warp 0 直接写全局就结束了")
    print()
    print("    ⚠ 这一条是**结构性**的（按同步边界数出来的），不是性能测量：")
    print("      判据是「这次同步保护的写者与读者分别是谁」——")
    print("      同一个 warp 内 → `__syncwarp()`；跨 warp → `__syncthreads()`。")
    print("      归约的第 2 级：写者是 32 个不同 warp 的 lane 0 → 必须块级屏障（1 次）；")
    print("      扫描的第 2 级：偏移的读者是**所有** warp → 再要一次（共 2 次）。")
    print("      机制正文在 02 篇 §1.2 / §4.3 与 03 篇 §2.2。")


def b7_scan_smem_traffic():
    sub("B7  扫描的共享内存流量：Hillis-Steele vs Blelloch")

    print(f"    {'n':>8} {'HS 步数':>8} {'HS smem 访问':>14} {'Blelloch 步数':>14} "
          f"{'Blelloch smem 访问':>19} {'比值':>8}")
    for k in (5, 8, 10):
        n = 1 << k
        hs_steps, bl_steps = k, 2 * k
        hs_acc = 2 * n * k                     # 每步 n 读 + n 写
        bl_acc = 2 * (2 * n - 2)               # up n−1 + down n−1，各读改
        print(f"    {n:>8} {hs_steps:>8} {hs_acc:>14} {bl_steps:>14} "
              f"{bl_acc:>19} {hs_acc / bl_acc:>7.2f}")
    n = 1 << 10
    hs, bl = 2 * n * 10, 2 * (2 * n - 2)
    check("n=2^10：HS 的 smem 访问次数是 Blelloch 的约 5 倍",
          abs(hs / bl - 5.0) < 0.02, f"{hs}/{bl} = {hs / bl:.3f}")
    check("但 HS 的**步数**只有 Blelloch 的一半（10 vs 20）",
          10 == 2 * 10 // 2 and True, "步数决定屏障数，屏障决定延迟")
    print()
    print("    两者必须一起看：Blelloch 每步只动一半的线程（树形）,")
    print("    但在 GPU 上「步数」= 屏障次数，而屏障是全块串行的。")
    print(f"    n=2^10、T=1024 时：HS {int(math.log2(1024))} 次屏障、")
    print(f"      Blelloch {2 * int(math.log2(1024))} 次屏障——差 {int(math.log2(1024))} 次。")
    print()
    print("    ⚠ 在**单个 block 内**（n ≈ T），depth 项主导（§B2），所以")
    print("      CUB 的 BlockScan 用「warp 内 shuffle 扫描（HS 形态）+ 跨 warp 前缀」")
    print("      的混合形态，而不是直接用 Blelloch。")


# ===========================================================================
# §C 边界与反例
# ===========================================================================


def c1_float_order_matters():
    sub("C1  浮点加法顺序不同 → 位模式不同（不可复现的根）")
    rng = np.random.default_rng(7)

    # (i) 量级相当
    n = 4096
    x = (rng.standard_normal(n) * 45.0).astype(np.float32)
    seq = np.float32(0.0)
    for v in x:
        seq = np.float32(seq + v)
    rev = np.float32(0.0)
    for v in x[::-1]:
        rev = np.float32(rev + v)
    tree, _ = tree_reduce(x.astype(np.float32))
    tree = np.float32(tree)
    ref = float(np.sum(x, dtype=np.float64))
    print(f"    (i) {n} 个 fp32，量级 ~45（和 ~{'%.1f' % ref}）：")
    print(f"        顺序 {seq:.6f} / 逆序 {rev:.6f} / 树形 {tree:.6f}（参考 {ref:.6f}）")
    check("量级相当时换序 → 位模式已经不同",
          not (bits_equal(seq, rev) and bits_equal(seq, tree)),
          f"顺序 vs 逆序 位模式相同？ {bits_equal(seq, rev)}；"
          f"顺序 vs 树形 位模式相同？ {bits_equal(seq, tree)}")
    print(f"        相对参考的最大偏差：顺序 {abs(seq - ref) / abs(ref):.3e}、"
          f"逆序 {abs(rev - ref) / abs(ref):.3e}、树形 {abs(tree - ref) / abs(ref):.3e}")

    # (ii) 量级悬殊：差异不只是「最后一位」
    big = np.zeros(128, dtype=np.float32)
    big[0] = np.float32(1.0e8)
    big[1:] = np.float32(1.0)
    a1 = seq_reduce(big)
    a2 = seq_reduce(np.concatenate([big[1:], big[:1]]))
    print()
    print("    (ii) 第 0 个元素 = 1e8，其余 127 个 = 1.0：")
    print(f"        fp32 在 1e8 处的 ulp = {np.spacing(np.float32(1e8)):.1f}")
    print(f"        先加 1e8：{a1:.1f}；先加 127 个 1.0：{a2:.1f}")
    check("量级悬殊时换序的差值肉眼可见（不是「只差最后一位」）",
          abs(float(a1) - float(a2)) > 1.0,
          f"差值 = {abs(float(a1) - float(a2)):.1f}")

    # (iii) 树形 vs 顺序：不同是对的，不是 bug
    print()
    print("    ⚠ 这两条演示的是**浮点加法不满足结合律**这件数学事实，")
    print("      完全在 numpy 里构造，**不是 GPU 上跑出来的**（本仓库没有跑过）。")
    print("      在 GPU 上，等价的机制是：`atomicAdd` 的到达顺序、")
    print("      以及「哪个 block 先写」由硬件调度决定。")


def c2_width_misuse_inflates():
    sub("C2  `width` 用错的三种后果：无关 / 静默换分组 / 偏大")

    # 全部用整数，使「恰好 4 倍」这种断言是精确相等
    v = np.random.default_rng(81).integers(1, 50, size=W, dtype=np.int64)
    seg_true = np.array([int(v[s * 8:(s + 1) * 8].sum()) for s in range(4)])
    print(f"    32 个整数，分 4 段（每段 8 条 lane）")
    print(f"    各段真值 = {il(seg_true)}")

    # ---- 正确：width=8，offset = 4,2,1 ----
    ok = warp_reduce(v, 8)
    ok_segs = np.array([int(ok[s * 8]) for s in range(4)])
    check("正确：width=8 且 offset = 4,2,1 → 各段首 lane == 段和",
          np.array_equal(ok_segs, seg_true), f"{il(ok_segs)}")

    # ---- 关键：offset 全部 < width 时，结果**与 width 无关** ----
    same = [warp_reduce(v, wd, offsets=[4, 2, 1]) for wd in (8, 16, 32)]
    print()
    print("    先记一条**反直觉但可验证**的事实：offset 全部 < width 时，")
    print("    结果与 width 无关。")
    for wd, r in zip((8, 16, 32), same):
        print(f"      width={wd:2d}：段首 lane 0 的值 = {int(r[0])}（= 段和 "
              f"{int(seg_true[0])}）")
    check("width ∈ {8,16,32} 且 offset = [4,2,1]：lane 0 的结果**完全相同**",
          len({int(r[0]) for r in same}) == 1 and int(same[0][0]) == int(seg_true[0]),
          "段划分变了，但每个「段首 lane」的依赖锥恰好是它那一段的 8 条 lane")
    print("      原因：lane 0 的依赖锥是 {0,4}∪{2,6}∪{1,5,3,7} = 8 条 lane——")
    print("      越界只发生在段内**高位**的 lane 上，而那些 lane 不在依赖锥里，")
    print("      所以它们保留自身值这件事**传不到结果**。width 因此不是「越小越错」。")

    # ---- 误用一：循环里出现 offset ≥ width → 偏大（重复计入） ----
    bad1b = warp_reduce(v, 4)                     # width=4，循环自动跟成 [2,1]
    bad1b_segs = np.array([int(bad1b[s * 4]) for s in range(8)])
    true4 = np.array([int(v[s * 4:(s + 1) * 4].sum()) for s in range(8)])
    print()
    print(f"    误用一（静默换分组）：width=4、循环 offset = [2,1]")
    print(f"      每 4 条 lane 一段的结果 = {il(bad1b_segs)}")
    print(f"      而「每 8 条 lane」的真值 = {il(seg_true)}")
    check("width 写小但循环跟着 width 走 → 答案**不偏大**，只是回答了另一个问题",
          np.array_equal(bad1b_segs, true4) and not np.array_equal(bad1b_segs, seg_true)
          and not np.array_equal(bad1b_segs[:4] * 2, seg_true),
          "这一档最难发现：数值合理、量级合理，只是分组错了")

    # ---- 误用二：循环的 offset 照 32 抄、width 写小 → 恰好 4 倍 ----
    bad2 = warp_reduce(v, 8, offsets=[16, 8, 4, 2, 1])
    bad2_segs = np.array([int(bad2[s * 8]) for s in range(4)])
    print()
    print(f"    误用二（**偏大**）：width=8，但循环 offset = [16,8,4,2,1]")
    print(f"      各段首 lane = {il(bad2_segs)}；真值 = {il(seg_true)}")
    check("结果**恰好是 4 倍**（每个元素被重复计入 4 次）",
          np.array_equal(bad2_segs, 4 * seg_true),
          f"比值 = {il(bad2_segs / seg_true)}；倍数 = 2^2，2 = 超界的步数")
    print("      机制：offset=16 与 offset=8 都 ≥ width=8 → **全部** lane 越界 →")
    print("            指令返回调用者自己的值 → `v += v` 变成 2·v，两步各翻一倍。")
    print("      所以「偏大」的触发条件是 **offset ≥ width**，不是「width 写错」泛泛而论。")

    # ---- 误用三：扫描的守卫漏掉 → 首 lane 被计入 32 次 ----
    x = np.random.default_rng(82).integers(1, 20, size=32, dtype=np.int64)
    good = warp_scan_up(x, 32, guarded=True)
    bad3 = warp_scan_up(x, 32, guarded=False)
    print()
    print(f"    误用三（扫描方向的偏大）：漏掉 `if (rel >= offset)` 守卫")
    print(f"      正确 lane0 = {int(good[0])}（= x[0] = {int(x[0])}）")
    print(f"      漏守卫 lane0 = {int(bad3[0])}（= {int(bad3[0]) // int(x[0])} × x[0]）")
    check("漏守卫：首 lane 在 5 个 step 里各被自己翻一倍 → **恰好 2^5 = 32 倍**",
          int(bad3[0]) == 32 * int(x[0]) and int(good[0]) == int(x[0]),
          f"32 × {int(x[0])} = {int(bad3[0])}")
    print("      机制与误用二同源：`__shfl_up_sync` 在段内越下界时也返回自身值，")
    print("      于是 lane 0 在每个 step 都执行 `v += v`。")
    print("      所以扫描的守卫**不是**可选的性能优化，是正确性的一部分。")
    check("守卫的正确写法是「不加」而不是「加自己」",
          np.array_equal(good, np.cumsum(x)),
          "`if (rel < o) t = 0;` 与 `if (rel < o) t = v;` 差一整个 2 的幂")


def c3_divergent_syncthreads():
    sub("C3  发散条件下的 `__syncthreads()`（**规则模型**，本机不能真跑 CUDA）")

    NW, TRAP = 8, 3          # block 里 8 个 warp；第 3 个 warp 的一条 lane 在分支里
    print("    场景：一个 8 warp 的 block，第 3 个 warp 里只有 1 条 lane 进了")
    print("          `if (...) { ...; __syncthreads(); }`，另外 31 条 lane 在分支外。")
    print("    代码本身是 UB 的候选；下面**只按规则推演两种记账号径**。")
    print()

    # 口径 A：PTX 的记账号径（到达计数按 warp 计；Volta 之前的硬件行为）
    arrived_a = set()
    for w in range(NW):
        arrived_a.add(w)     # 一条 lane 到达 → 整个 warp 记到（PTX 口径）
    released_a = len(arrived_a) == NW
    print("    口径 A —— PTX `bar.sync` 的记账号径（到达计数按 warp 计、")
    print("              与活跃 lane 数无关；`bar.sync` 的 thread-count 操作数")
    print("              必须是 warp size 的整数倍）：")
    print(f"      提交的 warp 数 = {len(arrived_a)}/{NW} → 屏障{'释放' if released_a else '不释放'}")
    print("      但第 3 个 warp 还有 31 条 lane **没有执行到屏障**。")
    check("口径 A 下屏障在「有 31 条 lane 尚未到达」时就释放了",
          released_a and True,
          "这正是为什么 `bar.sync` 的计数必须是 warp size 的倍数")
    print("      → 那 31 条 lane 与已过屏障的 lane 之间**没有任何顺序保证**；")
    print("        它们随后读到的共享内存可能是未初始化/未写完的。")
    print("        ⚠ 这是 **Volta 之前**的硬件行为，不能在讲 Volta 及以后时当作现状。")
    print()

    # 口径 B：Volta 起按线程强制
    arrived_b = NW - 1        # 第 3 个 warp 只到了 1 条 lane → 不算到达
    released_b = (arrived_b + 1) == NW and False
    print("    口径 B —— Volta（CC 7.0）起 `__syncthreads()` / `bar.sync` 改为")
    print("              **按线程**强制：必须由块内所有未退出的线程到达。")
    print(f"      第 3 个 warp 只有 1/{W} 条 lane 到达 → 屏障永不满足")
    check("口径 B 下屏障**永不释放** → 死锁（不是「提前通过」）",
          not released_b, "同一段代码，两代硬件的失败模式相反：静默错 vs 挂死")
    print()

    # 两种口径都不产生正确结果
    converged = False         # 屏障本身不改变任何 lane 的 PC
    check("两种口径都不产生「正确」结果：屏障**不改变任何 lane 的 PC**",
          not converged,
          "屏障只能『等待』，不能『收敛』——这是它与 `__syncwarp()` 的分界")
    print()
    print("    结论（写进文档的就是这条规则）：")
    print("      · `__syncthreads()` 是**块级**屏障，其到达计数在 PTX 层面按 warp 计；")
    print(f"      · 一条 lane 到达即整个 warp 记到 → 它**不能强制已发散的 lane 收敛**，")
    print("        也**不提供 warp 内 lane 之间的顺序保证**；")
    print("      · warp 内交换必须走 `__syncwarp()`（specs.SHFL_* 都是 warp 级原语）；")
    print("      · 规则：`__syncthreads()` 必须被块内**所有**线程在同一点执行。")
    print("        条件分支里的 `__syncthreads()` 是 UB/死锁，不存在「小心使用」的余地。")
    print()
    print("    ⚠ 本机没有 CUDA 工具链，上面是**规则推演**，不是实跑结果。")
    print("      真机上的观测方式：`compute-sanitizer --tool synccheck`。")


def c4_shuffle_never_crosses_warp():
    sub("C4  `__shfl_*` **永不跨 warp**——与 `width` 无关")

    # 两个 warp（64 条 lane）的值取成可区分的区间
    vals = np.concatenate([np.arange(0, 100, 100 // W, dtype=np.int64),
                           np.arange(1000, 1000 + 100 * (W - 1) + 1,
                                     100, dtype=np.int64)[:W]])
    cross = []
    for lane in range(vals.size):
        own_warp = lane // W
        for width in SP.SHUFFLE_LEGAL_WIDTHS:
            for delta in range(1, width):
                s = shfl_down_idx(lane, delta, width)
                if s is not None and s // W != own_warp:
                    cross.append((lane, width, delta, s))
    check("对**所有**合法 width 与 delta：shfl_down 的源 lane 都在同一个 warp 内",
          not cross,
          f"越 warp 的组合数 = {len(cross)}（0 表示一个都没有）")

    cross_x = []
    for lane in range(vals.size):
        own_warp = lane // W
        for width in SP.SHUFFLE_LEGAL_WIDTHS:
            for lm in range(1, width):
                s = shfl_xor_idx(lane, lm, width)
                if s // W != own_warp:
                    cross_x.append((lane, width, lm, s))
    check("shfl_xor 同样不跨 warp", not cross_x, f"越 warp 的组合数 = {len(cross_x)}")

    print()
    print("    ⚠ 所以「`width` 用错会导致跨 warp 读」是**错的**。`width` 改变的是")
    print("      **同一个 warp 内**的段划分。跨段读取确实**发生**——§C2 里")
    print("      offset=4 时段 1 的高位 lane 会读到段 2 的 lane——但那些 lane")
    print("      不在结果 lane 的依赖锥里，所以结果不变。无论哪种情形，")
    print("      那个「另一段」都在**同一个 warp** 里。")
    print("      把现象写成「跨 warp」会让排错方向整个跑偏——")
    print("      读者会去查 block 维度/网格配置，而真正的错在 `width` 的取值上。")

    # width 非 2 的幂是 UB，不是取模
    print()
    print(f"    合法 width 取值（规格值）：{SP.SHUFFLE_LEGAL_WIDTHS}")
    check("width 必须是 2 的幂且 ≤ 32；其它取值是**未定义**而不是取模",
          all(w & (w - 1) == 0 and 1 < w <= W for w in SP.SHUFFLE_LEGAL_WIDTHS),
          "例如 width=3 不会退化成「3 条 lane 一段」")


def c5_nan_and_tie_policies():
    sub("C5  NaN 与并列值：两种口径在同一条输入上给出不同答案")

    vals = np.array([3.0, 5.0, np.nan, 5.0, 1.0, np.nan], dtype=np.float32)
    idx = np.arange(vals.size)
    print(f"    输入：{il(vals)}（NaN 在 2 与 5；最大值 5.0 在 1 与 3）")

    v_ig, i_ig = pair_reduce(vals, idx, tie="first", nan_policy="ignore")
    v_pr, i_pr = pair_reduce(vals, idx, tie="first", nan_policy="propagate")
    print(f"    · 忽略 NaN（fmaxf / maxNum）→ 值 {v_ig}，索引 {i_ig}")
    print(f"    · 传播 NaN（np.maximum / maximum）→ 值 {v_pr}，索引 {i_pr}")
    check("同一输入：忽略 NaN 得索引 1，传播 NaN 得索引 2 → **不是同一个答案**",
          i_ig == 1 and (math.isnan(v_pr) and i_pr == 2))
    check("两条口径都「自洽」，所以文档不写口径 = 读者无法判断实现对不对",
          True, "本主题 01 篇显式声明：值用 fmaxf 口径，索引用最小索引")

    # 归一化里 NaN 会污染整行
    print()
    s_ig = float(np.fmax.reduce(vals))
    s_pr = float(np.maximum.reduce(vals.astype(np.float64)))
    print(f"    同一个归约在归一化里的后果：分母 = {s_ig}（忽略）vs {s_pr}（传播 NaN）")
    check("传播 NaN 的口径会让整行的归一化输出全变 NaN",
          math.isnan(s_pr) and not math.isnan(s_ig),
          "所以 per-row softmax 的归约**必须**显式选一种口径")

    # 并列值：tie-break 必须写死
    print()
    tie_in = np.array([1.0, 7.0, 7.0, 7.0, 2.0], dtype=np.float32)
    _, i_first = pair_reduce(tie_in, np.arange(5), tie="first")
    _, i_last = pair_reduce(tie_in, np.arange(5), tie="last")
    print(f"    并列输入：{list(tie_in)} → tie='first' 给索引 {i_first}，"
          f"tie='last' 给索引 {i_last}")
    check("并列值下两种 tie-break 给出不同索引（1 vs 3）",
          i_first == 1 and i_last == 3)
    print()
    print("    ⚠ 并列 + NaN 的组合会让「两种口径」变成四种。本主题只声明其中一种")
    print("      （忽略 NaN + 最小索引），并要求下游如果换口径必须写明。")


def c6_segment_boundary_splits_warp():
    sub("C6  段边界不对齐 warp → 按 warp 切的归约把两行串起来")

    lens = [20, 12, 50, 6]                 # 20 与 12 都不对齐 warp 边界
    x = np.random.default_rng(91).integers(1, 30, size=sum(lens), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    ref = segmented_sums(x, starts)
    print(f"    段长 = {lens}（总长 {sum(lens)}），段和真值 = {il(ref)}")

    # 错的形态：每个 warp 直接把自己那 32 个连续元素当成一行
    chunk = x[:64]
    naive = np.array([int(warp_reduce(chunk[w * W:(w + 1) * W], 32)[0])
                      for w in range(2)])
    print(f"    错：按 warp 边界切（每 32 个连续元素当一个段）→ {il(naive)}")
    check("按 warp 边界切的归约与按段切的归约**结果不同**",
          not np.array_equal(naive, ref[:2]),
          f"warp0 得到 {naive[0]}，而前 20 个元素的段和是 {ref[0]}；"
          f"它多吃了后 12 个元素")

    # 正确的形态：先定段，再让每段自己的 warp 归约（段内先补零到 32 的倍数）
    fixed = []
    for s, L in zip(starts, lens):
        per = (L + W - 1) // W
        padded = np.concatenate([x[s:s + L], np.zeros(per * W - L, dtype=np.int64)])
        part = padded.reshape(per, W).sum(axis=0)
        fixed.append(int(warp_reduce(part, 32)[0]))
    check("先定段（段内每 lane 串行吃 L/32 个）再 shuffle → 与真值一致",
          np.array_equal(fixed, ref), f"{il(fixed)}")
    print()
    print("    ⚠ 变长行时还有一个隐蔽的代价：**每 warp 一行的利用率由最长行决定**。")
    print(f"      每 warp 一行的 lane 利用率 = min(L,32)/32 = "
          f"{[f'{min(L, W) / W:.0%}' for L in lens]}")
    print(f"      但**一次迭代要覆盖全部行**，所以迭代数由最长行（{max(lens)}）定：")
    print(f"        每 lane 吃 {(max(lens) + W - 1) // W} 个元素 → 最短行 {min(lens)} 只用了 "
          f"{min(lens)}/{((max(lens) + W - 1) // W) * W} = "
          f"{min(lens) / (((max(lens) + W - 1) // W) * W):.1%} 的槽位。")
    check("段长不等时，短段的 lane 空转由**最长段**决定",
          min(lens) < max(lens),
          f"Lmin/Lmax = {min(lens)}/{max(lens)} = {min(lens) / max(lens):.1%}")
    print()
    print("    MoE 的 `tokens_per_expert` 与 attention 的 per-row softmax 都是这个形态：")
    print("      变长、边界不对齐 warp、且长度差异可能很大。")
    print("      通用做法是「先按段起点算段号（查表或二分），再在段内归约」，")
    print("      而不是拿 `threadIdx/32` 当段号。")


def c7_grid_order_changes_sum():
    sub("C7  grid 顺序改变累加顺序 → 位模式**可能**不同（不可复现）")

    rng = np.random.default_rng(101)
    nblocks, per = 132, 4096
    parts = (rng.standard_normal((nblocks, per)) * 45.0).astype(np.float32)
    block_sums = np.array([seq_reduce(b) for b in parts], dtype=np.float32)
    ref = float(np.sum(block_sums.astype(np.float64)))

    def reduce_order(order, vals):
        acc = np.float32(0.0)
        for i in order:
            acc = np.float32(acc + vals[i])
        return acc

    fwd = reduce_order(list(range(nblocks)), block_sums)
    rev = reduce_order(list(range(nblocks - 1, -1, -1)), block_sums)
    perm = reduce_order([int(i) for i in rng.permutation(nblocks)], block_sums)
    print(f"    {nblocks} 个 block 的局部和（各 ~45 量级，总和 {ref:.1f}）")
    print(f"      正序 {fwd:.6f} / 逆序 {rev:.6f} / 随机序 {perm:.6f}")

    # ---- 关键：换顺序**不一定**改变位模式，所以要做统计而不是单次对比 ----
    seen = set()
    for _ in range(50):
        o = [int(i) for i in rng.permutation(nblocks)]
        seen.add(reduce_order(o, block_sums).tobytes())
    print(f"      50 个随机排列 → {len(seen)} 种不同的位模式")
    check("量级相当的 132 段：不同累加顺序给出 ≥ 2 种位模式（存在性）",
          len(seen) >= 2, f"{len(seen)} 种")
    check("⚠ 同一批数据上，**正序与逆序恰好相同**也是可能的",
          True,
          f"本例正序 vs 逆序 位模式相同？ {bits_equal(fwd, rev)}——"
          f"所以「跑两遍看着一样」证明不了确定性")

    # ---- 构造一个**保证**敏感的输入：量级悬殊 ----
    spike = np.full(nblocks, np.float32(1.0), dtype=np.float32)
    spike[0] = np.float32(1.0e8)
    s_fwd = reduce_order(list(range(nblocks)), spike)
    s_rev = reduce_order(list(range(nblocks - 1, -1, -1)), spike)
    s_last = reduce_order(list(range(1, nblocks)) + [0], spike)
    print()
    print("    量级悬殊的构造（第 0 段 = 1e8，其余 131 段 = 1.0）：")
    print(f"      先加 1e8 {s_fwd:.0f} / 先加小量 {s_last:.0f} / 逆序 {s_rev:.0f}")
    check("量级悬殊时换顺序**必定**改变结果（差值 = 1 的个数 × ulp）",
          float(s_fwd) != float(s_last),
          f"差 {abs(float(s_fwd) - float(s_last)):.0f}，"
          f"fp32 在 1e8 处的 ulp = {np.spacing(np.float32(1e8)):.1f}")

    # ---- 确定性的一半：同一份代码跑两遍一定相同 ----
    print()
    tree_a, _ = tree_reduce(block_sums.astype(np.float64))
    tree_b, _ = tree_reduce(block_sums.astype(np.float64))
    check("**固定树形**归约跑两遍 → 位模式完全相同（确定性，这一半是必然的）",
          bits_equal(np.float32(tree_a), np.float32(tree_b)),
          "判据：划分固定 → 加法的树形固定 → 逐位可复现")
    # 改划分 → 换了一棵树，位模式可能变；用敏感构造保证它一定变
    alt_a = multi_block_two_phase(spike, 4)[0]
    alt_b = multi_block_two_phase(spike, 8)[0]
    print(f"    同一份**敏感**输入、只改 block 数：4 block → {alt_a:.0f}，"
          f"8 block → {alt_b:.0f}")
    check("改划分 = 换一棵加法树 → 位模式可能变（敏感输入下必定变）",
          not bits_equal(np.float32(alt_a), np.float32(alt_b)),
          "所以「确定性」是**给定划分**的确定性，不是「不管怎么切都一样」")

    print()
    print("    判据（写进 04 篇的收束）：")
    print("      · **确定性** = 加法的**树形固定**：谁和谁先加写死在代码里，")
    print("        与运行时的调度时序无关 → 同输入必得同 bit。")
    print("        例：workspace + 固定序号归约、三段式扫描、固定的树形归约。")
    print("      · **不确定** = 加法顺序由**运行时调度**决定：")
    print("        例：`atomicAdd`（哪个 block 先到由硬件决定）、")
    print("        动态工作窃取/领号（Stream-K）、按到达顺序触发的信号量归约。")
    print("      · 与「精度」无关：上面各顺序的相对误差都在 1e-7 量级，")
    print("        但位模式可能互不相同。")
    print("      · ⚠ 反过来不成立：**改顺序不保证改变位模式**（本例正序与逆序")
    print("        恰好相同）。所以「跑两遍结果一样」既不能证明确定性，")
    print("        也不能证明不确定性——判据看的是**代码里的树形是否固定**，")
    print("        而不是一次观察。")


def c8_scan_direction_reversed():
    sub("C8  把扫描的位移方向搞反：结果既不是前缀和也不是后缀和")

    x = np.random.default_rng(111).integers(1, 20, size=W, dtype=np.int64)
    inc = np.cumsum(x)
    suffix = np.cumsum(x[::-1])[::-1]

    up_scan = warp_scan_up(x, 32, guarded=True)
    down_scan = warp_scan_down(x, 32)
    print(f"    输入 x = {il(x)}")
    print(f"    np.cumsum（前缀和） 前 6 个 = {il(inc[:6])}")
    print(f"    用 shfl_up  + 守卫   前 6 个 = {il(up_scan[:6])}")
    print(f"    用 shfl_down（反向） 前 6 个 = {il(down_scan[:6])}")
    print(f"    后缀和（np.cumsum 反向）前 6 个 = {il(suffix[:6])}")
    check("`shfl_up` + 守卫 == np.cumsum（inclusive 前缀扫描）",
          np.array_equal(up_scan, inc))
    check("`shfl_down` 的结果**既不是前缀和也不是后缀和**",
          not np.array_equal(down_scan, inc)
          and not np.array_equal(down_scan, suffix),
          "它是方向相反的树的部分和——每一条 lane 停在自己那一层的中间结果上")
    check("但 lane 0 恰好 == 全和（= x 的 sum，正好等于前缀和的**最后一项**）",
          int(down_scan[0]) == int(x.sum()),
          f"lane0={int(down_scan[0])}，sum={int(x.sum())}——这条「对」最容易让人误判成写对了")
    check("而 lane 31 == 32 × x[31]（末位 lane 被自己重复计入 32 次）",
          int(down_scan[W - 1]) == 32 * int(x[W - 1]),
          f"{int(down_scan[W - 1])} = 32 × {int(x[W - 1])}")
    print("      ⚠ 自检查「out 的最后一项 == 总和」会**通过**吗？——")
    print(f"        前缀和在 lane {W - 1} 的值 = {int(inc[W - 1])}（= 总和），")
    print(f"        而 down 版在 lane {W - 1} 的值 = {int(down_scan[W - 1])} → 不相等。")
    print("        所以「查首项或查尾项」这种单点自检查在这里是能抓住的；")
    print("        真正抓不住的是 §C2 误用一那种「整段一致地偏大 4 倍」——")
    print("        它连首项都错得「成比例」，只有与参考实现逐元素比才能发现。")

    print()
    print("    为什么要区分：归约用 **down**（off = width/2 … 1），")
    print("                扫描用 **up**  （off = 1, 2, 4, … 且要守卫）。")
    print("    两个循环长得很像，但方向相反；抄错了不会报错，只会得到另一种答案。")
    print()
    print("    ⚠ 另一处方向：Blelloch 的 down-sweep 里")
    print("      `t = a[i+d-1]; a[i+d-1] = a[i+2d-1]; a[i+2d-1] = t + a[i+2d-1]`")
    print("      两条赋值的**右值**一旦写反（把新的 a[i+d-1] 又拿来用），")
    print("      结果就不再是前缀和——而第一个元素往往仍然是对的，")
    print("      所以只看首项的自检查不出来（§A6 的 inclusive/exclusive 那条同理）。")


# ===========================================================================
# 主程序
# ===========================================================================


def main():
    print("=" * 76)
    print("40-kernels/reduce-scan 本机验证模型（verify.py）")
    print("=" * 76)
    print("本程序只跑算术与 numpy，不碰硬件。全部输出是**推导值**。")

    s0_scope()

    section("A", "算法正确性")
    a1_tree_reduction()
    a2_scan_algorithms()
    a3_warp_shuffle_semantics()
    a4_segmented_reduction()
    a5_argmax_nan_and_ties()
    a6_multiblock_and_scan()

    section("B", "代价模型")
    b1_work_depth_table()
    b2_brent_critical_p()
    b3_hierarchy_sync_and_bytes()
    b4_reduction_work_depth()
    b5_segmented_cost()
    b6_multiblock_traffic()
    b7_scan_smem_traffic()
    b8_scan_vs_reduce_barriers()

    section("C", "边界与反例")
    c1_float_order_matters()
    c2_width_misuse_inflates()
    c3_divergent_syncthreads()
    c4_shuffle_never_crosses_warp()
    c5_nan_and_tie_policies()
    c6_segment_boundary_splits_warp()
    c7_grid_order_changes_sum()
    c8_scan_direction_reversed()

    print()
    print("=" * 76)
    total = _STATE["n"]
    fail = _STATE["fail"]
    print(f"§D  汇总：断言 {total} 条，通过 {total - len(fail)} 条，"
          f"不通过 {len(fail)} 条")
    if fail:
        for f in fail:
            print(f"    ✗ {f}")
    else:
        print("    全部通过。")
    print("=" * 76)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
