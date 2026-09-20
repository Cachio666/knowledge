# -*- coding: utf-8 -*-
"""独立参照实现。

★ 这个文件是 A 类断言的**可证伪性来源**。硬要求（沿用本仓库 CONTRIBUTING §2）：

  「A 类必须有一条**与被检验代码路径不同**的参照——不得用被测代码自己的输出
    当参照，也不得直接调用被测函数然后相信它的返回值。」

  为满足它，本文件里**没有一个函数调用 minitriton**，也**不 import 它**；
  全部用纯 Python 标量循环 + 标准库 math 写，故意不向量化，
  使它与 numpy 向量化实现、与三相归约模拟是**三条互不重叠的代码路径**。

  ★ 数值口径也不同：参照一律用 float64 累加（`math.fsum` 之外的显式加法），
    被测侧是 float32。因此比对必须给容差，容差按 f32 的机器 epsilon 定，
    不按「跑出来差不多」定。
"""

import math


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

def softmax_rows_ref(x):
    """对每一行做三段式 softmax：先求行最大、再求指数和、再归一。

    `x` 是 [[...]] 的 Python 列表。返回逐元素的 Python 列表。
    全部用 float64 标量运算。
    """
    out = []
    for row in x:
        m = float(row[0])
        for v in row:
            if float(v) > m:
                m = float(v)
        es = []
        s = 0.0
        for v in row:
            e = math.exp(float(v) - m)
            es.append(e)
            s += e
        out.append([e / s for e in es])
    return out


def softmax_rows_ref_shifted(x, c_index=0):
    """同一个问题的**另一种**参考式：用行内第 c_index 个元素当平移量。

    数学上 softmax(x) 对任意常数 c 满足 softmax(x)_i = exp(x_i-c)/sum_j exp(x_j-c)，
    所以结果应当与上面那段（用行最大当 c）一致到数值精度。
    这条参照**专门用来抓「最大值减错了 / 忘了减」这类错**：
    它根本不求最大，走的是完全不同的分支。
    """
    out = []
    for row in x:
        c = float(row[c_index])
        es = [math.exp(float(v) - c) for v in row]
        s = 0.0
        for e in es:
            s += e
        out.append([e / s for e in es])
    return out


def softmax_rows_stream_ref(x, block_n):
    """在线 softmax 的参照：**按块流式**跑，同样的递推但用标量循环写。

    递推（本文件独立写一遍，只作为参照）：
        m_new = max(m_old, max(blk))
        s_new = s_old * exp(m_old - m_new) + sum(exp(blk - m_new))
    返回逐元素的 Python 列表。
    """
    out = []
    for row in x:
        m = -float("inf")
        s = 0.0
        n = len(row)
        for b0 in range(0, n, block_n):
            blk = [float(v) for v in row[b0:b0 + block_n]]
            mb = max(blk)
            m_new = mb if mb > m else m
            if m == -float("inf"):
                s = s * 0.0
                s = 0.0
            s = s * math.exp(m - m_new) if m != -float("inf") else 0.0
            acc = 0.0
            for v in blk:
                acc += math.exp(v - m_new)
            s += acc
            m = m_new
        out.append([math.exp(float(v) - m) / s for v in row])
    return out


def row_max_ref(x):
    return [max(float(v) for v in row) for row in x]


def row_sum_ref(x):
    return [math.fsum(float(v) for v in row) for row in x]


# ---------------------------------------------------------------------------
# 掩码 / 越界
# ---------------------------------------------------------------------------

def masked_block_ref(src, n, block, pid, other=0.0, arange_start=None):
    """一个 program 的掩码读参考值。

    `src` 是长度 >= n 的 Python 列表；返回该 program 在 BLOCK 个位置上的期望值：
      - 下标 < n：src[下标]
      - 下标 >= n：`other`
    逐位置用 Python 循环判断，不复用被测侧的掩码表达式。
    """
    start = pid * block if arange_start is None else arange_start
    out = []
    for k in range(block):
        i = start + k
        if 0 <= i < n:
            out.append(float(src[i]))
        else:
            out.append(float(other))
    return out


def oob_positions(n, block, grid_size):
    """所有越界位置 (pid, k, 全局下标) 的清单。"""
    out = []
    for pid in range(grid_size):
        for k in range(block):
            i = pid * block + k
            if not (0 <= i < n):
                out.append((pid, k, i))
    return out


def masked_index_np_ref(n, block, pid, other=0.0, src=None):
    """掩码读的**第二条独立参照**：用 numpy 花式索引 + clip 写，不走标量循环。

    返回 (值数组, 掩码数组)。值在有效位置取 `src[下标]`，越界位置取 `other`。
    与 `masked_block_ref`（纯 Python 标量循环）是两条不同的代码路径。
    """
    import numpy as np
    idx = np.arange(pid * block, pid * block + block, dtype=np.int64)
    mask = idx < n
    if src is None:
        src = np.arange(n, dtype=np.float64)
    safe = np.clip(idx, 0, n - 1)
    vals = np.where(mask, np.asarray(src)[safe], other)
    return vals, mask


def softmax_prefix_np_ref(rows_np, n_valid):
    """「只对每行前 n_valid 个元素做 softmax」的 numpy 参照。"""
    import numpy as np
    pre = np.asarray(rows_np, dtype=np.float64)[:, :n_valid]
    m = pre.max(axis=1, keepdims=True)
    e = np.exp(pre - m)
    return e / e.sum(axis=1, keepdims=True)


def softmax_over_prefix_ref(x, n_valid):
    """只对每行的前 n_valid 个元素做 softmax 的参照（越界位置必须完全无贡献）。

    返回每行前 n_valid 个元素上的 softmax（float64 标量循环）。
    """
    out = []
    for row in x:
        pre = [float(v) for v in row[:n_valid]]
        m = max(pre)
        es = [math.exp(v - m) for v in pre]
        s = 0.0
        for e in es:
            s += e
        out.append([e / s for e in es])
    return out


# ---------------------------------------------------------------------------
# 与布局分析无关的通用小工具
# ---------------------------------------------------------------------------

def max_abs_err(a, b):
    """两个嵌套列表的逐元素最大绝对误差。"""
    worst = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            d = abs(float(va) - float(vb))
            if d > worst:
                worst = d
    return worst


def f32_eps():
    return 2.0 ** -23


def rel_err_bound(n_terms, extra_slack=64.0):
    """f32 上 n_terms 项求和的**保守**相对误差上界（简单的 n*eps 模型）。"""
    return float(n_terms) * f32_eps() * extra_slack
