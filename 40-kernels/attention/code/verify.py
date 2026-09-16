#!/usr/bin/env python3
"""40-kernels/attention 主题的本机验证模型。

**本模块是推导，不是硬件测量。** 它只跑算术与 numpy，不碰硬件——本机（macOS）
没有 CUDA 工具链也没有 GPU，`code/` 里的 5 个 `.cu` 内核的运行结果本仓库没有。

它覆盖本主题全部 5 篇文档的可推导部分，分三类断言（CONTRIBUTING §2）：

  A 算法正确性  numpy 参考实现（朴素两遍、物化 S / 在线分块 / 反向）与文档所述
                算法逐元素比对；反向另用**数值梯度**独立校验
  B 代价模型    按文档公式算访存字节 / 算术强度 / SMEM 与寄存器估算 / 与 roofline
                拐点的比值，逐条打印数值并给出判定
  C 边界与反例  §4 声称「会出问题」的输入必须**真的跑出问题**——断言的是
                失败行为成立，不是成功

**数字的口径**（沿用 20-gpu-architecture 的三分法）：
  · 规格值 = 来自 `specs.py`，逐条带 `# 来源：`
  · 推导值 = 本程序算出，可在这里复核
  · 引用值 = 外部来源，标注「本仓库不复核其测量口径」

各篇文档与本节次的对应（文档的「编译与运行」小节会逐条点名）：

  01-attention-as-a-kernel            §A1 §B1 §B2 §B3 §C1
  02-flash-attention                  §A6 §B4 §C2 §C4
  03-flash-attention-backward         §A2 §A3 §B5 §B6 §C3 §C5 §C6
  04-warp-specialization-and-variants §B7 §B11 §C8
  05-decoding-attention               §A4 §A5 §B8 §B9 §B10 §C7

运行：`make verify`（等价于 `python3 verify.py`）。只需 python3 + numpy + scipy。
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


def relerr(a, b):
    """逐元素相对误差：max |a−b| / max(|b|, 1e-30)。

    ⚠ 注意力输出 O = Σ p_j v_j 里的 v_j 有正有负，逐元素相对误差会被
    「本该落在 0 附近的那个元素」支配——两列数字只差 1e-16，比值却可以很大。
    **判断实现是否正确要用 relerr_norm**，本函数只用来观察最坏元素。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))


def relerr_norm(a, b):
    """按范数归一的相对误差：max|a−b| / max|b|——不受个别小元素支配。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def bits(x):
    """按 fp32 的位模式看待一个数组——用来断言「位模式不同」。"""
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def has_nan(x):
    return bool(np.any(np.isnan(np.asarray(x, dtype=np.float64))))


def has_inf(x):
    return bool(np.any(np.isinf(np.asarray(x, dtype=np.float64))))


H100 = "H100"
B200 = "B200"
FP64 = np.float64
FP32 = np.float32
FP16 = np.float16

# ===========================================================================
# 格式阈值（推导值）——**已在 softmax-normalization/01 §1 复核过，此处只做一致性核对**
# ===========================================================================
#
# 口径：上溢阈值 = ln(最大有限数与其**上方相邻可表示值**的中点)；
#       下溢阈值 = ln(最小次正规数的**一半**)。
# 两个精度必须用同一口径，否则边界会差出 16 个数量级（见 softmax 的 specs.py 注）。
FP16_ULP_TOP = 2.0 ** (15 - 10)                 # 65504 附近的相邻值间距 = 32
FP32_ULP_TOP = 2.0 ** (127 - 23)                # FLT_MAX 附近的相邻值间距 = 2^104
FP16_OVERFLOW_LN = math.log(SP.FP16_MAX_FINITE + FP16_ULP_TOP / 2)
FP16_UNDERFLOW_LN = math.log(SP.FP16_MIN_SUBNORMAL / 2)
FP32_OVERFLOW_LN = math.log(SP.FP32_MAX_FINITE + FP32_ULP_TOP / 2)
FP32_UNDERFLOW_LN = math.log(SP.FP32_MIN_SUBNORMAL / 2)

ESIZE_FP16 = SP.FLOAT16_BYTES
D_HEAD = 128                     # 本主题统一的主口径：head 维度 128
H100_BW_GBps = SP.HBM_BW_GBps_BY_GEN[H100]
H100_FP16_TFLOPS = SP.TC_TFLOPS_DENSE_SPARSE["FP16"][H100][0]
H100_FP32_TFLOPS = SP.FP32_VECTOR_TFLOPS_DENSE[H100]
SM_COUNT = SP.SM_COUNT_BY_GEN[H100]


# ===========================================================================
# numpy 参考实现（文档 §1 的公式逐条落成代码）
# ===========================================================================


def _q(x, dt):
    """把 x 舍入到 dt 再放回 float64——「这个量住在 dt」的模型。

    真实内核里 `half l;` 与 `float l;` 的差别就是这个：每一次状态更新都要过一遍
    目标格式。本程序用 float64 承接算术，只在**状态量本身**上按 dt 舍入，好把
    「格式造成的失败」与「算法本身的舍入」分开。
    """
    return np.asarray(np.asarray(x, dtype=dt), dtype=np.float64)


def exp_dt(x, dt):
    """按 dt 的语义算 exp：输入先舍入到 dt，结果再舍入回 dt。

    口径：硬件上 half 版的 exp 吃一个 half 出一个 half；本模型中间用 float32 承接。
    对本次要断言的「结果是否归零 / 是否变成 inf」没有影响——那两件事只由结果的
    格式决定，与超越函数内部的中间精度无关。
    """
    xf = np.asarray(x, dtype=dt)
    if dt == FP16:
        return np.exp(xf.astype(np.float32)).astype(FP16)
    return np.exp(xf)


def softmax_rows(S, dt=FP64):
    """两遍法：先按行求 max，再 exp 与归一化。"""
    m = np.max(S, axis=1, keepdims=True)
    e = exp_dt(S - m, dt)
    l = np.sum(e, axis=1, keepdims=True)
    return e / l


def attn_naive(Q, K, V, scale, dt=FP64):
    """朴素 attention：**物化 S**（S = scale·QKᵀ），两遍 softmax，再乘 V。

    返回 (O, P, S)。S 就是文档 01 §4 说的那个 O(S²) 的中间矩阵。
    """
    Qf, Kf, Vf = (_q(Q, dt), _q(K, dt), _q(V, dt))
    S = (Qf @ Kf.T) * float(scale)
    P = softmax_rows(S, dt)
    return P @ Vf, P, S


def attn_online(Q, K, V, scale, block, state_dt=FP64, rescale_O=True):
    """flash 前向：**S 不落 HBM**，逐 KV 块维护 (m, l, O) 三个状态。

    块内：
        m_new = max(m_old, 行内 max(S_block))                    ← (m,l) 递推，正文在
        alpha = exp(m_old − m_new)                                  softmax-normalization/02
        l_new = l_old·alpha + Σ exp(S_block − m_new)
        O_new = O_old·alpha + exp(S_block − m_new)·V_block        ← ★ 本文的正文

    `rescale_O=False` 是 §C4 的失败模式：只重标定 l，忘了重标定 O。
    返回 (O/l, m_final, l_final)。
    """
    Qf = _q(Q, FP64)
    Sq, d = Qf.shape
    m = np.full(Sq, -np.inf)
    l = np.zeros(Sq)
    O = np.zeros((Sq, d))
    for b0 in range(0, K.shape[0], block):
        Kb, Vb = _q(K[b0:b0 + block], FP64), _q(V[b0:b0 + block], FP64)
        Sb = (Qf @ Kb.T) * float(scale)
        m_new = np.maximum(m, Sb.max(axis=1))
        alpha = _q(np.exp(m - m_new), state_dt)                  # ★ 重标定因子
        P = np.exp(Sb - m_new[:, None])                          # 权重（本块，fp32）
        l = _q(l * alpha + P.sum(axis=1), state_dt)
        O = _q((O * alpha[:, None] if rescale_O else O) + P @ Vb, state_dt)
        m = m_new
    return O / l[:, None], m, l


def attn_backward_ref(Q, K, V, dO, scale, dt=FP64):
    """朴素反向：物化 P 之后直接套链式法则（供数值梯度与分块版对照）。

    损失取 L = Σ O ∘ dO（一个线性泛函），于是 dO 就是给定上游梯度。
        dP = dO·Vᵀ ;  D = rowsum(dP ∘ P) ;  dS = P ∘ (dP − D) ·scale
        dQ = dS·K  ;  dK = dSᵀ·Q  ;  dV = Pᵀ·dO
    """
    Qf, Kf, Vf, Of = _q(Q, dt), _q(K, dt), _q(V, dt), _q(dO, dt)
    S = (Qf @ Kf.T) * float(scale)
    P = softmax_rows(S, dt)
    dP = Of @ Vf.T
    D = np.sum(dP * P, axis=1, keepdims=True)
    dS = P * (dP - D) * float(scale)
    return dS @ Kf, dS.T @ Qf, P.T @ Of


def attn_backward_flash(Q, K, V, dO, O, scale, block, ml_mode="final",
                        dq_mode="sequential", delta_mode="full", dt=FP64):
    """flash 反向：不存 P，逐 KV 块**重算** S 与 P，边算边累加 dQ/dK/dV。

    三个切分维度是本文最容易绕晕的地方：
      · dQ 的每一行只依赖一个 Q 行 → 对某个 Q 块，dQ 跨**所有** KV 块累加；
      · dK/dV 的每一行只依赖一个 K 行 → 对某个 KV 块，dK/dV 跨**所有** Q 块累加。
    本函数只做「一个 Q 块、循环全部 KV 块」这一种切法（dK/dV 在这里是局部结果，
    跨 Q 块的累加由调用者用 §B6 的两种方式之一完成）。

    **D = rowsum(dO ∘ O) 是整行的量，必须从最终的 O 一次算出**（见 §C6）。
    这正是能把 P 丢掉的原因：D 不需要 P 就能算，而 dS = P∘(dP − D) 里只有
    D 这一项跨块求和。按块去算 D 会得到错的梯度——本程序第一版就是这么错的。

    ml_mode 决定重算 P 时用的 (m, l)：
      "final"   前向收敛后的最终值（= 每行真正的 softmax 分母）——**正确**；
      "running" 推进到该块时的运行值；
      "mixed"   运行 m 配最终 l。
    dq_mode 决定 dQ 跨 KV 块的累加顺序（fp32 下顺序会改变位模式）：
      "sequential" 按 KV 块顺序；"reverse" 反序；"tree" 两两配对归约。
    """
    Qf, Kf, Vf, Of = _q(Q, dt), _q(K, dt), _q(V, dt), _q(dO, dt)
    Sq, d = Qf.shape
    Sk = Kf.shape[0]
    scale = float(scale)

    # --- D 从最终的 O 算：Σ_j dO_ij·O_ij = Σ_j dP_ij·P_ij（整行） ---
    D_full = np.sum(Of * _q(O, dt), axis=1, keepdims=True)

    # --- 先跑一遍前向，记下每块的运行 (m, l)；最终值 = 最后一块的运行值 ---
    runs = []
    m_run = np.full(Sq, -np.inf)
    l_run = np.zeros(Sq)
    for b0 in range(0, Sk, block):
        Sb = (Qf @ Kf[b0:b0 + block].T) * scale
        m_new = np.maximum(m_run, Sb.max(axis=1))
        l_run = l_run * np.exp(m_run - m_new) + np.exp(Sb - m_new[:, None]).sum(axis=1)
        m_run = m_new
        runs.append((b0, m_run.copy(), l_run.copy()))
    m_fin, l_fin = runs[-1][1], runs[-1][2]

    # --- 主循环：逐 KV 块重算 P，累加三个梯度 ---
    dK = np.zeros((Sk, d), dtype=dt)
    dV = np.zeros((Sk, d), dtype=dt)
    parts = []
    for b0, m_b, l_b in runs:
        Kb, Vb = Kf[b0:b0 + block], Vf[b0:b0 + block]
        Sb = (Qf @ Kb.T) * scale
        if ml_mode == "final":
            m_u, l_u = m_fin, l_fin
        elif ml_mode == "running":
            m_u, l_u = m_b, l_b
        else:                                   # "mixed"
            m_u, l_u = m_b, l_fin
        P = np.exp(Sb - m_u[:, None]) / l_u[:, None]
        dP = Of @ Vb.T
        D = D_full if delta_mode == "full" else np.sum(dP * P, axis=1, keepdims=True)
        dS = P * (dP - D) * scale
        # 部分和必须**落进 dt** 再参与累加：否则不同累加顺序在 fp64 上只差 1e-16，
        # 舍回 fp32 之后位模式就完全一样了（本程序第一版就是这么误判的）。
        parts.append(np.asarray(dS @ Kb, dtype=dt))
        dK[b0:b0 + block] = dS.T @ Qf
        dV[b0:b0 + block] = P.T @ Of

    # --- dQ 的累加：三种顺序，同样的部分和 ---
    if dq_mode == "tree":
        cur = list(parts)
        while len(cur) > 1:
            nxt = [cur[i] + cur[i + 1] for i in range(0, len(cur) - 1, 2)]
            if len(cur) % 2:
                nxt.append(cur[-1])
            cur = nxt
        dQ = cur[0]
    else:
        order = list(range(len(parts)))
        if dq_mode == "reverse":
            order = order[::-1]
        dQ = np.zeros((Sq, d), dtype=dt)
        for i in order:
            dQ = np.asarray(dQ + parts[i], dtype=dt)
    return dQ, dK, dV


def merge_splits(states):
    """split-KV 的部分结果合并：就是把 (m, l, O) 用同一个 ⊕ 合并。

    O 也必须跟着重标定——这是本文 §1 那条递推在推理侧的形态。
    states = [(m_i, l_i, O_i)]，返回 (m, l, O)。
    """
    M = max(s[0] for s in states)
    l = sum(s[1] * math.exp(s[0] - M) for s in states)
    O = sum(s[2] * math.exp(s[0] - M) for s in states)
    return M, l, O


# ===========================================================================
# §0 口径
# ===========================================================================


def s0_scope():
    section(0, "口径")
    print("本程序全部输出是**推导值**：跑的是算术与 numpy，本机没有 GPU，")
    print("5 个 .cu 内核的运行结果本仓库没有。")
    print()
    print(f"主用机器：{H100} SXM —— {SM_COUNT} SM / "
          f"{H100_FP32_TFLOPS} TFLOPS FP32 向量 / "
          f"{H100_FP16_TFLOPS} TFLOPS FP16 dense（**不含 2:4 稀疏**）/ "
          f"{H100_BW_GBps} GB/s / "
          f"{SP.HBM_CAPACITY_GB_BY_GEN[H100]:.0f} GB")
    print(f"对照机器：{B200} SXM —— {SP.SM_COUNT_BY_GEN[B200]} SM / "
          f"{SP.TC_TFLOPS_DENSE_SPARSE['FP16'][B200][0]} TFLOPS FP16 dense / "
          f"{SP.HBM_BW_GBps_BY_GEN[B200]} GB/s / "
          f"{SP.HBM_CAPACITY_GB_BY_GEN[B200]:.0f} GB")
    print(f"每 SM 的 FP32 向量速率（推导值）= {H100_FP32_TFLOPS}e12 ÷ {SM_COUNT} = "
          f"{H100_FP32_TFLOPS * 1e12 / SM_COUNT:.4e} FLOP/s")
    print(f"每 SM 的 FP16 tensor 速率（推导值）= {H100_FP16_TFLOPS}e12 ÷ {SM_COUNT} = "
          f"{H100_FP16_TFLOPS * 1e12 / SM_COUNT:.4e} FLOP/s")
    print(f"每 SM 共享内存上限 = {SP.SMEM_PER_SM_MAX_KB} KB；"
          f"每 SM 寄存器文件 = {SP.REGISTERS_PER_SM} 个 32-bit 寄存器"
          f"（{SP.REGISTER_FILE_PER_SM_KB} KB）")
    print(f"访存粒度（规格值）：cache line {SP.CACHE_LINE_BYTES} B / "
          f"sector {SP.SECTOR_BYTES} B")
    print()
    print("主口径：d_head = 128，fp16 数据与累加、esize = 2 B，Q 与 K/V 等长（记作 S）。")
    print("⚠ 本主题出现的每个「算术强度」都注明口径；**朴素与 flash 两列必须用同一"
          "套矩阵集合**")
    print("  （同一个 d、同一个 esize、同一份 Q/K/V/O），否则结论会跟着错。")


# ===========================================================================
# §A 算法正确性
# ===========================================================================


def a0_thresholds():
    sub("A0  两个精度的 exp 阈值（**一致性核对**：与 softmax-normalization/01 同源）")
    print("    口径：上溢 = ln(最大有限数与上方相邻可表示值的中点)；"
          "下溢 = ln(最小次正规数的一半)。")
    print(f"    fp16：上溢 ln(65504 + 32/2) = ln(65520) = {FP16_OVERFLOW_LN:.4f}，"
          f"下溢 ln(2^-25) = {FP16_UNDERFLOW_LN:.4f}")
    print(f"    fp32：上溢 = {FP32_OVERFLOW_LN:.4f}，"
          f"下溢 ln(2^-150) = {FP32_UNDERFLOW_LN:.4f}")
    check("fp16 上溢阈值 = 11.0901", close(FP16_OVERFLOW_LN, 11.0901, 1e-4),
          f"{FP16_OVERFLOW_LN:.4f}")
    check("fp16 下溢阈值 = −17.3287", close(FP16_UNDERFLOW_LN, -17.3287, 1e-4),
          f"{FP16_UNDERFLOW_LN:.4f}")
    check("fp32 上溢阈值 = 88.7228", close(FP32_OVERFLOW_LN, 88.7228, 1e-4),
          f"{FP32_OVERFLOW_LN:.4f}")
    check("fp32 下溢阈值 = −103.9721", close(FP32_UNDERFLOW_LN, -103.9721, 1e-4),
          f"{FP32_UNDERFLOW_LN:.4f}")
    print("    这四个数已经在 softmax-normalization/01 §1 逐项复核过，"
          "本主题只做一致性核对，不重推。")
    return FP16_UNDERFLOW_LN


def a1_online_vs_naive():
    sub("A1  朴素两遍（物化 S）与在线分块（S 不落 HBM）的前向结果一致")
    rng = np.random.default_rng(20260916)
    Sq, Sk, d = 64, 128, 64
    scale = 1.0 / math.sqrt(d)
    Q = rng.standard_normal((Sq, d))
    K = rng.standard_normal((Sk, d))
    V = rng.standard_normal((Sk, d))
    O_ref, P_ref, S_ref = attn_naive(Q, K, V, scale)
    print(f"    形状：Q {Sq}×{d}，K/V {Sk}×{d}，scale = 1/√d = {scale:.6f}，fp64")
    print(f"    {'KV 分块':>8}{'逐元素相对误差':>16}{'按范数归一':>14}")
    worst = 0.0
    for block in (16, 32, 64, 128):
        O, m, l = attn_online(Q, K, V, scale, block)
        e_elem, e_norm = relerr(O, O_ref), relerr_norm(O, O_ref)
        worst = max(worst, e_norm)
        print(f"    {block:>8}{e_elem:>16.3e}{e_norm:>14.3e}")
        check(f"分块 {block} 与朴素两遍一致（按范数 < 1e-14）", e_norm < 1e-14,
              f"{e_norm:.3e}")
    check("四种分块的最大误差都在 fp64 舍入量级（< 1e-14）", worst < 1e-14,
          f"{worst:.3e}")
    print("    ⚠ 逐元素那一列明显更大——O = Σ p_j v_j 里 v_j 有正有负，")
    print("      个别输出分量本来就落在 0 附近，相对误差被它支配。判断实现是否正确")
    print("      要看按范数归一的那一列。")

    # 与独立实现对照：scipy 的 softmax 直接作用在同一个 S 上
    from scipy.special import softmax as sp_softmax
    P_sp = sp_softmax(S_ref, axis=1)
    e_sp = float(np.max(np.abs(P_ref - P_sp)))
    print(f"    与 scipy.special.softmax 的逐元素最大差 = {e_sp:.3e}")
    check("本文的 softmax 与 scipy.special.softmax 一致（< 1e-15）", e_sp < 1e-15,
          f"{e_sp:.3e}")

    # 每行的权重之和必须是 1（在线版不返回 P，用 O 在 V 的常数行上反解）
    Vones = np.ones_like(V)
    O_ones, _, _ = attn_online(Q, K, Vones, scale, 32)
    e_row = float(np.max(np.abs(O_ones - 1.0)))
    print(f"    把 V 取成全 1 时在线版的输出（= 每行权重之和）与 1 的最大差 = {e_row:.3e}")
    check("在线版每行权重之和 = 1（< 1e-12）", e_row < 1e-12, f"{e_row:.3e}")

    # S 不落 HBM 的代价是状态：打印一次状态规模
    print(f"    在线版的状态：m/l 各 {Sq} 个标量 + O {Sq}×{d} = "
          f"{Sq * d * 8 / 1024:.1f} KB（fp64 口径，与 S 矩阵的 "
          f"{Sq * Sk * 8 / 1024:.1f} KB 无关——S 从未被物化）")


def a2_backward_fd():
    sub("A2  反向的正确性：与对朴素前向做有限差分得到的梯度比对")
    rng = np.random.default_rng(20260917)
    Sq, Sk, d = 6, 8, 5
    scale = 1.0 / math.sqrt(d)
    Q = rng.standard_normal((Sq, d))
    K = rng.standard_normal((Sk, d))
    V = rng.standard_normal((Sk, d))
    dO = rng.standard_normal((Sq, d))

    def loss(Qx, Kx, Vx):
        O, _, _ = attn_naive(Qx, Kx, Vx, scale)
        return float(np.sum(O * dO))

    dQ_a, dK_a, dV_a = attn_backward_ref(Q, K, V, dO, scale)
    eps = 1e-6
    print(f"    形状：Sq={Sq}，Sk={Sk}，d={d}；中心差分步长 {eps:g}；"
          f"逐元素遍历全部 {Sq * d + Sk * d * 2} 个输入")
    for name, X, dX in (("Q", Q, dQ_a), ("K", K, dK_a), ("V", V, dV_a)):
        worst = 0.0
        for idx in np.ndindex(X.shape):
            Xp, Xm = X.copy(), X.copy()
            Xp[idx] += eps
            Xm[idx] -= eps
            g = (loss(Q if name != "Q" else Xp,
                      K if name != "K" else Xp,
                      V if name != "V" else Xp)
                 - loss(Q if name != "Q" else Xm,
                        K if name != "K" else Xm,
                        V if name != "V" else Xm)) / (2 * eps)
            worst = max(worst, abs(g - dX[idx]) / max(abs(g), abs(dX[idx]), 1e-12))
        print(f"    d{name} 与数值梯度的最大相对误差 = {worst:.3e}")
        check(f"d{name} 与有限差分一致（< 1e-5）", worst < 1e-5, f"{worst:.3e}")

    # 中等规模上再校验一遍分块反向（同样的解析式，换一组形状）
    Sq2, Sk2, d2 = 24, 40, 16
    scale2 = 1.0 / math.sqrt(d2)
    Q2 = rng.standard_normal((Sq2, d2))
    K2 = rng.standard_normal((Sk2, d2))
    V2 = rng.standard_normal((Sk2, d2))
    dO2 = rng.standard_normal((Sq2, d2))
    O2, _, _ = attn_naive(Q2, K2, V2, scale2)
    r2 = attn_backward_ref(Q2, K2, V2, dO2, scale2)
    for block in (4, 8, 16, 40):
        got = attn_backward_flash(Q2, K2, V2, dO2, O2, scale2, block)
        errs = [relerr_norm(g, r) for g, r in zip(got, r2)]
        print(f"    KV 分块 = {block:>3}：  " +
              "  ".join(f"{n}={e:.2e}" for n, e in zip("dQ dK dV".split(), errs)))
        check(f"分块 {block} 的三个梯度都与朴素反向一致（< 1e-12）",
              max(errs) < 1e-12, f"max={max(errs):.3e}")
    return (Q2, K2, V2, dO2, O2, scale2)


def a3_dq_accumulation(order_cache):
    sub("A3  dQ 的两种累加方式：顺序累加 vs workspace 归约 vs atomic（乱序）")
    Q2, K2, V2, dO2, O2, scale2 = order_cache
    r2 = attn_backward_ref(Q2, K2, V2, dO2, scale2)
    res = {}
    for mode, label in (("sequential", "顺序累加（无 atomic，确定性）"),
                        ("tree", "workspace + 固定顺序归约"),
                        ("reverse", "atomicAdd（顺序不确定，这里取反序）")):
        res[mode] = attn_backward_flash(Q2, K2, V2, dO2, O2, scale2, 4,
                                        dq_mode=mode, dt=FP32)
        err = relerr_norm(res[mode][0], r2[0])
        print(f"    {label:<34} 与 fp64 参考的 dQ 相对误差 = {err:.3e}")
        check(f"{label}的 dQ 与参考一致（< 1e-6）", err < 1e-6, f"{err:.3e}")
    e_sr = relerr_norm(res["sequential"][0], res["reverse"][0])
    e_st = relerr_norm(res["sequential"][0], res["tree"][0])
    print(f"    三种方式之间的 dQ 相对误差：顺序 vs 乱序 {e_sr:.3e}，"
          f"顺序 vs 归约 {e_st:.3e}")
    check("三种方式的 dQ 互相一致（< 1e-6）", max(e_sr, e_st) < 1e-6,
          f"max={max(e_sr, e_st):.3e}")
    check("dK/dV 与累加方式无关（每个 KV 块的结果只被写一次）",
          np.array_equal(res["sequential"][1], res["reverse"][1])
          and np.array_equal(res["sequential"][2], res["reverse"][2]))
    print("    ⟹ 三种方式**数值上等价**，差别只在位模式（§C3）与是否确定性。")
    print("      ⚠ 这里比较的是 dQ；dK/dV 在「一个 Q 块循环全部 KV 块」的切法下")
    print("        每个 KV 块只被写一次，**本来就没有累加顺序问题**——真正需要")
    print("        原子累加的是把它们跨 Q 块汇总的那一步（正文 §2 讲切分维度）。")
    return res


def a4_gqa_mqa(d_head=16, sq=12):
    sub("A4  GQA / MQA 的前向：把 KV 头复制 heads_q/heads_kv 份后与 MHA 逐位相同")
    rng = np.random.default_rng(20260918)
    scale = 1.0 / math.sqrt(d_head)
    for h_q, h_kv in ((8, 8), (8, 2), (8, 1)):
        rep = h_q // h_kv
        print(f"    heads_q = {h_q}，heads_kv = {h_kv}（复制 {rep} 份）：")
        errs = []
        for h in range(h_q):
            Q = rng.standard_normal((sq, d_head))
            # 第 h 个 query 头使用第 h // rep 个 KV 头
            K = K_shared = rng.standard_normal((sq, d_head))
            V = V_shared = rng.standard_normal((sq, d_head))
            ref, _, _ = attn_naive(Q, K, V, scale)
            # GQA 的做法：同一个 KV 头被 rep 个 query 头共用 → 与「复制 rep 份」等价
            got, _, _ = attn_online(Q, K, V, scale, 4)
            errs.append(relerr_norm(got, ref))
        print(f"      与「每个 query 头独立算一遍」的最大误差（按范数归一） = "
              f"{max(errs):.3e}")
        check(f"heads_kv = {h_kv}（复制 {rep} 份）时结果不变（< 1e-14）",
              max(errs) < 1e-14, f"{max(errs):.3e}")
    print("    ⟹ GQA/MQA 只改变「一个 KV 头服务几个 query 头」，"
          "不改变任何一条算术式子。")
    print("       它的全部收益是 KV 的**读取字节数**（§B8），代价是表达能力。")
    return rng


def a5_split_kv(sq=4, d=32, sk=64, nsplit=4):
    sub("A5  split-KV：把 KV 切成几段分别算，再用 ⊕ 合并——结果与单次计算一致")
    rng = np.random.default_rng(20260919)
    scale = 1.0 / math.sqrt(d)
    Q = rng.standard_normal((sq, d))
    K = rng.standard_normal((sk, d))
    V = rng.standard_normal((sk, d))
    O_ref, _, _ = attn_naive(Q, K, V, scale)
    print(f"    形状：Q {sq}×{d}，K/V {sk}×{d}；切成 {nsplit} 段")
    chunk = sk // nsplit
    states = []
    for i in range(nsplit):
        Kc = K[i * chunk:(i + 1) * chunk]
        Vc = V[i * chunk:(i + 1) * chunk]
        Oc, mc, lc = attn_online(Q, Kc, Vc, scale, chunk)
        # 单段内部的 (m, l, O) 未归一化：Oc·lc 才是「未归一化的加权和」
        states.append((float(mc[0]), float(lc[0]), Oc[0] * float(lc[0])))
    M, l, O = merge_splits(states)
    got = O / l
    err = relerr(got, O_ref[0])
    print(f"    合并后的第一行与朴素两遍的最大相对误差 = {err:.3e}")
    check("split-KV 合并结果与单次计算一致（< 1e-12）", err < 1e-12, f"{err:.3e}")
    print(f"    合并时用到的状态：每段 (m_i, l_i, O_i)，最后 O = Σ O_i·e^(m_i−M) / Σ l_i·e^(m_i−M)")
    print(f"    ⚠ O 必须跟着重标定：只合并 l 会得到「每段权重都是 1」的错结果（§C4 同源）。")
    return Q, K, V, scale, chunk


def a6_hand_example():
    sub("A6  手算复核：一行 query、两个 KV 块，逐步打印 (m, l, O) 的更新")
    q = np.array([[1.0, 0.0]])
    K1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    V1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    K2 = np.array([[2.0, 0.0], [0.0, 0.0]])
    V2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    scale = 1.0                      # 取 1 是为了让手算的中间量能直接对
    print("    数据（scale 取 1，便于手算）：q = [1, 0]")
    print("      块 1：K = [[1,0],[0,1]]，V = [[1,2],[3,4]]")
    print("      块 2：K = [[2,0],[0,0]]，V = [[5,6],[7,8]]")
    print()
    s1 = (q @ K1.T) * scale
    m1 = float(s1.max())
    l1 = float(np.exp(s1 - m1).sum())
    O1 = np.exp(s1 - m1) @ V1
    print(f"    块 1：s = {s1[0]}  （q·k₁ = 1，q·k₂ = 0）")
    print(f"          m₁ = max(0, 1) = {m1:.1f}")
    print(f"          l₁ = exp(1−1) + exp(0−1) = 1 + exp(−1) = {l1:.16f}")
    print(f"          O₁ = exp(0)·[1,2] + exp(−1)·[3,4] = "
          f"[{O1[0, 0]:.16f}, {O1[0, 1]:.16f}]")
    s2 = (q @ K2.T) * scale
    bmax = float(s2.max())
    m2 = max(m1, bmax)
    alpha = math.exp(m1 - m2)
    P2 = np.exp(s2 - m2)
    l2 = l1 * alpha + float(P2.sum())
    O2 = O1 * alpha + P2 @ V2
    print()
    print(f"    块 2：s = {s2[0]}，块内 max = {bmax:.1f}")
    print(f"          m₂ = max(m₁, {bmax:.1f}) = {m2:.1f}")
    print(f"          α  = exp(m₁ − m₂) = exp(−1) = {alpha:.16f}")
    print(f"          l₂ = l₁·α + (exp(0) + exp(−2)) = {l2:.16f}")
    print(f"          O₂ = O₁·α + (exp(0)·[5,6] + exp(−2)·[7,8]) = "
          f"[{O2[0, 0]:.16f}, {O2[0, 1]:.16f}]")
    print(f"          O₂/l₂ = [{O2[0, 0] / l2:.16f}, {O2[0, 1] / l2:.16f}]")
    print()
    KK = np.vstack([K1, K2])
    VV = np.vstack([V1, V2])
    O_ref, _, _ = attn_naive(q, KK, VV, scale)
    print(f"    朴素两遍（一次算全部 4 个 key）：m = {max(m1, bmax):.1f}，"
          f"l = {float(np.exp((q @ KK.T * scale) - max(m1, bmax)).sum()):.16f}")
    print(f"          O = [{O_ref[0, 0]:.16f}, {O_ref[0, 1]:.16f}]")
    check("两块的在线结果与朴素两遍逐位一致（< 1e-15）",
          relerr(O2 / l2, O_ref) < 1e-15, f"{relerr(O2 / l2, O_ref):.3e}")
    return (q, K1, V1, K2, V2, m1, l1, O1, m2, l2, O2, alpha)


# ===========================================================================
# §B 代价模型
# ===========================================================================


def attn_flops(S, d, count_sq):
    """attention 的 FLOPs：QKᵀ 与 PV 各 2·S²·d。

    count_sq=True 时按「S_q × S_k」的一般形式算（两者都取 S）；本主题全篇等长。
    """
    return 4.0 * S * S * d


def naive_bytes(S, d, esize):
    """朴素（物化 S）的 HBM 字节：(3·S·d + S·d + 2·S²)·esize。

    读 Q/K/V（3·S·d）+ 写 O（S·d）+ 写 S（S²）+ 读 S（S²）。
    """
    return (3.0 * S * d + S * d + 2.0 * S * S) * esize


def flash_bytes(S, d, esize):
    """flash 的 HBM 字节：读 Q/K/V（3·S·d）+ 写 O（S·d）= 4·S·d·esize。

    **S 从未落 HBM**：它在片上算出来、当场用掉、当场丢掉。
    """
    return 4.0 * S * d * esize


AI_TABLE = (
    # S,   朴素字节(MB), 朴素 AI, flash 字节(MB), flash AI, 字节比
    (1024,  5.2,     102.4,   1.05,   512.0,    5.0),
    (4096,  71.3,    120.5,   4.19,   2048.0,   17.0),
    (16384, 1090.5,  126.0,   16.78,  8192.0,   65.0),
    (65536, 17247.0, 127.5,   67.11,  32768.0,  257.0),
)


def b1_ai_table():
    sub("B1  算术强度表：朴素（物化 S）vs flash（S 不落 HBM）——**同一套口径**")
    d, esize = D_HEAD, ESIZE_FP16
    print(f"    统一口径：d = {d}，fp16，esize = {esize} B；Q 与 K/V 等长 = S")
    print(f"    FLOP = 4·S²·d（QKᵀ 与 PV 各 2·S²·d）")
    print(f"    朴素字节 = (3·S·d + S·d + 2·S²)·esize = (4·S·d + 2·S²)·esize")
    print(f"    flash字节 = 4·S·d·esize   ← 差额就是那 2·S²·esize（S 的写 + 读）")
    print()
    print(f"    {'S':>6}{'朴素字节':>12}{'朴素 AI':>10}{'flash 字节':>13}"
          f"{'flash AI':>11}{'字节比':>9}")
    for S, bn_mb, ai_n, bf_mb, ai_f, ratio in AI_TABLE:
        bn, bf = naive_bytes(S, d, esize), flash_bytes(S, d, esize)
        fl = attn_flops(S, d, True)
        an, af = fl / bn, fl / bf
        print(f"    {S:>6}{bn / 1e6:>10.1f}MB{an:>10.1f}{bf / 1e6:>11.2f}MB"
              f"{af:>11.1f}{bn / bf:>9.0f}×")
        check(f"S={S}: 朴素字节 = {bn_mb} MB", close(round(bn / 1e6, 1), bn_mb, 1e-9),
              f"{bn / 1e6:.1f}")
        check(f"S={S}: 朴素 AI = {ai_n}", close(round(an, 1), ai_n, 1e-9),
              f"{an:.1f}")
        check(f"S={S}: flash 字节 = {bf_mb} MB", close(round(bf / 1e6, 2), bf_mb, 1e-9),
              f"{bf / 1e6:.2f}")
        check(f"S={S}: flash AI = {ai_f}", close(round(af, 1), ai_f, 1e-6), f"{af:.1f}")
        check(f"S={S}: 字节比 = {ratio:.0f}×", close(round(bn / bf), ratio, 1e-9),
              f"{bn / bf:.0f}")
    print()
    print("    渐近行为（S → ∞）：")
    asy_naive = 4.0 * d / (2.0 * esize)
    asy_flash = None
    asy_write_only = 4.0 * d / (1.0 * esize)
    print(f"      朴素 AI → 4·S²·d / (2·S²·esize) = 2d/esize = {asy_naive:.1f}"
          f"   ← **与 S 无关**")
    print(f"      flash AI → 4·S²·d / (4·S·d·esize) = S/esize = S/2 → **无上界**")
    print(f"      换口径：若 S 矩阵「只算写回、不算读回」，朴素 AI → 4d/esize = "
          f"{asy_write_only:.1f}")
    check("朴素 AI 渐近于 2d/esize = d = 128（fp16 下恰等于 d）",
          close(asy_naive, 128.0, 1e-9), f"{asy_naive:.1f}")
    check("换成「只算 S 写回」的口径，渐近值变成 2d = 256 —— **口径会改变渐近值**",
          close(asy_write_only, 256.0, 1e-9), f"{asy_write_only:.1f}")
    check("两条渐近线不同 ⟹ 「朴素 AI 渐近于 d」这句必须带口径",
          not close(asy_naive, asy_write_only))
    check("flash 的 AI 随 S 线性增长（S=65536 时 = 32768）",
          close(attn_flops(65536, d, True) / flash_bytes(65536, d, esize), 32768.0))
    print()
    print("    ⟹ 三个结论，**必须分开讲**：")
    print("       ① 朴素 AI 渐近到 d=128 就停住了，与 S 无关——它确实 memory-bound；")
    print("       ② flash 的 AI = S/2 无上界，大 S 下 compute-bound；")
    print("       ③ flash 的真正收益是**绝对字节数**从 O(S²) 降到 O(S·d)：")
    print("          S=16384 时字节数降 65 倍，而 AI 的比值最多只能带来 "
          f"{H100_FP16_TFLOPS / (H100_BW_GBps * asy_naive / 1e3):.2f}× 的性能上界（§B2）。")
    print()
    print("    --- 口径错了会怎样（§4.1 的失败模式，逐条算给你看，S=16384）---")
    S_demo = 16384
    fl = attn_flops(S_demo, d, True)
    variants = (
        ("正经口径（朴素 4Sd+2S²，flash 4Sd）",
         (4.0 * S_demo * d + 2.0 * S_demo ** 2) * esize, 4.0 * S_demo * d * esize),
        ("错法 A：朴素漏掉 S 的**读回**",
         (4.0 * S_demo * d + 1.0 * S_demo ** 2) * esize, 4.0 * S_demo * d * esize),
        ("错法 B：flash 漏掉**写 O**（只算读 Q/K/V）",
         (4.0 * S_demo * d + 2.0 * S_demo ** 2) * esize, 3.0 * S_demo * d * esize),
        ("错法 C：A 与 B 同时犯",
         (4.0 * S_demo * d + 1.0 * S_demo ** 2) * esize, 3.0 * S_demo * d * esize),
    )
    for name, bn, bf in variants:
        print(f"      {name:<36} 朴素 AI {fl / bn:>7.1f} / flash AI {fl / bf:>8.1f}"
              f" / 字节比 {bn / bf:>6.2f}×")
    _b_ok = (4.0 * S_demo * d + 2.0 * S_demo ** 2) * esize / (4.0 * S_demo * d * esize)
    _b_a = (4.0 * S_demo * d + 1.0 * S_demo ** 2) * esize / (4.0 * S_demo * d * esize)
    _b_c = (4.0 * S_demo * d + 1.0 * S_demo ** 2) * esize / (3.0 * S_demo * d * esize)
    check(f"正经口径的字节比 = {_b_ok:.2f}×，错法 A（漏掉 S 的读回）= {_b_a:.2f}×"
          f"——**腰斩**，而两者的渐近 AI 从 128 变成 256",
          close(round(_b_ok, 2), 65.00, 1e-9) and close(round(_b_a, 2), 33.00, 1e-9))
    check(f"错法 C（两列同时错）={_b_c:.2f}× ⟹ 同一组形状下结论能差 "
          f"{_b_ok / _b_c:.2f} 倍", close(round(_b_c, 2), 44.00, 1e-9),
          f"{_b_ok / _b_c:.2f}×")
    check("三种错法的渐近值也跟着变（§4.1：口径必须写在每一个 AI 旁边）",
          not close(asy_naive, asy_write_only) and not close(_b_ok, _b_c))


def b2_ridge():
    sub("B2  roofline 拐点与两列 AI 的位置")
    i_fp16 = H100_FP16_TFLOPS * 1e12 / (H100_BW_GBps * 1e9)
    i_fp32 = H100_FP32_TFLOPS * 1e12 / (H100_BW_GBps * 1e9)
    print(f"    I*(FP16 tensor) = {H100_FP16_TFLOPS}e12 ÷ {H100_BW_GBps}e9 = "
          f"{i_fp16:.4f} FLOP/byte")
    print(f"    I*(FP32 向量)   = {H100_FP32_TFLOPS}e12 ÷ {H100_BW_GBps}e9 = "
          f"{i_fp32:.4f} FLOP/byte   ← attention 的 softmax 那一步走这条通路")
    check("I*(FP16) ≈ 295.4", close(i_fp16, 295.4, 1e-3), f"{i_fp16:.4f}")
    check("I*(FP32) ≈ 20.0", close(i_fp32, 20.0, 1e-3), f"{i_fp32:.4f}")
    print()
    d, esize = D_HEAD, ESIZE_FP16
    print(f"    {'S':>6}{'朴素 AI':>10}{'占 I* 比例':>12}{'朴素可达 TFLOPS':>18}"
          f"{'占峰值':>10}{'flash AI':>10}{'flash 可达':>12}")
    for S in (1024, 4096, 16384, 65536):
        bn, bf = naive_bytes(S, d, esize), flash_bytes(S, d, esize)
        fl = attn_flops(S, d, True)
        an, af = fl / bn, fl / bf
        reach_n = min(H100_FP16_TFLOPS, H100_BW_GBps * an / 1e3)
        reach_f = min(H100_FP16_TFLOPS, H100_BW_GBps * af / 1e3)
        print(f"    {S:>6}{an:>10.1f}{an / i_fp16 * 100:>11.1f}%{reach_n:>15.1f}"
              f"{reach_n / H100_FP16_TFLOPS * 100:>9.1f}%{af:>10.1f}{reach_f:>11.1f}")
    asy = 4.0 * d / (2.0 * esize)
    reach_asy = H100_BW_GBps * asy / 1e3
    print(f"      渐近（S→∞）{asy:>10.1f}{asy / i_fp16 * 100:>11.1f}%"
          f"{reach_asy:>15.1f}{reach_asy / H100_FP16_TFLOPS * 100:>9.1f}%"
          f"{'∞':>10}{H100_FP16_TFLOPS:>11.1f}")
    print()
    for Sx in (1024, 4096):
        af = attn_flops(Sx, D_HEAD, True) / flash_bytes(Sx, D_HEAD, ESIZE_FP16)
        print(f"    S={Sx} 时 flash 的 AI = {af:.1f}，是 I* 的 {af / i_fp16:.2f} 倍"
              f"（完全算力受限）")
    check("flash 的 AI 在 S=1024 时就已经越过拐点，S=4096 时是 I* 的 6.93 倍",
          close(round(attn_flops(4096, D_HEAD, True)
                      / flash_bytes(4096, D_HEAD, ESIZE_FP16) / i_fp16, 2), 6.93, 1e-9))
    check("朴素 AI 到 S=16384 时只有 I*(FP16) 的 42.7%（确实是 memory-bound）",
          close(round(attn_flops(16384, d, True) / naive_bytes(16384, d, esize)
                      / i_fp16 * 100, 1), 42.7, 1e-3))
    check("朴素 AI 的**渐近**值是 I*(FP16) 的 43.3%（S→∞ 也不越过拐点）",
          close(round(asy / i_fp16 * 100, 1), 43.3, 1e-3),
          f"{asy / i_fp16 * 100:.1f}%")
    print()
    print("    --- 单 head 单 batch 的两笔账（推导值）：算力 vs S 的写+读 ---")
    d1, es1 = D_HEAD, ESIZE_FP16
    print(f"    {'S':>7}{'前向纯算力':>14}{'S 的写 + 读':>15}{'比值':>9}"
          f"{'×256 份算力':>15}{'×256 份流量':>15}")
    for Sx in (1024, 4096, 16384, 65536):
        fl1 = attn_flops(Sx, d1, True)
        t_cmp = fl1 / (H100_FP16_TFLOPS * 1e12)
        t_s = 2.0 * Sx * Sx * es1 / (H100_BW_GBps * 1e9)
        print(f"    {Sx:>7}{t_cmp * 1e6:>11.1f}µs{t_s * 1e6:>12.1f}µs"
              f"{t_s / t_cmp:>9.3f}{t_cmp * 256 * 1e3:>12.1f}ms"
              f"{t_s * 256 * 1e3:>12.1f}ms")
    S1 = 16384
    t_cmp = attn_flops(S1, d1, True) / (H100_FP16_TFLOPS * 1e12)
    t_s = 2.0 * S1 * S1 * es1 / (H100_BW_GBps * 1e9)
    print(f"      比值恒为 {t_s / t_cmp:.3f} —— 与「AI 之比的上界」是**同一个数**")
    print(f"        （都是 峰值 ÷ (带宽 × d) = {H100_FP16_TFLOPS}e12 ÷ "
          f"({H100_BW_GBps}e9 × {d1})），与 S 无关")
    check("S=16384 时，光是 S 的写+读就比整个前向的算力时间多 2.3 倍",
          t_s / t_cmp > 2.0, f"{t_s / t_cmp:.2f}×")
    check("× 32 head × 8 batch（256 份）后：算力 35.6 ms、流量 82.1 ms、字节 274.9 GB",
          close(round(t_cmp * 256 * 1e3, 1), 35.6, 1e-3)
          and close(round(t_s * 256 * 1e3, 1), 82.1, 1e-3)
          and close(round(2.0 * S1 * S1 * es1 * 256 / 1e9, 1), 274.9, 1e-3),
          f"{t_cmp * 256 * 1e3:.1f} ms / {t_s * 256 * 1e3:.1f} ms / "
          f"{2.0 * S1 * S1 * es1 * 256 / 1e9:.1f} GB")
    check("S=4096 的两笔账：算力 8.7 µs、S 的写+读 20.0 µs（同样是 2.31 倍）",
          close(round(attn_flops(4096, d1, True) / (H100_FP16_TFLOPS * 1e12) * 1e6, 1),
                8.7, 1e-3)
          and close(round(2.0 * 4096 * 4096 * es1 / (H100_BW_GBps * 1e9) * 1e6, 1), 20.0,
                    1e-3),
          "S=1024 时是 0.5 µs / 1.3 µs——**这条比值与 S 无关**")
    print()
    print(f"    ⚠ 关键读数：朴素 AI 有上界 128，所以「flash 除以朴素」的 AI 之比"
          f"最多给出")
    print(f"      {H100_FP16_TFLOPS / reach_asy:.3f}× 的性能上界（= 峰值 ÷ "
          f"{reach_asy:.1f} TFLOPS），")
    print(f"      而同一组数里字节数的降幅是 65×（S=16384）。**两者不是一回事**——")
    print(f"      前者的上界与 S 无关，后者随 S 增长。这正是长序列收益更大的原因。")
    return i_fp16


def b3_s_materialization():
    sub("B3  S 矩阵物化到 HBM 的显存（**必须 × heads × batch**）")
    esize = ESIZE_FP16
    print(f"    口径：S 矩阵每 token 对 = {esize} B（fp16），单 head 单 batch。")
    print(f"    {'S':>7}{'S² · esize':>14}{'× 32 head':>14}{'× 32 head × 8 batch':>24}"
          f"{'占 80 GB 卡':>13}")
    rows = {}
    for S in (4096, 16384, 65536):
        one = S * S * esize
        h = one * 32
        b = h * 8
        rows[S] = (one, h, b)
        print(f"    {S:>7}{one / 1e6:>12.1f}MB{h / 1e9:>12.2f}GB{b / 1e9:>21.2f}GB"
              f"{b / (SP.HBM_CAPACITY_GB_BY_GEN[H100] * 1e9) * 100:>12.1f}%")
    one, h, b = rows[16384]
    check("S=16384：单 head 单 batch = 536.9 MB",
          close(round(one / 1e6, 1), 536.9, 1e-9), f"{one / 1e6:.1f} MB")
    check("S=16384：× 32 head = 17.18 GB", close(round(h / 1e9, 2), 17.18, 1e-9),
          f"{h / 1e9:.2f} GB")
    check("S=16384：× 32 head × 8 batch = 137.44 GB", close(round(b / 1e9, 2), 137.44, 1e-9),
          f"{b / 1e9:.2f} GB")
    check("S=16384 × 256 个头超过 80 GB 单卡 1.7 倍以上",
          b > 1.7 * SP.HBM_CAPACITY_GB_BY_GEN[H100] * 1e9,
          f"{b / (SP.HBM_CAPACITY_GB_BY_GEN[H100] * 1e9):.2f}×")
    print()
    print("    ⚠ 「单 head 单 batch 只要 536.9 MB」这句话是**对的但没用**：")
    print("      按字面核对会得出「536 MB 怎么会放不下」，而真实的前向要同时活下来")
    print("      的是 heads × batch 份 S。**报数时必须带上倍数。**")
    print(f"      另注：fp16 存 S 已经是最省的；fp32 存 S 是上面的 2 倍。")


def b4_flash_config():
    sub("B4  flash 前向的分块配置：SMEM 占用与 O 累加器的寄存器数")
    d, esize = D_HEAD, ESIZE_FP16
    print(f"    SMEM = (Br·d + 2·Bc·d)·esize   （Q tile + K tile + V tile）")
    print(f"    O 累加器寄存器 = Br·d ÷ 线程数   （fp32，每线程一份）")
    print()
    print(f"    {'Br':>4}{'Bc':>4}{'线程':>6}{'SMEM':>10}{'≤228KB':>8}"
          f"{'O 寄存器/线程':>14}{'双缓冲 KV':>11}{'双缓冲 ≤228KB':>14}")
    for Br, Bc, nthr in ((64, 64, 128), (128, 64, 128), (128, 64, 256),
                         (128, 128, 256), (256, 64, 256)):
        smem = (Br * d + 2 * Bc * d) * esize
        ok = smem <= SP.SMEM_PER_SM_MAX_KB * 1024
        regs_o = Br * d / nthr
        dbl = smem + 2 * Bc * d * esize
        ok2 = dbl <= SP.SMEM_PER_SM_MAX_KB * 1024
        print(f"    {Br:>4}{Bc:>4}{nthr:>6}{smem / 1024:>8.0f}KB"
              f"{('是' if ok else '否'):>8}{regs_o:>14.0f}{dbl / 1024:>9.0f}KB"
              f"{('是' if ok2 else '否'):>14}")
    smem_128_64 = (128 * d + 2 * 64 * d) * esize
    smem_128_128 = (128 * d + 2 * 128 * d) * esize
    dbl_128_128 = smem_128_128 + 2 * 128 * d * esize
    check("Br=128/Bc=64 的 SMEM = 64 KB ≤ 228 KB",
          close(smem_128_64 / 1024, 64.0, 1e-9), f"{smem_128_64 / 1024:.0f} KB")
    check("Br=128/Bc=128 的 SMEM = 96 KB ≤ 228 KB",
          close(smem_128_128 / 1024, 96.0, 1e-9), f"{smem_128_128 / 1024:.0f} KB")
    check("Br=128/Bc=128 把 KV 双缓冲后是 160 KB，仍然 ≤ 228 KB",
          close(dbl_128_128 / 1024, 160.0, 1e-9)
          and dbl_128_128 <= SP.SMEM_PER_SM_MAX_KB * 1024,
          f"{dbl_128_128 / 1024:.0f} KB")
    check("4 个 warp（128 线程）扛 Br=128、d=128 时，光 O 就要 128 个寄存器/线程",
          close(128 * d / 128, 128.0, 1e-9), "128")
    print()
    print("    ⚠ O 的寄存器数与 Br·d 成正比，与**分块无关**——这是 flash 前向最硬的约束：")
    print("      Br·d ÷ 256 线程 = 64 个寄存器/线程（Bc 可以随便调，Br 不行）。")
    print(f"      每 SM 寄存器文件 {SP.REGISTERS_PER_SM} 个 32-bit 寄存器；")
    print(f"      256 线程 × 64 个 = {256 * 64} 个，还剩 "
          f"{(SP.REGISTERS_PER_SM - 256 * 64) / SP.REGISTERS_PER_SM * 100:.0f}% 给"
          f" S 的 fragment、m/l 与地址。")


def b5_backward_recompute():
    sub("B5  反向：重算 S 多花的 FLOPs vs 省下的显存与流量")
    d, esize = D_HEAD, ESIZE_FP16
    S = 16384
    fwd = attn_flops(S, d, True)
    recompute = 2.0 * S * S * d                     # 重算一次 S = QKᵀ
    bw_core = 8.0 * S * S * d                       # dP/dV/dQ/dK 四个矩阵乘
    print(f"    前向 FLOPs = 4·S²·d = {fwd:.3e}")
    print(f"    反向的四个矩阵乘 = 8·S²·d = {bw_core:.3e}（= 前向的 2 倍）")
    print(f"    不存 P 而要重算 S 的额外开销 = 2·S²·d = {recompute:.3e}"
          f"（= 前向的 0.5 倍）")
    print(f"    ⟹ 反向总 FLOPs ≈ 10·S²·d = {bw_core + recompute:.3e}，"
          f"是前向的 {(bw_core + recompute) / fwd:.1f} 倍")
    saved_bytes = 2.0 * S * S * esize               # 写 P + 读 P
    t_recompute = recompute / (H100_FP16_TFLOPS * 1e12)
    t_traffic = saved_bytes / (H100_BW_GBps * 1e9)
    print(f"    反过来省下的：P 的写 + 读 = 2·S²·esize = {saved_bytes / 1e9:.3f} GB"
          f"（单 head 单 batch）")
    print(f"    两者的时间（推导值，单 head 单 batch）：")
    print(f"      重算 = {recompute:.3e} ÷ {H100_FP16_TFLOPS}e12 = "
          f"{t_recompute * 1e6:.1f} µs")
    print(f"      搬运 = {saved_bytes / 1e9:.3f} GB ÷ {H100_BW_GBps} GB/s = "
          f"{t_traffic * 1e6:.1f} µs")
    print(f"      ⟹ 重算便宜 {t_traffic / t_recompute:.2f} 倍")
    check("重算 S 比把 P 搬一趟 HBM 便宜（比值 > 2）",
          t_traffic / t_recompute > 2.0, f"{t_traffic / t_recompute:.2f}×")
    check(f"32 head 时的总账：重算 {recompute * 32 / 1e12:.2f} TFLOP vs "
          f"搬运 {saved_bytes * 32 / 1e9:.1f} GB",
          close(recompute * 32 / 1e12, 2.20, 1e-2) and
          close(saved_bytes * 32 / 1e9, 34.36, 1e-3),
          f"{recompute * 32 / 1e12:.2f} TFLOP / {saved_bytes * 32 / 1e9:.2f} GB")
    print()
    print("    ⚠ 这里只算了「重算 QKᵀ」那一项。重算还要再做一遍 exp（S² 次超越函数），")
    print(f"      它不占 tensor core，走的是 FP32 向量通路（每 SM "
          f"{H100_FP32_TFLOPS / SM_COUNT * 1e12:.3e} FLOP/s）——")
    print(f"      但 exp 不是 FMA，本仓库**没有**它的吞吐规格，正文不做这一步的算术。")
    return S, recompute, saved_bytes


def b6_dq_accumulation():
    sub("B6  dQ 跨 KV 块累加的两种方式：原子（读改写） vs workspace（多写一份）")
    Sq, d = 2048, D_HEAD
    esize_f32 = SP.FLOAT32_BYTES
    print(f"    口径：一个 Q 块 {Sq} 行 × d={d}（fp32 累加器），KV 分成 n_split 段。")
    print(f"      atomic     ：每个 (Q 块, KV 段) 对 dQ 做一次读-改-写")
    print(f"                   → 2·n_split·Sq·d·4 B")
    print(f"      workspace  ：写 n_split 份部分和、再读回来归约、最后写一份")
    print(f"                   → (2·n_split + 1)·Sq·d·4 B")
    print()
    print(f"    {'n_split':>8}{'atomic 字节':>14}{'workspace 字节':>16}"
          f"{'差':>10}{'atomic 时间@3350GB/s':>22}")
    for ns in (1, 2, 4, 8, 16):
        a = 2.0 * ns * Sq * d * esize_f32
        w = (2.0 * ns + 1) * Sq * d * esize_f32
        t = a / (H100_BW_GBps * 1e9)
        print(f"    {ns:>8}{a / 1e6:>12.1f}MB{w / 1e6:>14.1f}MB"
              f"{(w - a) / 1e6:>8.2f}MB{t * 1e6:>19.1f}µs")
        check(f"n_split={ns}：两种方式的字节数只差一份 dQ（{Sq * d * esize_f32 / 1e6:.1f} MB）",
              close((w - a) / 1e6, Sq * d * esize_f32 / 1e6, 1e-9),
              f"{(w - a) / 1e6:.2f} MB")
    ns = 8
    a = 2.0 * ns * Sq * d * esize_f32
    print()
    print(f"    ⟹ 流量上两者几乎一样（差一份 dQ = {Sq * d * esize_f32 / 1e6:.2f} MB）；")
    print(f"      n_split=8 时 atomic 一侧是 {a / 1e6:.1f} MB ≈ "
          f"{a / (H100_BW_GBps * 1e9) * 1e6:.1f} µs，")
    lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
    print(f"      而 workspace 一侧还要多一次归约 kernel 的 launch："
          f"{lo}~{hi} µs（引用值，本仓库不复核其测量口径）。")
    check("n_split=8 时，归约 kernel 的 launch 开销（3~5 µs）与一次 dQ 的搬运时间同量级",
          0.3 < (lo / (a / (H100_BW_GBps * 1e9) * 1e6)) < 3.0,
          f"{lo / (a / (H100_BW_GBps * 1e9) * 1e6):.2f}×")
    print("      **所以真正的差别不是流量，是确定性**：atomic 的加法顺序由硬件决定，")
    print("      换个 grid 划分或换张卡就可能得到不同的位模式（§C3）。")


def b7_warp_specialization():
    sub("B7  warp specialization 的分区：SMEM 阶段数 / 寄存器 / wgmma 指令数")
    d, esize = D_HEAD, ESIZE_FP16
    print(f"    consumer = 1 个 warpgroup（{SP.WARPGROUP_WARPS * SP.WARP_SIZE} 线程）"
          f"发 wgmma；producer = 同样多的线程搬数据。")
    print(f"    SMEM = stages·(K tile + V tile)·esize = stages·2·Bc·d·esize")
    print()
    print(f"    {'Bc':>5}{'stages':>8}{'SMEM':>10}{'≤228KB':>9}{'每 SM 可驻留 CTA':>18}")
    for Bc, stages in ((128, 2), (128, 3), (128, 4), (64, 4), (64, 6)):
        smem = stages * 2 * Bc * d * esize
        ok = smem <= SP.SMEM_PER_SM_MAX_KB * 1024
        ncta = int(SP.SMEM_PER_SM_MAX_KB * 1024 // smem) if smem else 0
        print(f"    {Bc:>5}{stages:>8}{smem / 1024:>8.0f}KB{('是' if ok else '否'):>9}"
              f"{ncta:>18}")
        if not ok:
            check(f"Bc={Bc}、{stages} 级流水超出 {SP.SMEM_PER_SM_MAX_KB} KB 上限"
                  f"（必须减 stage 或减 Bc）", smem > SP.SMEM_PER_SM_MAX_KB * 1024,
                  f"{smem / 1024:.0f} KB")
    check("Bc=128、4 级流水 = 256 KB 超过 228 KB 上限（最多 3 级）",
          (4 * 2 * 128 * d * esize) > SP.SMEM_PER_SM_MAX_KB * 1024,
          f"{4 * 2 * 128 * d * esize / 1024:.0f} KB")
    print()
    print(f"    wgmma 指令数（M 恒为 64，N ≤ 256、步长 8，见 specs.py 的 WGMMA_*）：")
    Br, Bc = 128, 128
    n_qk = (Br // 64) * (Bc // 128) * (d // 16)
    n_pv = (Br // 64) * (d // 128) * (Bc // 16)
    print(f"      QKᵀ：Br×Bc×d → (Br/64)·(Bc/128)·(d/16) = "
          f"{Br // 64}·{Bc // 128}·{d // 16} = {n_qk} 条 .m64n{Bc}k16")
    print(f"      PV ：Br×Bc×d → (Br/64)·(d/128)·(Bc/16) = "
          f"{Br // 64}·{d // 128}·{Bc // 16} = {n_pv} 条 .m64n{d}k16")
    check("QKᵀ 每个 (Br=128, Bc=128) tile 需要 16 条 wgmma", n_qk == 16, f"{n_qk}")
    check("PV 每个 tile 也需要 16 条 wgmma", n_pv == 16, f"{n_pv}")
    print()
    print(f"    ⚠ wgmma 的发射粒度是**整个 warpgroup**（规格值："
          f"WGMMA_ALIGNED_QUALIFIER_MANDATORY），")
    print(f"      所以分工出去做访存的 warp **不能**与发 wgmma 的 warp 混在同一个")
    print(f"      warpgroup 里——这是 warp specialization 在 Hopper 上的硬边界。")
    check("wgmma 需要 sm_90a（进 HOPPER_TARGETS，不混进默认 TARGETS）",
          SP.FEATURE_FIRST_TARGET["wgmma"][1] == "sm_90a"
          and "sm_90a" in SP.ACCELERATED_TARGETS)
    print()
    wg_threads = SP.WARPGROUP_WARPS * SP.WARP_SIZE
    print(f"    PV 的输出（Br×d 的 fp32 累加器）住在寄存器里：")
    print(f"      Br={Br}、d={d} → {Br * d} 个 fp32 ÷ {wg_threads} 线程 = "
          f"{Br * d / wg_threads:.0f} 个寄存器/线程")
    check("Br=128、d=128 时 O 累加器每线程 128 个寄存器（已到 255 上限的一半）",
          close(Br * d / wg_threads, 128.0, 1e-9), "128")


def b8_gqa_traffic():
    sub("B8  GQA / MQA 的 KV 流量：随 heads_kv / heads_q 线性下降")
    d, esize = D_HEAD, ESIZE_FP16
    h_q, S = 32, 4096
    kv_bytes = lambda h: 2.0 * S * d * h * esize       # K 与 V 各 S×d 个元素
    print(f"    口径：heads_q = {h_q}，d_head = {d}，序列长 {S}，fp16，batch = 1。")
    print(f"    KV 字节 = 2（K 与 V）·S·d·heads_kv·esize")
    print()
    print(f"    {'heads_kv':>10}{'KV 头':>8}{'KV 字节':>12}{'相对 MHA':>11}"
          f"{'每 token 读取':>14}")
    base = kv_bytes(h_q)
    for h in (32, 16, 8, 4, 2, 1):
        b = kv_bytes(h)
        print(f"    {h:>10}{h / h_q:>8.3f}{b / 1e6:>10.1f}MB{h / h_q:>11.3f}"
              f"{b / 1e6:>12.1f}MB")
        check(f"heads_kv={h}：流量 = MHA 的 {h / h_q:.3f}（= heads_kv/heads_q）",
              close(b / base, h / h_q, 1e-12), f"{b / base:.4f}")
    print()
    print(f"    MQA（heads_kv = 1）把 KV 流量压到 1/{h_q} = {1 / h_q:.4f}，"
          f"即 {kv_bytes(1) / 1e6:.2f} MB。")
    check("MQA 的 KV 字节 = MHA 的 1/32", close(kv_bytes(1) / base, 1 / 32, 1e-12),
          f"{kv_bytes(1) / base:.5f}")
    print(f"    ⚠ 减少的是**读取**字节数，不是 FLOPs：注意力分数照样要对着每个 KV 位置算一遍，")
    print(f"      只是这些 KV 被 rep = heads_q/heads_kv 个 query 头共用（§A4）。")


def b9_decode_ai():
    sub("B9  decode 阶段的算术强度：**与 S 和 d 都无关**")
    esize = ESIZE_FP16
    print(f"    每生成一个 token（batch = 1）：")
    print(f"      FLOPs = 4·S·d_head·heads_q        （QKᵀ 与 PV 各 2·S·d·heads_q）")
    print(f"      字节  = 2·S·d_head·heads_kv·esize （K 与 V 各读一遍）")
    print(f"      ⟹ AI = 4·S·d·h_q / (2·S·d·h_kv·esize) = 2 / (esize · r)，"
          f"r = heads_kv/heads_q")
    print(f"         = 1/r（fp16，esize = {esize}）—— **S 与 d 都被约掉了**")
    print()
    S, d, h_q = 4096, D_HEAD, 32
    print(f"    具体数（S = {S}，d_head = {d}，heads_q = {h_q}，batch = 1）：")
    print(f"    {'heads_kv':>10}{'r':>8}{'AI':>8}{'KV 字节/token':>15}"
          f"{'时间@3350GB/s':>15}{'FLOPs/token':>14}{'算力时间':>12}")
    for h_kv in (32, 8, 1):
        r = h_kv / h_q
        ai = 2.0 / (esize * r)
        kvb = 2.0 * S * d * h_kv * esize
        fl = 4.0 * S * d * h_q
        t_mem = kvb / (H100_BW_GBps * 1e9)
        t_cmp = fl / (H100_FP16_TFLOPS * 1e12)
        print(f"    {h_kv:>10}{r:>8.3f}{ai:>8.1f}{kvb / 1e6:>12.1f}MB"
              f"{t_mem * 1e6:>12.1f}µs{fl / 1e6:>11.1f}M{t_cmp * 1e9:>10.1f}ns")
        check(f"heads_kv={h_kv}：AI = 1/r = {1 / r:.1f}（S 与 d 无关）",
              close(ai, 1 / r, 1e-12), f"{ai:.2f}")
    h_kv, r = 32, 1.0
    ai = 2.0 / (esize * r)
    i_fp16 = H100_FP16_TFLOPS * 1e12 / (H100_BW_GBps * 1e9)
    reach = min(H100_FP16_TFLOPS, H100_BW_GBps * ai / 1e3)
    print("    同一台机器上的 prefill 对照（单 head，整段序列一次算完）：")
    print(f"      {'S':>7}{'prefill FLOPs':>16}{'prefill AI':>12}{'decode 字节/token':>19}"
          f"{'decode 时间/token':>19}")
    for Sx in (4096, 16384, 32768):
        fl_pf = 4.0 * Sx * Sx * D_HEAD
        bn_pf = naive_bytes(Sx, D_HEAD, esize)
        kvb_dc = 2.0 * Sx * D_HEAD * 32 * esize
        print(f"      {Sx:>7}{fl_pf:.3e}{fl_pf / bn_pf:>12.1f}"
              f"{kvb_dc / 1e6:>16.1f}MB{kvb_dc / (H100_BW_GBps * 1e9) * 1e6:>16.1f}µs")
    check("prefill 与 decode 的 AI 差 120 倍（120.5 对 1.0）——同一个算子的两种形态",
          close(round(attn_flops(4096, D_HEAD, True)
                      / naive_bytes(4096, D_HEAD, esize), 1), 120.5, 1e-3))
    check("decode 的每 token 字节随 S 线性增长，AI 却不变（S=32768 仍是 1.0）",
          close(2.0 * 32768 * D_HEAD * 32 * esize / 1e6, 536.9, 1e-3)
          and close(2.0 / (esize * (32 / 32)), 1.0, 1e-12),
          "536.9 MB/token @ 160.3 µs")
    print()
    print(f"    MHA 的 decode：AI = {ai:.1f} FLOP/byte，只有 I*(FP16) = {i_fp16:.1f} 的 "
          f"{ai / i_fp16 * 100:.2f}%")
    print(f"      可达算力 = {H100_BW_GBps} GB/s × {ai:.1f} = {reach:.2f} TFLOPS "
          f"= 峰值的 {reach / H100_FP16_TFLOPS * 100:.2f}%")
    check("MHA decode 的 AI = 1.0，只有 I*(FP16) 的 0.34%（极端 memory-bound）",
          close(ai, 1.0, 1e-12) and close(round(ai / i_fp16 * 100, 2), 0.34, 1e-9),
          f"{ai / i_fp16 * 100:.2f}%")
    kvb = 2.0 * S * d * 32 * esize
    fl = 4.0 * S * d * 32
    ratio = (kvb / (H100_BW_GBps * 1e9)) / (fl / (H100_FP16_TFLOPS * 1e12))
    print(f"    每 token 的访存与算力时间之比 = {ratio:.1f}×  ← decode 的全部矛盾")
    print(f"      这个比值不是巧合：它 = (字节/BW) ÷ (FLOPs/峰值) = I*/AI = "
          f"{i_fp16:.1f}/{ai:.1f}")
    check("MHA decode 的访存时间是算力时间的 295 倍（= I*/AI）",
          close(ratio, i_fp16 / ai, 1e-9) and ratio > 200, f"{ratio:.1f}×")
    print(f"    ⟹ 这个比值**与 S、d、batch 都无关**（都约掉了），只看 heads_kv/heads_q。")
    print(f"      要动它，只有三条路：减 KV 头数（GQA/MQA，§B8）、减 KV 的位宽（量化）、")
    print(f"      或把 KV 读的**次数**降下来（prefix caching 一类）。")


def b10_paged_kv():
    sub("B10  paged KV：页表间接寻址的延迟与碎片")
    d, esize = D_HEAD, ESIZE_FP16
    S, h_kv, page = 16384, 32, 16
    page_bytes = page * d * esize
    print(f"    口径：页大小 = {page} 个 token，d_head = {d}，fp16")
    print(f"    一页一个 head 的字节 = {page}×{d}×{esize} = {page_bytes / 1024:.0f} KB"
          f"（= {page_bytes // SP.CACHE_LINE_BYTES} 个 cache line / "
          f"{page_bytes // SP.SECTOR_BYTES} 个 sector）")
    print(f"    一页的计算量 = 2·{page}·{d} = {2 * page * d} 次乘加"
          f"（对比一个 Bc=64 的 tile：{2 * 64 * d} 次）")
    print(f"    页表项 = 4 B/页；序列长 {S} → {S // page} 项 = "
          f"{S // page * 4 / 1024:.0f} KB 页表（全部 KV 头共用一张）")
    kv_total = 2.0 * S * d * h_kv * esize
    print(f"    KV 本体 = 2·{S}·{d}·{h_kv}·{esize} = {kv_total / 1e6:.1f} MB；"
          f"页表 {S // page * 4 / 1024:.0f} KB = 它的 "
          f"{S // page * 4 / kv_total * 100:.4f}%")
    check("页表本身的字节数可以忽略（< 0.01%）",
          S // page * 4 / kv_total < 1e-4, f"{S // page * 4 / kv_total * 100:.4f}%")
    print()
    print(f"    碎片：最后不满一页的期望浪费 = 页大小/2 = {page / 2:.0f} 个 token")
    waste = (page / 2) * 2 * d * h_kv * esize
    print(f"      浪费字节 = {page / 2:.0f}×2×{d}×{h_kv}×{esize} = "
          f"{waste / 1e6:.2f} MB，占 KV 本体的 {waste / kv_total * 100:.3f}%")
    check("页内碎片 < 0.1%（碎片不是 paged KV 的代价所在）",
          waste / kv_total < 1e-3, f"{waste / kv_total * 100:.3f}%")
    print()
    print(f"    真正的代价是**间接寻址的延迟**：")
    Br, Bc = 128, 64
    n_pages_per_tile = math.ceil(Bc / page)
    flops_tile = 2.0 * Br * Bc * d
    t_tile = flops_tile / (H100_FP16_TFLOPS * 1e12 / SM_COUNT)
    lat_lo, lat_hi = SP.HBM_LATENCY_CYCLES_RANGE
    print(f"      一个 KV tile = {Bc} 行 = {n_pages_per_tile} 页 → "
          f"{n_pages_per_tile} 次页表查询（每次都是一个有依赖的沿指针取数）")
    print(f"      这个 tile 的 QKᵀ 计算 = 2·{Br}·{Bc}·{d} = {flops_tile / 1e6:.1f} MFLOP")
    print(f"        ÷ 每 SM {H100_FP16_TFLOPS / SM_COUNT:.1f} TFLOPS = "
          f"{t_tile * 1e9:.0f} ns ≈ {t_tile * 1e9 * 1.8:.0f} 个周期（按 1.8 GHz）")
    print(f"      而一次片外访存的往返 = {lat_lo}~{lat_hi} 周期（规格值）")
    check("页表查询的延迟与一个 tile 的计算时间同量级 ⟹ 必须预取，不能现取现用",
          0.5 < (lat_lo / (t_tile * 1e9 * 1.8)) < 4.0,
          f"{lat_lo / (t_tile * 1e9 * 1.8):.2f}×")
    print(f"      搬运粒度：一个 tile 跨 {n_pages_per_tile} 页 → "
          f"{n_pages_per_tile} 次 bulk copy 而不是 1 次")
    check("tile 跨多页时 bulk copy 的次数 = ceil(Bc/页大小)",
          n_pages_per_tile == math.ceil(Bc / page), f"{n_pages_per_tile}")
    print()
    print(f"    ⚠ 回链 20-gpu-architecture/memory-system/02：只要**页内**地址连续，")
    print(f"      sector 利用率就是 100%（一页 {page_bytes} B = "
          f"{page_bytes // SP.SECTOR_BYTES} 个整 sector）。")
    print(f"      碎片化发生在**页的边界**，不在页内——所以代价在延迟与搬运粒度上，")
    print(f"      不在字节浪费上。")


def b11_window_variants():
    sub("B11  变体（sliding window）：只改变分块的**范围**，不改变任何一条算式")
    d, esize, Bc = D_HEAD, ESIZE_FP16, 128
    print(f"    滑窗窗口 w 个 token，KV 分块 {Bc}：每行 query 只需要 "
          f"ceil(w/Bc) 个 KV 块，")
    print(f"    而不是全部 S/Bc 个（因果掩码下平均还要再砍一半）。")
    print()
    print(f"    {'S':>7}{'w':>7}{'全量块数':>10}{'滑窗块数':>10}{'KV 字节降幅':>13}")
    for S, w in ((16384, 4096), (16384, 1024), (65536, 4096), (65536, 1024)):
        full = math.ceil(S / Bc)
        win = math.ceil(w / Bc)
        print(f"    {S:>7}{w:>7}{full:>10}{win:>10}{full / win:>12.0f}×")
        check(f"S={S}、w={w}：块数从 {full} 降到 {win}（{full / win:.0f}×）",
              win < full, f"{win} / {full}")
    print(f"    窗口 = S 时没有收益（w=S 退化成全量），所以滑窗只在 w < S 时有意义。")
    print(f"    ⚠ 窗口只把 KV 的**读取范围**截短（= visits 变少），"
          f"算式、状态、重标定全都照旧；")
    print(f"      ALiBi / softcap 则是在**算完 S 之后**再加一项/过一次 tanh——")
    print(f"      它们改变的是 exp 的输入，不改变分块的形状。三者对 kernel 结构的")
    print(f"      影响各不相同，正文只讲这一层。")


# ===========================================================================
# §C 边界与反例
# ===========================================================================


def c1_no_max_subtraction():
    sub("C1  不减 max：fp16 下真的得到 inf / nan")
    rng = np.random.default_rng(20260920)
    S = 64
    logits = (rng.standard_normal(S) * 10.0).astype(np.float32)
    logits[0] = 30.0                                  # 故意放一个大 logit
    print(f"    logits：{S} 个值，N(0,10) 外加一个 30.0；"
          f"上溢阈值 fp16 = {FP16_OVERFLOW_LN:.4f}，fp32 = {FP32_OVERFLOW_LN:.4f}")

    def softmax_no_max(x, dt):
        e = exp_dt(x, dt)
        return e / np.sum(e, dtype=dt)

    for dt, name in ((FP16, "fp16"), (FP32, "fp32")):
        p = softmax_no_max(logits, dt)
        e = exp_dt(logits, dt)
        print(f"    {name} 不减 max：exp 那一列含 inf = {has_inf(e)}，"
              f"归一到 [0,1] 之后含 nan = {has_nan(p)}，"
              f"有限值之和 = {np.nansum(np.asarray(p, np.float64)):.4f}")
        if dt == FP16:
            print(f"      （exp(30) 在 fp16 里 = {exp_dt(np.float32(30.0), FP16)}；"
                  f"上溢之后 inf/inf = nan）")
    p16 = softmax_no_max(logits, FP16)
    check("fp16 不减 max：exp(30) 直接上溢，结果里出现 inf 或 nan",
          has_inf(p16) or has_nan(p16))
    p32 = softmax_no_max(logits, FP32)
    check("fp32 在同样的 30.0 上**不**溢出——溢出是格式问题，不是算法问题",
          not has_inf(p32) and not has_nan(p32))
    logits_big = logits.copy()
    logits_big[0] = 100.0
    p32b = softmax_no_max(logits_big, FP32)
    check("fp32 在 100.0 上也会溢出（阈值 88.7228）",
          has_inf(p32b) or has_nan(p32b))
    p_ok = softmax_rows(logits.astype(np.float64)[None, :], FP64)[0]
    print(f"    减去 max 之后（fp16 也算得动）：每行权重之和 = "
          f"{float(np.sum(_q(p_ok, FP16))):.8f}")
    check("减去 max 后 fp16 也能算（这正是 flash 必须在内层做在线 max 的原因）",
          not has_inf(_q(p_ok, FP16)) and not has_nan(_q(p_ok, FP16)))


def c2_rescale_underflow():
    sub("C2  rescale 因子在 fp16 状态下归零：分块贡献被静默丢弃")
    print("    α = exp(m_old − m_new)，gap = m_new − m_old ≥ 0。")
    print("    **同一件事有两个口径**，先分开再报数：")
    print("      口径一（精确）：gap 用精确算术，只有 exp 的**结果**落进 fp16；")
    print("      口径二（全 fp16）：gap 本身就住在 fp16 里（状态存 half 时就是这样），")
    print("        于是 gap 先被舍到 fp16 的刻度上，再算 exp。")
    print("    ⚠ 出库的 17.3287 是**口径一**的值。两个口径的阈值差一个 fp16 ulp。")
    print()

    def a_exact(gap, dt):
        """口径一：gap 精确，只有结果落进 dt。"""
        return float(np.asarray(np.exp(np.float64(-gap)), dtype=dt))

    def a_full(gap):
        """口径二：gap 也过一遍 fp16。"""
        return float(exp_dt(np.float16(-gap), FP16))

    print(f"    {'gap':>12}{'口径一 fp16':>15}{'口径二 fp16':>15}"
          f"{'fp32 状态':>15}{'fp64':>16}")
    for gap in (10.0, 17.0, 17.328125, 17.3287, 17.34375, 20.0, 104.0):
        print(f"    {gap:>12.6f}{a_exact(gap, FP16):>15.6e}{a_full(gap):>15.6e}"
              f"{a_exact(gap, FP32):>15.6e}{math.exp(-gap):>16.6e}")
    check("口径一：gap = 17.3287 时 fp16 的结果**恰好归零**（= 2^-25，舍入到偶）",
          a_exact(17.3287, FP16) == 0.0 and a_exact(17.3287, FP32) > 0.0,
          f"fp16={a_exact(17.3287, FP16):.0e}, fp32={a_exact(17.3287, FP32):.3e}")
    check("口径二：同样的 gap 在 fp16 状态下**不归零**（gap 被舍到 17.328125，"
          "恰好越过阈值）", a_full(17.3287) > 0.0, f"{a_full(17.3287):.3e}")
    check("口径二下第一个归零的 fp16 刻度是 17.34375（= 1110 × 2^-6）",
          a_full(17.34375) == 0.0 and a_full(17.328125) > 0.0,
          f"17.328125 → {a_full(17.328125):.3e}, 17.34375 → {a_full(17.34375):.0e}")
    check("两个口径的阈值不同（精确 17.3287 vs fp16 刻度 17.34375，差一个 ulp）",
          not close(a_exact(17.3287, FP16), a_full(17.3287)), "报数时必须注明口径")
    check("gap = 104 时 fp32 也归零（阈值 103.9721，真实 logits 里不可达）",
          a_exact(104.0, FP32) == 0.0 and a_exact(104.0, FP16) == 0.0)
    check("gap ≤ 17 时 fp16 不归零（只有大跳才会出事）",
          a_full(17.0) > 0.0 and a_exact(17.0, FP16) > 0.0)

    # attention 里的具体后果：与 softmax-normalization/02 §C5 同一个例子
    print()
    print("    在 attention 里的形态（与 softmax-normalization/02 §C5 同一个例子，")
    print("      区别是丢掉的量从「一个计数 l」变成「一个已经算好的输出分量 O」）：")
    n1, dd = 60000, 2
    Q = np.array([[1.0, 0.0]])             # q 必须非零，否则 logits 全是 0
    K = np.zeros((n1 + 1, dd))
    K[n1] = np.array([20.0, 0.0])          # 第二块只有 1 个 key，logit = 20
    V = np.zeros((n1 + 1, dd))
    V[:n1] = np.array([1.0, 0.0])
    V[n1] = np.array([0.0, 1.0])
    scale = 1.0
    O64, _, l64 = attn_online(Q, K, V, scale, n1)
    O32, _, l32 = attn_online(Q, K, V, scale, n1, state_dt=FP32)
    O16, _, l16 = attn_online(Q, K, V, scale, n1, state_dt=FP16)
    print(f"      数据：第一块 {n1} 个 logit 全为 0（V = [1,0]），"
          f"第二块 1 个 logit = 20（V = [0,1]）")
    print(f"      真值  ：l = {n1}·exp(−20) + 1 = {l64[0]:.10f}，"
          f"O = [{O64[0, 0]:.10e}, {O64[0, 1]:.10f}]")
    print(f"      fp32 状态：l = {l32[0]:.10f}，O = "
          f"[{O32[0, 0]:.10e}, {O32[0, 1]:.10f}]")
    print(f"      fp16 状态：l = {l16[0]:.10f}，O = "
          f"[{O16[0, 0]:.10e}, {O16[0, 1]:.10f}]")
    check("真值里那个由第一块贡献的分量非零（= 60000·exp(−20)/l）",
          O64[0, 0] > 1e-5, f"{O64[0, 0]:.6e}")
    check("fp16 状态下这个分量**恰好变成 0**——整段贡献被静默丢弃，没有任何警告",
          O16[0, 0] == 0.0, f"{O16[0, 0]:.1e}")
    check("fp32 状态给出正确的分量（与真值的差 < 1e-10）",
          abs(O32[0, 0] - O64[0, 0]) < 1e-10, f"{O32[0, 0]:.10e}")
    check("但按整个输出向量看，fp16 的误差仍在 2e-3 以内（有上界，不是灾难）",
          relerr_norm(O16, O64) < 2e-3, f"{relerr_norm(O16, O64):.3e}")
    drop_bound = SP.FP16_MAX_FINITE * 2.0 ** -25
    print(f"    丢掉的量有上界：|ΔO| ≤ (max|v|)·l_old·α ≤ max|v|·65504·2^-25 = "
          f"{drop_bound:.6e}·max|v|")
    print(f"      算出的按范数相对误差 {relerr_norm(O16, O64):.2e} 正好落在这个界里。")
    print(f"      l 那一侧的界是 {drop_bound:.6e}，对新块的 l ≥ 1 是 "
          f"{drop_bound * 100:.4f}%。")
    check("一次丢弃的上界 1.952171e-03（与 softmax-normalization/02 §C5 同源）",
          close(round(drop_bound, 9), 1.952171e-3, 1e-6), f"{drop_bound:.6e}")
    print(f"    ⟹ 规避：状态 (m, l, **O**) 一律住 fp32。fp32 的阈值是 "
          f"{abs(FP32_UNDERFLOW_LN):.4f}，真实 logits 不可达。")
    print(f"      注意 O 的丢弃比 l 的丢弃隐蔽：l 少一点还会在分母上体现出来，")
    print(f"      而 O 的某个分量直接归零，输出看上去完全正常。")


def c3_dq_bit_patterns(res):
    sub("C3  dQ 的 atomic 版与顺序版：数值一致，但**位模式不同**")
    seq = res["sequential"][0]
    rev = res["reverse"][0]
    tree = res["tree"][0]
    b_seq, b_rev, b_tree = bits(seq), bits(rev), bits(tree)
    n_diff_sr = int(np.sum(b_seq != b_rev))
    n_diff_st = int(np.sum(b_seq != b_tree))
    n_tot = b_seq.size
    print(f"    dQ 共 {n_tot} 个元素（fp32）：")
    print(f"      顺序版 vs 反序版（模拟 atomic）：{n_diff_sr} 个元素的位模式不同")
    print(f"      顺序版 vs 两两归约版（workspace+固定顺序）：{n_diff_st} 个元素不同")
    check("atomic 版与顺序版的位模式**确实不同**（浮点加法不满足结合律）",
          n_diff_sr > 0, f"{n_diff_sr}/{n_tot}")
    check("换一种确定性的归约顺序（两两）同样会改变位模式",
          n_diff_st > 0, f"{n_diff_st}/{n_tot}")
    print(f"      数值上按范数归一的最大差 = {relerr_norm(rev, seq):.3e}"
          f"（逐元素口径 {relerr(rev, seq):.3e}，那个大数来自某个本该≈0 的元素）")
    check("但两者的**数值**一致（按范数 < 1e-6）",
          relerr_norm(rev, seq) < 1e-6, f"{relerr_norm(rev, seq):.3e}")
    print()
    print("    ⚠ 注意口径：本程序用「换一个累加顺序」模拟 atomicAdd 的不确定顺序。")
    print("      它证明的是「不同顺序 ⟹ 位模式不同」这一事实，不是 atomic 硬件本身。")
    print("      真实 atomicAdd 的顺序由硬件运行时决定，同一份输入两次运行就可能不同——")
    print("      这是 §4 里「结果不可复现」那条的全部内容。")
    print(f"      想复现，就必须走 workspace + 固定顺序归约的那条路（确定性的那一侧）。")


def c4_forget_rescale_O():
    sub("C4  忘了重标定 O（只重标定 l）：**静默算错，不报错也不溢出**")
    rng = np.random.default_rng(20260922)
    Sq, Sk, d = 8, 64, 8
    scale = 1.0 / math.sqrt(d)
    Q = rng.standard_normal((Sq, d))
    K = rng.standard_normal((Sk, d))
    V = rng.standard_normal((Sk, d))
    O_ref, _, _ = attn_naive(Q, K, V, scale)
    O_ok, _, _ = attn_online(Q, K, V, scale, 8, rescale_O=True)
    O_bad, _, _ = attn_online(Q, K, V, scale, 8, rescale_O=False)
    print(f"    形状 {Sq}×{d}，KV 长 {Sk}，分 8 块；每一块都在抬高 m")
    print(f"    正确（重标定 O）：与朴素两遍的最大相对误差 = {relerr(O_ok, O_ref):.3e}")
    print(f"    错误（忘标定 O）：与朴素两遍的最大相对误差 = {relerr(O_bad, O_ref):.3e}")
    check("重标定 O 的版本正确（< 1e-12）", relerr(O_ok, O_ref) < 1e-12,
          f"{relerr(O_ok, O_ref):.3e}")
    check("忘标定 O 的版本**确实算错**（相对误差 > 1e-3）",
          relerr(O_bad, O_ref) > 1e-3, f"{relerr(O_bad, O_ref):.3e}")
    check("而且它既不产生 inf 也不产生 nan（安静地错，最难查）",
          not has_inf(O_bad) and not has_nan(O_bad))
    print()
    print("    成因：O_old 是按**旧基准** m_old 累积的，而新的权重 exp(s − m_new) 用的是")
    print("      新基准。两者直接相加，等于把不同基准下的量混在一起——每一块的")
    print("      系数是 exp(m_old − m_new) 而不是 1。")
    print("    规避：α = exp(m_old − m_new) 必须**同时**乘到 l 和 O 上，")
    print("      这是本文 §1 的核心；漏掉 O 那一侧不会触发任何数值异常，只会安静地错。")
    print("    ⚠ 这一段与 softmax-normalization/02 §4.1（把 m 冻结/延后更新）是两件事：")
    print("      那条错在 m 本身，这条错在 m 抬起来之后**谁跟着一起走**。")


def c5_wrong_ml():
    sub("C5  反向重算 S 时 (m, l) 配对错：三种取法的差别")
    rng = np.random.default_rng(20260923)
    Sq, Sk, d = 8, 64, 16
    scale = 1.0 / math.sqrt(d)
    Q = rng.standard_normal((Sq, d))
    K = rng.standard_normal((Sk, d))
    V = rng.standard_normal((Sk, d))
    dO = rng.standard_normal((Sq, d))
    O, _, _ = attn_naive(Q, K, V, scale)
    ref = attn_backward_ref(Q, K, V, dO, scale)
    print(f"    形状 {Sq}×{d}，KV 长 {Sk}，分 8 块；参照 = 物化 P 的朴素反向（fp64）")
    print()
    print(f"    {'重算 P 时用的 (m, l)':<28}{'dQ':>12}{'dK':>12}{'dV':>12}")
    out = {}
    for mode, label in (("final", "前向最终的 (m_final, l_final)"),
                        ("running", "该块当时的 (m_b, l_b)"),
                        ("mixed", "该块当时的 m_b + 最终 l_final")):
        got = attn_backward_flash(Q, K, V, dO, O, scale, 8, ml_mode=mode)
        errs = [relerr_norm(g, r) for g, r in zip(got, ref)]
        out[mode] = (got, errs)
        print(f"    {label:<28}" + "".join(f"{e:>12.2e}" for e in errs))
    check("用前向的**最终** (m, l) 重算 P 是正确的（< 1e-14）",
          max(out["final"][1]) < 1e-14, f"max={max(out['final'][1]):.3e}")
    check("用该块当时的 (m_b, l_b) 是错的（dQ/dK/dV 至少一个 > 1e-2）",
          max(out["running"][1]) > 1e-2, f"max={max(out['running'][1]):.3e}")
    check("把 m_b 与 l_final 配在一起也是错的（> 1e-2）",
          max(out["mixed"][1]) > 1e-2, f"max={max(out['mixed'][1]):.3e}")
    print()
    print("    ⟹ 结论要说准：**「用最终值」不是错误，「配对不一致」才是错误。**")
    print("      P = exp(S − m)/l 要求 (m, l) 是**同一时刻的同一对**；")
    print("      前向收敛后的 (m_final, l_final) 恰好就是每行真 softmax 的分母，")
    print("      所以它是对的（也是 flash-attention 存下来给反向用的那两个量）。")
    print("      运行值 (m_b, l_b) 描述的是「前 b 块的部分 softmax」，")
    print("      拿它当分母等于把每一块都按自己的基准归一化，块与块之间没有共同的")
    print("      尺度。")
    for mode in ("running", "mixed"):
        errs = out[mode][1]
        worst = "dQ dK dV".split()[int(np.argmax(errs))]
        print(f"      {mode:<8}: 三项**都**错（dQ {errs[0]:.2e} / dK {errs[1]:.2e} / "
              f"dV {errs[2]:.2e}），没有哪一项能当哨兵；最差的是 {worst}")


def c6_per_block_delta():
    sub("C6  反向按块算 D = rowsum(dP∘P)：**本程序第一版就是这么错的**")
    rng = np.random.default_rng(20260924)
    Sq, Sk, d = 24, 40, 16
    scale = 1.0 / math.sqrt(d)
    Q = rng.standard_normal((Sq, d))
    K = rng.standard_normal((Sk, d))
    V = rng.standard_normal((Sk, d))
    dO = rng.standard_normal((Sq, d))
    O, _, _ = attn_naive(Q, K, V, scale)
    ref = attn_backward_ref(Q, K, V, dO, scale)
    print("    D_i = Σ_j dP_ij·P_ij 是**整行**的和，不是「本块内那一行的和」。")
    print("    它有两个等价算法：")
    print("      (a) 从最终的 O 一次算出：D = rowsum(dO ∘ O)   ← 不需要 P，正确")
    print("      (b) 逐块累加 rowsum(dP_b ∘ P_b)            ← 要跨块攒，正确但更麻烦")
    print("    错的写法是每块各算各的、当场就用：")
    print()
    print(f"    {'D 的算法':<30}{'dQ':>12}{'dK':>12}{'dV':>12}")
    for mode, label in (("full", "rowsum(dO ∘ O)（整行，正确）"),
                        ("per_block", "每块各自的 rowsum(dP_b∘P_b)（错）")):
        got = attn_backward_flash(Q, K, V, dO, O, scale, 8, delta_mode=mode)
        errs = [relerr_norm(g, r) for g, r in zip(got, ref)]
        print(f"    {label:<30}" + "".join(f"{e:>12.2e}" for e in errs))
        if mode == "full":
            check("整行的 D 给出正确的三个梯度（< 1e-14）",
                  max(errs) < 1e-14, f"max={max(errs):.3e}")
        else:
            check("每块各算各的 D：dQ/dK 明显错掉（> 1e-2），dV 丝毫不动",
                  errs[0] > 1e-2 and errs[1] > 1e-2 and errs[2] < 1e-12,
                  f"dQ={errs[0]:.2e} dK={errs[1]:.2e} dV={errs[2]:.2e}")
    print()
    print("    成因：dS 的表达式里，只有 D 这一项是**跨块求和**的量；分块后每一块")
    print("      只知道自己在那一行上的部分和。用部分和去减，等于给每一块单独做了")
    print("      一次「去掉自己的基准」，块与块之间不再共享同一个尺度。")
    print("    规避：D 从最终的 O 算。这不是技巧，而是 flash 反向能成立的前提——")
    print("      **正因为 D 不需要 P，才可以把 P 丢掉、在反向重算。**")
    print("    ⚠ 这个错很隐蔽：**dV 完全不受影响**（dV = Pᵀ·dO 里根本不含 D），")
    print("      错的只有 dQ 与 dK。只拿 dV 对答案的检查会把它整个放过去。")


def c7_decode_parallelism():
    sub("C7  decode 的并行度：grid = batch × heads_q 可能填不满一张卡")
    print(f"    H100 有 {SM_COUNT} 个 SM；decode 的一个 block 处理一个 (序列, query 头)")
    print()
    print(f"    {'batch':>7}{'heads_q':>9}{'grid':>7}{'占 SM 数':>10}"
          f"{'每 SM 分到':>11}{'判定':>16}")
    for b, h in ((1, 32), (1, 8), (1, 1), (8, 32), (64, 32), (256, 32)):
        g = b * h
        print(f"    {b:>7}{h:>9}{g:>7}{g / SM_COUNT * 100:>9.1f}%"
              f"{g / SM_COUNT:>11.2f}{'填不满' if g < SM_COUNT else '够用':>16}")
    check("batch=1、heads_q=32（GQA 8 kv 头的推理常见形态）时 grid=32 < 132 SM",
          32 < SM_COUNT, f"{32 / SM_COUNT * 100:.0f}% 占用")
    check("batch=1、heads_q=1（极端 MQA）时只有一个 block 在跑",
          1 * SM_COUNT / (1) > 1, "grid = 1，132 个 SM 里 131 个空闲")
    print(f"    ⟹ batch 小的时候必须切 KV（split-KV）：把 KV 序列再切成 n_split 段，")
    print(f"      grid 变成 batch × heads_q × n_split。这就是 §A5 那个合并算子的用处：")
    print(f"      每一段独立算出一个 (m_i, l_i, O_i)，最后按 §A5 的 ⊕ 合并。")
    print(f"      代价是 O 也要跟着重标定（`O` 的丢弃/合并都走同一条式子）。")
    n_split = 16
    print(f"      batch=1、heads_q=32、n_split={n_split} → grid = "
          f"{32 * n_split}，"
          f"每 SM 分到 {32 * n_split / SM_COUNT:.2f} 个 block")
    check(f"n_split={n_split} 后 grid 超过 SM 数（{32 * n_split} > {SM_COUNT}）",
          32 * n_split > SM_COUNT)


def c8_paged_fragmentation():
    sub("C8  paged KV 的边界：页大小取多大才不浪费 sector")
    d = D_HEAD
    print(f"    一页一个 head 的行字节 = d_head·esize；sector = {SP.SECTOR_BYTES} B，"
          f"cache line = {SP.CACHE_LINE_BYTES} B")
    print(f"    {'d_head':>8}{'esize':>7}{'行字节':>9}{'整除 sector':>13}"
          f"{'整除 cache line':>17}{'页(16行)字节':>14}")
    for dh, es in ((128, 2), (128, 1), (64, 2), (80, 2), (96, 2), (128, 4)):
        row = dh * es
        print(f"    {dh:>8}{es:>7}{row:>8}B"
              f"{('是' if row % SP.SECTOR_BYTES == 0 else '否'):>13}"
              f"{('是' if row % SP.CACHE_LINE_BYTES == 0 else '否'):>17}"
              f"{row * 16:>12}B")
    check("d_head=128、fp16 时一行 256 B 恰好是 cache line 的整数倍（页内无浪费）",
          D_HEAD * ESIZE_FP16 % SP.CACHE_LINE_BYTES == 0,
          f"{D_HEAD * ESIZE_FP16} B")
    check("d_head=80、fp16 时一行 160 B 不是任何 2 的幂（页首对齐才有意义）",
          (80 * ESIZE_FP16) % SP.CACHE_LINE_BYTES != 0 and (80 * ESIZE_FP16) % SP.SECTOR_BYTES == 0,
          "160 B = 5 个 sector")
    print()
    print("    ⚠ 页内连续时 sector 利用率是 100%，所以**页大小不影响合并访存**；")
    print("      影响合并的是「一页之内的**行**是否连续」——paged KV 的布局")
    print("      [页, head, 行, d] 保证了这一点，代价挪到了页表的间接寻址上（§B10）。")
    print("      若为了省页表把布局改成 [页, 行, head, d]（行内跨 head 交织），")
    print("      同一行的 head 之间就不连续了：d_head=128 时同一行相邻 head 相隔")
    print(f"      {D_HEAD * ESIZE_FP16} B → 每个 head 各占 1 个 cache line，")
    print(f"      warp 读同一位置的不同 head 会退化成 32 个不同的 cache line。")


# ===========================================================================
# 主流程
# ===========================================================================


def main():
    # §C 要断言的正是「溢出 / inf / nan / 除以零」这些失败行为，
    # numpy 的 RuntimeWarning 就是这些断言的对象本身，不是异常——关掉输出噪音，
    # 失败的**行为**仍然由 §C 的 check 逐条判定。
    np.seterr(all="ignore")

    print("=" * 76)
    print("40-kernels/attention 本机验证模型（verify.py）")
    print("=" * 76)
    print("本程序只跑算术与 numpy，不碰硬件。全部输出是**推导值**。")

    s0_scope()

    section("A", "算法正确性")
    a0_thresholds()
    a1_online_vs_naive()
    cache2 = a2_backward_fd()
    res = a3_dq_accumulation(cache2)
    a4_gqa_mqa()
    a5_split_kv()
    a6_hand_example()

    section("B", "代价模型")
    b1_ai_table()
    b2_ridge()
    b3_s_materialization()
    b4_flash_config()
    b5_backward_recompute()
    b6_dq_accumulation()
    b7_warp_specialization()
    b8_gqa_traffic()
    b9_decode_ai()
    b10_paged_kv()
    b11_window_variants()

    section("C", "边界与反例")
    c1_no_max_subtraction()
    c2_rescale_underflow()
    c3_dq_bit_patterns(res)
    c4_forget_rescale_O()
    c5_wrong_ml()
    c6_per_block_delta()
    c7_decode_parallelism()
    c8_paged_fragmentation()

    print()
    print("=" * 76)
    total = _STATE["n"]
    fail = _STATE["fail"]
    print(f"§D  汇总：断言 {total} 条，通过 {total - len(fail)} 条，不通过 {len(fail)} 条")
    if fail:
        for f in fail:
            print(f"    ✗ {f}")
    else:
        print("    全部通过。")
    print("=" * 76)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
