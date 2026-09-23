"""
100-parallelism 探测 E：**ring attention 的跨块重缩放**（计划 §0.7-D 的 D6 / D7 / D14）。

要回答（`sequence-context-parallel/02` 的判据设计依据）：
  1. 把 KV 按 `CP` 切块、每块用自己的 running-max 做重缩放，结果与**全量 attention** 差多少？
  2. **不做重缩放**的朴素版会怎样？（原尺度下**看起来也对**，放大尺度就 `nan`）
  3. `np.einsum('bhst,bthd->bshd', p, v)` 的输出轴序到底是什么？（**轴序是独立的出错环节**）

★ **只打印、不断言**。可证伪的断言在 `sequence-context-parallel/code/` 的编号程序里。
★ **本主题的禁区**：**「在线 softmax」的 `(m, l)` 更新公式本身**（正文在
  `40-kernels/softmax-normalization/02-online-softmax.md`）与**注意力变体的结构**
  （MHA / MQA / GQA，正文在 `80-model-architectures/attention-variants/01`）——
  这里写的是**跨设备**的块统计量重缩放。
★ 本机**无 GPU、无多卡** ⇒ 「设备」在这里只是**索引**，通信是显式的张量搬运。
★ 不是 NCCL，也不是真机。
"""
import contextlib

import numpy as np

rng = np.random.default_rng(100)


@contextlib.contextmanager
def quiet():
    """★ **有意的**溢出/无效值静默：E2/E3 的朴素分支**就是要**让 `exp` 溢出成 `nan`。

    只在**这一段**里静默 —— 判据仍然靠打印出来的 `含 nan=True` 承担，不是靠警告。
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        yield


def show(tag, msg):
    print(f"[{tag}] {msg}")


B, H, S, Dh = 1, 2, 64, 16
SCALE = 1.0 / np.sqrt(Dh)
q = rng.standard_normal((B, H, S, Dh)) * 0.5
k = rng.standard_normal((B, H, S, Dh)) * 0.5
v = rng.standard_normal((B, H, S, Dh)) * 0.5

show("E0", f"形状：B={B} H={H} S={S} D={Dh}；SCALE = 1/sqrt(D) = {SCALE:.6f}")
show("E0", "注意：`np.split` 要求 S 能被 CP 整除 ⇒ 这里 S=64，CP 取 1/2/4/8/16")


def attn_full(q, k, v, rescale=True):
    """全量 attention（K、V 都在一张卡上）。rescale=False 时**不做 max 平移**。"""
    s = np.einsum("bhsd,bhtd->bhst", q, k) * SCALE
    if rescale:
        p = np.exp(s - s.max(axis=-1, keepdims=True))
    else:
        p = np.exp(s)                      # ← 没有 max 平移：数值上不安全
    l = p.sum(axis=-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", p / l, v)


def attn_ring(q, k, v, CP, rescale=True):
    """把 K/V 沿 S 切成 CP 块，按块顺序推进（**通信 = 显式的张量搬运**）。

    rescale=True：每来一块就更新 running-max，并把已有的 `acc`/`l` **按新 max 重缩放**。
    """
    kbs = np.split(k, CP, axis=2)
    vbs = np.split(v, CP, axis=2)
    neg_inf = np.full((B, H, S, 1), -np.inf)
    m = neg_inf.copy()
    l = np.zeros((B, H, S, 1))
    acc = np.zeros((B, H, S, Dh))
    for kb, vb in zip(kbs, vbs):
        sb = np.einsum("bhsd,bhtd->bhst", q, kb) * SCALE
        if rescale:
            m_new = np.maximum(m, sb.max(axis=-1, keepdims=True))
            corr = np.exp(m - m_new)                       # ← 跨块重缩放因子
            p = np.exp(sb - m_new)
            l = l * corr + p.sum(axis=-1, keepdims=True)
            acc = acc * corr + np.einsum("bhst,bhtd->bhsd", p, vb)
            m = m_new
        else:
            p = np.exp(sb)                                 # ← 朴素版：不重缩放
            l = l + p.sum(axis=-1, keepdims=True)
            acc = acc + np.einsum("bhst,bhtd->bhsd", p, vb)
    return acc / l


print()
show("E1", "── D6：带 running-max 的跨块重缩放 vs 全量 ──")
ref = attn_full(q, k, v)
print("  %6s %18s %18s" % ("CP", "max|diff|", "/ eps(fp64)"))
for CP in (1, 2, 4, 8, 16):
    out = attn_ring(q, k, v, CP)
    d = np.max(np.abs(out - ref))
    print("  %6d %18.3e %18.3f" % (CP, d, d / np.finfo(np.float64).eps))
show("E1", "★ 全部都是**机器精度档** ⇒ 分块本身不改变数学（含重缩放时）")
show("E1", "★ 注意 CP 越大块越多，误差**不一定单调上升**（多一次重缩放 = 多一次舍入，方向随机）")

print()
show("E2", "── D7 反例甲：**朴素版（不做 max 重缩放）**，原尺度 ──")
with quiet():
    naive_ok = attn_full(q, k, v, rescale=False)
    d_naive = np.max(np.abs(naive_ok - ref))
    show("E2", f"朴素全量 vs 带 max 全量： max|diff| = {d_naive:.3e} ⇒ **原尺度下看起来也对**")
    for CP in (1, 4, 8):
        out = attn_ring(q, k, v, CP, rescale=False)
        show("E2", f"     朴素 ring(CP={CP}) vs 全量： {np.max(np.abs(out - ref)):.3e}")

print()
show("E3", "── D7 反例乙：把 Q、K **放大 100 倍** ──")
Q2, K2 = q * 100.0, k * 100.0
ref2 = attn_full(Q2, K2, v)
for rescale in (False, True):
    with quiet():
        out = attn_full(Q2, K2, v, rescale=rescale)
    tag = "带 max" if rescale else "**朴素（无 max）**"
    show("E3", f"{tag:<22}： max|diff| = {np.max(np.abs(out - ref2)):.3e}"
               f"  含 inf={bool(np.any(np.isinf(out)))}  含 nan={bool(np.any(np.isnan(out)))}")
for rescale in (False, True):
    with quiet():
        out = attn_ring(Q2, K2, v, 8, rescale=rescale)
    tag = "带 max" if rescale else "**朴素（无 max）**"
    show("E3", f"ring(CP=8) {tag:<14}： max|diff| = {np.max(np.abs(out - ref2)):.3e}"
               f"  含 inf={bool(np.any(np.isinf(out)))}  含 nan={bool(np.any(np.isnan(out)))}")
show("E3", "★★ ⇒ 判据不能只看「原尺度下对不对」：**朴素版在原尺度也是 1e-16 档**，")
show("E3", "   只有把尺度推到溢出边界，「不做重缩放」这个缺陷才暴露 ⇒ 这是 C 类判据的构造方法")

print()
show("E4", "── D14：`einsum` 的输出轴序（**独立的出错环节**）──")
p = np.exp(np.einsum("bhsd,bhtd->bhst", q, k) * SCALE)
p = p / p.sum(axis=-1, keepdims=True)
v_th = np.transpose(v, (0, 2, 1, 3))        # (b,t,h,d) —— 换一种摆法
A = np.einsum("bhst,bthd->bshd", p, v_th)
Bt = np.einsum("bhst,bthd->bhsd", p, v_th)
show("E4", f"输入：p{p.shape}（b,h,s,t）  v_th{v_th.shape}（b,t,h,d）")
show("E4", f"`'bhst,bthd->bshd'`  ⇒  输出形状 {A.shape}（**`s` 与 `h` 交换了**）")
show("E4", f"`'bhst,bthd->bhsd'`  ⇒  输出形状 {Bt.shape}（这才是 (b,h,s,d)）")
show("E4", f"两者逐元素相等吗？ {np.array_equal(A.transpose(0, 2, 1, 3), Bt)}"
           f"（**是同一个数、不同的轴序** ⇒ 形状对不上时报的错会离现场很远）")
show("E4", "★ 本域纪律：凡写 `einsum` 必**显式写出输出轴**（`->`），不靠隐式推断")

print()
show("E9", "★ 本脚本只打印、不断言；这里量的是**分块的代数**，没有一次真实通信。")
show("E9", "★ 不是 NCCL，也不是真机。")
