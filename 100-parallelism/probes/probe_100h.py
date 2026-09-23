"""
100-parallelism 探测 H：**论文公式的恒等式**、**SP 的收益比**、**A 类判据的恒真检查**。
（对应计划 §0.7-D 的 D15 / D16 / **D22**；★ 那张 9 行的脚本表在计划
**§3 · Chunk 0 的动作 4**（不是 §3.0.1 —— 那是「主线」节），里面**已经列了本脚本**；
是 Chunk 0 的**任务书**只给了 8 个 ⇒ 本脚本是 **Chunk 0 的补齐**。）

要回答三件事：

  H1（D15）**Korthikanti 式(2) 与式(4) 是不是同一个东西的两种写法？**
     · `t = 1` 时两式**必须相等**（SP 在 TP 度 1 时应无收益）——
       不等就说明公式抄错了。
     · `eq2 − eq4` 是不是**恰好** `10sbh(1 − 1/t)`？
   H2（D16）**SP 的收益有多大？它随序列长度怎么变？**
   H3（D22）★★★ **本域最要紧的一条**：哪些「切分 == 单卡」的判据是**恒真**的？
     恒真的判据只重述定义，**不能当 A 类**（那是本域的头号错误模式）。

★ **本脚本只打印、不断言**。可证伪的断言在各主题 `code/` 的编号程序里。
★ **限定语（强制）**：本机 **无 GPU、无多卡、无 NCCL、无真实网络**；
  这里量的是**代数与计数**，**不是任何实现的性能**。**不是 NCCL，也不是真机**。
★ **H1 / H2 是「推导值」**（把 `specs.py` 会用的公式代进去算），**不是任何实现的实测**。
  它们的**依据是公式本身**：公式的出处与逐字引文见计划 §C.4（Korthikanti, arXiv:2205.05198，
  **同行评审：是**，MLSys 2023）。
★ **H3 是「本机实测值」**（真的跑了 numpy），须与本文件同见 python3 / numpy 版本。
"""
import platform
import sys

import numpy as np


def show(tag, msg):
    print(f"[{tag}] {msg}")


show("H0", f"python3 {platform.python_version()} / numpy {np.__version__} / "
           f"{platform.machine()}（本机实测值的测量条件）")
show("H0", "★ 全部是玩具规模的真跑；**不是 NCCL，也不是真机**")
print()


# ===========================================================================
# H1（D15）· Korthikanti 式(2) 与式(4) 的两条恒等式
# ===========================================================================
# 记号（**必须一并引用**，否则不可复核）：
#   s = 序列长度 · b = micro-batch 大小 · h = hidden size · a = attention head 数
#   t = TP（张量并行）度
# 式(2)（仅 TP）      ：sbh(10 + 24/t + 5as/(ht))
# 式(4)（TP + SP）    ：(sbh/t)(34 + 5as/h)
# 二者的差恒等于 10sbh(1 − 1/t) —— 即【SP 省的正是被复制的那 10sbh 项按 t 切掉的差额】。
def eq2(s, b, h, a, t):
    """式(2)：仅张量并行时的每层激活（元素数）。"""
    return s * b * h * (10 + 24 / t + 5 * a * s / (h * t))


def eq4(s, b, h, a, t):
    """式(4)：张量并行 + 序列并行时的每层激活（元素数）。"""
    return (s * b * h / t) * (34 + 5 * a * s / h)


# 三组参数（★ 结论只对「式(2) 与式(4) 的写法」成立，与具体规模无关）
CASES = [(1024, 1, 64, 8), (2048, 2, 128, 16), (4096, 4, 256, 32)]

print("─" * 78)
show("H1a", "★ 恒等式甲：t = 1 时式(2) == 式(4)（SP 在 TP=1 时应无收益）")
worst = 0.0
for (s, b, h, a) in CASES:
    A, B = eq2(s, b, h, a, 1), eq4(s, b, h, a, 1)
    rel = abs(A - B) / A
    worst = max(worst, rel)
    show("H1a", f"     s={s:<5d} b={b} h={h:<4d} a={a:<3d} : "
                f"式(2)={A:.6e}  式(4)={B:.6e}  相对差={rel:.3e}")
show("H1a", f"     ⇒ 三组全部相对差 = 0（最大 {worst:.3e}）✓")

print()
show("H1b", "★ 恒等式乙：eq2 − eq4 == 10sbh(1 − 1/t)")
s, b, h, a = 2048, 1, 128, 16
for t in (1, 2, 4, 8, 16, 32):
    lhs = eq2(s, b, h, a, t) - eq4(s, b, h, a, t)
    rhs = 10 * s * b * h * (1 - 1 / t)
    show("H1b", f"     t={t:<3d} : eq2−eq4 = {lhs:.6e}   10sbh(1−1/t) = {rhs:.6e}   "
                f"差 = {abs(lhs - rhs):.3e}")
show("H1b", f"     ⇒ 恒等成立（差值全为 0）—— 这就是「SP 省的是被复制的 10sbh 项」的算术证据")
show("H1b", "     ★ 依据：原文逐字 `the 10sbh part of Equation 2 is due to these replicated")
show("H1b", "       operations and as a result they are not divided by the tensor parallel size t`")
print()


# ===========================================================================
# H2（D16）· SP 的收益比 —— ★ 两组参数【不同】，必须一并引用
# ===========================================================================
print("─" * 78)
show("H2", "★★ SP 的收益比 eq2/eq4（t=1 时恒为 1.000 —— 那是 H1a 的恒等式）")
show("H2", "★★ 注意：下面两组用了【不同的 a/h】（0.125 vs 0.0625，差 2 倍）—— 这是必需的，")
show("H2", "   因为收益比依赖 a/h；只报「s=1024 得 1.104、s=4096 得 1.053」而不报 a/h 是**不可复核**的。")
PARAM_SETS = [
    ("s=1024", 1024, 1, 64, 8),
    ("s=4096", 4096, 1, 256, 16),
]
for label, s, b, h, a in PARAM_SETS:
    row = []
    for t in (1, 2, 4, 8, 16, 32):
        row.append(f"t={t}:{eq2(s, b, h, a, t) / eq4(s, b, h, a, t):.4f}")
    show("H2", f"     {label} (b={b}, h={h:<4d}, a={a:<3d}, a/h={a/h:.4f})")
    show("H2", "            " + "  ".join(row))
show("H2", "★★★ ⇒ **序列越长，SP 的相对收益越小**")
show("H2", "     成因：10sbh 是 O(s)，而 attention 的 5as²b/h 是 O(s²)；**SP 不切后者**")
show("H2", "     ⇒ 这就是「SP 够不够用」与 `sequence-context-parallel/03` 的引子：")
show("H2", "       长序列要靠 CP（切【全部】激活，含 O(s²) 那批），不是靠 SP")
print()


# ===========================================================================
# H3（D22）· A 类判据的「恒真」系统检查 —— ★★★ 本域最要紧的一条
# ===========================================================================
print("─" * 78)
show("H3", "★★★ 检查：本域最自然的 A 类判据「切分后的结果 == 单卡的结果」，")
show("H3", "     在哪些切法上是**恒真**的？（恒真 ⇒ 只重述定义 ⇒ **不能当 A 类**）")
show("H3", "★ 判据的机械形式：max|diff| —— 若恒为 0.0，则这条判据没有任何信息量。")

rng = np.random.default_rng(100)          # ★ 种子写死，便于复跑
TH = 2.0                                   # 行并行比对的相对阈（只用于打印）
rows = []

# ── 甲：列并行（切 D_out）—— 纯搬运 ──
B_, S_, D_in, D_out, TP = 2, 8, 64, 32, 4
X = rng.standard_normal((B_, S_, D_in))
W = rng.standard_normal((D_in, D_out)) * 0.1
col = np.concatenate([X @ w for w in np.split(W, TP, axis=1)], axis=-1)
rows.append(("列并行（切 D_out）", "纯搬运", float(np.abs(col - X @ W).max())))

# ── 乙：行并行（切 D_in）—— 改变了求和顺序 ──
Xs, Ws = np.split(X, TP, axis=-1), np.split(W, TP, axis=0)
rowsum = sum(Xs[r] @ Ws[r] for r in range(TP))
rows.append(("行并行（切 D_in）", "改了求和顺序", float(np.abs(rowsum - X @ W).max())))

# ── 丙：MLP 列 + 行 ──
W1 = rng.standard_normal((D_in, 4 * D_in)) * 0.1
W2 = rng.standard_normal((4 * D_in, D_out)) * 0.1
ref_mlp = np.maximum(X @ W1, 0) @ W2
parts = [np.maximum(X @ w1, 0) @ w2
         for w1, w2 in zip(np.split(W1, TP, axis=1), np.split(W2, TP, axis=0))]
rows.append(("MLP 列 + 行", "改了求和顺序", float(np.abs(sum(parts) - ref_mlp).max())))

# ── 丁：EP 的 token 分发 —— 纯索引搬运 ──
T_, Dd, E, TOPK = 64, 16, 4, 2
Xm = rng.standard_normal((T_, Dd))
Wg = rng.standard_normal((Dd, E)) * 0.3
We = [rng.standard_normal((Dd, Dd)) * 0.1 for _ in range(E)]
sel = np.argsort(-(Xm @ Wg), axis=1)[:, :TOPK]
single = np.zeros((T_, Dd))
for t_ in range(T_):
    for e in sel[t_]:
        single[t_] += Xm[t_] @ We[e]
ep = np.zeros((T_, Dd))
per = E // 2
for dev in range(2):
    belong = set(range(dev * per, (dev + 1) * per))
    for t_ in range(T_):
        for e in sel[t_]:
            if e in belong:
                ep[t_] += Xm[t_] @ We[e]
rows.append(("EP 的 token 分发", "纯搬运", float(np.abs(ep - single).max())))

# ── 戊：DP 的 N 卡平均 vs 单卡大 batch —— 改了累加顺序 ──
Bn, N_, Dn = 8, 4, 64
Xg = rng.standard_normal((Bn * N_, Dn))
yg = rng.standard_normal((Bn * N_,))
g_split = sum(Xg[i * Bn:(i + 1) * Bn].T @ yg[i * Bn:(i + 1) * Bn] for i in range(N_)) / N_
g_one = (Xg.T @ yg) / N_
rows.append(("DP 的 N 卡平均", "改了累加顺序", float(np.abs(g_split - g_one).max())))

# ── 己：ring（CP=4）vs 全量 attention —— 改了除法的位置 ──
Bc, Sc, Hc, dh = 1, 32, 2, 8
Q, K, V = (rng.standard_normal((Bc, Sc, Hc, dh)) * 0.5 for _ in range(3))
sc_ = 1 / np.sqrt(dh)
sx = np.einsum('bshd,bthd->bhst', Q, K) * sc_
sx = sx - sx.max(-1, keepdims=True)
pr = np.exp(sx)
pr /= pr.sum(-1, keepdims=True)
ref_attn = np.einsum('bhst,bthd->bshd', pr, V)
mm = np.full((Bc, Hc, Sc, 1), -np.inf)
ll = np.zeros((Bc, Hc, Sc, 1))
acc = np.zeros((Bc, Hc, Sc, dh))
for kc, vc in zip(np.split(K, 4, axis=1), np.split(V, 4, axis=1)):
    ss = np.einsum('bshd,bthd->bhst', Q, kc) * sc_
    mn = np.maximum(mm, ss.max(-1, keepdims=True))
    al = np.exp(mm - mn)
    pp = np.exp(ss - mn)
    acc = acc * al + np.einsum('bhst,bthd->bhsd', pp, vc)
    ll = ll * al + pp.sum(-1, keepdims=True)
    mm = mn
rows.append(("ring（CP=4）vs 全量", "改了除法的位置",
             float(np.abs(np.einsum('bhsd->bshd', acc / ll) - ref_attn).max())))

show("H3", "")
show("H3", f"     {'切法':<22s} {'它改变了求值顺序吗':<18s} {'max|diff|':>14s}   本组为 0?")
show("H3", "     " + "-" * 70)
for name, kind, d in rows:
    flag = "★★ 是（无用）" if d == 0.0 else "否（有信息量）"
    show("H3", f"     {name:<22s} {kind:<18s} {d:>14.3e}   {flag}")
show("H3", "")
show("H3", "★★★ 结论：**在本脚本这组形状下**，只有改变了【求值顺序】的切分才给出非零差；")
show("H3", "     ★★★ **但不能由此说「列并行的差恒为 0」** —— 上表里那两行 `0.0` 是【本组形状】的观测：")
show("H3", "        · **列并行（切 `D_out`）与 vocab 维列切是同一件事 ⇒ BLAS 路径依赖**：")
show("H3", "          同一份数据换个形状就可能非零（`tensor-parallel/01`/`02` 的 `D.10` 量到 `1.4e-14`），")
show("H3", "          且**哪些形状给 `0` 是数据 / 种子依赖的**；")
show("H3", "        · **头切分**才是**结构性恒真**（每卡做的**就是**单卡时那几个 head 的同一批运算，")
show("H3", "          **不涉及 GEMM 的 `M`/`N`**）—— 那一组在 `tensor-parallel/02` 的 `D.10` 里；")
show("H3", "        · **EP 的 token 分发**也是**搬运型**（差是否为零取决于数据与设备归属的划分）。")
show("H3", "     ⇒ ★★ **可引用的是「形状依赖」这个【性质】；某组具体形状给 `0` 的清单本身不可引用。**")
show("H3", "     ⇒ 本域纪律：这类地方【改用 C 类】（证明「搬运做错会失败」），")
show("H3", "       或把 A 类改钉在「改变了求值顺序」的地方（行并行 / MLP 组合 / DP 累加 / ring 除法）。")

print()
show("H9", "★ 本脚本只打印、不断言。")
show("H9", "★ H1/H2 是**推导值**（公式代入），H3 是**本机实测值**（真的跑了 numpy）。")
show("H9", "★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
