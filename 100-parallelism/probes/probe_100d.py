"""
100-parallelism 探测 D：**`(Σg)/N` 与 `Σ(g/N)` 到底一样不一样**（计划 §0.7-D 的 D5）。

★★★ 这一条的**口径是全部**：**用的是 Python 内建 `sum()`（顺序累加）**。
   换成 `np.sum()`（**pairwise / 分块累加**）会得到**不同的计数** ——
   本脚本**两列都打**，就是为了让这个口径差别看得见（计划 §D.4 ①：我和审查者
   各拿一个口径在争论同一组数）。
★ **只打印、不断言**。可证伪的断言在 `data-parallel/code/` 的编号程序里。
★ 模型：`N` 张卡各持一份**同形状**的梯度（向量长 `D`），
  「平均」有两种写法 —— 先全局求和再除 `N`，或者每卡先除 `N` 再求和。
  两者在实数上恒等，在浮点上**只在求和顺序一致时**逐位相同。
★ **本机无 GPU、无多卡、无 NCCL** ⇒ 这里量的是**浮点求和的结合律**，不是任何真机行为。
"""
import numpy as np

rng = np.random.default_rng(100)
TRIALS = 200


def show(tag, msg):
    print(f"[{tag}] {msg}")


def two_ways(g, N, use_builtin):
    """返回 ((Σg)/N, Σ(g/N)) 逐元素比是否**有任何一个元素不同**。

    ★ `use_builtin=True` 用 **Python 内建 sum()（顺序累加）**；
      `False` 用 `np.sum()`（**pairwise**）—— 两者是**不同口径**。
    """
    if use_builtin:
        a = sum(g) / N                       # 顺序累加，除一次
        b = sum(x / N for x in g)            # 每卡先除，再顺序累加
    else:
        a = np.sum(np.stack(g), axis=0) / N
        b = np.sum(np.stack([x / N for x in g]), axis=0)
    return bool(np.any(a != b)), a, b


print()
show("D0", f"每格 {TRIALS} 次试验；梯度向量长 D，卡数 N；元素来自 N(0,1)")
show("D0", "★ 口径：**Python 内建 sum()（顺序累加）** —— 见 `two_ways()` 的注释")

# ---------------------------------------------------------------- 1. 2 的幂
print()
show("D1", "── 卡数 N 是 **2 的幂** 时 ──")
show("D1", "★ 逐位相同的理由：除以 2 的幂是**精确**缩放，且**与舍入可交换**")
print("  %5s %6s %14s %14s" % ("D", "N", "内建 sum 非零", "np.sum 非零"))
POW2 = [(64, 2), (64, 4), (64, 8), (64, 16), (256, 32), (256, 1024 // 8)]
for D, N in POW2:
    nb = ns = 0
    for _ in range(TRIALS):
        g = [rng.standard_normal(D) for _ in range(N)]
        if two_ways(g, N, True)[0]:
            nb += 1
        if two_ways(g, N, False)[0]:
            ns += 1
    print("  %5d %6d %14s %14s" % (D, N, f"{nb}/{TRIALS}", f"{ns}/{TRIALS}"))

# ---------------------------------------------------------------- 2. 非 2 的幂
print()
show("D2", "── 卡数 N **不是** 2 的幂 时：非零次数随 D 上升 ──")
NS = [3, 5, 7, 10, 100]
DS = [2, 6, 16, 64, 1024]
print("  %6s" % "D" + "".join("%16s" % f"N={N}" for N in NS))
for D in DS:
    row = "  %6d" % D
    for N in NS:
        nb = 0
        for _ in range(TRIALS):
            g = [rng.standard_normal(D) for _ in range(N)]
            if two_ways(g, N, True)[0]:
                nb += 1
        row += "%16s" % f"{nb}/{TRIALS}"
    print(row)

print()
show("D2", "★ 每格都是 `非零次数 / 总次数`（分子大 = 两种写法在那个规模上**更容易分叉**）")
show("D2", "★ ⇒ **判据必须写明 D 与 N**：同一个 `assert diff != 0` 在小 D 上会随机失败")

# ---------------------------------------------------------------- 3. 一个能看的实例
print()
show("D3", "── 一个具体的分叉实例（首次命中即打印）──")
D, N = 6, 3
for k in range(TRIALS):
    g = [rng.standard_normal(D) for _ in range(N)]
    nz, a, b = two_ways(g, N, True)
    if nz:
        i = int(np.argmax(np.abs(a - b)))
        show("D3", f"第 {k + 1} 次试验命中：D={D}, N={N}")
        show("D3", f"     `(Σg)/N`[{i}] = {a[i]!r}")
        show("D3", f"     `Σ(g/N)`[{i}] = {b[i]!r}")
        show("D3", f"     差 = {abs(a[i] - b[i]):.3e}（相对 {abs(a[i] - b[i]) / max(abs(a[i]), 1e-300):.3e}）")
        break

# ---------------------------------------------------------------- 4. 两种口径的并列对照
print()
show("D4", "── **同一组随机数**上，两种累加口径的并列对照 ──")
show("D4", "★ 口径甲 = Python 内建 `sum()`（顺序累加）；口径乙 = `np.sum()`（pairwise 分块）")
TRIALS2 = 200
print("  %6s %6s %20s %20s %10s" % ("D", "N", "内建 sum() 非零", "np.sum() 非零", "差"))
for D, N in [(2, 3), (2, 100), (2, 500), (6, 100)]:
    nb = ns = 0
    for _ in range(TRIALS2):
        g = [rng.standard_normal(D) for _ in range(N)]
        if two_ways(g, N, True)[0]:
            nb += 1
        if two_ways(g, N, False)[0]:
            ns += 1
    print("  %6d %6d %20s %20s %10d" % (D, N, f"{nb}/{TRIALS2}", f"{ns}/{TRIALS2}", nb - ns))
show("D4", "★ 本构造下两列计数相同 —— 原因见下一段：**归约轴是跨步的**，")
show("D4", "  numpy 在这种情况下**不做 pairwise 分块**，于是与顺序累加重合")

print()
show("D5", "── ★★ **口径分叉的真实条件：被归约的那个轴连不连续** ──")
show("D5", "★ 同一个 `np.sum()`，归约**内轴（连续）**时走 pairwise、归约**外轴（跨步）**时走顺序累加")
show("D5", "  ⇒ 「用哪个函数」还不够，**「在哪个轴上归约」也是口径的一部分**")
TRIALS3 = 200
print("  %6s %6s %26s %26s" % ("D", "N", "归约跨步轴（axis=0）", "归约连续轴（axis=1）"))
for D, N in [(4, 100), (4, 137), (4, 500)]:
    hit_strided = hit_contig = 0
    for _ in range(TRIALS3):
        rows = [rng.standard_normal(D) for _ in range(N)]
        A = np.stack(rows, axis=0)                   # (N, D)：归约轴 0 **跨步**
        if np.any(sum(rows) / N != np.sum(A, axis=0) / N):
            hit_strided += 1
        cols = [rng.standard_normal(D) for _ in range(N)]
        B = np.stack(cols, axis=1)                   # (D, N)：归约轴 1 **连续**
        if np.any(sum(cols) / N != np.sum(B, axis=1) / N):
            hit_contig += 1
    print("  %6d %6d %26s %26s"
          % (D, N, f"{hit_strided}/{TRIALS3}", f"{hit_contig}/{TRIALS3}"))

print()
show("D5", "── 一个能看的实例（连续轴上首次命中即打印）──")
for _ in range(TRIALS3):
    D, N = 4, 100
    cols = [rng.standard_normal(D) for _ in range(N)]
    B = np.stack(cols, axis=1)
    r1 = sum(cols) / N
    r2 = np.sum(B, axis=1) / N
    if np.any(r1 != r2):
        i = int(np.argmax(np.abs(r1 - r2)))
        show("D5", f"     顺序累加（内置 sum）： {r1[i]!r}")
        show("D5", f"     pairwise（np.sum，连续轴）： {r2[i]!r}")
        show("D5", f"     差 = {abs(r1[i] - r2[i]):.3e}")
        break
show("D5", "★ ⇒ 凡报「N/M 次」这类计数，**必须同时报「用什么累加的」**（计划 §D.4 ①）")

print()
show("D9", "★ 本脚本只打印、不断言；数的是**浮点求和的结合律**，不是任何真机行为。")
show("D9", "★ 不是 NCCL，也不是真机。")
