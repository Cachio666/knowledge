"""
100-parallelism 探测 A：TP 的列 / 行并行 —— **切在哪个轴，连误差档位都不同**。

要回答（对应计划 §0.7-D 的 D1 / D2，也是 `tensor-parallel/01` 的 A 类判据设计依据）：
  1. 列并行（切 `D_out`）与单卡比，**在这组形状下是不是逐位相同**？
     ★★ 而「逐位相同」**不是恒真的**（见 `probe_100h.py` 的 `H3` 与 `tensor-parallel/01`/`02` 的 `D.10`）。
  2. 行并行（切 `D_in`）与单卡比，差多少（**几个 ulp**）？
  3. **忘记 all-reduce** 会怎样？（C 类判据要证明的「失败」）
  4. **切错轴** 会怎样？
  5. 局部部分和的**量级** —— 「一片是零、一片是全」这个直觉成立吗？

★ **形态**：**只打印、不断言**（可证伪的断言属于 `tensor-parallel/code/` 下的编号程序）。
★ **通信是一个显式的张量搬运**：`all_reduce = Σ局部`、`all_gather = concat`。
  理由见域 README「代码约定」：本域的知识对象是**切分策略**，不是通信原语的实现。
★ 本机**无 GPU、无多卡、无 NCCL** ⇒ 这里量的是**切分的代数**，**不是**任何真机性能。
  强制限定语：**「不是 NCCL，也不是真机」**。
★ 口径：`np.max(np.abs(a - b))`（逐元素绝对差的最大值），**不是**相对差、**不是**范数差。
"""
import numpy as np

rng = np.random.default_rng(100)


def show(tag, msg):
    print(f"[{tag}] {msg}")


B, S, D_IN, D_OUT = 2, 8, 64, 32
NP_DEV = 4                      # TP 度（设备数）
X = rng.standard_normal((B, S, D_IN))
W = rng.standard_normal((D_IN, D_OUT)) * 0.1
Y_SINGLE = X @ W                # 单卡（无切分）

show("A0", f"形状：X={X.shape}  W={W.shape}  NP_DEV={NP_DEV}（TP 度）")
show("A0", "口径：max|diff| = 逐元素绝对差的最大值")

# ============================================================
# A. 列并行：切 W 的 D_out（输出列）—— 免通信
# ============================================================
print()
show("A1", "── 列并行（切 `D_out`）—— **免通信** ──")
blocks = np.split(W, NP_DEV, axis=1)            # 每卡一份 (D_IN, D_OUT/NP_DEV)
Y_COL = np.concatenate([X @ w for w in blocks], axis=-1)   # all_gather = concat
d_col = np.max(np.abs(Y_COL - Y_SINGLE))
show("A1", f"切分（{NP_DEV} 卡各算各的输出列，再 concat）vs 单卡： max|diff| = {d_col:.3e}")
show("A1", f"     ★ 逐位相同吗？ {np.array_equal(Y_COL, Y_SINGLE)}"
           f"（**本组形状**下为 True：每个输出元素的点积没变）")
show("A1", "     ★★★ **但「列并行 == 单卡」不是一条【恒真】的判据 —— 它随形状变**："
           "同一份数据换个形状就可能非零（`tensor-parallel/01`/`02` 的 `D.10` 量到 `1.4e-14`），"
           "成因是 `GEMM` 的 `M`/`N` 变了 ⇒ BLAS 可能换分块 / 向量化路径。")
show("A1", "     ⇒ ★★ **可引用的是「形状依赖」这个性质；本组形状给 `0` 这件事本身不可引用**"
           "（⇒ 这条判据不能当 A 类）。")

# ============================================================
# B. 行并行：切 X 的 D_in 与 W 的「行」—— 要 all-reduce
# ============================================================
print()
show("A2", "── 行并行（切 `D_in`）—— **要 all-reduce** ──")
xs = np.split(X, NP_DEV, axis=-1)               # 每卡一份 (B, S, D_IN/NP_DEV)
ws = np.split(W, NP_DEV, axis=0)                # 每卡一份 (D_IN/NP_DEV, D_OUT)
partials = [x @ w for x, w in zip(xs, ws)]      # 各卡的**局部部分和**
Y_ROW = sum(partials)                           # all_reduce = Σ局部（显式搬运）
d_row = np.max(np.abs(Y_ROW - Y_SINGLE))
show("A2", f"切分（{NP_DEV} 卡各算局部部分和，再 Σ）vs 单卡： max|diff| = {d_row:.3e}")
show("A2", f"     ★ 与 float64 的 eps = {np.finfo(np.float64).eps:.3e} 比："
           f" {d_row / np.finfo(np.float64).eps:.3f} × eps"
           f"（求和顺序变了 ⇒ 应当是**机器精度档**，不是 0）")

# ============================================================
# C. MLP 的「列 + 行」：先列并行、再行并行（Megatron 的标准切法）
# ============================================================
print()
show("A3", "── MLP：第一层列并行、第二层行并行 ──")
D_FF = 128
W1 = rng.standard_normal((D_IN, D_FF)) * 0.1
W2 = rng.standard_normal((D_FF, D_OUT)) * 0.1
H_SINGLE = np.maximum(X @ W1, 0.0)
Y2_SINGLE = H_SINGLE @ W2
w1s = np.split(W1, NP_DEV, axis=1)              # 列并行：每卡一段 D_ff
w2s = np.split(W2, NP_DEV, axis=0)              # 行并行：每卡一段 D_ff
parts2 = [np.maximum(X @ a, 0.0) @ b for a, b in zip(w1s, w2s)]
Y2 = sum(parts2)                                # 第二层出口一次 all-reduce
d_mlp = np.max(np.abs(Y2 - Y2_SINGLE))
show("A3", f"MLP（列+行）vs 单卡： max|diff| = {d_mlp:.3e}"
           f"（{d_mlp / np.finfo(np.float64).eps:.3f} × eps）")
show("A3", "     ★ 两层的误差**只有一处来源**：第二层出口那一次 all-reduce（第一层是纯搬运）")

# ============================================================
# D. 忘记 all-reduce —— C 类判据要证明的那个「失败」
# ============================================================
print()
show("A4", "── **忘记 all-reduce**：只拿一卡的局部部分和当结果 ──")
Y_FORGOT = partials[0] * NP_DEV                  # 既不 Σ、又把 N 的除数丢了
d_forgot = np.max(np.abs(Y_FORGOT - Y_SINGLE))
d_forgot_sum = np.max(np.abs(partials[0] - Y_SINGLE))
show("A4", f"只取 device 0 的局部部分和： max|diff| = {d_forgot_sum:.6f}")
show("A4", f"拿了 Σ 的除数、但没 Σ（`partials[0] * N`）： max|diff| = {d_forgot:.6f}")
show("A4", f"     ★ 与上面 A2 的 {d_row:.3e} 相比，**大了 {d_forgot / max(d_row, 1e-300):.3e} 倍**")
show("A4", "     ⇒ 行并行的「忘了 all-reduce」是**可观测的失败**，不是精度问题")

# ============================================================
# E. 切错轴 —— 形状不匹配（也是 C 类）
# ============================================================
print()
show("A5", "── **切错轴**：把 X 按 `D_in` 切开、却仍与整个 W 相乘 ──")
try:
    _ = xs[0] @ W                                # (B,S,D_IN/N) @ (D_IN,D_OUT)
    show("A5", "!! 居然没报错 —— 这说明形状碰巧兼容，判据需要更严的构造")
except ValueError as e:
    show("A5", f"ValueError: {e}")
show("A5", f"     ★ 另一种切错：按**输出列**去切 X（X 的第二维是 D_in={D_IN}，"
           f"每卡会被切成 {D_IN // NP_DEV} 列）—— 形状上「能跑」，但**语义全错**")
X_WRONG = np.concatenate([X[:, :, i::NP_DEV] for i in range(NP_DEV)], axis=-1)  # 交错切，不是分块
d_wrongx = np.max(np.abs(X_WRONG @ W - Y_SINGLE))
show("A5", f"交错切 `D_in` 再乘： max|diff| = {d_wrongx:.6f} ⇒ 形状对、**索引错位**，"
           f"只有 C 类判据能抓住")

# ============================================================
# F. D2：局部部分和的量级 —— 证伪「一片是零、一片是全」
# ============================================================
print()
show("A6", "── 局部部分和的量级（D2）──")
gnorms = [float(np.linalg.norm(p)) for p in partials]
gnorm = float(np.linalg.norm(Y_SINGLE))
show("A6", "各卡局部部分和的 Frobenius 范数：" + "  ".join(f"{v:.6f}" for v in gnorms))
show("A6", f"合并后（单卡）范数 = {gnorm:.6f}；局部均值 / 全局 = "
           f"{np.mean(gnorms) / gnorm:.6f}")
show("A6", "     ★ 量级同阶 ⇒ 「一片接近 0、一片接近全局」的直觉**不成立**，"
           "误差不是来自「某一片没贡献」")

print()
show("A9", "★ 本脚本只打印、不断言；可证伪的断言在 `tensor-parallel/code/` 的编号程序里。")
show("A9", "★ 不是 NCCL，也不是真机：以上全部是 python3 + numpy 的单进程玩具规模。")
