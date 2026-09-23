"""
100-parallelism 探测 F：EP 的 **all-to-all 搬运**与专家负载（计划 §0.7-D 的 D8 / D9 / D21）。

要回答（`expert-parallel/02` 的核心）：
  1. 每台设备**发出**的量相等吗？**收到**的量相等吗？（★ 「均衡发送、不均接收」）
  2. 专家负载的分布长什么样？（`max/mean` 与 `min/mean`）
  3. 容量因子一压，**丢弃率**是多少？

★ **随机种子**：`np.random.default_rng(100)`（本文件顶部）—— 换种子这些数会变，
  **不是**「typically 如此」（计划 §4 的 T56）。
★ **只打印、不断言**。可证伪的断言在 `expert-parallel/code/` 的编号程序里。
★ **本主题的禁区**：**容量因子的定义与语义**（定义式、取整口径、
  dropping vs No-Token-Left-Behind）正文在 `80-model-architectures/moe/02-capacity-factor.md`；
  本脚本用的 `capacity = ceil(top_k·T/E·CF)` 只是一个**可算的记账模型**，
  **不是**在给这个词下定义。
★ 「设备」在这里只是**索引**；通信是显式的索引搬运（`all_to_all` 的搬运量 = 计数）。
★ 不是 NCCL，也不是真机。
"""
import numpy as np

SEED = 100
rng = np.random.default_rng(SEED)

T = 256          # token 数（全局 batch × 序列长）
E = 8            # 专家数
TOP_K = 2        # 每 token 选几个专家


def show(tag, msg):
    print(f"[{tag}] {msg}")


show("F0", f"种子 = {SEED}；T = {T}（token 数）、E = {E}（专家数）、top_k = {TOP_K}")
show("F0", f"⇒ 总投递条数（每个 token 的每一份拷贝算一条）= T·top_k = {T * TOP_K}")

scores = rng.standard_normal((T, E))
top = np.argsort(-scores, axis=1)[:, :TOP_K]          # 每 token 的 top_k 个专家
show("F0", f"路由：对 `scores`（{scores.shape}，N(0,1)）取每行最大的 {TOP_K} 个")

# ------------------------------------------------------------------ D9 专家负载
print()
show("F1", "── D9：**专家负载分布** ──")
load = np.bincount(top.ravel(), minlength=E)
mean = load.mean()
show("F1", "各专家收到的 token 数：" + "  ".join(str(int(x)) for x in load))
show("F1", f"mean = {mean:.6f}  max/mean = {load.max() / mean:.6f}  "
           f"min/mean = {load.min() / mean:.6f}")
show("F1", f"方差/in^2 的 CV： {load.std() / mean:.6f}"
           f"（**一个种子**的值，不是「typically」）")

# ------------------------------------------------------------------ D8 搬运矩阵
print()
show("F2", "── D8：**dispatch 的搬运量矩阵**（行 = 源设备，列 = 目的设备）──")
show("F2", "★ token 按 `token_id * EP // T` **轮流**分给 EP 台设备；专家按 `expert_id * EP // E` 分")
matrices = {}
for EP in (2, 4, 8):
    owner = np.arange(T) * EP // T                     # 每个 token 的源设备
    dest = top * EP // E                               # 每个 token 的每一份拷贝的去向
    M = np.zeros((EP, EP), dtype=int)
    for t in range(T):
        for d in dest[t]:
            M[owner[t], d] += 1
    matrices[EP] = M
    sends = M.sum(axis=1)
    recvs = M.sum(axis=0)
    show("F2", f"EP={EP}：发送量 {list(map(int, sends))}（**恒等**吗？ {sends.min() == sends.max()}）")
    show("F2", f"       接收量 {list(map(int, recvs))}  max/min = {recvs.max() / recvs.min():.6f}")
    show("F2", f"       每源设备发出 {T // EP * TOP_K} 条（= T·top_k/EP = {T * TOP_K}/{EP}）")

print("  搬运量矩阵（EP=4）：")
M4 = matrices[4]
for r in M4:
    print("   " + "".join("%8d" % v for v in r))

print()
show("F3", "── ★★ **均衡发送、但接收不均**：比值随 EP 上升 ──")
for EP, M in matrices.items():
    recvs = M.sum(axis=0)
    send_ratio = recvs.max() / recvs.min()
    show("F3", f"     EP={EP}：接收 max/min = {send_ratio:.6f}"
               f"（每设备收的平均 = {recvs.mean():.3f}）")
show("F3", "★ 直觉「设备越多越均衡」是**错的**：设备越多，每台摊到的 token 越少 ⇒ 相对涨落越大")

# ------------------------------------------------------------------ D21 容量因子
print()
show("F4", "── D21：**容量因子与丢弃率** ──")
show("F4", "模型（★ 只是记账，不是定义）：容量 = ceil(top_k·T/E·CF)；"
           "每个专家超过容量的那些投递**被丢弃**")
show("F4", f"top_k·T/E = {TOP_K * T / E:.1f}（平均每个专家该收多少）")
print("  %10s %14s %16s %16s" % ("CF", "容量/专家", "丢弃条数", "丢弃率"))
for CF in (1.0, 1.25, 1.5, 2.0):
    cap = int(np.ceil(TOP_K * T / E * CF))
    kept = int(np.minimum(load, cap).sum())
    dropped = int(load.sum()) - kept
    print("  %10.2f %14d %16d %15.2f%%" % (CF, cap, dropped, 100.0 * dropped / load.sum()))
show("F4", "★ CF=1.0 时**必然**有丢弃（因为负载本来就不均）⇒ 容量因子是「用丢弃换确定性」的旋钮")
show("F4", "★ 这些数是**一个种子**下的；`dropping` 与 `No-Token-Left-Behind` 的语义之别**不在本脚本**")
show("F4", "  （正文在 `80-model-architectures/moe/02-capacity-factor.md`）")

print()
show("F9", "★ 本脚本只打印、不断言；以上是**索引搬运的计数**，没有一次真实通信。")
show("F9", "★ 不是 NCCL，也不是真机。")
