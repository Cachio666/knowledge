"""
100-parallelism 探测 G：**SP 的激活账 + DP 的通信量复核 + PP 的 stage 不均**
（计划 §0.7-D 的 D11 / D10 / D12）。

三段各回答一件事：
  1. **SP 切的是什么** —— 哪些激活项**随 TP 度下降**、哪些**不随**（O(s) vs O(s²)）
  2. **DP 的通信量** —— `2Ψ` 的梯度过 IB / NVLink 各要多久（★ 与 `performance-models/02` 交叉复核）
  3. **PP 的 stage 显存不均** —— 首尾 stage（含 embedding / lm_head）比中间 stage 多多少

★ **只打印、不断言**。判据在 `sequence-context-parallel/code/`、`data-parallel/code/`、
  `pipeline-parallel/code/` 里。
★ **本主题的禁区**：**`16Ψ` 的字节分解**正文在 `90-training/checkpointing/01-state-and-bytes.md`
  （这里只**引用**梯度 `= 2Ψ` 这一个系数）；**Amdahl / 扩展效率模型**正文在
  `10-foundations/performance-models/02-amdahl-and-scaling.md`；**链路带宽的三种口径**正文在
  `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`。
★ ★★ **本脚本第 2 段用的是「引用值」**（50 GB/s / 450 GB/s 来自上面那份文档）——
  **本仓库不复核其测量口径**。第 1、3 段是**推导值**（按写明的假设算出来的）。
★ 不是 NCCL，也不是真机。
"""
def show(tag, msg):
    print(f"[{tag}] {msg}")


# ============================================================
# 1. SP 的激活账（D11）
# ============================================================
print()
show("G1", "── D11：SP 切的是哪一批激活 ──")
B, S, A_HEADS, D_HEAD, D_FF, LAYERS = 2, 1024, 8, 64, 256, 12
H_HID = A_HEADS * D_HEAD
show("G1", "假设（**必须一并引用**）："
           f"b={B}（batch）· s={S}（序列长）· a={A_HEADS}（注意力头数）· d={D_HEAD}（每头维）"
           f" ⇒ h={H_HID}（隐层）· d_ff={D_FF} · 层数={LAYERS}")

# ★ 逐 token 类：每一项都是 O(b*s*h)，**与 s 成正比**
PER_TOKEN = {
    "LayerNorm 输出": B * S * H_HID,
    "Dropout mask": B * S * H_HID,
    "残差流": B * S * H_HID,
    "MLP 后": B * S * H_HID,
}
# ★ O(s²) 类：注意力概率
PROBS = B * A_HEADS * S * S
small = sum(PER_TOKEN.values())
total = small + PROBS

show("G1", "（**每层**，单位：元素数）")
for k, v in PER_TOKEN.items():
    show("G1", f"     {k:<16} {v:>12,d}  （占合计 {100.0 * v / total:5.2f}%）")
show("G1", f"     {'逐 token 类小计':<16} {small:>12,d}  （占合计 {100.0 * small / total:5.2f}%）")
show("G1", f"     {'softmax 概率 (b,a,s,s)':<16} {PROBS:>12,d}  （占合计 {100.0 * PROBS / total:5.2f}%）")
show("G1", f"     合计 {total:,d} 元素/层；{LAYERS} 层 ⇒ {total * LAYERS:,d} 元素")
show("G1", "     ★★ **清单口径**：本段是 **5 项**（4 项逐 token + 1 项 softmax 概率）。")
show("G1", "        计划 §0.7-D 的 D11 行用的是**另一份 8 项清单**（多 QKV 投影 / attention 输出 / MLP 中间）")
show("G1", "        ⇒ 它的 `14.58% / 66.67%` 与本段的 `20.00% / 80.00%` **分母不同、不可比**；")
show("G1", "        两组数各自自洽。★★ **凡报占比必须一并给出【完整清单】与【总元素数】**。")

print()
show("G2", "── 同上，把 s 从 1024 提到 4096（其余不变）──")
S2 = 4096
small2 = 4 * B * S2 * H_HID
probs2 = B * A_HEADS * S2 * S2
show("G2", f"     s={S2}：逐 token 类占 {100.0 * small2 / (small2 + probs2):.2f}%，"
           f"softmax 概率占 {100.0 * probs2 / (small2 + probs2):.2f}%")
show("G2", f"     ⇒ 序列变 4 倍，O(s²) 项占比从 {100.0 * PROBS / total:.2f}% "
           f"升到 {100.0 * probs2 / (small2 + probs2):.2f}%")

print()
show("G3", "── ★ 这两批在 **TP 度 t** 下的行为（SP 只切前一批）──")
print("  %6s %22s %22s %16s" % ("t", "逐 token 类（每卡）", "softmax 概率（每卡）", "前者降了几倍"))
for t in (1, 2, 4, 8):
    show_pt = small / t                     # SP 把 O(s) 那批按 t 切
    show_pr = PROBS                         # O(s²) 那批**不被 SP 切**（要 CP 才能切）
    print("  %6d %22s %22s %16.1f"
          % (t, f"{show_pt:,.0f}", f"{show_pr:,.0f}", small / show_pt))
show("G3", "★ ⇒ **SP 的收益上限由 O(s) 那批的占比决定**；序列越长，这个上限越低")
show("G3", "  （要把 O(s²) 那批也切下去，需要 CP —— 正文在 `sequence-context-parallel/03`）")

# ============================================================
# 2. DP 的通信量复核（D10）
# ============================================================
print()
show("G4", "── D10：DP 的通信量（★ 与 `10-foundations/performance-models/02` 交叉复核）──")
PSI = 70e9
GRAD_BYTES = 2 * PSI                      # 引用值：梯度 = 2Ψ（分解正文在 checkpointing/01）
GB = 1e9
show("G4", f"Ψ = {PSI:.0e} ⇒ 梯度 = 2Ψ = {GRAD_BYTES / GB:.1f} GB")
show("G4", "★ ring all-reduce 的搬运量 = 2(N-1)/N × S；N 大时趋近 **2S**")
BW = {"IB NDR 单向（引用值）": 50.0, "NVLink4 每方向（引用值）": 450.0}
S_BYTES = GRAD_BYTES
print("  %-26s %14s %14s %14s" % ("链路 / 口径", "S=2Ψ(s)", "2S(N=8)", "2S(N→∞)"))
for name, bw in BW.items():
    col = []
    for vol in (S_BYTES, S_BYTES * 2 * 7 / 8, S_BYTES * 2.0):
        col.append(vol / GB / bw)
    print("  %-26s %14.3f %14.3f %14.3f" % (name, col[0], col[1], col[2]))
show("G4", "★ 表里的单位是**秒**；第三列即教科书口径的 `2S / BW`")
show("G4", "★ **引用值**：50 GB/s 与 450 GB/s 取自 "
           "`20-gpu-architecture/interconnect/01-bandwidth-conventions.md:196,:231`；")
show("G4", "  **本仓库不复核其测量口径**，也不把它们当成真机实测")
show("G4", "★ ★ 本段**只算搬运时间**：不含延迟、不含协议开销、不含计算重叠")

# ============================================================
# 3. PP 的 stage 不均（D12）
# ============================================================
print()
show("G5", "── D12：PP 的 stage 不均（首尾含 embedding / lm_head）──")
P_STAGES, LAY = 4, 12
B2, S3, H3 = 2, 1024, 512
per_layer = 4 * B2 * S3 * H3                    # 与 G1 的「逐 token 类」同一个口径
per_stage_layers = LAY // P_STAGES
mid = per_stage_layers * per_layer
edge = mid + B2 * S3 * H3                       # 首尾多一块 embedding / lm_head 的激活
show("G5", f"假设：p={P_STAGES} 级 · 共 {LAY} 层（每级 {per_stage_layers} 层）· "
           f"b={B2} · s={S3} · h={H3}")
show("G5", f"     中间级激活 = {mid:,d} 元素；首尾级 = {edge:,d} 元素 ⇒ "
           f"比值 = {edge / mid:.6f}")
show("G5", f"     ★ 比值只取决于「每级几层」与「embedding 折算几个层」："
           f"这里嵌入了 {B2 * S3 * H3 / per_layer:.4f} 个「层当量」")
show("G5", "     ⇒ 层数越多 / 级数越少，这个比值越接近 1（**不均是结构性的，不是配置没调好**）")
for p_ in (2, 4, 8):
    lay_p = LAY // p_
    show("G5", f"     p={p_}（每级 {lay_p} 层）：比值 = "
               f"{(lay_p * per_layer + B2 * S3 * H3) / (lay_p * per_layer):.6f}")

print()
show("G9", "★ 本脚本只打印、不断言。第 1、3 段是**推导值**（按写明的假设算）；")
show("G9", "  第 2 段的链路带宽是**引用值**（本仓库不复核其测量口径）。")
show("G9", "★ 不是 NCCL，也不是真机：这里没有一张卡、没有一条链路。")
