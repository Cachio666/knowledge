"""
100-parallelism 探测 B：ZeRO 三阶段的**每卡显存账**（对应计划 §0.7-D 的 D3）。

要回答（`zero-fsdp/01` 的核心）：**分片不是除法**。
  · 把优化器状态分成 `N` 份，每卡真的只省到 `1/N` 吗？
  · `N` 一直涨下去，每卡会趋近于 0 吗？

★ **本脚本量的是「字节账」**（一个**推导值**：`每卡字节 = Σ 各项 / 分片度`），
  不是任何实现的实测；**没有一个字节来自真机**。
★ **只打印、不断言**。可证伪的断言在 `zero-fsdp/code/` 的编号程序里。
★ **本主题的禁区**：`16Ψ` 的**组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）正文在
  `90-training/checkpointing/01-state-and-bytes.md`；这里**只引用**它的四项系数（参数 2 / 梯度 2 /
  优化器 12），**不重新推导**它。`4x`/`8x` 与 `P_os`/`P_os+g` 的对应关系正文在
  `90-training/checkpointing/02-sharded-save-and-load.md`。
★ 单位口径：**GB = 1e9 字节**（十进制），不是 GiB。`Ψ` = 参数量。
"""
import numpy as np

PSI = 7e9                    # 参数量 Ψ
BYTES_PER_PARAM = {"param": 2, "grad": 2, "opt": 12}    # 引用值：见上「禁区」
GB = 1e9

P, G, O = (BYTES_PER_PARAM[k] for k in ("param", "grad", "opt"))
TOTAL_PER_PARAM = P + G + O                             # = 16


def show(tag, msg):
    print(f"[{tag}] {msg}")


def gb(per_param_bytes):
    """每参数字节数 → 全模型该卡的 GB（按 Ψ 缩放）。"""
    return per_param_bytes * PSI / GB


show("B0", f"Ψ = {PSI:.0e} 个参数；字节/参数：参数 {P} + 梯度 {G} + 优化器 {O} = {TOTAL_PER_PARAM}")
show("B0", "单位口径：GB = 1e9 字节（十进制），**不是** GiB")
show("B0", f"⇒ 不分片时每卡 = {TOTAL_PER_PARAM}Ψ = {gb(TOTAL_PER_PARAM):.1f} GB")

# 四个 stage 的「每参数字节数」公式（N = 数据并行度）
#   ZeRO-0：都不分片                         P + G + O
#   ZeRO-1：只分片优化器状态                  P + G + O/N
#   ZeRO-2：分片梯度 + 优化器状态             P + (G + O)/N
#   ZeRO-3：全分片（含参数）                  (P + G + O)/N
FORMULA = {
    "ZeRO-0": lambda N: P + G + O,
    "ZeRO-1": lambda N: P + G + O / N,
    "ZeRO-2": lambda N: P + (G + O) / N,
    "ZeRO-3": lambda N: (P + G + O) / N,
}
NS = [1, 8, 64, 512]

print()
show("B1", "── 每卡显存（行 = stage，列 = 数据并行度 N）──")
head = "  %-8s" % "stage" + "".join("%18s" % f"N={N}" for N in NS)
print(head)
for stage, f in FORMULA.items():
    row = "  %-8s" % stage
    for N in NS:
        b = f(N)
        row += "%18s" % f"{gb(b):.1f} GB"
    print(row)
print()
show("B1", "── 同上，换成 Ψ 的倍数（= 每参数字节数）──")
print(head.replace("N=", "Ψ× N="))
for stage, f in FORMULA.items():
    row = "  %-8s" % stage
    for N in NS:
        row += "%18s" % f"{f(N):.3f}Ψ"
    print(row)

# ★★ 这条是本探测的**结论行**：分片范围决定了地板
print()
P_OS_FLOOR = P + G
show("B2", f"★★ `P_os` 的**下界**是 `{P}+{G} = {P_OS_FLOOR}Ψ`"
           f"（= {gb(P_OS_FLOOR):.1f} GB）")
show("B2", "     ★ 理由：ZeRO-1/2 的**参数与梯度那一块不在分片范围内** ⇒"
           " N 再大，每卡也不会低于这个地板")
for N in NS:
    z1 = FORMULA["ZeRO-1"](N)
    show("B2", f"     N={N:<4d} ZeRO-1：{z1:.4f}Ψ（= {gb(z1):.2f} GB）；"
               f"距离下界还有 {z1 - P_OS_FLOOR:.4f}Ψ")
show("B2", "     ⇒ 从 N=64 到 N=512（**8 倍**），ZeRO-1 每卡只从 "
           f"{FORMULA['ZeRO-1'](64):.3f}Ψ 降到 {FORMULA['ZeRO-1'](512):.3f}Ψ"
           f"（**{FORMULA['ZeRO-1'](512) / FORMULA['ZeRO-1'](64):.3f} 倍**）")

print()
show("B3", "── ZeRO-3 才是「接近除法」的那一档：N 翻 8 倍，每卡降 8 倍 ──")
for a, b_ in zip(NS[:-1], NS[1:]):
    r_a, r_b = FORMULA["ZeRO-3"](a), FORMULA["ZeRO-3"](b_)
    show("B3", f"     N: {a} → {b_}（×{b_ // a}） ⇒ "
               f"{r_a:.4f}Ψ → {r_b:.4f}Ψ（×{r_b / r_a:.4f}）")
show("B3", "     ★ 对照：同一段 N 上 ZeRO-1 的比值是 "
           f"{FORMULA['ZeRO-1'](NS[-1]) / FORMULA['ZeRO-1'](NS[0]):.4f}（**不是** 1/N 的倍数关系）")

print()
show("B9", "★ 本脚本只打印、不断言；以上是**推导值**（字节账），不是任何实现的实测。")
show("B9", "★ 不是 NCCL，也不是真机：本机无 GPU、无多卡，这里一个字节也没有真的分配过。")
