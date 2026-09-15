"""Amdahl 定律、Gustafson 定律与扩展性：为什么加卡最终不划算。

运行：python3 10_amdahl_scaling.py
依赖：numpy（仅用于格式化与对数轴；全部计算在本机 CPU 上完成，不需要 GPU）

硬件规格来自公开规格表，不凭记忆：
  · NVLink 4（H100）：NVIDIA 规格写 **900 GB/s 双向**，即每方向 450 GB/s。
  · InfiniBand NDR（400 Gb/s）：400 Gb/s ÷ 8 = **50 GB/s 单向**，
    双向合计 100 GB/s。
  这两个数字口径不同（一个双向、一个单向），放在一起比会得到 18 倍，
  同口径比只有 9 倍。本程序统一按**每方向**计算，并在用到的地方注明。
"""

import math

import numpy as np


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


rule("1. Amdahl 定律：串行部分给加速比封顶")

print("把程序按「能不能并行」切成两块。用归一化的说法：串行部分占总时间的")
print("比例是 s，可并行部分占 1-s。")
print()
print("  单处理器：T(1) = s + (1-s) = 1        （把单处理器总时间归一化为 1）")
print("  N 处理器：T(N) = s + (1-s)/N          （只有可并行部分被摊薄）")
print()
print("加速比就是二者之比：")
print()
print("    S(N) = T(1) / T(N) = 1 / ( s + (1-s)/N )")
print()
print("推导到这里已经能看出上限了。把分母里的 s 单独拎出来：")
print()
print("    S(N) = 1 / ( s + (1-s)/N )  <  1 / s        （因为 (1-s)/N > 0）")
print()
print("N → ∞ 时 (1-s)/N → 0，所以：")
print()
print("    lim_{N→∞} S(N) = 1 / s")
print()
print("**这就是 Amdahl 定律：无论投多少处理器，加速比都超不过 1/s。**")
print("s = 5% 时上限是 20 倍——哪怕你用一百万张卡。")

print()
print("解析计算得到的加速比 S(N)（这是公式算出来的，不是实测）：")
print()
NS = (1, 2, 4, 8, 16, 64, 256, 1024, 4096, 65536)
SS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5)
print("  %-8s" % "串行 s" + "".join("%9s" % ("N=%d" % n) for n in NS) + "%11s" % "上限 1/s")
print("  " + "-" * 106)
for s in SS:
    row = "  %-8s" % ("%.3f" % s)
    for n in NS:
        row += "%9.2f" % (1.0 / (s + (1 - s) / n))
    row += "%11s" % ("∞" if s == 0 else "%.1f" % (1 / s))
    print(row)

print()
print("读表要读三件事：")
print()
print("① **s 只要不为 0，上限就是有限的。** s = 0.05 那一行，N 从 1024 涨到 65536")
print("   （64 倍），加速比只从 %.1f 涨到 %.1f——几乎不动。"
      % (1 / (0.05 + 0.95 / 1024), 1 / (0.05 + 0.95 / 65536)))
print("   已经贴着上限 %.0f 了。" % (1 / 0.05))
print()
print("② **小 s 也照样封顶。** s = 0.001（千分之一串行）的上限是 %.0f。"
      % (1 / 0.001))
print("   听起来很高，但 N=4096 时只跑到 %.1f，只有上限的 %.0f%%。"
      % (1 / (0.001 + 0.999 / 4096), (1 / (0.001 + 0.999 / 4096)) / 1000 * 100))
print("   在 4096 卡上，千分之一的串行部分吃掉了 %.0f%% 的潜在加速"
      % ((1 - (1 / (0.001 + 0.999 / 4096)) / 4096) * 100))
print("   （理想是 4096 倍，实得 %.0f 倍）。" % (1 / (0.001 + 0.999 / 4096)))
print()
print("③ **s = 0 时 S(N) = N，这才是理想的线性加速。** 但 s 恒不为 0——")
print("   启动、收尾、I/O、通信、同步，任何一样都不是零。")

print()
print("一个常被引用的换算：给定目标加速比 S_t，需要多少处理器？")
print()
print("    由 S_t = 1/(s + (1-s)/N) 反解：")
print("    N = (1-s) / (1/S_t - s) = (1-s)·S_t / (1 - s·S_t)")
print()
print("  %-10s %-16s %s" % ("串行 s", "想达到的加速比", "需要 N"))
print("  " + "-" * 56)
for s, target in ((0.05, 10), (0.05, 20), (0.05, 100), (0.01, 100), (0.001, 512)):
    denom = 1 - s * target
    need = (1 - s) * target / denom if denom > 0 else float("inf")
    tag = "（上限只有 %.0f，无解）" % (1 / s) if denom <= 0 else ""
    print("  %-10.3f %-16d %s %s" % (s, target, ("%.0f" % need) if denom > 0 else "∞", tag))
print()
print("s = 0.05 时想要 100 倍加速，分母 1 - 0.05×100 = 0 直接变号——**无解**。")
print("这就是「上限」的实际含义：不是「很难」，是**数学上做不到**。")

print()
print("Amdahl 曲线（双对数，横轴 N，纵轴加速比，'-' 是理想线性）：")
print()
XLIM, YLIM = (1.0, 4096.0), (1.0, 4096.0)
W, H = 56, 20


def _col(x, lim, n):
    f = (math.log10(x) - math.log10(lim[0])) / (math.log10(lim[1]) - math.log10(lim[0]))
    return min(n - 1, max(0, int(round(f * (n - 1)))))


def _row(y, lim, n):
    f = (math.log10(y) - math.log10(lim[0])) / (math.log10(lim[1]) - math.log10(lim[0]))
    return min(n - 1, max(0, n - 1 - int(round(f * (n - 1)))))


grid = [[' '] * W for _ in range(H)]
for c in range(W):
    fx = c / (W - 1)
    x = 10 ** (math.log10(XLIM[0]) + fx * (math.log10(XLIM[1]) - math.log10(XLIM[0])))
    grid[_row(min(x, YLIM[1]), YLIM, H)][c] = '-'          # 理想线性加速 S = N
curve_chars = {0.0: 'O', 0.001: '1', 0.01: '2', 0.05: '3', 0.1: '4', 0.5: '5'}
legend = []
for s, ch in curve_chars.items():
    for c in range(W):
        fx = c / (W - 1)
        x = 10 ** (math.log10(XLIM[0]) + fx * (math.log10(XLIM[1]) - math.log10(XLIM[0])))
        y = 1.0 / (s + (1 - s) / x)
        if y > YLIM[1]:
            continue
        r = _row(y, YLIM, H)
        grid[r][c] = ch                                  # 曲线覆盖理想线，便于读
    legend.append((ch, s, 1.0 / s if s else float("inf")))

ylab = {}
for e in range(0, 4):
    ylab[_row(10.0 ** e, YLIM, H)] = "1e%d" % e
for r in range(H):
    print("  %-5s|%s" % (ylab.get(r, ""), "".join(grid[r])))
print("  %-5s+%s" % ("", "-" * W))
xlab = [" "] * W
for e in range(0, 4):
    s_ = "1e%d" % e
    c = _col(10.0 ** e, XLIM, W)
    if c + len(s_) < W:
        for i, ch in enumerate(s_):
            xlab[c + i] = ch
print("  %-5s %s" % ("", "".join(xlab)))
print()
print("图例： '-' 理想 S=N，" + "，".join(
    "%s s=%.3f（上限 %s）" % (ch, s, "∞" if s == 0 else "%.0f" % lim) for ch, s, lim in legend))
print()
print("图上每条曲线都在 N 增大后**变成水平线**。水平线的高度就是 1/s。")
print("'O'（s=0）与理想线 '-' 完全重合，所以图上的 '-' 全被 'O' 盖住了——")
print("s=0 是唯一一条不封顶、一路贴着理想线性走的曲线。")

rule("2. Gustafson 定律：换个固定量，结论就反过来")

print("Amdahl 固定的是**问题规模**：机器变多，每个处理器分到的活变少。")
print("Gustafson 固定的是**单处理器的工作量**：机器变多，总问题规模一起变大。")
print()
print("推导。固定「并行执行时」的总时间为 1：")
print()
print("    T(N) = 1，其中串行部分耗时 a，可并行部分耗时 1-a")
print()
print("现在问：这同一个问题放到 1 个处理器上要多久？")
print("可并行部分在 1 个处理器上要花 (1-a)·N 的时间（N 倍），串行部分不变：")
print()
print("    T(1) = a + (1-a)·N")
print()
print("于是加速比：")
print()
print("    S(N) = T(1) / T(N) = a + (1-a)·N")
print()
print("**这是一个关于 N 的线性函数，没有上限。**")
print()
print("  %-10s %-14s %-14s %-14s %s" % ("参数", "T(N) 固定为", "T(1)", "S(N)", "形态"))
print("  " + "-" * 68)
print("  %-10s %-14s %-14s %-14s %s" % ("Amdahl", "问题规模", "1", "1/(s+(1-s)/N)", "有上界 1/s"))
print("  %-10s %-14s %-14s %-14s %s" % ("Gustafson", "单卡工作量", "a+(1-a)N", "a+(1-a)N", "线性无界"))

print()
print("两者**不矛盾**，因为它们固定的量不同——这句话必须落到具体数字上才可信：")
print()
print("  %-10s %10s %14s %14s %14s" % ("串行占比", "N", "Amdahl S(N)", "Gustafson S(N)", "差多少倍"))
print("  " + "-" * 66)
for a in (0.05, 0.1):
    for n in (16, 256, 4096):
        amd = 1.0 / (a + (1 - a) / n)
        gus = a + (1 - a) * n
        print("  %-10.3f %10d %14.2f %14.2f %13.1f"
              % (a, n, amd, gus, gus / amd))
print()
print("注意这里的 a 与第 1 节的 s 是**同一个物理量**（串行占比），只是参照系不同。")
print("a = 0.05、N = 4096 时：Amdahl 说 %.1f 倍，Gustafson 说 %.0f 倍，差 %.0f 倍。"
      % (1 / (0.05 + 0.95 / 4096), 0.05 + 0.95 * 4096,
         (0.05 + 0.95 * 4096) / (1 / (0.05 + 0.95 / 4096))))
print()
print("**哪个对？取决于你在扩机器的时候有没有把 batch 一起放大。**")
print("  · 固定 batch、只加卡 → Amdahl 的处境，加速比封顶。")
print("  · 加卡的同时把 batch 也按比例放大 → Gustafson 的处境，加速比线性。")
print()
print("LLM 训练实际走的是后者：全局 batch 随卡数一起涨（这就是为什么要调")
print("learning rate scaling）。所以「训练任务能扩到几万卡」这个问题的答案，")
print("比 Amdahl 的 1/s 乐观得多——但**前提是 batch 还能继续放大**，")
print("而 batch 放大会撞上收敛性上限，那是另一个约束，本节不涉及。")

rule("3. 通信项：模型选择与最优并行度 N*")

print("前两节都假设并行部分免费扩展。现实里并行要通信，而通信开销随 N 增长。")
print("把时间写成三项：")
print()
print("    T(N) = t_s  +  t_p / N  +  t_c · f(N)")
print()
print("      t_s    ：完全不能并行的固定时间")
print("      t_p/N  ：计算部分，随 N 摊薄")
print("      t_c·f(N)：通信部分，随 N 增长")
print()
print("**关键是 f(N) 取什么。**有两个常见候选，它们对应两种不同的集合通信算法：")
print()
print("  %-10s %-34s %s" % ("f(N)", "对应算法", "为什么"))
print("  " + "-" * 76)
print("  %-10s %-34s %s" % ("N", "ring all-reduce", "步数 = 2(N-1)，每步有固定延迟 α ⇒ 2(N-1)α ∝ N"))
print("  %-10s %-34s %s" % ("log N", "tree / recursive halving", "步数 = 2·log₂N ⇒ 延迟 ∝ log N"))
print()
print("**本程序用线性项 t_c·N，理由是：**")
print()
print("① 数据量项**不是** t_c·N。ring all-reduce 每卡收发 2(N-1)/N × S 字节，")
print("   N 大时它**饱和到 2S**，是个常数，不随 N 增长。所以「通信量爆炸」")
print("   这个直觉在 all-reduce 上是错的——真正随 N 长的是**步数**。")
print()
print("② 步数才是线性的：ring 走 2(N-1) 步，每步一个固定延迟 α（kernel 启动 +")
print("   一跳网络往返）。小消息时这一项主导，大消息时带宽项主导。")
print("   LLM 训练的梯度 all-reduce 是**大消息**，所以真实系统里线性延迟项")
print("   往往很小——第 4 节会把这个数字算出来。")
print()
print("③ tree 的 log N 在 NCCL 里确实存在（小消息默认走 tree），但对本节的")
print("   大梯度场景，ring 是主路径。**选线性项不是因为它大，而是因为它对。**")
print("   本节末尾会给出 log 版本的闭式解做对照。")

print()
print("对线性模型求最优并行度。记 f(N) = N：")
print()
print("    T(N) = t_s + t_p/N + t_c·N")
print()
print("    对 N 求导：dT/dN = -t_p/N² + t_c")
print()
print("    令其为零：-t_p/N² + t_c = 0  ⟹  N² = t_p/t_c")
print()
print("    **N* = sqrt( t_p / t_c )**")
print()
print("二阶导 d²T/dN² = 2·t_p/N³ > 0，所以这是极小点。")
print("把 N* 代回去，得到最小时间和此时的加速比：")
print()
print("    T(N*) = t_s + t_p/sqrt(t_p/t_c) + t_c·sqrt(t_p/t_c)")
print("          = t_s + sqrt(t_p·t_c) + sqrt(t_p·t_c)")
print("          = **t_s + 2·sqrt(t_p·t_c)**")
print()
print("对照 log 模型 f(N) = ln N：")
print()
print("    dT/dN = -t_p/N² + t_c/N = 0  ⟹  N* = t_p/t_c")
print("    T(N*) = t_s + t_c·(1 + ln(t_p/t_c))")
print()
print("两个闭式解形状不同：线性模型是**平方根**（增长慢），log 模型是**线性**。")

print()
print("注意 N* 的表达式里**没有 t_s**。这不是笔误：")
print("t_s 是与 N 无关的常数，平移整条曲线但不改变它的形状，")
print("所以最优 N 完全由「t_p 与 t_c 的相对大小」决定。")
print()
print("取 t_p = 1（归一化），改变 t_c，看两个闭式解差多少：")
print()
print("  %-12s %14s %14s %16s" % ("t_c", "线性模型 N*", "log 模型 N*", "T(N*)-t_s"))
print("  " + "-" * 60)
for tc_ in (1e-2, 1e-4, 1e-6, 1e-8):
    print("  %-12.0e %14.0f %14.0f %16.4f"
          % (tc_, math.sqrt(1 / tc_), 1 / tc_, 2 * math.sqrt(tc_)))

print()
print("两列差了 1.5～4 个数量级（t_c=1e-4 时 100 对 10000）。")
print("**所以必须按实际用的是 ring 还是 tree 来选模型，不能随手挑一个。**")
print("选错的代价不是小数点，是「能扩到 100 卡」和「能扩到 10000 卡」的区别。")
print("下面代入真实数字。")

rule("4. LLM 训练的落点：数据并行的梯度 all-reduce")

# ---- 硬件与模型常量
P_PARAMS = 70e9
ESZ = 2                                  # BF16
# BF16 下每个参数的梯度占 2 字节，梯度总量与参数量同量级
S_GRAD = P_PARAMS * ESZ
P_PEAK = 989.5e12                        # H100 SXM BF16 dense
BW_NVLINK = 450e9                        # NVLink 4：900 GB/s 双向 ÷ 2
BW_IB = 50e9                             # IB NDR 400 Gb/s ÷ 8 = 50 GB/s 单向
ALPHA = 5e-6                             # ring 每一步的固定延迟，取 5 μs

print("参数：")
print("  模型            P = %.0f 参数，梯度以 BF16 存 ⇒ %.0f GB" % (P_PARAMS, S_GRAD / 1e9))
print("  峰值算力        %.1f TFLOPS（H100 SXM BF16 dense）" % (P_PEAK / 1e12))
print("  互联带宽        NVLink 4: %.0f GB/s 每方向（规格标 900 GB/s 双向）"
      % (BW_NVLINK / 1e9))
print("                  IB NDR : %.0f GB/s 每方向（400 Gb/s ÷ 8；双向合计 100 GB/s）"
      % (BW_IB / 1e9))
print()
print("**口径警告**：NVIDIA 写「NVLink 900 GB/s」，IB 写「400 Gb/s」。")
print("900 GB/s 是**双向合计**，400 Gb/s = 50 GB/s 是**单向**。")
print("直接拿 900 比 50 得到 18 倍；同口径（每方向）比是 %.0f 倍。"
      % (BW_NVLINK / BW_IB))
print("本程序一律用每方向带宽，因为 all-reduce 里每个 rank 的收发量相等，")
print("瓶颈在单向链路上。")

print()
print("--- 4.1 通信量为什么是个常数 ---")
print()
print("ring all-reduce 分两个阶段（reduce-scatter + all-gather），")
print("每个 rank 的收发量是：")
print()
print("    V(N) = 2(N-1)/N × S        S = 梯度总字节数 = %.0f GB" % (S_GRAD / 1e9))
print()
print("  %-8s %16s %16s" % ("N", "每卡收发 (GB)", "占 2S 的比例"))
print("  " + "-" * 44)
for n in (2, 8, 64, 256, 1024, 4096, 65536):
    v = 2 * (n - 1) / n * S_GRAD
    print("  %-8d %16.1f %15.1f%%" % (n, v / 1e9, v / (2 * S_GRAD) * 100))
print()
print("**N → ∞ 时 V(N) → 2S，是个常数。**N=1024 时已经是极限值 2S 的 %.1f%%。"
      % ((1024 - 1) / 1024 * 100))
print("所以数据并行的通信**数据量**不会随卡数爆炸——这一点常被说反。")
print()
print("但它也不随 N **下降**。这正是第 1 节那个串行分数 s 的物理来源：")
print("计算时间 t_p/N 一路降到 0，通信时间却停在 2S/BW 不动。")
print("把饱和的通信时间写成 t_s：")
print()
t_ib = 2 * S_GRAD / BW_IB
t_nvl = 2 * S_GRAD / BW_NVLINK
print("    IB     : t_s = 2S/BW = %.3e / %.1e = %.3f s" % (2 * S_GRAD, BW_IB, t_ib))
print("    NVLink : t_s = 2S/BW = %.3e / %.1e = %.3f s" % (2 * S_GRAD, BW_NVLINK, t_nvl))
print()
print("**这两个数就是数据并行的 Amdahl 串行项。**与 N 无关，永远在那儿。")

print()
print("--- 4.2 计算时间：需要先说清全局 batch ---")
print()
B_GLOBAL = 4_000_000
print("训练一个 token 的浮点运算量约 6P（前向 2P + 反向 4P），")
print("所以全局 batch 为 B 时，单卡跑完全部计算的时间是：")
print()
print("    t_p = 6·P·B / P_peak")
print()
print("取 B = %d token：" % B_GLOBAL)
t_p = 6 * P_PARAMS * B_GLOBAL / P_PEAK
print("    t_p = 6 × %.0e × %.0e / %.3e = %.1f s" % (P_PARAMS, B_GLOBAL, P_PEAK, t_p))
print()
print("这是在**一张**卡上跑完整个全局 batch 的时间。N 张卡时计算部分是 t_p/N。")

print()
print("--- 4.3 扩展效率：强扩展（固定全局 batch） ---")
print()
print("扩展效率定义为 实际加速比 / N。理想情况下 T(N) = t_p/N，")
print("实际 T(N) = t_p/N + t_comm(N)，所以：")
print()
print("    η(N) = t_p / ( t_p + N·t_comm(N) )")
print()
print("其中 t_comm(N) = 2(N-1)/N × S / BW。注意 N·t_comm(N) = 2(N-1)·S/BW。")


def eta(N, bw, overlap=False):
    """overlap=True 时假设梯度分桶，通信与反向计算重叠。"""
    t_comm = 2 * (N - 1) / N * S_GRAD / bw
    t_cmp = t_p / N
    if overlap:
        # 反向占总 FLOPs 的 4/6 = 2/3，只有超出这部分的反向计算才暴露出来
        exposed = max(0.0, t_comm - (2.0 / 3.0) * t_cmp)
    else:
        exposed = t_comm
    return t_cmp / (t_cmp + exposed), t_cmp, t_comm


print("  %-7s %13s %13s %11s %13s %13s %11s"
      % ("N", "计算 t_p/N", "通信(IB)", "η_IB", "通信(NVLink)", "η_NVL", "η_NVL 重叠"))
print("  " + "-" * 88)
for n in (1, 2, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
    e_ib, tc, cm_ib = eta(n, BW_IB)
    e_nv, _, cm_nv = eta(n, BW_NVLINK)
    e_nvo, _, _ = eta(n, BW_NVLINK, overlap=True)
    print("  %-7d %13.2f %13.3f %10.1f%% %13.3f %12.1f%% %10.1f%%"
          % (n, tc, cm_ib, e_ib * 100, cm_nv, e_nv * 100, e_nvo * 100))

print()
print("读表：")
print()
print("  · N=1 时没有通信，η = 100%。")
print("  · IB 那一列掉得很快：N=256 时只剩 %.1f%%，N=1024 时 %.1f%%。"
      % (eta(256, BW_IB)[0] * 100, eta(1024, BW_IB)[0] * 100))
print("    因为通信时间卡在 %.2f s 不动，而计算时间已经降到 %.2f s。" % (t_ib, t_p / 1024))
print("  · NVLink 好得多，但也逃不掉同一个上限。")
print("  · NVLink + 重叠几乎不掉——因为通信时间 %.2f s 相对反向计算太小了。"
      % t_nvl)

print()
print("**「多少卡之后加卡不划算」——用 η 跌破 50% 作为判据：**")
print()
print("  %-26s %16s %s" % ("情形", "η 跌破 50% 的 N", "依据"))
print("  " + "-" * 74)


def cross50(bw, overlap=False):
    for n in range(2, 200000):
        if eta(n, bw, overlap)[0] < 0.5:
            return n
    return None


for label, bw, ov in (("IB，无重叠", BW_IB, False),
                      ("IB，梯度分桶重叠", BW_IB, True),
                      ("NVLink，无重叠", BW_NVLINK, False),
                      ("NVLink，梯度分桶重叠", BW_NVLINK, True)):
    n50 = cross50(bw, ov)
    basis = "t_comm = t_cmp" if not ov else "暴露的通信 = t_cmp"
    print("  %-26s %16s %s" % (label, ("N ≈ %d" % n50) if n50 else "> 200000", basis))

print()
print("对照闭式解：无重叠时 η = 50% 等价于 t_comm(N) = t_p/N，即")
print("    2(N-1)·S/BW = t_p  ⟹  N = 1 + t_p·BW/(2S)")
print("  IB     : 1 + %.1f × %.1e / %.3e = %.1f"
      % (t_p, BW_IB, 2 * S_GRAD, 1 + t_p * BW_IB / (2 * S_GRAD)))
print("  NVLink : 1 + %.1f × %.1e / %.3e = %.1f"
      % (t_p, BW_NVLINK, 2 * S_GRAD, 1 + t_p * BW_NVLINK / (2 * S_GRAD)))
print("与表里数出来的值一致（表里是第一个使 η<50% 的整数 N）。")

print()
print("--- 4.4 这个 50% 拐点对 batch 极其敏感 ---")
print()
print("上面对应的是 B = %.0f token 的全局 batch。拐点 N₅₀ 正比于 t_p，也就是正比于 B："
      % B_GLOBAL)
print()
print("  %-14s %14s %16s %16s" % ("全局 batch", "t_p (s)", "N₅₀ (IB)", "N₅₀ (NVLink)"))
print("  " + "-" * 64)
for b in (1e5, 5e5, 1e6, 4e6, 1e7):
    tp = 6 * P_PARAMS * b / P_PEAK
    print("  %-14.0e %14.1f %16.0f %16.0f"
          % (b, tp, 1 + tp * BW_IB / (2 * S_GRAD), 1 + tp * BW_NVLINK / (2 * S_GRAD)))

print()
print("临界情形 t_p = 2S/BW（此时连第 2 张卡都不划算，N₅₀ = 1）：")
b_crit_ib = 2 * S_GRAD / BW_IB * P_PEAK / (6 * P_PARAMS)
b_crit_nvl = 2 * S_GRAD / BW_NVLINK * P_PEAK / (6 * P_PARAMS)
print("  IB     : B ≤ %.0f token 时，t_p ≤ %.2f s = 2S/BW，**加任何第二张卡都不划算**"
      % (b_crit_ib, t_ib))
print("  NVLink : B ≤ %.0f token 时才如此" % b_crit_nvl)
print()
print("这个阈值很有用：**小 batch 的训练任务根本不该做数据并行。**")
print("在 IB 上跑 70B，全局 batch 低于约 %.1f 万 token 时，" % (b_crit_ib / 1e4))
print("数据并行的通信时间已经超过全部计算时间——这时候该做的是")
print("梯度累积（把 batch 攒大）或者换并行策略（TP/PP），而不是加卡。")

print()
print("--- 4.5 用第 3 节的闭式解算 N*，看它是不是有效拐点 ---")
print()
print("现在代入 t_c。线性项的物理来源是 ring 的 2(N-1) 步延迟，")
print("每步 %.0f μs，所以 t_c = 2α = %.3e s/rank。" % (ALPHA * 1e6, 2 * ALPHA))
t_c = 2 * ALPHA
n_star_lin = math.sqrt(t_p / t_c)
n_star_log = t_p / t_c
print()
print("    线性模型 N* = sqrt(t_p/t_c) = sqrt(%.1f / %.1e) = %.0f"
      % (t_p, t_c, n_star_lin))
print("    log  模型 N* = t_p/t_c       = %.1f / %.1e = %.0f"
      % (t_p, t_c, n_star_log))
print()
print("**两个 N* 都远在一万卡以上（log 模型更是到了 1.7 亿）。但 4.3 节已经给出：")
print("效率在 N ≈ %.0f 就跌破 50%%了。**" % cross50(BW_IB))
print()
print("也就是说：**在本例的参数下，N* 不是有效的工程拐点。**")
print("原因很直接——线性延迟项每次同步 %.2f ms（N=1000 时累计 %.1f ms），"
      % (t_c * 1e3, t_c * 1e3 * 1000))
print("相对饱和的通信量 %.2f s 小了两三个数量级。" % t_ib)
print("真正卡住扩展的是那个**常数**通信量，不是步数延迟。")
print()
print("这里必须修正一个容易犯的错：把 N* = sqrt(t_p/t_c) 当成「最佳卡数」直接报出去，")
print("会严重高估可用规模（这里高估了 %.0f 倍）。" % (n_star_lin / cross50(BW_IB)))
print("**闭式解只对它所基于的那个模型成立**：它假设 t_c·N 是主要的通信开销，")
print("而本例里主要开销是一个与 N 无关的常数，所以这个 N* 约束不到任何东西。")

print()
print("把饱和的通信量也并进模型，看看修正量到底有多大。")
print("用三项 ring 模型（计算 + 步数延迟 + 饱和的带宽项）：")
print()
print("    T(N) = t_p/N + 2(N-1)α + 2(N-1)/N · S/BW")
print()
print("    dT/dN = -t_p/N² + 2α + 2S/(BW·N²)")
print("    令其为零：2α = (t_p - 2S/BW)/N²")
print()
print("    **N* = sqrt( (t_p - 2S/BW) / (2α) )**")
print()
print("分母里那个 2S/BW 就是饱和通信量。它从 t_p 里**减掉**——")
print("因为通信量本身不随 N 下降，它等效于把可摊薄的算力变小了。")
num = t_p - t_ib
n_star_r3 = math.sqrt(num / (2 * ALPHA))
print()
print("    IB : t_p - 2S/BW = %.1f - %.2f = %.1f ⇒ N* = sqrt(%.1f / %.1e) = %.0f"
      % (t_p, t_ib, num, num, 2 * ALPHA, n_star_r3))
print()
print("    **但这个修正只把 N* 从 %.0f 挪到 %.0f，改了 %.2f%%——微不足道。**"
      % (n_star_lin, n_star_r3, (n_star_lin - n_star_r3) / n_star_lin * 100))
print("    因为 2S/BW = %.2f s 相对 t_p = %.1f s 本来就很小。"
      % (t_ib, t_p))
print("    **修正量小这件事本身就是结论**：它再次说明饱和通信量在这里")
print("    远不足以改变延迟项给出的 N*，两者都没碰到真正的瓶颈。")
print()
print("    三项模型真正给出的新东西是**存在性条件**：")
print("    **当 t_p ≤ 2S/BW 时，(t_p - 2S/BW) ≤ 0，N* 不存在——**")
print("    此时 dT/dN > 0 对所有 N 成立，T(N) 单调上升，**加卡永远不划算**。")
print("    这正是 4.4 节算出的那个 batch 阈值：B ≤ %.0f token。" % b_crit_ib)

print()
print("--- 4.6 弱扩展：让 batch 跟着卡数涨 ---")
print()
print("上面固定 B。若改成每卡 batch 固定（B 随 N 线性涨），计算时间 t_p 与 N 无关，")
print("通信时间仍是常数 2S/BW，于是：")
print()
print("    T(N) = t_p(每卡算完自己那份) + 2S/BW + 2(N-1)α")
print("    η(N) = t_p / (t_p + 2S/BW + 2(N-1)α)")
print()
print("这是 Gustafson 的处境：效率**不随 N 崩塌**，只是被那个常数项压低。")
print()
print("  %-10s %14s %14s %14s" % ("每卡 batch", "每卡 t_p (s)", "η_IB 上界", "η_NVL 上界"))
print("  " + "-" * 58)
for b_local in (1024, 4096, 16384, 65536):
    tp_l = 6 * P_PARAMS * b_local / P_PEAK
    print("  %-10d %14.1f %13.1f%% %13.1f%%"
          % (b_local, tp_l, tp_l / (tp_l + t_ib) * 100,
             tp_l / (tp_l + t_nvl) * 100))
print()
print("这一列是**上界**：它只算了那个常数 2S/BW + %.1f·α 里 N→∞ 的带宽部分，"
      % 2)
print("没有算 2(N-1)α 的线性增长。N 大到 2(N-1)α 不可忽略时，η 仍会被拖下来。")
print()
print("关键读法：**弱扩展的效率不随 N 变化，只由「每卡算多少 token」决定。**")
print("这跟 4.4 节是同一件事的两面——效率取决于每卡的计算量，不取决于卡数。")
print("代价是总 batch 会随 N 一起涨，最终必须调大学习率，")
print("而学习率不能无限调大（收敛性约束），所以 Gustafson 视角也有尽头。")

rule("5. 断言自检")


def argmin_N(t_s_val):
    """数值法求 T(N) = t_s + t_p/N + t_c·N 的极小点，用来独立验证闭式解。"""
    return min(range(1, 400000), key=lambda n: t_s_val + t_p / n + t_c * n)


_nstar_num = argmin_N(0.0)

checks = [
    ("Amdahl 加速比恒不超过 1/s",
     all(1.0 / (s + (1 - s) / n) <= 1.0 / s + 1e-12 for n in NS[1:] for s in SS[1:])),
    ("Amdahl 的 S(N) 随 N 单调递增",
     all(1.0 / (0.05 + 0.95 / n) < 1.0 / (0.05 + 0.95 / (n * 2)) for n in (1, 8, 64, 1024))),
    # 注意容差必须是**相对**的：N=65536 时 S(N) ≈ 6.2e4，fp64 在该量级的
    # 绝对分辨率只有 ~1e-11，用固定的 1e-12 绝对容差会误报失败。
    ("Gustafson 的 S(N) 是线性的（相邻差分恒为 1-a）",
     all(abs(((0.05 + 0.95 * (n + 1)) - (0.05 + 0.95 * n)) - 0.95)
         <= 1e-12 * (0.05 + 0.95 * n) for n in (1, 8, 1024, 65536))),
    ("Gustafson 在 N=65536 时远超过 Amdahl 上限 1/s = 20",
     (0.05 + 0.95 * 65536) > 20 * 100),
    ("两个定律在 N=1 处给出相同答案（都是 1）",
     abs(1.0 / (0.05 + 0.95 / 1) - (0.05 + 0.95 * 1)) < 1e-12),
    ("N* 与 t_s 无关：t_s 从 0 变到 1e4，数值 argmin 不变",
     _nstar_num == argmin_N(10000.0)),
    ("数值 argmin 与闭式解 sqrt(t_p/t_c) 吻合（误差 <= 1）",
     abs(_nstar_num - round(math.sqrt(t_p / t_c))) <= 1),
    ("线性模型 N* 处的二阶导为正（是极小点）",
     2 * t_p / math.sqrt(t_p / t_c) ** 3 > 0),
    ("all-reduce 每卡数据量随 N 单调增，且上确界是 2S",
     all(2 * (n - 1) / n * S_GRAD < 2 * S_GRAD for n in (2, 8, 1024))
     and 2 * 65536 / 65537 * S_GRAD < 2 * S_GRAD),
    ("all-reduce 每卡数据量随 N 单调递增",
     all(2 * (n - 1) / n < 2 * (2 * n - 1) / (2 * n) for n in (2, 8, 64, 1024))),
    ("扩展效率随 N 单调下降（IB，强扩展）",
     all(eta(n, BW_IB)[0] > eta(n * 2, BW_IB)[0] for n in (1, 8, 64, 512, 2048))),
    ("NVLink 的扩展效率在任何 N 处都优于 IB",
     all(eta(n, BW_NVLINK)[0] >= eta(n, BW_IB)[0] for n in (2, 64, 1024, 8192))),
    ("重叠不降低效率",
     all(eta(n, BW_NVLINK, True)[0] >= eta(n, BW_NVLINK)[0] for n in (2, 64, 1024, 8192))),
    ("IB 的 N50 与闭式解 1 + t_p·BW/(2S) 吻合（误差 < 1）",
     abs(cross50(BW_IB) - (1 + t_p * BW_IB / (2 * S_GRAD))) < 1.0),
    ("饱和通信 2S/BW 大于 0 且与 N 无关",
     t_ib > 0 and abs(2 * (1024 - 1) / 1024 * S_GRAD / BW_IB - t_ib) < t_ib),
    ("t_p > 2S/BW（本例中 N* 存在）", t_p > t_ib),
    ("线性模型 N* 远大于 N50（闭式解不是有效拐点）", n_star_lin > 10 * cross50(BW_IB)),
    ("全局 batch 取临界值时 N50 = 1（第二张卡就不划算）",
     abs((1 + b_crit_ib * 6 * P_PARAMS / P_PEAK * BW_IB / (2 * S_GRAD)) - 2) < 0.5),
    ("三项模型的 N* 小于两项模型的 N*（减掉饱和通信量后最优规模变小）",
     math.sqrt((t_p - t_ib) / (2 * ALPHA)) < math.sqrt(t_p / (2 * ALPHA))),
    ("弱扩展效率有下界 t_p/(t_p + 2S/BW) 且随每卡 batch 增大而升高",
     (lambda f: f(1024) < f(65536) and f(65536) < 1.0)(
         lambda b: (6 * P_PARAMS * b / P_PEAK) / (6 * P_PARAMS * b / P_PEAK + t_ib))),
]
for desc, ok in checks:
    print("  [%s] %s" % ("通过" if ok else "**失败**", desc))
print()
print("全部 %d 项%s。" % (len(checks),
                        "通过" if all(o for _, o in checks) else "**未全部通过，见上**"))

rule("6. 本程序的边界")

print("  · 全部是解析计算。**没有在真实集群上跑过任何 all-reduce**，")
print("    η(N) 曲线是模型给的上界，不是实测吞吐。")
print("  · 只建模了**数据并行**的梯度 all-reduce。张量并行 / 流水线并行的通信")
print("    模式完全不同（TP 每层两次 all-reduce，PP 是点对点），")
print("    本程序的结论**不能**直接套到它们身上。")
print("  · 忽略了：优化器状态通信（ZeRO 会把通信量抬到 3 倍以上）、")
print("    通信与计算的实际重叠效率、网络拓扑的收敛比、straggler、")
print("    以及 ring 之外的其他算法（NCCL 会按消息大小自动切换）。")
print("  · α = %.0f μs 是一个**假设值**，不是实测。它只影响 N* 的量级，"
      % (ALPHA * 1e6))
print("    不影响第 4.3 节由带宽项决定的 N50——那个结论只依赖 S 和 BW。")
print("  · 峰值算力取 H100 SXM BF16 dense = 989.5 TFLOPS（不含 2:4 稀疏）。")
print("    用 sparse 峰值会让 t_p 减半、N50 减半。")
print("  · 第 2 节末尾提到「batch 放大会撞上收敛性上限」，这是训练动力学的")
print("    问题，本程序不涉及，只是指出 Gustafson 视角的前提并非无条件成立。")
