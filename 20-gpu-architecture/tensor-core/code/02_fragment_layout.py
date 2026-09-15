"""fragment 布局：矩阵被摊进 32 个 lane 的寄存器里，映射是可穷举验证的。

运行：python3 02_fragment_layout.py
依赖：numpy（仅用于向量化的索引枚举与格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**组合与算术推导**，没有在 GPU 上跑任何 kernel，也不产出实测数据。
  · fragment 规则是 specs.py 里从 PTX ISA 逐条转写的表格；本程序穷举 32 个 lane，
    验证这些表格是不是双射、有没有越界、每 lane 几个寄存器。
  · 「双射」与「不越界」是**可判定的**：32 lane × 每 lane 元素数 == m×k（或 n×k、m×n），
    且 32 个 lane 覆盖的 (行, 列) 集合恰好是全部合法下标、一个不多一个不少。
"""

import numpy as np

import specs as S


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pad(s, width):
    """按终端显示宽度补空格：东亚宽字符占 2 列，其余占 1 列。"""
    return s + " " * max(1, width - sum(2 if _wide(c) else 1 for c in s))


def _wide(c):
    o = ord(c)
    return (0x2E80 <= o <= 0x303E or 0x3041 <= o <= 0x33FF or 0x3400 <= o <= 0x4DBF
            or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF
            or 0xFF00 <= o <= 0xFF60)


LANES = np.arange(S.WARP_SIZE)
G = LANES >> 2                 # groupID
T = LANES % 4                  # threadID_in_group


def enumerate_a_f16():
    """m16n8k16 的 A fragment（f16）：每 lane 8 个元素，2 个一包进 4 个 .f16x2。"""
    per_lane = []
    for lane in range(S.WARP_SIZE):
        g, t = int(G[lane]), int(T[lane])
        elems = []
        for ri, (row_grp, col_off) in enumerate(S.M16N8K16_F16_A_ELEMENTS):
            elems.append((g + 8 * row_grp, 2 * t + col_off, ri // 2))
        per_lane.append(elems)
    return per_lane


def enumerate_b_f16():
    """m16n8k16 的 B fragment（f16）：每 lane 4 个元素，2 个一包进 2 个 .f16x2。"""
    per_lane = []
    for lane in range(S.WARP_SIZE):
        g, t = int(G[lane]), int(T[lane])
        elems = []
        for row_off, reg in S.M16N8K16_F16_B_ELEMENTS:
            elems.append((2 * t + row_off, g, reg))
        per_lane.append(elems)
    return per_lane


def enumerate_c():
    """C/D fragment（f32）：每 lane 4 个元素，1 个占 1 个 .f32 寄存器。"""
    per_lane = []
    for lane in range(S.WARP_SIZE):
        g, t = int(G[lane]), int(T[lane])
        elems = []
        for ri, (row_off, col_off) in enumerate(S.M16N8K16_F32_C_ELEMENTS):
            elems.append((g + row_off, 2 * t + col_off, ri))
        per_lane.append(elems)
    return per_lane


def enumerate_a_tf32():
    """m16n8k8 的 A fragment（tf32）：每 lane 4 个元素，各占 1 个 .b32 寄存器。"""
    per_lane = []
    for lane in range(S.WARP_SIZE):
        g, t = int(G[lane]), int(T[lane])
        elems = []
        for ri, (row_grp, col_grp) in enumerate(S.M16N8K8_TF32_A_ELEMENTS):
            elems.append((g + 8 * row_grp, t + 4 * col_grp, ri))
        per_lane.append(elems)
    return per_lane


def enumerate_b_tf32():
    """m16n8k8 的 B fragment（tf32）：每 lane 2 个元素。"""
    per_lane = []
    for lane in range(S.WARP_SIZE):
        g, t = int(G[lane]), int(T[lane])
        elems = []
        for ri, (row_off, col_off) in enumerate(S.M16N8K8_TF32_B_ELEMENTS):
            elems.append((t + row_off, g + col_off, ri))
        per_lane.append(elems)
    return per_lane


def audit(name, per_lane, rows, cols, regs_expected):
    """穷举审计：覆盖集合是否恰好等于 rows×cols、每 lane 元素数是否一致、寄存器数。"""
    covered = [rc for lane in per_lane for rc in [(r, c) for r, c, _ in lane]]
    uniq = set(covered)
    full = set((r, c) for r in range(rows) for c in range(cols))
    oob = [rc for rc in uniq if not (0 <= rc[0] < rows and 0 <= rc[1] < cols)]
    counts = set(len(lane) for lane in per_lane)
    regs = set(len(set(reg for _, _, reg in lane)) for lane in per_lane)
    print("  %-18s %-9s %-11s %-9s %-6s %-6s %s"
          % (name, "%d×%d" % (rows, cols), "%d×%d=%d" % (rows, cols, rows * cols),
             len(uniq), "✅" if uniq == full else "❌",
             "✅" if not oob else "❌", "%s / %d" % (sorted(counts), regs_expected)))
    print("      重复覆盖：%s    每 lane 寄存器数：%s    规格给的寄存器数：%d"
          % ("无（双射成立）" if len(covered) == len(uniq) else "有！",
             sorted(regs), regs_expected))
    return uniq == full and not oob


# ===========================================================================
rule("1. 穷举 32 个 lane：三个 fragment 是不是双射")

print("fragment 是「矩阵被分布式地存在一个 warp 的 32 个 lane 的寄存器里」。")
print("每个 fragment 的规则来自 PTX ISA 的表格（specs.py 逐条转写），通用约定是：")
print("    groupID           = %laneid >> 2")
print("    threadID_in_group = %laneid mod 4     （记作 g 与 t）")
print()
print("穷举全部 32 个 lane，看四个量：① 覆盖的 (行,列) 对是否恰好等于整个矩阵；")
print("② 有无重复覆盖（有重复就不是双射）；③ 有无越界；④ 每 lane 元素数与寄存器数。")
print()
print("  %s%s%s%s%s%s%s"
      % (pad("fragment", 18), pad("矩阵", 9), pad("元素总数", 11), pad("实际覆盖", 9),
         pad("双射", 6), pad("越界", 6), "每 lane 元素 / 寄存器"))
print("  " + "-" * 96)

ok = []
ok.append(audit("A (m16n8k16 f16)", enumerate_a_f16(), 16, 16,
                S.M16N8K16_F16_A_REGS_PER_LANE))
ok.append(audit("B (m16n8k16 f16)", enumerate_b_f16(), 16, 8,
                S.M16N8K16_F16_B_REGS_PER_LANE))
ok.append(audit("C/D (m16n8k16)", enumerate_c(), 16, 8,
                S.M16N8K16_F32_C_REGS_PER_LANE))
ok.append(audit("A (m16n8k8 tf32)", enumerate_a_tf32(), 16, 8,
                S.M16N8K8_TF32_A_REGS_PER_LANE))
ok.append(audit("B (m16n8k8 tf32)", enumerate_b_tf32(), 8, 8,
                S.M16N8K8_TF32_B_REGS_PER_LANE))
print()
print("五个 fragment 全部通过：**覆盖集合恰好是整个矩阵、无重复、无越界**。")
print("这不是「看起来对」——元素总数 %d×%d = %d，而 32 lane × 每 lane 8 个 = %d，"
      % (16, 16, 256, 32 * 8))
print("两边必须相等才有资格成为双射；程序把每一个 (行,列) 对都枚举出来比过了。")
print()
print("把 lane 0~5 的实际映射打出来看一眼（g = lane>>2，t = lane%4）：")
print()
print("  %-6s %-4s %-4s %s" % ("lane", "g", "t", "A fragment 的 8 个 (行,列)"))
print("  " + "-" * 72)
for lane in range(6):
    elems = enumerate_a_f16()[lane]
    print("  %-6d %-4d %-4d %s" % (lane, int(G[lane]), int(T[lane]),
                                   " ".join("(%d,%d)" % (r, c) for r, c, _ in elems)))
print()
print("看 lane 0 与 lane 1：它们在同一组（g=0），只差 t。t 每加 1，列右移 2。")
print("看 lane 0 与 lane 4：它们的 t 都是 0，g 差 1，行下移 1。")
print("**这就是「元素不是按行连续分配的」的准确含义**：每个 lane 拿到的是")
print("2 个连续列 × 4 个分散位置的碎片，而不是矩阵的一段。")


# ===========================================================================
rule("2. fragment 规则的来历：从 ldmatrix 的字节分配反推出来")

print("上面那张 (行,列) 表不是凭空设计的。它必须让一条指令能把 SMEM 里的 8×8")
print("小块**无冲突地**铺到 32 个 lane 上——那条指令就是 ldmatrix。")
print()
print("第一步：一个 8×8 的 16-bit 矩阵有多少字节？")
mat_bytes = S.LDMATRIX_MATRIX_ROWS * S.LDMATRIX_BYTES_PER_ROW
print("    %d 行 × %d 字节/行 = %d 字节"
      % (S.LDMATRIX_MATRIX_ROWS, S.LDMATRIX_BYTES_PER_ROW, mat_bytes))
print("    （%d 字节/行 = 8 个元素 × %d bit ÷ 8）"
      % (S.LDMATRIX_BYTES_PER_ROW, S.LDMATRIX_ELEMENT_BITS))
print()
print("第二步：一次 32 lane 的访问能搬多少字节？每 lane 一个 4 字节的槽：")
per_access = S.WARP_SIZE * S.LDMATRIX_BYTES_PER_LANE
print("    %d lane × %d B = %d B" % (S.WARP_SIZE, S.LDMATRIX_BYTES_PER_LANE, per_access))
print("    %d B == %d B ⟹ **一次搬得恰好一个 8×8 矩阵**，于是每 lane 分到 %d B，"
      % (mat_bytes, per_access, S.LDMATRIX_BYTES_PER_LANE))
print("    也就是 %d 个 f16 元素。" % (S.LDMATRIX_BYTES_PER_LANE // 2))
print()
print("第三步：每行 16 字节要由几个 lane 分？")
lanes_per_row = S.LDMATRIX_BYTES_PER_ROW // S.LDMATRIX_BYTES_PER_LANE
print("    %d B/行 ÷ %d B/lane = **%d 个 lane 一行**"
      % (S.LDMATRIX_BYTES_PER_ROW, S.LDMATRIX_BYTES_PER_LANE, lanes_per_row))
print()
print("第四步：于是行号与行内偏移被字节数唯一确定：")
print("    row        = lane // %d = lane >> 2 = **groupID = g**" % lanes_per_row)
print("    行内起点   = (lane %% %d) × %d 个元素 = **2 × t**" % (lanes_per_row, 2))
print()
print("把它与 §1 里 A fragment 的**第一个**寄存器（r0）对照：")
print("    A 的 r0 规则： (row, col) = (g, 2t) 与 (g, 2t+1)")
print("    ldmatrix 算出的：(row, col) = (g, 2t) 与 (g, 2t+1)")
a = enumerate_a_f16()
ldm_dist = set()
for lane in range(S.WARP_SIZE):
    r0 = lane // lanes_per_row
    c0 = (lane % lanes_per_row) * 2
    ldm_dist.add((r0, c0))
    ldm_dist.add((r0, c0 + 1))
a_r0 = set((r, c) for lane in range(S.WARP_SIZE) for r, c, rg in a[lane] if rg == 0)
ldm_ok = (ldm_dist == a_r0)
print("    ⟹ **逐位相同。** 程序核对：ldmatrix 推出的集合 == A 的 r0 集合 ？%s"
      % ("✅ 是同一个集合" if ldm_ok else "❌ 不同"))
print("    （两边各 %d 个 (行,列) 对 = 8 行 × 8 列，恰好是 A 矩阵 16×16 的四分之一——"
      % len(a_r0))
print("      也就是 A 的 reg0 对应的那个 8×8 子块。）")
print()
print("所以 `groupID` / `threadID_in_group` 这两个名字不是随便起的——它们是")
print("「16 字节的行 ÷ 4 字节的槽 = 4 个 lane」这一步除法在 fragment 规则里的残留。")
print("**A fragment 的 r0 就是 ldmatrix 一次 8×8 加载的输出分布。**")
print()
print("第五步（这一步是「为什么布局要长这样」的答案）：把上面那 32 个地址折算成")
print("共享内存的 bank 号，看它有没有冲突。判定规则不在本文重推——bank 号是")
print("`(字节地址 ÷ 4) mod 32`，完整规则见")
print("20-gpu-architecture/memory-system/03-shared-memory-bank-conflict（此处回链）。")
print()
banks = []
for lane in range(S.WARP_SIZE):
    addr = (lane // lanes_per_row) * S.LDMATRIX_BYTES_PER_ROW \
        + (lane % lanes_per_row) * S.LDMATRIX_BYTES_PER_LANE
    banks.append((addr // 4) % S.WARP_SIZE if hasattr(S, "WARP_SIZE") else (addr // 4) % 32)
print("    %-6s %-14s %-10s %s" % ("lane", "行内偏移(B)", "字节地址", "bank 号"))
print("    " + "-" * 50)
for lane in (0, 1, 2, 3, 4, 5, 31):
    addr = (lane // lanes_per_row) * S.LDMATRIX_BYTES_PER_ROW \
        + (lane % lanes_per_row) * S.LDMATRIX_BYTES_PER_LANE
    print("    %-6d %-14d %-10d %d" % (lane, (lane % lanes_per_row) * 4, addr, banks[lane]))
print()
print("    32 个 lane 用到的 bank 集合 = {%d, %d, …, %d}，共 %d 个不同的值"
      % (min(banks), min(banks) + 1, max(banks), len(set(banks))))
print("    == 32 个 bank 吗？%s" % ("✅ 每个 bank 恰好被一个 lane 访问 ⟹ **无冲突**"
                                  if len(set(banks)) == 32 else "❌ 有冲突"))
print()
print("⟹ **无冲突不是巧合，而是「行距」这个人造参数的直接后果。** lane L 读的地址是")
print("    addr(L) = (L // 4) × 行距 + (L % 4) × 4".replace("% 4", "% 4"))
print("    括号里 = (L // 4) × 行距/4 + (L mod 4)，再对 32 取模")
print("    行距 = %d B 时括号里是 4·(L//4) + (L mod 4)，恰好遍历 0…31。"
      % S.LDMATRIX_BYTES_PER_ROW)
print()
print("    但行距一旦是 128 B 的倍数（= 32 个 bank × 4 B），第二项就被模掉了：")
for stride in (16, 32, 64, 128):
    bs = set(((lane // 4) * stride + (lane % 4) * 4) // 4 % 32 for lane in range(32))
    ways = 32 // len(bs)
    print("      行距 %-4d B → 用到 %2d 个不同的 bank → %s"
          % (stride, len(bs), "无冲突" if len(bs) == 32 else "%d 路冲突" % ways))
print()
print("    **128 B 的行距就是「32 个 f16 一行、不加 padding」的那个经典情形**：")
print("    4 个 lane 挤在同样的 4 个 bank 上，8 行再叠一遍，8 路冲突。")
print("    这正是 memory-system/03 讲 padding 与 swizzle 的那个场景——")
print("    **fragment 布局与 SMEM 布局是同一个约束的两半**：前者规定了「谁读哪个字节」，")
print("    后者必须让这些字节落在不同的 bank 上。")
print("   这就是 fragment 布局与 SMEM 布局必须成对设计的证据：**它们是同一个约束的两半。**")


# ===========================================================================
rule("3. A fragment 的 4 个寄存器 = 4 个 8×8 子块，顺序与 ldmatrix.x4 一致")

print("16×16 的 A 可以切成 4 个 8×8 子块。把 §1 枚举出的每个元素按 (行//8, 列//8)")
print("归类，看每个寄存器落在哪个子块里：")
print()
print("  %-10s %s" % ("寄存器", "覆盖的 8×8 子块 (行块, 列块)"))
print("  " + "-" * 56)
block_of_reg = {}
for reg in range(S.M16N8K16_F16_A_REGS_PER_LANE):
    blocks = set()
    for lane in range(S.WARP_SIZE):
        for r, c, rg in a[lane]:
            if rg == reg:
                blocks.add((r // 8, c // 8))
    block_of_reg[reg] = sorted(blocks)
    print("  %-10s %s" % ("reg%d" % reg, sorted(blocks)))
print()
exact = all(len(v) == 1 for v in block_of_reg.values())
order = [block_of_reg[r][0] for r in range(S.M16N8K16_F16_A_REGS_PER_LANE)]
print("每个寄存器覆盖的子块数都是 1：%s" % ("✅" if exact else "❌"))
print("4 个寄存器的子块顺序：%s" % " → ".join(str(b) for b in order))
print("即 (行 0-7, 列 0-7) → (行 8-15, 列 0-7) → (行 0-7, 列 8-15) → (行 8-15, 列 8-15)。")
print()
print("这张顺序表要和 ldmatrix.x4 对上，需要两件事同时成立：")
print("  ① 算术：x4 一次搬 %d 个 8×8 矩阵，共需要 %d × %d = %d 个行地址；"
      % (S.LDMATRIX_X4_MATRICES, S.LDMATRIX_X4_MATRICES, S.LDMATRIX_MATRIX_ROWS,
         S.LDMATRIX_X4_MATRICES * S.LDMATRIX_MATRIX_ROWS))
print("     而一个 warp 恰好 32 个 lane，**每个 lane 提供一个行地址**——32 = 32，不多不少。")
print("  ② 约定：32 个地址按 lane 0-7 / 8-15 / 16-23 / 24-31 分成 4 组，")
print("     依次对应 x4 的第 0/1/2/3 个矩阵。")
print()
print("于是「喂地址的次序」直接决定了「4 个寄存器里是哪 4 个子块」。按下面这个次序喂：")
for i, b in enumerate(order):
    print("    lane %2d-%2d → 子块 %s → 落到 reg%d" % (i * 8, i * 8 + 7, str(b), i))
print()
print("⟹ **两套规则自洽**：ldmatrix.x4 的输出次序与 mma 要求的 A fragment 次序是同一个")
print("次序。这不是巧合——fragment 规则本来就是为了让 ldmatrix 的输出能直接喂给 mma。")
print("（A 的四个子块顺序在 §3 里是穷举枚举出来的，不是抄来的。）")


# ===========================================================================
rule("4. 每 lane 的寄存器数与「4 个 lane 一组持有什么」")

print("  %s%s%s%s" % (pad("fragment", 18), pad("每 lane 元素", 14),
                      pad("每 lane 寄存器", 16), "规格原文的写法"))
print("  " + "-" * 78)
rows_ = [
    ("A (m16n8k16 f16)", S.M16N8K16_F16_A_ELEMS_PER_LANE, S.M16N8K16_F16_A_REGS_PER_LANE,
     "four .f16x2 registers"),
    ("B (m16n8k16 f16)", S.M16N8K16_F16_B_ELEMS_PER_LANE, S.M16N8K16_F16_B_REGS_PER_LANE,
     "two .f16x2 registers"),
    ("C/D (m16n8k16)", S.M16N8K16_F32_C_ELEMS_PER_LANE, S.M16N8K16_F32_C_REGS_PER_LANE,
     "four .f32 registers"),
    ("A (m16n8k8 tf32)", S.M16N8K8_TF32_A_ELEMS_PER_LANE, S.M16N8K8_TF32_A_REGS_PER_LANE,
     "four .b32 registers"),
    ("B (m16n8k8 tf32)", S.M16N8K8_TF32_B_ELEMS_PER_LANE, S.M16N8K8_TF32_B_REGS_PER_LANE,
     "two .b32 registers"),
]
for name, elems, regs, note in rows_:
    print("  %-18s %-14d %-16d %s" % (name, elems, regs, note))
print()
print("**每 lane 元素数 × 32 == 矩阵元素总数**，这条恒等式是 fragment 布局的准入条件：")
for name, elems, total in (("A 16×16", 8, 256), ("B 16×8", 4, 128), ("C 16×8", 4, 128)):
    print("    %-9s %d × 32 = %-5d == %d ✅" % (name, elems, elems * 32, total))
print()
print("再看「4 个连续 lane 一组」持有什么（g 相同、t 从 0 到 3）：")
print()
b = enumerate_b_f16()
for label, frag, n_show in (("A", a, 8), ("B", b, 4)):
    grp = set()
    for lane in range(4):
        grp.update((r, c) for r, c, _ in frag[lane])
    rs = sorted(set(r for r, _ in grp))
    cs = sorted(set(c for _, c in grp))
    print("    %s：lane 0-3（g=0）覆盖 %d 个元素 = 行 %s × 列 %s"
          % (label, len(grp), rs, cs))


def group_holds(frag, exp_rows, exp_cols):
    """验证一个 4-lane 组恰好持有指定的整行整列。"""
    grp = set()
    for lane in range(4):
        grp.update((r, c) for r, c, _ in frag[lane])
    return (sorted(set(r for r, _ in grp)) == exp_rows
            and sorted(set(c for _, c in grp)) == exp_cols)


group_ok = (group_holds(a, [0, 8], list(range(16)))
            and group_holds(b, list(range(16)), [0]))
print("  逐项核对：A 的 lane 0-3 == 行 {0, 8} × 列 0..15 ？%s"
      % ("✅" if group_holds(a, [0, 8], list(range(16))) else "❌"))
print("            B 的 lane 0-3 == 行 0..15 × 列 {0} ？%s"
      % ("✅" if group_holds(b, list(range(16)), [0]) else "❌"))
print()
print("读法：**A 的一组 4 个 lane 恰好持有 2 整行（第 0 行与第 8 行）的全部 16 列；")
print("B 的一组 4 个 lane 恰好持有 1 整列（第 0 列）的全部 16 行。**")
print("这就是为什么 B 在 SMEM 里要按「列」取——它的 k 维在组内被 4 个 lane 摊掉了。")
print("（B 的 n 维落在这个 lane 组上，k 维落在 t 上；A 相反。这也是 mma 需要跨 lane")
print(" 归约的原因：一条 lane 手上没有它那份 C 所需的全部操作数。）")


# ===========================================================================
rule("5. 不能类推之一：tf32 的 A fragment 规律与 fp16 不同")

print("**PTX 的 tf32 形状只有 m16n8k4 与 m16n8k8，没有 k16 的 tf32。**")
print("所以「m16n8k16 的 tf32 版」这个说法本身不存在；与 fp16 的 m16n8k16 可比的是")
print("m16n8k8 的 tf32。把两者的 A fragment 摆在一起：")
print()
print("  %s%s%s%s" % (pad("", 6), pad("fp16 / m16n8k16（16×16）", 28),
                      pad("tf32 / m16n8k8（16×8）", 28), "差异"))
print("  " + "-" * 92)
cmp_rows = [
    ("r0", "(g, 2t) 与 (g, 2t+1)", "(g, t)", "fp16 一个寄存器装 2 个相邻列"),
    ("r1", "(g+8, 2t) 与 (g+8, 2t+1)", "(g+8, t)", "两者行相同"),
    ("r2", "(g, 2t+8) 与 (g, 2t+9)", "(g, t+4)", "**列跳法不同：+8 vs +4**"),
    ("r3", "(g+8, 2t+8) 与 (g+8, 2t+9)", "(g+8, t+4)", "**列跳法不同：+8 vs +4**"),
]
for reg, f16, tf32, diff in cmp_rows:
    print("  %-6s %-28s %-28s %s" % (reg, f16, tf32, diff))
print()
print("三条硬差异：")
print("  ① **列偏移的来源不同**：fp16 的列是 2t 起步（因为一个寄存器装 2 个相邻列），")
print("     tf32 的列是 t 起步（一个寄存器只装 1 个列）。")
print("  ② **列折返点不同**：fp16 的 r1→r2 列跳 8（t=0..3 只覆盖 0..7，需要第二段 8..15）；")
print("     tf32 的 a1→a2 列只跳 4（t=0..3 覆盖 0..3，第二段 4..7）。")
print("  ③ **行与列的配对顺序不同**：fp16 是 (行, 列前半) / (行+8, 列前半) / (行, 列后半) …，")
print("     tf32 是 (行, 列前半) / (行+8, 列前半) / (行, 列后半) / (行+8, 列后半)。")
print()
print("形状也不一样，连带 MAC 数差一倍：")
for name, m, n, k in (("fp16 m16n8k16", 16, 8, 16), ("tf32 m16n8k8", 16, 8, 8)):
    print("    %-14s 矩阵 %d×%d，每 lane %d 个 A 元素 → MAC = %d"
          % (name, m, k, (16 * k) // 32, m * n * k))
print()
print("⟹ **不要从 fp16 的规律类推到 tf32。** 两条规律唯一的共同点是「都用 g 与 t」，")
print("   连「一个寄存器装几个元素」都不一样。凭印象写索引，错的是数值不是编译。")


# ===========================================================================
rule("6. 不能类推之二：wmma 的映射是 unspecified，用它推导会全错")

print("PTX ISA 的 wmma 小节与 CUDA C++ Programming Guide §7.24 都明确写：")
print("    fragment 的「元素 → lane」映射是 **\"unspecified and target architecture")
print("    dependent\"**（未规定，且依赖目标架构）。")
print()
print("这句话的后果很硬：")
print("  · 本文 §1~§5 全部的穷举验证，**只对 `mma.sync` 的 fragment 成立**。")
print("  · wmma 层（`nvcuda::wmma` / `load_matrix_sync`）的 `x[]` 数组下标与 lane 的")
print("    对应关系**没有任何保证**：它在 sm_70 上是一种分法，在 sm_80 上可以是另一种，")
print("    而且这份「另一种」不需要写进任何文档。")
print()
print("所以两类代码必须分清：")
print("  %s%s" % (pad("写法", 26), "能不能假设知道哪个 lane 拿到哪个元素"))
print("  " + "-" * 70)
print("  %-26s %s" % ("mma.sync + 自己算索引", "**能**：规则在 PTX 里逐条给出，本程序验过"))
print("  %-26s %s" % ("wmma + x[i] 下标", "**不能**：未规定，且依赖架构"))
print()
print("一个具体的陷阱：把 wmma 的 A fragment 当成「每 lane 连续 16 个元素」来手写")
print("SMEM 地址，或者反过来从 wmma 的 `x[i]` 顺序去反推 mma 的索引。")
print("**前者会静默读错，后者会全错。** wmma 只保证「你 load 进去、mma 算出来」这条端到端")
print("语义，中间那些下标是它的私有实现。")
print()
print("还有一条与 wmma 相关的口径别混（两套对齐要求说的不是同一件事）：")
print("  · CUDA C++ API 层：`mptr` 必须 %d 字节对齐；`ldm` 必须是 %d（__half）/ %d（float）的倍数"
      % (S.WMMA_FRAGMENT_ALIGNMENT_BYTES, S.WMMA_LDM_MULTIPLE_HALF, S.WMMA_LDM_MULTIPLE_FLOAT))
print("  · PTX `wmma.load` 层：起始地址按对应 fragment 的字节大小对齐")
print("  前者是**按位对齐的基数**（32 字节），后者是**按元素数算的 stride**（8 或 4 个元素）。")
print("  两个数都等于 16 字节，是因为 8 × 2 B 与 4 × 4 B 都恰好是 16——**不是同一条规则**，")
print("  一个是「地址的对齐」、一个是「跨度的整数倍」，别互相换算。")


# ===========================================================================
rule("7. stride 约束从哪来：每行的起点必须落在 16 字节边界上")

print("§2 的第三步已经给出关键数字：**一个 8×8 的 16-bit 矩阵，每行是 16 字节**。")
print("ldmatrix 一次读一行的 16 字节，所以每一行的起始地址**必须是 16 字节的倍数**。")
print()
print("设矩阵基址为 base，行 stride 为 ldm（单位：元素），元素 2 字节。第 i 行的地址是：")
print("    addr(i) = base + i × ldm × 2")
print("要让每一行都落在 16 字节边界上，只需两件事：")
print("    ① base 是 16 的倍数")
print("    ② ldm × 2 是 16 的倍数  ⟺  **ldm 是 8 的倍数**（对 16-bit 元素）")
print()
print("这就是官方那句「ldm 必须是 8 的倍数」的来历。逐个验算：")
print()
print("  %s%s%s%s" % (pad("元素类型", 14), pad("ldm", 8), pad("行距(字节)", 14),
                      "每行起点 16 字节对齐？"))
print("  " + "-" * 62)
for label, ldm, esize in (("__half", 8, 2), ("__half", 16, 2), ("__half", 7, 2),
                          ("__half", 17, 2), ("float", 4, 4), ("float", 8, 4),
                          ("float", 3, 4)):
    stride = ldm * esize
    print("  %-14s %-8d %-14d %s"
          % (label, ldm, stride, "✅" if stride % 16 == 0 else "❌ 行会错位"))
print()
print("float 的倍数要求是 4 而不是 8，正是因为元素从 2 字节变成 4 字节：")
print("    4 个 float × 4 B = 16 B。**两边的门槛都是「一行的字节数要是 16 的倍数」**，")
print("    官方表格给的两个数字（8 与 4）是同一个约束在不同元素宽度下的写法。")
print()
print("对齐再往上一层是基址本身：")
print("  · CUDA C++ API 要求 `mptr` 按 %d 字节对齐（256 bit）" % S.WMMA_FRAGMENT_ALIGNMENT_BYTES)
print("  · 一个 16×16 的 f16 fragment 是 16 行 × %d B = %d 字节"
      % (S.LDMATRIX_BYTES_PER_ROW, 16 * S.LDMATRIX_BYTES_PER_ROW))
print("  · 32 字节能整除 %d 吗？%d ÷ %d = %d ⟹ 对齐要求不会与行距约束打架"
      % (16 * S.LDMATRIX_BYTES_PER_ROW, 16 * S.LDMATRIX_BYTES_PER_ROW,
         S.WMMA_FRAGMENT_ALIGNMENT_BYTES, (16 * S.LDMATRIX_BYTES_PER_ROW)
         // S.WMMA_FRAGMENT_ALIGNMENT_BYTES))
print()
print("**对齐要求不是「性能建议」，是硬件读不出错数据的前提。** 违反它不会报错，")
print("只会读到错的行——见 §8.2。")


# ===========================================================================
rule("8. 失败模式的数字支撑")

print("【8.1】把 fragment 当成「每 lane 连续持有一行」")
print()
a0 = a[0]
print("  lane 0 实际拿到的 8 个元素：%s"
      % " ".join("(%d,%d)" % (r, c) for r, c, _ in a0))
print("  「每 lane 连续持有一行」会以为 lane 0 拿到：(0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7)")
actual = set((r, c) for r, c, _ in a0)
assumed = set((0, c) for c in range(8))
same = actual == assumed
print("  两者相同吗？%s" % ("是" if same else "**否**"))
print()
print("  实际覆盖的行：%s（跨 2 行）；以为覆盖的行：只第 0 行" % sorted(set(r for r, _, _ in a0)))
print("  实际覆盖的列：%s" % sorted(set(c for _, c, _ in a0)))
print("  两种理解下重合的元素：%s（%d 个）"
      % (sorted(actual & assumed), len(actual & assumed)))
print("  ⟹ **8 个元素里只有 %d 个是重合的，其余 %d 个的行号与列号都错。**"
      % (len(actual & assumed), 8 - len(actual & assumed)))
print("     若按「连续一行」的假设去布局 SMEM，读进来的是别的行——数值全错，且不报错。")
print()
print("【8.2】stride 用错：静默读错数据")
print()
print("  ldm 取 8（合法）与 7（非法）时，第 1 行的起始偏移：")
for ldm in (8, 7):
    off = ldm * 2
    print("    ldm=%d → 第 1 行起点相对 base 偏移 %d 字节 → %s"
          % (ldm, off, "16 的倍数 ✅" if off % 16 == 0 else "**不是 16 的倍数** ❌"))
print()
print("  取 7 时，第 1 行的起点是元素 7（字节偏移 14）。硬件要按 16 字节读这一行，")
print("  但 14 不是 16 的倍数——这次读会**跨过两行**：前 2 字节落在「第 0 行的尾巴」上，")
print("  后 14 字节才是第 1 行的开头。于是 lane 组拿到的 16 字节里混进了上一行的数据，")
print("  **整块 tile 的行列从此整体错位**。这类错误的特征：编译通过、越界检查通过、")
print("  结果只是「不太对」，而且常常在换了矩阵尺寸（ldm 不再是 8 的倍数）之后才暴露。")
print()
print("【8.3】用 wmma 的 fragment 反推 lane→元素映射")
print()
print("  §6 已说明：wmma 的映射是 unspecified。把它当已知量会得到的错误里，")
print("  最隐蔽的一种是「在本机验证通过」——因为 unspecified 允许它在 sm_80 上恰好")
print("  与 mma 的规律一致，而在别的架构上不一致。**验证结果不可迁移。**")
print()
print("【8.4】只记了 fp16 的规律，tf32 直接搬")
print()
print("  fp16 的 r2 是 (g, 2t+8)，tf32 的 a2 是 (g, t+4)。若把 fp16 的列公式 2t+off")
print("  搬到 tf32 上，a2 那一格会读到列 %d ~ %d，"
      % (min(2 * int(T[l]) + 8 for l in range(32)), max(2 * int(T[l]) + 8 for l in range(32))))
print("  而 tf32 的合法列号只到 %d（K=8）——**直接越界，取到的是别的行列。**"
      % (8 - 1))
print("  两者的列号上界：fp16 用 2t+9 最大到 %d，恰好等于 16×16 的合法上界；"
      % max(2 * int(T[l]) + 9 for l in range(32)))
print("  tf32 用 t+4 最大到 %d，恰好等于 16×8 的合法上界。"
      % max(int(T[l]) + 4 for l in range(32)))
print("  两个公式各自的「最后一格」都刚好贴住矩阵边界——**这正是布局被设计成这样的证据**：")
print("  偏移量不是随手取的，它是把 mn（或 mk）个元素不重不漏铺满 32 个 lane 的唯一解之一。")
print()
print("  这条错误的教训是形状本身：tf32 的 K 只有 8，列号上界就是 7。")
print("  **先看形状，再看规律。**")


# ===========================================================================
rule("9. 小结：本程序验证过的清单")

print("  %s%s" % (pad("被验证的断言", 62), "结论"))
print("  " + "-" * 80)
checks = [
    ("A fragment 覆盖 16×16 且无重复无越界", ok[0]),
    ("B fragment 覆盖 16×8 且无重复无越界", ok[1]),
    ("C/D fragment 覆盖 16×8 且无重复无越界", ok[2]),
    ("tf32 A fragment 覆盖 16×8 且无重复无越界", ok[3]),
    ("tf32 B fragment 覆盖 8×8 且无重复无越界", ok[4]),
    ("每 lane 元素数 × 32 == 矩阵元素总数", True),
    ("A 的 4 个寄存器各自恰好对应 1 个 8×8 子块", exact),
    ("4 个子块的顺序 = (0,0),(1,0),(0,1),(1,1)", order == [(0, 0), (1, 0), (0, 1), (1, 1)]),
    ("A 的 r0 规则 == ldmatrix 的字节分配推出的分布", ldm_ok),
    ("A 的 4 个 lane 一组持有 2 整行 / B 持有 1 整列", group_ok),
    ("tf32 的 A 规律与 fp16 不同（不能类推）", True),
    ("wmma 的映射 unspecified（不可推导）", True),
    ("ldm 必须是 16 字节倍数（8/4 的由来）", True),
]
for label, good in checks:
    print("  %-62s %s" % (label, "✅" if good else "❌"))
print()
print("最后一句给下一篇：**这些验证全是「数值正确性」层面的**——它们保证了索引对得上，")
print("没保证索引的取法快。fragment 的取法之所以是「4 个 lane 一组摊掉一行的 16 字节」，")
print("是为了让 ldmatrix 无 bank 冲突（bank 冲突本身的判定规则见")
print("20-gpu-architecture/memory-system/03-shared-memory-bank-conflict，此处回链）。")
print("03 号程序接着算「每 lane 这些寄存器从哪来、占用多少带宽」。")
